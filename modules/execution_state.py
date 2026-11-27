# -*- coding: utf-8 -*-
"""
modules/execution_state.py
Decision State Machine, Reason Codes, Persistent Execution State, Daily Risk/Trade Counters,
Thread-Safe Execution Locks, dan Startup/Reconnect Reconciliation.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from modules.config import (
    AUTO_RESUME_REAL_FULL_AUTO,
    AUTO_TRADING,
    DAILY_RESET_TIMEZONE,
    ENTRY_MODE,
    FILE_STATE_AUTO_EXECUTION,
    SCANNER_MAGIC_NUMBER,
    SYMBOL,
    TRADING_MODE,
)
from modules.utils import WIB_TZ, _catat_error, format_wib_timestamp


# =============================================================================
# CANONICAL EXECUTION STATES (SECTION 12 & 16)
# =============================================================================
STATE_RADAR = "RADAR"
STATE_APPROACHING = "APPROACHING"
STATE_ZONE_TOUCHED = "ZONE_TOUCHED"
STATE_WAIT_TRIGGER = "WAIT_TRIGGER"
STATE_ENTRY_READY = "ENTRY_READY"
STATE_EXECUTING = "EXECUTING"
STATE_ORDER_PLACED = "ORDER_PLACED"
STATE_POSITION_ACTIVE = "POSITION_ACTIVE"
STATE_COMPLETED = "COMPLETED"
STATE_INVALIDATED = "INVALIDATED"
STATE_EXPIRED = "EXPIRED"
STATE_BLOCKED = "BLOCKED"

EXECUTION_STATES = frozenset({
    STATE_RADAR,
    STATE_APPROACHING,
    STATE_ZONE_TOUCHED,
    STATE_WAIT_TRIGGER,
    STATE_ENTRY_READY,
    STATE_EXECUTING,
    STATE_ORDER_PLACED,
    STATE_POSITION_ACTIVE,
    STATE_COMPLETED,
    STATE_INVALIDATED,
    STATE_EXPIRED,
    STATE_BLOCKED,
})

TERMINAL_EXECUTION_STATES = frozenset({
    STATE_COMPLETED,
    STATE_INVALIDATED,
    STATE_EXPIRED,
})

VALID_TRADING_MODES = frozenset({
    "SCANNER_ONLY",
    "DRY_RUN",
    "SEMI_AUTO",
    "FULL_AUTO",
})

VALID_ENTRY_MODES = frozenset({
    "EARLY_PENDING",
    "ADAPTIVE",
    "ADAPTIVE_ENTRY",
    "CONFIRMED",
    "CONFIRMED_ENTRY",
})


# =============================================================================
# CANONICAL WAIT / BLOCKED REASON CODES (SECTION 16)
# =============================================================================
REASON_OK = "OK"
REASON_WAIT_TRIGGER = "WAIT_TRIGGER"
REASON_WAIT_ZONE_TOUCH = "WAIT_ZONE_TOUCH"
REASON_GRADE_A_WAIT_CONFIRM = "GRADE_A_WAIT_CONFIRM"
REASON_WAIT_USER_APPROVAL = "WAIT_USER_APPROVAL"
REASON_SCANNER_ONLY_MODE = "SCANNER_ONLY_MODE"
REASON_AUTO_TRADING_OFF = "AUTO_TRADING_OFF"
REASON_HTF_CONFLICT = "HTF_CONFLICT"
REASON_DIRECTION_NOT_ALIGNED = "DIRECTION_NOT_ALIGNED"
REASON_ZONE_INVALIDATED = "ZONE_INVALIDATED"
REASON_ZONE_EXPIRED = "ZONE_EXPIRED"
REASON_ZONE_LOW_SCORE = "ZONE_LOW_SCORE"
REASON_ZONE_MITIGATED = "ZONE_MITIGATED"
REASON_SPREAD_TOO_HIGH = "SPREAD_TOO_HIGH"
REASON_RR_BELOW_MINIMUM = "RR_BELOW_MINIMUM"
REASON_MIN_LOT_EXCEEDS_RISK = "MIN_LOT_EXCEEDS_RISK"
REASON_DAILY_LOSS_LIMIT = "DAILY_LOSS_LIMIT"
REASON_MAX_EXPOSURE_EXCEEDED = "MAX_EXPOSURE_EXCEEDED"
REASON_MAX_ACTIVE_POSITIONS = "MAX_ACTIVE_POSITIONS"
REASON_MAX_ACTIVE_PENDING_ORDERS = "MAX_ACTIVE_PENDING_ORDERS"
REASON_MAX_TRADES_PER_DAY = "MAX_TRADES_PER_DAY"
REASON_DUPLICATE_SETUP = "DUPLICATE_SETUP"
REASON_STOPPED_OUT_SETUP = "STOPPED_OUT_SETUP"
REASON_PRICE_LEFT_ENTRY_RANGE = "PRICE_LEFT_ENTRY_RANGE"
REASON_ZONE_TOO_FAR = "ZONE_TOO_FAR"
REASON_NEWS_BLACKOUT = "NEWS_BLACKOUT"
REASON_STALE_MARKET_DATA = "STALE_MARKET_DATA"
REASON_STALE_NEWS_DATA = "STALE_NEWS_DATA"
REASON_MT5_DISCONNECTED = "MT5_DISCONNECTED"
REASON_ALGO_TRADING_DISABLED = "ALGO_TRADING_DISABLED"
REASON_ACCOUNT_CHANGED = "ACCOUNT_CHANGED"
REASON_REAL_ACCOUNT_UNCONFIRMED = "REAL_ACCOUNT_UNCONFIRMED"
REASON_DEMO_ONLY_RESTRICTION = "DEMO_ONLY_RESTRICTION"
REASON_MARKET_CLOSED = "MARKET_CLOSED"
REASON_INVALID_STOP_LOSS = "INVALID_STOP_LOSS"
REASON_INVALID_TAKE_PROFIT = "INVALID_TAKE_PROFIT"
REASON_INVALID_VOLUME = "INVALID_VOLUME"
REASON_INSUFFICIENT_MARGIN = "INSUFFICIENT_MARGIN"
REASON_BROKER_CONSTRAINTS = "BROKER_CONSTRAINTS"
REASON_UNPROTECTED_POSITION_NO_SL = "UNPROTECTED_POSITION_NO_SL"
REASON_EXECUTION_REJECTED = "EXECUTION_REJECTED"
REASON_EXECUTION_AMBIGUOUS = "EXECUTION_AMBIGUOUS"
# Session + PO3/AMD guard (hanya menahan EARLY_PENDING; ADAPTIVE_ENTRY saat sentuh zona tidak terpengaruh)
REASON_AMD_COUNTER_PHASE = "AMD_COUNTER_PHASE"
REASON_AMD_LIQUIDITY_EXPOSED = "AMD_LIQUIDITY_EXPOSED"


def canonical_zone_key(zone_id: str) -> str:
    """
    Normalisasi wrapper zone_id (mis. TF_PB_SNR_... atau TF_BR_SNR_...) ke zone_id sumber aslinya
    agar satu zona fisik tidak pernah dieksekusi ganda lewat jalur SMC/TF dan SNR sekaligus.
    """
    zid = str(zone_id or "").strip()
    for prefix in ("TF_PB_", "TF_BR_"):
        if zid.startswith(prefix):
            inner = zid[len(prefix):]
            if inner.startswith(("SNR_", "ZONE_", "SMC_")):
                return inner
    return zid


def make_setup_id(zone_id: str, direction: str, symbol: str = SYMBOL) -> str:
    """
    Buat setup_id deterministik dan persisten untuk satu zona + arah transaksi.
    Ketiga jalur entry (EARLY_PENDING, ADAPTIVE_ENTRY, CONFIRMED_ENTRY) berbagi setup_id
    yang sama pada zona yang sama agar saling eksklusif dan anti-duplicate.
    """
    raw_zid = str(zone_id or "").strip()
    if "::SCALE_IN" in raw_zid:
        base_part, suffix = raw_zid.split("::SCALE_IN", 1)
        zid = f"{canonical_zone_key(base_part)}::SCALE_IN{suffix}"
    else:
        zid = canonical_zone_key(raw_zid)
    dir_u = str(direction or "").strip().upper()
    sym_u = str(symbol or SYMBOL).strip().upper()
    raw = f"{sym_u}|{zid}|{dir_u}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"SETUP_{sym_u}_{dir_u}_{digest}"


def encode_order_comment(setup_id: str, entry_mode: str) -> str:
    """Format comment MT5 <= 31 karakter dengan identitas setup & mode."""
    mode_tag = {
        "EARLY_PENDING": "EP",
        "ADAPTIVE_ENTRY": "AD",
        "ADAPTIVE": "AD",
        "CONFIRMED_ENTRY": "CF",
        "CONFIRMED": "CF",
    }.get(str(entry_mode or "").upper(), "AU")
    sid_tail = str(setup_id or "")[-16:]
    comment = f"SMC_{mode_tag}_{sid_tail}"
    return comment[:31]


def current_trading_day_key(now_utc: Optional[datetime] = None, reset_tz: str = DAILY_RESET_TIMEZONE) -> str:
    """Key hari trading eksplisit (default WIB Asia/Jakarta UTC+7, konsisten lintas restart)."""
    dt = now_utc if isinstance(now_utc, datetime) else datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    if str(reset_tz or "WIB").upper() == "UTC":
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%d")
    return dt.astimezone(WIB_TZ).strftime("%Y-%m-%d")


def trading_day_start_epoch(day_key: Optional[str] = None, reset_tz: str = DAILY_RESET_TIMEZONE) -> float:
    """Epoch UTC awal hari trading untuk perhitungan daily realized loss & trade counter."""
    key = day_key or current_trading_day_key(reset_tz=reset_tz)
    try:
        y, m, d = (int(x) for x in key.split("-"))
        tz_obj = timezone.utc if str(reset_tz or "WIB").upper() == "UTC" else WIB_TZ
        dt_start = datetime(y, m, d, 0, 0, 0, tzinfo=tz_obj)
        return float(dt_start.timestamp())
    except Exception:
        now = datetime.now(WIB_TZ)
        dt_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return float(dt_start.timestamp())


class ExecutionStateManager:
    """
    Manajer state eksekusi persisten & thread-safe:
    - Menyimpan mode trading, akun terotorisasi, setup lifecycle, tiket order/posisi,
      daily loss counter, daily trade counter, dan riwayat setup SL/completed.
    - Menjamin satu setup tidak pernah dieksekusi ganda oleh dua loop berbeda (in-flight lock).
    - Menjalankan aman kebijakan restart (tidak auto-resume FULL_AUTO akun real tanpa izin).
    """

    def __init__(self, state_file: Optional[str] = None, auto_load: bool = True):
        self.state_file = state_file or FILE_STATE_AUTO_EXECUTION
        self._lock = threading.RLock()
        self._inflight_setups: set = set()
        self.state: Dict[str, Any] = self._default_state()
        if auto_load:
            self.load(apply_startup_safety=True)

    @staticmethod
    def _default_state() -> Dict[str, Any]:
        t_mode = TRADING_MODE if TRADING_MODE in VALID_TRADING_MODES else "SCANNER_ONLY"
        return {
            "schema_version": 1,
            "updated_at": format_wib_timestamp(default_now=True),
            "auto_trading": bool(AUTO_TRADING),
            "trading_mode": t_mode,
            "entry_mode": ENTRY_MODE if ENTRY_MODE in VALID_ENTRY_MODES else "ADAPTIVE",
            "authorized_account": {
                "login": None,
                "server": "",
                "currency": "",
                "account_mode": "",       # REAL | DEMO | CONTEST
                "real_confirmed": False,
                "confirmed_at": "",
            },
            "account_mismatch_halted": False,
            "account_mismatch_reason": "",
            "daily": {
                "day_key": current_trading_day_key(),
                "reset_timezone": DAILY_RESET_TIMEZONE,
                "starting_balance": 0.0,
                "starting_equity": 0.0,
                "realized_pnl": 0.0,
                "floating_pnl": 0.0,
                "trades_count": 0,
                "counted_deals": [],
                "counted_tickets": [],
            },
            "setups": {},                 # setup_id -> dict record
            "executed_setup_ids": [],     # setup_id yang sudah pernah order/fill
            "stopped_out_Zones": {},      # zone_id -> {sl_hit_at, bos_time, setup_id}
            "pending_approvals": {},      # setup_id -> plan dict untuk SEMI_AUTO
            "last_decision_snapshot": {},
            "emergency_halt": False,
            "emergency_reason": "",
        }

    @property
    def lock(self) -> threading.RLock:
        return self._lock

    def load(self, apply_startup_safety: bool = True) -> Dict[str, Any]:
        with self._lock:
            base = self._default_state()
            if os.path.exists(self.state_file):
                try:
                    with open(self.state_file, "r", encoding="utf-8") as f:
                        loaded = json.load(f)
                    if isinstance(loaded, dict):
                        for k, v in loaded.items():
                            if k in ("authorized_account", "daily") and isinstance(v, dict):
                                base[k].update(v)
                            else:
                                base[k] = v
                except Exception as exc:
                    _catat_error("execution_state_load", exc)

            # Pastikan reset harian hanya terjadi jika tanggal kalender berubah (BUKAN karena restart)
            self._ensure_daily_rollover_locked()

            # Aturan Keamanan Akun Real (Section 7 & 20):
            # Jangan pernah otomatis melanjutkan FULL_AUTO pada akun REAL saat aplikasi baru start
            # kecuali kebijakan AUTO_RESUME_REAL_FULL_AUTO diizinkan eksplisit.
            if apply_startup_safety:
                acc_auth = base.get("authorized_account") or {}
                if (
                    str(acc_auth.get("account_mode") or "").upper() == "REAL"
                    and not AUTO_RESUME_REAL_FULL_AUTO
                ):
                    if base.get("trading_mode") == "FULL_AUTO" or base.get("auto_trading"):
                        base["auto_trading"] = False
                        base["trading_mode"] = "SCANNER_ONLY"
                    acc_auth["real_confirmed"] = False
                    base["authorized_account"] = acc_auth

            # Bersihkan status EXECUTING gantung akibat crash sebelumnya -> tandai perlu rekonsiliasi
            setups = base.get("setups")
            if isinstance(setups, dict):
                for sid, rec in setups.items():
                    if isinstance(rec, dict) and rec.get("state") == STATE_EXECUTING:
                        rec["needs_reconciliation"] = True

            self.state = base
            return dict(self.state)

    def save(self) -> bool:
        with self._lock:
            self.state["updated_at"] = format_wib_timestamp(default_now=True)
            tmp_path = f"{self.state_file}.{os.getpid()}.{threading.get_ident()}.tmp"
            try:
                parent = os.path.dirname(os.path.abspath(self.state_file))
                if parent and not os.path.exists(parent):
                    os.makedirs(parent, exist_ok=True)
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(self.state, f, indent=2, ensure_ascii=True)
                os.replace(tmp_path, self.state_file)
                return True
            except Exception as exc:
                _catat_error("execution_state_save", exc)
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except Exception:
                    pass
                return False

    def _ensure_daily_rollover_locked(self, now_utc: Optional[datetime] = None, current_balance: float = 0.0, current_equity: float = 0.0) -> bool:
        daily = self.state.setdefault("daily", {})
        tz_name = str(daily.get("reset_timezone") or DAILY_RESET_TIMEZONE)
        today_key = current_trading_day_key(now_utc=now_utc, reset_tz=tz_name)
        if daily.get("day_key") != today_key:
            daily["day_key"] = today_key
            daily["starting_balance"] = float(current_balance or 0.0)
            daily["starting_equity"] = float(current_equity or current_balance or 0.0)
            daily["realized_pnl"] = 0.0
            daily["floating_pnl"] = 0.0
            daily["trades_count"] = 0
            daily["counted_deals"] = []
            daily["counted_tickets"] = []
            return True
        if current_balance > 0 and float(daily.get("starting_balance") or 0.0) <= 0:
            daily["starting_balance"] = float(current_balance)
        if current_equity > 0 and float(daily.get("starting_equity") or 0.0) <= 0:
            daily["starting_equity"] = float(current_equity)
        return False

    def ensure_daily_rollover(self, now_utc: Optional[datetime] = None, current_balance: float = 0.0, current_equity: float = 0.0) -> bool:
        with self._lock:
            changed = self._ensure_daily_rollover_locked(now_utc=now_utc, current_balance=current_balance, current_equity=current_equity)
            if changed:
                self.save()
            return changed

    # -------------------------------------------------------------------------
    # TRADING MODE & ACCOUNT AUTHORIZATION
    # -------------------------------------------------------------------------
    def get_mode_info(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "auto_trading": bool(self.state.get("auto_trading", False)),
                "trading_mode": str(self.state.get("trading_mode") or "SCANNER_ONLY").upper(),
                "entry_mode": str(self.state.get("entry_mode") or "ADAPTIVE").upper(),
                "authorized_account": dict(self.state.get("authorized_account") or {}),
                "account_mismatch_halted": bool(self.state.get("account_mismatch_halted", False)),
                "account_mismatch_reason": str(self.state.get("account_mismatch_reason") or ""),
                "emergency_halt": bool(self.state.get("emergency_halt", False)),
                "emergency_reason": str(self.state.get("emergency_reason") or ""),
            }

    def set_trading_mode(
        self,
        trading_mode: str,
        auto_trading: Optional[bool] = None,
        entry_mode: Optional[str] = None,
        clear_emergency: bool = False,
    ) -> Dict[str, Any]:
        mode_u = str(trading_mode or "SCANNER_ONLY").strip().upper()
        if mode_u not in VALID_TRADING_MODES:
            raise ValueError(f"Mode trading tidak dikenal: {trading_mode}")
        with self._lock:
            self.state["trading_mode"] = mode_u
            if auto_trading is not None:
                self.state["auto_trading"] = bool(auto_trading)
            elif mode_u == "SCANNER_ONLY":
                self.state["auto_trading"] = False
            if entry_mode is not None and str(entry_mode).upper() in VALID_ENTRY_MODES:
                self.state["entry_mode"] = str(entry_mode).upper()
            if clear_emergency:
                self.state["emergency_halt"] = False
                self.state["emergency_reason"] = ""
            self.save()
            return self.get_mode_info()

    def authorize_account(
        self,
        login: int,
        server: str,
        currency: str,
        account_mode: str,
        real_confirmed: bool = False,
        balance: float = 0.0,
        equity: float = 0.0,
    ) -> Dict[str, Any]:
        with self._lock:
            self.state["authorized_account"] = {
                "login": int(login),
                "server": str(server or ""),
                "currency": str(currency or "").upper(),
                "account_mode": str(account_mode or "").upper(),
                "real_confirmed": bool(real_confirmed),
                "confirmed_at": format_wib_timestamp(default_now=True) if real_confirmed else "",
            }
            self.state["account_mismatch_halted"] = False
            self.state["account_mismatch_reason"] = ""
            self._ensure_daily_rollover_locked(current_balance=balance, current_equity=equity)
            self.save()
            return dict(self.state["authorized_account"])

    def verify_bound_account(self, login: Any, server: str = "") -> Tuple[bool, str]:
        """
        Verifikasi apakah akun MT5 yang sedang login sama dengan akun yang telah diotorisasi.
        Jika belum ada akun terikat, mengikat secara pasif (tanpa real_confirmed) untuk mode non-real/read.
        Jika berubah di tengah jalan -> HENTIKAN eksekusi baru (ACCOUNT_CHANGED).
        """
        if login is None:
            return False, REASON_MT5_DISCONNECTED
        try:
            login_int = int(login)
        except (TypeError, ValueError):
            return False, REASON_MT5_DISCONNECTED

        with self._lock:
            if self.state.get("account_mismatch_halted"):
                return False, REASON_ACCOUNT_CHANGED
            auth = self.state.get("authorized_account") or {}
            bound_login = auth.get("login")
            bound_server = str(auth.get("server") or "").strip()
            cur_server = str(server or "").strip()
            if bound_login is None:
                return True, REASON_OK
            if int(bound_login) != login_int or (bound_server and cur_server and bound_server.lower() != cur_server.lower()):
                msg = f"Akun MT5 berubah dari {bound_login} ({bound_server}) ke {login_int} ({cur_server})"
                self.state["account_mismatch_halted"] = True
                self.state["account_mismatch_reason"] = msg
                self.state["auto_trading"] = False
                self.save()
                return False, REASON_ACCOUNT_CHANGED
            return True, REASON_OK

    def trigger_emergency_halt(self, reason: str = "EMERGENCY_COMMAND") -> None:
        with self._lock:
            self.state["emergency_halt"] = True
            self.state["emergency_reason"] = str(reason or "EMERGENCY_COMMAND")
            self.state["auto_trading"] = False
            self.state["trading_mode"] = "SCANNER_ONLY"
            self.save()

    # -------------------------------------------------------------------------
    # IN-FLIGHT EXECUTION LOCK & DUPLICATE PROTECTION
    # -------------------------------------------------------------------------
    @staticmethod
    def _is_unfilled_expired_pending(rec: Optional[Dict[str, Any]]) -> bool:
        if not isinstance(rec, dict):
            return False
        if rec.get("stopped_out") or rec.get("position_ticket"):
            return False
        st = str(rec.get("state") or "").upper()
        if st in (STATE_EXECUTING, STATE_ORDER_PLACED, STATE_POSITION_ACTIVE, STATE_COMPLETED, STATE_INVALIDATED):
            return False
        return int(rec.get("order_ticket") or 0) == 0 or st == STATE_EXPIRED

    def try_acquire_setup_execution(self, setup_id: str) -> Tuple[bool, str]:
        """
        Atomic check-and-lock untuk memastikan dua loop monitoring paralel tidak pernah
        mengeksekusi setup yang sama secara bersamaan, dan setup yang sudah aktif/selesai
        tidak dieksekusi ulang.
        """
        sid = str(setup_id or "").strip()
        if not sid:
            return False, "INVALID_SETUP_ID"
        with self._lock:
            if sid in self._inflight_setups:
                return False, REASON_DUPLICATE_SETUP
            rec = (self.state.get("setups") or {}).get(sid)
            if isinstance(rec, dict):
                st = str(rec.get("state") or "").upper()
                if st in (STATE_EXECUTING, STATE_ORDER_PLACED, STATE_POSITION_ACTIVE, STATE_COMPLETED):
                    return False, REASON_DUPLICATE_SETUP
                if rec.get("stopped_out"):
                    return False, REASON_STOPPED_OUT_SETUP
            if sid in set(self.state.get("executed_setup_ids") or []) and not self._is_unfilled_expired_pending(rec):
                return False, REASON_DUPLICATE_SETUP
            self._inflight_setups.add(sid)
            self.update_setup_state(sid, state=STATE_EXECUTING, save_now=False)
            return True, REASON_OK

    def release_setup_execution(self, setup_id: str) -> None:
        sid = str(setup_id or "").strip()
        with self._lock:
            self._inflight_setups.discard(sid)

    def find_stage1_for_scale_in(
        self,
        zone_id: str,
        direction: str,
        symbol: str = SYMBOL,
        max_scout_lot: float = 0.01,
    ) -> Tuple[Optional[Dict[str, Any]], bool]:
        """
        Cek apakah suatu zona (80+ / 90+) sedang memiliki posisi Stage-1 (Scout Limit <= max_scout_lot)
        yang SUDAH terisi (POSITION_ACTIVE) dan BELUM pernah mengeksekusi Stage-2 Scale-In:
        Mengembalikan (stage1_record, stage2_blocked_or_done).
        - Jika masih ada pending order belum terisi (ORDER_PLACED / EXECUTING) -> (None, True).
        - Jika Stage-2 sudah pernah jalan / selesai / stopped out -> (None, True).
        - Jika zona tercatat stopped_out_Zones -> (None, True).
        """
        zid = canonical_zone_key(zone_id)
        dir_u = str(direction or "").strip().upper()
        if not zid:
            return None, True

        stage2_sid = make_setup_id(f"{zid}::SCALE_IN", dir_u, symbol=symbol)
        with self._lock:
            if stage2_sid in self._inflight_setups:
                return None, True
            exec_ids = set(self.state.get("executed_setup_ids") or [])
            if stage2_sid in exec_ids:
                return None, True
            stopped_keys = {canonical_zone_key(k) for k in (self.state.get("stopped_out_Zones") or {}).keys()}
            if zid in stopped_keys:
                return None, True

            setups = self.state.get("setups") or {}
            s2_rec = setups.get(stage2_sid)
            if isinstance(s2_rec, dict):
                s2_st = str(s2_rec.get("state") or "").upper()
                if s2_st in (STATE_EXECUTING, STATE_ORDER_PLACED, STATE_POSITION_ACTIVE, STATE_COMPLETED) or s2_rec.get("stopped_out"):
                    return None, True

            stage1_candidate: Optional[Dict[str, Any]] = None
            for rec in setups.values():
                if not isinstance(rec, dict):
                    continue
                if canonical_zone_key(str(rec.get("zone_id") or "")) != zid:
                    continue
                st = str(rec.get("state") or "").upper()
                if st in (STATE_ORDER_PLACED, STATE_EXECUTING):
                    # Pending order Stage-1 belum terjemput -> jangan jalan Stage-2 dulu!
                    return None, True
                if rec.get("stopped_out"):
                    return None, True
                if rec.get("is_scale_in") and st in (STATE_ORDER_PLACED, STATE_POSITION_ACTIVE, STATE_COMPLETED):
                    return None, True
                if st == STATE_POSITION_ACTIVE and not rec.get("is_scale_in"):
                    if str(rec.get("entry_mode") or "").upper() != "EARLY_PENDING":
                        continue
                    r_dir = str(rec.get("direction") or "").upper()
                    if dir_u and r_dir and r_dir != dir_u:
                        continue
                    r_lot = float(rec.get("initial_lot") or rec.get("final_lot") or rec.get("actual_lot") or 0.0)
                    if 0.0 < r_lot <= (float(max_scout_lot) + 1e-6):
                        stage1_candidate = dict(rec)

            return stage1_candidate, (stage1_candidate is None)

    def is_setup_duplicate_or_locked(
        self,
        setup_id: str,
        zone_id: str = "",
        zone_bos_time: str = "",
        allow_scale_in_stage2: bool = False,
    ) -> Tuple[bool, str]:
        sid = str(setup_id or "").strip()
        zid = canonical_zone_key(zone_id)
        with self._lock:
            if sid in self._inflight_setups:
                return True, REASON_DUPLICATE_SETUP
            setups = self.state.get("setups") or {}
            rec = setups.get(sid) if sid else None
            if sid and sid in set(self.state.get("executed_setup_ids") or []) and not self._is_unfilled_expired_pending(rec):
                return True, REASON_DUPLICATE_SETUP
            if isinstance(rec, dict):
                st = str(rec.get("state") or "").upper()
                if st in (STATE_EXECUTING, STATE_ORDER_PLACED, STATE_POSITION_ACTIVE, STATE_COMPLETED):
                    return True, REASON_DUPLICATE_SETUP
                if rec.get("stopped_out"):
                    return True, REASON_STOPPED_OUT_SETUP
            # Cek juga apakah zona yang sama sedang memiliki order/posisi aktif di setup record lain
            if zid:
                active_non_scale_cnt = 0
                for other_sid, other_rec in setups.items():
                    if not isinstance(other_rec, dict):
                        continue
                    if canonical_zone_key(str(other_rec.get("zone_id") or "")) == zid:
                        o_st = str(other_rec.get("state") or "").upper()
                        if allow_scale_in_stage2:
                            if o_st in (STATE_EXECUTING, STATE_ORDER_PLACED):
                                return True, REASON_DUPLICATE_SETUP
                            if other_rec.get("is_scale_in") and o_st in (
                                STATE_EXECUTING, STATE_ORDER_PLACED, STATE_POSITION_ACTIVE, STATE_COMPLETED
                            ):
                                return True, REASON_DUPLICATE_SETUP
                            if o_st == STATE_POSITION_ACTIVE and not other_rec.get("is_scale_in"):
                                active_non_scale_cnt += 1
                                if active_non_scale_cnt > 1:
                                    return True, REASON_DUPLICATE_SETUP
                        else:
                            if o_st in (STATE_EXECUTING, STATE_ORDER_PLACED, STATE_POSITION_ACTIVE):
                                return True, REASON_DUPLICATE_SETUP
                stopped_map = self.state.get("stopped_out_Zones") or {}
                for stop_zid, prev_stop_raw in stopped_map.items():
                    if canonical_zone_key(stop_zid) == zid:
                        prev_stop = prev_stop_raw or {}
                        prev_bos = str(prev_stop.get("bos_time") or "")
                        # Jangan buka ulang setup yang sama setelah terkena SL kecuali terdapat pembentukan struktur baru
                        if not zone_bos_time or not prev_bos or zone_bos_time == prev_bos:
                            return True, REASON_STOPPED_OUT_SETUP
            return False, REASON_OK

    # -------------------------------------------------------------------------
    # SETUP STATE UPDATES & COUNTERS
    # -------------------------------------------------------------------------
    def get_setup(self, setup_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            rec = (self.state.get("setups") or {}).get(str(setup_id or ""))
            return dict(rec) if isinstance(rec, dict) else None

    def get_all_setups(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {k: dict(v) for k, v in (self.state.get("setups") or {}).items() if isinstance(v, dict)}

    def get_active_scanner_records(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Kembalikan (active_positions_records, active_pending_records) dari state internal."""
        with self._lock:
            pos_recs = []
            ord_recs = []
            for rec in (self.state.get("setups") or {}).values():
                if not isinstance(rec, dict):
                    continue
                st = str(rec.get("state") or "").upper()
                if st == STATE_POSITION_ACTIVE:
                    pos_recs.append(dict(rec))
                elif st == STATE_ORDER_PLACED:
                    ord_recs.append(dict(rec))
            return pos_recs, ord_recs

    def update_setup_state(
        self,
        setup_id: str,
        state: str,
        reason_code: str = "",
        reason_text: str = "",
        zone_id: str = "",
        symbol: str = SYMBOL,
        direction: str = "",
        entry_mode: str = "",
        sub_mode: str = "",
        order_ticket: Optional[int] = None,
        position_ticket: Optional[int] = None,
        planned_entry: Optional[float] = None,
        actual_entry: Optional[float] = None,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        target_lot: Optional[float] = None,
        final_lot: Optional[float] = None,
        actual_lot: Optional[float] = None,
        zone_score: Optional[float] = None,
        zone_grade: str = "",
        zone_quality: str = "",
        zone_timeframe: str = "",
        trigger_name: str = "",
        bos_time: str = "",
        extra: Optional[Dict[str, Any]] = None,
        save_now: bool = True,
    ) -> Dict[str, Any]:
        st_u = str(state or STATE_RADAR).upper()
        if st_u not in EXECUTION_STATES:
            st_u = STATE_BLOCKED
        sid = str(setup_id or "").strip()
        now_str = format_wib_timestamp(default_now=True)
        with self._lock:
            setups = self.state.setdefault("setups", {})
            rec = setups.get(sid)
            if not isinstance(rec, dict):
                rec = {
                    "setup_id": sid,
                    "created_at": now_str,
                    "created_epoch": time.time(),
                    "history": [],
                }
                setups[sid] = rec

            prev_state = rec.get("state")
            rec["state"] = st_u
            rec["updated_at"] = now_str
            rec["updated_epoch"] = time.time()
            if reason_code:
                rec["reason_code"] = str(reason_code)
            if reason_text:
                rec["reason_text"] = str(reason_text)
            if zone_id:
                rec["zone_id"] = str(zone_id)
            if symbol:
                rec["symbol"] = str(symbol)
            if direction:
                rec["direction"] = str(direction).upper()
            if entry_mode:
                rec["entry_mode"] = str(entry_mode).upper()
            if sub_mode:
                rec["sub_mode"] = str(sub_mode).upper()
            if order_ticket is not None:
                rec["order_ticket"] = int(order_ticket)
            if position_ticket is not None:
                rec["position_ticket"] = int(position_ticket)
            if planned_entry is not None:
                rec["planned_entry"] = float(planned_entry)
            if actual_entry is not None:
                rec["actual_entry"] = float(actual_entry)
            if sl is not None:
                rec["sl"] = float(sl)
            if tp is not None:
                rec["tp"] = float(tp)
            if target_lot is not None:
                rec["target_lot"] = float(target_lot)
            if final_lot is not None:
                rec["final_lot"] = float(final_lot)
            if actual_lot is not None:
                rec["actual_lot"] = float(actual_lot)
            if zone_score is not None:
                rec["zone_score"] = float(zone_score)
            if zone_grade:
                rec["zone_grade"] = str(zone_grade)
            if zone_quality:
                rec["zone_quality"] = str(zone_quality)
            if zone_timeframe:
                rec["zone_timeframe"] = str(zone_timeframe)
            if trigger_name:
                rec["trigger_name"] = str(trigger_name)
            if bos_time:
                rec["bos_time"] = str(bos_time)
            if isinstance(extra, dict):
                for k, v in extra.items():
                    rec[k] = v

            if st_u == STATE_ORDER_PLACED and prev_state != STATE_ORDER_PLACED:
                rec["order_placed_epoch"] = time.time()

            if st_u in (STATE_ORDER_PLACED, STATE_POSITION_ACTIVE):
                exec_list = self.state.setdefault("executed_setup_ids", [])
                if sid and sid not in exec_list:
                    exec_list.append(sid)
                    if len(exec_list) > 500:
                        del exec_list[:-500]
                rec["needs_reconciliation"] = False
            elif self._is_unfilled_expired_pending(rec):
                exec_list = self.state.get("executed_setup_ids")
                if isinstance(exec_list, list) and sid in exec_list:
                    exec_list.remove(sid)

            if prev_state != st_u:
                hist = rec.setdefault("history", [])
                hist.append({
                    "from": prev_state,
                    "to": st_u,
                    "at": now_str,
                    "reason": reason_code or "",
                })
                if len(hist) > 25:
                    del hist[:-25]

            # Pangkas setup lama terminal agar file state tetap ramping
            if len(setups) > 250:
                removable = [
                    (k, float(v.get("updated_epoch") or 0.0))
                    for k, v in setups.items()
                    if isinstance(v, dict) and v.get("state") in (STATE_RADAR, STATE_COMPLETED, STATE_INVALIDATED, STATE_EXPIRED, STATE_BLOCKED)
                ]
                removable.sort(key=lambda item: item[1])
                for k_rem, _ in removable[:max(0, len(setups) - 200)]:
                    setups.pop(k_rem, None)

            if save_now:
                self.save()
            return dict(rec)

    def record_trade_opened(self, setup_id: str, ticket: int) -> int:
        """Increment counter transaksi harian (persisten lintas restart, anti double-count per ticket)."""
        with self._lock:
            self._ensure_daily_rollover_locked()
            daily = self.state.setdefault("daily", {})
            counted = daily.setdefault("counted_tickets", [])
            tkt_int = int(ticket) if ticket else 0
            key_id = str(tkt_int) if tkt_int > 0 else str(setup_id)
            if key_id not in counted:
                counted.append(key_id)
                daily["trades_count"] = int(daily.get("trades_count") or 0) + 1
            self.save()
            return int(daily.get("trades_count") or 0)

    def record_realized_deal(
        self,
        deal_id: Any,
        net_pnl: float,
        setup_id: str = "",
        zone_id: str = "",
        hit_sl: bool = False,
        bos_time: str = "",
    ) -> float:
        """Catat realized P/L (termasuk komisi & swap) dan tandai setup yang kena SL agar tidak dibuka ulang."""
        with self._lock:
            self._ensure_daily_rollover_locked()
            daily = self.state.setdefault("daily", {})
            deals = daily.setdefault("counted_deals", [])
            d_key = str(deal_id or "")
            if d_key and d_key not in deals:
                deals.append(d_key)
                if math.isfinite(float(net_pnl)):
                    daily["realized_pnl"] = round(float(daily.get("realized_pnl") or 0.0) + float(net_pnl), 4)
            elif not d_key and math.isfinite(float(net_pnl)):
                daily["realized_pnl"] = round(float(daily.get("realized_pnl") or 0.0) + float(net_pnl), 4)

            if hit_sl:
                if setup_id and setup_id in (self.state.get("setups") or {}):
                    self.state["setups"][setup_id]["stopped_out"] = True
                if zone_id:
                    stopped_map = self.state.setdefault("stopped_out_Zones", {})
                    stopped_map[str(zone_id)] = {
                        "sl_hit_at": format_wib_timestamp(default_now=True),
                        "bos_time": str(bos_time or ""),
                        "setup_id": str(setup_id or ""),
                    }
            self.save()
            return float(daily.get("realized_pnl") or 0.0)

    def sync_daily_metrics_from_mt5(
        self,
        mt5_realized_pnl: Optional[float] = None,
        mt5_floating_pnl: Optional[float] = None,
        mt5_deal_ids: Optional[List[Any]] = None,
        mt5_trades_opened_today: Optional[int] = None,
        current_balance: float = 0.0,
        current_equity: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Sinkronisasi konservatif antara persisted counter lokal dan histori aktual broker MT5:
        - Mengambil kerugian harian terburuk (agar restart atau deal MT5 luar memori tidak pernah mengecilkan kerugian).
        - Mengambil jumlah transaksi tertinggi antara catatan lokal dan histori MT5 hari ini.
        """
        with self._lock:
            self._ensure_daily_rollover_locked(current_balance=current_balance, current_equity=current_equity)
            daily = self.state.setdefault("daily", {})
            if current_balance > 0 and mt5_realized_pnl is not None:
                inferred_start = float(current_balance) - float(mt5_realized_pnl)
                if inferred_start > 0 and float(daily.get("starting_balance") or 0.0) <= 0:
                    daily["starting_balance"] = round(inferred_start, 2)
                    daily["starting_equity"] = round(inferred_start, 2)

            if mt5_realized_pnl is not None and math.isfinite(float(mt5_realized_pnl)):
                persisted_pnl = float(daily.get("realized_pnl") or 0.0)
                if mt5_deal_ids:
                    # Jika daftar deal hari ini lengkap dari MT5, gabungkan secara konservatif (pilih nilai lebih negatif jika rugi)
                    if float(mt5_realized_pnl) < persisted_pnl:
                        daily["realized_pnl"] = round(float(mt5_realized_pnl), 4)
                    existing_deals = set(str(x) for x in (daily.get("counted_deals") or []))
                    for did in mt5_deal_ids:
                        if str(did) not in existing_deals:
                            existing_deals.add(str(did))
                    daily["counted_deals"] = sorted(existing_deals)[-300:]
                else:
                    if float(mt5_realized_pnl) < persisted_pnl:
                        daily["realized_pnl"] = round(float(mt5_realized_pnl), 4)

            if mt5_floating_pnl is not None and math.isfinite(float(mt5_floating_pnl)):
                daily["floating_pnl"] = round(float(mt5_floating_pnl), 4)

            if mt5_trades_opened_today is not None:
                daily["trades_count"] = max(int(daily.get("trades_count") or 0), int(mt5_trades_opened_today))

            return dict(daily)

    def get_daily_summary(self) -> Dict[str, Any]:
        with self._lock:
            self._ensure_daily_rollover_locked()
            return dict(self.state.get("daily") or {})

    # -------------------------------------------------------------------------
    # SEMI_AUTO APPROVAL QUEUE
    # -------------------------------------------------------------------------
    def queue_semi_auto_setup(self, setup_id: str, payload: Dict[str, Any]) -> None:
        with self._lock:
            approvals = self.state.setdefault("pending_approvals", {})
            approvals[str(setup_id)] = {
                **payload,
                "queued_at": format_wib_timestamp(default_now=True),
                "queued_epoch": time.time(),
            }
            self.save()

    def pop_semi_auto_setup(self, setup_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        with self._lock:
            approvals = self.state.get("pending_approvals") or {}
            if not approvals:
                return None
            target_key = str(setup_id) if setup_id else next(iter(approvals.keys()))
            item = approvals.pop(target_key, None)
            if item is not None:
                self.save()
            return item

    def get_pending_approvals(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {k: dict(v) for k, v in (self.state.get("pending_approvals") or {}).items() if isinstance(v, dict)}

    def record_last_decision(self, snapshot_summary: Dict[str, Any]) -> None:
        with self._lock:
            self.state["last_decision_snapshot"] = dict(snapshot_summary or {})

    def get_last_decision(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self.state.get("last_decision_snapshot") or {})

    # -------------------------------------------------------------------------
    # STARTUP & RECONNECT RECONCILIATION (SECTION 12 & 20)
    # -------------------------------------------------------------------------
    def reconcile_with_mt5(
        self,
        positions: List[Any],
        orders: List[Any],
        history_deals: Optional[List[Any]] = None,
        magic_number: int = SCANNER_MAGIC_NUMBER,
    ) -> Dict[str, Any]:
        """
        Rekonsiliasi penuh antara setup state lokal dengan posisi aktif, pending order,
        dan riwayat deal aktual pada broker MT5:
        - Order pending yang sudah terisi -> transisi otomatis ke POSITION_ACTIVE.
        - Posisi aktif yang sudah tertutup -> transisi ke COMPLETED dan catat P/L + SL status.
        - Posisi/order milik scanner (magic number cocok) yang belum tercatat di state -> adopsi ke state
          agar tidak terjadi duplicate order setelah restart.
        """
        pos_by_ticket: Dict[int, Any] = {}
        pos_by_comment: Dict[str, Any] = {}
        scanner_positions: List[Any] = []
        manual_positions: List[Any] = []

        for p in (positions or []):
            tkt = int(getattr(p, "ticket", 0) or (p.get("ticket", 0) if isinstance(p, dict) else 0) or 0)
            mgc = int(getattr(p, "magic", 0) or (p.get("magic", 0) if isinstance(p, dict) else 0) or 0)
            cmt = str(getattr(p, "comment", "") or (p.get("comment", "") if isinstance(p, dict) else "") or "")
            if tkt > 0:
                pos_by_ticket[tkt] = p
            if cmt:
                pos_by_comment[cmt] = p
            if mgc == int(magic_number) or cmt.startswith("SMC_"):
                scanner_positions.append(p)
            else:
                manual_positions.append(p)

        ord_by_ticket: Dict[int, Any] = {}
        ord_by_comment: Dict[str, Any] = {}
        scanner_orders: List[Any] = []
        manual_orders: List[Any] = []

        for o in (orders or []):
            tkt = int(getattr(o, "ticket", 0) or (o.get("ticket", 0) if isinstance(o, dict) else 0) or 0)
            mgc = int(getattr(o, "magic", 0) or (o.get("magic", 0) if isinstance(o, dict) else 0) or 0)
            cmt = str(getattr(o, "comment", "") or (o.get("comment", "") if isinstance(o, dict) else "") or "")
            if tkt > 0:
                ord_by_ticket[tkt] = o
            if cmt:
                ord_by_comment[cmt] = o
            if mgc == int(magic_number) or cmt.startswith("SMC_"):
                scanner_orders.append(o)
            else:
                manual_orders.append(o)

        deals_by_pos: Dict[int, List[Any]] = {}
        for d in (history_deals or []):
            pid = int(getattr(d, "position_id", 0) or (d.get("position_id", 0) if isinstance(d, dict) else 0) or 0)
            if pid > 0:
                deals_by_pos.setdefault(pid, []).append(d)

        transitions: List[Dict[str, Any]] = []
        adopted_tickets: List[int] = []

        with self._lock:
            setups = self.state.setdefault("setups", {})
            known_pos_tickets = set()
            known_ord_tickets = set()

            for sid, rec in list(setups.items()):
                if not isinstance(rec, dict):
                    continue
                st = str(rec.get("state") or "").upper()
                o_tkt = int(rec.get("order_ticket") or 0)
                p_tkt = int(rec.get("position_ticket") or 0)
                exp_cmt = encode_order_comment(sid, rec.get("entry_mode", "ADAPTIVE"))

                # 1. Cek jika setup menggantung di EXECUTING (ambiguous execution recovery)
                if st == STATE_EXECUTING or rec.get("needs_reconciliation"):
                    matched_pos = (
                        pos_by_ticket.get(p_tkt)
                        or pos_by_ticket.get(o_tkt)
                        or pos_by_comment.get(exp_cmt)
                    )
                    matched_ord = ord_by_ticket.get(o_tkt) or ord_by_comment.get(exp_cmt)
                    if matched_pos is not None:
                        pt = int(getattr(matched_pos, "ticket", 0) or matched_pos.get("ticket", 0))
                        px = float(getattr(matched_pos, "price_open", 0.0) or matched_pos.get("price_open", 0.0))
                        vol = float(getattr(matched_pos, "volume", 0.0) or matched_pos.get("volume", 0.0))
                        sl_v = float(getattr(matched_pos, "sl", 0.0) or matched_pos.get("sl", 0.0))
                        tp_v = float(getattr(matched_pos, "tp", 0.0) or matched_pos.get("tp", 0.0))
                        self.update_setup_state(
                            sid, STATE_POSITION_ACTIVE, reason_code="RECONCILED_ACTIVE_POSITION",
                            position_ticket=pt, actual_entry=px, actual_lot=vol, sl=sl_v, tp=tp_v, save_now=False,
                        )
                        known_pos_tickets.add(pt)
                        transitions.append({"setup_id": sid, "from": st, "to": STATE_POSITION_ACTIVE, "ticket": pt})
                        continue
                    if matched_ord is not None:
                        ot = int(getattr(matched_ord, "ticket", 0) or matched_ord.get("ticket", 0))
                        self.update_setup_state(
                            sid, STATE_ORDER_PLACED, reason_code="RECONCILED_PENDING_ORDER",
                            order_ticket=ot, save_now=False,
                        )
                        known_ord_tickets.add(ot)
                        transitions.append({"setup_id": sid, "from": st, "to": STATE_ORDER_PLACED, "ticket": ot})
                        continue
                    rec["needs_reconciliation"] = False
                    if st == STATE_EXECUTING and sid not in self._inflight_setups:
                        self.update_setup_state(
                            sid, STATE_BLOCKED, reason_code="RECONCILED_NOT_EXECUTED",
                            reason_text="Order tidak ditemukan di broker pasca-rekonsiliasi", save_now=False,
                        )
                        transitions.append({"setup_id": sid, "from": st, "to": STATE_BLOCKED, "ticket": 0})
                    continue

                # 2. Cek jika ORDER_PLACED sudah terisi menjadi posisi aktif atau sudah batal/expired
                if st == STATE_ORDER_PLACED:
                    if o_tkt in ord_by_ticket:
                        known_ord_tickets.add(o_tkt)
                        continue
                    matched_pos = pos_by_ticket.get(o_tkt) or pos_by_ticket.get(p_tkt) or pos_by_comment.get(exp_cmt)
                    if matched_pos is not None:
                        pt = int(getattr(matched_pos, "ticket", 0) or matched_pos.get("ticket", 0))
                        px = float(getattr(matched_pos, "price_open", 0.0) or matched_pos.get("price_open", 0.0))
                        vol = float(getattr(matched_pos, "volume", 0.0) or matched_pos.get("volume", 0.0))
                        sl_v = float(getattr(matched_pos, "sl", 0.0) or matched_pos.get("sl", 0.0))
                        tp_v = float(getattr(matched_pos, "tp", 0.0) or matched_pos.get("tp", 0.0))
                        self.update_setup_state(
                            sid, STATE_POSITION_ACTIVE, reason_code="PENDING_FILLED",
                            position_ticket=pt, actual_entry=px, actual_lot=vol, sl=sl_v, tp=tp_v, save_now=False,
                        )
                        self.record_trade_opened(sid, pt)
                        known_pos_tickets.add(pt)
                        transitions.append({"setup_id": sid, "from": st, "to": STATE_POSITION_ACTIVE, "ticket": pt})
                    else:
                        # Cek apakah sudah sempat fill & langsung close di history_deals
                        p_deals = deals_by_pos.get(o_tkt, [])
                        if p_deals:
                            pnl_sum = sum(
                                float(getattr(d, "profit", 0.0) or 0.0)
                                + float(getattr(d, "commission", 0.0) or 0.0)
                                + float(getattr(d, "swap", 0.0) or 0.0)
                                + float(getattr(d, "fee", 0.0) or 0.0)
                                for d in p_deals
                            )
                            sl_hit = any(int(getattr(d, "reason", -1) or -1) == 4 for d in p_deals) or (pnl_sum < -0.01)
                            for d in p_deals:
                                if int(getattr(d, "entry", -1) or -1) in (1, 2, 3):
                                    d_net = (
                                        float(getattr(d, "profit", 0.0) or 0.0)
                                        + float(getattr(d, "commission", 0.0) or 0.0)
                                        + float(getattr(d, "swap", 0.0) or 0.0)
                                        + float(getattr(d, "fee", 0.0) or 0.0)
                                    )
                                    self.record_realized_deal(
                                        getattr(d, "ticket", ""), d_net, setup_id=sid,
                                        zone_id=rec.get("zone_id", ""), hit_sl=sl_hit, bos_time=rec.get("bos_time", ""),
                                    )
                            self.update_setup_state(
                                sid, STATE_COMPLETED, reason_code="SL_HIT" if sl_hit else "TP_OR_MANUAL_CLOSE",
                                position_ticket=o_tkt, extra={"realized_pnl": round(pnl_sum, 2)}, save_now=False,
                            )
                            transitions.append({"setup_id": sid, "from": st, "to": STATE_COMPLETED, "ticket": o_tkt, "pnl": pnl_sum})
                        else:
                            self.update_setup_state(
                                sid, STATE_EXPIRED, reason_code="PENDING_CANCELLED_OR_EXPIRED", save_now=False,
                            )
                            transitions.append({"setup_id": sid, "from": st, "to": STATE_EXPIRED, "ticket": o_tkt})
                    continue

                # 3. Cek jika POSITION_ACTIVE sudah tertutup
                if st == STATE_POSITION_ACTIVE:
                    lookup_tkt = p_tkt or o_tkt
                    if lookup_tkt in pos_by_ticket:
                        known_pos_tickets.add(lookup_tkt)
                        cur_p = pos_by_ticket[lookup_tkt]
                        cur_vol = float(getattr(cur_p, "volume", 0.0) or (cur_p.get("volume", 0.0) if isinstance(cur_p, dict) else 0.0))
                        cur_sl = float(getattr(cur_p, "sl", 0.0) or (cur_p.get("sl", 0.0) if isinstance(cur_p, dict) else 0.0))
                        cur_tp = float(getattr(cur_p, "tp", 0.0) or (cur_p.get("tp", 0.0) if isinstance(cur_p, dict) else 0.0))
                        rec["actual_lot"] = cur_vol
                        rec["sl"] = cur_sl
                        rec["tp"] = cur_tp
                        continue

                    p_deals = deals_by_pos.get(lookup_tkt, [])
                    pnl_sum = 0.0
                    exit_reason = "CLOSED"
                    sl_hit = False
                    for d in p_deals:
                        d_entry = int(getattr(d, "entry", -1) if not isinstance(d, dict) else d.get("entry", -1))
                        if d_entry in (1, 2, 3):  # DEAL_ENTRY_OUT / INOUT / OUT_BY
                            d_net = (
                                float(getattr(d, "profit", 0.0) if not isinstance(d, dict) else d.get("profit", 0.0))
                                + float(getattr(d, "commission", 0.0) if not isinstance(d, dict) else d.get("commission", 0.0))
                                + float(getattr(d, "swap", 0.0) if not isinstance(d, dict) else d.get("swap", 0.0))
                                + float(getattr(d, "fee", 0.0) if not isinstance(d, dict) else d.get("fee", 0.0))
                            )
                            pnl_sum += d_net
                            d_rsn = int(getattr(d, "reason", -1) if not isinstance(d, dict) else d.get("reason", -1))
                            if d_rsn == 4:
                                exit_reason = "SL_HIT"
                                sl_hit = True
                            elif d_rsn == 5:
                                exit_reason = "TP_HIT"
                            d_tkt = getattr(d, "ticket", "") if not isinstance(d, dict) else d.get("ticket", "")
                            self.record_realized_deal(
                                d_tkt, d_net, setup_id=sid,
                                zone_id=rec.get("zone_id", ""), hit_sl=sl_hit, bos_time=rec.get("bos_time", ""),
                            )
                    if not p_deals and pnl_sum == 0.0:
                        exit_reason = "CLOSED_RECONCILED"
                    self.update_setup_state(
                        sid, STATE_COMPLETED, reason_code=exit_reason,
                        extra={"realized_pnl": round(pnl_sum, 2), "exit_reason": exit_reason}, save_now=False,
                    )
                    transitions.append({
                        "setup_id": sid, "from": st, "to": STATE_COMPLETED,
                        "ticket": lookup_tkt, "pnl": round(pnl_sum, 2), "exit_reason": exit_reason,
                    })

            # 4. Adopsi posisi aktif milik scanner yang belum ada di state (mis. file state sempat terhapus)
            for sp in scanner_positions:
                pt = int(getattr(sp, "ticket", 0) or (sp.get("ticket", 0) if isinstance(sp, dict) else 0) or 0)
                if pt <= 0 or pt in known_pos_tickets:
                    continue
                p_type = int(getattr(sp, "type", 0) if not isinstance(sp, dict) else sp.get("type", 0))
                p_dir = "BUY" if p_type == 0 else "SELL"
                p_sym = str(getattr(sp, "symbol", SYMBOL) if not isinstance(sp, dict) else sp.get("symbol", SYMBOL))
                p_px = float(getattr(sp, "price_open", 0.0) if not isinstance(sp, dict) else sp.get("price_open", 0.0))
                p_vol = float(getattr(sp, "volume", 0.0) if not isinstance(sp, dict) else sp.get("volume", 0.0))
                p_sl = float(getattr(sp, "sl", 0.0) if not isinstance(sp, dict) else sp.get("sl", 0.0))
                p_tp = float(getattr(sp, "tp", 0.0) if not isinstance(sp, dict) else sp.get("tp", 0.0))
                adopted_sid = f"ADOPTED_POS_{pt}"
                self.update_setup_state(
                    adopted_sid, STATE_POSITION_ACTIVE, reason_code="ADOPTED_SCANNER_POSITION",
                    symbol=p_sym, direction=p_dir, position_ticket=pt,
                    actual_entry=p_px, actual_lot=p_vol, final_lot=p_vol, sl=p_sl, tp=p_tp, save_now=False,
                )
                adopted_tickets.append(pt)

            for so in scanner_orders:
                ot = int(getattr(so, "ticket", 0) or (so.get("ticket", 0) if isinstance(so, dict) else 0) or 0)
                if ot <= 0 or ot in known_ord_tickets:
                    continue
                o_type = int(getattr(so, "type", 2) if not isinstance(so, dict) else so.get("type", 2))
                o_dir = "BUY" if o_type in (2, 4, 6) else "SELL"
                o_sym = str(getattr(so, "symbol", SYMBOL) if not isinstance(so, dict) else so.get("symbol", SYMBOL))
                o_px = float(getattr(so, "price_open", 0.0) if not isinstance(so, dict) else so.get("price_open", 0.0))
                o_vol = float(getattr(so, "volume_current", getattr(so, "volume_initial", 0.0)) if not isinstance(so, dict) else so.get("volume", 0.0))
                o_sl = float(getattr(so, "sl", 0.0) if not isinstance(so, dict) else so.get("sl", 0.0))
                o_tp = float(getattr(so, "tp", 0.0) if not isinstance(so, dict) else so.get("tp", 0.0))
                adopted_sid = f"ADOPTED_ORD_{ot}"
                self.update_setup_state(
                    adopted_sid, STATE_ORDER_PLACED, reason_code="ADOPTED_SCANNER_ORDER",
                    symbol=o_sym, direction=o_dir, entry_mode="EARLY_PENDING", order_ticket=ot,
                    planned_entry=o_px, final_lot=o_vol, sl=o_sl, tp=o_tp, save_now=False,
                )
                adopted_tickets.append(ot)

            self.save()

        return {
            "scanner_positions": scanner_positions,
            "manual_positions": manual_positions,
            "scanner_orders": scanner_orders,
            "manual_orders": manual_orders,
            "transitions": transitions,
            "adopted_tickets": adopted_tickets,
        }


_GLOBAL_EXEC_STATE_LOCK = threading.Lock()
_GLOBAL_EXEC_STATE: Optional[ExecutionStateManager] = None


def get_execution_state_manager(state_file: Optional[str] = None, force_reload: bool = False) -> ExecutionStateManager:
    """Singleton accessor untuk ExecutionStateManager utama."""
    global _GLOBAL_EXEC_STATE
    with _GLOBAL_EXEC_STATE_LOCK:
        if state_file and (_GLOBAL_EXEC_STATE is None or _GLOBAL_EXEC_STATE.state_file != state_file):
            return ExecutionStateManager(state_file=state_file, auto_load=True)
        if _GLOBAL_EXEC_STATE is None:
            _GLOBAL_EXEC_STATE = ExecutionStateManager(auto_load=True)
        elif force_reload:
            _GLOBAL_EXEC_STATE.load(apply_startup_safety=False)
        return _GLOBAL_EXEC_STATE
