# -*- coding: utf-8 -*-
"""
modules/execution_engine.py
MT5 Auto Execution Engine (Section 12):
- Integrasi langsung MetaTrader5 API untuk BUY MARKET, SELL MARKET, BUY LIMIT, SELL LIMIT,
  pembatalan pending order, modifikasi SL/TP, penutupan darurat posisi scanner, dan rekonsiliasi.
- Revalidasi harga live (Ask untuk BUY MARKET, Bid untuk SELL MARKET), spread, SL/TP, RR, volume step,
  dan margin tepat sebelum mt5.order_check() -> mt5.order_send().
- Penanganan partial fill, rejection, requote, invalid stops, insufficient margin, market closed, dan timeout.
- Rekonsiliasi terhadap orders_get, positions_get, dan history_deals_get pada hasil ambigu (tanpa blind retry).
- Menjaga kepemilikan ketat (magic number scanner) sehingga order/posisi manual user tidak pernah disentuh.
"""
from __future__ import annotations

import math
import threading
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

from modules.config import SYMBOL, get_auto_trading_config
from modules.dynamic_lot_manager import is_broker_volume_valid
from modules.execution_state import (
    ExecutionStateManager,
    REASON_ALGO_TRADING_DISABLED,
    REASON_BROKER_CONSTRAINTS,
    REASON_DUPLICATE_SETUP,
    REASON_EXECUTION_AMBIGUOUS,
    REASON_EXECUTION_REJECTED,
    REASON_INSUFFICIENT_MARGIN,
    REASON_INVALID_STOP_LOSS,
    REASON_INVALID_VOLUME,
    REASON_MARKET_CLOSED,
    REASON_MT5_DISCONNECTED,
    REASON_OK,
    REASON_PRICE_LEFT_ENTRY_RANGE,
    REASON_SPREAD_TOO_HIGH,
    REASON_ZONE_EXPIRED,
    REASON_ZONE_INVALIDATED,
    STATE_BLOCKED,
    STATE_COMPLETED,
    STATE_ORDER_PLACED,
    STATE_POSITION_ACTIVE,
    encode_order_comment,
)
from modules.mt5_conn import MT5_IPC_LOCK
from modules.risk_manager import inspect_mt5_account_and_permissions, validate_sl_tp_and_rr
from modules.utils import _catat_error, get_symbol_unit_spec


_EXEC_SEND_LOCK = threading.RLock()


# Kode retcode standar MT5
RETCODE_REQUOTE = 10004
RETCODE_REJECT = 10006
RETCODE_CANCEL = 10007
RETCODE_PLACED = 10008
RETCODE_DONE = 10009
RETCODE_DONE_PARTIAL = 10010
RETCODE_ERROR = 10011
RETCODE_TIMEOUT = 10012
RETCODE_INVALID = 10013
RETCODE_INVALID_VOLUME = 10014
RETCODE_INVALID_PRICE = 10015
RETCODE_INVALID_STOPS = 10016
RETCODE_TRADE_DISABLED = 10017
RETCODE_MARKET_CLOSED = 10018
RETCODE_NO_MONEY = 10019
RETCODE_PRICE_CHANGED = 10020
RETCODE_PRICE_OFF = 10021
RETCODE_INVALID_EXPIRATION = 10022
RETCODE_AUTOTRADING_DISABLED = 10027
RETCODE_CONNECTION = 10031


def _safe_retcode(res_obj: Any) -> int:
    """Ambil retcode integer secara aman tanpa mengubah nilai valid 0 menjadi -1."""
    if res_obj is None:
        return -1
    rc = getattr(res_obj, "retcode", None)
    if rc is None:
        return -1
    try:
        return int(rc)
    except (TypeError, ValueError):
        return -1


def map_mt5_retcode_to_reason(retcode: Any) -> str:
    """Petakan retcode MT5 ke canonical reason code sistem."""
    try:
        rc = int(retcode)
    except (TypeError, ValueError):
        return REASON_EXECUTION_AMBIGUOUS

    if rc in (0, RETCODE_DONE, RETCODE_PLACED, RETCODE_DONE_PARTIAL):
        return REASON_OK
    if rc == RETCODE_INVALID_VOLUME:
        return REASON_INVALID_VOLUME
    if rc == RETCODE_INVALID_STOPS:
        return REASON_INVALID_STOP_LOSS
    if rc == RETCODE_NO_MONEY:
        return REASON_INSUFFICIENT_MARGIN
    if rc == RETCODE_MARKET_CLOSED:
        return REASON_MARKET_CLOSED
    if rc in (RETCODE_TRADE_DISABLED, RETCODE_AUTOTRADING_DISABLED):
        return REASON_ALGO_TRADING_DISABLED
    if rc == RETCODE_CONNECTION:
        return REASON_MT5_DISCONNECTED
    if rc == RETCODE_TIMEOUT:
        return REASON_EXECUTION_AMBIGUOUS
    return REASON_EXECUTION_REJECTED


def is_scanner_owned_item(item: Any, magic_number: int) -> bool:
    """Pastikan order/posisi benar-benar milik scanner (bukan order/posisi manual user)."""
    mgc = int(getattr(item, "magic", 0) or (item.get("magic", 0) if isinstance(item, dict) else 0) or 0)
    cmt = str(getattr(item, "comment", "") or (item.get("comment", "") if isinstance(item, dict) else "") or "")
    return (mgc == int(magic_number)) or cmt.startswith("SMC_")


def resolve_broker_filling_mode(mt5_api: Any, sym_info: Any) -> int:
    """
    Pilih filling mode yang didukung simbol broker secara dinamis (bitmask symbol_info.filling_mode):
    1 = FOK, 2 = IOC, lainnya = RETURN.
    """
    fok = getattr(mt5_api, "ORDER_FILLING_FOK", 0)
    ioc = getattr(mt5_api, "ORDER_FILLING_IOC", 1)
    ret = getattr(mt5_api, "ORDER_FILLING_RETURN", 2)
    if sym_info is None:
        return ioc
    mask = int(getattr(sym_info, "filling_mode", 0) or 0)
    if mask & 1:
        return fok
    if mask & 2:
        return ioc
    return ret


def resolve_pending_expiration_spec(
    mt5_api: Any,
    sym_info: Any,
    tick: Any,
    expiration_hours: float = 8.0,
) -> Tuple[int, int, str]:
    """
    Tentukan type_time & expiration epoch untuk pending order berdasarkan expiration_mode broker.
    Jika broker mengizinkan ORDER_TIME_SPECIFIED (bitmask 4), pasang waktu kedaluwarsa server.
    Jika hanya GTC -> kembalikan peringatan eksplisit bahwa order server tetap aktif saat Python disconnect.
    """
    gtc = getattr(mt5_api, "ORDER_TIME_GTC", 0)
    specified = getattr(mt5_api, "ORDER_TIME_SPECIFIED", 2)
    exp_mask = int(getattr(sym_info, "expiration_mode", 0) or 0) if sym_info is not None else 0
    tick_ts = int(getattr(tick, "time", 0) or 0) if tick is not None else int(time.time())
    if tick_ts <= 0:
        tick_ts = int(time.time())

    if (exp_mask == 0 or (exp_mask & 4)) and expiration_hours > 0:
        exp_epoch = tick_ts + int(round(float(expiration_hours) * 3600))
        note = (
            f"ORDER_TIME_SPECIFIED ({expiration_hours:g}h); PERINGATAN: pending order di server broker "
            f"tetap aktif hingga waktu expiration meskipun script Python terputus."
        )
        return specified, exp_epoch, note

    note = (
        "ORDER_TIME_GTC (broker tidak mendukung ORDER_TIME_SPECIFIED); "
        "PERINGATAN: pending order di server broker tetap aktif apabila Python/MT5 terputus!"
    )
    return gtc, 0, note


def normalize_price_to_tick(price: float, sym_info: Any, symbol: str = SYMBOL) -> float:
    spec = get_symbol_unit_spec(symbol=symbol, sym_info=sym_info)
    digits = int(spec["digits"])
    t_size = float(spec["trade_tick_size"] or spec["point"] or 0.01)
    if price <= 0 or not math.isfinite(price):
        return 0.0
    if t_size > 0:
        return round(round(price / t_size) * t_size, digits)
    return round(price, digits)


def reconcile_ambiguous_execution(
    mt5_api: Any,
    setup_id: str,
    comment: str,
    symbol: str,
    magic_number: int,
    candidate_ticket: int = 0,
) -> Dict[str, Any]:
    """
    Rekonsiliasi penuh terhadap positions_get, orders_get, dan history_deals_get ketika
    order_send mengalami timeout, koneksi terputus, atau retcode ambigu.
    Mencegah blind retry yang memicu duplicate order!
    """
    if mt5_api is None:
        return {"found": False, "kind": "NONE"}

    with MT5_IPC_LOCK:
        positions = list(mt5_api.positions_get(symbol=symbol) or mt5_api.positions_get() or ())
        orders = list(mt5_api.orders_get(symbol=symbol) or mt5_api.orders_get() or ())

    for p in positions:
        p_tkt = int(getattr(p, "ticket", 0) or 0)
        p_cmt = str(getattr(p, "comment", "") or "")
        if not is_scanner_owned_item(p, magic_number):
            continue
        if (candidate_ticket > 0 and p_tkt == candidate_ticket) or (comment and p_cmt == comment) or (setup_id[-12:] in p_cmt):
            return {
                "found": True,
                "kind": "POSITION",
                "ticket": p_tkt,
                "price": float(getattr(p, "price_open", 0.0) or 0.0),
                "volume": float(getattr(p, "volume", 0.0) or 0.0),
                "sl": float(getattr(p, "sl", 0.0) or 0.0),
                "tp": float(getattr(p, "tp", 0.0) or 0.0),
                "raw": p,
            }

    for o in orders:
        o_tkt = int(getattr(o, "ticket", 0) or 0)
        o_cmt = str(getattr(o, "comment", "") or "")
        if not is_scanner_owned_item(o, magic_number):
            continue
        if (candidate_ticket > 0 and o_tkt == candidate_ticket) or (comment and o_cmt == comment) or (setup_id[-12:] in o_cmt):
            return {
                "found": True,
                "kind": "ORDER",
                "ticket": o_tkt,
                "price": float(getattr(o, "price_open", 0.0) or 0.0),
                "volume": float(getattr(o, "volume_current", getattr(o, "volume_initial", 0.0)) or 0.0),
                "sl": float(getattr(o, "sl", 0.0) or 0.0),
                "tp": float(getattr(o, "tp", 0.0) or 0.0),
                "raw": o,
            }

    # Cek history deals 15 menit terakhir bila posisi sudah langsung tereksekusi
    if hasattr(mt5_api, "history_deals_get"):
        try:
            now_u = datetime.now(timezone.utc)
            deals = mt5_api.history_deals_get(now_u - timedelta(minutes=30), now_u + timedelta(days=1)) or ()
            for d in reversed(list(deals)):
                d_cmt = str(getattr(d, "comment", "") or "")
                d_ord = int(getattr(d, "order", 0) or 0)
                if not is_scanner_owned_item(d, magic_number):
                    continue
                if (candidate_ticket > 0 and d_ord == candidate_ticket) or (comment and d_cmt == comment):
                    return {
                        "found": True,
                        "kind": "HISTORY_DEAL",
                        "ticket": int(getattr(d, "position_id", d_ord) or d_ord),
                        "order_ticket": d_ord,
                        "price": float(getattr(d, "price", 0.0) or 0.0),
                        "volume": float(getattr(d, "volume", 0.0) or 0.0),
                        "sl": 0.0,
                        "tp": 0.0,
                        "raw": d,
                    }
        except Exception:
            pass

    return {"found": False, "kind": "NONE"}


class MT5ExecutionEngine:
    """
    Engine eksekusi order & rekonsiliasi MT5 yang thread-safe dan mendukung injeksi `mt5_api`
    untuk pengujian mock maupun eksekusi langsung pada akun Real IDC / Demo.
    """

    def __init__(
        self,
        mt5_api: Any = None,
        state_mgr: Optional[ExecutionStateManager] = None,
        journal: Optional[Any] = None,
        cfg: Optional[Dict[str, Any]] = None,
    ):
        if mt5_api is None:
            import MetaTrader5 as _mt5_mod
            self.mt5 = _mt5_mod
        else:
            self.mt5 = mt5_api
        self.state_mgr = state_mgr
        self.journal = journal
        self.cfg = cfg or get_auto_trading_config()

    def execute_validated_setup(
        self,
        validation_result: Dict[str, Any],
        trading_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Eksekusi setup yang sudah mencapai status ENTRY_READY sesuai mode trading:
        - SCANNER_ONLY : Tidak mengirim order.
        - DRY_RUN      : Simulasi menggunakan harga live aktual tanpa mengirim order ke broker.
        - SEMI_AUTO    : Antrekan untuk persetujuan eksplisit user (kecuali user_approved=True).
        - FULL_AUTO    : Kirim ke broker MT5 setelah pre-send revalidation & order_check().
        """
        t0 = time.perf_counter()
        conf = self.cfg
        sm = self.state_mgr
        mode_u = str(
            trading_mode
            or (sm.get_mode_info().get("trading_mode") if sm else None)
            or validation_result.get("trading_mode")
            or conf.get("TRADING_MODE", "SCANNER_ONLY")
        ).upper()

        setup_id = str(validation_result.get("setup_id") or "")
        zone_id = str(validation_result.get("zone_id") or "")
        plan = validation_result.get("trade_plan") or {}
        symbol = str(validation_result.get("symbol") or plan.get("symbol") or conf.get("SYMBOL") or SYMBOL)
        direction = str(plan.get("direction") or "").upper()
        entry_mode = str(validation_result.get("entry_mode") or plan.get("entry_mode") or "ADAPTIVE_ENTRY").upper()
        order_type_str = str(plan.get("order_type") or f"{direction} MARKET").upper()
        is_market = ("LIMIT" not in order_type_str and "STOP" not in order_type_str)

        if not validation_result.get("can_execute") or validation_result.get("state") != "ENTRY_READY":
            return {
                "executed": False,
                "simulated": False,
                "state": validation_result.get("state", STATE_BLOCKED),
                "reason_code": validation_result.get("reason_code", REASON_EXECUTION_REJECTED),
                "reason_text": validation_result.get("reason_text", "Setup belum ENTRY_READY"),
                "setup_id": setup_id,
            }

        if mode_u == "SCANNER_ONLY":
            return {
                "executed": False,
                "simulated": False,
                "state": "ENTRY_READY",
                "execution_status": "SCANNER_ONLY_NO_ORDER",
                "reason_code": "SCANNER_ONLY_MODE",
                "reason_text": "Mode SCANNER_ONLY aktif — order tidak dikirim",
                "setup_id": setup_id,
            }

        if mode_u == "SEMI_AUTO" and not validation_result.get("user_approved"):
            if sm is not None:
                sm.queue_semi_auto_setup(setup_id, validation_result)
                sm.update_setup_state(
                    setup_id, "ENTRY_READY",
                    reason_code="WAIT_USER_APPROVAL",
                    reason_text="Menunggu persetujuan user ('auto approve')",
                    zone_id=zone_id, symbol=symbol, direction=direction, entry_mode=entry_mode,
                    planned_entry=float(plan.get("entry") or 0.0),
                    sl=float(plan.get("sl") or 0.0),
                    tp=float(plan.get("tp") or 0.0),
                    target_lot=float(validation_result.get("target_lot") or 0.0),
                    final_lot=float(validation_result.get("final_lot") or 0.0),
                )
            return {
                "executed": False,
                "simulated": False,
                "queued_for_approval": True,
                "state": "ENTRY_READY",
                "execution_status": "AWAITING_USER_APPROVAL",
                "reason_code": "WAIT_USER_APPROVAL",
                "reason_text": f"Setup {setup_id} siap; ketik 'auto approve' untuk mengeksekusi",
                "setup_id": setup_id,
            }

        # Atomic lock per setup_id & global execution lock (Scenario 34: dua loop paralel -> hanya 1 eksekusi)
        with _EXEC_SEND_LOCK:
            if sm is not None:
                acquired, lock_code = sm.try_acquire_setup_execution(setup_id)
                if not acquired:
                    return {
                        "executed": False,
                        "simulated": False,
                        "state": STATE_BLOCKED,
                        "execution_status": STATE_BLOCKED,
                        "reason_code": lock_code,
                        "reason_text": f"Setup {setup_id} sudah sedang/telah dieksekusi ({lock_code})",
                        "setup_id": setup_id,
                    }

            try:
                return self._execute_locked(
                    validation_result=validation_result,
                    mode_u=mode_u,
                    setup_id=setup_id,
                    zone_id=zone_id,
                    symbol=symbol,
                    direction=direction,
                    entry_mode=entry_mode,
                    is_market=is_market,
                    t_start=t0,
                )
            finally:
                if sm is not None:
                    sm.release_setup_execution(setup_id)

    def _execute_locked(
        self,
        validation_result: Dict[str, Any],
        mode_u: str,
        setup_id: str,
        zone_id: str,
        symbol: str,
        direction: str,
        entry_mode: str,
        is_market: bool,
        t_start: float,
    ) -> Dict[str, Any]:
        conf = self.cfg
        sm = self.state_mgr
        plan = validation_result.get("trade_plan") or {}
        zone_dict = validation_result.get("zone") or {}
        lot = round(float(validation_result.get("final_lot") or 0.0), 2)
        target_lot = round(float(validation_result.get("target_lot") or 0.0), 2)
        magic = int(conf.get("SCANNER_MAGIC_NUMBER", 20260991))

        # ---------------------------------------------------------------------
        # PRE-SEND LIVE REVALIDATION (Harga live, Spread, RR, Volume, Margin, Akun)
        # ---------------------------------------------------------------------
        acc_check = inspect_mt5_account_and_permissions(
            mt5_api=self.mt5,
            symbol=symbol,
            state_mgr=sm,
            require_exec_permission=(mode_u in ("FULL_AUTO", "SEMI_AUTO")),
            trading_mode=mode_u,
            cfg=conf,
        )
        if not acc_check.get("connected") or (mode_u != "DRY_RUN" and not acc_check.get("authorized")):
            r_code = str(acc_check.get("reason_code") or REASON_MT5_DISCONNECTED)
            r_txt = str(acc_check.get("reason_text") or "Koneksi/izin MT5 tidak valid sebelum order_send")
            if sm is not None:
                sm.update_setup_state(setup_id, STATE_BLOCKED, reason_code=r_code, reason_text=r_txt, zone_id=zone_id)
            return {
                "executed": False, "simulated": False, "state": STATE_BLOCKED,
                "execution_status": STATE_BLOCKED, "reason_code": r_code, "reason_text": r_txt, "setup_id": setup_id,
            }

        sym_info = acc_check.get("symbol_info")
        with MT5_IPC_LOCK:
            fresh_tick = self.mt5.symbol_info_tick(symbol)
        if fresh_tick is None:
            if sm is not None:
                sm.update_setup_state(setup_id, STATE_BLOCKED, reason_code=REASON_MT5_DISCONNECTED, reason_text="Tick live tidak tersedia", zone_id=zone_id)
            return {
                "executed": False, "simulated": False, "state": STATE_BLOCKED,
                "execution_status": STATE_BLOCKED, "reason_code": REASON_MT5_DISCONNECTED,
                "reason_text": "Tick live tidak tersedia saat pre-send check", "setup_id": setup_id,
            }

        bid = float(getattr(fresh_tick, "bid", 0.0) or 0.0)
        ask = float(getattr(fresh_tick, "ask", 0.0) or 0.0)
        if bid <= 0 or ask < bid:
            if sm is not None:
                sm.update_setup_state(setup_id, STATE_BLOCKED, reason_code=REASON_MT5_DISCONNECTED, reason_text="Harga Bid/Ask tidak valid", zone_id=zone_id)
            return {
                "executed": False, "simulated": False, "state": STATE_BLOCKED,
                "execution_status": STATE_BLOCKED, "reason_code": REASON_MT5_DISCONNECTED,
                "reason_text": "Harga Bid/Ask tidak valid", "setup_id": setup_id,
            }

        live_spread = round(ask - bid, 4)
        max_spread = float(conf.get("SPREAD_MAKS_POIN", 1.0))
        if live_spread > max_spread + 1e-9:
            msg_sp = f"Spread melonjak sebelum order_send ({live_spread:.2f} > {max_spread:.2f} USD)"
            if sm is not None:
                sm.update_setup_state(setup_id, STATE_BLOCKED, reason_code=REASON_SPREAD_TOO_HIGH, reason_text=msg_sp, zone_id=zone_id)
            self._log_journal_event("BLOCKED", validation_result, mode_u, acc_check, spread=live_spread, reason=REASON_SPREAD_TOO_HIGH)
            return {
                "executed": False, "simulated": False, "state": STATE_BLOCKED,
                "execution_status": STATE_BLOCKED, "reason_code": REASON_SPREAD_TOO_HIGH,
                "reason_text": msg_sp, "setup_id": setup_id,
            }

        # Gunakan Ask untuk BUY MARKET dan Bid untuk SELL MARKET
        if is_market:
            exec_price = ask if direction == "BUY" else bid
            # Anti-chase check tepat sebelum kirim order
            z_lo = float(zone_dict.get("price_low") or plan.get("zone_low") or 0.0)
            z_hi = float(zone_dict.get("price_high") or plan.get("zone_high") or 0.0)
            max_chase = float((validation_result.get("touch") or {}).get("max_chase_distance") or conf.get("EXECUTION_CHASE_MAX_POINTS", 1.5))
            if not conf.get("CHASE_PRICE", False) and z_lo > 0 and z_hi >= z_lo:
                if (direction == "BUY" and exec_price > z_hi + max_chase) or (direction == "SELL" and exec_price < z_lo - max_chase):
                    msg_ch = f"Harga live ({exec_price:.2f}) telah meninggalkan execution range zona ({z_lo:.2f}-{z_hi:.2f})"
                    if sm is not None:
                        sm.update_setup_state(setup_id, STATE_BLOCKED, reason_code=REASON_PRICE_LEFT_ENTRY_RANGE, reason_text=msg_ch, zone_id=zone_id)
                    self._log_journal_event("MISSED_ENTRY", validation_result, mode_u, acc_check, spread=live_spread, reason=REASON_PRICE_LEFT_ENTRY_RANGE)
                    return {
                        "executed": False, "simulated": False, "state": STATE_BLOCKED,
                        "execution_status": STATE_BLOCKED, "reason_code": REASON_PRICE_LEFT_ENTRY_RANGE,
                        "reason_text": msg_ch, "setup_id": setup_id,
                    }
        else:
            exec_price = normalize_price_to_tick(float(plan.get("entry") or 0.0), sym_info, symbol=symbol)
            # Validasi mekanisme harga limit terhadap pasar saat ini (BUY LIMIT < Ask, SELL LIMIT > Bid)
            if (direction == "BUY" and exec_price >= ask) or (direction == "SELL" and exec_price <= bid):
                msg_lim = f"Harga {direction} LIMIT ({exec_price:.2f}) sudah tersentuh/melewati harga pasar (Bid={bid:.2f}, Ask={ask:.2f})"
                if sm is not None:
                    sm.update_setup_state(setup_id, STATE_BLOCKED, reason_code=REASON_BROKER_CONSTRAINTS, reason_text=msg_lim, zone_id=zone_id)
                return {
                    "executed": False, "simulated": False, "state": STATE_BLOCKED,
                    "execution_status": STATE_BLOCKED, "reason_code": REASON_BROKER_CONSTRAINTS,
                    "reason_text": msg_lim, "setup_id": setup_id,
                }

        sl_px = normalize_price_to_tick(float(plan.get("sl") or 0.0), sym_info, symbol=symbol)
        tp_px = normalize_price_to_tick(float(plan.get("tp") or plan.get("tp1") or 0.0), sym_info, symbol=symbol)
        inval_px = float(plan.get("invalidation_price") or zone_dict.get("invalidation_price") or 0.0)

        # Revalidasi status zona & structural invalidation tepat sebelum order_send (Section 6)
        z_st_pre = str(zone_dict.get("status") or "ACTIVE").upper()
        z_fr_pre = str(zone_dict.get("freshness") or "FRESH").upper()
        if z_st_pre in ("INVALIDATED", "EXPIRED", "MITIGATED", "BROKEN", "REJECTED") or z_fr_pre in ("INVALIDATED", "EXPIRED", "MITIGATED"):
            r_z_code = REASON_ZONE_EXPIRED if "EXPIRED" in (z_st_pre, z_fr_pre) else REASON_ZONE_INVALIDATED
            msg_zi = f"Revalidasi pre-send: zona {zone_id} berstatus {z_st_pre}/{z_fr_pre}"
            if sm is not None:
                sm.update_setup_state(setup_id, STATE_BLOCKED, reason_code=r_z_code, reason_text=msg_zi, zone_id=zone_id)
            return {
                "executed": False, "simulated": False, "state": STATE_BLOCKED,
                "execution_status": STATE_BLOCKED, "reason_code": r_z_code,
                "reason_text": msg_zi, "setup_id": setup_id,
            }
        probe_px = float(bid if direction == "BUY" else ask)
        if inval_px > 0 and ((direction == "BUY" and probe_px < inval_px) or (direction == "SELL" and probe_px > inval_px)):
            msg_inv_pre = f"Revalidasi pre-send: harga live ({probe_px:.2f}) menembus structural invalidation ({inval_px:.2f})"
            if sm is not None:
                sm.update_setup_state(setup_id, STATE_BLOCKED, reason_code=REASON_ZONE_INVALIDATED, reason_text=msg_inv_pre, zone_id=zone_id)
            return {
                "executed": False, "simulated": False, "state": STATE_BLOCKED,
                "execution_status": STATE_BLOCKED, "reason_code": REASON_ZONE_INVALIDATED,
                "reason_text": msg_inv_pre, "setup_id": setup_id,
            }

        sl_tp_recheck = validate_sl_tp_and_rr(
            direction=direction,
            entry_price=exec_price,
            sl_price=sl_px,
            tp1_price=tp_px,
            tp2_price=float(plan.get("tp2") or 0.0),
            structural_invalidation=inval_px,
            sym_info=sym_info,
            bid=bid,
            ask=ask,
            is_market_order=is_market,
            symbol=symbol,
            cfg=conf,
        )
        if not sl_tp_recheck.get("valid"):
            rc_sl = str(sl_tp_recheck.get("reason_code"))
            rt_sl = str(sl_tp_recheck.get("reason_text"))
            if sm is not None:
                sm.update_setup_state(setup_id, STATE_BLOCKED, reason_code=rc_sl, reason_text=rt_sl, zone_id=zone_id)
            self._log_journal_event("BLOCKED", validation_result, mode_u, acc_check, spread=live_spread, reason=rc_sl)
            return {
                "executed": False, "simulated": False, "state": STATE_BLOCKED,
                "execution_status": STATE_BLOCKED, "reason_code": rc_sl, "reason_text": rt_sl, "setup_id": setup_id,
            }

        vol_ok, vol_msg = is_broker_volume_valid(lot, sym_info)
        if not vol_ok:
            if sm is not None:
                sm.update_setup_state(setup_id, STATE_BLOCKED, reason_code=REASON_INVALID_VOLUME, reason_text=vol_msg, zone_id=zone_id)
            return {
                "executed": False, "simulated": False, "state": STATE_BLOCKED,
                "execution_status": STATE_BLOCKED, "reason_code": REASON_INVALID_VOLUME,
                "reason_text": vol_msg, "setup_id": setup_id,
            }

        # Cek duplikat order/posisi scanner berdasarkan Zone ID, Magic Number, dan Ticket MT5 tepat sebelum kirim
        with MT5_IPC_LOCK:
            cur_pos = list(self.mt5.positions_get(symbol=symbol) or ())
            cur_ord = list(self.mt5.orders_get(symbol=symbol) or ())
        comment = encode_order_comment(setup_id, entry_mode)
        if sm is not None and zone_id:
            for s_rec in sm.get_all_setups().values():
                if str(s_rec.get("zone_id") or "") == zone_id and s_rec.get("state") in (STATE_ORDER_PLACED, STATE_POSITION_ACTIVE):
                    if sm is not None:
                        sm.update_setup_state(setup_id, STATE_BLOCKED, reason_code=REASON_DUPLICATE_SETUP, reason_text=f"Zone ID {zone_id} sudah memiliki order/posisi aktif", zone_id=zone_id)
                    return {
                        "executed": False, "simulated": False, "state": STATE_BLOCKED,
                        "execution_status": STATE_BLOCKED, "reason_code": REASON_DUPLICATE_SETUP,
                        "reason_text": f"Zone ID {zone_id} sudah memiliki order/posisi aktif", "setup_id": setup_id,
                    }
        for p in cur_pos:
            if is_scanner_owned_item(p, magic) and (
                str(getattr(p, "comment", "")) == comment
                or len([x for x in cur_pos if is_scanner_owned_item(x, magic)]) >= int(conf.get("MAX_ACTIVE_POSITIONS", 1))
            ):
                if sm is not None:
                    sm.update_setup_state(setup_id, STATE_BLOCKED, reason_code=REASON_DUPLICATE_SETUP, reason_text="Posisi scanner sudah aktif di MT5", zone_id=zone_id)
                return {
                    "executed": False, "simulated": False, "state": STATE_BLOCKED,
                    "execution_status": STATE_BLOCKED, "reason_code": REASON_DUPLICATE_SETUP,
                    "reason_text": "Posisi scanner sudah aktif di MT5", "setup_id": setup_id,
                }
        if not is_market:
            scanner_ords = [o for o in cur_ord if is_scanner_owned_item(o, magic)]
            if len(scanner_ords) >= int(conf.get("MAX_ACTIVE_PENDING_ORDERS", 1)):
                if sm is not None:
                    sm.update_setup_state(setup_id, STATE_BLOCKED, reason_code=REASON_DUPLICATE_SETUP, reason_text="Pending order scanner sudah aktif di MT5", zone_id=zone_id)
                return {
                    "executed": False, "simulated": False, "state": STATE_BLOCKED,
                    "execution_status": STATE_BLOCKED, "reason_code": REASON_DUPLICATE_SETUP,
                    "reason_text": "Pending order scanner sudah aktif di MT5", "setup_id": setup_id,
                }

        # ---------------------------------------------------------------------
        # MODE DRY_RUN: Simulasi penuh dengan harga aktual TANPA mengirim order ke MT5
        # ---------------------------------------------------------------------
        if mode_u == "DRY_RUN":
            latency_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
            sim_state = STATE_POSITION_ACTIVE if is_market else STATE_ORDER_PLACED
            self._log_journal_event(
                "DRY_RUN_SIMULATION", validation_result, mode_u, acc_check,
                actual_entry=exec_price, actual_lot=lot, actual_rr=sl_tp_recheck["actual_rr"],
                spread=live_spread, retcode=0, reason="DRY_RUN_NO_BROKER_ORDER", latency_ms=latency_ms,
            )
            return {
                "executed": False,
                "simulated": True,
                "state": "ENTRY_READY",
                "execution_status": f"DRY_RUN_SIMULATED ({sim_state})",
                "reason_code": REASON_OK,
                "reason_text": f"DRY_RUN: Simulasi {direction} {lot:.2f} lot @ {exec_price:.2f} (SL {sl_px:.2f}, TP {tp_px:.2f}) — TIDAK dikirim ke MT5",
                "setup_id": setup_id,
                "planned_entry": float(plan.get("entry") or exec_price),
                "simulated_entry": exec_price,
                "simulated_lot": lot,
                "sl": sl_px,
                "tp": tp_px,
                "actual_rr": sl_tp_recheck["actual_rr"],
                "latency_ms": latency_ms,
            }

        # ---------------------------------------------------------------------
        # SUSUN REQUEST MT5 -> ORDER_CHECK() -> ORDER_SEND()
        # ---------------------------------------------------------------------
        filling_type = resolve_broker_filling_mode(self.mt5, sym_info)
        deviation = int(conf.get("ORDER_DEVIATION_POINTS", 30))

        if is_market:
            action = getattr(self.mt5, "TRADE_ACTION_DEAL", 1)
            ord_type = getattr(self.mt5, "ORDER_TYPE_BUY", 0) if direction == "BUY" else getattr(self.mt5, "ORDER_TYPE_SELL", 1)
            type_time = getattr(self.mt5, "ORDER_TIME_GTC", 0)
            exp_epoch = 0
            exp_warning = ""
        else:
            action = getattr(self.mt5, "TRADE_ACTION_PENDING", 5)
            ord_type = getattr(self.mt5, "ORDER_TYPE_BUY_LIMIT", 2) if direction == "BUY" else getattr(self.mt5, "ORDER_TYPE_SELL_LIMIT", 3)
            type_time, exp_epoch, exp_warning = resolve_pending_expiration_spec(
                self.mt5, sym_info, fresh_tick, expiration_hours=float(conf.get("DEFAULT_PENDING_EXPIRATION_HOURS", 8))
            )

        request: Dict[str, Any] = {
            "action": action,
            "symbol": symbol,
            "volume": float(lot),
            "type": ord_type,
            "price": float(exec_price),
            "sl": float(sl_px),
            "tp": float(tp_px),
            "deviation": deviation,
            "magic": magic,
            "comment": comment,
            "type_time": type_time,
            "type_filling": filling_type,
        }
        if exp_epoch > 0:
            request["expiration"] = int(exp_epoch)

        with MT5_IPC_LOCK:
            check_res = self.mt5.order_check(request)

        if check_res is None:
            msg_chk = "mt5.order_check() mengembalikan None"
            if sm is not None:
                sm.update_setup_state(setup_id, STATE_BLOCKED, reason_code=REASON_EXECUTION_REJECTED, reason_text=msg_chk, zone_id=zone_id)
            self._log_journal_event("REJECTED", validation_result, mode_u, acc_check, spread=live_spread, retcode=-1, reason=msg_chk)
            return {
                "executed": False, "simulated": False, "state": STATE_BLOCKED,
                "execution_status": STATE_BLOCKED, "reason_code": REASON_EXECUTION_REJECTED,
                "reason_text": msg_chk, "setup_id": setup_id,
            }

        chk_rc = _safe_retcode(check_res)
        if chk_rc not in (0, RETCODE_DONE, RETCODE_PLACED):
            mapped_chk = map_mt5_retcode_to_reason(chk_rc)
            chk_cmt = str(getattr(check_res, "comment", "") or f"retcode={chk_rc}")
            msg_fail_chk = f"order_check ditolak broker ({chk_rc}: {chk_cmt})"
            if sm is not None:
                sm.update_setup_state(setup_id, STATE_BLOCKED, reason_code=mapped_chk, reason_text=msg_fail_chk, zone_id=zone_id)
            self._log_journal_event("REJECTED", validation_result, mode_u, acc_check, spread=live_spread, retcode=chk_rc, reason=msg_fail_chk)
            return {
                "executed": False, "simulated": False, "state": STATE_BLOCKED,
                "execution_status": STATE_BLOCKED, "reason_code": mapped_chk,
                "reason_text": msg_fail_chk, "retcode": chk_rc, "setup_id": setup_id,
            }

        # Kirim order ke broker MT5 dengan bounded retry aman pada requote / price changed (tanpa duplicate order!)
        max_retries = max(0, int(conf.get("ORDER_SEND_MAX_RETRIES", 2)))
        transient_price_rcs = {RETCODE_REQUOTE, RETCODE_PRICE_CHANGED, RETCODE_PRICE_OFF}
        send_res = None
        send_exc = None
        send_rc = -1
        attempts = 0

        for attempt_idx in range(max_retries + 1):
            attempts = attempt_idx + 1
            if attempt_idx > 0:
                # Sebelum retry: rekonsiliasi dulu untuk memastikan attempt sebelumnya tidak diam-diam masuk!
                pre_retry_recon = reconcile_ambiguous_execution(
                    self.mt5, setup_id=setup_id, comment=comment, symbol=symbol, magic_number=magic,
                )
                if pre_retry_recon.get("found"):
                    send_res = SimpleNamespace(
                        retcode=RETCODE_DONE,
                        order=pre_retry_recon.get("ticket", 0),
                        deal=0,
                        price=pre_retry_recon.get("price", exec_price),
                        volume=pre_retry_recon.get("volume", lot),
                        comment="Reconciled before retry",
                    )
                    send_rc = RETCODE_DONE
                    break

                # Refresh harga live & pastikan masih masuk batas spread & anti-chase sebelum retry
                with MT5_IPC_LOCK:
                    retry_tick = self.mt5.symbol_info_tick(symbol)
                if retry_tick is None:
                    break
                r_bid = float(getattr(retry_tick, "bid", 0.0) or 0.0)
                r_ask = float(getattr(retry_tick, "ask", 0.0) or 0.0)
                if r_bid <= 0 or r_ask < r_bid or (r_ask - r_bid) > max_spread + 1e-9:
                    break
                if is_market:
                    exec_price = r_ask if direction == "BUY" else r_bid
                    z_lo_r = float(zone_dict.get("price_low") or plan.get("zone_low") or 0.0)
                    z_hi_r = float(zone_dict.get("price_high") or plan.get("zone_high") or 0.0)
                    max_ch_r = float((validation_result.get("touch") or {}).get("max_chase_distance") or conf.get("EXECUTION_CHASE_MAX_POINTS", 1.5))
                    if not conf.get("CHASE_PRICE", False) and z_lo_r > 0 and z_hi_r >= z_lo_r:
                        if (direction == "BUY" and exec_price > z_hi_r + max_ch_r) or (direction == "SELL" and exec_price < z_lo_r - max_ch_r):
                            break
                    request["price"] = float(exec_price)

            send_res = None
            send_exc = None
            try:
                with MT5_IPC_LOCK:
                    send_res = self.mt5.order_send(request)
            except Exception as exc:
                send_exc = exc
                _catat_error("mt5_order_send", exc)
                break

            send_rc = _safe_retcode(send_res)
            if send_res is None or send_rc not in transient_price_rcs:
                break
            time.sleep(0.15)

        latency_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
        cand_order_tkt = int(getattr(send_res, "order", 0) or 0) if send_res is not None else 0

        # Jika send_res None, exception, atau TIMEOUT -> WAJIB REKONSILIASI SEBELUM MENGANGGAP GAGAL!
        if send_res is None or send_exc is not None or send_rc in (RETCODE_TIMEOUT, RETCODE_CONNECTION, RETCODE_ERROR, -1):
            recon = reconcile_ambiguous_execution(
                self.mt5, setup_id=setup_id, comment=comment, symbol=symbol,
                magic_number=magic, candidate_ticket=cand_order_tkt,
            )
            if recon.get("found"):
                r_kind = recon["kind"]
                r_tkt = int(recon["ticket"])
                r_px = float(recon.get("price") or exec_price)
                r_vol = float(recon.get("volume") or lot)
                r_sl = float(recon.get("sl") or sl_px)
                r_tp = float(recon.get("tp") or tp_px)
                final_st = STATE_ORDER_PLACED if r_kind == "ORDER" else STATE_POSITION_ACTIVE
                if sm is not None:
                    sm.update_setup_state(
                        setup_id, final_st,
                        reason_code="RECONCILED_AFTER_AMBIGUOUS_SEND",
                        reason_text=f"Order terverifikasi masuk di broker pasca-timeout ({r_kind} #{r_tkt})",
                        zone_id=zone_id, symbol=symbol, direction=direction, entry_mode=entry_mode,
                        order_ticket=r_tkt if r_kind == "ORDER" else cand_order_tkt,
                        position_ticket=r_tkt if r_kind != "ORDER" else None,
                        planned_entry=exec_price, actual_entry=r_px,
                        sl=r_sl, tp=r_tp, target_lot=target_lot, final_lot=lot, actual_lot=r_vol,
                        zone_score=float(zone_dict.get("score") or 0.0),
                        zone_grade=str(zone_dict.get("grade") or ""),
                        zone_quality=str(validation_result.get("zone_quality") or ""),
                        zone_timeframe=str(zone_dict.get("timeframe") or ""),
                        trigger_name=str((validation_result.get("triggers") or {}).get("primary_trigger") or ""),
                        bos_time=str(zone_dict.get("bos_candle_time") or ""),
                    )
                    if final_st == STATE_POSITION_ACTIVE:
                        sm.record_trade_opened(setup_id, r_tkt)
                self._log_journal_event(
                    "EXECUTED_RECONCILED", validation_result, mode_u, acc_check,
                    actual_entry=r_px, actual_lot=r_vol, actual_rr=sl_tp_recheck["actual_rr"],
                    spread=live_spread, retcode=send_rc, order_ticket=r_tkt, position_ticket=r_tkt if r_kind != "ORDER" else 0,
                    reason="RECONCILED_AFTER_TIMEOUT", latency_ms=latency_ms,
                )
                return {
                    "executed": True,
                    "reconciled": True,
                    "simulated": False,
                    "state": final_st,
                    "execution_status": final_st,
                    "reason_code": REASON_OK,
                    "reason_text": f"Rekonsiliasi mengonfirmasi {r_kind} #{r_tkt} aktif di broker",
                    "order_ticket": r_tkt,
                    "position_ticket": r_tkt if r_kind != "ORDER" else 0,
                    "actual_entry": r_px,
                    "actual_lot": r_vol,
                    "actual_sl": r_sl,
                    "actual_tp": r_tp,
                    "retcode": send_rc,
                    "latency_ms": latency_ms,
                    "setup_id": setup_id,
                }

            amb_msg = f"Hasil order_send ambigu/timeout (retcode={send_rc}) dan tidak ditemukan pada rekonsiliasi; tidak melakukan blind retry"
            if sm is not None:
                sm.update_setup_state(setup_id, STATE_BLOCKED, reason_code=REASON_EXECUTION_AMBIGUOUS, reason_text=amb_msg, zone_id=zone_id)
            self._log_journal_event("AMBIGUOUS_NOT_PLACED", validation_result, mode_u, acc_check, spread=live_spread, retcode=send_rc, reason=amb_msg, latency_ms=latency_ms)
            return {
                "executed": False,
                "reconciled": True,
                "simulated": False,
                "state": STATE_BLOCKED,
                "execution_status": STATE_BLOCKED,
                "reason_code": REASON_EXECUTION_AMBIGUOUS,
                "reason_text": amb_msg,
                "retcode": send_rc,
                "latency_ms": latency_ms,
                "setup_id": setup_id,
            }

        # Jika order ditolak eksplisit oleh broker (mis. REQUOTE, INVALID_STOPS, NO_MONEY, REJECT)
        if send_rc not in (RETCODE_DONE, RETCODE_PLACED, RETCODE_DONE_PARTIAL):
            mapped_rc = map_mt5_retcode_to_reason(send_rc)
            brk_cmt = str(getattr(send_res, "comment", "") or f"retcode={send_rc}")
            rej_msg = f"Order ditolak broker ({send_rc}: {brk_cmt})"
            if sm is not None:
                sm.update_setup_state(setup_id, STATE_BLOCKED, reason_code=mapped_rc, reason_text=rej_msg, zone_id=zone_id)
            self._log_journal_event("REJECTED", validation_result, mode_u, acc_check, spread=live_spread, retcode=send_rc, reason=rej_msg, latency_ms=latency_ms)
            return {
                "executed": False,
                "simulated": False,
                "state": STATE_BLOCKED,
                "execution_status": STATE_BLOCKED,
                "reason_code": mapped_rc,
                "reason_text": f"{rej_msg} [attempts={attempts}]",
                "retcode": send_rc,
                "attempts": attempts,
                "latency_ms": latency_ms,
                "setup_id": setup_id,
            }

        # ---------------------------------------------------------------------
        # ORDER DITERIMA BROKER (DONE / PLACED / DONE_PARTIAL) -> VERIFIKASI AKTUAL!
        # ---------------------------------------------------------------------
        order_ticket = int(getattr(send_res, "order", 0) or 0)
        deal_ticket = int(getattr(send_res, "deal", 0) or 0)
        res_price = float(getattr(send_res, "price", 0.0) or exec_price)
        res_vol = float(getattr(send_res, "volume", 0.0) or lot)

        actual_entry = res_price if res_price > 0 else exec_price
        actual_vol = res_vol if res_vol > 0 else lot
        actual_sl = sl_px
        actual_tp = tp_px
        position_ticket = order_ticket

        if is_market:
            # Verifikasi langsung dari positions_get untuk mendapatkan volume aktual (partial fill), SL, TP, & ticket posisi
            with MT5_IPC_LOCK:
                live_positions = list(self.mt5.positions_get(symbol=symbol) or ())
            matched_p = None
            for p in live_positions:
                if not is_scanner_owned_item(p, magic):
                    continue
                pt = int(getattr(p, "ticket", 0) or 0)
                if pt == order_ticket or str(getattr(p, "comment", "")) == comment:
                    matched_p = p
                    break
            if matched_p is not None:
                position_ticket = int(getattr(matched_p, "ticket", position_ticket) or position_ticket)
                actual_entry = float(getattr(matched_p, "price_open", actual_entry) or actual_entry)
                actual_vol = float(getattr(matched_p, "volume", actual_vol) or actual_vol)
                actual_sl = float(getattr(matched_p, "sl", 0.0) or 0.0)
                actual_tp = float(getattr(matched_p, "tp", 0.0) or 0.0)

            final_state = STATE_POSITION_ACTIVE
        else:
            with MT5_IPC_LOCK:
                live_orders = list(self.mt5.orders_get(symbol=symbol) or ())
            for o in live_orders:
                if int(getattr(o, "ticket", 0) or 0) == order_ticket:
                    actual_entry = float(getattr(o, "price_open", actual_entry) or actual_entry)
                    actual_vol = float(getattr(o, "volume_current", getattr(o, "volume_initial", actual_vol)) or actual_vol)
                    actual_sl = float(getattr(o, "sl", 0.0) or 0.0)
                    actual_tp = float(getattr(o, "tp", 0.0) or 0.0)
                    break
            final_state = STATE_ORDER_PLACED
            position_ticket = 0

        slippage_actual = round(abs(actual_entry - exec_price), 4)
        act_risk_dist = abs(actual_entry - actual_sl) if actual_sl > 0 else abs(actual_entry - sl_px)
        act_rew_dist = abs(actual_tp - actual_entry) if actual_tp > 0 else abs(tp_px - actual_entry)
        actual_rr = round(act_rew_dist / act_risk_dist, 2) if act_risk_dist > 0 else sl_tp_recheck["actual_rr"]
        is_partial = (send_rc == RETCODE_DONE_PARTIAL) or (actual_vol + 1e-6 < lot)

        if sm is not None:
            sm.update_setup_state(
                setup_id,
                final_state,
                reason_code="PARTIAL_FILL" if is_partial else REASON_OK,
                reason_text=exp_warning if not is_market else ("Partial fill terkonfirmasi" if is_partial else "Order terkonfirmasi broker"),
                zone_id=zone_id,
                symbol=symbol,
                direction=direction,
                entry_mode=entry_mode,
                order_ticket=order_ticket,
                position_ticket=position_ticket if is_market else None,
                planned_entry=exec_price,
                actual_entry=actual_entry,
                sl=actual_sl,
                tp=actual_tp,
                target_lot=target_lot,
                final_lot=lot,
                actual_lot=actual_vol,
                zone_score=float(zone_dict.get("score") or 0.0),
                zone_grade=str(zone_dict.get("grade") or ""),
                zone_quality=str(validation_result.get("zone_quality") or ""),
                zone_timeframe=str(zone_dict.get("timeframe") or ""),
                trigger_name=str((validation_result.get("triggers") or {}).get("primary_trigger") or ""),
                bos_time=str(zone_dict.get("bos_candle_time") or ""),
                extra={"planned_sl": float(sl_px), "planned_tp": float(tp_px)},
            )
            if is_market:
                sm.record_trade_opened(setup_id, position_ticket or order_ticket)

        self._log_journal_event(
            "ORDER_PLACED" if not is_market else ("PARTIAL_FILL" if is_partial else "POSITION_OPENED"),
            validation_result, mode_u, acc_check,
            actual_entry=actual_entry, actual_lot=actual_vol, actual_sl=actual_sl, actual_tp=actual_tp,
            actual_rr=actual_rr, spread=live_spread, slippage=slippage_actual,
            retcode=send_rc, order_ticket=order_ticket, position_ticket=position_ticket,
            reason="PARTIAL_FILL" if is_partial else REASON_OK, latency_ms=latency_ms,
        )

        return {
            "executed": True,
            "simulated": False,
            "partial_fill": is_partial,
            "state": final_state,
            "execution_status": final_state,
            "reason_code": REASON_OK,
            "reason_text": exp_warning if not is_market else ("Partial fill reconciled" if is_partial else "Broker confirmed execution"),
            "order_ticket": order_ticket,
            "deal_ticket": deal_ticket,
            "position_ticket": position_ticket,
            "planned_entry": exec_price,
            "actual_entry": actual_entry,
            "target_lot": target_lot,
            "final_lot": lot,
            "actual_lot": actual_vol,
            "actual_sl": actual_sl,
            "actual_tp": actual_tp,
            "actual_rr": actual_rr,
            "slippage": slippage_actual,
            "retcode": send_rc,
            "attempts": attempts,
            "latency_ms": latency_ms,
            "expiration_warning": exp_warning,
            "setup_id": setup_id,
        }

    # -------------------------------------------------------------------------
    # PENDING ORDER CANCELLATION, SL/TP MODIFICATION, & EMERGENCY CLOSE
    # -------------------------------------------------------------------------
    def cancel_scanner_pending_order(self, order_ticket: int, symbol: str = SYMBOL, reason: str = "") -> Dict[str, Any]:
        """
        Batalkan pending order MILIK SCANNER (memverifikasi magic number / comment).
        Order manual milik user TIDAK PERNAH dibatalkan!
        Jika order ternyata sudah terisi saat akan dibatalkan -> kembalikan already_filled=True.
        """
        magic = int(self.cfg.get("SCANNER_MAGIC_NUMBER", 20260991))
        tkt = int(order_ticket or 0)
        if tkt <= 0:
            return {"cancelled": False, "already_filled": False, "reason": "INVALID_TICKET"}

        with MT5_IPC_LOCK:
            orders = list(self.mt5.orders_get(ticket=tkt) or self.mt5.orders_get(symbol=symbol) or ())
        target_ord = next((o for o in orders if int(getattr(o, "ticket", 0) or 0) == tkt), None)

        if target_ord is None:
            # Periksa apakah order sudah berubah menjadi posisi aktif!
            with MT5_IPC_LOCK:
                positions = list(self.mt5.positions_get(symbol=symbol) or self.mt5.positions_get() or ())
            filled_pos = next((p for p in positions if int(getattr(p, "ticket", 0) or 0) == tkt and is_scanner_owned_item(p, magic)), None)
            if filled_pos is not None:
                return {
                    "cancelled": False,
                    "already_filled": True,
                    "position": filled_pos,
                    "position_ticket": tkt,
                    "reason": "ORDER_ALREADY_FILLED",
                }
            return {"cancelled": True, "already_filled": False, "reason": "ORDER_NOT_ACTIVE"}

        if not is_scanner_owned_item(target_ord, magic):
            return {"cancelled": False, "already_filled": False, "reason": "MANUAL_ORDER_PROTECTED"}

        req = {
            "action": getattr(self.mt5, "TRADE_ACTION_REMOVE", 8),
            "order": tkt,
            "symbol": str(getattr(target_ord, "symbol", symbol) or symbol),
            "magic": magic,
            "comment": f"CANCEL_{reason[:20]}" if reason else "CANCEL_SCANNER",
        }
        with MT5_IPC_LOCK:
            res = self.mt5.order_send(req)
        rc = _safe_retcode(res)

        # Verifikasi ulang pasca-remove apakah benar hilang atau justru terisi tepat bersamaan (Scenario 39!)
        with MT5_IPC_LOCK:
            rem_orders = list(self.mt5.orders_get(ticket=tkt) or ())
            positions_after = list(self.mt5.positions_get(symbol=symbol) or self.mt5.positions_get() or ())

        filled_pos_after = next(
            (p for p in positions_after if int(getattr(p, "ticket", 0) or 0) == tkt and is_scanner_owned_item(p, magic)),
            None,
        )
        if filled_pos_after is not None:
            return {
                "cancelled": False,
                "already_filled": True,
                "position": filled_pos_after,
                "position_ticket": tkt,
                "retcode": rc,
                "reason": "FILLED_DURING_CANCELLATION",
            }

        is_gone = len([o for o in rem_orders if int(getattr(o, "ticket", 0) or 0) == tkt]) == 0
        ok = (rc in (RETCODE_DONE, RETCODE_CANCEL, 0)) or is_gone
        return {
            "cancelled": ok,
            "already_filled": False,
            "retcode": rc,
            "reason": reason or ("CANCELLED" if ok else f"CANCEL_FAILED_{rc}"),
        }

    def modify_scanner_position_sl_tp(
        self,
        position_ticket: int,
        new_sl: float,
        new_tp: float,
        symbol: str = SYMBOL,
        allow_initial_sl_attach: bool = False,
    ) -> Dict[str, Any]:
        """
        Modifikasi SL/TP posisi MILIK SCANNER (memverifikasi magic number).
        DILARANG memperlebar SL yang sudah terpasang (`allow_initial_sl_attach` hanya untuk memasang SL darurat saat SL==0).
        Posisi manual milik user TIDAK PERNAH dimodifikasi!
        """
        magic = int(self.cfg.get("SCANNER_MAGIC_NUMBER", 20260991))
        tkt = int(position_ticket or 0)
        with MT5_IPC_LOCK:
            positions = list(self.mt5.positions_get(ticket=tkt) or self.mt5.positions_get(symbol=symbol) or ())
        pos = next((p for p in positions if int(getattr(p, "ticket", 0) or 0) == tkt), None)
        if pos is None:
            return {"modified": False, "reason": "POSITION_NOT_FOUND"}
        if not is_scanner_owned_item(pos, magic):
            return {"modified": False, "reason": "MANUAL_POSITION_PROTECTED"}

        pos_sym = str(getattr(pos, "symbol", symbol) or symbol)
        sym_info = self.mt5.symbol_info(pos_sym)
        is_buy = int(getattr(pos, "type", 0) or 0) == getattr(self.mt5, "ORDER_TYPE_BUY", 0)
        cur_sl = float(getattr(pos, "sl", 0.0) or 0.0)
        cur_tp = float(getattr(pos, "tp", 0.0) or 0.0)

        norm_sl = normalize_price_to_tick(float(new_sl if new_sl > 0 else cur_sl), sym_info, symbol=pos_sym)
        norm_tp = normalize_price_to_tick(float(new_tp if new_tp > 0 else cur_tp), sym_info, symbol=pos_sym)

        # Aturan Mutlak: DILARANG memperlebar SL!
        if cur_sl > 0 and norm_sl > 0:
            if (is_buy and norm_sl < cur_sl - 1e-9) or ((not is_buy) and norm_sl > cur_sl + 1e-9):
                return {"modified": False, "reason": "WIDENING_SL_FORBIDDEN"}

        req = {
            "action": getattr(self.mt5, "TRADE_ACTION_SLTP", 6),
            "position": tkt,
            "symbol": pos_sym,
            "sl": float(norm_sl),
            "tp": float(norm_tp),
            "magic": magic,
        }
        with MT5_IPC_LOCK:
            res = self.mt5.order_send(req)
        rc = _safe_retcode(res)
        ok = rc in (RETCODE_DONE, 0)
        return {
            "modified": ok,
            "retcode": rc,
            "sl": norm_sl,
            "tp": norm_tp,
            "reason": "MODIFIED" if ok else f"SLTP_MODIFY_FAILED_{rc}",
        }

    def close_scanner_position(self, position_ticket: int, symbol: str = SYMBOL, reason: str = "EMERGENCY") -> Dict[str, Any]:
        """
        Tutup posisi aktif MILIK SCANNER (hanya untuk Emergency Protection tanpa SL atau command eksplisit user).
        Posisi manual milik user TIDAK PERNAH ditutup!
        """
        magic = int(self.cfg.get("SCANNER_MAGIC_NUMBER", 20260991))
        tkt = int(position_ticket or 0)
        with MT5_IPC_LOCK:
            positions = list(self.mt5.positions_get(ticket=tkt) or self.mt5.positions_get(symbol=symbol) or ())
        pos = next((p for p in positions if int(getattr(p, "ticket", 0) or 0) == tkt), None)
        if pos is None:
            return {"closed": False, "reason": "POSITION_NOT_FOUND"}
        if not is_scanner_owned_item(pos, magic):
            return {"closed": False, "reason": "MANUAL_POSITION_PROTECTED"}

        pos_sym = str(getattr(pos, "symbol", symbol) or symbol)
        sym_info = self.mt5.symbol_info(pos_sym)
        tick = self.mt5.symbol_info_tick(pos_sym)
        if tick is None:
            return {"closed": False, "reason": "MISSING_TICK"}

        is_buy = int(getattr(pos, "type", 0) or 0) == getattr(self.mt5, "ORDER_TYPE_BUY", 0)
        close_type = getattr(self.mt5, "ORDER_TYPE_SELL", 1) if is_buy else getattr(self.mt5, "ORDER_TYPE_BUY", 0)
        close_px = float(getattr(tick, "bid", 0.0) if is_buy else getattr(tick, "ask", 0.0))
        vol = float(getattr(pos, "volume", 0.0) or 0.0)

        req = {
            "action": getattr(self.mt5, "TRADE_ACTION_DEAL", 1),
            "position": tkt,
            "symbol": pos_sym,
            "volume": vol,
            "type": close_type,
            "price": close_px,
            "deviation": int(self.cfg.get("ORDER_DEVIATION_POINTS", 30)),
            "magic": magic,
            "comment": f"CLS_{reason[:22]}",
            "type_time": getattr(self.mt5, "ORDER_TIME_GTC", 0),
            "type_filling": resolve_broker_filling_mode(self.mt5, sym_info),
        }
        with MT5_IPC_LOCK:
            res = self.mt5.order_send(req)
        rc = _safe_retcode(res)
        ok = rc in (RETCODE_DONE, RETCODE_DONE_PARTIAL, 0)
        return {
            "closed": ok,
            "retcode": rc,
            "close_price": close_px,
            "volume": vol,
            "reason": reason if ok else f"CLOSE_FAILED_{rc}",
        }

    def _log_journal_event(
        self,
        event_type: str,
        validation_result: Dict[str, Any],
        trading_mode: str,
        acc_check: Dict[str, Any],
        actual_entry: float = 0.0,
        actual_lot: float = 0.0,
        actual_sl: float = 0.0,
        actual_tp: float = 0.0,
        actual_rr: float = 0.0,
        spread: float = 0.0,
        slippage: float = 0.0,
        retcode: int = 0,
        order_ticket: int = 0,
        position_ticket: int = 0,
        reason: str = "",
        latency_ms: float = 0.0,
    ) -> None:
        if self.journal is None or not hasattr(self.journal, "record_execution_event"):
            return
        try:
            self.journal.record_execution_event(
                event_type=event_type,
                validation_result=validation_result,
                trading_mode=trading_mode,
                account_info=acc_check,
                actual_entry=actual_entry,
                actual_lot=actual_lot,
                actual_sl=actual_sl,
                actual_tp=actual_tp,
                actual_rr=actual_rr,
                spread=spread,
                slippage=slippage,
                retcode=retcode,
                order_ticket=order_ticket,
                position_ticket=position_ticket,
                decision_reason=reason,
                latency_ms=latency_ms,
            )
        except Exception as exc:
            _catat_error("execution_engine_journal", exc)
