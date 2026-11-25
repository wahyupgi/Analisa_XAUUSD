# -*- coding: utf-8 -*-
"""
modules/execution_monitor.py
Real-Time Execution Monitoring Engine, Mode Pantau Formatter, & CLI 'auto' Command Suite:
- Background daemon monitoring setiap 1 detik (configurable) yang berjalan independen dari
  rendering CLI dan analisis Gemini AI (Section 13).
- Rekonsiliasi otomatis saat startup & setelah reconnect MT5 (Section 20).
- Formatter panel eksekusi ringkas & faktual untuk Mode Pantau (Section 18).
- Handler lengkap seluruh CLI command 'auto ...' (Section 17).
"""
from __future__ import annotations

import math
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional

from modules.config import SYMBOL, get_auto_trading_config
from modules.execution_engine import MT5ExecutionEngine, is_scanner_owned_item
from modules.execution_journal import ExecutionJournal, get_execution_journal
from modules.execution_state import (
    ExecutionStateManager,
    REASON_MARKET_CLOSED,
    REASON_MT5_DISCONNECTED,
    REASON_PRICE_LEFT_ENTRY_RANGE,
    STATE_BLOCKED,
    STATE_COMPLETED,
    STATE_ENTRY_READY,
    STATE_ORDER_PLACED,
    STATE_POSITION_ACTIVE,
    get_execution_state_manager,
)
from modules.mt5_conn import MT5_IPC_LOCK, pastikan_mt5_aktif
from modules.order_manager import PendingOrderManager
from modules.position_manager import PositionManager
from modules.risk_manager import (
    compute_effective_risk_budget,
    evaluate_portfolio_risk_and_daily_limits,
    inspect_mt5_account_and_permissions,
)
from modules.trade_validator import select_candidate_zones, validate_trade_candidate
from modules.utils import _catat_error, format_wib_timestamp
from modules.zone_scanner import create_market_snapshot, evaluate_trade_decision, scan_xauusd_zones


class AutoExecutionMonitor:
    """
    Real-Time Monitoring & Auto-Execution Engine:
    - Update tick setiap MONITOR_INTERVAL_SECONDS (default 1s) tanpa membebani CPU.
    - Refresh struktur market penuh saat candle M5 baru terbentuk atau interval berkala.
    - Mengeksekusi setup segera ketika ENTRY_READY tercapai dalam mode yang diotorisasi.
    - Berhenti otomatis mengirim order baru saat MT5 terputus / market closed, lalu melakukan
      rekonsiliasi penuh saat koneksi pulih.
    """

    def __init__(
        self,
        mt5_api: Any = None,
        state_mgr: Optional[ExecutionStateManager] = None,
        journal: Optional[ExecutionJournal] = None,
        symbol: str = SYMBOL,
        cfg: Optional[Dict[str, Any]] = None,
        connect_fn: Optional[Callable[[], bool]] = None,
    ):
        if mt5_api is None:
            import MetaTrader5 as _mt5_mod
            self.mt5 = _mt5_mod
        else:
            self.mt5 = mt5_api
        self._custom_cfg_provided: bool = cfg is not None
        self.cfg = dict(cfg) if cfg is not None else get_auto_trading_config()
        self.symbol = str(symbol or self.cfg.get("SYMBOL") or SYMBOL)
        self.state_mgr = state_mgr or get_execution_state_manager()
        self.journal = journal or get_execution_journal()
        self.engine = MT5ExecutionEngine(
            mt5_api=self.mt5, state_mgr=self.state_mgr, journal=self.journal, cfg=self.cfg
        )
        self.order_mgr = PendingOrderManager(
            exec_engine=self.engine, state_mgr=self.state_mgr, cfg=self.cfg
        )
        self.pos_mgr = PositionManager(
            exec_engine=self.engine, state_mgr=self.state_mgr, cfg=self.cfg
        )
        self.connect_fn = connect_fn or pastikan_mt5_aktif

        self._cycle_lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._was_connected: bool = False
        self._startup_reconciled: bool = False
        self._cached_scan: Optional[Dict[str, Any]] = None
        self._cached_decision: Optional[Dict[str, Any]] = None
        self._cached_snr: Optional[Dict[str, Any]] = None
        self._cached_snapshot: Optional[Dict[str, Any]] = None
        self._last_structure_refresh_mono: float = 0.0
        self._last_m5_bucket: Optional[int] = None
        self._last_cycle_summary: Dict[str, Any] = {}

    # -------------------------------------------------------------------------
    # BACKGROUND THREAD LIFECYCLE
    # -------------------------------------------------------------------------
    def start_background(self) -> bool:
        if self._thread is not None and self._thread.is_alive():
            return True
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._background_loop,
            name="xauusd-auto-exec-monitor",
            daemon=True,
        )
        self._thread.start()
        return True

    def stop_background(self, timeout: float = 3.0) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self._thread = None

    @property
    def is_running(self) -> bool:
        return bool(self._thread is not None and self._thread.is_alive() and not self._stop_event.is_set())

    def _background_loop(self) -> None:
        interval = max(0.25, float(self.cfg.get("MONITOR_INTERVAL_SECONDS", 1.0)))
        while not self._stop_event.is_set():
            t0 = time.monotonic()
            try:
                mode_info = self.state_mgr.get_mode_info()
                # Jalankan siklus aktif apabila auto_trading ON atau masih ada posisi/order scanner aktif yang wajib dikawal
                pos_recs, ord_recs = self.state_mgr.get_active_scanner_records()
                should_tick = bool(
                    mode_info.get("auto_trading")
                    or mode_info.get("trading_mode") in ("DRY_RUN", "SEMI_AUTO", "FULL_AUTO")
                    or pos_recs
                    or ord_recs
                    or not self._startup_reconciled
                )
                if should_tick:
                    self.run_once()
            except Exception as exc:
                _catat_error("auto_execution_monitor_loop", exc)

            elapsed = time.monotonic() - t0
            wait_s = max(0.05, interval - elapsed)
            self._stop_event.wait(timeout=wait_s)

    # -------------------------------------------------------------------------
    # STARTUP & RECONNECT RECONCILIATION (SECTION 20)
    # -------------------------------------------------------------------------
    def reconcile_now(self) -> Dict[str, Any]:
        """Rekonsiliasi penuh posisi, order, histori deal, daily counter, dan proteksi SL."""
        magic = int(self.cfg.get("SCANNER_MAGIC_NUMBER", 20260991))
        acc_snap = inspect_mt5_account_and_permissions(
            mt5_api=self.mt5, symbol=self.symbol, state_mgr=self.state_mgr, cfg=self.cfg
        )
        if not acc_snap.get("connected"):
            self._was_connected = False
            return {"ok": False, "reason_code": REASON_MT5_DISCONNECTED, "account": acc_snap}

        with MT5_IPC_LOCK:
            positions = list(self.mt5.positions_get(symbol=self.symbol) or self.mt5.positions_get() or ())
            orders = list(self.mt5.orders_get(symbol=self.symbol) or self.mt5.orders_get() or ())
            deals = []
            if hasattr(self.mt5, "history_deals_get"):
                try:
                    now_u = datetime.now(timezone.utc)
                    deals = list(self.mt5.history_deals_get(now_u - timedelta(days=2), now_u + timedelta(days=1)) or ())
                except Exception:
                    deals = []

        recon = self.state_mgr.reconcile_with_mt5(
            positions=positions, orders=orders, history_deals=deals, magic_number=magic
        )
        # Catat ke jurnal jika terdapat transisi posisi yang tertutup saat offline/restart
        for tr in (recon.get("transitions") or []):
            if tr.get("to") == STATE_COMPLETED:
                sid_tr = str(tr.get("setup_id") or "")
                rec_tr = self.state_mgr.get_setup(sid_tr) or {"setup_id": sid_tr, "position_ticket": tr.get("ticket")}
                self.journal.record_trade_closed(
                    setup_rec=rec_tr,
                    realized_pnl=float(tr.get("pnl") or 0.0),
                    exit_reason=str(tr.get("exit_reason") or "CLOSED"),
                    account_mode=str(acc_snap.get("account_mode") or ""),
                    currency=str(acc_snap.get("currency") or ""),
                )

        # Verifikasi proteksi SL seluruh posisi scanner segera setelah reconnect/startup
        pos_audit = self.pos_mgr.audit_and_manage_positions(symbol=self.symbol)
        portfolio = evaluate_portfolio_risk_and_daily_limits(
            mt5_api=self.mt5, state_mgr=self.state_mgr, symbol=self.symbol, cfg=self.cfg
        )

        self._was_connected = True
        self._startup_reconciled = True
        return {
            "ok": True,
            "account": acc_snap,
            "reconciliation": recon,
            "position_audit": pos_audit,
            "portfolio": portfolio,
        }

    # -------------------------------------------------------------------------
    # SINGLE MONITORING & EXECUTION CYCLE
    # -------------------------------------------------------------------------
    def run_once(
        self,
        snapshot: Optional[Dict[str, Any]] = None,
        scan_result: Optional[Dict[str, Any]] = None,
        decision: Optional[Dict[str, Any]] = None,
        snr_result: Optional[Dict[str, Any]] = None,
        news_override: Optional[Dict[str, Any]] = None,
        force_structure_refresh: bool = False,
    ) -> Dict[str, Any]:
        """
        Jalankan 1 siklus monitoring real-time secara thread-safe:
        1. Cek koneksi MT5 & rekonsiliasi otomatis jika baru terhubung kembali.
        2. Kelola posisi aktif (verifikasi SL / emergency protection) & pending order (auto-cancel invalid).
        3. Refresh struktur market bila candle M5 berganti atau snapshot eksternal diberikan.
        4. Evaluasi kandidat zona terbaik dengan 7-layer validator.
        5. Jika ENTRY_READY & auto_trading aktif -> eksekusi sesuai mode trading!
        """
        with self._cycle_lock:
            if not getattr(self, "_custom_cfg_provided", False):
                self.cfg.update(get_auto_trading_config())
            mode_info = self.state_mgr.get_mode_info()
            trading_mode = str(mode_info.get("trading_mode") or "SCANNER_ONLY").upper()
            auto_enabled = bool(mode_info.get("auto_trading", False))

            # Pastikan koneksi MT5 hidup jika menggunakan live MT5
            if snapshot is None:
                try:
                    self.connect_fn()
                except Exception:
                    pass

            acc_snap = inspect_mt5_account_and_permissions(
                mt5_api=self.mt5,
                symbol=self.symbol,
                state_mgr=self.state_mgr,
                require_exec_permission=(auto_enabled and trading_mode in ("FULL_AUTO", "SEMI_AUTO")),
                trading_mode=trading_mode,
                cfg=self.cfg,
            )

            # Jika koneksi MT5 terputus -> hentikan eksekusi baru, jangan pakai tick lama!
            if not acc_snap.get("connected"):
                self._was_connected = False
                summary = {
                    "timestamp": format_wib_timestamp(default_now=True),
                    "connected": False,
                    "trading_mode": trading_mode,
                    "auto_trading": auto_enabled,
                    "state": STATE_BLOCKED,
                    "decision": STATE_BLOCKED,
                    "reason_code": REASON_MT5_DISCONNECTED,
                    "reason_text": acc_snap.get("reason_text") or "MT5 Disconnected — eksekusi baru dihentikan",
                    "account": acc_snap,
                    "pending_warning": "PERINGATAN: Pending order di server broker tetap aktif saat koneksi Python/MT5 terputus.",
                }
                self.state_mgr.record_last_decision(summary)
                self._last_cycle_summary = summary
                return summary

            # Jika baru tersambung (startup atau reconnect pasca-disconnect) -> rekonsiliasi penuh terlebih dahulu!
            if not self._was_connected or not self._startup_reconciled:
                self.reconcile_now()
            else:
                # Rekonsiliasi ringan posisi/order setiap siklus agar fill/close baru segera tercatat
                with MT5_IPC_LOCK:
                    pos_l = list(self.mt5.positions_get(symbol=self.symbol) or ())
                    ord_l = list(self.mt5.orders_get(symbol=self.symbol) or ())
                recon_step = self.state_mgr.reconcile_with_mt5(
                    pos_l, ord_l, magic_number=int(self.cfg.get("SCANNER_MAGIC_NUMBER", 20260991))
                )
                for tr in (recon_step.get("transitions") or []):
                    if tr.get("to") == STATE_COMPLETED:
                        sid_tr = str(tr.get("setup_id") or "")
                        rec_tr = self.state_mgr.get_setup(sid_tr) or {"setup_id": sid_tr, "position_ticket": tr.get("ticket")}
                        self.journal.record_trade_closed(
                            setup_rec=rec_tr,
                            realized_pnl=float(tr.get("pnl") or 0.0),
                            exit_reason=str(tr.get("exit_reason") or "CLOSED"),
                            account_mode=str(acc_snap.get("account_mode") or ""),
                            currency=str(acc_snap.get("currency") or ""),
                        )

            # Audit posisi scanner aktif (proteksi SL darurat, BEP/Trailing opsional)
            pos_audit = self.pos_mgr.audit_and_manage_positions(symbol=self.symbol)

            # Tentukan apakah struktur candle perlu di-refresh penuh atau cukup update live tick
            tick = acc_snap.get("tick")
            tick_ts = int(getattr(tick, "time", 0) or 0) if tick is not None else 0
            cur_m5_bucket = (tick_ts // 300) if tick_ts > 0 else None
            now_mono = time.monotonic()

            if snapshot is not None and scan_result is not None:
                eff_snap = snapshot
                eff_scan = scan_result
                eff_dec = decision if decision is not None else evaluate_trade_decision(snapshot=eff_snap, scan_result=eff_scan)
                eff_snr = snr_result if snr_result is not None else self._cached_snr
                self._cached_snapshot = eff_snap
                self._cached_scan = eff_scan
                self._cached_decision = eff_dec
                if snr_result is not None:
                    self._cached_snr = snr_result
                self._last_structure_refresh_mono = now_mono
                if cur_m5_bucket is not None:
                    self._last_m5_bucket = cur_m5_bucket
            else:
                need_full_refresh = (
                    force_structure_refresh
                    or self._cached_scan is None
                    or self._cached_snapshot is None
                    or (cur_m5_bucket is not None and cur_m5_bucket != self._last_m5_bucket)
                    or ((now_mono - self._last_structure_refresh_mono) >= 60.0)
                )
                if need_full_refresh:
                    eff_snap = create_market_snapshot(symbol=self.symbol, tick=tick)
                    eff_scan = scan_xauusd_zones(snapshot=eff_snap, symbol=self.symbol)
                    eff_dec = evaluate_trade_decision(snapshot=eff_snap, scan_result=eff_scan)
                    eff_snr = None
                    try:
                        from modules.snr import scan_snr as _run_snr_scan
                        _s_res, _s_dec, _ = _run_snr_scan(snapshot=eff_snap, live=False, detailed=False)
                        eff_snr = {"scan_result": _s_res, "decision": _s_dec}
                    except Exception:
                        eff_snr = None
                    self._cached_snapshot = eff_snap
                    self._cached_scan = eff_scan
                    self._cached_decision = eff_dec
                    self._cached_snr = eff_snr
                    self._last_structure_refresh_mono = now_mono
                    if cur_m5_bucket is not None:
                        self._last_m5_bucket = cur_m5_bucket
                else:
                    # Update harga live pada snapshot cached tanpa memanggil ulang copy_rates berat
                    eff_snap = dict(self._cached_snapshot)
                    eff_scan = dict(self._cached_scan)
                    eff_dec = self._cached_decision
                    eff_snr = self._cached_snr
                    if tick is not None:
                        b_v = float(getattr(tick, "bid", 0.0) or 0.0)
                        a_v = float(getattr(tick, "ask", 0.0) or 0.0)
                        if b_v > 0 and a_v >= b_v:
                            m_v = round((b_v + a_v) / 2.0, 2)
                            sp_v = round(a_v - b_v, 3)
                            eff_snap.update({"bid": b_v, "ask": a_v, "mid": m_v, "current_price": m_v, "spread": sp_v, "tick_age": 0.0})
                            eff_scan.update({"bid": b_v, "ask": a_v, "mid": m_v, "current_price": m_v, "spread": sp_v})

            # Audit pending order scanner terhadap struktur terbaru & news blackout
            ord_audit = self.order_mgr.audit_and_manage_pending_orders(
                snapshot=eff_snap,
                scan_result=eff_scan,
                news_override=news_override,
                symbol=self.symbol,
            )

            # Evaluasi kandidat zona terbaik (Hybrid SMC + SNR) menggunakan Validator 7-Lapis
            # Jika ada plan invalid/missed, jangan berhenti: release & cari kandidat pengganti otomatis!
            candidates = select_candidate_zones(eff_scan, eff_dec, snr_result=eff_snr)
            best_eval: Optional[Dict[str, Any]] = None
            evaluated_list: List[Dict[str, Any]] = []

            for cand_z in candidates[:6]:
                val_res = validate_trade_candidate(
                    zone_in=cand_z,
                    mt5_api=self.mt5,
                    state_mgr=self.state_mgr,
                    snapshot=eff_snap,
                    scan_result=eff_scan,
                    decision=eff_dec,
                    trading_mode=trading_mode,
                    news_override=news_override,
                    cfg=self.cfg,
                )
                evaluated_list.append(val_res)
                if val_res.get("state") in ("INVALIDATED", "EXPIRED") or val_res.get("reason_code") == REASON_PRICE_LEFT_ENTRY_RANGE:
                    # Release locked plan lama pada state_mgr agar tidak menghalangi pencarian kandidat pengganti
                    self.state_mgr.update_setup_state(
                        str(val_res["setup_id"]),
                        state=str(val_res["state"] if val_res["state"] in ("INVALIDATED", "EXPIRED") else "EXPIRED"),
                        reason_code=str(val_res["reason_code"]),
                        reason_text=str(val_res["reason_text"]),
                        zone_id=str(val_res["zone_id"]),
                    )
                if val_res.get("can_execute") and val_res.get("state") == STATE_ENTRY_READY:
                    best_eval = val_res
                    break

            if best_eval is None and evaluated_list:
                # Prioritaskan kandidat pengganti yang masih aktif/valid dibanding zona invalid/missed
                state_rank = {
                    STATE_POSITION_ACTIVE: 0,
                    STATE_ORDER_PLACED: 1,
                    "WAIT_TRIGGER": 2,
                    "ZONE_TOUCHED": 3,
                    "APPROACHING": 4,
                    "RADAR": 5,
                    STATE_BLOCKED: 6,
                    "EXPIRED": 8,
                    "INVALIDATED": 9,
                }
                evaluated_list.sort(key=lambda r: (state_rank.get(str(r.get("state")), 9), -float((r.get("zone") or {}).get("score") or 0.0)))
                best_eval = evaluated_list[0]

            execution_outcome = None
            if best_eval is not None:
                sid = str(best_eval["setup_id"])
                cur_rec = self.state_mgr.get_setup(sid)
                # Update state non-terminal apabila belum ORDER_PLACED / POSITION_ACTIVE
                if not (isinstance(cur_rec, dict) and cur_rec.get("state") in (STATE_ORDER_PLACED, STATE_POSITION_ACTIVE)):
                    self.state_mgr.update_setup_state(
                        sid,
                        state=str(best_eval["state"]),
                        reason_code=str(best_eval["reason_code"]),
                        reason_text=str(best_eval["reason_text"]),
                        zone_id=str(best_eval["zone_id"]),
                        symbol=self.symbol,
                        direction=str((best_eval.get("trade_plan") or {}).get("direction") or ""),
                        entry_mode=str(best_eval.get("entry_mode") or ""),
                        planned_entry=float((best_eval.get("trade_plan") or {}).get("entry") or 0.0),
                        sl=float((best_eval.get("trade_plan") or {}).get("sl") or 0.0),
                        tp=float((best_eval.get("trade_plan") or {}).get("tp") or 0.0),
                        target_lot=float(best_eval.get("target_lot") or 0.0),
                        final_lot=float(best_eval.get("final_lot") or 0.0),
                        zone_score=float((best_eval.get("zone") or {}).get("score") or 0.0),
                        zone_grade=str((best_eval.get("zone") or {}).get("grade") or ""),
                        zone_quality=str(best_eval.get("zone_quality") or ""),
                        zone_timeframe=str((best_eval.get("zone") or {}).get("timeframe") or ""),
                        trigger_name=str((best_eval.get("triggers") or {}).get("primary_trigger") or ""),
                    )

                # Catat missed entry ke jurnal bila harga meninggalkan zona
                log_blocked_pref = bool(self.cfg.get("JOURNAL_LOG_BLOCKED_EVENTS", False))
                if best_eval.get("reason_code") == REASON_PRICE_LEFT_ENTRY_RANGE:
                    self.journal.record_execution_event(
                        event_type="MISSED_ENTRY",
                        validation_result=best_eval,
                        trading_mode=trading_mode,
                        decision_reason=REASON_PRICE_LEFT_ENTRY_RANGE,
                    )
                elif log_blocked_pref and best_eval.get("state") == STATE_BLOCKED and best_eval.get("reason_code") not in (REASON_MARKET_CLOSED, "DUPLICATE_SETUP"):
                    self.journal.record_execution_event(
                        event_type="BLOCKED",
                        validation_result=best_eval,
                        trading_mode=trading_mode,
                        decision_reason=str(best_eval.get("reason_code") or ""),
                    )

                # EKSEKUSI OTOMATIS APABILA ENTRY_READY DAN AUTO_TRADING AKTIF!
                if best_eval.get("can_execute") and best_eval.get("state") == STATE_ENTRY_READY:
                    if auto_enabled and trading_mode in ("FULL_AUTO", "DRY_RUN", "SEMI_AUTO"):
                        execution_outcome = self.engine.execute_validated_setup(best_eval, trading_mode=trading_mode)
                        best_eval["execution_result"] = execution_outcome
                        if execution_outcome.get("executed"):
                            best_eval["execution_status"] = execution_outcome.get("state", STATE_POSITION_ACTIVE)
                            best_eval["state"] = execution_outcome.get("state", STATE_POSITION_ACTIVE)
                        elif execution_outcome.get("simulated"):
                            best_eval["execution_status"] = execution_outcome.get("execution_status", "DRY_RUN_SIMULATED")
                        elif execution_outcome.get("queued_for_approval"):
                            best_eval["execution_status"] = "AWAITING_USER_APPROVAL"
                        else:
                            best_eval["execution_status"] = execution_outcome.get("state", STATE_BLOCKED)
                            best_eval["decision"] = STATE_BLOCKED
                            best_eval["reason_code"] = execution_outcome.get("reason_code", best_eval["reason_code"])
                            best_eval["reason_text"] = execution_outcome.get("reason_text", best_eval["reason_text"])
                    else:
                        best_eval["execution_status"] = "AUTO_TRADING_DISABLED"

            cycle_summary = {
                "timestamp": format_wib_timestamp(default_now=True),
                "connected": True,
                "market_open": bool(acc_snap.get("market_open")),
                "trading_mode": trading_mode,
                "auto_trading": auto_enabled,
                "account": acc_snap,
                "best_evaluation": best_eval,
                "execution_outcome": execution_outcome,
                "position_audit": pos_audit,
                "order_audit": ord_audit,
                "premarket_plan": (eff_dec.get("best_trade_plan") if isinstance(eff_dec, dict) else None) if not acc_snap.get("market_open") else None,
            }
            self.state_mgr.record_last_decision({
                "timestamp": cycle_summary["timestamp"],
                "trading_mode": trading_mode,
                "auto_trading": auto_enabled,
                "setup_id": (best_eval or {}).get("setup_id", ""),
                "zone_id": (best_eval or {}).get("zone_id", ""),
                "decision": (best_eval or {}).get("decision", "NO_SETUP"),
                "execution_status": (best_eval or {}).get("execution_status", "IDLE"),
                "reason_code": (best_eval or {}).get("reason_code", ""),
                "reason_text": (best_eval or {}).get("reason_text", ""),
                "target_lot": (best_eval or {}).get("target_lot", 0.0),
                "final_lot": (best_eval or {}).get("final_lot", 0.0),
            })
            self._last_cycle_summary = cycle_summary
            return cycle_summary


# =============================================================================
# FORMATTER TAMPILAN EKSEKUSI MODE PANTAU (SECTION 18)
# =============================================================================
def format_adaptive_execution_panel(
    cycle_summary: Optional[Dict[str, Any]],
    state_mgr: Optional[ExecutionStateManager] = None,
) -> str:
    """
    Format ringkas & akurat status Adaptive Auto Trading untuk Mode Pantau (Section 18).
    Seluruh angka berasal dari hasil analisis Python, risk manager, dan respons aktual MT5.
    """
    sm = state_mgr or get_execution_state_manager()
    mode_info = sm.get_mode_info()
    cyc = cycle_summary or {}
    acc = cyc.get("account") or {}
    ev = cyc.get("best_evaluation") or {}

    acc_mode = str(acc.get("account_mode") or mode_info.get("authorized_account", {}).get("account_mode") or "UNKNOWN")
    currency = str(acc.get("currency") or mode_info.get("authorized_account", {}).get("currency") or "N/A")
    auto_on = bool(cyc.get("auto_trading", mode_info.get("auto_trading", False)))
    t_mode = str(cyc.get("trading_mode") or mode_info.get("trading_mode") or "SCANNER_ONLY")
    mt5_stat = "CONNECTED" if cyc.get("connected") else "DISCONNECTED"
    if cyc.get("connected") and not cyc.get("market_open", True):
        mt5_stat = "CONNECTED (MARKET CLOSED)"

    auto_lbl = f"ENABLED ({t_mode})" if auto_on else f"DISABLED ({t_mode})"

    lines = [
        "=" * 52,
        "       XAUUSD SMC ADAPTIVE AUTO TRADING",
        "=" * 52,
        f"ACCOUNT MODE : {acc_mode}",
        f"CURRENCY     : {currency}",
        f"AUTO TRADING : {auto_lbl}",
        f"MT5 STATUS   : {mt5_stat}",
    ]

    if not ev:
        lines.extend([
            "",
            "DECISION     : WAIT",
            "REASON       : NO_QUALIFIED_ZONE_CANDIDATE",
            "NEXT ACTION  : Terus memindai SMC & SNR secara otomatis untuk mencari kandidat zona valid berikutnya",
            "=" * 52,
        ])
        return "\n".join(lines)

    bias = ev.get("market_bias") or {}
    plan = ev.get("trade_plan") or {}
    zone = ev.get("zone") or {}
    touch = ev.get("touch") or {}
    trigs = ev.get("triggers") or {}
    exec_res = ev.get("execution_result") or cyc.get("execution_outcome") or {}

    live_px = float(ev.get("live_price") or 0.0)
    z_tf = str(zone.get("timeframe") or plan.get("zone_timeframe") or "").replace("+", " + ")
    z_type_raw = str(zone.get("zone_type") or "")
    z_status = "INVALIDATED" if ev.get("state") == "INVALIDATED" else ("EXPIRED" if ev.get("state") == "EXPIRED" else "ACTIVE")
    z_score = float(zone.get("score") or plan.get("zone_score") or 0.0)
    z_grade = str(zone.get("grade") or plan.get("zone_grade") or "-")

    lines.extend([
        "",
        f"LIVE PRICE   : {live_px:.2f}",
        f"DIRECTION    : {plan.get('direction', '-')}",
        "",
        f"H4           : {bias.get('H4', '-')}",
        f"H1           : {bias.get('H1', '-')}",
        f"M30          : {bias.get('M30', '-')}",
        f"M5           : {bias.get('M5', '-')}",
        "",
        f"ZONE TYPE    : {z_tf} {z_type_raw}".strip(),
        f"ZONE STATUS  : {z_status}",
        f"ZONE QUALITY : {ev.get('zone_quality', '-')}",
        f"ZONE SCORE   : {z_grade} / {z_score:.0f}",
        f"ZONE TOUCH   : {str(bool(touch.get('touched'))).upper()}",
        "",
        f"ENTRY MODE   : {str(ev.get('entry_mode', 'ADAPTIVE')).replace('_ENTRY', '')}",
        f"TRIGGER      : {trigs.get('primary_trigger', 'NONE')}",
        f"CHoCH        : {trigs.get('choch_status', 'NOT CONFIRMED')}",
        f"CISD         : {trigs.get('cisd_status', 'NOT CONFIRMED')}",
        "",
        f"ENTRY        : {float(plan.get('entry') or 0.0):.2f}",
        f"STRUCT INV   : {float(plan.get('invalidation_price') or zone.get('invalidation_price') or 0.0):.2f}",
        f"SL           : {float(plan.get('sl') or 0.0):.2f}",
        f"TP           : {float(plan.get('tp') or 0.0):.2f}",
        f"RR           : 1:{float(ev.get('actual_rr') or plan.get('planned_rr') or 0.0):.2f}",
        "",
        f"TARGET LOT   : {float(ev.get('target_lot') or 0.0):.2f}",
        f"FINAL LOT    : {float(ev.get('final_lot') or 0.0):.2f}",
        f"LOT REASON   : {ev.get('lot_reason', '-')}",
        "",
        f"RISK CHECK   : {ev.get('risk_check', '-')}",
        f"VALIDATION   : {ev.get('validation', '-')}",
        f"DECISION     : {ev.get('decision', '-')}",
        f"EXECUTION    : {ev.get('execution_status', '-')}",
    ])

    if ev.get("decision") != STATE_ENTRY_READY and ev.get("reason_code") and ev.get("reason_code") != "OK":
        lines.append(f"REASON       : {ev.get('reason_code')} ({ev.get('reason_text', '')})")
    if exec_res.get("retcode") is not None and not exec_res.get("simulated"):
        lines.append(f"MT5 RETCODE  : {exec_res.get('retcode')} (attempts={exec_res.get('attempts', 1)})")
    if ev.get("next_action"):
        lines.append(f"NEXT ACTION  : {ev.get('next_action')}")

    # Tampilkan detail tiket broker HANYA apabila benar-benar dikonfirmasi broker (bukan simulasi!)
    setup_rec = sm.get_setup(str(ev.get("setup_id") or "")) or {}
    if exec_res.get("executed") and not exec_res.get("simulated"):
        tkt_disp = exec_res.get("position_ticket") or exec_res.get("order_ticket")
        lines.extend([
            "-" * 52,
            f"ORDER TICKET : {tkt_disp}",
            f"ACTUAL ENTRY : {float(exec_res.get('actual_entry') or 0.0):.2f}",
            f"ACTUAL LOT   : {float(exec_res.get('actual_lot') or 0.0):.2f}",
            f"ACTUAL SL    : {float(exec_res.get('actual_sl') or 0.0):.2f}",
            f"ACTUAL TP    : {float(exec_res.get('actual_tp') or 0.0):.2f}",
        ])
    elif setup_rec.get("state") in (STATE_POSITION_ACTIVE, STATE_ORDER_PLACED):
        tkt_disp = setup_rec.get("position_ticket") or setup_rec.get("order_ticket")
        lines.extend([
            "-" * 52,
            f"ORDER TICKET : {tkt_disp}",
            f"ACTUAL ENTRY : {float(setup_rec.get('actual_entry') or setup_rec.get('planned_entry') or 0.0):.2f}",
            f"ACTUAL LOT   : {float(setup_rec.get('actual_lot') or setup_rec.get('final_lot') or 0.0):.2f}",
            f"ACTUAL SL    : {float(setup_rec.get('sl') or 0.0):.2f}",
            f"ACTUAL TP    : {float(setup_rec.get('tp') or 0.0):.2f}",
        ])

    lines.append("=" * 52)
    return "\n".join(lines)


# =============================================================================
# SINGLETON MONITOR & CLI 'auto ...' COMMAND HANDLER (SECTION 17)
# =============================================================================
_GLOBAL_AUTO_MONITOR_LOCK = threading.Lock()
_GLOBAL_AUTO_MONITOR: Optional[AutoExecutionMonitor] = None


def get_auto_execution_monitor() -> AutoExecutionMonitor:
    global _GLOBAL_AUTO_MONITOR
    with _GLOBAL_AUTO_MONITOR_LOCK:
        if _GLOBAL_AUTO_MONITOR is None:
            _GLOBAL_AUTO_MONITOR = AutoExecutionMonitor()
        return _GLOBAL_AUTO_MONITOR


def handle_auto_cli_command(
    raw_cmd: str,
    input_fn: Optional[Callable[[str], str]] = input,
    monitor: Optional[AutoExecutionMonitor] = None,
) -> Dict[str, Any]:
    """
    Handler terpusat untuk seluruh command CLI `auto ...` (Section 17):
    - auto status
    - auto on
    - auto off
    - auto real
    - auto demo
    - auto dryrun
    - auto semi (serta auto approve / auto reject)
    - auto positions
    - auto orders
    - auto journal
    - auto risk
    - auto emergency (serta auto emergency close)
    """
    mon = monitor or get_auto_execution_monitor()
    if not getattr(mon, "_custom_cfg_provided", False):
        mon.cfg.update(get_auto_trading_config())
    sm = mon.state_mgr
    conf = mon.cfg
    tokens = str(raw_cmd or "").strip().split()
    sub = tokens[1].lower() if len(tokens) > 1 else "status"
    arg2 = tokens[2].lower() if len(tokens) > 2 else ""

    try:
        mon.connect_fn()
    except Exception:
        pass

    acc = inspect_mt5_account_and_permissions(mt5_api=mon.mt5, symbol=mon.symbol, state_mgr=sm, cfg=conf)
    cur = acc.get("currency") or "UNIT"

    # 1. AUTO STATUS
    if sub == "status":
        cyc = mon.run_once()
        panel = format_adaptive_execution_panel(cyc, state_mgr=sm)
        approvals = sm.get_pending_approvals()
        extra = ""
        if approvals:
            extra = f"\n[SEMI_AUTO QUEUE]: {len(approvals)} setup menunggu persetujuan ('auto approve' / 'auto reject')."
        return {"ok": True, "subcommand": "status", "output": panel + extra, "cycle": cyc}

    # 2. AUTO REAL (Aktivasi FULL_AUTO eksplisit pada akun REAL IDC/USD dengan konfirmasi user!)
    if sub == "real":
        if not acc.get("connected"):
            return {"ok": False, "subcommand": "real", "output": f"[BLOCKED] MT5 tidak terhubung: {acc.get('reason_text')}"}
        if acc.get("account_mode") != "REAL":
            return {
                "ok": False,
                "subcommand": "real",
                "output": f"[BLOCKED] Akun yang sedang login bukan akun REAL (terdeteksi: {acc.get('account_mode')} #{acc.get('login')}). Gunakan 'auto demo' untuk akun demo.",
            }
        if not acc.get("algo_allowed"):
            return {
                "ok": False,
                "subcommand": "real",
                "output": "[BLOCKED] Algorithmic Trading / External Python API masih NONAKTIF di terminal MT5. Aktifkan tombol 'Algo Trading' di MT5 terlebih dahulu.",
            }

        summary_lines = [
            "=" * 64,
            "⚠️  KONFIRMASI AKTIVASI FULL_AUTO — REAL ACCOUNT MT5",
            "=" * 64,
            f"Account Login     : {acc.get('login')}",
            f"Server            : {acc.get('server')}",
            f"Account Mode      : {acc.get('account_mode')}",
            f"Account Currency  : {cur}",
            f"Balance           : {acc.get('balance', 0.0):,.2f} {cur}",
            f"Equity            : {acc.get('equity', 0.0):,.2f} {cur}",
            f"Free Margin       : {acc.get('free_margin', 0.0):,.2f} {cur}",
            f"Maximum Lot       : {conf.get('MAX_LOT', 0.05):.2f} lot",
            f"Risk per Trade    : Adaptive {conf.get('RISK_ADAPTIVE_PERCENT', 0.50):.2f}% | Early Pending {conf.get('RISK_EARLY_PENDING_PERCENT', 0.25):.2f}% (Abs Max {conf.get('MAX_RISK_PER_TRADE_PERCENT', 1.0):.2f}%)",
            f"Daily Loss Limit  : {conf.get('MAX_DAILY_LOSS_PERCENT', 2.0):.2f}%",
            f"Maximum Exposure  : {conf.get('MAX_TOTAL_EXPOSURE_PERCENT', 1.0):.2f}%",
            "=" * 64,
            "PERINGATAN: Sistem akan mengirim order LIVE secara otomatis pada akun REAL di atas",
            "ketika seluruh validasi ketat terpenuhi. Jangan pernah membagikan password akun Anda.",
        ]
        header_txt = "\n".join(summary_lines)
        if input_fn is None:
            return {"ok": False, "subcommand": "real", "output": header_txt + "\n[DIBATALKAN] Konfirmasi interaktif diperlukan."}

        print(header_txt)
        ans = str(input_fn("Ketik 'YES' atau 'KONFIRMASI REAL' untuk mengaktifkan FULL_AUTO REAL: ") or "").strip().upper()
        if ans not in ("YES", "KONFIRMASI REAL"):
            return {"ok": False, "subcommand": "real", "output": "[DIBATALKAN] Aktivasi FULL_AUTO pada akun REAL dibatalkan oleh user."}

        sm.authorize_account(
            login=int(acc["login"]),
            server=str(acc.get("server") or ""),
            currency=cur,
            account_mode="REAL",
            real_confirmed=True,
            balance=float(acc.get("balance") or 0.0),
            equity=float(acc.get("equity") or 0.0),
        )
        sm.set_trading_mode("FULL_AUTO", auto_trading=True, clear_emergency=True)
        mon.reconcile_now()
        mon.start_background()
        return {
            "ok": True,
            "subcommand": "real",
            "output": f"[AKTIF] FULL_AUTO pada akun REAL #{acc.get('login')} ({cur}) berhasil diotorisasi dan background monitor aktif.",
        }

    # 3. AUTO DEMO
    if sub == "demo":
        if not acc.get("connected"):
            return {"ok": False, "subcommand": "demo", "output": f"[BLOCKED] MT5 tidak terhubung: {acc.get('reason_text')}"}
        if acc.get("account_mode") == "REAL":
            return {
                "ok": False,
                "subcommand": "demo",
                "output": f"[BLOCKED] Akun yang sedang login adalah akun REAL (#{acc.get('login')}). Gunakan 'auto real' atau 'auto dryrun'.",
            }
        if not acc.get("algo_allowed"):
            return {"ok": False, "subcommand": "demo", "output": "[BLOCKED] Algo Trading belum diaktifkan di terminal MT5."}
        sm.authorize_account(
            login=int(acc["login"]),
            server=str(acc.get("server") or ""),
            currency=cur,
            account_mode=str(acc.get("account_mode") or "DEMO"),
            real_confirmed=False,
            balance=float(acc.get("balance") or 0.0),
            equity=float(acc.get("equity") or 0.0),
        )
        sm.set_trading_mode("FULL_AUTO", auto_trading=True, clear_emergency=True)
        mon.reconcile_now()
        mon.start_background()
        return {
            "ok": True,
            "subcommand": "demo",
            "output": f"[AKTIF] FULL_AUTO pada akun DEMO #{acc.get('login')} ({cur}) aktif.",
        }

    # 4. AUTO DRYRUN
    if sub in ("dryrun", "dry", "dry_run"):
        if acc.get("connected") and acc.get("login") is not None:
            sm.authorize_account(
                login=int(acc["login"]),
                server=str(acc.get("server") or ""),
                currency=cur,
                account_mode=str(acc.get("account_mode") or "UNKNOWN"),
                real_confirmed=False,
                balance=float(acc.get("balance") or 0.0),
                equity=float(acc.get("equity") or 0.0),
            )
        sm.set_trading_mode("DRY_RUN", auto_trading=True, clear_emergency=True)
        mon.start_background()
        return {
            "ok": True,
            "subcommand": "dryrun",
            "output": "[AKTIF] Mode DRY_RUN aktif — evaluasi real-time menggunakan harga aktual MT5 tanpa mengirim order.",
        }

    # 5. AUTO SEMI (+ APPROVE / REJECT)
    if sub == "semi":
        if acc.get("connected") and acc.get("login") is not None:
            sm.authorize_account(
                login=int(acc["login"]),
                server=str(acc.get("server") or ""),
                currency=cur,
                account_mode=str(acc.get("account_mode") or "UNKNOWN"),
                real_confirmed=(acc.get("account_mode") == "REAL"),
                balance=float(acc.get("balance") or 0.0),
                equity=float(acc.get("equity") or 0.0),
            )
        sm.set_trading_mode("SEMI_AUTO", auto_trading=True, clear_emergency=True)
        mon.start_background()
        return {
            "ok": True,
            "subcommand": "semi",
            "output": "[AKTIF] Mode SEMI_AUTO aktif — setiap setup ENTRY_READY akan meminta persetujuan ('auto approve' / 'auto reject').",
        }

    if sub == "approve":
        queued = sm.pop_semi_auto_setup(arg2 if arg2 else None)
        if not queued:
            return {"ok": False, "subcommand": "approve", "output": "[INFO] Tidak ada setup SEMI_AUTO dalam antrean."}
        queued["user_approved"] = True
        res_ex = mon.engine.execute_validated_setup(queued, trading_mode="SEMI_AUTO")
        return {
            "ok": bool(res_ex.get("executed")),
            "subcommand": "approve",
            "output": f"[SEMI_AUTO EXECUTION] Status={res_ex.get('state')} | Ticket={res_ex.get('position_ticket') or res_ex.get('order_ticket') or '-'} | {res_ex.get('reason_text')}",
            "result": res_ex,
        }

    if sub == "reject":
        queued = sm.pop_semi_auto_setup(arg2 if arg2 else None)
        if not queued:
            return {"ok": False, "subcommand": "reject", "output": "[INFO] Tidak ada setup SEMI_AUTO dalam antrean."}
        sm.update_setup_state(str(queued.get("setup_id") or ""), STATE_BLOCKED, reason_code="USER_REJECTED", reason_text="Ditolak manual oleh user")
        return {"ok": True, "subcommand": "reject", "output": f"[DITOLAK] Setup {queued.get('setup_id')} dibatalkan dari antrean SEMI_AUTO."}

    # 6. AUTO ON
    if sub == "on":
        m_inf = sm.get_mode_info()
        cur_tmode = str(m_inf.get("trading_mode") or "SCANNER_ONLY").upper()
        if acc.get("account_mode") == "REAL":
            auth = m_inf.get("authorized_account") or {}
            if not auth.get("real_confirmed") or auth.get("login") != acc.get("login"):
                return {
                    "ok": False,
                    "subcommand": "on",
                    "output": "[BLOCKED] Akun REAL terdeteksi. Demi keamanan, gunakan perintah 'auto real' untuk konfirmasi eksplisit akun real, atau 'auto dryrun' untuk simulasi.",
                }
            target_mode = "FULL_AUTO" if cur_tmode == "SCANNER_ONLY" else cur_tmode
        else:
            target_mode = "FULL_AUTO" if cur_tmode == "SCANNER_ONLY" else cur_tmode
            if acc.get("connected") and acc.get("login") is not None:
                sm.authorize_account(
                    login=int(acc["login"]),
                    server=str(acc.get("server") or ""),
                    currency=cur,
                    account_mode=str(acc.get("account_mode") or "DEMO"),
                    real_confirmed=False,
                    balance=float(acc.get("balance") or 0.0),
                    equity=float(acc.get("equity") or 0.0),
                )
        sm.set_trading_mode(target_mode, auto_trading=True, clear_emergency=True)
        mon.reconcile_now()
        mon.start_background()
        return {"ok": True, "subcommand": "on", "output": f"[AKTIF] Auto Trading ENABLED dalam mode {target_mode}."}

    # 7. AUTO OFF
    if sub == "off":
        sm.set_trading_mode("SCANNER_ONLY", auto_trading=False)
        cancelled = []
        if acc.get("connected"):
            cancelled = mon.order_mgr.cancel_all_scanner_pending_orders(symbol=mon.symbol, reason="AUTO_OFF")
        c_cnt = sum(1 for c in cancelled if c.get("cancelled"))
        return {
            "ok": True,
            "subcommand": "off",
            "output": (
                f"[NONAKTIF] Auto Trading dimatikan (mode SCANNER_ONLY). "
                f"Pending order scanner dibatalkan: {c_cnt}. Posisi aktif tetap berjalan dengan proteksi SL/TP broker."
            ),
        }

    # 8. AUTO EMERGENCY
    if sub == "emergency":
        sm.trigger_emergency_halt("USER_EMERGENCY_COMMAND")
        cancelled = []
        closed = []
        if acc.get("connected"):
            cancelled = mon.order_mgr.cancel_all_scanner_pending_orders(symbol=mon.symbol, reason="EMERGENCY")
            magic = int(conf.get("SCANNER_MAGIC_NUMBER", 20260991))
            with MT5_IPC_LOCK:
                sc_pos = [
                    p for p in (mon.mt5.positions_get(symbol=mon.symbol) or mon.mt5.positions_get() or ())
                    if is_scanner_owned_item(p, magic)
                ]
            if sc_pos:
                want_close = (arg2 == "close")
                if not want_close and input_fn is not None:
                    ans_e = str(input_fn(
                        f"Terdapat {len(sc_pos)} posisi aktif milik scanner. Ketik 'CLOSE' untuk menutup seluruh posisi scanner sekarang (Enter untuk biarkan dengan SL/TP broker): "
                    ) or "").strip().upper()
                    want_close = (ans_e == "CLOSE")
                if want_close:
                    closed = mon.pos_mgr.emergency_close_all_scanner_positions(symbol=mon.symbol)

        c_ord = sum(1 for x in cancelled if x.get("cancelled"))
        c_pos = sum(1 for x in closed if x.get("closed"))
        return {
            "ok": True,
            "subcommand": "emergency",
            "output": (
                f"[🚨 EMERGENCY HALT AKTIF] Eksekusi baru dihentikan. "
                f"Pending order scanner dibatalkan: {c_ord} | Posisi scanner ditutup: {c_pos} "
                f"(Posisi/order manual user tidak disentuh)."
            ),
        }

    # 9. AUTO POSITIONS
    if sub == "positions":
        if not acc.get("connected"):
            return {"ok": False, "subcommand": "positions", "output": "[DISCONNECTED] MT5 tidak terhubung."}
        magic = int(conf.get("SCANNER_MAGIC_NUMBER", 20260991))
        with MT5_IPC_LOCK:
            all_pos = list(mon.mt5.positions_get(symbol=mon.symbol) or mon.mt5.positions_get() or ())
        lines = [
            "=" * 68,
            f"POSISI AKTIF MT5 ({len(all_pos)} Total | Mata Uang: {cur})",
            "=" * 68,
        ]
        if not all_pos:
            lines.append("Tidak ada posisi terbuka saat ini.")
        for p in all_pos:
            owner = "SCANNER" if is_scanner_owned_item(p, magic) else "MANUAL"
            p_dir = "BUY" if int(getattr(p, "type", 0) or 0) == 0 else "SELL"
            lines.append(
                f"[{owner:<7}] Ticket #{getattr(p, 'ticket', '-')} | {getattr(p, 'symbol', mon.symbol)} | "
                f"{p_dir:<4} {float(getattr(p, 'volume', 0.0)):.2f} lot @ {float(getattr(p, 'price_open', 0.0)):.2f} | "
                f"SL {float(getattr(p, 'sl', 0.0)):.2f} | TP {float(getattr(p, 'tp', 0.0)):.2f} | "
                f"P/L: {float(getattr(p, 'profit', 0.0)):+,.2f} {cur}"
            )
        lines.append("=" * 68)
        return {"ok": True, "subcommand": "positions", "output": "\n".join(lines)}

    # 10. AUTO ORDERS
    if sub == "orders":
        if not acc.get("connected"):
            return {"ok": False, "subcommand": "orders", "output": "[DISCONNECTED] MT5 tidak terhubung."}
        magic = int(conf.get("SCANNER_MAGIC_NUMBER", 20260991))
        with MT5_IPC_LOCK:
            all_ord = list(mon.mt5.orders_get(symbol=mon.symbol) or mon.mt5.orders_get() or ())
        lines = [
            "=" * 68,
            f"PENDING ORDER AKTIF MT5 ({len(all_ord)} Total)",
            "=" * 68,
        ]
        if not all_ord:
            lines.append("Tidak ada pending order aktif saat ini.")
        for o in all_ord:
            owner = "SCANNER" if is_scanner_owned_item(o, magic) else "MANUAL"
            o_t = int(getattr(o, "type", 2) or 2)
            o_lbl = "BUY LIMIT" if o_t == 2 else ("SELL LIMIT" if o_t == 3 else f"TYPE_{o_t}")
            vol_o = float(getattr(o, "volume_current", getattr(o, "volume_initial", 0.0)) or 0.0)
            lines.append(
                f"[{owner:<7}] Ticket #{getattr(o, 'ticket', '-')} | {getattr(o, 'symbol', mon.symbol)} | "
                f"{o_lbl:<10} {vol_o:.2f} lot @ {float(getattr(o, 'price_open', 0.0)):.2f} | "
                f"SL {float(getattr(o, 'sl', 0.0)):.2f} | TP {float(getattr(o, 'tp', 0.0)):.2f}"
            )
        if any(is_scanner_owned_item(o, magic) for o in all_ord):
            lines.append("⚠️ CATATAN: Pending order di server broker tetap aktif meskipun aplikasi Python ditutup.")
        lines.append("=" * 68)
        return {"ok": True, "subcommand": "orders", "output": "\n".join(lines)}

    # 11. AUTO RISK
    if sub == "risk":
        port = evaluate_portfolio_risk_and_daily_limits(mt5_api=mon.mt5, state_mgr=sm, symbol=mon.symbol, cfg=conf)
        rb_ad = compute_effective_risk_budget("ADAPTIVE_ENTRY", port, cfg=conf)
        rb_ep = compute_effective_risk_budget("EARLY_PENDING", port, cfg=conf, is_pending_order=True)
        lines = [
            "=" * 68,
            f"AUDIT MANAJEMEN RISIKO & KUOTA HARIAN ({cur})",
            "=" * 68,
            f"Balance / Equity       : {port['balance']:,.2f} {cur} / {port['equity']:,.2f} {cur}",
            f"Scanner Positions/Ords : Pos={port['scanner_positions_count']}/{conf['MAX_ACTIVE_POSITIONS']} | Pending={port['scanner_orders_count']}/{conf['MAX_ACTIVE_PENDING_ORDERS']}",
            f"Manual Positions/Ords  : Pos={port['manual_positions_count']} | Pending={port['manual_orders_count']} (Dihitung ke exposure, tidak dimodifikasi)",
            f"Total Open Exposure    : {port['total_open_risk_nominal']:,.2f} {cur} ({port['open_exposure_pct']:.2f}% / maks {conf['MAX_TOTAL_EXPOSURE_PERCENT']:.2f}%)",
            f"Realized P/L Today     : {port['realized_pnl_today']:+,.2f} {cur}",
            f"Floating P/L Net       : {port['floating_pnl_net']:+,.2f} {cur}",
            f"Effective Daily Loss   : {port['daily_loss_nominal']:,.2f} {cur} ({port['daily_loss_pct']:.2f}% / maks {conf['MAX_DAILY_LOSS_PERCENT']:.2f}%)",
            f"Trades Count Today     : {port['trades_today']} / {conf['MAX_TRADES_PER_DAY']}",
            f"Adaptive Risk Budget   : {rb_ad['risk_budget_nominal']:,.2f} {cur} ({rb_ad['effective_risk_pct']:.2f}%) [{rb_ad['reason_code']}]",
            f"Early Pending Budget   : {rb_ep['risk_budget_nominal']:,.2f} {cur} ({rb_ep['effective_risk_pct']:.2f}%) [{rb_ep['reason_code']}]",
            f"Lot Tiers              : Standard={conf['LOT_STANDARD']} (70-79) | Medium={conf['LOT_MEDIUM']} (80-89) | Strong={conf['LOT_STRONG']} (90+ HTF Validated)",
            "=" * 68,
        ]
        return {"ok": True, "subcommand": "risk", "output": "\n".join(lines), "portfolio": port}

    # 12. AUTO JOURNAL
    if sub == "journal":
        stats = mon.journal.compute_analytics()
        rows = mon.journal.read_rows()
        lines = [
            "=" * 72,
            "📊 ANALITIK JURNAL EKSEKUSI AUTO TRADING",
            "=" * 72,
            f"File Jurnal          : {mon.journal.journal_path}",
            f"Total Event Tercatat : {stats['total_events']} | Order Tereksekusi: {stats['executed_count']} | Posisi Closed: {stats['closed_trades']}",
            f"Win / Loss / WR      : {stats['wins']}W / {stats['losses']}L | Win Rate: {stats['win_rate_pct']:.1f}%",
            f"Expectancy / PF      : {stats['expectancy']:+,.2f} per trade | Profit Factor: {stats['profit_factor']}",
            f"Net P/L / Max DD     : {stats['net_pnl']:+,.2f} | Max Drawdown: {stats['max_drawdown']:,.2f}",
            f"Average Actual RR    : 1:{stats['average_rr']:.2f}",
            f"Missed / Blocked     : Missed={stats['missed_entries']} | Blocked={stats['blocked_entries']} | Rejected={stats['rejected_orders']} ({stats['rejection_rate_pct']:.1f}%)",
            f"Avg Execution Latency: {stats['avg_latency_ms']:.2f} ms",
        ]
        if stats["by_entry_mode"]:
            lines.append("Per Entry Mode       : " + " | ".join(
                f"{k}: {v['trades']}tr, WR {v['win_rate_pct']}%, Net {v['net_pnl']:+.2f}" for k, v in stats["by_entry_mode"].items()
            ))
        if stats["by_zone_timeframe"]:
            lines.append("Per Zone Timeframe   : " + " | ".join(
                f"{k}: {v['trades']}tr, Net {v['net_pnl']:+.2f}" for k, v in stats["by_zone_timeframe"].items()
            ))
        if stats["by_lot_classification"]:
            lines.append("Per Lot Tier         : " + " | ".join(
                f"{k} lot: {v['trades']}tr, Net {v['net_pnl']:+.2f}" for k, v in stats["by_lot_classification"].items()
            ))
        if rows:
            lines.append("-" * 72)
            lines.append("5 Event Terbaru:")
            for r in rows[-5:]:
                lines.append(
                    f"  • {r.get('timestamp', '')[:19]} | {r.get('event_type', ''):<16} | "
                    f"{r.get('direction', '')} {r.get('final_lot', '')} lot @ {r.get('actual_entry') or r.get('planned_entry')} | "
                    f"Mode={r.get('entry_mode')} | Score={r.get('zone_score')} | Reason={r.get('decision_reason')}"
                )
        lines.append("=" * 72)
        return {"ok": True, "subcommand": "journal", "output": "\n".join(lines), "analytics": stats}

    help_auto = (
        "Perintah Auto Trading yang tersedia:\n"
        "  auto status    | auto on       | auto off       | auto real\n"
        "  auto demo      | auto dryrun   | auto semi      | auto approve / reject\n"
        "  auto positions | auto orders   | auto journal   | auto risk\n"
        "  auto emergency [close]"
    )
    return {"ok": False, "subcommand": sub, "output": help_auto}
