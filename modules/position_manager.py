# -*- coding: utf-8 -*-
"""
modules/position_manager.py
Automatic Position Management & Emergency SL Protection (Section 15):
- Memverifikasi setiap posisi aktif milik scanner memiliki SL & TP aktual.
- Menjalankan Emergency Protection Policy apabila posisi scanner tidak memiliki SL:
  mencoba memasang SL struktural valid; jika ditolak broker dan kebijakan mengizinkan,
  menutup posisi scanner secara aman agar tidak pernah dibiarkan tanpa proteksi.
- Tidak pernah memperlebar SL, tidak pernah martingale / averaging loss, dan tidak pernah
  menutup posisi hanya karena muncul zona baru.
- Tidak pernah memodifikasi atau menutup posisi manual milik user!
- Break-even dan Trailing Stop tersedia sebagai fitur opsional (default: disabled).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from modules.config import SYMBOL, get_auto_trading_config
from modules.execution_engine import MT5ExecutionEngine, is_scanner_owned_item, normalize_price_to_tick
from modules.execution_state import ExecutionStateManager, STATE_COMPLETED
from modules.mt5_conn import MT5_IPC_LOCK
from modules.utils import get_symbol_unit_spec, mt5_points_to_price_distance


class PositionManager:
    """Pengelola proteksi posisi aktif milik scanner (strictly non-intrusive terhadap posisi manual user)."""

    def __init__(
        self,
        exec_engine: MT5ExecutionEngine,
        state_mgr: ExecutionStateManager,
        cfg: Optional[Dict[str, Any]] = None,
    ):
        self.engine = exec_engine
        self.mt5 = exec_engine.mt5
        self.state_mgr = state_mgr
        self.cfg = cfg or get_auto_trading_config()

    def audit_and_manage_positions(self, symbol: str = SYMBOL) -> Dict[str, Any]:
        """
        Audit seluruh posisi terbuka:
        - Pisahkan posisi scanner vs posisi manual user.
        - Pastikan posisi scanner selalu terproteksi SL valid (Emergency Protection).
        - Jalankan Break-Even / Trailing Stop HANYA bila diaktifkan di konfigurasi (default False).
        """
        conf = self.cfg
        magic = int(conf.get("SCANNER_MAGIC_NUMBER", 20260991))

        with MT5_IPC_LOCK:
            all_positions = list(self.mt5.positions_get(symbol=symbol) or self.mt5.positions_get() or ())

        scanner_positions = [p for p in all_positions if is_scanner_owned_item(p, magic)]
        manual_positions = [p for p in all_positions if not is_scanner_owned_item(p, magic)]

        emergency_actions: List[Dict[str, Any]] = []
        protection_updates: List[Dict[str, Any]] = []

        setups = self.state_mgr.get_all_setups()
        setup_by_pos_tkt: Dict[int, Dict[str, Any]] = {}
        for sid, rec in setups.items():
            pt = int(rec.get("position_ticket") or rec.get("order_ticket") or 0)
            if pt > 0:
                setup_by_pos_tkt[pt] = rec

        for pos in scanner_positions:
            tkt = int(getattr(pos, "ticket", 0) or 0)
            pos_sym = str(getattr(pos, "symbol", symbol) or symbol)
            is_buy = int(getattr(pos, "type", 0) or 0) == getattr(self.mt5, "ORDER_TYPE_BUY", 0)
            entry_px = float(getattr(pos, "price_open", 0.0) or 0.0)
            cur_sl = float(getattr(pos, "sl", 0.0) or 0.0)
            cur_tp = float(getattr(pos, "tp", 0.0) or 0.0)
            rec = setup_by_pos_tkt.get(tkt) or {}
            sid = str(rec.get("setup_id") or "")

            sym_info = self.mt5.symbol_info(pos_sym)
            tick = self.mt5.symbol_info_tick(pos_sym)

            # 1. EMERGENCY PROTECTION: Posisi scanner tanpa SL valid!
            if cur_sl <= 0.0:
                planned_sl = float(rec.get("planned_sl") or rec.get("sl") or 0.0)
                fallback_dist = min(float(conf.get("SL_JARAK_MAKS", 20.0)), 10.0)
                if planned_sl <= 0.0:
                    planned_sl = (entry_px - fallback_dist) if is_buy else (entry_px + fallback_dist)
                target_tp = cur_tp if cur_tp > 0 else float(rec.get("planned_tp") or rec.get("tp") or 0.0)
                if target_tp <= 0.0:
                    target_tp = (entry_px + 1.5 * fallback_dist) if is_buy else (entry_px - 1.5 * fallback_dist)

                mod_res = self.engine.modify_scanner_position_sl_tp(
                    tkt, new_sl=planned_sl, new_tp=target_tp, symbol=pos_sym, allow_initial_sl_attach=True
                )
                if mod_res.get("modified"):
                    if sid:
                        self.state_mgr.update_setup_state(sid, "POSITION_ACTIVE", sl=mod_res["sl"], tp=mod_res["tp"])
                    emergency_actions.append({
                        "ticket": tkt,
                        "action": "EMERGENCY_SL_ATTACHED",
                        "sl": mod_res["sl"],
                        "tp": mod_res["tp"],
                    })
                    continue

                # Jika pemasangan SL darurat gagal dan kebijakan penutupan darurat aktif -> tutup posisi scanner!
                if conf.get("EMERGENCY_CLOSE_ON_MISSING_SL", True):
                    cls_res = self.engine.close_scanner_position(tkt, symbol=pos_sym, reason="EMERGENCY_NO_SL")
                    if cls_res.get("closed") and sid:
                        self.state_mgr.update_setup_state(
                            sid, STATE_COMPLETED, reason_code="EMERGENCY_CLOSED_NO_SL",
                            reason_text="Posisi scanner ditutup otomatis karena SL gagal dipasang",
                        )
                    emergency_actions.append({
                        "ticket": tkt,
                        "action": "EMERGENCY_POSITION_CLOSED" if cls_res.get("closed") else "EMERGENCY_CLOSE_FAILED",
                        "detail": cls_res,
                    })
                else:
                    emergency_actions.append({
                        "ticket": tkt,
                        "action": "EMERGENCY_SL_FAILED_MANUAL_ALERT",
                        "detail": mod_res,
                    })
                continue

            # 2. BREAK-EVEN & TRAILING STOP OPSIONAL (Default: Disabled!)
            if tick is None or sym_info is None or entry_px <= 0:
                continue

            bid = float(getattr(tick, "bid", 0.0) or 0.0)
            ask = float(getattr(tick, "ask", 0.0) or 0.0)
            if bid <= 0 or ask <= 0:
                continue

            u_spec = get_symbol_unit_spec(symbol=pos_sym, sym_info=sym_info)
            stops_pts = mt5_points_to_price_distance(u_spec["trade_stops_level"] or 0, spec=u_spec)
            freeze_pts = mt5_points_to_price_distance(u_spec["trade_freeze_level"] or 0, spec=u_spec)
            min_brk_dist = max(stops_pts, freeze_pts, float(u_spec["trade_tick_size"] or 0.01))

            initial_sl = float(rec.get("sl") or cur_sl)
            init_risk = abs(entry_px - initial_sl) if initial_sl > 0 else 0.0
            cur_close_px = bid if is_buy else ask
            profit_pts = (cur_close_px - entry_px) if is_buy else (entry_px - cur_close_px)
            cur_rr = (profit_pts / init_risk) if init_risk > 0 else 0.0

            candidate_sl = None

            if conf.get("BREAK_EVEN_ENABLED", False) and init_risk > 0:
                be_rr = float(conf.get("BREAK_EVEN_TRIGGER_RR", 1.0))
                be_lock = float(conf.get("BREAK_EVEN_LOCK_POINTS", 0.30))
                if cur_rr >= be_rr:
                    be_sl = (entry_px + be_lock) if is_buy else (entry_px - be_lock)
                    if (is_buy and be_sl > cur_sl) or ((not is_buy) and be_sl < cur_sl):
                        candidate_sl = be_sl

            if conf.get("TRAILING_STOP_ENABLED", False) and init_risk > 0:
                tr_rr = float(conf.get("TRAILING_STOP_ACTIVATE_RR", 1.5))
                tr_dist = float(conf.get("TRAILING_STOP_DISTANCE_POINTS", 2.5))
                if cur_rr >= tr_rr and tr_dist > 0:
                    tr_sl = (cur_close_px - tr_dist) if is_buy else (cur_close_px + tr_dist)
                    if candidate_sl is None:
                        if (is_buy and tr_sl > cur_sl) or ((not is_buy) and tr_sl < cur_sl):
                            candidate_sl = tr_sl
                    else:
                        candidate_sl = max(candidate_sl, tr_sl) if is_buy else min(candidate_sl, tr_sl)

            if candidate_sl is not None:
                norm_cand = normalize_price_to_tick(candidate_sl, sym_info, symbol=pos_sym)
                # Pastikan patuh stops/freeze level & hanya memperketat SL
                valid_tighten = (
                    (is_buy and norm_cand > cur_sl + 1e-6 and (bid - norm_cand) >= min_brk_dist)
                    or ((not is_buy) and norm_cand < cur_sl - 1e-6 and (norm_cand - ask) >= min_brk_dist)
                )
                if valid_tighten:
                    m_res = self.engine.modify_scanner_position_sl_tp(
                        tkt, new_sl=norm_cand, new_tp=cur_tp, symbol=pos_sym
                    )
                    if m_res.get("modified"):
                        protection_updates.append({
                            "ticket": tkt,
                            "old_sl": cur_sl,
                            "new_sl": m_res["sl"],
                        })

        return {
            "scanner_positions_count": len(scanner_positions),
            "manual_positions_count": len(manual_positions),
            "emergency_actions": emergency_actions,
            "protection_updates": protection_updates,
        }

    def emergency_close_all_scanner_positions(self, symbol: str = SYMBOL) -> List[Dict[str, Any]]:
        """Tutup seluruh posisi milik scanner (HANYA dipanggil jika user mengonfirmasi eksplisit pada `auto emergency close`)."""
        magic = int(self.cfg.get("SCANNER_MAGIC_NUMBER", 20260991))
        with MT5_IPC_LOCK:
            positions = list(self.mt5.positions_get(symbol=symbol) or self.mt5.positions_get() or ())
        results = []
        for p in positions:
            if not is_scanner_owned_item(p, magic):
                continue
            tkt = int(getattr(p, "ticket", 0) or 0)
            res = self.engine.close_scanner_position(tkt, symbol=symbol, reason="USER_EMERGENCY")
            res["ticket"] = tkt
            results.append(res)
        return results
