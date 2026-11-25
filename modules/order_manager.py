# -*- coding: utf-8 -*-
"""
modules/order_manager.py
Automatic Pending Order Management (Section 14):
- Memonitor pending order milik scanner (`magic == SCANNER_MAGIC_NUMBER` / `SMC_`).
- Membatalkan otomatis apabila zona invalidated, setup expired, konflik HTF muncul,
  risk budget/daily loss terlampaui, atau memasuki high-impact news blackout.
- Mencegah duplicate market order pada setup yang sudah memiliki pending order aktif.
- Menangani transisi aman Pending -> Market (Cancel -> Verify -> Check Fill -> Reconcile -> Revalidate -> Send).
- Tidak pernah menyentuh atau membatalkan pending order manual milik user.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

from modules.config import SYMBOL, get_auto_trading_config
from modules.execution_engine import MT5ExecutionEngine, is_scanner_owned_item
from modules.execution_state import (
    ExecutionStateManager,
    REASON_DAILY_LOSS_LIMIT,
    REASON_DUPLICATE_SETUP,
    REASON_HTF_CONFLICT,
    REASON_NEWS_BLACKOUT,
    REASON_ZONE_EXPIRED,
    REASON_ZONE_INVALIDATED,
    STATE_EXPIRED,
    STATE_INVALIDATED,
    STATE_ORDER_PLACED,
    STATE_POSITION_ACTIVE,
)
from modules.mt5_conn import MT5_IPC_LOCK
from modules.risk_manager import evaluate_portfolio_risk_and_daily_limits
from modules.trade_validator import evaluate_directional_alignment, evaluate_news_hard_filter, validate_zone_quality_layer


class PendingOrderManager:
    """Pengelola siklus hidup pending order scanner secara otomatis & aman."""

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

    def audit_and_manage_pending_orders(
        self,
        snapshot: Optional[Dict[str, Any]] = None,
        scan_result: Optional[Dict[str, Any]] = None,
        news_override: Optional[Dict[str, Any]] = None,
        symbol: str = SYMBOL,
    ) -> Dict[str, Any]:
        """
        Evaluasi seluruh pending order milik scanner dan batalkan segera apabila tidak lagi valid.
        Order manual milik user diabaikan sepenuhnya (read-only).
        """
        conf = self.cfg
        magic = int(conf.get("SCANNER_MAGIC_NUMBER", 20260991))
        snap = snapshot or {}
        sc = scan_result or {}
        frames = snap.get("frames") if isinstance(snap.get("frames"), dict) else None
        live_price = float(sc.get("current_price") or snap.get("current_price") or 0.0)
        bid = float(sc.get("bid") or snap.get("bid") or 0.0)
        ask = float(sc.get("ask") or snap.get("ask") or 0.0)
        spread = float(sc.get("spread") or snap.get("spread") or 0.0)

        with MT5_IPC_LOCK:
            raw_orders = list(self.mt5.orders_get(symbol=symbol) or self.mt5.orders_get() or ())

        scanner_orders = [o for o in raw_orders if is_scanner_owned_item(o, magic)]
        manual_orders = [o for o in raw_orders if not is_scanner_owned_item(o, magic)]

        cancelled_actions: List[Dict[str, Any]] = []
        filled_during_audit: List[Dict[str, Any]] = []

        if not scanner_orders:
            return {
                "scanner_orders_count": 0,
                "manual_orders_count": len(manual_orders),
                "cancelled": cancelled_actions,
                "filled": filled_during_audit,
            }

        # Peta zona live dari scan_result
        zones_by_id: Dict[str, Dict[str, Any]] = {}
        for z in (sc.get("active_zones") or []) + (sc.get("radar_zones") or []) + (sc.get("rejected_zones") or []):
            if isinstance(z, dict) and z.get("zone_id"):
                zones_by_id[str(z["zone_id"])] = z

        htf_ctx = sc.get("htf_context") or {}
        h4_bias = str(htf_ctx.get("H4") or sc.get("htf_bias") or "").upper()
        h1_bias = str(htf_ctx.get("H1") or sc.get("h1_structure") or "").upper()
        m30_status = str(htf_ctx.get("m30_status") or sc.get("m30_status") or "CONFIRMED").upper()
        has_htf_conflict = bool(htf_ctx.get("has_conflict"))

        news_ok, news_code, news_msg = evaluate_news_hard_filter(
            news_override=news_override, snapshot_news=snap.get("news"), cfg=conf
        )
        portfolio = evaluate_portfolio_risk_and_daily_limits(
            mt5_api=self.mt5, state_mgr=self.state_mgr, symbol=symbol, cfg=conf
        )

        setups_map = self.state_mgr.get_all_setups()
        setup_by_order_ticket: Dict[int, Dict[str, Any]] = {}
        for sid, rec in setups_map.items():
            ot = int(rec.get("order_ticket") or 0)
            if ot > 0:
                setup_by_order_ticket[ot] = rec

        now_epoch = time.time()
        max_ttl_sec = float(conf.get("DEFAULT_PENDING_EXPIRATION_HOURS", 8)) * 3600.0

        for ord_obj in scanner_orders:
            tkt = int(getattr(ord_obj, "ticket", 0) or 0)
            ord_type_code = int(getattr(ord_obj, "type", 2) or 2)
            direction = "BUY" if ord_type_code in (2, 4, 6) else "SELL"
            rec = setup_by_order_ticket.get(tkt) or {}
            sid = str(rec.get("setup_id") or "")
            zid = str(rec.get("zone_id") or "")

            cancel_code = ""
            cancel_reason = ""

            # 1. Cek High-impact news blackout
            if not news_ok and news_code == REASON_NEWS_BLACKOUT:
                cancel_code = REASON_NEWS_BLACKOUT
                cancel_reason = news_msg

            # 2. Cek Daily loss limit tercapai
            elif portfolio.get("daily_loss_limit_reached"):
                cancel_code = REASON_DAILY_LOSS_LIMIT
                cancel_reason = "Daily loss limit tercapai; membatalkan pending order scanner"

            # 3. Cek Expiration waktu setup
            elif conf.get("AUTO_CANCEL_EXPIRED_SETUP", True):
                created_ep = float(rec.get("created_epoch") or 0.0)
                ord_exp = int(getattr(ord_obj, "time_expiration", 0) or 0)
                if (created_ep > 0 and (now_epoch - created_ep) > max_ttl_sec) or (0 < ord_exp <= int(now_epoch)):
                    cancel_code = REASON_ZONE_EXPIRED
                    cancel_reason = f"Pending order #{tkt} melewati batas waktu kedaluwarsa"

            # 4. Cek Validitas Zona (Invalidated / Mitigated / Broken)
            if not cancel_code and conf.get("AUTO_CANCEL_INVALID_ZONE", True) and zid:
                z_live = zones_by_id.get(zid)
                if z_live is not None:
                    z_ok, _, z_code, z_msg = validate_zone_quality_layer(
                        z_live, frames=frames, current_price=live_price,
                        bid=bid if bid > 0 else None, ask=ask if ask > 0 else None,
                        spread=spread, cfg=conf,
                    )
                    if not z_ok and z_code in (REASON_ZONE_INVALIDATED, REASON_ZONE_EXPIRED, "ZONE_MITIGATED"):
                        cancel_code = z_code
                        cancel_reason = z_msg

            # 5. Cek Perubahan Struktur HTF (EARLY_PENDING membutuhkan H4 & H1 tetap mendukung)
            if not cancel_code and h4_bias in ("BULLISH", "BEARISH", "NETRAL") and sc:
                dir_ok, d_code, d_msg = evaluate_directional_alignment(
                    direction=direction,
                    entry_mode="EARLY_PENDING",
                    htf_bias=h4_bias,
                    h1_bias=h1_bias,
                    m30_status=m30_status,
                    has_explicit_htf_conflict=has_htf_conflict,
                    cfg=conf,
                )
                if not dir_ok:
                    cancel_code = REASON_HTF_CONFLICT if d_code == REASON_HTF_CONFLICT else d_code
                    cancel_reason = d_msg

            if cancel_code:
                c_res = self.engine.cancel_scanner_pending_order(tkt, symbol=symbol, reason=cancel_code)
                if c_res.get("already_filled"):
                    pos_obj = c_res.get("position")
                    if sid:
                        self.state_mgr.update_setup_state(
                            sid,
                            STATE_POSITION_ACTIVE,
                            reason_code="FILLED_DURING_CANCEL",
                            reason_text="Pending order terisi saat proses pembatalan berlangsung",
                            position_ticket=tkt,
                            actual_entry=float(getattr(pos_obj, "price_open", 0.0) or 0.0),
                            actual_lot=float(getattr(pos_obj, "volume", 0.0) or 0.0),
                        )
                        self.state_mgr.record_trade_opened(sid, tkt)
                    filled_during_audit.append({"ticket": tkt, "setup_id": sid, "reason": c_res.get("reason")})
                elif c_res.get("cancelled"):
                    new_st = STATE_EXPIRED if cancel_code == REASON_ZONE_EXPIRED else STATE_INVALIDATED
                    if sid:
                        self.state_mgr.update_setup_state(
                            sid, new_st, reason_code=cancel_code, reason_text=cancel_reason
                        )
                    cancelled_actions.append({
                        "ticket": tkt,
                        "setup_id": sid,
                        "zone_id": zid,
                        "reason_code": cancel_code,
                        "reason_text": cancel_reason,
                    })

        return {
            "scanner_orders_count": len(scanner_orders),
            "manual_orders_count": len(manual_orders),
            "cancelled": cancelled_actions,
            "filled": filled_during_audit,
        }

    def switch_pending_to_market_safely(
        self,
        setup_id: str,
        order_ticket: int,
        revalidate_fn: Callable[[], Dict[str, Any]],
        symbol: str = SYMBOL,
        trading_mode: str = "FULL_AUTO",
    ) -> Dict[str, Any]:
        """
        Prosedur aman perpindahan dari Pending Order ke Market Order (Section 14 & Scenario 39):
        1. Batalkan pending order terlebih dahulu.
        2. Verifikasi apakah pending order sudah terisi selama proses pembatalan.
        3. Lakukan rekonsiliasi posisi & order MT5.
        4. Jika sudah terisi -> JANGAN kirim market order tambahan (cegah duplicate exposure!).
        5. Jika benar-benar batal & 0 eksposur -> validasi ulang setup -> eksekusi market order.
        """
        cancel_res = self.engine.cancel_scanner_pending_order(
            order_ticket, symbol=symbol, reason="SWITCH_TO_MARKET"
        )
        if cancel_res.get("already_filled"):
            pos_obj = cancel_res.get("position")
            self.state_mgr.update_setup_state(
                setup_id,
                STATE_POSITION_ACTIVE,
                reason_code="PENDING_ALREADY_FILLED",
                reason_text="Pending order sudah terisi; membatalkan pengiriman market order agar tidak duplikat",
                position_ticket=int(order_ticket),
                actual_entry=float(getattr(pos_obj, "price_open", 0.0) or 0.0),
                actual_lot=float(getattr(pos_obj, "volume", 0.0) or 0.0),
            )
            self.state_mgr.record_trade_opened(setup_id, int(order_ticket))
            return {
                "switched": False,
                "already_filled": True,
                "executed": False,
                "state": STATE_POSITION_ACTIVE,
                "position_ticket": int(order_ticket),
                "reason_code": REASON_DUPLICATE_SETUP,
                "reason_text": "Pending order terisi saat pembatalan; market order dibatalkan (anti-duplicate)",
            }

        if not cancel_res.get("cancelled"):
            return {
                "switched": False,
                "already_filled": False,
                "executed": False,
                "state": STATE_ORDER_PLACED,
                "reason_code": "CANCEL_PENDING_FAILED",
                "reason_text": f"Gagal membatalkan pending order #{order_ticket}; market order ditahan",
            }

        # Bersihkan flag executed_setup_ids sementara untuk setup yang sama HANYA karena order pending-nya resmi dibatalkan tanpa fill
        with self.state_mgr.lock:
            exec_ids = self.state_mgr.state.get("executed_setup_ids") or []
            if setup_id in exec_ids:
                exec_ids.remove(setup_id)
            self.state_mgr.update_setup_state(
                setup_id, "ZONE_TOUCHED", reason_code="PENDING_CANCELLED_FOR_SWITCH", save_now=True
            )

        # Rekonsiliasi ulang terhadap MT5 untuk memastikan 0 posisi aktif
        with MT5_IPC_LOCK:
            positions = list(self.mt5.positions_get(symbol=symbol) or ())
            orders = list(self.mt5.orders_get(symbol=symbol) or ())
        self.state_mgr.reconcile_with_mt5(positions, orders, magic_number=int(self.cfg.get("SCANNER_MAGIC_NUMBER", 20260991)))

        cur_rec = self.state_mgr.get_setup(setup_id) or {}
        if cur_rec.get("state") in (STATE_POSITION_ACTIVE, STATE_ORDER_PLACED):
            return {
                "switched": False,
                "already_filled": True,
                "executed": False,
                "state": cur_rec.get("state"),
                "reason_code": REASON_DUPLICATE_SETUP,
                "reason_text": "Rekonsiliasi mendeteksi eksposur sudah aktif",
            }

        fresh_val = revalidate_fn()
        if not fresh_val.get("can_execute"):
            return {
                "switched": False,
                "already_filled": False,
                "executed": False,
                "state": fresh_val.get("state", "BLOCKED"),
                "reason_code": fresh_val.get("reason_code", "REVALIDATION_FAILED"),
                "reason_text": fresh_val.get("reason_text", "Revalidasi gagal pasca-pembatalan pending"),
            }

        exec_res = self.engine.execute_validated_setup(fresh_val, trading_mode=trading_mode)
        exec_res["switched"] = bool(exec_res.get("executed"))
        return exec_res

    def cancel_all_scanner_pending_orders(self, symbol: str = SYMBOL, reason: str = "AUTO_OFF") -> List[Dict[str, Any]]:
        """Batalkan seluruh pending order milik scanner (untuk `auto off` dan `auto emergency`)."""
        magic = int(self.cfg.get("SCANNER_MAGIC_NUMBER", 20260991))
        with MT5_IPC_LOCK:
            orders = list(self.mt5.orders_get(symbol=symbol) or self.mt5.orders_get() or ())
        results = []
        for o in orders:
            if not is_scanner_owned_item(o, magic):
                continue
            tkt = int(getattr(o, "ticket", 0) or 0)
            res = self.engine.cancel_scanner_pending_order(tkt, symbol=symbol, reason=reason)
            res["ticket"] = tkt
            results.append(res)
        return results
