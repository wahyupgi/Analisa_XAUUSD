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

from modules.config import SYMBOL, get_auto_trading_config, is_grade_a_wait_confirm_zone
from modules.execution_engine import MT5ExecutionEngine, is_scanner_owned_item
from modules.execution_state import (
    ExecutionStateManager,
    REASON_DAILY_LOSS_LIMIT,
    REASON_DUPLICATE_SETUP,
    REASON_GRADE_A_WAIT_CONFIRM,
    REASON_HTF_CONFLICT,
    REASON_NEWS_BLACKOUT,
    REASON_ZONE_EXPIRED,
    REASON_ZONE_INVALIDATED,
    REASON_ZONE_LOW_SCORE,
    REASON_ZONE_MITIGATED,
    REASON_ZONE_TOO_FAR,
    STATE_BLOCKED,
    STATE_EXPIRED,
    STATE_INVALIDATED,
    STATE_ORDER_PLACED,
    STATE_POSITION_ACTIVE,
    STATE_RADAR,
    make_setup_id,
)
from modules.mt5_conn import MT5_IPC_LOCK
from modules.risk_manager import evaluate_portfolio_risk_and_daily_limits
from modules.trade_validator import (
    evaluate_directional_alignment,
    evaluate_news_hard_filter,
    select_candidate_zones,
    validate_zone_quality_layer,
)


def _find_matching_live_zone(
    direction: str,
    ord_price: float,
    live_zones: List[Dict[str, Any]],
    tol: float = 1.5,
) -> Optional[Dict[str, Any]]:
    """Cari zona aktif/radar yang cocok dengan arah & harga pending order scanner."""
    if ord_price <= 0 or not live_zones:
        return None
    dir_u = str(direction or "").upper()
    best_zone: Optional[Dict[str, Any]] = None
    best_key = None
    for z in live_zones:
        if not isinstance(z, dict):
            continue
        z_dir = str(z.get("direction") or "").upper()
        z_type = str(z.get("zone_type") or z.get("type") or "").upper()
        if dir_u == "BUY":
            if not (z_dir == "BUY" or z_type in ("DEMAND", "SUPPORT")):
                continue
        elif dir_u == "SELL":
            if not (z_dir == "SELL" or z_type in ("SUPPLY", "RESISTANCE")):
                continue
        else:
            continue
        z_lo = float(z.get("price_low") or z.get("zone_low") or z.get("low") or 0.0)
        z_hi = float(z.get("price_high") or z.get("zone_high") or z.get("high") or 0.0)
        if z_lo <= 0 or z_hi <= 0:
            continue
        if z_lo > z_hi:
            z_lo, z_hi = z_hi, z_lo
        if (z_lo - tol) <= ord_price <= (z_hi + tol):
            mid = (z_lo + z_hi) / 2.0
            key = (abs(ord_price - mid), -float(z.get("score") or 0.0))
            if best_key is None or key < best_key:
                best_key = key
                best_zone = z
    return best_zone


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
        snr_result: Optional[Dict[str, Any]] = None,
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

        # Peta zona live dari scan_result (SMC) dan snr_result (Hybrid SNR)
        zones_by_id: Dict[str, Dict[str, Any]] = {}
        active_live_zones: List[Dict[str, Any]] = []
        seen_live_ids = set()
        for z in (sc.get("active_zones") or []) + (sc.get("radar_zones") or []):
            if isinstance(z, dict):
                zid_k = str(z.get("zone_id") or "").strip()
                if zid_k:
                    zones_by_id[zid_k] = z
                    if zid_k not in seen_live_ids:
                        seen_live_ids.add(zid_k)
                        active_live_zones.append(z)
                else:
                    active_live_zones.append(z)
        if snr_result:
            for sz in select_candidate_zones({}, None, snr_result=snr_result):
                if isinstance(sz, dict):
                    sz_id = str(sz.get("zone_id") or "").strip()
                    if sz_id and sz_id not in zones_by_id:
                        zones_by_id[sz_id] = sz
                    if sz_id and sz_id not in seen_live_ids:
                        seen_live_ids.add(sz_id)
                        active_live_zones.append(sz)
        rejected_live_zones = [z for z in (sc.get("rejected_zones") or []) if isinstance(z, dict)]
        for z in rejected_live_zones:
            if z.get("zone_id"):
                zones_by_id[str(z["zone_id"])] = z
        has_live_scan_zones = bool(active_live_zones or rejected_live_zones)

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
        zone_match_tol = max(1.5, float(conf.get("PRICE_ENTRY_TOLERANCE_USD", 1.5) or 1.5))

        for ord_obj in scanner_orders:
            tkt = int(getattr(ord_obj, "ticket", 0) or 0)
            ord_type_code = int(getattr(ord_obj, "type", 2) or 2)
            direction = "BUY" if ord_type_code in (2, 4, 6) else "SELL"
            ord_price = float(getattr(ord_obj, "price_open", 0.0) or 0.0)
            ord_sl = float(getattr(ord_obj, "sl", 0.0) or 0.0)
            ord_tp = float(getattr(ord_obj, "tp", 0.0) or 0.0)
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

            # 3. Cek Expiration waktu pending order (wajib dihitung sejak ORDER_PLACED, BUKAN sejak zona pertama masuk radar kemarin!)
            elif conf.get("AUTO_CANCEL_EXPIRED_SETUP", True):
                placed_ep = float(rec.get("order_placed_epoch") or 0.0)
                ord_exp = int(getattr(ord_obj, "time_expiration", 0) or 0)
                tick_ref_ts = 0
                try:
                    with MT5_IPC_LOCK:
                        _tk = self.mt5.symbol_info_tick(symbol)
                    tick_ref_ts = int(getattr(_tk, "time", 0) or 0) if _tk is not None else 0
                except Exception:
                    tick_ref_ts = 0
                exp_expired = bool(0 < ord_exp <= (tick_ref_ts if tick_ref_ts > 0 else int(now_epoch))) and (
                    placed_ep <= 0.0 or (now_epoch - placed_ep) > 60.0
                )
                if (placed_ep > 0.0 and (now_epoch - placed_ep) > max_ttl_sec) or exp_expired:
                    cancel_code = REASON_ZONE_EXPIRED
                    cancel_reason = f"Pending order #{tkt} melewati batas waktu kedaluwarsa"

            # 3b. Cek Invalidation berbasis harga live (SL pending order tertembus harga live)
            if not cancel_code and conf.get("AUTO_CANCEL_INVALID_ZONE", True) and live_price > 0:
                if direction == "SELL" and ord_sl > 0 and live_price >= ord_sl:
                    cancel_code = REASON_ZONE_INVALIDATED
                    cancel_reason = f"Live price ({live_price:.2f}) menembus SL pending order scanner #{tkt} ({ord_sl:.2f})"
                elif direction == "BUY" and ord_sl > 0 and live_price <= ord_sl:
                    cancel_code = REASON_ZONE_INVALIDATED
                    cancel_reason = f"Live price ({live_price:.2f}) menembus SL pending order scanner #{tkt} ({ord_sl:.2f})"

            # 3c. Cek Jarak Terlalu Jauh dari Harga Live (auto-cancel pending order yang sudah jauh dari harga)
            if not cancel_code and conf.get("AUTO_CANCEL_FAR_PENDING", True) and live_price > 0 and ord_price > 0:
                max_pend_dist = float(conf.get("PENDING_MAX_DISTANCE_USD", 40.0) or 0.0)
                cancel_hyst = max(0.0, float(conf.get("PENDING_CANCEL_HYSTERESIS_USD", 3.0) or 0.0))
                dist_usd = abs(ord_price - live_price)
                if max_pend_dist > 0.0 and dist_usd > (max_pend_dist + cancel_hyst):
                    cancel_code = REASON_ZONE_TOO_FAR
                    cancel_reason = (
                        f"Pending order #{tkt} ({direction} @ {ord_price:.2f}) sudah terlalu jauh dari harga live "
                        f"({live_price:.2f}, jarak {dist_usd:.2f} > batas {max_pend_dist + cancel_hyst:.2f} USD)"
                    )

            # 4. Cek Validitas Zona (termasuk order ADOPTED/orphan tanpa zone_id atau zona yang sudah hilang/jauh/invalid)
            if not cancel_code and conf.get("AUTO_CANCEL_INVALID_ZONE", True):
                z_live = zones_by_id.get(zid) if zid else None
                if z_live is None and has_live_scan_zones and ord_price > 0:
                    matched_z = _find_matching_live_zone(direction, ord_price, active_live_zones, tol=zone_match_tol)
                    if matched_z is not None:
                        z_live = matched_z
                        new_zid = str(matched_z.get("zone_id") or "").strip()
                        if sid and new_zid and new_zid != zid:
                            self.state_mgr.update_setup_state(
                                sid,
                                STATE_ORDER_PLACED,
                                reason_code=str(rec.get("reason_code") or "RELINKED_SCANNER_ORDER"),
                                zone_id=new_zid,
                            )
                            zid = new_zid
                    else:
                        cancel_code = REASON_ZONE_EXPIRED
                        if zid:
                            cancel_reason = f"Zona asal ({zid}) untuk pending order #{tkt} tidak lagi aktif pada hasil scan terbaru"
                        else:
                            cancel_reason = (
                                f"Pending order scanner #{tkt} ({direction} @ {ord_price:.2f}) "
                                "tidak cocok dengan zona aktif/radar manapun (stale/adopted orphan)"
                            )
                if not cancel_code and z_live is not None:
                    z_ok, _, z_code, z_msg = validate_zone_quality_layer(
                        z_live, frames=frames, current_price=live_price,
                        bid=bid if bid > 0 else None, ask=ask if ask > 0 else None,
                        spread=spread, cfg=conf,
                    )
                    if not z_ok and z_code in (
                        REASON_ZONE_INVALIDATED,
                        REASON_ZONE_EXPIRED,
                        REASON_ZONE_MITIGATED,
                        REASON_ZONE_TOO_FAR,
                        REASON_ZONE_LOW_SCORE,
                    ):
                        cancel_code = z_code
                        cancel_reason = z_msg

                # 4b. Jika zona berstatus Grade A / A+ / 80+ / 90+ dan kebijakan tunggu konfirmasi M5 aktif ->
                #     batalkan pending limit 0.01 lot agar setup bebas dieksekusi 0.02 / 0.03 lot saat harga menjemput + konfirmasi!
                if (
                    not cancel_code
                    and z_live is not None
                    and bool(conf.get("AUTO_CANCEL_GRADE_A_PENDING_LIMIT", True))
                    and is_grade_a_wait_confirm_zone(z_live, cfg=conf)
                ):
                    cancel_code = REASON_GRADE_A_WAIT_CONFIRM
                    cancel_reason = (
                        f"Zona {zid or z_live.get('zone_id')} berstatus Grade {z_live.get('grade', 'A')} "
                        f"({float(z_live.get('score') or 0.0):.1f}) — pending limit dibatalkan agar menunggu "
                        f"harga menjemput + konfirmasi M5 (0.02/0.03 lot)"
                    )

            # 5. Cek Perubahan Struktur HTF (EARLY_PENDING standar membutuhkan H4 & H1 mendukung;
            #    kecuali order Mid-Zone 50% Limit yang memang dipasang khusus untuk kondisi konflik/sideways)
            is_midzone_ord = (str(rec.get("sub_mode") or "") == "CONFLICT_MIDZONE_LIMIT")
            if not cancel_code and (not is_midzone_ord) and h4_bias in ("BULLISH", "BEARISH", "NETRAL") and sc:
                dir_ok, d_code, d_msg = evaluate_directional_alignment(
                    direction=direction,
                    entry_mode="EARLY_PENDING",
                    htf_bias=h4_bias,
                    h1_bias=h1_bias,
                    m30_status=m30_status,
                    has_explicit_htf_conflict=has_htf_conflict,
                    allow_conflict_midzone=False,
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
                    if cancel_code in (REASON_ZONE_TOO_FAR, REASON_GRADE_A_WAIT_CONFIRM):
                        new_st = STATE_RADAR
                    elif cancel_code == REASON_NEWS_BLACKOUT:
                        new_st = STATE_BLOCKED
                    elif cancel_code == REASON_ZONE_EXPIRED:
                        new_st = STATE_EXPIRED
                    else:
                        new_st = STATE_INVALIDATED
                    if sid:
                        with self.state_mgr.lock:
                            cur_rec_c = self.state_mgr.get_setup(sid) or {}
                            if not cur_rec_c.get("position_ticket") and not cur_rec_c.get("stopped_out"):
                                exec_ids = self.state_mgr.state.get("executed_setup_ids") or []
                                if sid in exec_ids:
                                    exec_ids.remove(sid)
                                if zid:
                                    canon_sid = make_setup_id(zid, direction, symbol=symbol)
                                    if canon_sid in exec_ids:
                                        exec_ids.remove(canon_sid)
                            self.state_mgr.update_setup_state(
                                sid,
                                new_st,
                                reason_code=cancel_code,
                                reason_text=cancel_reason,
                                order_ticket=0,
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
