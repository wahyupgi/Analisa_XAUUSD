# -*- coding: utf-8 -*-
"""
modules/smart_monitor_report.py
Presentation layer Smart Monitor / Unified Monitor: konteks smart monitor dan formatter laporan
(`pantau` ringkas & detail) serta blok Account Monitor.

Dipindahkan apa adanya dari modules/unified_monitor.py (structural refactoring, tanpa perubahan logika).
Modul ini tidak bergantung pada unified_monitor (dependency satu arah); unified_monitor
mengimpor kembali seluruh nama publiknya agar caller lama tetap kompatibel.
"""
from modules.snr.data_provider import safe_float
from modules.utils import format_wib_timestamp
from modules.zone_scanner import format_smc_zone_lines


def format_account_monitor_section(acc_data):
    """
    Menyusun blok `### C. ACCOUNT MONITOR` (STRICTLY READ-ONLY):
    - Balance, Equity, Margin, Free Margin, Margin Level (%)
    - Posisi aktif & Exposure Lot (Buy lot vs Sell lot)
    - Pending order
    - Profit/loss berjalan
    - Posisi atau pending order tanpa SL (Peringatan Keras)
    - Akumulasi total risiko akun (berdasarkan SL atau UNBOUNDED)
    - Informasi spread
    - Peringatan risiko penting
    """
    acc = acc_data or {}
    spread_str = f"{safe_float(acc.get('spread')):.2f} pts" if acc.get("spread") is not None else "N/A"

    if not acc.get("available", False):
        lines = [
            "### C. ACCOUNT MONITOR (STRICTLY READ-ONLY)",
            "BALANCE / EQUITY: N/A (MT5 Offline / Mock)",
            "OPEN POSITIONS  : N/A",
            "PENDING ORDERS  : N/A",
            "FLOATING P/L    : N/A",
            "TOTAL RISK (SL) : N/A",
            "WITHOUT SL      : N/A",
            f"SPREAD          : {spread_str}",
            "RISK ALERTS     : Data posisi akun MT5 belum tersedia.",
            "AUDIT MODE      : STRICTLY READ-ONLY (Tidak ada modifikasi order otomatis)",
        ]
        return "\n".join(lines)

    warns = acc.get("risk_warnings") or []
    warn_str = "; ".join(warns) if warns else "AMAN (Seluruh posisi/order terproteksi SL)"

    bal = safe_float(acc.get("balance", 0.0))
    eq = safe_float(acc.get("equity", 0.0))
    cur = acc.get("currency", "USD")
    m_lvl = safe_float(acc.get("margin_level", 0.0))
    m_val = safe_float(acc.get("margin", 0.0))
    fm_val = safe_float(acc.get("free_margin", 0.0))
    b_lots = safe_float(acc.get("buy_lots", 0.0))
    s_lots = safe_float(acc.get("sell_lots", 0.0))
    net_lots = safe_float(acc.get("net_lot_exposure", 0.0))

    lines = [
        "### C. ACCOUNT MONITOR (STRICTLY READ-ONLY)",
        f"BALANCE / EQUITY: {bal:,.2f} {cur} / {eq:,.2f} {cur} | MARGIN LEVEL: {m_lvl:.2f}% (Margin: {m_val:,.2f} {cur} | Free: {fm_val:,.2f} {cur})",
        f"OPEN POSITIONS  : {acc.get('open_positions', 0)} (Buy: {b_lots:.2f} lot | Sell: {s_lots:.2f} lot | Net: {net_lots:+.2f} lot)",
        f"PENDING ORDERS  : {acc.get('pending_orders', 0)}",
        f"FLOATING P/L    : {acc.get('floating_pnl', 0.0):+,.2f} {cur}",
        f"TOTAL RISK (SL) : {acc.get('total_risk_status', 'N/A')}",
        f"WITHOUT SL      : Position={acc.get('position_without_sl', 0)} | Pending={acc.get('pending_without_sl', 0)}",
        f"SPREAD          : {spread_str}",
        f"RISK ALERTS     : {warn_str}",
        "AUDIT MODE      : STRICTLY READ-ONLY (Dilarang auto-close / auto-modify / auto-trade)",
    ]
    return "\n".join(lines)
    return "\n".join(lines)


def _as_monitor_zone(zone):
    if isinstance(zone, dict):
        return zone
    to_dict = getattr(zone, "to_dict", None)
    if callable(to_dict):
        converted = to_dict()
        return converted if isinstance(converted, dict) else {}
    return {}


def _calc_zone_touch(z, p, stages_dict, trk_rec, life_st):
    z_lo = safe_float(z.get("price_low"), 0.0)
    z_hi = safe_float(z.get("price_high"), 0.0)
    if not z:
        return "NO_ZONE", "NO SMC SETUP"
    if z.get("is_valid") is False or life_st in ("INVALIDATED", "BROKEN") or z.get("status") == "INVALIDATED":
        return "INVALIDATED", "INVALIDATED"
    inv_p = safe_float(z.get("invalidation_price"), 0.0)
    z_t = str(z.get("zone_type", "")).upper()
    if inv_p > 0 and p > 0:
        if z_t == "DEMAND" and p < inv_p:
            return "INVALIDATED", "INVALIDATED"
        if z_t == "SUPPLY" and p > inv_p:
            return "INVALIDATED", "INVALIDATED"
    if z_lo > 0 and z_hi > 0 and p > 0 and (z_lo <= p <= z_hi):
        return "IN_ZONE", "IN ZONE"
    t_stage = (stages_dict.get("zone_touch") if isinstance(stages_dict, dict) else {}) or {}
    t_stage_st = str(t_stage.get("status") if isinstance(t_stage, dict) else t_stage or "").upper()
    t_cnt = int(z.get("touch_count", 0))
    app = str(z.get("approach", "")).upper()
    has_touched = (
        t_stage_st in ("CONFIRMED", "PASS", "PASSED")
        or app in ("TOUCHED", "ZONE_TOUCHED")
        or life_st in ("ZONE_TOUCHED", "TOUCHED", "CONFIRMED")
        or t_cnt > 0
    )
    if not has_touched and trk_rec:
        hist = trk_rec.get("history") or []
        has_touched = any(str(h.get("status", "")).upper() in ("ZONE_TOUCHED", "TOUCHED", "CONFIRMED") for h in hist)
        if not has_touched and str(trk_rec.get("touch_status", "")).upper() in ("IN_ZONE", "TOUCHED_PREVIOUSLY"):
            has_touched = True
    if has_touched:
        return "TOUCHED_PREVIOUSLY", "PREVIOUSLY TOUCHED"
    return "NOT_TOUCHED", "NOT TOUCHED"


def _smart_monitor_context(cycle_result):
    cycle = cycle_result if isinstance(cycle_result, dict) else {}
    snapshot = cycle.get("snapshot") if isinstance(cycle.get("snapshot"), dict) else {}
    smc = cycle.get("smc") if isinstance(cycle.get("smc"), dict) else {}
    snr = cycle.get("snr") if isinstance(cycle.get("snr"), dict) else {}
    scan = smc.get("scan_result") if isinstance(smc.get("scan_result"), dict) else {}
    decision = smc.get("decision") if isinstance(smc.get("decision"), dict) else {}
    summary = smc.get("summary") if isinstance(smc.get("summary"), dict) else {}
    snr_scan = snr.get("scan_result") if isinstance(snr.get("scan_result"), dict) else {}
    snr_decision = snr.get("decision") if isinstance(snr.get("decision"), dict) else {}

    zone = _as_monitor_zone(decision.get("zone"))
    if not zone or safe_float(zone.get("price_low", zone.get("low")), 0.0) <= 0:
        priority = decision.get("radar_priority") or scan.get("radar_priority") or {}
        smc_candidates = [
            decision.get("primary_zone"),
            priority.get("primary_directional_zone"),
            priority.get("primary_zone"),
            decision.get("best_radar_zone"),
            decision.get("nearest_valid_zone"),
            decision.get("next_radar_zone"),
            scan.get("primary_zone"),
            scan.get("nearest_zone"),
            (decision.get("active_setup") or {}).get("zone"),
            (decision.get("early_setup") or {}).get("best_setup"),
        ]
        for cand in smc_candidates:
            cz = _as_monitor_zone(cand)
            if cz and safe_float(cz.get("price_low", cz.get("low", cz.get("zone_low"))), 0.0) > 0:
                zone = dict(cz)
                if "price_low" not in zone and "low" in zone:
                    zone["price_low"] = zone["low"]
                elif "price_low" not in zone and "zone_low" in zone:
                    zone["price_low"] = zone["zone_low"]
                if "price_high" not in zone and "high" in zone:
                    zone["price_high"] = zone["high"]
                elif "price_high" not in zone and "zone_high" in zone:
                    zone["price_high"] = zone["zone_high"]
                if "zone_type" not in zone and "type" in zone:
                    zone["zone_type"] = zone["type"]
                break

    pending = decision.get("early_setup") or {}
    if not isinstance(pending, dict):
        pending = {}
    plan = pending.get("best_setup") or {}
    if not isinstance(plan, dict):
        plan = {}

    price = safe_float(snapshot.get("snapshot_price", snapshot.get("bid")), 0.0)
    lo = safe_float(zone.get("price_low", zone.get("low")), 0.0)
    hi = safe_float(zone.get("price_high", zone.get("high")), 0.0)
    distance = 0.0 if lo <= price <= hi and price > 0 else (
        lo - price if price > 0 and lo > price else (price - hi if price > hi else None)
    )
    zone_type = str(zone.get("zone_type") or zone.get("type") or "").upper()
    direction = str(decision.get("direction") or summary.get("direction") or "").upper()
    if direction not in ("BUY", "SELL"):
        direction = "BUY" if zone_type in ("DEMAND", "SUPPORT") else ("SELL" if zone_type in ("SUPPLY", "RESISTANCE") else "WAIT")

    # Structured Setup Identity for Monitored Active Setup (Single Source of Truth)
    z_id = str(zone.get("zone_id") or "N/A")
    setup_type = str(decision.get("setup_type") or zone.get("setup_type") or "TREND_PULLBACK")
    now_str = format_wib_timestamp(default_now=True)
    active_setup = dict(decision.get("active_setup") or {})
    active_setup.update({
        "setup_id": active_setup.get("setup_id") or (f"{z_id}:{direction}:{setup_type}" if z_id != "N/A" else "N/A"),
        "zone_id": z_id,
        "direction": direction,
        "zone_range": (lo, hi),
        "setup_type": setup_type,
        "timeframe": str(zone.get("timeframe") or "H1"),
        "grade": str(zone.get("grade") or "N/A"),
        "score": safe_float(zone.get("score"), 0.0),
        "created_at": zone.get("created_at") or zone.get("first_detected") or now_str,
        "last_updated": zone.get("last_updated") or now_str,
        "lifecycle_status": str(zone.get("lifecycle_status") or zone.get("status") or "RADAR").upper(),
    })

    snr_bias = str((snr_scan.get("market_bias") or {}).get("overall_bias") or "").upper()
    snr_zones = snr_scan.get("ranked_zones") or {}
    snr_zone = (
        snr_decision.get("target_zone")
        or (snr_scan.get("entry_validation") or {}).get("target_zone")
        or snr_decision.get("trend_entry_zone")
        or {}
    )
    snr_zone_direction = (
        str(snr_decision.get("signal")).upper() if snr_decision.get("signal") in ("BUY", "SELL")
        else "SELL" if snr_zone.get("zone_type") == "RESISTANCE" or "R" in str(snr_zone.get("rank_label", "")).upper()
        else "BUY" if snr_zone.get("zone_type") == "SUPPORT" or "S" in str(snr_zone.get("rank_label", "")).upper() else "NONE"
    )
    if not snr_zone and snr_bias == "BEARISH":
        choices = snr_zones.get("active_resistances") or []
        if choices:
            snr_zone, snr_zone_direction = choices[0], "SELL"
    elif not snr_zone and snr_bias == "BULLISH":
        choices = snr_zones.get("active_supports") or []
        if choices:
            snr_zone, snr_zone_direction = choices[0], "BUY"
    if not snr_zone:
        choices = snr_zones.get("active_supports") or []
        snr_zone = choices[0] if choices else {}
        snr_zone_direction = "BUY" if snr_zone else "NONE"
    if not snr_zone:
        choices = snr_zones.get("active_resistances") or []
        snr_zone = choices[0] if choices else {}
        snr_zone_direction = "SELL" if snr_zone else "NONE"

    # If SMC zone was not found, fall back to SNR zone for active setup display
    if (not zone or safe_float(zone.get("price_low", zone.get("low")), 0.0) <= 0) and snr_zone:
        s_lo = safe_float(snr_zone.get("low", snr_zone.get("price_low")), 0.0)
        s_hi = safe_float(snr_zone.get("high", snr_zone.get("price_high")), 0.0)
        if s_lo > 0 and s_hi > 0:
            s_zt = str(snr_zone.get("zone_type") or ("RESISTANCE" if snr_zone_direction == "SELL" else "SUPPORT")).upper()
            zone = {
                "zone_id": snr_zone.get("zone_id", f"SNR_{snr_zone.get('rank_label', 'ZONE')}"),
                "timeframe": snr_zone.get("timeframe", "H1"),
                "zone_type": "SUPPLY" if s_zt == "RESISTANCE" else "DEMAND",
                "price_low": s_lo,
                "price_high": s_hi,
                "grade": snr_zone.get("grade", "A"),
                "score": safe_float(snr_zone.get("score"), 80.0),
                "status": "RADAR",
                "source": "SNR",
                "rank_label": snr_zone.get("rank_label", "SNR ZONE"),
            }
            lo, hi = s_lo, s_hi
            distance = 0.0 if lo <= price <= hi and price > 0 else (
                lo - price if price > 0 and lo > price else (price - hi if price > hi else None)
            )
            zone_type = zone["zone_type"]
            direction = "SELL" if s_zt == "RESISTANCE" else "BUY"
            z_id = str(zone.get("zone_id") or "N/A")
            active_setup.update({
                "setup_id": f"{z_id}:{direction}:{setup_type}",
                "zone_id": z_id,
                "direction": direction,
                "zone_range": (lo, hi),
                "timeframe": str(zone.get("timeframe") or "H1"),
                "grade": str(zone.get("grade") or "N/A"),
                "score": safe_float(zone.get("score"), 0.0),
            })

    def _zone_is_usable(candidate):
        if not isinstance(candidate, dict) or candidate.get("is_valid") is False or candidate.get("is_active") is False:
            return False
        status = str(candidate.get("lifecycle_status") or candidate.get("status") or "ACTIVE").upper()
        freshness_status = str(candidate.get("freshness") or candidate.get("freshness_status") or "FRESH").upper()
        return (
            status not in ("INVALIDATED", "EXPIRED", "MITIGATED", "BROKEN")
            and freshness_status not in ("INVALIDATED", "EXPIRED", "MITIGATED", "STALE", "INVALID")
        )

    smc_zone_usable = _zone_is_usable(zone)
    snr_zone_usable = _zone_is_usable(snr_zone)
    if smc.get("status") != "OK" or snr.get("status") != "OK":
        confluence = "UNAVAILABLE"
    elif not smc_zone_usable and not snr_zone_usable:
        confluence = "UNAVAILABLE"
    elif not smc_zone_usable or not snr_zone_usable or direction not in ("BUY", "SELL") or snr_zone_direction == "NONE":
        confluence = "PARTIAL"
    elif direction != snr_zone_direction:
        confluence = "CONFLICT"
    else:
        snr_lo = safe_float(snr_zone.get("low"), 0.0)
        snr_hi = safe_float(snr_zone.get("high"), 0.0)
        confluence = "ALIGNED" if max(lo, snr_lo) <= min(hi, snr_hi) else "PARTIAL"

    tracker = decision.get("zone_tracker")
    tracked = tracker.get_zone(str(zone.get("zone_id"))) if tracker and zone.get("zone_id") and hasattr(tracker, "get_zone") else None
    lifecycle = str((tracked or {}).get("lifecycle_status") or (tracked or {}).get("status") or active_setup["lifecycle_status"]).upper()
    if not lifecycle:
        lifecycle = str(zone.get("approach") or zone.get("status") or "RADAR").upper()
    state_map = {
        "RADAR": "IDLE",
        "READY": "IDLE",
        "DETECTED": "IDLE",
        "APPROACHING": "APPROACHING",
        "ZONE_TOUCHED": "IN_ZONE",
        "TOUCHED": "IN_ZONE",
        "CONFIRMED": "CONFIRMED",
        "INVALIDATED": "INVALIDATED",
        "EXPIRED": "EXPIRED",
    }
    monitor_state = state_map.get(lifecycle, lifecycle)
    if monitor_state == "IN_ZONE":
        monitor_state = "AWAITING_CONFIRMATION"

    stages = decision.get("m5_stages") or decision.get("stages") or {}
    touch_status, touch_display = _calc_zone_touch(zone, price, stages, tracked, lifecycle)

    trigger_stage = stages.get("structure_trigger") or {}
    trigger = str(
        (trigger_stage.get("status") if isinstance(trigger_stage, dict) else trigger_stage)
        or summary.get("choch_bos") or "WAIT"
    ).upper()

    snapshot_status = str(snapshot.get("snapshot_status") or snapshot.get("data_status") or "UNKNOWN").upper()
    freshness = str(snapshot.get("data_freshness") or (snapshot.get("data_freshness_info") or {}).get("status") or "UNKNOWN").upper()
    data_invalid = (
        snapshot_status in ("DATA_INCOMPLETE", "INVALID", "ERROR", "MARKET_CLOSED")
        or freshness in ("STALE", "INVALID", "NO DATA")
        or snapshot.get("data_ok") is False
        or price <= 0
        or smc.get("status") == "ERROR"
    )
    validation_price = safe_float(decision.get("snapshot_price") or decision.get("current_price"), 0.0)
    needs_revalidation = validation_price > 0 and abs(validation_price - price) > 1e-8
    if needs_revalidation:
        data_invalid = True

    # 1. Pending Plan evaluation (Section 6)
    p_status_raw = str(plan.get("status") or "WAIT").upper()
    if data_invalid:
        p_status = "WAIT"
    elif p_status_raw in ("NONE", "NOT ACTIVE"):
        p_status = "WAIT"
    elif p_status_raw == "READY":
        p_status = "READY (NOT PLACED)"
    else:
        p_status = p_status_raw

    p_zid = str(plan.get("zone_id") or "")
    p_lo = safe_float(plan.get("zone_low", plan.get("price_low")), 0.0)
    p_hi = safe_float(plan.get("zone_high", plan.get("price_high")), 0.0)
    p_dir = "BUY" if "BUY" in str(plan.get("order_type", "")).upper() else ("SELL" if "SELL" in str(plan.get("order_type", "")).upper() else "NONE")
    pending_setup = {
        "setup_id": f"{p_zid}:{p_dir}:{plan.get('setup_type', 'PENDING')}" if p_zid else "N/A",
        "zone_id": p_zid or "N/A",
        "direction": p_dir,
        "zone_range": (p_lo, p_hi),
        "setup_type": str(plan.get("setup_type") or "PENDING"),
        "timeframe": str(plan.get("timeframe") or "H1"),
        "status": p_status,
        "is_placed": False,
    }
    if p_status.startswith("READY") and p_zid and z_id != "N/A" and p_zid != z_id:
        p_label = f"[{p_lo:.2f}-{p_hi:.2f}]" if (p_lo > 0 and p_hi > 0) else f"[{p_zid}]"
        pending_display = f"{p_status} {p_label}"
    else:
        pending_display = p_status

    # 2. Confirmed Entry validation (Section 5)
    dec_status = str(decision.get("status") or "").upper()
    dec_dir = str(decision.get("direction") or "").upper()
    conf_plan = decision.get("plan") or {}

    is_valid_confirmed = (
        dec_status == "VALID"
        and dec_dir in ("BUY", "SELL")
        and not data_invalid
        and bool(conf_plan)
        and bool(zone.get("zone_id"))
        and (not conf_plan.get("zone_id") or conf_plan.get("zone_id") == zone.get("zone_id"))
        and lifecycle not in ("INVALIDATED", "EXPIRED")
        and zone.get("is_valid") is not False
    )

    c_entry = safe_float(conf_plan.get("entry_acuan", conf_plan.get("entry")), 0.0)
    is_entry_stale = False

    conf_zid = str(conf_plan.get("zone_id") or (decision.get("zone") or {}).get("zone_id") or "")
    conf_lo = safe_float((decision.get("zone") or {}).get("price_low"), 0.0)
    conf_hi = safe_float((decision.get("zone") or {}).get("price_high"), 0.0)
    confirmed_setup = {
        "setup_id": f"{conf_zid}:{dec_dir}:{decision.get('setup_type', 'CONFIRMED')}" if conf_zid else "N/A",
        "zone_id": conf_zid or "N/A",
        "direction": dec_dir if is_valid_confirmed else "NONE",
        "zone_range": (conf_lo, conf_hi),
        "setup_type": str(decision.get("setup_type") or "TREND_PULLBACK"),
        "timeframe": str((decision.get("zone") or {}).get("timeframe") or "H1"),
        "is_valid": is_valid_confirmed,
        "is_stale": is_entry_stale,
    }

    snr_sig = str(snr_decision.get("signal", "WAIT")).upper()
    snr_confirmed = (
        snr.get("status") == "OK" and snr_sig in ("BUY", "SELL")
        and snr_decision.get("entry") is not None and not data_invalid
    )
    if not is_valid_confirmed:
        if snr_confirmed:
            confirmed_status = f"{snr_sig} [SNR]"
        else:
            confirmed_status = "WAIT"
    else:
        method_tag = "[SMC + SNR]" if snr_sig == dec_dir else "[SMC]"
        if is_entry_stale:
            confirmed_status = f"{dec_dir} {method_tag} (ENTRY STALE)"
        else:
            confirmed_status = f"{dec_dir} {method_tag}"

    if confirmed_status != "WAIT" and conf_zid and z_id != "N/A" and conf_zid != z_id:
        c_label = f"[{conf_lo:.2f}-{conf_hi:.2f}]" if (conf_lo > 0 and conf_hi > 0) else f"[{conf_zid}]"
        confirmed_display = f"{confirmed_status} {c_label}"
    else:
        confirmed_display = confirmed_status

    if is_valid_confirmed and not is_entry_stale:
        monitor_state = "CONFIRMED"
    elif monitor_state == "CONFIRMED":
        monitor_state = "AWAITING_CONFIRMATION"

    # State, Reason, Next Action, and Final Signal
    if data_invalid:
        reason = "PRICE CHANGED; VALIDATION SNAPSHOT EXPIRED" if needs_revalidation else "DATA STALE / UNAVAILABLE"
        next_action, final_signal = "WAIT FRESH VALIDATION", "WAIT"
    elif is_valid_confirmed and not is_entry_stale:
        reason, next_action, final_signal = "M5 TRIGGER VALID", "CHECK ENTRY VALIDITY", confirmed_status
    elif snr_confirmed:
        reason = str(snr_decision.get("reason") or "SNR ENTRY VALID")
        next_action, final_signal = "CHECK ENTRY VALIDITY", confirmed_status
    elif is_entry_stale:
        reason = f"Entry stale: harga ({price:.2f}) telah bergerak melewati entry {c_entry:.2f}"
        next_action, final_signal = "WAIT PULLBACK", "WAIT (ENTRY STALE)"
    elif monitor_state == "INVALIDATED" or touch_status == "INVALIDATED":
        reason, next_action, final_signal = "ZONE INVALIDATED", "CANCEL SETUP", "WAIT"
    elif monitor_state == "EXPIRED":
        reason, next_action, final_signal = "SETUP EXPIRED", "WAIT NEW SETUP", "WAIT"
    elif touch_status in ("IN_ZONE", "TOUCHED_PREVIOUSLY") or monitor_state == "AWAITING_CONFIRMATION":
        reason = str((decision.get("reasons") or ["ZONE TOUCHED; M5 TRIGGER PENDING"])[0])
        next_action, final_signal = "CHECK VALIDATION", "WAIT"
    elif monitor_state == "APPROACHING":
        reason, next_action, final_signal = "APPROACHING ENTRY ZONE", "WAIT ZONE TOUCH", "WAIT"
    elif not zone:
        reason, next_action, final_signal = "NO SMC SETUP", "WAIT SETUP", "WAIT"
    else:
        reason, next_action, final_signal = "WAITING FOR ZONE PULLBACK", "WAIT APPROACH", "WAIT"
    focus_status = cycle.get("focus_status") or {}
    if "RUNNING" in focus_status.values() and not is_valid_confirmed and not snr_confirmed:
        monitor_state = "RUNNING"
        touch_status = "TOUCHED_PREVIOUSLY"
        touch_display = "ZONA TERAMBIL (POSISI COCOK)"
        focus_detail = cycle.get("focus_detail") or {}
        running_uncertain = any(
            item.get("account_status") == "RUNNING" and item.get("uncertain")
            for item in focus_detail.values()
        )
        running_with_plan = any(
            item.get("account_status") == "RUNNING" and item.get("plan_valid")
            for item in focus_detail.values()
        )
        reason = "POSISI COCOK; RIWAYAT BELUM PASTI" if running_uncertain else "POSISI COCOK OTOMATIS; PERIKSA MT5"
        next_action = "PANTAU SL/TP1 ACUAN" if running_with_plan else "PERIKSA SL/TP POSISI MT5"
        final_signal = "WAIT (MONITORING)"
    elif not is_valid_confirmed and not snr_confirmed:
        focus_detail = cycle.get("focus_detail") or {}
        active_method = "SNR" if zone.get("source") == "SNR" else "SMC"
        act_item = focus_detail.get(active_method) or {}
        has_active_zone = bool(zone and safe_float(zone.get("price_low", zone.get("low")), 0.0) > 0 and smc_zone_usable)

        if act_item.get("touched"):
            monitor_state = "ZONE_TOUCHED"
            touch_status = "TOUCHED_PREVIOUSLY"
            touch_display = f"{active_method} ZONE TOUCHED"
            reason = f"{active_method} ZONE TOUCHED; ENTRY BELUM DIKONFIRMASI"
            next_action = "PERIKSA RIWAYAT DAN MT5" if act_item.get("uncertain") else "PANTAU KONFIRMASI M5"
            final_signal = "WAIT"
        elif has_active_zone and monitor_state in ("APPROACHING", "AWAITING_CONFIRMATION", "IN_ZONE"):
            # Pertahankan status radar / approaching / inside zone dari active setup
            pass
        elif has_active_zone and not act_item.get("historical_touch"):
            # Zona aktif sedang menunggu pullback (RADAR)
            monitor_state = "RADAR"
            touch_status = "NOT_TOUCHED"
            z_desc = f"{zone.get('timeframe', '')} {zone.get('zone_type', '')} {safe_float(zone.get('price_low', zone.get('low'))):.2f}-{safe_float(zone.get('price_high', zone.get('high'))):.2f}".strip()
            touch_display = f"UNTOUCHED ({distance:.2f} pt)" if distance is not None else "UNTOUCHED"
            reason = f"RADAR: Menunggu pullback ke {z_desc}"
            next_action = "WAIT PULLBACK"
            final_signal = "WAIT"
        elif act_item.get("historical_touch"):
            monitor_state = "HISTORICAL_TOUCH"
            touch_status = "TOUCHED_PREVIOUSLY"
            touch_display = f"{active_method} TOUCHED (HISTORICAL)"
            reason = f"{active_method} PERNAH DISENTUH; ENTRY BELUM TERBUKTI"
            next_action = "PERIKSA RIWAYAT DAN MT5" if act_item.get("uncertain") else "PANTAU KUNJUNGAN BARU"
            final_signal = "WAIT"
        elif act_item.get("uncertain"):
            monitor_state = "HISTORY_UNCERTAIN"
            touch_status = "NO_ZONE"
            touch_display = "SENTUHAN BELUM TERBUKTI"
            reason, next_action, final_signal = "FOKUS TERKUNCI: RIWAYAT BELUM PASTI", "PERIKSA RIWAYAT DAN MT5", "WAIT"
        elif not has_active_zone:
            historical = next(((m, it) for m, it in focus_detail.items() if it.get("historical_touch")), None)
            touched = next(((m, it) for m, it in focus_detail.items() if it.get("touched")), None)
            uncertain = next(((m, it) for m, it in focus_detail.items() if it.get("uncertain")), None)
            if touched:
                m_t, it_t = touched
                monitor_state = "ZONE_TOUCHED"
                touch_status = "TOUCHED_PREVIOUSLY"
                touch_display = f"{m_t} ZONE TOUCHED"
                reason = f"{m_t} ZONE TOUCHED; ENTRY BELUM DIKONFIRMASI"
                next_action = "PERIKSA RIWAYAT DAN MT5" if it_t.get("uncertain") else "PANTAU KONFIRMASI M5"
                final_signal = "WAIT"
            elif historical:
                m_hist, it_hist = historical
                monitor_state = "HISTORICAL_TOUCH"
                touch_status = "TOUCHED_PREVIOUSLY"
                touch_display = f"{m_hist} TOUCHED (HISTORICAL)"
                reason = f"{m_hist} PERNAH DISENTUH; ENTRY BELUM TERBUKTI"
                next_action = "PERIKSA RIWAYAT DAN MT5" if it_hist.get("uncertain") else "PANTAU KUNJUNGAN BARU"
                final_signal = "WAIT"
            elif uncertain:
                monitor_state = "HISTORY_UNCERTAIN"
                touch_status = "NO_ZONE"
                touch_display = "SENTUHAN BELUM TERBUKTI"
                reason, next_action, final_signal = "FOKUS TERKUNCI: RIWAYAT BELUM PASTI", "PERIKSA RIWAYAT DAN MT5", "WAIT"

    account = cycle.get("account") if isinstance(cycle.get("account"), dict) else {}
    warnings = list(account.get("risk_warnings") or [])
    warnings.extend(cycle.get("risk_warnings") or [])
    alerts = list(cycle.get("alerts") or [])
    if warnings:
        alerts.extend(warnings)

    # Clean alert display resolution:
    if is_valid_confirmed and not is_entry_stale:
        alert_display = f"{dec_dir} CONFIRMED"
    elif is_entry_stale:
        alert_display = "ENTRY STALE"
    elif monitor_state == "INVALIDATED" or touch_status == "INVALIDATED":
        alert_display = "ZONE INVALIDATED"
    else:
        best_al = "NONE"
        for al in alerts:
            al_str = str(al).replace("\n", " ").strip()
            if any(k in al_str for k in ("CONFIRMED", "TERKONFIRMASI")):
                best_al = al_str
                break
            elif any(k in al_str for k in ("INVALID", "BATAL")):
                best_al = al_str
                break
            elif any(k in al_str for k in ("FOCUS CHANGED", "PERUBAHAN ZONA")):
                best_al = al_str
                break
            elif any(k in al_str for k in ("TOUCH", "MENYENTUH", "DISENTUH", "HARGA DEKAT", "APPROACHING")):
                best_al = al_str
                break
        if best_al == "NONE" and alerts:
            best_al = str(alerts[0]).replace("\n", " ").strip()
        if best_al.startswith("[SMC] "):
            best_al = best_al[6:].strip()
        elif best_al.startswith("[SNR] "):
            best_al = best_al[6:].strip()
        elif best_al.startswith("[ALERT] "):
            best_al = best_al[8:].strip()
        if "TERKONFIRMASI" in best_al:
            best_al = best_al.replace("TERKONFIRMASI", "CONFIRMED")
        if "PERUBAHAN ZONA FOKUS -> " in best_al:
            best_al = "FOCUS CHANGED [" + best_al.replace("PERUBAHAN ZONA FOKUS -> ", "").strip() + "]"
        alert_display = best_al if best_al else "NONE"
    return {
        "snapshot": snapshot,
        "smc": smc,
        "snr": snr,
        "scan": scan,
        "decision": decision,
        "summary": summary,
        "snr_scan": snr_scan,
        "snr_decision": snr_decision,
        "snr_zone": snr_zone,
        "snr_zone_direction": snr_zone_direction,
        "zone": zone,
        "plan": plan,
        "price": price,
        "distance": distance,
        "direction": direction,
        "confluence": confluence,
        "lifecycle": lifecycle,
        "monitor_state": monitor_state,
        "touch": touch_display,
        "touch_status": touch_status,
        "touch_display": touch_display,
        "trigger": trigger,
        "confirmed": confirmed_status,
        "confirmed_display": confirmed_display,
        "pending_status": p_status,
        "pending_display": pending_display,
        "signal": final_signal,
        "signal_display": final_signal,
        "data_invalid": data_invalid,
        "freshness": freshness,
        "reason": reason,
        "reason_display": reason,
        "next_action": next_action,
        "alerts": alerts,
        "alert_display": alert_display,
        "account": account,
        "active_setup": active_setup,
        "pending_setup": pending_setup,
        "confirmed_setup": confirmed_setup,
    }


def format_smart_monitor_report(cycle_result):
    """Format default `pantau` output in one compact, non-repeating terminal view."""
    data = _smart_monitor_context(cycle_result)
    zone = data["zone"]
    summary = data["summary"]
    snapshot = data["snapshot"]
    quote_fresh = (
        snapshot.get("data_ok") is not False and snapshot.get("live") is True
        and str(snapshot.get("data_freshness") or "").upper() not in ("STALE", "INVALID", "NO DATA")
    )
    price_txt = f"{data['price']:.2f}" if data["price"] > 0 and (not data["data_invalid"] or quote_fresh) else "N/A"
    h4 = str(summary.get("h4_bias") or "N/A").upper()
    h1 = str(summary.get("h1_bias") or "N/A").upper()
    snr_h1 = str((data["snr_scan"].get("market_bias") or {}).get("h1") or "N/A").upper()
    htf_alignment = "ALIGNED" if h1 in ("BULLISH", "BEARISH") and h1 == snr_h1 else "PARTIAL"
    m30 = str(summary.get("m30_bias") or "N/A").upper()
    m5 = data["trigger"]
    lo = safe_float(zone.get("price_low", zone.get("low")), 0.0)
    hi = safe_float(zone.get("price_high", zone.get("high")), 0.0)
    score_val = safe_float(zone.get("score"), 0.0)
    score_disp = f"{int(round(score_val))}" if abs(score_val - round(score_val)) < 1e-4 else f"{score_val:.1f}"
    z_tf = str(zone.get("timeframe") or "").strip()
    z_type = str(zone.get("zone_type") or zone.get("type") or "").strip()
    z_tag = f" ({z_tf} {z_type})".strip() if z_tf or z_type else ""
    if z_tag == "()":
        z_tag = ""
    elif z_tag:
        z_tag = f" {z_tag}"

    if lo > 0 and hi > 0:
        zone_lines = [
            f"DIRECTION   : {data['direction']}",
            f"ZONE        : {lo:.2f} - {hi:.2f}{z_tag}",
            f"GRADE       : {zone.get('grade', 'N/A')} ({score_disp})",
        ]
    else:
        zone_lines = ["DIRECTION   : N/A", "ZONE        : NO ACTIVE SETUP", "GRADE       : N/A"]

    snr_z = data.get("snr_zone") or {}
    snr_lo = safe_float(snr_z.get("low", snr_z.get("price_low")), 0.0)
    snr_hi = safe_float(snr_z.get("high", snr_z.get("price_high")), 0.0)
    snr_zone_line = None
    if snr_lo > 0 and snr_hi > 0 and (abs(snr_lo - lo) > 0.01 or abs(snr_hi - hi) > 0.01):
        s_lbl = str(snr_z.get("rank_label") or "SNR").strip()
        s_tf = str(snr_z.get("timeframe") or "").strip()
        s_grd = str(snr_z.get("grade") or "N/A").strip()
        s_tag = f"{s_lbl} {s_tf}".strip()
        snr_zone_line = f"SNR ZONE    : {snr_lo:.2f} - {snr_hi:.2f} ({s_tag} | Grade {s_grd})"

    pending_disp = data.get("pending_display", data.get("pending_status", "WAIT"))
    confirmed_disp = data.get("confirmed_display", data.get("confirmed", "WAIT"))
    touch_disp = data.get("touch_display", "NOT TOUCHED")
    trigger_disp = data.get("trigger", "WAIT")
    focused_states = ("RUNNING", "ZONE_TOUCHED", "HISTORICAL_TOUCH", "HISTORY_UNCERTAIN")
    status_text = "WAIT" if data["data_invalid"] and data["monitor_state"] not in focused_states else data["monitor_state"]
    alert_text = data.get("alert_display", "NONE")
    alert_text = str(alert_text).replace("\n", " ").strip()
    if len(alert_text) > 60:
        alert_text = alert_text[:57].rstrip() + "..."
    signal = data.get("signal_display", data.get("signal", "WAIT"))
    main_reason = str(data["reason"] or summary.get("main_reason") or "N/A").replace("\n", " ").strip()
    if data["reason"] == "WAITING FOR ZONE PULLBACK" and summary.get("main_reason"):
        main_reason = str(summary["main_reason"]).replace("\n", " ").strip()
    if len(main_reason) > 60:
        main_reason = main_reason[:57].rstrip() + "..."

    smc_focus = zone or data["decision"].get("primary_zone") or {}
    smc_block = format_smc_zone_lines(
        smc_focus, data["decision"], allow_plan=not data["data_invalid"],
        role="SMC ACTIVE" if zone else "SMC PRIMARY",
    ) if smc_focus else []
    if zone:
        smc_block = smc_block[1:]

    lines = [
        "==========================================",
        "       XAUUSD SMART MONITOR v2.5",
        "==========================================",
        f"PRICE       : {price_txt}",
        f"SMC H4 / H1 : {h4} / {h1}",
        f"SNR H1      : {snr_h1} | HTF: {htf_alignment}",
        f"M30 / M5    : {m30} / {m5}",
        "------------------------------------------",
        "ACTIVE SETUP",
        "------------------------------------------",
        *zone_lines,
        *( [snr_zone_line] if snr_zone_line else [] ),
        f"CONFLUENCE  : {data['confluence']}",
        f"SMC + SNR   : {data['confluence']}",
        "ENTRY STATUS",
        f"PENDING     : {pending_disp}",
        f"CONFIRMED   : {confirmed_disp}",
        f"TOUCH       : {touch_disp}",
        f"TRIGGER     : {trigger_disp}",
        "SMART MONITOR",
        f"STATE       : {status_text}",
        f"NEXT        : {data['next_action']}",
        f"ALERT       : {alert_text}",
        "FINAL",
        f"SIGNAL      : {signal}",
        f"REASON      : {main_reason}",
        "==========================================",
    ]
    if not snapshot.get("snapshot_status") and data["price"] <= 0:
        lines[3] = "PRICE       : N/A"
    return "\n".join(lines)


def format_smart_monitor_detail(cycle_result):
    """Structured lifecycle, validation, data freshness, and blocker details."""
    data = _smart_monitor_context(cycle_result)
    zone = data["zone"]
    plan = data["plan"]
    snapshot = data["snapshot"]
    decision = data["decision"]
    summary = data["summary"]
    stages = decision.get("m5_stages") or decision.get("stages") or {}
    active_setup = data.get("active_setup") or {}
    zone_id = str(active_setup.get("zone_id") or zone.get("zone_id") or "N/A")
    setup_type = str(active_setup.get("setup_type") or decision.get("setup_type") or plan.get("setup_type") or "N/A")
    direction = data["direction"]
    setup_id = active_setup.get("setup_id") or (f"{zone_id}:{direction}:{setup_type}" if zone_id != "N/A" else "N/A")
    blockers = [str(x) for x in (decision.get("reasons") or []) if x]
    if data["data_invalid"]:
        blockers.insert(0, str(snapshot.get("error_reason") or f"Data freshness: {data['freshness']}"))
    if not blockers:
        blockers = [str(summary.get("main_reason") or "N/A")]
    pending_status = str(plan.get("status") or "WAIT").upper()
    exp = plan.get("suggested_expiration") or plan.get("expires_at") or "N/A"
    lines = [
        format_smart_monitor_report(cycle_result),
        "",
        "LIFECYCLE DETAIL",
        f"SETUP ID    : {setup_id}",
        f"ZONE STATUS : {data['lifecycle']}",
        f"MONITOR     : {data['monitor_state']}",
        f"SIGNAL      : {summary.get('signal_status', 'NO SETUP')}",
        "",
        "VALIDATION",
        f"H4 / H1     : {summary.get('h4_bias', 'N/A')} / {summary.get('h1_bias', 'N/A')}",
        f"M30         : {summary.get('m30_bias', 'N/A')}",
        f"ZONE TOUCH  : {data.get('touch_display', data['touch'])}",
        f"SWEEP       : {_stage_status(stages.get('liquidity_sweep'), summary.get('liquidity_sweep', 'WAIT'))}",
        f"STRUCTURE   : {data['trigger']}",
        f"DISPLACEMENT: {_stage_status(stages.get('displacement'), summary.get('displacement', 'WAIT'))}",
        f"R:R CHECK   : {_stage_status(stages.get('risk_reward'), summary.get('rr_stage', 'WAIT'))}",
        f"SPREAD      : {safe_float(snapshot.get('spread'), 0.0):.2f}" if snapshot.get("spread") is not None else "SPREAD      : N/A",
        f"FRESHNESS   : {data['freshness']} | CONNECTION: {snapshot.get('connection_status', 'UNKNOWN')}",
        f"LAST M5     : {(snapshot.get('last_candle_times') or {}).get('M5', snapshot.get('last_closed_m5', 'N/A'))}",
        "",
        "PENDING PLAN (NOT PLACED)",
        f"STATUS      : {pending_status}",
        f"EXPIRATION  : {exp}",
        "",
        "BLOCKERS",
    ]
    lines.extend(f"- {reason.replace(chr(10), ' ')[:180]}" for reason in blockers)
    lines.extend([
        "",
        f"SMC + SNR   : {data['confluence']} (informational only)",
        f"NEXT ACTION : {data['next_action']}",
    ])
    smc_detail = (data["smc"].get("formatted_section") or "").strip()
    snr_detail = (data["snr"].get("formatted_section") or "").strip()
    if smc_detail:
        lines.extend(["", "SMC SCANNER DETAIL", smc_detail])
    if snr_detail:
        lines.extend(["", "SNR SCANNER DETAIL", snr_detail])
    return "\n".join(lines)


def _stage_status(stage, default="WAIT"):
    if isinstance(stage, dict):
        return stage.get("status") or default
    return stage or default


