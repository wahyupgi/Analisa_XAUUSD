# -*- coding: utf-8 -*-
"""
formatter.py — Penyusun Output Ringkas (Section U) dan Detail Teknis 22 Bagian Audit (Section V)
untuk SNR Advanced Scanner.
"""
from modules.snr.breakout_detection import build_snr_breakout_watch, classify_snr_breakout_state
from modules.snr.data_provider import safe_float


def format_snr_lifecycle_status(scan_result, decision=None):
    """Render a short lifecycle snapshot without treating non-active states as entry zones."""
    sr = scan_result if isinstance(scan_result, dict) else {}
    dec = decision if isinstance(decision, dict) else sr.get("decision") or {}
    ranked = sr.get("ranked_zones") or {}
    all_zones = ranked.get("all_zones") or []
    pending_breaks = ranked.get("pending_break_zones") or []
    broken_zones = ranked.get("broken_zones") or []
    active_supports = ranked.get("active_supports") or []
    active_resistances = ranked.get("active_resistances") or []
    supports = [z for z in all_zones if str(z.get("zone_type", "")).upper() == "SUPPORT"]
    supports.sort(key=lambda z: safe_float(z.get("distance_points"), float("inf")))

    bo_watch_lc = sr.get("breakout_watch") or build_snr_breakout_watch(ranked, price_action=sr.get("price_action"))
    # Prioritaskan breakout confirmed / retest watch / pending break, lalu support aktif, broken, resistance aktif
    zone = None
    if bo_watch_lc.get("confirmed"):
        zone = bo_watch_lc["confirmed"][0].get("zone")
    elif bo_watch_lc.get("retest_watch"):
        zone = bo_watch_lc["retest_watch"][0].get("zone")
    elif bo_watch_lc.get("pending"):
        zone = bo_watch_lc["pending"][0].get("zone")
    elif pending_breaks:
        zone = pending_breaks[0]
    elif active_supports:
        zone = active_supports[0]
    elif broken_zones:
        zone = broken_zones[0]
    elif active_resistances:
        zone = active_resistances[0]
    elif supports:
        zone = supports[0]

    signal = str(dec.get("signal", "WAIT")).upper()
    entry_confirmed = dec.get("entry_confirmed", False) if isinstance(dec, dict) else False
    entry_status = "CONFIRMED" if signal in ("BUY", "SELL") and (entry_confirmed or dec.get("status") == "CONFIRMED") else ("CONFIRMED" if signal in ("BUY", "SELL") else "WAIT")

    if zone is None:
        disp_status = "UNCONFIRMED" if str(sr.get("market_status", "")).upper() == "DATA UNAVAILABLE" else "NONE"
        disp_event = "NONE"
        zone_label = "S1"
    else:
        z_type = str(zone.get("zone_type", "SUPPORT")).upper()
        lifecycle = str(zone.get("lifecycle_status", zone.get("status", "UNCONFIRMED"))).upper()
        event = str(zone.get("lifecycle_event") or lifecycle).upper()
        reclaim = str(zone.get("reclaim_status") or "WAIT").upper()
        default_lbl = "NEAREST SUPPORT" if z_type == "SUPPORT" else "NEAREST RESISTANCE"
        zone_label = str(zone.get("rank_label") or default_lbl)
        bo_cls = classify_snr_breakout_state(zone)
        bo_st = bo_cls["state"]

        if lifecycle == "RECLAIMED" or (lifecycle in ("BROKEN", "RETEST", "RECLAIM_PENDING") and reclaim == "CONFIRMED"):
            disp_status = "RECLAIMED"
            disp_event = "RECLAIM CONFIRMED"
        elif bo_st == "CONFIRMED":
            disp_status = "INVALID — BROKEN"
            disp_event = bo_cls["label"]
        elif bo_st == "RETEST_WATCH":
            disp_status = "INVALID — BROKEN"
            disp_event = bo_cls["label"]
        elif bo_st == "BROKEN" or lifecycle in ("BROKEN", "RETEST", "RECLAIM_PENDING", "INVALIDATED"):
            disp_status = "INVALID — BROKEN"
            disp_event = "RECLAIM PENDING" if lifecycle == "RECLAIM_PENDING" else bo_cls["label"]
        elif bo_st in ("BREAKDOWN_PENDING", "BREAKOUT_PENDING"):
            disp_status = bo_st
            disp_event = bo_cls["label"]
        elif lifecycle == "UNCONFIRMED":
            disp_status = "UNCONFIRMED"
            disp_event = "UNCONFIRMED"
        elif lifecycle in ("ACTIVE", "TOUCHED", "SWEEP"):
            z_rel_lc = str(zone.get("zone_relevance") or "ACTIVE").upper()
            disp_status = "FAR REFERENCE" if z_rel_lc == "FAR_REFERENCE" else ("RADAR" if z_rel_lc == "RADAR" else "ACTIVE")
            if event in ("FALSE_BREAKDOWN", "FALSE_BREAKOUT"):
                disp_event = event.replace("_", " ")
            else:
                disp_event = event if event in ("SWEEP", "TOUCHED") else ("TOUCHED" if event == "ACTIVE" and zone.get("touch_count", 0) > 0 else "NONE")
                if disp_event in ("SWEEP", "TOUCHED"):
                    disp_event += " (HISTORICAL)"
        else:
            disp_status = lifecycle
            disp_event = event if event not in ("", "N/A") else "NONE"

    # Dapatkan alert perubahan terbaru (hanya alert paling mutakhir)
    alerts = []
    if isinstance(sr, dict):
        if sr.get("alerts"):
            alerts.extend(sr["alerts"])
        elif isinstance(sr.get("radar"), dict) and sr["radar"].get("alerts"):
            alerts.extend(sr["radar"]["alerts"])
    if not alerts and isinstance(dec, dict) and dec.get("alerts"):
        alerts.extend(dec["alerts"])

    alert_text = str(alerts[-1]).replace("\n", " ").strip() if alerts else "NONE"
    if alert_text != "NONE" and alert_text.startswith("[SNR] "):
        alert_text = alert_text[6:].strip()

    lines = [
        f"{zone_label} STATUS : {disp_status}",
        f"LAST EVENT: {disp_event}",
        f"ENTRY     : {entry_status}",
        f"ALERT     : {alert_text}",
    ]
    overlaps = ranked.get("overlaps") or []
    if overlaps:
        overlap = next((item for item in overlaps if item.get("entry_conflict")), overlaps[0])
        state = "CONFLICT" if overlap.get("entry_conflict") else "GEOMETRIC"
        lines.insert(2, f"SNR OVERLAP: {overlap['support_label']} / {overlap['resistance_label']} {state}")
    return lines


def build_snr_zone_reason(zone, h4_bias=None, max_words=10):
    """Short technical basis for an SNR zone, using only evidence stored on the zone."""
    if not isinstance(zone, dict):
        return "N/A"
    tf = str(zone.get("timeframe", "")).upper()
    kind = str(zone.get("zone_type", "")).lower()
    parts = [f"{tf} {kind}".strip()]
    bias = str(h4_bias or "").upper()
    if bias in ("BULLISH", "BEARISH"):
        with_trend = (kind == "resistance" and bias == "BEARISH") or (kind == "support" and bias == "BULLISH")
        parts.append(f"with H4 {bias.lower()}" if with_trend else f"counter H4 {bias.lower()}")
    if str(zone.get("swing_class", "")).upper() == "MAJOR":
        parts.append("major swing")
    reactions = int(safe_float(zone.get("historical_reaction_count"), 0))
    if reactions >= 2:
        parts.append(f"{reactions} reactions")
    confluence = zone.get("confluence_info") or {}
    if confluence.get("has_confluence"):
        parts.append(f"{'+'.join(confluence.get('timeframes') or [])} confluence")
    if str(zone.get("consolidation_structure", "")).upper() == "IMPULSIVE_DEPARTURE":
        parts.append("impulsive departure")
    touch = str((zone.get("touch_analysis") or {}).get("classification", "")).upper()
    if touch == "UNTESTED":
        parts.append("fresh")
    elif touch.startswith("TESTED"):
        parts.append(touch.lower().replace("_", " ").replace("plus", "+"))
    if str(zone.get("lifecycle_event", "")).upper() == "SWEEP":
        parts.append("liquidity sweep")
    reason, words = [], 0
    for part in parts:
        count = len(part.split())
        if reason and words + count > max_words:
            continue
        reason.append(part)
        words += count
    return ", ".join(reason)


def _snr_status_code(*texts):
    for text in texts:
        head = str(text or "").split(":", 1)
        if len(head) == 2 and head[0].isupper() and head[0].replace("_", "").isalnum() and "_" in head[0]:
            return head[0]
    return "WAIT"


def format_snr_focus_plan(decision, entry_validation=None, zone_override=None, market_bias=None):
    """Compact zone block: grade, technical reason, risk plan (confirmed or provisional) and honest status."""
    dec = decision or {}
    ev = entry_validation or {}
    focus = dec.get("target_zone") or ev.get("target_zone") or dec.get("trend_entry_zone") or ev.get("trend_entry_zone")
    zone = zone_override or focus
    if not isinstance(zone, dict):
        return "\n".join([
            "SNR ZONE : NONE", "REASON   : no active SNR zone", "ENTRY    : N/A", "SL       : N/A",
            "TP1/TP2  : N/A / N/A", "RR       : NOT CHECKED", "STATUS   : WAIT",
        ])
    is_focus = bool(zone.get("zone_id")) and zone.get("zone_id") == (focus or {}).get("zone_id")
    label = (
        f"{zone.get('rank_label', 'ZONE')} {zone.get('timeframe', 'N/A')} "
        f"{safe_float(zone.get('low')):.2f}-{safe_float(zone.get('high')):.2f}"
    )
    score = f"{safe_float(zone.get('score')):.2f}" if zone.get("score") is not None else "N/A"
    reason = build_snr_zone_reason(zone, (market_bias or {}).get("h4") if isinstance(market_bias, dict) else market_bias)

    direction = str(dec.get("signal", "WAIT")).upper()
    plan = ev.get("provisional_plan") if is_focus and isinstance(ev.get("provisional_plan"), dict) else {}
    entry = sl = tp1 = tp2 = 0.0
    entry_text, rr_text, suffix = "N/A", "NOT CHECKED", ""
    candidate_rr = None
    candidate_detail = ""
    if is_focus and direction in ("BUY", "SELL"):
        entry, sl, tp1, tp2 = (safe_float(dec.get(key), 0.0) for key in ("effective_entry", "sl", "tp1", "tp2"))
        if entry > 0 and sl > 0 and tp1 > 0 and (sl < entry < tp1 if direction == "BUY" else tp1 < entry < sl):
            entry_text = f"{direction} {entry:.2f}"
            rr_text = f"1:{abs(tp1 - entry) / abs(entry - sl):.2f}"
        else:
            entry = sl = tp1 = tp2 = 0.0
    elif plan.get("status") == "PROVISIONAL":
        entry, sl, tp1, tp2 = (safe_float(plan.get(key), 0.0) for key in ("entry", "sl", "tp1", "tp2"))
        valid = entry > 0 and sl > 0 and tp1 > 0 and (
            sl < entry < tp1 if plan.get("direction") == "BUY" else tp1 < entry < sl
        )
        if valid:
            from modules.utils import format_price_distance, extract_unit_spec_from_container
            _u_fp = extract_unit_spec_from_container(plan.get("rr_calculation") or dec)
            reference = "CURRENT PRICE" if plan.get("reference") == "CURRENT PRICE" else f"{format_price_distance(safe_float(plan.get('distance')), spec=_u_fp, include_mt5_points=True)} away"
            entry_text = f"{plan['direction']} PLAN {entry:.2f} ({reference})"
            rr_text = f"1:{abs(tp1 - entry) / abs(entry - sl):.2f}"
            suffix = " (PROVISIONAL ONLY)"
        else:
            entry = sl = tp1 = tp2 = 0.0
            entry_text = "N/A (LEVELS INVALID)"
    elif plan:
        candidate = plan.get("rr_calculation") or {}
        if candidate.get("rr_failed") and candidate.get("rr_tp1") is not None:
            candidate_rr = safe_float(candidate["rr_tp1"])
            minimum = candidate.get("min_rr_required")
            reference = "CURRENT PRICE" if plan.get("reference") == "CURRENT PRICE" else "ZONE EDGE"
            candidate_detail = f" ({reference}" + (f", MIN 1:{safe_float(minimum):.2f}" if minimum is not None else "") + ")"
        else:
            entry_text = f"N/A ({plan.get('note', 'PLAN UNAVAILABLE')})"
    elif not is_focus:
        entry_text = "N/A (NOT FOCUS ZONE)"

    entry_rr = ev.get("rr_calculation") or {}
    entry_rr_text = "NOT CHECKED"
    if is_focus and entry_rr.get("checked"):
        entry_rr_text = str(entry_rr.get("rr_display") or "N/A")
        if entry_rr.get("rr_failed") and entry_rr.get("rr_tp1") is not None:
            entry_rr_text = f"1:{safe_float(entry_rr['rr_tp1']):.2f} | FAILED"

    lifecycle = str(zone.get("lifecycle_status") or zone.get("status") or "").upper()
    if is_focus and direction in ("BUY", "SELL"):
        status = "CONFIRMED" if entry > 0 else f"{direction} (LEVELS N/A)"
    elif lifecycle in ("INVALIDATED", "BROKEN"):
        status = "INVALID — BROKEN"
    elif lifecycle in ("EXPIRED", "MITIGATED"):
        status = lifecycle
    elif is_focus:
        status = _snr_status_code(dec.get("reason"), ev.get("reason")) + suffix
        if status == "WAIT" and (ev.get("rr_calculation") or {}).get("rr_failed"):
            status = "RR_FAILED"
    else:
        status = "RADAR"
    zid_snr = str(zone.get("zone_id") or "N/A")
    lines = [
        f"SNR ZONE : {label} | Grade {zone.get('grade', 'N/A')} ({score}) [ID: {zid_snr}]",
        f"REASON   : {reason}",
        *(["INTERAKSI: ZONA DISENTUH M5 CLOSED (4 CANDLE TERAKHIR)"] if is_focus and ev.get("recent_closed_touch") else []),
        f"ENTRY    : {entry_text}",
        f"SL       : {sl:.2f}" if sl > 0 else "SL       : N/A",
        f"TP1/TP2  : {tp1:.2f} / {tp2:.2f}" if tp1 > 0 and tp2 > 0 and tp2 != tp1 else (f"TP1/TP2  : {tp1:.2f} / N/A" if tp1 > 0 else "TP1/TP2  : N/A / N/A"),
        *([f"PLAN RR  : 1:{candidate_rr:.2f} | FAILED{candidate_detail}", f"ENTRY RR : {entry_rr_text}"] if candidate_rr is not None else
          [f"PLAN RR  : {rr_text} | PASSED", f"ENTRY RR : {entry_rr_text}"] if suffix else
          [f"ENTRY RR : {entry_rr_text}"] if entry_rr.get("rr_failed") else [f"RR       : {rr_text}"]),
        f"STATUS   : {status}",
    ]
    change = ev.get("focus_change") if is_focus else None
    if isinstance(change, dict):
        lines.append(f"FOCUS    : {change.get('previous')} -> {change.get('to')} ({change.get('reason')})")
    return "\n".join(lines)


def _is_snr_far_zone(zone):
    """True jika zona SNR berada di luar batas jarak ATR/timeframe (FAR REFERENCE / HISTORICAL)."""
    if not isinstance(zone, dict):
        return False
    if zone.get("is_far_reference") or str(zone.get("zone_relevance") or "").upper() == "FAR_REFERENCE":
        return True
    if zone.get("is_distant_historical"):
        return True
    life = str(zone.get("lifecycle_status") or zone.get("status") or "").upper()
    if life == "HISTORICAL":
        return True
    d_atr = safe_float(zone.get("distance_atr"), 0.0)
    return d_atr > 12.0


def format_snr_summary(scan_result, decision=None, detailed=False):
    """
    Menyusun output ringkas SNR Advanced Scanner:
    Ringkas, padat, tanpa kalimat berulang, namun informasi zona dan alasannya sangat jelas.
    """
    sr = scan_result or {}
    dec = decision or sr.get("decision") or {}
    mb = sr.get("market_bias") or {}
    ranked = sr.get("ranked_zones") or {}
    ev = sr.get("entry_validation") or {}

    price_val = safe_float(sr.get("price"), 0.0)
    if price_val <= 0:
        price_val = safe_float(sr.get("last_price"), 0.0)
    price_str = f"{price_val:.2f}" if price_val > 0 else "N/A"

    time_label = (
        sr.get("time_label")
        or sr.get("snapshot_time")
        or sr.get("tick_time")
        or (str(sr.get("server_time")) if sr.get("server_time") else "N/A")
    )

    mkt_status = str(sr.get("market_status") or dec.get("market_status") or "OPEN").strip().upper()
    if mkt_status in ("MARKET OPEN", "MARKET_OPEN"):
        mkt_status = "OPEN"
    elif mkt_status in ("MARKET CLOSED", "MARKET_CLOSED"):
        mkt_status = "CLOSED"
    elif mkt_status in ("UNAVAILABLE", "NO_DATA", "DATA UNAVAILABLE"):
        mkt_status = "DATA UNAVAILABLE"

    if mkt_status == "DATA UNAVAILABLE":
        price_str = "DATA UNAVAILABLE"
        time_label = "N/A"
        h4_b = h1_b = m30_b = m5_b = "DATA UNAVAILABLE"
    else:
        h4_b = mb.get("h4", "UNCONFIRMED")
        h1_b = mb.get("h1", "UNCONFIRMED")
        m30_b = mb.get("m30", "UNCONFIRMED")
        m5_b = "WAIT" if mkt_status == "CLOSED" else mb.get("m5", "WAIT")

    active_sups = ranked.get("active_supports") or []
    active_ress = ranked.get("active_resistances") or []

    lines = [
        "====================================================",
        "          XAUUSD SNR ADVANCED SCANNER",
        "====================================================",
        f"MARKET STATUS : {mkt_status}",
        f"PRICE         : {price_str} | TIME: {time_label}",
        f"MARKET BIAS   : H4: {h4_b} | H1: {h1_b} | M30: {m30_b} | M5: {m5_b}",
        "",
        "----------------------------------------------------",
        "SUPPORT ZONES (ACTIVE — STRUCTURAL VALIDITY ONLY, ENTRY WAIT [MARKET CLOSED])" if mkt_status == "CLOSED" else "SUPPORT ZONES (ACTIVE)",
        "----------------------------------------------------",
    ]

    if mkt_status == "DATA UNAVAILABLE":
        lines.append("S1 : DATA UNAVAILABLE")
    elif active_sups:
        for idx_s, sz in enumerate(active_sups[:2]):
            rank_lbl = sz.get("rank_label", f"S{idx_s + 1}")
            lo_str = f"{safe_float(sz.get('low')):.2f}"
            hi_str = f"{safe_float(sz.get('high')):.2f}"
            grade = sz.get("grade", "D")
            score_str = f"{safe_float(sz.get('score')):.2f}"
            strength = sz.get("strength_short", "MODERATE")
            rel_tag = str(sz.get("zone_relevance") or "ACTIVE").upper()
            if mkt_status == "CLOSED" and rel_tag == "ACTIVE":
                rel_tag = "ACTIVE — STRUCTURAL VALID | ENTRY WAIT"
            reason_str = build_snr_zone_reason(sz, mb.get("h4"))
            sz_id = str(sz.get("zone_id") or "N/A")
            lines.append(f"{rank_lbl} : {lo_str} - {hi_str} [{rel_tag}] (Grade {grade} | Score {score_str} | {strength}) [ID: {sz_id}]")
            lines.append(f"     Reason: {reason_str}")
            if idx_s == 0 and len(active_sups[:2]) > 1:
                lines.append("")
    else:
        lines.append("S1 : NONE (Belum ada zona support aktif)")

    lines.extend([
        "",
        "----------------------------------------------------",
        "RESISTANCE ZONES (ACTIVE — STRUCTURAL VALIDITY ONLY, ENTRY WAIT [MARKET CLOSED])" if mkt_status == "CLOSED" else "RESISTANCE ZONES (ACTIVE)",
        "----------------------------------------------------",
    ])

    if mkt_status == "DATA UNAVAILABLE":
        lines.append("R1 : DATA UNAVAILABLE")
    elif active_ress:
        for idx_r, rz in enumerate(active_ress[:2]):
            rank_lbl = rz.get("rank_label", f"R{idx_r + 1}")
            lo_str = f"{safe_float(rz.get('low')):.2f}"
            hi_str = f"{safe_float(rz.get('high')):.2f}"
            grade = rz.get("grade", "D")
            score_str = f"{safe_float(rz.get('score')):.2f}"
            strength = rz.get("strength_short", "MODERATE")
            rel_tag = str(rz.get("zone_relevance") or "ACTIVE").upper()
            if mkt_status == "CLOSED" and rel_tag == "ACTIVE":
                rel_tag = "ACTIVE — STRUCTURAL VALID | ENTRY WAIT"
            reason_str = build_snr_zone_reason(rz, mb.get("h4"))
            rz_id = str(rz.get("zone_id") or "N/A")
            lines.append(f"{rank_lbl} : {lo_str} - {hi_str} [{rel_tag}] (Grade {grade} | Score {score_str} | {strength}) [ID: {rz_id}]")
            lines.append(f"     Reason: {reason_str}")
            if idx_r == 0 and len(active_ress[:2]) > 1:
                lines.append("")
    else:
        lines.append("R1 : NONE (Belum ada zona resistance aktif)")

    bo_watch = sr.get("breakout_watch") or build_snr_breakout_watch(ranked, price_action=sr.get("price_action"))
    bo_confirmed = list(bo_watch.get("confirmed") or [])
    bo_retest = list(bo_watch.get("retest_watch") or [])
    bo_pending = list(bo_watch.get("pending") or [])
    bo_broken = list(bo_watch.get("broken") or [])

    far_ref_zs = []
    _seen_far = set()
    for hz in list(ranked.get("far_reference_zones") or []) + list(ranked.get("historical_zones") or []):
        if not isinstance(hz, dict):
            continue
        f_key = hz.get("zone_id") or (hz.get("timeframe"), hz.get("zone_type"), round(safe_float(hz.get("low")), 2), round(safe_float(hz.get("high")), 2))
        if f_key in _seen_far:
            continue
        _seen_far.add(f_key)
        far_ref_zs.append(hz)

    if mkt_status != "DATA UNAVAILABLE" and (bo_confirmed or bo_retest or bo_pending or bo_broken):
        lines.extend([
            "",
            "----------------------------------------------------",
            "BROKEN / INVALID & BREAKOUT WATCH (NON-ACTIVE)",
            "----------------------------------------------------",
        ])
        for cw in bo_confirmed[:2]:
            ts_note = f" @ {cw['event_time']}" if cw.get("event_time") else ""
            lines.append(
                f"• {cw['label']} : {cw['timeframe']} {cw['zone_type']} "
                f"{cw['low']:.2f}-{cw['high']:.2f}{ts_note} (closed candle terkonfirmasi)"
            )
        for rw in bo_retest[:2]:
            ts_note = f" @ {rw['event_time']}" if rw.get("event_time") else ""
            lines.append(
                f"• {rw['label']} : {rw['timeframe']} {rw['zone_type']} "
                f"{rw['low']:.2f}-{rw['high']:.2f}{ts_note} (pantau retest flip pasca-break)"
            )
        for pw in bo_pending[:2]:
            ts_note = f" @ {pw['event_time']}" if pw.get("event_time") else ""
            lines.append(
                f"• {pw['label']} : {pw['timeframe']} {pw['zone_type']} "
                f"{pw['low']:.2f}-{pw['high']:.2f}{ts_note} (menunggu close TF asal)"
            )
        seen_brk = set()
        shown_brk = 0
        for bw in bo_broken:
            b_key = (bw["timeframe"], bw["zone_type"], round(bw["low"], 2), round(bw["high"], 2))
            if b_key in seen_brk:
                continue
            seen_brk.add(b_key)
            lines.append(
                f"• INVALID — {bw['label']} : {bw['timeframe']} {bw['zone_type']} "
                f"{bw['low']:.2f}-{bw['high']:.2f} (keluar dari zona aktif)"
            )
            shown_brk += 1
            if shown_brk >= 3:
                break

    if mkt_status != "DATA UNAVAILABLE" and far_ref_zs:
        lines.extend([
            "",
            "----------------------------------------------------",
            "FAR REFERENCE ZONES (REFERENCE ONLY)",
            "----------------------------------------------------",
        ])
        from modules.utils import format_price_distance, extract_unit_spec_from_container
        _u_sum = extract_unit_spec_from_container(sr)
        for hz in far_ref_zs[:3]:
            h_zt = str(hz.get("zone_type", "")).upper()
            h_dist = safe_float(hz.get("distance_points"), 0.0)
            lines.append(
                f"• FAR REFERENCE {h_zt} : {hz.get('timeframe', '')} "
                f"{safe_float(hz.get('low')):.2f}-{safe_float(hz.get('high')):.2f} "
                f"(Grade {hz.get('grade', 'D')} | Dist {format_price_distance(h_dist, spec=_u_sum, include_mt5_points=True)})"
            )

    sig_final = dec.get("signal", "WAIT")
    dir_final = dec.get("direction", "NONE")
    reason_final = dec.get("reason", "Waiting for valid SNR confirmation.")
    focus_lbl = dec.get("zone_label") or ev.get("zone_label")
    target_z = dec.get("target_zone") or ev.get("target_zone") or dec.get("trend_entry_zone") or ev.get("trend_entry_zone")
    if (not focus_lbl or focus_lbl == "N/A") and target_z:
        t_lo = safe_float(target_z.get("low"))
        t_hi = safe_float(target_z.get("high"))
        focus_lbl = f"{target_z.get('rank_label', 'ZONE')} ({t_lo:.2f} - {t_hi:.2f})"
    elif focus_lbl and target_z and "(" not in focus_lbl:
        t_lo = safe_float(target_z.get("low"))
        t_hi = safe_float(target_z.get("high"))
        focus_lbl = f"{focus_lbl} ({t_lo:.2f} - {t_hi:.2f})"

    # Cek plan jika ada (Confirmed Entry atau Provisional Plan)
    plan_line = None
    if sig_final in ("BUY", "SELL"):
        e_val = safe_float(dec.get("effective_entry") or dec.get("entry"), 0.0)
        sl_val = safe_float(dec.get("sl"), 0.0)
        tp_val = safe_float(dec.get("tp1"), 0.0)
        rr_disp = dec.get("rr_display")
        if e_val > 0 and sl_val > 0 and tp_val > 0:
            rr_str = f" | RR {rr_disp}" if rr_disp and rr_disp != "NOT CHECKED" else ""
            plan_line = f"ENTRY {sig_final} : {e_val:.2f} | SL: {sl_val:.2f} | TP: {tp_val:.2f}{rr_str} (CONFIRMED)"
    else:
        prov = ev.get("provisional_plan")
        if isinstance(prov, dict) and prov.get("status") == "PROVISIONAL":
            pe = safe_float(prov.get("entry"), 0.0)
            psl = safe_float(prov.get("sl"), 0.0)
            ptp = safe_float(prov.get("tp1"), 0.0)
            if pe > 0 and psl > 0 and ptp > 0:
                p_dir = prov.get("direction", "PLAN")
                p_rr = f"1:{abs(ptp - pe) / abs(pe - psl):.2f}"
                if mkt_status == "CLOSED":
                    plan_line = f"{p_dir} PLAN : {pe:.2f} | SL: {psl:.2f} | TP: {ptp:.2f} (RR {p_rr}, WAIT — MARKET CLOSED; Revalidasi zona/gap/RR/SL/TP/M5 saat market OPEN)"
                else:
                    plan_line = f"{p_dir} PLAN : {pe:.2f} | SL: {psl:.2f} | TP: {ptp:.2f} (RR {p_rr}, PROVISIONAL)"

    lines.extend([
        "",
        "----------------------------------------------------",
        "FINAL DECISION",
        "----------------------------------------------------",
        f"SIGNAL     : {sig_final}",
    ])
    if dir_final not in ("NONE", "N/A"):
        lines.append(f"DIRECTION  : {dir_final}")
    target_is_far = bool(target_z and _is_snr_far_zone(target_z) and sig_final not in ("BUY", "SELL"))
    if target_z and not target_is_far and focus_lbl and focus_lbl != "N/A":
        tz_id_s = str(target_z.get("zone_id") or "")
        tz_id_tag = f" [ID: {tz_id_s}]" if tz_id_s else ""
        lines.append(f"ACTIVE FOCUS: {focus_lbl}{tz_id_tag}")
    else:
        lines.append("ACTIVE FOCUS: NONE | WAIT")
        far_ref_z = target_z if target_is_far else (ev.get("far_reference_zone") or (far_ref_zs[0] if far_ref_zs else None))
        if isinstance(far_ref_z, dict):
            from modules.utils import format_price_distance, extract_unit_spec_from_container
            _u_far = extract_unit_spec_from_container(sr)
            f_lo = safe_float(far_ref_z.get("low"), 0.0)
            f_hi = safe_float(far_ref_z.get("high"), 0.0)
            f_dist = safe_float(far_ref_z.get("distance_points"), 0.0)
            f_atr = safe_float(far_ref_z.get("distance_atr"), 0.0)
            f_lbl = far_ref_z.get("rank_label") or far_ref_z.get("zone_type") or "SNR"
            lines.append(
                f"FAR ZONE    : FAR ZONE / REFERENCE ONLY — {far_ref_z.get('timeframe', '')} {f_lbl} "
                f"{f_lo:.2f}-{f_hi:.2f} ({format_price_distance(f_dist, spec=_u_far, include_mt5_points=True)} / {f_atr:.1f}x ATR)".replace("  ", " ")
            )
    if plan_line and not target_is_far:
        lines.append(f"PLAN       : {plan_line}")
    lines.append(f"REASON     : {reason_final}")

    mt5_recon = sr.get("mt5_reconciliation")
    mt5_snap = sr.get("mt5_orders_snapshot")
    inv_ord_warns = sr.get("invalid_pending_warnings") or []
    if isinstance(mt5_recon, dict) or isinstance(mt5_snap, dict) or inv_ord_warns:
        lines.extend([
            "",
            "----------------------------------------------------",
            "MT5 ORDER SYNC (READ-ONLY)",
            "----------------------------------------------------",
        ])
        if isinstance(mt5_recon, dict):
            lines.append(f"PENDING    : {mt5_recon.get('compact_status') or mt5_recon.get('display_status') or 'UNKNOWN'}")
            if detailed or mt5_recon.get("is_placed"):
                lines.append(f"EVIDENCE   : {mt5_recon.get('evidence_summary') or 'N/A'}")
        if isinstance(mt5_snap, dict):
            try:
                from modules.mt5_conn import format_mt5_orders_summary_lines
                m_recons = [mt5_recon] if isinstance(mt5_recon, dict) else None
                for ln in format_mt5_orders_summary_lines(
                    mt5_snap,
                    invalid_zone_warnings=inv_ord_warns,
                    matched_recons=m_recons,
                    detailed=detailed,
                    invalid_zone_scope="SNR",
                ):
                    lines.append(ln)
            except Exception:
                if inv_ord_warns:
                    lines.append(f"- RISK: {len(inv_ord_warns)} Pending on INVALID Zone [SNR] [HIGH RISK]")
        elif inv_ord_warns:
            lines.append(f"- RISK: {len(inv_ord_warns)} Pending on INVALID Zone [SNR] [HIGH RISK]")

    lines.extend([
        "",
        "====================================================",
    ])

    return "\n".join(lines)


def format_snr_detail(scan_result, decision=None):
    """
    Menyusun laporan lengkap 22 bagian teknis yang dapat diaudit (Section V)
    untuk perintah `scan snr detail`.
    """
    sr = scan_result or {}
    dec = decision or sr.get("decision") or {}
    mb = sr.get("market_bias") or {}
    ranked = sr.get("ranked_zones") or {}
    pa = sr.get("price_action") or {}
    ev = sr.get("entry_validation") or {}
    atr_diag = sr.get("atr_diagnostics") or {}
    swings_by_tf = sr.get("swings_by_tf") or {}
    sup_cands_by_tf = sr.get("support_candidates_by_tf") or {}
    res_cands_by_tf = sr.get("resistance_candidates_by_tf") or {}
    clustered_by_tf = sr.get("clustered_zones_by_tf") or {}
    conf_data = sr.get("confluence") or {}
    mom_by_tf = sr.get("momentum_by_tf") or {}
    spec = sr.get("symbol_spec") or {}

    all_zones = ranked.get("all_zones") or []
    active_sups = ranked.get("active_supports") or []
    active_ress = ranked.get("active_resistances") or []
    broken_zones = ranked.get("broken_zones") or []
    hist_zones = ranked.get("historical_zones") or []
    inv_zones = ranked.get("invalidated_zones") or []

    lines = [
        format_snr_summary(sr, dec, detailed=True),
        "",
        "====================================================",
        "       XAUUSD SNR ADVANCED — FULL AUDIT DETAIL",
        "====================================================",
    ]

    # 1. Market Overview
    from modules.utils import format_price_distance, extract_unit_spec_from_container
    _u_det = extract_unit_spec_from_container(sr)
    _d_det = _u_det["digits"]
    _sp_det_txt = format_price_distance(safe_float(sr.get('spread')), spec=_u_det, include_mt5_points=True)
    lines.extend([
        "",
        "[1. MARKET OVERVIEW]",
        f"  • Symbol        : {sr.get('symbol', 'XAUUSD')}",
        f"  • Market Status : {sr.get('market_status', 'OPEN')} (Data Source: {sr.get('data_source', 'SNAPSHOT')})",
        f"  • Last Price    : {safe_float(sr.get('last_price', sr.get('price'))):.{_d_det}f} (Ask: {safe_float(sr.get('ask')):.{_d_det}f}, Spread: {_sp_det_txt})",
        f"  • Last Candle   : {sr.get('last_candle', sr.get('server_time', 'N/A'))} ({sr.get('time_label', 'N/A')})",
        f"  • Broker Spec   : Digits={_u_det['digits']}, Point={_u_det['point']:g}, TickSize={_u_det['trade_tick_size']:g}, TickValue={spec.get('tick_value', 1.0)}, Source={spec.get('source', _u_det['source'])}",
    ])

    # 2. ATR Diagnostics
    lines.append("")
    lines.append("[2. ATR DIAGNOSTICS]")
    for tf in ("H4", "H1", "M30", "M5"):
        ad = atr_diag.get(tf) or {}
        if ad.get("available"):
            lines.append(
                f"  • {tf:<3} : ATR14={ad.get('atr14', 0):.2f} | MeanATR={ad.get('atr_mean', 0):.2f} "
                f"| Ratio={ad.get('atr_ratio', 1):.2f}x | Regime={ad.get('volatility_regime', 'N/A')} | Bars={ad.get('bars', 0)}"
            )
        else:
            lines.append(f"  • {tf:<3} : N/A")

    # 3. Swing High/Low
    lines.append("")
    lines.append("[3. SWING HIGH / LOW DETECTION]")
    for tf in ("H4", "H1", "M30", "M5"):
        sw_tf = swings_by_tf.get(tf) or {}
        s_sum = sw_tf.get("summary") or {}
        lines.append(
            f"  • {tf:<3} : Confirmed={s_sum.get('total_confirmed', 0)} "
            f"(Highs={s_sum.get('swing_high_count', 0)}, Lows={s_sum.get('swing_low_count', 0)}, "
            f"Major={s_sum.get('major_count', 0)}, Minor={s_sum.get('minor_count', 0)}, "
            f"Dedup={s_sum.get('duplicates_filtered', 0)}, UnconfirmedExcluded={len(sw_tf.get('unconfirmed_candidates', []))})"
        )

    # 4. Support Candidates
    lines.append("")
    lines.append("[4. SUPPORT CANDIDATES]")
    for tf in ("H4", "H1", "M30", "M5"):
        sc_list = sup_cands_by_tf.get(tf) or []
        lines.append(f"  • {tf:<3} ({len(sc_list)} kandidat):")
        for sc in sc_list[-3:]:
            lines.append(
                f"      - {sc.get('low', 0):.2f}-{sc.get('high', 0):.2f} (w={sc.get('width', 0):.2f}, "
                f"class={sc.get('swing_class')}, bounce={sc.get('bounce_strength_atr', 0):.2f} ATR)"
            )

    # 5. Resistance Candidates
    lines.append("")
    lines.append("[5. RESISTANCE CANDIDATES]")
    for tf in ("H4", "H1", "M30", "M5"):
        rc_list = res_cands_by_tf.get(tf) or []
        lines.append(f"  • {tf:<3} ({len(rc_list)} kandidat):")
        for rc in rc_list[-3:]:
            lines.append(
                f"      - {rc.get('low', 0):.2f}-{rc.get('high', 0):.2f} (w={rc.get('width', 0):.2f}, "
                f"class={rc.get('swing_class')}, rejection={rc.get('bounce_strength_atr', 0):.2f} ATR)"
            )

    # 6. Zone Clustering
    lines.append("")
    lines.append("[6. ZONE CLUSTERING]")
    for tf in ("H4", "H1", "M30", "M5"):
        cz_list = clustered_by_tf.get(tf) or []
        lines.append(f"  • {tf:<3} : {len(cz_list)} zona terklaster")
        for cz in cz_list[:4]:
            lines.append(
                f"      - [{cz.get('zone_id')}] {cz.get('zone_type')} {cz.get('low', 0):.2f}-{cz.get('high', 0):.2f} "
                f"(swings={cz.get('source_swing_count', 1)}, width={cz.get('width', 0):.2f})"
            )

    # 7. Touch Analysis
    lines.append("")
    lines.append("[7. TOUCH ANALYSIS]")
    display_zones = (active_sups[:3] + active_ress[:3] + broken_zones[:2]) or all_zones[:6]
    if display_zones:
        for z in display_zones:
            ta = z.get("touch_analysis") or {}
            lines.append(
                f"  • [{z.get('timeframe')} {z.get('zone_type')} {z.get('low', 0):.2f}-{z.get('high', 0):.2f}] "
                f"Status={ta.get('classification', z.get('touch_status', 'N/A'))} | ValidTouches={ta.get('valid_touch_count', 0)} "
                f"| WickOnly={ta.get('wick_only_count', 0)} | AvgBounce={ta.get('avg_bounce_atr', 0):.2f} ATR"
            )
    else:
        lines.append("  • N/A")

    # 8. Zone Freshness
    lines.append("")
    lines.append("[8. ZONE FRESHNESS]")
    if display_zones:
        for z in display_zones:
            lines.append(
                f"  • [{z.get('timeframe')} {z.get('zone_type')} {z.get('low', 0):.2f}-{z.get('high', 0):.2f}] "
                f"FreshnessRatio={z.get('freshness_ratio', 0):.2f} | AgeBars={z.get('age_bars', 0)} | TouchState={z.get('touch_status', 'N/A')}"
            )
    else:
        lines.append("  • N/A")

    # 9. Historical Reactions
    lines.append("")
    lines.append("[9. HISTORICAL REACTIONS]")
    if display_zones:
        for z in display_zones:
            lines.append(
                f"  • [{z.get('timeframe')} {z.get('zone_type')} {z.get('low', 0):.2f}-{z.get('high', 0):.2f}] "
                f"Reactions={z.get('historical_reaction_count', 0)} | RepeatedRejections={z.get('repeated_rejections', 0)} "
                f"| MaxBounce={z.get('bounce_strength_atr', 0):.2f} ATR | Structure={z.get('consolidation_structure', 'N/A')}"
            )
    else:
        lines.append("  • N/A")

    # 10. Breakout Analysis
    lines.append("")
    lines.append("[10. BREAKOUT ANALYSIS]")
    bo_detail = pa.get("breakout_detail") or {}
    lines.append(f"  • Summary Status : {pa.get('breakout_summary', 'NOT CONFIRMED')} ({bo_detail.get('status', 'NO_BREAKOUT')} | Type={bo_detail.get('break_type', pa.get('break_type', 'NONE'))})")
    lines.append(f"  • Reclaim / Fail : ReclaimDetected={bool(bo_detail.get('reclaim_detected', False))} | ZoneFailed={bool(bo_detail.get('zone_failed', False))}")
    lines.append(f"  • Penetration    : {format_price_distance(safe_float(bo_detail.get('penetration_points', 0)), spec=_u_det, include_mt5_points=True)} ({bo_detail.get('penetration_atr', 0):.2f} ATR)")
    lines.append(f"  • Body & Mom     : Body={bo_detail.get('body_atr', 0):.2f} ATR | MomSupported={bo_detail.get('momentum_supported', False)} | Vol={bo_detail.get('volume_supported', 'N/A')}")
    lines.append(f"  • Reason         : {bo_detail.get('reason', 'N/A')}")

    # 11. Retest Analysis
    lines.append("")
    lines.append("[11. RETEST ANALYSIS]")
    rt_detail = pa.get("retest_detail") or {}
    lines.append(f"  • Summary Status : {pa.get('retest_summary', 'NOT CONFIRMED')} ({rt_detail.get('status', 'NO_RETEST')})")
    lines.append(f"  • Role Reversal  : {rt_detail.get('role_reversal', 'NONE')}")
    lines.append(f"  • Reaction Held  : {rt_detail.get('retest_held', False)} (Confirmed={rt_detail.get('reaction_confirmed', False)})")
    lines.append(f"  • Reason         : {rt_detail.get('reason', 'N/A')}")

    # 12. Rejection Analysis
    lines.append("")
    lines.append("[12. REJECTION ANALYSIS]")
    rej_detail = pa.get("rejection_detail") or {}
    lines.append(f"  • Summary Status : {pa.get('rejection_summary', 'WAIT')} ({rej_detail.get('status', 'NONE')})")
    lines.append(
        f"  • Wick & Body    : UpperWick={rej_detail.get('upper_wick', 0):.2f} | LowerWick={rej_detail.get('lower_wick', 0):.2f} "
        f"| Body={rej_detail.get('body', 0):.2f} | Wick/Body={rej_detail.get('wick_to_body', 0):.2f}x | Wick/Range={rej_detail.get('wick_to_range', 0):.2f}"
    )
    lines.append(
        f"  • Context & Next : InZone={rej_detail.get('in_zone_context', False)} | NextCandleConfirmed={rej_detail.get('next_candle_confirmed', False)}"
    )
    lines.append(f"  • Reason         : {rej_detail.get('reason', 'N/A')}")

    # 13. Momentum Analysis
    lines.append("")
    lines.append("[13. MOMENTUM ANALYSIS]")
    for tf in ("H4", "H1", "M30", "M5"):
        m_tf = mom_by_tf.get(tf) or {}
        lines.append(
            f"  • {tf:<3} : Dir={m_tf.get('direction', 'N/A')} ({m_tf.get('strength', 'N/A')}) | "
            f"Body={m_tf.get('body_atr', 0):.2f} ATR | Vel={m_tf.get('velocity_atr_per_bar', 0):+.2f} ATR/bar | "
            f"TickVol={m_tf.get('tick_volume_status', 'N/A')} ({m_tf.get('tick_volume', 'N/A')}) | Weakening={m_tf.get('momentum_weakening', False)}"
        )

    # 14. Multi-Timeframe Confluence
    lines.append("")
    lines.append("[14. MULTI-TIMEFRAME CONFLUENCE]")
    conf_groups = conf_data.get("confluence_groups") or []
    if conf_groups:
        for cg in conf_groups:
            lines.append(
                f"  • MULTI-TF {cg.get('zone_type')} CONFLUENCE ({'+'.join(cg.get('timeframes', []))}): "
                f"{cg.get('confluence_low', 0):.2f} - {cg.get('confluence_high', 0):.2f} (Score={cg.get('score_15', 0):.1f}/15)"
            )
    else:
        lines.append("  • Tidak ada irisan Multi-TF yang memenuhi batas lebar maksimum saat ini.")

    # 15. Zone Scoring Breakdown
    lines.append("")
    lines.append("[15. ZONE SCORING BREAKDOWN (0-100)]")
    top_scored = (active_sups[:2] + active_ress[:2]) or all_zones[:4]
    if top_scored:
        for z in top_scored:
            sb = z.get("score_breakdown") or {}
            lbl = z.get("rank_label", z.get("zone_id"))
            lines.append(
                f"  • {lbl} ({z.get('timeframe')} {z.get('zone_type')} {z.get('low', 0):.2f}-{z.get('high', 0):.2f}) "
                f"— Total={z.get('score', 0):.2f}/100 (Grade {z.get('grade', 'D')}):"
            )
            for comp_key, comp_label in (
                ("swing_structural_strength", "Swing/Structural (20)"),
                ("historical_price_reaction", "Hist. Reaction   (20)"),
                ("touch_quality", "Touch Quality    (15)"),
                ("multi_tf_confluence", "MTF Confluence   (15)"),
                ("rejection_breakout_quality", "Rej/Breakout     (15)"),
                ("freshness", "Freshness        (10)"),
                ("momentum_confirmation", "Momentum          (5)"),
            ):
                c_obj = sb.get(comp_key) or {}
                lines.append(f"      - {comp_label}: {c_obj.get('score', 0):.2f} | {c_obj.get('reason', '')}")
    else:
        lines.append("  • N/A")

    # 16. Zone Ranking
    pending_breaks = ranked.get("pending_break_zones") or []
    reclaimed_zs = ranked.get("reclaimed_zones") or []
    lines.append("")
    lines.append("[16. ZONE RANKING & LIFECYCLE SEPARATION]")
    lines.append(
        f"  • Active Supports={len(active_sups)} | Active Resistances={len(active_ress)} "
        f"| PendingBreak={len(pending_breaks)} | Reclaimed={len(reclaimed_zs)} "
        f"| Historical={len(hist_zones)} | Broken={len(broken_zones)} | Invalidated={len(inv_zones)}"
    )
    by_cat = ranked.get("by_category") or {}
    for cat_name in (
        "STRONG SUPPORT",
        "MODERATE SUPPORT",
        "WEAK SUPPORT",
        "STRONG RESISTANCE",
        "MODERATE RESISTANCE",
        "WEAK RESISTANCE",
    ):
        cat_items = by_cat.get(cat_name) or []
        lines.append(f"  • {cat_name:<20}: {len(cat_items)} zona")

    # 17. Market Bias Calculation
    lines.append("")
    lines.append("[17. MARKET BIAS CALCULATION]")
    lines.append(f"  • Overall Bias : {mb.get('overall_bias', 'UNCONFIRMED')} ({mb.get('snr_status', 'UNCONFIRMED')})")
    lines.append(f"  • Reason       : {mb.get('reason', 'N/A')}")
    per_tf_mb = mb.get("per_timeframe") or {}
    for tf in ("H4", "H1", "M30", "M5"):
        tf_b = per_tf_mb.get(tf) or {}
        lines.append(f"      - {tf:<3}: {tf_b.get('bias', 'N/A')} | {tf_b.get('reason', '')}")

    # 18. Entry Validation
    lines.append("")
    lines.append("[18. ENTRY VALIDATION]")
    lines.append(f"  • BUY Status   : {ev.get('buy_status', 'WAIT')}")
    lines.append(f"  • SELL Status  : {ev.get('sell_status', 'WAIT')}")
    lines.append(f"  • Setup Type   : {ev.get('setup_type', 'NONE')}")
    lines.append(f"  • Focus Zone   : {ev.get('zone_label', 'N/A')}")
    lines.append(f"  • Nearest Zone : {ev.get('nearest_snr_label') or dec.get('nearest_snr_label', 'N/A')}")
    lines.append(f"  • Trend Entry  : {ev.get('trend_entry_label') or dec.get('trend_entry_label', 'N/A')}")
    lines.append(f"  • Trigger Zone : {ev.get('entry_trigger_label') or dec.get('entry_trigger_label', 'N/A')}")
    lines.append(f"  • Trigger      : {ev.get('trigger_status', 'NOT CONFIRMED')}")
    if ev.get("snr_role_context"):
        lines.append(f"  • SNR Role     : {ev.get('snr_role_context')}")
    if ev.get("pullback_target_label") and ev.get("pullback_target_label") != "NONE":
        lines.append(f"  • Pullback Area: {ev.get('pullback_target_label')}")
    mc_det = sr.get("market_context") or {}
    if mc_det:
        lines.append(f"  • Context Class: {mc_det.get('classification', 'N/A')} | Short-Term: {mc_det.get('short_term', 'N/A')} | Bias: {mc_det.get('trade_bias', 'NEUTRAL / WAIT')}")

    # 19. SL/TP Calculation
    lines.append("")
    lines.append("[19. SL/TP CALCULATION]")
    rr_calc = ev.get("rr_calculation") or {}
    if rr_calc.get("checked"):
        lines.append(f"  • Effective Entry         : {rr_calc.get('effective_entry', 'N/A')}")
        lines.append(f"  • Structural Invalidation : {rr_calc.get('structural_invalidation', 'N/A')}")
        _vb_val = rr_calc.get("volatility_buffer")
        _vb_txt = format_price_distance(safe_float(_vb_val), spec=_u_det, include_mt5_points=True) if isinstance(_vb_val, (int, float)) else "N/A"
        _sp_rr_txt = format_price_distance(safe_float(rr_calc.get("spread_points", 0)), spec=_u_det, include_mt5_points=True)
        lines.append(f"  • Volatility Buffer       : {_vb_txt} (Spread={_sp_rr_txt})")
        lines.append(f"  • Stop Loss (SL)          : {rr_calc.get('sl', 'N/A')}")
        lines.append(f"  • Take Profit (TP1 / TP2) : {rr_calc.get('tp1', 'N/A')} / {rr_calc.get('tp2', 'N/A')}")
    else:
        lines.append("  • NOT CHECKED (Trigger entry belum terkonfirmasi sehingga kalkulasi SL/TP belum dijalankan).")

    # 20. Risk/Reward
    lines.append("")
    lines.append("[20. RISK / REWARD]")
    if rr_calc.get("checked"):
        rr2 = rr_calc.get("rr_tp2")
        rr2_text = f"1:{rr2:.2f}" if isinstance(rr2, (int, float)) else "N/A"
        lines.append(f"  • RR Ratio (TP1 / TP2) : {rr_calc.get('rr_display', 'N/A')} (TP2={rr2_text})")
        lines.append(f"  • Minimum Required     : 1 : {rr_calc.get('min_rr_required', 1.5):.2f}")
        lines.append(f"  • USD Risk / 1.0 Lot   : ${rr_calc.get('risk_usd_per_lot', 0):.2f} (Reward=${rr_calc.get('reward_usd_per_lot', 0):.2f})")
        lines.append(f"  • Validation Reason    : {rr_calc.get('reason', 'N/A')}")
    else:
        lines.append("  • RR : NOT CHECKED")

    # 21. Final Decision
    mkt_status = str(sr.get("market_status", dec.get("market_status", "OPEN"))).upper()
    live_entry = str(dec.get("live_entry", sr.get("live_entry", "WAIT — MARKET CLOSED" if mkt_status == "CLOSED" else ("WAIT — DATA UNAVAILABLE" if mkt_status == "DATA UNAVAILABLE" else dec.get("signal", "WAIT")))))
    lines.append("")
    lines.append("[21. FINAL DECISION]")
    lines.append(f"  • FINAL STATUS : {dec.get('final_status', 'MARKET CLOSED' if mkt_status == 'CLOSED' else ('DATA UNAVAILABLE' if mkt_status == 'DATA UNAVAILABLE' else dec.get('signal', 'WAIT')))}")
    lines.append(f"  • SIGNAL    : {dec.get('signal', 'WAIT')}")
    lines.append(f"  • LIVE ENTRY   : {live_entry}")
    lines.append(f"  • DIRECTION : {dec.get('direction', 'NONE')}")
    lines.append(f"  • REASON    : {dec.get('reason', 'N/A')}")

    # 22. Wait Reason Breakdown
    lines.append("")
    lines.append("[22. WAIT REASON BREAKDOWN]")
    wait_list = dec.get("wait_reasons") or ev.get("wait_reasons") or []
    if wait_list:
        for idx_w, wr in enumerate(wait_list, start=1):
            lines.append(f"  {idx_w}. {wr}")
    else:
        lines.append("  • Tidak ada kendala WAIT (seluruh syarat entry terpenuhi).")

    lines.append("")
    lines.append("====================================================")
    return "\n".join(lines)


def format_snr_presentation(scan_result, decision=None, detailed=False):
    """Fungsi pembungkus untuk memilih output ringkas atau detail SNR Scanner."""
    if detailed:
        return format_snr_detail(scan_result, decision=decision)
    return format_snr_summary(scan_result, decision=decision)

