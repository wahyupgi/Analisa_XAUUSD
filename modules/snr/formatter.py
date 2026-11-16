# -*- coding: utf-8 -*-
"""
formatter.py — Penyusun Output Ringkas (Section U) dan Detail Teknis 22 Bagian Audit (Section V)
untuk SNR Advanced Scanner.
"""
from modules.snr.data_provider import safe_float


def format_snr_lifecycle_status(scan_result, decision=None):
    """Render a short lifecycle snapshot without treating non-active states as entry zones."""
    sr = scan_result if isinstance(scan_result, dict) else {}
    dec = decision if isinstance(decision, dict) else sr.get("decision") or {}
    ranked = sr.get("ranked_zones") or {}
    supports = [
        zone for zone in (ranked.get("all_zones") or [])
        if str(zone.get("zone_type", "")).upper() == "SUPPORT"
    ]
    supports.sort(key=lambda zone: safe_float(zone.get("distance_points"), float("inf")))
    zone = supports[0] if supports else None
    signal = str(dec.get("signal", "WAIT")).upper()
    entry_confirmed = dec.get("entry_confirmed", False) if isinstance(dec, dict) else False
    entry_status = "CONFIRMED" if signal in ("BUY", "SELL") and (entry_confirmed or dec.get("status") == "CONFIRMED") else ("CONFIRMED" if signal in ("BUY", "SELL") else "WAIT")

    if zone is None:
        disp_status = "UNCONFIRMED" if str(sr.get("market_status", "")).upper() == "DATA UNAVAILABLE" else "NONE"
        disp_event = "NONE"
        zone_label = "S1"
    else:
        lifecycle = str(zone.get("lifecycle_status", zone.get("status", "UNCONFIRMED"))).upper()
        event = str(zone.get("lifecycle_event") or lifecycle).upper()
        reclaim = str(zone.get("reclaim_status") or "WAIT").upper()
        zone_label = str(zone.get("rank_label") or "S1")

        if lifecycle == "RECLAIMED" or (lifecycle in ("BROKEN", "RETEST", "RECLAIM_PENDING") and reclaim == "CONFIRMED"):
            disp_status = "RECLAIMED"
            disp_event = "RECLAIM CONFIRMED"
        elif lifecycle in ("BROKEN", "RETEST", "RECLAIM_PENDING"):
            disp_status = "BROKEN"
            disp_event = (
                "RETEST" if lifecycle == "RETEST"
                else ("RECLAIM PENDING" if lifecycle == "RECLAIM_PENDING" else "BREAKDOWN")
            )
        elif lifecycle == "INVALIDATED":
            disp_status = "INVALIDATED"
            disp_event = "INVALIDATED"
        elif lifecycle == "UNCONFIRMED":
            disp_status = "UNCONFIRMED"
            disp_event = "UNCONFIRMED"
        elif lifecycle in ("ACTIVE", "TOUCHED", "SWEEP"):
            disp_status = "ACTIVE"
            disp_event = event if event in ("SWEEP", "TOUCHED") else ("TOUCHED" if event == "ACTIVE" and zone.get("touch_count", 0) > 0 else "NONE")
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

    alert_text = str(alerts[-1]).strip() if alerts else "NONE"
    if alert_text != "NONE" and alert_text.startswith("[SNR] "):
        alert_text = alert_text[6:].strip()
    if len(alert_text) > 48:
        alert_text = alert_text[:45] + "..."

    lines = [
        f"{zone_label} STATUS : {disp_status}",
        f"LAST EVENT: {disp_event}",
        f"ENTRY     : {entry_status}",
        f"ALERT     : {alert_text}",
    ]
    return lines


def format_snr_summary(scan_result, decision=None):
    """
    Menyusun output ringkas SNR Advanced Scanner sesuai format Section U.
    Menggunakan hasil perhitungan aktual; informasi yang belum diperiksa ditampilkan
    sebagai N/A atau NOT CHECKED.
    """
    sr = scan_result or {}
    dec = decision or sr.get("decision") or {}
    mb = sr.get("market_bias") or {}
    ranked = sr.get("ranked_zones") or {}
    pa = sr.get("price_action") or {}
    ev = sr.get("entry_validation") or {}

    price_val = safe_float(sr.get("price"), 0.0)
    price_str = f"{price_val:.2f}" if price_val > 0 else "N/A"
    time_label = sr.get("time_label") or "N/A"

    mkt_status = str(sr.get("market_status") or dec.get("market_status") or "OPEN").strip().upper()
    if mkt_status in ("MARKET OPEN", "MARKET_OPEN"):
        mkt_status = "OPEN"
    elif mkt_status in ("MARKET CLOSED", "MARKET_CLOSED"):
        mkt_status = "CLOSED"
    elif mkt_status in ("UNAVAILABLE", "NO_DATA", "DATA UNAVAILABLE"):
        mkt_status = "DATA UNAVAILABLE"

    data_source = sr.get("data_source") or (
        "MT5 HISTORICAL"
        if mkt_status == "CLOSED"
        else ("DATA UNAVAILABLE" if mkt_status == "DATA UNAVAILABLE" else "MT5 LIVE")
    )
    last_price_val = safe_float(sr.get("last_price", price_val), 0.0)
    last_price_str = (
        f"{last_price_val:.2f}"
        if (last_price_val > 0 and mkt_status != "DATA UNAVAILABLE")
        else "DATA UNAVAILABLE"
    )
    last_candle_str = sr.get("last_candle") or (
        str(sr.get("server_time")) if (sr.get("server_time") and mkt_status != "DATA UNAVAILABLE") else "DATA UNAVAILABLE"
    )
    snr_analysis = sr.get("snr_analysis") or dec.get("snr_analysis") or (
        "DATA UNAVAILABLE" if mkt_status == "DATA UNAVAILABLE" else "AVAILABLE"
    )

    h4_b = "DATA UNAVAILABLE" if mkt_status == "DATA UNAVAILABLE" else mb.get("h4", "UNCONFIRMED")
    h1_b = "DATA UNAVAILABLE" if mkt_status == "DATA UNAVAILABLE" else mb.get("h1", "UNCONFIRMED")
    m30_b = "DATA UNAVAILABLE" if mkt_status == "DATA UNAVAILABLE" else mb.get("m30", "UNCONFIRMED")
    m5_b = "DATA UNAVAILABLE" if mkt_status == "DATA UNAVAILABLE" else ("WAIT" if mkt_status == "CLOSED" else mb.get("m5", "WAIT"))
    snr_status = "DATA UNAVAILABLE" if mkt_status == "DATA UNAVAILABLE" else mb.get("snr_status", "UNCONFIRMED")

    active_sups = ranked.get("active_supports") or []
    active_ress = ranked.get("active_resistances") or []

    if mkt_status == "DATA UNAVAILABLE":
        sup_summary = "DATA UNAVAILABLE"
        res_summary = "DATA UNAVAILABLE"
    else:
        if active_sups:
            s0 = active_sups[0]
            sup_summary = (
                f"{s0.get('rank_label', 'S1')} {safe_float(s0.get('low')):.2f} - {safe_float(s0.get('high')):.2f} "
                f"(Grade {s0.get('grade', 'D')} | Score {int(round(safe_float(s0.get('score'))))} | {s0.get('strength_short', 'MODERATE')})"
            )
        else:
            sup_summary = "NONE"

        if active_ress:
            r0 = active_ress[0]
            res_summary = (
                f"{r0.get('rank_label', 'R1')} {safe_float(r0.get('low')):.2f} - {safe_float(r0.get('high')):.2f} "
                f"(Grade {r0.get('grade', 'D')} | Score {int(round(safe_float(r0.get('score'))))} | {r0.get('strength_short', 'MODERATE')})"
            )
        else:
            res_summary = "NONE"

    sig_final = dec.get("signal", "WAIT")
    live_entry = dec.get("live_entry") or sr.get("live_entry") or (
        "WAIT — MARKET CLOSED"
        if mkt_status == "CLOSED"
        else ("WAIT — DATA UNAVAILABLE" if mkt_status == "DATA UNAVAILABLE" else sig_final)
    )

    lines = [
        "====================================================",
        "          XAUUSD SNR ADVANCED SCANNER",
        "====================================================",
        "",
        f"MARKET STATUS : {mkt_status}",
        f"DATA SOURCE   : {data_source}",
        f"LAST PRICE    : {last_price_str}",
        f"LAST CANDLE   : {last_candle_str}",
    ]
    if mkt_status == "OPEN":
        lines.append(f"PRICE       : {price_str}")
        lines.append(f"TIME        : {time_label}")
    lines.extend(format_snr_lifecycle_status(sr, dec))

    df_info = sr.get("data_freshness_info") or {}
    conn_disp = str(sr.get("connection_status") or df_info.get("connection_status") or "UNKNOWN")
    tick_st_disp = str(df_info.get("tick_status") or ("MARKET CLOSED" if mkt_status == "CLOSED" else sr.get("data_freshness", "N/A")))
    candle_st_disp = str(df_info.get("candle_status") or sr.get("data_freshness", "N/A"))
    last_tick_disp = str(sr.get("tick_time") or sr.get("snapshot_time") or "N/A")
    lct_map = sr.get("last_candle_times") or df_info.get("last_candle_times") or {}
    last_m5_disp = str(sr.get("last_closed_m5") or lct_map.get("M5") or last_candle_str or "N/A")
    fc_map = sr.get("forming_candles") or {}
    curr_m5_disp = str(sr.get("current_candle_m5") or (fc_map.get("M5") or {}).get("time") or "N/A")
    analysis_t_disp = str(sr.get("snapshot_time") or sr.get("server_time") or "N/A")
    data_val_disp = str(df_info.get("data_validation") or ("PASS" if sr.get("data_freshness") == "FRESH" else "FAIL"))

    lines.extend([
        "",
        "DATA INTEGRITY",
        "--------------------------------",
        f"MT5 CONNECTION   : {conn_disp}",
        f"TICK STATUS      : {tick_st_disp}",
        f"CANDLE STATUS    : {candle_st_disp}",
        f"LAST TICK        : {last_tick_disp}",
        f"LAST CLOSED M5   : {last_m5_disp}",
        f"CURRENT CANDLE   : {curr_m5_disp}",
        f"ANALYSIS TIME    : {analysis_t_disp}",
        f"DATA VALIDATION  : {data_val_disp}",
        "",
        "MULTI-TF MARKET BIAS",
        f"H4            : {h4_b}",
        f"H1            : {h1_b}",
        f"M30           : {m30_b}",
        f"M5            : {m5_b}",
        "",
        f"SNR STATUS  : {snr_status}",
        f"SUPPORT       : {sup_summary}",
        f"RESISTANCE    : {res_summary}",
        "",
        f"SNR ANALYSIS  : {snr_analysis}",
        f"LIVE ENTRY    : {live_entry}",
        "",
        "----------------------------------------------------",
        "SUPPORT ZONES",
        "----------------------------------------------------",
    ])

    if active_sups:
        for idx_s, sz in enumerate(active_sups[:2]):
            lbl = sz.get("rank_label", f"S{idx_s + 1}")
            s_lo = safe_float(sz.get("low"), 0.0)
            s_hi = safe_float(sz.get("high"), s_lo)
            s_gr = sz.get("grade", "D")
            s_sc = int(round(safe_float(sz.get("score"), 0.0)))
            s_st = sz.get("strength_short", "MODERATE")
            lines.append(f"{lbl} : {s_lo:.2f} - {s_hi:.2f}")
            lines.append(f"     Grade {s_gr} | Score {s_sc} | {s_st}")
            if idx_s == 0 and len(active_sups[:2]) > 1:
                lines.append("")
    else:
        lines.append("S1 : N/A (Belum ada zona support aktif)")

    lines.extend([
        "",
        "----------------------------------------------------",
        "RESISTANCE ZONES",
        "----------------------------------------------------",
    ])

    if active_ress:
        for idx_r, rz in enumerate(active_ress[:2]):
            lbl = rz.get("rank_label", f"R{idx_r + 1}")
            r_lo = safe_float(rz.get("low"), 0.0)
            r_hi = safe_float(rz.get("high"), r_lo)
            r_gr = rz.get("grade", "D")
            r_sc = int(round(safe_float(rz.get("score"), 0.0)))
            r_st = rz.get("strength_short", "MODERATE")
            lines.append(f"{lbl} : {r_lo:.2f} - {r_hi:.2f}")
            lines.append(f"     Grade {r_gr} | Score {r_sc} | {r_st}")
            if idx_r == 0 and len(active_ress[:2]) > 1:
                lines.append("")
    else:
        lines.append("R1 : N/A (Belum ada zona resistance aktif)")

    bo_label = pa.get("breakout_summary", "NOT CONFIRMED")
    rt_label = pa.get("retest_summary", "NOT CONFIRMED")
    rej_label = pa.get("rejection_summary", "NOT CONFIRMED")
    mom_label = pa.get("momentum_summary", "N/A")
    focus_pa_lbl = pa.get("focus_zone_label")
    bo_z_lbl = pa.get("breakout_zone_label")
    rt_z_lbl = pa.get("retest_zone_label")
    if bo_z_lbl and focus_pa_lbl and bo_z_lbl != focus_pa_lbl and pa.get("breakout_code") != "NO_BREAKOUT":
        bo_hist_tag = " (HISTORICAL)" if pa.get("breakout_is_historical") else ""
        bo_label = f"{bo_label}{bo_hist_tag} [{bo_z_lbl}]"
    if rt_z_lbl and focus_pa_lbl and rt_z_lbl != focus_pa_lbl and pa.get("retest_code") != "NO_RETEST":
        rt_hist_tag = " (HISTORICAL)" if pa.get("retest_is_historical") else ""
        rt_label = f"{rt_label}{rt_hist_tag} [{rt_z_lbl}]"

    buy_st = dec.get("buy_status", ev.get("buy_status", "WAIT"))
    sell_st = dec.get("sell_status", ev.get("sell_status", "WAIT"))
    zone_lbl = dec.get("zone_label", ev.get("zone_label", "N/A"))
    trig_st = dec.get("trigger_status", ev.get("trigger_status", "NOT CONFIRMED"))
    rr_st = dec.get("rr_display", ev.get("rr_status", "NOT CHECKED"))

    sig_final = dec.get("signal", "WAIT")
    dir_final = dec.get("direction", "NONE")
    reason_final = dec.get("reason", "Waiting for valid SNR confirmation.")

    final_status_lbl = dec.get(
        "final_status",
        "MARKET CLOSED"
        if mkt_status == "CLOSED"
        else ("DATA UNAVAILABLE" if mkt_status == "DATA UNAVAILABLE" else sig_final),
    )

    lines.extend([
        "",
        "----------------------------------------------------",
        "PRICE ACTION",
        "----------------------------------------------------",
        f"BREAKOUT  : {bo_label}",
        f"RETEST    : {rt_label}",
        f"REJECTION : {rej_label}",
        f"MOMENTUM  : {mom_label}",
        "",
        "----------------------------------------------------",
        "ENTRY VALIDATION",
        "----------------------------------------------------",
        f"BUY  : {buy_st}",
        f"SELL : {sell_st}",
        "",
        f"ZONE       : {zone_lbl}",
        f"TRIGGER    : {trig_st}",
        f"RR         : {rr_st}",
    ])

    mc = sr.get("market_context") or {}
    role_ctx = ev.get("snr_role_context")
    pb_tgt = ev.get("pullback_target_label")
    if mc or role_ctx:
        lines.extend([
            "",
            "----------------------------------------------------",
            "MARKET CONTEXT (SNR)",
            "----------------------------------------------------",
            f"CONTEXT CLASS : {mc.get('classification', 'N/A')}",
            f"PRIMARY TREND : {mc.get('primary_trend', mb.get('h4', 'UNCONFIRMED'))}",
            f"SHORT TERM    : {mc.get('short_term', mom_label)}",
            f"SNR CONTEXT   : {role_ctx or mc.get('snr_context', zone_lbl)}",
            f"NEAREST ZONE  : {dec.get('nearest_snr_label') or ev.get('nearest_snr_label', 'N/A')}",
            f"TREND ENTRY   : {dec.get('trend_entry_label') or ev.get('trend_entry_label', 'N/A')}",
            f"ENTRY TRIGGER : {dec.get('entry_trigger_label') or ev.get('entry_trigger_label', 'N/A')}",
            f"PULLBACK AREA : {pb_tgt or 'NONE'}",
            f"TRADE BIAS    : {mc.get('trade_bias', 'NEUTRAL / WAIT')}",
        ])

    lines.extend([
        "",
        "----------------------------------------------------",
        "FINAL DECISION",
        "----------------------------------------------------",
        f"FINAL STATUS : {final_status_lbl}",
        f"SIGNAL     : {sig_final}",
        f"LIVE ENTRY : {live_entry}",
        f"DIRECTION  : {dir_final}",
    ])

    if mkt_status == "OPEN" and sig_final in ("BUY", "SELL") and dec.get("entry") is not None:
        lines.append(f"ENTRY      : {safe_float(dec.get('entry')):.2f}")
        lines.append(f"SL         : {safe_float(dec.get('sl')):.2f}")
        lines.append(f"TP1        : {safe_float(dec.get('tp1')):.2f}")
        if dec.get("tp2") is not None:
            lines.append(f"TP2        : {safe_float(dec.get('tp2')):.2f}")
        lines.append(f"RR         : {rr_st}")

    lines.extend([
        f"REASON     : {reason_final}",
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
        format_snr_summary(sr, dec),
        "",
        "====================================================",
        "       XAUUSD SNR ADVANCED — FULL AUDIT DETAIL",
        "====================================================",
    ]

    # 1. Market Overview
    lines.extend([
        "",
        "[1. MARKET OVERVIEW]",
        f"  • Symbol        : {sr.get('symbol', 'XAUUSD')}",
        f"  • Market Status : {sr.get('market_status', 'OPEN')} (Data Source: {sr.get('data_source', 'SNAPSHOT')})",
        f"  • Last Price    : {safe_float(sr.get('last_price', sr.get('price'))):.2f} (Ask: {safe_float(sr.get('ask')):.2f}, Spread: {safe_float(sr.get('spread')):.2f})",
        f"  • Last Candle   : {sr.get('last_candle', sr.get('server_time', 'N/A'))} ({sr.get('time_label', 'N/A')})",
        f"  • Broker Spec   : Digits={spec.get('digits', 2)}, TickSize={spec.get('tick_size', 0.01)}, TickValue={spec.get('tick_value', 1.0)}, Source={spec.get('source', 'N/A')}",
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
    lines.append(f"  • Summary Status : {pa.get('breakout_summary', 'NOT CONFIRMED')} ({bo_detail.get('status', 'NO_BREAKOUT')})")
    lines.append(f"  • Penetration    : {bo_detail.get('penetration_points', 0):.2f} pts ({bo_detail.get('penetration_atr', 0):.2f} ATR)")
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
    lines.append("")
    lines.append("[16. ZONE RANKING & LIFECYCLE SEPARATION]")
    lines.append(
        f"  • Active Supports={len(active_sups)} | Active Resistances={len(active_ress)} "
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
        lines.append(f"  • Volatility Buffer       : {rr_calc.get('volatility_buffer', 'N/A')} (Spread={rr_calc.get('spread_points', 0)})")
        lines.append(f"  • Stop Loss (SL)          : {rr_calc.get('sl', 'N/A')}")
        lines.append(f"  • Take Profit (TP1 / TP2) : {rr_calc.get('tp1', 'N/A')} / {rr_calc.get('tp2', 'N/A')}")
    else:
        lines.append("  • NOT CHECKED (Trigger entry belum terkonfirmasi sehingga kalkulasi SL/TP belum dijalankan).")

    # 20. Risk/Reward
    lines.append("")
    lines.append("[20. RISK / REWARD]")
    if rr_calc.get("checked"):
        lines.append(f"  • RR Ratio (TP1 / TP2) : {rr_calc.get('rr_display', 'N/A')} (TP2=1:{rr_calc.get('rr_tp2', 0):.2f})")
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

