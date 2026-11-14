# -*- coding: utf-8 -*-
"""
formatter.py — Penyusun Output Ringkas (Section U) dan Detail Teknis 22 Bagian Audit (Section V)
untuk SNR Advanced Scanner.
"""
from modules.snr.data_provider import safe_float


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

    h4_b = mb.get("h4", "UNCONFIRMED")
    h1_b = mb.get("h1", "UNCONFIRMED")
    m30_b = mb.get("m30", "UNCONFIRMED")
    m5_b = mb.get("m5", "WAIT")
    snr_status = mb.get("snr_status", "UNCONFIRMED")

    active_sups = ranked.get("active_supports") or []
    active_ress = ranked.get("active_resistances") or []

    lines = [
        "====================================================",
        "          XAUUSD SNR ADVANCED SCANNER",
        "====================================================",
        "",
        f"PRICE       : {price_str}",
        f"TIME        : {time_label}",
        "",
        "MULTI-TF MARKET BIAS",
        f"H4          : {h4_b}",
        f"H1          : {h1_b}",
        f"M30         : {m30_b}",
        f"M5          : {m5_b}",
        "",
        f"SNR STATUS  : {snr_status}",
        "",
        "----------------------------------------------------",
        "SUPPORT ZONES",
        "----------------------------------------------------",
    ]

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
    rej_label = pa.get("rejection_summary", "WAIT")
    mom_label = pa.get("momentum_summary", "N/A")

    buy_st = dec.get("buy_status", ev.get("buy_status", "WAIT"))
    sell_st = dec.get("sell_status", ev.get("sell_status", "WAIT"))
    zone_lbl = dec.get("zone_label", ev.get("zone_label", "N/A"))
    trig_st = dec.get("trigger_status", ev.get("trigger_status", "NOT CONFIRMED"))
    rr_st = dec.get("rr_display", ev.get("rr_status", "NOT CHECKED"))

    sig_final = dec.get("signal", "WAIT")
    dir_final = dec.get("direction", "NONE")
    reason_final = dec.get("reason", "Waiting for valid SNR confirmation.")

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
        "",
        "----------------------------------------------------",
        "FINAL DECISION",
        "----------------------------------------------------",
        f"SIGNAL     : {sig_final}",
        f"DIRECTION  : {dir_final}",
    ])

    if sig_final in ("BUY", "SELL") and dec.get("entry") is not None:
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
        f"  • Symbol       : {sr.get('symbol', 'XAUUSD')}",
        f"  • Price / Bid  : {safe_float(sr.get('price')):.2f} (Ask: {safe_float(sr.get('ask')):.2f}, Spread: {safe_float(sr.get('spread')):.2f})",
        f"  • Server Time  : {sr.get('server_time', 'N/A')} ({sr.get('time_label', 'N/A')})",
        f"  • Broker Spec  : Digits={spec.get('digits', 2)}, TickSize={spec.get('tick_size', 0.01)}, TickValue={spec.get('tick_value', 1.0)}, Source={spec.get('source', 'N/A')}",
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
    lines.append(f"  • Trigger      : {ev.get('trigger_status', 'NOT CONFIRMED')}")

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
    lines.append("")
    lines.append("[21. FINAL DECISION]")
    lines.append(f"  • SIGNAL    : {dec.get('signal', 'WAIT')}")
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

