# -*- coding: utf-8 -*-
"""
scanner.py — Mesin Utama (Orchestrator) SNR Advanced Scanner Independen.
Menjalankan seluruh tahapan analisis H4, H1, M30, dan M5 secara mandiri tanpa
bergantung pada maupun mengubah hasil SMC Scanner.
"""
from modules.snr.breakout_detection import detect_zone_breakout
from modules.snr.config import SNR_STRUCTURAL_TIMEFRAMES, SNR_TIMEFRAMES, get_snr_config
from modules.snr.data_provider import (
    fetch_snr_market_data,
    log_snr_error,
    safe_float,
)
from modules.snr.decision_engine import build_snr_final_decision
from modules.snr.entry_validator import validate_snr_entry
from modules.snr.formatter import format_snr_presentation
from modules.snr.market_bias import evaluate_snr_market_bias
from modules.snr.momentum_analysis import analyze_momentum
from modules.snr.radar import evaluate_snr_radar
from modules.snr.rejection_detection import detect_zone_rejection
from modules.snr.resistance_detection import detect_resistance_candidates
from modules.snr.retest_detection import detect_zone_retest
from modules.snr.support_detection import detect_support_candidates
from modules.snr.swing_detection import detect_swings
from modules.snr.touch_analysis import analyze_zone_touches
from modules.snr.zone_clustering import cluster_zones
from modules.snr.zone_ranking import rank_zones
from modules.snr.zone_scoring import evaluate_mtf_confluence, score_zone
from modules.snr.zone_validation import validate_single_zone


def _build_price_action_summary(focus_zone, ranked_zones, m5_df, frames_dict, momentum_by_tf, cfg):
    """Menyusun ringkasan PRICE ACTION (BREAKOUT, RETEST, REJECTION, MOMENTUM) untuk output utama."""
    eval_df = m5_df if (m5_df is not None and not getattr(m5_df, "empty", True)) else None
    if eval_df is None:
        for tf_fallback in ("M30", "H1", "H4"):
            cand_df = frames_dict.get(tf_fallback)
            if cand_df is not None and not getattr(cand_df, "empty", True):
                eval_df = cand_df
                break

    m5_mom = momentum_by_tf.get("M5") or {}
    if not m5_mom.get("available"):
        for tf_fallback in ("M30", "H1", "H4"):
            if (momentum_by_tf.get(tf_fallback) or {}).get("available"):
                m5_mom = momentum_by_tf[tf_fallback]
                break

    if focus_zone is not None and eval_df is not None:
        bo_res = detect_zone_breakout(focus_zone, eval_df, momentum_result=m5_mom, config=cfg)
        rt_res = detect_zone_retest(focus_zone, eval_df, breakout_result=bo_res, config=cfg)
        rej_res = detect_zone_rejection(focus_zone, eval_df, config=cfg)
    else:
        bo_res = {
            "status": "NO_BREAKOUT",
            "summary_status": "NOT CONFIRMED",
            "direction": "NONE",
            "penetration_points": 0.0,
            "penetration_atr": 0.0,
            "body_atr": 0.0,
            "momentum_supported": False,
            "volume_supported": "N/A",
            "reason": "Tidak ada zona fokus aktif untuk dievaluasi.",
        }
        rt_res = {
            "status": "NO_RETEST",
            "summary_status": "NOT CONFIRMED",
            "direction": "NONE",
            "role_reversal": "NONE",
            "retest_held": False,
            "reaction_confirmed": False,
            "reason": "Belum ada retest.",
        }
        rej_res = {
            "status": "NONE",
            "summary_status": "WAIT",
            "direction": "NONE",
            "confirmed": False,
            "upper_wick": 0.0,
            "lower_wick": 0.0,
            "body": 0.0,
            "wick_to_body": 0.0,
            "wick_to_range": 0.0,
            "in_zone_context": False,
            "next_candle_confirmed": False,
            "reason": "Belum ada pola rejection.",
        }

    # Cek juga apakah ada broken zone yang sedang mengalami breakout/retest aktif
    broken_zones = (ranked_zones or {}).get("broken_zones") or []
    if bo_res.get("status") == "NO_BREAKOUT" and broken_zones and eval_df is not None:
        bz0 = broken_zones[0]
        bo_bz = bz0.get("breakout_info") or detect_zone_breakout(bz0, eval_df, momentum_result=m5_mom, config=cfg)
        rt_bz = bz0.get("retest_info") or detect_zone_retest(bz0, eval_df, breakout_result=bo_bz, config=cfg)
        if bo_bz.get("status") in ("VALID_BREAKOUT", "FALSE_BREAKOUT", "BREAKOUT_PENDING_CONFIRMATION"):
            bo_res = bo_bz
        if rt_bz.get("status") in ("RETEST_CONFIRMED", "RETEST_IN_PROGRESS", "RETEST_FAILED"):
            rt_res = rt_bz

    return {
        "breakout_code": bo_res.get("status", "NO_BREAKOUT"),
        "breakout_direction": bo_res.get("direction", "NONE"),
        "breakout_summary": bo_res.get("summary_status", "NOT CONFIRMED"),
        "breakout_detail": bo_res,
        "retest_code": rt_res.get("status", "NO_RETEST"),
        "retest_direction": rt_res.get("direction", "NONE"),
        "retest_summary": rt_res.get("summary_status", "NOT CONFIRMED"),
        "retest_detail": rt_res,
        "rejection_code": rej_res.get("status", "NONE"),
        "rejection_direction": rej_res.get("direction", "NONE"),
        "rejection_summary": rej_res.get("summary_status", "WAIT"),
        "rejection_detail": rej_res,
        "momentum_code": m5_mom.get("direction", "NEUTRAL"),
        "momentum_strength": m5_mom.get("strength", "WEAK"),
        "momentum_summary": m5_mom.get("summary_label", "NEUTRAL"),
        "momentum_detail": m5_mom,
    }


def run_snr_scan(
    symbol=None,
    snapshot=None,
    frames=None,
    tick=None,
    live=True,
    has_unclosed_last_bar=False,
    config=None,
):
    """
    Menjalankan keseluruhan pipeline analisis SNR Advanced Scanner secara independen.
    Mengembalikan dict `scan_result` yang memuat seluruh hasil ringkas maupun detail 22 bagian.
    """
    cfg = get_snr_config(config)
    market_data = fetch_snr_market_data(
        symbol=symbol,
        snapshot=snapshot,
        frames=frames,
        tick=tick,
        live=live,
        has_unclosed_last_bar=has_unclosed_last_bar,
        config=cfg,
    )

    prepared_frames = market_data["frames"]
    current_price = safe_float(market_data.get("price"), 0.0)
    atr_diag = market_data["atr_diagnostics"]
    symbol_spec = market_data["symbol_spec"]

    swings_by_tf = {}
    support_candidates_by_tf = {}
    resistance_candidates_by_tf = {}
    clustered_zones_by_tf = {}
    validated_zones_by_tf = {}
    momentum_by_tf = {}

    for tf in SNR_TIMEFRAMES:
        df_tf = prepared_frames.get(tf)
        mom_tf = analyze_momentum(df_tf, timeframe=tf, config=cfg, has_unclosed_last_bar=False)
        momentum_by_tf[tf] = mom_tf

        sw_tf = detect_swings(df_tf, timeframe=tf, config=cfg, has_unclosed_last_bar=False)
        swings_by_tf[tf] = sw_tf

        sup_cands = detect_support_candidates(df_tf, swing_result=sw_tf, timeframe=tf, config=cfg)
        res_cands = detect_resistance_candidates(df_tf, swing_result=sw_tf, timeframe=tf, config=cfg)
        support_candidates_by_tf[tf] = sup_cands
        resistance_candidates_by_tf[tf] = res_cands

        tf_atr = safe_float((atr_diag.get(tf) or {}).get("atr14"), 2.0)
        clust_sup = cluster_zones(sup_cands, timeframe=tf, zone_type="SUPPORT", atr=tf_atr, config=cfg)
        clust_res = cluster_zones(res_cands, timeframe=tf, zone_type="RESISTANCE", atr=tf_atr, config=cfg)
        all_clust_tf = clust_sup + clust_res
        clustered_zones_by_tf[tf] = all_clust_tf

        val_tf_list = []
        for cz in all_clust_tf:
            touched = analyze_zone_touches(cz, df_tf, config=cfg, has_unclosed_last_bar=False)
            validated = validate_single_zone(
                touched,
                df=df_tf,
                current_price=current_price,
                config=cfg,
                has_unclosed_last_bar=False,
            )
            bo_z = detect_zone_breakout(validated, df_tf, momentum_result=mom_tf, config=cfg)
            rt_z = detect_zone_retest(validated, df_tf, breakout_result=bo_z, config=cfg)
            rej_z = detect_zone_rejection(validated, df_tf, config=cfg)
            validated["breakout_info"] = bo_z
            validated["retest_info"] = rt_z
            validated["rejection_info"] = rej_z
            val_tf_list.append(validated)

        validated_zones_by_tf[tf] = val_tf_list

    # Multi-Timeframe Confluence pada timeframe struktural (H4, H1, M30)
    confluence_res = evaluate_mtf_confluence(validated_zones_by_tf, config=cfg)
    zone_conf_map = confluence_res.get("zone_confluence_map") or {}

    # Scoring (0-100) untuk setiap zona
    scored_zones_by_tf = {}
    structural_scored_zones = []
    for tf in SNR_TIMEFRAMES:
        tf_scored = []
        mom_tf = momentum_by_tf.get(tf)
        for z in validated_zones_by_tf.get(tf, []):
            c_info = zone_conf_map.get(z.get("zone_id"))
            sz = score_zone(
                z,
                confluence_info=c_info,
                breakout_info=z.get("breakout_info"),
                retest_info=z.get("retest_info"),
                rejection_info=z.get("rejection_info"),
                momentum_info=mom_tf,
                config=cfg,
            )
            tf_scored.append(sz)
            if tf in SNR_STRUCTURAL_TIMEFRAMES:
                structural_scored_zones.append(sz)
        scored_zones_by_tf[tf] = tf_scored

    # Jika structural timeframes kosong (misal hanya diberikan M5 pada unit test spesifik),
    # gunakan zona M5 sebagai fallback ranking
    zones_for_ranking = structural_scored_zones if structural_scored_zones else scored_zones_by_tf.get("M5", [])
    ranked_zones = rank_zones(zones_for_ranking, current_price=current_price)

    # Market Bias awal dari H4, H1, M30
    market_bias = evaluate_snr_market_bias(
        frames_dict=prepared_frames,
        swings_by_tf=swings_by_tf,
        zones_by_tf=scored_zones_by_tf,
        momentum_by_tf=momentum_by_tf,
        m5_entry_state="WAIT",
        current_price=current_price,
    )

    # Entry Validation pada M5 (atau fallback timeframe terendah yang tersedia jika M5 kosong pada test)
    entry_df = prepared_frames.get("M5")
    if entry_df is None or getattr(entry_df, "empty", True):
        for tf_fb in ("M30", "H1", "H4"):
            if prepared_frames.get(tf_fb) is not None and not getattr(prepared_frames.get(tf_fb), "empty", True):
                entry_df = prepared_frames[tf_fb]
                break

    entry_validation = validate_snr_entry(
        ranked_zones=ranked_zones,
        m5_df=entry_df,
        market_bias=market_bias,
        current_price=current_price,
        symbol_spec=symbol_spec,
        config=cfg,
        has_unclosed_last_bar=False,
    )

    # Perbarui label M5 pada market_bias jika entry_validation mengonfirmasi arah BUY/SELL
    if entry_validation.get("final_status") == "BUY":
        market_bias["m5"] = "BULLISH"
        if "M5" in market_bias.get("per_timeframe", {}):
            market_bias["per_timeframe"]["M5"]["bias"] = "BULLISH"
    elif entry_validation.get("final_status") == "SELL":
        market_bias["m5"] = "BEARISH"
        if "M5" in market_bias.get("per_timeframe", {}):
            market_bias["per_timeframe"]["M5"]["bias"] = "BEARISH"

    focus_zone = entry_validation.get("target_zone")
    price_action = _build_price_action_summary(
        focus_zone=focus_zone,
        ranked_zones=ranked_zones,
        m5_df=entry_df,
        frames_dict=prepared_frames,
        momentum_by_tf=momentum_by_tf,
        cfg=cfg,
    )

    # Sinkronkan summary price action jika entry_validation sudah mengonfirmasi rejection/retest
    if entry_validation.get("setup_type") in ("SUPPORT_REJECTION", "RESISTANCE_REJECTION"):
        rej_det = entry_validation.get("rejection_detail") or price_action.get("rejection_detail") or {}
        price_action["rejection_code"] = rej_det.get("status", price_action["rejection_code"])
        price_action["rejection_summary"] = rej_det.get("summary_status", "CONFIRMED")
        price_action["rejection_detail"] = rej_det
    elif entry_validation.get("setup_type") in ("RESISTANCE_BREAKOUT_RETEST", "SUPPORT_BREAKDOWN_RETEST"):
        price_action["breakout_code"] = "VALID_BREAKOUT"
        price_action["breakout_summary"] = "CONFIRMED"
        price_action["retest_code"] = "RETEST_CONFIRMED"
        price_action["retest_summary"] = "CONFIRMED"

    decision = build_snr_final_decision(
        entry_validation=entry_validation,
        market_bias=market_bias,
        price_action=price_action,
    )

    radar = evaluate_snr_radar(
        ranked_zones=ranked_zones,
        decision=decision,
        price_action=price_action,
        current_price=current_price,
        config=cfg,
    )

    return {
        "method": "SNR",
        "symbol": market_data["symbol"],
        "price": market_data["price"],
        "bid": market_data["bid"],
        "ask": market_data["ask"],
        "mid": market_data["mid"],
        "spread": market_data["spread"],
        "time_label": market_data["time_label"],
        "server_time": market_data["server_time"],
        "symbol_spec": symbol_spec,
        "atr_diagnostics": atr_diag,
        "swings_by_tf": swings_by_tf,
        "support_candidates_by_tf": support_candidates_by_tf,
        "resistance_candidates_by_tf": resistance_candidates_by_tf,
        "clustered_zones_by_tf": clustered_zones_by_tf,
        "scored_zones_by_tf": scored_zones_by_tf,
        "confluence": confluence_res,
        "ranked_zones": ranked_zones,
        "momentum_by_tf": momentum_by_tf,
        "market_bias": market_bias,
        "price_action": price_action,
        "entry_validation": entry_validation,
        "decision": decision,
        "radar": radar,
    }


def evaluate_snr_decision(scan_result):
    """Mengambil atau membangun objek keputusan akhir dari hasil scan SNR."""
    if not isinstance(scan_result, dict):
        return build_snr_final_decision({})
    if isinstance(scan_result.get("decision"), dict):
        return scan_result["decision"]
    return build_snr_final_decision(
        scan_result.get("entry_validation"),
        scan_result.get("market_bias"),
        scan_result.get("price_action"),
    )


def scan_snr(
    symbol=None,
    snapshot=None,
    frames=None,
    tick=None,
    live=True,
    detailed=False,
    config=None,
):
    """
    Menjalankan scan SNR dan mengembalikan tuple `(scan_result, decision, formatted_output)`.
    """
    try:
        res = run_snr_scan(
            symbol=symbol,
            snapshot=snapshot,
            frames=frames,
            tick=tick,
            live=live,
            config=config,
        )
        dec = evaluate_snr_decision(res)
        txt = format_snr_presentation(res, decision=dec, detailed=detailed)
        return res, dec, txt
    except Exception as exc:
        log_snr_error("scan_snr", exc)
        raise


def scan_snr_detail(
    symbol=None,
    snapshot=None,
    frames=None,
    tick=None,
    live=True,
    config=None,
):
    """Shortcut untuk menjalankan `scan snr detail`."""
    return scan_snr(
        symbol=symbol,
        snapshot=snapshot,
        frames=frames,
        tick=tick,
        live=live,
        detailed=True,
        config=config,
    )

