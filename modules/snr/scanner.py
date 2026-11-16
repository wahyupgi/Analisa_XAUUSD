# -*- coding: utf-8 -*-
"""
scanner.py — Mesin Utama (Orchestrator) SNR Advanced Scanner Independen.
Menjalankan seluruh tahapan analisis H4, H1, M30, dan M5 secara mandiri tanpa
bergantung pada maupun mengubah hasil SMC Scanner.
"""
import json
import os

from modules.snr.breakout_detection import detect_zone_breakout
from modules.snr.config import (
    FILE_STATE_SNR_LIFECYCLE,
    SNR_LIFECYCLE_STATE_KEY,
    SNR_STRUCTURAL_TIMEFRAMES,
    SNR_TIMEFRAMES,
    get_snr_config,
)
from modules.snr.data_provider import (
    fetch_snr_market_data,
    log_snr_error,
    prepare_snr_frame,
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


def _snr_lifecycle_key(zone):
    formation = zone.get("formation_time") or zone.get("confirmed_time")
    if not formation:
        formation = zone.get("formation_index", zone.get("zone_id", "UNKNOWN"))
    return "|".join((
        str(zone.get("symbol", "")),
        str(zone.get("timeframe", "")).upper(),
        str(zone.get("zone_type", "")).upper(),
        str(formation),
        f"{safe_float(zone.get('low'), 0.0):.2f}",
        f"{safe_float(zone.get('high'), 0.0):.2f}",
    ))


def _load_snr_lifecycle_state():
    if not os.path.exists(FILE_STATE_SNR_LIFECYCLE):
        return {}
    try:
        with open(FILE_STATE_SNR_LIFECYCLE, "r", encoding="utf-8") as state_file:
            data = json.load(state_file)
        if not isinstance(data, dict):
            raise ValueError("Shared zone state harus berupa object JSON.")
        zones = data.get(SNR_LIFECYCLE_STATE_KEY, {})
        if not isinstance(zones, dict):
            raise ValueError("Field lifecycle SNR pada shared state harus berupa object.")
        if any(not isinstance(value, dict) for value in zones.values()):
            raise ValueError("Setiap record lifecycle SNR harus berupa object.")
        return zones
    except (OSError, ValueError, TypeError) as exc:
        log_snr_error("snr_lifecycle_load", exc)
        raise RuntimeError("Lifecycle SNR state tidak dapat dibaca; scan dihentikan demi fail-safe.") from exc


def _save_snr_lifecycle_state(zones):
    temporary_path = FILE_STATE_SNR_LIFECYCLE + ".snr.tmp"
    try:
        if os.path.exists(FILE_STATE_SNR_LIFECYCLE):
            with open(FILE_STATE_SNR_LIFECYCLE, "r", encoding="utf-8") as state_file:
                data = json.load(state_file)
            if not isinstance(data, dict):
                raise ValueError("Shared zone state harus berupa object JSON.")
        else:
            data = {}
        data[SNR_LIFECYCLE_STATE_KEY] = zones
        with open(temporary_path, "w", encoding="utf-8") as state_file:
            json.dump(data, state_file, indent=2, ensure_ascii=True)
        os.replace(temporary_path, FILE_STATE_SNR_LIFECYCLE)
    except (OSError, TypeError, ValueError) as exc:
        log_snr_error("snr_lifecycle_save", exc)
        raise RuntimeError("Lifecycle SNR state tidak dapat disimpan; scan dihentikan demi fail-safe.") from exc


def _apply_saved_lifecycle(zone, saved):
    saved_status = str((saved or {}).get("lifecycle_status", "")).upper()
    broken_states = ("BROKEN", "RETEST", "RECLAIM_PENDING", "RECLAIMED")
    if saved_status not in broken_states + ("INVALIDATED", "BREAKDOWN_PENDING", "UNCONFIRMED"):
        return
    current_status = str(zone.get("lifecycle_status", "")).upper()
    if current_status == "INVALIDATED" and saved_status != "INVALIDATED":
        return

    touch = dict(zone.get("touch_analysis") or {})
    if saved_status in broken_states:
        # Kunci touch count pada nilai pra-breakdown; jangan menambah sentuhan saat retest pasca-breakdown
        touch_count = int((saved or {}).get("touch_count", 0) or 0)
    else:
        touch_count = max(
            int(zone.get("touch_count", 0) or 0),
            int((saved or {}).get("touch_count", 0) or 0),
        )
    saved_touch_status = (
        "INVALIDATED" if saved_status == "INVALIDATED"
        else ("BROKEN" if saved_status in broken_states else saved_status)
    )
    touch.update({
        "classification": saved_touch_status,
        "valid_touch_count": touch_count,
        "is_broken": saved_status in broken_states,
        "is_invalidated": saved_status == "INVALIDATED",
        "reason": "Lifecycle terminal dipulihkan dari penyimpanan SNR.",
    })
    zone.update({
        "lifecycle_status": saved_status,
        "status": saved_status,
        "touch_status": saved_touch_status,
        "touch_count": touch_count,
        "is_broken": saved_status in broken_states,
        "is_invalidated": saved_status == "INVALIDATED",
        "is_active": saved_status in ("TOUCHED", "SWEEP", "ACTIVE"),
        "touch_analysis": touch,
        "lifecycle_breakout_info": (saved or {}).get("breakout_info"),
        "lifecycle_retest_origin": (saved or {}).get("retest_origin"),
        "lifecycle_event": (saved or {}).get("lifecycle_event", saved_status),
        "reclaim_status": (saved or {}).get("reclaim_status", "WAIT"),
    })


def _evaluate_post_break_state(zone, df, breakout_info, config, fallback_status="BROKEN"):
    """Classify closed-candle retest/reclaim without making a broken zone entry-active."""
    z_type = str(zone.get("zone_type", "SUPPORT")).upper()
    result = {
        "lifecycle_status": fallback_status,
        "lifecycle_event": fallback_status,
        "reclaim_status": "WAIT",
        "retest_status": "WAIT",
        "retest_origin": "FROM_BELOW" if z_type == "SUPPORT" else "FROM_ABOVE",
    }
    clean_df, _ = prepare_snr_frame(df, has_unclosed_last_bar=False)
    if clean_df.empty or not breakout_info:
        result["reclaim_status"] = "UNCONFIRMED"
        return result

    break_time = str(breakout_info.get("breakout_bar_time") or "")
    bar_times = [str(value) for value in clean_df["Time"].tolist()]
    try:
        break_idx = bar_times.index(break_time)
    except ValueError:
        result["reclaim_status"] = "UNCONFIRMED"
        return result

    post_break = clean_df.iloc[break_idx + 1:].reset_index(drop=True)
    if post_break.empty:
        result["reclaim_status"] = "UNCONFIRMED"
        return result

    low = safe_float(zone.get("low"), 0.0)
    high = safe_float(zone.get("high"), low)
    last = post_break.iloc[-1]
    last_close = safe_float(last.get("Close"), 0.0)
    overlap = (
        (post_break["High"] >= low) & (post_break["Low"] <= high)
    ).any()

    if z_type == "SUPPORT":
        reclaim_zone_type = "RESISTANCE"
        is_reclaim_pending = last_close > low
        is_retest = bool(overlap and last_close <= low)
    else:
        reclaim_zone_type = "SUPPORT"
        is_reclaim_pending = last_close < high
        is_retest = bool(overlap and last_close >= high)

    if is_reclaim_pending:
        reclaim_zone = dict(zone)
        reclaim_zone["zone_type"] = reclaim_zone_type
        reclaim_zone["confirmed_index"] = -1
        reclaim_zone["formation_index"] = -1
        reclaim = detect_zone_breakout(
            reclaim_zone,
            post_break,
            config=config,
            has_unclosed_last_bar=False,
        )
        result["lifecycle_event"] = "RETEST" if overlap else "RECLAIM_PENDING"
        if reclaim.get("status") == "VALID_BREAKOUT":
            result["lifecycle_status"] = "RECLAIMED"
            result["lifecycle_event"] = "RECLAIMED"
            result["reclaim_status"] = "CONFIRMED"
            result["retest_status"] = "CONFIRMED"
        else:
            result["lifecycle_status"] = "RECLAIM_PENDING"
            result["reclaim_status"] = "PENDING"
    elif is_retest:
        result["lifecycle_status"] = "RETEST"
        result["lifecycle_event"] = "RETEST"
        result["retest_status"] = result["retest_origin"]
    return result


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

    # Cek juga apakah ada broken zone yang sedang mengalami breakout valid / retest aktif
    # (Jangan mengambil FALSE_BREAKOUT dari zona lain ke fokus zona aktif agar status PA tidak rancu)
    focus_label = (
        f"{focus_zone.get('rank_label', focus_zone.get('zone_type', 'FOCUS'))} "
        f"({focus_zone.get('low', 0):.2f}-{focus_zone.get('high', 0):.2f})"
        if focus_zone is not None
        else "NONE"
    )
    bo_zone_label = focus_label
    rt_zone_label = focus_label
    rej_zone_label = focus_label
    broken_zones = (ranked_zones or {}).get("broken_zones") or []
    if bo_res.get("status") == "NO_BREAKOUT" and broken_zones and eval_df is not None:
        bz0 = broken_zones[0]
        bz_label = f"{bz0.get('rank_label', bz0.get('zone_type', 'BROKEN'))} ({bz0.get('low', 0):.2f}-{bz0.get('high', 0):.2f})"
        bo_bz = bz0.get("breakout_info") or detect_zone_breakout(bz0, eval_df, momentum_result=m5_mom, config=cfg)
        rt_bz = bz0.get("retest_info") or detect_zone_retest(bz0, eval_df, breakout_result=bo_bz, config=cfg)
        if bo_bz.get("status") in ("VALID_BREAKOUT", "BREAKOUT_PENDING_CONFIRMATION"):
            bo_res = bo_bz
            bo_zone_label = bz_label
            if rt_bz.get("status") in ("RETEST_CONFIRMED", "RETEST_IN_PROGRESS", "RETEST_FAILED"):
                rt_res = rt_bz
                rt_zone_label = bz_label

    # Pastikan koherensi breakout vs retest pada zona yang sama
    if bo_res.get("status") == "FALSE_BREAKOUT" and bo_zone_label == rt_zone_label and rt_res.get("status") == "RETEST_IN_PROGRESS":
        rt_res = dict(rt_res)
        rt_res["status"] = "NO_RETEST"
        rt_res["summary_status"] = "NOT CONFIRMED"

    n_bars = len(eval_df) if eval_df is not None else 0
    bo_bar_idx = bo_res.get("breakout_bar_index")
    bo_bars_ago = (n_bars - 1 - int(bo_bar_idx)) if (bo_bar_idx is not None and n_bars > 0) else 999
    # Breakout dianggap historical jika terjadi lebih dari 1 candle yang lalu atau berasal dari zona broken lama
    bo_is_historical = bool(bo_bars_ago > 1 or (bo_zone_label != focus_label and bo_bars_ago > 0))
    bo_detail = dict(bo_res)
    bo_detail["bars_ago"] = bo_bars_ago
    bo_detail["is_historical"] = bo_is_historical
    bo_detail["zone_label"] = bo_zone_label

    rt_bar_idx = rt_res.get("retest_bar_index")
    rt_bars_ago = (n_bars - 1 - int(rt_bar_idx)) if (rt_bar_idx is not None and n_bars > 0) else 999
    rt_is_historical = bool(rt_bars_ago > 2)
    rt_detail = dict(rt_res)
    rt_detail["bars_ago"] = rt_bars_ago
    rt_detail["is_historical"] = rt_is_historical
    rt_detail["zone_label"] = rt_zone_label

    return {
        "focus_zone_label": focus_label,
        "breakout_zone_label": bo_zone_label,
        "retest_zone_label": rt_zone_label,
        "rejection_zone_label": rej_zone_label,
        "breakout_code": bo_res.get("status", "NO_BREAKOUT"),
        "breakout_direction": bo_res.get("direction", "NONE"),
        "breakout_summary": bo_res.get("summary_status", "NOT CONFIRMED"),
        "breakout_is_historical": bo_is_historical,
        "breakout_detail": bo_detail,
        "retest_code": rt_res.get("status", "NO_RETEST"),
        "retest_direction": rt_res.get("direction", "NONE"),
        "retest_summary": rt_res.get("summary_status", "NOT CONFIRMED"),
        "retest_is_historical": rt_is_historical,
        "retest_detail": rt_detail,
        "rejection_code": rej_res.get("status", "NONE"),
        "rejection_direction": rej_res.get("direction", "NONE"),
        "rejection_summary": rej_res.get("summary_status", "NOT CONFIRMED"),
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
    market_status=None,
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
        market_status=market_status,
        config=cfg,
    )

    prepared_frames = market_data["frames"]
    current_price = safe_float(market_data.get("price"), 0.0)
    atr_diag = market_data["atr_diagnostics"]
    symbol_spec = market_data["symbol_spec"]
    mkt_status = market_data.get("market_status", "OPEN")
    snr_analysis = market_data.get("snr_analysis", "AVAILABLE")

    swings_by_tf = {}
    support_candidates_by_tf = {}
    resistance_candidates_by_tf = {}
    clustered_zones_by_tf = {}
    validated_zones_by_tf = {}
    momentum_by_tf = {}
    lifecycle_state = _load_snr_lifecycle_state()

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
            validated.setdefault("symbol", market_data.get("symbol", symbol or cfg.get("symbol", "")))
            lifecycle_key = _snr_lifecycle_key(validated)
            saved_lifecycle = lifecycle_state.get(lifecycle_key) or {}
            _apply_saved_lifecycle(validated, saved_lifecycle)
            bo_z = detect_zone_breakout(validated, df_tf, momentum_result=mom_tf, config=cfg)
            rt_z = detect_zone_retest(validated, df_tf, breakout_result=bo_z, config=cfg)
            rej_z = detect_zone_rejection(validated, df_tf, config=cfg)
            current_lifecycle = str(validated.get("lifecycle_status", "ACTIVE")).upper()
            current_side = (
                "BELOW" if current_price < safe_float(validated.get("low"), 0.0)
                else "ABOVE" if current_price > safe_float(validated.get("high"), 0.0)
                else "INSIDE"
            )
            breakout_confirmed = (
                bo_z.get("status") == "VALID_BREAKOUT"
                or bool(bo_z.get("prior_confirmed_breakout"))
            )
            locked_states = ("BROKEN", "RETEST", "RECLAIM_PENDING", "RECLAIMED")
            breakout_side = "BELOW" if str(validated.get("zone_type", "")).upper() == "SUPPORT" else "ABOVE"
            if (
                breakout_confirmed
                and current_lifecycle in ("RETEST", "RECLAIM_PENDING")
                and current_side == breakout_side
            ):
                validated["lifecycle_status"] = "BROKEN"
                validated["status"] = "BROKEN"
                validated["is_broken"] = True
                validated["is_active"] = False
                validated["touch_status"] = "BROKEN"
                current_lifecycle = "BROKEN"
            elif breakout_confirmed and current_lifecycle not in locked_states + ("INVALIDATED",):
                validated["lifecycle_status"] = "BROKEN"
                validated["status"] = "BROKEN"
                validated["is_broken"] = True
                validated["is_active"] = False
                validated["touch_status"] = "BROKEN"
                touch = dict(validated.get("touch_analysis") or {})
                touch["classification"] = "BROKEN"
                touch["is_broken"] = True
                touch["reason"] = "Breakout terkonfirmasi oleh candle close dan validasi momentum."
                validated["touch_analysis"] = touch
                current_lifecycle = "BROKEN"
            elif bo_z.get("status") == "BREAKOUT_PENDING_CONFIRMATION" and current_lifecycle not in locked_states + ("INVALIDATED",):
                validated["lifecycle_status"] = "BREAKDOWN_PENDING"
                validated["status"] = "BREAKDOWN_PENDING"
                validated["is_active"] = False
                current_lifecycle = "BREAKDOWN_PENDING"

            breakout_info = (
                bo_z if breakout_confirmed
                else (
                    bo_z
                    if current_lifecycle == "BREAKDOWN_PENDING"
                    and bo_z.get("status") in ("FALSE_BREAKOUT", "BREAKOUT_PENDING_CONFIRMATION")
                    else saved_lifecycle.get("breakout_info") or validated.get("lifecycle_breakout_info")
                )
            )
            post_break_state = None
            if current_lifecycle in locked_states:
                if current_lifecycle == "RECLAIMED":
                    post_break_state = {
                        "lifecycle_status": "RECLAIMED",
                        "lifecycle_event": "RECLAIMED",
                        "reclaim_status": "CONFIRMED",
                        "retest_status": "CONFIRMED",
                        "retest_origin": "FROM_BELOW" if str(validated.get("zone_type")).upper() == "SUPPORT" else "FROM_ABOVE",
                    }
                else:
                    post_break_state = _evaluate_post_break_state(
                        validated,
                        df_tf,
                        breakout_info,
                        cfg,
                        fallback_status=current_lifecycle,
                    )
                    validated["lifecycle_status"] = post_break_state["lifecycle_status"]
                    validated["status"] = post_break_state["lifecycle_status"]
                    validated["is_active"] = False
                    current_lifecycle = post_break_state["lifecycle_status"]
            elif current_lifecycle == "BREAKDOWN_PENDING" and breakout_info:
                returned_to_zone = (
                    current_side != "BELOW"
                    if str(validated.get("zone_type", "")).upper() == "SUPPORT"
                    else current_side != "ABOVE"
                )
                if returned_to_zone or bo_z.get("status") == "FALSE_BREAKOUT":
                    post_break_state = _evaluate_post_break_state(
                        validated,
                        df_tf,
                        breakout_info,
                        cfg,
                        fallback_status="BREAKDOWN_PENDING",
                    )
                    validated["lifecycle_status"] = post_break_state["lifecycle_status"]
                    validated["status"] = post_break_state["lifecycle_status"]
                    validated["is_active"] = False
                    current_lifecycle = post_break_state["lifecycle_status"]

            retest_origin = validated.get("lifecycle_retest_origin") or saved_lifecycle.get("retest_origin")
            if post_break_state and post_break_state.get("retest_origin"):
                retest_origin = post_break_state["retest_origin"]
            if (
                not retest_origin
                and rt_z.get("status") in ("RETEST_IN_PROGRESS", "RETEST_CONFIRMED")
            ):
                retest_origin = rt_z.get("retest_origin")
            if current_lifecycle in locked_states:
                z_type = str(validated.get("zone_type", "")).upper()
                previous_side = str(saved_lifecycle.get("last_side", "")).upper()
                if not retest_origin and z_type == "SUPPORT" and previous_side == "BELOW" and current_side != "BELOW":
                    retest_origin = "FROM_BELOW"
                elif not retest_origin and z_type == "RESISTANCE" and previous_side == "ABOVE" and current_side != "ABOVE":
                    retest_origin = "FROM_ABOVE"
                elif not retest_origin and bo_z.get("prior_confirmed_breakout") and current_side != "BELOW" and z_type == "SUPPORT":
                    retest_origin = "FROM_BELOW"
                elif not retest_origin and bo_z.get("prior_confirmed_breakout") and current_side != "ABOVE" and z_type == "RESISTANCE":
                    retest_origin = "FROM_ABOVE"
                if retest_origin:
                    rt_z["retest_origin"] = retest_origin
                validated["lifecycle_retest_origin"] = retest_origin
                if post_break_state:
                    validated["lifecycle_event"] = post_break_state["lifecycle_event"]
                    validated["reclaim_status"] = post_break_state["reclaim_status"]
                    validated["retest_status"] = post_break_state["retest_status"]
            else:
                validated["lifecycle_event"] = validated.get("lifecycle_event", current_lifecycle)
                validated["reclaim_status"] = validated.get("reclaim_status", "WAIT")
                validated["retest_status"] = "WAIT"

            validated["breakout_info"] = bo_z
            validated["retest_info"] = rt_z
            validated["rejection_info"] = rej_z
            lifecycle_state[lifecycle_key] = {
                "lifecycle_status": validated.get("lifecycle_status", "ACTIVE"),
                "touch_count": int(validated.get("touch_count", 0) or 0),
                "last_side": current_side,
                "breakout_info": breakout_info,
                "retest_origin": retest_origin,
                "lifecycle_event": validated.get("lifecycle_event", "ACTIVE"),
                "reclaim_status": validated.get("reclaim_status", "WAIT"),
            }
            val_tf_list.append(validated)

        validated_zones_by_tf[tf] = val_tf_list

    if len(lifecycle_state) > 2000:
        lifecycle_state = dict(list(lifecycle_state.items())[-2000:])
    _save_snr_lifecycle_state(lifecycle_state)

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

    if mkt_status == "DATA UNAVAILABLE":
        market_bias["h4"] = "DATA UNAVAILABLE"
        market_bias["h1"] = "DATA UNAVAILABLE"
        market_bias["m30"] = "DATA UNAVAILABLE"
        market_bias["m5"] = "DATA UNAVAILABLE"
        market_bias["overall_bias"] = "DATA UNAVAILABLE"
        market_bias["snr_status"] = "DATA UNAVAILABLE"

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
        data_freshness=market_data.get("data_freshness", "FRESH"),
    )

    # Perbarui label M5 pada market_bias hanya ketika market OPEN dan entry_validation mengonfirmasi arah BUY/SELL
    if mkt_status == "OPEN":
        if entry_validation.get("final_status") == "BUY":
            market_bias["m5"] = "BULLISH"
            if "M5" in market_bias.get("per_timeframe", {}):
                market_bias["per_timeframe"]["M5"]["bias"] = "BULLISH"
        elif entry_validation.get("final_status") == "SELL":
            market_bias["m5"] = "BEARISH"
            if "M5" in market_bias.get("per_timeframe", {}):
                market_bias["per_timeframe"]["M5"]["bias"] = "BEARISH"
    elif mkt_status == "CLOSED":
        market_bias["m5"] = "WAIT"
        if "M5" in market_bias.get("per_timeframe", {}):
            market_bias["per_timeframe"]["M5"]["bias"] = "WAIT"

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

    # Pemisahan analisis historis vs klaim live ketika market CLOSED / DATA UNAVAILABLE
    if mkt_status == "CLOSED":
        if price_action.get("breakout_summary") == "CONFIRMED":
            price_action["breakout_summary"] = "HISTORICAL (CLOSED CANDLE)"
        if price_action.get("retest_summary") == "CONFIRMED":
            price_action["retest_summary"] = "HISTORICAL (CLOSED CANDLE)"
        if price_action.get("rejection_summary") == "CONFIRMED":
            price_action["rejection_summary"] = "HISTORICAL (CLOSED CANDLE)"

        entry_validation["historical_final_status"] = entry_validation.get("final_status", "WAIT")
        entry_validation["final_status"] = "WAIT"
        entry_validation["buy_status"] = "WAIT — MARKET CLOSED"
        entry_validation["sell_status"] = "WAIT — MARKET CLOSED"
        entry_validation["trigger_status"] = "WAIT — MARKET CLOSED"
    elif mkt_status == "DATA UNAVAILABLE":
        price_action["breakout_summary"] = "DATA UNAVAILABLE"
        price_action["retest_summary"] = "DATA UNAVAILABLE"
        price_action["rejection_summary"] = "DATA UNAVAILABLE"
        price_action["momentum_summary"] = "DATA UNAVAILABLE"
        entry_validation["final_status"] = "WAIT"
        entry_validation["buy_status"] = "DATA UNAVAILABLE"
        entry_validation["sell_status"] = "DATA UNAVAILABLE"
        entry_validation["trigger_status"] = "DATA UNAVAILABLE"

    decision = build_snr_final_decision(
        entry_validation=entry_validation,
        market_bias=market_bias,
        price_action=price_action,
        market_status=mkt_status,
        snr_analysis=snr_analysis,
    )

    radar = evaluate_snr_radar(
        ranked_zones=ranked_zones,
        decision=decision,
        price_action=price_action,
        current_price=current_price,
        config=cfg,
    )

    scan_payload = {
        "method": "SNR",
        "symbol": market_data["symbol"],
        "snapshot_id": market_data.get("snapshot_id"),
        "snapshot_time": market_data.get("snapshot_time"),
        "snapshot_price": market_data.get("snapshot_price", market_data["price"]),
        "price": market_data["price"],
        "last_price": market_data.get("last_price", market_data["price"]),
        "last_candle": market_data.get("last_candle", "DATA UNAVAILABLE"),
        "last_candle_times": market_data.get("last_candle_times", {}),
        "last_closed_m5": market_data.get("last_closed_m5", ""),
        "current_candle_m5": market_data.get("current_candle_m5", ""),
        "tick_time": market_data.get("tick_time", ""),
        "forming_candles": market_data.get("forming_candles", {}),
        "market_status": mkt_status,
        "market_status_label": market_data.get("market_status_label", mkt_status),
        "data_source": market_data.get("data_source", "SNAPSHOT"),
        "connection_status": market_data.get("connection_status", "UNKNOWN"),
        "data_freshness": market_data.get("data_freshness", "FRESH"),
        "data_freshness_info": market_data.get("data_freshness_info", {}),
        "snr_analysis": snr_analysis,
        "live_entry": decision.get("live_entry", "WAIT"),
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
    try:
        from modules.trend_following import build_unified_market_context
        scan_payload["market_context"] = build_unified_market_context(
            snr_scan=scan_payload,
            snr_dec=decision,
        )
    except Exception:
        scan_payload["market_context"] = {}
    return scan_payload


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
        market_status=scan_result.get("market_status", "OPEN"),
        snr_analysis=scan_result.get("snr_analysis", "AVAILABLE"),
    )


def scan_snr(
    symbol=None,
    snapshot=None,
    frames=None,
    tick=None,
    live=True,
    detailed=False,
    market_status=None,
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
            market_status=market_status,
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
    market_status=None,
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
        market_status=market_status,
        config=config,
    )

