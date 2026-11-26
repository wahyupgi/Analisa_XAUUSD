# -*- coding: utf-8 -*-
"""
scanner.py — Mesin Utama (Orchestrator) SNR Advanced Scanner Independen.
Menjalankan seluruh tahapan analisis H4, H1, M30, dan M5 secara mandiri tanpa
bergantung pada maupun mengubah hasil SMC Scanner.
"""
import json
import os

from modules.snr.breakout_detection import build_snr_breakout_watch, detect_zone_breakout
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
from modules.snr.retest_detection import classify_snr_flip_status, detect_zone_retest
from modules.snr.risk_reward import calculate_snr_risk_reward, opposing_risk_zones
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
    anchor = zone.get("anchor_price")
    # Batas zona ikut bergeser bersama ATR; pivot anchor adalah identitas stabil bila tersedia.
    bounds = (
        (f"{safe_float(anchor, 0.0):.2f}",)
        if anchor is not None
        else (f"{safe_float(zone.get('low'), 0.0):.2f}", f"{safe_float(zone.get('high'), 0.0):.2f}")
    )
    return "|".join((
        str(zone.get("symbol", "")),
        str(zone.get("timeframe", "")).upper(),
        str(zone.get("zone_type", "")).upper(),
        str(formation),
        *bounds,
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
    temporary_path = FILE_STATE_SNR_LIFECYCLE + f".snr.{os.getpid()}.tmp"
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
        for attempt in range(4):
            try:
                os.replace(temporary_path, FILE_STATE_SNR_LIFECYCLE)
                break
            except PermissionError:
                if attempt == 3:
                    raise
                time.sleep(0.05)
    except (OSError, TypeError, ValueError) as exc:
        log_snr_error("snr_lifecycle_save", exc)
        raise RuntimeError("Lifecycle SNR state tidak dapat disimpan; scan dihentikan demi fail-safe.") from exc


SNR_FOCUS_STATE_KEY = "snr_focus_state"


def _zone_brief(zone):
    label = zone.get("rank_label") or str(zone.get("zone_type", "ZONE"))[:1]
    return f"{label} {zone.get('timeframe', '')} {safe_float(zone.get('low')):.2f}-{safe_float(zone.get('high')):.2f}".strip()


def _track_snr_focus(focus_zone, ranked_zones, selection_reason=None):
    """Persist the focused SNR zone and explain a focus change; non-critical, never blocks a scan."""
    if not isinstance(focus_zone, dict) or not focus_zone.get("zone_id"):
        return None
    try:
        data = {}
        if os.path.exists(FILE_STATE_SNR_LIFECYCLE):
            with open(FILE_STATE_SNR_LIFECYCLE, "r", encoding="utf-8") as state_file:
                data = json.load(state_file)
        if not isinstance(data, dict):
            return None
        saved = data.get(SNR_FOCUS_STATE_KEY) if isinstance(data.get(SNR_FOCUS_STATE_KEY), dict) else {}
        current_id = focus_zone["zone_id"]
        if saved.get("zone_id") == current_id:
            return saved.get("change")
        change = None
        if saved.get("zone_id"):
            by_id = {z.get("zone_id"): z for z in (ranked_zones or {}).get("all_zones") or [] if z.get("zone_id")}
            previous = by_id.get(saved["zone_id"])
            if previous is not None:
                status = str(previous.get("lifecycle_status", previous.get("status", "ACTIVE"))).upper()
                active_ids = {
                    z.get("zone_id") for z in
                    ((ranked_zones or {}).get("active_supports") or []) + ((ranked_zones or {}).get("active_resistances") or [])
                }
                if status not in ("ACTIVE", "TOUCHED", "SWEEP"):
                    reason = f"PREVIOUS ZONE {status}"
                elif saved["zone_id"] in active_ids:
                    sel_head = str(selection_reason or "").split(" (", 1)[0].strip()
                    reason = sel_head if sel_head else "FOCUS RANKING OR SETUP CHANGED"
                else:
                    reason = "PREVIOUS ZONE NOT IN ACTIVE RANKING"
            else:
                rebuilt = next((
                    z for z in by_id.values()
                    if z.get("timeframe") == saved.get("timeframe") and z.get("zone_type") == saved.get("zone_type")
                    and min(safe_float(z.get("high")), safe_float(saved.get("high"))) >= max(safe_float(z.get("low")), safe_float(saved.get("low")))
                ), None)
                reason = "PREVIOUS ZONE REBUILT" if rebuilt is not None else "PREVIOUS ZONE NOT DETECTED"
            change = {"previous": saved.get("label", "N/A"), "reason": reason, "to": _zone_brief(focus_zone)}
        data[SNR_FOCUS_STATE_KEY] = {
            "zone_id": current_id, "label": _zone_brief(focus_zone), "timeframe": focus_zone.get("timeframe"),
            "zone_type": focus_zone.get("zone_type"), "low": focus_zone.get("low"), "high": focus_zone.get("high"),
            "change": change,
        }
        temporary_path = FILE_STATE_SNR_LIFECYCLE + f".snr_foc.{os.getpid()}.tmp"
        with open(temporary_path, "w", encoding="utf-8") as state_file:
            json.dump(data, state_file, indent=2, ensure_ascii=True)
        for attempt in range(4):
            try:
                os.replace(temporary_path, FILE_STATE_SNR_LIFECYCLE)
                break
            except PermissionError:
                if attempt == 3:
                    raise
                time.sleep(0.05)
        return change
    except (OSError, TypeError, ValueError) as exc:
        log_snr_error("snr_focus_state", exc)
        return None


def _build_snr_provisional_plan(entry_validation, ranked_zones, market_bias, current_price, symbol_spec, cfg, blocked=None):
    """Provisional SL/TP for the focused zone using the existing SNR risk module; never a signal."""
    zone = entry_validation.get("target_zone")
    plan = {"status": "N/A", "direction": "NONE", "note": "NO FOCUS ZONE"}
    if not isinstance(zone, dict):
        return plan
    z_type = str(zone.get("zone_type", "")).upper()
    direction = "SELL" if z_type == "RESISTANCE" else "BUY" if z_type == "SUPPORT" else "NONE"
    plan["direction"] = direction
    lifecycle = str(zone.get("lifecycle_status", zone.get("status", "ACTIVE"))).upper()
    h4 = str((market_bias or {}).get("h4", "")).upper()
    if entry_validation.get("final_status") in ("BUY", "SELL"):
        plan["note"] = "CONFIRMED PLAN"
        return plan
    if blocked:
        plan["note"] = blocked
    elif direction == "NONE":
        plan["note"] = "ZONE TYPE UNKNOWN"
    elif (
        str(zone.get("zone_relevance") or "").upper() in ("FAR_REFERENCE", "INVALID")
        or zone.get("is_far_reference")
        or zone.get("is_distant_historical")
    ):
        plan["note"] = "FAR REFERENCE ONLY (WAIT)"
    elif lifecycle not in ("ACTIVE", "TOUCHED", "SWEEP"):
        plan["note"] = f"ZONE {lifecycle}"
    elif (direction == "SELL" and h4 == "BULLISH") or (direction == "BUY" and h4 == "BEARISH"):
        plan["note"] = "COUNTER-TREND (H4 FILTER)"
    elif (market_bias or {}).get("has_conflict") or (market_bias or {}).get("m30_conflict") or str((market_bias or {}).get("overall_bias", "")).upper() in ("CONFLICT", "UNCONFIRMED", "DATA UNAVAILABLE"):
        plan["note"] = "BIAS CONFLICT OR UNCONFIRMED"
    elif safe_float((symbol_spec or {}).get("spread_points"), 0.0) > safe_float(cfg.get("risk", {}).get("max_spread_points"), 1.0):
        plan["note"] = "SPREAD TOO HIGH"
    else:
        low = safe_float(zone.get("low"), 0.0)
        high = safe_float(zone.get("high"), 0.0)
        edge = low if direction == "SELL" else high
        price = safe_float(current_price, 0.0)
        inside = low > 0 and low <= price <= high
        if price <= 0 or edge <= 0 or high < low:
            plan["note"] = "PRICE UNAVAILABLE"
        elif not inside and ((direction == "SELL" and price > high) or (direction == "BUY" and price < low)):
            plan["note"] = "PRICE PASSED ZONE"
        elif inside and str(entry_validation.get("final_status", "")).upper() == "INVALID":
            plan["note"] = "SETUP INVALID"
        elif inside and (entry_validation.get("rr_calculation") or {}).get("checked") and not (entry_validation.get("rr_calculation") or {}).get("valid"):
            plan["note"] = (entry_validation.get("rr_calculation") or {}).get("reason") or "ENTRY RISK INVALID"
        elif inside and not (symbol_spec or {}).get("quote_valid"):
            plan["note"] = "BID/ASK UNAVAILABLE"
        elif inside and ((symbol_spec or {}).get("stops_level") is None or not (symbol_spec or {}).get("tick_size_verified")):
            plan["note"] = "BROKER STOP LEVEL OR TICK SIZE UNAVAILABLE"
        else:
            entry_ref = price if inside else edge
            opposing = opposing_risk_zones(ranked_zones, direction)
            rr = calculate_snr_risk_reward(
                direction=direction, entry_price=entry_ref, active_zone=zone, opposing_zones=opposing,
                trigger_candle=None, symbol_spec=symbol_spec, config=cfg,
            )
            spec_digits = int((symbol_spec or {}).get("digits", 2))
            plan.update(
                entry=rr.get("effective_entry"),
                distance=round(abs(entry_ref - price), spec_digits),
                reference="CURRENT PRICE" if inside else "ZONE EDGE",
                rr_calculation=rr,
            )
            if rr.get("valid"):
                plan.update(status="PROVISIONAL", note="PROVISIONAL ONLY", sl=rr["sl"], tp1=rr["tp1"], tp2=rr["tp2"], rr=rr["rr_tp1"])
            else:
                plan["note"] = rr.get("reason") or rr.get("rr_display") or "PLAN UNAVAILABLE"
    return plan


def generate_snr_trade_plan(
    zone,
    ranked_zones,
    entry_validation,
    decision,
    market_bias,
    current_price,
    symbol_spec,
    cfg,
    market_status="OPEN",
    data_freshness="FRESH",
    m5_df=None,
    mt5_orders_snapshot=None,
):
    """
    Trade Plan Generator deterministik untuk satu zona SNR ACTIVE/RADAR.
    Mendelegasikan ke generator bersama `build_shared_trade_plan` dengan kalkulasi SL/TP existing
    (`calculate_snr_risk_reward` & `opposing_risk_zones`) agar level pending limit stabil per Zone ID.
    """
    from modules.config import SL_JARAK_MAKS
    from modules.snr.data_provider import prepare_snr_frame, zone_confirmation_start
    from modules.snr.entry_validator import _zone_touch_provenance
    from modules.snr.formatter import build_snr_zone_reason
    from modules.utils import build_shared_trade_plan, format_wib_timestamp

    z = zone if isinstance(zone, dict) else {}
    zid = str(z.get("zone_id") or "").strip()
    tf = str(z.get("timeframe") or "H1").strip().upper()
    z_type = str(z.get("zone_type") or "").strip().upper()
    direction = "BUY" if z_type == "SUPPORT" else ("SELL" if z_type == "RESISTANCE" else "NONE")
    grd = str(z.get("grade") or "N/A").strip()
    sc = safe_float(z.get("score"), 0.0)
    lo = safe_float(z.get("low"), 0.0)
    hi = safe_float(z.get("high"), lo)
    price = safe_float(current_price, 0.0)
    mb = market_bias if isinstance(market_bias, dict) else {}
    h4 = str(mb.get("h4") or "").upper()
    why_txt = build_snr_zone_reason(z, h4_bias=h4, h1_bias=mb.get("h1"))

    spec = symbol_spec if isinstance(symbol_spec, dict) else {}
    digits = int(spec.get("digits", 2))
    point = safe_float(spec.get("point"), 0.01)
    stops_lvl = spec.get("trade_stops_level")
    min_stop = (float(stops_lvl) * point) if (isinstance(stops_lvl, (int, float)) and stops_lvl > 0) else 0.0
    struct_inv = safe_float(z.get("structure_invalidation") or z.get("invalidation_price"), 0.0)
    if struct_inv <= 0 and lo > 0:
        struct_inv = lo if direction == "BUY" else (hi if direction == "SELL" else 0.0)

    mkt_u = str(market_status or "OPEN").upper()
    fresh_u = str(data_freshness or "FRESH").upper()
    data_unavail = bool(mkt_u == "DATA UNAVAILABLE" or fresh_u in ("STALE", "INVALID", "NO DATA"))
    is_closed = mkt_u in ("CLOSED", "MARKET_CLOSED")

    lifecycle = str(z.get("lifecycle_status") or z.get("status") or "ACTIVE").upper()
    bo_info = z.get("breakout_info") if isinstance(z.get("breakout_info"), dict) else {}
    bo_st = str(bo_info.get("status") or "").upper()
    is_bo_pending = bool(
        (lifecycle in ("BREAKOUT_PENDING", "BREAKDOWN_PENDING") or bo_st == "BREAKOUT_PENDING_CONFIRMATION")
        and not z.get("is_broken")
        and not z.get("is_invalidated")
        and lifecycle not in ("BROKEN", "INVALIDATED", "EXPIRED")
    )
    is_expired_z = bool(lifecycle == "EXPIRED" or z.get("is_expired"))
    is_invalid_z = bool(
        (not is_expired_z) and (not is_bo_pending) and (
            lifecycle not in ("ACTIVE", "TOUCHED", "SWEEP")
            or z.get("is_active") is False
            or z.get("is_broken") is True
            or z.get("is_invalidated") is True
        )
    )
    is_far_z = bool(
        (not is_invalid_z) and (
            str(z.get("zone_relevance") or "").upper() == "FAR_REFERENCE"
            or z.get("is_far_reference")
            or z.get("is_distant_historical")
        )
    )

    risk_cfg = (cfg or {}).get("risk", {}) if isinstance(cfg, dict) else {}
    spread_pts = safe_float(spec.get("spread_points"), 0.0)
    max_spr = safe_float(risk_cfg.get("max_spread_points"), 1.0)
    min_rr = safe_float(risk_cfg.get("min_rr_ratio"), 1.5)
    bid_v = safe_float(spec.get("bid"), 0.0) or None
    ask_v = safe_float(spec.get("ask"), 0.0) or None

    # 1. Existing CONFIRMED SNR decision valid untuk zona ini
    dec = decision if isinstance(decision, dict) else {}
    ev = entry_validation if isinstance(entry_validation, dict) else {}
    dec_sig = str(dec.get("signal") or "WAIT").upper()
    dec_tz = dec.get("target_zone") or ev.get("target_zone") or {}
    confirmed_payload = None
    if not is_closed and dec_sig == direction and zid and str(dec_tz.get("zone_id") or "") == zid:
        ce = safe_float(dec.get("effective_entry") or dec.get("entry"), 0.0)
        csl = safe_float(dec.get("sl"), 0.0)
        ctp1 = safe_float(dec.get("tp1"), 0.0)
        ctp2 = safe_float(dec.get("tp2"), 0.0)
        if (csl < ce < ctp1) if direction == "BUY" else (csl > ce > ctp1 > 0):
            confirmed_payload = {
                "entry": ce,
                "sl": csl,
                "tp1": ctp1,
                "tp2": ctp2,
                "struct_inv": safe_float((ev.get("rr_calculation") or {}).get("structural_invalidation"), struct_inv),
            }

    htf_conflict = bool(mb.get("has_conflict") or str(mb.get("overall_bias") or "").upper() == "CONFLICT")
    m30_conflict = bool(mb.get("m30_conflict"))
    counter_htf = bool((direction == "SELL" and h4 == "BULLISH") or (direction == "BUY" and h4 == "BEARISH"))
    if is_closed:
        cond_rsn = "MARKET_CLOSED"
    elif htf_conflict:
        cond_rsn = "HTF_CONFLICT"
    elif m30_conflict:
        cond_rsn = "M30_CONFLICT"
    elif counter_htf:
        cond_rsn = "COUNTER_HTF"
    else:
        cond_rsn = "WAIT_M5_CONFIRM"

    opposing = opposing_risk_zones(ranked_zones, direction)

    def _compute_snr_levels(planned_e):
        rr = calculate_snr_risk_reward(
            direction=direction,
            entry_price=planned_e,
            active_zone=z,
            opposing_zones=opposing,
            trigger_candle=None,
            symbol_spec=spec,
            config=cfg,
        )
        if not rr.get("valid"):
            rr_disp = str(rr.get("rr_display") or "").upper()
            if rr.get("rr_failed") and rr.get("rr_tp1") is not None:
                min_r = safe_float(rr.get("min_rr_required"), min_rr)
                return {"ok": False, "reason": f"RR_TOO_LOW (1:{safe_float(rr['rr_tp1']):.2f}<1:{min_r:g})"}
            if "TP N/A" in rr_disp:
                return {"ok": False, "reason": "NO_STRUCTURAL_TARGET"}
            if "STOP" in str(rr.get("reason") or "").upper():
                return {"ok": False, "reason": "BELOW_MT5_STOP_LEVEL"}
            if "SL" in rr_disp:
                return {"ok": False, "reason": rr_disp.replace(" ", "_")}
            return {"ok": False, "reason": "PLAN_INVALID"}
        return {
            "ok": True,
            "setup_sl": safe_float(rr.get("sl"), 0.0),
            "struct_inv": safe_float(rr.get("structural_invalidation"), struct_inv),
            "tp1": safe_float(rr.get("tp1"), 0.0),
            "tp2": safe_float(rr.get("tp2"), 0.0),
        }

    # Ekstrak bukti touch aktual & jendela validitas existing (max_entry_distance_atr & 4 bar M5 closed)
    clean_m5_tp = None
    if m5_df is not None:
        try:
            clean_m5_tp, _ = prepare_snr_frame(m5_df, has_unclosed_last_bar=False)
        except Exception:
            clean_m5_tp = None
    was_touched_snr = False
    touch_time_snr = None
    bars_since_touch_snr = None
    sl_breached_hist_snr = False
    if clean_m5_tp is not None and not clean_m5_tp.empty and lo > 0 and hi >= lo:
        try:
            start_i = max(0, int(zone_confirmation_start(z, clean_m5_tp)))
            post_m5 = clean_m5_tp.iloc[start_i:]
            if not post_m5.empty:
                ov_mask = (post_m5["High"] >= lo) & (post_m5["Low"] <= hi)
                if ov_mask.any():
                    was_touched_snr = True
                    hit_pos = [idx_i for idx_i, flag in enumerate(ov_mask.tolist()) if flag]
                    last_hit_rel = hit_pos[-1]
                    last_hit_abs = start_i + last_hit_rel
                    bars_since_touch_snr = max(0, (len(clean_m5_tp) - 1) - last_hit_abs)
                    raw_ts = post_m5["Time"].iloc[last_hit_rel] if "Time" in post_m5.columns else post_m5.index[last_hit_rel]
                    touch_time_snr = format_wib_timestamp(raw_ts, source_tz="MT5")
                from modules.utils import get_frozen_trade_plan, resolve_stored_setup_sl
                stored_sl_snr = resolve_stored_setup_sl("SNR", zid, timeframe=tf)
                if stored_sl_snr > 0.0:
                    post_sl_snr = post_m5
                    frz_snr = get_frozen_trade_plan("SNR", zid, timeframe=tf)
                    cr_ts_s = float((frz_snr or {}).get("created_at_ts") or 0.0)
                    if cr_ts_s > 0.0 and "Time" in post_m5.columns:
                        try:
                            t_ser_s = pd.to_datetime(post_m5["Time"], errors="coerce")
                            cr_dt_s = pd.Timestamp(cr_ts_s, unit="s")
                            if getattr(t_ser_s.dt, "tz", None) is not None:
                                cr_dt_s = cr_dt_s.tz_localize("UTC").tz_convert(t_ser_s.dt.tz)
                            m_after = t_ser_s >= cr_dt_s
                            if m_after.any():
                                post_sl_snr = post_m5.loc[m_after]
                        except Exception:
                            pass
                    if not post_sl_snr.empty:
                        if direction == "BUY" and bool((post_sl_snr["Low"] <= stored_sl_snr).any()):
                            sl_breached_hist_snr = True
                        elif direction == "SELL" and bool((post_sl_snr["High"] >= stored_sl_snr).any()):
                            sl_breached_hist_snr = True
        except Exception:
            pass
    if not was_touched_snr:
        prov_snr = _zone_touch_provenance(z, clean_m5=clean_m5_tp)
        if prov_snr.get("tested"):
            was_touched_snr = True
            if prov_snr.get("touch_time"):
                touch_time_snr = format_wib_timestamp(prov_snr.get("touch_time"), source_tz="MT5")

    max_entry_dist_atr = safe_float(risk_cfg.get("max_entry_distance_atr", 0.65), 0.65)
    m5_atr_tp = (
        safe_float(clean_m5_tp["ATR14"].iloc[-1], 0.0)
        if (clean_m5_tp is not None and not clean_m5_tp.empty and "ATR14" in clean_m5_tp.columns)
        else 0.0
    )
    ref_atr_tp = max(0.20, safe_float(z.get("ref_atr"), 0.0), m5_atr_tp)
    missed_thresh_snr = round(max_entry_dist_atr * ref_atr_tp, digits)
    m5_eval_st_snr = str(dec.get("m5_eval_status") or ev.get("m5_eval_status") or "").strip().upper() or None

    # Mode Conflict/Sideways Mid-Zone 50% Limit pada SNR (selaras dengan Auto Execution)
    early_mz_snr = None
    eff_min_rr_snr = min_rr
    try:
        from modules.adaptive_entry_engine import compute_midzone_scalp_levels, is_conflict_or_sideways_market
        from modules.config import get_auto_trading_config, ENTRY_JARAK_MAKS
        _acfg_snr = get_auto_trading_config()
        if (
            not is_closed
            and not is_invalid_z
            and not is_expired_z
            and not is_far_z
            and bool(_acfg_snr.get("CONFLICT_MIDZONE_LIMIT_ENABLED", False))
            and sc >= float(_acfg_snr.get("CONFLICT_MIDZONE_MIN_SCORE", 70.0))
            and is_conflict_or_sideways_market(
                direction=direction,
                htf_bias=h4,
                h1_bias=str(mb.get("h1") or "NETRAL"),
                m30_status=str(mb.get("m30") or ("CONFLICT" if m30_conflict else "CONFIRMED")),
                has_explicit_htf_conflict=(htf_conflict or m30_conflict),
            )
        ):
            _mz_cand_s = {
                "zone_id": zid,
                "timeframe": tf,
                "zone_type": "DEMAND" if direction == "BUY" else "SUPPLY",
                "price_low": lo,
                "price_high": hi,
                "score": sc,
                "grade": grd,
                "invalidation_price": struct_inv,
                "is_valid": True,
            }
            _mz_lvls = compute_midzone_scalp_levels(_mz_cand_s, spread=spread_pts, cfg=_acfg_snr, digits=digits)
            if _mz_lvls.get("valid"):
                _mid_e = float(_mz_lvls["entry_price"])
                _ref_m = float(ask_v if (direction == "BUY" and ask_v) else (bid_v if (direction == "SELL" and bid_v) else price))
                _gap_ok = (_mid_e < _ref_m - max(0.15, spread_pts)) if direction == "BUY" else (_mid_e > _ref_m + max(0.15, spread_pts))
                if _gap_ok and abs(_mid_e - _ref_m) <= float(ENTRY_JARAK_MAKS):
                    early_mz_snr = {
                        "entry": _mid_e,
                        "sl": float(_mz_lvls["sl"]),
                        "tp1": float(_mz_lvls["tp1"]),
                        "tp2": 0.0,
                        "ready": True,
                        "is_expired": False,
                        "is_midzone_limit": True,
                    }
                    struct_inv = float(_mz_lvls["invalidation_price"])
                    eff_min_rr_snr = min(min_rr, float(_acfg_snr.get("MIN_RR", 1.0)))
    except Exception:
        pass

    return build_shared_trade_plan(
        source="SNR",
        zone_id=zid,
        timeframe=tf,
        zone_type=z_type,
        direction=direction,
        grade=grd,
        score=sc,
        lo=lo,
        hi=hi,
        struct_inv=struct_inv,
        live_price=price,
        why=why_txt,
        bid=bid_v,
        ask=ask_v,
        spread=spread_pts,
        max_spread=max_spr,
        digits=digits,
        min_stop=min_stop,
        min_rr=eff_min_rr_snr,
        max_sl_dist=float(SL_JARAK_MAKS),
        atr_value=safe_float(z.get("ref_atr"), 0.0),
        is_closed=is_closed,
        data_unavailable=data_unavail,
        is_invalid_zone=is_invalid_z,
        invalid_reason=f"ZONE_{lifecycle}",
        is_expired_zone=is_expired_z,
        is_breakout_pending=is_bo_pending,
        is_far_zone=is_far_z,
        confirmed_plan=confirmed_payload,
        early_pending_plan=early_mz_snr,
        cond_reason=cond_rsn,
        compute_levels_fn=_compute_snr_levels,
        was_touched=(False if early_mz_snr else was_touched_snr),
        touch_time=touch_time_snr,
        m5_eval_status=m5_eval_st_snr,
        missed_move_threshold=missed_thresh_snr,
        bars_since_touch=bars_since_touch_snr,
        max_bars_after_touch=4,
        mt5_orders_snapshot=mt5_orders_snapshot,
        sl_breached_hist=sl_breached_hist_snr,
    )


def generate_snr_trade_plans(
    ranked_zones,
    entry_validation,
    decision,
    market_bias,
    current_price,
    symbol_spec,
    cfg,
    market_status="OPEN",
    data_freshness="FRESH",
    m5_df=None,
    mt5_orders_snapshot=None,
):
    """Evaluasi semua zona SNR ACTIVE/RADAR/BREAKOUT_PENDING unik untuk menghasilkan trade_plans & best_trade_plan."""
    from modules.utils import is_tracked_trade_plan_zone, resolve_trade_plan_focus
    from modules.zone_scanner import resolve_analysis_direction

    rk = ranked_zones if isinstance(ranked_zones, dict) else {}
    ev = entry_validation if isinstance(entry_validation, dict) else {}
    dec = decision if isinstance(decision, dict) else {}
    raw_cands = []
    for z_c in (ev.get("target_zone"), dec.get("target_zone")):
        if isinstance(z_c, dict):
            raw_cands.append(z_c)
    for z_c in (list(rk.get("active_supports") or []) + list(rk.get("active_resistances") or [])):
        if isinstance(z_c, dict):
            raw_cands.append(z_c)
    for z_c in (list(rk.get("all_zones") or []) + list(rk.get("invalidated_zones") or []) + list(rk.get("broken_zones") or [])):
        if not isinstance(z_c, dict):
            continue
        z_lc = str(z_c.get("lifecycle_status") or z_c.get("status") or "").upper()
        z_bo_st = str((z_c.get("breakout_info") or {}).get("status") or "").upper()
        if (
            z_lc in ("BREAKOUT_PENDING", "BREAKDOWN_PENDING")
            or z_bo_st == "BREAKOUT_PENDING_CONFIRMATION"
            or is_tracked_trade_plan_zone("SNR", str(z_c.get("zone_id") or ""))
        ):
            raw_cands.append(z_c)

    seen_ids = set()
    plans = []
    plan_by_zid = {}
    for z in raw_cands:
        zid = str(z.get("zone_id") or "").strip()
        if not zid or zid in seen_ids:
            continue
        is_tracked = is_tracked_trade_plan_zone("SNR", zid)
        if not is_tracked and (
            str(z.get("zone_relevance") or "").upper() in ("FAR_REFERENCE", "INVALID")
            or z.get("is_far_reference")
            or z.get("is_distant_historical")
        ):
            continue
        seen_ids.add(zid)
        tp_snr = generate_snr_trade_plan(
            z,
            ranked_zones=rk,
            entry_validation=ev,
            decision=dec,
            market_bias=market_bias,
            current_price=current_price,
            symbol_spec=symbol_spec,
            cfg=cfg,
            market_status=market_status,
            data_freshness=data_freshness,
            m5_df=m5_df,
            mt5_orders_snapshot=mt5_orders_snapshot,
        )
        plans.append(tp_snr)
        if safe_float(tp_snr.get("setup_sl"), 0.0) > 0.0:
            plan_by_zid[zid] = tp_snr

    if plan_by_zid:
        for z_ref in raw_cands:
            if not isinstance(z_ref, dict):
                continue
            rz_id = str(z_ref.get("zone_id") or "").strip()
            matched_tp = plan_by_zid.get(rz_id)
            if isinstance(matched_tp, dict):
                z_ref["setup_sl"] = safe_float(matched_tp.get("setup_sl"), 0.0)
                z_ref["setup_version"] = int(matched_tp.get("setup_version") or 1)

    dir_pref = resolve_analysis_direction(
        decision=dec,
        snr_bias=market_bias,
        market_status=market_status,
        data_ok=(str(market_status or "").upper() == "OPEN" and safe_float(current_price, 0.0) > 0),
    )
    spec = symbol_spec if isinstance(symbol_spec, dict) else {}
    best_tp = resolve_trade_plan_focus(
        plans,
        direction=dir_pref,
        scope="SNR",
        live_price=safe_float(current_price, 0.0),
        bid=safe_float(spec.get("bid"), 0.0) or None,
        ask=safe_float(spec.get("ask"), 0.0) or None,
        is_closed=(str(market_status or "").upper() in ("CLOSED", "MARKET_CLOSED")),
        mt5_orders_snapshot=mt5_orders_snapshot,
    )
    return plans, best_tp


def _is_valid_saved_breakout(saved_bo, tf, clean_df=None):
    """Verify persisted breakout_info matches the zone's own timeframe and is a confirmed break."""
    if not isinstance(saved_bo, dict):
        return True
    tf_up = str(tf or "").upper()
    eval_tf = str(saved_bo.get("eval_timeframe") or "").upper()
    if eval_tf and tf_up and eval_tf != tf_up:
        return False
    bo_status = str(saved_bo.get("status") or "").upper()
    if bo_status == "BREAKOUT_PENDING_CONFIRMATION" and not bool(saved_bo.get("reclaim_detected")):
        return False
    bo_time = str(saved_bo.get("breakout_bar_time") or "").strip()
    if tf_up != "M5" and bo_time and clean_df is not None and not clean_df.empty and "Time" in clean_df.columns:
        bar_times = [str(v) for v in clean_df["Time"].tolist()]
        if bar_times and bar_times[0] <= bo_time <= bar_times[-1] and bo_time not in bar_times:
            return False
    return True


def _apply_saved_lifecycle(zone, saved, clean_df=None):
    saved_status = str((saved or {}).get("lifecycle_status", "")).upper()
    broken_states = ("BROKEN", "RETEST", "RECLAIM_PENDING", "RECLAIMED")
    pending_states = ("BREAKDOWN_PENDING", "BREAKOUT_PENDING")
    if saved_status not in broken_states + ("INVALIDATED", "UNCONFIRMED") + pending_states:
        return
    z_type = str(zone.get("zone_type", "")).upper()
    tf = str(zone.get("timeframe", "")).upper()
    # Pending states bersifat sementara (bukan terminal): jangan pernah menurunkan zona yang sudah BROKEN/INVALIDATED
    # maupun memaksakan arah pending yang tidak cocok (mis. BREAKDOWN_PENDING pada RESISTANCE).
    if saved_status in pending_states:
        expected_pend = "BREAKDOWN_PENDING" if z_type == "SUPPORT" else ("BREAKOUT_PENDING" if z_type == "RESISTANCE" else "")
        saved_bo = (saved or {}).get("breakout_info")
        if (
            saved_status == expected_pend
            and str(zone.get("lifecycle_status", "")).upper() == expected_pend
            and isinstance(saved_bo, dict)
            and str(saved_bo.get("zone_type", z_type)).upper() == z_type
            and str(saved_bo.get("eval_timeframe", tf)).upper() == tf
        ):
            zone["lifecycle_breakout_info"] = saved_bo
        return
    current_status = str(zone.get("lifecycle_status", "")).upper()
    if saved_status == "UNCONFIRMED" and current_status != "UNCONFIRMED":
        return
    if current_status == "INVALIDATED" and saved_status != "INVALIDATED":
        return

    saved_bo = (saved or {}).get("breakout_info")
    if isinstance(saved_bo, dict) and str(saved_bo.get("zone_type", z_type)).upper() not in ("", z_type):
        saved_bo = None
    if (
        saved_status in broken_states
        and isinstance(saved_bo, dict)
        and not _is_valid_saved_breakout(saved_bo, tf, clean_df=clean_df)
        and not zone.get("is_broken")
        and not zone.get("is_invalidated")
    ):
        # Tolak restore state RECLAIM_PENDING/BROKEN yang berasal dari artefak sub-TF (mis. M5) / belum terkonfirmasi di TF zona.
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
    saved_ev = str((saved or {}).get("lifecycle_event") or saved_status).upper()
    if saved_ev in ("BREAKOUT_INVALIDATED", "BREAKDOWN_INVALIDATED") or saved_status == "RECLAIMED":
        if z_type == "SUPPORT":
            saved_ev = "BREAKDOWN_INVALIDATED"
        elif z_type == "RESISTANCE":
            saved_ev = "BREAKOUT_INVALIDATED"
    zone.update({
        "lifecycle_status": saved_status,
        "status": saved_status,
        "touch_status": saved_touch_status,
        "touch_count": touch_count,
        "is_broken": saved_status in broken_states,
        "is_invalidated": saved_status == "INVALIDATED",
        "is_active": saved_status in ("TOUCHED", "SWEEP", "ACTIVE"),
        "touch_analysis": touch,
        "lifecycle_breakout_info": saved_bo,
        "lifecycle_retest_origin": (saved or {}).get("retest_origin"),
        "lifecycle_event": saved_ev,
        "lifecycle_event_time": (saved or {}).get("lifecycle_event_time"),
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
        "event_time": None,
    }
    clean_df, _ = prepare_snr_frame(df, has_unclosed_last_bar=False)
    if clean_df.empty:
        result["reclaim_status"] = "UNCONFIRMED"
        return result

    bar_times = [str(value) for value in clean_df["Time"].tolist()]
    break_time = str((breakout_info or {}).get("breakout_bar_time") or "").strip()
    break_idx = None
    if break_time:
        try:
            break_idx = bar_times.index(break_time)
        except ValueError:
            if bar_times and break_time < bar_times[0]:
                # Historical break occurred before the current lookback window -> all bars in clean_df are post-break
                break_idx = -1
            else:
                try:
                    import pandas as _pd_pb
                    bo_ts_pb = _pd_pb.to_datetime(break_time, errors="coerce")
                    ts_ser_pb = _pd_pb.to_datetime(clean_df["Time"], errors="coerce")
                    if not _pd_pb.isna(bo_ts_pb) and not ts_ser_pb.isna().all():
                        pos_pb = int(ts_ser_pb.searchsorted(bo_ts_pb, side="left"))
                        break_idx = min(len(clean_df) - 1, max(0, pos_pb))
                except Exception:
                    break_idx = None
                if break_idx is None:
                    ta_idx_fb = (zone.get("touch_analysis") or {}).get("first_break_index")
                    if isinstance(ta_idx_fb, int) and 0 <= ta_idx_fb < len(clean_df):
                        break_idx = ta_idx_fb
                    else:
                        result["reclaim_status"] = "UNCONFIRMED"
                        return result
    else:
        ta_idx = (zone.get("touch_analysis") or {}).get("first_break_index")
        if isinstance(ta_idx, int) and 0 <= ta_idx < len(clean_df):
            break_idx = ta_idx
        else:
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
    last_atr = max(0.10, safe_float(last.get("ATR14"), safe_float(zone.get("ref_atr"), 1.0)))
    cfg_pb = config or get_snr_config()
    spread_pts = max(0.0, safe_float(zone.get("spread"), safe_float((cfg_pb.get("risk") or {}).get("default_spread_points", 0.25), 0.25)))
    fail_atr = safe_float((cfg_pb.get("retest") or {}).get("retest_fail_close_inside_atr", 0.18), 0.18)
    inv_atr = safe_float((cfg_pb.get("touch") or {}).get("invalidation_close_beyond_atr", 0.42), 0.42)
    inv_consec = max(2, int((cfg_pb.get("touch") or {}).get("consecutive_closes_for_invalidation", 2)))
    valid_pen_atr = safe_float((cfg_pb.get("breakout") or {}).get("valid_close_penetration_atr", 0.15), 0.15)
    fail_tol = fail_atr * last_atr + spread_pts
    inv_tol = inv_atr * last_atr + spread_pts
    valid_tol = valid_pen_atr * last_atr + spread_pts

    result["event_time"] = str(last.get("Time") or "") or None
    overlap = (
        (post_break["High"] >= (low - spread_pts)) & (post_break["Low"] <= (high + spread_pts))
    ).any()

    pb_closes = post_break["Close"].to_numpy(dtype=float)
    pb_atrs = post_break["ATR14"].to_numpy(dtype=float) if "ATR14" in post_break.columns else [last_atr] * len(pb_closes)
    pb_times = post_break["Time"].tolist() if "Time" in post_break.columns else [None] * len(pb_closes)
    run_edge = 0
    run_valid = 0
    is_breakout_invalidated = False
    is_retest_failed = False
    fail_event_time = None
    inv_event_time = None
    for idx_pb, c_val in enumerate(pb_closes):
        b_atr_pb = max(0.10, safe_float(pb_atrs[idx_pb], last_atr))
        fail_tol_k = fail_atr * b_atr_pb + spread_pts
        inv_tol_k = inv_atr * b_atr_pb + spread_pts
        valid_tol_k = valid_pen_atr * b_atr_pb + spread_pts
        pen_c = (c_val - high) if z_type == "SUPPORT" else (low - c_val)
        run_edge = (run_edge + 1) if pen_c > spread_pts else 0
        run_valid = (run_valid + 1) if pen_c >= valid_tol_k else 0
        if pen_c >= inv_tol_k or run_valid >= inv_consec:
            is_breakout_invalidated = True
            is_retest_failed = True
            inv_event_time = str(pb_times[idx_pb] or "") or inv_event_time
            break
        if pen_c >= fail_tol_k or run_edge >= 2:
            if not is_retest_failed:
                fail_event_time = str(pb_times[idx_pb] or "") or fail_event_time
            is_retest_failed = True

    pen_beyond = (last_close - high) if z_type == "SUPPORT" else (low - last_close)
    reclaim_zone_type = "RESISTANCE" if z_type == "SUPPORT" else "SUPPORT"
    is_retest = bool((overlap or pen_beyond > 0.0) and not is_retest_failed)
    inv_ev_name = "BREAKDOWN_INVALIDATED" if z_type == "SUPPORT" else "BREAKOUT_INVALIDATED"

    if is_retest_failed:
        reclaim_zone = dict(zone)
        reclaim_zone["zone_type"] = reclaim_zone_type
        reclaim_zone["confirmed_index"] = -1
        reclaim_zone["formation_index"] = -1
        reclaim = detect_zone_breakout(
            reclaim_zone,
            post_break,
            config=cfg_pb,
            has_unclosed_last_bar=False,
        )
        if reclaim.get("status") == "VALID_BREAKOUT" or is_breakout_invalidated:
            result["lifecycle_status"] = "RECLAIMED"
            result["lifecycle_event"] = inv_ev_name
            result["reclaim_status"] = "CONFIRMED"
            result["retest_status"] = "FAILED"
            result["breakout_invalidated"] = True
            if inv_event_time:
                result["event_time"] = inv_event_time
            elif reclaim.get("breakout_bar_time"):
                result["event_time"] = str(reclaim.get("breakout_bar_time"))
        else:
            result["lifecycle_status"] = "RECLAIM_PENDING"
            result["lifecycle_event"] = "RETEST_FAILED"
            result["reclaim_status"] = "PENDING"
            result["retest_status"] = "FAILED"
            result["breakout_invalidated"] = False
            if fail_event_time:
                result["event_time"] = fail_event_time
    elif is_retest:
        result["lifecycle_status"] = "RETEST"
        result["lifecycle_event"] = "RETEST"
        result["retest_status"] = result["retest_origin"]
        result["breakout_invalidated"] = False
    else:
        result["lifecycle_status"] = "BROKEN"
        result["lifecycle_event"] = "BROKEN"
        result["reclaim_status"] = "WAIT"
        result["breakout_invalidated"] = False
    return result


def _build_price_action_summary(focus_zone, ranked_zones, m5_df, frames_dict, momentum_by_tf, cfg):
    """Menyusun ringkasan PRICE ACTION (BREAKOUT, RETEST, REJECTION, MOMENTUM) untuk output utama."""
    eval_df = m5_df if (m5_df is not None and not getattr(m5_df, "empty", True)) else None
    m5_mom = momentum_by_tf.get("M5") or {}

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

    # Cek juga apakah ada zona yang baru mengalami potensi breakdown/breakout (pending_break_zones)
    # atau broken_zones dengan breakout valid / retest aktif di dekat harga.
    # JANGAN mengambil FALSE_BREAKOUT dari zona lain ke fokus zona aktif agar status PA tidak rancu.
    focus_label = (
        f"{focus_zone.get('rank_label', focus_zone.get('zone_type', 'FOCUS'))} "
        f"({focus_zone.get('low', 0):.2f}-{focus_zone.get('high', 0):.2f})"
        if focus_zone is not None
        else "NONE"
    )
    bo_zone_label = focus_label
    rt_zone_label = focus_label
    rej_zone_label = focus_label
    pending_zones = (ranked_zones or {}).get("pending_break_zones") or (ranked_zones or {}).get("unconfirmed_zones") or []
    broken_zones = (ranked_zones or {}).get("broken_zones") or []
    reclaimed_zones_pa = [
        rz for rz in ((ranked_zones or {}).get("reclaimed_zones") or [])
        if isinstance(rz, dict) and int(safe_float(rz.get("lifecycle_bars_ago"), 99)) <= 2 and safe_float(rz.get("distance_atr"), 0.0) <= 6.0
    ]
    if bo_res.get("status") == "NO_BREAKOUT" and eval_df is not None:
        bo_candidates = []
        seen_cids = set()
        for cand in list(pending_zones[:3]) + list(broken_zones[:3]) + list(reclaimed_zones_pa[:2]):
            if not isinstance(cand, dict):
                continue
            cid = cand.get("zone_id") or id(cand)
            if cid in seen_cids:
                continue
            seen_cids.add(cid)
            bo_candidates.append(cand)
        n_eval = len(eval_df)
        for cand in bo_candidates:
            c_label = f"{cand.get('rank_label', cand.get('zone_type', 'BROKEN'))} ({cand.get('low', 0):.2f}-{cand.get('high', 0):.2f})"
            cand_tf = str(cand.get("timeframe", "")).upper()
            cand_df = frames_dict.get(cand_tf) if (frames_dict and cand_tf in frames_dict) else eval_df
            cand_mom = momentum_by_tf.get(cand_tf) if (momentum_by_tf and cand_tf in momentum_by_tf) else m5_mom
            # Gunakan breakout_info yang sudah dievaluasi pada timeframe zona aslinya jika ada
            bo_cand = cand.get("breakout_info")
            if not isinstance(bo_cand, dict) or not bo_cand.get("status"):
                bo_cand = detect_zone_breakout(cand, cand_df if cand_df is not None else eval_df, momentum_result=cand_mom, config=cfg)
            if bo_cand.get("status") not in ("VALID_BREAKOUT", "BREAKOUT_PENDING_CONFIRMATION") and not (
                cand.get("lifecycle_status") in ("RECLAIM_PENDING", "RECLAIMED") and isinstance(cand.get("retest_info"), dict) and cand["retest_info"].get("status") == "RETEST_FAILED"
            ):
                continue
            # Jangan turunkan zona yang sudah confirmed broken di timeframe aslinya menjadi potential akibat intrabar M5
            cand_life = str(cand.get("lifecycle_status") or cand.get("status") or "").upper()
            if cand_life in ("BROKEN", "RETEST", "RECLAIM_PENDING", "RECLAIMED") and bo_cand.get("status") == "BREAKOUT_PENDING_CONFIRMATION":
                bo_cand = dict(bo_cand)
                bo_cand["status"] = "VALID_BREAKOUT"
                bo_cand["break_type"] = "CONFIRMED_BREAKDOWN" if str(cand.get("zone_type")).upper() == "SUPPORT" else "CONFIRMED_BREAKOUT"
                bo_cand["summary_status"] = "CONFIRMED"
            c_bar_idx = bo_cand.get("breakout_bar_index")
            c_bars_ago = (n_eval - 1 - int(c_bar_idx)) if (c_bar_idx is not None and n_eval > 0) else 999
            c_is_hist = bool(bo_cand.get("is_historical_break")) or safe_float(cand.get("distance_atr"), 0.0) > 12.0
            orig_rt = cand.get("retest_info") if isinstance(cand.get("retest_info"), dict) else None
            if orig_rt and str(orig_rt.get("status") or "").upper() in ("RETEST_CONFIRMED", "RETEST_IN_PROGRESS", "RETEST_FAILED"):
                rt_cand = dict(orig_rt)
            else:
                rt_cand = detect_zone_retest(cand, eval_df, breakout_result=bo_cand, config=cfg)
            is_active_retest = rt_cand.get("status") in ("RETEST_CONFIRMED", "RETEST_IN_PROGRESS", "RETEST_FAILED")
            if c_is_hist and not is_active_retest:
                continue
            # Promosikan jika fresh/recent breakout/breakdown (<= 6 bar M5) atau sedang retest aktif,
            # atau jika memang belum ada focus_zone aktif (selama bukan event historis/jauh).
            if (focus_zone is None and not c_is_hist) or c_bars_ago <= 6 or is_active_retest:
                bo_res = bo_cand
                bo_zone_label = c_label
                if is_active_retest:
                    rt_res = rt_cand
                    rt_zone_label = c_label
                break

    # Pastikan koherensi breakout vs retest pada zona yang sama
    if bo_res.get("status") == "FALSE_BREAKOUT" and bo_zone_label == rt_zone_label and rt_res.get("status") == "RETEST_IN_PROGRESS":
        rt_res = dict(rt_res)
        rt_res["status"] = "NO_RETEST"
        rt_res["flip_status"] = "INVALID"
        rt_res["summary_status"] = "INVALID"
    elif bo_res.get("status") == "BREAKOUT_PENDING_CONFIRMATION" and bo_zone_label == rt_zone_label and rt_res.get("status") == "RETEST_CONFIRMED":
        rt_res = dict(rt_res)
        rt_res["status"] = "NO_RETEST"
        rt_res["flip_status"] = "POTENTIAL_FLIP"
        rt_res["summary_status"] = "POTENTIAL FLIP"
        rt_res["reaction_confirmed"] = False

    n_bars = len(eval_df) if eval_df is not None else 0
    bo_bar_idx = bo_res.get("breakout_bar_index")
    bo_bars_ago = (n_bars - 1 - int(bo_bar_idx)) if (bo_bar_idx is not None and n_bars > 0) else 999
    # Breakout dianggap historical jika terjadi lebih dari 1 candle yang lalu atau berasal dari zona broken lama
    bo_is_historical = bool(
        bo_res.get("is_historical_break")
        or bo_bars_ago > 1
        or (bo_zone_label != focus_label and bo_bars_ago > 1)
    )
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
    pa_flip_status = str(rt_detail.get("flip_status") or "").upper()
    if not pa_flip_status or pa_flip_status == "NONE":
        if bo_res.get("status") == "BREAKOUT_PENDING_CONFIRMATION":
            pa_flip_status = "POTENTIAL_FLIP"
        elif bo_res.get("status") == "FALSE_BREAKOUT" or bo_res.get("reclaim_detected"):
            pa_flip_status = "INVALID"
        elif bo_res.get("status") == "VALID_BREAKOUT":
            pa_flip_status = "CONFIRMED_FLIP" if rt_res.get("status") == "RETEST_CONFIRMED" else "POTENTIAL_FLIP"
        else:
            pa_flip_status = "NONE"
    rt_detail["flip_status"] = pa_flip_status
    rej_detail = dict(rej_res)
    rej_bar_idx = rej_detail.get("bar_index")
    rej_bars_ago = (n_bars - 1 - int(rej_bar_idx)) if (rej_bar_idx is not None and n_bars > 0) else 999
    # Confirmed rejection pada bar k dikonfirmasi oleh candle k+1; hanya bar k = n_bars-2 (bars_ago == 1)
    # yang baru saja terkonfirmasi pada candle closed terakhir. Lebih lama dari itu bersifat historis.
    rej_is_historical = bool(rej_bars_ago > 1)
    rej_detail["bars_ago"] = rej_bars_ago
    rej_detail["is_historical"] = rej_is_historical
    rej_detail["zone_label"] = rej_zone_label
    if (
        rej_bar_idx is not None
        and eval_df is not None
        and "Time" in eval_df.columns
        and 0 <= int(rej_bar_idx) + 1 < n_bars
    ):
        rej_detail["confirm_bar_time"] = str(eval_df["Time"].iloc[int(rej_bar_idx) + 1])
    rej_summary = rej_res.get("summary_status", "NOT CONFIRMED")
    if rej_detail.get("confirmed") and rej_is_historical:
        rej_summary = f"HISTORICAL ({rej_summary})"

    return {
        "focus_zone_label": focus_label,
        "breakout_zone_label": bo_zone_label,
        "retest_zone_label": rt_zone_label,
        "rejection_zone_label": rej_zone_label,
        "breakout_code": bo_res.get("status", "NO_BREAKOUT"),
        "break_type": bo_res.get("break_type", "NONE"),
        "potential_breakdown": bool(bo_res.get("potential_breakdown", False)),
        "potential_breakout": bool(bo_res.get("potential_breakout", False)),
        "reclaim_detected": bool(bo_res.get("reclaim_detected", False)),
        "zone_failed": bool(bo_res.get("zone_failed", False)),
        "breakout_direction": bo_res.get("direction", "NONE"),
        "breakout_summary": bo_res.get("summary_status", "NOT CONFIRMED"),
        "breakout_is_historical": bo_is_historical,
        "breakout_detail": bo_detail,
        "retest_code": rt_res.get("status", "NO_RETEST"),
        "retest_direction": rt_res.get("direction", "NONE"),
        "retest_summary": rt_res.get("summary_status", "NOT CONFIRMED"),
        "retest_is_historical": rt_is_historical,
        "retest_detail": rt_detail,
        "flip_status": pa_flip_status,
        "rejection_code": rej_res.get("status", "NONE"),
        "rejection_direction": rej_res.get("direction", "NONE"),
        "rejection_summary": rej_summary,
        "rejection_is_historical": rej_is_historical,
        "rejection_detail": rej_detail,
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
    m5_df_ref = prepared_frames.get("M5")
    m5_mom_ref = (
        analyze_momentum(m5_df_ref, timeframe="M5", config=cfg, has_unclosed_last_bar=False)
        if (m5_df_ref is not None and not getattr(m5_df_ref, "empty", True))
        else None
    )

    for tf in SNR_TIMEFRAMES:
        df_tf = prepared_frames.get(tf)
        mom_tf = m5_mom_ref if (tf == "M5" and m5_mom_ref is not None) else analyze_momentum(df_tf, timeframe=tf, config=cfg, has_unclosed_last_bar=False)
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

        clean_df_tf, _ = prepare_snr_frame(df_tf, has_unclosed_last_bar=False)
        tf_bar_times = [str(v) for v in clean_df_tf["Time"].tolist()] if (not clean_df_tf.empty and "Time" in clean_df_tf.columns) else []
        tf_last_bar_time = tf_bar_times[-1] if tf_bar_times else None

        val_tf_list = []
        for cz in all_clust_tf:
            cz["time_tz"] = str((getattr(df_tf, "attrs", {}) or {}).get("time_tz", "WIB"))
            touched = analyze_zone_touches(cz, df_tf, config=cfg, has_unclosed_last_bar=False)
            validated = validate_single_zone(
                touched,
                df=df_tf,
                current_price=current_price,
                config=cfg,
                has_unclosed_last_bar=False,
            )
            validated.setdefault("symbol", market_data.get("symbol", symbol or cfg.get("symbol", "")))
            validated["spread"] = max(
                0.0,
                safe_float(
                    market_data.get("spread"),
                    safe_float((cfg.get("risk") or {}).get("default_spread_points", 0.25), 0.25),
                ),
            )
            lifecycle_key = _snr_lifecycle_key(validated)
            saved_lifecycle = lifecycle_state.get(lifecycle_key) or {}
            _apply_saved_lifecycle(validated, saved_lifecycle, clean_df=clean_df_tf)
            bo_z = detect_zone_breakout(validated, df_tf, momentum_result=mom_tf, config=cfg)
            bo_z["eval_timeframe"] = tf
            rt_z = detect_zone_retest(validated, df_tf, breakout_result=bo_z, config=cfg)
            rej_z = detect_zone_rejection(validated, df_tf, config=cfg)
            n_tf_bars = len(clean_df_tf)
            rej_z_idx = rej_z.get("bar_index")
            rej_z_ago = (n_tf_bars - 1 - int(rej_z_idx)) if (rej_z_idx is not None and n_tf_bars > 0) else 999
            rej_z["bars_ago"] = rej_z_ago
            rej_z["is_historical"] = bool(rej_z_ago > 1)
            z_type_cur = str(validated.get("zone_type", "")).upper()
            pend_state = "BREAKDOWN_PENDING" if z_type_cur == "SUPPORT" else "BREAKOUT_PENDING"
            pending_states = ("BREAKDOWN_PENDING", "BREAKOUT_PENDING")
            saved_prev_broken = str(saved_lifecycle.get("lifecycle_status", "")).upper() in (
                "BROKEN", "RETEST", "RECLAIM_PENDING", "RECLAIMED", "INVALIDATED"
            )
            if (
                bo_z.get("status") == "BREAKOUT_PENDING_CONFIRMATION"
                and not validated.get("is_invalidated")
                and not saved_prev_broken
                and not bo_z.get("prior_confirmed_breakout")
                and not bo_z.get("reclaim_detected")
            ):
                # Penetrasi tipis dari touch_analysis tidak boleh mempromosikan BREAKOUT_PENDING menjadi BROKEN/CONFIRMED
                # jika candle closed TF asal belum memenuhi threshold body/rasio/arah/momentum detect_zone_breakout.
                validated["lifecycle_status"] = pend_state
                validated["status"] = pend_state
                validated["is_broken"] = False
                validated["is_active"] = False
                validated["touch_status"] = pend_state
                validated["break_event"] = "POTENTIAL_BREAKDOWN" if z_type_cur == "SUPPORT" else "POTENTIAL_BREAKOUT"
            current_lifecycle = str(validated.get("lifecycle_status", "ACTIVE")).upper()
            current_side = (
                "BELOW" if current_price < safe_float(validated.get("low"), 0.0)
                else "ABOVE" if current_price > safe_float(validated.get("high"), 0.0)
                else "INSIDE"
            )
            breakout_confirmed = (
                bo_z.get("status") == "VALID_BREAKOUT"
                or bool(bo_z.get("reclaim_detected"))
            )
            locked_states = ("BROKEN", "RETEST", "RECLAIM_PENDING", "RECLAIMED")
            breakout_side = "BELOW" if z_type_cur == "SUPPORT" else "ABOVE"
            valid_side = "ABOVE" if z_type_cur == "SUPPORT" else "BELOW"

            if (
                breakout_confirmed
                and not bo_z.get("reclaim_detected")
                and current_lifecycle == "RETEST"
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
                touch["reason"] = (
                    "Support breakdown terkonfirmasi oleh candle close dan validasi momentum."
                    if z_type_cur == "SUPPORT"
                    else "Resistance breakout terkonfirmasi oleh candle close dan validasi momentum."
                )
                validated["touch_analysis"] = touch
                current_lifecycle = "BROKEN"
            elif bo_z.get("status") == "BREAKOUT_PENDING_CONFIRMATION" and current_lifecycle not in locked_states + ("INVALIDATED",):
                validated["lifecycle_status"] = pend_state
                validated["status"] = pend_state
                validated["is_active"] = False
                validated["break_event"] = "POTENTIAL_BREAKDOWN" if z_type_cur == "SUPPORT" else "POTENTIAL_BREAKOUT"
                current_lifecycle = pend_state

            # Deteksi dini potensi breakdown Support / breakout Resistance pada candle M5 closed
            # sebelum candle HTF selesai. M5 TIDAK boleh mengunci zona HTF sebagai BROKEN/RECLAIM_PENDING
            # tanpa breakdown terkonfirmasi di timeframe zona sendiri, dan tidak boleh mendowngrade zona yang sudah terkonfirmasi break.
            bo_m5 = None
            if (
                tf != "M5"
                and current_lifecycle not in locked_states + ("INVALIDATED", "BROKEN", "RECLAIM_PENDING", "RECLAIMED")
                and not validated.get("breakout_confirmed")
                and not validated.get("is_broken")
                and m5_df_ref is not None
                and not getattr(m5_df_ref, "empty", True)
            ):
                bo_m5 = detect_zone_breakout(validated, m5_df_ref, momentum_result=m5_mom_ref, config=cfg)
                bo_m5["eval_timeframe"] = "M5"
                validated["m5_breakout_info"] = bo_m5
                if (
                    bo_m5.get("status") in ("VALID_BREAKOUT", "BREAKOUT_PENDING_CONFIRMATION")
                    and bool(bo_m5.get("close_outside_zone"))
                ):
                    validated["lifecycle_status"] = pend_state
                    validated["status"] = pend_state
                    validated["is_active"] = False
                    validated["break_event"] = "POTENTIAL_BREAKDOWN" if z_type_cur == "SUPPORT" else "POTENTIAL_BREAKOUT"
                    current_lifecycle = pend_state
                elif bo_m5.get("reclaim_detected") and not validated.get("is_broken"):
                    validated["break_event"] = "FALSE_BREAKDOWN" if z_type_cur == "SUPPORT" else "FALSE_BREAKOUT"

            saved_bo_valid = saved_lifecycle.get("breakout_info") or validated.get("lifecycle_breakout_info")
            if isinstance(saved_bo_valid, dict) and not _is_valid_saved_breakout(saved_bo_valid, tf, clean_df=clean_df_tf):
                saved_bo_valid = None

            breakout_info = (
                bo_z if breakout_confirmed
                else (
                    bo_z
                    if current_lifecycle in pending_states
                    and bo_z.get("status") in ("FALSE_BREAKOUT", "BREAKOUT_PENDING_CONFIRMATION")
                    else saved_bo_valid
                )
            )
            post_break_state = None
            if current_lifecycle in locked_states:
                if current_lifecycle == "RECLAIMED":
                    post_break_state = {
                        "lifecycle_status": "RECLAIMED",
                        "lifecycle_event": "BREAKDOWN_INVALIDATED" if z_type_cur == "SUPPORT" else "BREAKOUT_INVALIDATED",
                        "reclaim_status": "CONFIRMED",
                        "retest_status": "FAILED",
                        "breakout_invalidated": True,
                        "retest_origin": "FROM_BELOW" if z_type_cur == "SUPPORT" else "FROM_ABOVE",
                        "event_time": saved_lifecycle.get("lifecycle_event_time"),
                    }
                else:
                    if (
                        isinstance(breakout_info, dict)
                        and breakout_info is not bo_z
                        and str(breakout_info.get("status") or "").upper() == "VALID_BREAKOUT"
                        and str((rt_z or {}).get("status") or "").upper() in ("NO_RETEST", "")
                    ):
                        rt_z = detect_zone_retest(validated, df_tf, breakout_result=breakout_info, config=cfg)
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
            elif current_lifecycle in pending_states:
                is_clean_false_break = (
                    not validated.get("is_broken")
                    and current_side == valid_side
                    and (
                        (bo_z.get("status") == "FALSE_BREAKOUT" and not bo_z.get("reclaim_detected"))
                        or (
                            bo_m5 is not None
                            and bo_m5.get("status") == "FALSE_BREAKOUT"
                            and bo_z.get("status") in ("NO_BREAKOUT", "FALSE_BREAKOUT")
                            and not bo_z.get("reclaim_detected")
                        )
                    )
                )
                if is_clean_false_break:
                    # False breakout / false breakdown murni (belum pernah confirmed broken di TF zona):
                    # zona tetap bertahan, dibedakan eksplisit dari reclaim zona yang sudah gagal.
                    fb_evt = "FALSE_BREAKDOWN" if z_type_cur == "SUPPORT" else "FALSE_BREAKOUT"
                    validated["lifecycle_status"] = "ACTIVE"
                    validated["status"] = "ACTIVE"
                    validated["is_active"] = True
                    validated["lifecycle_event"] = fb_evt
                    validated["break_event"] = fb_evt
                    validated["reclaim_status"] = "WAIT"
                    current_lifecycle = "ACTIVE"
                elif breakout_info and (current_side != breakout_side or bo_z.get("status") == "FALSE_BREAKOUT"):
                    # Jika sempat ada break terkonfirmasi di TF zona sebelum kembali -> evaluasi reclaim (tetap non-aktif)
                    if breakout_info.get("reclaim_detected") or breakout_info.get("status") == "VALID_BREAKOUT":
                        post_break_state = _evaluate_post_break_state(
                            validated,
                            df_tf,
                            breakout_info,
                            cfg,
                            fallback_status=pend_state,
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
                previous_side = str(saved_lifecycle.get("last_side", "")).upper()
                if not retest_origin and z_type_cur == "SUPPORT" and previous_side == "BELOW" and current_side != "BELOW":
                    retest_origin = "FROM_BELOW"
                elif not retest_origin and z_type_cur == "RESISTANCE" and previous_side == "ABOVE" and current_side != "ABOVE":
                    retest_origin = "FROM_ABOVE"
                elif not retest_origin and (bo_z.get("reclaim_detected") or bo_z.get("prior_confirmed_breakout")) and current_side != "BELOW" and z_type_cur == "SUPPORT":
                    retest_origin = "FROM_BELOW"
                elif not retest_origin and (bo_z.get("reclaim_detected") or bo_z.get("prior_confirmed_breakout")) and current_side != "ABOVE" and z_type_cur == "RESISTANCE":
                    retest_origin = "FROM_ABOVE"
                if retest_origin:
                    rt_z["retest_origin"] = retest_origin
                validated["lifecycle_retest_origin"] = retest_origin
                if post_break_state:
                    validated["lifecycle_event"] = post_break_state["lifecycle_event"]
                    validated["reclaim_status"] = post_break_state["reclaim_status"]
                    validated["retest_status"] = post_break_state["retest_status"]
            else:
                if bo_z.get("status") == "FALSE_BREAKOUT" and not bo_z.get("reclaim_detected"):
                    validated["lifecycle_event"] = "FALSE_BREAKDOWN" if z_type_cur == "SUPPORT" else "FALSE_BREAKOUT"
                else:
                    validated["lifecycle_event"] = validated.get("lifecycle_event", current_lifecycle)
                validated["reclaim_status"] = validated.get("reclaim_status", "WAIT")
                validated["retest_status"] = "WAIT"

            final_life_st = str(validated.get("lifecycle_status", "ACTIVE")).upper()
            final_life_ev = str(validated.get("lifecycle_event", final_life_st)).upper()
            prev_life_st = str(saved_lifecycle.get("lifecycle_status", "")).upper()
            prev_life_ev = str(saved_lifecycle.get("lifecycle_event", "")).upper()
            prev_ev_time = saved_lifecycle.get("lifecycle_event_time")
            if prev_life_st == final_life_st and prev_life_ev == final_life_ev and prev_ev_time:
                life_ev_time = str(prev_ev_time)
            else:
                life_ev_time = (
                    (post_break_state.get("event_time") if post_break_state else None)
                    or (bo_z.get("breakout_bar_time") if final_life_st in ("BROKEN", "BREAKDOWN_PENDING", "BREAKOUT_PENDING") and bo_z.get("breakout_bar_time") else None)
                    or ((validated.get("touch_analysis") or {}).get("lifecycle_event_time"))
                    or (
                        (breakout_info.get("breakout_bar_time") if isinstance(breakout_info, dict) else None)
                        or validated.get("confirmed_time")
                        or validated.get("formation_time")
                        if final_life_st == "RECLAIMED" and prev_life_st == "RECLAIMED"
                        else tf_last_bar_time
                    )
                )
            life_bars_ago = 0
            if life_ev_time and tf_bar_times:
                if life_ev_time in tf_bar_times:
                    life_bars_ago = (len(tf_bar_times) - 1) - tf_bar_times.index(life_ev_time)
                elif life_ev_time < tf_bar_times[0]:
                    life_bars_ago = len(tf_bar_times)
            life_is_historical = bool(
                (final_life_st in locked_states and life_bars_ago > 1)
                or (final_life_st == "RECLAIMED" and prev_life_st == "RECLAIMED")
                or (final_life_st == "BROKEN" and isinstance(breakout_info, dict) and breakout_info.get("is_historical_break"))
            )
            validated["lifecycle_event_time"] = life_ev_time
            validated["lifecycle_bars_ago"] = int(life_bars_ago)
            validated["lifecycle_is_historical"] = life_is_historical

            is_confirmed_broken_zone = bool(
                final_life_st in ("INVALIDATED", "BROKEN", "RETEST", "RECLAIM_PENDING", "RECLAIMED")
                or validated.get("is_broken")
                or validated.get("is_invalidated")
            )
            if is_confirmed_broken_zone and isinstance(bo_z, dict):
                if bo_z.get("status") == "BREAKOUT_PENDING_CONFIRMATION" and (validated.get("is_invalidated") or saved_prev_broken):
                    bo_z = dict(bo_z)
                    bo_z["status"] = "VALID_BREAKOUT"
                    bo_z["break_type"] = "CONFIRMED_BREAKDOWN" if z_type_cur == "SUPPORT" else "CONFIRMED_BREAKOUT"
                    bo_z["potential_breakdown"] = False
                    bo_z["potential_breakout"] = False
                    bo_z["zone_failed"] = True
                    bo_z["label"] = "VALID BREAKDOWN" if z_type_cur == "SUPPORT" else "VALID BREAKOUT"
                    bo_z["summary_status"] = "CONFIRMED"
                    bo_z["breakout_state"] = "BROKEN" if life_is_historical else "CONFIRMED"
                elif bo_z.get("status") == "VALID_BREAKOUT":
                    bo_z = dict(bo_z)
                    bo_z["potential_breakdown"] = False
                    bo_z["potential_breakout"] = False
                    bo_z["breakout_state"] = "BROKEN" if life_is_historical else "CONFIRMED"
                if validated.get("break_event") in ("POTENTIAL_BREAKDOWN", "POTENTIAL_BREAKOUT") and bo_z.get("status") == "VALID_BREAKOUT":
                    validated["break_event"] = "CONFIRMED_BREAKDOWN" if z_type_cur == "SUPPORT" else "CONFIRMED_BREAKOUT"

            # Sinkronisasi dua arah antara rt_z (closed candle TF asal) dan lifecycle_status pasca-breakout
            rt_st_cur = str(rt_z.get("status") or "").upper() if isinstance(rt_z, dict) else ""
            if isinstance(rt_z, dict) and rt_st_cur == "RETEST_FAILED" and final_life_st in ("BROKEN", "RETEST"):
                is_bo_inv_rt = bool(rt_z.get("breakout_invalidated"))
                final_life_st = "RECLAIMED" if is_bo_inv_rt else "RECLAIM_PENDING"
                validated["lifecycle_status"] = final_life_st
                validated["status"] = final_life_st
                validated["lifecycle_event"] = (
                    ("BREAKDOWN_INVALIDATED" if z_type_cur == "SUPPORT" else "BREAKOUT_INVALIDATED")
                    if is_bo_inv_rt
                    else "RETEST_FAILED"
                )
                validated["reclaim_status"] = "CONFIRMED" if is_bo_inv_rt else "PENDING"
                validated["retest_status"] = "FAILED"
                if rt_z.get("retest_bar_time"):
                    validated["lifecycle_event_time"] = str(rt_z["retest_bar_time"])
            elif isinstance(rt_z, dict) and rt_st_cur == "RETEST_CONFIRMED" and final_life_st in ("BROKEN", "RETEST"):
                validated["lifecycle_status"] = "RETEST"
                validated["status"] = "RETEST"
                validated["lifecycle_event"] = "RETEST_CONFIRMED"
                validated["retest_status"] = "CONFIRMED"
                final_life_st = "RETEST"
                if rt_z.get("retest_bar_time"):
                    validated["lifecycle_event_time"] = str(rt_z["retest_bar_time"])
            elif final_life_st in ("RECLAIMED", "RECLAIM_PENDING") and isinstance(rt_z, dict):
                is_bo_inv_lf = bool(final_life_st == "RECLAIMED" or (post_break_state and post_break_state.get("breakout_invalidated")))
                rt_z = dict(rt_z)
                rt_z["status"] = "RETEST_FAILED"
                rt_z["breakout_invalidated"] = is_bo_inv_lf
                rt_z["flip_status"] = "INVALID"
                inv_lbl_lf = "BREAKDOWN INVALIDATED" if z_type_cur == "SUPPORT" else "BREAKOUT INVALIDATED"
                rt_z["label"] = inv_lbl_lf if is_bo_inv_lf else "RETEST FAILED"
                rt_z["summary_status"] = "INVALIDATED" if is_bo_inv_lf else "FAILED"
                rt_z["reaction_confirmed"] = False
                rt_z["retest_held"] = False
                z_lo_inv = safe_float(validated.get("low"), 0.0)
                z_hi_inv = safe_float(validated.get("high"), z_lo_inv)
                if is_bo_inv_lf:
                    rt_z["reason"] = (
                        f"BREAKDOWN_INVALIDATED (INVALID): breakdown Support ({z_lo_inv:.2f}-{z_hi_inv:.2f}) dibatalkan oleh reclaim Support candle closed {tf}."
                        if z_type_cur == "SUPPORT"
                        else f"BREAKOUT_INVALIDATED (INVALID): breakout Resistance ({z_lo_inv:.2f}-{z_hi_inv:.2f}) dibatalkan oleh reclaim Resistance candle closed {tf}."
                    )
                else:
                    rt_z["reason"] = (
                        f"RETEST_FAILED (INVALID): candle closed {tf} gagal mempertahankan batas {z_type_cur} ({z_lo_inv:.2f}-{z_hi_inv:.2f}) melewati toleransi ATR+spread."
                    )

            # Ketika harga kembali ke zona breakout terkonfirmasi (live price atau M5 closed pasca-break)
            # sebelum retest/rejection benar-benar terkonfirmasi pada candle closed TF asal:
            # tandai RETEST IN PROGRESS (POTENTIAL_FLIP). Wick / intrabar dip di bawah batas zona BUKAN kegagalan otomatis.
            if (
                isinstance(bo_z, dict)
                and bo_z.get("status") == "VALID_BREAKOUT"
                and final_life_st in ("BROKEN", "RETEST")
                and not validated.get("is_invalidated")
                and isinstance(rt_z, dict)
                and str(rt_z.get("status") or "").upper() not in ("RETEST_CONFIRMED", "RETEST_FAILED")
            ):
                z_lo_rt = safe_float(validated.get("low"), 0.0)
                z_hi_rt = safe_float(validated.get("high"), z_lo_rt)
                atr_rt = max(0.10, safe_float(validated.get("ref_atr") or validated.get("atr_at_formation"), 1.0))
                spread_rt = max(0.0, safe_float(validated.get("spread"), safe_float((cfg.get("risk") or {}).get("default_spread_points", 0.25), 0.25)))
                touch_tol_rt = max(spread_rt, atr_rt * safe_float((cfg.get("radar") or {}).get("touch_distance_atr", 0.08), 0.08) + spread_rt)
                prox_tol_rt = max(touch_tol_rt, atr_rt * safe_float((cfg.get("retest") or {}).get("retest_proximity_atr", 0.28), 0.28) + spread_rt)
                live_in_zone = bool(
                    current_price > 0
                    and z_lo_rt > 0
                    and (z_lo_rt - touch_tol_rt) <= current_price <= (z_hi_rt + touch_tol_rt)
                )
                live_beyond_opposite = bool(
                    current_price > 0
                    and z_lo_rt > 0
                    and (
                        (z_type_cur == "RESISTANCE" and current_price < (z_lo_rt - touch_tol_rt))
                        or (z_type_cur == "SUPPORT" and current_price > (z_hi_rt + touch_tol_rt))
                    )
                )
                live_far_breakout_side = bool(
                    current_price > 0
                    and z_lo_rt > 0
                    and (
                        (z_type_cur == "RESISTANCE" and current_price > (z_hi_rt + prox_tol_rt))
                        or (z_type_cur == "SUPPORT" and current_price < (z_lo_rt - prox_tol_rt))
                    )
                )
                m5_returned = False
                bo_time_ref = str(bo_z.get("breakout_bar_time") or "").strip()
                if (
                    not live_in_zone
                    and not live_beyond_opposite
                    and not live_far_breakout_side
                    and tf != "M5"
                    and bo_time_ref
                    and m5_df_ref is not None
                    and not getattr(m5_df_ref, "empty", True)
                    and "Time" in m5_df_ref.columns
                ):
                    try:
                        m5_post = m5_df_ref[m5_df_ref["Time"].astype(str) > bo_time_ref]
                        if not m5_post.empty:
                            m5_ov = ((m5_post["High"] >= (z_lo_rt - touch_tol_rt)) & (m5_post["Low"] <= (z_hi_rt + touch_tol_rt))).any()
                            m5_returned = bool(m5_ov)
                    except Exception:
                        m5_returned = False
                rt_cur_code = str(rt_z.get("status") or "").upper()
                if live_in_zone or live_beyond_opposite or m5_returned or (final_life_st == "RETEST" and rt_cur_code == "RETEST_IN_PROGRESS" and not live_far_breakout_side):
                    rt_z = dict(rt_z)
                    rt_z["status"] = "RETEST_IN_PROGRESS"
                    rt_z["label"] = "RETEST IN PROGRESS"
                    rt_z["summary_status"] = "IN PROGRESS"
                    rt_z["retest_detected"] = True
                    rt_z["reaction_confirmed"] = False
                    rt_z["retest_held"] = False
                    rt_z["flip_status"] = "POTENTIAL_FLIP"
                    rt_z["intrabar_beyond_edge"] = bool(live_beyond_opposite)
                    rt_z["retest_origin"] = "FROM_BELOW" if z_type_cur == "SUPPORT" else "FROM_ABOVE"
                    if not rt_z.get("retest_bar_time") and bo_time_ref:
                        rt_z["retest_bar_time"] = bo_time_ref
                    brk_word = "breakdown" if z_type_cur == "SUPPORT" else "breakout"
                    rt_z["reason"] = (
                        f"Harga live menembus sisi sebaliknya ({z_lo_rt:.2f}-{z_hi_rt:.2f}) intrabar; "
                        f"tetap RETEST IN PROGRESS menunggu close {tf} (wick/intrabar bukan kegagalan otomatis)."
                        if live_beyond_opposite
                        else f"Harga kembali menguji zona {z_type_cur} ({z_lo_rt:.2f}-{z_hi_rt:.2f}) pasca-{brk_word}; "
                        "RETEST IN PROGRESS (menunggu konfirmasi rejection/retest closed candle — belum CONFIRMED_FLIP)."
                    )
                    validated["lifecycle_status"] = "RETEST"
                    validated["status"] = "RETEST"
                    validated["retest_status"] = "IN_PROGRESS"
                    final_life_st = "RETEST"
                elif final_life_st == "RETEST" and live_far_breakout_side and rt_cur_code in ("RETEST_WAITING", "NO_RETEST"):
                    # Hindari status RETEST stale ketika harga sudah menjauh di sisi breakout tanpa retest terkonfirmasi
                    validated["lifecycle_status"] = "BROKEN"
                    validated["status"] = "BROKEN"
                    validated["retest_status"] = "WAIT"
                    final_life_st = "BROKEN"

            validated["breakout_info"] = bo_z
            validated["retest_info"] = rt_z
            validated["rejection_info"] = rej_z
            validated["flip_status"] = classify_snr_flip_status(validated, breakout_info=bo_z, retest_info=rt_z)
            if isinstance(rt_z, dict):
                rt_z["flip_status"] = validated["flip_status"]
            if is_confirmed_broken_zone:
                validated["invalid_label"] = "INVALID — BROKEN"
                validated["is_active"] = False
            else:
                validated["invalid_label"] = ""
            saved_life_ev = str(validated.get("lifecycle_event") or "ACTIVE").upper()
            if saved_life_ev in ("BREAKOUT_INVALIDATED", "BREAKDOWN_INVALIDATED") or validated.get("lifecycle_status") == "RECLAIMED":
                saved_life_ev = "BREAKDOWN_INVALIDATED" if z_type_cur == "SUPPORT" else "BREAKOUT_INVALIDATED"
                validated["lifecycle_event"] = saved_life_ev
            lifecycle_state[lifecycle_key] = {
                "lifecycle_status": validated.get("lifecycle_status", "ACTIVE"),
                "touch_count": int(validated.get("touch_count", 0) or 0),
                "last_side": current_side,
                "breakout_info": breakout_info,
                "retest_origin": retest_origin,
                "lifecycle_event": saved_life_ev,
                "lifecycle_event_time": life_ev_time,
                "reclaim_status": validated.get("reclaim_status", "WAIT"),
                "invalid_label": validated.get("invalid_label", ""),
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
            lifecycle_k = _snr_lifecycle_key(z)
            saved_life = lifecycle_state.get(lifecycle_k) or {}
            if "initial_score" in saved_life and "initial_score" not in z:
                z["initial_score"] = saved_life["initial_score"]
            sz = score_zone(
                z,
                confluence_info=c_info,
                breakout_info=z.get("breakout_info"),
                retest_info=z.get("retest_info"),
                rejection_info=z.get("rejection_info"),
                momentum_info=mom_tf,
                config=cfg,
            )
            # Simpan locked initial_score ke state lifecycle jika zona berkualitas >= 70
            if float(sz.get("score") or 0.0) >= 70.0 and lifecycle_k in lifecycle_state:
                lifecycle_state[lifecycle_k]["initial_score"] = float(sz["score"])
            tf_scored.append(sz)
            if tf in SNR_STRUCTURAL_TIMEFRAMES:
                structural_scored_zones.append(sz)
        scored_zones_by_tf[tf] = tf_scored

    # Jika structural timeframes kosong (misal hanya diberikan M5 pada unit test spesifik),
    # gunakan zona M5 sebagai fallback ranking
    zones_for_ranking = structural_scored_zones if structural_scored_zones else scored_zones_by_tf.get("M5", [])
    h1_atr_for_rank = safe_float((atr_diag.get("H1") or {}).get("atr14"), 0.0) or None
    ranked_zones = rank_zones(zones_for_ranking, current_price=current_price, h1_atr=h1_atr_for_rank)

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

    # Entry Validation hanya menggunakan M5.
    entry_df = prepared_frames.get("M5")

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
    target_id = (entry_validation.get("target_zone") or {}).get("zone_id")
    entry_price = safe_float((entry_validation.get("rr_calculation") or {}).get("effective_entry"), 0.0)
    ranked_by_id = {
        zone.get("zone_id"): zone for zone in (
            (ranked_zones.get("active_supports") or []) + (ranked_zones.get("active_resistances") or [])
        ) if zone.get("zone_id")
    }
    conflicting = [
        overlap for overlap in ranked_zones.get("overlaps", [])
        if target_id and target_id in (overlap["support_id"], overlap["resistance_id"])
        and entry_price > 0
        and safe_float(
            (ranked_by_id.get(overlap["resistance_id"] if target_id == overlap["support_id"] else overlap["support_id"]) or {}).get("low")
        ) <= entry_price <= safe_float(
            (ranked_by_id.get(overlap["resistance_id"] if target_id == overlap["support_id"] else overlap["support_id"]) or {}).get("high")
        )
    ]
    if (
        conflicting
        and entry_validation.get("final_status") in ("BUY", "SELL")
        and entry_validation.get("setup_type") not in ("RESISTANCE_BREAKOUT_RETEST", "SUPPORT_BREAKDOWN_RETEST")
    ):
        overlap = conflicting[0]
        overlap["entry_conflict"] = True
        reason = f"BLOCKED_ZONE_CONFLICT: Entry berada di zona lawan {overlap['support_label']} / {overlap['resistance_label']}."
        entry_validation.update(
            final_status="WAIT", buy_status="WAIT", sell_status="WAIT", direction="NONE",
            reason=reason, wait_reasons=[reason], overlap_conflict=overlap,
        )

    data_state = str(market_data.get("data_freshness", "FRESH")).upper()
    blocked_plan = (
        "MARKET CLOSED" if mkt_status == "CLOSED"
        else "DATA UNAVAILABLE" if mkt_status == "DATA UNAVAILABLE"
        else f"DATA {data_state}" if data_state in ("STALE", "INVALID", "NO DATA")
        else None
    )
    entry_validation["provisional_plan"] = _build_snr_provisional_plan(
        entry_validation, ranked_zones, market_bias, current_price, symbol_spec, cfg, blocked=blocked_plan,
    )
    if live and mkt_status == "OPEN":
        entry_validation["focus_change"] = _track_snr_focus(
            entry_validation.get("target_zone"),
            ranked_zones,
            selection_reason=entry_validation.get("focus_selection_reason"),
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
    if (
        entry_validation.get("recent_closed_touch")
        and data_state == "FRESH"
        and isinstance(focus_zone, dict)
        and focus_zone.get("zone_id")
    ):
        price_action["recent_touch_zone_id"] = focus_zone["zone_id"]

    # Sinkronkan summary price action jika entry_validation sudah mengonfirmasi rejection/retest
    if entry_validation.get("setup_type") in ("SUPPORT_REJECTION", "RESISTANCE_REJECTION"):
        rej_det = dict(price_action.get("rejection_detail") or {})
        ev_rej = entry_validation.get("rejection_detail") or {}
        rej_det.update(ev_rej)
        n_m5 = len(entry_df) if (entry_df is not None and not getattr(entry_df, "empty", True)) else 0
        if rej_det.get("bar_index") is not None and n_m5 > 0:
            ev_ago = n_m5 - 1 - int(rej_det["bar_index"])
            rej_det["bars_ago"] = ev_ago
            rej_det["is_historical"] = bool(ev_ago > 1)
        price_action["rejection_code"] = rej_det.get("status", price_action["rejection_code"])
        price_action["rejection_is_historical"] = bool(rej_det.get("is_historical"))
        price_action["rejection_summary"] = rej_det.get("summary_status", "CONFIRMED")
        price_action["rejection_detail"] = rej_det
    elif entry_validation.get("setup_type") in ("RESISTANCE_BREAKOUT_RETEST", "SUPPORT_BREAKDOWN_RETEST"):
        price_action["breakout_code"] = "VALID_BREAKOUT"
        price_action["breakout_summary"] = "CONFIRMED"
        price_action["retest_code"] = "RETEST_CONFIRMED"
        price_action["retest_summary"] = "CONFIRMED"
        price_action["flip_status"] = "CONFIRMED_FLIP"
        if isinstance(price_action.get("retest_detail"), dict):
            price_action["retest_detail"]["flip_status"] = "CONFIRMED_FLIP"
    elif entry_validation.get("flip_status") in ("POTENTIAL_FLIP", "CONFIRMED_FLIP", "INVALID"):
        ev_fs = str(entry_validation["flip_status"]).upper()
        if price_action.get("flip_status") in (None, "", "NONE") or ev_fs == "INVALID":
            price_action["flip_status"] = ev_fs
            if isinstance(price_action.get("retest_detail"), dict):
                price_action["retest_detail"]["flip_status"] = ev_fs

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

    mt5_orders_snapshot = None
    mt5_reconciliation = None
    invalid_pending_warnings = []
    try:
        from modules.mt5_conn import (
            fetch_mt5_pending_orders_snapshot,
            find_pending_orders_on_invalid_zones,
            reconcile_pending_plan_with_mt5,
        )
        snap_ord = snapshot.get("mt5_orders_snapshot") if isinstance(snapshot, dict) else None
        if isinstance(snap_ord, dict) and "query_ok" in snap_ord and "orders" in snap_ord:
            mt5_orders_snapshot = snap_ord
        elif live:
            mt5_orders_snapshot = fetch_mt5_pending_orders_snapshot(
                symbol=market_data["symbol"],
                allow_live_query=True,
            )
        if isinstance(mt5_orders_snapshot, dict):
            snr_plan_for_recon = {"symbol": market_data["symbol"], "plan_status": "WAIT"}
            if decision.get("signal") in ("BUY", "SELL") and safe_float(decision.get("effective_entry") or decision.get("entry"), 0.0) > 0:
                tz_r = decision.get("target_zone") or {}
                snr_plan_for_recon = {
                    "symbol": market_data["symbol"],
                    "zone_id": tz_r.get("zone_id", ""),
                    "zone_type": tz_r.get("zone_type", ""),
                    "zone_low": safe_float(tz_r.get("low"), 0.0),
                    "zone_high": safe_float(tz_r.get("high"), 0.0),
                    "plan_status": "READY",
                    "direction": decision.get("signal"),
                    "order_type": "BUY LIMIT" if decision.get("signal") == "BUY" else "SELL LIMIT",
                    "entry_price": safe_float(decision.get("effective_entry") or decision.get("entry"), 0.0),
                    "sl": safe_float(decision.get("sl"), 0.0),
                    "tp1": safe_float(decision.get("tp1"), 0.0),
                    "tp2": safe_float(decision.get("tp2"), 0.0),
                }
            else:
                prov_r = entry_validation.get("provisional_plan")
                tz_r = entry_validation.get("target_zone") or {}
                if isinstance(prov_r, dict) and prov_r.get("status") == "PROVISIONAL" and safe_float(prov_r.get("entry"), 0.0) > 0 and mkt_status != "CLOSED":
                    p_dir_r = str(prov_r.get("direction") or "").upper()
                    snr_plan_for_recon = {
                        "symbol": market_data["symbol"],
                        "zone_id": tz_r.get("zone_id", ""),
                        "zone_type": tz_r.get("zone_type", ""),
                        "zone_low": safe_float(tz_r.get("low"), 0.0),
                        "zone_high": safe_float(tz_r.get("high"), 0.0),
                        "plan_status": "PROVISIONAL",
                        "direction": p_dir_r,
                        "order_type": "BUY LIMIT" if p_dir_r == "BUY" else ("SELL LIMIT" if p_dir_r == "SELL" else ""),
                        "entry_price": safe_float(prov_r.get("entry"), 0.0),
                        "sl": safe_float(prov_r.get("sl"), 0.0),
                        "tp1": safe_float(prov_r.get("tp1"), 0.0),
                        "tp2": safe_float(prov_r.get("tp2"), 0.0),
                    }
            mt5_reconciliation = reconcile_pending_plan_with_mt5(
                snr_plan_for_recon,
                mt5_snapshot=mt5_orders_snapshot,
                symbol=market_data["symbol"],
            )
            inv_pool_snr = [
                z_i for z_i in (list(ranked_zones.get("broken_zones") or []) + list(ranked_zones.get("invalidated_zones") or []))
                if not (isinstance(z_i, dict) and str(z_i.get("lifecycle_status") or z_i.get("status") or "").upper() == "RECLAIMED")
            ]
            valid_pool_snr = list(ranked_zones.get("active_supports") or []) + list(ranked_zones.get("active_resistances") or [])
            invalid_pending_warnings = find_pending_orders_on_invalid_zones(
                inv_pool_snr,
                mt5_snapshot=mt5_orders_snapshot,
                symbol=market_data["symbol"],
                default_source="SNR",
                active_valid_zones=valid_pool_snr,
            )
    except Exception as exc_mt5_snr:
        log_snr_error("snr_mt5_reconcile", exc_mt5_snr)

    breakout_watch = build_snr_breakout_watch(ranked_zones, price_action=price_action)

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
        "breakout_watch": breakout_watch,
        "entry_validation": entry_validation,
        "decision": decision,
        "radar": radar,
        "mt5_orders_snapshot": mt5_orders_snapshot,
        "mt5_reconciliation": mt5_reconciliation,
        "invalid_pending_warnings": invalid_pending_warnings,
    }
    try:
        snr_tp_list, snr_tp_best = generate_snr_trade_plans(
            ranked_zones=ranked_zones,
            entry_validation=entry_validation,
            decision=decision,
            market_bias=market_bias,
            current_price=current_price,
            symbol_spec=symbol_spec,
            cfg=cfg,
            market_status=mkt_status,
            data_freshness=data_state,
            m5_df=prepared_frames.get("M5"),
            mt5_orders_snapshot=mt5_orders_snapshot,
        )
        scan_payload["trade_plans"] = snr_tp_list
        scan_payload["best_trade_plan"] = snr_tp_best
    except Exception as exc_tp_snr:
        log_snr_error("generate_snr_trade_plans", exc_tp_snr)
        scan_payload["trade_plans"] = []
        scan_payload["best_trade_plan"] = None
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

