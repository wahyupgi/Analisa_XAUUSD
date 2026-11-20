# -*- coding: utf-8 -*-
"""
breakout_detection.py — Modul Khusus Identifikasi Breakout Zona Support & Resistance.

Membedakan 4 klasifikasi wajib:
1. VALID_BREAKOUT
2. FALSE_BREAKOUT
3. BREAKOUT_PENDING_CONFIRMATION
4. NO_BREAKOUT

Mempertimbangkan:
- Candle close di luar zona (bukan sekadar wick).
- Besarnya penetrasi terhadap zona (dalam satuan ATR).
- Ukuran body candle terhadap ATR dan range candle.
- Momentum dan konfirmasi candle penutupan lanjutan.
- Tick volume jika tersedia dan valid.
- Struktur harga sebelum breakout (pre-breakout structure).
"""
import numpy as np
from modules.snr.config import get_snr_config
from modules.snr.data_provider import prepare_snr_frame, safe_float, zone_confirmation_start
from modules.snr.momentum_analysis import analyze_momentum


def _evaluate_pre_breakout_structure(highs, lows, closes, idx_break, z_low, z_high, z_type, atr):
    """Mengevaluasi struktur harga tepat sebelum candle breakout (apakah ada kompresi/higher-low/lower-high)."""
    start = max(0, idx_break - 6)
    if idx_break <= start + 1:
        return "NORMAL_APPROACH"

    sub_highs = highs[start:idx_break]
    sub_lows = lows[start:idx_break]
    sub_span = float(sub_highs.max() - sub_lows.min())

    if sub_span <= (1.8 * atr):
        return "TIGHT_COMPRESSION_BUILDUP"

    if z_type == "RESISTANCE":
        if len(sub_lows) >= 3 and sub_lows[-1] > sub_lows[0]:
            return "ASCENDING_PRESSURE"
    else:
        if len(sub_highs) >= 3 and sub_highs[-1] < sub_highs[0]:
            return "DESCENDING_PRESSURE"

    return "IMPULSIVE_APPROACH"


def detect_zone_breakout(
    zone,
    df,
    momentum_result=None,
    config=None,
    has_unclosed_last_bar=False,
):
    """
    Mendeteksi status breakout pada satu zona Support atau Resistance menggunakan CLOSED candles.
    """
    cfg = config or get_snr_config()
    bo_cfg = cfg.get("breakout", {})
    touch_cfg = cfg.get("touch", {})

    valid_pen_atr = safe_float(bo_cfg.get("valid_close_penetration_atr", 0.15), 0.15)
    strong_pen_atr = safe_float(bo_cfg.get("strong_close_penetration_atr", 0.28), 0.28)
    invalid_pen_atr = safe_float(touch_cfg.get("invalidation_close_beyond_atr", 0.42), 0.42)
    min_body_atr = safe_float(bo_cfg.get("min_breakout_body_atr", 0.42), 0.42)
    min_body_ratio = safe_float(bo_cfg.get("min_breakout_body_ratio", 0.48), 0.48)
    vol_mult = safe_float(bo_cfg.get("volume_confirm_mult", 1.05), 1.05)
    lookback_bars = max(4, int(bo_cfg.get("lookback_bars", 12)))

    clean_df, _ = prepare_snr_frame(df, has_unclosed_last_bar=has_unclosed_last_bar)
    z_low = safe_float(zone.get("low"), 0.0)
    z_high = safe_float(zone.get("high"), z_low)
    z_mid = round((z_low + z_high) / 2.0, 2)
    z_type = str(zone.get("zone_type", "RESISTANCE")).upper()
    tf = str(zone.get("timeframe", "H1")).upper()
    ref_atr = max(0.10, safe_float(zone.get("ref_atr"), 1.0))

    if clean_df.empty or z_high <= 0:
        return {
            "zone_id": zone.get("zone_id", "UNKNOWN"),
            "timeframe": tf,
            "zone_type": z_type,
            "status": "NO_BREAKOUT",
            "break_type": "NONE",
            "potential_breakdown": False,
            "potential_breakout": False,
            "reclaim_detected": False,
            "zone_failed": False,
            "label": "NO BREAKOUT",
            "summary_status": "NOT CONFIRMED",
            "direction": "NONE",
            "breakout_bar_index": None,
            "breakout_bar_time": None,
            "close_outside_zone": False,
            "wick_penetration_only": False,
            "penetration_points": 0.0,
            "penetration_atr": 0.0,
            "body_atr": 0.0,
            "body_ratio": 0.0,
            "momentum_supported": False,
            "volume_supported": "N/A",
            "pre_breakout_structure": "N/A",
            "quality_score_15": 0.0,
            "reason": "Data candle tidak tersedia untuk evaluasi breakout.",
        }

    if momentum_result is None:
        momentum_result = analyze_momentum(clean_df, timeframe=tf, config=cfg, has_unclosed_last_bar=False)

    opens = clean_df["Open"].to_numpy(dtype=float)
    highs = clean_df["High"].to_numpy(dtype=float)
    lows = clean_df["Low"].to_numpy(dtype=float)
    closes = clean_df["Close"].to_numpy(dtype=float)
    atrs = clean_df["ATR14"].to_numpy(dtype=float)
    times = clean_df["Time"].tolist()
    n_bars = len(clean_df)

    start_idx = max(zone_confirmation_start(zone, clean_df), n_bars - lookback_bars)

    has_vol = bool(clean_df["HasTickVolume"].iloc[-1]) if "HasTickVolume" in clean_df.columns else False
    vols = clean_df["TickVolume"].to_numpy(dtype=float) if has_vol else None

    breakout_events = []
    wick_pierce_events = []

    for k in range(start_idx, n_bars):
        b_open = float(opens[k])
        b_high = float(highs[k])
        b_low = float(lows[k])
        b_close = float(closes[k])
        b_atr = max(0.10, safe_float(atrs[k], ref_atr))
        b_range = max(1e-6, b_high - b_low)
        b_body = abs(b_close - b_open)
        body_atr = b_body / b_atr
        body_ratio = b_body / b_range

        if z_type == "RESISTANCE":
            if b_close > z_high:
                pen_pts = b_close - z_high
                pen_atr = pen_pts / b_atr
                breakout_events.append({
                    "bar_index": int(k),
                    "bar_time": str(times[k]),
                    "direction": "BULLISH",
                    "close": round(b_close, 2),
                    "open": round(b_open, 2),
                    "high": round(b_high, 2),
                    "low": round(b_low, 2),
                    "penetration_points": round(pen_pts, 2),
                    "penetration_atr": round(pen_atr, 3),
                    "body_atr": round(body_atr, 3),
                    "body_ratio": round(body_ratio, 3),
                    "atr": round(b_atr, 4),
                    "is_directional_candle": bool(b_close > b_open),
                })
            elif b_high > z_high:
                wick_pen = (b_high - z_high) / b_atr
                wick_pierce_events.append({
                    "bar_index": int(k),
                    "bar_time": str(times[k]),
                    "direction": "BULLISH",
                    "wick_overshoot_atr": round(wick_pen, 3),
                    "close": round(b_close, 2),
                    "closed_below_zone": bool(b_close < z_low),
                })
        else:  # SUPPORT (Breakdown ke bawah)
            if b_close < z_low:
                pen_pts = z_low - b_close
                pen_atr = pen_pts / b_atr
                breakout_events.append({
                    "bar_index": int(k),
                    "bar_time": str(times[k]),
                    "direction": "BEARISH",
                    "close": round(b_close, 2),
                    "open": round(b_open, 2),
                    "high": round(b_high, 2),
                    "low": round(b_low, 2),
                    "penetration_points": round(pen_pts, 2),
                    "penetration_atr": round(pen_atr, 3),
                    "body_atr": round(body_atr, 3),
                    "body_ratio": round(body_ratio, 3),
                    "atr": round(b_atr, 4),
                    "is_directional_candle": bool(b_close < b_open),
                })
            elif b_low < z_low:
                wick_pen = (z_low - b_low) / b_atr
                wick_pierce_events.append({
                    "bar_index": int(k),
                    "bar_time": str(times[k]),
                    "direction": "BEARISH",
                    "wick_overshoot_atr": round(wick_pen, 3),
                    "close": round(b_close, 2),
                    "closed_above_zone": bool(b_close > z_high),
                })

    last_close = float(closes[-1])
    last_atr = max(0.10, safe_float(atrs[-1], ref_atr))
    expected_dir = "BULLISH" if z_type == "RESISTANCE" else "BEARISH"

    # Kasus 1: Tidak pernah ada candle closed di luar zona
    if not breakout_events:
        if wick_pierce_events:
            last_w = wick_pierce_events[-1]
            overshoot = float(last_w["wick_overshoot_atr"])
            w_idx = int(last_w["bar_index"])
            # Jangan langsung menetapkan FALSE_BREAKOUT hanya dari satu candle wick;
            # wajib ada candle closed berikutnya yang menembus/mengonfirmasi kegagalan balik ke dalam/sisi berlawanan.
            has_follow_through = w_idx < (n_bars - 1)
            follow_confirmed = False
            if has_follow_through:
                next_close = float(closes[w_idx + 1])
                if z_type == "RESISTANCE":
                    follow_confirmed = next_close < z_mid and last_close <= z_high
                else:
                    follow_confirmed = next_close > z_mid and last_close >= z_low
            is_deep_trap = overshoot >= 0.30 and follow_confirmed and (
                last_w.get("closed_below_zone", False) or last_w.get("closed_above_zone", False)
            )
            if is_deep_trap:
                false_bt = "FALSE_BREAKOUT" if z_type == "RESISTANCE" else "FALSE_BREAKDOWN"
                return {
                    "zone_id": zone.get("zone_id", "UNKNOWN"),
                    "timeframe": tf,
                    "zone_type": z_type,
                    "status": "FALSE_BREAKOUT",
                    "break_type": false_bt,
                    "potential_breakdown": False,
                    "potential_breakout": False,
                    "reclaim_detected": False,
                    "zone_failed": False,
                    "label": "FALSE BREAKOUT" if z_type == "RESISTANCE" else "FALSE BREAKDOWN",
                    "summary_status": "FALSE BREAKOUT",
                    "direction": expected_dir,
                    "breakout_bar_index": last_w["bar_index"],
                    "breakout_bar_time": last_w["bar_time"],
                    "close_outside_zone": False,
                    "wick_penetration_only": True,
                    "penetration_points": round(overshoot * last_atr, 2),
                    "penetration_atr": round(overshoot, 3),
                    "body_atr": 0.0,
                    "body_ratio": 0.0,
                    "momentum_supported": False,
                    "volume_supported": "N/A",
                    "pre_breakout_structure": "WICK_SWEEP_TRAP",
                    "quality_score_15": 4.0,
                    "reason": (
                        f"False {'breakout' if z_type == 'RESISTANCE' else 'breakdown'} (wick trap): ekor menembus {overshoot:.2f} ATR melewati {z_type} "
                        "tetapi gagal close di luar zona dan berbalik arah."
                    ),
                }

            return {
                "zone_id": zone.get("zone_id", "UNKNOWN"),
                "timeframe": tf,
                "zone_type": z_type,
                "status": "NO_BREAKOUT",
                "break_type": "NONE",
                "potential_breakdown": False,
                "potential_breakout": False,
                "reclaim_detected": False,
                "zone_failed": False,
                "label": "NO BREAKOUT",
                "summary_status": "NOT CONFIRMED",
                "direction": "NONE",
                "breakout_bar_index": last_w["bar_index"],
                "breakout_bar_time": last_w["bar_time"],
                "close_outside_zone": False,
                "wick_penetration_only": True,
                "penetration_points": round(overshoot * last_atr, 2),
                "penetration_atr": round(overshoot, 3),
                "body_atr": 0.0,
                "body_ratio": 0.0,
                "momentum_supported": False,
                "volume_supported": "N/A",
                "pre_breakout_structure": "WICK_TEST_ONLY",
                "quality_score_15": 0.0,
                "reason": (
                    f"Hanya penetrasi wick ({overshoot:.2f} ATR) tanpa penutupan candle (close) di luar zona; "
                    "bukan breakout."
                ),
            }

        return {
            "zone_id": zone.get("zone_id", "UNKNOWN"),
            "timeframe": tf,
            "zone_type": z_type,
            "status": "NO_BREAKOUT",
            "break_type": "NONE",
            "potential_breakdown": False,
            "potential_breakout": False,
            "reclaim_detected": False,
            "zone_failed": False,
            "label": "NO BREAKOUT",
            "summary_status": "NOT CONFIRMED",
            "direction": "NONE",
            "breakout_bar_index": None,
            "breakout_bar_time": None,
            "close_outside_zone": False,
            "wick_penetration_only": False,
            "penetration_points": 0.0,
            "penetration_atr": 0.0,
            "body_atr": 0.0,
            "body_ratio": 0.0,
            "momentum_supported": False,
            "volume_supported": "N/A",
            "pre_breakout_structure": "INTACT",
            "quality_score_15": 0.0,
            "reason": f"Harga belum menembus batas {z_type} ({z_low:.2f}-{z_high:.2f}).",
        }

    # Kasus 2: Ada candle yang sempat close di luar zona
    first_bo = breakout_events[0]
    latest_bo = breakout_events[-1]
    bo_idx = int(latest_bo["bar_index"])
    ref_bo = latest_bo

    # Cek apakah setelah close di luar zona, harga gagal mempertahankan breakout (FALSE BREAKOUT)
    # Bedakan false breakout (kegagalan terkonfirmasi: close melewati sisi berlawanan zona atau 2+ close di sisi dalam)
    # vs pullback wajar menuju tepi zona untuk retest (JANGAN tandai FALSE hanya karena 1 sentuhan/pullback di tepi zona).
    reclaimed_inside = False
    bars_since_first = (n_bars - 1) - int(first_bo["bar_index"])
    closes_after_first = [float(closes[m]) for m in range(int(first_bo["bar_index"]) + 1, n_bars)]
    if z_type == "RESISTANCE":
        mid_fail_count = sum(1 for c in closes_after_first if c < z_mid)
        if last_close < z_low:
            reclaimed_inside = True
        elif last_close < z_mid and mid_fail_count >= 2:
            reclaimed_inside = True
        elif bars_since_first <= 4 and any(c < z_low for c in closes_after_first) and last_close <= z_high:
            reclaimed_inside = True
            bo_idx = int(first_bo["bar_index"])
            ref_bo = first_bo
    else:  # SUPPORT
        mid_fail_count = sum(1 for c in closes_after_first if c > z_mid)
        if last_close > z_high:
            reclaimed_inside = True
        elif last_close > z_mid and mid_fail_count >= 2:
            reclaimed_inside = True
        elif bars_since_first <= 4 and any(c > z_high for c in closes_after_first) and last_close >= z_low:
            reclaimed_inside = True
            bo_idx = int(first_bo["bar_index"])
            ref_bo = first_bo

    pre_struct = _evaluate_pre_breakout_structure(highs, lows, closes, bo_idx, z_low, z_high, z_type, last_atr)

    # Evaluasi dukungan tick volume pada candle breakout jika tersedia
    vol_supported = "N/A"
    if vols is not None and bo_idx < len(vols):
        bo_v = safe_float(vols[bo_idx], 0.0)
        v_start = max(0, bo_idx - 15)
        hist_v = [float(v) for v in vols[v_start:bo_idx] if not np.isnan(v) and float(v) > 0]
        if bo_v > 0 and hist_v:
            avg_v = sum(hist_v) / float(len(hist_v))
            vol_supported = bool((bo_v / max(1e-6, avg_v)) >= vol_mult)

    if reclaimed_inside:
        prior_confirmed_breakout = any(
            float(event["penetration_atr"]) >= valid_pen_atr
            and bool(event["is_directional_candle"])
            and float(event["body_atr"]) >= min_body_atr
            and float(event["body_ratio"]) >= min_body_ratio
            for event in breakout_events
        )
        max_consec_outside = 0
        run_out = 0
        prev_idx = None
        for event in breakout_events:
            cur_i = int(event["bar_index"])
            if prev_idx is not None and cur_i == prev_idx + 1:
                run_out += 1
            else:
                run_out = 1
            prev_idx = cur_i
            if run_out > max_consec_outside:
                max_consec_outside = run_out

        if not prior_confirmed_breakout:
            for idx_event in range(1, len(breakout_events)):
                previous = breakout_events[idx_event - 1]
                current = breakout_events[idx_event]
                if (
                    int(current["bar_index"]) == int(previous["bar_index"]) + 1
                    and float(current["penetration_atr"]) >= valid_pen_atr
                    and max(
                        float(previous["penetration_atr"]),
                        float(current["penetration_atr"]),
                    ) >= strong_pen_atr
                    and max(
                        float(previous["body_atr"]),
                        float(current["body_atr"]),
                    ) >= min_body_atr * 0.85
                ):
                    prior_confirmed_breakout = True
                    break

        max_pen_out = max((float(event["penetration_atr"]) for event in breakout_events), default=0.0)
        already_broken_zone = bool(zone.get("is_broken")) or str(zone.get("lifecycle_status", "")).upper() in (
            "BROKEN", "RETEST", "RECLAIM_PENDING", "RECLAIMED"
        )
        # Bedakan FALSE BREAKOUT/BREAKDOWN (1 bar trap dangkal yang langsung gagal) vs
        # RECLAIM (harga sempat menerima/menembus kuat di luar zona lalu kembali mereclaim).
        reclaim_detected = bool(
            already_broken_zone
            or (prior_confirmed_breakout and (max_consec_outside >= 2 or max_pen_out >= invalid_pen_atr))
        )
        if reclaim_detected:
            bt_label = "RECLAIM_AFTER_BREAKOUT" if z_type == "RESISTANCE" else "RECLAIM_AFTER_BREAKDOWN"
            disp_lbl = "RECLAIM AFTER BREAKOUT" if z_type == "RESISTANCE" else "RECLAIM AFTER BREAKDOWN"
            reason_txt = (
                f"Reclaim pasca-{'breakout' if z_type == 'RESISTANCE' else 'breakdown'} pada {z_type}: "
                f"zona sempat ditembus kuat ({ref_bo['close']:.2f}, max {max_pen_out:.2f} ATR) lalu kembali ditutup di {last_close:.2f}; "
                "zona telah gagal dan tidak dipromosikan ulang sebagai S/R aktif."
            )
        else:
            bt_label = "FALSE_BREAKOUT" if z_type == "RESISTANCE" else "FALSE_BREAKDOWN"
            disp_lbl = "FALSE BREAKOUT" if z_type == "RESISTANCE" else "FALSE BREAKDOWN"
            reason_txt = (
                f"False {'breakout' if z_type == 'RESISTANCE' else 'breakdown'} pada {z_type}: "
                f"candle sempat close di luar zona ({ref_bo['close']:.2f}) tanpa penerimaan lanjutan "
                f"dan langsung kembali ditutup di {last_close:.2f}."
            )
        return {
            "zone_id": zone.get("zone_id", "UNKNOWN"),
            "timeframe": tf,
            "zone_type": z_type,
            "status": "FALSE_BREAKOUT",
            "break_type": bt_label,
            "potential_breakdown": False,
            "potential_breakout": False,
            "reclaim_detected": reclaim_detected,
            "zone_failed": reclaim_detected,
            "label": disp_lbl,
            "summary_status": "RECLAIM" if reclaim_detected else "FALSE BREAKOUT",
            "direction": expected_dir,
            "breakout_bar_index": bo_idx,
            "breakout_bar_time": ref_bo["bar_time"],
            "prior_confirmed_breakout": prior_confirmed_breakout,
            "max_consecutive_closes_outside": int(max_consec_outside),
            "close_outside_zone": False,
            "wick_penetration_only": False,
            "penetration_points": ref_bo["penetration_points"],
            "penetration_atr": ref_bo["penetration_atr"],
            "body_atr": ref_bo["body_atr"],
            "body_ratio": ref_bo["body_ratio"],
            "momentum_supported": False,
            "volume_supported": vol_supported,
            "pre_breakout_structure": pre_struct,
            "quality_score_15": 3.5,
            "reason": reason_txt,
        }

    # Harga masih ditutup di luar zona pada bar closed terakhir -> evaluasi kualitas & momentum breakout
    best_bo = max(breakout_events, key=lambda e: (e["body_atr"], e["penetration_atr"]))
    curr_pen_pts = (last_close - z_high) if z_type == "RESISTANCE" else (z_low - last_close)
    curr_pen_atr = curr_pen_pts / last_atr
    max_pen_atr = max(curr_pen_atr, float(best_bo["penetration_atr"]))

    conf_start_idx = zone_confirmation_start(zone, clean_df)
    consec_outside = 0
    for idx_c in range(n_bars - 1, conf_start_idx - 1, -1):
        if z_type == "RESISTANCE" and float(closes[idx_c]) > z_high:
            consec_outside += 1
        elif z_type == "SUPPORT" and float(closes[idx_c]) < z_low:
            consec_outside += 1
        else:
            break
    initial_bo_idx = max(conf_start_idx, n_bars - consec_outside) if consec_outside > 0 else int(first_bo["bar_index"])
    initial_bo_time = str(times[initial_bo_idx]) if 0 <= initial_bo_idx < n_bars else first_bo["bar_time"]
    is_historical_break = bool(consec_outside > lookback_bars)

    candle_has_momentum = (
        bool(best_bo["is_directional_candle"])
        and float(best_bo["body_atr"]) >= min_body_atr
        and float(best_bo["body_ratio"]) >= min_body_ratio
    )
    tf_mom_aligned = (
        momentum_result.get("direction") == expected_dir
        and momentum_result.get("strength") in ("STRONG", "MODERATE")
        and not bool(momentum_result.get("momentum_weakening", False))
    )
    momentum_supported = bool(candle_has_momentum or (consec_outside >= 2 and tf_mom_aligned))

    # Jika penetrasi close cukup dalam dan didukung momentum (atau 2+ candle close berturut-turut dengan penetrasi kuat)
    if curr_pen_atr >= valid_pen_atr and (
        candle_has_momentum
        or (consec_outside >= 2 and max_pen_atr >= strong_pen_atr and float(best_bo["body_atr"]) >= (min_body_atr * 0.85))
    ):
        q_score = 11.5
        if max_pen_atr >= strong_pen_atr:
            q_score += 1.5
        if candle_has_momentum and tf_mom_aligned:
            q_score += 1.2
        if vol_supported is True:
            q_score += 0.8
        q_score = min(15.0, round(q_score, 2))
        is_support = (z_type == "SUPPORT")
        rep_bo_idx = initial_bo_idx if is_historical_break else int(best_bo["bar_index"])
        rep_bo_time = initial_bo_time if is_historical_break else best_bo["bar_time"]
        return {
            "zone_id": zone.get("zone_id", "UNKNOWN"),
            "timeframe": tf,
            "zone_type": z_type,
            "status": "VALID_BREAKOUT",
            "breakout_state": "BROKEN" if is_historical_break else "CONFIRMED",
            "break_type": "CONFIRMED_BREAKDOWN" if is_support else "CONFIRMED_BREAKOUT",
            "potential_breakdown": False,
            "potential_breakout": False,
            "reclaim_detected": False,
            "zone_failed": True,
            "label": "VALID BREAKDOWN" if is_support else "VALID BREAKOUT",
            "summary_status": "CONFIRMED",
            "direction": expected_dir,
            "breakout_bar_index": rep_bo_idx,
            "breakout_bar_time": rep_bo_time,
            "initial_break_bar_index": initial_bo_idx,
            "initial_break_bar_time": initial_bo_time,
            "is_historical_break": is_historical_break,
            "close_outside_zone": True,
            "wick_penetration_only": False,
            "penetration_points": round(curr_pen_pts, 2),
            "penetration_atr": round(curr_pen_atr, 3),
            "body_atr": best_bo["body_atr"],
            "body_ratio": best_bo["body_ratio"],
            "consecutive_closes_outside": int(consec_outside),
            "momentum_supported": True,
            "volume_supported": vol_supported,
            "pre_breakout_structure": pre_struct,
            "quality_score_15": q_score,
            "reason": (
                f"Valid {expected_dir} {'breakdown' if is_support else 'breakout'} melewati {z_type} ({z_low:.2f}-{z_high:.2f}): "
                f"close={last_close:.2f} (penetrasi {curr_pen_atr:.2f} ATR, body {best_bo['body_atr']:.2f} ATR)."
            ),
        }

    # Jika sudah close di luar zona tetapi penetrasi tipis atau sedang pullback ke tepi zona pasca-breakout
    prior_confirmed_bo = any(
        float(event["penetration_atr"]) >= valid_pen_atr
        and bool(event["is_directional_candle"])
        and float(event["body_atr"]) >= min_body_atr
        and float(event["body_ratio"]) >= min_body_ratio
        for event in breakout_events
    )
    is_support = (z_type == "SUPPORT")
    if prior_confirmed_bo:
        # Breakdown/breakout SUDAH terkonfirmasi pada closed candle sebelumnya dan belum tereclaim:
        # pertahankan status VALID_BREAKOUT (CONFIRMED), jangan diturunkan menjadi potential akibat pullback/intrabar.
        q_score = 11.5
        if max_pen_atr >= strong_pen_atr:
            q_score += 1.5
        if candle_has_momentum and tf_mom_aligned:
            q_score += 1.2
        if vol_supported is True:
            q_score += 0.8
        q_score = min(15.0, round(q_score, 2))
        rep_bo_idx = initial_bo_idx if is_historical_break else int(best_bo["bar_index"])
        rep_bo_time = initial_bo_time if is_historical_break else best_bo["bar_time"]
        return {
            "zone_id": zone.get("zone_id", "UNKNOWN"),
            "timeframe": tf,
            "zone_type": z_type,
            "status": "VALID_BREAKOUT",
            "breakout_state": "BROKEN" if is_historical_break else "CONFIRMED",
            "break_type": "CONFIRMED_BREAKDOWN" if is_support else "CONFIRMED_BREAKOUT",
            "potential_breakdown": False,
            "potential_breakout": False,
            "prior_confirmed_breakout": True,
            "reclaim_detected": False,
            "zone_failed": True,
            "label": "VALID BREAKDOWN" if is_support else "VALID BREAKOUT",
            "summary_status": "CONFIRMED",
            "direction": expected_dir,
            "breakout_bar_index": rep_bo_idx,
            "breakout_bar_time": rep_bo_time,
            "initial_break_bar_index": initial_bo_idx,
            "initial_break_bar_time": initial_bo_time,
            "is_historical_break": is_historical_break,
            "close_outside_zone": True,
            "wick_penetration_only": False,
            "penetration_points": round(curr_pen_pts, 2),
            "penetration_atr": round(curr_pen_atr, 3),
            "body_atr": best_bo["body_atr"],
            "body_ratio": best_bo["body_ratio"],
            "consecutive_closes_outside": int(consec_outside),
            "momentum_supported": True,
            "volume_supported": vol_supported,
            "pre_breakout_structure": pre_struct,
            "quality_score_15": q_score,
            "reason": (
                f"Valid {expected_dir} {'breakdown' if is_support else 'breakout'} terkonfirmasi melewati {z_type} ({z_low:.2f}-{z_high:.2f}); "
                f"harga bertahan di luar zona (close={last_close:.2f})."
            ),
        }

    pend_bo_idx = initial_bo_idx if is_historical_break else int(latest_bo["bar_index"])
    pend_bo_time = initial_bo_time if is_historical_break else latest_bo["bar_time"]
    return {
        "zone_id": zone.get("zone_id", "UNKNOWN"),
        "timeframe": tf,
        "zone_type": z_type,
        "status": "BREAKOUT_PENDING_CONFIRMATION",
        "breakout_state": "BREAKDOWN_PENDING" if is_support else "BREAKOUT_PENDING",
        "break_type": "POTENTIAL_BREAKDOWN" if is_support else "POTENTIAL_BREAKOUT",
        "potential_breakdown": is_support,
        "potential_breakout": not is_support,
        "prior_confirmed_breakout": bool(prior_confirmed_bo),
        "reclaim_detected": False,
        "zone_failed": bool(prior_confirmed_bo),
        "label": "BREAKDOWN PENDING" if is_support else "BREAKOUT PENDING",
        "summary_status": "PENDING CONFIRMATION",
        "direction": expected_dir,
        "breakout_bar_index": pend_bo_idx,
        "breakout_bar_time": pend_bo_time,
        "initial_break_bar_index": initial_bo_idx,
        "initial_break_bar_time": initial_bo_time,
        "is_historical_break": is_historical_break,
        "close_outside_zone": bool(curr_pen_pts > 0),
        "wick_penetration_only": False,
        "penetration_points": round(curr_pen_pts, 2),
        "penetration_atr": round(curr_pen_atr, 3),
        "body_atr": best_bo["body_atr"],
        "body_ratio": best_bo["body_ratio"],
        "consecutive_closes_outside": int(consec_outside),
        "momentum_supported": bool(momentum_supported),
        "volume_supported": vol_supported,
        "pre_breakout_structure": pre_struct,
        "quality_score_15": 7.5,
        "reason": (
            f"Potensi {'breakdown Support' if is_support else 'breakout Resistance'} belum terkonfirmasi penuh "
            f"(penetrasi={curr_pen_atr:.2f} ATR, body={best_bo['body_atr']:.2f} ATR, "
            f"momentum_supported={momentum_supported}); menunggu candle konfirmasi lanjutan."
        ),
    }


def classify_snr_breakout_state(zone):
    """
    Klasifikasi kanonis status breakout zona SNR (single source of truth antara `scan snr` & `pantau`):
    - BREAKDOWN_PENDING / BREAKOUT_PENDING : belum terkonfirmasi closed candle TF asal (TIDAK pernah untuk zona broken)
    - CONFIRMED                            : baru saja terkonfirmasi break oleh candle closed TF asal
    - RETEST_WATCH                         : zona confirmed broken yang sedang menunggu / menjalani retest role-reversal
    - BROKEN                               : zona sudah confirmed broken / invalidated / reclaimed
    - NONE                                 : belum ada breakout
    """
    if not isinstance(zone, dict):
        return {
            "state": "NONE",
            "label": "NONE",
            "direction_word": "BREAKOUT",
            "is_pending": False,
            "is_confirmed": False,
            "is_retest_watch": False,
            "is_broken": False,
            "is_historical": False,
        }
    z_type = str(zone.get("zone_type") or "SUPPORT").upper()
    dir_word = "BREAKDOWN" if z_type == "SUPPORT" else "BREAKOUT"
    life = str(zone.get("lifecycle_status") or zone.get("status") or "ACTIVE").upper()
    bo = zone.get("breakout_info") if isinstance(zone.get("breakout_info"), dict) else {}
    rt = zone.get("retest_info") if isinstance(zone.get("retest_info"), dict) else {}
    bo_status = str(bo.get("status") or "NO_BREAKOUT").upper()
    rt_status = str(rt.get("status") or "NO_RETEST").upper()
    bars_ago = int(safe_float(zone.get("lifecycle_bars_ago"), 0))
    d_atr = safe_float(zone.get("distance_atr"), 0.0)
    is_hist = bool(
        zone.get("lifecycle_is_historical")
        or zone.get("is_distant_historical")
        or bo.get("is_historical_break")
        or d_atr > 12.0
    )
    is_confirmed_broken = bool(
        zone.get("is_broken")
        or zone.get("is_invalidated")
        or bo_status == "VALID_BREAKOUT"
        or bo.get("zone_failed")
        or bo.get("reclaim_detected")
        or life in ("BROKEN", "RETEST", "RECLAIM_PENDING", "RECLAIMED", "INVALIDATED")
    )

    if is_confirmed_broken:
        if life in ("RECLAIMED", "RECLAIM_PENDING") or (bo.get("reclaim_detected") and life in ("BROKEN", "RETEST", "INVALIDATED")):
            # RECLAIMED hanya jika lifecycle (urutan candle closed pasca-break) mengonfirmasi; reclaim_detected saja = PENDING
            tag_sub = life.replace("_", " ") if life in ("RECLAIMED", "RECLAIM_PENDING") else "RECLAIM PENDING"
            return {
                "state": "BROKEN",
                "label": f"BROKEN ({tag_sub})",
                "direction_word": dir_word,
                "is_pending": False,
                "is_confirmed": False,
                "is_retest_watch": False,
                "is_broken": True,
                "is_historical": bool(is_hist or (life == "RECLAIMED" and bars_ago > 1)),
            }
        if (
            (life == "RETEST" or rt_status in ("RETEST_WAITING", "RETEST_IN_PROGRESS", "RETEST_CONFIRMED"))
            and not zone.get("is_invalidated")
            and not zone.get("is_distant_historical")
            and d_atr <= 6.0
        ):
            rt_tag = "RETEST WATCH (CONFIRMED)" if rt_status == "RETEST_CONFIRMED" else "RETEST WATCH"
            return {
                "state": "RETEST_WATCH",
                "label": rt_tag,
                "direction_word": dir_word,
                "is_pending": False,
                "is_confirmed": True,
                "is_retest_watch": True,
                "is_broken": True,
                "is_historical": False,
            }
        if not is_hist and bars_ago <= 1 and life == "BROKEN" and bo_status == "VALID_BREAKOUT":
            return {
                "state": "CONFIRMED",
                "label": f"{dir_word} CONFIRMED",
                "direction_word": dir_word,
                "is_pending": False,
                "is_confirmed": True,
                "is_retest_watch": False,
                "is_broken": True,
                "is_historical": False,
            }
        return {
            "state": "BROKEN",
            "label": f"BROKEN ({z_type})",
            "direction_word": dir_word,
            "is_pending": False,
            "is_confirmed": False,
            "is_retest_watch": False,
            "is_broken": True,
            "is_historical": is_hist,
        }

    if life in ("BREAKDOWN_PENDING", "BREAKOUT_PENDING") or bo_status == "BREAKOUT_PENDING_CONFIRMATION":
        pend_st = "BREAKDOWN_PENDING" if z_type == "SUPPORT" else "BREAKOUT_PENDING"
        pend_lbl = "BREAKDOWN PENDING" if z_type == "SUPPORT" else "BREAKOUT PENDING"
        return {
            "state": pend_st,
            "label": pend_lbl,
            "direction_word": dir_word,
            "is_pending": True,
            "is_confirmed": False,
            "is_retest_watch": False,
            "is_broken": False,
            "is_historical": is_hist,
        }

    return {
        "state": "NONE",
        "label": "NONE",
        "direction_word": dir_word,
        "is_pending": False,
        "is_confirmed": False,
        "is_retest_watch": False,
        "is_broken": False,
        "is_historical": False,
    }


def build_snr_breakout_watch(ranked_zones, price_action=None):
    """
    Bangun ringkasan breakout watch terpadu (CONFIRMED / RETEST WATCH / BREAKDOWN PENDING / BROKEN)
    agar `scan snr` dan `pantau` menampilkan status yang identik dan tidak pernah menampilkan
    POTENTIAL BREAKDOWN pada zona yang sudah confirmed broken.
    """
    rz = ranked_zones if isinstance(ranked_zones, dict) else {}
    pa = price_action if isinstance(price_action, dict) else {}

    confirmed = []
    retest_watch = []
    pending = []
    broken = []
    historical = []

    seen_ids = set()
    pool = (
        list(rz.get("broken_zones") or [])
        + list(rz.get("pending_break_zones") or [])
        + list(rz.get("reclaimed_zones") or [])
        + list(rz.get("invalidated_zones") or [])
    )
    for z in pool:
        if not isinstance(z, dict):
            continue
        zid = z.get("zone_id") or (
            z.get("timeframe"),
            z.get("zone_type"),
            round(safe_float(z.get("low")), 2),
            round(safe_float(z.get("high")), 2),
        )
        if zid in seen_ids:
            continue
        seen_ids.add(zid)

        info = classify_snr_breakout_state(z)
        st = info["state"]
        if st == "NONE":
            continue

        d_atr = safe_float(z.get("distance_atr"), 0.0)
        entry = {
            "zone_id": z.get("zone_id"),
            "timeframe": str(z.get("timeframe") or "").upper(),
            "zone_type": str(z.get("zone_type") or "").upper(),
            "low": safe_float(z.get("low"), 0.0),
            "high": safe_float(z.get("high"), 0.0),
            "distance_points": safe_float(z.get("distance_points"), 0.0),
            "distance_atr": d_atr,
            "state": st,
            "label": info["label"],
            "lifecycle_status": str(z.get("lifecycle_status") or z.get("status") or "").upper(),
            "event_time": str(
                z.get("lifecycle_event_time")
                or (z.get("breakout_info") or {}).get("breakout_bar_time")
                or (z.get("m5_breakout_info") or {}).get("breakout_bar_time")
                or ""
            ).strip(),
            "is_historical": bool(info["is_historical"]),
            "zone": z,
        }

        if z.get("is_distant_historical") or d_atr > 12.0:
            historical.append(entry)
            continue

        if st == "CONFIRMED":
            confirmed.append(entry)
        elif st == "RETEST_WATCH":
            retest_watch.append(entry)
        elif st in ("BREAKDOWN_PENDING", "BREAKOUT_PENDING"):
            if not info["is_historical"]:
                pending.append(entry)
            else:
                historical.append(entry)
        elif st == "BROKEN":
            broken.append(entry)
            if info["is_historical"]:
                historical.append(entry)

    for bucket in (confirmed, retest_watch, pending, broken):
        bucket.sort(key=lambda item: (item["distance_points"], item["timeframe"], str(item["zone_id"] or "")))

    def _fmt_watch_line(item):
        rng = f"{item['low']:.2f}-{item['high']:.2f}"
        return f"{item['label']} ({item['timeframe']} {item['zone_type']} {rng})".replace("  ", " ").strip()

    primary = None
    if confirmed:
        primary = confirmed[0]
    elif retest_watch:
        primary = retest_watch[0]
    elif pending:
        primary = pending[0]
    else:
        recent_broken = [b for b in broken if not b["is_historical"]]
        if recent_broken:
            primary = recent_broken[0]
        else:
            nearby_broken = [b for b in broken if b["distance_atr"] <= 6.0]
            if nearby_broken:
                primary = nearby_broken[0]

    if primary is not None:
        return {
            "state": primary["state"],
            "label": primary["label"],
            "summary_line": _fmt_watch_line(primary),
            "zone_id": primary["zone_id"],
            "pending": pending,
            "confirmed": confirmed,
            "retest_watch": retest_watch,
            "broken": broken,
            "historical": historical,
        }

    # Fallback price_action hanya jika non-historical dan tidak bertentangan dengan zona broken
    pa_code = str(pa.get("breakout_code") or "NO_BREAKOUT").upper()
    if pa_code in ("VALID_BREAKOUT", "BREAKOUT_PENDING_CONFIRMATION") and not pa.get("breakout_is_historical"):
        pa_dir = str(pa.get("breakout_direction") or "").upper()
        dir_w = "BREAKDOWN" if pa_dir == "BEARISH" else "BREAKOUT"
        if pa_code == "VALID_BREAKOUT":
            st_f, lbl_f = "CONFIRMED", f"{dir_w} CONFIRMED"
        else:
            st_f = "BREAKDOWN_PENDING" if pa_dir == "BEARISH" else "BREAKOUT_PENDING"
            lbl_f = f"{dir_w} PENDING"
        pa_lbl = str(pa.get("breakout_zone_label") or "").strip()
        sum_ln = f"{lbl_f} ({pa_lbl})" if pa_lbl and pa_lbl != "NONE" else lbl_f
        return {
            "state": st_f,
            "label": lbl_f,
            "summary_line": sum_ln,
            "zone_id": None,
            "pending": pending,
            "confirmed": confirmed,
            "retest_watch": retest_watch,
            "broken": broken,
            "historical": historical,
        }

    return {
        "state": "NONE",
        "label": "",
        "summary_line": "",
        "zone_id": None,
        "pending": pending,
        "confirmed": confirmed,
        "retest_watch": retest_watch,
        "broken": broken,
        "historical": historical,
    }

