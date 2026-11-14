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
from modules.snr.data_provider import prepare_snr_frame, safe_float
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

    valid_pen_atr = safe_float(bo_cfg.get("valid_close_penetration_atr", 0.15), 0.15)
    strong_pen_atr = safe_float(bo_cfg.get("strong_close_penetration_atr", 0.28), 0.28)
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

    conf_idx = int(zone.get("confirmed_index", zone.get("formation_index", 0)))
    start_idx = max(conf_idx + 1, n_bars - lookback_bars)
    if start_idx >= n_bars:
        start_idx = max(0, n_bars - min(lookback_bars, n_bars))

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
            # Jika wick menembus cukup dalam (>= 0.30 ATR) namun gagal close di luar dan bahkan ditutup berbalik tajam
            is_deep_trap = overshoot >= 0.30 and (
                last_w.get("closed_below_zone", False) or last_w.get("closed_above_zone", False)
            )
            if is_deep_trap:
                return {
                    "zone_id": zone.get("zone_id", "UNKNOWN"),
                    "timeframe": tf,
                    "zone_type": z_type,
                    "status": "FALSE_BREAKOUT",
                    "label": "FALSE BREAKOUT",
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
                        f"False breakout (wick trap): ekor menembus {overshoot:.2f} ATR melewati {z_type} "
                        "tetapi gagal close di luar zona dan berbalik arah."
                    ),
                }

            return {
                "zone_id": zone.get("zone_id", "UNKNOWN"),
                "timeframe": tf,
                "zone_type": z_type,
                "status": "NO_BREAKOUT",
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
    bo_idx = int(first_bo["bar_index"])

    # Cek apakah setelah close di luar zona, harga justru kembali masuk/menutup di dalam/bawah zona (FALSE BREAKOUT)
    reclaimed_inside = False
    if z_type == "RESISTANCE":
        if last_close <= z_high:
            reclaimed_inside = True
        else:
            for m in range(bo_idx + 1, n_bars):
                if float(closes[m]) < z_mid:
                    reclaimed_inside = True
                    break
    else:  # SUPPORT
        if last_close >= z_low:
            reclaimed_inside = True
        else:
            for m in range(bo_idx + 1, n_bars):
                if float(closes[m]) > z_mid:
                    reclaimed_inside = True
                    break

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
        return {
            "zone_id": zone.get("zone_id", "UNKNOWN"),
            "timeframe": tf,
            "zone_type": z_type,
            "status": "FALSE_BREAKOUT",
            "label": "FALSE BREAKOUT",
            "summary_status": "FALSE BREAKOUT",
            "direction": expected_dir,
            "breakout_bar_index": bo_idx,
            "breakout_bar_time": first_bo["bar_time"],
            "close_outside_zone": False,
            "wick_penetration_only": False,
            "penetration_points": first_bo["penetration_points"],
            "penetration_atr": first_bo["penetration_atr"],
            "body_atr": first_bo["body_atr"],
            "body_ratio": first_bo["body_ratio"],
            "momentum_supported": False,
            "volume_supported": vol_supported,
            "pre_breakout_structure": pre_struct,
            "quality_score_15": 3.5,
            "reason": (
                f"False breakout pada {z_type}: candle sempat close di luar zona ({first_bo['close']:.2f}) "
                f"namun gagal bertahan dan kembali ditutup di {last_close:.2f}."
            ),
        }

    # Harga masih ditutup di luar zona pada bar closed terakhir -> evaluasi kualitas & momentum breakout
    best_bo = max(breakout_events, key=lambda e: (e["body_atr"], e["penetration_atr"]))
    curr_pen_pts = (last_close - z_high) if z_type == "RESISTANCE" else (z_low - last_close)
    curr_pen_atr = curr_pen_pts / last_atr
    max_pen_atr = max(curr_pen_atr, float(best_bo["penetration_atr"]))

    consec_outside = 0
    for idx_c in range(n_bars - 1, -1, -1):
        if z_type == "RESISTANCE" and float(closes[idx_c]) > z_high:
            consec_outside += 1
        elif z_type == "SUPPORT" and float(closes[idx_c]) < z_low:
            consec_outside += 1
        else:
            break

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

        return {
            "zone_id": zone.get("zone_id", "UNKNOWN"),
            "timeframe": tf,
            "zone_type": z_type,
            "status": "VALID_BREAKOUT",
            "label": "VALID BREAKOUT",
            "summary_status": "CONFIRMED",
            "direction": expected_dir,
            "breakout_bar_index": int(best_bo["bar_index"]),
            "breakout_bar_time": best_bo["bar_time"],
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
                f"Valid {expected_dir} breakout melewati {z_type} ({z_low:.2f}-{z_high:.2f}): "
                f"close={last_close:.2f} (penetrasi {curr_pen_atr:.2f} ATR, body {best_bo['body_atr']:.2f} ATR)."
            ),
        }

    # Jika sudah close di luar zona tetapi penetrasi tipis atau tanpa dukungan momentum yang memadai
    return {
        "zone_id": zone.get("zone_id", "UNKNOWN"),
        "timeframe": tf,
        "zone_type": z_type,
        "status": "BREAKOUT_PENDING_CONFIRMATION",
        "label": "BREAKOUT PENDING CONFIRMATION",
        "summary_status": "PENDING CONFIRMATION",
        "direction": expected_dir,
        "breakout_bar_index": int(latest_bo["bar_index"]),
        "breakout_bar_time": latest_bo["bar_time"],
        "close_outside_zone": True,
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
            f"Breakout belum terkonfirmasi penuh (penetrasi={curr_pen_atr:.2f} ATR, "
            f"body={best_bo['body_atr']:.2f} ATR, momentum_supported={momentum_supported}); "
            "menunggu candle konfirmasi lanjutan."
        ),
    }

