# -*- coding: utf-8 -*-
"""
rejection_detection.py — Sistem Identifikasi Candle Rejection Berbasis Konteks Lokasi Zona,
 Karakteristik Wick/Body, ATR, Posisi Close, dan Konfirmasi Candle Berikutnya.

Mengevaluasi:
- Upper wick & Lower wick
- Body candle & Rasio wick terhadap body / range
- Posisi closing candle
- Lokasi candle terhadap zona (Support / Resistance)
- ATR candle
- Konfirmasi candle berikutnya (next-candle confirmation)

Klasifikasi:
- BULLISH_REJECTION
- BEARISH_REJECTION
- WEAK_REJECTION
- FALSE_REJECTION
- NONE (WAIT)
"""
from modules.snr.config import get_snr_config
from modules.snr.data_provider import prepare_snr_frame, safe_float


def detect_zone_rejection(
    zone,
    df,
    config=None,
    has_unclosed_last_bar=False,
):
    """
    Mendeteksi apakah terdapat rejection yang valid, lemah, atau palsu pada suatu zona
    menggunakan CLOSED candles.
    """
    cfg = config or get_snr_config()
    rej_cfg = cfg.get("rejection", {})

    min_wick_body = safe_float(rej_cfg.get("min_rejection_wick_to_body", 1.10), 1.10)
    min_wick_range = safe_float(rej_cfg.get("min_rejection_wick_to_range", 0.36), 0.36)
    weak_wick_range = safe_float(rej_cfg.get("weak_rejection_wick_to_range", 0.22), 0.22)
    prox_atr = safe_float(rej_cfg.get("rejection_zone_proximity_atr", 0.25), 0.25)
    min_range_atr = safe_float(rej_cfg.get("min_candle_range_atr", 0.25), 0.25)
    lookback = max(2, int(rej_cfg.get("lookback_bars", 5)))

    clean_df, _ = prepare_snr_frame(df, has_unclosed_last_bar=has_unclosed_last_bar)
    z_low = safe_float(zone.get("low"), 0.0)
    z_high = safe_float(zone.get("high"), z_low)
    z_mid = round((z_low + z_high) / 2.0, 2)
    z_type = str(zone.get("zone_type", "SUPPORT")).upper()
    tf = str(zone.get("timeframe", "M5")).upper()
    ref_atr = max(0.10, safe_float(zone.get("ref_atr"), 1.0))

    if clean_df.empty or z_high <= 0:
        return {
            "zone_id": zone.get("zone_id", "UNKNOWN"),
            "timeframe": tf,
            "zone_type": z_type,
            "status": "NONE",
            "summary_status": "WAIT",
            "direction": "NONE",
            "confirmed": False,
            "bar_index": None,
            "bar_time": None,
            "in_zone_context": False,
            "upper_wick": 0.0,
            "lower_wick": 0.0,
            "body": 0.0,
            "wick_to_body": 0.0,
            "wick_to_range": 0.0,
            "close_position_ratio": 0.0,
            "range_atr": 0.0,
            "next_candle_confirmed": False,
            "quality_score_15": 0.0,
            "reason": "Data candle tidak tersedia untuk evaluasi rejection.",
        }

    opens = clean_df["Open"].to_numpy(dtype=float)
    highs = clean_df["High"].to_numpy(dtype=float)
    lows = clean_df["Low"].to_numpy(dtype=float)
    closes = clean_df["Close"].to_numpy(dtype=float)
    atrs = clean_df["ATR14"].to_numpy(dtype=float)
    times = clean_df["Time"].tolist()
    n_bars = len(clean_df)

    start_idx = max(0, n_bars - lookback)
    candidates = []

    for k in range(start_idx, n_bars):
        b_open = float(opens[k])
        b_high = float(highs[k])
        b_low = float(lows[k])
        b_close = float(closes[k])
        b_atr = max(0.10, safe_float(atrs[k], ref_atr))

        c_range = max(1e-6, b_high - b_low)
        c_body = abs(b_close - b_open)
        eff_body = max(c_body, 0.04 * b_atr)
        upper_wick = max(0.0, b_high - max(b_open, b_close))
        lower_wick = max(0.0, min(b_open, b_close) - b_low)
        close_pos = (b_close - b_low) / c_range
        range_atr = c_range / b_atr

        if z_type == "SUPPORT":
            in_zone = (b_low <= (z_high + prox_atr * b_atr)) and (b_high >= (z_low - 0.35 * b_atr))
            rel_wick = lower_wick
            wick_to_body = lower_wick / eff_body
            wick_to_range = lower_wick / c_range
            favorable_close = (b_close >= z_low) and (close_pos >= 0.52 or b_close >= b_open)

            # Evaluasi candle berikutnya (jika sudah ada candle closed setelah k)
            has_next = (k + 1) < n_bars
            next_confirmed = False
            next_failed = False
            if has_next:
                for m in range(k + 1, n_bars):
                    if float(closes[m]) < (z_low - 0.10 * b_atr) or float(closes[m]) < b_low:
                        next_failed = True
                        break
                if not next_failed:
                    n_open = float(opens[k + 1])
                    n_close = float(closes[k + 1])
                    last_c = float(closes[-1])
                    if (n_close >= b_close or n_close > n_open) and last_c >= z_low and last_c >= b_close:
                        next_confirmed = True

            if not in_zone:
                # Pola candlestick di luar konteks lokasi zona -> abaikan / tandai out-of-context
                if wick_to_range >= min_wick_range and wick_to_body >= min_wick_body:
                    candidates.append({
                        "priority": 1,
                        "status": "FALSE_REJECTION",
                        "summary_status": "FALSE REJECTION",
                        "direction": "NONE",
                        "confirmed": False,
                        "bar_index": int(k),
                        "bar_time": str(times[k]),
                        "in_zone_context": False,
                        "upper_wick": round(upper_wick, 2),
                        "lower_wick": round(lower_wick, 2),
                        "body": round(c_body, 2),
                        "wick_to_body": round(wick_to_body, 2),
                        "wick_to_range": round(wick_to_range, 3),
                        "close_position_ratio": round(close_pos, 3),
                        "range_atr": round(range_atr, 3),
                        "next_candle_confirmed": False,
                        "quality_score_15": 0.0,
                        "reason": "Pola wick muncul di luar lokasi zona Support (tanpa konteks struktural).",
                    })
                continue

            # Berada di dalam/dekat zona Support
            if next_failed and wick_to_range >= weak_wick_range:
                candidates.append({
                    "priority": 3,
                    "status": "FALSE_REJECTION",
                    "summary_status": "FALSE REJECTION",
                    "direction": "BULLISH",
                    "confirmed": False,
                    "bar_index": int(k),
                    "bar_time": str(times[k]),
                    "in_zone_context": True,
                    "upper_wick": round(upper_wick, 2),
                    "lower_wick": round(lower_wick, 2),
                    "body": round(c_body, 2),
                    "wick_to_body": round(wick_to_body, 2),
                    "wick_to_range": round(wick_to_range, 3),
                    "close_position_ratio": round(close_pos, 3),
                    "range_atr": round(range_atr, 3),
                    "next_candle_confirmed": False,
                    "quality_score_15": 2.0,
                    "reason": (
                        f"Rejection palsu (FALSE_REJECTION) di Support: lower wick pada bar #{k} "
                        "gagal ditindaklanjuti dan ditembus ke bawah oleh candle berikutnya."
                    ),
                })
            elif (
                favorable_close
                and wick_to_range >= min_wick_range
                and wick_to_body >= min_wick_body
                and range_atr >= min_range_atr
                and next_confirmed
            ):
                q_score = min(15.0, round(11.5 + min(2.0, (wick_to_range - min_wick_range) * 5.0) + min(1.5, range_atr * 0.8), 2))
                candidates.append({
                    "priority": 5,
                    "status": "BULLISH_REJECTION",
                    "summary_status": "CONFIRMED (BULLISH)",
                    "direction": "BULLISH",
                    "confirmed": True,
                    "bar_index": int(k),
                    "bar_time": str(times[k]),
                    "in_zone_context": True,
                    "upper_wick": round(upper_wick, 2),
                    "lower_wick": round(lower_wick, 2),
                    "body": round(c_body, 2),
                    "wick_to_body": round(wick_to_body, 2),
                    "wick_to_range": round(wick_to_range, 3),
                    "close_position_ratio": round(close_pos, 3),
                    "range_atr": round(range_atr, 3),
                    "next_candle_confirmed": True,
                    "quality_score_15": q_score,
                    "reason": (
                        f"Bullish Support Rejection terkonfirmasi pada bar #{k}: lower wick {wick_to_range:.0%} "
                        f"(wick/body {wick_to_body:.2f}x), close di atas area relevan, dan dikonfirmasi candle lanjutan."
                    ),
                })
            elif favorable_close and (wick_to_range >= weak_wick_range or (b_close > b_open and b_close >= z_mid)):
                why_weak = (
                    "menunggu konfirmasi candle berikutnya"
                    if (wick_to_range >= min_wick_range and not next_confirmed)
                    else f"rasio wick moderat ({wick_to_range:.0%})"
                )
                candidates.append({
                    "priority": 4,
                    "status": "WEAK_REJECTION",
                    "summary_status": "WEAK / PENDING",
                    "direction": "BULLISH",
                    "confirmed": False,
                    "bar_index": int(k),
                    "bar_time": str(times[k]),
                    "in_zone_context": True,
                    "upper_wick": round(upper_wick, 2),
                    "lower_wick": round(lower_wick, 2),
                    "body": round(c_body, 2),
                    "wick_to_body": round(wick_to_body, 2),
                    "wick_to_range": round(wick_to_range, 3),
                    "close_position_ratio": round(close_pos, 3),
                    "range_atr": round(range_atr, 3),
                    "next_candle_confirmed": bool(next_confirmed),
                    "quality_score_15": 7.5,
                    "reason": f"Weak Support Rejection pada bar #{k} ({why_weak}).",
                })

        else:  # RESISTANCE
            in_zone = (b_high >= (z_low - prox_atr * b_atr)) and (b_low <= (z_high + 0.35 * b_atr))
            wick_to_body = upper_wick / eff_body
            wick_to_range = upper_wick / c_range
            favorable_close = (b_close <= z_high) and (close_pos <= 0.48 or b_close <= b_open)

            has_next = (k + 1) < n_bars
            next_confirmed = False
            next_failed = False
            if has_next:
                for m in range(k + 1, n_bars):
                    if float(closes[m]) > (z_high + 0.10 * b_atr) or float(closes[m]) > b_high:
                        next_failed = True
                        break
                if not next_failed:
                    n_open = float(opens[k + 1])
                    n_close = float(closes[k + 1])
                    last_c = float(closes[-1])
                    if (n_close <= b_close or n_close < n_open) and last_c <= z_high and last_c <= b_close:
                        next_confirmed = True

            if not in_zone:
                if wick_to_range >= min_wick_range and wick_to_body >= min_wick_body:
                    candidates.append({
                        "priority": 1,
                        "status": "FALSE_REJECTION",
                        "summary_status": "FALSE REJECTION",
                        "direction": "NONE",
                        "confirmed": False,
                        "bar_index": int(k),
                        "bar_time": str(times[k]),
                        "in_zone_context": False,
                        "upper_wick": round(upper_wick, 2),
                        "lower_wick": round(lower_wick, 2),
                        "body": round(c_body, 2),
                        "wick_to_body": round(wick_to_body, 2),
                        "wick_to_range": round(wick_to_range, 3),
                        "close_position_ratio": round(close_pos, 3),
                        "range_atr": round(range_atr, 3),
                        "next_candle_confirmed": False,
                        "quality_score_15": 0.0,
                        "reason": "Pola wick muncul di luar lokasi zona Resistance (tanpa konteks struktural).",
                    })
                continue

            if next_failed and wick_to_range >= weak_wick_range:
                candidates.append({
                    "priority": 3,
                    "status": "FALSE_REJECTION",
                    "summary_status": "FALSE REJECTION",
                    "direction": "BEARISH",
                    "confirmed": False,
                    "bar_index": int(k),
                    "bar_time": str(times[k]),
                    "in_zone_context": True,
                    "upper_wick": round(upper_wick, 2),
                    "lower_wick": round(lower_wick, 2),
                    "body": round(c_body, 2),
                    "wick_to_body": round(wick_to_body, 2),
                    "wick_to_range": round(wick_to_range, 3),
                    "close_position_ratio": round(close_pos, 3),
                    "range_atr": round(range_atr, 3),
                    "next_candle_confirmed": False,
                    "quality_score_15": 2.0,
                    "reason": (
                        f"Rejection palsu (FALSE_REJECTION) di Resistance: upper wick pada bar #{k} "
                        "gagal bertahan dan ditembus ke atas oleh candle berikutnya."
                    ),
                })
            elif (
                favorable_close
                and wick_to_range >= min_wick_range
                and wick_to_body >= min_wick_body
                and range_atr >= min_range_atr
                and next_confirmed
            ):
                q_score = min(15.0, round(11.5 + min(2.0, (wick_to_range - min_wick_range) * 5.0) + min(1.5, range_atr * 0.8), 2))
                candidates.append({
                    "priority": 5,
                    "status": "BEARISH_REJECTION",
                    "summary_status": "CONFIRMED (BEARISH)",
                    "direction": "BEARISH",
                    "confirmed": True,
                    "bar_index": int(k),
                    "bar_time": str(times[k]),
                    "in_zone_context": True,
                    "upper_wick": round(upper_wick, 2),
                    "lower_wick": round(lower_wick, 2),
                    "body": round(c_body, 2),
                    "wick_to_body": round(wick_to_body, 2),
                    "wick_to_range": round(wick_to_range, 3),
                    "close_position_ratio": round(close_pos, 3),
                    "range_atr": round(range_atr, 3),
                    "next_candle_confirmed": True,
                    "quality_score_15": q_score,
                    "reason": (
                        f"Bearish Resistance Rejection terkonfirmasi pada bar #{k}: upper wick {wick_to_range:.0%} "
                        f"(wick/body {wick_to_body:.2f}x), close di bawah area relevan, dan dikonfirmasi candle lanjutan."
                    ),
                })
            elif favorable_close and (wick_to_range >= weak_wick_range or (b_close < b_open and b_close <= z_mid)):
                why_weak = (
                    "menunggu konfirmasi candle berikutnya"
                    if (wick_to_range >= min_wick_range and not next_confirmed)
                    else f"rasio wick moderat ({wick_to_range:.0%})"
                )
                candidates.append({
                    "priority": 4,
                    "status": "WEAK_REJECTION",
                    "summary_status": "WEAK / PENDING",
                    "direction": "BEARISH",
                    "confirmed": False,
                    "bar_index": int(k),
                    "bar_time": str(times[k]),
                    "in_zone_context": True,
                    "upper_wick": round(upper_wick, 2),
                    "lower_wick": round(lower_wick, 2),
                    "body": round(c_body, 2),
                    "wick_to_body": round(wick_to_body, 2),
                    "wick_to_range": round(wick_to_range, 3),
                    "close_position_ratio": round(close_pos, 3),
                    "range_atr": round(range_atr, 3),
                    "next_candle_confirmed": bool(next_confirmed),
                    "quality_score_15": 7.5,
                    "reason": f"Weak Resistance Rejection pada bar #{k} ({why_weak}).",
                })

    if candidates:
        # Jika ada FALSE_REJECTION yang lebih baru dari rejection sebelumnya, dahulukan status terbaru
        latest_cand = candidates[-1]
        if latest_cand["status"] == "FALSE_REJECTION" and latest_cand["in_zone_context"]:
            best = dict(latest_cand)
        else:
            best = dict(max(candidates, key=lambda c: (c["priority"], c["bar_index"])))
        best.pop("priority", None)
        best["zone_id"] = zone.get("zone_id", "UNKNOWN")
        best["timeframe"] = tf
        best["zone_type"] = z_type
        return best

    return {
        "zone_id": zone.get("zone_id", "UNKNOWN"),
        "timeframe": tf,
        "zone_type": z_type,
        "status": "NONE",
        "summary_status": "WAIT",
        "direction": "NONE",
        "confirmed": False,
        "bar_index": None,
        "bar_time": None,
        "in_zone_context": False,
        "upper_wick": 0.0,
        "lower_wick": 0.0,
        "body": 0.0,
        "wick_to_body": 0.0,
        "wick_to_range": 0.0,
        "close_position_ratio": 0.0,
        "range_atr": 0.0,
        "next_candle_confirmed": False,
        "quality_score_15": 0.0,
        "reason": f"Belum ada pola rejection pada area {z_type} ({z_low:.2f}-{z_high:.2f}).",
    }

