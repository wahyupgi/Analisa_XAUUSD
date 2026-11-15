# -*- coding: utf-8 -*-
"""
swing_detection.py — Pendeteksian Objektif Swing High & Swing Low (Pivot/Fractal)
untuk SNR Advanced Scanner.

Aturan Anti Look-Ahead Bias:
1. Hanya menggunakan candle yang sudah CLOSED.
2. Pivot pada indeks `i` hanya dianggap CONFIRMED setelah `right_bars` candle closed
   di sebelah kanannya selesai terbentuk (`i + right_bars <= last_closed_index`).
3. Kekuatan pivot (strength) dan klasifikasi MAJOR / MINOR dihitung murni menggunakan
   data hingga bar konfirmasi (`i + right_bars`) sehingga tidak pernah berubah secara
   retrospektif ketika candle masa depan terbentuk.
"""
import pandas as pd
from modules.snr.config import get_snr_config
from modules.snr.data_provider import prepare_snr_frame, safe_float


def detect_swings(
    df,
    timeframe="H1",
    config=None,
    has_unclosed_last_bar=False,
):
    """
    Mendeteksi Swing High dan Swing Low terkonfirmasi pada satu timeframe.

    Mengembalikan dict:
    - timeframe: str
    - swing_highs: list of dict (confirmed swing highs)
    - swing_lows: list of dict (confirmed swing lows)
    - all_swings: list of dict (gabungan urut kronologis)
    - unconfirmed_candidates: list of dict (kandidat di ujung kanan yang belum cukup right_bars)
    - summary: ringkasan statistik swing
    """
    cfg = config or get_snr_config()
    tf = str(timeframe).upper()
    swing_cfg = cfg.get("swing", {}).get(tf, cfg.get("swing", {}).get("H1", {}))

    left_bars = max(1, int(swing_cfg.get("left_bars", 3)))
    right_bars = max(1, int(swing_cfg.get("right_bars", 3)))
    major_atr_mult = safe_float(swing_cfg.get("major_atr_mult", 1.10), 1.10)
    min_dist_atr = safe_float(swing_cfg.get("min_pivot_distance_atr", 0.28), 0.28)
    dup_tol_atr = safe_float(swing_cfg.get("duplicate_tolerance_atr", 0.20), 0.20)

    clean_df, forming_candle = prepare_snr_frame(df, has_unclosed_last_bar=has_unclosed_last_bar)
    if clean_df.empty or len(clean_df) < (left_bars + right_bars + 1):
        return {
            "timeframe": tf,
            "left_bars": left_bars,
            "right_bars": right_bars,
            "swing_highs": [],
            "swing_lows": [],
            "all_swings": [],
            "unconfirmed_candidates": [],
            "forming_candle_excluded": forming_candle is not None,
            "summary": {
                "total_confirmed": 0,
                "swing_high_count": 0,
                "swing_low_count": 0,
                "major_count": 0,
                "minor_count": 0,
                "duplicates_filtered": 0,
            },
        }

    highs = clean_df["High"].to_numpy(dtype=float)
    lows = clean_df["Low"].to_numpy(dtype=float)
    opens = clean_df["Open"].to_numpy(dtype=float)
    closes = clean_df["Close"].to_numpy(dtype=float)
    atrs = clean_df["ATR14"].to_numpy(dtype=float)
    times = clean_df["Time"].tolist()
    n_bars = len(clean_df)
    last_confirmed_limit = n_bars - 1 - right_bars

    raw_highs = []
    raw_lows = []

    for i in range(left_bars, last_confirmed_limit + 1):
        atr_i = max(safe_float(atrs[i], 0.0), 0.10)
        h_i = float(highs[i])
        l_i = float(lows[i])

        left_h = highs[i - left_bars : i]
        right_h = highs[i + 1 : i + 1 + right_bars]

        # Swing High: tertinggi di kiri (>=) dan tegas lebih tinggi dari kanan (>)
        if h_i >= float(left_h.max()) and h_i > float(right_h.max()):
            past_start = max(0, i - left_bars * 2)
            left_low_min = float(lows[past_start : i + 1].min())
            right_low_min = float(lows[i + 1 : i + 1 + right_bars].min())
            left_exc = max(0.0, h_i - left_low_min) / atr_i
            right_exc = max(0.0, h_i - right_low_min) / atr_i
            excursion_atr = (left_exc * 0.55) + (right_exc * 0.45)

            upper_wick = max(0.0, h_i - max(float(opens[i]), float(closes[i])))
            wick_ratio = upper_wick / max(1e-6, h_i - l_i)
            strength_score = min(10.0, round((excursion_atr * 4.5) + (wick_ratio * 2.0), 2))
            swing_class = "MAJOR" if excursion_atr >= major_atr_mult else "MINOR"

            conf_idx = i + right_bars
            raw_highs.append({
                "pivot_id": f"{tf}_SH_{i}",
                "timeframe": tf,
                "type": "SWING_HIGH",
                "price": round(h_i, 2),
                "high": round(h_i, 2),
                "low": round(l_i, 2),
                "open": round(float(opens[i]), 2),
                "close": round(float(closes[i]), 2),
                "body_top": round(max(float(opens[i]), float(closes[i])), 2),
                "body_bottom": round(min(float(opens[i]), float(closes[i])), 2),
                "atr_at_pivot": round(atr_i, 4),
                "bar_index": int(i),
                "bar_time": str(times[i]),
                "confirmed": True,
                "confirmed_index": int(conf_idx),
                "confirmed_time": str(times[conf_idx]),
                "left_excursion_atr": round(left_exc, 3),
                "right_excursion_atr": round(right_exc, 3),
                "excursion_atr": round(excursion_atr, 3),
                "strength": strength_score,
                "swing_class": swing_class,
            })

        left_l = lows[i - left_bars : i]
        right_l = lows[i + 1 : i + 1 + right_bars]

        # Swing Low: terendah di kiri (<=) dan tegas lebih rendah dari kanan (<)
        if l_i <= float(left_l.min()) and l_i < float(right_l.min()):
            past_start = max(0, i - left_bars * 2)
            left_high_max = float(highs[past_start : i + 1].max())
            right_high_max = float(highs[i + 1 : i + 1 + right_bars].max())
            left_exc = max(0.0, left_high_max - l_i) / atr_i
            right_exc = max(0.0, right_high_max - l_i) / atr_i
            excursion_atr = (left_exc * 0.55) + (right_exc * 0.45)

            lower_wick = max(0.0, min(float(opens[i]), float(closes[i])) - l_i)
            wick_ratio = lower_wick / max(1e-6, h_i - l_i)
            strength_score = min(10.0, round((excursion_atr * 4.5) + (wick_ratio * 2.0), 2))
            swing_class = "MAJOR" if excursion_atr >= major_atr_mult else "MINOR"

            conf_idx = i + right_bars
            raw_lows.append({
                "pivot_id": f"{tf}_SL_{i}",
                "timeframe": tf,
                "type": "SWING_LOW",
                "price": round(l_i, 2),
                "high": round(h_i, 2),
                "low": round(l_i, 2),
                "open": round(float(opens[i]), 2),
                "close": round(float(closes[i]), 2),
                "body_top": round(max(float(opens[i]), float(closes[i])), 2),
                "body_bottom": round(min(float(opens[i]), float(closes[i])), 2),
                "atr_at_pivot": round(atr_i, 4),
                "bar_index": int(i),
                "bar_time": str(times[i]),
                "confirmed": True,
                "confirmed_index": int(conf_idx),
                "confirmed_time": str(times[conf_idx]),
                "left_excursion_atr": round(left_exc, 3),
                "right_excursion_atr": round(right_exc, 3),
                "excursion_atr": round(excursion_atr, 3),
                "strength": strength_score,
                "swing_class": swing_class,
            })

    # Hindari duplikasi level berdekatan pada tipe pivot yang sama tanpa look-ahead bias:
    # Jika pivot baru berada sangat dekat secara waktu (<= left_bars + right_bars + 2)
    # dan selisih harga <= duplicate_tolerance_atr * ATR terhadap pivot sebelumnya yang sudah confirmed,
    # tandai sebagai duplicate_filtered dan jangan masukkan ke daftar unik utama.
    duplicates_filtered = 0

    def _deduplicate_same_type(pivots):
        nonlocal duplicates_filtered
        filtered = []
        for p in pivots:
            if not filtered:
                p["distance_from_prev_same_type"] = 0.0
                p["distance_from_prev_same_type_atr"] = 0.0
                p["bars_from_prev_same_type"] = 0
                filtered.append(p)
                continue

            prev = filtered[-1]
            bar_gap = int(p["bar_index"] - prev["bar_index"])
            price_diff = abs(float(p["price"]) - float(prev["price"]))
            ref_atr = max(0.10, float(p["atr_at_pivot"]))
            price_diff_atr = price_diff / ref_atr

            if bar_gap <= (left_bars + right_bars + 2) and price_diff_atr <= dup_tol_atr:
                duplicates_filtered += 1
                continue

            p["distance_from_prev_same_type"] = round(price_diff, 2)
            p["distance_from_prev_same_type_atr"] = round(price_diff_atr, 3)
            p["bars_from_prev_same_type"] = bar_gap
            if price_diff_atr < min_dist_atr and p["swing_class"] == "MAJOR" and p["strength"] < prev["strength"]:
                p["swing_class"] = "MINOR"
            filtered.append(p)
        return filtered

    swing_highs = _deduplicate_same_type(raw_highs)
    swing_lows = _deduplicate_same_type(raw_lows)

    all_swings = sorted(
        swing_highs + swing_lows,
        key=lambda x: (x["bar_index"], 0 if x["type"] == "SWING_LOW" else 1),
    )

    for idx_s, sw in enumerate(all_swings):
        if idx_s == 0:
            sw["distance_from_prev_pivot"] = 0.0
            sw["bars_from_prev_pivot"] = 0
        else:
            prev_sw = all_swings[idx_s - 1]
            sw["distance_from_prev_pivot"] = round(abs(float(sw["price"]) - float(prev_sw["price"])), 2)
            sw["bars_from_prev_pivot"] = int(sw["bar_index"] - prev_sw["bar_index"])

    # Catat kandidat di ujung kanan yang belum terkonfirmasi (hanya untuk informasi diagnostik,
    # DILARANG dipakai sebagai level historis valid)
    unconfirmed_candidates = []
    for j in range(max(left_bars, last_confirmed_limit + 1), n_bars):
        h_j = float(highs[j])
        l_j = float(lows[j])
        left_h = highs[j - left_bars : j]
        left_l = lows[j - left_bars : j]
        right_h = highs[j + 1 : n_bars]
        right_l = lows[j + 1 : n_bars]

        if len(left_h) > 0 and h_j >= float(left_h.max()) and (len(right_h) == 0 or h_j > float(right_h.max())):
            unconfirmed_candidates.append({
                "timeframe": tf,
                "type": "UNCONFIRMED_HIGH",
                "price": round(h_j, 2),
                "bar_index": int(j),
                "bar_time": str(times[j]),
                "confirmed": False,
                "missing_right_bars": int(right_bars - len(right_h)),
            })
        if len(left_l) > 0 and l_j <= float(left_l.min()) and (len(right_l) == 0 or l_j < float(right_l.min())):
            unconfirmed_candidates.append({
                "timeframe": tf,
                "type": "UNCONFIRMED_LOW",
                "price": round(l_j, 2),
                "bar_index": int(j),
                "bar_time": str(times[j]),
                "confirmed": False,
                "missing_right_bars": int(right_bars - len(right_l)),
            })

    major_cnt = sum(1 for s in all_swings if s.get("swing_class") == "MAJOR")
    minor_cnt = sum(1 for s in all_swings if s.get("swing_class") == "MINOR")

    return {
        "timeframe": tf,
        "left_bars": left_bars,
        "right_bars": right_bars,
        "swing_highs": swing_highs,
        "swing_lows": swing_lows,
        "all_swings": all_swings,
        "unconfirmed_candidates": unconfirmed_candidates,
        "forming_candle_excluded": forming_candle is not None,
        "summary": {
            "total_confirmed": len(all_swings),
            "swing_high_count": len(swing_highs),
            "swing_low_count": len(swing_lows),
            "major_count": major_cnt,
            "minor_count": minor_cnt,
            "duplicates_filtered": duplicates_filtered,
        },
    }

