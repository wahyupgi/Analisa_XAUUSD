# -*- coding: utf-8 -*-
"""
resistance_detection.py — Pendeteksian Kandidat Zona Resistance Berbasis Swing High Terkonfirmasi,
Reaksi Harga Historis, Kekuatan Penolakan, Struktur Konsolidasi, Freshness, dan ATR Dinamis.
"""
from modules.snr.config import get_snr_config
from modules.snr.data_provider import prepare_snr_frame, safe_float
from modules.snr.swing_detection import detect_swings


def _compute_dynamic_resistance_bounds(pivot, tf_atr, zone_cfg):
    """
    Menghitung batas bawah dan atas zona resistance secara dinamis berdasarkan
    ATR timeframe dan struktur body/wick pivot (bukan lebar statis).
    """
    piv_atr = max(0.10, safe_float(pivot.get("atr_at_pivot"), tf_atr))
    ref_atr = (piv_atr * 0.65) + (max(0.10, tf_atr) * 0.35)

    half_mult = safe_float(zone_cfg.get("zone_half_width_atr", 0.18), 0.18)
    min_mult = safe_float(zone_cfg.get("min_zone_width_atr", 0.15), 0.15)
    max_mult = safe_float(zone_cfg.get("max_zone_width_atr", 0.78), 0.78)
    max_pts = safe_float(zone_cfg.get("max_zone_width_points", 10.0), 10.0)

    min_width = max(0.15, ref_atr * min_mult)
    max_width = min(max_pts, max(min_width + 0.10, ref_atr * max_mult))
    target_width = ref_atr * (half_mult * 2.0)

    high_price = safe_float(pivot.get("high"), safe_float(pivot.get("price"), 0.0))
    body_top = safe_float(pivot.get("body_top"), high_price - target_width)
    structural_span = max(0.0, high_price - body_top)

    raw_width = (structural_span * 0.55) + (target_width * 0.45)
    width = max(min_width, min(max_width, raw_width))

    zone_high = round(high_price, 2)
    zone_low = round(high_price - width, 2)
    if zone_high <= zone_low:
        zone_low = round(zone_high - max(0.15, min_width), 2)
    return zone_low, zone_high, round(zone_high - zone_low, 2), round(ref_atr, 4)


def detect_resistance_candidates(
    df,
    swing_result=None,
    timeframe="H1",
    config=None,
    has_unclosed_last_bar=False,
):
    """
    Mengidentifikasi kandidat zona Resistance dari confirmed swing high dan reaksi historis.
    Seluruh perhitungan hanya menggunakan CLOSED candles setelah pivot terkonfirmasi.
    """
    cfg = config or get_snr_config()
    tf = str(timeframe).upper()
    zone_cfg = cfg.get("zone", {}).get(tf, cfg.get("zone", {}).get("H1", {}))

    clean_df, _ = prepare_snr_frame(df, has_unclosed_last_bar=has_unclosed_last_bar)
    if clean_df.empty:
        return []

    if swing_result is None:
        swing_result = detect_swings(clean_df, timeframe=tf, config=cfg, has_unclosed_last_bar=False)

    swing_highs = swing_result.get("swing_highs", [])
    if not swing_highs:
        return []

    highs = clean_df["High"].to_numpy(dtype=float)
    lows = clean_df["Low"].to_numpy(dtype=float)
    opens = clean_df["Open"].to_numpy(dtype=float)
    closes = clean_df["Close"].to_numpy(dtype=float)
    atrs = clean_df["ATR14"].to_numpy(dtype=float)
    n_bars = len(clean_df)
    curr_tf_atr = max(0.10, safe_float(atrs[-1], 1.0))

    candidates = []
    for idx_cand, piv in enumerate(swing_highs):
        if not piv.get("confirmed", False):
            continue

        z_low, z_high, z_width, ref_atr = _compute_dynamic_resistance_bounds(piv, curr_tf_atr, zone_cfg)
        z_mid = round((z_low + z_high) / 2.0, 2)
        piv_idx = int(piv["bar_index"])
        conf_idx = int(piv["confirmed_index"])

        # Evaluasi struktur konsolidasi di sekitar pembentukan pivot (hanya hingga conf_idx)
        base_start = max(0, piv_idx - 3)
        base_end = min(n_bars, conf_idx + 1)
        base_bars_in_zone = 0
        for b in range(base_start, base_end):
            if highs[b] >= z_low - (0.15 * ref_atr) and closes[b] <= z_high + (0.10 * ref_atr):
                base_bars_in_zone += 1
        if base_bars_in_zone >= 4:
            consolidation_structure = "CONSOLIDATION_BASE"
        elif safe_float(piv.get("right_excursion_atr"), 0.0) >= 1.25:
            consolidation_structure = "IMPULSIVE_DEPARTURE"
        else:
            consolidation_structure = "STANDARD_PIVOT"

        # Evaluasi reaksi historis setelah pivot confirmed (conf_idx + 1 .. n_bars - 1)
        reaction_count = 0
        repeated_rejections = 0
        max_rejection_atr = safe_float(piv.get("right_excursion_atr"), 0.0)
        broken_bars = 0
        first_break_index = None
        last_touch_index = None

        k = conf_idx + 1
        while k < n_bars:
            bar_low = float(lows[k])
            bar_high = float(highs[k])
            bar_close = float(closes[k])
            bar_open = float(opens[k])
            bar_atr = max(0.10, safe_float(atrs[k], ref_atr))

            # Cek apakah candle menembus tegas ke atas resistance
            if bar_close > (z_high + 0.18 * bar_atr):
                broken_bars += 1
                if first_break_index is None:
                    first_break_index = int(k)

            # Cek apakah candle menguji area resistance dan tertolak turun
            if bar_high >= (z_low - 0.08 * bar_atr) and bar_low <= z_high:
                if last_touch_index is None or (k - last_touch_index) >= 2:
                    look_end = min(n_bars, k + 5)
                    future_min_low = float(lows[k:look_end].min()) if look_end > k else bar_low
                    drop_atr = max(0.0, min(bar_high, z_high) - future_min_low) / bar_atr
                    if bar_close <= z_high and drop_atr >= 0.25:
                        reaction_count += 1
                        max_rejection_atr = max(max_rejection_atr, drop_atr)
                        upper_wick = max(0.0, bar_high - max(bar_open, bar_close))
                        if upper_wick >= 0.30 * max(1e-6, bar_high - bar_low) and bar_close <= z_mid:
                            repeated_rejections += 1
                        last_touch_index = int(k)
            k += 1

        age_bars = max(0, (n_bars - 1) - piv_idx)
        freshness_ratio = max(0.15, min(1.0, 1.0 - (age_bars / max(60.0, float(n_bars) * 1.1))))
        if reaction_count == 0 and broken_bars == 0:
            freshness_state = "FRESH_UNTESTED"
        elif broken_bars == 0 and reaction_count <= 2:
            freshness_state = "ACTIVE_TESTED"
        elif broken_bars == 0:
            freshness_state = "FREQUENTLY_TESTED"
        else:
            freshness_state = "BROKEN_HISTORY"

        breakout_history = "CLEAN_HOLD" if broken_bars == 0 else ("BROKEN_ABOVE" if broken_bars >= 2 else "TESTED_DEEP")

        candidates.append({
            "candidate_id": f"{tf}_RES_CAND_{idx_cand + 1}_{piv_idx}",
            "timeframe": tf,
            "zone_type": "RESISTANCE",
            "low": z_low,
            "high": z_high,
            "mid": z_mid,
            "width": z_width,
            "width_atr": round(z_width / max(0.10, ref_atr), 3),
            "ref_atr": ref_atr,
            "origin_pivot": piv,
            "source_swings": [piv],
            "formation_index": piv_idx,
            "formation_time": piv.get("bar_time"),
            "confirmed_index": conf_idx,
            "confirmed_time": piv.get("confirmed_time"),
            "swing_class": piv.get("swing_class", "MINOR"),
            "swing_strength": safe_float(piv.get("strength"), 0.0),
            "bounce_strength_atr": round(max_rejection_atr, 3),
            "historical_reaction_count": int(reaction_count),
            "repeated_rejections": int(repeated_rejections),
            "consolidation_structure": consolidation_structure,
            "age_bars": int(age_bars),
            "freshness_ratio": round(freshness_ratio, 3),
            "freshness_state": freshness_state,
            "breakout_history": breakout_history,
            "first_break_index": first_break_index,
        })

    return candidates

