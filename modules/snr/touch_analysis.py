# -*- coding: utf-8 -*-
"""
touch_analysis.py — Sistem Analisis Pengujian (Touch Analysis) Zona Support & Resistance.

Klasifikasi Wajib:
- UNTESTED
- TESTED_1X
- TESTED_2X
- TESTED_3X_PLUS
- BROKEN
- INVALIDATED

Aturan Utama:
1. Tidak menghitung touch berulang dari candle yang sama atau kunjungan (visit) yang sama.
2. Membedakan sentuhan wick biasa (WICK_ONLY_CONTACT) dengan penolakan valid (VALID_REJECTION).
3. Semakin sering diuji tidak otomatis semakin kuat: penetrasi yang makin dalam atau pantulan
   yang melemah menurunkan kualitas zona.
"""
from modules.snr.config import get_snr_config
from modules.snr.data_provider import prepare_snr_frame, safe_float


def analyze_zone_touches(
    zone,
    df,
    config=None,
    has_unclosed_last_bar=False,
):
    """
    Menganalisis riwayat pengujian (touches) sebuah zona pada DataFrame candle closed.
    Mengembalikan salinan zona yang diperkaya dengan field `touch_analysis` dan `touch_status`.
    """
    cfg = config or get_snr_config()
    touch_cfg = cfg.get("touch", {})

    min_sep_bars = max(1, int(touch_cfg.get("min_touch_separation_bars", 2)))
    valid_bounce_atr = safe_float(touch_cfg.get("valid_rejection_bounce_atr", 0.32), 0.32)
    max_pen_atr = safe_float(touch_cfg.get("max_valid_penetration_atr", 0.45), 0.45)
    broken_close_atr = safe_float(touch_cfg.get("broken_close_beyond_atr", 0.15), 0.15)
    invalid_close_atr = safe_float(touch_cfg.get("invalidation_close_beyond_atr", 0.42), 0.42)
    consec_invalid = max(2, int(touch_cfg.get("consecutive_closes_for_invalidation", 2)))

    clean_df, _ = prepare_snr_frame(df, has_unclosed_last_bar=has_unclosed_last_bar)
    z_low = safe_float(zone.get("low"), 0.0)
    z_high = safe_float(zone.get("high"), z_low)
    z_width = max(0.05, z_high - z_low)
    z_type = str(zone.get("zone_type", "SUPPORT")).upper()
    ref_atr = max(0.10, safe_float(zone.get("ref_atr"), 1.0))

    if clean_df.empty or z_high <= 0:
        empty_res = {
            "classification": "UNTESTED",
            "valid_touch_count": 0,
            "wick_only_count": 0,
            "total_interactions": 0,
            "touches": [],
            "avg_bounce_atr": 0.0,
            "max_bounce_atr": 0.0,
            "avg_penetration_ratio": 0.0,
            "max_penetration_atr": 0.0,
            "avg_bars_between_touches": 0.0,
            "bounce_decay": False,
            "absorption_warning": False,
            "is_broken": False,
            "is_invalidated": False,
            "first_break_index": None,
            "touch_quality_score_15": 11.0,
            "reason": "No post-confirmation bars available; treated as UNTESTED.",
        }
        updated = dict(zone)
        updated["touch_status"] = "UNTESTED"
        updated["touch_count"] = 0
        updated["touch_analysis"] = empty_res
        return updated

    highs = clean_df["High"].to_numpy(dtype=float)
    lows = clean_df["Low"].to_numpy(dtype=float)
    opens = clean_df["Open"].to_numpy(dtype=float)
    closes = clean_df["Close"].to_numpy(dtype=float)
    atrs = clean_df["ATR14"].to_numpy(dtype=float)
    times = clean_df["Time"].tolist()
    n_bars = len(clean_df)

    conf_idx = int(zone.get("confirmed_index", zone.get("formation_index", 0)))
    start_idx = max(0, conf_idx + 1)

    valid_touches = []
    wick_only_contacts = []
    seen_bar_indices = set()

    is_broken = False
    is_invalidated = False
    first_break_index = None
    consecutive_beyond = 0
    price_left_zone_since_last_touch = True
    last_interaction_bar = None

    for k in range(start_idx, n_bars):
        if k in seen_bar_indices:
            continue
        seen_bar_indices.add(k)

        b_high = float(highs[k])
        b_low = float(lows[k])
        b_open = float(opens[k])
        b_close = float(closes[k])
        b_atr = max(0.10, safe_float(atrs[k], ref_atr))
        b_range = max(1e-6, b_high - b_low)

        if z_type == "SUPPORT":
            # Cek apakah harga sudah keluar cukup jauh di atas zona untuk mereset visit
            if b_low > z_high + (0.05 * b_atr):
                price_left_zone_since_last_touch = True

            # Cek penembusan ke bawah support
            if b_close < (z_low - broken_close_atr * b_atr):
                is_broken = True
                consecutive_beyond += 1
                if first_break_index is None:
                    first_break_index = int(k)
                if b_close < (z_low - invalid_close_atr * b_atr) or consecutive_beyond >= consec_invalid:
                    is_invalidated = True
                continue
            else:
                consecutive_beyond = 0

            # Cek apakah candle menyentuh/memasuki zona support
            touches_zone = (b_low <= z_high + 0.04 * b_atr) and (b_high >= z_low - 0.04 * b_atr)
            if not touches_zone:
                continue

            # Cegah penghitungan berulang dari visit yang sama atau jarak bar terlalu rapat
            if last_interaction_bar is not None:
                if (k - last_interaction_bar) < min_sep_bars or not price_left_zone_since_last_touch:
                    continue

            penetration_pts = max(0.0, z_high - b_low)
            overshoot_pts = max(0.0, z_low - b_low)
            penetration_ratio = min(2.0, penetration_pts / z_width)
            overshoot_atr = overshoot_pts / b_atr

            look_end = min(n_bars, k + 5)
            future_high = float(highs[k:look_end].max()) if look_end > k else b_high
            bounce_pts = max(0.0, future_high - max(b_low, z_low))
            bounce_atr = bounce_pts / b_atr

            lower_wick = max(0.0, min(b_open, b_close) - b_low)
            wick_ratio = lower_wick / b_range
            favorable_close = b_close >= z_low and (b_close >= b_open or b_close >= (z_low + 0.35 * z_width))

            if favorable_close and overshoot_atr <= max_pen_atr and (bounce_atr >= valid_bounce_atr or (wick_ratio >= 0.38 and bounce_atr >= 0.18)):
                valid_touches.append({
                    "bar_index": int(k),
                    "bar_time": str(times[k]),
                    "type": "VALID_SUPPORT_REJECTION",
                    "low": round(b_low, 2),
                    "high": round(b_high, 2),
                    "close": round(b_close, 2),
                    "bounce_atr": round(bounce_atr, 3),
                    "penetration_ratio": round(penetration_ratio, 3),
                    "overshoot_atr": round(overshoot_atr, 3),
                    "wick_ratio": round(wick_ratio, 3),
                })
                last_interaction_bar = int(k)
                price_left_zone_since_last_touch = False
            else:
                wick_only_contacts.append({
                    "bar_index": int(k),
                    "bar_time": str(times[k]),
                    "type": "WICK_OR_WEAK_CONTACT",
                    "bounce_atr": round(bounce_atr, 3),
                    "penetration_ratio": round(penetration_ratio, 3),
                })
                last_interaction_bar = int(k)

        else:  # RESISTANCE
            if b_high < z_low - (0.05 * b_atr):
                price_left_zone_since_last_touch = True

            # Cek penembusan ke atas resistance
            if b_close > (z_high + broken_close_atr * b_atr):
                is_broken = True
                consecutive_beyond += 1
                if first_break_index is None:
                    first_break_index = int(k)
                if b_close > (z_high + invalid_close_atr * b_atr) or consecutive_beyond >= consec_invalid:
                    is_invalidated = True
                continue
            else:
                consecutive_beyond = 0

            touches_zone = (b_high >= z_low - 0.04 * b_atr) and (b_low <= z_high + 0.04 * b_atr)
            if not touches_zone:
                continue

            if last_interaction_bar is not None:
                if (k - last_interaction_bar) < min_sep_bars or not price_left_zone_since_last_touch:
                    continue

            penetration_pts = max(0.0, b_high - z_low)
            overshoot_pts = max(0.0, b_high - z_high)
            penetration_ratio = min(2.0, penetration_pts / z_width)
            overshoot_atr = overshoot_pts / b_atr

            look_end = min(n_bars, k + 5)
            future_low = float(lows[k:look_end].min()) if look_end > k else b_low
            drop_pts = max(0.0, min(b_high, z_high) - future_low)
            bounce_atr = drop_pts / b_atr

            upper_wick = max(0.0, b_high - max(b_open, b_close))
            wick_ratio = upper_wick / b_range
            favorable_close = b_close <= z_high and (b_close <= b_open or b_close <= (z_high - 0.35 * z_width))

            if favorable_close and overshoot_atr <= max_pen_atr and (bounce_atr >= valid_bounce_atr or (wick_ratio >= 0.38 and bounce_atr >= 0.18)):
                valid_touches.append({
                    "bar_index": int(k),
                    "bar_time": str(times[k]),
                    "type": "VALID_RESISTANCE_REJECTION",
                    "low": round(b_low, 2),
                    "high": round(b_high, 2),
                    "close": round(b_close, 2),
                    "bounce_atr": round(bounce_atr, 3),
                    "penetration_ratio": round(penetration_ratio, 3),
                    "overshoot_atr": round(overshoot_atr, 3),
                    "wick_ratio": round(wick_ratio, 3),
                })
                last_interaction_bar = int(k)
                price_left_zone_since_last_touch = False
            else:
                wick_only_contacts.append({
                    "bar_index": int(k),
                    "bar_time": str(times[k]),
                    "type": "WICK_OR_WEAK_CONTACT",
                    "bounce_atr": round(bounce_atr, 3),
                    "penetration_ratio": round(penetration_ratio, 3),
                })
                last_interaction_bar = int(k)

    # Jika pada candle terakhir harga berhasil kembali ke sisi aktif setelah sempat tembus 1 candle,
    # evaluasi apakah itu false breakout (bukan invalidated)
    last_close = float(closes[-1])
    last_atr = max(0.10, safe_float(atrs[-1], ref_atr))
    if is_broken and not is_invalidated:
        if z_type == "SUPPORT" and last_close >= z_low:
            # Sempat tembus 1 candle namun sudah reclaim kembali ke atas support
            is_broken = False
        elif z_type == "RESISTANCE" and last_close <= z_high:
            is_broken = False

    if is_invalidated:
        # Pastikan jika pada akhirnya harga masih jauh di luar zona, status INVALIDATED tetap terkunci
        if z_type == "SUPPORT" and last_close >= z_high:
            is_invalidated = False
            is_broken = False
        elif z_type == "RESISTANCE" and last_close <= z_low:
            is_invalidated = False
            is_broken = False

    valid_cnt = len(valid_touches)
    wick_cnt = len(wick_only_contacts)

    if is_invalidated:
        classification = "INVALIDATED"
    elif is_broken:
        classification = "BROKEN"
    elif valid_cnt == 0:
        classification = "UNTESTED"
    elif valid_cnt == 1:
        classification = "TESTED_1X"
    elif valid_cnt == 2:
        classification = "TESTED_2X"
    else:
        classification = "TESTED_3X_PLUS"

    if valid_cnt > 0:
        bounces = [float(t["bounce_atr"]) for t in valid_touches]
        pens = [float(t["penetration_ratio"]) for t in valid_touches]
        overs = [float(t["overshoot_atr"]) for t in valid_touches]
        avg_bounce = sum(bounces) / len(bounces)
        max_bounce = max(bounces)
        avg_pen = sum(pens) / len(pens)
        max_over = max(overs)
        if valid_cnt >= 2:
            gaps = [
                int(valid_touches[idx_t]["bar_index"] - valid_touches[idx_t - 1]["bar_index"])
                for idx_t in range(1, valid_cnt)
            ]
            avg_gap = sum(gaps) / len(gaps)
            bounce_decay = bounces[-1] < (bounces[0] * 0.65)
            deepening_pen = pens[-1] > (pens[0] + 0.25)
        else:
            avg_gap = float(n_bars - start_idx)
            bounce_decay = False
            deepening_pen = False
    else:
        avg_bounce = safe_float(zone.get("bounce_strength_atr"), 0.8)
        max_bounce = avg_bounce
        avg_pen = 0.0
        max_over = 0.0
        avg_gap = 0.0
        bounce_decay = False
        deepening_pen = False

    absorption_warning = bool(valid_cnt >= 3 and (bounce_decay or deepening_pen or avg_gap < 4.0))

    # Hitung skor kualitas sentuhan (0 - 15 poin sesuai bobot Touch Quality)
    if classification == "INVALIDATED":
        q_score = 0.0
        reason = "Zona INVALIDATED karena penutupan candle menembus jauh melewati batas invalidasi."
    elif classification == "BROKEN":
        q_score = 3.5
        reason = "Zona BROKEN (ditembus candle closed); hanya relevan untuk skenario retest/role-reversal."
    elif classification == "UNTESTED":
        q_score = 11.5
        reason = "Zona UNTESTED (masih fresh sejak konfirmasi pivot)."
    elif classification in ("TESTED_1X", "TESTED_2X"):
        base_q = 12.0 if classification == "TESTED_1X" else 13.0
        bounce_bonus = min(2.0, max(0.0, (avg_bounce - 0.35) * 2.2))
        pen_penalty = max(0.0, (avg_pen - 0.55) * 3.0) + (1.0 if bounce_decay else 0.0)
        q_score = max(6.0, min(15.0, base_q + bounce_bonus - pen_penalty))
        reason = (
            f"{classification} dengan rata-rata pantulan {avg_bounce:.2f} ATR "
            f"dan penetrasi {avg_pen:.2f}x lebar zona."
        )
    else:  # TESTED_3X_PLUS
        # Semakin sering diuji tidak otomatis semakin kuat: penalti jika pantulan melemah / absorpsi
        base_q = 10.5
        if absorption_warning:
            q_score = max(4.5, base_q - 3.5 - min(2.0, max(0, valid_cnt - 3) * 0.7))
            reason = (
                f"TESTED_3X_PLUS ({valid_cnt}x) disertai tanda pelemahan/absorpsi "
                f"(bounce decay={bounce_decay}, gap={avg_gap:.1f} bar)."
            )
        else:
            bounce_bonus = min(2.0, max(0.0, (avg_bounce - 0.50) * 2.0))
            freq_penalty = min(2.5, max(0, valid_cnt - 3) * 0.6)
            q_score = max(6.5, min(13.0, base_q + bounce_bonus - freq_penalty))
            reason = f"TESTED_3X_PLUS ({valid_cnt}x) bertahan dengan rata-rata pantulan {avg_bounce:.2f} ATR."

    touch_info = {
        "classification": classification,
        "valid_touch_count": int(valid_cnt),
        "wick_only_count": int(wick_cnt),
        "total_interactions": int(valid_cnt + wick_cnt),
        "touches": valid_touches,
        "wick_contacts": wick_only_contacts,
        "avg_bounce_atr": round(avg_bounce, 3),
        "max_bounce_atr": round(max_bounce, 3),
        "avg_penetration_ratio": round(avg_pen, 3),
        "max_penetration_atr": round(max_over, 3),
        "avg_bars_between_touches": round(avg_gap, 2),
        "bounce_decay": bool(bounce_decay),
        "absorption_warning": bool(absorption_warning),
        "is_broken": bool(is_broken),
        "is_invalidated": bool(is_invalidated),
        "first_break_index": first_break_index,
        "touch_quality_score_15": round(q_score, 2),
        "reason": reason,
    }

    updated = dict(zone)
    updated["touch_status"] = classification
    updated["touch_count"] = int(valid_cnt)
    updated["wick_touch_count"] = int(wick_cnt)
    updated["is_broken"] = bool(is_broken)
    updated["is_invalidated"] = bool(is_invalidated)
    updated["is_active"] = not (is_broken or is_invalidated)
    updated["status"] = (
        "INVALIDATED"
        if is_invalidated
        else ("BROKEN" if is_broken else "ACTIVE")
    )
    updated["touch_analysis"] = touch_info
    return updated

