# -*- coding: utf-8 -*-
"""
market_bias.py — Perhitungan Market Bias Independen Khusus SNR Advanced Scanner
Berdasarkan H4, H1, dan M30 (dengan M5 sebagai Konfirmasi Entry).

Klasifikasi Wajib:
- BULLISH
- BEARISH
- SIDEWAYS
- UNCONFIRMED
- CONFLICT
"""
from modules.snr.data_provider import safe_float


def evaluate_single_tf_snr_bias(
    timeframe,
    df,
    swing_result,
    zones_tf,
    momentum_tf=None,
    current_price=None,
):
    """
    Mengevaluasi bias SNR pada satu timeframe berdasarkan:
    1. Posisi harga terhadap support/resistance.
    2. Breakout valid.
    3. Retest.
    4. Struktur swing (HH/HL vs LH/LL).
    5. Reaksi harga & rejection.
    6. Dominasi support vs resistance.
    """
    tf = str(timeframe).upper()
    if df is None or getattr(df, "empty", True):
        return {
            "timeframe": tf,
            "bias": "UNCONFIRMED",
            "score": 0.0,
            "swing_structure": "UNCONFIRMED",
            "breakout_state": "NONE",
            "retest_state": "NONE",
            "reaction_state": "NONE",
            "dominance": "NEUTRAL",
            "price_position": "UNKNOWN",
            "reason": f"Data candle {tf} tidak mencukupi.",
        }

    cp = safe_float(current_price, 0.0)
    if cp <= 0:
        cp = safe_float(df["Close"].iloc[-1], 0.0)

    sh_list = (swing_result or {}).get("swing_highs", [])
    sl_list = (swing_result or {}).get("swing_lows", [])

    if not sh_list and not sl_list and not zones_tf:
        return {
            "timeframe": tf,
            "bias": "UNCONFIRMED",
            "score": 0.0,
            "swing_structure": "UNCONFIRMED",
            "breakout_state": "NONE",
            "retest_state": "NONE",
            "reaction_state": "NONE",
            "dominance": "NEUTRAL",
            "price_position": "UNKNOWN",
            "reason": f"Belum ada swing terkonfirmasi pada {tf}.",
        }

    bias_score = 0.0
    reasons = []

    # 1. Struktur Swing (HH/HL vs LH/LL)
    swing_structure = "RANGE"
    if len(sh_list) >= 2 and len(sl_list) >= 2:
        hh = float(sh_list[-1]["price"]) > float(sh_list[-2]["price"])
        lh = float(sh_list[-1]["price"]) < float(sh_list[-2]["price"])
        hl = float(sl_list[-1]["price"]) > float(sl_list[-2]["price"])
        ll = float(sl_list[-1]["price"]) < float(sl_list[-2]["price"])
        if hh and hl:
            swing_structure = "BULLISH_HH_HL"
            bias_score += 1.6
            reasons.append("struktur HH+HL")
        elif lh and ll:
            swing_structure = "BEARISH_LH_LL"
            bias_score -= 1.6
            reasons.append("struktur LH+LL")
        elif hh and not ll:
            swing_structure = "MILD_BULLISH"
            bias_score += 0.8
            reasons.append("higher high terbentuk")
        elif ll and not hh:
            swing_structure = "MILD_BEARISH"
            bias_score -= 0.8
            reasons.append("lower low terbentuk")
    elif len(sh_list) >= 1 and len(sl_list) >= 1:
        last_sh = sh_list[-1]
        last_sl = sl_list[-1]
        if cp > float(last_sh["price"]):
            swing_structure = "ABOVE_LAST_SWING_HIGH"
            bias_score += 1.2
        elif cp < float(last_sl["price"]):
            swing_structure = "BELOW_LAST_SWING_LOW"
            bias_score -= 1.2

    # 2. Breakout, Retest, Reaksi Harga, dan Dominasi Zona
    sup_zones = [z for z in (zones_tf or []) if str(z.get("zone_type", "")).upper() == "SUPPORT"]
    res_zones = [z for z in (zones_tf or []) if str(z.get("zone_type", "")).upper() == "RESISTANCE"]

    active_sups = [z for z in sup_zones if z.get("lifecycle_status", "ACTIVE") == "ACTIVE"]
    active_ress = [z for z in res_zones if z.get("lifecycle_status", "ACTIVE") == "ACTIVE"]

    breakout_state = "NONE"
    retest_state = "NONE"
    reaction_state = "NONE"

    for z in zones_tf or []:
        bo = z.get("breakout_info") or {}
        rt = z.get("retest_info") or {}
        rej = z.get("rejection_info") or {}

        if bo.get("status") == "VALID_BREAKOUT":
            if bo.get("direction") == "BULLISH":
                bias_score += 1.5
                breakout_state = "BULLISH_BREAKOUT"
                reasons.append("valid resistance breakout")
            elif bo.get("direction") == "BEARISH":
                bias_score -= 1.5
                breakout_state = "BEARISH_BREAKDOWN"
                reasons.append("valid support breakdown")

        if rt.get("status") == "RETEST_CONFIRMED":
            if rt.get("direction") == "BULLISH":
                bias_score += 1.2
                retest_state = "BULLISH_RETEST_CONFIRMED"
                reasons.append("bullish retest confirmed")
            elif rt.get("direction") == "BEARISH":
                bias_score -= 1.2
                retest_state = "BEARISH_RETEST_CONFIRMED"
                reasons.append("bearish retest confirmed")

        if rej.get("status") == "BULLISH_REJECTION" and rej.get("confirmed"):
            bias_score += 1.1
            reaction_state = "BULLISH_SUPPORT_REJECTION"
            reasons.append("support rejection")
        elif rej.get("status") == "BEARISH_REJECTION" and rej.get("confirmed"):
            bias_score -= 1.1
            reaction_state = "BEARISH_RESISTANCE_REJECTION"
            reasons.append("resistance rejection")

    # 3. Dominasi Support vs Resistance & Posisi Harga terhadap S1 / R1
    best_sup_score = max((safe_float(z.get("score"), 0.0) for z in active_sups), default=0.0)
    best_res_score = max((safe_float(z.get("score"), 0.0) for z in active_ress), default=0.0)

    dominance = "NEUTRAL"
    if best_sup_score - best_res_score >= 6.0:
        dominance = "SUPPORT_DOMINANT"
        bias_score += 0.65
        reasons.append(f"dominasi support ({best_sup_score:.0f} vs {best_res_score:.0f})")
    elif best_res_score - best_sup_score >= 6.0:
        dominance = "RESISTANCE_DOMINANT"
        bias_score -= 0.65
        reasons.append(f"dominasi resistance ({best_res_score:.0f} vs {best_sup_score:.0f})")

    price_position = "MID_RANGE"
    nearest_sup = min(active_sups, key=lambda z: abs(cp - safe_float(z.get("high"), cp)), default=None)
    nearest_res = min(active_ress, key=lambda z: abs(cp - safe_float(z.get("low"), cp)), default=None)

    if nearest_sup and nearest_res:
        s_hi = safe_float(nearest_sup.get("high"), cp)
        r_lo = safe_float(nearest_res.get("low"), cp)
        if r_lo > s_hi:
            pos_ratio = (cp - s_hi) / max(1e-6, r_lo - s_hi)
            if pos_ratio <= 0.25:
                price_position = "NEAR_SUPPORT"
            elif pos_ratio >= 0.75:
                price_position = "NEAR_RESISTANCE"
    elif nearest_sup and not nearest_res:
        price_position = "ABOVE_SUPPORT_OPEN_AIR"
        bias_score += 0.5
    elif nearest_res and not nearest_sup:
        price_position = "BELOW_RESISTANCE_OPEN_AIR"
        bias_score -= 0.5

    # Konfirmasi ringan dari arah penutupan candle terbaru jika selaras struktur
    if momentum_tf and isinstance(momentum_tf, dict):
        m_dir = momentum_tf.get("direction", "NEUTRAL")
        if m_dir == "BULLISH" and bias_score > 0:
            bias_score += 0.35
        elif m_dir == "BEARISH" and bias_score < 0:
            bias_score -= 0.35

    if bias_score >= 1.10:
        final_tf_bias = "BULLISH"
    elif bias_score <= -1.10:
        final_tf_bias = "BEARISH"
    else:
        final_tf_bias = "SIDEWAYS"

    reason_text = ", ".join(reasons) if reasons else "keseimbangan antara support dan resistance"
    return {
        "timeframe": tf,
        "bias": final_tf_bias,
        "score": round(bias_score, 2),
        "swing_structure": swing_structure,
        "breakout_state": breakout_state,
        "retest_state": retest_state,
        "reaction_state": reaction_state,
        "dominance": dominance,
        "price_position": price_position,
        "reason": f"{tf} {final_tf_bias} (skor={bias_score:+.2f}: {reason_text}).",
    }


def evaluate_snr_market_bias(
    frames_dict,
    swings_by_tf,
    zones_by_tf,
    momentum_by_tf=None,
    m5_entry_state="WAIT",
    current_price=None,
):
    """
    Menghitung Market Bias Independen SNR dari H4, H1, dan M30,
    serta status konfirmasi M5.

    Klasifikasi overall_bias:
    - BULLISH
    - BEARISH
    - SIDEWAYS
    - UNCONFIRMED
    - CONFLICT
    """
    momentum_by_tf = momentum_by_tf or {}
    per_tf = {}

    for tf in ("H4", "H1", "M30"):
        per_tf[tf] = evaluate_single_tf_snr_bias(
            timeframe=tf,
            df=frames_dict.get(tf),
            swing_result=swings_by_tf.get(tf),
            zones_tf=zones_by_tf.get(tf),
            momentum_tf=momentum_by_tf.get(tf),
            current_price=current_price,
        )

    # M5 digunakan sebagai konfirmasi entry, bukan penentu tunggal bias utama
    m5_df = frames_dict.get("M5")
    m5_mom = momentum_by_tf.get("M5") or {}
    m5_zones = zones_by_tf.get("M5") or []
    m5_label = str(m5_entry_state or "WAIT").upper()

    if m5_label in ("WAIT", "NONE", ""):
        # Cek apakah pada M5 sudah ada breakout/rejection terkonfirmasi
        has_m5_bull = any(
            (z.get("rejection_info") or {}).get("status") == "BULLISH_REJECTION"
            or (z.get("breakout_info") or {}).get("status") == "VALID_BREAKOUT"
            and (z.get("breakout_info") or {}).get("direction") == "BULLISH"
            for z in m5_zones
        )
        has_m5_bear = any(
            (z.get("rejection_info") or {}).get("status") == "BEARISH_REJECTION"
            or (z.get("breakout_info") or {}).get("status") == "VALID_BREAKOUT"
            and (z.get("breakout_info") or {}).get("direction") == "BEARISH"
            for z in m5_zones
        )
        if has_m5_bull and not has_m5_bear:
            m5_display = "BULLISH"
        elif has_m5_bear and not has_m5_bull:
            m5_display = "BEARISH"
        else:
            m5_display = "WAIT"
    else:
        m5_display = m5_label

    per_tf["M5"] = {
        "timeframe": "M5",
        "bias": m5_display,
        "momentum_direction": m5_mom.get("direction", "NEUTRAL"),
        "reason": f"M5 berfungsi sebagai konfirmasi trigger entry (status={m5_display}).",
    }

    b_h4 = per_tf["H4"]["bias"]
    b_h1 = per_tf["H1"]["bias"]
    b_m30 = per_tf["M30"]["bias"]

    struct_biases = [b_h4, b_h1, b_m30]
    if all(b == "UNCONFIRMED" for b in struct_biases):
        overall = "UNCONFIRMED"
        status_label = "UNCONFIRMED"
        summary_reason = "Data H4, H1, dan M30 belum mencukupi untuk mengonfirmasi bias SNR."
    else:
        # Cek konflik struktural antara H4 dan H1
        h4_h1_conflict = (b_h4 == "BULLISH" and b_h1 == "BEARISH") or (b_h4 == "BEARISH" and b_h1 == "BULLISH")
        weighted_score = (
            safe_float(per_tf["H4"]["score"], 0.0) * 0.45
            + safe_float(per_tf["H1"]["score"], 0.0) * 0.35
            + safe_float(per_tf["M30"]["score"], 0.0) * 0.20
        )

        if h4_h1_conflict and b_m30 != b_h4 and abs(weighted_score) < 0.85:
            overall = "CONFLICT"
            status_label = "CONFLICT BIAS"
            summary_reason = f"Konflik struktur antar timeframe utama (H4={b_h4} vs H1={b_h1}, M30={b_m30})."
        elif (
            (b_h4 == "BULLISH" and b_h1 in ("BULLISH", "SIDEWAYS"))
            or (b_h1 == "BULLISH" and b_m30 == "BULLISH" and b_h4 == "SIDEWAYS")
            or weighted_score >= 0.95
        ):
            overall = "BULLISH"
            status_label = "BULLISH BIAS"
            summary_reason = f"Dominasi struktur & reaksi support mendukung arah BULLISH (H4={b_h4}, H1={b_h1}, M30={b_m30})."
        elif (
            (b_h4 == "BEARISH" and b_h1 in ("BEARISH", "SIDEWAYS"))
            or (b_h1 == "BEARISH" and b_m30 == "BEARISH" and b_h4 == "SIDEWAYS")
            or weighted_score <= -0.95
        ):
            overall = "BEARISH"
            status_label = "BEARISH BIAS"
            summary_reason = f"Dominasi struktur & reaksi resistance mendukung arah BEARISH (H4={b_h4}, H1={b_h1}, M30={b_m30})."
        elif h4_h1_conflict:
            overall = "CONFLICT"
            status_label = "CONFLICT BIAS"
            summary_reason = f"Arah H4 ({b_h4}) dan H1 ({b_h1}) berlawanan tanpa dominasi satu arah yang tegas."
        else:
            overall = "SIDEWAYS"
            status_label = "SIDEWAYS BIAS"
            summary_reason = f"Pasar bergerak dalam rentang keseimbangan support-resistance (H4={b_h4}, H1={b_h1}, M30={b_m30})."

    return {
        "overall_bias": overall,
        "snr_status": status_label,
        "h4": b_h4,
        "h1": b_h1,
        "m30": b_m30,
        "m5": m5_display,
        "per_timeframe": per_tf,
        "reason": summary_reason,
    }

