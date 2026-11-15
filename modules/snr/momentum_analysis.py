# -*- coding: utf-8 -*-
"""
momentum_analysis.py — Modul Analisis Momentum sebagai Pendukung Penilaian Kualitas
Breakout dan Rejection pada SNR Advanced Scanner.

Aturan:
- Jangan menjadikan momentum sebagai sinyal entry tunggal.
- Jika data tick volume tidak tersedia, tampilkan "N/A" dan jangan mengarang data volume.
"""
import numpy as np
from modules.snr.config import get_snr_config
from modules.snr.data_provider import prepare_snr_frame, safe_float


def analyze_momentum(
    df,
    timeframe="M5",
    config=None,
    has_unclosed_last_bar=False,
):
    """
    Mengevaluasi karakteristik momentum pada candle closed terakhir:
    - Candle body strength terhadap ATR
    - Candle range terhadap ATR
    - Consecutive bullish / bearish candles
    - Kecepatan pergerakan harga (price velocity dalam ATR/bar)
    - Tick volume (jika tersedia & valid, atau "N/A" jika tidak tersedia)
    - Indikasi pelemahan momentum (momentum weakening)
    """
    cfg = config or get_snr_config()
    mom_cfg = cfg.get("momentum", {})
    tf = str(timeframe).upper()

    strong_body_atr = safe_float(mom_cfg.get("strong_body_atr", 0.52), 0.52)
    mod_body_atr = safe_float(mom_cfg.get("moderate_body_atr", 0.30), 0.30)
    strong_range_atr = safe_float(mom_cfg.get("strong_range_atr", 0.80), 0.80)
    vel_lookback = max(2, int(mom_cfg.get("velocity_lookback_bars", 4)))
    vol_lookback = max(5, int(mom_cfg.get("volume_lookback_bars", 20)))

    clean_df, _ = prepare_snr_frame(df, has_unclosed_last_bar=has_unclosed_last_bar)
    if clean_df.empty:
        return {
            "timeframe": tf,
            "available": False,
            "direction": "NEUTRAL",
            "strength": "WEAK",
            "summary_label": "NEUTRAL",
            "body_atr": 0.0,
            "range_atr": 0.0,
            "body_to_range": 0.0,
            "consecutive_bullish": 0,
            "consecutive_bearish": 0,
            "velocity_atr_per_bar": 0.0,
            "tick_volume": "N/A",
            "tick_volume_avg": "N/A",
            "tick_volume_ratio": "N/A",
            "tick_volume_status": "N/A",
            "momentum_weakening": False,
            "weakening_reason": "NONE",
            "score_5": 0.0,
            "reason": "Data candle tidak tersedia.",
        }

    opens = clean_df["Open"].to_numpy(dtype=float)
    highs = clean_df["High"].to_numpy(dtype=float)
    lows = clean_df["Low"].to_numpy(dtype=float)
    closes = clean_df["Close"].to_numpy(dtype=float)
    atrs = clean_df["ATR14"].to_numpy(dtype=float)
    n_bars = len(clean_df)

    curr_atr = max(0.10, safe_float(atrs[-1], 1.0))
    last_open = float(opens[-1])
    last_high = float(highs[-1])
    last_low = float(lows[-1])
    last_close = float(closes[-1])

    signed_body = last_close - last_open
    abs_body = abs(signed_body)
    candle_range = max(1e-6, last_high - last_low)

    body_atr = abs_body / curr_atr
    range_atr = candle_range / curr_atr
    body_to_range = abs_body / candle_range

    # Hitung consecutive bullish / bearish candles
    consec_bull = 0
    consec_bear = 0
    for idx in range(n_bars - 1, -1, -1):
        diff = float(closes[idx] - opens[idx])
        if diff > 1e-6:
            if consec_bear > 0:
                break
            consec_bull += 1
        elif diff < -1e-6:
            if consec_bull > 0:
                break
            consec_bear += 1
        else:
            break

    # Kecepatan pergerakan harga (velocity)
    lb = min(vel_lookback, max(1, n_bars - 1))
    net_move = float(closes[-1] - closes[-1 - lb]) if n_bars > 1 else signed_body
    velocity_atr_per_bar = (net_move / float(lb)) / curr_atr

    # Evaluasi Tick Volume secara jujur (jangan mengarang jika tidak ada)
    has_vol = bool(clean_df["HasTickVolume"].iloc[-1]) if "HasTickVolume" in clean_df.columns else False
    tick_vol_val = "N/A"
    tick_vol_avg = "N/A"
    tick_vol_ratio = "N/A"
    tick_vol_status = "N/A"

    if has_vol and "TickVolume" in clean_df.columns:
        vols = clean_df["TickVolume"].to_numpy(dtype=float)
        curr_v = safe_float(vols[-1], 0.0)
        valid_vols = [float(v) for v in vols[-vol_lookback:] if not np.isnan(v) and float(v) > 0]
        if curr_v > 0 and valid_vols:
            avg_v = sum(valid_vols) / float(len(valid_vols))
            ratio_v = curr_v / max(1e-6, avg_v)
            tick_vol_val = round(curr_v, 1)
            tick_vol_avg = round(avg_v, 1)
            tick_vol_ratio = round(ratio_v, 2)
            if ratio_v >= 1.20:
                tick_vol_status = "HIGH"
            elif ratio_v <= 0.75:
                tick_vol_status = "LOW"
            else:
                tick_vol_status = "NORMAL"

    # Evaluasi pelemahan momentum (Momentum Weakening)
    momentum_weakening = False
    weakening_reasons = []
    if n_bars >= 3:
        prev_body_1 = abs(float(closes[-2] - opens[-2]))
        prev_body_2 = abs(float(closes[-3] - opens[-3]))
        if prev_body_2 > (0.40 * curr_atr) and prev_body_1 < prev_body_2 * 0.80 and abs_body < prev_body_1 * 0.80:
            momentum_weakening = True
            weakening_reasons.append("penyusutan body 3 candle berturut-turut")

        if net_move > (0.35 * curr_atr):
            upper_wick = max(0.0, last_high - max(last_open, last_close))
            if (upper_wick / candle_range) >= 0.48 and signed_body <= 0:
                momentum_weakening = True
                weakening_reasons.append("upper wick panjang menahan laju naik")
        elif net_move < -(0.35 * curr_atr):
            lower_wick = max(0.0, min(last_open, last_close) - last_low)
            if (lower_wick / candle_range) >= 0.48 and signed_body >= 0:
                momentum_weakening = True
                weakening_reasons.append("lower wick panjang menahan laju turun")

    # Tentukan arah dan kekuatan momentum
    if consec_bull >= 2 or (signed_body >= 0.22 * curr_atr and n_bars >= 2 and closes[-1] >= closes[-2]):
        direction = "BULLISH"
    elif consec_bear >= 2 or (signed_body <= -0.22 * curr_atr and n_bars >= 2 and closes[-1] <= closes[-2]):
        direction = "BEARISH"
    elif signed_body > 0.08 * curr_atr and velocity_atr_per_bar >= -0.05:
        direction = "BULLISH"
    elif signed_body < -0.08 * curr_atr and velocity_atr_per_bar <= 0.05:
        direction = "BEARISH"
    elif velocity_atr_per_bar >= 0.15:
        direction = "BULLISH"
    elif velocity_atr_per_bar <= -0.15:
        direction = "BEARISH"
    else:
        direction = "NEUTRAL"

    consec_active = max(consec_bull, consec_bear)
    if (body_atr >= strong_body_atr and body_to_range >= 0.55) or (
        body_atr >= mod_body_atr and consec_active >= 3 and abs(velocity_atr_per_bar) >= 0.25
    ):
        strength = "STRONG"
    elif body_atr >= mod_body_atr or abs(velocity_atr_per_bar) >= 0.15 or consec_active >= 2:
        strength = "MODERATE"
    else:
        strength = "WEAK"

    if momentum_weakening and strength == "STRONG":
        strength = "MODERATE"

    # Skor kontribusi momentum (0.0 - 5.0)
    score_5 = 1.5
    if direction in ("BULLISH", "BEARISH"):
        if strength == "STRONG":
            score_5 = 4.5
        elif strength == "MODERATE":
            score_5 = 3.5
        else:
            score_5 = 2.2
        if tick_vol_status == "HIGH":
            score_5 = min(5.0, score_5 + 0.5)
        if momentum_weakening:
            score_5 = max(1.0, score_5 - 1.2)
    else:
        score_5 = 1.0

    summary_label = direction if direction in ("BULLISH", "BEARISH") else "NEUTRAL"
    weak_str = ", ".join(weakening_reasons) if weakening_reasons else "NONE"
    reason = (
        f"Momentum {direction} ({strength}): body={body_atr:.2f} ATR, range={range_atr:.2f} ATR, "
        f"velocity={velocity_atr_per_bar:+.2f} ATR/bar, consec_bull={consec_bull}, consec_bear={consec_bear}, "
        f"tick_vol={tick_vol_status}, weakening={momentum_weakening}."
    )

    return {
        "timeframe": tf,
        "available": True,
        "direction": direction,
        "strength": strength,
        "summary_label": summary_label,
        "body_atr": round(body_atr, 3),
        "range_atr": round(range_atr, 3),
        "body_to_range": round(body_to_range, 3),
        "consecutive_bullish": int(consec_bull),
        "consecutive_bearish": int(consec_bear),
        "velocity_atr_per_bar": round(velocity_atr_per_bar, 3),
        "tick_volume": tick_vol_val,
        "tick_volume_avg": tick_vol_avg,
        "tick_volume_ratio": tick_vol_ratio,
        "tick_volume_status": tick_vol_status,
        "momentum_weakening": bool(momentum_weakening),
        "weakening_reason": weak_str,
        "score_5": round(score_5, 2),
        "reason": reason,
    }
