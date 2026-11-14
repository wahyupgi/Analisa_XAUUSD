# -*- coding: utf-8 -*-
"""
retest_detection.py — Modul Deteksi Retest & Role Reversal (Support <-> Resistance Flip)
Setelah Breakout Valid pada SNR Advanced Scanner.

Klasifikasi Wajib:
- RETEST_WAITING
- RETEST_IN_PROGRESS
- RETEST_CONFIRMED
- RETEST_FAILED
- NO_RETEST

Aturan:
- Jangan mengonfirmasi retest hanya berdasarkan harga menyentuh level.
- Diperlukan reaksi candle dan penutupan (close) yang sesuai dengan arah breakout.
"""
from modules.snr.breakout_detection import detect_zone_breakout
from modules.snr.config import get_snr_config
from modules.snr.data_provider import prepare_snr_frame, safe_float


def detect_zone_retest(
    zone,
    df,
    breakout_result=None,
    config=None,
    has_unclosed_last_bar=False,
):
    """
    Mendeteksi status retest setelah sebuah zona mengalami breakout pada CLOSED candles.
    """
    cfg = config or get_snr_config()
    rt_cfg = cfg.get("retest", {})

    prox_atr = safe_float(rt_cfg.get("retest_proximity_atr", 0.28), 0.28)
    max_lookback = max(5, int(rt_cfg.get("retest_max_lookback_bars", 28)))
    confirm_bounce_atr = safe_float(rt_cfg.get("retest_confirm_bounce_atr", 0.18), 0.18)
    fail_close_atr = safe_float(rt_cfg.get("retest_fail_close_inside_atr", 0.18), 0.18)

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
            "status": "NO_RETEST",
            "label": "NO RETEST",
            "summary_status": "NOT CONFIRMED",
            "direction": "NONE",
            "role_reversal": "NONE",
            "flipped_zone_type": None,
            "breakout_bar_index": None,
            "retest_bar_index": None,
            "retest_bar_time": None,
            "reaction_confirmed": False,
            "retest_held": False,
            "reason": "Data candle tidak tersedia untuk evaluasi retest.",
        }

    opens = clean_df["Open"].to_numpy(dtype=float)
    highs = clean_df["High"].to_numpy(dtype=float)
    lows = clean_df["Low"].to_numpy(dtype=float)
    closes = clean_df["Close"].to_numpy(dtype=float)
    atrs = clean_df["ATR14"].to_numpy(dtype=float)
    times = clean_df["Time"].tolist()
    n_bars = len(clean_df)

    if breakout_result is None:
        breakout_result = detect_zone_breakout(zone, clean_df, config=cfg, has_unclosed_last_bar=False)

    bo_status = str(breakout_result.get("status", "NO_BREAKOUT"))
    bo_idx = breakout_result.get("breakout_bar_index")

    # Cari histori breakout valid dalam jendela max_lookback jika breakout_result mencari lebih pendek
    conf_idx = int(zone.get("confirmed_index", zone.get("formation_index", 0)))
    search_start = max(conf_idx + 1, n_bars - max_lookback)
    if bo_idx is None or bo_status == "NO_BREAKOUT":
        found_idx = None
        for idx_b in range(search_start, max(search_start, n_bars - 1)):
            c_b = float(closes[idx_b])
            o_b = float(opens[idx_b])
            atr_b = max(0.10, safe_float(atrs[idx_b], ref_atr))
            if z_type == "RESISTANCE":
                if c_b > (z_high + 0.15 * atr_b) and (c_b - o_b) >= (0.35 * atr_b):
                    found_idx = idx_b
                    break
            else:
                if c_b < (z_low - 0.15 * atr_b) and (o_b - c_b) >= (0.35 * atr_b):
                    found_idx = idx_b
                    break
        if found_idx is not None:
            bo_idx = found_idx
        else:
            return {
                "zone_id": zone.get("zone_id", "UNKNOWN"),
                "timeframe": tf,
                "zone_type": z_type,
                "status": "NO_RETEST",
                "label": "NO RETEST",
                "summary_status": "NOT CONFIRMED",
                "direction": "NONE",
                "role_reversal": "NONE",
                "flipped_zone_type": None,
                "breakout_bar_index": None,
                "retest_bar_index": None,
                "retest_bar_time": None,
                "reaction_confirmed": False,
                "retest_held": False,
                "reason": f"Belum ada breakout valid pada {z_type} sehingga retest tidak relevan.",
            }

    bo_idx = int(bo_idx)
    expected_dir = "BULLISH" if z_type == "RESISTANCE" else "BEARISH"
    role_rev = "RESISTANCE_TO_SUPPORT" if z_type == "RESISTANCE" else "SUPPORT_TO_RESISTANCE"
    flipped_type = "SUPPORT" if z_type == "RESISTANCE" else "RESISTANCE"

    # Jika breakout baru saja terjadi pada candle closed paling akhir, maka belum ada candle retest setelahnya
    if bo_idx >= (n_bars - 1):
        return {
            "zone_id": zone.get("zone_id", "UNKNOWN"),
            "timeframe": tf,
            "zone_type": z_type,
            "status": "RETEST_WAITING",
            "label": "RETEST WAITING",
            "summary_status": "WAITING",
            "direction": expected_dir,
            "role_reversal": "PENDING_" + role_rev,
            "flipped_zone_type": flipped_type,
            "breakout_bar_index": bo_idx,
            "retest_bar_index": None,
            "retest_bar_time": None,
            "reaction_confirmed": False,
            "retest_held": False,
            "reason": f"Breakout baru terjadi pada bar #{bo_idx}; menunggu harga pullback untuk retest.",
        }

    # Evaluasi pergerakan setelah candle breakout (bo_idx + 1 .. n_bars - 1)
    retest_touch_bars = []
    failed_bar = None
    last_atr = max(0.10, safe_float(atrs[-1], ref_atr))

    for k in range(bo_idx + 1, n_bars):
        b_open = float(opens[k])
        b_high = float(highs[k])
        b_low = float(lows[k])
        b_close = float(closes[k])
        b_atr = max(0.10, safe_float(atrs[k], ref_atr))
        b_range = max(1e-6, b_high - b_low)

        if z_type == "RESISTANCE":
            # Harga kembali menembus turun ke bawah zona -> Retest Gagal
            if b_close < (z_low - fail_close_atr * b_atr):
                failed_bar = int(k)
                break
            # Harga menguji kembali area bekas resistance (sekarang kandidat support)
            if b_low <= (z_high + prox_atr * b_atr) and b_high >= z_low:
                lower_wick = max(0.0, min(b_open, b_close) - b_low)
                wick_ratio = lower_wick / b_range
                bullish_close = b_close > b_open and b_close >= z_high
                retest_touch_bars.append({
                    "bar_index": int(k),
                    "bar_time": str(times[k]),
                    "open": round(b_open, 2),
                    "high": round(b_high, 2),
                    "low": round(b_low, 2),
                    "close": round(b_close, 2),
                    "atr": round(b_atr, 4),
                    "wick_ratio": round(wick_ratio, 3),
                    "directional_close": bool(bullish_close),
                })
        else:  # SUPPORT yang ditembus ke bawah (sekarang kandidat resistance)
            # Harga kembali menembus naik ke atas zona -> Retest Gagal (reclaim support)
            if b_close > (z_high + fail_close_atr * b_atr):
                failed_bar = int(k)
                break
            # Harga menguji kembali area bekas support dari bawah
            if b_high >= (z_low - prox_atr * b_atr) and b_low <= z_high:
                upper_wick = max(0.0, b_high - max(b_open, b_close))
                wick_ratio = upper_wick / b_range
                bearish_close = b_close < b_open and b_close <= z_low
                retest_touch_bars.append({
                    "bar_index": int(k),
                    "bar_time": str(times[k]),
                    "open": round(b_open, 2),
                    "high": round(b_high, 2),
                    "low": round(b_low, 2),
                    "close": round(b_close, 2),
                    "atr": round(b_atr, 4),
                    "wick_ratio": round(wick_ratio, 3),
                    "directional_close": bool(bearish_close),
                })

    if failed_bar is not None:
        return {
            "zone_id": zone.get("zone_id", "UNKNOWN"),
            "timeframe": tf,
            "zone_type": z_type,
            "status": "RETEST_FAILED",
            "label": "RETEST FAILED",
            "summary_status": "FAILED",
            "direction": expected_dir,
            "role_reversal": "FAILED_" + role_rev,
            "flipped_zone_type": None,
            "breakout_bar_index": bo_idx,
            "retest_bar_index": failed_bar,
            "retest_bar_time": str(times[failed_bar]),
            "reaction_confirmed": False,
            "retest_held": False,
            "reason": (
                f"Retest gagal pada bar #{failed_bar}: harga menembus kembali melewati batas zona "
                f"({z_low:.2f}-{z_high:.2f}) dan membatalkan skenario role-reversal."
            ),
        }

    if not retest_touch_bars:
        return {
            "zone_id": zone.get("zone_id", "UNKNOWN"),
            "timeframe": tf,
            "zone_type": z_type,
            "status": "RETEST_WAITING",
            "label": "RETEST WAITING",
            "summary_status": "WAITING",
            "direction": expected_dir,
            "role_reversal": "PENDING_" + role_rev,
            "flipped_zone_type": flipped_type,
            "breakout_bar_index": bo_idx,
            "retest_bar_index": None,
            "retest_bar_time": None,
            "reaction_confirmed": False,
            "retest_held": False,
            "reason": f"Breakout valid telah terjadi; menunggu harga kembali menguji area {z_low:.2f}-{z_high:.2f}.",
        }

    # Harga sudah menyentuh area retest. Cek apakah sudah ada reaksi candle & penutupan searah breakout!
    last_rt = retest_touch_bars[-1]
    rt_idx = int(last_rt["bar_index"])
    last_close = float(closes[-1])
    last_open = float(opens[-1])

    if z_type == "RESISTANCE":
        min_retest_low = min(float(x["low"]) for x in retest_touch_bars)
        bounce_pts = max(0.0, last_close - max(z_low, min_retest_low))
        bounce_atr = bounce_pts / last_atr
        has_bullish_reaction = any(bool(x["directional_close"]) for x in retest_touch_bars) or (
            last_close > z_high and last_close > last_open and bounce_atr >= confirm_bounce_atr
        )
        held_outside = last_close >= z_high and float(closes[rt_idx:].min()) >= (z_low - 0.05 * last_atr)

        if has_bullish_reaction and held_outside and bounce_atr >= confirm_bounce_atr:
            return {
                "zone_id": zone.get("zone_id", "UNKNOWN"),
                "timeframe": tf,
                "zone_type": z_type,
                "status": "RETEST_CONFIRMED",
                "label": "RETEST CONFIRMED",
                "summary_status": "CONFIRMED",
                "direction": "BULLISH",
                "role_reversal": role_rev,
                "flipped_zone_type": "SUPPORT",
                "breakout_bar_index": bo_idx,
                "retest_bar_index": rt_idx,
                "retest_bar_time": last_rt["bar_time"],
                "reaction_confirmed": True,
                "retest_held": True,
                "bounce_atr": round(bounce_atr, 3),
                "reason": (
                    f"Retest BULLISH terkonfirmasi: Resistance ({z_low:.2f}-{z_high:.2f}) berubah fungsi "
                    f"menjadi Support dengan pantulan {bounce_atr:.2f} ATR dan penutupan di {last_close:.2f}."
                ),
            }
    else:  # SUPPORT breakdown -> retest dari bawah sebagai Resistance baru
        max_retest_high = max(float(x["high"]) for x in retest_touch_bars)
        drop_pts = max(0.0, min(z_high, max_retest_high) - last_close)
        drop_atr = drop_pts / last_atr
        has_bearish_reaction = any(bool(x["directional_close"]) for x in retest_touch_bars) or (
            last_close < z_low and last_close < last_open and drop_atr >= confirm_bounce_atr
        )
        held_outside = last_close <= z_low and float(closes[rt_idx:].max()) <= (z_high + 0.05 * last_atr)

        if has_bearish_reaction and held_outside and drop_atr >= confirm_bounce_atr:
            return {
                "zone_id": zone.get("zone_id", "UNKNOWN"),
                "timeframe": tf,
                "zone_type": z_type,
                "status": "RETEST_CONFIRMED",
                "label": "RETEST CONFIRMED",
                "summary_status": "CONFIRMED",
                "direction": "BEARISH",
                "role_reversal": role_rev,
                "flipped_zone_type": "RESISTANCE",
                "breakout_bar_index": bo_idx,
                "retest_bar_index": rt_idx,
                "retest_bar_time": last_rt["bar_time"],
                "reaction_confirmed": True,
                "retest_held": True,
                "bounce_atr": round(drop_atr, 3),
                "reason": (
                    f"Retest BEARISH terkonfirmasi: Support ({z_low:.2f}-{z_high:.2f}) berubah fungsi "
                    f"menjadi Resistance dengan penolakan {drop_atr:.2f} ATR dan penutupan di {last_close:.2f}."
                ),
            }

    # Jika harga menyentuh zona tetapi belum ada konfirmasi reaksi candle & close searah breakout
    return {
        "zone_id": zone.get("zone_id", "UNKNOWN"),
        "timeframe": tf,
        "zone_type": z_type,
        "status": "RETEST_IN_PROGRESS",
        "label": "RETEST IN PROGRESS",
        "summary_status": "IN PROGRESS",
        "direction": expected_dir,
        "role_reversal": "TESTING_" + role_rev,
        "flipped_zone_type": flipped_type,
        "breakout_bar_index": bo_idx,
        "retest_bar_index": rt_idx,
        "retest_bar_time": last_rt["bar_time"],
        "reaction_confirmed": False,
        "retest_held": True,
        "reason": (
            f"Harga sedang menguji area retest ({z_low:.2f}-{z_high:.2f}), "
            "namun belum ada reaksi candle dan penutupan terkonfirmasi."
        ),
    }

