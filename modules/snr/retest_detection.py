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
import pandas as pd


def classify_snr_flip_status(zone, breakout_info=None, retest_info=None):
    """
    Klasifikasi kanonis status role-reversal (flip) SNR berdasarkan bukti candle:
    - CONFIRMED_FLIP : breakout valid terkonfirmasi di TF asal zona + retest terkonfirmasi bertahan (reaction_confirmed & retest_held).
    - POTENTIAL_FLIP : zona asal masih BREAKOUT_PENDING/BREAKDOWN_PENDING, atau breakout TF asal terkonfirmasi namun retest masih WAITING/IN_PROGRESS.
    - INVALID        : zona RECLAIMED/RECLAIM_PENDING/INVALIDATED, atau FALSE_BREAKOUT/reclaim_detected, atau RETEST_FAILED.
    - NONE           : belum ada indikasi breakout/flip.
    """
    z = zone if isinstance(zone, dict) else {}
    life = str(z.get("lifecycle_status") or z.get("status") or "ACTIVE").upper()
    orig_bo = z.get("breakout_info") if isinstance(z.get("breakout_info"), dict) else {}
    bo = breakout_info if isinstance(breakout_info, dict) else orig_bo
    rt = retest_info if isinstance(retest_info, dict) else (z.get("retest_info") if isinstance(z.get("retest_info"), dict) else {})

    orig_bo_st = str(orig_bo.get("status") or bo.get("status") or "NO_BREAKOUT").upper()
    bo_st = str(bo.get("status") or "NO_BREAKOUT").upper()
    rt_st = str(rt.get("status") or "NO_RETEST").upper()

    if (
        life in ("RECLAIMED", "RECLAIM_PENDING", "INVALIDATED")
        or bool(z.get("is_invalidated"))
        or bool(orig_bo.get("reclaim_detected"))
        or bool(bo.get("reclaim_detected"))
        or orig_bo_st == "FALSE_BREAKOUT"
        or bo_st == "FALSE_BREAKOUT"
        or rt_st == "RETEST_FAILED"
    ):
        return "INVALID"

    origin_is_pending = (
        life in ("BREAKOUT_PENDING", "BREAKDOWN_PENDING")
        or orig_bo_st == "BREAKOUT_PENDING_CONFIRMATION"
    )
    if origin_is_pending:
        return "POTENTIAL_FLIP"

    origin_confirmed_break = (
        orig_bo_st == "VALID_BREAKOUT"
        and (life in ("BROKEN", "RETEST") or bool(z.get("is_broken")) or not orig_bo)
    )
    if origin_confirmed_break:
        if rt_st == "RETEST_CONFIRMED" and bool(rt.get("reaction_confirmed", True)) and bool(rt.get("retest_held", True)):
            return "CONFIRMED_FLIP"
        if rt_st in ("RETEST_WAITING", "RETEST_IN_PROGRESS", "NO_RETEST"):
            return "POTENTIAL_FLIP"

    if bo_st in ("VALID_BREAKOUT", "BREAKOUT_PENDING_CONFIRMATION"):
        return "POTENTIAL_FLIP"
    return "NONE"


def _resolve_breakout_bar_index(clean_df, breakout_result):
    """Map breakout bar index safely across timeframes (e.g., H1/M30 breakout_info evaluated on M5)."""
    bo_idx = breakout_result.get("breakout_bar_index")
    bo_time = str(breakout_result.get("breakout_bar_time") or "").strip()
    n_bars = len(clean_df)
    if n_bars == 0:
        return None
    times = clean_df["Time"].tolist() if "Time" in clean_df.columns else []
    if bo_idx is not None:
        try:
            idx_int = int(bo_idx)
            if 0 <= idx_int < n_bars and (not bo_time or not times or str(times[idx_int]) == bo_time):
                return idx_int
        except Exception:
            pass
    if bo_time and times:
        try:
            bo_ts = pd.to_datetime(bo_time, errors="coerce")
            ts_ser = pd.to_datetime(clean_df["Time"], errors="coerce")
            if not pd.isna(bo_ts) and not ts_ser.isna().all():
                pos = int(ts_ser.searchsorted(bo_ts, side="left"))
                if pos >= n_bars:
                    return n_bars - 1
                return max(0, pos)
        except Exception:
            pass
    if bo_idx is not None:
        try:
            idx_int = int(bo_idx)
            if 0 <= idx_int < n_bars:
                return idx_int
        except Exception:
            pass
    return None


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
            "flip_status": "NONE",
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

    bo_status = str(breakout_result.get("status", "NO_BREAKOUT")).upper()
    bo_idx = _resolve_breakout_bar_index(clean_df, breakout_result)
    z_life = str(zone.get("lifecycle_status") or zone.get("status") or "ACTIVE").upper()
    orig_bo = zone.get("breakout_info") if isinstance(zone.get("breakout_info"), dict) else {}
    orig_bo_st = str(orig_bo.get("status") or "").upper()
    origin_is_pending = (
        z_life in ("BREAKOUT_PENDING", "BREAKDOWN_PENDING")
        or orig_bo_st == "BREAKOUT_PENDING_CONFIRMATION"
    )
    origin_is_invalid = (
        z_life in ("RECLAIMED", "RECLAIM_PENDING", "INVALIDATED")
        or bool(zone.get("is_invalidated"))
        or bool(orig_bo.get("reclaim_detected"))
        or bool(breakout_result.get("reclaim_detected"))
    )

    if origin_is_invalid:
        life_ev_up = str(zone.get("lifecycle_event") or "").upper()
        is_inv_bo = bool(z_life == "RECLAIMED" or life_ev_up in ("BREAKOUT_INVALIDATED", "BREAKDOWN_INVALIDATED"))
        is_rt_fail = bool(is_inv_bo or z_life == "RECLAIM_PENDING" or life_ev_up == "RETEST_FAILED")
        inv_lbl = "BREAKDOWN INVALIDATED" if z_type == "SUPPORT" else "BREAKOUT INVALIDATED"
        inv_reason = (
            f"BREAKDOWN_INVALIDATED (INVALID): breakdown Support direclaim kembali oleh candle closed TF asal ({z_life})."
            if z_type == "SUPPORT"
            else f"BREAKOUT_INVALIDATED (INVALID): breakout Resistance direclaim kembali oleh candle closed TF asal ({z_life})."
        )
        return {
            "zone_id": zone.get("zone_id", "UNKNOWN"),
            "timeframe": tf,
            "zone_type": z_type,
            "status": "RETEST_FAILED" if is_rt_fail else "NO_RETEST",
            "breakout_invalidated": is_inv_bo,
            "flip_status": "INVALID",
            "label": inv_lbl if is_inv_bo else ("RETEST FAILED" if is_rt_fail else "NO RETEST"),
            "summary_status": "INVALIDATED" if is_inv_bo else ("FAILED" if is_rt_fail else "INVALID"),
            "direction": "NONE",
            "role_reversal": "INVALID",
            "flipped_zone_type": None,
            "breakout_bar_index": bo_idx,
            "retest_bar_index": None,
            "retest_bar_time": zone.get("lifecycle_event_time"),
            "retest_origin": "FROM_ABOVE" if z_type == "RESISTANCE" else "FROM_BELOW",
            "reaction_confirmed": False,
            "retest_held": False,
            "reason": (
                inv_reason
                if is_inv_bo
                else f"RETEST_FAILED (INVALID): penutupan candle closed TF asal gagal mempertahankan zona {z_type} ({z_life})."
                if is_rt_fail
                else f"Flip tidak valid (INVALID): zona {z_type} berstatus {z_life} / reclaim terdeteksi."
            ),
        }

    has_valid_bo = (
        not origin_is_pending
        and (
            (bo_status == "VALID_BREAKOUT")
            or (bo_status in ("BREAKOUT_PENDING_CONFIRMATION", "FALSE_BREAKOUT") and breakout_result.get("prior_confirmed_breakout"))
        )
    )
    if not has_valid_bo or bo_idx is None:
        is_pend_flip = bool(origin_is_pending or bo_status == "BREAKOUT_PENDING_CONFIRMATION")
        is_false_bo = bo_status == "FALSE_BREAKOUT"
        return {
            "zone_id": zone.get("zone_id", "UNKNOWN"),
            "timeframe": tf,
            "zone_type": z_type,
            "status": "NO_RETEST",
            "flip_status": "INVALID" if is_false_bo else ("POTENTIAL_FLIP" if is_pend_flip else "NONE"),
            "label": "NO RETEST",
            "summary_status": "INVALID" if is_false_bo else ("POTENTIAL FLIP" if is_pend_flip else "NOT CONFIRMED"),
            "direction": "NONE",
            "role_reversal": "POTENTIAL_FLIP" if is_pend_flip else ("INVALID" if is_false_bo else "NONE"),
            "flipped_zone_type": None,
            "breakout_bar_index": bo_idx if is_pend_flip else None,
            "retest_bar_index": None,
            "retest_bar_time": None,
            "retest_origin": "FROM_ABOVE" if z_type == "RESISTANCE" else "FROM_BELOW",
            "reaction_confirmed": False,
            "retest_held": False,
            "reason": (
                f"POTENTIAL_FLIP: Zona asal {tf} {z_type} masih {z_life if origin_is_pending else 'BREAKOUT PENDING'}; "
                "belum dapat dikonfirmasi sebagai FLIP sebelum candle closed TF asal mengonfirmasi breakout."
                if is_pend_flip
                else (
                    f"Flip INVALID: {z_type} mengalami FALSE BREAKOUT."
                    if is_false_bo
                    else (
                        f"Retest {z_type} menunggu breakout valid dari candle closed; "
                        "wick atau breakout pending tidak cukup untuk mengubah peran zona."
                    )
                )
            ),
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
            "flip_status": "POTENTIAL_FLIP",
            "label": "RETEST WAITING",
            "summary_status": "WAITING",
            "direction": expected_dir,
            "role_reversal": "PENDING_" + role_rev,
            "flipped_zone_type": flipped_type,
            "breakout_bar_index": bo_idx,
            "retest_bar_index": None,
            "retest_bar_time": None,
            "retest_origin": "FROM_ABOVE" if z_type == "RESISTANCE" else "FROM_BELOW",
            "reaction_confirmed": False,
            "retest_held": False,
            "reason": f"Breakout baru terjadi pada bar #{bo_idx}; menunggu harga pullback untuk retest (POTENTIAL_FLIP).",
        }
    if (n_bars - 1 - bo_idx) > max_lookback:
        return {
            "zone_id": zone.get("zone_id", "UNKNOWN"),
            "timeframe": tf,
            "zone_type": z_type,
            "status": "NO_RETEST",
            "flip_status": "POTENTIAL_FLIP",
            "label": "NO RETEST",
            "summary_status": "NOT CONFIRMED",
            "direction": expected_dir,
            "role_reversal": "PENDING_" + role_rev,
            "flipped_zone_type": flipped_type,
            "breakout_bar_index": bo_idx,
            "retest_bar_index": None,
            "retest_bar_time": None,
            "retest_origin": "FROM_ABOVE" if z_type == "RESISTANCE" else "FROM_BELOW",
            "reaction_confirmed": False,
            "retest_held": False,
            "reason": "Breakout valid terlalu lama untuk mengonfirmasi retest dari histori candle yang tersedia.",
        }

    # Evaluasi pergerakan setelah candle breakout (bo_idx + 1 .. n_bars - 1)
    retest_touch_bars = []
    failed_bar = None
    breakout_invalidated_flag = False
    last_atr = max(0.10, safe_float(atrs[-1], ref_atr))
    spread_pts = max(0.0, safe_float(zone.get("spread"), safe_float((cfg.get("risk") or {}).get("default_spread_points", 0.25), 0.25)))
    inv_close_atr = safe_float((cfg.get("touch") or {}).get("invalidation_close_beyond_atr", 0.42), 0.42)
    valid_pen_atr = safe_float((cfg.get("breakout") or {}).get("valid_close_penetration_atr", 0.15), 0.15)

    for k in range(bo_idx + 1, n_bars):
        b_open = float(opens[k])
        b_high = float(highs[k])
        b_low = float(lows[k])
        b_close = float(closes[k])
        b_atr = max(0.10, safe_float(atrs[k], ref_atr))
        b_range = max(1e-6, b_high - b_low)
        fail_thr = fail_close_atr * b_atr + spread_pts
        inv_thr = inv_close_atr * b_atr + spread_pts
        valid_thr = valid_pen_atr * b_atr + spread_pts

        if z_type == "RESISTANCE":
            # Hanya CLOSED candle di bawah z_low melewati toleransi ATR+spread yang gagal; wick penetration BUKAN kegagalan otomatis.
            pen_fail = z_low - b_close
            next_stay_below = (k + 1) < n_bars and float(closes[k + 1]) < (z_low - spread_pts)
            next_stay_valid = (k + 1) < n_bars and float(closes[k + 1]) < (z_low - valid_thr)
            if pen_fail >= inv_thr or (pen_fail >= valid_thr and next_stay_valid):
                failed_bar = int(k)
                breakout_invalidated_flag = True
                break
            if pen_fail >= fail_thr or (b_close < (z_low - spread_pts) and next_stay_below):
                failed_bar = int(k)
                breakout_invalidated_flag = False
                break
            # Harga menguji kembali area bekas resistance (termasuk wick di bawah z_low selama close bertahan)
            if b_low <= (z_high + prox_atr * b_atr + spread_pts) and b_high >= (z_low - spread_pts):
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
            pen_fail = b_close - z_high
            next_stay_above = (k + 1) < n_bars and float(closes[k + 1]) > (z_high + spread_pts)
            next_stay_valid = (k + 1) < n_bars and float(closes[k + 1]) > (z_high + valid_thr)
            if pen_fail >= inv_thr or (pen_fail >= valid_thr and next_stay_valid):
                failed_bar = int(k)
                breakout_invalidated_flag = True
                break
            if pen_fail >= fail_thr or (b_close > (z_high + spread_pts) and next_stay_above):
                failed_bar = int(k)
                breakout_invalidated_flag = False
                break
            # Harga menguji kembali area bekas support dari bawah (termasuk wick di atas z_high selama close bertahan)
            if b_high >= (z_low - prox_atr * b_atr - spread_pts) and b_low <= (z_high + spread_pts):
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
        inv_lbl_fb = "BREAKDOWN INVALIDATED" if z_type == "SUPPORT" else "BREAKOUT INVALIDATED"
        inv_reason_fb = (
            f"BREAKDOWN_INVALIDATED pada bar #{failed_bar}: candle closed mereclaim kembali di atas Support "
            f"({z_low:.2f}-{z_high:.2f}) melewati toleransi invalidasi; breakdown support & flip dibatalkan."
            if z_type == "SUPPORT"
            else f"BREAKOUT_INVALIDATED pada bar #{failed_bar}: candle closed menembus balik di bawah Resistance "
            f"({z_low:.2f}-{z_high:.2f}) melewati toleransi invalidasi; breakout resistance & flip dibatalkan."
        )
        return {
            "zone_id": zone.get("zone_id", "UNKNOWN"),
            "timeframe": tf,
            "zone_type": z_type,
            "status": "RETEST_FAILED",
            "breakout_invalidated": bool(breakout_invalidated_flag),
            "flip_status": "INVALID",
            "label": inv_lbl_fb if breakout_invalidated_flag else "RETEST FAILED",
            "summary_status": "INVALIDATED" if breakout_invalidated_flag else "FAILED",
            "direction": expected_dir,
            "role_reversal": "INVALIDATED_" + role_rev if breakout_invalidated_flag else "FAILED_" + role_rev,
            "flipped_zone_type": None,
            "breakout_bar_index": bo_idx,
            "retest_bar_index": failed_bar,
            "retest_bar_time": str(times[failed_bar]),
            "retest_origin": "FROM_ABOVE" if z_type == "RESISTANCE" else "FROM_BELOW",
            "reaction_confirmed": False,
            "retest_held": False,
            "reason": (
                inv_reason_fb
                if breakout_invalidated_flag
                else f"RETEST_FAILED pada bar #{failed_bar}: candle closed melewati batas zona {z_type} "
                f"({z_low:.2f}-{z_high:.2f}) melebihi toleransi ATR+spread dan membatalkan role-reversal."
            ),
        }

    if bo_status == "FALSE_BREAKOUT":
        return {
            "zone_id": zone.get("zone_id", "UNKNOWN"),
            "timeframe": tf,
            "zone_type": z_type,
            "status": "NO_RETEST",
            "flip_status": "INVALID",
            "label": "NO RETEST",
            "summary_status": "INVALID",
            "direction": "NONE",
            "role_reversal": "INVALID",
            "flipped_zone_type": None,
            "breakout_bar_index": bo_idx,
            "retest_bar_index": None,
            "retest_bar_time": None,
            "retest_origin": "FROM_ABOVE" if z_type == "RESISTANCE" else "FROM_BELOW",
            "reaction_confirmed": False,
            "retest_held": False,
            "reason": f"Breakout pada {z_type} tergolong FALSE BREAKOUT (INVALID) sehingga tidak valid untuk RETEST IN PROGRESS.",
        }

    if not retest_touch_bars:
        return {
            "zone_id": zone.get("zone_id", "UNKNOWN"),
            "timeframe": tf,
            "zone_type": z_type,
            "status": "RETEST_WAITING",
            "flip_status": "POTENTIAL_FLIP",
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
            "reason": f"Breakout valid telah terjadi (POTENTIAL_FLIP); menunggu harga kembali menguji area {z_low:.2f}-{z_high:.2f}.",
        }

    # Harga sudah menyentuh area retest. Cek apakah sudah ada reaksi candle & penutupan searah breakout!
    # Wajib minimal 2 bar pasca-breakout agar tidak terjadi sinyal premature pada satu candle.
    last_rt = retest_touch_bars[-1]
    rt_idx = int(last_rt["bar_index"])
    last_close = float(closes[-1])
    last_open = float(opens[-1])
    bars_after_bo = (len(closes) - 1) - int(bo_idx)

    if z_type == "RESISTANCE":
        min_retest_low = min(float(x["low"]) for x in retest_touch_bars)
        bounce_pts = max(0.0, last_close - max(z_low, min_retest_low))
        bounce_atr = bounce_pts / last_atr
        has_closed_bull_bar = any(
            float(closes[m]) > float(opens[m]) and float(closes[m]) >= z_high
            for m in range(rt_idx, n_bars)
        )
        has_bullish_reaction = (bars_after_bo >= 2) and (
            has_closed_bull_bar
            or (
                (last_close > last_open)
                and (
                    any(bool(x["directional_close"]) for x in retest_touch_bars)
                    or (last_close > z_high and bounce_atr >= confirm_bounce_atr)
                )
            )
        )
        held_outside = last_close >= z_high and float(closes[rt_idx:].min()) >= (z_low - 0.05 * last_atr)

        if has_bullish_reaction and held_outside and bounce_atr >= confirm_bounce_atr:
            return {
                "zone_id": zone.get("zone_id", "UNKNOWN"),
                "timeframe": tf,
                "zone_type": z_type,
                "status": "RETEST_CONFIRMED",
                "flip_status": "CONFIRMED_FLIP",
                "label": "RETEST CONFIRMED",
                "summary_status": "CONFIRMED",
                "direction": "BULLISH",
                "role_reversal": role_rev,
                "flipped_zone_type": "SUPPORT",
                "breakout_bar_index": bo_idx,
                "retest_bar_index": rt_idx,
                "retest_bar_time": last_rt["bar_time"],
                "retest_origin": "FROM_ABOVE",
                "reaction_confirmed": True,
                "retest_held": True,
                "bounce_atr": round(bounce_atr, 3),
                "reason": (
                    f"CONFIRMED_FLIP — Retest BULLISH terkonfirmasi: Resistance ({z_low:.2f}-{z_high:.2f}) berubah fungsi "
                    f"menjadi Support dengan pantulan {bounce_atr:.2f} ATR dan penutupan di {last_close:.2f}."
                ),
            }
    else:  # SUPPORT breakdown -> retest dari bawah sebagai Resistance baru
        max_retest_high = max(float(x["high"]) for x in retest_touch_bars)
        drop_pts = max(0.0, min(z_high, max_retest_high) - last_close)
        drop_atr = drop_pts / last_atr
        has_closed_bear_bar = any(
            float(closes[m]) < float(opens[m]) and float(closes[m]) <= z_low
            for m in range(rt_idx, n_bars)
        )
        has_bearish_reaction = (bars_after_bo >= 2) and (
            has_closed_bear_bar
            or (
                (last_close < last_open)
                and (
                    any(bool(x["directional_close"]) for x in retest_touch_bars)
                    or (last_close < z_low and drop_atr >= confirm_bounce_atr)
                )
            )
        )
        held_outside = last_close <= z_low and float(closes[rt_idx:].max()) <= (z_high + 0.05 * last_atr)

        if has_bearish_reaction and held_outside and drop_atr >= confirm_bounce_atr:
            return {
                "zone_id": zone.get("zone_id", "UNKNOWN"),
                "timeframe": tf,
                "zone_type": z_type,
                "status": "RETEST_CONFIRMED",
                "flip_status": "CONFIRMED_FLIP",
                "label": "RETEST CONFIRMED",
                "summary_status": "CONFIRMED",
                "direction": "BEARISH",
                "role_reversal": role_rev,
                "flipped_zone_type": "RESISTANCE",
                "breakout_bar_index": bo_idx,
                "retest_bar_index": rt_idx,
                "retest_bar_time": last_rt["bar_time"],
                "retest_origin": "FROM_BELOW",
                "reaction_confirmed": True,
                "retest_held": True,
                "bounce_atr": round(drop_atr, 3),
                "reason": (
                    f"CONFIRMED_FLIP — Retest BEARISH terkonfirmasi: Support ({z_low:.2f}-{z_high:.2f}) berubah fungsi "
                    f"menjadi Resistance dengan penolakan {drop_atr:.2f} ATR dan penutupan di {last_close:.2f}."
                ),
            }

    # Jika harga menyentuh zona tetapi belum ada konfirmasi reaksi candle & close searah breakout
    return {
        "zone_id": zone.get("zone_id", "UNKNOWN"),
        "timeframe": tf,
        "zone_type": z_type,
        "status": "RETEST_IN_PROGRESS",
        "flip_status": "POTENTIAL_FLIP",
        "label": "RETEST IN PROGRESS",
        "summary_status": "IN PROGRESS",
        "direction": expected_dir,
        "role_reversal": "TESTING_" + role_rev,
        "flipped_zone_type": flipped_type,
        "breakout_bar_index": bo_idx,
        "retest_bar_index": rt_idx,
        "retest_bar_time": last_rt["bar_time"],
        "retest_origin": "FROM_ABOVE" if z_type == "RESISTANCE" else "FROM_BELOW",
        "reaction_confirmed": False,
        "retest_held": True,
        "reason": (
            f"POTENTIAL_FLIP — Harga sedang menguji area retest ({z_low:.2f}-{z_high:.2f}), "
            "namun belum ada reaksi candle dan penutupan terkonfirmasi."
        ),
    }

