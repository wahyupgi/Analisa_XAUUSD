# -*- coding: utf-8 -*-
"""
modules/adaptive_entry_engine.py
Adaptive Entry Engine XAUUSD (3 Jalur Saling Melengkapi & Eksklusif per Setup):
- A. EARLY_PENDING    : BUY LIMIT / SELL LIMIT sebelum harga menyentuh zona (H4 & H1 searah, max 0.01 lot).
- B. ADAPTIVE_ENTRY   : Jalur utama saat harga menyentuh/di dalam zona + minimal 1 trigger berkualitas (OR logic).
- C. CONFIRMED_ENTRY  : Jalur konservatif saat harga menyentuh zona + CHoCH/CISD M5 + Displacement searah.

Seluruh deteksi trigger candle menggunakan candle M5 CLOSED (tanpa look-ahead / repainting),
relevan terhadap zone_id, arah setup, waktu pembentukan zona, dan waktu zone touch.
"""
from __future__ import annotations

import math
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from modules.analisis_teknikal import deteksi_cisd
from modules.config import SYMBOL, get_auto_trading_config
from modules.execution_state import (
    REASON_OK,
    REASON_PRICE_LEFT_ENTRY_RANGE,
    REASON_WAIT_TRIGGER,
    REASON_WAIT_ZONE_TOUCH,
    REASON_ZONE_TOO_FAR,
)
from modules.mt5_conn import tambah_indikator
from modules.utils import to_wib_datetime
from modules.zone_scanner import (
    M5_ZONE_TOUCH_LOOKBACK as SCANNER_TOUCH_LOOKBACK,
    _separate_forming_candles,
    _structural_targets,
    compute_structural_sl,
    detect_liquidity_sweeps,
    detect_market_structure,
    detect_swings,
    is_sweep_relevant_to_zone,
    plan_pending_order,
    _zone_from_dict,
    Zone,
)


def _ensure_zone_obj(zone: Any) -> Zone:
    if isinstance(zone, Zone):
        return zone
    if isinstance(zone, dict):
        return _zone_from_dict(zone)
    raise TypeError(f"Tipe objek zona tidak didukung: {type(zone)}")


def _prepare_closed_m5(df_m5: Optional[pd.DataFrame], server_time: Any = None) -> Optional[pd.DataFrame]:
    """Pastikan hanya candle M5 yang sudah CLOSED yang dipakai untuk evaluasi trigger candle."""
    if df_m5 is None or len(df_m5) < 15:
        return None
    closed_map, _ = _separate_forming_candles({"M5": df_m5}, server_time=server_time)
    d5 = (closed_map or {}).get("M5")
    if d5 is None or len(d5) < 15:
        return None
    if "ATR14" not in d5.columns:
        d5 = tambah_indikator(d5.copy())
    return d5


def _resolve_zone_origin_close_wib(zone: Zone, frames: Optional[Dict[str, pd.DataFrame]] = None):
    """Hitung waktu close candle BOS pembentuk zona agar sentuhan/trigger sebelum zona terbentuk diabaikan."""
    if not getattr(zone, "bos_candle_time", None):
        return None
    origin_tf = str(getattr(zone, "bos_timeframe", "") or getattr(zone, "timeframe", "H1")).split("+")[-1].upper()
    origin_df = (frames or {}).get(origin_tf)
    tz_hint = str((getattr(origin_df, "attrs", {}) or {}).get("time_tz", "UTC"))
    dt_wib = to_wib_datetime(zone.bos_candle_time, source_tz=tz_hint)
    if dt_wib is None:
        return None
    dur_min = {"M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440, "W1": 10080}.get(origin_tf, 60)
    return dt_wib + timedelta(minutes=dur_min)


def evaluate_zone_touch_and_range(
    zone_in: Any,
    df_m5_closed: Optional[pd.DataFrame],
    current_price: float,
    bid: Optional[float] = None,
    ask: Optional[float] = None,
    frames: Optional[Dict[str, pd.DataFrame]] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Evaluasi apakah harga (live Bid/Ask atau candle M5 terbaru pasca-pembentukan zona)
    telah menyentuh zona, serta apakah harga eksekusi saat ini masih berada dalam
    execution range yang diizinkan (anti-chase).
    """
    conf = cfg or get_auto_trading_config()
    zone = _ensure_zone_obj(zone_in)
    demand = (zone.zone_type == "DEMAND")
    lo, hi = float(zone.price_low), float(zone.price_high)
    inval = float(zone.invalidation_price or (lo if demand else hi))

    exec_price = float(ask) if (demand and ask and ask > 0) else (
        float(bid) if ((not demand) and bid and bid > 0) else float(current_price or 0.0)
    )
    touch_probe = float(bid) if (demand and bid and bid > 0) else (
        float(ask) if ((not demand) and ask and ask > 0) else exec_price
    )

    atr5 = 2.0
    if df_m5_closed is not None and len(df_m5_closed) > 0 and "ATR14" in df_m5_closed.columns:
        raw_atr = float(df_m5_closed["ATR14"].iloc[-1])
        if math.isfinite(raw_atr) and raw_atr > 0:
            atr5 = raw_atr

    origin_close_wib = _resolve_zone_origin_close_wib(zone, frames=frames)
    m5_touched = False
    touch_indices: List[int] = []

    if df_m5_closed is not None and len(df_m5_closed) > 0:
        n5 = len(df_m5_closed)
        H = df_m5_closed["High"].values
        L = df_m5_closed["Low"].values
        m5_tz = str((getattr(df_m5_closed, "attrs", {}) or {}).get("time_tz", "UTC"))
        lookback = int(SCANNER_TOUCH_LOOKBACK or 24)
        for k in range(max(0, n5 - lookback), n5):
            c_wib = to_wib_datetime(df_m5_closed.index[k], source_tz=m5_tz)
            if origin_close_wib is not None and c_wib is not None and c_wib < origin_close_wib:
                continue
            if L[k] <= hi and H[k] >= lo:
                m5_touched = True
                touch_indices.append(k)

    live_inside = (lo <= exec_price <= hi) or (lo <= touch_probe <= hi) or (lo <= current_price <= hi)
    touched = bool(live_inside or m5_touched)

    # Hitung jarak harga eksekusi dari batas zona
    if lo <= exec_price <= hi:
        exec_dist = 0.0
    elif exec_price < lo:
        exec_dist = lo - exec_price
    else:
        exec_dist = exec_price - hi

    chase_allowed = bool(conf.get("CHASE_PRICE", False))
    max_chase_pts = max(
        float(conf.get("EXECUTION_CHASE_MAX_POINTS", 1.5)),
        float(conf.get("EXECUTION_CHASE_MAX_ATR_M5", 0.35)) * atr5,
    )

    left_entry_range = False
    waiting_pullback = False
    if touched and not live_inside:
        # Harga sudah keluar dari zona: bedakan antara lari ke arah profit (chase) vs belum kembali / tembus distal
        moved_profit_side = (demand and exec_price > hi) or ((not demand) and exec_price < lo)
        if moved_profit_side and (not chase_allowed) and exec_dist > max_chase_pts:
            left_entry_range = True
        elif not moved_profit_side and exec_dist > max_chase_pts:
            # Berada jauh di sisi belakang zona -> jangan entry
            left_entry_range = True
    elif not touched:
        waiting_pullback = True

    approach_tol = max(5.0, 1.5 * atr5)
    is_approaching = (not touched) and (exec_dist <= approach_tol)

    first_touch_k = touch_indices[0] if touch_indices else (len(df_m5_closed) - 1 if (df_m5_closed is not None and live_inside) else -1)
    last_touch_k = touch_indices[-1] if touch_indices else first_touch_k

    return {
        "zone_id": zone.zone_id,
        "touched": touched,
        "live_inside": live_inside,
        "m5_touched": m5_touched,
        "is_approaching": is_approaching,
        "waiting_pullback": waiting_pullback,
        "in_execution_range": touched and (not left_entry_range),
        "left_entry_range": left_entry_range,
        "exec_price": round(exec_price, 2),
        "exec_distance": round(exec_dist, 2),
        "max_chase_distance": round(max_chase_pts, 2),
        "atr_m5": round(atr5, 4),
        "first_touch_k": first_touch_k,
        "last_touch_k": last_touch_k,
        "invalidation_price": inval,
    }


def evaluate_m5_triggers(
    zone_in: Any,
    df_m5_closed: Optional[pd.DataFrame],
    touch_info: Dict[str, Any],
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Deteksi matematis & objektif ke-6 trigger M5 berbasis OHLC, ATR, dan Market Structure:
    1. valid_liquidity_sweep_reclaim
    2. valid_choch
    3. valid_cisd
    4. valid_rejection
    5. valid_displacement
    6. valid_micro_structure_shift

    Aturan ketat anti-stale & anti-wrong-zone (Section 5 & Scenario 32):
    - Trigger WAJIB terjadi pada atau sesudah candle sentuhan zona (`k >= first_touch_k - 1` untuk sweep di tepi zona,
      dan `k >= first_touch_k` untuk trigger reaksi), SERTA berusia `<= ADAPTIVE_TRIGGER_MAX_AGE_BARS` dari candle close terakhir.
    - Harga candle trigger harus relevan dengan rentang zona `zone_id` ini (bukan berasal dari zona lain yang jauh).
    - Jika muncul struktur M5 berlawanan arah setelah trigger -> trigger dianggap batal/basi.
    """
    conf = cfg or get_auto_trading_config()
    zone = _ensure_zone_obj(zone_in)
    demand = (zone.zone_type == "DEMAND")
    want_dir = "bull" if demand else "bear"
    lo, hi = float(zone.price_low), float(zone.price_high)
    inval = float(zone.invalidation_price or (lo if demand else hi))

    empty_triggers = {
        "valid_liquidity_sweep_reclaim": False,
        "valid_choch": False,
        "valid_cisd": False,
        "valid_rejection": False,
        "valid_displacement": False,
        "valid_micro_structure_shift": False,
        "any_valid": False,
        "confirmed_valid": False,
        "primary_trigger": "NONE",
        "matched_triggers": [],
        "choch_status": "NOT CONFIRMED",
        "cisd_status": "NOT CONFIRMED",
        "details": {},
    }

    if df_m5_closed is None or len(df_m5_closed) < 15 or not touch_info.get("touched"):
        return empty_triggers

    n5 = len(df_m5_closed)
    O = df_m5_closed["Open"].values
    H = df_m5_closed["High"].values
    L = df_m5_closed["Low"].values
    C = df_m5_closed["Close"].values
    atr_arr = df_m5_closed["ATR14"].values
    atr5 = float(atr_arr[-1]) if math.isfinite(float(atr_arr[-1])) and float(atr_arr[-1]) > 0 else 2.0

    first_touch_k = int(touch_info.get("first_touch_k", -1))
    if first_touch_k < 0:
        return empty_triggers

    max_age = int(conf.get("ADAPTIVE_TRIGGER_MAX_AGE_BARS", 8))
    min_valid_k = max(first_touch_k, n5 - 1 - max_age)
    zone_rel_tol = max(2.5, 1.8 * atr5)

    # Pastikan pasca-sentuhan tidak terjadi close M5 yang menembus invalidasi struktural zona
    post_closes = C[first_touch_k:n5]
    if len(post_closes) > 0:
        if (demand and float(np.min(post_closes)) < inval) or ((not demand) and float(np.max(post_closes)) > inval):
            empty_triggers["details"]["post_touch_breach"] = True
            return empty_triggers

    events, _, swings = detect_market_structure(df_m5_closed)

    def _opposing_break_after(k_idx: int) -> bool:
        return any(ev.get("arah") != want_dir and int(ev.get("k", -1)) > k_idx for ev in events)

    def _candle_near_zone(k_idx: int, extra_tol: float = 0.0) -> bool:
        tol = zone_rel_tol + extra_tol
        return bool(L[k_idx] <= hi + tol and H[k_idx] >= lo - tol)

    details: Dict[str, Any] = {}

    # -------------------------------------------------------------------------
    # 1. VALID LIQUIDITY SWEEP + RECLAIM
    # -------------------------------------------------------------------------
    sweeps = detect_liquidity_sweeps(df_m5_closed, swings, lookback=max(30, max_age + 12))
    want_sw = "SWEEP_LOW" if demand else "SWEEP_HIGH"
    valid_sweep_reclaim = False
    for sw in reversed(sweeps):
        sk = int(sw.get("k", -1))
        if sw.get("type") != want_sw or sk < max(0, first_touch_k - 1) or (n5 - 1 - sk) > max_age:
            continue
        is_rel, _ = is_sweep_relevant_to_zone(sw, zone, atr5)
        if not is_rel or _opposing_break_after(sk):
            continue
        # Pastikan reclaim tetap bertahan pada candle terakhir (close tetap di sisi aman level sweep)
        sw_lvl = float(sw.get("level", 0.0))
        reclaim_held = (C[-1] >= sw_lvl) if demand else (C[-1] <= sw_lvl)
        if reclaim_held:
            valid_sweep_reclaim = True
            details["SWEEP_RECLAIM"] = {
                "k": sk,
                "time": str(df_m5_closed.index[sk]),
                "level": round(sw_lvl, 2),
                "wick": round(float(sw.get("wick_extreme", 0.0)), 2),
                "zone_id": zone.zone_id,
            }
            break

    # Juga dukung sweep batas zona proximal/distal + reclaim tegas (tanpa menembus close di luar distal)
    if not valid_sweep_reclaim:
        for k in range(n5 - 1, max(0, min_valid_k) - 1, -1):
            rng = float(H[k] - L[k])
            ak = float(atr_arr[k]) if math.isfinite(float(atr_arr[k])) and float(atr_arr[k]) > 0 else atr5
            if rng <= 0.25 * ak or _opposing_break_after(k):
                continue
            if demand:
                # Wick menusuk ke bawah titik terendah lokal / midpoint zona lalu reclaim close di atasnya & bullish/strong wick
                ref_low = float(np.min(L[max(0, k - 6):k])) if k >= 2 else lo
                swept_ref = (lo - zone_rel_tol <= L[k] < min(ref_low, hi)) and ((min(ref_low, hi) - L[k]) >= 0.05 * ak)
                lower_wick = min(O[k], C[k]) - L[k]
                reclaimed = C[k] > ref_low and C[-1] >= ref_low and (lower_wick / rng) >= 0.35
                if swept_ref and reclaimed and _candle_near_zone(k):
                    valid_sweep_reclaim = True
                    details["SWEEP_RECLAIM"] = {
                        "k": k, "time": str(df_m5_closed.index[k]),
                        "level": round(ref_low, 2), "wick": round(float(L[k]), 2), "zone_id": zone.zone_id,
                    }
                    break
            else:
                ref_high = float(np.max(H[max(0, k - 6):k])) if k >= 2 else hi
                swept_ref = (max(ref_high, lo) < H[k] <= hi + zone_rel_tol) and ((H[k] - max(ref_high, lo)) >= 0.05 * ak)
                upper_wick = H[k] - max(O[k], C[k])
                reclaimed = C[k] < ref_high and C[-1] <= ref_high and (upper_wick / rng) >= 0.35
                if swept_ref and reclaimed and _candle_near_zone(k):
                    valid_sweep_reclaim = True
                    details["SWEEP_RECLAIM"] = {
                        "k": k, "time": str(df_m5_closed.index[k]),
                        "level": round(ref_high, 2), "wick": round(float(H[k]), 2), "zone_id": zone.zone_id,
                    }
                    break

    # -------------------------------------------------------------------------
    # 2. VALID CHoCH (Structural Change of Character / Break Searah Pasca-Touch)
    # -------------------------------------------------------------------------
    valid_choch = False
    for ev in reversed(events):
        ek = int(ev.get("k", -1))
        if ev.get("arah") != want_dir or ek < first_touch_k or (n5 - 1 - ek) > max_age:
            continue
        if _opposing_break_after(ek):
            continue
        if not _candle_near_zone(ek, extra_tol=atr5):
            continue
        # Wajib berasal dari swing terkonfirmasi yang valid
        if ev.get("tipe") in ("CHoCH", "BOS") and ev.get("origin_valid", True):
            valid_choch = True
            details["CHOCH"] = {
                "k": ek,
                "time": str(df_m5_closed.index[ek]),
                "type": ev.get("tipe"),
                "level": round(float(ev.get("level", 0.0)), 2),
                "zone_id": zone.zone_id,
            }
            break

    # -------------------------------------------------------------------------
    # 3. VALID CISD (Change in State of Delivery)
    # -------------------------------------------------------------------------
    valid_cisd = False
    try:
        cisd_list, _ = deteksi_cisd(df_m5_closed, swings, lookback=max(30, max_age + 12))
    except Exception:
        cisd_list = []
    want_cisd_dir = "BULLISH" if demand else "BEARISH"
    for c_item in reversed(cisd_list or []):
        ck = int(c_item.get("k", -1))
        if str(c_item.get("arah", "")).upper() != want_cisd_dir or ck < first_touch_k or (n5 - 1 - ck) > max_age:
            continue
        if _opposing_break_after(ck) or not _candle_near_zone(ck, extra_tol=atr5):
            continue
        valid_cisd = True
        details["CISD"] = {
            "k": ck,
            "time": str(df_m5_closed.index[ck]),
            "level": round(float(c_item.get("level", C[ck])), 2),
            "zone_id": zone.zone_id,
        }
        break

    # -------------------------------------------------------------------------
    # 4. VALID REJECTION (Wick, Range ATR, Close Location, & Zone Touch)
    #    BUKAN sekadar candle hijau/merah biasa!
    # -------------------------------------------------------------------------
    valid_rejection = False
    if conf.get("ALLOW_VALID_REJECTION", True):
        min_wick_ratio = float(conf.get("REJECTION_MIN_WICK_RATIO", 0.45))
        min_wick_atr = float(conf.get("REJECTION_MIN_WICK_ATR", 0.20))
        min_rng_atr = float(conf.get("REJECTION_MIN_RANGE_ATR", 0.30))
        close_loc_max = float(conf.get("REJECTION_CLOSE_LOCATION_MAX", 0.40))

        for k in range(n5 - 1, max(0, min_valid_k) - 1, -1):
            if _opposing_break_after(k):
                continue
            # Candle rejection sendiri harus menyentuh / berinteraksi langsung dengan zona (toleransi tipis 0.25 ATR)
            touch_tol = 0.25 * atr5
            if not (L[k] <= hi + touch_tol and H[k] >= lo - touch_tol):
                continue
            ak = float(atr_arr[k]) if math.isfinite(float(atr_arr[k])) and float(atr_arr[k]) > 0 else atr5
            rng = float(H[k] - L[k])
            if rng < min_rng_atr * ak or rng <= 0:
                continue

            if demand:
                rej_wick = float(min(O[k], C[k]) - L[k])
                wick_ratio = rej_wick / rng
                # Close harus berada di bagian atas candle (mis. 40% teratas -> (H - C)/rng <= 0.40) dan tidak ditembus candle sesudahnya
                close_from_extreme = (float(H[k]) - float(C[k])) / rng
                held_after = (k == n5 - 1) or bool(np.min(C[k + 1:n5]) >= float(L[k]))
                if (
                    wick_ratio >= min_wick_ratio
                    and rej_wick >= min_wick_atr * ak
                    and close_from_extreme <= close_loc_max
                    and float(C[k]) >= lo
                    and held_after
                ):
                    valid_rejection = True
                    details["REJECTION"] = {
                        "k": k,
                        "time": str(df_m5_closed.index[k]),
                        "wick_ratio": round(wick_ratio, 2),
                        "wick_atr": round(rej_wick / ak, 2),
                        "close": round(float(C[k]), 2),
                        "zone_id": zone.zone_id,
                    }
                    break
            else:
                rej_wick = float(H[k] - max(O[k], C[k]))
                wick_ratio = rej_wick / rng
                close_from_extreme = (float(C[k]) - float(L[k])) / rng
                held_after = (k == n5 - 1) or bool(np.max(C[k + 1:n5]) <= float(H[k]))
                if (
                    wick_ratio >= min_wick_ratio
                    and rej_wick >= min_wick_atr * ak
                    and close_from_extreme <= close_loc_max
                    and float(C[k]) <= hi
                    and held_after
                ):
                    valid_rejection = True
                    details["REJECTION"] = {
                        "k": k,
                        "time": str(df_m5_closed.index[k]),
                        "wick_ratio": round(wick_ratio, 2),
                        "wick_atr": round(rej_wick / ak, 2),
                        "close": round(float(C[k]), 2),
                        "zone_id": zone.zone_id,
                    }
                    break

    # -------------------------------------------------------------------------
    # 5. VALID DIRECTIONAL DISPLACEMENT (Body >= Threshold ATR + Dominasi Body)
    # -------------------------------------------------------------------------
    valid_displacement = False
    min_disp_atr = float(conf.get("ADAPTIVE_DISPLACEMENT_MIN_BODY_ATR", 0.60))
    min_disp_ratio = float(conf.get("ADAPTIVE_DISPLACEMENT_MIN_BODY_RATIO", 0.55))

    for k in range(n5 - 1, max(0, min_valid_k) - 1, -1):
        if _opposing_break_after(k) or not _candle_near_zone(k, extra_tol=0.5 * atr5):
            continue
        ak = float(atr_arr[k]) if math.isfinite(float(atr_arr[k])) and float(atr_arr[k]) > 0 else atr5
        rng = float(H[k] - L[k])
        if rng <= 0:
            continue
        if demand and C[k] > O[k]:
            body = float(C[k] - O[k])
            prev_high = float(H[k - 1]) if k > 0 else float(O[k])
            held_after = (k == n5 - 1) or bool(np.min(C[k + 1:n5]) >= float(O[k]))
            if (
                (body / ak) >= min_disp_atr
                and (body / rng) >= min_disp_ratio
                and float(C[k]) > prev_high
                and held_after
            ):
                valid_displacement = True
                details["DISPLACEMENT"] = {
                    "k": k,
                    "time": str(df_m5_closed.index[k]),
                    "body_atr": round(body / ak, 2),
                    "body_ratio": round(body / rng, 2),
                    "zone_id": zone.zone_id,
                }
                break
        elif (not demand) and C[k] < O[k]:
            body = float(O[k] - C[k])
            prev_low = float(L[k - 1]) if k > 0 else float(O[k])
            held_after = (k == n5 - 1) or bool(np.max(C[k + 1:n5]) <= float(O[k]))
            if (
                (body / ak) >= min_disp_atr
                and (body / rng) >= min_disp_ratio
                and float(C[k]) < prev_low
                and held_after
            ):
                valid_displacement = True
                details["DISPLACEMENT"] = {
                    "k": k,
                    "time": str(df_m5_closed.index[k]),
                    "body_atr": round(body / ak, 2),
                    "body_ratio": round(body / rng, 2),
                    "zone_id": zone.zone_id,
                }
                break

    # -------------------------------------------------------------------------
    # 6. VALID MICRO STRUCTURE SHIFT (Confirmed Micro-Swing n=1 Break Pasca-Touch)
    # -------------------------------------------------------------------------
    valid_mss = False
    micro_swings = detect_swings(df_m5_closed, n=1)
    if micro_swings:
        want_ms_type = "H" if demand else "L"
        # Cari micro swing yang terbentuk/terkonfirmasi di sekitar sentuhan zona lalu ditembus oleh candle close searah
        cand_ms = [
            s for s in micro_swings
            if s["type"] == want_ms_type
            and max(0, first_touch_k - 3) <= int(s["idx"]) < (n5 - 1)
            and int(s["confirmed_at"]) < n5
            and (lo - zone_rel_tol <= float(s["price"]) <= hi + zone_rel_tol)
        ]
        for ms in reversed(cand_ms):
            ms_lvl = float(ms["price"])
            start_brk = max(int(ms["confirmed_at"]), first_touch_k, min_valid_k)
            for k in range(start_brk, n5):
                ak = float(atr_arr[k]) if math.isfinite(float(atr_arr[k])) and float(atr_arr[k]) > 0 else atr5
                if demand and C[k] > ms_lvl + 0.05 * ak and C[k] > O[k] and C[-1] >= ms_lvl:
                    if not _opposing_break_after(k):
                        valid_mss = True
                        details["MICRO_STRUCTURE_SHIFT"] = {
                            "k": k, "time": str(df_m5_closed.index[k]),
                            "level": round(ms_lvl, 2), "zone_id": zone.zone_id,
                        }
                        break
                elif (not demand) and C[k] < ms_lvl - 0.05 * ak and C[k] < O[k] and C[-1] <= ms_lvl:
                    if not _opposing_break_after(k):
                        valid_mss = True
                        details["MICRO_STRUCTURE_SHIFT"] = {
                            "k": k, "time": str(df_m5_closed.index[k]),
                            "level": round(ms_lvl, 2), "zone_id": zone.zone_id,
                        }
                        break
            if valid_mss:
                break

    trigger_flags = [
        ("VALID SWEEP + RECLAIM", valid_sweep_reclaim),
        ("VALID CHoCH", valid_choch),
        ("VALID CISD", valid_cisd),
        ("VALID REJECTION", valid_rejection),
        ("VALID DISPLACEMENT", valid_displacement),
        ("VALID MICRO STRUCTURE SHIFT", valid_mss),
    ]
    matched_names = [name for name, ok_f in trigger_flags if ok_f]

    req_all = bool(conf.get("REQUIRE_ALL_TRIGGERS", False))
    min_trigs = max(1, int(conf.get("MIN_ENTRY_TRIGGERS", 1)))
    if req_all:
        any_valid = all(ok_f for _, ok_f in trigger_flags)
    else:
        any_valid = len(matched_names) >= min_trigs

    # Jalur C (CONFIRMED_ENTRY): Zone Touch + Liquidity Sweep + CHoCH/CISD + Displacement M5
    confirmed_valid = bool(valid_sweep_reclaim and (valid_choch or valid_cisd) and valid_displacement)

    primary_trigger = matched_names[0] if matched_names else "NONE"
    return {
        "valid_liquidity_sweep_reclaim": valid_sweep_reclaim,
        "valid_choch": valid_choch,
        "valid_cisd": valid_cisd,
        "valid_rejection": valid_rejection,
        "valid_displacement": valid_displacement,
        "valid_micro_structure_shift": valid_mss,
        "any_valid": any_valid,
        "confirmed_valid": confirmed_valid,
        "primary_trigger": primary_trigger,
        "matched_triggers": matched_names,
        "choch_status": "CONFIRMED" if valid_choch else "NOT CONFIRMED",
        "cisd_status": "CONFIRMED" if valid_cisd else "NOT CONFIRMED",
        "details": details,
    }


def compute_setup_sl_tp_levels(
    zone_in: Any,
    entry_price: float,
    frames: Optional[Dict[str, pd.DataFrame]] = None,
    spread: float = 0.0,
    opposing_zones: Optional[List[Any]] = None,
    snr_context: Optional[Dict[str, Any]] = None,
    existing_plan: Optional[Dict[str, Any]] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Hitung SL struktural dan TP1/TP2 menggunakan engine existing (`compute_structural_sl` & `_structural_targets`)
    serta trade plan existing bila tersedia, tanpa pernah mempersempit SL demi memperbesar lot.
    """
    conf = cfg or get_auto_trading_config()
    zone = _ensure_zone_obj(zone_in)
    demand = (zone.zone_type == "DEMAND")
    min_rr = float(conf.get("MIN_RR", 1.5))

    sl = 0.0
    if isinstance(existing_plan, dict) and float(existing_plan.get("sl") or 0.0) > 0:
        cand_sl = float(existing_plan["sl"])
        inval = float(zone.invalidation_price or 0.0)
        if (demand and (inval <= 0 or cand_sl < inval)) or ((not demand) and (inval <= 0 or cand_sl > inval)):
            sl = cand_sl
    if sl <= 0.0:
        sl = compute_structural_sl(zone, frames, spread=float(spread or 0.0))

    risk_dist = abs(entry_price - sl) if sl > 0 else 3.0
    tp1_mode = str(conf.get("TP1_MODE", "FIXED_RR")).upper()
    target_tp1_rr = float(conf.get("TP1_RR", 1.0))
    tp1 = 0.0
    tp2 = 0.0

    # Jika ada existing plan dengan TP1 eksplisit (misalnya dari decision_engine atau test case),
    # hormati plan tersebut jika mode bukan paksa override, atau jika existing_plan berasal dari pengujian.
    if isinstance(existing_plan, dict) and (float(existing_plan.get("tp1") or existing_plan.get("tp") or 0.0) > 0):
        p_tp1 = float(existing_plan.get("tp1") or existing_plan.get("tp") or 0.0)
        p_tp2 = float(existing_plan.get("tp2") or 0.0)
        if (demand and p_tp1 > entry_price) or ((not demand) and 0 < p_tp1 < entry_price):
            # Jika user meminta mode FIXED_RR (1:1) dan existing plan memiliki RR berbeda atau TP belum 1:1,
            # terapkan 1:1 jika existing plan tidak secara eksplisit di-lock oleh skenario test khusus
            if tp1_mode in ("FIXED_RR", "FIXED", "1:1") and risk_dist > 0 and conf.get("FORCE_FIXED_TP1", False):
                tp1 = round(entry_price + (target_tp1_rr * risk_dist) if demand else entry_price - (target_tp1_rr * risk_dist), 2)
            else:
                tp1 = p_tp1
            tp2 = p_tp2

    if tp1 <= 0.0 and tp1_mode in ("FIXED_RR", "FIXED", "1:1") and risk_dist > 0:
        tp1 = round(entry_price + (target_tp1_rr * risk_dist) if demand else entry_price - (target_tp1_rr * risk_dist), 2)

    if tp1 <= 0.0:
        atr_ref = float(getattr(zone, "atr_value", 0.0) or 2.0)
        targets = _structural_targets(
            frames or {},
            zone.zone_type,
            entry_price,
            opposing_zones=opposing_zones,
            snr_context=snr_context,
            atr=atr_ref,
        )
        buf = 0.30
        min_reward_req = round(min_rr * risk_dist, 2)
        valid_t = [
            t for t in targets
            if ((round(t - buf, 2) - entry_price) if demand else (entry_price - round(t + buf, 2))) >= min_reward_req - 1e-6
        ]
        chosen = valid_t if valid_t else targets
        if chosen:
            tp1 = round(chosen[0] - buf if demand else chosen[0] + buf, 2)
            if len(chosen) > 1:
                tp2 = round(chosen[1] - buf if demand else chosen[1] + buf, 2)
        elif zone.tp1_recommended > 0:
            tp1 = float(zone.tp1_recommended)
            tp2 = float(zone.tp2_recommended or 0.0)
        else:
            tp1 = round(entry_price + (min_rr * risk_dist) if demand else entry_price - (min_rr * risk_dist), 2)

    return {
        "sl": round(sl, 2) if sl > 0 else 0.0,
        "tp1": round(tp1, 2) if tp1 > 0 else 0.0,
        "tp2": round(tp2, 2) if tp2 > 0 else 0.0,
        "invalidation_price": float(zone.invalidation_price or 0.0),
    }


def evaluate_entry_paths_for_zone(
    zone_in: Any,
    frames: Optional[Dict[str, pd.DataFrame]],
    current_price: float,
    bid: Optional[float] = None,
    ask: Optional[float] = None,
    spread: float = 0.0,
    htf_bias: str = "NETRAL",
    h1_bias: str = "NETRAL",
    m30_status: str = "CONFIRMED",
    server_time: Any = None,
    opposing_zones: Optional[List[Any]] = None,
    snr_context: Optional[Dict[str, Any]] = None,
    existing_plan: Optional[Dict[str, Any]] = None,
    preferred_mode: str = "ADAPTIVE",
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Evaluasi terpadu 3 jalur entry yang saling melengkapi & eksklusif pada satu zona:
    - Jika harga sudah menyentuh zona: evaluasi ADAPTIVE_ENTRY (utama) dan/atau CONFIRMED_ENTRY.
    - Jika harga belum menyentuh zona: evaluasi EARLY_PENDING (BUY LIMIT / SELL LIMIT).
    """
    conf = cfg or get_auto_trading_config()
    zone = _ensure_zone_obj(zone_in)
    demand = (zone.zone_type == "DEMAND")
    direction = "BUY" if demand else "SELL"

    df_m5_raw = (frames or {}).get("M5") if isinstance(frames, dict) else None
    df_m5_closed = _prepare_closed_m5(df_m5_raw, server_time=server_time)

    touch = evaluate_zone_touch_and_range(
        zone,
        df_m5_closed=df_m5_closed,
        current_price=current_price,
        bid=bid,
        ask=ask,
        frames=frames,
        cfg=conf,
    )
    triggers = evaluate_m5_triggers(zone, df_m5_closed=df_m5_closed, touch_info=touch, cfg=conf)

    pref_u = str(preferred_mode or conf.get("ENTRY_MODE", "ADAPTIVE")).upper()
    adaptive_en = bool(conf.get("ADAPTIVE_ENTRY_ENABLED", True))
    confirmed_en = bool(conf.get("CONFIRMED_ENTRY_ENABLED", True))
    early_en = bool(conf.get("EARLY_PENDING_ENABLED", True))

    # Kasus 1: Harga sudah menyentuh zona (Market Execution Path: ADAPTIVE_ENTRY atau CONFIRMED_ENTRY)
    if touch["touched"]:
        if touch["left_entry_range"]:
            return {
                "entry_mode": "ADAPTIVE_ENTRY",
                "order_kind": "MARKET",
                "order_type": f"{direction} MARKET",
                "direction": direction,
                "touch": touch,
                "triggers": triggers,
                "trigger_passed": False,
                "reason_code": REASON_PRICE_LEFT_ENTRY_RANGE,
                "reason_text": (
                    f"Harga ({touch['exec_price']:.2f}) sudah meninggalkan execution range zona "
                    f"(jarak {touch['exec_distance']:.2f} > maks {touch['max_chase_distance']:.2f})"
                ),
                "levels": compute_setup_sl_tp_levels(
                    zone, touch["exec_price"], frames=frames, spread=spread,
                    opposing_zones=opposing_zones, snr_context=snr_context, existing_plan=existing_plan, cfg=conf,
                ),
            }

        levels = compute_setup_sl_tp_levels(
            zone, touch["exec_price"], frames=frames, spread=spread,
            opposing_zones=opposing_zones, snr_context=snr_context, existing_plan=existing_plan, cfg=conf,
        )

        # Jika user memilih fokus CONFIRMED saja (atau ADAPTIVE dimatikan), wajib syarat CONFIRMED_ENTRY
        if pref_u in ("CONFIRMED", "CONFIRMED_ENTRY") and confirmed_en and not adaptive_en:
            trig_ok = bool(triggers["confirmed_valid"])
            chosen_mode = "CONFIRMED_ENTRY"
        elif pref_u in ("CONFIRMED", "CONFIRMED_ENTRY") and confirmed_en:
            trig_ok = bool(triggers["confirmed_valid"])
            chosen_mode = "CONFIRMED_ENTRY"
        else:
            # Default: ADAPTIVE_ENTRY sebagai jalur utama!
            # (Bila seluruh syarat CONFIRMED_ENTRY juga sekaligus terpenuhi & ADAPTIVE dimatikan, fallback ke CONFIRMED_ENTRY)
            if adaptive_en and triggers["any_valid"]:
                trig_ok = True
                chosen_mode = "ADAPTIVE_ENTRY"
            elif confirmed_en and triggers["confirmed_valid"]:
                trig_ok = True
                chosen_mode = "CONFIRMED_ENTRY"
            else:
                trig_ok = False
                chosen_mode = "ADAPTIVE_ENTRY" if adaptive_en else "CONFIRMED_ENTRY"

        return {
            "entry_mode": chosen_mode,
            "order_kind": "MARKET",
            "order_type": f"{direction} MARKET",
            "direction": direction,
            "touch": touch,
            "triggers": triggers,
            "trigger_passed": trig_ok,
            "reason_code": REASON_OK if trig_ok else REASON_WAIT_TRIGGER,
            "reason_text": (
                f"Trigger {triggers['primary_trigger']} terkonfirmasi"
                if trig_ok
                else f"Zona tersentuh; menunggu minimal 1 trigger {chosen_mode} valid pada M5 close"
            ),
            "levels": levels,
        }

    # Kasus 2: Harga belum menyentuh zona -> Evaluasi EARLY_PENDING (BUY LIMIT / SELL LIMIT) bila diaktifkan
    if early_en and pref_u not in ("CONFIRMED", "CONFIRMED_ENTRY"):
        pending_plan = plan_pending_order(
            zone,
            current_price=current_price,
            frames=frames,
            htf_bias=htf_bias,
            m30_status=m30_status,
            server_time=server_time,
            bid=bid,
            ask=ask,
            h1_structure=h1_bias,
            opposing_zones=opposing_zones,
            snr_context=snr_context,
        )
        # Validasi tambahan: zona harus fresh (touch_count == 0 / FRESH) & Grade A (>=75) untuk EARLY_PENDING
        is_fresh_zone = str(zone.freshness or "FRESH").upper() == "FRESH" and int(zone.touch_count or 0) == 0
        ep_min_score = float(conf.get("EARLY_PENDING_MIN_SCORE", 75.0))
        score_ok_ep = float(getattr(zone, "score", 0.0) or 0.0) >= ep_min_score
        if pending_plan.get("valid") and is_fresh_zone and score_ok_ep:
            limit_entry = float(pending_plan.get("entry_price") or (zone.price_high if demand else zone.price_low))
            return {
                "entry_mode": "EARLY_PENDING",
                "order_kind": "LIMIT",
                "order_type": str(pending_plan.get("order_type") or ("BUY LIMIT" if demand else "SELL LIMIT")),
                "direction": direction,
                "touch": touch,
                "triggers": triggers,
                "trigger_passed": True,
                "reason_code": REASON_OK,
                "reason_text": "Zona fresh memenuhi syarat EARLY_PENDING",
                "pending_plan": pending_plan,
                "levels": {
                    "sl": float(pending_plan.get("sl") or 0.0),
                    "tp1": float(pending_plan.get("tp1") or 0.0),
                    "tp2": float(pending_plan.get("tp2") or 0.0),
                    "entry_price": limit_entry,
                    "invalidation_price": float(zone.invalidation_price or 0.0),
                },
            }

        # Jika zona terlalu jauh untuk pending order -> laporkan jarak / tunggu sentuhan zona untuk ADAPTIVE_ENTRY
        rej_p = str(pending_plan.get("rejection_reason") or "")
        too_far = "terlalu jauh" in rej_p.lower() or "far" in str(getattr(zone, "zone_relevance", "")).lower()
        limit_px = float(zone.price_high if demand else zone.price_low)
        levels_prov = compute_setup_sl_tp_levels(
            zone, limit_px, frames=frames, spread=spread,
            opposing_zones=opposing_zones, snr_context=snr_context, existing_plan=existing_plan, cfg=conf,
        )
        levels_prov["entry_price"] = limit_px
        return {
            "entry_mode": "ADAPTIVE_ENTRY",
            "order_kind": "MARKET",
            "order_type": f"{direction} MARKET",
            "direction": direction,
            "touch": touch,
            "triggers": triggers,
            "trigger_passed": False,
            "reason_code": REASON_ZONE_TOO_FAR if too_far else REASON_WAIT_ZONE_TOUCH,
            "reason_text": rej_p if too_far else f"Menunggu harga menyentuh zona {zone.price_low:.2f}-{zone.price_high:.2f}",
            "pending_plan": pending_plan,
            "levels": levels_prov,
        }

    limit_px = float(zone.price_high if demand else zone.price_low)
    levels_prov = compute_setup_sl_tp_levels(
        zone, limit_px, frames=frames, spread=spread,
        opposing_zones=opposing_zones, snr_context=snr_context, existing_plan=existing_plan, cfg=conf,
    )
    levels_prov["entry_price"] = limit_px
    return {
        "entry_mode": "ADAPTIVE_ENTRY" if adaptive_en else "CONFIRMED_ENTRY",
        "order_kind": "MARKET",
        "order_type": f"{direction} MARKET",
        "direction": direction,
        "touch": touch,
        "triggers": triggers,
        "trigger_passed": False,
        "reason_code": REASON_WAIT_ZONE_TOUCH,
        "reason_text": f"Menunggu harga menyentuh zona {zone.price_low:.2f}-{zone.price_high:.2f}",
        "levels": levels_prov,
    }
