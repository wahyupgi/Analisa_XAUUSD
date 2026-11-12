# -*- coding: utf-8 -*-
"""
modules/zone_scanner.py
Python-Only XAUUSD Zone Scanner, Multi-Timeframe Validator & Entry Validator.

Independen dari Gemini/AI. Semua deteksi, scoring, validasi, dan keputusan
(VALID / WAIT / INVALID) dihitung deterministik dari OHLCV MT5 (candle CLOSE saja).

Rantai keputusan (evaluate_trade_decision):
    DATA -> H4 (bias + zona utama) -> H1 (konfirmasi) -> M30 (validasi struktur)
         -> M5 (trigger) -> RISK (SL/TP struktural, R:R) -> VALID / WAIT / INVALID
Setiap kegagalan wajib => WAIT (atau INVALID bila setup sudah mati secara struktural).
"""
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union, NamedTuple

import numpy as np
import pandas as pd

from modules.config import (
    SYMBOL, TIMEFRAMES, SWING_N, RR_MIN, SL_JARAK_MAKS, ENTRY_LEBAR_MAKS,
    SPREAD_MAKS_POIN, BLOKIR_JIKA_BERITA_GAGAL, _DIR_SCRIPT,
    FILE_STATE_ZONE_TRACKER, DEFAULT_PENDING_EXPIRATION_HOURS, ZONE_APPROACH_DISTANCE_POINTS,
)
from modules.utils import _catat_error
from modules.mt5_conn import ambil_data, pastikan_mt5_aktif, tambah_indikator

# =============================================================================
# KONSTANTA
# =============================================================================
MIN_DISPLACEMENT_ATR = 1.2      # impuls (OB -> puncak impuls) relatif ATR (disesuaikan untuk volatilitas XAUUSD)
MIN_IMPULSE_BODY_ATR = 0.5      # minimal satu candle impuls dengan body >= 0.5 ATR
MIN_BOS_STRENGTH_ATR = 0.08     # close harus melewati level BOS minimal 0.08 ATR
MAX_BOS_DISTANCE = 20           # BOS harus terjadi <= 20 candle setelah OB (impuls langsung)
IMPULSE_WINDOW = 12             # jendela candle impuls setelah OB
MIN_FVG_ATR = 0.25              # minimal imbalance FVG 3-candle
FVG_ADJACENT_TOL_ATR = 1.0      # toleransi jarak FVG terhadap OB candle pada leg impuls
MAX_ZONE_WIDTH_ATR = 3.0        # batas lebar zona relatif ATR
MAX_ZONE_WIDTH_POINTS = 20.0    # batas lebar zona H4 (H1 dibatasi 14.0 pt)
MIN_ZONE_WIDTH_POINTS = 0.4     # batas minimal lebar zona
MAX_TOUCH_COUNT = 2             # maksimal sentuhan sebelum status MITIGATED
MAX_ZONE_DISTANCE_ATR = 35.0    # jarak maksimal zona aktif dari harga berjalan (~140 pt)
SCORE_MIN_ACTIVE = 60.0         # skor minimal zona aktif lolos validasi
SCORE_GRADE_A_PLUS = 85.0
SCORE_GRADE_A = 75.0
SCORE_GRADE_B = 60.0
SWEEP_MIN_PENETRATION_ATR = 0.05
SWEEP_MIN_WICK_RATIO = 0.30
M5_EVENT_MAX_AGE = 12           # BOS/CHoCH M5 maksimal 12 candle lalu
M5_MIN_DISPLACEMENT_ATR = 0.8
M5_ZONE_TOUCH_LOOKBACK = 24
TF_MINUTES = {"M5": 5, "M30": 30, "H1": 60, "H4": 240, "D1": 1440}
MIN_BARS_DECISION = 60
FRAMES_CACHE_TTL = 10.0
SCAN_CACHE_TTL = 30.0
AUDIT_DEDUP_DETIK = 60.0

FILE_AUDIT_ZONA = os.path.join(_DIR_SCRIPT, "audit_zone_scanner.jsonl")

_FRAMES_CACHE: Dict[str, Any] = {"ts": 0.0, "data": None}
_SCAN_CACHE: Dict[str, Any] = {"key": None, "ts": 0.0, "data": None}
_AUDIT_LAST: Dict[str, Tuple[str, float]] = {}


# =============================================================================
# DATA STRUCTURE
# =============================================================================
@dataclass
class Zone:
    zone_id: str
    symbol: str
    timeframe: str
    zone_type: str                       # DEMAND / SUPPLY
    price_low: float
    price_high: float
    midpoint: float
    score: float
    grade: str                           # A+ / A / B / REJECTED
    origin_bos: Dict[str, Any]
    fvg_confirmation: Optional[Dict[str, Any]]
    displacement_strength: float
    freshness: str                       # FRESH / TESTED_1X / TESTED_2X / MITIGATED / INVALIDATED
    touch_count: int
    liquidity_sweep_status: str
    invalidation_price: float
    is_valid: bool
    rejection_reason: str
    ob_candle_time: Any = None
    bos_candle_time: Any = None
    sl_recommended: float = 0.0
    tp1_recommended: float = 0.0         # estimasi kasar; TP struktural dihitung di pipeline keputusan
    tp2_recommended: float = 0.0
    score_breakdown: Dict[str, float] = field(default_factory=dict)
    merged_from: List[str] = field(default_factory=list)
    htf_alignment: Dict[str, str] = field(default_factory=dict)
    audit: List[str] = field(default_factory=list)
    status: str = "RADAR"                # RADAR / READY / CONFIRMED / INVALIDATED / EXPIRED
    priority_score: float = 0.0
    distance: float = 0.0
    direction_status: str = ""           # ABOVE_PRICE / BELOW_PRICE / INSIDE_ZONE
    approach: str = "UNTOUCHED"          # TOUCHED / APPROACHING / UNTOUCHED / FAR
    is_aligned: bool = False
    priority_labels: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return _clean(asdict(self))


def _clean(o: Any) -> Any:
    """Konversi rekursif ke tipe JSON-aman (Timestamp, numpy, NaN)."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (pd.Timestamp, datetime)):
        return str(o)
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


# =============================================================================
# STRUKTUR PASAR
# =============================================================================
def detect_swings(df: pd.DataFrame, n: int = SWING_N) -> List[Dict[str, Any]]:
    """Swing High/Low; swing di candle i baru terkonfirmasi pada candle i+n (tanpa look-ahead)."""
    if df is None or len(df) < 2 * n + 1:
        return []
    H, L, idx = df["High"].values, df["Low"].values, df.index
    out = []
    for i in range(n, len(df) - n):
        if H[i] == np.max(H[i - n:i + n + 1]):
            out.append({"idx": i, "type": "H", "price": float(H[i]), "time": idx[i], "confirmed_at": i + n})
        if L[i] == np.min(L[i - n:i + n + 1]):
            out.append({"idx": i, "type": "L", "price": float(L[i]), "time": idx[i], "confirmed_at": i + n})
    out.sort(key=lambda s: s["idx"])
    return out


def detect_market_structure(df: pd.DataFrame, n: int = SWING_N):
    """
    BOS / CHoCH berbasis CLOSE (wick tidak dihitung).
    Event memuat origin_idx (swing lawan terakhir = asal impuls) dan origin_valid
    (False bila tidak ada swing lawan terkonfirmasi sehingga origin tidak dapat diidentifikasi).
    Return: (events, trend[1/-1/0], swings)
    """
    if df is None or len(df) < 2 * n + 1:
        return [], 0, []
    swings = detect_swings(df, n)
    C, idx = df["Close"].values, df.index
    conf: Dict[int, List[Dict[str, Any]]] = {}
    for s in swings:
        conf.setdefault(s["confirmed_at"], []).append(s)
    last_high = last_low = None          # level yang masih "hidup" (dipakai untuk deteksi break)
    ref_high = ref_low = None            # swing lawan terkonfirmasi terakhir (TIDAK di-reset) = asal impuls
    trend, events = 0, []
    for k in range(len(df)):
        for s in conf.get(k, []):
            if s["type"] == "H":
                last_high = ref_high = s
            else:
                last_low = ref_low = s
        if last_high is not None and C[k] > last_high["price"]:
            events.append({
                "k": k, "time": idx[k], "tipe": "CHoCH" if trend == -1 else "BOS", "arah": "bull",
                "level": float(last_high["price"]),
                "origin_idx": ref_low["idx"] if ref_low else max(0, k - 12),
                "origin_valid": ref_low is not None,
                "swing_idx": last_high["idx"], "swing_time": last_high["time"],
            })
            last_high, trend = None, 1
        elif last_low is not None and C[k] < last_low["price"]:
            events.append({
                "k": k, "time": idx[k], "tipe": "CHoCH" if trend == 1 else "BOS", "arah": "bear",
                "level": float(last_low["price"]),
                "origin_idx": ref_high["idx"] if ref_high else max(0, k - 12),
                "origin_valid": ref_high is not None,
                "swing_idx": last_low["idx"], "swing_time": last_low["time"],
            })
            last_low, trend = None, -1
    return events, trend, swings


def _atr_array(df: pd.DataFrame) -> np.ndarray:
    if "ATR14" in df.columns:
        return df["ATR14"].values
    return tambah_indikator(df.copy())["ATR14"].values


# =============================================================================
# LIQUIDITY SWEEP
# =============================================================================
def detect_liquidity_sweeps(df: pd.DataFrame, swings: List[Dict[str, Any]],
                            lookback: Optional[int] = 40) -> List[Dict[str, Any]]:
    """
    Sweep valid = wick menembus swing terkonfirmasi yang BELUM pernah ditembus, dengan:
      - penetrasi >= 0.05 ATR (bukan sentuhan kosmetik),
      - candle CLOSE kembali ke sisi dalam level,
      - wick rejection >= 30% dari range candle,
      - bukan breakout: candle berikutnya tidak close melewati ekstrem wick.
    lookback=None -> seluruh histori.
    """
    if df is None or len(df) < 2 * SWING_N + 1 or not swings:
        return []
    O, H, L, C = (df[c].values for c in ("Open", "High", "Low", "Close"))
    atr = _atr_array(df)
    idx = df.index
    srt = sorted(swings, key=lambda s: s["confirmed_at"])
    start_k = 0 if lookback is None else max(0, len(df) - lookback)
    res, ptr, act = [], 0, {"H": [], "L": []}
    for k in range(len(df)):
        while ptr < len(srt) and srt[ptr]["confirmed_at"] < k:
            act[srt[ptr]["type"]].append(srt[ptr])
            ptr += 1
        if k < start_k:
            continue
        a = atr[k] if np.isfinite(atr[k]) and atr[k] > 0 else 0.0
        rng = H[k] - L[k]
        if rng <= 0:
            continue
        for s in act["H"][-6:]:
            p = s["price"]
            if H[k] > p and C[k] <= p and (H[k] - p) >= SWEEP_MIN_PENETRATION_ATR * a:
                if k > s["idx"] + 1 and np.max(H[s["idx"] + 1:k]) > p:
                    continue  # level sudah pernah ditembus
                if (H[k] - max(O[k], C[k])) / rng < SWEEP_MIN_WICK_RATIO:
                    continue
                if k + 1 < len(df) and C[k + 1] > H[k]:
                    continue  # breakout lanjutan, bukan sweep
                is_eqh = any(abs(sh["price"] - p) <= max(0.25 * a, 0.5) for sh in act["H"] if sh["idx"] != s["idx"])
                pool_lbl = "EQH (Equal Highs)" if is_eqh else "Swing High"
                res.append({"k": k, "time": idx[k], "type": "SWEEP_HIGH", "level": p,
                            "wick_extreme": float(H[k]), "close_price": float(C[k]), "swing_time": s["time"],
                            "pool_type": "EQH" if is_eqh else "SWING_HIGH",
                            "pool_description": f"Buy-Side Liquidity ({pool_lbl}) @ {p:.2f}",
                            "description": f"Buy-Side Sweep ({pool_lbl} {p:.2f}, wick {H[k]:.2f}, close {C[k]:.2f})"})
        for s in act["L"][-6:]:
            p = s["price"]
            if L[k] < p and C[k] >= p and (p - L[k]) >= SWEEP_MIN_PENETRATION_ATR * a:
                if k > s["idx"] + 1 and np.min(L[s["idx"] + 1:k]) < p:
                    continue
                if (min(O[k], C[k]) - L[k]) / rng < SWEEP_MIN_WICK_RATIO:
                    continue
                if k + 1 < len(df) and C[k + 1] < L[k]:
                    continue
                is_eql = any(abs(sl["price"] - p) <= max(0.25 * a, 0.5) for sl in act["L"] if sl["idx"] != s["idx"])
                pool_lbl = "EQL (Equal Lows)" if is_eql else "Swing Low"
                res.append({"k": k, "time": idx[k], "type": "SWEEP_LOW", "level": p,
                            "wick_extreme": float(L[k]), "close_price": float(C[k]), "swing_time": s["time"],
                            "pool_type": "EQL" if is_eql else "SWING_LOW",
                            "pool_description": f"Sell-Side Liquidity ({pool_lbl}) @ {p:.2f}",
                            "description": f"Sell-Side Sweep ({pool_lbl} {p:.2f}, wick {L[k]:.2f}, close {C[k]:.2f})"})
    return res


def is_sweep_relevant_to_zone(sweep: Dict[str, Any], zone: Zone, atr: float) -> Tuple[bool, str]:
    """
    Validasi relevansi harga & lokasi sweep terhadap zona entry (TASK C.1, C.4):
      - DEMAND (BUY): sweep wajib SWEEP_LOW (Sell-Side Liquidity), level dan wick sweep
        berada di area zona (toleransi berbasis ATR & ukuran zona). Tidak boleh jauh di atas atau di bawah.
      - SUPPLY (SELL): sweep wajib SWEEP_HIGH (Buy-Side Liquidity), level dan wick sweep
        berada di area zona (toleransi berbasis ATR & ukuran zona). Tidak boleh jauh di bawah atau di atas.
    """
    lo, hi = float(zone.price_low), float(zone.price_high)
    demand = zone.zone_type == "DEMAND"
    atr_val = atr if (isinstance(atr, (int, float)) and math.isfinite(atr) and atr > 0) else 2.0
    tol_atr = max(2.0, 1.5 * atr_val)

    sw_type = sweep.get("type", "")
    level = float(sweep.get("level", 0.0))
    wick = float(sweep.get("wick_extreme", level))

    if demand:
        if sw_type != "SWEEP_LOW":
            return False, f"Tipe sweep {sw_type} tidak sesuai untuk Demand (wajib SWEEP_LOW)"
        if level > hi + tol_atr or wick > hi + tol_atr:
            return False, f"Sweep terjadi terlalu jauh di atas zona ({max(level, wick):.2f} > {hi + tol_atr:.2f})"
        if wick < lo - tol_atr:
            return False, f"Sweep menembus terlalu jauh di bawah zona ({wick:.2f} < {lo - tol_atr:.2f})"
        return True, "Sweep relevan dengan zona Demand"
    else:
        if sw_type != "SWEEP_HIGH":
            return False, f"Tipe sweep {sw_type} tidak sesuai untuk Supply (wajib SWEEP_HIGH)"
        if level < lo - tol_atr or wick < lo - tol_atr:
            return False, f"Sweep terjadi terlalu jauh di bawah zona ({min(level, wick):.2f} < {lo - tol_atr:.2f})"
        if wick > hi + tol_atr:
            return False, f"Sweep menembus terlalu jauh di atas zona ({wick:.2f} > {hi + tol_atr:.2f})"
        return True, "Sweep relevan dengan zona Supply"


# =============================================================================
# FVG
# =============================================================================
def detect_fvgs(df: pd.DataFrame, min_atr_mult: float = MIN_FVG_ATR) -> List[Dict[str, Any]]:
    """FVG 3-candle >= min_atr_mult*ATR dengan status UNTOUCHED / PARTIALLY_MITIGATED / FULLY_MITIGATED."""
    if df is None or len(df) < 3:
        return []
    atr, H, L, idx = _atr_array(df), df["High"].values, df["Low"].values, df.index
    out = []
    for i in range(2, len(df)):
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            continue
        if L[i] > H[i - 2] and (L[i] - H[i - 2]) >= min_atr_mult * a:
            lo, hi, typ = float(H[i - 2]), float(L[i]), "BULLISH_FVG"
            sub = L[i + 1:]
            st = "UNTOUCHED" if len(sub) == 0 or sub.min() >= hi else ("FULLY_MITIGATED" if sub.min() <= lo else "PARTIALLY_MITIGATED")
        elif H[i] < L[i - 2] and (L[i - 2] - H[i]) >= min_atr_mult * a:
            lo, hi, typ = float(H[i]), float(L[i - 2]), "BEARISH_FVG"
            sub = H[i + 1:]
            st = "UNTOUCHED" if len(sub) == 0 or sub.max() <= lo else ("FULLY_MITIGATED" if sub.max() >= hi else "PARTIALLY_MITIGATED")
        else:
            continue
        out.append({"idx": i - 1, "time": idx[i - 1], "type": typ, "price_low": lo, "price_high": hi,
                    "size_points": float(hi - lo), "size_atr": float((hi - lo) / a), "status": st})
    return out


# =============================================================================
# ORDER BLOCK
# =============================================================================
def detect_order_blocks(df, events, fvgs, sweeps, timeframe, max_events: int = 25):
    """
    OB = candle lawan terakhir pada leg [origin_idx .. BOS] sebelum impuls yang menghasilkan BOS/CHoCH.
    Acuan leg adalah origin_idx (swing lawan = asal impuls); swing_idx (level yang ditembus) TIDAK dipakai
    sebagai pengganti. Hard validation (semua wajib): origin teridentifikasi, candle lawan ada, BOS dekat
    (impuls langsung), BOS tegas (close melewati level >= 0.08 ATR), displacement >= 1.2 ATR,
    ada candle impuls dengan body >= 0.5 ATR, FVG searah yang relevan & belum terisi penuh,
    belum invalid (close menembus OB), sentuhan <= 2, lebar zona wajar.
    Return: (valid_zones, rejected_zones) -- keduanya lengkap dengan audit alasan.
    """
    if df is None or len(df) < 2 * SWING_N + 1 or not events:
        return [], []
    O, H, L, C = (df[c].values for c in ("Open", "High", "Low", "Close"))
    atr, idx = _atr_array(df), df.index
    ok_list: List[Zone] = []
    bad_list: List[Zone] = []
    seen_ob: set = set()

    for ev in events[-max_events:]:
        k, arah = ev["k"], ev["arah"]
        origin = int(ev["origin_idx"])
        reasons: List[str] = []
        if not ev.get("origin_valid", True):
            reasons.append("Asal impuls (swing lawan terkonfirmasi) tidak teridentifikasi")
        if origin >= k:
            continue
        bull = arah == "bull"
        zone_type = "DEMAND" if bull else "SUPPLY"
        m = origin + int(np.argmin(L[origin:k + 1]) if bull else np.argmax(H[origin:k + 1]))
        ob_idx = next((x for x in range(m, origin - 1, -1) if (C[x] < O[x] if bull else C[x] > O[x])), None)
        if ob_idx is None:
            reasons.append("Tidak ada candle lawan arah pada leg impuls")
            ob_idx = m
        if ob_idx >= k:
            continue

        # Cegah duplikasi zona dari OB candle yang sama
        ob_key = (zone_type, ob_idx)
        if ob_key in seen_ob:
            continue
        seen_ob.add(ob_key)

        ob_lo, ob_hi = float(L[ob_idx]), float(H[ob_idx])
        width, mid = ob_hi - ob_lo, (ob_lo + ob_hi) / 2.0
        a = float(atr[ob_idx]) if np.isfinite(atr[ob_idx]) and atr[ob_idx] > 0 else (
            float(atr[k]) if np.isfinite(atr[k]) and atr[k] > 0 else 2.0)

        # BOS & impuls
        ak = float(atr[k]) if np.isfinite(atr[k]) and atr[k] > 0 else a
        bos_strength = abs(C[k] - ev["level"]) / ak
        j_end = min(k, ob_idx + IMPULSE_WINDOW)
        seg = slice(ob_idx + 1, j_end + 1)
        if bull:
            disp = (float(np.max(H[seg])) - ob_lo) / a
        else:
            disp = (ob_hi - float(np.min(L[seg]))) / a
        max_body = float(np.max(np.abs(C[seg] - O[seg]))) / a if j_end >= ob_idx + 1 else 0.0
        if k - ob_idx > MAX_BOS_DISTANCE:
            reasons.append(f"BOS terlalu jauh dari OB ({k - ob_idx} candle > {MAX_BOS_DISTANCE}); bukan impuls langsung")
        if bos_strength < MIN_BOS_STRENGTH_ATR:
            reasons.append(f"BOS tipis (close hanya {bos_strength:.2f} ATR melewati level)")
        if disp < MIN_DISPLACEMENT_ATR:
            reasons.append(f"Displacement tidak memadai ({disp:.2f}x ATR < {MIN_DISPLACEMENT_ATR}x)")
        if max_body < MIN_IMPULSE_BODY_ATR:
            reasons.append(f"Tidak ada candle impuls bertubuh kuat (max body {max_body:.2f} ATR)")

        # FVG konfirmasi: searah, terbentuk pada leg impuls, belum terisi penuh, relevan harga dengan OB
        want = "BULLISH_FVG" if bull else "BEARISH_FVG"
        best = None
        j_fvg_end = min(len(df) - 1, k + 4)
        for f in fvgs:
            if f["type"] != want or f["status"] == "FULLY_MITIGATED" or not (ob_idx <= f["idx"] <= j_fvg_end):
                continue
            overlap = min(ob_hi, f["price_high"]) - max(ob_lo, f["price_low"])
            if overlap < -max(FVG_ADJACENT_TOL_ATR * a, 2.5 * a):
                continue
            rank = (f["status"] == "UNTOUCHED", f["size_atr"])
            if best is None or rank > best[0]:
                best = (rank, f)
        fvg = best[1] if best else None
        if fvg is None:
            reasons.append("Tidak ada FVG searah yang relevan/belum terisi penuh pada impuls OB")

        # Sentuhan & invalidasi (hanya candle CLOSE setelah BOS)
        touches, invalid = 0, False
        for x in range(k + 1, len(df)):
            if bull and C[x] < ob_lo:
                invalid = True
                reasons.append(f"Zona invalid (close {C[x]:.2f} < {ob_lo:.2f})")
                break
            if (not bull) and C[x] > ob_hi:
                invalid = True
                reasons.append(f"Zona invalid (close {C[x]:.2f} > {ob_hi:.2f})")
                break
            inside = L[x] <= ob_hi and H[x] >= ob_lo
            prev_inside = x > k + 1 and L[x - 1] <= ob_hi and H[x - 1] >= ob_lo
            if inside and not prev_inside:
                touches += 1
        if touches > MAX_TOUCH_COUNT:
            reasons.append(f"Zona sudah dimitigasi ({touches}x > {MAX_TOUCH_COUNT}x sentuhan)")
        fresh = ("INVALIDATED" if invalid else "FRESH" if touches == 0 else "TESTED_1X" if touches == 1
                 else "TESTED_2X" if touches == 2 else "MITIGATED")

        cap = min(MAX_ZONE_WIDTH_POINTS if "H4" in timeframe else 14.0, MAX_ZONE_WIDTH_ATR * a)
        if width > cap:
            reasons.append(f"Zona terlalu lebar ({width:.2f} pt > {cap:.2f})")
        if width < MIN_ZONE_WIDTH_POINTS:
            reasons.append(f"Zona terlalu sempit ({width:.2f} pt)")

        # Sweep pendukung (sisi yang tepat, di sekitar pembentukan OB)
        want_sw = "SWEEP_LOW" if bull else "SWEEP_HIGH"
        sw = next((s for s in sweeps if s["type"] == want_sw and ob_idx - 6 <= s["k"] <= ob_idx + 2), None)
        sweep_status = ("CONFIRMED_SWEEP_LOW" if bull else "CONFIRMED_SWEEP_HIGH") if sw else "NONE"

        buf = max(1.8, round(0.8 * a, 2))
        if bull:
            sl = round(ob_lo - buf, 2); risk = max(2.0, mid - sl)
            tp1, tp2 = round(mid + RR_MIN * risk, 2), round(mid + 2.5 * risk, 2)
        else:
            sl = round(ob_hi + buf, 2); risk = max(2.0, sl - mid)
            tp1, tp2 = round(mid - RR_MIN * risk, 2), round(mid - 2.5 * risk, 2)

        ev_out = dict(ev)
        ev_out["strength_atr"] = round(float(bos_strength), 2)
        ev_out["impulse_max_body_atr"] = round(max_body, 2)
        z = Zone(
            zone_id=f"ZONE_{timeframe}_{zone_type}_{int(idx[ob_idx].timestamp())}", symbol=SYMBOL,
            timeframe=timeframe, zone_type=zone_type, price_low=round(ob_lo, 2), price_high=round(ob_hi, 2),
            midpoint=round(mid, 2), score=0.0, grade="REJECTED", origin_bos=ev_out, fvg_confirmation=fvg,
            displacement_strength=round(float(disp), 2), freshness=fresh, touch_count=touches,
            liquidity_sweep_status=sweep_status, invalidation_price=round(ob_lo if bull else ob_hi, 2),
            is_valid=not reasons, rejection_reason="; ".join(reasons), ob_candle_time=idx[ob_idx],
            bos_candle_time=idx[k], sl_recommended=sl, tp1_recommended=tp1, tp2_recommended=tp2,
            audit=(["OB lolos seluruh hard validation"] if not reasons else list(reasons)),
        )
        (ok_list if not reasons else bad_list).append(z)
    return ok_list, bad_list


# =============================================================================
# SCORING
# =============================================================================
_ALIGN_WEIGHT = {"H4": 4.0, "H1": 3.0, "M30": 3.0}


def score_zone(zone: Zone, htf_bias_aligned: bool = True, htf_is_neutral: bool = False,
               htf_alignment: Optional[Dict[str, str]] = None) -> Zone:
    """
    Skor 0-100: struktur/BOS 25 (jenis BOS/CHoCH, TF, kekuatan close) + displacement 20 + FVG 20
    + freshness 15 + keselarasan HTF 10 (H4=4, H1=3, M30=3; netral setengah) + sweep 10.
    Skor TIDAK menggantikan hard validation: zona is_valid=False tetap 0/REJECTED.
    """
    if not zone.is_valid:
        zone.score, zone.grade = 0.0, "REJECTED"
        return zone
    ev = zone.origin_bos or {}
    base = 17.0 if ev.get("tipe") == "BOS" else 13.0
    tf_factor = 1.0 if "H4" in zone.timeframe else 0.8
    p_struct = min(25.0, base * tf_factor + min(8.0, 4.0 * float(ev.get("strength_atr", 0.0))))
    d = zone.displacement_strength
    p_disp = 20.0 if d >= 2.5 else 16.0 if d >= 2.0 else 12.0 if d >= MIN_DISPLACEMENT_ATR else 0.0
    st = (zone.fvg_confirmation or {}).get("status")
    p_fvg = 20.0 if st == "UNTOUCHED" else 14.0 if st == "PARTIALLY_MITIGATED" else 0.0
    p_fresh = {"FRESH": 15.0, "TESTED_1X": 8.0, "TESTED_2X": 3.0}.get(zone.freshness, 0.0)
    if htf_alignment is not None:
        p_htf = sum(w * (1.0 if htf_alignment.get(tf) == "ALIGNED" else 0.5 if htf_alignment.get(tf) == "NEUTRAL" else 0.0)
                    for tf, w in _ALIGN_WEIGHT.items())
        zone.htf_alignment = dict(htf_alignment)
    else:
        p_htf = 10.0 if htf_bias_aligned else 5.0 if htf_is_neutral else 0.0
    p_sweep = 10.0 if zone.liquidity_sweep_status.startswith("CONFIRMED") else 0.0
    total = round(p_struct + p_disp + p_fvg + p_fresh + p_htf + p_sweep, 1)
    zone.score = total
    zone.score_breakdown = {"struktur_bos": round(p_struct, 1), "displacement": p_disp, "fvg_konfirmasi": p_fvg,
                            "freshness": p_fresh, "konteks_htf": round(p_htf, 1), "liquidity_sweep": p_sweep}
    zone.grade = _grade(total)
    if zone.grade == "REJECTED":
        zone.is_valid = False
        zone.rejection_reason = f"Skor {total:.1f} < {SCORE_MIN_ACTIVE}"
        zone.audit.append(zone.rejection_reason)
    else:
        zone.audit.append(f"Diterima: grade {zone.grade}, skor {total:.1f}")
    return zone


def _grade(s: float) -> str:
    return "A+" if s >= SCORE_GRADE_A_PLUS else "A" if s >= SCORE_GRADE_A else "B" if s >= SCORE_GRADE_B else "REJECTED"


# =============================================================================
# MERGE
# =============================================================================
def merge_overlapping_zones(zones: List[Zone]) -> List[Zone]:
    """
    Gabungkan zona searah yang overlap harga. Konservatif: tanpa bonus skor lintas-TF
    (zona overlap umumnya dari pergerakan yang sama, bukan konfirmasi independen),
    freshness/touch mengambil yang terburuk.
    """
    if len(zones) <= 1:
        return zones
    out: List[Zone] = []
    order = {"FRESH": 0, "TESTED_1X": 1, "TESTED_2X": 2, "MITIGATED": 3, "INVALIDATED": 4}
    for z in sorted(zones, key=lambda x: (x.zone_type, x.price_low)):
        last = out[-1] if out else None
        if last is None or last.zone_type != z.zone_type or min(last.price_high, z.price_high) < max(last.price_low, z.price_low):
            out.append(z)
            continue
        lo, hi = min(last.price_low, z.price_low), max(last.price_high, z.price_high)
        max_w = MAX_ZONE_WIDTH_POINTS if any("H4" in tf for tf in (last.timeframe, z.timeframe)) else 14.0
        if hi - lo > max_w:
            lo, hi = max(last.price_low, z.price_low), min(last.price_high, z.price_high)
        best = last if last.score >= z.score else z
        tfs = sorted({*last.timeframe.split("+"), *z.timeframe.split("+")}, key=lambda t: -TF_MINUTES.get(t, 0))
        demand = last.zone_type == "DEMAND"
        mid = round((lo + hi) / 2.0, 2)
        out[-1] = Zone(
            zone_id=f"ZONE_{'+'.join(tfs)}_{last.zone_type}_{int(mid)}", symbol=last.symbol,
            timeframe="+".join(tfs), zone_type=last.zone_type, price_low=round(lo, 2), price_high=round(hi, 2),
            midpoint=mid, score=best.score, grade=best.grade, origin_bos=best.origin_bos,
            fvg_confirmation=best.fvg_confirmation or last.fvg_confirmation or z.fvg_confirmation,
            displacement_strength=max(last.displacement_strength, z.displacement_strength),
            freshness=max(last.freshness, z.freshness, key=lambda f: order.get(f, 9)),
            touch_count=max(last.touch_count, z.touch_count),
            liquidity_sweep_status=last.liquidity_sweep_status if last.liquidity_sweep_status != "NONE" else z.liquidity_sweep_status,
            invalidation_price=round(lo if demand else hi, 2), is_valid=True, rejection_reason="",
            ob_candle_time=best.ob_candle_time, bos_candle_time=best.bos_candle_time,
            sl_recommended=min(last.sl_recommended, z.sl_recommended) if demand else max(last.sl_recommended, z.sl_recommended),
            tp1_recommended=best.tp1_recommended, tp2_recommended=best.tp2_recommended,
            score_breakdown=best.score_breakdown,
            merged_from=sorted({*last.merged_from, *z.merged_from, last.zone_id, z.zone_id}),
            htf_alignment=best.htf_alignment, audit=best.audit + [f"Digabung dari {last.zone_id} & {z.zone_id}"],
            status=getattr(best, "status", "RADAR"),
            priority_score=getattr(best, "priority_score", 0.0),
            distance=getattr(best, "distance", 0.0),
            approach=getattr(best, "approach", "UNTOUCHED"),
            is_aligned=getattr(best, "is_aligned", False),
            priority_labels=list(getattr(best, "priority_labels", [])),
        )
    return out


# =============================================================================
# KONTEKS MULTI-TIMEFRAME & MARKET REGIME
# =============================================================================
def detect_market_regime(df: Optional[pd.DataFrame], n: int = SWING_N) -> Dict[str, Any]:
    """
    Identifikasi kondisi pasar dan audit struktur SMC (D1, H4, H1, M30):
      - Swing High dan Swing Low (HH/HL vs LH/LL)
      - BOS dan CHoCH berbasis candle close
      - Struktur aktif yang membatasi harga berjalan
      - Perubahan struktur terbaru
      - Penentuan regime: TRENDING_BULLISH, TRENDING_BEARISH, RANGING, CHOPPY
      - Diagnostic reason spesifik saat kondisi NETRAL
    """
    if df is None or len(df) < 2 * n + 20:
        return {
            "regime": "UNKNOWN", "bias": "NO_DATA",
            "detail": "Data tidak cukup",
            "diagnostic_reason": "Data candle tidak cukup untuk evaluasi struktur",
            "active_structure": {},
        }

    d = df if "EMA20" in df.columns else tambah_indikator(df.copy())
    events, trend, swings = detect_market_structure(d, n=n)
    last = d.iloc[-1]
    c_now = float(last["Close"])
    atr_val = float(d["ATR14"].iloc[-1]) if ("ATR14" in d.columns and float(d["ATR14"].iloc[-1]) > 0) else 2.0

    highs = [s for s in swings if s["type"] == "H"]
    lows = [s for s in swings if s["type"] == "L"]

    # Evaluasi HH / HL vs LH / LL
    is_hh = (highs[-1]["price"] > highs[-2]["price"]) if len(highs) >= 2 else None
    is_lh = (highs[-1]["price"] < highs[-2]["price"]) if len(highs) >= 2 else None
    is_hl = (lows[-1]["price"] > lows[-2]["price"]) if len(lows) >= 2 else None
    is_ll = (lows[-1]["price"] < lows[-2]["price"]) if len(lows) >= 2 else None

    # Tentukan level struktur aktif yang membatasi harga berjalan
    active_high = highs[-1]["price"] if highs else float(d["High"].max())
    active_low = lows[-1]["price"] if lows else float(d["Low"].min())
    latest_event = events[-1] if events else None

    # 1. Deteksi True Choppy / Whipsaw:
    # Terjadi >= 3 event bolak-balik dengan arah berlawanan dalam 20 candle terakhir
    recent_events = [e for e in events if len(d) - 1 - e["k"] <= 20]
    is_whipsaw = False
    if len(recent_events) >= 3:
        alternating = sum(1 for i in range(1, len(recent_events)) if recent_events[i]["arah"] != recent_events[i - 1]["arah"])
        if alternating >= 2:
            is_whipsaw = True

    if is_whipsaw:
        diag = f"Whipsaw: {len(recent_events)} event BOS/CHoCH bolak-balik berlawanan arah dalam 20 candle tanpa follow-through"
        return {
            "regime": "CHOPPY", "bias": "NETRAL", "detail": diag, "diagnostic_reason": diag,
            "active_structure": {"active_high": active_high, "active_low": active_low, "latest_event": latest_event},
        }

    # 2. Deteksi Ranging (Horizontal Consolidation):
    look = min(20, len(d))
    high_range = float(d["High"].iloc[-look:].max())
    low_range = float(d["Low"].iloc[-look:].min())
    range_w = high_range - low_range
    ema_diff = abs(float(last["EMA20"]) - float(last["EMA50"]))
    if range_w <= 2.6 * atr_val and ema_diff < 0.4 * atr_val:
        diag = f"Konsolidasi horizontal: range sempit {low_range:.2f}-{high_range:.2f} (lebar {range_w:.2f} pt <= 2.6 ATR) dengan EMA20/EMA50 mendatar"
        return {
            "regime": "RANGING", "bias": "NETRAL", "detail": diag, "diagnostic_reason": diag,
            "active_structure": {"active_high": active_high, "active_low": active_low, "latest_event": latest_event},
        }

    # 3. Deteksi Symmetrical Compression / Inside Structure:
    # Swing high makin rendah (LH) DAN swing low makin tinggi (HL) tanpa breakout baru
    if is_lh is True and is_hl is True:
        bars_since_ev = (len(d) - 1 - latest_event["k"]) if latest_event else 999
        if bars_since_ev > 8:
            diag = (f"Kompresi struktur: Pola inside/triangle (LH {highs[-1]['price']:.2f} < {highs[-2]['price']:.2f} & "
                    f"HL {lows[-1]['price']:.2f} > {lows[-2]['price']:.2f}), belum ada candle close breakout")
            return {
                "regime": "RANGING", "bias": "NETRAL", "detail": diag, "diagnostic_reason": diag,
                "active_structure": {"active_high": active_high, "active_low": active_low, "latest_event": latest_event},
            }

    # 4. Evaluasi Bullish / Bearish berdasarkan Struktur Aktif & Event Terbaru
    ema_bull = last["EMA20"] > last["EMA50"] and last["Close"] > last["EMA20"]
    ema_bear = last["EMA20"] < last["EMA50"] and last["Close"] < last["EMA20"]

    # A. Cek struktur dari event terbaru (BOS / CHoCH)
    if latest_event:
        ev_arah = latest_event.get("arah")
        ev_tipe = latest_event.get("tipe", "BOS")
        ev_lvl = float(latest_event.get("level", 0.0))

        if ev_arah == "bull":
            # Struktur bullish aktif: harga close belum menembus active_low
            if c_now >= active_low:
                detail = f"Struktur {ev_tipe} bullish @ {ev_lvl:.2f} aktif (active low {active_low:.2f} bertahan)"
                return {
                    "regime": "TRENDING_BULLISH", "bias": "BULLISH", "detail": detail,
                    "diagnostic_reason": detail,
                    "active_structure": {"active_high": active_high, "active_low": active_low, "latest_event": latest_event},
                }
        elif ev_arah == "bear":
            # Struktur bearish aktif: harga close belum menembus active_high
            if c_now <= active_high:
                detail = f"Struktur {ev_tipe} bearish @ {ev_lvl:.2f} aktif (active high {active_high:.2f} bertahan)"
                return {
                    "regime": "TRENDING_BEARISH", "bias": "BEARISH", "detail": detail,
                    "diagnostic_reason": detail,
                    "active_structure": {"active_high": active_high, "active_low": active_low, "latest_event": latest_event},
                }

    # B. Cek sekuens HH/HL vs LH/LL jika belum ada event terbaru
    if is_hh is True and is_hl is True:
        detail = f"Sekuens Swing Bullish: Higher High ({highs[-1]['price']:.2f}) & Higher Low ({lows[-1]['price']:.2f})"
        return {
            "regime": "TRENDING_BULLISH", "bias": "BULLISH", "detail": detail, "diagnostic_reason": detail,
            "active_structure": {"active_high": active_high, "active_low": active_low, "latest_event": latest_event},
        }
    elif is_lh is True and is_ll is True:
        detail = f"Sekuens Swing Bearish: Lower High ({highs[-1]['price']:.2f}) & Lower Low ({lows[-1]['price']:.2f})"
        return {
            "regime": "TRENDING_BEARISH", "bias": "BEARISH", "detail": detail, "diagnostic_reason": detail,
            "active_structure": {"active_high": active_high, "active_low": active_low, "latest_event": latest_event},
        }

    # C. Cek EMA Stack & Trend Value
    if ema_bull:
        detail = "EMA/Struktur tren bullish (Close > EMA20 > EMA50)"
        return {
            "regime": "TRENDING_BULLISH", "bias": "BULLISH", "detail": detail, "diagnostic_reason": detail,
            "active_structure": {"active_high": active_high, "active_low": active_low, "latest_event": latest_event},
        }
    elif ema_bear:
        detail = "EMA/Struktur tren bearish (Close < EMA20 < EMA50)"
        return {
            "regime": "TRENDING_BEARISH", "bias": "BEARISH", "detail": detail, "diagnostic_reason": detail,
            "active_structure": {"active_high": active_high, "active_low": active_low, "latest_event": latest_event},
        }
    elif trend == 1 and c_now >= active_low:
        detail = f"Struktur BOS bullish aktif (fase transisi, active low {active_low:.2f} bertahan)"
        return {
            "regime": "TRENDING_BULLISH", "bias": "BULLISH", "detail": detail, "diagnostic_reason": detail,
            "active_structure": {"active_high": active_high, "active_low": active_low, "latest_event": latest_event},
        }
    elif trend == -1 and c_now <= active_high:
        detail = f"Struktur BOS bearish aktif (fase transisi, active high {active_high:.2f} bertahan)"
        return {
            "regime": "TRENDING_BEARISH", "bias": "BEARISH", "detail": detail, "diagnostic_reason": detail,
            "active_structure": {"active_high": active_high, "active_low": active_low, "latest_event": latest_event},
        }

    # D. Fallback NETRAL dengan Diagnostic Reason Spesifik
    diag = f"Struktur netral: Harga ({c_now:.2f}) bergerak di antara Swing High {active_high:.2f} dan Swing Low {active_low:.2f} tanpa konfirmasi candle close breakout baru"
    return {
        "regime": "RANGING", "bias": "NETRAL", "detail": diag, "diagnostic_reason": diag,
        "active_structure": {"active_high": active_high, "active_low": active_low, "latest_event": latest_event},
    }


def _tf_label(df: Optional[pd.DataFrame]) -> str:
    """BULLISH / BEARISH / NETRAL / NO_DATA: struktur (BOS/CHoCH close) + filter EMA; harus searah, bukan sekadar salah satu."""
    reg = detect_market_regime(df)
    return reg.get("bias", "NETRAL")


def evaluate_htf_context(frames: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """D1 (konteks), H4 (bias utama), H1 (konfirmasi), M30 (validasi). Konflik H4 vs H1 => WAIT."""
    regimes = {tf: detect_market_regime(frames.get(tf)) for tf in ("D1", "H4", "H1", "M30")}
    r = {tf: regimes[tf]["bias"] for tf in ("D1", "H4", "H1", "M30")}
    r["regimes"] = {tf: regimes[tf]["regime"] for tf in ("D1", "H4", "H1", "M30")}
    r["h4_diagnostic"] = regimes["H4"].get("diagnostic_reason", regimes["H4"].get("detail", ""))
    r["h4_structure"] = regimes["H4"].get("active_structure", {})
    r["h4_detail"] = regimes["H4"]
    opp = {"BULLISH": "BEARISH", "BEARISH": "BULLISH"}
    conflict = opp.get(r["H4"]) == r["H1"]
    r["has_conflict"] = conflict
    r["m30_conflict"] = opp.get(r["H4"]) == r["M30"]

    h4_reg = regimes["H4"]["regime"]
    if h4_reg == "CHOPPY":
        r["has_conflict"] = True
        r["bias_utama"] = "WAIT (MARKET CHOPPY H4)"
        r["detail"] = f"H4 Choppy: {regimes['H4']['detail']}"
    elif conflict:
        r["bias_utama"] = "WAIT (KONFLIK H4 vs H1)"
        r["detail"] = f"Konflik H4 {r['H4']} vs H1 {r['H1']}"
    else:
        r["bias_utama"] = r["H4"] if r["H4"] in ("BULLISH", "BEARISH") else "NETRAL"
        r["detail"] = f"D1 {r['D1']}, H4 {r['H4']} ({h4_reg}), H1 {r['H1']}, M30 {r['M30']}"
    return r


def _alignment(zone_type: str, ctx: Dict[str, Any]) -> Dict[str, str]:
    want = "BULLISH" if zone_type == "DEMAND" else "BEARISH"
    res = {}
    for tf in ("H4", "H1", "M30"):
        lab = ctx.get(tf)
        res[tf] = "ALIGNED" if lab == want else "NEUTRAL" if lab in ("NETRAL", "NO_DATA") else "OPPOSED"
    return res


def evaluate_m30_momentum(df_m30: Optional[pd.DataFrame], h4_bias: str) -> Dict[str, Any]:
    """
    Validasi mendalam momentum & struktur M30 (TASK Section D):
      1. Keselarasan directional bias H4
      2. Struktur market M30 (BOS/CHoCH)
      3. Kondisi momentum (RSI14, EMA20/EMA50)
      4. Kualitas displacement M30
      5. Kondisi market ranging atau choppy
      6. Hindari entry ketika momentum berlawanan dengan arah setup
    Status output:
      'CONFIRMED' | 'WEAK' | 'CONFLICT' | 'INVALID'
    """
    if df_m30 is None or len(df_m30) < 20:
        return {"status": "INVALID", "ok": False, "msg": "Data M30 tidak mencukupi",
                "detail": "Data kosong / bar kurang", "rsi": 50.0, "regime": "NO_DATA"}

    d30 = df_m30 if "ATR14" in df_m30.columns else tambah_indikator(df_m30.copy())
    last = d30.iloc[-1]
    rsi_raw = float(last["RSI14"]) if "RSI14" in last else 50.0
    rsi = 50.0 if (not math.isfinite(rsi_raw) or math.isnan(rsi_raw)) else rsi_raw
    close = float(last["Close"])
    ema20 = float(last["EMA20"]) if "EMA20" in last else close
    ema50 = float(last["EMA50"]) if "EMA50" in last else close
    atr = float(last["ATR14"]) if ("ATR14" in last and float(last["ATR14"]) > 0) else 2.0

    reg = detect_market_regime(d30)
    regime = reg.get("regime", "UNKNOWN")
    bias = reg.get("bias", "NETRAL")
    events, trend, swings = detect_market_structure(d30)

    want_bias = "BULLISH" if h4_bias == "BULLISH" else ("BEARISH" if h4_bias == "BEARISH" else "NETRAL")
    opp_bias = "BEARISH" if h4_bias == "BULLISH" else ("BULLISH" if h4_bias == "BEARISH" else "NETRAL")

    if want_bias == "NETRAL":
        return {"status": "WEAK", "ok": False, "msg": f"Bias H4 {h4_bias} netral, M30 tidak dapat divalidasi",
                "detail": reg.get("detail", ""), "rsi": rsi, "regime": regime}

    # 1. Deteksi Konflik Struktural Langsung
    if bias == opp_bias or (want_bias == "BULLISH" and regime == "TRENDING_BEARISH") or (want_bias == "BEARISH" and regime == "TRENDING_BULLISH"):
        return {"status": "CONFLICT", "ok": False,
                "msg": f"M30 {bias} ({regime}) bertentangan dengan H4 {h4_bias}",
                "detail": reg.get("detail", ""), "rsi": rsi, "regime": regime}

    # 2. Deteksi Momentum Berlawanan Ekstrem
    if want_bias == "BULLISH":
        if (close < ema20 and ema20 < ema50 and rsi < 42):
            return {"status": "CONFLICT", "ok": False,
                    "msg": f"Momentum M30 berlawanan (RSI {rsi:.1f} < 42, Close < EMA20 < EMA50)",
                    "detail": "Momentum bearish berlawanan", "rsi": rsi, "regime": regime}
        if len(d30) >= 3:
            bear_impulse = max([d30["Open"].iloc[k] - d30["Close"].iloc[k] for k in range(-3, 0)])
            bull_counter = max([d30["Close"].iloc[k] - d30["Open"].iloc[k] for k in range(-3, 0)])
            if bear_impulse >= 1.6 * atr and bull_counter < 0.6 * atr:
                return {"status": "WEAK", "ok": False,
                        "msg": f"Displacement M30 berlawanan ({bear_impulse:.2f} pt >= 1.6 ATR)",
                        "detail": "Displacement bear lawan", "rsi": rsi, "regime": regime}
    else:  # BEARISH
        if (close > ema20 and ema20 > ema50 and rsi > 58):
            return {"status": "CONFLICT", "ok": False,
                    "msg": f"Momentum M30 berlawanan (RSI {rsi:.1f} > 58, Close > EMA20 > EMA50)",
                    "detail": "Momentum bullish berlawanan", "rsi": rsi, "regime": regime}
        if len(d30) >= 3:
            bull_impulse = max([d30["Close"].iloc[k] - d30["Open"].iloc[k] for k in range(-3, 0)])
            bear_counter = max([d30["Open"].iloc[k] - d30["Close"].iloc[k] for k in range(-3, 0)])
            if bull_impulse >= 1.6 * atr and bear_counter < 0.6 * atr:
                return {"status": "WEAK", "ok": False,
                        "msg": f"Displacement M30 berlawanan ({bull_impulse:.2f} pt >= 1.6 ATR)",
                        "detail": "Displacement bull lawan", "rsi": rsi, "regime": regime}

    # 3. Kondisi Market Choppy / Whipsaw
    if regime == "CHOPPY":
        return {"status": "WEAK", "ok": False,
                "msg": f"Market M30 Choppy / Whipsaw ({reg.get('detail', 'whipsaw')})",
                "detail": reg.get("detail", ""), "rsi": rsi, "regime": regime}

    # 4. Kondisi Market Ranging / Momentum Datar / Lemah
    if regime == "RANGING" or bias == "NETRAL":
        if want_bias == "BULLISH":
            if rsi < 46:
                return {"status": "WEAK", "ok": False,
                        "msg": f"Momentum M30 lemah / ranging (RSI {rsi:.1f} < 46)",
                        "detail": "RSI di bawah netral", "rsi": rsi, "regime": regime}
            if close < ema20 - 0.5 * atr:
                return {"status": "WEAK", "ok": False,
                        "msg": f"Momentum M30 di bawah EMA20 ({close:.2f} < {ema20:.2f})",
                        "detail": "Harga tertahan di bawah EMA20", "rsi": rsi, "regime": regime}
        else:
            if rsi > 54:
                return {"status": "WEAK", "ok": False,
                        "msg": f"Momentum M30 lemah / ranging (RSI {rsi:.1f} > 54)",
                        "detail": "RSI di atas netral", "rsi": rsi, "regime": regime}
            if close > ema20 + 0.5 * atr:
                return {"status": "WEAK", "ok": False,
                        "msg": f"Momentum M30 di atas EMA20 ({close:.2f} > {ema20:.2f})",
                        "detail": "Harga tertahan di atas EMA20", "rsi": rsi, "regime": regime}

    # 5. Status Terkonfirmasi (CONFIRMED)
    if want_bias == "BULLISH":
        return {"status": "CONFIRMED", "ok": True,
                "msg": f"M30 BULLISH (RSI {rsi:.1f}, {regime})",
                "detail": reg.get("detail", ""), "rsi": rsi, "regime": regime}
    else:
        return {"status": "CONFIRMED", "ok": True,
                "msg": f"M30 BEARISH (RSI {rsi:.1f}, {regime})",
                "detail": reg.get("detail", ""), "rsi": rsi, "regime": regime}


# =============================================================================
# DATA LIVE (MT5) & CACHE
# =============================================================================
def _load_frames(ttl: float = FRAMES_CACHE_TTL) -> Optional[Dict[str, pd.DataFrame]]:
    """Ambil candle CLOSE dari MT5 (cache singkat). None bila MT5 tidak aktif."""
    now = time.time()
    if _FRAMES_CACHE["data"] is not None and now - _FRAMES_CACHE["ts"] < ttl:
        return _FRAMES_CACHE["data"]
    if not pastikan_mt5_aktif():
        return None
    frames = {}
    for name, (tf, bars) in TIMEFRAMES.items():
        if name == "W1":
            continue
        df = ambil_data(tf, bars)
        frames[name] = tambah_indikator(df) if df is not None and len(df) >= 20 else None
    _FRAMES_CACHE.update(ts=now, data=frames)
    return frames


def _live_context() -> Dict[str, Any]:
    """Harga bid, spread, waktu server, dan status blackout berita (None bila tak tersedia)."""
    ctx = {"price": None, "spread": None, "server_time": None, "news": None, "news_error": None}
    try:
        import MetaTrader5 as mt5
        tick = mt5.symbol_info_tick(SYMBOL)
        if tick and tick.bid > 0:
            ctx.update(price=float(tick.bid), spread=float(tick.ask - tick.bid),
                       server_time=pd.Timestamp(int(tick.time), unit="s"))
    except Exception as e:
        _catat_error("zone_scanner_live_tick", e)
    try:
        from modules.analisis_pasar import berita_blackout_aktif
        ctx["news"] = berita_blackout_aktif() or None
    except Exception as e:
        ctx["news_error"] = repr(e)
        _catat_error("zone_scanner_berita", e)
    return ctx


def _stale_reason(frames: Dict[str, pd.DataFrame], server_time) -> Optional[str]:
    if server_time is None:
        return None
    for tf in ("M5", "M30", "H1", "H4"):
        df = frames.get(tf)
        if df is None or len(df) == 0:
            continue
        age = (pd.Timestamp(server_time) - df.index[-1]).total_seconds() / 60.0
        if age > TF_MINUTES[tf] * 3 + 5:
            return f"Data {tf} terlambat ({age:.0f} menit sejak candle terakhir; market tutup/feed tertinggal)"
    return None


# =============================================================================
# AUDIT TRAIL (JSONL, throttled)
# =============================================================================
def _audit(kind: str, payload: Dict[str, Any], sig: str) -> None:
    """Tulis satu baris JSON; duplikat identik dalam 60 detik dilewati agar tidak memperlambat/membanjiri."""
    try:
        now = time.time()
        last = _AUDIT_LAST.get(kind)
        if last and last[0] == sig and now - last[1] < AUDIT_DEDUP_DETIK:
            return
        _AUDIT_LAST[kind] = (sig, now)
        rec = {"ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "kind": kind, "symbol": SYMBOL, **payload}
        with open(FILE_AUDIT_ZONA, "a", encoding="utf-8") as f:
            f.write(json.dumps(_clean(rec), ensure_ascii=False) + "\n")
    except Exception as e:
        _catat_error("zone_scanner_audit", e)


def _sig(*parts) -> str:
    return hashlib.md5(json.dumps(_clean(list(parts)), sort_keys=True, default=str).encode()).hexdigest()


# =============================================================================
# RADAR ZONE DISTANCE & PRIORITY SYSTEM
# =============================================================================
class ZoneDistanceResult(NamedTuple):
    distance: float
    status: str


def calculate_zone_distance(
    zone_low: float,
    zone_high: float,
    current_price: float,
    digits: int = 2,
) -> ZoneDistanceResult:
    """
    Kalkulasi jarak harga snapshot terhadap batas zona dengan presisi digits:

    1. Zona di atas harga (current_price < zone_low):
       Distance = round(zone_low - current_price, digits)
       status = "ABOVE_PRICE"

    2. Zona di bawah harga (current_price > zone_high):
       Distance = round(current_price - zone_high, digits)
       status = "BELOW_PRICE"

    3. Harga di dalam zona (zone_low <= current_price <= zone_high):
       Distance = 0.0
       status = "INSIDE_ZONE"

    Returns:
        ZoneDistanceResult(distance, status)
    """
    eps = 1e-7
    if current_price < zone_low - eps:
        dist = round(zone_low - current_price, digits)
        status = "ABOVE_PRICE"
    elif current_price > zone_high + eps:
        dist = round(current_price - zone_high, digits)
        status = "BELOW_PRICE"
    else:
        dist = 0.0
        status = "INSIDE_ZONE"
    return ZoneDistanceResult(dist, status)


def calculate_zone_approach(
    distance: float,
    status: str,
    atr: float = 2.0,
) -> str:
    """
    Menentukan status pendekatan harga terhadap zona:
    - Jika status == "INSIDE_ZONE" atau distance == 0.0: "TOUCHED"
    - Jika distance <= max(3.0, 1.5 * atr): "APPROACHING"
    - Jika distance > max(35.0, 15.0 * atr): "FAR"
    - Selainnya: "UNTOUCHED"
    """
    if status == "INSIDE_ZONE" or distance == 0.0:
        return "TOUCHED"
    atr_val = atr if (isinstance(atr, (int, float)) and math.isfinite(atr) and atr > 0) else 2.0
    far_threshold = max(35.0, 15.0 * atr_val)
    approach_threshold = max(3.0, 1.5 * atr_val)
    if distance <= approach_threshold:
        return "APPROACHING"
    if distance > far_threshold:
        return "FAR"
    return "UNTOUCHED"


def prioritize_radar_zones(
    radar_zones: List[Union[Zone, Dict[str, Any]]],
    current_price: float,
    htf_bias: str = "NETRAL",
    h1_structure: str = "NETRAL",
    atr: float = 2.0,
    m5_stages: Optional[Dict[str, Any]] = None,
    digits: int = 2,
) -> Dict[str, Any]:
    """
    Sistem Prioritas Zona RADAR dengan 3 Kategori Terpisah:
      A. PRIMARY DIRECTIONAL ZONE: Zona kualitas terbaik selaras bias H4/H1.
      B. NEAREST RADAR ZONE: Zona valid terdekat dengan harga saat ini (tanpa memandang arah).
      C. BEST ENTRY MONITORING ZONE: Zona paling layak dipantau untuk setup berdasarkan gabungan
         HTF alignment, grade/score, OB/FVG, mitigasi, touch count, jarak, dan R:R.
    
    Ketentuan:
      - Zona yang dekat tetapi bertentangan dengan bias utama TIDAK otomatis menjadi prioritas entry (hanya support/reaction).
      - Zona yang jauh tetap ditampilkan sebagai RADAR, tetapi diberi label FAR.
      - Grade A/B tidak otomatis berarti READY atau CONFIRMED.
    """
    if not radar_zones:
        return {
            "primary_directional_zone": None,
            "nearest_radar_zone": None,
            "best_entry_monitoring_zone": None,
            "primary_zone": None,
            "secondary_zone": None,
            "countertrend_zone": None,
            "all_prioritized": [],
            "aligned_zones": [],
            "counter_zones": [],
            "primary_reason": "NO RADAR ZONE DETECTED.",
            "distance_to_zone_str": "N/A",
            "missing_conditions": ["NO RADAR ZONE DETECTED."],
            "dynamic_wait_reason": "NO RADAR ZONE DETECTED.",
        }

    atr_val = atr if (isinstance(atr, (int, float)) and math.isfinite(atr) and atr > 0) else 2.0
    far_threshold = max(35.0, 15.0 * atr_val)
    approach_threshold = max(3.0, 1.5 * atr_val)

    processed: List[Dict[str, Any]] = []
    for z in radar_zones:
        zd = z.to_dict() if hasattr(z, "to_dict") else dict(z)
        z_type = str(zd.get("zone_type", "")).upper()
        lo = float(zd.get("price_low", 0.0))
        hi = float(zd.get("price_high", 0.0))
        tf = str(zd.get("timeframe", "H1"))
        grd = str(zd.get("grade", "B"))
        score = float(zd.get("score", 60.0))
        fresh = str(zd.get("freshness", "FRESH")).upper()
        status = str(zd.get("status", "RADAR"))

        # 1. Jarak & Status Pendekatan menggunakan snapshot formula standar
        dist_res = calculate_zone_distance(lo, hi, current_price, digits=digits)
        dist = dist_res.distance
        dir_status = dist_res.status
        approach = calculate_zone_approach(dist, dir_status, atr=atr_val)

        # 2. Keselarasan Arah HTF
        if htf_bias == "BULLISH":
            is_aligned = (z_type == "DEMAND")
            is_counter = (z_type == "SUPPLY")
        elif htf_bias == "BEARISH":
            is_aligned = (z_type == "SUPPLY")
            is_counter = (z_type == "DEMAND")
        else:  # NETRAL
            is_aligned = False
            is_counter = False

        # 3. Skor Prioritas Komprehensif
        p_score = score
        if is_aligned:
            p_score += 50.0
            if h1_structure == htf_bias:
                p_score += 15.0
        elif is_counter:
            p_score -= 80.0  # Penalti drastis: zona countertrend dilarang menjadi entry pro-trend
        else:
            p_score += 5.0

        if fresh == "FRESH":
            p_score += 10.0
        elif fresh == "TESTED_1X":
            p_score += 5.0
        elif "INVALID" in fresh:
            p_score -= 200.0

        if approach == "TOUCHED":
            p_score += 15.0
        elif approach == "APPROACHING":
            p_score += 8.0
        elif approach == "FAR":
            p_score -= 25.0
        else:
            p_score -= min(15.0, dist * 0.15)

        if grd == "A+":
            p_score += 10.0
        elif grd == "A":
            p_score += 5.0

        # Label prioritas
        labels = [status]
        if approach == "FAR":
            labels.append("FAR")
        if is_counter:
            labels.append("COUNTERTREND")

        zd["score"] = score
        zd["grade"] = grd
        zd["freshness"] = fresh
        zd["timeframe"] = tf
        zd["zone_type"] = z_type
        zd["price_low"] = lo
        zd["price_high"] = hi
        zd["priority_score"] = round(p_score, 1)
        zd["distance"] = dist
        zd["direction_status"] = dir_status
        zd["distance_status"] = dir_status
        zd["approach"] = approach
        zd["is_aligned"] = is_aligned
        zd["is_counter"] = is_counter
        zd["priority_labels"] = labels
        processed.append(zd)

    # 4. Pemisahan 3 Kategori Prioritas:
    # A. PRIMARY DIRECTIONAL ZONE: Kualitas terbaik selaras bias HTF
    # B. NEAREST RADAR ZONE: Jarak terdekat ke snapshot harga (apa pun arahnya)
    # C. BEST ENTRY MONITORING ZONE: Paling layak dipantau untuk aksi setup pro-trend
    if htf_bias in ("BULLISH", "BEARISH"):
        aligned = [z for z in processed if z["is_aligned"]]
        counter = [z for z in processed if z["is_counter"]]
        aligned.sort(key=lambda x: -x["priority_score"])
        counter.sort(key=lambda x: x["distance"])  # Diurutkan dari jarak terdekat
        primary_directional = aligned[0] if aligned else None
        secondary_aligned = aligned[1] if len(aligned) > 1 else None
        countertrend_nearest = counter[0] if counter else None
        primary_fallback = primary_directional
    else:  # NETRAL
        processed.sort(key=lambda x: -x["priority_score"])
        demands = [z for z in processed if z["zone_type"] == "DEMAND"]
        supplies = [z for z in processed if z["zone_type"] == "SUPPLY"]
        demands.sort(key=lambda x: x["distance"])
        supplies.sort(key=lambda x: x["distance"])
        primary_directional = None  # Pada HTF Netral, tidak ada directional bias
        primary_fallback = supplies[0] if supplies else (demands[0] if demands else None)
        secondary_aligned = demands[0] if (supplies and demands) else (supplies[1] if len(supplies) > 1 else None)
        countertrend_nearest = None
        aligned = processed
        counter = []

    # B. NEAREST RADAR ZONE (Zona valid dengan jarak terkecil)
    valid_cands = [z for z in processed if "INVALID" not in z.get("freshness", "")]
    if valid_cands:
        sorted_by_dist = sorted(valid_cands, key=lambda x: x["distance"])
        nearest_radar = sorted_by_dist[0]
    else:
        nearest_radar = None

    # C. BEST ENTRY MONITORING ZONE (Optimal pro-trend setup)
    if htf_bias in ("BULLISH", "BEARISH") and aligned:
        def _calc_monitoring_score(z: Dict[str, Any]) -> float:
            m_score = float(z.get("score", 60.0))
            if z.get("is_aligned"):
                m_score += 25.0
            fresh_val = z.get("freshness", "FRESH")
            if fresh_val == "FRESH":
                m_score += 15.0
            elif fresh_val == "TESTED_1X":
                m_score += 8.0
            elif fresh_val == "TESTED_2X":
                m_score += 2.0
            elif "MITIGATED" in fresh_val or "INVALID" in fresh_val:
                return -999.0
            tc = int(z.get("touch_count", 0))
            if tc > 2:
                return -999.0
            elif tc == 1:
                m_score -= 3.0
            elif tc == 2:
                m_score -= 8.0
            app = z.get("approach", "UNTOUCHED")
            d = float(z.get("distance", 0.0))
            if app == "TOUCHED" or d == 0.0:
                m_score += 20.0
            elif app == "APPROACHING":
                m_score += 15.0
            elif app == "FAR":
                m_score -= 25.0
            else:
                m_score -= min(15.0, d * 0.2)
            w = float(z.get("price_high", 0.0)) - float(z.get("price_low", 0.0))
            if 1.0 <= w <= 8.0:
                m_score += 5.0
            elif w > 15.0:
                m_score -= 10.0
            return m_score

        aligned_valid = [z for z in aligned if "INVALID" not in z.get("freshness", "") and "MITIGATED" not in z.get("freshness", "")]
        best_entry_monitoring = max(aligned_valid, key=_calc_monitoring_score) if aligned_valid else None
    else:
        best_entry_monitoring = None

    primary = primary_directional if primary_directional is not None else primary_fallback
    secondary = secondary_aligned
    countertrend = countertrend_nearest

    # Alasan Prioritas Zona Utama
    if primary_directional:
        align_str = f"searah bias {htf_bias}"
        fvg_st = (primary_directional.get("fvg_confirmation") or {}).get("status", "-")
        app_lbl = f" [{primary_directional['approach']}]" if primary_directional['approach'] in ("FAR", "TOUCHED", "APPROACHING") else ""
        primary_reason = (
            f"{primary_directional['timeframe']} {primary_directional['zone_type']} ({primary_directional['price_low']:.2f}-{primary_directional['price_high']:.2f}){app_lbl} "
            f"diprioritaskan karena {align_str}, Grade {primary_directional['grade']} (Skor {primary_directional['score']:.1f}), {primary_directional['freshness']}, FVG {fvg_st}."
        )
    elif htf_bias == "NETRAL" and primary:
        fvg_st = (primary.get("fvg_confirmation") or {}).get("status", "-")
        primary_reason = f"{primary['timeframe']} {primary['zone_type']} ({primary['price_low']:.2f}-{primary['price_high']:.2f}) dipantau sebagai batas range reaksi (bias NETRAL, FVG {fvg_st})."
    else:
        primary_reason = f"Tidak ada zona aktif yang selaras dengan arah bias {htf_bias}."

    # Jarak ke Setiap Zona Relevan
    dist_parts = []
    if primary_directional:
        p_lbl = "Primary " + primary_directional["zone_type"].capitalize()
        dist_parts.append(f"{primary_directional['distance']:.2f} pt ke {p_lbl} ({primary_directional['price_low']:.2f}-{primary_directional['price_high']:.2f} [{primary_directional['approach']}])")
    elif primary:
        p_lbl = "Primary " + primary["zone_type"].capitalize()
        dist_parts.append(f"{primary['distance']:.2f} pt ke {p_lbl} ({primary['price_low']:.2f}-{primary['price_high']:.2f} [{primary['approach']}])")
    if secondary:
        s_lbl = "Secondary " + secondary["zone_type"].capitalize()
        dist_parts.append(f"{secondary['distance']:.2f} pt ke {s_lbl} ({secondary['price_low']:.2f}-{secondary['price_high']:.2f} [{secondary['approach']}])")
    if countertrend:
        ct_lbl = "Countertrend " + countertrend["zone_type"].capitalize()
        dist_parts.append(f"{countertrend['distance']:.2f} pt ke {ct_lbl} ({countertrend['price_low']:.2f}-{countertrend['price_high']:.2f} [{countertrend['approach']}])")
    dist_str = "; ".join(dist_parts) if dist_parts else "N/A"

    # Missing Entry Conditions
    missing: List[str] = []
    eval_target = primary_directional if primary_directional is not None else (primary if htf_bias != "NETRAL" else None)
    if eval_target:
        st_zt = (m5_stages or {}).get("zone_touch", {}).get("status")
        st_sw = (m5_stages or {}).get("liquidity_sweep", {}).get("status")
        st_tr = (m5_stages or {}).get("structure_trigger", {}).get("status")
        st_dp = (m5_stages or {}).get("displacement", {}).get("status")
        st_rr = (m5_stages or {}).get("risk_reward", {}).get("status")

        if eval_target["approach"] == "TOUCHED" or st_zt == "CONFIRMED":
            pass
        else:
            missing.append(f"[ZONE_TOUCH]: Harga belum mencapai atau menyentuh {eval_target['zone_type'].capitalize()} ({eval_target['price_low']:.2f}-{eval_target['price_high']:.2f}, jarak {eval_target['distance']:.2f} pt)")

        if st_sw == "CONFIRMED":
            pass
        elif st_sw == "WAIT" and "jauh" in (m5_stages or {}).get("liquidity_sweep", {}).get("msg", "").lower():
            missing.append("[LIQUIDITY_SWEEP]: Sweep likuiditas M5 terjadi terlalu jauh dari area zona")
        else:
            want_sw = "Buy-Side Liquidity (EQH/Swing High)" if eval_target["zone_type"] == "SUPPLY" else "Sell-Side Liquidity (EQL/Swing Low)"
            missing.append(f"[LIQUIDITY_SWEEP]: Belum ada validasi sweep {want_sw} M5 di area zona")

        if st_tr == "CONFIRMED":
            pass
        else:
            want_tr = "CHoCH/BOS/CISD bearish" if eval_target["zone_type"] == "SUPPLY" else "CHoCH/BOS/CISD bullish"
            missing.append(f"[STRUCTURE_TRIGGER]: Belum ada konfirmasi perubahan struktur {want_tr} M5")

        if st_dp == "CONFIRMED":
            pass
        elif st_dp in ("WEAK", "WAIT"):
            dp_m = (m5_stages or {}).get("displacement", {}).get("msg", "Displacement lemah")
            missing.append(f"[DISPLACEMENT]: {dp_m}")
        else:
            missing.append("[DISPLACEMENT]: Belum ada konfirmasi dorongan displacement M5 searah")

        # RISK/REWARD UNTUK CONFIRMED ENTRY:
        # Pisahkan dari Early Pending RR!
        # Hanya dievaluasi jika pemicu struktur & displacement M5 sudah terkonfirmasi.
        # Sebelum trigger lengkap, statusnya adalah NOT_CHECKED (bukan FAILED/NOT PASSED).
        m5_triggers_ready = (st_tr == "CONFIRMED" and st_dp == "CONFIRMED")
        if st_rr == "CONFIRMED":
            confirmed_entry_rr_status = "PASSED"
        elif m5_triggers_ready:
            if st_rr in ("FAILED", "WAIT"):
                confirmed_entry_rr_status = "FAILED"
                missing.append("[CONFIRMED_ENTRY_RISK_REWARD]: Entry konfirmasi M5 tidak memenuhi Risk/Reward minimum 1:1.5")
            else:
                confirmed_entry_rr_status = "WAIT"
        else:
            confirmed_entry_rr_status = "NOT_CHECKED"
            # Jangan menambahkan pesan [RISK_REWARD] ketika M5 trigger belum lengkap!
    else:
        confirmed_entry_rr_status = "NOT_CHECKED"
        missing.append(f"Tidak ada zona yang selaras dengan arah bias HTF ({htf_bias})")

    # Alasan Dinamis Final WAIT
    if htf_bias == "NETRAL":
        dynamic_wait = f"WAIT: Bias H4 NETRAL. Belum ada directional bias yang jelas. Zona Supply dan Demand dipantau sebagai batas range reaksi, bukan sinyal entry."
    elif eval_target:
        p_type = eval_target["zone_type"]
        p_name = f"{eval_target['timeframe']} {p_type}"
        p_range = f"{eval_target['price_low']:.2f}-{eval_target['price_high']:.2f}"
        p_dist = eval_target["distance"]
        p_app = eval_target["approach"]

        ct_note = ""
        if countertrend:
            ct_type = countertrend["zone_type"]
            ct_range = f"{countertrend['price_low']:.2f}-{countertrend['price_high']:.2f}"
            opp_sig = "BUY" if ct_type == "DEMAND" else "SELL"
            ct_role = "area reaksi/support" if ct_type == "DEMAND" else "area reaksi/resistance"
            ct_note = f" {ct_type.capitalize()} di bawah harga ({ct_range}) hanya menjadi {ct_role}, bukan sinyal {opp_sig}."

        if p_app == "TOUCHED":
            st_sw = (m5_stages or {}).get("liquidity_sweep", {}).get("status")
            st_tr = (m5_stages or {}).get("structure_trigger", {}).get("status")
            st_dp = (m5_stages or {}).get("displacement", {}).get("status")
            if st_sw != "CONFIRMED":
                sw_msg = (m5_stages or {}).get("liquidity_sweep", {}).get("msg", "")
                if "jauh" in sw_msg.lower():
                    dynamic_wait = f"WAIT: Harga telah menyentuh {p_name} ({p_range}), namun Liquidity Sweep M5 terjadi tidak relevan (terlalu jauh dari area zona). Menunggu sweep yang valid.{ct_note}"
                else:
                    dynamic_wait = f"WAIT: Harga telah memasuki {p_name} ({p_range}), tetapi belum ada konfirmasi Liquidity Sweep M5 yang valid di area zona. Menunggu sweep likuiditas dan konfirmasi struktur M5.{ct_note}"
            elif st_tr != "CONFIRMED":
                dynamic_wait = f"WAIT: Harga telah menyentuh {p_name} dan membentuk Liquidity Sweep M5, tetapi belum terkonfirmasi pembalikan/kelanjutan struktur (CHoCH/BOS M5). Menunggu penembusan struktur.{ct_note}"
            elif st_dp != "CONFIRMED":
                dynamic_wait = f"WAIT: Sweep dan CHoCH M5 terkonfirmasi pada {p_name}, namun displacement M5 masih lemah. Menunggu dorongan momentum yang memadai.{ct_note}"
            else:
                dynamic_wait = f"WAIT: Konfirmasi trigger M5 pada {p_name} terpenuhi, tetapi kalkulasi Risk/Reward belum mencapai minimum 1:1.5.{ct_note}"
        else:
            app_str = "sudah mendekati" if p_app == "APPROACHING" else ("berjarak jauh dari" if p_app == "FAR" else "belum mencapai")
            dynamic_wait = f"WAIT: Bias H4/H1 {htf_bias.lower()}, tetapi harga {app_str} {p_type.capitalize()} RADAR ({p_name} @ {p_range}, jarak {p_dist:.2f} pt). Belum ada validasi sweep dan struktur M5.{ct_note}"
    else:
        if countertrend:
            ct_type = countertrend["zone_type"]
            ct_range = f"{countertrend['price_low']:.2f}-{countertrend['price_high']:.2f}"
            opp_sig = "BUY" if ct_type == "DEMAND" else "SELL"
            dynamic_wait = f"WAIT: Bias H4/H1 {htf_bias.lower()}, tidak ada zona yang selaras dengan bias utama. Zona {ct_type.capitalize()} di {ct_range} hanya area reaksi/support, bukan sinyal {opp_sig}."
        else:
            dynamic_wait = f"WAIT: Bias H4/H1 {htf_bias.lower()}, tidak ditemukan zona aktif yang memenuhi syarat."

    return {
        "primary_directional_zone": primary_directional,
        "nearest_radar_zone": nearest_radar,
        "best_entry_monitoring_zone": best_entry_monitoring,
        "primary_zone": primary_directional if primary_directional is not None else (None if htf_bias in ("BULLISH", "BEARISH") else primary),
        "secondary_zone": secondary,
        "countertrend_zone": countertrend,
        "all_prioritized": processed,
        "aligned_zones": aligned,
        "counter_zones": counter,
        "primary_reason": primary_reason,
        "distance_to_zone_str": dist_str,
        "missing_conditions": missing,
        "dynamic_wait_reason": dynamic_wait,
        "confirmed_entry_rr": confirmed_entry_rr_status,
    }


# =============================================================================
# SCANNER UTAMA
# =============================================================================
def scan_xauusd_zones(frames: Optional[Dict[str, pd.DataFrame]] = None, current_price: Optional[float] = None,
                      symbol: str = SYMBOL) -> Dict[str, Any]:
    """
    Pindai zona H4 & H1 (OB + konfirmasi FVG + sweep), pisahkan deteksi vs validasi,
    skor, gabung, dan kembalikan hasil terstruktur ke dalam 3 kategori:
      1. valid_zones (active_zones): Lolos hard validation, skor >= 60, belum invalid, jarak wajar.
      2. radar_zones: Seluruh zona reaksi terdeteksi yang utuh struktural (termasuk zona jauh & saat konflik HTF).
      3. rejected_zones: Seluruh zona yang gagal validasi beserta alasan spesifik.
    Raise-safe untuk data buruk (mengembalikan data_ok=False dengan empty_reason).
    """
    out: Dict[str, Any] = {
        "symbol": symbol, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "current_price": current_price,
        "valid_zones": [], "active_zones": [], "radar_zones": [], "detected_zones": [], "rejected_zones": [],
        "best_demand": None, "best_supply": None, "htf_context": {}, "empty_reason": "", "summary_text": "",
        "diagnostics": {}, "diagnostic_text": "", "data_ok": False,
    }
    if frames is None:
        frames = _load_frames()
        if frames is None:
            out["empty_reason"] = "MT5 tidak aktif; data tidak dapat diambil"
            out["summary_text"] = "MT5 offline - scanner tidak berjalan (WAIT)."
            return out

    need = 2 * SWING_N + 20
    for tf in ("H4", "H1"):
        df_tf = frames.get(tf)
        if df_tf is None or len(df_tf) < need:
            out["empty_reason"] = f"Data {tf} tidak mencukupi (minimal {need} candle)"
            out["summary_text"] = f"Data {tf} belum cukup (WAIT)."
            return out
        # Cek integritas data OHLC & ATR
        for col in ("Open", "High", "Low", "Close"):
            if col not in df_tf.columns or not np.isfinite(df_tf[col].values).all():
                out["empty_reason"] = f"Data {tf} korup (terdapat nilai non-finite pada kolom {col})"
                out["summary_text"] = f"Data {tf} tidak valid (WAIT)."
                return out
        if ((df_tf["High"] < df_tf["Low"]).any() or (df_tf["High"] < df_tf["Open"]).any() or
                (df_tf["High"] < df_tf["Close"]).any() or (df_tf["Low"] > df_tf["Open"]).any() or
                (df_tf["Low"] > df_tf["Close"]).any()):
            out["empty_reason"] = f"Data {tf} korup (struktur OHLC tidak konsisten High/Low)"
            out["summary_text"] = f"Data {tf} tidak valid (WAIT)."
            return out
        if "ATR14" in df_tf.columns:
            atr_chk = float(df_tf["ATR14"].iloc[-1])
            if not math.isfinite(atr_chk) or atr_chk <= 0:
                out["empty_reason"] = f"ATR {tf} tidak valid ({atr_chk})"
                out["summary_text"] = f"ATR {tf} tidak valid (WAIT)."
                return out

    if current_price is None:
        ref = frames.get("M5") if frames.get("M5") is not None and len(frames["M5"]) else frames["H1"]
        current_price = float(ref["Close"].iloc[-1])
    if not (isinstance(current_price, (int, float)) and math.isfinite(current_price) and current_price > 0):
        out["empty_reason"] = "Harga referensi tidak valid"
        out["summary_text"] = "Harga tidak valid (WAIT)."
        return out

    snapshot_price = round(float(current_price), 2)
    snapshot_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S WIB")
    out["current_price"] = snapshot_price
    out["snapshot_price"] = snapshot_price
    out["snapshot_time"] = snapshot_time
    out["bid"] = None
    out["ask"] = None
    try:
        import MetaTrader5 as mt5
        tick = mt5.symbol_info_tick(symbol)
        if tick and getattr(tick, "bid", 0) > 0:
            out["bid"] = round(float(tick.bid), 2)
            out["ask"] = round(float(tick.ask), 2)
    except Exception:
        pass
    if out.get("bid") is not None and out.get("bid") > 0:
        out["price_source"] = f"MT5 Live Tick (Bid: {out['bid']:.2f} | Ask: {out['ask']:.2f}) — Presisi Jarak terhadap Proximal Edge"
    else:
        out["price_source"] = f"Candle Close Snapshot ({snapshot_price:.2f})"
    out["data_ok"] = True

    ck = _sig([(tf, len(frames[tf]), str(frames[tf].index[-1]), float(frames[tf]["Close"].iloc[-1]))
               for tf in ("H4", "H1", "M30", "M5") if frames.get(tf) is not None and len(frames[tf])],
              round(float(snapshot_price), 1))
    if _SCAN_CACHE["key"] == ck and time.time() - _SCAN_CACHE["ts"] < SCAN_CACHE_TTL:
        return _SCAN_CACHE["data"]

    ctx = evaluate_htf_context(frames)
    out["htf_context"] = ctx

    valid_all: List[Zone] = []
    radar_all: List[Zone] = []
    rejected: List[Zone] = []

    # Diagnostik tracker
    total_candles = 0
    total_swings = 0
    total_swings_h = 0
    total_swings_l = 0
    total_bos_choch = 0
    total_ob_inspected = 0
    total_fvg = 0
    total_hard_pass = 0
    total_hard_reject = 0
    total_invalidated = 0
    total_mitigated = 0
    total_too_far = 0
    total_scored_pass = 0
    rejection_reasons_counter: Dict[str, int] = {}
    tf_diag: Dict[str, Any] = {}
    inspected_zones_log: List[Dict[str, Any]] = []

    tfs_to_scan = ["H4", "H1"]
    if frames.get("M30") is not None and len(frames["M30"]) >= need:
        df_m30 = frames["M30"]
        m30_valid = True
        for col in ("Open", "High", "Low", "Close"):
            if col not in df_m30.columns or not np.isfinite(df_m30[col].values).all():
                m30_valid = False
                break
        if m30_valid and not ((df_m30["High"] < df_m30["Low"]).any() or (df_m30["High"] < df_m30["Open"]).any()):
            tfs_to_scan.append("M30")

    for tf in tfs_to_scan:
        df = frames[tf] if "ATR14" in frames[tf].columns else tambah_indikator(frames[tf].copy())
        events, _, swings = detect_market_structure(df)
        sweeps = detect_liquidity_sweeps(df, swings, lookback=None)
        fvgs = detect_fvgs(df)
        ok, bad = detect_order_blocks(df, events, fvgs, sweeps, tf, max_events=25)

        n_bars = len(df)
        total_candles += n_bars
        n_sw = len(swings)
        total_swings += n_sw
        n_sw_h = sum(1 for s in swings if s["type"] == "H")
        n_sw_l = sum(1 for s in swings if s["type"] == "L")
        total_swings_h += n_sw_h
        total_swings_l += n_sw_l
        n_ev = len(events)
        total_bos_choch += n_ev
        n_fvg = len(fvgs)
        total_fvg += n_fvg
        n_ob_cands = len(ok) + len(bad)
        total_ob_inspected += n_ob_cands
        total_hard_pass += len(ok)
        total_hard_reject += len(bad)

        tf_diag[tf] = {
            "bars": n_bars, "swings": n_sw, "events": n_ev, "fvgs": n_fvg,
            "ob_candidates": n_ob_cands, "hard_pass": len(ok), "hard_reject": len(bad),
        }

        atr_now = float(df["ATR14"].iloc[-1]) if ("ATR14" in df.columns and float(df["ATR14"].iloc[-1]) > 0) else 2.0

        # Evaluasi kandidat yang lolos hard validation
        for z in ok:
            score_zone(z, htf_alignment=_alignment(z.zone_type, ctx))
            dist_res = calculate_zone_distance(z.price_low, z.price_high, snapshot_price, digits=2)
            dist_val = dist_res.distance
            dir_st = dist_res.status
            z.distance = dist_val
            z.direction_status = dir_st
            z.approach = calculate_zone_approach(dist_val, dir_st, atr=atr_now)

            if z.score >= SCORE_MIN_ACTIVE:
                total_scored_pass += 1
                z.status = "RADAR"
                if dist_val <= MAX_ZONE_DISTANCE_ATR * atr_now:
                    valid_all.append(z)
                    radar_all.append(z)
                    rej_desc = "PASSED_VALID_ACTIVE"
                else:
                    total_too_far += 1
                    rej_desc = f"Zona radar HTF jauh dari harga aktual ({dist_val:.1f} pt > {MAX_ZONE_DISTANCE_ATR * atr_now:.1f} pt)"
                    z.audit.append(rej_desc)
                    radar_all.append(z)
            else:
                z.is_valid = False
                z.grade = "REJECTED"
                z.status = "INVALIDATED" if ("invalid" in (z.rejection_reason or "").lower()) else "RADAR"
                rej_key = f"Skor {z.score:.1f} < {SCORE_MIN_ACTIVE}"
                rejection_reasons_counter[rej_key] = rejection_reasons_counter.get(rej_key, 0) + 1
                rej_desc = rej_key
                rejected.append(z)

            inspected_zones_log.append({
                "source_timeframe": z.timeframe,
                "zone_id": z.zone_id,
                "zone_type": z.zone_type,
                "price_low": z.price_low,
                "price_high": z.price_high,
                "grade": z.grade,
                "score": z.score,
                "ob_type": (z.origin_bos or {}).get("tipe", "OB"),
                "fvg_status": (z.fvg_confirmation or {}).get("status", "NONE"),
                "mitigation_status": z.freshness,
                "touch_count": z.touch_count,
                "invalidation_level": z.invalidation_price,
                "distance": dist_val,
                "direction_status": dir_st,
                "rejection_reason": rej_desc if not z.is_valid else ("TOO_FAR" if dist_val > MAX_ZONE_DISTANCE_ATR * atr_now else "PASSED"),
                "is_valid": z.is_valid,
                "status": z.status,
            })

        # Evaluasi kandidat yang gagal hard validation
        for z in bad:
            dist_res = calculate_zone_distance(z.price_low, z.price_high, snapshot_price, digits=2)
            dist_val = dist_res.distance
            dir_st = dist_res.status
            z.distance = dist_val
            z.direction_status = dir_st
            z.approach = calculate_zone_approach(dist_val, dir_st, atr=atr_now)

            rej = z.rejection_reason or "Gagal hard validation"
            for part in rej.split("; "):
                part_clean = part.split("(")[0].strip()
                rejection_reasons_counter[part_clean] = rejection_reasons_counter.get(part_clean, 0) + 1

            if "Zona invalid" in rej:
                total_invalidated += 1
                z.status = "INVALIDATED"
            if "Zona sudah dimitigasi" in rej:
                total_mitigated += 1

            # Kategori A: Radar Zone untuk zona yang utuh secara struktural
            is_inval = (z.freshness == "INVALIDATED" or "Zona invalid" in rej)
            is_mit = (z.touch_count > MAX_TOUCH_COUNT or "Zona sudah dimitigasi" in rej)
            origin_ok = z.origin_bos.get("origin_valid", True)
            if (not is_inval) and (not is_mit) and origin_ok:
                z.grade = "RADAR"
                z.status = "RADAR"
                z.audit.append("Kandidat zona reaksi (RADAR SMC)")
                radar_all.append(z)

            rejected.append(z)

            inspected_zones_log.append({
                "source_timeframe": z.timeframe,
                "zone_id": z.zone_id,
                "zone_type": z.zone_type,
                "price_low": z.price_low,
                "price_high": z.price_high,
                "grade": z.grade,
                "score": z.score,
                "ob_type": (z.origin_bos or {}).get("tipe", "OB"),
                "fvg_status": (z.fvg_confirmation or {}).get("status", "NONE"),
                "mitigation_status": z.freshness,
                "touch_count": z.touch_count,
                "invalidation_level": z.invalidation_price,
                "distance": dist_val,
                "direction_status": dir_st,
                "rejection_reason": rej,
                "is_valid": False,
                "status": z.status,
            })

    active = merge_overlapping_zones(valid_all)
    radar_zones = merge_overlapping_zones(radar_all)

    def _zone_sort_key(z: Zone) -> Tuple:
        rank = {"A+": 0, "A": 1, "B": 2, "RADAR": 3}
        tf_rank = 0 if ("H4" in z.timeframe and "H1" in z.timeframe) else (1 if "H4" in z.timeframe else (2 if "H1" in z.timeframe else 3))
        fresh_rank = {"FRESH": 0, "TESTED_1X": 1, "TESTED_2X": 2}.get(z.freshness, 9)
        return (
            rank.get(z.grade, 9),
            -round(float(z.score), 1),
            tf_rank,
            fresh_rank,
            -round(float(z.displacement_strength), 2),
            str(z.zone_id),
        )

    active.sort(key=_zone_sort_key)
    radar_zones.sort(key=_zone_sort_key)

    # Penanganan Konflik HTF (Requirement 4):
    # Jangan hilangkan zona radar! Beri label [CONFLICT H4 vs H1] dan pertahankan di radar.
    if ctx["has_conflict"]:
        for z in active:
            z.audit.append("Status: Terpantau [CONFLICT H4 vs H1]")
        for z in radar_zones:
            z.audit.append("Status: Terpantau [CONFLICT H4 vs H1]")

    # Bangun dictionary diagnostik lengkap
    diagnostics = {
        "symbol": symbol,
        "candles_checked": total_candles,
        "swings_found": total_swings,
        "swings_high": total_swings_h,
        "swings_low": total_swings_l,
        "bos_choch_count": total_bos_choch,
        "ob_candidates_count": total_ob_inspected,
        "fvg_count": total_fvg,
        "zones_detected": len(radar_all) + len(rejected),
        "zones_passed_hard_validation": total_hard_pass,
        "zones_rejected": len(rejected),
        "zones_invalidated": total_invalidated,
        "zones_mitigated": total_mitigated,
        "zones_too_far": total_too_far,
        "zones_passed_scoring": total_scored_pass,
        "zones_valid_active": len(active),
        "zones_radar": len(radar_zones),
        "rejected_reasons_breakdown": rejection_reasons_counter,
        "htf_conflict": ctx.get("has_conflict", False),
        "by_timeframe": tf_diag,
        "inspected_zones_log": inspected_zones_log,
    }
    out["diagnostics"] = diagnostics

    ref_atr = float(frames["H1"]["ATR14"].iloc[-1]) if ("H1" in frames and frames["H1"] is not None and "ATR14" in frames["H1"].columns and float(frames["H1"]["ATR14"].iloc[-1]) > 0) else 2.0
    radar_prio = prioritize_radar_zones(
        radar_zones=radar_zones,
        current_price=snapshot_price,
        htf_bias=ctx.get("H4", "NETRAL"),
        h1_structure=ctx.get("H1", "NETRAL"),
        atr=ref_atr,
        digits=2,
    )
    out["radar_priority"] = radar_prio

    # Sinkronisasi atribut prioritas ke objek Zone
    p_map = {z_d["zone_id"]: z_d for z_d in radar_prio.get("all_prioritized", [])}
    for z in radar_zones:
        if z.zone_id in p_map:
            p_item = p_map[z.zone_id]
            z.distance = p_item["distance"]
            z.direction_status = p_item.get("direction_status", "")
            z.approach = p_item["approach"]
            z.is_aligned = p_item["is_aligned"]
            z.priority_score = p_item["priority_score"]
            z.priority_labels = p_item["priority_labels"]

    out["valid_zones"] = [z.to_dict() for z in active]
    out["active_zones"] = [z.to_dict() for z in active]
    out["radar_zones"] = [z.to_dict() for z in radar_zones]
    out["detected_zones"] = [z.to_dict() for z in (radar_zones + rejected)]
    out["rejected_zones"] = [z.to_dict() for z in rejected]
    out["best_demand"] = next((z.to_dict() for z in active if z.zone_type == "DEMAND"), None)
    out["best_supply"] = next((z.to_dict() for z in active if z.zone_type == "SUPPLY"), None)

    if not active:
        if ctx["has_conflict"]:
            out["empty_reason"] = "Konflik arah H4 vs H1; zona dipantau sebagai RADAR"
        elif rejected:
            out["empty_reason"] = f"Tidak ada zona lolos standar. Contoh penolakan: {rejected[0].rejection_reason}"
        else:
            out["empty_reason"] = "Tidak ditemukan OB dengan BOS, displacement, dan FVG yang memenuhi syarat"

    lines = ["=" * 70, f"[PYTHON-ONLY XAUUSD ZONE SCANNER] Harga {current_price:.2f}",
             f"Konteks: D1 {ctx['D1']} | H4 {ctx['H4']} | H1 {ctx['H1']} | M30 {ctx['M30']} -> bias [{ctx['bias_utama']}]",
             "=" * 70]
    if active:
        lines.append(f"DITEMUKAN {len(active)} ZONA VALID AKTIF (TOTAL RADAR: {len(radar_zones)}):")
        for i, z in enumerate(active, 1):
            fs = (z.fvg_confirmation or {}).get("status", "-")
            lines.append(f"  {i}. [{z.grade}] {z.timeframe} {z.zone_type} {z.price_low:.2f}-{z.price_high:.2f} "
                         f"(skor {z.score:.1f} | {z.freshness} | FVG {fs} | sweep {z.liquidity_sweep_status} | SL {z.sl_recommended:.2f})")
    elif radar_zones:
        lines.append(f"TIDAK ADA ZONA VALID AKTIF, DITEMUKAN {len(radar_zones)} ZONA RADAR (STATUS: WAIT):")
        for i, z in enumerate(radar_zones[:3], 1):
            lines.append(f"  {i}. [RADAR] {z.timeframe} {z.zone_type} {z.price_low:.2f}-{z.price_high:.2f} ({z.freshness})")
    else:
        lines += ["TIDAK ADA ZONA VALID MAUPUN RADAR (STATUS: WAIT)", f"  Alasan: {out['empty_reason']}"]
    lines.append(f"Kandidat ditolak: {len(rejected)}")
    for z in rejected[:3]:
        lines.append(f"  - {z.timeframe} {z.zone_type} {z.price_low:.2f}-{z.price_high:.2f}: {z.rejection_reason}")
    lines.append("=" * 70)
    out["summary_text"] = "\n".join(lines)
    out["diagnostic_text"] = format_diagnostic_report(out)

    _audit("zone_scan", {
        "price": current_price, "bias": ctx["bias_utama"],
        "zones": [{"id": z.zone_id, "tf": z.timeframe, "type": z.zone_type, "low": z.price_low, "high": z.price_high,
                   "bos": z.origin_bos.get("tipe"), "fvg": (z.fvg_confirmation or {}).get("status"),
                   "sweep": z.liquidity_sweep_status, "fresh": z.freshness, "score": z.score,
                   "status": "ACCEPTED" if z.is_valid else "REJECTED", "reason": z.rejection_reason or "; ".join(z.audit[-1:])}
                  for z in (active + rejected)[:30]],
    }, _sig([z.zone_id for z in active], [z.zone_id for z in rejected], ctx["bias_utama"]))
    _SCAN_CACHE.update(key=ck, ts=time.time(), data=out)
    return out

    _audit("zone_scan", {
        "price": current_price, "bias": ctx["bias_utama"],
        "zones": [{"id": z.zone_id, "tf": z.timeframe, "type": z.zone_type, "low": z.price_low, "high": z.price_high,
                   "bos": z.origin_bos.get("tipe"), "fvg": (z.fvg_confirmation or {}).get("status"),
                   "sweep": z.liquidity_sweep_status, "fresh": z.freshness, "score": z.score,
                   "status": "ACCEPTED" if z.is_valid else "REJECTED", "reason": z.rejection_reason or "; ".join(z.audit[-1:])}
                  for z in (active + rejected)[:30]],
    }, _sig([z.zone_id for z in active], [z.zone_id for z in rejected], ctx["bias_utama"]))
    _SCAN_CACHE.update(key=ck, ts=time.time(), data=out)
    return out


# =============================================================================
# ENTRY VALIDATOR (M5) + RISK
# =============================================================================
def _structural_targets(frames: Dict[str, pd.DataFrame], zone_type: str, entry: float) -> List[float]:
    """Likuiditas struktural searah transaksi (swing H1/H4 yang belum ditembus), diurutkan dari terdekat."""
    levels: List[float] = []
    for tf in ("H1", "H4"):
        df = frames.get(tf)
        if df is None or len(df) < 2 * SWING_N + 1:
            continue
        H, L = df["High"].values, df["Low"].values
        for s in detect_swings(df):
            i, p = s["idx"], s["price"]
            if zone_type == "DEMAND" and s["type"] == "H" and p > entry and (i + 1 >= len(H) or H[i + 1:].max() <= p):
                levels.append(p)
            elif zone_type == "SUPPLY" and s["type"] == "L" and p < entry and (i + 1 >= len(L) or L[i + 1:].min() >= p):
                levels.append(p)
    levels = sorted(set(round(x, 2) for x in levels), reverse=(zone_type == "SUPPLY"))
    return levels


def _entry_check(zone: Zone, price: float, df_m5: Optional[pd.DataFrame], spread: float = 0.0,
                 news_blackout=False, htf_frames: Optional[Dict[str, pd.DataFrame]] = None,
                 require_sweep: Optional[bool] = None, server_time: Any = None) -> Dict[str, Any]:
    """Hasil terstruktur: status VALID/WAIT/INVALID, reasons, stages, plan.
    Urutan validasi ketat konsisten untuk semua grade zona:
      1. Validasi zona & integritas data candle close selesai
      2. Harga mencapai atau memasuki zona entry (dan belum terlambat)
      3. Terdapat indikasi Liquidity Sweep yang valid dan relevan dengan area zona
      4. Terdapat konfirmasi pembalikan atau kelanjutan struktur (CHoCH/BOS/CISD) SETELAH sweep
      5. Terdapat displacement yang memadai searah transaksi
      6. Risk Reward memenuhi syarat (>= 1.5) dan jarak SL wajar (<= 20 pt)
    """
    stages = {
        "zone_validation": {"status": "NOT_CHECKED", "msg": ""},
        "zone_touch": {"status": "NOT_CHECKED", "msg": ""},
        "liquidity_sweep": {"status": "NOT_CHECKED", "msg": "", "pool": None},
        "structure_trigger": {"status": "NOT_CHECKED", "msg": ""},
        "displacement": {"status": "NOT_CHECKED", "msg": "", "atr": 0.0},
        "entry_validation": {"status": "NOT_CHECKED", "msg": ""},
        "risk_reward": {"status": "NOT_CHECKED", "msg": "", "rr": None},
    }
    res: Dict[str, Any] = {"status": "WAIT", "reasons": [], "plan": None, "m5": {}, "stages": stages}
    W = res["reasons"]

    # 1. Validasi Zona
    if not zone.is_valid:
        res["status"] = "INVALID"
        msg = f"Zona {zone.zone_id} tidak valid: {zone.rejection_reason}"
        stages["zone_validation"] = {"status": "INVALID", "msg": msg}
        stages["entry_validation"] = {"status": "INVALID", "msg": msg}
        W.append(msg)
        return res

    if not (isinstance(price, (int, float)) and math.isfinite(price) and price > 0):
        stages["zone_validation"] = {"status": "WAIT", "msg": "Harga tidak valid"}
        W.append("Harga tidak valid")
        return res

    demand = zone.zone_type == "DEMAND"
    if (demand and price < zone.invalidation_price) or ((not demand) and price > zone.invalidation_price):
        res["status"] = "INVALID"
        msg = f"Harga {price:.2f} sudah menembus level invalidasi {zone.invalidation_price:.2f}"
        stages["zone_validation"] = {"status": "INVALID", "msg": msg}
        stages["entry_validation"] = {"status": "INVALID", "msg": msg}
        W.append(msg)
        return res

    if news_blackout:
        stages["zone_validation"] = {"status": "WAIT", "msg": f"Blackout berita: {news_blackout}"}
        W.append(f"Blackout berita: {news_blackout}")
        return res

    if df_m5 is None or len(df_m5) < MIN_BARS_DECISION:
        stages["zone_validation"] = {"status": "WAIT", "msg": "Data M5 tidak mencukupi"}
        W.append("Data M5 tidak mencukupi")
        return res

    # Cek candle M5 terakhir belum selesai (running)
    if df_m5.attrs.get("is_closed") is False:
        stages["zone_validation"] = {"status": "WAIT", "msg": "Candle terakhir M5 belum selesai (masih berjalan)"}
        W.append("Candle terakhir M5 belum selesai (masih berjalan)")
        return res

    if server_time is not None:
        diff_sec = (pd.Timestamp(server_time) - df_m5.index[-1]).total_seconds()
        if diff_sec < 5 * 60 - 1e-6:
            stages["zone_validation"] = {"status": "WAIT", "msg": "Candle terakhir M5 belum selesai (masih berjalan)"}
            W.append("Candle terakhir M5 belum selesai (masih berjalan)")
            return res

    if spread is not None and spread > max(SPREAD_MAKS_POIN, 0.0):
        stages["zone_validation"] = {"status": "WAIT", "msg": f"Spread {spread:.2f} pt melebihi batas {SPREAD_MAKS_POIN:g}"}
        W.append(f"Spread {spread:.2f} pt melebihi batas {SPREAD_MAKS_POIN:g}")
        return res

    d5 = df_m5 if "ATR14" in df_m5.columns else tambah_indikator(df_m5.copy())
    atr5 = float(d5["ATR14"].iloc[-1])
    if not math.isfinite(atr5) or atr5 <= 0:
        stages["zone_validation"] = {"status": "WAIT", "msg": "ATR M5 tidak valid"}
        W.append("ATR M5 tidak valid")
        return res

    if abs(price - float(d5["Close"].iloc[-1])) > max(30.0, 6 * atr5):
        stages["zone_validation"] = {"status": "WAIT", "msg": "Harga menyimpang jauh dari candle M5 terakhir"}
        W.append("Harga menyimpang jauh dari candle M5 terakhir (data tidak sinkron)")
        return res

    stages["zone_validation"] = {"status": "CONFIRMED", "msg": f"Zona {zone.grade} {zone.timeframe} {zone.zone_type} valid"}

    lo, hi = float(zone.price_low), float(zone.price_high)
    n5 = len(d5)
    O, H, L, C = (d5[c].values for c in ("Open", "High", "Low", "Close"))

    # 2. Pengecekan apakah harga mencapai atau memasuki zona entry
    touched = False
    touch_indices = []
    for x in range(max(0, n5 - M5_ZONE_TOUCH_LOOKBACK), n5):
        if (demand and L[x] <= hi and H[x] >= lo) or ((not demand) and H[x] >= lo and L[x] <= hi):
            touched = True
            touch_indices.append(x)
    if not touched and (lo <= price <= hi):
        touched = True
        touch_indices.append(n5 - 1)

    if not touched:
        stages["zone_touch"] = {"status": "WAIT", "msg": f"Harga belum mencapai atau menyentuh zona entry {lo:.2f}-{hi:.2f}"}
        W.append(f"Harga belum mencapai atau menyentuh zona entry {lo:.2f}-{hi:.2f}")
        return res

    first_touch_k = touch_indices[0] if touch_indices else 0
    dist = 0.0 if lo <= price <= hi else (lo - price if price < lo else price - hi)
    approach = max(3.0, 1.5 * atr5)
    if dist > approach:
        stages["zone_touch"] = {"status": "WAIT", "msg": f"Entry terlambat: harga sudah meninggalkan zona entry ({dist:.2f} pt > {approach:.2f} pt)"}
        W.append(f"Entry terlambat: harga sudah meninggalkan zona entry (jarak {dist:.2f} > toleransi {approach:.2f})")
        return res

    stages["zone_touch"] = {"status": "CONFIRMED", "msg": f"Zona tersentuh (jarak {dist:.2f} pt)"}

    events, _, swings = detect_market_structure(d5)
    sweeps = detect_liquidity_sweeps(d5, swings, lookback=40)
    want_dir = "bull" if demand else "bear"
    want_sw = "SWEEP_LOW" if demand else "SWEEP_HIGH"

    m5 = res["m5"]
    m5.update(structure_event=None, sweep=False, displacement_atr=0.0, cisd=False, zone_touched=touched)

    # 3. Indikasi Liquidity Sweep yang Valid dan Relevan terhadap Zona
    need_sweep = True if require_sweep is None else require_sweep
    relevant_sweeps = []
    for s in sweeps:
        if s["type"] == want_sw and n5 - 1 - s["k"] <= 40:
            is_rel, rel_msg = is_sweep_relevant_to_zone(s, zone, atr5)
            if is_rel:
                relevant_sweeps.append(s)

    if need_sweep and not relevant_sweeps:
        any_sweeps = [s for s in sweeps if s["type"] == want_sw and n5 - 1 - s["k"] <= 40]
        if any_sweeps:
            stages["liquidity_sweep"] = {"status": "WAIT", "msg": f"M5 Liquidity Sweep terjadi jauh dari area zona {lo:.2f}-{hi:.2f}"}
            W.append(f"M5 Liquidity Sweep terjadi jauh dari area zona {lo:.2f}-{hi:.2f}")
        else:
            stages["liquidity_sweep"] = {"status": "WAIT", "msg": f"M5 Liquidity Sweep tidak terkonfirmasi ({want_sw} belum terjadi di area zona)"}
            W.append(f"M5 Liquidity Sweep tidak terkonfirmasi ({want_sw} belum terjadi di area zona)")
        return res

    best_sweep = relevant_sweeps[-1] if relevant_sweeps else None
    k_sweep = best_sweep["k"] if best_sweep else -1
    sweep_pool = best_sweep.get("pool_description", "Liquidity Pool") if best_sweep else None

    stages["liquidity_sweep"] = {
        "status": "CONFIRMED" if best_sweep else "NOT_CHECKED",
        "msg": best_sweep.get("description", "Sweep terkonfirmasi") if best_sweep else "Sweep dilewati",
        "pool": sweep_pool,
        "k": k_sweep,
    }
    m5["sweep"] = bool(best_sweep)
    m5["sweep_k"] = k_sweep
    m5["pool"] = sweep_pool

    # 4. Konfirmasi Struktur (CHoCH / BOS / CISD) SETELAH Sweep
    cl = []
    try:
        from modules.analisis_teknikal import deteksi_cisd
        cl, _ = deteksi_cisd(d5, swings, lookback=40)
    except Exception:
        pass

    matching_events = [e for e in reversed(events) if e["arah"] == want_dir and n5 - 1 - e["k"] <= M5_EVENT_MAX_AGE and (k_sweep < 0 or e["k"] >= k_sweep)]
    matching_cisd = [c for c in cl if c["arah"] == ("BULLISH" if demand else "BEARISH") and n5 - 1 - c["k"] <= M5_EVENT_MAX_AGE and (k_sweep < 0 or c["k"] >= k_sweep)]

    if not matching_events and not matching_cisd:
        prior_events = [e for e in reversed(events) if e["arah"] == want_dir and n5 - 1 - e["k"] <= M5_EVENT_MAX_AGE]
        prior_cisd = [c for c in cl if c["arah"] == ("BULLISH" if demand else "BEARISH") and n5 - 1 - c["k"] <= M5_EVENT_MAX_AGE]
        if prior_events or prior_cisd:
            stages["structure_trigger"] = {"status": "WAIT", "msg": "Konfirmasi struktur terjadi sebelum liquidity sweep (urutan tidak valid)"}
            W.append("Konfirmasi struktur terjadi sebelum liquidity sweep (urutan tidak valid)")
        else:
            stages["structure_trigger"] = {"status": "WAIT", "msg": f"Belum ada konfirmasi struktur pembalikan/kelanjutan M5 ({'BOS/CHoCH/CISD bullish' if demand else 'BOS/CHoCH/CISD bearish'})"}
            W.append(f"Belum ada konfirmasi struktur pembalikan/kelanjutan M5 ({'BOS/CHoCH/CISD bullish' if demand else 'BOS/CHoCH/CISD bearish'})")
        return res

    ev = matching_events[0] if matching_events else None
    k_ref = ev["k"] if ev else matching_cisd[0]["k"]
    cisd_ok = bool(matching_cisd and (not ev or matching_cisd[0]["k"] >= ev["k"]))
    m5["cisd"] = cisd_ok
    struct_str = f"{ev['tipe']} {ev['arah']} @ {ev['level']:.2f}" if ev else ("CISD " + ("BULLISH" if demand else "BEARISH"))
    m5["structure_event"] = f"{struct_str} (k={k_ref})"
    stages["structure_trigger"] = {"status": "CONFIRMED", "msg": m5["structure_event"], "k": k_ref}

    # 5. Displacement yang Mendukung Arah Entry
    k0 = max(0, k_ref - 3)
    if demand:
        dir_bodies = [C[x] - O[x] for x in range(k0, min(n5, k_ref + 2)) if C[x] > O[x]]
    else:
        dir_bodies = [O[x] - C[x] for x in range(k0, min(n5, k_ref + 2)) if O[x] > C[x]]
    disp_body = max(dir_bodies) if dir_bodies else 0.0
    disp5 = float(max(disp_body, np.max(np.abs(C[k0:k_ref + 1] - O[k0:k_ref + 1])))) / atr5
    disp5 = round(disp5, 2)
    m5["displacement_atr"] = disp5

    if disp5 < M5_MIN_DISPLACEMENT_ATR:
        stages["displacement"] = {"status": "WEAK", "msg": f"Displacement M5 lemah ({disp5:.2f} ATR < {M5_MIN_DISPLACEMENT_ATR:g})", "atr": disp5}
        W.append(f"Displacement M5 lemah ({disp5:.2f} ATR < {M5_MIN_DISPLACEMENT_ATR:g})")
        return res

    stages["displacement"] = {"status": "CONFIRMED", "msg": f"Displacement M5 {disp5:.2f} ATR", "atr": disp5}

    # 6. Risk Reward & Risk Management
    if demand:
        e_lo, e_hi = (lo, hi) if hi - lo <= ENTRY_LEBAR_MAKS else (round(hi - ENTRY_LEBAR_MAKS, 2), hi)
    else:
        e_lo, e_hi = (lo, hi) if hi - lo <= ENTRY_LEBAR_MAKS else (lo, round(lo + ENTRY_LEBAR_MAKS, 2))
    entry = round((e_lo + e_hi) / 2.0, 2)
    sl = zone.sl_recommended
    risk = round((entry - sl) if demand else (sl - entry), 2)

    if risk <= 0.5:
        stages["risk_reward"] = {"status": "WAIT", "msg": "Jarak SL tidak realistis"}
        W.append("Jarak SL tidak realistis")
        return res
    if risk > SL_JARAK_MAKS:
        stages["risk_reward"] = {"status": "WAIT", "msg": f"Jarak SL struktural {risk:.2f} pt melebihi batas maksimal {SL_JARAK_MAKS:g}"}
        W.append(f"Jarak SL struktural {risk:.2f} pt melebihi batas maksimal {SL_JARAK_MAKS:g}")
        return res

    targets = _structural_targets(htf_frames or {}, zone.zone_type, entry)
    if not targets:
        stages["risk_reward"] = {"status": "WAIT", "msg": "Tidak ada target likuiditas struktural H1/H4 searah transaksi"}
        W.append("Tidak ada target likuiditas struktural H1/H4 searah transaksi")
        return res

    buf = 0.3
    tp1 = round(targets[0] - buf if demand else targets[0] + buf, 2)
    tp2 = round(targets[1] - buf if demand else targets[1] + buf, 2) if len(targets) > 1 else tp1
    reward = round((tp1 - entry) if demand else (entry - tp1), 2)
    rr = round(reward / risk, 2) if risk > 0 else 0.0
    res["rr"] = rr

    if reward <= 0 or rr < RR_MIN - 1e-9:
        stages["risk_reward"] = {"status": "WAIT", "msg": f"R:R aktual ke TP1 struktural {rr:.2f} < 1:{RR_MIN:g}", "rr": rr}
        W.append(f"R:R aktual ke TP1 struktural {rr:.2f} < 1:{RR_MIN:g}")
        return res

    stages["risk_reward"] = {"status": "CONFIRMED", "msg": f"1:{rr:.2f}", "rr": rr}
    stages["entry_validation"] = {"status": "PASSED", "msg": f"Setup {zone.grade} valid"}

    res["plan"] = {"arah": "BUY" if demand else "SELL", "entry_min": e_lo, "entry_max": e_hi, "entry_acuan": entry,
                   "sl": sl, "tp1": tp1, "tp2": tp2, "rr": rr, "zone_id": zone.zone_id,
                   "timeframe": zone.timeframe, "grade": zone.grade, "score": zone.score,
                   "keterangan": f"Zona {zone.grade} {zone.timeframe} {zone.zone_type} + trigger M5 {m5['structure_event']}"}
    res["status"] = "VALID"
    W.append(f"Entry valid: {m5['structure_event']}, displacement M5 {disp5:.2f} ATR, R:R {rr:.2f}")
    return res


def validate_entry_setup(zone: Zone, current_price: float, df_m5: pd.DataFrame, spread_points: float = 0.0,
                         is_news_blackout=False, htf_frames: Optional[Dict[str, pd.DataFrame]] = None,
                         require_sweep: Optional[bool] = None,
                         server_time: Any = None) -> Tuple[bool, str, Dict[str, Any]]:
    """Kompatibel dgn versi lama: (ok, pesan, plan). Status lengkap ada di plan['status'] bila ok; pesan diawali VALID/WAIT/INVALID."""
    r = _entry_check(zone, current_price, df_m5, spread_points, is_news_blackout, htf_frames, require_sweep, server_time=server_time)
    return r["status"] == "VALID", f"{r['status']}: " + "; ".join(r["reasons"]), (r["plan"] or {})


# =============================================================================
# EARLY SETUP DETECTION & PENDING ORDER PLANNER (MODE A)
# =============================================================================
def calculate_suggested_expiration(
    base_time: Any = None,
    duration_hours: Union[int, float] = DEFAULT_PENDING_EXPIRATION_HOURS,
    target_time: Any = None,
) -> Dict[str, Any]:
    """
    Hitung suggested expiration secara dinamis berdasarkan current/server timestamp.
    Menghitung durasi aktual (selisih waktu nyata antara expiration dan base_time):
      - Misal snapshot 14:46 dan expiration 15:46 -> "1 jam"
      - Misal snapshot 14:46 dan expiration 22:46 -> "8 jam"
    Expiration selalu >= base_time.
    """
    if base_time is None:
        base_dt = datetime.now()
    elif isinstance(base_time, (int, float)):
        base_dt = datetime.fromtimestamp(base_time)
    elif isinstance(base_time, str):
        try:
            clean_base = str(base_time).replace(" WIB", "").strip()
            base_dt = pd.Timestamp(clean_base).to_pydatetime()
        except Exception:
            base_dt = datetime.now()
    elif hasattr(base_time, "to_pydatetime"):
        base_dt = base_time.to_pydatetime()
    elif isinstance(base_time, datetime):
        base_dt = base_time
    else:
        base_dt = datetime.now()

    if target_time is not None:
        if isinstance(target_time, (int, float)):
            target_dt = datetime.fromtimestamp(target_time)
        elif isinstance(target_time, str):
            try:
                clean_target = str(target_time).replace(" WIB", "").strip()
                target_dt = pd.Timestamp(clean_target).to_pydatetime()
            except Exception:
                target_dt = base_dt + timedelta(hours=max(0.0, float(duration_hours)))
        elif hasattr(target_time, "to_pydatetime"):
            target_dt = target_time.to_pydatetime()
        elif isinstance(target_time, datetime):
            target_dt = target_time
        else:
            target_dt = base_dt + timedelta(hours=max(0.0, float(duration_hours)))
    else:
        dur = max(0.0, float(duration_hours))
        target_dt = base_dt + timedelta(hours=dur)

    if base_dt.tzinfo is not None and target_dt.tzinfo is None:
        target_dt = target_dt.replace(tzinfo=base_dt.tzinfo)
    elif base_dt.tzinfo is None and target_dt.tzinfo is not None:
        base_dt = base_dt.replace(tzinfo=target_dt.tzinfo)

    if target_dt < base_dt:
        target_dt = base_dt

    delta = target_dt - base_dt
    total_sec = max(0, int(delta.total_seconds()))

    hours = total_sec // 3600
    minutes = (total_sec % 3600) // 60

    if hours > 0 and minutes == 0:
        dur_str = f"{hours} jam"
    elif hours > 0 and minutes > 0:
        dur_str = f"{hours} jam {minutes} menit"
    elif minutes > 0:
        dur_str = f"{minutes} menit"
    else:
        dur_str = "0 jam"

    exp_str = target_dt.strftime("%Y-%m-%d %H:%M")
    formatted_str = f"{exp_str} ({dur_str})"

    return {
        "base_time": base_dt.strftime("%Y-%m-%d %H:%M"),
        "expires_at": exp_str,
        "duration_hours": round(total_sec / 3600.0, 2),
        "duration_text": dur_str,
        "suggested_expiration": formatted_str,
    }


def plan_pending_order(
    zone: Zone,
    current_price: float,
    frames: Optional[Dict[str, pd.DataFrame]] = None,
    htf_bias: Optional[str] = None,
    m30_status: Optional[str] = None,
    server_time: Any = None,
    expiration_hours: int = DEFAULT_PENDING_EXPIRATION_HOURS,
) -> Dict[str, Any]:
    """
    Rencana Pending Order (BUY LIMIT / SELL LIMIT) untuk Early Setup Detection (Mode A - Aggressive Pending).
    
    Persyaratan:
    1. Filter HTF: H4 directional bias selaras (Demand wajib Bullish, Supply wajib Bearish).
    2. Filter M30: Momentum tidak boleh CONFLICT, INVALID, atau WEAK.
    3. Invalidation Level: Harga belum menembus batas pembatalan zona.
    4. Minimum RR: Wajib memenuhi 1:1.5.
    5. Entry Price: Dihitung secara objektif dari batas zona aktual (proximal edge).
    """
    demand = (getattr(zone, "zone_type", "") == "DEMAND")
    order_type = "BUY LIMIT" if demand else "SELL LIMIT"
    
    res: Dict[str, Any] = {
        "valid": False,
        "order_type": order_type,
        "zone_id": getattr(zone, "zone_id", ""),
        "zone_type": getattr(zone, "zone_type", ""),
        "timeframe": getattr(zone, "timeframe", ""),
        "zone_timeframe": getattr(zone, "timeframe", ""),
        "zone_grade": getattr(zone, "grade", ""),
        "zone_score": float(getattr(zone, "score", 0.0)),
        "zone_high": float(getattr(zone, "price_high", 0.0)),
        "zone_low": float(getattr(zone, "price_low", 0.0)),
        "entry_price": 0.0,
        "sl": 0.0,
        "tp1": 0.0,
        "tp2": 0.0,
        "risk": 0.0,
        "risk_points": 0.0,
        "reward": 0.0,
        "reward_tp1": 0.0,
        "reward_tp2": 0.0,
        "rr": 0.0,
        "rr_tp1": 0.0,
        "rr_tp2": 0.0,
        "invalidation_price": float(getattr(zone, "invalidation_price", 0.0)),
        "structural_invalidation": float(getattr(zone, "invalidation_price", 0.0)),
        "invalidation_buffer": 0.0,
        "zone_boundary": f"{float(getattr(zone, 'price_low', 0.0)):.2f} - {float(getattr(zone, 'price_high', 0.0)):.2f}",
        "level_explanations": {},
        "suggested_expiration": "",
        "expires_at": "",
        "entry_reason": "",
        "status": "WAIT",
        "early_pending_status": "WAIT",
        "early_pending_status_label": "TIDAK_AKTIF",
        "early_pending_rr": "DATA_UNAVAILABLE",
        "rejection_reason": "",
    }
    
    # 1. Validasi Zona Dasar
    if not getattr(zone, "is_valid", False):
        res["status"] = "INVALID"
        res["early_pending_status"] = "INVALID"
        res["rejection_reason"] = f"Zona {res['zone_id']} tidak valid: {getattr(zone, 'rejection_reason', '')}"
        return res
        
    if not (isinstance(current_price, (int, float)) and math.isfinite(current_price) and current_price > 0):
        res["status"] = "WAIT"
        res["early_pending_status"] = "WAIT"
        res["rejection_reason"] = "Harga saat ini tidak valid"
        return res

    # 2. Invalidation Level Check
    inval = res["invalidation_price"]
    if demand and current_price < inval:
        res["status"] = "INVALID"
        res["early_pending_status"] = "INVALID"
        res["rejection_reason"] = f"Harga ({current_price:.2f}) telah menembus invalidation level Demand ({inval:.2f})"
        return res
    if (not demand) and current_price > inval:
        res["status"] = "INVALID"
        res["early_pending_status"] = "INVALID"
        res["rejection_reason"] = f"Harga ({current_price:.2f}) telah menembus invalidation level Supply ({inval:.2f})"
        return res

    # 3. Filter Arah HTF (H4)
    if htf_bias is not None:
        if demand and htf_bias != "BULLISH":
            res["status"] = "CONFLICT"
            res["early_pending_status"] = "CONFLICT"
            res["rejection_reason"] = f"Bias H4 ({htf_bias}) tidak selaras dengan setup BUY LIMIT (harus BULLISH)"
            return res
        if (not demand) and htf_bias != "BEARISH":
            res["status"] = "CONFLICT"
            res["early_pending_status"] = "CONFLICT"
            res["rejection_reason"] = f"Bias H4 ({htf_bias}) tidak selaras dengan setup SELL LIMIT (harus BEARISH)"
            return res

    # 4. Filter Momentum M30
    if m30_status is not None:
        if m30_status == "CONFLICT":
            res["status"] = "CONFLICT"
            res["early_pending_status"] = "CONFLICT"
            res["rejection_reason"] = "Momentum M30 berlawanan dengan arah setup"
            return res
        if m30_status == "INVALID":
            res["status"] = "INVALID"
            res["early_pending_status"] = "INVALID"
            res["rejection_reason"] = "Data atau struktur M30 invalid"
            return res
        if m30_status == "WEAK":
            res["status"] = "WEAK"
            res["early_pending_status"] = "WEAK"
            res["rejection_reason"] = "Momentum M30 lemah / ranging"
            return res

    # 5. Perhitungan Entry Price (Batas Proximal Edge Zona Aktual)
    # BUY LIMIT: Batas atas zona Demand (price_high)
    # SELL LIMIT: Batas bawah zona Supply (price_low)
    if demand:
        entry = round(res["zone_high"], 2)
        entry_reason = "Batas atas (proximal edge) zona Demand H1/H4 untuk mitigasi awal"
    else:
        entry = round(res["zone_low"], 2)
        entry_reason = "Batas bawah (proximal edge) zona Supply H1/H4 untuk mitigasi awal"

    # 6. Stop Loss & Invalidation Buffer
    # Stop Loss wajib selalu berada di luar invalidation level dengan buffer yang valid:
    # Demand: SL < Invalidation (invalidation_buffer = inval - SL)
    # Supply: SL > Invalidation (invalidation_buffer = SL - inval)
    raw_sl = round(getattr(zone, "sl_recommended", 0.0), 2)
    min_buf = 1.50
    if demand:
        if raw_sl > 0.0 and raw_sl < inval:
            sl = raw_sl
        else:
            sl = round(inval - min_buf, 2)
        inval_buffer = round(inval - sl, 2)
    else:
        if raw_sl > 0.0 and raw_sl > inval:
            sl = raw_sl
        else:
            sl = round(inval + min_buf, 2)
        inval_buffer = round(sl - inval, 2)
    
    risk_points = round((entry - sl) if demand else (sl - entry), 2)
    if risk_points <= 0.5:
        res["status"] = "INVALID"
        res["early_pending_status"] = "INVALID"
        res["rejection_reason"] = f"Jarak SL ({risk_points:.2f} pt) terlalu sempit / tidak realistis"
        return res
    if risk_points > SL_JARAK_MAKS:
        res["status"] = "INVALID"
        res["early_pending_status"] = "INVALID"
        res["rejection_reason"] = f"Jarak SL ({risk_points:.2f} pt) melebihi batas toleransi maksimal ({SL_JARAK_MAKS:g} pt)"
        return res

    # 7. Take Profit Struktural & Risk/Reward
    targets = _structural_targets(frames or {}, getattr(zone, "zone_type", ""), entry) if frames else []
    buf = 0.3
    if demand:
        if targets:
            tp1 = round(targets[0] - buf, 2)
            tp2 = round(targets[1] - buf, 2) if len(targets) > 1 else round(entry + 2.5 * risk_points, 2)
        else:
            tp1 = round(entry + max(RR_MIN, 2.0) * risk_points, 2)
            tp2 = round(entry + 3.0 * risk_points, 2)
        reward_tp1 = round(tp1 - entry, 2)
        reward_tp2 = round(tp2 - entry, 2)
    else:
        if targets:
            tp1 = round(targets[0] + buf, 2)
            tp2 = round(targets[1] + buf, 2) if len(targets) > 1 else round(entry - 2.5 * risk_points, 2)
        else:
            tp1 = round(entry - max(RR_MIN, 2.0) * risk_points, 2)
            tp2 = round(entry - 3.0 * risk_points, 2)
        reward_tp1 = round(entry - tp1, 2)
        reward_tp2 = round(entry - tp2, 2)

    rr_tp1 = round(reward_tp1 / risk_points, 2) if risk_points > 0 else 0.0
    rr_tp2 = round(reward_tp2 / risk_points, 2) if risk_points > 0 else 0.0

    if reward_tp1 <= 0 or rr_tp1 < RR_MIN - 1e-9:
        res["status"] = "INVALID"
        res["early_pending_status"] = "INVALID"
        res["early_pending_rr"] = "FAILED"
        res["rr"] = rr_tp1
        res["rr_tp1"] = rr_tp1
        res["rejection_reason"] = f"Risk/Reward ke TP1 struktural (1:{rr_tp1:.2f}) di bawah minimum 1:{RR_MIN:g}"
        return res

    # 8. Suggested Expiration Dinamis
    exp_info = calculate_suggested_expiration(base_time=server_time, duration_hours=expiration_hours)

    # 9. Penjelasan Objektif Level
    level_explanations = {
        "zone_boundary": f"Area reaksi institusional ({res['zone_low']:.2f} - {res['zone_high']:.2f}) tempat order berpotensi terisi",
        "structural_invalidation": f"Level ({inval:.2f}) di mana struktur zona resmi batal jika tertembus",
        "stop_loss": f"Level proteksi modal ({sl:.2f}) yang diletakkan di luar invalidation dengan buffer ({inval_buffer:.2f} pt) untuk menghindari market noise/spread",
    }

    res.update({
        "valid": True,
        "entry_price": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "risk": risk_points,
        "risk_points": risk_points,
        "reward": reward_tp1,
        "reward_tp1": reward_tp1,
        "reward_tp2": reward_tp2,
        "rr": rr_tp1,
        "rr_tp1": rr_tp1,
        "rr_tp2": rr_tp2,
        "structural_invalidation": inval,
        "invalidation_buffer": inval_buffer,
        "zone_boundary": f"{res['zone_low']:.2f} - {res['zone_high']:.2f}",
        "level_explanations": level_explanations,
        "suggested_expiration": exp_info["suggested_expiration"],
        "expires_at": exp_info["expires_at"],
        "duration_hours": exp_info["duration_hours"],
        "duration_text": exp_info["duration_text"],
        "entry_reason": entry_reason,
        "status": "READY",
        "early_pending_status": "READY",
        "early_pending_status_label": "READY — PENDING PLAN (BELUM DIPASANG DI MT5)",
        "early_pending_rr": f"PASSED (1:{rr_tp1:.1f})",
        "rejection_reason": "",
    })
    return res


def detect_early_setups(
    frames: Optional[Dict[str, pd.DataFrame]] = None,
    current_price: Optional[float] = None,
    server_time: Any = None,
    scan: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Deteksi peluang zona potensial lebih awal sebelum konfirmasi M5 (Mode A - Aggressive Pending).
    """
    if scan is None:
        if frames is None:
            frames = _load_frames()
            if frames is None:
                return {"best_setup": None, "all_setups": [], "empty_reason": "MT5 offline; data tidak tersedia",
                        "htf_bias": "NETRAL", "h1_structure": "NETRAL", "m30_momentum": "NOT_CHECKED"}
        if current_price is None:
            ref = frames.get("M5") if frames.get("M5") is not None and len(frames["M5"]) else frames.get("H1")
            if ref is not None and len(ref):
                current_price = float(ref["Close"].iloc[-1])
        scan = scan_xauusd_zones(frames=frames, current_price=current_price)

    ctx = scan.get("htf_context") or (evaluate_htf_context(frames) if frames else {})
    h4_bias = ctx.get("H4", "NETRAL")
    h1_struct = ctx.get("H1", "NETRAL")
    
    if frames and frames.get("M30") is not None:
        m30_val = evaluate_m30_momentum(frames.get("M30"), h4_bias)
    else:
        m30_val = {"status": "CONFIRMED", "msg": "M30 ok"}
    m30_status = m30_val.get("status", "CONFIRMED")

    if current_price is None:
        current_price = scan.get("current_price", 0.0)

    out: Dict[str, Any] = {
        "best_setup": None,
        "all_setups": [],
        "htf_bias": h4_bias,
        "h1_structure": h1_struct,
        "m30_momentum": m30_status,
        "m30_detail": m30_val,
        "empty_reason": "",
    }

    if h4_bias not in ("BULLISH", "BEARISH"):
        out["empty_reason"] = f"Bias H4 netral / tidak ada directional bias ({h4_bias})"
        return out

    if ctx.get("has_conflict"):
        out["empty_reason"] = f"Konflik struktur H4 ({h4_bias}) vs H1 ({h1_struct})"
        return out

    if m30_status in ("CONFLICT", "INVALID", "WEAK"):
        out["empty_reason"] = f"Momentum M30 tidak mendukung aggressive pending ({m30_status}: {m30_val.get('msg', '')})"
        return out

    wanted_type = "DEMAND" if h4_bias == "BULLISH" else "SUPPLY"
    cands = [z for z in scan.get("active_zones", []) if z.get("zone_type") == wanted_type]
    if not cands:
        out["empty_reason"] = f"Tidak ada zona {wanted_type} berkualitas yang aktif pada H4/H1"
        return out

    valid_plans = []
    for zd in cands:
        z_obj = _zone_from_dict(zd) if isinstance(zd, dict) else zd
        plan = plan_pending_order(z_obj, float(current_price), frames, h4_bias, m30_status, server_time)
        if plan.get("valid"):
            valid_plans.append(plan)

    out["all_setups"] = valid_plans
    if valid_plans:
        out["best_setup"] = valid_plans[0]
    else:
        out["empty_reason"] = "Tidak ada zona yang memenuhi syarat Risk/Reward >= 1.5 atau batas invalidasi"
    return out


# =============================================================================
# ZONE TRACKING SYSTEM
# =============================================================================
class ZoneTracker:
    def __init__(self, filepath: Optional[str] = None, auto_save: bool = False):
        self.filepath = filepath or FILE_STATE_ZONE_TRACKER
        self.auto_save = auto_save
        self.tracked_zones: Dict[str, Dict[str, Any]] = {}
        self.alerts: List[Dict[str, Any]] = []
        if self.filepath and os.path.exists(self.filepath):
            self.load()

    def update(
        self,
        zones: List[Any],
        current_price: float,
        frames: Optional[Dict[str, pd.DataFrame]] = None,
        htf_bias: Optional[str] = None,
        m30_status: Optional[str] = None,
        server_time: Any = None,
        conservative_dec: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perbarui siklus hidup zona:
        DETECTED -> READY -> APPROACHING -> TOUCHED -> CONFIRMED -> INVALIDATED / EXPIRED.
        Menghasilkan notifikasi/alert dan mencegah duplikasi identitas zona.
        """
        now_dt = pd.Timestamp(server_time).to_pydatetime() if server_time is not None else datetime.now()
        now_str = now_dt.strftime("%Y-%m-%d %H:%M:%S")
        new_alerts: List[Dict[str, Any]] = []

        # 1. Daftarkan zona baru tanpa duplikasi
        for z in (zones or []):
            z_id = getattr(z, "zone_id", None) if not isinstance(z, dict) else z.get("zone_id")
            if not z_id:
                continue
            z_obj = z if isinstance(z, Zone) else _zone_from_dict(z)
            if z_id not in self.tracked_zones:
                exp_dt = now_dt + timedelta(hours=DEFAULT_PENDING_EXPIRATION_HOURS)
                self.tracked_zones[z_id] = {
                    "zone_id": z_id,
                    "symbol": getattr(z_obj, "symbol", SYMBOL),
                    "timeframe": getattr(z_obj, "timeframe", ""),
                    "zone_type": getattr(z_obj, "zone_type", ""),
                    "price_low": float(getattr(z_obj, "price_low", 0.0)),
                    "price_high": float(getattr(z_obj, "price_high", 0.0)),
                    "invalidation_price": float(getattr(z_obj, "invalidation_price", 0.0)),
                    "grade": getattr(z_obj, "grade", ""),
                    "score": float(getattr(z_obj, "score", 0.0)),
                    "first_detected": now_str,
                    "last_updated": now_str,
                    "status": "RADAR",
                    "expires_at": exp_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "pending_plan": None,
                    "history": [{"status": "RADAR", "time": now_str, "reason": "Zona teridentifikasi sebagai RADAR oleh scanner"}],
                    "rejection_reason": "",
                }
            else:
                self.tracked_zones[z_id]["last_updated"] = now_str

        # 2. Evaluasi transisi state untuk seluruh zona yang sedang dilacak
        for z_id, rec in list(self.tracked_zones.items()):
            curr_st = rec.get("status", "RADAR")
            z_type = rec.get("zone_type", "")
            inval = rec.get("invalidation_price", 0.0)

            # A. Cek Expiration
            exp_str = rec.get("expires_at", "")
            try:
                exp_dt = datetime.strptime(exp_str, "%Y-%m-%d %H:%M:%S")
                is_expired = (now_dt > exp_dt)
            except Exception:
                is_expired = False

            if is_expired and curr_st != "EXPIRED":
                rec["status"] = "EXPIRED"
                rec["pending_plan"] = None
                rec["rejection_reason"] = f"Kedaluwarsa (batas waktu {exp_str})"
                rec["history"].append({"status": "EXPIRED", "time": now_str, "reason": rec["rejection_reason"]})
                msg = f"[ALERT] Zona {z_id} telah EXPIRED"
                new_alerts.append({"type": "EXPIRED", "zone_id": z_id, "msg": msg, "time": now_str})
                continue

            if curr_st in ("EXPIRED", "INVALIDATED"):
                continue

            # B. Cek Invalidation Level & Market Change
            inv_reason = None
            if z_type == "DEMAND" and current_price < inval:
                inv_reason = f"Harga ({current_price:.2f}) menembus invalidation level ({inval:.2f})"
            elif z_type == "SUPPLY" and current_price > inval:
                inv_reason = f"Harga ({current_price:.2f}) menembus invalidation level ({inval:.2f})"
            elif z_type == "DEMAND" and htf_bias == "BEARISH":
                inv_reason = "Bias H4 berbalik menjadi BEARISH (berlawanan dengan zona DEMAND)"
            elif z_type == "SUPPLY" and htf_bias == "BULLISH":
                inv_reason = "Bias H4 berbalik menjadi BULLISH (berlawanan dengan zona SUPPLY)"
            elif m30_status == "CONFLICT":
                inv_reason = "Momentum M30 berbalik menjadi CONFLICT"

            if inv_reason:
                rec["status"] = "INVALIDATED"
                rec["pending_plan"] = None
                rec["rejection_reason"] = inv_reason
                rec["history"].append({"status": "INVALIDATED", "time": now_str, "reason": inv_reason})
                msg = f"[ALERT] Zona {z_id} INVALID: {inv_reason}"
                new_alerts.append({"type": "INVALIDATED", "zone_id": z_id, "msg": msg, "time": now_str})
                continue

            # C. Hitung Pending Order Plan & Update Status
            z_mock = Zone(
                zone_id=z_id, symbol=rec["symbol"], timeframe=rec["timeframe"], zone_type=rec["zone_type"],
                price_low=rec["price_low"], price_high=rec["price_high"], midpoint=(rec["price_low"] + rec["price_high"]) / 2,
                score=rec["score"], grade=rec["grade"], origin_bos={}, fvg_confirmation=None, displacement_strength=1.5,
                freshness="FRESH", touch_count=0, liquidity_sweep_status="NONE", invalidation_price=inval,
                is_valid=True, rejection_reason="", sl_recommended=(rec["price_low"] - 1.5 if z_type == "DEMAND" else rec["price_high"] + 1.5),
            )
            plan = plan_pending_order(z_mock, current_price, frames, htf_bias, m30_status, server_time)
            if plan.get("valid"):
                rec["pending_plan"] = plan
                if rec["status"] in ("RADAR", "DETECTED"):
                    rec["status"] = "READY"
                    rec["history"].append({"status": "READY", "time": now_str, "reason": "Memenuhi kriteria Aggressive Pending Order"})
                    msg = f"[INFO] Zona {z_id} READY untuk {plan['order_type']} @ {plan['entry_price']:.2f}"
                    new_alerts.append({"type": "READY", "zone_id": z_id, "msg": msg, "time": now_str})
            else:
                rec["pending_plan"] = None
                rec["rejection_reason"] = plan.get("rejection_reason", "")
                if rec["status"] == "READY":
                    rec["status"] = "RADAR"

            # D. Cek Jarak & Touch
            z_lo, z_hi = rec["price_low"], rec["price_high"]
            is_touched = (z_lo <= current_price <= z_hi)
            dist = 0.0 if is_touched else (z_lo - current_price if current_price < z_lo else current_price - z_hi)
            approach_thresh = ZONE_APPROACH_DISTANCE_POINTS

            if is_touched:
                if rec["status"] in ("RADAR", "DETECTED", "READY", "APPROACHING"):
                    rec["status"] = "TOUCHED"
                    rec["history"].append({"status": "TOUCHED", "time": now_str, "reason": f"Harga ({current_price:.2f}) menyentuh zona {z_lo:.2f}-{z_hi:.2f}"})
                    msg = f"[ALERT] Harga telah menyentuh zona {z_id} ({current_price:.2f})"
                    new_alerts.append({"type": "TOUCHED", "zone_id": z_id, "msg": msg, "time": now_str})
            elif dist <= approach_thresh:
                if rec["status"] in ("RADAR", "DETECTED", "READY"):
                    rec["status"] = "APPROACHING"
                    rec["history"].append({"status": "APPROACHING", "time": now_str, "reason": f"Harga ({current_price:.2f}) mendekati zona (jarak {dist:.2f} pt)"})
                    msg = f"[ALERT] Harga mendekati zona {z_id} ({current_price:.2f}, jarak {dist:.2f} pt)"
                    new_alerts.append({"type": "APPROACHING", "zone_id": z_id, "msg": msg, "time": now_str})

            # E. Cek Conservative Confirmation (M5)
            if conservative_dec and conservative_dec.get("status") == "VALID":
                c_zone = conservative_dec.get("zone") or {}
                c_zid = c_zone.get("zone_id") if isinstance(c_zone, dict) else getattr(c_zone, "zone_id", None)
                if c_zid == z_id and rec["status"] != "CONFIRMED":
                    rec["status"] = "CONFIRMED"
                    rec["history"].append({"status": "CONFIRMED", "time": now_str, "reason": "Konfirmasi M5 terpenuhi (sweep, CHoCH, displacement, RR)"})
                    msg = f"[ALERT] Setup terkonfirmasi di zona {z_id}! (Trigger M5 terverifikasi)"
                    new_alerts.append({"type": "CONFIRMED", "zone_id": z_id, "msg": msg, "time": now_str})

        self.alerts.extend(new_alerts)
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]

        if self.auto_save and self.filepath:
            self.save()

        return new_alerts

    def get_tracked_zones(self, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        zones = list(self.tracked_zones.values())
        if status_filter:
            zones = [z for z in zones if z.get("status") == status_filter]
        return zones

    def get_active_zones(self) -> List[Dict[str, Any]]:
        """Mengembalikan zona yang masih aktif (belum INVALIDATED atau EXPIRED)."""
        return [z for z in self.tracked_zones.values() if z.get("status") not in ("INVALIDATED", "EXPIRED")]

    def get_recent_alerts(self, limit: int = 5) -> List[str]:
        return [a["msg"] for a in self.alerts[-limit:]]

    def reset(self):
        self.tracked_zones = {}
        self.alerts = []
        if self.filepath and os.path.exists(self.filepath):
            try:
                os.remove(self.filepath)
            except Exception:
                pass

    def save(self):
        if not self.filepath:
            return
        try:
            data = {"tracked_zones": self.tracked_zones, "alerts": self.alerts[-50:]}
            tmp = self.filepath + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(_clean(data), f, indent=2)
            os.replace(tmp, self.filepath)
        except Exception as e:
            _catat_error("zone_tracker_save", e)

    def load(self):
        if not self.filepath or not os.path.exists(self.filepath):
            return
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.tracked_zones = data.get("tracked_zones", {})
            self.alerts = data.get("alerts", [])
        except Exception as e:
            _catat_error("zone_tracker_load", e)


_ZONE_TRACKER: Optional[ZoneTracker] = None


def get_zone_tracker(auto_save: bool = False) -> ZoneTracker:
    global _ZONE_TRACKER
    if _ZONE_TRACKER is None:
        _ZONE_TRACKER = ZoneTracker(auto_save=auto_save)
    return _ZONE_TRACKER


# =============================================================================
# PIPELINE KEPUTUSAN: DATA -> H4 -> H1 -> M30 -> M5 -> RISK
# =============================================================================
def evaluate_trade_decision(frames: Optional[Dict[str, pd.DataFrame]] = None, current_price: Optional[float] = None,
                            spread: Optional[float] = None, news_blackout: Any = None,
                            server_time: Any = None, live: Optional[bool] = None) -> Dict[str, Any]:
    """
    Keputusan akhir VALID / WAIT / INVALID (fail-safe: exception, data kosong/terlambat, harga tak valid => WAIT).
    live=None: bila frames None, data/harga/spread/berita/waktu server diambil dari MT5.
    """
    dec: Dict[str, Any] = {"status": "WAIT", "direction": None, "reasons": [], "zone": None, "plan": None,
                           "stages": {}, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "symbol": SYMBOL}
    try:
        dec = _decide(dec, frames, current_price, spread, news_blackout, server_time, live)
    except Exception as e:
        _catat_error("evaluate_trade_decision", e)
        dec.update(status="WAIT", direction=None, plan=None, zone=None)
        dec["reasons"].append(f"Error internal scanner ({type(e).__name__}): fail-safe WAIT")
        dec["stages"]["ERROR"] = {"ok": False, "msg": repr(e)}
        _audit("entry_decision", _entry_audit(dec), _sig(dec["status"], dec["reasons"]))

    if "early_setup" not in dec:
        try:
            eff_p = current_price or dec.get("current_price") or 0.0
            dec["early_setup"] = detect_early_setups(frames=frames, current_price=eff_p, server_time=server_time)
            tracker = get_zone_tracker(auto_save=False)
            dec["zone_tracker"] = tracker
        except Exception:
            pass

    return dec


def _entry_audit(dec: Dict[str, Any]) -> Dict[str, Any]:
    st = dec.get("stages", {})
    plan = dec.get("plan") or {}
    return {"status": dec["status"], "arah": dec.get("direction"), "bias_H4": st.get("H4", {}).get("msg"),
            "konfirmasi_H1": st.get("H1", {}).get("msg"), "validasi_M30": st.get("M30", {}).get("msg"),
            "trigger_M5": st.get("M5", {}).get("msg"), "entry": plan.get("entry_acuan"), "sl": plan.get("sl"),
            "tp1": plan.get("tp1"), "tp2": plan.get("tp2"), "rr": plan.get("rr"), "alasan": dec["reasons"]}


def _decide(dec, frames, price, spread, news, server_time, live):
    st = dec["stages"]

    def _attach_early_and_tracker(dec_obj, scan_obj=None):
        try:
            eff_p = price if price is not None else (scan_obj.get("current_price") if scan_obj else 0.0)
            dec_obj["early_setup"] = detect_early_setups(frames=frames, current_price=eff_p, server_time=server_time, scan=scan_obj)
            tracker = get_zone_tracker(auto_save=False)
            h4_b = ctx.get("H4") if ("ctx" in locals() and ctx) else None
            m30_s = st.get("M30", {}).get("status")
            zones_to_track = scan_obj.get("active_zones", []) if scan_obj else ([dec_obj["zone"]] if dec_obj.get("zone") else [])
            tracker.update(zones_to_track, eff_p or 0.0, frames=frames, htf_bias=h4_b, m30_status=m30_s, server_time=server_time, conservative_dec=dec_obj)
            dec_obj["zone_tracker"] = tracker
        except Exception as e_att:
            _catat_error("attach_early_tracker", e_att)

    def finish(status, reason):
        dec["status"] = status
        if reason:
            dec["reasons"].append(reason)
        if status != "VALID":
            dec["plan"] = None
        for comp in ("zone_validation", "zone_touch", "liquidity_sweep", "structure_trigger", "displacement", "entry_validation", "risk_reward"):
            if comp not in st:
                st[comp] = {"status": "NOT_CHECKED", "msg": "Belum diperiksa"}

        # Attach radar priority information
        if "scan" in locals() and isinstance(scan, dict):
            all_r = scan.get("radar_zones") or scan.get("active_zones", [])
            h4_bias_val = ctx.get("H4", "NETRAL") if ("ctx" in locals() and ctx) else "NETRAL"
            h1_struct_val = ctx.get("H1", "NETRAL") if ("ctx" in locals() and ctx) else "NETRAL"
            ref_atr_val = 2.0
            if "frames" in locals() and frames and frames.get("M5") is not None and "ATR14" in frames["M5"].columns:
                ref_atr_val = float(frames["M5"]["ATR14"].iloc[-1])
            prio = prioritize_radar_zones(
                radar_zones=all_r,
                current_price=float(price or 0.0),
                htf_bias=h4_bias_val,
                h1_structure=h1_struct_val,
                atr=ref_atr_val,
                m5_stages=st,
            )
            dec["radar_priority"] = prio
            if status == "WAIT" and prio.get("dynamic_wait_reason"):
                dyn_w = prio["dynamic_wait_reason"]
                if dyn_w not in dec["reasons"]:
                    dec["reasons"].insert(0, dyn_w)

        _attach_early_and_tracker(dec, scan if ("scan" in locals() and isinstance(scan, dict)) else None)
        _audit("entry_decision", _entry_audit(dec), _sig(status, dec["reasons"], dec.get("direction")))
        return dec

    use_live = (frames is None) if live is None else live
    if frames is None:
        frames = _load_frames()
        if frames is None:
            st["DATA"] = {"ok": False, "msg": "MT5 tidak aktif"}
            return finish("WAIT", "MT5 tidak aktif / data tidak tersedia")
    if use_live:
        lc = _live_context()
        price = price if price is not None else lc["price"]
        spread = spread if spread is not None else lc["spread"]
        server_time = server_time if server_time is not None else lc["server_time"]
        if news is None:
            news = lc["news"]
            if lc["news_error"] and BLOKIR_JIKA_BERITA_GAGAL:
                return finish("WAIT", "Pemeriksaan berita gagal (kebijakan blokir aktif)")

    # --- DATA ---
    for tf in ("H4", "H1", "M30", "M5"):
        df = frames.get(tf)
        if df is None or len(df) < MIN_BARS_DECISION:
            st["DATA"] = {"ok": False, "msg": f"Data {tf} kosong/tidak cukup"}
            return finish("WAIT", f"Data {tf} kosong/tidak cukup")
    if not (isinstance(price, (int, float)) and math.isfinite(price) and price > 0):
        st["DATA"] = {"ok": False, "msg": "Harga tidak valid"}
        return finish("WAIT", "Harga tidak valid/tidak tersedia")

    stale = _stale_reason(frames, server_time)
    if stale:
        st["DATA"] = {"ok": False, "msg": stale}
        return finish("WAIT", stale)

    # Cek candle berjalan / belum selesai jika server_time disediakan (TASK C.7, H.12)
    if server_time is not None:
        for tf in ("M5", "M30", "H1", "H4"):
            df_tf = frames.get(tf)
            if df_tf is not None and len(df_tf) > 0:
                tf_sec = TF_MINUTES[tf] * 60
                last_ts = pd.Timestamp(df_tf.index[-1])
                st_ts = pd.Timestamp(server_time)
                diff = (st_ts - last_ts).total_seconds()
                if 0 <= diff < tf_sec - 1e-6 or df_tf.attrs.get("is_closed") is False:
                    st["DATA"] = {"ok": False, "msg": f"Candle terakhir {tf} belum selesai (masih berjalan)"}
                    return finish("WAIT", f"Candle terakhir {tf} belum selesai (masih berjalan): konfirmasi entry hanya dari candle close")
    st["DATA"] = {"ok": True, "msg": "Data H4/H1/M30/M5 lengkap & segar"}

    scan = scan_xauusd_zones(frames=frames, current_price=price)
    ctx = scan.get("htf_context") or evaluate_htf_context(frames)
    if not scan.get("data_ok"):
        return finish("WAIT", scan.get("empty_reason") or "Scanner tidak dapat berjalan")

    # --- H4: bias & zona utama ---
    h4 = ctx["H4"]
    if h4 not in ("BULLISH", "BEARISH"):
        st["H4"] = {"ok": False, "msg": f"Bias H4 {h4}"}
        return finish("WAIT", f"Bias H4 tidak jelas ({h4})")
    direction = "BUY" if h4 == "BULLISH" else "SELL"
    dec["direction"] = direction
    ztype = "DEMAND" if direction == "BUY" else "SUPPLY"
    if ctx["has_conflict"]:
        st["H1"] = {"ok": False, "msg": f"H1 {ctx['H1']} bertentangan dengan H4 {h4}"}
        st["H4"] = {"ok": True, "msg": f"Bias H4 {h4}"}
        return finish("WAIT", f"Konflik struktur H4 {h4} vs H1 {ctx['H1']}")
    cands = [z for z in scan["active_zones"] if z["zone_type"] == ztype]
    if not cands:
        other = [z for z in scan["active_zones"] if z["zone_type"] != ztype]
        st["H4"] = {"ok": False, "msg": f"Bias H4 {h4}, tidak ada zona {ztype} valid"}
        return finish("WAIT", scan.get("empty_reason") or (
            "Zona aktif hanya berlawanan dengan bias H4" if other else "Tidak ada zona valid"))
    def _cand_sort(z: Dict[str, Any]) -> Tuple:
        rank = {"A+": 0, "A": 1, "B": 2}
        tf_rank = 0 if "H4" in z.get("timeframe", "") and "H1" in z.get("timeframe", "") else (1 if "H4" in z.get("timeframe", "") else 2)
        fresh_rank = {"FRESH": 0, "TESTED_1X": 1, "TESTED_2X": 2}.get(z.get("freshness", ""), 9)
        # Prioritas teknikal: Grade, Skor, Timeframe HTF, Freshness, Displacement, ID deterministik
        return (
            rank.get(z.get("grade", ""), 9),
            -round(float(z.get("score", 0.0)), 1),
            tf_rank,
            fresh_rank,
            -round(float(z.get("displacement_strength", 0.0)), 2),
            str(z.get("zone_id", "")),
        )

    cands.sort(key=_cand_sort)
    best_candidate_dict = cands[0]
    best_candidate_zone = _zone_from_dict(best_candidate_dict)
    dec["zone"] = best_candidate_dict

    # --- H1: konfirmasi (struktur tidak bertentangan, zona tidak ditembus close H1) ---
    st["H1"] = {"ok": True, "msg": f"H1 {ctx['H1']} tidak bertentangan dengan H4 {h4}; invalidasi aman"}

    # --- M30: keselarasan & momentum (TASK Section D) ---
    m30_val = evaluate_m30_momentum(frames.get("M30"), h4)
    st["M30"] = m30_val
    if ctx.get("m30_conflict") or m30_val["status"] == "CONFLICT":
        st["M30"] = {"ok": False, "status": "CONFLICT", "msg": m30_val.get("msg") or f"M30 bertentangan dengan H4 {h4}"}
        return finish("WAIT", f"M30 bertentangan dengan struktur utama H4 {h4}")
    if m30_val["status"] == "INVALID":
        st["M30"] = {"ok": False, "status": "INVALID", "msg": m30_val["msg"]}
        return finish("WAIT", f"M30 INVALID: {m30_val['msg']}")
    if m30_val["status"] == "WEAK":
        st["M30"] = {"ok": False, "status": "WEAK", "msg": m30_val["msg"]}
        return finish("WAIT", f"M30 tidak memiliki momentum (status WEAK: {m30_val['msg']})")

    st["M30"] = {"ok": True, "status": "CONFIRMED", "msg": m30_val["msg"]}

    # Evaluasi zona kandidat berurutan dari yang paling berkualitas tinggi:
    # Zona pertama yang memenuhi seluruh validasi M5 + Risk akan menghasilkan sinyal VALID
    valid_plan = None
    winning_zone = None
    winning_zdict = None
    first_reasons = []

    for zd in cands:
        zn = _zone_from_dict(zd)
        inval = zn.invalidation_price
        bos_t = pd.Timestamp(zd["bos_candle_time"]) if zd.get("bos_candle_time") else None

        def closed_beyond(df):
            sub = df[df.index > bos_t] if bos_t is not None else df
            c = sub["Close"].values
            return len(c) > 0 and ((c < inval).any() if ztype == "DEMAND" else (c > inval).any())

        if closed_beyond(frames["H1"]):
            if zd == cands[0]:
                st["H1"] = {"ok": False, "msg": "Close H1 menembus invalidasi zona"}
                return finish("INVALID", f"Zona invalid: close H1 menembus {inval:.2f}")
            continue

        if closed_beyond(frames["M30"]):
            if zd == cands[0]:
                st["M30"] = {"ok": False, "status": "INVALID", "msg": "Close M30 menembus invalidasi zona"}
                return finish("INVALID", f"Zona invalid: close M30 menembus {inval:.2f}")
            continue

        r = _entry_check(zn, float(price), frames["M5"], spread if spread is not None else 0.0, news, frames, server_time=server_time)
        if not first_reasons:
            first_reasons = r["reasons"]
            for k_st, v_st in r.get("stages", {}).items():
                st[k_st] = v_st
        if r["status"] == "VALID":
            valid_plan = r["plan"]
            winning_zone = zn
            winning_zdict = zd
            dec["reasons"].extend(r["reasons"])
            for k_st, v_st in r.get("stages", {}).items():
                st[k_st] = v_st
            break

    if valid_plan and winning_zone and winning_zdict:
        dec["zone"] = winning_zdict
        dec["plan"] = valid_plan
        st["H4"] = {"ok": True, "msg": f"Bias H4 {h4}; zona {winning_zone.grade} {winning_zone.timeframe} {winning_zone.zone_type} {winning_zone.price_low:.2f}-{winning_zone.price_high:.2f}"}
        st["M5"] = {"ok": True, "msg": "; ".join(dec["reasons"])}
        st["RISK"] = {"ok": True, "msg": f"R:R 1:{valid_plan.get('rr', '-')}"}
        return finish("VALID", "")

    # Jika tidak ada yang lolos, gunakan zona terbaik untuk pelaporan status WAIT
    dec["zone"] = best_candidate_dict
    dec["reasons"].extend(first_reasons)
    st["H4"] = {"ok": True, "msg": f"Bias H4 {h4}; zona {best_candidate_zone.grade} {best_candidate_zone.timeframe} {best_candidate_zone.zone_type} {best_candidate_zone.price_low:.2f}-{best_candidate_zone.price_high:.2f}"}
    st["M5"] = {"ok": False, "msg": "; ".join(first_reasons)}
    if "risk_reward" not in st or st["risk_reward"].get("status") == "NOT_CHECKED":
        st["RISK"] = {"ok": False, "msg": "Menunggu konfirmasi entry M5 & R:R"}
    else:
        st["RISK"] = {"ok": False, "msg": st["risk_reward"].get("msg", "Menunggu R:R")}
    for comp in ("zone_validation", "zone_touch", "liquidity_sweep", "structure_trigger", "displacement", "entry_validation", "risk_reward"):
        if comp not in st:
            st[comp] = {"status": "NOT_CHECKED", "msg": "Belum diperiksa"}
    return finish("WAIT", "")


def _zone_from_dict(d: Dict[str, Any]) -> Zone:
    keys = Zone.__dataclass_fields__.keys()
    return Zone(**{k: d[k] for k in keys if k in d})


def format_diagnostic_report(scan: Optional[Dict[str, Any]] = None) -> str:
    """Format laporan diagnostik komprehensif scanner XAUUSD (TASK Section 5)."""
    if scan is None:
        scan = scan_xauusd_zones()

    if not scan.get("data_ok"):
        return (
            "=" * 70 + "\n"
            "XAUUSD ZONE SCANNER DIAGNOSTIC REPORT\n"
            "=" * 70 + "\n"
            f"Status Data        : TIDAK VALID / OFFLINE\n"
            f"Alasan             : {scan.get('empty_reason', 'Data tidak tersedia')}\n"
            "=" * 70
        )

    diag = scan.get("diagnostics") or {}
    ctx = scan.get("htf_context") or {}
    price = scan.get("current_price", 0.0)
    ts = scan.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    sym = scan.get("symbol", SYMBOL)

    by_tf = diag.get("by_timeframe", {})
    tf_summary_parts = []
    for tf_key in ("H4", "H1", "M30"):
        if tf_key in by_tf:
            tf_summary_parts.append(f"{tf_key}: {by_tf[tf_key].get('bars', 0)} bars")
    tf_bars_str = " (" + ", ".join(tf_summary_parts) + ")" if tf_summary_parts else ""

    lines = [
        "=" * 70,
        "XAUUSD ZONE SCANNER DIAGNOSTIC REPORT",
        "=" * 70,
        f"Timestamp          : {ts}",
        f"Broker / Symbol    : {sym}",
        f"Current Price      : {price:.2f}",
        "",
        "1. DATA & STRUKTUR PASAR:",
        f"- Candle Diperiksa : {diag.get('candles_checked', 0)} bars total{tf_bars_str}",
        f"- Swing Ditemukan  : {diag.get('swings_found', 0)} swings ({diag.get('swings_high', 0)} High, {diag.get('swings_low', 0)} Low)",
        f"- BOS / CHoCH      : {diag.get('bos_choch_count', 0)} event struktural teridentifikasi",
        f"- FVG Imbalance    : {diag.get('fvg_count', 0)} FVG terdeteksi",
        "",
        "2. DETEKSI & VALIDASI ZONA:",
        f"- Kandidat OB      : {diag.get('ob_candidates_count', 0)} leg impuls diperiksa",
        f"- Total Terdeteksi : {diag.get('zones_detected', 0)} zona (Radar: {diag.get('zones_radar', 0)}, Valid: {diag.get('zones_valid_active', 0)})",
        f"- Lolos Hard Valid : {diag.get('zones_passed_hard_validation', 0)} zona",
        f"- Lolos Scoring    : {diag.get('zones_passed_scoring', 0)} zona (Skor >= {SCORE_MIN_ACTIVE})",
        f"- Zona Valid Aktif : {diag.get('zones_valid_active', 0)} zona (setelah merge overlap)",
        "",
        "3. BREAKDOWN ZONA PER TIMEFRAME (H4, H1, M30):",
    ]

    if by_tf:
        for tf_k in ("H4", "H1", "M30"):
            if tf_k in by_tf:
                tf_info = by_tf[tf_k]
                lines.append(f"  • {tf_k} : {tf_info.get('ob_candidates', 0)} OB candidates ({tf_info.get('hard_pass', 0)} lolos hard valid, {tf_info.get('hard_reject', 0)} ditolak)")
    else:
        lines.append("  • Belum ada data per timeframe.")

    lines.extend([
        "",
        "4. BREAKDOWN FILTER & PENOLAKAN:",
        f"- Zona Invalidated (Tembus Close) : {diag.get('zones_invalidated', 0)} zona",
        f"- Zona Termitigasi (> 2 Sentuhan) : {diag.get('zones_mitigated', 0)} zona",
        f"- Zona Terlalu Jauh (> 35 ATR)   : {diag.get('zones_too_far', 0)} zona",
        f"- Total Zona Ditolak / Gagal Valid: {diag.get('zones_rejected', 0)} kandidat",
    ])

    reasons = diag.get("rejected_reasons_breakdown", {})
    if reasons:
        lines.append("  Rincian alasan penolakan kandidat:")
        for r_name, r_cnt in sorted(reasons.items(), key=lambda x: -x[1])[:8]:
            lines.append(f"    * {r_name}: {r_cnt}x")

    iz_log = diag.get("inspected_zones_log", [])
    if iz_log:
        lines.append("")
        lines.append("5. AUDIT LOG ZONA MULTI-TIMEFRAME (H4/H1/M30):")
        for rz in iz_log[:10]:
            st_flag = "LOLOS" if rz["is_valid"] else f"DITOLAK: {rz['rejection_reason']}"
            fvg_st = rz.get("fvg_status", "-")
            lines.append(f"  • [{rz['source_timeframe']}] {rz['zone_type']} {rz['price_low']:.2f}-{rz['price_high']:.2f} "
                         f"[Grade: {rz['grade']}, Skor: {rz['score']:.1f}, FVG: {fvg_st}, Touch: {rz['touch_count']}x, Jarak: {rz['distance']:.2f} pt ({rz['direction_status']})] -> {st_flag}")

    conflict_lbl = "CONFLICT" if diag.get("htf_conflict") else "ALIGNED"
    lines.extend([
        "",
        "6. STATUS HTF & KONFLIK:",
        f"- H4 Bias          : {ctx.get('H4', 'NETRAL')}",
        f"- H1 Structure     : {ctx.get('H1', 'NETRAL')}",
        f"- M30 Momentum     : {ctx.get('M30', 'NETRAL')}",
        f"- Status Konflik   : {conflict_lbl} ({ctx.get('bias_utama', '-')})",
        "=" * 70,
    ])

    return "\n".join(lines)


def format_scanner_audit(dec: Dict[str, Any]) -> str:
    """Format audit log standar [SCANNER] sesuai spesifikasi audit trading (TASK Section E)."""
    status = dec.get("status", "WAIT")
    direction = dec.get("direction")
    stages = dec.get("stages", {})
    plan = dec.get("plan") or {}

    lines = []
    # 1. H4 Bias
    h4_stage = stages.get("H4", {})
    h4_msg = h4_stage.get("msg", "-")
    if h4_stage.get("ok"):
        h4_bias = "BULLISH" if "BULLISH" in h4_msg.upper() else ("BEARISH" if "BEARISH" in h4_msg.upper() else "NETRAL")
    elif "H4" in stages:
        h4_bias = "UNCONFIRMED"
    else:
        h4_bias = "NOT_CHECKED"
    lines.append(f"[SCANNER] H4 Bias: {h4_bias}")

    # 2. H1 Structure
    h1_stage = stages.get("H1", {})
    if h1_stage.get("ok"):
        h1_str = "BULLISH" if direction == "BUY" else ("BEARISH" if direction == "SELL" else "ALIGNED")
    elif "H1" in stages:
        h1_str = "CONFLICT" if ("bertentangan" in h1_stage.get("msg", "").lower() or "konflik" in h1_stage.get("msg", "").lower()) else "UNCONFIRMED"
    else:
        h1_str = "NOT_CHECKED"
    lines.append(f"[SCANNER] H1 Structure: {h1_str}")

    # 3. M30 Momentum
    m30_stage = stages.get("M30", {})
    if "status" in m30_stage:
        m30_status = m30_stage["status"]
    elif m30_stage.get("ok"):
        m30_status = "CONFIRMED"
    elif "M30" in stages:
        m30_status = "UNCONFIRMED"
    else:
        m30_status = "NOT_CHECKED"
    lines.append(f"[SCANNER] M30 Momentum: {m30_status}")

    # 4. Zone Validation
    zv_stage = stages.get("zone_validation", {})
    zv_status = zv_stage.get("status")
    if not zv_status:
        zv_status = "CONFIRMED" if status == "VALID" else ("NOT_CHECKED" if "zone_validation" not in stages else "WAIT")
    lines.append(f"[SCANNER] Zone Validation: {zv_status}")

    # 5. Zone Touch
    zt_stage = stages.get("zone_touch", {})
    zt_status = zt_stage.get("status")
    if not zt_status:
        zt_status = "CONFIRMED" if status == "VALID" else ("NOT_CHECKED" if "zone_touch" not in stages else "WAIT")
    lines.append(f"[SCANNER] Zone Touch: {zt_status}")

    # 6. Liquidity Sweep
    sw_stage = stages.get("liquidity_sweep", {})
    sw_status = sw_stage.get("status")
    if not sw_status:
        if status == "VALID":
            sw_status = "CONFIRMED"
        elif "liquidity_sweep" in stages:
            sw_status = "NOT CONFIRMED"
        else:
            sw_status = "NOT_CHECKED"
    elif sw_status == "WAIT":
        sw_status = "NOT CONFIRMED"
    lines.append(f"[SCANNER] M5 Liquidity Sweep: {sw_status}")

    # 7. CHoCH / BOS / CISD
    tr_stage = stages.get("structure_trigger", {})
    tr_status = tr_stage.get("status")
    if not tr_status:
        tr_status = "CONFIRMED" if status == "VALID" else ("NOT_CHECKED" if "structure_trigger" not in stages else "WAIT")
    lines.append(f"[SCANNER] CHoCH / BOS / CISD: {tr_status}")

    # 8. Displacement
    dp_stage = stages.get("displacement", {})
    dp_status = dp_stage.get("status")
    if not dp_status:
        dp_status = "CONFIRMED" if status == "VALID" else ("NOT_CHECKED" if "displacement" not in stages else "NOT CONFIRMED")
    elif dp_status in ("WAIT", "WEAK"):
        dp_status = "WEAK" if dp_stage.get("status") == "WEAK" else "NOT CONFIRMED"
    lines.append(f"[SCANNER] M5 Displacement: {dp_status}")

    # 9. Entry Validation
    ev_stage = stages.get("entry_validation", {})
    ev_status = ev_stage.get("status")
    if not ev_status:
        ev_status = "PASSED" if status == "VALID" else ("INVALID" if status == "INVALID" else "WAIT")
    lines.append(f"[SCANNER] Entry Validation: {ev_status}")

    # 10. Risk / Reward & Final Decision
    rr_stage = stages.get("risk_reward", {})
    rr_val = plan.get("rr") or rr_stage.get("rr")
    if status == "VALID" and rr_val:
        lines.append(f"[SCANNER] Risk Reward: 1:{rr_val:.1f}")
        lines.append(f"[SCANNER] Final Decision: {direction} VALID")
    else:
        rr_status = rr_stage.get("status", "NOT_CHECKED" if "risk_reward" not in stages else "WAIT")
        lines.append(f"[SCANNER] Risk Reward: {rr_status}")
        lines.append(f"[SCANNER] Final Decision: {status}")
        reasons = [r.replace("[SCANNER] Rejection Reason: ", "") for r in dec.get("reasons", []) if r]
        rej_msg = reasons[0] if reasons else "Persyaratan entry belum terpenuhi"
        lines.append(f"[SCANNER] Rejection Reason: {rej_msg}")

    return "\n".join(lines)


def format_scan_presentation(
    scan: Optional[Dict[str, Any]] = None,
    dec: Optional[Dict[str, Any]] = None,
    early_setups: Optional[Dict[str, Any]] = None,
    tracker: Optional[Any] = None,
    detailed: bool = False,
) -> str:
    """
    Format ringkas visual command 'scan' sesuai aturan Otoritas Keputusan Python Scanner:
    5 Bagian Utama:
      A. KONDISI PASAR (Snapshot harga, sumber tick/close, status multi-TF bias)
      B. RADAR ZONE (Primary Directional Zone + Nearest Radar Zone; kandidat hanya tampil jika detailed=True)
      C. ENTRY PLAN (Early Pending Plan vs Confirmed Entry M5)
      D. FINAL DECISION & REASON (Keputusan resmi Python Scanner + dynamic reason)
      E. RISK WARNING (Peringatan singkat MM, pending order & kewajiban SL)
    """
    if scan is None:
        scan = scan_xauusd_zones()
    if dec is None:
        dec = evaluate_trade_decision()

    price = scan.get("current_price") or dec.get("current_price") or 0.0
    ctx = scan.get("htf_context") or {}
    h4_bias = ctx.get("H4", "NETRAL")
    h1_struct = ctx.get("H1", "NETRAL")
    m30_val = dec.get("stages", {}).get("M30", {})
    m30_status = m30_val.get("status", "CONFIRMED" if m30_val.get("ok") else "UNCONFIRMED")

    zone = dec.get("zone")
    if not zone and scan.get("active_zones"):
        zone = scan["active_zones"][0]

    stages = dec.get("stages", {})
    status = dec.get("status", "WAIT")
    direction = dec.get("direction")
    plan = dec.get("plan")

    # Early Setups (Aggressive Pending)
    if early_setups is None:
        early_setups = dec.get("early_setup")
    if early_setups is None:
        try:
            early_setups = detect_early_setups(current_price=price, scan=scan)
        except Exception:
            early_setups = {"best_setup": None, "all_setups": [], "empty_reason": "Data tidak tersedia"}

    # Zone Tracker
    if tracker is None:
        tracker = dec.get("zone_tracker")
    if tracker is None:
        try:
            tracker = get_zone_tracker(auto_save=False)
            tracker.update(scan.get("active_zones", []), price, htf_bias=h4_bias, m30_status=m30_status, conservative_dec=dec)
        except Exception:
            tracker = None

    # M5 Trigger status summary
    zt = stages.get("zone_touch", {}).get("status", "CONFIRMED" if status == "VALID" else "WAIT")
    sw = stages.get("liquidity_sweep", {}).get("status", "CONFIRMED" if status == "VALID" else "WAIT")
    tr = stages.get("structure_trigger", {}).get("status", "CONFIRMED" if status == "VALID" else "WAIT")
    dp = stages.get("displacement", {}).get("status", "CONFIRMED" if status == "VALID" else "WAIT")
    rr_st = stages.get("risk_reward", {}).get("status", "CONFIRMED" if (status == "VALID" and plan) else "WAIT")

    sw_desc = stages.get("liquidity_sweep", {}).get("msg")
    tr_desc = stages.get("structure_trigger", {}).get("msg")
    dp_desc = stages.get("displacement", {}).get("msg")

    if status == "VALID":
        m5_trig_str = f"CONFIRMED ({direction} Trigger Valid)"
    elif sw != "CONFIRMED":
        m5_trig_str = "WAIT (Belum ada Sweep M5 yang relevan)"
    elif tr != "CONFIRMED":
        m5_trig_str = "WAIT (Menunggu CHoCH/BOS/CISD M5)"
    else:
        m5_trig_str = "WAIT (Menunggu konfirmasi M5)"

    out = []
    out.append("=" * 70)
    out.append("XAUUSD PYTHON SMC SCANNER")
    out.append("=" * 70)
    out.append("")

    # A. KONDISI PASAR (MARKET OVERVIEW & BIAS)
    out.append("A. KONDISI PASAR — 2. MARKET BIAS (MARKET OVERVIEW)")
    h4_diag = ctx.get("h4_diagnostic") or ""
    if h4_bias == "NETRAL" and h4_diag:
        h4_str = f"{h4_bias} ({h4_diag})"
    else:
        h4_str = f"{h4_bias}"

    snapshot_time = scan.get("snapshot_time") or scan.get("timestamp") or datetime.now().strftime("%Y-%m-%d %H:%M:%S WIB")
    bid_val = scan.get("bid")
    ask_val = scan.get("ask")
    bid_ask_str = f" | Bid: {bid_val:.2f} | Ask: {ask_val:.2f}" if (bid_val is not None and ask_val is not None and bid_val > 0) else ""
    price_src = scan.get("price_source") or ("MT5 Live Tick (Bid/Ask) — Presisi Jarak terhadap Proximal Edge" if bid_val else f"Candle Close Snapshot ({price:.2f})")

    conflict_tag = " [CONFLICT H4 vs H1]" if ctx.get("has_conflict") else ""
    out.append(f"- Current Price    : {price:.2f} (Snapshot: {snapshot_time}{bid_ask_str})" if price > 0 else "- Current Price    : N/A")
    out.append(f"- Sumber Harga     : {price_src}")
    out.append(f"- Multi-TF Bias    : H4: {h4_bias} | H1: {h1_struct}{conflict_tag} | M30: {m30_status} | M5: {'CONFIRMED' if status == 'VALID' else 'WAIT'}")
    out.append(f"- H4 Bias          : {h4_str}")
    out.append(f"- H1 Structure     : {h1_struct}")
    out.append(f"- M30 Momentum     : {m30_status}")
    out.append(f"- M5 Trigger       : {m5_trig_str}")
    out.append("")

    # B. RADAR ZONE (ZONA PEMANTAUAN & ZONE ANALYSIS)
    out.append("B. RADAR ZONE — 3. RADAR ZONE (ZONA PEMANTAUAN & ZONE ANALYSIS)")
    all_radar = scan.get("radar_zones") or scan.get("active_zones", [])
    if not all_radar:
        out.append("  NO RADAR ZONE DETECTED.")
    else:
        # Sistem Prioritas Radar Zone
        r_prio = dec.get("radar_priority") or scan.get("radar_priority")
        if not r_prio:
            r_prio = prioritize_radar_zones(all_radar, price, htf_bias=h4_bias, h1_structure=h1_struct, m5_stages=stages)

        pz = r_prio.get("primary_directional_zone") or r_prio.get("primary_zone")
        nz = r_prio.get("nearest_radar_zone")
        bz = r_prio.get("best_entry_monitoring_zone")
        sz = r_prio.get("secondary_zone")
        cz = r_prio.get("countertrend_zone")

        out.append("  [3 KATEGORI PRIORITAS RADAR ZONE — SELEKSI RESMI PYTHON SCANNER]:")
        if pz:
            pz_lbl = f"[{pz.get('status', 'RADAR')}]" + (f" [{pz.get('approach')}]" if pz.get('approach') in ("FAR", "INSIDE_ZONE") else "")
            pz_dir = pz.get("direction_status") or ("ABOVE_PRICE" if pz["price_low"] > price else ("BELOW_PRICE" if pz["price_high"] < price else "INSIDE_ZONE"))
            fresh_str = f"Freshness: {pz.get('freshness', 'FRESH')} | " if pz.get('freshness') else ""
            out.append(f"  • A. PRIMARY DIRECTIONAL ZONE (Primary Radar Zone) : {pz['timeframe']} {pz['zone_type']} {pz['price_low']:.2f} - {pz['price_high']:.2f} [Grade: {pz['grade']} | Skor: {pz.get('score', 0.0):.1f}] | {fresh_str}Jarak: {pz['distance']:.2f} pt ({pz_dir}) | Status: {pz_lbl}")
            out.append(f"    - Alasan Directional                             : {r_prio.get('primary_reason', '-')}")
        else:
            out.append(f"  • A. PRIMARY DIRECTIONAL ZONE (Primary Radar Zone) : - (Tidak ada zona yang selaras dengan bias {h4_bias})")

        if nz:
            nz_lbl = f"[{nz.get('status', 'RADAR')}]" + (" [COUNTERTREND]" if nz.get("is_counter") else "") + (f" [{nz.get('approach')}]" if nz.get('approach') in ("FAR", "INSIDE_ZONE") else "")
            nz_dir = nz.get("direction_status") or ("ABOVE_PRICE" if nz["price_low"] > price else ("BELOW_PRICE" if nz["price_high"] < price else "INSIDE_ZONE"))
            nz_role = " (Countertrend/Support - pantau reaksi, bukan sinyal)" if nz.get("is_counter") else ""
            fresh_nz = f"Freshness: {nz.get('freshness', 'FRESH')} | " if nz.get('freshness') else ""
            out.append(f"  • B. NEAREST RADAR ZONE                             : {nz['timeframe']} {nz['zone_type']} {nz['price_low']:.2f} - {nz['price_high']:.2f} [Grade: {nz['grade']} | Skor: {nz.get('score', 0.0):.1f}] | {fresh_nz}Jarak: {nz['distance']:.2f} pt ({nz_dir}) | Status: {nz_lbl}{nz_role}")
        else:
            out.append("  • B. NEAREST RADAR ZONE                             : - (Tidak ada zona radar aktif)")

        if cz and cz != nz:
            cz_dir = cz.get("direction_status") or ("ABOVE_PRICE" if cz["price_low"] > price else ("BELOW_PRICE" if cz["price_high"] < price else "INSIDE_ZONE"))
            opp_act = "BUY" if cz['zone_type'] == "DEMAND" else "SELL"
            opp_role = "support/pantau reaksi" if cz['zone_type'] == "DEMAND" else "resistance/pantau reaksi"
            out.append(f"  • C. COUNTERTREND / REACTION ZONE                   : {cz['timeframe']} {cz['zone_type']} {cz['price_low']:.2f} - {cz['price_high']:.2f} [Grade: {cz['grade']}] | Jarak: {cz['distance']:.2f} pt ({cz_dir}) | Status: [RADAR] [COUNTERTREND]")
            out.append(f"    - Catatan Countertrend                            : Area {opp_role} (bukan sinyal {opp_act} karena melawan bias HTF).")

        if detailed:
            if bz and bz != pz and bz != nz and bz != cz:
                bz_lbl = f"[{bz.get('status', 'RADAR')}]" + (f" [{bz.get('approach')}]" if bz.get('approach') in ("FAR", "INSIDE_ZONE") else "")
                bz_dir = bz.get("direction_status") or ("ABOVE_PRICE" if bz["price_low"] > price else ("BELOW_PRICE" if bz["price_high"] < price else "INSIDE_ZONE"))
                out.append(f"  • Best Entry Monitoring                             : {bz['timeframe']} {bz['zone_type']} {bz['price_low']:.2f} - {bz['price_high']:.2f} [Grade: {bz['grade']}] | Jarak: {bz['distance']:.2f} pt ({bz_dir}) | Status: {bz_lbl}")
                out.append(f"    - Alasan Monitoring                               : Zona pro-trend optimal: Grade {bz['grade']} (Skor {bz['score']:.1f}), {bz['freshness']}, jarak {bz['distance']:.2f} pt ({bz_dir}).")

            if sz and sz != bz and sz != pz and sz != nz and sz != cz:
                sz_lbl = f"[{sz.get('status', 'RADAR')}]" + (f" [{sz.get('approach')}]" if sz.get('approach') == "FAR" else "")
                out.append(f"  • Secondary Radar Zone                              : {sz['timeframe']} {sz['zone_type']} {sz['price_low']:.2f} - {sz['price_high']:.2f} [Grade: {sz['grade']}] | Jarak: {sz['distance']:.2f} pt | Status: {sz_lbl}")

            out.append("")
            out.append("  [Kandidat Zona Pemantauan — BUKAN Sinyal Entry / Pending Order Aktif]:")
            demands = [z for z in all_radar if z.get("zone_type") == "DEMAND"]
            supplies = [z for z in all_radar if z.get("zone_type") == "SUPPLY"]
            has_conflict = ctx.get("has_conflict", False)

            if demands:
                for zd in demands[:3]:
                    z_lo, z_hi = float(zd.get("price_low", 0.0)), float(zd.get("price_high", 0.0))
                    d_val, d_st = calculate_zone_distance(z_lo, z_hi, price, digits=2)
                    inval = float(zd.get("invalidation_price", z_lo))
                    tf = zd.get("timeframe", "H1")
                    grd = zd.get("grade", "A")
                    z_st = zd.get("status", "RADAR")
                    fvg_c = zd.get("fvg_confirmation") or {}
                    fvg_str = f"FVG: {fvg_c.get('status', '-')}"
                    ob_c = zd.get("origin_bos") or {}
                    ob_str = f"OB: {ob_c.get('tipe', 'OB')}"
                    is_far = (d_val > max(35.0, 15.0 * 2.0))
                    far_lbl = " [FAR]" if is_far else ""
                    ct_lbl = " [COUNTERTREND]" if (h4_bias == "BEARISH") else ""
                    st_lbl = f"[{z_st}]{far_lbl}{ct_lbl} " + ("CONFLICT H4 vs H1" if has_conflict else (f"Jauh {d_val:.1f} pt ({d_st})" if is_far else f"Terpantau ({d_st})"))
                    out.append(f"- Kandidat Demand  : {tf} DEMAND {z_lo:.2f} - {z_hi:.2f} [Grade: {grd}] | Jarak: {d_val:.2f} pt ({d_st}) | Invalidation: {inval:.2f} | {ob_str} | {fvg_str} | Status: {st_lbl}")
            else:
                out.append("- Kandidat Demand  : Tidak ada zona Demand aktif yang memenuhi syarat")

            if supplies:
                for zs_z in supplies[:3]:
                    z_lo, z_hi = float(zs_z.get("price_low", 0.0)), float(zs_z.get("price_high", 0.0))
                    d_val, d_st = calculate_zone_distance(z_lo, z_hi, price, digits=2)
                    inval = float(zs_z.get("invalidation_price", z_hi))
                    tf = zs_z.get("timeframe", "H1")
                    grd = zs_z.get("grade", "A")
                    z_st = zs_z.get("status", "RADAR")
                    fvg_c = zs_z.get("fvg_confirmation") or {}
                    fvg_str = f"FVG: {fvg_c.get('status', '-')}"
                    ob_c = zs_z.get("origin_bos") or {}
                    ob_str = f"OB: {ob_c.get('tipe', 'OB')}"
                    is_far = (d_val > max(35.0, 15.0 * 2.0))
                    far_lbl = " [FAR]" if is_far else ""
                    ct_lbl = " [COUNTERTREND]" if (h4_bias == "BULLISH") else ""
                    st_lbl = f"[{z_st}]{far_lbl}{ct_lbl} " + ("CONFLICT H4 vs H1" if has_conflict else (f"Jauh {d_val:.1f} pt ({d_st})" if is_far else f"Terpantau ({d_st})"))
                    out.append(f"- Kandidat Supply  : {tf} SUPPLY {z_lo:.2f} - {z_hi:.2f} [Grade: {grd}] | Jarak: {d_val:.2f} pt ({d_st}) | Invalidation: {inval:.2f} | {ob_str} | {fvg_str} | Status: {st_lbl}")
            else:
                out.append("- Kandidat Supply  : Tidak ada zona Supply aktif yang memenuhi syarat")
        else:
            out.append("  (Ketik 'scan detail' untuk melihat rincian seluruh kandidat zona)")
    out.append("")

    # C. ENTRY PLAN (ENTRY VALIDATION & OTORITAS SCANNER)
    out.append("C. ENTRY PLAN — 4. ENTRY PLAN (ENTRY VALIDATION & OTORITAS SCANNER)")
    best_p = (early_setups or {}).get("best_setup")

    # A. EARLY PENDING PLAN
    out.append("A. EARLY PENDING PLAN (Strategi Agresif - Risiko Lebih Tinggi):")
    if best_p and best_p.get("valid") and h4_bias in ("BULLISH", "BEARISH") and m30_status not in ("CONFLICT", "INVALID", "WEAK"):
        out.append("   * Status         : READY — PENDING PLAN (BELUM DIPASANG DI MT5)")
        out.append("   * EARLY_PENDING_STATUS : READY")
        out.append(f"   * Order Type     : {best_p['order_type']}")
        out.append(f"   * Entry Price    : {best_p['entry_price']:.2f} ({best_p['entry_reason']})")
        out.append(f"   * Stop Loss      : {best_p['sl']:.2f} (Risk: {best_p['risk']:.2f} pt)")
        out.append(f"   * Take Profit 1  : {best_p['tp1']:.2f} (Reward: {best_p['reward']:.2f} pt)")
        out.append(f"   * Take Profit 2  : {best_p['tp2']:.2f}")
        out.append(f"   * Risk Points    : {best_p.get('risk_points', best_p.get('risk', 0.0)):.2f} pt")
        out.append(f"   * Reward TP1     : {best_p.get('reward_tp1', best_p.get('reward', 0.0)):.2f} pt")
        out.append(f"   * Reward TP2     : {best_p.get('reward_tp2', 0.0):.2f} pt")
        out.append(f"   * EARLY_PENDING_RR (TP1) : PASSED (1:{best_p.get('rr_tp1', best_p.get('rr', 0.0)):.1f})")
        out.append(f"   * EARLY_PENDING_RR (TP2) : 1:{best_p.get('rr_tp2', 0.0):.1f}")
        out.append(f"   * Risk/Reward    : 1:{best_p['rr']:.1f}")
        out.append(f"   * Invalidation   : {best_p['invalidation_price']:.2f}")
        out.append(f"   * Structural Invalidation : {best_p['invalidation_price']:.2f}")
        out.append(f"   * Invalidation Buffer : {best_p.get('invalidation_buffer', 0.0):.2f} pt")
        out.append(f"   * Zone Boundary  : {best_p.get('zone_low', 0.0):.2f} - {best_p.get('zone_high', 0.0):.2f}")
        out.append(f"   * Suggested Exp  : {best_p['suggested_expiration']}")
        out.append("   * Keterangan     : BELUM DIPASANG DI MT5 (Rencana pending order agresif; eksekusi manual oleh trader).")
        out.append("   * Peringatan     : Setup agresif sebelum konfirmasi M5; pasang limit manual hanya jika siap menanggung risiko.")
        if detailed:
            out.append("   * Penjelasan Level:")
            out.append(f"     - Zone Boundary           : Area reaksi institusional ({best_p.get('zone_low', 0.0):.2f} - {best_p.get('zone_high', 0.0):.2f}) tempat order berpotensi terisi.")
            out.append(f"     - Structural Invalidation : Level ({best_p['invalidation_price']:.2f}) di mana struktur zona resmi batal jika tertembus.")
            out.append(f"     - Stop Loss               : Level proteksi modal ({best_p['sl']:.2f}) yang diletakkan di luar Invalidation dengan buffer ({best_p.get('invalidation_buffer', 0.0):.2f} pt) untuk menghindari market noise/spread.")
    else:
        rej_early = (early_setups or {}).get("empty_reason") or (best_p.get("rejection_reason") if best_p else "Syarat pending order belum terpenuhi")
        if h4_bias == "NETRAL":
            rej_early = "H4 belum memberikan directional bias yang jelas (NETRAL)"
        out.append(f"   * Status         : TIDAK AKTIF ({rej_early})")
        out.append("   * EARLY_PENDING_STATUS : TIDAK_AKTIF")
        out.append("   * EARLY_PENDING_RR : DATA_UNAVAILABLE")
        out.append("   * Recommendation : NO PENDING SETUP (WAIT)")
        out.append("   * Entry Price    : -")
        out.append("   * Stop Loss      : -")
        out.append("   * Take Profit    : -")
        out.append("   * Risk/Reward    : -")

    out.append("")
    # B. CONFIRMED ENTRY (CONSERVATIVE M5)
    out.append("B. CONFIRMED ENTRY (Strategi Konservatif - Sinyal Resmi Terkonfirmasi):")
    if status == "VALID" and plan:
        confirmed_rr_val = f"PASSED (1:{plan.get('rr', 0.0):.1f})"
    elif tr == "CONFIRMED" and dp == "CONFIRMED" and rr_st == "CONFIRMED":
        confirmed_rr_val = "PASSED"
    elif tr == "CONFIRMED" and dp == "CONFIRMED" and (rr_st in ("WAIT", "FAILED")):
        confirmed_rr_val = "FAILED"
    else:
        confirmed_rr_val = "NOT_CHECKED (Menunggu trigger M5 lengkap)"

    out.append(f"   * Zone Touch     : {zt}")
    out.append(f"   * Liquidity Sweep: {sw}" + (f" ({sw_desc})" if sw == "CONFIRMED" and sw_desc else ""))
    out.append(f"   * CHoCH/BOS/CISD : {tr}" + (f" ({tr_desc})" if tr == "CONFIRMED" and tr_desc else ""))
    out.append(f"   * Displacement   : {dp}" + (f" ({dp_desc})" if dp == "CONFIRMED" and dp_desc else ""))
    out.append(f"   * CONFIRMED_ENTRY_RR : {confirmed_rr_val}")
    out.append(f"   * Risk/Reward    : {rr_st}" + (f" (1:{plan['rr']:.1f})" if (status == "VALID" and plan and plan.get("rr")) else ""))

    if status == "VALID" and plan:
        out.append(f"   * BUY / SELL / WAIT / INVALID: {direction} VALID")
        out.append(f"   * Sinyal Resmi   : {direction} CONFIRMED")
        out.append(f"   * Entry Price    : {plan.get('entry_acuan', 0.0):.2f} (Area: {plan.get('entry_min', 0.0):.2f} - {plan.get('entry_max', 0.0):.2f})")
        out.append(f"   * Stop Loss      : {plan.get('sl', 0.0):.2f}")
        out.append(f"   * Take Profit 1  : {plan.get('tp1', 0.0):.2f}")
        out.append(f"   * Take Profit 2  : {plan.get('tp2', 0.0):.2f}" if plan.get("tp2") else f"   * Take Profit 2  : -")
        out.append(f"   * Risk/Reward    : 1:{plan.get('rr', 0.0):.1f}")
        out.append(f"   * Setup Quality  : HIGH ({zone.get('grade', 'A+') if zone else 'A+'})")
    else:
        out.append(f"   * BUY / SELL / WAIT / INVALID: {status} (NO VALID SETUP)")
        out.append(f"   * Sinyal Resmi   : TIDAK ADA SINYAL AKTIF ({status})")
        out.append("   * Entry Price    : - (Setup belum valid)")
        out.append("   * Stop Loss      : -")
        out.append("   * Take Profit    : -")
        out.append("   * Risk/Reward    : -")
        out.append("   * Setup Quality  : -")
    out.append("")

    # D. FINAL DECISION & REASON
    out.append("D. FINAL DECISION — 1. FINAL DECISION & 5. REASON")
    if status == "VALID" and plan:
        out.append(f"- Status           : {direction} VALID")
        out.append(f"- Recommendation   : {direction} CONFIRMED (Eksekusi Sesuai Rencana)")
        out.append(f"- Direction        : {direction}")
    else:
        out.append(f"- Status           : {status} (NO VALID SETUP)")
        out.append("- Recommendation   : NO ENTRY / WAIT (Dilarang Buka Posisi Baru)")
        out.append("- Direction        : -")

    r_prio = dec.get("radar_priority") or scan.get("radar_priority")
    if not r_prio and all_radar:
        r_prio = prioritize_radar_zones(all_radar, price, htf_bias=h4_bias, h1_structure=h1_struct, m5_stages=stages)

    if status == "VALID" and plan:
        reasons_txt = "; ".join(dec.get("reasons", [])) or "Setup lolos validasi multi-timeframe"
        out.append(f"- Final Decision Reason    : {reasons_txt}")
        out.append(f"- Alasan Keputusan         : {reasons_txt}")
        if r_prio and r_prio.get("primary_reason"):
            out.append(f"- Zone Priority Reason     : {r_prio['primary_reason']}")
        if r_prio and r_prio.get("distance_to_zone_str"):
            out.append(f"- Distance to Zone         : {r_prio['distance_to_zone_str']}")
        out.append("- Validated Conditions     : Seluruh 6 tahap validasi entry (Zone Touch, Sweep, CHoCH/BOS, Displacement, R:R) terkonfirmasi.")
    else:
        explicit_reasons = [r.replace("[SCANNER] Rejection Reason: ", "") for r in dec.get("reasons", []) if r]
        dyn_wait = r_prio.get("dynamic_wait_reason") if r_prio else None

        # Jika terdapat konflik struktur H4 vs H1
        if ctx.get("has_conflict") or (explicit_reasons and any("konflik" in r.lower() for r in explicit_reasons)):
            conflict_msg = next((r for r in explicit_reasons if "konflik" in r.lower()), f"Konflik struktur H4 {h4_bias} vs H1 {h1_struct}")
            dyn_wait = f"WAIT: {conflict_msg}. Seluruh zona dipantau sebagai RADAR."
        elif not dyn_wait:
            if h4_bias == "NETRAL":
                h4_diag_str = f" ({h4_diag})" if h4_diag else ""
                dyn_wait = f"H4 belum memberikan directional bias yang jelas (NETRAL){h4_diag_str}. Menunggu breakout struktur."
            else:
                dyn_wait = explicit_reasons[0] if explicit_reasons else (scan.get("empty_reason") or "Persyaratan entry belum terpenuhi")

        # Jika h4_diag spesifik ada dan h4 netral, pastikan termuat
        if h4_bias == "NETRAL" and h4_diag and h4_diag not in dyn_wait:
            dyn_wait = f"WAIT: Bias H4 NETRAL ({h4_diag}). Belum ada directional bias yang jelas. Zona Supply dan Demand dipantau sebagai batas range reaksi, bukan sinyal entry."

        out.append(f"- Final WAIT Reason        : {dyn_wait}")
        out.append(f"- Alasan Keputusan         : {dyn_wait}")
        if r_prio and r_prio.get("primary_reason"):
            out.append(f"- Zone Priority Reason     : {r_prio['primary_reason']}")
        if r_prio and r_prio.get("distance_to_zone_str"):
            out.append(f"- Distance to Zone         : {r_prio['distance_to_zone_str']}")
        if r_prio and r_prio.get("missing_conditions"):
            out.append("- Missing Entry Conditions :")
            for mc in r_prio["missing_conditions"]:
                out.append(f"  * {mc}")
    out.append("")

    # E. RISK WARNING
    out.append("E. RISK WARNING — 6. RISK WARNING")
    if detailed:
        out.append("- Peringatan Risiko Pasar: Pergerakan XAUUSD memiliki volatilitas tinggi.")
        out.append("  * Manajemen Risiko Modal:")
        out.append("    - 0,5%: risiko lebih konservatif (modal terlindungi secara ketat).")
        out.append("    - 1,0%: risiko lebih tinggi dibandingkan 0,5% (eksposur risiko modal standar/maksimal).")
        out.append("  * Perbedaan Persentase Modal vs Ukuran Lot: Persentase risiko adalah toleransi kerugian modal; ukuran lot dihitung secara matematis dari nominal risiko dibagi jarak SL x contract size MT5.")
        out.append("  * Integritas Perhitungan: Jangan mengubah Entry, SL, TP, atau RR resmi yang dihitung oleh Python Scanner.")
        out.append("  * Status Lot: Jika data kontrak MT5 tidak lengkap atau koneksi terputus, kalkulasi lot dinyatakan DATA UNAVAILABLE (tidak mengarang ukuran lot).")
        out.append("  * Status WAIT: Dilarang membuka posisi baru sebelum seluruh syarat validasi terpenuhi.")
        out.append("  * Pending Plan (READY): Bersifat agresif sebelum konfirmasi M5; alokasi risiko disarankan 0,5% (lebih konservatif).")
        out.append("  * Perlindungan Modal: Wajib pasang Stop Loss di balik invalidation level struktur.")
    else:
        out.append("- Peringatan Risiko: Pergerakan XAUUSD memiliki volatilitas tinggi. Disiplin MM (alokasi risiko modal 0,5% konservatif vs 1,0% standar).")
        out.append("- Status Pending Plan: Early Pending Plan bersifat rencana analisis dan BELUM dipasang di MT5 (eksekusi manual oleh trader).")
        out.append("- Kewajiban Stop Loss: Wajib pasang Stop Loss di balik level invalidation struktur institusional.")
    out.append("=" * 70)
    return "\n".join(out)


def format_decision_text(dec: Dict[str, Any]) -> str:
    audit_txt = format_scanner_audit(dec)
    lines = [f"[ENTRY DECISION] {dec['status']} {dec.get('direction') or ''}".rstrip()]
    for name in ("DATA", "H4", "H1", "M30", "M5", "RISK", "ERROR"):
        if name in dec["stages"]:
            s = dec["stages"][name]
            lines.append(f"  {'OK ' if s['ok'] else 'XX '}{name}: {s['msg']}")
    for r in dec["reasons"]:
        lines.append(f"  - {r}")
    lines.append("-" * 40)
    lines.append(audit_txt)
    return "\n".join(lines)
