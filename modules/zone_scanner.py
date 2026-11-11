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
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from modules.config import (
    SYMBOL, TIMEFRAMES, SWING_N, RR_MIN, SL_JARAK_MAKS, ENTRY_LEBAR_MAKS,
    SPREAD_MAKS_POIN, BLOKIR_JIKA_BERITA_GAGAL, _DIR_SCRIPT,
)
from modules.utils import _catat_error
from modules.mt5_conn import ambil_data, pastikan_mt5_aktif, tambah_indikator

# =============================================================================
# KONSTANTA
# =============================================================================
MIN_DISPLACEMENT_ATR = 1.5      # impuls (OB -> puncak impuls) relatif ATR
MIN_IMPULSE_BODY_ATR = 0.7      # minimal satu candle impuls dengan body >= 0.7 ATR
MIN_BOS_STRENGTH_ATR = 0.10     # close harus melewati level BOS minimal 0.10 ATR
MAX_BOS_DISTANCE = 12           # BOS harus terjadi <= 12 candle setelah OB (impuls langsung)
IMPULSE_WINDOW = 8              # jendela candle impuls setelah OB
MIN_FVG_ATR = 0.30
FVG_ADJACENT_TOL_ATR = 0.10     # FVG dianggap relevan bila overlap/bersinggungan dengan OB
MAX_ZONE_WIDTH_ATR = 2.5
MAX_ZONE_WIDTH_POINTS = 10.0
MIN_ZONE_WIDTH_POINTS = 0.4
MAX_TOUCH_COUNT = 2
MAX_ZONE_DISTANCE_ATR = 10.0
SCORE_MIN_ACTIVE = 60.0
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
                res.append({"k": k, "time": idx[k], "type": "SWEEP_HIGH", "level": p,
                            "wick_extreme": float(H[k]), "close_price": float(C[k]), "swing_time": s["time"],
                            "description": f"Buy-Side Sweep (High {p:.2f}, wick {H[k]:.2f}, close {C[k]:.2f})"})
        for s in act["L"][-6:]:
            p = s["price"]
            if L[k] < p and C[k] >= p and (p - L[k]) >= SWEEP_MIN_PENETRATION_ATR * a:
                if k > s["idx"] + 1 and np.min(L[s["idx"] + 1:k]) < p:
                    continue
                if (min(O[k], C[k]) - L[k]) / rng < SWEEP_MIN_WICK_RATIO:
                    continue
                if k + 1 < len(df) and C[k + 1] < L[k]:
                    continue
                res.append({"k": k, "time": idx[k], "type": "SWEEP_LOW", "level": p,
                            "wick_extreme": float(L[k]), "close_price": float(C[k]), "swing_time": s["time"],
                            "description": f"Sell-Side Sweep (Low {p:.2f}, wick {L[k]:.2f}, close {C[k]:.2f})"})
    return res


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
def detect_order_blocks(df, events, fvgs, sweeps, timeframe, max_events: int = 10):
    """
    OB = candle lawan terakhir pada leg [origin_idx .. BOS] sebelum impuls yang menghasilkan BOS/CHoCH.
    Acuan leg adalah origin_idx (swing lawan = asal impuls); swing_idx (level yang ditembus) TIDAK dipakai
    sebagai pengganti. Hard validation (semua wajib): origin teridentifikasi, candle lawan ada, BOS dekat
    (impuls langsung), BOS tegas (close melewati level >= 0.10 ATR), displacement >= 1.5 ATR,
    ada candle impuls dengan body >= 0.7 ATR, FVG searah yang relevan & belum terisi penuh,
    belum invalid (close menembus OB), sentuhan <= 2, lebar zona wajar.
    Return: (valid_zones, rejected_zones) -- keduanya lengkap dengan audit alasan.
    """
    if df is None or len(df) < 2 * SWING_N + 1 or not events:
        return [], []
    O, H, L, C = (df[c].values for c in ("Open", "High", "Low", "Close"))
    atr, idx = _atr_array(df), df.index
    ok_list: List[Zone] = []
    bad_list: List[Zone] = []

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
        for f in fvgs:
            if f["type"] != want or f["status"] == "FULLY_MITIGATED" or not (ob_idx <= f["idx"] <= k):
                continue
            overlap = min(ob_hi, f["price_high"]) - max(ob_lo, f["price_low"])
            if overlap < -FVG_ADJACENT_TOL_ATR * a:
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

        cap = min(MAX_ZONE_WIDTH_POINTS, MAX_ZONE_WIDTH_ATR * a)
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
        if hi - lo > MAX_ZONE_WIDTH_POINTS:
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
        )
    return out


# =============================================================================
# KONTEKS MULTI-TIMEFRAME & MARKET REGIME
# =============================================================================
def detect_market_regime(df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """
    Identifikasi kondisi pasar (Trending, Ranging, Choppy):
      - TRENDING_BULLISH: Struktur BOS/trend Bullish + EMA bullish (Close > EMA20 > EMA50)
      - TRENDING_BEARISH: Struktur BOS/trend Bearish + EMA bearish (Close < EMA20 < EMA50)
      - RANGING: Osilasi horizontal dalam rentang sempit (lebar rentang <= 2.6 ATR)
      - CHOPPY: Bolak-balik break tanpa arah jelas (whipsaw / swing berdekatan)
      - UNKNOWN: Data tidak cukup
    """
    if df is None or len(df) < 2 * SWING_N + 20:
        return {"regime": "UNKNOWN", "bias": "NO_DATA", "detail": "Data tidak cukup"}

    d = df if "EMA20" in df.columns else tambah_indikator(df.copy())
    events, trend, swings = detect_market_structure(d)
    last = d.iloc[-1]

    # 1. Deteksi Choppy: dalam 15 candle terakhir terdapat >= 2 event BOS/CHoCH bolak-balik berlawanan arah
    recent_events = [e for e in events if len(d) - 1 - e["k"] <= 15]
    if len(recent_events) >= 2:
        arahs = [e["arah"] for e in recent_events]
        if "bull" in arahs and "bear" in arahs:
            return {"regime": "CHOPPY", "bias": "NETRAL", "detail": "Whipsaw: bolak-balik BOS/CHoCH berlawanan dalam 15 candle"}

    # 2. Deteksi Ranging: 20 candle terakhir bergerak dalam rentang horizontal sempit <= 2.6 ATR
    look = min(20, len(d))
    high_range = float(d["High"].iloc[-look:].max())
    low_range = float(d["Low"].iloc[-look:].min())
    atr_val = float(d["ATR14"].iloc[-1]) if "ATR14" in d.columns else 2.0
    if (high_range - low_range) <= 2.6 * atr_val and abs(float(last["EMA20"]) - float(last["EMA50"])) < 0.4 * atr_val:
        return {"regime": "RANGING", "bias": "NETRAL", "detail": f"Konsolidasi horisontal {low_range:.2f}-{high_range:.2f} (lebar {high_range-low_range:.2f} pt)"}

    # 3. Trending
    ema_bull = last["EMA20"] > last["EMA50"] and last["Close"] > last["EMA20"]
    ema_bear = last["EMA20"] < last["EMA50"] and last["Close"] < last["EMA20"]

    if ema_bull:
        return {"regime": "TRENDING_BULLISH", "bias": "BULLISH", "detail": "EMA/Struktur tren bullish"}
    elif ema_bear:
        return {"regime": "TRENDING_BEARISH", "bias": "BEARISH", "detail": "EMA/Struktur tren bearish"}
    elif trend == 1:
        return {"regime": "TRENDING_BULLISH", "bias": "BULLISH", "detail": "Struktur BOS bullish (EMA fase transisi)"}
    elif trend == -1:
        return {"regime": "TRENDING_BEARISH", "bias": "BEARISH", "detail": "Struktur BOS bearish (EMA fase transisi)"}

    return {"regime": "RANGING", "bias": "NETRAL", "detail": "Tidak ada struktur tren yang dominan"}


def _tf_label(df: Optional[pd.DataFrame]) -> str:
    """BULLISH / BEARISH / NETRAL / NO_DATA: struktur (BOS/CHoCH close) + filter EMA; harus searah, bukan sekadar salah satu."""
    reg = detect_market_regime(df)
    return reg.get("bias", "NETRAL")


def evaluate_htf_context(frames: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """D1 (konteks), H4 (bias utama), H1 (konfirmasi), M30 (validasi). Konflik H4 vs H1 => WAIT."""
    regimes = {tf: detect_market_regime(frames.get(tf)) for tf in ("D1", "H4", "H1", "M30")}
    r = {tf: regimes[tf]["bias"] for tf in ("D1", "H4", "H1", "M30")}
    r["regimes"] = {tf: regimes[tf]["regime"] for tf in ("D1", "H4", "H1", "M30")}
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
# SCANNER UTAMA
# =============================================================================
def scan_xauusd_zones(frames: Optional[Dict[str, pd.DataFrame]] = None, current_price: Optional[float] = None,
                      symbol: str = SYMBOL) -> Dict[str, Any]:
    """
    Pindai zona H4 & H1 (OB + konfirmasi FVG + sweep), skor, gabung, dan kembalikan hasil terstruktur.
    Tidak ada zona lolos => active_zones == [] dengan empty_reason. Tidak ada fallback/pemaksaan zona.
    Raise-safe untuk data buruk (mengembalikan data_ok=False); exception internal tetap naik ke pemanggil
    (pemanggil pipeline keputusan menangkapnya dan menghasilkan WAIT).
    """
    out: Dict[str, Any] = {
        "symbol": symbol, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "current_price": current_price,
        "active_zones": [], "rejected_zones": [], "best_demand": None, "best_supply": None,
        "htf_context": {}, "empty_reason": "", "summary_text": "", "data_ok": False,
    }
    if frames is None:
        frames = _load_frames()
        if frames is None:
            out["empty_reason"] = "MT5 tidak aktif; data tidak dapat diambil"
            out["summary_text"] = "MT5 offline - scanner tidak berjalan (WAIT)."
            return out
    need = 2 * SWING_N + 20
    for tf in ("H4", "H1"):
        if frames.get(tf) is None or len(frames[tf]) < need:
            out["empty_reason"] = f"Data {tf} tidak mencukupi"
            out["summary_text"] = f"Data {tf} belum cukup (WAIT)."
            return out
    if current_price is None:
        ref = frames.get("M5") if frames.get("M5") is not None and len(frames["M5"]) else frames["H1"]
        current_price = float(ref["Close"].iloc[-1])
    if not (isinstance(current_price, (int, float)) and math.isfinite(current_price) and current_price > 0):
        out["empty_reason"] = "Harga referensi tidak valid"
        out["summary_text"] = "Harga tidak valid (WAIT)."
        return out
    out["current_price"], out["data_ok"] = float(current_price), True

    ck = _sig([(tf, len(frames[tf]), str(frames[tf].index[-1]), float(frames[tf]["Close"].iloc[-1]))
               for tf in ("H4", "H1", "M30", "M5") if frames.get(tf) is not None and len(frames[tf])],
              round(float(current_price), 1))
    if _SCAN_CACHE["key"] == ck and time.time() - _SCAN_CACHE["ts"] < SCAN_CACHE_TTL:
        return _SCAN_CACHE["data"]

    ctx = evaluate_htf_context(frames)
    out["htf_context"] = ctx
    valid_all: List[Zone] = []
    rejected: List[Zone] = []
    for tf in ("H4", "H1"):
        df = frames[tf] if "ATR14" in frames[tf].columns else tambah_indikator(frames[tf].copy())
        events, _, swings = detect_market_structure(df)
        sweeps = detect_liquidity_sweeps(df, swings, lookback=None)
        fvgs = detect_fvgs(df)
        ok, bad = detect_order_blocks(df, events, fvgs, sweeps, tf, max_events=8)
        rejected += bad
        atr_now = float(df["ATR14"].iloc[-1])
        for z in ok:
            score_zone(z, htf_alignment=_alignment(z.zone_type, ctx))
            if z.is_valid and abs(z.midpoint - current_price) > MAX_ZONE_DISTANCE_ATR * atr_now:
                z.is_valid, z.grade = False, "REJECTED"
                z.rejection_reason = f"Zona terlalu jauh dari harga ({abs(z.midpoint - current_price):.1f} pt)"
                z.audit.append(z.rejection_reason)
            (valid_all if z.is_valid else rejected).append(z)

    active = merge_overlapping_zones(valid_all)

    def _zone_sort_key(z: Zone) -> Tuple:
        rank = {"A+": 0, "A": 1, "B": 2}
        tf_rank = 0 if "H4" in z.timeframe and "H1" in z.timeframe else (1 if "H4" in z.timeframe else 2)
        fresh_rank = {"FRESH": 0, "TESTED_1X": 1, "TESTED_2X": 2}.get(z.freshness, 9)
        # Prioritas teknikal objektif: Grade, Skor tertinggi, Timeframe HTF, Freshness terbersih, Displacement terkuat, Zone ID (deterministik)
        return (
            rank.get(z.grade, 9),
            -round(float(z.score), 1),
            tf_rank,
            fresh_rank,
            -round(float(z.displacement_strength), 2),
            str(z.zone_id),
        )

    active.sort(key=_zone_sort_key)
    if ctx["has_conflict"]:
        dropped = [z for z in active if z.grade != "A+"]
        for z in dropped:
            z.audit.append("Diturunkan: konflik H4 vs H1 (hanya A+ dipertahankan)")
        rejected += dropped
        active = [z for z in active if z.grade == "A+"]

    out["active_zones"] = [z.to_dict() for z in active]
    out["rejected_zones"] = [z.to_dict() for z in rejected]
    out["best_demand"] = next((z.to_dict() for z in active if z.zone_type == "DEMAND"), None)
    out["best_supply"] = next((z.to_dict() for z in active if z.zone_type == "SUPPLY"), None)
    if not active:
        if ctx["has_conflict"]:
            out["empty_reason"] = "Konflik arah H4 vs H1; tidak ada zona A+"
        elif rejected:
            out["empty_reason"] = f"Tidak ada zona lolos standar. Contoh penolakan: {rejected[0].rejection_reason}"
        else:
            out["empty_reason"] = "Tidak ditemukan OB dengan BOS, displacement, dan FVG yang memenuhi syarat"

    lines = ["=" * 70, f"[PYTHON-ONLY XAUUSD ZONE SCANNER] Harga {current_price:.2f}",
             f"Konteks: D1 {ctx['D1']} | H4 {ctx['H4']} | H1 {ctx['H1']} | M30 {ctx['M30']} -> bias [{ctx['bias_utama']}]",
             "=" * 70]
    if active:
        lines.append(f"DITEMUKAN {len(active)} ZONA VALID:")
        for i, z in enumerate(active, 1):
            fs = (z.fvg_confirmation or {}).get("status", "-")
            lines.append(f"  {i}. [{z.grade}] {z.timeframe} {z.zone_type} {z.price_low:.2f}-{z.price_high:.2f} "
                         f"(skor {z.score:.1f} | {z.freshness} | FVG {fs} | sweep {z.liquidity_sweep_status} | SL {z.sl_recommended:.2f})")
    else:
        lines += ["TIDAK ADA ZONA VALID (STATUS: WAIT)", f"  Alasan: {out['empty_reason']}"]
    lines.append(f"Kandidat ditolak: {len(rejected)}")
    for z in rejected[:3]:
        lines.append(f"  - {z.timeframe} {z.zone_type} {z.price_low:.2f}-{z.price_high:.2f}: {z.rejection_reason}")
    lines.append("=" * 70)
    out["summary_text"] = "\n".join(lines)

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
                 require_sweep: Optional[bool] = None) -> Dict[str, Any]:
    """Hasil terstruktur: status VALID/WAIT/INVALID, reasons, stages, plan.
    Urutan validasi ketat konsisten untuk semua grade zona:
      1. Harga mencapai atau memasuki zona entry (dan belum terlambat)
      2. Terdapat indikasi Liquidity Sweep yang valid (semua grade zona wajib)
      3. Terdapat konfirmasi pembalikan atau kelanjutan struktur (CHoCH/BOS/CISD)
      4. Terdapat displacement yang memadai
      5. Konfirmasi candle dihitung dari candle yang sudah selesai (close)
      6. Risk Reward memenuhi syarat (>= 1.5) dan jarak SL wajar (<= 20 pt)
    """
    res: Dict[str, Any] = {"status": "WAIT", "reasons": [], "plan": None, "m5": {}}
    W = res["reasons"]
    if not zone.is_valid:
        res["status"] = "INVALID"
        W.append(f"Zona {zone.zone_id} tidak valid: {zone.rejection_reason}")
        return res
    if not (isinstance(price, (int, float)) and math.isfinite(price) and price > 0):
        W.append("Harga tidak valid")
        return res
    demand = zone.zone_type == "DEMAND"
    if (demand and price < zone.invalidation_price) or ((not demand) and price > zone.invalidation_price):
        res["status"] = "INVALID"
        W.append(f"Harga {price:.2f} sudah menembus level invalidasi {zone.invalidation_price:.2f}")
        return res
    if news_blackout:
        W.append(f"Blackout berita: {news_blackout}")
        return res
    if df_m5 is None or len(df_m5) < MIN_BARS_DECISION:
        W.append("Data M5 tidak mencukupi")
        return res
    if spread is not None and spread > max(SPREAD_MAKS_POIN, 0.0):
        W.append(f"Spread {spread:.2f} pt melebihi batas {SPREAD_MAKS_POIN:g}")
        return res

    d5 = df_m5 if "ATR14" in df_m5.columns else tambah_indikator(df_m5.copy())
    atr5 = float(d5["ATR14"].iloc[-1])
    if not math.isfinite(atr5) or atr5 <= 0:
        W.append("ATR M5 tidak valid")
        return res
    if abs(price - float(d5["Close"].iloc[-1])) > max(30.0, 6 * atr5):
        W.append("Harga menyimpang jauh dari candle M5 terakhir (data tidak sinkron)")
        return res

    lo, hi = zone.price_low, zone.price_high
    n5 = len(d5)
    O, H, L, C = (d5[c].values for c in ("Open", "High", "Low", "Close"))

    # 1. Pengecekan apakah harga mencapai atau memasuki zona entry
    touched = False
    for x in range(max(0, n5 - M5_ZONE_TOUCH_LOOKBACK), n5):
        if (demand and L[x] <= hi and H[x] >= lo) or ((not demand) and H[x] >= lo and L[x] <= hi):
            touched = True
            break
    if not touched and (lo <= price <= hi):
        touched = True

    if not touched:
        W.append(f"Harga belum mencapai atau menyentuh zona entry {lo:.2f}-{hi:.2f}")
        return res

    # Cek keterlambatan entry (harga sudah lari terlalu jauh meninggalkan zona)
    dist = 0.0 if lo <= price <= hi else (lo - price if price < lo else price - hi)
    approach = max(3.0, 1.5 * atr5)
    if dist > approach:
        W.append(f"Entry terlambat: harga sudah meninggalkan zona entry (jarak {dist:.2f} > toleransi {approach:.2f})")
        return res

    events, _, swings = detect_market_structure(d5)
    sweeps = detect_liquidity_sweeps(d5, swings, lookback=40)
    want_dir = "bull" if demand else "bear"
    want_sw = "SWEEP_LOW" if demand else "SWEEP_HIGH"

    m5 = res["m5"]
    m5.update(structure_event=None, sweep=False, displacement_atr=0.0, cisd=False, zone_touched=touched)

    # 2. Indikasi Liquidity Sweep yang valid (semua grade zona A+, A, B wajib melewati validasi ini)
    confirmed_sweep = any(s["type"] == want_sw and n5 - 1 - s["k"] <= 40 for s in sweeps)
    m5["sweep"] = confirmed_sweep
    need_sweep = True if require_sweep is None else require_sweep
    if need_sweep and not confirmed_sweep:
        W.append(f"M5 Liquidity Sweep tidak terkonfirmasi ({want_sw} belum terjadi di area zona)")
        return res

    # 3. Konfirmasi pembalikan atau kelanjutan struktur (CHoCH / BOS / CISD)
    ev = next((e for e in reversed(events) if e["arah"] == want_dir and n5 - 1 - e["k"] <= M5_EVENT_MAX_AGE), None)
    cisd_ok = False
    try:
        from modules.analisis_teknikal import deteksi_cisd
        cl, _ = deteksi_cisd(d5, swings, lookback=40)
        cisd_ok = any(c["arah"] == ("BULLISH" if demand else "BEARISH") and n5 - 1 - c["k"] <= M5_EVENT_MAX_AGE for c in cl)
        m5["cisd"] = cisd_ok
    except Exception:
        pass

    if ev is None and not cisd_ok:
        W.append(f"Belum ada konfirmasi struktur pembalikan/kelanjutan M5 ({'BOS/CHoCH/CISD bullish' if demand else 'BOS/CHoCH/CISD bearish'})")
        return res

    m5["structure_event"] = f"{ev['tipe']} {ev['arah']} @ {ev['level']:.2f} (k={ev['k']})" if ev else ("CISD " + ("BULLISH" if demand else "BEARISH"))

    # 4. Displacement yang mendukung arah entry
    k_ref = ev["k"] if ev else n5 - 1
    k0 = max(0, k_ref - 3)
    disp5 = float(np.max(np.abs(C[k0:k_ref + 1] - O[k0:k_ref + 1]))) / atr5
    m5["displacement_atr"] = round(disp5, 2)
    if disp5 < M5_MIN_DISPLACEMENT_ATR:
        W.append(f"Displacement M5 lemah ({disp5:.2f} ATR < {M5_MIN_DISPLACEMENT_ATR:g})")
        return res

    # 5. Konfirmasi candle: seluruh data di atas terverifikasi dari candle close M5 yang sudah selesai

    # 6. Risk Reward & Risk Management
    if demand:
        e_lo, e_hi = (lo, hi) if hi - lo <= ENTRY_LEBAR_MAKS else (round(hi - ENTRY_LEBAR_MAKS, 2), hi)
    else:
        e_lo, e_hi = (lo, hi) if hi - lo <= ENTRY_LEBAR_MAKS else (lo, round(lo + ENTRY_LEBAR_MAKS, 2))
    entry = round((e_lo + e_hi) / 2.0, 2)
    sl = zone.sl_recommended
    risk = round((entry - sl) if demand else (sl - entry), 2)
    if risk <= 0.5:
        W.append("Jarak SL tidak realistis")
        return res
    if risk > SL_JARAK_MAKS:
        W.append(f"Jarak SL struktural {risk:.2f} pt melebihi batas maksimal {SL_JARAK_MAKS:g}")
        return res
    targets = _structural_targets(htf_frames or {}, zone.zone_type, entry)
    if not targets:
        W.append("Tidak ada target likuiditas struktural H1/H4 searah transaksi")
        return res
    buf = 0.3
    tp1 = round(targets[0] - buf if demand else targets[0] + buf, 2)
    tp2 = round(targets[1] - buf if demand else targets[1] + buf, 2) if len(targets) > 1 else tp1
    reward = round((tp1 - entry) if demand else (entry - tp1), 2)
    rr = round(reward / risk, 2) if risk > 0 else 0.0
    res["rr"] = rr
    if reward <= 0 or rr < RR_MIN - 1e-9:
        W.append(f"R:R aktual ke TP1 struktural {rr:.2f} < 1:{RR_MIN:g}")
        return res

    res["plan"] = {"arah": "BUY" if demand else "SELL", "entry_min": e_lo, "entry_max": e_hi, "entry_acuan": entry,
                   "sl": sl, "tp1": tp1, "tp2": tp2, "rr": rr, "zone_id": zone.zone_id,
                   "timeframe": zone.timeframe, "grade": zone.grade, "score": zone.score,
                   "keterangan": f"Zona {zone.grade} {zone.timeframe} {zone.zone_type} + trigger M5 {m5['structure_event']}"}
    res["status"] = "VALID"
    W.append(f"Entry valid: {m5['structure_event']}, displacement M5 {disp5:.2f} ATR, R:R {rr:.2f}")
    return res
    res["status"] = "VALID"
    W.append(f"Entry valid: {m5['structure_event']}, displacement M5 {disp5:.2f} ATR, R:R {rr:.2f}")
    return res


def validate_entry_setup(zone: Zone, current_price: float, df_m5: pd.DataFrame, spread_points: float = 0.0,
                         is_news_blackout=False, htf_frames: Optional[Dict[str, pd.DataFrame]] = None,
                         require_sweep: Optional[bool] = None) -> Tuple[bool, str, Dict[str, Any]]:
    """Kompatibel dgn versi lama: (ok, pesan, plan). Status lengkap ada di plan['status'] bila ok; pesan diawali VALID/WAIT/INVALID."""
    r = _entry_check(zone, current_price, df_m5, spread_points, is_news_blackout, htf_frames, require_sweep)
    return r["status"] == "VALID", f"{r['status']}: " + "; ".join(r["reasons"]), (r["plan"] or {})


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
        return _decide(dec, frames, current_price, spread, news_blackout, server_time, live)
    except Exception as e:
        _catat_error("evaluate_trade_decision", e)
        dec.update(status="WAIT", direction=None, plan=None, zone=None)
        dec["reasons"].append(f"Error internal scanner ({type(e).__name__}): fail-safe WAIT")
        dec["stages"]["ERROR"] = {"ok": False, "msg": repr(e)}
        _audit("entry_decision", _entry_audit(dec), _sig(dec["status"], dec["reasons"]))
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

    def finish(status, reason):
        dec["status"] = status
        if reason:
            dec["reasons"].append(reason)
        if status != "VALID":
            dec["plan"] = None
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

    # --- M30: keselarasan & momentum ---
    if ctx["m30_conflict"]:
        st["M30"] = {"ok": False, "msg": f"M30 {ctx['M30']} bertentangan dengan H4 {h4}"}
        return finish("WAIT", f"M30 {ctx['M30']} bertentangan dengan struktur utama H4 {h4}")
    st["M30"] = {"ok": True, "msg": f"M30 {ctx['M30']} selaras/netral terhadap H4 {h4}"}

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
                st["M30"] = {"ok": False, "msg": "Close M30 menembus invalidasi zona"}
                return finish("INVALID", f"Zona invalid: close M30 menembus {inval:.2f}")
            continue

        r = _entry_check(zn, float(price), frames["M5"], spread if spread is not None else 0.0, news, frames)
        if not first_reasons:
            first_reasons = r["reasons"]
        if r["status"] == "VALID":
            valid_plan = r["plan"]
            winning_zone = zn
            winning_zdict = zd
            dec["reasons"].extend(r["reasons"])
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
    st["RISK"] = {"ok": False, "msg": "Menunggu konfirmasi entry M5 & R:R"}
    return finish("WAIT", "")


def _zone_from_dict(d: Dict[str, Any]) -> Zone:
    keys = Zone.__dataclass_fields__.keys()
    return Zone(**{k: d[k] for k in keys if k in d})


def format_scanner_audit(dec: Dict[str, Any]) -> str:
    """Format audit log standar [SCANNER] sesuai spesifikasi audit trading."""
    status = dec.get("status", "WAIT")
    direction = dec.get("direction")
    stages = dec.get("stages", {})
    plan = dec.get("plan") or {}

    lines = []
    # H4 Bias
    h4_msg = stages.get("H4", {}).get("msg", "-")
    h4_bias = "BULLISH" if "BULLISH" in h4_msg.upper() else ("BEARISH" if "BEARISH" in h4_msg.upper() else "NETRAL")
    lines.append(f"[SCANNER] H4 Bias: {h4_bias}")

    # H1 Structure
    h1_ok = stages.get("H1", {}).get("ok", False)
    lines.append(f"[SCANNER] H1 Structure: {'BULLISH' if (h1_ok and direction == 'BUY') else ('BEARISH' if (h1_ok and direction == 'SELL') else 'UNCONFIRMED')}")

    # M30 Momentum
    m30_ok = stages.get("M30", {}).get("ok", False)
    lines.append(f"[SCANNER] M30 Momentum: {'CONFIRMED' if m30_ok else 'UNCONFIRMED'}")

    # M5 validations
    m5_ok = stages.get("M5", {}).get("ok", False)
    lines.append(f"[SCANNER] M5 Liquidity Sweep: {'CONFIRMED' if m5_ok else 'NOT CONFIRMED'}")
    lines.append(f"[SCANNER] M5 Displacement: {'CONFIRMED' if m5_ok else 'NOT CONFIRMED'}")
    lines.append(f"[SCANNER] Entry Validation: {'PASSED' if status == 'VALID' else 'WAIT'}")

    if status == "VALID" and plan.get("rr"):
        lines.append(f"[SCANNER] Risk Reward: 1:{plan['rr']:.1f}")
        lines.append(f"[SCANNER] Final Decision: {direction} VALID")
    else:
        lines.append(f"[SCANNER] Final Decision: {status}")
        reasons = [r.replace("[SCANNER] Rejection Reason: ", "") for r in dec.get("reasons", []) if r]
        rej_msg = reasons[0] if reasons else "Persyaratan entry belum terpenuhi"
        lines.append(f"[SCANNER] Rejection Reason: {rej_msg}")

    return "\n".join(lines)


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
