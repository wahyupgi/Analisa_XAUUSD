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
import inspect
import json
import math
import os
import re
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union, NamedTuple

import numpy as np
import pandas as pd

from modules.config import (
    SYMBOL, TIMEFRAMES, SWING_N, RR_MIN, SL_JARAK_MAKS, ENTRY_LEBAR_MAKS,
    SPREAD_MAKS_POIN, BLOKIR_JIKA_BERITA_GAGAL, _DIR_SCRIPT,
    FILE_STATE_ZONE_TRACKER, DEFAULT_PENDING_EXPIRATION_HOURS, ZONE_APPROACH_DISTANCE_POINTS,
    MULTI_TIER_MAX_ZONES, HIGH_RISK_MAX_ATR_RATIO, HIGH_RISK_MAX_DISTANCE_PT, HIGH_RISK_MIN_SCORE,
    MEDIUM_RISK_MAX_ATR_RATIO, MEDIUM_RISK_MAX_DISTANCE_PT, MEDIUM_RISK_MIN_SCORE,
    LOW_RISK_MAX_ATR_RATIO, LOW_RISK_MAX_DISTANCE_PT, LOW_RISK_MIN_SCORE,
)
from modules.utils import _catat_error, WIB_TZ, get_current_wib, to_wib_datetime
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
TF_MINUTES = {"M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440, "W1": 10080}
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
    scoring_status: str = "VALID"        # VALID / DATA_INCOMPLETE
    score_reason: str = ""
    distance: float = 0.0
    direction_status: str = ""           # ABOVE_PRICE / BELOW_PRICE / INSIDE_ZONE
    approach: str = "UNTOUCHED"          # TOUCHED / APPROACHING / UNTOUCHED / FAR
    is_aligned: bool = False
    priority_labels: List[str] = field(default_factory=list)
    distance_atr_ratio: Optional[float] = None
    atr_value: Optional[float] = None
    risk_tier: str = ""                       # HIGH_RISK / MEDIUM_RISK / LOW_RISK / UNCLASSIFIED
    risk_tier_reason: str = ""
    radar_eligibility: str = "QUALIFIED"      # QUALIFIED / NOT_QUALIFIED
    pending_eligibility: str = "NOT_QUALIFIED"# READY / NOT_QUALIFIED / WAIT
    confirmed_entry_eligibility: str = "WAIT" # CONFIRMED / WAIT / FAILED
    pending_plan: Optional[Dict[str, Any]] = None
    proximal_edge: float = 0.0
    distal_edge: float = 0.0
    zone_width: float = 0.0

    def __post_init__(self):
        if self.price_high > 0 and self.price_low > 0:
            if self.zone_width == 0.0:
                self.zone_width = round(self.price_high - self.price_low, 2)
            if self.proximal_edge == 0.0:
                self.proximal_edge = round(self.price_high if self.zone_type == "DEMAND" else self.price_low, 2)
            if self.distal_edge == 0.0:
                self.distal_edge = round(self.price_low if self.zone_type == "DEMAND" else self.price_high, 2)

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
# =============================================================================
# SCORING
# =============================================================================
_ALIGN_WEIGHT = {"H4": 4.0, "H1": 3.0, "M30": 3.0}


def score_zone(
    zone: Zone,
    htf_bias_aligned: bool = True,
    htf_is_neutral: bool = False,
    htf_alignment: Optional[Dict[str, str]] = None,
    allow_radar: bool = False,
    current_price: Optional[float] = None,
    htf_bias: Optional[str] = None,
    h1_bias: Optional[str] = None,
) -> Zone:
    """
    Skor 0-100 dengan Breakdown Transparan:
      1. Higher Timeframe Alignment (HTF Context) : 10 poin (H4=4.0, H1=3.0, M30=3.0; Netral=0.5x, Berlawanan=0.0)
      2. Market Structure Alignment (Struktur BOS): 25 poin (BOS=17 base, CHoCH=13 base, TF factor, strength_atr)
      3. Zone Strength (Displacement & Body)      : 20 poin (>=2.5: 20, >=2.0: 16, >=1.2: 12)
      4. FVG Quality (Konfirmasi Imbalance)       : 20 poin (UNTOUCHED: 20, PARTIALLY_MITIGATED: 14, NONE: 0)
      5. Freshness (Status Pengujian Zona)       : 15 poin (FRESH: 15, TESTED_1X: 8, TESTED_2X: 3, MITIGATED/INVALID: 0)
      6. Liquidity Context (Sweep Likuiditas)     : 10 poin (CONFIRMED_SWEEP: 10, NONE: 0)
      7. Proximity (Jarak & Pendekatan Harga)     : Dicatat transparan pada score_breakdown & prioritas.

    Audit & Penanganan Data:
      - Jika data tidak tersedia/tidak lengkap: status DATA_INCOMPLETE, bukan menyamarkannya sebagai skor valid 0.0.
      - Membedakan skor 0.0 hasil evaluasi objektif dengan kegagalan kalkulasi data tidak lengkap.
      - Zona RADAR yang utuh struktural tetapi gagal 1 hard filter (misal tanpa FVG) tetap dihitung skornya secara transparan.
      - Zona yang benar-benar invalid / dibatalkan tetap grade REJECTED / is_valid = False.
    """
    # 0. HTF Bias argument mapping
    if htf_bias is not None:
        if htf_bias == "NETRAL":
            htf_is_neutral = True
            htf_bias_aligned = False
        elif (zone.zone_type == "DEMAND" and htf_bias == "BULLISH") or (zone.zone_type == "SUPPLY" and htf_bias == "BEARISH"):
            htf_bias_aligned = True
            htf_is_neutral = False
        else:
            htf_bias_aligned = False
            htf_is_neutral = False

    # 1. Audit Kelengkapan Data (DATA INCOMPLETE Check)
    ev = getattr(zone, "origin_bos", None)
    is_data_incomplete = getattr(zone, "incomplete_data", False)
    if ev is None or not isinstance(ev, dict):
        is_data_incomplete = True
    elif (not ev) and (not getattr(zone, "is_valid", False)) and ("incomplete" in (getattr(zone, "rejection_reason", "") or "").lower()):
        is_data_incomplete = True
    elif getattr(zone, "price_low", 0.0) <= 0 or getattr(zone, "price_high", 0.0) <= 0 or getattr(zone, "price_high", 0.0) < getattr(zone, "price_low", 0.0):
        is_data_incomplete = True
    elif not getattr(zone, "timeframe", ""):
        is_data_incomplete = True

    if is_data_incomplete:
        zone.scoring_status = "DATA_INCOMPLETE"
        zone.score = 0.0
        zone.grade = "DATA_INCOMPLETE"
        zone.score_breakdown = {
            "status": "DATA_INCOMPLETE",
            "konteks_htf": 0.0, "struktur_bos": 0.0, "displacement": 0.0,
            "fvg_konfirmasi": 0.0, "freshness": 0.0, "liquidity_sweep": 0.0,
            "htf_alignment": 0.0, "market_structure": 0.0, "zone_strength": 0.0,
            "fvg_quality": 0.0, "liquidity_context": 0.0,
            "proximity": getattr(zone, "approach", "UNTOUCHED"),
        }
        zone.score_reason = "DATA INCOMPLETE: Data struktur atau harga zona tidak tersedia / tidak valid"
        zone.audit.append(zone.score_reason)
        return zone

    # 2. Hard Validation Rejection Check (Non-Radar)
    is_inval_rej = "invalid" in (zone.rejection_reason or "").lower() or zone.freshness == "INVALIDATED"
    if not zone.is_valid and (not allow_radar or is_inval_rej):
        zone.score, zone.grade = 0.0, "REJECTED"
        zone.scoring_status = "VALID"
        zone.score_breakdown = {
            "status": "VALID",
            "konteks_htf": 0.0, "struktur_bos": 0.0, "displacement": 0.0,
            "fvg_konfirmasi": 0.0, "freshness": 0.0, "liquidity_sweep": 0.0,
            "htf_alignment": 0.0, "market_structure": 0.0, "zone_strength": 0.0,
            "fvg_quality": 0.0, "liquidity_context": 0.0,
            "proximity": getattr(zone, "approach", "UNTOUCHED"),
        }
        zone.score_reason = f"Skor 0.0 (Hard validation rejected: {zone.rejection_reason or 'tidak valid'})"
        return zone

    # 3. Kalkulasi Komponen Skor
    # A. Market Structure Alignment (BOS/CHoCH, TF, kekuatan close) - max 25.0
    base = 17.0 if ev.get("tipe") == "BOS" else 13.0
    tf_factor = 1.0 if "H4" in zone.timeframe else 0.8
    p_struct = min(25.0, base * tf_factor + min(8.0, 4.0 * float(ev.get("strength_atr", 0.0))))

    # B. Zone Strength / Displacement - max 20.0
    d = zone.displacement_strength
    p_disp = 20.0 if d >= 2.5 else 16.0 if d >= 2.0 else 12.0 if d >= MIN_DISPLACEMENT_ATR else 0.0

    # C. FVG Quality - max 20.0
    st = (zone.fvg_confirmation or {}).get("status")
    p_fvg = 20.0 if st == "UNTOUCHED" else 14.0 if st == "PARTIALLY_MITIGATED" else 0.0

    # D. Freshness - max 15.0
    p_fresh = {"FRESH": 15.0, "TESTED_1X": 8.0, "TESTED_2X": 3.0}.get(zone.freshness, 0.0)

    # E. Higher Timeframe Alignment - max 10.0
    if htf_alignment is not None:
        p_htf = sum(w * (1.0 if htf_alignment.get(tf) == "ALIGNED" else 0.5 if htf_alignment.get(tf) == "NEUTRAL" else 0.0)
                    for tf, w in _ALIGN_WEIGHT.items())
        zone.htf_alignment = dict(htf_alignment)
    else:
        p_htf = 10.0 if htf_bias_aligned else 5.0 if htf_is_neutral else 0.0

    # F. Liquidity Context (Sweep) - max 10.0
    p_sweep = 10.0 if zone.liquidity_sweep_status.startswith("CONFIRMED") else 0.0

    total = round(p_struct + p_disp + p_fvg + p_fresh + p_htf + p_sweep, 1)
    zone.score = total
    zone.scoring_status = "VALID"

    # G. Proximity Status
    prox_status = getattr(zone, "approach", "UNTOUCHED")
    if current_price is not None and current_price > 0 and getattr(zone, "price_low", 0) > 0:
        dist_res = calculate_zone_distance(zone.price_low, zone.price_high, current_price, digits=2)
        prox_status = dist_res.status
        zone.distance = dist_res.distance
        zone.direction_status = dist_res.status

    zone.score_breakdown = {
        # Backward-compatible keys:
        "struktur_bos": round(p_struct, 1),
        "displacement": p_disp,
        "fvg_konfirmasi": p_fvg,
        "freshness": p_fresh,
        "konteks_htf": round(p_htf, 1),
        "liquidity_sweep": p_sweep,
        # Transparent 7-factor breakdown aliases:
        "market_structure": round(p_struct, 1),
        "zone_strength": p_disp,
        "fvg_quality": p_fvg,
        "htf_alignment": round(p_htf, 1),
        "liquidity_context": p_sweep,
        "proximity": prox_status,
        "status": "VALID",
    }

    # Format score_reason penjelasan transparansi:
    reasons_list = [
        f"Struktur BOS {ev.get('tipe', 'BOS')} (+{p_struct:.1f})",
        f"Zone Strength / Disp {d:.2f}x (+{p_disp:.1f})",
        f"FVG {st or 'NONE'} (+{p_fvg:.1f})",
        f"Freshness {zone.freshness} (+{p_fresh:.1f})",
        f"HTF Alignment (+{p_htf:.1f})",
        f"Liquidity Sweep (+{p_sweep:.1f})" if p_sweep > 0 else "No Sweep (0.0)",
    ]

    if not zone.is_valid and allow_radar:
        zone.grade = "RADAR"
        zone.score_reason = f"Skor {total:.1f} [RADAR]: " + ", ".join(reasons_list)
        zone.audit.append(f"Kandidat RADAR: grade {zone.grade}, skor {total:.1f}")
    else:
        zone.grade = _grade(total)
        zone.score_reason = f"Skor {total:.1f} [{zone.grade}]: " + ", ".join(reasons_list)
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
    
    Penyempurnaan Multi-Tier SMC:
    Jika terdapat zona LTF refinement (M5/M15) di dalam zona HTF (H1/H4) dengan lebar zona
    HTF jauh lebih lebar (HTF width >= 1.8 * LTF width), pertahankan struktur independen
    agar presisi entry dan proximal/distal edge LTF tidak tereliminasi oleh zona makro HTF.
    Zona pada timeframe berdekatan / sebanding tetap digabung menjadi konfluensi multi-TF.
    """
    if len(zones) <= 1:
        return zones
    out: List[Zone] = []
    order = {"FRESH": 0, "TESTED_1X": 1, "TESTED_2X": 2, "MITIGATED": 3, "INVALIDATED": 4}
    ltf_tfs = {"M5", "M15"}
    htf_tfs = {"H1", "H4", "D1", "W1"}

    for z in sorted(zones, key=lambda x: (x.zone_type, x.price_low)):
        last = out[-1] if out else None
        if last is None or last.zone_type != z.zone_type or min(last.price_high, z.price_high) < max(last.price_low, z.price_low):
            out.append(z)
            continue

        # Evaluasi apakah salah satu zona merupakan LTF refinement independen
        last_is_ltf = any(t in ltf_tfs for t in last.timeframe.split("+"))
        z_is_ltf = any(t in ltf_tfs for t in z.timeframe.split("+"))
        last_is_htf = any(t in htf_tfs for t in last.timeframe.split("+"))
        z_is_htf = any(t in htf_tfs for t in z.timeframe.split("+"))

        last_w = round(last.price_high - last.price_low, 2)
        z_w = round(z.price_high - z.price_low, 2)

        is_independent_refinement = False
        if (last_is_ltf and z_is_htf and z_w >= 1.8 * last_w) or (z_is_ltf and last_is_htf and last_w >= 1.8 * z_w):
            is_independent_refinement = True

        if is_independent_refinement:
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
            scoring_status=getattr(best, "scoring_status", "VALID"),
            score_reason=getattr(best, "score_reason", ""),
            distance=getattr(best, "distance", 0.0),
            approach=getattr(best, "approach", "UNTOUCHED"),
            is_aligned=getattr(best, "is_aligned", False),
            priority_labels=list(getattr(best, "priority_labels", [])),
            distance_atr_ratio=getattr(best, "distance_atr_ratio", None),
            atr_value=getattr(best, "atr_value", None),
            risk_tier=getattr(best, "risk_tier", ""),
            radar_eligibility=getattr(best, "radar_eligibility", "QUALIFIED"),
            pending_eligibility=getattr(best, "pending_eligibility", "NOT_QUALIFIED"),
            confirmed_entry_eligibility=getattr(best, "confirmed_entry_eligibility", "WAIT"),
            pending_plan=getattr(best, "pending_plan", None),
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


def _generate_snapshot_id() -> str:
    """Buat ID unik deterministik-waktu untuk satu siklus snapshot."""
    return f"SNAP_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def _live_context(symbol: str = SYMBOL, tick: Any = None) -> Dict[str, Any]:
    """Harga bid/ask/mid snapshot tunggal, spread, waktu server, dan status blackout berita.

    tick: objek tick MT5 yang sudah diambil pemanggil (dipakai apa adanya, TANPA memanggil
    mt5.symbol_info_tick lagi). Jika None, satu tick baru diambil dari MT5.
    """
    ctx = {"price": None, "spread": None, "server_time": None, "news": None, "news_error": None, "snapshot": None}
    try:
        if tick is None:
            import MetaTrader5 as mt5
            tick = mt5.symbol_info_tick(symbol)
        if tick:
            b_val = float(getattr(tick, "bid", 0.0) or 0.0)
            a_val = float(getattr(tick, "ask", 0.0) or 0.0)
            if math.isfinite(b_val) and math.isfinite(a_val) and b_val > 0 and a_val >= b_val:
                bid = round(b_val, 2)
                ask = round(a_val, 2)
                mid = round((bid + ask) / 2.0, 2)
                spread = round(ask - bid, 2)
                if hasattr(tick, "time") and tick.time > 0:
                    tick_time = datetime.fromtimestamp(tick.time).strftime("%Y-%m-%d %H:%M:%S WIB")
                    tick_age = round(max(0.0, time.time() - tick.time), 1)
                    srv_ts = pd.Timestamp(int(tick.time), unit="s")
                else:
                    tick_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S WIB")
                    tick_age = 0.0
                    srv_ts = None
                snap = {
                    "snapshot_id": _generate_snapshot_id(),
                    "symbol": symbol,
                    "bid": bid,
                    "ask": ask,
                    "mid": mid,
                    "spread": spread,
                    "snapshot_price": mid,
                    "snapshot_time": tick_time,
                    "tick_time": tick_time,
                    "tick_age": tick_age,
                    "server_time": srv_ts,
                    "price_source": f"MT5 Live Tick (Bid: {bid:.2f} | Ask: {ask:.2f} | Mid: {mid:.2f} | Spread: {spread:.2f} pt)",
                    "snapshot_status": "OK",
                }
                ctx.update(price=mid, spread=spread, server_time=srv_ts, snapshot=snap)
            elif b_val > 0 or a_val > 0 or not math.isfinite(b_val) or not math.isfinite(a_val):
                ctx["snapshot"] = {"snapshot_id": _generate_snapshot_id(), "symbol": symbol, "snapshot_status": "DATA_INCOMPLETE"}
    except Exception as e:
        _catat_error("zone_scanner_live_tick", e)
    try:
        from modules.analisis_pasar import berita_blackout_aktif
        ctx["news"] = berita_blackout_aktif() or None
    except Exception as e:
        ctx["news_error"] = repr(e)
        _catat_error("zone_scanner_berita", e)
    return ctx


def create_market_snapshot(
    symbol: str = SYMBOL,
    frames: Optional[Dict[str, pd.DataFrame]] = None,
    current_price: Optional[float] = None,
    spread: Optional[float] = None,
    server_time: Any = None,
    news_blackout: Any = None,
    live: Optional[bool] = None,
    force_refresh_frames: bool = True,
    tick: Any = None,
) -> Dict[str, Any]:
    """
    Buat satu objek snapshot pasar terpusat untuk satu siklus scan -> evaluate -> report.
    Menjamin satu pengambilan MT5 tick dan satu pengambilan candle OHLC per siklus.

    Sumber data (dicatat pada key `data_sources`):
      - frames: "caller_supplied" bila frames diberikan, selain itu "mt5_fetched".
      - tick  : "caller_supplied" bila `tick` diberikan (tidak ada fetch tick baru),
                "mt5_fetched" bila diambil di sini, "none" bila tidak live.
      - price : "snapshot_mid" (default) atau "caller_override" bila current_price eksplisit berbeda dari mid.
    """
    snapshot_id = _generate_snapshot_id()
    use_live = (frames is None) if live is None else bool(live)
    if tick is not None:
        use_live = True
    _frames_src = "caller_supplied" if frames is not None else "mt5_fetched"
    _tick_src = "caller_supplied" if tick is not None else ("mt5_fetched" if use_live else "none")
    _price_arg = current_price

    if frames is None:
        frames = _load_frames(ttl=0.0 if force_refresh_frames else FRAMES_CACHE_TTL)
        if frames is None:
            return {
                "snapshot_id": snapshot_id,
                "symbol": symbol,
                "bid": None,
                "ask": None,
                "mid": None,
                "spread": spread,
                "current_price": None,
                "snapshot_price": None,
                "snapshot_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S WIB"),
                "tick_time": None,
                "tick_age": None,
                "server_time": server_time,
                "price_source": "MT5 Offline",
                "snapshot_status": "DATA_INCOMPLETE",
                "frames": None,
                "atr_diagnostics": None,
                "news": news_blackout,
                "news_error": None,
                "data_ok": False,
                "error_reason": "MT5 tidak aktif; data tidak dapat diambil",
            }

    bid = None
    ask = None
    mid = None
    tick_time = None
    tick_age = None
    price_source = None
    snapshot_status = "OK"
    news_err = None
    error_reason = ""

    if use_live:
        lc = _live_context(symbol=symbol, tick=tick)
        lc_snap = lc.get("snapshot")
        if isinstance(lc_snap, dict) and lc_snap.get("snapshot_status") == "DATA_INCOMPLETE":
            return {
                "snapshot_id": snapshot_id,
                "symbol": symbol,
                "bid": None,
                "ask": None,
                "mid": None,
                "spread": spread,
                "current_price": None,
                "snapshot_price": None,
                "snapshot_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S WIB"),
                "tick_time": None,
                "tick_age": None,
                "server_time": server_time,
                "price_source": "MT5 Live Tick (DATA_INCOMPLETE)",
                "snapshot_status": "DATA_INCOMPLETE",
                "frames": frames,
                "atr_diagnostics": extract_atr_diagnostics(frames, preferred_tf="H1"),
                "news": news_blackout if news_blackout is not None else lc.get("news"),
                "news_error": lc.get("news_error"),
                "data_ok": False,
                "error_reason": "DATA_INCOMPLETE: Snapshot MT5 Bid/Ask tidak lengkap atau tidak valid",
            }
        if isinstance(lc_snap, dict) and lc_snap.get("snapshot_status") == "OK":
            bid = lc_snap.get("bid")
            ask = lc_snap.get("ask")
            mid = lc_snap.get("mid")
            if spread is None:
                spread = lc_snap.get("spread")
            tick_time = lc_snap.get("tick_time")
            tick_age = lc_snap.get("tick_age")
            price_source = lc_snap.get("price_source")
            if server_time is None:
                server_time = lc.get("server_time")
            if current_price is None:
                current_price = mid
        elif current_price is None:
            return {
                "snapshot_id": snapshot_id,
                "symbol": symbol,
                "bid": None,
                "ask": None,
                "mid": None,
                "spread": spread,
                "current_price": None,
                "snapshot_price": None,
                "snapshot_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S WIB"),
                "tick_time": None,
                "tick_age": None,
                "server_time": server_time,
                "price_source": "MT5 Live Tick Unavailable (DATA_INCOMPLETE)",
                "snapshot_status": "DATA_INCOMPLETE",
                "frames": frames,
                "atr_diagnostics": extract_atr_diagnostics(frames, preferred_tf="H1"),
                "news": news_blackout if news_blackout is not None else lc.get("news"),
                "news_error": lc.get("news_error"),
                "data_ok": False,
                "error_reason": "DATA_INCOMPLETE: Snapshot MT5 tick tidak tersedia",
            }
        if news_blackout is None:
            news_blackout = lc.get("news")
        news_err = lc.get("news_error")

    if current_price is None:
        if mid is not None:
            current_price = mid
        else:
            ref = frames.get("M5") if frames.get("M5") is not None and len(frames["M5"]) else frames.get("H1")
            if ref is not None and len(ref):
                current_price = float(ref["Close"].iloc[-1])

    if not (isinstance(current_price, (int, float)) and math.isfinite(current_price) and current_price > 0):
        return {
            "snapshot_id": snapshot_id,
            "symbol": symbol,
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "spread": spread,
            "current_price": None,
            "snapshot_price": None,
            "snapshot_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S WIB"),
            "tick_time": tick_time,
            "tick_age": tick_age,
            "server_time": server_time,
            "price_source": price_source or "Invalid Price",
            "snapshot_status": "DATA_INCOMPLETE",
            "frames": frames,
            "atr_diagnostics": extract_atr_diagnostics(frames, preferred_tf="H1"),
            "news": news_blackout,
            "news_error": news_err,
            "data_ok": False,
            "error_reason": "DATA_INCOMPLETE: Harga referensi tidak valid",
        }

    snap_price = round(float(current_price), 2)
    snap_time = tick_time or datetime.now().strftime("%Y-%m-%d %H:%M:%S WIB")
    if price_source is None:
        price_source = f"Candle Close Snapshot ({snap_price:.2f})"
    atr_diag = extract_atr_diagnostics(frames, preferred_tf="H1")

    return {
        "snapshot_id": snapshot_id,
        "symbol": symbol,
        "bid": bid,
        "ask": ask,
        "mid": mid,
        "spread": spread,
        "current_price": snap_price,
        "snapshot_price": snap_price,
        "snapshot_time": snap_time,
        "tick_time": tick_time,
        "tick_age": tick_age,
        "server_time": server_time,
        "price_source": price_source,
        "snapshot_status": snapshot_status,
        "frames": frames,
        "atr_diagnostics": atr_diag,
        "news": news_blackout,
        "news_error": news_err,
        "data_ok": True,
        "error_reason": error_reason,
        "data_sources": {
            "frames": _frames_src,
            "tick": _tick_src,
            "price": "caller_override" if (_price_arg is not None and mid is not None and round(float(_price_arg), 2) != mid) else (
                "snapshot_mid" if mid is not None else ("caller_price" if _price_arg is not None else "candle_close")),
        },
    }


def _stale_reason(frames: Dict[str, pd.DataFrame], server_time) -> Optional[str]:
    if server_time is None:
        return None
    for tf in ("M5", "M30", "H1", "H4"):
        df = frames.get(tf)
        if df is None or len(df) == 0:
            continue
        age = (pd.Timestamp(server_time) - df.index[-1]).total_seconds() / 60.0
        if age < -TF_MINUTES[tf]:
            return f"Tick lebih lama dari candle {tf} terakhir ({abs(age):.0f} menit; stale tick)"
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
class ZoneStatus(str):
    """
    Subclass string untuk status zona yang mendukung backward compatibility
    antara ZONE_TOUCHED dan TOUCHED.
    """
    def __eq__(self, other):
        if str(self) == "ZONE_TOUCHED" and other == "TOUCHED":
            return True
        if str(self) == "TOUCHED" and other == "ZONE_TOUCHED":
            return True
        return super().__eq__(other)

    def __hash__(self):
        return hash(str(self))


class ValidationStatus(str):
    """
    Subclass string untuk status evaluasi validasi M5 & kondisi entry:
    Mendukung 4 status konsisten:
      - PASS: pemeriksaan dijalankan dan berhasil
      - FAIL: pemeriksaan dijalankan dan tidak memenuhi syarat
      - NOT_CHECKED: pemeriksaan belum dijalankan
      - DATA_INCOMPLETE: data yang diperlukan tidak tersedia atau tidak cukup
    Serta kompatibilitas dengan kode lama ("CONFIRMED", "WAIT", "PASSED", "FAILED").
    """
    def __new__(cls, val: str, code: Optional[str] = None):
        obj = str.__new__(cls, val)
        obj.code = code or val
        return obj

    def __eq__(self, other):
        if not isinstance(other, str):
            return False
        s_self = str(self).upper()
        s_other = other.upper()
        if s_self == s_other:
            return True
        c_code = getattr(self, "code", "").upper()
        if c_code == s_other:
            return True
        if s_self in ("PASS", "CONFIRMED", "PASSED") and s_other in ("PASS", "CONFIRMED", "PASSED"):
            return True
        if s_self in ("FAIL", "WAIT", "FAILED", "WEAK") and s_other in ("FAIL", "WAIT", "FAILED", "WEAK"):
            return True
        return False

    def __hash__(self):
        return hash(str(self))


def sanitize_wait_reason(text: str) -> str:
    """
    Menghilangkan duplikasi awalan 'WAIT:', pengulangan frasa identik,
    dan tanda baca ganda (e.g. '..', '::', ';;') dari alasan WAIT.
    """
    if not text:
        return ""
    cleaned = str(text).strip()
    while re.match(r"^\s*WAIT\s*:\s*", cleaned, flags=re.IGNORECASE):
        cleaned = re.sub(r"^\s*WAIT\s*:\s*", "", cleaned, count=1, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"\.{2,}", ".", cleaned)
    cleaned = re.sub(r":{2,}", ":", cleaned)
    cleaned = re.sub(r";{2,}", ";", cleaned)
    cleaned = re.sub(r",\s*,+", ",", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _format_radar_status_clause(
    radar_qualified_count: int,
    far_zones_count: int = 0,
    invalidated_count: int = 0,
    mitigated_count: int = 0,
    expired_count: int = 0,
) -> str:
    """Bangun kalimat status zona yang akurat berdasarkan jumlah status aktual."""
    if radar_qualified_count > 0:
        active_part = f"{radar_qualified_count} zona Radar Qualified aktif dipantau"
    else:
        active_part = "Tidak ada zona Radar Qualified yang aktif dipantau"

    excluded_parts: List[str] = []
    if far_zones_count > 0:
        excluded_parts.append(f"{far_zones_count} zona Far Zone Excluded (dikeluarkan dari pemantauan intraday)")
    if invalidated_count > 0:
        excluded_parts.append(f"{invalidated_count} zona Invalidated (tidak valid)")
    if mitigated_count > 0:
        excluded_parts.append(f"{mitigated_count} zona Mitigated (sudah termitigasi)")
    if expired_count > 0:
        excluded_parts.append(f"{expired_count} zona Expired (kedaluwarsa)")

    if excluded_parts:
        return f"{active_part}; {'; '.join(excluded_parts)}."
    return f"{active_part}."


def is_zone_active_and_valid(z: Union[Zone, Dict[str, Any]], current_price: Optional[float] = None) -> bool:
    """
    Pemeriksaan integritas validitas struktural zona SMC:
    Menjamin hard rejection jika:
      - is_valid == False
      - Rejection reason mengandung 'invalid', 'expired', 'dimitigasi', 'mitigated', 'lebar', 'sempit'
      - Freshness mengandung 'INVALID', 'EXPIRED', 'MITIGATED'
      - Status zona == 'INVALIDATED', 'EXPIRED', 'MITIGATED'
      - is_expired == True
      - touch_count > 2
      - Origin BOS invalid
      - Current price telah menembus level distal / invalidasi (breached)
    """
    if z is None:
        return False
    zd = z.to_dict() if hasattr(z, "to_dict") else z
    if not isinstance(zd, dict):
        return False
    if not zd.get("is_valid", True):
        return False
    rej = str(zd.get("rejection_reason", "")).lower()
    if any(k in rej for k in ("invalid", "expired", "dimitigasi", "mitigated", "terlalu lebar", "terlalu sempit")):
        return False
    fresh = str(zd.get("freshness", "")).upper()
    if any(k in fresh for k in ("INVALID", "EXPIRED", "MITIGATED")):
        return False
    st = str(zd.get("status", "")).upper()
    if st in ("INVALIDATED", "EXPIRED", "MITIGATED"):
        return False
    if zd.get("is_expired"):
        return False
    if int(zd.get("touch_count", 0)) > 2:
        return False
    orig_bos = zd.get("origin_bos") or {}
    if isinstance(orig_bos, dict) and not orig_bos.get("origin_valid", True):
        return False
    if current_price is not None and isinstance(current_price, (int, float)) and math.isfinite(current_price) and current_price > 0:
        z_type = str(zd.get("zone_type", "")).upper()
        lo = float(zd.get("price_low", 0.0))
        hi = float(zd.get("price_high", 0.0))
        inval = float(zd.get("invalidation_price") or (lo if z_type == "DEMAND" else hi))
        if z_type == "DEMAND" and current_price < inval:
            return False
        if z_type == "SUPPLY" and current_price > inval:
            return False
    return True


class ZoneDistanceResult:
    __slots__ = (
        "distance",
        "status",
        "reference_price",
        "reference_price_type",
        "nearest_boundary",
        "atr_value",
        "distance_atr_ratio",
        "ratio_str",
    )

    def __init__(
        self,
        distance: float,
        status: str,
        reference_price: float = 0.0,
        reference_price_type: str = "MID",
        nearest_boundary: float = 0.0,
        atr_value: Optional[float] = None,
        distance_atr_ratio: Optional[float] = None,
        ratio_str: str = "",
    ):
        self.distance = distance
        self.status = status
        self.reference_price = reference_price
        self.reference_price_type = reference_price_type
        self.nearest_boundary = nearest_boundary
        self.atr_value = atr_value
        self.distance_atr_ratio = distance_atr_ratio
        self.ratio_str = ratio_str

    def __iter__(self):
        # Kompatibilitas mundur: dist, status = calculate_zone_distance(...)
        return iter((self.distance, self.status))

    def __getitem__(self, index):
        return (self.distance, self.status)[index]

    def __repr__(self):
        return (
            f"ZoneDistanceResult(distance={self.distance}, status={self.status!r}, "
            f"reference_price={self.reference_price}, reference_price_type={self.reference_price_type!r}, "
            f"nearest_boundary={self.nearest_boundary}, distance_atr_ratio={self.distance_atr_ratio}, "
            f"ratio_str={self.ratio_str!r})"
        )


def calculate_zone_distance(
    zone_low: float,
    zone_high: float,
    current_price: float,
    digits: int = 2,
    zone_type: Optional[str] = None,
    bid: Optional[float] = None,
    ask: Optional[float] = None,
    atr: Optional[float] = None,
) -> ZoneDistanceResult:
    """
    Kalkulasi jarak harga snapshot terhadap batas zona terdekat (proximal edge) dengan presisi digits.
    Mendukung konteks Bid/Ask untuk BUY (Demand) vs SELL (Supply):

    1. SUPPLY (SELL):
       - Proximal edge: zone_low (batas terdekat yang dicapai harga dari bawah).
       - Distal edge: zone_high (level pembatalan / invalidasi jika tertembus ke atas).
       - Harga referensi eksekusi / kenaikan: Ask (atau Mid / current_price jika Ask tidak tersedia).
       - Jika Ask < zone_low: zona berada di atas harga -> status = "ABOVE_PRICE", dist = zone_low - Ask.
       - Jika Bid > zone_high: harga telah menembus distal edge -> status = "BELOW_PRICE" (BREACHED), dist = Bid - zone_high.
       - Jika berada di antara zone_low dan zone_high: dist = 0.0, status = "INSIDE_ZONE".

    2. DEMAND (BUY):
       - Proximal edge: zone_high (batas terdekat yang dicapai harga dari atas).
       - Distal edge: zone_low (level pembatalan / invalidasi jika tertembus ke bawah).
       - Harga referensi eksekusi / penurunan: Bid (atau Mid / current_price jika Bid tidak tersedia).
       - Jika Bid > zone_high: zona berada di bawah harga -> status = "BELOW_PRICE", dist = Bid - zone_high.
       - Jika Ask < zone_low: harga telah menembus distal edge -> status = "ABOVE_PRICE" (BREACHED), dist = zone_low - Ask.
       - Jika berada di antara zone_low dan zone_high: dist = 0.0, status = "INSIDE_ZONE".

    3. Tanpa Tipe / Default Mid Price:
       - current_price < zone_low: dist = zone_low - current_price, status = "ABOVE_PRICE", nearest = zone_low.
       - current_price > zone_high: dist = current_price - zone_high, status = "BELOW_PRICE", nearest = zone_high.
       - zone_low <= current_price <= zone_high: dist = 0.0, status = "INSIDE_ZONE".

    Normalisasi ATR Ratio dihitung secara langsung tanpa pembulatan prematur.
    """
    eps = 1e-7
    z_type = (zone_type or "").upper()
    c_price = float(current_price) if (current_price is not None and math.isfinite(current_price)) else 0.0
    lo = float(zone_low)
    hi = float(zone_high)

    if z_type == "SUPPLY":
        nearest_bnd = lo
        if ask is not None and math.isfinite(ask) and ask > 0:
            ref_p = float(ask)
            ref_t = "ASK"
        elif c_price > 0:
            ref_p = c_price
            ref_t = "MID"
        elif bid is not None and math.isfinite(bid) and bid > 0:
            ref_p = float(bid)
            ref_t = "BID"
        else:
            ref_p = 0.0
            ref_t = "UNKNOWN"

        if ref_p < lo - eps:
            raw_d = lo - ref_p
            status = "ABOVE_PRICE"
        elif ref_p > hi + eps:
            raw_d = ref_p - hi
            status = "BELOW_PRICE"
        else:
            raw_d = 0.0
            status = "INSIDE_ZONE"

    elif z_type == "DEMAND":
        nearest_bnd = hi
        if bid is not None and math.isfinite(bid) and bid > 0:
            ref_p = float(bid)
            ref_t = "BID"
        elif c_price > 0:
            ref_p = c_price
            ref_t = "MID"
        elif ask is not None and math.isfinite(ask) and ask > 0:
            ref_p = float(ask)
            ref_t = "ASK"
        else:
            ref_p = 0.0
            ref_t = "UNKNOWN"

        if ref_p > hi + eps:
            raw_d = ref_p - hi
            status = "BELOW_PRICE"
        elif ref_p < lo - eps:
            raw_d = lo - ref_p
            status = "ABOVE_PRICE"
        else:
            raw_d = 0.0
            status = "INSIDE_ZONE"

    else:
        ref_p = c_price
        ref_t = "MID"
        if ref_p < lo - eps:
            raw_d = lo - ref_p
            nearest_bnd = lo
            status = "ABOVE_PRICE"
        elif ref_p > hi + eps:
            raw_d = ref_p - hi
            nearest_bnd = hi
            status = "BELOW_PRICE"
        else:
            raw_d = 0.0
            nearest_bnd = round((lo + hi) / 2.0, digits)
            status = "INSIDE_ZONE"

    # Perhitungan rasio ATR tanpa pembulatan awal
    atr_val = float(atr) if (atr is not None and isinstance(atr, (int, float)) and math.isfinite(atr) and atr > 0) else None
    if atr_val is not None:
        ratio_val = round(raw_d / atr_val, 2)
        ratio_str = f"{ratio_val:.1f}x"
    else:
        ratio_val = None
        ratio_str = "DATA INCOMPLETE"

    dist_val = round(raw_d, digits)
    return ZoneDistanceResult(
        distance=dist_val,
        status=status,
        reference_price=round(ref_p, digits),
        reference_price_type=ref_t,
        nearest_boundary=round(nearest_bnd, digits),
        atr_value=round(atr_val, digits) if atr_val is not None else None,
        distance_atr_ratio=ratio_val,
        ratio_str=ratio_str,
    )


def calculate_zone_approach(
    distance: float,
    status: str,
    atr: float = 2.0,
) -> str:
    """
    Menentukan status pendekatan harga terhadap zona:
    - Jika status == "INSIDE_ZONE" atau distance == 0.0: ZoneStatus("ZONE_TOUCHED")
    - Jika distance <= max(3.0, 1.5 * atr): "APPROACHING"
    - Jika distance > max(35.0, 15.0 * atr): "FAR"
    - Selainnya: "UNTOUCHED"
    """
    if status == "INSIDE_ZONE" or distance == 0.0:
        return ZoneStatus("ZONE_TOUCHED")
    atr_is_valid = (isinstance(atr, (int, float)) and math.isfinite(atr) and atr > 0)
    atr_val = float(atr) if atr_is_valid else None
    ref_atr = atr_val if atr_val is not None else 2.0
    far_threshold = max(35.0, 15.0 * ref_atr)
    approach_threshold = max(3.0, 1.5 * ref_atr)
    if distance <= approach_threshold:
        return "APPROACHING"
    if distance > far_threshold:
        return "FAR"
    return "UNTOUCHED"


# =============================================================================
# MULTI-TIER RISK CLASSIFICATION & HYBRID ENTRY INTEGRATION
# =============================================================================
def extract_atr_diagnostics(
    frames: Optional[Dict[str, pd.DataFrame]],
    preferred_tf: str = "H1",
    atr_override: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Ekstrak informasi diagnostik ATR secara transparan tanpa fallback diam-diam ke angka statis.
    """
    if frames and isinstance(frames, dict):
        df = frames.get(preferred_tf)
        if df is not None and len(df) >= 14 and "ATR14" in df.columns:
            raw_atr = df["ATR14"].iloc[-1]
            if isinstance(raw_atr, (int, float, np.floating)) and math.isfinite(float(raw_atr)) and float(raw_atr) > 0:
                raw_ts = None
                if len(df.index) > 0 and isinstance(df.index[-1], (pd.Timestamp, datetime)):
                    raw_ts = df.index[-1]
                elif "Time" in df.columns and len(df["Time"]) > 0:
                    raw_ts = df["Time"].iloc[-1]
                elif "time" in df.columns and len(df["time"]) > 0:
                    raw_ts = df["time"].iloc[-1]

                if raw_ts is not None:
                    last_ts = pd.Timestamp(raw_ts)
                    if last_ts.tzinfo is not None:
                        ts_str = last_ts.strftime("%Y-%m-%d %H:%M:%S %Z").strip()
                    else:
                        ts_str = f"{last_ts.strftime('%Y-%m-%d %H:%M:%S')} UTC (MT5 Server)"
                elif len(df.index) > 0:
                    ts_str = str(df.index[-1])
                else:
                    ts_str = "N/A"
                return {
                    "atr_source_timeframe": preferred_tf,
                    "atr_period": 14,
                    "atr_value": round(float(raw_atr), 4),
                    "atr_candle_timestamp": ts_str,
                    "atr_calculation_status": "VALID",
                    "atr_fallback_status": f"NONE (Primary {preferred_tf})",
                    "atr_candle_mode": "CLOSED_CANDLE",
                }
        return {
            "atr_source_timeframe": preferred_tf,
            "atr_period": 14,
            "atr_value": None,
            "atr_candle_timestamp": None,
            "atr_calculation_status": "DATA_INCOMPLETE",
            "atr_fallback_status": f"UNAVAILABLE ({preferred_tf} ATR tidak tersedia/invalid)",
            "atr_candle_mode": "CLOSED_CANDLE",
        }

    if atr_override is not None and isinstance(atr_override, (int, float, np.floating)) and math.isfinite(float(atr_override)) and float(atr_override) > 0:
        return {
            "atr_source_timeframe": preferred_tf,
            "atr_period": 14,
            "atr_value": round(float(atr_override), 4),
            "atr_candle_timestamp": "PARAM_PROVIDED",
            "atr_calculation_status": "VALID",
            "atr_fallback_status": "NONE (Explicit Parameter)",
            "atr_candle_mode": "CLOSED_CANDLE",
        }

    return {
        "atr_source_timeframe": preferred_tf,
        "atr_period": 14,
        "atr_value": None,
        "atr_candle_timestamp": None,
        "atr_calculation_status": "DATA_INCOMPLETE",
        "atr_fallback_status": "UNAVAILABLE",
        "atr_candle_mode": "CLOSED_CANDLE",
    }


def calculate_distance_atr_ratio(distance: float, atr: Optional[float]) -> Tuple[Optional[float], str]:
    """
    Normalisasi jarak zona terhadap volatilitas (ATR).
    Formula: Distance ATR Ratio = Absolute Distance To Zone / ATR Relevant Timeframe.
    Jika ATR tidak tersedia atau invalid, mengembalikan (None, 'DATA INCOMPLETE').
    """
    if atr is None or not (isinstance(atr, (int, float)) and math.isfinite(atr) and atr > 0):
        return None, "DATA INCOMPLETE"
    if distance is None or not (isinstance(distance, (int, float)) and math.isfinite(distance)):
        return None, "DATA INCOMPLETE"
    ratio = round(abs(float(distance)) / float(atr), 2)
    return ratio, f"{ratio:.1f}x"


def classify_zone_risk_tier(
    zone: Union[Zone, Dict[str, Any]],
    current_price: float,
    htf_bias: str = "NETRAL",
    h1_structure: str = "NETRAL",
    atr: Optional[float] = None,
    bid: Optional[float] = None,
    ask: Optional[float] = None,
) -> str:
    """
    Three-Tier Risk Classification Engine dengan 8 Tahap Validasi SMC:
      1. Data Completeness
      2. Structural Validity (Hard Rejection jika INVALID)
      3. Invalidation Status
      4. Mitigation Status
      5. Freshness
      6. Quality Scoring
      7. Risk Classification (Distance ATR Ratio, Volatility, HTF Alignment, TF, Zone Strength)
      8. Ranking
      
      Returns:
        "LOW_RISK" / "MEDIUM_RISK" / "HIGH_RISK" / "FAR_ZONE" / "UNCLASSIFIED"
    """
    zd = zone.to_dict() if hasattr(zone, "to_dict") else dict(zone)

    def _finalize(tier: str, reason: str) -> str:
        zd["risk_tier_reason"] = reason
        if hasattr(zone, "risk_tier_reason"):
            zone.risk_tier_reason = reason
        elif isinstance(zone, dict):
            zone["risk_tier_reason"] = reason
        return tier

    tf = str(zd.get("timeframe", ""))
    z_type = str(zd.get("zone_type", "")).upper()
    score = float(zd.get("score", 0.0))
    score_st = zd.get("scoring_status", "VALID")
    fresh = str(zd.get("freshness", "FRESH")).upper()
    grade = str(zd.get("grade", "B"))
    lo = float(zd.get("price_low", 0.0))
    hi = float(zd.get("price_high", 0.0))
    inval = float(zd.get("invalidation_price", lo if z_type == "DEMAND" else hi))
    is_valid = bool(zd.get("is_valid", True))
    origin_ok = bool((zd.get("origin_bos") or {}).get("origin_valid", True))
    rejection = str(zd.get("rejection_reason", "")).lower()
    touch_count = int(zd.get("touch_count", 0))

    # --- TAHAP 1: DATA COMPLETENESS ---
    if score_st == "DATA_INCOMPLETE" or score is None:
        return _finalize("UNCLASSIFIED", "Data teknikal tidak lengkap (DATA_INCOMPLETE)")
    if not (isinstance(current_price, (int, float)) and math.isfinite(current_price) and current_price > 0):
        return _finalize("UNCLASSIFIED", "Harga running tidak valid")
    if lo <= 0 or hi <= 0 or hi < lo:
        return _finalize("UNCLASSIFIED", "Batas zona tidak valid")

    # --- TAHAP 2: STRUCTURAL VALIDITY (HARD REJECTION) ---
    # Structural Validity INVALID harus menjadi hard rejection untuk kandidat Multi-Tier aktif
    if not is_valid or not origin_ok or "invalid" in rejection:
        return _finalize("UNCLASSIFIED", f"Struktur zona tidak valid / origin BOS patah ({zd.get('rejection_reason') or 'INVALID'})")

    # --- TAHAP 3: INVALIDATION STATUS ---
    if fresh == "INVALIDATED" or zd.get("status") == "INVALIDATED":
        return _finalize("UNCLASSIFIED", "Struktur zona telah invalid / tertembus")
    if z_type == "DEMAND" and current_price < inval:
        return _finalize("UNCLASSIFIED", f"Level invalidasi Demand tertembus (harga {current_price:.2f} < {inval:.2f})")
    if z_type == "SUPPLY" and current_price > inval:
        return _finalize("UNCLASSIFIED", f"Level invalidasi Supply tertembus (harga {current_price:.2f} > {inval:.2f})")

    # --- TAHAP 4: MITIGATION & EXPIRATION STATUS ---
    if fresh == "EXPIRED" or zd.get("status") == "EXPIRED" or zd.get("is_expired") or "expired" in rejection:
        return _finalize("UNCLASSIFIED", "Batas waktu pemantauan zona telah habis (EXPIRED)")
    if touch_count > 2 or fresh == "MITIGATED" or zd.get("status") == "MITIGATED" or "dimitigasi" in rejection:
        return _finalize("UNCLASSIFIED", f"Zona telah termitigasi ({fresh} / touch count: {touch_count})")

    # --- TAHAP 5: FRESHNESS ---
    if fresh not in ("FRESH", "TESTED_1X", "TESTED_2X"):
        return _finalize("UNCLASSIFIED", f"Status freshness ({fresh}) tidak memenuhi syarat radar")

    # --- TAHAP 6: QUALITY SCORING ---
    if score < 50.0:
        return _finalize("UNCLASSIFIED", f"Skor teknikal ({score:.1f}) di bawah ambang batas minimal 50.0")

    # --- TAHAP 7: RISK CLASSIFICATION & DISTANCE ATR RATIO ---
    # Hitung jarak aktual dinamis terhadap current_price
    if lo > 0 and hi > 0 and current_price > 0:
        dist_res = calculate_zone_distance(lo, hi, current_price, digits=2, zone_type=z_type, bid=bid, ask=ask)
        dist = dist_res.distance
    else:
        dist = float(zd.get("distance", 0.0))

    dist_ratio, _ = calculate_distance_atr_ratio(dist, atr)
    if dist_ratio is None:
        return _finalize("DATA_INCOMPLETE", "ATR tidak valid atau tidak tersedia — klasifikasi Multi-Tier ditangguhkan (DATA INCOMPLETE)")
    dist_ratio_str = f"{dist_ratio:.1f}x ATR"

    # Filter zona makro yang terlalu jauh untuk trading intraday aktif (> 25.0 ATR atau > 75.0 pt)
    over_atr = (dist_ratio is not None and dist_ratio > LOW_RISK_MAX_ATR_RATIO)
    over_pt = (dist > LOW_RISK_MAX_DISTANCE_PT)
    if over_atr or over_pt:
        if over_atr and over_pt:
            exceed_desc = f"melebihi batas ATR ratio (> {LOW_RISK_MAX_ATR_RATIO:.1f}x ATR) dan jarak absolut (> {LOW_RISK_MAX_DISTANCE_PT:.1f} pt)"
        elif over_atr:
            exceed_desc = f"melebihi batas ATR ratio maksimal (> {LOW_RISK_MAX_ATR_RATIO:.1f}x ATR)"
        else:
            exceed_desc = f"melebihi batas jarak absolut maksimal (> {LOW_RISK_MAX_DISTANCE_PT:.1f} pt)"
        return _finalize("FAR_ZONE", f"Jarak zona ({dist:.2f} pt / {dist_ratio_str}) {exceed_desc}")

    w = round(abs(hi - lo), 2)
    is_abnormal_width = (w > 25.0 or w < 0.5)

    # Evaluasi komponen timeframe secara eksak (mendukung zona gabungan seperti M15+M5 tanpa mencocokkan M5 murni ke M15)
    tf_parts = [p.strip().upper() for p in tf.split("+") if p.strip()]
    has_htf = any(t in tf_parts for t in ("H4", "H1", "D1", "W1"))
    has_mtf = any(t in tf_parts for t in ("M15", "M30", "H1"))
    has_ltf = any(t in tf_parts for t in ("M5", "M15"))

    # HTF Alignment
    is_aligned = False
    if htf_bias == "BULLISH" and z_type == "DEMAND":
        is_aligned = True
    elif htf_bias == "BEARISH" and z_type == "SUPPLY":
        is_aligned = True

    # Ambang jarak proporsional dan volatilitas (ATR)
    is_close = (dist_ratio is not None and dist_ratio <= HIGH_RISK_MAX_ATR_RATIO and dist <= HIGH_RISK_MAX_DISTANCE_PT) or (dist_ratio is None and dist <= HIGH_RISK_MAX_DISTANCE_PT)
    is_moderate_dist = (dist_ratio is not None and dist_ratio <= MEDIUM_RISK_MAX_ATR_RATIO and dist <= MEDIUM_RISK_MAX_DISTANCE_PT) or (dist_ratio is None and dist <= MEDIUM_RISK_MAX_DISTANCE_PT)
    is_conservative_dist = (dist_ratio is not None and dist_ratio <= LOW_RISK_MAX_ATR_RATIO and dist <= LOW_RISK_MAX_DISTANCE_PT) or (dist_ratio is None and dist <= LOW_RISK_MAX_DISTANCE_PT)

    # 1. Kategori LOW RISK (Conservative Zone)
    # H1 / H4, HTF Alignment Kuat, Grade A / A+ (Score >= 75.0), Fresh / Tested 1x, Jarak wajar (<= 25 ATR DAN <= 75 pt)
    if has_htf and is_aligned and (grade in ("A+", "A") or score >= LOW_RISK_MIN_SCORE) and fresh in ("FRESH", "TESTED_1X") and is_conservative_dist and not is_abnormal_width:
        # Zona LTF murni (M5/M15) tidak boleh masuk ke Low Risk konservatif
        if not (has_ltf and "H" not in tf):
            return _finalize("LOW_RISK", f"Konservatif HTF ({tf}): Grade {grade} (Skor {score:.1f}), Fresh, selaras bias {htf_bias}, jarak proporsional ({dist:.2f} pt / {dist_ratio_str})")

    # 2. Kategori HIGH RISK (Aggressive Zone)
    # Dekat dengan harga (<= 6.0 ATR DAN <= 18 pt), M5/M15/M30 (atau H1 dekat), Fresh, Score >= 55.0
    if is_close and score >= HIGH_RISK_MIN_SCORE and fresh in ("FRESH", "TESTED_1X", "TESTED_2X") and not is_abnormal_width:
        align_tag = "Pro-Trend" if is_aligned else ("Countertrend" if (htf_bias in ("BULLISH", "BEARISH")) else "Netral")
        return _finalize("HIGH_RISK", f"Agresif {align_tag} ({tf}): Dekat harga ({dist:.2f} pt / {dist_ratio_str}), Grade {grade} (Skor {score:.1f}), {fresh}")

    # 3. Kategori MEDIUM RISK (Balanced Zone)
    # M15, M30, H1, Score >= 65.0, Jarak proporsional (<= 16.0 ATR DAN <= 50 pt), Fresh/Tested 1x
    if has_mtf and score >= MEDIUM_RISK_MIN_SCORE and is_moderate_dist and fresh in ("FRESH", "TESTED_1X") and not is_abnormal_width:
        if is_aligned:
            align_desc = f"Balanced Pro-Trend (Searah bias HTF {htf_bias})"
        elif htf_bias in ("BULLISH", "BEARISH"):
            align_desc = f"Balanced Countertrend (Melawan bias HTF {htf_bias} — Pantau Reaksi Saja)"
        else:
            align_desc = f"Balanced Netral (Bias HTF {htf_bias})"
        return _finalize("MEDIUM_RISK", f"Moderat {align_desc} ({tf}): Grade {grade} (Skor {score:.1f}), {fresh}, jarak seimbang ({dist:.2f} pt / {dist_ratio_str})")

    # Fallback terkontrol dalam batas jarak ketat (dilarang menerima zona jarak berlebih atau TF tidak sesuai)
    if is_close and score >= HIGH_RISK_MIN_SCORE and not is_abnormal_width:
        align_tag = "Pro-Trend" if is_aligned else ("Countertrend" if (htf_bias in ("BULLISH", "BEARISH")) else "Netral")
        return _finalize("HIGH_RISK", f"Agresif Dekat {align_tag} ({tf}): Skor {score:.1f}, jarak {dist:.2f} pt ({dist_ratio_str})")
    if has_mtf and is_moderate_dist and score >= MEDIUM_RISK_MIN_SCORE and fresh in ("FRESH", "TESTED_1X") and not is_abnormal_width:
        align_tag = "Pro-Trend" if is_aligned else ("Countertrend" if (htf_bias in ("BULLISH", "BEARISH")) else "Netral")
        return _finalize("MEDIUM_RISK", f"Moderat {align_tag} ({tf}): Skor {score:.1f}, jarak {dist:.2f} pt ({dist_ratio_str})")
    if has_htf and is_aligned and is_conservative_dist and score >= LOW_RISK_MIN_SCORE and fresh in ("FRESH", "TESTED_1X") and not is_abnormal_width:
        return _finalize("LOW_RISK", f"Konservatif HTF ({tf}): Skor {score:.1f}, selaras bias {htf_bias}, jarak {dist:.2f} pt")

    missed = []
    if not is_close:
        missed.append(f"High Risk butuh jarak <= {HIGH_RISK_MAX_ATR_RATIO:.1f}x ATR & <= {HIGH_RISK_MAX_DISTANCE_PT:.1f} pt (aktual: {dist:.2f} pt / {dist_ratio_str})")
    elif score < HIGH_RISK_MIN_SCORE:
        missed.append(f"High Risk butuh skor >= {HIGH_RISK_MIN_SCORE:.1f} (aktual: {score:.1f})")

    if not has_mtf:
        missed.append(f"Medium Risk butuh TF M15/M30/H1 (aktual TF: {tf})")
    elif not is_moderate_dist:
        missed.append(f"Medium Risk butuh jarak <= {MEDIUM_RISK_MAX_ATR_RATIO:.1f}x ATR & <= {MEDIUM_RISK_MAX_DISTANCE_PT:.1f} pt (aktual: {dist:.2f} pt / {dist_ratio_str})")
    elif score < MEDIUM_RISK_MIN_SCORE:
        missed.append(f"Medium Risk butuh skor >= {MEDIUM_RISK_MIN_SCORE:.1f} (aktual: {score:.1f})")

    if not has_htf:
        missed.append(f"Low Risk butuh TF H1/H4 (aktual TF: {tf})")
    elif not is_aligned:
        missed.append(f"Low Risk butuh searah bias HTF {htf_bias}")
    elif not is_conservative_dist:
        missed.append(f"Low Risk butuh jarak <= {LOW_RISK_MAX_ATR_RATIO:.1f}x ATR & <= {LOW_RISK_MAX_DISTANCE_PT:.1f} pt (aktual: {dist:.2f} pt / {dist_ratio_str})")
    elif score < LOW_RISK_MIN_SCORE:
        missed.append(f"Low Risk butuh skor >= {LOW_RISK_MIN_SCORE:.1f} (aktual: {score:.1f})")

    if is_abnormal_width:
        missed.append(f"Lebar zona abnormal ({w:.2f} pt)")

    detail_str = "; ".join(missed) if missed else f"Skor: {score:.1f}, Grade: {grade}, Jarak: {dist:.2f} pt"
    return _finalize("UNCLASSIFIED", f"Tidak memenuhi kriteria High/Medium/Low Risk — {detail_str}")


def evaluate_zone_hybrid_eligibility(
    zone: Union[Zone, Dict[str, Any]],
    current_price: float,
    frames: Optional[Dict[str, pd.DataFrame]] = None,
    htf_bias: str = "NETRAL",
    h1_struct: str = "NETRAL",
    m30_status: str = "CONFIRMED",
    m5_stages: Optional[Dict[str, Any]] = None,
    server_time: Any = None,
    atr: Optional[float] = None,
    bid: Optional[float] = None,
    ask: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Integrasi Hybrid Entry Engine untuk setiap zona di seluruh kategori:
    Menghasilkan status independen:
      - Radar Eligibility: QUALIFIED / NOT_QUALIFIED
      - Pending Eligibility: READY / NOT_QUALIFIED (dengan alasan spesifik)
      - Confirmed Entry Eligibility: CONFIRMED / WAIT / FAILED
    Serta parameter trading kalkulasi Python: Entry, SL, TP1, TP2, Risk, Reward, RR.
    """
    zd = zone.to_dict() if hasattr(zone, "to_dict") else dict(zone)
    z_obj = zone if isinstance(zone, Zone) else _zone_from_dict(zd)
    z_type = str(zd.get("zone_type", "")).upper()
    direction = "BUY" if z_type == "DEMAND" else "SELL"
    lo = float(zd.get("price_low", 0.0))
    hi = float(zd.get("price_high", 0.0))
    grade = str(zd.get("grade", "B"))
    score = float(zd.get("score", 0.0))
    fresh = str(zd.get("freshness", "FRESH")).upper()
    inval = float(zd.get("invalidation_price", lo if z_type == "DEMAND" else hi))
    is_valid = bool(zd.get("is_valid", True))
    origin_ok = bool((zd.get("origin_bos") or {}).get("origin_valid", True))
    rej = str(zd.get("rejection_reason", "")).lower()

    # 1. RADAR ELIGIBILITY
    radar_elig = "QUALIFIED"
    radar_reason = "Zona valid untuk pemantauan radar"
    if not is_valid or not origin_ok or "invalid" in rej:
        radar_elig = "NOT_QUALIFIED"
        radar_reason = f"Struktur zona tidak valid ({zd.get('rejection_reason') or 'INVALID'})"
    elif fresh == "INVALIDATED" or (z_type == "DEMAND" and current_price < inval) or (z_type == "SUPPLY" and current_price > inval) or zd.get("status") == "INVALIDATED":
        radar_elig = "NOT_QUALIFIED"
        radar_reason = "Zona telah tertembus (INVALIDATED)"
    elif fresh == "EXPIRED" or zd.get("status") == "EXPIRED" or zd.get("is_expired") or "expired" in rej:
        radar_elig = "NOT_QUALIFIED"
        radar_reason = "Batas waktu pemantauan zona telah habis (EXPIRED)"
    elif zd.get("touch_count", 0) > 2 or "dimitigasi" in rej or fresh == "MITIGATED" or zd.get("status") == "MITIGATED":
        radar_elig = "NOT_QUALIFIED"
        radar_reason = "Zona telah termitigasi penuh (> 2x touch)"
    elif score < 50.0:
        radar_elig = "NOT_QUALIFIED"
        radar_reason = f"Skor teknikal ({score:.1f}) di bawah ambang batas minimal 50.0"

    # 2. PENDING ELIGIBILITY (Aturan Ketat Pending Limit)
    # PENDING LIMIT WAJIB MEMENUHI:
    # - H4 searah
    # - H1 searah / tidak konflik
    # - M30 momentum tidak CONFLICT/INVALID/WEAK
    # - Zona searah
    # - Zona minimal Grade A (skor >= 75.0)
    # - Fresh & belum termitigasi (> 2x touch)
    # - Struktur valid
    # - Invalidation belum tertembus
    # - RR TP1 >= 1.5
    pending_elig = "NOT_QUALIFIED"
    pending_reason = ""
    pending_plan = None

    if radar_elig != "QUALIFIED":
        pending_reason = radar_reason
    elif z_type == "DEMAND" and htf_bias != "BULLISH":
        pending_reason = f"Bias H4 ({htf_bias}) tidak selaras dengan setup BUY LIMIT (harus BULLISH)"
    elif z_type == "SUPPLY" and htf_bias != "BEARISH":
        pending_reason = f"Bias H4 ({htf_bias}) tidak selaras dengan setup SELL LIMIT (harus BEARISH)"
    elif (htf_bias == "BULLISH" and h1_struct == "BEARISH") or (htf_bias == "BEARISH" and h1_struct == "BULLISH"):
        pending_reason = f"Konflik struktur H4 ({htf_bias}) vs H1 ({h1_struct})"
    elif m30_status in ("CONFLICT", "INVALID", "WEAK"):
        pending_reason = f"Momentum M30 tidak mendukung pending ({m30_status})"
    elif not (grade in ("A+", "A") or score >= LOW_RISK_MIN_SCORE):
        # High Risk Grade B boleh muncul di Radar, tetapi NOT QUALIFIED untuk Pending Limit!
        pending_reason = f"Kualitas zona ({grade} / Skor {score:.1f}) di bawah standar Grade A untuk Pending Limit"
    elif fresh not in ("FRESH", "TESTED_1X"):
        pending_reason = f"Status zona ({fresh}) sudah termitigasi"
    else:
        # Jalankan plan_pending_order untuk kalkulasi level dan verifikasi RR
        plan_res = plan_pending_order(
            z_obj, current_price, frames=frames, htf_bias=htf_bias,
            m30_status=m30_status, server_time=server_time, bid=bid, ask=ask
        )
        if plan_res.get("valid"):
            pending_elig = "READY"
            pending_plan = plan_res
            pending_reason = "Memenuhi seluruh 10 persyaratan Pending Limit (Grade A, H4/H1/M30 selaras, RR >= 1:1.5)"
        else:
            pending_reason = plan_res.get("rejection_reason") or "Tidak memenuhi parameter validasi pending order"

    # 3. CONFIRMED ENTRY ELIGIBILITY (M5 Triggers)
    confirmed_elig = "WAIT"
    confirmed_reason = ""
    if m5_stages:
        st_zt = m5_stages.get("zone_touch", {}).get("status")
        st_sw = m5_stages.get("liquidity_sweep", {}).get("status")
        st_tr = m5_stages.get("structure_trigger", {}).get("status")
        st_dp = m5_stages.get("displacement", {}).get("status")
        st_rr = m5_stages.get("risk_reward", {}).get("status")

        if st_zt != "CONFIRMED":
            confirmed_elig = "WAIT"
            confirmed_reason = "Harga belum menyentuh zona (Zone Touch belum terkonfirmasi)"
        elif st_sw != "CONFIRMED":
            confirmed_elig = "WAIT"
            confirmed_reason = "Menunggu konfirmasi Liquidity Sweep M5 di area zona"
        elif st_tr != "CONFIRMED":
            confirmed_elig = "WAIT"
            confirmed_reason = "Menunggu konfirmasi pembalikan/kelanjutan struktur CHoCH/BOS/CISD M5"
        elif st_dp != "CONFIRMED":
            confirmed_elig = "WAIT"
            confirmed_reason = "Menunggu dorongan displacement momentum M5"
        elif st_rr in ("FAILED", "WAIT"):
            confirmed_elig = "FAILED"
            confirmed_reason = "Risk/Reward kalkulasi M5 tidak memenuhi minimum 1:1.5"
        else:
            confirmed_elig = "CONFIRMED"
            confirmed_reason = "Seluruh 5 tahap konfirmasi M5 (Touch, Sweep, CHoCH/BOS, Displacement, RR) terpenuhi"
    else:
        confirmed_elig = "WAIT"
        confirmed_reason = "Menunggu data konfirmasi trigger M5"

    # Trade parameters (fallback jika pending_plan belum READY)
    entry_val = pending_plan.get("entry_price") if pending_plan else (hi if z_type == "DEMAND" else lo)
    sl_val = pending_plan.get("sl") if pending_plan else (getattr(z_obj, "sl_recommended", 0.0) or (lo - 2.0 if z_type == "DEMAND" else hi + 2.0))
    tp1_val = pending_plan.get("tp1") if pending_plan else (getattr(z_obj, "tp1_recommended", 0.0) or (entry_val + 2.0 * abs(entry_val - sl_val) if z_type == "DEMAND" else entry_val - 2.0 * abs(entry_val - sl_val)))
    tp2_val = pending_plan.get("tp2") if pending_plan else (getattr(z_obj, "tp2_recommended", 0.0) or (entry_val + 3.0 * abs(entry_val - sl_val) if z_type == "DEMAND" else entry_val - 3.0 * abs(entry_val - sl_val)))
    risk_val = pending_plan.get("risk") if pending_plan else round(abs(entry_val - sl_val), 2)
    reward_val = pending_plan.get("reward") if pending_plan else round(abs(tp1_val - entry_val), 2)
    rr_val = pending_plan.get("rr") if pending_plan else (round(reward_val / risk_val, 2) if risk_val > 0 else 0.0)

    return {
        "zone_id": zd.get("zone_id", ""),
        "direction": direction,
        "radar_eligibility": radar_elig,
        "radar_rejection_reason": radar_reason,
        "pending_eligibility": pending_elig,
        "pending_rejection_reason": pending_reason,
        "confirmed_entry_eligibility": confirmed_elig,
        "confirmed_rejection_reason": confirmed_reason,
        "entry_price": entry_val,
        "sl": sl_val,
        "tp1": tp1_val,
        "tp2": tp2_val,
        "risk": risk_val,
        "reward": reward_val,
        "rr": rr_val,
        "pending_plan": pending_plan,
    }


def prioritize_radar_zones(
    radar_zones: List[Union[Zone, Dict[str, Any]]],
    current_price: float,
    htf_bias: str = "NETRAL",
    h1_structure: str = "NETRAL",
    atr: float = 2.0,
    m5_stages: Optional[Dict[str, Any]] = None,
    digits: int = 2,
    validated_entry_zone: Optional[Union[Zone, Dict[str, Any]]] = None,
    frames: Optional[Dict[str, pd.DataFrame]] = None,
    m30_status: str = "CONFIRMED",
    server_time: Any = None,
    rejected_zones: Optional[List[Union[Zone, Dict[str, Any]]]] = None,
    primary_target_override: Optional[Union[Zone, Dict[str, Any]]] = None,
    bid: Optional[float] = None,
    ask: Optional[float] = None,
    snapshot_time: Optional[str] = None,
    snapshot_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Sistem Seleksi & Evaluasi Multi-Zona RADAR dengan 4 Kategori Terpisah & Multi-Tier SMC:
      A. PRIMARY DIRECTIONAL ZONE : Zona kualitas terbaik selaras bias HTF (H4/H1).
      B. NEAREST RADAR ZONE       : Zona valid terdekat dengan harga saat ini (bisa Demand/Supply).
      C. BEST VALIDATED ENTRY ZONE: Zona yang telah memenuhi SELURUH syarat validasi M5 + Risk/Reward.
                                    Wajib NONE jika belum ada zona yang terkonfirmasi penuh.
      D. ALTERNATIVE ZONE         : Zona cadangan/alternatif pro-trend berikutnya atau area reaksi valid.
      
      Multi-Tier Risk Classification:
        - HIGH RISK — AGGRESSIVE ZONE (Maksimal MULTI_TIER_MAX_ZONES): LTF/MTF dekat, score >= 55.0
        - MEDIUM RISK — BALANCED ZONE (Maksimal MULTI_TIER_MAX_ZONES): MTF/HTF balanced, score >= 65.0
        - LOW RISK — CONSERVATIVE ZONE (Maksimal MULTI_TIER_MAX_ZONES): HTF H1/H4, Grade A/A+, score >= 75.0
    
    Ketentuan Teknis:
      - Evaluasi seluruh kandidat zona secara independen.
      - Zona terdekat tidak otomatis menjadi zona terbaik.
      - Zona Grade A yang jauh juga tidak otomatis menjadi pilihan entry terbaik (terkena penalti jarak).
      - Zona yang melawan bias utama (countertrend) tetap dideteksi tetapi dilarang diprioritaskan sebagai peluang entry.
      - Status DATA INCOMPLETE dipertahankan secara transparan dan tidak disamarkan sebagai skor 0.0.
    """
    if not radar_zones:
        return {
            "primary_directional_zone": None,
            "nearest_radar_zone": None,
            "best_validated_entry_zone": None,
            "alternative_zone": None,
            "best_entry_monitoring_zone": None,
            "primary_zone": None,
            "secondary_zone": None,
            "countertrend_zone": None,
            "all_prioritized": [],
            "aligned_zones": [],
            "counter_zones": [],
            "multi_tier_zones": {"high_risk": [], "medium_risk": [], "low_risk": []},
            "high_risk_zones": [],
            "medium_risk_zones": [],
            "low_risk_zones": [],
            "best_high_risk": None,
            "best_medium_risk": None,
            "best_low_risk": None,
            "best_pending_candidate": None,
            "best_confirmed_candidate": None,
            "primary_reason": "NO RADAR ZONE DETECTED.",
            "distance_to_zone_str": "N/A",
            "missing_conditions": ["NO RADAR ZONE DETECTED."],
            "dynamic_wait_reason": "NO RADAR ZONE DETECTED.",
            "zone_scoring_summary": {
                "ranked_zones": [],
                "prioritized_reasons": [],
                "unprioritized_reasons": [],
                "has_incomplete_data": False,
            },
            "snapshot_id": snapshot_id,
            "snapshot_time": snapshot_time,
            "snapshot_price": round(float(current_price), 2) if isinstance(current_price, (int, float)) else None,
            "priority_calc_timestamp": snapshot_time or datetime.now().strftime("%Y-%m-%d %H:%M:%S WIB"),
        }

    atr_is_valid = (isinstance(atr, (int, float)) and math.isfinite(atr) and atr > 0)
    atr_val = float(atr) if atr_is_valid else None
    ref_atr = atr_val if atr_val is not None else 2.0
    far_threshold = max(35.0, 15.0 * ref_atr)
    approach_threshold = max(3.0, 1.5 * ref_atr)

    processed: List[Dict[str, Any]] = []
    for z in radar_zones:
        zd = z.to_dict() if hasattr(z, "to_dict") else dict(z)
        z_type = str(zd.get("zone_type", "")).upper()
        lo = float(zd.get("price_low", 0.0))
        hi = float(zd.get("price_high", 0.0))
        tf = str(zd.get("timeframe", "H1"))
        grd = str(zd.get("grade", "B"))
        score_st = zd.get("scoring_status", "VALID")
        raw_score = zd.get("score")
        if score_st == "DATA_INCOMPLETE" or raw_score is None:
            score = 0.0
            score_st = "DATA_INCOMPLETE"
        else:
            score = float(raw_score)
        fresh = str(zd.get("freshness", "FRESH")).upper()
        is_valid = bool(zd.get("is_valid", True))
        rejection = str(zd.get("rejection_reason", ""))
        origin_ok = bool((zd.get("origin_bos") or {}).get("origin_valid", True))
        inval = float(zd.get("invalidation_price") or (lo if z_type == "DEMAND" else hi))

        # 1. Jarak & Status Pendekatan (Snapshot Real-Time)
        dist_res = calculate_zone_distance(
            lo, hi, current_price, digits=digits, zone_type=z_type,
            bid=bid, ask=ask, atr=atr_val
        )
        dist = dist_res.distance
        dir_status = dist_res.status
        approach = calculate_zone_approach(dist, dir_status, atr=atr_val)

        # Normalisasi ATR Ratio
        dist_ratio, _ = calculate_distance_atr_ratio(dist, atr_val)
        zd["distance"] = dist
        zd["direction_status"] = dir_status
        zd["distance_status"] = dir_status
        zd["approach"] = approach
        zd["distance_atr_ratio"] = dist_ratio
        zd["atr_value"] = atr_val
        zd["reference_price"] = dist_res.reference_price
        zd["reference_price_type"] = dist_res.reference_price_type
        zd["nearest_boundary"] = dist_res.nearest_boundary
        if snapshot_time is not None:
            zd["snapshot_time"] = snapshot_time

        # Edge Proximal / Distal & Lebar Zone
        zd["proximal_edge"] = hi if z_type == "DEMAND" else lo
        zd["distal_edge"] = lo if z_type == "DEMAND" else hi
        zd["zone_width"] = round(abs(hi - lo), 2)

        # Status pemantauan zona MT5:
        # RADAR, APPROACHING, ZONE_TOUCHED, MITIGATED, INVALIDATED, EXPIRED
        if not is_valid or not origin_ok or "invalid" in rejection.lower() or fresh == "INVALIDATED" or (z_type == "DEMAND" and current_price < inval) or (z_type == "SUPPLY" and current_price > inval):
            zone_status = "INVALIDATED"
        elif zd.get("touch_count", 0) > 2 or fresh == "MITIGATED" or "dimitigasi" in rejection.lower():
            zone_status = "MITIGATED"
        elif approach in ("TOUCHED", "ZONE_TOUCHED"):
            zone_status = ZoneStatus("ZONE_TOUCHED")
        elif approach == "APPROACHING":
            zone_status = "APPROACHING"
        else:
            zone_status = "RADAR"
        zd["status"] = zone_status

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

        # 3. Skor Prioritas Komprehensif (Menyeimbangkan Kualitas Struktur, Bias, Jarak & Freshness)
        # Single Source of Truth: jika sudah dikunci untuk snapshot yang benar-benar sama, pertahankan nilai tunggalnya.
        snap_p_key = round(float(current_price), 2)
        _same_snap_id = (
            (snapshot_id is not None and zd.get("_priority_snapshot_id") == snapshot_id)
            or (snapshot_id is None and zd.get("_priority_snapshot_id") is None)
        )
        if (
            zd.get("_priority_locked")
            and zd.get("priority_score") is not None
            and zd.get("_priority_snapshot_price") == snap_p_key
            and _same_snap_id
        ):
            p_score = float(zd["priority_score"])
        else:
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
            elif "INVALID" in fresh or not is_valid or not origin_ok:
                p_score -= 200.0

            # Komponen jarak menggunakan bucket pendekatan untuk TOUCHED (+15), APPROACHING (+8,
            # yaitu 0 < dist <= max(3.0, 1.5 * ref_atr)), dan FAR (-25), serta penalti jarak
            # kontinu -min(15.0, dist * 0.15) hanya saat status pendekatan UNTOUCHED.
            if approach in ("TOUCHED", "ZONE_TOUCHED"):
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

            if score_st == "DATA_INCOMPLETE":
                p_score -= 100.0

        # Label prioritas
        labels = [str(zone_status)]
        if approach == "FAR":
            labels.append("FAR")
        if is_counter:
            labels.append("COUNTERTREND")

        # Risk Tier Classification (Three-Tier Engine)
        tier = classify_zone_risk_tier(
            zd, current_price, htf_bias, h1_structure,
            atr=atr_val, bid=bid, ask=ask
        )
        zd["risk_tier"] = tier

        # Hybrid Entry Eligibility (Radar, Pending, Confirmed M5)
        h_eval = evaluate_zone_hybrid_eligibility(
            zd, current_price, frames=frames, htf_bias=htf_bias,
            h1_struct=h1_structure, m30_status=m30_status,
            m5_stages=m5_stages, server_time=server_time, atr=atr_val,
            bid=bid, ask=ask
        )
        zd["radar_eligibility"] = h_eval["radar_eligibility"]
        zd["radar_rejection_reason"] = h_eval["radar_rejection_reason"]
        zd["pending_eligibility"] = h_eval["pending_eligibility"]
        zd["pending_rejection_reason"] = h_eval["pending_rejection_reason"]
        zd["confirmed_entry_eligibility"] = h_eval["confirmed_entry_eligibility"]
        zd["confirmed_rejection_reason"] = h_eval["confirmed_rejection_reason"]
        zd["trade_parameters"] = h_eval
        if h_eval.get("pending_plan"):
            zd["pending_plan"] = h_eval["pending_plan"]

        zd["score"] = score
        zd["scoring_status"] = score_st
        zd["grade"] = grd
        zd["freshness"] = fresh
        zd["timeframe"] = tf
        zd["zone_type"] = z_type
        zd["price_low"] = lo
        zd["price_high"] = hi
        zd["priority_score"] = round(p_score, 1)
        zd["_priority_locked"] = True
        zd["_priority_snapshot_price"] = snap_p_key
        zd["_priority_snapshot_id"] = snapshot_id
        zd["_priority_calc_timestamp"] = snapshot_time or datetime.now().strftime("%Y-%m-%d %H:%M:%S WIB")
        zd["is_aligned"] = is_aligned
        zd["is_counter"] = is_counter
        zd["priority_labels"] = labels
        processed.append(zd)

    # 4. Pemisahan 4 Kategori Evaluasi:
    # A. PRIMARY DIRECTIONAL ZONE: Kualitas terbaik selaras bias HTF
    # B. NEAREST RADAR ZONE: Jarak terdekat ke snapshot harga (tanpa memandang arah)
    # C. BEST VALIDATED ENTRY ZONE: Terverifikasi penuh (Trigger M5 + R:R), NONE jika belum lengkap
    # D. ALTERNATIVE ZONE: Zona cadangan pro-trend berikutnya atau area reaksi
    if htf_bias in ("BULLISH", "BEARISH"):
        aligned = [
            z for z in processed
            if z["is_aligned"]
            and is_zone_active_and_valid(z, current_price)
            and z.get("risk_tier") != "FAR_ZONE"
        ]
        counter = [
            z for z in processed
            if z["is_counter"]
            and is_zone_active_and_valid(z, current_price)
            and z.get("risk_tier") != "FAR_ZONE"
        ]
        aligned.sort(key=lambda x: -x["priority_score"])
        counter.sort(key=lambda x: x["distance"])  # Diurutkan dari jarak terdekat
        primary_directional = aligned[0] if aligned else None
        if primary_target_override:
            ov_dict = primary_target_override if isinstance(primary_target_override, dict) else primary_target_override.to_dict()
            ov_match = next((z for z in aligned if z.get("zone_id") == ov_dict.get("zone_id") or
                            (abs(float(z.get("price_low", 0.0)) - float(ov_dict.get("price_low", 0.0))) < 1e-4 and
                             abs(float(z.get("price_high", 0.0)) - float(ov_dict.get("price_high", 0.0))) < 1e-4)), None)
            if ov_match:
                primary_directional = ov_match
        secondary_aligned = aligned[1] if len(aligned) > 1 else None
        countertrend_nearest = counter[0] if counter else None
        primary_fallback = primary_directional
    else:  # NETRAL
        processed.sort(key=lambda x: -x["priority_score"])
        demands = [
            z for z in processed
            if z["zone_type"] == "DEMAND"
            and is_zone_active_and_valid(z, current_price)
            and z.get("risk_tier") != "FAR_ZONE"
        ]
        supplies = [
            z for z in processed
            if z["zone_type"] == "SUPPLY"
            and is_zone_active_and_valid(z, current_price)
            and z.get("risk_tier") != "FAR_ZONE"
        ]
        demands.sort(key=lambda x: x["distance"])
        supplies.sort(key=lambda x: x["distance"])
        primary_directional = None  # Pada HTF Netral, tidak ada directional bias
        primary_fallback = supplies[0] if supplies else (demands[0] if demands else None)
        secondary_aligned = demands[0] if (supplies and demands) else (supplies[1] if len(supplies) > 1 else None)
        countertrend_nearest = None
        aligned = [z for z in processed if is_zone_active_and_valid(z, current_price) and z.get("risk_tier") != "FAR_ZONE"]
        counter = []

    # B. NEAREST RADAR ZONE (Zona valid dengan jarak terkecil)
    valid_cands = [
        z for z in processed
        if is_zone_active_and_valid(z, current_price)
        and z.get("risk_tier") != "FAR_ZONE"
    ]
    if valid_cands:
        sorted_by_dist = sorted(valid_cands, key=lambda x: x["distance"])
        nearest_radar = sorted_by_dist[0]
    else:
        nearest_radar = None

    # C. BEST VALIDATED ENTRY ZONE (Hanya terisi jika SELURUH syarat validasi entry M5 terpenuhi)
    best_validated_entry = None
    if validated_entry_zone:
        vz_dict = validated_entry_zone if isinstance(validated_entry_zone, dict) else validated_entry_zone.to_dict()
        best_validated_entry = vz_dict

    # D. ALTERNATIVE ZONE
    alternative_zone = secondary_aligned if secondary_aligned is not None else countertrend_nearest

    # Backwards compatibility: Best Entry Monitoring Zone
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
        score_disp = f"Skor {primary_directional['score']:.1f}" if primary_directional.get("scoring_status") != "DATA_INCOMPLETE" else "DATA INCOMPLETE"
        primary_reason = (
            f"{primary_directional['timeframe']} {primary_directional['zone_type']} ({primary_directional['price_low']:.2f}-{primary_directional['price_high']:.2f}){app_lbl} "
            f"diprioritaskan karena {align_str}, Grade {primary_directional['grade']} ({score_disp}), {primary_directional['freshness']}, FVG {fvg_st}."
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
    confirmed_entry_rr_status = "NOT_CHECKED"
    eval_target = primary_directional if primary_directional is not None else (primary if htf_bias != "NETRAL" else None)
    if eval_target:
        st_zt = (m5_stages or {}).get("zone_touch", {}).get("status")
        st_sw = (m5_stages or {}).get("liquidity_sweep", {}).get("status")
        st_tr = (m5_stages or {}).get("structure_trigger", {}).get("status")
        st_dp = (m5_stages or {}).get("displacement", {}).get("status")
        st_rr = (m5_stages or {}).get("risk_reward", {}).get("status")

        is_touched = (eval_target.get("approach") in ("TOUCHED", "ZONE_TOUCHED") or st_zt in ("CONFIRMED", "PASS", "PASSED"))
        if not is_touched:
            missing.append(f"[ZONE_TOUCH]: Harga belum mencapai atau menyentuh {eval_target['zone_type'].capitalize()} ({eval_target['price_low']:.2f}-{eval_target['price_high']:.2f}, jarak {eval_target['distance']:.2f} pt)")
        else:
            if st_sw in ("CONFIRMED", "PASS", "PASSED"):
                pass
            elif st_sw in ("WAIT", "FAIL", "FAILED") and "jauh" in (m5_stages or {}).get("liquidity_sweep", {}).get("msg", "").lower():
                missing.append("[LIQUIDITY_SWEEP]: Sweep likuiditas M5 terjadi terlalu jauh dari area zona")
            else:
                want_sw = "Buy-Side Liquidity (EQH/Swing High)" if eval_target["zone_type"] == "SUPPLY" else "Sell-Side Liquidity (EQL/Swing Low)"
                missing.append(f"[LIQUIDITY_SWEEP]: Belum ada validasi sweep {want_sw} M5 di area zona")

            if st_tr in ("CONFIRMED", "PASS", "PASSED"):
                pass
            else:
                want_tr = "CHoCH/BOS/CISD bearish" if eval_target["zone_type"] == "SUPPLY" else "CHoCH/BOS/CISD bullish"
                missing.append(f"[STRUCTURE_TRIGGER]: Belum ada konfirmasi perubahan struktur {want_tr} M5")

            if st_dp in ("CONFIRMED", "PASS", "PASSED"):
                pass
            elif st_dp in ("WEAK", "WAIT", "FAIL", "FAILED"):
                dp_m = (m5_stages or {}).get("displacement", {}).get("msg", "Displacement lemah")
                missing.append(f"[DISPLACEMENT]: {dp_m}")
            else:
                missing.append("[DISPLACEMENT]: Belum ada konfirmasi dorongan displacement M5 searah")

            m5_triggers_ready = (st_tr in ("CONFIRMED", "PASS", "PASSED") and st_dp in ("CONFIRMED", "PASS", "PASSED"))
            if st_rr in ("CONFIRMED", "PASS", "PASSED"):
                confirmed_entry_rr_status = "PASSED"
            elif m5_triggers_ready:
                if st_rr in ("FAILED", "FAIL", "WAIT"):
                    confirmed_entry_rr_status = "FAILED"
                    missing.append("[CONFIRMED_ENTRY_RISK_REWARD]: Entry konfirmasi M5 tidak memenuhi Risk/Reward minimum 1:1.5")
                else:
                    confirmed_entry_rr_status = "WAIT"
            else:
                confirmed_entry_rr_status = "NOT_CHECKED"
    else:
        confirmed_entry_rr_status = "NOT_CHECKED"
        missing.append(f"Tidak ada zona yang selaras dengan arah bias HTF ({htf_bias})")

    # Alasan Dinamis Final WAIT & Structured WAIT Breakdown (Requirement 5 & 6)
    primary_blocker = ""
    pending_blockers = []
    confirmed_blockers = []
    sec_warnings = []

    ct_note = ""
    if countertrend:
        ct_type = countertrend["zone_type"]
        ct_range = f"{countertrend['price_low']:.2f}-{countertrend['price_high']:.2f}"
        opp_sig = "BUY" if ct_type == "DEMAND" else "SELL"
        ct_role = "area reaksi/support" if ct_type == "DEMAND" else "area reaksi/resistance"
        ct_note = f" {ct_type.capitalize()} di bawah harga ({ct_range}) hanya menjadi {ct_role}, bukan sinyal {opp_sig}."
        sec_warnings.append(f"Zona {ct_type.capitalize()} di {ct_range} berlawanan arah bias HTF ({htf_bias}) — hanya sebagai {ct_role}, bukan sinyal {opp_sig}.")

    if htf_bias == "NETRAL":
        primary_blocker = "Bias H4 NETRAL: Belum ada directional bias yang jelas. Seluruh zona dipantau sebagai batas range reaksi, bukan sinyal entry."
        dynamic_wait = f"WAIT: {primary_blocker}"
    elif (htf_bias == "BULLISH" and h1_structure == "BEARISH") or (htf_bias == "BEARISH" and h1_structure == "BULLISH"):
        primary_blocker = f"Konflik struktur Higher Timeframe: H4 {htf_bias} vs H1 {h1_structure}. Mode pasar tidak sinkron."
        dynamic_wait = f"WAIT: {primary_blocker}"
    elif eval_target:
        p_type = eval_target["zone_type"]
        p_name = f"{eval_target['timeframe']} {p_type}"
        p_range = f"{eval_target['price_low']:.2f}-{eval_target['price_high']:.2f}"
        p_dist = eval_target["distance"]
        p_app = eval_target["approach"]

        if p_app in ("TOUCHED", "ZONE_TOUCHED"):
            st_sw = (m5_stages or {}).get("liquidity_sweep", {}).get("status")
            st_tr = (m5_stages or {}).get("structure_trigger", {}).get("status")
            st_dp = (m5_stages or {}).get("displacement", {}).get("status")
            st_rr = (m5_stages or {}).get("risk_reward", {}).get("status")

            if st_sw not in ("CONFIRMED", "PASS", "PASSED"):
                sw_msg = (m5_stages or {}).get("liquidity_sweep", {}).get("msg", "")
                if "jauh" in str(sw_msg).lower():
                    primary_blocker = f"Harga telah menyentuh {p_name} ({p_range}), namun Liquidity Sweep M5 terjadi tidak relevan (terlalu jauh dari area zona). Menunggu sweep yang valid."
                else:
                    primary_blocker = f"Harga telah memasuki {p_name} ({p_range}), tetapi belum ada konfirmasi Liquidity Sweep M5 yang valid di area zona."
                dynamic_wait = f"WAIT: {primary_blocker}{ct_note}"
            elif st_tr not in ("CONFIRMED", "PASS", "PASSED"):
                primary_blocker = f"Harga telah menyentuh {p_name} dan membentuk Liquidity Sweep M5, tetapi belum terkonfirmasi pembalikan/kelanjutan struktur (CHoCH/BOS M5)."
                dynamic_wait = f"WAIT: {primary_blocker}{ct_note}"
            elif st_dp not in ("CONFIRMED", "PASS", "PASSED"):
                primary_blocker = f"Sweep dan CHoCH M5 terkonfirmasi pada {p_name}, namun displacement M5 masih lemah."
                dynamic_wait = f"WAIT: {primary_blocker}{ct_note}"
            elif st_rr in ("FAILED", "FAIL", "WAIT"):
                primary_blocker = f"Konfirmasi trigger M5 pada {p_name} terpenuhi, tetapi kalkulasi Risk/Reward belum mencapai minimum 1:1.5."
                dynamic_wait = f"WAIT: {primary_blocker}{ct_note}"
            else:
                primary_blocker = f"Menunggu penyelesaian validasi final pada {p_name}."
                dynamic_wait = f"WAIT: {primary_blocker}"
        else:
            app_str = "sudah mendekati" if p_app == "APPROACHING" else ("berjarak jauh dari" if p_app == "FAR" else "belum mencapai")
            primary_blocker = f"Bias H4/H1 {htf_bias.lower()}, tetapi harga {app_str} {p_type.capitalize()} RADAR ({p_name} @ {p_range}, jarak {p_dist:.2f} pt). Belum ada validasi sweep dan struktur M5."
            dynamic_wait = f"WAIT: {primary_blocker}{ct_note}"
    else:
        if countertrend:
            ct_type = countertrend["zone_type"]
            ct_range = f"{countertrend['price_low']:.2f}-{countertrend['price_high']:.2f}"
            opp_sig = "BUY" if ct_type == "DEMAND" else "SELL"
            primary_blocker = f"Bias H4/H1 {htf_bias.lower()}, tidak ada zona yang selaras dengan bias utama. Zona {ct_type.capitalize()} di {ct_range} hanya area reaksi/support, bukan sinyal {opp_sig}."
            dynamic_wait = f"WAIT: {primary_blocker}"
        else:
            primary_blocker = f"Bias H4/H1 {htf_bias.lower()}, tidak ditemukan zona aktif yang memenuhi syarat."
            dynamic_wait = f"WAIT: {primary_blocker}"

    # PENDING LIMIT BLOCKERS & CONFIRMED ENTRY BLOCKERS
    if (htf_bias == "BULLISH" and h1_structure == "BEARISH") or (htf_bias == "BEARISH" and h1_structure == "BULLISH"):
        pending_blockers.append(f"Konflik struktur H4 ({htf_bias}) vs H1 ({h1_structure})")
    if m30_status in ("CONFLICT", "INVALID", "WEAK"):
        pending_blockers.append(f"Momentum M30 tidak mendukung pending ({m30_status})")

    if eval_target:
        if eval_target.get("grade") not in ("A+", "A"):
            pending_blockers.append(f"Kualitas zona ({eval_target.get('grade')} / Skor {eval_target.get('score', 0.0):.1f}) di bawah standar Grade A (skor >= 75.0)")
        if eval_target.get("freshness") not in ("FRESH", "TESTED_1X"):
            pending_blockers.append(f"Status zona ({eval_target.get('freshness')}) sudah termitigasi")
        p_rsn = eval_target.get("pending_rejection_reason")
        if p_rsn and p_rsn not in pending_blockers and "memenuhi" not in str(p_rsn).lower():
            pending_blockers.append(p_rsn)

        # Confirmed M5 blockers: Berurutan, tidak menampilkan sweep/trigger gagal jika zone belum tersentuh
        st_zt = (m5_stages or {}).get("zone_touch", {}).get("status")
        st_sw = (m5_stages or {}).get("liquidity_sweep", {}).get("status")
        st_tr = (m5_stages or {}).get("structure_trigger", {}).get("status")
        st_dp = (m5_stages or {}).get("displacement", {}).get("status")
        st_rr = (m5_stages or {}).get("risk_reward", {}).get("status")

        is_touched = (eval_target.get("approach") in ("TOUCHED", "ZONE_TOUCHED") or st_zt in ("CONFIRMED", "PASS", "PASSED"))
        if not is_touched:
            confirmed_blockers.append(f"[ZONE_TOUCH]: Harga belum mencapai atau menyentuh {eval_target['zone_type'].capitalize()} ({eval_target['price_low']:.2f}-{eval_target['price_high']:.2f}, jarak {eval_target['distance']:.2f} pt) — konfirmasi M5 menunggu zone touch")
        else:
            if st_sw not in ("CONFIRMED", "PASS", "PASSED"):
                want_sw = "Buy-Side Liquidity (EQH/Swing High)" if eval_target["zone_type"] == "SUPPLY" else "Sell-Side Liquidity (EQL/Swing Low)"
                sw_m = (m5_stages or {}).get("liquidity_sweep", {}).get("msg", "")
                if "jauh" in str(sw_m).lower():
                    confirmed_blockers.append("[LIQUIDITY_SWEEP]: Sweep likuiditas M5 terjadi terlalu jauh dari area zona")
                else:
                    confirmed_blockers.append(f"[LIQUIDITY_SWEEP]: Belum ada validasi sweep {want_sw} M5 di area zona")
            elif st_tr not in ("CONFIRMED", "PASS", "PASSED"):
                want_tr = "CHoCH/BOS/CISD bearish" if eval_target["zone_type"] == "SUPPLY" else "CHoCH/BOS/CISD bullish"
                confirmed_blockers.append(f"[STRUCTURE_TRIGGER]: Belum ada konfirmasi perubahan struktur {want_tr} M5 setelah sweep")
            elif st_dp not in ("CONFIRMED", "PASS", "PASSED"):
                dp_m = (m5_stages or {}).get("displacement", {}).get("msg", "Displacement lemah")
                confirmed_blockers.append(f"[DISPLACEMENT]: Belum ada konfirmasi dorongan displacement momentum M5 searah ({dp_m})")
            elif st_rr in ("FAILED", "FAIL", "WAIT"):
                confirmed_blockers.append("[RISK_REWARD]: Kalkulasi Risk/Reward belum mencapai minimum 1:1.5")
    else:
        pending_blockers.append(f"Tidak ada zona Grade A yang selaras dengan bias HTF ({htf_bias})")
        confirmed_blockers.append(f"Tidak ada zona aktif yang selaras dengan arah bias HTF ({htf_bias})")

    if eval_target and eval_target.get("approach") == "FAR":
        sec_warnings.append(f"Zona prioritas berjarak jauh ({eval_target['distance']:.2f} pt / {eval_target.get('distance_atr_ratio', 0.0)}x ATR).")
    if any(rz.get("scoring_status") == "DATA_INCOMPLETE" for rz in processed):
        sec_warnings.append("Sebagian zona memiliki data teknikal tidak lengkap (DATA INCOMPLETE).")

    wait_breakdown = {
        "primary_blocker": sanitize_wait_reason(primary_blocker),
        "pending_limit_blockers": [sanitize_wait_reason(pb) for pb in pending_blockers if pb],
        "confirmed_entry_blockers": [sanitize_wait_reason(cb) for cb in confirmed_blockers if cb],
        "secondary_warnings": [sanitize_wait_reason(sw) for sw in sec_warnings if sw],
    }

    # 5. Kompilasi ZONE SCORING SUMMARY
    ranked_by_score = sorted(processed, key=lambda x: -float(x.get("score", 0.0)))
    prioritized_reasons = []
    unprioritized_reasons = []
    has_incomplete = any(rz.get("scoring_status") == "DATA_INCOMPLETE" for rz in processed)

    for rz in processed:
        z_name = f"{rz['timeframe']} {rz['zone_type']} ({rz['price_low']:.2f}-{rz['price_high']:.2f})"
        score_val_str = f"{rz['score']:.1f}" if rz.get("scoring_status") != "DATA_INCOMPLETE" else "DATA INCOMPLETE"
        if rz == primary_directional:
            prioritized_reasons.append(
                f"{z_name}: Diprioritaskan sebagai Primary Directional Zone karena searah bias {htf_bias}, Grade {rz['grade']} (Skor {score_val_str}), {rz['freshness']}, jarak {rz['distance']:.2f} pt ({rz['direction_status']})."
            )
        elif rz.get("is_counter"):
            unprioritized_reasons.append(
                f"{z_name}: Tidak diprioritaskan untuk entry karena berlawanan dengan bias HTF ({htf_bias}) — hanya area reaksi/support."
            )
        elif rz.get("approach") == "FAR":
            unprioritized_reasons.append(
                f"{z_name}: Belum diprioritaskan untuk entry dekat karena jaraknya jauh ({rz['distance']:.2f} pt)."
            )
        elif rz.get("freshness") in ("TESTED_1X", "TESTED_2X"):
            unprioritized_reasons.append(
                f"{z_name}: Kualitas berkurang karena sudah pernah teruji ({rz['freshness']})."
            )
        elif rz != primary_directional:
            if primary_directional is None:
                unprioritized_reasons.append(
                    f"{z_name}: Tidak menjadi Primary Directional Zone karena belum ada zona aktif yang memenuhi syarat searah bias {htf_bias}."
                )
            else:
                p_score = float(primary_directional.get("score") or 0.0)
                p_dist = float(primary_directional.get("distance") or 0.0)
                p_prio = float(primary_directional.get("priority_score") or 0.0)
                r_score = float(rz.get("score") or 0.0)
                r_dist = float(rz.get("distance") or 0.0)
                r_prio = float(rz.get("priority_score") or 0.0)
                p_name = f"{primary_directional['timeframe']} {primary_directional['zone_type']}"

                if r_score > p_score and r_dist > p_dist:
                    reason = (
                        f"{z_name}: Skor teknikal lebih tinggi ({score_val_str} vs {p_score:.1f}), "
                        f"namun jarak lebih jauh ({r_dist:.2f} pt vs {p_dist:.2f} pt dari {p_name}) "
                        f"sehingga priority komposit lebih rendah ({r_prio:.1f} vs {p_prio:.1f})."
                    )
                elif r_score > p_score:
                    reason = (
                        f"{z_name}: Skor teknikal lebih tinggi ({score_val_str} vs {p_score:.1f}), "
                        f"namun kalah pada bobot prioritas komposit ({r_prio:.1f} vs {p_prio:.1f}) dibanding {p_name}."
                    )
                elif r_dist > p_dist:
                    reason = (
                        f"{z_name}: Skor teknikal ({score_val_str} vs {p_score:.1f}) dan jarak lebih jauh "
                        f"({r_dist:.2f} pt vs {p_dist:.2f} pt) dibanding {p_name}."
                    )
                else:
                    reason = (
                        f"{z_name}: Skor teknikal ({score_val_str}) berada di bawah {p_name} ({p_score:.1f})."
                    )
                unprioritized_reasons.append(reason)

    # 6. ATURAN PRIORITAS VALIDASI (Requirement 1, 2, 3):
    # Zona INVALIDATED / EXPIRED / MITIGATED / FAR dilarang masuk Radar aktif & Multi-Tier aktif!
    radar_qualified = [
        z for z in processed
        if is_zone_active_and_valid(z, current_price)
        and z.get("radar_eligibility") == "QUALIFIED"
        and z.get("risk_tier") != "FAR_ZONE"
    ]

    # Zona historis/invalid untuk bagian pelaporan terpisah (Requirement 1 & 2)
    invalidated_historical = [
        z for z in processed
        if not is_zone_active_and_valid(z, current_price)
    ]
    if rejected_zones:
        for rz in rejected_zones:
            rz_dict = rz.to_dict() if hasattr(rz, "to_dict") else dict(rz)
            if not is_zone_active_and_valid(rz_dict, current_price):
                if not any(iz.get("zone_id") == rz_dict.get("zone_id") for iz in invalidated_historical):
                    invalidated_historical.append(rz_dict)

    # 1. HIGH RISK — AGGRESSIVE ZONE (Prioritaskan M5/M15/M30, dekat harga, freshness, score)
    high_risk_candidates = [z for z in radar_qualified if z.get("risk_tier") == "HIGH_RISK"]
    high_risk_candidates.sort(key=lambda x: (
        0 if x.get("approach") in ("TOUCHED", "ZONE_TOUCHED") else (1 if x.get("approach") == "APPROACHING" else 2),
        x.get("distance", 999.0),
        0 if x.get("freshness") == "FRESH" else 1,
        -float(x.get("score", 0.0))
    ))
    high_risk_zones = high_risk_candidates[:MULTI_TIER_MAX_ZONES]

    # 2. MEDIUM RISK — BALANCED ZONE (Keseimbangan kualitas struktur & jarak harga)
    medium_risk_candidates = [z for z in radar_qualified if z.get("risk_tier") == "MEDIUM_RISK"]
    medium_risk_candidates.sort(key=lambda x: (
        0 if x.get("is_aligned") else 1,
        0 if x.get("freshness") == "FRESH" else 1,
        -float(x.get("score", 0.0)),
        x.get("distance", 999.0),
    ))
    medium_risk_zones = medium_risk_candidates[:MULTI_TIER_MAX_ZONES]

    # 3. LOW RISK — CONSERVATIVE ZONE (Prioritas H1/H4, HTF alignment, Grade A+/A, score >= 75.0, jarak wajar)
    low_risk_candidates = [z for z in radar_qualified if z.get("risk_tier") == "LOW_RISK"]
    low_risk_candidates.sort(key=lambda x: (
        0 if x.get("is_aligned") else 1,
        0 if x.get("grade") == "A+" else (1 if x.get("grade") == "A" else 2),
        0 if x.get("freshness") == "FRESH" else 1,
        x.get("distance", 999.0),
        -float(x.get("score", 0.0)),
    ))
    low_risk_zones = low_risk_candidates[:MULTI_TIER_MAX_ZONES]

    best_high_risk = high_risk_zones[0] if high_risk_zones else None
    best_medium_risk = medium_risk_zones[0] if medium_risk_zones else None
    best_low_risk = low_risk_zones[0] if low_risk_zones else None

    # Pemisahan Sistem Ranking (5 Sistem Independen — Requirement 2):
    # A. Raw / Historical Candidate Scoring Ranking (Audit Only — Tanpa Otoritas Entry)
    all_raw_cands = list(processed)
    if rejected_zones:
        for rz in rejected_zones:
            rz_dict = rz.to_dict() if hasattr(rz, "to_dict") else dict(rz)
            if not any(x.get("zone_id") == rz_dict.get("zone_id") for x in all_raw_cands):
                all_raw_cands.append(rz_dict)

    raw_scoring_ranking = []
    for idx, rz in enumerate(sorted(all_raw_cands, key=lambda x: -float(x.get("score", 0.0))), 1):
        z_is_v = is_zone_active_and_valid(rz, current_price)
        if not z_is_v:
            raw_st = rz.get("status")
            if raw_st in ("INVALIDATED", "EXPIRED", "MITIGATED"):
                structural_st = raw_st
                v_st = raw_st
            else:
                structural_st = "STRUCTURALLY_REJECTED"
                v_st = "STRUCTURALLY_REJECTED"
            radar_st = "EXCLUDED"
        elif rz.get("risk_tier") == "FAR_ZONE":
            structural_st = "STRUCTURALLY_VALID"
            radar_st = "FAR_ZONE_EXCLUDED"
            v_st = "STRUCTURALLY_VALID | FAR_ZONE_EXCLUDED (Audit Only)"
        elif rz.get("radar_eligibility") == "QUALIFIED":
            structural_st = "STRUCTURALLY_VALID"
            radar_st = "RADAR_QUALIFIED"
            v_st = "STRUCTURALLY_VALID | RADAR_QUALIFIED"
        else:
            structural_st = "STRUCTURALLY_VALID"
            radar_st = "NOT_RADAR_QUALIFIED"
            v_st = "STRUCTURALLY_VALID"

        if rz.get("confirmed_entry_eligibility") == "CONFIRMED":
            entry_st = "CONFIRMED_ENTRY_QUALIFIED"
        elif rz.get("pending_eligibility") == "READY" and rz.get("risk_tier") != "FAR_ZONE":
            entry_st = "PENDING_QUALIFIED"
        else:
            entry_st = "NO_ENTRY_AUTHORITY"

        raw_scoring_ranking.append({
            "zone_id": rz.get("zone_id", ""),
            "rank": idx,
            "zone": f"{rz.get('timeframe', 'H1')} {rz.get('zone_type', '')} {rz.get('price_low', 0.0):.2f}-{rz.get('price_high', 0.0):.2f}",
            "score": float(rz.get("score", 0.0)),
            "scoring_status": rz.get("scoring_status", "VALID"),
            "structural_status": structural_st,
            "radar_status": radar_st,
            "entry_status": entry_st,
            "validity_status": v_st,
            "is_valid": z_is_v,
            "reason": rz.get("risk_tier_reason") if rz.get("risk_tier") == "FAR_ZONE" else (rz.get("rejection_reason") or rz.get("score_reason") or f"Grade {rz.get('grade')} | {rz.get('freshness')}"),
            "entry_authority": False,
            "zone_dict": rz,
        })

    # B. Active Valid Zone Ranking (Hanya zona structurally valid, not invalidated/mitigated/expired)
    tech_rank = sorted(radar_qualified, key=lambda x: (-float(x.get("score", 0.0)), 0 if x.get("grade") == "A+" else 1))
    active_valid_ranking = []
    for idx, az in enumerate(tech_rank, 1):
        item_dict = {
            "zone_id": az.get("zone_id", ""),
            "rank": idx,
            "zone": f"{az.get('timeframe', 'H1')} {az.get('zone_type', '')} {az.get('price_low', 0.0):.2f}-{az.get('price_high', 0.0):.2f}",
            "score": float(az.get("score", 0.0)),
            "grade": az.get("grade"),
            "freshness": az.get("freshness"),
            "distance": az.get("distance"),
            "risk_tier": az.get("risk_tier"),
            "scoring_status": az.get("scoring_status", "VALID"),
            "reason": az.get("risk_tier_reason") or f"Grade {az.get('grade')} | {az.get('freshness')}",
            "zone_dict": az,
        }
        for k_az, v_az in az.items():
            if k_az not in item_dict:
                item_dict[k_az] = v_az
        active_valid_ranking.append(item_dict)

    # C. Directional Priority Ranking (Kesesuaian arah HTF + Kualitas SMC)
    dir_rank = sorted(radar_qualified, key=lambda x: (0 if x.get("is_aligned") else (2 if x.get("is_counter") else 1), -float(x.get("score", 0.0)), x.get("distance", 999.0)))

    # D. Nearest Zone Ranking (Berdasarkan jarak snapshot harga)
    near_rank = sorted(radar_qualified, key=lambda x: (x.get("distance", 999.0), -float(x.get("score", 0.0))))

    # E. Entry Candidate Ranking (Hanya kandidat lolos Pending READY atau Confirmed CONFIRMED)
    pending_ready = [z for z in radar_qualified if z.get("pending_eligibility") == "READY"]
    if pending_ready:
        pending_ready.sort(key=lambda x: (
            0 if x.get("grade") == "A+" else 1,
            -float(x.get("score", 0.0)),
            x.get("distance", 999.0)
        ))
        best_pending_candidate = pending_ready[0]
    else:
        best_pending_candidate = None

    confirmed_ready = [z for z in radar_qualified if z.get("confirmed_entry_eligibility") == "CONFIRMED"]
    if confirmed_ready:
        confirmed_ready.sort(key=lambda x: (
            -float(x.get("score", 0.0)),
            x.get("distance", 999.0)
        ))
        best_confirmed_candidate = confirmed_ready[0]
    elif best_validated_entry:
        best_confirmed_candidate = best_validated_entry
    else:
        best_confirmed_candidate = None

    entry_candidate_ranking = []
    if best_confirmed_candidate:
        entry_candidate_ranking.append({"type": "CONFIRMED_ENTRY", "status": "CONFIRMED", "zone": best_confirmed_candidate})
    if best_pending_candidate:
        entry_candidate_ranking.append({"type": "PENDING_LIMIT", "status": "READY", "zone": best_pending_candidate})

    ranking_systems = {
        "raw_scoring_ranking": raw_scoring_ranking,
        "active_valid_ranking": active_valid_ranking,
        "technical_quality_ranking": tech_rank,
        "directional_priority_ranking": dir_rank,
        "nearest_zone_ranking": near_rank,
        "entry_candidate_ranking": entry_candidate_ranking,
        "pending_eligibility_ranking": pending_ready,
        "confirmed_entry_ranking": confirmed_ready,
    }

    # Rekonsiliasi Penghitungan Zona (Requirement 1):
    # Total Radar Qualified vs Multi-Tier vs Display Limit vs Advanced Rejection
    multi_tier_displayed_count = len(high_risk_zones) + len(medium_risk_zones) + len(low_risk_zones)
    unclassified_tier_zones = [z for z in radar_qualified if z.get("risk_tier") in ("UNCLASSIFIED", "DATA_INCOMPLETE")]
    backup_tier_zones = (
        high_risk_candidates[MULTI_TIER_MAX_ZONES:] +
        medium_risk_candidates[MULTI_TIER_MAX_ZONES:] +
        low_risk_candidates[MULTI_TIER_MAX_ZONES:]
    )
    far_zones = []
    _seen_far_ids = set()
    for z in processed:
        if z.get("risk_tier") == "FAR_ZONE":
            fz_id = z.get("zone_id") or f"{z.get('timeframe')}_{z.get('zone_type')}_{z.get('price_low')}_{z.get('price_high')}"
            if fz_id not in _seen_far_ids:
                _seen_far_ids.add(fz_id)
                far_zones.append(z)
    active_ranking_display_cap = 5
    hidden_active_ranking_zones = radar_qualified[active_ranking_display_cap:] if len(radar_qualified) > active_ranking_display_cap else []

    zone_count_reconciliation = {
        "radar_qualified_count": len(radar_qualified),
        "multi_tier_displayed_count": multi_tier_displayed_count,
        "high_risk_count": len(high_risk_zones),
        "medium_risk_count": len(medium_risk_zones),
        "low_risk_count": len(low_risk_zones),
        "unclassified_tier_count": len(unclassified_tier_zones),
        "unclassified_tier_reasons": [
            (
                f"{z.get('timeframe')} {z.get('zone_type')} ({z.get('price_low', 0):.2f}-{z.get('price_high', 0):.2f}): "
                f"{z.get('risk_tier_reason') or 'Tidak memenuhi ambang batas tier'} | "
                f"Status: Tetap aktif di Radar"
                + (", Dipantau sebagai Primary Directional Zone" if (primary_directional and z.get("zone_id") == primary_directional.get("zone_id")) else "")
            )
            for z in unclassified_tier_zones
        ],
        "backup_tier_count": len(backup_tier_zones),
        "backup_tier_reasons": [
            f"{z.get('timeframe')} {z.get('zone_type')} ({z.get('price_low', 0):.2f}-{z.get('price_high', 0):.2f}): Cadangan di luar batas kuota tampilan top {MULTI_TIER_MAX_ZONES} pada tier {z.get('risk_tier')}"
            for z in backup_tier_zones
        ],
        "far_zones_count": len(far_zones),
        "far_zones_reasons": [
            f"{z.get('timeframe')} {z.get('zone_type')} ({z.get('price_low', 0):.2f}-{z.get('price_high', 0):.2f}): {z.get('risk_tier_reason') or 'Jarak zona melebihi batas intraday maksimal (> 25.0 ATR atau > 75.0 pt)'}"
            for z in far_zones
        ],
        "hidden_active_ranking_count": len(hidden_active_ranking_zones),
        "hidden_active_ranking_reasons": [
            f"{z.get('timeframe')} {z.get('zone_type')} ({z.get('price_low', 0):.2f}-{z.get('price_high', 0):.2f}): Berada di luar batas tampilan {active_ranking_display_cap} teratas (Rank #{idx})"
            for idx, z in enumerate(hidden_active_ranking_zones, active_ranking_display_cap + 1)
        ],
    }

    scoring_summary = {
        "ranking": active_valid_ranking,
        "active_ranking": active_valid_ranking,
        "raw_ranking": raw_scoring_ranking,
        "ranked_zones": active_valid_ranking,
        "prioritized_reasons": prioritized_reasons,
        "deprioritized_reasons": unprioritized_reasons,
        "unprioritized_reasons": unprioritized_reasons,
        "has_incomplete_data": has_incomplete,
    }

    return {
        "primary_directional_zone": primary_directional,
        "eval_target": eval_target,
        "nearest_radar_zone": nearest_radar,
        "best_validated_entry_zone": best_validated_entry,
        "alternative_zone": alternative_zone,
        "best_entry_monitoring_zone": best_entry_monitoring,
        "primary_zone": primary_directional if primary_directional is not None else (None if htf_bias in ("BULLISH", "BEARISH") else primary),
        "secondary_zone": secondary,
        "countertrend_zone": countertrend,
        "all_prioritized": processed,
        "aligned_zones": aligned,
        "counter_zones": counter,
        "multi_tier_zones": {
            "high_risk": high_risk_zones,
            "medium_risk": medium_risk_zones,
            "low_risk": low_risk_zones,
        },
        "high_risk_zones": high_risk_zones,
        "medium_risk_zones": medium_risk_zones,
        "low_risk_zones": low_risk_zones,
        "best_high_risk": best_high_risk,
        "best_medium_risk": best_medium_risk,
        "best_low_risk": best_low_risk,
        "best_pending_candidate": best_pending_candidate,
        "best_confirmed_candidate": best_confirmed_candidate,
        "ranking_systems": ranking_systems,
        "primary_reason": primary_reason,
        "distance_to_zone_str": dist_str,
        "missing_conditions": missing,
        "dynamic_wait_reason": dynamic_wait,
        "wait_breakdown": wait_breakdown,
        "invalidated_zones": invalidated_historical,
        "historical_zones": invalidated_historical,
        "radar_qualified": radar_qualified,
        "confirmed_entry_rr": confirmed_entry_rr_status,
        "zone_scoring_summary": scoring_summary,
        "zone_count_reconciliation": zone_count_reconciliation,
        "snapshot_id": snapshot_id,
        "snapshot_time": snapshot_time,
        "snapshot_price": round(float(current_price), 2),
        "priority_calc_timestamp": snapshot_time or datetime.now().strftime("%Y-%m-%d %H:%M:%S WIB"),
    }


# =============================================================================
# SCANNER UTAMA
# =============================================================================
def scan_xauusd_zones(frames: Optional[Dict[str, pd.DataFrame]] = None, current_price: Optional[float] = None,
                      symbol: str = SYMBOL, snapshot: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Pindai zona H4 & H1 (OB + konfirmasi FVG + sweep), pisahkan deteksi vs validasi,
    skor, gabung, dan kembalikan hasil terstruktur ke dalam 3 kategori:
      1. valid_zones (active_zones): Lolos hard validation, skor >= 60, belum invalid, jarak wajar.
      2. radar_zones: Seluruh zona reaksi terdeteksi yang utuh struktural (termasuk zona jauh & saat konflik HTF).
      3. rejected_zones: Seluruh zona yang gagal validasi beserta alasan spesifik.
    Raise-safe untuk data buruk (mengembalikan data_ok=False dengan empty_reason).
    """
    snapshot_id = (
        snapshot.get("snapshot_id")
        if isinstance(snapshot, dict) and snapshot.get("snapshot_id")
        else _generate_snapshot_id()
    )
    out: Dict[str, Any] = {
        "symbol": symbol, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "current_price": current_price,
        "snapshot_id": snapshot_id,
        "valid_zones": [], "active_zones": [], "radar_zones": [], "detected_zones": [], "rejected_zones": [],
        "best_demand": None, "best_supply": None, "htf_context": {}, "empty_reason": "", "summary_text": "",
        "diagnostics": {}, "diagnostic_text": "", "data_ok": False,
    }
    if isinstance(snapshot, dict) and snapshot.get("snapshot_status") == "DATA_INCOMPLETE":
        out["snapshot_status"] = "DATA_INCOMPLETE"
        out["empty_reason"] = snapshot.get("error_reason") or "DATA_INCOMPLETE: Snapshot MT5 Bid/Ask tidak lengkap atau tidak valid"
        out["summary_text"] = "Snapshot MT5 tidak lengkap (DATA_INCOMPLETE)."
        return out
    if frames is None and isinstance(snapshot, dict) and snapshot.get("frames") is not None:
        frames = snapshot.get("frames")
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

    # Sinkronisasi Snapshot MT5 Tunggal & Immutable (Live Tick vs Candle Close)
    bid = None
    ask = None
    mid = None
    spread = None
    tick_time = None
    tick_age = None
    srv_ts_scan = None
    price_source = None
    snapshot_status = "OK"

    if isinstance(snapshot, dict) and snapshot.get("snapshot_status") == "OK":
        bid = snapshot.get("bid")
        ask = snapshot.get("ask")
        mid = snapshot.get("mid")
        spread = snapshot.get("spread")
        tick_time = snapshot.get("tick_time") or snapshot.get("snapshot_time")
        tick_age = snapshot.get("tick_age")
        srv_ts_scan = snapshot.get("server_time")
        price_source = snapshot.get("price_source")
        if current_price is None:
            _sp = snapshot.get("snapshot_price")
            current_price = _sp if _sp is not None else mid
    elif isinstance(snapshot, dict) and snapshot.get("snapshot_status") == "DATA_INCOMPLETE":
        snapshot_status = "DATA_INCOMPLETE"
    else:
        try:
            import MetaTrader5 as mt5
            tick = mt5.symbol_info_tick(symbol)
            if tick:
                b_val = float(getattr(tick, "bid", 0.0) or 0.0)
                a_val = float(getattr(tick, "ask", 0.0) or 0.0)
                if math.isfinite(b_val) and math.isfinite(a_val) and b_val > 0 and a_val >= b_val:
                    bid = round(b_val, 2)
                    ask = round(a_val, 2)
                    mid = round((bid + ask) / 2.0, 2)
                    spread = round(ask - bid, 2)
                    if hasattr(tick, "time") and tick.time > 0:
                        tick_time = datetime.fromtimestamp(tick.time).strftime("%Y-%m-%d %H:%M:%S WIB")
                        tick_age = round(max(0.0, time.time() - tick.time), 1)
                        srv_ts_scan = pd.Timestamp(int(tick.time), unit="s")
                    else:
                        tick_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S WIB")
                        tick_age = 0.0
                    price_source = f"MT5 Live Tick (Bid: {bid:.2f} | Ask: {ask:.2f} | Mid: {mid:.2f} | Spread: {spread:.2f} pt)"
                elif b_val > 0 or a_val > 0:
                    snapshot_status = "DATA_INCOMPLETE"
        except Exception:
            pass

    if current_price is None:
        if mid is not None:
            # Standar konvensi: Mid Price digunakan untuk evaluasi jarak zona SMC terhadap fair market value
            current_price = mid
        elif snapshot_status == "DATA_INCOMPLETE":
            out["snapshot_status"] = "DATA_INCOMPLETE"
            out["empty_reason"] = "DATA_INCOMPLETE: Snapshot MT5 Bid/Ask tidak lengkap atau tidak valid"
            out["summary_text"] = "Snapshot MT5 tidak lengkap (DATA_INCOMPLETE)."
            return out
        else:
            ref = frames.get("M5") if frames.get("M5") is not None and len(frames["M5"]) else frames["H1"]
            current_price = float(ref["Close"].iloc[-1])

    if not (isinstance(current_price, (int, float)) and math.isfinite(current_price) and current_price > 0):
        out["snapshot_status"] = "DATA_INCOMPLETE"
        out["empty_reason"] = "DATA_INCOMPLETE: Harga referensi tidak valid"
        out["summary_text"] = "Harga tidak valid (WAIT)."
        return out

    snapshot_price = round(float(current_price), 2)
    snapshot_time = tick_time or datetime.now().strftime("%Y-%m-%d %H:%M:%S WIB")
    out["snapshot_status"] = snapshot_status

    if price_source is None:
        price_source = f"Candle Close Snapshot ({snapshot_price:.2f})"

    out["current_price"] = snapshot_price
    out["snapshot_price"] = snapshot_price
    out["snapshot_time"] = snapshot_time
    out["snapshot_id"] = snapshot_id
    # Simpan objek snapshot yang dipakai (referensi eksplisit) agar evaluator/laporan yang menerima
    # hasil scan ini dapat memakai frames/tick/ATR yang sama tanpa fetch ulang. Bila scan dipanggil
    # tanpa snapshot (pemanggil lama), snapshot sintetis dibangun dari data yang baru saja dipakai scan.
    if isinstance(snapshot, dict):
        out["snapshot"] = snapshot
    else:
        out["snapshot"] = {
            "snapshot_id": snapshot_id, "symbol": symbol, "bid": bid, "ask": ask, "mid": mid,
            "spread": spread, "snapshot_price": snapshot_price, "current_price": snapshot_price,
            "snapshot_time": snapshot_time, "tick_time": tick_time, "tick_age": tick_age,
            "price_source": price_source, "snapshot_status": snapshot_status, "frames": frames,
            "atr_diagnostics": None, "news": None, "news_error": None, "server_time": srv_ts_scan,
            "data_ok": snapshot_status == "OK", "error_reason": "",
            "data_sources": {"frames": "caller_supplied", "tick": "scan_internal", "price": "scan_internal"},
        }
    out["bid"] = bid
    out["ask"] = ask
    out["mid"] = mid
    out["spread"] = spread
    out["tick_time"] = tick_time
    out["tick_age_seconds"] = tick_age
    out["price_source"] = price_source
    out["data_ok"] = True
    out["is_cached"] = False
    out["cache_timestamp"] = None
    out["cache_age_seconds"] = 0.0
    out["data_source_mode"] = "LIVE_SNAPSHOT" if isinstance(snapshot, dict) else "LIVE_SCAN"

    _ck_snap = snapshot.get("snapshot_id", "") if isinstance(snapshot, dict) else ""
    ck = _sig([(tf, len(frames[tf]), str(frames[tf].index[-1]), float(frames[tf]["Close"].iloc[-1]))
               for tf in ("H4", "H1", "M30", "M5") if frames.get(tf) is not None and len(frames[tf])],
              round(float(snapshot_price), 1), _ck_snap)
    if _SCAN_CACHE["key"] == ck and time.time() - _SCAN_CACHE["ts"] < SCAN_CACHE_TTL:
        cached = dict(_SCAN_CACHE["data"])
        cached["is_cached"] = True
        cached["cache_timestamp"] = datetime.fromtimestamp(_SCAN_CACHE["ts"]).strftime("%Y-%m-%d %H:%M:%S WIB")
        cached["cache_age_seconds"] = round(max(0.0, time.time() - _SCAN_CACHE["ts"]), 2)
        cached["data_source_mode"] = "CACHE"
        return cached

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
    total_expired = 0
    total_too_far = 0
    total_scored_pass = 0
    rejection_reasons_counter: Dict[str, int] = {}
    tf_diag: Dict[str, Any] = {}
    inspected_zones_log: List[Dict[str, Any]] = []

    tfs_to_scan = ["H4", "H1"]
    for tf_cand in ("M30", "M15", "M5"):
        if frames.get(tf_cand) is not None and len(frames[tf_cand]) >= need:
            df_c = frames[tf_cand]
            c_valid = True
            for col in ("Open", "High", "Low", "Close"):
                if col not in df_c.columns or not np.isfinite(df_c[col].values).all():
                    c_valid = False
                    break
            if c_valid and not ((df_c["High"] < df_c["Low"]).any() or (df_c["High"] < df_c["Open"]).any()):
                tfs_to_scan.append(tf_cand)

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
            dist_res = calculate_zone_distance(
                z.price_low, z.price_high, snapshot_price, digits=2,
                zone_type=z.zone_type, bid=bid, ask=ask, atr=atr_now
            )
            dist_val = dist_res.distance
            dir_st = dist_res.status
            z.distance = dist_val
            z.direction_status = dir_st
            z.approach = calculate_zone_approach(dist_val, dir_st, atr=atr_now)
            dist_ratio, _ = calculate_distance_atr_ratio(dist_val, atr_now)
            z.distance_atr_ratio = dist_ratio
            z.atr_value = atr_now
            z.proximal_edge = z.price_high if z.zone_type == "DEMAND" else z.price_low
            z.distal_edge = z.price_low if z.zone_type == "DEMAND" else z.price_high
            z.zone_width = round(abs(z.price_high - z.price_low), 2)

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
            dist_res = calculate_zone_distance(
                z.price_low, z.price_high, snapshot_price, digits=2,
                zone_type=z.zone_type, bid=bid, ask=ask, atr=atr_now
            )
            dist_val = dist_res.distance
            dir_st = dist_res.status
            z.distance = dist_val
            z.direction_status = dir_st
            z.approach = calculate_zone_approach(dist_val, dir_st, atr=atr_now)
            dist_ratio, _ = calculate_distance_atr_ratio(dist_val, atr_now)
            z.distance_atr_ratio = dist_ratio
            z.atr_value = atr_now
            z.proximal_edge = z.price_high if z.zone_type == "DEMAND" else z.price_low
            z.distal_edge = z.price_low if z.zone_type == "DEMAND" else z.price_high
            z.zone_width = round(abs(z.price_high - z.price_low), 2)

            rej = z.rejection_reason or "Gagal hard validation"
            for part in rej.split("; "):
                part_clean = part.split("(")[0].strip()
                rejection_reasons_counter[part_clean] = rejection_reasons_counter.get(part_clean, 0) + 1

            if "Zona invalid" in rej:
                total_invalidated += 1
                z.status = "INVALIDATED"
            if "Zona sudah dimitigasi" in rej:
                total_mitigated += 1
            if "expired" in rej.lower() or z.freshness == "EXPIRED":
                total_expired += 1
                z.status = "EXPIRED"

            # Kategori A: Radar Zone untuk zona yang utuh secara struktural
            is_inval = (z.freshness == "INVALIDATED" or "Zona invalid" in rej)
            is_mit = (z.touch_count > MAX_TOUCH_COUNT or "Zona sudah dimitigasi" in rej)
            origin_ok = z.origin_bos.get("origin_valid", True)
            is_struct_valid = is_zone_active_and_valid(z, snapshot_price)
            if (not is_inval) and (not is_mit) and origin_ok and is_struct_valid:
                z.grade = "RADAR"
                z.status = "RADAR"
                z.is_valid = True
                score_zone(z, htf_alignment=_alignment(z.zone_type, ctx), allow_radar=True, current_price=snapshot_price)
                z.audit.append(f"Kandidat zona reaksi (RADAR SMC | Skor: {z.score:.1f})")
                radar_all.append(z)
            else:
                z.is_valid = False
                if is_inval:
                    z.status = "INVALIDATED"
                elif is_mit:
                    z.status = "MITIGATED"
                elif "expired" in rej.lower() or z.freshness == "EXPIRED":
                    z.status = "EXPIRED"
                else:
                    z.status = "REJECTED"

            rejected.append(z)

            inspected_zones_log.append({
                "source_timeframe": z.timeframe,
                "zone_id": z.zone_id,
                "zone_type": z.zone_type,
                "price_low": z.price_low,
                "price_high": z.price_high,
                "grade": z.grade,
                "score": z.score,
                "scoring_status": getattr(z, "scoring_status", "VALID"),
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
        "zones_expired": total_expired,
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

    atr_diag = (
        snapshot.get("atr_diagnostics")
        if isinstance(snapshot, dict) and isinstance(snapshot.get("atr_diagnostics"), dict)
        else extract_atr_diagnostics(frames, preferred_tf="H1")
    )
    ref_atr = atr_diag["atr_value"]
    out["atr_diagnostics"] = atr_diag
    radar_prio = prioritize_radar_zones(
        radar_zones=radar_zones,
        current_price=snapshot_price,
        htf_bias=ctx.get("H4", "NETRAL"),
        h1_structure=ctx.get("H1", "NETRAL"),
        atr=ref_atr,
        digits=2,
        frames=frames,
        rejected_zones=rejected,
        bid=bid,
        ask=ask,
        snapshot_time=snapshot_time,
        snapshot_id=snapshot_id,
    )
    radar_prio["atr_diagnostics"] = atr_diag
    out["radar_priority"] = radar_prio
    out["multi_tier_zones"] = radar_prio.get("multi_tier_zones", {})
    out["high_risk_zones"] = radar_prio.get("high_risk_zones", [])
    out["medium_risk_zones"] = radar_prio.get("medium_risk_zones", [])
    out["low_risk_zones"] = radar_prio.get("low_risk_zones", [])
    out["best_high_risk"] = radar_prio.get("best_high_risk")
    out["best_medium_risk"] = radar_prio.get("best_medium_risk")
    out["best_low_risk"] = radar_prio.get("best_low_risk")
    out["best_pending_candidate"] = radar_prio.get("best_pending_candidate")
    out["best_confirmed_candidate"] = radar_prio.get("best_confirmed_candidate")
    out["invalidated_zones"] = radar_prio.get("invalidated_zones", [])

    # Sinkronisasi atribut prioritas & multi-tier ke objek Zone
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
            z.distance_atr_ratio = p_item.get("distance_atr_ratio")
            z.atr_value = p_item.get("atr_value")
            z.risk_tier = p_item.get("risk_tier", "")
            z.status = p_item.get("status", "RADAR")
            z.radar_eligibility = p_item.get("radar_eligibility", "QUALIFIED")
            z.pending_eligibility = p_item.get("pending_eligibility", "NOT_QUALIFIED")
            z.confirmed_entry_eligibility = p_item.get("confirmed_entry_eligibility", "WAIT")
            z.pending_plan = p_item.get("pending_plan")

    out["valid_zones"] = [z.to_dict() for z in active]
    out["active_zones"] = [z.to_dict() for z in active]
    out["radar_zones"] = [z.to_dict() for z in radar_zones]
    out["detected_zones"] = [z.to_dict() for z in (radar_zones + rejected)]
    out["rejected_zones"] = [z.to_dict() for z in rejected]
    out["best_demand"] = next((z.to_dict() for z in active if z.zone_type == "DEMAND"), None)
    out["best_supply"] = next((z.to_dict() for z in active if z.zone_type == "SUPPLY"), None)

    # ZONE EVALUATION SUMMARY (Pemisahan Metrik Independen & Rekonsiliasi Eksklusif: Requirement 2)
    all_logged = inspected_zones_log
    count_raw = len(all_logged)
    count_sup = sum(1 for z in all_logged if z.get("zone_type") == "SUPPLY")
    count_dem = sum(1 for z in all_logged if z.get("zone_type") == "DEMAND")

    # Rekonsiliasi Status Mutually Exclusive (Setiap kandidat dihitung tepat 1 kali):
    count_inval = 0
    count_exp = 0
    count_mit = 0
    count_frsh_valid = 0
    count_tested_valid = 0
    count_struct_rej = 0

    for z in all_logged:
        z_mit_st = str(z.get("mitigation_status", "")).upper()
        z_st = str(z.get("status", "")).upper()
        z_rej = str(z.get("rejection_reason", "")).lower()
        z_is_v = bool(z.get("is_valid", False))
        z_touch = int(z.get("touch_count", 0))

        if z_st == "INVALIDATED" or z_mit_st == "INVALIDATED" or "invalid" in z_rej:
            count_inval += 1
        elif z_st == "EXPIRED" or z_mit_st == "EXPIRED" or "expired" in z_rej:
            count_exp += 1
        elif z_st == "MITIGATED" or z_mit_st == "MITIGATED" or "dimitigasi" in z_rej or z_touch > MAX_TOUCH_COUNT:
            count_mit += 1
        elif z_is_v:
            if z_mit_st.startswith("TESTED"):
                count_tested_valid += 1
            else:
                count_frsh_valid += 1
        else:
            count_struct_rej += 1

    count_struct_valid = count_frsh_valid + count_tested_valid
    count_radar_qualified = len(radar_prio.get("radar_qualified", []))
    count_pending_qualified = len(radar_prio.get("pending_eligibility_ranking", []))
    count_confirmed_qualified = len(radar_prio.get("confirmed_entry_ranking", []))

    out["zone_evaluation_summary"] = {
        "raw_candidates": count_raw,
        "total_candidates": count_raw,
        "total_raw": count_raw,
        "structurally_valid": count_struct_valid,
        "structurally_rejected": count_struct_rej,
        "total_validated": count_struct_valid,
        "radar_qualified": count_radar_qualified,
        "initial_radar_count": count_radar_qualified,
        "pending_qualified": count_pending_qualified,
        "confirmed_entry_qualified": count_confirmed_qualified,
        "entry_validated_count": count_confirmed_qualified,
        "entry_qualified": count_confirmed_qualified,
        "total_supply": count_sup,
        "total_demand": count_dem,
        "total_fresh": count_frsh_valid,
        "total_tested": count_tested_valid,
        "total_mitigated": count_mit,
        "total_invalidated": count_inval,
        "total_expired": count_exp,
        "structurally_valid_definition": "Fresh Valid + Tested Valid (tidak termasuk Mitigated/Invalidated/Expired/Structurally Rejected)",
        # Backward compatibility
        "count_supply": count_sup,
        "count_demand": count_dem,
        "count_fresh": count_frsh_valid,
        "count_tested": count_tested_valid,
        "count_mitigated": count_mit,
        "count_invalidated": count_inval,
        "count_expired": count_exp,
        "count_structurally_rejected": count_struct_rej,
        "count_initial_pass": count_radar_qualified,
        "count_entry_validated": count_confirmed_qualified,
    }

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
        stages["zone_touch"] = {"status": "FAIL", "msg": f"Harga belum mencapai atau menyentuh zona entry {lo:.2f}-{hi:.2f}"}
        W.append(f"Harga belum mencapai atau menyentuh zona entry {lo:.2f}-{hi:.2f}")
        return res

    first_touch_k = touch_indices[0] if touch_indices else 0
    dist = 0.0 if lo <= price <= hi else (lo - price if price < lo else price - hi)
    approach = max(3.0, 1.5 * atr5)
    if dist > approach:
        stages["zone_touch"] = {"status": "FAIL", "msg": f"Entry terlambat: harga sudah meninggalkan zona entry ({dist:.2f} pt > {approach:.2f} pt)"}
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
            stages["liquidity_sweep"] = {"status": "FAIL", "msg": f"M5 Liquidity Sweep terjadi jauh dari area zona {lo:.2f}-{hi:.2f}"}
            W.append(f"M5 Liquidity Sweep terjadi jauh dari area zona {lo:.2f}-{hi:.2f}")
        else:
            stages["liquidity_sweep"] = {"status": "FAIL", "msg": f"M5 Liquidity Sweep tidak terkonfirmasi ({want_sw} belum terjadi di area zona)"}
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
    current_time: Any = None,
    hours: Optional[Union[int, float]] = None,
    server_time: Any = None,
) -> Dict[str, Any]:
    """
    Hitung suggested expiration secara dinamis berdasarkan satu sumber waktu konsisten dalam WIB (Asia/Jakarta, UTC+7).
    
    Ketentuan:
      - Gunakan timezone WIB yang konsisten untuk snapshot, created at, dan expiration.
      - Expiration dihitung dari created_at + durasi aktual konfigurasi.
      - Menghitung sisa durasi (remaining time) secara dinamis terhadap waktu sekarang.
      - Jika input waktu tidak valid / korup -> tampilkan status EXPIRATION ERROR.
      - Format output terstruktur:
          Pending Plan Expiration:
          - Created At: 08:25 WIB
          - Duration: 8 Hours
          - Expiration: 16:25 WIB
          - Remaining: 07h 59m
    """
    if hours is not None:
        duration_hours = hours
    if server_time is not None and base_time is None:
        base_time = server_time
    # 1. Konversi base_time ke WIB
    if base_time is None:
        base_dt = get_current_wib()
    else:
        base_dt = to_wib_datetime(base_time)

    if base_dt is None:
        return {
            "base_time": "EXPIRATION ERROR",
            "expires_at": "EXPIRATION ERROR",
            "created_at": "EXPIRATION ERROR",
            "expiration": "EXPIRATION ERROR",
            "duration_hours": 0.0,
            "duration_text": "EXPIRATION ERROR",
            "duration_hours_str": "EXPIRATION ERROR",
            "remaining": "EXPIRATION ERROR",
            "suggested_expiration": "EXPIRATION ERROR",
            "is_error": True,
            "formatted_block": "Pending Plan Expiration:\n- Status: EXPIRATION ERROR (Data waktu tidak valid)",
        }

    # 2. Validasi Durasi
    if not isinstance(duration_hours, (int, float)) or not math.isfinite(duration_hours) or duration_hours <= 0:
        return {
            "base_time": base_dt.strftime("%Y-%m-%d %H:%M"),
            "expires_at": "EXPIRATION ERROR",
            "created_at": base_dt.strftime("%H:%M WIB"),
            "expiration": "EXPIRATION ERROR",
            "duration_hours": 0.0,
            "duration_text": "EXPIRATION ERROR",
            "duration_hours_str": "EXPIRATION ERROR",
            "remaining": "EXPIRATION ERROR",
            "suggested_expiration": "EXPIRATION ERROR",
            "is_error": True,
            "formatted_block": "Pending Plan Expiration:\n- Status: EXPIRATION ERROR (Durasi tidak valid)",
        }

    # 3. Hitung target_dt (Expiration) dalam WIB
    if target_time is not None:
        target_dt = to_wib_datetime(target_time)
        if target_dt is None:
            return {
                "base_time": base_dt.strftime("%Y-%m-%d %H:%M"),
                "expires_at": "EXPIRATION ERROR",
                "created_at": base_dt.strftime("%H:%M WIB"),
                "expiration": "EXPIRATION ERROR",
                "duration_hours": 0.0,
                "duration_text": "EXPIRATION ERROR",
                "duration_hours_str": "EXPIRATION ERROR",
                "remaining": "EXPIRATION ERROR",
                "suggested_expiration": "EXPIRATION ERROR",
                "is_error": True,
                "formatted_block": "Pending Plan Expiration:\n- Status: EXPIRATION ERROR (Target waktu tidak valid)",
            }
    else:
        target_dt = base_dt + timedelta(hours=float(duration_hours))

    if target_dt < base_dt:
        target_dt = base_dt

    # 4. Durasi Aktual (Selisih Waktu Nyata)
    delta = target_dt - base_dt
    total_sec = max(0, int(delta.total_seconds()))
    hours = total_sec // 3600
    minutes = (total_sec % 3600) // 60
    dur_hours_val = round(total_sec / 3600.0, 2)

    if hours > 0 and minutes == 0:
        dur_str = f"{hours} jam"
        dur_str_en = f"{hours} Hours"
    elif hours > 0 and minutes > 0:
        dur_str = f"{hours} jam {minutes} menit"
        dur_str_en = f"{hours} Hours {minutes} Mins"
    elif minutes > 0:
        dur_str = f"{minutes} menit"
        dur_str_en = f"{minutes} Minutes"
    else:
        dur_str = "0 jam"
        dur_str_en = "0 Hours"

    # 5. Sisa Durasi (Remaining Time terhadap waktu acuan sekarang dalam WIB)
    curr_dt = to_wib_datetime(current_time) if current_time is not None else get_current_wib()
    if curr_dt is None:
        curr_dt = base_dt

    rem_sec = max(0, int((target_dt - curr_dt).total_seconds()))
    rem_h = rem_sec // 3600
    rem_m = (rem_sec % 3600) // 60
    rem_str = f"{rem_h:02d}h {rem_m:02d}m"

    created_wib_str = f"{base_dt.strftime('%H:%M')} WIB"
    exp_wib_str = f"{target_dt.strftime('%H:%M')} WIB"
    exp_full_str = target_dt.strftime("%Y-%m-%d %H:%M")
    formatted_str = f"{exp_full_str} ({dur_str})"

    formatted_block = (
        "Pending Plan Expiration:\n"
        f"- Created At: {created_wib_str}\n"
        f"- Duration: {dur_str_en}\n"
        f"- Expiration: {exp_wib_str}\n"
        f"- Remaining: {rem_str}"
    )

    return {
        "base_time": base_dt.strftime("%Y-%m-%d %H:%M"),
        "expires_at": exp_full_str,
        "created_at": created_wib_str,
        "created_at_str": created_wib_str,
        "expiration": exp_wib_str,
        "expiration_str": exp_wib_str,
        "duration_hours": dur_hours_val,
        "duration_text": dur_str,
        "duration_str": dur_str_en,
        "duration_hours_str": dur_str_en,
        "remaining": rem_str,
        "remaining_str": rem_str,
        "target_time": target_dt,
        "suggested_expiration": formatted_str,
        "is_error": False,
        "formatted_block": formatted_block,
    }


def plan_pending_order(
    zone: Zone,
    current_price: float,
    frames: Optional[Dict[str, pd.DataFrame]] = None,
    htf_bias: Optional[str] = None,
    m30_status: Optional[str] = None,
    server_time: Any = None,
    expiration_hours: int = DEFAULT_PENDING_EXPIRATION_HOURS,
    created_at: Any = None,
    target_time: Any = None,
    current_time: Any = None,
    bid: Optional[float] = None,
    ask: Optional[float] = None,
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
        "created_at": "",
        "expiration": "",
        "remaining": "",
        "expiration_block": "",
        "is_expiration_error": False,
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
    # BUY LIMIT: Batas atas zona Demand (price_high), wajib < Ask (atau current_price)
    # SELL LIMIT: Batas bawah zona Supply (price_low), wajib > Bid (atau current_price)
    if demand:
        entry = round(res["zone_high"], 2)
        entry_reason = "Batas atas (proximal edge) zona Demand H1/H4 untuk mitigasi awal"
        ref_mkt = float(ask) if (ask is not None and math.isfinite(ask) and ask > 0) else float(current_price)
        if entry >= ref_mkt:
            res["status"] = "WAIT"
            res["early_pending_status"] = "WAIT"
            res["rejection_reason"] = (
                f"Harga {'Ask' if (ask is not None and ask > 0) else 'pasar'} ({ref_mkt:.2f}) "
                f"sudah di bawah/di dalam entry BUY LIMIT ({entry:.2f}); gunakan Confirmed Entry"
            )
            return res
    else:
        entry = round(res["zone_low"], 2)
        entry_reason = "Batas bawah (proximal edge) zona Supply H1/H4 untuk mitigasi awal"
        ref_mkt = float(bid) if (bid is not None and math.isfinite(bid) and bid > 0) else float(current_price)
        if entry <= ref_mkt:
            res["status"] = "WAIT"
            res["early_pending_status"] = "WAIT"
            res["rejection_reason"] = (
                f"Harga {'Bid' if (bid is not None and bid > 0) else 'pasar'} ({ref_mkt:.2f}) "
                f"sudah di atas/di dalam entry SELL LIMIT ({entry:.2f}); gunakan Confirmed Entry"
            )
            return res

    # 6. Stop Loss & Invalidation Buffer
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

    # 8. Suggested Expiration Dinamis dalam WIB
    base_ref = created_at if created_at is not None else server_time
    exp_info = calculate_suggested_expiration(
        base_time=base_ref,
        duration_hours=expiration_hours,
        target_time=target_time,
        current_time=current_time,
    )

    # 9. Penjelasan Objektif Level
    level_explanations = {
        "zone_boundary": f"Area reaksi institusional ({res['zone_low']:.2f} - {res['zone_high']:.2f}) tempat order berpotensi terisi",
        "structural_invalidation": f"Level ({inval:.2f}) di mana struktur zona resmi batal jika tertembus",
        "stop_loss": f"Level proteksi modal ({sl:.2f}) yang diletakkan di luar invalidation dengan buffer ({inval_buffer:.2f} pt) untuk menghindari market noise/spread",
    }

    res.update({
        "valid": not exp_info.get("is_error", False),
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
        "created_at": exp_info.get("created_at", ""),
        "expiration": exp_info.get("expiration", ""),
        "remaining": exp_info.get("remaining", ""),
        "duration_hours": exp_info["duration_hours"],
        "duration_text": exp_info["duration_text"],
        "duration_hours_str": exp_info.get("duration_hours_str", ""),
        "expiration_block": exp_info.get("formatted_block", ""),
        "is_expiration_error": exp_info.get("is_error", False),
        "entry_reason": entry_reason,
        "status": "READY" if not exp_info.get("is_error", False) else "INVALID",
        "early_pending_status": "READY" if not exp_info.get("is_error", False) else "INVALID",
        "early_pending_status_label": "READY — PENDING PLAN (BELUM DIPASANG DI MT5)" if not exp_info.get("is_error", False) else "INVALID (EXPIRATION ERROR)",
        "early_pending_rr": f"PASSED (1:{rr_tp1:.1f})",
        "rejection_reason": "EXPIRATION ERROR" if exp_info.get("is_error", False) else "",
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
    tracker = get_zone_tracker(auto_save=False)
    tracked_zones = tracker.tracked_zones if tracker else {}

    for zd in cands:
        z_obj = _zone_from_dict(zd) if isinstance(zd, dict) else zd
        z_id = getattr(z_obj, "zone_id", "")
        existing_rec = tracked_zones.get(z_id)
        prev_created_at = None
        prev_expires_at = None
        if existing_rec and existing_rec.get("pending_plan") and existing_rec.get("status") in ("READY", "APPROACHING", "TOUCHED"):
            p_plan = existing_rec["pending_plan"]
            prev_created_at = p_plan.get("created_at") or existing_rec.get("first_detected")
            prev_expires_at = p_plan.get("expires_at") or existing_rec.get("expires_at")

        plan = plan_pending_order(
            z_obj, float(current_price), frames, h4_bias, m30_status, server_time,
            created_at=prev_created_at, target_time=prev_expires_at
        )
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

    def get_zone(self, zone_id: str) -> Optional[Dict[str, Any]]:
        return self.tracked_zones.get(zone_id)

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
                    "created_at": now_str,
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
            prev_plan = rec.get("pending_plan") or {}
            prev_c_at = prev_plan.get("created_at") or rec.get("first_detected")
            prev_e_at = prev_plan.get("expires_at") or rec.get("expires_at")
            plan = plan_pending_order(
                z_mock, current_price, frames, htf_bias, m30_status, server_time,
                created_at=prev_c_at, target_time=prev_e_at
            )
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
                    rec["status"] = ZoneStatus("ZONE_TOUCHED")
                    rec["history"].append({"status": "ZONE_TOUCHED", "time": now_str, "reason": f"Harga ({current_price:.2f}) menyentuh zona {z_lo:.2f}-{z_hi:.2f}"})
                    msg = f"[ALERT] Harga telah menyentuh zona {z_id} ({current_price:.2f})"
                    new_alerts.append({"type": "ZONE_TOUCHED", "zone_id": z_id, "msg": msg, "time": now_str})
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

    def clear(self):
        self.reset()

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
                            server_time: Any = None, live: Optional[bool] = None,
                            snapshot: Optional[Dict[str, Any]] = None,
                            scan_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Keputusan akhir VALID / WAIT / INVALID (fail-safe: exception, data kosong/terlambat, harga tak valid => WAIT).
    live=None: bila frames None, data/harga/spread/berita/waktu server diambil dari MT5.
    Jika snapshot dan/atau scan_result disediakan, evaluator menggunakan data snapshot/scan tersebut
    tanpa melakukan pemindaian ulang atau pengambilan tick baru.
    """
    snap_id = (
        snapshot.get("snapshot_id") if isinstance(snapshot, dict) and snapshot.get("snapshot_id")
        else (scan_result.get("snapshot_id") if isinstance(scan_result, dict) else None)
    )
    dec: Dict[str, Any] = {"status": "WAIT", "direction": None, "reasons": [], "zone": None, "plan": None,
                           "stages": {}, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "symbol": SYMBOL,
                           "snapshot_id": snap_id}
    try:
        dec = _decide(dec, frames, current_price, spread, news_blackout, server_time, live,
                      snapshot=snapshot, scan_result=scan_result)
    except Exception as e:
        _catat_error("evaluate_trade_decision", e)
        dec.update(status="WAIT", direction=None, plan=None, zone=None)
        dec["reasons"].append(f"Error internal scanner ({type(e).__name__}): fail-safe WAIT")
        dec["stages"]["ERROR"] = {"ok": False, "msg": repr(e)}
        _audit("entry_decision", _entry_audit(dec), _sig(dec["status"], dec["reasons"]))

    if "early_setup" not in dec and "ERROR" not in dec.get("stages", {}) and snapshot is None and scan_result is None:
        try:
            eff_p = current_price or dec.get("current_price") or dec.get("snapshot_price") or 0.0
            eff_frames = frames if frames is not None else (snapshot.get("frames") if isinstance(snapshot, dict) else None)
            dec["early_setup"] = detect_early_setups(
                frames=eff_frames,
                current_price=eff_p,
                server_time=server_time,
                scan=scan_result if isinstance(scan_result, dict) else None,
            )
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


def _decide(dec, frames, price, spread, news, server_time, live,
            snapshot: Optional[Dict[str, Any]] = None,
            scan_result: Optional[Dict[str, Any]] = None):
    st = dec["stages"]

    def _attach_early_and_tracker(dec_obj, scan_obj=None):
        try:
            eff_p = price if price is not None else (scan_obj.get("current_price") if scan_obj else 0.0)
            if scan_obj is None and (snapshot is not None or scan_result is not None):
                # Mode single-snapshot tanpa scan yang valid: JANGAN memuat frames/tick baru (fail-safe).
                dec_obj["early_setup"] = {"best_setup": None, "all_setups": [], "htf_bias": "NETRAL",
                                          "h1_structure": "NETRAL", "m30_momentum": "NOT_CHECKED",
                                          "empty_reason": "Data snapshot/scan tidak tersedia (DATA_INCOMPLETE)"}
                return
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
            atr_diag_dec = scan.get("atr_diagnostics") or extract_atr_diagnostics(frames if "frames" in locals() else None, preferred_tf="H1")
            ref_atr_val = atr_diag_dec.get("atr_value")
            dec["atr_diagnostics"] = atr_diag_dec
            winning_cand = winning_zdict if ("winning_zdict" in locals() and status == "VALID") else None
            cand_override = winning_zdict if ("winning_zdict" in locals() and winning_zdict) else (best_candidate_dict if "best_candidate_dict" in locals() else None)
            prio = prioritize_radar_zones(
                radar_zones=all_r,
                current_price=float(price or 0.0),
                htf_bias=h4_bias_val,
                h1_structure=h1_struct_val,
                atr=ref_atr_val,
                m5_stages=st,
                validated_entry_zone=winning_cand,
                frames=frames,
                m30_status=st.get("M30", {}).get("status", "CONFIRMED"),
                server_time=server_time,
                primary_target_override=cand_override,
                bid=scan.get("bid"),
                ask=scan.get("ask"),
                snapshot_time=scan.get("snapshot_time"),
                snapshot_id=scan.get("snapshot_id") or dec.get("snapshot_id"),
            )
            dec["radar_priority"] = prio
            dec["multi_tier_zones"] = prio.get("multi_tier_zones", {})
            dec["best_high_risk"] = prio.get("best_high_risk")
            dec["best_medium_risk"] = prio.get("best_medium_risk")
            dec["best_low_risk"] = prio.get("best_low_risk")
            dec["best_pending_candidate"] = prio.get("best_pending_candidate")
            dec["best_confirmed_candidate"] = prio.get("best_confirmed_candidate")
            dec["wait_breakdown"] = prio.get("wait_breakdown", {})
            dec["invalidated_zones"] = prio.get("invalidated_zones", [])
            dec["zone_count_reconciliation"] = prio.get("zone_count_reconciliation", {})
            if status == "WAIT" and prio.get("dynamic_wait_reason"):
                dyn_w = prio["dynamic_wait_reason"]
                if dyn_w not in dec["reasons"]:
                    dec["reasons"].insert(0, dyn_w)

            # ENTRY VALIDATION SUMMARY (Sinkronkan zona yang dipantau dengan satu sumber data kandidat)
            monitored_ref = (winning_zdict if ("winning_zdict" in locals() and winning_zdict) else
                             (best_candidate_dict if "best_candidate_dict" in locals() else 
                              (prio.get("primary_directional_zone") or (all_r[0] if all_r else None))))
            if monitored_ref:
                z_lbl = f"{monitored_ref.get('timeframe', 'H1')} {monitored_ref.get('zone_type', '')} ({monitored_ref.get('price_low', 0.0):.2f} - {monitored_ref.get('price_high', 0.0):.2f})"
                dec["zone"] = monitored_ref
                dec["monitored_zone_identity"] = {
                    "zone_id": monitored_ref.get("zone_id", ""),
                    "timeframe": monitored_ref.get("timeframe", ""),
                    "zone_type": monitored_ref.get("zone_type", ""),
                    "direction": "SELL" if monitored_ref.get("zone_type") == "SUPPLY" else "BUY",
                    "proximal": float(monitored_ref.get("price_low" if monitored_ref.get("zone_type") == "SUPPLY" else "price_high", 0.0)),
                    "distal": float(monitored_ref.get("price_high" if monitored_ref.get("zone_type") == "SUPPLY" else "price_low", 0.0)),
                    "price_low": float(monitored_ref.get("price_low", 0.0)),
                    "price_high": float(monitored_ref.get("price_high", 0.0)),
                    "status": monitored_ref.get("status", "RADAR"),
                }
            else:
                z_lbl = "NONE"
                dec["monitored_zone_identity"] = None

            if monitored_ref and monitored_ref.get("approach") not in ("TOUCHED", "ZONE_TOUCHED"):
                if st.get("zone_touch", {}).get("status") in (None, "NOT_CHECKED", "WAIT"):
                    st["zone_touch"] = {
                        "status": "FAIL",
                        "msg": (
                            f"Harga belum menyentuh zona {monitored_ref.get('zone_type', '')} "
                            f"({monitored_ref.get('price_low', 0.0):.2f}-{monitored_ref.get('price_high', 0.0):.2f}, "
                            f"jarak {float(monitored_ref.get('distance') or 0.0):.2f} pt)"
                        ),
                    }

            conds_pass = []
            conds_fail = []
            conds_not_checked = []
            conds_incomplete = []
            for c_name, c_label in [("zone_touch", "Zone Touch"), ("liquidity_sweep", "Liquidity Sweep M5"),
                                    ("structure_trigger", "CHoCH/BOS/CISD M5"), ("displacement", "Displacement M5"),
                                    ("risk_reward", "Risk/Reward >= 1:1.5")]:
                st_item = st.get(c_name, {})
                c_st = st_item.get("status") if isinstance(st_item, dict) else "NOT_CHECKED"
                c_msg = st_item.get("msg", "") if isinstance(st_item, dict) else ""
                if c_st in ("CONFIRMED", "PASS", "PASSED"):
                    conds_pass.append(f"{c_label} [PASS]")
                elif c_st == "DATA_INCOMPLETE" or ("data" in str(c_msg).lower() and "tidak" in str(c_msg).lower()):
                    conds_incomplete.append(f"{c_label} [DATA_INCOMPLETE: {c_msg or 'Data tidak cukup'}]")
                elif c_st == "NOT_CHECKED":
                    conds_not_checked.append(f"{c_label} [NOT_CHECKED]")
                else:
                    fail_desc = f": {c_msg}" if c_msg else ""
                    conds_fail.append(f"{c_label} [FAIL{fail_desc}]")

            sanitized_reasons = [sanitize_wait_reason(r) for r in dec.get("reasons", []) if r]
            final_reason_text = sanitized_reasons[0] if sanitized_reasons else ("Setup memenuhi validasi" if status == "VALID" else "Menunggu validasi entry")
            final_reason_text = sanitize_wait_reason(final_reason_text)

            dec["entry_validation_summary"] = {
                "monitored_zone": z_lbl,
                "conditions_met": conds_pass,
                "fulfilled_conditions": conds_pass,
                "conditions_pass": conds_pass,
                "conditions_fail": conds_fail,
                "conditions_unmet": conds_fail,
                "unfulfilled_conditions": conds_fail,
                "conditions_not_checked": conds_not_checked,
                "not_checked_conditions": conds_not_checked,
                "conditions_incomplete": conds_incomplete,
                "incomplete_conditions": conds_incomplete,
                "final_decision_reason": f"{status}: {final_reason_text}" if not final_reason_text.startswith(status) else final_reason_text,
                "final_verdict": f"{dec.get('direction', '')} VALID" if status == "VALID" else status,
            }

        _attach_early_and_tracker(dec, scan if ("scan" in locals() and isinstance(scan, dict)) else None)
        _audit("entry_decision", _entry_audit(dec), _sig(status, dec["reasons"], dec.get("direction")))
        return dec

    live_snap = snapshot if isinstance(snapshot, dict) else None
    if (isinstance(snapshot, dict) and isinstance(scan_result, dict)
            and snapshot.get("snapshot_id") and scan_result.get("snapshot_id")
            and snapshot["snapshot_id"] != scan_result["snapshot_id"]):
        dec["snapshot_status"] = "DATA_INCOMPLETE"
        msg_mm = (f"SNAPSHOT_MISMATCH: snapshot {snapshot['snapshot_id']} != scan_result "
                  f"{scan_result['snapshot_id']}; evaluasi dibatalkan (fail-safe)")
        st["DATA"] = {"ok": False, "msg": msg_mm}
        return finish("WAIT", msg_mm)
    if live_snap is None and isinstance(scan_result, dict) and isinstance(scan_result.get("snapshot"), dict):
        live_snap = scan_result["snapshot"]

    if live_snap is not None:
        dec["snapshot_id"] = live_snap.get("snapshot_id") or dec.get("snapshot_id")
        dec["snapshot_time"] = live_snap.get("snapshot_time")
        dec["snapshot_price"] = live_snap.get("snapshot_price")
        if live_snap.get("snapshot_status") == "DATA_INCOMPLETE":
            dec["snapshot_status"] = "DATA_INCOMPLETE"
            err_msg = live_snap.get("error_reason") or "DATA_INCOMPLETE: Snapshot MT5 tidak lengkap atau tidak valid"
            st["DATA"] = {"ok": False, "msg": err_msg}
            return finish("WAIT", err_msg)
        if frames is None:
            frames = live_snap.get("frames")
        if price is None:
            price = live_snap.get("snapshot_price") if live_snap.get("snapshot_price") is not None else live_snap.get("mid")
        if spread is None:
            spread = live_snap.get("spread")
        if server_time is None:
            server_time = live_snap.get("server_time")
        if news is None:
            news = live_snap.get("news")
            if live_snap.get("news_error") and BLOKIR_JIKA_BERITA_GAGAL:
                return finish("WAIT", "Pemeriksaan berita gagal (kebijakan blokir aktif)")
        if price is not None:
            dec["current_price"] = price
    elif isinstance(scan_result, dict):
        dec["snapshot_id"] = scan_result.get("snapshot_id") or dec.get("snapshot_id")
        dec["snapshot_time"] = scan_result.get("snapshot_time")
        dec["snapshot_price"] = scan_result.get("snapshot_price") or scan_result.get("current_price")
        if price is None:
            price = scan_result.get("snapshot_price") or scan_result.get("current_price")
        if spread is None:
            spread = scan_result.get("spread")
        if price is not None:
            dec["current_price"] = price

    use_live = (frames is None) if live is None else live
    if live_snap is None and not isinstance(scan_result, dict):
        # Jalur kompatibilitas pemanggil lama (tanpa snapshot & tanpa scan_result): evaluator mengambil
        # data sendiri. Jalur ini TIDAK PERNAH dipakai bila snapshot atau scan_result diberikan.
        if frames is None:
            frames = _load_frames()
            if frames is None:
                st["DATA"] = {"ok": False, "msg": "MT5 tidak aktif"}
                return finish("WAIT", "MT5 tidak aktif / data tidak tersedia")
        if use_live:
            lc = _live_context()
            live_snap = lc.get("snapshot")
            if isinstance(live_snap, dict):
                dec["snapshot_id"] = live_snap.get("snapshot_id") or dec.get("snapshot_id")
                dec["snapshot_time"] = live_snap.get("snapshot_time")
                dec["snapshot_price"] = live_snap.get("snapshot_price")
            price = price if price is not None else lc["price"]
            spread = spread if spread is not None else lc["spread"]
            server_time = server_time if server_time is not None else lc["server_time"]
            if news is None:
                news = lc["news"]
                if lc["news_error"] and BLOKIR_JIKA_BERITA_GAGAL:
                    return finish("WAIT", "Pemeriksaan berita gagal (kebijakan blokir aktif)")
    elif frames is None:
        # snapshot/scan_result tersedia tetapi OHLC tidak ada: fail-safe, JANGAN fetch ulang diam-diam.
        dec["snapshot_status"] = "DATA_INCOMPLETE"
        st["DATA"] = {"ok": False, "msg": "DATA_INCOMPLETE: Data OHLC tidak tersedia pada snapshot/scan_result"}
        return finish("WAIT", "DATA_INCOMPLETE: Data OHLC tidak tersedia pada snapshot/scan_result")

    if isinstance(scan_result, dict):
        scan = scan_result

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

    if isinstance(scan_result, dict):
        scan = scan_result
    else:
        _target_scan_fn = getattr(scan_xauusd_zones, "side_effect", None)
        if not callable(_target_scan_fn):
            _target_scan_fn = scan_xauusd_zones
        _scan_sig = inspect.signature(_target_scan_fn)
        _accepts_snapshot = "snapshot" in _scan_sig.parameters or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in _scan_sig.parameters.values()
        )
        if _accepts_snapshot:
            scan = scan_xauusd_zones(frames=frames, current_price=price, snapshot=live_snap)
        else:
            _catat_error(
                "evaluate_trade_decision_signature",
                RuntimeError("scan_xauusd_zones tidak menerima parameter snapshot; menggunakan pemanggilan kompatibilitas lama"),
            )
            scan = scan_xauusd_zones(frames=frames, current_price=price)
    if not dec.get("snapshot_id") and isinstance(scan, dict):
        dec["snapshot_id"] = scan.get("snapshot_id")
    ctx = scan.get("htf_context") or evaluate_htf_context(frames)
    if not scan.get("data_ok"):
        return finish("WAIT", scan.get("empty_reason") or "Scanner tidak dapat berjalan")
    if scan.get("snapshot_status") == "DATA_INCOMPLETE":
        dec["snapshot_status"] = "DATA_INCOMPLETE"
        return finish("WAIT", scan.get("empty_reason") or "DATA_INCOMPLETE: Snapshot MT5 tidak lengkap atau tidak valid")
    if scan.get("is_cached") and float(scan.get("cache_age_seconds") or 0.0) > SCAN_CACHE_TTL:
        return finish("WAIT", "Data cache scanner kedaluwarsa (stale cache)")

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

    # Harmonisasikan pilihan kandidat dengan Primary Directional Zone yang dipilih Python Scanner
    prio_existing = scan.get("radar_priority") or scan.get("prioritized")
    prio_dir = prio_existing.get("primary_directional_zone") if prio_existing else None

    target_match = None
    if prio_dir:
        target_match = next((z for z in cands if z.get("zone_id") == prio_dir.get("zone_id") or 
                            (abs(float(z.get("price_low", 0.0)) - float(prio_dir.get("price_low", 0.0))) < 1e-4 and
                             abs(float(z.get("price_high", 0.0)) - float(prio_dir.get("price_high", 0.0))) < 1e-4)), None)

    if target_match:
        best_candidate_dict = target_match
    else:
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
            if zd == cands[0] and len(cands) == 1:
                st["H1"] = {"ok": False, "msg": "Close H1 menembus invalidasi zona"}
                return finish("INVALID", f"Zona invalid: close H1 menembus {inval:.2f}")
            continue

        if closed_beyond(frames["M30"]):
            if zd == cands[0] and len(cands) == 1:
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
    lo = float(d.get("price_low", 0.0))
    hi = float(d.get("price_high", 0.0))
    mid = float(d.get("midpoint", (lo + hi) / 2.0 if (lo + hi) > 0 else 0.0))
    zt = str(d.get("zone_type", "DEMAND")).upper()
    inval = float(d.get("invalidation_price", lo if zt == "DEMAND" else hi))

    defaults = {
        "zone_id": d.get("zone_id", "ZONE_UNKNOWN"),
        "symbol": d.get("symbol", SYMBOL),
        "timeframe": d.get("timeframe", "H1"),
        "zone_type": zt,
        "price_low": lo,
        "price_high": hi,
        "midpoint": mid,
        "score": float(d.get("score", 70.0)),
        "grade": d.get("grade", "B"),
        "origin_bos": d.get("origin_bos", {}),
        "fvg_confirmation": d.get("fvg_confirmation"),
        "displacement_strength": float(d.get("displacement_strength", 1.5)),
        "freshness": d.get("freshness", "FRESH"),
        "touch_count": int(d.get("touch_count", 0)),
        "liquidity_sweep_status": d.get("liquidity_sweep_status", "NONE"),
        "invalidation_price": inval,
        "is_valid": bool(d.get("is_valid", True)),
        "rejection_reason": d.get("rejection_reason", ""),
    }
    merged = {**defaults, **d}
    keys = Zone.__dataclass_fields__.keys()
    return Zone(**{k: merged[k] for k in keys if k in merged})


def format_diagnostic_report(scan: Optional[Dict[str, Any]] = None) -> str:
    """Format laporan diagnostik komprehensif scanner XAUUSD (TASK Section 5)."""
    if scan is None:
        _diag_snap = create_market_snapshot()
        scan = scan_xauusd_zones(snapshot=_diag_snap)

    if not scan.get("data_ok"):
        return (
            "=" * 70 + "\n"
            "XAUUSD ZONE SCANNER DIAGNOSTIC REPORT\n"
            "=" * 70 + "\n"
            f"Snapshot ID        : {scan.get('snapshot_id', 'N/A')}\n"
            f"Status Data        : {scan.get('snapshot_status') or 'TIDAK VALID / OFFLINE'}\n"
            f"Alasan             : {scan.get('empty_reason', 'Data tidak tersedia')}\n"
            "=" * 70
        )

    diag = scan.get("diagnostics") or {}
    ctx = scan.get("htf_context") or {}
    price = scan.get("current_price", 0.0)
    ts = scan.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    sym = scan.get("symbol", SYMBOL)
    snap_id = scan.get("snapshot_id", "N/A")
    snap_ts = scan.get("snapshot_time") or ts
    snap_price = scan.get("snapshot_price", price)
    r_prio = scan.get("radar_priority") or {}
    prio_ts = r_prio.get("priority_calc_timestamp") or snap_ts
    pz_diag = r_prio.get("primary_directional_zone")
    pz_comp = f"{pz_diag.get('priority_score'):.1f}" if isinstance(pz_diag, dict) and pz_diag.get("priority_score") is not None else "N/A"

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
        f"Snapshot ID        : {snap_id}",
        f"Snapshot Timestamp : {snap_ts}",
        f"Snapshot Price     : {snap_price:.2f}" if isinstance(snap_price, (int, float)) else f"Snapshot Price     : {snap_price}",
        f"Priority Calc Time : {prio_ts}",
        f"Priority Composite : {pz_comp}",
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
        # Tidak ada scan: buat SATU snapshot baru + scan baru. Keputusan (dec) lama yang diberikan berasal
        # dari snapshot lain, sehingga TIDAK boleh digabung dengan scan baru -> dievaluasi ulang dari snapshot baru.
        _pres_snap = create_market_snapshot()
        scan = scan_xauusd_zones(snapshot=_pres_snap)
        dec = evaluate_trade_decision(snapshot=_pres_snap, scan_result=scan)
    elif dec is None:
        # Ada scan, tanpa dec: evaluasi memakai objek snapshot yang tersimpan pada scan (tanpa fetch baru).
        _pres_snap = scan.get("snapshot") if isinstance(scan.get("snapshot"), dict) else None
        dec = evaluate_trade_decision(snapshot=_pres_snap, scan_result=scan)

    price = scan.get("current_price") or dec.get("current_price") or 0.0
    ctx = scan.get("htf_context") or {}
    h4_bias = ctx.get("H4", "NETRAL")
    h1_struct = ctx.get("H1", "NETRAL")
    m30_val = dec.get("stages", {}).get("M30", {})
    m30_status = m30_val.get("status", "CONFIRMED" if m30_val.get("ok") else "UNCONFIRMED")

    zone = dec.get("zone")
    if not zone and scan.get("active_zones"):
        zone = scan["active_zones"][0]
    zone_dict = zone.to_dict() if hasattr(zone, "to_dict") else (dict(zone) if isinstance(zone, dict) else {})

    stages = dec.get("stages", {})
    status = dec.get("status", "WAIT")
    direction = dec.get("direction")
    plan = dec.get("plan")

    def _cand_label(c_dict):
        if not c_dict:
            return "NONE"
        sc = f"Skor {c_dict.get('score', 0.0):.1f}" if c_dict.get("scoring_status") != "DATA_INCOMPLETE" else "DATA INCOMPLETE"
        return f"{c_dict.get('timeframe')} {c_dict.get('zone_type')} ({c_dict.get('price_low', 0.0):.2f}-{c_dict.get('price_high', 0.0):.2f}) [Grade {c_dict.get('grade')} | {sc}]"

    def _format_multi_tier_zone_item(z: Dict[str, Any], curr_p: float) -> List[str]:
        tf = z.get("timeframe", "H1")
        zt = z.get("zone_type", "")
        lo = float(z.get("price_low", 0.0))
        hi = float(z.get("price_high", 0.0))
        prox = float(z.get("proximal_edge") or (hi if zt == "DEMAND" else lo))
        distal = float(z.get("distal_edge") or (lo if zt == "DEMAND" else hi))
        w = float(z.get("zone_width") or round(abs(hi - lo), 2))
        grd = z.get("grade", "B")
        sc_st = z.get("scoring_status", "VALID")
        sc_val = f"{z.get('score', 0.0):.1f}" if sc_st != "DATA_INCOMPLETE" else "DATA INCOMPLETE"
        frsh = z.get("freshness", "FRESH")
        inval = float(z.get("invalidation_price") or distal)
        dist = float(z.get("distance", 0.0))
        d_st = z.get("direction_status") or ("ABOVE_PRICE" if lo > curr_p else ("BELOW_PRICE" if hi < curr_p else "INSIDE_ZONE"))
        app = z.get("approach", "UNTOUCHED")
        atr_r = z.get("distance_atr_ratio")
        atr_str = f"{atr_r:.1f}x ATR" if (atr_r is not None and isinstance(atr_r, (int, float))) else "DATA INCOMPLETE"

        # Reference price & nearest boundary
        ref_p = z.get("reference_price")
        ref_t = z.get("reference_price_type")
        if ref_p is None or not (isinstance(ref_p, (int, float)) and ref_p > 0):
            if zt == "SUPPLY" and isinstance(scan.get("ask"), (int, float)) and scan.get("ask") > 0:
                ref_p = float(scan["ask"])
                ref_t = "ASK"
            elif zt == "DEMAND" and isinstance(scan.get("bid"), (int, float)) and scan.get("bid") > 0:
                ref_p = float(scan["bid"])
                ref_t = "BID"
            else:
                ref_p = curr_p
                ref_t = "MID"
        elif not ref_t:
            ref_t = "ASK" if zt == "SUPPLY" else "BID"
        ref_str = f" | Ref: {ref_p:.2f} ({ref_t}) -> Proximal: {prox:.2f}"

        # Directional context transparan
        is_al = z.get("is_aligned", False)
        is_ct = z.get("is_counter", False)
        if is_al:
            dir_context = f"PRO-TREND (Searah bias HTF {h4_bias})"
        elif is_ct or (h4_bias in ("BULLISH", "BEARISH") and ((h4_bias == "BULLISH" and zt == "SUPPLY") or (h4_bias == "BEARISH" and zt == "DEMAND"))):
            dir_context = f"COUNTERTREND (Melawan bias HTF {h4_bias} — Pantau reaksi, BUKAN sinyal entry)"
        else:
            dir_context = f"NETRAL (Bias HTF {h4_bias})"

        struct_st = "VALID" if (z.get("is_valid", True) and "invalid" not in str(z.get("rejection_reason", "")).lower()) else "INVALID"
        p_elig = z.get("pending_eligibility", "NOT_QUALIFIED")
        p_rsn = z.get("pending_rejection_reason", "")
        c_elig = z.get("confirmed_entry_eligibility", "WAIT")
        c_rsn = z.get("confirmed_rejection_reason", "")

        return [
            f"    - Timeframe & Tipe   : {tf} {zt} [{lo:.2f} - {hi:.2f}] (Proximal: {prox:.2f}, Distal: {distal:.2f}, Lebar: {w:.2f} pt)",
            f"    - Kualitas & Skor    : Grade {grd} | Skor: {sc_val} | {frsh} | Invalidation: {inval:.2f}",
            f"    - Directional Context: {dir_context}",
            f"    - Jarak & Volatilitas: {dist:.2f} pt ({atr_str}) [{app}] ({d_st}){ref_str}",
            f"    - Status Monitoring  : {z.get('status', 'RADAR')}",
            f"    - Alasan Klasifikasi : {z.get('risk_tier_reason') or 'Sesuai ambang batas tier'}",
            f"    - Validitas Struktur : {struct_st}",
            f"    - Pending Eligibility: {p_elig}" + (f" ({p_rsn})" if p_rsn else ""),
            f"    - Confirmed M5       : {c_elig}" + (f" ({c_rsn})" if c_rsn else ""),
        ]

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
    elif zt in ("FAIL", "WAIT") and sw in ("NOT_CHECKED", "WAIT"):
        m5_trig_str = "WAIT (Zone Touch belum terpenuhi — trigger zona M5 belum dijalankan)"
    elif sw != "CONFIRMED":
        m5_trig_str = "WAIT (Sweep M5 zona tidak terpenuhi)"
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
    mid_val = scan.get("mid")
    spread_val = scan.get("spread")
    tick_age = scan.get("tick_age_seconds")
    if bid_val is not None and ask_val is not None and bid_val > 0:
        if mid_val is None:
            mid_val = round((bid_val + ask_val) / 2.0, 2)
        if spread_val is None:
            spread_val = round(ask_val - bid_val, 2)
        bid_ask_str = f" | Bid: {bid_val:.2f} | Ask: {ask_val:.2f}"
    else:
        bid_ask_str = ""
    price_src = scan.get("price_source") or (f"MT5 Live Tick (Bid: {bid_val:.2f} | Ask: {ask_val:.2f} | Mid: {mid_val:.2f} | Spread: {spread_val:.2f} pt)" if bid_val else f"Candle Close Snapshot ({price:.2f})")

    conflict_tag = " [CONFLICT H4 vs H1]" if ctx.get("has_conflict") else ""
    out.append(f"- Current Price    : {price:.2f} (Snapshot: {snapshot_time}{bid_ask_str})" if price > 0 else "- Current Price    : N/A")
    out.append(f"- Sumber Harga     : {price_src}")
    out.append(f"- Multi-TF Bias    : H4: {h4_bias} | H1: {h1_struct}{conflict_tag} | M30: {m30_status} | M5: {'CONFIRMED' if status == 'VALID' else 'WAIT'}")
    out.append(f"- H4 Bias          : {h4_str}")
    out.append(f"- H1 Structure     : {h1_struct}")
    out.append(f"- M30 Momentum     : {m30_status}")
    out.append(f"- M5 Trigger       : {m5_trig_str}")
    atr_diag_pres = dec.get("atr_diagnostics") or scan.get("atr_diagnostics")
    if atr_diag_pres:
        atr_v_str = f"{atr_diag_pres['atr_value']:.2f} pt" if isinstance(atr_diag_pres.get("atr_value"), (int, float)) else "DATA_INCOMPLETE"
        out.append(
            f"- ATR Diagnostics  : Source: {atr_diag_pres.get('atr_source_timeframe', 'H1')} | "
            f"Period: {atr_diag_pres.get('atr_period', 14)} ({atr_diag_pres.get('atr_candle_mode', 'CLOSED_CANDLE')}) | "
            f"Value: {atr_v_str} | Candle: {atr_diag_pres.get('atr_candle_timestamp') or 'N/A'} | "
            f"Status: {atr_diag_pres.get('atr_calculation_status', 'N/A')} | Fallback: {atr_diag_pres.get('atr_fallback_status', 'N/A')}"
        )
    out.append("")

    # B. RADAR ZONE (ZONA PEMANTAUAN & ZONE ANALYSIS)
    out.append("B. RADAR ZONE — 3. RADAR ZONE (ZONA PEMANTAUAN & ZONE ANALYSIS)")
    raw_radar = scan.get("radar_zones") or scan.get("active_zones", [])
    all_radar = [z.to_dict() if hasattr(z, "to_dict") else dict(z) for z in raw_radar]
    r_prio = dec.get("radar_priority") or scan.get("radar_priority")
    _pres_atr = atr_diag_pres.get("atr_value") if (isinstance(atr_diag_pres, dict) and atr_diag_pres.get("atr_value")) else 2.0
    if not r_prio:
        r_prio = prioritize_radar_zones(
            all_radar, price, htf_bias=h4_bias, h1_structure=h1_struct,
            atr=_pres_atr, m5_stages=stages, bid=scan.get("bid"), ask=scan.get("ask"),
            snapshot_time=scan.get("snapshot_time"), snapshot_id=scan.get("snapshot_id")
        )
    b_hr = r_prio.get("best_high_risk") if r_prio else None
    b_mr = r_prio.get("best_medium_risk") if r_prio else None
    b_lr = r_prio.get("best_low_risk") if r_prio else None
    b_pend = r_prio.get("best_pending_candidate") if r_prio else None
    b_conf = r_prio.get("best_confirmed_candidate") if r_prio else None

    if not all_radar:
        out.append("  NO RADAR ZONE DETECTED.")
    else:
        pz = r_prio.get("primary_directional_zone") or r_prio.get("primary_zone")
        nz = r_prio.get("nearest_radar_zone")
        bz = r_prio.get("best_entry_monitoring_zone")
        sz = r_prio.get("secondary_zone")
        cz = r_prio.get("countertrend_zone")

        # Zone Evaluation Summary
        eval_sum = scan.get("zone_evaluation_summary") or {}
        if not eval_sum and all_radar:
            tot_sup = sum(1 for z in all_radar if z.get("zone_type") == "SUPPLY")
            tot_dem = sum(1 for z in all_radar if z.get("zone_type") == "DEMAND")
            tot_fresh = sum(1 for z in all_radar if z.get("freshness") == "FRESH")
            tot_tested = sum(1 for z in all_radar if str(z.get("freshness", "")).startswith("TESTED"))
            eval_sum = {
                "total_candidates": len(all_radar),
                "total_raw": len(all_radar),
                "total_supply": tot_sup,
                "total_demand": tot_dem,
                "total_fresh": tot_fresh,
                "total_tested": tot_tested,
                "total_mitigated": 0,
                "total_invalidated": 0,
                "initial_radar_count": len(all_radar),
                "radar_qualified": len(all_radar),
                "entry_validated_count": 1 if (status == "VALID") else 0,
                "entry_qualified": 1 if (status == "VALID") else 0,
            }

        if eval_sum:
            tot_cand = eval_sum.get('total_candidates', len(all_radar))
            tot_raw = eval_sum.get('raw_candidates', eval_sum.get('total_raw', tot_cand))
            tot_struct = eval_sum.get('structurally_valid', eval_sum.get('count_struct_valid', tot_cand))
            tot_struct_rej = eval_sum.get('structurally_rejected', eval_sum.get('count_structurally_rejected', 0))
            tot_sup = eval_sum.get('total_supply', eval_sum.get('count_supply', 0))
            tot_dem = eval_sum.get('total_demand', eval_sum.get('count_demand', 0))
            tot_fresh = eval_sum.get('total_fresh', eval_sum.get('count_fresh', 0))
            tot_tested = eval_sum.get('total_tested', eval_sum.get('count_tested', 0))
            tot_mit = eval_sum.get('total_mitigated', eval_sum.get('count_mitigated', 0))
            tot_inval = eval_sum.get('total_invalidated', eval_sum.get('count_invalidated', 0))
            tot_exp = eval_sum.get('total_expired', eval_sum.get('count_expired', 0))
            tot_radar = eval_sum.get('radar_qualified', eval_sum.get('initial_radar_count', len(all_radar)))
            tot_pend = eval_sum.get('pending_qualified', 1 if b_pend else 0)
            tot_conf = eval_sum.get('confirmed_entry_qualified', 1 if (status == "VALID" and b_conf) else 0)

            out.append("  [ZONE EVALUATION SUMMARY]:")
            out.append(f"  • Total Kandidat Zona      : {tot_cand}")
            out.append(f"  • Raw Candidates           : {tot_raw}")
            out.append(f"  • Structurally Valid       : {tot_struct}")
            out.append("  • Definisi Struct. Valid   : Fresh Valid + Tested Valid (Mitigated ditampilkan terpisah)")
            out.append(f"  • Radar Qualified          : {tot_radar} (Zona Aktif Pemantauan)")
            out.append(f"  • Pending Qualified (Limit): {tot_pend}")
            out.append(f"  • Confirmed Entry Qualified: {tot_conf}")
            out.append(f"  • Jumlah Supply            : {tot_sup}")
            out.append(f"  • Jumlah Demand            : {tot_dem}")
            out.append(f"  • Jumlah Zona Fresh        : {tot_fresh}")
            out.append(f"  • Detail Granular          : Fresh: {tot_fresh} | Tested: {tot_tested} | Mitigated: {tot_mit} | Invalidated: {tot_inval}")
            out.append(f"  • Rekonsiliasi Status      : Fresh Valid ({tot_fresh}) + Tested Valid ({tot_tested}) + Mitigated ({tot_mit}) + Invalidated ({tot_inval}) + Structurally Rejected ({tot_struct_rej}) + Expired ({tot_exp}) = Total Raw ({tot_raw})")
            out.append("")

        # ZONE COUNT RECONCILIATION (Requirement 1 & 2)
        recon = (r_prio.get("zone_count_reconciliation") if r_prio else None) or dec.get("zone_count_reconciliation") or {}
        if recon:
            out.append("  [ZONE COUNT RECONCILIATION — AUDIT DISTRIBUSI KANDIDAT]:")
            out.append(f"  • Structurally Valid                       : {tot_struct} (Lolos validasi struktur awal)")
            out.append(f"  • Radar Qualified (Zona Aktif)             : {recon.get('radar_qualified_count', tot_radar)} (Memenuhi syarat radar aktif)")
            out.append(f"  • Total Ditampilkan di Multi-Tier          : {recon.get('multi_tier_displayed_count', 0)} (High: {recon.get('high_risk_count', 0)}, Medium: {recon.get('medium_risk_count', 0)}, Low: {recon.get('low_risk_count', 0)})")
            out.append(f"  • Di Luar Kategori High/Medium/Low         : {recon.get('unclassified_tier_count', 0)}")
            for uc_r in recon.get("unclassified_tier_reasons", [])[:3]:
                out.append(f"    - {uc_r}")
            out.append(f"  • Cadangan Tier di Luar Batas Kuota Top 3  : {recon.get('backup_tier_count', 0)}")
            for bk_r in recon.get("backup_tier_reasons", [])[:3]:
                out.append(f"    - {bk_r}")
            fz_reasons = recon.get("far_zones_reasons", [])
            fz_count = recon.get("far_zones_count", len(fz_reasons))
            fz_shown = len(fz_reasons)
            fz_hidden = max(0, fz_count - fz_shown)
            fz_meta = f"Detail Displayed: {fz_shown}/{fz_count}" + (f" | Hidden Details: {fz_hidden}" if fz_hidden > 0 else "")
            out.append(f"  • Ditolak Evaluasi Lanjutan (Far Zone)     : {fz_count} ({fz_meta})")
            for fz_r in fz_reasons:
                out.append(f"    - {fz_r}")
            out.append(f"  • Tidak Ditampilkan Batas Tampilan Ranking : {recon.get('hidden_active_ranking_count', 0)}")
            for hr_r in recon.get("hidden_active_ranking_reasons", [])[:3]:
                out.append(f"    - {hr_r}")
            out.append("")

        # MULTI-TIER RADAR ZONE (Three-Tier Risk Classification Engine)
        out.append("  [MULTI-TIER RADAR ZONE — RISK-BASED SELEKSI RESMI PYTHON SCANNER]:")
        hr_zones = r_prio.get("high_risk_zones") or r_prio.get("multi_tier_zones", {}).get("high_risk", [])
        mr_zones = r_prio.get("medium_risk_zones") or r_prio.get("multi_tier_zones", {}).get("medium_risk", [])
        lr_zones = r_prio.get("low_risk_zones") or r_prio.get("multi_tier_zones", {}).get("low_risk", [])

        out.append("  • A. HIGH RISK ZONES (Aggressive — Peluang Dekat / LTF-MTF):")
        if hr_zones:
            for idx, hz in enumerate(hr_zones[:MULTI_TIER_MAX_ZONES], 1):
                out.append(f"    [{idx}] {hz.get('timeframe')} {hz.get('zone_type')} @ {hz.get('price_low', 0.0):.2f}-{hz.get('price_high', 0.0):.2f}:")
                out.extend(_format_multi_tier_zone_item(hz, price))
        else:
            out.append("    NONE (Tidak ada zona High Risk yang memenuhi kriteria)")

        out.append("  • B. MEDIUM RISK ZONES (Balanced — Kualitas Struktur & Jarak Proporsional):")
        if mr_zones:
            for idx, mz in enumerate(mr_zones[:MULTI_TIER_MAX_ZONES], 1):
                out.append(f"    [{idx}] {mz.get('timeframe')} {mz.get('zone_type')} @ {mz.get('price_low', 0.0):.2f}-{mz.get('price_high', 0.0):.2f}:")
                out.extend(_format_multi_tier_zone_item(mz, price))
        else:
            out.append("    NONE (Tidak ada zona Medium Risk yang memenuhi kriteria)")

        out.append("  • C. LOW RISK ZONES (Conservative — Kualitas Tertinggi & Selaras HTF):")
        if lr_zones:
            for idx, lz in enumerate(lr_zones[:MULTI_TIER_MAX_ZONES], 1):
                out.append(f"    [{idx}] {lz.get('timeframe')} {lz.get('zone_type')} @ {lz.get('price_low', 0.0):.2f}-{lz.get('price_high', 0.0):.2f}:")
                out.extend(_format_multi_tier_zone_item(lz, price))
        else:
            out.append("    NONE (Tidak ada zona Low Risk yang memenuhi kriteria)")
        out.append("")

        out.append("  [4 KATEGORI EVALUASI ZONA — SELEKSI RESMI PYTHON SCANNER]:")
        if pz:
            pz_lbl = f"[{pz.get('status', 'RADAR')}]" + (f" [{pz.get('approach')}]" if pz.get('approach') in ("FAR", "INSIDE_ZONE") else "")
            pz_dir = pz.get("direction_status") or ("ABOVE_PRICE" if pz["price_low"] > price else ("BELOW_PRICE" if pz["price_high"] < price else "INSIDE_ZONE"))
            fresh_str = f"Freshness: {pz.get('freshness', 'FRESH')} | " if pz.get('freshness') else ""
            skor_str = f"Skor: {pz.get('score', 0.0):.1f}" if pz.get("scoring_status") != "DATA_INCOMPLETE" else "Skor: DATA INCOMPLETE"
            out.append(f"  • A. PRIMARY DIRECTIONAL ZONE (Primary Radar Zone) : {pz['timeframe']} {pz['zone_type']} {pz['price_low']:.2f} - {pz['price_high']:.2f} [Grade: {pz['grade']} | {skor_str}] | {fresh_str}Jarak: {pz['distance']:.2f} pt ({pz_dir}) | Status: {pz_lbl}")
            out.append(f"    - Alasan Directional                             : {r_prio.get('primary_reason', '-')}")
        else:
            out.append(f"  • A. PRIMARY DIRECTIONAL ZONE (Primary Radar Zone) : - (Tidak ada zona yang selaras dengan bias {h4_bias})")

        if nz:
            nz_lbl = f"[{nz.get('status', 'RADAR')}]" + (" [COUNTERTREND]" if nz.get("is_counter") else "") + (f" [{nz.get('approach')}]" if nz.get('approach') in ("FAR", "INSIDE_ZONE") else "")
            nz_dir = nz.get("direction_status") or ("ABOVE_PRICE" if nz["price_low"] > price else ("BELOW_PRICE" if nz["price_high"] < price else "INSIDE_ZONE"))
            nz_role = " (Countertrend/Support - pantau reaksi, bukan sinyal)" if nz.get("is_counter") else ""
            fresh_nz = f"Freshness: {nz.get('freshness', 'FRESH')} | " if nz.get('freshness') else ""
            skor_nz = f"Skor: {nz.get('score', 0.0):.1f}" if nz.get("scoring_status") != "DATA_INCOMPLETE" else "Skor: DATA INCOMPLETE"
            out.append(f"  • B. NEAREST RADAR ZONE                             : {nz['timeframe']} {nz['zone_type']} {nz['price_low']:.2f} - {nz['price_high']:.2f} [Grade: {nz['grade']} | {skor_nz}] | {fresh_nz}Jarak: {nz['distance']:.2f} pt ({nz_dir}) | Status: {nz_lbl}{nz_role}")
        else:
            out.append("  • B. NEAREST RADAR ZONE                             : - (Tidak ada zona radar aktif)")

        vz = r_prio.get("best_validated_entry_zone")
        if vz and status == "VALID":
            vz_dir = vz.get("direction_status") or ("ABOVE_PRICE" if vz["price_low"] > price else ("BELOW_PRICE" if vz["price_high"] < price else "INSIDE_ZONE"))
            out.append(f"  • C. BEST VALIDATED ENTRY ZONE                      : {vz['timeframe']} {vz['zone_type']} {vz['price_low']:.2f} - {vz['price_high']:.2f} [Grade: {vz.get('grade', 'A+')}] | Jarak: {vz.get('distance', 0.0):.2f} pt ({vz_dir}) | Status: [VALID]")
            out.append("    - Alasan Validasi                                : Seluruh syarat validasi entry (Zone Touch, Sweep, Trigger M5, Displacement, RR) terpenuhi.")
        else:
            out.append("  • C. BEST VALIDATED ENTRY ZONE                      : NONE")

        az = r_prio.get("alternative_zone")
        if az:
            is_ct = az.get("is_counter") or (cz and (az.get("price_low") == cz.get("price_low") and az.get("zone_type") == cz.get("zone_type")))
            ct_tag = " (COUNTERTREND / REACTION ZONE)" if is_ct else ""
            az_dir = az.get("direction_status") or ("ABOVE_PRICE" if az["price_low"] > price else ("BELOW_PRICE" if az["price_high"] < price else "INSIDE_ZONE"))
            az_skor = f"Skor: {az.get('score', 0.0):.1f}" if az.get("scoring_status") != "DATA_INCOMPLETE" else "Skor: DATA INCOMPLETE"
            out.append(f"  • D. ALTERNATIVE ZONE{ct_tag}                               : {az['timeframe']} {az['zone_type']} {az['price_low']:.2f} - {az['price_high']:.2f} [Grade: {az['grade']} | {az_skor}] | Jarak: {az.get('distance', 0.0):.2f} pt ({az_dir}) | Status: [{az.get('status', 'RADAR')}]{' [COUNTERTREND]' if is_ct else ''}")
            if is_ct:
                opp_act = "BUY" if az['zone_type'] == "DEMAND" else "SELL"
                opp_role = "support/pantau reaksi" if az['zone_type'] == "DEMAND" else "resistance/pantau reaksi"
                out.append(f"    - Catatan Countertrend                            : Area {opp_role} (bukan sinyal {opp_act} karena melawan bias HTF).")
        else:
            out.append("  • D. ALTERNATIVE ZONE                               : NONE")

        if cz and cz != nz and (not az or (az.get("price_low") != cz.get("price_low") or az.get("zone_type") != cz.get("zone_type"))):
            cz_dir = cz.get("direction_status") or ("ABOVE_PRICE" if cz["price_low"] > price else ("BELOW_PRICE" if cz["price_high"] < price else "INSIDE_ZONE"))
            opp_act = "BUY" if cz['zone_type'] == "DEMAND" else "SELL"
            opp_role = "support/pantau reaksi" if cz['zone_type'] == "DEMAND" else "resistance/pantau reaksi"
            out.append(f"  • COUNTERTREND / REACTION ZONE                     : {cz['timeframe']} {cz['zone_type']} {cz['price_low']:.2f} - {cz['price_high']:.2f} [Grade: {cz['grade']}] | Jarak: {cz['distance']:.2f} pt ({cz_dir}) | Status: [RADAR] [COUNTERTREND]")
            out.append(f"    - Catatan Countertrend                            : Area {opp_role} (bukan sinyal {opp_act} karena melawan bias HTF).")
        out.append("")

        # Zone Scoring Summary
        score_sum = r_prio.get("zone_scoring_summary")
        if score_sum:
            out.append("  [ZONE SCORING SUMMARY — DEFINISI RANKING TERPISAH]:")
            act_rank = score_sum.get("active_ranking") or score_sum.get("ranking", [])
            if act_rank:
                out.append("  - A. Active Valid Zone Ranking (Zona Aktif Memenuhi Syarat Radar):")
                for item in act_rank[:5]:
                    st_info = f"Skor {item.get('score', 0.0):.1f}" if item.get("scoring_status") != "DATA_INCOMPLETE" else "DATA INCOMPLETE"
                    grd_info = f"Grade {item.get('grade', '-')}"
                    fresh_info = item.get("freshness", "-")
                    dist_info = f"Jarak {item.get('distance', 0.0):.2f} pt"
                    out.append(f"    {item.get('rank')}. {item.get('zone')} — [{grd_info} | {st_info} | {fresh_info} | {dist_info}]")
                if len(act_rank) > 5:
                    out.append(f"    (Catatan: {len(act_rank) - 5} zona aktif lainnya berada di luar batas tampilan 5 teratas. Ketik 'scan detail' untuk rincian lengkap).")
            else:
                out.append("  - A. Active Valid Zone Ranking: NONE (Tidak ada zona aktif yang valid)")

            raw_rank = score_sum.get("raw_ranking", [])
            if raw_rank:
                out.append("  - B. Raw / Historical Candidate Scoring (Audit Only — Tanpa Otoritas Entry):")
                for item in raw_rank[:5]:
                    st_info = f"{item.get('score', 0.0):.1f}" if item.get("scoring_status") != "DATA_INCOMPLETE" else "DATA INCOMPLETE"
                    v_st = item.get("validity_status", "VALID")
                    rsn = item.get("reason", "")
                    out.append(f"    {item.get('rank')}. {item.get('zone')} (Skor: {st_info}) | Status: {v_st} — {rsn}")
                if len(raw_rank) > 5:
                    out.append(f"    (Catatan: {len(raw_rank) - 5} kandidat mentah lainnya berada di luar batas tampilan 5 teratas).")
                raw_scored_ok = sum(1 for r in raw_rank if r.get("scoring_status") != "DATA_INCOMPLETE")
                raw_excl_inval = sum(1 for r in raw_rank if r.get("validity_status") in ("INVALIDATED", "EXPIRED", "MITIGATED", "REJECTED", "STRUCTURALLY_REJECTED"))
                raw_excl_far = recon.get("far_zones_count", 0) if recon else 0
                out.append(
                    f"    (Audit Raw Scoring: Masuk={len(raw_rank)} | Dinilai={raw_scored_ok} | "
                    f"Top 5 Tampil={min(5, len(raw_rank))} | Di Luar Batas Tampilan={max(0, len(raw_rank) - 5)} | "
                    f"Dikeluarkan dari Active Ranking: Invalid/Exp/Mit/Rej={raw_excl_inval}, Far Zone={raw_excl_far})."
                )

            prio_reasons = score_sum.get("prioritized_reasons", [])
            if prio_reasons:
                out.append("  - Alasan Zona Diprioritaskan:")
                for pr in prio_reasons[:3]:
                    out.append(f"    • {pr}")
            deprio_reasons = score_sum.get("deprioritized_reasons", [])
            if deprio_reasons:
                out.append("  - Alasan Zona Tidak Diprioritaskan:")
                for dr in deprio_reasons[:3]:
                    out.append(f"    • {dr}")
            out.append("")

        # Invalidated / Historical Zones
        inval_list = r_prio.get("invalidated_zones") or dec.get("invalidated_zones") or []
        if inval_list:
            out.append("  [INVALIDATED / HISTORICAL ZONES (Zona historis / tidak aktif untuk entry)]:")
            for iz in inval_list[:5]:
                iz_tf = iz.get("timeframe", "H1")
                iz_t = iz.get("zone_type", "")
                iz_lo = float(iz.get("price_low", 0.0))
                iz_hi = float(iz.get("price_high", 0.0))
                iz_sc = f"{iz.get('score', 0.0):.1f}" if iz.get("scoring_status") != "DATA_INCOMPLETE" else "DATA INCOMPLETE"
                iz_st = iz.get("status") if iz.get("status") in ("INVALIDATED", "EXPIRED", "MITIGATED") else "INVALIDATED"
                iz_rsn = iz.get("rejection_reason") or "Struktur invalid / level tertembus (breached)"
                out.append(f"    • {iz_tf} {iz_t} [{iz_lo:.2f} - {iz_hi:.2f}] (Skor: {iz_sc}) — Status: {iz_st} (Alasan: {iz_rsn})")
            out.append("")
        elif detailed:
            out.append("  [INVALIDATED / HISTORICAL ZONES (Zona historis / tidak aktif untuk entry)]:")
            out.append("    • NONE (Tidak ada zona yang berstatus invalid / breached)")
            out.append("")

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

    # HYBRID ENTRY CANDIDATES SUMMARY
    b_hr = r_prio.get("best_high_risk") if r_prio else None
    b_mr = r_prio.get("best_medium_risk") if r_prio else None
    b_lr = r_prio.get("best_low_risk") if r_prio else None
    b_pend = r_prio.get("best_pending_candidate") if r_prio else None
    b_conf = r_prio.get("best_confirmed_candidate") if r_prio else None

    out.append("  [HYBRID ENTRY CANDIDATES SUMMARY]:")
    out.append(f"  • Best High Risk Candidate   : {_cand_label(b_hr)}")
    out.append(f"  • Best Medium Risk Candidate : {_cand_label(b_mr)}")
    out.append(f"  • Best Low Risk Candidate    : {_cand_label(b_lr)}")
    if b_pend:
        p_pln = b_pend.get("pending_plan") or {}
        out.append(f"  • Best Pending Candidate     : {_cand_label(b_pend)} — Status: READY (Entry: {p_pln.get('entry_price', 0.0):.2f}, SL: {p_pln.get('sl', 0.0):.2f}, TP1: {p_pln.get('tp1', 0.0):.2f}, RR: 1:{p_pln.get('rr', 0.0):.1f})")
    else:
        out.append("  • Best Pending Candidate     : NONE (Belum ada zona yang memenuhi syarat Pending Limit Grade A)")

    if b_conf and status == "VALID":
        out.append(f"  • Best Confirmed Candidate   : {_cand_label(b_conf)} — Status: CONFIRMED (Trigger M5 Lengkap)")
    else:
        out.append("  • Best Confirmed Candidate   : WAIT (Menunggu konfirmasi M5 Touch/Sweep/CHoCH/Displacement/RR)")
    out.append("")

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
        if best_p.get("expiration_block"):
            for el in best_p["expiration_block"].splitlines():
                if el.startswith("Pending Plan Expiration:"):
                    out.append(f"   * {el}")
                else:
                    out.append(f"     {el}")
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

    global_m5_sw = dec.get("global_m5_sweep_status") or scan.get("global_m5_sweep_status") or "NOT_CHECKED"
    out.append(f"   * GLOBAL_M5_SWEEP_STATUS     : {global_m5_sw}")
    out.append(f"   * ZONE_SPECIFIC_SWEEP_STATUS : {sw}" + (f" ({sw_desc})" if sw == "CONFIRMED" and sw_desc else ""))
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
        out.append(f"   * Setup Quality  : HIGH ({zone_dict.get('grade', 'A+') if zone_dict else 'A+'})")
    else:
        out.append(f"   * BUY / SELL / WAIT / INVALID: {status} (NO VALID SETUP)")
        out.append(f"   * Sinyal Resmi   : TIDAK ADA SINYAL AKTIF ({status})")
        out.append("   * Entry Price    : - (Setup belum valid)")
        out.append("   * Stop Loss      : -")
        out.append("   * Take Profit    : -")
        out.append("   * Risk/Reward    : -")
        out.append("   * Setup Quality  : -")
    out.append("")

    # Entry Validation Summary
    val_sum = dec.get("entry_validation_summary")
    if not val_sum:
        mon_zone = (
            f"{zone_dict.get('timeframe', 'H1')} {zone_dict.get('zone_type', '')} {zone_dict.get('price_low', 0.0):.2f}-{zone_dict.get('price_high', 0.0):.2f}"
            if zone_dict else "NONE"
        )
        passed_c = []
        missing_c = []
        for name, val in [("Zone Touch", zt), ("Liquidity Sweep", sw), ("CHoCH/BOS/CISD", tr), ("Displacement", dp), ("Risk/Reward", rr_st)]:
            if val == "CONFIRMED":
                passed_c.append(name)
            else:
                missing_c.append(name)
        val_sum = {
            "monitored_zone": mon_zone,
            "fulfilled_conditions": passed_c,
            "unfulfilled_conditions": missing_c,
            "final_verdict": f"{direction} VALID" if status == "VALID" else status,
        }

    out.append("  [ENTRY VALIDATION SUMMARY]:")
    out.append(f"  • Zona yang Dipantau       : {val_sum.get('monitored_zone', 'NONE')}")
    m_ident = dec.get("monitored_zone_identity")
    if m_ident:
        out.append(f"  • Identitas Zona Terpilih  : ID {m_ident.get('zone_id')} | Arah: {m_ident.get('direction')} | Proximal: {m_ident.get('proximal', 0.0):.2f} | Distal: {m_ident.get('distal', 0.0):.2f} | Status: {m_ident.get('status')}")
    fc = val_sum.get("conditions_met") or val_sum.get("fulfilled_conditions") or val_sum.get("conditions_pass") or []
    out.append(f"  • Kondisi Terpenuhi        : {', '.join(fc) if fc else 'Belum ada'}")
    fail_c = val_sum.get("conditions_unmet") or val_sum.get("unfulfilled_conditions") or val_sum.get("conditions_fail") or []
    if fail_c:
        ufc_text = ', '.join(fail_c)
    elif status == "VALID":
        ufc_text = "None (Semua syarat terpenuhi)"
    elif m_ident or (val_sum.get("monitored_zone") and val_sum.get("monitored_zone") != "NONE"):
        ufc_text = "Zona aktif tersedia dan dipantau; evaluasi trigger M5 menunggu harga menyentuh zona"
    else:
        ufc_text = "Tidak ada zona aktif yang memenuhi syarat untuk dipantau"
    out.append(f"  • Kondisi Belum Terpenuhi  : {ufc_text}")
    nc_c = val_sum.get("conditions_not_checked") or val_sum.get("not_checked_conditions") or []
    if nc_c:
        out.append(f"  • Belum Dijalankan         : {', '.join(nc_c)}")
    inc_c = val_sum.get("conditions_incomplete") or val_sum.get("incomplete_conditions") or []
    if inc_c:
        out.append(f"  • Data Tidak Lengkap       : {', '.join(inc_c)}")
    verdict = sanitize_wait_reason(val_sum.get('final_decision_reason') or val_sum.get('final_verdict') or status)
    out.append(f"  • Alasan Akhir             : {verdict}")
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

    out.append("  [CANDIDATE SELECTION OVERVIEW]:")
    pz_ov = r_prio.get("primary_directional_zone") if r_prio else None
    nz_ov = r_prio.get("nearest_radar_zone") if r_prio else None
    m_ident_ov = dec.get("monitored_zone_identity")
    mon_ov_label = (
        f"{m_ident_ov.get('timeframe')} {m_ident_ov.get('zone_type')} ({m_ident_ov.get('price_low', 0.0):.2f}-{m_ident_ov.get('price_high', 0.0):.2f}) [ID: {m_ident_ov.get('zone_id')}]"
        if m_ident_ov else _cand_label(pz_ov)
    )
    out.append(f"  • Primary Directional Zone       : {_cand_label(pz_ov)}")
    out.append(f"  • Nearest Radar Zone             : {_cand_label(nz_ov)}")
    out.append(f"  • Confirmed Entry Monitoring Zone: {mon_ov_label}")
    if nz_ov and m_ident_ov and nz_ov.get("zone_id") != m_ident_ov.get("zone_id"):
        if nz_ov.get("is_counter"):
            nz_skip_reason = f"berlawanan arah bias HTF ({h4_bias})"
        elif pz_ov and pz_ov.get("zone_id") != nz_ov.get("zone_id"):
            nz_skip_reason = f"prioritas komposit lebih rendah ({float(nz_ov.get('priority_score') or 0.0):.1f} vs {float(pz_ov.get('priority_score') or 0.0):.1f})"
        else:
            nz_skip_reason = f"evaluator entry memantau kandidat {m_ident_ov.get('timeframe')} {m_ident_ov.get('zone_type')} ({m_ident_ov.get('zone_id')})"
        out.append(f"    - Catatan Nearest vs Monitoring: Nearest Radar Zone tidak dipilih sebagai Confirmed Entry Monitoring Zone karena {nz_skip_reason}.")
    out.append(f"  • Best High Risk Zone        : {_cand_label(b_hr)}")
    out.append(f"  • Best Medium Risk Zone      : {_cand_label(b_mr)}")
    out.append(f"  • Best Low Risk Zone         : {_cand_label(b_lr)}")
    out.append(f"  • Best Pending Candidate     : {_cand_label(b_pend) if b_pend else 'NONE'}")
    out.append(f"  • Best Confirmed Candidate   : {_cand_label(b_conf) if (b_conf and status == 'VALID') else 'NONE'}")
    out.append("")

    if not r_prio and all_radar:
        r_prio = prioritize_radar_zones(
            all_radar, price, htf_bias=h4_bias, h1_structure=h1_struct,
            atr=_pres_atr, m5_stages=stages, bid=scan.get("bid"), ask=scan.get("ask"),
            snapshot_time=scan.get("snapshot_time")
        )

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
            eval_sum_w = scan.get("zone_evaluation_summary") or {}
            recon_w = (r_prio.get("zone_count_reconciliation") if r_prio else None) or dec.get("zone_count_reconciliation") or {}
            rq_cnt = recon_w.get("radar_qualified_count")
            if rq_cnt is None:
                rq_cnt = eval_sum_w.get("radar_qualified")
            if rq_cnt is None:
                rq_cnt = len(r_prio.get("radar_qualified", [])) if r_prio else len(all_radar)
            fz_cnt = int(recon_w.get("far_zones_count", 0) or 0)
            inval_list_w = (r_prio.get("invalidated_zones") if r_prio else None) or dec.get("invalidated_zones") or []
            inval_cnt = int(eval_sum_w.get("total_invalidated", 0) or 0)
            mit_cnt = int(eval_sum_w.get("total_mitigated", 0) or 0)
            exp_cnt = int(eval_sum_w.get("total_expired", 0) or 0)
            if not eval_sum_w and inval_list_w:
                inval_cnt = sum(1 for iz in inval_list_w if str(iz.get("status", "")).upper() == "INVALIDATED")
                mit_cnt = sum(1 for iz in inval_list_w if str(iz.get("status", "")).upper() == "MITIGATED")
                exp_cnt = sum(1 for iz in inval_list_w if str(iz.get("status", "")).upper() == "EXPIRED")
            radar_clause = _format_radar_status_clause(
                radar_qualified_count=int(rq_cnt or 0),
                far_zones_count=fz_cnt,
                invalidated_count=inval_cnt,
                mitigated_count=mit_cnt,
                expired_count=exp_cnt,
            )
            dyn_wait = f"Konflik struktur H4 {h4_bias} vs H1 {h1_struct}. {radar_clause}"
        elif not dyn_wait:
            if h4_bias == "NETRAL":
                h4_diag_str = f" ({h4_diag})" if h4_diag else ""
                dyn_wait = f"H4 belum memberikan directional bias yang jelas (NETRAL){h4_diag_str}. Menunggu breakout struktur."
            else:
                dyn_wait = explicit_reasons[0] if explicit_reasons else (scan.get("empty_reason") or "Persyaratan entry belum terpenuhi")

        # Jika h4_diag spesifik ada dan h4 netral, pastikan termuat
        if h4_bias == "NETRAL" and h4_diag and h4_diag not in dyn_wait:
            dyn_wait = f"Bias H4 NETRAL ({h4_diag}). Belum ada directional bias yang jelas. Zona Supply dan Demand dipantau sebagai batas range reaksi, bukan sinyal entry."

        dyn_wait = sanitize_wait_reason(dyn_wait)
        out.append(f"- Final WAIT Reason        : {dyn_wait}")
        out.append(f"- Alasan Keputusan         : {dyn_wait}")

        wb = (r_prio.get("wait_breakdown") if r_prio else None) or dec.get("wait_breakdown") or {}
        p_block = sanitize_wait_reason(wb.get("primary_blocker") or dyn_wait)
        pend_blocks = [sanitize_wait_reason(pb) for pb in (wb.get("pending_limit_blockers") or []) if pb]
        conf_blocks = [sanitize_wait_reason(cb) for cb in (wb.get("confirmed_entry_blockers") or []) if cb]
        sec_warns = [sanitize_wait_reason(sw) for sw in (wb.get("secondary_warnings") or []) if sw]

        out.append("  [WAIT REASON BREAKDOWN]:")
        out.append(f"  • PRIMARY BLOCKER        : {p_block}")
        out.append("  • PENDING LIMIT BLOCKERS :")
        if pend_blocks:
            for pb in pend_blocks:
                out.append(f"    - {pb}")
        else:
            out.append("    - (Syarat Pending Limit terpenuhi)")

        out.append("  • CONFIRMED ENTRY BLOCKERS:")
        if conf_blocks:
            for cb in conf_blocks:
                out.append(f"    - {cb}")
        else:
            out.append("    - (Menunggu konfirmasi M5 lengkap)")

        if sec_warns:
            out.append("  • SECONDARY WARNINGS     :")
            for sw_item in sec_warns:
                out.append(f"    - {sw_item}")

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
        out.append("  * Klasifikasi Risiko Multi-Tier: Klasifikasi HIGH RISK / MEDIUM RISK / LOW RISK merefleksikan karakteristik teknikal setup, BUKAN pembenaran untuk meningkatkan ukuran lot atau risiko modal!")
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
        out.append("- Klasifikasi Risiko Multi-Tier: Klasifikasi HIGH RISK / MEDIUM RISK / LOW RISK merefleksikan karakteristik teknikal setup, BUKAN pembenaran untuk meningkatkan ukuran lot atau risiko modal!")
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


def _extract_risk_monitor_snapshot(
    pos_list: Optional[Any] = None,
    orders_list: Optional[Any] = None,
    risk_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Mengambil ringkasan fakta risiko akun MT5 (posisi running, pending order, dan status SL/TP)
    tanpa mengubah logika trading maupun mengeksekusi transaksi.
    """
    if isinstance(risk_data, dict):
        rp = int(risk_data.get("running_position", 0) or 0)
        po = int(risk_data.get("pending_orders", 0) or 0)
        p_nosl = int(risk_data.get("position_without_sl", 0) or 0)
        o_nosl = int(risk_data.get("pending_without_sl", 0) or 0)
        p_notp = int(risk_data.get("position_without_tp", 0) or 0)
        o_notp = int(risk_data.get("pending_without_tp", 0) or 0)
        return {
            "available": bool(risk_data.get("available", True)),
            "running_position": rp,
            "pending_orders": po,
            "position_without_sl": p_nosl,
            "pending_without_sl": o_nosl,
            "position_without_tp": p_notp,
            "pending_without_tp": o_notp,
            "positions": list(pos_list or ()),
            "orders": list(orders_list or ()),
        }

    available = (pos_list is not None) or (orders_list is not None)
    if pos_list is None or orders_list is None:
        try:
            import MetaTrader5 as mt5
            import modules.mt5_conn as _mt5_conn

            is_conn = bool(getattr(_mt5_conn, "TERHUBUNG_MT5", False))
            if not is_conn:
                try:
                    is_conn = mt5.terminal_info() is not None
                except Exception:
                    is_conn = False
            if is_conn:
                if pos_list is None:
                    pos_list = mt5.positions_get(symbol=SYMBOL) or ()
                if orders_list is None:
                    orders_list = mt5.orders_get(symbol=SYMBOL) or ()
                available = True
        except Exception:
            pass

    positions = list(pos_list or ())
    orders = list(orders_list or ())

    def _get_float(obj: Any, key: str) -> float:
        if isinstance(obj, dict):
            val = obj.get(key, 0.0)
        else:
            val = getattr(obj, key, 0.0)
        try:
            return float(val or 0.0)
        except Exception:
            return 0.0

    pos_no_sl = sum(1 for p in positions if _get_float(p, "sl") <= 0.0)
    ord_no_sl = sum(1 for o in orders if _get_float(o, "sl") <= 0.0)
    pos_no_tp = sum(1 for p in positions if _get_float(p, "tp") <= 0.0)
    ord_no_tp = sum(1 for o in orders if _get_float(o, "tp") <= 0.0)

    return {
        "available": available,
        "running_position": len(positions),
        "pending_orders": len(orders),
        "position_without_sl": pos_no_sl,
        "pending_without_sl": ord_no_sl,
        "position_without_tp": pos_no_tp,
        "pending_without_tp": ord_no_tp,
        "positions": positions,
        "orders": orders,
    }


def format_risk_alert(
    pos_list: Optional[Any] = None,
    orders_list: Optional[Any] = None,
    risk_data: Optional[Dict[str, Any]] = None,
) -> str:
    """Memformat blok RISK MONITOR ringkas beserta peringatan SL/TP kritis."""
    snap_r = _extract_risk_monitor_snapshot(
        pos_list=pos_list, orders_list=orders_list, risk_data=risk_data
    )
    lines = [
        "-" * 56,
        "RISK MONITOR",
        "-" * 56,
    ]
    if not snap_r["available"]:
        lines.extend([
            "RUNNING POSITION : N/A",
            "PENDING ORDERS   : N/A",
            "POSITION WITHOUT SL : N/A",
            "PENDING WITHOUT SL  : N/A",
            "",
            "WARNING:",
            "Data posisi/pending MT5 belum tersedia (MT5 tidak terhubung).",
        ])
        return "\n".join(lines)

    rp = snap_r["running_position"]
    po = snap_r["pending_orders"]
    p_nosl = snap_r["position_without_sl"]
    o_nosl = snap_r["pending_without_sl"]
    p_notp = snap_r["position_without_tp"]
    o_notp = snap_r["pending_without_tp"]

    lines.extend([
        f"RUNNING POSITION : {rp}",
        f"PENDING ORDERS   : {po}",
        f"POSITION WITHOUT SL : {p_nosl}",
        f"PENDING WITHOUT SL  : {o_nosl}",
    ])

    warnings = []
    if p_nosl > 0 and o_nosl > 0:
        warnings.append("Ada posisi dan pending order yang belum memiliki Stop Loss.")
    elif p_nosl > 0:
        warnings.append("Ada posisi running yang belum memiliki Stop Loss.")
    elif o_nosl > 0:
        warnings.append("Ada pending order yang belum memiliki Stop Loss.")

    if p_notp > 0 or o_notp > 0:
        warnings.append("Ada posisi/pending order yang belum memiliki Take Profit.")

    if warnings:
        lines.append("")
        lines.append("WARNING:")
        lines.extend(warnings)

    return "\n".join(lines)


def _norm_stage_label(val: Any, default: str = "NOT CHECKED") -> str:
    if val is None:
        return default
    s = str(val).strip()
    if not s:
        return default
    if s.upper() == "NOT_CHECKED":
        return "NOT CHECKED"
    return s.upper()


def _format_wib_summary_time(raw_ts: Any) -> str:
    dt = to_wib_datetime(raw_ts)
    if dt is None:
        dt = get_current_wib()
    return dt.strftime("%d-%m-%Y %H:%M WIB")


def format_scan_summary(
    scan: Optional[Dict[str, Any]] = None,
    dec: Optional[Dict[str, Any]] = None,
    early_setups: Optional[Dict[str, Any]] = None,
    tracker: Optional[Any] = None,
    pos_list: Optional[Any] = None,
    orders_list: Optional[Any] = None,
    risk_data: Optional[Dict[str, Any]] = None,
) -> str:
    """
    DEFAULT MODE formatter untuk command `scan` (Two-Level Output Display).
    Menampilkan ringkasan singkat, informatif, dan terstruktur tanpa mengubah logika analisis.
    """
    if scan is None:
        _snap = create_market_snapshot()
        scan = scan_xauusd_zones(snapshot=_snap)
        dec = evaluate_trade_decision(snapshot=_snap, scan_result=scan)
    elif dec is None:
        _snap = scan.get("snapshot") if isinstance(scan.get("snapshot"), dict) else None
        dec = evaluate_trade_decision(snapshot=_snap, scan_result=scan)

    price_raw = scan.get("current_price") or dec.get("current_price") or 0.0
    try:
        price = float(price_raw)
    except Exception:
        price = 0.0

    time_str = _format_wib_summary_time(scan.get("snapshot_time") or scan.get("timestamp"))

    ctx = scan.get("htf_context") or scan.get("context") or {}
    stages = dec.get("stages") or {}
    status = str(dec.get("status", "WAIT")).upper()
    direction = dec.get("direction")

    h4_bias = str(ctx.get("H4", "N/A")).upper()
    h1_struct = str(ctx.get("H1", "N/A")).upper()
    m30_stage = stages.get("M30") or {}
    m30_status = str(
        m30_stage.get("status")
        or ("CONFIRMED" if m30_stage.get("ok") else "UNCONFIRMED")
    ).upper()
    m5_status = "CONFIRMED" if status == "VALID" else ("INVALID" if status == "INVALID" else "WAIT")

    if ctx.get("has_conflict"):
        htf_status = "CONFLICT"
    elif h4_bias in ("BULLISH", "BEARISH") and h4_bias == h1_struct:
        htf_status = "ALIGNED"
    elif h4_bias in ("NETRAL", "NEUTRAL"):
        htf_status = "NEUTRAL"
    else:
        htf_status = "UNCONFIRMED"

    raw_radar = scan.get("radar_zones") or scan.get("active_zones") or []
    all_radar = [z.to_dict() if hasattr(z, "to_dict") else dict(z) for z in raw_radar]

    atr_diag_pres = dec.get("atr_diagnostics") or scan.get("atr_diagnostics") or {}
    _pres_atr = atr_diag_pres.get("atr_value") if isinstance(atr_diag_pres, dict) and atr_diag_pres.get("atr_value") else 2.0

    r_prio = dec.get("radar_priority") or scan.get("radar_priority")
    if not r_prio and all_radar:
        r_prio = prioritize_radar_zones(
            all_radar,
            price,
            htf_bias=h4_bias,
            h1_structure=h1_struct,
            atr=_pres_atr,
            m5_stages=stages,
            bid=scan.get("bid"),
            ask=scan.get("ask"),
            snapshot_time=scan.get("snapshot_time"),
            snapshot_id=scan.get("snapshot_id"),
        )
    r_prio = r_prio or {}

    zone_fallback = dec.get("zone") or (all_radar[0] if all_radar else None)
    if zone_fallback and hasattr(zone_fallback, "to_dict"):
        zone_fallback = zone_fallback.to_dict()

    pz = (
        r_prio.get("primary_directional_zone")
        or r_prio.get("primary_zone")
        or r_prio.get("nearest_radar_zone")
        or zone_fallback
    )
    cz = r_prio.get("countertrend_zone")
    sz = r_prio.get("secondary_zone")
    az = sz if sz is not None else r_prio.get("alternative_zone")
    if az and cz and az.get("price_low") == cz.get("price_low") and az.get("price_high") == cz.get("price_high") and az.get("zone_type") == cz.get("zone_type"):
        az = None

    def _short_zone_line(z_item: Optional[Dict[str, Any]]) -> str:
        if not z_item:
            return "NONE"
        tf = z_item.get("timeframe", "N/A")
        zt_name = z_item.get("zone_type", "N/A")
        lo = float(z_item.get("price_low", 0.0) or 0.0)
        hi = float(z_item.get("price_high", 0.0) or 0.0)
        return f"{tf} {zt_name} {lo:.2f} - {hi:.2f}"

    # Entry Validation states (gunakan hasil stage scanner secara langsung)
    zt_stage = stages.get("zone_touch") or {}
    zt = _norm_stage_label(zt_stage.get("status"), "CONFIRMED" if status == "VALID" else "WAIT")
    zt_ok = bool(zt_stage.get("ok")) or zt in ("CONFIRMED", "PASS", "PASSED", "TOUCHED", "INSIDE")

    if not zt_ok and status != "VALID":
        sw_raw = (stages.get("liquidity_sweep") or {}).get("status")
        tr_raw = (stages.get("structure_trigger") or {}).get("status")
        dp_raw = (stages.get("displacement") or {}).get("status")
        rr_raw = (stages.get("risk_reward") or {}).get("status")
        sw = _norm_stage_label(sw_raw if sw_raw not in (None, "WAIT") else "NOT CHECKED")
        tr = _norm_stage_label(tr_raw if tr_raw not in (None, "WAIT") else "NOT CHECKED")
        dp = _norm_stage_label(dp_raw if dp_raw not in (None, "WAIT") else "NOT CHECKED")
        rr_val_str = _norm_stage_label(rr_raw if rr_raw not in (None, "WAIT") else "NOT CHECKED")
    else:
        sw = _norm_stage_label((stages.get("liquidity_sweep") or {}).get("status"), "CONFIRMED" if status == "VALID" else "WAIT")
        tr = _norm_stage_label((stages.get("structure_trigger") or {}).get("status"), "CONFIRMED" if status == "VALID" else "WAIT")
        dp = _norm_stage_label((stages.get("displacement") or {}).get("status"), "CONFIRMED" if status == "VALID" else "WAIT")
        rr_stage = stages.get("risk_reward") or {}
        rr_st = _norm_stage_label(rr_stage.get("status"), "CONFIRMED" if status == "VALID" else "NOT CHECKED")
        rr_num = rr_stage.get("rr")
        if rr_st == "CONFIRMED" and isinstance(rr_num, (int, float)) and rr_num > 0:
            rr_val_str = f"CONFIRMED (1:{float(rr_num):.2f})"
        else:
            rr_val_str = rr_st

    plan = dec.get("plan") or dec.get("entry_plan")
    if not direction and isinstance(plan, dict):
        direction = plan.get("direction")

    if status == "VALID" and direction:
        sig_str = f"{direction} VALID"
        dir_str = str(direction).upper()
        if isinstance(plan, dict) and "entry_min" in plan and "entry_max" in plan:
            entry_str = (
                f"{float(plan['entry_min']):.2f} - {float(plan['entry_max']):.2f} "
                f"(SL: {float(plan.get('sl', 0.0)):.2f} | TP1: {float(plan.get('tp1', 0.0)):.2f} | "
                f"RR: 1:{float(plan.get('rr', 0.0)):.2f})"
            )
        else:
            entry_str = "CONFIRMED"
        reason_str = "; ".join(str(r) for r in (dec.get("reasons") or ["M5 CONFIRMED"])[:2])
    else:
        sig_str = status
        dir_str = "NONE"
        entry_str = "NONE"
        wb = (r_prio.get("wait_breakdown") if r_prio else None) or dec.get("wait_breakdown") or {}
        if ctx.get("has_conflict") and m5_status != "CONFIRMED":
            reason_str = "HTF CONFLICT + M5 NOT CONFIRMED"
        elif wb.get("primary_blocker"):
            reason_str = sanitize_wait_reason(str(wb["primary_blocker"]))
        elif dec.get("reasons"):
            reason_str = sanitize_wait_reason(str(dec["reasons"][0]).replace("[SCANNER] Rejection Reason: ", ""))
        else:
            reason_str = "WAIT FOR CONFIRMATION"

    # Radar Summary counts (ambil langsung dari scan / reconciliation)
    eval_sum = scan.get("zone_evaluation_summary") or {}
    recon = (r_prio.get("zone_count_reconciliation") if r_prio else None) or dec.get("zone_count_reconciliation") or {}

    active_zones_cnt = recon.get("radar_qualified_count", eval_sum.get("radar_qualified", len(all_radar)))
    fresh_cnt = eval_sum.get("total_fresh", sum(1 for z in all_radar if z.get("freshness") == "FRESH"))
    invalidated_cnt = eval_sum.get("total_invalidated", 0)
    mitigated_cnt = eval_sum.get("total_mitigated", 0)
    far_excluded_cnt = recon.get("far_zones_count", len(recon.get("far_zones_reasons", [])))

    out = [
        "=" * 56,
        "             XAUUSD PYTHON SMC SCANNER",
        "=" * 56,
        "",
        f"PRICE       : {price:.2f}" if price > 0 else "PRICE       : N/A",
        f"TIME        : {time_str}",
        "",
        "MARKET BIAS",
        f"H4          : {h4_bias}",
        f"H1          : {h1_struct}",
        f"M30         : {m30_status}",
        f"M5          : {m5_status}",
        f"HTF STATUS  : {htf_status}",
        "",
        "-" * 56,
        "PRIMARY RADAR ZONE",
        "-" * 56,
    ]

    if pz:
        pz_type = f"{pz.get('timeframe', 'N/A')} {pz.get('zone_type', 'N/A')}".strip()
        pz_lo = float(pz.get("price_low", 0.0) or 0.0)
        pz_hi = float(pz.get("price_high", 0.0) or 0.0)
        pz_grd = pz.get("grade", "N/A")
        pz_sc = pz.get("score")
        grade_str = f"{pz_grd} ({float(pz_sc):.1f})" if isinstance(pz_sc, (int, float)) else str(pz_grd)
        pz_fresh = pz.get("freshness", "N/A")
        pz_dist = pz.get("distance")
        dist_str = f"{float(pz_dist):.2f} pt" if isinstance(pz_dist, (int, float)) else "N/A"
        pz_stat = str(pz.get("approach") or pz.get("status") or "RADAR").upper()
        out.extend([
            f"TYPE        : {pz_type}",
            f"ZONE        : {pz_lo:.2f} - {pz_hi:.2f}",
            f"GRADE       : {grade_str}",
            f"FRESHNESS   : {pz_fresh}",
            f"DISTANCE    : {dist_str}",
            f"STATUS      : {pz_stat}",
        ])
    else:
        out.extend([
            "TYPE        : NONE",
            "ZONE        : N/A",
            "GRADE       : N/A",
            "FRESHNESS   : N/A",
            "DISTANCE    : N/A",
            "STATUS      : NONE",
        ])

    out.extend([
        "",
        f"ALTERNATIVE : {_short_zone_line(az)}",
        f"COUNTERTREND: {_short_zone_line(cz)}",
        "",
        "-" * 56,
        "ENTRY VALIDATION",
        "-" * 56,
        f"ZONE TOUCH      : {zt}",
        f"LIQUIDITY SWEEP : {sw}",
        f"CHoCH / BOS     : {tr}",
        f"DISPLACEMENT    : {dp}",
        f"RR VALIDATION   : {rr_val_str}",
        "",
        "-" * 56,
        "FINAL DECISION",
        "-" * 56,
        f"SIGNAL      : {sig_str}",
        f"DIRECTION   : {dir_str}",
        f"ENTRY       : {entry_str}",
        f"REASON      : {reason_str}",
        "",
        "-" * 56,
        "RADAR SUMMARY",
        "-" * 56,
        f"ACTIVE ZONES    : {active_zones_cnt}",
        f"FRESH           : {fresh_cnt} (total scanner freshness count)",
        f"INVALIDATED     : {invalidated_cnt}",
        f"MITIGATED       : {mitigated_cnt}",
        f"FAR EXCLUDED    : {far_excluded_cnt}",
        "",
        format_risk_alert(pos_list=pos_list, orders_list=orders_list, risk_data=risk_data),
        "",
        "=" * 56,
    ])
    return "\n".join(out)


def format_scan_detail(
    scan: Optional[Dict[str, Any]] = None,
    dec: Optional[Dict[str, Any]] = None,
    early_setups: Optional[Dict[str, Any]] = None,
    tracker: Optional[Any] = None,
    pos_list: Optional[Any] = None,
    orders_list: Optional[Any] = None,
    risk_data: Optional[Dict[str, Any]] = None,
    include_mt5_positions: bool = True,
) -> str:
    """
    DETAIL MODE formatter untuk command `scan detail` (Two-Level Output Display).
    Mempertahankan seluruh informasi teknis lengkap dari scanner + rekonsiliasi Smart Radar + detail posisi/pending MT5.
    """
    if scan is None:
        _snap = create_market_snapshot()
        scan = scan_xauusd_zones(snapshot=_snap)
        dec = evaluate_trade_decision(snapshot=_snap, scan_result=scan)
    elif dec is None:
        _snap = scan.get("snapshot") if isinstance(scan.get("snapshot"), dict) else None
        dec = evaluate_trade_decision(snapshot=_snap, scan_result=scan)

    base = format_scan_presentation(
        scan, dec, early_setups=early_setups, tracker=tracker, detailed=True
    )
    sections = [base]

    # Tambahan rekonsiliasi Smart Radar penuh
    try:
        from modules.pantau import collect_radar_watchlist_with_reconciliation
        wl, rec = collect_radar_watchlist_with_reconciliation(scan, dec)
        if rec:
            rec_lines = [
                "",
                "-" * 70,
                "F. SMART RADAR RECONCILIATION & LIVE ACCOUNT RISK DETAIL",
                "-" * 70,
                (
                    f"• Rekonsiliasi Smart Radar : Scanner Qualified={rec.get('scanner_radar_qualified_count', 0)} | "
                    f"Received={rec.get('monitor_received_count', 0)} | Active={rec.get('monitor_active_count', 0)} | "
                    f"Excluded={rec.get('excluded_count', 0)} | Status={rec.get('transfer_status', 'OK')}"
                ),
            ]
            for ex in rec.get("excluded_details", [])[:10]:
                rec_lines.append(
                    f"  - Excluded [{ex.get('zone_id')}]: {ex.get('timeframe')} {ex.get('zone_type')} "
                    f"({ex.get('initial_status')} -> {ex.get('final_status')}) | Reason: {ex.get('reason')}"
                )
            sections.append("\n".join(rec_lines))
    except Exception:
        pass

    # Risk Monitor & Detail Posisi Running / Pending Order MT5
    if include_mt5_positions and pos_list is None and orders_list is None and risk_data is None:
        try:
            from modules.mt5_conn import cek_posisi_terbuka_mt5
            import modules.mt5_conn as _mt5_conn
            if getattr(_mt5_conn, "TERHUBUNG_MT5", False):
                mt5_info = cek_posisi_terbuka_mt5(SYMBOL)
                if mt5_info:
                    sections.append(str(mt5_info).strip())
        except Exception:
            pass

    sections.append(format_risk_alert(pos_list=pos_list, orders_list=orders_list, risk_data=risk_data))
    sections.append("")
    sections.append(format_scanner_audit(dec))
    sections.append("=" * 70)
    return "\n".join(sections)
