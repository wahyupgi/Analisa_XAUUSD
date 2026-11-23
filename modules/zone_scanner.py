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
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple, Union, NamedTuple

import numpy as np
import pandas as pd

from modules.config import (
    SYMBOL, TIMEFRAMES, SWING_N, RR_MIN, SL_JARAK_MAKS, ENTRY_LEBAR_MAKS,
    SPREAD_MAKS_POIN, BLOKIR_JIKA_BERITA_GAGAL, _DIR_SCRIPT,
    FILE_STATE_ZONE_TRACKER, DEFAULT_PENDING_EXPIRATION_HOURS, ZONE_APPROACH_DISTANCE_POINTS,
    PANTAU_EXPIRY_WARNING_MINUTES,
    MULTI_TIER_MAX_ZONES, HIGH_RISK_MAX_ATR_RATIO, HIGH_RISK_MAX_DISTANCE_PT, HIGH_RISK_MIN_SCORE,
    MEDIUM_RISK_MAX_ATR_RATIO, MEDIUM_RISK_MAX_DISTANCE_PT, MEDIUM_RISK_MIN_SCORE,
    LOW_RISK_MAX_ATR_RATIO, LOW_RISK_MAX_DISTANCE_PT, LOW_RISK_MIN_SCORE,
    ZONE_RELEVANCE_TF_LIMITS,
    SIGNAL_STATUS_NO_SETUP, SIGNAL_STATUS_SETUP_DETECTED, SIGNAL_STATUS_WAITING_PULLBACK,
    SIGNAL_STATUS_WAITING_CONFIRMATION, SIGNAL_STATUS_SIGNAL_CONFIRMED, SIGNAL_STATUS_ENTRY_MISSED,
    INVALIDATION_SETUP_SL_BREACHED, INVALIDATION_ZONE_STRUCTURAL,
)
from modules.utils import (
    _catat_error,
    WIB_TZ,
    get_current_wib,
    to_wib_datetime,
    format_wib_timestamp,
    get_symbol_unit_spec,
    extract_unit_spec_from_container,
    calc_price_distance,
    price_distance_to_mt5_points,
    price_distance_to_pips,
    format_price_distance,
)
from modules.mt5_conn import MT5_IPC_LOCK, ambil_data, pastikan_mt5_aktif, tambah_indikator
from modules.trend_following import (
    _is_snr_zone_broken_or_inactive,
    _snr_frames_signature,
    assess_m30_condition,
    evaluate_h1_trend_confirmation,
    evaluate_h4_primary_trend,
    evaluate_snr_opposing_barrier,
    evaluate_trend_following_setup,
    extract_snr_supporting_context,
)

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
    bos_timeframe: str = ""
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
    is_broken: bool = False
    invalid_label: str = ""
    invalidation_type: str = ""
    setup_invalidation_price: float = 0.0
    original_grade: str = ""
    zone_relevance: str = "RADAR"             # ACTIVE / RADAR / FAR_REFERENCE / INVALID
    zone_relevance_reason: str = ""
    liquidity_sweep_evidence: Optional[Dict[str, Any]] = None
    evidence: Dict[str, Any] = field(default_factory=dict)

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
    """FVG 3-candle >= min_atr_mult*ATR dengan candle impuls searah dan status UNTOUCHED / PARTIALLY_MITIGATED / FULLY_MITIGATED."""
    if df is None or len(df) < 3:
        return []
    atr = _atr_array(df)
    O, H, L, C, idx = df["Open"].values, df["High"].values, df["Low"].values, df["Close"].values, df.index
    out = []
    for i in range(2, len(df)):
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            continue
        if L[i] > H[i - 2] and C[i - 1] > O[i - 1] and (L[i] - H[i - 2]) >= min_atr_mult * a:
            lo, hi, typ = float(H[i - 2]), float(L[i]), "BULLISH_FVG"
            sub = L[i + 1:]
            st = "UNTOUCHED" if len(sub) == 0 or sub.min() >= hi else ("FULLY_MITIGATED" if sub.min() <= lo else "PARTIALLY_MITIGATED")
        elif H[i] < L[i - 2] and C[i - 1] < O[i - 1] and (L[i - 2] - H[i]) >= min_atr_mult * a:
            lo, hi, typ = float(H[i]), float(L[i - 2]), "BEARISH_FVG"
            sub = H[i + 1:]
            st = "UNTOUCHED" if len(sub) == 0 or sub.max() <= lo else ("FULLY_MITIGATED" if sub.max() >= hi else "PARTIALLY_MITIGATED")
        else:
            continue
        out.append({
            "idx": i - 1, "time": idx[i - 1], "c1_time": idx[i - 2], "c3_time": idx[i],
            "type": typ, "price_low": lo, "price_high": hi,
            "size_points": float(hi - lo), "size_atr": float((hi - lo) / a), "status": st,
        })
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
            dir_bodies = [float(C[x] - O[x]) for x in range(ob_idx + 1, j_end + 1) if C[x] > O[x]]
        else:
            disp = (ob_hi - float(np.min(L[seg]))) / a
            dir_bodies = [float(O[x] - C[x]) for x in range(ob_idx + 1, j_end + 1) if C[x] < O[x]]
        max_body = (max(dir_bodies) / a) if dir_bodies else 0.0
        if k - ob_idx > MAX_BOS_DISTANCE:
            reasons.append(f"BOS terlalu jauh dari OB ({k - ob_idx} candle > {MAX_BOS_DISTANCE}); bukan impuls langsung")
        if bos_strength < MIN_BOS_STRENGTH_ATR:
            reasons.append(f"BOS tipis (close hanya {bos_strength:.2f} ATR melewati level)")
        if disp < MIN_DISPLACEMENT_ATR:
            reasons.append(f"Displacement tidak memadai ({disp:.2f}x ATR < {MIN_DISPLACEMENT_ATR}x)")
        if max_body < MIN_IMPULSE_BODY_ATR:
            reasons.append(f"Tidak ada candle impuls searah bertubuh kuat (max body {max_body:.2f} ATR)")

        # FVG konfirmasi: searah, terbentuk pada leg impuls setelah OB, belum terisi penuh, relevan harga dengan OB, candle c3 sudah close
        want = "BULLISH_FVG" if bull else "BEARISH_FVG"
        best = None
        j_fvg_end = min(len(df) - 2, k + 4)
        for f in fvgs:
            if f["type"] != want or f["status"] == "FULLY_MITIGATED" or not (ob_idx + 1 <= f["idx"] <= j_fvg_end):
                continue
            if (bull and f["price_high"] < ob_lo) or ((not bull) and f["price_low"] > ob_hi):
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

        # Sentuhan & invalidasi (candle CLOSE pada timeframe asal; invalidasi dicek pasca-OB, sentuhan pasca-BOS)
        touches, invalid = 0, False
        break_candle_time, break_close = None, None
        for x in range(ob_idx + 1, len(df)):
            if bull and C[x] < ob_lo:
                invalid = True
                break_candle_time, break_close = idx[x], float(C[x])
                reasons.append(f"INVALID — BROKEN (Close {timeframe} {C[x]:.2f} < {ob_lo:.2f})")
                break
            if (not bull) and C[x] > ob_hi:
                invalid = True
                break_candle_time, break_close = idx[x], float(C[x])
                reasons.append(f"INVALID — BROKEN (Close {timeframe} {C[x]:.2f} > {ob_hi:.2f})")
                break
            if x >= k + 1:
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
            reasons.append(f"Zona terlalu lebar ({width:.2f} USD > {cap:.2f} USD)")
        if width < MIN_ZONE_WIDTH_POINTS:
            reasons.append(f"Zona terlalu sempit ({width:.2f} USD)")

        # Sweep pendukung (sisi yang tepat, di sekitar waktu & harga pembentukan OB, tidak boleh sesudah BOS)
        want_sw = "SWEEP_LOW" if bull else "SWEEP_HIGH"
        sw_tol = max(1.5 * a, 2.0)
        sw = next(
            (
                s for s in sweeps
                if s["type"] == want_sw
                and ob_idx - 6 <= s["k"] <= min(ob_idx + 2, k)
                and (ob_lo - sw_tol <= float(s.get("wick_extreme", s.get("level", 0.0))) <= ob_hi + sw_tol)
            ),
            None,
        )
        sweep_status = ("CONFIRMED_SWEEP_LOW" if bull else "CONFIRMED_SWEEP_HIGH") if sw else "NONE"
        sweep_ev = (
            {
                "type": sw.get("type"),
                "time": str(sw.get("time")),
                "level": round(float(sw.get("level", 0.0)), 2),
                "wick_extreme": round(float(sw.get("wick_extreme", 0.0)), 2),
                "pool_type": sw.get("pool_type"),
                "description": sw.get("description"),
            }
            if sw else None
        )

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
        zone_evidence = {
            "timeframe": timeframe,
            "candle_closed": True,
            "ob_candle_time": str(idx[ob_idx]),
            "bos_candle_time": str(idx[k]),
            "ob_to_bos_bars": int(k - ob_idx),
            "bos_type": ev.get("tipe"),
            "bos_level": round(float(ev.get("level", 0.0)), 2),
            "bos_strength_atr": round(float(bos_strength), 2),
            "ob_range": [round(ob_lo, 2), round(ob_hi, 2)],
            "fvg_range": [round(float(fvg["price_low"]), 2), round(float(fvg["price_high"]), 2)] if fvg else None,
            "fvg_time": str(fvg["time"]) if fvg else None,
            "fvg_c3_time": str(fvg["c3_time"]) if fvg and fvg.get("c3_time") is not None else None,
            "fvg_status": fvg["status"] if fvg else "NONE",
            "displacement_atr": round(float(disp), 2),
            "impulse_max_body_atr": round(max_body, 2),
            "freshness": fresh,
            "touch_count": touches,
            "mitigation_status": fresh,
            "break_candle_time": str(break_candle_time) if break_candle_time is not None else None,
            "break_close": round(break_close, 2) if break_close is not None else None,
            "sweep": sweep_ev,
            "score_reason": "",
        }
        z = Zone(
            zone_id=f"ZONE_{timeframe}_{zone_type}_{int(idx[ob_idx].timestamp())}", symbol=SYMBOL,
            timeframe=timeframe, zone_type=zone_type, price_low=round(ob_lo, 2), price_high=round(ob_hi, 2),
            midpoint=round(mid, 2), score=0.0, grade="REJECTED", origin_bos=ev_out, fvg_confirmation=fvg,
            displacement_strength=round(float(disp), 2), freshness=fresh, touch_count=touches,
            liquidity_sweep_status=sweep_status, invalidation_price=round(ob_lo if bull else ob_hi, 2),
            is_valid=not reasons, rejection_reason="; ".join(reasons), ob_candle_time=idx[ob_idx],
            bos_candle_time=idx[k], bos_timeframe=timeframe, sl_recommended=sl, tp1_recommended=tp1, tp2_recommended=tp2,
            audit=(["OB lolos seluruh hard validation"] if not reasons else list(reasons)),
            is_broken=invalid, invalid_label=("INVALID — BROKEN" if invalid else ""),
            liquidity_sweep_evidence=sweep_ev,
            evidence=zone_evidence,
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
        if isinstance(getattr(zone, "evidence", None), dict):
            zone.evidence["score_reason"] = zone.score_reason
        return zone

    # 2. Hard Validation Rejection Check (Non-Radar) & Mitigated/Invalidated Gate
    touch_cnt = int(getattr(zone, "touch_count", 0) or 0)
    is_dead_zone = zone.freshness in ("INVALIDATED", "MITIGATED") or touch_cnt > MAX_TOUCH_COUNT
    if is_dead_zone:
        zone.is_valid = False
        if not zone.rejection_reason:
            zone.rejection_reason = f"Zona sudah {zone.freshness} (sentuhan={touch_cnt})"
    is_inval_rej = "invalid" in (zone.rejection_reason or "").lower() or is_dead_zone
    if not zone.is_valid and (not allow_radar or is_inval_rej):
        zone.score, zone.grade = 0.0, "REJECTED"
        zone.scoring_status = "VALID"
        zone.score_breakdown = {
            "status": "VALID",
            "konteks_htf": 0.0, "struktur_bos": 0.0, "displacement": 0.0,
            "fvg_konfirmasi": 0.0, "freshness": 0.0, "liquidity_sweep": 0.0,
            "htf_alignment": 0.0, "market_structure": 0.0, "zone_strength": 0.0,
            "fvg_quality": 0.0, "liquidity_context": 0.0,
            "impulse_body_atr": round(float(ev.get("impulse_max_body_atr", ev.get("impulse_body_atr", 0.0)) or 0.0), 2),
            "proximity": getattr(zone, "approach", "UNTOUCHED"),
        }
        zone.score_reason = f"Skor 0.0 (Hard validation rejected: {zone.rejection_reason or 'tidak valid'})"
        if isinstance(getattr(zone, "evidence", None), dict):
            zone.evidence["score_reason"] = zone.score_reason
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
        dist_res = calculate_zone_distance(zone.price_low, zone.price_high, current_price)
        prox_status = dist_res.status
        zone.distance = dist_res.distance
        zone.direction_status = dist_res.status

    impulse_body_atr = round(float(ev.get("impulse_max_body_atr", ev.get("impulse_body_atr", 0.0)) or 0.0), 2)
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
        "impulse_body_atr": impulse_body_atr,
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
        raw_grade = _grade(total)
        cap_reasons: List[str] = []
        final_grade = raw_grade
        if raw_grade in ("A+", "A"):
            if zone.freshness == "TESTED_2X" or touch_cnt >= 2:
                final_grade = "B"
                cap_reasons.append("zona sudah TESTED_2X")
            elif p_fvg <= 0:
                final_grade = "B"
                cap_reasons.append("tanpa FVG valid")
            elif p_disp <= 0:
                final_grade = "B"
                cap_reasons.append("displacement lemah")
            elif p_htf <= 0:
                final_grade = "B"
                cap_reasons.append("melawan arah HTF")
            elif raw_grade == "A+":
                if zone.freshness != "FRESH" or touch_cnt > 0 or p_disp < 16.0:
                    final_grade = "A"
                    cap_reasons.append("A+ memerlukan FRESH + displacement >= 2.0 ATR")
        zone.grade = final_grade
        cap_note = f" [Cap: {', '.join(cap_reasons)}]" if cap_reasons else ""
        zone.score_reason = f"Skor {total:.1f} [{zone.grade}]{cap_note}: " + ", ".join(reasons_list)
        if zone.grade == "REJECTED":
            zone.is_valid = False
            zone.rejection_reason = f"Skor {total:.1f} < {SCORE_MIN_ACTIVE}"
            zone.audit.append(zone.rejection_reason)
        else:
            zone.audit.append(f"Diterima: grade {zone.grade}, skor {total:.1f}{cap_note}")

    if isinstance(getattr(zone, "evidence", None), dict):
        zone.evidence["score_reason"] = zone.score_reason
        zone.evidence["freshness"] = zone.freshness
        zone.evidence["touch_count"] = int(getattr(zone, "touch_count", 0) or 0)
        zone.evidence["mitigation_status"] = zone.freshness

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
    Zona pada timeframe berdekatan / sebanding tetap digabung menjadi konfluensi multi-TF,
    termasuk ketika terpisahkan oleh zona refinement LTF di antaranya.
    """
    if len(zones) <= 1:
        return zones
    out: List[Zone] = []
    order = {"FRESH": 0, "TESTED_1X": 1, "TESTED_2X": 2, "MITIGATED": 3, "INVALIDATED": 4}
    ltf_tfs = {"M5", "M15"}
    htf_tfs = {"H1", "H4", "D1", "W1"}

    for z in sorted(zones, key=lambda x: (x.zone_type, x.price_low)):
        z_tfs = z.timeframe.split("+")
        z_is_ltf = any(t in ltf_tfs for t in z_tfs)
        z_is_htf = any(t in htf_tfs for t in z_tfs)
        z_w = round(z.price_high - z.price_low, 2)

        merged_idx = None
        for idx in range(len(out) - 1, -1, -1):
            cand = out[idx]
            if cand.zone_type != z.zone_type:
                continue
            cand_tfs = cand.timeframe.split("+")
            cand_is_ltf = any(t in ltf_tfs for t in cand_tfs)
            cand_is_htf = any(t in htf_tfs for t in cand_tfs)
            same_tier = (set(cand_tfs) == set(z_tfs)) or (cand_is_ltf and z_is_ltf) or (cand_is_htf and z_is_htf)
            gap_tol = 0.35 if same_tier else 0.0
            if min(cand.price_high, z.price_high) + gap_tol < max(cand.price_low, z.price_low):
                continue

            cand_w = round(cand.price_high - cand.price_low, 2)

            if (cand_is_ltf and z_is_htf and z_w >= 1.8 * cand_w) or (z_is_ltf and cand_is_htf and cand_w >= 1.8 * z_w):
                continue
            merged_idx = idx
            break

        if merged_idx is None:
            out.append(z)
            continue

        last = out[merged_idx]
        lo, hi = min(last.price_low, z.price_low), max(last.price_high, z.price_high)
        max_w = MAX_ZONE_WIDTH_POINTS if any("H4" in tf for tf in (last.timeframe, z.timeframe)) else 14.0
        if hi - lo > max_w:
            inter_lo, inter_hi = max(last.price_low, z.price_low), min(last.price_high, z.price_high)
            if inter_hi - inter_lo >= MIN_ZONE_WIDTH_POINTS:
                lo, hi = inter_lo, inter_hi
            else:
                best_b = last if last.score >= z.score else z
                lo, hi = best_b.price_low, best_b.price_high
        best = last if last.score >= z.score else z
        tfs = sorted({*last.timeframe.split("+"), *z.timeframe.split("+")}, key=lambda t: -TF_MINUTES.get(t, 0))
        demand = last.zone_type == "DEMAND"
        mid = round((lo + hi) / 2.0, 2)
        merged_fresh = max(last.freshness, z.freshness, key=lambda f: order.get(f, 9))
        merged_valid = (merged_fresh not in ("MITIGATED", "INVALIDATED")) and bool(last.is_valid or z.is_valid)
        constituents = sorted({*last.merged_from, *z.merged_from, last.zone_id, z.zone_id})
        identity = hashlib.sha256("|".join(constituents).encode("utf-8")).hexdigest()[:16]
        # Jaga batas invalidasi struktural distal konstituen & waktu BOS terbaru agar tidak terinvalidasi prematur
        merged_inval = round(
            min(lo, last.invalidation_price, z.invalidation_price)
            if demand
            else max(hi, last.invalidation_price, z.invalidation_price),
            2,
        )
        merged_bos_time = best.bos_candle_time
        merged_bos_tf = best.bos_timeframe or best.timeframe
        if last.bos_candle_time and z.bos_candle_time:
            try:
                _t_last = pd.Timestamp(last.bos_candle_time)
                _t_z = pd.Timestamp(z.bos_candle_time)
                if _t_z > _t_last:
                    merged_bos_time = z.bos_candle_time
                    merged_bos_tf = z.bos_timeframe or z.timeframe
                else:
                    merged_bos_time = last.bos_candle_time
                    merged_bos_tf = last.bos_timeframe or last.timeframe
            except Exception:
                pass
        elif z.bos_candle_time and not merged_bos_time:
            merged_bos_time = z.bos_candle_time
            merged_bos_tf = z.bos_timeframe or z.timeframe
        elif last.bos_candle_time and not merged_bos_time:
            merged_bos_time = last.bos_candle_time
            merged_bos_tf = last.bos_timeframe or last.timeframe
        merged_fvg = best.fvg_confirmation or last.fvg_confirmation or z.fvg_confirmation
        merged_sweep_ev = (
            getattr(best, "liquidity_sweep_evidence", None)
            or getattr(last, "liquidity_sweep_evidence", None)
            or getattr(z, "liquidity_sweep_evidence", None)
        )
        out[merged_idx] = Zone(
            zone_id=f"ZONE_{'+'.join(tfs)}_{last.zone_type}_{identity}", symbol=last.symbol,
            timeframe="+".join(tfs), zone_type=last.zone_type, price_low=round(lo, 2), price_high=round(hi, 2),
            midpoint=mid, score=best.score, grade=best.grade, origin_bos=best.origin_bos,
            fvg_confirmation=merged_fvg,
            displacement_strength=max(last.displacement_strength, z.displacement_strength),
            freshness=merged_fresh,
            touch_count=max(last.touch_count, z.touch_count),
            liquidity_sweep_status=last.liquidity_sweep_status if last.liquidity_sweep_status != "NONE" else z.liquidity_sweep_status,
            invalidation_price=merged_inval,
            is_valid=merged_valid,
            rejection_reason="" if merged_valid else (last.rejection_reason or z.rejection_reason or f"Zona {merged_fresh}"),
            ob_candle_time=best.ob_candle_time, bos_candle_time=merged_bos_time,
            bos_timeframe=merged_bos_tf,
            sl_recommended=min(last.sl_recommended, z.sl_recommended) if demand else max(last.sl_recommended, z.sl_recommended),
            tp1_recommended=best.tp1_recommended, tp2_recommended=best.tp2_recommended,
            score_breakdown=best.score_breakdown,
            merged_from=constituents,
            htf_alignment=best.htf_alignment, audit=best.audit + [f"Digabung dari {last.zone_id} & {z.zone_id}"],
            status=getattr(best, "status", "RADAR") if merged_valid else "INVALIDATED",
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
            zone_relevance=getattr(best, "zone_relevance", "RADAR"),
            zone_relevance_reason=getattr(best, "zone_relevance_reason", ""),
            liquidity_sweep_evidence=merged_sweep_ev,
            evidence={
                **dict(getattr(best, "evidence", None) or {}),
                "timeframe": "+".join(tfs),
                "candle_closed": True,
                "ob_candle_time": str(best.ob_candle_time) if best.ob_candle_time is not None else None,
                "bos_candle_time": str(merged_bos_time) if merged_bos_time is not None else None,
                "ob_range": [round(lo, 2), round(hi, 2)],
                "fvg_range": [round(float(merged_fvg["price_low"]), 2), round(float(merged_fvg["price_high"]), 2)] if isinstance(merged_fvg, dict) else None,
                "fvg_time": str(merged_fvg["time"]) if isinstance(merged_fvg, dict) and merged_fvg.get("time") is not None else None,
                "fvg_c3_time": str(merged_fvg["c3_time"]) if isinstance(merged_fvg, dict) and merged_fvg.get("c3_time") is not None else None,
                "fvg_status": merged_fvg.get("status", "NONE") if isinstance(merged_fvg, dict) else "NONE",
                "displacement_atr": round(float(max(last.displacement_strength, z.displacement_strength)), 2),
                "freshness": merged_fresh,
                "touch_count": max(last.touch_count, z.touch_count),
                "mitigation_status": merged_fresh,
                "sweep": merged_sweep_ev,
                "merged_from": list(constituents),
                "score_reason": getattr(best, "score_reason", ""),
            },
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
        diag = f"Konsolidasi horizontal: range sempit {low_range:.2f}-{high_range:.2f} (lebar {range_w:.2f} USD <= 2.6 ATR) dengan EMA20/EMA50 mendatar"
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
    """D1 (konteks), H4 (bias utama), H1 (konfirmasi), M30 (setup/pullback vs konflik serius)."""
    regimes = {tf: detect_market_regime(frames.get(tf)) for tf in ("D1", "H4", "H1", "M30")}
    r = {tf: regimes[tf]["bias"] for tf in ("D1", "H4", "H1", "M30")}
    r["regimes"] = {tf: regimes[tf]["regime"] for tf in ("D1", "H4", "H1", "M30")}
    r["h4_diagnostic"] = regimes["H4"].get("diagnostic_reason", regimes["H4"].get("detail", ""))
    r["h4_structure"] = regimes["H4"].get("active_structure", {})
    r["h4_detail"] = regimes["H4"]

    h4_info = evaluate_h4_primary_trend(frames.get("H4"), regime_h4=regimes["H4"])
    h1_conf = evaluate_h1_trend_confirmation(frames.get("H1"), h4_info, regime_h1=regimes["H1"])
    r["h1_regime"] = regimes["H1"]
    r["m30_regime"] = regimes["M30"]
    r["h1_confirmation"] = h1_conf.get("status", "UNCONFIRMED")
    r["h1_detail"] = h1_conf

    conflict = bool(h1_conf.get("has_conflict"))
    r["has_conflict"] = conflict

    df_m30 = frames.get("M30")
    cp_m30 = float(df_m30["Close"].iloc[-1]) if df_m30 is not None and len(df_m30) > 0 else 0.0
    m30_eval = (
        evaluate_m30_momentum(df_m30, r["H4"])
        if r["H4"] in ("BULLISH", "BEARISH")
        else {"status": "UNCONFIRMED" if r["H4"] == "NO_DATA" else "WAIT", "ok": False, "msg": "Menunggu bias arah H4"}
    )
    m30_cond = (
        assess_m30_condition(frames, h4_info, h1_conf, cp_m30, regime_m30=regimes["M30"])
        if r["H4"] in ("BULLISH", "BEARISH") and not conflict
        else {"status": m30_eval.get("status", "WAIT"), "condition": "CONFLICT" if conflict else "UNCONFIRMED", "ok": False, "msg": m30_eval.get("msg", "")}
    )

    m30_conflict = bool(m30_cond.get("status") == "CONFLICT")
    r["m30_conflict"] = m30_conflict
    r["m30_eval"] = m30_eval
    r["m30_condition"] = m30_cond
    r["m30_setup_condition"] = m30_cond.get("condition", "WAIT")
    r["m30_status"] = "CONFLICT" if m30_conflict else m30_cond.get("status", m30_eval.get("status", "WAIT"))

    h4_reg = regimes["H4"]["regime"]
    if h4_reg == "CHOPPY":
        r["has_conflict"] = True
        r["bias_utama"] = "WAIT (MARKET CHOPPY H4)"
        r["htf_status"] = "CONFLICT"
        r["detail"] = f"H4 Choppy: {regimes['H4']['detail']}"
    elif conflict:
        r["bias_utama"] = "WAIT (KONFLIK H4 vs H1)"
        r["htf_status"] = "CONFLICT"
        r["detail"] = h1_conf.get("reason") or f"Konflik H4 {r['H4']} vs H1 {r['H1']}"
    elif r["H4"] == "NO_DATA" or (r["H4"] == "NETRAL" and r["H1"] == "NO_DATA"):
        r["bias_utama"] = r["H4"] if r["H4"] in ("BULLISH", "BEARISH") else "NETRAL"
        r["htf_status"] = "UNCONFIRMED"
        r["detail"] = f"D1 {r['D1']}, H4 {r['H4']} ({h4_reg}), H1 {r['H1']}, M30 {r['M30']}"
    else:
        r["bias_utama"] = r["H4"] if r["H4"] in ("BULLISH", "BEARISH") else "NETRAL"
        r["htf_status"] = (
            "M30_CONFLICT"
            if r["m30_conflict"]
            else (
                "PULLBACK"
                if r["h1_confirmation"] == "PULLBACK" or r["m30_setup_condition"] == "PULLBACK"
                else ("ALIGNED" if r["H4"] in ("BULLISH", "BEARISH") else "WAIT")
            )
        )
        r["detail"] = f"D1 {r['D1']}, H4 {r['H4']} ({h4_reg}), H1 {r['H1']} ({r['h1_confirmation']}), M30 {r['M30']} ({r['m30_setup_condition']})"
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
                        "msg": f"Displacement M30 berlawanan ({bear_impulse:.2f} USD >= 1.6 ATR)",
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
                        "msg": f"Displacement M30 berlawanan ({bull_impulse:.2f} USD >= 1.6 ATR)",
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
    """Buat ID unik deterministik-waktu WIB untuk satu siklus snapshot."""
    return f"SNAP_{get_current_wib().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def _get_df_last_timestamp(df: Any) -> Optional[pd.Timestamp]:
    """Ekstrak timestamp candle terakhir dari DatetimeIndex atau kolom Time/time."""
    if df is None or not hasattr(df, "__len__") or len(df) == 0:
        return None
    ts = None
    try:
        if isinstance(df.index, pd.DatetimeIndex):
            ts = df.index[-1]
        elif hasattr(df, "columns"):
            for col in ("Time", "time", "timestamp"):
                if col in df.columns:
                    ts = pd.to_datetime(df[col].iloc[-1], errors="coerce")
                    break
        if ts is None:
            ts = pd.to_datetime(df.index[-1], errors="coerce")
    except Exception:
        return None
    if ts is None or pd.isna(ts):
        return None
    try:
        return pd.Timestamp(ts)
    except Exception:
        return None


def _align_timestamps_for_diff(ts_a: pd.Timestamp, ts_b: pd.Timestamp, b_tz_hint: str = "WIB") -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Samakan tz-awareness antara dua pd.Timestamp sebelum pengurangan."""
    a_tz = getattr(ts_a, "tzinfo", None) is not None
    b_tz = getattr(ts_b, "tzinfo", None) is not None
    if a_tz == b_tz:
        return ts_a, ts_b
    if a_tz and not b_tz:
        try:
            tz = "UTC" if str(b_tz_hint or "WIB").upper() == "UTC" else "Asia/Jakarta"
            return ts_a, ts_b.tz_localize(tz)
        except Exception:
            return ts_a, ts_b
    try:
        return ts_a, ts_b.tz_convert("UTC").tz_localize(None)
    except Exception:
        try:
            return ts_a, ts_b.tz_localize(None)
        except Exception:
            return ts_a, ts_b


def _separate_forming_candles(
    frames: Optional[Dict[str, pd.DataFrame]],
    server_time: Any = None,
) -> Tuple[Optional[Dict[str, pd.DataFrame]], Dict[str, Any]]:
    """
    Pisahkan candle yang masih berjalan (forming/unclosed) dari candle yang sudah close.
    Tidak mengubah dataframe sumber secara destruktif dan tetap mempertahankan flag attrs['is_closed']=False
    agar pemeriksaan downstream dapat memblokir konfirmasi dari candle yang belum selesai.
    """
    if not isinstance(frames, dict):
        return frames, {}

    closed_frames: Dict[str, pd.DataFrame] = {}
    forming_candles: Dict[str, Any] = {}
    modified = False
    for tf, df in frames.items():
        if df is None or not isinstance(df, pd.DataFrame) or len(df) == 0:
            closed_frames[tf] = df
            continue

        attrs = dict(getattr(df, "attrs", {}) or {})
        src_tz = str(attrs.get("time_tz", "WIB"))
        if "is_closed" in df.columns:
            unclosed_mask = df["is_closed"].astype(bool) == False  # noqa: E712
            if unclosed_mask.any():
                u_df = df[unclosed_mask]
                last_f = u_df.iloc[-1]
                forming_candles[tf] = {
                    "time": format_wib_timestamp(_get_df_last_timestamp(u_df), default_now=False, source_tz=src_tz),
                    "open": float(last_f["Open"]),
                    "high": float(last_f["High"]),
                    "low": float(last_f["Low"]),
                    "close": float(last_f["Close"]),
                    "is_closed": False,
                    "source": "is_closed_col",
                }
                clean = df[~unclosed_mask].copy()
                clean.attrs = dict(attrs)
                clean.attrs["had_unclosed_bar"] = True
                closed_frames[tf] = clean
                modified = True
                continue

        if "is_forming" in df.columns:
            forming_mask = df["is_forming"].astype(bool) == True  # noqa: E712
            if forming_mask.any():
                f_df = df[forming_mask]
                last_f = f_df.iloc[-1]
                forming_candles[tf] = {
                    "time": format_wib_timestamp(_get_df_last_timestamp(f_df), default_now=False, source_tz=src_tz),
                    "open": float(last_f["Open"]),
                    "high": float(last_f["High"]),
                    "low": float(last_f["Low"]),
                    "close": float(last_f["Close"]),
                    "is_closed": False,
                    "source": "is_forming_col",
                }
                clean = df[~forming_mask].copy()
                clean.attrs = dict(attrs)
                clean.attrs["had_unclosed_bar"] = True
                closed_frames[tf] = clean
                modified = True
                continue

        if attrs.get("is_closed") is False or attrs.get("has_unclosed_last_bar") is True:
            last_f = df.iloc[-1]
            forming_candles[tf] = {
                "time": format_wib_timestamp(_get_df_last_timestamp(df), default_now=False, source_tz=src_tz),
                "open": float(last_f["Open"]),
                "high": float(last_f["High"]),
                "low": float(last_f["Low"]),
                "close": float(last_f["Close"]),
                "is_closed": False,
                "source": "attrs_flag",
            }
            closed_frames[tf] = df
            continue

        if server_time is not None and tf in TF_MINUTES:
            last_ts = _get_df_last_timestamp(df)
            if last_ts is not None:
                try:
                    srv_ts, lts = _align_timestamps_for_diff(pd.Timestamp(server_time), last_ts, b_tz_hint=src_tz)
                    age_min = (srv_ts - lts).total_seconds() / 60.0
                    if 0.0 <= age_min < float(TF_MINUTES[tf]):
                        forming_candles[tf] = {
                            "time": format_wib_timestamp(last_ts, default_now=False, source_tz=src_tz),
                            "age_minutes": round(age_min, 2),
                            "is_closed": True,
                            "clock_in_bar": True,
                        }
                except Exception:
                    pass

        closed_frames[tf] = df

    return (closed_frames if modified else frames), forming_candles


def validate_data_freshness(
    frames: Optional[Dict[str, pd.DataFrame]],
    server_time: Any = None,
    tick: Any = None,
    tick_age: Optional[float] = None,
    market_status: Optional[str] = None,
    connection_status: Optional[str] = None,
    snapshot_status: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Validasi DATA FRESHNESS secara eksplisit: FRESH / STALE / INVALID / NO DATA.
    Membedakan MARKET CLOSED dari MT5 DISCONNECTED atau data korup/invalid.
    """
    mkt_status = str(market_status or "UNKNOWN").upper()
    conn_status = str(connection_status or "UNKNOWN").upper()
    last_candle_times: Dict[str, str] = {}
    ages_minutes: Dict[str, float] = {}
    stale_tfs: List[str] = []
    missing_tfs: List[str] = []
    unclosed_tfs: List[str] = []

    def _make_fresh_res(
        status: str,
        ok: bool,
        reason: str,
        is_closed: bool = False,
        tick_st: Optional[str] = None,
        candle_st: Optional[str] = None,
        conn_st_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        eff_conn = conn_st_override or conn_status
        eff_tick_st = tick_st or (
            "MARKET CLOSED" if mkt_status == "CLOSED"
            else ("INVALID" if snapshot_status == "DATA_INCOMPLETE" else status)
        )
        eff_candle_st = candle_st or status
        return {
            "status": status,
            "ok": ok,
            "reason": reason,
            "market_status": "OPEN" if (status == "FRESH" and mkt_status == "UNKNOWN") else ("CLOSED" if is_closed else mkt_status),
            "connection_status": eff_conn,
            "tick_status": eff_tick_st,
            "candle_status": eff_candle_st,
            "data_validation": "PASS" if ok else "FAIL",
            "last_candle_times": last_candle_times,
            "ages_minutes": ages_minutes,
            "stale_timeframes": stale_tfs,
            "missing_timeframes": missing_tfs,
            "unclosed_timeframes": unclosed_tfs,
            "has_unclosed_candle": bool(unclosed_tfs),
            "is_market_closed": is_closed,
        }

    if tick is not None:
        try:
            tb = getattr(tick, "bid", None) if not isinstance(tick, dict) else tick.get("bid")
            ta = getattr(tick, "ask", None) if not isinstance(tick, dict) else tick.get("ask")
            fb = float(tb) if tb is not None else 0.0
            fa = float(ta) if ta is not None else 0.0
            if not math.isfinite(fb) or not math.isfinite(fa) or fb < 0 or (fa > 0 and fa < fb):
                return _make_fresh_res("INVALID", False, "Tick Bid/Ask tidak valid", is_closed=(mkt_status == "CLOSED"), tick_st="INVALID")
            t_epoch = getattr(tick, "time", None) if not isinstance(tick, dict) else tick.get("time")
            if server_time is None and t_epoch is not None and float(t_epoch) > 0:
                server_time = pd.Timestamp(int(float(t_epoch)), unit="s", tz="UTC")
        except Exception:
            return _make_fresh_res("INVALID", False, "Tick Bid/Ask korup", is_closed=(mkt_status == "CLOSED"), tick_st="INVALID")

    if frames is None or not isinstance(frames, dict):
        missing_tfs.extend(["H4", "H1", "M30", "M5"])
        return _make_fresh_res(
            "NO DATA",
            False,
            "Data candle tidak tersedia (MT5 terputus atau feed kosong)",
            is_closed=(mkt_status == "CLOSED"),
            conn_st_override="DISCONNECTED" if conn_status == "UNKNOWN" else conn_status,
        )

    non_empty = [tf for tf in ("H4", "H1", "M30", "M5") if frames.get(tf) is not None and len(frames.get(tf)) > 0]
    missing_tfs.extend(tf for tf in ("H4", "H1", "M30", "M5") if tf not in non_empty)
    if not non_empty:
        return _make_fresh_res(
            "NO DATA",
            False,
            "Tidak ada timeframe utama (H4/H1/M30/M5) yang memiliki data candle",
            is_closed=(mkt_status == "CLOSED"),
        )

    # Validasi integritas OHLC, duplikasi/urutan timestamp, dan kumpulkan waktu candle terakhir dalam WIB
    for tf in ("D1", "H4", "H1", "M30", "M5"):
        df = frames.get(tf)
        if df is None or len(df) == 0:
            continue
        df_attrs = getattr(df, "attrs", {}) or {}
        if df_attrs.get("is_closed") is False or (
            ("is_closed" in df.columns and (df["is_closed"].astype(bool) == False).any())  # noqa: E712
            or ("is_forming" in df.columns and (df["is_forming"].astype(bool) == True).any())  # noqa: E712
        ):
            unclosed_tfs.append(tf)
        src_tz = str(df_attrs.get("time_tz", "WIB"))
        lts = _get_df_last_timestamp(df)
        if lts is not None:
            last_candle_times[tf] = format_wib_timestamp(lts, default_now=False, source_tz=src_tz) or str(lts)

        ts_series = None
        if isinstance(df.index, pd.DatetimeIndex):
            ts_series = df.index
        elif "Time" in df.columns:
            ts_series = pd.to_datetime(df["Time"], errors="coerce")
        elif "time" in df.columns:
            ts_series = pd.to_datetime(df["time"], errors="coerce")

        if ts_series is not None and len(ts_series) > 1:
            if pd.isna(ts_series).any():
                return _make_fresh_res("INVALID", False, f"Timestamp NaT terdeteksi pada {tf}", is_closed=(mkt_status == "CLOSED"), candle_st="INVALID")
            if ts_series.duplicated().any():
                return _make_fresh_res("INVALID", False, f"Candle duplikat terdeteksi pada {tf}", is_closed=(mkt_status == "CLOSED"), candle_st="INVALID")
            if not ts_series.is_monotonic_increasing:
                return _make_fresh_res("INVALID", False, f"Urutan timestamp candle {tf} tidak berurutan (non-monotonic)", is_closed=(mkt_status == "CLOSED"), candle_st="INVALID")

        for col in ("Open", "High", "Low", "Close"):
            if col not in df.columns:
                return _make_fresh_res("INVALID", False, f"Kolom {col} hilang pada data {tf}", is_closed=(mkt_status == "CLOSED"), candle_st="INVALID")
        tail = df.iloc[-min(len(df), 5):]
        if tail[["Open", "High", "Low", "Close"]].isna().any().any():
            return _make_fresh_res("INVALID", False, f"Nilai NaN terdeteksi pada OHLC {tf}", is_closed=(mkt_status == "CLOSED"), candle_st="INVALID")
        if (tail["High"] < tail["Low"]).any() or (tail["Close"] <= 0).any():
            return _make_fresh_res("INVALID", False, f"OHLC tidak konsisten / harga <= 0 pada {tf}", is_closed=(mkt_status == "CLOSED"), candle_st="INVALID")

    if snapshot_status == "DATA_INCOMPLETE":
        return _make_fresh_res(
            "INVALID",
            False,
            "Snapshot harga MT5 tidak lengkap atau tidak valid (DATA_INCOMPLETE)",
            is_closed=(mkt_status == "CLOSED"),
            tick_st="INVALID",
            candle_st="FRESH" if non_empty else "NO DATA",
        )

    # Hitung usia candle terhadap server_time
    stale_msg = None
    tick_stale = False
    candle_stale = False
    if server_time is not None:
        try:
            srv_ts = pd.Timestamp(server_time)
            if not pd.isna(srv_ts):
                for tf in ("M5", "M30", "H1", "H4"):
                    df = frames.get(tf)
                    if df is None or len(df) == 0:
                        continue
                    lts = _get_df_last_timestamp(df)
                    if lts is None:
                        continue
                    src_tz = str((getattr(df, "attrs", {}) or {}).get("time_tz", "WIB"))
                    s_al, l_al = _align_timestamps_for_diff(srv_ts, lts, b_tz_hint=src_tz)
                    age = (s_al - l_al).total_seconds() / 60.0
                    ages_minutes[tf] = round(age, 2)
                    if age < -TF_MINUTES[tf]:
                        stale_tfs.append(tf)
                        tick_stale = True
                        stale_msg = stale_msg or f"Tick lebih lama dari candle {tf} terakhir ({abs(age):.0f} menit; stale tick)"
                    elif age > TF_MINUTES[tf] * 3 + 5:
                        stale_tfs.append(tf)
                        candle_stale = True
                        stale_msg = stale_msg or f"Data {tf} terlambat ({age:.0f} menit sejak candle terakhir; market tutup/feed tertinggal)"
        except Exception:
            pass

    if mkt_status == "CLOSED":
        return _make_fresh_res(
            "STALE",
            False,
            stale_msg or "Market sedang tutup (data sesi penutupan terakhir; bukan koneksi MT5 terputus)",
            is_closed=True,
            tick_st="MARKET CLOSED",
            candle_st="HISTORICAL (CLOSED)",
        )

    if stale_msg:
        return _make_fresh_res(
            "STALE",
            False,
            stale_msg,
            is_closed=False,
            tick_st="STALE" if tick_stale else "FRESH",
            candle_st="STALE" if candle_stale else "FRESH",
        )

    return _make_fresh_res("FRESH", True, "Data candle dan snapshot pasar valid & aktual", is_closed=False, tick_st="FRESH", candle_st="FRESH")


def _live_context(symbol: str = SYMBOL, tick: Any = None) -> Dict[str, Any]:
    """Harga bid/ask/mid snapshot tunggal, spread, waktu server, dan status blackout berita.

    tick: objek tick MT5 yang sudah diambil pemanggil (dipakai apa adanya, TANPA memanggil
    mt5.symbol_info_tick lagi). Jika None, satu tick baru diambil dari MT5.
    """
    ctx = {"price": None, "spread": None, "server_time": None, "news": None, "news_error": None, "snapshot": None}
    try:
        if tick is None:
            import MetaTrader5 as mt5
            with MT5_IPC_LOCK:
                tick = mt5.symbol_info_tick(symbol)
        if tick:
            b_val = float(getattr(tick, "bid", 0.0) or 0.0)
            a_val = float(getattr(tick, "ask", 0.0) or 0.0)
            if math.isfinite(b_val) and math.isfinite(a_val) and b_val > 0 and a_val >= b_val:
                u_spec = get_symbol_unit_spec(symbol=symbol)
                d_px = u_spec["digits"]
                bid = round(b_val, d_px)
                ask = round(a_val, d_px)
                mid = round((bid + ask) / 2.0, d_px)
                spread = round(ask - bid, d_px)
                if hasattr(tick, "time") and tick.time > 0:
                    tick_time = format_wib_timestamp(tick.time, default_now=True)
                    tick_age = round(max(0.0, time.time() - tick.time), 1)
                    srv_ts = pd.Timestamp(int(tick.time), unit="s", tz="UTC")
                else:
                    tick_time = format_wib_timestamp(default_now=True)
                    tick_age = 0.0
                    srv_ts = None
                spread_lbl = format_price_distance(spread, spec=u_spec, include_mt5_points=True)
                snap = {
                    "snapshot_id": _generate_snapshot_id(),
                    "symbol": symbol,
                    "digits": d_px,
                    "point": u_spec["point"],
                    "trade_tick_size": u_spec["trade_tick_size"],
                    "symbol_spec": u_spec,
                    "bid": bid,
                    "ask": ask,
                    "mid": mid,
                    "spread": spread,
                    "spread_mt5_points": price_distance_to_mt5_points(spread, spec=u_spec),
                    "snapshot_price": mid,
                    "snapshot_time": tick_time,
                    "tick_time": tick_time,
                    "tick_time_msc": int(getattr(tick, "time_msc", 0) or 0),
                    "tick_bid_raw": b_val,
                    "tick_ask_raw": a_val,
                    "tick_age": tick_age,
                    "server_time": srv_ts,
                    "price_source": f"MT5 Live Tick (Bid: {bid:.{d_px}f} | Ask: {ask:.{d_px}f} | Mid: {mid:.{d_px}f} | Spread: {spread_lbl})",
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
    market_status: Optional[str] = None,
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
    req_market_status = str(market_status or "").upper() or None

    if frames is None:
        frames = _load_frames(ttl=0.0 if force_refresh_frames else FRAMES_CACHE_TTL)
        if frames is None:
            fresh_info = validate_data_freshness(
                None,
                server_time=server_time,
                market_status=req_market_status or "UNKNOWN",
                connection_status="DISCONNECTED",
                snapshot_status="DATA_INCOMPLETE",
            )
            return {
                "snapshot_id": snapshot_id,
                "symbol": symbol,
                "bid": None,
                "ask": None,
                "mid": None,
                "spread": spread,
                "current_price": None,
                "snapshot_price": None,
                "snapshot_time": format_wib_timestamp(default_now=True),
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
                "error_reason": "MT5 tidak aktif; koneksi terputus dan data tidak dapat diambil",
                "data_freshness": fresh_info["status"],
                "data_freshness_info": fresh_info,
                "market_status": fresh_info["market_status"],
                "connection_status": "DISCONNECTED",
                "last_candle_times": {},
                "forming_candles": {},
            }

    closed_frames, forming_candles = _separate_forming_candles(frames, server_time=server_time)
    frames = closed_frames if isinstance(closed_frames, dict) else frames

    detected_mkt_status = req_market_status or ""
    detected_conn_status = "CONNECTED" if _frames_src == "mt5_fetched" else "CALLER_SUPPLIED"
    if not detected_mkt_status:
        try:
            import MetaTrader5 as mt5
            with MT5_IPC_LOCK:
                term_info = getattr(mt5, "terminal_info", lambda: None)()
                sym_info = getattr(mt5, "symbol_info", lambda s: None)(symbol) if term_info is not None else None
            if term_info is not None:
                detected_conn_status = "CONNECTED"
                if sym_info is not None and getattr(sym_info, "trade_mode", None) == 0:
                    detected_mkt_status = "CLOSED"
                if not detected_mkt_status:
                    now_utc = datetime.now(timezone.utc)
                    wd, hr = now_utc.weekday(), now_utc.hour
                    if wd == 5 or (wd == 4 and hr >= 22) or (wd == 6 and hr < 21):
                        detected_mkt_status = "CLOSED"
        except Exception:
            pass

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
            fresh_info = validate_data_freshness(
                frames,
                server_time=server_time,
                market_status=detected_mkt_status or "UNKNOWN",
                connection_status=detected_conn_status,
                snapshot_status="DATA_INCOMPLETE",
            )
            return {
                "snapshot_id": snapshot_id,
                "symbol": symbol,
                "bid": None,
                "ask": None,
                "mid": None,
                "spread": spread,
                "current_price": None,
                "snapshot_price": None,
                "snapshot_time": format_wib_timestamp(default_now=True),
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
                "data_freshness": fresh_info["status"],
                "data_freshness_info": fresh_info,
                "market_status": fresh_info["market_status"],
                "connection_status": detected_conn_status,
                "last_candle_times": fresh_info["last_candle_times"],
                "forming_candles": forming_candles,
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
                if detected_mkt_status == "CLOSED":
                    ref_c = frames.get("M5") if frames.get("M5") is not None and len(frames["M5"]) else frames.get("H1")
                    last_c = float(ref_c["Close"].iloc[-1]) if ref_c is not None and len(ref_c) else None
                    if last_c is not None and math.isfinite(last_c) and last_c > 0:
                        current_price = round(last_c, 2)
                        price_source = f"MT5 HISTORICAL (MARKET CLOSED) — Last Close: {current_price:.2f}" + (f" (Last Tick: {mid:.2f})" if mid else "")
                    else:
                        current_price = mid
                else:
                    current_price = mid
        elif current_price is None:
            if detected_mkt_status == "CLOSED":
                ref_c = frames.get("M5") if frames.get("M5") is not None and len(frames["M5"]) else frames.get("H1")
                last_c = float(ref_c["Close"].iloc[-1]) if ref_c is not None and len(ref_c) else None
                if last_c is not None and math.isfinite(last_c) and last_c > 0:
                    snap_price = round(last_c, 2)
                    fresh_info = validate_data_freshness(
                        frames,
                        server_time=server_time,
                        market_status="CLOSED",
                        connection_status=detected_conn_status,
                        snapshot_status="MARKET_CLOSED",
                    )
                    return {
                        "snapshot_id": snapshot_id,
                        "symbol": symbol,
                        "bid": None,
                        "ask": None,
                        "mid": snap_price,
                        "spread": spread,
                        "current_price": snap_price,
                        "snapshot_price": snap_price,
                        "snapshot_time": format_wib_timestamp(default_now=True),
                        "tick_time": None,
                        "tick_age": None,
                        "server_time": server_time,
                        "price_source": f"Market Closed (Last Close: {snap_price:.2f})",
                        "snapshot_status": "MARKET_CLOSED",
                        "frames": frames,
                        "atr_diagnostics": extract_atr_diagnostics(frames, preferred_tf="H1"),
                        "news": news_blackout if news_blackout is not None else lc.get("news"),
                        "news_error": lc.get("news_error"),
                        "data_ok": False,
                        "error_reason": "Market sedang tutup (data penutupan sesi terakhir; bukan koneksi MT5 terputus)",
                        "data_freshness": fresh_info["status"],
                        "data_freshness_info": fresh_info,
                        "market_status": "CLOSED",
                        "connection_status": detected_conn_status,
                        "last_candle_times": fresh_info["last_candle_times"],
                        "forming_candles": forming_candles,
                    }
            fresh_info = validate_data_freshness(
                frames,
                server_time=server_time,
                market_status=detected_mkt_status or "UNKNOWN",
                connection_status="DISCONNECTED" if _tick_src == "mt5_fetched" else detected_conn_status,
                snapshot_status="DATA_INCOMPLETE",
            )
            return {
                "snapshot_id": snapshot_id,
                "symbol": symbol,
                "bid": None,
                "ask": None,
                "mid": None,
                "spread": spread,
                "current_price": None,
                "snapshot_price": None,
                "snapshot_time": format_wib_timestamp(default_now=True),
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
                "data_freshness": fresh_info["status"],
                "data_freshness_info": fresh_info,
                "market_status": fresh_info["market_status"],
                "connection_status": fresh_info["connection_status"],
                "last_candle_times": fresh_info["last_candle_times"],
                "forming_candles": forming_candles,
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
        fresh_info = validate_data_freshness(
            frames,
            server_time=server_time,
            market_status=detected_mkt_status or "UNKNOWN",
            connection_status=detected_conn_status,
            snapshot_status="DATA_INCOMPLETE",
        )
        return {
            "snapshot_id": snapshot_id,
            "symbol": symbol,
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "spread": spread,
            "current_price": None,
            "snapshot_price": None,
            "snapshot_time": format_wib_timestamp(default_now=True),
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
            "data_freshness": "INVALID",
            "data_freshness_info": fresh_info,
            "market_status": fresh_info["market_status"],
            "connection_status": detected_conn_status,
            "last_candle_times": fresh_info["last_candle_times"],
            "forming_candles": forming_candles,
        }

    snap_price = round(float(current_price), 2)
    snap_time = tick_time or format_wib_timestamp(server_time, default_now=True)
    if price_source is None:
        price_source = f"Candle Close Snapshot ({snap_price:.2f})"
    atr_diag = extract_atr_diagnostics(frames, preferred_tf="H1")
    fresh_info = validate_data_freshness(
        frames,
        server_time=server_time,
        tick=tick,
        tick_age=tick_age,
        market_status=detected_mkt_status or "OPEN",
        connection_status=detected_conn_status,
        snapshot_status=snapshot_status,
    )
    if detected_mkt_status == "CLOSED":
        snapshot_status = "MARKET_CLOSED"
        error_reason = error_reason or fresh_info["reason"]
        ref_c = frames.get("M5") if frames.get("M5") is not None and len(frames["M5"]) else frames.get("H1")
        last_c = float(ref_c["Close"].iloc[-1]) if ref_c is not None and len(ref_c) else None
        if last_c is not None and math.isfinite(last_c) and last_c > 0:
            current_price = round(last_c, 2)
            snap_price = current_price
            price_source = f"MT5 HISTORICAL (MARKET CLOSED) — Last Close: {current_price:.2f}" + (f" (Last Tick: {mid:.2f})" if mid else "")

    if use_live and "M5" not in forming_candles and detected_mkt_status != "CLOSED":
        try:
            from modules.mt5_conn import ambil_candle_berjalan
            tf_m5 = TIMEFRAMES.get("M5", (None,))[0]
            if tf_m5 is not None:
                rc = ambil_candle_berjalan(tf_m5)
                if rc is not None and int(rc["time"]) > 0:
                    forming_candles["M5"] = {
                        "time": format_wib_timestamp(int(rc["time"]), default_now=False, source_tz="UTC"),
                        "open": float(rc["open"]),
                        "high": float(rc["high"]),
                        "low": float(rc["low"]),
                        "close": float(rc["close"]),
                        "is_closed": False,
                        "source": "mt5_pos0",
                    }
        except Exception:
            pass

    return {
        "snapshot_id": snapshot_id,
        "symbol": symbol,
        "live": bool(use_live),
        "bid": bid,
        "ask": ask,
        "mid": mid,
        "spread": spread,
        "current_price": snap_price,
        "snapshot_price": snap_price,
        "snapshot_time": snap_time,
        "tick_time": tick_time,
        "tick_time_msc": lc_snap.get("tick_time_msc") if use_live and isinstance(lc_snap, dict) else None,
        "tick_bid_raw": lc_snap.get("tick_bid_raw") if use_live and isinstance(lc_snap, dict) else None,
        "tick_ask_raw": lc_snap.get("tick_ask_raw") if use_live and isinstance(lc_snap, dict) else None,
        "tick_age": tick_age,
        "server_time": server_time,
        "price_source": price_source,
        "snapshot_status": snapshot_status,
        "frames": frames,
        "atr_diagnostics": atr_diag,
        "news": news_blackout,
        "news_error": news_err,
        "data_ok": True if detected_mkt_status != "CLOSED" else False,
        "error_reason": error_reason,
        "data_freshness": fresh_info["status"],
        "data_freshness_info": fresh_info,
        "market_status": fresh_info["market_status"],
        "connection_status": detected_conn_status,
        "last_candle_times": fresh_info["last_candle_times"],
        "forming_candles": forming_candles,
        "last_closed_m5": fresh_info["last_candle_times"].get("M5", ""),
        "current_candle_m5": (forming_candles.get("M5") or {}).get("time", ""),
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
    try:
        srv_ts = pd.Timestamp(server_time)
    except Exception:
        return None
    if pd.isna(srv_ts):
        return None
    for tf in ("M5", "M30", "H1", "H4"):
        df = frames.get(tf)
        if df is None or len(df) == 0:
            continue
        last_ts = _get_df_last_timestamp(df)
        if last_ts is None:
            continue
        s_al, l_al = _align_timestamps_for_diff(srv_ts, last_ts, b_tz_hint=str((getattr(df, "attrs", {}) or {}).get("time_tz", "WIB")))
        age = (s_al - l_al).total_seconds() / 60.0
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


def _zone_origin_timeframe(z: Union[Zone, Dict[str, Any]], frames: Optional[Dict[str, pd.DataFrame]] = None) -> Optional[str]:
    """Menentukan timeframe asal zona (mis. H4 untuk H4+H1) agar invalidasi hanya memakai candle close TF asal."""
    zd = z.to_dict() if hasattr(z, "to_dict") else (z if isinstance(z, dict) else {})
    raw_tf = str(zd.get("bos_timeframe") or zd.get("timeframe") or "").upper()
    parts = [p.strip() for p in re.split(r"[+/, ]+", raw_tf) if p.strip()]
    ordered = sorted(parts, key=lambda t: TF_MINUTES.get(t, 0), reverse=True)
    if isinstance(frames, dict):
        for tf_cand in ordered:
            df_c = frames.get(tf_cand)
            if df_c is not None and len(df_c) > 0:
                return tf_cand
    return ordered[0] if ordered else None


def _smc_zone_invalid_key(z: Union[Zone, Dict[str, Any]]) -> str:
    """Kunci identitas stabil zona SMC untuk persistensi status INVALID — BROKEN antar-scan."""
    zd = z.to_dict() if hasattr(z, "to_dict") else (z if isinstance(z, dict) else {})
    zid = str(zd.get("zone_id") or "").strip()
    if zid and not zid.startswith("ZONE_UNKNOWN"):
        return zid
    tf = str(zd.get("timeframe") or "").upper()
    zt = str(zd.get("zone_type") or "").upper()
    lo = float(zd.get("price_low", zd.get("low", 0.0)) or 0.0)
    hi = float(zd.get("price_high", zd.get("high", 0.0)) or 0.0)
    return f"{zd.get('symbol', SYMBOL)}|{tf}|{zt}|{lo:.2f}|{hi:.2f}"


def _smc_zone_geo_key(zd: Dict[str, Any]) -> str:
    """Kunci geometri stabil (TF|tipe|low|high) agar regenerasi zone_id tidak melepas status INVALID."""
    tf = str(zd.get("timeframe") or "").upper()
    zt = str(zd.get("zone_type") or zd.get("type") or "").upper()
    lo = float(zd.get("price_low", zd.get("low", 0.0)) or 0.0)
    hi = float(zd.get("price_high", zd.get("high", 0.0)) or 0.0)
    return f"GEO|{zd.get('symbol', SYMBOL)}|{tf}|{zt}|{lo:.2f}|{hi:.2f}"


_ZONE_ID_EPOCH_RE = re.compile(r"_(\d{9,11})$")
_ZONE_ID_ISO_RE = re.compile(r"(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})")


def _resolve_smc_zone_origin_time(
    zd: Dict[str, Any],
    work_df: Optional[pd.DataFrame] = None,
) -> Tuple[Optional[pd.Timestamp], bool]:
    """
    Resolve timestamp pembentukan/BOS zona SMC dalam timezone yang sama dengan `work_df.index`
    (atau naive UTC bila `work_df` tidak diberikan).
    Return: (origin_ts, is_bos_candle_time). Jika tidak dapat ditentukan, kembalikan (None, False)
    agar evaluasi invalidasi TIDAK pernah memindai candle historis sebelum zona terbentuk.
    """
    df_tz = work_df.index.tz if (work_df is not None and isinstance(work_df.index, pd.DatetimeIndex)) else None
    tz_hint = str((getattr(work_df, "attrs", {}) or {}).get("time_tz", "UTC") if work_df is not None else "UTC").upper()

    def _align_ts(ts_val: pd.Timestamp, naive_source_tz: str) -> Optional[pd.Timestamp]:
        if ts_val is None or pd.isna(ts_val):
            return None
        src_tz = "UTC" if str(naive_source_tz or "UTC").upper() == "UTC" else "Asia/Jakarta"
        tgt_tz = "UTC" if tz_hint == "UTC" else "Asia/Jakarta"
        if df_tz is not None:
            if ts_val.tzinfo is None:
                return ts_val.tz_localize(src_tz).tz_convert(df_tz)
            return ts_val.tz_convert(df_tz)
        if ts_val.tzinfo is not None:
            return ts_val.tz_convert(tgt_tz).tz_localize(None)
        if src_tz != tgt_tz:
            return ts_val.tz_localize(src_tz).tz_convert(tgt_tz).tz_localize(None)
        return ts_val

    def _parse_candidate(raw_val: Any, default_tz: str = "UTC") -> Optional[pd.Timestamp]:
        if raw_val is None or raw_val == "":
            return None
        if isinstance(raw_val, (int, float)) and math.isfinite(float(raw_val)) and float(raw_val) > 1e8:
            ts_ep = pd.Timestamp(int(float(raw_val)), unit="s", tz="UTC")
            return _align_ts(ts_ep, "UTC")
        if isinstance(raw_val, str):
            raw_s = raw_val.strip()
            if not raw_s or raw_s.lower() in ("none", "nan", "nat", "-"):
                return None
            has_wib = "WIB" in raw_s.upper()
            cleaned = raw_s.replace(" WIB", "").replace(" wib", "").strip()
            ts_str = pd.Timestamp(cleaned)
            return _align_ts(ts_str, "WIB" if has_wib else default_tz)
        ts_obj = pd.Timestamp(raw_val)
        return _align_ts(ts_obj, default_tz)

    # 1. Timestamp BOS / OB / formation eksplisit (menggunakan timezone frame asal)
    orig_bos = zd.get("origin_bos") if isinstance(zd.get("origin_bos"), dict) else {}
    form_tz = str(zd.get("formation_time_tz") or tz_hint or "UTC").upper()
    candidates = [
        (zd.get("bos_candle_time"), True, tz_hint),
        (orig_bos.get("time"), True, tz_hint),
        (zd.get("ob_candle_time"), False, tz_hint),
        (zd.get("formation_time"), False, form_tz),
    ]
    for raw_v, is_bos_flag, src_tz_c in candidates:
        try:
            aligned = _parse_candidate(raw_v, default_tz=src_tz_c)
            if aligned is not None:
                return aligned, is_bos_flag
        except Exception:
            continue

    # 2. Timestamp yang tertanam pada zone_id (ZONE_<TF>_<TYPE>_<epoch> atau TF_PB_FVG_<TF>_<DIR>_<ISO>)
    zid = str(zd.get("zone_id") or "").strip()
    if zid:
        m_ep = _ZONE_ID_EPOCH_RE.search(zid)
        if m_ep:
            try:
                aligned = _parse_candidate(int(m_ep.group(1)), default_tz="UTC")
                if aligned is not None:
                    return aligned, False
            except Exception:
                pass
        m_iso = _ZONE_ID_ISO_RE.search(zid)
        if m_iso:
            try:
                aligned = _parse_candidate(m_iso.group(1), default_tz=tz_hint)
                if aligned is not None:
                    return aligned, False
            except Exception:
                pass

    # 3. Fallback konservatif: waktu deteksi pertama (WIB -> timezone frame)
    for det_key in ("first_detected", "created_at"):
        try:
            aligned = _parse_candidate(zd.get(det_key), default_tz="WIB")
            if aligned is not None:
                return aligned, False
        except Exception:
            continue

    return None, False


def _parse_ts_utc_naive(raw_val: Any) -> Optional[pd.Timestamp]:
    """Parse timestamp (termasuk akhiran WIB / tz-aware) menjadi UTC naive."""
    if not raw_val:
        return None
    try:
        raw_s = str(raw_val).strip()
        if not raw_s or raw_s.lower() in ("none", "nan", "nat", "-"):
            return None
        is_wib = "WIB" in raw_s.upper()
        cleaned = raw_s.replace(" WIB", "").replace(" wib", "").strip()
        ts = pd.Timestamp(cleaned)
        if pd.isna(ts):
            return None
        if ts.tzinfo is not None:
            return ts.tz_convert("UTC").tz_localize(None)
        if is_wib:
            return ts.tz_localize("Asia/Jakarta").tz_convert("UTC").tz_localize(None)
        return ts
    except Exception:
        return None


def _is_impossible_pre_creation_invalidation(
    zone_or_rec: Dict[str, Any],
    inv_rec: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    True jika catatan invalidasi memiliki break_time <= waktu pembentukan/OB/BOS zona
    (bukti artefak pemindaian candle historis pra-pembentukan zona, bukan invalidasi nyata).
    """
    target = inv_rec if isinstance(inv_rec, dict) else zone_or_rec
    if not isinstance(target, dict):
        return False
    bt_utc = _parse_ts_utc_naive(target.get("break_time"))
    if bt_utc is None or pd.isna(bt_utc):
        return False

    merged_meta = dict(target)
    if isinstance(zone_or_rec, dict):
        for k_m in ("zone_id", "bos_candle_time", "ob_candle_time", "formation_time", "origin_bos", "first_detected", "created_at"):
            if zone_or_rec.get(k_m) and not merged_meta.get(k_m):
                merged_meta[k_m] = zone_or_rec.get(k_m)
    origin_utc, _ = _resolve_smc_zone_origin_time(merged_meta, work_df=None)
    if origin_utc is None or pd.isna(origin_utc):
        return False
    return bool(bt_utc <= origin_utc)


def _is_breach_invalidation_valid(
    zone_or_rec: Dict[str, Any],
    inv_rec: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Verifikasi integritas catatan invalidasi: tolak jika artefak pra-pembentukan ATAU jika
    harga/close tercatat belum benar-benar menembus batas invalidasi (distal untuk struktur, SL untuk setup).
    """
    if _is_impossible_pre_creation_invalidation(zone_or_rec, inv_rec):
        return False
    target = inv_rec if isinstance(inv_rec, dict) else zone_or_rec
    if not isinstance(target, dict):
        return True
    src = zone_or_rec if isinstance(zone_or_rec, dict) and (zone_or_rec.get("price_low") or zone_or_rec.get("low")) else target
    z_type = str(src.get("zone_type") or target.get("zone_type") or "").upper()
    inv_type = str(target.get("invalidation_type") or target.get("invalid_label") or "").upper()
    if inv_type == INVALIDATION_SETUP_SL_BREACHED:
        b_px = target.get("break_price")
        sl_ref = float(
            target.get("setup_sl")
            or target.get("setup_invalidation_price")
            or src.get("setup_sl")
            or src.get("setup_invalidation_price")
            or src.get("sl")
            or 0.0
        )
        if isinstance(b_px, (int, float)) and math.isfinite(float(b_px)) and float(b_px) > 0 and sl_ref > 0:
            if z_type == "DEMAND" and float(b_px) > sl_ref:
                return False
            if z_type == "SUPPLY" and float(b_px) < sl_ref:
                return False
        return True
    b_close = target.get("break_close")
    if not isinstance(b_close, (int, float)) or not math.isfinite(float(b_close)) or float(b_close) <= 0:
        return True
    lo = float(src.get("price_low", src.get("low", target.get("price_low", 0.0))) or 0.0)
    hi = float(src.get("price_high", src.get("high", target.get("price_high", 0.0))) or 0.0)
    raw_inv = float(src.get("invalidation_price") or target.get("invalidation_price") or 0.0)
    if z_type == "DEMAND":
        eff_inv = min(lo, raw_inv) if (lo > 0 and raw_inv > 0) else (lo or raw_inv)
        if eff_inv > 0 and float(b_close) >= eff_inv:
            return False
    elif z_type == "SUPPLY":
        eff_inv = max(hi, raw_inv) if (hi > 0 and raw_inv > 0) else (hi or raw_inv)
        if eff_inv > 0 and float(b_close) <= eff_inv:
            return False
    return True


def resolve_smc_structure_invalidation_price(zone_or_setup: Union[Zone, Dict[str, Any]]) -> float:
    """
    Sumber tunggal `STRUCTURE_INV` (batas invalidasi struktur zona SMC — terpisah dari SETUP_SL):
    - DEMAND/BUY  : batas distal bawah zona (min(low, invalidation_price))
    - SUPPLY/SELL : batas distal atas zona (max(high, invalidation_price))
    """
    zd = zone_or_setup.to_dict() if hasattr(zone_or_setup, "to_dict") else (zone_or_setup if isinstance(zone_or_setup, dict) else {})
    if not zd:
        return 0.0
    z_type = str(zd.get("zone_type") or zd.get("type") or "").upper()
    direction = str(zd.get("direction") or zd.get("arah") or "").upper()
    if z_type not in ("DEMAND", "SUPPLY"):
        if direction == "BUY":
            z_type = "DEMAND"
        elif direction == "SELL":
            z_type = "SUPPLY"
        else:
            return 0.0
    demand = (z_type == "DEMAND")
    lo = float(zd.get("price_low", zd.get("low", zd.get("zone_low", 0.0))) or 0.0)
    hi = float(zd.get("price_high", zd.get("high", zd.get("zone_high", 0.0))) or 0.0)
    raw = float(
        zd.get("structure_invalidation")
        or zd.get("structural_invalidation")
        or zd.get("invalidation_price")
        or (lo if demand else hi)
        or 0.0
    )
    if demand and lo > 0 and raw > 0:
        return round(min(lo, raw), 2)
    if (not demand) and hi > 0 and raw > 0:
        return round(max(hi, raw), 2)
    return round(raw, 2) if raw > 0 else 0.0


def resolve_smc_setup_invalidation_price(
    zone_or_setup: Union[Zone, Dict[str, Any]],
    frames: Optional[Dict[str, pd.DataFrame]] = None,
    spread: float = 0.0,
) -> float:
    """
    Sumber tunggal `SETUP_SL` untuk setup aktif SMC (berbasis SL/buffer existing — terpisah dari STRUCTURE_INV):
    - DEMAND/BUY  : batas bawah SL entry (<= distal edge)
    - SUPPLY/SELL : batas atas SL entry (>= distal edge)
    Tidak pernah menyamakan batas struktur zona mentah sebagai SL tanpa dasar perhitungan buffer/SL existing.
    """
    zd = zone_or_setup.to_dict() if hasattr(zone_or_setup, "to_dict") else (zone_or_setup if isinstance(zone_or_setup, dict) else {})
    if not zd:
        return 0.0
    z_type = str(zd.get("zone_type") or zd.get("type") or "").upper()
    direction = str(zd.get("direction") or zd.get("arah") or "").upper()
    if z_type not in ("DEMAND", "SUPPLY"):
        if direction == "BUY":
            z_type = "DEMAND"
        elif direction == "SELL":
            z_type = "SUPPLY"
        else:
            return 0.0
    demand = (z_type == "DEMAND")
    lo = float(zd.get("price_low", zd.get("low", zd.get("zone_low", 0.0))) or 0.0)
    hi = float(zd.get("price_high", zd.get("high", zd.get("zone_high", 0.0))) or 0.0)
    struct_inv = resolve_smc_structure_invalidation_price(zd)

    def _side_ok(val: float) -> bool:
        if not (isinstance(val, (int, float)) and math.isfinite(val) and val > 0):
            return False
        if demand and lo > 0:
            return val <= lo + 1e-6
        if (not demand) and hi > 0:
            return val >= hi - 1e-6
        return True

    # 0. Sumber tunggal SetupSL tersimpan (frozen/active setup version) agar tidak pernah dihitung ulang dari live tick
    zid = str(zd.get("zone_id") or "").strip()
    if zid:
        try:
            from modules.utils import resolve_stored_setup_sl
            z_tf = str(zd.get("timeframe") or "").strip().upper() or None
            z_ver = zd.get("setup_version")
            stored_sl = resolve_stored_setup_sl(
                "SMC",
                zid,
                timeframe=z_tf,
                setup_version=int(z_ver) if isinstance(z_ver, int) and z_ver > 0 else None,
            )
            if _side_ok(stored_sl):
                return round(stored_sl, 2)
        except Exception:
            pass

    # 1. Explicit setup SL / setup_invalidation_price (plan / pending_plan)
    pending_p = zd.get("pending_plan") if isinstance(zd.get("pending_plan"), dict) else {}
    for raw_c in (
        zd.get("setup_sl"),
        zd.get("setup_invalidation_price"),
        zd.get("sl"),
        pending_p.get("sl"),
    ):
        try:
            v = float(raw_c or 0.0)
        except (TypeError, ValueError):
            v = 0.0
        if _side_ok(v):
            return round(v, 2)

    # 2. Compute structural SL using existing buffer logic (ATR/spread)
    try:
        z_obj = zone_or_setup if isinstance(zone_or_setup, Zone) else _zone_from_dict(zd)
        sl_comp = compute_structural_sl(z_obj, frames, float(spread or 0.0))
        if _side_ok(sl_comp):
            return round(sl_comp, 2)
    except Exception:
        pass

    # 3. Fallback sl_recommended or structural invalidation + explicit positive buffer only
    try:
        sl_rec = float(zd.get("sl_recommended") or 0.0)
        if _side_ok(sl_rec):
            return round(sl_rec, 2)
    except (TypeError, ValueError):
        pass
    try:
        buf = float(zd.get("invalidation_buffer") or 0.0)
    except (TypeError, ValueError):
        buf = 0.0
    if struct_inv > 0 and buf > 0:
        return round(struct_inv - buf if demand else struct_inv + buf, 2)
    return 0.0


def evaluate_smc_setup_live_invalidation(
    zone_or_setup: Union[Zone, Dict[str, Any]],
    current_price: Optional[float] = None,
    bid: Optional[float] = None,
    ask: Optional[float] = None,
    frames: Optional[Dict[str, pd.DataFrame]] = None,
    spread: float = 0.0,
    market_status: Optional[str] = None,
) -> Tuple[bool, str, Optional[float], Optional[float], str]:
    """
    Evaluasi invalidasi setup aktif SMC berbasis harga MT5 live Bid/Ask sesuai arah & buffer SL existing:
    - BUY/DEMAND : invalid (`SETUP_INVALID_SL_BREACHED`) jika live Bid <= setup_invalidation_price (batas bawah)
    - SELL/SUPPLY: invalid (`SETUP_INVALID_SL_BREACHED`) jika live Ask >= setup_invalidation_price (batas atas)
    Tidak pernah membatalkan struktur zona (struktur tetap pakai closed candle TF asal).
    Return: (is_invalid, reason, live_price_used, invalidation_level, price_side).
    """
    if str(market_status or "").strip().upper() in ("CLOSED", "MARKET_CLOSED"):
        return False, "", None, None, ""
    zd = zone_or_setup.to_dict() if hasattr(zone_or_setup, "to_dict") else (zone_or_setup if isinstance(zone_or_setup, dict) else {})
    if not zd:
        return False, "", None, None, ""
    z_type = str(zd.get("zone_type") or zd.get("type") or "").upper()
    direction = str(zd.get("direction") or zd.get("arah") or "").upper()
    if direction not in ("BUY", "SELL"):
        if z_type == "DEMAND":
            direction = "BUY"
        elif z_type == "SUPPLY":
            direction = "SELL"
        else:
            return False, "", None, None, ""

    eff_spread = float(spread or 0.0)
    if eff_spread <= 0 and isinstance(bid, (int, float)) and isinstance(ask, (int, float)) and ask >= bid > 0:
        eff_spread = round(float(ask) - float(bid), 2)

    setup_inv = resolve_smc_setup_invalidation_price(zone_or_setup, frames=frames, spread=eff_spread)
    if not (isinstance(setup_inv, (int, float)) and math.isfinite(setup_inv) and setup_inv > 0):
        return False, "", None, None, ""

    if direction == "BUY":
        if isinstance(bid, (int, float)) and math.isfinite(bid) and bid > 0:
            live_px, side = float(bid), "BID"
        elif isinstance(current_price, (int, float)) and math.isfinite(current_price) and current_price > 0:
            live_px, side = float(current_price), "PRICE"
        else:
            return False, "", None, setup_inv, ""
        if live_px <= setup_inv:
            rsn = f"{INVALIDATION_SETUP_SL_BREACHED} (BUY live {side} {live_px:.2f} <= SETUP_SL {setup_inv:.2f})"
            return True, rsn, round(live_px, 2), setup_inv, side
        return False, "", round(live_px, 2), setup_inv, side

    if isinstance(ask, (int, float)) and math.isfinite(ask) and ask > 0:
        live_px, side = float(ask), "ASK"
    elif isinstance(current_price, (int, float)) and math.isfinite(current_price) and current_price > 0:
        live_px, side = float(current_price), "PRICE"
    else:
        return False, "", None, setup_inv, ""
    if live_px >= setup_inv:
        rsn = f"{INVALIDATION_SETUP_SL_BREACHED} (SELL live {side} {live_px:.2f} >= SETUP_SL {setup_inv:.2f})"
        return True, rsn, round(live_px, 2), setup_inv, side
    return False, "", round(live_px, 2), setup_inv, side


def resolve_analysis_direction(
    ctx: Optional[Dict[str, Any]] = None,
    decision: Optional[Dict[str, Any]] = None,
    scan: Optional[Dict[str, Any]] = None,
    snr_bias: Optional[Dict[str, Any]] = None,
    frames: Optional[Dict[str, pd.DataFrame]] = None,
    market_status: Optional[str] = "OPEN",
    data_ok: bool = True,
) -> str:
    """
    Arah analisis deterministik: saat MARKET OPEN dengan data valid WAJIB mengembalikan BUY atau SELL
    tanpa memalsukan sinyal entry, SL, TP, atau confidence.
    """
    mkt = str(market_status or "OPEN").strip().upper()
    if mkt in ("CLOSED", "MARKET_CLOSED", "DATA UNAVAILABLE", "UNAVAILABLE", "NO_DATA") or not data_ok:
        return "N/A"
    dec = decision if isinstance(decision, dict) else {}
    sc = scan if isinstance(scan, dict) else {}
    c = ctx if isinstance(ctx, dict) else (sc.get("htf_context") or dec.get("context") or {})
    sb = snr_bias if isinstance(snr_bias, dict) else {}

    # 1. Valid confirmed direction
    if str(dec.get("status") or "").upper() == "VALID" and str(dec.get("direction") or "").upper() in ("BUY", "SELL"):
        return str(dec["direction"]).upper()
    if str(dec.get("signal") or "").upper() in ("BUY", "SELL"):
        return str(dec["signal"]).upper()

    # 2. HTF bias cascade (SMC)
    for k in ("H4", "H1", "M30", "D1", "htf_bias", "h1_structure", "bias_utama"):
        val = str(c.get(k) or "").upper()
        if val == "BULLISH":
            return "BUY"
        if val == "BEARISH":
            return "SELL"

    # 3. SNR market bias cascade & weighted score
    for k in ("overall_bias", "h4", "h1", "m30"):
        val = str(sb.get(k) or "").upper()
        if val == "BULLISH":
            return "BUY"
        if val == "BEARISH":
            return "SELL"
    per_tf = sb.get("per_timeframe") if isinstance(sb.get("per_timeframe"), dict) else {}
    if per_tf:
        w_score = 0.0
        for tf_k, w in (("H4", 0.45), ("H1", 0.35), ("M30", 0.20)):
            tf_sc = (per_tf.get(tf_k) or {}).get("score")
            if isinstance(tf_sc, (int, float)) and math.isfinite(tf_sc):
                w_score += float(tf_sc) * w
        if abs(w_score) > 1e-9:
            return "BUY" if w_score > 0 else "SELL"

    # 4. Primary/active zone direction
    for z_cand in (dec.get("zone"), dec.get("primary_zone"), dec.get("nearest_valid_zone")):
        if isinstance(z_cand, dict):
            zt = str(z_cand.get("zone_type") or z_cand.get("type") or "").upper()
            if zt in ("DEMAND", "SUPPORT"):
                return "BUY"
            if zt in ("SUPPLY", "RESISTANCE"):
                return "SELL"

    # 5. Deterministic tie-breaker from OHLC / EMA when pure sideways
    f_src = frames if isinstance(frames, dict) else (sc.get("snapshot") or {}).get("frames") if isinstance(sc.get("snapshot"), dict) else None
    if isinstance(f_src, dict):
        for tf_k in ("H4", "H1", "M30", "M5"):
            df_t = f_src.get(tf_k)
            if df_t is not None and len(df_t) >= 2 and "Close" in df_t.columns:
                c_last = float(df_t["Close"].iloc[-1])
                if "EMA50" in df_t.columns and math.isfinite(float(df_t["EMA50"].iloc[-1])):
                    return "BUY" if c_last >= float(df_t["EMA50"].iloc[-1]) else "SELL"
                o_last = float(df_t["Open"].iloc[-1]) if "Open" in df_t.columns else float(df_t["Close"].iloc[-2])
                return "BUY" if c_last >= o_last else "SELL"
    return "BUY"


def evaluate_smc_zone_closed_invalidation(
    z: Union[Zone, Dict[str, Any]],
    frames: Optional[Dict[str, pd.DataFrame]] = None,
) -> Tuple[bool, str, Optional[float], Optional[str], Optional[str]]:
    """
    Evaluasi invalidasi zona SMC hanya dari candle CLOSED pada timeframe asal zona (wick saja tidak cukup).
    Supply invalid hanya jika candle closed menembus batas invalidasi atas (> max(high, invalidation_price));
    Demand invalid hanya jika candle closed menembus batas invalidasi bawah (< min(low, invalidation_price)).
    Hanya memeriksa candle CLOSED setelah waktu pembentukan/BOS zona (tidak pernah memindai histori pra-zona).
    Return: (is_invalid, reason, break_close, break_time, origin_tf).
    """
    zd = z.to_dict() if hasattr(z, "to_dict") else (z if isinstance(z, dict) else {})
    if not zd or not isinstance(frames, dict):
        return False, "", None, None, None
    z_type = str(zd.get("zone_type") or zd.get("type") or "").upper()
    if z_type not in ("DEMAND", "SUPPLY"):
        return False, "", None, None, None
    lo = float(zd.get("price_low", zd.get("low", 0.0)) or 0.0)
    hi = float(zd.get("price_high", zd.get("high", 0.0)) or 0.0)
    raw_inval = float(zd.get("invalidation_price") or (lo if z_type == "DEMAND" else hi))
    if z_type == "SUPPLY":
        inval = max(hi, raw_inval) if hi > 0 else raw_inval
    else:
        inval = min(lo, raw_inval) if lo > 0 else raw_inval
    if not (math.isfinite(inval) and inval > 0):
        return False, "", None, None, None

    origin_tf = _zone_origin_timeframe(zd, frames)
    if not origin_tf or origin_tf not in frames:
        return False, "", None, None, origin_tf
    df = frames.get(origin_tf)
    if df is None or len(df) == 0 or "Close" not in df.columns:
        return False, "", None, None, origin_tf

    # Jika frame masih menyertakan candle forming (is_closed is False), kecualikan bar terakhir
    work_df = df.iloc[:-1] if (getattr(df, "attrs", {}) or {}).get("is_closed") is False else df
    if len(work_df) == 0:
        return False, "", None, None, origin_tf

    if isinstance(work_df.index, pd.DatetimeIndex):
        ref_t, is_bos_ref = _resolve_smc_zone_origin_time(zd, work_df=work_df)
        if ref_t is None or pd.isna(ref_t):
            # Fail-safe: tanpa referensi waktu pembentukan/BOS zona, JANGAN memindai histori pra-zona
            return False, "", None, None, origin_tf
        tf_min = TF_MINUTES.get(origin_tf, 0)
        if tf_min > 0 and is_bos_ref:
            post_df = work_df.loc[work_df.index >= (ref_t + timedelta(minutes=tf_min))]
        else:
            post_df = work_df.loc[work_df.index > ref_t]
    else:
        return False, "", None, None, origin_tf

    if len(post_df) == 0:
        return False, "", None, None, origin_tf

    closes = post_df["Close"]
    mask = (closes < inval) if z_type == "DEMAND" else (closes > inval)
    if bool(mask.any()):
        first_idx = mask[mask].index[0]
        b_close = float(closes.loc[first_idx])
        b_time = str(first_idx)
        cmp_op = "<" if z_type == "DEMAND" else ">"
        rsn = f"{INVALIDATION_ZONE_STRUCTURAL} (Close {origin_tf} {b_close:.2f} {cmp_op} batas invalidasi {inval:.2f})"
        return True, rsn, b_close, b_time, origin_tf
    return False, "", None, None, origin_tf


def is_zone_active_and_valid(
    z: Union[Zone, Dict[str, Any]],
    current_price: Optional[float] = None,
    frames: Optional[Dict[str, pd.DataFrame]] = None,
) -> bool:
    """
    Pemeriksaan integritas validitas struktural zona SMC:
    Menjamin hard rejection jika:
      - is_valid == False atau is_broken == True
      - Rejection reason mengandung 'invalid', 'broken', 'expired', 'dimitigasi', 'mitigated', 'lebar', 'sempit'
      - Freshness mengandung 'INVALID', 'BROKEN', 'EXPIRED', 'MITIGATED'
      - Status zona == 'INVALIDATED', 'BROKEN', 'EXPIRED', 'MITIGATED'
      - is_expired == True
      - touch_count > 2
      - Origin BOS invalid
      - Candle CLOSED pada timeframe asal zona telah menembus batas invalidasi (wick saja tidak cukup)
    """
    if z is None:
        return False
    zd = z.to_dict() if hasattr(z, "to_dict") else z
    if not isinstance(zd, dict):
        return False
    if not zd.get("is_valid", True) or zd.get("is_broken") is True:
        return False
    if str(zd.get("invalidation_type") or "").upper() in (INVALIDATION_SETUP_SL_BREACHED, INVALIDATION_ZONE_STRUCTURAL):
        return False
    rej = str(zd.get("rejection_reason", "")).lower()
    if any(k in rej for k in ("invalid", "broken", "expired", "dimitigasi", "mitigated", "terlalu lebar", "terlalu sempit")):
        return False
    fresh = str(zd.get("freshness", "")).upper()
    if any(k in fresh for k in ("INVALID", "BROKEN", "EXPIRED", "MITIGATED")):
        return False
    st = str(zd.get("status", "")).upper()
    if st in ("INVALIDATED", "BROKEN", "EXPIRED", "MITIGATED"):
        return False
    if zd.get("is_expired"):
        return False
    if int(zd.get("touch_count", 0)) > 2:
        return False
    orig_bos = zd.get("origin_bos") or {}
    if isinstance(orig_bos, dict) and not orig_bos.get("origin_valid", True):
        return False
    if isinstance(frames, dict):
        is_brk, _, _, _, _ = evaluate_smc_zone_closed_invalidation(zd, frames)
        if is_brk:
            return False
    return True


class ZoneDistanceResult:
    __slots__ = (
        "distance",
        "price_distance",
        "mt5_points",
        "pips",
        "point",
        "digits",
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
        mt5_points: Optional[float] = None,
        pips: Optional[float] = None,
        point: Optional[float] = None,
        digits: int = 2,
    ):
        self.distance = distance
        self.price_distance = distance
        self.mt5_points = mt5_points
        self.pips = pips
        self.point = point
        self.digits = digits
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
            f"ZoneDistanceResult(price_distance={self.distance}, mt5_points={self.mt5_points}, status={self.status!r}, "
            f"reference_price={self.reference_price}, reference_price_type={self.reference_price_type!r}, "
            f"nearest_boundary={self.nearest_boundary}, distance_atr_ratio={self.distance_atr_ratio}, "
            f"ratio_str={self.ratio_str!r})"
        )


def calculate_zone_distance(
    zone_low: float,
    zone_high: float,
    current_price: float,
    digits: Optional[int] = None,
    zone_type: Optional[str] = None,
    bid: Optional[float] = None,
    ask: Optional[float] = None,
    atr: Optional[float] = None,
    symbol: Optional[str] = None,
    spec: Optional[Dict[str, Any]] = None,
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
    u_spec = get_symbol_unit_spec(symbol=symbol, spec=spec)
    eff_digits = int(digits) if digits is not None else int(u_spec["digits"])
    eff_point = float(u_spec["point"])
    z_type = (zone_type or "").upper()
    c_price = float(current_price) if (current_price is not None and math.isfinite(current_price)) else 0.0
    b_val = float(bid) if (bid is not None and math.isfinite(bid) and bid > 0) else 0.0
    a_val = float(ask) if (ask is not None and math.isfinite(ask) and ask > 0) else 0.0
    lo = float(zone_low)
    hi = float(zone_high)
    if lo > hi:
        lo, hi = hi, lo

    # Konsisten gunakan harga snapshot MT5 (c_price) sebagai referensi tunggal jarak & status zona
    if c_price > 0:
        ref_p = c_price
        ref_t = "MID"
    elif b_val > 0 and a_val > 0:
        ref_p = (b_val + a_val) / 2.0
        ref_t = "MID"
    elif z_type == "SUPPLY" and a_val > 0:
        ref_p = a_val
        ref_t = "ASK"
    elif z_type == "DEMAND" and b_val > 0:
        ref_p = b_val
        ref_t = "BID"
    elif b_val > 0:
        ref_p = b_val
        ref_t = "BID"
    elif a_val > 0:
        ref_p = a_val
        ref_t = "ASK"
    else:
        ref_p = 0.0
        ref_t = "UNKNOWN"

    if z_type == "SUPPLY":
        nearest_bnd = lo
    elif z_type == "DEMAND":
        nearest_bnd = hi
    elif ref_p < lo - eps:
        nearest_bnd = lo
    elif ref_p > hi + eps:
        nearest_bnd = hi
    else:
        nearest_bnd = round((lo + hi) / 2.0, eff_digits)

    if ref_p <= 0:
        raw_d = 0.0
        status = "UNKNOWN"
    elif ref_p < lo - eps:
        raw_d = abs(lo - ref_p)
        status = "ABOVE_PRICE"
    elif ref_p > hi + eps:
        raw_d = abs(ref_p - hi)
        status = "BELOW_PRICE"
    else:
        raw_d = 0.0
        status = "INSIDE_ZONE"

    # Perhitungan rasio ATR tanpa pembulatan awal
    atr_val = float(atr) if (atr is not None and isinstance(atr, (int, float)) and math.isfinite(atr) and atr > 0) else None
    if atr_val is not None:
        ratio_val = round(raw_d / atr_val, 2)
        ratio_str = f"{ratio_val:.1f}x"
    else:
        ratio_val = None
        ratio_str = "DATA INCOMPLETE"

    if status == "INSIDE_ZONE":
        dist_val = 0.0
    elif raw_d > 0:
        dist_val = max(round(raw_d, eff_digits), 10 ** (-eff_digits))
    else:
        dist_val = round(raw_d, eff_digits)

    mt5_pts_val = price_distance_to_mt5_points(dist_val, point=eff_point)
    pips_val = price_distance_to_pips(dist_val, pip_size=u_spec.get("pip_size"))

    return ZoneDistanceResult(
        distance=dist_val,
        status=status,
        reference_price=round(ref_p, eff_digits),
        reference_price_type=ref_t,
        nearest_boundary=round(nearest_bnd, eff_digits),
        atr_value=round(atr_val, eff_digits) if atr_val is not None else None,
        distance_atr_ratio=ratio_val,
        ratio_str=ratio_str,
        mt5_points=mt5_pts_val,
        pips=pips_val,
        point=eff_point,
        digits=eff_digits,
    )


def calculate_zone_approach(
    distance: float,
    status: str,
    atr: float = 2.0,
) -> str:
    """
    Menentukan status pendekatan harga terhadap zona berbasis jarak aktual ke batas zona:
    - Jika status == "INSIDE_ZONE": ZoneStatus("ZONE_TOUCHED")
    - Jika 0.0 < distance <= ZONE_APPROACH_DISTANCE_POINTS: "APPROACHING"
    - Jika distance > max(35.0, 15.0 * atr): "FAR"
    - Selainnya: "UNTOUCHED"
    """
    if status == "INSIDE_ZONE":
        return ZoneStatus("ZONE_TOUCHED")
    atr_is_valid = (isinstance(atr, (int, float)) and math.isfinite(atr) and atr > 0)
    atr_val = float(atr) if atr_is_valid else None
    ref_atr = atr_val if atr_val is not None else 2.0
    far_threshold = max(35.0, 15.0 * ref_atr)
    approach_threshold = float(ZONE_APPROACH_DISTANCE_POINTS)
    if distance <= approach_threshold:
        return "APPROACHING"
    if distance > far_threshold:
        return "FAR"
    return "UNTOUCHED"


def resolve_radar_state(
    zone: Optional[Union[Zone, Dict[str, Any]]],
    decision: Optional[Dict[str, Any]] = None,
    current_price: Optional[float] = None,
    atr: Optional[float] = None,
) -> str:
    """
    Klasifikasi status zona Early Entry Radar secara deterministik:
      - INVALID       : zona broken/mitigated/expired/rejected/invalid
      - CONFIRMED     : sinyal resmi VALID pada zona ini (seluruh konfirmasi M5 + risiko terpenuhi)
      - PENDING READY : setup pending limit READY tervalidasi (HTF/M30/SL/RR lolos, market OPEN)
      - APPROACHING   : harga mendekati / menyentuh zona (belum CONFIRMED / PENDING READY)
      - RADAR         : zona berkualitas terdeteksi dan dipantau sebelum harga menyentuhnya
    """
    if not zone:
        return "INVALID"
    zd = zone.to_dict() if hasattr(zone, "to_dict") else (zone if isinstance(zone, dict) else {})
    dec = decision if isinstance(decision, dict) else {}

    lifecycle = str(zd.get("lifecycle_status") or zd.get("status") or "").upper()
    fresh = str(zd.get("freshness") or "").upper()
    grd = str(zd.get("grade") or "").upper()
    if (
        zd.get("is_valid") is False
        or zd.get("is_broken") is True
        or lifecycle in ("INVALIDATED", "BROKEN", "EXPIRED", "MITIGATED", "REJECTED")
        or fresh in ("INVALIDATED", "EXPIRED", "MITIGATED")
        or grd == "REJECTED"
    ):
        return "INVALID"

    zid = str(zd.get("zone_id") or "")
    is_mkt_closed = (
        str(dec.get("market_status") or zd.get("market_status") or "").upper() == "CLOSED"
        or str(dec.get("snapshot_status") or "").upper() == "MARKET_CLOSED"
    )
    dec_zone = dec.get("zone") if isinstance(dec.get("zone"), dict) else {}
    if (
        not is_mkt_closed
        and str(dec.get("status") or "").upper() == "VALID"
        and zid
        and str(dec_zone.get("zone_id") or "") == zid
    ):
        return "CONFIRMED"

    if not is_mkt_closed:
        early = dec.get("early_setup") if isinstance(dec.get("early_setup"), dict) else {}
        cand_plans = list(early.get("all_setups") or [])
        if isinstance(early.get("best_setup"), dict):
            cand_plans.append(early["best_setup"])
        has_ready_pending = any(
            isinstance(p, dict)
            and p.get("valid")
            and str(p.get("status") or "").upper() == "READY"
            and not p.get("is_expired")
            and (zid and str(p.get("zone_id") or "") == zid)
            for p in cand_plans
        )
        if has_ready_pending or str(zd.get("pending_eligibility") or "").upper() == "READY":
            return "PENDING READY"

    app = str(zd.get("approach") or "").upper()
    d_st = str(zd.get("direction_status") or "").upper()
    lo = float(zd.get("price_low", zd.get("low", 0.0)) or 0.0)
    hi = float(zd.get("price_high", zd.get("high", 0.0)) or 0.0)
    if current_price is not None and current_price > 0 and lo > 0 and hi >= lo:
        dist_v = 0.0 if lo <= current_price <= hi else (lo - current_price if current_price < lo else current_price - hi)
        d_st = "INSIDE_ZONE" if dist_v == 0.0 else ("ABOVE_PRICE" if lo > current_price else "BELOW_PRICE")
        eff_atr = float(atr if (isinstance(atr, (int, float)) and atr > 0) else (zd.get("atr_value") or 2.0))
        app = str(calculate_zone_approach(dist_v, d_st, atr=eff_atr)).upper()

    if not is_mkt_closed and (d_st == "INSIDE_ZONE" or app in ("APPROACHING", "ZONE_TOUCHED", "TOUCHED", "INSIDE_ZONE")):
        return "APPROACHING"
    return "RADAR"


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


def _is_pure_m5_tf(timeframe: str) -> bool:
    parts = [p.strip().upper() for p in re.split(r"[+/, ]+", str(timeframe or "")) if p.strip()]
    return parts == ["M5"]


def _intraday_tf_rank(
    timeframe: str,
    m30_status: str = "CONFIRMED",
    h1_aligned: bool = True,
) -> int:
    """
    Hierarki timeframe intraday:
      - H4/H1 = konteks arah & struktur; M30 = konfirmasi momentum & konteks zona;
      - M15/M5 diprioritaskan untuk entry intraday bila struktur H4/H1/M30 mendukung.
      - M5 tidak pernah dipaksakan jika struktur/momentum belum mendukung.
    """
    parts = [p.strip().upper() for p in re.split(r"[+/, ]+", str(timeframe or "")) if p.strip()]
    m30_u = str(m30_status or "").upper()
    m30_ok = m30_u not in ("CONFLICT", "INVALID", "WEAK")
    m30_strong = m30_u == "CONFIRMED"
    has_m15 = "M15" in parts
    has_m5 = "M5" in parts
    has_m30 = "M30" in parts
    has_h1 = "H1" in parts
    has_h4 = "H4" in parts

    if m30_ok:
        if has_m15:
            return 0
        if has_m5 and m30_strong and h1_aligned:
            return 1
        if has_m30:
            return 2
        if has_h1:
            return 3
        if has_h4:
            return 4
        if has_m5:
            return 5
    else:
        if has_h1:
            return 0
        if has_h4:
            return 1
        if has_m30:
            return 2
        if has_m15:
            return 3
        if has_m5:
            return 4
    return 6


def _resolve_relevance_limits(origin_tf: str, atr: Optional[float]) -> Tuple[float, float, float, float]:
    """Kembalikan (active_max_pt, radar_max_pt, active_max_atr, radar_max_atr) proporsional terhadap ATR & TF."""
    tf_key = str(origin_tf or "H1").upper()
    a_atr, r_atr, a_min_pt, a_max_pt, r_min_pt, r_max_pt = ZONE_RELEVANCE_TF_LIMITS.get(
        tf_key, ZONE_RELEVANCE_TF_LIMITS["H1"]
    )
    if atr is not None and isinstance(atr, (int, float)) and math.isfinite(atr) and atr > 0:
        active_pt = min(a_max_pt, max(a_min_pt, round(a_atr * float(atr), 2)))
        radar_pt = min(r_max_pt, max(r_min_pt, round(r_atr * float(atr), 2)))
    else:
        active_pt = a_max_pt
        radar_pt = r_max_pt
    radar_pt = min(radar_pt, float(LOW_RISK_MAX_DISTANCE_PT))
    active_pt = min(active_pt, radar_pt)
    return round(active_pt, 2), round(radar_pt, 2), float(a_atr), float(r_atr)


def classify_zone_relevance(
    zone: Union[Zone, Dict[str, Any]],
    current_price: Optional[float],
    atr: Optional[float] = None,
    htf_bias: str = "NETRAL",
    h1_structure: str = "NETRAL",
    m30_status: str = "CONFIRMED",
    frames: Optional[Dict[str, pd.DataFrame]] = None,
    bid: Optional[float] = None,
    ask: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Klasifikasi Distance-Aware Single Source of Truth untuk seluruh zona SMC:
      - INVALID       : Struktur rusak / broken candle close TF asal / mitigated (>2x) / expired / tidak valid.
      - FAR_REFERENCE : Struktur masih valid, tetapi di luar batas jarak ATR/poin sesuai TF asal
                        (simpan sebagai referensi saja, dilarang menjadi Active Focus atau pending intraday).
      - RADAR         : Struktur valid & dalam jangkauan intraday, namun belum memenuhi kriteria eksekusi ACTIVE
                        (jarak radar, countertrend, M5 tanpa dukungan penuh, atau kualitas menengah).
      - ACTIVE        : Struktur valid, berkualitas, searah H4/H1, struktur M30 mendukung (untuk LTF),
                        dan berada dalam jarak eksekusi intraday proporsional terhadap ATR.
    Catatan: Harga menjauh TIDAK PERNAH mengubah zona valid menjadi INVALID.
    """
    zd = zone.to_dict() if hasattr(zone, "to_dict") else (dict(zone) if isinstance(zone, dict) else {})
    if not zd:
        return {
            "relevance": "INVALID",
            "reason": "Data zona kosong",
            "origin_tf": "H1",
            "distance": 0.0,
            "distance_atr_ratio": None,
            "active_limit_pt": 0.0,
            "radar_limit_pt": 0.0,
        }

    lo = float(zd.get("price_low", zd.get("low", 0.0)) or 0.0)
    hi = float(zd.get("price_high", zd.get("high", 0.0)) or 0.0)
    z_type = str(zd.get("zone_type") or zd.get("type") or "").upper()
    tf_raw = str(zd.get("timeframe") or "H1").upper()
    origin_tf = _zone_origin_timeframe(zd, frames) or (tf_raw.split("+")[0] if tf_raw else "H1")

    atr_val: Optional[float] = None
    if atr is not None and isinstance(atr, (int, float)) and math.isfinite(atr) and atr > 0:
        atr_val = float(atr)
    elif isinstance(zd.get("atr_value"), (int, float)) and math.isfinite(float(zd["atr_value"])) and float(zd["atr_value"]) > 0:
        atr_val = float(zd["atr_value"])
    elif isinstance(frames, dict):
        diag_h1 = extract_atr_diagnostics(frames, preferred_tf="H1")
        if isinstance(diag_h1.get("atr_value"), (int, float)) and diag_h1["atr_value"] > 0:
            atr_val = float(diag_h1["atr_value"])

    active_pt, radar_pt, active_atr, radar_atr = _resolve_relevance_limits(origin_tf, atr_val)

    c_price = float(current_price) if (isinstance(current_price, (int, float)) and math.isfinite(current_price) and current_price > 0) else 0.0
    if lo > 0 and hi >= lo and c_price > 0:
        dist_res = calculate_zone_distance(lo, hi, c_price, zone_type=z_type, bid=bid, ask=ask, atr=atr_val)
        dist = dist_res.distance
        dir_st = dist_res.status
        dist_ratio = dist_res.distance_atr_ratio
    else:
        dist = float(zd.get("distance", 0.0) or 0.0)
        dir_st = str(zd.get("direction_status") or "")
        dist_ratio, _ = calculate_distance_atr_ratio(dist, atr_val)

    # 1. INVALID gate (hanya kerusakan struktural / closed break / mitigated / expired)
    # Status INVALID tersimpan (tracker) berlaku konsisten untuk scan / scan snr / pantau, juga tanpa frames.
    _persist_rec = None
    try:
        _trk_rel = _ZONE_TRACKER if _ZONE_TRACKER is not None else None
        if _trk_rel is not None:
            _p_inv, _persist_rec = _trk_rel.is_zone_invalidated(zd)
            if not _p_inv:
                _persist_rec = None
    except Exception:
        _persist_rec = None
    if _persist_rec is not None or not is_zone_active_and_valid(zd, c_price if c_price > 0 else None, frames=frames):
        rej_r = str((_persist_rec or {}).get("reason") or zd.get("rejection_reason") or zd.get("invalid_label") or zd.get("status") or "INVALID").strip()
        if _persist_rec is not None:
            rej_r = f"{rej_r} | TF {_persist_rec.get('timeframe', '')} | Close {_persist_rec.get('break_close')} @ {_persist_rec.get('break_time')} | dicatat {_persist_rec.get('invalidated_at')}"
        return {
            "relevance": "INVALID",
            "reason": rej_r,
            "origin_tf": origin_tf,
            "distance": dist,
            "distance_atr_ratio": dist_ratio,
            "active_limit_pt": active_pt,
            "radar_limit_pt": radar_pt,
        }

    # 2. FAR REFERENCE gate (sebelum pemilihan Primary/Active Focus & Pending)
    over_global_pt = dist > LOW_RISK_MAX_DISTANCE_PT
    over_global_atr = (dist_ratio is not None and dist_ratio > LOW_RISK_MAX_ATR_RATIO)
    over_tf_pt = dist > radar_pt
    if over_global_pt or over_global_atr or over_tf_pt:
        ratio_s = f"{dist_ratio:.1f}x ATR" if dist_ratio is not None else "N/A ATR"
        is_ext_far = bool(dist_ratio is not None and dist_ratio > 15.0)
        far_cat = "EXTREME_FAR" if is_ext_far else "FAR_INTRADAY"
        if is_ext_far:
            far_rsn = f"EXTREME_FAR ({origin_tf}): jarak {dist:.2f} USD ({ratio_s} > 15.0x ATR) melebihi batas intraday {origin_tf} (<= {radar_pt:.1f} USD)"
        else:
            far_rsn = f"FAR_INTRADAY ({origin_tf}): jarak {dist:.2f} USD ({ratio_s}) melebihi batas intraday {origin_tf} (<= {radar_pt:.1f} USD / {radar_atr:.1f}x ATR)"
        return {
            "relevance": "FAR_REFERENCE",
            "far_category": far_cat,
            "reason": far_rsn,
            "origin_tf": origin_tf,
            "distance": dist,
            "distance_atr_ratio": dist_ratio,
            "active_limit_pt": active_pt,
            "radar_limit_pt": radar_pt,
        }

    # 3. Evaluasi struktur market, arah HTF, dan kualitas zona untuk ACTIVE vs RADAR
    htf_u = str(htf_bias or "NETRAL").upper()
    h1_u = str(h1_structure or "NETRAL").upper()
    m30_u = str(m30_status or "CONFIRMED").upper()
    is_aligned = (htf_u == "BULLISH" and z_type == "DEMAND") or (htf_u == "BEARISH" and z_type == "SUPPLY")
    is_counter = (htf_u == "BULLISH" and z_type == "SUPPLY") or (htf_u == "BEARISH" and z_type == "DEMAND")
    h1_conflict = (htf_u == "BULLISH" and h1_u == "BEARISH") or (htf_u == "BEARISH" and h1_u == "BULLISH")
    m30_bad = m30_u in ("CONFLICT", "INVALID", "WEAK")
    score = float(zd.get("score", 0.0) or 0.0)
    fresh = str(zd.get("freshness", "FRESH")).upper()
    has_fvg = bool(zd.get("fvg_confirmation"))
    disp_ok = float(zd.get("displacement_strength", 0.0) or 0.0) >= MIN_DISPLACEMENT_ATR
    wrong_side_intrabar = (
        (z_type == "DEMAND" and dir_st == "ABOVE_PRICE")
        or (z_type == "SUPPLY" and dir_st == "BELOW_PRICE")
    )

    if is_counter:
        return {
            "relevance": "RADAR",
            "reason": f"RADAR Countertrend ({origin_tf}): melawan bias H4 {htf_u}, pantau reaksi saja ({dist:.2f} USD)",
            "origin_tf": origin_tf,
            "distance": dist,
            "distance_atr_ratio": dist_ratio,
            "active_limit_pt": active_pt,
            "radar_limit_pt": radar_pt,
        }

    if htf_u not in ("BULLISH", "BEARISH") or h1_conflict or wrong_side_intrabar:
        why_r = "bias H4 netral" if htf_u not in ("BULLISH", "BEARISH") else ("konflik H4 vs H1" if h1_conflict else "harga intrabar melewati sisi distal")
        return {
            "relevance": "RADAR",
            "reason": f"RADAR ({origin_tf}): {why_r} ({dist:.2f} USD)",
            "origin_tf": origin_tf,
            "distance": dist,
            "distance_atr_ratio": dist_ratio,
            "active_limit_pt": active_pt,
            "radar_limit_pt": radar_pt,
        }

    # Jangan memaksakan zona M5 jika struktur H1/M30 atau kualitas tidak mendukung
    if _is_pure_m5_tf(tf_raw):
        m5_struct_ok = is_aligned and (h1_u == htf_u) and (not m30_bad) and fresh in ("FRESH", "TESTED_1X") and score >= 70.0 and has_fvg and disp_ok
        if not m5_struct_ok:
            return {
                "relevance": "RADAR",
                "reason": f"RADAR (M5): struktur H1/M30 atau kualitas M5 belum cukup kuat untuk Active Focus ({dist:.2f} USD)",
                "origin_tf": origin_tf,
                "distance": dist,
                "distance_atr_ratio": dist_ratio,
                "active_limit_pt": active_pt,
                "radar_limit_pt": radar_pt,
            }

    # Zona ACTIVE: searah H4/H1, berkualitas (fresh/tested_1x, skor memadai), dan dalam batas ACTIVE proporsional ATR
    quality_for_active = (
        score >= SCORE_MIN_ACTIVE
        and fresh in ("FRESH", "TESTED_1X")
        and disp_ok
    )
    if is_aligned and dist <= active_pt and quality_for_active and not (m30_bad and origin_tf in ("M5", "M15")):
        ratio_s = f"{dist_ratio:.1f}x ATR" if dist_ratio is not None else "N/A ATR"
        return {
            "relevance": "ACTIVE",
            "reason": f"ACTIVE ({origin_tf}): searah H4 {htf_u}, {fresh}, jarak {dist:.2f} USD ({ratio_s} <= {active_pt:.1f} USD)",
            "origin_tf": origin_tf,
            "distance": dist,
            "distance_atr_ratio": dist_ratio,
            "active_limit_pt": active_pt,
            "radar_limit_pt": radar_pt,
        }

    ratio_s = f"{dist_ratio:.1f}x ATR" if dist_ratio is not None else "N/A ATR"
    return {
        "relevance": "RADAR",
        "reason": f"RADAR ({origin_tf}): valid dalam jangkauan intraday ({dist:.2f} USD / {ratio_s} <= {radar_pt:.1f} USD)",
        "origin_tf": origin_tf,
        "distance": dist,
        "distance_atr_ratio": dist_ratio,
        "active_limit_pt": active_pt,
        "radar_limit_pt": radar_pt,
    }


def is_far_reference_zone(
    zone: Union[Zone, Dict[str, Any], None],
    current_price: Optional[float] = None,
    atr: Optional[float] = None,
) -> bool:
    """Single source of truth untuk mendeteksi zona FAR_REFERENCE / FAR_ZONE lintas `scan` dan `pantau`."""
    if zone is None:
        return False
    zd = zone.to_dict() if hasattr(zone, "to_dict") else (zone if isinstance(zone, dict) else {})
    if not zd:
        return False
    rel = str(zd.get("zone_relevance") or "").upper()
    if rel == "FAR_REFERENCE":
        return True
    if str(zd.get("risk_tier") or "").upper() == "FAR_ZONE":
        return True
    if zd.get("is_distant_historical"):
        return True
    d_ratio = zd.get("distance_atr_ratio")
    if isinstance(d_ratio, (int, float)) and math.isfinite(d_ratio) and d_ratio > LOW_RISK_MAX_ATR_RATIO:
        return True
    d_pt = zd.get("distance", zd.get("distance_points"))
    if isinstance(d_pt, (int, float)) and math.isfinite(d_pt) and d_pt > LOW_RISK_MAX_DISTANCE_PT:
        return True
    origin_tf = _zone_origin_timeframe(zd) or "H1"
    eff_atr = atr if (isinstance(atr, (int, float)) and math.isfinite(atr) and atr > 0) else zd.get("atr_value")
    _, radar_pt, _, _ = _resolve_relevance_limits(origin_tf, eff_atr)
    if current_price is not None and isinstance(current_price, (int, float)) and math.isfinite(current_price) and current_price > 0:
        lo = float(zd.get("price_low", zd.get("low", 0.0)) or 0.0)
        hi = float(zd.get("price_high", zd.get("high", 0.0)) or 0.0)
        if lo > 0 and hi >= lo:
            live_dist = 0.0 if lo <= current_price <= hi else (lo - current_price if current_price < lo else current_price - hi)
            if live_dist > radar_pt:
                return True
    elif isinstance(d_pt, (int, float)) and math.isfinite(d_pt) and d_pt > radar_pt:
        return True
    return False


def _zones_substantially_overlap(
    z1: Dict[str, Any],
    z2: Dict[str, Any],
    overlap_ratio_thr: float = 0.50,
    gap_tol: float = 0.25,
) -> bool:
    """True bila dua zona searah tumpang tindih signifikan atau duplikat di kantong harga yang sama."""
    if not isinstance(z1, dict) or not isinstance(z2, dict):
        return False
    zt1 = str(z1.get("zone_type") or "").upper()
    zt2 = str(z2.get("zone_type") or "").upper()
    if zt1 and zt2 and zt1 != zt2:
        return False
    id1, id2 = str(z1.get("zone_id") or ""), str(z2.get("zone_id") or "")
    if id1 and id2 and id1 == id2:
        return True
    lo1 = float(z1.get("price_low", z1.get("zone_low", 0.0)) or 0.0)
    hi1 = float(z1.get("price_high", z1.get("zone_high", 0.0)) or 0.0)
    lo2 = float(z2.get("price_low", z2.get("zone_low", 0.0)) or 0.0)
    hi2 = float(z2.get("price_high", z2.get("zone_high", 0.0)) or 0.0)
    if not (lo1 > 0 and hi1 >= lo1 and lo2 > 0 and hi2 >= lo2):
        return False
    ov = min(hi1, hi2) - max(lo1, lo2)
    min_w = max(0.1, min(hi1 - lo1, hi2 - lo2))
    if ov >= 0 and (ov / min_w) >= overlap_ratio_thr:
        return True
    tf1 = str(z1.get("timeframe") or "").upper()
    tf2 = str(z2.get("timeframe") or "").upper()
    if tf1 == tf2 and ov >= -gap_tol:
        return True
    return False


def _deduplicate_overlapping_zones(
    zones: List[Dict[str, Any]],
    overlap_ratio_thr: float = 0.50,
    gap_tol: float = 0.25,
) -> Tuple[List[Dict[str, Any]], List[Tuple[Dict[str, Any], Dict[str, Any]]]]:
    """
    Hapus duplikat / zona tumpang tindih yang berada di urutan prioritas lebih rendah.
    Input diasumsikan sudah terurut dari prioritas tertinggi ke terendah.
    Return: (kept_zones, suppressed_pairs[(suppressed_zone, winner_zone)]).
    """
    kept: List[Dict[str, Any]] = []
    suppressed: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for z in zones:
        winner = next((k for k in kept if _zones_substantially_overlap(k, z, overlap_ratio_thr=overlap_ratio_thr, gap_tol=gap_tol)), None)
        if winner is not None:
            suppressed.append((z, winner))
        else:
            kept.append(z)
    return kept, suppressed


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

    # --- TAHAP 3: INVALIDATION STATUS (Hanya dari candle closed TF asal / status terkonfirmasi; wick tidak cukup) ---
    if fresh == "INVALIDATED" or zd.get("status") in ("INVALIDATED", "BROKEN") or zd.get("is_broken") is True:
        return _finalize("UNCLASSIFIED", "Struktur zona telah INVALID — BROKEN")

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

    # --- TAHAP 7: RISK CLASSIFICATION & DISTANCE ATR RATIO (TF + ATR AWARE) ---
    # Hitung jarak aktual dinamis terhadap current_price
    if lo > 0 and hi > 0 and current_price > 0:
        dist_res = calculate_zone_distance(lo, hi, current_price, zone_type=z_type, bid=bid, ask=ask)
        dist = dist_res.distance
    else:
        dist = float(zd.get("distance", 0.0))

    dist_ratio, _ = calculate_distance_atr_ratio(dist, atr)
    if dist_ratio is None:
        return _finalize("DATA_INCOMPLETE", "ATR tidak valid atau tidak tersedia — klasifikasi Multi-Tier ditangguhkan (DATA INCOMPLETE)")
    dist_ratio_str = f"{dist_ratio:.1f}x ATR"

    # Filter jarak ATR & timeframe asal (Distance-Aware Relevance Gate)
    rel_info = classify_zone_relevance(
        zd, current_price, atr=atr, htf_bias=htf_bias, h1_structure=h1_structure, bid=bid, ask=ask
    )
    if hasattr(zone, "zone_relevance"):
        zone.zone_relevance = rel_info["relevance"]
        zone.zone_relevance_reason = rel_info["reason"]
    elif isinstance(zone, dict):
        zone["zone_relevance"] = rel_info["relevance"]
        zone["zone_relevance_reason"] = rel_info["reason"]
    if rel_info["relevance"] == "FAR_REFERENCE":
        return _finalize("FAR_ZONE", rel_info["reason"])

    # Filter zona makro yang terlalu jauh untuk trading intraday aktif (> 25.0 ATR atau > 75.0 USD)
    over_atr = (dist_ratio is not None and dist_ratio > LOW_RISK_MAX_ATR_RATIO)
    over_pt = (dist > LOW_RISK_MAX_DISTANCE_PT)
    if over_atr or over_pt:
        if over_atr and over_pt:
            exceed_desc = f"melebihi batas ATR ratio (> {LOW_RISK_MAX_ATR_RATIO:.1f}x ATR) dan jarak absolut (> {LOW_RISK_MAX_DISTANCE_PT:.1f} USD)"
        elif over_atr:
            exceed_desc = f"melebihi batas ATR ratio maksimal (> {LOW_RISK_MAX_ATR_RATIO:.1f}x ATR)"
        else:
            exceed_desc = f"melebihi batas jarak absolut maksimal (> {LOW_RISK_MAX_DISTANCE_PT:.1f} USD)"
        return _finalize("FAR_ZONE", f"Jarak zona ({dist:.2f} USD / {dist_ratio_str}) {exceed_desc}")

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
    # H1 / H4, HTF Alignment Kuat, Grade A / A+ (Score >= 75.0), Fresh / Tested 1x, Jarak wajar (<= 25 ATR DAN <= 75 USD)
    if has_htf and is_aligned and (grade in ("A+", "A") or score >= LOW_RISK_MIN_SCORE) and fresh in ("FRESH", "TESTED_1X") and is_conservative_dist and not is_abnormal_width:
        # Zona LTF murni (M5/M15) tidak boleh masuk ke Low Risk konservatif
        if not (has_ltf and "H" not in tf):
            return _finalize("LOW_RISK", f"Konservatif HTF ({tf}): Grade {grade} (Skor {score:.1f}), Fresh, selaras bias {htf_bias}, jarak proporsional ({dist:.2f} USD / {dist_ratio_str})")

    # 2. Kategori HIGH RISK (Aggressive Zone)
    # Dekat dengan harga (<= 6.0 ATR DAN <= 18 USD), M5/M15/M30 (atau H1 dekat), Fresh, Score >= 55.0
    if is_close and score >= HIGH_RISK_MIN_SCORE and fresh in ("FRESH", "TESTED_1X", "TESTED_2X") and not is_abnormal_width:
        align_tag = "Pro-Trend" if is_aligned else ("Countertrend" if (htf_bias in ("BULLISH", "BEARISH")) else "Netral")
        return _finalize("HIGH_RISK", f"Agresif {align_tag} ({tf}): Dekat harga ({dist:.2f} USD / {dist_ratio_str}), Grade {grade} (Skor {score:.1f}), {fresh}")

    # 3. Kategori MEDIUM RISK (Balanced Zone)
    # M15, M30, H1, Score >= 65.0, Jarak proporsional (<= 16.0 ATR DAN <= 50 USD), Fresh/Tested 1x
    if has_mtf and score >= MEDIUM_RISK_MIN_SCORE and is_moderate_dist and fresh in ("FRESH", "TESTED_1X") and not is_abnormal_width:
        if is_aligned:
            align_desc = f"Balanced Pro-Trend (Searah bias HTF {htf_bias})"
        elif htf_bias in ("BULLISH", "BEARISH"):
            align_desc = f"Balanced Countertrend (Melawan bias HTF {htf_bias} — Pantau Reaksi Saja)"
        else:
            align_desc = f"Balanced Netral (Bias HTF {htf_bias})"
        return _finalize("MEDIUM_RISK", f"Moderat {align_desc} ({tf}): Grade {grade} (Skor {score:.1f}), {fresh}, jarak seimbang ({dist:.2f} USD / {dist_ratio_str})")

    # Fallback terkontrol dalam batas jarak ketat (dilarang menerima zona jarak berlebih atau TF tidak sesuai)
    if is_close and score >= HIGH_RISK_MIN_SCORE and not is_abnormal_width:
        align_tag = "Pro-Trend" if is_aligned else ("Countertrend" if (htf_bias in ("BULLISH", "BEARISH")) else "Netral")
        return _finalize("HIGH_RISK", f"Agresif Dekat {align_tag} ({tf}): Skor {score:.1f}, jarak {dist:.2f} USD ({dist_ratio_str})")
    if has_mtf and is_moderate_dist and score >= MEDIUM_RISK_MIN_SCORE and fresh in ("FRESH", "TESTED_1X") and not is_abnormal_width:
        align_tag = "Pro-Trend" if is_aligned else ("Countertrend" if (htf_bias in ("BULLISH", "BEARISH")) else "Netral")
        return _finalize("MEDIUM_RISK", f"Moderat {align_tag} ({tf}): Skor {score:.1f}, jarak {dist:.2f} USD ({dist_ratio_str})")
    if has_htf and is_aligned and is_conservative_dist and score >= LOW_RISK_MIN_SCORE and fresh in ("FRESH", "TESTED_1X") and not is_abnormal_width:
        return _finalize("LOW_RISK", f"Konservatif HTF ({tf}): Skor {score:.1f}, selaras bias {htf_bias}, jarak {dist:.2f} USD")

    missed = []
    if not is_close:
        missed.append(f"High Risk butuh jarak <= {HIGH_RISK_MAX_ATR_RATIO:.1f}x ATR & <= {HIGH_RISK_MAX_DISTANCE_PT:.1f} USD (aktual: {dist:.2f} USD / {dist_ratio_str})")
    elif score < HIGH_RISK_MIN_SCORE:
        missed.append(f"High Risk butuh skor >= {HIGH_RISK_MIN_SCORE:.1f} (aktual: {score:.1f})")

    if not has_mtf:
        missed.append(f"Medium Risk butuh TF M15/M30/H1 (aktual TF: {tf})")
    elif not is_moderate_dist:
        missed.append(f"Medium Risk butuh jarak <= {MEDIUM_RISK_MAX_ATR_RATIO:.1f}x ATR & <= {MEDIUM_RISK_MAX_DISTANCE_PT:.1f} USD (aktual: {dist:.2f} USD / {dist_ratio_str})")
    elif score < MEDIUM_RISK_MIN_SCORE:
        missed.append(f"Medium Risk butuh skor >= {MEDIUM_RISK_MIN_SCORE:.1f} (aktual: {score:.1f})")

    if not has_htf:
        missed.append(f"Low Risk butuh TF H1/H4 (aktual TF: {tf})")
    elif not is_aligned:
        missed.append(f"Low Risk butuh searah bias HTF {htf_bias}")
    elif not is_conservative_dist:
        missed.append(f"Low Risk butuh jarak <= {LOW_RISK_MAX_ATR_RATIO:.1f}x ATR & <= {LOW_RISK_MAX_DISTANCE_PT:.1f} USD (aktual: {dist:.2f} USD / {dist_ratio_str})")
    elif score < LOW_RISK_MIN_SCORE:
        missed.append(f"Low Risk butuh skor >= {LOW_RISK_MIN_SCORE:.1f} (aktual: {score:.1f})")

    if is_abnormal_width:
        missed.append(f"Lebar zona abnormal ({w:.2f} USD)")

    detail_str = "; ".join(missed) if missed else f"Skor: {score:.1f}, Grade: {grade}, Jarak: {dist:.2f} USD"
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
    market_status: Optional[str] = None,
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
    is_mkt_closed = str(market_status or zd.get("market_status") or "").upper() in ("CLOSED", "MARKET_CLOSED")

    # 1. RADAR ELIGIBILITY (Distance-Aware ATR + Struktur Gate)
    rel_eval = classify_zone_relevance(
        zd, current_price, atr=atr, htf_bias=htf_bias, h1_structure=h1_struct,
        m30_status=m30_status, frames=frames, bid=bid, ask=ask
    )
    rel_class = rel_eval["relevance"]
    radar_elig = "QUALIFIED"
    radar_reason = "Zona valid untuk pemantauan radar"
    closed_inv_h, closed_inv_rsn_h, _, _, _ = evaluate_smc_zone_closed_invalidation(zd, frames)
    if not is_valid or not origin_ok or "invalid" in rej or grade.upper() == "REJECTED":
        radar_elig = "NOT_QUALIFIED"
        radar_reason = f"Struktur zona tidak valid ({zd.get('rejection_reason') or 'INVALID'})"
    elif fresh == "INVALIDATED" or closed_inv_h or zd.get("is_broken") is True or zd.get("status") in ("INVALIDATED", "BROKEN"):
        radar_elig = "NOT_QUALIFIED"
        radar_reason = closed_inv_rsn_h or "Zona telah INVALID — BROKEN"
    elif fresh == "EXPIRED" or zd.get("status") == "EXPIRED" or zd.get("is_expired") or "expired" in rej:
        radar_elig = "NOT_QUALIFIED"
        radar_reason = "Batas waktu pemantauan zona telah habis (EXPIRED)"
    elif zd.get("touch_count", 0) > 2 or "dimitigasi" in rej or fresh == "MITIGATED" or zd.get("status") == "MITIGATED":
        radar_elig = "NOT_QUALIFIED"
        radar_reason = "Zona telah termitigasi penuh (> 2x touch)"
    elif score < 50.0:
        radar_elig = "NOT_QUALIFIED"
        radar_reason = f"Skor teknikal ({score:.1f}) di bawah ambang batas minimal 50.0"
    elif rel_class == "FAR_REFERENCE" or str(zd.get("risk_tier") or "").upper() == "FAR_ZONE":
        radar_elig = "NOT_QUALIFIED"
        radar_reason = rel_eval["reason"]

    if is_valid and not closed_inv_h and not zd.get("is_broken") and grade.upper() != "REJECTED" and fresh not in ("INVALIDATED", "MITIGATED", "EXPIRED"):
        zone_quality = f"VALID (Grade {grade} | Score {score:.1f} | {fresh})"
    else:
        zone_quality = f"REJECTED ({radar_reason})"

    # 2. PENDING ELIGIBILITY (Aturan Ketat Pending Limit — Zona FAR DILARANG!)
    pending_elig = "NOT_QUALIFIED"
    pending_reason = ""
    pending_plan = None
    plan_res = None
    validation_reasons: List[str] = []
    far_cat_h = rel_eval.get("far_category") or (
        "EXTREME_FAR"
        if (isinstance(rel_eval.get("distance_atr_ratio"), (int, float)) and float(rel_eval["distance_atr_ratio"]) > 15.0)
        else ("FAR_INTRADAY" if rel_class == "FAR_REFERENCE" else "IN_RANGE")
    )

    # Evaluasi SL struktural terlebih dahulu agar konsisten di ZONA UTAMA, KANDIDAT, dan RADAR
    sl_exceeded_msg = ""
    try:
        _ep_chk = hi if z_type == "DEMAND" else lo
        _sl_chk = compute_structural_sl(z_obj, frames)
        if _ep_chk > 0 and _sl_chk > 0:
            _rk_chk = round((_ep_chk - _sl_chk) if z_type == "DEMAND" else (_sl_chk - _ep_chk), 2)
            if _rk_chk > SL_JARAK_MAKS:
                sl_exceeded_msg = f"Jarak SL ({_rk_chk:.2f} USD) > maks {SL_JARAK_MAKS:g} USD"
    except Exception:
        pass

    if rel_class == "FAR_REFERENCE" or str(zd.get("risk_tier") or "").upper() == "FAR_ZONE":
        validation_reasons.append(rel_eval["reason"])
    if radar_elig != "QUALIFIED" and radar_reason not in validation_reasons:
        validation_reasons.append(radar_reason)
    if z_type == "DEMAND" and htf_bias != "BULLISH":
        validation_reasons.append(f"Bias H4 ({htf_bias}) tidak selaras dengan setup BUY LIMIT (harus BULLISH)")
    elif z_type == "SUPPLY" and htf_bias != "BEARISH":
        validation_reasons.append(f"Bias H4 ({htf_bias}) tidak selaras dengan setup SELL LIMIT (harus BEARISH)")
    if (htf_bias == "BULLISH" and h1_struct == "BEARISH") or (htf_bias == "BEARISH" and h1_struct == "BULLISH"):
        validation_reasons.append(f"Konflik struktur H4 ({htf_bias}) vs H1 ({h1_struct})")
    if m30_status in ("CONFLICT", "INVALID", "WEAK"):
        validation_reasons.append(f"Momentum M30 tidak mendukung pending ({m30_status})")
    if not (grade in ("A+", "A") or score >= LOW_RISK_MIN_SCORE):
        validation_reasons.append(f"Kualitas zona ({grade} / Skor {score:.1f}) di bawah standar Grade A untuk Pending Limit")
    if fresh not in ("FRESH", "TESTED_1X"):
        validation_reasons.append(f"Status zona ({fresh}) sudah termitigasi")
    if _is_pure_m5_tf(str(zd.get("timeframe") or "")) and (fresh != "FRESH" or m30_status != "CONFIRMED" or h1_struct != htf_bias):
        validation_reasons.append("Zona M5 murni membutuhkan H1 searah H4, M30 CONFIRMED, dan status FRESH untuk Pending Limit")
    if sl_exceeded_msg:
        validation_reasons.append(sl_exceeded_msg)

    if rel_class == "FAR_REFERENCE" or str(zd.get("risk_tier") or "").upper() == "FAR_ZONE":
        pending_reason = f"{rel_eval['reason']}; {sl_exceeded_msg}" if sl_exceeded_msg else rel_eval["reason"]
    elif radar_elig != "QUALIFIED":
        pending_reason = f"{radar_reason}; {sl_exceeded_msg}" if sl_exceeded_msg else radar_reason
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
    elif _is_pure_m5_tf(str(zd.get("timeframe") or "")) and (fresh != "FRESH" or m30_status != "CONFIRMED" or h1_struct != htf_bias):
        pending_reason = "Zona M5 murni membutuhkan H1 searah H4, M30 CONFIRMED, dan status FRESH untuk Pending Limit"
    else:
        # Jalankan plan_pending_order untuk kalkulasi level dan verifikasi SL / RR
        plan_res = plan_pending_order(
            z_obj, current_price, frames=frames, htf_bias=htf_bias,
            m30_status=m30_status, server_time=server_time, bid=bid, ask=ask,
            h1_structure=h1_struct, premarket=is_mkt_closed,
        )
        if plan_res.get("valid"):
            pending_plan = plan_res
            if is_mkt_closed:
                pending_elig = "WAIT"
                pending_reason = "Setup layak secara struktur & risiko, namun tetap WAIT saat MARKET CLOSED (wajib revalidasi saat market OPEN)"
            else:
                pending_elig = "READY"
                pending_reason = "Memenuhi seluruh 10 persyaratan Pending Limit (Grade A, H4/H1/M30 selaras, SL <= 20 USD, RR >= 1:1.5)"
        else:
            pending_reason = plan_res.get("rejection_reason") or sl_exceeded_msg or "Tidak memenuhi parameter validasi pending order"
            if pending_reason not in validation_reasons:
                validation_reasons.append(pending_reason)

    if pending_plan is not None:
        setup_elig = "ELIGIBLE (PROVISIONAL — MARKET CLOSED)" if is_mkt_closed else "ELIGIBLE"
    else:
        setup_elig = f"NOT_ELIGIBLE ({pending_reason})"

    # 3. CONFIRMED ENTRY ELIGIBILITY (M5 Triggers)
    confirmed_elig = "WAIT"
    confirmed_reason = ""
    if is_mkt_closed:
        confirmed_elig = "WAIT"
        confirmed_reason = "MARKET CLOSED — seluruh confirmed entry & rekomendasi pending tetap WAIT sampai revalidasi market OPEN"
    elif not setup_elig.startswith("ELIGIBLE") and ("jarak sl" in pending_reason.lower() or "melebihi batas" in pending_reason.lower()):
        confirmed_elig = "FAILED"
        confirmed_reason = pending_reason
    elif m5_stages:
        st_zt = m5_stages.get("zone_touch", {}).get("status")
        st_sw = m5_stages.get("liquidity_sweep", {}).get("status")
        st_tr = m5_stages.get("structure_trigger", {}).get("status")
        st_dp = m5_stages.get("displacement", {}).get("status")
        st_rr = m5_stages.get("risk_reward", {}).get("status")
        st_ev = m5_stages.get("entry_validation", {}).get("status")
        rr_msg = str(m5_stages.get("risk_reward", {}).get("msg") or "").strip()
        ev_msg = str(m5_stages.get("entry_validation", {}).get("msg") or "").strip()

        if st_zt not in ("CONFIRMED", "NOT_REQUIRED", "PASS", "PASSED"):
            confirmed_elig = "WAIT"
            confirmed_reason = "Harga belum menyentuh zona (Zone Touch belum terkonfirmasi)"
        elif st_sw not in ("CONFIRMED", "OPTIONAL_OK", "PASS", "PASSED"):
            confirmed_elig = "WAIT"
            confirmed_reason = "Menunggu konfirmasi Liquidity Sweep M5 di area zona"
        elif st_tr not in ("CONFIRMED", "PASS", "PASSED"):
            confirmed_elig = "WAIT"
            confirmed_reason = "Menunggu konfirmasi pembalikan/kelanjutan struktur CHoCH/BOS/CISD M5"
        elif st_dp not in ("CONFIRMED", "PASS", "PASSED"):
            confirmed_elig = "WAIT"
            confirmed_reason = "Menunggu dorongan displacement momentum M5"
        elif st_rr in ("FAILED", "FAIL", "WAIT") or st_ev in ("FAILED", "FAIL", "INVALID", "WAIT"):
            confirmed_elig = "FAILED"
            confirmed_reason = rr_msg or ev_msg or "Risk/Reward kalkulasi M5 tidak memenuhi minimum 1:1.5"
        else:
            confirmed_elig = "CONFIRMED"
            confirmed_reason = "Seluruh 5 tahap konfirmasi M5 (Touch, Sweep, CHoCH/BOS, Displacement, RR) terpenuhi"
    else:
        confirmed_elig = "WAIT"
        confirmed_reason = "Menunggu data konfirmasi trigger M5"

    if is_mkt_closed:
        entry_perm = "WAIT (MARKET CLOSED)"
    elif confirmed_elig == "CONFIRMED" and setup_elig.startswith("ELIGIBLE"):
        entry_perm = "GRANTED"
    elif confirmed_elig == "FAILED" or not setup_elig.startswith("ELIGIBLE"):
        entry_perm = f"WAIT — BLOCKED ({confirmed_reason if confirmed_elig == 'FAILED' else pending_reason})"
    else:
        entry_perm = f"WAIT ({confirmed_reason})"

    # Trade parameters: hanya tampilkan level aktif jika setup eligible (jangan tampilkan level READY jika gagal filter risiko)
    if pending_plan:
        entry_val = float(pending_plan.get("entry_price") or 0.0)
        sl_val = float(pending_plan.get("sl") or 0.0)
        tp1_val = float(pending_plan.get("tp1") or 0.0)
        tp2_val = float(pending_plan.get("tp2") or 0.0)
        risk_val = float(pending_plan.get("risk") or 0.0)
        reward_val = float(pending_plan.get("reward") or 0.0)
        rr_val = float(pending_plan.get("rr") or 0.0)
    else:
        entry_val = 0.0
        sl_val = 0.0
        tp1_val = 0.0
        tp2_val = 0.0
        risk_val = float((plan_res or {}).get("risk_points") or 0.0) if isinstance(plan_res, dict) else 0.0
        reward_val = 0.0
        rr_val = float((plan_res or {}).get("rr") or 0.0) if isinstance(plan_res, dict) else 0.0

    return {
        "zone_id": zd.get("zone_id", ""),
        "direction": direction,
        "far_category": far_cat_h,
        "validation_reasons": validation_reasons,
        "zone_quality": zone_quality,
        "setup_eligibility": setup_elig,
        "entry_permission": entry_perm,
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
    market_status: Optional[str] = None,
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
            "far_reference_zones": [],
            "far_reference_zone": None,
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
        closed_inv_p, closed_rsn_p, _, _, _ = evaluate_smc_zone_closed_invalidation(zd, frames)
        live_inv_p, live_rsn_p, live_px_p, live_inv_lvl_p, _ = evaluate_smc_setup_live_invalidation(
            zd, current_price=current_price, bid=bid, ask=ask, frames=frames, market_status=market_status,
        )
        if live_inv_lvl_p and live_inv_lvl_p > 0:
            zd["setup_invalidation_price"] = live_inv_lvl_p
        is_mkt_closed = str(market_status or "").upper() in ("CLOSED", "MARKET_CLOSED")
        if (
            not is_valid or not origin_ok or "invalid" in rejection.lower()
            or fresh == "INVALIDATED" or closed_inv_p or live_inv_p
            or zd.get("is_broken") is True
            or str(zd.get("status") or "").upper() in ("INVALIDATED", "BROKEN")
            or str(zd.get("invalidation_type") or "").upper() in (INVALIDATION_SETUP_SL_BREACHED, INVALIDATION_ZONE_STRUCTURAL)
        ):
            zone_status = "INVALIDATED"
            zd["is_valid"] = False
            zd["freshness"] = "INVALIDATED"
            zd["zone_relevance"] = "INVALID"
            if closed_inv_p or zd.get("is_broken") is True or str(zd.get("invalidation_type") or "").upper() == INVALIDATION_ZONE_STRUCTURAL:
                zd["is_broken"] = True
                zd["invalidation_type"] = INVALIDATION_ZONE_STRUCTURAL
                zd["invalid_label"] = INVALIDATION_ZONE_STRUCTURAL
                if closed_rsn_p and not zd.get("rejection_reason"):
                    zd["rejection_reason"] = closed_rsn_p
            elif live_inv_p or str(zd.get("invalidation_type") or "").upper() == INVALIDATION_SETUP_SL_BREACHED:
                zd["invalidation_type"] = INVALIDATION_SETUP_SL_BREACHED
                zd["invalid_label"] = INVALIDATION_SETUP_SL_BREACHED
                if live_rsn_p:
                    zd["rejection_reason"] = live_rsn_p
                if live_px_p is not None:
                    zd["break_price"] = live_px_p
            else:
                zd["is_broken"] = True
                zd["invalidation_type"] = zd.get("invalidation_type") or INVALIDATION_ZONE_STRUCTURAL
                zd["invalid_label"] = zd["invalidation_type"]
        elif zd.get("touch_count", 0) > 2 or fresh == "MITIGATED" or "dimitigasi" in rejection.lower():
            zone_status = "MITIGATED"
        elif approach in ("TOUCHED", "ZONE_TOUCHED"):
            zone_status = ZoneStatus("ZONE_TOUCHED")
        elif approach == "APPROACHING":
            if is_mkt_closed:
                zone_status = "RADAR"
                approach = "UNTOUCHED"
                zd["approach"] = "UNTOUCHED"
            else:
                zone_status = "APPROACHING"
        else:
            zone_status = "RADAR"
        zd["status"] = zone_status

        # 1.5 Klasifikasi Distance-Aware Relevance SEBELUM pemilihan zona utama (ACTIVE / RADAR / FAR_REFERENCE / INVALID)
        rel_info = classify_zone_relevance(
            zd, current_price, atr=atr_val, htf_bias=htf_bias,
            h1_structure=h1_structure, m30_status=m30_status,
            frames=frames, bid=bid, ask=ask,
        )
        zone_relevance = rel_info["relevance"]
        zd["zone_relevance"] = zone_relevance
        zd["zone_relevance_reason"] = rel_info["reason"]
        if zone_relevance == "FAR_REFERENCE":
            approach = "FAR"
            zd["approach"] = "FAR"

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

        # 3. Skor Prioritas Komprehensif (Kualitas Struktur + HTF + Freshness; TANPA inflasi skor hanya karena dekat harga)
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

            # Penalti jarak proporsional volatilitas tanpa menaikkan skor zona lemah hanya karena dekat harga
            if zone_relevance == "FAR_REFERENCE" or approach == "FAR":
                p_score -= 40.0
            elif zone_relevance == "RADAR":
                if dist_ratio is not None:
                    p_score -= min(12.0, max(0.0, dist_ratio - 2.0) * 1.2)
                else:
                    p_score -= min(12.0, dist * 0.15)

            if grd == "A+":
                p_score += 10.0
            elif grd == "A":
                p_score += 5.0

            if score_st == "DATA_INCOMPLETE":
                p_score -= 100.0

        # Label prioritas
        labels = [str(zone_status)]
        if zone_relevance not in labels:
            labels.append(zone_relevance)
        if approach == "FAR" and "FAR" not in labels:
            labels.append("FAR")
        if is_counter:
            labels.append("COUNTERTREND")

        # Risk Tier Classification (Three-Tier Engine)
        tier = classify_zone_risk_tier(
            zd, current_price, htf_bias, h1_structure,
            atr=atr_val, bid=bid, ask=ask
        )
        if zone_relevance == "FAR_REFERENCE" and tier != "DATA_INCOMPLETE":
            tier = "FAR_ZONE"
            zd["risk_tier_reason"] = rel_info["reason"]
        zd["risk_tier"] = tier

        # Hybrid Entry Eligibility (Radar, Pending, Confirmed M5)
        h_eval = evaluate_zone_hybrid_eligibility(
            zd, current_price, frames=frames, htf_bias=htf_bias,
            h1_struct=h1_structure, m30_status=m30_status,
            m5_stages=(m5_stages if m5_stages and m5_stages.get("_zone_id") and m5_stages["_zone_id"] == zd.get("zone_id") else None),
            server_time=server_time, atr=atr_val,
            bid=bid, ask=ask, market_status=market_status,
        )
        # Jika zona tidak memenuhi kategori High/Medium/Low tetapi memenuhi kriteria pending independen,
        # tetapkan klasifikasi sebagai UNTIERED PENDING tanpa membatalkan status READY (dilarang untuk FAR_REFERENCE).
        if tier == "UNCLASSIFIED" and zone_relevance != "FAR_REFERENCE" and h_eval.get("pending_eligibility") == "READY":
            tier = "UNTIERED PENDING"
            zd["risk_tier"] = tier
            zd["risk_tier_reason"] = (
                f"Untiered Pending ({zd.get('timeframe')}): Memenuhi kriteria pending independen "
                f"(Grade {zd.get('grade')}, Skor {zd.get('score', 0.0):.1f}, Fresh) di luar batasan High/Medium/Low Risk tier"
            )

        zd["zone_quality"] = h_eval["zone_quality"]
        zd["setup_eligibility"] = h_eval["setup_eligibility"]
        zd["entry_permission"] = h_eval["entry_permission"]
        zd["far_category"] = h_eval.get("far_category") or rel_info.get("far_category") or ("IN_RANGE" if zone_relevance != "FAR_REFERENCE" else "FAR_INTRADAY")
        zd["validation_reasons"] = h_eval.get("validation_reasons") or []
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

    # 4. Pemisahan 4 Kategori Evaluasi (Filter Distance-Aware ATR SEBELUM seleksi utama!):
    # A. PRIMARY DIRECTIONAL ZONE : Kualitas terbaik dalam jangkauan (ACTIVE/RADAR) selaras bias HTF (H4/H1)
    # B. NEAREST RADAR ZONE       : Zona valid dalam jangkauan terdekat ke snapshot harga (tanpa memandang arah)
    # C. BEST VALIDATED ENTRY ZONE: Terverifikasi penuh (Trigger M5 + R:R), NONE jika belum lengkap
    # D. ALTERNATIVE ZONE         : Zona cadangan pro-trend non-overlap berikutnya atau area reaksi
    h1_aligned_flag = (htf_bias in ("BULLISH", "BEARISH") and h1_structure == htf_bias)

    def _in_range_valid_pred(z: Dict[str, Any]) -> bool:
        return (
            is_zone_active_and_valid(z, current_price)
            and not z.get("is_broken")
            and str(z.get("status", "")).upper() not in ("INVALIDATED", "BROKEN", "EXPIRED", "MITIGATED")
            and str(z.get("zone_relevance", "")).upper() != "FAR_REFERENCE"
            and str(z.get("risk_tier", "")).upper() != "FAR_ZONE"
            and not is_far_reference_zone(z, current_price=current_price, atr=atr_val)
        )

    def _primary_intraday_rank(z: Dict[str, Any]) -> Tuple:
        is_far = (
            z.get("zone_relevance") == "FAR_REFERENCE"
            or z.get("risk_tier") == "FAR_ZONE"
            or z.get("approach") == "FAR"
            or float(z.get("distance", 0.0)) > far_threshold
        )
        rel_is_active = (z.get("zone_relevance") == "ACTIVE") and not is_far
        qual_ok = (
            z.get("grade") in ("A+", "A")
            or (float(z.get("score", 0.0)) >= 70.0 and z.get("freshness") in ("FRESH", "TESTED_1X"))
        )
        if is_far:
            bucket = 4  # FAR_REFERENCE: dilarang mendominasi zona ACTIVE / RADAR
        elif rel_is_active and qual_ok:
            bucket = 0
        elif rel_is_active:
            bucket = 1
        elif qual_ok:
            bucket = 2
        else:
            bucket = 3
        fresh_r = {"FRESH": 0, "TESTED_1X": 1, "TESTED_2X": 2}.get(z.get("freshness", ""), 9)
        tf_r = _intraday_tf_rank(z.get("timeframe", ""), m30_status=m30_status, h1_aligned=h1_aligned_flag)
        return (
            bucket,
            fresh_r,
            round(float(z.get("distance", 9999.0)), 2),
            tf_r,
            -round(float(z.get("priority_score", 0.0)), 1),
            -round(float(z.get("score", 0.0)), 1),
        )

    if htf_bias in ("BULLISH", "BEARISH"):
        aligned_raw = [z for z in processed if z["is_aligned"] and _in_range_valid_pred(z)]
        counter_raw = [z for z in processed if z["is_counter"] and _in_range_valid_pred(z)]
        aligned_raw.sort(key=_primary_intraday_rank)
        counter_raw.sort(key=lambda x: (x["distance"], -float(x.get("score", 0.0))))
        aligned, aligned_dups = _deduplicate_overlapping_zones(aligned_raw)
        counter, counter_dups = _deduplicate_overlapping_zones(counter_raw)
        primary_directional = aligned[0] if aligned else None
        if primary_target_override:
            ov_dict = primary_target_override if isinstance(primary_target_override, dict) else primary_target_override.to_dict()
            ov_match = next((z for z in aligned if z.get("zone_id") == ov_dict.get("zone_id") or
                            (abs(float(z.get("price_low", 0.0)) - float(ov_dict.get("price_low", 0.0))) < 1e-4 and
                             abs(float(z.get("price_high", 0.0)) - float(ov_dict.get("price_high", 0.0))) < 1e-4)), None)
            if ov_match:
                primary_directional = ov_match
            elif isinstance(ov_dict, dict) and ov_dict.get("price_low") is not None and not is_far_reference_zone(ov_dict, current_price=current_price, atr=atr_val):
                primary_directional = ov_dict
        secondary_aligned = next((z for z in aligned if z is not primary_directional and (not primary_directional or not _zones_substantially_overlap(primary_directional, z))), None)
        countertrend_nearest = counter[0] if counter else None
        primary_fallback = primary_directional
    else:  # NETRAL
        aligned_dups = []
        counter_dups = []
        demands = [z for z in processed if z["zone_type"] == "DEMAND" and _in_range_valid_pred(z)]
        supplies = [z for z in processed if z["zone_type"] == "SUPPLY" and _in_range_valid_pred(z)]
        demands.sort(key=lambda x: (x["distance"], -float(x.get("score", 0.0))))
        supplies.sort(key=lambda x: (x["distance"], -float(x.get("score", 0.0))))
        demands, dem_dups = _deduplicate_overlapping_zones(demands)
        supplies, sup_dups = _deduplicate_overlapping_zones(supplies)
        aligned_dups = dem_dups + sup_dups
        primary_directional = None  # Pada HTF Netral, tidak ada directional bias
        primary_fallback = supplies[0] if supplies else (demands[0] if demands else None)
        secondary_aligned = demands[0] if (supplies and demands) else (supplies[1] if len(supplies) > 1 else None)
        countertrend_nearest = None
        aligned = sorted(demands + supplies, key=lambda x: -float(x.get("priority_score", 0.0)))
        counter = []

    # Tandai duplikat tumpang tindih agar konsisten antara scan & pantau (tanpa menghapus jejak audit)
    suppressed_dup_ids = set()
    for dup_z, win_z in (aligned_dups + counter_dups):
        dup_id = dup_z.get("zone_id")
        if dup_id:
            suppressed_dup_ids.add(dup_id)
        dup_z["radar_eligibility"] = "NOT_QUALIFIED"
        dup_z["radar_rejection_reason"] = (
            f"DUPLICATE_OVERLAP: Tumpang tindih dengan zona prioritas {win_z.get('timeframe')} "
            f"{win_z.get('zone_type')} ({win_z.get('price_low', 0.0):.2f}-{win_z.get('price_high', 0.0):.2f})"
        )
        dup_z["pending_eligibility"] = "NOT_QUALIFIED"
        dup_z["pending_rejection_reason"] = dup_z["radar_rejection_reason"]

    # B. NEAREST RADAR ZONE (Zona valid dalam jangkauan dengan jarak terkecil — terpisah dari Primary Directional)
    valid_cands = [
        z for z in processed
        if _in_range_valid_pred(z) and z.get("zone_id") not in suppressed_dup_ids
    ]
    if valid_cands:
        sorted_by_dist = sorted(valid_cands, key=lambda x: (x["distance"], -float(x.get("score", 0.0))))
        nearest_radar = sorted_by_dist[0]
    else:
        nearest_radar = None

    # C. BEST VALIDATED ENTRY ZONE (Hanya terisi jika SELURUH syarat validasi entry M5 terpenuhi & bukan FAR)
    best_validated_entry = None
    if validated_entry_zone:
        vz_dict = validated_entry_zone if isinstance(validated_entry_zone, dict) else validated_entry_zone.to_dict()
        if not is_far_reference_zone(vz_dict, current_price=current_price, atr=atr_val):
            best_validated_entry = vz_dict

    # D. ALTERNATIVE ZONE (non-overlap dengan Primary)
    alternative_zone = secondary_aligned if secondary_aligned is not None else countertrend_nearest

    # Backwards compatibility: Best Entry Monitoring Zone (tanpa inflasi skor kedekatan)
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
            rel_v = z.get("zone_relevance", "RADAR")
            d_r = z.get("distance_atr_ratio")
            d = float(z.get("distance", 0.0))
            if rel_v == "FAR_REFERENCE":
                m_score -= 40.0
            elif rel_v == "RADAR":
                m_score -= min(12.0, float(d_r) * 1.2) if isinstance(d_r, (int, float)) else min(12.0, d * 0.15)
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
        dist_parts.append(f"{primary_directional['distance']:.2f} USD ke {p_lbl} ({primary_directional['price_low']:.2f}-{primary_directional['price_high']:.2f} [{primary_directional['approach']}])")
    elif primary:
        p_lbl = "Primary " + primary["zone_type"].capitalize()
        dist_parts.append(f"{primary['distance']:.2f} USD ke {p_lbl} ({primary['price_low']:.2f}-{primary['price_high']:.2f} [{primary['approach']}])")
    if secondary:
        s_lbl = "Secondary " + secondary["zone_type"].capitalize()
        dist_parts.append(f"{secondary['distance']:.2f} USD ke {s_lbl} ({secondary['price_low']:.2f}-{secondary['price_high']:.2f} [{secondary['approach']}])")
    if countertrend:
        ct_lbl = "Countertrend " + countertrend["zone_type"].capitalize()
        dist_parts.append(f"{countertrend['distance']:.2f} USD ke {ct_lbl} ({countertrend['price_low']:.2f}-{countertrend['price_high']:.2f} [{countertrend['approach']}])")
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
            missing.append(f"[ZONE_TOUCH]: Harga belum mencapai atau menyentuh {eval_target['zone_type'].capitalize()} ({eval_target['price_low']:.2f}-{eval_target['price_high']:.2f}, jarak {eval_target['distance']:.2f} USD)")
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
            st_ev = (m5_stages or {}).get("entry_validation", {}).get("status")
            rr_msg_rad = str((m5_stages or {}).get("risk_reward", {}).get("msg") or "").strip()
            ev_msg_rad = str((m5_stages or {}).get("entry_validation", {}).get("msg") or "").strip()
            block_detail_rad = rr_msg_rad or ev_msg_rad
            if st_rr in ("CONFIRMED", "PASS", "PASSED") and st_ev not in ("FAILED", "FAIL", "INVALID", "WAIT"):
                confirmed_entry_rr_status = "PASSED"
            elif m5_triggers_ready:
                if st_rr in ("FAILED", "FAIL", "WAIT") or st_ev in ("FAILED", "FAIL", "INVALID", "WAIT"):
                    confirmed_entry_rr_status = "FAILED"
                    missing.append(
                        f"[CONFIRMED_ENTRY_RISK_REWARD]: {block_detail_rad}"
                        if block_detail_rad
                        else "[CONFIRMED_ENTRY_RISK_REWARD]: Entry konfirmasi M5 tidak memenuhi Risk/Reward minimum 1:1.5"
                    )
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
            st_ev = (m5_stages or {}).get("entry_validation", {}).get("status")
            rr_m_blk = str((m5_stages or {}).get("risk_reward", {}).get("msg") or "").strip()
            ev_m_blk = str((m5_stages or {}).get("entry_validation", {}).get("msg") or "").strip()
            blk_m = rr_m_blk or ev_m_blk

            if st_sw not in ("CONFIRMED", "OPTIONAL_OK", "PASS", "PASSED"):
                sw_msg = (m5_stages or {}).get("liquidity_sweep", {}).get("msg", "")
                if "jauh" in str(sw_msg).lower():
                    primary_blocker = f"Harga telah menyentuh {p_name} ({p_range}), namun Liquidity Sweep M5 terjadi tidak relevan (terlalu jauh dari area zona). Menunggu sweep yang valid."
                else:
                    primary_blocker = f"Harga telah memasuki {p_name} ({p_range}), tetapi belum ada konfirmasi Liquidity Sweep M5 yang valid di area zona."
                dynamic_wait = f"WAIT: {primary_blocker}{ct_note}"
            elif st_tr not in ("CONFIRMED", "PASS", "PASSED"):
                tr_m = str((m5_stages or {}).get("structure_trigger", {}).get("msg") or "").strip()
                primary_blocker = (
                    tr_m
                    if ("berbalik" in tr_m.lower() or "historis" in tr_m.lower())
                    else f"Harga telah menyentuh {p_name} dan membentuk Liquidity Sweep M5, tetapi belum terkonfirmasi pembalikan/kelanjutan struktur (CHoCH/BOS M5)."
                )
                dynamic_wait = f"WAIT: {primary_blocker}{ct_note}"
            elif st_dp not in ("CONFIRMED", "PASS", "PASSED"):
                primary_blocker = f"Sweep dan CHoCH M5 terkonfirmasi pada {p_name}, namun displacement M5 masih lemah."
                dynamic_wait = f"WAIT: {primary_blocker}{ct_note}"
            elif st_rr in ("FAILED", "FAIL", "WAIT") or st_ev in ("FAILED", "FAIL", "INVALID", "WAIT"):
                if blk_m and any(k in blk_m.lower() for k in ("snr", "s&r", "konflik", "terhalang", "historis", "terlewat")):
                    primary_blocker = f"Konfirmasi teknikal M5 pada {p_name} terpenuhi, namun entry ditahan: {blk_m}"
                else:
                    primary_blocker = f"Konfirmasi trigger M5 pada {p_name} terpenuhi, tetapi kalkulasi Risk/Reward belum mencapai minimum 1:1.5."
                dynamic_wait = f"WAIT: {primary_blocker}{ct_note}"
            else:
                primary_blocker = f"Menunggu penyelesaian validasi final pada {p_name}."
                dynamic_wait = f"WAIT: {primary_blocker}"
        else:
            app_str = "sudah mendekati" if p_app == "APPROACHING" else ("berjarak jauh dari" if p_app == "FAR" else "belum mencapai")
            primary_blocker = f"Bias H4/H1 {htf_bias.lower()}, tetapi harga {app_str} {p_type.capitalize()} RADAR ({p_name} @ {p_range}, jarak {p_dist:.2f} USD). Belum ada validasi sweep dan struktur M5."
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
    htf_conflict_blk = (htf_bias == "BULLISH" and h1_structure == "BEARISH") or (htf_bias == "BEARISH" and h1_structure == "BULLISH")
    if htf_conflict_blk:
        pending_blockers.append(f"Konflik struktur H4 ({htf_bias}) vs H1 ({h1_structure})")
        confirmed_blockers.append(
            f"[HTF_CONFLICT]: Konflik struktur H4 ({htf_bias}) vs H1 ({h1_structure}) — belum ada setup eligible sehingga trigger M5 tidak dijalankan"
        )
    if m30_status in ("CONFLICT", "INVALID", "WEAK"):
        pending_blockers.append(f"Momentum M30 tidak mendukung pending ({m30_status})")
        if not htf_conflict_blk:
            confirmed_blockers.append(
                f"[M30_BLOCKER]: Momentum/struktur M30 ({m30_status}) belum selaras dengan H4 ({htf_bias}) — trigger M5 tidak dijalankan"
            )

    if eval_target:
        if eval_target.get("grade") not in ("A+", "A"):
            pending_blockers.append(f"Kualitas zona ({eval_target.get('grade')} / Skor {eval_target.get('score', 0.0):.1f}) di bawah standar Grade A (skor >= 75.0)")
        if eval_target.get("freshness") not in ("FRESH", "TESTED_1X"):
            pending_blockers.append(f"Status zona ({eval_target.get('freshness')}) sudah termitigasi")
        p_rsn = eval_target.get("pending_rejection_reason")
        if p_rsn and p_rsn not in pending_blockers and "memenuhi" not in str(p_rsn).lower():
            pending_blockers.append(p_rsn)

        # Confirmed M5 blockers: Berurutan, tidak menampilkan sweep/trigger gagal jika zone belum tersentuh atau HTF/M30 konflik
        if not htf_conflict_blk and m30_status not in ("CONFLICT", "INVALID", "WEAK"):
            st_zt = (m5_stages or {}).get("zone_touch", {}).get("status")
            st_sw = (m5_stages or {}).get("liquidity_sweep", {}).get("status")
            st_tr = (m5_stages or {}).get("structure_trigger", {}).get("status")
            st_dp = (m5_stages or {}).get("displacement", {}).get("status")
            st_rr = (m5_stages or {}).get("risk_reward", {}).get("status")
            st_ev = (m5_stages or {}).get("entry_validation", {}).get("status")
            rr_m_blk = str((m5_stages or {}).get("risk_reward", {}).get("msg") or "").strip()
            ev_m_blk = str((m5_stages or {}).get("entry_validation", {}).get("msg") or "").strip()
            blk_m = rr_m_blk or ev_m_blk

            is_touched = (eval_target.get("approach") in ("TOUCHED", "ZONE_TOUCHED") or st_zt in ("CONFIRMED", "PASS", "PASSED"))
            if not is_touched:
                confirmed_blockers.append(f"[ZONE_TOUCH]: Harga belum mencapai atau menyentuh {eval_target['zone_type'].capitalize()} ({eval_target['price_low']:.2f}-{eval_target['price_high']:.2f}, jarak {eval_target['distance']:.2f} USD) — konfirmasi M5 menunggu zone touch")
            else:
                if st_sw not in ("CONFIRMED", "OPTIONAL_OK", "PASS", "PASSED"):
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
                elif st_rr in ("FAILED", "FAIL", "WAIT") or st_ev in ("FAILED", "FAIL", "INVALID", "WAIT"):
                    confirmed_blockers.append(
                        f"[RISK_REWARD]: {blk_m}"
                        if blk_m
                        else "[RISK_REWARD]: Kalkulasi Risk/Reward belum mencapai minimum 1:1.5"
                    )
    else:
        pending_blockers.append(f"Tidak ada zona Grade A yang selaras dengan bias HTF ({htf_bias})")
        if not htf_conflict_blk and m30_status not in ("CONFLICT", "INVALID", "WEAK"):
            confirmed_blockers.append(f"Tidak ada zona aktif yang selaras dengan arah bias HTF ({htf_bias}) — trigger M5 tidak dijalankan")

    if eval_target and eval_target.get("approach") == "FAR":
        sec_warnings.append(f"Zona prioritas berjarak jauh ({eval_target['distance']:.2f} USD / {eval_target.get('distance_atr_ratio', 0.0)}x ATR).")
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
                f"{z_name}: Diprioritaskan sebagai Primary Directional Zone karena searah bias {htf_bias}, Grade {rz['grade']} (Skor {score_val_str}), {rz['freshness']}, jarak {rz['distance']:.2f} USD ({rz['direction_status']})."
            )
        elif rz.get("is_counter"):
            unprioritized_reasons.append(
                f"{z_name}: Tidak diprioritaskan untuk entry karena berlawanan dengan bias HTF ({htf_bias}) — hanya area reaksi/support."
            )
        elif rz.get("approach") == "FAR":
            unprioritized_reasons.append(
                f"{z_name}: Belum diprioritaskan untuk entry dekat karena jaraknya jauh ({rz['distance']:.2f} USD)."
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
                        f"namun jarak lebih jauh ({r_dist:.2f} USD vs {p_dist:.2f} USD dari {p_name}) "
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
                        f"({r_dist:.2f} USD vs {p_dist:.2f} USD) dibanding {p_name}."
                    )
                else:
                    reason = (
                        f"{z_name}: Skor teknikal ({score_val_str}) berada di bawah {p_name} ({p_score:.1f})."
                    )
                unprioritized_reasons.append(reason)

    # 6. ATURAN PRIORITAS VALIDASI (Requirement 1, 2, 3):
    # Zona INVALIDATED / EXPIRED / MITIGATED / FAR_REFERENCE / DUPLICATE_OVERLAP dilarang masuk Radar aktif & Multi-Tier aktif!
    radar_qualified = [
        z for z in processed
        if is_zone_active_and_valid(z, current_price)
        and z.get("radar_eligibility") == "QUALIFIED"
        and z.get("risk_tier") != "FAR_ZONE"
        and z.get("zone_relevance") in ("ACTIVE", "RADAR")
    ]

    # Zona historis/invalid untuk bagian pelaporan terpisah (Requirement 1 & 2)
    def _is_genuinely_broken_zone(z_cand: Dict[str, Any]) -> bool:
        if not isinstance(z_cand, dict):
            return False
        st_u = str(z_cand.get("lifecycle_status") or z_cand.get("status") or "").upper()
        fr_u = str(z_cand.get("freshness") or "").upper()
        inv_t = str(z_cand.get("invalidation_type") or z_cand.get("invalid_label") or "").upper()
        if inv_t in (INVALIDATION_SETUP_SL_BREACHED, INVALIDATION_ZONE_STRUCTURAL):
            return True
        if not (z_cand.get("is_broken") is True or st_u in ("INVALIDATED", "BROKEN") or fr_u == "INVALIDATED"):
            return False
        if st_u in ("REJECTED", "MITIGATED", "EXPIRED"):
            return False
        rej_u = str(z_cand.get("rejection_reason") or "").lower()
        if any(
            kw in rej_u
            for kw in (
                "tidak ada fvg",
                "displacement tidak memadai",
                "bos tipis",
                "bos terlalu jauh",
                "tidak ada candle impuls",
                "zona terlalu lebar",
                "zona terlalu sempit",
                "zona sudah dimitigasi",
                "gagal hard validation",
                "skor ",
            )
        ):
            return False
        return True

    invalidated_historical = [
        z for z in processed
        if not is_zone_active_and_valid(z, current_price) and _is_genuinely_broken_zone(z)
    ]
    if rejected_zones:
        for rz in rejected_zones:
            rz_dict = rz.to_dict() if hasattr(rz, "to_dict") else dict(rz)
            if not is_zone_active_and_valid(rz_dict, current_price) and _is_genuinely_broken_zone(rz_dict):
                if not any(iz.get("zone_id") == rz_dict.get("zone_id") for iz in invalidated_historical):
                    invalidated_historical.append(rz_dict)

    # 1. HIGH RISK — AGGRESSIVE ZONE (Prioritaskan M5/M15/M30, kualitas & freshness, tanpa duplikat overlap)
    high_risk_candidates = [z for z in radar_qualified if z.get("risk_tier") == "HIGH_RISK"]
    high_risk_candidates.sort(key=lambda x: (
        0 if x.get("is_aligned") else 1,
        0 if x.get("freshness") == "FRESH" else 1,
        _intraday_tf_rank(x.get("timeframe", ""), m30_status=m30_status, h1_aligned=h1_aligned_flag),
        -float(x.get("score", 0.0)),
        x.get("distance", 999.0),
    ))
    high_risk_candidates, _ = _deduplicate_overlapping_zones(high_risk_candidates)
    high_risk_zones = high_risk_candidates[:MULTI_TIER_MAX_ZONES]

    # 2. MEDIUM RISK — BALANCED ZONE (Keseimbangan kualitas struktur & jarak harga)
    medium_risk_candidates = [z for z in radar_qualified if z.get("risk_tier") == "MEDIUM_RISK"]
    medium_risk_candidates.sort(key=lambda x: (
        0 if x.get("is_aligned") else 1,
        0 if x.get("freshness") == "FRESH" else 1,
        -float(x.get("score", 0.0)),
        x.get("distance", 999.0),
    ))
    medium_risk_candidates, _ = _deduplicate_overlapping_zones(medium_risk_candidates)
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
    low_risk_candidates, _ = _deduplicate_overlapping_zones(low_risk_candidates)
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

    # E. Entry Candidate Ranking (Hanya kandidat lolos Pending READY atau Confirmed CONFIRMED — bebas FAR & overlap)
    pending_ready = [
        z for z in radar_qualified
        if z.get("pending_eligibility") == "READY"
        and z.get("zone_relevance") in ("ACTIVE", "RADAR")
        and z.get("risk_tier") != "FAR_ZONE"
    ]
    if pending_ready:
        pending_ready.sort(key=lambda x: (
            0 if x.get("zone_relevance") == "ACTIVE" else 1,
            0 if x.get("freshness") == "FRESH" else 1,
            _intraday_tf_rank(x.get("timeframe", ""), m30_status=m30_status, h1_aligned=h1_aligned_flag),
            0 if x.get("grade") == "A+" else 1,
            -float(x.get("score", 0.0)),
            x.get("distance", 999.0),
        ))
        pending_ready, _ = _deduplicate_overlapping_zones(pending_ready)
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
    unclassified_tier_zones = [z for z in radar_qualified if z.get("risk_tier") in ("UNCLASSIFIED", "UNTIERED PENDING", "DATA_INCOMPLETE")]
    backup_tier_zones = (
        high_risk_candidates[MULTI_TIER_MAX_ZONES:] +
        medium_risk_candidates[MULTI_TIER_MAX_ZONES:] +
        low_risk_candidates[MULTI_TIER_MAX_ZONES:]
    )
    far_zones = []
    _seen_far_ids = set()
    for z in processed:
        if z.get("zone_relevance") == "FAR_REFERENCE" or z.get("risk_tier") == "FAR_ZONE":
            if is_zone_active_and_valid(z, current_price):
                fz_id = z.get("zone_id") or f"{z.get('timeframe')}_{z.get('zone_type')}_{z.get('price_low')}_{z.get('price_high')}"
                if fz_id not in _seen_far_ids:
                    _seen_far_ids.add(fz_id)
                    far_zones.append(z)
    far_zones.sort(key=lambda x: (0 if x.get("is_aligned") else 1, float(x.get("distance", 9999.0)), -float(x.get("score", 0.0))))
    far_aligned_ref = [z for z in far_zones if z.get("is_aligned")]
    far_reference_zone = far_aligned_ref[0] if far_aligned_ref else (far_zones[0] if far_zones else None)
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
            f"{z.get('timeframe')} {z.get('zone_type')} ({z.get('price_low', 0):.2f}-{z.get('price_high', 0):.2f}): {z.get('risk_tier_reason') or 'Jarak zona melebihi batas intraday maksimal (> 25.0 ATR atau > 75.0 USD)'}"
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
        "far_reference_zones": far_zones,
        "far_reference_zone": far_reference_zone,
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
                      symbol: str = SYMBOL, snapshot: Optional[Dict[str, Any]] = None,
                      market_status: Optional[str] = None) -> Dict[str, Any]:
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
    eff_mkt_status = str(
        market_status
        or (snapshot.get("market_status") if isinstance(snapshot, dict) else "")
        or "UNKNOWN"
    ).upper()
    eff_conn_status = str(
        (snapshot.get("connection_status") if isinstance(snapshot, dict) else "")
        or "UNKNOWN"
    ).upper()
    out: Dict[str, Any] = {
        "symbol": symbol, "timestamp": format_wib_timestamp(default_now=True), "current_price": current_price,
        "snapshot_id": snapshot_id,
        "valid_zones": [], "active_zones": [], "radar_zones": [], "detected_zones": [], "rejected_zones": [],
        "best_demand": None, "best_supply": None, "htf_context": {}, "empty_reason": "", "summary_text": "",
        "diagnostics": {}, "diagnostic_text": "", "data_ok": False,
        "data_freshness": "NO DATA", "data_freshness_info": {}, "last_candle_times": {}, "forming_candles": {},
        "market_status": eff_mkt_status, "connection_status": eff_conn_status,
        "htf_bias": "NETRAL", "h1_structure": "NETRAL", "m30_status": "WAIT", "htf_status": "UNCONFIRMED",
    }
    if isinstance(snapshot, dict) and snapshot.get("snapshot_status") == "DATA_INCOMPLETE":
        out["snapshot_status"] = "DATA_INCOMPLETE"
        out["data_freshness"] = snapshot.get("data_freshness") or "INVALID"
        out["data_freshness_info"] = snapshot.get("data_freshness_info") or {}
        out["last_candle_times"] = snapshot.get("last_candle_times") or {}
        out["forming_candles"] = snapshot.get("forming_candles") or {}
        out["empty_reason"] = snapshot.get("error_reason") or "DATA_INCOMPLETE: Snapshot MT5 Bid/Ask tidak lengkap atau tidak valid"
        out["summary_text"] = "Snapshot MT5 tidak lengkap (DATA_INCOMPLETE)."
        return out
    if frames is None and isinstance(snapshot, dict) and "frames" in snapshot:
        frames = snapshot.get("frames")
        if frames is None:
            out["snapshot_status"] = snapshot.get("snapshot_status") or "DATA_INCOMPLETE"
            out["data_freshness"] = snapshot.get("data_freshness") or "NO DATA"
            out["data_freshness_info"] = snapshot.get("data_freshness_info") or {}
            out["empty_reason"] = snapshot.get("error_reason") or "DATA_INCOMPLETE: Snapshot tidak memiliki data OHLC"
            out["summary_text"] = "Data snapshot tidak tersedia (WAIT)."
            return out
    if frames is None:
        frames = _load_frames()
        if frames is None:
            out["data_freshness"] = "NO DATA"
            out["connection_status"] = "DISCONNECTED"
            out["empty_reason"] = "MT5 tidak aktif; data tidak dapat diambil"
            out["summary_text"] = "MT5 offline - scanner tidak berjalan (WAIT)."
            return out
        if eff_conn_status == "UNKNOWN":
            eff_conn_status = "CONNECTED"
            out["connection_status"] = eff_conn_status

    need = 2 * SWING_N + 20
    for tf in ("H4", "H1"):
        df_tf = frames.get(tf)
        if df_tf is None or len(df_tf) < need:
            out["data_freshness"] = "NO DATA" if df_tf is None or len(df_tf) == 0 else "INVALID"
            out["empty_reason"] = f"Data {tf} tidak mencukupi (minimal {need} candle)"
            out["summary_text"] = f"Data {tf} belum cukup (WAIT)."
            return out
        # Cek integritas data OHLC & ATR
        for col in ("Open", "High", "Low", "Close"):
            if col not in df_tf.columns or not np.isfinite(df_tf[col].values).all():
                out["data_freshness"] = "INVALID"
                out["empty_reason"] = f"Data {tf} korup (terdapat nilai non-finite pada kolom {col})"
                out["summary_text"] = f"Data {tf} tidak valid (WAIT)."
                return out
        if ((df_tf["High"] < df_tf["Low"]).any() or (df_tf["High"] < df_tf["Open"]).any() or
                (df_tf["High"] < df_tf["Close"]).any() or (df_tf["Low"] > df_tf["Open"]).any() or
                (df_tf["Low"] > df_tf["Close"]).any()):
            out["data_freshness"] = "INVALID"
            out["empty_reason"] = f"Data {tf} korup (struktur OHLC tidak konsisten High/Low)"
            out["summary_text"] = f"Data {tf} tidak valid (WAIT)."
            return out
        if "ATR14" in df_tf.columns:
            atr_chk = float(df_tf["ATR14"].iloc[-1])
            if not math.isfinite(atr_chk) or atr_chk <= 0:
                out["data_freshness"] = "INVALID"
                out["empty_reason"] = f"ATR {tf} tidak valid ({atr_chk})"
                out["summary_text"] = f"ATR {tf} tidak valid (WAIT)."
                return out

    # Sinkronisasi Snapshot MT5 Tunggal & Immutable (Live Tick vs Candle Close)
    u_spec_scan = extract_unit_spec_from_container(snapshot, symbol=symbol)
    d_scan_px = u_spec_scan["digits"]
    bid = None
    ask = None
    mid = None
    spread = None
    tick_time = None
    tick_age = None
    srv_ts_scan = None
    price_source = None
    snapshot_status = "OK"

    if isinstance(snapshot, dict) and snapshot.get("snapshot_status") in ("OK", "MARKET_CLOSED"):
        bid = snapshot.get("bid")
        ask = snapshot.get("ask")
        mid = snapshot.get("mid")
        spread = snapshot.get("spread")
        tick_time = snapshot.get("tick_time") or snapshot.get("snapshot_time")
        tick_age = snapshot.get("tick_age")
        srv_ts_scan = snapshot.get("server_time")
        price_source = snapshot.get("price_source")
        if snapshot.get("snapshot_status") == "MARKET_CLOSED":
            snapshot_status = "MARKET_CLOSED"
            eff_mkt_status = "CLOSED"
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
                    bid = round(b_val, d_scan_px)
                    ask = round(a_val, d_scan_px)
                    mid = round((bid + ask) / 2.0, d_scan_px)
                    spread = round(ask - bid, d_scan_px)
                    if hasattr(tick, "time") and tick.time > 0:
                        tick_time = format_wib_timestamp(tick.time, default_now=True)
                        tick_age = round(max(0.0, time.time() - tick.time), 1)
                        srv_ts_scan = pd.Timestamp(int(tick.time), unit="s", tz="UTC")
                    else:
                        tick_time = format_wib_timestamp(default_now=True)
                        tick_age = 0.0
                    _sp_lbl_s = format_price_distance(spread, spec=u_spec_scan, include_mt5_points=True)
                    price_source = f"MT5 Live Tick (Bid: {bid:.{d_scan_px}f} | Ask: {ask:.{d_scan_px}f} | Mid: {mid:.{d_scan_px}f} | Spread: {_sp_lbl_s})"
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
            out["data_freshness"] = "INVALID"
            out["empty_reason"] = "DATA_INCOMPLETE: Snapshot MT5 Bid/Ask tidak lengkap atau tidak valid"
            out["summary_text"] = "Snapshot MT5 tidak lengkap (DATA_INCOMPLETE)."
            return out
        else:
            ref = frames.get("M5") if frames.get("M5") is not None and len(frames["M5"]) else frames["H1"]
            current_price = float(ref["Close"].iloc[-1])

    if not (isinstance(current_price, (int, float)) and math.isfinite(current_price) and current_price > 0):
        out["snapshot_status"] = "DATA_INCOMPLETE"
        out["data_freshness"] = "INVALID"
        out["empty_reason"] = "DATA_INCOMPLETE: Harga referensi tidak valid"
        out["summary_text"] = "Harga tidak valid (WAIT)."
        return out

    # Pisahkan candle berjalan (forming/unclosed) agar kalkulasi struktur SMC hanya memakai candle close
    closed_frames, forming_candles = _separate_forming_candles(frames, server_time=srv_ts_scan)
    if isinstance(closed_frames, dict):
        frames = closed_frames
    if isinstance(snapshot, dict) and isinstance(snapshot.get("forming_candles"), dict) and snapshot.get("forming_candles"):
        forming_candles = snapshot.get("forming_candles") or forming_candles

    fresh_info = validate_data_freshness(
        frames,
        server_time=srv_ts_scan,
        tick_age=tick_age,
        market_status=eff_mkt_status,
        connection_status=eff_conn_status,
        snapshot_status=snapshot_status,
    )
    if eff_mkt_status == "UNKNOWN" and fresh_info.get("market_status"):
        eff_mkt_status = fresh_info["market_status"]
    if eff_conn_status == "UNKNOWN" and fresh_info.get("connection_status"):
        eff_conn_status = fresh_info["connection_status"]

    snapshot_price = round(float(current_price), d_scan_px)
    snapshot_time = tick_time or format_wib_timestamp(srv_ts_scan, default_now=True)
    out["snapshot_status"] = snapshot_status
    out["digits"] = d_scan_px
    out["point"] = u_spec_scan["point"]
    out["trade_tick_size"] = u_spec_scan["trade_tick_size"]
    out["symbol_spec"] = u_spec_scan

    if price_source is None:
        price_source = f"Candle Close Snapshot ({snapshot_price:.{d_scan_px}f})"

    out["current_price"] = snapshot_price
    out["snapshot_price"] = snapshot_price
    out["snapshot_time"] = snapshot_time
    out["snapshot_id"] = snapshot_id
    out["data_freshness"] = fresh_info["status"]
    out["data_freshness_info"] = fresh_info
    out["last_candle_times"] = fresh_info.get("last_candle_times", {})
    out["forming_candles"] = forming_candles
    out["market_status"] = eff_mkt_status
    out["connection_status"] = eff_conn_status
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
            "data_freshness": fresh_info["status"], "data_freshness_info": fresh_info,
            "market_status": eff_mkt_status, "connection_status": eff_conn_status,
            "last_candle_times": out["last_candle_times"], "forming_candles": forming_candles,
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
    _inv_cnt = len(get_zone_tracker(auto_save=False).invalidated_zones)
    ck = _sig(
        _snr_frames_signature(frames), float(snapshot_price), bid, ask, spread, tick_time,
        _ck_snap, eff_mkt_status, fresh_info["status"], _inv_cnt,
    )
    if _SCAN_CACHE["key"] == ck and time.time() - _SCAN_CACHE["ts"] < SCAN_CACHE_TTL:
        cached = dict(_SCAN_CACHE["data"])
        cached["is_cached"] = True
        cached["cache_timestamp"] = format_wib_timestamp(_SCAN_CACHE["ts"], default_now=True)
        cached["cache_age_seconds"] = round(max(0.0, time.time() - _SCAN_CACHE["ts"]), 2)
        cached["data_source_mode"] = "CACHE"
        return cached

    ctx = evaluate_htf_context(frames)
    out["htf_context"] = ctx
    out["htf_bias"] = ctx.get("H4", "NETRAL")
    out["h1_structure"] = ctx.get("H1", "NETRAL")
    out["m30_status"] = ctx.get("m30_status", "WAIT")
    out["htf_status"] = ctx.get("htf_status", "UNCONFIRMED")

    # Ekstrak ATR H1 acuan SEBELUM pemindaian & pemilihan zona utama (Distance-Aware Pre-Filter)
    atr_diag = (
        snapshot.get("atr_diagnostics")
        if isinstance(snapshot, dict) and isinstance(snapshot.get("atr_diagnostics"), dict)
        else extract_atr_diagnostics(frames, preferred_tf="H1")
    )
    ref_atr = atr_diag["atr_value"]
    out["atr_diagnostics"] = atr_diag

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
        _scan_trk = get_zone_tracker(auto_save=False)
        _newly_invalidated = False

        # Evaluasi kandidat yang lolos hard validation (dengan filter jarak ATR H1 + TF asal SEBELUM masuk active_zones)
        eff_scan_atr = ref_atr if (isinstance(ref_atr, (int, float)) and math.isfinite(ref_atr) and ref_atr > 0) else atr_now
        for z in ok:
            score_zone(z, htf_alignment=_alignment(z.zone_type, ctx), current_price=snapshot_price)
            dist_res = calculate_zone_distance(
                z.price_low, z.price_high, snapshot_price, digits=2,
                zone_type=z.zone_type, bid=bid, ask=ask, atr=eff_scan_atr
            )
            dist_val = dist_res.distance
            dir_st = dist_res.status
            z.distance = dist_val
            z.direction_status = dir_st
            z.approach = calculate_zone_approach(dist_val, dir_st, atr=eff_scan_atr)
            dist_ratio, _ = calculate_distance_atr_ratio(dist_val, eff_scan_atr)
            z.distance_atr_ratio = dist_ratio
            z.atr_value = eff_scan_atr
            z.proximal_edge = z.price_high if z.zone_type == "DEMAND" else z.price_low
            z.distal_edge = z.price_low if z.zone_type == "DEMAND" else z.price_high
            z.zone_width = round(abs(z.price_high - z.price_low), 2)

            persisted_inv, persisted_rec = _scan_trk.is_zone_invalidated(z) if _scan_trk else (False, None)
            closed_inv, closed_rsn, closed_px, closed_tm, closed_tf = evaluate_smc_zone_closed_invalidation(z, frames)
            eff_spr_scan = round(float(ask) - float(bid), 2) if (bid and ask and ask >= bid) else 0.0
            live_inv, live_rsn, live_px, live_inv_lvl, _ = evaluate_smc_setup_live_invalidation(
                z, current_price=snapshot_price, bid=bid, ask=ask,
                frames=frames, spread=eff_spr_scan, market_status=eff_mkt_status,
            )
            if live_inv_lvl and live_inv_lvl > 0:
                z.setup_invalidation_price = live_inv_lvl
            live_valid = (not persisted_inv) and (not closed_inv) and (not live_inv) and is_zone_active_and_valid(z, snapshot_price, frames=frames)
            if not live_valid:
                z.is_valid = False
                z.freshness = "INVALIDATED"
                z.status = "INVALIDATED"
                z.zone_relevance = "INVALID"
                pers_type = str((persisted_rec or {}).get("invalidation_type") or "").upper() if isinstance(persisted_rec, dict) else ""
                if closed_inv or pers_type == INVALIDATION_ZONE_STRUCTURAL:
                    inv_kind = INVALIDATION_ZONE_STRUCTURAL
                    z.is_broken = True
                elif live_inv or pers_type == INVALIDATION_SETUP_SL_BREACHED:
                    inv_kind = INVALIDATION_SETUP_SL_BREACHED
                    z.is_broken = False
                else:
                    inv_kind = INVALIDATION_ZONE_STRUCTURAL
                    z.is_broken = True
                z.invalidation_type = inv_kind
                z.invalid_label = inv_kind
                if not z.original_grade and z.grade and str(z.grade).upper() != "REJECTED":
                    z.original_grade = str(z.grade)
                z.grade = "REJECTED"
                _inv_lvl_msg = (
                    f"SETUP_SL {z.setup_invalidation_price:.2f}"
                    if (inv_kind == INVALIDATION_SETUP_SL_BREACHED and z.setup_invalidation_price > 0)
                    else f"STRUCTURE_INV {z.invalidation_price:.2f}"
                )
                z.rejection_reason = (
                    closed_rsn
                    or live_rsn
                    or (persisted_rec.get("reason") if isinstance(persisted_rec, dict) and persisted_rec.get("reason") else "")
                    or f"{inv_kind} ({z.timeframe} {_inv_lvl_msg})"
                )
                z.zone_relevance_reason = z.rejection_reason
                z.audit.append(z.rejection_reason)
                if _scan_trk and (closed_inv or live_inv) and not persisted_inv:
                    _scan_trk.mark_invalidated(
                        z,
                        z.rejection_reason,
                        break_close=closed_px if closed_inv else None,
                        break_time=closed_tm if closed_inv else snapshot_time,
                        origin_tf=closed_tf if closed_inv else z.timeframe,
                        invalidation_type=inv_kind,
                        break_price=live_px if (live_inv and not closed_inv) else closed_px,
                    )
                    _newly_invalidated = True
                total_invalidated += 1
                rej_desc = z.rejection_reason
                rej_bucket = (
                    f"{INVALIDATION_ZONE_STRUCTURAL} (Candle closed menembus batas invalidasi)"
                    if inv_kind == INVALIDATION_ZONE_STRUCTURAL
                    else f"{INVALIDATION_SETUP_SL_BREACHED} (Live price menembus batas SL setup)"
                )
                rejection_reasons_counter[rej_bucket] = rejection_reasons_counter.get(rej_bucket, 0) + 1
                rejected.append(z)
            elif z.is_valid and z.score >= SCORE_MIN_ACTIVE:
                total_scored_pass += 1
                z.status = "RADAR"
                rel_pre = classify_zone_relevance(
                    z, snapshot_price, atr=eff_scan_atr,
                    htf_bias=ctx.get("H4", "NETRAL"),
                    h1_structure=ctx.get("H1", "NETRAL"),
                    m30_status=ctx.get("m30_status", "CONFIRMED"),
                    frames=frames, bid=bid, ask=ask,
                )
                z.zone_relevance = rel_pre["relevance"]
                z.zone_relevance_reason = rel_pre["reason"]
                if rel_pre["relevance"] in ("ACTIVE", "RADAR"):
                    valid_all.append(z)
                    radar_all.append(z)
                    rej_desc = "PASSED_VALID_ACTIVE"
                else:
                    total_too_far += 1
                    z.risk_tier = "FAR_ZONE"
                    z.risk_tier_reason = rel_pre["reason"]
                    z.approach = "FAR"
                    z.radar_eligibility = "QUALIFIED"
                    z.pending_eligibility = "WATCHLIST"
                    rej_desc = rel_pre["reason"]
                    z.audit.append(rej_desc)
                    valid_all.append(z)
                    radar_all.append(z)
            else:
                z.is_valid = False
                z.grade = "REJECTED"
                z.zone_relevance = "INVALID"
                z.status = "INVALIDATED" if ("invalid" in (z.rejection_reason or "").lower() or z.freshness == "INVALIDATED") else "RADAR"
                rej_key = z.rejection_reason or f"Skor {z.score:.1f} < {SCORE_MIN_ACTIVE}"
                z.zone_relevance_reason = rej_key
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
                "zone_relevance": z.zone_relevance,
                "rejection_reason": rej_desc if not z.is_valid else ("TOO_FAR" if z.zone_relevance == "FAR_REFERENCE" else "PASSED"),
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

            persisted_inv_b, persisted_rec_b = _scan_trk.is_zone_invalidated(z) if _scan_trk else (False, None)
            closed_inv_b, closed_rsn_b, closed_px_b, closed_tm_b, closed_tf_b = evaluate_smc_zone_closed_invalidation(z, frames)
            eff_spr_b = round(float(ask) - float(bid), 2) if (bid and ask and ask >= bid) else 0.0
            live_inv_b, live_rsn_b, live_px_b, live_inv_lvl_b, _ = evaluate_smc_setup_live_invalidation(
                z, current_price=snapshot_price, bid=bid, ask=ask,
                frames=frames, spread=eff_spr_b, market_status=eff_mkt_status,
            )
            if live_inv_lvl_b and live_inv_lvl_b > 0:
                z.setup_invalidation_price = live_inv_lvl_b
            _non_break_reasons_b = [
                rp for rp in (rej.split("; ") if rej else [])
                if rp and "INVALID — BROKEN" not in rp and INVALIDATION_ZONE_STRUCTURAL not in rp and INVALIDATION_SETUP_SL_BREACHED not in rp and "Zona invalid" not in rp
            ]
            if "Zona invalid" in rej or "INVALID — BROKEN" in rej or INVALIDATION_ZONE_STRUCTURAL in rej or z.is_broken or persisted_inv_b or closed_inv_b or live_inv_b:
                total_invalidated += 1
                z.status = "INVALIDATED" if not _non_break_reasons_b else "REJECTED"
                z.freshness = "INVALIDATED"
                if closed_inv_b or z.is_broken or "Zona invalid" in rej or INVALIDATION_ZONE_STRUCTURAL in rej:
                    inv_kind_b = INVALIDATION_ZONE_STRUCTURAL
                    z.is_broken = True
                elif live_inv_b:
                    inv_kind_b = INVALIDATION_SETUP_SL_BREACHED
                    z.is_broken = False
                else:
                    inv_kind_b = str((persisted_rec_b or {}).get("invalidation_type") or INVALIDATION_ZONE_STRUCTURAL)
                    z.is_broken = (inv_kind_b == INVALIDATION_ZONE_STRUCTURAL)
                z.invalidation_type = inv_kind_b
                z.invalid_label = inv_kind_b
                if live_inv_b and not closed_inv_b and live_rsn_b:
                    z.rejection_reason = live_rsn_b
                if _scan_trk and not persisted_inv_b and not _non_break_reasons_b:
                    _scan_trk.mark_invalidated(
                        z,
                        closed_rsn_b or live_rsn_b or rej,
                        break_close=closed_px_b if closed_inv_b else None,
                        break_time=closed_tm_b if closed_inv_b else snapshot_time,
                        origin_tf=closed_tf_b or z.timeframe,
                        invalidation_type=inv_kind_b,
                        break_price=live_px_b if (live_inv_b and not closed_inv_b) else closed_px_b,
                    )
                    _newly_invalidated = True
            if "Zona sudah dimitigasi" in rej:
                total_mitigated += 1
            if "expired" in rej.lower() or z.freshness == "EXPIRED":
                total_expired += 1
                z.status = "EXPIRED"

            # Kategori A: Radar Zone untuk zona yang utuh secara struktural (TIDAK pernah memasukkan zona INVALID)
            is_inval = (
                z.freshness == "INVALIDATED" or z.is_broken or "Zona invalid" in rej
                or "INVALID — BROKEN" in rej or INVALIDATION_ZONE_STRUCTURAL in rej
                or INVALIDATION_SETUP_SL_BREACHED in rej
                or persisted_inv_b or closed_inv_b or live_inv_b
            )
            is_mit = (z.touch_count > MAX_TOUCH_COUNT or "Zona sudah dimitigasi" in rej)
            origin_ok = z.origin_bos.get("origin_valid", True)
            is_struct_valid = is_zone_active_and_valid(z, snapshot_price, frames=frames)
            if (not is_inval) and (not is_mit) and origin_ok and is_struct_valid:
                z.grade = "RADAR"
                z.status = "RADAR"
                z.is_valid = True
                score_zone(z, htf_alignment=_alignment(z.zone_type, ctx), allow_radar=True, current_price=snapshot_price)
                rel_bad = classify_zone_relevance(
                    z, snapshot_price, atr=eff_scan_atr,
                    htf_bias=ctx.get("H4", "NETRAL"),
                    h1_structure=ctx.get("H1", "NETRAL"),
                    m30_status=ctx.get("m30_status", "CONFIRMED"),
                    frames=frames, bid=bid, ask=ask,
                )
                z.zone_relevance = rel_bad["relevance"]
                z.zone_relevance_reason = rel_bad["reason"]
                z.atr_value = eff_scan_atr
                z.distance_atr_ratio = rel_bad["distance_atr_ratio"]
                if rel_bad["relevance"] == "FAR_REFERENCE":
                    z.risk_tier = "FAR_ZONE"
                    z.risk_tier_reason = rel_bad["reason"]
                    z.approach = "FAR"
                    z.radar_eligibility = "QUALIFIED"
                    z.pending_eligibility = "WATCHLIST"
                z.audit.append(f"Kandidat zona reaksi (RADAR SMC | Skor: {z.score:.1f})")
                radar_all.append(z)
            else:
                z.is_valid = False
                if is_inval:
                    z.status = "INVALIDATED"
                    z.zone_relevance = "INVALID"
                    z.is_broken = True
                    z.invalid_label = "INVALID — BROKEN"
                elif is_mit:
                    z.status = "MITIGATED"
                    z.zone_relevance = "INVALID"
                elif "expired" in rej.lower() or z.freshness == "EXPIRED":
                    z.status = "EXPIRED"
                    z.zone_relevance = "INVALID"
                else:
                    z.status = "REJECTED"
                    z.zone_relevance = "INVALID"

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

        if _newly_invalidated and _scan_trk and (not isinstance(snapshot, dict) or snapshot.get("live") is True):
            _scan_trk.save()

    active = [z for z in merge_overlapping_zones(valid_all) if not _scan_trk.is_zone_invalidated(z)[0]]
    radar_zones = [z for z in merge_overlapping_zones(radar_all) if not _scan_trk.is_zone_invalidated(z)[0]]

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

    radar_prio = prioritize_radar_zones(
        radar_zones=radar_zones,
        current_price=snapshot_price,
        htf_bias=ctx.get("H4", "NETRAL"),
        h1_structure=ctx.get("H1", "NETRAL"),
        atr=ref_atr,
        digits=2,
        frames=frames,
        m30_status=ctx.get("m30_status", "CONFIRMED"),
        rejected_zones=rejected,
        bid=bid,
        ask=ask,
        snapshot_time=snapshot_time,
        snapshot_id=snapshot_id,
        market_status=eff_mkt_status,
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
    out["far_reference_zones"] = radar_prio.get("far_reference_zones", [])
    out["far_reference_zone"] = radar_prio.get("far_reference_zone")

    # Sinkronisasi atribut prioritas, relevansi & multi-tier ke objek Zone
    p_map = {z_d["zone_id"]: z_d for z_d in radar_prio.get("all_prioritized", [])}
    seen_sync_objs = set()
    for z in (*radar_zones, *active):
        if id(z) in seen_sync_objs:
            continue
        seen_sync_objs.add(id(z))
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
            z.risk_tier_reason = p_item.get("risk_tier_reason", "")
            z.status = p_item.get("status", "RADAR")
            z.radar_eligibility = p_item.get("radar_eligibility", "QUALIFIED")
            z.pending_eligibility = p_item.get("pending_eligibility", "NOT_QUALIFIED")
            z.confirmed_entry_eligibility = p_item.get("confirmed_entry_eligibility", "WAIT")
            z.pending_plan = p_item.get("pending_plan")
            z.zone_relevance = p_item.get("zone_relevance", z.zone_relevance)
            z.zone_relevance_reason = p_item.get("zone_relevance_reason", z.zone_relevance_reason)
        else:
            d_res = calculate_zone_distance(
                z.price_low, z.price_high, snapshot_price, digits=2,
                zone_type=z.zone_type, bid=bid, ask=ask, atr=ref_atr,
            )
            z.distance = d_res.distance
            z.direction_status = d_res.status
            z.approach = calculate_zone_approach(d_res.distance, d_res.status, atr=ref_atr)
            z.distance_atr_ratio, _ = calculate_distance_atr_ratio(d_res.distance, ref_atr)
            z.atr_value = ref_atr
            rel_sync = classify_zone_relevance(
                z, snapshot_price, atr=ref_atr,
                htf_bias=ctx.get("H4", "NETRAL"),
                h1_structure=ctx.get("H1", "NETRAL"),
                m30_status=ctx.get("m30_status", "CONFIRMED"),
                frames=frames, bid=bid, ask=ask,
            )
            z.zone_relevance = rel_sync["relevance"]
            z.zone_relevance_reason = rel_sync["reason"]
            z.risk_tier = classify_zone_risk_tier(
                z, snapshot_price,
                htf_bias=ctx.get("H4", "NETRAL"),
                h1_structure=ctx.get("H1", "NETRAL"),
                atr=ref_atr, bid=bid, ask=ask,
            )

    # Pastikan active_zones hanya memuat zona in-range (ACTIVE/RADAR) dan bukan FAR_REFERENCE
    active = [
        z for z in active
        if z.zone_relevance in ("ACTIVE", "RADAR")
        and z.risk_tier != "FAR_ZONE"
        and is_zone_active_and_valid(z, snapshot_price, frames=frames)
    ]
    diagnostics["zones_valid_active"] = len(active)

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
    out["snr_context"] = extract_snr_supporting_context(
        frames,
        current_price,
        market_status=eff_mkt_status,
        snapshot=snapshot if isinstance(snapshot, dict) else out.get("snapshot"),
    )
    # SMC + SNR Confluence Evaluation (Supporting Context Only)
    target_smc_c = (out.get("radar_priority") or {}).get("primary_directional_zone") or (active[0] if active else (radar_zones[0] if radar_zones else None))
    out["smc_snr_confluence"] = evaluate_smc_snr_confluence(
        target_smc_c,
        out.get("snr_context"),
        current_price=current_price,
        htf_bias=ctx.get("H4", "NETRAL"),
    )
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
# SMC + SNR CONFLUENCE (SUPPORTING CONTEXT ONLY)
# =============================================================================
def evaluate_smc_snr_confluence(
    smc_zone: Optional[Union[Zone, Dict[str, Any]]],
    snr_context: Optional[Dict[str, Any]],
    current_price: Optional[float] = None,
    htf_bias: str = "NETRAL",
) -> Dict[str, Any]:
    """
    Evaluasi Konfluensi SMC + SNR sebagai INFORMASI PENDUKUNG (Supporting Context Only).
    
    Aturan Ketat:
      - Konfluensi tidak boleh menambah skor secara ganda (no double-counting).
      - Konfluensi tidak boleh mengubah status WAIT menjadi BUY/SELL/READY tanpa syarat lengkap.
      - Menghitung overlap poin dan persentase terhadap lebar zona SMC.
      - Mendeteksi kedekatan batas jika tidak overlap (jarak <= 3.0 pt).
      - Menampilkan status keselarasan arah dan HTF alignment.
    """
    if isinstance(snr_context, list):
        snr_context = {
            "available": len(snr_context) > 0,
            "resistances": [z for z in snr_context if str(z.get("zone_type", "")).upper() in ("RESISTANCE", "RES")],
            "supports": [z for z in snr_context if str(z.get("zone_type", "")).upper() in ("SUPPORT", "SUP")],
        }

    has_snr_zones = isinstance(snr_context, dict) and bool(
        snr_context.get("supports")
        or snr_context.get("resistances")
        or snr_context.get("support")
        or snr_context.get("resistance")
    )
    if not smc_zone or not isinstance(snr_context, dict) or (not snr_context.get("available", False) and not has_snr_zones):
        return {
            "available": False,
            "status": "NOT_AVAILABLE",
            "confluence_status": "NONE",
            "is_aligned": False,
            "is_historical_context": False,
            "overlap_points": 0.0,
            "overlap_percentage": 0.0,
            "gap_points": 0.0,
            "smc_zone": None,
            "snr_zone": None,
            "opposing_conflict": False,
            "opposing_snr_zone": None,
            "opposing_overlap_points": 0.0,
            "htf_alignment": "NONE",
            "compact_summary": "NONE (Data SNR tidak mencukupi)",
            "display_block": "SMC + SNR CONFLUENCE: NOT AVAILABLE (Data SNR tidak mencukupi)",
        }

    is_hist_ctx = bool(snr_context.get("historical_only") or snr_context.get("is_stale") or not snr_context.get("available", False))

    zd = smc_zone.to_dict() if hasattr(smc_zone, "to_dict") else dict(smc_zone)
    z_type = str(zd.get("zone_type", "")).upper()
    smc_lo = float(zd.get("price_low", zd.get("low", 0.0)) or 0.0)
    smc_hi = float(zd.get("price_high", zd.get("high", 0.0)) or 0.0)
    if smc_lo > smc_hi:
        smc_lo, smc_hi = smc_hi, smc_lo
    smc_w = max(0.1, smc_hi - smc_lo)
    smc_tf = str(zd.get("timeframe", "M5"))
    smc_fresh = str(zd.get("freshness", "FRESH"))
    smc_grd = str(zd.get("grade", "A"))

    def _snr_zone_list(key_plural: str, key_single: str) -> List[Dict[str, Any]]:
        vals = snr_context.get(key_plural)
        if isinstance(vals, list) and vals:
            return [z for z in vals if isinstance(z, dict)]
        single = snr_context.get(key_single)
        if isinstance(single, list):
            return [z for z in single if isinstance(z, dict)]
        if isinstance(single, dict):
            return [single]
        return []

    # Cari zona SNR yang sejenis (Supply -> Resistance, Demand -> Support)
    # serta deteksi konflik terhadap zona SNR lawan yang masih aktif/unbroken
    matching_snr_zones: List[Dict[str, Any]] = []
    opposing_snr_zones: List[Dict[str, Any]] = []
    opp_dir_for_zone = "SELL" if z_type == "SUPPLY" else "BUY"
    if z_type == "SUPPLY":
        matching_snr_zones = _snr_zone_list("resistances", "resistance")
        opposing_snr_zones = _snr_zone_list("supports", "support")
    elif z_type == "DEMAND":
        matching_snr_zones = _snr_zone_list("supports", "support")
        opposing_snr_zones = _snr_zone_list("resistances", "resistance")

    opposing_snr_zone: Optional[Dict[str, Any]] = None
    opp_overlap_pts = 0.0
    for oz in opposing_snr_zones:
        if _is_snr_zone_broken_or_inactive(oz, opp_dir_for_zone, snr_context=snr_context):
            continue
        o_grade = str(oz.get("grade") or "").upper()
        o_score = float(oz.get("score") or 0.0)
        o_strength = str(oz.get("strength_short") or oz.get("strength") or "").upper()
        o_material = (
            o_grade in ("A+", "A", "B", "C")
            or o_score >= 60.0
            or ("STRONG" in o_strength)
            or ("MODERATE" in o_strength)
            or ("KUAT" in o_strength)
            or ("SEDANG" in o_strength)
        )
        if not o_material:
            continue
        o_lo = float(oz.get("low", 0.0) or 0.0)
        o_hi = float(oz.get("high", 0.0) or 0.0)
        if o_lo <= 0 or o_hi <= 0:
            continue
        if o_lo > o_hi:
            o_lo, o_hi = o_hi, o_lo
        opp_ov_lo = max(smc_lo, o_lo)
        opp_ov_hi = min(smc_hi, o_hi)
        opp_ov = max(0.0, opp_ov_hi - opp_ov_lo)
        if opp_ov > 0 and (opposing_snr_zone is None or opp_ov > opp_overlap_pts):
            opp_overlap_pts = opp_ov
            opposing_snr_zone = oz

    opposing_conflict = opposing_snr_zone is not None
    opp_conflict_line = ""
    if opposing_conflict and opposing_snr_zone is not None:
        o_tf = str(opposing_snr_zone.get("timeframe", "H1"))
        o_lbl = str(opposing_snr_zone.get("label") or opposing_snr_zone.get("rank_label") or ("Support" if z_type == "SUPPLY" else "Resistance"))
        o_grd = str(opposing_snr_zone.get("grade") or "-")
        o_lo = float(opposing_snr_zone.get("low", 0.0) or 0.0)
        o_hi = float(opposing_snr_zone.get("high", 0.0) or 0.0)
        opp_conflict_line = (
            f"KONFLIK SNR LAWAN: {smc_tf} {z_type} bertabrakan dengan {o_tf} "
            f"{o_lbl} Grade {o_grd} ({o_lo:.2f} - {o_hi:.2f}, overlap {opp_overlap_pts:.2f} USD)"
        )

    best_match = None
    best_overlap = -1.0
    best_gap = 999.0
    match_dir_for_zone = "SELL" if z_type == "SUPPLY" else "BUY"

    for sz in matching_snr_zones:
        if _is_snr_zone_broken_or_inactive(sz, match_dir_for_zone, snr_context=snr_context):
            continue
        s_lo = float(sz.get("low", sz.get("price_low", 0.0)) or 0.0)
        s_hi = float(sz.get("high", sz.get("price_high", 0.0)) or 0.0)
        if s_lo <= 0 or s_hi <= 0:
            continue
        if s_lo > s_hi:
            s_lo, s_hi = s_hi, s_lo
        # Validasi irisan rentang harga sebenarnya (> 0)
        ov_lo = max(smc_lo, s_lo)
        ov_hi = min(smc_hi, s_hi)
        raw_overlap = ov_hi - ov_lo
        overlap = round(raw_overlap, 2) if raw_overlap > 0.0 else 0.0
        if raw_overlap > 0.0 and overlap > 0.0:
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = (sz, ov_lo, ov_hi, overlap, 0.0)
        else:
            gap = (smc_lo - s_hi) if smc_lo > s_hi else (s_lo - smc_hi)
            if best_overlap <= 0 and gap < best_gap:
                best_gap = gap
                best_match = (sz, 0.0, 0.0, 0.0, max(0.0, gap))

    if not best_match:
        no_match_disp = "SMC + SNR CONFLUENCE: NONE (Tidak ditemukan zona SNR searah yang berdekatan)"
        if opposing_conflict:
            no_match_disp = f"{no_match_disp}\n  • {opp_conflict_line}"
        return {
            "available": True,
            "status": "CONFLICT" if opposing_conflict else "NO_CONFLUENCE",
            "confluence_status": "CONFLICT" if opposing_conflict else "NONE",
            "is_aligned": False,
            "is_historical_context": is_hist_ctx,
            "overlap_points": 0.0,
            "overlap_percentage": 0.0,
            "gap_points": 0.0,
            "smc_zone": zd,
            "snr_zone": None,
            "opposing_conflict": opposing_conflict,
            "opposing_snr_zone": opposing_snr_zone,
            "opposing_overlap_points": round(opp_overlap_pts, 2),
            "htf_alignment": "NONE",
            "compact_summary": (f"CONFLICT ({opp_conflict_line})" if opposing_conflict else "NONE"),
            "display_block": no_match_disp,
        }

    sz, ov_lo, ov_hi, overlap_pts, gap_pts = best_match
    snr_tf = str(sz.get("timeframe") or "H1")
    snr_lbl = str(sz.get("rank_label") or sz.get("label") or ("R1" if z_type == "SUPPLY" else "S1"))
    snr_type = "RESISTANCE" if z_type == "SUPPLY" else "SUPPORT"
    snr_fresh = str(sz.get("freshness") or "FRESH")
    s_lo = float(sz.get("low", sz.get("price_low", 0.0)) or 0.0)
    s_hi = float(sz.get("high", sz.get("price_high", 0.0)) or 0.0)
    if s_lo > s_hi:
        s_lo, s_hi = s_hi, s_lo

    is_aligned = (overlap_pts > 0.0)
    overlap_pct = round((overlap_pts / smc_w) * 100.0, 1) if smc_w > 0 else 0.0
    snr_w = max(0.1, s_hi - s_lo)
    overlap_pct_snr = round((overlap_pts / snr_w) * 100.0, 1) if snr_w > 0 else 0.0
    union_w = max(0.1, max(smc_hi, s_hi) - min(smc_lo, s_lo))
    jaccard_ratio = round(overlap_pts / union_w, 3) if overlap_pts > 0 else 0.0
    if opposing_conflict:
        confl_quality = "CONFLICT"
    elif is_aligned and overlap_pct >= 30.0:
        confl_quality = "STRONG_CONFLUENCE"
    elif is_aligned:
        confl_quality = "PARTIAL_CONFLUENCE"
    else:
        confl_quality = "SEPARATED"

    htf_align_str = "ALIGNED" if ((htf_bias == "BEARISH" and z_type == "SUPPLY") or (htf_bias == "BULLISH" and z_type == "DEMAND")) else "COUNTERTREND"

    ctx_note = " (HISTORICAL/MARKET CLOSED)" if is_hist_ctx else ""
    if overlap_pts > 0:
        area_desc = f"{ov_lo:.2f} - {ov_hi:.2f} ({overlap_pts:.2f} USD | {overlap_pct:.1f}% SMC / {overlap_pct_snr:.1f}% SNR)"
        compact_summary = (
            f"OVERLAP {overlap_pts:.2f} USD ({overlap_pct:.1f}% SMC) @ {ov_lo:.2f}-{ov_hi:.2f} "
            f"vs SNR {snr_lbl} {snr_tf} {s_lo:.2f}-{s_hi:.2f} [SUPPORTING ONLY{ctx_note}]"
        )
    else:
        area_desc = f"TIDAK OVERLAP — Terpisah {gap_pts:.2f} USD (SMC Active Focus terpisah dari SNR Reference)"
        compact_summary = f"SEPARATED (Gap {gap_pts:.2f} USD vs SNR {snr_lbl} {s_lo:.2f}-{s_hi:.2f})"
    if opposing_conflict:
        compact_summary += f" | {opp_conflict_line}"

    block_lines = [
        f"SMC + SNR CONFLUENCE: ALIGNED (OVERLAP){ctx_note}" if is_aligned else f"SMC + SNR CONFLUENCE: SEPARATED (NO OVERLAP — BUKAN FULL CONFLUENCE){ctx_note}",
        f"  • Status         : SUPPORTING CONTEXT{ctx_note} — NOT ENTRY CONFIRMATION",
        f"  • Overlap Area   : {area_desc}",
        f"  • SMC Active     : {smc_tf} {z_type} ({smc_lo:.2f} - {smc_hi:.2f}) [Grade {smc_grd}] | Freshness: {smc_fresh}",
        f"  • SNR Reference  : {snr_tf} {snr_type} {snr_lbl} ({s_lo:.2f} - {s_hi:.2f}) | Freshness: {snr_fresh}",
        f"  • HTF Alignment  : {htf_align_str} with H4 {htf_bias}",
        "  • Keterangan     : Informasi pendukung konfluensi zona (tidak menambah skor, tidak mengubah NOT_ELIGIBLE / filter).",
    ]
    if opposing_conflict:
        block_lines.append(f"  • Peringatan     : {opp_conflict_line}")

    return {
        "available": True,
        "status": "ALIGNED" if is_aligned else "SEPARATED",
        "confluence_status": "ALIGNED" if is_aligned else "SEPARATED",
        "is_aligned": is_aligned,
        "is_historical_context": is_hist_ctx,
        "overlap_points": round(overlap_pts, 2),
        "overlap_percentage": overlap_pct,
        "overlap_percentage_snr": overlap_pct_snr,
        "jaccard_ratio": jaccard_ratio,
        "confluence_quality": confl_quality,
        "gap_points": round(gap_pts, 2),
        "overlap_area": (round(ov_lo, 2), round(ov_hi, 2)) if overlap_pts > 0 else None,
        "smc_zone": zd,
        "snr_zone": sz,
        "smc_zone_id": str(zd.get("zone_id") or ""),
        "snr_zone_id": str(sz.get("zone_id") or ""),
        "opposing_conflict": opposing_conflict,
        "opposing_snr_zone": opposing_snr_zone,
        "opposing_overlap_points": round(opp_overlap_pts, 2),
        "htf_alignment": htf_align_str,
        "compact_summary": compact_summary,
        "display_block": "\n".join(block_lines),
    }


# =============================================================================
# ENTRY VALIDATOR (M5) + RISK
# =============================================================================
def _structural_targets(
    frames: Dict[str, pd.DataFrame],
    zone_type: str,
    entry: float,
    opposing_zones: Optional[List[Any]] = None,
    snr_context: Optional[Dict[str, Any]] = None,
    atr: Optional[float] = None,
) -> List[float]:
    """Likuiditas struktural searah transaksi (swing M15/M30/H1/H4 yang belum ditembus + opposing barrier), diurutkan dari terdekat."""
    levels: List[float] = []
    z_type_u = str(zone_type or "").upper()
    is_demand = (z_type_u == "DEMAND")

    # 1. Swings dari M15, M30, H1, H4
    for tf in ("M15", "M30", "H1", "H4"):
        df = (frames or {}).get(tf)
        if df is None or len(df) < 2 * SWING_N + 1:
            continue
        H, L = df["High"].values, df["Low"].values
        for s in detect_swings(df):
            i, p = s["idx"], s["price"]
            if is_demand and s["type"] == "H" and p > entry and (i + 1 >= len(H) or H[i + 1:].max() <= p):
                levels.append(p)
            elif (not is_demand) and s["type"] == "L" and p < entry and (i + 1 >= len(L) or L[i + 1:].min() >= p):
                levels.append(p)

    # 2. Opposing SMC zones (penghalang / target likuiditas terdekat)
    if opposing_zones:
        for oz in opposing_zones:
            oz_d = oz if isinstance(oz, dict) else (oz.to_dict() if hasattr(oz, "to_dict") else {})
            if not oz_d or not oz_d.get("is_valid", True):
                continue
            oz_type = str(oz_d.get("zone_type") or "").upper()
            oz_lo = float(oz_d.get("price_low") or 0.0)
            oz_hi = float(oz_d.get("price_high") or 0.0)
            if is_demand and oz_type == "SUPPLY" and oz_lo > entry:
                levels.append(oz_lo)
            elif (not is_demand) and oz_type == "DEMAND" and 0 < oz_hi < entry:
                levels.append(oz_hi)

    # 3. Opposing SNR context (penghalang / support-resistance material)
    if isinstance(snr_context, dict):
        if is_demand:
            for r in (snr_context.get("resistances") or []):
                if isinstance(r, dict):
                    r_lo = float(r.get("low") or 0.0)
                    if r_lo > entry:
                        levels.append(r_lo)
        else:
            for s in (snr_context.get("supports") or []):
                if isinstance(s, dict):
                    s_hi = float(s.get("high") or 0.0)
                    if 0 < s_hi < entry:
                        levels.append(s_hi)

    levels = sorted(set(round(x, 2) for x in levels), reverse=(not is_demand))

    # 4. Fallback jika tidak ada swing searah
    if not levels and atr is not None and isinstance(atr, (int, float)) and atr > 0:
        step = max(3.0, 2.0 * float(atr))
        if is_demand:
            levels = [round(entry + step, 2), round(entry + 2 * step, 2)]
        else:
            levels = [round(entry - step, 2), round(entry - 2 * step, 2)]

    return levels


def _entry_check(zone: Zone, price: float, df_m5: Optional[pd.DataFrame], spread: float = 0.0,
                 news_blackout=False, htf_frames: Optional[Dict[str, pd.DataFrame]] = None,
                 require_sweep: Optional[bool] = None, server_time: Any = None,
                 snr_context: Optional[Dict[str, Any]] = None,
                 bid: Optional[float] = None, ask: Optional[float] = None,
                 market_status: Optional[str] = None) -> Dict[str, Any]:
    """Hasil terstruktur: status VALID/WAIT/INVALID, reasons, stages, plan.
    Urutan validasi ketat konsisten untuk semua grade zona:
      1. Validasi zona & integritas data candle close selesai
      2. Harga mencapai atau memasuki zona entry (dan belum terlambat)
      3. Terdapat indikasi Liquidity Sweep yang valid dan relevan dengan area zona
      4. Terdapat konfirmasi pembalikan atau kelanjutan struktur (CHoCH/BOS/CISD) SETELAH sweep
      5. Terdapat displacement yang memadai searah transaksi
      6. Risk Reward memenuhi syarat (>= 1.5), jarak SL wajar (<= 20 pt), dan bebas penghalang SNR material
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
    res: Dict[str, Any] = {
        "status": "WAIT",
        "reasons": [],
        "plan": None,
        "m5": {},
        "stages": stages,
        "smc_technical_confirmed": False,
        "entry_permission": "WAIT",
    }
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
    closed_inv_ec, closed_rsn_ec, _, _, _ = evaluate_smc_zone_closed_invalidation(zone, htf_frames)
    if closed_inv_ec or getattr(zone, "is_broken", False) or str(getattr(zone, "status", "")).upper() in ("INVALIDATED", "BROKEN"):
        res["status"] = "INVALID"
        res["invalidation_type"] = INVALIDATION_ZONE_STRUCTURAL
        msg = closed_rsn_ec or f"{INVALIDATION_ZONE_STRUCTURAL}: Zona {zone.zone_id} telah ditembus candle close"
        stages["zone_validation"] = {"status": "INVALID", "msg": msg}
        stages["entry_validation"] = {"status": "INVALID", "msg": msg}
        W.append(msg)
        return res
    live_inv_ec, live_rsn_ec, _, _, _ = evaluate_smc_setup_live_invalidation(
        zone, current_price=price, bid=bid, ask=ask,
        frames=htf_frames, spread=float(spread or 0.0), market_status=market_status,
    )
    if live_inv_ec:
        res["status"] = "INVALID"
        res["invalidation_type"] = INVALIDATION_SETUP_SL_BREACHED
        stages["zone_validation"] = {"status": "INVALID", "msg": live_rsn_ec}
        stages["entry_validation"] = {"status": "INVALID", "msg": live_rsn_ec}
        W.append(live_rsn_ec)
        return res
    if (demand and price < zone.invalidation_price) or ((not demand) and price > zone.invalidation_price):
        # Wick / harga intrabar melewati batas distal namun belum tembus SL buffer & tanpa candle closed TF asal -> WAIT
        msg = f"Harga intrabar ({price:.2f}) melewati batas distal {zone.invalidation_price:.2f} (wick belum membatalkan struktur); menunggu close candle TF asal"
        stages["zone_validation"] = {"status": "WAIT", "msg": msg}
        stages["entry_validation"] = {"status": "WAIT", "msg": msg}
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
        last_m5_ts = _get_df_last_timestamp(df_m5)
        if last_m5_ts is not None:
            try:
                srv_al, m5_al = _align_timestamps_for_diff(
                    pd.Timestamp(server_time), last_m5_ts,
                    b_tz_hint=str(df_m5.attrs.get("time_tz", "WIB")),
                )
                diff_sec = (srv_al - m5_al).total_seconds()
                if diff_sec < 5 * 60 - 1e-6:
                    stages["zone_validation"] = {"status": "WAIT", "msg": "Candle terakhir M5 belum selesai (masih berjalan)"}
                    W.append("Candle terakhir M5 belum selesai (masih berjalan)")
                    return res
            except Exception:
                pass

    if spread is not None and spread > max(SPREAD_MAKS_POIN, 0.0):
        sp_txt = format_price_distance(spread, include_mt5_points=True)
        stages["zone_validation"] = {"status": "WAIT", "msg": f"Spread {sp_txt} melebihi batas {SPREAD_MAKS_POIN:g} USD"}
        W.append(f"Spread {sp_txt} melebihi batas {SPREAD_MAKS_POIN:g} USD")
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
    origin_close = None
    if zone.bos_candle_time:
        origin_tf = str(zone.bos_timeframe or zone.timeframe).upper()
        origin_frame = (htf_frames or {}).get(origin_tf)
        origin_tz = str((getattr(origin_frame, "attrs", {}) or {}).get("time_tz", "WIB"))
        origin_time = to_wib_datetime(zone.bos_candle_time, source_tz=origin_tz)
        duration = {"M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240}.get(origin_tf)
        if origin_time is None or duration is None:
            stages["zone_validation"] = {"status": "DATA_INCOMPLETE", "msg": "Waktu konfirmasi zona tidak valid."}
            W.append(stages["zone_validation"]["msg"])
            return res
        origin_close = origin_time + timedelta(minutes=duration)
    m5_tz = str((getattr(d5, "attrs", {}) or {}).get("time_tz", "WIB"))
    for x in range(max(0, n5 - M5_ZONE_TOUCH_LOOKBACK), n5):
        candle_time = to_wib_datetime(d5.index[x], source_tz=m5_tz)
        if origin_close is not None and (candle_time is None or candle_time < origin_close):
            continue
        if (demand and L[x] <= hi and H[x] >= lo) or ((not demand) and H[x] >= lo and L[x] <= hi):
            touched = True
            touch_indices.append(x)
    live_touch_only = not touched and lo <= price <= hi
    if live_touch_only:
        touched = True
        touch_indices.append(n5 - 1)

    if not touched:
        stages["zone_touch"] = {"status": "FAIL", "msg": f"Harga belum mencapai atau menyentuh zona entry {lo:.2f}-{hi:.2f}"}
        W.append(f"Harga belum mencapai atau menyentuh zona entry {lo:.2f}-{hi:.2f}")
        return res

    first_touch_k = n5 if live_touch_only else touch_indices[0]
    dist = 0.0 if lo <= price <= hi else (lo - price if price < lo else price - hi)
    approach = max(3.0, 1.5 * atr5)
    if dist > approach:
        moved_away_profit = (demand and price > hi) or ((not demand) and price < lo)
        detail = (
            f"Entry terlambat: harga telah melewati zona entry ({dist:.2f} USD > toleransi {approach:.2f} USD)"
            if moved_away_profit else
            f"Harga masih di luar zona; menunggu pullback ({dist:.2f} USD > toleransi {approach:.2f} USD)"
        )
        stages["zone_touch"] = {"status": "FAIL", "msg": detail}
        W.append(detail)
        return res

    touch_time_str = str(d5.index[touch_indices[0]]) if touch_indices and touch_indices[0] < n5 else None
    stages["zone_touch"] = {"status": "CONFIRMED", "msg": f"Zona tersentuh (jarak {dist:.2f} USD)", "candle_time": touch_time_str}

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
        if s["type"] == want_sw and first_touch_k <= s["k"] and n5 - 1 - s["k"] <= 40:
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
        "candle_time": str(best_sweep.get("time")) if best_sweep else None,
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

    def _has_opposing_m5_after(trig_k: int) -> bool:
        return any(e2.get("arah") != want_dir and int(e2.get("k", -1)) > trig_k for e2 in events)

    raw_matching_events = [e for e in reversed(events) if e["arah"] == want_dir and n5 - 1 - e["k"] <= M5_EVENT_MAX_AGE and e["k"] >= max(first_touch_k, k_sweep)]
    raw_matching_cisd = [c for c in cl if c["arah"] == ("BULLISH" if demand else "BEARISH") and n5 - 1 - c["k"] <= M5_EVENT_MAX_AGE and c["k"] >= max(first_touch_k, k_sweep)]
    matching_events = [e for e in raw_matching_events if not _has_opposing_m5_after(int(e["k"]))]
    matching_cisd = [c for c in raw_matching_cisd if not _has_opposing_m5_after(int(c["k"]))]

    if not matching_events and not matching_cisd:
        if raw_matching_events or raw_matching_cisd:
            stages["structure_trigger"] = {"status": "WAIT", "msg": "Struktur M5 pasca-trigger sudah berbalik arah (sinyal M5 historis tidak berlaku)"}
            W.append(stages["structure_trigger"]["msg"])
            return res
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
    zid_cur = str(getattr(zone, "zone_id", "") or "")
    trig_time_str = str(d5.index[k_ref]) if 0 <= k_ref < n5 else None
    stages["structure_trigger"] = {"status": "CONFIRMED", "msg": m5["structure_event"], "k": k_ref, "zone_id": zid_cur, "candle_time": trig_time_str}

    # 5. Displacement yang Mendukung Arah Entry
    k0 = max(first_touch_k, k_ref - 3)
    if demand:
        dir_bodies = [(x, C[x] - O[x]) for x in range(k0, min(n5, k_ref + 2)) if C[x] > O[x]]
    else:
        dir_bodies = [(x, O[x] - C[x]) for x in range(k0, min(n5, k_ref + 2)) if O[x] > C[x]]
    disp_k, disp_body = max(dir_bodies, key=lambda item: item[1]) if dir_bodies else (-1, 0.0)
    disp5 = round(float(disp_body) / atr5, 2)
    m5["displacement_atr"] = disp5

    if disp5 < M5_MIN_DISPLACEMENT_ATR:
        stages["displacement"] = {"status": "WEAK", "msg": f"Displacement M5 lemah ({disp5:.2f} ATR < {M5_MIN_DISPLACEMENT_ATR:g})", "atr": disp5}
        W.append(f"Displacement M5 lemah ({disp5:.2f} ATR < {M5_MIN_DISPLACEMENT_ATR:g})")
        return res

    disp_time_str = str(d5.index[disp_k]) if 0 <= disp_k < n5 else None
    stages["displacement"] = {"status": "CONFIRMED", "msg": f"Displacement M5 {disp5:.2f} ATR", "atr": disp5, "candle_time": disp_time_str}

    # Cek invalidasi M5 pasca-trigger
    inval = float(getattr(zone, "invalidation_price", 0.0) or (lo if demand else hi))
    if k_ref < n5:
        post_c = C[k_ref:n5]
        breached_m5 = len(post_c) > 0 and (
            (demand and float(np.min(post_c)) < inval)
            or ((not demand) and float(np.max(post_c)) > inval)
        )
        if breached_m5:
            res["status"] = "INVALID"
            msg_inv = f"Close M5 pasca-trigger menembus invalidation zona {inval:.2f}"
            stages["zone_validation"] = {"status": "INVALID", "msg": msg_inv}
            stages["entry_validation"] = {"status": "INVALID", "msg": msg_inv}
            W.append(msg_inv)
            return res

    res["smc_technical_confirmed"] = True
    m5["technical_confirmed"] = True

    bars_since_trig = (n5 - 1) - k_ref
    chase_tol = max(1.0, 0.5 * atr5)
    if bars_since_trig > 2 and (
        (demand and price > hi + chase_tol)
        or ((not demand) and price < lo - chase_tol)
    ):
        msg_missed = f"Sinyal M5 sudah historis/terlewat (trigger {bars_since_trig} bar lalu); harga telah meninggalkan zona entry"
        stages["entry_validation"] = {"status": "WAIT", "msg": msg_missed}
        res["entry_permission"] = "WAIT"
        res["is_missed"] = True
        res["signal_status"] = SIGNAL_STATUS_ENTRY_MISSED
        W.append(msg_missed)
        return res

    # 6. Risk Reward & Risk Management
    if demand:
        e_lo, e_hi = (lo, hi) if hi - lo <= ENTRY_LEBAR_MAKS else (round(hi - ENTRY_LEBAR_MAKS, 2), hi)
    else:
        e_lo, e_hi = (lo, hi) if hi - lo <= ENTRY_LEBAR_MAKS else (lo, round(lo + ENTRY_LEBAR_MAKS, 2))
    entry = round((e_lo + e_hi) / 2.0, 2)
    sl = compute_structural_sl(zone, htf_frames, float(spread or 0.0))
    if sl <= 0.0 or (demand and sl >= inval) or ((not demand) and sl <= inval):
        stages["risk_reward"] = {"status": "WAIT", "msg": "SL struktural tidak dapat ditentukan"}
        W.append("SL struktural tidak dapat ditentukan")
        return res
    risk = round((entry - sl) if demand else (sl - entry), 2)

    if risk <= 0.5:
        res["risk_rejected"] = True
        res["setup_eligibility"] = f"NOT_ELIGIBLE (SL {risk:.2f} USD terlalu sempit)"
        stages["risk_reward"] = {"status": "FAIL", "msg": "Jarak SL tidak realistis"}
        stages["entry_validation"] = {"status": "FAIL", "msg": "Setup ditolak filter risiko: Jarak SL tidak realistis"}
        W.append("Jarak SL tidak realistis")
        return res
    if risk > SL_JARAK_MAKS:
        sl_msg_rej = f"Jarak SL struktural {risk:.2f} USD melebihi batas maksimal {SL_JARAK_MAKS:g} USD"
        res["risk_rejected"] = True
        res["setup_eligibility"] = f"NOT_ELIGIBLE ({sl_msg_rej})"
        res["entry_permission"] = "WAIT"
        stages["risk_reward"] = {"status": "FAIL", "msg": sl_msg_rej}
        stages["entry_validation"] = {"status": "FAIL", "msg": f"Setup ditolak filter risiko: {sl_msg_rej}"}
        W.append(sl_msg_rej)
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

    if k_ref + 1 < n5:
        post_tp_hit = (
            bool(np.any(H[k_ref + 1 : n5] >= tp1))
            if demand
            else bool(np.any(L[k_ref + 1 : n5] <= tp1))
        )
        if post_tp_hit:
            msg_tp = f"Target TP1 {tp1:.2f} sudah tersentuh pasca-trigger M5 (sinyal historis/terlewat)"
            stages["risk_reward"] = {"status": "WAIT", "msg": msg_tp, "rr": rr}
            stages["entry_validation"] = {"status": "WAIT", "msg": msg_tp}
            res["entry_permission"] = "WAIT"
            W.append(msg_tp)
            return res

    barrier = evaluate_snr_opposing_barrier(
        "BUY" if demand else "SELL",
        entry,
        sl,
        tp1=tp1,
        current_price=price,
        snr_context=snr_context,
    )
    if barrier.get("blocked"):
        clean_rr = barrier.get("clean_rr")
        if clean_rr is not None:
            res["rr"] = clean_rr
        stages["risk_reward"] = {"status": "WAIT", "msg": barrier["reason"], "rr": res["rr"]}
        stages["entry_validation"] = {"status": "WAIT", "msg": barrier["reason"]}
        res["entry_permission"] = "WAIT"
        W.append(barrier["reason"])
        return res

    if reward <= 0 or reward / risk < RR_MIN - 1e-9:
        msg_rr = f"R:R aktual ke TP1 struktural {rr:.2f} < 1:{RR_MIN:g}"
        stages["risk_reward"] = {"status": "WAIT", "msg": msg_rr, "rr": rr}
        stages["entry_validation"] = {"status": "WAIT", "msg": msg_rr}
        res["entry_permission"] = "WAIT"
        W.append(msg_rr)
        return res

    stages["risk_reward"] = {"status": "CONFIRMED", "msg": f"1:{rr:.2f}", "rr": rr}
    stages["entry_validation"] = {"status": "PASSED", "msg": f"Setup {zone.grade} valid"}

    res["plan"] = {"arah": "BUY" if demand else "SELL", "entry_min": e_lo, "entry_max": e_hi, "entry_acuan": entry,
                   "sl": sl, "tp1": tp1, "tp2": tp2, "rr": rr, "zone_id": zone.zone_id,
                   "timeframe": zone.timeframe, "grade": zone.grade, "score": zone.score,
                   "keterangan": f"Zona {zone.grade} {zone.timeframe} {zone.zone_type} + trigger M5 {m5['structure_event']}"}
    res["status"] = "VALID"
    res["entry_permission"] = "GRANTED"
    W.append(f"Entry valid: {m5['structure_event']}, displacement M5 {disp5:.2f} ATR, R:R {rr:.2f}")
    return res


def validate_entry_setup(zone: Zone, current_price: float, df_m5: pd.DataFrame, spread_points: float = 0.0,
                         is_news_blackout=False, htf_frames: Optional[Dict[str, pd.DataFrame]] = None,
                         require_sweep: Optional[bool] = None,
                         server_time: Any = None,
                         snr_context: Optional[Dict[str, Any]] = None) -> Tuple[bool, str, Dict[str, Any]]:
    """Kompatibel dgn versi lama: (ok, pesan, plan). Status lengkap ada di plan['status'] bila ok; pesan diawali VALID/WAIT/INVALID."""
    r = _entry_check(zone, current_price, df_m5, spread_points, is_news_blackout, htf_frames, require_sweep, server_time=server_time, snr_context=snr_context)
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
    Hitung expiration pending dari waktu absolut yang dinormalisasi ke WIB.
    Jangan gunakan waktu chart broker tanpa timezone terverifikasi sebagai jam lokal.
    
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


def compute_structural_sl(zone: Zone, frames: Optional[Dict[str, pd.DataFrame]] = None,
                          spread: float = 0.0) -> float:
    """Sumber tunggal SL (dipakai pending & confirmed -> scan/pantau konsisten).
    SL = di luar invalidasi struktur ATAU wick sweep OB->BOS (mana yang lebih jauh) + buffer ATR/spread adaptif."""
    demand = (getattr(zone, "zone_type", "") == "DEMAND")
    inval = float(getattr(zone, "invalidation_price", 0.0) or 0.0)
    if inval <= 0:
        return 0.0
    base = inval
    atr = float(getattr(zone, "atr_value", 0.0) or 0.0)
    try:
        df = (frames or {}).get(str(getattr(zone, "timeframe", "") or ""))
        t0, t1 = getattr(zone, "ob_candle_time", None), getattr(zone, "bos_candle_time", None)
        if df is not None and len(df) and t0 is not None and t1 is not None:
            seg = df.loc[(df.index >= t0) & (df.index <= t1)]
            if len(seg):
                if demand:
                    base = min(base, float(seg["Low"].min()))
                else:
                    base = max(base, float(seg["High"].max()))
            if atr <= 0:
                atr = float(_atr_array(df)[-1])
    except Exception:
        pass
    buf = max(1.0, 0.3 * atr if math.isfinite(atr) else 0.0) + max(0.0, float(spread or 0.0))
    return round(base - buf if demand else base + buf, 2)


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
    h1_structure: Optional[str] = None,
    premarket: bool = False,
    opposing_zones: Optional[List[Any]] = None,
    snr_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Rencana Pending Order (BUY LIMIT / SELL LIMIT) untuk Early Setup Detection (Mode A - Aggressive Pending).
    
    Persyaratan:
    1. Filter HTF: H4 directional bias selaras (Demand wajib Bullish, Supply wajib Bearish).
    2. Filter H1: Struktur H1 tidak konflik dengan bias H4.
    3. Filter M30: Momentum tidak boleh CONFLICT, INVALID, atau WEAK.
    4. Invalidation & Freshness: Harga belum menembus batas pembatalan, zona belum termitigasi (> 2x touch).
    5. Minimum RR: Wajib memenuhi 1:1.5.
    6. Entry Price: Dihitung secara objektif dari batas zona aktual (proximal edge).
    7. Data MT5: Data candle fresh dan valid.
    8. Jarak Zona: Jarak zona dalam batas pending wajar.
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
    
    # 0. Validasi Data Freshness (MT5)
    if frames and not premarket:
        fresh_check = validate_data_freshness(frames, server_time=server_time)
        if not fresh_check.get("ok", True) or fresh_check.get("status") in ("STALE", "INVALID", "NO DATA", "NO_DATA"):
            res["status"] = "WAIT"
            res["early_pending_status"] = "WAIT"
            res["rejection_reason"] = f"Data MT5 tidak fresh / stale ({fresh_check.get('status')}: {fresh_check.get('reason')})"
            return res

    # 1. Validasi Zona Dasar & Struktur
    z_fresh = str(getattr(zone, "freshness", "FRESH")).upper()
    z_stat = str(getattr(zone, "status", "ACTIVE")).upper()
    z_tc = int(getattr(zone, "touch_count", 0))
    if not getattr(zone, "is_valid", False) or z_stat in ("INVALIDATED", "EXPIRED", "MITIGATED") or z_fresh in ("INVALIDATED", "EXPIRED"):
        res["status"] = "INVALID"
        res["early_pending_status"] = "INVALID"
        res["rejection_reason"] = f"Zona {res['zone_id']} tidak valid / invalidated: {getattr(zone, 'rejection_reason', '') or z_stat}"
        return res

    if z_fresh == "MITIGATED" or z_tc > 2 or getattr(zone, "is_expired", False):
        res["status"] = "INVALID"
        res["early_pending_status"] = "INVALID"
        res["rejection_reason"] = f"Zona {res['zone_id']} sudah termitigasi ({z_fresh}, touch count: {z_tc})"
        return res
        
    if not (isinstance(current_price, (int, float)) and math.isfinite(current_price) and current_price > 0):
        res["status"] = "WAIT"
        res["early_pending_status"] = "WAIT"
        res["rejection_reason"] = "Harga saat ini tidak valid"
        return res

    # 2. Invalidation Level Check (Closed candle TF asal = ZONE_STRUCTURALLY_INVALID; Live Bid/Ask tembus SL = SETUP_INVALID_SL_BREACHED)
    inval = res["invalidation_price"]
    trk_pend = get_zone_tracker(auto_save=False)
    pers_inv_p, pers_rec_p = trk_pend.is_zone_invalidated(zone) if trk_pend else (False, None)
    closed_inv_p, closed_rsn_p, _, _, _ = evaluate_smc_zone_closed_invalidation(zone, frames)
    eff_spr_p = round(float(ask) - float(bid), 2) if (bid and ask and ask >= bid) else 0.0
    live_inv_p, live_rsn_p, live_px_p, live_inv_lvl_p, _ = evaluate_smc_setup_live_invalidation(
        zone, current_price=current_price, bid=bid, ask=ask,
        frames=frames, spread=eff_spr_p, market_status=("CLOSED" if premarket else "OPEN"),
    )
    if live_inv_lvl_p and live_inv_lvl_p > 0:
        res["setup_invalidation_price"] = live_inv_lvl_p
    if pers_inv_p or closed_inv_p or live_inv_p or getattr(zone, "is_broken", False):
        inv_kind_p = (
            INVALIDATION_ZONE_STRUCTURAL
            if (closed_inv_p or getattr(zone, "is_broken", False))
            else (
                INVALIDATION_SETUP_SL_BREACHED
                if live_inv_p
                else str((pers_rec_p or {}).get("invalidation_type") or INVALIDATION_ZONE_STRUCTURAL)
            )
        )
        res["status"] = "INVALID"
        res["early_pending_status"] = "INVALID"
        res["invalidation_type"] = inv_kind_p
        res["rejection_reason"] = (
            closed_rsn_p
            or live_rsn_p
            or (pers_rec_p.get("reason") if isinstance(pers_rec_p, dict) and pers_rec_p.get("reason") else "")
            or f"{inv_kind_p}: Zona {res['zone_id']} telah menembus batas invalidasi ({inval:.2f})"
        )
        return res

    # 3. Filter Arah HTF (H4 & H1)
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

    h1_chk = h1_structure
    if h1_chk is None and frames:
        try:
            h1_ctx = evaluate_htf_context(frames)
            h1_chk = h1_ctx.get("H1")
        except Exception:
            pass
    if htf_bias in ("BULLISH", "BEARISH") and h1_chk in ("BULLISH", "BEARISH"):
        if (htf_bias == "BULLISH" and h1_chk == "BEARISH") or (htf_bias == "BEARISH" and h1_chk == "BULLISH"):
            res["status"] = "CONFLICT"
            res["early_pending_status"] = "CONFLICT"
            res["rejection_reason"] = f"Konflik struktur H4 ({htf_bias}) vs H1 ({h1_chk})"
            return res

    # 3b. Validasi Jarak Zona Proporsional ATR & TF Asal (Zona FAR DILARANG untuk pending intraday!)
    z_lo_val = float(getattr(zone, "price_low", 0.0))
    z_hi_val = float(getattr(zone, "price_high", 0.0))
    atr_p_val = None
    if frames:
        atr_p_diag = extract_atr_diagnostics(frames, preferred_tf="H1")
        if isinstance(atr_p_diag.get("atr_value"), (int, float)) and atr_p_diag["atr_value"] > 0:
            atr_p_val = float(atr_p_diag["atr_value"])
    if atr_p_val is None:
        z_atr_attr = getattr(zone, "atr_value", None)
        if isinstance(z_atr_attr, (int, float)) and math.isfinite(z_atr_attr) and z_atr_attr > 0:
            atr_p_val = float(z_atr_attr)

    rel_p = classify_zone_relevance(
        zone, current_price, atr=atr_p_val,
        htf_bias=htf_bias or "NETRAL",
        h1_structure=h1_chk or "NETRAL",
        m30_status=m30_status or "CONFIRMED",
        frames=frames, bid=bid, ask=ask,
    )
    d_val_res = float(rel_p.get("distance", 0.0))
    is_far_cand = (
        rel_p["relevance"] == "FAR_REFERENCE"
        or str(getattr(zone, "risk_tier", "") or "").upper() == "FAR_ZONE"
        or d_val_res > LOW_RISK_MAX_DISTANCE_PT
    )

    # Jangan memaksakan pending pada zona M5 murni bila struktur H1/M30 atau freshness tidak mendukung
    if _is_pure_m5_tf(str(getattr(zone, "timeframe", "") or "")) and htf_bias is not None:
        if z_fresh != "FRESH" or (m30_status is not None and m30_status != "CONFIRMED") or (h1_chk is not None and h1_chk != htf_bias):
            res["status"] = "WAIT"
            res["early_pending_status"] = "WAIT"
            res["rejection_reason"] = "Zona M5 membutuhkan struktur H4/H1 searah, momentum M30 CONFIRMED, dan status FRESH untuk pending intraday"
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

    # 4b. Validasi Spread MT5 (bila Bid/Ask tersedia)
    if bid is not None and ask is not None and math.isfinite(bid) and math.isfinite(ask) and bid > 0 and ask >= bid:
        curr_spread = round(float(ask) - float(bid), 2)
        if curr_spread > max(SPREAD_MAKS_POIN, 0.0):
            res["status"] = "WAIT"
            res["early_pending_status"] = "WAIT"
            res["rejection_reason"] = f"Spread {format_price_distance(curr_spread, include_mt5_points=True)} melebihi batas {SPREAD_MAKS_POIN:g} USD"
            return res

    # 5. Perhitungan Entry Price (Batas Proximal Edge Zona Aktual)
    # BUY LIMIT: Batas atas zona Demand (price_high), wajib < Ask (atau current_price)
    # SELL LIMIT: Batas bawah zona Supply (price_low), wajib > Bid (atau current_price)
    if demand:
        entry = round(res["zone_high"], 2)
        entry_reason = f"Batas atas (proximal edge) zona Demand {getattr(zone, 'timeframe', 'UNKNOWN')} untuk mitigasi awal"
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
        entry_reason = f"Batas bawah (proximal edge) zona Supply {getattr(zone, 'timeframe', 'UNKNOWN')} untuk mitigasi awal"
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
    _spr = round(float(ask) - float(bid), 2) if (bid and ask and ask >= bid) else 0.0
    sl = compute_structural_sl(zone, frames, _spr)
    if sl <= 0.0 or (demand and sl >= inval) or ((not demand) and sl <= inval):
        res["status"] = "WAIT"
        res["early_pending_status"] = "WAIT"
        res["rejection_reason"] = "SL struktural tidak dapat ditentukan (invalidasi zona tidak valid)"
        return res
    inval_buffer = round(inval - sl, 2) if demand else round(sl - inval, 2)
    
    risk_points = round((entry - sl) if demand else (sl - entry), 2)
    if risk_points <= 0.5:
        res["status"] = "INVALID"
        res["early_pending_status"] = "INVALID"
        res["rejection_reason"] = f"Jarak SL ({risk_points:.2f} USD) terlalu sempit / tidak realistis"
        return res
    if risk_points > SL_JARAK_MAKS:
        res["status"] = "INVALID"
        res["early_pending_status"] = "INVALID"
        res["rejection_reason"] = f"Jarak SL ({risk_points:.2f} USD) melebihi batas toleransi maksimal ({SL_JARAK_MAKS:g} USD)"
        return res

    # 7. Take Profit Struktural & Risk/Reward
    ref_atr_z = getattr(zone, "atr_value", None)
    if not ref_atr_z and frames and "H1" in frames and "ATR14" in frames["H1"].columns:
        ref_atr_z = float(frames["H1"]["ATR14"].iloc[-1])
    targets = _structural_targets(
        frames or {}, getattr(zone, "zone_type", ""), entry,
        opposing_zones=opposing_zones,
        snr_context=snr_context,
        atr=ref_atr_z,
    ) if frames else []
    if not targets:
        res["status"] = "REJECTED"
        res["early_pending_status"] = "REJECTED"
        res["entry_price"] = entry
        res["sl"] = sl
        res["rejection_reason"] = "NO_TARGET: Tidak ada target likuiditas struktural searah transaksi"
        return res
    buf = 0.3
    if demand:
        tp1 = round(targets[0] - buf, 2)
        tp2 = round(targets[1] - buf, 2) if len(targets) > 1 else tp1
        reward_tp1 = round(tp1 - entry, 2)
        reward_tp2 = round(tp2 - entry, 2)
    else:
        tp1 = round(targets[0] + buf, 2)
        tp2 = round(targets[1] + buf, 2) if len(targets) > 1 else tp1
        reward_tp1 = round(entry - tp1, 2)
        reward_tp2 = round(entry - tp2, 2)

    rr_tp1 = round(reward_tp1 / risk_points, 2) if risk_points > 0 else 0.0
    rr_tp2 = round(reward_tp2 / risk_points, 2) if risk_points > 0 else 0.0
    if rr_tp1 > 8.0:
        res["rr_justification"] = f"TP1 struktural {tp1:.2f} (R:R 1:{rr_tp1:.2f})"

    if reward_tp1 <= 0 or reward_tp1 / risk_points < RR_MIN - 1e-9:
        res["status"] = "REJECTED"
        res["early_pending_status"] = "REJECTED"
        res["early_pending_rr"] = "FAILED"
        res["entry_price"] = entry
        res["sl"] = sl
        res["tp1"] = tp1
        res["rr"] = rr_tp1
        res["rr_tp1"] = rr_tp1
        res["rejection_reason"] = f"RR_TOO_LOW: Risk/Reward ke TP1 struktural (1:{rr_tp1:.2f}) di bawah minimum 1:{RR_MIN:g}"
        return res

    # Pending-plan lifecycle memakai jam absolut lokal WIB, bukan wall-clock candle broker.
    # server_time dapat berupa jam chart MT5 (offset broker tidak diketahui).
    # created_at historis yang sudah disimpan tetap dihormati tanpa konversi spekulatif.
    base_ref = created_at if created_at is not None else get_current_wib()
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
        "stop_loss": f"Level proteksi modal ({sl:.2f}) yang diletakkan di luar invalidation dengan buffer ({inval_buffer:.2f} USD) untuk menghindari market noise/spread",
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
    if is_far_cand:
        res["is_far"] = True
        res["risk_tier"] = "FAR_ZONE"
        if premarket:
            res["valid"] = True
            res["provisional"] = True
            res["status"] = "PROVISIONAL"
            res["early_pending_status"] = "PROVISIONAL"
            res["early_pending_status_label"] = "PROVISIONAL (FAR_ZONE)"
        else:
            res["valid"] = False
            res["status"] = "WATCHLIST"
            res["early_pending_status"] = "WATCHLIST"
            res["early_pending_status_label"] = "WATCHLIST (FAR_ZONE)"
            res["rejection_reason"] = f"FAR_ZONE: Jarak zona jauh ({d_val_res:.2f} USD); masuk WATCHLIST"
    elif premarket:
        res["valid"] = True
        res["provisional"] = True
        res["status"] = "PROVISIONAL"
        res["early_pending_status"] = "PROVISIONAL"
        res["early_pending_status_label"] = "PROVISIONAL (MARKET CLOSED)"
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

    ctx_m30_status = str(ctx.get("m30_status") or "").upper()
    if ctx.get("m30_conflict") or ctx_m30_status == "CONFLICT":
        m30_status = "CONFLICT"
        ctx_m30_msg = (ctx.get("m30_condition") or {}).get("msg") or m30_val.get("msg") or f"M30 bertentangan dengan H4 {h4_bias}"
        m30_val = {**m30_val, "status": "CONFLICT", "ok": False, "msg": ctx_m30_msg}
    elif ctx_m30_status in ("INVALID", "WEAK") and m30_status not in ("CONFLICT", "INVALID", "WEAK"):
        m30_status = ctx_m30_status
        ctx_m30_msg = (ctx.get("m30_condition") or {}).get("msg") or m30_val.get("msg") or ""
        m30_val = {**m30_val, "status": m30_status, "ok": False, "msg": ctx_m30_msg}

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

    if str(scan.get("market_status") or "").upper() == "CLOSED" or str(scan.get("snapshot_status") or "").upper() == "MARKET_CLOSED":
        out["empty_reason"] = "MARKET CLOSED — seluruh rekomendasi pending & confirmed entry tetap WAIT sampai revalidasi market OPEN"
        return out

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
    atr_early = (scan.get("atr_diagnostics") or {}).get("atr_value")
    if atr_early is None and frames:
        atr_early = extract_atr_diagnostics(frames, preferred_tf="H1").get("atr_value")
    h1_aligned_early = (h1_struct == h4_bias)

    raw_cands = [z for z in scan.get("active_zones", []) if z.get("zone_type") == wanted_type]
    cands = []
    far_skipped_reasons = []
    for zd in raw_cands:
        live_inv_e, _, _, _, _ = evaluate_smc_setup_live_invalidation(
            zd, current_price=float(current_price), bid=scan.get("bid"), ask=scan.get("ask"),
            frames=frames, market_status=scan.get("market_status"),
        )
        if live_inv_e:
            continue
        rel_e = classify_zone_relevance(
            zd, float(current_price), atr=atr_early,
            htf_bias=h4_bias, h1_structure=h1_struct, m30_status=m30_status,
            frames=frames, bid=scan.get("bid"), ask=scan.get("ask"),
        )
        if rel_e["relevance"] == "INVALID":
            continue
        if rel_e["relevance"] == "FAR_REFERENCE" or str(zd.get("risk_tier") or "").upper() == "FAR_ZONE":
            far_skipped_reasons.append(rel_e["reason"])
            continue
        if str(zd.get("grade") or "") not in ("A+", "A") and float(zd.get("score", 0.0) or 0.0) < LOW_RISK_MIN_SCORE:
            continue
        if str(zd.get("freshness") or "FRESH").upper() not in ("FRESH", "TESTED_1X"):
            continue
        zd_copy = dict(zd)
        zd_copy["zone_relevance"] = rel_e["relevance"]
        zd_copy["distance"] = rel_e["distance"]
        cands.append(zd_copy)

    cands.sort(key=lambda z: (
        0 if z.get("zone_relevance") == "ACTIVE" else 1,
        0 if str(z.get("freshness") or "").upper() == "FRESH" else 1,
        _intraday_tf_rank(str(z.get("timeframe") or ""), m30_status=m30_status, h1_aligned=h1_aligned_early),
        0 if str(z.get("grade") or "") == "A+" else 1,
        -round(float(z.get("score", 0.0) or 0.0), 1),
        round(float(z.get("distance", 9999.0) or 9999.0), 2),
    ))
    cands, _ = _deduplicate_overlapping_zones(cands)

    if not cands:
        if far_skipped_reasons:
            out["empty_reason"] = f"Zona searah berada di luar jangkauan intraday (FAR REFERENCE — dilarang pending intraday): {far_skipped_reasons[0]}"
        else:
            out["empty_reason"] = f"Tidak ada zona {wanted_type} berkualitas dalam jangkauan intraday saat ini"
        return out

    valid_plans = []
    rejection_reasons = []
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
            created_at=prev_created_at, target_time=prev_expires_at,
            bid=scan.get("bid"), ask=scan.get("ask"), h1_structure=h1_struct,
        )
        if plan.get("valid"):
            if not any(_zones_substantially_overlap(ep, plan) for ep in valid_plans):
                valid_plans.append(plan)
        elif plan.get("rejection_reason"):
            rejection_reasons.append(str(plan["rejection_reason"]))

    out["all_setups"] = valid_plans
    if valid_plans:
        out["best_setup"] = valid_plans[0]
    else:
        out["empty_reason"] = (
            rejection_reasons[0]
            if rejection_reasons
            else "Syarat pending order belum terpenuhi"
        )
    return out


# =============================================================================
# ZONE TRACKING SYSTEM
# =============================================================================
class ZoneTracker:
    INVALID_STATE_KEY = "invalidated_smc_zones"

    def __init__(self, filepath: Optional[str] = FILE_STATE_ZONE_TRACKER, auto_save: bool = False):
        self.filepath = filepath
        self.auto_save = auto_save
        self.tracked_zones: Dict[str, Dict[str, Any]] = {}
        self.invalidated_zones: Dict[str, Dict[str, Any]] = {}
        self.alerts: List[Dict[str, Any]] = []
        self.extra_state: Dict[str, Any] = {}
        if self.filepath and os.path.exists(self.filepath):
            self.load()

    def get_zone(self, zone_id: str) -> Optional[Dict[str, Any]]:
        return self.tracked_zones.get(zone_id)

    def mark_invalidated(
        self,
        zone: Union[Zone, Dict[str, Any]],
        reason: str = INVALIDATION_ZONE_STRUCTURAL,
        break_close: Optional[float] = None,
        break_time: Optional[str] = None,
        origin_tf: Optional[str] = None,
        invalidation_type: str = INVALIDATION_ZONE_STRUCTURAL,
        break_price: Optional[float] = None,
    ) -> None:
        """Simpan status invalidasi secara permanen agar zona/setup lama tidak muncul kembali tanpa setup/zone ID baru."""
        zd = zone.to_dict() if hasattr(zone, "to_dict") else (zone if isinstance(zone, dict) else {})
        if not zd:
            return
        inv_key = _smc_zone_invalid_key(zd)
        z_id = str(zd.get("zone_id") or inv_key).strip()
        inv_type_u = str(invalidation_type or INVALIDATION_ZONE_STRUCTURAL).upper()
        if inv_type_u not in (INVALIDATION_SETUP_SL_BREACHED, INVALIDATION_ZONE_STRUCTURAL):
            inv_type_u = INVALIDATION_ZONE_STRUCTURAL
        now_str = get_current_wib().strftime("%Y-%m-%d %H:%M:%S WIB")
        struct_inv_px = float(
            zd.get("structure_invalidation")
            or zd.get("structural_invalidation")
            or zd.get("invalidation_price")
            or 0.0
        )
        setup_inv_px = float(
            zd.get("setup_sl")
            or zd.get("setup_invalidation_price")
            or zd.get("sl")
            or 0.0
        )
        orig_grade = str(zd.get("original_grade") or "").strip()
        raw_grade = str(zd.get("grade") or "").strip()
        saved_grade = orig_grade if orig_grade else (raw_grade if raw_grade.upper() != "REJECTED" else "")
        saved_score = float(zd.get("score") or 0.0) if zd.get("score") is not None else None
        origin_reason_txt = str(zd.get("origin_reason") or "").strip()
        if not origin_reason_txt:
            try:
                origin_reason_txt = build_smc_zone_reason(zd)
            except Exception:
                origin_reason_txt = ""
        rec_inv = {
            "zone_id": z_id,
            "key": inv_key,
            "symbol": str(zd.get("symbol") or SYMBOL),
            "timeframe": str(origin_tf or zd.get("timeframe") or ""),
            "zone_type": str(zd.get("zone_type") or "").upper(),
            "price_low": float(zd.get("price_low", zd.get("low", 0.0)) or 0.0),
            "price_high": float(zd.get("price_high", zd.get("high", 0.0)) or 0.0),
            "invalidation_price": struct_inv_px,
            "structure_invalidation": struct_inv_px,
            "setup_invalidation_price": setup_inv_px,
            "setup_sl": setup_inv_px,
            "grade": saved_grade,
            "original_grade": saved_grade,
            "score": saved_score,
            "origin_reason": origin_reason_txt,
            "status": "INVALIDATED",
            "invalidation_type": inv_type_u,
            "invalid_label": inv_type_u,
            "is_broken": (inv_type_u == INVALIDATION_ZONE_STRUCTURAL),
            "reason": str(reason or inv_type_u),
            "break_close": float(break_close) if break_close is not None else None,
            "break_price": float(break_price) if break_price is not None else (float(break_close) if break_close is not None else None),
            "break_time": str(break_time) if break_time else None,
            "invalidated_at": now_str,
        }
        geo_key = _smc_zone_geo_key(zd)
        rec_inv["geo_key"] = geo_key
        if inv_type_u == INVALIDATION_ZONE_STRUCTURAL:
            self.invalidated_zones[geo_key] = rec_inv
        self.invalidated_zones[inv_key] = rec_inv
        if z_id and z_id != inv_key:
            self.invalidated_zones[z_id] = rec_inv
        if len(self.invalidated_zones) > 500:
            self.invalidated_zones = dict(list(self.invalidated_zones.items())[-500:])
        if z_id in self.tracked_zones:
            tr = self.tracked_zones[z_id]
            tr["status"] = "INVALIDATED"
            tr["lifecycle_status"] = "INVALIDATED"
            tr["invalidation_type"] = inv_type_u
            tr["invalid_label"] = inv_type_u
            tr["is_broken"] = (inv_type_u == INVALIDATION_ZONE_STRUCTURAL)
            tr["pending_plan"] = None
            tr["rejection_reason"] = rec_inv["reason"]
            tr["last_updated"] = now_str

    def is_zone_invalidated(self, zone: Union[Zone, Dict[str, Any]]) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Memeriksa apakah zona sudah pernah ditandai INVALID — BROKEN di state tersimpan (tolak artefak pra-pembentukan)."""
        zd = zone.to_dict() if hasattr(zone, "to_dict") else (zone if isinstance(zone, dict) else {})
        if not zd:
            return False, None
        z_id = str(zd.get("zone_id") or "").strip()
        inv_key = _smc_zone_invalid_key(zd)
        geo_key = _smc_zone_geo_key(zd)
        for k_chk in (inv_key, z_id, geo_key):
            if k_chk and k_chk in self.invalidated_zones:
                rec_inv = self.invalidated_zones[k_chk]
                if not _is_breach_invalidation_valid(zd, rec_inv):
                    # Koreksi state tercatat (satu-satunya jalur pemulihan zona)
                    self.invalidated_zones.pop(k_chk, None)
                    _corr = self.extra_state.setdefault("invalid_corrections", [])
                    if isinstance(_corr, list):
                        _corr.append({
                            "key": k_chk, "zone_id": z_id,
                            "break_time": rec_inv.get("break_time"),
                            "reason": "catatan invalidasi tidak valid (pra-pembentukan / break_close tidak menembus distal)",
                            "corrected_at": get_current_wib().strftime("%Y-%m-%d %H:%M:%S WIB"),
                        })
                        del _corr[:-50]
                    continue
                if k_chk == geo_key and k_chk != inv_key and k_chk != z_id:
                    # Geometri sama tetapi terbentuk SETELAH break = zona baru, bukan kemunculan ulang
                    try:
                        _o, _ = _resolve_smc_zone_origin_time(zd, work_df=None)
                        _bt = _parse_ts_utc_naive(rec_inv.get("break_time"))
                        if _o is not None and not pd.isna(_o) and _bt is not None and not pd.isna(_bt) and _o > _bt:
                            continue
                    except Exception:
                        pass
                return True, rec_inv
        for k_src in (zd.get("merged_from") or []):
            if k_src in self.invalidated_zones:
                rec_src = self.invalidated_zones[k_src]
                if not _is_breach_invalidation_valid({"zone_id": k_src, "zone_type": zd.get("zone_type")}, rec_src):
                    self.invalidated_zones.pop(k_src, None)
                    continue
                return True, rec_src
        tr = self.tracked_zones.get(z_id) if z_id else None
        if isinstance(tr, dict) and (tr.get("is_broken") is True or str(tr.get("invalidation_type") or "").upper() in (INVALIDATION_SETUP_SL_BREACHED, INVALIDATION_ZONE_STRUCTURAL)):
            if _is_breach_invalidation_valid(zd, tr):
                return True, tr
        return False, None

    def update(
        self,
        zones: List[Any],
        current_price: float,
        frames: Optional[Dict[str, pd.DataFrame]] = None,
        htf_bias: Optional[str] = None,
        m30_status: Optional[str] = None,
        server_time: Any = None,
        conservative_dec: Optional[Dict[str, Any]] = None,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
        spread: float = 0.0,
        market_status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perbarui siklus hidup zona:
        DETECTED -> READY -> APPROACHING -> TOUCHED -> CONFIRMED -> INVALIDATED / EXPIRED.
        Menghasilkan notifikasi/alert dan mencegah duplikasi identitas zona.
        """
        # Waktu pencatatan lifecycle independen dari timezone server chart MT5.
        now_dt = to_wib_datetime(server_time) or get_current_wib()
        now_str = now_dt.strftime("%Y-%m-%d %H:%M:%S WIB")
        new_alerts: List[Dict[str, Any]] = []
        created_any = False

        # 1. Daftarkan zona baru tanpa duplikasi (tolak zona yang sudah INVALID — BROKEN)
        for z in (zones or []):
            z_id = getattr(z, "zone_id", None) if not isinstance(z, dict) else z.get("zone_id")
            if not z_id:
                continue
            already_inv, _ = self.is_zone_invalidated(z)
            if already_inv:
                continue
            z_obj = z if isinstance(z, Zone) else _zone_from_dict(z)
            if z_id not in self.tracked_zones:
                created_any = True
                exp_dt = now_dt + timedelta(hours=DEFAULT_PENDING_EXPIRATION_HOURS)
                z_lo = float(getattr(z_obj, "price_low", 0.0))
                z_hi = float(getattr(z_obj, "price_high", 0.0))
                z_type_str = str(getattr(z_obj, "zone_type", "")).upper()
                z_dir_str = "SELL" if z_type_str == "SUPPLY" else ("BUY" if z_type_str == "DEMAND" else "NONE")
                z_stype_str = str((z.get("setup_type") if isinstance(z, dict) else getattr(z_obj, "setup_type", None)) or "TREND_PULLBACK")
                z_form_t = str((z.get("formation_time") if isinstance(z, dict) else getattr(z_obj, "formation_time", None)) or "")
                self.tracked_zones[z_id] = {
                    "setup_id": f"{z_id}:{z_dir_str}:{z_stype_str}",
                    "zone_id": z_id,
                    "symbol": getattr(z_obj, "symbol", SYMBOL),
                    "direction": z_dir_str,
                    "timeframe": getattr(z_obj, "timeframe", ""),
                    "bos_timeframe": getattr(z_obj, "bos_timeframe", "") or getattr(z_obj, "timeframe", ""),
                    "bos_candle_time": str(getattr(z_obj, "bos_candle_time", "") or ""),
                    "ob_candle_time": str(getattr(z_obj, "ob_candle_time", "") or ""),
                    "formation_time": z_form_t,
                    "zone_type": getattr(z_obj, "zone_type", ""),
                    "price_low": z_lo,
                    "price_high": z_hi,
                    "zone_range": (z_lo, z_hi),
                    "setup_type": z_stype_str,
                    "invalidation_price": float(getattr(z_obj, "invalidation_price", 0.0)),
                    "grade": getattr(z_obj, "grade", ""),
                    "score": float(getattr(z_obj, "score", 0.0)),
                    "first_detected": now_str,
                    "created_at": now_str,
                    "last_updated": now_str,
                    "status": "RADAR",
                    "lifecycle_status": "RADAR",
                    "touch_status": "NOT_TOUCHED",
                    "expires_at": exp_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "pending_plan": None,
                    "history": [{"status": "RADAR", "time": now_str, "reason": "Zona teridentifikasi sebagai RADAR oleh scanner"}],
                    "rejection_reason": "",
                    "merged_from": list(z.get("merged_from") or []) if isinstance(z, dict) else list(getattr(z, "merged_from", []) or []),
                    "monitor_only": bool(z.get("monitor_only")) if isinstance(z, dict) else False,
                }
            else:
                self.tracked_zones[z_id]["last_updated"] = now_str
                if isinstance(z, dict) and z.get("merged_from"):
                    self.tracked_zones[z_id]["merged_from"] = list(z["merged_from"])
                if not self.tracked_zones[z_id].get("bos_candle_time") and getattr(z_obj, "bos_candle_time", None):
                    self.tracked_zones[z_id]["bos_candle_time"] = str(getattr(z_obj, "bos_candle_time"))
                    self.tracked_zones[z_id]["bos_timeframe"] = str(getattr(z_obj, "bos_timeframe", "") or getattr(z_obj, "timeframe", ""))
                if not self.tracked_zones[z_id].get("ob_candle_time") and getattr(z_obj, "ob_candle_time", None):
                    self.tracked_zones[z_id]["ob_candle_time"] = str(getattr(z_obj, "ob_candle_time"))
                if not self.tracked_zones[z_id].get("formation_time") and isinstance(z, dict) and z.get("formation_time"):
                    self.tracked_zones[z_id]["formation_time"] = str(z["formation_time"])
                if "setup_id" not in self.tracked_zones[z_id]:
                    z_type_str = str(self.tracked_zones[z_id].get("zone_type", "")).upper()
                    z_dir_str = "SELL" if z_type_str == "SUPPLY" else ("BUY" if z_type_str == "DEMAND" else "NONE")
                    z_stype_str = str(self.tracked_zones[z_id].get("setup_type", "TREND_PULLBACK") or "TREND_PULLBACK")
                    self.tracked_zones[z_id]["setup_id"] = f"{z_id}:{z_dir_str}:{z_stype_str}"
                    self.tracked_zones[z_id]["direction"] = z_dir_str
                    self.tracked_zones[z_id]["zone_range"] = (self.tracked_zones[z_id].get("price_low", 0.0), self.tracked_zones[z_id].get("price_high", 0.0))

        # 2. Evaluasi transisi state untuk seluruh zona yang sedang dilacak
        for z_id, rec in list(self.tracked_zones.items()):
            curr_st = rec.get("status", "RADAR")
            z_type = rec.get("zone_type", "")
            inval = rec.get("invalidation_price", 0.0)

            # A. Cek Expiration (hanya mengkedaluwarsakan rencana pending / monitor aktif — JANGAN membuang zona struktural utuh saat harga menjauh)
            exp_str = rec.get("expires_at", "")
            try:
                exp_dt = to_wib_datetime(exp_str)
                is_expired = exp_dt is not None and now_dt > exp_dt
            except (TypeError, ValueError):
                is_expired = False

            has_active_plan_or_monitor = bool(
                rec.get("pending_plan")
                or rec.get("monitor_only")
                or curr_st in ("READY", "APPROACHING", "ZONE_TOUCHED", "CONFIRMED")
            )
            if is_expired and curr_st != "EXPIRED":
                if has_active_plan_or_monitor:
                    rec["status"] = "EXPIRED"
                    rec["lifecycle_status"] = "EXPIRED"
                    rec["pending_plan"] = None
                    rec["rejection_reason"] = f"Kedaluwarsa rencana pending (batas waktu {exp_str})"
                    rec["history"].append({"status": "EXPIRED", "time": now_str, "reason": rec["rejection_reason"]})
                    msg = f"[ALERT] Pending plan zona {z_id} telah EXPIRED"
                    new_alerts.append({"type": "EXPIRED", "zone_id": z_id, "msg": msg, "time": now_str})
                    continue
                else:
                    # Zona struktural RADAR tanpa pending aktif yang belum broken tetap valid -> perbarui jendela tracking
                    rec["expires_at"] = (now_dt + timedelta(hours=DEFAULT_PENDING_EXPIRATION_HOURS)).strftime("%Y-%m-%d %H:%M:%S")
                    exp_dt = now_dt + timedelta(hours=DEFAULT_PENDING_EXPIRATION_HOURS)

            remaining_seconds = (exp_dt - now_dt).total_seconds() if exp_dt is not None else None
            if (
                remaining_seconds is not None
                and 0 < remaining_seconds <= PANTAU_EXPIRY_WARNING_MINUTES * 60
                and not rec.get("expiry_warning_emitted")
            ):
                rec["expiry_warning_emitted"] = True
                remaining_minutes = max(1, int(remaining_seconds // 60))
                msg = f"[WARN] Pending plan {z_id} mendekati expiration ({remaining_minutes} menit)"
                new_alerts.append({"type": "EXPIRING", "zone_id": z_id, "msg": msg, "time": now_str})

            if curr_st in ("EXPIRED", "INVALIDATED"):
                continue

            # B. Cek Invalidation Level:
            # 1) Struktur zona: closed candle TF asal -> ZONE_STRUCTURALLY_INVALID (wick tidak membatalkan struktur)
            # 2) Setup aktif: live Bid/Ask tembus SL/invalidasi setup -> SETUP_INVALID_SL_BREACHED
            inv_reason = None
            inv_kind_t = ""
            is_price_broken = False
            b_close_val, b_time_val, b_tf_val, b_px_val = None, None, None, None
            if isinstance(frames, dict):
                is_price_broken, closed_rsn_t, b_close_val, b_time_val, b_tf_val = evaluate_smc_zone_closed_invalidation(rec, frames)
                if is_price_broken:
                    inv_reason = closed_rsn_t
                    inv_kind_t = INVALIDATION_ZONE_STRUCTURAL
                    b_px_val = b_close_val

            if not is_price_broken:
                live_inv_t, live_rsn_t, live_px_t, live_inv_lvl_t, _ = evaluate_smc_setup_live_invalidation(
                    rec, current_price=current_price, bid=bid, ask=ask,
                    frames=frames, spread=spread, market_status=market_status,
                )
                if live_inv_lvl_t and live_inv_lvl_t > 0:
                    rec["setup_invalidation_price"] = live_inv_lvl_t
                if live_inv_t:
                    is_price_broken = True
                    inv_reason = live_rsn_t
                    inv_kind_t = INVALIDATION_SETUP_SL_BREACHED
                    b_px_val = live_px_t

            if is_price_broken and inv_reason:
                rec["status"] = "INVALIDATED"
                rec["lifecycle_status"] = "INVALIDATED"
                rec["pending_plan"] = None
                rec["rejection_reason"] = inv_reason
                rec["invalidation_type"] = inv_kind_t
                rec["is_broken"] = (inv_kind_t == INVALIDATION_ZONE_STRUCTURAL)
                rec["invalid_label"] = inv_kind_t
                self.mark_invalidated(
                    rec,
                    inv_reason,
                    break_close=b_close_val,
                    break_time=b_time_val or now_str,
                    origin_tf=b_tf_val or rec.get("timeframe"),
                    invalidation_type=inv_kind_t,
                    break_price=b_px_val,
                )
                rec["history"].append({"status": "INVALIDATED", "time": now_str, "reason": inv_reason})
                msg = f"[ALERT] Zona {z_id} {inv_kind_t}: {inv_reason}"
                new_alerts.append({"type": "INVALIDATED", "zone_id": z_id, "msg": msg, "time": now_str})
                continue

            context_pause_reason = None
            if z_type == "DEMAND" and htf_bias == "BEARISH":
                context_pause_reason = "Bias H4 berbalik menjadi BEARISH (pending DEMAND ditangguhkan; struktur zona tetap utuh)"
            elif z_type == "SUPPLY" and htf_bias == "BULLISH":
                context_pause_reason = "Bias H4 berbalik menjadi BULLISH (pending SUPPLY ditangguhkan; struktur zona tetap utuh)"
            elif m30_status == "CONFLICT":
                context_pause_reason = "Momentum M30 CONFLICT (pending ditangguhkan sementara; struktur zona tetap utuh)"
            if context_pause_reason:
                rec["pending_plan"] = None
                if rec.get("status") == "READY":
                    rec["status"] = "RADAR"
                    rec["lifecycle_status"] = "RADAR"
                rec["rejection_reason"] = context_pause_reason
                continue

            # C. Hitung Pending Order Plan & Update Status
            z_mock = Zone(
                zone_id=z_id, symbol=rec["symbol"], timeframe=rec["timeframe"], zone_type=rec["zone_type"],
                price_low=rec["price_low"], price_high=rec["price_high"], midpoint=(rec["price_low"] + rec["price_high"]) / 2,
                score=rec["score"], grade=rec["grade"], origin_bos={}, fvg_confirmation=None, displacement_strength=1.5,
                freshness="FRESH", touch_count=0, liquidity_sweep_status="NONE", invalidation_price=inval,
                is_valid=True, rejection_reason="",
                ob_candle_time=rec.get("ob_candle_time") or rec.get("formation_time") or None,
                bos_candle_time=rec.get("bos_candle_time") or None,
                bos_timeframe=rec.get("bos_timeframe") or rec.get("timeframe", ""),
                sl_recommended=(rec["price_low"] - 1.5 if z_type == "DEMAND" else rec["price_high"] + 1.5),
            )
            prev_plan = rec.get("pending_plan") or {}
            prev_c_at = prev_plan.get("created_at") or rec.get("first_detected")
            prev_e_at = prev_plan.get("expires_at") or rec.get("expires_at")
            plan = (
                {"valid": False, "rejection_reason": "Monitoring setup; bukan kandidat pending planner."}
                if rec.get("monitor_only")
                else plan_pending_order(
                    z_mock, current_price, frames, htf_bias, m30_status, server_time,
                    created_at=prev_c_at, target_time=prev_e_at
                )
            )
            if plan.get("valid"):
                rec["pending_plan"] = plan
                rec["pending_rejection_reason"] = ""
                if rec["status"] in ("RADAR", "DETECTED"):
                    rec["status"] = "READY"
                    rec["history"].append({"status": "READY", "time": now_str, "reason": "Memenuhi kriteria Aggressive Pending Order"})
                    msg = f"[INFO] Zona {z_id} READY untuk {plan['order_type']} @ {plan['entry_price']:.2f}"
                    new_alerts.append({"type": "READY", "zone_id": z_id, "msg": msg, "time": now_str})
            else:
                rec["pending_plan"] = None
                rec["pending_rejection_reason"] = plan.get("rejection_reason", "")
                if rec["status"] == "READY":
                    rec["status"] = "RADAR"
                    rec["lifecycle_status"] = "RADAR"

            # D. Cek Jarak & Touch berbasis jarak aktual ke batas zona
            z_lo, z_hi = rec["price_low"], rec["price_high"]
            is_touched = (z_lo <= current_price <= z_hi)
            dist = 0.0 if is_touched else round(z_lo - current_price if current_price < z_lo else current_price - z_hi, 2)
            approach_thresh = float(ZONE_APPROACH_DISTANCE_POINTS)

            if is_touched:
                rec["touch_status"] = "IN_ZONE"
                if rec["status"] in ("RADAR", "DETECTED", "READY", "APPROACHING"):
                    rec["status"] = ZoneStatus("ZONE_TOUCHED")
                    rec["lifecycle_status"] = "ZONE_TOUCHED"
                    rec["signal_status"] = SIGNAL_STATUS_WAITING_CONFIRMATION
                    rec["history"].append({"status": "ZONE_TOUCHED", "time": now_str, "reason": f"Harga ({current_price:.2f}) menyentuh zona {z_lo:.2f}-{z_hi:.2f}"})
                    msg = f"[ALERT] Harga telah menyentuh zona {z_id} ({current_price:.2f})"
                    new_alerts.append({"type": "ZONE_TOUCHED", "zone_id": z_id, "msg": msg, "time": now_str})
            else:
                if rec.get("status") in ("ZONE_TOUCHED", "TOUCHED") or rec.get("touch_status") in ("IN_ZONE", "TOUCHED_PREVIOUSLY"):
                    rec["touch_status"] = "TOUCHED_PREVIOUSLY"
                if dist <= approach_thresh:
                    if rec["status"] in ("RADAR", "DETECTED", "READY", "ZONE_TOUCHED", "TOUCHED"):
                        prev_st_d = rec["status"]
                        rec["status"] = "APPROACHING"
                        rec["lifecycle_status"] = "APPROACHING"
                        rec["signal_status"] = SIGNAL_STATUS_WAITING_PULLBACK
                        if prev_st_d not in ("ZONE_TOUCHED", "TOUCHED"):
                            dist_trk_txt = format_price_distance(dist, include_mt5_points=True)
                            rec["history"].append({"status": "APPROACHING", "time": now_str, "reason": f"Harga ({current_price:.2f}) mendekati zona (jarak {dist_trk_txt})"})
                            msg = f"[ALERT] Harga mendekati zona {z_id} ({current_price:.2f}, jarak {dist_trk_txt})"
                            new_alerts.append({"type": "APPROACHING", "zone_id": z_id, "msg": msg, "time": now_str})
                elif rec["status"] in ("APPROACHING", "ZONE_TOUCHED", "TOUCHED"):
                    reset_st = "READY" if rec.get("pending_plan") else "RADAR"
                    rec["status"] = reset_st
                    rec["lifecycle_status"] = reset_st
                    rec["signal_status"] = SIGNAL_STATUS_WAITING_PULLBACK

            # E. Cek Conservative Confirmation (M5)
            if conservative_dec:
                c_zone = conservative_dec.get("zone") or {}
                c_zid = c_zone.get("zone_id") if isinstance(c_zone, dict) else getattr(c_zone, "zone_id", None)
                if conservative_dec.get("status") == "VALID" and c_zid == z_id and rec["status"] != "CONFIRMED":
                    rec["status"] = "CONFIRMED"
                    rec["lifecycle_status"] = "CONFIRMED"
                    rec["signal_status"] = str(conservative_dec.get("signal_status") or SIGNAL_STATUS_SIGNAL_CONFIRMED)
                    rec["setup_type"] = str(conservative_dec.get("setup_type") or "TREND_PULLBACK")
                    rec["history"].append({"status": "CONFIRMED", "time": now_str, "reason": "Konfirmasi M5 terpenuhi (Trend Following + SMC Confirmation)"})
                    msg = f"[ALERT] Setup terkonfirmasi di zona {z_id}! (Trigger M5 terverifikasi)"
                    new_alerts.append({"type": "CONFIRMED", "zone_id": z_id, "msg": msg, "time": now_str})
                elif rec["status"] == "CONFIRMED" and (conservative_dec.get("status") != "VALID" or c_zid != z_id):
                    rec["status"] = "ZONE_TOUCHED" if rec.get("touch_status") in ("IN_ZONE", "TOUCHED_PREVIOUSLY") else "RADAR"
                    rec["lifecycle_status"] = rec["status"]
                    rec["signal_status"] = SIGNAL_STATUS_WAITING_CONFIRMATION if rec["status"] == "ZONE_TOUCHED" else SIGNAL_STATUS_WAITING_PULLBACK

            if "signal_status" not in rec:
                rec["signal_status"] = (
                    SIGNAL_STATUS_SETUP_DETECTED
                    if rec.get("status") == "RADAR"
                    else (SIGNAL_STATUS_WAITING_PULLBACK if rec.get("status") in ("READY", "APPROACHING") else str(rec.get("status")))
                )

        self.alerts.extend(new_alerts)
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]

        if self.auto_save and self.filepath and (created_any or new_alerts):
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
        self.invalidated_zones = {}
        self.alerts = []
        if self.filepath and os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, dict):
                    raise ValueError("Shared zone state harus berupa object JSON.")
                self.extra_state = {
                    key: value for key, value in data.items()
                    if key not in ("tracked_zones", "alerts", self.INVALID_STATE_KEY)
                }
                if self.extra_state:
                    self.save()
                else:
                    os.remove(self.filepath)
            except Exception as e:
                _catat_error("zone_tracker_reset", e)

    def clear(self):
        self.reset()

    def save(self):
        if not self.filepath:
            return
        try:
            data = dict(self.extra_state)
            if os.path.exists(self.filepath):
                with open(self.filepath, "r", encoding="utf-8") as f:
                    current = json.load(f)
                if not isinstance(current, dict):
                    raise ValueError("Shared zone state harus berupa object JSON.")
                data.update({
                    key: value for key, value in current.items()
                    if key not in ("tracked_zones", "alerts", self.INVALID_STATE_KEY)
                })
            data.update({
                "tracked_zones": self.tracked_zones,
                self.INVALID_STATE_KEY: self.invalidated_zones,
                "alerts": self.alerts[-50:],
            })
            self.extra_state = {
                key: value for key, value in data.items()
                if key not in ("tracked_zones", "alerts", self.INVALID_STATE_KEY)
            }
            tmp = self.filepath + ".zone.tmp"
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
            raw_inv = data.get(self.INVALID_STATE_KEY, {})
            self.invalidated_zones = raw_inv if isinstance(raw_inv, dict) else {}
            self.alerts = data.get("alerts", [])
            self.extra_state = {
                key: value for key, value in data.items()
                if key not in ("tracked_zones", "alerts", self.INVALID_STATE_KEY)
            }
            # Audit & bersihkan catatan invalidasi palsu (pra-pembentukan, gagal formasi OB/FVG, atau break_close tidak menembus distal)
            cleaned_any = False
            for inv_k in list(self.invalidated_zones.keys()):
                inv_rec = self.invalidated_zones.get(inv_k)
                tr_match = self.tracked_zones.get(inv_k) or (
                    self.tracked_zones.get(inv_rec.get("zone_id")) if isinstance(inv_rec, dict) else None
                )
                inv_rsn_l = str((inv_rec or {}).get("reason") or "").lower() if isinstance(inv_rec, dict) else ""
                is_form_rej = any(
                    kw in inv_rsn_l
                    for kw in (
                        "tidak ada fvg",
                        "displacement tidak memadai",
                        "bos tipis",
                        "bos terlalu jauh",
                        "tidak ada candle impuls",
                        "zona terlalu lebar",
                        "zona terlalu sempit",
                        "zona sudah dimitigasi",
                        "gagal hard validation",
                    )
                )
                if isinstance(inv_rec, dict) and (is_form_rej or not _is_breach_invalidation_valid(tr_match or inv_rec, inv_rec)):
                    self.invalidated_zones.pop(inv_k, None)
                    cleaned_any = True
                    if isinstance(tr_match, dict) and tr_match.get("status") == "INVALIDATED":
                        tr_match["status"] = "RADAR"
                        tr_match["lifecycle_status"] = "RADAR"
                        tr_match["is_broken"] = False
                        tr_match["invalid_label"] = ""
                        tr_match["rejection_reason"] = ""
            for tr_z in self.tracked_zones.values():
                if not isinstance(tr_z, dict):
                    continue
                tr_st = str(tr_z.get("status") or "").upper()
                tr_rej = str(tr_z.get("rejection_reason") or "")
                if tr_st != "INVALIDATED" and "INVALID — BROKEN" in tr_rej:
                    tr_z["rejection_reason"] = ""
                    cleaned_any = True
            if cleaned_any and self.auto_save:
                self.save()
        except Exception as e:
            _catat_error("zone_tracker_load", e)


_ZONE_TRACKER: Optional[ZoneTracker] = None


def get_zone_tracker(auto_save: bool = False) -> ZoneTracker:
    global _ZONE_TRACKER
    if _ZONE_TRACKER is None:
        _ZONE_TRACKER = ZoneTracker(filepath=FILE_STATE_ZONE_TRACKER, auto_save=auto_save)
    elif auto_save:
        _ZONE_TRACKER.auto_save = True
    return _ZONE_TRACKER


# =============================================================================
# PIPELINE KEPUTUSAN: DATA -> H4 -> H1 -> M30 -> M5 -> RISK
# =============================================================================
def evaluate_trade_decision(frames: Optional[Dict[str, pd.DataFrame]] = None, current_price: Optional[float] = None,
                            spread: Optional[float] = None, news_blackout: Any = None,
                            server_time: Any = None, live: Optional[bool] = None,
                            snapshot: Optional[Dict[str, Any]] = None,
                            scan_result: Optional[Dict[str, Any]] = None,
                            market_status: Optional[str] = None) -> Dict[str, Any]:
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
    eff_mkt_status = str(
        market_status
        or (snapshot.get("market_status") if isinstance(snapshot, dict) else "")
        or (scan_result.get("market_status") if isinstance(scan_result, dict) else "")
        or "UNKNOWN"
    ).upper()
    dec: Dict[str, Any] = {
        "status": "WAIT", "direction": None, "reasons": [], "zone": None, "plan": None,
        "stages": {}, "timestamp": format_wib_timestamp(default_now=True), "symbol": SYMBOL,
        "snapshot_id": snap_id,
        "data_freshness": "NO DATA", "data_freshness_info": {},
        "market_status": eff_mkt_status, "connection_status": "UNKNOWN",
        "last_candle_times": {}, "forming_candles": {},
    }
    try:
        dec = _decide(dec, frames, current_price, spread, news_blackout, server_time, live,
                      snapshot=snapshot, scan_result=scan_result, market_status=eff_mkt_status)
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


def build_premarket_analysis(frames, snap, max_plans: int = 3, scan_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """PRE-MARKET ANALYSIS (market CLOSED): rencana PROVISIONAL dari candle MT5 terakhir yang valid.
    Memakai scanner + plan_pending_order yang sama (filter H4/H1/M30, ATR, RR, SL struktural); bukan sinyal live."""
    out: Dict[str, Any] = {"status": "PROVISIONAL", "plans": [], "candidates": [], "rejected_candidates": [], "far_references": [], "reason": "", "price": None,
                           "candle_times": dict((snap or {}).get("last_candle_times") or {})}
    try:
        price = (snap or {}).get("snapshot_price")
        if (price is None or not math.isfinite(float(price)) or float(price) <= 0) and frames:
            ref_c = frames.get("M5") if frames.get("M5") is not None and len(frames["M5"]) else frames.get("H1")
            if ref_c is not None and len(ref_c):
                price = float(ref_c["Close"].iloc[-1])
        if not frames or price is None or not math.isfinite(float(price)) or float(price) <= 0:
            out["eligibility_status"] = "DATA_UNAVAILABLE"
            out["reason"] = "DATA_UNAVAILABLE: Candle/harga MT5 valid tidak tersedia"
            return out
        price = float(price)
        out["price"] = price
        out["last_tick_time"] = (snap or {}).get("tick_time")
        out["price_time"] = out["candle_times"].get("M5") or out["candle_times"].get("H1")
        if not out["price_time"] and frames:
            for tf_k in ("M5", "M15", "M30", "H1", "H4"):
                df_k = frames.get(tf_k)
                if df_k is not None and len(df_k):
                    out["price_time"] = format_wib_timestamp(_get_df_last_timestamp(df_k))
                    break
        if not out["price_time"]:
            out["eligibility_status"] = "DATA_UNAVAILABLE"
            out["reason"] = "DATA_UNAVAILABLE: Timestamp candle terakhir tidak tersedia; referensi harga tidak valid"
            return out
        scan = scan_result or scan_xauusd_zones(frames=frames, current_price=price, snapshot=snap, market_status="CLOSED")
        ctx = scan.get("htf_context") or evaluate_htf_context(frames)
        h4, m30 = ctx.get("H4"), ctx.get("m30_status")
        out["htf"] = {"H4": h4, "H1": ctx.get("H1"), "M30": m30}
        
        # ZONE DISCOVERY: kumpulkan seluruh kandidat zona dari MT5 scan
        raw_cands = []
        seen_ids = set()
        for pool in (scan.get("radar_zones") or [], scan.get("active_zones") or [], (scan.get("radar_priority") or {}).get("all_prioritized") or [], scan.get("detected_zones") or []):
            for z in pool:
                zd = z.to_dict() if hasattr(z, "to_dict") else dict(z)
                zid = zd.get("zone_id")
                if zid and zid not in seen_ids:
                    seen_ids.add(zid)
                    raw_cands.append(zd)
        
        # ZONE VALIDATION: periksa keutuhan struktur (bukan broken / invalidated / structurally rejected)
        valid_struct_cands = []
        for zd in raw_cands:
            is_inval = (
                zd.get("is_valid") is False
                or str(zd.get("grade") or "").upper() == "REJECTED"
                or str(zd.get("freshness") or "").upper() in ("INVALIDATED", "MITIGATED", "EXPIRED")
                or zd.get("is_broken")
                or str(zd.get("status") or "").upper() in ("INVALIDATED", "BROKEN", "REJECTED", "MITIGATED", "EXPIRED")
            )
            if not is_inval:
                valid_struct_cands.append(zd)
                
        # Deduplikasi lintas timeframe
        cands_to_rank = valid_struct_cands
        dedup_cands, _ = _deduplicate_overlapping_zones(cands_to_rank)
        out["total_zones_found"] = len(dedup_cands)

        ref_atr = float(
            (scan.get("atr_diagnostics") or {}).get("atr_value")
            or (frames.get("H1")["ATR14"].iloc[-1] if frames.get("H1") is not None and "ATR14" in frames["H1"].columns else 2.0)
        )
        far_threshold = max(35.0, 15.0 * ref_atr)

        # Update distance, relevance & far_category (FAR_INTRADAY vs EXTREME_FAR) pada seluruh kandidat
        for zd in dedup_cands:
            z_lo = float(zd.get("price_low", 0.0))
            z_hi = float(zd.get("price_high", 0.0))
            if price > 0 and z_lo > 0:
                dist_val = max(z_lo - price, price - z_hi, 0.0)
                zd["distance"] = dist_val
                dist_ratio, _ = calculate_distance_atr_ratio(dist_val, ref_atr)
                zd["distance_atr_ratio"] = dist_ratio
                rel = classify_zone_relevance(zd, price, atr=ref_atr, htf_bias=h4, h1_structure=ctx.get("H1"), m30_status=m30, frames=frames)
                zd["zone_relevance"] = rel["relevance"]
                zd["zone_relevance_reason"] = rel["reason"]
                zd["origin_tf"] = rel.get("origin_tf") or zd.get("timeframe", "H1")
                zd["radar_limit_pt"] = rel.get("radar_limit_pt")
                if isinstance(dist_ratio, (int, float)) and dist_ratio > 15.0:
                    zd["far_category"] = "EXTREME_FAR"
                elif str(zd.get("zone_relevance") or "").upper() == "FAR_REFERENCE":
                    zd["far_category"] = "FAR_INTRADAY"
                else:
                    zd["far_category"] = "IN_RANGE"
        
        # Ranking berdasarkan kualitas terpadu (Relevance Tier, HTF Alignment, Validitas, Freshness, Grade, Jarak, Score, Timeframe)
        def _premarket_sort_key(z_dict):
            rel = str(z_dict.get("zone_relevance") or "").upper()
            f_cat = str(z_dict.get("far_category") or "IN_RANGE").upper()
            dist = float(z_dict.get("distance") or 999.0)
            if rel == "ACTIVE":
                rel_tier = 0
            elif rel == "RADAR" and f_cat == "IN_RANGE":
                rel_tier = 1
            elif f_cat == "FAR_INTRADAY":
                rel_tier = 2
            else:
                rel_tier = 3
            val_r = 0 if z_dict.get("is_valid") else 1
            is_alg = (h4 == "BULLISH" and z_dict.get("zone_type") == "DEMAND") or (h4 == "BEARISH" and z_dict.get("zone_type") == "SUPPLY")
            alg_r = 0 if is_alg else 1
            fresh_r = {"FRESH": 0, "TESTED_1X": 1, "TESTED_2X": 2}.get(z_dict.get("freshness"), 9)
            grd = str(z_dict.get("grade") or "B")
            grd_rank = {"A+": 0, "A": 1, "B": 2}.get(grd, 3)
            sc = float(z_dict.get("score") or 0.0)
            tf = str(z_dict.get("timeframe") or "H1")
            tf_r = 0 if "H4" in tf else (1 if "H1" in tf else (2 if "M30" in tf else 3))
            return (rel_tier, alg_r, val_r, fresh_r, grd_rank, dist, -sc, tf_r)
            
        dedup_cands.sort(key=_premarket_sort_key)
        
        # Pisahkan IN_RANGE vs FAR_INTRADAY (<= 15x ATR tetapi > batas TF/poin) vs EXTREME_FAR (> 15x ATR)
        actionable_cands = [z for z in dedup_cands if z.get("far_category") == "IN_RANGE"]
        far_intraday_cands = [z for z in dedup_cands if z.get("far_category") == "FAR_INTRADAY"]
        extreme_far_cands = [z for z in dedup_cands if z.get("far_category") == "EXTREME_FAR"]
        far_cands = sorted(far_intraday_cands + extreme_far_cands, key=lambda z_f: (0 if z_f.get("far_category") == "FAR_INTRADAY" else 1, float(z_f.get("distance", 9999.0))))
        out["far_references"] = far_cands
        out["far_intraday"] = far_intraday_cands
        out["extreme_far"] = extreme_far_cands
        out["far_intraday_count"] = len(far_intraday_cands)
        out["extreme_far_count"] = len(extreme_far_cands)

        # Evaluasi kandidat actionable (IN_RANGE) + kandidat FAR_INTRADAY terdekat (agar alasan penolakan selalu tercatat konsisten)
        eval_pool = actionable_cands + far_intraday_cands
        for zd in eval_pool:
            z_obj = _zone_from_dict(zd)
            plan = plan_pending_order(z_obj, price, frames, h4, m30, (snap or {}).get("server_time"),
                                      h1_structure=ctx.get("H1"), premarket=True,
                                      opposing_zones=cands_to_rank, snr_context=scan.get("snr_context"))
            d = "BUY" if zd.get("zone_type") == "DEMAND" else "SELL"
            z_lo = float(zd.get("price_low", 0.0))
            z_hi = float(zd.get("price_high", 0.0))
            dist_val = float(zd.get("distance") or max(z_lo - price, price - z_hi, 0.0))
            dist_ratio = zd.get("distance_atr_ratio")
            grd = zd.get("grade") or "B"
            sc = float(zd.get("score") or 0.0)
            f_cat = str(zd.get("far_category") or "IN_RANGE")
            
            confl_parts = []
            if (zd.get("origin_bos") or {}).get("tipe"):
                confl_parts.append(str(zd["origin_bos"]["tipe"]))
            if (zd.get("fvg_confirmation") or {}).get("status") in ("FVG_CONFIRMED", "ALIGNED"):
                confl_parts.append("FVG Aligned")
            if zd.get("liquidity_sweep"):
                confl_parts.append("Liquidity Swept")
            confl_str = " + ".join(confl_parts) or zd.get("zone_relevance_reason") or "Struktur MT5 valid"
            
            ep = float(plan.get("entry_price") or (z_hi if d == "BUY" else z_lo))
            sl = float(plan.get("sl") or compute_structural_sl(z_obj, frames))
            tp_list = _structural_targets(frames, zd.get("zone_type", ""), ep,
                                         opposing_zones=cands_to_rank,
                                         snr_context=scan.get("snr_context"),
                                         atr=ref_atr) if frames else []
            tp1 = float(plan.get("tp1") or (tp_list[0] if tp_list else 0.0))
            risk_px = round(abs(ep - sl), 2) if (ep > 0 and sl > 0) else 0.0
            reward_px = round(abs(tp1 - ep), 2) if (ep > 0 and tp1 > 0) else 0.0
            rr = round(reward_px / risk_px, 2) if risk_px > 0 else float(plan.get("rr", 0.0) or 0.0)
            
            rej_r = str(zd.get("pending_rejection_reason") or plan.get("rejection_reason") or zd.get("rejection_reason") or "")
            is_eligible_setup = bool(
                f_cat == "IN_RANGE"
                and zd.get("is_valid")
                and plan.get("valid")
                and rr >= RR_MIN
                and 0.5 < risk_px <= SL_JARAK_MAKS
            )
            # Susun alasan penolakan utama yang konsisten dengan ZONA UTAMA & RADAR (pertahankan seluruh alasan internal)
            val_reasons = list(zd.get("validation_reasons") or [])
            if is_eligible_setup:
                status_c = "PROVISIONAL (ELIGIBLE — WAIT MARKET OPEN)"
                status_reason_c = "Memenuhi validasi HTF, SL struktural <= 20 USD, TP likuiditas, RR >= 1:1.5 (tetap WAIT saat market CLOSED)"
            else:
                reason_parts = []
                if f_cat == "FAR_INTRADAY":
                    orig_tf_c = zd.get("origin_tf") or zd.get("timeframe", "H1")
                    lim_pt_c = float(zd.get("radar_limit_pt") or 30.0)
                    rat_s_c = f"{dist_ratio:.1f}x ATR" if isinstance(dist_ratio, (int, float)) else "N/A ATR"
                    reason_parts.append(f"FAR_INTRADAY ({orig_tf_c}: jarak {dist_val:.2f} USD / {rat_s_c} > batas intraday {lim_pt_c:.0f} USD)")
                if risk_px > SL_JARAK_MAKS or "Jarak SL" in rej_r or "melebihi batas" in rej_r:
                    reason_parts.append(f"Jarak SL ({risk_px:.2f} USD) > maks {SL_JARAK_MAKS:g} USD")
                if "HTF_CONFLICT" in rej_r or "Konflik" in rej_r or "tidak selaras" in rej_r:
                    reason_parts.append(f"HTF_CONFLICT (Bias H4 {h4} berlawanan)")
                elif "RR_TOO_LOW" in rej_r or (rr > 0 and rr < RR_MIN):
                    reason_parts.append(f"RR_TOO_LOW (1:{rr:.2f} < 1:{RR_MIN:g})")
                elif "NO_TARGET" in rej_r or not tp_list:
                    reason_parts.append("NO_TARGET (Tidak ada target likuiditas searah)")
                elif not zd.get("is_valid") or sc < 50.0:
                    reason_parts.append(f"LOW_GRADE (Skor {sc:.1f} < minimum)")
                elif not reason_parts and rej_r:
                    reason_parts.append(rej_r)
                status_c = "REJECTED_RISK" if risk_px > SL_JARAK_MAKS else ("FAR_INTRADAY" if f_cat == "FAR_INTRADAY" else "REJECTED")
                status_reason_c = "; ".join(reason_parts) or "Tidak memenuhi filter kelayakan setup pending"
                for rp_item in reason_parts:
                    if rp_item not in val_reasons:
                        val_reasons.append(rp_item)

            if is_eligible_setup and len(out["candidates"]) < max_plans:
                cand_item = {
                    "direction": d,
                    "order_type": f"{d} LIMIT",
                    "zone_timeframe": zd.get("timeframe", "H1"),
                    "timeframe": zd.get("timeframe", "H1"),
                    "zone_type": zd.get("zone_type", "DEMAND"),
                    "zone_subtype": (zd.get("origin_bos") or {}).get("tipe", "OB"),
                    "zone_low": z_lo,
                    "zone_high": z_hi,
                    "distance": dist_val,
                    "distance_atr_ratio": dist_ratio,
                    "zone_grade": grd,
                    "grade": grd,
                    "zone_score": sc,
                    "score": sc,
                    "zone_quality": f"VALID (Grade {grd} | {sc:.1f})",
                    "setup_eligibility": "ELIGIBLE (PROVISIONAL)",
                    "entry_permission": "WAIT (MARKET CLOSED)",
                    "entry_price": ep,
                    "sl": sl,
                    "tp1": tp1,
                    "rr": rr,
                    "rr_tp1": rr,
                    "status": status_c,
                    "status_reason": status_reason_c,
                    "validation_reasons": val_reasons,
                    "validity_reason": confl_str,
                    "confluence": confl_str,
                    "provisional": True,
                }
                out["candidates"].append(cand_item)
                out["plans"].append(cand_item)
            else:
                r_rsn = status_reason_c if not is_eligible_setup else "Zona cadangan eligible (di luar kuota top 3 kandidat utama)"
                out["rejected_candidates"].append({
                    "timeframe": zd.get("timeframe"),
                    "zone_type": zd.get("zone_type"),
                    "price_low": z_lo,
                    "price_high": z_hi,
                    "distance": dist_val,
                    "distance_atr_ratio": dist_ratio,
                    "far_category": f_cat,
                    "grade": grd,
                    "score": sc,
                    "zone_quality": f"VALID (Grade {grd} | {sc:.1f})",
                    "setup_eligibility": "NOT_ELIGIBLE" if not is_eligible_setup else "ELIGIBLE_BACKUP",
                    "entry_permission": "WAIT (MARKET CLOSED)",
                    "reason": r_rsn,
                    "validation_reasons": val_reasons,
                    "rr": rr if is_eligible_setup else 0.0,
                })

        out["rejected_candidates"].sort(key=lambda rc_item: (
            0 if ((h4 == "BULLISH" and rc_item.get("zone_type") == "DEMAND") or (h4 == "BEARISH" and rc_item.get("zone_type") == "SUPPLY")) else 1,
            float(rc_item.get("distance", 9999.0)),
            -float(rc_item.get("score", 0.0)),
        ))

        if not dedup_cands:
            out["eligibility_status"] = "NO_ZONES_FOUND"
            out["reason"] = "NO_ZONES_FOUND: Tidak ada zona struktural valid yang terdeteksi dari candle MT5"
        elif not out["candidates"]:
            out["eligibility_status"] = "NO_ELIGIBLE_SETUPS"
            top_rej = out["rejected_candidates"][0] if out["rejected_candidates"] else None
            top_rej_note = (
                f" — Terdekat {top_rej.get('timeframe')} {top_rej.get('zone_type')} "
                f"{top_rej.get('price_low', 0.0):.2f}-{top_rej.get('price_high', 0.0):.2f} ditolak ({top_rej.get('reason')})"
                if top_rej else ""
            )
            out["reason"] = f"NO_ELIGIBLE_SETUPS: {len(dedup_cands)} zona struktural valid terdeteksi di RADAR, namun 0 lolos filter kelayakan setup{top_rej_note}"
        else:
            out["eligibility_status"] = "ELIGIBLE_SETUPS_FOUND"
            out["reason"] = f"ELIGIBLE_SETUPS_FOUND: {len(out['candidates'])} kandidat setup layak (tetap WAIT saat market CLOSED)"
    except Exception as exc:
        _catat_error("premarket_analysis", exc)
        out["eligibility_status"] = "DATA_UNAVAILABLE"
        out["reason"] = "DATA_UNAVAILABLE: Analisis pre-market gagal (data tidak cukup)"
    return out


def _provisional_zone_plans(dec, price, frames, h4_bias, m30_status, server_time, bid=None, ask=None):
    """Run the existing pending-order planner once per displayed zone so every command shows the same levels."""
    plans: Dict[str, Dict[str, Any]] = {}
    r_prio = dec.get("radar_priority") or {}
    is_mkt_closed_p = (
        str(dec.get("market_status") or "").upper() == "CLOSED"
        or str(dec.get("snapshot_status") or "").upper() == "MARKET_CLOSED"
    )
    cands = (
        dec.get("zone"),
        dec.get("primary_zone"),
        dec.get("nearest_valid_zone"),
        dec.get("next_radar_zone"),
        r_prio.get("primary_directional_zone"),
        r_prio.get("nearest_radar_zone"),
        r_prio.get("best_high_risk"),
    )
    for zd in cands:
        zid = zd.get("zone_id") if isinstance(zd, dict) else None
        if not zid or zid in plans:
            continue
        try:
            z_obj = _zone_from_dict(zd)
            plan = plan_pending_order(
                z_obj, float(price), frames, h4_bias, m30_status, server_time,
                bid=bid, ask=ask, premarket=is_mkt_closed_p,
            )
            if str(zd.get("setup_eligibility") or "").upper().startswith("NOT_ELIGIBLE") or zd.get("risk_rejected") or str(zd.get("zone_relevance") or "").upper() == "FAR_REFERENCE":
                plan["valid"] = False
                plan["provisional"] = False
                plan["status"] = "REJECTED"
                plan["early_pending_status"] = "REJECTED"
                if zd.get("pending_rejection_reason"):
                    plan["rejection_reason"] = zd["pending_rejection_reason"]
            z_type = str(zd.get("zone_type") or "").upper()
            aligned_bias = (h4_bias == "BULLISH" and z_type == "DEMAND") or (h4_bias == "BEARISH" and z_type == "SUPPLY")
            if not plan.get("valid") and aligned_bias and plan.get("status") in ("CONFLICT", "WEAK"):
                calc = plan_pending_order(
                    z_obj, float(price), frames, None, None, server_time, bid=bid, ask=ask, h1_structure=None,
                )
                if calc.get("valid"):
                    calc["valid"] = False
                    calc["provisional"] = True
                    calc["status"] = "PROVISIONAL"
                    calc["early_pending_status"] = "PROVISIONAL"
                    calc["early_pending_status_label"] = f"PROVISIONAL ONLY — {plan.get('rejection_reason') or 'VALIDASI FINAL WAIT'}"
                    calc["rejection_reason"] = plan.get("rejection_reason") or "Gagal validasi HTF/M30"
                    plan = calc
            plans[zid] = plan
        except Exception as exc:
            _catat_error("provisional_zone_plan", exc)
            plans[zid] = {"valid": False, "status": "WAIT", "rejection_reason": "PLAN ERROR"}
    return plans


def generate_smc_trade_plan(
    zone_dict: Dict[str, Any],
    price: float,
    frames: Optional[Dict[str, pd.DataFrame]] = None,
    *,
    dec: Optional[Dict[str, Any]] = None,
    scan: Optional[Dict[str, Any]] = None,
    bid: Optional[float] = None,
    ask: Optional[float] = None,
    spread: float = 0.0,
    opposing_zones: Optional[List[Any]] = None,
    snr_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Trade Plan Generator deterministik untuk satu zona SMC ACTIVE/RADAR.
    Mendelegasikan ke generator bersama `build_shared_trade_plan` agar PLANNED_ENTRY, SetupSL,
    TP1, TP2, dan RR stabil per Zone ID + setup_version serta terpisah dari LIVE_ENTRY/Market RR.
    """
    from modules.utils import build_shared_trade_plan

    zd = zone_dict if isinstance(zone_dict, dict) else (zone_dict.to_dict() if hasattr(zone_dict, "to_dict") else {})
    zid = str(zd.get("zone_id") or "").strip()
    tf = str(zd.get("timeframe") or "H1").strip().upper()
    z_type = str(zd.get("zone_type") or zd.get("type") or "").strip().upper()
    direction = "BUY" if z_type == "DEMAND" else ("SELL" if z_type == "SUPPLY" else "NONE")
    grd = str(zd.get("original_grade") or zd.get("grade") or "N/A").strip()
    sc = float(zd.get("score") or 0.0) if zd.get("score") is not None else 0.0
    lo = float(zd.get("price_low", zd.get("low", 0.0)) or 0.0)
    hi = float(zd.get("price_high", zd.get("high", 0.0)) or 0.0)
    if lo > hi > 0:
        lo, hi = hi, lo

    d_obj = dec if isinstance(dec, dict) else {}
    s_obj = scan if isinstance(scan, dict) else {}
    ctx = d_obj.get("context") or s_obj.get("htf_context") or {}
    h4_bias = str(ctx.get("H4") or ctx.get("htf_bias") or "NETRAL").upper()
    h1_bias = str(ctx.get("H1") or ctx.get("h1_structure") or "NETRAL").upper()
    stages = d_obj.get("stages") or {}
    m30_st = str((stages.get("M30") or {}).get("status") or ctx.get("m30_status") or "").upper()
    mkt_st = str(d_obj.get("market_status") or s_obj.get("market_status") or "OPEN").upper()
    is_closed = mkt_st in ("CLOSED", "MARKET_CLOSED") or str(d_obj.get("snapshot_status") or "").upper() == "MARKET_CLOSED"

    try:
        why_txt = build_smc_zone_reason(zd, h4_bias=h4_bias)
    except Exception:
        why_txt = ""

    struct_inv = resolve_smc_structure_invalidation_price(zd)
    if struct_inv <= 0 and lo > 0:
        struct_inv = lo if direction == "BUY" else (hi if direction == "SELL" else 0.0)

    z_life = str(zd.get("lifecycle_status") or zd.get("status") or "").upper()
    z_fr = str(zd.get("freshness") or "FRESH").upper()
    is_expired_z = bool(zd.get("is_expired") or z_life == "EXPIRED" or z_fr == "EXPIRED")
    is_invalid_z = bool(
        (not is_expired_z) and (
            zd.get("is_valid") is False
            or zd.get("is_broken") is True
            or z_life in ("INVALIDATED", "BROKEN", "MITIGATED", "REJECTED")
            or z_fr in ("INVALIDATED", "BROKEN", "MITIGATED")
            or str(zd.get("grade") or "").upper() == "REJECTED"
        )
    )
    is_far_z = bool(
        str(zd.get("zone_relevance") or "").upper() == "FAR_REFERENCE"
        or (price > 0 and is_far_reference_zone(zd, current_price=price))
    )

    u_spec = extract_unit_spec_from_container(s_obj if s_obj else d_obj)
    digits = int(u_spec.get("digits", 2))
    point = float(u_spec.get("point") or 0.01)
    stops_lvl = u_spec.get("trade_stops_level")
    min_stop = (float(stops_lvl) * point) if (isinstance(stops_lvl, (int, float)) and stops_lvl > 0) else 0.0
    eff_spread = float(spread or 0.0)
    if eff_spread <= 0 and bid and ask and ask >= bid > 0:
        eff_spread = round(float(ask) - float(bid), max(digits, 2))

    htf_conflict = bool(
        ctx.get("has_conflict")
        or (h4_bias in ("BULLISH", "BEARISH") and h1_bias in ("BULLISH", "BEARISH") and h4_bias != h1_bias)
    )
    counter_htf = bool(
        (h4_bias == "BULLISH" and direction == "SELL")
        or (h4_bias == "BEARISH" and direction == "BUY")
    )
    m30_bad = m30_st in ("CONFLICT", "INVALID", "WEAK")
    demand = (direction == "BUY")
    inside = bool(lo > 0 and hi >= lo and lo <= price <= hi)

    if is_closed:
        cond_rsn = "MARKET_CLOSED"
    elif htf_conflict:
        cond_rsn = "HTF_CONFLICT"
    elif counter_htf:
        cond_rsn = "COUNTER_HTF"
    elif m30_bad:
        cond_rsn = f"M30_{m30_st}"
    elif inside:
        cond_rsn = "WAIT_M5_CONFIRM"
    else:
        cond_rsn = "WAIT_CONFIRM"

    # 1. Existing CONFIRMED entry plan (jika valid untuk zona ini)
    confirmed_payload = None
    conf_plan = d_obj.get("plan") if isinstance(d_obj.get("plan"), dict) else None
    conf_zid = str((d_obj.get("zone") or {}).get("zone_id") or "")
    if (
        not is_closed
        and str(d_obj.get("status") or "").upper() == "VALID"
        and conf_plan
        and str(d_obj.get("direction") or "").upper() == direction
        and zid and conf_zid == zid
    ):
        ce = float(conf_plan.get("entry_acuan", conf_plan.get("entry", 0.0)) or 0.0)
        csl = float(conf_plan.get("sl") or 0.0)
        ctp1 = float(conf_plan.get("tp1") or 0.0)
        ctp2 = float(conf_plan.get("tp2") or 0.0)
        if (csl < ce < ctp1) if demand else (csl > ce > ctp1 > 0):
            confirmed_payload = {
                "entry": ce,
                "sl": csl,
                "tp1": ctp1,
                "tp2": ctp2,
                "struct_inv": struct_inv,
            }

    # 2. Existing Early Pending plan (jika ada untuk zona ini)
    early_payload = None
    early = d_obj.get("early_setup") if isinstance(d_obj.get("early_setup"), dict) else {}
    early_pool = ([early["best_setup"]] if isinstance(early.get("best_setup"), dict) else []) + list(early.get("all_setups") or [])
    matched_early = next(
        (ep for ep in early_pool if isinstance(ep, dict) and str(ep.get("zone_id") or "") == zid),
        None,
    )
    if matched_early:
        ep_ready = bool(
            matched_early.get("valid")
            and str(matched_early.get("status") or "").upper() == "READY"
            and not matched_early.get("is_expired")
            and not is_closed and not htf_conflict and not counter_htf and not m30_bad
        )
        early_payload = {
            "entry": float(matched_early.get("entry_price") or 0.0),
            "sl": float(matched_early.get("sl") or 0.0),
            "tp1": float(matched_early.get("tp1") or 0.0),
            "tp2": float(matched_early.get("tp2") or 0.0),
            "ready": ep_ready,
            "is_expired": bool(matched_early.get("is_expired")),
        }

    ref_atr_z = zd.get("atr_value")
    if not ref_atr_z and frames and "H1" in frames and "ATR14" in frames["H1"].columns:
        try:
            ref_atr_z = float(frames["H1"]["ATR14"].iloc[-1])
        except Exception:
            ref_atr_z = 0.0
    atr_num = float(ref_atr_z) if (isinstance(ref_atr_z, (int, float)) and math.isfinite(ref_atr_z) and ref_atr_z > 0) else 0.0

    def _compute_smc_levels(planned_e: float) -> Dict[str, Any]:
        try:
            z_obj = _zone_from_dict(zd)
            setup_sl_v = compute_structural_sl(z_obj, frames, eff_spread)
        except Exception:
            setup_sl_v = 0.0
        if setup_sl_v <= 0.0:
            return {"ok": False, "reason": "SL_UNAVAILABLE"}
        targets = _structural_targets(
            frames or {},
            z_type,
            planned_e,
            opposing_zones=opposing_zones,
            snr_context=snr_context,
            atr=atr_num if atr_num > 0 else None,
        ) if frames else []
        if not targets:
            return {"ok": False, "reason": "NO_STRUCTURAL_TARGET"}
        tp_buf = 0.3
        if demand:
            tp1_v = round(targets[0] - tp_buf, digits)
            tp2_v = round(targets[1] - tp_buf, digits) if len(targets) > 1 else 0.0
        else:
            tp1_v = round(targets[0] + tp_buf, digits)
            tp2_v = round(targets[1] + tp_buf, digits) if len(targets) > 1 else 0.0
        return {
            "ok": True,
            "setup_sl": round(setup_sl_v, digits),
            "struct_inv": round(struct_inv, digits),
            "tp1": tp1_v,
            "tp2": tp2_v,
        }

    is_bo_pending_z = bool(
        (not is_invalid_z) and (not is_expired_z) and (
            z_life in ("BREAKOUT_PENDING", "BREAKDOWN_PENDING", "SUSPENDED")
            or (lo > 0 and hi >= lo and price > 0 and ((direction == "SELL" and price > hi) or (direction == "BUY" and price < lo)))
        )
    )

    # Bukti touch aktual & jendela validitas existing SMC (bars_since_trig > 2 & chase_tol = max(1.0, 0.5*atr5))
    was_touched_smc = bool(int(zd.get("touch_count") or 0) > 0 or z_life in ("TOUCHED", "ZONE_TOUCHED"))
    touch_time_smc = None
    bars_since_touch_smc = None
    atr5_smc = 1.5
    m5_df_tp = frames.get("M5") if isinstance(frames, dict) else None
    if m5_df_tp is not None and not m5_df_tp.empty and lo > 0 and hi >= lo:
        try:
            if "ATR14" in m5_df_tp.columns:
                _a5 = float(m5_df_tp["ATR14"].iloc[-1])
                if math.isfinite(_a5) and _a5 > 0:
                    atr5_smc = _a5
            origin_dt = _resolve_smc_zone_origin_time(zd, frames)
            post_m5 = m5_df_tp
            if origin_dt is not None:
                try:
                    t_col = m5_df_tp["Time"] if "Time" in m5_df_tp.columns else m5_df_tp.index
                    t_ser = pd.to_datetime(t_col, errors="coerce")
                    orig_ts = pd.Timestamp(origin_dt)
                    if getattr(t_ser.dt if hasattr(t_ser, "dt") else t_ser, "tz", None) is None and orig_ts.tzinfo is not None:
                        orig_ts = orig_ts.tz_localize(None)
                    mask_post = t_ser > orig_ts
                    if mask_post.any():
                        post_m5 = m5_df_tp.loc[mask_post]
                except Exception:
                    post_m5 = m5_df_tp.tail(24)
            else:
                post_m5 = m5_df_tp.tail(24)
            if not post_m5.empty:
                ov_m = (post_m5["High"] >= lo) & (post_m5["Low"] <= hi)
                if ov_m.any():
                    was_touched_smc = True
                    hit_idx_list = [idx_i for idx_i, flag in enumerate(ov_m.tolist()) if flag]
                    last_rel = hit_idx_list[-1]
                    last_label = post_m5.index[last_rel]
                    try:
                        last_abs = int(m5_df_tp.index.get_loc(last_label))
                        bars_since_touch_smc = max(0, (len(m5_df_tp) - 1) - last_abs)
                    except Exception:
                        bars_since_touch_smc = max(0, (len(post_m5) - 1) - last_rel)
                    raw_t = post_m5["Time"].iloc[last_rel] if "Time" in post_m5.columns else last_label
                    touch_time_smc = format_wib_timestamp(raw_t, source_tz="MT5")
        except Exception:
            pass

    chase_tol_smc = round(max(1.0, 0.5 * atr5_smc), digits)
    is_missed_smc = bool((d_obj.get("is_missed") or (d_obj.get("m5") or {}).get("is_missed")) and conf_zid == zid)
    m5_eval_st_smc = str(d_obj.get("m5_eval_status") or (stages.get("M5") or {}).get("status") or "").strip().upper() or None
    mt5_snap_smc = s_obj.get("mt5_orders_snapshot") or d_obj.get("mt5_orders_snapshot")

    return build_shared_trade_plan(
        source="SMC",
        zone_id=zid,
        timeframe=tf,
        zone_type=z_type,
        direction=direction,
        grade=grd,
        score=sc,
        lo=lo,
        hi=hi,
        struct_inv=struct_inv,
        live_price=price,
        why=why_txt,
        bid=bid,
        ask=ask,
        spread=eff_spread,
        max_spread=max(float(SPREAD_MAKS_POIN), 0.0),
        digits=digits,
        min_stop=min_stop,
        min_rr=float(RR_MIN),
        max_sl_dist=float(SL_JARAK_MAKS),
        atr_value=atr_num,
        is_closed=is_closed,
        is_invalid_zone=is_invalid_z,
        invalid_reason="ZONE_INVALID",
        is_expired_zone=is_expired_z,
        is_breakout_pending=is_bo_pending_z,
        is_far_zone=is_far_z,
        confirmed_plan=confirmed_payload,
        early_pending_plan=early_payload,
        cond_reason=cond_rsn,
        compute_levels_fn=_compute_smc_levels,
        was_touched=was_touched_smc,
        touch_time=touch_time_smc,
        m5_eval_status=m5_eval_st_smc,
        is_missed=is_missed_smc,
        missed_move_threshold=chase_tol_smc,
        bars_since_touch=bars_since_touch_smc,
        max_bars_after_touch=2,
        mt5_orders_snapshot=mt5_snap_smc,
    )


def generate_smc_trade_plans(
    dec: Optional[Dict[str, Any]],
    scan: Optional[Dict[str, Any]],
    frames: Optional[Dict[str, pd.DataFrame]],
    price: float,
    bid: Optional[float] = None,
    ask: Optional[float] = None,
    spread: float = 0.0,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Evaluasi semua zona SMC ACTIVE/RADAR unik untuk menghasilkan trade_plans & best_trade_plan."""
    from modules.utils import is_tracked_trade_plan_zone, resolve_trade_plan_focus

    d_obj = dec if isinstance(dec, dict) else {}
    s_obj = scan if isinstance(scan, dict) else {}
    r_prio = d_obj.get("radar_priority") or s_obj.get("radar_priority") or {}
    snr_ctx = d_obj.get("snr_context") or s_obj.get("snr_context")
    all_raw = []
    for item in (
        d_obj.get("zone"),
        d_obj.get("primary_zone"),
        d_obj.get("nearest_valid_zone"),
        d_obj.get("next_radar_zone"),
        r_prio.get("primary_directional_zone"),
        r_prio.get("nearest_radar_zone"),
    ):
        if isinstance(item, dict):
            all_raw.append(item)
    for z_list in (s_obj.get("active_zones") or [], s_obj.get("radar_zones") or []):
        for item in z_list:
            zd = item.to_dict() if hasattr(item, "to_dict") else (item if isinstance(item, dict) else None)
            if isinstance(zd, dict):
                all_raw.append(zd)
    for z_list in (s_obj.get("invalidated_zones") or [], s_obj.get("broken_zones") or [], s_obj.get("expired_zones") or []):
        for item in z_list:
            zd = item.to_dict() if hasattr(item, "to_dict") else (item if isinstance(item, dict) else None)
            if isinstance(zd, dict) and is_tracked_trade_plan_zone("SMC", str(zd.get("zone_id") or "")):
                all_raw.append(zd)

    seen_ids = set()
    unique_zones: List[Dict[str, Any]] = []
    for zd in all_raw:
        zid = str(zd.get("zone_id") or "").strip()
        if not zid or zid in seen_ids:
            continue
        is_tracked = is_tracked_trade_plan_zone("SMC", zid)
        if not is_tracked:
            if str(zd.get("zone_relevance") or "").upper() == "FAR_REFERENCE" or is_far_reference_zone(zd, current_price=price if price > 0 else None):
                continue
            if zd.get("is_valid") is False or zd.get("is_broken") is True:
                continue
            z_st = str(zd.get("lifecycle_status") or zd.get("status") or "").upper()
            if z_st in ("INVALIDATED", "BROKEN", "EXPIRED", "MITIGATED", "REJECTED"):
                continue
        seen_ids.add(zid)
        unique_zones.append(zd)

    plans: List[Dict[str, Any]] = []
    plan_by_zid: Dict[str, Dict[str, Any]] = {}
    for zd in unique_zones:
        opp = [oz for oz in unique_zones if oz.get("zone_type") != zd.get("zone_type")]
        tp = generate_smc_trade_plan(
            zd,
            price=price,
            frames=frames,
            dec=d_obj,
            scan=s_obj,
            bid=bid,
            ask=ask,
            spread=spread,
            opposing_zones=opp,
            snr_context=snr_ctx,
        )
        plans.append(tp)
        tp_zid = str(tp.get("zone_id") or "").strip()
        if tp_zid and float(tp.get("setup_sl") or 0.0) > 0.0:
            plan_by_zid[tp_zid] = tp

    if plan_by_zid:
        for zd_ref in all_raw:
            if not isinstance(zd_ref, dict):
                continue
            rz = str(zd_ref.get("zone_id") or "").strip()
            matched_tp = plan_by_zid.get(rz)
            if isinstance(matched_tp, dict):
                zd_ref["setup_sl"] = float(matched_tp["setup_sl"])
                zd_ref["setup_version"] = int(matched_tp.get("setup_version") or 1)

    mkt_st_p = str(d_obj.get("market_status") or s_obj.get("market_status") or "OPEN").upper()
    is_closed_p = mkt_st_p in ("CLOSED", "MARKET_CLOSED") or str(d_obj.get("snapshot_status") or "").upper() == "MARKET_CLOSED"

    # Sinkronkan Trade Plan SMC + SNR bersama agar `scan`, `scan snr`, dan `pantau` memakai selector & fokus UNIFIED yang sama
    if not isinstance(snr_ctx, dict) and isinstance(frames, dict) and price > 0:
        try:
            snr_ctx = extract_snr_supporting_context(
                frames,
                float(price),
                market_status=("CLOSED" if is_closed_p else mkt_st_p),
                snapshot=(s_obj if isinstance(s_obj, dict) else None),
            )
            if isinstance(d_obj, dict) and isinstance(snr_ctx, dict):
                d_obj["snr_context"] = snr_ctx
        except Exception:
            snr_ctx = None
    if isinstance(snr_ctx, dict):
        snr_tp_pool = list(snr_ctx.get("trade_plans") or [])
        if not snr_tp_pool and isinstance(snr_ctx.get("best_trade_plan"), dict):
            snr_tp_pool = [snr_ctx["best_trade_plan"]]
        seen_tp_keys = {
            f"{str(p.get('source') or 'SMC').upper()}:{str(p.get('zone_id') or '').strip()}"
            for p in plans if isinstance(p, dict) and p.get("zone_id")
        }
        for sp in snr_tp_pool:
            if not isinstance(sp, dict):
                continue
            sk = f"{str(sp.get('source') or 'SNR').upper()}:{str(sp.get('zone_id') or '').strip()}"
            if sk not in seen_tp_keys:
                seen_tp_keys.add(sk)
                plans.append(sp)

    best_tp = resolve_trade_plan_focus(
        plans,
        direction=d_obj.get("analysis_direction"),
        scope="UNIFIED",
        live_price=price,
        bid=bid,
        ask=ask,
        is_closed=is_closed_p,
        mt5_orders_snapshot=(s_obj.get("mt5_orders_snapshot") or d_obj.get("mt5_orders_snapshot")),
    )
    return plans, best_tp


def _decide(dec, frames, price, spread, news, server_time, live,
            snapshot: Optional[Dict[str, Any]] = None,
            scan_result: Optional[Dict[str, Any]] = None,
            market_status: Optional[str] = None):
    st = dec["stages"]
    scan = None
    ctx = {}
    best_candidate_dict = None
    winning_zdict = None
    directional = []

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
            zones_to_track = list(scan_obj.get("active_zones", [])) if scan_obj else []
            decision_zone = dec_obj.get("zone") or {}
            if decision_zone.get("zone_id") and not any(z.get("zone_id") == decision_zone["zone_id"] for z in zones_to_track):
                zones_to_track.append({**decision_zone, "monitor_only": True})
            dec_obj["zone_tracker"] = tracker
            if not isinstance(eff_p, (int, float)) or not math.isfinite(eff_p) or eff_p <= 0 or st.get("DATA", {}).get("ok") is False:
                return
            old_focus = next((rec for rec in tracker.tracked_zones.values() if rec.get("monitor_focus")), None)
            tracker.update(
                zones_to_track, eff_p, frames=frames, htf_bias=None, m30_status=None,
                server_time=server_time, conservative_dec=dec_obj,
                bid=(scan_obj.get("bid") if isinstance(scan_obj, dict) else None),
                ask=(scan_obj.get("ask") if isinstance(scan_obj, dict) else None),
                spread=float(spread or 0.0),
                market_status=dec_obj.get("market_status"),
            )
            focus_id = (dec_obj.get("zone") or {}).get("zone_id")
            focus_evaluated = bool(dec_obj.get("focus_selection_evaluated")) or bool(focus_id)
            if old_focus and focus_evaluated and old_focus.get("zone_id") != focus_id:
                old_status = str(old_focus.get("status", "RADAR")).upper()
                old_id = old_focus.get("zone_id")
                detected = next((z for z in zones_to_track if z.get("zone_id") == old_id), None)
                if old_status in ("EXPIRED", "INVALIDATED"):
                    reason = f"PREVIOUS ZONE {old_status}"
                elif detected is None:
                    new_zone = dec_obj.get("zone") or {}
                    old_sources = set(old_focus.get("merged_from") or [old_id])
                    new_sources = set(new_zone.get("merged_from") or [focus_id])
                    reason = (
                        "ZONE REBUILT; SETUP REVALIDATED"
                        if old_sources.intersection(new_sources)
                        else "PREVIOUS ZONE NOT DETECTED OR FILTERED"
                    )
                elif focus_id:
                    reason = "HIGHER PRIORITY OR VALIDATED SETUP"
                else:
                    reason = "NO ACTIVE SETUP"
                dec_obj["previous_focus"] = {
                    "zone_id": old_id, "price_low": old_focus.get("price_low"),
                    "price_high": old_focus.get("price_high"), "status": old_status,
                    "timeframe": old_focus.get("timeframe"), "zone_type": old_focus.get("zone_type"),
                    "grade": old_focus.get("grade"), "score": old_focus.get("score", 0.0),
                }
                dec_obj["focus_change_reason"] = reason
            if focus_evaluated:
                for rec in tracker.tracked_zones.values():
                    rec["monitor_focus"] = rec.get("zone_id") == focus_id
            if isinstance(live_snap, dict) and live_snap.get("live") is True:
                tracker.save()
        except Exception as e_att:
            _catat_error("attach_early_tracker", e_att)

    def finish(status, reason):
        dec["status"] = status
        is_closed_fin = (
            str(dec.get("market_status") or "").upper() == "CLOSED"
            or str(dec.get("snapshot_status") or "").upper() == "MARKET_CLOSED"
        )
        m5_tech_ok = (
            not is_closed_fin
            and (st.get("structure_trigger") or {}).get("status") == "CONFIRMED"
            and (st.get("displacement") or {}).get("status") == "CONFIRMED"
        )
        dec["smc_technical_status"] = "CONFIRMED" if (status == "VALID" or m5_tech_ok) else "WAIT"
        dec["entry_permission"] = "WAIT (MARKET CLOSED)" if is_closed_fin else ("GRANTED" if status == "VALID" else "WAIT")
        if directional:
            focus_id = (dec.get("zone") or {}).get("zone_id")
            remaining = [z for z in directional if z.get("zone_id") != focus_id]
            dec["next_radar_zone"] = min(
                remaining,
                key=lambda z: max(float(z["price_low"]) - price, price - float(z["price_high"]), 0.0),
            ) if remaining else None
        if reason:
            dec["reasons"].append(reason)
        if status != "VALID":
            dec["plan"] = None
            if dec.get("signal_status") == SIGNAL_STATUS_SIGNAL_CONFIRMED:
                dec["signal_status"] = SIGNAL_STATUS_WAITING_CONFIRMATION
        for comp in ("zone_validation", "zone_touch", "liquidity_sweep", "structure_trigger", "displacement", "entry_validation", "risk_reward"):
            if comp not in st:
                st[comp] = {"status": "NOT_CHECKED", "msg": "Belum diperiksa"}

        # Attach radar priority information
        if isinstance(scan, dict) and isinstance(frames, dict) and isinstance(price, (int, float)) and math.isfinite(price) and price > 0:
            all_r = scan.get("radar_zones") or scan.get("active_zones", [])
            h4_bias_val = ctx.get("H4", "NETRAL")
            h1_struct_val = ctx.get("H1", "NETRAL")
            atr_diag_dec = scan.get("atr_diagnostics") or extract_atr_diagnostics(frames, preferred_tf="H1")
            ref_atr_val = atr_diag_dec.get("atr_value")
            dec["atr_diagnostics"] = atr_diag_dec
            winning_cand = winning_zdict if status == "VALID" else None
            prio = prioritize_radar_zones(
                radar_zones=all_r,
                current_price=float(price or 0.0),
                htf_bias=h4_bias_val,
                h1_structure=h1_struct_val,
                atr=ref_atr_val,
                m5_stages={**st, "_zone_id": (dec.get("zone") or {}).get("zone_id")},
                validated_entry_zone=winning_cand,
                frames=frames,
                m30_status=st.get("M30", {}).get("status", "CONFIRMED"),
                server_time=server_time,
                bid=scan.get("bid"),
                ask=scan.get("ask"),
                snapshot_time=scan.get("snapshot_time"),
                snapshot_id=scan.get("snapshot_id") or dec.get("snapshot_id"),
                market_status=dec.get("market_status"),
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
            dec["far_reference_zones"] = prio.get("far_reference_zones", [])
            if not dec.get("far_reference_zone") and prio.get("far_reference_zone"):
                dec["far_reference_zone"] = prio.get("far_reference_zone")
            if dec.get("primary_zone") is None and prio.get("primary_directional_zone") is not None:
                dec["primary_zone"] = prio.get("primary_directional_zone")
            dec["zone_count_reconciliation"] = prio.get("zone_count_reconciliation", {})
            dec["zone_plans"] = _provisional_zone_plans(
                dec, price, frames, h4_bias_val, st.get("M30", {}).get("status", "CONFIRMED"), server_time,
                bid=scan.get("bid"), ask=scan.get("ask"),
            )
            # ENTRY VALIDATION SUMMARY (Sinkronkan zona yang dipantau tanpa pernah mempromosikan FAR_REFERENCE / countertrend / all_r[0])
            locked_signal_zone = None
            try:
                import modules.state_manager as _sm
                _st_data = _sm.muat_state_sinyal() or {}
                _sig_data = _st_data.get("sinyal") if isinstance(_st_data, dict) else None
                if isinstance(_sig_data, dict):
                    _arah = str(_sig_data.get("arah", "")).upper()
                    _lo = float(_sig_data.get("lo", 0.0) or 0.0)
                    _hi = float(_sig_data.get("hi", 0.0) or 0.0)
                    _sl = float(_sig_data.get("sl", 0.0) or 0.0)
                    _cp = float(price or 0.0)
                    if _lo > 0 and _hi > 0 and _arah in ("BUY", "SELL"):
                        if _lo > _hi:
                            _lo, _hi = _hi, _lo
                        _invalid = False
                        if _cp > 0 and _sl > 0:
                            if _arah == "SELL" and _cp > _sl:
                                _invalid = True
                            if _arah == "BUY" and _cp < _sl:
                                _invalid = True
                        if not _invalid:
                            _ztype = "SUPPLY" if _arah == "SELL" else "DEMAND"
                            _match = next((z for z in (all_r or [])
                                           if str(z.get("zone_type", "")).upper() == _ztype
                                           and abs(float(z.get("price_low", 0.0)) - _lo) <= 1.5
                                           and abs(float(z.get("price_high", 0.0)) - _hi) <= 1.5), None)
                            if _match is not None:
                                locked_signal_zone = _match
                            else:
                                locked_signal_zone = {
                                    "zone_id": f"LOCKED_SIGNAL_{_arah}_{_lo:.2f}_{_hi:.2f}",
                                    "timeframe": "ACTIVE_SIGNAL",
                                    "zone_type": _ztype,
                                    "price_low": _lo,
                                    "price_high": _hi,
                                    "status": "SIGNAL_LOCKED",
                                    "approach": "TOUCHED" if (_cp > 0 and _lo <= _cp <= _hi) else "UNTOUCHED",
                                }
            except Exception:
                locked_signal_zone = None

            def _is_valid_active_focus(cand_z):
                if not isinstance(cand_z, dict) or not cand_z.get("zone_id"):
                    return False
                if str(cand_z.get("timeframe") or "").upper() == "ACTIVE_SIGNAL":
                    return True
                if str(cand_z.get("grade") or "").upper() == "REJECTED" or cand_z.get("risk_rejected") is True:
                    return False
                s_elig_f = str(cand_z.get("setup_eligibility") or "").upper()
                if s_elig_f.startswith("NOT_ELIGIBLE") and ("JARAK SL" in s_elig_f or "MELEBIHI BATAS" in s_elig_f):
                    return False
                if not is_zone_active_and_valid(cand_z, float(price or 0.0)):
                    return False
                if is_far_reference_zone(cand_z, current_price=float(price or 0.0), atr=ref_atr_val):
                    return False
                if h4_bias_val in ("BULLISH", "BEARISH"):
                    zt_f = str(cand_z.get("zone_type") or "").upper()
                    want_f = "DEMAND" if h4_bias_val == "BULLISH" else "SUPPLY"
                    if cand_z.get("is_counter") or (zt_f in ("DEMAND", "SUPPLY") and zt_f != want_f):
                        return False
                return True

            monitored_ref = None
            raw_focus_cands = [
                dec.get("zone"),
                winning_zdict if ("winning_zdict" in locals() and winning_zdict and status == "VALID") else None,
                best_candidate_dict if "best_candidate_dict" in locals() else None,
                locked_signal_zone,
            ]
            if not dec.get("focus_selection_evaluated"):
                raw_focus_cands.append(prio.get("best_confirmed_candidate"))
                if status == "VALID":
                    raw_focus_cands.append(prio.get("best_pending_candidate"))
            for fc_item in raw_focus_cands:
                if _is_valid_active_focus(fc_item):
                    monitored_ref = fc_item
                    break

            if monitored_ref:
                z_lo_val = float(monitored_ref.get("price_low", 0.0))
                z_hi_val = float(monitored_ref.get("price_high", 0.0))
                if z_lo_val > 0 and z_hi_val >= z_lo_val and price > 0:
                    d_sync = calculate_zone_distance(
                        z_lo_val, z_hi_val, price,
                        zone_type=monitored_ref.get("zone_type"),
                        bid=scan.get("bid"), ask=scan.get("ask"), atr=ref_atr_val,
                    )
                    app_sync = str(calculate_zone_approach(d_sync.distance, d_sync.status, ref_atr_val or 2.0))
                    monitored_ref["distance"] = d_sync.distance
                    monitored_ref["direction_status"] = d_sync.status
                    monitored_ref["approach"] = app_sync
                    if d_sync.distance_atr_ratio is not None:
                        monitored_ref["distance_atr_ratio"] = d_sync.distance_atr_ratio
                    if status == "VALID":
                        monitored_ref["status"] = "CONFIRMED"
                        monitored_ref["lifecycle_status"] = "CONFIRMED"
                    elif monitored_ref.get("timeframe") != "ACTIVE_SIGNAL":
                        if d_sync.status == "INSIDE_ZONE":
                            monitored_ref["status"] = "ZONE_TOUCHED"
                        elif app_sync == "APPROACHING":
                            monitored_ref["status"] = "APPROACHING"
                        else:
                            monitored_ref["status"] = "RADAR"
                        monitored_ref["lifecycle_status"] = monitored_ref["status"]
                z_lbl = f"{monitored_ref.get('timeframe', 'H1')} {monitored_ref.get('zone_type', '')} ({monitored_ref.get('price_low', 0.0):.2f} - {monitored_ref.get('price_high', 0.0):.2f})"
                z_dir_val = "SELL" if monitored_ref.get("zone_type") == "SUPPLY" else ("BUY" if monitored_ref.get("zone_type") == "DEMAND" else "NONE")
                z_stype_val = str(dec.get("setup_type") or monitored_ref.get("setup_type") or "TREND_PULLBACK")
                z_id_val = str(monitored_ref.get("zone_id", ""))
                z_now_val = format_wib_timestamp(default_now=True)
                dec["zone"] = monitored_ref
                dec["monitored_zone_identity"] = {
                    "setup_id": f"{z_id_val}:{z_dir_val}:{z_stype_val}" if z_id_val else "N/A",
                    "zone_id": z_id_val,
                    "direction": z_dir_val,
                    "zone_range": (z_lo_val, z_hi_val),
                    "setup_type": z_stype_val,
                    "timeframe": monitored_ref.get("timeframe", "H1"),
                    "created_at": monitored_ref.get("created_at") or monitored_ref.get("first_detected") or z_now_val,
                    "last_updated": z_now_val,
                    "lifecycle_status": monitored_ref.get("lifecycle_status") or monitored_ref.get("status", "RADAR"),
                    "zone_type": monitored_ref.get("zone_type", ""),
                    "proximal": float(monitored_ref.get("price_low" if monitored_ref.get("zone_type") == "SUPPLY" else "price_high", 0.0)),
                    "distal": float(monitored_ref.get("price_high" if monitored_ref.get("zone_type") == "SUPPLY" else "price_low", 0.0)),
                    "price_low": z_lo_val,
                    "price_high": z_hi_val,
                    "status": monitored_ref.get("status", "RADAR"),
                }
            else:
                z_lbl = "NONE"
                dec["zone"] = None
                dec["monitored_zone_identity"] = None
            _confl_target = monitored_ref or dec.get("primary_zone")
            _snr_for_confl = dec.get("snr_context") or (scan.get("snr_context") if isinstance(scan, dict) else None)
            if isinstance(_snr_for_confl, dict):
                _confl_res = evaluate_smc_snr_confluence(
                    _confl_target, _snr_for_confl, current_price=price, htf_bias=h4_bias_val
                )
                dec["smc_snr_confluence"] = _confl_res
                if isinstance(scan, dict):
                    scan["smc_snr_confluence"] = _confl_res

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

        _attach_early_and_tracker(dec, scan if isinstance(scan, dict) else None)
        zone = dec.get("zone") or {}
        identity = dec.get("monitored_zone_identity") or {}
        zone_id = zone.get("zone_id")
        pending_setup = (dec.get("early_setup") or {}).get("best_setup") or {}
        _act_struct_inv = resolve_smc_structure_invalidation_price(zone) if isinstance(zone, dict) and zone else 0.0
        _act_setup_sl = (
            float((dec.get("plan") or {}).get("sl") or 0.0)
            or resolve_smc_setup_invalidation_price(zone, frames=frames, spread=float(spread or 0.0))
        ) if isinstance(zone, dict) and zone else 0.0
        dec["active_setup"] = {
            **identity,
            "zone_id": zone_id,
            "structure_invalidation": _act_struct_inv,
            "invalidation_price": _act_struct_inv,
            "setup_sl": _act_setup_sl,
            "setup_invalidation_price": _act_setup_sl,
            "invalidation_type": str(zone.get("invalidation_type") or "") if isinstance(zone, dict) else "",
            "snapshot_id": dec.get("snapshot_id"),
            "validation_time": dec.get("snapshot_time") or dec.get("timestamp"),
            "last_closed_m5": (dec.get("last_candle_times") or {}).get("M5"),
            "zone_status": (st.get("zone_validation") or {}).get("status", "NOT_CHECKED"),
            "touch_status": (st.get("zone_touch") or {}).get("status", "NOT_CHECKED"),
            "trigger_status": (st.get("structure_trigger") or {}).get("status", "NOT_CHECKED"),
            "entry_status": (st.get("entry_validation") or {}).get("status", "NOT_CHECKED"),
            "pending_status": pending_setup.get("status", "WAIT") if zone_id and pending_setup.get("zone_id") == zone_id else "NOT_APPLICABLE",
            "smc_technical_status": dec.get("smc_technical_status", "WAIT"),
            "entry_permission": dec.get("entry_permission", "WAIT"),
            "final_decision": dec.get("direction") if status == "VALID" else status,
            "blocker": (dec.get("reasons") or [""])[0] if status != "VALID" else "",
        }
        dec["analysis_direction"] = resolve_analysis_direction(
            ctx=ctx,
            decision=dec,
            scan=scan if isinstance(scan, dict) else None,
            frames=frames if isinstance(frames, dict) else None,
            market_status=dec.get("market_status"),
            data_ok=(st.get("DATA") or {}).get("ok") is not False and bool(not isinstance(scan, dict) or scan.get("data_ok", True)),
        )
        try:
            _smc_tp_list, _smc_tp_best = generate_smc_trade_plans(
                dec=dec,
                scan=scan if isinstance(scan, dict) else None,
                frames=frames if isinstance(frames, dict) else None,
                price=float(price or 0.0),
                bid=(scan.get("bid") if isinstance(scan, dict) else None),
                ask=(scan.get("ask") if isinstance(scan, dict) else None),
                spread=float(spread or 0.0),
            )
            dec["trade_plans"] = _smc_tp_list
            dec["best_trade_plan"] = _smc_tp_best
        except Exception as _tp_exc:
            _catat_error("generate_smc_trade_plans", _tp_exc)
            dec["trade_plans"] = []
            dec["best_trade_plan"] = None
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
        if live_snap.get("data_freshness"):
            dec["data_freshness"] = live_snap.get("data_freshness")
        if isinstance(live_snap.get("data_freshness_info"), dict):
            dec["data_freshness_info"] = live_snap.get("data_freshness_info")
        if live_snap.get("market_status"):
            dec["market_status"] = live_snap.get("market_status")
        if live_snap.get("connection_status"):
            dec["connection_status"] = live_snap.get("connection_status")
        if isinstance(live_snap.get("last_candle_times"), dict):
            dec["last_candle_times"] = live_snap.get("last_candle_times")
        if isinstance(live_snap.get("forming_candles"), dict):
            dec["forming_candles"] = live_snap.get("forming_candles")
        if live_snap.get("snapshot_status") == "MARKET_CLOSED" or dec.get("market_status") == "CLOSED":
            dec["snapshot_status"] = "MARKET_CLOSED"
            dec["data_freshness"] = live_snap.get("data_freshness") or "STALE"
            dec["market_status"] = "CLOSED"
            if frames is None:
                frames = live_snap.get("frames")
            if price is None:
                price = live_snap.get("snapshot_price") if live_snap.get("snapshot_price") is not None else live_snap.get("mid")
            if price is not None:
                dec["current_price"] = price
            if not dec.get("context") and ctx:
                dec["context"] = ctx
            if "M30" not in st and ctx:
                st["M30"] = ctx.get("m30_eval") or {"status": ctx.get("m30_status", "CONFIRMED"), "ok": True, "msg": "Data historis closed candle"}
            closed_msg = live_snap.get("error_reason") or "Market sedang tutup (data sesi penutupan terakhir; bukan koneksi MT5 terputus)"
            st["DATA"] = {"ok": False, "msg": closed_msg}
            if not scan:
                scan = scan_result or scan_xauusd_zones(frames=live_snap.get("frames") or frames, snapshot=live_snap, market_status="CLOSED")
            if isinstance(scan, dict) and not dec.get("context") and scan.get("htf_context"):
                ctx = scan["htf_context"]
                dec["context"] = ctx
            dec["premarket"] = build_premarket_analysis(live_snap.get("frames") or frames, live_snap, scan_result=scan)
            dec["live_signal_disabled"] = True
            dec["signal_status"] = "NO_LIVE_SIGNAL"
            dec["entry_permission"] = "WAIT (MARKET CLOSED)"
            return finish("WAIT", closed_msg)
        if live_snap.get("snapshot_status") == "DATA_INCOMPLETE":
            dec["snapshot_status"] = "DATA_INCOMPLETE"
            dec["data_freshness"] = live_snap.get("data_freshness") or "INVALID"
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
        if scan_result.get("data_freshness"):
            dec["data_freshness"] = scan_result.get("data_freshness")
        if isinstance(scan_result.get("data_freshness_info"), dict):
            dec["data_freshness_info"] = scan_result.get("data_freshness_info")
        if scan_result.get("market_status"):
            dec["market_status"] = scan_result.get("market_status")
        if scan_result.get("connection_status"):
            dec["connection_status"] = scan_result.get("connection_status")
        if isinstance(scan_result.get("last_candle_times"), dict):
            dec["last_candle_times"] = scan_result.get("last_candle_times")
        if isinstance(scan_result.get("forming_candles"), dict):
            dec["forming_candles"] = scan_result.get("forming_candles")
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
                dec["data_freshness"] = "NO DATA"
                dec["connection_status"] = "DISCONNECTED"
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
        dec["data_freshness"] = "NO DATA"
        st["DATA"] = {"ok": False, "msg": "DATA_INCOMPLETE: Data OHLC tidak tersedia pada snapshot/scan_result"}
        return finish("WAIT", "DATA_INCOMPLETE: Data OHLC tidak tersedia pada snapshot/scan_result")

    if isinstance(scan_result, dict):
        scan = scan_result

    # --- DATA ---
    for tf in ("H4", "H1", "M30", "M5"):
        df = frames.get(tf)
        if df is None or len(df) < MIN_BARS_DECISION:
            dec["data_freshness"] = "NO DATA" if (df is None or len(df) == 0) else "INVALID"
            st["DATA"] = {"ok": False, "msg": f"Data {tf} kosong/tidak cukup"}
            return finish("WAIT", f"Data {tf} kosong/tidak cukup")
    if not (isinstance(price, (int, float)) and math.isfinite(price) and price > 0):
        dec["data_freshness"] = "INVALID"
        st["DATA"] = {"ok": False, "msg": "Harga tidak valid"}
        return finish("WAIT", "Harga tidak valid/tidak tersedia")

    eff_mkt_dec = str(market_status or dec.get("market_status") or "UNKNOWN").upper()
    if eff_mkt_dec == "CLOSED":
        dec["market_status"] = "CLOSED"
        dec["data_freshness"] = "STALE"
        closed_msg = "Market sedang tutup (bukan error koneksi MT5)"
        st["DATA"] = {"ok": False, "msg": closed_msg}
        if not scan:
            scan = scan_result or scan_xauusd_zones(frames=frames, current_price=price, market_status="CLOSED")
        dec["premarket"] = build_premarket_analysis(frames, live_snap, scan_result=scan)
        dec["live_signal_disabled"] = True
        dec["signal_status"] = "NO_LIVE_SIGNAL"
        dec["entry_permission"] = "WAIT (MARKET CLOSED)"
        return finish("WAIT", closed_msg)

    stale = _stale_reason(frames, server_time)
    if stale:
        dec["data_freshness"] = "STALE"
        st["DATA"] = {"ok": False, "msg": stale}
        return finish("WAIT", stale)

    # Cek candle berjalan / belum selesai (baik dari atribut df maupun server_time)
    for tf in ("M5", "M30", "H1", "H4"):
        df_tf = frames.get(tf)
        if df_tf is not None and len(df_tf) > 0:
            if df_tf.attrs.get("is_closed") is False:
                st["DATA"] = {"ok": False, "msg": f"Candle terakhir {tf} belum selesai (masih berjalan)"}
                return finish("WAIT", f"Candle terakhir {tf} belum selesai (masih berjalan): konfirmasi entry hanya dari candle close")
            if server_time is not None:
                tf_sec = TF_MINUTES[tf] * 60
                last_ts = _get_df_last_timestamp(df_tf)
                if last_ts is not None:
                    try:
                        st_al, last_al = _align_timestamps_for_diff(
                            pd.Timestamp(server_time), last_ts,
                            b_tz_hint=str(df_tf.attrs.get("time_tz", "WIB")),
                        )
                        diff = (st_al - last_al).total_seconds()
                        if 0 <= diff < tf_sec - 1e-6:
                            st["DATA"] = {"ok": False, "msg": f"Candle terakhir {tf} belum selesai (masih berjalan)"}
                            return finish("WAIT", f"Candle terakhir {tf} belum selesai (masih berjalan): konfirmasi entry hanya dari candle close")
                    except Exception:
                        pass

    fresh_dec = validate_data_freshness(
        frames,
        server_time=server_time,
        market_status=eff_mkt_dec,
        connection_status=dec.get("connection_status", "UNKNOWN"),
        snapshot_status=dec.get("snapshot_status", "OK"),
    )
    if dec.get("data_freshness") in ("NO DATA", "UNKNOWN", "", None):
        dec["data_freshness"] = fresh_dec["status"]
    if not dec.get("data_freshness_info"):
        dec["data_freshness_info"] = fresh_dec
    if not dec.get("last_candle_times"):
        dec["last_candle_times"] = fresh_dec["last_candle_times"]
    if dec.get("data_freshness") in ("STALE", "INVALID", "NO DATA"):
        st["DATA"] = {"ok": False, "msg": fresh_dec.get("reason") or f"Data status {dec['data_freshness']}"}
        return finish("WAIT", st["DATA"]["msg"])
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
        _accepts_mkt = "market_status" in _scan_sig.parameters or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in _scan_sig.parameters.values()
        )
        if _accepts_snapshot:
            scan_kwargs: Dict[str, Any] = {"frames": frames, "current_price": price, "snapshot": live_snap}
            if _accepts_mkt and eff_mkt_dec != "UNKNOWN":
                scan_kwargs["market_status"] = eff_mkt_dec
            scan = scan_xauusd_zones(**scan_kwargs)
        else:
            _catat_error(
                "evaluate_trade_decision_signature",
                RuntimeError("scan_xauusd_zones tidak menerima parameter snapshot; menggunakan pemanggilan kompatibilitas lama"),
            )
            scan = scan_xauusd_zones(frames=frames, current_price=price)
    if not dec.get("snapshot_id") and isinstance(scan, dict):
        dec["snapshot_id"] = scan.get("snapshot_id")
    if isinstance(scan, dict):
        if scan.get("data_freshness") and dec.get("data_freshness") in ("FRESH", "NO DATA", "UNKNOWN"):
            dec["data_freshness"] = scan.get("data_freshness")
        if scan.get("last_candle_times"):
            dec["last_candle_times"] = scan.get("last_candle_times")
        if scan.get("forming_candles"):
            dec["forming_candles"] = scan.get("forming_candles")
        if scan.get("market_status") and dec.get("market_status") in ("UNKNOWN", ""):
            dec["market_status"] = scan.get("market_status")
        if isinstance(scan.get("snr_context"), dict):
            dec["snr_context"] = scan.get("snr_context")
    ctx = scan.get("htf_context") or evaluate_htf_context(frames)
    if not scan.get("data_ok"):
        return finish("WAIT", scan.get("empty_reason") or "Scanner tidak dapat berjalan")
    if scan.get("snapshot_status") == "DATA_INCOMPLETE":
        dec["snapshot_status"] = "DATA_INCOMPLETE"
        return finish("WAIT", scan.get("empty_reason") or "DATA_INCOMPLETE: Snapshot MT5 tidak lengkap atau tidak valid")
    if scan.get("data_freshness") in ("STALE", "INVALID", "NO DATA") or scan.get("market_status") == "CLOSED":
        if scan.get("market_status") == "CLOSED":
            dec["market_status"] = "CLOSED"
            closed_msg = "Market sedang tutup (bukan error koneksi MT5)"
            dec["premarket"] = build_premarket_analysis(frames, live_snap, scan_result=scan)
            dec["live_signal_disabled"] = True
            dec["signal_status"] = "NO_LIVE_SIGNAL"
            dec["entry_permission"] = "WAIT (MARKET CLOSED)"
            return finish("WAIT", closed_msg)
        f_msg = (scan.get("data_freshness_info") or {}).get("reason") or f"Data tidak segar ({scan.get('data_freshness')})"
        st["DATA"] = {"ok": False, "msg": f_msg}
        return finish("WAIT", f_msg)
    if scan.get("is_cached") and float(scan.get("cache_age_seconds") or 0.0) > SCAN_CACHE_TTL:
        return finish("WAIT", "Data cache scanner kedaluwarsa (stale cache)")

    # --- H4: bias & arah tren utama ---
    h4 = ctx["H4"]
    if h4 not in ("BULLISH", "BEARISH"):
        st["H4"] = {"ok": False, "msg": f"Bias H4 {h4}"}
        dec["setup_type"] = "NONE"
        dec["signal_status"] = SIGNAL_STATUS_NO_SETUP
        return finish("WAIT", f"Bias H4 tidak jelas ({h4})")
    direction = "BUY" if h4 == "BULLISH" else "SELL"
    dec["direction"] = direction
    dec["setup_type"] = "NONE"
    dec["signal_status"] = SIGNAL_STATUS_NO_SETUP
    ztype = "DEMAND" if direction == "BUY" else "SUPPLY"

    h4_eval_tf = evaluate_h4_primary_trend(frames.get("H4"), regime_h4=ctx.get("h4_detail"))
    if h4_eval_tf["bias"] != h4:
        h4_eval_tf["bias"] = h4
        h4_eval_tf["ok"] = True
    h1_eval_tf = evaluate_h1_trend_confirmation(
        frames.get("H1"), h4_eval_tf, regime_h1=ctx.get("h1_regime"), explicit_conflict=bool(ctx.get("has_conflict"))
    )

    def _eval_m5_during_conflict(conflict_code: str, conflict_summary: str):
        """Saat HTF/M30 konflik, tetap evaluasi M5 pada zona eligible yang didekati/disentuh/baru ditinggalkan tanpa auto-signal."""
        m5_df_c = frames.get("M5") if isinstance(frames, dict) else None
        if m5_df_c is None or len(m5_df_c) < MIN_BARS_DECISION or not isinstance(scan, dict):
            dec["m5_eval_status"] = "DATA_UNAVAILABLE"
            st["M5"] = {"ok": False, "status": "DATA_UNAVAILABLE", "msg": f"{conflict_code}: data M5 tidak tersedia/belum cukup"}
            return finish("WAIT", f"{conflict_summary} ({conflict_code})")
        recent_m5_c = m5_df_c.tail(6)
        pool_c = []
        seen_c = set()
        fallback_pool_c = []
        for zl_c in (scan.get("active_zones") or [], scan.get("radar_zones") or []):
            for item_c in zl_c:
                zd_c = item_c.to_dict() if hasattr(item_c, "to_dict") else (item_c if isinstance(item_c, dict) else None)
                if not isinstance(zd_c, dict):
                    continue
                zid_c = str(zd_c.get("zone_id") or "").strip()
                if not zid_c or zid_c in seen_c:
                    continue
                if zd_c.get("is_valid") is False or zd_c.get("is_broken") is True:
                    continue
                if str(zd_c.get("grade") or "").upper() == "REJECTED" or zd_c.get("risk_rejected") is True:
                    continue
                if is_far_reference_zone(zd_c, current_price=float(price)):
                    continue
                lo_c = float(zd_c.get("price_low", zd_c.get("low", 0.0)) or 0.0)
                hi_c = float(zd_c.get("price_high", zd_c.get("high", 0.0)) or 0.0)
                if lo_c <= 0 or hi_c < lo_c:
                    continue
                dist_c = max(lo_c - float(price), float(price) - hi_c, 0.0)
                recently_touched_c = bool(
                    not recent_m5_c.empty
                    and ((recent_m5_c["Low"] <= hi_c) & (recent_m5_c["High"] >= lo_c)).any()
                )
                was_touched_c = bool(
                    recently_touched_c
                    or int(zd_c.get("touch_count") or 0) > 0
                    or str(zd_c.get("lifecycle_status") or zd_c.get("status") or "").upper() in ("TOUCHED", "ZONE_TOUCHED", "BREAKOUT_PENDING")
                )
                seen_c.add(zid_c)
                if dist_c <= float(ZONE_APPROACH_DISTANCE_POINTS) or was_touched_c:
                    # Prioritaskan zona yang disentuh/baru ditinggalkan sebelum approaching
                    pool_c.append((0 if (dist_c == 0.0 or recently_touched_c) else 1, dist_c, zd_c))
                else:
                    fallback_pool_c.append((2, dist_c, zd_c))
        if not pool_c and fallback_pool_c:
            # HTF_CONFLICT tetap filter entry, bukan alasan melewati evaluasi M5 bila zona SMC eligible tersedia.
            pool_c = fallback_pool_c
        if not pool_c:
            snr_c_fb = dec.get("snr_context") or scan.get("snr_context") or {}
            snr_m5_fb = str(
                (snr_c_fb.get("decision") or {}).get("m5_eval_status")
                or (snr_c_fb.get("entry_validation") or {}).get("m5_eval_status")
                or ""
            ).strip().upper()
            if snr_m5_fb in ("CONFIRMED", "UNCONFIRMED"):
                dec["m5_conflict_evaluated"] = True
                dec["m5_eval_status"] = snr_m5_fb
                st["M5"] = {"ok": False, "status": snr_m5_fb, "blocked_by": conflict_code, "msg": f"M5 {snr_m5_fb} (SNR) namun diblokir {conflict_code}"}
                return finish("WAIT", f"{conflict_code}: M5 {snr_m5_fb} (SNR) — diblokir filter risiko {conflict_code}")
            dec["m5_eval_status"] = "NOT_CHECKED"
            st["M5"] = {"ok": False, "status": "NOT_CHECKED", "msg": f"{conflict_code}: belum ada zona eligible didekati/disentuh"}
            return finish("WAIT", f"{conflict_summary} — belum ada zona eligible didekati/disentuh ({conflict_code})")
        pool_c.sort(key=lambda item: (item[0], round(item[1], 2), -round(float(item[2].get("score") or 0.0), 1), str(item[2].get("zone_id") or "")))
        _, _, target_zd = pool_c[0]
        target_zn = _zone_from_dict(target_zd)
        snr_c = dec.get("snr_context") or scan.get("snr_context")
        r_c = _entry_check(
            target_zn,
            float(price),
            m5_df_c,
            float(spread or 0.0),
            news,
            frames,
            server_time=server_time,
            snr_context=snr_c,
            bid=scan.get("bid"),
            ask=scan.get("ask"),
            market_status=eff_mkt_dec,
        )
        for k_st, v_st in (r_c.get("stages") or {}).items():
            st[k_st] = v_st
        dec["m5"] = dict(r_c.get("m5") or {})
        dec["m5_conflict_evaluated"] = True
        if dec.get("zone") is None:
            dec["zone"] = target_zd
        m5_trig_ok = bool(r_c.get("smc_technical_confirmed") or r_c.get("status") == "VALID")
        m5_rsn_short = (r_c.get("reasons") or ["M5 dievaluasi (belum konfirmasi)"])[0]
        if m5_trig_ok:
            dec["m5_eval_status"] = "CONFIRMED"
            st["M5"] = {"ok": False, "status": "CONFIRMED", "blocked_by": conflict_code, "msg": f"M5 CONFIRMED di {target_zn.timeframe} {target_zn.zone_type} namun diblokir {conflict_code}"}
            return finish("WAIT", f"{conflict_code}: M5 CONFIRMED di {target_zn.timeframe} {target_zn.zone_type} ({target_zn.price_low:.2f}-{target_zn.price_high:.2f}), diblokir filter risiko {conflict_code}")
        dec["m5_eval_status"] = "UNCONFIRMED"
        st["M5"] = {"ok": False, "status": "UNCONFIRMED", "blocked_by": conflict_code, "msg": f"M5 UNCONFIRMED ({conflict_code}): {m5_rsn_short}"}
        return finish("WAIT", f"{conflict_code}: M5 UNCONFIRMED di {target_zn.timeframe} {target_zn.zone_type} — {m5_rsn_short}")

    if ctx.get("has_conflict"):
        # Saat H4 vs H1 konflik serius: periksa strict Reversal HTF sebelum menolak
        tf_rev = evaluate_trend_following_setup(
            frames,
            float(price),
            scan=scan,
            spread=float(spread if spread is not None else 0.0),
            news_blackout=news,
            explicit_h1_conflict=True,
            explicit_m30_conflict=bool(ctx.get("m30_conflict")),
        )
        rev_id = (tf_rev.get("best_candidate") or {}).get("zone_id")
        rev_record = (get_zone_tracker(auto_save=False).get_zone(rev_id) or {}) if rev_id else {}
        rev_expiry = to_wib_datetime(rev_record.get("expires_at")) if rev_record.get("expires_at") else None
        if rev_record.get("status") in ("INVALIDATED", "EXPIRED") or (
            rev_expiry is not None and (to_wib_datetime(server_time) or get_current_wib()) > rev_expiry
        ):
            tf_rev["status"] = "WAIT"
        if tf_rev.get("status") == "VALID" and tf_rev.get("setup_type") == "REVERSAL" and tf_rev.get("plan"):
            bc = tf_rev["best_candidate"]
            plan_rev = tf_rev["plan"]
            winning_zdict = {
                "zone_id": bc.get("zone_id", "TF_REVERSAL"),
                "symbol": dec.get("symbol", SYMBOL),
                "timeframe": bc.get("timeframe", "H1/M30"),
                "zone_type": bc.get("zone_type", "DEMAND" if tf_rev["direction"] == "BUY" else "SUPPLY"),
                "price_low": float(bc["price_low"]),
                "price_high": float(bc["price_high"]),
                "midpoint": round((float(bc["price_low"]) + float(bc["price_high"])) / 2.0, 2),
                "invalidation_price": float(
                    bc.get("invalidation_price")
                    or (float(bc["price_low"]) if str(bc.get("zone_type") or ("DEMAND" if tf_rev["direction"] == "BUY" else "SUPPLY")).upper() == "DEMAND" else float(bc["price_high"]))
                ),
                "setup_sl": float(plan_rev["sl"]),
                "setup_invalidation_price": float(plan_rev["sl"]),
                "sl_recommended": float(plan_rev["sl"]),
                "grade": bc.get("grade", "B"),
                "score": float(bc.get("score", 74.0)),
                "is_valid": True,
                "setup_type": "REVERSAL",
            }
            winning_zone = _zone_from_dict(winning_zdict)
            valid_plan = plan_rev
            dec["direction"] = tf_rev["direction"]
            dec["setup_type"] = "REVERSAL"
            dec["signal_status"] = SIGNAL_STATUS_SIGNAL_CONFIRMED
            dec["trend_following"] = tf_rev
            dec["snr_context"] = tf_rev.get("snr")
            dec["zone"] = winning_zdict
            dec["plan"] = plan_rev
            for k_st, v_st in (tf_rev.get("m5", {}).get("stages") or {}).items():
                st[k_st] = v_st
            st["H4"] = {"ok": True, "msg": f"Transisi Reversal dari H4 {h4}"}
            st["H1"] = {"ok": True, "msg": "CHoCH H1 terkonfirmasi untuk Reversal"}
            st["M30"] = {"ok": True, "status": "CONFIRMED", "msg": tf_rev.get("m30", {}).get("msg", "CHoCH M30 terkonfirmasi")}
            st["M5"] = {"ok": True, "msg": "; ".join(tf_rev.get("reasons", []))}
            st["RISK"] = {"ok": True, "msg": f"R:R 1:{plan_rev.get('rr', '-')}"}
            dec["reasons"].extend(tf_rev.get("reasons", []))
            return finish("VALID", "")

        st["H1"] = {"ok": False, "msg": f"H1 {ctx['H1']} bertentangan dengan H4 {h4}"}
        st["H4"] = {"ok": True, "msg": f"Bias H4 {h4}"}
        m30_c_dir = str(ctx.get("M30") or "").upper()
        if m30_c_dir in ("BULLISH", "BEARISH") and m30_c_dir == str(ctx.get("H1") or "").upper() and m30_c_dir != h4:
            conflict_reason = f"Konflik struktur H4 {h4} vs H1/M30 {ctx['H1']}"
        else:
            conflict_reason = f"Konflik struktur H4 {h4} vs H1 {ctx['H1']}"
        return _eval_m5_during_conflict("HTF_CONFLICT", conflict_reason)

    # --- H1: konfirmasi / pullback sehat ---
    h1_tag = ctx.get("h1_confirmation") or h1_eval_tf.get("status", "ALIGNED")
    st["H1"] = {
        "ok": True,
        "status": h1_tag,
        "msg": f"H1 {ctx['H1']} ({h1_tag}) terhadap H4 {h4}; struktur utama terjaga",
    }

    # --- M30: bedakan ALIGNED / PULLBACK / CONSOLIDATION vs CONFLICT serius / WEAK datar ---
    m30_val = evaluate_m30_momentum(frames.get("M30"), h4)
    m30_cond = assess_m30_condition(
        frames, h4_eval_tf, h1_eval_tf, float(price),
        regime_m30=ctx.get("m30_regime"),
    )
    if ctx.get("m30_conflict") or m30_cond.get("status") == "CONFLICT":
        ctx_m30_cond = ctx.get("m30_condition") if isinstance(ctx.get("m30_condition"), dict) else {}
        m30_msg = (
            (m30_cond.get("msg") if m30_cond.get("status") == "CONFLICT" else None)
            or (ctx_m30_cond.get("msg") if ctx.get("m30_conflict") else None)
            or m30_val.get("msg")
            or f"M30 bertentangan dengan H4 {h4}"
        )
        st["M30"] = {"ok": False, "status": "CONFLICT", "msg": m30_msg}
        return _eval_m5_during_conflict("M30_CONFLICT", m30_msg if f"H4 {h4}" in m30_msg else f"M30 konflik terhadap H4 {h4}")
    if m30_cond.get("status") == "INVALID" or m30_val.get("status") == "INVALID":
        st["M30"] = {"ok": False, "status": "INVALID", "msg": m30_val.get("msg") or m30_cond.get("msg", "Data M30 tidak valid")}
        return finish("WAIT", f"M30 INVALID: {st['M30']['msg']}")
    if m30_cond.get("status") == "WEAK":
        st["M30"] = {"ok": False, "status": "WEAK", "msg": m30_cond.get("msg") or m30_val.get("msg")}
        return finish("WAIT", f"M30 tidak memiliki momentum (status WEAK: {st['M30']['msg']})")

    m30_effective_status = (
        "CONFIRMED"
        if m30_val.get("status") == "CONFIRMED" or m30_cond.get("status") == "CONFIRMED"
        else m30_cond.get("status", "CONFIRMED")
    )
    st["M30"] = {
        "ok": True,
        "status": m30_effective_status,
        "condition": m30_cond.get("condition", "ALIGNED"),
        "msg": m30_cond.get("msg") or m30_val.get("msg"),
    }

    dec["focus_selection_evaluated"] = True
    ref_atr_dec = (scan.get("atr_diagnostics") or {}).get("atr_value")
    if ref_atr_dec is None and isinstance(frames, dict):
        ref_atr_dec = extract_atr_diagnostics(frames, preferred_tf="H1").get("atr_value")
    h1_aligned_dec = (ctx.get("H1") == h4)

    # Kumpulkan kandidat searah dari active_zones dan radar_zones (agar FAR tetap tercatat sebagai referensi tanpa masuk active)
    cands = [z for z in (scan.get("active_zones") or []) if z.get("zone_type") == ztype]
    seen_cids = {str(z.get("zone_id") or "") for z in cands}
    for rz_c in (scan.get("radar_zones") or []):
        if isinstance(rz_c, dict) and rz_c.get("zone_type") == ztype and str(rz_c.get("zone_id") or "") not in seen_cids:
            if rz_c.get("is_valid", True):
                cands.append(rz_c)
                seen_cids.add(str(rz_c.get("zone_id") or ""))

    def _cand_sort(z: Dict[str, Any]) -> Tuple:
        rel_is_active = str(z.get("zone_relevance") or "").upper() == "ACTIVE"
        qual_ok = (
            z.get("grade") in ("A+", "A")
            or (float(z.get("score", 0.0) or 0.0) >= 70.0 and str(z.get("freshness") or "").upper() in ("FRESH", "TESTED_1X"))
        )
        bucket = 0 if (rel_is_active and qual_ok) else (1 if rel_is_active else (2 if qual_ok else 3))
        rank = {"A+": 0, "A": 1, "B": 2}
        tf_rank = _intraday_tf_rank(str(z.get("timeframe", "")), m30_status=m30_effective_status, h1_aligned=h1_aligned_dec)
        fresh_rank = {"FRESH": 0, "TESTED_1X": 1, "TESTED_2X": 2}.get(z.get("freshness", ""), 9)
        return (
            bucket,
            fresh_rank,
            round(float(z.get("distance", 9999.0) or 9999.0), 2),
            tf_rank,
            rank.get(z.get("grade", ""), 9),
            -round(float(z.get("score", 0.0) or 0.0), 1),
            -round(float(z.get("displacement_strength", 0.0) or 0.0), 2),
            str(z.get("zone_id", "")),
        )

    def _post_bos_closes(df: pd.DataFrame, z_dict: Dict[str, Any]) -> pd.Series:
        raw_bos = z_dict.get("bos_candle_time")
        if not raw_bos or not isinstance(df.index, pd.DatetimeIndex):
            return df["Close"]
        try:
            bos_t = pd.Timestamp(raw_bos)
        except Exception:
            return df["Close"]
        if pd.isna(bos_t):
            return df["Close"]
        bos_tf = str(z_dict.get("bos_timeframe") or z_dict.get("timeframe") or "").upper()
        bos_min = TF_MINUTES.get(bos_tf, 0) or max(
            (TF_MINUTES.get(p.strip(), 0) for p in bos_tf.split("+") if p.strip()),
            default=0,
        )
        df_tz = df.index.tz
        if df_tz is not None and bos_t.tzinfo is None:
            origin_df = frames.get(bos_tf) if isinstance(frames, dict) else None
            tz_hint = str((getattr(origin_df, "attrs", {}) or {}).get("time_tz") or (getattr(df, "attrs", {}) or {}).get("time_tz", "WIB"))
            bos_t = bos_t.tz_localize("UTC" if tz_hint.upper() == "UTC" else "Asia/Jakarta").tz_convert(df_tz)
        elif df_tz is None and bos_t.tzinfo is not None:
            df_tz_hint = str((getattr(df, "attrs", {}) or {}).get("time_tz", "WIB")).upper()
            bos_t = bos_t.tz_convert("UTC" if df_tz_hint == "UTC" else "Asia/Jakarta").tz_localize(None)
        if bos_min > 0:
            return df.loc[df.index >= (bos_t + timedelta(minutes=bos_min)), "Close"]
        return df.loc[df.index > bos_t, "Close"]

    best_candidate_dict = None
    best_candidate_zone = None
    eligible: List[Dict[str, Any]] = []
    tracker = get_zone_tracker(auto_save=False)
    inval_persisted_dirty = False

    def _is_far_zone_cand(z):
        if not isinstance(z, dict):
            return False
        if is_far_reference_zone(z, current_price=float(price), atr=ref_atr_dec):
            return True
        rel_c = classify_zone_relevance(
            z, float(price), atr=ref_atr_dec,
            htf_bias=h4, h1_structure=ctx.get("H1", "NETRAL"),
            m30_status=m30_effective_status, frames=frames,
            bid=scan.get("bid"), ask=scan.get("ask"),
        )
        if rel_c["relevance"] == "FAR_REFERENCE":
            return True
        if str(z.get("radar_eligibility") or "").upper() == "NOT_QUALIFIED" and "DUPLICATE_OVERLAP" not in str(z.get("radar_rejection_reason") or ""):
            return True
        return False

    if cands:
        now = to_wib_datetime(server_time) or get_current_wib()

        def _focus_eligible(z):
            nonlocal inval_persisted_dirty
            if z.get("is_valid") is False or z.get("is_broken") is True:
                return False
            z_st = str(z.get("lifecycle_status") or z.get("status") or "").upper()
            z_fr = str(z.get("freshness") or "").upper()
            if z_st in ("INVALIDATED", "BROKEN", "EXPIRED", "MITIGATED") or z_fr in ("INVALIDATED", "BROKEN", "EXPIRED", "MITIGATED"):
                return False
            pers_inv_f, _ = tracker.is_zone_invalidated(z) if tracker else (False, None)
            if pers_inv_f:
                return False
            closed_inv_f, closed_rsn_f, b_px_f, b_tm_f, b_tf_f = evaluate_smc_zone_closed_invalidation(z, frames)
            live_inv_f, live_rsn_f, live_px_f, live_inv_lvl_f, _ = evaluate_smc_setup_live_invalidation(
                z, current_price=float(price), bid=scan.get("bid"), ask=scan.get("ask"),
                frames=frames, spread=float(spread or 0.0), market_status=eff_mkt_dec,
            )
            if live_inv_lvl_f and live_inv_lvl_f > 0:
                z["setup_invalidation_price"] = live_inv_lvl_f
            if closed_inv_f or live_inv_f:
                inv_kind_f = INVALIDATION_ZONE_STRUCTURAL if closed_inv_f else INVALIDATION_SETUP_SL_BREACHED
                inv_rsn_f = closed_rsn_f if closed_inv_f else live_rsn_f
                z["is_valid"] = False
                z["is_broken"] = (inv_kind_f == INVALIDATION_ZONE_STRUCTURAL)
                z["status"] = "INVALIDATED"
                z["lifecycle_status"] = "INVALIDATED"
                z["freshness"] = "INVALIDATED"
                z["zone_relevance"] = "INVALID"
                z["invalidation_type"] = inv_kind_f
                z["invalid_label"] = inv_kind_f
                z["rejection_reason"] = inv_rsn_f
                if tracker:
                    tracker.mark_invalidated(
                        z, inv_rsn_f,
                        break_close=b_px_f if closed_inv_f else None,
                        break_time=b_tm_f if closed_inv_f else dec.get("snapshot_time"),
                        origin_tf=b_tf_f if closed_inv_f else z.get("timeframe"),
                        invalidation_type=inv_kind_f,
                        break_price=live_px_f if (live_inv_f and not closed_inv_f) else b_px_f,
                    )
                    inval_persisted_dirty = True
                return False
            # Klasifikasi Distance-Aware Relevance SEBELUM sorting & pemilihan zona utama
            rel_f = classify_zone_relevance(
                z, float(price), atr=ref_atr_dec,
                htf_bias=h4, h1_structure=ctx.get("H1", "NETRAL"),
                m30_status=m30_effective_status, frames=frames,
                bid=scan.get("bid"), ask=scan.get("ask"),
            )
            z["zone_relevance"] = rel_f["relevance"]
            z["zone_relevance_reason"] = rel_f["reason"]
            z["distance"] = rel_f["distance"]
            z["distance_atr_ratio"] = rel_f["distance_atr_ratio"]
            if rel_f["relevance"] == "FAR_REFERENCE":
                z["risk_tier"] = "FAR_ZONE"
                z["risk_tier_reason"] = rel_f["reason"]
            return rel_f["relevance"] != "INVALID"

        def _is_zone_risk_rejected(z_cand: Dict[str, Any]) -> bool:
            if not isinstance(z_cand, dict):
                return True
            if str(z_cand.get("grade") or "").upper() == "REJECTED" or z_cand.get("is_valid") is False:
                return True
            if z_cand.get("risk_rejected") is True:
                return True
            p_rsn_c = str(z_cand.get("pending_rejection_reason") or z_cand.get("setup_eligibility") or "").lower()
            if "jarak sl" in p_rsn_c and "melebihi batas" in p_rsn_c:
                z_cand["risk_rejected"] = True
                return True
            try:
                zn_c = _zone_from_dict(z_cand)
                z_lo_c = float(z_cand.get("price_low", 0.0))
                z_hi_c = float(z_cand.get("price_high", 0.0))
                dem_c = (str(z_cand.get("zone_type") or "").upper() == "DEMAND")
                e_prox = z_hi_c if dem_c else z_lo_c
                sl_c = compute_structural_sl(zn_c, frames, float(spread or 0.0))
                if sl_c > 0 and e_prox > 0:
                    r_px_c = round((e_prox - sl_c) if dem_c else (sl_c - e_prox), 2)
                    if r_px_c > SL_JARAK_MAKS:
                        sl_rej_txt = f"Jarak SL ({r_px_c:.2f} USD) melebihi batas toleransi maksimal ({SL_JARAK_MAKS:g} USD)"
                        z_cand["risk_rejected"] = True
                        z_cand["pending_eligibility"] = "NOT_QUALIFIED"
                        z_cand["pending_rejection_reason"] = sl_rej_txt
                        z_cand["setup_eligibility"] = f"NOT_ELIGIBLE ({sl_rej_txt})"
                        return True
            except Exception:
                pass
            return False

        eligible = [z for z in cands if _focus_eligible(z)]
        for _ez in eligible:
            _is_zone_risk_rejected(_ez)
        # Filter jarak ATR & TF asal SEBELUM pemilihan Primary Zone / Active Focus
        in_range_raw = [z for z in eligible if z.get("zone_relevance") in ("ACTIVE", "RADAR") and not _is_far_zone_cand(z)]
        in_range_raw.sort(key=lambda z_s: (1 if z_s.get("risk_rejected") else 0, _cand_sort(z_s)))
        in_range_eligible, _ = _deduplicate_overlapping_zones(in_range_raw)
        far_eligible = [z for z in eligible if z.get("zone_relevance") == "FAR_REFERENCE" or _is_far_zone_cand(z)]
        far_eligible.sort(key=lambda z_f: (
            round(float(z_f.get("distance", 9999.0) or 9999.0), 2),
            -round(float(z_f.get("score", 0.0) or 0.0), 1),
            str(z_f.get("zone_id", "")),
        ))
        far_eligible, _ = _deduplicate_overlapping_zones(far_eligible)
        dec["primary_zone"] = in_range_eligible[0] if in_range_eligible else None

        def _distance(z):
            return max(float(z["price_low"]) - price, price - float(z["price_high"]), 0.0)

        if far_eligible:
            dec["far_reference_zone"] = min(far_eligible, key=_distance)

        directional = [
            z for z in in_range_eligible
            if (
                (ztype == "SUPPLY" and float(z["price_high"]) >= price)
                or (ztype == "DEMAND" and float(z["price_low"]) <= price)
            )
        ]
        dec["nearest_valid_zone"] = min(directional, key=_distance) if directional else (in_range_eligible[0] if in_range_eligible else None)
        setup_ready_in_range = [z for z in in_range_eligible if not z.get("risk_rejected")]
        inside = [z for z in setup_ready_in_range if _distance(z) == 0]
        old_focus = next((rec for rec in tracker.tracked_zones.values() if rec.get("monitor_focus")), None)
        previous = next((z for z in setup_ready_in_range if (tracker.get_zone(z.get("zone_id")) or {}).get("monitor_focus")), None)
        if not previous and old_focus and old_focus.get("status") not in ("INVALIDATED", "EXPIRED"):
            expires = to_wib_datetime(old_focus.get("expires_at")) if old_focus.get("expires_at") else None
            old_sources = set(old_focus.get("merged_from") or [old_focus.get("zone_id")])
            matches = [
                z for z in setup_ready_in_range
                if (expires is None or now <= expires)
                and old_focus.get("zone_type") == z.get("zone_type")
                and old_sources.intersection(z.get("merged_from") or [z.get("zone_id")])
                and min(float(old_focus["price_high"]), float(z["price_high"]))
                >= max(float(old_focus["price_low"]), float(z["price_low"]))
            ]
            if len(matches) == 1:
                previous = matches[0]
        atr_approach_pt = float(ZONE_APPROACH_DISTANCE_POINTS)
        nearby = [z for z in setup_ready_in_range if _distance(z) <= atr_approach_pt]
        active_high_qual = [
            z for z in setup_ready_in_range
            if z.get("zone_relevance") == "ACTIVE"
            and (z.get("grade") in ("A+", "A") or (float(z.get("score", 0.0) or 0.0) >= 70.0 and str(z.get("freshness") or "").upper() in ("FRESH", "TESTED_1X")))
        ]
        previous_record = old_focus if previous and old_focus else {}
        awaiting = bool(
            previous
            and previous_record.get("touch_status") in ("IN_ZONE", "TOUCHED_PREVIOUSLY")
        )
        if previous and (awaiting or _distance(previous) == 0):
            best_candidate_dict = previous
        elif inside:
            # Pilih kualitas tertinggi di dalam zona (tanpa memaksakan M5 lemah)
            best_candidate_dict = inside[0]
        elif previous and _distance(previous) <= atr_approach_pt:
            best_candidate_dict = previous
        elif nearby:
            best_candidate_dict = nearby[0]
        elif active_high_qual:
            best_candidate_dict = active_high_qual[0]
        elif not in_range_eligible and far_eligible:
            best_candidate_dict = far_eligible[0]
        remaining = [
            z for z in directional
            if z is not best_candidate_dict
            and (not best_candidate_dict or not _zones_substantially_overlap(best_candidate_dict, z))
        ]
        dec["next_radar_zone"] = min(remaining, key=_distance) if remaining else None
        # Evaluasi entry pada kandidat (fokus in-range jika ada, atau far jika in-range kosong)
        if best_candidate_dict:
            cands = [best_candidate_dict] + [z for z in (in_range_eligible or far_eligible) if z is not best_candidate_dict]
        else:
            cands = in_range_eligible or far_eligible
    if best_candidate_dict:
        best_candidate_zone = _zone_from_dict(best_candidate_dict)
        dec["zone"] = best_candidate_dict

    valid_plan = None
    winning_zone = None
    winning_zdict = None
    first_reasons: List[str] = []
    first_tech_confirmed = False

    if not isinstance(dec.get("snr_context"), dict):
        dec["snr_context"] = extract_snr_supporting_context(
            frames,
            float(price),
            market_status=eff_mkt_dec,
            snapshot=live_snap if isinstance(live_snap, dict) else (scan if isinstance(scan, dict) else None),
        )
    pre_snr_ctx = dec.get("snr_context")

    # Jalur 1: Zona OB+FVG aktif searah tren dalam jangkauan (konfirmasi SMC klasik)
    valid_cands: List[Dict[str, Any]] = []
    newly_broken_cands: List[Dict[str, Any]] = []
    _dec_bid = live_snap.get("bid") if isinstance(live_snap, dict) else (scan.get("bid") if isinstance(scan, dict) else None)
    _dec_ask = live_snap.get("ask") if isinstance(live_snap, dict) else (scan.get("ask") if isinstance(scan, dict) else None)
    for zd in cands:
        zn = _zone_from_dict(zd)
        pers_inv_c, pers_rec_c = tracker.is_zone_invalidated(zd) if tracker else (False, None)
        closed_inv_c, closed_rsn_c, b_px_c, b_tm_c, b_tf_c = evaluate_smc_zone_closed_invalidation(zd, frames)
        live_inv_c, live_rsn_c, live_px_c, live_inv_lvl_c, _ = evaluate_smc_setup_live_invalidation(
            zd,
            current_price=float(price),
            bid=_dec_bid,
            ask=_dec_ask,
            frames=frames,
            spread=float(spread or 0.0),
            market_status=eff_mkt_dec,
        )
        if pers_inv_c or closed_inv_c or live_inv_c:
            inv_kind_c = (
                INVALIDATION_ZONE_STRUCTURAL
                if closed_inv_c
                else (
                    INVALIDATION_SETUP_SL_BREACHED
                    if live_inv_c
                    else str((pers_rec_c or {}).get("invalidation_type") or INVALIDATION_ZONE_STRUCTURAL)
                )
            )
            inv_reason_c = (
                closed_rsn_c
                or live_rsn_c
                or (pers_rec_c.get("reason") if isinstance(pers_rec_c, dict) and pers_rec_c.get("reason") else "")
                or f"{inv_kind_c} (Close {zn.timeframe} menembus invalidasi {zn.invalidation_price:.2f})"
            )
            zd["is_valid"] = False
            zd["is_broken"] = True
            zd["status"] = "INVALIDATED"
            zd["lifecycle_status"] = "INVALIDATED"
            zd["freshness"] = "INVALIDATED"
            zd["zone_relevance"] = "INVALID"
            zd["invalidation_type"] = inv_kind_c
            zd["invalid_label"] = inv_kind_c
            if live_inv_lvl_c and live_inv_lvl_c > 0:
                zd["setup_sl"] = float(live_inv_lvl_c)
                zd["setup_invalidation_price"] = float(live_inv_lvl_c)
            zd["rejection_reason"] = inv_reason_c
            if tracker and (closed_inv_c or live_inv_c) and not pers_inv_c:
                tracker.mark_invalidated(
                    zd,
                    inv_reason_c,
                    break_close=b_px_c if closed_inv_c else live_px_c,
                    break_time=b_tm_c if closed_inv_c else None,
                    origin_tf=b_tf_c if closed_inv_c else zn.timeframe,
                    invalidation_type=inv_kind_c,
                )
                inval_persisted_dirty = True
            newly_broken_cands.append(zd)
            if best_candidate_dict is not None and (
                zd is best_candidate_dict or zd.get("zone_id") == best_candidate_dict.get("zone_id")
            ):
                best_candidate_dict = None
                best_candidate_zone = None
                dec["zone"] = None
            continue

        valid_cands.append(zd)
        if best_candidate_dict is None and zd.get("zone_relevance") == "ACTIVE" and not _is_far_zone_cand(zd) and not zd.get("risk_rejected"):
            best_candidate_dict = zd
            best_candidate_zone = zn
            dec["zone"] = zd

        r = _entry_check(
            zn,
            float(price),
            frames["M5"],
            spread if spread is not None else 0.0,
            news,
            frames,
            server_time=server_time,
            snr_context=pre_snr_ctx,
        )
        if r.get("risk_rejected"):
            zd["risk_rejected"] = True
            zd["setup_eligibility"] = r.get("setup_eligibility") or "NOT_ELIGIBLE (Risk filter)"
            zd["pending_eligibility"] = "NOT_QUALIFIED"
            if r.get("reasons"):
                zd["pending_rejection_reason"] = str(r["reasons"][-1])
            if best_candidate_dict is not None and (zd is best_candidate_dict or zd.get("zone_id") == best_candidate_dict.get("zone_id")):
                best_candidate_dict = None
                best_candidate_zone = None
                dec["zone"] = None
        is_r_tech = bool(r.get("smc_technical_confirmed")) and not _is_far_zone_cand(zd) and not bool(r.get("risk_rejected"))
        if not first_reasons or (is_r_tech and not first_tech_confirmed):
            first_reasons = list(r["reasons"])
            for k_st, v_st in r.get("stages", {}).items():
                st[k_st] = v_st
            if is_r_tech:
                first_tech_confirmed = True
                best_candidate_dict = zd
                best_candidate_zone = zn
                dec["zone"] = zd
                dec["m5"] = dict(r.get("m5") or {})
        if r["status"] == "VALID" and not _is_far_zone_cand(zd):
            valid_plan = dict(r["plan"])
            valid_plan.setdefault("setup_type", "TREND_PULLBACK")
            winning_zone = zn
            winning_zdict = zd
            dec["setup_type"] = "TREND_PULLBACK"
            dec["signal_status"] = SIGNAL_STATUS_SIGNAL_CONFIRMED
            dec["reasons"].extend(r["reasons"])
            for k_st, v_st in r.get("stages", {}).items():
                st[k_st] = v_st
            break

    cands = valid_cands
    if newly_broken_cands:
        broken_ids = {str(bz.get("zone_id") or "") for bz in newly_broken_cands}
        dec["invalidated_zones"] = newly_broken_cands
        if isinstance(dec.get("primary_zone"), dict) and str(dec["primary_zone"].get("zone_id") or "") in broken_ids:
            dec["primary_zone"] = valid_cands[0] if valid_cands else None
        if isinstance(dec.get("nearest_valid_zone"), dict) and str(dec["nearest_valid_zone"].get("zone_id") or "") in broken_ids:
            dec["nearest_valid_zone"] = valid_cands[0] if valid_cands else None

    if valid_plan and winning_zone and winning_zdict:
        if inval_persisted_dirty and tracker:
            tracker.save()
        dec["zone"] = winning_zdict
        dec["plan"] = valid_plan
        st["H4"] = {"ok": True, "msg": f"Bias H4 {h4}; zona {winning_zone.grade} {winning_zone.timeframe} {winning_zone.zone_type} {winning_zone.price_low:.2f}-{winning_zone.price_high:.2f}"}
        st["M5"] = {"ok": True, "msg": "; ".join(dec["reasons"])}
        st["RISK"] = {"ok": True, "msg": f"R:R 1:{valid_plan.get('rr', '-')}"}
        return finish("VALID", "")

    # Jalur 2: Multi-Timeframe Trend Following (Pullback FVG/SNR/OTE, Trend Continuation, Breakout Retest)
    tracker_for_focus = tracker
    prior_tf_focus = next(
        (rec for rec in tracker_for_focus.tracked_zones.values() if rec.get("monitor_focus") and rec.get("monitor_only")),
        None,
    )
    prior_tf_expiry = to_wib_datetime(prior_tf_focus.get("expires_at")) if prior_tf_focus and prior_tf_focus.get("expires_at") else None
    if prior_tf_focus and (
        prior_tf_focus.get("status") in ("INVALIDATED", "EXPIRED")
        or (prior_tf_expiry is not None and (to_wib_datetime(server_time) or get_current_wib()) > prior_tf_expiry)
    ):
        prior_tf_focus = None
    tf_res = evaluate_trend_following_setup(
        frames,
        float(price),
        scan=scan,
        spread=float(spread if spread is not None else 0.0),
        news_blackout=news,
        explicit_h1_conflict=bool(ctx.get("has_conflict")),
        explicit_m30_conflict=bool(ctx.get("m30_conflict")),
        preferred_zone_id=prior_tf_focus.get("zone_id") if prior_tf_focus else None,
    )
    tf_candidate = tf_res.get("best_candidate") or {}
    tf_id = tf_candidate.get("zone_id")
    tf_record = (tracker_for_focus.get_zone(tf_id) or {}) if tf_id else {}
    tf_rec_status = str(tf_record.get("status") or "").upper()
    tf_rec_reason = str(tf_record.get("rejection_reason") or "").strip()
    tf_persisted_inv, tf_persisted_rec = (
        tracker_for_focus.is_zone_invalidated(tf_candidate) if tf_candidate else (False, None)
    )
    tf_closed_inv, tf_closed_rsn, tf_b_px, tf_b_tm, tf_b_tf = (
        evaluate_smc_zone_closed_invalidation(tf_candidate, frames)
        if tf_candidate
        else (False, "", None, None, None)
    )
    tf_live_inv, tf_live_rsn, tf_live_px, tf_live_lvl, _ = (
        evaluate_smc_setup_live_invalidation(
            tf_candidate,
            current_price=float(price),
            bid=_dec_bid,
            ask=_dec_ask,
            frames=frames,
            spread=float(spread or 0.0),
            market_status=eff_mkt_dec,
        )
        if tf_candidate
        else (False, "", None, None, None)
    )
    if tf_candidate and (tf_closed_inv or tf_live_inv):
        tf_inv_kind = INVALIDATION_ZONE_STRUCTURAL if tf_closed_inv else INVALIDATION_SETUP_SL_BREACHED
        tf_candidate["is_valid"] = False
        tf_candidate["is_broken"] = True
        tf_candidate["status"] = "INVALIDATED"
        tf_candidate["invalidation_type"] = tf_inv_kind
        tf_candidate["invalid_label"] = tf_inv_kind
        if tf_live_lvl and tf_live_lvl > 0:
            tf_candidate["setup_sl"] = float(tf_live_lvl)
            tf_candidate["setup_invalidation_price"] = float(tf_live_lvl)
        if not tf_persisted_inv:
            tracker_for_focus.mark_invalidated(
                tf_candidate,
                tf_closed_rsn or tf_live_rsn,
                break_close=tf_b_px if tf_closed_inv else tf_live_px,
                break_time=tf_b_tm if tf_closed_inv else None,
                origin_tf=tf_b_tf if tf_closed_inv else tf_candidate.get("timeframe"),
                invalidation_type=tf_inv_kind,
            )
            inval_persisted_dirty = True
    tf_cand_in_eligible = bool(tf_id and any(z.get("zone_id") == tf_id for z in valid_cands))
    tf_tracker_invalidated = bool(
        tf_rec_status == "INVALIDATED"
        and ("invalidation" in tf_rec_reason.lower() or "broken" in tf_rec_reason.lower() or "sl_breached" in tf_rec_reason.lower())
        and not tf_cand_in_eligible
    )
    tf_blocked = bool(tf_candidate) and (tf_closed_inv or tf_live_inv or tf_persisted_inv or tf_tracker_invalidated)
    tf_block_reason = ""
    if tf_blocked:
        if tf_closed_inv:
            tf_block_reason = f"{INVALIDATION_ZONE_STRUCTURAL} ({tf_id or ''}): {tf_closed_rsn}".strip()
        elif tf_live_inv:
            tf_block_reason = f"{INVALIDATION_SETUP_SL_BREACHED} ({tf_id or ''}): {tf_live_rsn}".strip()
        elif tf_persisted_inv:
            pers_msg = (tf_persisted_rec or {}).get("reason") if isinstance(tf_persisted_rec, dict) else ""
            tf_block_reason = f"Kandidat trend-following {tf_id or ''} INVALID — BROKEN: {pers_msg or 'tercatat invalid pada scan sebelumnya'}".strip()
        elif tf_rec_reason:
            tf_block_reason = f"Kandidat trend-following {tf_id or ''} INVALID — BROKEN: {tf_rec_reason}".strip()
        else:
            tf_block_reason = "Kandidat trend-following sebelumnya INVALID — BROKEN pada batas invalidation."
        tf_res["status"] = "WAIT"
        tf_res["plan"] = None
        tf_res["signal_status"] = SIGNAL_STATUS_NO_SETUP
        tf_res["reasons"] = [tf_block_reason]
    if inval_persisted_dirty and tracker_for_focus:
        tracker_for_focus.save()
    dec["trend_following"] = tf_res
    dec["snr_context"] = tf_res.get("snr") or pre_snr_ctx

    if tf_res.get("status") == "VALID" and tf_res.get("plan"):
        dec["setup_type"] = tf_res.get("setup_type", "NONE")
        dec["signal_status"] = tf_res.get("signal_status", SIGNAL_STATUS_NO_SETUP)
        bc = tf_res["best_candidate"]
        tf_plan = tf_res["plan"]
        winning_zdict = {
            "zone_id": bc.get("zone_id", f"TF_{tf_res['setup_type']}"),
            "symbol": dec.get("symbol", SYMBOL),
            "timeframe": bc.get("timeframe", "M30"),
            "zone_type": bc.get("zone_type", ztype),
            "price_low": float(bc["price_low"]),
            "price_high": float(bc["price_high"]),
            "midpoint": round((float(bc["price_low"]) + float(bc["price_high"])) / 2.0, 2),
            "invalidation_price": float(bc.get("invalidation_price", tf_plan["sl"])),
            "bos_candle_time": bc.get("bos_candle_time") or bc.get("formation_time"),
            "bos_timeframe": bc.get("bos_timeframe") or bc.get("formation_timeframe") or bc.get("timeframe", "M30"),
            "formation_time": bc.get("formation_time"),
            "sl_recommended": float(tf_plan["sl"]),
            "setup_sl": float(tf_plan["sl"]),
            "grade": bc.get("grade", "B"),
            "score": float(bc.get("score", 75.0)),
            "freshness": "FRESH",
            "is_valid": True,
            "setup_type": tf_res["setup_type"],
            "source_kind": bc.get("source_kind"),
            "source_zone_id": bc.get("source_zone_id"),
        }
        winning_zone = _zone_from_dict(winning_zdict)
        valid_plan = tf_plan
        dec["zone"] = winning_zdict
        dec["plan"] = tf_plan
        for k_st, v_st in (tf_res.get("m5", {}).get("stages") or {}).items():
            st[k_st] = v_st
        dec["reasons"].extend(tf_res.get("reasons", []))
        st["H4"] = {"ok": True, "msg": tf_res.get("h4", {}).get("reason") or f"Bias H4 {h4}"}
        st["H1"] = {"ok": True, "msg": tf_res.get("h1", {}).get("reason") or f"H1 {h1_tag}"}
        st["M30"] = {"ok": True, "status": tf_res.get("m30", {}).get("status", m30_effective_status), "msg": tf_res.get("m30", {}).get("msg", "")}
        st["M5"] = {"ok": True, "msg": "; ".join(dec["reasons"])}
        st["RISK"] = {"ok": True, "msg": f"R:R 1:{tf_plan.get('rr', '-')}"}
        return finish("VALID", "")

    # Tidak ada setup VALID pada jalur 1 & 2 -> WAIT transparan dengan stage & alasan lengkap
    tf_m5_res = tf_res.get("m5") or {}
    tf_tech_confirmed = bool(tf_m5_res.get("smc_technical_confirmed"))
    keep_tf_focus = bool(
        prior_tf_focus and not tf_blocked
        and tf_candidate.get("zone_id") == prior_tf_focus.get("zone_id")
        and (
            prior_tf_focus.get("touch_status") in ("IN_ZONE", "TOUCHED_PREVIOUSLY")
            or max(float(tf_candidate["price_low"]) - price, price - float(tf_candidate["price_high"]), 0.0) <= ZONE_APPROACH_DISTANCE_POINTS
        )
    )
    if cands and best_candidate_dict and best_candidate_zone and (first_tech_confirmed or not keep_tf_focus) and not (tf_tech_confirmed and not first_tech_confirmed):
        dec["zone"] = best_candidate_dict
        dec["setup_type"] = "TREND_PULLBACK"
        dec["signal_status"] = (
            SIGNAL_STATUS_WAITING_CONFIRMATION
            if st.get("zone_touch", {}).get("status") == "CONFIRMED"
            else SIGNAL_STATUS_WAITING_PULLBACK
        )
        dec["reasons"].extend(first_reasons)
        st["H4"] = {"ok": True, "msg": f"Bias H4 {h4}; zona {best_candidate_zone.grade} {best_candidate_zone.timeframe} {best_candidate_zone.zone_type} {best_candidate_zone.price_low:.2f}-{best_candidate_zone.price_high:.2f}"}
        st["M5"] = {"ok": False, "msg": "; ".join(first_reasons)}
    else:
        dec["setup_type"] = tf_res.get("setup_type", "NONE")
        dec["signal_status"] = tf_res.get("signal_status", SIGNAL_STATUS_NO_SETUP)
        tf_reasons = tf_res.get("reasons") or []
        wait_msg = tf_reasons[0] if tf_reasons else (scan.get("empty_reason") or f"Menunggu setup {direction} valid")
        tf_touch = (tf_m5_res.get("stages") or {}).get("zone_touch") or {}
        tf_monitorable = bool(
            tf_candidate and not tf_blocked and not _is_far_zone_cand(tf_candidate) and (
                keep_tf_focus
                or tf_tech_confirmed
                or tf_touch.get("status") == "CONFIRMED"
                or max(float(tf_candidate["price_low"]) - price, price - float(tf_candidate["price_high"]), 0.0) <= ZONE_APPROACH_DISTANCE_POINTS
            )
        )
        if tf_monitorable:
            bc = tf_res["best_candidate"]
            bc_zt = str(bc.get("zone_type", ztype)).upper()
            bc_def_inval = bc["price_low"] if bc_zt == "DEMAND" else bc["price_high"]
            dec["zone"] = {
                "zone_id": bc.get("zone_id"),
                "timeframe": bc.get("timeframe", "M30"),
                "zone_type": bc_zt,
                "price_low": float(bc["price_low"]),
                "price_high": float(bc["price_high"]),
                "grade": bc.get("grade", "B"),
                "score": float(bc.get("score", 70.0)),
                "invalidation_price": float(bc.get("invalidation_price", bc_def_inval)),
                "bos_candle_time": bc.get("bos_candle_time") or bc.get("formation_time"),
                "bos_timeframe": bc.get("bos_timeframe") or bc.get("formation_timeframe") or bc.get("timeframe", "M30"),
                "formation_time": bc.get("formation_time"),
                "is_valid": True,
                "setup_type": bc.get("setup_type"),
                "source_kind": bc.get("source_kind"),
                "source_zone_id": bc.get("source_zone_id"),
            }
        else:
            dec["zone"] = None
            dec["setup_type"] = "NONE"
            dec["signal_status"] = SIGNAL_STATUS_WAITING_PULLBACK
            nz_ref = dec.get("nearest_valid_zone") or dec.get("primary_zone")
            if nz_ref and nz_ref.get("risk_rejected"):
                nz_lo = float(nz_ref.get("price_low", 0.0))
                nz_hi = float(nz_ref.get("price_high", 0.0))
                r_rej_txt = nz_ref.get("pending_rejection_reason") or (first_reasons[-1] if first_reasons else f"Jarak SL melebihi batas maksimal {SL_JARAK_MAKS:g} USD")
                wait_msg = (
                    f"Zona {nz_ref.get('timeframe', '')} {nz_ref.get('zone_type', '')} ({nz_lo:.2f}-{nz_hi:.2f}) "
                    f"valid secara struktur (Grade {nz_ref.get('grade', 'A')} {float(nz_ref.get('score', 0.0)):.1f} — tetap di RADAR), "
                    f"namun ditolak sebagai setup aktif: {r_rej_txt}"
                ).strip()
                tf_reasons = [wait_msg]
            elif nz_ref and (tf_blocked or (tf_candidate and tf_touch.get("status") in ("WAIT", "FAIL")) or not tf_candidate):
                nz_lo = float(nz_ref.get("price_low", 0.0))
                nz_hi = float(nz_ref.get("price_high", 0.0))
                nz_dist = max(nz_lo - float(price), float(price) - nz_hi, 0.0)
                if nz_ref.get("risk_tier") == "FAR_ZONE":
                    wait_msg = (
                        f"Zona {nz_ref.get('timeframe', '')} {nz_ref.get('zone_type', '')} "
                        f"({nz_lo:.2f}-{nz_hi:.2f}) berjarak {format_price_distance(nz_dist, include_mt5_points=True)} (TOO FAR dari batas intraday); menunggu pullback dan konfirmasi M5."
                    ).strip()
                else:
                    wait_msg = (
                        f"Harga belum mendekati zona radar {nz_ref.get('timeframe', '')} {nz_ref.get('zone_type', '')} "
                        f"({nz_lo:.2f}-{nz_hi:.2f}, jarak {format_price_distance(nz_dist, include_mt5_points=True)}); menunggu pullback dan konfirmasi M5."
                    ).strip()
                tf_reasons = [wait_msg]
            elif not nz_ref:
                far_aligned = [
                    z for z in (scan.get("radar_zones") or [])
                    if isinstance(z, dict)
                    and z.get("is_valid", True)
                    and z.get("zone_type") == ztype
                    and z.get("risk_tier") == "FAR_ZONE"
                    and str(z.get("status", "")).upper() not in ("INVALIDATED", "EXPIRED", "MITIGATED")
                    and str(z.get("freshness", "")).upper() not in ("INVALIDATED", "EXPIRED", "MITIGATED")
                ]
                if far_aligned:
                    fz_ref = min(
                        far_aligned,
                        key=lambda z: max(float(z.get("price_low", 0.0)) - float(price), float(price) - float(z.get("price_high", 0.0)), 0.0),
                    )
                    dec.setdefault("far_reference_zone", fz_ref)
                    fz_lo = float(fz_ref.get("price_low", 0.0))
                    fz_hi = float(fz_ref.get("price_high", 0.0))
                    fz_dist = max(fz_lo - float(price), float(price) - fz_hi, 0.0)
                    wait_msg = (
                        f"Tidak ada zona radar intraday aktif; zona valid searah terdekat {fz_ref.get('timeframe', '')} {fz_ref.get('zone_type', '')} "
                        f"({fz_lo:.2f}-{fz_hi:.2f}) berjarak {format_price_distance(fz_dist, include_mt5_points=True)} (TOO FAR); menunggu pullback."
                    ).strip()
                    tf_reasons = [wait_msg]
                elif tf_blocked:
                    wait_msg = tf_block_reason
                    tf_reasons = [wait_msg]
                elif tf_candidate and tf_touch.get("status") in ("WAIT", "FAIL"):
                    wait_msg = "WAIT_ZONE_APPROACH: Belum ada zona aktif yang berinteraksi dengan harga."
                    tf_reasons = [wait_msg]
                elif not tf_reasons:
                    wait_msg = f"Tidak ada zona {ztype} aktif yang memenuhi syarat saat ini."
                    tf_reasons = [wait_msg]
            elif tf_blocked:
                wait_msg = tf_block_reason
                tf_reasons = [wait_msg]
            elif tf_candidate and not tf_blocked and tf_touch.get("status") in ("WAIT", "FAIL"):
                wait_msg = "WAIT_ZONE_APPROACH: Belum ada zona aktif yang berinteraksi dengan harga."
                tf_reasons = [wait_msg]
        if not first_tech_confirmed or tf_tech_confirmed:
            tf_m5_st = (tf_m5_res.get("stages") or {}) if tf_monitorable else {}
            for component in ("zone_validation", "zone_touch", "liquidity_sweep", "structure_trigger", "displacement", "entry_validation", "risk_reward"):
                st.pop(component, None)
            for k_st, v_st in tf_m5_st.items():
                st[k_st] = v_st
        dec["reasons"].extend(tf_reasons if tf_reasons else [wait_msg])
        st["H4"] = {"ok": True, "msg": f"Bias H4 {h4}"}
        st["M5"] = {"ok": False, "msg": wait_msg}

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


# =============================================================================
# COMPATIBILITY RE-EXPORT: formatter dipindahkan ke modules/zone_formatter.py
# (harus berada di akhir file karena zone_formatter mengimpor nama dari modul ini)
# =============================================================================
from modules.zone_formatter import (  # noqa: E402
    _PLAN_NOTES,
    _extract_risk_monitor_snapshot,
    _format_wib_summary_time,
    _norm_stage_label,
    _num,
    build_smc_invalidation_reason,
    build_smc_zone_reason,
    format_decision_text,
    format_diagnostic_report,
    format_risk_alert,
    format_scan_detail,
    format_scan_presentation,
    format_scan_summary,
    format_scanner_audit,
    format_smc_invalid_why,
    format_smc_zone_lines,
    format_smc_zone_plan,
    format_zone_evidence,
    premarket_lines,
    smc_zone_summary,
)
