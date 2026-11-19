# -*- coding: utf-8 -*-
"""
trend_following.py — Engine Multi-Timeframe Trend Following + SMC Confirmation (XAUUSD).

Hierarki Peran Timeframe:
  - H4  : Primary Trend (BULLISH / BEARISH / UNCONFIRMED) berdasarkan HH/HL, LH/LL, BOS/CHoCH.
  - H1  : Trend Confirmation (ALIGNED / PULLBACK / CONFLICT / UNCONFIRMED).
  - M30 : Setup Detection (TREND_PULLBACK, TREND_CONTINUATION, BREAKOUT_RETEST, REVERSAL,
          serta membedakan pullback normal vs konflik struktural serius vs kondisi lemah datar).
  - M5  : Entry Confirmation berbasis CLOSED CANDLES (Sweep/CHoCH/BOS/CISD/Displacement/Rejection/Retest).

Prinsip:
  - Tidak pernah memutasi snapshot/dataframe input (copy-on-write untuk indikator).
  - Data stale / invalid / market closed / candle running -> selalu fail-safe WAIT (tidak ada sinyal).
  - R:R minimum >= RR_MIN (1:1.5), spread & SL dalam batas konfigurasi.
  - Terintegrasi read-only dengan SNR Advanced untuk level breakout/retest & rintangan S&R,
    tanpa memaksa hasil SMC & SNR menjadi identik.
"""

from __future__ import annotations

import hashlib
import math
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from modules.config import (
    ENTRY_JARAK_MAKS,
    ENTRY_LEBAR_MAKS,
    RR_MIN,
    SIGNAL_STATUS_NO_SETUP,
    SIGNAL_STATUS_SETUP_DETECTED,
    SIGNAL_STATUS_SIGNAL_CONFIRMED,
    SIGNAL_STATUS_WAITING_CONFIRMATION,
    SIGNAL_STATUS_WAITING_PULLBACK,
    SL_JARAK_MAKS,
    SPREAD_MAKS_POIN,
    SYMBOL,
    TF_BREAKOUT_MIN_BODY_ATR,
    TF_CONTINUATION_MIN_DISPLACEMENT_ATR,
    TF_OPPOSING_IMPULSE_BLOCK_ATR,
    TF_PULLBACK_MAX_RETRACE,
    TF_PULLBACK_MIN_RETRACE,
)
from modules.mt5_conn import tambah_indikator
from modules.utils import format_wib_timestamp, to_wib_datetime

_SWING_N = 5
_M5_EVENT_MAX_AGE = 12
_M5_TOUCH_LOOKBACK = 18
_M5_MIN_DISP_ATR = 0.80


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        v = float(val)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _ensure_indicators(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Pastikan dataframe memiliki kolom indikator tanpa memutasi dataframe asli."""
    if df is None or not isinstance(df, pd.DataFrame) or len(df) == 0:
        return None
    need_cols = ("ATR14", "EMA20", "EMA50", "RSI14")
    if all(c in df.columns for c in need_cols):
        return df
    try:
        return tambah_indikator(df.copy())
    except Exception:
        return df.copy()


def _detect_swings(df: Optional[pd.DataFrame], n: int = _SWING_N) -> List[Dict[str, Any]]:
    """Swing High/Low terkonfirmasi tanpa look-ahead bias (konfirmasi pada bar i + n)."""
    if df is None or len(df) < 2 * n + 1:
        return []
    H = df["High"].to_numpy(dtype=float)
    L = df["Low"].to_numpy(dtype=float)
    idx = df.index
    out: List[Dict[str, Any]] = []
    for i in range(n, len(df) - n):
        if H[i] == np.max(H[i - n : i + n + 1]):
            out.append({"idx": i, "type": "H", "price": float(H[i]), "time": idx[i], "confirmed_at": i + n})
        if L[i] == np.min(L[i - n : i + n + 1]):
            out.append({"idx": i, "type": "L", "price": float(L[i]), "time": idx[i], "confirmed_at": i + n})
    out.sort(key=lambda s: s["idx"])
    return out


def _detect_structure(df: Optional[pd.DataFrame], n: int = _SWING_N) -> Tuple[List[Dict[str, Any]], int, List[Dict[str, Any]]]:
    """BOS / CHoCH berbasis CLOSE candle terkonfirmasi (wick tidak dihitung)."""
    if df is None or len(df) < 2 * n + 1:
        return [], 0, []
    swings = _detect_swings(df, n=n)
    C = df["Close"].to_numpy(dtype=float)
    idx = df.index
    conf: Dict[int, List[Dict[str, Any]]] = {}
    for s in swings:
        conf.setdefault(s["confirmed_at"], []).append(s)
    last_high = last_low = None
    ref_high = ref_low = None
    trend = 0
    events: List[Dict[str, Any]] = []
    for k in range(len(df)):
        for s in conf.get(k, []):
            if s["type"] == "H":
                last_high = ref_high = s
            else:
                last_low = ref_low = s
        if last_high is not None and C[k] > last_high["price"]:
            events.append({
                "k": k,
                "time": idx[k],
                "tipe": "CHoCH" if trend == -1 else "BOS",
                "arah": "bull",
                "level": float(last_high["price"]),
                "origin_idx": ref_low["idx"] if ref_low else max(0, k - 12),
                "origin_valid": ref_low is not None,
                "swing_idx": last_high["idx"],
                "swing_time": last_high["time"],
            })
            last_high, trend = None, 1
        elif last_low is not None and C[k] < last_low["price"]:
            events.append({
                "k": k,
                "time": idx[k],
                "tipe": "CHoCH" if trend == 1 else "BOS",
                "arah": "bear",
                "level": float(last_low["price"]),
                "origin_idx": ref_high["idx"] if ref_high else max(0, k - 12),
                "origin_valid": ref_high is not None,
                "swing_idx": last_low["idx"],
                "swing_time": last_low["time"],
            })
            last_low, trend = None, -1
    return events, trend, swings


def _detect_sweeps(df: pd.DataFrame, swings: List[Dict[str, Any]], lookback: int = 40) -> List[Dict[str, Any]]:
    """Deteksi liquidity sweep M5/M30 yang valid (wick menembus swing belum tersapu, close kembali ke dalam)."""
    if df is None or len(df) < 2 * _SWING_N + 1 or not swings:
        return []
    O = df["Open"].to_numpy(dtype=float)
    H = df["High"].to_numpy(dtype=float)
    L = df["Low"].to_numpy(dtype=float)
    C = df["Close"].to_numpy(dtype=float)
    d_ind = _ensure_indicators(df)
    atr = d_ind["ATR14"].to_numpy(dtype=float) if d_ind is not None and "ATR14" in d_ind.columns else np.full(len(df), 1.5)
    idx = df.index
    srt = sorted(swings, key=lambda s: s["confirmed_at"])
    start_k = max(0, len(df) - lookback) if lookback else 0
    res: List[Dict[str, Any]] = []
    ptr = 0
    act: Dict[str, List[Dict[str, Any]]] = {"H": [], "L": []}

    for k in range(len(df)):
        while ptr < len(srt) and srt[ptr]["confirmed_at"] < k:
            act[srt[ptr]["type"]].append(srt[ptr])
            ptr += 1
        if k < start_k:
            continue
        a = atr[k] if np.isfinite(atr[k]) and atr[k] > 0 else 1.5
        rng = H[k] - L[k]
        if rng <= 0:
            continue
        for s in act["H"][-6:]:
            p = s["price"]
            if H[k] > p and C[k] <= p and (H[k] - p) >= 0.05 * a:
                if k > s["idx"] + 1 and np.max(H[s["idx"] + 1 : k]) > p:
                    continue
                if (H[k] - max(O[k], C[k])) / rng < 0.30:
                    continue
                if k + 1 < len(df) and C[k + 1] > H[k]:
                    continue
                res.append({
                    "k": k, "time": idx[k], "type": "SWEEP_HIGH", "level": float(p),
                    "wick_extreme": float(H[k]), "close_price": float(C[k]),
                    "description": f"Buy-Side Sweep @ {p:.2f} (wick {H[k]:.2f}, close {C[k]:.2f})",
                })
        for s in act["L"][-6:]:
            p = s["price"]
            if L[k] < p and C[k] >= p and (p - L[k]) >= 0.05 * a:
                if k > s["idx"] + 1 and np.min(L[s["idx"] + 1 : k]) < p:
                    continue
                if (min(O[k], C[k]) - L[k]) / rng < 0.30:
                    continue
                if k + 1 < len(df) and C[k + 1] < L[k]:
                    continue
                res.append({
                    "k": k, "time": idx[k], "type": "SWEEP_LOW", "level": float(p),
                    "wick_extreme": float(L[k]), "close_price": float(C[k]),
                    "description": f"Sell-Side Sweep @ {p:.2f} (wick {L[k]:.2f}, close {C[k]:.2f})",
                })
    return res


def _detect_fvgs(df: Optional[pd.DataFrame], tf_name: str, min_atr_ratio: float = 0.20) -> List[Dict[str, Any]]:
    """Deteksi FVG (Fair Value Gap) yang masih aktif / belum tertutup penuh."""
    if df is None or len(df) < 10:
        return []
    d = _ensure_indicators(df)
    if d is None:
        return []
    H = d["High"].to_numpy(dtype=float)
    L = d["Low"].to_numpy(dtype=float)
    C = d["Close"].to_numpy(dtype=float)
    atr_arr = d["ATR14"].to_numpy(dtype=float) if "ATR14" in d.columns else np.full(len(d), 2.0)
    idx = d.index
    n = len(d)
    fvgs: List[Dict[str, Any]] = []
    start = max(2, n - 45)
    for i in range(start, n):
        atr_i = atr_arr[i] if np.isfinite(atr_arr[i]) and atr_arr[i] > 0 else 2.0
        # Bullish FVG: Low[i] > High[i-2]
        if L[i] > H[i - 2]:
            lo, hi = float(H[i - 2]), float(L[i])
            if (hi - lo) >= min_atr_ratio * atr_i:
                after_min = float(np.min(L[i + 1 :])) if i + 1 < n else hi
                after_close_min = float(np.min(C[i + 1 :])) if i + 1 < n else hi
                if after_close_min >= lo - 0.15 * atr_i and after_min > lo:
                    status = "UNTOUCHED" if after_min >= hi else "PARTIAL"
                    fvgs.append({
                        "timeframe": tf_name,
                        "direction": "BUY",
                        "zone_type": "DEMAND",
                        "source_kind": "FVG",
                        "low": round(lo, 2),
                        "high": round(hi, 2),
                        "invalidation_price": round(lo - max(0.6, 0.35 * atr_i), 2),
                        "status": status,
                        "k": i,
                        "time": str(idx[i]),
                        "atr": round(atr_i, 3),
                    })
        # Bearish FVG: High[i] < Low[i-2]
        if H[i] < L[i - 2]:
            lo, hi = float(H[i]), float(L[i - 2])
            if (hi - lo) >= min_atr_ratio * atr_i:
                after_max = float(np.max(H[i + 1 :])) if i + 1 < n else lo
                after_close_max = float(np.max(C[i + 1 :])) if i + 1 < n else lo
                if after_close_max <= hi + 0.15 * atr_i and after_max < hi:
                    status = "UNTOUCHED" if after_max <= lo else "PARTIAL"
                    fvgs.append({
                        "timeframe": tf_name,
                        "direction": "SELL",
                        "zone_type": "SUPPLY",
                        "source_kind": "FVG",
                        "low": round(lo, 2),
                        "high": round(hi, 2),
                        "invalidation_price": round(hi + max(0.6, 0.35 * atr_i), 2),
                        "status": status,
                        "k": i,
                        "time": str(idx[i]),
                        "atr": round(atr_i, 3),
                    })
    return fvgs


def _extract_recent_impulse_leg(df: Optional[pd.DataFrame], bias: str) -> Optional[Dict[str, Any]]:
    """Identifikasi leg impuls terukur terkini & rasio retracement harga saat ini."""
    if df is None or len(df) < 2 * _SWING_N + 12 or bias not in ("BULLISH", "BEARISH"):
        return None
    d = _ensure_indicators(df)
    if d is None:
        return None
    _, _, swings = _detect_structure(d, n=_SWING_N)
    if not swings:
        return None
    c_now = float(d["Close"].iloc[-1])
    atr_now = max(0.5, _safe_float(d["ATR14"].iloc[-1] if "ATR14" in d.columns else 2.0, 2.0))
    highs = [s for s in swings if s["type"] == "H"]
    lows = [s for s in swings if s["type"] == "L"]

    if bias == "BULLISH":
        if not highs:
            return None
        sw_h = highs[-1]
        prior_lows = [s for s in lows if s["idx"] < sw_h["idx"]]
        if not prior_lows:
            # fallback window lookback untuk leg impuls satu arah tanpa swing low internal
            win_start = max(0, sw_h["idx"] - 45)
            lo_val = float(d["Low"].iloc[win_start : sw_h["idx"] + 1].min())
            origin_idx = int(win_start)
        else:
            sw_l = prior_lows[-1]
            lo_val = float(sw_l["price"])
            origin_idx = int(sw_l["idx"])
        hi_val = float(sw_h["price"])
        span = hi_val - lo_val
        if span < 1.2 * atr_now:
            return None
        retrace = (hi_val - c_now) / span if span > 0 else 0.0
        return {
            "direction": "BUY",
            "origin_price": round(lo_val, 2),
            "extreme_price": round(hi_val, 2),
            "origin_idx": origin_idx,
            "extreme_idx": int(sw_h["idx"]),
            "span": round(span, 2),
            "span_atr": round(span / atr_now, 2),
            "retrace_ratio": round(retrace, 3),
            "atr": round(atr_now, 3),
        }
    else:
        if not lows:
            return None
        sw_l = lows[-1]
        prior_highs = [s for s in highs if s["idx"] < sw_l["idx"]]
        if not prior_highs:
            win_start = max(0, sw_l["idx"] - 45)
            hi_val = float(d["High"].iloc[win_start : sw_l["idx"] + 1].max())
            origin_idx = int(win_start)
        else:
            sw_h = prior_highs[-1]
            hi_val = float(sw_h["price"])
            origin_idx = int(sw_h["idx"])
        lo_val = float(sw_l["price"])
        span = hi_val - lo_val
        if span < 1.2 * atr_now:
            return None
        retrace = (c_now - lo_val) / span if span > 0 else 0.0
        return {
            "direction": "SELL",
            "origin_price": round(hi_val, 2),
            "extreme_price": round(lo_val, 2),
            "origin_idx": origin_idx,
            "extreme_idx": int(sw_l["idx"]),
            "span": round(span, 2),
            "span_atr": round(span / atr_now, 2),
            "retrace_ratio": round(retrace, 3),
            "atr": round(atr_now, 3),
        }


def _detect_rejection_candle(df: pd.DataFrame, direction: str, lo: float, hi: float, atr: float, lookback: int = 5) -> Optional[Dict[str, Any]]:
    """Deteksi candle rejection closed M5 valid (Pin Bar / Hammer / Engulfing) pada rentang [lo, hi]."""
    if df is None or len(df) < 3:
        return None
    O = df["Open"].to_numpy(dtype=float)
    H = df["High"].to_numpy(dtype=float)
    L = df["Low"].to_numpy(dtype=float)
    C = df["Close"].to_numpy(dtype=float)
    n = len(df)
    tol = max(0.6, 0.45 * atr)
    start = max(1, n - lookback)

    for k in range(n - 1, start - 1, -1):
        rng = H[k] - L[k]
        if rng <= 0:
            continue
        body = abs(C[k] - O[k])
        if direction == "BUY":
            touches = L[k] <= hi + tol and H[k] >= lo - tol
            if not touches:
                continue
            lower_wick = min(O[k], C[k]) - L[k]
            pinbar = lower_wick >= 0.45 * rng and C[k] >= O[k] - 0.15 * atr and C[k] >= lo
            engulf = (C[k] > O[k] and C[k - 1] < O[k - 1] and C[k] > O[k - 1] and body >= 0.55 * atr)
            strong_close = (C[k] > O[k] and body >= 0.65 * atr and C[k] >= (lo + hi) * 0.5)
            if pinbar or engulf or strong_close:
                lbl = "Bullish Engulfing" if engulf else ("Bullish Pin Bar" if pinbar else "Bullish Rejection Close")
                return {"k": k, "label": lbl, "wick_extreme": float(L[k]), "close": float(C[k])}
        else:
            touches = H[k] >= lo - tol and L[k] <= hi + tol
            if not touches:
                continue
            upper_wick = H[k] - max(O[k], C[k])
            pinbar = upper_wick >= 0.45 * rng and C[k] <= O[k] + 0.15 * atr and C[k] <= hi
            engulf = (C[k] < O[k] and C[k - 1] > O[k - 1] and C[k] < O[k - 1] and body >= 0.55 * atr)
            strong_close = (C[k] < O[k] and body >= 0.65 * atr and C[k] <= (lo + hi) * 0.5)
            if pinbar or engulf or strong_close:
                lbl = "Bearish Engulfing" if engulf else ("Bearish Pin Bar" if pinbar else "Bearish Rejection Close")
                return {"k": k, "label": lbl, "wick_extreme": float(H[k]), "close": float(C[k])}
    return None


def _is_sustained_opposing_trend(df: Optional[pd.DataFrame], target_dir: str) -> Tuple[bool, str]:
    """
    Bedakan konflik struktural serius (multi-swing trend berlawanan + EMA stack + break)
    dari sekadar pullback / konsolidasi ringan.
    """
    if df is None or len(df) < 2 * _SWING_N + 12:
        return False, "Data belum cukup"
    d = _ensure_indicators(df)
    if d is None:
        return False, "Indikator tidak tersedia"
    events, trend, swings = _detect_structure(d, n=_SWING_N)
    last = d.iloc[-1]
    c_now = float(last["Close"])
    ema20 = _safe_float(last.get("EMA20"), c_now)
    ema50 = _safe_float(last.get("EMA50"), c_now)
    rsi = _safe_float(last.get("RSI14"), 50.0)

    opp_ev_dir = "bear" if target_dir == "BULLISH" else "bull"
    recent_opp_events = [e for e in events[-5:] if e.get("arah") == opp_ev_dir]

    highs = [s for s in swings if s["type"] == "H"]
    lows = [s for s in swings if s["type"] == "L"]

    if target_dir == "BULLISH":
        seq_opp = (len(highs) >= 2 and highs[-1]["price"] < highs[-2]["price"] and
                   len(lows) >= 2 and lows[-1]["price"] < lows[-2]["price"])
        ema_opp = (c_now < ema20 and ema20 < ema50 and rsi < 44.0)
        latest_aligned = bool(events and events[-1].get("arah") == "bull")
        if latest_aligned and not ema_opp:
            return False, "Struktur terbaru sudah kembali searah BULLISH"
        if (len(recent_opp_events) >= 2 and seq_opp) or (seq_opp and ema_opp) or (len(recent_opp_events) >= 3 and ema_opp):
            return True, f"Konflik struktural serius: LH/LL + {len(recent_opp_events)}x Bearish BOS + EMA20<EMA50 (RSI {rsi:.1f})"
    else:
        seq_opp = (len(highs) >= 2 and highs[-1]["price"] > highs[-2]["price"] and
                   len(lows) >= 2 and lows[-1]["price"] > lows[-2]["price"])
        ema_opp = (c_now > ema20 and ema20 > ema50 and rsi > 56.0)
        latest_aligned = bool(events and events[-1].get("arah") == "bear")
        if latest_aligned and not ema_opp:
            return False, "Struktur terbaru sudah kembali searah BEARISH"
        if (len(recent_opp_events) >= 2 and seq_opp) or (seq_opp and ema_opp) or (len(recent_opp_events) >= 3 and ema_opp):
            return True, f"Konflik struktural serius: HH/HL + {len(recent_opp_events)}x Bullish BOS + EMA20>EMA50 (RSI {rsi:.1f})"

    return False, "Tidak ada konflik struktural berkelanjutan"


# =============================================================================
# EVALUASI PERAN TIMEFRAME: H4 -> H1 -> M30
# =============================================================================
def evaluate_h4_primary_trend(df_h4: Optional[pd.DataFrame], regime_h4: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """H4 — Primary Trend Direction."""
    if regime_h4 and isinstance(regime_h4, dict):
        bias = str(regime_h4.get("bias") or "UNCONFIRMED").upper()
        if bias in ("NETRAL", "NO_DATA", "UNKNOWN", ""):
            bias = "UNCONFIRMED"
        reg = str(regime_h4.get("regime") or "UNKNOWN").upper()
        act = regime_h4.get("active_structure") or {}
        imp = _extract_recent_impulse_leg(df_h4, bias) if bias in ("BULLISH", "BEARISH") else None
        return {
            "bias": bias,
            "ok": bias in ("BULLISH", "BEARISH") and reg != "CHOPPY",
            "regime": reg,
            "active_high": act.get("active_high"),
            "active_low": act.get("active_low"),
            "latest_event": act.get("latest_event"),
            "impulse_leg": imp,
            "reason": regime_h4.get("diagnostic_reason") or regime_h4.get("detail") or f"H4 {bias}",
        }

    if df_h4 is None or len(df_h4) < 2 * _SWING_N + 20:
        return {"bias": "UNCONFIRMED", "ok": False, "regime": "NO_DATA", "active_high": None,
                "active_low": None, "latest_event": None, "impulse_leg": None, "reason": "Data H4 tidak cukup"}

    d4 = _ensure_indicators(df_h4)
    events, trend, swings = _detect_structure(d4, n=_SWING_N)
    highs = [s for s in swings if s["type"] == "H"]
    lows = [s for s in swings if s["type"] == "L"]
    active_high = highs[-1]["price"] if highs else float(d4["High"].max())
    active_low = lows[-1]["price"] if lows else float(d4["Low"].min())
    c_now = float(d4["Close"].iloc[-1])
    ema20 = _safe_float(d4["EMA20"].iloc[-1], c_now)
    ema50 = _safe_float(d4["EMA50"].iloc[-1], c_now)

    recent = [e for e in events if len(d4) - 1 - e["k"] <= 20]
    if len(recent) >= 3 and sum(1 for i in range(1, len(recent)) if recent[i]["arah"] != recent[i - 1]["arah"]) >= 2:
        return {"bias": "UNCONFIRMED", "ok": False, "regime": "CHOPPY", "active_high": active_high,
                "active_low": active_low, "latest_event": events[-1], "impulse_leg": None, "reason": "H4 choppy / whipsaw"}

    is_hh = len(highs) >= 2 and highs[-1]["price"] > highs[-2]["price"]
    is_hl = len(lows) >= 2 and lows[-1]["price"] > lows[-2]["price"]
    is_lh = len(highs) >= 2 and highs[-1]["price"] < highs[-2]["price"]
    is_ll = len(lows) >= 2 and lows[-1]["price"] < lows[-2]["price"]
    latest = events[-1] if events else None

    bias = "UNCONFIRMED"
    reason = "Struktur H4 belum terkonfirmasi"
    if latest and latest["arah"] == "bull" and c_now >= active_low:
        bias = "BULLISH"
        reason = f"H4 {latest['tipe']} bullish @ {latest['level']:.2f} (active low {active_low:.2f} bertahan)"
    elif latest and latest["arah"] == "bear" and c_now <= active_high:
        bias = "BEARISH"
        reason = f"H4 {latest['tipe']} bearish @ {latest['level']:.2f} (active high {active_high:.2f} bertahan)"
    elif is_hh and is_hl:
        bias = "BULLISH"
        reason = f"H4 Higher High & Higher Low ({highs[-1]['price']:.2f} / {lows[-1]['price']:.2f})"
    elif is_lh and is_ll:
        bias = "BEARISH"
        reason = f"H4 Lower High & Lower Low ({highs[-1]['price']:.2f} / {lows[-1]['price']:.2f})"
    elif ema20 > ema50 and c_now > ema20:
        bias = "BULLISH"
        reason = "H4 tren bullish (Close > EMA20 > EMA50)"
    elif ema20 < ema50 and c_now < ema20:
        bias = "BEARISH"
        reason = "H4 tren bearish (Close < EMA20 < EMA50)"

    imp = _extract_recent_impulse_leg(d4, bias) if bias in ("BULLISH", "BEARISH") else None
    return {
        "bias": bias,
        "ok": bias in ("BULLISH", "BEARISH"),
        "regime": f"TRENDING_{bias}" if bias in ("BULLISH", "BEARISH") else "RANGING",
        "active_high": active_high,
        "active_low": active_low,
        "latest_event": latest,
        "impulse_leg": imp,
        "reason": reason,
    }


def evaluate_h1_trend_confirmation(
    df_h1: Optional[pd.DataFrame],
    h4_info: Dict[str, Any],
    regime_h1: Optional[Dict[str, Any]] = None,
    explicit_conflict: bool = False,
) -> Dict[str, Any]:
    """
    H1 — Trend Confirmation:
    Membedakan ALIGNED, PULLBACK sehat di dalam struktur H4, dan CONFLICT struktural serius.
    """
    h4_bias = str((h4_info or {}).get("bias") or "UNCONFIRMED").upper()
    if h4_bias not in ("BULLISH", "BEARISH"):
        return {
            "status": "UNCONFIRMED", "has_conflict": False, "ok": False,
            "active_high": None, "active_low": None, "impulse_leg": None,
            "reason": f"H4 masih {h4_bias}; konfirmasi H1 belum dapat ditentukan",
        }

    if explicit_conflict:
        return {
            "status": "CONFLICT", "has_conflict": True, "ok": False,
            "active_high": (regime_h1 or {}).get("active_structure", {}).get("active_high"),
            "active_low": (regime_h1 or {}).get("active_structure", {}).get("active_low"),
            "impulse_leg": None,
            "reason": f"Konflik struktur H4 {h4_bias} vs H1",
        }

    if df_h1 is None or len(df_h1) < 2 * _SWING_N + 12:
        return {
            "status": "UNCONFIRMED", "has_conflict": False, "ok": False,
            "active_high": None, "active_low": None, "impulse_leg": None,
            "reason": "Data H1 tidak mencukupi",
        }

    d1 = _ensure_indicators(df_h1)
    events, trend, swings = _detect_structure(d1, n=_SWING_N)
    highs = [s for s in swings if s["type"] == "H"]
    lows = [s for s in swings if s["type"] == "L"]
    active_high = highs[-1]["price"] if highs else float(d1["High"].max())
    active_low = lows[-1]["price"] if lows else float(d1["Low"].min())
    c_now = float(d1["Close"].iloc[-1])

    reg_str = str((regime_h1 or {}).get("regime") or "").upper()
    bias_str = str((regime_h1 or {}).get("bias") or "").upper()
    if not bias_str:
        # Hitung lokal jika tidak diberikan
        if trend == 1 and c_now >= active_low:
            bias_str = "BULLISH"
        elif trend == -1 and c_now <= active_high:
            bias_str = "BEARISH"
        else:
            bias_str = "NETRAL"

    if reg_str == "CHOPPY":
        return {
            "status": "CONFLICT", "has_conflict": True, "ok": False,
            "active_high": active_high, "active_low": active_low, "impulse_leg": None,
            "reason": "Struktur H1 Choppy / Whipsaw",
        }

    h4_act_lo = h4_info.get("active_low")
    h4_act_hi = h4_info.get("active_high")
    imp = _extract_recent_impulse_leg(d1, h4_bias)

    if bias_str == h4_bias:
        return {
            "status": "ALIGNED", "has_conflict": False, "ok": True,
            "active_high": active_high, "active_low": active_low, "impulse_leg": imp,
            "reason": f"H1 {bias_str} selaras penuh dengan H4 {h4_bias}",
        }

    # Cek apakah H1 melanggar struktur inti H4 atau membentuk tren lawan berkelanjutan
    broke_h4 = (
        (h4_bias == "BULLISH" and isinstance(h4_act_lo, (int, float)) and c_now < float(h4_act_lo))
        or (h4_bias == "BEARISH" and isinstance(h4_act_hi, (int, float)) and c_now > float(h4_act_hi))
    )
    sustained_opp, opp_reason = _is_sustained_opposing_trend(d1, h4_bias)
    if broke_h4 or sustained_opp:
        why = opp_reason if sustained_opp else f"Harga menembus level struktur inti H4"
        return {
            "status": "CONFLICT", "has_conflict": True, "ok": False,
            "active_high": active_high, "active_low": active_low, "impulse_leg": imp,
            "reason": f"Konflik struktur H1 vs H4 {h4_bias}: {why}",
        }

    # Jika tidak ada konflik struktural serius: evaluasi apakah H1 dalam pullback sehat atau konsolidasi
    if bias_str in ("NETRAL", "RANGING") or (imp and 0.15 <= imp["retrace_ratio"] <= 0.85):
        return {
            "status": "PULLBACK", "has_conflict": False, "ok": True,
            "active_high": active_high, "active_low": active_low, "impulse_leg": imp,
            "reason": f"H1 melakukan pullback / konsolidasi sehat dalam struktur utama H4 {h4_bias}",
        }

    # Bila H1 menunjukkan arah berlawanan secara label meski belum 2x BOS (mis. test lama mock), tetap cek hati-hati:
    if (h4_bias == "BULLISH" and bias_str == "BEARISH") or (h4_bias == "BEARISH" and bias_str == "BULLISH"):
        return {
            "status": "CONFLICT", "has_conflict": True, "ok": False,
            "active_high": active_high, "active_low": active_low, "impulse_leg": imp,
            "reason": f"Konflik H4 {h4_bias} vs H1 {bias_str}",
        }

    return {
        "status": "UNCONFIRMED", "has_conflict": False, "ok": False,
        "active_high": active_high, "active_low": active_low, "impulse_leg": imp,
        "reason": f"H1 belum memberikan konfirmasi arah ({bias_str})",
    }


def assess_m30_condition(
    frames: Dict[str, pd.DataFrame],
    h4_info: Dict[str, Any],
    h1_info: Dict[str, Any],
    current_price: float,
    regime_m30: Optional[Dict[str, Any]] = None,
    explicit_m30_conflict: bool = False,
) -> Dict[str, Any]:
    """
    M30 — Setup Detection & Condition Classifier:
    Membedakan ALIGNED, PULLBACK sehat, CONSOLIDATION, vs CONFLICT serius / WEAK datar.
    """
    h4_bias = str((h4_info or {}).get("bias") or "UNCONFIRMED").upper()
    if h4_bias not in ("BULLISH", "BEARISH"):
        return {"status": "INVALID", "condition": "UNCONFIRMED", "ok": False,
                "msg": f"Bias H4 {h4_bias} belum jelas", "impulse_leg": None}

    df_m30 = (frames or {}).get("M30")
    if df_m30 is None or len(df_m30) < 20:
        return {"status": "INVALID", "condition": "NO_DATA", "ok": False,
                "msg": "Data M30 tidak mencukupi", "impulse_leg": None}

    d30 = _ensure_indicators(df_m30)
    last = d30.iloc[-1]
    c_now = float(last["Close"])
    ema20 = _safe_float(last.get("EMA20"), c_now)
    ema50 = _safe_float(last.get("EMA50"), c_now)
    rsi = _safe_float(last.get("RSI14"), 50.0)
    atr30 = max(0.4, _safe_float(last.get("ATR14"), 2.0))

    events, trend, swings = _detect_structure(d30, n=_SWING_N)
    look_20_pre = min(20, len(d30))
    rng_20_pre = float(d30["High"].iloc[-look_20_pre:].max() - d30["Low"].iloc[-look_20_pre:].min())
    ema_diff_pre = abs(ema20 - ema50)
    reg = str((regime_m30 or {}).get("regime") or "").upper()
    m30_bias = str((regime_m30 or {}).get("bias") or "").upper()
    if not reg:
        if rng_20_pre <= 2.6 * atr30 and ema_diff_pre < 0.4 * atr30:
            m30_bias, reg = "NETRAL", "RANGING"
        elif trend == 1 or (ema20 > ema50 and c_now > ema20):
            m30_bias, reg = "BULLISH", "TRENDING_BULLISH"
        elif trend == -1 or (ema20 < ema50 and c_now < ema20):
            m30_bias, reg = "BEARISH", "TRENDING_BEARISH"
        else:
            m30_bias, reg = "NETRAL", "RANGING"

    h1_imp = h1_info.get("impulse_leg") if isinstance(h1_info, dict) else None
    m30_imp = _extract_recent_impulse_leg(d30, h4_bias)
    active_imp = m30_imp or h1_imp

    h1_act_lo = (h1_info or {}).get("active_low") or (h4_info or {}).get("active_low")
    h1_act_hi = (h1_info or {}).get("active_high") or (h4_info or {}).get("active_high")

    # 1. Evaluasi apakah harga saat ini berada DALAM batas struktur H1/H4 yang utuh (Pullback Zone Check)
    within_h1_structure = False
    if h1_imp is not None:
        if h4_bias == "BULLISH" and c_now >= float(h1_imp["origin_price"]) and -0.10 <= h1_imp["retrace_ratio"] <= TF_PULLBACK_MAX_RETRACE:
            within_h1_structure = True
        elif h4_bias == "BEARISH" and c_now <= float(h1_imp["origin_price"]) and -0.10 <= h1_imp["retrace_ratio"] <= TF_PULLBACK_MAX_RETRACE:
            within_h1_structure = True

    # 2. Deteksi Konflik Struktural Serius pada M30
    sustained_opp, opp_msg = _is_sustained_opposing_trend(d30, h4_bias)
    broke_h1_origin = (
        (h4_bias == "BULLISH" and isinstance(h1_act_lo, (int, float)) and c_now < float(h1_act_lo) - 0.5 * atr30)
        or (h4_bias == "BEARISH" and isinstance(h1_act_hi, (int, float)) and c_now > float(h1_act_hi) + 0.5 * atr30)
    )

    if explicit_m30_conflict and not within_h1_structure:
        if sustained_opp:
            return {"status": "CONFLICT", "condition": "CONFLICT", "ok": False,
                    "msg": f"M30 bertentangan serius dengan H4 {h4_bias}: {opp_msg}", "impulse_leg": active_imp}
        if m30_bias in ("BULLISH", "BEARISH") and m30_bias != h4_bias:
            return {"status": "CONFLICT", "condition": "CONFLICT", "ok": False,
                    "msg": f"M30 {m30_bias} bertentangan dengan H4 {h4_bias}", "impulse_leg": active_imp}
        if reg == "CHOPPY":
            return {"status": "WEAK", "condition": "CHOPPY", "ok": False,
                    "msg": "Market M30 Choppy / Whipsaw", "impulse_leg": active_imp}
        return {"status": "CONFLICT", "condition": "CONFLICT", "ok": False,
                "msg": f"Konflik konteks M30 terhadap struktur H4 {h4_bias} (M30 {m30_bias}/{reg}, RSI {rsi:.1f})",
                "impulse_leg": active_imp}

    if sustained_opp and (broke_h1_origin or not within_h1_structure):
        return {"status": "CONFLICT", "condition": "CONFLICT", "ok": False,
                "msg": f"M30 bertentangan serius dengan H4 {h4_bias}: {opp_msg}", "impulse_leg": active_imp}

    # 3. Cek Choppy M30
    if reg == "CHOPPY":
        return {"status": "WEAK", "condition": "CHOPPY", "ok": False,
                "msg": "Market M30 Choppy / Whipsaw", "impulse_leg": active_imp}

    # 4. Cek Kondisi Selaras Penuh (ALIGNED / CONFIRMED)
    want_bull = h4_bias == "BULLISH"
    if (want_bull and m30_bias == "BULLISH") or ((not want_bull) and m30_bias == "BEARISH"):
        # Pastikan tidak sedang ada impuls berlawanan ekstrem melawan H4 tepat di 3 bar terakhir
        if len(d30) >= 3:
            if want_bull:
                opp_imp = max(float(d30["Open"].iloc[k] - d30["Close"].iloc[k]) for k in range(-3, 0))
                pro_imp = max(float(d30["Close"].iloc[k] - d30["Open"].iloc[k]) for k in range(-3, 0))
            else:
                opp_imp = max(float(d30["Close"].iloc[k] - d30["Open"].iloc[k]) for k in range(-3, 0))
                pro_imp = max(float(d30["Open"].iloc[k] - d30["Close"].iloc[k]) for k in range(-3, 0))
            if opp_imp >= 1.6 * atr30 and pro_imp < 0.6 * atr30:
                return {"status": "PULLBACK", "condition": "PULLBACK", "ok": True,
                        "msg": f"M30 pullback tajam ke area nilai dalam tren H4 {h4_bias} (menunggu konfirmasi M5)",
                        "impulse_leg": active_imp}
        return {"status": "CONFIRMED", "condition": "ALIGNED", "ok": True,
                "msg": f"M30 {h4_bias} selaras (RSI {rsi:.1f}, {reg})", "impulse_leg": active_imp}

    # 5. Cek apakah M30 merupakan HEALTHY PULLBACK / CONSOLIDATION pasca-impuls (anti-false WAIT!)
    look_box = min(14, len(d30) - 2)
    prior_high = float(d30["High"].iloc[-look_box - 2 : -1].max())
    prior_low = float(d30["Low"].iloc[-look_box - 2 : -1].min())
    breakout_bull = want_bull and c_now > prior_high and (c_now - float(last["Open"])) >= TF_BREAKOUT_MIN_BODY_ATR * atr30
    breakout_bear = (not want_bull) and c_now < prior_low and (float(last["Open"]) - c_now) >= TF_BREAKOUT_MIN_BODY_ATR * atr30

    if breakout_bull or breakout_bear:
        return {"status": "CONFIRMED", "condition": "CONSOLIDATION_BREAKOUT", "ok": True,
                "msg": f"M30 breakout konsolidasi searah H4 {h4_bias} (close {c_now:.2f})",
                "impulse_leg": active_imp, "box_low": prior_low, "box_high": prior_high}

    # Cek pasar M30 datar & lemah berkepanjangan (bukan pullback tren aktif)
    look_20 = min(20, len(d30))
    rng_20 = float(d30["High"].iloc[-look_20:].max() - d30["Low"].iloc[-look_20:].min())
    look_40 = min(40, len(d30))
    rng_40 = float(d30["High"].iloc[-look_40:].max() - d30["Low"].iloc[-look_40:].min())
    if m30_bias in ("NETRAL", "RANGING") and rng_20 <= 2.6 * atr30 and rng_40 <= 3.6 * atr30:
        if (want_bull and (rsi < 46.0 or c_now < ema20 - 0.5 * atr30)) or ((not want_bull) and (rsi > 54.0 or c_now > ema20 + 0.5 * atr30)):
            return {"status": "WEAK", "condition": "WEAK", "ok": False,
                    "msg": f"Momentum M30 lemah / datar (RSI {rsi:.1f}, range {rng_20:.2f} pt)", "impulse_leg": active_imp}

    # Healthy pullback: ada leg impuls terukur di M30 atau H1 & retracement berada dalam rentang sehat & struktur H1/H4 utuh
    if active_imp is not None and not broke_h1_origin:
        rt = active_imp["retrace_ratio"]
        if TF_PULLBACK_MIN_RETRACE <= rt <= TF_PULLBACK_MAX_RETRACE:
            return {"status": "PULLBACK", "condition": "PULLBACK", "ok": True,
                    "msg": f"M30 pullback sehat ({rt * 100:.0f}% retracement leg {h4_bias}) menjaga struktur H1/H4",
                    "impulse_leg": active_imp}
        if -0.05 <= rt < TF_PULLBACK_MIN_RETRACE and ((want_bull and rsi >= 46) or ((not want_bull) and rsi <= 54)):
            return {"status": "CONSOLIDATION", "condition": "CONSOLIDATION", "ok": True,
                    "msg": f"M30 konsolidasi dekat puncak/dasar tren H4 {h4_bias} (potensi continuation/breakout)",
                    "impulse_leg": active_imp, "box_low": prior_low, "box_high": prior_high}

    # 6. Flat / weak market tanpa impuls maupun breakout dengan RSI/posisi EMA berlawanan -> WEAK
    if (want_bull and (rsi < 46.0 or c_now < ema20 - 0.5 * atr30)) or ((not want_bull) and (rsi > 54.0 or c_now > ema20 + 0.5 * atr30)):
        return {"status": "WEAK", "condition": "WEAK", "ok": False,
                "msg": f"Momentum M30 lemah / datar (RSI {rsi:.1f}, range {rng_20:.2f} pt)", "impulse_leg": active_imp}

    if reg == "RANGING":
        return {"status": "CONFIRMED", "condition": "CONSOLIDATION", "ok": True,
                "msg": f"M30 stabil netral mendukung struktur H4 {h4_bias} (RSI {rsi:.1f})",
                "impulse_leg": active_imp}

    return {"status": "PULLBACK", "condition": "PULLBACK", "ok": True,
            "msg": f"M30 transisi/pullback moderat dalam struktur H4 {h4_bias} (RSI {rsi:.1f})",
            "impulse_leg": active_imp}


# =============================================================================
# INTEGRASI SNR READ-ONLY (BREAKOUT / RETEST / SUPPORT & RESISTANCE BARRIERS)
# =============================================================================
def _strip_unclosed_for_snr(df: pd.DataFrame) -> pd.DataFrame:
    """Salin DataFrame secara independen hanya dengan CLOSED candles (tanpa memutasi input)."""
    attrs = dict(getattr(df, "attrs", {}) or {})
    work = df.copy(deep=True)
    had_unclosed = False
    if "is_closed" in work.columns:
        unclosed_mask = work["is_closed"].astype(bool) == False  # noqa: E712
        if unclosed_mask.any():
            had_unclosed = True
            work = work[~unclosed_mask].copy()
    elif "is_forming" in work.columns:
        forming_mask = work["is_forming"].astype(bool) == True  # noqa: E712
        if forming_mask.any():
            had_unclosed = True
            work = work[~forming_mask].copy()
    elif attrs.get("is_closed") is False or attrs.get("has_unclosed_last_bar") is True:
        had_unclosed = True
        work = work.iloc[:-1].copy() if len(work) > 1 else work.iloc[0:0].copy()

    work.attrs = dict(attrs)
    if had_unclosed:
        work.attrs["had_unclosed_bar"] = True
    return work


def _snr_frames_signature(frames: Optional[Dict[str, Any]]) -> Tuple:
    """Fingerprint all candle inputs used by the shared SNR snapshot."""
    if not isinstance(frames, dict):
        return ()
    parts = []
    for tf in ("H4", "H1", "M30", "M15", "M5"):
        df = frames.get(tf)
        if not isinstance(df, pd.DataFrame) or df.empty:
            parts.append((tf, 0, "", 0.0, bool(getattr(df, "attrs", {}).get("is_closed", True)) if isinstance(df, pd.DataFrame) else True))
            continue
        columns = tuple(c for c in (
            "Time", "time", "Open", "High", "Low", "Close", "ATR14",
            "Volume", "TickVolume", "tick_volume", "is_closed", "is_forming",
        ) if c in df.columns)
        digest = hashlib.sha256(pd.util.hash_pandas_object(df.loc[:, list(columns)], index=True).values.tobytes()).hexdigest()
        attrs = getattr(df, "attrs", {}) or {}
        parts.append((tf, len(df), columns, digest, attrs.get("is_closed"), attrs.get("has_unclosed_last_bar"), attrs.get("time_tz")))
    return tuple(parts)


def _is_snr_ctx_usable(snr_context: Optional[Dict[str, Any]]) -> bool:
    """Pastikan konteks SNR benar-benar tersedia, segar (PASS), dan tidak STALE/INVALID."""
    if not isinstance(snr_context, dict) or not snr_context.get("available"):
        return False
    if snr_context.get("is_stale") or snr_context.get("historical_only"):
        return False
    if snr_context.get("usable_for_entry") is False:
        return False
    fresh = snr_context.get("freshness")
    if isinstance(fresh, dict):
        f_st = str(fresh.get("status") or "PASS").upper()
        if f_st in ("STALE", "INVALID", "NO DATA") or fresh.get("ok") is False:
            return False
    return True


def _match_snr_full_zone(zone_id: Any, zones_pool: List[Dict[str, Any]], fallback: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    zid = str(zone_id or "").strip()
    if zid:
        for z in zones_pool:
            if isinstance(z, dict) and str(z.get("zone_id") or "") == zid:
                return z
    if isinstance(fallback, dict) and fallback.get("low") is not None and fallback.get("high") is not None:
        return fallback
    return None


def extract_snr_supporting_context(
    frames: Optional[Dict[str, pd.DataFrame]],
    current_price: float,
    market_status: str = "OPEN",
    snapshot: Optional[Dict[str, Any]] = None,
    snr_scan_override: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Mengambil konteks SNR secara read-only pada salinan frame (tidak pernah memutasi snapshot/dataframe).
    Menggunakan API SNR resmi (run_snr_scan, rank_zones, detect_zone_breakout,
    detect_zone_retest, detect_zone_rejection), membedakan status AVAILABLE / PARTIAL / UNAVAILABLE,
    menjaga aturan freshness (PASS / STALE / NO DATA / INVALID), serta mencatat error integrasi
    secara transparan tanpa menyembunyikan kegagalan.
    """
    from modules.utils import _catat_error

    base: Dict[str, Any] = {
        "status": "UNAVAILABLE",
        "available": False,
        "reason": "SNR context tidak tersedia",
        "summary": "[UNAVAILABLE] SNR context tidak tersedia",
        "support": None,
        "resistance": None,
        "supports": [],
        "resistances": [],
        "broken_zones": [],
        "price_action": {},
        "freshness": {
            "status": "NO DATA",
            "data_freshness": "NO DATA",
            "data_validation": "FAIL",
            "ok": False,
            "reason": "SNR belum dievaluasi",
            "stale_timeframes": [],
            "missing_timeframes": [],
            "ages_minutes": {},
            "last_candle_times": {},
            "is_market_closed": False,
            "tick_status": "NO DATA",
            "candle_status": "NO DATA",
        },
        "is_stale": False,
        "historical_only": False,
        "usable_for_entry": False,
        "breakout_code": "NO_BREAKOUT",
        "breakout_direction": "NONE",
        "breakout_zone": None,
        "retest_code": "NO_RETEST",
        "retest_direction": "NONE",
        "retest_zone": None,
        "rejection_code": "NONE",
        "rejection_zone": None,
        "market_bias": {},
        "momentum_by_tf": {},
        "entry_validation": {},
        "available_timeframes": [],
        "missing_timeframes": [],
        "warnings": [],
        "error_source": None,
        "error_type": None,
        "error_reason": None,
    }

    def _fail(
        err_type: str,
        reason_msg: str,
        exc: Optional[Exception] = None,
        extra_missing: Optional[List[str]] = None,
        source: str = "processing",
        fresh_status: str = "NO DATA",
        fresh_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        out_fail = dict(base)
        fi = fresh_info if isinstance(fresh_info, dict) else {}
        out_fail["error_source"] = source
        out_fail["error_type"] = err_type
        out_fail["error_reason"] = reason_msg
        out_fail["reason"] = reason_msg
        out_fail["missing_timeframes"] = list(extra_missing or [])
        out_fail["summary"] = f"[UNAVAILABLE] ({source}) {reason_msg}"
        out_fail["freshness"] = {
            "status": fresh_status,
            "data_freshness": str(fi.get("status") or fresh_status),
            "data_validation": "FAIL",
            "ok": False,
            "reason": reason_msg,
            "stale_timeframes": list(fi.get("stale_timeframes") or []),
            "missing_timeframes": list(extra_missing or []),
            "ages_minutes": dict(fi.get("ages_minutes") or {}),
            "last_candle_times": dict(fi.get("last_candle_times") or {}),
            "is_market_closed": bool(fi.get("is_market_closed", False)),
            "tick_status": fi.get("tick_status", fresh_status),
            "candle_status": fi.get("candle_status", fresh_status),
        }
        try:
            _catat_error("extract_snr_supporting_context", exc if exc is not None else RuntimeError(f"{err_type} [{source}]: {reason_msg}"))
        except Exception:
            pass
        try:
            from modules.snr.data_provider import log_snr_error
            log_snr_error("extract_snr_supporting_context", exc if exc is not None else RuntimeError(f"{err_type} [{source}]: {reason_msg}"))
        except Exception:
            pass
        return out_fail

    if not frames or not isinstance(frames, dict):
        return _fail("MissingFramesError", "Frames OHLCV kosong atau tidak valid.", source="empty_data", fresh_status="NO DATA")

    if isinstance(current_price, bool) or not isinstance(current_price, (int, float)):
        try:
            cp = _safe_float(current_price, 0.0)
            if isinstance(current_price, bool) or cp <= 0:
                return _fail("InvalidPriceError", f"Parameter current_price tidak valid ({current_price!r}).", source="parameter", fresh_status="INVALID")
        except Exception as p_err:
            return _fail("TypeError", f"Kesalahan parameter current_price ({current_price!r}): {p_err}", exc=p_err, source="parameter", fresh_status="INVALID")
    else:
        cp = _safe_float(current_price, 0.0)
        if cp <= 0:
            return _fail("InvalidPriceError", f"current_price tidak valid ({current_price!r}).", source="parameter", fresh_status="INVALID")

    # 1. Validasi & impor API resmi SNR secara eksplisit (tidak boleh diam-diam gagal)
    try:
        from modules.snr.config import get_snr_config
        from modules.snr.scanner import run_snr_scan
        from modules.snr.zone_ranking import rank_zones
        from modules.snr.breakout_detection import detect_zone_breakout
        from modules.snr.retest_detection import detect_zone_retest
        from modules.snr.rejection_detection import detect_zone_rejection
    except ImportError as imp_err:
        return _fail("ImportError", f"Gagal mengimpor API SNR: {imp_err}", exc=imp_err, source="import", fresh_status="INVALID")
    except Exception as api_err:
        return _fail(type(api_err).__name__, f"Kesalahan memuat modul SNR: {api_err}", exc=api_err, source="import", fresh_status="INVALID")

    # Referensi eksplisit agar API inti tervalidasi signature-nya
    _required_apis = (get_snr_config, run_snr_scan, rank_zones, detect_zone_breakout, detect_zone_retest, detect_zone_rejection)
    if any(not callable(fn) for fn in _required_apis):
        return _fail("APIValidationError", "Salah satu fungsi API SNR tidak callable.", source="import", fresh_status="INVALID")

    # 2. Siapkan salinan frame CLOSED candles tanpa memutasi input
    frames_copy: Dict[str, pd.DataFrame] = {}
    available_tfs: List[str] = []
    missing_tfs: List[str] = []
    unclosed_tfs: List[str] = []
    warnings: List[str] = []

    for tf in ("H4", "H1", "M30", "M5"):
        raw_df = frames.get(tf)
        if not isinstance(raw_df, pd.DataFrame) or raw_df.empty:
            missing_tfs.append(f"{tf}(kosong)")
            continue
        req_cols = ("Open", "High", "Low", "Close")
        lower_map = {str(c).strip().lower(): c for c in raw_df.columns}
        if not all(k.lower() in lower_map for k in req_cols):
            missing_tfs.append(f"{tf}(kolom OHLC kurang)")
            continue
        closed_df = _strip_unclosed_for_snr(raw_df)
        if getattr(closed_df, "attrs", {}).get("had_unclosed_bar") or getattr(closed_df, "attrs", {}).get("is_closed") is False:
            unclosed_tfs.append(tf)
        if len(closed_df) < 20:
            missing_tfs.append(f"{tf}({len(closed_df)} bar < 20)")
            continue
        if "Time" not in closed_df.columns and isinstance(closed_df.index, pd.DatetimeIndex):
            closed_df["Time"] = closed_df.index
        closed_attrs = dict(getattr(closed_df, "attrs", {}) or {})
        ind_df = _ensure_indicators(closed_df)
        if isinstance(ind_df, pd.DataFrame):
            ind_df.attrs = dict(closed_attrs)
        frames_copy[tf] = ind_df
        available_tfs.append(tf)

    if unclosed_tfs:
        warnings.append(f"Candle berjalan (unclosed) pada {', '.join(unclosed_tfs)} dipisahkan; analisis hanya memakai candle tertutup")

    if not available_tfs or not any(tf in frames_copy for tf in ("M5", "M30", "H1")):
        is_all_empty = all(
            (frames.get(tf) is None) or (isinstance(frames.get(tf), pd.DataFrame) and frames.get(tf).empty)
            for tf in ("H4", "H1", "M30", "M5")
        )
        return _fail(
            "IncompleteDataError",
            f"Data candle CLOSED tidak mencukupi untuk analisis SNR ({', '.join(missing_tfs) or 'tanpa TF valid'}).",
            extra_missing=missing_tfs,
            source="empty_data" if is_all_empty else "incomplete_data",
            fresh_status="NO DATA" if is_all_empty else "INVALID",
        )

    # 3. Jalankan pipeline resmi run_snr_scan secara read-only pada salinan snapshot/frames
    try:
        cfg = get_snr_config()
        snr_res: Optional[Dict[str, Any]] = None
        snap_id = snapshot.get("snapshot_id") if isinstance(snapshot, dict) else None

        if isinstance(snr_scan_override, dict):
            ov_id = snr_scan_override.get("snapshot_id")
            if snap_id and ov_id and ov_id != snap_id:
                return _fail(
                    "SnapshotMismatchError",
                    f"Snapshot override SNR ({ov_id}) tidak cocok dengan snapshot aktif ({snap_id}).",
                    extra_missing=missing_tfs,
                    source="parameter",
                    fresh_status="INVALID",
                )
            snr_res = snr_scan_override
        elif isinstance(snapshot, dict) and isinstance(snapshot.get("_shared_snr_scan"), dict):
            cand_shared = snapshot.get("_shared_snr_scan")
            is_real_fn = (
                getattr(run_snr_scan, "__module__", "") == "modules.snr.scanner"
                and not hasattr(run_snr_scan, "assert_called")
            )
            if (
                is_real_fn
                and snap_id
                and cand_shared.get("snapshot_id") == snap_id
                and snapshot.get("_shared_snr_sig") == _snr_frames_signature(frames)
                and snapshot.get("_shared_snr_quote") == tuple(
                    snapshot.get(key) for key in (
                        "snapshot_price", "bid", "ask", "spread", "tick_time", "server_time",
                        "snapshot_time", "market_status", "data_freshness",
                    )
                )
            ):
                snr_res = cand_shared

        if snr_res is None:
            snap_ro: Optional[Dict[str, Any]] = None
            tick_arg: Optional[Dict[str, Any]] = None
            if isinstance(snapshot, dict):
                snap_ro = {
                    k: v
                    for k, v in snapshot.items()
                    if k not in ("frames", "_shared_snr_scan", "_shared_snr_sig", "_shared_snr_quote")
                }
                snap_ro["frames"] = frames_copy
                snap_ro["live"] = False
                snap_bid = _safe_float(snap_ro.get("bid"), 0.0)
                snap_price = _safe_float(snap_ro.get("snapshot_price") or snap_ro.get("current_price") or snap_ro.get("mid"), 0.0)
                if snap_bid <= 0 and snap_price <= 0 and str(market_status or "").upper() != "CLOSED":
                    tick_arg = {"bid": cp, "ask": cp}
            else:
                if str(market_status or "").upper() != "CLOSED":
                    tick_arg = {"bid": cp, "ask": cp}

            snr_res = run_snr_scan(
                symbol=str((snapshot or {}).get("symbol") or SYMBOL) if isinstance(snapshot, dict) else SYMBOL,
                snapshot=snap_ro,
                frames=frames_copy,
                tick=tick_arg,
                live=False,
                has_unclosed_last_bar=False,
                market_status=market_status,
                config=cfg,
            )

        if not isinstance(snr_res, dict) or not snr_res:
            return _fail("EmptyScanResultError", "run_snr_scan mengembalikan hasil kosong.", extra_missing=missing_tfs, source="empty_data", fresh_status="NO DATA")

        fresh_info = snr_res.get("data_freshness_info") if isinstance(snr_res.get("data_freshness_info"), dict) else {}
        fresh_raw = str(snr_res.get("data_freshness") or fresh_info.get("status") or "FRESH").upper()
        mkt_res = str(snr_res.get("market_status") or market_status or "OPEN").upper()

        if fresh_raw in ("INVALID", "NO DATA"):
            f_reason = str(fresh_info.get("reason") or f"Validasi data SNR gagal ({fresh_raw}).")
            return _fail(
                "DataValidationError",
                f_reason,
                extra_missing=missing_tfs,
                source="data_validation",
                fresh_status=fresh_raw,
                fresh_info=fresh_info,
            )

        ranked = snr_res.get("ranked_zones") or {}
        supports = list(ranked.get("active_supports") or [])
        resistances = list(ranked.get("active_resistances") or [])
        broken_zones = list(ranked.get("broken_zones") or [])
        all_zones_pool = list(broken_zones) + list(supports) + list(resistances) + list(ranked.get("all_zones") or [])

        pa = dict(snr_res.get("price_action") or {})
        ev = dict(snr_res.get("entry_validation") or {})
        mb = dict(snr_res.get("market_bias") or {})
        moms = dict(snr_res.get("momentum_by_tf") or {})

        bo_det = pa.get("breakout_detail") or {}
        rt_det = pa.get("retest_detail") or {}
        rej_det = pa.get("rejection_detail") or {}

        focus_z = ev.get("target_zone") if isinstance(ev.get("target_zone"), dict) else None
        bo_zone = _match_snr_full_zone(bo_det.get("zone_id"), all_zones_pool, fallback=(broken_zones[0] if broken_zones else focus_z))
        rt_zone = _match_snr_full_zone(rt_det.get("zone_id"), all_zones_pool, fallback=(bo_zone or focus_z))
        rej_zone = _match_snr_full_zone(rej_det.get("zone_id"), all_zones_pool, fallback=focus_z)

        bo_code = str(pa.get("breakout_code") or "NO_BREAKOUT").upper()
        rt_code = str(pa.get("retest_code") or "NO_RETEST").upper()
        rej_code = str(pa.get("rejection_code") or "NONE").upper()

        is_stale = (fresh_raw == "STALE") or (mkt_res == "CLOSED")
        fresh_norm_status = "STALE" if is_stale else ("PASS" if fresh_raw == "FRESH" else fresh_raw)
        freshness_payload = {
            "status": fresh_norm_status,
            "data_freshness": fresh_raw,
            "data_validation": "FAIL" if is_stale else str(fresh_info.get("data_validation") or "PASS"),
            "ok": not is_stale and fresh_raw == "FRESH",
            "reason": str(fresh_info.get("reason") or ("Data SNR lengkap dan segar" if not is_stale else "Data SNR terlambat / market closed (STALE)")),
            "stale_timeframes": list(fresh_info.get("stale_timeframes") or []),
            "missing_timeframes": list(missing_tfs),
            "ages_minutes": dict(fresh_info.get("ages_minutes") or {}),
            "last_candle_times": dict(fresh_info.get("last_candle_times") or snr_res.get("last_candle_times") or {}),
            "is_market_closed": bool(fresh_info.get("is_market_closed", mkt_res == "CLOSED")),
            "tick_status": str(fresh_info.get("tick_status") or ("MARKET CLOSED" if mkt_res == "CLOSED" else ("STALE" if is_stale else "FRESH"))),
            "candle_status": str(fresh_info.get("candle_status") or ("HISTORICAL (CLOSED)" if mkt_res == "CLOSED" else ("STALE" if is_stale else "FRESH"))),
        }

        if missing_tfs:
            warnings.append(f"Timeframe SNR tidak lengkap: {', '.join(missing_tfs)}")
        if not supports and not resistances and not broken_zones:
            warnings.append("Belum terbentuk zona Support/Resistance aktif dari data saat ini")

        nearest_sup = supports[0] if supports else None
        nearest_res = resistances[0] if resistances else None

        if is_stale:
            stale_reason = freshness_payload["reason"]
            warnings.append(f"Data SNR STALE ({stale_reason}); hanya ditampilkan sebagai konteks historis dan tidak dipakai untuk konfirmasi entry")
            has_hist_zones = bool(supports or resistances or broken_zones)
            status_str = "PARTIAL" if has_hist_zones else "UNAVAILABLE"
            reason_str = f"Data SNR berstatus STALE ({stale_reason}); hanya berlaku sebagai konteks historis."
            summary = (
                f"[STALE - HISTORIS SAJA] Breakout={pa.get('breakout_summary', 'NOT CONFIRMED')} | "
                f"Retest={pa.get('retest_summary', 'NOT CONFIRMED')} | Rejection={pa.get('rejection_summary', 'NOT CONFIRMED')}"
            )
            return {
                "status": status_str,
                "available": False,
                "reason": reason_str,
                "summary": summary,
                "support": nearest_sup,
                "resistance": nearest_res,
                "supports": supports,
                "resistances": resistances,
                "broken_zones": broken_zones,
                "price_action": pa,
                "freshness": freshness_payload,
                "is_stale": True,
                "historical_only": True,
                "usable_for_entry": False,
                "breakout_code": bo_code,
                "breakout_direction": str(pa.get("breakout_direction") or "NONE").upper(),
                "breakout_zone": bo_zone,
                "retest_code": rt_code,
                "retest_direction": str(pa.get("retest_direction") or "NONE").upper(),
                "retest_zone": rt_zone,
                "rejection_code": rej_code,
                "rejection_zone": rej_zone,
                "market_bias": mb,
                "momentum_by_tf": moms,
                "entry_validation": ev,
                "available_timeframes": available_tfs,
                "missing_timeframes": missing_tfs,
                "warnings": warnings,
                "error_source": "stale_data",
                "error_type": "StaleDataWarning",
                "error_reason": stale_reason,
            }

        is_partial = bool(missing_tfs or unclosed_tfs or (not supports and not resistances and not broken_zones))
        status_str = "PARTIAL" if is_partial else "AVAILABLE"
        if missing_tfs:
            prefix = f"[PARTIAL: {', '.join(missing_tfs)}]"
            reason_str = f"Sebagian timeframe SNR tidak lengkap ({', '.join(missing_tfs)}); dihitung dari {', '.join(available_tfs)}."
        elif unclosed_tfs:
            prefix = f"[PARTIAL: unclosed {', '.join(unclosed_tfs)}]"
            reason_str = f"Candle berjalan pada {', '.join(unclosed_tfs)} dipisahkan; dihitung hanya dari candle tertutup."
        elif not supports and not resistances and not broken_zones:
            prefix = "[PARTIAL]"
            reason_str = "Belum terbentuk zona Support/Resistance aktif dari data saat ini."
        else:
            prefix = "[AVAILABLE]"
            reason_str = "Konteks SNR lengkap dan segar."
        summary = (
            f"{prefix} Breakout={pa.get('breakout_summary', 'NOT CONFIRMED')} | "
            f"Retest={pa.get('retest_summary', 'NOT CONFIRMED')} | Rejection={pa.get('rejection_summary', 'NOT CONFIRMED')}"
        )

        return {
            "status": status_str,
            "available": True,
            "reason": reason_str,
            "summary": summary,
            "support": nearest_sup,
            "resistance": nearest_res,
            "supports": supports,
            "resistances": resistances,
            "broken_zones": broken_zones,
            "price_action": pa,
            "freshness": freshness_payload,
            "is_stale": False,
            "historical_only": False,
            "usable_for_entry": ("M5" not in unclosed_tfs),
            "breakout_code": bo_code,
            "breakout_direction": str(pa.get("breakout_direction") or "NONE").upper(),
            "breakout_zone": bo_zone,
            "retest_code": rt_code,
            "retest_direction": str(pa.get("retest_direction") or "NONE").upper(),
            "retest_zone": rt_zone,
            "rejection_code": rej_code,
            "rejection_zone": rej_zone,
            "market_bias": mb,
            "momentum_by_tf": moms,
            "entry_validation": ev,
            "available_timeframes": available_tfs,
            "missing_timeframes": missing_tfs,
            "warnings": warnings,
            "error_source": None,
            "error_type": None,
            "error_reason": None,
        }
    except TypeError as param_exc:
        return _fail("TypeError", f"Kesalahan parameter pemanggilan SNR: {param_exc}", exc=param_exc, extra_missing=missing_tfs, source="parameter", fresh_status="INVALID")
    except Exception as exc:
        return _fail(type(exc).__name__, f"Kegagalan kalkulasi SNR: {exc}", exc=exc, extra_missing=missing_tfs, source="processing", fresh_status="INVALID")


# =============================================================================
# PEMBANGUN KANDIDAT SETUP MULTI-TIPE (PULLBACK, CONTINUATION, BREAKOUT RETEST, REVERSAL)
# =============================================================================
def _dedup_push_candidate(cands: List[Dict[str, Any]], new_c: Dict[str, Any]) -> None:
    lo = float(new_c["price_low"])
    hi = float(new_c["price_high"])
    st = new_c["setup_type"]
    dr = new_c["direction"]
    for existing in cands:
        if existing["setup_type"] == st and existing["direction"] == dr:
            overlap = min(hi, float(existing["price_high"])) - max(lo, float(existing["price_low"]))
            if overlap >= -0.4:
                return
    cands.append(new_c)


def build_trend_following_candidates(
    frames: Dict[str, pd.DataFrame],
    h4_info: Dict[str, Any],
    h1_info: Dict[str, Any],
    m30_info: Dict[str, Any],
    current_price: float,
    scan: Optional[Dict[str, Any]] = None,
    snr_context: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Membangun kandidat setup Trend Pullback, Trend Continuation, Breakout Retest, dan Reversal."""
    h4_bias = str((h4_info or {}).get("bias") or "UNCONFIRMED").upper()
    if h4_bias not in ("BULLISH", "BEARISH") or current_price <= 0:
        return []

    direction = "BUY" if h4_bias == "BULLISH" else "SELL"
    ztype = "DEMAND" if direction == "BUY" else "SUPPLY"
    df_h1 = _ensure_indicators(frames.get("H1"))
    df_m30 = _ensure_indicators(frames.get("M30"))
    df_m5 = _ensure_indicators(frames.get("M5"))
    atr_h1 = max(0.8, _safe_float(df_h1["ATR14"].iloc[-1] if df_h1 is not None and "ATR14" in df_h1.columns else 3.0, 3.0))
    atr_m30 = max(0.5, _safe_float(df_m30["ATR14"].iloc[-1] if df_m30 is not None and "ATR14" in df_m30.columns else 2.0, 2.0))
    atr_m5 = max(0.4, _safe_float(df_m5["ATR14"].iloc[-1] if df_m5 is not None and "ATR14" in df_m5.columns else 1.2, 1.2))
    cands: List[Dict[str, Any]] = []

    # 1. TREND_PULLBACK Candidates:
    #    1a) Active SMC Zones (OB + FVG)
    for az in (scan or {}).get("active_zones") or []:
        azd = az.to_dict() if hasattr(az, "to_dict") else dict(az)
        if (
            str(azd.get("zone_type", "")).upper() != ztype
            or not azd.get("is_valid", True)
            or str(azd.get("risk_tier") or "").upper() == "FAR_ZONE"
        ):
            continue
        _dedup_push_candidate(cands, {
            "zone_id": azd.get("zone_id") or f"TF_PB_OB_{direction}_{float(azd['price_low']):.2f}_{float(azd['price_high']):.2f}",
            "setup_type": "TREND_PULLBACK",
            "source_kind": "SMC_OB_FVG",
            "timeframe": azd.get("timeframe", "H1"),
            "formation_time": azd.get("bos_candle_time"),
            "formation_timeframe": azd.get("bos_timeframe") or azd.get("timeframe", "H1"),
            "zone_type": ztype,
            "direction": direction,
            "price_low": float(azd["price_low"]),
            "price_high": float(azd["price_high"]),
            "invalidation_price": float(azd.get("invalidation_price") or (azd["price_low"] if direction == "BUY" else azd["price_high"])),
            "grade": azd.get("grade", "A"),
            "score": float(azd.get("score", 78.0)),
            "priority": 100.0 + float(azd.get("score", 78.0)),
            "label": f"SMC {azd.get('timeframe', 'H1')} {ztype} OB+FVG",
        })

    #    1b) Pro-Trend Unfilled FVG (H4 / H1 / M30)
    for tf_fvg in ("H4", "H1", "M30"):
        for fv in _detect_fvgs(frames.get(tf_fvg), tf_fvg, min_atr_ratio=0.20):
            if fv["direction"] != direction:
                continue
            dist_f = 0.0 if fv["low"] <= current_price <= fv["high"] else min(abs(current_price - fv["low"]), abs(current_price - fv["high"]))
            if dist_f <= max(18.0, 4.0 * atr_h1):
                base_sc = 80.0 if tf_fvg == "H4" else (76.0 if tf_fvg == "H1" else 70.0)
                _dedup_push_candidate(cands, {
                    "zone_id": f"TF_PB_FVG_{tf_fvg}_{direction}_{fv['time']}",
                    "setup_type": "TREND_PULLBACK",
                    "source_kind": f"FVG_{tf_fvg}",
                    "timeframe": tf_fvg,
                    "formation_time": fv["time"],
                    "zone_type": ztype,
                    "direction": direction,
                    "price_low": float(fv["low"]),
                    "price_high": float(fv["high"]),
                    "invalidation_price": float(fv["invalidation_price"]),
                    "grade": "A" if tf_fvg in ("H4", "H1") else "B",
                    "score": base_sc,
                    "priority": 85.0 + base_sc - dist_f * 0.5,
                    "label": f"{tf_fvg} Fair Value Gap ({fv['status']})",
                })

    #    1c) SNR Key Support / Resistance Aligned Pullback Area
    snr_usable = _is_snr_ctx_usable(snr_context)
    snr_list = (snr_context or {}).get("supports" if direction == "BUY" else "resistances") or [] if snr_usable else []
    for sz in snr_list[:3]:
        s_lo, s_hi = float(sz.get("low", 0.0)), float(sz.get("high", 0.0))
        if s_lo <= 0 or s_hi < s_lo:
            continue
        dist_s = 0.0 if s_lo <= current_price <= s_hi else min(abs(current_price - s_lo), abs(current_price - s_hi))
        if dist_s <= max(15.0, 3.5 * atr_h1):
            inv = round(s_lo - max(0.8, 0.45 * atr_m30), 2) if direction == "BUY" else round(s_hi + max(0.8, 0.45 * atr_m30), 2)
            _dedup_push_candidate(cands, {
                "zone_id": f"TF_PB_{sz['zone_id']}" if sz.get("zone_id") else f"TF_PB_SNR_{direction}_{s_lo:.2f}_{s_hi:.2f}",
                "source_zone_id": sz.get("zone_id"),
                "setup_type": "TREND_PULLBACK",
                "source_kind": "SNR_ZONE",
                "timeframe": sz.get("timeframe", "H1"),
                "formation_time": sz.get("confirmed_time"),
                "formation_time_tz": sz.get("time_tz"),
                "zone_type": ztype,
                "direction": direction,
                "price_low": round(s_lo, 2),
                "price_high": round(s_hi, 2),
                "invalidation_price": inv,
                "grade": sz.get("grade", "A"),
                "score": float(sz.get("score", 74.0)),
                "priority": 75.0 + float(sz.get("score", 74.0)) - dist_s * 0.5,
                "label": f"SNR {sz.get('timeframe', 'H1')} {'Support' if direction == 'BUY' else 'Resistance'} Pullback",
            })

    #    1d) Dynamic M30/H1 Impulse OTE / EMA20-50 Pullback Pocket (saat tidak ada OB+FVG tetapi leg tren jelas)
    imp = (m30_info or {}).get("impulse_leg") or (h1_info or {}).get("impulse_leg")
    if imp and TF_PULLBACK_MIN_RETRACE <= imp["retrace_ratio"] <= TF_PULLBACK_MAX_RETRACE and df_m30 is not None:
        last30 = df_m30.iloc[-1]
        ema20 = _safe_float(last30.get("EMA20"), current_price)
        ema50 = _safe_float(last30.get("EMA50"), current_price)
        half_w = min(1.25, max(0.60, 0.45 * atr_m30))
        if direction == "BUY":
            # equilibrium & EMA pocket
            eq_mid = float(imp["extreme_price"]) - 0.50 * float(imp["span"])
            poc_center = (eq_mid + ema20) * 0.5 if abs(eq_mid - ema20) <= 3.0 * atr_m30 else current_price
            lo_p = round(min(poc_center, current_price) - half_w * 0.6, 2)
            hi_p = round(lo_p + min(ENTRY_LEBAR_MAKS, 2.0 * half_w), 2)
            inv_p = round(max(float(imp["origin_price"]) - 0.3 * atr_m30, lo_p - max(3.0, 1.5 * atr_m30)), 2)
        else:
            eq_mid = float(imp["extreme_price"]) + 0.50 * float(imp["span"])
            poc_center = (eq_mid + ema20) * 0.5 if abs(eq_mid - ema20) <= 3.0 * atr_m30 else current_price
            hi_p = round(max(poc_center, current_price) + half_w * 0.6, 2)
            lo_p = round(hi_p - min(ENTRY_LEBAR_MAKS, 2.0 * half_w), 2)
            inv_p = round(min(float(imp["origin_price"]) + 0.3 * atr_m30, hi_p + max(3.0, 1.5 * atr_m30)), 2)
        dist_p = 0.0 if lo_p <= current_price <= hi_p else min(abs(current_price - lo_p), abs(current_price - hi_p))
        if dist_p <= max(6.0, 2.5 * atr_m30):
            _dedup_push_candidate(cands, {
                "zone_id": f"TF_PB_DYNAMIC_{direction}_{lo_p:.2f}_{hi_p:.2f}",
                "setup_type": "TREND_PULLBACK",
                "source_kind": "DYNAMIC_OTE_EMA",
                "timeframe": "M30",
                "zone_type": ztype,
                "direction": direction,
                "price_low": lo_p,
                "price_high": hi_p,
                "invalidation_price": inv_p,
                "grade": "B",
                "score": 72.0,
                "priority": 68.0 + imp.get("span_atr", 1.5) * 3.0 - dist_p,
                "label": f"Dynamic M30 Pullback Pocket (Retrace {imp['retrace_ratio']*100:.0f}% / EMA20-50)",
            })

    # 2. BREAKOUT_RETEST Candidates (SNR Verified Breakout+Retest + M30/H1 Swing Flip Close Breakout):
    if snr_usable:
        bo_code = snr_context.get("breakout_code", "")
        bo_dir = snr_context.get("breakout_direction", "")
        rt_code = snr_context.get("retest_code", "")
        rt_dir = snr_context.get("retest_direction", "")
        bz = snr_context.get("retest_zone") or snr_context.get("breakout_zone")
        if bz and bo_code == "VALID_BREAKOUT" and bo_dir == direction and rt_code not in ("FAILED_RETEST",):
            b_lo, b_hi = float(bz.get("low", 0.0)), float(bz.get("high", 0.0))
            if 0 < b_lo <= b_hi:
                inv = round(b_lo - max(1.0, 0.6 * atr_m5), 2) if direction == "BUY" else round(b_hi + max(1.0, 0.6 * atr_m5), 2)
                _dedup_push_candidate(cands, {
                    "zone_id": f"TF_BR_{bz.get('zone_id') or f'{direction}_{b_lo:.2f}_{b_hi:.2f}'}",
                    "setup_type": "BREAKOUT_RETEST",
                    "source_kind": "SNR_BREAKOUT_RETEST",
                    "timeframe": bz.get("timeframe", "H1"),
                    "zone_type": ztype,
                    "direction": direction,
                    "price_low": round(b_lo, 2),
                    "price_high": round(b_hi, 2),
                    "invalidation_price": inv,
                    "grade": "A",
                    "score": 82.0,
                    "priority": 96.0,
                    "label": f"SNR Breakout Retest @ {b_lo:.2f}-{b_hi:.2f} ({rt_code})",
                })

    #    2b) Structural Swing Breakout Retest on H1 / M30 (tanpa bergantung pada SNR bila SNR kosong)
    for tf_br, df_br, atr_br in (("H1", df_h1, atr_h1), ("M30", df_m30, atr_m30)):
        if df_br is None or len(df_br) < 25:
            continue
        evs_br, _, swings_br = _detect_structure(df_br, n=_SWING_N)
        matching_bos = [e for e in evs_br[-4:] if e["arah"] == ("bull" if direction == "BUY" else "bear") and (len(df_br) - 1 - e["k"]) <= 14]
        for ev in reversed(matching_bos):
            lvl = float(ev["level"])
            k_ev = int(ev["k"])
            c_br = float(df_br["Close"].iloc[k_ev])
            o_br = float(df_br["Open"].iloc[k_ev])
            # Pastikan breakout benar-benar closed kuat (bukan false breakout wick!)
            body_past = (c_br - lvl) if direction == "BUY" else (lvl - c_br)
            if body_past < TF_BREAKOUT_MIN_BODY_ATR * atr_br:
                continue
            # Cek apakah candle setelah breakout bertahan di sisi breakout & mendekati retest level
            post_closes = df_br["Close"].iloc[k_ev + 1 :]
            if len(post_closes) > 0:
                if direction == "BUY" and (post_closes < lvl - 0.5 * atr_br).any():
                    continue  # failed retest / false breakout!
                if direction == "SELL" and (post_closes > lvl + 0.5 * atr_br).any():
                    continue
            tol_lvl = min(1.2, max(0.5, 0.45 * atr_br))
            z_lo, z_hi = round(lvl - tol_lvl, 2), round(lvl + tol_lvl, 2)
            dist_lvl = 0.0 if z_lo <= current_price <= z_hi else min(abs(current_price - z_lo), abs(current_price - z_hi))
            if dist_lvl <= max(4.5, 1.8 * atr_br):
                inv = round(z_lo - max(1.0, 0.6 * atr_m30), 2) if direction == "BUY" else round(z_hi + max(1.0, 0.6 * atr_m30), 2)
                _dedup_push_candidate(cands, {
                    "zone_id": f"TF_BR_{tf_br}_{direction}_{df_br.index[k_ev]}",
                    "setup_type": "BREAKOUT_RETEST",
                    "source_kind": f"SWING_BREAKOUT_RETEST_{tf_br}",
                    "timeframe": tf_br,
                    "formation_time": str(df_br.index[k_ev]),
                    "zone_type": ztype,
                    "direction": direction,
                    "price_low": z_lo,
                    "price_high": z_hi,
                    "invalidation_price": inv,
                    "grade": "A" if tf_br == "H1" else "B",
                    "score": 80.0 if tf_br == "H1" else 74.0,
                    "priority": 90.0 - dist_lvl,
                    "label": f"{tf_br} Structural Breakout Retest @ {lvl:.2f}",
                })
                break

    # 3. TREND_CONTINUATION Candidates (saat H4 & H1 aligned/valid dan M30 tidak CONFLICT/WEAK):
    h1_ok_cont = (h1_info or {}).get("status") == "ALIGNED" or (m30_info or {}).get("condition") in ("ALIGNED", "CONSOLIDATION_BREAKOUT", "CONSOLIDATION")
    if h1_ok_cont and (m30_info or {}).get("status") not in ("CONFLICT", "INVALID", "WEAK") and df_m5 is not None and len(df_m5) >= 20:
        evs_m5, _, _ = _detect_structure(df_m5, n=_SWING_N)
        want_m5 = "bull" if direction == "BUY" else "bear"
        recent_m5_bos = [e for e in evs_m5 if e["arah"] == want_m5 and (len(df_m5) - 1 - e["k"]) <= _M5_EVENT_MAX_AGE]
        # Bangun shelf / M5 FVG atau continuation shelf di sekitar harga breakout / pullback kecil terbaru
        cont_fvg = [fv for fv in _detect_fvgs(df_m5, "M5", min_atr_ratio=0.15) if fv["direction"] == direction]
        half_c = min(1.2, max(0.55, 0.75 * atr_m5))
        if cont_fvg:
            cf = cont_fvg[-1]
            c_lo, c_hi = float(cf["low"]), float(cf["high"])
        else:
            c_now_m5 = float(df_m5["Close"].iloc[-1])
            c_lo = round(current_price - half_c, 2) if direction == "BUY" else round(current_price - 0.3 * half_c, 2)
            c_hi = round(current_price + 0.3 * half_c, 2) if direction == "BUY" else round(current_price + half_c, 2)
            if abs(c_now_m5 - current_price) > 2.5 * atr_m5:
                c_lo = c_hi = 0.0
        if c_lo > 0 and c_hi >= c_lo and (recent_m5_bos or (m30_info or {}).get("condition") == "CONSOLIDATION_BREAKOUT"):
            # Anti-overextension guard: hindari kejar harga terlalu jauh dari EMA20 M30 (> 2.2 ATR M30)
            overext_ok = True
            if df_m30 is not None and len(df_m30) >= 20 and "EMA20" in df_m30.columns:
                ema20_30 = _safe_float(df_m30["EMA20"].iloc[-1], current_price)
                if (direction == "BUY" and (current_price - ema20_30) > 2.2 * atr_m30) or (direction == "SELL" and (ema20_30 - current_price) > 2.2 * atr_m30):
                    overext_ok = False
            if overext_ok:
                sw_m5 = _detect_swings(df_m5, n=_SWING_N)
                if direction == "BUY":
                    recent_lows = [s["price"] for s in sw_m5 if s["type"] == "L" and s["price"] < current_price]
                    anchor_sl = recent_lows[-1] if recent_lows else float(df_m5["Low"].iloc[-8:].min())
                    inv_c = round(min(c_lo, anchor_sl) - max(0.5, 0.4 * atr_m5), 2)
                else:
                    recent_highs = [s["price"] for s in sw_m5 if s["type"] == "H" and s["price"] > current_price]
                    anchor_sl = recent_highs[-1] if recent_highs else float(df_m5["High"].iloc[-8:].max())
                    inv_c = round(max(c_hi, anchor_sl) + max(0.5, 0.4 * atr_m5), 2)
                _dedup_push_candidate(cands, {
                    "zone_id": f"TF_CONT_{direction}_{c_lo:.2f}_{c_hi:.2f}",
                    "setup_type": "TREND_CONTINUATION",
                    "source_kind": "M30_M5_CONTINUATION",
                    "timeframe": "M30",
                    "zone_type": ztype,
                    "direction": direction,
                    "price_low": round(c_lo, 2),
                    "price_high": round(c_hi, 2),
                    "invalidation_price": inv_c,
                    "grade": "A" if (h1_info or {}).get("status") == "ALIGNED" else "B",
                    "score": 78.0 if (h1_info or {}).get("status") == "ALIGNED" else 71.0,
                    "priority": 82.0,
                    "label": f"Trend Continuation H4/H1/M30 ({(m30_info or {}).get('condition', 'ALIGNED')})",
                })

    cands.sort(key=lambda c: -float(c.get("priority", 0.0)))
    return cands


def build_strict_reversal_candidate(
    frames: Dict[str, pd.DataFrame],
    h4_info: Dict[str, Any],
    current_price: float,
) -> Optional[Dict[str, Any]]:
    """
    Reversal STRICT ONLY:
    Hanya mengizinkan setup Reversal apabila ada Liquidity Sweep pada ekstrem H1/M30,
    CHoCH H1 & M30 terkonfirmasi candle-close ke arah baru, dan displacement kuat.
    """
    h4_bias = str((h4_info or {}).get("bias") or "UNCONFIRMED").upper()
    if h4_bias not in ("BULLISH", "BEARISH"):
        return None
    rev_dir = "SELL" if h4_bias == "BULLISH" else "BUY"
    rev_ev_dir = "bear" if rev_dir == "SELL" else "bull"
    rev_sw_type = "SWEEP_HIGH" if rev_dir == "SELL" else "SWEEP_LOW"
    df_h1 = _ensure_indicators(frames.get("H1"))
    df_m30 = _ensure_indicators(frames.get("M30"))
    if df_h1 is None or df_m30 is None or len(df_h1) < 25 or len(df_m30) < 25:
        return None

    evs_h1, _, sw_h1 = _detect_structure(df_h1, n=_SWING_N)
    evs_m30, _, sw_m30 = _detect_structure(df_m30, n=_SWING_N)
    has_h1_choch = any(e["arah"] == rev_ev_dir and e["tipe"] == "CHoCH" and len(df_h1) - 1 - e["k"] <= 6 for e in evs_h1)
    has_m30_choch = any(e["arah"] == rev_ev_dir and e["tipe"] == "CHoCH" and len(df_m30) - 1 - e["k"] <= 10 for e in evs_m30)
    if not (has_h1_choch and has_m30_choch):
        return None

    sweeps_h1 = _detect_sweeps(df_h1, sw_h1, lookback=15)
    sweeps_m30 = _detect_sweeps(df_m30, sw_m30, lookback=20)
    has_htf_sweep = any(s["type"] == rev_sw_type for s in (sweeps_h1 + sweeps_m30))
    if not has_htf_sweep:
        return None

    atr30 = max(0.5, _safe_float(df_m30["ATR14"].iloc[-1], 2.0))
    if rev_dir == "BUY":
        disp_m30 = max(float(df_m30["Close"].iloc[k] - df_m30["Open"].iloc[k]) for k in range(-5, 0)) / atr30
        sw_l = [s["price"] for s in sw_m30 if s["type"] == "L"]
        inval = round((sw_l[-1] if sw_l else current_price - 2.0 * atr30) - 0.5 * atr30, 2)
    else:
        disp_m30 = max(float(df_m30["Open"].iloc[k] - df_m30["Close"].iloc[k]) for k in range(-5, 0)) / atr30
        sw_h = [s["price"] for s in sw_m30 if s["type"] == "H"]
        inval = round((sw_h[-1] if sw_h else current_price + 2.0 * atr30) + 0.5 * atr30, 2)

    if disp_m30 < 1.1:
        return None

    half = min(1.2, max(0.5, 0.5 * atr30))
    return {
        "zone_id": f"TF_REV_{rev_dir}_{int(round(current_price))}",
        "setup_type": "REVERSAL",
        "source_kind": "STRICT_HTF_REVERSAL",
        "timeframe": "H1/M30",
        "zone_type": "DEMAND" if rev_dir == "BUY" else "SUPPLY",
        "direction": rev_dir,
        "price_low": round(current_price - half, 2),
        "price_high": round(current_price + half, 2),
        "invalidation_price": inval,
        "grade": "B",
        "score": 74.0,
        "priority": 60.0,
        "label": f"Strict HTF Reversal (H1+M30 CHoCH + Sweep + Disp {disp_m30:.2f} ATR)",
    }


# =============================================================================
# VALIDASI ENTRY M5 + RISK/REWARD PER TIPE SETUP (CLOSED CANDLES ONLY)
# =============================================================================
def _is_snr_zone_broken_or_inactive(
    zone: Dict[str, Any],
    direction: str,
    snr_context: Optional[Dict[str, Any]] = None,
) -> bool:
    """True bila zona SNR lawan sudah breakout/invalid/mitigated/non-aktif sehingga bukan penghalang aktif."""
    lifecycle = str(zone.get("lifecycle") or "").upper()
    if lifecycle in ("BROKEN", "INVALIDATED", "EXPIRED", "MITIGATED", "RETEST", "RECLAIM_PENDING", "RECLAIMED"):
        return True
    if zone.get("is_broken") or zone.get("is_invalidated") or (zone.get("is_active") is False):
        return True
    bo_info = zone.get("breakout_info") if isinstance(zone.get("breakout_info"), dict) else {}
    if str(bo_info.get("status") or "").upper() == "VALID_BREAKOUT":
        return True
    if isinstance(snr_context, dict):
        want_bo_dirs = ("BUY", "BULLISH") if direction == "BUY" else ("SELL", "BEARISH")
        if (
            str(snr_context.get("breakout_code") or "").upper() == "VALID_BREAKOUT"
            and str(snr_context.get("breakout_direction") or "").upper() in want_bo_dirs
        ):
            bo_z = snr_context.get("breakout_zone")
            if isinstance(bo_z, dict):
                z_id = str(zone.get("zone_id") or "").strip()
                bo_id = str(bo_z.get("zone_id") or "").strip()
                if z_id and bo_id and z_id == bo_id:
                    return True
                z_lo = _safe_float(zone.get("low"), -1.0)
                z_hi = _safe_float(zone.get("high"), -1.0)
                bo_lo = _safe_float(bo_z.get("low"), -2.0)
                bo_hi = _safe_float(bo_z.get("high"), -2.0)
                if z_lo > 0 and z_hi > 0 and abs(z_lo - bo_lo) < 1e-6 and abs(z_hi - bo_hi) < 1e-6:
                    return True
    return False


def evaluate_snr_opposing_barrier(
    direction: str,
    entry: float,
    sl: float,
    tp1: Optional[float] = None,
    current_price: Optional[float] = None,
    snr_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Evaluasi apakah zona Support/Resistance SNR lawan menjadi penghalang material
    bagi rencana entry SMC berdasarkan arah transaksi, posisi entry/harga saat ini,
    kekuatan zona, status lifecycle/breakout, serta jarak bersih ke TP1 / R:R minimum.
    """
    clean = {
        "blocked": False,
        "reason": "",
        "barrier_zone": None,
        "clean_rr": None,
        "conflict_type": "NONE",
    }
    dir_u = str(direction or "").upper()
    if dir_u not in ("BUY", "SELL") or entry <= 0 or not _is_snr_ctx_usable(snr_context):
        return clean

    risk = round(abs(float(entry) - float(sl)), 2)
    opp_list = list(snr_context.get("resistances" if dir_u == "BUY" else "supports") or [])
    if not opp_list:
        fallback_z = snr_context.get("resistance" if dir_u == "BUY" else "support")
        if isinstance(fallback_z, dict):
            opp_list = [fallback_z]

    grade_rank = {"A+": 5, "A": 4, "B": 3, "C": 2, "D": 1}
    best_block: Optional[Dict[str, Any]] = None
    best_priority = (-1, -1, 0.0)

    for z in opp_list:
        if not isinstance(z, dict):
            continue
        if _is_snr_zone_broken_or_inactive(z, dir_u, snr_context=snr_context):
            continue

        z_lo = _safe_float(z.get("low"), 0.0)
        z_hi = _safe_float(z.get("high"), 0.0)
        if z_lo <= 0 or z_hi <= 0:
            continue
        if z_lo > z_hi:
            z_lo, z_hi = z_hi, z_lo

        grade = str(z.get("grade") or "").upper()
        score = _safe_float(z.get("score"), 0.0)
        strength = str(z.get("strength_short") or z.get("strength") or "").upper()
        is_major = (
            grade in ("A+", "A", "B")
            or score >= 70.0
            or ("STRONG" in strength)
            or ("KUAT" in strength)
        )
        is_material = (
            is_major
            or grade == "C"
            or score >= 60.0
            or ("MODERATE" in strength)
            or ("SEDANG" in strength)
        )
        if not is_material:
            continue

        label = str(z.get("label") or z.get("zone_id") or ("Resistance" if dir_u == "BUY" else "Support"))
        tf_lbl = str(z.get("timeframe") or "H1")
        g_lbl = f" Grade {grade}" if grade else ""
        near_edge = z_lo if dir_u == "BUY" else z_hi
        px = float(current_price) if current_price is not None and current_price > 0 else float(entry)

        # A. Entry atau harga aktif berada langsung di dalam zona SNR lawan yang belum tembus
        if (z_lo <= entry <= z_hi) or (z_lo <= px <= z_hi):
            reason = (
                f"Konflik SMC-SNR: entry {dir_u} @ {entry:.2f} berada di dalam zona lawan aktif "
                f"{label} ({tf_lbl}{g_lbl} {z_lo:.2f}-{z_hi:.2f}) yang belum breakout"
            )
            prio = (2, grade_rank.get(grade, 0), score)
            if prio > best_priority:
                best_priority = prio
                best_block = {
                    "blocked": True,
                    "reason": reason,
                    "barrier_zone": z,
                    "clean_rr": 0.0,
                    "conflict_type": "ENTRY_INSIDE_OPPOSING_SNR",
                }
            continue

        # B. Penghalang SNR mayor di depan entry memotong ruang bersih ke bawah RR_MIN
        is_ahead = (dir_u == "BUY" and near_edge > entry) or (dir_u == "SELL" and near_edge < entry)
        if is_ahead and is_major and risk > 0:
            clean_reward = (near_edge - entry) if dir_u == "BUY" else (entry - near_edge)
            clean_rr = round(clean_reward / risk, 2)
            crosses_barrier = (
                (tp1 is None)
                or (dir_u == "BUY" and float(tp1) > near_edge)
                or (dir_u == "SELL" and float(tp1) < near_edge)
            )
            if crosses_barrier and clean_rr < RR_MIN - 1e-9:
                reason = (
                    f"Konflik SMC-SNR: target {dir_u} terhalang {label} "
                    f"({tf_lbl}{g_lbl} {z_lo:.2f}-{z_hi:.2f}); "
                    f"ruang bersih sebelum SNR hanya 1:{clean_rr:.2f} < 1:{RR_MIN:g}"
                )
                prio = (1, grade_rank.get(grade, 0), score)
                if prio > best_priority:
                    best_priority = prio
                    best_block = {
                        "blocked": True,
                        "reason": reason,
                        "barrier_zone": z,
                        "clean_rr": clean_rr,
                        "conflict_type": "TARGET_BLOCKED_BY_SNR",
                    }

    return best_block if best_block is not None else clean


def _compute_targets_and_rr(
    frames: Dict[str, pd.DataFrame],
    direction: str,
    entry: float,
    sl: float,
    setup_type: str,
    snr_context: Optional[Dict[str, Any]] = None,
    current_price: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Hitung TP1 & TP2 berdasarkan struktur H1/H4/M30 dan S&R aktif:
      - Memastikan tidak ada obstacle mayor S&R/swing H1/H4 yang menghalangi TP1 dengan R:R < 1.5,
        termasuk entry yang berada langsung di dalam zona SNR lawan aktif.
      - Jika target struktural tidak tersedia, tunggu tanpa membuat target proyeksi.
    """
    risk = round(abs(entry - sl), 2)
    if risk <= 0 or risk > SL_JARAK_MAKS:
        return {"ok": False, "rr": 0.0, "tp1": None, "tp2": None,
                "reason": f"Jarak SL {risk:.2f} pt di luar batas aman (maks {SL_JARAK_MAKS:g} pt)"}

    pre_barrier = evaluate_snr_opposing_barrier(
        direction=direction,
        entry=entry,
        sl=sl,
        tp1=None,
        current_price=current_price,
        snr_context=snr_context,
    )
    if pre_barrier.get("blocked"):
        bz = pre_barrier.get("barrier_zone") or {}
        bz_edge = (
            _safe_float(bz.get("low"), entry) if direction == "BUY" else _safe_float(bz.get("high"), entry)
        )
        return {
            "ok": False,
            "rr": float(pre_barrier.get("clean_rr") or 0.0),
            "tp1": round(bz_edge, 2) if bz_edge > 0 else None,
            "tp2": None,
            "snr_blocked": True,
            "snr_conflict_type": pre_barrier.get("conflict_type"),
            "reason": pre_barrier["reason"],
        }

    raw_targets: List[Tuple[float, str]] = []
    for tf_name in ("H4", "H1", "M30"):
        df_t = frames.get(tf_name)
        if df_t is None or len(df_t) < 2 * _SWING_N + 1:
            continue
        H_arr = df_t["High"].to_numpy(dtype=float)
        L_arr = df_t["Low"].to_numpy(dtype=float)
        for sw in _detect_swings(df_t, n=_SWING_N):
            i, p = int(sw["idx"]), float(sw["price"])
            if direction == "BUY" and sw["type"] == "H" and p > entry:
                if i + 1 >= len(H_arr) or H_arr[i + 1 :].max() <= p:
                    raw_targets.append((round(p - 0.25, 2), f"{tf_name} Swing High"))
            elif direction == "SELL" and sw["type"] == "L" and p < entry:
                if i + 1 >= len(L_arr) or L_arr[i + 1 :].min() >= p:
                    raw_targets.append((round(p + 0.25, 2), f"{tf_name} Swing Low"))

    # Tambahkan level penghalang SNR lawan aktif (Resistance untuk BUY, Support untuk SELL)
    major_obstacles: List[float] = []
    if _is_snr_ctx_usable(snr_context):
        opp_key = "resistances" if direction == "BUY" else "supports"
        for z in snr_context.get(opp_key) or []:
            if not isinstance(z, dict) or _is_snr_zone_broken_or_inactive(z, direction, snr_context=snr_context):
                continue
            edge = float(z.get("low") if direction == "BUY" else z.get("high"))
            if (direction == "BUY" and edge > entry) or (direction == "SELL" and edge < entry):
                raw_targets.append((round(edge, 2), f"SNR {z.get('timeframe', 'H1')}"))
                if float(z.get("score", 0.0)) >= 70.0 or str(z.get("grade", "")) in ("A+", "A"):
                    major_obstacles.append(edge)

    # Sort dari terdekat
    raw_targets.sort(key=lambda x: x[0], reverse=(direction == "SELL"))

    # Cek apakah ada major S&R obstacle tepat di depan yang membuat R:R < RR_MIN
    if major_obstacles:
        nearest_obs = min(major_obstacles) if direction == "BUY" else max(major_obstacles)
        obs_reward = (nearest_obs - entry) if direction == "BUY" else (entry - nearest_obs)
        obs_rr = obs_reward / risk if risk > 0 else 0.0
        if obs_rr < RR_MIN - 1e-9:
            return {"ok": False, "rr": round(obs_rr, 2), "tp1": round(nearest_obs, 2), "tp2": None,
                    "snr_blocked": True,
                    "reason": f"Terhalang zona S&R mayor di {nearest_obs:.2f} (R:R aktual 1:{obs_rr:.2f} < 1:{RR_MIN:g})"}

    if raw_targets:
        nearest_reward = (raw_targets[0][0] - entry) if direction == "BUY" else (entry - raw_targets[0][0])
        if nearest_reward / risk < RR_MIN - 1e-9:
            return {"ok": False, "rr": round(nearest_reward / risk, 2), "tp1": raw_targets[0][0], "tp2": None,
                    "reason": f"R:R aktual ke target terdekat {raw_targets[0][0]:.2f} hanya 1:{nearest_reward / risk:.2f} (< 1:{RR_MIN:g})"}

    valid_tps = [
        (t, src) for (t, src) in raw_targets
        if (((t - entry) if direction == "BUY" else (entry - t)) / risk) >= RR_MIN - 1e-9
    ]

    if valid_tps:
        tp1, tp1_src = valid_tps[0]
        tp2 = valid_tps[1][0] if len(valid_tps) > 1 else tp1
    elif raw_targets:
        # Hanya ada target struktural lebih dekat dan tidak memenuhi R:R minimum -> tolak!
        first_t = raw_targets[0][0]
        rew_0 = (first_t - entry) if direction == "BUY" else (entry - first_t)
        rr_0 = round(rew_0 / risk, 2)
        return {"ok": False, "rr": rr_0, "tp1": first_t, "tp2": None,
                "reason": f"R:R aktual ke target terdekat {first_t:.2f} hanya 1:{rr_0:.2f} (< 1:{RR_MIN:g})"}
    else:
        return {"ok": False, "rr": 0.0, "tp1": None, "tp2": None,
                "reason": "Tidak ada target likuiditas struktural searah"}

    post_barrier = evaluate_snr_opposing_barrier(
        direction=direction,
        entry=entry,
        sl=sl,
        tp1=tp1,
        current_price=current_price,
        snr_context=snr_context,
    )
    if post_barrier.get("blocked"):
        return {
            "ok": False,
            "rr": float(post_barrier.get("clean_rr") or 0.0),
            "tp1": tp1,
            "tp2": None,
            "snr_blocked": True,
            "snr_conflict_type": post_barrier.get("conflict_type"),
            "reason": post_barrier["reason"],
        }

    reward = abs(tp1 - entry)
    rr = round(reward / risk, 2) if risk > 0 else 0.0
    if rr < RR_MIN - 1e-9:
        return {"ok": False, "rr": rr, "tp1": tp1, "tp2": tp2,
                "reason": f"R:R aktual 1:{rr:.2f} di bawah minimum 1:{RR_MIN:g}"}

    return {"ok": True, "rr": rr, "tp1": tp1, "tp2": tp2, "risk": risk, "target_source": tp1_src}


def validate_m5_entry_confirmation(
    candidate: Dict[str, Any],
    frames: Dict[str, pd.DataFrame],
    price: float,
    spread: float = 0.0,
    news_blackout: Any = False,
    snr_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Validasi entry M5 spesifik untuk masing-masing tipe setup (CLOSED CANDLES ONLY).
    Mengembalikan dict dengan status ('VALID'|'WAIT'|'INVALID'), stages, m5, plan, reasons.
    """
    stages = {
        "zone_validation": {"status": "NOT_CHECKED", "msg": ""},
        "zone_touch": {"status": "NOT_CHECKED", "msg": ""},
        "liquidity_sweep": {"status": "NOT_CHECKED", "msg": "", "pool": None},
        "structure_trigger": {"status": "NOT_CHECKED", "msg": ""},
        "displacement": {"status": "NOT_CHECKED", "msg": "", "atr": 0.0},
        "entry_validation": {"status": "WAIT", "msg": ""},
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

    direction = str(candidate.get("direction") or "").upper()
    setup_type = str(candidate.get("setup_type") or "TREND_PULLBACK").upper()
    lo = float(candidate.get("price_low", 0.0))
    hi = float(candidate.get("price_high", 0.0))
    inval = float(candidate.get("invalidation_price", lo if direction == "BUY" else hi))

    if direction not in ("BUY", "SELL") or lo <= 0 or hi < lo:
        stages["zone_validation"] = {"status": "INVALID", "msg": "Kandidat setup tidak valid"}
        res["status"] = "INVALID"
        W.append("Kandidat setup tidak valid")
        return res

    if news_blackout:
        stages["zone_validation"] = {"status": "WAIT", "msg": f"Blackout berita: {news_blackout}"}
        W.append(f"Blackout berita: {news_blackout}")
        return res

    if spread > SPREAD_MAKS_POIN:
        stages["zone_validation"] = {"status": "WAIT", "msg": f"Spread {spread:.2f} pt melebihi batas {SPREAD_MAKS_POIN:g}"}
        W.append(f"Spread {spread:.2f} pt melebihi batas {SPREAD_MAKS_POIN:g}")
        return res

    df_m5_raw = frames.get("M5")
    if isinstance(df_m5_raw, pd.DataFrame) and getattr(df_m5_raw, "attrs", {}).get("is_closed") is False:
        stages["zone_validation"] = {"status": "WAIT", "msg": "Candle terakhir M5 belum selesai (masih berjalan)"}
        W.append("Candle terakhir M5 belum selesai (masih berjalan): konfirmasi entry hanya dari candle close")
        return res

    df_m5 = _ensure_indicators(df_m5_raw)
    if df_m5 is None or len(df_m5) < 25:
        stages["zone_validation"] = {"status": "WAIT", "msg": "Data M5 tidak mencukupi"}
        W.append("Data M5 tidak mencukupi")
        return res

    atr5 = max(0.3, _safe_float(df_m5["ATR14"].iloc[-1], 1.2))
    c_last = float(df_m5["Close"].iloc[-1])
    if abs(price - c_last) > max(30.0, 6.0 * atr5):
        stages["zone_validation"] = {"status": "WAIT", "msg": "Harga live menyimpang dari candle close M5"}
        W.append("Harga live menyimpang jauh dari candle close M5")
        return res

    # Anti-invalidasi: harga tidak boleh melewati invalidasi
    if (direction == "BUY" and price < inval) or (direction == "SELL" and price > inval):
        stages["zone_validation"] = {"status": "INVALID", "msg": f"Harga menembus invalidation {inval:.2f}"}
        res["status"] = "INVALID"
        W.append(f"Harga {price:.2f} menembus invalidation {inval:.2f}")
        return res

    # Anti-extreme counter impulse pada candle terakhir M5
    O = df_m5["Open"].to_numpy(dtype=float)
    H = df_m5["High"].to_numpy(dtype=float)
    L = df_m5["Low"].to_numpy(dtype=float)
    C = df_m5["Close"].to_numpy(dtype=float)
    n5 = len(df_m5)
    opp_body = (O[-1] - C[-1]) if direction == "BUY" else (C[-1] - O[-1])
    if opp_body >= TF_OPPOSING_IMPULSE_BLOCK_ATR * atr5:
        stages["zone_validation"] = {"status": "WAIT", "msg": f"Terdapat impuls berlawanan tajam M5 ({opp_body/atr5:.2f} ATR)"}
        W.append(stages["zone_validation"]["msg"])
        return res

    # Cek anti-false-breakout dari SNR saat BREAKOUT_RETEST
    if setup_type == "BREAKOUT_RETEST" and _is_snr_ctx_usable(snr_context):
        bo_c = snr_context.get("breakout_code", "")
        rt_c = snr_context.get("retest_code", "")
        if bo_c == "FALSE_BREAKOUT" or rt_c == "FAILED_RETEST":
            stages["zone_validation"] = {"status": "WAIT", "msg": f"SNR mendeteksi {bo_c}/{rt_c} — setup retest ditolak"}
            W.append(stages["zone_validation"]["msg"])
            return res

    stages["zone_validation"] = {"status": "CONFIRMED", "msg": f"{setup_type}: {candidate.get('label', direction)}"}

    # 2. Pengecekan Sentuhan Area Setup & Anti-Late-Entry
    touched = False
    touch_indices = []
    formation_time = candidate.get("formation_time")
    formation_close = None
    if formation_time:
        formation_tf = str(candidate.get("formation_timeframe") or candidate.get("timeframe", "")).upper()
        formation_df = frames.get(formation_tf)
        formation_tz = candidate.get("formation_time_tz") or (getattr(formation_df, "attrs", {}) or {}).get("time_tz", "WIB")
        formation_dt = to_wib_datetime(formation_time, source_tz=formation_tz)
        duration = {"M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240}.get(formation_tf)
        if formation_dt is None or duration is None:
            stages["zone_validation"] = {"status": "DATA_INCOMPLETE", "msg": "Waktu konfirmasi zona tidak valid."}
            W.append(stages["zone_validation"]["msg"])
            return res
        formation_close = pd.Timestamp(formation_dt) + pd.Timedelta(minutes=duration)
    for x in range(max(0, n5 - _M5_TOUCH_LOOKBACK), n5):
        m5_time = to_wib_datetime(df_m5.index[x], source_tz=str((getattr(df_m5, "attrs", {}) or {}).get("time_tz", "WIB")))
        if formation_close is not None and (m5_time is None or pd.Timestamp(m5_time) < formation_close):
            continue
        if L[x] <= hi and H[x] >= lo:
            touched = True
            touch_indices.append(x)
    if not touched and (lo <= price <= hi):
        touched = True

    dist = 0.0 if lo <= price <= hi else (lo - price if price < lo else price - hi)
    approach_tol = max(3.5, 2.0 * atr5) if setup_type in ("TREND_CONTINUATION", "BREAKOUT_RETEST") else max(3.0, 1.5 * atr5)

    if not touched and setup_type != "TREND_CONTINUATION":
        stages["zone_touch"] = {"status": "WAIT", "msg": f"Harga belum mencapai area {setup_type} ({lo:.2f}-{hi:.2f})"}
        W.append(stages["zone_touch"]["msg"])
        return res

    if dist > min(ENTRY_JARAK_MAKS, approach_tol):
        stages["zone_touch"] = {"status": "FAIL", "msg": f"Harga sudah menjauh dari area setup ({dist:.2f} > {approach_tol:.2f} pt)"}
        W.append(stages["zone_touch"]["msg"])
        return res

    if touched and lo <= price <= hi and not touch_indices:
        first_touch_k = n5
    elif touch_indices:
        first_touch_k = touch_indices[0]
    else:
        first_touch_k = 0
    stages["zone_touch"] = {
        "status": "CONFIRMED" if touched else "NOT_REQUIRED",
        "msg": f"Harga berada di area {setup_type} (jarak {dist:.2f} pt)" if touched else "Touch tidak diwajibkan untuk trend continuation",
    }

    # 3. Analisis M5 Structure (BOS/CHoCH), CISD, Liquidity Sweep, Rejection, Displacement
    events, _, swings = _detect_structure(df_m5, n=_SWING_N)
    sweeps = _detect_sweeps(df_m5, swings, lookback=35)
    want_ev = "bull" if direction == "BUY" else "bear"
    want_sw = "SWEEP_LOW" if direction == "BUY" else "SWEEP_HIGH"

    tol_sw = max(2.5, 1.6 * atr5)
    rel_sweeps = [
        s for s in sweeps
        if s["type"] == want_sw and first_touch_k <= s["k"] and n5 - 1 - s["k"] <= 35
        and (lo - tol_sw <= s["wick_extreme"] <= hi + tol_sw)
        and (lo - tol_sw <= s["level"] <= hi + tol_sw)
    ]
    best_sweep = rel_sweeps[-1] if rel_sweeps else None
    k_sweep = best_sweep["k"] if best_sweep else -1

    cisd_list: List[Dict[str, Any]] = []
    try:
        from modules.analisis_teknikal import deteksi_cisd
        cisd_list, _ = deteksi_cisd(df_m5, swings, lookback=35)
    except Exception:
        cisd_list = []

    def _has_opposing_m5_event_after(trig_k: int) -> bool:
        return any(e.get("arah") != want_ev and int(e.get("k", -1)) > trig_k for e in events)

    raw_match_evs = [
        e for e in reversed(events)
        if e["arah"] == want_ev and n5 - 1 - e["k"] <= _M5_EVENT_MAX_AGE and e["k"] >= max(first_touch_k, k_sweep)
    ]
    raw_match_cisd = [
        c for c in reversed(cisd_list)
        if c.get("arah") == ("BULLISH" if direction == "BUY" else "BEARISH")
        and n5 - 1 - int(c.get("k", 0)) <= _M5_EVENT_MAX_AGE
        and int(c.get("k", 0)) >= max(first_touch_k, k_sweep)
    ]
    match_evs = [e for e in raw_match_evs if not _has_opposing_m5_event_after(int(e["k"]))]
    match_cisd = [c for c in raw_match_cisd if not _has_opposing_m5_event_after(int(c.get("k", 0)))]
    rej = _detect_rejection_candle(df_m5, direction, lo, hi, atr5, lookback=6)
    if rej and (int(rej["k"]) < first_touch_k or _has_opposing_m5_event_after(int(rej["k"]))):
        rej = None

    # Set stage sweep
    if best_sweep:
        stages["liquidity_sweep"] = {"status": "CONFIRMED", "msg": best_sweep["description"], "pool": want_sw, "k": k_sweep}
    else:
        stages["liquidity_sweep"] = {
            "status": "WAIT" if setup_type == "REVERSAL" else "OPTIONAL_OK" if (rej or setup_type == "TREND_CONTINUATION") else "WAIT",
            "msg": f"Tidak ada {want_sw}" if not rej else f"Diganti konfirmasi {rej['label']} searah tren",
            "pool": None,
        }

    # Evaluasi Structure Trigger & Rejection per tipe setup
    ev = match_evs[0] if match_evs else None
    cisd_ev = match_cisd[0] if match_cisd else None
    has_struct = bool(ev or cisd_ev)
    reversed_after_trig = bool(not has_struct and (raw_match_evs or raw_match_cisd))

    if setup_type == "REVERSAL":
        if not best_sweep or not has_struct:
            msg_st = (
                "Struktur M5 pasca-trigger sudah berbalik arah (sinyal historis tidak berlaku)"
                if (best_sweep and reversed_after_trig)
                else "Reversal wajib Sweep + CHoCH/BOS M5 terkonfirmasi"
            )
            stages["structure_trigger"] = {"status": "WAIT", "msg": msg_st}
            W.append(msg_st)
            return res
    elif setup_type == "TREND_PULLBACK":
        # Trend Pullback: Sweep + CHoCH/BOS/CISD ATAU CHoCH/BOS/CISD + Rejection jelas dari pullback zone
        if not has_struct or (not best_sweep and not rej):
            msg_st = (
                "Struktur M5 pasca-trigger sudah berbalik arah (sinyal historis tidak berlaku)"
                if reversed_after_trig
                else "Menunggu M5 Sweep + CHoCH/BOS/CISD atau CHoCH/BOS + Rejection jelas di area pullback"
            )
            stages["structure_trigger"] = {"status": "WAIT", "msg": msg_st}
            W.append(msg_st)
            return res
    elif setup_type == "BREAKOUT_RETEST":
        # Breakout Retest: minimal Rejection jelas dari level retest ATAU CHoCH/BOS/CISD M5
        if not has_struct and not rej:
            msg_st = (
                "Struktur M5 pasca-trigger sudah berbalik arah (sinyal historis tidak berlaku)"
                if reversed_after_trig
                else "Menunggu Rejection atau CHoCH/BOS M5 pada area Breakout Retest"
            )
            stages["structure_trigger"] = {"status": "WAIT", "msg": msg_st}
            W.append(msg_st)
            return res
    else:  # TREND_CONTINUATION
        if not has_struct:
            msg_st = (
                "Struktur M5 pasca-trigger sudah berbalik arah (sinyal historis tidak berlaku)"
                if reversed_after_trig
                else f"Menunggu M5 BOS/CHoCH/CISD {direction} untuk Trend Continuation"
            )
            stages["structure_trigger"] = {"status": "WAIT", "msg": msg_st}
            W.append(msg_st)
            return res

    k_ref = int(ev["k"]) if ev else (int(cisd_ev["k"]) if cisd_ev else int(rej["k"]))
    m5_tz_tf = str((getattr(df_m5, "attrs", {}) or {}).get("time_tz", "WIB"))
    trig_ts = format_wib_timestamp(df_m5.index[k_ref], default_now=False, source_tz=m5_tz_tf) or str(df_m5.index[k_ref])
    cand_zid = candidate.get("zone_id")
    cand_tf = str(candidate.get("timeframe") or "M30")
    cand_zt = str(candidate.get("zone_type") or ("DEMAND" if direction == "BUY" else "SUPPLY")).upper()
    trig_msg = (
        f"{ev['tipe']} {ev['arah']} @ {ev['level']:.2f}"
        if ev
        else (f"CISD {direction}" if cisd_ev else f"{rej['label']} @ retest")
    )
    if rej and ev:
        trig_msg = f"{trig_msg} + {rej['label']}"
    stages["structure_trigger"] = {
        "status": "CONFIRMED",
        "msg": trig_msg,
        "k": k_ref,
        "zone_id": cand_zid,
        "timeframe": cand_tf,
        "zone_type": cand_zt,
        "zone_range": (round(lo, 2), round(hi, 2)),
        "timestamp": trig_ts,
    }

    # 4. Displacement M5
    k0 = max(first_touch_k, k_ref - 3)
    if direction == "BUY":
        bodies = [C[x] - O[x] for x in range(k0, min(n5, k_ref + 3)) if C[x] > O[x]]
    else:
        bodies = [O[x] - C[x] for x in range(k0, min(n5, k_ref + 3)) if O[x] > C[x]]
    disp_body = max(bodies) if bodies else 0.0
    disp_atr = round(float(disp_body) / atr5, 2)
    min_disp = (
        max(1.0, _M5_MIN_DISP_ATR)
        if setup_type == "REVERSAL"
        else (0.65 if (setup_type == "BREAKOUT_RETEST" and rej is not None) else TF_CONTINUATION_MIN_DISPLACEMENT_ATR)
    )
    if disp_atr < min_disp:
        stages["displacement"] = {"status": "WEAK", "msg": f"Displacement M5 {disp_atr:.2f} ATR < {min_disp:.2f}", "atr": disp_atr}
        W.append(stages["displacement"]["msg"])
        return res
    stages["displacement"] = {
        "status": "CONFIRMED",
        "msg": f"Displacement M5 {disp_atr:.2f} ATR",
        "atr": disp_atr,
        "zone_id": cand_zid,
        "timeframe": cand_tf,
        "timestamp": trig_ts,
    }

    if k_ref + 1 < n5:
        post_c = C[k_ref + 1 : n5]
        post_inval = bool(np.any(post_c < inval)) if direction == "BUY" else bool(np.any(post_c > inval))
        if post_inval:
            msg_inv = f"Trigger M5 historis telah terinvalidasi pasca-konfirmasi ({inval:.2f})"
            stages["structure_trigger"] = {"status": "WAIT", "msg": msg_inv, "k": k_ref}
            W.append(msg_inv)
            return res

    res["smc_technical_confirmed"] = True
    m5_tech_payload = {
        "zone_touched": touched,
        "sweep": bool(best_sweep),
        "structure_event": trig_msg,
        "rejection": rej["label"] if rej else None,
        "displacement_atr": disp_atr,
        "technical_confirmed": True,
        "zone_id": cand_zid,
        "timeframe": cand_tf,
        "zone_type": cand_zt,
        "zone_range": (round(lo, 2), round(hi, 2)),
        "timestamp": trig_ts,
        "confirmed_at": trig_ts,
    }
    res["m5"] = dict(m5_tech_payload)
    res["m5_confirmation"] = {
        "zone_id": cand_zid,
        "timeframe": cand_tf,
        "zone_type": cand_zt,
        "zone_range": (round(lo, 2), round(hi, 2)),
        "timestamp": trig_ts,
        "status": "CONFIRMED",
        "structure_event": trig_msg,
        "displacement_atr": disp_atr,
    }

    bars_since_trig = (n5 - 1) - k_ref
    chase_tol = max(1.0, 0.5 * atr5)
    if bars_since_trig > 2 and (
        (direction == "BUY" and price > hi + chase_tol)
        or (direction == "SELL" and price < lo - chase_tol)
    ):
        msg_missed = (
            f"Sinyal M5 sudah historis/terlewat (trigger {bars_since_trig} bar lalu; "
            f"harga {price:.2f} sudah meninggalkan area entry {lo:.2f}-{hi:.2f})"
        )
        stages["entry_validation"] = {"status": "WAIT", "msg": msg_missed}
        res["entry_permission"] = "WAIT"
        W.append(msg_missed)
        return res

    # 5. Kalkulasi Entry Range, Stop Loss Struktural, dan Target TP / RR
    w_zone = max(0.4, hi - lo)
    if w_zone <= ENTRY_LEBAR_MAKS:
        e_lo, e_hi = round(lo, 2), round(hi, 2)
    else:
        if direction == "BUY":
            e_hi = round(hi, 2)
            e_lo = round(e_hi - ENTRY_LEBAR_MAKS, 2)
        else:
            e_lo = round(lo, 2)
            e_hi = round(e_lo + ENTRY_LEBAR_MAKS, 2)
    entry = round(price if e_lo <= price <= e_hi else (e_lo + e_hi) / 2.0, 2)

    df_m30_ref = frames.get("M30")
    atr_m30_ref = _safe_float(df_m30_ref["ATR14"].iloc[-1] if df_m30_ref is not None and "ATR14" in df_m30_ref.columns else atr5 * 1.5, atr5 * 1.5)
    min_sl_floor = max(1.2, 0.8 * atr5, min(6.0, 0.65 * atr_m30_ref))
    buf = max(0.5 * atr5, spread + 0.5)
    sweep_ext = float(best_sweep["wick_extreme"]) if best_sweep else (float(rej["wick_extreme"]) if rej else None)
    use_local_sl = abs(entry - inval) > 0.80 * SL_JARAK_MAKS or setup_type in ("REVERSAL", "TREND_CONTINUATION")
    if direction == "BUY":
        m5_low_anchor = float(np.min(L[max(0, k_ref - 4) : min(n5, k_ref + 2)]))
        anchors = [m5_low_anchor] if use_local_sl else [inval, m5_low_anchor]
        if sweep_ext is not None:
            anchors.append(sweep_ext)
        sl_raw = min(anchors) - buf
        sl = round(min(sl_raw, entry - min_sl_floor), 2)
    else:
        m5_high_anchor = float(np.max(H[max(0, k_ref - 4) : min(n5, k_ref + 2)]))
        anchors = [m5_high_anchor] if use_local_sl else [inval, m5_high_anchor]
        if sweep_ext is not None:
            anchors.append(sweep_ext)
        sl_raw = max(anchors) + buf
        sl = round(max(sl_raw, entry + min_sl_floor), 2)

    rr_info = _compute_targets_and_rr(
        frames, direction, entry, sl, setup_type, snr_context=snr_context, current_price=price
    )
    if not rr_info.get("ok"):
        stages["risk_reward"] = {"status": "WAIT", "msg": rr_info["reason"], "rr": rr_info.get("rr", 0.0)}
        stages["entry_validation"] = {"status": "WAIT", "msg": rr_info["reason"]}
        res["entry_permission"] = "WAIT"
        res["m5"] = dict(m5_tech_payload)
        W.append(rr_info["reason"])
        return res

    rr = float(rr_info["rr"])
    tp1 = float(rr_info["tp1"])
    tp2 = float(rr_info["tp2"])

    if k_ref + 1 < n5:
        post_tp_hit = (
            bool(np.any(H[k_ref + 1 : n5] >= tp1))
            if direction == "BUY"
            else bool(np.any(L[k_ref + 1 : n5] <= tp1))
        )
        if post_tp_hit:
            msg_tp = f"Target TP1 {tp1:.2f} sudah tersentuh pasca-trigger M5 (sinyal historis/terlewat)"
            stages["risk_reward"] = {"status": "WAIT", "msg": msg_tp, "rr": rr}
            stages["entry_validation"] = {"status": "WAIT", "msg": msg_tp}
            res["entry_permission"] = "WAIT"
            res["m5"] = dict(m5_tech_payload)
            W.append(msg_tp)
            return res

    stages["risk_reward"] = {"status": "CONFIRMED", "msg": f"1:{rr:.2f} ({rr_info['target_source']})", "rr": rr}
    stages["entry_validation"] = {"status": "PASSED", "msg": f"{setup_type} terkonfirmasi penuh"}

    res["status"] = "VALID"
    res["entry_permission"] = "GRANTED"
    res["m5"] = dict(m5_tech_payload)
    res["plan"] = {
        "arah": direction,
        "setup_type": setup_type,
        "source_kind": candidate.get("source_kind"),
        "entry_min": e_lo,
        "entry_max": e_hi,
        "entry_acuan": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "rr": rr,
        "invalidation_price": inval,
        "zone_id": candidate.get("zone_id"),
        "timeframe": candidate.get("timeframe", "M30"),
        "grade": candidate.get("grade", "B"),
        "score": candidate.get("score", 75.0),
        "keterangan": f"{setup_type} ({candidate.get('label', '')}) + M5 {trig_msg}",
    }
    W.append(f"Entry {setup_type} valid: M5 {trig_msg}, Displacement {disp_atr:.2f} ATR, R:R 1:{rr:.2f}")
    return res


# =============================================================================
# ORKESTRATOR ENTRY LOGIC TERPADU (MULTI-TIMEFRAME TREND FOLLOWING + SMC + SNR)
# =============================================================================
def evaluate_trend_following_setup(
    frames: Dict[str, pd.DataFrame],
    current_price: float,
    scan: Optional[Dict[str, Any]] = None,
    spread: float = 0.0,
    news_blackout: Any = False,
    explicit_h1_conflict: bool = False,
    explicit_m30_conflict: bool = False,
    preferred_zone_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Orkestrator utama Multi-Timeframe Trend Following + SMC Confirmation.
    Mengembalikan evaluasi H4 -> H1 -> M30 -> M5 -> Risk + SNR context + Signal Lifecycle Status.
    """
    scan_dict = scan if isinstance(scan, dict) else {}
    scan_snap = scan_dict.get("snapshot") if isinstance(scan_dict.get("snapshot"), dict) else None
    eff_mkt = str(scan_dict.get("market_status") or (scan_snap or {}).get("market_status") or "OPEN").upper()
    eff_fresh = str(scan_dict.get("data_freshness") or (scan_snap or {}).get("data_freshness") or "FRESH").upper()
    if eff_mkt == "CLOSED" or eff_fresh in ("STALE", "INVALID", "NO DATA"):
        wait_rsn = (
            "Market sedang tutup (MARKET CLOSED); sinyal entry baru diblokir."
            if eff_mkt == "CLOSED"
            else f"Data pasar berstatus {eff_fresh}; sinyal entry baru diblokir."
        )
        return {
            "status": "WAIT",
            "signal_status": SIGNAL_STATUS_NO_SETUP,
            "direction": None,
            "setup_type": "NONE",
            "h4": {"bias": "UNCONFIRMED", "ok": False, "regime": eff_fresh, "reason": wait_rsn},
            "h1": {"status": "UNCONFIRMED", "has_conflict": False, "ok": False, "reason": wait_rsn},
            "m30": {"status": "INVALID", "condition": eff_fresh, "ok": False, "msg": wait_rsn},
            "m5": {},
            "snr": scan_dict.get("snr_context") if isinstance(scan_dict.get("snr_context"), dict) else {"available": False, "status": "UNAVAILABLE", "summary": f"[{eff_fresh}] {wait_rsn}"},
            "candidates": [],
            "best_candidate": None,
            "plan": None,
            "reasons": [wait_rsn],
        }

    ctx = scan_dict.get("htf_context") or {}
    h4_reg = ctx.get("h4_detail")
    h4_info = evaluate_h4_primary_trend(frames.get("H4"), regime_h4=h4_reg)

    # Hormati explicit bias override dari scan mock jika ada
    if ctx.get("H4") in ("BULLISH", "BEARISH", "NETRAL", "UNCONFIRMED"):
        over_h4 = "UNCONFIRMED" if ctx["H4"] in ("NETRAL", "UNCONFIRMED") else ctx["H4"]
        if over_h4 != h4_info["bias"]:
            h4_info["bias"] = over_h4
            h4_info["ok"] = over_h4 in ("BULLISH", "BEARISH")
            h4_info["impulse_leg"] = _extract_recent_impulse_leg(frames.get("H4"), over_h4) if h4_info["ok"] else None

    if not h4_info["ok"]:
        return {
            "status": "WAIT",
            "signal_status": SIGNAL_STATUS_NO_SETUP,
            "direction": None,
            "setup_type": "NONE",
            "h4": h4_info,
            "h1": {"status": "UNCONFIRMED", "has_conflict": False, "ok": False, "reason": "Menunggu bias H4 jelas"},
            "m30": {"status": "INVALID", "condition": "UNCONFIRMED", "ok": False, "msg": "Menunggu H4"},
            "m5": {},
            "snr": {"available": False, "summary": "Belum dicek"},
            "candidates": [],
            "plan": None,
            "best_candidate": None,
            "reasons": [f"H4 Primary Trend belum terkonfirmasi ({h4_info['bias']})"],
        }

    h1_info = evaluate_h1_trend_confirmation(frames.get("H1"), h4_info, explicit_conflict=explicit_h1_conflict)
    pre_snr_ctx = (scan or {}).get("snr_context") if isinstance(scan, dict) else None
    if h1_info["has_conflict"]:
        # Cek khusus apakah memenuhi syarat Reversal STRICT sebelum benar-benar menolak
        rev_cand = build_strict_reversal_candidate(frames, h4_info, current_price)
        if rev_cand:
            snr_ctx = (
                pre_snr_ctx
                if (isinstance(pre_snr_ctx, dict) and "status" in pre_snr_ctx)
                else extract_snr_supporting_context(frames, current_price, market_status=eff_mkt, snapshot=scan_snap)
            )
            if snr_ctx.get("is_stale"):
                stale_rsn = snr_ctx.get("reason") or "Data pasar STALE; sinyal entry diblokir."
                return {
                    "status": "WAIT",
                    "signal_status": SIGNAL_STATUS_NO_SETUP,
                    "direction": None,
                    "setup_type": "NONE",
                    "h4": h4_info,
                    "h1": h1_info,
                    "m30": {"status": "INVALID", "condition": "STALE", "ok": False, "msg": stale_rsn},
                    "m5": {},
                    "snr": snr_ctx,
                    "candidates": [],
                    "best_candidate": None,
                    "plan": None,
                    "reasons": [stale_rsn],
                }
            rev_eval = validate_m5_entry_confirmation(rev_cand, frames, current_price, spread=spread, news_blackout=news_blackout, snr_context=snr_ctx)
            if rev_eval["status"] == "VALID":
                return {
                    "status": "VALID",
                    "signal_status": SIGNAL_STATUS_SIGNAL_CONFIRMED,
                    "direction": rev_cand["direction"],
                    "setup_type": "REVERSAL",
                    "h4": h4_info,
                    "h1": h1_info,
                    "m30": {"status": "CONFIRMED", "condition": "REVERSAL", "ok": True, "msg": rev_cand["label"]},
                    "m5": rev_eval,
                    "snr": snr_ctx,
                    "candidates": [rev_cand],
                    "best_candidate": rev_cand,
                    "plan": rev_eval["plan"],
                    "reasons": rev_eval["reasons"],
                }
        return {
            "status": "WAIT",
            "signal_status": SIGNAL_STATUS_NO_SETUP,
            "direction": None,
            "setup_type": "NONE",
            "h4": h4_info,
            "h1": h1_info,
            "m30": {"status": "INVALID", "condition": "CONFLICT", "ok": False, "msg": h1_info["reason"]},
            "m5": {},
            "snr": {"available": False, "summary": "Dilewati (Konflik H4 vs H1)"},
            "candidates": [],
            "best_candidate": None,
            "plan": None,
            "reasons": [h1_info["reason"]],
        }

    m30_info = assess_m30_condition(frames, h4_info, h1_info, current_price, explicit_m30_conflict=explicit_m30_conflict)
    if m30_info["status"] in ("CONFLICT", "INVALID", "WEAK"):
        return {
            "status": "WAIT",
            "signal_status": SIGNAL_STATUS_NO_SETUP,
            "direction": "BUY" if h4_info["bias"] == "BULLISH" else "SELL",
            "setup_type": "NONE",
            "h4": h4_info,
            "h1": h1_info,
            "m30": m30_info,
            "m5": {},
            "snr": {"available": False, "summary": "Dilewati"},
            "candidates": [],
            "best_candidate": None,
            "plan": None,
            "reasons": [m30_info["msg"]],
        }

    snr_ctx = (
        pre_snr_ctx
        if (isinstance(pre_snr_ctx, dict) and "status" in pre_snr_ctx)
        else extract_snr_supporting_context(frames, current_price, market_status=eff_mkt, snapshot=scan_snap)
    )
    if snr_ctx.get("is_stale"):
        stale_rsn = snr_ctx.get("reason") or "Data pasar STALE; sinyal entry diblokir."
        return {
            "status": "WAIT",
            "signal_status": SIGNAL_STATUS_NO_SETUP,
            "direction": None,
            "setup_type": "NONE",
            "h4": h4_info,
            "h1": h1_info,
            "m30": m30_info,
            "m5": {},
            "snr": snr_ctx,
            "candidates": [],
            "best_candidate": None,
            "plan": None,
            "reasons": [stale_rsn],
        }
    cands = build_trend_following_candidates(frames, h4_info, h1_info, m30_info, current_price, scan=scan, snr_context=snr_ctx)
    if preferred_zone_id:
        cands.sort(key=lambda candidate: candidate.get("zone_id") != preferred_zone_id)
    if not cands:
        return {
            "status": "WAIT",
            "signal_status": SIGNAL_STATUS_WAITING_PULLBACK,
            "direction": "BUY" if h4_info["bias"] == "BULLISH" else "SELL",
            "setup_type": "NONE",
            "h4": h4_info,
            "h1": h1_info,
            "m30": m30_info,
            "m5": {},
            "snr": snr_ctx,
            "candidates": [],
            "best_candidate": None,
            "plan": None,
            "reasons": ["Belum terbentuk area pullback/continuation/breakout-retest yang valid"],
        }

    first_eval: Optional[Dict[str, Any]] = None
    first_cand: Optional[Dict[str, Any]] = None
    for cand in cands:
        ev = validate_m5_entry_confirmation(cand, frames, current_price, spread=spread, news_blackout=news_blackout, snr_context=snr_ctx)
        if first_eval is None or (ev.get("smc_technical_confirmed") and not first_eval.get("smc_technical_confirmed")):
            first_eval, first_cand = ev, cand
        if ev["status"] == "VALID":
            return {
                "status": "VALID",
                "signal_status": SIGNAL_STATUS_SIGNAL_CONFIRMED,
                "direction": cand["direction"],
                "setup_type": cand["setup_type"],
                "h4": h4_info,
                "h1": h1_info,
                "m30": m30_info,
                "m5": ev,
                "snr": snr_ctx,
                "candidates": cands,
                "best_candidate": cand,
                "plan": ev["plan"],
                "reasons": ev["reasons"],
            }

    rev_cand_fb = build_strict_reversal_candidate(frames, h4_info, current_price)
    if rev_cand_fb:
        rev_eval_fb = validate_m5_entry_confirmation(rev_cand_fb, frames, current_price, spread=spread, news_blackout=news_blackout, snr_context=snr_ctx)
        if rev_eval_fb["status"] == "VALID":
            return {
                "status": "VALID",
                "signal_status": SIGNAL_STATUS_SIGNAL_CONFIRMED,
                "direction": rev_cand_fb["direction"],
                "setup_type": "REVERSAL",
                "h4": h4_info,
                "h1": h1_info,
                "m30": {"status": "CONFIRMED", "condition": "REVERSAL", "ok": True, "msg": rev_cand_fb["label"]},
                "m5": rev_eval_fb,
                "snr": snr_ctx,
                "candidates": cands + [rev_cand_fb],
                "best_candidate": rev_cand_fb,
                "plan": rev_eval_fb["plan"],
                "reasons": rev_eval_fb["reasons"],
            }
        if rev_eval_fb.get("smc_technical_confirmed") and (first_eval is None or not first_eval.get("smc_technical_confirmed")):
            first_eval, first_cand = rev_eval_fb, rev_cand_fb

    zt_st = ((first_eval or {}).get("stages") or {}).get("zone_touch", {}).get("status", "WAIT")
    sig_st = SIGNAL_STATUS_WAITING_PULLBACK if zt_st in ("WAIT", "FAIL") else SIGNAL_STATUS_WAITING_CONFIRMATION
    return {
        "status": "WAIT",
        "signal_status": sig_st,
        "direction": (first_cand or {}).get("direction"),
        "setup_type": (first_cand or {}).get("setup_type", "NONE"),
        "h4": h4_info,
        "h1": h1_info,
        "m30": m30_info,
        "m5": first_eval or {},
        "snr": snr_ctx,
        "candidates": cands,
        "best_candidate": first_cand,
        "plan": None,
        "reasons": (first_eval or {}).get("reasons") or ["Menunggu konfirmasi M5"],
    }


# Compatibility re-export: dipindahkan ke modules/trend_following_context.py (harus di akhir file)
from modules.trend_following_context import (  # noqa: E402,F401
    _norm_dir_bias,
    _pick_smc_focus_zone_label,
    build_unified_market_context,
)
