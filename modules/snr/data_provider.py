# -*- coding: utf-8 -*-
"""
data_provider.py — Penyedia Data Pasar & Diagnostik ATR Independen untuk SNR Advanced Scanner.
Mendukung pembacaan dari MT5 secara read-only maupun penggunaan bersama (shared snapshot)
dengan Unified Monitor tanpa memodifikasi data milik SMC Scanner.
"""
from datetime import datetime
import math
import traceback
import numpy as np
import pandas as pd

try:
    import MetaTrader5 as mt5
except Exception:  # pragma: no cover
    mt5 = None

from modules.config import SYMBOL as DEFAULT_APP_SYMBOL
from modules.mt5_conn import ambil_data, pastikan_mt5_aktif
from modules.snr.config import (
    FILE_LOG_SNR_AUDIT,
    FILE_LOG_SNR_ERROR,
    SNR_TIMEFRAMES,
    get_snr_config,
)


def log_snr_error(konteks, err):
    """Mencatat error internal SNR ke file log terpisah agar mudah diaudit."""
    try:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        tb = traceback.format_exc()
        with open(FILE_LOG_SNR_ERROR, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] [SNR_ERROR] {konteks}: {type(err).__name__}: {err}\n")
            if tb and "NoneType: None" not in tb:
                f.write(tb + "\n")
    except Exception:
        pass


def log_snr_audit(pesan):
    """Mencatat jejak audit SNR ke file log terpisah."""
    try:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(FILE_LOG_SNR_AUDIT, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] [SNR_AUDIT] {pesan}\n")
    except Exception:
        pass


def safe_float(val, default=0.0):
    """Konversi nilai ke float secara aman tanpa melempar exception."""
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return float(default)
        return f
    except Exception:
        return float(default)


def compute_atr14_series(df, period=14):
    """Menghitung seri ATR(14) secara independen tanpa bergantung pada modul SMC."""
    high = pd.to_numeric(df["High"], errors="coerce")
    low = pd.to_numeric(df["Low"], errors="coerce")
    close = pd.to_numeric(df["Close"], errors="coerce")
    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / float(period), min_periods=1, adjust=False).mean()
    return atr


def prepare_snr_frame(df, has_unclosed_last_bar=False):
    """
    Menyiapkan DataFrame candle untuk analisis SNR:
    - Membuat salinan independen (tidak memodifikasi DataFrame asli).
    - Memisahkan candle yang belum closed (jika ada kolom is_closed=False / is_forming=True
      atau parameter has_unclosed_last_bar=True) sehingga validasi struktur dan sinyal final
      100% hanya menggunakan CLOSED candles.
    - Memastikan kolom Time, Open, High, Low, Close, ATR14, dan TickVolume tersedia.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(), None

    work = df.copy(deep=True)

    # Normalisasi nama kolom dasar jika menggunakan huruf kecil
    col_map = {}
    for c in work.columns:
        cl = str(c).strip().lower()
        if cl == "open" and "Open" not in work.columns:
            col_map[c] = "Open"
        elif cl == "high" and "High" not in work.columns:
            col_map[c] = "High"
        elif cl == "low" and "Low" not in work.columns:
            col_map[c] = "Low"
        elif cl == "close" and "Close" not in work.columns:
            col_map[c] = "Close"
        elif cl in ("time", "timestamp", "datetime", "date") and "Time" not in work.columns:
            col_map[c] = "Time"
    if col_map:
        work = work.rename(columns=col_map)

    for req_col in ("Open", "High", "Low", "Close"):
        if req_col not in work.columns:
            return pd.DataFrame(), None
        work[req_col] = pd.to_numeric(work[req_col], errors="coerce")

    work = work.dropna(subset=["Open", "High", "Low", "Close"]).reset_index(drop=True)
    if work.empty:
        return pd.DataFrame(), None

    if "Time" not in work.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            work["Time"] = df.index[: len(work)]
        else:
            work["Time"] = pd.date_range("2026-01-01", periods=len(work), freq="5min")

    # Urutkan berdasarkan waktu secara kronologis untuk mencegah ketidakurutan timestamp
    try:
        work = work.sort_values("Time", kind="mergesort").reset_index(drop=True)
    except Exception:
        work = work.reset_index(drop=True)

    # Pisahkan candle yang belum closed (forming/open candle)
    forming_candle = None
    if "is_closed" in work.columns:
        unclosed_mask = work["is_closed"].astype(bool) == False  # noqa: E712
        if unclosed_mask.any():
            forming_candle = work[unclosed_mask].iloc[-1].to_dict()
            work = work[~unclosed_mask].reset_index(drop=True)
    elif "is_forming" in work.columns:
        forming_mask = work["is_forming"].astype(bool) == True  # noqa: E712
        if forming_mask.any():
            forming_candle = work[forming_mask].iloc[-1].to_dict()
            work = work[~forming_mask].reset_index(drop=True)
    elif has_unclosed_last_bar and len(work) > 1:
        forming_candle = work.iloc[-1].to_dict()
        work = work.iloc[:-1].reset_index(drop=True)

    if work.empty:
        return pd.DataFrame(), forming_candle

    # Hitung ATR14 pada candle closed jika belum tersedia atau bernilai NaN
    if "ATR14" not in work.columns or work["ATR14"].isna().all():
        work["ATR14"] = compute_atr14_series(work, period=14)
    else:
        work["ATR14"] = pd.to_numeric(work["ATR14"], errors="coerce")
        if work["ATR14"].isna().any():
            work["ATR14"] = work["ATR14"].bfill().ffill()
            if work["ATR14"].isna().any():
                work["ATR14"] = compute_atr14_series(work, period=14)

    # Identifikasi ketersediaan Tick Volume secara jujur (jangan mengarang volume)
    vol_col = None
    for candidate_col in ("tick_volume", "TickVolume", "Volume", "Vol", "real_volume"):
        if candidate_col in work.columns:
            vol_col = candidate_col
            break

    if vol_col is not None:
        v_series = pd.to_numeric(work[vol_col], errors="coerce")
        if v_series.notna().any() and float(v_series.fillna(0).max()) > 0:
            work["TickVolume"] = v_series.fillna(0.0)
            work["HasTickVolume"] = True
        else:
            work["TickVolume"] = np.nan
            work["HasTickVolume"] = False
    else:
        work["TickVolume"] = np.nan
        work["HasTickVolume"] = False

    return work, forming_candle


def get_symbol_spec(symbol=None, tick=None, config=None):
    """
    Mengambil spesifikasi simbol dari MT5 secara read-only (digits, point, tick_size,
    tick_value, contract_size, spread). Jika MT5 offline/mock, gunakan fallback konfigurasi.
    """
    cfg = config or get_snr_config()
    risk_cfg = cfg.get("risk", {})
    sym = symbol or cfg.get("symbol", DEFAULT_APP_SYMBOL)

    digits = int(risk_cfg.get("default_digits", 2))
    point = 10 ** (-digits)
    tick_size = safe_float(risk_cfg.get("default_tick_size", 0.01), 0.01)
    tick_value = safe_float(risk_cfg.get("default_tick_value", 1.0), 1.0)
    contract_size = safe_float(risk_cfg.get("default_contract_size", 100.0), 100.0)
    spread_points = safe_float(risk_cfg.get("default_spread_points", 0.25), 0.25)
    source = "CONFIG_DEFAULT"

    if mt5 is not None:
        try:
            info = mt5.symbol_info(sym)
            if info is not None:
                d_val = getattr(info, "digits", None)
                if d_val is not None and int(d_val) >= 0:
                    digits = int(d_val)
                p_val = safe_float(getattr(info, "point", 0.0), 0.0)
                if p_val > 0:
                    point = p_val
                ts_val = safe_float(getattr(info, "trade_tick_size", 0.0), 0.0)
                if ts_val > 0:
                    tick_size = ts_val
                tv_val = safe_float(getattr(info, "trade_tick_value", 0.0), 0.0)
                if tv_val > 0:
                    tick_value = tv_val
                cs_val = safe_float(getattr(info, "trade_contract_size", 0.0), 0.0)
                if cs_val > 0:
                    contract_size = cs_val
                sp_raw = safe_float(getattr(info, "spread", 0.0), 0.0)
                if sp_raw > 0 and point > 0:
                    spread_points = round(sp_raw * point, max(2, digits))
                source = "MT5_SYMBOL_INFO"
        except Exception:
            pass

    if tick is not None:
        bid = safe_float(getattr(tick, "bid", None) if not isinstance(tick, dict) else tick.get("bid"), 0.0)
        ask = safe_float(getattr(tick, "ask", None) if not isinstance(tick, dict) else tick.get("ask"), 0.0)
        if bid > 0 and ask >= bid:
            spread_points = round(ask - bid, max(2, digits))

    return {
        "symbol": sym,
        "digits": digits,
        "point": point,
        "tick_size": tick_size,
        "tick_value": tick_value,
        "contract_size": contract_size,
        "spread_points": spread_points,
        "source": source,
    }


def compute_atr_diagnostics(frames_dict):
    """Menghitung diagnostik ATR14 untuk seluruh timeframe SNR (H4, H1, M30, M5)."""
    diag = {}
    for tf in SNR_TIMEFRAMES:
        df = frames_dict.get(tf)
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            diag[tf] = {
                "timeframe": tf,
                "available": False,
                "bars": 0,
                "atr14": 0.0,
                "atr_mean": 0.0,
                "atr_ratio": 1.0,
                "volatility_regime": "N/A",
                "last_close": 0.0,
                "last_time": "N/A",
            }
            continue

        atr_series = pd.to_numeric(df["ATR14"], errors="coerce").dropna()
        curr_atr = safe_float(atr_series.iloc[-1], 0.0) if not atr_series.empty else 0.0
        mean_atr = safe_float(atr_series.tail(50).mean(), curr_atr) if not atr_series.empty else 0.0
        if curr_atr <= 0:
            rng = (df["High"] - df["Low"]).tail(14)
            curr_atr = safe_float(rng.mean(), 1.0)
            mean_atr = curr_atr
        ratio = curr_atr / mean_atr if mean_atr > 1e-9 else 1.0
        if ratio >= 1.25:
            regime = "HIGH"
        elif ratio <= 0.80:
            regime = "LOW"
        else:
            regime = "NORMAL"

        last_close = safe_float(df["Close"].iloc[-1], 0.0)
        last_time = str(df["Time"].iloc[-1]) if "Time" in df.columns else "N/A"
        diag[tf] = {
            "timeframe": tf,
            "available": True,
            "bars": int(len(df)),
            "atr14": round(curr_atr, 4),
            "atr_mean": round(mean_atr, 4),
            "atr_ratio": round(ratio, 3),
            "volatility_regime": regime,
            "last_close": round(last_close, 2),
            "last_time": last_time,
        }
    return diag


def fetch_snr_market_data(
    symbol=None,
    snapshot=None,
    frames=None,
    tick=None,
    live=True,
    has_unclosed_last_bar=False,
    config=None,
):
    """
    Mengumpulkan dan menormalisasi data pasar untuk SNR Advanced Scanner.
    Urutan prioritas sumber data:
    1. `frames` eksplisit jika diberikan.
    2. `snapshot["frames"]` jika diberikan (misal dari Unified Monitor agar 1x fetch per siklus).
    3. `ambil_data(symbol, tf, bars)` dari MT5 jika `live=True`.
    """
    cfg = config or get_snr_config()
    sym = symbol or (snapshot.get("symbol") if isinstance(snapshot, dict) else None) or cfg.get("symbol", DEFAULT_APP_SYMBOL)
    tf_bars = cfg.get("timeframe_bars", {})

    raw_frames = {}
    if isinstance(frames, dict) and frames:
        raw_frames = dict(frames)
    elif isinstance(snapshot, dict) and isinstance(snapshot.get("frames"), dict):
        raw_frames = dict(snapshot["frames"])

    mt5_ready = False
    if live and len(raw_frames) < len(SNR_TIMEFRAMES):
        try:
            mt5_ready = bool(pastikan_mt5_aktif())
        except Exception:
            mt5_ready = False

    prepared_frames = {}
    forming_candles = {}

    for tf in SNR_TIMEFRAMES:
        df_src = raw_frames.get(tf)
        if (df_src is None or (isinstance(df_src, pd.DataFrame) and df_src.empty)) and live and mt5_ready:
            try:
                df_src = ambil_data(sym, tf, int(tf_bars.get(tf, 220)))
            except Exception as exc:
                log_snr_error(f"fetch_snr_market_data({tf})", exc)
                df_src = None

        df_clean, forming_bar = prepare_snr_frame(df_src, has_unclosed_last_bar=has_unclosed_last_bar)
        prepared_frames[tf] = df_clean
        if forming_bar is not None:
            forming_candles[tf] = forming_bar

    # Ambil informasi harga bid/ask/mid
    bid = 0.0
    ask = 0.0
    time_label = "SNAPSHOT"
    server_time = None

    if isinstance(snapshot, dict):
        bid = safe_float(snapshot.get("bid"), 0.0)
        ask = safe_float(snapshot.get("ask"), 0.0)
        server_time = snapshot.get("server_time")
        if snapshot.get("live", False) or snapshot.get("mt5_connected", False):
            time_label = "MT5 LIVE"

    if tick is not None:
        t_bid = safe_float(getattr(tick, "bid", None) if not isinstance(tick, dict) else tick.get("bid"), 0.0)
        t_ask = safe_float(getattr(tick, "ask", None) if not isinstance(tick, dict) else tick.get("ask"), 0.0)
        if t_bid > 0:
            bid = t_bid
        if t_ask > 0:
            ask = t_ask
        time_label = "MT5 LIVE"
    elif bid <= 0 and live and mt5_ready and mt5 is not None:
        try:
            live_tick = mt5.symbol_info_tick(sym)
            if live_tick is not None:
                bid = safe_float(getattr(live_tick, "bid", 0.0), 0.0)
                ask = safe_float(getattr(live_tick, "ask", 0.0), 0.0)
                tick = live_tick
                time_label = "MT5 LIVE"
        except Exception:
            pass

    # Fallback harga dari close candle closed terakhir (M5 -> M30 -> H1 -> H4)
    if bid <= 0:
        for tf in ("M5", "M30", "H1", "H4"):
            df_tf = prepared_frames.get(tf)
            if df_tf is not None and not df_tf.empty:
                bid = safe_float(df_tf["Close"].iloc[-1], 0.0)
                if bid > 0:
                    if server_time is None and "Time" in df_tf.columns:
                        server_time = str(df_tf["Time"].iloc[-1])
                    break

    spec = get_symbol_spec(symbol=sym, tick=tick, config=cfg)
    if ask <= 0 and bid > 0:
        ask = round(bid + spec["spread_points"], spec["digits"])
    spread = round(max(0.0, ask - bid), max(2, spec["digits"])) if (bid > 0 and ask > 0) else spec["spread_points"]
    spec["spread_points"] = spread
    price = round(bid, spec["digits"]) if bid > 0 else 0.0

    atr_diag = compute_atr_diagnostics(prepared_frames)

    return {
        "symbol": sym,
        "price": price,
        "bid": round(bid, spec["digits"]) if bid > 0 else 0.0,
        "ask": round(ask, spec["digits"]) if ask > 0 else 0.0,
        "mid": round((bid + ask) / 2.0, spec["digits"]) if (bid > 0 and ask > 0) else price,
        "spread": spread,
        "time_label": time_label,
        "server_time": server_time or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "frames": prepared_frames,
        "forming_candles": forming_candles,
        "symbol_spec": spec,
        "atr_diagnostics": atr_diag,
    }

