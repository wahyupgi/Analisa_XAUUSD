# -*- coding: utf-8 -*-
"""
data_provider.py — Penyedia Data Pasar & Diagnostik ATR Independen untuk SNR Advanced Scanner.
Mendukung pembacaan dari MT5 secara read-only maupun penggunaan bersama (shared snapshot)
dengan Unified Monitor tanpa memodifikasi data milik SMC Scanner.
"""
from datetime import datetime, timezone
import math
import time
import traceback
import numpy as np
import pandas as pd

try:
    import MetaTrader5 as mt5
except Exception:  # pragma: no cover
    mt5 = None

from modules.config import SYMBOL as DEFAULT_APP_SYMBOL, TIMEFRAMES as APP_TIMEFRAMES
from modules.utils import format_wib_timestamp, get_current_wib
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


def zone_confirmation_start(zone, frame):
    """First candle opening after the zone's structural confirmation candle closed."""
    confirmed_time = zone.get("confirmed_time")
    tf = str(zone.get("timeframe", "")).upper()
    minutes = {"M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240}.get(tf)
    if not confirmed_time or minutes is None:
        return len(frame)
    confirmed = pd.to_datetime(confirmed_time, errors="coerce", utc=True)
    times = pd.to_datetime(frame["Time"], errors="coerce", utc=True)
    if pd.isna(confirmed) or times.isna().any():
        return len(frame)
    zone_tz = str(zone.get("time_tz") or frame.attrs.get("time_tz", "WIB")).upper()
    frame_tz = str(frame.attrs.get("time_tz", "WIB")).upper()
    if pd.Timestamp(confirmed_time).tzinfo is None and zone_tz != "UTC":
        confirmed -= pd.Timedelta(hours=7)
    if len(frame) and pd.Timestamp(frame["Time"].iloc[0]).tzinfo is None and frame_tz != "UTC":
        times = times - pd.Timedelta(hours=7)
    return int(times.searchsorted(confirmed + pd.Timedelta(minutes=minutes), side="left"))


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
            empty_df = pd.DataFrame()
            empty_df.attrs["invalid_ohlc"] = True
            return empty_df, None
        work[req_col] = pd.to_numeric(work[req_col], errors="coerce")

    had_invalid_ohlc = False
    if not work.empty:
        tail_chk = work.iloc[-min(len(work), 5):]
        if tail_chk[["Open", "High", "Low", "Close"]].isna().any().any():
            had_invalid_ohlc = True
        bad_ohlc = (
            (work["High"] < work["Low"])
            | (work["Close"] <= 0)
            | (work["Open"] <= 0)
            | (work["High"] <= 0)
            | (work["Low"] <= 0)
        )
        if bad_ohlc.any():
            had_invalid_ohlc = True

    work = work.dropna(subset=["Open", "High", "Low", "Close"]).reset_index(drop=True)
    if not work.empty:
        valid_ohlc = (work["High"] >= work["Low"]) & (work["Close"] > 0) & (work["Open"] > 0)
        work = work[valid_ohlc].reset_index(drop=True)
    if work.empty:
        empty_df = pd.DataFrame()
        if had_invalid_ohlc:
            empty_df.attrs["invalid_ohlc"] = True
        return empty_df, None

    if "Time" not in work.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            work["Time"] = df.index[: len(work)]
        else:
            work["Time"] = pd.date_range("2026-01-01", periods=len(work), freq="5min")

    # Deteksi duplikasi & ketidakurutan timestamp sebelum normalisasi
    src_attrs = dict(getattr(df, "attrs", {}) or {})
    had_duplicate_ts = False
    had_out_of_order_ts = False
    try:
        ts_chk = pd.to_datetime(work["Time"], errors="coerce")
        if len(ts_chk) > 1:
            if ts_chk.duplicated().any():
                had_duplicate_ts = True
            if not ts_chk.is_monotonic_increasing:
                had_out_of_order_ts = True
    except Exception:
        pass

    # Urutkan berdasarkan waktu secara kronologis untuk mencegah ketidakurutan timestamp
    try:
        work = work.sort_values("Time", kind="mergesort").reset_index(drop=True)
    except Exception:
        work = work.reset_index(drop=True)

    # Pisahkan candle yang belum closed (forming/open candle)
    src_is_unclosed = src_attrs.get("is_closed") is False or src_attrs.get("has_unclosed_last_bar") is True
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
    elif (has_unclosed_last_bar or src_is_unclosed) and len(work) > 1:
        forming_candle = work.iloc[-1].to_dict()
        work = work.iloc[:-1].reset_index(drop=True)

    if work.empty:
        empty_df = pd.DataFrame()
        if had_invalid_ohlc:
            empty_df.attrs["invalid_ohlc"] = True
        return empty_df, forming_candle

    if had_invalid_ohlc:
        work.attrs["invalid_ohlc"] = True
    if had_duplicate_ts:
        work.attrs["duplicate_timestamps"] = True
    if had_out_of_order_ts:
        work.attrs["out_of_order_timestamps"] = True
    if "time_tz" in src_attrs:
        work.attrs["time_tz"] = src_attrs["time_tz"]
    if src_is_unclosed:
        work.attrs["is_closed"] = False

    # Hitung ATR14 pada candle closed jika belum tersedia atau bernilai NaN
    if "ATR14" not in work.columns or work["ATR14"].isna().all():
        work["ATR14"] = compute_atr14_series(work, period=14)
    else:
        work["ATR14"] = pd.to_numeric(work["ATR14"], errors="coerce")
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
    stops_level = None
    tick_size_verified = False
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
                raw_stops = getattr(info, "trade_stops_level", None)
                if p_val > 0 and raw_stops is not None and safe_float(raw_stops, -1.0) >= 0:
                    stops_level = safe_float(raw_stops) * point
                ts_val = safe_float(getattr(info, "trade_tick_size", 0.0), 0.0)
                if ts_val > 0:
                    tick_size = ts_val
                    tick_size_verified = True
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
        "tick_size_verified": tick_size_verified,
        "tick_value": tick_value,
        "contract_size": contract_size,
        "spread_points": spread_points,
        "stops_level": stops_level,
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


def _resolve_mt5_tf_constant(tf_name):
    """Memetakan nama timeframe ('H4', 'H1', 'M30', 'M5') ke konstanta integer MT5."""
    if isinstance(APP_TIMEFRAMES, dict) and tf_name in APP_TIMEFRAMES:
        tf_entry = APP_TIMEFRAMES[tf_name]
        if isinstance(tf_entry, (tuple, list)) and len(tf_entry) >= 1:
            return tf_entry[0]
        return tf_entry
    if mt5 is not None:
        return getattr(mt5, f"TIMEFRAME_{tf_name}", None)
    return tf_name


def _fetch_mt5_historical_candles(symbol, tf_name, bars):
    """
    Mengambil candle historis yang SUDAH TERTUTUP (posisi >= 1) dari MT5 untuk timeframe terkait.
    Tidak bergantung pada adanya tick baru sehingga tetap berhasil saat market CLOSED.
    """
    tf_const = _resolve_mt5_tf_constant(tf_name)
    df_res = None

    # 1. Gunakan ambil_data standar (posisi 1 = candle closed terakhir)
    try:
        df_res = ambil_data(tf_const, int(bars))
    except TypeError:
        # Kompatibilitas jika fungsi ambil_data di-mock dengan 3 argumen (symbol, tf, bars)
        try:
            df_res = ambil_data(symbol, tf_name, int(bars))
        except Exception as exc_inner:
            log_snr_error(f"fetch_snr_market_data({tf_name})", exc_inner)
            df_res = None
    except Exception as exc:
        log_snr_error(f"fetch_snr_market_data({tf_name})", exc)
        df_res = None

    # 2. Fallback langsung ke mt5.copy_rates_from_pos jika simbol berbeda atau percobaan pertama kosong
    if (df_res is None or (isinstance(df_res, pd.DataFrame) and df_res.empty)) and mt5 is not None and isinstance(tf_const, int):
        for sym_try in (symbol, DEFAULT_APP_SYMBOL, "XAUUSDc", "XAUUSD", "XAUUSDm"):
            if not sym_try:
                continue
            try:
                rates = mt5.copy_rates_from_pos(sym_try, tf_const, 1, int(bars))
                if rates is not None and len(rates) > 0:
                    df_tmp = pd.DataFrame(rates)
                    df_tmp["time"] = pd.to_datetime(df_tmp["time"], unit="s")
                    df_tmp = df_tmp.set_index("time")
                    df_res = df_tmp.rename(
                        columns={
                            "open": "Open",
                            "high": "High",
                            "low": "Low",
                            "close": "Close",
                            "tick_volume": "Volume",
                        }
                    )
                    df_res.attrs["time_tz"] = "UTC"
                    break
            except Exception:
                continue

    return df_res


def detect_snr_market_status(
    prepared_frames,
    symbol=None,
    tick=None,
    live=True,
    snapshot=None,
    market_status_override=None,
    now_utc=None,
    config=None,
):
    """
    Mendeteksi kondisi pasar secara deterministik:
    - MARKET OPEN   (market_status='OPEN', data_source='MT5 LIVE' / 'SNAPSHOT', snr_analysis='AVAILABLE')
    - MARKET CLOSED (market_status='CLOSED', data_source='MT5 HISTORICAL', snr_analysis='AVAILABLE')
    - DATA UNAVAILABLE (market_status='DATA UNAVAILABLE', data_source='DATA UNAVAILABLE', snr_analysis='DATA UNAVAILABLE')

    Tidak menyimpulkan kegagalan data hanya karena tidak adanya tick baru selama
    candle historis tertutup tetap tersedia.
    """
    frames_map = prepared_frames or {}
    last_close = 0.0
    last_candle_str = "DATA UNAVAILABLE"
    last_candle_dt = None

    for tf in ("M5", "M30", "H1", "H4"):
        df_tf = frames_map.get(tf)
        if isinstance(df_tf, tuple) and len(df_tf) >= 1:
            df_tf = df_tf[0]
        if df_tf is not None and isinstance(df_tf, pd.DataFrame) and not df_tf.empty:
            c_val = safe_float(df_tf["Close"].iloc[-1], 0.0)
            if c_val > 0:
                last_close = c_val
                if "Time" in df_tf.columns:
                    raw_t = df_tf["Time"].iloc[-1]
                    last_candle_str = str(raw_t)
                    try:
                        last_candle_dt = pd.Timestamp(raw_t).to_pydatetime().replace(tzinfo=None)
                    except Exception:
                        last_candle_dt = None
                break

    has_historical_data = bool(last_close > 0)

    # Cek override eksplisit (dari parameter, snapshot, atau config)
    ov = market_status_override
    if ov is None and isinstance(snapshot, dict):
        if snapshot.get("market_status") is not None:
            ov = snapshot.get("market_status")
        elif snapshot.get("market_closed") is not None:
            ov = "CLOSED" if snapshot.get("market_closed") else "OPEN"
    if ov is None and isinstance(config, dict) and config.get("market_status") is not None:
        ov = config.get("market_status")

    ov_str = str(ov).strip().upper() if ov is not None else ""

    if not has_historical_data or ov_str in ("DATA UNAVAILABLE", "UNAVAILABLE", "NO_DATA"):
        return {
            "market_status": "DATA UNAVAILABLE",
            "market_status_label": "DATA UNAVAILABLE",
            "data_source": "DATA UNAVAILABLE",
            "snr_analysis": "DATA UNAVAILABLE",
            "is_market_open": False,
            "is_market_closed": False,
            "data_available": False,
            "last_price": 0.0,
            "last_candle": "DATA UNAVAILABLE",
            "reason": "Data candle historis MT5 tidak tersedia.",
        }

    if ov_str in ("CLOSED", "MARKET CLOSED", "MARKET_CLOSED"):
        return {
            "market_status": "CLOSED",
            "market_status_label": "MARKET CLOSED",
            "data_source": "MT5 HISTORICAL",
            "snr_analysis": "AVAILABLE",
            "is_market_open": False,
            "is_market_closed": True,
            "data_available": True,
            "last_price": last_close,
            "last_candle": last_candle_str,
            "reason": "Market XAUUSD sedang tutup (MARKET CLOSED); menggunakan data historis tertutup terakhir.",
        }

    is_live_mode = bool(live) or (isinstance(snapshot, dict) and bool(snapshot.get("live", False)))
    t_bid = safe_float(getattr(tick, "bid", None) if (tick is not None and not isinstance(tick, dict)) else (tick or {}).get("bid"), 0.0)
    has_live_tick = bool(t_bid > 0)

    if ov_str in ("OPEN", "MARKET OPEN", "MARKET_OPEN"):
        src = "MT5 LIVE" if (is_live_mode or has_live_tick) else "SNAPSHOT"
        return {
            "market_status": "OPEN",
            "market_status_label": "MARKET OPEN",
            "data_source": src,
            "snr_analysis": "AVAILABLE",
            "is_market_open": True,
            "is_market_closed": False,
            "data_available": True,
            "last_price": t_bid if has_live_tick else last_close,
            "last_candle": last_candle_str,
            "reason": "Market XAUUSD sedang buka (MARKET OPEN).",
        }

    # Mode offline / snapshot test tanpa override eksplisit -> perlakukan sebagai OPEN (SNAPSHOT)
    if not is_live_mode:
        return {
            "market_status": "OPEN",
            "market_status_label": "MARKET OPEN",
            "data_source": "SNAPSHOT",
            "snr_analysis": "AVAILABLE",
            "is_market_open": True,
            "is_market_closed": False,
            "data_available": True,
            "last_price": t_bid if has_live_tick else last_close,
            "last_candle": last_candle_str,
            "reason": "Menggunakan data snapshot pasar.",
        }

    # Evaluasi otomatis kondisi MARKET CLOSED vs MARKET OPEN pada mode live
    closed_reasons = []

    # 1. Cek symbol_info.trade_mode dari MT5
    if mt5 is not None and symbol:
        try:
            s_info = mt5.symbol_info(symbol)
            if s_info is not None:
                tm = getattr(s_info, "trade_mode", None)
                disabled_modes = (
                    getattr(mt5, "SYMBOL_TRADE_MODE_DISABLED", 0),
                    getattr(mt5, "SYMBOL_TRADE_MODE_CLOSEONLY", 3),
                )
                if tm is not None and tm in disabled_modes:
                    closed_reasons.append("MT5 symbol trade_mode disabled/close-only")
        except Exception:
            pass

    # 2. Cek jadwal sesi global XAUUSD (Weekend Closure & Daily Rollover Break dalam UTC)
    ref_utc = now_utc if isinstance(now_utc, datetime) else datetime.now(timezone.utc)
    wk = ref_utc.weekday()  # 0=Mon .. 4=Fri, 5=Sat, 6=Sun
    hr = ref_utc.hour
    if wk == 5 or (wk == 4 and hr >= 22) or (wk == 6 and hr < 22):
        closed_reasons.append("Akhir pekan (XAUUSD Weekend Market Closed)")
    elif wk in (0, 1, 2, 3) and hr == 21:
        closed_reasons.append("Jeda rollover harian XAUUSD (Daily Market Break)")

    # 3. Cek ketiadaan tick live atau usia candle terakhir (> 2 jam dari waktu UTC saat ini)
    if not has_live_tick:
        closed_reasons.append("Tidak ada live tick aktif; menggunakan harga penutupan candle terakhir")
    elif last_candle_dt is not None:
        utc_naive = ref_utc.replace(tzinfo=None)
        age_sec = (utc_naive - last_candle_dt).total_seconds()
        if age_sec > 7200:
            closed_reasons.append("Candle terakhir lebih lama dari 2 jam (sesi perdagangan tutup)")

    if closed_reasons:
        return {
            "market_status": "CLOSED",
            "market_status_label": "MARKET CLOSED",
            "data_source": "MT5 HISTORICAL",
            "snr_analysis": "AVAILABLE",
            "is_market_open": False,
            "is_market_closed": True,
            "data_available": True,
            "last_price": last_close,
            "last_candle": last_candle_str,
            "reason": "; ".join(closed_reasons),
        }

    return {
        "market_status": "OPEN",
        "market_status_label": "MARKET OPEN",
        "data_source": "MT5 LIVE",
        "snr_analysis": "AVAILABLE",
        "is_market_open": True,
        "is_market_closed": False,
        "data_available": True,
        "last_price": t_bid if has_live_tick else last_close,
        "last_candle": last_candle_str,
        "reason": "Market XAUUSD sedang buka (MARKET OPEN).",
    }


def _snr_align_ts(ts_a, ts_b, b_tz_hint="WIB"):
    a_tz = getattr(ts_a, "tzinfo", None) is not None
    b_tz = getattr(ts_b, "tzinfo", None) is not None
    if a_tz == b_tz:
        return ts_a, ts_b
    if a_tz and not b_tz:
        try:
            if str(b_tz_hint or "WIB").upper() == "UTC":
                return ts_a, ts_b.tz_localize("UTC")
            return ts_a.tz_localize(None), ts_b
        except Exception:
            return ts_a, ts_b
    try:
        return ts_a, ts_b.tz_convert("UTC").tz_localize(None)
    except Exception:
        try:
            return ts_a, ts_b.tz_localize(None)
        except Exception:
            return ts_a, ts_b


def validate_snr_data_freshness(
    prepared_frames,
    server_time=None,
    tick=None,
    market_status="OPEN",
    connection_status="UNKNOWN",
    bid=0.0,
    ask=0.0,
    snapshot=None,
):
    """
    Validasi DATA FRESHNESS untuk SNR Advanced Scanner:
    FRESH / STALE / INVALID / NO DATA.
    """
    tf_minutes = {"M5": 5, "M30": 30, "H1": 60, "H4": 240, "D1": 1440}
    last_candle_times = {}
    ages_minutes = {}
    stale_tfs = []
    frames_map = prepared_frames or {}
    mkt_st = str(market_status or "OPEN").upper()
    conn_st = str(connection_status or "UNKNOWN").upper()

    if isinstance(snapshot, dict) and snapshot.get("snapshot_status") == "DATA_INCOMPLETE":
        return {
            "status": "INVALID",
            "ok": False,
            "reason": snapshot.get("error_reason") or "Snapshot pasar DATA_INCOMPLETE",
            "market_status": mkt_st,
            "connection_status": conn_st,
            "last_candle_times": last_candle_times,
            "ages_minutes": ages_minutes,
            "stale_timeframes": stale_tfs,
            "is_market_closed": mkt_st == "CLOSED",
        }

    if isinstance(snapshot, dict) and snapshot.get("data_freshness") in ("INVALID", "NO DATA", "STALE"):
        snap_df_status = str(snapshot.get("data_freshness")).upper()
        snap_df_info = snapshot.get("data_freshness_info") if isinstance(snapshot.get("data_freshness_info"), dict) else {}
        # Evaluasi waktu candle tetap dikumpulkan
        for tf in ("H4", "H1", "M30", "M5"):
            df_tf = frames_map.get(tf)
            if df_tf is not None and not df_tf.empty and "Time" in df_tf.columns:
                raw_t = df_tf["Time"].iloc[-1]
                src_tz = str((getattr(df_tf, "attrs", {}) or {}).get("time_tz", "WIB"))
                last_candle_times[tf] = format_wib_timestamp(raw_t, default_now=False, source_tz=src_tz) or str(raw_t)
        default_tick_st = "MARKET CLOSED" if mkt_st == "CLOSED" else snap_df_status
        default_candle_st = "HISTORICAL (CLOSED)" if mkt_st == "CLOSED" else snap_df_status
        return {
            "status": snap_df_status,
            "ok": False,
            "reason": snap_df_info.get("reason") or snapshot.get("error_reason") or f"Status snapshot: {snap_df_status}",
            "market_status": mkt_st,
            "connection_status": conn_st,
            "tick_status": snap_df_info.get("tick_status") or default_tick_st,
            "candle_status": snap_df_info.get("candle_status") or default_candle_st,
            "data_validation": snap_df_info.get("data_validation") or "FAIL",
            "last_candle_times": last_candle_times or dict(snap_df_info.get("last_candle_times") or {}),
            "ages_minutes": dict(snap_df_info.get("ages_minutes") or ages_minutes),
            "stale_timeframes": list(snap_df_info.get("stale_timeframes") or stale_tfs),
            "is_market_closed": bool(snap_df_info.get("is_market_closed", mkt_st == "CLOSED")),
        }

    # Cek validitas tick bila diberikan
    if tick is not None:
        tb = getattr(tick, "bid", None) if not isinstance(tick, dict) else tick.get("bid")
        ta = getattr(tick, "ask", None) if not isinstance(tick, dict) else tick.get("ask")
        try:
            fb = float(tb) if tb is not None else 0.0
            fa = float(ta) if ta is not None else 0.0
            if not math.isfinite(fb) or not math.isfinite(fa) or (fa > 0 and fa < fb) or fb < 0:
                return {
                    "status": "INVALID",
                    "ok": False,
                    "reason": "Tick Bid/Ask tidak valid",
                    "market_status": mkt_st,
                    "connection_status": conn_st,
                    "tick_status": "INVALID",
                    "candle_status": "HISTORICAL (CLOSED)" if mkt_st == "CLOSED" else "UNKNOWN",
                    "data_validation": "FAIL",
                    "last_candle_times": last_candle_times,
                    "ages_minutes": ages_minutes,
                    "stale_timeframes": stale_tfs,
                    "is_market_closed": mkt_st == "CLOSED",
                }
        except Exception:
            return {
                "status": "INVALID",
                "ok": False,
                "reason": "Tick Bid/Ask korup",
                "market_status": mkt_st,
                "connection_status": conn_st,
                "tick_status": "INVALID",
                "candle_status": "HISTORICAL (CLOSED)" if mkt_st == "CLOSED" else "UNKNOWN",
                "data_validation": "FAIL",
                "last_candle_times": last_candle_times,
                "ages_minutes": ages_minutes,
                "stale_timeframes": stale_tfs,
                "is_market_closed": mkt_st == "CLOSED",
            }

    if bid > 0 and ask > 0 and ask < bid:
        return {
            "status": "INVALID",
            "ok": False,
            "reason": f"Ask ({ask:.2f}) lebih kecil dari Bid ({bid:.2f})",
            "market_status": mkt_st,
            "connection_status": conn_st,
            "tick_status": "INVALID",
            "candle_status": "HISTORICAL (CLOSED)" if mkt_st == "CLOSED" else "UNKNOWN",
            "data_validation": "FAIL",
            "last_candle_times": last_candle_times,
            "ages_minutes": ages_minutes,
            "stale_timeframes": stale_tfs,
            "is_market_closed": mkt_st == "CLOSED",
        }

    non_empty = [tf for tf in ("H4", "H1", "M30", "M5") if isinstance(frames_map.get(tf), pd.DataFrame) and not frames_map[tf].empty]
    any_invalid = any(
        isinstance(frames_map.get(tf), pd.DataFrame) and bool(getattr(frames_map[tf], "attrs", {}).get("invalid_ohlc", False))
        for tf in ("H4", "H1", "M30", "M5")
    )
    any_dup = any(
        isinstance(frames_map.get(tf), pd.DataFrame) and bool(getattr(frames_map[tf], "attrs", {}).get("duplicate_timestamps", False))
        for tf in ("H4", "H1", "M30", "M5")
    )
    any_ooo = any(
        isinstance(frames_map.get(tf), pd.DataFrame) and bool(getattr(frames_map[tf], "attrs", {}).get("out_of_order_timestamps", False))
        for tf in ("H4", "H1", "M30", "M5")
    )
    if any_invalid or any_dup or any_ooo:
        inv_rsn = (
            "Candle duplikat terdeteksi pada data historis"
            if any_dup
            else (
                "Urutan timestamp candle tidak berurutan (out-of-order)"
                if any_ooo
                else "Data OHLC mengandung nilai NaN / tidak konsisten (High < Low atau <= 0)"
            )
        )
        return {
            "status": "INVALID",
            "ok": False,
            "reason": inv_rsn,
            "market_status": mkt_st,
            "connection_status": conn_st,
            "tick_status": "INVALID" if mkt_st != "CLOSED" else "MARKET CLOSED",
            "candle_status": "INVALID",
            "data_validation": "FAIL",
            "last_candle_times": last_candle_times,
            "ages_minutes": ages_minutes,
            "stale_timeframes": stale_tfs,
            "is_market_closed": mkt_st == "CLOSED",
        }

    if not non_empty or mkt_st == "DATA UNAVAILABLE":
        return {
            "status": "NO DATA",
            "ok": False,
            "reason": "Data candle historis MT5 tidak tersedia",
            "market_status": mkt_st,
            "connection_status": "DISCONNECTED" if conn_st == "UNKNOWN" else conn_st,
            "tick_status": "NO DATA",
            "candle_status": "NO DATA",
            "data_validation": "FAIL",
            "last_candle_times": last_candle_times,
            "ages_minutes": ages_minutes,
            "stale_timeframes": stale_tfs,
            "is_market_closed": False,
        }

    for tf in ("H4", "H1", "M30", "M5"):
        df_tf = frames_map.get(tf)
        if df_tf is None or df_tf.empty or "Time" not in df_tf.columns:
            continue
        raw_t = df_tf["Time"].iloc[-1]
        src_tz = str((getattr(df_tf, "attrs", {}) or {}).get("time_tz", "WIB"))
        last_candle_times[tf] = format_wib_timestamp(raw_t, default_now=False, source_tz=src_tz) or str(raw_t)

    # Hitung usia candle terhadap server_time (hanya jika server_time bertipe waktu valid)
    stale_msg = None
    tick_stale = False
    candle_stale = False
    ref_srv_ts = None
    if isinstance(snapshot, dict) and snapshot.get("server_time") is not None:
        ref_srv_ts = snapshot.get("server_time")
    elif tick is not None and getattr(tick, "time", 0) and getattr(tick, "time", 0) > 0:
        ref_srv_ts = pd.Timestamp(int(getattr(tick, "time")), unit="s")
    elif isinstance(server_time, (pd.Timestamp, datetime)) or (isinstance(server_time, str) and server_time and "WIB" not in server_time):
        ref_srv_ts = server_time

    if ref_srv_ts is not None:
        try:
            srv_ts = pd.Timestamp(ref_srv_ts)
            if not pd.isna(srv_ts):
                for tf in ("M5", "M30", "H1", "H4"):
                    df_tf = frames_map.get(tf)
                    if df_tf is None or df_tf.empty or "Time" not in df_tf.columns:
                        continue
                    lts = pd.Timestamp(df_tf["Time"].iloc[-1])
                    if pd.isna(lts):
                        continue
                    src_tz = str((getattr(df_tf, "attrs", {}) or {}).get("time_tz", "WIB"))
                    s_al, l_al = _snr_align_ts(srv_ts, lts, b_tz_hint=src_tz)
                    age_min = (s_al - l_al).total_seconds() / 60.0
                    ages_minutes[tf] = round(age_min, 2)
                    tf_min = tf_minutes[tf]
                    if age_min < -tf_min:
                        stale_tfs.append(tf)
                        tick_stale = True
                        stale_msg = stale_msg or f"Tick lebih lama dari candle {tf} terakhir ({abs(age_min):.0f} menit; stale tick)"
                    elif mkt_st == "OPEN" and 0 <= age_min < tf_min:
                        stale_tfs.append(tf)
                        candle_stale = True
                        stale_msg = stale_msg or f"Candle {tf} terakhir belum dapat dipastikan tertutup."
                    elif age_min > tf_min * 3 + 5:
                        stale_tfs.append(tf)
                        candle_stale = True
                        stale_msg = stale_msg or f"Data {tf} terlambat ({age_min:.0f} menit sejak candle terakhir)"
        except Exception:
            pass

    if mkt_st == "CLOSED":
        return {
            "status": "STALE",
            "ok": False,
            "reason": stale_msg or "Market sedang tutup (data sesi penutupan terakhir; bukan error koneksi MT5)",
            "market_status": "CLOSED",
            "connection_status": conn_st,
            "tick_status": "MARKET CLOSED",
            "candle_status": "HISTORICAL (CLOSED)",
            "data_validation": "FAIL",
            "last_candle_times": last_candle_times,
            "ages_minutes": ages_minutes,
            "stale_timeframes": stale_tfs,
            "is_market_closed": True,
        }

    if stale_msg:
        return {
            "status": "STALE",
            "ok": False,
            "reason": stale_msg,
            "market_status": mkt_st,
            "connection_status": conn_st,
            "tick_status": "STALE" if tick_stale else "FRESH",
            "candle_status": "STALE" if candle_stale else "FRESH",
            "data_validation": "FAIL",
            "last_candle_times": last_candle_times,
            "ages_minutes": ages_minutes,
            "stale_timeframes": stale_tfs,
            "is_market_closed": False,
        }

    return {
        "status": "FRESH",
        "ok": True,
        "reason": "Data SNR lengkap dan segar",
        "market_status": mkt_st,
        "connection_status": conn_st,
        "tick_status": "FRESH",
        "candle_status": "FRESH",
        "data_validation": "PASS",
        "last_candle_times": last_candle_times,
        "ages_minutes": ages_minutes,
        "stale_timeframes": [],
        "is_market_closed": False,
    }


def fetch_snr_market_data(
    symbol=None,
    snapshot=None,
    frames=None,
    tick=None,
    live=True,
    has_unclosed_last_bar=False,
    market_status=None,
    config=None,
):
    """
    Mengumpulkan dan menormalisasi data pasar untuk SNR Advanced Scanner.
    Urutan prioritas sumber data:
    1. `frames` eksplisit jika diberikan.
    2. `snapshot["frames"]` jika diberikan (misal dari Unified Monitor agar 1x fetch per siklus).
    3. `_fetch_mt5_historical_candles(symbol, tf, bars)` dari MT5 jika `live=True`.
    """
    cfg = config or get_snr_config()
    sym = symbol or (snapshot.get("symbol") if isinstance(snapshot, dict) else None) or cfg.get("symbol", DEFAULT_APP_SYMBOL)
    tf_bars = cfg.get("timeframe_bars", {})
    snapshot_id = snapshot.get("snapshot_id") if isinstance(snapshot, dict) else None
    has_snapshot = isinstance(snapshot, dict) and ("frames" in snapshot or bool(snapshot_id))

    raw_frames = {}
    if isinstance(frames, dict) and frames:
        raw_frames = dict(frames)
    elif isinstance(snapshot, dict) and isinstance(snapshot.get("frames"), dict):
        raw_frames = dict(snapshot["frames"])

    mt5_ready = False
    if live and not has_snapshot and len(raw_frames) < len(SNR_TIMEFRAMES):
        try:
            mt5_ready = bool(pastikan_mt5_aktif())
        except Exception:
            mt5_ready = False
    elif live and not has_snapshot and mt5 is not None:
        try:
            mt5_ready = getattr(mt5, "terminal_info", lambda: None)() is not None
        except Exception:
            mt5_ready = False
    elif isinstance(snapshot, dict) and snapshot.get("mt5_connected") is not None:
        mt5_ready = bool(snapshot.get("mt5_connected"))

    prepared_frames = {}
    forming_candles = {}

    for tf in SNR_TIMEFRAMES:
        df_src = raw_frames.get(tf)
        if (df_src is None or (isinstance(df_src, pd.DataFrame) and df_src.empty)) and live and mt5_ready and not has_snapshot:
            df_src = _fetch_mt5_historical_candles(sym, tf, int(tf_bars.get(tf, 220)))

        df_clean, forming_bar = prepare_snr_frame(df_src, has_unclosed_last_bar=has_unclosed_last_bar)
        prepared_frames[tf] = df_clean
        if forming_bar is not None:
            forming_candles[tf] = forming_bar

    if isinstance(snapshot, dict) and isinstance(snapshot.get("forming_candles"), dict) and not forming_candles:
        forming_candles = dict(snapshot.get("forming_candles") or {})

    # Ambil informasi harga bid/ask/mid
    bid = 0.0
    ask = 0.0
    quote_valid = False
    time_label = "SNAPSHOT"
    server_time = None
    snapshot_time_wib = None
    snap_price_override = None

    if isinstance(snapshot, dict):
        bid = safe_float(snapshot.get("bid"), 0.0)
        ask = safe_float(snapshot.get("ask"), 0.0)
        quote_valid = bid > 0 and ask >= bid
        snap_price_override = safe_float(snapshot.get("snapshot_price") or snapshot.get("current_price") or snapshot.get("mid"), 0.0)
        if bid <= 0:
            bid = snap_price_override
        server_time = snapshot.get("server_time")
        snapshot_time_wib = snapshot.get("snapshot_time")
        if snapshot.get("live", False) or snapshot.get("mt5_connected", False):
            time_label = "MT5 LIVE"

    if tick is not None:
        t_bid = safe_float(getattr(tick, "bid", None) if not isinstance(tick, dict) else tick.get("bid"), 0.0)
        t_ask = safe_float(getattr(tick, "ask", None) if not isinstance(tick, dict) else tick.get("ask"), 0.0)
        quote_valid = t_bid > 0 and t_ask >= t_bid
        if t_bid > 0:
            bid = t_bid
        if t_ask > 0:
            ask = t_ask
        t_time = getattr(tick, "time", None) if not isinstance(tick, dict) else tick.get("time")
        if t_time and t_time > 0:
            snapshot_time_wib = snapshot_time_wib or format_wib_timestamp(t_time, default_now=True)
            if server_time is None:
                server_time = pd.Timestamp(int(t_time), unit="s", tz="UTC")
        time_label = "MT5 LIVE"
    elif bid <= 0 and live and mt5_ready and mt5 is not None and not has_snapshot:
        for sym_tick in (sym, DEFAULT_APP_SYMBOL, "XAUUSDc", "XAUUSD"):
            if not sym_tick:
                continue
            try:
                live_tick = mt5.symbol_info_tick(sym_tick)
                if live_tick is not None and safe_float(getattr(live_tick, "bid", 0.0), 0.0) > 0:
                    bid = safe_float(getattr(live_tick, "bid", 0.0), 0.0)
                    ask = safe_float(getattr(live_tick, "ask", 0.0), 0.0)
                    quote_valid = bid > 0 and ask >= bid
                    tick = live_tick
                    t_time = getattr(live_tick, "time", 0)
                    if t_time and t_time > 0:
                        snapshot_time_wib = snapshot_time_wib or format_wib_timestamp(t_time, default_now=True)
                        if server_time is None:
                            server_time = pd.Timestamp(int(t_time), unit="s", tz="UTC")
                    time_label = "MT5 LIVE"
                    break
            except Exception:
                continue

    # Deteksi status pasar (MARKET OPEN / MARKET CLOSED / DATA UNAVAILABLE)
    mkt_info = detect_snr_market_status(
        prepared_frames=prepared_frames,
        symbol=sym,
        tick=tick if tick is not None else ({"bid": bid, "ask": ask} if bid > 0 else None),
        live=live,
        snapshot=snapshot,
        market_status_override=market_status,
        config=cfg,
    )
    mkt_status = mkt_info["market_status"]
    data_source = mkt_info["data_source"]
    snr_analysis = mkt_info["snr_analysis"]
    last_candle = mkt_info["last_candle"]
    ref_tf_tz = "WIB"
    for tf_tz_cand in ("M5", "M30", "H1", "H4"):
        df_cand = prepared_frames.get(tf_tz_cand)
        if df_cand is not None and not df_cand.empty:
            ref_tf_tz = str((getattr(df_cand, "attrs", {}) or {}).get("time_tz", "WIB"))
            break
    last_candle_wib = format_wib_timestamp(last_candle, default_now=False, source_tz=ref_tf_tz) if last_candle != "DATA UNAVAILABLE" else "DATA UNAVAILABLE"

    # Saat MARKET CLOSED, gunakan harga penutupan candle terakhir (jangan klaim harga live)
    if mkt_status == "CLOSED":
        bid = safe_float(mkt_info["last_price"], 0.0)
        ask = 0.0
        time_label = "MT5 HISTORICAL (MARKET CLOSED)"
        if server_time is None and last_candle != "DATA UNAVAILABLE":
            server_time = last_candle
    elif mkt_status == "DATA UNAVAILABLE":
        bid = 0.0
        ask = 0.0
        time_label = "DATA UNAVAILABLE"
    else:
        # Fallback harga dari close candle closed terakhir (M5 -> M30 -> H1 -> H4) jika bid belum terisi
        if bid <= 0:
            bid = safe_float(mkt_info["last_price"], 0.0)
            if server_time is None and last_candle != "DATA UNAVAILABLE":
                server_time = last_candle

    spec = get_symbol_spec(symbol=sym, tick=tick if mkt_status == "OPEN" else None, config=cfg)
    spec["quote_valid"] = quote_valid and mkt_status == "OPEN"
    if ask <= 0 and bid > 0:
        ask = round(bid + spec["spread_points"], spec["digits"])
    spread = round(max(0.0, ask - bid), max(2, spec["digits"])) if (bid > 0 and ask > 0) else spec["spread_points"]
    spec["spread_points"] = spread
    price = round(bid, spec["digits"]) if bid > 0 else 0.0
    if isinstance(snapshot, dict) and snapshot.get("spread") is not None and safe_float(snapshot.get("spread"), -1.0) >= 0:
        # Pertahankan spread eksplisit dari shared snapshot bila ada
        snap_sp = safe_float(snapshot.get("spread"), spread)
        if snap_sp >= 0 and (tick is None):
            spread = round(snap_sp, max(2, spec["digits"]))
            spec["spread_points"] = spread
            if quote_valid and abs(snap_sp - (ask - bid)) > 0.5 * (10 ** -spec["digits"]) + 1e-9:
                spec["quote_valid"] = False

    conn_status = (
        str(snapshot.get("connection_status")).upper()
        if isinstance(snapshot, dict) and snapshot.get("connection_status")
        else ("CONNECTED" if mt5_ready else ("CALLER_SUPPLIED" if raw_frames else "DISCONNECTED"))
    )
    fresh_info = validate_snr_data_freshness(
        prepared_frames,
        server_time=server_time,
        tick=tick,
        market_status=mkt_status,
        connection_status=conn_status,
        bid=bid,
        ask=ask,
        snapshot=snapshot,
    )
    if fresh_info["status"] == "INVALID":
        snr_analysis = "DATA INVALID"
    elif fresh_info["status"] == "NO DATA":
        snr_analysis = "DATA UNAVAILABLE"

    if live and mt5_ready and not has_snapshot and mkt_status == "OPEN" and "M5" not in forming_candles:
        try:
            from modules.mt5_conn import ambil_candle_berjalan
            tf_m5 = _resolve_mt5_tf_constant("M5")
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

    snap_time_final = snapshot_time_wib or format_wib_timestamp(server_time, default_now=True, source_tz=ref_tf_tz if server_time == last_candle else "WIB")
    srv_time_final = format_wib_timestamp(server_time, default_now=True, source_tz=ref_tf_tz if server_time == last_candle else "WIB")
    tick_time_final = (
        snapshot.get("tick_time")
        if isinstance(snapshot, dict) and snapshot.get("tick_time")
        else (snapshot_time_wib or (snap_time_final if tick is not None else ""))
    )
    mid_val = round((bid + ask) / 2.0, spec["digits"]) if (bid > 0 and ask > 0) else price
    snap_price_final = (
        round(snap_price_override, spec["digits"])
        if (mkt_status == "OPEN" and snap_price_override and snap_price_override > 0)
        else price
    )

    atr_diag = compute_atr_diagnostics(prepared_frames)

    return {
        "symbol": sym,
        "snapshot_id": snapshot_id,
        "snapshot_time": snap_time_final,
        "snapshot_price": snap_price_final,
        "price": price,
        "last_price": price,
        "last_candle": last_candle_wib or last_candle,
        "last_candle_times": fresh_info["last_candle_times"],
        "last_closed_m5": fresh_info["last_candle_times"].get("M5", ""),
        "current_candle_m5": (forming_candles.get("M5") or {}).get("time", ""),
        "tick_time": tick_time_final,
        "market_status": mkt_status,
        "market_status_label": mkt_info["market_status_label"],
        "data_source": data_source,
        "snr_analysis": snr_analysis,
        "market_status_reason": mkt_info["reason"],
        "connection_status": fresh_info["connection_status"],
        "data_freshness": fresh_info["status"],
        "data_freshness_info": fresh_info,
        "bid": round(bid, spec["digits"]) if bid > 0 else 0.0,
        "ask": round(ask, spec["digits"]) if ask > 0 else 0.0,
        "mid": mid_val,
        "spread": spread,
        "time_label": time_label,
        "server_time": srv_time_final,
        "frames": prepared_frames,
        "forming_candles": forming_candles,
        "symbol_spec": spec,
        "atr_diagnostics": atr_diag,
    }

