# -*- coding: utf-8 -*-
"""
modules/utils.py
Fungsi helper umum: penanganan error, notifikasi suara, kalkulasi jarak zona, dan offset UTC/DST.
"""
import os
import queue
import threading
import time
import traceback
from datetime import datetime, timedelta, timezone
try:
    import winsound
except ImportError:
    winsound = None

import pandas as pd
from typing import Optional, Any
from modules.config import FILE_ERROR_LOG, SYMBOL, APP_VERSION, API_KEY

WIB_TZ = timezone(timedelta(hours=7))


def get_current_wib() -> datetime:
    """Mengembalikan waktu saat ini dalam timezone WIB (Asia/Jakarta, UTC+7)."""
    return datetime.now(WIB_TZ)


def to_wib_datetime(dt_val: Any) -> Optional[datetime]:
    """
    Konversi berbagai format waktu (timestamp epoch, string, pd.Timestamp, datetime)
    ke objek datetime dengan timezone WIB (UTC+7) yang valid.
    Mengembalikan None jika input tidak valid.
    """
    if dt_val is None:
        return None
    try:
        if isinstance(dt_val, (int, float)):
            if not pd.isna(dt_val) and dt_val > 0:
                return datetime.fromtimestamp(dt_val, tz=timezone.utc).astimezone(WIB_TZ)
            return None
        if isinstance(dt_val, str):
            s = str(dt_val).replace(" WIB", "").strip()
            if not s or s.lower() in ("none", "nan", "nat", "-"):
                return None
            ts = pd.Timestamp(s)
            if pd.isna(ts):
                return None
            if ts.tzinfo is not None:
                return ts.to_pydatetime().astimezone(WIB_TZ)
            return ts.to_pydatetime().replace(tzinfo=WIB_TZ)
        if hasattr(dt_val, "tzinfo") and getattr(dt_val, "tzinfo") is not None:
            return dt_val.astimezone(WIB_TZ)
        if isinstance(dt_val, datetime):
            return dt_val.replace(tzinfo=WIB_TZ)
        if hasattr(dt_val, "to_pydatetime"):
            pdt = dt_val.to_pydatetime()
            if getattr(pdt, "tzinfo", None) is not None:
                return pdt.astimezone(WIB_TZ)
            return pdt.replace(tzinfo=WIB_TZ)
    except Exception:
        return None
    return None

_ANTRIAN_BUNYI = queue.Queue()
_THREAD_BUNYI = None


class KonteksError(Exception):
    """Gagal menyusun konteks dari MT5 (koneksi/simbol/data). Ditangani pemanggil, tidak mematikan aplikasi."""


def _worker_bunyi():
    while True:
        f, d = _ANTRIAN_BUNYI.get()
        try:
            if winsound is not None:
                winsound.Beep(int(f), int(d))
        except Exception as e:
            _catat_error("_worker_bunyi", e)


def bunyi(freq, durasi):
    global _THREAD_BUNYI
    if _THREAD_BUNYI is None or not _THREAD_BUNYI.is_alive():
        _THREAD_BUNYI = threading.Thread(target=_worker_bunyi, daemon=True)
        _THREAD_BUNYI.start()
    _ANTRIAN_BUNYI.put((freq, durasi))


def _catat_error(konteks, err):
    try:
        with open(FILE_ERROR_LOG, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {konteks}: {err!r}\n")
            f.write(traceback.format_exc() + "\n")
    except Exception:
        pass


def wkt(t):
    return t.strftime("%m-%d %H:%M")


def jarak_zona(lo, hi, harga):
    if lo <= harga <= hi:
        return "HARGA DI DALAM ZONA"
    if lo > harga:
        return f"{lo - harga:.1f} di atas harga"
    return f"{harga - hi:.1f} di bawah harga"


def selisih_zona(z, harga):
    return 0 if z["lo"] <= harga <= z["hi"] else min(abs(z["lo"] - harga), abs(z["hi"] - harga))


def us_dst(dt):
    d = datetime(dt.year, 3, 8)
    while d.weekday() != 6:
        d += timedelta(days=1)
    mulai = d + timedelta(hours=7)
    e = datetime(dt.year, 11, 1)
    while e.weekday() != 6:
        e += timedelta(days=1)
    return mulai <= dt < e + timedelta(hours=6)


def deteksi_offset(tick, df_m5=None):
    """Deteksi offset UTC server broker dari tick live; jika market tutup, estimasi dari selisih jam bulat candle M5 terakhir."""
    if tick is not None:
        diff = (tick.time - time.time()) / 3600
        off = round(diff)
        if -3 <= off <= 5 and abs(diff - off) < 0.25:
            return off
    if df_m5 is not None and len(df_m5) > 0:
        last_ts = df_m5.index[-1]
        # Kebanyakan broker menutup Jumat 21:00/22:00 UTC (23:55 waktu server UTC+2 / UTC+3)
        if last_ts.weekday() == 4 and last_ts.hour in (22, 23):
            return 2 if last_ts.hour == 22 else 3
    return None


def format_startup_presentation(
    symbol: str = SYMBOL,
    mt5_connected: Optional[bool] = None,
    tick: Optional[Any] = None,
    engine_ready: Optional[bool] = None,
    ai_status: Optional[str] = None,
    version: str = APP_VERSION,
) -> str:
    """
    Format tampilan startup awal XAUUSD SMC Scanner:
    1. HEADER (Nama, deskripsi singkat, versi aplikasi)
    2. SYSTEM STATUS (MT5 Connection, Market Data, Python Scanner Engine, Gemini AI)
    3. MARKET OVERVIEW (Harga XAUUSD terkini, waktu update terakhir atau DATA UNAVAILABLE)
    4. COMMAND MENU (Daftar perintah utama ringkas)
    """
    out = []
    out.append("=" * 70)
    out.append(f"XAUUSD SMC SCANNER {version}".center(70))
    out.append("=" * 70)
    out.append("")

    # 1. Header
    out.append("1. HEADER")
    out.append("• Aplikasi             : XAUUSD SMC SCANNER")
    out.append("• Deskripsi            : SMC Multi-Timeframe Analysis & Institutional Risk Management")
    out.append(f"• Versi                : {version}")
    out.append("")

    # 2. System Status
    if mt5_connected is None:
        try:
            import MetaTrader5 as mt5
            mt5_connected = bool(mt5.terminal_info() is not None)
        except Exception:
            mt5_connected = False

    st_mt5 = "CONNECTED" if mt5_connected else "DISCONNECTED"

    if tick is None and mt5_connected:
        try:
            import MetaTrader5 as mt5
            tick = mt5.symbol_info_tick(symbol)
        except Exception:
            tick = None

    has_tick = bool(tick and getattr(tick, "bid", 0) > 0)
    if mt5_connected:
        st_market = f"READY ({symbol})" if has_tick else "DATA UNAVAILABLE"
    else:
        st_market = "DISCONNECTED"

    if engine_ready is None:
        engine_ready = True
    st_engine = "READY" if engine_ready else "ERROR"

    if ai_status is None:
        api_k = os.environ.get("GEMINI_API_KEY", "").strip() or API_KEY
        st_ai = "READY" if api_k else "DISCONNECTED"
    else:
        st_ai = ai_status

    out.append("2. SYSTEM STATUS")
    out.append(f"• MT5 Connection       : {st_mt5}")
    out.append(f"• Market Data          : {st_market}")
    out.append(f"• Python Scanner Engine: {st_engine}")
    out.append(f"• Gemini AI            : {st_ai}")
    out.append("")

    # 3. Market Overview
    out.append("3. MARKET OVERVIEW")
    if has_tick:
        bid_p = float(tick.bid)
        ask_p = float(tick.ask)
        spread_pt = abs(ask_p - bid_p)
        t_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S WIB")
        out.append(f"• Simbol               : {symbol}")
        out.append(f"• Harga Terkini        : Bid {bid_p:.2f} | Ask {ask_p:.2f} (Spread: {spread_pt:.2f} pt)")
        out.append(f"• Waktu Update         : {t_str}")
    else:
        out.append(f"• Simbol               : {symbol}")
        out.append("• Harga Terkini        : DATA UNAVAILABLE")
        out.append("• Waktu Update         : DATA UNAVAILABLE")
    out.append("")

    # 4. Command Menu
    out.append("4. COMMAND MENU")
    out.append("• scan       : Jalankan Scanner Zona SMC (5 Bagian Ringkas)")
    out.append("• scan detail: Analisis detail seluruh kandidat zona & breakdown skor")
    out.append("• zona       : Radar zona Order Block & FVG aktif")
    out.append("• pantau     : Monitor interaktif live price & kawal posisi terbuka")
    out.append("• jurnal     : Lihat riwayat & rekap performa sinyal jurnal")
    out.append("• audit      : Audit fakta akun MT5, pending order & status ADR")
    out.append("• refresh    : Update data real-time MT5 & perbarui analisa")
    out.append("• help       : Panduan lengkap seluruh perintah terminal")
    out.append("• exit       : Keluar dari aplikasi")
    out.append("=" * 70)

    return "\n".join(out)
