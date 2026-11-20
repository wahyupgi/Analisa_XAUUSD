# -*- coding: utf-8 -*-
"""
modules/utils.py
Fungsi helper umum: penanganan error, notifikasi suara, kalkulasi jarak zona, dan offset UTC/DST.
"""
import math
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
from typing import Optional, Any, Dict
import modules.config as _app_config
from modules.config import FILE_ERROR_LOG, SYMBOL, APP_VERSION, API_KEY

WIB_TZ = timezone(timedelta(hours=7))


def get_current_wib() -> datetime:
    """Mengembalikan waktu saat ini dalam timezone WIB (Asia/Jakarta, UTC+7)."""
    return datetime.now(WIB_TZ)


def to_wib_datetime(dt_val: Any, source_tz: str = "WIB") -> Optional[datetime]:
    """
    Konversi berbagai format waktu (timestamp epoch, string, pd.Timestamp, datetime)
    ke objek datetime dengan timezone WIB (UTC+7) yang valid.
    Konversi dilakukan tepat satu kali:
    - Jika input berupa epoch (int/float) atau sudah tz-aware, langsung dikonversi ke WIB.
    - Jika input tz-naive dan source_tz == "UTC", diperlakukan sebagai UTC lalu dikonversi ke WIB (+7).
    - Selain itu, input tz-naive dianggap sudah dalam WIB.
    Mengembalikan None jika input tidak valid.
    """
    if dt_val is None:
        return None
    src_is_utc = str(source_tz or "WIB").strip().upper() == "UTC"
    naive_tz = timezone.utc if src_is_utc else WIB_TZ
    try:
        if isinstance(dt_val, (int, float)):
            if not pd.isna(dt_val) and dt_val > 0:
                return datetime.fromtimestamp(dt_val, tz=timezone.utc).astimezone(WIB_TZ)
            return None
        if isinstance(dt_val, str):
            raw_s = str(dt_val).strip()
            has_wib_suffix = "WIB" in raw_s.upper()
            s = raw_s.replace(" WIB", "").replace(" wib", "").strip()
            if not s or s.lower() in ("none", "nan", "nat", "-"):
                return None
            ts = pd.Timestamp(s)
            if pd.isna(ts):
                return None
            if ts.tzinfo is not None:
                return ts.to_pydatetime().astimezone(WIB_TZ)
            tz_use = WIB_TZ if has_wib_suffix else naive_tz
            return ts.to_pydatetime().replace(tzinfo=tz_use).astimezone(WIB_TZ)
        if hasattr(dt_val, "tzinfo") and getattr(dt_val, "tzinfo") is not None:
            return dt_val.astimezone(WIB_TZ)
        if isinstance(dt_val, datetime):
            return dt_val.replace(tzinfo=naive_tz).astimezone(WIB_TZ)
        if hasattr(dt_val, "to_pydatetime"):
            pdt = dt_val.to_pydatetime()
            if getattr(pdt, "tzinfo", None) is not None:
                return pdt.astimezone(WIB_TZ)
            return pdt.replace(tzinfo=naive_tz).astimezone(WIB_TZ)
    except Exception:
        return None
    return None


def format_wib_timestamp(dt_val: Any = None, default_now: bool = False, source_tz: str = "WIB") -> str:
    """
    Memformat berbagai input waktu menjadi string standar WIB: 'YYYY-MM-DD HH:MM:SS WIB'.
    Jika dt_val tidak valid/None dan default_now=True, menggunakan waktu WIB saat ini.
    """
    dt = to_wib_datetime(dt_val, source_tz=source_tz) if dt_val is not None else None
    if dt is None:
        if not default_now:
            return ""
        dt = get_current_wib()
    return dt.strftime("%Y-%m-%d %H:%M:%S WIB")

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


# =============================================================================
# HELPER TERPUSAT SPESIFIKASI SIMBOL MT5 & KONVERSI SATUAN JARAK (XAUUSD)
# =============================================================================
# Membedakan secara tegas:
# 1. Price Distance (USD/oz) : abs(harga_zona_terdekat - harga_saat_ini), 0 jika di dalam zona
# 2. MT5 Points              : Price Distance / symbol.point
# 3. Pips                    : Price Distance / PIP_SIZE_PRICE (HANYA bila dikonfigurasi eksplisit)

def _safe_pos_float(val: Any) -> Optional[float]:
    try:
        f = float(val)
        return f if (math.isfinite(f) and f > 0.0) else None
    except (TypeError, ValueError):
        return None


def _configured_pip_size(pip_size: Optional[float] = None) -> Optional[float]:
    raw = pip_size if pip_size is not None else getattr(_app_config, "PIP_SIZE_PRICE", None)
    return _safe_pos_float(raw)


def get_symbol_unit_spec(
    symbol: Optional[str] = None,
    sym_info: Any = None,
    spec: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Mengambil `digits`, `point`, dan `trade_tick_size` langsung dari `mt5.symbol_info()`
    secara terpusat tanpa hardcode (fallback minimal hanya saat MT5 tidak tersedia/offline).
    """
    target_sym = str(symbol or SYMBOL or "XAUUSDc").strip() or SYMBOL
    pip_size = _configured_pip_size()

    if isinstance(spec, dict):
        sp_d = spec.get("digits")
        sp_p = _safe_pos_float(spec.get("point"))
        sp_ts = _safe_pos_float(spec.get("trade_tick_size", spec.get("tick_size")))
        if sp_d is not None and sp_p is not None and sp_ts is not None:
            d_int = max(0, min(8, int(sp_d)))
            return {
                "symbol": str(spec.get("symbol") or target_sym),
                "digits": d_int,
                "point": sp_p,
                "trade_tick_size": sp_ts,
                "tick_size": sp_ts,
                "tick_size_verified": bool(spec.get("tick_size_verified", spec.get("source") == "MT5_SYMBOL_INFO")),
                "trade_tick_value": _safe_pos_float(spec.get("trade_tick_value", spec.get("tick_value"))),
                "trade_contract_size": _safe_pos_float(spec.get("trade_contract_size", spec.get("contract_size"))),
                "trade_stops_level": spec.get("trade_stops_level"),
                "trade_freeze_level": spec.get("trade_freeze_level"),
                "spread_mt5_points": _safe_pos_float(spec.get("spread_mt5_points")),
                "pip_size": _configured_pip_size(spec.get("pip_size")),
                "source": str(spec.get("source") or "SNAPSHOT_SPEC"),
            }

    info = sym_info
    if info is None:
        try:
            import MetaTrader5 as mt5
            info = mt5.symbol_info(target_sym)
            if info is None and target_sym != SYMBOL:
                info = mt5.symbol_info(SYMBOL)
        except Exception:
            info = None

    digits: Optional[int] = None
    point: Optional[float] = None
    tick_size: Optional[float] = None
    tick_value: Optional[float] = None
    contract_size: Optional[float] = None
    stops_level: Optional[int] = None
    freeze_level: Optional[int] = None
    spread_mt5_pts: Optional[float] = None
    tick_size_verified = False
    source = "FALLBACK"

    if info is not None:
        d_raw = getattr(info, "digits", None)
        if d_raw is not None:
            try:
                d_cand = int(d_raw)
                if d_cand >= 0:
                    digits = max(0, min(8, d_cand))
            except (TypeError, ValueError):
                pass
        point = _safe_pos_float(getattr(info, "point", None))
        tick_size = _safe_pos_float(getattr(info, "trade_tick_size", None))
        if tick_size is not None:
            tick_size_verified = True
        tick_value = _safe_pos_float(getattr(info, "trade_tick_value", None))
        contract_size = _safe_pos_float(getattr(info, "trade_contract_size", None))
        raw_stops = getattr(info, "trade_stops_level", None)
        if raw_stops is not None:
            try:
                if int(raw_stops) >= 0:
                    stops_level = int(raw_stops)
            except (TypeError, ValueError):
                pass
        raw_freeze = getattr(info, "trade_freeze_level", None)
        if raw_freeze is not None:
            try:
                if int(raw_freeze) >= 0:
                    freeze_level = int(raw_freeze)
            except (TypeError, ValueError):
                pass
        spread_mt5_pts = _safe_pos_float(getattr(info, "spread", None))
        if digits is not None or point is not None or tick_size is not None:
            source = "MT5_SYMBOL_INFO"

    if digits is None:
        if point is not None and point > 0:
            digits = max(0, min(8, -int(math.floor(math.log10(point) + 1e-9))))
        else:
            digits = 2
    if point is None:
        point = round(10.0 ** (-digits), max(digits, 6))
    if tick_size is None:
        tick_size = point

    return {
        "symbol": target_sym,
        "digits": digits,
        "point": point,
        "trade_tick_size": tick_size,
        "tick_size": tick_size,
        "tick_size_verified": tick_size_verified,
        "trade_tick_value": tick_value,
        "trade_contract_size": contract_size,
        "trade_stops_level": stops_level,
        "trade_freeze_level": freeze_level,
        "spread_mt5_points": spread_mt5_pts,
        "pip_size": pip_size,
        "source": source,
    }


def extract_unit_spec_from_container(container: Any = None, symbol: Optional[str] = None) -> Dict[str, Any]:
    """Ambil unit spec dari dict scan/snapshot/symbol_spec jika sudah ada, atau langsung dari mt5.symbol_info()."""
    sym = symbol
    if isinstance(container, dict):
        sym = container.get("symbol") or sym
        for sub_key in ("symbol_spec", "unit_spec", "snapshot"):
            sub = container.get(sub_key)
            if isinstance(sub, dict):
                if sub.get("digits") is not None and _safe_pos_float(sub.get("point")) is not None:
                    return get_symbol_unit_spec(symbol=sym, spec=sub)
        if container.get("digits") is not None and _safe_pos_float(container.get("point")) is not None:
            return get_symbol_unit_spec(symbol=sym, spec=container)
    return get_symbol_unit_spec(symbol=sym)


def calc_price_distance(zone_low: float, zone_high: float, current_price: float) -> float:
    """
    Price Distance (USD/oz) = abs(harga zona terdekat - harga saat ini), atau 0.0 jika harga berada dalam zona.
    """
    lo = float(zone_low)
    hi = float(zone_high)
    cp = float(current_price)
    if lo > hi:
        lo, hi = hi, lo
    if lo <= cp <= hi:
        return 0.0
    return abs(lo - cp) if cp < lo else abs(cp - hi)


def price_distance_to_mt5_points(
    price_distance: float,
    point: Optional[float] = None,
    symbol: Optional[str] = None,
    sym_info: Any = None,
    spec: Optional[Dict[str, Any]] = None,
) -> float:
    """MT5 Points = Price Distance / symbol.point."""
    p_val = _safe_pos_float(point)
    if p_val is None:
        p_val = get_symbol_unit_spec(symbol=symbol, sym_info=sym_info, spec=spec)["point"]
    raw = float(price_distance) / p_val
    rounded_int = round(raw)
    if abs(raw - rounded_int) < 1e-6:
        return int(rounded_int)
    return round(raw, 2)


def mt5_points_to_price_distance(
    mt5_points: float,
    point: Optional[float] = None,
    symbol: Optional[str] = None,
    sym_info: Any = None,
    spec: Optional[Dict[str, Any]] = None,
) -> float:
    """Price Distance (USD/oz) = MT5 Points * symbol.point."""
    u_spec = get_symbol_unit_spec(symbol=symbol, sym_info=sym_info, spec=spec)
    p_val = _safe_pos_float(point) or u_spec["point"]
    return round(float(mt5_points) * p_val, max(2, int(u_spec["digits"])))


def price_distance_to_pips(
    price_distance: float,
    pip_size: Optional[float] = None,
) -> Optional[float]:
    """Konversi Price Distance ke Pips HANYA bila konvensi `PIP_SIZE_PRICE` dikonfigurasi secara eksplisit."""
    ps = _configured_pip_size(pip_size)
    if ps is None:
        return None
    raw = float(price_distance) / ps
    rounded_int = round(raw)
    if abs(raw - rounded_int) < 1e-6:
        return float(rounded_int)
    return round(raw, 2)


def format_mt5_points(mt5_pts: float, signed: bool = False) -> str:
    """Format nilai MT5 Points secara rapi (bulat jika eksak, 1-2 desimal jika pecahan)."""
    val = float(mt5_pts)
    if abs(val - round(val)) < 1e-6:
        iv = int(round(val))
        return f"{iv:+d} MT5 pts" if signed else f"{iv} MT5 pts"
    return f"{val:+.2f} MT5 pts" if signed else f"{val:.2f} MT5 pts"


def format_price_distance(
    price_distance: float,
    symbol: Optional[str] = None,
    sym_info: Any = None,
    spec: Optional[Dict[str, Any]] = None,
    include_mt5_points: bool = False,
    include_pips: bool = False,
    signed: bool = False,
    digits: Optional[int] = None,
    unit_label: str = "USD",
) -> str:
    """
    Memformat Price Distance (USD/oz) secara konsisten sesuai `digits` dan `point` dari `mt5.symbol_info()`.
    - Default: `5.25 USD`
    - Dengan `include_mt5_points=True`: `5.25 USD (525 MT5 pts)`
    - Pips hanya ditampilkan jika `include_pips=True` DAN `PIP_SIZE_PRICE` dikonfigurasi secara eksplisit.
    """
    u_spec = get_symbol_unit_spec(symbol=symbol, sym_info=sym_info, spec=spec)
    d = max(0, min(8, int(digits))) if digits is not None else int(u_spec["digits"])
    val = float(price_distance)
    base = f"{val:+.{d}f} {unit_label}" if signed else f"{val:.{d}f} {unit_label}"
    extras = []
    if include_mt5_points:
        pts_val = price_distance_to_mt5_points(val, spec=u_spec)
        extras.append(format_mt5_points(pts_val, signed=signed))
    if include_pips:
        pips_val = price_distance_to_pips(val, pip_size=u_spec.get("pip_size"))
        if pips_val is not None:
            if abs(pips_val - round(pips_val)) < 1e-6:
                extras.append(f"{int(round(pips_val)):+d} pips" if signed else f"{int(round(pips_val))} pips")
            else:
                extras.append(f"{pips_val:+.1f} pips" if signed else f"{pips_val:.1f} pips")
    if extras:
        return f"{base} ({' | '.join(extras)})"
    return base


def jarak_zona(lo, hi, harga, symbol=None, spec=None):
    dist = calc_price_distance(lo, hi, harga)
    if dist == 0.0:
        return "HARGA DI DALAM ZONA"
    dist_txt = format_price_distance(dist, symbol=symbol, spec=spec, include_mt5_points=True)
    if float(lo) > float(harga):
        return f"{dist_txt} di atas harga"
    return f"{dist_txt} di bawah harga"


def selisih_zona(z, harga):
    return calc_price_distance(z["lo"], z["hi"], harga)


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
        u_spec = get_symbol_unit_spec(symbol=symbol)
        d_px = u_spec["digits"]
        bid_p = float(tick.bid)
        ask_p = float(tick.ask)
        spread_usd = abs(ask_p - bid_p)
        spread_str = format_price_distance(spread_usd, spec=u_spec, include_mt5_points=True)
        t_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S WIB")
        out.append(f"• Simbol               : {symbol} (Digits: {d_px} | Point: {u_spec['point']:g} | TickSize: {u_spec['trade_tick_size']:g})")
        out.append(f"• Harga Terkini        : Bid {bid_p:.{d_px}f} | Ask {ask_p:.{d_px}f} (Spread: {spread_str})")
        out.append(f"• Waktu Update         : {t_str}")
    else:
        out.append(f"• Simbol               : {symbol}")
        out.append("• Harga Terkini        : DATA UNAVAILABLE")
        out.append("• Waktu Update         : DATA UNAVAILABLE")
    out.append("")

    # 4. Command Menu
    out.append("4. COMMAND MENU")
    out.append("ANALYSIS:")
    out.append("- scan             : Scan SMC")
    out.append("- scan detail      : Scan SMC detail")
    out.append("- scan snr         : Scan SNR")
    out.append("- scan snr detail  : Scan SNR detail")
    out.append("")
    out.append("MONITORING:")
    out.append("- pantau           : Unified Monitor (SMC + SNR + Account)")
    out.append("- pantau detail    : Unified Monitor detail")
    out.append("- pantau smc       : Monitor SMC saja")
    out.append("- pantau smc detail: Monitor SMC detail")
    out.append("- pantau snr       : Monitor SNR saja")
    out.append("- pantau snr detail: Monitor SNR detail")
    out.append("")
    out.append("OTHER:")
    out.append("- zona             : Radar zona Order Block & FVG aktif")
    out.append("- jurnal           : Lihat riwayat & rekap performa sinyal jurnal")
    out.append("- audit            : Audit fakta akun MT5, pending order & status ADR")
    out.append("- refresh          : Update data real-time MT5 & perbarui analisa")
    out.append("- help             : Panduan lengkap seluruh perintah terminal")
    out.append("- exit             : Keluar dari aplikasi")
    out.append("=" * 70)

    return "\n".join(out)
