# -*- coding: utf-8 -*-
"""
modules/utils.py
Fungsi helper umum: penanganan error, notifikasi suara, kalkulasi jarak zona, dan offset UTC/DST.
"""
import json
import math
import os
import queue
import re
import threading
import time
import traceback
from datetime import datetime, timedelta, timezone
try:
    import winsound
except ImportError:
    winsound = None

import pandas as pd
from typing import Optional, Any, Dict, List, Tuple
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


def _parse_event_timestamp_epoch(raw_ts) -> float:
    """Konversi aman timestamp event (epoch / ISO / string WIB / Timestamp) ke epoch detik; 0.0 bila gagal."""
    if raw_ts is None or isinstance(raw_ts, bool):
        return 0.0
    if isinstance(raw_ts, (int, float)):
        val = float(raw_ts)
        if not math.isfinite(val) or val <= 0:
            return 0.0
        return val / 1000.0 if val > 1e12 else val
    if hasattr(raw_ts, "timestamp"):
        try:
            val = float(raw_ts.timestamp())
            return val if math.isfinite(val) and val > 0 else 0.0
        except Exception:
            return 0.0
    txt = str(raw_ts).strip()
    if not txt or txt.upper() in ("N/A", "NONE", "UNKNOWN"):
        return 0.0
    # Strip trailing timezone tags like "WIB" / "UTC"
    cleaned = txt
    tz_offset_sec = 0.0
    if cleaned.upper().endswith(" WIB"):
        cleaned = cleaned[:-4].strip()
        tz_offset_sec = 7 * 3600.0
    elif cleaned.upper().endswith(" UTC"):
        cleaned = cleaned[:-4].strip()
    elif cleaned.endswith("Z"):
        cleaned = cleaned[:-1].strip()
    cleaned = cleaned.replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(cleaned[:19] if len(cleaned) >= 19 else cleaned, fmt)
            return float(dt.replace(tzinfo=timezone.utc).timestamp()) - tz_offset_sec
        except Exception:
            continue
    try:
        dt_iso = datetime.fromisoformat(txt.replace("Z", "+00:00"))
        if dt_iso.tzinfo is None:
            dt_iso = dt_iso.replace(tzinfo=timezone.utc)
        return float(dt_iso.timestamp())
    except Exception:
        return 0.0


def resolve_entry_blocker_code(
    reason_text=None,
    *,
    market_status=None,
    data_ok=True,
    htf_conflict=False,
    m30_conflict=False,
    h4_bias=None,
    has_active_zone=True,
    zone_touch=None,
    monitor_state=None,
    rr_failed=False,
    is_missed=False,
    is_stale=False,
) -> str:
    """Petakan alasan blocker entry menjadi kode singkat kanonis tanpa memotong kalimat mentah."""
    mkt = str(market_status or "").strip().upper()
    if mkt in ("CLOSED", "MARKET_CLOSED", "MARKET CLOSED"):
        return "MARKET_CLOSED"
    if mkt in ("DATA UNAVAILABLE", "DATA_UNAVAILABLE", "UNAVAILABLE", "NO_DATA"):
        return "DATA_UNAVAILABLE"

    txt = str(reason_text or "").replace("\n", " ").strip()
    up = txt.upper()
    low = txt.lower()

    if not data_ok:
        if "stale" in low or "expired" in low or "price changed" in low:
            return "DATA_STALE"
        return "DATA_UNAVAILABLE"
    if is_stale or "entry stale" in low:
        return "ENTRY_STALE"
    if is_missed or "terlewat" in low or "entry missed" in low or "entry_missed" in low:
        return "ENTRY_MISSED"
    if htf_conflict or m30_conflict or any(tok in low for tok in ("konflik struktur", "konflik h4", "konflik m30", "htf_conflict", "htf conflict")):
        return "HTF_CONFLICT"
    if "konflik smc" in low or "terhalang zona s&r" in low or "snr conflict" in low or "overlap" in low and "konflik" in low:
        return "SNR_CONFLICT"
    if "berita" in low or "news" in low:
        return "NEWS_BLOCKED"
    if "spread" in low and any(tok in low for tok in ("tinggi", "melebar", "wide", "high", "batas")):
        return "SPREAD_HIGH"
    if rr_failed or "rr_failed" in low or "risk/reward" in low or "rr <" in low or "rr tidak" in low or "min rr" in low:
        return "RR_FAILED"
    if "jarak sl" in low or ("sl" in low and any(tok in low for tok in ("melebihi batas", "invalid", "tidak valid", "terlalu"))):
        return "SL_INVALID"
    if str(monitor_state or "").upper() == "INVALIDATED" or any(tok in low for tok in ("zone invalidated", "zona invalid", "terinvalidasi", "sl_breached", "broken")):
        return "ZONE_INVALID"
    if str(monitor_state or "").upper() == "EXPIRED" or "expired" in low or "kedaluwarsa" in low:
        return "SETUP_EXPIRED"
    h4_up = str(h4_bias or "").strip().upper()
    if h4_up in ("NETRAL", "NEUTRAL", "SIDEWAYS") or "bias h4 netral" in low or "choppy" in low:
        return "HTF_NEUTRAL"
    if (
        not has_active_zone
        or any(tok in low for tok in (
            "no_setup", "no setup", "no active", "tidak ada zona", "belum ada setup",
            "far reference", "far_reference", "di luar batas intraday", "no_actionable",
        ))
    ):
        return "NO_SETUP"
    zt_up = str(zone_touch or "").strip().upper()
    if (
        zt_up in ("FAIL", "WAIT", "NOT_TOUCHED", "NO_ZONE")
        or str(monitor_state or "").upper() in ("APPROACHING", "RADAR", "IDLE")
        or any(tok in low for tok in (
            "belum menyentuh", "approaching", "menunggu pullback", "waiting for zone",
            "wait zone touch", "di luar zona", "belum di zona",
        ))
    ):
        return "WAIT_ZONE_TOUCH"
    if any(tok in low for tok in (
        "m5", "sweep", "choch", "bos", "cisd", "displacement", "rejection",
        "retest", "trigger", "belum dikonfirmasi", "belum terkonfirmasi", "konfirmasi",
        "awaiting_confirmation", "zone_touched", "zone touched",
    )):
        return "M5_UNCONFIRMED"
    if up in ("HTF_CONFLICT", "NO_SETUP", "M5_UNCONFIRMED", "WAIT_ZONE_TOUCH", "RR_FAILED", "MARKET_CLOSED", "DATA_STALE", "DATA_UNAVAILABLE"):
        return up
    return "M5_UNCONFIRMED" if has_active_zone else "NO_SETUP"


def format_mt5_sync_short(mt5_snapshot=None, fallback_ts=None) -> str:
    """Format singkat timestamp sinkronisasi MT5 read-only ('HH:MM:SS WIB') untuk audit antar-snapshot."""
    raw = None
    if isinstance(mt5_snapshot, dict):
        raw = mt5_snapshot.get("retrieved_at") or mt5_snapshot.get("synced_at")
    if not raw and fallback_ts:
        raw = fallback_ts
    if not raw:
        return "N/A"
    s = str(raw).strip()
    if len(s) >= 19 and s[10] in (" ", "T") and s[13] == ":" and s[16] == ":":
        return f"{s[11:19]} WIB"
    dt = to_wib_datetime(s)
    if dt is not None:
        return dt.strftime("%H:%M:%S WIB")
    return s


def _norm_direction_for_invalid(direction=None) -> str:
    u = str(direction or "").strip().upper()
    if u in ("BUY", "BULLISH"):
        return "BUY"
    if u in ("SELL", "BEARISH"):
        return "SELL"
    return ""


def _invalid_grade_rank(item_or_zone, inner) -> int:
    """Peringkat kualitas zona INVALID: 0=A+/A/B, 1=C/unknown, 2=D/low."""
    raw_g = ""
    for src_d in (inner, item_or_zone):
        if not isinstance(src_d, dict):
            continue
        for k_g in ("original_grade", "grade", "zone_grade"):
            g_cand = str(src_d.get(k_g) or "").strip().upper()
            if g_cand and g_cand not in ("REJECTED", "N/A", "NONE"):
                raw_g = g_cand
                break
        if raw_g:
            break
    if raw_g in ("A+", "A", "B"):
        return 0
    if raw_g == "C":
        return 1
    if raw_g == "D":
        return 2
    sc_val = None
    for src_d in (inner, item_or_zone):
        if isinstance(src_d, dict) and src_d.get("score") is not None:
            try:
                f = float(src_d.get("score"))
                if math.isfinite(f) and f > 0.0:
                    sc_val = f
                    break
            except (TypeError, ValueError):
                pass
    if sc_val is not None:
        if sc_val >= 70.0:
            return 0
        if sc_val >= 55.0:
            return 1
        return 2
    return 1


def _extract_invalid_zone_meta(item_or_zone, current_price=None, atr=None, direction=None):
    """Ekstrak metadata aman (zone dict, bounds, distance, event_ts, quality, direction_match, zone_id) untuk seleksi INVALID."""
    if not isinstance(item_or_zone, dict):
        return None
    inner = item_or_zone.get("zone") if isinstance(item_or_zone.get("zone"), dict) else item_or_zone
    lo = item_or_zone.get("price_low", item_or_zone.get("low"))
    hi = item_or_zone.get("price_high", item_or_zone.get("high"))
    if lo is None:
        lo = inner.get("price_low", inner.get("low"))
    if hi is None:
        hi = inner.get("price_high", inner.get("high"))
    try:
        lo_f = float(lo or 0.0)
        hi_f = float(hi if hi is not None else lo_f)
    except (TypeError, ValueError):
        return None
    if lo_f <= 0.0 and hi_f <= 0.0:
        return None
    if lo_f > hi_f > 0.0:
        lo_f, hi_f = hi_f, lo_f

    cp = 0.0
    try:
        cp = float(current_price or 0.0)
    except (TypeError, ValueError):
        cp = 0.0

    if cp > 0.0:
        if lo_f <= cp <= hi_f:
            dist = 0.0
        else:
            dist = min(abs(cp - lo_f), abs(cp - hi_f))
    else:
        raw_d = item_or_zone.get("distance_points", item_or_zone.get("distance", inner.get("distance_points", inner.get("distance"))))
        try:
            dist = float(raw_d) if raw_d is not None else float("inf")
        except (TypeError, ValueError):
            dist = float("inf")

    eff_atr = 0.0
    for atr_cand in (atr, inner.get("ref_atr"), inner.get("atr_value"), inner.get("atr_at_formation")):
        try:
            v = float(atr_cand or 0.0)
            if v > 0.0 and math.isfinite(v):
                eff_atr = v
                break
        except (TypeError, ValueError):
            continue
    if eff_atr <= 0.0:
        eff_atr = 10.0

    raw_d_atr = item_or_zone.get("distance_atr", inner.get("distance_atr"))
    d_atr = None
    if cp > 0.0 and math.isfinite(dist):
        d_atr = dist / eff_atr
    elif raw_d_atr is not None:
        try:
            v = float(raw_d_atr)
            if math.isfinite(v) and v >= 0.0:
                d_atr = v
        except (TypeError, ValueError):
            d_atr = None

    bo = inner.get("breakout_info") if isinstance(inner.get("breakout_info"), dict) else {}
    rt = inner.get("retest_info") if isinstance(inner.get("retest_info"), dict) else {}
    is_hist_flag = bool(
        item_or_zone.get("is_distant_historical")
        or item_or_zone.get("is_far_reference")
        or inner.get("is_distant_historical")
        or inner.get("is_far_reference")
        or inner.get("lifecycle_is_historical")
        or bo.get("is_historical_break")
        or str(inner.get("zone_relevance") or "").upper() == "FAR_REFERENCE"
    )
    near_dist_cap = max(35.0, 6.0 * eff_atr)
    mod_dist_cap = max(75.0, 12.0 * eff_atr)
    is_near = bool((not is_hist_flag) and ((d_atr is not None and d_atr <= 6.0) or (math.isfinite(dist) and dist <= near_dist_cap)))
    is_moderate = bool((not is_hist_flag) and (not is_near) and ((d_atr is not None and d_atr <= 12.0) or (math.isfinite(dist) and dist <= mod_dist_cap)))
    is_far_or_hist = not (is_near or is_moderate)
    relevance_tier = 0 if is_near else (1 if is_moderate else 2)

    inv_type_u = str(
        item_or_zone.get("invalidation_type")
        or inner.get("invalidation_type")
        or item_or_zone.get("invalid_label")
        or inner.get("invalid_label")
        or ""
    ).strip().upper()
    is_setup_sl_breach = inv_type_u == "SETUP_INVALID_SL_BREACHED"
    grade_rank = _invalid_grade_rank(item_or_zone, inner)
    is_low_quality = bool(grade_rank >= 2 and not (is_setup_sl_breach and not is_far_or_hist))

    # Prioritaskan timestamp event invalidasi aktual (break_time / bar break) di atas waktu scan umum (invalidated_at)
    if is_setup_sl_breach:
        ts_candidates = (
            item_or_zone.get("break_time"),
            inner.get("break_time"),
            item_or_zone.get("invalidation_time"),
            inner.get("invalidation_time"),
            item_or_zone.get("invalidated_at"),
            inner.get("invalidated_at"),
            item_or_zone.get("event_time"),
        )
    else:
        ts_candidates = (
            item_or_zone.get("break_time"),
            inner.get("break_time"),
            rt.get("retest_bar_time") if (rt.get("breakout_invalidated") or str(rt.get("status") or "").upper() == "RETEST_FAILED") else None,
            bo.get("breakout_bar_time"),
            item_or_zone.get("event_time"),
            item_or_zone.get("lifecycle_event_time"),
            inner.get("lifecycle_event_time"),
            item_or_zone.get("invalidation_time"),
            inner.get("invalidation_time"),
            inner.get("broken_at"),
            item_or_zone.get("invalidated_at"),
            inner.get("invalidated_at"),
        )
    ts_epoch = 0.0
    for cand_ts in ts_candidates:
        ep = _parse_event_timestamp_epoch(cand_ts)
        if ep > 0.0:
            ts_epoch = ep
            break

    zid = str(item_or_zone.get("zone_id") or inner.get("zone_id") or "").strip()
    tf = str(item_or_zone.get("timeframe") or inner.get("timeframe") or "").strip().upper()
    zt = str(item_or_zone.get("zone_type") or inner.get("zone_type") or inner.get("type") or "").strip().upper()

    norm_dir = _norm_direction_for_invalid(direction)
    if norm_dir == "BUY" and zt in ("DEMAND", "SUPPORT"):
        dir_mismatch = 0
    elif norm_dir == "SELL" and zt in ("SUPPLY", "RESISTANCE"):
        dir_mismatch = 0
    elif norm_dir and zt in ("DEMAND", "SUPPORT", "SUPPLY", "RESISTANCE"):
        dir_mismatch = 1
    else:
        dir_mismatch = 0

    if (not is_far_or_hist) and (not is_low_quality) and dir_mismatch == 0:
        priority_bucket = 0
    elif (not is_far_or_hist) and (not is_low_quality):
        priority_bucket = 1
    elif is_near and is_low_quality:
        priority_bucket = 2
    elif not is_low_quality:
        priority_bucket = 3
    else:
        priority_bucket = 4

    dedup_key = zid.upper() if zid and zid.upper() not in ("N/A", "UNKNOWN") else f"{tf}_{zt}_{lo_f:.2f}_{hi_f:.2f}"
    return {
        "item": item_or_zone,
        "zone": inner,
        "low": lo_f,
        "high": hi_f,
        "dist": dist,
        "d_atr": d_atr,
        "relevance_tier": relevance_tier,
        "is_near": is_near,
        "is_far_or_hist": is_far_or_hist,
        "grade_rank": grade_rank,
        "is_low_quality": is_low_quality,
        "is_setup_sl_breach": is_setup_sl_breach,
        "dir_mismatch": dir_mismatch,
        "priority_bucket": priority_bucket,
        "ts_epoch": ts_epoch,
        "zone_id": zid,
        "dedup_key": dedup_key,
    }


def select_relevant_invalid_zone(candidates, current_price=None, atr=None, direction=None):
    """
    Pilih satu zona INVALID paling relevan terhadap harga, direction, kualitas, dan terbaru:
    1. Prioritaskan zona relevan (non-historis/non-jauh) berkualitas memadai & searah direction (Bucket 0 -> 1).
    2. Zona Grade D historis jauh (Bucket 4) tidak boleh mengalahkan invalidasi setup/zona relevan yang lebih baru.
    3. Dalam bucket yang sama, prioritaskan timestamp event invalidasi terbaru, lalu kedekatan harga & grade.
    """
    if not candidates:
        return None
    best_by_key = {}
    for cand in candidates:
        meta = _extract_invalid_zone_meta(cand, current_price=current_price, atr=atr, direction=direction)
        if meta is None:
            continue
        sort_key = (
            meta["priority_bucket"],
            0 if meta["ts_epoch"] > 0.0 else 1,
            -meta["ts_epoch"],
            0 if meta["is_near"] else 1,
            meta["grade_rank"],
            round(meta["dist"], 4) if math.isfinite(meta["dist"]) else float("inf"),
            meta["zone_id"] or meta["dedup_key"],
        )
        dk = meta["dedup_key"]
        prev = best_by_key.get(dk)
        if prev is None or sort_key < prev[0]:
            best_by_key[dk] = (sort_key, meta["item"])
    if not best_by_key:
        return None
    best_item = min(best_by_key.values(), key=lambda pair: pair[0])[1]
    return best_item


def format_invalid_zone_compact(item_or_zone, current_price=None, unit_spec=None, status_override=None) -> str:
    """Format zona INVALID konsisten dengan pemisahan tegas StructInv (batas zona) vs SetupSL (SL entry)."""
    meta = _extract_invalid_zone_meta(item_or_zone, current_price=current_price)
    if meta is None:
        return "NONE"
    raw = meta["item"]
    z = meta["zone"]
    tf = str(raw.get("timeframe") or z.get("timeframe") or "").strip().upper()
    zt = str(raw.get("zone_type") or z.get("zone_type") or z.get("type") or "ZONE").strip().upper()
    rk = str(z.get("rank_label") or raw.get("rank_label") or "").strip().upper()
    head_lbl = f"{tf} {rk} {zt}".strip() if (rk and rk not in zt) else f"{tf} {zt}".strip()
    lo_f, hi_f = meta["low"], meta["high"]

    grd = ""
    for src_g in (z, raw):
        if isinstance(src_g, dict):
            for k_g in ("original_grade", "grade", "zone_grade"):
                g_c = str(src_g.get(k_g) or "").strip()
                if g_c and g_c.upper() not in ("REJECTED", "N/A", "NONE"):
                    grd = g_c
                    break
        if grd:
            break
    sc_raw = z.get("score") if z.get("score") is not None else raw.get("score")
    g_part = ""
    if grd:
        if sc_raw is not None and str(z.get("scoring_status") or "").upper() != "DATA_INCOMPLETE":
            try:
                g_part = f" | G:{grd}/{float(sc_raw):.1f}"
            except (TypeError, ValueError):
                g_part = f" | G:{grd}"
        else:
            g_part = f" | G:{grd}"

    if math.isfinite(meta["dist"]):
        d_txt = "IN ZONE" if meta["dist"] <= 0.0 else format_price_distance(meta["dist"], spec=unit_spec)
        dist_part = f" | Dist {d_txt}"
    else:
        dist_part = ""

    # Pisahkan STRUCTURE_INV (batas invalidasi zona) dari SETUP_SL (SL entry)
    struct_inv = 0.0
    for s_k in ("structure_invalidation", "structural_invalidation", "invalidation_price", "invalidation"):
        try:
            v = float(z.get(s_k) or raw.get(s_k) or 0.0)
            if v > 0.0 and math.isfinite(v):
                struct_inv = v
                break
        except (TypeError, ValueError):
            continue
    if struct_inv <= 0.0 and lo_f > 0.0:
        struct_inv = lo_f if zt in ("DEMAND", "SUPPORT") else (hi_f if zt in ("SUPPLY", "RESISTANCE") else 0.0)

    setup_sl = 0.0
    for sl_k in ("setup_sl", "setup_invalidation_price", "sl"):
        try:
            v = float(z.get(sl_k) or raw.get(sl_k) or 0.0)
            if v > 0.0 and math.isfinite(v):
                setup_sl = v
                break
        except (TypeError, ValueError):
            continue
    # Jangan tampilkan SetupSL yang identik dengan batas zona tanpa perhitungan SL/buffer
    if setup_sl > 0.0 and struct_inv > 0.0 and abs(setup_sl - struct_inv) < 1e-6:
        if not (z.get("sl") or raw.get("sl")):
            setup_sl = 0.0

    st_txt = str(status_override or "").strip().upper()
    if not st_txt:
        st_txt = str(
            raw.get("state")
            or z.get("invalidation_type")
            or z.get("lifecycle_event")
            or z.get("invalid_label")
            or z.get("lifecycle_status")
            or z.get("status")
            or "INVALIDATED"
        ).strip().upper()
    if st_txt in ("BREAKOUT_INVALIDATED", "BREAKDOWN_INVALIDATED"):
        if zt == "SUPPORT":
            st_txt = "BREAKDOWN_INVALIDATED"
        elif zt == "RESISTANCE":
            st_txt = "BREAKOUT_INVALIDATED"

    inv_parts = []
    if st_txt == "SETUP_INVALID_SL_BREACHED":
        if setup_sl > 0.0:
            inv_parts.append(f"SetupSL {setup_sl:.2f}")
        if struct_inv > 0.0:
            inv_parts.append(f"StructInv {struct_inv:.2f}")
    else:
        if struct_inv > 0.0:
            inv_parts.append(f"StructInv {struct_inv:.2f}")
        if setup_sl > 0.0 and abs(setup_sl - struct_inv) >= 0.01:
            inv_parts.append(f"SetupSL {setup_sl:.2f}")
    inv_part = (" | " + " | ".join(inv_parts)) if inv_parts else ""

    zid = meta["zone_id"]
    id_part = f" [ID:{zid}]" if zid and zid.upper() not in ("N/A", "UNKNOWN") else ""
    return f"{head_lbl} {lo_f:.2f}-{hi_f:.2f}{g_part}{dist_part}{inv_part} | {st_txt}{id_part}".replace("  ", " ").strip()


_FROZEN_TRADE_PLANS: Dict[str, Dict[str, Any]] = {}
_ACTIVE_TRADE_PLAN_FOCUS: Dict[str, Dict[str, Any]] = {}
_PREVIOUS_TRADE_PLANS: Dict[str, Dict[str, Any]] = {}
_INVALIDATED_SETUP_VERSIONS: Dict[str, Dict[str, Any]] = {}
_EMITTED_PLAN_SWITCHES: List[str] = []
_TRADE_PLAN_STATE_LOADED = False
_TRADE_PLAN_ATR_RECALC_RATIO = 0.20

_ELIGIBLE_FOCUS_STATUSES = frozenset({"READY", "TOUCHED", "TOUCHED_UNCONFIRMED", "CONDITIONAL", "FILLED"})
_SUSPENDED_FOCUS_STATUSES = frozenset({"BREAKOUT_PENDING", "SUSPENDED"})
_ACTIVE_FOCUS_STATUSES = _ELIGIBLE_FOCUS_STATUSES | _SUSPENDED_FOCUS_STATUSES
_QUALIFIED_NEXT_STATUSES = frozenset({"READY", "TOUCHED", "TOUCHED_UNCONFIRMED", "CONDITIONAL"})
_TERMINAL_PLAN_STATUSES = frozenset({"INVALID", "SL_BREACHED", "EXPIRED", "MISSED"})


def _get_trade_plan_state_path() -> str:
    try:
        from modules.config import FILE_STATE_ZONE_TRACKER
        return str(FILE_STATE_ZONE_TRACKER or "")
    except Exception:
        return ""


def _load_trade_plan_state_once() -> None:
    """Muat state Trade Plan (frozen, focus, previous, invalidated, switch) sekali dari state_zone_tracker.json."""
    global _TRADE_PLAN_STATE_LOADED
    if _TRADE_PLAN_STATE_LOADED:
        return
    _TRADE_PLAN_STATE_LOADED = True
    path = _get_trade_plan_state_path()
    if not path or not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        tp_st = raw.get("trade_plan_state") if isinstance(raw, dict) else None
        if not isinstance(tp_st, dict):
            return
        for k, v in (tp_st.get("frozen") or {}).items():
            if isinstance(v, dict) and k not in _FROZEN_TRADE_PLANS:
                rec = dict(v)
                sig = rec.get("struct_sig")
                if isinstance(sig, list):
                    rec["struct_sig"] = tuple(sig)
                _FROZEN_TRADE_PLANS[str(k)] = rec
        for k, v in (tp_st.get("focus") or {}).items():
            if isinstance(v, dict) and k not in _ACTIVE_TRADE_PLAN_FOCUS:
                _ACTIVE_TRADE_PLAN_FOCUS[str(k).upper()] = dict(v)
        for k, v in (tp_st.get("previous") or {}).items():
            if isinstance(v, dict) and k not in _PREVIOUS_TRADE_PLANS:
                _PREVIOUS_TRADE_PLANS[str(k).upper()] = dict(v)
        for k, v in (tp_st.get("invalidated") or {}).items():
            if isinstance(v, dict) and k not in _INVALIDATED_SETUP_VERSIONS:
                _INVALIDATED_SETUP_VERSIONS[str(k)] = dict(v)
        for k, v in (tp_st.get("setup_order_links") or {}).items():
            if isinstance(v, list) and k not in _SETUP_ORDER_LINKS:
                _SETUP_ORDER_LINKS[str(k)] = [int(x) for x in v if str(x).isdigit()]
        for sid in (tp_st.get("emitted_switches") or []):
            s_txt = str(sid or "").strip()
            if s_txt and s_txt not in _EMITTED_PLAN_SWITCHES:
                _EMITTED_PLAN_SWITCHES.append(s_txt)
        if len(_EMITTED_PLAN_SWITCHES) > 60:
            del _EMITTED_PLAN_SWITCHES[:-60]
    except Exception:
        pass


def _save_trade_plan_state() -> None:
    """Simpan state Trade Plan secara atomik ke key `trade_plan_state` di state_zone_tracker.json."""
    path = _get_trade_plan_state_path()
    if not path:
        return
    try:
        frozen_out = {}
        for k, v in list(_FROZEN_TRADE_PLANS.items())[-120:]:
            if isinstance(v, dict):
                rec = dict(v)
                if isinstance(rec.get("struct_sig"), tuple):
                    rec["struct_sig"] = list(rec["struct_sig"])
                frozen_out[k] = rec
        payload = {
            "frozen": frozen_out,
            "focus": {k: dict(v) for k, v in _ACTIVE_TRADE_PLAN_FOCUS.items() if isinstance(v, dict)},
            "previous": {k: dict(v) for k, v in _PREVIOUS_TRADE_PLANS.items() if isinstance(v, dict)},
            "invalidated": dict(list(_INVALIDATED_SETUP_VERSIONS.items())[-200:]),
            "setup_order_links": dict(list(_SETUP_ORDER_LINKS.items())[-200:]),
            "emitted_switches": list(_EMITTED_PLAN_SWITCHES[-60:]),
        }
        data: Dict[str, Any] = {}
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    cur = json.load(f)
                if isinstance(cur, dict):
                    data = cur
            except Exception:
                data = {}
        data["trade_plan_state"] = payload
        tmp = f"{path}.plan.{os.getpid()}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=True, default=str)
        os.replace(tmp, path)
    except Exception:
        pass


_SETUP_ORDER_LINKS: Dict[str, List[int]] = {}


def link_mt5_ticket_to_setup(source: str, timeframe: str, zone_id: str, setup_version: int, ticket: Any) -> None:
    """Simpan mapping eksplisit ticket/position_id/order_id MT5 ke setup version tertentu (read-only tracking)."""
    _load_trade_plan_state_once()
    vk = _trade_plan_version_key(source, timeframe, zone_id, setup_version)
    if not vk or ticket is None:
        return
    try:
        t_int = int(ticket)
    except (TypeError, ValueError):
        return
    if t_int <= 0:
        return
    cur = _SETUP_ORDER_LINKS.setdefault(vk, [])
    if t_int not in cur:
        cur.append(t_int)
        _save_trade_plan_state()


def get_linked_mt5_tickets_for_setup(source: str, timeframe: str, zone_id: str, setup_version: int) -> List[int]:
    """Ambil daftar ticket/position_id/order_id MT5 yang terbukti terhubung ke setup version."""
    _load_trade_plan_state_once()
    vk = _trade_plan_version_key(source, timeframe, zone_id, setup_version)
    if not vk:
        return []
    return list(_SETUP_ORDER_LINKS.get(vk) or [])


def _parse_zone_formation_ts_from_id(zone_id: str) -> float:
    m_ep = re.search(r"_(\d{9,11})$", str(zone_id or "").strip())
    if m_ep:
        try:
            return float(int(m_ep.group(1)))
        except ValueError:
            pass
    return 0.0


def verify_mt5_fill_for_plan(
    direction: str,
    lo: float,
    hi: float,
    planned_entry: float,
    mt5_orders_snapshot: Optional[Dict[str, Any]] = None,
    *,
    source: str = "SMC",
    timeframe: str = "H1",
    zone_id: str = "",
    setup_version: int = 1,
    setup_created_ts: float = 0.0,
    zone_formation_ts: float = 0.0,
    linked_tickets: Optional[List[int]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Verifikasi read-only apakah Trade Plan benar-benar FILLED berdasarkan posisi/deal MT5 aktual.
    - HANYA mengembalikan verified=True (status FILLED) bila setup_match == 'CONFIRMED' (ada bukti mapping
      order/deal/position ke setup) DAN posisi/deal benar-benar tereksekusi (OPEN/CLOSED).
    - Posisi lama (EXISTING) atau kecocokan harga/arah/waktu tanpa bukti mapping (INFERRED/UNRELATED/UNKNOWN)
      TIDAK boleh menandai setup baru sebagai FILLED.
    """
    if not isinstance(mt5_orders_snapshot, dict):
        return None
    if not (mt5_orders_snapshot.get("query_ok") or mt5_orders_snapshot.get("positions_query_ok")):
        return None
    d = str(direction or "").strip().upper()
    if d not in ("BUY", "SELL"):
        return None
    positions = mt5_orders_snapshot.get("positions") or []
    deals = mt5_orders_snapshot.get("deals") or []
    if not positions and not deals:
        return None
    tol = float(mt5_orders_snapshot.get("entry_tolerance") or 0.0)
    if tol <= 0.0:
        tol = max(1.0, (hi - lo) * 0.25) if (lo > 0 and hi >= lo) else 1.0

    from modules.mt5_conn import evaluate_mt5_setup_match, format_mt5_wib

    all_linked = list(linked_tickets or [])
    if zone_id:
        all_linked.extend(get_linked_mt5_tickets_for_setup(source, timeframe, zone_id, setup_version))
    z_ts = float(zone_formation_ts or 0.0) or _parse_zone_formation_ts_from_id(zone_id)
    s_ts = float(setup_created_ts or 0.0)

    overlap_existing = None
    for pos in positions:
        if not isinstance(pos, dict):
            continue
        ev = evaluate_mt5_setup_match(
            pos,
            direction=d,
            zone_low=lo,
            zone_high=hi,
            plan_entry=planned_entry,
            zone_id=zone_id,
            setup_version=setup_version,
            setup_created_ts=s_ts,
            zone_formation_ts=z_ts,
            linked_tickets=all_linked,
            entry_tol=tol,
            is_order=False,
            method=source,
        )
        op = float(pos.get("open_price") or pos.get("price_open") or pos.get("entry_price") or 0.0)
        ex_wib = str(
            pos.get("executed_wib_short")
            or pos.get("position_opened_wib_short")
            or format_mt5_wib(pos.get("executed_ts") or pos.get("position_opened_ts") or ev.get("item_ts"), short=True)
            or ""
        )
        if ev["setup_match"] == "CONFIRMED" and ev["mt5_status"] in ("OPEN", "CLOSED"):
            return {
                "verified": True,
                "ticket": pos.get("ticket"),
                "position_id": pos.get("position_id") or pos.get("ticket"),
                "open_price": op,
                "source": "MT5_POSITION",
                "mt5_status": ev["mt5_status"],
                "setup_match": "CONFIRMED",
                "executed_wib": ex_wib,
                "position_opened_ts": float(pos.get("position_opened_ts") or ev.get("item_ts") or 0.0),
            }
        if ev.get("price_matches") and overlap_existing is None:
            overlap_existing = {
                "verified": False,
                "ticket": pos.get("ticket"),
                "position_id": pos.get("position_id") or pos.get("ticket"),
                "direction": d,
                "open_price": op,
                "mt5_status": ev["mt5_status"],
                "setup_match": ev["setup_match"],
                "is_existing_position": bool(ev.get("is_existing_position")),
                "position_opened_wib_short": ex_wib,
                "position_opened_ts": float(pos.get("position_opened_ts") or ev.get("item_ts") or 0.0),
                "reason": ev.get("reason") or "",
            }

    for dl in deals:
        if not isinstance(dl, dict):
            continue
        ev = evaluate_mt5_setup_match(
            dl,
            direction=d,
            zone_low=lo,
            zone_high=hi,
            plan_entry=planned_entry,
            zone_id=zone_id,
            setup_version=setup_version,
            setup_created_ts=s_ts,
            zone_formation_ts=z_ts,
            linked_tickets=all_linked,
            entry_tol=tol,
            is_order=False,
            method=source,
        )
        if ev["setup_match"] == "CONFIRMED":
            dp = float(dl.get("price") or dl.get("open_price") or 0.0)
            ex_wib = str(dl.get("executed_wib_short") or format_mt5_wib(ev.get("item_ts"), short=True) or "")
            return {
                "verified": True,
                "ticket": dl.get("ticket") or dl.get("order") or dl.get("position_id"),
                "position_id": dl.get("position_id") or dl.get("ticket"),
                "open_price": dp,
                "source": "MT5_DEAL",
                "mt5_status": ev["mt5_status"],
                "setup_match": "CONFIRMED",
                "executed_wib": ex_wib,
                "position_opened_ts": float(ev.get("item_ts") or 0.0),
            }
    return overlap_existing


def _trade_plan_cache_key(source: str, timeframe: str, zone_id: str) -> str:
    src = str(source or "SMC").strip().upper()
    tf = str(timeframe or "H1").strip().upper()
    zid = str(zone_id or "").strip()
    if not zid:
        return ""
    return f"{src}:{tf}:{zid}"


def _trade_plan_version_key(source: str, timeframe: str, zone_id: str, setup_version: int) -> str:
    base = _trade_plan_cache_key(source, timeframe, zone_id)
    if not base:
        return ""
    ver = int(setup_version) if (isinstance(setup_version, int) and setup_version > 0) else 1
    return f"{base}:v{ver}"


def _record_invalidated_setup_version(
    source: str,
    timeframe: str,
    zone_id: str,
    setup_version: int,
    *,
    status: str,
    setup_status: str,
    zone_status: str,
    reason: str,
) -> None:
    _load_trade_plan_state_once()
    vk = _trade_plan_version_key(source, timeframe, zone_id, setup_version)
    if not vk or vk in _INVALIDATED_SETUP_VERSIONS:
        return
    _INVALIDATED_SETUP_VERSIONS[vk] = {
        "status": str(status or "INVALID").strip().upper(),
        "setup_status": str(setup_status or "REJECTED").strip().upper(),
        "zone_status": str(zone_status or "STRUCTURALLY_INVALID").strip().upper(),
        "reason": str(reason or "SETUP_INVALID").strip(),
    }
    _save_trade_plan_state()


def _get_invalidated_setup_version(source: str, timeframe: str, zone_id: str, setup_version: int) -> Optional[Dict[str, Any]]:
    _load_trade_plan_state_once()
    vk = _trade_plan_version_key(source, timeframe, zone_id, setup_version)
    if not vk:
        return None
    rec = _INVALIDATED_SETUP_VERSIONS.get(vk)
    return dict(rec) if isinstance(rec, dict) else None


def is_tracked_trade_plan_zone(source: str, zone_id: str) -> bool:
    """True bila (source, zone_id) sedang menjadi fokus aktif pada scope mana pun (SMC/SNR/UNIFIED)."""
    _load_trade_plan_state_once()
    src = str(source or "").strip().upper()
    zid = str(zone_id or "").strip()
    if not src or not zid:
        return False
    for fp in _ACTIVE_TRADE_PLAN_FOCUS.values():
        if isinstance(fp, dict):
            if str(fp.get("source") or "").strip().upper() == src and str(fp.get("zone_id") or "").strip() == zid:
                return True
    return False


def get_frozen_trade_plan(
    source: str,
    zone_id: str,
    timeframe: Optional[str] = None,
    setup_version: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Ambil record trade plan beku untuk (source, zone_id) + setup_version dari shared state."""
    _load_trade_plan_state_once()
    src = str(source or "").strip().upper()
    zid = str(zone_id or "").strip()
    if not src or not zid:
        return None
    req_ver = int(setup_version) if (isinstance(setup_version, int) and setup_version > 0) else None
    tf = str(timeframe or "").strip().upper()
    if tf:
        rec = _FROZEN_TRADE_PLANS.get(_trade_plan_cache_key(src, tf, zid))
        if isinstance(rec, dict):
            if req_ver is None or int(rec.get("setup_version") or 1) == req_ver:
                return dict(rec)
    suffix = f":{zid}"
    prefix = f"{src}:"
    for k, rec in _FROZEN_TRADE_PLANS.items():
        if isinstance(rec, dict) and k.startswith(prefix) and k.endswith(suffix):
            if req_ver is None or int(rec.get("setup_version") or 1) == req_ver:
                return dict(rec)
    return None


def resolve_stored_setup_sl(
    source: str,
    zone_id: str,
    trade_plans: Optional[List[Dict[str, Any]]] = None,
    best_trade_plan: Optional[Dict[str, Any]] = None,
    timeframe: Optional[str] = None,
    setup_version: Optional[int] = None,
) -> float:
    """
    Sumber tunggal SetupSL tersimpan untuk (source, zone_id, setup_version).
    Tidak pernah menghitung ulang dari live tick; membaca dari best_trade_plan -> trade_plans -> frozen state.
    """
    src = str(source or "").strip().upper()
    zid = str(zone_id or "").strip()
    if not src or not zid:
        return 0.0
    req_ver = int(setup_version) if (isinstance(setup_version, int) and setup_version > 0) else None

    def _extract_sl(plan: Any) -> float:
        if not isinstance(plan, dict):
            return 0.0
        p_src = str(plan.get("source") or src).strip().upper()
        p_zid = str(plan.get("zone_id") or "").strip()
        if p_src != src or p_zid != zid:
            return 0.0
        if req_ver is not None and int(plan.get("setup_version") or 1) != req_ver:
            return 0.0
        try:
            v = float(plan.get("setup_sl") or plan.get("sl") or 0.0)
        except (TypeError, ValueError):
            return 0.0
        return round(v, 2) if (math.isfinite(v) and v > 0.0) else 0.0

    sl_best = _extract_sl(best_trade_plan)
    if sl_best > 0.0:
        return sl_best
    for tp in (trade_plans or []):
        sl_tp = _extract_sl(tp)
        if sl_tp > 0.0:
            return sl_tp
    frozen = get_frozen_trade_plan(src, zid, timeframe=timeframe, setup_version=req_ver)
    if isinstance(frozen, dict):
        try:
            v = float(frozen.get("setup_sl") or 0.0)
        except (TypeError, ValueError):
            v = 0.0
        if math.isfinite(v) and v > 0.0:
            return round(v, 2)
    return 0.0


def _compute_rr_pair(direction: str, entry: float, setup_sl: float, tp1: float, tp2: float) -> Tuple[float, float]:
    """Hitung RR1 dan RR2 secara deterministik dari pasangan (entry, setup_sl, tp1, tp2)."""
    d = str(direction or "").strip().upper()
    if entry <= 0.0 or setup_sl <= 0.0 or tp1 <= 0.0:
        return 0.0, 0.0
    order_ok = (setup_sl < entry < tp1) if d == "BUY" else (setup_sl > entry > tp1 > 0.0 if d == "SELL" else False)
    if not order_ok:
        return 0.0, 0.0
    risk = abs(entry - setup_sl)
    if risk <= 1e-9:
        return 0.0, 0.0
    rr1 = round(abs(tp1 - entry) / risk, 2)
    has_tp2 = bool(
        tp2 > 0.0
        and abs(tp2 - tp1) >= 0.01
        and ((tp2 > tp1) if d == "BUY" else (tp2 < tp1))
    )
    rr2 = round(abs(tp2 - entry) / risk, 2) if has_tp2 else 0.0
    return rr1, rr2


def evaluate_trade_plan_lifecycle(
    direction: str,
    lo: float,
    hi: float,
    planned_entry: float,
    setup_sl: float,
    struct_inv: float,
    tp1: float,
    tp2: float,
    live_price: float,
    bid: Optional[float] = None,
    ask: Optional[float] = None,
    digits: int = 2,
    is_closed: bool = False,
    is_invalid_zone: bool = False,
    invalid_reason: str = "",
    is_expired: bool = False,
    is_breakout_pending: bool = False,
    confirmed_entry: float = 0.0,
    early_pending_ready: bool = False,
    cond_reason: str = "WAIT_CONFIRM",
    was_touched: bool = False,
    touch_time: Optional[str] = None,
    m5_eval_status: Optional[str] = None,
    m5_eval_at_touch: Optional[str] = None,
    is_missed: bool = False,
    missed_reason: str = "",
    missed_move_threshold: float = 0.0,
    bars_since_touch: Optional[int] = None,
    max_bars_after_touch: Optional[int] = None,
    verified_mt5_fill: Optional[Dict[str, Any]] = None,
    sl_breached_hist: bool = False,
    htf_conflict: bool = False,
    breakdown_state: Optional[str] = None,
    breakdown_reason: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluasi status lifecycle aktual & LIVE_ENTRY/Market RR tanpa menggeser PLANNED_ENTRY/SL/TP."""
    d = str(direction or "").strip().upper()
    demand = (d == "BUY")
    lp = float(live_price) if (isinstance(live_price, (int, float)) and math.isfinite(live_price) and live_price > 0) else 0.0
    b_val = float(bid) if (isinstance(bid, (int, float)) and math.isfinite(bid) and bid > 0) else 0.0
    a_val = float(ask) if (isinstance(ask, (int, float)) and math.isfinite(ask) and ask > 0) else 0.0
    mkt_entry = (a_val if a_val > 0 else lp) if demand else (b_val if b_val > 0 else lp)
    sl_check_px = (b_val if b_val > 0 else lp) if demand else (a_val if a_val > 0 else lp)

    live_entry = round(confirmed_entry, digits) if (confirmed_entry and confirmed_entry > 0) else (round(mkt_entry, digits) if mkt_entry > 0 else 0.0)
    live_rr1, live_rr2 = _compute_rr_pair(d, live_entry, setup_sl, tp1, tp2)

    inside_zone_now = bool(lo > 0 and hi >= lo and lp > 0 and lo <= lp <= hi)
    touched_now = False
    if lp > 0 and planned_entry > 0:
        if demand:
            touched_now = (lp <= planned_entry) or (mkt_entry > 0 and mkt_entry <= planned_entry)
        else:
            touched_now = (lp >= planned_entry) or (mkt_entry > 0 and mkt_entry >= planned_entry)
    touched_now = bool(touched_now or inside_zone_now)

    exited_far_edge = False
    if lp > 0 and lo > 0 and hi >= lo:
        exited_far_edge = (lp < lo) if demand else (lp > hi)

    eff_was_touched = bool(was_touched or touched_now)
    eff_touch_time = str(touch_time).strip() if touch_time else (format_wib_timestamp(default_now=True) if touched_now else None)
    norm_m5_st = str(m5_eval_status or "").strip().upper() or None
    eff_m5_at_touch = (
        str(m5_eval_at_touch).strip().upper()
        if m5_eval_at_touch
        else (norm_m5_st if eff_was_touched and norm_m5_st else None)
    )

    def _base(
        status: str,
        zone_status: str,
        setup_status: str,
        reason: str,
        transition_reason: str,
        entry_method: str = "PENDING_LIMIT",
        is_executable: bool = False,
        is_suspended: bool = False,
        touched_flag: Optional[bool] = None,
        fill_ticket: Any = None,
        live_e_override: Optional[float] = None,
    ) -> Dict[str, Any]:
        le_v = round(float(live_e_override), digits) if (live_e_override and live_e_override > 0) else live_entry
        lrr1, lrr2 = _compute_rr_pair(d, le_v, setup_sl, tp1, tp2) if live_e_override else (live_rr1, live_rr2)
        return {
            "status": status,
            "zone_status": zone_status,
            "setup_status": setup_status,
            "reason": reason,
            "transition_reason": transition_reason,
            "entry_method": entry_method,
            "is_executable": is_executable,
            "is_suspended": is_suspended,
            "live_price": round(lp, digits) if lp > 0 else 0.0,
            "live_entry": le_v,
            "live_rr1": lrr1,
            "live_rr2": lrr2,
            "was_touched": eff_was_touched if touched_flag is None else bool(touched_flag),
            "touch_time": eff_touch_time,
            "m5_eval_status": norm_m5_st,
            "m5_eval_at_touch": eff_m5_at_touch,
            "mt5_fill_ticket": fill_ticket,
        }

    # 0. FILLED — HANYA dari verifikasi order/deal/posisi MT5 aktual dengan setup_match == CONFIRMED
    if isinstance(verified_mt5_fill, dict) and verified_mt5_fill.get("verified"):
        tkt = verified_mt5_fill.get("ticket")
        fill_px = float(verified_mt5_fill.get("open_price") or live_entry or planned_entry)
        exec_wib = str(verified_mt5_fill.get("executed_wib") or "").strip()
        rsn_fill = f"MT5_FILLED#{tkt} {exec_wib}".strip() if tkt else (f"MT5_FILLED {exec_wib}".strip() if exec_wib else "MT5_FILLED")
        res_fill = _base(
            status="FILLED",
            zone_status="TOUCHED" if inside_zone_now else "ACTIVE",
            setup_status="FILLED",
            reason=rsn_fill,
            transition_reason=f"CONFIRMED_{verified_mt5_fill.get('source', 'MT5_ORDER')}",
            entry_method="CONFIRMED",
            is_executable=False,
            is_suspended=False,
            touched_flag=True,
            fill_ticket=tkt,
            live_e_override=fill_px,
        )
        res_fill["mt5_fill_time_wib"] = exec_wib
        res_fill["mt5_status"] = str(verified_mt5_fill.get("mt5_status") or "OPEN").upper()
        res_fill["setup_match"] = "CONFIRMED"
        return res_fill

    # 1. Evaluasi breach SetupSL (live & histori tersedia) dan structural invalidation SEBELUM filter lain
    live_sl_hit = bool(
        sl_check_px > 0
        and setup_sl > 0
        and ((demand and sl_check_px <= setup_sl) or ((not demand) and sl_check_px >= setup_sl))
    )
    sl_breached = bool(live_sl_hit or sl_breached_hist)
    if is_invalid_zone and sl_breached:
        inv_r = invalid_reason or "ZONE_STRUCTURALLY_INVALID"
        comb_r = inv_r if "SL_BREACHED" in inv_r.upper() else f"SL_BREACHED+{inv_r}"
        return _base("INVALID", "STRUCTURALLY_INVALID", "SL_BREACHED", comb_r, "SETUP_SL_BREACHED_AND_ZONE_INVALID")

    if sl_breached:
        sl_r = "SL_BREACHED" if live_sl_hit else "SL_BREACHED_HIST"
        z_st_sl = "BREAKOUT_PENDING" if (is_breakout_pending or exited_far_edge) else "ACTIVE"
        return _base("SL_BREACHED", z_st_sl, "SL_BREACHED", sl_r, "SETUP_SL_BREACHED", is_suspended=False)

    if is_invalid_zone:
        inv_r = invalid_reason or "ZONE_STRUCTURALLY_INVALID"
        return _base("INVALID", "STRUCTURALLY_INVALID", "INVALID", inv_r, inv_r)

    brk_st_u = str(breakdown_state or "").strip().upper()
    if brk_st_u in ("STRUCTURAL_INVALIDATION", "STRUCTURALLY_INVALID"):
        brk_r = str(breakdown_reason or "STRUCTURAL_INVALIDATION (CLOSED_CANDLE)")
        return _base("INVALID", "STRUCTURALLY_INVALID", "INVALID", brk_r, "STRUCTURAL_INVALIDATION_CLOSED_CANDLE")
    if brk_st_u in ("CONFIRMED_BREAKDOWN", "CONFIRMED_BREAKOUT", "BREAKDOWN_CONFIRMED", "BREAKOUT_CONFIRMED", "BROKEN"):
        brk_r = str(breakdown_reason or f"{brk_st_u} (CLOSED_CANDLE)")
        return _base("INVALID", "ZONE_BROKEN", "INVALID", brk_r, f"{brk_st_u}_CLOSED_CANDLE")

    # 2. EXPIRED (aturan validitas existing habis; zona tetap ACTIVE bila tidak invalid)
    if is_expired:
        return _base("EXPIRED", "TOUCHED" if inside_zone_now else "ACTIVE", "EXPIRED", "SETUP_EXPIRED", "VALIDITY_WINDOW_EXPIRED")

    blocked_by_htf_or_m5 = bool(
        htf_conflict
        or (norm_m5_st in ("UNCONFIRMED", "CONFLICT", "CONFIRMED_BLOCKED_HTF"))
    )

    # 3. CONFIRMED (market/confirmed entry tervalidasi -> READY executable; diblokir oleh HTF_CONFLICT / M5 UNCONFIRMED)
    if (
        (not is_closed)
        and (not blocked_by_htf_or_m5)
        and confirmed_entry
        and confirmed_entry > 0
        and not is_breakout_pending
        and not exited_far_edge
        and brk_st_u not in ("POTENTIAL_BREAKDOWN", "POTENTIAL_BREAKOUT")
    ):
        return _base(
            status="READY",
            zone_status="TOUCHED" if inside_zone_now else "ACTIVE",
            setup_status="READY",
            reason="CONFIRMED",
            transition_reason="M5_ENTRY_CONFIRMED",
            entry_method="CONFIRMED",
            is_executable=True,
            touched_flag=True,
        )

    # 4. SUSPENDED (potential breakdown/breakout belum closed-candle confirm & belum tembus SetupSL)
    if brk_st_u in ("POTENTIAL_BREAKDOWN", "POTENTIAL_BREAKOUT"):
        brk_z_st = "POTENTIAL_BREAKDOWN" if "BREAKDOWN" in brk_st_u else "POTENTIAL_BREAKOUT"
        brk_r = str(breakdown_reason or f"{brk_z_st} (AWAIT_CLOSED_CANDLE)")
        return _base(
            status="SUSPENDED",
            zone_status=brk_z_st,
            setup_status="SUSPENDED",
            reason=brk_r,
            transition_reason=f"{brk_z_st}_AWAIT_CLOSED_CANDLE",
            is_suspended=True,
        )

    if is_breakout_pending or exited_far_edge:
        return _base(
            status="SUSPENDED",
            zone_status="BREAKOUT_PENDING",
            setup_status="SUSPENDED",
            reason="SUSPENDED_UNCONFIRMED_BO",
            transition_reason="EXITED_FAR_EDGE_UNCONFIRMED_BO",
            is_suspended=True,
        )

    # 5. MISSED — harga mencapai TP1 ATAU bergerak menjauh melampaui jendela validitas existing tanpa konfirmasi/MT5 fill
    if lp > 0 and tp1 > 0:
        if (demand and lp >= tp1) or ((not demand) and lp <= tp1):
            return _base(
                status="MISSED",
                zone_status="TOUCHED" if inside_zone_now else "ACTIVE",
                setup_status="MISSED",
                reason="TP1_REACHED_UNCONFIRMED",
                transition_reason="TP1_REACHED_WITHOUT_CONFIRM_OR_MT5_FILL",
            )

    dist_tp_dir = 0.0
    if lp > 0 and lo > 0 and hi >= lo:
        dist_tp_dir = max(0.0, lp - hi) if demand else max(0.0, lo - lp)
    moved_away_outside_window = bool(
        eff_was_touched
        and (not inside_zone_now)
        and dist_tp_dir > 0.0
        and (
            (missed_move_threshold > 0.0 and dist_tp_dir > missed_move_threshold)
            or (
                max_bars_after_touch is not None
                and bars_since_touch is not None
                and bars_since_touch > max_bars_after_touch
            )
        )
    )
    if is_missed or moved_away_outside_window:
        m_rsn = missed_reason or ("NO_CONFIRM_MOVED_AWAY" if moved_away_outside_window else "ENTRY_MISSED")
        return _base(
            status="MISSED",
            zone_status="ACTIVE",
            setup_status="MISSED",
            reason=m_rsn,
            transition_reason=f"TOUCHED_NO_CONFIRM_MOVED_AWAY (M5:{eff_m5_at_touch or 'UNCONFIRMED'})",
            touched_flag=True,
        )

    # 6. TOUCHED_UNCONFIRMED (harga menyentuh/pernah menyentuh zona tanpa konfirmasi entry & masih dalam jendela validitas)
    if eff_was_touched:
        return _base(
            status="TOUCHED_UNCONFIRMED",
            zone_status="TOUCHED" if inside_zone_now else "ACTIVE",
            setup_status="TOUCHED_UNCONFIRMED",
            reason=cond_reason or "WAIT_M5_CONFIRM",
            transition_reason=f"TOUCHED_NO_CONFIRM (M5:{eff_m5_at_touch or 'UNCONFIRMED'})",
            touched_flag=True,
        )

    # 7. PENDING_LIMIT READY (diblokir oleh HTF_CONFLICT dan M5 UNCONFIRMED; tetap tampil sebagai CONDITIONAL)
    if (not is_closed) and (not blocked_by_htf_or_m5) and early_pending_ready:
        return _base(
            status="READY",
            zone_status="ACTIVE",
            setup_status="READY",
            reason="PENDING_READY",
            transition_reason="PENDING_LIMIT_READY",
            is_executable=True,
            touched_flag=False,
        )

    eff_cond_reason = cond_reason or "WAIT_CONFIRM"
    if htf_conflict and "HTF_CONFLICT" not in eff_cond_reason.upper():
        eff_cond_reason = "HTF_CONFLICT_CONDITIONAL"
    elif norm_m5_st == "UNCONFIRMED" and "UNCONFIRMED" not in eff_cond_reason.upper():
        eff_cond_reason = "M5_UNCONFIRMED_CONDITIONAL"

    # 8. CONDITIONAL (pending limit valid secara numerik, belum disentuh, menunggu kondisi/konfirmasi)
    return _base(
        status="CONDITIONAL",
        zone_status="ACTIVE",
        setup_status="CONDITIONAL",
        reason=eff_cond_reason,
        transition_reason="AWAITING_TOUCH_AND_CONFIRM",
        touched_flag=False,
    )


def _evaluate_plan_revalidation_from_ctx(
    plan: Optional[Dict[str, Any]],
    revalidation_ctx: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Revalidasi locked plan terhadap status breakdown/invalidasi closed candle di SNR/SMC."""
    out = {"breakdown_state": None, "breakdown_reason": None, "is_invalid_zone": False, "invalid_reason": ""}
    if not isinstance(plan, dict) or not isinstance(revalidation_ctx, dict):
        return out
    zid = str(plan.get("zone_id") or "").strip()
    p_dir = str(plan.get("direction") or "").strip().upper()
    if not zid:
        return out

    pool: List[Dict[str, Any]] = []
    snr_zones = revalidation_ctx.get("snr_zones") if isinstance(revalidation_ctx.get("snr_zones"), dict) else {}
    for k in ("all_zones", "active_supports", "active_resistances", "broken_zones", "invalidated_zones"):
        for z in (snr_zones.get(k) or []):
            if isinstance(z, dict):
                pool.append(z)
    for z in (revalidation_ctx.get("scan_zones") or []):
        if isinstance(z, dict):
            pool.append(z)

    matched_z = next((z for z in pool if str(z.get("zone_id") or "").strip() == zid), None)
    if isinstance(matched_z, dict):
        life = str(matched_z.get("lifecycle_status") or matched_z.get("status") or "").strip().upper()
        zt = str(matched_z.get("zone_type") or "").strip().upper()
        is_sup = zt in ("SUPPORT", "DEMAND") or (not zt and p_dir == "BUY")
        if matched_z.get("is_invalidated") or matched_z.get("closed_invalidated") or life == "INVALIDATED":
            out["breakdown_state"] = "STRUCTURAL_INVALIDATION"
            out["breakdown_reason"] = "STRUCTURAL_INVALIDATION (CLOSED_CANDLE)"
            out["is_invalid_zone"] = True
            out["invalid_reason"] = out["breakdown_reason"]
            return out
        if matched_z.get("is_broken") or life in ("BROKEN", "BREAKDOWN_CONFIRMED", "BREAKOUT_CONFIRMED"):
            brk_lbl = "CONFIRMED_BREAKDOWN" if is_sup else "CONFIRMED_BREAKOUT"
            out["breakdown_state"] = brk_lbl
            out["breakdown_reason"] = f"{brk_lbl} (CLOSED_CANDLE)"
            return out
        if matched_z.get("unconfirmed_break") or life in ("BREAKDOWN_PENDING", "BREAKOUT_PENDING"):
            pend_lbl = "POTENTIAL_BREAKDOWN" if is_sup else "POTENTIAL_BREAKOUT"
            out["breakdown_state"] = pend_lbl
            out["breakdown_reason"] = f"{pend_lbl} (AWAIT_CLOSED_CANDLE)"
            return out

    # Cek price_action SNR untuk breakdown/breakout yang baru terjadi pada zona dengan ID sama
    snr_pa = revalidation_ctx.get("snr_price_action") if isinstance(revalidation_ctx.get("snr_price_action"), dict) else {}
    bo_zid = str(snr_pa.get("breakout_zone_id") or "").strip()
    if bo_zid and bo_zid == zid:
        bo_st = str(snr_pa.get("breakout_state") or "").strip().upper()
        bo_dir = str(snr_pa.get("breakout_dir") or "").strip().upper()
        if bo_st == "VALID_BREAKOUT":
            brk_lbl = "CONFIRMED_BREAKDOWN" if (bo_dir == "BEARISH" or p_dir == "BUY") else "CONFIRMED_BREAKOUT"
            out["breakdown_state"] = brk_lbl
            out["breakdown_reason"] = f"{brk_lbl} (CLOSED_CANDLE)"
            return out
        if snr_pa.get("potential_breakdown") or bo_st in ("POTENTIAL_BREAKOUT", "POTENTIAL_BREAKDOWN", "BREAKDOWN_PENDING", "BREAKOUT_PENDING"):
            pend_lbl = "POTENTIAL_BREAKDOWN" if (bo_dir == "BEARISH" or p_dir == "BUY") else "POTENTIAL_BREAKOUT"
            out["breakdown_state"] = pend_lbl
            out["breakdown_reason"] = f"{pend_lbl} (AWAIT_CLOSED_CANDLE)"
            return out
    return out


def refresh_trade_plan_live_state(
    plan: Optional[Dict[str, Any]],
    live_price: float,
    bid: Optional[float] = None,
    ask: Optional[float] = None,
    is_closed: bool = False,
    m5_eval_status: Optional[str] = None,
    mt5_orders_snapshot: Optional[Dict[str, Any]] = None,
    htf_conflict: bool = False,
    revalidation_ctx: Optional[Dict[str, Any]] = None,
    h4_bias: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Perbarui LIVE PRICE, LIVE ENTRY, Market RR, dan status lifecycle pada live tick tanpa menggeser level plan beku."""
    if not isinstance(plan, dict) or not plan:
        return plan
    out = dict(plan)
    digits = int(out.get("digits") or 2)
    lp = float(live_price) if (isinstance(live_price, (int, float)) and math.isfinite(live_price) and live_price > 0) else float(out.get("live_price") or 0.0)
    if lp > 0:
        out["live_price"] = round(lp, digits)
    st_cur = str(out.get("status") or "").strip().upper()
    if st_cur in ("REJECTED", "NO_ACTIVE_PLAN"):
        return out
    pe = float(out.get("planned_entry") or out.get("entry") or 0.0)
    sl = float(out.get("setup_sl") or out.get("sl") or 0.0)
    tp1 = float(out.get("tp1") or 0.0)
    tp2 = float(out.get("tp2") or 0.0)
    if pe <= 0.0 or sl <= 0.0 or tp1 <= 0.0 or lp <= 0.0:
        return out

    src = str(out.get("source") or "SMC").strip().upper()
    tf = str(out.get("timeframe") or "H1").strip().upper()
    zid = str(out.get("zone_id") or "").strip()
    cur_ver = int(out.get("setup_version") or 1)
    # Setup terminal tidak boleh aktif kembali pada ID + version yang sama
    inv_rec = _get_invalidated_setup_version(src, tf, zid, cur_ver)
    if inv_rec is not None:
        out["status"] = inv_rec["status"]
        out["setup_status"] = inv_rec["setup_status"]
        out["zone_status"] = inv_rec["zone_status"]
        out["reason"] = inv_rec["reason"]
        out["is_executable"] = False
        out["is_suspended"] = (inv_rec["zone_status"] in ("SUSPENDED", "BREAKOUT_PENDING"))
        out["is_breakout_pending"] = (inv_rec["zone_status"] == "BREAKOUT_PENDING")
        out["entry"] = pe
        return out

    cache_k = _trade_plan_cache_key(src, tf, zid)
    cached = _FROZEN_TRADE_PLANS.get(cache_k) if cache_k else None
    was_touched = bool(out.get("was_touched") or (cached.get("was_touched") if isinstance(cached, dict) else False))
    touch_time = out.get("touch_time") or (cached.get("touch_time") if isinstance(cached, dict) else None)
    eff_m5_st = m5_eval_status or out.get("m5_eval_status")
    m5_at_touch = out.get("m5_eval_at_touch") or (cached.get("m5_eval_at_touch") if isinstance(cached, dict) else None)
    is_confirmed = (st_cur == "READY" and str(out.get("entry_method") or "").upper() == "CONFIRMED")
    is_pend_ready = (st_cur == "READY" and str(out.get("entry_method") or "").upper() == "PENDING_LIMIT")
    conf_e = float(out.get("live_entry") or 0.0) if is_confirmed else 0.0
    lo = float(out.get("zone_low") or 0.0)
    hi = float(out.get("zone_high") or 0.0)
    s_inv = float(out.get("struct_inv") or 0.0)
    cond_rsn = str(out.get("cond_reason") or out.get("reason") or "WAIT_CONFIRM")
    d_up = str(out.get("direction") or "").strip().upper()
    still_outside = bool(lo > 0 and hi >= lo and ((lp < lo) if d_up == "BUY" else (lp > hi)))
    bo_pend = bool(still_outside or (out.get("is_breakout_pending") and still_outside))
    setup_cr_ts = float(out.get("created_at_ts") or (cached.get("created_at_ts") if isinstance(cached, dict) else 0.0) or 0.0)
    zone_fm_ts = float(out.get("zone_formation_ts") or (cached.get("zone_formation_ts") if isinstance(cached, dict) else 0.0) or 0.0)
    mt5_fill = verify_mt5_fill_for_plan(
        d_up, lo, hi, pe, mt5_orders_snapshot,
        source=src, timeframe=tf, zone_id=zid, setup_version=cur_ver,
        setup_created_ts=setup_cr_ts, zone_formation_ts=zone_fm_ts,
    )
    if (mt5_fill is None or not mt5_fill.get("verified")) and st_cur == "FILLED" and out.get("mt5_fill_ticket") and str(out.get("setup_match") or "").upper() == "CONFIRMED":
        mt5_fill = {
            "verified": True,
            "ticket": out.get("mt5_fill_ticket"),
            "open_price": out.get("live_entry") or pe,
            "source": "MT5_POSITION",
            "mt5_status": out.get("mt5_status") or "OPEN",
            "setup_match": "CONFIRMED",
            "executed_wib": out.get("mt5_fill_time_wib") or "",
        }

    reval = _evaluate_plan_revalidation_from_ctx(out, revalidation_ctx)
    eff_inv_zone = bool(st_cur == "INVALID" or str(out.get("zone_status") or "").upper() == "STRUCTURALLY_INVALID" or reval.get("is_invalid_zone"))
    eff_inv_rsn = str(reval.get("invalid_reason") or (out.get("reason") or "") if eff_inv_zone else "")
    lc = evaluate_trade_plan_lifecycle(
        direction=d_up,
        lo=lo,
        hi=hi,
        planned_entry=pe,
        setup_sl=sl,
        struct_inv=s_inv,
        tp1=tp1,
        tp2=tp2,
        live_price=lp,
        bid=bid,
        ask=ask,
        digits=digits,
        is_closed=is_closed,
        is_invalid_zone=eff_inv_zone,
        invalid_reason=eff_inv_rsn,
        is_expired=(st_cur == "EXPIRED" or str(out.get("setup_status") or "").upper() == "EXPIRED"),
        is_breakout_pending=bo_pend,
        confirmed_entry=conf_e,
        early_pending_ready=is_pend_ready,
        cond_reason=cond_rsn,
        was_touched=was_touched,
        touch_time=touch_time,
        m5_eval_status=eff_m5_st,
        m5_eval_at_touch=m5_at_touch,
        missed_move_threshold=float(out.get("missed_move_threshold") or (cached.get("missed_move_threshold") if isinstance(cached, dict) else 0.0) or 0.0),
        verified_mt5_fill=mt5_fill,
        htf_conflict=htf_conflict,
        breakdown_state=reval.get("breakdown_state"),
        breakdown_reason=reval.get("breakdown_reason"),
    )
    eff_h4 = str(h4_bias or out.get("h4_bias") or "").strip().upper()
    if eff_h4 in ("BULLISH", "BEARISH"):
        out["h4_bias"] = eff_h4
        out["is_counter_trend"] = bool((d_up == "BUY" and eff_h4 == "BEARISH") or (d_up == "SELL" and eff_h4 == "BULLISH"))
    if lc["status"] in ("INVALID", "SL_BREACHED", "EXPIRED", "MISSED") or lc["setup_status"] in ("SL_BREACHED", "EXPIRED", "MISSED") or lc["zone_status"] == "STRUCTURALLY_INVALID":
        _record_invalidated_setup_version(
            src, tf, zid, cur_ver,
            status=lc["status"],
            setup_status=lc["setup_status"],
            zone_status=lc["zone_status"],
            reason=lc["reason"],
        )
    if isinstance(cached, dict) and lc.get("was_touched"):
        cached["was_touched"] = True
        if lc.get("touch_time") and not cached.get("touch_time"):
            cached["touch_time"] = lc["touch_time"]
        if lc.get("m5_eval_at_touch") and not cached.get("m5_eval_at_touch"):
            cached["m5_eval_at_touch"] = lc["m5_eval_at_touch"]
    out["live_price"] = lc["live_price"]
    out["live_entry"] = lc["live_entry"]
    out["live_rr1"] = lc["live_rr1"]
    out["live_rr2"] = lc["live_rr2"]
    out["was_touched"] = lc["was_touched"]
    out["touch_time"] = lc["touch_time"]
    out["m5_eval_status"] = lc["m5_eval_status"]
    out["m5_eval_at_touch"] = lc["m5_eval_at_touch"]
    out["transition_reason"] = lc["transition_reason"]
    out["mt5_fill_ticket"] = lc["mt5_fill_ticket"]
    out["mt5_fill_time_wib"] = lc.get("mt5_fill_time_wib") or out.get("mt5_fill_time_wib") or ""
    if isinstance(mt5_fill, dict) and not mt5_fill.get("verified"):
        out["existing_position_info"] = dict(mt5_fill)
        out["mt5_status"] = mt5_fill.get("mt5_status") or "OPEN"
        out["setup_match"] = mt5_fill.get("setup_match") or "UNKNOWN"
    elif lc.get("setup_match"):
        out["mt5_status"] = lc.get("mt5_status") or "OPEN"
        out["setup_match"] = lc.get("setup_match")
        out.pop("existing_position_info", None)
    else:
        out.pop("existing_position_info", None)
    out["status"] = lc["status"]
    out["zone_status"] = lc["zone_status"]
    out["setup_status"] = lc["setup_status"]
    out["reason"] = lc["reason"]
    out["entry_method"] = lc["entry_method"]
    out["is_executable"] = lc["is_executable"]
    out["is_suspended"] = lc["is_suspended"]
    out["is_breakout_pending"] = (lc["zone_status"] == "BREAKOUT_PENDING")
    out["entry"] = lc["live_entry"] if lc["entry_method"] == "CONFIRMED" and lc["live_entry"] > 0 else pe
    return out


def build_shared_trade_plan(
    *,
    source: str,
    zone_id: str,
    timeframe: str,
    zone_type: str,
    direction: str,
    grade: str,
    score: float,
    lo: float,
    hi: float,
    struct_inv: float,
    live_price: float,
    why: str = "",
    bid: Optional[float] = None,
    ask: Optional[float] = None,
    spread: float = 0.0,
    max_spread: float = 1.0,
    digits: int = 2,
    min_stop: float = 0.0,
    min_rr: float = 1.5,
    max_sl_dist: float = 20.0,
    atr_value: float = 0.0,
    is_closed: bool = False,
    data_unavailable: bool = False,
    is_invalid_zone: bool = False,
    invalid_reason: str = "",
    is_expired_zone: bool = False,
    is_breakout_pending: bool = False,
    is_far_zone: bool = False,
    confirmed_plan: Optional[Dict[str, Any]] = None,
    early_pending_plan: Optional[Dict[str, Any]] = None,
    cond_reason: str = "WAIT_CONFIRM",
    compute_levels_fn=None,
    setup_version: Optional[int] = None,
    force_recalc: bool = False,
    was_touched: bool = False,
    touch_time: Optional[str] = None,
    m5_eval_status: Optional[str] = None,
    is_missed: bool = False,
    missed_reason: str = "",
    missed_move_threshold: float = 0.0,
    bars_since_touch: Optional[int] = None,
    max_bars_after_touch: Optional[int] = None,
    mt5_orders_snapshot: Optional[Dict[str, Any]] = None,
    sl_breached_hist: bool = False,
) -> Dict[str, Any]:
    """
    Trade Plan Generator bersama untuk SMC dan SNR.
    - Memisahkan PLANNED_ENTRY (pending limit deterministik dari proximal edge zona) vs LIVE_ENTRY (market/confirmed).
    - Membekukan PLANNED_ENTRY, SetupSL, TP1, TP2, RR selama Zone ID & setup_version sama (tidak bergeser oleh live tick).
    - Merevisi hanya bila struktur berubah, ATR memicu recalculation (>=20%), atau setup_version baru.
    - Mengubah status secara jujur ke TOUCHED / BREAKOUT_PENDING / MISSED / EXPIRED / INVALID saat kondisi aktual berubah.
    """
    _load_trade_plan_state_once()
    src = str(source or "SMC").strip().upper()
    zid = str(zone_id or "").strip()
    tf = str(timeframe or "H1").strip().upper()
    zt = str(zone_type or "").strip().upper()
    d = str(direction or "NONE").strip().upper()
    demand = (d == "BUY")
    lp = float(live_price) if (isinstance(live_price, (int, float)) and math.isfinite(live_price) and live_price > 0) else 0.0
    dist_val = 0.0 if (lp > 0 and lo > 0 and lo <= lp <= hi) else (min(abs(lp - lo), abs(lp - hi)) if (lp > 0 and lo > 0) else 9999.0)
    cache_k = _trade_plan_cache_key(src, tf, zid)
    cached = _FROZEN_TRADE_PLANS.get(cache_k) if cache_k else None
    cur_ver = int(setup_version) if (isinstance(setup_version, int) and setup_version > 0) else (int(cached.get("setup_version", 1)) if isinstance(cached, dict) else 1)

    def _rej(reason_code: str, status_override: str = "REJECTED", s_inv_v: float = 0.0) -> Dict[str, Any]:
        if is_invalid_zone or status_override == "INVALID":
            z_st_r = "STRUCTURALLY_INVALID"
        elif reason_code == "FAR_ZONE":
            z_st_r = "FAR_REFERENCE"
        elif is_breakout_pending:
            z_st_r = "BREAKOUT_PENDING"
        else:
            z_st_r = "ACTIVE"
        if status_override == "EXPIRED":
            s_st_r = "EXPIRED"
        elif status_override == "SL_BREACHED":
            s_st_r = "SL_BREACHED"
        elif status_override == "INVALID":
            s_st_r = "INVALID"
        elif reason_code == "FAR_ZONE":
            s_st_r = "FAR_ZONE"
        else:
            s_st_r = "REJECTED"
        return {
            "source": src,
            "zone_id": zid,
            "setup_version": cur_ver,
            "timeframe": tf,
            "zone_type": zt,
            "grade": str(grade or "N/A").strip(),
            "score": float(score or 0.0),
            "distance": round(dist_val, 2),
            "zone_low": round(lo, digits) if lo > 0 else 0.0,
            "zone_high": round(hi, digits) if hi > 0 else 0.0,
            "digits": digits,
            "direction": d,
            "entry_method": "NONE",
            "entry": 0.0,
            "planned_entry": 0.0,
            "live_entry": 0.0,
            "live_price": round(lp, digits) if lp > 0 else 0.0,
            "struct_inv": round(s_inv_v, digits) if s_inv_v > 0 else (round(struct_inv, digits) if struct_inv > 0 else 0.0),
            "setup_sl": 0.0,
            "sl": 0.0,
            "tp1": 0.0,
            "tp2": 0.0,
            "rr1": 0.0,
            "rr2": 0.0,
            "planned_rr1": 0.0,
            "planned_rr2": 0.0,
            "live_rr1": 0.0,
            "live_rr2": 0.0,
            "status": status_override,
            "zone_status": z_st_r,
            "setup_status": s_st_r,
            "reason": reason_code,
            "cond_reason": cond_reason,
            "why": str(why or "").strip(),
            "is_executable": False,
            "is_suspended": False,
            "is_breakout_pending": (z_st_r == "BREAKOUT_PENDING"),
            "was_touched": False,
        }

    if data_unavailable:
        return _rej("DATA_UNAVAILABLE")
    if d not in ("BUY", "SELL") or lo <= 0.0 or hi < lo or lp <= 0.0:
        return _rej("DATA_INCOMPLETE")

    s_inv = float(struct_inv) if (isinstance(struct_inv, (int, float)) and struct_inv > 0) else (lo if demand else hi)

    # Prioritas 1: Cek lock invalidasi permanen & breach/structural invalidation SEBELUM distance filter (FAR_ZONE)
    if not isinstance(cached, dict):
        inv_pre = _get_invalidated_setup_version(src, tf, zid, cur_ver)
        if inv_pre is not None:
            return _rej(inv_pre["reason"], status_override=inv_pre["status"], s_inv_v=s_inv)
        if is_invalid_zone or is_expired_zone:
            if is_expired_zone:
                _record_invalidated_setup_version(src, tf, zid, cur_ver, status="EXPIRED", setup_status="EXPIRED", zone_status="ACTIVE", reason="SETUP_EXPIRED")
                return _rej("SETUP_EXPIRED", status_override="EXPIRED", s_inv_v=s_inv)
            inv_code = invalid_reason or "ZONE_INVALID"
            _record_invalidated_setup_version(src, tf, zid, cur_ver, status="INVALID", setup_status="INVALID", zone_status="STRUCTURALLY_INVALID", reason=inv_code)
            return _rej(inv_code, status_override="INVALID", s_inv_v=s_inv)
        # FAR_ZONE hanya filter jarak sementara (tidak memicu invalidasi permanen)
        if is_far_zone:
            return _rej("FAR_ZONE", s_inv_v=s_inv)
        if not is_closed and max_spread > 0.0 and spread > max_spread:
            return _rej(f"SPREAD_HIGH ({spread:.2f}>{max_spread:g})", s_inv_v=s_inv)

    # PLANNED_ENTRY deterministik dari proximal edge zona (bukan midpoint, bukan live tick)
    det_planned_entry = round(hi, digits) if demand else round(lo, digits)
    if isinstance(early_pending_plan, dict):
        ep_e = float(early_pending_plan.get("entry") or early_pending_plan.get("entry_price") or 0.0)
        if ep_e > 0 and lo - 1e-6 <= ep_e <= hi + 1e-6:
            det_planned_entry = round(ep_e, digits)

    struct_sig = (d, round(lo, digits), round(hi, digits), round(s_inv, digits))
    atr_now = float(atr_value) if (isinstance(atr_value, (int, float)) and math.isfinite(atr_value) and atr_value > 0) else 0.0

    # Cek apakah plan beku masih berlaku (Zone ID & struktur & ATR & setup_version sama)
    use_cached = False
    revision_reason = ""
    revised_at = ""
    if isinstance(cached, dict) and not force_recalc:
        same_struct = (cached.get("struct_sig") == struct_sig)
        cached_ver = int(cached.get("setup_version", 1))
        same_ver = (setup_version is None or int(setup_version) == cached_ver)
        cached_invalid = (_get_invalidated_setup_version(src, tf, zid, cached_ver) is not None)
        c_sl_val = float(cached.get("setup_sl") or 0.0)
        c_sl_px = (float(bid) if (isinstance(bid, (int, float)) and bid > 0) else lp) if demand else (float(ask) if (isinstance(ask, (int, float)) and ask > 0) else lp)
        cached_sl_hit = bool(
            sl_breached_hist
            or (c_sl_val > 0.0 and c_sl_px > 0.0 and ((demand and c_sl_px <= c_sl_val) or ((not demand) and c_sl_px >= c_sl_val)))
        )
        atr_ref = float(cached.get("atr_ref") or 0.0)
        atr_shift = bool(
            (not cached_invalid)
            and (not cached_sl_hit)
            and (not is_invalid_zone)
            and (not is_expired_zone)
            and (not is_far_zone)
            and atr_ref > 0.0 and atr_now > 0.0
            and abs(atr_now - atr_ref) / atr_ref >= _TRADE_PLAN_ATR_RECALC_RATIO
        )
        if same_struct and same_ver and not atr_shift:
            use_cached = True
        elif (cached_invalid or cached_sl_hit or is_invalid_zone or is_expired_zone or is_far_zone) and (setup_version is None or int(setup_version) <= cached_ver):
            # Setup terminal/breached/far pada ID + version yang sama tidak boleh auto-bump versi / aktif kembali
            use_cached = True
        elif not is_invalid_zone and not is_expired_zone:
            cur_ver = int(setup_version) if (isinstance(setup_version, int) and setup_version > cached_ver) else (cached_ver + 1)
            if not same_struct:
                revision_reason = "STRUCTURE_CHANGED"
            elif atr_shift:
                revision_reason = "ATR_SHIFT"
            else:
                revision_reason = "EXPLICIT_VERSION"
            revised_at = format_wib_timestamp(default_now=True)
    elif isinstance(cached, dict) and force_recalc:
        cached_ver = int(cached.get("setup_version", 1))
        cur_ver = int(setup_version) if (isinstance(setup_version, int) and setup_version > cached_ver) else (cached_ver + 1)
        revision_reason = "FORCE_RECALC"
        revised_at = format_wib_timestamp(default_now=True)

    if use_cached:
        planned_entry = float(cached["planned_entry"])
        setup_sl = float(cached["setup_sl"])
        s_inv_final = float(cached["struct_inv"])
        tp1 = float(cached["tp1"])
        tp2 = float(cached["tp2"])
        rr1 = float(cached["rr1"])
        rr2 = float(cached["rr2"])
        cur_ver = int(cached["setup_version"])
        revision_reason = str(cached.get("revision_reason") or "")
        revised_at = str(cached.get("revised_at") or "")
        plan_created_ts = float(cached.get("created_at_ts") or 0.0)
        plan_created_at = str(cached.get("created_at") or "")
        eff_was_touched_in = bool(cached.get("was_touched", False) or was_touched)
        eff_touch_time_in = cached.get("touch_time") or touch_time
        eff_m5_at_touch_in = cached.get("m5_eval_at_touch")
        if callable(compute_levels_fn) and not is_invalid_zone and not is_expired_zone and not is_far_zone:
            try:
                fresh_lvls = compute_levels_fn(planned_entry)
                if isinstance(fresh_lvls, dict) and fresh_lvls.get("ok"):
                    new_tp1 = round(float(fresh_lvls.get("tp1") or 0.0), digits)
                    new_tp2 = round(float(fresh_lvls.get("tp2") or 0.0), digits)
                    if new_tp1 > 0.0 and ((demand and planned_entry < new_tp1 < tp1 - 0.05) or ((not demand) and planned_entry > new_tp1 > tp1 + 0.05)):
                        n_rr1, _ = _compute_rr_pair(d, planned_entry, setup_sl, new_tp1, tp1)
                        if n_rr1 >= min_rr - 1e-9:
                            cand_tp2 = new_tp2 if (new_tp2 > 0.0 and ((demand and new_tp1 < new_tp2 <= tp1) or ((not demand) and new_tp1 > new_tp2 >= tp1))) else tp1
                            n_rr1, n_rr2 = _compute_rr_pair(d, planned_entry, setup_sl, new_tp1, cand_tp2)
                            tp1, tp2, rr1, rr2 = new_tp1, (cand_tp2 if n_rr2 > 0 else 0.0), n_rr1, n_rr2
                            cached["tp1"] = tp1
                            cached["tp2"] = tp2
                            cached["rr1"] = rr1
                            cached["rr2"] = rr2
                            _save_trade_plan_state()
            except Exception:
                pass
    else:
        planned_entry = det_planned_entry
        setup_sl = 0.0
        s_inv_final = round(s_inv, digits)
        tp1 = 0.0
        tp2 = 0.0
        plan_created_ts = time.time()
        plan_created_at = format_wib_timestamp(default_now=True)
        eff_was_touched_in = bool(was_touched)
        eff_touch_time_in = touch_time
        eff_m5_at_touch_in = None

        # Prioritaskan level SL/TP dari early_pending_plan atau confirmed_plan bila valid, jika tidak hitung via compute_levels_fn
        src_levels = None
        if isinstance(early_pending_plan, dict) and early_pending_plan.get("ready"):
            ep_sl = float(early_pending_plan.get("sl") or 0.0)
            ep_tp1 = float(early_pending_plan.get("tp1") or 0.0)
            ep_tp2 = float(early_pending_plan.get("tp2") or 0.0)
            ep_rr1, ep_rr2 = _compute_rr_pair(d, planned_entry, ep_sl, ep_tp1, ep_tp2)
            ep_risk = abs(planned_entry - ep_sl)
            ep_tp1_d = abs(ep_tp1 - planned_entry)
            if ep_rr1 >= min_rr - 1e-9 and ep_risk >= min_stop and ep_tp1_d >= min_stop:
                src_levels = {
                    "ok": True,
                    "setup_sl": round(ep_sl, digits),
                    "struct_inv": round(s_inv, digits),
                    "tp1": round(ep_tp1, digits),
                    "tp2": round(ep_tp2, digits) if ep_rr2 > 0 else 0.0,
                }
        if src_levels is None and callable(compute_levels_fn):
            src_levels = compute_levels_fn(planned_entry)
        if (src_levels is None or not src_levels.get("ok")) and isinstance(confirmed_plan, dict):
            c_sl = float(confirmed_plan.get("sl") or 0.0)
            c_tp1 = float(confirmed_plan.get("tp1") or 0.0)
            c_tp2 = float(confirmed_plan.get("tp2") or 0.0)
            c_sinv = float(confirmed_plan.get("struct_inv") or s_inv)
            c_rr1, c_rr2 = _compute_rr_pair(d, planned_entry, c_sl, c_tp1, c_tp2)
            if c_rr1 > 0.0:
                src_levels = {
                    "ok": True,
                    "setup_sl": round(c_sl, digits),
                    "struct_inv": round(c_sinv, digits),
                    "tp1": round(c_tp1, digits),
                    "tp2": round(c_tp2, digits) if c_rr2 > 0 else 0.0,
                }

        if not isinstance(src_levels, dict) or not src_levels.get("ok"):
            err_rsn = str((src_levels or {}).get("reason") or "PLAN_INVALID")
            return _rej(err_rsn, s_inv_v=s_inv)

        setup_sl = round(float(src_levels.get("setup_sl") or 0.0), digits)
        s_inv_final = round(float(src_levels.get("struct_inv") or s_inv), digits)
        tp1 = round(float(src_levels.get("tp1") or 0.0), digits)
        tp2 = round(float(src_levels.get("tp2") or 0.0), digits)

        if setup_sl <= 0.0:
            return _rej("SL_UNAVAILABLE", s_inv_v=s_inv_final)
        if (demand and not (setup_sl < planned_entry and setup_sl < s_inv_final + 1e-6)) or ((not demand) and not (setup_sl > planned_entry and setup_sl > s_inv_final - 1e-6)):
            return _rej("SL_INVALID", s_inv_v=s_inv_final)
        risk_pts = round(abs(planned_entry - setup_sl), digits)
        if risk_pts <= 0.5:
            return _rej(f"SL_TOO_TIGHT ({risk_pts:.2f})", s_inv_v=s_inv_final)
        if max_sl_dist > 0.0 and risk_pts > max_sl_dist:
            return _rej(f"SL_EXCEEDS_MAX ({risk_pts:.2f}>{max_sl_dist:g})", s_inv_v=s_inv_final)
        if min_stop > 0.0 and risk_pts < min_stop:
            return _rej("BELOW_MT5_STOP_LEVEL", s_inv_v=s_inv_final)

        rr1, rr2 = _compute_rr_pair(d, planned_entry, setup_sl, tp1, tp2)
        if rr1 <= 0.0:
            return _rej("TARGET_INVALID", s_inv_v=s_inv_final)
        reward_tp1 = round(abs(tp1 - planned_entry), digits)
        if min_stop > 0.0 and reward_tp1 < min_stop:
            return _rej("BELOW_MT5_STOP_LEVEL", s_inv_v=s_inv_final)
        if rr1 < min_rr - 1e-9 and not isinstance(confirmed_plan, dict):
            return _rej(f"RR_TOO_LOW (1:{rr1:.2f}<1:{min_rr:g})", s_inv_v=s_inv_final)
        if rr2 <= 0.0:
            tp2 = 0.0

        if cache_k:
            _FROZEN_TRADE_PLANS[cache_k] = {
                "setup_version": cur_ver,
                "created_at_ts": plan_created_ts,
                "created_at": plan_created_at,
                "revision_reason": revision_reason,
                "revised_at": revised_at,
                "struct_sig": struct_sig,
                "atr_ref": atr_now,
                "planned_entry": planned_entry,
                "setup_sl": setup_sl,
                "struct_inv": s_inv_final,
                "tp1": tp1,
                "tp2": tp2,
                "rr1": rr1,
                "rr2": rr2,
                "was_touched": bool(eff_was_touched_in),
                "touch_time": eff_touch_time_in,
                "m5_eval_at_touch": eff_m5_at_touch_in,
                "missed_move_threshold": float(missed_move_threshold or 0.0),
            }
            _save_trade_plan_state()

    conf_e = float((confirmed_plan or {}).get("entry") or 0.0) if isinstance(confirmed_plan, dict) else 0.0
    early_ready = bool(isinstance(early_pending_plan, dict) and early_pending_plan.get("ready"))
    early_exp = bool(isinstance(early_pending_plan, dict) and early_pending_plan.get("is_expired"))
    mt5_fill = verify_mt5_fill_for_plan(
        d, lo, hi, planned_entry, mt5_orders_snapshot,
        source=src, timeframe=tf, zone_id=zid, setup_version=cur_ver,
        setup_created_ts=plan_created_ts,
        zone_formation_ts=_parse_zone_formation_ts_from_id(zid),
    )

    inv_locked = _get_invalidated_setup_version(src, tf, zid, cur_ver)
    if inv_locked is not None:
        lc = {
            "status": inv_locked["status"],
            "zone_status": inv_locked["zone_status"],
            "setup_status": inv_locked["setup_status"],
            "reason": inv_locked["reason"],
            "transition_reason": inv_locked["reason"],
            "entry_method": "PENDING_LIMIT",
            "is_executable": False,
            "is_suspended": (inv_locked["zone_status"] in ("SUSPENDED", "BREAKOUT_PENDING")),
            "live_price": round(lp, digits) if lp > 0 else 0.0,
            "live_entry": round(lp, digits) if lp > 0 else 0.0,
            "live_rr1": 0.0,
            "live_rr2": 0.0,
            "was_touched": bool(eff_was_touched_in),
            "touch_time": eff_touch_time_in,
            "m5_eval_status": m5_eval_status,
            "m5_eval_at_touch": eff_m5_at_touch_in,
            "mt5_fill_ticket": None,
        }
    else:
        lc = evaluate_trade_plan_lifecycle(
            direction=d,
            lo=lo,
            hi=hi,
            planned_entry=planned_entry,
            setup_sl=setup_sl,
            struct_inv=s_inv_final,
            tp1=tp1,
            tp2=tp2,
            live_price=lp,
            bid=bid,
            ask=ask,
            digits=digits,
            is_closed=is_closed,
            is_invalid_zone=is_invalid_zone,
            invalid_reason=invalid_reason,
            is_expired=(is_expired_zone or early_exp),
            is_breakout_pending=is_breakout_pending,
            confirmed_entry=conf_e,
            early_pending_ready=early_ready,
            cond_reason=cond_reason,
            was_touched=eff_was_touched_in,
            touch_time=eff_touch_time_in,
            m5_eval_status=m5_eval_status,
            m5_eval_at_touch=eff_m5_at_touch_in,
            is_missed=is_missed,
            missed_reason=missed_reason,
            missed_move_threshold=float(missed_move_threshold or 0.0),
            bars_since_touch=bars_since_touch,
            max_bars_after_touch=max_bars_after_touch,
            verified_mt5_fill=mt5_fill,
            sl_breached_hist=sl_breached_hist,
        )
        if lc["status"] in ("INVALID", "SL_BREACHED", "EXPIRED", "MISSED") or lc["setup_status"] in ("SL_BREACHED", "EXPIRED", "MISSED") or lc["zone_status"] == "STRUCTURALLY_INVALID":
            _record_invalidated_setup_version(
                src, tf, zid, cur_ver,
                status=lc["status"],
                setup_status=lc["setup_status"],
                zone_status=lc["zone_status"],
                reason=lc["reason"],
            )
        elif is_far_zone and lc["status"] != "FILLED":
            # Filter jarak sementara (FAR_ZONE): JANGAN catat ke _INVALIDATED_SETUP_VERSIONS agar tetap dibedakan dari invalidasi permanen
            lc["status"] = "REJECTED"
            lc["zone_status"] = "FAR_REFERENCE"
            lc["setup_status"] = "FAR_ZONE"
            lc["reason"] = "FAR_ZONE"
            lc["transition_reason"] = "DISTANCE_FILTER_FAR_ZONE"
            lc["is_executable"] = False
            lc["is_suspended"] = False
    if cache_k and cache_k in _FROZEN_TRADE_PLANS:
        if lc.get("was_touched"):
            _FROZEN_TRADE_PLANS[cache_k]["was_touched"] = True
        if lc.get("touch_time") and not _FROZEN_TRADE_PLANS[cache_k].get("touch_time"):
            _FROZEN_TRADE_PLANS[cache_k]["touch_time"] = lc["touch_time"]
        if lc.get("m5_eval_at_touch") and not _FROZEN_TRADE_PLANS[cache_k].get("m5_eval_at_touch"):
            _FROZEN_TRADE_PLANS[cache_k]["m5_eval_at_touch"] = lc["m5_eval_at_touch"]
        if missed_move_threshold and missed_move_threshold > 0:
            _FROZEN_TRADE_PLANS[cache_k]["missed_move_threshold"] = float(missed_move_threshold)

    entry_compat = lc["live_entry"] if (lc["entry_method"] == "CONFIRMED" and lc["live_entry"] > 0) else planned_entry
    return {
        "source": src,
        "zone_id": zid,
        "setup_version": cur_ver,
        "created_at_ts": plan_created_ts,
        "created_at": plan_created_at,
        "zone_formation_ts": _parse_zone_formation_ts_from_id(zid),
        "mt5_fill_ticket": lc.get("mt5_fill_ticket"),
        "mt5_fill_time_wib": lc.get("mt5_fill_time_wib") or "",
        "mt5_status": lc.get("mt5_status") or (mt5_fill.get("mt5_status") if isinstance(mt5_fill, dict) else ""),
        "setup_match": lc.get("setup_match") or (mt5_fill.get("setup_match") if isinstance(mt5_fill, dict) else ""),
        "existing_position_info": (dict(mt5_fill) if (isinstance(mt5_fill, dict) and not mt5_fill.get("verified")) else None),
        "revision_reason": revision_reason,
        "revised_at": revised_at,
        "timeframe": tf,
        "zone_type": zt,
        "grade": str(grade or "N/A").strip(),
        "score": float(score or 0.0),
        "distance": round(dist_val, 2),
        "zone_low": round(lo, digits),
        "zone_high": round(hi, digits),
        "digits": digits,
        "direction": d,
        "entry_method": lc["entry_method"],
        "entry": entry_compat,
        "planned_entry": planned_entry,
        "live_entry": lc["live_entry"],
        "live_price": lc["live_price"],
        "struct_inv": s_inv_final,
        "setup_sl": setup_sl,
        "sl": setup_sl,
        "tp1": tp1,
        "tp2": tp2,
        "rr1": rr1,
        "rr2": rr2,
        "planned_rr1": rr1,
        "planned_rr2": rr2,
        "live_rr1": lc["live_rr1"],
        "live_rr2": lc["live_rr2"],
        "status": lc["status"],
        "zone_status": lc["zone_status"],
        "setup_status": lc["setup_status"],
        "reason": lc["reason"],
        "transition_reason": lc["transition_reason"],
        "cond_reason": cond_reason,
        "why": str(why or "").strip(),
        "is_executable": lc["is_executable"],
        "is_suspended": lc["is_suspended"],
        "is_breakout_pending": (lc["zone_status"] == "BREAKOUT_PENDING"),
        "was_touched": lc["was_touched"],
        "touch_time": lc["touch_time"],
        "m5_eval_status": lc["m5_eval_status"],
        "m5_eval_at_touch": lc["m5_eval_at_touch"],
        "missed_move_threshold": float(missed_move_threshold or 0.0),
        "mt5_fill_ticket": lc["mt5_fill_ticket"],
    }


def _trade_plan_grade_rank(grade: Any) -> int:
    g = str(grade or "").strip().upper()
    if g == "A+":
        return 0
    if g == "A":
        return 1
    if g == "B":
        return 2
    if g == "C":
        return 3
    if g == "D":
        return 4
    return 3


def select_best_trade_plan(plans, direction=None, allow_rejected: bool = True):
    """Pilih satu trade plan terbaik secara deterministik: READY > TOUCHED > CONDITIONAL > BREAKOUT_PENDING > MISSED > EXPIRED > INVALID > REJECTED."""
    if not plans:
        return None
    norm_dir = _norm_direction_for_invalid(direction)
    status_order = {
        "FILLED": 0,
        "READY": 0,
        "TOUCHED_UNCONFIRMED": 1,
        "TOUCHED": 1,
        "CONDITIONAL": 2,
        "SUSPENDED": 3,
        "BREAKOUT_PENDING": 3,
        "MISSED": 4,
        "EXPIRED": 5,
        "SL_BREACHED": 6,
        "INVALID": 6,
        "REJECTED": 7,
    }
    best_by_id = {}
    for p in plans:
        if not isinstance(p, dict):
            continue
        st = str(p.get("status") or "REJECTED").strip().upper()
        if st == "NO_ACTIVE_PLAN":
            continue
        if st not in status_order:
            st = "REJECTED"
        if not allow_rejected and st == "REJECTED":
            continue
        p_dir = str(p.get("direction") or "").strip().upper()
        dir_mm = 0 if (not norm_dir or p_dir == norm_dir) else 1
        try:
            rr1 = float(p.get("planned_rr1") or p.get("rr1") or 0.0)
            if not math.isfinite(rr1):
                rr1 = 0.0
        except (TypeError, ValueError):
            rr1 = 0.0
        try:
            sc = float(p.get("score") or 0.0)
            if not math.isfinite(sc):
                sc = 0.0
        except (TypeError, ValueError):
            sc = 0.0
        try:
            dist = float(p.get("distance") if p.get("distance") is not None else 9999.0)
            if not math.isfinite(dist):
                dist = 9999.0
        except (TypeError, ValueError):
            dist = 9999.0
        src = str(p.get("source") or "SMC").strip().upper()
        zid = str(p.get("zone_id") or "").strip()
        dedup_k = f"{src}:{zid}" if zid else f"{src}:{p_dir}:{p.get('planned_entry') or p.get('entry')}:{st}"
        sort_k = (
            status_order[st],
            dir_mm,
            _trade_plan_grade_rank(p.get("grade")),
            -round(rr1, 2),
            -round(sc, 1),
            round(dist, 2),
            0 if src == "SMC" else 1,
            zid,
        )
        prev = best_by_id.get(dedup_k)
        if prev is None or sort_k < prev[0]:
            best_by_id[dedup_k] = (sort_k, p)
    if not best_by_id:
        return None
    return min(best_by_id.values(), key=lambda pair: pair[0])[1]


def _reconcile_archived_far_zone_plan(
    prev: Optional[Dict[str, Any]],
    live_price: float = 0.0,
    bid: Optional[float] = None,
    ask: Optional[float] = None,
) -> Tuple[Optional[Dict[str, Any]], bool]:
    """
    Rekonsiliasi arsip PREVIOUS PLAN yang sebelumnya tercatat REJECTED (FAR_ZONE):
    - Pulihkan level beku (planned_entry, setup_sl, struct_inv, tp1, tp2, rr1) bila tersedia.
    - Evaluasi breach SetupSL & structural invalidation terlebih dahulu dari harga terbaru / state zona.
    - Jika terbukti breached/invalid, ubah status ke INVALID/SL_BREACHED dan kunci permanen (Zone ID + version).
    - Jika benar-benar hanya di luar jarak pada sisi valid, beri label eksplisit DISTANCE_FILTER (FAR_ZONE).
    """
    if not isinstance(prev, dict) or not prev:
        return prev, False
    p_rsn = str(prev.get("reason") or "").strip().upper()
    p_arch = str(prev.get("archive_reason") or "").strip().upper()
    if " | " in str(prev.get("reason") or "") or " | " in str(prev.get("archive_reason") or ""):
        out_fix = dict(prev)
        if " | " in str(out_fix.get("reason") or ""):
            out_fix["reason"] = str(out_fix["reason"]).replace(" | ", "+")
        if " | " in str(out_fix.get("archive_reason") or ""):
            out_fix["archive_reason"] = str(out_fix["archive_reason"]).replace(" | ", "+")
        prev = out_fix
        if p_rsn != "FAR_ZONE" and "FAR_ZONE" not in p_arch:
            return prev, True
    if p_rsn != "FAR_ZONE" and "FAR_ZONE" not in p_arch:
        return prev, False

    p_src = str(prev.get("source") or "SMC").strip().upper()
    p_zid = str(prev.get("zone_id") or "").strip()
    p_ver = int(prev.get("setup_version") or 1)
    p_dir = str(prev.get("direction") or "").strip().upper()
    demand = (p_dir == "BUY")
    changed = False
    out = dict(prev)

    # Temukan timeframe & frozen record bila ada
    p_tf = str(out.get("timeframe") or "").strip().upper()
    frozen = get_frozen_trade_plan(p_src, p_zid, timeframe=p_tf or None, setup_version=p_ver)
    if not p_tf and p_zid:
        m_tf = re.search(r"^(?:ZONE|SNR)_([A-Z0-9+]+)_", p_zid.upper())
        if m_tf:
            p_tf = m_tf.group(1)
        else:
            suffix = f":{p_zid}"
            prefix = f"{p_src}:"
            for fk in _FROZEN_TRADE_PLANS:
                if fk.startswith(prefix) and fk.endswith(suffix):
                    parts = fk.split(":")
                    if len(parts) >= 3:
                        p_tf = parts[1].upper()
                        break
    if p_tf and not out.get("timeframe"):
        out["timeframe"] = p_tf
        changed = True

    if isinstance(frozen, dict):
        for k_f in ("planned_entry", "setup_sl", "struct_inv", "tp1", "tp2", "rr1"):
            if float(out.get(k_f) or 0.0) <= 0.0 and float(frozen.get(k_f) or 0.0) > 0.0:
                out[k_f] = float(frozen[k_f])
                changed = True

    sl_v = float(out.get("setup_sl") or 0.0)
    sinv_v = float(out.get("struct_inv") or 0.0)
    lp = float(live_price) if (isinstance(live_price, (int, float)) and math.isfinite(live_price) and live_price > 0) else 0.0
    sl_px = (float(bid) if (isinstance(bid, (int, float)) and bid > 0) else lp) if demand else (float(ask) if (isinstance(ask, (int, float)) and ask > 0) else lp)

    sl_hit = bool(sl_v > 0.0 and sl_px > 0.0 and ((demand and sl_px <= sl_v) or ((not demand) and sl_px >= sl_v)))
    struct_broken = False
    path = _get_trade_plan_state_path()
    if p_zid and path and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                st_raw = json.load(f)
            if isinstance(st_raw, dict):
                if p_src == "SMC":
                    inv_smc = (st_raw.get("invalidated_smc_zones") or {}).get(p_zid)
                    trk_smc = (st_raw.get("tracked_zones") or {}).get(p_zid)
                    if isinstance(inv_smc, dict) and (inv_smc.get("is_broken") or str(inv_smc.get("status") or "").upper() in ("INVALIDATED", "BROKEN")):
                        struct_broken = True
                    elif isinstance(trk_smc, dict) and str(trk_smc.get("status") or "").upper() in ("INVALIDATED", "BROKEN"):
                        struct_broken = True
                    elif sinv_v > 0.0 and lp > 0.0 and ((demand and lp < sinv_v) or ((not demand) and lp > sinv_v)) and sl_hit:
                        # Harga sudah jauh melampaui StructInv & SetupSL secara bersamaan
                        struct_broken = True
                elif p_src == "SNR":
                    for snr_rec in (st_raw.get("snr_lifecycle_zones") or {}).values():
                        if isinstance(snr_rec, dict):
                            bo_i = snr_rec.get("breakout_info") if isinstance(snr_rec.get("breakout_info"), dict) else {}
                            if str(bo_i.get("zone_id") or "") == p_zid and str(snr_rec.get("lifecycle_status") or "").upper() in ("INVALIDATED", "BROKEN"):
                                struct_broken = True
                                break
        except Exception:
            pass

    if sl_hit and struct_broken:
        new_st, new_s_st, new_z_st = "INVALID", "SL_BREACHED", "STRUCTURALLY_INVALID"
        new_rsn = "SL_BREACHED+ZONE_STRUCTURALLY_INVALID"
        new_arch = f"{new_st} ({new_rsn})"
    elif sl_hit:
        new_st, new_s_st, new_z_st = "SL_BREACHED", "SL_BREACHED", "ACTIVE"
        new_rsn = "SL_BREACHED"
        new_arch = "SL_BREACHED"
    elif struct_broken:
        new_st, new_s_st, new_z_st = "INVALID", "INVALID", "STRUCTURALLY_INVALID"
        new_rsn = "ZONE_STRUCTURALLY_INVALID"
        new_arch = f"{new_st} ({new_rsn})"
    else:
        if out.get("archive_reason") == "REJECTED (FAR_ZONE)":
            out["archive_reason"] = "DISTANCE_FILTER (FAR_ZONE)"
            changed = True
        return out, changed

    if out.get("status") != new_st or out.get("reason") != new_rsn or out.get("archive_reason") != new_arch:
        out["status"] = new_st
        out["reason"] = new_rsn
        out["archive_reason"] = new_arch
        changed = True
    if p_tf and p_zid:
        _record_invalidated_setup_version(
            p_src, p_tf, p_zid, p_ver,
            status=new_st,
            setup_status=new_s_st,
            zone_status=new_z_st,
            reason=new_rsn,
        )
    return out, changed


def _archive_trade_plan(scope_key: str, plan: Dict[str, Any], archive_reason: str) -> Dict[str, Any]:
    """Arsipkan plan lama sebagai PREVIOUS PLAN dengan alasan & timestamp WIB tanpa menggeser level."""
    prev_existing = _PREVIOUS_TRADE_PLANS.get(scope_key)
    p_src = str(plan.get("source") or "").strip().upper()
    p_zid = str(plan.get("zone_id") or "").strip()
    p_tf = str(plan.get("timeframe") or "").strip().upper()
    p_ver = int(plan.get("setup_version") or 1)
    p_st = str(plan.get("status") or "").strip().upper()
    p_rsn = str(plan.get("reason") or "").strip()
    eff_arch_rsn = "DISTANCE_FILTER (FAR_ZONE)" if (p_st == "REJECTED" and p_rsn.upper() == "FAR_ZONE") else str(archive_reason or p_st).strip()
    if (
        isinstance(prev_existing, dict)
        and str(prev_existing.get("source") or "").strip().upper() == p_src
        and str(prev_existing.get("zone_id") or "").strip() == p_zid
        and int(prev_existing.get("setup_version") or 1) == p_ver
        and str(prev_existing.get("status") or "").strip().upper() == p_st
        and str(prev_existing.get("archive_reason") or "").strip() == eff_arch_rsn
    ):
        return prev_existing
    frozen_fb = get_frozen_trade_plan(p_src, p_zid, timeframe=p_tf or None, setup_version=p_ver) if p_zid else None
    pe_v = float(plan.get("planned_entry") or plan.get("entry") or (frozen_fb.get("planned_entry") if isinstance(frozen_fb, dict) else 0.0) or 0.0)
    sl_v = float(plan.get("setup_sl") or plan.get("sl") or (frozen_fb.get("setup_sl") if isinstance(frozen_fb, dict) else 0.0) or 0.0)
    sinv_v = float(plan.get("struct_inv") or (frozen_fb.get("struct_inv") if isinstance(frozen_fb, dict) else 0.0) or 0.0)
    tp1_v = float(plan.get("tp1") or (frozen_fb.get("tp1") if isinstance(frozen_fb, dict) else 0.0) or 0.0)
    tp2_v = float(plan.get("tp2") or (frozen_fb.get("tp2") if isinstance(frozen_fb, dict) else 0.0) or 0.0)
    rr1_v = float(plan.get("planned_rr1") or plan.get("rr1") or (frozen_fb.get("rr1") if isinstance(frozen_fb, dict) else 0.0) or 0.0)
    archived = {
        "source": p_src,
        "timeframe": p_tf,
        "zone_id": p_zid,
        "setup_version": p_ver,
        "direction": str(plan.get("direction") or "N/A").strip().upper(),
        "planned_entry": pe_v,
        "setup_sl": sl_v,
        "struct_inv": sinv_v,
        "tp1": tp1_v,
        "tp2": tp2_v,
        "rr1": rr1_v,
        "status": p_st,
        "reason": p_rsn,
        "archive_reason": eff_arch_rsn,
        "archived_at": format_wib_timestamp(default_now=True),
    }
    _PREVIOUS_TRADE_PLANS[scope_key] = archived
    return archived


def resolve_trade_plan_focus(
    plans,
    direction=None,
    scope: str = "UNIFIED",
    live_price: float = 0.0,
    bid: Optional[float] = None,
    ask: Optional[float] = None,
    is_closed: bool = False,
    mt5_orders_snapshot: Optional[Dict[str, Any]] = None,
    htf_conflict: bool = False,
    m5_eval_status: Optional[str] = None,
    revalidation_ctx: Optional[Dict[str, Any]] = None,
    h4_bias: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Selector fokus Trade Plan bersama untuk SMC, SNR, dan UNIFIED (`scan`, `scan snr`, `pantau`).
    - Tidak pernah mengganti FOCUS aktif/conditional/touched/suspended dengan kandidat REJECTED.
    - Mempertahankan plan fokus berdasarkan exact (source, zone_id) meskipun SNR sementara NONE karena BREAKOUT_PENDING.
    - Mengarsipkan plan yang benar-benar terminal (INVALID/EXPIRED/MISSED) sebagai PREVIOUS PLAN + timestamp,
      lalu memilih NEXT PLAN hanya jika lolos filter arah, kualitas, risiko, dan RR.
    """
    scope_k = str(scope or "UNIFIED").strip().upper()
    _load_trade_plan_state_once()
    if isinstance(_PREVIOUS_TRADE_PLANS.get(scope_k), dict):
        rec_prev, rec_chg = _reconcile_archived_far_zone_plan(
            _PREVIOUS_TRADE_PLANS.get(scope_k), live_price=live_price, bid=bid, ask=ask,
        )
        if rec_chg and isinstance(rec_prev, dict):
            _PREVIOUS_TRADE_PLANS[scope_k] = rec_prev
            if isinstance(_ACTIVE_TRADE_PLAN_FOCUS.get(scope_k), dict):
                _ACTIVE_TRADE_PLAN_FOCUS[scope_k]["previous_plan"] = rec_prev
            _save_trade_plan_state()
    elif isinstance((_ACTIVE_TRADE_PLAN_FOCUS.get(scope_k) or {}).get("previous_plan"), dict):
        rec_prev, rec_chg = _reconcile_archived_far_zone_plan(
            _ACTIVE_TRADE_PLAN_FOCUS[scope_k]["previous_plan"], live_price=live_price, bid=bid, ask=ask,
        )
        if rec_chg and isinstance(rec_prev, dict):
            _PREVIOUS_TRADE_PLANS[scope_k] = rec_prev
            _ACTIVE_TRADE_PLAN_FOCUS[scope_k]["previous_plan"] = rec_prev
            _save_trade_plan_state()
    norm_dir = _norm_direction_for_invalid(direction)
    valid_plans = [p for p in (plans or []) if isinstance(p, dict) and str(p.get("status") or "").strip().upper() != "NO_ACTIVE_PLAN"]

    by_key: Dict[str, Dict[str, Any]] = {}
    for p in valid_plans:
        p_src = str(p.get("source") or "SMC").strip().upper()
        p_zid = str(p.get("zone_id") or "").strip()
        if p_zid:
            by_key[f"{p_src}:{p_zid}"] = p

    qualified_pool = [
        p for p in valid_plans
        if str(p.get("status") or "").strip().upper() in _QUALIFIED_NEXT_STATUSES
        and float(p.get("planned_rr1") or p.get("rr1") or 0.0) > 0.0
    ]
    dir_qualified_pool = [
        p for p in qualified_pool
        if (not norm_dir or str(p.get("direction") or "").strip().upper() == norm_dir)
    ]
    active_pool = [
        p for p in valid_plans
        if str(p.get("status") or "").strip().upper() in _ACTIVE_FOCUS_STATUSES
        and float(p.get("planned_entry") or p.get("entry") or 0.0) > 0.0
    ]
    terminal_pool = [
        p for p in valid_plans
        if str(p.get("status") or "").strip().upper() in _TERMINAL_PLAN_STATUSES
    ]
    rejected_pool = [
        p for p in valid_plans
        if str(p.get("status") or "").strip().upper() == "REJECTED"
    ]
    best_terminal = select_best_trade_plan(terminal_pool, direction=norm_dir, allow_rejected=False) if terminal_pool else None
    best_rejected = select_best_trade_plan(rejected_pool, direction=norm_dir, allow_rejected=True) if rejected_pool else None

    def _attach_switch_notice(chosen_plan: Dict[str, Any], old_src: str, old_zid: str, sw_reason: str) -> None:
        new_src = str(chosen_plan.get("source") or "SMC").strip().upper()
        new_zid = str(chosen_plan.get("zone_id") or "").strip()
        if old_src and new_src and old_src != new_src:
            sw_id = f"{scope_k}:{old_src}:{old_zid}->{new_src}:{new_zid}:{sw_reason}"
            if sw_id not in _EMITTED_PLAN_SWITCHES:
                chosen_plan["plan_switch_notice"] = f"{old_src}→{new_src} | {sw_reason}"
                chosen_plan["plan_switch_id"] = sw_id

    def _attach_selection_summary(chosen_plan: Dict[str, Any], held_focus: bool = False) -> None:
        if scope_k != "UNIFIED" or not isinstance(chosen_plan, dict):
            return
        ch_src = str(chosen_plan.get("source") or "").strip().upper()
        if ch_src not in ("SMC", "SNR"):
            return
        ch_dir = str(chosen_plan.get("direction") or norm_dir or "").strip().upper()
        mode_sfx = " (LOCKED)" if held_focus else ""
        other_src = "SNR" if ch_src == "SMC" else "SMC"
        other_plans = [p for p in valid_plans if str(p.get("source") or "").strip().upper() == other_src]
        if not other_plans:
            chosen_plan["plan_selection_summary"] = f"{ch_src}{mode_sfx} | {other_src}: NONE"
            return
        other_best = select_best_trade_plan(other_plans, direction=norm_dir, allow_rejected=True)
        if not isinstance(other_best, dict):
            chosen_plan["plan_selection_summary"] = f"{ch_src}{mode_sfx} | {other_src}: NONE"
            return
        o_st = str(other_best.get("status") or "REJECTED").strip().upper()
        o_rsn = str(other_best.get("reason") or "").strip()
        o_dir = str(other_best.get("direction") or "").strip().upper()
        if o_st == "REJECTED":
            blk = o_rsn if (o_rsn and o_rsn.upper() not in ("UNQUALIFIED", "PLAN_INVALID")) else "FILTER_REJECTED"
            chosen_plan["plan_selection_summary"] = f"{ch_src}{mode_sfx} | {other_src} BLOCKED: {blk}"
        elif o_st in _TERMINAL_PLAN_STATUSES or o_st in _SUSPENDED_FOCUS_STATUSES:
            blk = "SUSPENDED" if o_st in _SUSPENDED_FOCUS_STATUSES else o_st
            chosen_plan["plan_selection_summary"] = f"{ch_src}{mode_sfx} | {other_src} BLOCKED: {blk}"
        elif o_rsn.upper() == "COUNTER_HTF" or (ch_dir in ("BUY", "SELL") and o_dir in ("BUY", "SELL") and o_dir != ch_dir):
            chosen_plan["plan_selection_summary"] = f"{ch_src}{mode_sfx} | {other_src} BLOCKED: COUNTER_HTF"
        else:
            mode_tag = "LOCKED" if held_focus else "PRIORITY"
            o_sfx = f" ({o_rsn})" if (o_st == "CONDITIONAL" and o_rsn and o_rsn.upper() != o_st) else ""
            chosen_plan["plan_selection_summary"] = f"{ch_src} ({mode_tag}) | {other_src}: {o_st}{o_sfx}"

    prev_focus = _ACTIVE_TRADE_PLAN_FOCUS.get(scope_k)
    if isinstance(prev_focus, dict):
        pf_src = str(prev_focus.get("source") or "SMC").strip().upper()
        pf_zid = str(prev_focus.get("zone_id") or "").strip()
        pf_key = f"{pf_src}:{pf_zid}"
        updated_focus = by_key.get(pf_key)
        if updated_focus is None:
            # Zona lama tidak muncul di pool aktif (mis. SNR jadi NONE saat BREAKOUT_PENDING): tetap pantau sampai lifecycle selesai
            updated_focus = refresh_trade_plan_live_state(
                prev_focus,
                live_price=live_price,
                bid=bid,
                ask=ask,
                is_closed=is_closed,
                m5_eval_status=m5_eval_status,
                mt5_orders_snapshot=mt5_orders_snapshot,
                htf_conflict=htf_conflict,
                revalidation_ctx=revalidation_ctx,
                h4_bias=h4_bias,
            )
        elif isinstance(updated_focus, dict):
            if (
                htf_conflict
                or m5_eval_status
                or revalidation_ctx is not None
                or h4_bias
                or (isinstance(mt5_orders_snapshot, dict) and not updated_focus.get("existing_position_info") and not updated_focus.get("mt5_fill_ticket"))
            ):
                updated_focus = refresh_trade_plan_live_state(
                    updated_focus,
                    live_price=live_price,
                    bid=bid,
                    ask=ask,
                    is_closed=is_closed,
                    m5_eval_status=m5_eval_status,
                    mt5_orders_snapshot=mt5_orders_snapshot,
                    htf_conflict=htf_conflict,
                    revalidation_ctx=revalidation_ctx,
                    h4_bias=h4_bias,
                )
            elif mt5_orders_snapshot is None and isinstance(prev_focus.get("existing_position_info"), dict) and not updated_focus.get("existing_position_info"):
                updated_focus = dict(updated_focus)
                updated_focus["existing_position_info"] = dict(prev_focus["existing_position_info"])
                updated_focus["mt5_status"] = prev_focus.get("mt5_status") or "OPEN"
                updated_focus["setup_match"] = prev_focus.get("setup_match") or "UNRELATED"
        cur_st = str((updated_focus or {}).get("status") or "").strip().upper()
        cur_rsn = str((updated_focus or {}).get("reason") or "").strip()

        if isinstance(updated_focus, dict) and cur_st in _ACTIVE_FOCUS_STATUSES:
            # Jika fokus lama SUSPENDED/BREAKOUT_PENDING (belum eligible), pindah jika ada kandidat qualified lain
            if cur_st in _SUSPENDED_FOCUS_STATUSES:
                repl_pool = [
                    p for p in dir_qualified_pool
                    if f"{str(p.get('source') or '').upper()}:{str(p.get('zone_id') or '').strip()}" != pf_key
                ]
                best_repl = select_best_trade_plan(repl_pool, direction=norm_dir, allow_rejected=False) if repl_pool else None
                if isinstance(best_repl, dict):
                    sw_rsn = f"PREV_{cur_st}" + (f" ({cur_rsn})" if cur_rsn and cur_rsn != cur_st else "")
                    archived = _archive_trade_plan(scope_k, updated_focus, sw_rsn)
                    chosen = dict(best_repl)
                    chosen["previous_plan"] = archived
                    _attach_switch_notice(chosen, pf_src, pf_zid, sw_rsn)
                    _attach_selection_summary(chosen, held_focus=False)
                    _ACTIVE_TRADE_PLAN_FOCUS[scope_k] = chosen
                    _save_trade_plan_state()
                    return chosen

            # Hanya izinkan pindah fokus bila ada kandidat READY yang lolos arah & kualitas saat fokus lama belum READY/TOUCHED/FILLED
            if cur_st not in ("READY", "TOUCHED", "TOUCHED_UNCONFIRMED", "FILLED"):
                ready_cands = [
                    p for p in dir_qualified_pool
                    if str(p.get("status") or "").strip().upper() == "READY"
                    and f"{str(p.get('source') or '').upper()}:{str(p.get('zone_id') or '').strip()}" != pf_key
                ]
                best_ready = select_best_trade_plan(ready_cands, direction=norm_dir, allow_rejected=False) if ready_cands else None
                if isinstance(best_ready, dict):
                    sw_rsn = f"UPGRADED_TO_READY (prev {cur_st})"
                    archived = _archive_trade_plan(scope_k, updated_focus, f"REPLACED_BY_READY ({cur_st})")
                    chosen = dict(best_ready)
                    chosen["previous_plan"] = archived
                    _attach_switch_notice(chosen, pf_src, pf_zid, sw_rsn)
                    _attach_selection_summary(chosen, held_focus=False)
                    _ACTIVE_TRADE_PLAN_FOCUS[scope_k] = chosen
                    _save_trade_plan_state()
                    return chosen

            # Pertahankan fokus lama (JANGAN pernah ganti karena tick/skor selama masih layak)
            held = dict(updated_focus)
            if isinstance(_PREVIOUS_TRADE_PLANS.get(scope_k), dict):
                held["previous_plan"] = _PREVIOUS_TRADE_PLANS[scope_k]
            # Pertahankan notice switch hanya bila belum pernah ditampilkan (one-shot across same-cycle calls)
            prev_sw_id = str(prev_focus.get("plan_switch_id") or "").strip()
            if prev_sw_id and prev_sw_id not in _EMITTED_PLAN_SWITCHES and prev_focus.get("plan_switch_notice"):
                held["plan_switch_notice"] = prev_focus["plan_switch_notice"]
                held["plan_switch_id"] = prev_sw_id
            _attach_selection_summary(held, held_focus=True)
            _ACTIVE_TRADE_PLAN_FOCUS[scope_k] = held
            _save_trade_plan_state()
            return held

        # Fokus lama benar-benar tidak eligible (INVALID / SL_BREACHED / EXPIRED / MISSED / REJECTED) -> arsipkan sebagai PREVIOUS PLAN
        if isinstance(updated_focus, dict):
            arch_rsn = f"{cur_st} ({cur_rsn})" if (cur_rsn and cur_rsn != cur_st) else cur_st
            archived = _archive_trade_plan(scope_k, updated_focus, arch_rsn)
        else:
            arch_rsn = cur_st or "NOT_ELIGIBLE"
            archived = _PREVIOUS_TRADE_PLANS.get(scope_k)
        _ACTIVE_TRADE_PLAN_FOCUS.pop(scope_k, None)

        # Pilih NEXT PLAN hanya bila lolos filter arah, kualitas, risiko, dan RR (bukan REJECTED)
        next_pool = [
            p for p in dir_qualified_pool
            if f"{str(p.get('source') or '').upper()}:{str(p.get('zone_id') or '').strip()}" != pf_key
        ]
        next_plan = select_best_trade_plan(next_pool, direction=norm_dir, allow_rejected=False) if next_pool else None
        if isinstance(next_plan, dict):
            chosen = dict(next_plan)
            chosen["previous_plan"] = archived
            _attach_switch_notice(chosen, pf_src, pf_zid, f"PREV_{arch_rsn}")
            _attach_selection_summary(chosen, held_focus=False)
            _ACTIVE_TRADE_PLAN_FOCUS[scope_k] = chosen
            _save_trade_plan_state()
            return chosen

        _save_trade_plan_state()
        # Tidak ada kandidat qualified: kembalikan NO_ACTIVE_PLAN + PREVIOUS PLAN + diagnostik REJECTED (bukan fokus utama)
        return {
            "source": scope_k,
            "zone_id": "",
            "setup_version": 1,
            "status": "NO_ACTIVE_PLAN",
            "reason": f"PREV_{pf_zid}_{cur_st}" if pf_zid else f"PREV_{cur_st}_NO_QUALIFIED_NEXT",
            "live_price": round(float(live_price), 2) if (isinstance(live_price, (int, float)) and live_price > 0) else 0.0,
            "previous_plan": archived,
            "rejected_diagnostic": best_rejected,
            "is_executable": False,
        }

    # Belum ada fokus aktif sebelumnya: pilih kandidat aktif terbaik (bukan REJECTED)
    initial_best = select_best_trade_plan(active_pool, direction=norm_dir, allow_rejected=False) if active_pool else None
    if isinstance(initial_best, dict):
        chosen = refresh_trade_plan_live_state(
            dict(initial_best),
            live_price=live_price,
            bid=bid,
            ask=ask,
            is_closed=is_closed,
            m5_eval_status=m5_eval_status,
            mt5_orders_snapshot=mt5_orders_snapshot,
            htf_conflict=htf_conflict,
            revalidation_ctx=revalidation_ctx,
            h4_bias=h4_bias,
        ) or dict(initial_best)
        if isinstance(_PREVIOUS_TRADE_PLANS.get(scope_k), dict):
            chosen.setdefault("previous_plan", _PREVIOUS_TRADE_PLANS[scope_k])
        _attach_selection_summary(chosen, held_focus=False)
        _ACTIVE_TRADE_PLAN_FOCUS[scope_k] = chosen
        _save_trade_plan_state()
        return chosen

    # Tidak ada kandidat aktif: kembalikan NO_ACTIVE_PLAN dengan alasan eksplisit (jangan kembalikan None diam-diam)
    prev_arch = _PREVIOUS_TRADE_PLANS.get(scope_k)
    if not isinstance(prev_arch, dict) and isinstance(best_terminal, dict):
        bt_st = str(best_terminal.get("status") or "EXPIRED").strip().upper()
        bt_rsn = str(best_terminal.get("reason") or "").strip()
        bt_arch_rsn = f"{bt_st} ({bt_rsn})" if (bt_rsn and bt_rsn != bt_st) else bt_st
        prev_arch = _archive_trade_plan(scope_k, best_terminal, bt_arch_rsn)
        _save_trade_plan_state()

    suspended_pool = [
        p for p in valid_plans
        if str(p.get("status") or "").strip().upper() in _SUSPENDED_FOCUS_STATUSES
    ]
    if isinstance(best_terminal, dict):
        bt_st = str(best_terminal.get("status") or "EXPIRED").strip().upper()
        bt_rsn = str(best_terminal.get("reason") or "").strip()
        none_reason = f"{bt_st}_{bt_rsn}" if (bt_rsn and bt_rsn != bt_st) else bt_st
    elif isinstance(best_rejected, dict):
        rej_r = str(best_rejected.get("reason") or "").strip()
        none_reason = rej_r if (rej_r and rej_r.upper() not in ("UNQUALIFIED", "PLAN_INVALID")) else "FILTER_REJECTED"
    elif suspended_pool:
        sp0 = suspended_pool[0]
        sp_st = str(sp0.get("status") or "SUSPENDED").strip().upper()
        sp_rsn = str(sp0.get("reason") or "").strip()
        none_reason = f"SUSPENDED ({sp_rsn})" if (sp_rsn and sp_rsn != sp_st) else "SUSPENDED"
    elif qualified_pool and not dir_qualified_pool:
        none_reason = "FILTER_REJECTED"
    elif isinstance(prev_arch, dict):
        none_reason = f"PREV_{str(prev_arch.get('status') or 'ENDED').strip().upper()}_NO_QUALIFIED_NEXT"
    else:
        none_reason = "MARKET_CLOSED" if is_closed else "NO_QUALIFIED_SETUP"

    return {
        "source": scope_k,
        "zone_id": "",
        "setup_version": 1,
        "status": "NO_ACTIVE_PLAN",
        "reason": none_reason,
        "live_price": round(float(live_price), 2) if (isinstance(live_price, (int, float)) and live_price > 0) else 0.0,
        "previous_plan": prev_arch if isinstance(prev_arch, dict) else None,
        "rejected_diagnostic": best_rejected,
        "is_executable": False,
    }


def _format_previous_plan_compact(prev: Optional[Dict[str, Any]], include_source: bool = False, h4_bias: Optional[str] = None) -> str:
    """Format 1 baris ringkas untuk PREVIOUS PLAN yang diarsipkan beserta alasan & timestamp."""
    if not isinstance(prev, dict) or not prev:
        return ""
    p_dir = str(prev.get("direction") or "N/A").strip().upper()
    h4_prev = str(h4_bias or prev.get("h4_bias") or "").strip().upper()
    is_ct_prev = bool(prev.get("is_counter_trend") or (p_dir == "BUY" and h4_prev == "BEARISH") or (p_dir == "SELL" and h4_prev == "BULLISH"))
    p_dir_disp = f"{p_dir} (COUNTER-TREND)" if (is_ct_prev and p_dir in ("BUY", "SELL")) else p_dir
    pe = float(prev.get("planned_entry") or 0.0)
    sl = float(prev.get("setup_sl") or 0.0)
    tp1 = float(prev.get("tp1") or 0.0)
    ver = int(prev.get("setup_version") or 1)
    zid = str(prev.get("zone_id") or "").strip()
    src = str(prev.get("source") or "").strip().upper()
    arch_rsn = str(prev.get("archive_reason") or prev.get("status") or "ARCHIVED").strip()
    arch_ts = str(prev.get("archived_at") or "").strip()
    ts_short = arch_ts[11:19] + " WIB" if len(arch_ts) >= 19 else arch_ts
    src_part = f" [{src}]" if (include_source and src) else ""
    id_part = f" [ID:{zid}]" if zid else ""
    lvl_part = f"PLAN ENTRY {pe:.2f} | SetupSL {sl:.2f} | TP1 {tp1:.2f} | " if (pe > 0 and sl > 0 and tp1 > 0) else ""
    ts_part = f" @ {ts_short}" if ts_short else ""
    return f"{p_dir_disp} | {lvl_part}{arch_rsn}{ts_part}{src_part} [v{ver}]{id_part}".strip()


def format_trade_plan_compact(plan, include_source=False, broker_tag=None) -> str:
    """Format ringkas satu Trade Plan dengan Zone ID, setup version, PLAN ENTRY, LIVE PRICE, SetupSL, TP1, TP2, RR, dan status."""
    if not isinstance(plan, dict) or not plan:
        return ""
    st = str(plan.get("status") or "REJECTED").strip().upper()
    zid = str(plan.get("zone_id") or "").strip()
    id_part = f" [ID:{zid}]" if zid and zid.upper() not in ("N/A", "NONE", "UNKNOWN") else ""
    ver = int(plan.get("setup_version") or 1)
    ver_part = f" [v{ver}]"
    src = str(plan.get("source") or "").strip().upper()
    src_part = f" [{src}]" if (include_source and src) else ""
    rsn = str(plan.get("reason") or "").strip()
    live_price = float(plan.get("live_price") or 0.0)
    lp_str = f"LIVE PRICE {live_price:.2f}" if live_price > 0.0 else "LIVE PRICE N/A"

    if st == "NO_ACTIVE_PLAN":
        rsn_txt = rsn or "NO_QUALIFIED_SETUP"
        return f"NONE | {rsn_txt} | {lp_str}".strip()

    if st == "REJECTED":
        rsn_txt = rsn or "UNQUALIFIED"
        return f"REJECTED (DIAG) | {rsn_txt} | {lp_str}{src_part}{ver_part}{id_part}".strip()

    p_dir = str(plan.get("direction") or "N/A").strip().upper()
    h4_b = str(plan.get("h4_bias") or "").strip().upper()
    is_ct = bool(plan.get("is_counter_trend") or (p_dir == "BUY" and h4_b == "BEARISH") or (p_dir == "SELL" and h4_b == "BULLISH"))
    p_dir_disp = f"{p_dir} (COUNTER-TREND)" if (is_ct and p_dir in ("BUY", "SELL")) else p_dir
    planned_entry = float(plan.get("planned_entry") or plan.get("entry") or 0.0)
    live_entry = float(plan.get("live_entry") or 0.0)
    setup_sl = float(plan.get("setup_sl") or plan.get("sl") or 0.0)
    struct_inv = float(plan.get("struct_inv") or 0.0)
    tp1 = float(plan.get("tp1") or 0.0)
    tp2 = float(plan.get("tp2") or 0.0)
    rr1 = float(plan.get("planned_rr1") or plan.get("rr1") or 0.0)
    rr2 = float(plan.get("planned_rr2") or plan.get("rr2") or 0.0)
    live_rr1 = float(plan.get("live_rr1") or 0.0)
    entry_method = str(plan.get("entry_method") or "").strip().upper()

    if planned_entry <= 0.0 or setup_sl <= 0.0 or tp1 <= 0.0:
        return f"{st} | {rsn or 'LEVELS_INVALID'} | {lp_str}{src_part}{ver_part}{id_part}".strip()

    if entry_method == "CONFIRMED" and live_entry > 0.0:
        mkt_rr_txt = f", MktRR 1:{live_rr1:.2f}" if live_rr1 > 0.0 else ""
        live_part = f"{lp_str} (LIVE ENTRY {live_entry:.2f}{mkt_rr_txt})"
    elif st in ("TOUCHED", "TOUCHED_UNCONFIRMED") and live_entry > 0.0 and live_rr1 > 0.0 and abs(live_entry - planned_entry) >= 0.01:
        live_part = f"{lp_str} (MktRR 1:{live_rr1:.2f})"
    else:
        live_part = lp_str

    sl_part = (
        f"SetupSL {setup_sl:.2f} (StructInv {struct_inv:.2f})"
        if (struct_inv > 0.0 and abs(struct_inv - setup_sl) >= 0.01)
        else f"SetupSL {setup_sl:.2f}"
    )
    has_tp2 = bool(tp2 > 0.0 and abs(tp2 - tp1) >= 0.01)
    tp2_part = f"TP2 {tp2:.2f}" if has_tp2 else "TP2 -"
    rr_part = f"RR 1:{rr1:.2f}/1:{rr2:.2f}" if (has_tp2 and rr2 > 0.0) else f"RR 1:{rr1:.2f}"
    grd_txt = str(plan.get("grade") or "").strip()
    sc_val = float(plan.get("score") or 0.0)
    qual_part = (
        f" | Grade {grd_txt}/{sc_val:.0f}"
        if (grd_txt and grd_txt.upper() not in ("N/A", "NONE") and sc_val > 0.0)
        else (f" | Grade {grd_txt}" if (grd_txt and grd_txt.upper() not in ("N/A", "NONE")) else "")
    )
    z_st = str(plan.get("zone_status") or "").strip().upper()
    s_st = str(plan.get("setup_status") or "").strip().upper()
    t_raw = str(plan.get("touch_time") or "").strip()
    t_short = t_raw[11:16] + " WIB" if len(t_raw) >= 16 else t_raw
    m5_at_t = str(plan.get("m5_eval_at_touch") or "").strip().upper()
    touch_meta_parts = []
    if st in ("TOUCHED_UNCONFIRMED", "TOUCHED", "MISSED") and t_short:
        touch_meta_parts.append(f"Touch {t_short}")
    if st in ("TOUCHED_UNCONFIRMED", "TOUCHED", "MISSED") and m5_at_t:
        touch_meta_parts.append(f"M5:{m5_at_t}")
    touch_meta = (" | " + " ".join(touch_meta_parts)) if touch_meta_parts else ""
    if z_st and s_st:
        rsn_core = f"{rsn}{touch_meta}" if (rsn and rsn != s_st) else touch_meta.lstrip(" | ")
        rsn_sfx = f" ({rsn_core})" if rsn_core else ""
        st_part = f"Zone:{z_st} Setup:{s_st}{rsn_sfx}"
    elif st in ("BREAKOUT_PENDING", "SUSPENDED"):
        st_part = f"SUSPENDED/BREAKOUT_PENDING ({rsn})" if rsn else "SUSPENDED/BREAKOUT_PENDING"
    elif rsn and st in ("CONDITIONAL", "TOUCHED", "TOUCHED_UNCONFIRMED", "MISSED", "EXPIRED", "INVALID", "SL_BREACHED", "FILLED"):
        st_part = f"{st} ({rsn}{touch_meta})"
    else:
        st_part = st
    brk_part = f" | {broker_tag}" if broker_tag else ""
    ex_pos = plan.get("existing_position_info") if isinstance(plan.get("existing_position_info"), dict) else None
    ex_part = ""
    if ex_pos and st != "FILLED":
        ex_m = str(ex_pos.get("setup_match") or "UNKNOWN").upper()
        if ex_m in ("CONFIRMED", "INFERRED"):
            ex_tkt = ex_pos.get("ticket")
            ex_t = str(ex_pos.get("position_opened_wib_short") or "").strip()
            ex_t_sfx = f" {ex_t}" if ex_t else ""
            ex_part = f" | EXISTING POSITION #{ex_tkt}{ex_t_sfx} | MATCH:{ex_m}" if ex_tkt else f" | EXISTING POSITION | MATCH:{ex_m}"
    return (
        f"{p_dir_disp} | PLAN ENTRY {planned_entry:.2f} | {live_part} | {sl_part} | "
        f"TP1 {tp1:.2f} | {tp2_part} | {rr_part}{qual_part} | {st_part}{brk_part}{ex_part}{src_part}{ver_part}{id_part}"
    ).strip()


def format_trade_plan_lines(
    plan,
    prefix="TRADE PLAN   : ",
    why_prefix="  WHY PLAN   : ",
    width=100,
    include_source=False,
    broker_tag=None,
    skip_why_for_zone_id=None,
    h4_bias=None,
):
    """Susun baris PLAN SWITCH (one-shot) + TRADE PLAN + WHY PLAN + PREV PLAN / DIAG PLAN ter-wrap rapi tanpa truncation `...`."""
    if not isinstance(plan, dict) or not plan:
        return []
    from modules.terminal_ui import wrap_dashboard_field
    txt = format_trade_plan_compact(plan, include_source=include_source, broker_tag=broker_tag)
    if not txt:
        return []
    lines = []
    sw_notice = str(plan.get("plan_switch_notice") or "").strip()
    sw_id = str(plan.get("plan_switch_id") or "").strip()
    if sw_notice and sw_id and sw_id not in _EMITTED_PLAN_SWITCHES:
        _EMITTED_PLAN_SWITCHES.append(sw_id)
        if len(_EMITTED_PLAN_SWITCHES) > 60:
            del _EMITTED_PLAN_SWITCHES[:-60]
        sw_prefix = " PLAN SWITCH : " if prefix.startswith(" ") else "PLAN SWITCH  : "
        lines.extend(wrap_dashboard_field(sw_prefix, sw_notice, width, max_lines=None))
        for fp in _ACTIVE_TRADE_PLAN_FOCUS.values():
            if isinstance(fp, dict) and str(fp.get("plan_switch_id") or "").strip() == sw_id:
                fp.pop("plan_switch_notice", None)
                fp.pop("plan_switch_id", None)
        _save_trade_plan_state()
    lines.extend(wrap_dashboard_field(prefix, txt, width, max_lines=None))
    sel_sum = str(plan.get("plan_selection_summary") or "").strip()
    if include_source and sel_sum:
        sel_prefix = " PLAN SOURCE : " if prefix.startswith(" ") else "PLAN SOURCE  : "
        lines.extend(wrap_dashboard_field(sel_prefix, sel_sum, width, max_lines=None))
    st = str(plan.get("status") or "").strip().upper()
    why_txt = str(plan.get("why") or "").strip()
    zid = str(plan.get("zone_id") or "").strip()
    if why_txt and st in ("READY", "CONDITIONAL", "TOUCHED", "TOUCHED_UNCONFIRMED", "BREAKOUT_PENDING", "SUSPENDED", "FILLED"):
        if not (skip_why_for_zone_id and zid and str(skip_why_for_zone_id).strip() == zid):
            lines.extend(wrap_dashboard_field(why_prefix, why_txt, width, max_lines=None))
    prev_plan = plan.get("previous_plan") if isinstance(plan.get("previous_plan"), dict) else None
    if prev_plan:
        prev_txt = _format_previous_plan_compact(prev_plan, include_source=include_source, h4_bias=h4_bias or plan.get("h4_bias"))
        if prev_txt:
            prev_prefix = "   PREV PLAN : " if prefix.startswith(" ") else "PREV PLAN    : "
            lines.extend(wrap_dashboard_field(prev_prefix, prev_txt, width, max_lines=None))
    if st == "NO_ACTIVE_PLAN":
        diag_rej = plan.get("rejected_diagnostic") if isinstance(plan.get("rejected_diagnostic"), dict) else None
        if diag_rej:
            diag_txt = format_trade_plan_compact(diag_rej, include_source=include_source)
            if diag_txt:
                diag_prefix = "   DIAG PLAN : " if prefix.startswith(" ") else "DIAG PLAN    : "
                lines.extend(wrap_dashboard_field(diag_prefix, diag_txt, width, max_lines=None))
    return lines

