# -*- coding: utf-8 -*-
"""
modules/config.py
Konfigurasi parameter trading, timeframe, API key, threshold risiko, status sinyal, dan model AI.
"""
import os
import json
import time
import shutil
import threading
import MetaTrader5 as mt5

SHARED_ZONE_STATE_LOCK = threading.RLock()

# Direktori root project
_DIR_SCRIPT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_VERSION = "v2.5"


def _env_cfg(key: str, default, cast_type=str):
    """Baca override konfigurasi dari environment variable atau .env secara live."""
    raw = os.environ.get(key, "").strip()
    if not raw:
        env_path = os.path.join(_DIR_SCRIPT, ".env")
        if os.path.exists(env_path):
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    prefix = f"{key}="
                    for line in f:
                        s = line.strip()
                        if s.startswith(prefix):
                            raw = s.split("=", 1)[1].strip().strip('"').strip("'")
                            break
            except Exception:
                raw = ""
    if not raw:
        return default
    try:
        if cast_type is bool:
            return raw.lower() in ("1", "true", "yes", "on", "ya")
        return cast_type(raw)
    except (TypeError, ValueError):
        return default


def set_env_value(key: str, value: str) -> bool:
    """Tulis/ubah satu baris KEY=VALUE pada .env (baris lain tidak diubah) & sinkronkan os.environ."""
    env_path = os.path.join(_DIR_SCRIPT, ".env")
    lines = []
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
    new_line = f"{key}={value}"
    replaced = False
    for i, ln in enumerate(lines):
        if ln.strip().startswith(f"{key}="):
            lines[i] = new_line
            replaced = True
            break
    if not replaced:
        lines.append(new_line)
    with open(env_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    os.environ[key] = str(value)
    return True


def _muat_api_key():
    """Baca GEMINI_API_KEY dari environment variable atau file .env di folder script (fallback ke nilai default)."""
    kunci = os.environ.get("GEMINI_API_KEY", "").strip()
    if kunci:
        return kunci
    env_path = os.path.join(_DIR_SCRIPT, ".env")
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for baris in f:
                    baris = baris.strip()
                    if baris.startswith("GEMINI_API_KEY="):
                        val = baris.split("=", 1)[1].strip().strip('"').strip("'")
                        if val:
                            return val
        except Exception:
            pass
    return ""


API_KEY = _muat_api_key()

# Nama simbol bisa berbeda tiap broker (XAUUSD, XAUUSDc, XAUUSDm, GOLD, dll.)
SYMBOL = "XAUUSDc"

# Konvensi Pips opsional (Price Distance dalam USD/oz per 1 pip).
# None = Pips tidak ditampilkan; isi nilai eksplisit (mis. 0.10) jika ingin menampilkan pips.
PIP_SIZE_PRICE = None

# Top-down: W1/D1/H4 (zona jauh ke kiri) -> H1 -> M30 (konfirmasi) -> M5 (entry presisi)
TIMEFRAMES = {
    "W1": (mt5.TIMEFRAME_W1, 260),    # ~5 tahun
    "D1": (mt5.TIMEFRAME_D1, 1000),   # ~3-4 tahun
    "H4": (mt5.TIMEFRAME_H4, 3000),   # ~1.5-2 tahun, zona jauh ke kiri
    "H1": (mt5.TIMEFRAME_H1, 400),
    "M30": (mt5.TIMEFRAME_M30, 300),
    "M15": (mt5.TIMEFRAME_M15, 300),
    "M5": (mt5.TIMEFRAME_M5, 300),
}
SWING_N = 3  # sebuah candle = swing jika High/Low-nya ekstrem dibanding N candle kiri & kanan

# Selisih jam waktu server broker terhadap UTC (contoh: 0, 2, 3).
# None = deteksi otomatis dari tick MT5 (hanya akurat saat market buka).
SERVER_UTC_OFFSET_JAM = None

# --- Konfigurasi Multi-Tier SMC Zone Engine (satuan jarak internal: Price Distance USD/oz) ---
MULTI_TIER_MAX_ZONES = 3              # Maksimal zona ditampilkan per kategori di Radar (default: 3)
HIGH_RISK_MAX_ATR_RATIO = 6.0         # Batas maksimal rasio jarak/ATR untuk kategori High Risk (Aggressive)
HIGH_RISK_MAX_DISTANCE_PT = 18.0      # Batas maksimal Price Distance (USD/oz) untuk kategori High Risk
HIGH_RISK_MIN_SCORE = 55.0            # Skor minimal kandidat High Risk (Grade B)
MEDIUM_RISK_MAX_ATR_RATIO = 16.0      # Batas maksimal rasio jarak/ATR untuk kategori Medium Risk (Balanced)
MEDIUM_RISK_MAX_DISTANCE_PT = 50.0    # Batas maksimal Price Distance (USD/oz) untuk kategori Medium Risk
MEDIUM_RISK_MIN_SCORE = 65.0          # Skor minimal kandidat Medium Risk
LOW_RISK_MAX_ATR_RATIO = 25.0         # Batas maksimal rasio jarak/ATR untuk kategori Low Risk intraday wajar
LOW_RISK_MAX_DISTANCE_PT = 75.0       # Batas maksimal Price Distance (USD/oz) untuk kategori Low Risk intraday wajar
LOW_RISK_MIN_SCORE = 75.0             # Skor minimal kandidat Low Risk (Grade A/A+)

# Distance-Aware Zone Relevance Limits per Origin Timeframe (basis ATR14 H1 + floor/cap Price Distance USD/oz)
# Format: TF -> (active_max_atr, radar_max_atr, active_min_usd, active_max_usd, radar_min_usd, radar_max_usd)
ZONE_RELEVANCE_TF_LIMITS = {
    "M5":  (1.8,  3.5,  3.5, 18.0,  7.0, 30.0),
    "M15": (2.5,  5.0,  5.0, 25.0, 10.0, 40.0),
    "M30": (3.2,  6.5,  6.5, 32.0, 13.0, 50.0),
    "H1":  (4.0,  8.0,  8.0, 40.0, 16.0, 60.0),
    "H4":  (4.5, 10.0, 10.0, 45.0, 20.0, 75.0),
    "D1":  (4.5, 10.0, 10.0, 45.0, 20.0, 75.0),
    "W1":  (4.5, 10.0, 10.0, 45.0, 20.0, 75.0),
}

# --- Gerbang sinyal & kontrol risiko: ditegakkan oleh KODE (satuan jarak internal: Price Distance USD/oz) ---
RR_MIN = 1.5                       # R:R minimal ke TP1
ENTRY_LEBAR_MAKS = 2.5             # lebar area entry maksimal (Price Distance USD/oz)
ENTRY_JARAK_MAKS = 40.0            # jarak entry dari harga live maksimal (Price Distance USD/oz)
SL_JARAK_MAKS = _env_cfg("SL_JARAK_MAKS", _env_cfg("AUTO_SL_JARAK_MAKS", 12.0, float), float)  # jarak SL maksimal dari entry (USD/oz)
SPREAD_MAKS_POIN = 1.0             # spread maksimal (Price Distance USD/oz) untuk sinyal baru
SPREAD_MAKS_RASIO_ATR = 0.35       # ...dan maksimal sekian x ATR M5 (lantai 0.5 USD/oz)
SINYAL_TTL_BELUM_TERISI_JAM = 6    # sinyal yang belum terjemput melewati batas ini dianggap kedaluwarsa
BERITA_BLACKOUT_MENIT = 15         # tidak menerbitkan sinyal baru +/- sekian menit dari berita High-impact USD
MAKS_RISIKO_TERBUKA_PCT = 3.0      # total risiko posisi + pending (jarak ke SL x lot) terhadap equity
MAKS_RUGI_HARIAN_PCT = 3.0         # batas rugi tertutup hari ini (hari server) terhadap saldo awal hari
LEBAR_CLUSTER_MAKS = 8.0           # lebar maksimal zona Golden Confluence (Price Distance USD/oz)
FILTER_TREND_AKTIF = True          # True = hanya sinyal searah trend HTF; False = matikan filter
IZINKAN_COUNTER_TREND_MEDIUM = False  # True = counter-trend boleh bila sinyal MEDIUM RISK
FILTER_SESI_AKTIF = True           # True = hanya sinyal di sesi London & New York (UTC); False = semua jam
SESI_DIIZINKAN_UTC = [(7, 16), (12, 21)]  # London (07:00-16:00 UTC) dan New York (12:00-21:00 UTC)
BIAS_MIN_SKOR = 2                  # skor minimal (-5..+5) agar bias dianggap BULLISH/BEARISH
JARAK_CONFLUENCE_MAKS = 100.0      # confluence yang lebih jauh dari ini tidak ditampilkan

# Path file penyimpanan data
FILE_JURNAL_SINYAL = os.path.join(_DIR_SCRIPT, "jurnal_sinyal_xauusd.csv")
FILE_JURNAL_ENTRY = os.path.join(_DIR_SCRIPT, "jurnal_entry_xauusd.csv")
FILE_JURNAL_ENTRY_XLSX = os.path.join(_DIR_SCRIPT, "jurnal_entry_xauusd.xlsx")
FILE_ERROR_LOG = os.path.join(_DIR_SCRIPT, "error_xauusd.log")
FILE_STATE_SINYAL = os.path.join(_DIR_SCRIPT, "state_sinyal_xauusd.json")
FILE_STATE_ZONE_TRACKER = os.path.join(_DIR_SCRIPT, "state_zone_tracker.json")


def read_shared_zone_state(filepath: str = None) -> dict:
    """
    Baca state_zone_tracker.json secara thread-safe dengan auto-recovery bila file terpotong/corrupt:
    1. Coba parse file utama.
    2. Bila gagal (JSONDecodeError/corrupt), pulihkan otomatis dari `.bak` terakhir yang valid.
    3. Bila `.bak` juga tidak ada, pindahkan file rusak ke `.corrupt.bak` dan kembalikan `{}` tanpa crash.
    """
    path = str(filepath or FILE_STATE_ZONE_TRACKER)
    if not path or not os.path.exists(path):
        return {}
    bak_path = path + ".bak"
    with SHARED_ZONE_STATE_LOCK:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            pass

        # Coba pulihkan dari backup valid terakhir (.bak)
        if os.path.exists(bak_path):
            try:
                with open(bak_path, "r", encoding="utf-8") as bf:
                    bak_data = json.load(bf)
                if isinstance(bak_data, dict):
                    try:
                        shutil.copyfile(path, path + ".corrupt.bak")
                    except Exception:
                        pass
                    try:
                        shutil.copyfile(bak_path, path)
                    except Exception:
                        pass
                    return bak_data
            except Exception:
                pass

        # Fallback terakhir: amankan file rusak & mulai state bersih agar pantau/scanner tidak crash
        try:
            shutil.replace(path, path + ".corrupt.bak")
        except Exception:
            pass
        return {}


def update_shared_zone_state(updater_fn, filepath: str = None, tag: str = "state", clean_fn=None) -> dict:
    """
    Read-modify-write atomik & thread-safe pada state_zone_tracker.json:
    - Mengunci SHARED_ZONE_STATE_LOCK agar main thread `pantau` dan background `AutoExecutionMonitor` tidak tabrakan.
    - Memakai nama file `.tmp` unik per PID + Thread ID.
    - Menyimpan salinan `.bak` valid sebelum `os.replace` dengan retry aman di Windows.
    """
    path = str(filepath or FILE_STATE_ZONE_TRACKER)
    if not path:
        return {}
    tmp_path = f"{path}.{tag}.{os.getpid()}.{threading.get_ident()}.tmp"
    bak_path = path + ".bak"
    with SHARED_ZONE_STATE_LOCK:
        data = read_shared_zone_state(path)
        if not isinstance(data, dict):
            data = {}
        updater_fn(data)
        out_obj = clean_fn(data) if callable(clean_fn) else data
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(out_obj, f, indent=2, ensure_ascii=True, default=str)
            for attempt in range(5):
                try:
                    os.replace(tmp_path, path)
                    break
                except PermissionError:
                    if attempt == 4:
                        raise
                    time.sleep(0.03)
            try:
                shutil.copyfile(path, bak_path)
            except Exception:
                pass
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
        return data
DEFAULT_PENDING_EXPIRATION_HOURS = 8
ZONE_APPROACH_DISTANCE_POINTS = 5.0
PANTAU_INTERVALS_SECONDS = {
    "IDLE": 5.0,
    "APPROACHING": 2.0,
    "AWAITING_CONFIRMATION": 1.0,
    "CONFIRMED": 5.0,
    "INVALIDATED": 10.0,
    "EXPIRED": 10.0,
}
PANTAU_FULL_REFRESH_SECONDS = 300.0
PANTAU_EXPIRY_WARNING_MINUTES = 15.0
PANTAU_TICK_POLL_SECONDS = 0.15       # Polling tick harga MT5 ringan (~150 ms) terpisah dari siklus analisis SMC/SNR
PANTAU_TICK_STALE_SECONDS = 30.0      # Ambang stale berbasis timestamp tick MT5 saat market OPEN

# Flag penanganan berita
BLOKIR_JIKA_BERITA_GAGAL = False

# Definisi status hasil sinyal
ST_BATAL_TERLEWAT = "BATAL / TERLEWAT (TP1 tercapai tanpa entry terisi)"
ST_BATAL_SL = "BATAL / SL JEBOL"
ST_BATAL_KEDALUWARSA = "BATAL / KEDALUWARSA"
ST_SELESAI_TP2 = "SELESAI / TP2 HIT"
ST_TP1_TERCAPAI = "TP1 TERCAPAI (RUNNING KE TP2)"
ST_AKTIF_ENTRY = "AKTIF DI AREA ENTRY"
ST_VALID_BERJALAN = "MASIH VALID (SUDAH TERJEMPUT, BERJALAN)"
ST_VALID_PULLBACK = "MASIH VALID (MENUNGGU PULLBACK)"
ST_VALID_DEKAT_SL = "MASIH VALID (DI DEKAT SL)"

STATUS_TERMINAL_SINYAL = frozenset({
    ST_BATAL_TERLEWAT,
    ST_BATAL_SL,
    ST_BATAL_KEDALUWARSA,
    ST_SELESAI_TP2,
})

# Status Siklus Hidup Sinyal Terpadu (Signal Manager & Decision Consistency)
SIGNAL_STATUS_NO_SETUP = "NO SETUP"
SIGNAL_STATUS_SETUP_DETECTED = "SETUP DETECTED"
SIGNAL_STATUS_WAITING_PULLBACK = "WAITING PULLBACK"
SIGNAL_STATUS_WAITING_CONFIRMATION = "WAITING CONFIRMATION"
SIGNAL_STATUS_SIGNAL_CONFIRMED = "SIGNAL CONFIRMED"
SIGNAL_STATUS_ENTRY_REACHED = "ENTRY REACHED"
SIGNAL_STATUS_ENTRY_MISSED = "ENTRY MISSED"
SIGNAL_STATUS_ACTIVE_TRADE = "ACTIVE TRADE"
SIGNAL_STATUS_TP_HIT = "TP HIT"
SIGNAL_STATUS_SL_HIT = "SL HIT"
SIGNAL_STATUS_CANCELLED = "CANCELLED"
SIGNAL_STATUS_EXPIRED = "EXPIRED"

# Pembedaan eksplisit invalidasi setup aktif (live Bid/Ask tembus SL) vs struktur zona (closed candle TF asal)
INVALIDATION_SETUP_SL_BREACHED = "SETUP_INVALID_SL_BREACHED"
INVALIDATION_ZONE_STRUCTURAL = "ZONE_STRUCTURALLY_INVALID"

CANONICAL_SIGNAL_STATUSES = frozenset({
    SIGNAL_STATUS_NO_SETUP,
    SIGNAL_STATUS_SETUP_DETECTED,
    SIGNAL_STATUS_WAITING_PULLBACK,
    SIGNAL_STATUS_WAITING_CONFIRMATION,
    SIGNAL_STATUS_SIGNAL_CONFIRMED,
    SIGNAL_STATUS_ENTRY_REACHED,
    SIGNAL_STATUS_ENTRY_MISSED,
    SIGNAL_STATUS_ACTIVE_TRADE,
    SIGNAL_STATUS_TP_HIT,
    SIGNAL_STATUS_SL_HIT,
    SIGNAL_STATUS_CANCELLED,
    SIGNAL_STATUS_EXPIRED,
})

TERMINAL_CANONICAL_STATUSES = frozenset({
    SIGNAL_STATUS_ENTRY_MISSED,
    SIGNAL_STATUS_TP_HIT,
    SIGNAL_STATUS_SL_HIT,
    SIGNAL_STATUS_CANCELLED,
    SIGNAL_STATUS_EXPIRED,
})

# Setup Entry Multi-Timeframe Trend Following + SMC Confirmation
SETUP_TYPES = ("TREND_PULLBACK", "TREND_CONTINUATION", "BREAKOUT_RETEST", "REVERSAL")
TF_PULLBACK_MIN_RETRACE = 0.236
TF_PULLBACK_MAX_RETRACE = 0.85
TF_CONTINUATION_MIN_DISPLACEMENT_ATR = 0.80
TF_BREAKOUT_MIN_BODY_ATR = 0.30
TF_OPPOSING_IMPULSE_BLOCK_ATR = 1.50

KOLOM_JURNAL_SINYAL = [
    "id", "waktu_lokal", "arah", "lo", "hi", "lo_asli", "hi_asli", "sl", "tp1", "tp2",
    "harga_buat", "risiko", "ket", "catatan", "model", "spread_buat", "status_hasil", "r_hasil", "diambil"
]

KOLOM_JURNAL_ENTRY = [
    "ticket", "waktu_entry", "waktu_close", "symbol", "arah", "lot",
    "harga_entry", "sl", "tp", "harga_close", "status_posisi", "profit_usd",
    "status_zona", "sumber_zona", "zone_id", "zona_range", "grade_zona", "keterangan",
    "position_id", "order_id", "deal_id", "mt5_status", "setup_match",
    "waktu_order_created", "waktu_executed",
]

# Daftar model Gemini yang dicoba berurutan saat server padat / kuota habis
MODELS_TO_TRY = [
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.6-flash",
    "gemini-3.8-flash",
    "gemini-3.7-flash",
]

# Menyimpan model yang terbukti aktif dan paling cepat di sesi ini, serta membuang model yang 404 / kuota habis
MODEL_AKTIF_TERCEPAT = None
MODEL_404_BLACKLIST = set()
MODEL_COOLDOWN = {}  # model_name -> timestamp bebas cooldown (untuk 429 / 503)


# =============================================================================
# KONFIGURASI XAUUSD SMC ADAPTIVE AUTO TRADING SYSTEM (SINGLE SOURCE OF TRUTH)
# =============================================================================
AUTO_TRADING = _env_cfg("AUTO_TRADING", False, bool)
TRADING_MODE = _env_cfg("TRADING_MODE", "SCANNER_ONLY", str).upper()   # SCANNER_ONLY | DRY_RUN | SEMI_AUTO | FULL_AUTO
ENTRY_MODE = _env_cfg("ENTRY_MODE", "ADAPTIVE", str).upper()           # ADAPTIVE | EARLY_PENDING | CONFIRMED

ALLOW_REAL_ACCOUNT = _env_cfg("ALLOW_REAL_ACCOUNT", True, bool)
DEMO_ONLY = _env_cfg("DEMO_ONLY", False, bool)
REQUIRE_REAL_ACCOUNT_CONFIRMATION = _env_cfg("REQUIRE_REAL_ACCOUNT_CONFIRMATION", True, bool)
AUTO_RESUME_REAL_FULL_AUTO = _env_cfg("AUTO_RESUME_REAL_FULL_AUTO", False, bool)

LOT_MANAGEMENT = _env_cfg("LOT_MANAGEMENT", "DYNAMIC_ZONE_QUALITY", str)

LOT_STANDARD = _env_cfg("LOT_STANDARD", 0.01, float)
LOT_MEDIUM = _env_cfg("LOT_MEDIUM", 0.02, float)
LOT_STRONG = _env_cfg("LOT_STRONG", 0.05, float)
MAX_LOT = _env_cfg("MAX_LOT", 0.05, float)

MIN_ZONE_SCORE = _env_cfg("MIN_ZONE_SCORE", 70.0, float)
EARLY_PENDING_MIN_SCORE = _env_cfg("EARLY_PENDING_MIN_SCORE", 75.0, float)
MEDIUM_ZONE_SCORE = _env_cfg("MEDIUM_ZONE_SCORE", 80.0, float)
STRONG_ZONE_SCORE = _env_cfg("STRONG_ZONE_SCORE", 90.0, float)

EARLY_PENDING_ENABLED = _env_cfg("EARLY_PENDING_ENABLED", True, bool)
ADAPTIVE_ENTRY_ENABLED = _env_cfg("ADAPTIVE_ENTRY_ENABLED", True, bool)
CONFIRMED_ENTRY_ENABLED = _env_cfg("CONFIRMED_ENTRY_ENABLED", True, bool)

EARLY_PENDING_MAX_LOT = _env_cfg("EARLY_PENDING_MAX_LOT", 0.01, float)
LTF_MAX_LOT = _env_cfg("LTF_MAX_LOT", 0.02, float)
HTF_STRONG_MAX_LOT = _env_cfg("HTF_STRONG_MAX_LOT", 0.05, float)
PENDING_LIMIT_ZONE_DEPTH = _env_cfg("PENDING_LIMIT_ZONE_DEPTH", 0.50, float)  # Fallback default 50%
LIMIT_DEPTH_HIGH_RISK = _env_cfg("LIMIT_DEPTH_HIGH_RISK", 0.50, float)        # Zona HIGH_RISK atau Grade < B (C/D) -> 50% tengah zona
LIMIT_DEPTH_GRADE_B = _env_cfg("LIMIT_DEPTH_GRADE_B", 0.25, float)            # Zona Grade B (non-HIGH_RISK) -> 25% masuk zona
LIMIT_DEPTH_GRADE_A = _env_cfg("LIMIT_DEPTH_GRADE_A", 0.25, float)            # Zona Grade A/A+ (non-HIGH_RISK) -> 25% masuk zona


def resolve_zone_limit_depth(
    zone=None,
    grade: str = None,
    score: float = None,
    risk_tier: str = None,
    cfg: dict = None,
) -> float:
    """
    Tentukan kedalaman harga entry LIMIT di dalam zona secara adaptif berdasarkan kualitas & risiko zona:
    - Zona HIGH_RISK atau Grade rendah (< B, mis. C/D atau score < 70) -> 50% (tengah zona, LIMIT_DEPTH_HIGH_RISK).
    - Zona Grade B (bukan HIGH_RISK) -> 25% masuk ke dalam zona (LIMIT_DEPTH_GRADE_B).
    - Zona Grade A / A+ (bukan HIGH_RISK) -> 25% masuk ke dalam zona (LIMIT_DEPTH_GRADE_A).
    Tidak pernah 0% (awal zona banget).
    """
    def _g(obj, key, default=None):
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    c = cfg if isinstance(cfg, dict) else {}
    d_high = float(c.get("LIMIT_DEPTH_HIGH_RISK", _env_cfg("LIMIT_DEPTH_HIGH_RISK", 0.50, float)))
    d_b = float(c.get("LIMIT_DEPTH_GRADE_B", _env_cfg("LIMIT_DEPTH_GRADE_B", 0.25, float)))
    d_a = float(c.get("LIMIT_DEPTH_GRADE_A", _env_cfg("LIMIT_DEPTH_GRADE_A", 0.25, float)))
    d_def = float(c.get("PENDING_LIMIT_ZONE_DEPTH", _env_cfg("PENDING_LIMIT_ZONE_DEPTH", 0.50, float)))

    rt = str(risk_tier if risk_tier is not None else (_g(zone, "risk_tier", "") or "")).strip().upper().replace(" ", "_")
    grd = str(grade if grade is not None else (_g(zone, "grade", "") or "")).strip().upper()
    sc_raw = score if score is not None else _g(zone, "score", None)
    try:
        sc_val = float(sc_raw) if sc_raw is not None and sc_raw != "" else None
    except (TypeError, ValueError):
        sc_val = None

    if "HIGH_RISK" in rt or rt == "HIGH":
        chosen = d_high
    elif (grd and not grd.startswith(("A", "B")) and grd not in ("N/A", "NONE", "UNKNOWN")) or (not grd and sc_val is not None and sc_val < 70.0):
        chosen = d_high
    elif grd.startswith("B") or (not grd and sc_val is not None and 70.0 <= sc_val < 75.0):
        chosen = d_b
    elif grd.startswith("A") or (not grd and sc_val is not None and sc_val >= 75.0):
        chosen = d_a
    else:
        chosen = d_def

    return min(0.70, max(0.15, chosen))


def compute_zone_limit_entry_price(
    lo: float,
    hi: float,
    demand: bool,
    depth: float = None,
    digits: int = 2,
    zone=None,
    grade: str = None,
    score: float = None,
    risk_tier: str = None,
    cfg: dict = None,
) -> float:
    """
    Hitung harga entry untuk order LIMIT (BUY LIMIT / SELL LIMIT) di dalam zona,
    BUKAN di tepi awal zona (proximal edge 0%).
    - Jika zone/grade/score/risk_tier diberikan (dan depth=None): otomatis 50% untuk HIGH_RISK / Grade < B,
      dan 25% untuk Grade B / A.
    - Di-clamp aman sehingga tidak pernah jatuh di awal zona (0%) dan tidak terlalu mepet SL.
    """
    lo_f = float(lo or 0.0)
    hi_f = float(hi or 0.0)
    if lo_f <= 0.0 or hi_f <= lo_f:
        return round(hi_f if demand else lo_f, digits)
    if depth is not None:
        raw_d = float(depth)
        eff_d = min(0.70, raw_d if raw_d >= 0.15 else 0.25)
    elif zone is not None or grade is not None or score is not None or risk_tier is not None:
        eff_d = resolve_zone_limit_depth(zone=zone, grade=grade, score=score, risk_tier=risk_tier, cfg=cfg)
    else:
        raw_d = float((cfg or {}).get("PENDING_LIMIT_ZONE_DEPTH", _env_cfg("PENDING_LIMIT_ZONE_DEPTH", 0.50, float)))
        eff_d = min(0.70, raw_d if raw_d >= 0.15 else 0.25)
    width = hi_f - lo_f
    px = (hi_f - eff_d * width) if demand else (lo_f + eff_d * width)
    return round(px, digits)

MIN_RR = _env_cfg("MIN_RR", 1.0, float)
TP1_RR = _env_cfg("TP1_RR", 1.0, float)
TP1_MODE = _env_cfg("TP1_MODE", "FIXED_RR", str)  # "FIXED_RR" (1:1 / user-defined) atau "STRUCTURAL"
FORCE_FIXED_TP1 = _env_cfg("FORCE_FIXED_TP1", True, bool)  # True = paksa TP1 = Entry +/- (TP1_RR * jarak SL struktural)
REAL_LOT_001_FIXED_TP_ENABLED = _env_cfg("REAL_LOT_001_FIXED_TP_ENABLED", True, bool)
REAL_LOT_001_FIXED_TP_PIPS = _env_cfg("REAL_LOT_001_FIXED_TP_PIPS", 110.0, float)
GRADE_A_WAIT_CONFIRMATION_ENABLED = _env_cfg("GRADE_A_WAIT_CONFIRMATION_ENABLED", False, bool)
GRADE_A_MIN_SCORE = _env_cfg("GRADE_A_MIN_SCORE", 75.0, float)
AUTO_CANCEL_GRADE_A_PENDING_LIMIT = _env_cfg("AUTO_CANCEL_GRADE_A_PENDING_LIMIT", False, bool)
SCALE_IN_80PLUS_ENABLED = _env_cfg("SCALE_IN_80PLUS_ENABLED", True, bool)
SCALE_IN_MIN_SCORE = _env_cfg("SCALE_IN_MIN_SCORE", 80.0, float)


def is_scale_in_80plus_zone(
    zone=None,
    score: float = None,
    cfg: dict = None,
) -> bool:
    """
    Cek apakah zona memenuhi syarat 2-Tahap (Scout Pending Limit 0.01 lot -> setelah terjemput,
    tunggu konfirmasi M5 valid untuk Scale-In Market Order +0.02 / +0.03 lot maksimal 1x per zona).
    """
    c = cfg if isinstance(cfg, dict) else {}
    if not bool(c.get("SCALE_IN_80PLUS_ENABLED", _env_cfg("SCALE_IN_80PLUS_ENABLED", True, bool))):
        return False
    min_sc = float(c.get("SCALE_IN_MIN_SCORE", _env_cfg("SCALE_IN_MIN_SCORE", 80.0, float)))
    if score is not None:
        sc_raw = score
    elif isinstance(zone, dict):
        sc_raw = zone.get("score")
    else:
        sc_raw = getattr(zone, "score", None)
    try:
        sc_val = float(sc_raw) if sc_raw is not None and sc_raw != "" else None
    except (TypeError, ValueError):
        sc_val = None
    return bool(sc_val is not None and sc_val >= min_sc - 1e-9)


def is_grade_a_wait_confirm_zone(
    zone=None,
    grade: str = None,
    score: float = None,
    cfg: dict = None,
) -> bool:
    """
    Cek apakah zona termasuk kategori Grade A / A+ (atau score >= MEDIUM_ZONE_SCORE 80.0 /
    SMC Grade A >= GRADE_A_MIN_SCORE 75.0) yang TIDAK dipasang Order Limit 0.01 lot,
    melainkan menunggu harga menjemput zona + konfirmasi M5 close untuk eksekusi 0.02 / 0.03 lot.
    Zona yang secara eksplisit berlabel Grade B/C/D dengan skor < 80 (mis. SNR Grade B 70-79.99)
    tetap mengembalikan False sehingga tetap menggunakan jalur Order Limit 0.01 lot.
    """
    def _g(obj, key, default=None):
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    c = cfg if isinstance(cfg, dict) else {}
    if not bool(c.get("GRADE_A_WAIT_CONFIRMATION_ENABLED", _env_cfg("GRADE_A_WAIT_CONFIRMATION_ENABLED", True, bool))):
        return False
    med_score = float(c.get("MEDIUM_ZONE_SCORE", _env_cfg("MEDIUM_ZONE_SCORE", 80.0, float)))
    min_a = float(c.get("GRADE_A_MIN_SCORE", _env_cfg("GRADE_A_MIN_SCORE", 75.0, float)))
    grd = str(grade if grade is not None else (_g(zone, "grade", "") or "")).strip().upper()
    sc_raw = score if score is not None else _g(zone, "score", None)
    try:
        sc_val = float(sc_raw) if sc_raw is not None and sc_raw != "" else None
    except (TypeError, ValueError):
        sc_val = None

    if sc_val is not None and sc_val >= med_score - 1e-9:
        return True
    if grd and grd not in ("N/A", "NONE", "UNKNOWN", "-"):
        return bool(grd.startswith("A") and (sc_val is None or sc_val >= min_a - 1e-9))
    return bool(sc_val is not None and sc_val >= min_a - 1e-9)


def resolve_real_lot_001_fixed_tp_distance(cfg: dict = None) -> float:
    """
    Hitung jarak TP tetap dalam USD/oz untuk aturan lot 0.01 pada akun REAL (FULL_AUTO).
    Konvensi XAUUSD sistem: 1 pip = PIP_SIZE_PRICE (bila > 0) atau default 0.10 USD/oz
    (sehingga 110 pips = 11.00 USD).
    """
    c = cfg if isinstance(cfg, dict) else {}
    enabled = bool(c.get("REAL_LOT_001_FIXED_TP_ENABLED", _env_cfg("REAL_LOT_001_FIXED_TP_ENABLED", True, bool)))
    if not enabled:
        return 0.0
    pips = float(c.get("REAL_LOT_001_FIXED_TP_PIPS", _env_cfg("REAL_LOT_001_FIXED_TP_PIPS", 110.0, float)) or 0.0)
    if pips <= 0.0:
        return 0.0
    pip_sz = float(PIP_SIZE_PRICE) if (PIP_SIZE_PRICE is not None and float(PIP_SIZE_PRICE) > 0.0) else 0.10
    return round(pips * pip_sz, 4)


def should_use_real_lot_001_fixed_tp(
    account_mode: str,
    trading_mode: str,
    lot: float,
    cfg: dict = None,
) -> bool:
    """
    Cek apakah aturan TP tetap (default 110 pips / 11.00 USD) berlaku:
    - Hanya saat akun REAL (`account_mode == 'REAL'`) dalam mode `FULL_AUTO` (`auto real`).
    - Hanya saat `final_lot` bernilai 0.01 lot.
    """
    if str(account_mode or "").strip().upper() != "REAL":
        return False
    if str(trading_mode or "").strip().upper() != "FULL_AUTO":
        return False
    try:
        lot_f = round(float(lot or 0.0), 2)
    except (TypeError, ValueError):
        return False
    if abs(lot_f - 0.01) > 1e-6:
        return False
    return resolve_real_lot_001_fixed_tp_distance(cfg) > 0.0


def compute_fixed_tp_price(
    entry_price: float,
    direction: str,
    tp_distance_usd: float,
    digits: int = 2,
) -> float:
    """Hitung harga TP dari Entry +/- jarak tetap USD/oz sesuai arah BUY/SELL."""
    ep = float(entry_price or 0.0)
    dist = float(tp_distance_usd or 0.0)
    if ep <= 0.0 or dist <= 0.0:
        return 0.0
    is_buy = str(direction or "").strip().upper() == "BUY"
    return round((ep + dist) if is_buy else (ep - dist), digits)
AUTO_SL_JARAK_MAKS = _env_cfg("AUTO_SL_JARAK_MAKS", SL_JARAK_MAKS, float)
CONFLICT_MIDZONE_LIMIT_ENABLED = _env_cfg("CONFLICT_MIDZONE_LIMIT_ENABLED", True, bool)
CONFLICT_MIDZONE_MIN_SCORE = _env_cfg("CONFLICT_MIDZONE_MIN_SCORE", 70.0, float)
CONFLICT_MIDZONE_MAX_WIDTH = _env_cfg("CONFLICT_MIDZONE_MAX_WIDTH", 16.0, float)
CONFLICT_MIDZONE_SL_BUFFER = _env_cfg("CONFLICT_MIDZONE_SL_BUFFER", 0.60, float)
MIN_ENTRY_TRIGGERS = _env_cfg("MIN_ENTRY_TRIGGERS", 1, int)

REQUIRE_ALL_TRIGGERS = _env_cfg("REQUIRE_ALL_TRIGGERS", False, bool)
ALLOW_NEUTRAL_H1 = _env_cfg("ALLOW_NEUTRAL_H1", True, bool)
ALLOW_VALID_REJECTION = _env_cfg("ALLOW_VALID_REJECTION", True, bool)
ALLOW_ENTRY_WITHOUT_NEW_FVG = _env_cfg("ALLOW_ENTRY_WITHOUT_NEW_FVG", True, bool)

RISK_EARLY_PENDING_PERCENT = _env_cfg("RISK_EARLY_PENDING_PERCENT", 0.25, float)
RISK_ADAPTIVE_PERCENT = _env_cfg("RISK_ADAPTIVE_PERCENT", 0.5, float)
RISK_CONFIRMED_PERCENT = _env_cfg("RISK_CONFIRMED_PERCENT", 0.5, float)
MAX_RISK_PER_TRADE_PERCENT = _env_cfg("MAX_RISK_PER_TRADE_PERCENT", 1.0, float)
MAX_TOTAL_EXPOSURE_PERCENT = _env_cfg("MAX_TOTAL_EXPOSURE_PERCENT", 1.0, float)

MAX_DAILY_LOSS_PERCENT = _env_cfg("MAX_DAILY_LOSS_PERCENT", 2.0, float)
MAX_ACTIVE_POSITIONS = _env_cfg("MAX_ACTIVE_POSITIONS", 1, int)
MAX_ACTIVE_PENDING_ORDERS = _env_cfg("MAX_ACTIVE_PENDING_ORDERS", 0, int)  # 0 = unlimited pending per zona valid (anti-duplicate per zona tetap aktif)
MAX_TRADES_PER_DAY = _env_cfg("MAX_TRADES_PER_DAY", 3, int)

REQUIRE_STOP_LOSS = _env_cfg("REQUIRE_STOP_LOSS", True, bool)
REQUIRE_TAKE_PROFIT = _env_cfg("REQUIRE_TAKE_PROFIT", True, bool)
JOURNAL_LOG_BLOCKED_EVENTS = _env_cfg("JOURNAL_LOG_BLOCKED_EVENTS", False, bool)  # Jika False: hanya catat yang penting (ORDER, CLOSE, MISSED)

PREVENT_DUPLICATE_ORDER = _env_cfg("PREVENT_DUPLICATE_ORDER", True, bool)
AUTO_CANCEL_INVALID_ZONE = _env_cfg("AUTO_CANCEL_INVALID_ZONE", True, bool)
AUTO_CANCEL_EXPIRED_SETUP = _env_cfg("AUTO_CANCEL_EXPIRED_SETUP", True, bool)
AUTO_CANCEL_FAR_PENDING = _env_cfg("AUTO_CANCEL_FAR_PENDING", True, bool)
PENDING_MAX_DISTANCE_USD = _env_cfg("PENDING_MAX_DISTANCE_USD", 40.0, float)
PENDING_CANCEL_HYSTERESIS_USD = _env_cfg("PENDING_CANCEL_HYSTERESIS_USD", 3.0, float)
ALLOW_PENDING_TO_MARKET_SWITCH = _env_cfg("ALLOW_PENDING_TO_MARKET_SWITCH", False, bool)

NEWS_FILTER = _env_cfg("NEWS_FILTER", True, bool)
BLOCK_ON_STALE_NEWS_DATA = _env_cfg("BLOCK_ON_STALE_NEWS_DATA", True, bool)
NEWS_STALE_MAX_SECONDS = _env_cfg("NEWS_STALE_MAX_SECONDS", 21600.0, float)
CHASE_PRICE = _env_cfg("CHASE_PRICE", False, bool)
EXECUTION_CHASE_MAX_ATR_M5 = _env_cfg("EXECUTION_CHASE_MAX_ATR_M5", 0.35, float)
EXECUTION_CHASE_MAX_POINTS = _env_cfg("EXECUTION_CHASE_MAX_POINTS", 1.5, float)

MARTINGALE = False
GRID_TRADING = False
AVERAGING_LOSS = False

BREAK_EVEN_ENABLED = _env_cfg("BREAK_EVEN_ENABLED", False, bool)
BREAK_EVEN_TRIGGER_RR = _env_cfg("BREAK_EVEN_TRIGGER_RR", 1.0, float)
BREAK_EVEN_LOCK_POINTS = _env_cfg("BREAK_EVEN_LOCK_POINTS", 0.30, float)
TRAILING_STOP_ENABLED = _env_cfg("TRAILING_STOP_ENABLED", False, bool)
TRAILING_STOP_ACTIVATE_RR = _env_cfg("TRAILING_STOP_ACTIVATE_RR", 1.5, float)
TRAILING_STOP_DISTANCE_POINTS = _env_cfg("TRAILING_STOP_DISTANCE_POINTS", 2.5, float)
PARTIAL_CLOSE_ENABLED = _env_cfg("PARTIAL_CLOSE_ENABLED", False, bool)
BE_SINGLE_LOT_ENABLED = _env_cfg("BE_SINGLE_LOT_ENABLED", True, bool)
BE_SINGLE_LOT_TRIGGER_POINTS = _env_cfg("BE_SINGLE_LOT_TRIGGER_POINTS", 8.0, float)  # 8.0 USD = 80 pips pada XAUUSD untuk lot 0.01 (tidak bisa partial)

MONITOR_INTERVAL_SECONDS = _env_cfg("MONITOR_INTERVAL_SECONDS", 1.0, float)
SCANNER_MAGIC_NUMBER = _env_cfg("SCANNER_MAGIC_NUMBER", 20260991, int)
ORDER_DEVIATION_POINTS = _env_cfg("ORDER_DEVIATION_POINTS", 30, int)
ORDER_SEND_MAX_RETRIES = _env_cfg("ORDER_SEND_MAX_RETRIES", 2, int)
SLIPPAGE_BUFFER_PRICE = _env_cfg("SLIPPAGE_BUFFER_PRICE", 0.20, float)
COMMISSION_PER_LOT_USD_EST = _env_cfg("COMMISSION_PER_LOT_USD_EST", 7.0, float)
EMERGENCY_CLOSE_ON_MISSING_SL = _env_cfg("EMERGENCY_CLOSE_ON_MISSING_SL", True, bool)
DAILY_RESET_TIMEZONE = _env_cfg("DAILY_RESET_TIMEZONE", "WIB", str).upper()

# Threshold objektif trigger M5 Adaptive Entry
REJECTION_MIN_WICK_RATIO = _env_cfg("REJECTION_MIN_WICK_RATIO", 0.45, float)
REJECTION_MIN_WICK_ATR = _env_cfg("REJECTION_MIN_WICK_ATR", 0.20, float)
REJECTION_MIN_RANGE_ATR = _env_cfg("REJECTION_MIN_RANGE_ATR", 0.30, float)
REJECTION_CLOSE_LOCATION_MAX = _env_cfg("REJECTION_CLOSE_LOCATION_MAX", 0.40, float)
ADAPTIVE_DISPLACEMENT_MIN_BODY_ATR = _env_cfg("ADAPTIVE_DISPLACEMENT_MIN_BODY_ATR", 0.60, float)
ADAPTIVE_DISPLACEMENT_MIN_BODY_RATIO = _env_cfg("ADAPTIVE_DISPLACEMENT_MIN_BODY_RATIO", 0.55, float)
ADAPTIVE_TRIGGER_MAX_AGE_BARS = _env_cfg("ADAPTIVE_TRIGGER_MAX_AGE_BARS", 8, int)
STRONG_ZONE_MIN_DISPLACEMENT_ATR = _env_cfg("STRONG_ZONE_MIN_DISPLACEMENT_ATR", 1.5, float)

FILE_STATE_AUTO_EXECUTION = os.path.join(_DIR_SCRIPT, "state_auto_execution.json")
FILE_JURNAL_AUTO_EXECUTION = os.path.join(_DIR_SCRIPT, "jurnal_eksekusi_auto.csv")


def _apply_loose_overrides(conf: dict) -> dict:
    """
    LOOSE MODE (data-collection / eksperimen, default OFF via AUTO_LOOSE_MODE=false).
    Melonggarkan: jumlah order, skor/grade zona, konflik HTF, trigger, news, batas risiko.
    TETAP berlaku: SL & TP wajib, SL di luar structural invalidation, RR tetap, lot <= 0.01 untuk grade < B,
    anti-duplikat per zona, zona INVALIDATED/EXPIRED/BROKEN, market/MT5/data/spread, order_check & verifikasi broker.
    """
    if not _env_cfg("AUTO_LOOSE_MODE", False, bool):
        conf["AUTO_LOOSE_MODE"] = False
        return conf
    rr = _env_cfg("LOOSE_RR", 2.0, float)
    max_active = _env_cfg("LOOSE_MAX_ACTIVE", 5, int)
    conf.update({
        "AUTO_LOOSE_MODE": True,
        "LOOSE_LOT": _env_cfg("LOOSE_LOT", 0.01, float),
        "MIN_ZONE_SCORE": 0.0,
        "EARLY_PENDING_MIN_SCORE": 0.0,
        "MIN_RR": rr,
        "TP1_RR": rr,
        "TP1_MODE": "FIXED_RR",
        "FORCE_FIXED_TP1": True,
        "MAX_ACTIVE_POSITIONS": max_active,
        "MAX_ACTIVE_PENDING_ORDERS": max_active,
        "MAX_TRADES_PER_DAY": _env_cfg("LOOSE_MAX_TRADES_PER_DAY", 999, int),
        "MAX_TOTAL_EXPOSURE_PERCENT": _env_cfg("LOOSE_MAX_EXPOSURE_PERCENT", 100.0, float),
        "MAX_DAILY_LOSS_PERCENT": _env_cfg("LOOSE_MAX_DAILY_LOSS_PERCENT", 100.0, float),
        "MAX_RISK_PER_TRADE_PERCENT": 100.0,
        "RISK_EARLY_PENDING_PERCENT": 100.0,
        "RISK_ADAPTIVE_PERCENT": 100.0,
        "RISK_CONFIRMED_PERCENT": 100.0,
        "NEWS_FILTER": False,
        "ALLOW_NEUTRAL_H1": True,
        "CONFLICT_MIDZONE_LIMIT_ENABLED": False,
    })
    return conf


def get_auto_trading_config() -> dict:
    """Kembalikan snapshot dictionary konfigurasi Auto Trading terpusat (membaca .env terbaru secara live)."""
    return _apply_loose_overrides(_build_auto_trading_config())


def _build_auto_trading_config() -> dict:
    return {
        "SYMBOL": SYMBOL,
        "AUTO_TRADING": _env_cfg("AUTO_TRADING", False, bool),
        "TRADING_MODE": _env_cfg("TRADING_MODE", "SCANNER_ONLY", str).upper(),
        "ENTRY_MODE": _env_cfg("ENTRY_MODE", "ADAPTIVE", str).upper(),
        "ALLOW_REAL_ACCOUNT": _env_cfg("ALLOW_REAL_ACCOUNT", True, bool),
        "DEMO_ONLY": _env_cfg("DEMO_ONLY", False, bool),
        "REQUIRE_REAL_ACCOUNT_CONFIRMATION": _env_cfg("REQUIRE_REAL_ACCOUNT_CONFIRMATION", True, bool),
        "AUTO_RESUME_REAL_FULL_AUTO": _env_cfg("AUTO_RESUME_REAL_FULL_AUTO", False, bool),
        "LOT_MANAGEMENT": _env_cfg("LOT_MANAGEMENT", "DYNAMIC_ZONE_QUALITY", str),
        "LOT_STANDARD": _env_cfg("LOT_STANDARD", 0.01, float),
        "LOT_MEDIUM": _env_cfg("LOT_MEDIUM", 0.02, float),
        "LOT_STRONG": _env_cfg("LOT_STRONG", 0.05, float),
        "MAX_LOT": _env_cfg("MAX_LOT", 0.05, float),
        "MIN_ZONE_SCORE": _env_cfg("MIN_ZONE_SCORE", 70.0, float),
        "EARLY_PENDING_MIN_SCORE": _env_cfg("EARLY_PENDING_MIN_SCORE", 75.0, float),
        "MEDIUM_ZONE_SCORE": _env_cfg("MEDIUM_ZONE_SCORE", 80.0, float),
        "STRONG_ZONE_SCORE": _env_cfg("STRONG_ZONE_SCORE", 90.0, float),
        "EARLY_PENDING_ENABLED": _env_cfg("EARLY_PENDING_ENABLED", True, bool),
        "ADAPTIVE_ENTRY_ENABLED": _env_cfg("ADAPTIVE_ENTRY_ENABLED", True, bool),
        "CONFIRMED_ENTRY_ENABLED": _env_cfg("CONFIRMED_ENTRY_ENABLED", True, bool),
        "EARLY_PENDING_MAX_LOT": _env_cfg("EARLY_PENDING_MAX_LOT", 0.01, float),
        "PENDING_LIMIT_ZONE_DEPTH": _env_cfg("PENDING_LIMIT_ZONE_DEPTH", 0.50, float),
        "LIMIT_DEPTH_HIGH_RISK": _env_cfg("LIMIT_DEPTH_HIGH_RISK", 0.50, float),
        "LIMIT_DEPTH_GRADE_B": _env_cfg("LIMIT_DEPTH_GRADE_B", 0.25, float),
        "LIMIT_DEPTH_GRADE_A": _env_cfg("LIMIT_DEPTH_GRADE_A", 0.25, float),
        "LTF_MAX_LOT": _env_cfg("LTF_MAX_LOT", 0.02, float),
        "HTF_STRONG_MAX_LOT": _env_cfg("HTF_STRONG_MAX_LOT", 0.05, float),
        "MIN_RR": _env_cfg("MIN_RR", 1.0, float),
        "TP1_RR": _env_cfg("TP1_RR", 1.0, float),
        "TP1_MODE": _env_cfg("TP1_MODE", "FIXED_RR", str),
        "FORCE_FIXED_TP1": _env_cfg("FORCE_FIXED_TP1", True, bool),
        "REAL_LOT_001_FIXED_TP_ENABLED": _env_cfg("REAL_LOT_001_FIXED_TP_ENABLED", True, bool),
        "REAL_LOT_001_FIXED_TP_PIPS": _env_cfg("REAL_LOT_001_FIXED_TP_PIPS", 110.0, float),
        "GRADE_A_WAIT_CONFIRMATION_ENABLED": _env_cfg("GRADE_A_WAIT_CONFIRMATION_ENABLED", False, bool),
        "GRADE_A_MIN_SCORE": _env_cfg("GRADE_A_MIN_SCORE", 75.0, float),
        "AUTO_CANCEL_GRADE_A_PENDING_LIMIT": _env_cfg("AUTO_CANCEL_GRADE_A_PENDING_LIMIT", False, bool),
        "SCALE_IN_80PLUS_ENABLED": _env_cfg("SCALE_IN_80PLUS_ENABLED", True, bool),
        "SCALE_IN_MIN_SCORE": _env_cfg("SCALE_IN_MIN_SCORE", 80.0, float),
        "MIN_ENTRY_TRIGGERS": _env_cfg("MIN_ENTRY_TRIGGERS", 1, int),
        "REQUIRE_ALL_TRIGGERS": _env_cfg("REQUIRE_ALL_TRIGGERS", False, bool),
        "ALLOW_NEUTRAL_H1": _env_cfg("ALLOW_NEUTRAL_H1", True, bool),
        "ALLOW_VALID_REJECTION": _env_cfg("ALLOW_VALID_REJECTION", True, bool),
        "ALLOW_ENTRY_WITHOUT_NEW_FVG": _env_cfg("ALLOW_ENTRY_WITHOUT_NEW_FVG", True, bool),
        "RISK_EARLY_PENDING_PERCENT": _env_cfg("RISK_EARLY_PENDING_PERCENT", 0.25, float),
        "RISK_ADAPTIVE_PERCENT": _env_cfg("RISK_ADAPTIVE_PERCENT", 0.5, float),
        "RISK_CONFIRMED_PERCENT": _env_cfg("RISK_CONFIRMED_PERCENT", 0.5, float),
        "MAX_RISK_PER_TRADE_PERCENT": _env_cfg("MAX_RISK_PER_TRADE_PERCENT", 1.0, float),
        "MAX_TOTAL_EXPOSURE_PERCENT": _env_cfg("MAX_TOTAL_EXPOSURE_PERCENT", 1.0, float),
        "MAX_DAILY_LOSS_PERCENT": _env_cfg("MAX_DAILY_LOSS_PERCENT", 2.0, float),
        "MAX_ACTIVE_POSITIONS": _env_cfg("MAX_ACTIVE_POSITIONS", 1, int),
        "MAX_ACTIVE_PENDING_ORDERS": _env_cfg("MAX_ACTIVE_PENDING_ORDERS", 0, int),
        "MAX_TRADES_PER_DAY": _env_cfg("MAX_TRADES_PER_DAY", 3, int),
        "REQUIRE_STOP_LOSS": _env_cfg("REQUIRE_STOP_LOSS", True, bool),
        "REQUIRE_TAKE_PROFIT": _env_cfg("REQUIRE_TAKE_PROFIT", True, bool),
        "JOURNAL_LOG_BLOCKED_EVENTS": _env_cfg("JOURNAL_LOG_BLOCKED_EVENTS", False, bool),
        "PREVENT_DUPLICATE_ORDER": _env_cfg("PREVENT_DUPLICATE_ORDER", True, bool),
        "AUTO_CANCEL_INVALID_ZONE": _env_cfg("AUTO_CANCEL_INVALID_ZONE", True, bool),
        "AUTO_CANCEL_EXPIRED_SETUP": _env_cfg("AUTO_CANCEL_EXPIRED_SETUP", True, bool),
        "AUTO_CANCEL_FAR_PENDING": _env_cfg("AUTO_CANCEL_FAR_PENDING", True, bool),
        "PENDING_MAX_DISTANCE_USD": _env_cfg("PENDING_MAX_DISTANCE_USD", 40.0, float),
        "PENDING_CANCEL_HYSTERESIS_USD": _env_cfg("PENDING_CANCEL_HYSTERESIS_USD", 3.0, float),
        "ALLOW_PENDING_TO_MARKET_SWITCH": _env_cfg("ALLOW_PENDING_TO_MARKET_SWITCH", False, bool),
        "NEWS_FILTER": _env_cfg("NEWS_FILTER", True, bool),
        "BLOCK_ON_STALE_NEWS_DATA": _env_cfg("BLOCK_ON_STALE_NEWS_DATA", True, bool),
        "NEWS_STALE_MAX_SECONDS": _env_cfg("NEWS_STALE_MAX_SECONDS", 21600.0, float),
        "CHASE_PRICE": _env_cfg("CHASE_PRICE", False, bool),
        "EXECUTION_CHASE_MAX_ATR_M5": _env_cfg("EXECUTION_CHASE_MAX_ATR_M5", 0.35, float),
        "EXECUTION_CHASE_MAX_POINTS": _env_cfg("EXECUTION_CHASE_MAX_POINTS", 1.5, float),
        "MARTINGALE": False,
        "GRID_TRADING": False,
        "AVERAGING_LOSS": False,
        "BREAK_EVEN_ENABLED": _env_cfg("BREAK_EVEN_ENABLED", False, bool),
        "BREAK_EVEN_TRIGGER_RR": _env_cfg("BREAK_EVEN_TRIGGER_RR", 1.0, float),
        "BREAK_EVEN_LOCK_POINTS": _env_cfg("BREAK_EVEN_LOCK_POINTS", 0.30, float),
        "TRAILING_STOP_ENABLED": _env_cfg("TRAILING_STOP_ENABLED", False, bool),
        "TRAILING_STOP_ACTIVATE_RR": _env_cfg("TRAILING_STOP_ACTIVATE_RR", 1.5, float),
        "TRAILING_STOP_DISTANCE_POINTS": _env_cfg("TRAILING_STOP_DISTANCE_POINTS", 2.5, float),
        "PARTIAL_CLOSE_ENABLED": _env_cfg("PARTIAL_CLOSE_ENABLED", False, bool),
        "PARTIAL_STEP_POINTS": _env_cfg("PARTIAL_STEP_POINTS", 5.0, float),
        "PARTIAL_CLOSE_LOT_STEP": _env_cfg("PARTIAL_CLOSE_LOT_STEP", 0.01, float),
        "PARTIAL_MIN_REMAINING_LOT": _env_cfg("PARTIAL_MIN_REMAINING_LOT", 0.01, float),
        "BE_AFTER_PARTIAL_ENABLED": _env_cfg("BE_AFTER_PARTIAL_ENABLED", True, bool),
        "BE_PLUS_LOCK_POINTS": _env_cfg("BE_PLUS_LOCK_POINTS", 0.30, float),
        "BE_SINGLE_LOT_ENABLED": _env_cfg("BE_SINGLE_LOT_ENABLED", True, bool),
        "BE_SINGLE_LOT_TRIGGER_POINTS": _env_cfg("BE_SINGLE_LOT_TRIGGER_POINTS", 8.0, float),
        "MONITOR_INTERVAL_SECONDS": _env_cfg("MONITOR_INTERVAL_SECONDS", 1.0, float),
        "SNR_ZONE_MIN_STABLE_SECONDS": _env_cfg("SNR_ZONE_MIN_STABLE_SECONDS", 120.0, float),
        "SCANNER_MAGIC_NUMBER": _env_cfg("SCANNER_MAGIC_NUMBER", 20260991, int),
        "ORDER_DEVIATION_POINTS": _env_cfg("ORDER_DEVIATION_POINTS", 30, int),
        "ORDER_SEND_MAX_RETRIES": _env_cfg("ORDER_SEND_MAX_RETRIES", 2, int),
        "SLIPPAGE_BUFFER_PRICE": _env_cfg("SLIPPAGE_BUFFER_PRICE", 0.20, float),
        "COMMISSION_PER_LOT_USD_EST": _env_cfg("COMMISSION_PER_LOT_USD_EST", 7.0, float),
        "EMERGENCY_CLOSE_ON_MISSING_SL": _env_cfg("EMERGENCY_CLOSE_ON_MISSING_SL", True, bool),
        "DAILY_RESET_TIMEZONE": _env_cfg("DAILY_RESET_TIMEZONE", "WIB", str).upper(),
        "SPREAD_MAKS_POIN": _env_cfg("SPREAD_MAKS_POIN", SPREAD_MAKS_POIN, float),
        "BERITA_BLACKOUT_MENIT": _env_cfg("BERITA_BLACKOUT_MENIT", BERITA_BLACKOUT_MENIT, int),
        "SL_JARAK_MAKS": _env_cfg("AUTO_SL_JARAK_MAKS", _env_cfg("SL_JARAK_MAKS", SL_JARAK_MAKS, float), float),
        "CONFLICT_MIDZONE_LIMIT_ENABLED": _env_cfg("CONFLICT_MIDZONE_LIMIT_ENABLED", CONFLICT_MIDZONE_LIMIT_ENABLED, bool),
        "CONFLICT_MIDZONE_MIN_SCORE": _env_cfg("CONFLICT_MIDZONE_MIN_SCORE", CONFLICT_MIDZONE_MIN_SCORE, float),
        "CONFLICT_MIDZONE_MAX_WIDTH": _env_cfg("CONFLICT_MIDZONE_MAX_WIDTH", CONFLICT_MIDZONE_MAX_WIDTH, float),
        "CONFLICT_MIDZONE_SL_BUFFER": _env_cfg("CONFLICT_MIDZONE_SL_BUFFER", CONFLICT_MIDZONE_SL_BUFFER, float),
        "DEFAULT_PENDING_EXPIRATION_HOURS": DEFAULT_PENDING_EXPIRATION_HOURS,
        "PANTAU_TICK_STALE_SECONDS": PANTAU_TICK_STALE_SECONDS,
        "REJECTION_MIN_WICK_RATIO": _env_cfg("REJECTION_MIN_WICK_RATIO", 0.45, float),
        "REJECTION_MIN_WICK_ATR": _env_cfg("REJECTION_MIN_WICK_ATR", 0.20, float),
        "REJECTION_MIN_RANGE_ATR": _env_cfg("REJECTION_MIN_RANGE_ATR", 0.30, float),
        "REJECTION_CLOSE_LOCATION_MAX": _env_cfg("REJECTION_CLOSE_LOCATION_MAX", 0.40, float),
        "ADAPTIVE_DISPLACEMENT_MIN_BODY_ATR": _env_cfg("ADAPTIVE_DISPLACEMENT_MIN_BODY_ATR", 0.60, float),
        "ADAPTIVE_DISPLACEMENT_MIN_BODY_RATIO": _env_cfg("ADAPTIVE_DISPLACEMENT_MIN_BODY_RATIO", 0.55, float),
        "ADAPTIVE_TRIGGER_MAX_AGE_BARS": _env_cfg("ADAPTIVE_TRIGGER_MAX_AGE_BARS", 8, int),
        "STRONG_ZONE_MIN_DISPLACEMENT_ATR": _env_cfg("STRONG_ZONE_MIN_DISPLACEMENT_ATR", 1.5, float),
    }


