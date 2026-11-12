# -*- coding: utf-8 -*-
"""
modules/config.py
Konfigurasi parameter trading, timeframe, API key, threshold risiko, status sinyal, dan model AI.
"""
import os
import MetaTrader5 as mt5

# Direktori root project
_DIR_SCRIPT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_VERSION = "v2.5"


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

# Top-down: W1/D1/H4 (zona jauh ke kiri) -> H1 -> M30 (konfirmasi) -> M5 (entry presisi)
TIMEFRAMES = {
    "W1": (mt5.TIMEFRAME_W1, 260),    # ~5 tahun
    "D1": (mt5.TIMEFRAME_D1, 1000),   # ~3-4 tahun
    "H4": (mt5.TIMEFRAME_H4, 3000),   # ~1.5-2 tahun, zona jauh ke kiri
    "H1": (mt5.TIMEFRAME_H1, 400),
    "M30": (mt5.TIMEFRAME_M30, 300),
    "M5": (mt5.TIMEFRAME_M5, 300),
}
SWING_N = 3  # sebuah candle = swing jika High/Low-nya ekstrem dibanding N candle kiri & kanan

# Selisih jam waktu server broker terhadap UTC (contoh: 0, 2, 3).
# None = deteksi otomatis dari tick MT5 (hanya akurat saat market buka).
SERVER_UTC_OFFSET_JAM = None

# --- Gerbang sinyal & kontrol risiko: ditegakkan oleh KODE (bukan hanya prompt AI) ---
RR_MIN = 1.5                       # R:R minimal ke TP1
ENTRY_LEBAR_MAKS = 2.5             # lebar area entry maksimal (poin)
ENTRY_JARAK_MAKS = 40.0            # jarak entry dari harga live maksimal (poin)
SL_JARAK_MAKS = 20.0               # jarak SL maksimal dari entry (poin)
SPREAD_MAKS_POIN = 1.0             # spread maksimal (poin) untuk sinyal baru
SPREAD_MAKS_RASIO_ATR = 0.35       # ...dan maksimal sekian x ATR M5 (lantai 0.5 poin)
SINYAL_TTL_BELUM_TERISI_JAM = 6    # sinyal yang belum terjemput melewati batas ini dianggap kedaluwarsa
BERITA_BLACKOUT_MENIT = 15         # tidak menerbitkan sinyal baru +/- sekian menit dari berita High-impact USD
MAKS_RISIKO_TERBUKA_PCT = 3.0      # total risiko posisi + pending (jarak ke SL x lot) terhadap equity
MAKS_RUGI_HARIAN_PCT = 3.0         # batas rugi tertutup hari ini (hari server) terhadap saldo awal hari
LEBAR_CLUSTER_MAKS = 8.0           # lebar maksimal zona Golden Confluence (poin)
FILTER_TREND_AKTIF = True          # True = hanya sinyal searah trend HTF; False = matikan filter
IZINKAN_COUNTER_TREND_MEDIUM = False  # True = counter-trend boleh bila sinyal MEDIUM RISK
FILTER_SESI_AKTIF = True           # True = hanya sinyal di sesi London & New York (UTC); False = semua jam
SESI_DIIZINKAN_UTC = [(7, 16), (12, 21)]  # London (07:00-16:00 UTC) dan New York (12:00-21:00 UTC)
BIAS_MIN_SKOR = 2                  # skor minimal (-5..+5) agar bias dianggap BULLISH/BEARISH
JARAK_CONFLUENCE_MAKS = 100.0      # confluence yang lebih jauh dari ini tidak ditampilkan

# Path file penyimpanan data
FILE_JURNAL_SINYAL = os.path.join(_DIR_SCRIPT, "jurnal_sinyal_xauusd.csv")
FILE_ERROR_LOG = os.path.join(_DIR_SCRIPT, "error_xauusd.log")
FILE_STATE_SINYAL = os.path.join(_DIR_SCRIPT, "state_sinyal_xauusd.json")
FILE_STATE_ZONE_TRACKER = os.path.join(_DIR_SCRIPT, "state_zone_tracker.json")
DEFAULT_PENDING_EXPIRATION_HOURS = 8
ZONE_APPROACH_DISTANCE_POINTS = 5.0

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

KOLOM_JURNAL_SINYAL = [
    "id", "waktu_lokal", "arah", "lo", "hi", "lo_asli", "hi_asli", "sl", "tp1", "tp2",
    "harga_buat", "risiko", "ket", "catatan", "model", "spread_buat", "status_hasil", "r_hasil", "diambil"
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

