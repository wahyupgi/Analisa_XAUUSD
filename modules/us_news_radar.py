# -*- coding: utf-8 -*-
"""
modules/us_news_radar.py — US Fundamental News Radar, Gold Market News Intelligence & Shared Alert Engine (informational only, v2.5).

SUMBER DATA RESMI & LEGAL:
  1. US Economic Calendar:
     - Federal Reserve Board (www.federalreserve.gov): fomccalendars.htm & json/calendar.json
       (TIDAK lagi menggunakan FRED release_id=101 karena berisi seri harian DFEDTARU/DFEDTARL)
     - FRED API (Federal Reserve Bank of St. Louis, perlu FRED_API_KEY):
       Jadwal resmi BLS/BEA/US DOL/US Census + Actual & revisi Previous terverifikasi
       (Forecast konsensus tidak disediakan FRED/Fed -> N/A, pasca-rilis tanpa forecast -> UNCONFIRMED).
  2. Gold Market News (Headline, Publisher, URL, Timestamp, Excerpt saja — tanpa scraping paywall/artikel penuh):
     - RSS/Atom resmi tanpa API key: Federal Reserve (press_all.xml, speeches.xml), BLS (bls_latest.rss),
       BEA (rss.xml), CNBC Finance & Investing RSS, MarketWatch Top Stories RSS, FXStreet News RSS.
     - Market News API opsional (aktif otomatis bila API key tersedia di .env / environment):
       FINNHUB_API_KEY, ALPHAVANTAGE_API_KEY, MARKETAUX_API_KEY, NEWSAPI_KEY.
     - Catatan: Reuters telah menonaktifkan feed RSS publik gratis dan TradingView tidak menyediakan API berita gratis;
       modul ini tidak melakukan scraping ilegal atau bypass paywall.

Modul ini TIDAK mengubah keputusan trading, zona SMC/SNR, risk management, atau order MT5.
Semua error jaringan diisolasi sepenuhnya.
"""
import hashlib
import html as _html
from html.entities import name2codepoint as _HTML_ENTITIES
import json
import math
import os
import re
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

try:
    from zoneinfo import ZoneInfo
    TZ_NY = ZoneInfo("America/New_York")
    TZ_WIB = ZoneInfo("Asia/Jakarta")
except Exception:  # tzdata tidak ada -> radar UNAVAILABLE, scanner tetap jalan
    TZ_NY = TZ_WIB = None

_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_FILE = os.path.join(_DIR, "news_cache.json")
CACHE_SCHEMA_VERSION = 2

FRED = "https://api.stlouisfed.org/fred"
FED_FOMC_CAL_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
FED_JSON_CAL_URL = "https://www.federalreserve.gov/json/calendar.json"
HTTP_HEADERS = {"User-Agent": "XAUUSD-SMC-Scanner/2.5 (EconomicCalendarAuditor)"}

TIMEOUT = 5                 # detik per request
RETRIES = 2                 # percobaan per request (backoff 1.5s, 3s)
REFRESH_DEADLINE = 25       # batas total satu siklus refresh (detik)
TTL_NORMAL = 3600           # detik
TTL_ACTIVE = 60             # detik dekat/sesudah jadwal rilis untuk update actual
MAX_STALE_SECONDS = 6 * 3600  # batas keras; cache lebih tua tidak boleh dipakai sebagai live
FAIL_BACKOFF = 120          # jeda minimal antar siklus refresh yang gagal
WINDOW_PAST_DAYS = 10
WINDOW_FUTURE_DAYS = 35
POST_RELEASE_WINDOW_MIN = 30
DELAYED_AFTER_MIN = 30

# Threshold alert configurable (menit / detik)
ALERT_THRESHOLDS = {
    "upcoming_min": 60,                   # T-60: UPCOMING
    "caution_min": 30,                    # T-30: CAUTION
    "high_alert_min": 15,                 # T-15: HIGH ALERT
    "imminent_min": 5,                    # T-5 : RELEASE IMMINENT
    "post_release_min": 30,               # T+0..T+30: POST-RELEASE VOLATILITY
    "min_poll_interval_sec": 30,          # jangan panggil API setiap tick
    "post_release_poll_interval_sec": 60, # refresh actual/revisi pasca rilis
    "gold_news_poll_interval_sec": 180,   # throttle evaluasi polling headline emas
}

# Konfigurasi Gold Market News (terpisah dari kalender ekonomi; tanpa scraping ilegal)
GOLD_NEWS_TTL = 300                       # 5 menit untuk refresh headline
GOLD_NEWS_MAX_STALE_SECONDS = 6 * 3600    # batas keras stale cache headline
GOLD_NEWS_MAX_AGE_HOURS = 72              # filter artikel lebih lama dari 72 jam
GOLD_NEWS_ALERT_MAX_AGE_HOURS = 6         # alert monitor hanya untuk berita baru (<=6 jam)
GOLD_NEWS_REFRESH_DEADLINE = 18           # batas total satu siklus refresh headline (detik)
GOLD_NEWS_FAIL_BACKOFF = 120              # jeda minimal antar refresh headline yang gagal
GOLD_NEWS_MAX_STORED = 40                 # batas artikel disimpan di cache
GOLD_NEWS_ALERT_MIN_RELEVANCE = 35        # ambang relevansi minimum untuk alert monitor
GOLD_NEWS_MAX_ALERTS_PER_CYCLE = 2        # anti-spam maksimal alert berita per siklus

# Feed RSS/Atom resmi (akses publik legal, headline/metadata/excerpt saja)
# Urutan memprioritaskan sumber emas/FX & Fed/keuangan utama sebelum feed statistik umum.
# Tuple: (provider_id, publisher_default, url, credibility_tier, default_lang)
GOLD_RSS_FEEDS = [
    ("FXSTREET_RSS",    "FXStreet",        "https://www.fxstreet.com/rss",                                  2, "en"),
    ("CNBC_FIN_RSS",    "CNBC",            "https://www.cnbc.com/id/10000664/device/rss/rss.html",          2, "en"),
    ("CNBC_INV_RSS",    "CNBC",            "https://www.cnbc.com/id/15839069/device/rss/rss.html",          2, "en"),
    ("FED_PRESS_RSS",   "Federal Reserve", "https://www.federalreserve.gov/feeds/press_all.xml",            1, "en"),
    ("FED_SPEECH_RSS",  "Federal Reserve", "https://www.federalreserve.gov/feeds/speeches.xml",             1, "en"),
    ("MARKETWATCH_RSS", "MarketWatch",     "https://feeds.marketwatch.com/marketwatch/topstories/",         2, "en"),
    ("BLS_RSS",         "BLS",             "https://www.bls.gov/feed/bls_latest.rss",                       1, "en"),
    ("BEA_RSS",         "BEA",             "https://apps.bea.gov/rss/rss.xml",                              1, "en"),
]
GOLD_PROVIDER_COOLDOWN_SEC = 1800  # nonaktifkan sementara feed gagal permanen/400/404/429 agar tidak mengganggu feed lain
_gold_provider_cooldown = {}       # provider_id -> (until_epoch, reason)

# API berita pasar dokumentasi resmi (opsional via environment variable / .env)
OPTIONAL_NEWS_API_KEYS = (
    ("FINNHUB",      "FINNHUB_API_KEY"),
    ("ALPHAVANTAGE", "ALPHAVANTAGE_API_KEY"),
    ("MARKETAUX",    "MARKETAUX_API_KEY"),
    ("NEWSAPI",      "NEWSAPI_KEY"),
)

# Aturan impact (configurable): event_type/key -> HIGH / MEDIUM / LOW
IMPACT_RULES = {
    "FOMC_STATEMENT": "HIGH",
    "FOMC_RATE_DECISION": "HIGH",
    "FOMC_MINUTES": "HIGH",
    "CPI": "HIGH",
    "CORE_CPI": "HIGH",
    "NFP": "HIGH",
    "UNRATE": "HIGH",
    "CORE_PCE": "HIGH",
    "PCE": "MEDIUM",
    "RETAIL": "MEDIUM",
    "GDP": "MEDIUM",
    "PPI": "MEDIUM",
    "CORE_PPI": "MEDIUM",
    "CLAIMS": "MEDIUM",
    "FED_SPEECH": "LOW",
}

_IMPACT_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
_VERIFY_ORDER = {"VERIFIED": 0, "UNVERIFIED": 1}

# Indikator makro resmi melalui FRED (BLS / BEA / US Dept. of Labor / US Census).
# CATATAN: FRED release_id=101 (FOMC Press Release) DILARANG digunakan untuk jadwal FOMC
# karena memuat seri harian DFEDTARU/DFEDTARL sehingga mengembalikan setiap hari kalender.
# Tuple: (key, name, release_id, release_kw, series_id, mode, unit, time_et, neutral_band, hot_bearish, agency, lag_days)
INDICATORS = [
    ("CPI",      "US CPI (YoY)",              10,  "Consumer Price",       "CPIAUCNS",        "yoy",   "%", "08:30", 0.05,   True,  "BLS",               5),
    ("CORE_CPI", "US Core CPI (YoY)",         10,  "Consumer Price",       "CPILFENS",        "yoy",   "%", "08:30", 0.05,   True,  "BLS",               5),
    ("NFP",      "US Nonfarm Payrolls (chg)", 50,  "Employment Situation", "PAYEMS",          "diff",  "K", "08:30", 25.0,   True,  "BLS",               5),
    ("UNRATE",   "US Unemployment Rate",      50,  "Employment Situation", "UNRATE",          "level", "%", "08:30", 0.05,   False, "BLS",               5),
    ("PCE",      "US PCE (YoY)",              54,  "Personal Income",      "PCEPI",           "yoy",   "%", "08:30", 0.05,   True,  "BEA",               5),
    ("CORE_PCE", "US Core PCE (YoY)",         54,  "Personal Income",      "PCEPILFE",        "yoy",   "%", "08:30", 0.05,   True,  "BEA",               5),
    ("RETAIL",   "US Retail Sales (MoM)",     9,   "Retail",               "RSAFS",           "mom",   "%", "08:30", 0.1,    True,  "US Census Bureau",  5),
    ("GDP",      "US GDP (q/q SAAR)",         53,  "Gross Domestic",       "A191RL1Q225SBEA", "level", "%", "08:30", 0.1,    True,  "BEA",               5),
    ("PPI",      "US PPI Final Demand (MoM)", 46,  "Producer Price",       "PPIFIS",          "mom",   "%", "08:30", 0.1,    True,  "BLS",               5),
    ("CORE_PPI", "US Core PPI (MoM)",         46,  "Producer Price",       "PPIFES",          "mom",   "%", "08:30", 0.1,    True,  "BLS",               5),
    ("CLAIMS",   "US Initial Jobless Claims", 180, "Claims",               "ICSA",            "level", "",  "08:30", 5000.0, False, "US Dept. of Labor", 2),
]
_IND_BY_KEY = {i[0]: i for i in INDICATORS}

_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}

_lock = threading.Lock()
_file_lock = threading.Lock()
_mem = {"cache": None, "last_attempt": 0.0, "last_ok": False, "error": "", "refreshing": False}
_gold_mem = {"cache": None, "last_attempt": 0.0, "last_ok": False, "error": "", "refreshing": False}


# ------------------------------------------------------------------ util
def _env_key(var_name):
    k = os.environ.get(var_name, "").strip()
    if k:
        return k
    p = os.path.join(_DIR, ".env")
    prefix = f"{var_name}="
    try:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith(prefix):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    except Exception:
        pass
    return ""


def _api_key():
    return _env_key("FRED_API_KEY")


class _Fail(Exception):
    pass


def _http_get(url, params=None, deadline=None, expect_json=True, headers=None, retries=None, expect_xml=False):
    import requests
    hdr = dict(HTTP_HEADERS)
    if headers:
        hdr.update(headers)
    max_tries = max(1, int(retries if retries is not None else RETRIES))
    last = "error"
    for i in range(max_tries):
        if deadline is not None and time.time() > deadline:
            raise _Fail("deadline")
        req_timeout = TIMEOUT
        if deadline is not None:
            rem = deadline - time.time()
            if rem <= 0.2:
                raise _Fail("deadline")
            req_timeout = max(1.0, min(TIMEOUT, rem))
        try:
            r = requests.get(url, params=params, headers=hdr, timeout=req_timeout)
            if r.status_code == 200:
                text = r.content.decode("utf-8-sig", errors="replace")
                if expect_json:
                    d = json.loads(text)
                    if not isinstance(d, (dict, list)):
                        raise _Fail("schema")
                    return d
                if expect_xml:
                    ctype = str(r.headers.get("Content-Type") or "").lower()
                    head_snip = text.lstrip()[:160].lower()
                    if ("text/html" in ctype or head_snip.startswith(("<!doctype html", "<html"))) and "<rss" not in head_snip and "<feed" not in head_snip:
                        raise _Fail("non_xml_html")
                return text
            last = "HTTP 429 (rate_limited)" if r.status_code == 429 else f"HTTP {r.status_code}"
            if r.status_code in (400, 401, 403, 404, 410, 429):
                raise _Fail(last)
        except _Fail:
            raise
        except Exception as e:  # jangan bocorkan URL+api_key di pesan error
            last = type(e).__name__
        if i < max_tries - 1:
            time.sleep(1.5 * (2 ** i))
    raise _Fail(last)


def _get_fred(path, params, deadline):
    p = dict(params)
    p["api_key"] = _api_key()
    p["file_type"] = "json"
    d = _http_get(FRED + path, params=p, deadline=deadline, expect_json=True)
    if not isinstance(d, dict):
        raise _Fail("schema")
    return d


def _now_utc():
    return datetime.now(timezone.utc)


def _event_dt_info(d_iso, hhmm):
    """Konversi jadwal America/New_York (DST otomatis) -> UTC + label zona EST/EDT."""
    h, m = hhmm.split(":")
    local = datetime(int(d_iso[:4]), int(d_iso[5:7]), int(d_iso[8:10]), int(h), int(m), tzinfo=TZ_NY)
    return local.astimezone(timezone.utc), (local.tzname() or "ET")


def _event_dt(d_iso, hhmm):
    return _event_dt_info(d_iso, hhmm)[0]


def _wib(dt_utc):
    return dt_utc.astimezone(TZ_WIB)


def _is_verified(ev):
    return str(ev.get("verification", "")).upper() == "VERIFIED"


def _sort_key(ev):
    return (
        str(ev.get("dt_utc") or ""),
        _IMPACT_ORDER.get(str(ev.get("impact") or "LOW").upper(), 2),
        _VERIFY_ORDER.get(str(ev.get("verification") or "UNVERIFIED").upper(), 1),
        str(ev.get("key") or ""),
        str(ev.get("id") or ""),
    )


def _parse_time_et(raw_time):
    """Parse timestamp ET resmi seperti '2:00 p.m.', '10:30 AM', '14:00', 'Noon' -> 'HH:MM' atau None."""
    s = str(raw_time or "").strip().lower()
    if not s:
        return None
    if "noon" in s:
        return "12:00"
    m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)\b", s)
    if m:
        hh = int(m.group(1))
        mm = int(m.group(2) or "0")
        ampm = m.group(3).replace(".", "")
        if not (1 <= hh <= 12 and 0 <= mm <= 59):
            return None
        if ampm == "pm" and hh != 12:
            hh += 12
        elif ampm == "am" and hh == 12:
            hh = 0
        return f"{hh:02d}:{mm:02d}"
    m24 = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", s)
    if m24:
        return f"{int(m24.group(1)):02d}:{int(m24.group(2)):02d}"
    return None


def _strip_tags(raw):
    return re.sub(r"\s+", " ", _html.unescape(re.sub(r"<[^>]+>", " ", str(raw or "")))).strip()


# ------------------------------------------------------------------ official Federal Reserve calendar
def _parse_fomc_calendars_html(html_text, lo, hi):
    """
    Parse halaman resmi https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm.
    Hanya mengambil jadwal resmi yang tertulis eksplisit (FOMC Statement & Minutes yg dirilis).
    Tidak pernah menebak tanggal FOMC Minutes.
    """
    out = {}
    if not html_text:
        return out
    panels = re.split(r'(?i)(?=<[a-z]+[^>]+class="[^"]*panel-heading[^"]*")', html_text)
    for panel in panels:
        ym = re.search(r"\b(20\d{2})\s+FOMC\s+Meetings\b", _strip_tags(panel[:600]), re.I)
        if not ym:
            continue
        year = int(ym.group(1))
        # Pisahkan baris meeting berdasarkan fomc-meeting__month
        rows = re.split(r'(?i)(?=<[a-z]+[^>]+class="[^"]*fomc-meeting__month[^"]*")', panel)
        for row in rows[1:]:
            mm = re.search(r'class="[^"]*fomc-meeting__month[^"]*"[^>]*>(.*?)</(?:div|span|td)>', row, re.I | re.S)
            dm = re.search(r'class="[^"]*fomc-meeting__date[^"]*"[^>]*>(.*?)</(?:div|span|td)>', row, re.I | re.S)
            if not mm or not dm:
                continue
            month_txt = _strip_tags(mm.group(1))
            date_txt = _strip_tags(dm.group(1))
            if not month_txt or not date_txt:
                continue
            if "cancel" in date_txt.lower():
                continue
            has_sep = "*" in date_txt or "*" in month_txt
            m_parts = [p.strip().lower().rstrip("*.") for p in re.split(r"[/\-]", month_txt) if p.strip()]
            m_nums = [_MONTHS.get(p) for p in m_parts if _MONTHS.get(p)]
            d_nums = [int(x) for x in re.findall(r"\b(\d{1,2})\b", date_txt)]
            if not m_nums or not d_nums:
                continue
            first_day, last_day = d_nums[0], d_nums[-1]
            ev_year = year
            if len(m_nums) >= 2 and last_day < first_day:
                ev_month = m_nums[-1]
                if m_nums[0] == 12 and ev_month == 1:
                    ev_year += 1
            else:
                ev_month = m_nums[0]
            try:
                stmt_date = date(ev_year, ev_month, last_day)
            except ValueError:
                continue
            if lo <= stmt_date <= hi:
                d_iso = stmt_date.isoformat()
                dt_utc, tz_et = _event_dt_info(d_iso, "14:00")
                eid = f"FOMC_STATEMENT|{d_iso}"
                out[eid] = {
                    "id": eid,
                    "key": "FOMC_STATEMENT",
                    "event_type": "FOMC_STATEMENT",
                    "name": "US FOMC Statement" + (" (SEP)" if has_sep else ""),
                    "impact": IMPACT_RULES["FOMC_STATEMENT"],
                    "release_id": None,
                    "release_name": "FOMC Meeting Statement & Policy Decision",
                    "official_id": f"FED-FOMC-STMT-{d_iso}",
                    "date_et": d_iso,
                    "time_et": "14:00",
                    "tz_et": tz_et,
                    "dt_utc": dt_utc.isoformat(timespec="minutes"),
                    "unit": "",
                    "forecast": None,
                    "actual": None,
                    "previous": None,
                    "revised": False,
                    "previous_original": None,
                    "period": None,
                    "source": "Federal Reserve (federalreserve.gov)",
                    "official_agency": "Federal Reserve",
                    "verification": "VERIFIED",
                    "time_verification": "STANDARD_RULE",
                    "series": None,
                }

            # Ekstrak tanggal rilis resmi Minutes hanya bila tertulis eksplisit "(Released Month Day, Year)"
            row_txt = _strip_tags(row)
            for rel_m in re.finditer(r"Released\s+([A-Za-z]+)\s+(\d{1,2}),\s*(20\d{2})", row_txt, re.I):
                m_num = _MONTHS.get(rel_m.group(1).lower())
                if not m_num:
                    continue
                try:
                    min_date = date(int(rel_m.group(3)), m_num, int(rel_m.group(2)))
                except ValueError:
                    continue
                if lo <= min_date <= hi:
                    m_iso = min_date.isoformat()
                    dt_utc, tz_et = _event_dt_info(m_iso, "14:00")
                    mid = f"FOMC_MINUTES|{m_iso}"
                    out[mid] = {
                        "id": mid,
                        "key": "FOMC_MINUTES",
                        "event_type": "FOMC_MINUTES",
                        "name": "US FOMC Minutes",
                        "impact": IMPACT_RULES["FOMC_MINUTES"],
                        "release_id": None,
                        "release_name": "Minutes of the Federal Open Market Committee",
                        "official_id": f"FED-FOMC-MIN-{m_iso}",
                        "date_et": m_iso,
                        "time_et": "14:00",
                        "tz_et": tz_et,
                        "dt_utc": dt_utc.isoformat(timespec="minutes"),
                        "unit": "",
                        "forecast": None,
                        "actual": None,
                        "previous": None,
                        "revised": False,
                        "previous_original": None,
                        "period": None,
                        "source": "Federal Reserve (federalreserve.gov)",
                        "official_agency": "Federal Reserve",
                        "verification": "VERIFIED",
                        "time_verification": "STANDARD_RULE",
                        "series": None,
                    }
    return out


def _classify_fed_calendar_item(title, ev_type, desc):
    """
    Klasifikasi ketat non-overlapping untuk event resmi federalreserve.gov/json/calendar.json.
    Membedakan secara eksplisit: FOMC_MINUTES, FOMC_RATE_DECISION, FOMC_STATEMENT, FED_SPEECH.
    """
    tl = str(title or "").strip().lower()
    el = str(ev_type or "").strip().lower()
    dl = str(desc or "").strip().lower()
    if not tl:
        return None, None, None
    # Abaikan item non-rilis pasar (blackout, transkrip historis, beige book, agenda, press conference terpisah)
    if any(skip in tl for skip in ("blackout", "transcript", "tealbook", "beige book", "agenda", "press conference")):
        return None, None, None
    is_fomc = ("fomc" in tl) or ("federal open market committee" in tl) or (el == "fomc")
    if is_fomc and "minutes" in tl:
        return "FOMC_MINUTES", "US FOMC Minutes", IMPACT_RULES["FOMC_MINUTES"]
    if is_fomc and ("rate decision" in tl or "target range" in tl):
        return "FOMC_RATE_DECISION", "US FOMC Rate Decision", IMPACT_RULES["FOMC_RATE_DECISION"]
    if is_fomc and ("statement" in tl or "meeting" in tl):
        return "FOMC_STATEMENT", "US FOMC Statement", IMPACT_RULES["FOMC_STATEMENT"]
    if any(k in el for k in ("speech", "testimony")) or any(k in tl for k in ("speech", "testimony", "remarks")):
        is_chair = ("chair" in (tl + " " + dl)) and ("vice chair" not in tl)
        imp = "MEDIUM" if is_chair else IMPACT_RULES["FED_SPEECH"]
        short_title = re.sub(r"\s+", " ", str(title or "").strip())[:52]
        return "FED_SPEECH", f"Fed Speech: {short_title}", imp
    return None, None, None


def _parse_fed_calendar_json(payload, lo, hi, existing):
    """Parse https://www.federalreserve.gov/json/calendar.json dan merge timestamp resmi tanpa duplikasi."""
    out = dict(existing or {})
    rows = payload.get("events", []) if isinstance(payload, dict) else (payload if isinstance(payload, list) else [])
    for item in rows:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        ev_type_raw = str(item.get("type") or "").strip()
        desc = str(item.get("description") or "").strip()
        ekey, ename, eimp = _classify_fed_calendar_item(title, ev_type_raw, desc)
        if not ekey:
            continue

        # Parse bulan & hari dari format Fed calendar.json ("month": "2026-11", "days": "18" atau "8, 9")
        m_str = str(item.get("month") or "").strip()
        d_str = str(item.get("days") or item.get("day") or "").strip()
        ym = re.fullmatch(r"(20\d{2})-(\d{1,2})", m_str)
        if not ym:
            continue
        year, month = int(ym.group(1)), int(ym.group(2))
        day_nums = [int(x) for x in re.findall(r"\b(\d{1,2})\b", d_str)]
        if not day_nums:
            continue
        day_val = day_nums[-1]  # untuk meeting 2 hari ("8, 9"), rilis statement adalah hari ke-2
        try:
            ev_date = date(year, month, day_val)
        except ValueError:
            continue
        if not (lo <= ev_date <= hi):
            continue
        d_iso = ev_date.isoformat()

        parsed_hhmm = _parse_time_et(item.get("time"))
        if parsed_hhmm:
            hhmm, t_ver = parsed_hhmm, "OFFICIAL"
        elif ekey in ("FOMC_STATEMENT", "FOMC_MINUTES", "FOMC_RATE_DECISION"):
            hhmm, t_ver = "14:00", "STANDARD_RULE"
        else:
            # Fed speech tanpa jam resmi -> jangan menebak jam
            continue

        dt_utc, tz_et = _event_dt_info(d_iso, hhmm)
        if ekey == "FED_SPEECH":
            slug = re.sub(r"[^a-z0-9]+", "", title.lower())[:16] or "speech"
            eid = f"FED_SPEECH|{d_iso}|{hhmm}|{slug}"
        else:
            eid = f"{ekey}|{d_iso}"

        if eid in out:
            # Upgrade timestamp bila calendar.json menyediakan jam resmi
            if t_ver == "OFFICIAL":
                out[eid]["time_et"] = hhmm
                out[eid]["tz_et"] = tz_et
                out[eid]["dt_utc"] = dt_utc.isoformat(timespec="minutes")
                out[eid]["time_verification"] = "OFFICIAL"
            continue

        out[eid] = {
            "id": eid,
            "key": ekey,
            "event_type": ekey,
            "name": ename,
            "impact": eimp,
            "release_id": None,
            "release_name": title[:80],
            "official_id": f"FED-CAL-{ekey}-{d_iso}",
            "date_et": d_iso,
            "time_et": hhmm,
            "tz_et": tz_et,
            "dt_utc": dt_utc.isoformat(timespec="minutes"),
            "unit": "",
            "forecast": None,
            "actual": None,
            "previous": None,
            "revised": False,
            "previous_original": None,
            "period": None,
            "source": "Federal Reserve (federalreserve.gov)",
            "official_agency": "Federal Reserve",
            "verification": "VERIFIED",
            "time_verification": t_ver,
            "series": None,
        }
    return out


def _fetch_fed_events(lo, hi, deadline):
    """Ambil jadwal resmi FOMC Statement, Minutes, Rate Decision & Fed Speech dari federalreserve.gov."""
    events = {}
    notes = []
    ok_any = False
    try:
        cal_html = _http_get(FED_FOMC_CAL_URL, deadline=deadline, expect_json=False)
        events = _parse_fomc_calendars_html(cal_html, lo, hi)
        ok_any = True
    except _Fail as e:
        notes.append(f"fed fomccalendars: {e}")

    if time.time() < deadline:
        try:
            cal_json = _http_get(
                FED_JSON_CAL_URL,
                deadline=deadline,
                expect_json=True,
                headers={"Referer": "https://www.federalreserve.gov/newsevents/calendar.htm"},
            )
            events = _parse_fed_calendar_json(cal_json, lo, hi, events)
            ok_any = True
        except _Fail as e:
            notes.append(f"fed calendar.json: {e}")

    return events, ok_any, notes


# ------------------------------------------------------------------ FRED data fetch
def _obs(sid, rd, deadline):
    d = _get_fred(
        "/series/observations",
        {"series_id": sid, "realtime_start": rd, "realtime_end": rd, "sort_order": "desc", "limit": 30},
        deadline,
    )
    rows = d.get("observations")
    if not isinstance(rows, list):
        raise _Fail("schema")
    out = []
    for o in rows:
        try:
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(o["date"])):
                continue
            val = float(o["value"])
            if not math.isfinite(val):
                continue
            out.append((o["date"], val))
        except Exception:
            continue  # "." = tidak ada nilai
    return out


def _pct(a, b):
    return (a / b - 1.0) * 100.0 if b else None


def _transform(obs, mode):
    """obs: [(date, val)] terbaru dulu. Return (periode, current, previous) atau None."""
    try:
        if mode == "level":
            return obs[0][0], obs[0][1], obs[1][1]
        if mode == "diff":
            return obs[0][0], obs[0][1] - obs[1][1], obs[1][1] - obs[2][1]
        if mode == "mom":
            return obs[0][0], _pct(obs[0][1], obs[1][1]), _pct(obs[1][1], obs[2][1])
        if mode == "yoy":
            def yo(i):
                if obs[i][0][:4] != str(int(obs[i + 12][0][:4]) + 1) or obs[i][0][5:7] != obs[i + 12][0][5:7]:
                    raise ValueError("gap")
                return _pct(obs[i][1], obs[i + 12][1])
            return obs[0][0], yo(0), yo(1)
    except Exception:
        return None
    return None


def _fetch_actual(ind, rd, today_et, deadline):
    """
    Actual terverifikasi hanya jika vintage rilis memuat periode BARU dibanding vintage H-1.
    Mengakomodasi jeda ingest FRED terbatas (<= lag_days) tanpa tumpang tindih ke rilis periode berikutnya.
    """
    sid, mode, lag_days = ind[4], ind[5], ind[11]
    rd_date = date.fromisoformat(rd)
    rd_prev = (rd_date - timedelta(days=1)).isoformat()
    a = _obs(sid, rd, deadline)
    b = _obs(sid, rd_prev, deadline)
    if (not a or not b or a[0][0] <= b[0][0]) and today_et > rd_date:
        rd_late = min(today_et, rd_date + timedelta(days=lag_days)).isoformat()
        if rd_late > rd and time.time() < deadline:
            a_late = _obs(sid, rd_late, deadline)
            if a_late and b and a_late[0][0] > b[0][0]:
                a = a_late
    if not a or not b or a[0][0] <= b[0][0]:
        return None  # belum ada data baru -> PENDING RELEASE / DELAYED
    ta, tb = _transform(a, mode), _transform(b, mode)
    if not ta or ta[1] is None or ta[2] is None:
        return None
    res = {
        "period": ta[0],
        "actual": round(ta[1], 4),
        "previous": round(ta[2], 4),
        "previous_original": None,
        "revised": False,
        "actual_asof": _now_utc().isoformat(timespec="seconds"),
    }
    if tb and tb[1] is not None and abs(tb[1] - ta[2]) > 1e-6 and tb[0] != ta[0]:
        res["previous_original"] = round(tb[1], 4)
        res["revised"] = True
    return res


def _refresh(old):
    """Satu siklus refresh multi-sumber resmi (Federal Reserve + FRED). Error per sumber diisolasi."""
    if TZ_NY is None or TZ_WIB is None:
        raise _Fail("tzdata missing")
    deadline = time.time() + REFRESH_DEADLINE
    now = _now_utc()
    today = now.astimezone(TZ_NY).date()
    lo, hi = today - timedelta(days=WINDOW_PAST_DAYS), today + timedelta(days=WINDOW_FUTURE_DAYS)
    old_ev = {e["id"]: e for e in (old or {}).get("events", []) if isinstance(e, dict) and e.get("id")}
    names = dict((old or {}).get("release_names", {}))
    notes = []
    source_status = {"FED": "UNAVAILABLE", "FRED": "UNAVAILABLE"}
    events = {}

    # 1. Official Federal Reserve Calendar (FOMC Statement, FOMC Minutes, Rate Decision, Fed Speech)
    fed_events, fed_ok, fed_notes = _fetch_fed_events(lo, hi, deadline)
    notes.extend(fed_notes)
    if fed_ok:
        source_status["FED"] = "OK"
        events.update(fed_events)

    # 2. Official Macro Releases via FRED API (BLS / BEA / US Dept. of Labor / US Census)
    key_str = _api_key()
    fred_dates = {}
    if not key_str:
        notes.append("FRED_API_KEY missing")
    elif not re.fullmatch(r"[0-9a-z]{32}", key_str):
        notes.append("FRED_API_KEY format invalid (32 char lowercase/digit)")
    else:
        for rid in sorted({i[2] for i in INDICATORS}):
            if time.time() > deadline:
                notes.append("FRED deadline reached")
                break
            try:
                if str(rid) not in names:
                    d = _get_fred("/release", {"release_id": rid}, deadline)
                    names[str(rid)] = str((d.get("releases") or [{}])[0].get("name", ""))
                kw = next(i[3] for i in INDICATORS if i[2] == rid)
                if kw.lower() not in names[str(rid)].lower():
                    notes.append(f"release {rid} name mismatch")
                    names.pop(str(rid), None)
                    continue
                d = _get_fred(
                    "/release/dates",
                    {"release_id": rid, "include_release_dates_with_no_data": "true", "sort_order": "desc", "limit": 100},
                    deadline,
                )
                rows = d.get("release_dates")
                if not isinstance(rows, list):
                    raise _Fail("schema")
                valid_dates = []
                for r in rows:
                    ds = str(r.get("date", ""))
                    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", ds):
                        continue
                    if not (lo.isoformat() <= ds <= hi.isoformat()):
                        continue
                    if date.fromisoformat(ds).weekday() >= 5:  # rilis makro resmi bukan Sabtu/Minggu
                        continue
                    valid_dates.append(ds)
                fred_dates[rid] = sorted(set(valid_dates))
            except _Fail as e:
                notes.append(f"FRED release {rid}: {e}")
        if fred_dates:
            source_status["FRED"] = "OK"

    if source_status["FED"] != "OK" and source_status["FRED"] != "OK":
        raise _Fail("; ".join(notes[:2]) or "all official providers unavailable")

    pending_actuals = []
    for ind in INDICATORS:
        key, name, rid = ind[0], ind[1], ind[2]
        agency = ind[10]
        for d_iso in fred_dates.get(rid, []):
            eid = f"{key}|{d_iso}"
            dt, tz_et = _event_dt_info(d_iso, ind[7])
            ev = {
                "id": eid,
                "key": key,
                "event_type": key,
                "name": name,
                "impact": IMPACT_RULES.get(key, "LOW"),
                "release_id": rid,
                "release_name": names.get(str(rid), ""),
                "official_id": f"FRED-REL{rid}-{key}-{d_iso}",
                "date_et": d_iso,
                "time_et": ind[7],
                "tz_et": tz_et,
                "dt_utc": dt.isoformat(timespec="minutes"),
                "unit": ind[6],
                "forecast": None,
                "actual": None,
                "previous": None,
                "revised": False,
                "previous_original": None,
                "period": None,
                "source": f"FRED/St. Louis Fed ({agency})",
                "official_agency": agency,
                "verification": "VERIFIED",
                "time_verification": "STANDARD_RULE",
                "series": ind[4],
            }
            prev = old_ev.get(eid)
            if prev and prev.get("actual") is not None:
                ev.update({
                    k: prev.get(k)
                    for k in ("actual", "previous", "revised", "previous_original", "period", "actual_asof")
                })
            elif ind[4] and now >= dt and now - dt <= timedelta(days=WINDOW_PAST_DAYS):
                pending_actuals.append((dt, ind, d_iso, eid))
            events[eid] = ev

    # Prioritaskan pengambilan actual dari rilis terbaru lebih dulu agar tidak kalah oleh deadline
    for _dt, ind, d_iso, eid in sorted(pending_actuals, key=lambda x: x[0], reverse=True):
        if time.time() >= deadline:
            break
        try:
            r = _fetch_actual(ind, d_iso, today, deadline)
            if r:
                events[eid].update(r)
        except _Fail as e:
            notes.append(f"{ind[0]}: {e}")

    ordered = sorted(events.values(), key=_sort_key)
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "fetched_at": now.isoformat(timespec="seconds"),
        "events": ordered,
        "release_names": names,
        "source_status": source_status,
        "notes": notes[:6],
    }


def _sanitize_cached_events(events):
    """Filter defensif: buang event legacy FRED release_id=101 dan pastikan urutan deterministik."""
    clean = []
    seen = set()
    for e in events:
        if not isinstance(e, dict) or not e.get("id") or not e.get("dt_utc"):
            continue
        if e.get("release_id") == 101:
            continue
        eid = e["id"]
        if eid in seen:
            continue
        seen.add(eid)
        clean.append(e)
    return sorted(clean, key=_sort_key)


def _load():
    if _mem["cache"] is None:
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                c = json.load(f)
            if (
                isinstance(c, dict)
                and c.get("schema_version") == CACHE_SCHEMA_VERSION
                and isinstance(c.get("events"), list)
                and c.get("fetched_at")
            ):
                c["events"] = _sanitize_cached_events(c["events"])
                _mem["cache"] = c
            if isinstance(c, dict) and _gold_mem["cache"] is None:
                gn = c.get("gold_news")
                if isinstance(gn, dict) and isinstance(gn.get("articles"), list) and gn.get("fetched_at"):
                    _gold_mem["cache"] = gn
        except Exception:
            pass
    return _mem["cache"]


def _save_cache_atomic(payload, gold_news=None):
    """Simpan cache secara atomik dengan tetap menjaga schema_version=2 & bagian gold_news."""
    with _file_lock:
        merged = dict(payload) if isinstance(payload, dict) else None
        existing = {}
        try:
            if os.path.exists(CACHE_FILE):
                with open(CACHE_FILE, "r", encoding="utf-8") as f:
                    d = json.load(f)
                if isinstance(d, dict):
                    existing = d
        except Exception:
            existing = {}

        if merged is None:
            # Hanya update bagian gold_news tanpa merusak data kalender ekonomi yang ada
            merged = dict(existing)
            if "schema_version" not in merged:
                merged["schema_version"] = CACHE_SCHEMA_VERSION
            merged.setdefault("events", [])
            merged.setdefault("fetched_at", None)
            merged.setdefault("release_names", {})
            merged.setdefault("source_status", {})
            merged.setdefault("notes", [])
        else:
            if gold_news is None and isinstance(existing.get("gold_news"), dict):
                merged["gold_news"] = existing["gold_news"]
        if gold_news is not None:
            merged["gold_news"] = gold_news

        tmp = CACHE_FILE + f".{os.getpid()}.{threading.get_ident()}.tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(merged, f)
            os.replace(tmp, CACHE_FILE)
        except Exception:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass


def _ttl(cache, now):
    for e in cache.get("events", []):
        try:
            dt = datetime.fromisoformat(e["dt_utc"])
        except Exception:
            continue
        if (
            _is_verified(e)
            and e.get("actual") is None
            and e.get("series")
            and dt - timedelta(minutes=5) <= now <= dt + timedelta(hours=2)
        ):
            return TTL_ACTIVE
    return TTL_NORMAL


def _do_refresh_locked(old_cache):
    try:
        new = _refresh(old_cache)
        with _lock:
            gn = _gold_mem["cache"]
        _save_cache_atomic(new, gold_news=gn)
        with _lock:
            _mem.update(cache=new, last_ok=True, error="", refreshing=False)
        return new
    except Exception as e:
        err = str(e)[:90] if isinstance(e, _Fail) else type(e).__name__
        with _lock:
            _mem.update(last_ok=False, error=err, refreshing=False)
        return None


def get_state(force=False, non_blocking=False):
    """
    Return {status: READY/STALE/UNAVAILABLE, events, fetched_at, error, source_status, notes}.
    Tidak pernah raise. Cache kedaluwarsa (> MAX_STALE_SECONDS) tidak pernah dipakai sebagai data live.
    """
    try:
        now = _now_utc()
        with _lock:
            cache = _load()
            age = (now - datetime.fromisoformat(cache["fetched_at"])).total_seconds() if cache else None
            ttl = _ttl(cache, now) if cache else TTL_NORMAL
            need = force or cache is None or age is None or age > ttl
            can_attempt = force or (time.time() - _mem["last_attempt"] >= FAIL_BACKOFF) or (_mem["last_ok"] and age is not None and age > ttl)
            should_spawn = need and can_attempt and not _mem["refreshing"]
            if should_spawn:
                _mem["last_attempt"] = time.time()
                _mem["refreshing"] = True

        if should_spawn:
            if non_blocking and not force:
                t = threading.Thread(target=_do_refresh_locked, args=(cache,), daemon=True)
                t.start()
            else:
                new_c = _do_refresh_locked(cache)
                if new_c is not None:
                    cache = new_c

        with _lock:
            cache = _mem["cache"]
            last_err = _mem["error"]

        if not cache:
            return {
                "status": "UNAVAILABLE",
                "events": [],
                "fetched_at": None,
                "error": last_err or "provider unavailable / no valid cache",
                "source_status": {"FED": "UNAVAILABLE", "FRED": "UNAVAILABLE"},
                "notes": [],
            }

        age = (now - datetime.fromisoformat(cache["fetched_at"])).total_seconds()
        ttl = _ttl(cache, now)
        if age > MAX_STALE_SECONDS:
            return {
                "status": "UNAVAILABLE",
                "events": [],
                "fetched_at": cache["fetched_at"],
                "error": last_err or f"cache expired ({int(age // 3600)}h old); refresh required",
                "source_status": cache.get("source_status") or {},
                "notes": cache.get("notes") or [],
            }

        fresh = (age <= ttl) or (not last_err and age <= ttl * 2)
        return {
            "status": "READY" if fresh else "STALE",
            "events": cache["events"],
            "fetched_at": cache["fetched_at"],
            "error": last_err,
            "source_status": cache.get("source_status") or {},
            "notes": cache.get("notes") or [],
        }
    except Exception as e:
        return {
            "status": "UNAVAILABLE",
            "events": [],
            "fetched_at": None,
            "error": type(e).__name__,
            "source_status": {},
            "notes": [],
        }


# ------------------------------------------------------------------ Gold Market News Intelligence (verified live headlines)
_TZ_ABBR_OFFSETS = {
    "EST": "-0500", "EDT": "-0400", "CST": "-0600", "CDT": "-0500",
    "MST": "-0700", "MDT": "-0600", "PST": "-0800", "PDT": "-0700",
    "GMT": "+0000", "UTC": "+0000",
}

_TRACKING_PARAMS = frozenset({
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "fbclid", "gclid", "mc_cid", "mc_eid", "ref", "ref_src", "cmpid", "mod",
    "rss", "source", "sref", "ocid", "guccounter", "guce_referrer",
})

_CREDIBILITY_TIER1_PUBLISHERS = (
    "reuters", "associated press", "ap news", "federal reserve", "bls",
    "bureau of labor statistics", "bea", "bureau of economic analysis",
    "bloomberg", "wall street journal", "wsj", "financial times",
)
_CREDIBILITY_TIER2_PUBLISHERS = (
    "cnbc", "marketwatch", "fxstreet", "kitco", "barron", "investing.com",
    "yahoo finance", "dow jones", "cme group", "world gold council",
)

# 9 topik prioritas fundamental XAUUSD: (label, weight, strong_patterns, secondary_patterns)
_GOLD_TOPIC_RULES = (
    (
        "Gold / Precious Metals", 34,
        (r"\bxau/?usd\b", r"\bspot gold\b", r"\bgold prices?\b", r"\bgold futures?\b",
         r"\bcomex gold\b", r"\bbullion\b", r"\bprecious metals?\b", r"\bemas\b"),
        (r"\bgold\b", r"\bsilver\b", r"\bxau\b"),
    ),
    (
        "Central Bank Gold", 30,
        (r"\bcentral bank(?:s)?\s+(?:gold|buy|buying|purchase|purchases|reserves?)\b",
         r"\bgold reserves?\b", r"\bpboc\b.*\bgold\b", r"\bsovereign gold\b"),
        (r"\bworld gold council\b",),
    ),
    (
        "Gold ETF Flows", 28,
        (r"\bgold etf(?:s)?\b", r"\bspdr gold\b", r"\bgold (?:inflows?|outflows?)\b",
         r"\bphysical gold demand\b"),
        (r"\bgld\b", r"\biau\b"),
    ),
    (
        "Fed Policy", 26,
        (r"\bfederal reserve\b", r"\bfomc\b", r"\bfed chair\b", r"\bfed funds\b",
         r"\binterest rate(?:s)?\b", r"\brate cut(?:s)?\b", r"\brate hike(?:s)?\b",
         r"\bmonetary policy\b", r"\bfed minutes\b", r"\bdot plot\b"),
        (r"\bfed\b", r"\bpowell\b", r"\bwarsh\b", r"\bhawkish\b", r"\bdovish\b",
         r"\bquantitative tightening\b", r"\bpolicy rate\b"),
    ),
    (
        "US Treasury Yields", 24,
        (r"\btreasury yields?\b", r"\b10-year (?:treasury )?yield\b", r"\b2-year (?:treasury )?yield\b",
         r"\breal yields?\b", r"\bu\.?s\.? treasur(?:y|ies)\b", r"\bbond yields?\b"),
        (r"\byields?\b", r"\btips yield\b"),
    ),
    (
        "US Dollar / DXY", 22,
        (r"\bu\.?s\.? dollar\b", r"\bdollar index\b", r"\bdxy\b", r"\bgreenback\b"),
        (r"\busd\b", r"\bdollar\b"),
    ),
    (
        "US Inflation (CPI/PCE)", 24,
        (r"\bconsumer price index\b", r"\bcore cpi\b", r"\bcore pce\b", r"\bpce inflation\b",
         r"\bu\.?s\.? inflation\b", r"\bproducer price index\b"),
        (r"\bcpi\b", r"\bpce\b", r"\bppi\b", r"\binflation\b", r"\bdisinflation\b", r"\bstagflation\b"),
    ),
    (
        "US Employment (NFP)", 24,
        (r"\bnonfarm payroll(?:s)?\b", r"\bjobs report\b", r"\bunemployment rate\b",
         r"\bjobless claims\b", r"\bemployment situation\b"),
        (r"\bnfp\b", r"\bpayrolls?\b", r"\blabor market\b", r"\bjolts\b"),
    ),
    (
        "Geopolitical Risk", 20,
        (r"\bsafe[- ]haven\b", r"\bgeopolitical\b", r"\bmiddle east\b", r"\btrade war\b",
         r"\bmilitary strike(?:s)?\b", r"\bsanctions?\b", r"\btariffs?\b"),
        (r"\bconflict\b", r"\bescalat(?:ion|es|ing)\b", r"\bceasefire\b", r"\bwar fears?\b"),
    ),
)

_NOISE_PATTERNS = (
    r"\bolympic gold\b", r"\bgold medal(?:s|ist)?\b", r"\bgolden globe(?:s)?\b",
    r"\bgolden state\b", r"\bgoldfish\b", r"\bgoldie\b", r"\bbox office\b",
    r"\bnfl\b", r"\bnba\b", r"\bmlb\b", r"\bcelebrity\b", r"\brecipe\b",
    r"\bmovie review\b", r"\balbum review\b", r"\bstreaming\b", r"\bfantasy football\b",
    r"\bgolden retriever\b", r"\bgolden bachelor\b", r"\bbitcoin meme\b",
)

_HIGH_IMPACT_CUES = (
    r"\bfomc\b", r"\brate decision\b", r"\bcuts? (?:interest )?rates?\b",
    r"\bhikes? (?:interest )?rates?\b", r"\bfed chair\b", r"\bnonfarm payroll(?:s)?\b",
    r"\bcpi\b", r"\bcore pce\b", r"\bconsumer price index\b", r"\bemergency\b",
    r"\bmilitary strike\b", r"\bwar\b", r"\bsurge(?:s|d)?\b", r"\bplunge(?:s|d)?\b",
    r"\brecord high\b", r"\ball[- ]time high\b",
)

_LOW_IMPACT_CUES = (
    r"\bpreview\b", r"\bweek ahead\b", r"\bwhat to expect\b", r"\bhow to\b",
    r"\bschedule\b", r"\bagenda\b", r"\bquiet\b", r"\blittle changed\b",
    r"\bopinion\b", r"\bpodcast\b", r"\bwebinar\b", r"\btechnical analysis\b",
)

# Kanal sentimen fundamental XAUUSD (berbasis bukti teks)
_SENTIMENT_BULLISH = (
    ("Fed Policy", (
        r"\brate cut(?:s)?\b", r"\bcuts? rates?\b", r"\bdovish\b", r"\beasing cycle\b",
        r"\bfed pivot\b", r"\bpause(?:s|d)? (?:rate )?hikes?\b", r"\blower (?:interest )?rates?\b",
        r"\baccommodative\b", r"\bfavor(?:s|ed)? (?:a )?cut\b",
    )),
    ("USD & Yields", (
        r"\bdollar (?:falls?|fell|slips?|slipped|drops?|dropped|weakens?|weakened|eases?|retreats?|slides?)\b",
        r"\bweaker (?:u\.?s\.? )?dollar\b", r"\bdxy (?:falls?|drops?|slips?|slides?)\b",
        r"\byields? (?:fall|fell|drop|dropped|slip|slipped|decline|declined|ease|eased|tumble|retreat)\b",
        r"\blower (?:treasury |bond |real )?yields?\b",
    )),
    ("Inflation & Jobs", (
        r"\binflation (?:cools?|cooled|eases?|eased|slows?|slowed|falls?|fell|moderates?)\b",
        r"\bcooler[- ]than[- ]expected\b", r"\bsofter (?:u\.?s\.? )?(?:cpi|pce|inflation|jobs|payrolls?)\b",
        r"\bdisinflation\b", r"\b(?:cpi|pce) (?:below|misses?|missed|cools?|eases?)\b",
        r"\bpayrolls? (?:miss|missed|slow|slowed|below)\b", r"\bweak(?:er)? (?:jobs|labor|employment)\b",
        r"\bunemployment (?:rises?|rose|jumps?|climbs?|ticks? up)\b",
        r"\bjobless claims (?:rise|rose|jump|jumped|surge|surged|climb)\b",
    )),
    ("Safe-Haven / Geopolitics", (
        r"\bsafe[- ]haven (?:demand|buying|flows?|bid|appeal)\b",
        r"\bflight to safety\b", r"\bgeopolitical (?:tensions?|risks?|fears?|escalation)\b",
        r"\b(?:conflict|tensions?|war) escalat(?:es|ed|ing)\b", r"\bwar fears?\b",
        r"\bmiddle east tensions?\b", r"\btariff (?:fears?|threats?|uncertainty)\b",
        r"\btrade war escalat(?:es|ed|ing)\b",
    )),
    ("Central Bank / ETF Flows", (
        r"\bcentral bank(?:s)? (?:buy|buying|add|added|increase|increased|boost)[^.]{0,30}\bgold\b",
        r"\bgold reserves? (?:rise|rose|increase|increased|climb|grow)\b",
        r"\bgold etf(?:s)? (?:inflows?|add|added)\b", r"\betf inflows?\b",
        r"\bstrong physical (?:gold )?demand\b",
    )),
)

_SENTIMENT_BEARISH = (
    ("Fed Policy", (
        r"\brate hike(?:s)?\b", r"\bhikes? rates?\b", r"\bhawkish\b", r"\bhigher[- ]for[- ]longer\b",
        r"\bdelay(?:s|ed|ing)? rate cuts?\b", r"\bfewer (?:rate )?cuts?\b", r"\bskip(?:s|ped)? (?:a )?cut\b",
        r"\brestrictive (?:policy|stance)\b", r"\btightening\b", r"\bno rush to cut\b",
    )),
    ("USD & Yields", (
        r"\bdollar (?:rises?|rose|rallies?|rallied|climbs?|climbed|gains?|gained|strengthens?|strengthened|jumps?|surges?)\b",
        r"\bstronger (?:u\.?s\.? )?dollar\b", r"\bdxy (?:rises?|rallies?|climbs?|jumps?|gains?)\b",
        r"\byields? (?:rise|rose|jump|jumped|surge|surged|climb|climbed|spike|spiked|advance|gain)\b",
        r"\bhigher (?:treasury |bond |real )?yields?\b",
    )),
    ("Inflation & Jobs", (
        r"\bhotter[- ]than[- ]expected\b", r"\bsticky inflation\b",
        r"\binflation (?:rises?|rose|accelerates?|accelerated|jumps?|surges?|heats? up|beats?)\b",
        r"\b(?:cpi|pce) (?:beats?|above|hotter|jumps?|surges?|accelerates?)\b",
        r"\bpayrolls? (?:beat|surged?|jump(?:ed)?|stronger|above)\b",
        r"\bstrong(?:er)? (?:jobs|labor|employment|payrolls?)\b",
        r"\bunemployment (?:falls?|fell|drops?|dropped|declines?)\b",
        r"\bjobless claims (?:fall|fell|drop|dropped|decline|lower)\b",
    )),
    ("Safe-Haven / Geopolitics", (
        r"\bceasefire\b", r"\bpeace (?:talks?|deal|agreement)\b",
        r"\btensions? ease(?:d|s)?\b", r"\bde-?escalation\b",
        r"\bsafe[- ]haven (?:demand )?(?:fades?|eases?|wanes?)\b",
        r"\brisk[- ]on rally\b", r"\btrade deal\b",
    )),
    ("Central Bank / ETF Flows", (
        r"\bgold etf(?:s)? (?:outflows?|shed|drop)\b", r"\betf outflows?\b",
        r"\bcentral bank(?:s)? (?:sell|sells|sold|trim|trimmed)[^.]{0,30}\bgold\b",
        r"\bprofit[- ]taking\b", r"\bweak(?:er)? physical (?:gold )?demand\b",
    )),
)

_PRICED_IN_CUES = (
    r"\balready priced[- ]in\b", r"\bpriced[- ]in\b", r"\bwidely expected\b",
    r"\bas expected\b", r"\bin line with (?:expectations|forecasts?|estimates?)\b",
    r"\bbaked in\b", r"\blittle changed\b", r"\bmuted reaction\b",
    r"\bheld? steady\b", r"\bholds? (?:rates? )?steady\b", r"\bunchanged\b",
)

_SPECULATIVE_OR_PREVIEW_CUES = (
    r"^(?:will|could|should|can|why|what|how|when|is|are)\b",
    r"\bpreview\b", r"\bweek ahead\b", r"\bwhat to expect\b", r"\bwhat to watch\b",
    r"\bahead of\b", r"\bawaits?\b", r"\bin focus\b", r"\bto speak\b", r"\bscheduled to\b",
)

_EN_STOPWORDS = frozenset({"the", "and", "for", "with", "from", "that", "after", "amid", "says", "will", "this", "over"})
_ID_STOPWORDS = frozenset({"yang", "dan", "untuk", "dengan", "dari", "pada", "dalam", "harga", "emas", "akan", "atau"})


def _local_tag(tag):
    return str(tag or "").rsplit("}", 1)[-1].lower()


def _clean_text(raw, max_len=None):
    txt = _strip_tags(raw)
    if max_len and len(txt) > max_len:
        txt = txt[:max_len].rstrip()
    return txt


def _canonicalize_url(raw_url):
    u = str(raw_url or "").strip()
    if not u:
        return "", ""
    try:
        p = urlparse(u)
        if p.scheme.lower() not in ("http", "https") or not p.netloc:
            return "", ""
        q = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=False) if k.lower() not in _TRACKING_PARAMS]
        path = re.sub(r"/{2,}", "/", p.path or "/").rstrip("/") or "/"
        canon = urlunparse((p.scheme.lower(), p.netloc.lower(), path, "", urlencode(sorted(q)), ""))
        return u, canon
    except Exception:
        return "", ""


def _parse_article_timestamp(raw_val, now=None):
    """Parse RFC 2822 / ISO 8601 / Unix epoch / AlphaVantage YYYYMMDDTHHMMSS -> ISO UTC string atau None."""
    if raw_val is None or raw_val == "":
        return None
    now = now or _now_utc()
    dt = None
    try:
        if isinstance(raw_val, (int, float)) and math.isfinite(raw_val) and raw_val > 946684800:
            dt = datetime.fromtimestamp(float(raw_val), tz=timezone.utc)
        else:
            s = str(raw_val).strip()
            if not s:
                return None
            if re.fullmatch(r"\d{10}(?:\.\d+)?", s):
                dt = datetime.fromtimestamp(float(s), tz=timezone.utc)
            elif re.fullmatch(r"\d{8}T\d{6}", s):
                dt = datetime.strptime(s, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
            else:
                iso_try = s.replace("Z", "+00:00")
                try:
                    dt = datetime.fromisoformat(iso_try)
                except ValueError:
                    # RFC 2822 (map EST/EDT/CST/PST agar kompatibel penuh di semua platform)
                    norm = re.sub(
                        r"\b(EST|EDT|CST|CDT|MST|MDT|PST|PDT|GMT|UTC)\b",
                        lambda m: _TZ_ABBR_OFFSETS[m.group(1)],
                        s,
                    )
                    dt = parsedate_to_datetime(norm)
        if dt is None:
            return None
        if dt.tzinfo is None:
            # Timestamp tanpa zona waktu eksplisit dianggap tidak pasti -> asumsi UTC hanya jika ISO/RFC valid, else None
            dt = dt.replace(tzinfo=timezone.utc)
        dt_utc = dt.astimezone(timezone.utc)
        # Tolak timestamp masa depan (> +15 menit) atau terlalu kuno (< 2020)
        if dt_utc > now + timedelta(minutes=15) or dt_utc.year < 2020:
            return None
        return dt_utc.isoformat(timespec="seconds")
    except Exception:
        return None


def _detect_language(raw_lang, text):
    rl = str(raw_lang or "").strip().lower()
    if rl:
        code = re.split(r"[-_]", rl)[0]
        if re.fullmatch(r"[a-z]{2,3}", code):
            return code
    words = set(re.findall(r"[a-z]{2,}", str(text or "").lower()))
    en_hits = len(words & _EN_STOPWORDS)
    id_hits = len(words & _ID_STOPWORDS)
    if id_hits >= 2 and id_hits > en_hits:
        return "id"
    if en_hits >= 2:
        return "en"
    return "unknown"


def _publisher_credibility_tier(publisher, default_tier=3):
    pl = str(publisher or "").strip().lower()
    if any(k in pl for k in _CREDIBILITY_TIER1_PUBLISHERS):
        return 1
    if any(k in pl for k in _CREDIBILITY_TIER2_PUBLISHERS):
        return 2
    return int(default_tier or 3)


def _normalize_headline_key(headline):
    s = str(headline or "").lower().strip()
    # Hapus suffix publisher umum (" - Reuters", " | CNBC")
    s = re.sub(r"\s+[\-|–—]\s+[a-z0-9 .&]{2,30}$", "", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _make_article_id(canonical_url, headline, publisher):
    seed = canonical_url or f"{_normalize_headline_key(headline)}|{str(publisher or '').strip().lower()}"
    return "GN-" + hashlib.sha256(seed.encode("utf-8", errors="ignore")).hexdigest()[:16]


def _validate_article_integrity(art, now):
    """Validasi integritas metadata wajib artikel: VERIFIED / INCOMPLETE / UNVERIFIED."""
    notes = []
    headline = str(art.get("headline") or "").strip()
    publisher = str(art.get("publisher") or "").strip()
    url = str(art.get("url") or "").strip()
    pub_at = art.get("published_at")
    ret_at = art.get("retrieved_at")
    lang = str(art.get("language") or "").strip().lower()

    hard_fail = False
    if len(headline) < 8:
        notes.append("headline missing/too short")
        hard_fail = True
    if not publisher:
        notes.append("publisher missing")
        hard_fail = True
    if not url or not art.get("canonical_url"):
        notes.append("invalid source url")
        hard_fail = True

    if hard_fail:
        return "UNVERIFIED", notes

    incomplete = False
    if not pub_at:
        if art.get("updated_at"):
            notes.append("missing published_at (only updated_at provided)")
        else:
            notes.append("missing published_at")
        incomplete = True
    else:
        try:
            dt_p = datetime.fromisoformat(pub_at)
            if dt_p > now + timedelta(minutes=15):
                notes.append("future published_at")
                return "UNVERIFIED", notes
        except Exception:
            notes.append("invalid published_at")
            incomplete = True

    if not ret_at:
        notes.append("missing retrieved_at")
        incomplete = True
    if not lang or lang == "unknown":
        notes.append("language unknown")
        incomplete = True

    return ("INCOMPLETE" if incomplete else "VERIFIED"), notes


def _article_freshness(art, now, cache_status="READY"):
    """Hitung status freshness dinamis terhadap waktu sekarang dan status cache."""
    if cache_status == "STALE":
        return "STALE_CACHE"
    if cache_status == "UNAVAILABLE":
        return "UNAVAILABLE"
    ts = art.get("published_at")
    if not ts:
        return "UNKNOWN"
    try:
        age_h = (now - datetime.fromisoformat(ts)).total_seconds() / 3600.0
    except Exception:
        return "UNKNOWN"
    if age_h < 0:
        return "UNVERIFIED_TIME"
    if age_h <= 2.0:
        return "FRESH (<2h)"
    if age_h <= 12.0:
        return "RECENT (<12h)"
    if age_h <= 36.0:
        return "TODAY/YESTERDAY"
    if age_h <= GOLD_NEWS_MAX_AGE_HOURS:
        return "AGING (<72h)"
    return "STALE (>72h)"


def _score_and_classify_article(art):
    """
    Filter relevansi XAUUSD, klasifikasi topik & impact (tidak menganggap semua berita emas HIGH).
    Return dict klasifikasi atau None bila tidak relevan / noise.
    """
    headline = str(art.get("headline") or "")
    summary = str(art.get("summary") or "")
    hl_l = headline.lower()
    txt_l = f"{headline} {summary}".lower()

    # Saring noise non-pasar (olahraga, hiburan, dsb.)
    if any(re.search(pat, hl_l) for pat in _NOISE_PATTERNS):
        return None

    matched_topics = []
    topic_scores = {}
    for label, weight, strong_pats, sec_pats in _GOLD_TOPIC_RULES:
        h_strong = any(re.search(p, hl_l) for p in strong_pats)
        s_strong = any(re.search(p, txt_l) for p in strong_pats)
        h_sec = any(re.search(p, hl_l) for p in sec_pats)
        s_sec = any(re.search(p, txt_l) for p in sec_pats)
        sc = 0
        if h_strong:
            sc += weight
        elif s_strong:
            sc += int(weight * 0.72)
        if h_sec:
            sc += int(weight * 0.55)
        elif s_sec:
            sc += int(weight * 0.30)
        if sc > 0:
            topic_scores[label] = sc
            matched_topics.append((sc, label))

    if not matched_topics:
        return None

    # Hindari false positive seperti "Goldman Sachs" tanpa topik emas/makro nyata
    if "goldman" in txt_l and not any(
        t in topic_scores for t in ("Fed Policy", "US Treasury Yields", "US Dollar / DXY",
                                    "US Inflation (CPI/PCE)", "US Employment (NFP)",
                                    "Central Bank Gold", "Gold ETF Flows", "Geopolitical Risk")
    ):
        if not re.search(r"\b(?:spot gold|gold prices?|gold futures?|xau/?usd|bullion|precious metals?)\b", txt_l):
            return None

    # Untuk feed resmi Fed/BLS/BEA, saring halaman indeks/direktori umum & rilis non-moneter
    is_official_agency = art.get("provider") in ("FED_PRESS_RSS", "FED_SPEECH_RSS", "BLS_RSS", "BEA_RSS")
    canon_url_l = str(art.get("canonical_url") or art.get("url") or "").lower().rstrip("/")
    if is_official_agency:
        if any(skip in hl_l for skip in (
            "enforcement action", "prohibition", "cease and desist",
            "merger", "bank holding", "community reinvestment",
            "workplace injuries", "fatal occupational", "state employment",
            "major economic indicators", "latest numbers", "schedule of releases",
            "release schedule", "frequently asked questions", "technical note", "methodology",
        )):
            return None
        if canon_url_l in ("https://www.bls.gov/bls", "https://www.bls.gov", "https://www.bea.gov"):
            return None
        # Untuk BLS/BEA, judul rilis sendiri harus spesifik memuat topik inflasi/ketenagakerjaan/makro utama
        if art.get("provider") in ("BLS_RSS", "BEA_RSS"):
            if not re.search(
                r"\b(?:consumer price|cpi|producer price|ppi|pce|personal income|personal consumption|"
                r"employment situation|nonfarm payroll|unemployment|jobless|gross domestic|gdp|inflation)\b",
                hl_l,
            ):
                return None

    matched_topics.sort(key=lambda x: x[0], reverse=True)
    primary_topic = matched_topics[0][1]
    topics_list = [lbl for _, lbl in matched_topics[:3]]

    # Saring berita minyak mentah / energi murni tanpa transmisi langsung ke emas/Fed/USD/yield/inflasi AS
    has_direct_gold = any(t in topic_scores for t in ("Gold / Precious Metals", "Central Bank Gold", "Gold ETF Flows"))
    has_macro_transmission = any(
        t in topic_scores for t in ("Fed Policy", "US Treasury Yields", "US Dollar / DXY", "US Inflation (CPI/PCE)", "US Employment (NFP)")
    )
    if re.search(r"\b(?:crude oil|oil prices?|oil declines?|oil rises?|oil jumps?|oil falls?|brent|wti|opec|barrel|natural gas|gasoline)\b", hl_l):
        if not has_direct_gold and not has_macro_transmission:
            return None

    raw_rel = sum(sc for sc, _ in matched_topics[:3])
    # Artikel geopolitik tanpa kaitan emas/USD/yield/Fed/safe-haven harus memiliki transmisi emas yang jelas (bukan minyak/saham umum)
    if primary_topic == "Geopolitical Risk" and len(matched_topics) == 1:
        if not re.search(r"\b(?:safe[- ]haven|gold|bullion|precious metals?|xau/?usd|dollar|dxy|treasury|yields?|federal reserve|tariff|trade war)\b", txt_l):
            return None

    tier = _publisher_credibility_tier(art.get("publisher"), art.get("credibility_tier", 3))
    tier_bonus = 14 if tier == 1 else (7 if tier == 2 else 0)
    total_score = min(100, raw_rel + tier_bonus)

    if total_score < 24:
        return None

    # Kalibrasi impact secara proporsional — JANGAN menganggap semua berita emas HIGH!
    has_high_cue = any(re.search(p, hl_l) for p in _HIGH_IMPACT_CUES)
    has_low_cue = any(re.search(p, hl_l) for p in _LOW_IMPACT_CUES)
    macro_core = any(t in topic_scores for t in ("Fed Policy", "US Inflation (CPI/PCE)", "US Employment (NFP)"))
    multi_driver = len(matched_topics) >= 2

    if art.get("integrity_status") != "VERIFIED" or has_low_cue:
        impact = "LOW"
    elif has_high_cue and (macro_core or tier == 1 or multi_driver):
        impact = "HIGH"
    elif (macro_core and total_score >= 42) or (multi_driver and total_score >= 45) or (tier == 1 and total_score >= 35):
        impact = "MEDIUM"
    else:
        impact = "LOW"

    return {
        "topic": primary_topic,
        "topics": topics_list,
        "credibility_tier": tier,
        "relevance_score": int(total_score),
        "impact_level": impact,
    }


def analyze_gold_article_sentiment(article):
    """
    Analisis sentimen fundamental XAUUSD berbasis bukti teks yang benar-benar tersedia.
    Output arah WAJIB salah satu dari:
      POTENTIAL BULLISH | POTENTIAL BEARISH | MIXED | NEUTRAL | UNCONFIRMED
    Tidak menyimpulkan arah hanya dari judul bila konteks tidak cukup.
    """
    integrity = str((article or {}).get("integrity_status") or "UNVERIFIED").upper()
    headline = str((article or {}).get("headline") or "").strip()
    summary = str((article or {}).get("summary") or "").strip()

    if integrity != "VERIFIED" or not headline:
        return {
            "direction": "UNCONFIRMED",
            "confidence": "NONE",
            "priced_in_risk": "UNKNOWN",
            "drivers": [],
            "evidence_basis": "INSUFFICIENT",
            "rationale": f"metadata artikel {integrity}; arah fundamental tidak disimpulkan",
        }

    hl_l = headline.lower()
    sum_l = summary.lower()
    has_summary = len(summary) >= 45 and _normalize_headline_key(summary) != _normalize_headline_key(headline)
    evidence_basis = "HEADLINE+SUMMARY" if has_summary else "HEADLINE_ONLY"
    text_l = f"{hl_l}. {sum_l}" if has_summary else hl_l

    # Pertanyaan / preview / jadwal pidato tanpa substansi arah -> jangan menebak arah
    is_question_or_preview = headline.endswith("?") or any(re.search(p, hl_l) for p in _SPECULATIVE_OR_PREVIEW_CUES)

    bull_channels = []
    bear_channels = []
    bull_hits = 0
    bear_hits = 0

    for chan, patterns in _SENTIMENT_BULLISH:
        hits = sum(1 for p in patterns if re.search(p, text_l))
        if hits:
            bull_hits += hits
            bull_channels.append(chan)

    for chan, patterns in _SENTIMENT_BEARISH:
        hits = sum(1 for p in patterns if re.search(p, text_l))
        if hits:
            bear_hits += hits
            bear_channels.append(chan)

    priced_in_hits = sum(1 for p in _PRICED_IN_CUES if re.search(p, text_l))
    priced_in_risk = "HIGH" if priced_in_hits >= 2 else ("POSSIBLE" if priced_in_hits == 1 else "LOW")

    drivers = []
    if bull_channels:
        drivers.append("Bullish cues: " + ", ".join(bull_channels))
    if bear_channels:
        drivers.append("Bearish cues: " + ", ".join(bear_channels))
    if priced_in_hits:
        drivers.append("Priced-in/unchanged cue terdeteksi")

    # Jika hanya judul (tanpa excerpt/summary), syarat bukti jauh lebih ketat agar tidak menebak dari judul tipis
    if not has_summary:
        if is_question_or_preview:
            return {
                "direction": "UNCONFIRMED",
                "confidence": "LOW",
                "priced_in_risk": priced_in_risk,
                "drivers": drivers,
                "evidence_basis": evidence_basis,
                "rationale": "konteks hanya dari judul bersifat preview/pertanyaan; bukti tidak cukup",
            }
        if bull_hits == 0 and bear_hits == 0:
            if priced_in_hits >= 1:
                return {
                    "direction": "NEUTRAL",
                    "confidence": "LOW",
                    "priced_in_risk": priced_in_risk,
                    "drivers": drivers,
                    "evidence_basis": evidence_basis,
                    "rationale": "judul mengindikasikan kondisi stabil/sesuai ekspektasi tanpa katalis baru",
                }
            return {
                "direction": "UNCONFIRMED",
                "confidence": "LOW",
                "priced_in_risk": priced_in_risk,
                "drivers": drivers,
                "evidence_basis": evidence_basis,
                "rationale": "hanya tersedia judul tanpa excerpt dan tanpa pendorong makro eksplisit",
            }
        if bull_hits > 0 and bear_hits > 0:
            return {
                "direction": "MIXED",
                "confidence": "LOW",
                "priced_in_risk": priced_in_risk,
                "drivers": drivers,
                "evidence_basis": evidence_basis,
                "rationale": "judul memuat pendorong bullish & bearish yang saling berlawanan (tanpa excerpt)",
            }
        # Butuh minimal 2 bukti kanal/frasa makro bila hanya dari judul, jika tidak -> UNCONFIRMED
        if max(bull_hits, bear_hits) < 2 and len(bull_channels) + len(bear_channels) < 2:
            return {
                "direction": "UNCONFIRMED",
                "confidence": "LOW",
                "priced_in_risk": priced_in_risk,
                "drivers": drivers,
                "evidence_basis": evidence_basis,
                "rationale": "konteks hanya dari judul dengan bukti tunggal; tidak cukup untuk memastikan arah",
            }

    # Evaluasi berbasis HEADLINE+SUMMARY (atau bukti multi-kanal eksplisit)
    if is_question_or_preview and (bull_hits + bear_hits) < 2:
        return {
            "direction": "UNCONFIRMED",
            "confidence": "LOW",
            "priced_in_risk": priced_in_risk,
            "drivers": drivers,
            "evidence_basis": evidence_basis,
            "rationale": "artikel bersifat preview/menunggu rilis; belum ada konfirmasi fundamental",
        }

    if bull_hits == 0 and bear_hits == 0:
        if priced_in_hits >= 1:
            return {
                "direction": "NEUTRAL",
                "confidence": "MODERATE" if has_summary else "LOW",
                "priced_in_risk": priced_in_risk,
                "drivers": drivers,
                "evidence_basis": evidence_basis,
                "rationale": "isi mengindikasikan data/kebijakan sesuai ekspektasi atau sudah priced-in",
            }
        return {
            "direction": "UNCONFIRMED",
            "confidence": "LOW",
            "priced_in_risk": priced_in_risk,
            "drivers": drivers,
            "evidence_basis": evidence_basis,
            "rationale": "tidak ditemukan sinyal fundamental terarah yang cukup pada teks tersedia",
        }

    if bull_hits > 0 and bear_hits > 0:
        return {
            "direction": "MIXED",
            "confidence": "MODERATE" if has_summary else "LOW",
            "priced_in_risk": priced_in_risk,
            "drivers": drivers,
            "evidence_basis": evidence_basis,
            "rationale": f"faktor saling tarik: {', '.join(bull_channels)} (bullish) vs {', '.join(bear_channels)} (bearish)",
        }

    # Arah tunggal: cek apakah sudah sangat priced-in dengan sinyal lemah
    if priced_in_hits >= 2 and max(bull_hits, bear_hits) <= 1:
        return {
            "direction": "NEUTRAL",
            "confidence": "MODERATE" if has_summary else "LOW",
            "priced_in_risk": priced_in_risk,
            "drivers": drivers,
            "evidence_basis": evidence_basis,
            "rationale": "katalis ringan dan indikasi kuat informasi sudah priced-in di pasar",
        }

    caveat = "; catatan: sebagian mungkin sudah priced-in" if priced_in_hits >= 1 else ""
    conf = "MODERATE" if (has_summary and max(bull_hits, bear_hits) >= 2) else "LOW"
    if bull_hits > 0:
        return {
            "direction": "POTENTIAL BULLISH",
            "confidence": conf,
            "priced_in_risk": priced_in_risk,
            "drivers": drivers,
            "evidence_basis": evidence_basis,
            "rationale": f"didukung {', '.join(bull_channels)} -> potensi dukungan ke XAUUSD{caveat}",
        }
    return {
        "direction": "POTENTIAL BEARISH",
        "confidence": conf,
        "priced_in_risk": priced_in_risk,
        "drivers": drivers,
        "evidence_basis": evidence_basis,
        "rationale": f"tekanan dari {', '.join(bear_channels)} -> potensi tekanan ke XAUUSD{caveat}",
    }


_XML_ENTITY_RE = re.compile(r"&([a-zA-Z][a-zA-Z0-9]+);")
_BARE_AMP_RE = re.compile(r"&(?!(?:amp|lt|gt|quot|apos|#\d+|#x[0-9a-fA-F]+);)")


def _sanitize_rss_xml(xml_str):
    """Perbaiki entitas HTML umum (&nbsp;, &mdash;, dll.) dan '&' tak ter-escape pada feed XML resmi."""
    def _rep_entity(match):
        name = match.group(1)
        if name in ("amp", "lt", "gt", "quot", "apos"):
            return match.group(0)
        cp = _HTML_ENTITIES.get(name)
        if cp is not None:
            return f"&#{cp};"
        return f"&amp;{name};"

    fixed = _XML_ENTITY_RE.sub(_rep_entity, xml_str)
    fixed = _BARE_AMP_RE.sub("&amp;", fixed)
    return fixed


def _parse_rss_or_atom(xml_text, provider_id, publisher_default, credibility_tier, default_lang, retrieved_at, now):
    """Parser aman RSS 2.0 / Atom 1.0 / RDF (stdlib ElementTree, tanpa scraping halaman artikel)."""
    if not xml_text:
        return []
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", str(xml_text).lstrip("\ufeff \t\r\n"))
    head_l = cleaned[:200].lower()
    if head_l.startswith(("<!doctype html", "<html")) and "<rss" not in head_l and "<feed" not in head_l:
        raise _Fail("non_xml_html")
    try:
        root = ET.fromstring(cleaned)
    except ET.ParseError as e:
        try:
            root = ET.fromstring(_sanitize_rss_xml(cleaned))
        except ET.ParseError:
            raise _Fail(f"xml_parse:{type(e).__name__}")

    root_tag = _local_tag(root.tag)
    feed_lang = default_lang or ""
    items = []

    if root_tag == "feed":
        # Atom 1.0
        for attr_k, attr_v in root.attrib.items():
            if attr_k.lower().endswith("lang") and attr_v:
                feed_lang = attr_v
                break
        entries = [el for el in root if _local_tag(el.tag) == "entry"]
        for entry in entries[:40]:
            title = ""
            summary = ""
            url_raw = ""
            pub_raw = ""
            upd_raw = ""
            pub_name = publisher_default
            for child in entry:
                t = _local_tag(child.tag)
                if t == "title" and not title:
                    title = _clean_text("".join(child.itertext()), max_len=240)
                elif t in ("summary", "content") and not summary:
                    summary = _clean_text("".join(child.itertext()), max_len=420)
                elif t == "link":
                    rel = str(child.attrib.get("rel") or "alternate").lower()
                    href = str(child.attrib.get("href") or "").strip()
                    if href and (rel == "alternate" or not url_raw) and rel not in ("self", "enclosure"):
                        url_raw = href
                elif t == "published" and not pub_raw:
                    pub_raw = (child.text or "").strip()
                elif t == "updated" and not upd_raw:
                    upd_raw = (child.text or "").strip()
                elif t == "source":
                    for sc in child:
                        if _local_tag(sc.tag) == "title" and (sc.text or "").strip():
                            pub_name = _clean_text(sc.text, max_len=60)
                            break
            items.append((title, summary, url_raw, pub_raw, upd_raw, pub_name, feed_lang))
    else:
        # RSS 2.0 atau RDF
        channel = next((el for el in root if _local_tag(el.tag) == "channel"), root)
        for ch_el in channel:
            if _local_tag(ch_el.tag) == "language" and (ch_el.text or "").strip():
                feed_lang = ch_el.text.strip()
                break
        raw_items = [el for el in channel if _local_tag(el.tag) == "item"]
        if not raw_items:
            raw_items = [el for el in root if _local_tag(el.tag) == "item"]
        for item in raw_items[:40]:
            title = ""
            summary = ""
            url_raw = ""
            guid_url = ""
            pub_raw = ""
            upd_raw = ""
            pub_name = publisher_default
            for child in item:
                t = _local_tag(child.tag)
                txt = (child.text or "").strip()
                if t == "title" and not title:
                    title = _clean_text("".join(child.itertext()), max_len=240)
                elif t in ("description", "summary") and not summary:
                    summary = _clean_text("".join(child.itertext()), max_len=420)
                elif t == "link" and not url_raw:
                    url_raw = txt or str(child.attrib.get("href") or "").strip()
                elif t == "guid" and not guid_url:
                    if str(child.attrib.get("isPermaLink", "true")).lower() != "false" and txt.startswith(("http://", "https://")):
                        guid_url = txt
                elif t in ("pubdate", "published", "date") and not pub_raw:
                    pub_raw = txt
                elif t in ("updated", "modified") and not upd_raw:
                    upd_raw = txt
                elif t == "source" and txt:
                    pub_name = _clean_text(txt, max_len=60)
            items.append((title, summary, url_raw or guid_url, pub_raw, upd_raw, pub_name, feed_lang))

    out = []
    for title, summary, url_raw, pub_raw, upd_raw, pub_name, lang_raw in items:
        orig_url, canon_url = _canonicalize_url(url_raw)
        pub_iso = _parse_article_timestamp(pub_raw, now=now)
        upd_iso = _parse_article_timestamp(upd_raw, now=now)
        if pub_iso and upd_iso and upd_iso <= pub_iso:
            upd_iso = None
        art = {
            "article_id": _make_article_id(canon_url, title, pub_name),
            "canonical_url": canon_url,
            "headline": title,
            "summary": summary,
            "publisher": pub_name,
            "provider": provider_id,
            "url": orig_url,
            "published_at": pub_iso,
            "updated_at": upd_iso,
            "retrieved_at": retrieved_at,
            "language": _detect_language(lang_raw, f"{title} {summary}"),
            "credibility_tier": credibility_tier,
        }
        out.append(art)
    return out


def _fetch_optional_market_apis(deadline, retrieved_at, now):
    """Fetch dari Market News API resmi jika API key dikonfigurasi di .env/env (error per provider terisolasi)."""
    articles = []
    status = {}
    notes = []
    missing = []

    # 1. Finnhub Market News API
    fk = _env_key("FINNHUB_API_KEY")
    if not fk:
        missing.append("FINNHUB_API_KEY")
    elif time.time() < deadline:
        try:
            got_any = False
            for cat in ("forex", "general"):
                if time.time() >= deadline:
                    break
                rows = _http_get(
                    "https://finnhub.io/api/v1/news",
                    params={"category": cat, "token": fk},
                    deadline=deadline,
                    expect_json=True,
                )
                if isinstance(rows, list):
                    got_any = True
                    for r in rows[:35]:
                        if not isinstance(r, dict):
                            continue
                        orig_url, canon_url = _canonicalize_url(r.get("url"))
                        title = _clean_text(r.get("headline"), max_len=240)
                        summary = _clean_text(r.get("summary"), max_len=420)
                        pub = _clean_text(r.get("source"), max_len=60)
                        pub_iso = _parse_article_timestamp(r.get("datetime"), now=now)
                        articles.append({
                            "article_id": _make_article_id(canon_url, title, pub),
                            "canonical_url": canon_url,
                            "headline": title,
                            "summary": summary,
                            "publisher": pub,
                            "provider": "FINNHUB",
                            "url": orig_url,
                            "published_at": pub_iso,
                            "updated_at": None,
                            "retrieved_at": retrieved_at,
                            "language": _detect_language("en", f"{title} {summary}"),
                            "credibility_tier": _publisher_credibility_tier(pub, 2),
                        })
            status["FINNHUB"] = "OK" if got_any else "EMPTY"
        except _Fail as e:
            status["FINNHUB"] = f"ERR:{e}"
            notes.append(f"FINNHUB: {e}")

    # 2. Alpha Vantage News & Sentiment API
    ak = _env_key("ALPHAVANTAGE_API_KEY")
    if not ak:
        missing.append("ALPHAVANTAGE_API_KEY")
    elif time.time() < deadline:
        try:
            d = _http_get(
                "https://www.alphavantage.co/query",
                params={"function": "NEWS_SENTIMENT", "topics": "economy_monetary,economy_macro,financial_markets", "limit": 30, "apikey": ak},
                deadline=deadline,
                expect_json=True,
            )
            feed = d.get("feed") if isinstance(d, dict) else None
            if isinstance(feed, list):
                for r in feed[:30]:
                    if not isinstance(r, dict):
                        continue
                    orig_url, canon_url = _canonicalize_url(r.get("url"))
                    title = _clean_text(r.get("title"), max_len=240)
                    summary = _clean_text(r.get("summary"), max_len=420)
                    pub = _clean_text(r.get("source"), max_len=60)
                    pub_iso = _parse_article_timestamp(r.get("time_published"), now=now)
                    articles.append({
                        "article_id": _make_article_id(canon_url, title, pub),
                        "canonical_url": canon_url,
                        "headline": title,
                        "summary": summary,
                        "publisher": pub,
                        "provider": "ALPHAVANTAGE",
                        "url": orig_url,
                        "published_at": pub_iso,
                        "updated_at": None,
                        "retrieved_at": retrieved_at,
                        "language": _detect_language("en", f"{title} {summary}"),
                        "credibility_tier": _publisher_credibility_tier(pub, 2),
                    })
                status["ALPHAVANTAGE"] = "OK"
            else:
                msg = str((d or {}).get("Information") or (d or {}).get("Note") or "no_feed")[:45]
                status["ALPHAVANTAGE"] = "RATE_LIMITED"
                notes.append(f"ALPHAVANTAGE: {msg}")
        except _Fail as e:
            status["ALPHAVANTAGE"] = f"ERR:{e}"
            notes.append(f"ALPHAVANTAGE: {e}")

    # 3. Marketaux Financial News API
    mk = _env_key("MARKETAUX_API_KEY")
    if not mk:
        missing.append("MARKETAUX_API_KEY")
    elif time.time() < deadline:
        try:
            d = _http_get(
                "https://api.marketaux.com/v1/news/all",
                params={"search": "gold|xauusd|federal reserve|inflation|treasury", "language": "en", "limit": 20, "api_token": mk},
                deadline=deadline,
                expect_json=True,
            )
            rows = d.get("data") if isinstance(d, dict) else None
            if isinstance(rows, list):
                for r in rows[:20]:
                    if not isinstance(r, dict):
                        continue
                    orig_url, canon_url = _canonicalize_url(r.get("url"))
                    title = _clean_text(r.get("title"), max_len=240)
                    summary = _clean_text(r.get("description") or r.get("snippet"), max_len=420)
                    pub = _clean_text(r.get("source"), max_len=60)
                    pub_iso = _parse_article_timestamp(r.get("published_at"), now=now)
                    articles.append({
                        "article_id": _make_article_id(canon_url, title, pub),
                        "canonical_url": canon_url,
                        "headline": title,
                        "summary": summary,
                        "publisher": pub,
                        "provider": "MARKETAUX",
                        "url": orig_url,
                        "published_at": pub_iso,
                        "updated_at": None,
                        "retrieved_at": retrieved_at,
                        "language": _detect_language(r.get("language"), f"{title} {summary}"),
                        "credibility_tier": _publisher_credibility_tier(pub, 2),
                    })
                status["MARKETAUX"] = "OK"
        except _Fail as e:
            status["MARKETAUX"] = f"ERR:{e}"
            notes.append(f"MARKETAUX: {e}")

    # 4. NewsAPI.org
    nk = _env_key("NEWSAPI_KEY")
    if not nk:
        missing.append("NEWSAPI_KEY")
    elif time.time() < deadline:
        try:
            d = _http_get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": '(gold OR XAUUSD OR "Federal Reserve" OR "Treasury yields" OR "spot gold")',
                    "language": "en",
                    "sortBy": "publishedAt",
                    "pageSize": 20,
                    "apiKey": nk,
                },
                deadline=deadline,
                expect_json=True,
            )
            rows = d.get("articles") if isinstance(d, dict) else None
            if isinstance(rows, list):
                for r in rows[:20]:
                    if not isinstance(r, dict):
                        continue
                    orig_url, canon_url = _canonicalize_url(r.get("url"))
                    title = _clean_text(r.get("title"), max_len=240)
                    summary = _clean_text(r.get("description"), max_len=420)
                    src_obj = r.get("source") if isinstance(r.get("source"), dict) else {}
                    pub = _clean_text(src_obj.get("name"), max_len=60)
                    pub_iso = _parse_article_timestamp(r.get("publishedAt"), now=now)
                    articles.append({
                        "article_id": _make_article_id(canon_url, title, pub),
                        "canonical_url": canon_url,
                        "headline": title,
                        "summary": summary,
                        "publisher": pub,
                        "provider": "NEWSAPI",
                        "url": orig_url,
                        "published_at": pub_iso,
                        "updated_at": None,
                        "retrieved_at": retrieved_at,
                        "language": _detect_language("en", f"{title} {summary}"),
                        "credibility_tier": _publisher_credibility_tier(pub, 3),
                    })
                status["NEWSAPI"] = "OK"
        except _Fail as e:
            status["NEWSAPI"] = f"ERR:{e}"
            notes.append(f"NEWSAPI: {e}")

    return articles, status, notes, missing


def _title_tokens(norm_title):
    return {w for w in norm_title.split() if len(w) > 2}


def _deduplicate_articles(articles):
    """Deduplikasi berdasarkan canonical URL / article_id, judul ternormalisasi, dan kemiripan Jaccard >= 0.78."""
    def _quality(a):
        int_rank = 2 if a.get("integrity_status") == "VERIFIED" else (1 if a.get("integrity_status") == "INCOMPLETE" else 0)
        tier_rank = 4 - int(a.get("credibility_tier") or 3)
        has_sum = 1 if len(str(a.get("summary") or "")) >= 45 else 0
        has_upd = 1 if a.get("updated_at") else 0
        return (int_rank, tier_rank, has_sum, has_upd, int(a.get("relevance_score") or 0), str(a.get("published_at") or ""))

    ranked = sorted(articles, key=_quality, reverse=True)
    kept = []
    seen_urls = set()
    seen_ids = set()
    seen_titles = []

    for art in ranked:
        aid = art.get("article_id") or ""
        curl = art.get("canonical_url") or ""
        ntitle = _normalize_headline_key(art.get("headline"))
        if not ntitle:
            continue
        if curl and curl in seen_urls:
            continue
        if aid and aid in seen_ids:
            continue
        toks = _title_tokens(ntitle)
        is_dup = False
        for prev_title, prev_toks in seen_titles:
            if ntitle == prev_title:
                is_dup = True
                break
            if toks and prev_toks:
                inter = len(toks & prev_toks)
                union = len(toks | prev_toks)
                if union > 0 and (inter / union) >= 0.78:
                    is_dup = True
                    break
        if is_dup:
            continue
        if curl:
            seen_urls.add(curl)
        if aid:
            seen_ids.add(aid)
        seen_titles.append((ntitle, toks))
        kept.append(art)
    return kept


def _article_sort_key(art):
    """Prioritaskan terbaru dalam kelompok kredibel & relevan."""
    int_ord = 0 if art.get("integrity_status") == "VERIFIED" else (1 if art.get("integrity_status") == "INCOMPLETE" else 2)
    ts = str(art.get("updated_at") or art.get("published_at") or "")
    return (int_ord, ts, -int(art.get("credibility_tier") or 3), int(art.get("relevance_score") or 0))


def _select_top_gold_articles(articles, limit=5, now=None):
    """
    Pilih artikel paling relevan & kredibel, lalu tampilkan yang terbaru terlebih dahulu.
    """
    now = now or _now_utc()
    if not articles:
        return []

    def _priority_rank(a):
        int_bonus = 40 if a.get("integrity_status") == "VERIFIED" else 0
        tier_bonus = 24 if a.get("credibility_tier") == 1 else (12 if a.get("credibility_tier") == 2 else 0)
        imp_bonus = 18 if a.get("impact_level") == "HIGH" else (8 if a.get("impact_level") == "MEDIUM" else 0)
        rel = int(a.get("relevance_score") or 0)
        recency_bonus = 0
        ts = a.get("published_at")
        if ts:
            try:
                age_h = max(0.0, (now - datetime.fromisoformat(ts)).total_seconds() / 3600.0)
                recency_bonus = max(0, int(36 - age_h * 0.5))
            except Exception:
                pass
        return int_bonus + tier_bonus + imp_bonus + rel + recency_bonus

    top_pool = sorted(articles, key=_priority_rank, reverse=True)[: max(limit, 5)]
    # Setelah seleksi kualitas (relevansi & kredibilitas), urutkan TERBARU terlebih dahulu
    return sorted(
        top_pool,
        key=lambda a: (
            str(a.get("updated_at") or a.get("published_at") or ""),
            -int(a.get("credibility_tier") or 3),
            int(a.get("relevance_score") or 0),
        ),
        reverse=True,
    )[:limit]


def _refresh_gold_news(old_gold, force=False):
    """Satu siklus refresh multi-provider Gold Market News (RSS resmi paralel + optional Market News API)."""
    deadline = time.time() + GOLD_NEWS_REFRESH_DEADLINE
    now = _now_utc()
    now_ep = time.time()
    retrieved_at = now.isoformat(timespec="seconds")
    raw_articles = []
    provider_status = {}
    notes = []
    ok_any = False

    if force:
        _gold_provider_cooldown.clear()

    # 1. Official RSS / Atom Feeds secara paralel agar satu feed lambat/error tidak membuat feed lain timeout
    rss_headers = {"Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml;q=0.9, */*;q=0.5"}
    feeds_to_fetch = []
    for feed_spec in GOLD_RSS_FEEDS:
        pid = feed_spec[0]
        cd = _gold_provider_cooldown.get(pid)
        if cd and cd[0] > now_ep:
            provider_status[pid] = f"UNAVAILABLE(cooldown:{cd[1]})"
            continue
        feeds_to_fetch.append(feed_spec)

    def _fetch_one_rss(spec):
        pid, pub_default, url, tier, def_lang = spec
        xml_txt = _http_get(
            url,
            deadline=deadline,
            expect_json=False,
            headers=rss_headers,
            retries=1,
            expect_xml=True,
        )
        return _parse_rss_or_atom(xml_txt, pid, pub_default, tier, def_lang, retrieved_at, now)

    if feeds_to_fetch:
        pool_workers = min(4, len(feeds_to_fetch))
        with ThreadPoolExecutor(max_workers=pool_workers) as pool:
            fut_map = {pool.submit(_fetch_one_rss, spec): spec[0] for spec in feeds_to_fetch}
            wait_sec = max(1.0, deadline - time.time())
            try:
                for fut in as_completed(fut_map, timeout=wait_sec):
                    pid = fut_map[fut]
                    try:
                        parsed = fut.result()
                        raw_articles.extend(parsed)
                        provider_status[pid] = "OK"
                        _gold_provider_cooldown.pop(pid, None)
                        ok_any = True
                    except _Fail as e:
                        err_s = str(e)
                        provider_status[pid] = f"ERR:{err_s}"
                        notes.append(f"{pid}: {err_s}")
                        if any(tok in err_s for tok in ("HTTP 400", "HTTP 404", "HTTP 410", "HTTP 429", "non_xml_html")):
                            _gold_provider_cooldown[pid] = (time.time() + GOLD_PROVIDER_COOLDOWN_SEC, err_s)
                    except Exception as e:
                        err_s = type(e).__name__
                        provider_status[pid] = f"ERR:{err_s}"
                        notes.append(f"{pid}: {err_s}")
            except TimeoutError:
                for fut, pid in fut_map.items():
                    if pid not in provider_status:
                        provider_status[pid] = "ERR:deadline"
                        notes.append(f"{pid}: deadline")
                        fut.cancel()

    # 2. Optional documented Market News APIs (aktif jika API key ada di .env / env)
    api_arts, api_status, api_notes, missing_keys = _fetch_optional_market_apis(deadline, retrieved_at, now)
    raw_articles.extend(api_arts)
    provider_status.update(api_status)
    notes.extend(api_notes)
    if any(v == "OK" for v in api_status.values()):
        ok_any = True

    if not ok_any:
        raise _Fail("; ".join(notes[:2]) or "all gold news providers unavailable")

    # 3. Validasi integritas, filter umur (>72h), relevansi XAUUSD, dan sentimen berbasis bukti
    filtered = []
    for art in raw_articles:
        integrity, int_notes = _validate_article_integrity(art, now)
        if integrity == "UNVERIFIED":
            continue
        art["integrity_status"] = integrity
        art["integrity_notes"] = int_notes

        # Filter artikel terlalu lama (> GOLD_NEWS_MAX_AGE_HOURS)
        if art.get("published_at"):
            try:
                age_h = (now - datetime.fromisoformat(art["published_at"])).total_seconds() / 3600.0
                if age_h > GOLD_NEWS_MAX_AGE_HOURS:
                    continue
            except Exception:
                continue

        cls = _score_and_classify_article(art)
        if not cls:
            continue
        art.update(cls)
        art["sentiment"] = analyze_gold_article_sentiment(art)
        art["freshness"] = _article_freshness(art, now, cache_status="READY")
        filtered.append(art)

    deduped = _deduplicate_articles(filtered)
    deduped.sort(
        key=lambda a: (
            str(a.get("updated_at") or a.get("published_at") or ""),
            -int(a.get("credibility_tier") or 3),
            int(a.get("relevance_score") or 0),
        ),
        reverse=True,
    )

    return {
        "fetched_at": retrieved_at,
        "articles": deduped[:GOLD_NEWS_MAX_STORED],
        "provider_status": provider_status,
        "optional_keys_missing": missing_keys,
        "notes": notes[:6],
    }


def _load_gold_news():
    if _gold_mem["cache"] is None:
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                c = json.load(f)
            gn = c.get("gold_news") if isinstance(c, dict) else None
            if isinstance(gn, dict) and isinstance(gn.get("articles"), list) and gn.get("fetched_at"):
                _gold_mem["cache"] = gn
        except Exception:
            pass
    return _gold_mem["cache"]


def _do_gold_refresh_locked(old_gold, force=False):
    try:
        new_gn = _refresh_gold_news(old_gold, force=force)
        with _lock:
            cal = _mem["cache"]
        _save_cache_atomic(cal, gold_news=new_gn)
        with _lock:
            _gold_mem.update(cache=new_gn, last_ok=True, error="", refreshing=False)
        return new_gn
    except Exception as e:
        err = str(e)[:90] if isinstance(e, _Fail) else type(e).__name__
        with _lock:
            _gold_mem.update(last_ok=False, error=err, refreshing=False)
        return None


def get_gold_news_state(force=False, non_blocking=False):
    """
    Return {status: READY/STALE/UNAVAILABLE, articles, fetched_at, error, provider_status, optional_keys_missing, notes}.
    Terisolasi dari US Economic Calendar & tidak pernah mengembalikan berita palsu saat provider gagal.
    """
    try:
        now = _now_utc()
        with _lock:
            cache = _load_gold_news()
            age = (now - datetime.fromisoformat(cache["fetched_at"])).total_seconds() if cache else None
            need = force or cache is None or age is None or age > GOLD_NEWS_TTL
            can_attempt = (
                force
                or (time.time() - _gold_mem["last_attempt"] >= GOLD_NEWS_FAIL_BACKOFF)
                or (_gold_mem["last_ok"] and age is not None and age > GOLD_NEWS_TTL)
            )
            should_spawn = need and can_attempt and not _gold_mem["refreshing"]
            if should_spawn:
                _gold_mem["last_attempt"] = time.time()
                _gold_mem["refreshing"] = True

        if should_spawn:
            if non_blocking and not force:
                t = threading.Thread(target=_do_gold_refresh_locked, args=(cache, False), daemon=True)
                t.start()
            else:
                new_g = _do_gold_refresh_locked(cache, force=force)
                if new_g is not None:
                    cache = new_g

        with _lock:
            cache = _gold_mem["cache"]
            last_err = _gold_mem["error"]

        missing_keys = [env_var for _, env_var in OPTIONAL_NEWS_API_KEYS if not _env_key(env_var)]
        if not cache:
            return {
                "status": "UNAVAILABLE",
                "articles": [],
                "fetched_at": None,
                "error": last_err or "gold news provider unavailable / no valid cache",
                "provider_status": {},
                "optional_keys_missing": missing_keys,
                "notes": [],
            }

        age = (now - datetime.fromisoformat(cache["fetched_at"])).total_seconds()
        if age > GOLD_NEWS_MAX_STALE_SECONDS:
            return {
                "status": "UNAVAILABLE",
                "articles": [],
                "fetched_at": cache["fetched_at"],
                "error": last_err or f"gold news cache expired ({int(age // 3600)}h old); refresh required",
                "provider_status": cache.get("provider_status") or {},
                "optional_keys_missing": cache.get("optional_keys_missing") or missing_keys,
                "notes": cache.get("notes") or [],
            }

        fresh = (age <= GOLD_NEWS_TTL) or (not last_err and age <= GOLD_NEWS_TTL * 2)
        status = "READY" if fresh else "STALE"

        # Hitung ulang freshness & validasi relevansi secara dinamis agar artikel tidak relevan/stale tidak disajikan
        live_articles = []
        for a in cache.get("articles") or []:
            if not isinstance(a, dict) or not a.get("headline"):
                continue
            item = dict(a)
            f_lbl = _article_freshness(item, now, cache_status=status)
            if f_lbl == "STALE (>72h)":
                continue
            if _score_and_classify_article(item) is None:
                continue
            item["freshness"] = f_lbl
            live_articles.append(item)

        return {
            "status": status,
            "articles": live_articles,
            "fetched_at": cache["fetched_at"],
            "error": last_err,
            "provider_status": cache.get("provider_status") or {},
            "optional_keys_missing": cache.get("optional_keys_missing") or missing_keys,
            "notes": cache.get("notes") or [],
        }
    except Exception as e:
        return {
            "status": "UNAVAILABLE",
            "articles": [],
            "fetched_at": None,
            "error": type(e).__name__,
            "provider_status": {},
            "optional_keys_missing": [env_var for _, env_var in OPTIONAL_NEWS_API_KEYS if not _env_key(env_var)],
            "notes": [],
        }


# ------------------------------------------------------------------ analysis (deterministik)
def _fmt(v, unit):
    if v is None:
        return "N/A"
    if unit == "K":
        return f"{v:+,.0f}K"
    if unit == "":
        return f"{v:,.0f}"
    return f"{v:.2f}{unit}"


def _ev_status(ev, now):
    """Bedakan eksplisit: SCHEDULED, PENDING RELEASE, DELAYED, RELEASED, REVISED, UNVERIFIED."""
    if not _is_verified(ev):
        return "UNVERIFIED"
    if ev.get("actual") is not None:
        return "REVISED" if ev.get("revised") else "RELEASED"
    dt = datetime.fromisoformat(ev["dt_utc"])
    delta_min = (dt - now).total_seconds() / 60.0
    if delta_min > 0:
        return "SCHEDULED"
    if ev.get("series") is None:
        return "POST-RELEASE" if -delta_min <= POST_RELEASE_WINDOW_MIN else "RELEASED"
    return "DELAYED" if -delta_min > DELAYED_AFTER_MIN else "PENDING RELEASE"


def analyze_detail(ev):
    """
    Analisis dampak makro deterministik pasca-rilis (tanpa klaim reaksi harga aktual).
    Memeriksa verifikasi, satuan, periode, aturan spesifik indikator, dan revisi previous.
    """
    base = {
        "xau_impact": "UNCONFIRMED",
        "usd_impact": "UNCONFIRMED",
        "yield_impact": "UNCONFIRMED",
        "surprise": None,
        "unit": ev.get("unit", ""),
        "period": ev.get("period"),
        "reason": "",
    }
    if not _is_verified(ev):
        base["reason"] = "event unverified"
        return base
    actual = ev.get("actual")
    if actual is None or not isinstance(actual, (int, float)) or not math.isfinite(actual):
        base["reason"] = "belum dirilis / actual belum terverifikasi"
        return base
    ind = _IND_BY_KEY.get(ev.get("key"))
    if not ind or ind[8] is None:
        base["reason"] = "indikator non-numerik"
        return base
    if ev.get("unit") != ind[6] or not ev.get("period"):
        base["reason"] = "satuan atau periode indikator tidak lengkap"
        return base
    forecast = ev.get("forecast")
    if forecast is None or not isinstance(forecast, (int, float)) or not math.isfinite(forecast):
        base["reason"] = "forecast N/A (tidak ada sumber konsensus forecast terverifikasi)"
        return base

    band = ind[8]
    hot_bearish = bool(ind[9])
    s = round(float(actual) - float(forecast), 4)
    base["surprise"] = s

    if abs(s) < band:
        base.update(
            xau_impact="NEUTRAL",
            usd_impact="NEUTRAL",
            yield_impact="NEUTRAL",
            reason=f"surprise ({s:+.2f}{ind[6]}) dalam band netral (+/-{band}{ind[6]}); tanpa data reaksi harga",
        )
        return base

    hot = (s > 0) == hot_bearish
    if ev.get("revised") and ev.get("previous_original") is not None and ev.get("previous") is not None:
        rev = float(ev["previous"]) - float(ev["previous_original"])
        if abs(rev) >= band and ((rev > 0) != (s > 0)):
            base.update(
                xau_impact="MIXED",
                usd_impact="MIXED",
                yield_impact="MIXED",
                reason="surprise actual berlawanan arah dengan revisi previous; tanpa data reaksi harga",
            )
            return base

    if hot:
        base.update(
            xau_impact="POTENTIAL BEARISH",
            usd_impact="POTENTIAL UP",
            yield_impact="POTENTIAL UP",
            reason="surprise cenderung hawkish/kuat -> potensi USD & yield naik, emas tertekan (tanpa klaim reaksi aktual)",
        )
    else:
        base.update(
            xau_impact="POTENTIAL BULLISH",
            usd_impact="POTENTIAL DOWN",
            yield_impact="POTENTIAL DOWN",
            reason="surprise cenderung dovish/lemah -> potensi USD & yield turun, emas terdukung (tanpa klaim reaksi aktual)",
        )
    return base


def analyze(ev):
    """Kompatibel dengan API lama: return (label, alasan, surprise)."""
    d = analyze_detail(ev)
    return d["xau_impact"], d["reason"], d["surprise"]


def _risk_for_event(ev, now=None):
    if not _is_verified(ev):
        return "UNVERIFIED"
    if ev["impact"] == "HIGH":
        return "HIGH VOLATILITY"
    if ev["impact"] == "MEDIUM":
        return "MODERATE VOLATILITY"
    return "LOW"


def news_risk(st, now=None):
    """Hitung tingkat risiko berita hanya dari event VERIFIED (UNVERIFIED tidak pernah dipromosikan)."""
    now = now or _now_utc()
    if st["status"] == "UNAVAILABLE":
        return "UNKNOWN"
    lvl = "LOW"
    for ev in st["events"]:
        if not _is_verified(ev):
            continue
        d = (datetime.fromisoformat(ev["dt_utc"]) - now).total_seconds() / 60.0
        if ev["impact"] == "HIGH":
            if -POST_RELEASE_WINDOW_MIN <= d <= 120:
                return "HIGH"
            if 0 < d <= 1440:
                lvl = "MEDIUM"
        elif ev["impact"] == "MEDIUM" and -POST_RELEASE_WINDOW_MIN <= d <= 120 and lvl == "LOW":
            lvl = "MEDIUM"
    return lvl


def _next_event(st, now, high_only=False):
    """
    Pilih event relevan berikutnya secara deterministik.
    Event UNVERIFIED tidak pernah dipromosikan sebagai VERIFIED HIGH-IMPACT NEWS.
    """
    if st["status"] == "UNAVAILABLE":
        return None
    for ev in st["events"]:
        if not _is_verified(ev):
            continue
        if high_only and (ev["impact"] != "HIGH" or st["status"] != "READY"):
            continue
        dt = datetime.fromisoformat(ev["dt_utc"])
        delta_min = (dt - now).total_seconds() / 60.0
        if delta_min >= 0:
            return ev
        if -POST_RELEASE_WINDOW_MIN <= delta_min < 0:
            if ev.get("series") is None or ev.get("actual") is None:
                return ev
    return None


def _countdown(ev, now):
    s = int((datetime.fromisoformat(ev["dt_utc"]) - now).total_seconds())
    if s < 0:
        elapsed_m = (-s) // 60
        if ev.get("actual") is not None:
            return "RELEASED"
        if ev.get("series") is None:
            return f"T+{elapsed_m}m" if elapsed_m <= POST_RELEASE_WINDOW_MIN else "RELEASED"
        return f"T+{elapsed_m}m PENDING" if elapsed_m <= DELAYED_AFTER_MIN else "DELAYED"
    if s < 60:
        return f"in <1m"
    d, r = divmod(s, 86400)
    h, r = divmod(r, 3600)
    return (f"in {d}d " if d else "in ") + f"{h:02d}h {r // 60:02d}m"


def _short(ev):
    name = str(ev.get("name") or "")
    return name.replace("US ", "", 1).split(" (")[0]


def _wib_str(ev):
    return _wib(datetime.fromisoformat(ev["dt_utc"])).strftime("%a %d-%b %H:%M WIB")


def _wib_hhmm(ev):
    return _wib(datetime.fromisoformat(ev["dt_utc"])).strftime("%H:%M WIB")


def _source_tag(ev, st):
    if st["status"] == "STALE":
        return "STALE (UNVERIFIED LIVE)"
    if not _is_verified(ev):
        return "UNVERIFIED"
    agency = ev.get("official_agency") or ev.get("source") or "OFFICIAL"
    return f"VERIFIED ({agency})"


def _iso_wib_str(iso_str, fmt="%Y-%m-%d %H:%M WIB"):
    if not iso_str:
        return "N/A"
    try:
        dt = datetime.fromisoformat(str(iso_str))
        if TZ_WIB is not None:
            return _wib(dt).strftime(fmt)
        return dt.strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return "N/A"


# ------------------------------------------------------------------ shared Smart News Alert Engine
class NewsAlertEngine:
    """
    Shared US News & Gold Market News Alert Engine untuk seluruh mode monitoring
    (`pantau`, `pantau detail`, `pantau smc`, `pantau smc detail`, `pantau snr`,
    `pantau snr detail`, `scan gabungan`).

    - Independen dari BUY/SELL/WAIT dan zone touch.
    - Non-blocking dan di-throttle agar tidak memanggil API setiap tick.
    - Deduplikasi kalender per (event_id, stage): T-60 -> T-30 -> T-15 -> T-5 -> T+0..T+30 -> actual.
    - Deduplikasi artikel emas per canonical_url / article_id + deteksi pembaruan material.
    """

    def __init__(self, thresholds=None):
        cfg = dict(ALERT_THRESHOLDS)
        if isinstance(thresholds, dict):
            cfg.update(thresholds)
        self.thresholds = cfg
        self._emitted = {}             # event_id -> stage signature terakhir
        self._last_unavail_msg = None  # dedup alert DATA UNAVAILABLE
        self._last_poll_mono = 0.0
        self._cached_state = None
        self._seen_gold = {}           # canonical_url/article_id -> (updated_at, norm_title, sentiment_dir)
        self._gold_initialized = False
        self._last_gold_poll_mono = 0.0
        self._cached_gold_state = None

    def reset(self):
        self._emitted.clear()
        self._last_unavail_msg = None
        self._last_poll_mono = 0.0
        self._cached_state = None
        self._seen_gold.clear()
        self._gold_initialized = False
        self._last_gold_poll_mono = 0.0
        self._cached_gold_state = None

    def _stage_for(self, ev, now):
        """Return (stage_code, stage_label, delta_min) atau (None, None, delta_min)."""
        if not _is_verified(ev):
            return None, None, None
        if ev.get("impact") not in ("HIGH", "MEDIUM"):
            return None, None, None
        dt = datetime.fromisoformat(ev["dt_utc"])
        delta_min = (dt - now).total_seconds() / 60.0
        th = self.thresholds
        if th["caution_min"] < delta_min <= th["upcoming_min"]:
            return "T60_UPCOMING", "UPCOMING", delta_min
        if th["high_alert_min"] < delta_min <= th["caution_min"]:
            return "T30_CAUTION", "CAUTION", delta_min
        if th["imminent_min"] < delta_min <= th["high_alert_min"]:
            return "T15_HIGH_ALERT", "HIGH ALERT", delta_min
        if 0 < delta_min <= th["imminent_min"]:
            return "T5_IMMINENT", "RELEASE IMMINENT", delta_min
        if -th["post_release_min"] <= delta_min <= 0:
            if ev.get("actual") is not None:
                return "POST_ACTUAL", "POST-RELEASE UPDATE", delta_min
            return "POST_VOLATILITY", "POST-RELEASE VOLATILITY", delta_min
        return None, None, delta_min

    @staticmethod
    def _format_alert(ev, st, stage_code, stage_label, delta_min):
        short_name = _short(ev)
        src = _source_tag(ev, st)
        if stage_code == "POST_ACTUAL":
            u = ev.get("unit", "")
            act_s = _fmt(ev.get("actual"), u)
            fc_s = _fmt(ev.get("forecast"), u)
            prev_s = _fmt(ev.get("previous"), u)
            rev_tag = " [REVISED PREV]" if ev.get("revised") else ""
            d = analyze_detail(ev)
            imp_extra = ""
            if d["usd_impact"] != "UNCONFIRMED":
                imp_extra = f" | USD: {d['usd_impact']} | YIELD: {d['yield_impact']}"
            return (
                f"[US NEWS UPDATE] {short_name} | ACTUAL {act_s} vs FORECAST {fc_s} (PREV {prev_s}){rev_tag}\n"
                f"XAUUSD IMPACT : {d['xau_impact']}{imp_extra} | SOURCE: {src}"
            )

        if delta_min >= 0:
            t_disp = f"T-{max(1, int(math.ceil(delta_min)))}m"
        else:
            t_disp = f"T+{int(abs(delta_min))}m"
        risk_lbl = _risk_for_event(ev)
        return (
            f"[US NEWS ALERT] {ev['impact']} IMPACT | {short_name} | {t_disp} ({stage_label})\n"
            f"RELEASE : {_wib_hhmm(ev)} | SOURCE: {src}\n"
            f"XAUUSD RISK : {risk_lbl}"
        )

    @staticmethod
    def _format_gold_alert(art, is_update=False):
        tag = "UPDATED" if is_update else "NEW"
        topic = str(art.get("topic") or "Gold Market")
        sent_dir = str((art.get("sentiment") or {}).get("direction") or "UNCONFIRMED")
        hl = str(art.get("headline") or "").strip()
        if len(hl) > 95:
            hl = hl[:92].rstrip() + "..."
        pub = str(art.get("publisher") or "UNKNOWN")
        t_str = _iso_wib_str(art.get("updated_at") or art.get("published_at"), fmt="%d-%b %H:%M WIB")
        return (
            f"[GOLD NEWS] {tag} | {topic} | Potential Impact: {sent_dir}\n"
            f"  {hl} | {pub} ({t_str})"
        )

    def _get_throttled_state(self, now):
        mono = time.monotonic()
        poll_sec = self.thresholds.get("min_poll_interval_sec", 30)
        if self._cached_state is not None:
            # Jika background refresh baru saja mengisi _mem["cache"], langsung serap tanpa memanggil HTTP
            with _lock:
                mem_c = _mem["cache"]
                mem_ft = mem_c.get("fetched_at") if isinstance(mem_c, dict) else None
            if mem_ft and mem_ft != self._cached_state.get("fetched_at"):
                self._last_poll_mono = mono
                self._cached_state = get_state(non_blocking=True)
                return self._cached_state
            # Cek apakah ada event dalam jendela post-release -> gunakan interval post-release
            for ev in self._cached_state.get("events", []):
                try:
                    d_min = (datetime.fromisoformat(ev["dt_utc"]) - now).total_seconds() / 60.0
                    if -self.thresholds["post_release_min"] <= d_min <= 5 and ev.get("series") and ev.get("actual") is None:
                        poll_sec = min(poll_sec, self.thresholds.get("post_release_poll_interval_sec", 60))
                        break
                except Exception:
                    continue
            if (mono - self._last_poll_mono) < poll_sec:
                return self._cached_state
        self._last_poll_mono = mono
        self._cached_state = get_state(non_blocking=True)
        return self._cached_state

    def _get_throttled_gold_state(self, now):
        mono = time.monotonic()
        poll_sec = self.thresholds.get("gold_news_poll_interval_sec", 180)
        if self._cached_gold_state is not None:
            with _lock:
                mem_g = _gold_mem["cache"]
                mem_ft = mem_g.get("fetched_at") if isinstance(mem_g, dict) else None
            if mem_ft and mem_ft != self._cached_gold_state.get("fetched_at"):
                self._last_gold_poll_mono = mono
                self._cached_gold_state = get_gold_news_state(non_blocking=True)
                return self._cached_gold_state
            if (mono - self._last_gold_poll_mono) < poll_sec:
                return self._cached_gold_state
        self._last_gold_poll_mono = mono
        self._cached_gold_state = get_gold_news_state(non_blocking=True)
        return self._cached_gold_state

    def _evaluate_gold_news(self, now, gst):
        """Evaluasi Gold Market News secara terisolasi & ter-deduplikasi."""
        g_status = (gst or {}).get("status", "UNAVAILABLE")
        arts = list((gst or {}).get("articles") or [])
        if g_status == "UNAVAILABLE":
            return {
                "gold_news_status": "UNAVAILABLE",
                "gold_news_summary": "GOLD NEWS : UNAVAILABLE",
                "gold_articles": [],
                "gold_alerts": [],
                "gold_active_alerts": [],
            }

        top_arts = _select_top_gold_articles(arts, limit=5, now=now)
        if g_status == "STALE":
            summary = f"GOLD NEWS : {len(arts)} headlines (STALE CACHE)"
            # Jangan promosikan cache STALE sebagai alert headline baru
            return {
                "gold_news_status": "STALE",
                "gold_news_summary": summary,
                "gold_articles": top_arts,
                "gold_alerts": [],
                "gold_active_alerts": [],
            }

        if top_arts:
            t0 = top_arts[0]
            s0 = (t0.get("sentiment") or {}).get("direction") or "UNCONFIRMED"
            summary = f"GOLD NEWS : {len(arts)} headlines | Latest: {t0.get('topic', 'Gold')} ({s0})"
        else:
            summary = "GOLD NEWS : 0 relevant headlines"

        gold_new_alerts = []
        gold_active_alerts = []

        # Kandidat alert: hanya VERIFIED, relevan, dan cukup baru (<= GOLD_NEWS_ALERT_MAX_AGE_HOURS)
        fresh_candidates = []
        for a in arts:
            if a.get("integrity_status") != "VERIFIED":
                continue
            if int(a.get("relevance_score") or 0) < GOLD_NEWS_ALERT_MIN_RELEVANCE:
                continue
            pub_ts = a.get("updated_at") or a.get("published_at")
            if not pub_ts:
                continue
            try:
                age_h = (now - datetime.fromisoformat(pub_ts)).total_seconds() / 3600.0
            except Exception:
                continue
            if age_h > GOLD_NEWS_ALERT_MAX_AGE_HOURS:
                continue
            fresh_candidates.append(a)
        fresh_candidates.sort(
            key=lambda a: (
                str(a.get("updated_at") or a.get("published_at") or ""),
                -int(a.get("credibility_tier") or 3),
                int(a.get("relevance_score") or 0),
            ),
            reverse=True,
        )

        if not self._gold_initialized:
            # Pada siklus pertama, tampilkan maks GOLD_NEWS_MAX_ALERTS_PER_CYCLE berita paling baru/relevan
            # dan tandai seluruh artikel saat ini sebagai seen agar tidak spam pada siklus berikutnya.
            for a in fresh_candidates[:GOLD_NEWS_MAX_ALERTS_PER_CYCLE]:
                msg = self._format_gold_alert(a, is_update=False)
                gold_new_alerts.append(msg)
                gold_active_alerts.append(msg)
            for a in arts:
                dk = a.get("canonical_url") or a.get("article_id")
                if dk:
                    sd = (a.get("sentiment") or {}).get("direction") or "UNCONFIRMED"
                    self._seen_gold[dk] = (a.get("updated_at"), _normalize_headline_key(a.get("headline")), sd)
            self._gold_initialized = True
        else:
            emitted_this_cycle = 0
            for a in fresh_candidates:
                dk = a.get("canonical_url") or a.get("article_id")
                if not dk:
                    continue
                sd = (a.get("sentiment") or {}).get("direction") or "UNCONFIRMED"
                sig = (a.get("updated_at"), _normalize_headline_key(a.get("headline")), sd)
                prev_sig = self._seen_gold.get(dk)
                if prev_sig is None:
                    self._seen_gold[dk] = sig
                    if emitted_this_cycle < GOLD_NEWS_MAX_ALERTS_PER_CYCLE:
                        msg = self._format_gold_alert(a, is_update=False)
                        gold_new_alerts.append(msg)
                        gold_active_alerts.append(msg)
                        emitted_this_cycle += 1
                else:
                    # Deteksi pembaruan material: updated_at baru DAN judul atau arah sentimen berubah
                    upd_newer = bool(sig[0] and (not prev_sig[0] or sig[0] > prev_sig[0]))
                    material_change = (sig[1] != prev_sig[1]) or (sig[2] != prev_sig[2])
                    if upd_newer and material_change:
                        self._seen_gold[dk] = sig
                        if emitted_this_cycle < GOLD_NEWS_MAX_ALERTS_PER_CYCLE:
                            msg = self._format_gold_alert(a, is_update=True)
                            gold_new_alerts.append(msg)
                            gold_active_alerts.append(msg)
                            emitted_this_cycle += 1
            # Tandai artikel non-alert yang sudah masuk cache agar tidak memicu false NEW di siklus berikutnya
            for a in arts:
                dk = a.get("canonical_url") or a.get("article_id")
                if dk and dk not in self._seen_gold:
                    sd = (a.get("sentiment") or {}).get("direction") or "UNCONFIRMED"
                    self._seen_gold[dk] = (a.get("updated_at"), _normalize_headline_key(a.get("headline")), sd)
            if len(self._seen_gold) > 250:
                for k in list(self._seen_gold.keys())[:-200]:
                    self._seen_gold.pop(k, None)

        return {
            "gold_news_status": g_status,
            "gold_news_summary": summary,
            "gold_articles": top_arts,
            "gold_alerts": gold_new_alerts,
            "gold_active_alerts": gold_active_alerts,
        }

    def evaluate(self, now=None, state=None, gold_state=None):
        """
        Evaluasi status US Economic Calendar & Gold Market News dan hasilkan alert baru
        (kalender: hanya saat stage berubah; headline emas: hanya berita baru/pembaruan material).
        """
        now = now or _now_utc()
        st = state if isinstance(state, dict) else self._get_throttled_state(now)
        if isinstance(gold_state, dict):
            gst = gold_state
        elif state is None:
            gst = self._get_throttled_gold_state(now)
        elif isinstance(state.get("gold_news"), dict):
            gst = state["gold_news"]
        else:
            gst = {"status": "UNAVAILABLE", "articles": []}

        gold_eval = self._evaluate_gold_news(now, gst)
        status = st.get("status", "UNAVAILABLE")

        if status == "UNAVAILABLE":
            err = st.get("error") or "provider unavailable"
            summary = "NEWS RISK : UNKNOWN | NEXT EVENT : UNAVAILABLE"
            new_alerts = []
            if self._last_unavail_msg != err:
                new_alerts.append(
                    f"[US NEWS ALERT] DATA UNAVAILABLE | {err}\n"
                    f"NEWS RISK : UNKNOWN | NEXT EVENT : UNAVAILABLE"
                )
                self._last_unavail_msg = err
            combined_new = new_alerts + gold_eval["gold_alerts"]
            combined_active = list(new_alerts) + gold_eval["gold_active_alerts"]
            return {
                "status": "UNAVAILABLE",
                "risk": "UNKNOWN",
                "summary": summary,
                "next_event": None,
                "alerts": combined_new,
                "active_alerts": combined_active,
                **gold_eval,
            }

        self._last_unavail_msg = None
        risk = news_risk(st, now)
        nxt = _next_event(st, now)
        stale_sfx = " (STALE)" if status == "STALE" else ""
        if nxt:
            summary = f"NEWS RISK : {risk}{stale_sfx} | NEXT EVENT : {_short(nxt)} ({_wib_str(nxt)}, {_countdown(nxt, now)})"
        else:
            summary = f"NEWS RISK : {risk}{stale_sfx} | NEXT EVENT : NONE"

        new_alerts = []
        active_alerts = []
        active_ids = set()

        for ev in st.get("events", []):
            stage_code, stage_label, delta_min = self._stage_for(ev, now)
            if not stage_code:
                continue
            # Jangan promosikan cache STALE sebagai verified live alert
            if status == "STALE":
                continue
            eid = ev["id"]
            active_ids.add(eid)
            sig = (stage_code, ev.get("actual"), ev.get("forecast"), bool(ev.get("revised")))
            msg = self._format_alert(ev, st, stage_code, stage_label, delta_min)
            active_alerts.append(msg)
            if self._emitted.get(eid) != sig:
                self._emitted[eid] = sig
                new_alerts.append(msg)

        # Bersihkan tracking event yang sudah melewati jendela T+30m
        for old_id in list(self._emitted.keys()):
            if old_id not in active_ids:
                self._emitted.pop(old_id, None)

        return {
            "status": status,
            "risk": risk,
            "summary": summary,
            "next_event": nxt,
            "alerts": new_alerts + gold_eval["gold_alerts"],
            "active_alerts": active_alerts + gold_eval["gold_active_alerts"],
            **gold_eval,
        }


_shared_alert_engine = NewsAlertEngine()


def get_shared_alert_engine():
    return _shared_alert_engine


# ------------------------------------------------------------------ formatting
def startup_lines():
    try:
        st = get_state()
        gst = get_gold_news_state(non_blocking=True)
        now = _now_utc()
        src_st = st.get("source_status") or {}
        src_note = ""
        if st["status"] != "UNAVAILABLE" and src_st:
            src_note = f" (FED:{src_st.get('FED', '?')} | FRED:{src_st.get('FRED', '?')})"
        out = [f"US NEWS RADAR : {st['status']}{src_note}"]
        ev = _next_event(st, now, high_only=True)
        if ev:
            out.append(
                f"NEXT HIGH IMPACT : {_short(ev)} | {_wib_str(ev)} | {_countdown(ev, now)} | {_source_tag(ev, st)}"
            )
        elif st["status"] != "UNAVAILABLE":
            out.append("NEXT HIGH IMPACT : NONE")
        else:
            out.append(f"NEWS SETUP : {st['error']} (lihat modul us_news_radar)")

        g_st = gst.get("status", "UNAVAILABLE")
        g_cnt = len(gst.get("articles") or [])
        if g_st == "READY":
            out.append(f"GOLD MARKET NEWS : READY ({g_cnt} headlines; ketik 'news' / 'news detail')")
        elif g_st == "STALE":
            out.append(f"GOLD MARKET NEWS : STALE ({g_cnt} cached; ketik 'news refresh')")
        else:
            out.append("GOLD MARKET NEWS : LOADING / UNAVAILABLE (ketik 'news refresh')")
        return out
    except Exception:
        return ["US NEWS RADAR : UNAVAILABLE"]


def context_lines():
    """Ringkasan berita ringkas (1-3 baris) untuk command scan. Informasional saja."""
    try:
        st = get_state(non_blocking=True)
        gst = get_gold_news_state(non_blocking=True)
        now = _now_utc()
        ev_eval = NewsAlertEngine().evaluate(now=now, state=st, gold_state=gst)
        lines = []
        if st["status"] == "UNAVAILABLE":
            lines.append(f"NEWS RISK : UNKNOWN | DATA UNAVAILABLE ({st.get('error') or 'provider error'})")
        elif ev_eval["active_alerts"]:
            lines.append(ev_eval["active_alerts"][0])
        else:
            lines.append(ev_eval["summary"])
        g_sum = ev_eval.get("gold_news_summary")
        if g_sum:
            lines.append(g_sum)
        return lines
    except Exception:
        return ["NEWS RISK : UNKNOWN | NEXT EVENT : UNAVAILABLE"]


def _block(ev, st, now, detail=False):
    stat = _ev_status(ev, now)
    stale = " | STALE DATA" if st["status"] == "STALE" else ""
    u = ev["unit"]
    d = analyze_detail(ev)
    act = _fmt(ev["actual"], u) if ev.get("actual") is not None else (
        "N/A (non-numeric)" if ev.get("series") is None else stat
    )
    L = [
        "-" * 40,
        f"TIME WIB       : {_wib(datetime.fromisoformat(ev['dt_utc'])).strftime('%Y-%m-%d %H:%M WIB')}",
        f"EVENT          : {ev['name']} [{ev.get('event_type', ev['key'])}]",
        f"IMPACT         : {ev['impact']} | VERIFICATION: {ev.get('verification', 'UNVERIFIED')}",
        f"PREVIOUS       : {_fmt(ev.get('previous'), u) if ev.get('actual') is not None else 'N/A'}"
        + (f" (REVISED from {_fmt(ev['previous_original'], u)})" if ev.get("revised") else ""),
        f"FORECAST       : {_fmt(ev.get('forecast'), u)}",
        f"ACTUAL         : {act}",
    ]
    if ev.get("actual") is not None:
        L.append(f"SURPRISE       : {('%+.2f' % d['surprise']) + u if d['surprise'] is not None else 'N/A (no forecast)'}")
    L += [
        f"DATA SOURCE    : {ev.get('source', 'UNKNOWN')}",
        f"DATA STATUS    : {stat}{stale}",
        f"XAUUSD IMPACT  : {d['xau_impact']} (USD: {d['usd_impact']} | YIELD: {d['yield_impact']})",
        f"NEWS RISK      : {_risk_for_event(ev, now)}",
    ]
    if detail:
        t_ver = "OFFICIAL timestamp" if ev.get("time_verification") == "OFFICIAL" else "TIME rule-based (standar resmi)"
        L += [
            f"EVENT ID       : {ev.get('official_id') or ev['id']}",
            f"REF PERIOD     : {ev.get('period') or 'N/A'}  (series {ev.get('series') or '-'})",
            f"RELEASE TIME   : {ev['date_et']} {ev['time_et']} {ev.get('tz_et', 'ET')} -> WIB (DST otomatis)",
            f"VALIDATION     : {ev.get('verification', 'UNVERIFIED')} | {t_ver}"
            + (" | ACTUAL verified by vintage" if ev.get("actual") is not None else ""),
            f"ANALYSIS       : {d['reason']}",
        ]
    return L


def _gold_article_block(art, idx, gst_status, detail=False):
    sent = art.get("sentiment") if isinstance(art.get("sentiment"), dict) else analyze_gold_article_sentiment(art)
    s_dir = sent.get("direction", "UNCONFIRMED")
    pub_wib = _iso_wib_str(art.get("published_at"))
    upd_wib = _iso_wib_str(art.get("updated_at")) if art.get("updated_at") else "- (tidak ada pembaruan terpisah)"
    ret_wib = _iso_wib_str(art.get("retrieved_at"))
    fresh = art.get("freshness") or ("STALE_CACHE" if gst_status == "STALE" else "UNKNOWN")
    integ = art.get("integrity_status") or "UNVERIFIED"
    imp = art.get("impact_level") or "LOW"
    topic = art.get("topic") or "Gold Market"
    pub = art.get("publisher") or "UNKNOWN"
    url = art.get("url") or "N/A"
    hl = art.get("headline") or "(no headline)"

    if not detail:
        upd_tag = f" | UPD {upd_wib}" if art.get("updated_at") else ""
        return [
            "-" * 40,
            f"[#{idx}] {pub_wib}{upd_tag} | {fresh}",
            f"HEADLINE       : {hl}",
            f"TOPIC / IMPACT : {topic} | Impact: {imp} | Sentiment: {s_dir}",
            f"PUBLISHER      : {pub} ({art.get('provider', 'FEED')}) | Lang: {art.get('language', 'unknown')} | {integ}",
            f"URL            : {url}",
        ]

    topics_all = ", ".join(art.get("topics") or [topic])
    drivers_txt = "; ".join(sent.get("drivers") or []) or "tidak ada pendorong terarah terverifikasi"
    excerpt = str(art.get("summary") or "").strip()
    if len(excerpt) > 240:
        excerpt = excerpt[:237].rstrip() + "..."
    int_notes = ", ".join(art.get("integrity_notes") or [])
    integ_disp = f"{integ} ({int_notes})" if int_notes else integ
    return [
        "-" * 40,
        f"[#{idx}] HEADLINE : {hl}",
        f"PUBLISHER      : {pub} (Provider: {art.get('provider', 'FEED')} | Tier {art.get('credibility_tier', 3)})",
        f"URL            : {url}",
        f"PUBLISHED      : {pub_wib} ({art.get('published_at') or 'N/A'})",
        f"UPDATED        : {upd_wib}",
        f"RETRIEVED      : {ret_wib}",
        f"INTEGRITY      : {integ_disp} | LANG: {art.get('language', 'unknown')} | FRESHNESS: {fresh}",
        f"TOPIC / IMPACT : {topic} (All: {topics_all}) | Impact: {imp} | Relevance: {art.get('relevance_score', 0)}",
        f"SENTIMENT      : {s_dir} (Confidence: {sent.get('confidence', 'LOW')} | Basis: {sent.get('evidence_basis', 'INSUFFICIENT')} | Priced-in: {sent.get('priced_in_risk', 'UNKNOWN')})",
        f"DRIVERS        : {drivers_txt}",
        f"ANALYSIS       : {sent.get('rationale', 'N/A')}",
        f"EXCERPT        : {excerpt or 'N/A (snippet tidak disediakan provider)'}",
    ]


def _header(st, gst=None):
    ft = _iso_wib_str(st.get("fetched_at")) if st.get("fetched_at") else "-"
    src_st = st.get("source_status") or {}
    src_line = f"FED={src_st.get('FED', 'N/A')} | FRED={src_st.get('FRED', 'N/A')}"
    L = [
        "XAUUSD US FUNDAMENTAL & GOLD NEWS RADAR",
        f"CALENDAR STATUS: {st['status']}  | LAST FETCH: {ft} | SOURCES: {src_line}",
    ]
    if isinstance(gst, dict):
        gft = _iso_wib_str(gst.get("fetched_at")) if gst.get("fetched_at") else "-"
        p_st = gst.get("provider_status") or {}
        ok_cnt = sum(1 for v in p_st.values() if v == "OK")
        tot_cnt = len(p_st)
        p_note = f"{ok_cnt}/{tot_cnt} providers OK" if tot_cnt else "no providers polled"
        L.append(f"GOLD NEWS STAT : {gst.get('status', 'UNAVAILABLE')}  | LAST FETCH: {gft} | {p_note} | {len(gst.get('articles') or [])} relevant")
    L.append("NOTE           : verified official/RSS/API data only (no scraping/paywall bypass); informational only")
    if st["status"] == "STALE":
        L.append("WARNING        : CALENDAR STALE DATA - jangan dianggap jadwal live terverifikasi")
    if st["status"] == "UNAVAILABLE":
        L.append(f"CAL UNAVAILABLE: {st['error']} -> cek koneksi / FRED_API_KEY di .env")
    if isinstance(gst, dict) and gst.get("status") == "STALE":
        L.append("WARNING        : GOLD NEWS CACHE STALE - jangan mengklaim ini headline terbaru")
    if isinstance(gst, dict) and gst.get("status") == "UNAVAILABLE":
        L.append(f"NEWS UNAVAIL   : {gst.get('error') or 'provider unavailable'} -> cek koneksi / API key opsional di .env")
    if st.get("notes"):
        L.append(f"CAL NOTES      : {'; '.join(st['notes'][:3])}")
    if isinstance(gst, dict) and gst.get("notes"):
        L.append(f"NEWS NOTES     : {'; '.join(gst['notes'][:3])}")
    return L


def _build_trade_bias_prompt(upcoming_verified, released_verified, active_signal=None, gold_articles=None):
    """
    Prompt ketat untuk Gemini: HANYA menjelaskan data terverifikasi yang dikirim
    (Kalender Ekonomi AS + Headline Berita Emas terverifikasi).
    DILARANG mengarang jadwal, actual, forecast, berita palsu, atau mengubah sinyal trading.
    """
    up_info = []
    for e in [x for x in (upcoming_verified or []) if _is_verified(x)][:3]:
        up_info.append(
            f"- {e['name']} [{e.get('event_type', e['key'])}] | Jadwal: {_wib_str(e)} | "
            f"Impact: {e['impact']} | Source: {e.get('source')} | Forecast: {_fmt(e.get('forecast'), e.get('unit', ''))}"
        )

    rel_info = []
    for e in [x for x in (released_verified or []) if _is_verified(x) and x.get("actual") is not None][-3:]:
        u = e.get("unit", "")
        d = analyze_detail(e)
        rel_info.append(
            f"- {e['name']} | Period: {e.get('period') or 'N/A'} | Actual: {_fmt(e.get('actual'), u)} | "
            f"Forecast: {_fmt(e.get('forecast'), u)} | Previous: {_fmt(e.get('previous'), u)} | "
            f"Deterministic Impact: {d['xau_impact']} ({d['reason']})"
        )

    art_info = []
    for a in [x for x in (gold_articles or []) if isinstance(x, dict) and x.get("integrity_status") == "VERIFIED"][:5]:
        s = a.get("sentiment") or {}
        snip = str(a.get("summary") or "").strip()[:180]
        art_info.append(
            f"- [{a.get('topic', 'Gold')} | Impact:{a.get('impact_level', 'LOW')} | Sentimen:{s.get('direction', 'UNCONFIRMED')}] "
            f"{a.get('headline')} | Publisher: {a.get('publisher')} | Published: {_iso_wib_str(a.get('published_at'))}"
            + (f" | Excerpt: {snip}" if snip else " | Excerpt: N/A")
        )

    sig_str = "TIDAK ADA SETUP AKTIF"
    if active_signal and isinstance(active_signal, dict) and active_signal.get("lo"):
        sig_str = (
            f"{active_signal.get('arah', 'SETUP')} @ {active_signal.get('lo', 0):.2f} - {active_signal.get('hi', 0):.2f} "
            f"(SL: {active_signal.get('sl', 0):.2f}, TP: {active_signal.get('tp1', 0):.2f})"
        )

    prompt = f"""
Anda adalah Senior XAUUSD Fundamental & Macro Risk Analyst (STRICTLY INFORMATIONAL).
ATURAN WAJIB:
1. HANYA ringkas dan jelaskan data terverifikasi yang tertulis di bawah ini.
2. DILARANG mengarang angka actual, forecast, jadwal rilis, kutipan artikel, atau fakta di luar input.
3. DILARANG mengklaim reaksi harga aktual XAUUSD/USD/yield tanpa data pasar pendukung.
4. Jangan menganggap angka lebih tinggi selalu bearish untuk emas (contoh: Unemployment Rate / Jobless Claims lebih tinggi = potensi bullish emas).
5. Jika konteks berita atau forecast tidak cukup, WAJIB tulis UNCONFIRMED (jangan memaksa arah).
6. DILARANG mengubah atau menggantikan sinyal teknikal SMC/SNR.

[SETUP TEKNIKAL XAUUSD SAAT INI (REFERENSI SAJA)]:
{sig_str}

[1. US ECONOMIC CALENDAR TERVERIFIKASI]:
Event Mendatang:
{chr(10).join(up_info) if up_info else "- NONE"}

Event Baru Rilis (Actual Terverifikasi):
{chr(10).join(rel_info) if rel_info else "- NONE"}

[2. LATEST GOLD MARKET NEWS TERVERIFIKASI]:
{chr(10).join(art_info) if art_info else "- NONE"}

Format Output Ringkas (maks 6 baris):
[RINGKASAN KONTEKS FUNDAMENTAL & BERITA EMAS (VERIFIED DATA ONLY)]:
• Status Data     : (PRE-RELEASE / POST-RELEASE / NEWS-DRIVEN / UNCONFIRMED)
• Tema Utama Emas : (1 kalimat ringkas tema utama dari headline terverifikasi di atas, atau NONE)
• Potensi Dampak  : (POTENTIAL BULLISH / POTENTIAL BEARISH / MIXED / NEUTRAL / UNCONFIRMED — pertimbangkan Fed, USD, yield, inflasi/employment, safe-haven, priced-in)
• Catatan Risiko  : (1 kalimat risiko volatilitas berdasarkan data di atas; tanpa mengubah sinyal SMC/SNR)
"""
    return prompt.strip()


def _ai_text(events, ask_ai, active_signal=None, upcoming=None, gold_articles=None):
    """Gemini hanya menjelaskan data terverifikasi (tidak pernah menjadi sumber data)."""
    try:
        p = _build_trade_bias_prompt(upcoming or [], events or [], active_signal, gold_articles=gold_articles)
        txt = str(ask_ai(p) or "").strip()[:1200]
        return "\n" + txt if txt else ""
    except Exception as e:
        return f"\n[RINGKASAN FUNDAMENTAL]: Tidak tersedia ({type(e).__name__})"


def _news_text(detail, ask_ai, active_signal=None):
    st = get_state()
    gst = get_gold_news_state()
    now = _now_utc()
    L = _header(st, gst=gst)

    # --- SECTION 1: US ECONOMIC CALENDAR ---
    L += ["", "=== US ECONOMIC CALENDAR ==="]
    evs = st["events"]
    show_rel = []
    upcoming = []
    if not evs:
        L += ["NEWS RISK : UNKNOWN | NEXT EVENT : NONE", "NO VERIFIED ECONOMIC CALENDAR DATA."]
    else:
        released = [
            e for e in evs
            if _is_verified(e) and e.get("actual") is not None and now - datetime.fromisoformat(e["dt_utc"]) <= timedelta(hours=72)
        ]
        upcoming = [
            e for e in evs
            if _is_verified(e) and datetime.fromisoformat(e["dt_utc"]) >= now - timedelta(minutes=POST_RELEASE_WINDOW_MIN)
            and (e.get("actual") is None)
        ]
        if detail:
            show_rel = [e for e in evs if e.get("actual") is not None][-8:]
            show_up = upcoming[:12]
        else:
            show_rel = [e for e in released if e["impact"] == "HIGH"][-2:]
            nh = _next_event(st, now, high_only=True) or _next_event(st, now, high_only=False)
            show_up = [nh] if nh else []
        for e in show_rel + show_up:
            L += _block(e, st, now, detail)
        L.append("-" * 40)
        if not detail:
            more = [e for e in upcoming if e not in show_up][:4]
            for e in more:
                L.append(f"UPCOMING  {_wib_str(e)} | {e['impact']:<6} | {_short(e)} [{e.get('event_type', e['key'])}]")

    # --- SECTION 2: LATEST GOLD MARKET NEWS ---
    L += ["", "=== LATEST GOLD MARKET NEWS ==="]
    g_status = gst.get("status", "UNAVAILABLE")
    raw_arts = gst.get("articles") or []
    show_arts = _select_top_gold_articles(raw_arts, limit=(10 if detail else 5), now=now)
    if g_status == "UNAVAILABLE" or not show_arts:
        if g_status == "UNAVAILABLE":
            L.append(f"STATUS : UNAVAILABLE ({gst.get('error') or 'provider tidak dapat diakses'})")
        else:
            L.append("STATUS : READY (0 artikel emas relevan dalam 72 jam terakhir)")
        L.append("NO VERIFIED GOLD HEADLINES = NO FABRICATED FALLBACK.")
        missing = gst.get("optional_keys_missing") or []
        if missing:
            L.append(f"OPTIONAL API KEYS (.env): {', '.join(missing)} (opsional untuk feed berita pasar tambahan)")
    else:
        if g_status == "STALE":
            L.append("WARNING: Menampilkan cache STALE — bukan headline live terbaru; jalankan 'news refresh'.")
        for idx, art in enumerate(show_arts, 1):
            L += _gold_article_block(art, idx, g_status, detail=detail)
        L.append("-" * 40)
        if detail:
            p_st = gst.get("provider_status") or {}
            if p_st:
                p_items = [f"{k}={v}" for k, v in sorted(p_st.items())]
                L.append("PROVIDER AUDIT : " + " | ".join(p_items))
            missing = gst.get("optional_keys_missing") or []
            if missing:
                L.append("OPTIONAL KEYS  : " + ", ".join(missing) + " (belum diset di .env)")

    if not detail:
        L.append("Gunakan 'news detail' untuk metadata & analisis lengkap, 'news refresh' untuk paksa refresh.")

    if ask_ai and (st["status"] == "READY" or g_status == "READY"):
        ai_resp = _ai_text(
            show_rel if st["status"] == "READY" else [],
            ask_ai,
            active_signal=active_signal,
            upcoming=upcoming if st["status"] == "READY" else [],
            gold_articles=show_arts if g_status == "READY" else [],
        )
        if ai_resp:
            L.append(ai_resp)

    return "\n".join(L)


def _clean(s, n=40):
    return re.sub(r"[^\w .%+\-/()]", "", str(s))[:n]


def handle_command(cmd, ask_ai=None, active_signal=None):
    """Return teks untuk command 'news ...' atau None bila bukan command news."""
    t = str(cmd or "").strip().lower().split()
    if not t or t[0] != "news":
        return None
    try:
        sub = t[1] if len(t) > 1 else ""
        if sub == "refresh":
            get_state(force=True)
            get_gold_news_state(force=True)
            return _news_text(False, ask_ai, active_signal=active_signal)
        if sub in ("", "detail"):
            return _news_text(sub == "detail", ask_ai, active_signal=active_signal)
        return "Gunakan: news | news detail | news refresh"
    except Exception as e:
        return f"US News Radar error terisolasi: {type(e).__name__}"
