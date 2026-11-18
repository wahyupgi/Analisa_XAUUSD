# -*- coding: utf-8 -*-
"""
modules/us_news_radar.py — US Fundamental News Radar (informational only, v1).

SUMBER DATA: FRED API (Federal Reserve Bank of St. Louis), gratis, perlu FRED_API_KEY
(https://fred.stlouisfed.org/docs/api/api_key.html). FRED menyajikan data resmi BLS/BEA/Fed.
  - Jadwal  : /release/dates   (TANGGAL rilis resmi; JAM rilis = aturan standar, ditandai TIME RULE-BASED)
  - Actual  : /series/observations dengan vintage (realtime) per tanggal rilis -> revisi 'previous' terdeteksi
  - Forecast: TIDAK tersedia dari sumber gratis resmi -> selalu N/A (tidak pernah dikarang)
Tidak tercakup: Fed Chair speech, ISM PMI (tidak ada di FRED), konsensus forecast.

Modul ini TIDAK mengubah keputusan/zona/order. Semua error jaringan diisolasi.
"""
import json
import os
import re
import time
from datetime import date, datetime, timedelta, timezone

try:
    from zoneinfo import ZoneInfo
    TZ_NY = ZoneInfo("America/New_York")
    TZ_WIB = ZoneInfo("Asia/Jakarta")
except Exception:  # tzdata tidak ada -> radar UNAVAILABLE, scanner tetap jalan
    TZ_NY = TZ_WIB = None

_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_FILE = os.path.join(_DIR, "news_cache.json")
FRED = "https://api.stlouisfed.org/fred"

TIMEOUT = 5            # detik per request
RETRIES = 2            # percobaan per request (backoff 1.5s, 3s)
REFRESH_DEADLINE = 25  # batas total satu siklus refresh (detik)
TTL_NORMAL = 3600      # detik
TTL_ACTIVE = 300       # dekat/ sesudah jadwal rilis
FAIL_BACKOFF = 120     # jeda minimal antar siklus refresh yang gagal
WINDOW_PAST_DAYS = 10
WINDOW_FUTURE_DAYS = 30

# Aturan impact (bisa diedit): key -> HIGH / MEDIUM / LOW
IMPACT_RULES = {
    "FOMC": "HIGH", "CPI": "HIGH", "CORE_CPI": "HIGH", "NFP": "HIGH", "UNRATE": "HIGH",
    "CORE_PCE": "HIGH", "PCE": "MEDIUM", "RETAIL": "MEDIUM", "GDP": "MEDIUM",
    "PPI": "MEDIUM", "CORE_PPI": "MEDIUM", "CLAIMS": "MEDIUM",
}

# key, nama, release_id, keyword validasi nama rilis, series FRED, mode, unit, jam ET, band netral, hot_bearish
#   mode: yoy / mom (persen), diff (selisih level), level (nilai langsung). hot_bearish: actual > forecast => USD/yield naik => emas tertekan
#   FOMC (series None) = hanya jadwal, tanpa actual numerik.
INDICATORS = [
    ("FOMC",     "US FOMC Statement",          101, "FOMC",             None,            None,    "",      "14:00", None, None),
    ("CPI",      "US CPI (YoY)",               10,  "Consumer Price",   "CPIAUCNS",      "yoy",   "%",     "08:30", 0.05, True),
    ("CORE_CPI", "US Core CPI (YoY)",          10,  "Consumer Price",   "CPILFENS",      "yoy",   "%",     "08:30", 0.05, True),
    ("NFP",      "US Nonfarm Payrolls (chg)",  50,  "Employment Situation", "PAYEMS",    "diff",  "K",     "08:30", 25.0, True),
    ("UNRATE",   "US Unemployment Rate",       50,  "Employment Situation", "UNRATE",    "level", "%",     "08:30", 0.05, False),
    ("PCE",      "US PCE (YoY)",               54,  "Personal Income",  "PCEPI",         "yoy",   "%",     "08:30", 0.05, True),
    ("CORE_PCE", "US Core PCE (YoY)",          54,  "Personal Income",  "PCEPILFE",      "yoy",   "%",     "08:30", 0.05, True),
    ("RETAIL",   "US Retail Sales (MoM)",      9,   "Retail",           "RSAFS",         "mom",   "%",     "08:30", 0.1, True),
    ("GDP",      "US GDP (q/q SAAR)",          53,  "Gross Domestic",   "A191RL1Q225SBEA", "level", "%",   "08:30", 0.1, True),
    ("PPI",      "US PPI Final Demand (MoM)",  46,  "Producer Price",   "PPIFIS",        "mom",   "%",     "08:30", 0.1, True),
    ("CORE_PPI", "US Core PPI (MoM)",          46,  "Producer Price",   "PPIFES",        "mom",   "%",     "08:30", 0.1, True),
    ("CLAIMS",   "US Initial Jobless Claims",  180, "Claims",           "ICSA",          "level", "",      "08:30", 5000.0, False),
]

_mem = {"cache": None, "last_attempt": 0.0, "last_ok": False, "error": ""}


# ------------------------------------------------------------------ util
def _api_key():
    k = os.environ.get("FRED_API_KEY", "").strip()
    if k:
        return k
    p = os.path.join(_DIR, ".env")
    try:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("FRED_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    except Exception:
        pass
    return ""


class _Fail(Exception):
    pass


def _get(path, params, deadline):
    import requests
    p = dict(params)
    p["api_key"] = _api_key()
    p["file_type"] = "json"
    last = "error"
    for i in range(RETRIES):
        if time.time() > deadline:
            raise _Fail("deadline")
        try:
            r = requests.get(FRED + path, params=p, timeout=TIMEOUT)
            if r.status_code == 200:
                d = r.json()
                if not isinstance(d, dict):
                    raise _Fail("schema")
                return d
            last = f"HTTP {r.status_code}"
            if r.status_code in (400, 401, 403, 404):
                raise _Fail(last)
        except _Fail:
            raise
        except Exception as e:  # pesan exception requests memuat URL+key -> hanya tipe
            last = type(e).__name__
        if i < RETRIES - 1:
            time.sleep(1.5 * (2 ** i))
    raise _Fail(last)


def _now_utc():
    return datetime.now(timezone.utc)


def _event_dt(d_iso, hhmm):
    h, m = hhmm.split(":")
    local = datetime(int(d_iso[:4]), int(d_iso[5:7]), int(d_iso[8:10]), int(h), int(m), tzinfo=TZ_NY)
    return local.astimezone(timezone.utc)


def _wib(dt_utc):
    return dt_utc.astimezone(TZ_WIB)


# ------------------------------------------------------------------ data fetch
def _obs(sid, rd, deadline):
    d = _get("/series/observations", {"series_id": sid, "realtime_start": rd, "realtime_end": rd,
                                      "sort_order": "desc", "limit": 20}, deadline)
    rows = d.get("observations")
    if not isinstance(rows, list):
        raise _Fail("schema")
    out = []
    for o in rows:
        try:
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(o["date"])):
                continue
            out.append((o["date"], float(o["value"])))
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


def _fetch_actual(ind, rd, deadline):
    """Actual terverifikasi hanya jika vintage tanggal rilis memuat periode BARU dibanding vintage H-1."""
    sid, mode = ind[4], ind[5]
    rd_prev = (date.fromisoformat(rd) - timedelta(days=1)).isoformat()
    a = _obs(sid, rd, deadline)
    b = _obs(sid, rd_prev, deadline)
    if not a or not b or a[0][0] <= b[0][0]:
        return None  # belum ada data baru -> PENDING RELEASE
    ta, tb = _transform(a, mode), _transform(b, mode)
    if not ta or ta[1] is None or ta[2] is None:
        return None
    res = {"period": ta[0], "actual": round(ta[1], 4), "previous": round(ta[2], 4),
           "previous_original": None, "revised": False,
           "actual_asof": _now_utc().isoformat(timespec="seconds")}
    if tb and tb[1] is not None and abs(tb[1] - ta[2]) > 1e-6 and tb[0] != ta[0]:
        res["previous_original"] = round(tb[1], 4)
        res["revised"] = True
    return res


def _refresh(old):
    """Satu siklus refresh. Raise _Fail bila kalender gagal total; error per-indikator diisolasi."""
    if TZ_NY is None:
        raise _Fail("tzdata missing")
    if not _api_key():
        raise _Fail("FRED_API_KEY missing")
    if not re.fullmatch(r"[0-9a-z]{32}", _api_key()):
        raise _Fail("FRED_API_KEY format invalid (harus 32 karakter huruf kecil/angka)")
    deadline = time.time() + REFRESH_DEADLINE
    now = _now_utc()
    today = now.astimezone(TZ_NY).date()
    lo, hi = today - timedelta(days=WINDOW_PAST_DAYS), today + timedelta(days=WINDOW_FUTURE_DAYS)
    old_ev = {e["id"]: e for e in (old or {}).get("events", [])}
    names = dict((old or {}).get("release_names", {}))
    dates, notes = {}, []
    for rid in sorted({i[2] for i in INDICATORS}):
        try:
            if str(rid) not in names:
                d = _get("/release", {"release_id": rid}, deadline)
                names[str(rid)] = str((d.get("releases") or [{}])[0].get("name", ""))
            kw = next(i[3] for i in INDICATORS if i[2] == rid)
            if kw.lower() not in names[str(rid)].lower():
                notes.append(f"release {rid} name mismatch")
                names.pop(str(rid), None)
                continue
            d = _get("/release/dates", {"release_id": rid, "include_release_dates_with_no_data": "true",
                                        "sort_order": "desc", "limit": 100}, deadline)
            rows = d.get("release_dates")
            if not isinstance(rows, list):
                raise _Fail("schema")
            dates[rid] = sorted({r["date"] for r in rows
                                 if re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(r.get("date", "")))
                                 and lo.isoformat() <= r["date"] <= hi.isoformat()})
        except _Fail as e:
            if not dates and not names:
                raise
            notes.append(f"release {rid}: {e}")
    if not dates:
        raise _Fail("calendar empty")
    events = {}
    for ind in INDICATORS:
        key, name, rid = ind[0], ind[1], ind[2]
        for d_iso in dates.get(rid, []):
            eid = f"{key}|{d_iso}"
            dt = _event_dt(d_iso, ind[7])
            ev = {"id": eid, "key": key, "name": name, "impact": IMPACT_RULES.get(key, "LOW"),
                  "release_id": rid, "release_name": names.get(str(rid), ""), "date_et": d_iso,
                  "time_et": ind[7], "dt_utc": dt.isoformat(timespec="minutes"),
                  "unit": ind[6], "forecast": None, "actual": None, "previous": None,
                  "revised": False, "previous_original": None, "period": None,
                  "source": "FRED/St. Louis Fed", "series": ind[4]}
            prev = old_ev.get(eid)
            if prev and prev.get("actual") is not None:
                ev.update({k: prev.get(k) for k in ("actual", "previous", "revised", "previous_original",
                                                    "period", "actual_asof")})
            elif ind[4] and now >= dt and now - dt <= timedelta(days=WINDOW_PAST_DAYS) and time.time() < deadline:
                try:
                    r = _fetch_actual(ind, d_iso, deadline)
                    if r:
                        ev.update(r)
                except _Fail as e:
                    notes.append(f"{key}: {e}")
            events[eid] = ev  # dedup berdasarkan key+tanggal
    return {"fetched_at": now.isoformat(timespec="seconds"), "events": sorted(events.values(), key=lambda e: e["dt_utc"]),
            "release_names": names, "notes": notes[:5]}


def _load():
    if _mem["cache"] is None:
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                c = json.load(f)
            if isinstance(c, dict) and isinstance(c.get("events"), list) and c.get("fetched_at"):
                _mem["cache"] = c
        except Exception:
            pass
    return _mem["cache"]


def _ttl(cache, now):
    for e in cache.get("events", []):
        try:
            dt = datetime.fromisoformat(e["dt_utc"])
        except Exception:
            continue
        if e.get("actual") is None and e.get("series") and dt - timedelta(minutes=5) <= now <= dt + timedelta(hours=3):
            return TTL_ACTIVE
    return TTL_NORMAL


def get_state(force=False):
    """Return {status: READY/STALE/UNAVAILABLE, events, fetched_at, error}. Tidak pernah raise."""
    try:
        now = _now_utc()
        cache = _load()
        age = None
        if cache:
            age = (now - datetime.fromisoformat(cache["fetched_at"])).total_seconds()
        need = force or cache is None or age > _ttl(cache, now)
        if need and (force or time.time() - _mem["last_attempt"] >= FAIL_BACKOFF or _mem["last_ok"]):
            _mem["last_attempt"] = time.time()
            try:
                new = _refresh(cache)
                _mem.update(cache=new, last_ok=True, error="")
                try:
                    with open(CACHE_FILE, "w", encoding="utf-8") as f:
                        json.dump(new, f)
                except Exception:
                    pass
                cache, age = new, 0.0
            except Exception as e:
                _mem.update(last_ok=False, error=str(e)[:80] if isinstance(e, _Fail) else type(e).__name__)
        if not cache:
            return {"status": "UNAVAILABLE", "events": [], "fetched_at": None, "error": _mem["error"] or "no cache"}
        age = (now - datetime.fromisoformat(cache["fetched_at"])).total_seconds()
        fresh = _mem["last_ok"] and age <= _ttl(cache, now) * 2
        return {"status": "READY" if fresh else "STALE", "events": cache["events"],
                "fetched_at": cache["fetched_at"], "error": _mem["error"]}
    except Exception as e:
        return {"status": "UNAVAILABLE", "events": [], "fetched_at": None, "error": type(e).__name__}


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
    if ev.get("actual") is not None:
        return "RELEASED"
    return "SCHEDULED" if now < datetime.fromisoformat(ev["dt_utc"]) else "PENDING RELEASE"


def analyze(ev):
    """Return (label, alasan). Surprise = Actual - Forecast; tanpa forecast terverifikasi -> UNCONFIRMED."""
    ind = next((i for i in INDICATORS if i[0] == ev["key"]), None)
    if ev.get("actual") is None:
        return "UNCONFIRMED", "belum dirilis", None
    if not ind or ind[8] is None:
        return "UNCONFIRMED", "indikator non-numerik", None
    if ev.get("forecast") is None:
        return "UNCONFIRMED", "forecast N/A (tidak ada sumber forecast terverifikasi)", None
    s = ev["actual"] - ev["forecast"]
    if abs(s) < ind[8]:
        lab = "NEUTRAL"
    else:
        hot = (s > 0) == bool(ind[9])  # hot = data lebih kuat/inflasi lebih tinggi
        lab = "POTENTIAL BEARISH" if hot else "POTENTIAL BULLISH"
        # revisi 'previous' berlawanan arah -> MIXED
        if ev.get("revised") and ev.get("previous_original") is not None:
            rev = ev["previous"] - ev["previous_original"]
            if abs(rev) >= ind[8] and ((rev > 0) != (s > 0)):
                lab = "MIXED"
    return lab, "surprise = actual - forecast (tanpa data reaksi pasar)", s


def _risk_for_event(ev, now):
    dt = datetime.fromisoformat(ev["dt_utc"])
    if ev["impact"] == "HIGH":
        return "HIGH VOLATILITY"
    if ev["impact"] == "MEDIUM":
        return "MODERATE VOLATILITY"
    return "LOW"


def news_risk(st, now=None):
    now = now or _now_utc()
    if st["status"] == "UNAVAILABLE":
        return "UNKNOWN"
    lvl = "LOW"
    for ev in st["events"]:
        d = (datetime.fromisoformat(ev["dt_utc"]) - now).total_seconds() / 60.0
        if ev["impact"] == "HIGH":
            if -60 <= d <= 120:
                return "HIGH"
            if 0 < d <= 1440:
                lvl = "MEDIUM"
        elif ev["impact"] == "MEDIUM" and -60 <= d <= 120 and lvl == "LOW":
            lvl = "MEDIUM"
    return lvl


def _next_event(st, now, high_only=False):
    for ev in st["events"]:
        if high_only and ev["impact"] != "HIGH":
            continue
        dt = datetime.fromisoformat(ev["dt_utc"])
        if dt >= now or (ev.get("actual") is None and now - dt <= timedelta(hours=3)):
            return ev
    return None


def _countdown(ev, now):
    s = int((datetime.fromisoformat(ev["dt_utc"]) - now).total_seconds())
    if s < 0:
        return "PENDING RELEASE" if ev.get("actual") is None else "RELEASED"
    d, r = divmod(s, 86400)
    h, r = divmod(r, 3600)
    return (f"in {d}d " if d else "in ") + f"{h:02d}h {r // 60:02d}m"


def _short(ev):
    return ev["name"].replace("US ", "", 1).split(" (")[0]


def _wib_str(ev):
    return _wib(datetime.fromisoformat(ev["dt_utc"])).strftime("%a %d-%b %H:%M WIB")


# ------------------------------------------------------------------ formatting
def startup_lines():
    try:
        st = get_state()
        now = _now_utc()
        out = [f"US NEWS RADAR : {st['status']}"]
        ev = _next_event(st, now, high_only=True)
        if ev:
            out.append(f"NEXT HIGH IMPACT : {_short(ev)} | {_wib_str(ev)} | {_countdown(ev, now)}")
        elif st["status"] != "UNAVAILABLE":
            out.append("NEXT HIGH IMPACT : none in cached window")
        else:
            out.append(f"NEWS SETUP : {st['error']} (lihat modul us_news_radar)")
        return out
    except Exception:
        return ["US NEWS RADAR : UNAVAILABLE"]


def context_lines():
    """Maks 2 baris untuk scan/pantau. Informasional saja."""
    try:
        st = get_state()
        now = _now_utc()
        sfx = " (STALE DATA)" if st["status"] == "STALE" else ""
        out = [f"NEWS RISK : {news_risk(st, now)}{sfx}"]
        ev = _next_event(st, now)
        if ev:
            out.append(f"NEXT EVENT : {_short(ev)} | {_wib_str(ev)}")
        return out
    except Exception:
        return ["NEWS RISK : UNKNOWN"]


def _block(ev, st, now, detail=False):
    stat = _ev_status(ev, now)
    stale = " | STALE DATA" if st["status"] == "STALE" else ""
    u = ev["unit"]
    lab, why, s = analyze(ev)
    act = _fmt(ev["actual"], u) if ev.get("actual") is not None else "PENDING RELEASE"
    L = ["-" * 40,
         f"TIME WIB       : {_wib(datetime.fromisoformat(ev['dt_utc'])).strftime('%Y-%m-%d %H:%M')}",
         f"EVENT          : {ev['name']}",
         f"IMPACT         : {ev['impact']}",
         f"PREVIOUS       : {_fmt(ev.get('previous'), u) if ev.get('actual') is not None else 'N/A'}"
         + (f" (REVISED from {_fmt(ev['previous_original'], u)})" if ev.get("revised") else ""),
         f"FORECAST       : {_fmt(ev.get('forecast'), u)}",
         f"ACTUAL         : {act}"]
    if ev.get("actual") is not None:
        L.append(f"SURPRISE       : {('%+.2f' % s) if s is not None else 'N/A (no forecast)'}")
    L += [f"DATA SOURCE    : {ev['source']} (original: BLS/BEA/Fed)",
          f"DATA STATUS    : {stat}{stale}",
          f"XAUUSD IMPACT  : {lab}",
          f"NEWS RISK      : {_risk_for_event(ev, now)}"]
    if detail:
        L += [f"REF PERIOD     : {ev.get('period') or 'N/A'}  (series {ev.get('series') or '-'})",
              f"RELEASE TIME   : {ev['date_et']} {ev['time_et']} America/New_York -> WIB (DST otomatis)",
              "VALIDATION     : DATE verified (FRED calendar) | TIME rule-based (standar rilis)"
              + (" | ACTUAL verified by vintage" if ev.get("actual") is not None else ""),
              f"ANALYSIS       : {why}"]
    return L


def _header(st):
    ft = "-"
    if st.get("fetched_at"):
        ft = _wib(datetime.fromisoformat(st["fetched_at"])).strftime("%Y-%m-%d %H:%M WIB")
    L = ["XAUUSD US FUNDAMENTAL RADAR", f"RADAR STATUS   : {st['status']}  | LAST FETCH: {ft}",
         "NOTE           : delayed/polled data (bukan real-time); informational only"]
    if st["status"] == "STALE":
        L.append("WARNING        : STALE DATA - jangan dianggap berita terbaru")
    if st["status"] == "UNAVAILABLE":
        L.append(f"UNAVAILABLE    : {st['error']} -> set FRED_API_KEY di .env (gratis di fred.stlouisfed.org)")
    return L


def _build_trade_bias_prompt(upcoming_high, released_recent, active_signal=None):
    """
    Menyusun prompt analitis untuk AI:
    - Sebelum News: evaluasi tren data sebelumnya (Previous trend), ekspektasi Fed, dan sentimen makro saat ini
      untuk menentukan KECENDERUNGAN BIAS (Condong BUY atau Condong SELL XAUUSD) menjelang news.
    - Sesudah News (news refresh): evaluasi data Actual yang baru rilis untuk menentukan konfirmasi FINAL BUY atau SELL.
    """
    up_info = []
    for e in upcoming_high[:3]:
        up_info.append(f"- {e['name']} (Jadwal: {_wib_str(e)}, Status: {e.get('actual') or 'PENDING RELEASE'}, Prev: {e.get('previous') or 'N/A'}, Unit: {e.get('unit') or ''})")

    rel_info = []
    for e in released_recent[-3:]:
        u = e.get("unit", "")
        rel_info.append(f"- {e['name']} (Period: {e.get('period') or 'N/A'}, Actual: {_fmt(e.get('actual'), u)}, Prev: {_fmt(e.get('previous'), u)})")

    sig_str = "TIDAK ADA SETUP AKTIF"
    if active_signal and isinstance(active_signal, dict) and active_signal.get("lo"):
        sig_str = f"{active_signal.get('arah', 'SETUP')} @ {active_signal.get('lo', 0):.2f} - {active_signal.get('hi', 0):.2f} (SL: {active_signal.get('sl', 0):.2f}, TP: {active_signal.get('tp1', 0):.2f})"

    is_post_release = bool(released_recent and any(e.get("actual") is not None for e in released_recent))

    prompt = f"""
Anda adalah Senior Macro Quant & XAUUSD Risk Specialist.
Tugas Anda adalah menyaring data berita ekonomi AS dan memberikan arahan BIAS yang tegas (Condong BUY atau Condong SELL) untuk XAUUSD.

[SETUP TEKNIKAL XAUUSD SAAT INI]:
{sig_str}

[DATA BERITA EKONOMI AS]:
1. Berita Mendatang (Upcoming):
{chr(10).join(up_info) if up_info else "- Tidak ada high-impact event dalam waktu dekat."}

2. Berita Terakhir / Baru Dirilis (Released):
{chr(10).join(rel_info) if rel_info else "- Belum ada event baru rilis 24-48 jam terakhir."}

INSTRUKSI ANALISIS:
1. JIKA SEBELUM NEWS (Data Actual masih PENDING):
   - Baca tren data 'Previous', ekspektasi suku bunga Fed, dan arah inflasi/tenaga kerja AS terkini.
   - Saring dan putuskan KECENDERUNGAN BIAS: Apakah pasar saat ini lebih CONDONG BUY (Bullish XAUUSD) atau CONDONG SELL (Bearish XAUUSD)?
   - Berikan alasan 1-2 kalimat mengapa sentimen makronya condong ke arah tersebut.
   - Berikan panduan eksekusi sebelum news: apakah setup teknikal boleh diambil, di-hold, atau wajib tunggu konfirmasi rilis.

2. JIKA SETELAH NEWS (Ada data Actual baru rilis):
   - Bandingkan data Actual vs Previous.
   - Putuskan KONFIRMASI FINAL PASCA-NEWS: BUY atau SELL untuk XAUUSD.
   - Jelaskan apakah setup teknikal saat ini SEARAH (Valid/Kuat) atau BERLAWANAN (Bahaya/Skip).

Format Output WAJIB Tepat Seperti Ini:
💡 [KONFIRMASI FUNDAMENTAL & TEKNIKAL XAUUSD]:
• Status Berita           : (SEBELUM RILIS / PASCA RILIS AKTUAL)
• Arah Kecenderungan Bias : [CONDONG BUY (BULLISH)] atau [CONDONG SELL (BEARISH)] atau [NETRAL]
• Alasan Makro Fundamental: (Penjelasan singkat 1-2 kalimat berbasis data)
• Sinkronisasi Setup      : (Searah dengan Setup / Berlawanan Arah / Menunggu Setup)
• Panduan Tindakan        : (Instruksi tindakan praktis untuk trader)
"""
    return prompt.strip()


def _ai_text(events, ask_ai, active_signal=None, upcoming=None):
    """Gemini menjelaskan data terverifikasi & memberikan konfirmasi arah/skenario."""
    try:
        p = _build_trade_bias_prompt(upcoming or [], events or [], active_signal)
        txt = str(ask_ai(p) or "").strip()[:1400]
        return "\n" + txt if txt else ""
    except Exception as e:
        return f"\n💡 [KONFIRMASI FUNDAMENTAL]: Tidak tersedia ({type(e).__name__})"


def _news_text(detail, ask_ai, active_signal=None):
    st = get_state()
    now = _now_utc()
    L = _header(st)
    evs = st["events"]
    if not evs:
        return "\n".join(L + ["NO VERIFIED DATA = NO FUNDAMENTAL DIRECTION."])
    released = [e for e in evs if e.get("actual") is not None and now - datetime.fromisoformat(e["dt_utc"]) <= timedelta(hours=48)]
    upcoming = [e for e in evs if e.get("actual") is None and datetime.fromisoformat(e["dt_utc"]) >= now - timedelta(hours=3)]
    if detail:
        show_rel = [e for e in evs if e.get("actual") is not None][-8:]
        show_up = upcoming[:12]
    else:
        show_rel = [e for e in released if e["impact"] == "HIGH"][-2:]
        nh = _next_event(st, now, high_only=True)
        show_up = [nh] if nh else []
    for e in show_rel + show_up:
        L += _block(e, st, now, detail)
    L.append("-" * 40)
    if not detail:
        more = [e for e in upcoming if e not in show_up][:4]
        for e in more:
            L.append(f"UPCOMING  {_wib_str(e)} | {e['impact']:<6} | {_short(e)}")
        L.append("Gunakan 'news detail' untuk data lengkap, 'news refresh' untuk paksa refresh.")

    # Tampilkan konfirmasi AI (baik di 'news' ringkas maupun 'news detail') jika ask_ai tersedia
    if ask_ai:
        ai_resp = _ai_text(show_rel, ask_ai, active_signal=active_signal, upcoming=upcoming)
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
            return _news_text(False, ask_ai, active_signal=active_signal)
        if sub in ("", "detail"):
            return _news_text(sub == "detail", ask_ai, active_signal=active_signal)
        return "Gunakan: news | news detail | news refresh"
    except Exception as e:
        return f"US News Radar error terisolasi: {type(e).__name__}"
