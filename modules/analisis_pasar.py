# -*- coding: utf-8 -*-
"""
modules/analisis_pasar.py
Analisis makro & kondisi pasar: Sesi trading (London/NY), Kalender Berita USD,
Blackout waktu rilis berita, Korelasi US Dollar (DXY/EURUSD), dan ADR Exhaustion.
"""
import json
import time
import urllib.request
from datetime import datetime, timedelta, timezone
import MetaTrader5 as mt5
import numpy as np
import pandas as pd

from modules.config import SYMBOL, BERITA_BLACKOUT_MENIT, BLOKIR_JIKA_BERITA_GAGAL
from modules.utils import _catat_error, deteksi_offset, us_dst, wkt
from modules.mt5_conn import (
    ambil_data, pastikan_mt5_aktif, ambil_candle_berjalan, _suffix_simbol
)

_CACHE_BERITA = {"ts": 0, "waktu": 0.0, "events": [], "gagal": False, "err": ""}


def analisa_sesi(m30, m5, offset):
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    ny_off = -4 if us_dst(now) else -5
    ny = now + timedelta(hours=ny_off)
    h = ny.hour + ny.minute / 60
    hu = now.hour + now.minute / 60
    wib = now + timedelta(hours=7)
    to_wib = lambda hn: f"{int((hn + 7 - ny_off) % 24):02d}:00"

    sesi = [nm for nm, a, b in (("Asia/Tokyo", 0, 8), ("London", 7, 16), ("New York", 12, 21)) if a <= hu < b]
    if h >= 20:
        kz = "Asia Kill Zone"
    elif 2 <= h < 5:
        kz = "London Open Kill Zone"
    elif 7 <= h < 10:
        kz = "New York Open Kill Zone"
    elif 10 <= h < 12:
        kz = "London Close Kill Zone"
    else:
        kz = None

    out = ["\n##### SESI TRADING & KILL ZONE #####",
           f"Waktu sekarang: {wib:%Y-%m-%d %H:%M} WIB | {now:%H:%M} UTC | {ny:%H:%M} New York (offset server broker: UTC{offset:+d})",
           f"Sesi aktif: {', '.join(sesi) if sesi else 'di antara sesi (sepi)'}",
           f"Kill Zone sekarang: {kz if kz else 'TIDAK sedang di kill zone (probabilitas setup lebih rendah)'}",
           f"Jadwal kill zone hari ini (WIB): London Open {to_wib(2)}-{to_wib(5)} | New York Open {to_wib(7)}-{to_wib(10)} | London Close {to_wib(10)}-{to_wib(12)}"]
    if now.weekday() >= 5:
        out.append("CATATAN: akhir pekan, market emas tutup.")

    idx = m30.index - pd.Timedelta(hours=offset)
    x = pd.DataFrame({"H": m30["High"].values, "L": m30["Low"].values,
                      "day": [t.date() for t in idx], "hr": [t.hour + t.minute / 60 for t in idx]})
    ref_day = idx[-1].date()
    hari_lalu = sorted(set(d for d in x["day"] if d < ref_day))
    today = x[x["day"] == ref_day]
    prev = x[x["day"] == hari_lalu[-1]] if hari_lalu else x.iloc[0:0]
    asia = today[today["hr"] < 7]   # 00:00-07:00 UTC (London Open 07:00 UTC tidak ikut agar sweep-nya bisa terdeteksi)
    lon = today[(today["hr"] >= 7) & (today["hr"] < 16)]
    mulai_hari = datetime(ref_day.year, ref_day.month, ref_day.day)
    levels = []
    if len(prev):
        levels += [("Previous Day High (PDH)", prev["H"].max(), "H", mulai_hari),
                   ("Previous Day Low (PDL)", prev["L"].min(), "L", mulai_hari)]
    if len(asia):
        sel = mulai_hari + timedelta(hours=7)
        levels += [("Asia High", asia["H"].max(), "H", sel), ("Asia Low", asia["L"].min(), "L", sel)]
    if len(lon):
        sel = mulai_hari + timedelta(hours=16)
        levels += [("London High", lon["H"].max(), "H", sel), ("London Low", lon["L"].min(), "L", sel)]

    idx5 = m5.index - pd.Timedelta(hours=offset)
    out.append("Level likuiditas sesi (target stop hunt) & status M5:")
    for nm, val, tipe, sel in levels:
        sel_ts = pd.Timestamp(sel)
        msk = idx5 >= sel_ts
        status = "range sesi belum selesai" if sel_ts > idx5[-1] else "belum disentuh"
        if sel_ts <= idx5[-1] and msk.any():
            H5, L5, C5 = (m5[k].values[msk] for k in ("High", "Low", "Close"))
            tt = idx5[msk]
            if tipe == "H":
                sw_ = np.where((H5 > val) & (C5 < val))[0]
                br_ = np.where(C5 > val)[0]
            else:
                sw_ = np.where((L5 < val) & (C5 > val))[0]
                br_ = np.where(C5 < val)[0]
            if len(sw_):
                status = f"SUDAH DISAPU (sweep) pada {wkt(tt[sw_[-1]] + pd.Timedelta(hours=offset))} lalu close balik"
            elif len(br_):
                status = "ditembus (close di luar level)"
        out.append(f"  - {nm}: {val:.2f} | {status}")
    return "\n".join(out) + "\n"


def analisa_korelasi_usd():
    """Mengecek kekuatan US Dollar (DXY / EURUSD / USDJPY) di MT5 untuk konfirmasi korelasi XAUUSD."""
    out = ["\n===== KORELASI US DOLLAR (KONFIRMASI MAKRO) ====="]
    kandidat_simbol = [
        ("USDX", "DXY", 1), ("DXY", "DXY", 1), ("DX", "DXY", 1), ("USDX.a", "DXY", 1),
        ("EURUSD", "EURUSD", -1), ("EURUSDm", "EURUSD", -1),
        ("GBPUSD", "GBPUSD", -1), ("GBPUSDm", "GBPUSD", -1),
        ("USDJPY", "USDJPY", 1), ("USDJPYm", "USDJPY", 1),
    ]
    suf = _suffix_simbol()
    if suf:   # broker cent/ECN: pasangan biasanya memakai akhiran yang sama dengan XAUUSD
        kandidat_simbol = [(b + suf, g, a) for b, g, a in (
            ("USDX", "DXY", 1), ("DXY", "DXY", 1), ("EURUSD", "EURUSD", -1),
            ("GBPUSD", "GBPUSD", -1), ("USDJPY", "USDJPY", 1))] + kandidat_simbol
    ditemukan = []
    for sym, grup, arah_usd in kandidat_simbol:
        if any(g == grup for g, _ in ditemukan):
            continue
        if mt5.symbol_select(sym, True):
            rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_H1, 1, 60)
            if rates is not None and len(rates) >= 50:
                df = pd.DataFrame(rates)
                c = df["close"]
                ema20 = c.ewm(span=20, adjust=False).mean().iloc[-1]
                ema50 = c.ewm(span=50, adjust=False).mean().iloc[-1]
                chg = (c.iloc[-1] - c.iloc[-5]) / c.iloc[-5] * 100
                if (c.iloc[-1] > ema20 > ema50 and arah_usd == 1) or (c.iloc[-1] < ema20 < ema50 and arah_usd == -1):
                    status_usd = "USD MENGUAT (tekanan BEARISH untuk XAUUSD)"
                elif (c.iloc[-1] < ema20 < ema50 and arah_usd == 1) or (c.iloc[-1] > ema20 > ema50 and arah_usd == -1):
                    status_usd = "USD MELEMAH (dorongan BULLISH untuk XAUUSD)"
                else:
                    status_usd = "USD NETRAL / SIDEWAYS"
                ditemukan.append((grup, f"  - {sym} (H1): Close {c.iloc[-1]:.4f} (perubahan 4 jam: {chg:+.2f}%) -> {status_usd}"))
    if ditemukan:
        out.extend(teks for _, teks in ditemukan)
    else:
        out.append("  - Simbol korelasi USD tidak tersedia.")
    return "\n".join(out) + "\n"


def analisa_adr_exhaustion(df_d1, harga):
    """ADR 14 hari dari candle D1 yang sudah close vs range HARI INI dari candle D1 yang sedang berjalan + harga live.
    (df_d1 dibuat dari posisi 1, jadi df_d1.iloc[-1] adalah KEMARIN, bukan hari ini.)"""
    adr14 = (df_d1["High"] - df_d1["Low"]).iloc[-14:].mean()
    cur = ambil_candle_berjalan(mt5.TIMEFRAME_D1)
    if cur is not None:
        hi_today = max(float(cur["high"]), harga)
        lo_today = min(float(cur["low"]), harga)
    else:
        hi_today = lo_today = harga
    range_skrg = hi_today - lo_today
    persen = (range_skrg / adr14 * 100) if adr14 > 0 else 0

    proj_high = lo_today + adr14
    proj_low = hi_today - adr14

    if persen >= 100:
        status = f"⚠️ EXHAUSTION / BENSIN HABIS ({persen:.0f}% dari ADR)! Rawan Reversal / Pullback tajam, JANGAN kejar pucuk/dasar!"
    elif persen >= 80:
        status = f"Hampir Penuh ({persen:.0f}% dari ADR) -> mulai waspada potensi koreksi/profit taking."
    else:
        status = f"Masih Ada Ruang Gerak ({persen:.0f}% dari ADR terpakai)."

    return (
        f"\n===== ADR (AVERAGE DAILY RANGE 14 HARI) & BATAS BENSIN HARIAN =====\n"
        f"Rata-rata Range Harian (ADR14): {adr14:.2f} poin ($)\n"
        f"Range Hari Ini: {range_skrg:.2f} poin (Low {lo_today:.2f} s/d High {hi_today:.2f}) -> {status}\n"
        f"Proyeksi Batas Atap Harian (ADR High): {proj_high:.2f} | Batas Lantai Harian (ADR Low): {proj_low:.2f}\n"
    )


def _ambil_events_berita():
    """Event USD High/Medium minggu ini (cache 15 menit; bila gagal, coba lagi 60 dtk & pakai data lama jika ada)."""
    ttl = 60 if _CACHE_BERITA["gagal"] else 900
    if _CACHE_BERITA["waktu"] and (time.time() - _CACHE_BERITA["waktu"]) < ttl:
        return _CACHE_BERITA["events"]
    url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=4) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        events = []
        for item in data:
            if item.get("country") != "USD" or item.get("impact") not in ("High", "Medium"):
                continue
            try:
                dt_utc = datetime.fromisoformat(item.get("date", "")).astimezone(timezone.utc)
            except Exception as e_dt:
                _catat_error("parse_tanggal_berita", e_dt)
                continue
            events.append(dict(dt=dt_utc, impact=item.get("impact"), title=item.get("title"),
                               forecast=item.get("forecast"), previous=item.get("previous")))
        _CACHE_BERITA.update(events=events, waktu=time.time(), gagal=False, err="")
    except Exception as e:
        _catat_error("_ambil_events_berita", e)
        _CACHE_BERITA.update(waktu=time.time(), gagal=True, err=str(e))   # events lama (jika ada) tetap dipakai
    return _CACHE_BERITA["events"]


def berita_blackout_aktif():
    """Judul berita High-impact USD yang rilis dalam +/- BERITA_BLACKOUT_MENIT, atau None."""
    events = _ambil_events_berita()
    if not events:
        if _CACHE_BERITA["gagal"] and BLOKIR_JIKA_BERITA_GAGAL:
            return f"kalender berita gagal diambil ({_CACHE_BERITA['err'] or 'offline'}) — fail-safe aktif"
        return None
    now_utc = datetime.now(timezone.utc)
    for e in events:
        if e["impact"] != "High":
            continue
        sel = (e["dt"] - now_utc).total_seconds() / 60
        if -BERITA_BLACKOUT_MENIT <= sel <= BERITA_BLACKOUT_MENIT:
            ket = f"rilis {int(-sel)} menit lalu" if sel < 0 else f"{int(sel)} menit lagi"
            return f"{e['title']} ({ket})"
    return None


def cek_kalender_berita_usd():
    """Jadwal berita High/Medium USD hari ini (WIB). Hitung mundur dihitung ulang setiap panggilan (tidak ikut di-cache)."""
    out = ["\n===== RADAR BERITA EKONOMI HIGH-IMPACT USD HARI INI ====="]
    events = _ambil_events_berita()
    if events is None:
        out.append("  - Jadwal kalender otomatis tidak terjangkau saat ini (periksa rilis news USD pada jam 19:30 / 21:00 WIB).")
        return "\n".join(out) + "\n"
    now_utc = datetime.now(timezone.utc)
    hari_wib = (now_utc + timedelta(hours=7)).date()
    berita_penting = []
    for e in events:
        dt_wib = e["dt"] + timedelta(hours=7)
        if dt_wib.date() != hari_wib:
            continue
        selisih_mnt = (e["dt"] - now_utc).total_seconds() / 60
        if -120 <= selisih_mnt <= 720:
            status_wkt = (
                f"🚨 SEBENTAR LAGI ({int(selisih_mnt)} menit lagi!)"
                if 0 <= selisih_mnt <= 60
                else (f"Baru rilis {abs(int(selisih_mnt))} menit lalu" if selisih_mnt < 0 else f"{selisih_mnt / 60:.1f} jam lagi")
            )
            berita_penting.append(
                f"  - [{str(e['impact']).upper()}] {dt_wib:%H:%M} WIB: {e['title']} "
                f"(Forecast: {e['forecast'] or '-'}, Prev: {e['previous'] or '-'}) -> {status_wkt}"
            )
    if berita_penting:
        out.extend(berita_penting)
    else:
        out.append("  - Tidak ada jadwal berita USD High/Medium Impact dalam waktu dekat hari ini.")
    return "\n".join(out) + "\n"
