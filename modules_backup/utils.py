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
from modules.config import FILE_ERROR_LOG

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
