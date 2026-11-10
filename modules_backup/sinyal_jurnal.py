# -*- coding: utf-8 -*-
"""
modules/sinyal_jurnal.py
Siklus hidup sinyal & jurnal: validasi struktur sinyal, gerbang kualitas sinyal,
simulasi candle M1, penguncian sinyal aktif (Signal Lock anti-plin-plan),
pencatatan ke CSV jurnal, evaluasi TP/SL, dan pembuatan tabel rekapitulasi.
"""
import csv
import json
import math
import os
import re
import shutil
import time
from datetime import datetime, timezone
import MetaTrader5 as mt5
import numpy as np
import pandas as pd

from modules.config import (
    SYMBOL, RR_MIN, ENTRY_LEBAR_MAKS, ENTRY_JARAK_MAKS, SL_JARAK_MAKS,
    SPREAD_MAKS_POIN, SPREAD_MAKS_RASIO_ATR, SINYAL_TTL_BELUM_TERISI_JAM,
    FILTER_TREND_AKTIF, IZINKAN_COUNTER_TREND_MEDIUM, FILTER_SESI_AKTIF, SESI_DIIZINKAN_UTC,
    MAKS_RISIKO_TERBUKA_PCT, MAKS_RUGI_HARIAN_PCT, SERVER_UTC_OFFSET_JAM,
    FILE_JURNAL_SINYAL, FILE_ERROR_LOG, KOLOM_JURNAL_SINYAL,
    ST_BATAL_TERLEWAT, ST_BATAL_SL, ST_BATAL_KEDALUWARSA, ST_SELESAI_TP2, ST_TP1_TERCAPAI,
    ST_AKTIF_ENTRY, ST_VALID_BERJALAN, ST_VALID_PULLBACK, ST_VALID_DEKAT_SL, STATUS_TERMINAL_SINYAL
)
from modules.utils import _catat_error, deteksi_offset
from modules.mt5_conn import (
    pastikan_mt5_aktif, ambil_data, tambah_indikator, _ambil_m1_sejak,
    cek_posisi_terbuka_mt5, status_risiko_akun, _info_tipe_pending
)
from modules.analisis_teknikal import bias_htf_cached
from modules.analisis_pasar import berita_blackout_aktif

_STATE_PARSER = {"ditolak": []}


def validasi_sinyal_struktur(sig):
    """Aturan intrinsik sinyal (tidak bergantung kondisi pasar saat ini): sisi SL/TP, lebar entry, jarak SL, R:R."""
    alasan = []
    arah = sig["arah"]
    lo, hi = sig.get("lo_asli", sig["lo"]), sig.get("hi_asli", sig["hi"])
    sl, tp1 = sig["sl"], sig["tp1"]
    tp2 = sig.get("tp2", tp1)
    mid = (lo + hi) / 2.0
    if hi - lo > ENTRY_LEBAR_MAKS:
        alasan.append(f"area entry terlalu lebar ({hi - lo:.1f} pt > {ENTRY_LEBAR_MAKS:g})")
    sisi_ok = True
    if arah == "BUY":
        if sl >= lo:
            alasan.append("SL BUY tidak berada di bawah area entry"); sisi_ok = False
        if tp1 <= hi:
            alasan.append("TP1 BUY tidak berada di atas area entry"); sisi_ok = False
        if tp2 < tp1 - 1e-6:
            alasan.append(f"TP2 BUY ({tp2:.2f}) lebih rendah dari TP1 ({tp1:.2f})")
            sig["tp2"] = tp1
    else:
        if sl <= hi:
            alasan.append("SL SELL tidak berada di atas area entry"); sisi_ok = False
        if tp1 >= lo:
            alasan.append("TP1 SELL tidak berada di bawah area entry"); sisi_ok = False
        if tp2 > tp1 + 1e-6:
            alasan.append(f"TP2 SELL ({tp2:.2f}) lebih tinggi dari TP1 ({tp1:.2f})")
            sig["tp2"] = tp1
    if sisi_ok:
        risiko = abs(mid - sl)
        if risiko < 0.5:
            alasan.append(f"jarak SL terlalu dekat ({risiko:.2f} pt)")
        elif risiko > SL_JARAK_MAKS:
            alasan.append(f"jarak SL terlalu jauh ({risiko:.1f} pt > {SL_JARAK_MAKS:g})")
        else:
            rr = abs(tp1 - mid) / risiko
            if rr < RR_MIN - 1e-9:
                alasan.append(f"R:R TP1 hanya 1:{rr:.2f} (minimal 1:{RR_MIN:g})")
    return (not alasan), alasan


def _simulasi_jalur(sig, rates, spread=0.0):
    """Menelusuri candle M1 sejak sinyal terbit: apakah entry terjemput, SL/TP tersentuh, atau TERLEWAT.
    Konservatif: bila SL dan TP tersentuh pada candle yang sama, SL dianggap lebih dulu.
    Gaya entry diturunkan dari harga saat terbit: limit (menunggu pullback), break (menunggu tembus), atau langsung (di dalam zona).
    PERBAIKAN ANTI-PHANTOM TP1: Untuk order 'limit' (mis. BUY Limit di bawah harga sekarang), jika saat sinyal dibuat
    harga memang masih di sekitar/di atas TP1, jangan langsung dianggap 'terlewat' kecuali harga sudah sempat
    mendekati area entry (atau melewati titik tengah menuju entry) lalu memantul menyentuh TP1, atau jika SL ditembus."""
    buy = str(sig.get("arah", "BUY")).upper() == "BUY"
    lo, hi = float(sig["lo"]), float(sig["hi"])
    sl, tp1 = float(sig["sl"]), float(sig["tp1"])
    tp2 = float(sig.get("tp2", tp1))
    mid = (float(sig.get("lo_asli", lo)) + float(sig.get("hi_asli", hi))) / 2.0
    hb0 = sig.get("harga_buat")
    if hb0 is None:
        gaya = "limit"
    else:
        hb0 = float(hb0)
        if buy:
            gaya = "limit" if hb0 > hi else ("break" if hb0 < lo else "langsung")
        else:
            gaya = "limit" if hb0 < lo else ("break" if hb0 > hi else "langsung")

    h = dict(terisi=False, tp1=False, tp2=False, sl=False, terlewat=False, batal_sebelum_isi=False, t_isi=None, t_akhir=None, mfe=0.0, mae=0.0)
    # Jika saat sinyal dicatat harga_buat sudah berada di luar batas SL (sinyal usang/invalid sejak lahir),
    # tandai batal sebelum terisi agar tidak dihitung sebagai trade loss -1R palsu.
    if hb0 is not None and ((buy and hb0 <= sl) or (not buy and hb0 >= sl)):
        h["batal_sebelum_isi"] = True
        h["terlewat"] = True
        return h

    terisi = (gaya == "langsung")
    # Untuk limit order: catat apakah harga pernah turun/naik mendekati separuh jalan ke zona entry
    sudah_mendekat = (gaya != "limit")
    batas_dekat_buy = hi + min(2.5, 0.45 * max(0.0, tp1 - hi))
    batas_dekat_sell = lo - min(2.5, 0.45 * max(0.0, lo - tp1))

    for r in rates:
        hb, lb = float(r["high"]), float(r["low"])      # harga bid
        ha, la = hb + spread, lb + spread               # harga ask
        t = int(r["time"])
        baru_isi = False
        if not terisi:
            if buy:                                       # BUY terisi di ask
                if lb <= batas_dekat_buy:
                    sudah_mendekat = True
                kena = (la <= hi) if gaya == "limit" else (ha >= lo and lb <= hi)
                # Hanya dianggap terlewat (missed entry) jika sudah sempat mendekati entry lalu terbang ke TP1,
                # ATAU jika harga saat buat < TP1 lalu langsung terbang ke TP1 + 1.5 pt tanpa retrace
                tembus_tp = (hb >= tp1) and (sudah_mendekat or (hb0 is not None and hb0 < tp1 - 1.0 and hb >= tp1 + 0.5))
            else:                                         # SELL terisi di bid
                if hb >= batas_dekat_sell:
                    sudah_mendekat = True
                kena = (hb >= lo) if gaya == "limit" else (lb <= hi and hb >= lo)
                tembus_tp = (lb <= tp1) and (sudah_mendekat or (hb0 is not None and hb0 > tp1 + 1.0 and lb <= tp1 - 0.5))
            if not kena:
                # Jika belum pernah menyentuh zona entry namun harga langsung menjebol SL
                if (buy and lb <= sl) or (not buy and ha >= sl):
                    h["batal_sebelum_isi"], h["t_akhir"] = True, t
                    break
                if tembus_tp:
                    h["terlewat"], h["t_akhir"] = True, t
                    break
                continue
            terisi = True
            h["t_isi"] = t
            # Entry break-style: titik terburuk candle ini terjadi SEBELUM harga menembus zona -> bukan bagian dari trade
            baru_isi = (gaya == "break")
        h["terisi"] = True
        if buy:
            h["mfe"] = max(h["mfe"], hb - mid)
            if not baru_isi:
                h["mae"] = max(h["mae"], mid - lb)
            if not h["tp1"]:
                if lb <= sl and not baru_isi:
                    h["sl"], h["t_akhir"] = True, t
                    break
                if hb >= tp1:
                    h["tp1"] = True
            else:
                # Setelah TP1 tercapai, jika harga balik menyentuh BEP (mid) atau SL sebelum TP2, runner selesai di TP1/BEP
                if hb >= tp2:
                    h["tp2"], h["t_akhir"] = True, t
                    break
                if lb <= mid:
                    h["t_akhir"] = t
                    break
        else:
            h["mfe"] = max(h["mfe"], mid - lb)
            if not baru_isi:
                h["mae"] = max(h["mae"], ha - mid)
            if not h["tp1"]:
                if ha >= sl and not baru_isi:
                    h["sl"], h["t_akhir"] = True, t
                    break
                if la <= tp1:
                    h["tp1"] = True
            else:
                if la <= tp2:
                    h["tp2"], h["t_akhir"] = True, t
                    break
                if ha >= mid:
                    h["t_akhir"] = t
                    break
    return h


def sinyal_masih_aktif(status_str):
    """True bila sinyal belum BATAL / KEDALUWARSA / SELESAI TP2 (tidak bergantung pencocokan substring rapuh)."""
    if not status_str or status_str == "UNKNOWN":
        return False
    if status_str in STATUS_TERMINAL_SINYAL:
        return False
    return not status_str.startswith(("BATAL", "SELESAI"))


def _status_dari_hasil(sig, h, bid, umur_jam):
    lo, hi = float(sig["lo"]), float(sig["hi"])
    sl, tp1 = float(sig["sl"]), float(sig["tp1"])
    tp2 = float(sig.get("tp2", tp1))
    if h.get("batal_sebelum_isi"):
        return ST_BATAL_SL, f"Harga menembus batas SL ({sl:.2f}) sebelum sempat menjemput area entry ({lo:.2f} - {hi:.2f}). Setup batal tanpa terisi."
    if h["terlewat"]:
        return ST_BATAL_TERLEWAT, f"Harga sudah memantul mencapai TP1 ({tp1:.2f}) tanpa menjemput area entry ({lo:.2f} - {hi:.2f}). Jangan dikejar."
    if h["sl"]:
        return ST_BATAL_SL, f"Sejak sinyal terbit harga pernah menyentuh SL ({sl:.2f}) (riwayat M1), walau harga sekarang {bid:.2f}. Setup tidak lagi valid."
    if h["tp2"]:
        return ST_SELESAI_TP2, f"Harga telah menyapu target penuh TP2 ({tp2:.2f}) setelah entry terjemput. Siklus setup selesai."
    if h["tp1"]:
        return ST_TP1_TERCAPAI, f"Harga telah mencapai TP1 ({tp1:.2f}) setelah entry terjemput. SL seharusnya sudah dikunci ke BEP+."
    if h["terisi"]:
        if lo <= bid <= hi:
            return ST_AKTIF_ENTRY, f"Harga ({bid:.2f}) berada di area entry ({lo:.2f} - {hi:.2f}); entry sudah terjemput."
        return ST_VALID_BERJALAN, f"Area entry ({lo:.2f} - {hi:.2f}) sudah tersentuh; belum menyentuh SL ({sl:.2f}) maupun TP1 ({tp1:.2f}). Harga sekarang {bid:.2f}."
    if umur_jam > SINYAL_TTL_BELUM_TERISI_JAM:
        return ST_BATAL_KEDALUWARSA, f"Belum terjemput setelah {umur_jam:.1f} jam (batas {SINYAL_TTL_BELUM_TERISI_JAM} jam). Kondisi pasar sudah berubah."
    if lo <= bid <= hi:
        return ST_AKTIF_ENTRY, f"Harga ({bid:.2f}) sedang berada di area entry ({lo:.2f} - {hi:.2f}). Siap eksekusi."
    return ST_VALID_PULLBACK, f"Harga ({bid:.2f}) belum menjemput area entry ({lo:.2f} - {hi:.2f}); menunggu pullback ke zona (SL {sl:.2f} belum tersentuh)."


def evaluasi_validitas_sinyal(sig, harga_bid):
    """Validitas sinyal. Jika sinyal punya cap waktu ('t_buat'), dievaluasi dari RIWAYAT harga M1 sejak terbit
    (SL yang pernah tersentuh tetap dihitung walau harga sudah kembali; entry yang tak terjemput; kedaluwarsa).
    Tanpa cap waktu (state lama) dipakai penilaian titik harga saat ini."""
    if not sig or not isinstance(sig, dict) or harga_bid <= 0:
        return "UNKNOWN", "Data sinyal atau harga tidak tersedia"
    t_buat = sig.get("t_buat")
    if t_buat and pastikan_mt5_aktif():
        try:
            tick = mt5.symbol_info_tick(SYMBOL)
            rates = _ambil_m1_sejak(int(t_buat))
            if tick and rates is not None and len(rates) > 0:
                h = _simulasi_jalur(sig, rates, max(float(tick.ask - tick.bid), 0.0))
                return _status_dari_hasil(sig, h, harga_bid, (tick.time - int(t_buat)) / 3600.0)
        except Exception as e:
            _catat_error("evaluasi_validitas_sinyal", e)
    return _evaluasi_validitas_titik(sig, harga_bid)


def _evaluasi_validitas_titik(sig, harga_bid):
    """(Penilaian titik harga tanpa data M1) Mengevaluasi apakah sinyal masih valid, kena SL, atau menunggu pullback."""
    if not sig or not isinstance(sig, dict) or harga_bid <= 0:
        return "UNKNOWN", "Data sinyal atau harga tidak tersedia"

    arah = sig.get("arah", "BUY").upper()
    sl = float(sig.get("sl", 0.0))
    tp1 = float(sig.get("tp1", 0.0))
    tp2 = float(sig.get("tp2", tp1))
    lo = float(sig.get("lo", 0.0))
    hi = float(sig.get("hi", 0.0))
    hb0 = sig.get("harga_buat")

    if arah == "BUY":
        if harga_bid <= sl:
            return ST_BATAL_SL, f"Harga ({harga_bid:.2f}) telah menembus batas Stop Loss ({sl:.2f}). Setup tidak lagi valid."
        elif lo <= harga_bid <= hi:
            return ST_AKTIF_ENTRY, f"Harga ({harga_bid:.2f}) sedang berada tepat di area entry ({lo:.2f} - {hi:.2f}). Siap eksekusi."
        elif harga_bid > hi:
            # Jika ini sinyal BUY Limit (harga_buat > hi) dan belum ada bukti terisi, maka harga > hi adalah MENUNGGU PULLBACK
            if hb0 is not None and float(hb0) > hi and "TP1 Tercapai" not in str(sig.get("ket", "")):
                return ST_VALID_PULLBACK, f"Harga ({harga_bid:.2f}) masih berada di atas area entry ({lo:.2f} - {hi:.2f}). Menunggu pullback menjemput zona."
            if harga_bid >= tp2:
                return ST_SELESAI_TP2, f"Harga ({harga_bid:.2f}) telah menyapu target penuh TP2 ({tp2:.2f}). Siklus setup selesai."
            if harga_bid >= tp1:
                return ST_TP1_TERCAPAI, f"Harga ({harga_bid:.2f}) telah mencapai target TP1 ({tp1:.2f}). SL seharusnya sudah dikunci ke BEP+."
            return ST_VALID_PULLBACK, f"Harga ({harga_bid:.2f}) berada di atas entry ({lo:.2f} - {hi:.2f}). Menunggu harga pullback menjemput zona."
        else:
            return ST_VALID_DEKAT_SL, f"Harga ({harga_bid:.2f}) berada di bawah entry namun masih bertahan di atas batas SL ({sl:.2f})."
    else:
        if harga_bid >= sl:
            return ST_BATAL_SL, f"Harga ({harga_bid:.2f}) telah menembus batas Stop Loss ({sl:.2f}). Setup tidak lagi valid."
        elif lo <= harga_bid <= hi:
            return ST_AKTIF_ENTRY, f"Harga ({harga_bid:.2f}) sedang berada tepat di area entry ({lo:.2f} - {hi:.2f}). Siap eksekusi."
        elif harga_bid < lo:
            if hb0 is not None and float(hb0) < lo and "TP1 Tercapai" not in str(sig.get("ket", "")):
                return ST_VALID_PULLBACK, f"Harga ({harga_bid:.2f}) masih berada di bawah area entry ({lo:.2f} - {hi:.2f}). Menunggu pullback menjemput zona."
            if harga_bid <= tp2:
                return ST_SELESAI_TP2, f"Harga ({harga_bid:.2f}) telah menyapu target penuh TP2 ({tp2:.2f}). Siklus setup selesai."
            if harga_bid <= tp1:
                return ST_TP1_TERCAPAI, f"Harga ({harga_bid:.2f}) telah mencapai target TP1 ({tp1:.2f}). SL seharusnya sudah dikunci ke BEP+."
            return ST_VALID_PULLBACK, f"Harga ({harga_bid:.2f}) berada di bawah entry ({lo:.2f} - {hi:.2f}). Menunggu harga pullback menjemput zona."
        else:
            return ST_VALID_DEKAT_SL, f"Harga ({harga_bid:.2f}) berada di atas entry namun masih bertahan di bawah batas SL ({sl:.2f})."


def putuskan_sinyal_terkunci(
    sinyal_kandidat,
    sinyal_aktif,
    status_aktif=None,
    paksa_ganti=False,
    izinkan_balik_arah=False,
    konteks="QA"
):
    """
    Evaluasi terpusat untuk SIGNAL LOCK & DIRECTIONAL LOCK (Jalur STARTUP, REFRESH, QA).

    Menentukan apakah sinyal_kandidat baru boleh menggantikan sinyal_aktif yang sedang berjalan.

    Parameter:
    - sinyal_kandidat: dict sinyal baru dari AI (atau None)
    - sinyal_aktif: dict sinyal aktif yang sedang dipegang sistem (atau None)
    - status_aktif: string status validitas sinyal_aktif (hasil evaluasi_validitas_sinyal)
    - paksa_ganti: bool, True jika trader secara eksplisit meminta 'sinyal baru'/'reset sinyal' (hanya di QA)
    - izinkan_balik_arah: bool, True jika trader eksplisit menyertakan kata kunci 'balik arah'/'reversal' (hanya di QA)
    - konteks: str, nama jalur pemanggil ('STARTUP', 'REFRESH', 'QA', atau 'PANTAU' di masa depan)

    Mengembalikan tuple:
      (sinyal_terpilih, sinyal_riwayat_baru, pesan_log, disetujui_ubah)
      - sinyal_terpilih: sinyal yang harus dipakai (kandidat baru atau sinyal_aktif yang dipertahankan).
      - sinyal_riwayat_baru: sinyal lama yang diarsipkan ke SINYAL_SEBELUMNYA_DINAMIS jika terjadi penggantian nyata,
                            atau None jika tidak ada pengarsipan.
      - pesan_log: list teks string untuk dicetak ke terminal (bila ada lock/update).
      - disetujui_ubah: boolean True bila sinyal_kandidat diterima menggantikan sinyal_aktif.
    """
    pesan_log = []
    masih_aktif = sinyal_masih_aktif(status_aktif) if (status_aktif and status_aktif != "UNKNOWN") else True

    if not sinyal_kandidat or not isinstance(sinyal_kandidat, dict) or not sinyal_kandidat.get("lo"):
        if sinyal_aktif and isinstance(sinyal_aktif, dict) and sinyal_aktif.get("lo") and not masih_aktif:
            st_lbl = status_aktif or "NON-AKTIF"
            pesan_log.append(
                f"\n   [ℹ️ RESET SINYAL/{konteks}: Sinyal sebelumnya ({sinyal_aktif['arah']} @ {sinyal_aktif['lo']:.2f}-{sinyal_aktif['hi']:.2f}) "
                f"sudah selesai/batal ({st_lbl}) dan belum ada sinyal baru yang lolos filter -> Status menjadi WAIT.]"
            )
            return None, sinyal_aktif, pesan_log, True
        return sinyal_aktif, None, pesan_log, False

    if not sinyal_aktif or not isinstance(sinyal_aktif, dict) or not sinyal_aktif.get("lo"):
        # Belum ada sinyal aktif (status WAIT) -> sinyal baru diterima
        return sinyal_kandidat, None, pesan_log, True

    beda_arah = (sinyal_kandidat["arah"] != sinyal_aktif["arah"])
    beda_angka = (
        beda_arah
        or abs(float(sinyal_kandidat["lo"]) - float(sinyal_aktif["lo"])) > 0.05
        or abs(float(sinyal_kandidat["hi"]) - float(sinyal_aktif["hi"])) > 0.05
        or abs(float(sinyal_kandidat["sl"]) - float(sinyal_aktif["sl"])) > 0.05
    )
    # Syarat pengarsipan ke SINYAL_SEBELUMNYA_DINAMIS: arah atau lo berbeda (toleransi 0.05)
    beda_arsip = (
        beda_arah
        or abs(float(sinyal_kandidat["lo"]) - float(sinyal_aktif["lo"])) > 0.05
    )

    # 1. KASUS SINYAL LAMA SUDAH SELESAI / BATAL / EXPIRED
    if not masih_aktif:
        sinyal_riwayat = sinyal_aktif if beda_arsip else None
        st_lbl = status_aktif or "NON-AKTIF"
        pesan_log.append(
            f"\n   [ℹ️ UPDATE SINYAL/{konteks}: Sinyal sebelumnya sudah selesai/batal ({st_lbl}). "
            f"AI menerbitkan Sinyal Baru ({sinyal_kandidat['arah']} @ {sinyal_kandidat['lo']:.2f}-{sinyal_kandidat['hi']:.2f}).]"
        )
        return sinyal_kandidat, sinyal_riwayat, pesan_log, True

    # 2. KASUS SINYAL LAMA MASIH AKTIF
    # 2A. Trader secara eksplisit meminta ganti sinyal (paksa_ganti=True, hanya di QA)
    if paksa_ganti:
        if beda_arah and not izinkan_balik_arah:
            # 🔒 DIRECTIONAL LOCK: Tahan pembalikan arah tanpa konfirmasi kata kunci balik arah
            pesan_log.append(
                f"\n   [🔒 DIRECTIONAL LOCK/{konteks}: Sinyal aktif tetap {sinyal_aktif['arah']} @ "
                f"{sinyal_aktif['lo']:.2f}-{sinyal_aktif['hi']:.2f}. Sinyal lawan ({sinyal_kandidat['arah']}) tidak diaktifkan "
                f"karena setup sebelumnya belum selesai ({status_aktif or 'AKTIF'}). Gunakan kata kunci 'balik arah' jika sengaja switch.]"
            )
            return sinyal_aktif, None, pesan_log, False
        else:
            sinyal_riwayat = sinyal_aktif if beda_arsip else None
            pesan_log.append(
                f"\n   [ℹ️ GANTI SINYAL/{konteks}: Sinyal aktif digantikan oleh setup baru "
                f"({sinyal_kandidat['arah']} @ {sinyal_kandidat['lo']:.2f}-{sinyal_kandidat['hi']:.2f}) atas permintaan trader.]"
            )
            return sinyal_kandidat, sinyal_riwayat, pesan_log, True

    # 2B. Normal (Startup, Refresh, atau QA pertanyaan biasa tanpa paksa ganti)
    if beda_angka:
        st_lbl = status_aktif or "MASIH VALID"
        pesan_log.append(
            f"\n   [🔒 SIGNAL LOCK/{konteks}: Sinyal baru AI ({sinyal_kandidat['arah']} @ {sinyal_kandidat['lo']:.2f}-{sinyal_kandidat['hi']:.2f}) "
            f"DITAHAN karena sinyal aktif ({sinyal_aktif['arah']} @ {sinyal_aktif['lo']:.2f}-{sinyal_aktif['hi']:.2f}, SL {sinyal_aktif['sl']:.2f}) "
            f"masih valid ({st_lbl})! Angka asli tetap dipertahankan.]"
        )
        return sinyal_aktif, None, pesan_log, False

    # Sinyal kandidat identik dengan sinyal aktif
    return sinyal_aktif, None, pesan_log, False


def cek_status_eksekusi_sinyal(sig, pos_list=None, order_list=None):
    """Mengecek apakah suatu sinyal AI diambil (open posisi), dipasang pending order, atau TIDAK DIAMBIL oleh trader di MT5.
    PENTING: Jangan menulis 'Tidak ada posisi terbuka di MT5' jika trader sebenarnya sedang memegang posisi arah berlawanan!"""
    if not sig or not isinstance(sig, dict) or not sig.get("lo"):
        return "TIDAK ADA SINYAL", "TIDAK ADA", None

    arah_s = sig.get("arah", "").upper()
    if pos_list is None and pastikan_mt5_aktif():
        pos_list = mt5.positions_get(symbol=SYMBOL) or ()
    if order_list is None and pastikan_mt5_aktif():
        order_list = mt5.orders_get(symbol=SYMBOL) or ()

    pos_list = pos_list or ()
    order_list = order_list or ()

    lo_s, hi_s = float(sig["lo"]), float(sig["hi"])
    mid_s = (lo_s + hi_s) / 2.0

    pos_searah = [
        p for p in pos_list
        if (arah_s == "BUY" and p.type == mt5.ORDER_TYPE_BUY)
        or (arah_s == "SELL" and p.type == mt5.ORDER_TYPE_SELL)
    ]
    pos_lawan = [
        p for p in pos_list
        if (arah_s == "BUY" and p.type == mt5.ORDER_TYPE_SELL)
        or (arah_s == "SELL" and p.type == mt5.ORDER_TYPE_BUY)
    ]
    ord_searah_dekat = [
        o for o in order_list
        if ((arah_s == "BUY" and "BUY" in _info_tipe_pending(o.type)[1])
            or (arah_s == "SELL" and "SELL" in _info_tipe_pending(o.type)[1]))
        and (lo_s - 5.0 <= o.price_open <= hi_s + 5.0)
    ]

    if pos_searah:
        p0 = min(pos_searah, key=lambda x: abs(x.price_open - mid_s))
        ringkasan = f"Ticket #{p0.ticket} ({p0.volume} lot @ {p0.price_open:.2f})"
        return f"SINYAL {arah_s} SEDANG DIAMBIL", "DIAMBIL", ringkasan
    elif ord_searah_dekat:
        o0 = min(ord_searah_dekat, key=lambda x: abs(x.price_open - mid_s))
        nm0 = _info_tipe_pending(o0.type)[0]
        ringkasan = f"{nm0} #{o0.ticket} @ {o0.price_open:.2f}"
        return f"SINYAL {arah_s} DIPASANG PENDING", "LIMIT", ringkasan
    else:
        if pos_lawan:
            arah_l = "BUY" if pos_lawan[0].type == mt5.ORDER_TYPE_BUY else "SELL"
            daftar_l = ", ".join(f"#{p.ticket} ({p.volume} lot @ {p.price_open:.2f})" for p in pos_lawan[:3])
            info_l = f"Sinyal {arah_s} ({lo_s:.2f}-{hi_s:.2f}) ini tidak diambil; Trader SEDANG MEMEGANG {len(pos_lawan)} POSISI {arah_l}: {daftar_l}"
            return f"SINYAL {arah_s} TIDAK DIAMBIL (TRADER PEGANG {len(pos_lawan)} POSISI {arah_l})", "TIDAK DIAMBIL", info_l
        if order_list:
            return f"SINYAL {arah_s} BELUM DIAMBIL", "TIDAK DIAMBIL", f"Tidak ada posisi/pending di {lo_s:.2f}-{hi_s:.2f} (ada {len(order_list)} pending order di level lain)"
        return f"SINYAL {arah_s} YANG TIDAK DIAMBIL", "TIDAK DIAMBIL", "Tidak ada posisi running untuk sinyal ini di MT5"


def evaluasi_limit_kena_kentut(sig, harga_bid=0.0, order_list=None):
    """Mengevaluasi kasus 'Entrian Kena Kentut' (Shallow Tap / Front-Run pada Zona Entry):
    Misal zona BUY 4107-4110, trader pasang Buy Limit di 4107, tapi harga hanya turun sampai 4108 lalu terbang naik.
    Mengembalikan dict:
      - status_kode: 'WAJIB_CANCEL' | 'MASIH_VALID_SEMENTARA' | 'PANDUAN_PASANG'
      - judul: ringkasan status tegas
      - catatan: penjelasan lengkap & level pasti kapan harus Cancel vs Tahan
      - batas_cancel: angka level 60% menuju TP1 (batas zona dianggap sudah mitigated/habis bensin)
      - rek_entry: rentang ideal pasang limit (50% equilibrium s/d tepi depan zona)
    """
    if not sig or not isinstance(sig, dict) or not sig.get("lo"):
        return None

    arah = str(sig.get("arah", "BUY")).upper()
    lo = float(sig["lo"])
    hi = float(sig["hi"])
    tp1 = float(sig["tp1"])
    mid = round((lo + hi) / 2.0, 2)

    if order_list is None and pastikan_mt5_aktif():
        order_list = mt5.orders_get(symbol=SYMBOL) or ()
    order_list = order_list or ()

    ord_searah = [
        o for o in order_list
        if ((arah == "BUY" and "BUY" in _info_tipe_pending(o.type)[1])
            or (arah == "SELL" and "SELL" in _info_tipe_pending(o.type)[1]))
        and (lo - 5.0 <= o.price_open <= hi + 5.0)
    ]
    o_dekat = min(ord_searah, key=lambda x: abs(x.price_open - mid)) if ord_searah else None

    # Cari titik terdalam harga yang pernah masuk ke zona sejak sinyal dibuat
    pernah_masuk_zona = False
    titik_terdalam = sig.get("_ekstrem_dalam_zona")
    t_buat = sig.get("t_buat")
    if t_buat and pastikan_mt5_aktif():
        try:
            rates = _ambil_m1_sejak(int(t_buat))
            if rates is not None and len(rates) > 0:
                if arah == "BUY":
                    lows_in = [float(r["low"]) for r in rates if lo - 0.5 <= float(r["low"]) <= hi]
                    if lows_in:
                        pernah_masuk_zona = True
                        titik_terdalam = min(lows_in) if titik_terdalam is None else min(float(titik_terdalam), min(lows_in))
                else:
                    highs_in = [float(r["high"]) for r in rates if lo <= float(r["high"]) <= hi + 0.5]
                    if highs_in:
                        pernah_masuk_zona = True
                        titik_terdalam = max(highs_in) if titik_terdalam is None else max(float(titik_terdalam), max(highs_in))
        except Exception:
            pass
    elif titik_terdalam is not None:
        pernah_masuk_zona = True

    if arah == "BUY":
        batas_cancel = round(hi + 0.60 * max(tp1 - hi, 1.0), 2)
        rek_lo, rek_hi = mid, round(hi - 0.15, 2) if (hi - mid) > 0.25 else hi
        lvl_limit = float(o_dekat.price_open) if o_dekat else lo
        lbl_limit = f"BUY LIMIT #{o_dekat.ticket} @ {lvl_limit:.2f}" if o_dekat else f"Buy Limit di ujung bawah ({lo:.2f})"

        # Cek apakah pernah colek zona (misal 4108) tapi limit di bawahnya (misal 4107) tidak kena, lalu harga naik keluar zona
        if pernah_masuk_zona and titik_terdalam is not None and titik_terdalam > lvl_limit and harga_bid > hi + 0.5:
            selisih_meleset = round(titik_terdalam - lvl_limit, 2)
            if harga_bid >= batas_cancel or harga_bid >= tp1:
                return dict(
                    status_kode="WAJIB_CANCEL",
                    judul="❌ WAJIB CANCEL BUY LIMIT SEKARANG (ZONA SUDAH MITIGATED / HABIS BENSIN!)",
                    catatan=(
                        f"Harga tadi sudah colek zona sampai {titik_terdalam:.2f} ({lbl_limit} meleset tipis {selisih_meleset:.2f} pt / 'kena kentut') "
                        f"dan sekarang sudah terbang jauh ke {harga_bid:.2f} (melewati Batas Mitigasi 60% di {batas_cancel:.2f} menuju TP1 {tp1:.2f}). "
                        f"Order institusi di zona {lo:.2f}-{hi:.2f} SUDAH TERPAKAI! Jika harga nanti turun balik ke {lvl_limit:.2f}, zona ini sudah tidak fresh & sangat rawan jebol SL. SEGERA CANCEL LIMIT ANDA!"
                    ),
                    batas_cancel=batas_cancel,
                    rek_entry=f"{rek_lo:.2f}-{rek_hi:.2f}",
                )
            else:
                return dict(
                    status_kode="MASIH_VALID_SEMENTARA",
                    judul=f"⚠️ MASIH VALID SEMENTARA (TUNGGU RETEST KE-2) — BATAS CANCEL DI {batas_cancel:.2f}",
                    catatan=(
                        f"Harga tadi baru colek tepi atas zona sampai {titik_terdalam:.2f} ({lbl_limit} kurang {selisih_meleset:.2f} pt / 'kena kentut') "
                        f"dan baru memantul ke {harga_bid:.2f} (BELUM melewati Batas Mitigasi 60% di {batas_cancel:.2f}). "
                        f"KEPUTUSAN: Limit Anda di {lvl_limit:.2f} MASIH VALID dipertahankan untuk menunggu retest/jarum kedua ke dalam zona, "
                        f"TETAPI WAJIB SEGERA CANCEL jika harga lanjut terbang menyentuh {batas_cancel:.2f} atau TP1 ({tp1:.2f})!"
                    ),
                    batas_cancel=batas_cancel,
                    rek_entry=f"{rek_lo:.2f}-{rek_hi:.2f}",
                )
        return dict(
            status_kode="PANDUAN_PASANG",
            judul=f"💡 TIPS ANTI 'KENA KENTUT' (BATAS CANCEL LIMIT: {batas_cancel:.2f})",
            catatan=(
                f"Agar tidak ketinggalan kereta di zona BUY {lo:.2f}-{hi:.2f}, jangan taruh 100% Buy Limit di dasar zona ({lo:.2f}). "
                f"Pasang di area tengah/tepi atas ({rek_lo:.2f} - {hi:.2f}) atau bagi 2 peluru. "
                f"ATURAN: Jika harga cuma colek tepi zona (mis. {mid:.2f}-{hi:.2f}) lalu memantul: selama masih di bawah {batas_cancel:.2f}, Limit bawah MASIH VALID ditunggu; "
                f"tapi jika sudah terbang melewati {batas_cancel:.2f} (60% ke TP1 {tp1:.2f}), WAJIB CANCEL LIMIT karena zona sudah terpakai (mitigated)!"
            ),
            batas_cancel=batas_cancel,
            rek_entry=f"{rek_lo:.2f}-{hi:.2f}",
        )
    else:
        batas_cancel = round(lo - 0.60 * max(lo - tp1, 1.0), 2)
        rek_lo, rek_hi = (round(lo + 0.15, 2) if (mid - lo) > 0.25 else lo), mid
        lvl_limit = float(o_dekat.price_open) if o_dekat else hi
        lbl_limit = f"SELL LIMIT #{o_dekat.ticket} @ {lvl_limit:.2f}" if o_dekat else f"Sell Limit di ujung atas ({hi:.2f})"

        if pernah_masuk_zona and titik_terdalam is not None and titik_terdalam < lvl_limit and 0 < harga_bid < lo - 0.5:
            selisih_meleset = round(lvl_limit - titik_terdalam, 2)
            if harga_bid <= batas_cancel or harga_bid <= tp1:
                return dict(
                    status_kode="WAJIB_CANCEL",
                    judul="❌ WAJIB CANCEL SELL LIMIT SEKARANG (ZONA SUDAH MITIGATED / HABIS BENSIN!)",
                    catatan=(
                        f"Harga tadi sudah colek zona sampai {titik_terdalam:.2f} ({lbl_limit} meleset tipis {selisih_meleset:.2f} pt / 'kena kentut') "
                        f"dan sekarang sudah terjun jauh ke {harga_bid:.2f} (melewati Batas Mitigasi 60% di {batas_cancel:.2f} menuju TP1 {tp1:.2f}). "
                        f"Order institusi di zona {lo:.2f}-{hi:.2f} SUDAH TERPAKAI! Jika harga nanti naik balik ke {lvl_limit:.2f}, zona ini sudah tidak fresh & sangat rawan jebol SL. SEGERA CANCEL LIMIT ANDA!"
                    ),
                    batas_cancel=batas_cancel,
                    rek_entry=f"{lo:.2f}-{rek_hi:.2f}",
                )
            else:
                return dict(
                    status_kode="MASIH_VALID_SEMENTARA",
                    judul=f"⚠️ MASIH VALID SEMENTARA (TUNGGU RETEST KE-2) — BATAS CANCEL DI {batas_cancel:.2f}",
                    catatan=(
                        f"Harga tadi baru colek tepi bawah zona sampai {titik_terdalam:.2f} ({lbl_limit} kurang {selisih_meleset:.2f} pt / 'kena kentut') "
                        f"dan baru turun ke {harga_bid:.2f} (BELUM melewati Batas Mitigasi 60% di {batas_cancel:.2f}). "
                        f"KEPUTUSAN: Limit Anda di {lvl_limit:.2f} MASIH VALID dipertahankan untuk menunggu retest/jarum kedua ke dalam zona, "
                        f"TETAPI WAJIB SEGERA CANCEL jika harga lanjut terjun menyentuh {batas_cancel:.2f} atau TP1 ({tp1:.2f})!"
                    ),
                    batas_cancel=batas_cancel,
                    rek_entry=f"{lo:.2f}-{rek_hi:.2f}",
                )
        return dict(
            status_kode="PANDUAN_PASANG",
            judul=f"💡 TIPS ANTI 'KENA KENTUT' (BATAS CANCEL LIMIT: {batas_cancel:.2f})",
            catatan=(
                f"Agar tidak ketinggalan kereta di zona SELL {lo:.2f}-{hi:.2f}, jangan taruh 100% Sell Limit di pucuk zona ({hi:.2f}). "
                f"Pasang di area tepi bawah/tengah ({lo:.2f} - {rek_hi:.2f}) atau bagi 2 peluru. "
                f"ATURAN: Jika harga cuma colek tepi zona (mis. {lo:.2f}-{mid:.2f}) lalu turun: selama masih di atas {batas_cancel:.2f}, Limit atas MASIH VALID ditunggu; "
                f"tapi jika sudah terjun melewati {batas_cancel:.2f} (60% ke TP1 {tp1:.2f}), WAJIB CANCEL LIMIT karena zona sudah terpakai (mitigated)!"
            ),
            batas_cancel=batas_cancel,
            rek_entry=f"{lo:.2f}-{rek_hi:.2f}",
        )


def _migrasi_header_jurnal():
    """Memeriksa dan memigrasikan header jurnal CSV lama jika kolom 'model', 'spread_buat', 'status_hasil', 'r_hasil', atau 'diambil' belum ada (dengan backup .bak)."""
    if not os.path.exists(FILE_JURNAL_SINYAL):
        return
    try:
        with open(FILE_JURNAL_SINYAL, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header or all(k in header for k in ("model", "spread_buat", "status_hasil", "r_hasil", "diambil")):
                return  # Sudah termigrasi lengkap atau file kosong
        # Buat backup file lama sebelum ditimpa
        shutil.copy2(FILE_JURNAL_SINYAL, FILE_JURNAL_SINYAL + ".bak")
        with open(FILE_JURNAL_SINYAL, "r", encoding="utf-8") as f:
            baris_lama = list(csv.DictReader(f))
        with open(FILE_JURNAL_SINYAL, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=KOLOM_JURNAL_SINYAL, extrasaction="ignore")
            writer.writeheader()
            for row in baris_lama:
                if not row.get("model"):
                    row["model"] = ""
                if not row.get("spread_buat"):
                    row["spread_buat"] = ""
                if not row.get("status_hasil"):
                    row["status_hasil"] = ""
                if not row.get("r_hasil"):
                    row["r_hasil"] = ""
                if not row.get("diambil"):
                    row["diambil"] = ""
                writer.writerow(row)
    except Exception as e:
        _catat_error("migrasi_header_jurnal", e)


def _catat_sinyal_csv(sig, catatan=""):
    """Tambahkan sinyal baru ke jurnal CSV (dasar evaluasi/backtest: perintah 'rekap' / 'jurnal')."""
    try:
        _migrasi_header_jurnal()
        baru = not os.path.exists(FILE_JURNAL_SINYAL)
        diambil_awal = str(sig.get("diambil") or "").strip()
        if not diambil_awal:
            try:
                _, kd_eks, info_eks = cek_status_eksekusi_sinyal(sig)
                if kd_eks == "DIAMBIL":
                    m_tkt = re.search(r"#(\d+)", str(info_eks))
                    diambil_awal = f"YA (#{m_tkt.group(1)})" if m_tkt else "YA"
                elif kd_eks == "LIMIT":
                    diambil_awal = "LIMIT"
                else:
                    diambil_awal = "-"
            except Exception:
                diambil_awal = "-"
        with open(FILE_JURNAL_SINYAL, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if baru:
                w.writerow(KOLOM_JURNAL_SINYAL)
            w.writerow([
                sig["t_buat"], datetime.now().strftime("%Y-%m-%d %H:%M:%S"), sig["arah"],
                sig["lo"], sig["hi"], sig.get("lo_asli", sig["lo"]), sig.get("hi_asli", sig["hi"]),
                sig["sl"], sig["tp1"], sig.get("tp2", sig["tp1"]), sig.get("harga_buat", ""),
                tentukan_klasifikasi_risiko_sinyal(sig)[0], str(sig.get("ket", ""))[:100], str(catatan)[:60],
                str(sig.get("model", ""))[:50],
                f"{float(sig['spread_buat']):.3f}" if sig.get("spread_buat") is not None else "",
                sig.get("status_hasil", "WAIT / RUNNING"),
                sig.get("r_hasil", ""),
                diambil_awal
            ])
    except Exception as e:
        _catat_error("catat_sinyal_csv", e)


def tandai_sinyal_diambil_jurnal(target_id_atau_no, status_baru="YA", mode_by_id=False):
    """Menandai atau memperbarui kolom 'diambil' pada jurnal CSV berdasarkan nomor urut tabel (1-based)
    atau berdasarkan ID timestamp ('t_buat' jika mode_by_id=True). Mengembalikan (berhasil, pesan_info)."""
    _migrasi_header_jurnal()
    if not os.path.exists(FILE_JURNAL_SINYAL):
        return False, "File jurnal belum ada."
    try:
        with open(FILE_JURNAL_SINYAL, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return False, "Jurnal masih kosong."

        diubah_list = []
        if mode_by_id:
            str_id = str(target_id_atau_no).strip()
            for idx, r in enumerate(rows, 1):
                if str(r.get("id", "")).strip() == str_id:
                    lama = str(r.get("diambil") or "").strip().upper()
                    if lama == "TIDAK (MANUAL)":
                        continue
                    if lama != status_baru:
                        r["diambil"] = status_baru
                        diubah_list.append(f"#{idx} ({r.get('arah')} {r.get('lo')}-{r.get('hi')})")
        else:
            # target_id_atau_no berupa list/int nomor baris (1-based)
            daftar_no = target_id_atau_no if isinstance(target_id_atau_no, (list, tuple, set)) else [target_id_atau_no]
            for no in daftar_no:
                idx0 = int(no) - 1
                if 0 <= idx0 < len(rows):
                    r = rows[idx0]
                    r["diambil"] = status_baru
                    diubah_list.append(f"#{no} ({r.get('arah')} @ {float(r.get('lo', 0)):.2f}-{float(r.get('hi', 0)):.2f})")

        if not diubah_list:
            return False, "Tidak ada nomor zona yang cocok atau status sudah sama."

        with open(FILE_JURNAL_SINYAL, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=KOLOM_JURNAL_SINYAL, extrasaction="ignore")
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        return True, ", ".join(diubah_list)
    except Exception as e:
        _catat_error("tandai_sinyal_diambil_jurnal", e)
        return False, f"Gagal menyimpan ke CSV: {e}"


def _cek_duplikat_jurnal_terbaru(sig, ts_skrg, cooldown_detik=900):
    """Cek apakah dalam `cooldown_detik` terakhir (default 15 menit) sudah ada sinyal searah
    dengan zona/SL yang berdekatan di jurnal CSV agar tidak tercatat ganda (anti-spam jurnal)."""
    if not os.path.exists(FILE_JURNAL_SINYAL):
        return None
    try:
        with open(FILE_JURNAL_SINYAL, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        arah_baru = str(sig.get("arah", "")).upper()
        mid_baru = (float(sig["lo"]) + float(sig["hi"])) / 2.0
        sl_baru = float(sig["sl"])
        for r in reversed(rows[-8:]):
            try:
                t_lama = int(r["id"])
                if abs(ts_skrg - t_lama) > cooldown_detik:
                    continue
                if str(r.get("arah", "")).upper() != arah_baru:
                    continue
                mid_lama = (float(r["lo"]) + float(r["hi"])) / 2.0
                sl_lama = float(r["sl"])
                if abs(mid_baru - mid_lama) <= 3.5 or abs(sl_baru - sl_lama) <= 3.0:
                    return r
            except (ValueError, KeyError):
                continue
    except Exception:
        pass
    return None


def _stempel_sinyal(sig, data_lama, catatan=""):
    """Beri cap waktu server ('t_buat'), harga saat terbit ('harga_buat'), & spread ('spread_buat') pada sinyal baru. Sinyal identik dengan yang
    sudah tersimpan mewarisi cap lamanya (supaya riwayat & kedaluwarsa tidak ter-reset oleh penerbitan ulang)."""
    if sig.get("_abaikan_jurnal"):
        return
    lama = (data_lama or {}).get("sinyal")
    if (isinstance(lama, dict) and lama.get("t_buat") and lama.get("arah") == sig.get("arah")
            and abs(float(lama.get("lo", 0)) - float(sig["lo"])) < 0.05
            and (abs(float(lama.get("sl", 0)) - float(sig["sl"])) < 0.05 or sig.get("sudah_tp1"))):
        sig["t_buat"] = lama["t_buat"]
        if lama.get("harga_buat") is not None:
            sig["harga_buat"] = lama["harga_buat"]
        if lama.get("spread_buat") is not None:
            sig["spread_buat"] = lama["spread_buat"]
        if lama.get("model") and not sig.get("model"):
            sig["model"] = lama["model"]
        return
    tick = mt5.symbol_info_tick(SYMBOL) if pastikan_mt5_aktif() else None
    if tick:
        arah_s = str(sig.get("arah", "BUY")).upper()
        sl_s = float(sig.get("sl", 0.0))
        # Jangan pernah mencatat sinyal yang saat lahir harganya sudah menembus SL
        if (arah_s == "BUY" and tick.bid <= sl_s) or (arah_s == "SELL" and tick.ask >= sl_s):
            return
        ts_now = int(tick.time)
        dup = _cek_duplikat_jurnal_terbaru(sig, ts_now, cooldown_detik=900)
        if dup is not None:
            sig["t_buat"] = int(dup["id"])
            sig["harga_buat"] = float(dup["harga_buat"]) if dup.get("harga_buat") else float(tick.bid)
            sig["spread_buat"] = float(dup["spread_buat"]) if dup.get("spread_buat") else round(float(tick.ask - tick.bid), 3)
            return
        sig["t_buat"] = ts_now
        sig["harga_buat"] = float(tick.bid)
        sig["spread_buat"] = round(float(tick.ask - tick.bid), 3)
        _catat_sinyal_csv(sig, catatan)


def gerbang_sinyal_baru(sig):
    """Gerbang keras (kode) untuk penerbitan sinyal baru: validitas SL, buffer ATR, jarak entry, spread, blackout berita High-impact USD,
    batas rugi harian, batas total risiko terbuka. Mengembalikan (ok, [alasan]). Bila MT5 tak tersedia: tidak memblokir."""
    alasan = []
    if not pastikan_mt5_aktif():
        return True, alasan
    tick = mt5.symbol_info_tick(SYMBOL)
    atr_val = 2.0
    df = ambil_data(mt5.TIMEFRAME_M5, 60, percobaan=1)
    if df is not None and len(df) >= 20:
        atr_val = float(tambah_indikator(df.copy())["ATR14"].iloc[-1])

    arah_cek = str(sig.get("arah", "")).upper()
    lo_cek, hi_cek = float(sig.get("lo", 0.0)), float(sig.get("hi", 0.0))
    sl_cek = float(sig.get("sl", 0.0))
    tp1_cek = float(sig.get("tp1", 0.0))
    tp2_cek = float(sig.get("tp2", tp1_cek))

    if tick and tick.bid > 0:
        # 1. Pastikan harga live belum menjebol SL atau sudah melewati TP1
        if (arah_cek == "BUY" and tick.bid <= sl_cek) or (arah_cek == "SELL" and tick.ask >= sl_cek):
            alasan.append(f"harga live ({tick.bid:.2f}) sudah menembus level SL ({sl_cek:.2f})")
        if (arah_cek == "BUY" and tick.bid >= tp1_cek) or (arah_cek == "SELL" and tick.bid <= tp1_cek):
            alasan.append(f"harga live ({tick.bid:.2f}) sudah mencapai/melewati TP1 ({tp1_cek:.2f})")

        # 2. Pastikan buffer SL dari tepi luar zona cukup aman terhadap volatilitas (ATR M5) agar tidak mudah kena sweep tipis
        min_buf_sl = max(2.2, round(0.9 * atr_val, 2))
        if arah_cek == "BUY":
            if (lo_cek - sl_cek) < min_buf_sl:
                sl_aman = round(lo_cek - min_buf_sl, 2)
                mid_c = (lo_cek + hi_cek) / 2.0
                if (mid_c - sl_aman) <= SL_JARAK_MAKS:
                    sig["sl"] = sl_aman
                    sl_cek = sl_aman
                    risiko_baru = mid_c - sl_aman
                    if (tp1_cek - mid_c) / max(risiko_baru, 0.1) < RR_MIN:
                        sig["tp1"] = round(mid_c + RR_MIN * risiko_baru, 2)
                        sig["tp2"] = round( max(tp2_cek, mid_c + 2.6 * risiko_baru), 2)
        elif arah_cek == "SELL":
            if (sl_cek - hi_cek) < min_buf_sl:
                sl_aman = round(hi_cek + min_buf_sl, 2)
                mid_c = (lo_cek + hi_cek) / 2.0
                if (sl_aman - mid_c) <= SL_JARAK_MAKS:
                    sig["sl"] = sl_aman
                    sl_cek = sl_aman
                    risiko_baru = sl_aman - mid_c
                    if (mid_c - tp1_cek) / max(risiko_baru, 0.1) < RR_MIN:
                        sig["tp1"] = round(mid_c - RR_MIN * risiko_baru, 2)
                        sig["tp2"] = round(min(tp2_cek, mid_c - 2.6 * risiko_baru), 2)

        mid = (sig.get("lo_asli", sig["lo"]) + sig.get("hi_asli", sig["hi"])) / 2.0
        if abs(mid - tick.bid) > ENTRY_JARAK_MAKS:
            alasan.append(f"entry {mid:.2f} terlalu jauh dari harga {tick.bid:.2f} (> {ENTRY_JARAK_MAKS:g} pt)")
        spread = float(tick.ask - tick.bid)
        batas = min(SPREAD_MAKS_POIN, max(0.5, SPREAD_MAKS_RASIO_ATR * atr_val))
        if spread > batas:
            alasan.append(f"spread {spread:.2f} pt terlalu lebar (batas {batas:.2f} pt)")
    blk = berita_blackout_aktif()
    if blk:
        alasan.append(f"blackout berita High-impact USD: {blk}")
    st = status_risiko_akun()
    if st:
        if st["rugi_hari_pct"] >= MAKS_RUGI_HARIAN_PCT:
            alasan.append(f"batas rugi harian tercapai ({st['rugi_hari_pct']:.2f}% >= {MAKS_RUGI_HARIAN_PCT:g}%)")
        if st["risiko_terbuka_pct"] >= MAKS_RISIKO_TERBUKA_PCT:
            alasan.append(f"total risiko terbuka {st['risiko_terbuka_pct']:.2f}% >= {MAKS_RISIKO_TERBUKA_PCT:g}%")
    # --- Filter Trend: sinyal harus searah bias HTF ---
    if FILTER_TREND_AKTIF:
        try:
            bias = bias_htf_cached()
            arah = str(sig.get("arah", "")).upper()
            lawan = (arah == "BUY" and bias["bias"] == "BEARISH") or (arah == "SELL" and bias["bias"] == "BULLISH")
            if lawan:
                if IZINKAN_COUNTER_TREND_MEDIUM:
                    kat, _ = tentukan_klasifikasi_risiko_sinyal(sig)
                    if kat == "MEDIUM RISK":
                        alasan.append(f"⚠️ COUNTER-TREND ({arah} vs bias HTF {bias['bias']}, skor {bias['skor']:+d}) — diizinkan karena MEDIUM RISK")
                    else:
                        alasan.append(f"melawan trend HTF ({arah} vs bias HTF {bias['bias']}, skor {bias['skor']:+d}) — aktifkan IZINKAN_COUNTER_TREND_MEDIUM untuk mengizinkan")
                else:
                    alasan.append(f"melawan trend HTF ({arah} vs bias HTF {bias['bias']}, skor {bias['skor']:+d}) — ubah FILTER_TREND_AKTIF=False untuk menonaktifkan")
        except Exception as e:
            _catat_error("gerbang_trend", e)
    # --- Filter Sesi: sinyal baru hanya diizinkan di sesi London & New York ---
    if FILTER_SESI_AKTIF and tick and getattr(tick, "time", 0) > 0:
        offset = SERVER_UTC_OFFSET_JAM
        if offset is None:
            offset = deteksi_offset(tick, df if "df" in locals() else None)
        if offset is not None:
            ts_utc = int(tick.time) - int(offset * 3600)
            dt_utc = datetime.fromtimestamp(ts_utc, tz=timezone.utc)
            jam_utc = dt_utc.hour + dt_utc.minute / 60.0
            di_sesi = any(awal <= jam_utc < akhir for awal, akhir in SESI_DIIZINKAN_UTC)
            if not di_sesi:
                alasan.append(f"di luar sesi London/New York (jam UTC {dt_utc.strftime('%H:%M')})")
    return (not alasan), alasan


def terapkan_gerbang(sig):
    """Kembalikan sig bila lolos gerbang; None bila ditolak (alasan dicetak). Alasan penolakan parser ikut dilaporkan."""
    if sig is None:
        if _STATE_PARSER.get("ditolak"):
            print("\n   [🛑 VALIDASI SINYAL: sinyal AI ditolak — " + "; ".join(_STATE_PARSER["ditolak"]) + "]")
            _STATE_PARSER["ditolak"] = []
        return None
    ok, alasan = gerbang_sinyal_baru(sig)
    if ok:
        return sig
    print("\n   [🛑 GERBANG SINYAL: sinyal AI tidak diterbitkan — " + "; ".join(alasan) + "]")
    return None


def evaluasi_dan_perbarui_jurnal_sinyal():
    """Mengevaluasi setiap sinyal di jurnal CSV terhadap riwayat pergerakan harga M1 dari MT5.
    Memperbarui dan menyimpan kolom 'status_hasil' (HIT SL, HIT TP1, HIT TP2, RUNNING, TERLEWAT, WAIT)
    serta 'r_hasil' langsung ke file CSV agar dapat menjadi acuan permanen jurnal trading."""
    _migrasi_header_jurnal()
    if not os.path.exists(FILE_JURNAL_SINYAL):
        return [], False

    try:
        with open(FILE_JURNAL_SINYAL, "r", encoding="utf-8") as f:
            rows_raw = list(csv.DictReader(f))
    except Exception as e:
        _catat_error("baca_jurnal_sinyal", e)
        return [], False

    if not rows_raw:
        return [], False

    # Bersihkan baris duplikat beruntun (mis. 2 tiket SL di detik yang sama pada zona & SL yang sama)
    rows = []
    perubahan = False
    for r in rows_raw:
        if rows:
            prev = rows[-1]
            try:
                if (str(r.get("arah", "")).upper() == str(prev.get("arah", "")).upper()
                        and abs(int(r["id"]) - int(prev["id"])) <= 120
                        and abs(float(r["sl"]) - float(prev["sl"])) <= 1.0
                        and abs(float(r["lo"]) - float(prev["lo"])) <= 2.0):
                    perubahan = True
                    continue
            except (ValueError, KeyError):
                pass
        rows.append(r)

    mt5_aktif = pastikan_mt5_aktif()
    tick = mt5.symbol_info_tick(SYMBOL) if mt5_aktif else None
    spread_live = float(tick.ask - tick.bid) if tick else 0.0

    pos_live = (mt5.positions_get(symbol=SYMBOL) or ()) if mt5_aktif else ()
    orders_live = (mt5.orders_get(symbol=SYMBOL) or ()) if mt5_aktif else ()
    deals_in = []
    if mt5_aktif:
        try:
            all_deals = mt5.history_deals_get(datetime(2020, 1, 1), datetime(2030, 1, 1)) or ()
            deals_in = [
                d for d in all_deals
                if getattr(d, "symbol", "") == SYMBOL
                and getattr(d, "entry", 0) in (getattr(mt5, "DEAL_ENTRY_IN", 0), getattr(mt5, "DEAL_ENTRY_INOUT", 2))
                and getattr(d, "type", -1) in (getattr(mt5, "DEAL_TYPE_BUY", 0), getattr(mt5, "DEAL_TYPE_SELL", 1))
            ]
        except Exception as e_dl:
            _catat_error("evaluasi_deals_jurnal", e_dl)

    # 1. Inferensi historis dari catatan Stop-Hunt Recovery Posisi #<ticket>:
    # Jika pada baris k tercatat "Stop-Hunt Recovery Posisi BUY/SELL #<ticket>", berarti trader sebelumnya
    # benar-benar mengambil setup searah tersebut yang baru saja terkena SL.
    sudah_dipasangkan_recovery = set()
    for k_idx, r_k in enumerate(rows):
        cat_k = str(r_k.get("catatan") or "")
        m_rec = re.search(r"Stop-Hunt Recovery Posisi\s+(BUY|SELL)\s+#(\d+)", cat_k, re.I)
        if not m_rec:
            continue
        arah_rec = m_rec.group(1).upper()
        tkt_rec = m_rec.group(2)
        try:
            t_k = int(r_k["id"])
            lo_k = float(r_k["lo"])
            hi_k = float(r_k["hi"])
        except (ValueError, KeyError):
            continue
        kandidat_j = None
        jarak_terbaik = 999999.0
        for j_idx in range(k_idx - 1, -1, -1):
            if j_idx in sudah_dipasangkan_recovery:
                continue
            r_j = rows[j_idx]
            if str(r_j.get("arah") or "").upper() != arah_rec:
                continue
            try:
                t_j = int(r_j["id"])
                lo_j = float(r_j["lo"])
                hi_j = float(r_j["hi"])
                sl_j = float(r_j["sl"])
            except (ValueError, KeyError):
                continue
            if not (0 <= (t_k - t_j) <= 6 * 3600):
                continue
            # Pada sinyal recovery: BUY memiliki hi_k ~ open_r dan lo_k ~ sl_r; SELL memiliki lo_k ~ open_r dan hi_k ~ sl_r
            ref_open = hi_k if arah_rec == "BUY" else lo_k
            ref_sl = lo_k if arah_rec == "BUY" else hi_k
            mid_j = (lo_j + hi_j) / 2.0
            skor_jarak = min(abs(mid_j - ref_open), abs(lo_j - ref_open), abs(hi_j - ref_open)) + 0.5 * abs(sl_j - ref_sl)
            if skor_jarak < jarak_terbaik:
                jarak_terbaik = skor_jarak
                kandidat_j = j_idx
        target_idx = kandidat_j if (kandidat_j is not None and jarak_terbaik <= 18.0) else k_idx
        sudah_dipasangkan_recovery.add(target_idx)
        val_lama = str(rows[target_idx].get("diambil") or "").strip().upper()
        if val_lama not in ("TIDAK (MANUAL)", "YA (MANUAL)") and not val_lama.startswith("YA"):
            rows[target_idx]["diambil"] = f"YA (#{tkt_rec})"
            perubahan = True

    deals_terpakai = set()
    for row in rows:
        try:
            sp_buat = float(row["spread_buat"]) if (row.get("spread_buat") and str(row["spread_buat"]).strip()) else None
            sig = dict(
                arah=row["arah"],
                lo=float(row["lo"]),
                hi=float(row["hi"]),
                lo_asli=float(row.get("lo_asli") or row["lo"]),
                hi_asli=float(row.get("hi_asli") or row["hi"]),
                sl=float(row["sl"]),
                tp1=float(row["tp1"]),
                tp2=float(row.get("tp2") or row["tp1"]),
                t_buat=int(row["id"]),
                harga_buat=float(row["harga_buat"]) if row.get("harga_buat") else None,
                spread_buat=sp_buat
            )
        except (ValueError, KeyError):
            continue

        # 2. Deteksi otomatis kolom 'diambil' dari MT5 Deal History & Posisi/Order Aktif
        diambil_skrg = str(row.get("diambil") or "").strip()
        diambil_upper = diambil_skrg.upper()
        arah_u = str(sig["arah"]).upper()
        if diambil_upper not in ("TIDAK (MANUAL)", "YA (MANUAL)"):
            status_ambil_baru = diambil_skrg if diambil_skrg else "-"
            # Cek dari riwayat transaksi MT5 (DEAL_ENTRY_IN) jika belum tercatat YA
            if mt5_aktif and deals_in and not status_ambil_baru.upper().startswith("YA"):
                for d in deals_in:
                    if d.ticket in deals_terpakai:
                        continue
                    arah_d = "BUY" if d.type == getattr(mt5, "DEAL_TYPE_BUY", 0) else "SELL"
                    if arah_d != arah_u:
                        continue
                    t_deal = int(getattr(d, "time", 0))
                    if (sig["t_buat"] - 300) <= t_deal <= (sig["t_buat"] + 6 * 3600):
                        if (sig["lo"] - 4.5) <= float(d.price) <= (sig["hi"] + 4.5):
                            deals_terpakai.add(d.ticket)
                            tkt_id = getattr(d, "position_id", 0) or getattr(d, "ticket", 0)
                            status_ambil_baru = f"YA (#{tkt_id})" if tkt_id else "YA"
                            break
            # Cek dari posisi live / pending order jika sinyal masih baru
            if mt5_aktif and not status_ambil_baru.upper().startswith("YA"):
                if tick and abs(int(tick.time) - sig["t_buat"]) <= 12 * 3600:
                    _, kd_live, info_live = cek_status_eksekusi_sinyal(sig, pos_live, orders_live)
                    if kd_live == "DIAMBIL":
                        m_t = re.search(r"#(\d+)", str(info_live))
                        status_ambil_baru = f"YA (#{m_t.group(1)})" if m_t else "YA"
                    elif kd_live == "LIMIT":
                        status_ambil_baru = "LIMIT"
            if row.get("diambil") != status_ambil_baru:
                row["diambil"] = status_ambil_baru
                perubahan = True

        st_lama = str(row.get("status_hasil") or "").strip().upper()
        hb_awal = sig.get("harga_buat")
        # Jika saat sinyal dicatat harga_buat sudah berada di luar batas SL (sinyal usang yang tidak pernah terisi)
        if hb_awal is not None and ((arah_u == "BUY" and hb_awal <= sig["sl"]) or (arah_u == "SELL" and hb_awal >= sig["sl"])):
            if row.get("status_hasil") != "TERLEWAT" or row.get("r_hasil") != "0.00R":
                row["status_hasil"] = "TERLEWAT"
                row["r_hasil"] = "0.00R"
                perubahan = True
            continue

        # Jika MT5 aktif, evaluasi ulang riwayat candle M1 sejak sinyal terbit
        if mt5_aktif:
            rates = _ambil_m1_sejak(sig["t_buat"])
            if rates is not None and len(rates) > 0:
                spread_eval = sig["spread_buat"] if sig["spread_buat"] is not None else spread_live
                h = _simulasi_jalur(sig, rates, spread_eval)
                mid = (sig["lo_asli"] + sig["hi_asli"]) / 2.0
                risiko = abs(mid - sig["sl"]) or 1.0

                baru_st = "WAIT"
                baru_r = "0.00R"
                if h.get("batal_sebelum_isi"):
                    baru_st = "TERLEWAT"
                    baru_r = "0.00R"
                elif h["sl"]:
                    baru_st = "HIT SL"
                    baru_r = "-1.00R"
                elif h["tp2"]:
                    gain = abs(sig["tp2"] - mid) / risiko
                    baru_st = "HIT TP2"
                    baru_r = f"+{gain:.2f}R"
                elif h["tp1"]:
                    gain = abs(sig["tp1"] - mid) / risiko
                    baru_st = "HIT TP1"
                    baru_r = f"+{gain:.2f}R"
                elif h["terisi"]:
                    baru_st = "RUNNING"
                    baru_r = "0.00R"
                elif h["terlewat"]:
                    baru_st = "TERLEWAT"
                    baru_r = "0.00R"
                else:
                    baru_st = "WAIT"
                    baru_r = "0.00R"

                if row.get("status_hasil") != baru_st or row.get("r_hasil") != baru_r:
                    row["status_hasil"] = baru_st
                    row["r_hasil"] = baru_r
                    perubahan = True
        elif not st_lama:
            row["status_hasil"] = "WAIT / BELUM CEK"
            row["r_hasil"] = "0.00R"

    # Simpan kembali pembaruan hasil evaluasi ke file CSV jika ada perubahan
    if perubahan:
        try:
            with open(FILE_JURNAL_SINYAL, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=KOLOM_JURNAL_SINYAL, extrasaction="ignore")
                writer.writeheader()
                for r in rows:
                    writer.writerow(r)
        except PermissionError:
            print("   [⚠️ File jurnal CSV sedang dibuka di program lain (mis. Excel). Tutup file agar hasil evaluasi tersimpan permanen.]")
        except Exception as e:
            _catat_error("simpan_evaluasi_jurnal", e)

    return rows, mt5_aktif


def _format_label_diambil(raw_diambil):
    """Mengubah nilai kolom 'diambil' dari CSV menjadi label singkat dan jelas untuk tabel."""
    val = str(raw_diambil or "").strip()
    up = val.upper()
    if up.startswith("YA"):
        m_t = re.search(r"#(\d+)", val)
        if m_t:
            return f"✅ DIAMBIL #{m_t.group(1)}", True
        return "✅ DIAMBIL", True
    if up == "LIMIT":
        return "📌 LIMIT", False
    return "⚪ -", False


def susun_tabel_jurnal_markdown(rows):
    """Menghasilkan teks tabel Markdown untuk jurnal sinyal & evaluasi zona TP/SL."""
    if not rows:
        return "*(Belum ada sinyal yang tercatat di jurnal)*\n"
    md = [
        "| No | Waktu Terbit | Arah | Zona Entry | SL | TP1 / TP2 | Diambil? | Status Hasil | R Gain | Risiko | Setup / Keterangan |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |"
    ]
    for idx, row in enumerate(rows, 1):
        arah = (row.get("arah") or "").upper()
        wkt = str(row.get("waktu_lokal") or "")[:16]
        lo = float(row.get("lo") or 0.0)
        hi = float(row.get("hi") or 0.0)
        sl = float(row.get("sl") or 0.0)
        tp1 = float(row.get("tp1") or 0.0)
        tp2 = float(row.get("tp2") or tp1)
        st = str(row.get("status_hasil") or "WAIT").strip().upper()
        r = str(row.get("r_hasil") or "-").strip()
        risiko = str(row.get("risiko") or "-").strip()
        ket = str(row.get("ket") or row.get("catatan") or "-").strip().replace("|", "/")
        lbl_ambil, _ = _format_label_diambil(row.get("diambil"))

        if "HIT TP2" in st:
            ikon = f"🟢 **HIT TP2 (TP Penuh)**"
        elif "HIT TP1" in st:
            ikon = f"🟢 **HIT TP1**"
        elif "HIT SL" in st or "SL" in st:
            ikon = f"🔴 **HIT SL (Kalah)**"
        elif "RUNNING" in st:
            ikon = f"🟡 RUNNING"
        elif "TERLEWAT" in st:
            ikon = f"⚪ TERLEWAT"
        else:
            ikon = f"⏳ {st}"

        md.append(f"| {idx} | {wkt} | {arah} | {lo:.2f} - {hi:.2f} | {sl:.2f} | {tp1:.2f} / {tp2:.2f} | {lbl_ambil} | {ikon} | {r} | {risiko} | {ket} |")
    return "\n".join(md) + "\n"


def ringkasan_jurnal_untuk_ai(jumlah=5):
    """Menyusun ringkasan singkat sinyal terakhir dari jurnal CSV untuk dimasukkan ke konteks prompt AI."""
    _migrasi_header_jurnal()
    if not os.path.exists(FILE_JURNAL_SINYAL):
        return ""
    try:
        with open(FILE_JURNAL_SINYAL, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return ""
    if not rows:
        return ""

    terakhir = rows[-jumlah:]
    teks = ["[RIWAYAT EVALUASI ZONA SINYAL TERAKHIR (HASIL TP / SL DARI JURNAL)]:\n"]
    for idx, r in enumerate(terakhir, 1):
        st = r.get("status_hasil") or "BELUM EVALUASI"
        lbl_a, _ = _format_label_diambil(r.get("diambil"))
        teks.append(
            f"{idx}. {r.get('waktu_lokal', '')[:16]} | {r.get('arah', '')} @ {r.get('lo', '')}-{r.get('hi', '')} | "
            f"SL {r.get('sl', '')} | TP1 {r.get('tp1', '')} | Eksekusi: [{lbl_a}] -> HASIL: [{st}] ({r.get('r_hasil', '')}) | Setup: {r.get('ket', r.get('catatan', ''))}"
        )
    teks.append("Gunakan fakta historis di atas sebagai acuan performa zona sebelumnya agar analisa Anda konsisten.\n")
    return "\n".join(teks)


def tampilkan_rekap():
    """Menampilkan tabel detail semua zona sinyal di jurnal CSV:
    keterangan mana zona yang diambil oleh user, mana zona yang kena SL dan mana yang kena TP,
    lengkap dengan win-rate, total R, serta ringkasan per kelas risiko & model AI."""
    rows, mt5_aktif = evaluasi_dan_perbarui_jurnal_sinyal()
    if not rows:
        print("\n   Belum ada sinyal tercatat di jurnal CSV.")
        return

    lebar_garis = 138
    print("\n" + "=" * lebar_garis)
    print("📖 JURNAL SINYAL & EVALUASI ZONA (ACUAN HASIL HIT TP vs HIT SL + PENANDA ZONA YANG DIAMBIL)")
    print("=" * lebar_garis)
    print(f"{'#':<3} | {'Waktu Lokal':<16} | {'Arah':<4} | {'Zona Entry':<19} | {'SL':<8} | {'TP1 / TP2':<19} | {'Diambil?':<14} | {'Status Hasil':<16} | {'Hasil R':<8} | {'Setup / Keterangan'}")
    print("-" * lebar_garis)

    tot_sinyal = len(rows)
    tot_trade = 0
    tot_menang = 0
    tot_kalah = 0
    tot_r = 0.0
    tot_running = 0
    tot_terlewat = 0

    ambil_n = 0
    ambil_trade = 0
    ambil_menang = 0
    ambil_kalah = 0
    ambil_r = 0.0
    ambil_nomor_list = []

    kelas_stat = {}
    model_stat = {}

    for idx, row in enumerate(rows, 1):
        arah = (row.get("arah") or "").upper()
        wkt_str = str(row.get("waktu_lokal") or "")[:16]
        lo = float(row.get("lo") or 0.0)
        hi = float(row.get("hi") or 0.0)
        sl = float(row.get("sl") or 0.0)
        tp1 = float(row.get("tp1") or 0.0)
        tp2 = float(row.get("tp2") or tp1)
        r_val_str = str(row.get("r_hasil") or "").strip()
        st_hasil = str(row.get("status_hasil") or "WAIT").strip().upper()
        risiko_kat = str(row.get("risiko") or "UNKNOWN").strip().upper()
        nama_mdl = str(row.get("model") or "").strip() or "TIDAK DIKETAHUI"
        ket_teks = str(row.get("ket") or row.get("catatan") or "-").strip()
        lbl_ambil, is_diambil = _format_label_diambil(row.get("diambil"))

        # Format keterangan & ikon status hasil
        if "HIT TP2" in st_hasil:
            ikon_st = "🟢 HIT TP2 (Penuh)"
        elif "HIT TP1" in st_hasil:
            ikon_st = "🟢 HIT TP1"
        elif "HIT SL" in st_hasil or "SL" in st_hasil:
            ikon_st = "🔴 HIT SL"
        elif "RUNNING" in st_hasil:
            ikon_st = "🟡 RUNNING"
        elif "TERLEWAT" in st_hasil:
            ikon_st = "⚪ TERLEWAT"
        else:
            ikon_st = "⏳ WAIT / CEK"

        # Parsing nilai R numerik untuk statistik
        r_num = 0.0
        try:
            if r_val_str and "R" in r_val_str.upper():
                r_num = float(r_val_str.upper().replace("R", "").replace("+", "").strip())
                if "HIT SL" in st_hasil:
                    r_num = -abs(r_num) if r_num != 0 else -1.0
            elif "HIT SL" in st_hasil:
                r_num = -1.0
        except ValueError:
            r_num = 0.0

        # Tabulasi kelas & model
        k = kelas_stat.setdefault(risiko_kat, dict(n=0, menang=0, kalah=0, running=0, terlewat=0, total_r=0.0))
        k["n"] += 1
        m = model_stat.setdefault(nama_mdl, dict(n=0, menang=0, kalah=0, running=0, terlewat=0, total_r=0.0))
        m["n"] += 1

        if is_diambil:
            ambil_n += 1
            ambil_nomor_list.append(f"#{idx} ({arah} {lo:.2f}-{hi:.2f})")

        if "HIT TP" in st_hasil:
            tot_menang += 1
            tot_trade += 1
            tot_r += r_num
            k["menang"] += 1
            k["total_r"] += r_num
            m["menang"] += 1
            m["total_r"] += r_num
            if is_diambil:
                ambil_menang += 1
                ambil_trade += 1
                ambil_r += r_num
        elif "HIT SL" in st_hasil:
            tot_kalah += 1
            tot_trade += 1
            tot_r += r_num
            k["kalah"] += 1
            k["total_r"] += r_num
            m["kalah"] += 1
            m["total_r"] += r_num
            if is_diambil:
                ambil_kalah += 1
                ambil_trade += 1
                ambil_r += r_num
        elif "RUNNING" in st_hasil:
            tot_running += 1
            k["running"] += 1
            m["running"] += 1
        else:
            tot_terlewat += 1
            k["terlewat"] += 1
            m["terlewat"] += 1

        zona_str = f"{lo:.2f} - {hi:.2f}"
        tp_str = f"{tp1:.2f} / {tp2:.2f}"
        ket_pendek = (ket_teks[:32] + "..") if len(ket_teks) > 34 else ket_teks
        r_display = f"{r_val_str:>7}" if r_val_str else "      -"
        ambil_col = "✅ DIAMBIL" if is_diambil else ("📌 LIMIT" if "LIMIT" in lbl_ambil else "⚪ -")

        print(f"{idx:>3} | {wkt_str:<16} | {arah:<4} | {zona_str:<19} | {sl:>8.2f} | {tp_str:<19} | {ambil_col:<13} | {ikon_st:<16} | {r_display} | {ket_pendek}")

    print("=" * lebar_garis)
    wr_total = f"{(tot_menang / tot_trade * 100):.1f}%" if tot_trade > 0 else "-"
    exp_total = f"{(tot_r / tot_trade):+.2f}R" if tot_trade > 0 else "-"
    print("📊 RINGKASAN PERFORMA JURNAL SINYAL (SEMUA SINYAL):")
    print(f"• Total Sinyal Tercatat : {tot_sinyal} sinyal")
    print(f"• Selesai (Trade Closed): {tot_trade} trade (🟢 Menang: {tot_menang} | 🔴 Kalah: {tot_kalah})")
    print(f"• Win-Rate Keseluruhan  : {wr_total}")
    print(f"• Akumulasi Profit/Loss : {tot_r:+.2f}R")
    print(f"• Expectancy per Trade  : {exp_total} / trade")
    print(f"• Posisi Berjalan/Wait  : {tot_running} Running | {tot_terlewat} Terlewat/Belum Terisi")

    print("-" * lebar_garis)
    wr_ambil = f"{(ambil_menang / ambil_trade * 100):.1f}%" if ambil_trade > 0 else "-"
    daftar_ambil_str = ", ".join(ambil_nomor_list) if ambil_nomor_list else "Belum ada zona yang ditandai diambil"
    print("🎯 RINGKASAN KHUSUS ZONA YANG ANDA AMBIL (✅ DIAMBIL):")
    print(f"• Zona Diambil Tercatat : {ambil_n} zona -> {daftar_ambil_str}")
    print(f"• Hasil Eksekusi Anda   : Selesai {ambil_trade} trade (🟢 Menang: {ambil_menang} | 🔴 Kalah: {ambil_kalah}) | Win-Rate Anda: {wr_ambil} | Total Hasil: {ambil_r:+.2f}R")
    print("  💡 Cara Menandai Manual: Ketik 'ambil <nomor>' (contoh: ambil 7 atau ambil 7,8,20) untuk menandai zona yang Anda ambil,")
    print("                          atau ketik 'skip <nomor>' (contoh: skip 9) untuk menghapus tanda diambil.")
    if not mt5_aktif:
        print("  *(MT5 saat ini offline; status di atas diambil dari riwayat jurnal tersimpan terakhir)*")

    print("-" * lebar_garis)
    print("BREAKDOWN PER KELAS RISIKO:")
    for nama_k, stat_k in sorted(kelas_stat.items()):
        tr_k = stat_k["menang"] + stat_k["kalah"]
        wr_k = f"{(stat_k['menang'] / tr_k * 100):.0f}%" if tr_k > 0 else "-"
        ex_k = f"{(stat_k['total_r'] / tr_k):+.2f}R" if tr_k > 0 else "-"
        print(f"• {nama_k:<12} : {stat_k['n']:>2} sinyal | Selesai {tr_k:>2} (Menang {stat_k['menang']}, Kalah {stat_k['kalah']}) | Win-Rate {wr_k} | Total {stat_k['total_r']:+.2f}R | Exp {ex_k}/tr | Running {stat_k['running']}")

    if len(model_stat) > 1 or any(k != "TIDAK DIKETAHUI" for k in model_stat):
        print("-" * lebar_garis)
        print("BREAKDOWN PER MODEL AI:")
        for nama_m, stat_m in sorted(model_stat.items()):
            tr_m = stat_m["menang"] + stat_m["kalah"]
            wr_m = f"{(stat_m['menang'] / tr_m * 100):.0f}%" if tr_m > 0 else "-"
            ex_m = f"{(stat_m['total_r'] / tr_m):+.2f}R" if tr_m > 0 else "-"
            print(f"• {nama_m:<25} : {stat_m['n']:>2} sinyal | Selesai {tr_m:>2} (Menang {stat_m['menang']}, Kalah {stat_m['kalah']}) | Win-Rate {wr_m} | Total {stat_m['total_r']:+.2f}R | Exp {ex_m}/tr")
    print("=" * lebar_garis)


def tentukan_klasifikasi_risiko_sinyal(sig):
    """Menentukan apakah sinyal berkategori HIGH RISK (sinyal dadakan), MEDIUM RISK, atau LOW RISK."""
    if not sig or not isinstance(sig, dict):
        return "UNKNOWN", "Tidak ada sinyal"

    # 1. Jika ada explicit key 'risiko'
    if sig.get("risiko"):
        r_val = str(sig["risiko"]).upper()
        if "HIGH" in r_val:
            return "HIGH RISK", sig.get("risiko_detail", "Sinyal Dadakan LTF M5 (Tanpa Pijakan HTF Lengkap)")
        elif "LOW" in r_val:
            return "LOW RISK", sig.get("risiko_detail", "Konfluensi Golden Zone HTF + LTF Trigger Matang")
        elif "MEDIUM" in r_val:
            return "MEDIUM RISK", sig.get("risiko_detail", "Reaksi Intraday Terkonfirmasi di Zona Supply/Demand")

    ket = str(sig.get("ket", "")).upper()
    # 2. Deteksi sinyal dadakan di LTF M5 atau re-entry stop hunt
    if any(k in ket for k in ("DADAKAN", "OPEN DELIVERY", "WICK", "STOP-HUNT", "RE-ENTRY", "2ND CHANCE")):
        return "HIGH RISK", "Sinyal Dadakan LTF M5 (Pembalikan Cepat / Re-entry Spontan)"
    elif any(k in ket for k in ("GOLDEN", "4 PILAR", "SWING UTAMA", "LOW RISK")):
        return "LOW RISK", "Setup A+ Golden Confluence HTF"
    elif any(k in ket for k in ("DEMAND", "SUPPLY", "ZONA", "REAKSI", "OB", "FVG", "SUPPORT", "RESISTANCE", "CHOCH", "BOS")):
        return "MEDIUM RISK", "Reaksi Intraday Terkonfirmasi di Zona Supply/Demand"

    return "HIGH RISK", "Sinyal Dadakan Intraday M5"
