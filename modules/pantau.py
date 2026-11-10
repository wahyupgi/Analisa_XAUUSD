# -*- coding: utf-8 -*-
"""
modules/pantau.py
Mode Monitoring Real-Time: loop pemantauan tick/harga MT5, alert BEP / trailing SL,
peringatan dini posisi berisiko, bedah forensik saat zona jebol / kena SL,
dan otomatisasi pencarian serta pengaktifan zona institusional berikutnya.
"""
import csv
import math
import os
import re
import shutil
import sys
import time
from datetime import datetime
import MetaTrader5 as mt5
import numpy as np
import pandas as pd

from modules.config import (
    SYMBOL, TIMEFRAMES, SWING_N, BIAS_MIN_SKOR,
    FILE_JURNAL_SINYAL, FILE_ERROR_LOG,
    ST_BATAL_SL, ST_SELESAI_TP2, ST_TP1_TERCAPAI,
    ST_AKTIF_ENTRY, ST_VALID_BERJALAN, ST_VALID_PULLBACK, ST_VALID_DEKAT_SL
)
from modules.utils import bunyi, wkt, _catat_error
from modules.mt5_conn import (
    pastikan_mt5_aktif, ambil_data, tambah_indikator,
    cek_posisi_terbuka_mt5, _info_tipe_pending, _nilai_per_poin, hitung_mm_sinyal
)
from modules.analisis_teknikal import (
    struktur, order_blocks, deteksi_cisd, liquidity_sweeps,
    deteksi_pola, deteksi_institusi_dan_ranging,
    bias_htf_cached, invalidasi_cache_bias, _cari_zona_valid_berikutnya_mt5
)
from modules.sinyal_jurnal import (
    evaluasi_validitas_sinyal, cek_status_eksekusi_sinyal,
    tentukan_klasifikasi_risiko_sinyal, evaluasi_limit_kena_kentut,
    tandai_sinyal_diambil_jurnal, gerbang_sinyal_baru
)
from modules.ai_gemini import ekstrak_target_ai
from modules.state_manager import (
    simpan_state_sinyal, muat_state_sinyal,
    SINYAL_AKTIF_DINAMIS, SINYAL_SEBELUMNYA_DINAMIS,
    ZONA_ATAS_DINAMIS, ZONA_BAWAH_DINAMIS, CATATAN_JEBOL_PANTAU
)
import modules.state_manager as SM


def diagnosa_forensik_penyebab_sl(df_m5):
    """Mendiagnosa penyebab pasti kena SL atau jebolnya zona:
    1. Apakah karena ORDERAN BESAR INSTITUSI / IMPULS BERITA? (Volume spike >= 1.5x, Marubozu bodi >= 1.3x ATR, One-Way Impulse M5)
    2. Atau karena PERGERAKAN NORMAL / RETRACEMENT BIASA? (tanpa lonjakan institusi)
    Mengembalikan dict berisi: is_institusi, alasan_singkat, faktor, detail, tipe_zona, saran_tindakan, waktu."""
    d = tambah_indikator(df_m5.copy())
    O, H, L, C, V = (d[k].values for k in ("Open", "High", "Low", "Close", "Volume"))
    atr = d["ATR14"].values
    vol_ma = pd.Series(V).rolling(20, min_periods=5).mean().values
    vol_ratio = V[-3:] / np.maximum(vol_ma[-3:], 1)
    bodi_ratio = np.abs(C[-3:] - O[-3:]) / np.maximum(atr[-3:], 1e-6)
    rng_3 = H[-3:].max() - L[-3:].min()
    waktu_jebol = wkt(d.index[-1])

    info_inst = deteksi_institusi_dan_ranging(df_m5, atr)
    ada_impuls_inst = "PERINGATAN LONJAKAN MOMENTUM (TICK-VOLUME, BUKAN DATA INSTITUSI)" in info_inst
    max_vol = float(np.max(vol_ratio))
    max_bodi = float(np.max(bodi_ratio))

    faktor = []
    is_institusi = False
    if max_vol >= 1.5:
        faktor.append(f"Lonjakan Volume Institusi ({max_vol:.1f}x lipat rata-rata)")
        is_institusi = True
    if max_bodi >= 1.3:
        faktor.append(f"Candle Impuls Marubozu Besar ({max_bodi:.1f}x ATR M5)")
        is_institusi = True
    if ada_impuls_inst:
        faktor.append("Institutional One-Way Impulse Aktif (Candle searah bertubi-tubi)")
        is_institusi = True

    if is_institusi:
        alasan_singkat = "ORDERAN BESAR INSTITUSI / IMPULS VOLATILITAS TINGGI / BERITA"
        tipe_zona = "ZONA PALING VALID & KUAT (HTF MAJOR FORTRESS ZONE)"
        saran_tindakan = "Zona minor terdekat DIABAIKAN karena rawan bablas tertabrak arus! Mengarahkan ke Benteng Zona Terkuat (HTF Major Confluence)."
    else:
        alasan_singkat = "PERGERAKAN PASAR NORMAL / RETRACEMENT BIASA (NON-INSTITUSI)"
        tipe_zona = "ZONA TERDEKAT PERSIS PASCA-SL (IMMEDIATE NEARBY ZONE)"
        saran_tindakan = "Tekanan normal tanpa lonjakan institusi besar. Mengaktifkan Zona Valid Terdekat tepat setelah batas level SL."
        faktor.append("Tekanan pergerakan normal (Volume dan ukuran bodi candle terkendali)")

    detail = (
        f"Diagnosa Penyebab: {alasan_singkat} ({', '.join(faktor)})\n"
        f"Data Forensik 3 Candle M5: Rentang gerak {rng_3:.2f} pt | Volume Max {max_vol:.1f}x | Bodi Max {max_bodi:.1f}x ATR | RSI14 {d['RSI14'].iloc[-1]:.1f}"
    )

    return {
        "is_institusi": is_institusi,
        "alasan_singkat": alasan_singkat,
        "faktor": faktor,
        "detail": detail,
        "tipe_zona": tipe_zona,
        "saran_tindakan": saran_tindakan,
        "waktu": waktu_jebol,
    }


def bedah_forensik_jebol(df_m5, ket_zona, harga_jebol):
    """Menganalisis data candle M5 tepat saat zona jebol (apakah karena volume institusi, marubozu, atau berita)."""
    diag = diagnosa_forensik_penyebab_sl(df_m5)
    return (
        f"ZONA YANG BARU SAJA JEBOL: {ket_zona} | Harga Close Penjebolan: {harga_jebol:.2f} pada candle M5 {diag['waktu']}\n"
        f"{diag['detail']}\n"
        f"Tindakan Logis AI: {diag['saran_tindakan']}"
    )


def _aktifkan_zona_valid_berikutnya(otomatis, sudah_notif_masuk, arah_cari, batas_lewat, harga_skrg, atr_m5, alasan_jebol, df_m5=None, hanya_preview=False):
    """Mendiagnosa penyebab SL/jebol, lalu mencari dan mengaktifkan Zona Selanjutnya sesuai logika:
    - Kena institusi -> Zona Paling Valid & Kuat
    - Kena normal -> Zona Terdekat persis setelah SL."""
    global ZONA_ATAS_DINAMIS, ZONA_BAWAH_DINAMIS

    diag = diagnosa_forensik_penyebab_sl(df_m5) if df_m5 is not None else None
    is_institusi = diag["is_institusi"] if diag else False

    utama, cadangan = _cari_zona_valid_berikutnya_mt5(arah_cari, batas_lewat, harga_skrg, atr_m5, is_institusi=is_institusi)
    if not utama:
        return None

    lbl_jenis = "DEMAND / BUY" if arah_cari == "BUY" else "SUPPLY / SELL"
    print("\n   " + "━" * 70)
    if diag:
        icon_sebab = "🚨" if diag["is_institusi"] else "✅"
        print(f"   [{icon_sebab} INVESTIGASI FORENSIK PENYEBAB KENAL SL / JEBOL]")
        print(f"   • Pemicu Teknis      : {diag['alasan_singkat']}")
        print(f"   • Fakta Data M5      : {', '.join(diag['faktor'])}")
        print(f"   • Strategi Pemilihan : {diag['tipe_zona']}")
        print(f"   • Pertimbangan AI    : {diag['saran_tindakan']}")
        print("   " + "─" * 70)

    if hanya_preview:
        print(f"   [🔭 RADAR CADANGAN: ZONA {lbl_jenis} BERIKUTNYA (JIKA ZONA SAAT INI JEBOL TOTAL)]")
        print("   • STATUS SISTEM      : Zona/Sinyal saat ini MASIH BERJALAN. Ini HANYA peta cadangan darurat jika lantai jebol!")
        label_setup = "Rencana Setup Cadangan"
    else:
        print(f"   [⏳ ZONA {lbl_jenis} BERIKUTNYA DISIAGAKAN — WATCHLIST (Menunggu Harga Menjemput)]")
        label_setup = "Setup Sinyal Baru"
    print(f"   • Pemicu             : {alasan_jebol}")
    print(f"   • Zona Rekomendasi   : {utama['lo']:.2f} - {utama['hi']:.2f} (Jarak ~{utama['jarak']:.2f} pt dari harga {harga_skrg:.2f})")
    print(f"   • Kategori Zona      : {utama.get('tipe_strategi', 'Valid')}")
    print(f"   • Dasar Konfluensi   : {utama['sumber']}")
    print(f"   • {label_setup:<19} : {utama['arah']} @ {utama['lo']:.2f}-{utama['hi']:.2f} | SL/Invalidasi: {utama['sl']:.2f} | TP1: {utama['tp1']:.2f} | TP2: {utama['tp2']:.2f}")
    mm_u = hitung_mm_sinyal(utama['lo'], utama['hi'], utama['sl'], utama['tp1'], utama['tp2'])
    if mm_u:
        print(f"   • Rasio Risk/Reward  : TP1 1:{mm_u['rrr_tp1']:.1f} | TP2 1:{mm_u['rrr_tp2']:.1f} (Jarak SL: {mm_u['jarak_sl']:.2f} pt)")
        print(f"   • Rekomendasi Lot MM : {mm_u['lot_05']:.2f} lot (0.5% modal / -${mm_u['risk_05_usd']:.2f}) s/d {mm_u['lot_10']:.2f} lot (1.0% modal / -${mm_u['risk_10_usd']:.2f})")
        print(f"   • Batas Toleransi MM : Maksimal lot {mm_u['lot_max']:.2f} lot (Maks 2.0% modal). DILARANG over-lot!")
    if hanya_preview:
        print(f"   • Catatan Eksekusi   : JANGAN terburu-buru entri di {utama['lo']:.2f}-{utama['hi']:.2f}! Zona atas masih aktif.")
    else:
        print(f"   • Protokol Eksekusi  : Tunggu harga menjemput {utama['lo']:.2f}-{utama['hi']:.2f} + muncul Rejection / Sweep / CISD M5!")
    if cadangan:
        mm_c = hitung_mm_sinyal(cadangan['lo'], cadangan['hi'], cadangan['sl'], cadangan['tp1'], cadangan['tp2'])
        print("   " + "─" * 70)
        print(f"   [🛡️ ZONA CADANGAN BERIKUTNYA — JIKA ZONA DI ATAS TERKENA SL LAGI]")
        print(f"   • Area Pertahanan Ke-2 : {cadangan['arah']} @ {cadangan['lo']:.2f} - {cadangan['hi']:.2f} (Jarak ~{cadangan['jarak']:.2f} pt dari harga sekarang)")
        print(f"   • Batas SL & Target TP : SL {cadangan['sl']:.2f} | TP1 {cadangan['tp1']:.2f} | TP2 {cadangan['tp2']:.2f}")
        if mm_c:
            print(f"   • Rasio R:R & Lot MM   : R:R TP1 1:{mm_c['rrr_tp1']:.1f} | TP2 1:{mm_c['rrr_tp2']:.1f} | Lot Disiplin: {mm_c['lot_05']:.2f} lot (0.5% / -${mm_c['risk_05_usd']:.2f}) s/d {mm_c['lot_10']:.2f} lot (1% / -${mm_c['risk_10_usd']:.2f})")
        print(f"   • Dasar Konfluensi     : {cadangan['sumber']}")

        sebab_sl_lagi = "penurunan institusional lanjutan" if arah_cari == "BUY" else "lonjakan dorongan bullish lanjutan"
        target_likuiditas = "kolam likuiditas diskon HTF yang lebih dalam" if arah_cari == "BUY" else "kolam likuiditas premium HTF yang lebih tinggi"
        print(f"   • ALASAN PEMILIHAN ZONA CADANGAN :")
        print(f"     1) Analisa Skenario   : Jika Zona Rekomendasi Utama ({utama['lo']:.2f}-{utama['hi']:.2f}) gagal dan terkena SL lagi,")
        print(f"        maka pasar mengonfirmasi adanya {sebab_sl_lagi} yang berupaya menjemput {target_likuiditas}.")
        print(f"     2) Kekuatan Benteng   : Zona {cadangan['lo']:.2f}-{cadangan['hi']:.2f} diperkuat oleh {cadangan['sumber']},")
        print(f"        menjadikannya benteng 'Golden Fortress' terakhir dengan probabilitas pantulan tinggi untuk menahan harga.")
        print(f"     3) Protokol Eksekusi  : JANGAN langsung pasang Limit pasif di sini sekarang! Jika zona utama jebol,")
        print(f"        tunggu harga menyentuh zona cadangan ini dan memunculkan Rejection / Sweep / CISD M5 baru eksekusi.")
    print("   " + "━" * 70)

    if not hanya_preview:
        if arah_cari == "BUY":
            otomatis["bawah"] = (utama["lo"], utama["hi"], utama["jebol"])
            ZONA_BAWAH_DINAMIS = otomatis["bawah"]
            sudah_notif_masuk.discard("bawah")
        else:
            otomatis["atas"] = (utama["lo"], utama["hi"], utama["jebol"])
            ZONA_ATAS_DINAMIS = otomatis["atas"]
            sudah_notif_masuk.discard("atas")
        # ⚠️ PENTING: Zona baru hanya di-WATCHLIST (menunggu harga menjemput).
        # Sinyal TIDAK langsung aktif — biarkan Section 3 (cek otomatis["atas"/"bawah"])
        # yang mengaktifkan sinyal ketika harga benar-benar menyentuh zona tersebut.
        _update_sinyal_aktif_pantau(
            otomatis,
            sudah_notif_masuk,
            None,  # Set WAIT — sinyal aktif hanya ketika harga masuk zona
            f"Zona {lbl_jenis} ({utama['lo']:.2f}-{utama['hi']:.2f}) DISIAGAKAN — Menunggu harga menjemput untuk entry konfirmasi",
        )
    return utama


def _update_sinyal_aktif_pantau(otomatis, sudah_notif_masuk, sinyal_baru, alasan):
    """Memperbarui Sinyal Aktif secara otomatis di dalam Mode Pantau ketika terjadi peringatan penting."""
    global SINYAL_AKTIF_DINAMIS
    sinyal_lama = otomatis.get("sinyal")

    if sinyal_baru and isinstance(sinyal_baru, dict):
        is_bep_update = bool(sinyal_baru.get("sudah_tp1") or "TP1 Tercapai" in str(sinyal_baru.get("ket", "")))
        if is_bep_update:
            sinyal_baru["sudah_tp1"] = True
            sinyal_baru["_abaikan_jurnal"] = True
            if isinstance(sinyal_lama, dict):
                for k_inh in ("t_buat", "lo_asli", "hi_asli", "harga_buat", "spread_buat", "model"):
                    if k_inh in sinyal_lama and k_inh not in sinyal_baru:
                        sinyal_baru[k_inh] = sinyal_lama[k_inh]
        else:
            # Semua sinyal baru otomatis di Mode Pantau WAJIB lolos gerbang validasi (Trend HTF, Spread, Blackout, Buffer SL)
            ok_g, alasan_g = gerbang_sinyal_baru(sinyal_baru)
            if not ok_g:
                print(f"\n   [🛑 FILTER AKURASI PANTAU: Sinyal otomatis {sinyal_baru.get('arah')} ({sinyal_baru.get('ket', '')}) DITAHAN — {'; '.join(alasan_g)}]")
                return False

    otomatis["sinyal"] = sinyal_baru
    SINYAL_AKTIF_DINAMIS = sinyal_baru
    simpan_state_sinyal(sinyal_baru, atas=otomatis.get("atas"), bawah=otomatis.get("bawah"), catatan=alasan)

    if sinyal_baru and isinstance(sinyal_baru, dict) and sinyal_baru.get("sudah_tp1"):
        # Untuk update BEP+ pasca TP1: pertahankan status entry & tp1, dan kunci agar tidak memicu re-entry stop-hunt
        sudah_notif_masuk.discard("sig_sl_wick")
        sudah_notif_masuk.add("sig_sh_reentry")
    else:
        is_reentry_sig = bool(
            sinyal_baru and isinstance(sinyal_baru, dict)
            and any(k in str(sinyal_baru.get("ket", "")).upper() for k in ("RE-ENTRY", "2ND CHANCE", "STOP-HUNT"))
        )
        for k_flag in ("sig_entry", "sig_sl_wick", "sig_sh_reentry", "tp1"):
            sudah_notif_masuk.discard(k_flag)
        # Jika ini sudah merupakan sinyal Re-Entry 2nd Chance, kunci agar TIDAK BISA memicu Re-Entry ke-3/ke-4!
        if is_reentry_sig:
            sudah_notif_masuk.add("sig_sh_reentry")

    print("\n   " + "=" * 66)
    print("   [🔄 SINYAL AKTIF BERUBAH OTOMATIS DI MODE PANTAU!]")
    print(f"   • Pemicu Perubahan : {alasan}")
    if sinyal_baru:
        ket_s = sinyal_baru.get("ket", "Setup Aktif")
        kat_r, ket_r = tentukan_klasifikasi_risiko_sinyal(sinyal_baru)
        tag_icon = "⚠️" if kat_r == "HIGH RISK" else ("🛡️" if kat_r == "LOW RISK" else "⚡")

        print(f"   • Sinyal Baru      : {sinyal_baru['arah']} ({ket_s})")
        print(f"   • TINGKAT RISIKO   : {tag_icon} [{kat_r}] — {ket_r}")
        if kat_r == "HIGH RISK":
            print("     -> ⚠️ PERTIMBANGAN TRADER: Ini adalah Sinyal Dadakan LTF M5. Anda bebas SKIP/LEWATKAN jika ingin aman, ATAU ambil dengan lot mini!")
        print(f"   • Area Entry Baru  : {sinyal_baru['lo']:.2f} - {sinyal_baru['hi']:.2f}")
        print(f"   • Stop Loss Baru   : {sinyal_baru['sl']:.2f}")
        print(f"   • Target TP Baru   : TP1 {sinyal_baru['tp1']:.2f} | TP2 {sinyal_baru['tp2']:.2f}")

        # Protokol Money Management Presisi Berdasarkan Akun MT5 Live
        mm = hitung_mm_sinyal(sinyal_baru['lo'], sinyal_baru['hi'], sinyal_baru['sl'], sinyal_baru.get('tp1', 0), sinyal_baru.get('tp2', 0))
        if mm:
            print("   " + "─" * 66)
            print("   [💼 PROTOKOL MONEY MANAGEMENT & UKURAN LOT AKUN MT5]")
            print(f"   • Modal Real (Equity) : ${mm['equity']:,.2f} {mm['currency']}")
            print(f"   • Jarak Risiko SL     : {mm['jarak_sl']:.2f} pt ({mm['jarak_sl']*10:.0f} pips)")
            print(f"   • Rasio Risk/Reward   : TP1 1:{mm['rrr_tp1']:.1f} | TP2 1:{mm['rrr_tp2']:.1f}")
            print(f"   • REKOMENDASI UKURAN LOT MT5:")
            if kat_r == "HIGH RISK":
                print(f"     -> ⚠️ KHUSUS HIGH RISK   : {mm.get('lot_025', mm['lot_05']):.2f} lot (Maks risiko loss: -${mm.get('risk_025_usd', mm['risk_05_usd']):.2f}) [0.25% Modal]")
                print(f"     -> 🛑 OPSI BIJAK TRADER : Boleh SKIP (TIDAK DIAMBIL) tanpa penyesalan!")
            print(f"     -> DISIPLIN (0.5% Modal) : {mm['lot_05']:.2f} lot (Maks risiko loss: -${mm['risk_05_usd']:.2f}) [DIANJURKAN]")
            print(f"     -> STANDAR  (1.0% Modal) : {mm['lot_10']:.2f} lot (Maks risiko loss: -${mm['risk_10_usd']:.2f})")
            print(f"   • BATAS KERAS (SAFETY)     : DILARANG pasang lot > {mm['lot_max']:.2f} lot (Batas risiko maks 2.0% modal)!")
            print("   " + "─" * 66)
    else:
        print("   • Sinyal Baru      : WAIT / DIBATALKAN (Menunggu setup/konfirmasi baru yang aman)")
    print("   " + "=" * 66)


def mode_pantau_mt5(arg="", teks_ai_terakhir=""):
    """Memantau otomatis rekomendasi AI, angka manual, DAN mengawal posisi trading yang sedang running:
    1) Harga masuk zona entry + cek reaksi M5 (pantul / CHoCH / Sweep)
    2) Deteksi jika zona DITEMBUS (Breakout / SL jebol) -> beri peringatan & tawarkan analisa ulang otomatis!
    3) Deteksi jika menyentuh TP1 / TP2
    4) Pengawal Posisi Running Otomatis (Auto-Guard): Peringatan BEP, Partial TP, Bahaya Struktur Berlawanan, & Risiko SL.
    5) Sinyal Aktif Dinamis: Sinyal Aktif otomatis berubah mengikuti peringatan Stop-Hunt, Impuls, Konfirmasi Zona, atau Breakout."""
    global SINYAL_AKTIF_DINAMIS, ZONA_ATAS_DINAMIS, ZONA_BAWAH_DINAMIS, CATATAN_JEBOL_PANTAU
    kata = arg.split()[1:] if len(arg.split()) > 1 else []
    angka = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", " ".join(kata))]
    otomatis = ekstrak_target_ai(teks_ai_terakhir)
    if SINYAL_AKTIF_DINAMIS is not None:
        otomatis["sinyal"] = SINYAL_AKTIF_DINAMIS
    elif otomatis["sinyal"] is not None:
        SINYAL_AKTIF_DINAMIS = otomatis["sinyal"]
    if ZONA_ATAS_DINAMIS is not None:
        otomatis["atas"] = ZONA_ATAS_DINAMIS
    elif otomatis["atas"] is not None:
        ZONA_ATAS_DINAMIS = otomatis["atas"]
    if ZONA_BAWAH_DINAMIS is not None:
        otomatis["bawah"] = ZONA_BAWAH_DINAMIS
    elif otomatis["bawah"] is not None:
        ZONA_BAWAH_DINAMIS = otomatis["bawah"]

    lo_manual = min(angka[:2]) if len(angka) >= 2 else (angka[0] - 1.5 if len(angka) == 1 else None)
    hi_manual = max(angka[:2]) if len(angka) >= 2 else (angka[0] + 1.5 if len(angka) == 1 else None)
    sl_manual = angka[2] if len(angka) >= 3 else None

    if not pastikan_mt5_aktif():
        print("Gagal konek MT5 untuk mode pantau.")
        return None

    print("\n" + "=" * 70)
    print("MODE PANTAU PINTAR & PENGAWAL POSISI RUNNING (SMART RADAR ALERT) AKTIF")

    pos_awal = mt5.positions_get(symbol=SYMBOL) or ()
    orders_awal = mt5.orders_get(symbol=SYMBOL) or ()
    tick_awal = mt5.symbol_info_tick(SYMBOL)
    harga_awal_pantau = tick_awal.bid if tick_awal else 0.0

    # Tampilkan riwayat sinyal sebelumnya (jika ada) di awal mode pantau
    state_pantau = muat_state_sinyal()
    sig_prev_pantau = SINYAL_SEBELUMNYA_DINAMIS or (state_pantau.get("sinyal_sebelumnya") if state_pantau else None)
    if sig_prev_pantau and isinstance(sig_prev_pantau, dict) and sig_prev_pantau.get("lo"):
        st_prev, al_prev = evaluasi_validitas_sinyal(sig_prev_pantau, harga_awal_pantau)
        lbl_eks_prev, _, info_eks_prev = cek_status_eksekusi_sinyal(sig_prev_pantau, pos_awal, orders_awal)
        print("─" * 70)
        print(f"  📜 RIWAYAT SINYAL SEBELUMNYA [{lbl_eks_prev}]:")
        print(f"     {sig_prev_pantau['arah']} @ {sig_prev_pantau['lo']:.2f} - {sig_prev_pantau['hi']:.2f} | SL {sig_prev_pantau['sl']:.2f} | TP1 {sig_prev_pantau['tp1']:.2f} | TP2 {sig_prev_pantau.get('tp2', sig_prev_pantau['tp1']):.2f}")
        print(f"     Status: [{st_prev}] -> {al_prev}")
        print(f"     Sumber: {sig_prev_pantau.get('ket', '-')}")
        print("─" * 70)

    if lo_manual is not None:
        batas_atas = sl_manual if (sl_manual and sl_manual > hi_manual) else hi_manual + 3.0
        batas_bawah = sl_manual if (sl_manual and sl_manual < lo_manual) else lo_manual - 3.0
        print(f"- Target Zona Manual : {lo_manual:.2f} - {hi_manual:.2f}")
        print(f"- Batas Jebol (Break): Di atas {batas_atas:.2f} atau di bawah {batas_bawah:.2f} (close M5)")
    else:
        print("- Membaca otomatis level rekomendasi dari AI terakhir:")
        if otomatis["sinyal"]:
            s = otomatis["sinyal"]
            lbl_eksekusi, kode_eks, info_det = cek_status_eksekusi_sinyal(s, pos_awal, orders_awal)
            print(f"  * Sinyal {s['arah']} Aktif [{lbl_eksekusi}]: Entry {s['lo']:.2f}-{s['hi']:.2f} | SL: {s['sl']:.2f} | TP1: {s['tp1']:.2f} | TP2: {s['tp2']:.2f}")
            if kode_eks == "TIDAK DIAMBIL":
                print(f"    -> [INFO STATUS AKUN]: Sinyal {s['arah']} ini BELUM Anda ambil di MT5 (belum ada posisi terbuka).")
            else:
                print(f"    -> [INFO STATUS AKUN]: Terkoneksi dengan posisi akun: {info_det}")
        if otomatis["atas"]:
            z = otomatis["atas"]
            print(f"  * Zona Supply Atas : {z[0]:.2f} - {z[1]:.2f} (Jebol/Breakout jika close M5 > {z[2]:.2f})")
        if otomatis["bawah"]:
            z = otomatis["bawah"]
            print(f"  * Zona Demand Bawah: {z[0]:.2f} - {z[1]:.2f} (Jebol/Breakout jika close M5 < {z[2]:.2f})")
        if not any(otomatis.values()):
            print("  * Memantau pembentukan CHoCH / Liquidity Sweep baru di M5.")

    if pos_awal:
        print(f"- Pengawal Posisi Running Aktif ({len(pos_awal)} posisi terdeteksi di {SYMBOL}):")
        for p in pos_awal:
            arah_p = "BUY" if p.type == mt5.ORDER_TYPE_BUY else "SELL"
            print(f"  * Ticket #{p.ticket} | {arah_p} {p.volume} lot @ {p.price_open:.2f} | SL: {p.sl if p.sl > 0 else 'BELUM DISET'} | TP: {p.tp if p.tp > 0 else 'BELUM DISET'} | Floating: ${p.profit:+.2f}")
    else:
        print("- Pengawal Posisi Running Siaga: Otomatis mengawal begitu Anda klik BUY/SELL di MT5.")

    if orders_awal:
        print(f"- Pengawal Validitas Zona Pending Order Aktif ({len(orders_awal)} limit/stop order terdeteksi):")
        for o in orders_awal:
            nm_o, arah_o = _info_tipe_pending(o.type)
            vol_o = getattr(o, "volume_current", getattr(o, "volume_initial", 0.0))
            jrk_o = abs(harga_awal_pantau - o.price_open) if harga_awal_pantau > 0 else 0.0
            print(f"  * Ticket #{o.ticket} | {nm_o} {vol_o} lot @ {o.price_open:.2f} (Jarak: {jrk_o:.2f} pt) | SL: {o.sl if o.sl > 0 else 'BELUM DISET'} | TP: {o.tp if o.tp > 0 else 'BELUM DISET'}")
    else:
        print("- Pengawal Validitas Pending Order Siaga: Otomatis mengawasi validitas zona begitu Anda pasang Buy/Sell Limit di MT5.")
    print("Tekan Ctrl+C kapan saja untuk kembali ke mode Tanya Jawab.")
    print("=" * 70)

    last_event_time = None
    last_sweep_time = None
    last_cisd_time = None
    last_dadakan_ts = 0.0
    last_candle_check = 0.0
    _mm_acc = _mm_sym = None
    _mm_ts = 0.0
    _tick_kosong = 0
    _err_beruntun = 0
    df_m5 = None
    close_m5 = 0.0
    atr_m5 = 1.0
    evs, sw, sweeps, cisd_m5, pola_m5, ev_terakhir, calon_cisd_m5 = [], [], [], [], [], "-", "-"
    info_inst_cache = ""
    sudah_notif_masuk = set()
    riwayat_sl_terkena = []
    tiket_terpantau = {
        p.ticket: {
            "arah": "BUY" if p.type == mt5.ORDER_TYPE_BUY else "SELL",
            "volume": p.volume,
            "open": p.price_open,
            "sl": p.sl,
            "profit": p.profit,
            "poin": 0.0,
        }
        for p in pos_awal
    }
    pending_terpantau = {
        o.ticket: {
            "nama": _info_tipe_pending(o.type)[0],
            "arah": _info_tipe_pending(o.type)[1],
            "volume": getattr(o, "volume_current", getattr(o, "volume_initial", 0.0)),
            "open": o.price_open,
            "sl": o.sl,
            "tp": o.tp,
            "min_jarak": abs(harga_awal_pantau - o.price_open) if harga_awal_pantau > 0 else 9999.0,
        }
        for o in orders_awal
    }

    try:
        while True:
            try:
                tick = mt5.symbol_info_tick(SYMBOL)
                if tick is None:
                    _tick_kosong += 1
                    if _tick_kosong in (1, 30) or _tick_kosong % 300 == 0:
                        ti_ = mt5.terminal_info()
                        ket_ = ' (terminal tidak terhubung ke broker)' if (ti_ is not None and not ti_.connected) else ''
                        print(f'\n[⚠️ Tidak ada tick dari MT5{ket_} — menunggu...]')
                    time.sleep(1.0)
                    continue
                _tick_kosong = 0
                now_s = time.time()
                # Tarik data candle M5 setiap 1 detik, sedangkan harga Bid di-update LIVE tanpa jeda (tick-by-tick)
                if df_m5 is None or (now_s - last_candle_check) >= 1.0:
                    df_baru = ambil_data(mt5.TIMEFRAME_M5, 120, percobaan=1)
                    # Struktur/CISD/sweep hanya dihitung ulang saat candle M5 BARU close (data close tidak berubah antar detik)
                    if df_baru is not None and (df_m5 is None or df_baru.index[-1] != df_m5.index[-1]):
                        df_m5 = df_baru
                        close_m5 = df_m5["Close"].iloc[-1]
                        df_m5_ind = tambah_indikator(df_m5.copy())
                        atr_m5 = float(df_m5_ind["ATR14"].iloc[-1])
                        evs, _, sw = struktur(df_m5_ind)
                        sweeps = liquidity_sweeps(df_m5_ind, sw, SWING_N, lookback=3)
                        cisd_m5, calon_cisd_m5 = deteksi_cisd(df_m5_ind, sw, lookback=15)
                        pola_m5 = deteksi_pola(df_m5)[-2:]
                        info_inst_cache = deteksi_institusi_dan_ranging(df_m5_ind, df_m5_ind["ATR14"].values)
                        invalidasi_cache_bias()  # D1/H4 mungkin baru close — hitung ulang bias HTF
                        ev_terakhir = f"{evs[-1]['tipe']} {evs[-1]['arah']} @ {evs[-1]['level']:.2f} ({wkt(df_m5.index[evs[-1]['k']])})" if evs else "-"
                        ev_bar = f"{evs[-1]['tipe']} {evs[-1]['arah']} {evs[-1]['level']:.1f}" if evs else "-"
                    last_candle_check = now_s

                if tick and df_m5 is not None:
                    harga = tick.bid
                    pos_live = mt5.positions_get(symbol=SYMBOL) or ()
                    orders_live = mt5.orders_get(symbol=SYMBOL) or ()
                    info_floating_bar = ""
                    if pos_live:
                        tot_profit = sum(p.profit for p in pos_live)
                        p0 = pos_live[0]
                        arah0 = "BUY" if p0.type == mt5.ORDER_TYPE_BUY else "SELL"
                        poin0 = (tick.bid - p0.price_open) if arah0 == "BUY" else (p0.price_open - tick.ask)
                        label_pos = f"{arah0} {poin0:+.1f}pt" if len(pos_live) == 1 else f"{len(pos_live)}Pos"
                        info_floating_bar += f" | {label_pos}(${tot_profit:+.1f})"
                    if orders_live:
                        o0 = min(orders_live, key=lambda x: abs(harga - x.price_open))
                        nm0, _ = _info_tipe_pending(o0.type)
                        jrk0 = abs(harga - o0.price_open)
                        label_ord = f"{nm0}@{o0.price_open:.1f}({jrk0:.1f}pt)" if len(orders_live) == 1 else f"{len(orders_live)}Lim({jrk0:.1f}pt)"
                        info_floating_bar += f" | {label_ord}"
                    if otomatis["sinyal"]:
                        sg = otomatis["sinyal"]
                        _, kode_ambil_bar, _ = cek_status_eksekusi_sinyal(sg, pos_live, orders_live)
                        kat_r_bar, _ = tentukan_klasifikasi_risiko_sinyal(sg)
                        tag_risk_bar = "[HIGH]" if kat_r_bar == "HIGH RISK" else ""
                        tag_ambil_bar = "[DIAMBIL]" if kode_ambil_bar == "DIAMBIL" else ("[LIMIT]" if kode_ambil_bar == "LIMIT" else "[TDK DIAMBIL]")
                        info_floating_bar += f" | Sig:{sg['arah']} {sg['lo']:.1f}-{sg['hi']:.1f}{tag_risk_bar}{tag_ambil_bar}"
                    else:
                        # Tampilkan zona watchlist jika ada (menunggu harga menjemput)
                        wl_info = []
                        if otomatis.get("bawah"):
                            lb, hb, _ = otomatis["bawah"]
                            wl_info.append(f"BUY@{lb:.1f}-{hb:.1f}")
                        if otomatis.get("atas"):
                            la, ha, _ = otomatis["atas"]
                            wl_info.append(f"SEL@{la:.1f}-{ha:.1f}")
                        if wl_info:
                            info_floating_bar += f" | Sig:WAIT(Watch:{'/'.join(wl_info)})"
                        else:
                            info_floating_bar += " | Sig:WAIT"

                    cols = shutil.get_terminal_size((100, 20)).columns
                    lebar_aman = max(40, cols - 1)
                    _bias_bar = bias_htf_cached()
                    bias_lbl = f"{'BUL' if _bias_bar['bias'] == 'BULLISH' else ('BEA' if _bias_bar['bias'] == 'BEARISH' else 'NET')}({_bias_bar['skor']:+d})"
                    teks_bar = f"[{datetime.now():%H:%M:%S}] Bid:{harga:.2f} | C_M5:{close_m5:.2f} | HTF:{bias_lbl} | M5:{ev_bar}{info_floating_bar}"
                    baris_bersih = teks_bar[:lebar_aman].ljust(lebar_aman)
                    print(f"\r{baris_bersih}", end="", flush=True)

                    # 0A. PENGAWAL VALIDITAS ZONA PENDING ORDER (BUY LIMIT / SELL LIMIT / STOP)
                    pending_aktif_skrg = set()
                    o_last_c, c_last_c = df_m5["Open"].iloc[-1], df_m5["Close"].iloc[-1]
                    t_candle_last_c = df_m5.index[-1]
                    bodi_last_c = abs(c_last_c - o_last_c)
                    if (now_s - _mm_ts) >= 5.0:   # info akun cukup disegarkan tiap 5 detik
                        _mm_acc, _mm_sym, _mm_ts = mt5.account_info(), mt5.symbol_info(SYMBOL), now_s
                    acc_live_mm, sym_live_mm = _mm_acc, _mm_sym
                    cur_live_mm = acc_live_mm.currency if acc_live_mm else "USD"
                    eq_live_mm = float(acc_live_mm.equity if acc_live_mm and acc_live_mm.equity > 0 else (acc_live_mm.balance if acc_live_mm else 1000.0))
                    contract_live_mm = _nilai_per_poin(harga, True, float(sym_live_mm.trade_contract_size if sym_live_mm else 100.0))

                    for o in orders_live:
                        tkt_o = o.ticket
                        pending_aktif_skrg.add(tkt_o)
                        nm_o, arah_o = _info_tipe_pending(o.type)
                        vol_o = getattr(o, "volume_current", getattr(o, "volume_initial", 0.0))
                        p_limit = o.price_open
                        jarak_skrg = abs(harga - p_limit)

                        if tkt_o not in pending_terpantau:
                            print(f"\n\n[PENDING ORDER BARU TERDETEKSI #{tkt_o}!] {nm_o} {vol_o} lot @ {p_limit:.2f} (Jarak: {jarak_skrg:.2f} pt) | SL: {o.sl if o.sl > 0 else 'BELUM DISET'} | TP: {o.tp if o.tp > 0 else 'BELUM DISET'}")
                            print("   -> Pengawal Validitas Zona Limit aktif! Sistem akan memberi tahu jika zona limit ini tetap VALID atau berubah BAHAYA/TIDAK VALID.")
                            for _ in range(2):
                                bunyi(1350, 180)
                            pending_terpantau[tkt_o] = {
                                "nama": nm_o,
                                "arah": arah_o,
                                "volume": vol_o,
                                "open": p_limit,
                                "sl": o.sl,
                                "tp": o.tp,
                                "min_jarak": jarak_skrg,
                            }
                        else:
                            pending_terpantau[tkt_o]["sl"] = o.sl
                            pending_terpantau[tkt_o]["tp"] = o.tp
                            if jarak_skrg < pending_terpantau[tkt_o]["min_jarak"]:
                                pending_terpantau[tkt_o]["min_jarak"] = jarak_skrg

                        min_jrk_hist = pending_terpantau[tkt_o]["min_jarak"]

                        # Evaluasi Money Management pada Pending Order
                        if vol_o > 0 and o.sl > 0:
                            jrk_sl_o = abs(p_limit - o.sl)
                            potensi_loss_o = jrk_sl_o * contract_live_mm * vol_o
                            pct_risk_o = (potensi_loss_o / max(eq_live_mm, 1.0)) * 100.0
                            if pct_risk_o > 2.5 and f"ord_overlot_{tkt_o}" not in sudah_notif_masuk:
                                sudah_notif_masuk.add(f"ord_overlot_{tkt_o}")
                                print(f"\n\n[🚨 PERINGATAN MM PENDING ORDER #{tkt_o}!] {nm_o} {vol_o} lot memiliki risiko loss -${potensi_loss_o:.2f} ({pct_risk_o:.1f}% modal) jika terjemput!")
                                print("   -> TINDAKAN MM: Disarankan turunkan volume lot sebelum order terjemput agar tidak melanggar batas risiko maksimal 1%-2% modal.")
                                for _ in range(3):
                                    bunyi(520, 300)

                        # (1) Peringatan jika Pending Order belum dipasang SL
                        if o.sl == 0 and f"ord_nosl_{tkt_o}" not in sudah_notif_masuk:
                            sudah_notif_masuk.add(f"ord_nosl_{tkt_o}")
                            jarak_sl_aman = max(3.0, round(1.5 * atr_m5, 2))
                            sl_saran = (p_limit - jarak_sl_aman) if arah_o == "BUY" else (p_limit + jarak_sl_aman)
                            print(f"\n\n[PERINGATAN PENDING TANPA SL #{tkt_o}!] {nm_o} @ {p_limit:.2f} BELUM DIPASANG STOP LOSS!")
                            print(f"   -> SARAN: Segera set SL pada {nm_o} Anda di sekitar {sl_saran:.2f} (~{jarak_sl_aman:.2f} pt) agar jika terjemput tidak terbuka tanpa pengaman.")
                            for _ in range(2):
                                bunyi(850, 250)

                        # (2) Evaluasi saat harga mendekati Zona Limit Order (<= max(5.0, 2.5x ATR M5))
                        radius_evaluasi = max(5.0, round(2.5 * atr_m5, 2))
                        if jarak_skrg <= radius_evaluasi and "LIMIT" in nm_o:
                            # Cek apakah harga menghantam ke arah limit dengan Candle Impuls Kuat / Marubozu Lawan Arah
                            impuls_menghantam_limit = (
                                (arah_o == "BUY" and c_last_c < o_last_c and bodi_last_c >= 1.35 * atr_m5)
                                or (arah_o == "SELL" and c_last_c > o_last_c and bodi_last_c >= 1.35 * atr_m5)
                            )
                            key_bahaya_lim = f"ord_danger_{tkt_o}_{t_candle_last_c}"
                            if impuls_menghantam_limit and key_bahaya_lim not in sudah_notif_masuk:
                                sudah_notif_masuk.add(key_bahaya_lim)
                                print(f"\n\n[🚨 PERINGATAN ZONA {nm_o} TIDAK VALID / RAWAN JEBOL #{tkt_o}!]")
                                print(f"   Harga mendekati {nm_o} @ {p_limit:.2f} (sisa jarak {jarak_skrg:.2f} pt), TETAPI dihantam Candle Impuls/Marubozu M5 kuat (Body {bodi_last_c:.2f} pt)!")
                                print(f"   -> STATUS ZONA: TIDAK VALID UNTUK LIMIT PASIF (Risiko 'Pisau Jatuh' / langsung bablas menembus SL!).")
                                print(f"   -> TINDAKAN: Segera HAPUS / CANCEL {nm_o} #{tkt_o} sekarang! Tunggu di luar sampai muncul Sweep + CHoCH/CISD M5 baru entry manual!")
                                for _ in range(5):
                                    bunyi(620, 320)
                            elif jarak_skrg <= max(2.5, round(1.2 * atr_m5, 2)) and not impuls_menghantam_limit and f"ord_valid_{tkt_o}" not in sudah_notif_masuk:
                                sudah_notif_masuk.add(f"ord_valid_{tkt_o}")
                                print(f"\n\n[✅ SIAGA {nm_o} — ZONA MASIH VALID #{tkt_o}!]")
                                print(f"   Harga mendekati {nm_o} @ {p_limit:.2f} (jarak {jarak_skrg:.2f} pt) dengan ritme pullback/koreksi yang terkendali (Body M5 {bodi_last_c:.2f} pt, tidak ada impuls penghancur).")
                                print(f"   -> STATUS ZONA: MASIH VALID & AMAN. Order siap terjemput, pastikan SL ({o.sl if o.sl > 0 else 'BELUM DISET!'}) sudah terpasang!")
                                for _ in range(2):
                                    bunyi(1300, 220)

                        # (3) Deteksi Zona Limit Sudah Terpakai / Front-Run (Harga nyaris sentuh <= 1.2 pt lalu terbang jauh >= 6.0 pt)
                        batas_frontrun = max(1.2, round(0.6 * atr_m5, 2))
                        batas_terbang = max(6.0, round(3.0 * atr_m5, 2))
                        arah_terbang_benar = (arah_o == "BUY" and harga > p_limit) or (arah_o == "SELL" and harga < p_limit)
                        if "LIMIT" in nm_o and min_jrk_hist <= batas_frontrun and jarak_skrg >= batas_terbang and arah_terbang_benar and f"ord_mit_{tkt_o}" not in sudah_notif_masuk:
                            sudah_notif_masuk.add(f"ord_mit_{tkt_o}")
                            print(f"\n\n[⚠️ PERINGATAN ZONA {nm_o} SUDAH TERPAKAI (FRONT-RUN / MITIGATED) #{tkt_o}!]")
                            print(f"   Harga tadi sudah nyaris menyentuh {nm_o} @ {p_limit:.2f} (jarak terdekat {min_jrk_hist:.2f} pt) lalu memantul terbang sejauh {jarak_skrg:.2f} pt tanpa menjemput order Anda!")
                            print(f"   -> STATUS ZONA: SUDAH LEMAH / MITIGATED (Jika harga balik lagi ke {p_limit:.2f} untuk kedua kalinya, zona ini sudah tidak fresh dan rawan jebol!).")
                            print(f"   -> TINDAKAN: Disarankan HAPUS / CANCEL {nm_o} #{tkt_o} agar tidak terjebak sapuan likuiditas saat harga kembali.")
                            for _ in range(4):
                                bunyi(780, 280)

                    # Cek jika ada Pending Order yang baru saja hilang (karena TEREKSEKUSI jadi posisi running atau DIHAPUS manual)
                    tiket_pos_skrg = {p.ticket for p in pos_live}
                    for tkt_o_lama in list(pending_terpantau.keys()):
                        if tkt_o_lama not in pending_aktif_skrg:
                            inf_o = pending_terpantau.pop(tkt_o_lama)
                            # Cek apakah tiketnya berubah menjadi posisi running atau ada posisi baru di harga yang sama
                            terjemput = (tkt_o_lama in tiket_pos_skrg) or any(
                                abs(p.price_open - inf_o["open"]) <= 1.5 and ("BUY" if p.type == mt5.ORDER_TYPE_BUY else "SELL") == inf_o["arah"]
                                for p in pos_live
                            )
                            if terjemput:
                                print(f"\n\n[🎯 PENDING ORDER TEREKSEKUSI #{tkt_o_lama}!] {inf_o['nama']} {inf_o['volume']} lot @ {inf_o['open']:.2f} telah TERJEMPUT dan aktif menjadi Posisi Running!")
                                print("   -> Pengawal Posisi Otomatis langsung mengambil alih untuk mengawal BEP, Partial TP, dan keamanan SL Anda!")
                                for _ in range(3):
                                    bunyi(1500, 200)
                            else:
                                print(f"\n\n[ℹ️ PENDING ORDER DIBATALKAN/DIHAPUS #{tkt_o_lama}] {inf_o['nama']} {inf_o['volume']} lot @ {inf_o['open']:.2f} telah dihapus dari MT5.")
                                bunyi(1050, 200)

                    # 0B. PENGAWAL POSISI RUNNING OTOMATIS (LIVE TRADE GUARD)
                    tiket_aktif_skrg = set()
                    for p in pos_live:
                        tkt = p.ticket
                        tiket_aktif_skrg.add(tkt)
                        arah_p = "BUY" if p.type == mt5.ORDER_TYPE_BUY else "SELL"
                        poin_p = (tick.bid - p.price_open) if arah_p == "BUY" else (p.price_open - tick.ask)

                        # Deteksi jika trader baru saja membuka posisi baru saat mode pantau sedang jalan
                        if tkt not in tiket_terpantau:
                            print(f"\n\n[POSISI BARU TERDETEKSI #{tkt}!] {arah_p} {p.volume} lot @ {p.price_open:.2f} | SL: {p.sl if p.sl > 0 else 'BELUM DISET'} | TP: {p.tp if p.tp > 0 else 'BELUM DISET'}")
                            print("   -> Pengawal Posisi Otomatis langsung aktif mengawasi posisi ini!")
                            for _ in range(2):
                                bunyi(1300, 200)

                        tiket_terpantau[tkt] = {
                            "arah": arah_p,
                            "volume": p.volume,
                            "open": p.price_open,
                            "sl": p.sl,
                            "profit": p.profit,
                            "poin": poin_p,
                        }

                        # (a) Peringatan jika posisi belum dipasang SL
                        if p.sl == 0 and f"nosl_{tkt}" not in sudah_notif_masuk:
                            sudah_notif_masuk.add(f"nosl_{tkt}")
                            jarak_sl_aman = max(3.0, round(1.5 * atr_m5, 2))
                            sl_saran = (p.price_open - jarak_sl_aman) if arah_p == "BUY" else (p.price_open + jarak_sl_aman)
                            print(f"\n\n[PERINGATAN RISIKO #{tkt}!] Posisi {arah_p} {p.volume} lot @ {p.price_open:.2f} BELUM DIPASANG STOP LOSS!")
                            print(f"   -> SARAN: Segera pasang SL maksimal di sekitar {sl_saran:.2f} (~{jarak_sl_aman:.2f} poin) demi keamanan akun real Anda.")
                            for _ in range(2):
                                bunyi(800, 250)

                        # (b) Peringatan Geser SL ke BEP (Impas) saat sudah profit >= 2.5 poin (atau 1.2x ATR M5)
                        batas_bep = max(2.5, round(1.2 * atr_m5, 2))
                        sl_belum_bep = (arah_p == "BUY" and p.sl < p.price_open) or (arah_p == "SELL" and (p.sl == 0 or p.sl > p.price_open))
                        if poin_p >= batas_bep and sl_belum_bep and f"bep_{tkt}" not in sudah_notif_masuk:
                            sudah_notif_masuk.add(f"bep_{tkt}")
                            bep_plus = (p.price_open + 0.3) if arah_p == "BUY" else (p.price_open - 0.3)
                            print(f"\n\n[ALERT AMANKAN MODAL / BEP #{tkt}!] Posisi {arah_p} @ {p.price_open:.2f} sudah running profit +{poin_p:.2f} poin (${p.profit:+.2f})!")
                            print(f"   -> TINDAKAN: Segera geser Stop Loss ke BEP+ ({bep_plus:.2f}) agar transaksi ini 100% BEBAS RISIKO (Risk-Free)!")
                            for _ in range(3):
                                bunyi(1450, 220)

                        # (c) Peringatan Partial TP / Kunci Profit Besar saat profit >= 5.0 poin (atau 2.5x ATR M5)
                        batas_tp_parsial = max(5.0, round(2.5 * atr_m5, 2))
                        if poin_p >= batas_tp_parsial and f"lock_{tkt}" not in sudah_notif_masuk:
                            sudah_notif_masuk.add(f"lock_{tkt}")
                            kunci_sl = (p.price_open + round(poin_p * 0.5, 2)) if arah_p == "BUY" else (p.price_open - round(poin_p * 0.5, 2))
                            print(f"\n\n[ALERT KUNCI PROFIT BESAR #{tkt}!] Posisi {arah_p} @ {p.price_open:.2f} sudah profit +{poin_p:.2f} poin (${p.profit:+.2f})!")
                            print(f"   -> TINDAKAN: Amankan sebagian profit (Partial Close) atau geser Trailing SL ke {kunci_sl:.2f} untuk mengunci profit!")
                            for _ in range(4):
                                bunyi(1600, 200)

                        # (d) Peringatan Bahaya: Candle Impuls M5 Kuat Melawan Arah Posisi
                        o_last, c_last = df_m5["Open"].iloc[-1], df_m5["Close"].iloc[-1]
                        t_candle_last = df_m5.index[-1]
                        bodi_last = abs(c_last - o_last)
                        melawan_impuls = (arah_p == "BUY" and c_last < o_last and bodi_last >= 1.2 * atr_m5) or (
                            arah_p == "SELL" and c_last > o_last and bodi_last >= 1.2 * atr_m5
                        )
                        key_impuls = f"imp_{tkt}_{t_candle_last}"
                        if melawan_impuls and key_impuls not in sudah_notif_masuk:
                            sudah_notif_masuk.add(key_impuls)
                            saran_imp = "Segera amankan profit / pasang BEP!" if poin_p > 0 else "Waspada tekanan lawan arah, perketat pengawasan SL / siap Cut terukur jika struktur jebol!"
                            print(f"\n\n[PERINGATAN TEKANAN LAWAN ARAH #{tkt}!] Posisi Anda {arah_p} (Floating: {poin_p:+.2f} pt / ${p.profit:+.2f}),")
                            print(f"   tetapi candle M5 ({wkt(t_candle_last)}) membentuk impuls kuat melawan posisi Anda (Body {bodi_last:.2f} pt)!")
                            print(f"   -> TINDAKAN: {saran_imp}")
                            for _ in range(3):
                                bunyi(750, 300)

                        # (e) Peringatan Kritis Mendekati SL atau Floating Minus Besar Tanpa SL
                        if p.sl > 0:
                            jarak_ke_sl = (tick.bid - p.sl) if arah_p == "BUY" else (p.sl - tick.ask)
                            if 0 < jarak_ke_sl <= 1.2 and f"nearsl_{tkt}" not in sudah_notif_masuk:
                                sudah_notif_masuk.add(f"nearsl_{tkt}")
                                print(f"\n\n[AWAS MENDEKATI SL #{tkt}!] Harga saat ini tinggal {jarak_ke_sl:.2f} poin dari SL Anda ({p.sl:.2f})!")
                                print("   -> DISIPLIN: Jangan menggeser/melebarkan SL agar batas risiko akun real Anda tetap terjaga.")
                                for _ in range(3):
                                    bunyi(650, 300)
                        else:
                            batas_minus_darurat = -max(3.5, round(1.8 * atr_m5, 2))
                            if poin_p <= batas_minus_darurat and f"dd_{tkt}" not in sudah_notif_masuk:
                                sudah_notif_masuk.add(f"dd_{tkt}")
                                print(f"\n\n[BAHAYA FLOATING MINUS TANPA SL #{tkt}!] Posisi {arah_p} @ {p.price_open:.2f} sudah minus {poin_p:.2f} poin (${p.profit:+.2f}) tanpa SL!")
                                print("   -> TINDAKAN DARURAT: Segera pasang batas Stop Loss atau lakukan Cut Loss jika zona pertahanan sudah jebol!")
                                for _ in range(5):
                                    bunyi(600, 350)

                        # (f) Peringatan Money Management: Over-Leverage / Lot Kebesaran
                        if p.sl > 0:
                            jarak_sl_real = abs(p.price_open - p.sl)
                            potensi_loss_usd = jarak_sl_real * contract_live_mm * p.volume
                            pct_risk_real = (potensi_loss_usd / max(eq_live_mm, 1.0)) * 100.0
                            if pct_risk_real > 2.5 and f"mm_overlot_{tkt}" not in sudah_notif_masuk:
                                sudah_notif_masuk.add(f"mm_overlot_{tkt}")
                                print(f"\n\n[🚨 PERINGATAN MONEY MANAGEMENT (OVER-LEVERAGE) #{tkt}!]")
                                print(f"   Posisi {arah_p} {p.volume} lot @ {p.price_open:.2f} berisiko kerugian -${potensi_loss_usd:.2f} ({pct_risk_real:.1f}% dari modal ${eq_live_mm:,.2f})!")
                                print("   -> STANDAR RISIKO SEHAT: Maksimal 0.5% - 1.0% per trade (Toleransi keras: 2.0%).")
                                print("   -> TINDAKAN MM: Segera lakukan Partial Close (kurangi volume) agar modal terlindungi dari ancaman Margin Call!")
                                for _ in range(4):
                                    bunyi(520, 320)

                    # Evaluasi Total Akumulasi Risiko Portofolio (Portfolio Exposure Guard)
                    if pos_live:
                        tot_risk_usd = 0.0
                        for p_chk in pos_live:
                            sl_chk = p_chk.sl if p_chk.sl > 0 else (p_chk.price_open - 3.5 if p_chk.type == mt5.ORDER_TYPE_BUY else p_chk.price_open + 3.5)
                            tot_risk_usd += abs(p_chk.price_open - sl_chk) * contract_live_mm * p_chk.volume
                        tot_risk_pct = (tot_risk_usd / max(eq_live_mm, 1.0)) * 100.0
                        if tot_risk_pct > 5.0 and "mm_tot_risk" not in sudah_notif_masuk:
                            sudah_notif_masuk.add("mm_tot_risk")
                            print(f"\n\n[⚠️ PORTFOLIO MM GUARD: TOTAL RISIKO TERBUKA MENCAPAI {tot_risk_pct:.1f}% MODAL!]")
                            print(f"   Akumulasi potensi kerugian dari {len(pos_live)} posisi terbuka Anda adalah -${tot_risk_usd:.2f} (Modal: ${eq_live_mm:,.2f}).")
                            print("   -> DISIPLIN KERAS: DILARANG menambah posisi baru (Averaging / Martingale)! Kunci BEP pada posisi yang sudah profit!")
                            for _ in range(3):
                                bunyi(650, 300)

                    # Deteksi jika ada posisi yang baru saja ditutup (manual / kena TP / kena SL)
                    for tkt_lama in list(tiket_terpantau.keys()):
                        if tkt_lama not in tiket_aktif_skrg:
                            info_lama = tiket_terpantau.pop(tkt_lama)
                            status_akhir = "PROFIT" if info_lama["profit"] >= 0 else "LOSS"
                            print(f"\n\n[POSISI SELESAI / DITUTUP #{tkt_lama}] Posisi {info_lama['arah']} {info_lama['volume']} lot @ {info_lama['open']:.2f} telah tertutup ({status_akhir} terakhir: ${info_lama['profit']:+.2f} / {info_lama['poin']:+.2f} pt).")
                            if status_akhir == "LOSS":
                                lvl_sl_kena = info_lama.get("sl") if info_lama.get("sl", 0) > 0 else harga
                                # Pastikan ini benar-benar SL risiko (bukan SL BEP+ yang tertutup tipis karena komisi/slippage)
                                is_sl_risiko = (
                                    (info_lama["arah"] == "BUY" and lvl_sl_kena < info_lama["open"] - 0.4)
                                    or (info_lama["arah"] == "SELL" and lvl_sl_kena > info_lama["open"] + 0.4)
                                )
                                # Cegah duplikasi jika ada beberapa tiket di arah & level SL yang sama
                                sudah_ada_sl_serupa = any(
                                    not r_ex["selesai_evaluasi"]
                                    and r_ex["arah"] == info_lama["arah"]
                                    and abs(r_ex["sl"] - lvl_sl_kena) <= 5.0
                                    for r_ex in riwayat_sl_terkena
                                )
                                if is_sl_risiko and not sudah_ada_sl_serupa:
                                    riwayat_sl_terkena.append({
                                        "ticket": tkt_lama,
                                        "arah": info_lama["arah"],
                                        "open": info_lama["open"],
                                        "sl": lvl_sl_kena,
                                        "ekstrem": harga,
                                        "selesai_evaluasi": False,
                                    })
                                    print(f"   -> [🔍 RADAR INVESTIGASI PASCA-SL AKTIF UNTUK #{tkt_lama}!]")
                                    print("      Sistem otomatis mengawasi apakah tersentuhnya SL ini adalah:")
                                    print("      (a) STOP-HUNT / LIQUIDITY SWEEP (harga berbalik ke zona dengan konfirmasi M5 -> peluang Re-Entry 2nd Chance!), atau")
                                    print("      (b) VALID BREAKOUT (jebol beneran -> jangan balas dendam / tunggu zona baru di bawah ini).")
                                zona_masih_utuh = False
                                if info_lama["arah"] == "BUY" and otomatis.get("bawah"):
                                    lb_z, hb_z, jb_z = otomatis["bawah"]
                                    if harga >= jb_z:
                                        zona_masih_utuh = True
                                        print(f"   -> [ℹ️ STATUS ZONA UTAMA]: Posisi #{tkt_lama} terkena SL individual di {lvl_sl_kena:.2f},")
                                        print(f"      TETAPI Zona Demand Utama ({lb_z:.2f} - {hb_z:.2f} | Batas SL Zona {jb_z:.2f}) dan Pending Order Anda MASIH VALID!")
                                        print("      Harga saat ini hanya koreksi di dalam zona (belum jebol batas zona). Sinyal utama TIDAK berubah.")
                                elif info_lama["arah"] == "SELL" and otomatis.get("atas"):
                                    la_z, ha_z, ja_z = otomatis["atas"]
                                    if harga <= ja_z:
                                        zona_masih_utuh = True
                                        print(f"   -> [ℹ️ STATUS ZONA UTAMA]: Posisi #{tkt_lama} terkena SL individual di {lvl_sl_kena:.2f},")
                                        print(f"      TETAPI Zona Supply Utama ({la_z:.2f} - {ha_z:.2f} | Batas SL Zona {ja_z:.2f}) dan Pending Order Anda MASIH VALID!")
                                        print("      Harga saat ini hanya retest di dalam zona (belum jebol batas zona). Sinyal utama TIDAK berubah.")

                                if not zona_masih_utuh and not sudah_ada_sl_serupa:
                                    _aktifkan_zona_valid_berikutnya(
                                        otomatis,
                                        sudah_notif_masuk,
                                        info_lama["arah"],
                                        lvl_sl_kena,
                                        harga,
                                        atr_m5,
                                        f"Posisi {info_lama['arah']} #{tkt_lama} terkena SL di {lvl_sl_kena:.2f}",
                                        df_m5=df_m5,
                                        hanya_preview=True,
                                    )
                            for _ in range(2):
                                bunyi(1100, 200)

                    # 0C. EVALUASI REAL-TIME PASCA-SL (DETEKSI STOP-HUNT RE-ENTRY vs VALID BREAKOUT)
                    low_m5_now = float(df_m5["Low"].iloc[-1])
                    high_m5_now = float(df_m5["High"].iloc[-1])
                    open_m5_now = float(df_m5["Open"].iloc[-1])
                    rng_m5_now = max(high_m5_now - low_m5_now, 1e-6)
                    wick_bawah_m5 = min(open_m5_now, close_m5) - low_m5_now
                    wick_atas_m5 = high_m5_now - max(open_m5_now, close_m5)

                    for r_sl in riwayat_sl_terkena:
                        if r_sl["selesai_evaluasi"]:
                            continue
                        tkt_r = r_sl["ticket"]
                        arah_r = r_sl["arah"]
                        sl_r = r_sl["sl"]
                        open_r = r_sl["open"]
                        lawan_inst_re = (
                            "PERINGATAN LONJAKAN MOMENTUM" in info_inst_cache
                            and ((arah_r == "BUY" and "BEARISH" in info_inst_cache.upper()) or (arah_r == "SELL" and "BULLISH" in info_inst_cache.upper()))
                        )
                        if arah_r == "BUY":
                            r_sl["ekstrem"] = min(r_sl["ekstrem"], harga, low_m5_now)
                            # Cek Skenario 1: Harga berbalik NAIK kembali melewati SL + konfirmasi candle M5 (bukan sekadar pantulan tick sesaat!)
                            batas_pulih = sl_r + max(1.2, round(0.6 * atr_m5, 2))
                            konf_candle_buy = (close_m5 > open_m5_now and close_m5 > sl_r) or (wick_bawah_m5 / rng_m5_now >= 0.38 and close_m5 > sl_r)
                            if harga >= batas_pulih and konf_candle_buy and not lawan_inst_re:
                                for r_oth in riwayat_sl_terkena:
                                    if r_oth["arah"] == "BUY" and abs(r_oth["sl"] - sl_r) <= 6.0:
                                        r_oth["selesai_evaluasi"] = True
                                sl_reentry = r_sl["ekstrem"] - max(2.2, round(1.15 * atr_m5, 2))
                                risiko_re = max(2.5, harga - sl_reentry)
                                sinyal_re = dict(
                                    arah="BUY",
                                    lo=round(min(sl_r, harga - 0.5), 2),
                                    hi=round(max(open_r, harga + 0.8), 2),
                                    sl=round(sl_reentry, 2),
                                    tp1=round(harga + 1.8 * risiko_re, 2),
                                    tp2=round(harga + 3.0 * risiko_re, 2),
                                    ket="Re-Entry 2nd Chance Pasca Stop-Hunt",
                                )
                                print(f"\n\n[🔥 ALERT STOP-HUNT TERDETEKSI — PELUANG RE-ENTRY (2ND CHANCE) #{tkt_r}!]")
                                print(f"   Posisi BUY Anda (@ {open_r:.2f}) tadi terkena SL di {sl_r:.2f} (Titik jarum terendah: {r_sl['ekstrem']:.2f}),")
                                print(f"   TETAPI harga tidak lanjut jatuh dan candle M5 ditutup pulih ke {close_m5:.2f} (Harga: {harga:.2f})!")
                                print(f"   -> DIAGNOSA: Penurunan ke SL tadi adalah LIQUIDITY SWEEP / STOP-HUNT institusi untuk mengambil likuiditas!")
                                print(f"   -> PANDUAN RE-ENTRY AMAN (SECOND CHANCE): SL baru telah diberi buffer aman di bawah ujung jarum sweep ({sl_reentry:.2f})!")
                                _update_sinyal_aktif_pantau(otomatis, sudah_notif_masuk, sinyal_re, f"Stop-Hunt Recovery Posisi BUY #{tkt_r} (SL digeser ke bawah wick sweep {r_sl['ekstrem']:.2f})")
                                for _ in range(4):
                                    bunyi(1450, 230)
                            # Cek Skenario 2: Candle M5 resmi close jebol jauh di bawah SL (Valid Breakout!)
                            elif close_m5 < sl_r - max(1.8, round(0.9 * atr_m5, 2)) and harga < close_m5:
                                for r_oth in riwayat_sl_terkena:
                                    if r_oth["arah"] == "BUY" and abs(r_oth["sl"] - sl_r) <= 6.0:
                                        r_oth["selesai_evaluasi"] = True
                                print(f"\n\n[⛔ KONFIRMASI VALID BREAKOUT PASCA-SL #{tkt_r} — PINDAH KE ZONA VALID SELANJUTNYA!]")
                                print(f"   Setelah menyentuh SL ({sl_r:.2f}), candle M5 resmi CLOSE jebol di {close_m5:.2f} (Harga sekarang: {harga:.2f}).")
                                print("   -> DIAGNOSA: Zona lama benar-benar JEBOL (Valid Breakout), BUKAN sekadar Stop-Hunt!")
                                print("   -> TINDAKAN: JANGAN balas dendam di tempat yang sama! Sistem langsung mengarahkan ke Zona Demand Valid Selanjutnya:")
                                _aktifkan_zona_valid_berikutnya(
                                    otomatis,
                                    sudah_notif_masuk,
                                    "BUY",
                                    min(sl_r, close_m5),
                                    harga,
                                    atr_m5,
                                    f"Valid Breakout Turun menembus SL BUY #{tkt_r} ({sl_r:.2f})",
                                    df_m5=df_m5,
                                    hanya_preview=False,
                                )
                                for _ in range(3):
                                    bunyi(600, 320)
                        else:
                            r_sl["ekstrem"] = max(r_sl["ekstrem"], harga, high_m5_now)
                            batas_pulih = sl_r - max(1.2, round(0.6 * atr_m5, 2))
                            konf_candle_sell = (close_m5 < open_m5_now and close_m5 < sl_r) or (wick_atas_m5 / rng_m5_now >= 0.38 and close_m5 < sl_r)
                            if harga <= batas_pulih and konf_candle_sell and not lawan_inst_re:
                                for r_oth in riwayat_sl_terkena:
                                    if r_oth["arah"] == "SELL" and abs(r_oth["sl"] - sl_r) <= 6.0:
                                        r_oth["selesai_evaluasi"] = True
                                sl_reentry = r_sl["ekstrem"] + max(2.2, round(1.15 * atr_m5, 2))
                                risiko_re = max(2.5, sl_reentry - harga)
                                sinyal_re = dict(
                                    arah="SELL",
                                    lo=round(min(open_r, harga - 0.8), 2),
                                    hi=round(max(sl_r, harga + 0.5), 2),
                                    sl=round(sl_reentry, 2),
                                    tp1=round(harga - 1.8 * risiko_re, 2),
                                    tp2=round(harga - 3.0 * risiko_re, 2),
                                    ket="Re-Entry 2nd Chance Pasca Stop-Hunt",
                                )
                                print(f"\n\n[🔥 ALERT STOP-HUNT TERDETEKSI — PELUANG RE-ENTRY (2ND CHANCE) #{tkt_r}!]")
                                print(f"   Posisi SELL Anda (@ {open_r:.2f}) tadi terkena SL di {sl_r:.2f} (Titik jarum tertinggi: {r_sl['ekstrem']:.2f}),")
                                print(f"   TETAPI harga tidak lanjut naik dan candle M5 ditutup pulih ke {close_m5:.2f} (Harga: {harga:.2f})!")
                                print(f"   -> DIAGNOSA: Kenaikan ke SL tadi adalah LIQUIDITY SWEEP / STOP-HUNT institusi untuk mengambil likuiditas!")
                                print(f"   -> PANDUAN RE-ENTRY AMAN (SECOND CHANCE): SL baru telah diberi buffer aman di atas ujung jarum sweep ({sl_reentry:.2f})!")
                                _update_sinyal_aktif_pantau(otomatis, sudah_notif_masuk, sinyal_re, f"Stop-Hunt Recovery Posisi SELL #{tkt_r} (SL digeser ke atas wick sweep {r_sl['ekstrem']:.2f})")
                                for _ in range(4):
                                    bunyi(1450, 230)
                            elif close_m5 > sl_r + max(1.8, round(0.9 * atr_m5, 2)) and harga > close_m5:
                                for r_oth in riwayat_sl_terkena:
                                    if r_oth["arah"] == "SELL" and abs(r_oth["sl"] - sl_r) <= 6.0:
                                        r_oth["selesai_evaluasi"] = True
                                print(f"\n\n[⛔ KONFIRMASI VALID BREAKOUT PASCA-SL #{tkt_r} — PINDAH KE ZONA VALID SELANJUTNYA!]")
                                print(f"   Setelah menyentuh SL ({sl_r:.2f}), candle M5 resmi CLOSE jebol di {close_m5:.2f} (Harga sekarang: {harga:.2f}).")
                                print("   -> DIAGNOSA: Zona lama benar-benar JEBOL (Valid Breakout), BUKAN sekadar Stop-Hunt!")
                                print("   -> TINDAKAN: JANGAN balas dendam di tempat yang sama! Sistem langsung mengarahkan ke Zona Supply Valid Selanjutnya:")
                                _aktifkan_zona_valid_berikutnya(
                                    otomatis,
                                    sudah_notif_masuk,
                                    "SELL",
                                    max(sl_r, close_m5),
                                    harga,
                                    atr_m5,
                                    f"Valid Breakout Naik menembus SL SELL #{tkt_r} ({sl_r:.2f})",
                                    df_m5=df_m5,
                                    hanya_preview=False,
                                )
                                for _ in range(3):
                                    bunyi(600, 320)

                    # 1. Cek target manual jika user mengetik angka
                    if lo_manual is not None:
                        if lo_manual <= harga <= hi_manual and "manual" not in sudah_notif_masuk:
                            sudah_notif_masuk.add("manual")
                            pola_info = ", ".join(pola_m5) if pola_m5 else "belum ada pola pinbar/engulfing"
                            print(f"\n\n[ALERT MASUK ZONA!] Harga {harga:.2f} menyentuh zona ({lo_manual:.2f} - {hi_manual:.2f})!")
                            print(f"   -> Status M5 saat ini: {ev_terakhir} | Pola candle: {pola_info}")
                            print("   -> Pantau terus: menunggu konfirmasi pantulan atau deteksi jika zona jebol...")
                            for _ in range(3):
                                bunyi(1200, 250)

                        # Cek jika zona manual JEBOL (candle M5 close menembus batas)
                        if close_m5 > batas_atas or close_m5 < batas_bawah:
                            arah_jebol = "ATAS (Breakout Naik)" if close_m5 > batas_atas else "BAWAH (Breakout Turun)"
                            arah_next_m = "SELL" if close_m5 > batas_atas else "BUY"
                            batas_next_m = batas_atas if close_m5 > batas_atas else batas_bawah
                            print(f"\n\n[PERINGATAN: ZONA MANUAL DITEMBUS / JEBOL!] Candle M5 close di {close_m5:.2f} menembus batas {arah_jebol}!")
                            print("   -> Skenario pantulan di zona ini BATAL (Invalidated). Mencari zona valid selanjutnya...")
                            _aktifkan_zona_valid_berikutnya(
                                otomatis,
                                sudah_notif_masuk,
                                arah_next_m,
                                batas_next_m,
                                harga,
                                atr_m5,
                                f"Zona Manual {lo_manual:.2f}-{hi_manual:.2f} Jebol ({arah_jebol})",
                                df_m5=df_m5,
                                hanya_preview=False,
                            )
                            for _ in range(5):
                                bunyi(600, 350)
                            return bedah_forensik_jebol(df_m5, f"Zona Manual {lo_manual:.2f}-{hi_manual:.2f} ({arah_jebol})", close_m5)

                    # 2. Cek Sinyal Aktif dari AI (Entry, SL/Invalidasi, Stop-Hunt Re-Entry, TP1, TP2)
                    if otomatis["sinyal"]:
                        s = otomatis["sinyal"]
                        lbl_ambil_now, kode_ambil_now, ringkas_ambil = cek_status_eksekusi_sinyal(s, pos_live, orders_live)
                        if s.get("t_buat") and kode_ambil_now in ("DIAMBIL", "LIMIT"):
                            st_jurnal_target = "YA" if kode_ambil_now == "DIAMBIL" else "LIMIT"
                            if kode_ambil_now == "DIAMBIL":
                                m_tkt_p = re.search(r"#(\d+)", str(ringkas_ambil))
                                if m_tkt_p:
                                    st_jurnal_target = f"YA (#{m_tkt_p.group(1)})"
                            if s.get("_status_jurnal_ambil") != st_jurnal_target:
                                tandai_sinyal_diambil_jurnal(s["t_buat"], st_jurnal_target, mode_by_id=True)
                                s["_status_jurnal_ambil"] = st_jurnal_target
                        sudah_tp1_aktif = bool(s.get("sudah_tp1") or "TP1 Tercapai" in str(s.get("ket", "")))

                        if s["lo"] <= harga <= s["hi"]:
                            if s["arah"] == "BUY":
                                s["_ekstrem_dalam_zona"] = min(float(s.get("_ekstrem_dalam_zona", harga)), harga, low_m5_now)
                            else:
                                s["_ekstrem_dalam_zona"] = max(float(s.get("_ekstrem_dalam_zona", harga)), harga, high_m5_now)

                        if s["lo"] <= harga <= s["hi"] and "sig_entry" not in sudah_notif_masuk and not sudah_tp1_aktif:
                            sudah_notif_masuk.add("sig_entry")
                            kat_entry, ket_re = tentukan_klasifikasi_risiko_sinyal(s)
                            tag_risk_entry = " [⚠️ HIGH RISK]" if kat_entry == "HIGH RISK" else ""
                            info_k0 = evaluasi_limit_kena_kentut(s, harga, orders_live)
                            if kode_ambil_now == "DIAMBIL":
                                print(f"\n\n[ALERT ENTRY — SINYAL {s['arah']}{tag_risk_entry} SEDANG DIAMBIL!] Harga {harga:.2f} masuk area ENTRY {s['arah']} ({s['lo']:.2f}-{s['hi']:.2f})!")
                                print(f"   -> POSISI ANDA: {ringkas_ambil} sedang aktif mengawal setup ini.")
                            elif kode_ambil_now == "LIMIT":
                                print(f"\n\n[ALERT ENTRY — PENDING ORDER SIAP TERJEMPUT!] Harga {harga:.2f} masuk area ENTRY {s['arah']} ({s['lo']:.2f}-{s['hi']:.2f})!")
                                print(f"   -> PENDING ORDER: {ringkas_ambil} bersiap tereksekusi market.")
                                if info_k0:
                                    print(f"   -> 📌 NOTE ANTI 'KENA KENTUT': Jika harga cuma menyentuh tepi zona lalu memantul tanpa menjemput {ringkas_ambil}, Limit Anda MASIH VALID ditunggu selama belum terbang melewati {info_k0['batas_cancel']:.2f} (60% ke TP1). Jika lewat {info_k0['batas_cancel']:.2f}, WAJIB CANCEL!")
                            else:
                                print(f"\n\n[ALERT ENTRY — SINYAL {s['arah']}{tag_risk_entry} YANG TIDAK DIAMBIL!] Harga {harga:.2f} masuk area ENTRY {s['arah']} ({s['lo']:.2f}-{s['hi']:.2f})!")
                                if kat_entry == "HIGH RISK":
                                    print(f"   -> ⚠️ TINGKAT RISIKO: HIGH RISK ({ket_re}).")
                                    print("   -> PANDUAN TRADER : Sinyal ini dadakan di LTF M5. Anda bebas SKIP (TIDAK DIAMBIL) jika ingin santai, ATAU ambil dengan lot mini (0.25%).")
                                else:
                                    print(f"   -> STATUS AKUN: Anda BELUM mengambil sinyal {s['arah']} ini di MT5 (belum ada posisi terbuka).")
                                    print(f"   -> TINDAKAN: Jika ingin mengeksekusi manual sekarang, batasi risiko dengan batas SL di {s['sl']:.2f}.")
                                if info_k0:
                                    print(f"   -> 📌 NOTE LIMIT: {info_k0['catatan']}")
                            for _ in range(3):
                                bunyi(1200, 250)

                        # Deteksi Real-Time jika harga sempat colek zona ("sig_entry"), lalu terbang keluar zona tanpa menjemput Limit di ujung ("Kena Kentut")
                        if not sudah_tp1_aktif and kode_ambil_now != "DIAMBIL" and ("sig_entry" in sudah_notif_masuk or s.get("_ekstrem_dalam_zona") is not None):
                            keluar_zona_arah_tp = (s["arah"] == "BUY" and harga > s["hi"] + 1.2) or (s["arah"] == "SELL" and harga < s["lo"] - 1.2)
                            if keluar_zona_arah_tp:
                                info_kentut = evaluasi_limit_kena_kentut(s, harga, orders_live)
                                if info_kentut:
                                    if info_kentut["status_kode"] == "MASIH_VALID_SEMENTARA" and "limit_kentut_warn" not in sudah_notif_masuk:
                                        sudah_notif_masuk.add("limit_kentut_warn")
                                        print(f"\n\n[📌 NOTE ENTRI 'KENA KENTUT' (SHALLOW TAP ZONA {s['arah']}) — {info_kentut['judul']}]")
                                        print(f"   -> {info_kentut['catatan']}")
                                        bunyi(1100, 220)
                                    elif info_kentut["status_kode"] == "WAJIB_CANCEL" and "limit_kentut_cancel" not in sudah_notif_masuk:
                                        sudah_notif_masuk.add("limit_kentut_cancel")
                                        print(f"\n\n[🚨 PERINGATAN: ENTRI 'KENA KENTUT' SUDAH TERBANG JAUH — {info_kentut['judul']}]")
                                        print(f"   -> {info_kentut['catatan']}")
                                        for _ in range(3):
                                            bunyi(800, 280)

                        kena_sl_tick = (s["arah"] == "BUY" and harga <= s["sl"]) or (s["arah"] == "SELL" and harga >= s["sl"])
                        close_jebol_sl = (s["arah"] == "BUY" and close_m5 < s["sl"] - 0.35 * atr_m5) or (
                            s["arah"] == "SELL" and close_m5 > s["sl"] + 0.35 * atr_m5
                        )

                        # JIKA SUDAH TP1 (SL DI BEP+): Kena BEP+ bukanlah SL Loss atau Stop-Hunt, melainkan siklus Runner selesai!
                        if sudah_tp1_aktif:
                            if (s["arah"] == "BUY" and harga >= s["tp2"]) or (s["arah"] == "SELL" and harga <= s["tp2"]):
                                print(f"\n\n[🏆 TARGET PENUH TP2 TERCAPAI — SINYAL {s['arah']} SUKSES MAKSIMAL!] Harga {harga:.2f} menyapu TP2 ({s['tp2']:.2f})!")
                                _update_sinyal_aktif_pantau(otomatis, sudah_notif_masuk, None, f"TP2 Penuh ({s['tp2']:.2f}) Tercapai -> Siklus Sinyal {s['arah']} Selesai")
                                for _ in range(4):
                                    bunyi(1600, 250)
                            elif kena_sl_tick:
                                print(f"\n\n[✅ RUNNER SELESAI DI BEP+ PASCA TP1!] Harga ({harga:.2f}) kembali menyentuh batas BEP+ ({s['sl']:.2f}) setelah sebelumnya sukses mencapai TP1 ({s['tp1']:.2f}).")
                                print("   -> Siklus trade selesai dengan aman (Profit TP1 sudah diamankan, sisa posisi tertutup bebas risiko di BEP+).")
                                _update_sinyal_aktif_pantau(otomatis, sudah_notif_masuk, None, f"Siklus Sinyal {s['arah']} selesai di BEP+ ({s['sl']:.2f}) pasca TP1")
                                for _ in range(2):
                                    bunyi(1300, 220)
                        else:
                            if kena_sl_tick:
                                s["ekstrem_sl"] = min(s.get("ekstrem_sl", harga), harga) if s["arah"] == "BUY" else max(s.get("ekstrem_sl", harga), harga)
                                if "sig_sl_wick" not in sudah_notif_masuk and not close_jebol_sl:
                                    sudah_notif_masuk.add("sig_sl_wick")
                                    if kode_ambil_now == "TIDAK DIAMBIL":
                                        print(f"\n\n[LEVEL SL DISENTUH — SINYAL {s['arah']} YANG TIDAK DIAMBIL!] Harga {harga:.2f} menyentuh batas SL ({s['sl']:.2f})!")
                                        print(f"   -> STATUS AKUN: Akun Anda aman (bebas floating loss) karena sinyal {s['arah']} ini tidak Anda ambil.")
                                    else:
                                        print(f"\n\n[⚠️ WASPADA: LEVEL SL SINYAL ({s['sl']:.2f}) DISENTUH HARGA ({harga:.2f})!]")
                                        print("   -> Menunggu konfirmasi penutupan candle M5: Mengecek apakah ini hanya STOP-HUNT (jarum sweep lalu naik lagi) atau VALID BREAKOUT...")
                                    _aktifkan_zona_valid_berikutnya(
                                        otomatis,
                                        sudah_notif_masuk,
                                        s["arah"],
                                        s["sl"],
                                        harga,
                                        atr_m5,
                                        f"Level SL Sinyal {s['arah']} ({s['sl']:.2f}) tersentuh harga ({harga:.2f})",
                                        df_m5=df_m5,
                                        hanya_preview=True,
                                    )
                                    for _ in range(3):
                                        bunyi(720, 250)

                            # Jika tadi sempat menyentuh SL (wick), tapi harga berbalik masuk kembali ke zona Entry (Stop-Hunt Recovery!)
                            # CATATAN: Jangan izinkan Re-Entry di atas sinyal yang memang sudah Re-Entry / Dadakan!
                            bukan_sinyal_reentry_atau_dadakan = not any(
                                k in str(s.get("ket", "")).upper() for k in ("RE-ENTRY", "2ND CHANCE", "STOP-HUNT", "DADAKAN")
                            )
                            if "sig_sl_wick" in sudah_notif_masuk and "sig_sh_reentry" not in sudah_notif_masuk and bukan_sinyal_reentry_atau_dadakan:
                                lawan_inst_sh = (
                                    "PERINGATAN LONJAKAN MOMENTUM" in info_inst_cache
                                    and ((s["arah"] == "BUY" and "BEARISH" in info_inst_cache.upper()) or (s["arah"] == "SELL" and "BULLISH" in info_inst_cache.upper()))
                                )
                                konf_m5_sh = (
                                    (s["arah"] == "BUY" and close_m5 > s["sl"] and (close_m5 > open_m5_now or wick_bawah_m5 / rng_m5_now >= 0.38))
                                    or (s["arah"] == "SELL" and close_m5 < s["sl"] and (close_m5 < open_m5_now or wick_atas_m5 / rng_m5_now >= 0.38))
                                )
                                pulih_ke_entry = (
                                    ((s["arah"] == "BUY" and harga >= s["lo"]) or (s["arah"] == "SELL" and harga <= s["hi"]))
                                    and konf_m5_sh
                                    and not lawan_inst_sh
                                )
                                if pulih_ke_entry:
                                    sudah_notif_masuk.add("sig_sh_reentry")
                                    eks = s.get("ekstrem_sl", s["sl"])
                                    sl_baru_s = (eks - max(2.2, round(1.15 * atr_m5, 2))) if s["arah"] == "BUY" else (eks + max(2.2, round(1.15 * atr_m5, 2)))
                                    sinyal_sh = dict(
                                        arah=s["arah"],
                                        lo=s["lo"],
                                        hi=s["hi"],
                                        sl=round(sl_baru_s, 2),
                                        tp1=s["tp1"],
                                        tp2=s["tp2"],
                                        ket="Re-Entry 2nd Chance Pasca Stop-Hunt",
                                        risiko="HIGH RISK (Re-Entry Stop-Hunt)",
                                    )
                                    print(f"\n\n[🔥 STOP-HUNT PADA SINYAL {s['arah']} TERKONFIRMASI — PELUANG RE-ENTRY (2ND CHANCE — ⚠️ HIGH RISK)!]")
                                    print(f"   Harga sempat menyapu level SL ({s['sl']:.2f}) hingga {eks:.2f}, namun menarik ekor (wick) dan candle M5 kembali ke area Entry ({harga:.2f})!")
                                    print(f"   -> Ini adalah indikasi LIQUIDITY SWEEP institusi! Anda bisa pertimbangkan Re-Entry {s['arah']} dengan SL baru di {sl_baru_s:.2f}!")
                                    print("   -> KLASIFIKASI: ⚠️ HIGH RISK — Pertimbangkan hanya ambil jika siap risiko atau gunakan lot mini (0.25%). Boleh di-skip!")
                                    _update_sinyal_aktif_pantau(otomatis, sudah_notif_masuk, sinyal_sh, f"Stop-Hunt Sweep di {eks:.2f} & Re-Entry 2nd Chance [HIGH RISK]")
                                    sudah_notif_masuk.add("sig_sh_reentry")
                                    for _ in range(4):
                                        bunyi(1450, 230)

                            # Jika candle M5 resmi CLOSE menembus di balik SL (Valid Breakout / Setup Batal -> Langsung Geser ke Zona Valid Selanjutnya!)
                            if close_jebol_sl:
                                arah_lama = s["arah"]
                                lo_lama, hi_lama, sl_lama = s["lo"], s["hi"], s["sl"]
                                laporan_jebol = bedah_forensik_jebol(df_m5, f"Sinyal {arah_lama} Entry {lo_lama:.2f}-{hi_lama:.2f} (SL/Invalidasi {sl_lama:.2f} Jebol Close M5)", close_m5)
                                CATATAN_JEBOL_PANTAU.append(laporan_jebol)
                                if kode_ambil_now == "TIDAK DIAMBIL":
                                    print(f"\n\n[⛔ ZONA/SL SINYAL {arah_lama} JEBOL — SINYAL TIDAK DIAMBIL!] Candle M5 close di {close_m5:.2f} menembus SL ({sl_lama:.2f})!")
                                    print(f"   -> KEPUTUSAN TEPAT: Anda tidak mengambil sinyal {arah_lama} ini sehingga terhindar dari kerugian.")
                                else:
                                    print(f"\n\n[⛔ PERINGATAN: ZONA/SL SINYAL {arah_lama} ({sl_lama:.2f}) JEBOL CLOSE M5 ({close_m5:.2f})!]")
                                print(f"   -> Forensik Singkat: {laporan_jebol.splitlines()[-1]}")
                                batas_cari = min(sl_lama, close_m5) if arah_lama == "BUY" else max(sl_lama, close_m5)
                                _aktifkan_zona_valid_berikutnya(
                                    otomatis,
                                    sudah_notif_masuk,
                                    arah_lama,
                                    batas_cari,
                                    harga,
                                    atr_m5,
                                    f"Zona {arah_lama} Sebelumnya ({lo_lama:.2f}-{hi_lama:.2f}) Terkena SL / Jebol Close M5 di {close_m5:.2f}",
                                    df_m5=df_m5,
                                    hanya_preview=False,
                                )
                                print("   -> Mode Pantau TERUS BERJALAN mengawasi Zona Valid Selanjutnya di atas! (Tekan Ctrl+C lalu ketik 'refresh' jika ingin analisa penuh AI).")
                                for _ in range(5):
                                    bunyi(600, 350)
                            elif (s["arah"] == "BUY" and harga >= s["tp1"]) or (s["arah"] == "SELL" and harga <= s["tp1"]):
                                st_now_tp, _ = evaluasi_validitas_sinyal(s, harga)
                                sudah_masuk_zona = (
                                    "sig_entry" in sudah_notif_masuk
                                    or kode_ambil_now == "DIAMBIL"
                                    or st_now_tp in (ST_TP1_TERCAPAI, ST_SELESAI_TP2, ST_VALID_BERJALAN, ST_AKTIF_ENTRY)
                                )
                                if sudah_masuk_zona and "tp1" not in sudah_notif_masuk:
                                    sudah_notif_masuk.add("tp1")
                                    mid_entry = round((s["lo"] + s["hi"]) / 2.0, 2)
                                    bep_sl = round(mid_entry + 0.3, 2) if s["arah"] == "BUY" else round(mid_entry - 0.3, 2)
                                    sinyal_bep = dict(s, sl=bep_sl, sudah_tp1=True, _abaikan_jurnal=True, ket="TP1 Tercapai — SL Geser ke BEP+ (Runner ke TP2)")
                                    if kode_ambil_now == "TIDAK DIAMBIL":
                                        print(f"\n\n[TARGET TP1 TERCAPAI — SINYAL {s['arah']} YANG TIDAK DIAMBIL!] Harga {harga:.2f} menyentuh TP1 ({s['tp1']:.2f})!")
                                        print(f"   -> CATATAN: Sinyal {s['arah']} ini tadi tidak Anda ambil di akun MT5, namun analisa setup sukses mencapai target TP1.")
                                        print(f"   -> 📌 NOTE LIMIT: Jika Anda masih memasang Limit Order di zona {s['lo']:.2f}-{s['hi']:.2f} yang tadi terlewat ('kena kentut'), WAJIB SEGERA CANCEL karena zona sudah terpakai (mitigated)!")
                                    elif kode_ambil_now == "LIMIT":
                                        print(f"\n\n[🚨 TARGET TP1 TERCAPAI TANPA JEMPUT LIMIT ANDA ('KENA KENTUT') — WAJIB CANCEL {ringkas_ambil}!]")
                                        print(f"   -> Harga ({harga:.2f}) sudah terbang menyentuh TP1 ({s['tp1']:.2f}), sedangkan Pending Order Anda ({ringkas_ambil}) tertinggal di bawah/atas!")
                                        print(f"   -> TINDAKAN WAJIB: SEGERA CANCEL / HAPUS {ringkas_ambil} sekarang! Order institusi di zona {s['lo']:.2f}-{s['hi']:.2f} sudah habis terpakai (mitigated) dan sangat rawan jebol SL jika harga kembali lagi!")
                                    else:
                                        print(f"\n\n[TARGET TERCAPAI — POSISI {s['arah']} ANDA!] Harga {harga:.2f} menyentuh TP1 ({s['tp1']:.2f})! Amankan profit / geser SL ke BEP.")
                                    _update_sinyal_aktif_pantau(otomatis, sudah_notif_masuk, sinyal_bep, f"TP1 ({s['tp1']:.2f}) Tercapai -> SL Sinyal Aktif dikunci ke BEP+ ({bep_sl:.2f})")
                                    sudah_notif_masuk.add("tp1")
                                    for _ in range(4):
                                        bunyi(1500, 250)

                    # 3. Cek Zona Supply Atas & Zona Demand Bawah dari AI (Wajib Konfirmasi Trigger M5 sebelum aktif jadi sinyal!)
                    if otomatis["atas"]:
                        lo_a, hi_a, jebol_a = otomatis["atas"]
                        if lo_a <= harga <= hi_a and "atas" not in sudah_notif_masuk:
                            sudah_notif_masuk.add("atas")
                            pola_info = ", ".join(pola_m5) if pola_m5 else "belum ada pola rejection"
                            print(f"\n\n[ALERT ZONA SUPPLY!] Harga {harga:.2f} masuk Zona Supply Atas ({lo_a:.2f} - {hi_a:.2f})!")
                            print(f"   -> Menunggu konfirmasi Sweep / CISD / Rejection Bearish di M5 sebelum menerbitkan sinyal entry. Pola M5: {pola_info} | {calon_cisd_m5}")
                            for _ in range(3):
                                bunyi(1200, 250)

                        # Cek apakah saat harga di/dekat Zona Supply sudah muncul TRIGGER M5 Bearish (Rejection/Sweep/CISD) & tidak sedang dihantam Marubozu Bullish
                        if (lo_a - 0.8 <= harga <= hi_a + 1.2) and (not otomatis["sinyal"] or otomatis["sinyal"]["arah"] != "SELL") and "atas_konf" not in sudah_notif_masuk:
                            impuls_bull_menghantam = (close_m5 > open_m5_now and (close_m5 - open_m5_now) >= 1.15 * atr_m5) or (
                                "PERINGATAN LONJAKAN MOMENTUM" in info_inst_cache and "BULLISH" in info_inst_cache.upper()
                            )
                            ada_pola_bear = any(k in " ".join(pola_m5).upper() for k in ("SHOOTING STAR", "BEARISH ENGULFING"))
                            ada_cisd_bear = bool(cisd_m5 and cisd_m5[-1]["arah"] == "BEARISH" and ( len(df_m5) - 1 - cisd_m5[-1]["k"] ) <= 3)
                            ada_sweep_bear = bool(sweeps and "BUY-SIDE" in sweeps[-1][1] and sweeps[-1][0] >= df_m5.index[-3])
                            ada_wick_bear = (wick_atas_m5 / rng_m5_now >= 0.40 and close_m5 <= hi_a and close_m5 < open_m5_now + 0.2 * atr_m5)
                            if not impuls_bull_menghantam and (ada_pola_bear or ada_cisd_bear or ada_sweep_bear or ada_wick_bear):
                                sudah_notif_masuk.add("atas_konf")
                                sl_sup = round(max(jebol_a, hi_a + max(2.2, 1.15 * atr_m5)), 2)
                                r_sup = max(2.5, sl_sup - lo_a)
                                sig_sup = dict(
                                    arah="SELL",
                                    lo=lo_a,
                                    hi=hi_a,
                                    sl=sl_sup,
                                    tp1=round(lo_a - 1.6 * r_sup, 2),
                                    tp2=round(lo_a - 3.0 * r_sup, 2),
                                    ket="Terkonfirmasi Reaksi Zona Supply Atas (Trigger M5 Valid)",
                                    risiko="MEDIUM RISK",
                                )
                                _update_sinyal_aktif_pantau(otomatis, sudah_notif_masuk, sig_sup, f"Trigger M5 Bearish Terkonfirmasi di Zona Supply ({lo_a:.2f}-{hi_a:.2f})")

                        if close_m5 > max(jebol_a, hi_a + 0.5 * atr_m5):
                            laporan_jebol_a = bedah_forensik_jebol(df_m5, f"Zona Supply Atas ({lo_a:.2f}-{hi_a:.2f}) Jebol ke Atas (Breakout Naik)", close_m5)
                            CATATAN_JEBOL_PANTAU.append(laporan_jebol_a)
                            print(f"\n\n[PERINGATAN: ZONA SUPPLY ({lo_a:.2f}-{hi_a:.2f}) JEBOL / BREAKOUT NAIK!] Candle M5 close di {close_m5:.2f} menembus batas SL/Invalidasi {jebol_a:.2f}!")
                            print(f"   -> Forensik Singkat: {laporan_jebol_a.splitlines()[-1]}")
                            print(f"   -> Catatan Flip Zone: Eks-Supply {lo_a:.2f}-{hi_a:.2f} kini berpotensi menjadi Flip Demand jika harga retest dari atas.")
                            sudah_notif_masuk.discard("atas_konf")
                            _aktifkan_zona_valid_berikutnya(
                                otomatis,
                                sudah_notif_masuk,
                                "SELL",
                                max(jebol_a, close_m5),
                                harga,
                                atr_m5,
                                f"Zona Supply Sebelumnya ({lo_a:.2f}-{hi_a:.2f}) Jebol ke Atas di {close_m5:.2f}",
                                df_m5=df_m5,
                                hanya_preview=False,
                            )
                            print("   -> Mode Pantau TERUS BERJALAN mengawasi Zona Supply Valid Selanjutnya di atas!")
                            for _ in range(5):
                                bunyi(600, 350)

                    if otomatis["bawah"]:
                        lo_b, hi_b, jebol_b = otomatis["bawah"]
                        if lo_b <= harga <= hi_b and "bawah" not in sudah_notif_masuk:
                            sudah_notif_masuk.add("bawah")
                            pola_info = ", ".join(pola_m5) if pola_m5 else "belum ada pola rejection"
                            print(f"\n\n[ALERT ZONA DEMAND!] Harga {harga:.2f} masuk Zona Demand Bawah ({lo_b:.2f} - {hi_b:.2f})!")
                            print(f"   -> Menunggu konfirmasi Sweep / CISD / Rejection Bullish di M5 sebelum menerbitkan sinyal entry. Pola M5: {pola_info} | {calon_cisd_m5}")
                            for _ in range(3):
                                bunyi(1200, 250)

                        # Cek apakah saat harga di/dekat Zona Demand sudah muncul TRIGGER M5 Bullish & tidak sedang dihantam Marubozu Bearish
                        if (lo_b - 1.2 <= harga <= hi_b + 0.8) and (not otomatis["sinyal"] or otomatis["sinyal"]["arah"] != "BUY") and "bawah_konf" not in sudah_notif_masuk:
                            impuls_bear_menghantam = (close_m5 < open_m5_now and (open_m5_now - close_m5) >= 1.15 * atr_m5) or (
                                "PERINGATAN LONJAKAN MOMENTUM" in info_inst_cache and "BEARISH" in info_inst_cache.upper()
                            )
                            ada_pola_bull = any(k in " ".join(pola_m5).upper() for k in ("HAMMER", "BULLISH ENGULFING"))
                            ada_cisd_bull = bool(cisd_m5 and cisd_m5[-1]["arah"] == "BULLISH" and (len(df_m5) - 1 - cisd_m5[-1]["k"]) <= 3)
                            ada_sweep_bull = bool(sweeps and "SELL-SIDE" in sweeps[-1][1] and sweeps[-1][0] >= df_m5.index[-3])
                            ada_wick_bull = (wick_bawah_m5 / rng_m5_now >= 0.40 and close_m5 >= lo_b and close_m5 > open_m5_now - 0.2 * atr_m5)
                            if not impuls_bear_menghantam and (ada_pola_bull or ada_cisd_bull or ada_sweep_bull or ada_wick_bull):
                                sudah_notif_masuk.add("bawah_konf")
                                sl_dem = round(min(jebol_b, lo_b - max(2.2, 1.15 * atr_m5)), 2)
                                r_dem = max(2.5, hi_b - sl_dem)
                                sig_dem = dict(
                                    arah="BUY",
                                    lo=lo_b,
                                    hi=hi_b,
                                    sl=sl_dem,
                                    tp1=round(hi_b + 1.6 * r_dem, 2),
                                    tp2=round(hi_b + 3.0 * r_dem, 2),
                                    ket="Terkonfirmasi Reaksi Zona Demand Bawah (Trigger M5 Valid)",
                                    risiko="MEDIUM RISK",
                                )
                                _update_sinyal_aktif_pantau(otomatis, sudah_notif_masuk, sig_dem, f"Trigger M5 Bullish Terkonfirmasi di Zona Demand ({lo_b:.2f}-{hi_b:.2f})")

                        if close_m5 < min(jebol_b, lo_b - 0.5 * atr_m5):
                            laporan_jebol_b = bedah_forensik_jebol(df_m5, f"Zona Demand Bawah ({lo_b:.2f}-{hi_b:.2f}) Jebol ke Bawah (Breakout Turun)", close_m5)
                            CATATAN_JEBOL_PANTAU.append(laporan_jebol_b)
                            print(f"\n\n[PERINGATAN: ZONA DEMAND ({lo_b:.2f}-{hi_b:.2f}) JEBOL / BREAKOUT TURUN!] Candle M5 close di {close_m5:.2f} menembus batas SL/Invalidasi {jebol_b:.2f}!")
                            print(f"   -> Forensik Singkat: {laporan_jebol_b.splitlines()[-1]}")
                            print(f"   -> Catatan Flip Zone: Eks-Demand {lo_b:.2f}-{hi_b:.2f} kini berpotensi menjadi Flip Supply jika harga retest dari bawah.")
                            sudah_notif_masuk.discard("bawah_konf")
                            _aktifkan_zona_valid_berikutnya(
                                otomatis,
                                sudah_notif_masuk,
                                "BUY",
                                min(jebol_b, close_m5),
                                harga,
                                atr_m5,
                                f"Zona Demand Sebelumnya ({lo_b:.2f}-{hi_b:.2f}) Jebol ke Bawah di {close_m5:.2f}",
                                df_m5=df_m5,
                                hanya_preview=False,
                            )
                            print("   -> Mode Pantau TERUS BERJALAN mengawasi Zona Demand Valid Selanjutnya di bawah!")
                            for _ in range(5):
                                bunyi(600, 350)

                    # 4. Cek CHoCH, CISD, Liquidity Sweep, atau Ledakan Orderan Institusi / Sinyal Reversal Pasca-Sideways di M5
                    info_inst = info_inst_cache
                    if "PERINGATAN LONJAKAN MOMENTUM (TICK-VOLUME, BUKAN DATA INSTITUSI)" in info_inst and "inst" not in sudah_notif_masuk:
                        sudah_notif_masuk.add("inst")
                        baris_inst = info_inst.split("\n")[0]
                        print(f"\n\n[ALERT INSTITUSI!] {baris_inst}")
                        for tkt_o, inf_o in pending_terpantau.items():
                            lawan_inst = (inf_o["arah"] == "BUY" and "BEARISH" in baris_inst.upper()) or (inf_o["arah"] == "SELL" and "BULLISH" in baris_inst.upper())
                            if lawan_inst and "LIMIT" in inf_o["nama"]:
                                print(f"   -> [🚨 PERINGATAN PENDING ORDER #{tkt_o}!] {inf_o['nama']} Anda @ {inf_o['open']:.2f} MELAWAN Orderan Besar Institusi (One-Way Impulse)!")
                                print(f"      STATUS ZONA: TIDAK VALID / RAWAN JEBOL! Segera CANCEL/HAPUS {inf_o['nama']} #{tkt_o} agar tidak tertabrak impuls institusi!")
                        if otomatis["sinyal"]:
                            sg_now = otomatis["sinyal"]
                            sig_lawan_inst = (sg_now["arah"] == "BUY" and "BEARISH" in baris_inst.upper()) or (sg_now["arah"] == "SELL" and "BULLISH" in baris_inst.upper())
                            if sig_lawan_inst:
                                _update_sinyal_aktif_pantau(otomatis, sudah_notif_masuk, None, f"Dibatalkan Otomatis karena melawan {baris_inst}")
                        for _ in range(4):
                            bunyi(900, 300)
                    if "SINYAL PEMBALIKAN" in info_inst and "rev_box" not in sudah_notif_masuk:
                        sudah_notif_masuk.add("rev_box")
                        print(f"\n\n[ALERT PEMBALIKAN PASCA-SIDEWAYS!] {info_inst.split('->')[-1].strip()}")
                        for _ in range(4):
                            bunyi(1350, 300)

                    if cisd_m5:
                        t_c = cisd_m5[-1]["waktu"]
                        if last_cisd_time is None:
                            last_cisd_time = t_c
                        elif t_c > last_cisd_time:
                            last_cisd_time = t_c
                            c_last = cisd_m5[-1]
                            print(f"\n\n[ALERT CISD M5 (CHANGE IN STATE OF DELIVERY)!] {c_last['ket']} pada {wkt(t_c)}!")
                            for tkt, inf in tiket_terpantau.items():
                                if (inf["arah"] == "BUY" and c_last["arah"] == "BEARISH") or (inf["arah"] == "SELL" and c_last["arah"] == "BULLISH"):
                                    aksi_c = "Segera amankan profit / pasang BEP!" if inf["poin"] > 0 else "CISD M5 berbalik melawan posisi Anda! Waspada pembalikan dini, pertimbangkan perketat SL / Cut terukur."
                                    print(f"   -> [BAHAYA POSISI #{tkt}!] Posisi Anda {inf['arah']} ({inf['poin']:+.2f} pt / ${inf['profit']:+.2f}) berlawanan dengan {c_last['arah']} CISD M5! {aksi_c}")
                                    for _ in range(3):
                                        bunyi(700, 300)
                            # Jika muncul CISD M5:
                            # 🔒 DIRECTIONAL LOCK: CISD berlawanan TIDAK BOLEH mengganti sinyal aktif!
                            # Hanya boleh membuat sinyal baru jika status WAIT (tidak ada sinyal aktif),
                            # DAN jika di luar zona HTF ("Dadakan") WAJIB berupa [SWEEP + CISD VALID], searah Bias HTF, & lolos cooldown!
                            if c_last["arah"] == "BEARISH":
                                di_zona_supply = bool(otomatis.get("atas") and (otomatis["atas"][0] - 2.5 <= harga <= otomatis["atas"][1] + 2.5))
                                if not otomatis["sinyal"]:
                                    searah_htf_bear = (_bias_bar["bias"] == "BEARISH" and abs(_bias_bar["skor"]) >= BIAS_MIN_SKOR)
                                    layak_dadakan_bear = bool(c_last.get("is_sweep") and searah_htf_bear and (now_s - last_dadakan_ts) >= 1200)
                                    if di_zona_supply or layak_dadakan_bear:
                                        sl_base = max(c_last["ekstrem"], otomatis["atas"][1]) if di_zona_supply else c_last["ekstrem"]
                                        sl_c = round(sl_base + max(2.2, 1.15 * atr_m5), 2)
                                        r_c = max(2.5, sl_c - harga)
                                        tp1_c = round(harga - 1.8 * r_c, 2)
                                        tp2_c = round(harga - 3.0 * r_c, 2)
                                        risiko_c = "MEDIUM RISK" if di_zona_supply else "HIGH RISK"
                                        ket_c = "Terkonfirmasi Bearish CISD di Zona Supply" if di_zona_supply else f"Bearish Sweep+CISD M5 Searah HTF (Open Delivery {c_last['level']:.2f})"
                                        sig_c = dict(arah="SELL", lo=round(harga - 0.5, 2), hi=round(max(harga + 1.0, c_last["level"]), 2), sl=sl_c, tp1=tp1_c, tp2=tp2_c, ket=ket_c, risiko=risiko_c)
                                        if _update_sinyal_aktif_pantau(otomatis, sudah_notif_masuk, sig_c, f"Bearish CISD M5 [{risiko_c}] ({ket_c})") is not False and not di_zona_supply:
                                            last_dadakan_ts = now_s
                                    else:
                                        print("   -> [ℹ️ FILTER AKURASI M5]: Bearish CISD ini terjadi di luar Zona Supply HTF / tanpa Sweep+HTF Alignment — hanya peringatan, tidak dijadikan sinyal entry.")
                                elif otomatis["sinyal"]["arah"] == "SELL":
                                    print(f"   -> [💪 KONFIRMASI PENGUATAN SINYAL SELL!] Bearish CISD M5 terbentuk di {c_last['level']:.2f}! Setup SELL yang sedang berjalan semakin valid!")
                                elif otomatis["sinyal"]["arah"] == "BUY":
                                    # 🔒 DIRECTIONAL LOCK: JANGAN ganti sinyal BUY ke SELL — hanya PERINGATAN!
                                    print(f"   -> [⚠️ PERINGATAN!] Bearish CISD M5 terbentuk di {c_last['level']:.2f} melawan sinyal BUY aktif.")
                                    print(f"      🔒 DIRECTIONAL LOCK: Sinyal BUY Anda ({otomatis['sinyal']['lo']:.2f}-{otomatis['sinyal']['hi']:.2f}) TETAP DIPERTAHANKAN.")
                                    print(f"      Sinyal BUY hanya batal jika harga close M5 menembus SL ({otomatis['sinyal']['sl']:.2f}).")
                            elif c_last["arah"] == "BULLISH":
                                di_zona_demand = bool(otomatis.get("bawah") and (otomatis["bawah"][0] - 2.5 <= harga <= otomatis["bawah"][1] + 2.5))
                                if not otomatis["sinyal"]:
                                    searah_htf_bull = (_bias_bar["bias"] == "BULLISH" and abs(_bias_bar["skor"]) >= BIAS_MIN_SKOR)
                                    layak_dadakan_bull = bool(c_last.get("is_sweep") and searah_htf_bull and (now_s - last_dadakan_ts) >= 1200)
                                    if di_zona_demand or layak_dadakan_bull:
                                        sl_base = min(c_last["ekstrem"], otomatis["bawah"][0]) if di_zona_demand else c_last["ekstrem"]
                                        sl_c = round(sl_base - max(2.2, 1.15 * atr_m5), 2)
                                        r_c = max(2.5, harga - sl_c)
                                        tp1_c = round(harga + 1.8 * r_c, 2)
                                        tp2_c = round(harga + 3.0 * r_c, 2)
                                        risiko_c = "MEDIUM RISK" if di_zona_demand else "HIGH RISK"
                                        ket_c = "Terkonfirmasi Bullish CISD di Zona Demand" if di_zona_demand else f"Bullish Sweep+CISD M5 Searah HTF (Open Delivery {c_last['level']:.2f})"
                                        sig_c = dict(arah="BUY", lo=round(min(harga - 1.0, c_last["level"]), 2), hi=round(harga + 0.5, 2), sl=sl_c, tp1=tp1_c, tp2=tp2_c, ket=ket_c, risiko=risiko_c)
                                        if _update_sinyal_aktif_pantau(otomatis, sudah_notif_masuk, sig_c, f"Bullish CISD M5 [{risiko_c}] ({ket_c})") is not False and not di_zona_demand:
                                            last_dadakan_ts = now_s
                                    else:
                                        print("   -> [ℹ️ FILTER AKURASI M5]: Bullish CISD ini terjadi di luar Zona Demand HTF / tanpa Sweep+HTF Alignment — hanya peringatan, tidak dijadikan sinyal entry.")
                                elif otomatis["sinyal"]["arah"] == "BUY":
                                    print(f"   -> [💪 KONFIRMASI PENGUATAN SINYAL BUY!] Bullish CISD M5 terbentuk di {c_last['level']:.2f}! Setup BUY yang sedang berjalan semakin valid!")
                                elif otomatis["sinyal"]["arah"] == "SELL":
                                    # 🔒 DIRECTIONAL LOCK: JANGAN ganti sinyal SELL ke BUY — hanya PERINGATAN!
                                    print(f"   -> [⚠️ PERINGATAN!] Bullish CISD M5 terbentuk di {c_last['level']:.2f} melawan sinyal SELL aktif.")
                                    print(f"      🔒 DIRECTIONAL LOCK: Sinyal SELL Anda ({otomatis['sinyal']['lo']:.2f}-{otomatis['sinyal']['hi']:.2f}) TETAP DIPERTAHANKAN.")
                                    print(f"      Sinyal SELL hanya batal jika harga close M5 menembus SL ({otomatis['sinyal']['sl']:.2f}).")
                            print("   -> TINDAKAN: Konfirmasi pembalikan dini ICT (CISD) terbentuk! Tekan Ctrl+C lalu ketik 'refresh' jika ingin re-analisa!")
                            for _ in range(2):
                                bunyi(1250, 220)

                    if evs:
                        t_ev = df_m5.index[evs[-1]["k"]]
                        if last_event_time is None:
                            last_event_time = t_ev
                        elif t_ev > last_event_time:
                            last_event_time = t_ev
                            ev_item = evs[-1]
                            print(f"\n\n[ALERT STRUKTUR M5!] Terbentuk {ev_terakhir}!")
                            for tkt, inf in tiket_terpantau.items():
                                if (inf["arah"] == "BUY" and ev_item["arah"] == "BEARISH") or (inf["arah"] == "SELL" and ev_item["arah"] == "BULLISH"):
                                    aksi_saran = "Segera amankan profit / pasang BEP!" if inf["poin"] > 0 else "Struktur M5 berbalik melawan posisi Anda! Pertimbangkan Cut Loss terukur / perketat SL."
                                    print(f"   -> [BAHAYA POSISI #{tkt}!] Posisi Anda {inf['arah']} ({inf['poin']:+.2f} pt / ${inf['profit']:+.2f}) berlawanan dengan {ev_item['tipe']} {ev_item['arah']} M5! {aksi_saran}")
                                    for _ in range(3):
                                        bunyi(700, 300)
                            print("   -> TINDAKAN: Tekan Ctrl+C lalu ketik 'refresh' jika ini konfirmasi searah zona incaran Anda!")
                            for _ in range(2):
                                bunyi(1000, 250)
                    if sweeps:
                        t_sw = sweeps[-1][0]
                        if last_sweep_time is None:
                            last_sweep_time = t_sw
                        elif t_sw > last_sweep_time:
                            last_sweep_time = t_sw
                            sw_terakhir = sweeps[-1]
                            arah_saran = "SELL (pasang SL di atas ujung wick)" if "BUY-SIDE" in sw_terakhir[1] else "BUY (pasang SL di bawah ujung wick)"
                            print(f"\n\n[ALERT SWEEP M5!] {sw_terakhir[1]} di level {sw_terakhir[2]:.2f} (Wick: {sw_terakhir[3]:.2f}, Close: {sw_terakhir[4]:.2f})!")
                            for tkt, inf in tiket_terpantau.items():
                                melawan_sw = (inf["arah"] == "BUY" and "BUY-SIDE" in sw_terakhir[1]) or (inf["arah"] == "SELL" and "SELL-SIDE" in sw_terakhir[1])
                                if melawan_sw:
                                    print(f"   -> [BAHAYA POSISI #{tkt}!] Sapuan likuiditas ini berpotensi membalikkan harga melawan posisi {inf['arah']} Anda ({inf['poin']:+.2f} pt)! Segera amankan posisi / BEP!")
                                    for _ in range(3):
                                        bunyi(700, 300)
                            # Jika terjadi sweep:
                            # 🔒 DIRECTIONAL LOCK: Sweep berlawanan TIDAK BOLEH mengganti sinyal aktif!
                            if "BUY-SIDE" in sw_terakhir[1]:
                                di_zona_supply = bool(otomatis.get("atas") and (otomatis["atas"][0] - 2.5 <= sw_terakhir[3] <= otomatis["atas"][1] + 3.5))
                                if not otomatis["sinyal"]:
                                    searah_htf_sw_bear = (_bias_bar["bias"] == "BEARISH" and abs(_bias_bar["skor"]) >= BIAS_MIN_SKOR)
                                    wick_kuat_bear = (sw_terakhir[3] - sw_terakhir[2]) >= 0.35 * atr_m5
                                    if di_zona_supply or (searah_htf_sw_bear and wick_kuat_bear and (now_s - last_dadakan_ts) >= 1200):
                                        sl_sw = round(sw_terakhir[3] + max(2.2, 1.15 * atr_m5), 2)
                                        r_sw = max(2.5, sl_sw - harga)
                                        risiko_sw = "MEDIUM RISK" if di_zona_supply else "HIGH RISK"
                                        ket_sw = "Terkonfirmasi Buy-Side Sweep di Zona Supply" if di_zona_supply else f"Buy-Side Sweep M5 Searah HTF (Wick {sw_terakhir[3]:.2f})"
                                        sig_sw = dict(arah="SELL", lo=round(harga - 0.5, 2), hi=round(max(harga + 1.0, sw_terakhir[2]), 2), sl=sl_sw, tp1=round(harga - 1.8 * r_sw, 2), tp2=round(harga - 3.0 * r_sw, 2), ket=ket_sw, risiko=risiko_sw)
                                        if _update_sinyal_aktif_pantau(otomatis, sudah_notif_masuk, sig_sw, f"Buy-Side Sweep [{risiko_sw}] ({ket_sw})") is not False and not di_zona_supply:
                                            last_dadakan_ts = now_s
                                    else:
                                        print("   -> [ℹ️ FILTER AKURASI M5]: Sweep ini di luar Zona Supply HTF / melawan Bias HTF — hanya peringatan, tidak dijadikan sinyal entry.")
                                elif otomatis["sinyal"]["arah"] == "SELL":
                                    print(f"   -> [💪 KONFIRMASI PENGUATAN SINYAL SELL!] Buy-Side Sweep di {sw_terakhir[2]:.2f} menolak kenaikan! Posisi SELL Anda semakin terkonfirmasi!")
                                elif otomatis["sinyal"]["arah"] == "BUY":
                                    # 🔒 DIRECTIONAL LOCK: JANGAN ganti sinyal BUY ke SELL
                                    print(f"   -> [⚠️ PERINGATAN!] Buy-Side Sweep di {sw_terakhir[2]:.2f} berlawanan arah dengan sinyal BUY aktif.")
                                    print(f"      🔒 DIRECTIONAL LOCK: Sinyal BUY ({otomatis['sinyal']['lo']:.2f}-{otomatis['sinyal']['hi']:.2f}) TETAP DIPERTAHANKAN sampai SL ({otomatis['sinyal']['sl']:.2f}) terbukti jebol.")
                            elif "SELL-SIDE" in sw_terakhir[1]:
                                di_zona_demand = bool(otomatis.get("bawah") and (otomatis["bawah"][0] - 3.5 <= sw_terakhir[3] <= otomatis["bawah"][1] + 2.5))
                                if not otomatis["sinyal"]:
                                    searah_htf_sw_bull = (_bias_bar["bias"] == "BULLISH" and abs(_bias_bar["skor"]) >= BIAS_MIN_SKOR)
                                    wick_kuat_bull = (sw_terakhir[2] - sw_terakhir[3]) >= 0.35 * atr_m5
                                    if di_zona_demand or (searah_htf_sw_bull and wick_kuat_bull and (now_s - last_dadakan_ts) >= 1200):
                                        sl_sw = round(sw_terakhir[3] - max(2.2, 1.15 * atr_m5), 2)
                                        r_sw = max(2.5, harga - sl_sw)
                                        risiko_sw = "MEDIUM RISK" if di_zona_demand else "HIGH RISK"
                                        ket_sw = "Terkonfirmasi Sell-Side Sweep di Zona Demand" if di_zona_demand else f"Sell-Side Sweep M5 Searah HTF (Wick {sw_terakhir[3]:.2f})"
                                        sig_sw = dict(arah="BUY", lo=round(min(harga - 1.0, sw_terakhir[2]), 2), hi=round(harga + 0.5, 2), sl=sl_sw, tp1=round(harga + 1.8 * r_sw, 2), tp2=round(harga + 3.0 * r_sw, 2), ket=ket_sw, risiko=risiko_sw)
                                        if _update_sinyal_aktif_pantau(otomatis, sudah_notif_masuk, sig_sw, f"Sell-Side Sweep [{risiko_sw}] ({ket_sw})") is not False and not di_zona_demand:
                                            last_dadakan_ts = now_s
                                    else:
                                        print("   -> [ℹ️ FILTER AKURASI M5]: Sweep ini di luar Zona Demand HTF / melawan Bias HTF — hanya peringatan, tidak dijadikan sinyal entry.")
                                elif otomatis["sinyal"]["arah"] == "BUY":
                                    print(f"   -> [💪 KONFIRMASI PENGUATAN SINYAL BUY!] Sell-Side Sweep di {sw_terakhir[2]:.2f} menolak penurunan! Posisi BUY Anda semakin terkonfirmasi!")
                                elif otomatis["sinyal"]["arah"] == "SELL":
                                    # 🔒 DIRECTIONAL LOCK: JANGAN ganti sinyal SELL ke BUY
                                    print(f"   -> [⚠️ PERINGATAN!] Sell-Side Sweep di {sw_terakhir[2]:.2f} berlawanan arah dengan sinyal SELL aktif.")
                                    print(f"      🔒 DIRECTIONAL LOCK: Sinyal SELL ({otomatis['sinyal']['lo']:.2f}-{otomatis['sinyal']['hi']:.2f}) TETAP DIPERTAHANKAN sampai SL ({otomatis['sinyal']['sl']:.2f}) terbukti jebol.")
                            print(f"   -> TINDAKAN: Ini konfirmasi sapuan likuiditas untuk {arah_saran}. Tekan Ctrl+C lalu ketik 'refresh' untuk cek entry!")
                            for _ in range(2):
                                bunyi(1400, 300)

                _err_beruntun = 0
            except KeyboardInterrupt:
                raise
            except Exception as e_loop:
                _err_beruntun += 1
                _catat_error('mode_pantau', e_loop)
                if _err_beruntun in (1, 10) or _err_beruntun % 100 == 0:
                    print(f'\n[⚠️ Error di loop pantau (#{_err_beruntun}): {e_loop!r} — dicatat ke {FILE_ERROR_LOG}; pemantauan dilanjutkan]')
                time.sleep(1.0)
                continue
            time.sleep(0.1)  # Live update 10x per detik (halus, stabil, dan bebas lag)
    except KeyboardInterrupt:
        print("\nMode pantau dihentikan (koneksi live MT5 tetap aktif).")
    return None
