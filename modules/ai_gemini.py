# -*- coding: utf-8 -*-
"""
modules/ai_gemini.py
Integrasi Google Gemini API: pengumpulan konteks lengkap data MT5,
ekstraksi rekomendasi sinyal & zona [RADAR_TARGET], multi-model failover,
serta live snapshot kontekstual untuk Tanya Jawab AI.
"""
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import MetaTrader5 as mt5
import pandas as pd
from google import genai
from google.genai import types

from modules.config import (
    SYMBOL, TIMEFRAMES, SWING_N, FILTER_TREND_AKTIF, SERVER_UTC_OFFSET_JAM,
    MODELS_TO_TRY, MODEL_AKTIF_TERCEPAT, MODEL_404_BLACKLIST, MODEL_COOLDOWN
)
from modules.utils import KonteksError, _catat_error, deteksi_offset, wkt
from modules.mt5_conn import (
    pastikan_mt5_aktif, ambil_data, tambah_indikator, ambil_candle_berjalan,
    cek_posisi_terbuka_mt5, analisa_akun_dan_lot
)
from modules.analisis_teknikal import (
    ringkas_tf, level_psikologis_ekstrem, ringkas_zona,
    analisa_metode, cari_golden_confluence, hitung_bias_htf,
    _cari_zona_valid_berikutnya_mt5, analisa_smt_divergence,
    bias_htf_cached, teks_bias_htf, struktur, deteksi_cisd,
    liquidity_sweeps, deteksi_institusi_dan_ranging, deteksi_pola
)
from modules.analisis_pasar import (
    analisa_sesi, cek_kalender_berita_usd, analisa_korelasi_usd,
    analisa_adr_exhaustion
)
from modules.sinyal_jurnal import (
    validasi_sinyal_struktur, cek_status_eksekusi_sinyal,
    evaluasi_validitas_sinyal, sinyal_masih_aktif, _STATE_PARSER
)
from modules.state_manager import (
    SINYAL_AKTIF_DINAMIS, SINYAL_SEBELUMNYA_DINAMIS,
    ZONA_ATAS_DINAMIS, ZONA_BAWAH_DINAMIS, CATATAN_JEBOL_PANTAU
)
import modules.state_manager as SM
from modules.zone_scanner import scan_xauusd_zones, validate_entry_setup, evaluate_trade_decision


def kumpulkan_konteks_mt5(is_refresh=False):
    if not is_refresh:
        print("1. Menghubungkan ke MetaTrader5...")
    if not pastikan_mt5_aktif():
        raise KonteksError(f"Gagal initialize MT5 (pastikan aplikasi MT5 terbuka dan login): {mt5.last_error()}")

    if not mt5.symbol_select(SYMBOL, True):
        raise KonteksError(f"Simbol {SYMBOL} tidak ditemukan di broker Anda. Cek nama simbol di Market Watch.")
    frames = {nama: ambil_data(tf, n) for nama, (tf, n) in TIMEFRAMES.items()}
    for nama, df in frames.items():   # validasi SEBELUM dipakai fungsi analisa lain
        if df is None or len(df) < 60:
            raise KonteksError(f"Data {nama} tidak cukup/gagal diambil dari MT5 (history belum tersinkron? coba lagi sebentar).")
    tick = mt5.symbol_info_tick(SYMBOL)
    acc_info = mt5.account_info()
    sym_info = mt5.symbol_info(SYMBOL)
    usd_text = analisa_korelasi_usd()
    smt_text = analisa_smt_divergence(frames)
    harga_awal = tick.bid if tick else frames["M5"]["Close"].iloc[-1]
    posisi_text = cek_posisi_terbuka_mt5(harga_awal)

    # Bandingkan dengan jam server dari tick (bukan UTC) supaya benar untuk broker dengan offset != 0
    selisih = (pd.Timestamp(int(tick.time), unit="s") - frames["M5"].index[-1]) if tick else pd.Timedelta(0)
    if is_refresh:
        print(f"   [LIVE MT5] Bid: {tick.bid if tick else harga_awal:.2f} | Candle M5 terakhir: {frames['M5'].index[-1]}")
    else:
        print(f"   Candle M5 terakhir: {frames['M5'].index[-1]} (waktu server broker)")
    if tick is not None and selisih.total_seconds() > 3600 * 6:
        print("   PERINGATAN: data terakhir sudah lama, market mungkin tutup. Hasil analisa kurang relevan.")

    atr_m5_val = tambah_indikator(frames["M5"].copy())["ATR14"].iloc[-1]
    spread_val = (tick.ask - tick.bid) if tick else 0.0
    harga_sekarang = f"Bid {tick.bid:.2f} / Ask {tick.ask:.2f} (spread {spread_val:.2f}) | Volatilitas ATR14 M5: {atr_m5_val:.2f} poin" if tick else "tidak tersedia"
    harga_ref = tick.bid if tick else frames["M5"]["Close"].iloc[-1]

    zona_text = "".join(ringkas_zona(n, frames[n], harga_ref) for n in ("W1", "D1", "H4"))
    zona_text += level_psikologis_ekstrem(frames["W1"], frames["D1"], harga_ref)
    # Bias HTF — dicantumkan di konteks agar AI tidak lupa arah trend utama
    _bias_ctx = hitung_bias_htf(frames)
    _bias_detail_ctx = "\n".join(f"  {k}: {v}" for k, v in _bias_ctx["detail"].items())
    zona_text += (
        f"\n===== BIAS TREND HTF (FILTER ARAH SINYAL) =====\n"
        f"Bias Gabungan D1+H4+H1: {_bias_ctx['bias']} (skor {_bias_ctx['skor']:+d}/5)\n"
        f"{_bias_detail_ctx}\n"
        f"{'Filter Trend AKTIF: hanya sinyal SEARAH bias diterbitkan.' if FILTER_TREND_AKTIF else 'Filter Trend NONAKTIF: semua arah diizinkan (hati-hati).'}\n"
    )
    zona_text += analisa_adr_exhaustion(frames["D1"], harga_ref)
    zona_text += cari_golden_confluence(frames, harga_ref)
    zona_text += usd_text
    zona_text += smt_text
    zona_text += cek_kalender_berita_usd()
    zona_text += posisi_text
    zona_text += analisa_akun_dan_lot(acc_info, sym_info, atr_m5_val, harga_ref)

    if SERVER_UTC_OFFSET_JAM is not None:
        offset = SERVER_UTC_OFFSET_JAM
        if not is_refresh:
            print(f"   Offset server broker: UTC{offset:+d} (manual)")
    else:
        offset = deteksi_offset(tick, frames.get("M5"))
        if offset is None:
            offset = 0
            if not is_refresh:
                print("   PERINGATAN: offset server broker tidak bisa dideteksi (market tutup?). Dipakai UTC+0.")
                print("   Jika jam candle MT5 Anda beda dari UTC, isi SERVER_UTC_OFFSET_JAM di bagian atas script.")
        else:
            if not is_refresh:
                print(f"   Offset server broker terdeteksi: UTC{offset:+d}")

    smc_text = "".join(analisa_metode(n, frames[n], offset, harga_ref) for n in ("H4", "H1", "M30", "M5"))
    smc_text += analisa_sesi(frames["M30"], frames["M5"], offset)

    # Hitung kandidat zona algoritmik siap pakai (BUY & SELL) dari data MT5 agar AI punya jangkar angka presisi
    z_buy_k, z_buy_c2 = _cari_zona_valid_berikutnya_mt5("BUY", harga_ref, harga_ref, atr_m5_val)
    z_sell_k, z_sell_c2 = _cari_zona_valid_berikutnya_mt5("SELL", harga_ref, harga_ref, atr_m5_val)
    kandidat_algo_txt = "\n===== KANDIDAT SETUP ALGORITMIK MT5 SIAP PAKAI (JANGKAR PRESISI ANTI-HALUSINASI) =====\n"
    kandidat_algo_txt += "  (Gunakan angka di bawah ini sebagai acuan utama ZONA_BAWAH / ZONA_ATAS dan SINYAL_AKTIF agar 100% akurat sesuai struktur MT5 & lolos gerbang jarak <= 20 poin):\n"
    if z_buy_k:
        kandidat_algo_txt += (
            f"  * KANDIDAT BUY #1 (Terdekat): ENTRY {z_buy_k['lo']:.2f}-{z_buy_k['hi']:.2f} | SL {z_buy_k['sl']:.2f} | "
            f"TP1 {z_buy_k['tp1']:.2f} | TP2 {z_buy_k['tp2']:.2f} (Jarak {harga_ref - z_buy_k['hi']:.2f} pt | {z_buy_k['sumber']})\n"
        )
    if z_buy_c2:
        kandidat_algo_txt += (
            f"  * KANDIDAT BUY #2 (Cadangan): ENTRY {z_buy_c2['lo']:.2f}-{z_buy_c2['hi']:.2f} | SL {z_buy_c2['sl']:.2f} | "
            f"TP1 {z_buy_c2['tp1']:.2f} | TP2 {z_buy_c2['tp2']:.2f} (Jarak {harga_ref - z_buy_c2['hi']:.2f} pt | {z_buy_c2['sumber']})\n"
        )
    if z_sell_k:
        kandidat_algo_txt += (
            f"  * KANDIDAT SELL #1 (Terdekat): ENTRY {z_sell_k['lo']:.2f}-{z_sell_k['hi']:.2f} | SL {z_sell_k['sl']:.2f} | "
            f"TP1 {z_sell_k['tp1']:.2f} | TP2 {z_sell_k['tp2']:.2f} (Jarak {z_sell_k['lo'] - harga_ref:.2f} pt | {z_sell_k['sumber']})\n"
        )
    if z_sell_c2:
        kandidat_algo_txt += (
            f"  * KANDIDAT SELL #2 (Cadangan): ENTRY {z_sell_c2['lo']:.2f}-{z_sell_c2['hi']:.2f} | SL {z_sell_c2['sl']:.2f} | "
            f"TP1 {z_sell_c2['tp1']:.2f} | TP2 {z_sell_c2['tp2']:.2f} (Jarak {z_sell_c2['lo'] - harga_ref:.2f} pt | {z_sell_c2['sumber']})\n"
        )
    zona_text += kandidat_algo_txt

    data_text = (ringkas_tf("W1", frames["W1"], 6)
                 + ringkas_tf("D1", frames["D1"], 10)
                 + ringkas_tf("H4", frames["H4"], 12)
                 + ringkas_tf("H1", frames["H1"], 15)
                 + ringkas_tf("M30", frames["M30"], 20)
                 + ringkas_tf("M5", frames["M5"], 40))

    return f"""
Kamu adalah analis teknikal emas ({SYMBOL}) institusional yang SANGAT TELITI, DISIPLIN, dan ANTI-HALUSINASI.
Gunakan 100% HANYA fakta angka dari data MT5 di bawah ini (waktu = waktu server broker).
Semua candle di tabel sudah CLOSE (valid). Swing high/low, Order Block, FVG, dan Golden Confluence sudah dihitung secara matematis oleh Python dari data real-time MT5 — GUNAKAN ANGKA TERSEBUT, DILARANG KERAS MENGARANG ANGKA SENDIRI!
Harga saat ini: {harga_sekarang}

🛡️ [ATURAN KERAS ANTI-HALUSINASI & AUDIT AKUN REAL MT5 (WAJIB DITAATI MUTLAK)]:
1. DILARANG MENGARANG LEVEL HARGA / INDIKATOR: Setiap angka Support/Resistance, Order Block, FVG, High/Low candle, EMA, RSI, ATR, maupun Balance/Equity/Lot yang kamu sebutkan WAJIB ada secara eksplisit di dalam data di bawah. Jangan pernah menebak atau mengira-ngira angka di luar data!
2. BACA MATA UANG & POSISI AKUN REAL DENGAN TELITI:
   - Cek bagian "FAKTA AKUN REAL MT5, POSISI RUNNING & PENDING ORDER". Jika mata uang akun tertulis `IDC` (atau selain USD), WAJIB tulis `IDC` (JANGAN PERNAH menulis simbol `$` atau `USD` untuk nilai saldo/profit `IDC`!).
   - Jika ada POSISI RUNNING (misal SELL atau BUY yang sedang terbuka), sebutkan nomor Ticket, lot, harga open, SL, dan profit poinnya secara presisi!
   - TELITI MEMBACA STATUS SL BEP+: Pada posisi SELL, jika angka SL < Harga Open (misal Open 4130.88, SL 4130.55), itu berarti SL SUDAH DIKUNCI DI AREA PROFIT (BEP+)! Pada posisi BUY, jika angka SL > Harga Open, itu juga berarti SL SUDAH DIKUNCI DI BEP+! Jangan menyuruh trader "segera geser SL ke BEP" jika di data sudah tertulis `[✅ SUDAH DIKUNCI BEP+]`, melainkan apresiasi dan berikan panduan target TP / trailing SL selanjutnya!
3. LOGIKA TP1 PADA ORDER LIMIT:
   - Jika memberikan sinyal BUY LIMIT di bawah harga sekarang, pastikan TP1 berada di atas area entry dengan R:R >= 1:1.5.
   - Jika memberikan sinyal SELL LIMIT di atas harga sekarang, pastikan TP1 berada di bawah area entry dengan R:R >= 1:1.5.
4. KONSISTENSI 100% ANTARA TEKS DAN [RADAR_TARGET]: Angka Area Entry, SL, TP1, dan TP2 yang kamu tulis di bagian penjelasan sinyal WAJIB SAMA PERSIS 100% sampai ke desimalnya dengan yang kamu tulis di blok `[RADAR_TARGET]` paling bawah!

{zona_text}

{smc_text}

{data_text}

ATURAN SELEKSI ZONA (LEBIH PENTING DARIPADA JUMLAH ZONA):
- METODE INTI: Market Structure (BOS/CHoCH), Liquidity Sweep, dan Order Block.
- FVG hanya konfirmasi zona: wajib >= 0.30 ATR, belum terisi penuh, dan harus overlap secara harga dengan OB fresh H4/H1.
- Trigger entry: M5 wajib menunjukkan Sweep lalu CHoCH/BOS atau CISD yang searah. Rejection candle saja tidak cukup untuk setup A+.
- Supply & Demand terpisah, Breaker Block, Fibonacci/OTE, Volume Profile/POC/VAH/VAL, dan swing historis TIDAK BOLEH membentuk zona, menambah skor zona, atau menjadi alasan entry.
- EMA, RSI, VWAP, SMT, EQH/EQL hanya konteks/filter pendukung; tidak boleh membuat zona sendiri dan tidak boleh mengalahkan struktur H4/H1.
- DILARANG melonggarkan ambang FVG, memasukkan zona mitigated/OB tested, memakai fallback historis/psikologis, atau membuat zona pengganti saat tidak ada kandidat.
- Jika tidak ada OB fresh + FVG untouched yang overlap pada H4/H1, status zona = TIDAK ADA ZONA VALID dan keputusan = WAIT.
- OB dan FVG dari impuls yang sama dihitung sebagai satu setup OB-terkonfirmasi-FVG, bukan dua konfirmasi independen.

METODE TOP-DOWN (ikuti urutan ini):
0. DETEKSI LONJAKAN MOMENTUM SATU ARAH & FASE SIDEWAYS-REVERSAL (WAJIB DICEK PERTAMA!):
   - Cek bagian "[Order Flow Institusi & Sideways Reversal]" pada M30 dan M5.
   - Jika tertulis "🚨 PERINGATAN LONJAKAN MOMENTUM (TICK-VOLUME, BUKAN DATA INSTITUSI) (ONE-WAY IMPULSE AKTIF)", maka DILARANG KERAS memberi sinyal lawan arah (Counter-Trend) karena level teknikal kecil pasti jebol terus! Hanya boleh ikuti arah impuls saat pullback ke FVG/OB searah, atau beri rekomendasi WAIT sampai terbentuk kotak Sideways/Ranging.
   - Jika pasar baru selesai impuls lalu membentuk "Kotak Ranging / Sideways Pasca-Impuls", cari setup PEMBALIKAN HARGA (Reversal) berkualitas tinggi: yaitu saat harga melakukan Sweep/Manipulasi di batas atas/bawah kotak sideways lalu masuk kembali disertai CHoCH!
1. ZONA UTAMA: gunakan hanya RADAR ZONA SELEKTIF (OB + FVG) dari H4/H1. Swing W1/D1 hanya referensi likuiditas, bukan zona entry. Prioritaskan A+ lintas H4/H1; jika hanya A satu timeframe, tunggu konfirmasi lebih kuat.
2. H4 -> bias besar (Market Structure BOS/CHoCH, EMA sebagai filter, korelasi USD) dan pilih zona paling relevan. Sebutkan rentang harga, asal zona, dan waktu terbentuknya.
3. H1 -> perhalus zona OB fresh + FVG untouched yang overlap. OB tested atau breaker tidak boleh menjadi zona aktif. Cek CISD, EQH/EQL, RSI Divergence, dan VWAP sebagai konteks.
4. M30 -> konfirmasi konteks dan reaksi harga di zona H4/H1. M30 tidak boleh menciptakan zona entry baru. Periksa sweep, CHoCH/BOS, CISD, retest, EMA/RSI/VWAP.
5. M5 -> ENTRY presisi: zona harus sudah tervalidasi dari H4/H1. Tunggu Liquidity Sweep M5 lalu CHoCH/BOS atau CISD M5 yang close dan searah. Sweep tanpa perubahan struktur/delivery bukan trigger; CHoCH/CISD tanpa berada di zona valid juga bukan trigger. Jika belum lengkap, WAIT.
6. SESI, BERITA & MAKRO: nilai apakah waktu sekarang berada di Kill Zone, apakah ADR harian sudah Exhausted (bensin habis), apakah ada SMT Divergence XAUUSD vs XAGUSD, dan apakah ada rilis berita High-Impact USD sebentar lagi.

SISTEM 4 PILAR KONFLUENSI (Evaluasi tiap setup wajib berdasarkan 4 pilar ini):

PILAR 1: HTF BIAS & STRUKTUR TREN (W1 / D1 / H4 / H1) — WAJIB SEARAH TREND
ATURAN KERAS: Lihat bagian "BIAS TREND HTF" di atas. Jika bias BULLISH → HANYA kasih setup BUY (atau WAIT). Jika bias BEARISH → HANYA kasih setup SELL (atau WAIT). Counter-trend DILARANG. Reversal hanya boleh dipertimbangkan setelah CHoCH H4 dan H1 terkonfirmasi candle-close serta zona OB+FVG baru terbentuk. Pullback/retest ke zona DEMAND (bias bullish) atau SUPPLY (bias bearish) adalah entry ideal — BUKAN melawan trend.
- Tren Struktur Pasar: Apakah BOS/CHoCH H4 & H1 searah dengan sinyal (HH-HL untuk Bullish, LH-LL untuk Bearish)?
- Valuasi & Moving Average: Apakah struktur H4/H1 searah? EMA200 dan Daily VWAP hanya filter pendukung; Fibonacci tidak digunakan untuk membentuk zona.
- Dukungan Makro & Korelasi: Korelasi USD / DXY (DXY melemah mendukung Gold naik, DXY menguat menekan Gold) dan RSI Divergence di HTF.
- Syarat Lolos Pilar 1: Sinyal harus searah tren HTF ATAU berupa skenario Pullback/Retest terukur di level struktural kunci.

PILAR 2: INSTITUTIONAL FOOTPRINT & AREA OF VALUE (ZONA UTAMA)
- Keberadaan Zona Institusional: Harga berada di dalam atau sangat dekat dengan Zona OB fresh yang overlap dengan FVG untouched pada H4/H1. FVG tanpa OB bukan zona entry.
- Level psikologis hanya konteks tambahan; tidak boleh membentuk zona atau menggantikan syarat OB+FVG.
- Syarat Lolos Pilar 2: Setup HARUS berpijak pada Zona Institusional nyata. DILARANG KERAS entry di tengah range kosong ("no man's land") tanpa zona penahan.

PILAR 3: LTF CONFIRMATION TRIGGER (M5 / M30 EKSEKUSI PRESISI)
- Pemicu Validasi Dini (Trigger Entry):
  a) Liquidity Sweep: Ujung wick menyapu swing high/low sebelumnya atau EQH/EQL, lalu candle close kembali ke dalam range.
  b) CISD (Change in State of Delivery): Candle M5 close menembus level Open rangkaian candle pengiriman sebelumnya (konfirmasi reversal dini ICT).
  c) CHoCH / BOS M5: Terbentuk karakter baru atau pola candle rejection kuat (Pin Bar / Hammer / Engulfing) di batas zona / kotak sideways pasca-impuls.
- Syarat Lolos Pilar 3: wajib ada Sweep M5 dan konfirmasi lanjutan CHoCH/BOS atau CISD M5 yang sudah candle-close dan searah. Satu sinyal tunggal tanpa rangkaian konfirmasi tidak cukup.

PILAR 4: KONTEKS LINGKUNGAN PASAR & MANAJEMEN RISIKO
- Filter Impulse Institusi: TIDAK MELAWAN Institutional One-Way Impulse aktif. Jika ada impuls liar satu arah, hanya boleh ikut arah impuls saat pullback atau WAIT.
- Volatilitas & Bensin Harian: ADR (Average Daily Range) belum exhausted (masih ada ruang gerak harian), atau setup mengantisipasi reversal di batas ekstrem ADR.
- Waktu & Makro Event: Berada di dalam/menjelang Sesi Aktif (London/NY Kill Zone), tidak ada rilis High-Impact News USD dalam 30 menit ke depan, dan SMT Divergence (XAU vs XAG) mendukung atau netral.
- Rasio Risk-to-Reward: R:R terukur minimal 1:1.5 hingga 1:2+ dengan jarak SL logis di balik batas struktur/wick.

OUTPUT:
A. Status Order Flow Institusi, ADR Harian, Radar Berita USD, Evaluasi Posisi Terbuka & Pending Order (jika user punya posisi floating di MT5, beri saran tegas berdasarkan status SL sebenarnya: apakah SL sudah BEP+ [PERTAHANKAN / TRAILING PROFIT], belum BEP [GESER SL KE BEP], atau melanggar struktur [CUT LOSS]; jika user sedang memasang Pending Order seperti BUY LIMIT / SELL LIMIT di MT5, WAJIB evaluasi satu per satu nomor tiketnya apakah zona limit tersebut MASIH VALID [PERTAHANKAN], PERLU DIGESER [ADJUST LEVEL], atau SUDAH TIDAK VALID / TERLALU JAUH / TANPA SL [SEGERA PASANG SL / CANCEL] beserta alasan struktur/likuiditasnya), serta Tabel Ringkasan Zona Utama & Golden Confluence.
B. Daftar sinyal. HANYA tampilkan sinyal berlabel LOW RISK atau MEDIUM RISK. DILARANG menampilkan sinyal HIGH RISK atau sinyal yang melawan Institutional Impulse aktif!

   ══════════════════════════════════════════════════════════════════════
   PROTOKOL STANDAR AKURASI TINGGI SINYAL XAUUSD (WAJIB DIPATUHI):
   ══════════════════════════════════════════════════════════════════════
   1. TIPE ORDER & ANTI-FOMO (Dilarang Kejar Harga Saat Candle Running):
      - Wajib tentukan tipe eksekusi:
        * LIMIT / PULLBACK ENTRY (Rekomendasi Utama Akurasi): Antri di area FVG / OB / 50% Equilibrium Wick Sweep saat harga retrace.
        * CONFIRMED BREAK ENTRY: Masuk HANYA jika candle M5 resmi CLOSE menembus level trigger tertentu. Dilarang menyuruh entry saat candle masih berjalan/floating!
      - Rentang area entry dibatasi ketat maksimal 1.5 - 2.0 poin (contoh lebar area: 1.6 pt). Jangan berikan rentang lebar yang membingungkan.
   2. ATURAN STOP LOSS ANTI-HUNT & BUFFER SPREAD:
      - DILARANG meletakkan SL mepet di pucuk/lembah wick tanpa buffer (risiko tersapu spread atau secondary wick hunt).
      - Formula SL BUY: Di bawah Swing Low terendah / demand base dikurangi buffer aman (minimal 1.5 - 2.5 poin atau 1x ATR M5).
      - Formula SL SELL: Di atas Swing High tertinggi / supply base ditambah buffer aman (minimal 1.5 - 2.5 poin atau 1x ATR M5) + toleransi spread broker.
      - Sebutkan level struktural perlindungan SL (misal: "SL di <level>, di balik ujung wick sweep <level> + buffer 1.7 pt").
   3. TARGET TP OBJEKTIF & PENGAMANAN MODAL DUA TAHAP:
      - TP1 (Likuiditas Internal): Di opposing FVG / Swing M15 terdekat dengan R:R minimal 1:1.5.
        Instruksi WAJIB: "Saat harga menyentuh TP1 -> Amankan 50% profit (Partial Close) & Geser SL ke BEP+ (Titik Impas)!"
      - TP2 (Likuiditas Eksternal): Di Golden Confluence Zone lawan / batas ekstrem ADR / level likuiditas HTF (R:R 1:2.5 s/d 1:4+).
      - FILTER MATEMATIKA R:R: Jika jarak Entry ke TP1 lebih kecil dari jarak ke SL (R:R < 1:1.5), sinyal GUGUR / DILARANG TAMPIL.
   4. KONDISI INVALIDASI EKSPLISIT & NOTE ANTI "KENA KENTUT" (SHALLOW TAP LIMIT):
      - Wajib sebutkan level pasti pembatalan bawah/atas (SL) DAN aturan jika harga hanya menyentuh tepi zona lalu terbang tanpa menjemput Limit di ujung dalam ("kena kentut"):
        * Sarankan pasang Limit di tepi depan / 50% tengah zona (Equilibrium) + buffer spread (atau bagi 2 peluru), JANGAN taruh 100% di ujung paling dalam zona!
        * Jika harga sudah menyentuh tepi zona (misal zona BUY 4107-4110 tapi harga cuma turun ke 4108 lalu naik): selama kenaikan belum melewati 60% jarak ke TP1, Limit di 4107 MASIH VALID ditunggu untuk retest kedua. TETAPI jika harga sudah terbang >= 60% jarak ke TP1 (atau mencapai TP1), WAJIB CANCEL Limit karena zona sudah mitigated (habis bensin & rawan jebol jika balik lagi).
   5. UKURAN LOT & MANAJEMEN RISIKO MT5 REALISTIS:
      - Ambil langsung rekomendasi ukuran lot dan mata uang akun (`IDC` / `USD`) dari bagian "MANAJEMEN RISIKO & KALKULATOR LOT OTOMATIS" sesuai jarak SL dan equity akun MT5 (1% untuk Low Risk, 0.5% untuk Medium Risk).

   FORMAT SETIAP SINYAL WAJIB DITULIS RAPI SEPERTI INI:
   ----------------------------------------------------------------------
   [SINYAL <LOW RISK / MEDIUM RISK>] <BUY / SELL> XAUUSD (<Tipe Order: Limit Pullback / Confirmed Break>)
   • Area Entry   : <bawah> - <atas> (Jarak dari harga sekarang: X pt)
   • Stop Loss    : <angka> (Jarak risiko: X pt / X pips | Alasan: X pt di balik wick/level X)
   • Target TP    :
     - TP1        : <angka> (R:R 1:X | Amankan 50% lot & Geser SL ke BEP+)
     - TP2        : <angka> (R:R 1:X | Runner ke likuiditas eksternal)
   • Manajemen Lot: <X.XX> lot (Maksimum risiko sesuai mata uang akun / X% modal)
   • Pemicu (Trigger) : Syarat candle M5 (level CISD / Sweep / CHoCH spesifik)
   • Note Limit (Anti Kena Kentut) : Pasang di tengah/tepi depan (<angka tengah>-<angka tepi depan>); jika harga cuma colek tepi zona lalu terbang melewati <angka 60% ke TP1> / TP1 tanpa jemput limit -> WAJIB CANCEL LIMIT (zona sudah mitigated)!
   • Invalidasi   : Setup BATAL jika candle M5 close melewati <angka level> sebelum terjemput
   • TABEL 4 PILAR KONFLUENSI:
     | Pilar | Status | Bukti Data / Alasan Singkat |
     | Pilar 1: HTF Bias & Struktur        | [✔ / ✘ / ➖] | (Struktur H4/H1, EMA filter, USD) |
     | Pilar 2: Area of Value (Zona Kunci) | [✔ / ✘ / ➖] | (OB fresh + FVG untouched yang overlap pada H4/H1) |
     | Pilar 3: LTF Confirmation Trigger   | [✔ / ✘ / ➖] | (Sweep M5, CISD M5, CHoCH, Rejection) |
     | Pilar 4: Konteks Pasar & Risiko     | [✔ / ✘ / ➖] | (Sesi Kill Zone, ADR, Berita USD, R:R) |
     Total Skor: X / 4 Pilar
   ----------------------------------------------------------------------

   Kriteria Klasifikasi Risiko:
   - LOW RISK (Swing / Setup Utama A+):
     * Skor: 4/4 Pilar Terpenuhi.
     * Berpijak pada zona OB fresh + FVG untouched yang overlap di H4/H1, ada Sweep M5 + CISD/CHoCH searah, R:R minimal 1:2.
     * Alokasi risiko lot 1.0% modal.
   - MEDIUM RISK (Intraday / Scalping Reaksi Zona Terdekat B+):
     * Skor: minimal 3/4 Pilar, tetapi Pilar 2 (zona OB+FVG valid) dan Pilar 3 (trigger M5) wajib terpenuhi.
     * Jika harga SAAT INI sedang menyentuh Zona zona OB+FVG terdekat (H1/M30/H4) dan di M5 sudah muncul Liquidity Sweep, CISD, CHoCH, atau pola rejection (Pin Bar/Engulfing) — selama TIDAK ada Institutional Impulse ekstrem lawan arah — berikan rencana eksekusi MEDIUM RISK di zona terdekat tersebut jika tetap lolos 3/4 Pilar dan R:R >= 1:1.5 (jangan hanya menyuruh tunggu di zona yang jauh, tetapi jangan memaksa entry bila syarat tidak terpenuhi). R:R minimal 1:1.5.
     * Alokasi risiko lot 0.5% modal.
   - HIGH RISK (Kurang dari 3 pilar, melawan Institutional Impulse aktif, di tengah range kosong tanpa zona, atau R:R di bawah 1:1.5): DILARANG ditampilkan.
C. Jika tidak ada sinyal yang memenuhi minimal 3/4 Pilar, tulis jelas "WAIT: belum ada setup LOW/MEDIUM RISK", jelaskan pilar mana yang belum terpenuhi,
   dan sebutkan level/kondisi apa yang perlu ditunggu (Golden Zone mana, sweep/CHoCH apa, jam kill zone berikutnya) sebelum entry.
D. Di baris paling akhir responsmu, WAJIB tuliskan blok ringkasan target pantau otomatis dengan format persis seperti ini:
[RADAR_TARGET]
ZONA_ATAS: <bawah>-<atas> (batas_jebol: <angka_di_atas_zona>)
ZONA_BAWAH: <bawah>-<atas> (batas_jebol: <angka_di_bawah_zona>)
SINYAL_AKTIF: <BUY/SELL/WAIT> | ENTRY: <bawah>-<atas> | SL: <angka> | TP1: <angka> | TP2: <angka>
[/RADAR_TARGET]

ATURAN KETAT ZONA RADAR & SINYAL (WAJIB INTRADAY REALISTIS & KONSISTEN 100%):
1. ZONA_BAWAH: Wajib pilih Zona Demand TERDEKAT di bawah harga saat ini (M30/H1/H4/M5) dengan jarak realistis 5 - 20 poin dari harga sekarang! DILARANG KERAS mengambil zona ekstrem ratusan poin yang jauh dari harga saat ini (zona D1/W1 yang jauh bukan target radar intraday)! Jika Anda menerbitkan SINYAL_AKTIF: BUY, maka angka ZONA_BAWAH wajib SAMA PERSIS dengan angka ENTRY BUY (dan batas_jebol = SL BUY) agar tidak ada dua angka berbeda!
2. ZONA_ATAS: Wajib pilih Zona Supply TERDEKAT di atas harga saat ini (M30/H1/H4/M5) dengan jarak realistis 5 - 20 poin dari harga sekarang! Jika Anda menerbitkan SINYAL_AKTIF: SELL, maka angka ZONA_ATAS wajib SAMA PERSIS dengan angka ENTRY SELL (dan batas_jebol = SL SELL)!
3. SINYAL_AKTIF:
   - JIKA harga di dekat zona pantau DAN ada konfirmasi LTF M5 (CHoCH/CISD/Sweep/Rejection) DAN setup lolos minimal 3/4 Pilar serta R:R >= 1:1.5, berikan sinyal eksekusi (BUY atau SELL) dengan Entry ketat (maks 2 pt), SL terukur di balik wick, dan TP realistis.
   - Jika syarat di atas TIDAK terpenuhi, tulis SINYAL_AKTIF: WAIT. Lebih baik tidak entry daripada memaksa setup. Sinyal dengan R:R < 1:1.5 atau SL di sisi yang salah akan ditolak otomatis oleh sistem.
4. KONSISTENSI MUTLAK & ANTI-PLIN-PLAN (DILARANG MENGGESER ENTRY/SL ATAU FLIP-FLOP):
   - Jika pada data historis/sesi sebelumnya SUDAH ada Sinyal Aktif (misal BUY) dan sinyal tersebut MASIH VALID (harga belum tembus SL), Anda DILARANG KERAS:
     (x) Membalikkan sinyal menjadi arah berlawanan (misal tiba-tiba memberi SELL), ATAU
     (x) Menggeser-geser angka Area Entry / SL ke harga lain setiap kali ditanya ulang!
   - Mengubah angka entry dari pertanyaan pertama ke pertanyaan kedua (padahal SL belum jebol) membuat sinyal terlihat tidak valid dan membingungkan trader.
   - Sinyal baru atau perubahan angka Entry/SL HANYA boleh diterbitkan JIKA:
     a) Sinyal lama SUDAH TERBUKTI BATAL/JEBOL (harga menyentuh/menembus batas SL), ATAU
     b) Target penuh TP2 sudah tersapu bersih, ATAU sinyal lama sudah KEDALUWARSA.
   - Selama sinyal lama masih valid, pertahankan angka Entry, SL, dan TP yang SAMA PERSIS 100%!
Aturan: jangan mengarang angka di luar data, jangan memaksa entry. Ini bukan nasihat finansial.
"""


def _normalisasi_radar(teks):
    """Ambil blok [RADAR_TARGET] terakhir (jika ada), buang markdown (*, `) dan pemisah ribuan (4,031.50 -> 4031.50)."""
    blok = re.findall(r"\[RADAR_TARGET\](.*?)\[/RADAR_TARGET\]", teks, re.S | re.I)
    if blok:
        teks = blok[-1]
    teks = teks.replace("*", "").replace("`", "")
    return re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", teks)


def _ekstrak_sinyal_dari_narasi(teks_penuh):
    """Fallback parser: jika AI menuliskan sinyal di badan narasi (misal saat Q&A) tetapi lupa menyertakan atau
    tidak sinkron dengan [RADAR_TARGET], ekstrak angka BUY/SELL, Area Entry, SL, TP1, TP2 dari narasi."""
    if not teks_penuh:
        return None
    bersih = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", teks_penuh.replace("*", "").replace("`", ""))
    # Cari header [SINYAL ...] BUY/SELL atau rekomendasi eksplisit
    m_hdr = list(re.finditer(r"\[SINYAL[^\]]*\]\s*(BUY|SELL)\b", bersih, re.I))
    segmen = bersih[m_hdr[-1].start():] if m_hdr else bersih
    arah = m_hdr[-1].group(1).upper() if m_hdr else None
    if not arah:
        m_arah = re.search(r"(?:Setup|Sinyal|Rekomendasi|Order)\s*(?:Utama|Aktif|Baru)?\s*[:\-]?\s*(BUY|SELL)\s*(?:LIMIT|STOP|XAUUSD)?", segmen, re.I)
        if m_arah:
            arah = m_arah.group(1).upper()
    if not arah:
        return None
    m_entry = re.search(r"(?:Area\s*Entry|Entry\s*Zone|Entry)\s*[:\-]\s*(\d{4}(?:\.\d+)?)\s*(?:-\s*|\s+s/d\s+|\s+sampai\s+)(\d{4}(?:\.\d+)?)", segmen, re.I)
    if not m_entry:
        return None
    m_sl = re.search(r"(?:Stop\s*Loss|SL)\s*[:\-]\s*(\d{4}(?:\.\d+)?)", segmen, re.I)
    m_tp1 = re.search(r"TP\s*1\s*[:\-]\s*(\d{4}(?:\.\d+)?)", segmen, re.I)
    m_tp2 = re.search(r"TP\s*2\s*[:\-]\s*(\d{4}(?:\.\d+)?)", segmen, re.I)
    if not (m_sl and m_tp1):
        return None
    e1, e2 = float(m_entry.group(1)), float(m_entry.group(2))
    sl = float(m_sl.group(1))
    tp1 = float(m_tp1.group(1))
    tp2 = float(m_tp2.group(1)) if m_tp2 else tp1
    return dict(
        arah=arah,
        lo=round(min(e1, e2), 2),
        hi=round(max(e1, e2), 2),
        sl=sl,
        tp1=tp1,
        tp2=tp2,
        lo_asli=round(min(e1, e2), 2),
        hi_asli=round(max(e1, e2), 2),
    )



def _validasi_trigger_m5(sig):
    """Gerbang kode: M5 wajib punya Sweep lalu CHoCH/BOS atau CISD searah.

    Selain itu harga harus masih dekat dengan area entry agar AI tidak mengejar
    harga setelah trigger bergerak terlalu jauh. Gagal mengambil data = ditolak.
    """
    try:
        if not pastikan_mt5_aktif():
            return False, "MT5 tidak aktif; trigger M5 tidak dapat diverifikasi"
        df = ambil_data(mt5.TIMEFRAME_M5, 120)
        if df is None or len(df) < 60:
            return False, "Data M5 tidak cukup untuk memverifikasi trigger"
        d = tambah_indikator(df.copy())
        events, _, sw = struktur(d)
        sweeps = liquidity_sweeps(d, sw, SWING_N, lookback=20)
        cisd_list, _ = deteksi_cisd(d, sw, lookback=20)
        arah = str(sig.get("arah", "")).upper()
        arah_struktur = "bull" if arah == "BUY" else "bear"
        arah_cisd = "BULLISH" if arah == "BUY" else "BEARISH"
        jenis_sweep = "SELL-SIDE" if arah == "BUY" else "BUY-SIDE"
        batas_idx = len(d) - 7
        trigger_valid = False
        bukti = ""
        for waktu, ket, *_ in reversed(sweeps):
            if jenis_sweep not in ket.upper():
                continue
            try:
                idx_sweep = int(d.index.get_loc(waktu))
            except Exception:
                continue
            if idx_sweep < batas_idx:
                continue
            konfirmasi_struktur = next((e for e in reversed(events)
                if e["arah"] == arah_struktur and idx_sweep < e["k"] < len(d)), None)
            konfirmasi_cisd = next((c for c in reversed(cisd_list)
                if c.get("arah") == arah_cisd and idx_sweep < c.get("k", -1) < len(d)), None)
            if konfirmasi_struktur or konfirmasi_cisd:
                bukti = "CHoCH/BOS" if konfirmasi_struktur else "CISD"
                trigger_valid = True
                break
        if not trigger_valid:
            return False, f"Trigger M5 belum lengkap: wajib {jenis_sweep} Sweep lalu CHoCH/BOS atau CISD {arah_cisd}"

        harga = float(d["Close"].iloc[-1])
        atr = float(d["ATR14"].iloc[-1])
        lo, hi = float(sig["lo"]), float(sig["hi"])
        jarak = 0.0 if lo <= harga <= hi else min(abs(harga - lo), abs(harga - hi))
        batas_jarak = max(3.0, 1.5 * atr)
        if jarak > batas_jarak:
            return False, f"Trigger M5 sudah ada ({bukti}), tetapi harga terlalu jauh dari zona entry ({jarak:.2f} pt > {batas_jarak:.2f} pt); jangan kejar harga"
        return True, f"M5 valid: {jenis_sweep} Sweep lalu {bukti} searah; harga dekat zona"
    except Exception as exc:
        _catat_error("validasi_trigger_m5", exc)
        return False, f"Trigger M5 gagal diverifikasi ({type(exc).__name__}); fail-safe WAIT"

def _terapkan_otoritas_scanner(target):
    """Python Zone Scanner = satu-satunya sumber kebenaran teknikal (Single Source of Truth).

    Aturan Mutlak:
      1. Zona entry (ZONA_ATAS / ZONA_BAWAH) HANYA berasal dari Python Scanner (best_supply & best_demand).
      2. Sinyal trading dihasilkan langsung dari evaluasi mandiri Python Scanner:
         - Jika Python Scanner berstatus VALID (BUY/SELL): sinyal aktif dikunci ke plan Python.
         - Jika Python Scanner berstatus WAIT atau INVALID: sinyal aktif bernilai None (WAIT).
      3. AI Gemini DILARANG membuat atau mengganti angka entry, SL, TP, atau zona secara bebas.
      4. Tidak ada proses lanjutan atau pending order yang boleh sembarangan menimpa keputusan scanner.
    """
    try:
        dec = evaluate_trade_decision()   # tidak pernah raise; error internal -> status WAIT
        scan = scan_xauusd_zones()
    except Exception as e_scan:
        _catat_error("ekstrak_target_ai_zone_scanner", e_scan)
        target["sinyal"], target["atas"], target["bawah"] = None, None, None
        _STATE_PARSER["ditolak"] = [f"Zone Scanner error ({type(e_scan).__name__}): fail-safe WAIT"]
        return target

    bd, bs = scan.get("best_demand"), scan.get("best_supply")
    target["bawah"] = (float(bd["price_low"]), float(bd["price_high"]), float(bd["sl_recommended"])) if bd else None
    target["atas"] = (float(bs["price_low"]), float(bs["price_high"]), float(bs["sl_recommended"])) if bs else None

    status_dec = dec.get("status", "WAIT")
    plan = dec.get("plan")
    arah_dec = dec.get("direction")

    # Tolak jika terdapat kandidat sinyal berlawanan arah dengan keputusan teknikal scanner
    sig_in = target.get("sinyal")
    if sig_in and isinstance(sig_in, dict) and sig_in.get("arah"):
        arah_in = str(sig_in["arah"]).upper()
        if arah_dec and arah_in != arah_dec:
            target["sinyal"] = None
            _STATE_PARSER["ditolak"] = [f"Ditolak: Arah usulan ({arah_in}) bertentangan dengan arah Python Scanner ({arah_dec})"]
            return target

    # Keputusan mandiri Python Scanner:
    if status_dec == "VALID" and plan and arah_dec:
        sig_python = dict(
            arah=str(plan["arah"]).upper(),
            lo=round(float(plan["entry_min"]), 2),
            hi=round(float(plan["entry_max"]), 2),
            lo_asli=round(float(plan["entry_min"]), 2),
            hi_asli=round(float(plan["entry_max"]), 2),
            sl=float(plan["sl"]),
            tp1=float(plan["tp1"]),
            tp2=float(plan["tp2"]),
            ket=str(plan.get("keterangan", "Python Scanner Valid"))[:100],
            tipe_strategi="PYTHON SCANNER VALID",
        )
        ok, alasan_v = validasi_sinyal_struktur(sig_python)
        if ok:
            target["sinyal"] = sig_python
            _STATE_PARSER["ditolak"] = []
        else:
            target["sinyal"] = None
            _STATE_PARSER["ditolak"] = ["Rencana Python gagal validasi struktur: " + "; ".join(alasan_v)]
    else:
        alasan = "; ".join(dec.get("reasons") or []) or f"status scanner {status_dec}"
        target["sinyal"] = None
        _STATE_PARSER["ditolak"] = [f"Ditolak Python Scanner ({status_dec}, arah {arah_dec}): {alasan}"]

    return target


def ekstrak_target_ai(teks_ai):
    """Membaca target dan sinyal dengan Python Zone Scanner sebagai satu-satunya sumber kebenaran teknikal.
    AI Gemini hanya memberikan narasi dan analisis tambahan, tanpa otoritas mengubah angka teknikal."""
    target = {"atas": None, "bawah": None, "sinyal": None}
    _STATE_PARSER["ditolak"] = []

    # 🛡️ OTORITAS MUTLAK PYTHON SCANNER:
    # Mengambil zona dan keputusan entry mandiri dari Python
    _terapkan_otoritas_scanner(target)
    return target

    return target


def _coba_satu_model(client, model_name, contents):
    t0 = time.time()
    res = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=types.GenerateContentConfig(temperature=0.0),
    )
    return res.text, model_name, time.time() - t0


def snapshot_live_qa():
    """Mengambil harga Bid/Ask detik ini + update kilat M5 & M30 + Akun & Zona MT5 setiap kali user bertanya di Q&A,
    supaya AI selalu tahu posisi akun real, zona valid MT5 terdekat, serta struktur M5 terbaru tanpa berhalusinasi!"""
    if not pastikan_mt5_aktif():
        return ""
    tick = mt5.symbol_info_tick(SYMBOL)
    df_m5 = ambil_data(mt5.TIMEFRAME_M5, 120)
    df_m30 = ambil_data(mt5.TIMEFRAME_M30, 120)
    pos_txt = cek_posisi_terbuka_mt5(tick.bid if tick else 0.0)

    if df_m5 is None:
        return ""

    harga = tick.bid if tick else df_m5["Close"].iloc[-1]
    spread = (tick.ask - tick.bid) if tick else 0.0
    d5 = tambah_indikator(df_m5.copy())
    atr5 = d5["ATR14"].values
    evs5, tr5, sw5 = struktur(d5)
    cisd5, calon_cisd5 = deteksi_cisd(d5, sw5, lookback=15)
    sweeps5 = liquidity_sweeps(d5, sw5, SWING_N, lookback=10)
    pola5 = deteksi_pola(d5)[-3:]
    inst5 = deteksi_institusi_dan_ranging(d5, atr5)
    kol = ["Open", "High", "Low", "Close", "EMA20", "RSI14", "ATR14"]
    tabel5 = d5[kol].tail(10).round(2).to_string()

    ev_txt = "; ".join(f"{e['tipe']} {e['arah']} @ {e['level']:.2f} ({wkt(d5.index[e['k']])})" for e in evs5[-3:]) or "-"
    cisd_txt = "; ".join(f"{wkt(c['waktu'])}: {c['ket']}" for c in cisd5[-2:]) or "belum ada"
    sw_txt = "; ".join(f"{wkt(t)}: {ket} level {p:.2f} (wick {wk:.2f}, close {cl:.2f})" for t, ket, p, wk, cl in sweeps5[-3:]) or "tidak ada"

    m30_info = ""
    if df_m30 is not None:
        d30 = tambah_indikator(df_m30.copy())
        evs30, _, sw30 = struktur(d30)
        cisd30, _ = deteksi_cisd(d30, sw30, lookback=12)
        swp30 = liquidity_sweeps(d30, sw30, SWING_N, lookback=8)
        m30_info = (
            f"Update M30: Close {d30['Close'].iloc[-1]:.2f} | RSI14 {d30['RSI14'].iloc[-1]:.1f} | "
            f"Struktur: {evs30[-1]['tipe'] + ' ' + evs30[-1]['arah'] if evs30 else '-'} | "
            f"CISD M30: {cisd30[-1]['ket'] if cisd30 else 'tidak ada'} | "
            f"Sweep M30: {swp30[-1][1] + ' @ ' + f'{swp30[-1][2]:.2f}' if swp30 else 'tidak ada'}\n"
        )

    # Sertakan high/low candle H4 & H1 yang sedang berjalan (belum close) agar sweep/range tidak tertinggal
    cur_h4 = ambil_candle_berjalan(mt5.TIMEFRAME_H4)
    cur_h1 = ambil_candle_berjalan(mt5.TIMEFRAME_H1)
    htf_running_txt = ""
    if cur_h4 or cur_h1:
        htf_running_txt = "- Candle Berjalan (belum close — untuk konteks sweep/range): "
        if cur_h4:
            htf_running_txt += f"H4 High {float(cur_h4['high']):.2f} Low {float(cur_h4['low']):.2f} | "
        if cur_h1:
            htf_running_txt += f"H1 High {float(cur_h1['high']):.2f} Low {float(cur_h1['low']):.2f}"
        htf_running_txt = htf_running_txt.rstrip(" |") + "\n"

    # Sertakan Zona Demand (BUY) & Supply (SELL) Valid Terdekat hasil kalkulasi algoritmik MT5 (H4/H1/M30/M15)
    # agar di mode Q&A AI TIDAK PERNAH mengarang/menebak angka zona sendiri!
    z_buy_qa, z_buy_cad = _cari_zona_valid_berikutnya_mt5("BUY", harga, harga, float(atr5[-1]))
    z_sell_qa, z_sell_cad = _cari_zona_valid_berikutnya_mt5("SELL", harga, harga, float(atr5[-1]))
    zona_mt5_qa_txt = "- [REFERENSI ZONA TEKNIKAL DARI DATA MT5 REAL-TIME]:\n"
    if z_buy_qa:
        zona_mt5_qa_txt += (
            f"  * KANDIDAT BUY LIMIT UTAMA (DEMAND BAWAH TERDEKAT): ENTRY {z_buy_qa['lo']:.2f}-{z_buy_qa['hi']:.2f} | "
            f"SL {z_buy_qa['sl']:.2f} | TP1 {z_buy_qa['tp1']:.2f} | TP2 {z_buy_qa['tp2']:.2f} "
            f"(Jarak: {harga - z_buy_qa['hi']:.2f} pt di bawah harga sekarang | Sumber: {z_buy_qa['ket']})\n"
        )
    if z_buy_cad:
        zona_mt5_qa_txt += (
            f"  * KANDIDAT BUY LIMIT KE-2 (DEMAND CADANGAN): ENTRY {z_buy_cad['lo']:.2f}-{z_buy_cad['hi']:.2f} | "
            f"SL {z_buy_cad['sl']:.2f} | TP1 {z_buy_cad['tp1']:.2f} | TP2 {z_buy_cad['tp2']:.2f} ({z_buy_cad['ket']})\n"
        )
    if z_sell_qa:
        zona_mt5_qa_txt += (
            f"  * KANDIDAT SELL LIMIT UTAMA (SUPPLY ATAS TERDEKAT): ENTRY {z_sell_qa['lo']:.2f}-{z_sell_qa['hi']:.2f} | "
            f"SL {z_sell_qa['sl']:.2f} | TP1 {z_sell_qa['tp1']:.2f} | TP2 {z_sell_qa['tp2']:.2f} "
            f"(Jarak: {z_sell_qa['lo'] - harga:.2f} pt di atas harga sekarang | Sumber: {z_sell_qa['ket']})\n"
        )
    if z_sell_cad:
        zona_mt5_qa_txt += (
            f"  * KANDIDAT SELL LIMIT KE-2 (SUPPLY CADANGAN): ENTRY {z_sell_cad['lo']:.2f}-{z_sell_cad['hi']:.2f} | "
            f"SL {z_sell_cad['sl']:.2f} | TP1 {z_sell_cad['tp1']:.2f} | TP2 {z_sell_cad['tp2']:.2f} ({z_sell_cad['ket']})\n"
        )

    # Sertakan analisa akun & tabel lot presisi
    acc_qa = mt5.account_info()
    sym_qa = mt5.symbol_info(SYMBOL)
    info_akun_qa = analisa_akun_dan_lot(acc_qa, sym_qa, float(atr5[-1]), harga)

    sig_live_txt = ""
    if SINYAL_AKTIF_DINAMIS and isinstance(SINYAL_AKTIF_DINAMIS, dict) and SINYAL_AKTIF_DINAMIS.get("lo"):
        sd = SINYAL_AKTIF_DINAMIS
        st_sd, al_sd = evaluasi_validitas_sinyal(sd, harga)
        if sinyal_masih_aktif(st_sd):
            lbl_eks_sd, _, info_eks_sd = cek_status_eksekusi_sinyal(sd)
            sig_live_txt = (
                f"- 🔒 SINYAL UTAMA AKTIF (TERKUNCI): "
                f"{sd['arah']} | ENTRY: {sd['lo']:.2f}-{sd['hi']:.2f} | SL: {sd['sl']:.2f} | TP1: {sd['tp1']:.2f} | TP2: {sd.get('tp2', sd['tp1']):.2f} "
                f"-> Status Real-Time: [{st_sd}] ({al_sd}) | Status di Akun MT5: [{lbl_eks_sd}] ({info_eks_sd})\n"
            )
        else:
            sig_live_txt = f"- Status Sinyal Sebelumnya: SUDAH SELESAI/BATAL [{st_sd}]. Status Saat Ini: WAIT (Mencari peluang setup baru).\n"

    scan_qa_txt = ""
    try:
        from modules.zone_scanner import scan_xauusd_zones
        scan_qa = scan_xauusd_zones()
        dec_qa = scan_qa.get("decision", {})
        reasons_qa = "; ".join(dec_qa.get("reasons", [])) or "tidak ada alasan"
        scan_qa_txt = f"- Status Python Zone Scanner: [{dec_qa.get('status', 'WAIT')}] | Bias: {scan_qa.get('bias', '-')} | Catatan: {reasons_qa}\n"
    except Exception:
        pass

    _bias_qa = bias_htf_cached()
    _detail_qa = _bias_qa.get("detail", {}) if isinstance(_bias_qa, dict) else {}
    _detail_txt = " | ".join(str(_detail_qa.get(k, "-")) for k in ("D1", "H4", "H1"))
    return (
        f"\n[UPDATE DATA LIVE MT5 DETIK INI SAAT TRADER BERTANYA]:\n"
        f"- Harga Sekarang: Bid {harga:.2f} / Ask {tick.ask if tick else harga:.2f} (Spread {spread:.2f}) | Volatilitas ATR14 M5: {atr5[-1]:.2f} pt\n"
        f"- {teks_bias_htf(_bias_qa)} | Detail: {_detail_txt}\n"
        f"{scan_qa_txt}"
        f"{htf_running_txt}"
        f"{sig_live_txt}"
        f"{zona_mt5_qa_txt}"
        f"- Struktur M5 Terbaru: {ev_txt}\n"
        f"- CISD (Change in State of Delivery) M5 Terbaru: {cisd_txt} | Trigger Calon Saat Ini: {calon_cisd5}\n"
        f"- Liquidity Sweep M5 Terbaru: {sw_txt}\n"
        f"- Pola Candle M5 Terbaru: {', '.join(pola5) if pola5 else 'tidak ada'}\n"
        f"- Status Order Flow & Sideways M5: {inst5}\n"
        f"- {m30_info}"
        f"{pos_txt}\n"
        f"{info_akun_qa}\n"
        f"10 Candle M5 Terakhir yang Sudah Close:\n{tabel5}\n"
    )


def _tangani_error_model(model_name, err):
    pesan = str(err).lower()
    if "404" in pesan or "not_found" in pesan or "no longer available" in pesan:
        MODEL_404_BLACKLIST.add(model_name)
    elif "429" in pesan or "resource_exhausted" in pesan:
        MODEL_COOLDOWN[model_name] = time.time() + 120  # istirahatkan model yang kena limit 2 menit
    elif "503" in pesan or "unavailable" in pesan or "high demand" in pesan:
        MODEL_COOLDOWN[model_name] = time.time() + 60
    elif "504" in pesan or "deadline" in pesan or "timeout" in pesan or "timed out" in pesan:
        MODEL_COOLDOWN[model_name] = time.time() + 120


def panggil_gemini(client, contents, stream_output=False):
    """Memilih otomatis server AI yang paling longgar & cepat:
    - Mengingat model tercepat sebelumnya & mem-blacklist model 404 / cooldown 429 agar tidak buang waktu.
    - Mendukung mode Streaming (kata langsung keluar real-time tanpa menunggu seluruh paragraf selesai).
    - Jika gagal/lambat, menembak beberapa server kandidat secara paralel (balapan server tercepat)."""
    global MODEL_AKTIF_TERCEPAT

    now = time.time()
    kandidat = [
        m for m in MODELS_TO_TRY
        if m not in MODEL_404_BLACKLIST and MODEL_COOLDOWN.get(m, 0) <= now
    ]
    if not kandidat:
        # Jika semua sedang cooldown, coba kembali semua yang tidak 404
        kandidat = [m for m in MODELS_TO_TRY if m not in MODEL_404_BLACKLIST]

    if MODEL_AKTIF_TERCEPAT and MODEL_AKTIF_TERCEPAT in kandidat:
        kandidat.remove(MODEL_AKTIF_TERCEPAT)
        kandidat.insert(0, MODEL_AKTIF_TERCEPAT)

    # 1. Jika mode streaming aktif (untuk sesi Tanya Jawab agar teks langsung muncul instan)
    if stream_output and kandidat:
        for model_name in kandidat[:2]:
            try:
                stream = client.models.generate_content_stream(
                    model=model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(temperature=0.0),
                )
                teks_penuh = []
                printed_header = False
                for chunk in stream:
                    if chunk.text:
                        if not printed_header:
                            print(f"\n[Jawaban AI ({model_name})]:")
                            printed_header = True
                        print(chunk.text, end="", flush=True)
                        teks_penuh.append(chunk.text)
                if teks_penuh:
                    print()
                    MODEL_AKTIF_TERCEPAT = model_name
                    return "".join(teks_penuh), model_name
            except Exception as e:
                print(f"\n   [⚠️ Server {model_name} padat/kendala ({type(e).__name__}). Mencoba cadangan...]")
                _tangani_error_model(model_name, e)
                continue

    # 2. Balapan paralel (Fastest-Server Race) ke hingga 3 server sekaligus, ambil yang merespon paling cepat!
    batch_utama = [m for m in kandidat if m not in MODEL_404_BLACKLIST and MODEL_COOLDOWN.get(m, 0) <= time.time()][:3]
    if batch_utama:
        pool = ThreadPoolExecutor(max_workers=len(batch_utama))
        try:
            futures = {pool.submit(_coba_satu_model, client, m, contents): m for m in batch_utama}
            for fut in as_completed(futures):
                m_name = futures[fut]
                try:
                    teks, mdl, durasi = fut.result()
                    if teks:
                        MODEL_AKTIF_TERCEPAT = mdl
                        if stream_output:
                            print(f"\n[Jawaban AI ({mdl} — {durasi:.1f} dtk)]:\n{teks}")
                        return teks, mdl
                except Exception as e:
                    _tangani_error_model(m_name, e)
        finally:
            # JANGAN 'with ThreadPoolExecutor': __exit__ menunggu semua thread (termasuk yang kalah) sehingga tidak ada keuntungan balapan.
            pool.shutdown(wait=False, cancel_futures=True)

    # 3. Fallback sisa model jika batch utama sibuk
    sisa = [m for m in MODELS_TO_TRY if m not in MODEL_404_BLACKLIST and m not in batch_utama]
    for model_name in sisa:
        try:
            teks, mdl, _ = _coba_satu_model(client, model_name, contents)
            if teks:
                MODEL_AKTIF_TERCEPAT = mdl
                if stream_output:
                    print(f"\n[Jawaban AI ({mdl})]:\n{teks}")
                return teks, mdl
        except Exception as e:
            _tangani_error_model(model_name, e)
            continue

    return None, None


def validasi_model_tersedia(client):
    """Periksa nama di MODELS_TO_TRY terhadap daftar model akun/API Anda; yang tidak ada dilewati (tidak diam-diam 404)."""
    try:
        tersedia = {str(m.name).replace("models/", "") for m in client.models.list()}
    except Exception as e:
        print(f"   [ℹ️ Daftar model tidak bisa diperiksa ({e}); lanjut apa adanya.]")
        return
    if not tersedia:
        return
    hilang = [m for m in MODELS_TO_TRY if m not in tersedia]
    for m in hilang:
        MODEL_404_BLACKLIST.add(m)
    if hilang:
        print(f"   [⚠️ Model berikut tidak tersedia di akun/API Anda dan dilewati: {', '.join(hilang)}]")
    if len(hilang) == len(MODELS_TO_TRY):
        sys.exit("Tidak ada model di MODELS_TO_TRY yang tersedia. Perbarui daftar model (lihat client.models.list()).")
