# -*- coding: utf-8 -*-
"""
modules/ai_gemini.py
Integrasi Google Gemini API: pengumpulan konteks lengkap data MT5,
ekstraksi rekomendasi sinyal & zona [RADAR_TARGET], multi-model failover,
serta live snapshot kontekstual untuk Tanya Jawab AI.
"""
import inspect
import json
import os
import re
import sys
import time
from typing import Optional, Dict, Any, List, Tuple
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
from modules.zone_scanner import (
    scan_xauusd_zones, validate_entry_setup, evaluate_trade_decision, create_market_snapshot
)


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

    # SATU snapshot untuk seluruh siklus konteks ini: frames & tick yang sama dengan yang sudah diambil di atas
    # (tidak ada fetch tick/candle baru). Dipakai oleh golden confluence DAN evaluasi scanner di bawah.
    try:
        if tick is not None:
            snap_ctx = create_market_snapshot(frames=frames, tick=tick)
        else:
            snap_ctx = create_market_snapshot(frames=frames, current_price=float(harga_ref), live=False)
    except Exception as e_snap:
        _catat_error("kumpulkan_konteks_snapshot", e_snap)
        snap_ctx = None
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
    zona_text += cari_golden_confluence(frames, harga_ref, snapshot=snap_ctx)
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

    # Evaluasi keputusan teknikal dari Python Scanner secara langsung (single snapshot)
    try:
        snap_scan = snap_ctx if snap_ctx is not None else create_market_snapshot(
            frames=frames, current_price=float(harga_ref), live=False)
        scan_res = scan_xauusd_zones(frames=frames, snapshot=snap_scan)
        dec_scan = evaluate_trade_decision(
            frames=frames, live=False,
            snapshot=snap_scan, scan_result=scan_res,
        )
    except Exception:
        dec_scan = {"status": "WAIT", "reasons": ["Scanner dalam mode fallback"]}
        scan_res = {}

    st_dec = dec_scan.get("status", "WAIT")
    dir_dec = dec_scan.get("direction")
    plan_dec = dec_scan.get("plan")
    reasons_dec = dec_scan.get("reasons", [])
    reasons_txt = "; ".join(reasons_dec) if reasons_dec else "Konfirmasi scanner belum terpenuhi"
    ctx_htf = scan_res.get("htf_context", {})
    h4_bias_txt = ctx_htf.get("H4", "NETRAL")
    h1_struct_txt = ctx_htf.get("H1", "NETRAL")

    if st_dec == "VALID" and plan_dec and dir_dec:
        kandidat_algo_txt = (
            f"\n===== KEPUTUSAN RESMI PYTHON SCANNER: VALID ({dir_dec}) =====\n"
            f"  * STATUS FINAL PYTHON SCANNER: VALID ({dir_dec})\n"
            f"  * REKOMENDASI RESMI ENTRY     : {dir_dec} @ {plan_dec['entry_min']:.2f} - {plan_dec['entry_max']:.2f}\n"
            f"  * STOP LOSS RESMI             : {plan_dec['sl']:.2f}\n"
            f"  * TARGET TP1 / TP2            : TP1 {plan_dec['tp1']:.2f} | TP2 {plan_dec['tp2']:.2f}\n"
            f"  * RISK/REWARD RESMI           : 1:{plan_dec['rr']:.1f}\n"
            f"  * ALASAN VALIDASI             : {reasons_txt}\n"
            f"  (AI WAJIB MENGIKUTI DAN MENGUNCI ANGKA RESMI PYTHON SCANNER DI ATAS 100%!)\n"
        )
    else:
        # Ambil zona radar resmi langsung dari Python zone_scanner.py
        all_radar = scan_res.get("radar_zones") or scan_res.get("active_zones", [])
        if not all_radar:
            kandidat_algo_txt = "\n===== RADAR ZONE PYTHON SCANNER =====\nNO RADAR ZONE DETECTED.\n(Python Scanner tidak menemukan zona yang memenuhi syarat; AI DILARANG mengarang zona!)\n"
        else:
            kandidat_algo_txt = f"\n===== RADAR ZONE PYTHON SCANNER (STATUS SCANNER: {st_dec} — BUKAN SINYAL ENTRY / PENDING ORDER) =====\n"
            kandidat_algo_txt += (
                f"  (PERINGATAN KERAS OTORITAS PYTHON SCANNER: Status saat ini {st_dec} [{reasons_txt}].\n"
                f"   Semua zona di bawah WAJIB bersumber dari Python zone_scanner.py tanpa interpretasi perkiraan!\n"
                f"   DILARANG KERAS memberikan sinyal rekomendasi BUY atau SELL dalam bentuk apapun!\n"
                f"   DILARANG memberikan rekomendasi pending order (BUY LIMIT / SELL LIMIT) seolah-olah sudah valid!\n"
                f"   Zona di bawah HANYA kandidat radar untuk dipantau, BUKAN rekomendasi entry aktif):\n"
            )
            demands = [z for z in all_radar if z.get("zone_type") == "DEMAND"]
            supplies = [z for z in all_radar if z.get("zone_type") == "SUPPLY"]
            for i, zd in enumerate(demands[:2], 1):
                z_lo, z_hi = float(zd.get("price_low", 0.0)), float(zd.get("price_high", 0.0))
                inval = float(zd.get("invalidation_price", z_lo))
                tf = zd.get("timeframe", "H1")
                st = zd.get("status", "RADAR")
                ob = zd.get("origin_bos", {}).get("tipe", "OB")
                fvg = (zd.get("fvg_confirmation") or {}).get("status", "-")
                dist = 0.0 if (z_lo <= harga_ref <= z_hi) else (z_lo - harga_ref if harga_ref < z_lo else harga_ref - z_hi)
                kandidat_algo_txt += (
                    f"  * DEMAND ZONE #{i}: {tf} DEMAND {z_lo:.2f}-{z_hi:.2f} | Status: [{st}] | Invalidation: {inval:.2f} | "
                    f"OB: {ob} | FVG: {fvg} | Jarak: {dist:.2f} pt di bawah harga [HANYA PANTAU, BUKAN ENTRY]\n"
                )
            for i, zs in enumerate(supplies[:2], 1):
                z_lo, z_hi = float(zs.get("price_low", 0.0)), float(zs.get("price_high", 0.0))
                inval = float(zs.get("invalidation_price", z_hi))
                tf = zs.get("timeframe", "H1")
                st = zs.get("status", "RADAR")
                ob = zs.get("origin_bos", {}).get("tipe", "OB")
                fvg = (zs.get("fvg_confirmation") or {}).get("status", "-")
                dist = 0.0 if (z_lo <= harga_ref <= z_hi) else (z_lo - harga_ref if harga_ref < z_lo else harga_ref - z_hi)
                kandidat_algo_txt += (
                    f"  * SUPPLY ZONE #{i}: {tf} SUPPLY {z_lo:.2f}-{z_hi:.2f} | Status: [{st}] | Invalidation: {inval:.2f} | "
                    f"OB: {ob} | FVG: {fvg} | Jarak: {dist:.2f} pt di atas harga [HANYA PANTAU, BUKAN ENTRY]\n"
                )
    zona_text += kandidat_algo_txt

    if st_dec != "VALID":
        otoritas_scanner_txt = f"""
🛡️ [OTORITAS MUTLAK PYTHON SCANNER — KEPUTUSAN FINAL: {st_dec}]:
- Hasil Evaluasi Python Scanner: {st_dec} (Directional Bias H4: {h4_bias_txt}, Struktur H1: {h1_struct_txt})
- Catatan / Alasan: {reasons_txt}
- LARANGAN KERAS UNTUK ANALISIS AI (WAJIB DITAATI 100% TANPA PENGECUALIAN):
  1. DILARANG KERAS MENGELUARKAN REKOMENDASI BUY ATAU SELL DALAM BENTUK APAPUN!
  2. DILARANG MEMBERIKAN PENDING ORDER (BUY LIMIT / SELL LIMIT) SEOLAH-OLAH SUDAH VALID!
  3. DILARANG MENETAPKAN AREA ENTRY, STOP LOSS, DAN TAKE PROFIT SEBAGAI REKOMENDASI RESMI!
  4. DILARANG MENGUBAH STATUS WAIT MENJADI LOW RISK MAUPUN MEDIUM RISK!
  5. DILARANG MENGGUNAKAN SKOR KONFLUENSI INTERNAL AI UNTUK MELEWATI VALIDASI PYTHON SCANNER!
  6. AI WAJIB menuliskan status keputusan secara tegas: "WAIT: Belum ada setup valid ({reasons_txt})".
  7. Pada blok [RADAR_TARGET] di akhir jawaban, WAJIB menuliskan:
     SINYAL_AKTIF: WAIT | ENTRY: 0.00-0.00 | SL: 0.00 | TP1: 0.00 | TP2: 0.00
"""
    else:
        otoritas_scanner_txt = f"""
🎯 [OTORITAS MUTLAK PYTHON SCANNER — KEPUTUSAN FINAL: VALID ({dir_dec})]:
- Python Scanner telah MENYETUJUI dan MEMVALIDASI setup {dir_dec}!
- Parameter Resmi Terkonfirmasi:
  * Arah       : {dir_dec}
  * Area Entry : {plan_dec['entry_min']:.2f} - {plan_dec['entry_max']:.2f}
  * Stop Loss  : {plan_dec['sl']:.2f}
  * TP1 / TP2  : {plan_dec['tp1']:.2f} / {plan_dec['tp2']:.2f}
  * Risk/Reward: 1:{plan_dec['rr']:.1f}
- AI WAJIB menyelaraskan analisa dengan angka resmi di atas tanpa mengubah angka sedikitpun!
"""

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

{otoritas_scanner_txt}

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
5. OUTPUT RINGKAS, PROFESIONAL & FOKUS KONTEKS:
   - DILARANG menduplikasi atau mencetak ulang tabel OHLC mentah berbaris-baris atau mengulang daftar seluruh kandidat zona di dalam respons penjelasanmu.
   - FOKUSKAN penjelasanmu pada konteks makroekonomi, katalis berita High-Impact USD, korelasi Dollar Index (DXY), SMT Divergence (XAU vs XAG), dan konteks ADR (Average Daily Range) batas bensin harian.
6. PATUHI KEPUTUSAN PYTHON SCANNER 100%:
   - Python Scanner adalah SATU-SATUNYA otoritas keputusan teknikal.
   - Jika Python Scanner berstatus WAIT: dilarang memberikan rekomendasi entry aktif baru atau mengarang level sendiri. Jelaskan alasan WAIT secara logis, disiplin, dan profesional.

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
      - Ambil langsung rekomendasi ukuran lot dan mata uang akun (`IDC` / `USD`) dari bagian "MANAJEMEN RISIKO & KALKULATOR LOT OTOMATIS" sesuai jarak SL dan equity akun MT5 (0.5% untuk risiko lebih konservatif, 1.0% untuk risiko lebih tinggi dibandingkan 0.5%).
      - Jika data kalkulasi lot atau kontrak simbol MT5 tidak lengkap/tersedia, tampilkan "DATA UNAVAILABLE: Kalkulasi lot belum tersedia" dan DILARANG MENGARANG ukuran lot!
      - Bedakan persentase risiko modal dari ukuran lot. DILARANG MENGUBAH Entry, SL, TP, atau RR resmi dari Python Scanner!

   FORMAT SETIAP SINYAL WAJIB DITULIS RAPI SEPERTI INI:
   ----------------------------------------------------------------------
   [SINYAL <KONSERVATIF (0.5%) / STANDAR (1.0%)>] <BUY / SELL> XAUUSD (<Tipe Order: Limit Pullback / Confirmed Break>)
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

   Kriteria Klasifikasi Risiko & Alokasi Modal:
   - RISIKO LEBIH KONSERVATIF (0.5% modal):
     * Risiko lebih konservatif, perlindungan modal lebih disiplin.
   - RISIKO LEBIH TINGGI (1.0% modal):
     * Memiliki eksposur risiko lebih besar dibandingkan 0.5%.
     * DILARANG menyebut 1.0% sebagai Low Risk jika 0.5% disebut Medium Risk!
     * Bedakan persentase risiko modal dari ukuran lot fisik.
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
        _eval_sig = inspect.signature(evaluate_trade_decision)
        if "scan_result" in _eval_sig.parameters and "snapshot" in _eval_sig.parameters:
            _snap_ot = create_market_snapshot()
            scan = scan_xauusd_zones(snapshot=_snap_ot)
            dec = evaluate_trade_decision(snapshot=_snap_ot, scan_result=scan)
        else:
            dec = evaluate_trade_decision()   # kompatibel dengan mock 0-argumen pada unit test
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


def sanitasi_output_gemini(
    teks_ai: str,
    dec: Optional[Dict[str, Any]] = None,
    scan: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Menegakkan OTORITAS MUTLAK PYTHON SCANNER terhadap output Gemini:
    1. Python VALID: sinyal resmi boleh ditampilkan, angka entry/SL/TP dikunci ke rencana Python.
    2. Python WAIT / INVALID:
       - DILARANG menampilkan rekomendasi sinyal BUY/SELL atau pending limit aktif.
       - Blok [SINYAL ...] dinetralisir menjadi [STATUS TRADING: WAIT / INVALID].
       - Area Entry, SL, TP dinonaktifkan (diisi tanda strip '-' / WAIT).
       - Blok [RADAR_TARGET] disanitasi: SINYAL_AKTIF menjadi WAIT.
       - Menambahkan keterangan jelas bahwa Python Scanner berstatus WAIT/INVALID.
    3. Sumber Radar Zone:
       - Jika Python Scanner tidak menemukan zona: tampilkan NO RADAR ZONE DETECTED.
       - Jika Python Scanner memiliki radar zone: kunci harga zona ke hasil Python (AI dilarang mengubah harga).
    """
    if not teks_ai or not isinstance(teks_ai, str):
        return teks_ai

    if dec is None:
        try:
            from modules.zone_scanner import evaluate_trade_decision
            dec = evaluate_trade_decision()
        except Exception:
            dec = {"status": "WAIT", "direction": None, "reasons": ["Evaluasi scanner default WAIT"]}

    status_scanner = str(dec.get("status", "WAIT")).upper()
    direction_scanner = dec.get("direction")
    plan = dec.get("plan")
    reasons = dec.get("reasons", [])
    alasan_txt = "; ".join(reasons) if reasons else "Persyaratan validasi scanner belum terpenuhi"

    # Penegakan Sumber Radar Zone (Section 1 Audit)
    if scan is not None:
        radar_list = scan.get("radar_zones") or scan.get("active_zones", [])
        if not radar_list:
            def _bersihkan_radar_kosong(m):
                isi = m.group(1)
                isi = re.sub(r"ZONA_ATAS:[^\n]+", "ZONA_ATAS: NO RADAR ZONE DETECTED", isi, flags=re.I)
                isi = re.sub(r"ZONA_BAWAH:[^\n]+", "ZONA_BAWAH: NO RADAR ZONE DETECTED", isi, flags=re.I)
                if "NO RADAR ZONE DETECTED" not in isi:
                    isi += "\nZONA_ATAS: NO RADAR ZONE DETECTED\nZONA_BAWAH: NO RADAR ZONE DETECTED"
                return f"[RADAR_TARGET]{isi}[/RADAR_TARGET]"

            if "[RADAR_TARGET]" in teks_ai:
                teks_ai = re.sub(r"\[RADAR_TARGET\](.*?)\[/RADAR_TARGET\]", _bersihkan_radar_kosong, teks_ai, flags=re.S | re.I)
            else:
                teks_ai += "\n[RADAR_TARGET]\nZONA_ATAS: NO RADAR ZONE DETECTED\nZONA_BAWAH: NO RADAR ZONE DETECTED\nSINYAL_AKTIF: WAIT | ENTRY: 0.00-0.00 | SL: 0.00 | TP1: 0.00 | TP2: 0.00\n[/RADAR_TARGET]\n"

            teks_ai = re.sub(r"(?:•|\*|-)?\s*Kandidat\s+(?:Demand|Supply)[^\n]+", "NO RADAR ZONE DETECTED.", teks_ai, flags=re.I)
            if "NO RADAR ZONE DETECTED" not in teks_ai:
                teks_ai += "\n[RADAR ZONE]: NO RADAR ZONE DETECTED.\n"
        else:
            bs = scan.get("best_supply") or next((z for z in radar_list if z.get("zone_type") == "SUPPLY"), None)
            bd = scan.get("best_demand") or next((z for z in radar_list if z.get("zone_type") == "DEMAND"), None)

            def _kunci_radar_python(m):
                isi = m.group(1)
                if bs:
                    sl_s = bs.get('sl_recommended', bs.get('invalidation_price', bs['price_high']))
                    garis_atas = f"ZONA_ATAS: {bs['price_low']:.2f}-{bs['price_high']:.2f} (batas_jebol: {sl_s:.2f})"
                    isi = re.sub(r"ZONA_ATAS:[^\n]+", garis_atas, isi, flags=re.I)
                if bd:
                    sl_d = bd.get('sl_recommended', bd.get('invalidation_price', bd['price_low']))
                    garis_bawah = f"ZONA_BAWAH: {bd['price_low']:.2f}-{bd['price_high']:.2f} (batas_jebol: {sl_d:.2f})"
                    isi = re.sub(r"ZONA_BAWAH:[^\n]+", garis_bawah, isi, flags=re.I)
                return f"[RADAR_TARGET]{isi}[/RADAR_TARGET]"

            if "[RADAR_TARGET]" in teks_ai:
                teks_ai = re.sub(r"\[RADAR_TARGET\](.*?)\[/RADAR_TARGET\]", _kunci_radar_python, teks_ai, flags=re.S | re.I)

    if status_scanner == "VALID" and plan and direction_scanner:
        # Valid setup! Ensure AI output matches Python direction and plan
        teks_hasil = teks_ai
        def _perbaiki_radar_valid(m):
            isi = m.group(1)
            garis_sig = f"SINYAL_AKTIF: {direction_scanner} | ENTRY: {plan['entry_min']:.2f}-{plan['entry_max']:.2f} | SL: {plan['sl']:.2f} | TP1: {plan['tp1']:.2f} | TP2: {plan['tp2']:.2f}"
            isi_baru = re.sub(r"SINYAL_AKTIF:[^\n]+", garis_sig, isi, flags=re.I)
            if "SINYAL_AKTIF:" not in isi_baru:
                isi_baru += f"\n{garis_sig}"
            return f"[RADAR_TARGET]{isi_baru}[/RADAR_TARGET]"

        if "[RADAR_TARGET]" in teks_hasil:
            teks_hasil = re.sub(r"\[RADAR_TARGET\](.*?)\[/RADAR_TARGET\]", _perbaiki_radar_valid, teks_hasil, flags=re.S | re.I)
        return teks_hasil

    # JIKA SCANNER WAIT ATAU INVALID:
    # Netralisir seluruh rekomendasi sinyal BUY/SELL dan pending limit!
    teks_bersih = teks_ai

    # 1. Netralisir blok header sinyal [SINYAL <RISK>] <BUY/SELL>
    teks_bersih = re.sub(
        r"(?:###\s*)?(?:\*\*)?\[SINYAL[^\]]*\](?:\*\*)?\s*(?:BUY|SELL)[^\n]*",
        f"[STATUS TRADING: {status_scanner} — TIDAK ADA REKOMENDASI ENTRY AKTIF]",
        teks_bersih,
        flags=re.I
    )

    # 2. Netralisir rekomendasi langsung BUY/SELL LIMIT di baris terpisah
    teks_bersih = re.sub(
        r"([•\*\-]?\s*(?:Rekomendasi|Sinyal|Order)\s*(?:Utama|Aktif)?\s*:\s*)(?:BUY|SELL)\s*(?:LIMIT|STOP)?[^\n]*",
        rf"\1WAIT (Tidak ada pending order aktif — status scanner {status_scanner})",
        teks_bersih,
        flags=re.I
    )

    # 3. Netralisir baris Area Entry
    teks_bersih = re.sub(
        r"([•\*\-]?\s*(?:Area\s*Entry|Entry\s*Zone|Entry)\s*:\s*)[^\n]+",
        rf"\1- (WAIT: Tidak ada rekomendasi entry aktif dari Python Scanner)",
        teks_bersih,
        flags=re.I
    )

    # 4. Netralisir baris Stop Loss yang menyertai sinyal
    teks_bersih = re.sub(
        r"([•\*\-]?\s*(?:Stop\s*Loss|SL)\s*:\s*)\d{4}(?:\.\d+)?[^\n]*",
        r"\1- (Tidak ada posisi aktif)",
        teks_bersih,
        flags=re.I
    )

    # 5. Netralisir baris Target TP (TP1 & TP2)
    teks_bersih = re.sub(
        r"([•\*\-]?\s*Target\s*TP\s*:)[^\n]*",
        r"\1 -",
        teks_bersih,
        flags=re.I
    )
    teks_bersih = re.sub(
        r"(\s*-\s*TP1\s*:\s*)\d{4}(?:\.\d+)?[^\n]*",
        r"\1-",
        teks_bersih,
        flags=re.I
    )
    teks_bersih = re.sub(
        r"(\s*-\s*TP2\s*:\s*)\d{4}(?:\.\d+)?[^\n]*",
        r"\1-",
        teks_bersih,
        flags=re.I
    )

    # 6. Netralisir Manajemen Lot jika merekomendasikan lot aktif
    teks_bersih = re.sub(
        r"([•\*\-]?\s*Manajemen\s*Lot\s*:\s*)\d+(?:\.\d+)?\s*lot[^\n]*",
        r"\g<1>0.00 lot (Menunggu sinyal valid)",
        teks_bersih,
        flags=re.I
    )

    # 7. Netralisir Note Limit (Anti Kena Kentut) jika menyuruh pasang limit
    teks_bersih = re.sub(
        r"([•\*\-]?\s*Note\s*Limit[^\n:]*:\s*)[^\n]+",
        r"\1Jangan memasang pending limit; scanner belum memvalidasi setup.",
        teks_bersih,
        flags=re.I
    )

    # 8. Netralisir [RADAR_TARGET]
    def _perbaiki_radar_wait(m):
        isi = m.group(1)
        garis_sig = "SINYAL_AKTIF: WAIT | ENTRY: 0.00-0.00 | SL: 0.00 | TP1: 0.00 | TP2: 0.00"
        isi_baru = re.sub(r"SINYAL_AKTIF:[^\n]+", garis_sig, isi, flags=re.I)
        if "SINYAL_AKTIF:" not in isi_baru:
            isi_baru += f"\n{garis_sig}"
        return f"[RADAR_TARGET]{isi_baru}[/RADAR_TARGET]"

    if "[RADAR_TARGET]" in teks_bersih:
        teks_bersih = re.sub(r"\[RADAR_TARGET\](.*?)\[/RADAR_TARGET\]", _perbaiki_radar_wait, teks_bersih, flags=re.S | re.I)

    # 9. Pastikan ada peringatan otoritas scanner jika teks tampak memberikan sinyal
    header_otoritas = f"\n[OTORITAS PYTHON SCANNER — KEPUTUSAN FINAL: {status_scanner}]\nCatatan: {alasan_txt}\nTidak ada rekomendasi BUY/SELL atau pending order yang aktif.\n"
    if "[OTORITAS PYTHON SCANNER" not in teks_bersih:
        teks_bersih = header_otoritas + teks_bersih

    return teks_bersih


def ekstrak_target_ai(teks_ai):
    """Membaca target dan sinyal dengan Python Zone Scanner sebagai satu-satunya sumber kebenaran teknikal.
    AI Gemini hanya memberikan narasi dan analisis tambahan, tanpa otoritas mengubah angka teknikal."""
    teks_ai = sanitasi_output_gemini(teks_ai)
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

    # Periksa status scanner lebih awal agar penamaan zona sinkron dengan otoritas Python
    scan_qa_txt = ""
    st_qa = "WAIT"
    dec_qa = {"status": "WAIT", "reasons": []}
    try:
        from modules.zone_scanner import scan_xauusd_zones, evaluate_trade_decision, create_market_snapshot
        _snap_qa = create_market_snapshot()
        scan_qa = scan_xauusd_zones(snapshot=_snap_qa)
        dec_qa = evaluate_trade_decision(snapshot=_snap_qa, scan_result=scan_qa)
        st_qa = dec_qa.get("status", "WAIT")
        reasons_qa = "; ".join(dec_qa.get("reasons", [])) or "tidak ada alasan"
        scan_qa_txt = f"- Status Python Zone Scanner: [{st_qa}] | Bias: {scan_qa.get('bias', '-')} | Catatan: {reasons_qa}\n"
    except Exception:
        pass

    # Ambil zona radar resmi langsung dari Python zone_scanner.py
    radar_qa_list = scan_qa.get("radar_zones") or scan_qa.get("active_zones", [])
    if st_qa == "VALID" and dec_qa.get("plan"):
        plan_qa = dec_qa["plan"]
        zona_mt5_qa_txt = (
            f"- [SINYAL RESMI TERKONFIRMASI PYTHON SCANNER DETIK INI]:\n"
            f"  * ARAH: {dec_qa.get('direction')} | ENTRY: {plan_qa['entry_min']:.2f}-{plan_qa['entry_max']:.2f} | "
            f"SL: {plan_qa['sl']:.2f} | TP1: {plan_qa['tp1']:.2f} | TP2: {plan_qa['tp2']:.2f} (RR 1:{plan_qa['rr']:.1f})\n"
        )
    elif not radar_qa_list:
        zona_mt5_qa_txt = "- [RADAR ZONE PYTHON SCANNER]: NO RADAR ZONE DETECTED. (Tidak ada zona pemantauan valid).\n"
        scan_qa_txt += "  [ATURAN OTORITAS MUTLAK]: Status Python Scanner WAIT/INVALID. Dilarang memberikan rekomendasi BUY/SELL atau pending order limit!\n"
    else:
        zona_mt5_qa_txt = "- [RADAR ZONA PEMANTAUAN MT5 — BUKAN REKOMENDASI PENDING ORDER / ENTRY AKTIF]:\n"
        demands_qa = [z for z in radar_qa_list if z.get("zone_type") == "DEMAND"]
        supplies_qa = [z for z in radar_qa_list if z.get("zone_type") == "SUPPLY"]
        for i, zd in enumerate(demands_qa[:2], 1):
            z_lo, z_hi = float(zd.get("price_low", 0.0)), float(zd.get("price_high", 0.0))
            inval = float(zd.get("invalidation_price", z_lo))
            tf = zd.get("timeframe", "H1")
            st = zd.get("status", "RADAR")
            dist = 0.0 if (z_lo <= harga <= z_hi) else (z_lo - harga if harga < z_lo else harga - z_hi)
            zona_mt5_qa_txt += (
                f"  * DEMAND ZONE #{i}: {tf} DEMAND {z_lo:.2f}-{z_hi:.2f} | Status: [{st}] | Invalidation: {inval:.2f} | "
                f"Jarak: {dist:.2f} pt di bawah harga [HANYA PANTAU, BUKAN ENTRY]\n"
            )
        for i, zs in enumerate(supplies_qa[:2], 1):
            z_lo, z_hi = float(zs.get("price_low", 0.0)), float(zs.get("price_high", 0.0))
            inval = float(zs.get("invalidation_price", z_hi))
            tf = zs.get("timeframe", "H1")
            st = zs.get("status", "RADAR")
            dist = 0.0 if (z_lo <= harga <= z_hi) else (z_lo - harga if harga < z_lo else harga - z_hi)
            zona_mt5_qa_txt += (
                f"  * SUPPLY ZONE #{i}: {tf} SUPPLY {z_lo:.2f}-{z_hi:.2f} | Status: [{st}] | Invalidation: {inval:.2f} | "
                f"Jarak: {dist:.2f} pt di atas harga [HANYA PANTAU, BUKAN ENTRY]\n"
            )
        scan_qa_txt += "  [ATURAN OTORITAS MUTLAK]: Status Python Scanner WAIT/INVALID. Dilarang memberikan rekomendasi BUY/SELL atau pending order limit!\n"

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

    if client is None:
        return None, None

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
                    raw_text = "".join(teks_penuh)
                    MODEL_AKTIF_TERCEPAT = model_name
                    sanitized_text = sanitasi_output_gemini(raw_text)
                    if sanitized_text != raw_text:
                        print("\n[🛡️ OTORITAS PYTHON SCANNER]: Rekomendasi sinyal dinetralisir sesuai keputusan resmi Python Scanner (WAIT/INVALID).")
                    return sanitized_text, model_name
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
                        teks = sanitasi_output_gemini(teks)
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
                teks = sanitasi_output_gemini(teks)
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
