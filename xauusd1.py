# -*- coding: utf-8 -*-
"""
xauusd1.py — Sistem Analisis Trading Emas (XAUUSD) Terintegrasi MT5 dan Gemini AI.
Dimodularisasi ke dalam package modules/ agar kode rapi, cepat, dan mudah dipelihara.
"""
import csv
import os
import re
import sys
import time
from datetime import datetime

# Pastikan folder project selalu ada di urutan pertama sys.path agar package modules selalu terbaca
_DIR_PROYEK = os.path.dirname(os.path.abspath(__file__))
if _DIR_PROYEK not in sys.path:
    sys.path.insert(0, _DIR_PROYEK)

import MetaTrader5 as mt5
from google import genai
from google.genai import types

# Import dan re-export seluruh submodule agar kompatibel penuh 100%
import modules.config as _m_config
import modules.utils as _m_utils
import modules.mt5_conn as _m_mt5
import modules.analisis_teknikal as _m_tek
import modules.analisis_pasar as _m_pasar
import modules.sinyal_jurnal as _m_jurnal
import modules.state_manager as _m_state
import modules.ai_gemini as _m_ai
import modules.pantau as _m_pantau
import modules.zone_scanner as _m_zone
import modules.state_manager as SM

# Expose every function, variable, and helper (including _catat_error, _tutup_mt5, etc.)
for _mod in (_m_config, _m_utils, _m_mt5, _m_tek, _m_pasar, _m_jurnal, _m_state, _m_ai, _m_pantau, _m_zone):
    for _k, _v in _mod.__dict__.items():
        if not _k.startswith('__'):
            globals()[_k] = _v


def main():
    global SINYAL_AKTIF_DINAMIS, SINYAL_SEBELUMNYA_DINAMIS, ZONA_ATAS_DINAMIS, ZONA_BAWAH_DINAMIS
    state_sebelumnya = muat_state_sinyal()
    sig_sebelumnya = None
    sig_riwayat_lama = None
    if state_sebelumnya:
        cand_s = state_sebelumnya.get("sinyal")
        if not (cand_s and isinstance(cand_s, dict) and cand_s.get("lo")):
            cand_s = state_sebelumnya.get("sinyal_sebelumnya")
        if cand_s and isinstance(cand_s, dict) and cand_s.get("lo"):
            sig_sebelumnya = cand_s
        # Ambil juga riwayat sinyal yang lebih lama
        cand_r = state_sebelumnya.get("sinyal_sebelumnya")
        if cand_r and isinstance(cand_r, dict) and cand_r.get("lo") and cand_r != sig_sebelumnya:
            sig_riwayat_lama = cand_r

    teks_memori_prompt = ""

    if sig_sebelumnya and isinstance(sig_sebelumnya, dict):
        SINYAL_SEBELUMNYA_DINAMIS = sig_sebelumnya
        tick_cek = mt5.symbol_info_tick(SYMBOL) if pastikan_mt5_aktif() else None
        bid_cek = tick_cek.bid if tick_cek else 0.0
        status_lama, alasan_lama = evaluasi_validitas_sinyal(sig_sebelumnya, bid_cek)
        lbl_eks_start, kode_eks_start, info_eks_start = cek_status_eksekusi_sinyal(sig_sebelumnya)

        print("\n" + "=" * 70)
        print("📌 [MEMORI SINYAL DARI SESI SEBELUMNYA DITEMUKAN]")
        print(f"• Waktu Dicatat     : {state_sebelumnya.get('waktu_simpan', '-')}")

        # Tampilkan riwayat sinyal yang lebih lama terlebih dahulu (jika ada)
        if sig_riwayat_lama:
            st_riwayat, al_riwayat = evaluasi_validitas_sinyal(sig_riwayat_lama, bid_cek)
            lbl_rw, _, info_rw = cek_status_eksekusi_sinyal(sig_riwayat_lama)
            print("─" * 70)
            print(f"  📜 Riwayat Sinyal Lama: {sig_riwayat_lama['arah']} @ {sig_riwayat_lama['lo']:.2f} - {sig_riwayat_lama['hi']:.2f} | SL {sig_riwayat_lama['sl']:.2f} | TP1 {sig_riwayat_lama['tp1']:.2f}")
            print(f"     Status: [{st_riwayat}] | Akun: [{lbl_rw}]")
            print(f"     Sumber: {sig_riwayat_lama.get('ket', '-')}")
            print(f"     Alasan: {al_riwayat}")
            print("─" * 70)

        print(f"• Setup Sinyal Aktif: {sig_sebelumnya['arah']} @ {sig_sebelumnya['lo']:.2f} - {sig_sebelumnya['hi']:.2f}")
        print(f"• Stop Loss & TP    : SL {sig_sebelumnya['sl']:.2f} | TP1 {sig_sebelumnya['tp1']:.2f} | TP2 {sig_sebelumnya.get('tp2', sig_sebelumnya['tp1']):.2f}")
        print(f"• Status Validitas  : [{status_lama}]")
        print(f"• Keterangan Fakta  : {alasan_lama}")
        kat_start, ket_r_start = tentukan_klasifikasi_risiko_sinyal(sig_sebelumnya)
        tag_start = "⚠️ [HIGH RISK]" if kat_start == "HIGH RISK" else ("🛡️ [LOW RISK]" if kat_start == "LOW RISK" else "⚡ [MEDIUM RISK]")
        print(f"• Tingkat Risiko    : {tag_start} ({ket_r_start})")
        if kat_start == "HIGH RISK":
            print("  -> Panduan Trader : Sinyal dadakan LTF. Anda bebas mempertimbangkan untuk ambil atau skip!")
        print(f"• Status di Akun MT5: [{lbl_eks_start}] ({info_eks_start})")
        print(f"• Sumber Setup      : {sig_sebelumnya.get('ket', '-')}")
        print("=" * 70 + "\n")

        _riwayat_teks = ""
        if sig_riwayat_lama:
            st_riwayat2, al_riwayat2 = evaluasi_validitas_sinyal(sig_riwayat_lama, bid_cek)
            _riwayat_teks = (
                f"- Riwayat Sinyal Lama (HISTORIS): {sig_riwayat_lama['arah']} @ {sig_riwayat_lama['lo']:.2f} - {sig_riwayat_lama['hi']:.2f} | SL {sig_riwayat_lama['sl']:.2f} | TP1 {sig_riwayat_lama['tp1']:.2f} ({sig_riwayat_lama.get('ket', '-')}) -> [{st_riwayat2}]\n"
            )

        teks_memori_prompt = (
            f"\n\n[MEMORI SINYAL DARI SESI SEBELUMNYA — WAJIB DITAMPILKAN & DIEVALUASI VALIDITASNYA]:\n"
            f"Trader pada sesi sebelumnya memegang sinyal berikut:\n"
            f"{_riwayat_teks}"
            f"- 🔒 SINYAL AKTIF TERKUNCI: {sig_sebelumnya['arah']} | ENTRY: {sig_sebelumnya['lo']:.2f}-{sig_sebelumnya['hi']:.2f} | SL: {sig_sebelumnya['sl']:.2f} | TP1: {sig_sebelumnya['tp1']:.2f} | TP2: {sig_sebelumnya.get('tp2', sig_sebelumnya['tp1']):.2f} ({sig_sebelumnya.get('ket', '-')})\n"
            f"- Status Akun MT5: [{lbl_eks_start}] ({info_eks_start})\n"
            f"- Fakta Algoritmik Harga Sekarang ({bid_cek:.2f}): [{status_lama}] -> {alasan_lama}\n\n"
            f"INSTRUKSI ANTI-PLIN-PLAN & PENGUNCIAN ANGKA SINYAL (WAJIB DIPATUHI 100%):\n"
            f"1. BAGIAN WAJIB 1 — STATUS SINYAL SEBELUMNYA:\n"
            f"   Tampilkan kembali setup Sinyal Sebelumnya di atas dengan angka yang SAMA PERSIS. Beri label tegas: [MASIH VALID & AKTIF] atau [SUDAH TIDAK VALID / BATAL] / [TARGET TERCAPAI].\n"
            f"2. BAGIAN WAJIB 2 — LARANGAN MENGGESER ANGKA ENTRY/SL SELAMA MASIH VALID:\n"
            f"   - JIKA Fakta Algoritmik di atas menyatakan sinyal MASIH VALID (belum kena SL {sig_sebelumnya['sl']:.2f}, belum kedaluwarsa, dan belum TP2), Anda DILARANG KERAS mengubah/menggeser angka Area Entry ({sig_sebelumnya['lo']:.2f}-{sig_sebelumnya['hi']:.2f}) maupun SL ({sig_sebelumnya['sl']:.2f})!\n"
            f"   - Menggeser-geser angka entry setiap kali ditanya (misal dari {sig_sebelumnya['lo']:.2f}-{sig_sebelumnya['hi']:.2f} menjadi angka lain padahal SL {sig_sebelumnya['sl']:.2f} belum jebol) membuat trader bingung dan kehilangan kepercayaan!\n"
            f"   - Pada blok [RADAR_TARGET] di akhir jawaban, jika sinyal masih valid, wajib tulis ulang PERSIS:\n"
            f"     SINYAL_AKTIF: {sig_sebelumnya['arah']} | ENTRY: {sig_sebelumnya['lo']:.2f}-{sig_sebelumnya['hi']:.2f} | SL: {sig_sebelumnya['sl']:.2f} | TP1: {sig_sebelumnya['tp1']:.2f} | TP2: {sig_sebelumnya.get('tp2', sig_sebelumnya['tp1']):.2f}\n"
            f"   - Anda HANYA BOLEH menerbitkan angka Entry/SL baru jika sinyal lama SUDAH TERBUKTI BATAL (kena SL / lewat TTL / terlewat ke TP1).\n"
        )

    if not API_KEY:
        sys.exit("API key Gemini belum diset. Isi variabel lingkungan GEMINI_API_KEY (mis. setx GEMINI_API_KEY \"...\" lalu buka terminal baru).")
    try:
        info_jurnal = ringkasan_jurnal_untuk_ai(5)
        prompt = kumpulkan_konteks_mt5() + (f"\n\n{info_jurnal}" if info_jurnal else "") + teks_memori_prompt
    except KonteksError as e_ctx:
        sys.exit(str(e_ctx))
    print("2. Mengirim data ke Gemini (streaming real-time)...")
    client = genai.Client(
        api_key=API_KEY,
        http_options=types.HttpOptions(
            timeout=25000,
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )

    validasi_model_tersedia(client)
    jawaban_awal, model_aktif = panggil_gemini(client, prompt, stream_output=True)
    if not jawaban_awal:
        print("\n   [⚠️ Semua model Gemini sedang padat antrean. Mode mekanis & pantau tetap diaktifkan — ketik 'refresh' untuk coba lagi.]")
        jawaban_awal = ""
    _tgt_awal = ekstrak_target_ai(jawaban_awal)
    _cand_awal = _tgt_awal.get("sinyal")
    if _cand_awal and model_aktif:
        _cand_awal["model"] = model_aktif
    sinyal_baru_ai = terapkan_gerbang(_cand_awal)
    if sinyal_baru_ai and model_aktif:
        sinyal_baru_ai["model"] = model_aktif

    tick_st = mt5.symbol_info_tick(SYMBOL) if pastikan_mt5_aktif() else None
    bid_st = tick_st.bid if tick_st else 0.0
    st_cek = "UNKNOWN"
    if sig_sebelumnya and isinstance(sig_sebelumnya, dict) and bid_st > 0:
        st_cek, _ = evaluasi_validitas_sinyal(sig_sebelumnya, bid_st)

    # 🔒 SIGNAL & DIRECTIONAL LOCK via helper terpusat (konteks STARTUP)
    sig_pilih, sig_rw, logs_lock, diubah = putuskan_sinyal_terkunci(
        sinyal_baru_ai,
        sig_sebelumnya,
        status_aktif=st_cek,
        paksa_ganti=False,
        izinkan_balik_arah=False,
        konteks="STARTUP",
    )
    for p in logs_lock:
        print(p)
    if sig_rw:
        # Pengarsipan di Startup jika sinyal lama digantikan sinyal baru pasca-batal/selesai
        SINYAL_SEBELUMNYA_DINAMIS = sig_rw

    SINYAL_AKTIF_DINAMIS = sig_pilih

    # Selaraskan zona pantau dengan sinyal aktif agar tidak ada dua angka zona berbeda di layar
    if SINYAL_AKTIF_DINAMIS and isinstance(SINYAL_AKTIF_DINAMIS, dict):
        if SINYAL_AKTIF_DINAMIS["arah"] == "BUY":
            ZONA_BAWAH_DINAMIS = (SINYAL_AKTIF_DINAMIS["lo"], SINYAL_AKTIF_DINAMIS["hi"], SINYAL_AKTIF_DINAMIS["sl"])
            ZONA_ATAS_DINAMIS = _tgt_awal.get("atas") or (tuple(state_sebelumnya["atas"]) if state_sebelumnya and state_sebelumnya.get("atas") else None)
        else:
            ZONA_ATAS_DINAMIS = (SINYAL_AKTIF_DINAMIS["lo"], SINYAL_AKTIF_DINAMIS["hi"], SINYAL_AKTIF_DINAMIS["sl"])
            ZONA_BAWAH_DINAMIS = _tgt_awal.get("bawah") or (tuple(state_sebelumnya["bawah"]) if state_sebelumnya and state_sebelumnya.get("bawah") else None)
    else:
        ZONA_ATAS_DINAMIS = _tgt_awal.get("atas") or (tuple(state_sebelumnya["atas"]) if state_sebelumnya and state_sebelumnya.get("atas") else None)
        ZONA_BAWAH_DINAMIS = _tgt_awal.get("bawah") or (tuple(state_sebelumnya["bawah"]) if state_sebelumnya and state_sebelumnya.get("bawah") else None)

    # Simpan state awal yang memuat sinyal aktif & sinyal sebelumnya
    simpan_state_sinyal(SINYAL_AKTIF_DINAMIS, SINYAL_SEBELUMNYA_DINAMIS, ZONA_ATAS_DINAMIS, ZONA_BAWAH_DINAMIS, "Startup Awal")

    # =====================================================================
    #  FITUR TANYA JAWAB INTERAKTIF (Q&A CHAT DENGAN KONTEKS DATA MT5)
    # =====================================================================
    riwayat_chat = [
        {"role": "user", "parts": [{"text": prompt}]},
        {"role": "model", "parts": [{"text": jawaban_awal}]},
    ]

    print("\n" + "=" * 70)
    print("MODE TANYA JAWAB & ASISTEN TRADING INTERAKTIF AKTIF")
    print("- Ketik pertanyaan bebas tentang market / setup di atas.")
    print("- Ketik 'pantau'                -> OTOMATIS memantau zona AI & MENGAWAL POSISI RUNNING (Alert BEP/SL/Bahaya)!")
    print("- Ketik 'pantau <bawah> <atas>' -> memantau zona harga manual pilihan Anda + mengawal posisi running.")
    print("- Ketik 'sinyal' atau 'status'  -> tampilkan riwayat sinyal sebelumnya & status sinyal aktif saat ini.")
    print("- Ketik 'jurnal' atau 'rekap'   -> lihat daftar zona sinyal (mana yang diambil, kena SL & kena TP).")
    print("- Ketik 'ambil <nomor>'         -> tandai manual nomor zona di jurnal sebagai DIAMBIL (misal: ambil 7,8).")
    print("- Ketik 'skip <nomor>'          -> hapus tanda diambil pada nomor zona di jurnal (misal: skip 9).")
    print("- Ketik 'refresh' atau 'update' -> tarik ulang data terbaru dari MT5.")
    print("- Ketik 'zona' atau 'scan'      -> tampilkan laporan scanner zona SMC Python-only.")
    print("- Ketik 'simpan'                -> simpan hasil analisa, tanya jawab, & tabel jurnal ke file Jurnal (.md).")
    print("- Ketik 'exit' atau 'keluar'    -> berhenti.")
    print("=" * 70)

    while True:
        try:
            tanya = input("\n[Tanya AI] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSelesai. Semoga trading Anda lancar!")
            break

        if not tanya:
            continue
        cmd = tanya.lower()
        if cmd in ("exit", "keluar", "quit", "q"):
            print("Selesai. Semoga trading Anda lancar!")
            break

        if cmd in ("sinyal", "status", "status sinyal", "signal", "setup"):
            state_aktif = muat_state_sinyal()
            sig_aktif = SINYAL_AKTIF_DINAMIS or (state_aktif.get("sinyal") if state_aktif else None)
            sig_lama = SINYAL_SEBELUMNYA_DINAMIS or (state_aktif.get("sinyal_sebelumnya") if state_aktif else None)

            tick_now = mt5.symbol_info_tick(SYMBOL) if pastikan_mt5_aktif() else None
            bid_now = tick_now.bid if tick_now else 0.0

            print("\n" + "=" * 70)
            print("📊 [MONITORING DUA GENERASI SINYAL: SEBELUMNYA vs TERBARU]")
            print(f"• Harga Bid MT5 Saat Ini: {bid_now:.2f}")

            # 1. Tampilkan Sinyal Sebelumnya & Keterangan Validitasnya
            if sig_lama and isinstance(sig_lama, dict) and sig_lama.get("lo"):
                st_lama, al_lama = evaluasi_validitas_sinyal(sig_lama, bid_now)
                lbl_eks_lama, _, info_lama = cek_status_eksekusi_sinyal(sig_lama)
                print("\n" + "─" * 70)
                print(f"1. 📌 SINYAL SEBELUMNYA (SESI LALU) — [{lbl_eks_lama}]:")
                print(f"   • Setup Sinyal   : {sig_lama['arah']} @ {sig_lama['lo']:.2f} - {sig_lama['hi']:.2f}")
                print(f"   • Stop Loss & TP : SL {sig_lama['sl']:.2f} | TP1 {sig_lama['tp1']:.2f} | TP2 {sig_lama.get('tp2', sig_lama['tp1']):.2f}")
                print(f"   • STATUS VALID   : [{st_lama}]")
                print(f"   • Keterangan     : {al_lama}")
                print(f"   • Status di Akun : {lbl_eks_lama} ({info_lama})")
                print(f"   • Sumber Setup   : {sig_lama.get('ket', '-')}")
            else:
                print("\n" + "─" * 70)
                print("1. 📌 SINYAL SEBELUMNYA: Tidak ada riwayat sinyal sebelumnya yang tersimpan.")

            # 2. Tampilkan Sinyal Aktif Terbaru
            print("\n" + "─" * 70)
            if sig_aktif and isinstance(sig_aktif, dict) and sig_aktif.get("lo"):
                st_aktif, al_aktif = evaluasi_validitas_sinyal(sig_aktif, bid_now)
                lbl_eks_aktif, _, info_aktif = cek_status_eksekusi_sinyal(sig_aktif)
                kat_aktif, ket_r_aktif = tentukan_klasifikasi_risiko_sinyal(sig_aktif)
                tag_risk_icon = "⚠️" if kat_aktif == "HIGH RISK" else ("🛡️" if kat_aktif == "LOW RISK" else "⚡")

                print(f"2. 🎯 SINYAL AKTIF TERBARU — [{lbl_eks_aktif}]:")
                print(f"   • Setup Sinyal   : {sig_aktif['arah']} @ {sig_aktif['lo']:.2f} - {sig_aktif['hi']:.2f}")
                print(f"   • Tingkat Risiko : {tag_risk_icon} [{kat_aktif}] ({ket_r_aktif})")
                if kat_aktif == "HIGH RISK":
                    print("   • Pertimbangan   : ⚠️ Sinyal ini dadakan di LTF M5. Anda bebas SKIP / TIDAK DIAMBIL jika ingin santai, ATAU ambil dengan lot mini (0.25%).")
                print(f"   • Stop Loss & TP : SL {sig_aktif['sl']:.2f} | TP1 {sig_aktif['tp1']:.2f} | TP2 {sig_aktif.get('tp2', sig_aktif['tp1']):.2f}")
                print(f"   • Status Saat Ini: [{st_aktif}] -> {al_aktif}")
                print(f"   • Status di Akun : {lbl_eks_aktif} ({info_aktif})")
                info_kentut_st = evaluasi_limit_kena_kentut(sig_aktif, bid_now)
                if info_kentut_st:
                    print(f"   • Note Limit     : [{info_kentut_st['judul']}]")
                    print(f"     (Anti Kentut)    {info_kentut_st['catatan']}")
                mm_s = hitung_mm_sinyal(sig_aktif['lo'], sig_aktif['hi'], sig_aktif['sl'], sig_aktif['tp1'], sig_aktif.get('tp2', 0))
                if mm_s:
                    print(f"   • Rasio R:R      : TP1 1:{mm_s['rrr_tp1']:.1f} | TP2 1:{mm_s['rrr_tp2']:.1f} (Jarak SL: {mm_s['jarak_sl']:.2f} pt)")
                    if kat_aktif == "HIGH RISK":
                        print(f"   • Lot High Risk  : {mm_s.get('lot_025', mm_s['lot_05']):.2f} lot (Risiko mini 0.25%: -${mm_s.get('risk_025_usd', mm_s['risk_05_usd']):.2f}) [batas maksimum bila tetap diambil]")
                    print(f"   • Rekomendasi Lot: 0.5% (Disiplin): {mm_s['lot_05']:.2f} lot (-${mm_s['risk_05_usd']:.2f}) | 1.0% (Standar): {mm_s['lot_10']:.2f} lot (-${mm_s['risk_10_usd']:.2f})")
                    if mm_s.get("melebihi_batas"):
                        print(f"   ⚠️ Lot minimum broker ({mm_s['lot_05']:.2f}) membuat risiko melebihi 0.5% modal — pertimbangkan SL lebih dekat atau lewati setup.")
            else:
                print("2. 🎯 SINYAL AKTIF TERBARU: WAIT / BELUM ADA SETUP (Menunggu konfirmasi jemputan zona).")

            if state_aktif:
                print("\n" + "─" * 70)
                print(f"• Terakhir Disimpan : {state_aktif.get('waktu_simpan', '-')}")
                if state_aktif.get("bawah"):
                    lb, hb, jb = state_aktif["bawah"]
                    print(f"• Zona Demand Bawah : {lb:.2f} - {hb:.2f} (Batas SL Zona: {jb:.2f})")
                if state_aktif.get("atas"):
                    la, ha, ja = state_aktif["atas"]
                    print(f"• Zona Supply Atas  : {la:.2f} - {ha:.2f} (Batas SL Zona: {ja:.2f})")
            print("=" * 70)
            continue

        catatan_jebol_otomatis = None
        if cmd.startswith("pantau") or cmd.startswith("watch"):
            teks_ai_terakhir = next((m["parts"][0]["text"] for m in reversed(riwayat_chat) if m["role"] == "model"), "")
            hasil_pantau = mode_pantau_mt5(tanya, teks_ai_terakhir)
            if hasil_pantau:
                catatan_jebol_otomatis = hasil_pantau
                jawab_refresh = input("\nZona telah ditembus! Ingin AI menjelaskan alasan kenapa jebol & mencari zona baru sekarang? (y/n) [y]: ").strip().lower()
                if jawab_refresh in ("", "y", "ya", "yes"):
                    cmd = "refresh"
                else:
                    continue
            else:
                continue

        if cmd.startswith(("ambil ", "tandai ", "skip ", "batal ambil ", "tidak ambil ")):
            is_batal = cmd.startswith(("skip ", "batal ambil ", "tidak ambil "))
            angka_list = [int(x) for x in re.findall(r"\d+", cmd)]
            if not angka_list and "terakhir" in cmd:
                if os.path.exists(FILE_JURNAL_SINYAL):
                    with open(FILE_JURNAL_SINYAL, "r", encoding="utf-8") as f_cnt:
                        jml_baris = len(list(csv.DictReader(f_cnt)))
                    if jml_baris > 0:
                        angka_list = [jml_baris]
            if not angka_list:
                print("   [ℹ️ Gunakan format: 'ambil <nomor>' (contoh: ambil 7 atau ambil 7,8,20) atau 'skip <nomor>' (contoh: skip 9).]")
                continue
            status_set = "TIDAK (MANUAL)" if is_batal else "YA (MANUAL)"
            ok_t, info_t = tandai_sinyal_diambil_jurnal(angka_list, status_set, mode_by_id=False)
            if ok_t:
                label_aksi = "DIHAPUS dari daftar diambil (⚪ -)" if is_batal else "DITANDAI sebagai ✅ DIAMBIL"
                print(f"\n   [✅ BERHASIL]: Zona {info_t} telah {label_aksi}!")
                tampilkan_rekap()
            else:
                print(f"\n   [⚠️ GAGAL]: {info_t}")
            continue

        if cmd in ("rekap", "statistik", "stat", "jurnal", "journal", "evaluasi", "riwayat sinyal") or any(k in cmd for k in ("jurnal sinyal", "rekap sinyal", "cek jurnal", "tabel sinyal", "evaluasi sinyal", "zona kena sl", "zona kena tp", "mana zona")):
            tampilkan_rekap()
            continue

        if cmd in ("zona", "zone", "scanner", "scan", "smc"):
            try:
                print("\n" + "=" * 70)
                print("🔍 [PYTHON-ONLY XAUUSD ZONE SCANNER (SMC ENGINE)]")
                hasil_scan = scan_xauusd_zones()
                print(hasil_scan.get("summary_text", "Tidak ada data."))
                print("=" * 70)
            except Exception as e_sc:
                print(f"Gagal menjalankan Zone Scanner: {e_sc}")
            continue

        if cmd in ("simpan", "save"):
            nama_file = os.path.join(_DIR_SCRIPT, f"Jurnal_XAUUSD_{datetime.now():%Y%m%d_%H%M%S}.md")
            rows_jurnal, _ = evaluasi_dan_perbarui_jurnal_sinyal()
            tabel_md = susun_tabel_jurnal_markdown(rows_jurnal)
            with open(nama_file, "w", encoding="utf-8") as f:
                f.write(f"# Jurnal Analisis {SYMBOL} — {datetime.now():%Y-%m-%d %H:%M:%S}\n\n")
                f.write("## 📋 Tabel Jurnal Sinyal & Evaluasi Zona (Hit TP vs Hit SL)\n\n")
                f.write(tabel_md)
                f.write("\n---\n\n")
                for msg in riwayat_chat[1:]:
                    role = "AI Analis" if msg["role"] == "model" else "Trader"
                    f.write(f"## {role}\n{msg['parts'][0]['text']}\n\n---\n\n")
            print(f"   Jurnal berhasil disimpan ke: {nama_file}")
            continue

        if cmd in ("refresh", "update"):
            # Jika user ketik refresh manual, cek juga apakah zona dari AI sebelumnya sudah ditembus oleh harga terbaru
            teks_ai_lama = next((m["parts"][0]["text"] for m in reversed(riwayat_chat) if m["role"] == "model"), "")
            target_lama = ekstrak_target_ai(teks_ai_lama)
            try:
                info_jurnal_ref = ringkasan_jurnal_untuk_ai(5)
                prompt_baru = kumpulkan_konteks_mt5(is_refresh=True) + (f"\n\n{info_jurnal_ref}" if info_jurnal_ref else "")
            except KonteksError as e_ctx:
                print(f"\n   Gagal menarik data MT5: {e_ctx}")
                continue

            if not catatan_jebol_otomatis and CATATAN_JEBOL_PANTAU:
                catatan_jebol_otomatis = "\n".join(CATATAN_JEBOL_PANTAU)
                CATATAN_JEBOL_PANTAU.clear()

            if not catatan_jebol_otomatis and any(target_lama.values()):
                m_harga = re.search(r"Bid\s+(\d+(?:\.\d+)?)", prompt_baru)
                if m_harga:
                    h_skrg = float(m_harga.group(1))
                    if target_lama["atas"] and h_skrg > target_lama["atas"][2]:
                        catatan_jebol_otomatis = f"Zona Supply Atas sebelumnya ({target_lama['atas'][0]:.2f}-{target_lama['atas'][1]:.2f}) telah JEBOL ke atas (Harga sekarang {h_skrg:.2f} > batas {target_lama['atas'][2]:.2f})."
                    elif target_lama["bawah"] and h_skrg < target_lama["bawah"][2]:
                        catatan_jebol_otomatis = f"Zona Demand Bawah sebelumnya ({target_lama['bawah'][0]:.2f}-{target_lama['bawah'][1]:.2f}) telah JEBOL ke bawah (Harga sekarang {h_skrg:.2f} < batas {target_lama['bawah'][2]:.2f})."
                    elif target_lama["sinyal"]:
                        s = target_lama["sinyal"]
                        if (s["arah"] == "BUY" and h_skrg <= s["sl"]) or (s["arah"] == "SELL" and h_skrg >= s["sl"]):
                            catatan_jebol_otomatis = f"Sinyal {s['arah']} sebelumnya ({s['lo']:.2f}-{s['hi']:.2f}) telah BATAL / JEBOL SL ({s['sl']:.2f}) karena harga sekarang di {h_skrg:.2f}."

            if catatan_jebol_otomatis:
                prompt_baru += (
                    f"\n\n[PENTING — EVALUASI FORENSIK ZONA SEBELUMNYA YANG JEBOL]:\n{catatan_jebol_otomatis}\n"
                    "INSTRUKSI KHUSUS: Pada bagian paling atas jawabanmu, buat bagian berjudul "
                    "'🔍 EVALUASI KENAPA ZONA SEBELUMNYA JEBOL (POST-BREAKOUT AUTOPSY)'. "
                    "Jelaskan secara objektif dan detail dari data terbaru: kenapa zona tersebut gagal menahan harga "
                    "(apakah karena lonjakan volume/orderan besar institusi, dampak korelasi USD/berita, ketiadaan CHoCH konfirmasi di M5, atau likuiditas di balik zona yang sedang diincar), "
                    "serta apakah penjebolan ini adalah VALID BREAKOUT (sehingga zona lama berubah menjadi Breaker/Flip Zone untuk dicari Retest-nya) atau FAKEOUT/LIQUIDITY SWEEP!"
                )

            print("   Mengirim data live ke AI (streaming real-time)...", end="\r")
            jwb_baru, mdl = panggil_gemini(client, prompt_baru, stream_output=True)
            if jwb_baru:
                _tgt_baru = ekstrak_target_ai(jwb_baru)
                _cand_baru = _tgt_baru.get("sinyal")
                if _cand_baru and mdl:
                    _cand_baru["model"] = mdl
                sinyal_kandidat = terapkan_gerbang(_cand_baru)
                if sinyal_kandidat and mdl:
                    sinyal_kandidat["model"] = mdl
                tick_ref = mt5.symbol_info_tick(SYMBOL) if pastikan_mt5_aktif() else None
                bid_ref = tick_ref.bid if tick_ref else 0.0
                st_aktif_ref = "UNKNOWN"
                if SINYAL_AKTIF_DINAMIS and isinstance(SINYAL_AKTIF_DINAMIS, dict) and bid_ref > 0:
                    st_aktif_ref, _ = evaluasi_validitas_sinyal(SINYAL_AKTIF_DINAMIS, bid_ref)

                # 🔒 SIGNAL & DIRECTIONAL LOCK via helper terpusat (konteks REFRESH)
                sig_pilih, sig_rw, logs_lock, diubah = putuskan_sinyal_terkunci(
                    sinyal_kandidat,
                    SINYAL_AKTIF_DINAMIS,
                    status_aktif=st_aktif_ref,
                    paksa_ganti=False,
                    izinkan_balik_arah=False,
                    konteks="REFRESH",
                )
                for p in logs_lock:
                    print(p)
                if sig_rw:
                    # Pengarsipan di Refresh adalah perubahan perilaku yang disengaja (jika sinyal lama diganti karena selesai/batal)
                    SINYAL_SEBELUMNYA_DINAMIS = sig_rw

                SINYAL_AKTIF_DINAMIS = sig_pilih
                if SINYAL_AKTIF_DINAMIS and isinstance(SINYAL_AKTIF_DINAMIS, dict):
                    if SINYAL_AKTIF_DINAMIS["arah"] == "BUY":
                        ZONA_BAWAH_DINAMIS = (SINYAL_AKTIF_DINAMIS["lo"], SINYAL_AKTIF_DINAMIS["hi"], SINYAL_AKTIF_DINAMIS["sl"])
                        ZONA_ATAS_DINAMIS = _tgt_baru.get("atas") or ZONA_ATAS_DINAMIS
                    else:
                        ZONA_ATAS_DINAMIS = (SINYAL_AKTIF_DINAMIS["lo"], SINYAL_AKTIF_DINAMIS["hi"], SINYAL_AKTIF_DINAMIS["sl"])
                        ZONA_BAWAH_DINAMIS = _tgt_baru.get("bawah") or ZONA_BAWAH_DINAMIS
                else:
                    ZONA_ATAS_DINAMIS = _tgt_baru.get("atas")
                    ZONA_BAWAH_DINAMIS = _tgt_baru.get("bawah")
                simpan_state_sinyal(SINYAL_AKTIF_DINAMIS, SINYAL_SEBELUMNYA_DINAMIS, atas=ZONA_ATAS_DINAMIS, bawah=ZONA_BAWAH_DINAMIS, catatan="Refresh Manual")
                riwayat_chat = [
                    {"role": "user", "parts": [{"text": prompt_baru}]},
                    {"role": "model", "parts": [{"text": jwb_baru}]},
                ]
            else:
                print("\nGagal memperbarui analisa (server padat). Coba lagi sebentar.")
            continue

        tanya_lower = tanya.lower()
        kata_kunci_sinyal = (
            "sinyal baru", "setup baru", "ganti sinyal", "reset sinyal",
            "buatkan sinyal", "minta sinyal", "minta setup", "rekomendasi sinyal",
            "cari sinyal baru", "cari setup baru", "buat sinyal baru"
        )
        minta_baru = any(k in tanya_lower for k in kata_kunci_sinyal)
        tanya_posisi = any(k in tanya_lower for k in (
            "posisi", "floating", "order", "pending", "tiket", "ticket",
            "lot", "akun", "saldo", "balance", "equity", "bep"
        ))

        live_snap = snapshot_live_qa()
        teks_kunci_qa = ""
        if SINYAL_AKTIF_DINAMIS and isinstance(SINYAL_AKTIF_DINAMIS, dict):
            sd_q = SINYAL_AKTIF_DINAMIS
            tick_q0 = mt5.symbol_info_tick(SYMBOL) if pastikan_mt5_aktif() else None
            bid_q0 = tick_q0.bid if tick_q0 else 0.0
            st_q0, al_q0 = evaluasi_validitas_sinyal(sd_q, bid_q0)
            if sinyal_masih_aktif(st_q0):
                teks_kunci_qa = (
                    f"\n[STATUS SINYAL AKTIF TERKUNCI]:\n"
                    f"- Sinyal Saat Ini: {sd_q['arah']} @ {sd_q['lo']:.2f}-{sd_q['hi']:.2f} | SL {sd_q['sl']:.2f} | TP1 {sd_q['tp1']:.2f} | TP2 {sd_q.get('tp2', sd_q['tp1']):.2f}\n"
                    f"- Status Validitas: [{st_q0}] -> {al_q0}\n"
                    f"- Jaga konsistensi: jangan menggeser/mengganti angka sinyal aktif ini kecuali trader meminta 'ganti sinyal' / 'sinyal baru'.\n"
                )

        if minta_baru:
            instruksi_qa = (
                f"Pertanyaan trader: \"{tanya}\"\n\n"
                f"{live_snap}\n"
                f"{teks_kunci_qa}\n"
                "🛡️ INSTRUKSI JAWABAN (PERMINTAAN SETUP / SINYAL BARU):\n"
                "1. Trader secara eksplisit meminta setup/sinyal baru.\n"
                "2. Periksa status Python Zone Scanner & konfirmasi multi-timeframe di atas. JIKA status pasar WAIT atau terdapat konflik tren HTF (misal H4 Bullish vs H1 Bearish): JELASKAN secara singkat, tegas, dan padat bahwa belum ada sinyal valid, pasar dalam fase WAIT, dan sebutkan zona mana yang paling logis untuk ditunggu. DILARANG memaksakan sinyal entry jika syarat belum terpenuhi!\n"
                "3. JIKA dan HANYA JIKA ada setup yang lolos validasi penuh: tuliskan rencana setup secara ringkas dan WAJIB sertakan blok [RADAR_TARGET] di baris paling akhir jawaban.\n"
                "4. Jawab langsung to the point tanpa basa-basi dan tanpa template A/B/C."
            )
        else:
            instruksi_qa = (
                f"Pertanyaan trader: \"{tanya}\"\n\n"
                f"{live_snap}\n"
                f"{teks_kunci_qa}\n"
                "🛡️ INSTRUKSI JAWABAN (MODE TANYA JAWAB CEPAT & TO THE POINT):\n"
                "1. JAWAB LANGSUNG KE INTI PERTANYAAN (MAKSIMAL 1-3 PARAGRAF RINGKAS, PADAT, DAN JELAS).\n"
                "2. DILARANG KERAS MENGGUNAKAN FORMAT TEMPLATE: DILARANG membuat bagian A, B, C; DILARANG menulis header '### A. Status Order Flow...', '### B...', dsb.\n"
                "3. DILARANG MEMBUAT BOX [SINYAL ...] DAN DILARANG MENYERTAKAN BLOK [RADAR_TARGET]! Trader HANYA bertanya biasa seputar market/zona/kondisi teknikal, BUKAN meminta sinyal baru.\n"
                f"{'4. Jelaskan status posisi/order/akun trader secara spesifik sesuai pertanyaannya.' if tanya_posisi else '4. DILARANG mengulang audit pending order atau saldo akun karena trader tidak bertanya tentang akun/posisi.'}\n"
                "5. Jika trader menanyakan zona teknikal tertentu (misal: 'zona X masih valid ga dan ada apa disitu' atau 'apakah ada zona buy yang valid'):\n"
                "   - Jelaskan langsung apa yang ada di area harga tersebut (misal: FVG H1, Order Block, Swing High/Low, Liquidity Pool, dsb. dari data MT5 di atas).\n"
                "   - Jelaskan apakah zona itu valid atau tidak berdasar bias HTF dan struktur terkini.\n"
                "   - Berikan arahan singkat: apakah layak ditunggu atau sebaiknya WAIT.\n"
                "6. Gunakan gaya bahasa profesional, lugas, santai namun presisi seperti rekan trader ahli (tanpa basa-basi)."
            )

        # Untuk sesi Tanya Jawab, gunakan hanya percakapan Q&A terakhir (tanpa prompt startup yang membawa template A/B/C)
        percakapan_qa = [m for m in riwayat_chat[2:]] if len(riwayat_chat) > 2 else []
        if len(percakapan_qa) > 6:
            percakapan_qa = percakapan_qa[-6:]
        konten_qa = percakapan_qa + [{"role": "user", "parts": [{"text": instruksi_qa}]}]

        print("   Menghubungi server AI...", end="\r")
        jawaban_qa, mdl = panggil_gemini(client, konten_qa, stream_output=True)
        if jawaban_qa:
            if minta_baru:
                _tgt_qa = ekstrak_target_ai(jawaban_qa)
                _cand_qa = _tgt_qa.get("sinyal")
                if _cand_qa and mdl:
                    _cand_qa["model"] = mdl
                sig_qa = terapkan_gerbang(_cand_qa)
                if sig_qa and mdl:
                    sig_qa["model"] = mdl
                izinkan_balik = any(k in tanya_lower for k in ("balik arah", "reversal", "ganti sell", "ganti buy", "switch"))
                tick_q = mt5.symbol_info_tick(SYMBOL) if pastikan_mt5_aktif() else None
                bid_q = tick_q.bid if tick_q else 0.0
                st_q_curr = "UNKNOWN"
                if SINYAL_AKTIF_DINAMIS and isinstance(SINYAL_AKTIF_DINAMIS, dict) and bid_q > 0:
                    st_q_curr, _ = evaluasi_validitas_sinyal(SINYAL_AKTIF_DINAMIS, bid_q)

                if sig_qa:
                    # 🔒 SIGNAL & DIRECTIONAL LOCK via helper terpusat (konteks QA)
                    sig_pilih, sig_rw, logs_lock, diubah = putuskan_sinyal_terkunci(
                        sig_qa,
                        SINYAL_AKTIF_DINAMIS,
                        status_aktif=st_q_curr,
                        paksa_ganti=minta_baru,
                        izinkan_balik_arah=izinkan_balik,
                        konteks="QA",
                    )
                    for p in logs_lock:
                        print(p)
                    if sig_rw:
                        SINYAL_SEBELUMNYA_DINAMIS = sig_rw
                    if diubah:
                        SINYAL_AKTIF_DINAMIS = sig_pilih
                        if SINYAL_AKTIF_DINAMIS["arah"] == "BUY":
                            ZONA_BAWAH_DINAMIS = (SINYAL_AKTIF_DINAMIS["lo"], SINYAL_AKTIF_DINAMIS["hi"], SINYAL_AKTIF_DINAMIS["sl"])
                            ZONA_ATAS_DINAMIS = _tgt_qa.get("atas") or ZONA_ATAS_DINAMIS
                        else:
                            ZONA_ATAS_DINAMIS = (SINYAL_AKTIF_DINAMIS["lo"], SINYAL_AKTIF_DINAMIS["hi"], SINYAL_AKTIF_DINAMIS["sl"])
                            ZONA_BAWAH_DINAMIS = _tgt_qa.get("bawah") or ZONA_BAWAH_DINAMIS
                        simpan_state_sinyal(SINYAL_AKTIF_DINAMIS, SINYAL_SEBELUMNYA_DINAMIS, ZONA_ATAS_DINAMIS, ZONA_BAWAH_DINAMIS, catatan=f"Update QA: {tanya[:30]}")
            else:
                # Tanya jawab biasa: bersihkan status penolakan parser agar tidak bocor ke giliran berikutnya
                _STATE_PARSER["ditolak"] = []

            riwayat_chat.append({"role": "user", "parts": [{"text": f"Pertanyaan trader: {tanya}"}]})
            riwayat_chat.append({"role": "model", "parts": [{"text": jawaban_qa}]})
        else:
            print("   Maaf, server AI sedang sibuk. Silakan kirim ulang pertanyaan Anda.")



if __name__ == "__main__":
    try:
        main()
    finally:
        _tutup_mt5()
