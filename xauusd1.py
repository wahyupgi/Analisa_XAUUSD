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
    state_sebelumnya = muat_state_sinyal() or {}

    # Hubungkan MT5 lebih awal agar evaluasi harga real-time akurat
    mt5_terhubung = pastikan_mt5_aktif()
    tick_cek = mt5.symbol_info_tick(SYMBOL) if mt5_terhubung else None
    bid_cek = tick_cek.bid if tick_cek else 0.0

    # Ambil kandidat sinyal dari state sebelumnya (prioritaskan sinyal aktif, lalu sinyal_sebelumnya)
    cand_s = state_sebelumnya.get("sinyal")
    if not (cand_s and isinstance(cand_s, dict) and cand_s.get("lo")):
        cand_s = state_sebelumnya.get("sinyal_sebelumnya")

    sig_sebelumnya = None
    if cand_s and isinstance(cand_s, dict) and cand_s.get("lo"):
        status_lama, alasan_lama = evaluasi_validitas_sinyal(cand_s, bid_cek)
        if sinyal_masih_aktif(status_lama):
            sig_sebelumnya = cand_s
            SINYAL_AKTIF_DINAMIS = sig_sebelumnya
            SINYAL_SEBELUMNYA_DINAMIS = sig_sebelumnya
        else:
            simpan_state_sinyal(None, None, state_sebelumnya.get("atas"), state_sebelumnya.get("bawah"), "Reset Sinyal Invalid", hapus_riwayat=True)
            SINYAL_AKTIF_DINAMIS = None
            SINYAL_SEBELUMNYA_DINAMIS = None
            sig_sebelumnya = None

    # Status Komponen Sistem
    mt5_ok = bool(mt5_terhubung and mt5.terminal_info() is not None)
    tick_live = mt5.symbol_info_tick(SYMBOL) if mt5_ok else None
    engine_ok = True

    client = None
    ai_status = "DISCONNECTED"
    if API_KEY:
        try:
            client = genai.Client(
                api_key=API_KEY,
                http_options=types.HttpOptions(
                    timeout=25000,
                    retry_options=types.HttpRetryOptions(attempts=1),
                ),
            )
            ai_status = "READY"
        except Exception as e_ai:
            ai_status = f"ERROR ({type(e_ai).__name__})"
            client = None
    else:
        ai_status = "DISCONNECTED"

    # TAMPILKAN STARTUP AWAL RINGKAS (4 BAGIAN: HEADER, SYSTEM STATUS, MARKET OVERVIEW, COMMAND MENU)
    print(format_startup_presentation(
        symbol=SYMBOL,
        mt5_connected=mt5_ok,
        tick=tick_live,
        engine_ready=engine_ok,
        ai_status=ai_status,
        version=APP_VERSION,
    ))

    # Sinkronisasi target zona awal Python Scanner ke state jika belum diset
    if not ZONA_ATAS_DINAMIS or not ZONA_BAWAH_DINAMIS:
        try:
            _tgt_init = ekstrak_target_ai("")
            if not ZONA_ATAS_DINAMIS and _tgt_init.get("atas"):
                ZONA_ATAS_DINAMIS = _tgt_init["atas"]
            if not ZONA_BAWAH_DINAMIS and _tgt_init.get("bawah"):
                ZONA_BAWAH_DINAMIS = _tgt_init["bawah"]
        except Exception:
            pass

    riwayat_chat = []

    while True:
        try:
            tanya = input("\n[XAUUSD SMC] > ").strip()
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
            print("📊 [MONITORING STATUS SINYAL]")
            print(f"• Harga Bid MT5 Saat Ini: {bid_now:.2f}")

            # Evaluasi apakah ada sinyal yang benar-benar MASIH VALID
            sinyal_ditampilkan = None
            if sig_aktif and isinstance(sig_aktif, dict) and sig_aktif.get("lo"):
                st_aktif, al_aktif = evaluasi_validitas_sinyal(sig_aktif, bid_now)
                if sinyal_masih_aktif(st_aktif):
                    sinyal_ditampilkan = (sig_aktif, st_aktif, al_aktif)
                else:
                    # Sinyal sudah tidak valid -> bersihkan dari memori
                    simpan_state_sinyal(None, None, state_aktif.get("atas") if state_aktif else None, state_aktif.get("bawah") if state_aktif else None, "Reset Sinyal Invalid", hapus_riwayat=True)
                    SINYAL_AKTIF_DINAMIS = None

            if not sinyal_ditampilkan and sig_lama and isinstance(sig_lama, dict) and sig_lama.get("lo"):
                st_lama, al_lama = evaluasi_validitas_sinyal(sig_lama, bid_now)
                if sinyal_masih_aktif(st_lama):
                    sinyal_ditampilkan = (sig_lama, st_lama, al_lama)
                else:
                    simpan_state_sinyal(None, None, state_aktif.get("atas") if state_aktif else None, state_aktif.get("bawah") if state_aktif else None, "Reset Sinyal Invalid", hapus_riwayat=True)
                    SINYAL_SEBELUMNYA_DINAMIS = None

            if sinyal_ditampilkan:
                sig_t, st_t, al_t = sinyal_ditampilkan
                lbl_eks_t, _, info_eks_t = cek_status_eksekusi_sinyal(sig_t)
                kat_t, ket_r_t = tentukan_klasifikasi_risiko_sinyal(sig_t)
                tag_risk_icon = "⚠️" if kat_t == "HIGH RISK" else ("🛡️" if kat_t == "LOW RISK" else "⚡")

                print("\n" + "─" * 70)
                print(f"🎯 SINYAL AKTIF (MASIH VALID) — [{lbl_eks_t}]:")
                print(f"   • Setup Sinyal   : {sig_t['arah']} @ {sig_t['lo']:.2f} - {sig_t['hi']:.2f}")
                print(f"   • Tingkat Risiko : {tag_risk_icon} [{kat_t}] ({ket_r_t})")
                if kat_t == "HIGH RISK":
                    print("   • Pertimbangan   : ⚠️ Sinyal ini dadakan di LTF M5. Anda bebas SKIP / TIDAK DIAMBIL jika ingin santai, ATAU ambil dengan lot mini (0.25%).")
                print(f"   • Stop Loss & TP : SL {sig_t['sl']:.2f} | TP1 {sig_t['tp1']:.2f} | TP2 {sig_t.get('tp2', sig_t['tp1']):.2f}")
                print(f"   • Status Saat Ini: [{st_t}] -> {al_t}")
                print(f"   • Status di Akun : {lbl_eks_t} ({info_eks_t})")
                info_kentut_st = evaluasi_limit_kena_kentut(sig_t, bid_now)
                if info_kentut_st:
                    print(f"   • Note Limit     : [{info_kentut_st['judul']}]")
                    print(f"     (Anti Kentut)    {info_kentut_st['catatan']}")
                mm_s = hitung_mm_sinyal(sig_t['lo'], sig_t['hi'], sig_t['sl'], sig_t['tp1'], sig_t.get('tp2', 0))
                if mm_s:
                    print(f"   • Rasio R:R      : TP1 1:{mm_s['rrr_tp1']:.1f} | TP2 1:{mm_s['rrr_tp2']:.1f} (Jarak SL: {mm_s['jarak_sl']:.2f} pt)")
                    if kat_t == "HIGH RISK":
                        print(f"   • Lot High Risk  : {mm_s.get('lot_025', mm_s['lot_05']):.2f} lot (Risiko mini 0.25%: -${mm_s.get('risk_025_usd', mm_s['risk_05_usd']):.2f}) [batas maksimum bila tetap diambil]")
                    print(f"   • Rekomendasi Lot: 0.5% (Konservatif): {mm_s['lot_05']:.2f} lot (-${mm_s['risk_05_usd']:.2f}) | 1.0% (Risiko Lebih Tinggi): {mm_s['lot_10']:.2f} lot (-${mm_s['risk_10_usd']:.2f})")
                    if mm_s.get("melebihi_batas"):
                        print(f"   ⚠️ Lot minimum broker ({mm_s['lot_05']:.2f}) membuat risiko melebihi 0.5% modal — pertimbangkan SL lebih dekat atau lewati setup.")
            else:
                print("\n" + "─" * 70)
                print("🎯 STATUS SINYAL: WAIT / BELUM ADA SETUP AKTIF (Menunggu konfirmasi jemputan zona baru).")
                print("   (Sinyal sebelumnya yang sudah tidak valid/selesai telah dibersihkan).")

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

        if cmd in ("scan detail", "scanner detail", "zona detail", "zone detail"):
            try:
                hasil_scan = scan_xauusd_zones()
                dec = evaluate_trade_decision()
                print("\n" + format_scan_presentation(hasil_scan, dec, detailed=True))
            except Exception as e_sc:
                print(f"Gagal menjalankan Zone Scanner Detail: {e_sc}")
            continue

        if cmd in ("zona", "zone", "scanner", "scan", "smc"):
            try:
                hasil_scan = scan_xauusd_zones()
                dec = evaluate_trade_decision()
                print("\n" + format_scan_presentation(hasil_scan, dec, detailed=False))
            except Exception as e_sc:
                print(f"Gagal menjalankan Zone Scanner: {e_sc}")
            continue

        if cmd in ("audit", "audit mt5", "cek akun", "cek order", "audit order"):
            try:
                print("\n" + "=" * 70)
                print("AUDIT FAKTA AKUN MT5, PENDING ORDER & VOLATILITAS PASAR")
                print("=" * 70)
                info_mt5 = cek_posisi_terbuka_mt5(SYMBOL)
                print(info_mt5)
                try:
                    df_d1 = ambil_data(SYMBOL, "D1", 20)
                    if df_d1 is not None and len(df_d1) >= 14:
                        ref_price = float(df_d1["Close"].iloc[-1])
                        adr_text = analisa_adr_exhaustion(df_d1, ref_price)
                        print(adr_text)
                except Exception as e_adr:
                    print(f"ADR Info tidak dapat dimuat: {e_adr}")
                print("=" * 70)
            except Exception as e_aud:
                print(f"Gagal melakukan audit MT5: {e_aud}")
            continue

        if cmd in ("diag", "diagnostik", "diagnostic", "scan diag"):
            try:
                hasil_scan = scan_xauusd_zones()
                print("\n" + format_diagnostic_report(hasil_scan))
            except Exception as e_diag:
                print(f"Gagal menjalankan diagnostik scanner: {e_diag}")
            continue

        if cmd in ("help", "bantuan", "menu", "?", "panduan"):
            print("\n" + "=" * 70)
            print("📖 DAFTAR PERINTAH XAUUSD SMC SCANNER")
            print("=" * 70)
            print("1. PERINTAH SCAN & ANALISIS:")
            print("   • scan         : Ringkasan 5 bagian (Kondisi Pasar, Radar Zone, Entry Plan,")
            print("                    Keputusan Resmi Python Scanner, Risk Warning)")
            print("   • scan detail  : Analisis lengkap seluruh kandidat zona HTF & detail teknis")
            print("   • zona         : Cek status zona supply & demand aktif")
            print("   • diag         : Diagnostik lengkap filter dan zona yang terdeteksi/ditolak")
            print("\n2. PERINTAH PEMANTAUAN & EKSEKUSI:")
            print("   • pantau       : Mode pemantauan real-time MT5 hingga zona tersentuh")
            print("   • sinyal       : Cek status sinyal aktif yang tersimpan di sistem")
            print("   • audit        : Audit fakta akun MT5, floating order, pending order & ADR")
            print("\n3. PERINTAH JURNAL & MANAJEMEN:")
            print("   • jurnal       : Rekap riwayat sinyal, winrate, dan evaluasi performa")
            print("   • ambil <no>   : Tandai sinyal pada jurnal sebagai diambil")
            print("   • skip <no>    : Batalkan tanda diambil pada sinyal jurnal")
            print("   • simpan       : Simpan hasil analisis & jurnal ke file Markdown")
            print("   • refresh      : Perbarui analisis dengan data market MT5 terkini")
            print("\n4. SISTEM:")
            print("   • help         : Menampilkan panduan dan daftar perintah ini")
            print("   • exit         : Keluar dari aplikasi (q / quit / keluar)")
            print("=" * 70)
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
            if client is None:
                print("\n   [INFO]: Gemini AI tidak aktif. Memperbarui data market dan Python Scanner...")
                try:
                    hasil_scan = scan_xauusd_zones()
                    dec = evaluate_trade_decision()
                    print("\n" + format_scan_presentation(hasil_scan, dec, detailed=False))
                except Exception as e_rf:
                    print(f"Gagal memperbarui scanner: {e_rf}")
                continue

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
                jwb_baru = sanitasi_output_gemini(jwb_baru)
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

        # Jika Gemini AI offline, infokan kepada trader dan arahkan ke perintah lokal
        if client is None:
            print("\n   [INFO]: Layanan Gemini AI tidak aktif (GEMINI_API_KEY belum dikonfigurasi atau offline).")
            print("   Gunakan perintah lokal scanner seperti 'scan', 'zona', 'pantau', 'jurnal', 'audit', atau 'help'.")
            continue

        # Untuk sesi Tanya Jawab, gunakan hanya percakapan Q&A terakhir (hingga 6 giliran terakhir)
        percakapan_qa = list(riwayat_chat[-6:]) if riwayat_chat else []
        konten_qa = percakapan_qa + [{"role": "user", "parts": [{"text": instruksi_qa}]}]

        print("   Menghubungi server AI...", end="\r")
        jawaban_qa, mdl = panggil_gemini(client, konten_qa, stream_output=True)
        if jawaban_qa:
            jawaban_qa = sanitasi_output_gemini(jawaban_qa)
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
