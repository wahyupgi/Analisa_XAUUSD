# -*- coding: utf-8 -*-
"""
modules/jurnal_laporan.py
Laporan jurnal sinyal: tabel markdown, ringkasan untuk AI, dan tampilkan_rekap.

Dipindahkan apa adanya dari modules/sinyal_jurnal.py (structural refactoring, tanpa perubahan logika).
Modul ini DIMUAT dari bagian akhir modules.sinyal_jurnal (compatibility re-export); impor selalu lewat modul tersebut
agar tidak terjadi circular import.
"""
import csv
import os
import re
from modules.config import FILE_JURNAL_SINYAL
from modules.sinyal_jurnal import (
    _migrasi_header_jurnal,
    evaluasi_dan_perbarui_jurnal_sinyal,
)
from modules.terminal_ui import print_terminal


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
        print_terminal("\n   Belum ada sinyal tercatat di jurnal CSV.")
        return

    lebar_garis = 138
    print_terminal("\n" + "=" * lebar_garis)
    print_terminal("📖 JURNAL SINYAL & EVALUASI ZONA (ACUAN HASIL HIT TP vs HIT SL + PENANDA ZONA YANG DIAMBIL)")
    print_terminal("=" * lebar_garis)
    print_terminal(f"{'#':<3} | {'Waktu Lokal':<16} | {'Arah':<4} | {'Zona Entry':<19} | {'SL':<8} | {'TP1 / TP2':<19} | {'Diambil?':<14} | {'Status Hasil':<16} | {'Hasil R':<8} | {'Setup / Keterangan'}")
    print_terminal("-" * lebar_garis)

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

        print_terminal(f"{idx:>3} | {wkt_str:<16} | {arah:<4} | {zona_str:<19} | {sl:>8.2f} | {tp_str:<19} | {ambil_col:<13} | {ikon_st:<16} | {r_display} | {ket_pendek}")

    print_terminal("=" * lebar_garis)
    wr_total = f"{(tot_menang / tot_trade * 100):.1f}%" if tot_trade > 0 else "-"
    exp_total = f"{(tot_r / tot_trade):+.2f}R" if tot_trade > 0 else "-"
    print_terminal("📊 RINGKASAN PERFORMA JURNAL SINYAL (SEMUA SINYAL):")
    print_terminal(f"• Total Sinyal Tercatat : {tot_sinyal} sinyal")
    print_terminal(f"• Selesai (Trade Closed): {tot_trade} trade (🟢 Menang: {tot_menang} | 🔴 Kalah: {tot_kalah})")
    print_terminal(f"• Win-Rate Keseluruhan  : {wr_total}")
    print_terminal(f"• Akumulasi Profit/Loss : {tot_r:+.2f}R")
    print_terminal(f"• Expectancy per Trade  : {exp_total} / trade")
    print_terminal(f"• Posisi Berjalan/Wait  : {tot_running} Running | {tot_terlewat} Terlewat/Belum Terisi")

    print_terminal("-" * lebar_garis)
    wr_ambil = f"{(ambil_menang / ambil_trade * 100):.1f}%" if ambil_trade > 0 else "-"
    daftar_ambil_str = ", ".join(ambil_nomor_list) if ambil_nomor_list else "Belum ada zona yang ditandai diambil"
    print_terminal("🎯 RINGKASAN KHUSUS ZONA YANG ANDA AMBIL (✅ DIAMBIL):")
    print_terminal(f"• Zona Diambil Tercatat : {ambil_n} zona -> {daftar_ambil_str}")
    print_terminal(f"• Hasil Eksekusi Anda   : Selesai {ambil_trade} trade (🟢 Menang: {ambil_menang} | 🔴 Kalah: {ambil_kalah}) | Win-Rate Anda: {wr_ambil} | Total Hasil: {ambil_r:+.2f}R")
    print_terminal("  💡 Cara Menandai Manual: Ketik 'ambil <nomor>' (contoh: ambil 7 atau ambil 7,8,20) untuk menandai zona yang Anda ambil,")
    print_terminal("                          atau ketik 'skip <nomor>' (contoh: skip 9) untuk menghapus tanda diambil.")
    if not mt5_aktif:
        print_terminal("  *(MT5 saat ini offline; status di atas diambil dari riwayat jurnal tersimpan terakhir)*")

    print_terminal("-" * lebar_garis)
    print_terminal("BREAKDOWN PER KELAS RISIKO:")
    for nama_k, stat_k in sorted(kelas_stat.items()):
        tr_k = stat_k["menang"] + stat_k["kalah"]
        wr_k = f"{(stat_k['menang'] / tr_k * 100):.0f}%" if tr_k > 0 else "-"
        ex_k = f"{(stat_k['total_r'] / tr_k):+.2f}R" if tr_k > 0 else "-"
        print_terminal(f"• {nama_k:<12} : {stat_k['n']:>2} sinyal | Selesai {tr_k:>2} (Menang {stat_k['menang']}, Kalah {stat_k['kalah']}) | Win-Rate {wr_k} | Total {stat_k['total_r']:+.2f}R | Exp {ex_k}/tr | Running {stat_k['running']}")

    if len(model_stat) > 1 or any(k != "TIDAK DIKETAHUI" for k in model_stat):
        print_terminal("-" * lebar_garis)
        print_terminal("BREAKDOWN PER MODEL AI:")
        for nama_m, stat_m in sorted(model_stat.items()):
            tr_m = stat_m["menang"] + stat_m["kalah"]
            wr_m = f"{(stat_m['menang'] / tr_m * 100):.0f}%" if tr_m > 0 else "-"
            ex_m = f"{(stat_m['total_r'] / tr_m):+.2f}R" if tr_m > 0 else "-"
            print_terminal(f"• {nama_m:<25} : {stat_m['n']:>2} sinyal | Selesai {tr_m:>2} (Menang {stat_m['menang']}, Kalah {stat_m['kalah']}) | Win-Rate {wr_m} | Total {stat_m['total_r']:+.2f}R | Exp {ex_m}/tr")
    print_terminal("=" * lebar_garis)
