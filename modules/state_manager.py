# -*- coding: utf-8 -*-
"""
modules/state_manager.py
Penyimpanan dan pemulihan memori sinyal (persistent state) antar-sesi program.
Menjaga konsistensi Sinyal Aktif, Sinyal Riwayat Sebelumnya, dan Zona Dinamis.
"""
import json
import os
from datetime import datetime

from modules.config import FILE_STATE_SINYAL, FILE_ERROR_LOG
from modules.utils import _catat_error
from modules.sinyal_jurnal import _stempel_sinyal

# Menyimpan Sinyal Aktif Dinamis & Zona Dinamis yang otomatis berubah saat Mode Pantau mendeteksi peringatan / SL jebol
SINYAL_AKTIF_DINAMIS = None
SINYAL_SEBELUMNYA_DINAMIS = None
ZONA_ATAS_DINAMIS = None
ZONA_BAWAH_DINAMIS = None
CATATAN_JEBOL_PANTAU = []
_STATE_PARSER = {"ditolak": []}


def simpan_state_sinyal(sinyal, sinyal_sebelumnya=None, atas=None, bawah=None, catatan="", hapus_riwayat=False):
    """Menyimpan status sinyal aktif & sinyal sebelumnya ke file lokal secara permanen agar riwayat tidak pernah hilang."""
    global SINYAL_SEBELUMNYA_DINAMIS, SINYAL_AKTIF_DINAMIS

    # Jika trader meminta hapus riwayat / reset sinyal invalid
    if hapus_riwayat:
        sinyal = None if not (sinyal and isinstance(sinyal, dict) and sinyal.get("lo")) else sinyal
        sinyal_sebelumnya = None
        SINYAL_AKTIF_DINAMIS = sinyal
        SINYAL_SEBELUMNYA_DINAMIS = None

    # Deteksi jika dipanggil secara legacy: simpan_state_sinyal(sinyal, atas, bawah, catatan)
    # Cirinya: argumen ke-2 (sinyal_sebelumnya) adalah tuple/list zona [lo, hi, jebol], bukan dict sinyal.
    elif isinstance(sinyal_sebelumnya, (tuple, list)):
        catatan = bawah if isinstance(bawah, str) else catatan
        bawah = atas
        atas = sinyal_sebelumnya
        sinyal_sebelumnya = None

    # Pastikan sinyal_sebelumnya jika ada HARUS berupa dict, bukan list/tuple/string
    if sinyal_sebelumnya is not None and not isinstance(sinyal_sebelumnya, dict):
        sinyal_sebelumnya = None

    # Baca file lama terlebih dahulu agar tidak menimpa data yang sudah ada dengan null
    data_lama = muat_state_sinyal() or {}
    if sinyal and isinstance(sinyal, dict) and sinyal.get("lo") and not sinyal.get("t_buat"):
        _stempel_sinyal(sinyal, data_lama, catatan)   # in-place: objek SINYAL_AKTIF_DINAMIS ikut mendapat cap waktu
    riwayat_lama = data_lama.get("sinyal_sebelumnya")
    if not isinstance(riwayat_lama, dict):
        riwayat_lama = None
    sinyal_aktif_lama = data_lama.get("sinyal")
    if not isinstance(sinyal_aktif_lama, dict):
        sinyal_aktif_lama = None

    if not hapus_riwayat:
        # Jika sinyal baru adalah dict valid (setup nyata):
        if sinyal and isinstance(sinyal, dict) and sinyal.get("lo"):
            # Jika sinyal aktif lama juga valid dan angkanya berbeda, jadikan sinyal lama sebagai riwayat
            if sinyal_aktif_lama and (sinyal_aktif_lama.get("lo") != sinyal.get("lo") or sinyal_aktif_lama.get("arah") != sinyal.get("arah")):
                sinyal_sebelumnya = sinyal_aktif_lama
            elif sinyal_sebelumnya is None:
                sinyal_sebelumnya = riwayat_lama or (SINYAL_SEBELUMNYA_DINAMIS if isinstance(SINYAL_SEBELUMNYA_DINAMIS, dict) else None)
        else:
            # Jika sinyal baru adalah None (status WAIT), JANGAN hapus riwayat sinyal sebelumnya kecuali diminta hapus_riwayat
            if sinyal_sebelumnya is None:
                sinyal_sebelumnya = sinyal_aktif_lama or riwayat_lama or (SINYAL_SEBELUMNYA_DINAMIS if isinstance(SINYAL_SEBELUMNYA_DINAMIS, dict) else None)

    if sinyal_sebelumnya and isinstance(sinyal_sebelumnya, dict):
        SINYAL_SEBELUMNYA_DINAMIS = sinyal_sebelumnya
    elif hapus_riwayat:
        SINYAL_SEBELUMNYA_DINAMIS = None

    # Pastikan atas dan bawah adalah list numerik valid
    atas_val = list(atas) if isinstance(atas, (tuple, list)) else (data_lama.get("atas") if isinstance(data_lama.get("atas"), list) else None)
    bawah_val = list(bawah) if isinstance(bawah, (tuple, list)) else (data_lama.get("bawah") if isinstance(data_lama.get("bawah"), list) else None)

    data = {
        "waktu_simpan": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sinyal": sinyal if isinstance(sinyal, dict) else None,
        "sinyal_sebelumnya": sinyal_sebelumnya if isinstance(sinyal_sebelumnya, dict) else None,
        "atas": atas_val,
        "bawah": bawah_val,
        "catatan": str(catatan),
    }
    try:   # tulis atomik: file sementara lalu os.replace, agar crash saat menulis tidak merusak state
        tmp = FILE_STATE_SINYAL + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, FILE_STATE_SINYAL)
    except Exception as e:
        print(f"\n   [⚠️ Gagal menyimpan state sinyal: {e}]")
        _catat_error("simpan_state_sinyal", e)


def muat_state_sinyal():
    """Memuat sinyal dan zona pantau dari sesi sebelumnya jika tersedia."""
    if not os.path.exists(FILE_STATE_SINYAL):
        return None
    try:
        with open(FILE_STATE_SINYAL, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"\n   [⚠️ State sinyal tidak terbaca ({e}). Dicatat ke {FILE_ERROR_LOG}.]")
        _catat_error("muat_state_sinyal", e)
        return None
