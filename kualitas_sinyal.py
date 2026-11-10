# -*- coding: utf-8 -*-
"""
kualitas_sinyal.py  —  Uji objektif fitur SMC/ICT + gerbang kualitas sinyal untuk xauusd1.py

TARUH di folder yang sama dengan xauusd1.py (script utama kamu). Dua kegunaan:

  1) BACKTEST (jalankan sekali-sekali, MT5 harus terbuka & login):
        python kualitas_sinyal.py
     - Mengambil data historis MT5, lalu di tiap titik waktu menghitung fitur (bias HTF, OB, FVG, Fib OTE,
       sweep, CISD, kill zone, dst.) HANYA dari data sampai titik itu (tanpa mengintip masa depan).
     - Tiap kandidat entry disimulasikan dengan aturan SL/TP TETAP (supaya yang diukur kualitas entry-nya).
     - Data dibagi kronologis: TRAIN (pemilihan aturan) dan TEST (ujian satu kali). Aturan hanya disimpan ke
       aturan_kualitas.json bila lolos di TRAIN *dan* terbukti di TEST. Kalau tidak ada yang lolos: tidak ada
       file aturan, dan itu hasil yang sah.

  2) GERBANG LIVE (dipanggil dari xauusd1.py, lihat CONTOH_INTEGRASI di bagian bawah):
        ok, alasan, fitur = lolos_gerbang_kualitas("BUY", harga_acuan=4110.5)

  Perintah lain:  python kualitas_sinyal.py cek BUY   -> tampilkan nilai fitur saat ini (live) untuk debugging.

Bukan nasihat finansial. Hasil backtest = perkiraan; spread/volume historis broker bisa beda dari kondisi live.
"""
import importlib
import itertools
import json
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

MODUL_UTAMA = "xauusd1"          # nama file script utama (tanpa .py)

# ======================= KONFIGURASI =======================
HARI_BACKTEST = 60               # makin panjang makin baik (selama history MT5 ada); idealnya >= 90
STEP_MENIT = 30                  # titik sampel tiap sekian menit (kelipatan 5)
SESI_UTC = (7, 21)               # hanya sampel di jam UTC ini (London + New York)
OFFSET_TETAP = None              # isi jam offset server broker (mis. 2 / 3) bila pola otomatis di bawah salah;
                                 # None = pola umum broker: UTC+2 musim dingin, UTC+3 saat DST Amerika
# --- aturan simulasi (TETAP untuk semua kandidat) ---
SL_ATR_MULT = 1.5                # jarak SL = max(SL_MIN_POIN, SL_ATR_MULT x ATR M5)
SL_MIN_POIN = 3.0
TP_R_LIST = (1.5, 2.0, 3.0)      # target dalam kelipatan R
TP_R_UTAMA = 2.0                 # target yang dipakai untuk memilih aturan
MAKS_TAHAN_MENIT = 360           # posisi ditutup di harga close M1 terakhir jika belum kena SL/TP
SPREAD_POIN = 0.30               # biaya per trade (poin) = spread + slippage; SESUAIKAN dengan broker kamu
SLIPPAGE_POIN = 0.10
TOL_ZONA_ATR = 0.3               # toleransi "harga di dalam zona" (x ATR M5)
# --- pemilihan aturan (anti-overfitting) ---
COOLDOWN_MENIT = 60              # satu arah hanya dihitung 1 entry per jendela ini (meniru "sedikit entry")
FRAC_TRAIN = 0.6
MIN_N_TRAIN = 40                 # minimal entry independen di TRAIN
MIN_HARI_TRAIN = 8
MIN_N_TEST = 15
MAKS_FITUR_KOMBINASI = 3
TOP_K_FITUR = 9                  # jumlah fitur terbaik (di TRAIN) yang dikombinasikan
# --- perilaku gerbang live ---
BLOKIR_JIKA_TANPA_ATURAN = False   # True = tolak semua sinyal bila aturan_kualitas.json belum ada
BLOKIR_JIKA_GAGAL_HITUNG = False   # True = tolak sinyal bila perhitungan fitur live gagal

FILE_HASIL_CSV = os.path.join(_DIR, "backtest_fitur_xauusd.csv")
FILE_ATURAN = os.path.join(_DIR, "aturan_kualitas.json")

FITUR_UJI = [
    "bias_searah", "bias_kuat", "ema200_h1", "vwap_searah",
    "ob_h4", "ob_h1", "ob_m30", "fvg_h1", "fvg_m30", "fib_ote_h1", "konfluens_2",
    "sweep_m5", "cisd_m5", "kill_zone", "adr_ok",
]
# nama TF -> (menit per bar, jumlah bar yang dipakai per titik)
TF_JENDELA = {"M5": (5, 300), "M30": (30, 300), "H1": (60, 400), "H4": (240, 300), "D1": (1440, 120)}
MIN_BAR = {"M5": 150, "M30": 100, "H1": 100, "H4": 100, "D1": 60}


def _modul_utama():
    m = sys.modules.get("__main__")
    if m is not None and hasattr(m, "order_blocks") and hasattr(m, "struktur"):
        return m                  # dipanggil dari dalam xauusd1.py yang sedang berjalan
    return importlib.import_module(MODUL_UTAMA)


X = _modul_utama()
_EPOCH = pd.Timestamp("1970-01-01")


# ======================= WAKTU =======================
def offset_pada(t_server):
    """Offset jam server broker terhadap UTC pada waktu t_server (naive)."""
    if OFFSET_TETAP is not None:
        return OFFSET_TETAP
    if getattr(X, "SERVER_UTC_OFFSET_JAM", None) is not None:
        return X.SERVER_UTC_OFFSET_JAM
    return 3 if X.us_dst(pd.Timestamp(t_server).to_pydatetime() - timedelta(hours=3)) else 2


def _epoch_detik(index):
    return ((index - _EPOCH) // pd.Timedelta(seconds=1)).to_numpy().astype(np.int64)


# ======================= ANALISIS PER TIMEFRAME (dengan cache) =======================
_CACHE = {}


def _analisis_tf(nama, df):
    kunci = (nama, df.index[-1])
    if nama in _CACHE and _CACHE[nama][0] == kunci:
        return _CACHE[nama][1]
    d = X.tambah_indikator(df.copy())
    events, trend, sw = X.struktur(d)
    obs = X.order_blocks(d, events, maks_event=12)
    hasil = dict(d=d, events=events, trend=trend, sw=sw, obs=obs)
    _CACHE[nama] = (kunci, hasil)
    return hasil


def _skor_bias(A):
    """Meniru hitung_bias_htf() di script utama, tetapi dari data historis (D1 bobot 2, H4 bobot 2, H1 bobot 1)."""
    total = 0
    for nama, bobot in (("D1", 2), ("H4", 2), ("H1", 1)):
        a = A[nama]
        last = a["d"].iloc[-1]
        if last["EMA20"] > last["EMA50"] and last["Close"] > last["EMA20"]:
            s_ema = 1
        elif last["EMA20"] < last["EMA50"] and last["Close"] < last["EMA20"]:
            s_ema = -1
        else:
            s_ema = 0
        rsi = float(last["RSI14"])
        s_rsi = 1 if rsi > 55 else (-1 if rsi < 45 else 0)
        s200 = 1 if last["Close"] > last["EMA200"] else -1
        raw = a["trend"] + s_ema + s_rsi + s200
        total += (1 if raw >= 2 else (-1 if raw <= -2 else 0)) * bobot
    return total


# ======================= FITUR =======================
def _fvg_aktif(d, harga):
    out = []
    for tipe, b, a, st, _t in X.cari_fvg(d, harga, maks=6):
        if "mitigated" in st:
            continue
        out.append(("Bullish" in tipe, float(b), float(a)))
    return out


def _dalam_ob(a, harga, arah, tol):
    tipe = "bull" if arah > 0 else "bear"
    for o in a["obs"]:
        if o["status"] in ("fresh", "tested") and o["arah"] == tipe and o["lo"] - tol <= harga <= o["hi"] + tol:
            return 1
    return 0


def _dalam_fvg(daftar, harga, arah, tol):
    for bull, b, a in daftar:
        if bull == (arah > 0) and b - tol <= harga <= a + tol:
            return 1
    return 0


def _fib_ote(a, harga, arah):
    sw, d = a["sw"], a["d"]
    if len(sw) < 2:
        return 0
    last = sw[-1]
    prev = next((s for s in reversed(sw[:-1]) if s[1] != last[1]), None)
    if prev is None:
        return 0
    H, L = d["High"].values, d["Low"].values
    if arah > 0:
        if last[1] != "H":
            return 0
        low, high = prev[2], max(last[2], H[last[0]:].max())
        rng = high - low
        r = (high - harga) / rng if rng > 0 else -1
    else:
        if last[1] != "L":
            return 0
        high, low = prev[2], min(last[2], L[last[0]:].min())
        rng = high - low
        r = (harga - low) / rng if rng > 0 else -1
    return int(0.618 <= r <= 0.786)


def siapkan_konteks(A, harga, t_server, offset):
    """Hitungan yang tidak bergantung arah (dipakai bersama oleh BUY & SELL)."""
    a5 = A["M5"]
    d5 = a5["d"]
    atr = float(d5["ATR14"].iloc[-1])
    t_utc = (pd.Timestamp(t_server) - pd.Timedelta(hours=offset)).to_pydatetime()
    ny_off = -4 if X.us_dst(t_utc) else -5
    h_ny = (t_utc + timedelta(hours=ny_off)).hour
    d1 = A["D1"]["d"]
    adr = float((d1["High"] - d1["Low"]).iloc[-14:].mean())
    hari = d5.index[-1].normalize()
    h_ini = d5[d5.index >= hari]
    rng_hari = float(h_ini["High"].max() - h_ini["Low"].min())
    return dict(
        atr=atr, tol=TOL_ZONA_ATR * atr, skor_bias=_skor_bias(A),
        fvg_h1=_fvg_aktif(A["H1"]["d"], harga), fvg_m30=_fvg_aktif(A["M30"]["d"], harga),
        sweeps=X.liquidity_sweeps(d5, a5["sw"], X.SWING_N, lookback=6),
        cisd=X.deteksi_cisd(d5, a5["sw"], lookback=12)[0], n5=len(d5),
        vwap=float(X.vwap_harian(d5, offset)[-1]),
        kill=int(2 <= h_ny < 5 or 7 <= h_ny < 12),
        sesi=int(SESI_UTC[0] <= t_utc.hour < SESI_UTC[1]),
        adr_ok=int(adr > 0 and rng_hari < 0.9 * adr),
        ema200_bull=int(A["H1"]["d"]["Close"].iloc[-1] > A["H1"]["d"]["EMA200"].iloc[-1]),
    )


def fitur_arah(K, A, harga, arah):
    tol = K["tol"]
    f = {}
    f["bias_searah"] = int(K["skor_bias"] * arah >= X.BIAS_MIN_SKOR)
    f["bias_kuat"] = int(K["skor_bias"] * arah >= 4)
    f["ema200_h1"] = int(bool(K["ema200_bull"]) == (arah > 0))
    f["vwap_searah"] = int((harga > K["vwap"]) == (arah > 0))
    f["ob_h4"] = _dalam_ob(A["H4"], harga, arah, tol)
    f["ob_h1"] = _dalam_ob(A["H1"], harga, arah, tol)
    f["ob_m30"] = _dalam_ob(A["M30"], harga, arah, tol)
    f["fvg_h1"] = _dalam_fvg(K["fvg_h1"], harga, arah, tol)
    f["fvg_m30"] = _dalam_fvg(K["fvg_m30"], harga, arah, tol)
    f["fib_ote_h1"] = _fib_ote(A["H1"], harga, arah)
    f["skor_zona"] = f["ob_h4"] + f["ob_h1"] + f["ob_m30"] + f["fvg_h1"] + f["fvg_m30"] + f["fib_ote_h1"]
    f["konfluens_2"] = int(f["skor_zona"] >= 2)
    kata = "bias bullish" if arah > 0 else "bias bearish"
    f["sweep_m5"] = int(any(kata in s[1] for s in K["sweeps"]))
    f["cisd_m5"] = int(any((c["arah"] == "BULLISH") == (arah > 0) and (K["n5"] - 1 - c["k"]) <= 6 for c in K["cisd"]))
    f["kill_zone"] = K["kill"]
    f["adr_ok"] = K["adr_ok"]
    return f


# ======================= DATA HISTORIS & SIMULASI =======================
def muat_data_mt5():
    if not X.pastikan_mt5_aktif():
        sys.exit("MT5 tidak aktif. Buka MetaTrader 5 dan login dulu.")
    mt5 = X.mt5
    h = HARI_BACKTEST
    kebutuhan = {
        "M1": (mt5.TIMEFRAME_M1, h * 1440), "M5": (mt5.TIMEFRAME_M5, h * 288 + 400),
        "M30": (mt5.TIMEFRAME_M30, h * 48 + 400), "H1": (mt5.TIMEFRAME_H1, h * 24 + 500),
        "H4": (mt5.TIMEFRAME_H4, h * 6 + 400), "D1": (mt5.TIMEFRAME_D1, 600),
    }
    data = {}
    for nama, (tf, n) in kebutuhan.items():
        df = None
        while df is None and n >= 200:
            df = X.ambil_data(tf, n)
            if df is None:
                n //= 2
        if df is None:
            sys.exit(f"Gagal mengambil data {nama} dari MT5.")
        data[nama] = df
        print(f"   {nama}: {len(df):>7} bar  ({df.index[0]} s/d {df.index[-1]})")
    return data


def _pertama(mask):
    k = int(mask.argmax())
    return k if mask[k] else None


def simulasi_R(hi, lo, cl, arah, entry, risk, biaya):
    """R per target (setelah biaya). SL didahulukan bila SL & TP kena di candle M1 yang sama (konservatif)."""
    sl = entry - arah * risk
    i_sl = _pertama(lo <= sl) if arah > 0 else _pertama(hi >= sl)
    out, timeout = {}, False
    for r in TP_R_LIST:
        tp = entry + arah * r * risk
        i_tp = _pertama(hi >= tp) if arah > 0 else _pertama(lo <= tp)
        if i_sl is None and i_tp is None:
            R, timeout = (cl[-1] - entry) * arah / risk, True
        elif i_tp is None or (i_sl is not None and i_sl <= i_tp):
            R = -1.0
        else:
            R = r
        out[r] = R - biaya / risk
    return out, timeout


def _potong(data, idx, t_int):
    A = {}
    for nama, (menit, look) in TF_JENDELA.items():
        n = int(np.searchsorted(idx[nama], t_int - menit * 60, side="right"))
        if n < MIN_BAR[nama]:
            return None
        A[nama] = _analisis_tf(nama, data[nama].iloc[max(0, n - look):n])
    return A


def bangun_dataset(data, log_tiap=200):
    idx = {k: _epoch_detik(v.index) for k, v in data.items()}
    m1_t, m1_h, m1_l, m1_c = idx["M1"], data["M1"]["High"].values, data["M1"]["Low"].values, data["M1"]["Close"].values
    m5 = data["M5"]
    baris, dihitung = [], 0
    biaya = SPREAD_POIN + SLIPPAGE_POIN
    mulai_m1 = m1_t[0] if len(m1_t) else 0
    for i in range(len(m5)):
        t_open = m5.index[i]
        if t_open.minute % STEP_MENIT != 0:
            continue
        t = t_open + pd.Timedelta(minutes=5)
        t_int = int((t - _EPOCH).total_seconds())
        if t_int < mulai_m1:
            continue
        off = offset_pada(t)
        t_utc = (t - pd.Timedelta(hours=off)).to_pydatetime()
        if t_utc.weekday() >= 5 or not (SESI_UTC[0] <= t_utc.hour < SESI_UTC[1]):
            continue
        j = int(np.searchsorted(m1_t, t_int, side="left"))
        e = int(np.searchsorted(m1_t, t_int + MAKS_TAHAN_MENIT * 60, side="left"))
        if e - j < 30:
            continue
        A = _potong(data, idx, t_int)
        if A is None:
            continue
        harga = float(m5["Close"].iloc[i])
        K = siapkan_konteks(A, harga, t, off)
        risk = max(SL_MIN_POIN, SL_ATR_MULT * K["atr"])
        for arah in (1, -1):
            f = fitur_arah(K, A, harga, arah)
            R, timeout = simulasi_R(m1_h[j:e], m1_l[j:e], m1_c[j:e], arah, harga, risk, biaya)
            baris.append(dict(waktu=t, hari=t.date(), arah=arah, harga=harga, atr=K["atr"], risk=risk,
                              skor_bias=K["skor_bias"], timeout=int(timeout), **f,
                              **{f"R_{r}": R[r] for r in TP_R_LIST}))
        dihitung += 1
        if dihitung % log_tiap == 0:
            print(f"   ... {dihitung} titik waktu diproses (terakhir {t})")
    return pd.DataFrame(baris)


# ======================= STATISTIK & PEMILIHAN ATURAN =======================
def tanpa_tumpang_tindih(df, menit=COOLDOWN_MENIT):
    keep, last = [], {}
    for row in df.sort_values("waktu").itertuples():
        w = last.get(row.arah)
        if w is not None and (row.waktu - w).total_seconds() < menit * 60:
            continue
        last[row.arah] = row.waktu
        keep.append(row.Index)
    return df.loc[keep]


def ringkas(df, kol, n_boot=1500, seed=0):
    if len(df) == 0:
        return dict(n=0, mean=np.nan, lo=np.nan, hi=np.nan, win=np.nan, hari=0, total=0.0)
    g = df.groupby("hari")[kol].agg(["sum", "count"])
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    ix = np.random.default_rng(seed).integers(0, len(g), (n_boot, len(g)))     # bootstrap per HARI (antar-hari independen)
    m = s[ix].sum(1) / c[ix].sum(1)
    return dict(n=len(df), mean=float(df[kol].mean()), lo=float(np.percentile(m, 2.5)), hi=float(np.percentile(m, 97.5)),
                win=float((df[kol] > 0).mean()), hari=len(g), total=float(df[kol].sum()))


def _fmt(r):
    if r["n"] == 0:
        return "n=0"
    return (f"n={r['n']:>4} hari={r['hari']:>2} | win {r['win'] * 100:4.0f}% | "
            f"rata2 {r['mean']:+.2f}R [CI95 {r['lo']:+.2f} .. {r['hi']:+.2f}] | total {r['total']:+.1f}R")


def cari_aturan(train, kol):
    uni = []
    print("\nFITUR TUNGGAL di TRAIN (hanya entry independen, cooldown diterapkan):")
    base = ringkas(tanpa_tumpang_tindih(train), kol)
    print(f"  {'BASELINE (semua kandidat)':<26} {_fmt(base)}")
    for f in FITUR_UJI:
        r = ringkas(tanpa_tumpang_tindih(train[train[f] == 1]), kol)
        uni.append((f, r))
        print(f"  {f:<26} {_fmt(r)}")
    layak = [(f, r) for f, r in uni if r["n"] >= MIN_N_TRAIN // 2 and not np.isnan(r["mean"])]
    top = [f for f, _ in sorted(layak, key=lambda x: -x[1]["mean"])[:TOP_K_FITUR]]
    diuji, lolos = 0, []
    for k in range(1, MAKS_FITUR_KOMBINASI + 1):
        for kombo in itertools.combinations(top, k):
            diuji += 1
            m = np.ones(len(train), dtype=bool)
            for f in kombo:
                m &= (train[f].to_numpy() == 1)
            r = ringkas(tanpa_tumpang_tindih(train[m]), kol)
            if r["n"] >= MIN_N_TRAIN and r["hari"] >= MIN_HARI_TRAIN and r["lo"] > 0:
                lolos.append((kombo, r))
    lolos.sort(key=lambda x: -x[1]["lo"])
    print(f"\nKombinasi diuji: {diuji} | lolos syarat TRAIN (n, hari, batas bawah CI > 0): {len(lolos)}")
    for kombo, r in lolos[:5]:
        print(f"  {' + '.join(kombo):<60} {_fmt(r)}")
    return base, lolos


def jalankan_backtest():
    print("1. Mengambil data historis MT5 ...")
    data = muat_data_mt5()
    print("2. Menghitung fitur & simulasi hasil (bisa beberapa menit) ...")
    df = bangun_dataset(data)
    if df.empty:
        sys.exit("Tidak ada sampel. Perpanjang HARI_BACKTEST atau cek SESI_UTC/offset server.")
    df.to_csv(FILE_HASIL_CSV, index=False)
    analisis_dan_simpan(df)


def analisis_dan_simpan(df):
    kol = f"R_{TP_R_UTAMA}"
    df = df.sort_values("waktu").reset_index(drop=True)
    hari = sorted(df["hari"].unique())
    if len(hari) < 15:
        print(f"\nPERINGATAN: hanya {len(hari)} hari perdagangan. Hasil belum bisa dipercaya; perpanjang HARI_BACKTEST.")
    batas = hari[int(len(hari) * FRAC_TRAIN)]
    train, test = df[df["hari"] < batas], df[df["hari"] >= batas]
    print(f"\nData: {len(df)} baris kandidat, {len(hari)} hari | TRAIN < {batas} ({len(train)}) | TEST >= {batas} ({len(test)})")
    print(f"Aturan simulasi: SL = max({SL_MIN_POIN}, {SL_ATR_MULT} x ATR M5), TP utama {TP_R_UTAMA}R, biaya {SPREAD_POIN + SLIPPAGE_POIN:.2f} poin/trade, "
          f"cooldown {COOLDOWN_MENIT} menit.")
    base, lolos = cari_aturan(train, kol)
    base_test = ringkas(tanpa_tumpang_tindih(test), kol)
    print(f"\nBASELINE di TEST (semua kandidat): {_fmt(base_test)}")
    if not lolos:
        print("\nHASIL: TIDAK ADA aturan yang lolos di TRAIN. Artinya, pada data ini, tidak ada kombinasi fitur yang terbukti")
        print("       memberi edge setelah biaya. aturan_kualitas.json TIDAK dibuat. Perpanjang data / tinjau ulang konsepnya.")
        return None
    kombo, r_train = lolos[0]
    m = np.ones(len(test), dtype=bool)
    for f in kombo:
        m &= (test[f].to_numpy() == 1)
    r_test = ringkas(tanpa_tumpang_tindih(test[m]), kol)
    print(f"\nKANDIDAT TERBAIK : {' + '.join(kombo)}")
    print(f"  TRAIN : {_fmt(r_train)}")
    print(f"  TEST  : {_fmt(r_test)}   <- ujian satu kali, tidak dipakai untuk memilih")
    for r in TP_R_LIST:
        rr = ringkas(tanpa_tumpang_tindih(df[(np.all([df[f] == 1 for f in kombo], axis=0))]), f"R_{r}")
        print(f"  Semua data, TP {r}R : {_fmt(rr)}")
    diterima = r_test["n"] >= MIN_N_TEST and r_test["mean"] > 0
    if not diterima:
        print(f"\nHASIL: DITOLAK. Di TEST n={r_test['n']} (min {MIN_N_TEST}) rata2 {r_test['mean']:+.2f}R. Kemungkinan overfit/kebetulan.")
        print("       aturan_kualitas.json TIDAK dibuat.")
        return None
    sub = tanpa_tumpang_tindih(df[(np.all([df[f] == 1 for f in kombo], axis=0))]).copy()
    sub["minggu"] = pd.to_datetime(sub["waktu"]).dt.strftime("%G-W%V")
    print("\nStabilitas per minggu (aturan terpilih, semua data):")
    for mg, g in sub.groupby("minggu")[kol]:
        print(f"  {mg}: {len(g):>3} entry | {g.sum():+6.1f}R")
    aturan = dict(fitur=list(kombo), tp_r=TP_R_UTAMA, sl_atr_mult=SL_ATR_MULT, sl_min_poin=SL_MIN_POIN,
                  biaya_poin=SPREAD_POIN + SLIPPAGE_POIN, dibuat=datetime.now().strftime("%Y-%m-%d %H:%M"),
                  hari_data=len(hari), train=r_train, test=r_test)
    with open(FILE_ATURAN, "w", encoding="utf-8") as f:
        json.dump(aturan, f, indent=2, default=float)
    n_per_minggu = len(sub) / max(1, len(hari) / 5)
    print(f"\nHASIL: DITERIMA. Disimpan ke {FILE_ATURAN}  (~{n_per_minggu:.1f} entry/minggu jika semua dieksekusi)")
    print("       Tetap uji ulang tiap beberapa minggu dengan data baru; jika expectancy turun, aturan perlu ditinjau.")
    return aturan


# ======================= GERBANG LIVE =======================
def muat_aturan():
    if not os.path.exists(FILE_ATURAN):
        return None
    try:
        with open(FILE_ATURAN, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        X._catat_error("muat_aturan_kualitas", e)
        return None


def hitung_fitur_live(arah_txt, harga_acuan=None):
    """Fitur untuk sinyal BUY/SELL berdasarkan data MT5 saat ini (hanya candle yang sudah close)."""
    if not X.pastikan_mt5_aktif():
        raise RuntimeError("MT5 tidak aktif")
    mt5 = X.mt5
    tf_map = {"M5": mt5.TIMEFRAME_M5, "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1,
              "H4": mt5.TIMEFRAME_H4, "D1": mt5.TIMEFRAME_D1}
    A = {}
    for nama, (_m, look) in TF_JENDELA.items():
        df = X.ambil_data(tf_map[nama], look)
        if df is None or len(df) < MIN_BAR[nama]:
            raise RuntimeError(f"data {nama} tidak cukup")
        d = X.tambah_indikator(df.copy())
        events, trend, sw = X.struktur(d)
        A[nama] = dict(d=d, events=events, trend=trend, sw=sw, obs=X.order_blocks(d, events, maks_event=12))
    tick = mt5.symbol_info_tick(X.SYMBOL)
    t_server = pd.Timestamp(int(tick.time), unit="s").tz_localize(None) if tick else A["M5"]["d"].index[-1] + pd.Timedelta(minutes=5)
    off = getattr(X, "SERVER_UTC_OFFSET_JAM", None)
    if off is None:
        off = X.deteksi_offset(tick, A["M5"]["d"]) if tick else None
    if off is None:
        off = offset_pada(t_server)
    arah = 1 if str(arah_txt).upper() == "BUY" else -1
    harga = float(harga_acuan) if harga_acuan else float(tick.bid if tick else A["M5"]["d"]["Close"].iloc[-1])
    K = siapkan_konteks(A, harga, t_server, off)
    return fitur_arah(K, A, harga, arah)


def lolos_gerbang_kualitas(arah_txt, harga_acuan=None):
    """Mengembalikan (ok, [alasan], fitur). Dipanggil dari gerbang_sinyal_baru() di script utama."""
    aturan = muat_aturan()
    if aturan is None:
        return (not BLOKIR_JIKA_TANPA_ATURAN), (["aturan_kualitas.json belum ada (jalankan: python kualitas_sinyal.py)"]
                                                 if BLOKIR_JIKA_TANPA_ATURAN else []), {}
    try:
        f = hitung_fitur_live(arah_txt, harga_acuan)
    except Exception as e:
        X._catat_error("gerbang_kualitas", e)
        return (not BLOKIR_JIKA_GAGAL_HITUNG), ([f"gagal menghitung fitur kualitas: {e}"] if BLOKIR_JIKA_GAGAL_HITUNG else []), {}
    kurang = [k for k in aturan["fitur"] if not f.get(k)]
    if kurang:
        return False, ["filter kualitas: belum terpenuhi -> " + ", ".join(kurang)
                       + f" (aturan: {' + '.join(aturan['fitur'])})"], f
    return True, [], f


CONTOH_INTEGRASI = '''
# Di xauusd1.py, dalam gerbang_sinyal_baru(sig), TEPAT SEBELUM baris terakhir
#   return (not alasan), alasan
# tambahkan:
    try:
        import kualitas_sinyal as KS
        mid_q = (sig.get("lo_asli", sig["lo"]) + sig.get("hi_asli", sig["hi"])) / 2.0
        ok_q, al_q, _ = KS.lolos_gerbang_kualitas(sig["arah"], harga_acuan=mid_q)
        if not ok_q:
            alasan.extend(al_q)
    except Exception as e_kq:
        _catat_error("gerbang_kualitas_import", e_kq)
'''

if __name__ == "__main__":
    try:
        if len(sys.argv) >= 3 and sys.argv[1].lower() == "cek":
            ff = hitung_fitur_live(sys.argv[2])
            print(json.dumps(ff, indent=2))
            ar = muat_aturan()
            print("Aturan aktif:", ar["fitur"] if ar else "(belum ada)")
        else:
            jalankan_backtest()
    finally:
        X._tutup_mt5()
