# -*- coding: utf-8 -*-
"""
modules/analisis_teknikal.py
Perhitungan teknikal SMC/ICT: Swing, Struktur (BOS/CHoCH), Order Blocks, FVG,
Liquidity Sweeps, Equal Highs/Lows, Bias HTF, dan SMT Divergence. Generator zona aktif dibatasi pada OB fresh + FVG untouched.
"""
import math
import time
from datetime import datetime, timedelta, timezone
import MetaTrader5 as mt5
import numpy as np
import pandas as pd

from modules.config import SYMBOL, TIMEFRAMES, SWING_N, BIAS_MIN_SKOR, LEBAR_CLUSTER_MAKS, JARAK_CONFLUENCE_MAKS
from modules.utils import wkt, jarak_zona, selisih_zona, _catat_error
from modules.mt5_conn import (
    ambil_data, pastikan_mt5_aktif, tambah_indikator,
    ambil_candle_berjalan, _suffix_simbol
)

_CACHE_BIAS = {}


def cari_swing(df, n=SWING_N):
    """Swing high/low yang valid (sudah terkonfirmasi n candle di kanan) berbasis array NumPy cepat."""
    H, L = df["High"].values, df["Low"].values
    idx = df.index
    highs, lows = [], []
    for i in range(n, len(df) - n):
        if H[i] == H[i - n:i + n + 1].max():
            highs.append((idx[i], float(H[i])))
        if L[i] == L[i - n:i + n + 1].min():
            lows.append((idx[i], float(L[i])))
    return highs, lows


def deteksi_pola(df):
    """Deteksi pola candle sederhana pada 10 candle terakhir yang sudah close."""
    hasil = []
    for i in range(len(df) - 10, len(df)):
        o, h, l, cl = (df[k].iloc[i] for k in ("Open", "High", "Low", "Close"))
        po, pcl = df["Open"].iloc[i - 1], df["Close"].iloc[i - 1]
        body = abs(cl - o)
        rng = h - l
        if rng == 0:
            continue
        atas = h - max(o, cl)
        bawah = min(o, cl) - l
        waktu = df.index[i].strftime("%m-%d %H:%M")
        if cl > o and pcl < po and cl >= po and o <= pcl:
            hasil.append(f"{waktu}: Bullish Engulfing")
        if cl < o and pcl > po and cl <= po and o >= pcl:
            hasil.append(f"{waktu}: Bearish Engulfing")
        if bawah >= 2 * body and bawah >= 0.6 * rng and atas <= 0.15 * rng:
            hasil.append(f"{waktu}: Pin Bar / Hammer (rejection bawah, bullish)")
        if atas >= 2 * body and atas >= 0.6 * rng and bawah <= 0.15 * rng:
            hasil.append(f"{waktu}: Shooting Star (rejection atas, bearish)")
        if body <= 0.1 * rng:
            hasil.append(f"{waktu}: Doji (ragu-ragu)")
    return hasil


def ringkas_tf(nama, df, tail):
    df = tambah_indikator(df)
    last = df.iloc[-1]
    if last["EMA20"] > last["EMA50"] and last["Close"] > last["EMA20"]:
        bias = "BULLISH (Close > EMA20 > EMA50)"
    elif last["EMA20"] < last["EMA50"] and last["Close"] < last["EMA20"]:
        bias = "BEARISH (Close < EMA20 < EMA50)"
    else:
        bias = "NETRAL / TRANSISI (EMA & harga tidak searah)"

    highs, lows = cari_swing(df)
    fmt = lambda lst: ", ".join(f"{p:.2f} ({t.strftime('%m-%d %H:%M')})" for t, p in lst[-4:]) or "-"
    kolom = ["Open", "High", "Low", "Close", "EMA20", "EMA50", "RSI14", "ATR14"]
    tabel = df[kolom].tail(tail).round(2).to_string()
    teks = f"""
##### TIMEFRAME {nama} #####
Bias mekanis: {bias}
Close terakhir: {last['Close']:.2f} | RSI14: {last['RSI14']:.1f} | ATR14: {last['ATR14']:.2f}
Swing High terkonfirmasi (resistance): {fmt(highs)}
Swing Low terkonfirmasi (support): {fmt(lows)}
Range {len(df)} candle: High {df['High'].max():.2f} / Low {df['Low'].min():.2f}
"""
    if nama in ("M30", "M5"):
        pola = deteksi_pola(df)
        teks += "Pola candle (10 candle terakhir): " + ("; ".join(pola) if pola else "tidak ada pola jelas") + "\n"
    teks += f"{tail} candle terakhir {nama}:\n{tabel}\n"
    return teks


def cari_fvg(df, harga, maks=3):
    """FVG ketat: hanya gap >= 0.30 ATR yang belum termitigasi penuh.

    Tidak ada fallback ke ukuran gap lebih kecil atau FVG yang sudah terisi.
    FVG hanya menjadi konfirmasi zona OB, bukan generator zona entry mandiri.
    """
    if df is None or len(df) < 20:
        return []
    d = tambah_indikator(df.copy())
    atr = d["ATR14"].values
    H, L = d["High"].values, d["Low"].values
    zona = []
    mulai = max(2, len(d) - 180)  # hindari FVG historis yang sudah terlalu lama
    for i in range(mulai, len(d)):
        if not np.isfinite(atr[i]) or atr[i] <= 0:
            continue
        if L[i] > H[i - 2] and (L[i] - H[i - 2]) >= 0.30 * atr[i]:
            bawah, atas, tipe = float(H[i - 2]), float(L[i]), "Bullish FVG (demand)"
            low_after = L[i + 1:]
            if len(low_after) and float(np.min(low_after)) <= bawah:
                continue  # sudah terisi penuh: invalid sebagai FVG aktif
            status = "sebagian terisi" if len(low_after) and float(np.min(low_after)) < atas else "belum tersentuh"
        elif H[i] < L[i - 2] and (L[i - 2] - H[i]) >= 0.30 * atr[i]:
            bawah, atas, tipe = float(H[i]), float(L[i - 2]), "Bearish FVG (supply)"
            high_after = H[i + 1:]
            if len(high_after) and float(np.max(high_after)) >= atas:
                continue
            status = "sebagian terisi" if len(high_after) and float(np.max(high_after)) > bawah else "belum tersentuh"
        else:
            continue
        zona.append((tipe, bawah, atas, status, d.index[i - 1]))

    diatas = sorted((z for z in zona if z[1] >= harga), key=lambda z: z[1])[:maks]
    dibawah = sorted((z for z in zona if z[2] <= harga), key=lambda z: -z[2])[:maks]
    dalam = [z for z in zona if z[1] < harga < z[2]][-1:]
    return dalam + diatas + dibawah

def swing_belum_ditembus(df, harga, n=5, maks=4):
    """Swing high/low lama yang belum pernah ditembus. Jika kosong (misal di ATH), ambil swing ekstrem historis terdekat."""
    highs, lows = cari_swing(df, n)
    H, L = df["High"], df["Low"]
    res = [(t, p, "unmitigated") for t, p in highs if p > harga and H[H.index > t].max() <= p] if len(highs) else []
    sup = [(t, p, "unmitigated") for t, p in lows if p < harga and L[L.index > t].min() >= p] if len(lows) else []
    if not res and len(highs):
        res = [(t, p, "flip/historis") for t, p in highs if p > harga]
    if not sup and len(lows):
        sup = [(t, p, "flip/historis") for t, p in lows if p < harga]
    res = sorted(res, key=lambda x: x[1])[:maks]
    sup = sorted(sup, key=lambda x: -x[1])[:maks]
    return res, sup


def level_psikologis_ekstrem(df_w1, df_d1, harga):
    """Level angka bulat psikologis & ATH/ATL historis sebagai referensi jika di area harga baru."""
    ath, t_ath = df_w1["High"].max(), df_w1["High"].idxmax()
    atl, t_atl = df_w1["Low"].min(), df_w1["Low"].idxmin()
    cw = ambil_candle_berjalan(mt5.TIMEFRAME_W1)   # candle minggu ini (belum close) ikut dihitung
    if cw is not None:
        if float(cw["high"]) > ath:
            ath, t_ath = float(cw["high"]), pd.Timestamp(int(cw["time"]), unit="s")
        if float(cw["low"]) < atl:
            atl, t_atl = float(cw["low"]), pd.Timestamp(int(cw["time"]), unit="s")
    d_hi, d_lo = df_d1["High"].tail(60).max(), df_d1["Low"].tail(60).min()
    base50 = int(harga // 50) * 50
    bulat = sorted({base50 + k * 25 for k in range(-3, 5)})
    atas = [x for x in bulat if x > harga][:3]
    bawah = [x for x in bulat if x <= harga][-3:]
    return (
        f"\n===== LEVEL EKSTREM & PSIKOLOGIS (referensi tambahan / ATH) =====\n"
        f"Rekor di histori W1: Tertinggi (ATH) {ath:.2f} ({t_ath:%Y-%m-%d}) | Terendah {atl:.2f} ({t_atl:%Y-%m-%d})\n"
        f"Range 60 hari terakhir (D1): High {d_hi:.2f} | Low {d_lo:.2f}\n"
        f"Level psikologis (angka bulat) di atas harga: {', '.join(f'{x:.0f}' for x in atas)}\n"
        f"Level psikologis (angka bulat) di bawah harga: {', '.join(f'{x:.0f}' for x in bawah)}\n"
    )


def ringkas_zona(nama, df, harga):
    """Ringkasan level likuiditas dan FVG; swing bukan zona entry otomatis."""
    fmt_t = lambda t: t.strftime("%Y-%m-%d %H:%M")
    fvg = cari_fvg(df, harga)
    highs, lows = cari_swing(df, 5)
    res = [(t, px) for t, px in highs if px > harga][-3:]
    sup = [(t, px) for t, px in lows if px < harga][-3:]
    teks = f"\n===== REFERENSI LIKUIDITAS & FVG {nama} ({len(df)} candle) =====\n"
    teks += "Swing High (referensi likuiditas, BUKAN zona entry): "
    teks += ", ".join(f"{px:.2f} ({fmt_t(t)})" for t, px in res) or "-"
    teks += "\nSwing Low (referensi likuiditas, BUKAN zona entry): "
    teks += ", ".join(f"{px:.2f} ({fmt_t(t)})" for t, px in sup) or "-"
    teks += "\nFVG aktif (konfirmasi OB; hanya gap >= 0.30 ATR):\n"
    if fvg:
        for tipe, b, a, st, t in fvg:
            posisi = "HARGA DI DALAM" if b < harga < a else (f"{b - harga:.1f} di atas" if b >= harga else f"{harga - a:.1f} di bawah")
            teks += f"  - {tipe} {b:.2f}-{a:.2f} | {st} | {fmt_t(t)} | {posisi} harga\n"
    else:
        teks += "  - tidak ada FVG aktif berkualitas\n"
    teks += "Catatan: swing dan FVG tidak boleh diperlakukan sebagai zona entry mandiri tanpa Order Block valid.\n"
    return teks

def idx_swings(df, n=3):
    """Swing (i, 'H'/'L', harga); terkonfirmasi setelah n candle di kanan."""
    H, L = df["High"].values, df["Low"].values
    out = []
    for i in range(n, len(df) - n):
        if H[i] == H[i - n:i + n + 1].max():
            out.append((i, "H", H[i]))
        if L[i] == L[i - n:i + n + 1].min():
            out.append((i, "L", L[i]))
    return sorted(out, key=lambda s: s[0])


def struktur(df, n=3):
    """BOS / CHoCH berbasis CLOSE yang menembus swing terakhir."""
    C = df["Close"].values
    sw = idx_swings(df, n)
    konfirmasi = {}
    for s in sw:
        konfirmasi.setdefault(s[0] + n, []).append(s)
    last_h = last_l = None
    trend = 0
    events = []
    for k in range(len(df)):
        for (i, t, p) in konfirmasi.get(k, []):
            if t == "H":
                last_h = (i, p)
            else:
                last_l = (i, p)
        if last_h is not None and C[k] > last_h[1]:
            tipe = "CHoCH" if trend == -1 else "BOS"
            origin = last_l[0] if last_l else max(0, k - 30)
            events.append(dict(k=k, tipe=tipe, arah="bull", level=last_h[1], origin=origin))
            last_h, trend = None, 1
        elif last_l is not None and C[k] < last_l[1]:
            tipe = "CHoCH" if trend == 1 else "BOS"
            origin = last_h[0] if last_h else max(0, k - 30)
            events.append(dict(k=k, tipe=tipe, arah="bear", level=last_l[1], origin=origin))
            last_l, trend = None, -1
    return events, trend, sw


def deteksi_cisd(df, sw, lookback=35):
    """Deteksi CISD (Change in State of Delivery - ICT):
    Konfirmasi pembalikan dini ketika harga membuat titik ekstrem (Low/High atau Liquidity Sweep),
    lalu candle berikutnya CLOSE menembus harga OPEN dari rangkaian candle searah pertama yang mengantarkan harga ke ekstrem tersebut.
    Mengembalikan: (list_cisd_terkonfirmasi, info_calon_cisd_aktif)"""
    O, H, L, C = (df[k].values for k in ("Open", "High", "Low", "Close"))
    n_len = len(df)
    start_idx = max(5, n_len - lookback)
    hasil = []
    seen_level = set()

    for m in range(start_idx, n_len - 1):
        sw_highs = [s[2] for s in sw if s[1] == "H" and s[0] < m - 1]
        sw_lows = [s[2] for s in sw if s[1] == "L" and s[0] < m - 1]
        # 1. Cek Titik Low Lokal (kandidat Bullish CISD)
        win_l = L[max(0, m - 3):min(n_len, m + 3)]
        if L[m] == win_l.min():
            # Cari candle bearish pada m atau m-1 yang mengantarkan ke low ini
            idx_bear = m if C[m] < O[m] else (m - 1 if m >= 1 and C[m - 1] < O[m - 1] else None)
            if idx_bear is not None:
                first_bear = idx_bear
                while first_bear > 0 and C[first_bear - 1] < O[first_bear - 1] and (idx_bear - first_bear) < 5:
                    first_bear -= 1
                lvl_open = float(O[first_bear])
                low_ekstrem = float(L[m])
                # Abaikan kaki delivery mikro (< 1.2 poin) agar tidak terjebak noise M5
                if (lvl_open - low_ekstrem) >= 1.2:
                    # Cek apakah dalam 1..5 candle setelah m ada yang CLOSE tegas di atas lvl_open tanpa menjebol low_ekstrem
                    for k in range(m + 1, min(n_len, m + 6)):
                        if L[k] < low_ekstrem:
                            break
                        if C[k] > lvl_open + 0.15 and (C[k] - O[k]) >= 0.4:
                            key = ("BULL", round(lvl_open, 2))
                            if key not in seen_level:
                                seen_level.add(key)
                                is_sweep = any(low_ekstrem < sl_p and C[m] >= sl_p for sl_p in sw_lows[-8:])
                                tag_sw = " [SWEEP + CISD VALID!]" if is_sweep else ""
                                hasil.append(dict(
                                    k=k,
                                    waktu=df.index[k],
                                    arah="BULLISH",
                                    level=lvl_open,
                                    close=float(C[k]),
                                    ekstrem=low_ekstrem,
                                    is_sweep=is_sweep,
                                    ket=f"Bullish CISD{tag_sw} (Close {C[k]:.2f} > Open delivery {lvl_open:.2f}, SL di bawah {low_ekstrem:.2f})",
                                ))
                            break

        # 2. Cek Titik High Lokal (kandidat Bearish CISD)
        win_h = H[max(0, m - 3):min(n_len, m + 3)]
        if H[m] == win_h.max():
            idx_bull = m if C[m] > O[m] else (m - 1 if m >= 1 and C[m - 1] > O[m - 1] else None)
            if idx_bull is not None:
                first_bull = idx_bull
                while first_bull > 0 and C[first_bull - 1] > O[first_bull - 1] and (idx_bull - first_bull) < 5:
                    first_bull -= 1
                lvl_open = float(O[first_bull])
                high_ekstrem = float(H[m])
                # Abaikan kaki delivery mikro (< 1.2 poin) agar tidak terjebak noise M5
                if (high_ekstrem - lvl_open) >= 1.2:
                    for k in range(m + 1, min(n_len, m + 6)):
                        if H[k] > high_ekstrem:
                            break
                        if C[k] < lvl_open - 0.15 and (O[k] - C[k]) >= 0.4:
                            key = ("BEAR", round(lvl_open, 2))
                            if key not in seen_level:
                                seen_level.add(key)
                                is_sweep = any(high_ekstrem > sh_p and C[m] <= sh_p for sh_p in sw_highs[-8:])
                                tag_sw = " [SWEEP + CISD VALID!]" if is_sweep else ""
                                hasil.append(dict(
                                    k=k,
                                    waktu=df.index[k],
                                    arah="BEARISH",
                                    level=lvl_open,
                                    close=float(C[k]),
                                    ekstrem=high_ekstrem,
                                    is_sweep=is_sweep,
                                    ket=f"Bearish CISD{tag_sw} (Close {C[k]:.2f} < Open delivery {lvl_open:.2f}, SL di atas {high_ekstrem:.2f})",
                                ))
                            break

    hasil.sort(key=lambda x: x["k"])

    # 3. Hitung Calon Level CISD Aktif (pada rangkaian candle paling kanan / terbaru)
    calon_info = "-"
    if n_len >= 4:
        last_idx = n_len - 1
        if C[last_idx] < O[last_idx]:
            fb = last_idx
            while fb > 0 and C[fb - 1] < O[fb - 1] and (last_idx - fb) < 5:
                fb -= 1
            low_leg = float(L[fb:last_idx + 1].min())
            calon_info = f"Menunggu Bullish CISD: butuh candle close > {O[fb]:.2f} (SL di bawah {low_leg:.2f})"
        elif C[last_idx] > O[last_idx]:
            fb = last_idx
            while fb > 0 and C[fb - 1] > O[fb - 1] and (last_idx - fb) < 5:
                fb -= 1
            high_leg = float(H[fb:last_idx + 1].max())
            calon_info = f"Menunggu Bearish CISD: butuh candle close < {O[fb]:.2f} (SL di atas {high_leg:.2f})"

    return hasil, calon_info


def order_blocks(df, events, maks_event=8):
    """OB = candle lawan terakhir sebelum dorongan yang menghasilkan BOS/CHoCH.
    Jika ditutup menembus OB -> berubah jadi Breaker Block."""
    O, H, L, C = (df[k].values for k in ("Open", "High", "Low", "Close"))
    obs = []
    for ev in events[-maks_event:]:
        k, o = ev["k"], ev["origin"]
        if o >= k:
            continue
        if ev["arah"] == "bull":
            m = o + int(np.argmin(L[o:k + 1]))
            j = next((x for x in range(m, max(m - 6, -1), -1) if C[x] < O[x]), m)
        else:
            m = o + int(np.argmax(H[o:k + 1]))
            j = next((x for x in range(m, max(m - 6, -1), -1) if C[x] > O[x]), m)
        lo, hi = L[j], H[j]
        status, flip = "fresh", None
        for x in range(k + 1, len(df)):
            if ev["arah"] == "bull":
                if C[x] < lo:
                    status, flip = "breaker", x
                    break
                if L[x] <= hi:
                    status = "tested"
            else:
                if C[x] > hi:
                    status, flip = "breaker", x
                    break
                if H[x] >= lo:
                    status = "tested"
        retest = False
        if flip is not None:
            if ev["arah"] == "bull":   # sekarang jadi supply
                retest = bool((H[flip + 1:] >= lo).any()) if flip + 1 < len(df) else False
            else:                      # sekarang jadi demand
                retest = bool((L[flip + 1:] <= hi).any()) if flip + 1 < len(df) else False
        obs.append(dict(arah=ev["arah"], lo=lo, hi=hi, t=df.index[j], status=status,
                        asal=ev["tipe"], retest=retest))
    uniq, seen = [], set()
    for o in obs:   # buang zona kembar (beberapa event bisa menunjuk candle OB yang sama)
        key = (round(o["lo"], 2), round(o["hi"], 2))
        if key not in seen:
            seen.add(key)
            uniq.append(o)
    return uniq


def liquidity_sweeps(df, sw, n, lookback=40):
    """Wick menembus swing lama tapi CLOSE kembali di dalam = stop hunt / liquidity grab."""
    H, L, C = df["High"].values, df["Low"].values, df["Close"].values
    res, seen = [], set()
    for k in range(max(0, len(df) - lookback), len(df)):
        for tipe in ("H", "L"):
            kandidat = [s for s in sw if s[1] == tipe and s[0] + n < k][-8:]
            for (i, _, p) in kandidat:
                key = (tipe, round(p, 2))
                if key in seen:
                    continue
                if tipe == "H":
                    seg = H[i + 1:k]
                    if H[k] > p and C[k] < p and (len(seg) == 0 or seg.max() <= p):
                        res.append((df.index[k], "Sweep BUY-SIDE liquidity (high disapu, close balik) -> bias bearish", p, H[k], C[k]))
                        seen.add(key)
                else:
                    seg = L[i + 1:k]
                    if L[k] < p and C[k] > p and (len(seg) == 0 or seg.min() >= p):
                        res.append((df.index[k], "Sweep SELL-SIDE liquidity (low disapu, close balik) -> bias bullish", p, L[k], C[k]))
                        seen.add(key)
    return res


def equal_highs_lows(df, sw, atr_val, harga, maks=3):
    """Deteksi Equal Highs (EQH) & Equal Lows (EQL) yang belum disapu -> kolam likuiditas / magnet harga."""
    H, L = df["High"].values, df["Low"].values
    tol = 0.25 * atr_val
    eqh, eql = [], []
    highs = [s for s in sw if s[1] == "H"][-25:]
    lows = [s for s in sw if s[1] == "L"][-25:]
    for idx_a in range(len(highs)):
        for idx_b in range(idx_a + 1, len(highs)):
            i1, _, p1 = highs[idx_a]
            i2, _, p2 = highs[idx_b]
            if i2 - i1 < 3 or abs(p1 - p2) > tol:
                continue
            lvl = max(p1, p2)
            if len(H[i2 + 1:]) and H[i2 + 1:].max() > lvl + 0.1 * atr_val:
                continue
            eqh.append((lvl, df.index[i1], df.index[i2]))
    for idx_a in range(len(lows)):
        for idx_b in range(idx_a + 1, len(lows)):
            i1, _, p1 = lows[idx_a]
            i2, _, p2 = lows[idx_b]
            if i2 - i1 < 3 or abs(p1 - p2) > tol:
                continue
            lvl = min(p1, p2)
            if len(L[i2 + 1:]) and L[i2 + 1:].min() < lvl - 0.1 * atr_val:
                continue
            eql.append((lvl, df.index[i1], df.index[i2]))
    eqh = sorted(eqh, key=lambda x: abs(x[0] - harga))[:maks]
    eql = sorted(eql, key=lambda x: abs(x[0] - harga))[:maks]
    return eqh, eql


def rsi_divergence(df, sw):
    """Deteksi Regular Bullish & Bearish Divergence antara Harga dan RSI14 pada swing terakhir."""
    rsi = df["RSI14"].values
    highs = [s for s in sw if s[1] == "H"][-4:]
    lows = [s for s in sw if s[1] == "L"][-4:]
    hasil = []
    if len(highs) >= 2:
        (i1, _, p1), (i2, _, p2) = highs[-2], highs[-1]
        if p2 > p1 and rsi[i2] < rsi[i1] - 1.5:
            hasil.append(f"Bearish Divergence (Harga Higher High {p1:.2f}->{p2:.2f}, RSI Lower High {rsi[i1]:.1f}->{rsi[i2]:.1f} pada {wkt(df.index[i2])})")
    if len(lows) >= 2:
        (i1, _, p1), (i2, _, p2) = lows[-2], lows[-1]
        if p2 < p1 and rsi[i2] > rsi[i1] + 1.5:
            hasil.append(f"Bullish Divergence (Harga Lower Low {p1:.2f}->{p2:.2f}, RSI Higher Low {rsi[i1]:.1f}->{rsi[i2]:.1f} pada {wkt(df.index[i2])})")
    return hasil


def deteksi_institusi_dan_ranging(df, atr):
    """Mendeteksi:
    1) Institutional Order Flow / Impuls Satu Arah (Volume spike + Marubozu/bodi besar beruntun -> bahaya lawan arah / teknikal jebol terus)
    2) Fase Ranging / Sideways (Akumulasi/Distribusi) setelah impuls + apakah sudah ada Sweep/Manipulasi & CHoCH pembalikan."""
    O, H, L, C, V = (df[k].values for k in ("Open", "High", "Low", "Close", "Volume"))
    vol_ma = pd.Series(V).rolling(20, min_periods=5).mean().values
    hasil = []

    # --- 1. Deteksi Impuls Institusi Satu Arah (6 candle terakhir) ---
    bodi = np.abs(C[-6:] - O[-6:])
    rng = np.maximum(H[-6:] - L[-6:], 1e-6)
    atr6 = atr[-6:]
    vol_ratio = V[-6:] / np.maximum(vol_ma[-6:], 1)
    arah6 = np.sign(C[-6:] - O[-6:])

    # Cek apakah 4 dari 5 candle terakhir searah dengan bodi besar / volume melonjak
    naik_kuat = np.sum(arah6[-5:] > 0) >= 4 and (np.sum(bodi[-5:] > 1.2 * atr6[-5:]) >= 2 or np.max(vol_ratio[-5:]) >= 1.6)
    turun_kuat = np.sum(arah6[-5:] < 0) >= 4 and (np.sum(bodi[-5:] > 1.2 * atr6[-5:]) >= 2 or np.max(vol_ratio[-5:]) >= 1.6)
    ledakan_1candle = bodi[-1] >= 2.0 * atr[-1] and (bodi[-1] / rng[-1]) >= 0.7 and vol_ratio[-1] >= 1.5

    if naik_kuat or (ledakan_1candle and arah6[-1] > 0):
        hasil.append(
            f"🚨 PERINGATAN LONJAKAN MOMENTUM (TICK-VOLUME, BUKAN DATA INSTITUSI) (ONE-WAY BULLISH IMPULSE AKTIF)! "
            f"(Volume {np.max(vol_ratio[-5:]):.1f}x rata-rata, rentang gerak {H[-5:].max() - L[-5:].min():.1f} poin). "
            f"Zona Supply kecil berisiko JEBOL TERUS! DILARANG SELL lawan arah sebelum terbentuk Sideways + CHoCH turun!"
        )
    elif turun_kuat or (ledakan_1candle and arah6[-1] < 0):
        hasil.append(
            f"🚨 PERINGATAN LONJAKAN MOMENTUM (TICK-VOLUME, BUKAN DATA INSTITUSI) (ONE-WAY BEARISH IMPULSE AKTIF)! "
            f"(Volume {np.max(vol_ratio[-5:]):.1f}x rata-rata, rentang gerak {H[-5:].max() - L[-5:].min():.1f} poin). "
            f"Zona Demand kecil berisiko JEBOL TERUS! DILARANG BUY lawan arah (tangkap pisau jatuh) sebelum terbentuk Sideways + CHoCH naik!"
        )
    else:
        hasil.append("Aliran Order (Order Flow): Normal / Seimbang (tidak ada impuls satu arah ekstrem saat ini).")

    # --- 2. Deteksi Kotak Ranging / Sideways setelah Impuls + Sinyal Pembalikan (AMD / Wyckoff) ---
    found_box = None
    for win in (10, 14, 18):
        if len(df) < win + 12:
            continue
        # Cek kotak di akhir (sedang sideways sekarang) atau 1-3 candle sebelum sekarang (baru saja keluar dari sideways)
        for offset_akhir in (0, 2, 4):
            end_i = len(df) - offset_akhir
            start_i = end_i - win
            if start_i < 10:
                continue
            hi_box = H[start_i:end_i].max()
            lo_box = L[start_i:end_i].min()
            lebar_box = hi_box - lo_box
            atr_box = np.mean(atr[start_i:end_i])
            if lebar_box <= 2.6 * atr_box:
                # Cek tren/impuls sebelum kotak sideways terbentuk (10 candle sebelum start_i)
                pre_move = C[start_i] - C[start_i - 10]
                arah_impuls_awal = "NAIK (Bullish Impulse)" if pre_move > 1.5 * atr_box else ("TURUN (Bearish Impulse)" if pre_move < -1.5 * atr_box else "Netral")
                found_box = (start_i, end_i, lo_box, hi_box, arah_impuls_awal)
                break
        if found_box:
            break

    if found_box:
        s_i, e_i, lo_b, hi_b, imp_awal = found_box
        t_mulai, t_akhir = wkt(df.index[s_i]), wkt(df.index[e_i - 1])
        harga_skrg = C[-1]
        H_after, L_after, C_after = H[e_i - 2:], L[e_i - 2:], C[e_i - 2:]

        # Cek apakah ada tusukan manipulasi (Judas Swing / Sweep) di atas/bawah kotak lalu masuk lagi / pembalikan
        sweep_atas = bool((H_after > hi_b).any() and C[-1] < hi_b)
        sweep_bawah = bool((L_after < lo_b).any() and C[-1] > lo_b)
        break_atas = C[-1] > hi_b + 0.25 * atr[-1]
        break_bawah = C[-1] < lo_b - 0.25 * atr[-1]

        if sweep_atas:
            status_box = (
                f"🎯 SINYAL PEMBALIKAN BEARISH SETELAH RANGING! Harga sempat menyapu batas atas kotak ({hi_b:.2f}) "
                f"[Manipulasi/Distribution] lalu masuk kembali ke bawah {hi_b:.2f} (Close {harga_skrg:.2f}). Potensi kuat Reversal Turun!"
            )
        elif sweep_bawah:
            status_box = (
                f"🎯 SINYAL PEMBALIKAN BULLISH SETELAH RANGING! Harga sempat menyapu batas bawah kotak ({lo_b:.2f}) "
                f"[Manipulasi/Accumulation] lalu masuk kembali ke atas {lo_b:.2f} (Close {harga_skrg:.2f}). Potensi kuat Reversal Naik!"
            )
        elif break_atas:
            status_box = f"Harga sudah BREAKOUT NAIK keluar dari kotak sideways di atas {hi_b:.2f} (area retest demand baru: {hi_b - 0.5 * atr[-1]:.2f}-{hi_b:.2f})."
        elif break_bawah:
            status_box = f"Harga sudah BREAKOUT TURUN keluar dari kotak sideways di bawah {lo_b:.2f} (area retest supply baru: {lo_b:.2f}-{lo_b + 0.5 * atr[-1]:.2f})."
        else:
            status_box = (
                f"Harga SEDANG SIDEWAYS di dalam kotak ({lo_b:.2f} - {hi_b:.2f}). "
                f"Tunggu Sweep/Manipulasi di tepi kotak ({lo_b:.2f} atau {hi_b:.2f}) + CHoCH untuk tangkap pembalikan harga!"
            )
        hasil.append(
            f"[Ranging / Sideways Pasca-Impuls] Kotak konsolidasi {lo_b:.2f} - {hi_b:.2f} ({t_mulai} s/d {t_akhir}, pasca {imp_awal}):\n"
            f"    -> {status_box}"
        )
    else:
        hasil.append("[Ranging / Sideways Pasca-Impuls]: Tidak ada kotak sideways sempit saat ini (pasar sedang bergerak aktif/trending).")

    return "\n  ".join(hasil)


def supply_demand(df, atr, impuls=1.5, termasuk_tertembus=False):
    """Zona base -> impulse (candle dorongan besar didahului candle kecil/base).
    termasuk_tertembus=True: zona yang sudah ditembus close ikut dikembalikan (status 'mitigated')."""
    O, H, L, C = (df[k].values for k in ("Open", "High", "Low", "Close"))
    zona, last_k = [], -10
    for k in range(3, len(df)):
        if k - last_k < 3:
            continue
        if abs(C[k] - O[k]) < impuls * atr[k - 1]:
            continue
        base = []
        for x in range(k - 1, max(k - 4, -1), -1):
            if (H[x] - L[x]) <= 0.9 * atr[x]:
                base.append(x)
            else:
                break
        if not base:
            continue
        lo, hi = min(L[x] for x in base), max(H[x] for x in base)
        if hi - lo > 1.5 * atr[k - 1]:
            continue
        status = "fresh"
        if C[k] > O[k]:
            arah = "demand"
            if len(C[k + 1:]) and C[k + 1:].min() < lo:
                if not termasuk_tertembus:
                    continue
                status = "mitigated"
            elif len(L[k + 1:]) and L[k + 1:].min() <= hi:
                status = "tested"
        else:
            arah = "supply"
            if len(C[k + 1:]) and C[k + 1:].max() > hi:
                if not termasuk_tertembus:
                    continue
                status = "mitigated"
            elif len(H[k + 1:]) and H[k + 1:].max() >= lo:
                status = "tested"
        zona.append(dict(arah=arah, lo=lo, hi=hi, t=df.index[base[-1]], status=status))
        last_k = k
    return zona


def breakout_retest(df, events, atr, lookback=80):
    C, L, H = df["Close"].values, df["Low"].values, df["High"].values
    res = []
    for ev in events[-5:]:
        k, lvl = ev["k"], ev["level"]
        if k < len(df) - lookback:
            continue
        a = atr[k]
        status = "belum retest (menunggu harga kembali ke level)"
        for x in range(k + 1, len(df)):
            if ev["arah"] == "bull":
                if C[x] < lvl - 0.3 * a:
                    status = "GAGAL / fakeout (close kembali di bawah level)"
                    break
                if L[x] <= lvl + 0.3 * a:
                    status = "RETEST terjadi, level bertahan sebagai support"
            else:
                if C[x] > lvl + 0.3 * a:
                    status = "GAGAL / fakeout (close kembali di atas level)"
                    break
                if H[x] >= lvl - 0.3 * a:
                    status = "RETEST terjadi, level bertahan sebagai resistance"
        res.append((df.index[k], ev["arah"], lvl, status))
    return res


def fibonacci(df, harga, n=5):
    sw = idx_swings(df, n)
    if len(sw) < 2:
        return "data swing belum cukup"
    last = sw[-1]
    prev = next((s for s in reversed(sw[:-1]) if s[1] != last[1]), None)
    if prev is None:
        return "data swing belum cukup"
    Hh, Ll = df["High"].values, df["Low"].values

    def hitung(arah_naik, last, prev):
        if arah_naik:      # leg naik: low -> high, retracement turun
            low = prev[2] if prev else last[2]
            high = max(last[2], Hh[last[0]:].max()) if last[1] == "H" else Hh[last[0]:].max()
            rng = high - low
            return low, high, rng, ((high - harga) / rng if rng else 0)
        high = prev[2] if prev else last[2]   # leg turun: high -> low, retracement naik
        low = min(last[2], Ll[last[0]:].min()) if last[1] == "L" else Ll[last[0]:].min()
        rng = high - low
        return low, high, rng, ((harga - low) / rng if rng else 0)

    naik = last[1] == "H"
    low, high, rng, rasio = hitung(naik, last, prev)
    if rasio > 1 or rasio < 0:
        # leg lama sudah tertembus -> pakai leg dari swing terakhir ke ekstrem terbaru
        naik = last[1] == "L"
        low, high, rng, rasio = hitung(naik, last, None)
    arah = "NAIK (Low->High)" if naik else "TURUN (High->Low)"
    harga_lvl = (lambda r: high - r * rng) if naik else (lambda r: low + r * rng)
    lvl = " | ".join(f"{r:g}={harga_lvl(r):.2f}" for r in (0.382, 0.5, 0.618, 0.705, 0.786))
    if rasio > 1 or rasio < 0:
        zona = "harga di luar leg (belum ada retracement)"
    elif rasio >= 0.618:
        zona = "di area OTE / Golden Pocket (0.618-0.786) -> area reaksi terkuat"
    elif rasio >= 0.5:
        zona = "retracement dalam (0.5-0.618)"
    elif rasio >= 0.382:
        zona = "retracement dangkal (0.382-0.5)"
    else:
        zona = "retracement sangat dangkal/belum retrace"
    pd_ = "DISCOUNT (di bawah 0.5)" if (naik and rasio > 0.5) or (not naik and rasio < 0.5) else "PREMIUM (di atas 0.5)"
    if rasio > 1 or rasio < 0:
        pd_ = "harga sedang membuat high/low baru (breakout leg)"
    return (f"leg {arah} {low:.2f} -> {high:.2f}; harga sekarang retrace {rasio * 100:.0f}% ({zona}); {pd_}.\n"
            f"  Level: {lvl}")


def vwap_harian(df, offset):
    idx = df.index - pd.Timedelta(hours=offset)
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    v = df["Volume"].clip(lower=1)
    s = pd.DataFrame({"tpv": (tp * v).values, "v": v.values, "day": [t.date() for t in idx]})
    return (s.groupby("day")["tpv"].cumsum() / s.groupby("day")["v"].cumsum()).values


def volume_profile(df, harga, lookback=120, bins=24):
    """Menghitung Volume Profile (POC, VAH, VAL — 70% Value Area) dari distribusi tick volume."""
    sub = df.tail(lookback)
    lo_p, hi_p = sub["Low"].min(), sub["High"].max()
    if hi_p <= lo_p:
        return "Data tidak cukup", None, None, None
    edges = np.linspace(lo_p, hi_p, bins + 1)
    vols = np.zeros(bins)
    tp = ((sub["High"] + sub["Low"] + sub["Close"]) / 3).values
    v = sub["Volume"].values
    idx = np.clip(np.digitize(tp, edges) - 1, 0, bins - 1)
    for b_i, vol in zip(idx, v):
        vols[b_i] += vol
    poc_idx = int(np.argmax(vols))
    poc = 0.5 * (edges[poc_idx] + edges[poc_idx + 1])

    # Hitung 70% Value Area di sekitar POC
    total_v = vols.sum()
    cur_v = vols[poc_idx]
    l_i = r_i = poc_idx
    while cur_v < 0.70 * total_v and (l_i > 0 or r_i < bins - 1):
        v_left = vols[l_i - 1] if l_i > 0 else -1
        v_right = vols[r_i + 1] if r_i < bins - 1 else -1
        if v_left >= v_right:
            l_i -= 1
            cur_v += vols[l_i]
        else:
            r_i += 1
            cur_v += vols[r_i]
    val = edges[l_i]
    vah = edges[r_i + 1]
    pos = "DI ATAS Value Area (Premium/Breakout)" if harga > vah else ("DI BAWAH Value Area (Discount/Breakdown)" if harga < val else "DI DALAM Value Area (Seimbang, magnet ke POC)")
    teks = f"POC (Volume Terbesar): {poc:.2f} (jarak {harga - poc:+.1f}) | VAH: {vah:.2f} | VAL: {val:.2f} -> Posisi: {pos}"
    return teks, poc, val, vah


def analisa_smt_divergence(frames):
    """SMT Divergence XAUUSD vs XAGUSD. Swing perak dicari pada WAKTU yang sama dengan swing emas (+/-2 candle),
    dan hanya dihitung jika swing terbaru emas masih dalam 40 candle terakhir."""
    out = ["\n===== SMT DIVERGENCE (XAUUSD vs XAGUSD / SILVER — DETEKTOR MANIPULASI ICT) ====="]
    suf = _suffix_simbol()
    kandidat_silver = []
    if suf:
        kandidat_silver.append("XAGUSD" + suf)
    kandidat_silver += ["XAGUSD", "XAGUSDm", "SILVER", "XAGUSD.a"]
    sym_ag = next((s for s in kandidat_silver if mt5.symbol_select(s, True)), None)
    if not sym_ag:
        out.append("  - Simbol Perak (XAGUSD) tidak ditemukan di Market Watch broker.")
        return "\n".join(out) + "\n"

    for tf_nama, tf_mt5, menit in (("H1", mt5.TIMEFRAME_H1, 60), ("M30", mt5.TIMEFRAME_M30, 30)):
        rates_ag = mt5.copy_rates_from_pos(sym_ag, tf_mt5, 1, 120)
        if rates_ag is None or len(rates_ag) < 60:
            continue
        df_ag = pd.DataFrame(rates_ag)
        df_ag["time"] = pd.to_datetime(df_ag["time"], unit="s")
        df_ag = df_ag.set_index("time").rename(columns={"high": "High", "low": "Low", "close": "Close"})
        df_au = frames[tf_nama].tail(120)
        n_au = len(df_au)
        tol = pd.Timedelta(minutes=menit)

        def ekstrem_ag(t, kolom, fungsi):
            pos = df_ag.index.get_indexer([t], method="nearest")[0]
            if pos < 0 or abs(df_ag.index[pos] - t) > tol:
                return None
            return float(fungsi(df_ag[kolom].iloc[max(0, pos - 2):pos + 3]))

        sw_au = idx_swings(df_au, 3)
        h_au = [s for s in sw_au if s[1] == "H"][-2:]
        l_au = [s for s in sw_au if s[1] == "L"][-2:]

        smt_list = []
        if len(h_au) == 2 and h_au[1][0] >= n_au - 40:
            a1 = ekstrem_ag(df_au.index[h_au[0][0]], "High", max)
            a2 = ekstrem_ag(df_au.index[h_au[1][0]], "High", max)
            if a1 is not None and a2 is not None:
                au_hh = h_au[1][2] > h_au[0][2]
                ag_hh = a2 > a1
                if au_hh != ag_hh:
                    smt_list.append(
                        f"🎯 BEARISH SMT DIVERGENCE di {tf_nama}! "
                        f"({'XAUUSD buat Higher High tapi XAGUSD gagal/Lower High' if au_hh else 'XAGUSD buat Higher High tapi XAUUSD gagal/Lower High'}) "
                        f"-> Tanda Manipulasi Buy-Side & Reversal Turun!"
                    )
        if len(l_au) == 2 and l_au[1][0] >= n_au - 40:
            b1 = ekstrem_ag(df_au.index[l_au[0][0]], "Low", min)
            b2 = ekstrem_ag(df_au.index[l_au[1][0]], "Low", min)
            if b1 is not None and b2 is not None:
                au_ll = l_au[1][2] < l_au[0][2]
                ag_ll = b2 < b1
                if au_ll != ag_ll:
                    smt_list.append(
                        f"🎯 BULLISH SMT DIVERGENCE di {tf_nama}! "
                        f"({'XAUUSD buat Lower Low tapi XAGUSD menolak turun/Higher Low' if au_ll else 'XAGUSD buat Lower Low tapi XAUUSD bertahan/Higher Low'}) "
                        f"-> Tanda Manipulasi Sell-Side & Reversal Naik!"
                    )
        if smt_list:
            out.extend(f"  - {s}" for s in smt_list)
        else:
            out.append(f"  - {tf_nama}: Pergerakan XAUUSD & {sym_ag} masih searah/sinkron (tidak ada SMT Divergence pada swing terbaru).")
    return "\n".join(out) + "\n"


def analisa_metode(nama, df, offset, harga):
    """Analisis selektif: struktur, sweep, OB, FVG konfirmasi, dan trigger CISD.

    Supply/Demand, Breaker, Fibonacci, dan Volume Profile tidak lagi membuat
    zona atau menambah skor zona. EMA/RSI/VWAP hanya konteks pendukung.
    """
    d = tambah_indikator(df.copy())
    atr = d["ATR14"].values
    events, trend, sw = struktur(d)
    T = lambda i: wkt(d.index[i])
    out = [f"\n##### ANALISIS INTI SMC/ICT TF {nama} #####"]

    ttxt = {1: "BULLISH (HH-HL)", -1: "BEARISH (LH-LL)", 0: "NETRAL / BELUM JELAS"}[trend]
    out.append(f"[MARKET STRUCTURE] {ttxt}")
    for e in events[-3:]:
        out.append(f"  - {e['tipe']} {'bullish' if e['arah'] == 'bull' else 'bearish'}: close menembus {e['level']:.2f} ({len(d)-1-e['k']} candle lalu)")
    if not events:
        out.append("  - belum ada BOS/CHoCH terkonfirmasi")

    cisd_list, calon_cisd = deteksi_cisd(d, sw, lookback=35)
    out.append(f"[CISD / TRIGGER] {calon_cisd}")
    for item in cisd_list[-2:]:
        out.append(f"  - {wkt(item['waktu'])}: {item['ket']}")
    if not cisd_list:
        out.append("  - belum ada CISD baru yang terkonfirmasi")

    sweeps = liquidity_sweeps(d, sw, SWING_N, lookback=40)
    out.append("[LIQUIDITY SWEEP] (wick menembus swing lalu close kembali):")
    for t, ket, level, wick, close in sweeps[-2:]:
        out.append(f"  - {wkt(t)}: {ket}; level {level:.2f}, wick {wick:.2f}, close {close:.2f}")
    if not sweeps:
        out.append("  - tidak ada sweep valid")

    obs = order_blocks(d, events, maks_event=8)
    aktif = [o for o in obs if o["status"] == "fresh"]
    aktif.sort(key=lambda o: selisih_zona(o, harga))
    out.append("[ORDER BLOCK] (hanya candle lawan sebelum BOS/CHoCH; zona fresh):")
    for o in aktif[:3]:
        arah = "BULLISH DEMAND" if o["arah"] == "bull" else "BEARISH SUPPLY"
        out.append(f"  - {arah} {o['lo']:.2f}-{o['hi']:.2f} | fresh | asal {o['asal']} | {wkt(o['t'])} | {jarak_zona(o['lo'], o['hi'], harga)}")
    if not aktif:
        out.append("  - tidak ada OB fresh; OB tested/breaker tidak ditampilkan sebagai zona entry")

    fvgs = cari_fvg(d, harga, maks=3)
    out.append("[FVG] (hanya konfirmasi OB; >= 0.30 ATR; mitigated penuh dilarang):")
    for tipe, lo, hi, status, waktu in fvgs:
        out.append(f"  - {tipe} {lo:.2f}-{hi:.2f} | {status} | {wkt(waktu)}")
    if not fvgs:
        out.append("  - tidak ada FVG aktif berkualitas")

    eqh, eql = equal_highs_lows(d, sw, float(atr[-1]), harga)
    pool = [f"EQH {lvl:.2f}" for lvl, _, _ in eqh] + [f"EQL {lvl:.2f}" for lvl, _, _ in eql]
    out.append("[LIQUIDITY POOL] " + (", ".join(pool) if pool else "tidak ada EQH/EQL dekat"))
    div = rsi_divergence(d, sw)
    out.append("[RSI DIVERGENCE - pendukung, bukan trigger] " + ("; ".join(div) if div else "tidak ada"))
    out.append(f"[ORDER FLOW / RANGE] {deteksi_institusi_dan_ranging(d, atr)}")
    br = breakout_retest(d, events, atr)
    out.append("[BREAKOUT RETEST - konteks struktur] " + ("; ".join(f"{wkt(t)} {a} {lvl:.2f}: {st}" for t, a, lvl, st in br[-2:]) if br else "tidak ada breakout baru"))

    last = d.iloc[-1]
    out.append(f"[FILTER EMA] EMA20 {last['EMA20']:.2f} | EMA50 {last['EMA50']:.2f} | EMA200 {last['EMA200']:.2f} | Close {last['Close']:.2f}")
    out.append(f"[RSI/ATR] RSI14 {last['RSI14']:.1f} | ATR14 {last['ATR14']:.2f}")
    if nama in ("H1", "M30", "M5"):
        vw = vwap_harian(d, offset)[-1]
        out.append(f"[VWAP - filter pendukung] {vw:.2f}; harga {'di atas' if harga > vw else 'di bawah'} VWAP")
    out.append("[ATURAN ZONA] Hanya OB fresh yang bertumpuk/overlap dengan FVG aktif boleh menjadi kandidat zona. Tidak ada fallback ke S/D, breaker, Fibonacci, POC, atau swing.")
    return "\n".join(out) + "\n"

def cari_golden_confluence(frames, harga):
    """Zona A/A+ selektif menggunakan Python-Only Zone Scanner (OB fresh + FVG untouched).
    
    S/D, Breaker, swing, POC, Fibonacci dan metode lain tidak menjadi generator zona.
    OB dan FVG dari impuls yang sama diperlakukan sebagai satu setup, bukan dua konfirmasi independen.
    """
    try:
        from modules.zone_scanner import scan_xauusd_zones, evaluate_trade_decision, format_decision_text
        scan_res = scan_xauusd_zones(frames=frames, current_price=harga)
        teks = "\n" + scan_res.get("summary_text", "") + "\n"
        dec = evaluate_trade_decision(frames=frames, current_price=harga, live=True)
        teks += format_decision_text(dec) + "\n"
        teks += ("KEPUTUSAN PYTHON MENGIKAT: AI hanya boleh menafsirkan; jika status bukan VALID maka SINYAL_AKTIF wajib WAIT "
                 "dan AI dilarang menambah zona di luar daftar scanner.\n")
        return teks
    except Exception as e:
        _catat_error("cari_golden_confluence_scanner", e)
        return "\n===== RADAR ZONA SELEKTIF (OB + FVG) =====\n  - Gagal menjalankan scanner zona.\n"

def hitung_bias_htf(frames=None):
    """Gabungkan sinyal trend D1+H4+H1 menjadi satu bias terarah.
    Bobot: D1=2, H4=2, H1=1 -> total rentang -5..+5.
    Mengembalikan dict: bias ("BULLISH"/"BEARISH"/"NETRAL"), skor, detail per TF."""
    if not pastikan_mt5_aktif():
        return dict(bias="NETRAL", skor=0, detail={}, alasan="MT5 tidak aktif")
    konfigurasi = [
        ("D1", mt5.TIMEFRAME_D1, 100, 2),
        ("H4", mt5.TIMEFRAME_H4, 200, 2),
        ("H1", mt5.TIMEFRAME_H1, 200, 1),
    ]
    skor_total = 0
    detail = {}
    for nama, tf, jumlah, bobot in konfigurasi:
        df_tf = frames[nama].copy() if (frames and nama in frames) else ambil_data(tf, jumlah)
        if df_tf is None or len(df_tf) < 50:
            detail[nama] = "data tidak cukup (skor 0)"
            continue
        d = tambah_indikator(df_tf)
        last = d.iloc[-1]
        _, trend_val, _ = struktur(d)
        # Komponen: struktur + EMA stack + RSI + EMA200
        if last["EMA20"] > last["EMA50"] and last["Close"] > last["EMA20"]:
            skor_ema, label_ema = 1, "EMA bullish"
        elif last["EMA20"] < last["EMA50"] and last["Close"] < last["EMA20"]:
            skor_ema, label_ema = -1, "EMA bearish"
        else:
            skor_ema, label_ema = 0, "EMA netral"
        rsi = float(last["RSI14"])
        skor_rsi = 1 if rsi > 55 else (-1 if rsi < 45 else 0)
        label_rsi = f"RSI {rsi:.0f} ({'bullish' if skor_rsi==1 else 'bearish' if skor_rsi==-1 else 'netral'})"
        skor_ema200, label_ema200 = 0, ""
        if "EMA200" in d.columns:
            skor_ema200 = 1 if last["Close"] > last["EMA200"] else -1
            label_ema200 = f"EMA200 {'di bawah (bullish)' if skor_ema200==1 else 'di atas (bearish)'}"
        raw = trend_val + skor_ema + skor_rsi + skor_ema200
        arah_tf = 1 if raw >= 2 else (-1 if raw <= -2 else 0)
        skor_total += arah_tf * bobot
        label_str = {1: "BOS/trend bullish", -1: "BOS/trend bearish", 0: "belum jelas"}.get(trend_val, "?")
        detail[nama] = (f"{'↑' if arah_tf==1 else ('↓' if arah_tf==-1 else '→')} "
                        f"(struktur: {label_str}, {label_ema}, {label_rsi}"
                        f"{', ' + label_ema200 if label_ema200 else ''} | raw {raw:+d} -> skor {arah_tf:+d} x bobot {bobot})")
    bias = "BULLISH" if skor_total >= BIAS_MIN_SKOR else ("BEARISH" if skor_total <= -BIAS_MIN_SKOR else "NETRAL")
    return dict(bias=bias, skor=skor_total, detail=detail, alasan="")


def bias_htf_cached(frames=None, ttl=300):
    """Bias HTF dengan cache 5 menit. Gunakan di loop pantau dan gerbang sinyal."""
    if _CACHE_BIAS["data"] is None or (time.time() - _CACHE_BIAS["ts"]) > ttl:
        _CACHE_BIAS["data"] = hitung_bias_htf(frames)
        _CACHE_BIAS["ts"] = time.time()
    return _CACHE_BIAS["data"]


def invalidasi_cache_bias():
    """Paksa hitung ulang bias HTF (panggil saat candle D1/H4 baru close)."""
    _CACHE_BIAS["ts"] = 0.0


def teks_bias_htf(bias_dict):
    """Format ringkas untuk status bar dan konteks AI."""
    b = bias_dict
    icon = "🟢" if b["bias"] == "BULLISH" else ("🔴" if b["bias"] == "BEARISH" else "🟡")
    return f"{icon} BIAS HTF: {b['bias']} (skor {b['skor']:+d}/5)"


def _cari_zona_valid_berikutnya_mt5(arah_cari, batas_lewat, harga_skrg, atr_m5, is_institusi=False):
    """Cari zona lanjutan menggunakan Python-Only Zone Scanner (OB fresh + FVG untouched).
    
    Sepenuhnya independen dari AI, divalidasi oleh modules/zone_scanner.py.
    Jika tidak ada zona yang memenuhi syarat, kembalikan None, None (WAIT).
    """
    arah_cari = str(arah_cari).upper()
    if arah_cari not in ("BUY", "SELL") or not pastikan_mt5_aktif():
        return None, None

    try:
        from modules.zone_scanner import scan_xauusd_zones
        scan_res = scan_xauusd_zones(current_price=float(harga_skrg))
        active = scan_res.get("active_zones", [])
        target_type = "DEMAND" if arah_cari == "BUY" else "SUPPLY"
        matching = [z for z in active if z.get("zone_type") == target_type]
        if not matching:
            return None, None

        margin = max(1.0, 0.5 * float(atr_m5 or 0.0))
        batas_aman = (min(float(batas_lewat), float(harga_skrg)) - margin) if arah_cari == "BUY" else (max(float(batas_lewat), float(harga_skrg)) + margin)

        kandidat_aman = []
        for z in matching:
            lo_z, hi_z = float(z["price_low"]), float(z["price_high"])
            if arah_cari == "BUY" and hi_z > batas_aman:
                continue
            if arah_cari == "SELL" and lo_z < batas_aman:
                continue
            kandidat_aman.append(z)

        if not kandidat_aman:
            return None, None  # tanpa fallback: zona yang sudah terlewati tidak boleh dipakai ulang

        def bentuk_dari_scanner(z):
            lo_z, hi_z = round(float(z["price_low"]), 2), round(float(z["price_high"]), 2)
            sl_z = round(float(z["sl_recommended"]), 2)
            tp1_z = round(float(z["tp1_recommended"]), 2)
            tp2_z = round(float(z["tp2_recommended"]), 2)
            jarak_z = round(abs(float(harga_skrg) - (hi_z if target_type == "DEMAND" else lo_z)), 2)
            ket = f"Zona {arah_cari} [{z['grade']}]: {z['timeframe']} {z['zone_type']} (Skor {z['score']:.1f} | {z['freshness']})"
            return dict(
                arah=arah_cari,
                lo=lo_z,
                hi=hi_z,
                sl=sl_z,
                jebol=sl_z,
                tp1=tp1_z,
                tp2=tp2_z,
                ket=ket,
                sumber=ket,
                jarak=jarak_z,
                tipe_strategi=f"ZONE SCANNER {z['grade']}",
                grade=z["grade"],
                score=z["score"],
            )

        utama = bentuk_dari_scanner(kandidat_aman[0])
        cadangan = bentuk_dari_scanner(kandidat_aman[1]) if len(kandidat_aman) > 1 else None
        return utama, cadangan
    except Exception as e:
        _catat_error("_cari_zona_valid_berikutnya_mt5", e)
        return None, None

