import json
import os
import re
import shutil
import sys
import time
import urllib.request
import winsound
from datetime import datetime, timedelta, timezone

import numpy as np

import MetaTrader5 as mt5
import pandas as pd
from google import genai

API_KEY = os.getenv("GEMINI_API_KEY", "")

# Nama simbol bisa berbeda tiap broker (XAUUSD, XAUUSDm, GOLD, XAUUSD.a, dll.)
SYMBOL = "XAUUSDc"

# Top-down: W1/D1/H4 (zona jauh ke kiri) -> H1 -> M30 (konfirmasi) -> M5 (entry presisi)
TIMEFRAMES = {
    "W1": (mt5.TIMEFRAME_W1, 260),    # ~5 tahun
    "D1": (mt5.TIMEFRAME_D1, 1000),   # ~3-4 tahun
    "H4": (mt5.TIMEFRAME_H4, 3000),   # ~1.5-2 tahun, zona jauh ke kiri
    "H1": (mt5.TIMEFRAME_H1, 400),
    "M30": (mt5.TIMEFRAME_M30, 300),
    "M5": (mt5.TIMEFRAME_M5, 300),
}
SWING_N = 3  # sebuah candle = swing jika High/Low-nya ekstrem dibanding N candle kiri & kanan

# Selisih jam waktu server broker terhadap UTC (contoh: 0, 2, 3).
# None = deteksi otomatis dari tick MT5 (hanya akurat saat market buka).
SERVER_UTC_OFFSET_JAM = None


def ambil_data(tf, jumlah):
    # Posisi 0 = candle berjalan (belum selesai), jadi mulai dari 1
    rates = mt5.copy_rates_from_pos(SYMBOL, tf, 1, jumlah)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    # Waktu MT5 = waktu server broker (dibaca sebagai UTC), bukan WIB asli
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.set_index("time")
    return df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                              "close": "Close", "tick_volume": "Volume"})


def tambah_indikator(df):
    c = df["Close"]
    df["EMA20"] = c.ewm(span=20, adjust=False).mean()
    df["EMA50"] = c.ewm(span=50, adjust=False).mean()
    df["EMA200"] = c.ewm(span=200, adjust=False).mean()
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    df["RSI14"] = 100 - 100 / (1 + gain / loss)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - c.shift()).abs(),
        (df["Low"] - c.shift()).abs(),
    ], axis=1).max(axis=1)
    df["ATR14"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    return df


def cari_swing(df, n=SWING_N):
    """Swing high/low yang valid (sudah terkonfirmasi n candle di kanan)."""
    highs, lows = [], []
    for i in range(n, len(df) - n):
        jendela = df.iloc[i - n:i + n + 1]
        if df["High"].iloc[i] == jendela["High"].max():
            highs.append((df.index[i], df["High"].iloc[i]))
        if df["Low"].iloc[i] == jendela["Low"].min():
            lows.append((df.index[i], df["Low"].iloc[i]))
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


def cari_fvg(df, harga, maks=4):
    """FVG (3 candle). Jika yang belum terisi penuh kosong, longgarkan ukuran gap & sertakan FVG mitigated (flip zone)."""
    atr = tambah_indikator(df.copy())["ATR14"].values
    H, L = df["High"].values, df["Low"].values

    def scan(min_atr, ikut_terisi):
        zona = []
        for i in range(2, len(df)):
            if L[i] > H[i - 2] and (L[i] - H[i - 2]) >= min_atr * atr[i]:      # bullish FVG
                bawah, atas, tipe = H[i - 2], L[i], "Bullish FVG (demand)"
                sisa = L[i + 1:] if i + 1 < len(df) else []
                if len(sisa) and sisa.min() <= bawah:
                    if not ikut_terisi:
                        continue
                    status = "mitigated / flip zone"
                else:
                    status = "sebagian terisi" if len(sisa) and sisa.min() < atas else "belum tersentuh"
            elif H[i] < L[i - 2] and (L[i - 2] - H[i]) >= min_atr * atr[i]:    # bearish FVG
                bawah, atas, tipe = H[i], L[i - 2], "Bearish FVG (supply)"
                sisa = H[i + 1:] if i + 1 < len(df) else []
                if len(sisa) and sisa.max() >= atas:
                    if not ikut_terisi:
                        continue
                    status = "mitigated / flip zone"
                else:
                    status = "sebagian terisi" if len(sisa) and sisa.max() > bawah else "belum tersentuh"
            else:
                continue
            zona.append((tipe, bawah, atas, status, df.index[i - 1]))
        return zona

    zona = scan(0.3, False)
    if not any(z[1] >= harga for z in zona) or not any(z[2] <= harga for z in zona):
        zona = scan(0.15, False)
    if not any(z[1] >= harga for z in zona) or not any(z[2] <= harga for z in zona):
        zona = scan(0.2, True)
    diatas = sorted([z for z in zona if z[1] >= harga], key=lambda z: z[1])[:maks]
    dibawah = sorted([z for z in zona if z[2] <= harga], key=lambda z: -z[2])[:maks]
    dalam = [z for z in zona if z[1] < harga < z[2]][-2:]
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
    fmt_t = lambda t: t.strftime("%Y-%m-%d %H:%M")
    n = 3 if nama == "W1" else 5
    res, sup = swing_belum_ditembus(df, harga, n)
    fvg = cari_fvg(df, harga)
    teks = f"\n===== ZONA DARI CANDLE LAMA {nama} ({len(df)} candle, {fmt_t(df.index[0])} s/d {fmt_t(df.index[-1])}) =====\n"
    teks += "Resistance lama (potensi pantul turun/reversal): "
    teks += ", ".join(f"{p:.2f} [{st}] (swing {fmt_t(t)}, jarak {p - harga:.1f})" for t, p, st in res) or "tidak ada (harga di area ATH)"
    teks += "\nSupport lama (potensi pantul naik/reversal): "
    teks += ", ".join(f"{p:.2f} [{st}] (swing {fmt_t(t)}, jarak {harga - p:.1f})" for t, p, st in sup) or "-"
    teks += "\nFVG (belum terisi & flip zone terdekat):\n"
    if fvg:
        for tipe, b, a, st, t in fvg:
            posisi = "HARGA DI DALAM ZONA" if b < harga < a else (f"{b - harga:.1f} di atas harga" if b >= harga else f"{harga - a:.1f} di bawah harga")
            teks += f"  - {tipe} {b:.2f}-{a:.2f} | {st} | terbentuk {fmt_t(t)} | {posisi}\n"
    else:
        teks += "  - tidak ada\n"
    return teks


# =====================================================================
#  METODE SMC / ICT & KLASIK (semua dihitung di Python, bukan ditebak AI)
# =====================================================================
def wkt(t):
    return t.strftime("%m-%d %H:%M")


def jarak_zona(lo, hi, harga):
    if lo <= harga <= hi:
        return "HARGA DI DALAM ZONA"
    if lo > harga:
        return f"{lo - harga:.1f} di atas harga"
    return f"{harga - hi:.1f} di bawah harga"


def selisih_zona(z, harga):
    return 0 if z["lo"] <= harga <= z["hi"] else min(abs(z["lo"] - harga), abs(z["hi"] - harga))


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

    sw_highs = [s[2] for s in sw if s[1] == "H" and s[0] < start_idx]
    sw_lows = [s[2] for s in sw if s[1] == "L" and s[0] < start_idx]

    for m in range(start_idx, n_len - 1):
        # 1. Cek Titik Low Lokal (kandidat Bullish CISD)
        win_l = L[max(0, m - 2):min(n_len, m + 3)]
        if L[m] == win_l.min():
            # Cari candle bearish pada m atau m-1 yang mengantarkan ke low ini
            idx_bear = m if C[m] < O[m] else (m - 1 if m >= 1 and C[m - 1] < O[m - 1] else None)
            if idx_bear is not None:
                first_bear = idx_bear
                while first_bear > 0 and C[first_bear - 1] < O[first_bear - 1] and (idx_bear - first_bear) < 5:
                    first_bear -= 1
                lvl_open = float(O[first_bear])
                low_ekstrem = float(L[m])
                # Cek apakah dalam 1..5 candle setelah m ada yang CLOSE di atas lvl_open tanpa menjebol low_ekstrem
                for k in range(m + 1, min(n_len, m + 6)):
                    if L[k] < low_ekstrem:
                        break
                    if C[k] > lvl_open:
                        key = ("BULL", round(lvl_open, 2))
                        if key not in seen_level:
                            seen_level.add(key)
                            is_sweep = any(low_ekstrem < sl_p and C[m] >= sl_p for sl_p in sw_lows[-6:])
                            tag_sw = " [SWEEP + CISD VALID!]" if is_sweep else ""
                            hasil.append(dict(
                                k=k,
                                waktu=df.index[k],
                                arah="BULLISH",
                                level=lvl_open,
                                close=float(C[k]),
                                ekstrem=low_ekstrem,
                                ket=f"Bullish CISD{tag_sw} (Close {C[k]:.2f} > Open delivery {lvl_open:.2f}, SL di bawah {low_ekstrem:.2f})",
                            ))
                        break

        # 2. Cek Titik High Lokal (kandidat Bearish CISD)
        win_h = H[max(0, m - 2):min(n_len, m + 3)]
        if H[m] == win_h.max():
            idx_bull = m if C[m] > O[m] else (m - 1 if m >= 1 and C[m - 1] > O[m - 1] else None)
            if idx_bull is not None:
                first_bull = idx_bull
                while first_bull > 0 and C[first_bull - 1] > O[first_bull - 1] and (idx_bull - first_bull) < 5:
                    first_bull -= 1
                lvl_open = float(O[first_bull])
                high_ekstrem = float(H[m])
                for k in range(m + 1, min(n_len, m + 6)):
                    if H[k] > high_ekstrem:
                        break
                    if C[k] < lvl_open:
                        key = ("BEAR", round(lvl_open, 2))
                        if key not in seen_level:
                            seen_level.add(key)
                            is_sweep = any(high_ekstrem > sh_p and C[m] <= sh_p for sh_p in sw_highs[-6:])
                            tag_sw = " [SWEEP + CISD VALID!]" if is_sweep else ""
                            hasil.append(dict(
                                k=k,
                                waktu=df.index[k],
                                arah="BEARISH",
                                level=lvl_open,
                                close=float(C[k]),
                                ekstrem=high_ekstrem,
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
            f"🚨 PERINGATAN ORDERAN BESAR INSTITUSI (ONE-WAY BULLISH IMPULSE AKTIF)! "
            f"(Volume {np.max(vol_ratio[-5:]):.1f}x rata-rata, rentang gerak {H[-5:].max() - L[-5:].min():.1f} poin). "
            f"Zona Supply kecil berisiko JEBOL TERUS! DILARANG SELL lawan arah sebelum terbentuk Sideways + CHoCH turun!"
        )
    elif turun_kuat or (ledakan_1candle and arah6[-1] < 0):
        hasil.append(
            f"🚨 PERINGATAN ORDERAN BESAR INSTITUSI (ONE-WAY BEARISH IMPULSE AKTIF)! "
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


def analisa_metode(nama, df, offset, harga):
    d = tambah_indikator(df.copy())
    atr = d["ATR14"].values
    events, trend, sw = struktur(d)
    n_sw = SWING_N
    T = lambda i: wkt(d.index[i])
    out = [f"\n##### METODE SMC/ICT & KLASIK TF {nama} (waktu server broker) #####"]

    # 1. Struktur BOS / CHoCH
    ttxt = {1: "BULLISH (HH-HL)", -1: "BEARISH (LH-LL)", 0: "belum jelas"}[trend]
    out.append(f"[BOS/CHoCH] Struktur terakhir: {ttxt}")
    for e in events[-4:]:
        out.append(f"  - {e['tipe']} {'bullish' if e['arah'] == 'bull' else 'bearish'}: close menembus {e['level']:.2f} "
                   f"pada {T(e['k'])} ({len(d) - 1 - e['k']} candle lalu)")
    if not events:
        out.append("  - belum ada")

    # 1a. CISD (Change in State of Delivery - Konfirmasi Pembalikan Dini ICT)
    cisd_list, calon_cisd = deteksi_cisd(d, sw, lookback=35)
    out.append(f"[CISD (Change in State of Delivery - ICT)] Trigger Calon Saat Ini: {calon_cisd}")
    for c_item in cisd_list[-3:]:
        out.append(f"  - {wkt(c_item['waktu'])} ({len(d) - 1 - c_item['k']} candle lalu): {c_item['ket']}")
    if not cisd_list:
        out.append("  - belum ada CISD terkonfirmasi baru-baru ini")

    # 1b. Deteksi Orderan Besar Institusi (One-Way Impulse) & Pola Sideways -> Pembalikan (AMD)
    out.append(f"[Order Flow Institusi & Sideways Reversal]:\n  {deteksi_institusi_dan_ranging(d, atr)}")

    # 2. Liquidity sweep
    sweeps = liquidity_sweeps(d, sw, n_sw)
    out.append("[Liquidity Sweep] 40 candle terakhir:")
    for t, ket, p, wick, cl in sweeps[-3:]:
        out.append(f"  - {wkt(t)}: {ket}; level {p:.2f}, wick {wick:.2f}, close {cl:.2f}")
    if not sweeps:
        out.append("  - tidak ada sweep valid")

    # 3. Order Block & 4. Breaker Block
    obs = order_blocks(d, events)
    aktif = sorted([o for o in obs if o["status"] != "breaker"], key=lambda o: selisih_zona(o, harga))[:3]
    brk = sorted([o for o in obs if o["status"] == "breaker"], key=lambda o: selisih_zona(o, harga))[:2]
    out.append("[Order Block] (candle lawan terakhir sebelum BOS/CHoCH):")
    for o in aktif:
        nm = "Bullish OB (demand)" if o["arah"] == "bull" else "Bearish OB (supply)"
        out.append(f"  - {nm} {o['lo']:.2f}-{o['hi']:.2f} | {o['status']} | candle {wkt(o['t'])} | asal {o['asal']} | {jarak_zona(o['lo'], o['hi'], harga)}")
    if not aktif:
        out.append("  - tidak ada OB aktif")
    out.append("[Breaker Block] (OB gagal yang ditembus lalu berbalik fungsi):")
    for o in brk:
        nm = "Bearish Breaker (supply)" if o["arah"] == "bull" else "Bullish Breaker (demand)"
        out.append(f"  - {nm} {o['lo']:.2f}-{o['hi']:.2f} | {'sudah di-retest' if o['retest'] else 'belum di-retest'} | {jarak_zona(o['lo'], o['hi'], harga)}")
    if not brk:
        out.append("  - tidak ada breaker")

    # 5. Supply & Demand (dengan pelonggaran bertahap + fallback zona mitigated jika kosong)
    sd = supply_demand(d, atr, impuls=1.5, termasuk_tertembus=False)
    dem = [z for z in sd if z["arah"] == "demand"]
    sup_ = [z for z in sd if z["arah"] == "supply"]
    if not dem or not sup_:
        sd_longgar = supply_demand(d, atr, impuls=1.15, termasuk_tertembus=False)
        if not dem:
            dem = [z for z in sd_longgar if z["arah"] == "demand"]
        if not sup_:
            sup_ = [z for z in sd_longgar if z["arah"] == "supply"]
    if not dem or not sup_:
        sd_mit = supply_demand(d, atr, impuls=1.15, termasuk_tertembus=True)
        if not dem:
            dem = [z for z in sd_mit if z["arah"] == "demand"][-3:]
        if not sup_:
            sup_ = [z for z in sd_mit if z["arah"] == "supply"][-3:]
    sd_pilih = sorted(dem, key=lambda z: selisih_zona(z, harga))[:2] + \
               sorted(sup_, key=lambda z: selisih_zona(z, harga))[:2]
    out.append("[Supply & Demand] (base -> impulse, otomatis melonggar jika kosong):")
    for z in sd_pilih:
        out.append(f"  - {z['arah'].upper()} {z['lo']:.2f}-{z['hi']:.2f} | {z['status']} | base {wkt(z['t'])} | {jarak_zona(z['lo'], z['hi'], harga)}")
    if not sd_pilih:
        out.append("  - tidak ada zona valid")

    # 6. Breakout & Retest
    br = breakout_retest(d, events, atr)
    out.append("[Breakout & Retest] (level BOS/CHoCH 80 candle terakhir):")
    for t, arah, lvl, st in br:
        out.append(f"  - {wkt(t)} breakout {'naik' if arah == 'bull' else 'turun'} level {lvl:.2f}: {st}")
    if not br:
        out.append("  - tidak ada breakout baru")

    # 7. Fibonacci
    out.append(f"[Fibonacci Retracement] {fibonacci(d, harga)}")

    # 7b. Equal Highs / Equal Lows (EQH/EQL) & RSI Divergence
    eqh, eql = equal_highs_lows(d, sw, atr[-1], harga)
    eq_txt = []
    for lvl, t1, t2 in eqh:
        eq_txt.append(f"EQH (Buy-Side Liquidity Pool) di {lvl:.2f} ({wkt(t1)} & {wkt(t2)}, jarak {lvl - harga:+.1f})")
    for lvl, t1, t2 in eql:
        eq_txt.append(f"EQL (Sell-Side Liquidity Pool) di {lvl:.2f} ({wkt(t1)} & {wkt(t2)}, jarak {lvl - harga:+.1f})")
    out.append("[Equal Highs/Lows (Magnet Likuiditas)]: " + ("; ".join(eq_txt) if eq_txt else "tidak ada EQH/EQL dekat"))

    div = rsi_divergence(d, sw)
    out.append("[RSI Divergence]: " + ("; ".join(div) if div else "tidak ada divergence"))

    # 8. MA, VWAP & Volume Profile (POC/VAH/VAL)
    last = d.iloc[-1]
    ma = (f"[MA] EMA20 {last['EMA20']:.2f}, EMA50 {last['EMA50']:.2f}, EMA200 {last['EMA200']:.2f}; "
          f"harga {'di atas' if harga > last['EMA200'] else 'di bawah'} EMA200, "
          f"{'di atas' if harga > last['EMA50'] else 'di bawah'} EMA50, "
          f"{'di atas' if harga > last['EMA20'] else 'di bawah'} EMA20")
    out.append(ma)
    vp_txt, _, _, _ = volume_profile(d, harga)
    out.append(f"[Volume Profile] {vp_txt}")
    if nama in ("H1", "M30", "M5"):
        vw = vwap_harian(d, offset)[-1]
        out.append(f"[VWAP harian] {vw:.2f}; harga {'di atas' if harga > vw else 'di bawah'} VWAP "
                   f"({harga - vw:+.1f}) -> {'bias bullish intraday' if harga > vw else 'bias bearish intraday'}")
    return "\n".join(out) + "\n"


def us_dst(dt):
    d = datetime(dt.year, 3, 8)
    while d.weekday() != 6:
        d += timedelta(days=1)
    mulai = d + timedelta(hours=7)
    e = datetime(dt.year, 11, 1)
    while e.weekday() != 6:
        e += timedelta(days=1)
    return mulai <= dt < e + timedelta(hours=6)


def deteksi_offset(tick):
    if tick is None:
        return None
    diff = (tick.time - time.time()) / 3600
    off = round(diff)
    return off if (-3 <= off <= 5 and abs(diff - off) < 0.25) else None


def analisa_sesi(m30, m5, offset):
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    ny_off = -4 if us_dst(now) else -5
    ny = now + timedelta(hours=ny_off)
    h = ny.hour + ny.minute / 60
    hu = now.hour + now.minute / 60
    wib = now + timedelta(hours=7)
    to_wib = lambda hn: f"{int((hn + 7 - ny_off) % 24):02d}:00"

    sesi = [nm for nm, a, b in (("Asia/Tokyo", 0, 8), ("London", 7, 16), ("New York", 12, 21)) if a <= hu < b]
    if h >= 20:
        kz = "Asia Kill Zone"
    elif 2 <= h < 5:
        kz = "London Open Kill Zone"
    elif 7 <= h < 10:
        kz = "New York Open Kill Zone"
    elif 10 <= h < 12:
        kz = "London Close Kill Zone"
    else:
        kz = None

    out = ["\n##### SESI TRADING & KILL ZONE #####",
           f"Waktu sekarang: {wib:%Y-%m-%d %H:%M} WIB | {now:%H:%M} UTC | {ny:%H:%M} New York (offset server broker: UTC{offset:+d})",
           f"Sesi aktif: {', '.join(sesi) if sesi else 'di antara sesi (sepi)'}",
           f"Kill Zone sekarang: {kz if kz else 'TIDAK sedang di kill zone (probabilitas setup lebih rendah)'}",
           f"Jadwal kill zone hari ini (WIB): London Open {to_wib(2)}-{to_wib(5)} | New York Open {to_wib(7)}-{to_wib(10)} | London Close {to_wib(10)}-{to_wib(12)}"]
    if now.weekday() >= 5:
        out.append("CATATAN: akhir pekan, market emas tutup.")

    idx = m30.index - pd.Timedelta(hours=offset)
    x = pd.DataFrame({"H": m30["High"].values, "L": m30["Low"].values,
                      "day": [t.date() for t in idx], "hr": [t.hour + t.minute / 60 for t in idx]})
    ref_day = idx[-1].date()
    hari_lalu = sorted(set(d for d in x["day"] if d < ref_day))
    today = x[x["day"] == ref_day]
    prev = x[x["day"] == hari_lalu[-1]] if hari_lalu else x.iloc[0:0]
    asia = today[today["hr"] < 8]
    lon = today[(today["hr"] >= 7) & (today["hr"] < 16)]
    mulai_hari = datetime(ref_day.year, ref_day.month, ref_day.day)
    levels = []
    if len(prev):
        levels += [("Previous Day High (PDH)", prev["H"].max(), "H", mulai_hari),
                   ("Previous Day Low (PDL)", prev["L"].min(), "L", mulai_hari)]
    if len(asia):
        sel = mulai_hari + timedelta(hours=8)
        levels += [("Asia High", asia["H"].max(), "H", sel), ("Asia Low", asia["L"].min(), "L", sel)]
    if len(lon):
        sel = mulai_hari + timedelta(hours=16)
        levels += [("London High", lon["H"].max(), "H", sel), ("London Low", lon["L"].min(), "L", sel)]

    idx5 = m5.index - pd.Timedelta(hours=offset)
    out.append("Level likuiditas sesi (target stop hunt) & status M5:")
    for nm, val, tipe, sel in levels:
        sel_ts = pd.Timestamp(sel)
        msk = idx5 >= sel_ts
        status = "range sesi belum selesai" if sel_ts > idx5[-1] else "belum disentuh"
        if sel_ts <= idx5[-1] and msk.any():
            H5, L5, C5 = (m5[k].values[msk] for k in ("High", "Low", "Close"))
            tt = idx5[msk]
            if tipe == "H":
                sw_ = np.where((H5 > val) & (C5 < val))[0]
                br_ = np.where(C5 > val)[0]
            else:
                sw_ = np.where((L5 < val) & (C5 > val))[0]
                br_ = np.where(C5 < val)[0]
            if len(sw_):
                status = f"SUDAH DISAPU (sweep) pada {wkt(tt[sw_[-1]] + pd.Timedelta(hours=offset))} lalu close balik"
            elif len(br_):
                status = "ditembus (close di luar level)"
        out.append(f"  - {nm}: {val:.2f} | {status}")
    return "\n".join(out) + "\n"


def cari_golden_confluence(frames, harga):
    """Menggabungkan seluruh zona (FVG, OB, Breaker, S/D, Swing Lama, Fibo OTE) lintas timeframe
    yang saling bertumpuk (overlap) menjadi 'Golden Confluence Zones' beserta skor kekuatannya."""
    kandidat = []
    bobot_tf = {"W1": 3, "D1": 3, "H4": 2, "H1": 2, "M30": 1}

    for tf in ("W1", "D1", "H4", "H1", "M30"):
        df = tambah_indikator(frames[tf].copy())
        atr_v = df["ATR14"].iloc[-1]
        w = bobot_tf[tf]

        # Volume Profile POC (H4, H1)
        if tf in ("H4", "H1"):
            _, poc, _, _ = volume_profile(df, harga)
            if poc is not None:
                arah_poc = "DEMAND" if poc < harga else "SUPPLY"
                kandidat.append(dict(arah=arah_poc, lo=poc - 0.15 * atr_v, hi=poc + 0.15 * atr_v,
                                     komponen=f"{tf} Volume POC {poc:.2f}", skor=w))

        # FVG
        for tipe, b, a, st, t in cari_fvg(df, harga, maks=4):
            arah = "DEMAND" if "Bullish" in tipe else "SUPPLY"
            kandidat.append(dict(arah=arah, lo=b, hi=a, komponen=f"{tf} FVG ({st})", skor=w))

        # Swing Lama (W1, D1, H4)
        if tf in ("W1", "D1", "H4"):
            res, sup = swing_belum_ditembus(df, harga, n=3 if tf == "W1" else 5, maks=3)
            for t, p, st in res:
                kandidat.append(dict(arah="SUPPLY", lo=p - 0.15 * atr_v, hi=p + 0.15 * atr_v,
                                     komponen=f"{tf} Swing High {p:.2f}", skor=w))
            for t, p, st in sup:
                kandidat.append(dict(arah="DEMAND", lo=p - 0.15 * atr_v, hi=p + 0.15 * atr_v,
                                     komponen=f"{tf} Swing Low {p:.2f}", skor=w))

        # OB, Breaker, Supply/Demand (H4, H1, M30)
        if tf in ("H4", "H1", "M30"):
            evs, _, _ = struktur(df)
            for o in order_blocks(df, evs):
                if o["status"] == "breaker":
                    arah = "SUPPLY" if o["arah"] == "bull" else "DEMAND"
                    kandidat.append(dict(arah=arah, lo=o["lo"], hi=o["hi"],
                                         komponen=f"{tf} Breaker Block", skor=w))
                else:
                    arah = "DEMAND" if o["arah"] == "bull" else "SUPPLY"
                    kandidat.append(dict(arah=arah, lo=o["lo"], hi=o["hi"],
                                         komponen=f"{tf} OB ({o['status']})", skor=w))
            for z in supply_demand(df, df["ATR14"].values, impuls=1.2, termasuk_tertembus=False):
                arah = "DEMAND" if z["arah"] == "demand" else "SUPPLY"
                kandidat.append(dict(arah=arah, lo=z["lo"], hi=z["hi"],
                                     komponen=f"{tf} {arah.title()} Base", skor=w))

    # Gabungkan zona dengan arah sama yang saling overlap / berdekatan (<= 2.0 poin)
    clusters = []
    for z in sorted(kandidat, key=lambda x: x["lo"]):
        merged = False
        for c in clusters:
            if c["arah"] == z["arah"] and z["lo"] <= c["hi"] + 2.0 and z["hi"] >= c["lo"] - 2.0:
                # Persempit/fokuskan ke irisan atau rata-rata tertimbang agar zona tetap tajam
                c["lo"] = max(c["lo"], z["lo"]) if max(c["lo"], z["lo"]) < min(c["hi"], z["hi"]) else min(c["lo"], z["lo"])
                c["hi"] = min(c["hi"], z["hi"]) if c["lo"] < min(c["hi"], z["hi"]) else max(c["hi"], z["hi"])
                if z["komponen"] not in c["komponen"]:
                    c["komponen"].append(z["komponen"])
                    c["skor"] += z["skor"]
                merged = True
                break
        if not merged:
            clusters.append(dict(arah=z["arah"], lo=z["lo"], hi=z["hi"],
                                 komponen=[z["komponen"]], skor=z["skor"]))

    multi = [c for c in clusters if len(c["komponen"]) >= 2]
    multi.sort(key=lambda c: (-c["skor"], selisih_zona(c, harga)))

    out = ["\n===== GOLDEN CONFLUENCE ZONES (ZONA BERTUMPUK MULTI-METODE & MULTI-TF) ====="]
    dem_top = [c for c in multi if c["arah"] == "DEMAND" and c["lo"] <= harga + 5][:4]
    sup_top = [c for c in multi if c["arah"] == "SUPPLY" and c["hi"] >= harga - 5][:4]

    for label, lst in (("ZONA DEMAND (BUY) TERKUAT", dem_top), ("ZONA SUPPLY (SELL) TERKUAT", sup_top)):
        out.append(f"{label}:")
        if not lst:
            out.append("  - Tidak ada zona bertumpuk dekat harga")
            continue
        for c in lst:
            bintang = "★" * min(5, max(1, c["skor"] // 2))
            out.append(
                f"  - [{bintang} Skor {c['skor']}] {c['arah']} {c['lo']:.2f} - {c['hi']:.2f} "
                f"({jarak_zona(c['lo'], c['hi'], harga)}) | Gabungan {len(c['komponen'])} faktor: {', '.join(c['komponen'])}"
            )
    return "\n".join(out) + "\n"


def analisa_akun_dan_lot(acc_info, sym_info, atr_m5, harga):
    """Membaca saldo MT5 asli & menghitung ukuran lot otomatis untuk risiko 0.5% (Medium) dan 1% (Low)."""
    out = ["\n===== MANAJEMEN RISIKO & KALKULATOR LOT OTOMATIS (AKUN MT5) ====="]
    if acc_info is None or sym_info is None:
        out.append("Info akun MT5 tidak tersedia.")
        return "\n".join(out) + "\n"

    bal = acc_info.balance
    eq = acc_info.equity
    cur = acc_info.currency
    contract = sym_info.trade_contract_size or 100.0
    vol_min = sym_info.volume_min or 0.01
    vol_step = sym_info.volume_step or 0.01

    # Nilai $ per 1.00 poin pergerakan harga emas untuk 1.00 lot
    nilai_per_poin_1lot = contract  # XAUUSD standar: 100 oz -> $100 per $1 gerak harga per 1 lot
    out.append(f"Saldo (Balance): {bal:.2f} {cur} | Equity: {eq:.2f} {cur} | Contract Size: {contract:g} | Lot Min: {vol_min}")

    risk_low = eq * 0.01      # 1.0% untuk LOW RISK
    risk_med = eq * 0.005     # 0.5% untuk MEDIUM RISK
    out.append(f"Batas Risiko per Trade -> LOW RISK (1.0%): ${risk_low:.2f} | MEDIUM RISK (0.5%): ${risk_med:.2f}")
    out.append("Tabel Acuan Ukuran Lot berdasarkan jarak Stop Loss (poin/dolar emas):")
    for sl_poin in (3.0, 5.0, 8.0, 10.0, 15.0):
        rugi_per_lot = sl_poin * nilai_per_poin_1lot
        lot_low = max(vol_min, round((risk_low / rugi_per_lot) / vol_step) * vol_step) if rugi_per_lot else vol_min
        lot_med = max(vol_min, round((risk_med / rugi_per_lot) / vol_step) * vol_step) if rugi_per_lot else vol_min
        out.append(f"  - Jika jarak SL = {sl_poin:4.1f} poin ({sl_poin * 10:.0f} pips) -> Lot LOW RISK (1%): {lot_low:.2f} lot | Lot MEDIUM RISK (0.5%): {lot_med:.2f} lot")
    return "\n".join(out) + "\n"


def hitung_mm_sinyal(entry_lo, entry_hi, sl, tp1=0.0, tp2=0.0):
    """Menghitung Money Management presisi berdasarkan Equity akun MT5 asli dan jarak SL sinyal:
    - Ukuran lot rekomendasi (0.5% disiplin, 1.0% standar)
    - Batas keras maksimal lot (2.0% safety)
    - Nilai risiko dalam Dollar ($)
    - Rasio Risk to Reward (R:R) TP1 & TP2"""
    if not pastikan_mt5_aktif():
        return None
    acc = mt5.account_info()
    sym = mt5.symbol_info(SYMBOL)
    if not acc or not sym:
        return None

    eq = float(acc.equity if acc.equity > 0 else acc.balance)
    cur = acc.currency
    contract = float(sym.trade_contract_size or 100.0)
    vol_min = float(sym.volume_min or 0.01)
    vol_step = float(sym.volume_step or 0.01)
    vol_max = float(sym.volume_max or 100.0)

    entry_mid = (float(entry_lo) + float(entry_hi)) / 2.0
    jarak_sl = abs(entry_mid - float(sl))
    if jarak_sl <= 0.01:
        return None

    rugi_per_lot = jarak_sl * contract

    risk_025_usd = eq * 0.0025  # 0.25% High Risk / Mini Lot
    risk_05_usd = eq * 0.005   # 0.5% Medium Risk
    risk_10_usd = eq * 0.010   # 1.0% Low Risk
    risk_20_usd = eq * 0.020   # 2.0% Max Risk

    lot_025 = max(vol_min, min(vol_max, round((risk_025_usd / rugi_per_lot) / vol_step) * vol_step))
    lot_05 = max(vol_min, min(vol_max, round((risk_05_usd / rugi_per_lot) / vol_step) * vol_step))
    lot_10 = max(vol_min, min(vol_max, round((risk_10_usd / rugi_per_lot) / vol_step) * vol_step))
    lot_max = max(vol_min, min(vol_max, round((risk_20_usd / rugi_per_lot) / vol_step) * vol_step))

    real_risk_025 = lot_025 * rugi_per_lot
    real_risk_05 = lot_05 * rugi_per_lot
    real_risk_10 = lot_10 * rugi_per_lot
    real_risk_max = lot_max * rugi_per_lot

    jarak_tp1 = abs(float(tp1) - entry_mid) if tp1 > 0 else 0
    jarak_tp2 = abs(float(tp2) - entry_mid) if tp2 > 0 else 0
    rrr_tp1 = (jarak_tp1 / jarak_sl) if jarak_sl > 0 else 0
    rrr_tp2 = (jarak_tp2 / jarak_sl) if jarak_sl > 0 else 0

    return {
        "equity": eq,
        "balance": float(acc.balance),
        "currency": cur,
        "jarak_sl": jarak_sl,
        "lot_025": lot_025,
        "lot_05": lot_05,
        "lot_10": lot_10,
        "lot_max": lot_max,
        "risk_025_usd": real_risk_025,
        "risk_05_usd": real_risk_05,
        "risk_10_usd": real_risk_10,
        "risk_max_usd": real_risk_max,
        "rrr_tp1": rrr_tp1,
        "rrr_tp2": rrr_tp2,
    }


def analisa_korelasi_usd():
    """Mengecek kekuatan US Dollar (DXY / EURUSD / USDJPY) di MT5 untuk konfirmasi korelasi XAUUSD."""
    out = ["\n===== KORELASI US DOLLAR (KONFIRMASI MAKRO) ====="]
    kandidat_simbol = [
        ("USDX", "DXY", 1), ("DXY", "DXY", 1), ("DX", "DXY", 1), ("USDX.a", "DXY", 1),
        ("EURUSD", "EURUSD", -1), ("EURUSDm", "EURUSD", -1),
        ("GBPUSD", "GBPUSD", -1), ("GBPUSDm", "GBPUSD", -1),
        ("USDJPY", "USDJPY", 1), ("USDJPYm", "USDJPY", 1),
    ]
    ditemukan = []
    for sym, grup, arah_usd in kandidat_simbol:
        if any(g == grup for g, _ in ditemukan):
            continue
        if mt5.symbol_select(sym, True):
            rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_H1, 1, 60)
            if rates is not None and len(rates) >= 50:
                df = pd.DataFrame(rates)
                c = df["close"]
                ema20 = c.ewm(span=20, adjust=False).mean().iloc[-1]
                ema50 = c.ewm(span=50, adjust=False).mean().iloc[-1]
                chg = (c.iloc[-1] - c.iloc[-5]) / c.iloc[-5] * 100
                if (c.iloc[-1] > ema20 > ema50 and arah_usd == 1) or (c.iloc[-1] < ema20 < ema50 and arah_usd == -1):
                    status_usd = "USD MENGUAT (tekanan BEARISH untuk XAUUSD)"
                elif (c.iloc[-1] < ema20 < ema50 and arah_usd == 1) or (c.iloc[-1] > ema20 > ema50 and arah_usd == -1):
                    status_usd = "USD MELEMAH (dorongan BULLISH untuk XAUUSD)"
                else:
                    status_usd = "USD NETRAL / SIDEWAYS"
                ditemukan.append((grup, f"  - {sym} (H1): Close {c.iloc[-1]:.4f} (perubahan 4 jam: {chg:+.2f}%) -> {status_usd}"))
    if ditemukan:
        out.extend(teks for _, teks in ditemukan)
    else:
        out.append("  - Simbol korelasi USD tidak tersedia.")
    return "\n".join(out) + "\n"


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


def analisa_adr_exhaustion(df_d1, harga):
    """Menghitung Average Daily Range (ADR 14 hari) untuk mendeteksi apakah 'bensin' gerak harian sudah habis."""
    rng_harian = (df_d1["High"] - df_d1["Low"]).iloc[-15:-1]
    adr14 = rng_harian.mean()
    hari_ini = df_d1.iloc[-1]
    hi_today = max(hari_ini["High"], harga)
    lo_today = min(hari_ini["Low"], harga)
    range_skrg = hi_today - lo_today
    persen = (range_skrg / adr14 * 100) if adr14 > 0 else 0

    proj_high = lo_today + adr14
    proj_low = hi_today - adr14

    if persen >= 100:
        status = f"⚠️ EXHAUSTION / BENSIN HABIS ({persen:.0f}% dari ADR)! Rawan Reversal / Pullback tajam, JANGAN kejar pucuk/dasar!"
    elif persen >= 80:
        status = f"Hampir Penuh ({persen:.0f}% dari ADR) -> mulai waspada potensi koreksi/profit taking institusi."
    else:
        status = f"Masih Ada Ruang Gerak ({persen:.0f}% dari ADR terpakai)."

    return (
        f"\n===== ADR (AVERAGE DAILY RANGE 14 HARI) & BATAS BENSIN HARIAN =====\n"
        f"Rata-rata Range Harian (ADR14): {adr14:.2f} poin ($)\n"
        f"Range Hari Ini: {range_skrg:.2f} poin (Low {lo_today:.2f} s/d High {hi_today:.2f}) -> {status}\n"
        f"Proyeksi Batas Atap Harian (ADR High): {proj_high:.2f} | Batas Lantai Harian (ADR Low): {proj_low:.2f}\n"
    )


def analisa_smt_divergence(frames):
    """Mengecek SMT Divergence (Smart Money Technique) antara Emas (XAUUSD) dan Perak (XAGUSD) pada H1 dan M30."""
    out = ["\n===== SMT DIVERGENCE (XAUUSD vs XAGUSD / SILVER — DETEKTOR MANIPULASI ICT) ====="]
    kandidat_silver = ["XAGUSD", "XAGUSDm", "SILVER", "XAGUSD.a"]
    sym_ag = next((s for s in kandidat_silver if mt5.symbol_select(s, True)), None)
    if not sym_ag:
        out.append("  - Simbol Perak (XAGUSD) tidak ditemukan di Market Watch broker.")
        return "\n".join(out) + "\n"

    for tf_nama, tf_mt5 in (("H1", mt5.TIMEFRAME_H1), ("M30", mt5.TIMEFRAME_M30)):
        rates_ag = mt5.copy_rates_from_pos(sym_ag, tf_mt5, 1, 120)
        if rates_ag is None or len(rates_ag) < 60:
            continue
        df_ag = pd.DataFrame(rates_ag).rename(columns={"high": "High", "low": "Low", "close": "Close"})
        df_au = frames[tf_nama].tail(120)

        # Bandingkan 2 swing high & 2 swing low terakhir dalam 40 candle terakhir
        sw_au = idx_swings(df_au, 3)
        sw_ag = idx_swings(df_ag, 3)
        h_au = [s for s in sw_au if s[1] == "H"][-2:]
        l_au = [s for s in sw_au if s[1] == "L"][-2:]
        h_ag = [s for s in sw_ag if s[1] == "H"][-2:]
        l_ag = [s for s in sw_ag if s[1] == "L"][-2:]

        smt_list = []
        if len(h_au) == 2 and len(h_ag) == 2:
            au_hh = h_au[1][2] > h_au[0][2]
            ag_hh = h_ag[1][2] > h_ag[0][2]
            if au_hh != ag_hh:
                smt_list.append(
                    f"🎯 BEARISH SMT DIVERGENCE di {tf_nama}! "
                    f"({'XAUUSD buat Higher High tapi XAGUSD gagal/Lower High' if au_hh else 'XAGUSD buat Higher High tapi XAUUSD gagal/Lower High'}) "
                    f"-> Tanda Manipulasi Buy-Side & Reversal Turun!"
                )
        if len(l_au) == 2 and len(l_ag) == 2:
            au_ll = l_au[1][2] < l_au[0][2]
            ag_ll = l_ag[1][2] < l_ag[0][2]
            if au_ll != ag_ll:
                smt_list.append(
                    f"🎯 BULLISH SMT DIVERGENCE di {tf_nama}! "
                    f"({'XAUUSD buat Lower Low tapi XAGUSD menolak turun/Higher Low' if au_ll else 'XAGUSD buat Lower Low tapi XAUUSD bertahan/Higher Low'}) "
                    f"-> Tanda Manipulasi Sell-Side & Reversal Naik!"
                )
        if smt_list:
            out.extend(f"  - {s}" for s in smt_list)
        else:
            out.append(f"  - {tf_nama}: Pergerakan XAUUSD & {sym_ag} masih searah/sinkron (tidak ada SMT Divergence).")
    return "\n".join(out) + "\n"


def _info_tipe_pending(tipe):
    mapping = {
        mt5.ORDER_TYPE_BUY_LIMIT: ("BUY LIMIT", "BUY"),
        mt5.ORDER_TYPE_SELL_LIMIT: ("SELL LIMIT", "SELL"),
        mt5.ORDER_TYPE_BUY_STOP: ("BUY STOP", "BUY"),
        mt5.ORDER_TYPE_SELL_STOP: ("SELL STOP", "SELL"),
        getattr(mt5, "ORDER_TYPE_BUY_STOP_LIMIT", 6): ("BUY STOP LIMIT", "BUY"),
        getattr(mt5, "ORDER_TYPE_SELL_STOP_LIMIT", 7): ("SELL STOP LIMIT", "SELL"),
    }
    return mapping.get(tipe, (f"PENDING({tipe})", "BUY"))


def cek_posisi_terbuka_mt5(harga_skrg):
    """Membaca posisi trading yang sedang running (floating) DAN Pending Order (Buy Limit / Sell Limit) di MT5."""
    out = ["\n===== POSISI RUNNING & PENDING ORDER (BUY/SELL LIMIT) DI AKUN MT5 ANDA ====="]
    pos = mt5.positions_get(symbol=SYMBOL)
    orders = mt5.orders_get(symbol=SYMBOL)

    if not pos and not orders:
        out.append("  - Tidak ada posisi running maupun pending order (Buy/Sell Limit) pada simbol ini saat ini.")
        return "\n".join(out) + "\n"

    if pos:
        out.append("  [POSISI RUNNING / FLOATING]:")
        for p in pos:
            arah = "BUY" if p.type == mt5.ORDER_TYPE_BUY else "SELL"
            poin = (harga_skrg - p.price_open) if arah == "BUY" else (p.price_open - harga_skrg)
            out.append(
                f"  - Ticket #{p.ticket} | {arah} {p.volume} lot @ {p.price_open:.2f} | "
                f"SL: {p.sl if p.sl > 0 else 'BELUM DISET'} | TP: {p.tp if p.tp > 0 else 'BELUM DISET'} | "
                f"Floating: ${p.profit:+.2f} ({poin:+.2f} poin)"
            )

    if orders:
        out.append("  [PENDING ORDER TERPASANG (WAJIB DIEVALUASI VALIDITAS ZONANYA OLEH AI)]:")
        for o in orders:
            nama_ord, arah_ord = _info_tipe_pending(o.type)
            vol_ord = getattr(o, "volume_current", getattr(o, "volume_initial", 0.0))
            jarak_ord = abs(harga_skrg - o.price_open) if harga_skrg > 0 else 0.0
            out.append(
                f"  - Ticket #{o.ticket} | {nama_ord} ({arah_ord}) {vol_ord} lot @ {o.price_open:.2f} "
                f"(Jarak dari harga sekarang: {jarak_ord:.2f} poin) | "
                f"SL: {o.sl if o.sl > 0 else 'BELUM DISET'} | TP: {o.tp if o.tp > 0 else 'BELUM DISET'}"
            )
    return "\n".join(out) + "\n"


# Cache berita USD agar saat refresh tidak perlu download ulang terus-menerus
_CACHE_BERITA = {"teks": None, "waktu": 0.0}
MT5_TERHUBUNG = False


def pastikan_mt5_aktif():
    """Menjaga koneksi MT5 tetap terbuka (persistent) selama aplikasi berjalan agar tidak perlu buka-tutup berulang kali."""
    global MT5_TERHUBUNG
    if MT5_TERHUBUNG and mt5.terminal_info() is not None:
        return True
    if mt5.initialize():
        mt5.symbol_select(SYMBOL, True)
        MT5_TERHUBUNG = True
        return True
    MT5_TERHUBUNG = False
    return False


def cek_kalender_berita_usd():
    """Menarik jadwal berita High-Impact USD hari ini dari feed publik ForexFactory untuk filter keamanan news."""
    if _CACHE_BERITA["teks"] and (time.time() - _CACHE_BERITA["waktu"]) < 900:
        return _CACHE_BERITA["teks"]
    out = ["\n===== RADAR BERITA EKONOMI HIGH-IMPACT USD HARI INI ====="]
    url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=4) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        now_utc = datetime.now(timezone.utc)
        hari_wib = (now_utc + timedelta(hours=7)).date()
        berita_penting = []
        for item in data:
            if item.get("country") != "USD" or item.get("impact") not in ("High", "Medium"):
                continue
            dt_str = item.get("date", "")
            try:
                dt_utc = datetime.fromisoformat(dt_str).astimezone(timezone.utc)
            except Exception:
                continue
            dt_wib = dt_utc + timedelta(hours=7)
            if dt_wib.date() == hari_wib:
                selisih_mnt = (dt_utc - now_utc).total_seconds() / 60
                if -120 <= selisih_mnt <= 720:
                    status_wkt = (
                        f"🚨 SEBENTAR LAGI ({int(selisih_mnt)} menit lagi!)"
                        if 0 <= selisih_mnt <= 60
                        else (f"Baru rilis {abs(int(selisih_mnt))} menit lalu" if selisih_mnt < 0 else f"{selisih_mnt / 60:.1f} jam lagi")
                    )
                    berita_penting.append(
                        f"  - [{item.get('impact').upper()}] {dt_wib:%H:%M} WIB: {item.get('title')} "
                        f"(Forecast: {item.get('forecast') or '-'}, Prev: {item.get('previous') or '-'}) -> {status_wkt}"
                    )
        if berita_penting:
            out.extend(berita_penting)
        else:
            out.append("  - Tidak ada jadwal berita USD High/Medium Impact dalam waktu dekat hari ini.")
    except Exception:
        out.append("  - Jadwal kalender otomatis tidak terjangkau saat ini (periksa rilis news USD pada jam 19:30 / 21:00 WIB).")
    hasil_berita = "\n".join(out) + "\n"
    _CACHE_BERITA["teks"] = hasil_berita
    _CACHE_BERITA["waktu"] = time.time()
    return hasil_berita


def kumpulkan_konteks_mt5(is_refresh=False):
    if not is_refresh:
        print("1. Menghubungkan ke MetaTrader5...")
    if not pastikan_mt5_aktif():
        sys.exit(f"Gagal initialize MT5 (pastikan aplikasi MT5 terbuka dan login): {mt5.last_error()}")

    if not mt5.symbol_select(SYMBOL, True):
        sys.exit(f"Simbol {SYMBOL} tidak ditemukan di broker Anda. Cek nama simbol di Market Watch.")
    frames = {nama: ambil_data(tf, n) for nama, (tf, n) in TIMEFRAMES.items()}
    tick = mt5.symbol_info_tick(SYMBOL)
    acc_info = mt5.account_info()
    sym_info = mt5.symbol_info(SYMBOL)
    usd_text = analisa_korelasi_usd()
    smt_text = analisa_smt_divergence(frames)
    harga_awal = tick.bid if tick else frames["M5"]["Close"].iloc[-1]
    posisi_text = cek_posisi_terbuka_mt5(harga_awal)

    for nama, df in frames.items():
        if df is None or len(df) < 60:
            sys.exit(f"Data {nama} tidak cukup/gagal diambil dari MT5.")

    selisih = pd.Timestamp.utcnow().tz_localize(None) - frames["M5"].index[-1]
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
        offset = deteksi_offset(tick)
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

    data_text = (ringkas_tf("W1", frames["W1"], 6)
                 + ringkas_tf("D1", frames["D1"], 10)
                 + ringkas_tf("H4", frames["H4"], 12)
                 + ringkas_tf("H1", frames["H1"], 15)
                 + ringkas_tf("M30", frames["M30"], 20)
                 + ringkas_tf("M5", frames["M5"], 40))

    return f"""
Kamu adalah analis teknikal emas ({SYMBOL}) yang disiplin dan konservatif. Gunakan HANYA data di bawah (waktu = waktu server broker).
Semua candle sudah CLOSE (valid). Swing high/low sudah dihitung dan terkonfirmasi, gunakan itu sebagai dasar zona.
Harga saat ini: {harga_sekarang}

{zona_text}

{smc_text}

{data_text}

METODE TOP-DOWN (ikuti urutan ini):
0. DETEKSI ORDERAN BESAR INSTITUSI & FASE SIDEWAYS-REVERSAL (WAJIB DICEK PERTAMA!):
   - Cek bagian "[Order Flow Institusi & Sideways Reversal]" pada M30 dan M5.
   - Jika tertulis "🚨 PERINGATAN ORDERAN BESAR INSTITUSI (ONE-WAY IMPULSE AKTIF)", maka DILARANG KERAS memberi sinyal lawan arah (Counter-Trend) karena level teknikal kecil pasti jebol terus! Hanya boleh ikuti arah impuls saat pullback ke FVG/OB searah, atau beri rekomendasi WAIT sampai terbentuk kotak Sideways/Ranging.
   - Jika pasar baru selesai impuls lalu membentuk "Kotak Ranging / Sideways Pasca-Impuls", cari setup PEMBALIKAN HARGA (Reversal) berkualitas tinggi: yaitu saat harga melakukan Sweep/Manipulasi di batas atas/bawah kotak sideways lalu masuk kembali disertai CHoCH!
1. ZONA LAMA (W1, D1, H4) & GOLDEN CONFLUENCE ZONES: gunakan daftar "ZONA DARI CANDLE LAMA" dan "GOLDEN CONFLUENCE ZONES" sebagai area potensi PEMBALIKAN atau PANTULAN utama.
   Prioritaskan Golden Confluence Zone (zona bertumpuk multi-metode & multi-TF dengan skor tertinggi) yang terdekat dengan harga sekarang.
2. H4 -> bias besar (struktur BOS/CHoCH, EMA200, Fibonacci premium/discount, korelasi USD) dan pilih zona paling relevan. Sebutkan rentang harga, asal zona, dan waktu terbentuknya.
3. H1 -> perhalus zona; cek apakah harga berada di zona, mendekati zona, atau di tengah range (jika di tengah range, tunggu). Cek OB/Breaker H1, CISD, EQH/EQL, RSI Divergence, dan VWAP.
4. M30 -> konfirmasi di zona: CHoCH/BOS atau CISD (Change in State of Delivery) searah, liquidity sweep, kotak sideways pasca-impuls, OB/Breaker, retest breakout, RSI Divergence, pola candle, EMA/RSI/VWAP.
5. M5 -> ENTRY presisi: trigger CISD (konfirmasi dini ICT saat candle close menembus Open rangkaian candle delivery) atau CHoCH/BOS M5 setelah sweep (atau setelah fase sideways/akumulasi selesai), entry di OB/FVG/breaker/level CISD M5, SL di balik OB atau ujung wick sweep + buffer ATR M5, TP ke likuiditas berikutnya.
6. SESI, BERITA & MAKRO: nilai apakah waktu sekarang berada di Kill Zone, apakah ADR harian sudah Exhausted (bensin habis), apakah ada SMT Divergence XAUUSD vs XAGUSD, dan apakah ada rilis berita High-Impact USD sebentar lagi.

SISTEM 4 PILAR KONFLUENSI (Evaluasi tiap setup wajib berdasarkan 4 pilar ini):

PILAR 1: HTF BIAS & STRUKTUR TREN (W1 / D1 / H4 / H1)
- Tren Struktur Pasar: Apakah BOS/CHoCH H4 & H1 searah dengan sinyal (HH-HL untuk Bullish, LH-LL untuk Bearish)?
- Valuasi & Moving Average: Apakah harga berada di area Discount untuk BUY atau Premium untuk SELL (Fibonacci OTE)? Posisi terhadap EMA200 dan Daily VWAP.
- Dukungan Makro & Korelasi: Korelasi USD / DXY (DXY melemah mendukung Gold naik, DXY menguat menekan Gold) dan RSI Divergence di HTF.
- Syarat Lolos Pilar 1: Sinyal harus searah tren HTF ATAU berupa skenario Pullback/Retest terukur di level struktural kunci.

PILAR 2: INSTITUTIONAL FOOTPRINT & AREA OF VALUE (ZONA UTAMA)
- Keberadaan Zona Institusional: Harga berada di dalam atau sangat dekat dengan Golden Confluence Zone, Unmitigated Order Block (OB), Breaker Block, Fair Value Gap (FVG), atau High Volume Node (POC/VAH/VAL Volume Profile).
- Level Psikologis: Reaksi di sekitar level angka bulat ($XX00 / $XX50 / $XX20).
- Syarat Lolos Pilar 2: Setup HARUS berpijak pada Zona Institusional nyata. DILARANG KERAS entry di tengah range kosong ("no man's land") tanpa zona penahan.

PILAR 3: LTF CONFIRMATION TRIGGER (M5 / M30 EKSEKUSI PRESISI)
- Pemicu Validasi Dini (Trigger Entry):
  a) Liquidity Sweep: Ujung wick menyapu swing high/low sebelumnya atau EQH/EQL, lalu candle close kembali ke dalam range.
  b) CISD (Change in State of Delivery): Candle M5 close menembus level Open rangkaian candle pengiriman sebelumnya (konfirmasi reversal dini ICT).
  c) CHoCH / BOS M5: Terbentuk karakter baru atau pola candle rejection kuat (Pin Bar / Hammer / Engulfing) di batas zona / kotak sideways pasca-impuls.
- Syarat Lolos Pilar 3: Wajib ada minimal satu pemicu LTF (Sweep, CISD, atau CHoCH) yang sudah valid atau siap dieksekusi di batas zona. Jangan menebak pisau jatuh tanpa reaksi harga!

PILAR 4: KONTEKS LINGKUNGAN PASAR & MANAJEMEN RISIKO
- Filter Impulse Institusi: TIDAK MELAWAN Institutional One-Way Impulse aktif. Jika ada impuls liar satu arah, hanya boleh ikut arah impuls saat pullback atau WAIT.
- Volatilitas & Bensin Harian: ADR (Average Daily Range) belum exhausted (masih ada ruang gerak harian), atau setup mengantisipasi reversal di batas ekstrem ADR.
- Waktu & Makro Event: Berada di dalam/menjelang Sesi Aktif (London/NY Kill Zone), tidak ada rilis High-Impact News USD dalam 30 menit ke depan, dan SMT Divergence (XAU vs XAG) mendukung atau netral.
- Rasio Risk-to-Reward: R:R terukur minimal 1:1.5 hingga 1:2+ dengan jarak SL logis di balik batas struktur/wick.

OUTPUT:
A. Status Order Flow Institusi, ADR Harian, Radar Berita USD, Evaluasi Posisi Terbuka & Pending Order (jika user punya posisi floating di MT5, beri saran tegas: HOLD / GESER SL KE BEP / CUT LOSS; jika user sedang memasang Pending Order seperti BUY LIMIT / SELL LIMIT di MT5, WAJIB evaluasi apakah zona limit tersebut MASIH VALID [PERTAHANKAN], PERLU DIGESER [ADJUST LEVEL], atau SUDAH TIDAK VALID / RAWAN JEBOL [SEGERA CANCEL/HAPUS] beserta alasan struktur/likuiditasnya), serta Tabel Ringkasan Zona Utama & Golden Confluence.
B. Daftar sinyal. HANYA tampilkan sinyal berlabel LOW RISK atau MEDIUM RISK. DILARANG menampilkan sinyal HIGH RISK atau sinyal yang melawan Institutional Impulse aktif!

   ══════════════════════════════════════════════════════════════════════
   PROTOKOL STANDAR AKURASI TINGGI SINYAL XAUUSD (WAJIB DIPATUHI):
   ══════════════════════════════════════════════════════════════════════
   1. TIPE ORDER & ANTI-FOMO (Dilarang Kejar Harga Saat Candle Running):
      - Wajib tentukan tipe eksekusi:
        * LIMIT / PULLBACK ENTRY (Rekomendasi Utama Akurasi): Antri di area FVG / OB / 50% Equilibrium Wick Sweep saat harga retrace.
        * CONFIRMED BREAK ENTRY: Masuk HANYA jika candle M5 resmi CLOSE menembus level trigger tertentu. Dilarang menyuruh entry saat candle masih berjalan/floating!
      - Rentang area entry dibatasi ketat maksimal 1.5 - 2.0 poin (contoh: 2650.20 - 2651.80). Jangan berikan rentang lebar yang membingungkan.
   2. ATURAN STOP LOSS ANTI-HUNT & BUFFER SPREAD:
      - DILARANG meletakkan SL mepet di pucuk/lembah wick tanpa buffer (risiko tersapu spread atau secondary wick hunt).
      - Formula SL BUY: Di bawah Swing Low terendah / demand base dikurangi buffer aman (minimal 1.5 - 2.5 poin atau 1x ATR M5).
      - Formula SL SELL: Di atas Swing High tertinggi / supply base ditambah buffer aman (minimal 1.5 - 2.5 poin atau 1x ATR M5) + toleransi spread broker.
      - Sebutkan level struktural perlindungan SL (misal: "SL di 2642.50, di balik ujung wick sweep 2644.20 + buffer 1.7 pt").
   3. TARGET TP OBJEKTIF & PENGAMANAN MODAL DUA TAHAP:
      - TP1 (Likuiditas Internal): Di opposing FVG / Swing M15 terdekat dengan R:R minimal 1:1.5.
        Instruksi WAJIB: "Saat harga menyentuh TP1 -> Amankan 50% profit (Partial Close) & Geser SL ke BEP+ (Titik Impas)!"
      - TP2 (Likuiditas Eksternal): Di Golden Confluence Zone lawan / batas ekstrem ADR / level likuiditas HTF (R:R 1:2.5 s/d 1:4+).
      - FILTER MATEMATIKA R:R: Jika jarak Entry ke TP1 lebih kecil dari jarak ke SL (R:R < 1:1.5), sinyal GUGUR / DILARANG TAMPIL.
   4. KONDISI INVALIDASI EKSPLISIT (GARIS PEMBATALAN SETUP):
      - Wajib sebutkan level pasti pembatalan: "Setup ini BATAL / GUGUR jika candle M5 close menembus [Level X] SEBELUM menjemput area entry."
   5. UKURAN LOT & MANAJEMEN RISIKO MT5 REALISTIS:
      - Ambil langsung rekomendasi ukuran lot dari tabel "MANAJEMEN RISIKO & KALKULATOR LOT OTOMATIS" sesuai jarak SL dan equity akun MT5 (1% untuk Low Risk, 0.5% untuk Medium Risk).

   FORMAT SETIAP SINYAL WAJIB DITULIS RAPI SEPERTI INI:
   ----------------------------------------------------------------------
   [SINYAL <LOW RISK / MEDIUM RISK>] <BUY / SELL> XAUUSD (<Tipe Order: Limit Pullback / Confirmed Break>)
   • Area Entry   : <bawah> - <atas> (Jarak dari harga sekarang: X pt)
   • Stop Loss    : <angka> (Jarak risiko: X pt / X pips | Alasan: X pt di balik wick/level X)
   • Target TP    :
     - TP1        : <angka> (R:R 1:X | Amankan 50% lot & Geser SL ke BEP+)
     - TP2        : <angka> (R:R 1:X | Runner ke likuiditas eksternal)
   • Manajemen Lot: <X.XX> lot (Maksimum risiko: $XX.XX / X% modal)
   • Pemicu (Trigger) : Syarat candle M5 (level CISD / Sweep / CHoCH spesifik)
   • Invalidasi   : Setup BATAL jika candle M5 close melewati <angka level> sebelum terjemput
   • TABEL 4 PILAR KONFLUENSI:
     | Pilar | Status | Bukti Data / Alasan Singkat |
     | Pilar 1: HTF Bias & Struktur        | [✔ / ✘ / ➖] | (Struktur H4, EMA200, Fibonacci, USD) |
     | Pilar 2: Area of Value (Zona Kunci) | [✔ / ✘ / ➖] | (Golden Confluence, OB/FVG, Volume POC) |
     | Pilar 3: LTF Confirmation Trigger   | [✔ / ✘ / ➖] | (Sweep M5, CISD M5, CHoCH, Rejection) |
     | Pilar 4: Konteks Pasar & Risiko     | [✔ / ✘ / ➖] | (Sesi Kill Zone, ADR, Berita USD, R:R) |
     Total Skor: X / 4 Pilar
   ----------------------------------------------------------------------

   Kriteria Klasifikasi Risiko:
   - LOW RISK (Swing / Setup Utama A+):
     * Skor: 4/4 Pilar Terpenuhi (atau 3 Pilar Sangat Kuat + 1 Netral).
     * Berpijak pada Golden Confluence Zone HTF, ada konfirmasi M5 (Sweep + CISD/CHoCH), R:R minimal 1:2.
     * Alokasi risiko lot 1.0% modal.
   - MEDIUM RISK (Intraday / Scalping Reaksi Zona Terdekat B+):
     * Skor: 3/4 Pilar Terpenuhi.
     * Jika harga SAAT INI sedang menyentuh Zona Supply/Demand terdekat (H1/M30/H4) dan di M5 sudah muncul Liquidity Sweep, CISD, CHoCH, atau pola rejection (Pin Bar/Engulfing) — selama TIDAK ada Institutional Impulse ekstrem lawan arah — WAJIB berikan rencana eksekusi MEDIUM RISK di zona terdekat tersebut (jangan hanya menyuruh tunggu di zona atas/bawah yang jauh!). R:R minimal 1:1.5.
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

ATURAN KETAT ZONA RADAR & SINYAL (WAJIB INTRADAY REALISTIS, DILARANG ZONA JAUH EKSTREM):
1. ZONA_BAWAH: Wajib pilih Zona Demand TERDEKAT di bawah harga saat ini (M30/H1/H4/M5) dengan jarak realistis 5 - 20 poin dari harga sekarang! DILARANG KERAS mengambil zona ekstrem ratusan poin yang jauh (seperti zona D1/W1 3890-an saat harga di 4038)! Jika user punya Pending BUY LIMIT (misal 4031), gunakan zona tersebut sebagai ZONA_BAWAH!
2. ZONA_ATAS: Wajib pilih Zona Supply TERDEKAT di atas harga saat ini (M30/H1/H4/M5) dengan jarak realistis 5 - 20 poin dari harga sekarang! DILARANG KERAS mengambil zona ekstrem ratusan poin yang jauh (seperti 4138-an saat harga di 4038)!
3. SINYAL_AKTIF:
   - JIKA harga sedang berada di dekat zona pantau atau ada konfirmasi LTF M5 (CHoCH/CISD/Sweep/Rejection), WAJIB berikan sinyal eksekusi nyata (BUY atau SELL) dengan Entry ketat (1.5-2 pt), SL terukur di balik wick, dan TP yang realistis!
   - DILARANG menulis WAIT jika pasar sedang membentuk pola reaksi/pembalikan jelas (seperti Bullish CISD atau CHoCH M5 di zona demand)!
4. KONSISTENSI ARAH & DIRECTIONAL INTEGRITY (DILARANG PLIN-PLAN / FLIP-FLOP EKSTREM):
   - Jika pada data historis/sesi sebelumnya SUDAH ada Sinyal Aktif (misal BUY) dan sinyal tersebut MASIH VALID (harga belum tembus SL), Anda DILARANG KERAS tiba-tiba membalikkan sinyal menjadi arah berlawanan (misal tiba-tiba memberi SELL)!
   - Jangan membuat trader bingung dengan sinyal yang berubah-ubah saat setup baru saja terjemput atau sedang retest.
   - Sinyal berlawanan (Reversal) HANYA boleh diterbitkan JIKA:
     a) Sinyal lama SUDAH TERBUKTI BATAL/JEBOL (harga close M5 di balik batas SL), ATAU
     b) Target penuh TP2 sudah tersapu bersih dan muncul konfirmasi pembalikan kuat (CHoCH + CISD M5).
   - Selama sinyal lama masih valid dan running, FOKUS Anda adalah mengawal sinyal tersebut sampai tuntas (update trailing SL / BEP / TP), BUKAN malah menerbitkan sinyal lawan yang memicu hedging berisiko tinggi!
Aturan: jangan mengarang angka di luar data, jangan memaksa entry. Ini bukan nasihat finansial.
"""


def ekstrak_target_ai(teks_ai):
    """Membaca otomatis zona pantau, batas jebol (breakout/SL), dan TP dari teks jawaban AI,
    serta memastikan zona radar berada di jarak realistis (intraday) bukan outlier ratusan poin."""
    target = {"atas": None, "bawah": None, "sinyal": None}
    if not teks_ai:
        return target

    m_atas = re.search(r"ZONA_ATAS:\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)(?:.*?batas_jebol:\s*(\d+(?:\.\d+)?))?", teks_ai, re.I)
    if m_atas:
        a, b = float(m_atas.group(1)), float(m_atas.group(2))
        jebol = float(m_atas.group(3)) if m_atas.group(3) else max(a, b) + 3.0
        if a > 0 and b > 0:
            target["atas"] = (min(a, b), max(a, b), jebol)

    m_bawah = re.search(r"ZONA_BAWAH:\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)(?:.*?batas_jebol:\s*(\d+(?:\.\d+)?))?", teks_ai, re.I)
    if m_bawah:
        a, b = float(m_bawah.group(1)), float(m_bawah.group(2))
        jebol = float(m_bawah.group(3)) if m_bawah.group(3) else min(a, b) - 3.0
        if a > 0 and b > 0:
            target["bawah"] = (min(a, b), max(a, b), jebol)

    m_sig = re.search(
        r"SINYAL_AKTIF:\s*(BUY|SELL)\s*\|\s*ENTRY:\s*(\d+(?:\.\d+)?)(?:\s*-\s*(\d+(?:\.\d+)?))?\s*\|\s*SL:\s*(\d+(?:\.\d+)?)\s*\|\s*TP1:\s*(\d+(?:\.\d+)?)(?:\s*\|\s*TP2:\s*(\d+(?:\.\d+)?))?",
        teks_ai, re.I
    )
    if m_sig:
        arah = m_sig.group(1).upper()
        e1 = float(m_sig.group(2))
        e2 = float(m_sig.group(3)) if m_sig.group(3) else e1
        sl = float(m_sig.group(4))
        tp1 = float(m_sig.group(5))
        tp2 = float(m_sig.group(6)) if m_sig.group(6) else tp1
        if e1 > 0 and sl > 0:
            target["sinyal"] = dict(arah=arah, lo=min(e1, e2) - 0.5, hi=max(e1, e2) + 0.5, sl=sl, tp1=tp1, tp2=tp2)

    # 🛡️ FILTER ANTI-HALUSINASI ZONA JAUH:
    # Jika AI memberikan zona yang terlalu jauh (> 35 poin dari harga real-time MT5),
    # cari otomatis zona intraday nyata terdekat dari MT5 agar radar pantau tidak sia-sia!
    if pastikan_mt5_aktif():
        tick_cek = mt5.symbol_info_tick(SYMBOL)
        if tick_cek and tick_cek.bid > 0:
            h_bid = tick_cek.bid
            df_m5_temp = ambil_data(mt5.TIMEFRAME_M5, 30)
            atr_m5_temp = 2.0
            if df_m5_temp is not None and len(df_m5_temp) >= 15:
                atr_m5_temp = float(tambah_indikator(df_m5_temp.copy())["ATR14"].iloc[-1])

            # Cek jika ZONA_BAWAH terlalu jauh (misal harga 4038 tapi zona 3893 / jarak > 35 pt)
            if target["bawah"]:
                jarak_bawah = h_bid - target["bawah"][1]
                if jarak_bawah > 35.0 or target["bawah"][1] > h_bid:
                    u_b, _ = _cari_zona_valid_berikutnya_mt5("BUY", h_bid, h_bid, atr_m5_temp)
                    if u_b:
                        target["bawah"] = (u_b["lo"], u_b["hi"], u_b["jebol"])

            # Cek jika ZONA_ATAS terlalu jauh (misal harga 4038 tapi zona 4138 / jarak > 35 pt)
            if target["atas"]:
                jarak_atas = target["atas"][0] - h_bid
                if jarak_atas > 35.0 or target["atas"][0] < h_bid:
                    u_a, _ = _cari_zona_valid_berikutnya_mt5("SELL", h_bid, h_bid, atr_m5_temp)
                    if u_a:
                        target["atas"] = (u_a["lo"], u_a["hi"], u_a["jebol"])

            # Cek pending order user di MT5: Jika user punya pending order, prioritaskan zona pending order tersebut!
            orders_live_chk = mt5.orders_get(symbol=SYMBOL) or ()
            for o in orders_live_chk:
                p_ord = o.price_open
                if o.type in (mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_BUY_STOP):
                    if target["bawah"] is None or abs(h_bid - target["bawah"][1]) > 25.0:
                        sl_o = o.sl if o.sl > 0 else (p_ord - max(3.0, 1.5 * atr_m5_temp))
                        target["bawah"] = (round(p_ord - 1.5, 2), round(p_ord + 1.5, 2), round(sl_o, 2))
                elif o.type in (mt5.ORDER_TYPE_SELL_LIMIT, mt5.ORDER_TYPE_SELL_STOP):
                    if target["atas"] is None or abs(target["atas"][0] - h_bid) > 25.0:
                        sl_o = o.sl if o.sl > 0 else (p_ord + max(3.0, 1.5 * atr_m5_temp))
                        target["atas"] = (round(p_ord - 1.5, 2), round(p_ord + 1.5, 2), round(sl_o, 2))

    return target


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
    ada_impuls_inst = "PERINGATAN ORDERAN BESAR INSTITUSI" in info_inst
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


# Menyimpan Sinyal Aktif Dinamis & Zona Dinamis yang otomatis berubah saat Mode Pantau mendeteksi peringatan / SL jebol
SINYAL_AKTIF_DINAMIS = None
SINYAL_SEBELUMNYA_DINAMIS = None
ZONA_ATAS_DINAMIS = None
ZONA_BAWAH_DINAMIS = None
CATATAN_JEBOL_PANTAU = []
FILE_STATE_SINYAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state_sinyal_xauusd.json")


def evaluasi_validitas_sinyal(sig, harga_bid):
    """Mengevaluasi secara objektif apakah suatu sinyal masih valid, sudah kena SL, atau sudah kena TP."""
    if not sig or not isinstance(sig, dict) or harga_bid <= 0:
        return "UNKNOWN", "Data sinyal atau harga tidak tersedia"

    arah = sig.get("arah", "BUY").upper()
    sl = float(sig.get("sl", 0.0))
    tp1 = float(sig.get("tp1", 0.0))
    tp2 = float(sig.get("tp2", tp1))
    lo = float(sig.get("lo", 0.0))
    hi = float(sig.get("hi", 0.0))

    if arah == "BUY":
        if harga_bid <= sl:
            return "BATAL / SL JEBOL", f"Harga ({harga_bid:.2f}) telah menembus batas Stop Loss ({sl:.2f}). Setup tidak lagi valid."
        elif harga_bid >= tp2:
            return "SELESAI / TP2 HIT", f"Harga ({harga_bid:.2f}) telah menyapu target penuh TP2 ({tp2:.2f}). Siklus setup selesai."
        elif harga_bid >= tp1:
            return "TP1 TERCAPAI (RUNNING KE TP2)", f"Harga ({harga_bid:.2f}) telah mencapai target TP1 ({tp1:.2f}). SL seharusnya sudah dikunci ke BEP+."
        elif lo <= harga_bid <= hi:
            return "AKTIF DI AREA ENTRY", f"Harga ({harga_bid:.2f}) sedang berada tepat di area entry ({lo:.2f} - {hi:.2f}). Siap eksekusi."
        elif harga_bid > hi:
            return "MASIH VALID (MENUNGGU PULLBACK)", f"Harga ({harga_bid:.2f}) berada di atas entry ({lo:.2f} - {hi:.2f}). Menunggu harga pullback menjemput zona."
        else:
            return "MASIH VALID (DI DEKAT SL)", f"Harga ({harga_bid:.2f}) berada di bawah entry namun masih bertahan di atas batas SL ({sl:.2f})."
    else:
        if harga_bid >= sl:
            return "BATAL / SL JEBOL", f"Harga ({harga_bid:.2f}) telah menembus batas Stop Loss ({sl:.2f}). Setup tidak lagi valid."
        elif harga_bid <= tp2:
            return "SELESAI / TP2 HIT", f"Harga ({harga_bid:.2f}) telah menyapu target penuh TP2 ({tp2:.2f}). Siklus setup selesai."
        elif harga_bid <= tp1:
            return "TP1 TERCAPAI (RUNNING KE TP2)", f"Harga ({harga_bid:.2f}) telah mencapai target TP1 ({tp1:.2f}). SL seharusnya sudah dikunci ke BEP+."
        elif lo <= harga_bid <= hi:
            return "AKTIF DI AREA ENTRY", f"Harga ({harga_bid:.2f}) sedang berada tepat di area entry ({lo:.2f} - {hi:.2f}). Siap eksekusi."
        elif harga_bid < lo:
            return "MASIH VALID (MENUNGGU PULLBACK)", f"Harga ({harga_bid:.2f}) berada di bawah entry ({lo:.2f} - {hi:.2f}). Menunggu harga pullback menjemput zona."
        else:
            return "MASIH VALID (DI DEKAT SL)", f"Harga ({harga_bid:.2f}) berada di atas entry namun masih bertahan di bawah batas SL ({sl:.2f})."


def cek_status_eksekusi_sinyal(sig, pos_list=None, order_list=None):
    """Mengecek apakah suatu sinyal AI diambil (open posisi), dipasang pending order, atau TIDAK DIAMBIL oleh trader di MT5."""
    if not sig or not isinstance(sig, dict) or not sig.get("lo"):
        return "TIDAK ADA SINYAL", "TIDAK ADA", None

    arah_s = sig.get("arah", "").upper()
    if pos_list is None and pastikan_mt5_aktif():
        pos_list = mt5.positions_get(symbol=SYMBOL) or ()
    if order_list is None and pastikan_mt5_aktif():
        order_list = mt5.orders_get(symbol=SYMBOL) or ()

    pos_list = pos_list or ()
    order_list = order_list or ()

    pos_searah = [
        p for p in pos_list
        if (arah_s == "BUY" and p.type == mt5.ORDER_TYPE_BUY)
        or (arah_s == "SELL" and p.type == mt5.ORDER_TYPE_SELL)
    ]
    ord_searah = [
        o for o in order_list
        if (arah_s == "BUY" and "BUY" in _info_tipe_pending(o.type)[1])
        or (arah_s == "SELL" and "SELL" in _info_tipe_pending(o.type)[1])
    ]

    if pos_searah:
        p0 = pos_searah[0]
        ringkasan = f"Ticket #{p0.ticket} ({p0.volume} lot @ {p0.price_open:.2f})"
        return f"SINYAL {arah_s} SEDANG DIAMBIL", "DIAMBIL", ringkasan
    elif ord_searah:
        o0 = ord_searah[0]
        nm0 = _info_tipe_pending(o0.type)[0]
        ringkasan = f"{nm0} #{o0.ticket} @ {o0.price_open:.2f}"
        return f"SINYAL {arah_s} DIPASANG PENDING", "LIMIT", ringkasan
    else:
        return f"SINYAL {arah_s} YANG TIDAK DIAMBIL", "TIDAK DIAMBIL", "Tidak ada posisi terbuka di MT5"


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


def simpan_state_sinyal(sinyal, sinyal_sebelumnya=None, atas=None, bawah=None, catatan=""):
    """Menyimpan status sinyal aktif & sinyal sebelumnya ke file lokal secara permanen agar riwayat tidak pernah hilang."""
    global SINYAL_SEBELUMNYA_DINAMIS

    # Deteksi jika dipanggil secara legacy: simpan_state_sinyal(sinyal, atas, bawah, catatan)
    # Cirinya: argumen ke-2 (sinyal_sebelumnya) adalah tuple/list zona [lo, hi, jebol], bukan dict sinyal.
    if isinstance(sinyal_sebelumnya, (tuple, list)):
        catatan = bawah if isinstance(bawah, str) else catatan
        bawah = atas
        atas = sinyal_sebelumnya
        sinyal_sebelumnya = None

    # Pastikan sinyal_sebelumnya jika ada HARUS berupa dict, bukan list/tuple/string
    if sinyal_sebelumnya is not None and not isinstance(sinyal_sebelumnya, dict):
        sinyal_sebelumnya = None

    # Baca file lama terlebih dahulu agar tidak menimpa data yang sudah ada dengan null
    data_lama = muat_state_sinyal() or {}
    riwayat_lama = data_lama.get("sinyal_sebelumnya")
    if not isinstance(riwayat_lama, dict):
        riwayat_lama = None
    sinyal_aktif_lama = data_lama.get("sinyal")
    if not isinstance(sinyal_aktif_lama, dict):
        sinyal_aktif_lama = None

    # Jika sinyal baru adalah dict valid (setup nyata):
    if sinyal and isinstance(sinyal, dict) and sinyal.get("lo"):
        # Jika sinyal aktif lama juga valid dan angkanya berbeda, jadikan sinyal lama sebagai riwayat
        if sinyal_aktif_lama and (sinyal_aktif_lama.get("lo") != sinyal.get("lo") or sinyal_aktif_lama.get("arah") != sinyal.get("arah")):
            sinyal_sebelumnya = sinyal_aktif_lama
        elif sinyal_sebelumnya is None:
            sinyal_sebelumnya = riwayat_lama or (SINYAL_SEBELUMNYA_DINAMIS if isinstance(SINYAL_SEBELUMNYA_DINAMIS, dict) else None)
    else:
        # Jika sinyal baru adalah None (status WAIT), JANGAN hapus riwayat sinyal sebelumnya!
        if sinyal_sebelumnya is None:
            sinyal_sebelumnya = sinyal_aktif_lama or riwayat_lama or (SINYAL_SEBELUMNYA_DINAMIS if isinstance(SINYAL_SEBELUMNYA_DINAMIS, dict) else None)

    if sinyal_sebelumnya and isinstance(sinyal_sebelumnya, dict):
        SINYAL_SEBELUMNYA_DINAMIS = sinyal_sebelumnya

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
    try:
        with open(FILE_STATE_SINYAL, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


def muat_state_sinyal():
    """Memuat sinyal dan zona pantau dari sesi sebelumnya jika tersedia."""
    if not os.path.exists(FILE_STATE_SINYAL):
        return None
    try:
        with open(FILE_STATE_SINYAL, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception:
        return None


def _cari_zona_valid_berikutnya_mt5(arah_cari, batas_lewat, harga_skrg, atr_m5, is_institusi=False):
    """Memindai M30, H1, H4, dan D1 secara real-time dari MT5 untuk menemukan ZONA VALID SELANJUTNYA:
    - JIKA KARENA INSTITUSI / BERITA: Memberikan ZONA YANG PALING VALID & KUAT (HTF Fortress Zone, abaikan zona minor).
    - JIKA BUKAN KARENA INSTITUSI (NORMAL): Memberikan ZONA TERDEKAT PERSIS SETELAH SL."""
    margin_aman = max(1.0, round(0.5 * atr_m5, 2))
    batas_aman = (min(batas_lewat, harga_skrg) - margin_aman) if arah_cari == "BUY" else (max(batas_lewat, harga_skrg) + margin_aman)

    kandidat_mentah = []
    daftar_tf = [
        ("M30", mt5.TIMEFRAME_M30, 180, 2.0),
        ("H1", mt5.TIMEFRAME_H1, 200, 3.5),
        ("H4", mt5.TIMEFRAME_H4, 180, 5.0),
        ("D1", mt5.TIMEFRAME_D1, 120, 6.5),
    ]

    for nama_tf, kode_tf, jml_bar, bobot_tf in daftar_tf:
        df_tf = ambil_data(kode_tf, jml_bar)
        if df_tf is None or len(df_tf) < 30:
            continue
        evs_tf, _, _ = struktur(df_tf)
        obs_tf = order_blocks(df_tf, evs_tf, maks_event=12)
        for o in obs_tf:
            if o["status"] not in ("fresh", "tested"):
                continue
            lbl_st = "Fresh" if o["status"] == "fresh" else "Tested"
            skor_ob = bobot_tf + (3.0 if o["status"] == "fresh" else 1.2)
            if arah_cari == "BUY" and o["arah"] == "bull" and o["hi"] <= batas_aman:
                kandidat_mentah.append(dict(lo=float(o["lo"]), hi=float(o["hi"]), skor=skor_ob, ket=f"{nama_tf} {lbl_st} Demand OB ({o['asal']})", tf=nama_tf))
            elif arah_cari == "SELL" and o["arah"] == "bear" and o["lo"] >= batas_aman:
                kandidat_mentah.append(dict(lo=float(o["lo"]), hi=float(o["hi"]), skor=skor_ob, ket=f"{nama_tf} {lbl_st} Supply OB ({o['asal']})", tf=nama_tf))

        fvg_tf = cari_fvg(df_tf, harga_skrg, maks=6)
        for tipe_f, b_f, a_f, st_f, _t_f in fvg_tf:
            if "mitigated" in st_f:
                continue
            skor_fvg = bobot_tf + (2.0 if "belum" in st_f else 1.0)
            if arah_cari == "BUY" and "Bullish" in tipe_f and a_f <= batas_aman:
                kandidat_mentah.append(dict(lo=float(b_f), hi=float(a_f), skor=skor_fvg, ket=f"{nama_tf} Bullish FVG ({st_f})", tf=nama_tf))
            elif arah_cari == "SELL" and "Bearish" in tipe_f and b_f >= batas_aman:
                kandidat_mentah.append(dict(lo=float(b_f), hi=float(a_f), skor=skor_fvg, ket=f"{nama_tf} Bearish FVG ({st_f})", tf=nama_tf))

        res_tf, sup_tf = swing_belum_ditembus(df_tf, harga_skrg, n=3 if nama_tf in ("H4", "D1") else 5, maks=5)
        lebar_sw = max(1.6, round(0.8 * atr_m5, 2))
        if arah_cari == "BUY":
            for _t_s, p_s, st_s in sup_tf:
                if p_s <= batas_aman:
                    skor_sw = bobot_tf + (1.6 if st_s == "unmitigated" else 0.6)
                    kandidat_mentah.append(dict(lo=float(p_s - 0.6 * lebar_sw), hi=float(p_s + 0.4 * lebar_sw), skor=skor_sw, ket=f"{nama_tf} Support Swing [{st_s}]", tf=nama_tf))
        else:
            for _t_s, p_s, st_s in res_tf:
                if p_s >= batas_aman:
                    skor_sw = bobot_tf + (1.6 if st_s == "unmitigated" else 0.6)
                    kandidat_mentah.append(dict(lo=float(p_s - 0.4 * lebar_sw), hi=float(p_s + 0.6 * lebar_sw), skor=skor_sw, ket=f"{nama_tf} Resistance Swing [{st_s}]", tf=nama_tf))

    # Urutkan sementara dari yang terdekat dengan batas_aman
    if arah_cari == "BUY":
        kandidat_mentah.sort(key=lambda z: -z["hi"])
    else:
        kandidat_mentah.sort(key=lambda z: z["lo"])

    # Gabungkan zona yang berhimpitan / konfluensi (jarak <= 2.2 poin)
    klaster = []
    for z in kandidat_mentah:
        digabung = False
        for kl in klaster:
            overlap = not (z["hi"] < kl["lo"] - 2.2 or z["lo"] > kl["hi"] + 2.2)
            if overlap:
                kl["lo"] = min(kl["lo"], z["lo"])
                kl["hi"] = max(kl["hi"], z["hi"])
                kl["skor"] += z["skor"] * 0.8
                if z["ket"] not in kl["kets"] and len(kl["kets"]) < 3:
                    kl["kets"].append(z["ket"])
                if z["tf"] not in kl["tfs"]:
                    kl["tfs"].append(z["tf"])
                digabung = True
                break
        if not digabung:
            klaster.append(dict(lo=z["lo"], hi=z["hi"], skor=z["skor"], kets=[z["ket"]], tfs=[z["tf"]]))

    # Jika di area ATH/ATL belum ada klaster historis, gunakan level psikologis bulat
    if not klaster:
        step_psiko = 10.0
        if arah_cari == "BUY":
            lvl_p = (int(batas_aman // step_psiko) * step_psiko)
            if lvl_p >= batas_aman:
                lvl_p -= step_psiko
            klaster.append(dict(lo=lvl_p - 1.0, hi=lvl_p + 1.0, skor=3.0, kets=[f"Level Psikologis Bulat {lvl_p:.0f}"], tfs=["H1"]))
        else:
            lvl_p = (int(batas_aman // step_psiko) + 1) * step_psiko
            if lvl_p <= batas_aman:
                lvl_p += step_psiko
            klaster.append(dict(lo=lvl_p - 1.0, hi=lvl_p + 1.0, skor=3.0, kets=[f"Level Psikologis Bulat {lvl_p:.0f}"], tfs=["H1"]))

    # -------------------------------------------------------------------------
    # PEMILIHAN ZONA BERDASARKAN PENYEBAB JEBOL / SL:
    # 1. KASUS INSTITUSI / IMPULS: Pilih ZONA PALING VALID & KUAT (HTF Major Fortress)
    # 2. KASUS NORMAL / NON-INSTITUSI: Pilih ZONA TERDEKAT PERSIS SETELAH SL
    # -------------------------------------------------------------------------
    def _nilai_urut(kl):
        jarak = (harga_skrg - kl["hi"]) if arah_cari == "BUY" else (kl["lo"] - harga_skrg)
        if is_institusi:
            # Jika dijebol institusi besar: utamakan skor validitas tertinggi & beri penalti zona yang terlalu mepet (< 2.5 pt)
            penalti_mepet = 25.0 if jarak < max(2.5, round(1.2 * atr_m5, 2)) else 0.0
            bonus_htf = 6.0 if any(t in ("H4", "D1") for t in kl.get("tfs", [])) else (3.0 if "H1" in kl.get("tfs", []) else 0.0)
            # Nilai lebih kecil = diprioritaskan
            return - (kl["skor"] * 3.5 + bonus_htf) + penalti_mepet + (jarak * 0.15)
        else:
            # Jika pergerakan normal: utamakan jarak terdekat persis setelah SL
            penalti_terlalu_nempel = 12.0 if jarak < 0.6 else 0.0
            return jarak + penalti_terlalu_nempel - min(kl["skor"] * 0.35, 3.0)

    klaster.sort(key=_nilai_urut)

    hasil_zona = []
    lebar_maks = max(1.8, min(3.2, round(1.4 * atr_m5, 2)))
    buffer_sl = max(1.8, round(1.0 * atr_m5, 2))

    for kl in klaster[:2]:
        lo_z, hi_z = kl["lo"], kl["hi"]
        # Pertajam ketebalan zona agar sesuai Protokol Akurasi Tinggi (1.8 - 3.2 poin)
        if (hi_z - lo_z) > lebar_maks:
            if arah_cari == "BUY":
                mid_z = (lo_z + hi_z) / 2.0
                hi_z = round(min(hi_z, mid_z + 0.5 * lebar_maks), 2)
                lo_z = round(hi_z - lebar_maks, 2)
            else:
                mid_z = (lo_z + hi_z) / 2.0
                lo_z = round(max(lo_z, mid_z - 0.5 * lebar_maks), 2)
                hi_z = round(lo_z + lebar_maks, 2)
        else:
            lo_z, hi_z = round(lo_z, 2), round(hi_z, 2)
            if hi_z - lo_z < 1.2:
                if arah_cari == "BUY":
                    lo_z = round(hi_z - 1.5, 2)
                else:
                    hi_z = round(lo_z + 1.5, 2)

        mid_entry = round((lo_z + hi_z) / 2.0, 2)
        if arah_cari == "BUY":
            sl_z = round(lo_z - buffer_sl, 2)
            risiko_z = max(2.5, mid_entry - sl_z)
            tp1_z = round(hi_z + 1.6 * risiko_z, 2)
            tp2_z = round(hi_z + 3.0 * risiko_z, 2)
        else:
            sl_z = round(hi_z + buffer_sl, 2)
            risiko_z = max(2.5, sl_z - mid_entry)
            tp1_z = round(lo_z - 1.6 * risiko_z, 2)
            tp2_z = round(lo_z - 3.0 * risiko_z, 2)

        tag_konf = f" [KONFLUENSI {len(kl['kets'])} PILAR]" if len(kl["kets"]) >= 2 else ""
        tag_tipe = " [ZONA PALING VALID HTF]" if is_institusi else " [ZONA TERDEKAT PASCA-SL]"
        ket_gabung = " + ".join(kl["kets"]) + tag_konf + tag_tipe
        hasil_zona.append(dict(
            arah=arah_cari,
            lo=lo_z,
            hi=hi_z,
            sl=sl_z,
            jebol=sl_z,
            tp1=tp1_z,
            tp2=tp2_z,
            ket=f"Zona {arah_cari} Selanjutnya: {ket_gabung}",
            sumber=ket_gabung,
            jarak=round(abs(harga_skrg - (hi_z if arah_cari == "BUY" else lo_z)), 2),
            tipe_strategi="PALING VALID (HTF)" if is_institusi else "TERDEKAT PASCA-SL",
        ))

    utama = hasil_zona[0] if hasil_zona else None
    cadangan = hasil_zona[1] if len(hasil_zona) > 1 else None
    return utama, cadangan


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
        print(f"   [❌ STATUS SETUP SEBELUMNYA: GAGAL / SUDAH BATAL KENA SL]")
        print(f"   • Fakta Lapangan     : {alasan_jebol}")
        print(f"   • PERINGATAN KERAS   : DILARANG ENTRY DI SETUP LAMA! Setup tersebut sudah gugur/tidak berlaku.")
        print("   " + "─" * 70)
        print(f"   [⏳ ZONA {lbl_jenis} BERIKUTNYA DISIAGAKAN — WATCHLIST (Menunggu Harga Menjemput)]")
        label_setup = "Setup Sinyal Baru"
    print(f"   • Pemicu Watchlist   : Mencari pijakan zona aman berikutnya pasca-SL jebol")
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
    otomatis["sinyal"] = sinyal_baru
    SINYAL_AKTIF_DINAMIS = sinyal_baru
    simpan_state_sinyal(sinyal_baru, atas=otomatis.get("atas"), bawah=otomatis.get("bawah"), catatan=alasan)
    for k_flag in ("sig_entry", "sig_sl_wick", "sig_sh_reentry", "tp1"):
        sudah_notif_masuk.discard(k_flag)

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
        print("   • Status Sinyal Baru : ⏳ WAIT / NETRAL (TIDAK ADA SINYAL ENTRY SEKARANG)")
        print("   • Status Setup Lama  : ❌ GAGAL / KENA SL (RESMI DIBATALKAN & DIBERSIHKAN)")
        print("   • Tindakan Trader    : JANGAN ENTRY! Tunggu harga menjemput zona baru dengan konfirmasi valid.")
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
        is_gagal_prev = "BATAL" in st_prev or "JEBOL" in st_prev or "SL" in st_prev
        tag_prev = "❌ [SETUP GAGAL / KENA SL]" if is_gagal_prev else f"📌 [{lbl_eks_prev}]"
        print("─" * 70)
        print(f"  📜 RIWAYAT SINYAL SEBELUMNYA — {tag_prev}:")
        print(f"     Setup : {sig_prev_pantau['arah']} @ {sig_prev_pantau['lo']:.2f} - {sig_prev_pantau['hi']:.2f} | SL {sig_prev_pantau['sl']:.2f} | TP1 {sig_prev_pantau['tp1']:.2f} | TP2 {sig_prev_pantau.get('tp2', sig_prev_pantau['tp1']):.2f}")
        print(f"     Status: [{st_prev}] -> {al_prev}")
        if is_gagal_prev:
            print("     ⚠️ CATATAN: Setup di atas SUDAH TIDAK BERLAKU (gugur). Jangan dijadikan acuan entry!")
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
    last_candle_check = 0.0
    df_m5 = None
    close_m5 = 0.0
    atr_m5 = 1.0
    evs, sw, sweeps, cisd_m5, pola_m5, ev_terakhir, calon_cisd_m5 = [], [], [], [], [], "-", "-"
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
            tick = mt5.symbol_info_tick(SYMBOL)
            now_s = time.time()
            # Tarik data candle M5 setiap 1 detik, sedangkan harga Bid di-update LIVE tanpa jeda (tick-by-tick)
            if df_m5 is None or (now_s - last_candle_check) >= 1.0:
                df_baru = ambil_data(mt5.TIMEFRAME_M5, 120)
                if df_baru is not None:
                    df_m5 = df_baru
                    close_m5 = df_m5["Close"].iloc[-1]
                    atr_m5 = tambah_indikator(df_m5.copy())["ATR14"].iloc[-1]
                    evs, _, sw = struktur(df_m5)
                    sweeps = liquidity_sweeps(df_m5, sw, SWING_N, lookback=3)
                    cisd_m5, calon_cisd_m5 = deteksi_cisd(df_m5, sw, lookback=15)
                    pola_m5 = deteksi_pola(df_m5)[-2:]
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
                teks_bar = f"[{datetime.now():%H:%M:%S}] Bid:{harga:.2f} | C_M5:{close_m5:.2f} | M5:{ev_bar}{info_floating_bar}"
                baris_bersih = teks_bar[:lebar_aman].ljust(lebar_aman)
                print(f"\r{baris_bersih}", end="", flush=True)

                # 0A. PENGAWAL VALIDITAS ZONA PENDING ORDER (BUY LIMIT / SELL LIMIT / STOP)
                pending_aktif_skrg = set()
                o_last_c, c_last_c = df_m5["Open"].iloc[-1], df_m5["Close"].iloc[-1]
                t_candle_last_c = df_m5.index[-1]
                bodi_last_c = abs(c_last_c - o_last_c)
                acc_live_mm = mt5.account_info()
                sym_live_mm = mt5.symbol_info(SYMBOL)
                eq_live_mm = float(acc_live_mm.equity if acc_live_mm and acc_live_mm.equity > 0 else (acc_live_mm.balance if acc_live_mm else 1000.0))
                contract_live_mm = float(sym_live_mm.trade_contract_size if sym_live_mm else 100.0)

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
                            winsound.Beep(1350, 180)
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
                                winsound.Beep(520, 300)

                    # (1) Peringatan jika Pending Order belum dipasang SL
                    if o.sl == 0 and f"ord_nosl_{tkt_o}" not in sudah_notif_masuk:
                        sudah_notif_masuk.add(f"ord_nosl_{tkt_o}")
                        jarak_sl_aman = max(3.0, round(1.5 * atr_m5, 2))
                        sl_saran = (p_limit - jarak_sl_aman) if arah_o == "BUY" else (p_limit + jarak_sl_aman)
                        print(f"\n\n[PERINGATAN PENDING TANPA SL #{tkt_o}!] {nm_o} @ {p_limit:.2f} BELUM DIPASANG STOP LOSS!")
                        print(f"   -> SARAN: Segera set SL pada {nm_o} Anda di sekitar {sl_saran:.2f} (~{jarak_sl_aman:.2f} pt) agar jika terjemput tidak terbuka tanpa pengaman.")
                        for _ in range(2):
                            winsound.Beep(850, 250)

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
                                winsound.Beep(620, 320)
                        elif jarak_skrg <= max(2.5, round(1.2 * atr_m5, 2)) and not impuls_menghantam_limit and f"ord_valid_{tkt_o}" not in sudah_notif_masuk:
                            sudah_notif_masuk.add(f"ord_valid_{tkt_o}")
                            print(f"\n\n[✅ SIAGA {nm_o} — ZONA MASIH VALID #{tkt_o}!]")
                            print(f"   Harga mendekati {nm_o} @ {p_limit:.2f} (jarak {jarak_skrg:.2f} pt) dengan ritme pullback/koreksi yang terkendali (Body M5 {bodi_last_c:.2f} pt, tidak ada impuls penghancur).")
                            print(f"   -> STATUS ZONA: MASIH VALID & AMAN. Order siap terjemput, pastikan SL ({o.sl if o.sl > 0 else 'BELUM DISET!'}) sudah terpasang!")
                            for _ in range(2):
                                winsound.Beep(1300, 220)

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
                            winsound.Beep(780, 280)

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
                                winsound.Beep(1500, 200)
                        else:
                            print(f"\n\n[ℹ️ PENDING ORDER DIBATALKAN/DIHAPUS #{tkt_o_lama}] {inf_o['nama']} {inf_o['volume']} lot @ {inf_o['open']:.2f} telah dihapus dari MT5.")
                            winsound.Beep(1050, 200)

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
                            winsound.Beep(1300, 200)

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
                            winsound.Beep(800, 250)

                    # (b) Peringatan Geser SL ke BEP (Impas) saat sudah profit >= 2.5 poin (atau 1.2x ATR M5)
                    batas_bep = max(2.5, round(1.2 * atr_m5, 2))
                    sl_belum_bep = (arah_p == "BUY" and p.sl < p.price_open) or (arah_p == "SELL" and (p.sl == 0 or p.sl > p.price_open))
                    if poin_p >= batas_bep and sl_belum_bep and f"bep_{tkt}" not in sudah_notif_masuk:
                        sudah_notif_masuk.add(f"bep_{tkt}")
                        bep_plus = (p.price_open + 0.3) if arah_p == "BUY" else (p.price_open - 0.3)
                        print(f"\n\n[ALERT AMANKAN MODAL / BEP #{tkt}!] Posisi {arah_p} @ {p.price_open:.2f} sudah running profit +{poin_p:.2f} poin (${p.profit:+.2f})!")
                        print(f"   -> TINDAKAN: Segera geser Stop Loss ke BEP+ ({bep_plus:.2f}) agar transaksi ini 100% BEBAS RISIKO (Risk-Free)!")
                        for _ in range(3):
                            winsound.Beep(1450, 220)

                    # (c) Peringatan Partial TP / Kunci Profit Besar saat profit >= 5.0 poin (atau 2.5x ATR M5)
                    batas_tp_parsial = max(5.0, round(2.5 * atr_m5, 2))
                    if poin_p >= batas_tp_parsial and f"lock_{tkt}" not in sudah_notif_masuk:
                        sudah_notif_masuk.add(f"lock_{tkt}")
                        kunci_sl = (p.price_open + round(poin_p * 0.5, 2)) if arah_p == "BUY" else (p.price_open - round(poin_p * 0.5, 2))
                        print(f"\n\n[ALERT KUNCI PROFIT BESAR #{tkt}!] Posisi {arah_p} @ {p.price_open:.2f} sudah profit +{poin_p:.2f} poin (${p.profit:+.2f})!")
                        print(f"   -> TINDAKAN: Amankan sebagian profit (Partial Close) atau geser Trailing SL ke {kunci_sl:.2f} untuk mengunci profit!")
                        for _ in range(4):
                            winsound.Beep(1600, 200)

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
                            winsound.Beep(750, 300)

                    # (e) Peringatan Kritis Mendekati SL atau Floating Minus Besar Tanpa SL
                    if p.sl > 0:
                        jarak_ke_sl = (tick.bid - p.sl) if arah_p == "BUY" else (p.sl - tick.ask)
                        if 0 < jarak_ke_sl <= 1.2 and f"nearsl_{tkt}" not in sudah_notif_masuk:
                            sudah_notif_masuk.add(f"nearsl_{tkt}")
                            print(f"\n\n[AWAS MENDEKATI SL #{tkt}!] Harga saat ini tinggal {jarak_ke_sl:.2f} poin dari SL Anda ({p.sl:.2f})!")
                            print("   -> DISIPLIN: Jangan menggeser/melebarkan SL agar batas risiko akun real Anda tetap terjaga.")
                            for _ in range(3):
                                winsound.Beep(650, 300)
                    else:
                        batas_minus_darurat = -max(3.5, round(1.8 * atr_m5, 2))
                        if poin_p <= batas_minus_darurat and f"dd_{tkt}" not in sudah_notif_masuk:
                            sudah_notif_masuk.add(f"dd_{tkt}")
                            print(f"\n\n[BAHAYA FLOATING MINUS TANPA SL #{tkt}!] Posisi {arah_p} @ {p.price_open:.2f} sudah minus {poin_p:.2f} poin (${p.profit:+.2f}) tanpa SL!")
                            print("   -> TINDAKAN DARURAT: Segera pasang batas Stop Loss atau lakukan Cut Loss jika zona pertahanan sudah jebol!")
                            for _ in range(5):
                                winsound.Beep(600, 350)

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
                                winsound.Beep(520, 320)

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
                            winsound.Beep(650, 300)

                # Deteksi jika ada posisi yang baru saja ditutup (manual / kena TP / kena SL)
                for tkt_lama in list(tiket_terpantau.keys()):
                    if tkt_lama not in tiket_aktif_skrg:
                        info_lama = tiket_terpantau.pop(tkt_lama)
                        status_akhir = "PROFIT" if info_lama["profit"] >= 0 else "LOSS"
                        print(f"\n\n[POSISI SELESAI / DITUTUP #{tkt_lama}] Posisi {info_lama['arah']} {info_lama['volume']} lot @ {info_lama['open']:.2f} telah tertutup ({status_akhir} terakhir: ${info_lama['profit']:+.2f} / {info_lama['poin']:+.2f} pt).")
                        if status_akhir == "LOSS":
                            lvl_sl_kena = info_lama.get("sl") if info_lama.get("sl", 0) > 0 else harga
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
                            print("      (a) STOP-HUNT / LIQUIDITY SWEEP (harga berbalik naik ke zona -> peluang Re-Entry 2nd Chance!), atau")
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

                            if not zona_masih_utuh:
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
                            winsound.Beep(1100, 200)

                # 0C. EVALUASI REAL-TIME PASCA-SL (DETEKSI STOP-HUNT RE-ENTRY vs VALID BREAKOUT)
                low_m5_now = float(df_m5["Low"].iloc[-1])
                high_m5_now = float(df_m5["High"].iloc[-1])
                for r_sl in riwayat_sl_terkena:
                    if r_sl["selesai_evaluasi"]:
                        continue
                    tkt_r = r_sl["ticket"]
                    arah_r = r_sl["arah"]
                    sl_r = r_sl["sl"]
                    open_r = r_sl["open"]
                    if arah_r == "BUY":
                        r_sl["ekstrem"] = min(r_sl["ekstrem"], harga, low_m5_now)
                        # Cek Skenario 1: Harga berbalik NAIK kembali melewati SL menuju zona entry (Stop-Hunt / Sweep Recovery!)
                        batas_pulih = sl_r + max(1.0, round(0.5 * atr_m5, 2))
                        if harga >= batas_pulih and close_m5 > sl_r:
                            r_sl["selesai_evaluasi"] = True
                            sl_reentry = r_sl["ekstrem"] - max(1.5, round(0.8 * atr_m5, 2))
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
                            print(f"   TETAPI harga tidak lanjut jatuh dan kini BERBALIK NAIK kembali ke {harga:.2f} (Close M5: {close_m5:.2f})!")
                            print(f"   -> DIAGNOSA: Penurunan ke SL tadi adalah LIQUIDITY SWEEP / STOP-HUNT institusi untuk mengambil likuiditas!")
                            print(f"   -> PANDUAN RE-ENTRY AMAN (SECOND CHANCE): Jika candle M5 close bertahan di atas {sl_r:.2f} (atau muncul Bullish CISD/Pin Bar),")
                            print(f"      Anda bisa pertimbangkan Re-Entry BUY dengan SL baru di bawah ujung jarum sweep ({sl_reentry:.2f})!")
                            _update_sinyal_aktif_pantau(otomatis, sudah_notif_masuk, sinyal_re, f"Stop-Hunt Recovery Posisi BUY #{tkt_r} (SL digeser ke bawah wick sweep {r_sl['ekstrem']:.2f})")
                            for _ in range(4):
                                winsound.Beep(1450, 230)
                        # Cek Skenario 2: Candle M5 resmi close jebol jauh di bawah SL (Valid Breakout!)
                        elif close_m5 < sl_r - max(1.8, round(0.9 * atr_m5, 2)) and harga < close_m5:
                            r_sl["selesai_evaluasi"] = True
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
                                winsound.Beep(600, 320)
                    else:
                        r_sl["ekstrem"] = max(r_sl["ekstrem"], harga, high_m5_now)
                        batas_pulih = sl_r - max(1.0, round(0.5 * atr_m5, 2))
                        if harga <= batas_pulih and close_m5 < sl_r:
                            r_sl["selesai_evaluasi"] = True
                            sl_reentry = r_sl["ekstrem"] + max(1.5, round(0.8 * atr_m5, 2))
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
                            print(f"   TETAPI harga tidak lanjut naik dan kini BERBALIK TURUN kembali ke {harga:.2f} (Close M5: {close_m5:.2f})!")
                            print(f"   -> DIAGNOSA: Kenaikan ke SL tadi adalah LIQUIDITY SWEEP / STOP-HUNT institusi untuk mengambil likuiditas!")
                            print(f"   -> PANDUAN RE-ENTRY AMAN (SECOND CHANCE): Jika candle M5 close bertahan di bawah {sl_r:.2f} (atau muncul Bearish CISD/Shooting Star),")
                            print(f"      Anda bisa pertimbangkan Re-Entry SELL dengan SL baru di atas ujung jarum sweep ({sl_reentry:.2f})!")
                            _update_sinyal_aktif_pantau(otomatis, sudah_notif_masuk, sinyal_re, f"Stop-Hunt Recovery Posisi SELL #{tkt_r} (SL digeser ke atas wick sweep {r_sl['ekstrem']:.2f})")
                            for _ in range(4):
                                winsound.Beep(1450, 230)
                        elif close_m5 > sl_r + max(1.8, round(0.9 * atr_m5, 2)) and harga > close_m5:
                            r_sl["selesai_evaluasi"] = True
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
                                winsound.Beep(600, 320)

                # 1. Cek target manual jika user mengetik angka
                if lo_manual is not None:
                    if lo_manual <= harga <= hi_manual and "manual" not in sudah_notif_masuk:
                        sudah_notif_masuk.add("manual")
                        pola_info = ", ".join(pola_m5) if pola_m5 else "belum ada pola pinbar/engulfing"
                        print(f"\n\n[ALERT MASUK ZONA!] Harga {harga:.2f} menyentuh zona ({lo_manual:.2f} - {hi_manual:.2f})!")
                        print(f"   -> Status M5 saat ini: {ev_terakhir} | Pola candle: {pola_info}")
                        print("   -> Pantau terus: menunggu konfirmasi pantulan atau deteksi jika zona jebol...")
                        for _ in range(3):
                            winsound.Beep(1200, 250)

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
                            winsound.Beep(600, 350)
                        return bedah_forensik_jebol(df_m5, f"Zona Manual {lo_manual:.2f}-{hi_manual:.2f} ({arah_jebol})", close_m5)

                # 2. Cek Sinyal Aktif dari AI (Entry, SL/Invalidasi, Stop-Hunt Re-Entry, TP1, TP2)
                if otomatis["sinyal"]:
                    s = otomatis["sinyal"]
                    lbl_ambil_now, kode_ambil_now, ringkas_ambil = cek_status_eksekusi_sinyal(s, pos_live, orders_live)

                    if s["lo"] <= harga <= s["hi"] and "sig_entry" not in sudah_notif_masuk:
                        sudah_notif_masuk.add("sig_entry")
                        kat_entry, ket_re = tentukan_klasifikasi_risiko_sinyal(s)
                        tag_risk_entry = " [⚠️ HIGH RISK]" if kat_entry == "HIGH RISK" else ""
                        if kode_ambil_now == "DIAMBIL":
                            print(f"\n\n[ALERT ENTRY — SINYAL {s['arah']}{tag_risk_entry} SEDANG DIAMBIL!] Harga {harga:.2f} masuk area ENTRY {s['arah']} ({s['lo']:.2f}-{s['hi']:.2f})!")
                            print(f"   -> POSISI ANDA: {ringkas_ambil} sedang aktif mengawal setup ini.")
                        elif kode_ambil_now == "LIMIT":
                            print(f"\n\n[ALERT ENTRY — PENDING ORDER SIAP TERJEMPUT!] Harga {harga:.2f} masuk area ENTRY {s['arah']} ({s['lo']:.2f}-{s['hi']:.2f})!")
                            print(f"   -> PENDING ORDER: {ringkas_ambil} bersiap tereksekusi market.")
                        else:
                            print(f"\n\n[ALERT ENTRY — SINYAL {s['arah']}{tag_risk_entry} YANG TIDAK DIAMBIL!] Harga {harga:.2f} masuk area ENTRY {s['arah']} ({s['lo']:.2f}-{s['hi']:.2f})!")
                            if kat_entry == "HIGH RISK":
                                print(f"   -> ⚠️ TINGKAT RISIKO: HIGH RISK ({ket_re}).")
                                print("   -> PANDUAN TRADER : Sinyal ini dadakan di LTF M5. Anda bebas SKIP (TIDAK DIAMBIL) jika ingin santai, ATAU ambil dengan lot mini (0.25%).")
                            else:
                                print(f"   -> STATUS AKUN: Anda BELUM mengambil sinyal {s['arah']} ini di MT5 (belum ada posisi terbuka).")
                                print(f"   -> TINDAKAN: Jika ingin mengeksekusi manual sekarang, batasi risiko dengan batas SL di {s['sl']:.2f}.")
                        for _ in range(3):
                            winsound.Beep(1200, 250)

                    kena_sl_tick = (s["arah"] == "BUY" and harga <= s["sl"]) or (s["arah"] == "SELL" and harga >= s["sl"])
                    close_jebol_sl = (s["arah"] == "BUY" and close_m5 < s["sl"] - 0.35 * atr_m5) or (
                        s["arah"] == "SELL" and close_m5 > s["sl"] + 0.35 * atr_m5
                    )
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
                                winsound.Beep(720, 250)

                    # Jika tadi sempat menyentuh SL (wick), tapi harga berbalik masuk kembali ke zona Entry (Stop-Hunt Recovery!)
                    if "sig_sl_wick" in sudah_notif_masuk and "sig_sh_reentry" not in sudah_notif_masuk:
                        pulih_ke_entry = (s["arah"] == "BUY" and harga >= s["lo"] and close_m5 > s["sl"]) or (
                            s["arah"] == "SELL" and harga <= s["hi"] and close_m5 < s["sl"]
                        )
                        if pulih_ke_entry:
                            sudah_notif_masuk.add("sig_sh_reentry")
                            eks = s.get("ekstrem_sl", s["sl"])
                            sl_baru_s = (eks - max(1.5, round(0.8 * atr_m5, 2))) if s["arah"] == "BUY" else (eks + max(1.5, round(0.8 * atr_m5, 2)))
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
                            print(f"   Harga sempat menyapu level SL ({s['sl']:.2f}) hingga {eks:.2f}, namun menarik ekor (wick) dan kembali ke area Entry ({harga:.2f})!")
                            print(f"   -> Ini adalah indikasi kuat LIQUIDITY SWEEP institusi! Jika muncul konfirmasi M5 (CISD/Pin Bar), Anda bisa Re-Entry {s['arah']} dengan SL baru di {sl_baru_s:.2f}!")
                            print("   -> KLASIFIKASI: ⚠️ HIGH RISK — Pertimbangkan hanya ambil jika siap risiko atau gunakan lot mini (0.25%). Boleh di-skip!")
                            _update_sinyal_aktif_pantau(otomatis, sudah_notif_masuk, sinyal_sh, f"Stop-Hunt Sweep di {eks:.2f} & Re-Entry 2nd Chance [HIGH RISK]")
                            sudah_notif_masuk.add("sig_sh_reentry")
                            for _ in range(4):
                                winsound.Beep(1450, 230)

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
                            winsound.Beep(600, 350)
                    elif (s["arah"] == "BUY" and harga >= s["tp1"]) or (s["arah"] == "SELL" and harga <= s["tp1"]):
                        if "tp1" not in sudah_notif_masuk:
                            sudah_notif_masuk.add("tp1")
                            mid_entry = round((s["lo"] + s["hi"]) / 2.0, 2)
                            bep_sl = round(mid_entry + 0.3, 2) if s["arah"] == "BUY" else round(mid_entry - 0.3, 2)
                            sinyal_bep = dict(s, sl=bep_sl, ket="TP1 Tercapai — SL Geser ke BEP+ (Runner ke TP2)")
                            if kode_ambil_now == "TIDAK DIAMBIL":
                                print(f"\n\n[TARGET TP1 TERCAPAI — SINYAL {s['arah']} YANG TIDAK DIAMBIL!] Harga {harga:.2f} menyentuh TP1 ({s['tp1']:.2f})!")
                                print(f"   -> CATATAN: Sinyal {s['arah']} ini tadi tidak Anda ambil di akun MT5, namun analisa setup sukses mencapai target TP1.")
                            else:
                                print(f"\n\n[TARGET TERCAPAI — POSISI {s['arah']} ANDA!] Harga {harga:.2f} menyentuh TP1 ({s['tp1']:.2f})! Amankan profit / geser SL ke BEP.")
                            _update_sinyal_aktif_pantau(otomatis, sudah_notif_masuk, sinyal_bep, f"TP1 ({s['tp1']:.2f}) Tercapai -> SL Sinyal Aktif dikunci ke BEP+ ({bep_sl:.2f})")
                            sudah_notif_masuk.add("tp1")
                            for _ in range(4):
                                winsound.Beep(1500, 250)
                            for _ in range(4):
                                winsound.Beep(1500, 250)

                # 3. Cek Zona Supply Atas & Zona Demand Bawah dari AI
                if otomatis["atas"]:
                    lo_a, hi_a, jebol_a = otomatis["atas"]
                    if lo_a <= harga <= hi_a and "atas" not in sudah_notif_masuk:
                        sudah_notif_masuk.add("atas")
                        pola_info = ", ".join(pola_m5) if pola_m5 else "belum ada pola rejection"
                        print(f"\n\n[ALERT ZONA SUPPLY!] Harga {harga:.2f} masuk Zona Supply Atas ({lo_a:.2f} - {hi_a:.2f})!")
                        print(f"   -> Tunggu konfirmasi Sweep / CISD / CHoCH Bearish di M5. Pola M5: {pola_info} | {calon_cisd_m5}")
                        if not otomatis["sinyal"] or otomatis["sinyal"]["arah"] != "SELL":
                            sl_sup = round(max(jebol_a, hi_a + max(1.8, 1.0 * atr_m5)), 2)
                            r_sup = max(2.5, sl_sup - lo_a)
                            sig_sup = dict(arah="SELL", lo=lo_a, hi=hi_a, sl=sl_sup, tp1=round(lo_a - 1.6 * r_sup, 2), tp2=round(lo_a - 3.0 * r_sup, 2), ket="Reaksi Zona Supply Atas (Menunggu Trigger M5)")
                            _update_sinyal_aktif_pantau(otomatis, sudah_notif_masuk, sig_sup, f"Harga Masuk Zona Supply Atas ({lo_a:.2f}-{hi_a:.2f})")
                        for _ in range(3):
                            winsound.Beep(1200, 250)
                    if close_m5 > max(jebol_a, hi_a + 0.5 * atr_m5):
                        laporan_jebol_a = bedah_forensik_jebol(df_m5, f"Zona Supply Atas ({lo_a:.2f}-{hi_a:.2f}) Jebol ke Atas (Breakout Naik)", close_m5)
                        CATATAN_JEBOL_PANTAU.append(laporan_jebol_a)
                        print(f"\n\n[PERINGATAN: ZONA SUPPLY ({lo_a:.2f}-{hi_a:.2f}) JEBOL / BREAKOUT NAIK!] Candle M5 close di {close_m5:.2f} menembus batas SL/Invalidasi {jebol_a:.2f}!")
                        print(f"   -> Forensik Singkat: {laporan_jebol_a.splitlines()[-1]}")
                        print(f"   -> Catatan Flip Zone: Eks-Supply {lo_a:.2f}-{hi_a:.2f} kini berpotensi menjadi Flip Demand jika harga retest dari atas.")
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
                            winsound.Beep(600, 350)

                if otomatis["bawah"]:
                    lo_b, hi_b, jebol_b = otomatis["bawah"]
                    if lo_b <= harga <= hi_b and "bawah" not in sudah_notif_masuk:
                        sudah_notif_masuk.add("bawah")
                        pola_info = ", ".join(pola_m5) if pola_m5 else "belum ada pola rejection"
                        print(f"\n\n[ALERT ZONA DEMAND!] Harga {harga:.2f} masuk Zona Demand Bawah ({lo_b:.2f} - {hi_b:.2f})!")
                        print(f"   -> Tunggu konfirmasi Sweep / CISD / CHoCH Bullish di M5. Pola M5: {pola_info} | {calon_cisd_m5}")
                        if not otomatis["sinyal"] or otomatis["sinyal"]["arah"] != "BUY":
                            sl_dem = round(min(jebol_b, lo_b - max(1.8, 1.0 * atr_m5)), 2)
                            r_dem = max(2.5, hi_b - sl_dem)
                            sig_dem = dict(arah="BUY", lo=lo_b, hi=hi_b, sl=sl_dem, tp1=round(hi_b + 1.6 * r_dem, 2), tp2=round(hi_b + 3.0 * r_dem, 2), ket="Reaksi Zona Demand Bawah (Menunggu Trigger M5)")
                            _update_sinyal_aktif_pantau(otomatis, sudah_notif_masuk, sig_dem, f"Harga Masuk Zona Demand Bawah ({lo_b:.2f}-{hi_b:.2f})")
                        for _ in range(3):
                            winsound.Beep(1200, 250)
                    if close_m5 < min(jebol_b, lo_b - 0.5 * atr_m5):
                        laporan_jebol_b = bedah_forensik_jebol(df_m5, f"Zona Demand Bawah ({lo_b:.2f}-{hi_b:.2f}) Jebol ke Bawah (Breakout Turun)", close_m5)
                        CATATAN_JEBOL_PANTAU.append(laporan_jebol_b)
                        print(f"\n\n[PERINGATAN: ZONA DEMAND ({lo_b:.2f}-{hi_b:.2f}) JEBOL / BREAKOUT TURUN!] Candle M5 close di {close_m5:.2f} menembus batas SL/Invalidasi {jebol_b:.2f}!")
                        print(f"   -> Forensik Singkat: {laporan_jebol_b.splitlines()[-1]}")
                        print(f"   -> Catatan Flip Zone: Eks-Demand {lo_b:.2f}-{hi_b:.2f} kini berpotensi menjadi Flip Supply jika harga retest dari bawah.")
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
                            winsound.Beep(600, 350)

                # 4. Cek CHoCH, CISD, Liquidity Sweep, atau Ledakan Orderan Institusi / Sinyal Reversal Pasca-Sideways di M5
                info_inst = deteksi_institusi_dan_ranging(df_m5, tambah_indikator(df_m5.copy())["ATR14"].values)
                if "PERINGATAN ORDERAN BESAR INSTITUSI" in info_inst and "inst" not in sudah_notif_masuk:
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
                        winsound.Beep(900, 300)
                if "SINYAL PEMBALIKAN" in info_inst and "rev_box" not in sudah_notif_masuk:
                    sudah_notif_masuk.add("rev_box")
                    print(f"\n\n[ALERT PEMBALIKAN PASCA-SIDEWAYS!] {info_inst.split('->')[-1].strip()}")
                    for _ in range(4):
                        winsound.Beep(1350, 300)

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
                                    winsound.Beep(700, 300)
                        # Jika muncul CISD M5:
                        # 🔒 DIRECTIONAL LOCK: CISD berlawanan TIDAK BOLEH mengganti sinyal aktif!
                        # Hanya boleh membuat sinyal baru jika status WAIT (tidak ada sinyal aktif),
                        # atau CISD searah memperkuat sinyal yang sudah ada.
                        if c_last["arah"] == "BEARISH":
                            di_zona_supply = bool(otomatis.get("atas") and (otomatis["atas"][0] - 2.5 <= harga <= otomatis["atas"][1] + 2.5))
                            if not otomatis["sinyal"]:
                                sl_base = max(c_last["ekstrem"], otomatis["atas"][1]) if di_zona_supply else c_last["ekstrem"]
                                sl_c = round(sl_base + max(1.5, 0.8 * atr_m5), 2)
                                r_c = max(2.5, sl_c - harga)
                                tp1_c = round(harga - 1.8 * r_c, 2)
                                tp2_c = round(harga - 3.0 * r_c, 2)
                                risiko_c = "MEDIUM RISK" if di_zona_supply else "HIGH RISK"
                                ket_c = "Terkonfirmasi Bearish CISD di Zona Supply" if di_zona_supply else f"Bearish CISD M5 Dadakan (Open Delivery {c_last['level']:.2f})"
                                sig_c = dict(arah="SELL", lo=round(harga - 0.5, 2), hi=round(max(harga + 1.0, c_last["level"]), 2), sl=sl_c, tp1=tp1_c, tp2=tp2_c, ket=ket_c, risiko=risiko_c)
                                _update_sinyal_aktif_pantau(otomatis, sudah_notif_masuk, sig_c, f"Bearish CISD M5 [{risiko_c}] ({ket_c})")
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
                                sl_base = min(c_last["ekstrem"], otomatis["bawah"][0]) if di_zona_demand else c_last["ekstrem"]
                                sl_c = round(sl_base - max(1.5, 0.8 * atr_m5), 2)
                                r_c = max(2.5, harga - sl_c)
                                tp1_c = round(harga + 1.8 * r_c, 2)
                                tp2_c = round(harga + 3.0 * r_c, 2)
                                risiko_c = "MEDIUM RISK" if di_zona_demand else "HIGH RISK"
                                ket_c = "Terkonfirmasi Bullish CISD di Zona Demand" if di_zona_demand else f"Bullish CISD M5 Dadakan (Open Delivery {c_last['level']:.2f})"
                                sig_c = dict(arah="BUY", lo=round(min(harga - 1.0, c_last["level"]), 2), hi=round(harga + 0.5, 2), sl=sl_c, tp1=tp1_c, tp2=tp2_c, ket=ket_c, risiko=risiko_c)
                                _update_sinyal_aktif_pantau(otomatis, sudah_notif_masuk, sig_c, f"Bullish CISD M5 [{risiko_c}] ({ket_c})")
                            elif otomatis["sinyal"]["arah"] == "BUY":
                                print(f"   -> [💪 KONFIRMASI PENGUATAN SINYAL BUY!] Bullish CISD M5 terbentuk di {c_last['level']:.2f}! Setup BUY yang sedang berjalan semakin valid!")
                            elif otomatis["sinyal"]["arah"] == "SELL":
                                # 🔒 DIRECTIONAL LOCK: JANGAN ganti sinyal SELL ke BUY — hanya PERINGATAN!
                                print(f"   -> [⚠️ PERINGATAN!] Bullish CISD M5 terbentuk di {c_last['level']:.2f} melawan sinyal SELL aktif.")
                                print(f"      🔒 DIRECTIONAL LOCK: Sinyal SELL Anda ({otomatis['sinyal']['lo']:.2f}-{otomatis['sinyal']['hi']:.2f}) TETAP DIPERTAHANKAN.")
                                print(f"      Sinyal SELL hanya batal jika harga close M5 menembus SL ({otomatis['sinyal']['sl']:.2f}).")
                        print("   -> TINDAKAN: Konfirmasi pembalikan dini ICT (CISD) terbentuk! Tekan Ctrl+C lalu ketik 'refresh' jika ingin re-analisa!")
                        for _ in range(2):
                            winsound.Beep(1250, 220)

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
                                    winsound.Beep(700, 300)
                        print("   -> TINDAKAN: Tekan Ctrl+C lalu ketik 'refresh' jika ini konfirmasi searah zona incaran Anda!")
                        for _ in range(2):
                            winsound.Beep(1000, 250)
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
                                    winsound.Beep(700, 300)
                        # Jika terjadi sweep:
                        # 🔒 DIRECTIONAL LOCK: Sweep berlawanan TIDAK BOLEH mengganti sinyal aktif!
                        if "BUY-SIDE" in sw_terakhir[1]:
                            di_zona_supply = bool(otomatis.get("atas") and (otomatis["atas"][0] - 2.5 <= sw_terakhir[3] <= otomatis["atas"][1] + 3.5))
                            if not otomatis["sinyal"]:
                                sl_sw = round(sw_terakhir[3] + max(1.5, 0.8 * atr_m5), 2)
                                r_sw = max(2.5, sl_sw - harga)
                                risiko_sw = "MEDIUM RISK" if di_zona_supply else "HIGH RISK"
                                ket_sw = "Terkonfirmasi Buy-Side Sweep di Zona Supply" if di_zona_supply else f"Buy-Side Sweep M5 Dadakan (Wick {sw_terakhir[3]:.2f})"
                                sig_sw = dict(arah="SELL", lo=round(harga - 0.5, 2), hi=round(max(harga + 1.0, sw_terakhir[2]), 2), sl=sl_sw, tp1=round(harga - 1.8 * r_sw, 2), tp2=round(harga - 3.0 * r_sw, 2), ket=ket_sw, risiko=risiko_sw)
                                _update_sinyal_aktif_pantau(otomatis, sudah_notif_masuk, sig_sw, f"Buy-Side Sweep [{risiko_sw}] ({ket_sw})")
                            elif otomatis["sinyal"]["arah"] == "SELL":
                                print(f"   -> [💪 KONFIRMASI PENGUATAN SINYAL SELL!] Buy-Side Sweep di {sw_terakhir[2]:.2f} menolak kenaikan! Posisi SELL Anda semakin terkonfirmasi!")
                            elif otomatis["sinyal"]["arah"] == "BUY":
                                # 🔒 DIRECTIONAL LOCK: JANGAN ganti sinyal BUY ke SELL
                                print(f"   -> [⚠️ PERINGATAN!] Buy-Side Sweep di {sw_terakhir[2]:.2f} berlawanan arah dengan sinyal BUY aktif.")
                                print(f"      🔒 DIRECTIONAL LOCK: Sinyal BUY ({otomatis['sinyal']['lo']:.2f}-{otomatis['sinyal']['hi']:.2f}) TETAP DIPERTAHANKAN sampai SL ({otomatis['sinyal']['sl']:.2f}) terbukti jebol.")
                        elif "SELL-SIDE" in sw_terakhir[1]:
                            di_zona_demand = bool(otomatis.get("bawah") and (otomatis["bawah"][0] - 3.5 <= sw_terakhir[3] <= otomatis["bawah"][1] + 2.5))
                            if not otomatis["sinyal"]:
                                sl_sw = round(sw_terakhir[3] - max(1.5, 0.8 * atr_m5), 2)
                                r_sw = max(2.5, harga - sl_sw)
                                risiko_sw = "MEDIUM RISK" if di_zona_demand else "HIGH RISK"
                                ket_sw = "Terkonfirmasi Sell-Side Sweep di Zona Demand" if di_zona_demand else f"Sell-Side Sweep M5 Dadakan (Wick {sw_terakhir[3]:.2f})"
                                sig_sw = dict(arah="BUY", lo=round(min(harga - 1.0, sw_terakhir[2]), 2), hi=round(harga + 0.5, 2), sl=sl_sw, tp1=round(harga + 1.8 * r_sw, 2), tp2=round(harga + 3.0 * r_sw, 2), ket=ket_sw, risiko=risiko_sw)
                                _update_sinyal_aktif_pantau(otomatis, sudah_notif_masuk, sig_sw, f"Sell-Side Sweep [{risiko_sw}] ({ket_sw})")
                            elif otomatis["sinyal"]["arah"] == "BUY":
                                print(f"   -> [💪 KONFIRMASI PENGUATAN SINYAL BUY!] Sell-Side Sweep di {sw_terakhir[2]:.2f} menolak penurunan! Posisi BUY Anda semakin terkonfirmasi!")
                            elif otomatis["sinyal"]["arah"] == "SELL":
                                # 🔒 DIRECTIONAL LOCK: JANGAN ganti sinyal SELL ke BUY
                                print(f"   -> [⚠️ PERINGATAN!] Sell-Side Sweep di {sw_terakhir[2]:.2f} berlawanan arah dengan sinyal SELL aktif.")
                                print(f"      🔒 DIRECTIONAL LOCK: Sinyal SELL ({otomatis['sinyal']['lo']:.2f}-{otomatis['sinyal']['hi']:.2f}) TETAP DIPERTAHANKAN sampai SL ({otomatis['sinyal']['sl']:.2f}) terbukti jebol.")
                        print(f"   -> TINDAKAN: Ini konfirmasi sapuan likuiditas untuk {arah_saran}. Tekan Ctrl+C lalu ketik 'refresh' untuk cek entry!")
                        for _ in range(2):
                            winsound.Beep(1400, 300)

            time.sleep(0.1)  # Live update 10x per detik (halus, stabil, dan bebas lag)
    except KeyboardInterrupt:
        print("\nMode pantau dihentikan (koneksi live MT5 tetap aktif).")
    return None


from concurrent.futures import ThreadPoolExecutor, as_completed
from google.genai import types

MODELS_TO_TRY = [
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.6-flash",
    "gemini-3.8-flash",
]

# Menyimpan model yang terbukti aktif dan paling cepat di sesi ini, serta membuang model yang 404 / kuota habis
MODEL_AKTIF_TERCEPAT = None
MODEL_404_BLACKLIST = set()
MODEL_COOLDOWN = {}  # model_name -> timestamp bebas cooldown (untuk 429 / 503)


def _coba_satu_model(client, model_name, contents):
    t0 = time.time()
    res = client.models.generate_content(model=model_name, contents=contents)
    return res.text, model_name, time.time() - t0


def snapshot_live_qa():
    """Mengambil harga Bid/Ask detik ini + update kilat M5 & M30 (0.1 detik) setiap kali user bertanya di Q&A,
    supaya AI selalu tahu kalau harga baru saja menjemput zona atau baru saja membentuk Sweep/CISD/CHoCH di M5!"""
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

    sig_live_txt = ""
    if SINYAL_AKTIF_DINAMIS:
        sd = SINYAL_AKTIF_DINAMIS
        sig_live_txt = (
            f"- Status Sinyal Aktif Dinamis (Terupdate dari Mode Pantau): {sd['arah']} @ {sd['lo']:.2f}-{sd['hi']:.2f} | "
            f"SL {sd['sl']:.2f} | TP1 {sd['tp1']:.2f} | TP2 {sd['tp2']:.2f} ({sd.get('status_pantau', 'Sinyal Awal AI')})\n"
        )

    return (
        f"\n[UPDATE DATA LIVE MT5 DETIK INI SAAT TRADER BERTANYA]:\n"
        f"- Harga Sekarang: Bid {harga:.2f} (Spread {spread:.2f}) | Volatilitas ATR14 M5: {atr5[-1]:.2f} pt\n"
        f"{sig_live_txt}"
        f"- Struktur M5 Terbaru: {ev_txt}\n"
        f"- CISD (Change in State of Delivery) M5 Terbaru: {cisd_txt} | Trigger Calon Saat Ini: {calon_cisd5}\n"
        f"- Liquidity Sweep M5 Terbaru: {sw_txt}\n"
        f"- Pola Candle M5 Terbaru: {', '.join(pola5) if pola5 else 'tidak ada'}\n"
        f"- Status Order Flow & Sideways M5: {inst5}\n"
        f"- {m30_info}"
        f"{pos_txt}"
        f"10 Candle M5 Terakhir yang Sudah Close:\n{tabel5}\n"
    )


def _tangani_error_model(model_name, err):
    pesan = str(err)
    if "404" in pesan or "NOT_FOUND" in pesan or "no longer available" in pesan:
        MODEL_404_BLACKLIST.add(model_name)
    elif "429" in pesan or "RESOURCE_EXHAUSTED" in pesan:
        MODEL_COOLDOWN[model_name] = time.time() + 120  # istirahatkan model yang kena limit 2 menit
    elif "503" in pesan or "UNAVAILABLE" in pesan:
        MODEL_COOLDOWN[model_name] = time.time() + 30


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
                stream = client.models.generate_content_stream(model=model_name, contents=contents)
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
                _tangani_error_model(model_name, e)
                continue

    # 2. Balapan paralel (Fastest-Server Race) ke hingga 3 server sekaligus, ambil yang merespon paling cepat!
    batch_utama = [m for m in kandidat if m not in MODEL_404_BLACKLIST and MODEL_COOLDOWN.get(m, 0) <= time.time()][:3]
    if batch_utama:
        with ThreadPoolExecutor(max_workers=len(batch_utama)) as pool:
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


# =====================================================================
#  MEMORI STATE SINYAL & KELANJUTAN ANALISA (PERSISTENT RESTART MEMORY)
# =====================================================================
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

    is_gagal_start = "BATAL" in status_lama or "JEBOL" in status_lama or "SL" in status_lama
    if is_gagal_start:
        print(f"• STATUS SINYAL LAMA : ❌ [SETUP GAGAL / BATAL KENA SL] (SUDAH TIDAK BERLAKU!)")
        print(f"• Rincian Yang Batal : {sig_sebelumnya['arah']} @ {sig_sebelumnya['lo']:.2f} - {sig_sebelumnya['hi']:.2f} | SL {sig_sebelumnya['sl']:.2f} | TP1 {sig_sebelumnya['tp1']:.2f}")
        print(f"• Status Validitas   : [{status_lama}]")
        print(f"• Keterangan Fakta   : {alasan_lama}")
        print("• PANDUAN TRADER     : ⚠️ JANGAN ENTRY DI AREA INI! Setup ini sudah tewas/gugur.")
    else:
        print(f"• Setup Sinyal Aktif : {sig_sebelumnya['arah']} @ {sig_sebelumnya['lo']:.2f} - {sig_sebelumnya['hi']:.2f}")
        print(f"• Stop Loss & TP     : SL {sig_sebelumnya['sl']:.2f} | TP1 {sig_sebelumnya['tp1']:.2f} | TP2 {sig_sebelumnya.get('tp2', sig_sebelumnya['tp1']):.2f}")
        print(f"• Status Validitas   : [{status_lama}]")
        print(f"• Keterangan Fakta   : {alasan_lama}")
    kat_start, ket_r_start = tentukan_klasifikasi_risiko_sinyal(sig_sebelumnya)
    tag_start = "⚠️ [HIGH RISK]" if kat_start == "HIGH RISK" else ("🛡️ [LOW RISK]" if kat_start == "LOW RISK" else "⚡ [MEDIUM RISK]")
    print(f"• Tingkat Risiko     : {tag_start} ({ket_r_start})")
    if kat_start == "HIGH RISK":
        print("  -> Panduan Trader  : Sinyal dadakan LTF. Anda bebas mempertimbangkan untuk ambil atau skip!")
    print(f"• Status di Akun MT5 : [{lbl_eks_start}] ({info_eks_start})")
    print(f"• Sumber Setup       : {sig_sebelumnya.get('ket', '-')}")
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
        f"- Sinyal Aktif Terakhir: {sig_sebelumnya['arah']} @ {sig_sebelumnya['lo']:.2f} - {sig_sebelumnya['hi']:.2f} | SL {sig_sebelumnya['sl']:.2f} | TP1 {sig_sebelumnya['tp1']:.2f} | TP2 {sig_sebelumnya.get('tp2', sig_sebelumnya['tp1']):.2f} ({sig_sebelumnya.get('ket', '-')})\n"
        f"- Status Akun MT5: [{lbl_eks_start}] ({info_eks_start})\n"
        f"- Fakta Algoritmik Harga Sekarang ({bid_cek:.2f}): [{status_lama}] -> {alasan_lama}\n\n"
        f"INSTRUKSI STRUKTUR JAWABAN (DUA GENERASI SINYAL):\n"
        f"1. BAGIAN WAJIB 1 — STATUS SINYAL SEBELUMNYA:\n"
        f"   Tampilkan kembali setup Sinyal Sebelumnya di atas. Beri label status tegas: [MASIH VALID & AKTIF] atau [SUDAH TIDAK VALID / BATAL] / [TARGET TERCAPAI].\n"
        f"   Jelaskan alasan teknis kenapa sinyal lama masih valid atau sudah tidak valid terhadap aksi harga terkini.\n"
        f"2. BAGIAN WAJIB 2 — REKOMENDASI SINYAL BARU (BOLEH DIBERIKAN):\n"
        f"   Anda DIPERBOLEHKAN memberikan Sinyal Baru jika ada setup baru yang lebih segar, penyesuaian zona entri intraday, atau reaksi di zona terdekat.\n"
        f"   PENTING — KONSISTENSI ARAH SINYAL:\n"
        f"   - Jika sinyal sebelumnya MASIH VALID (belum kena SL/belum dibatalkan), Anda DILARANG KERAS langsung memberi sinyal baru berlawanan arah (misal sinyal lama BUY masih valid, tiba-tiba kasih SELL).\n"
        f"   - Sinyal berlawanan arah HANYA boleh diberikan jika sinyal lama SUDAH TERBUKTI BATAL (harga tembus SL) dan ada konfirmasi struktur M5 berlawanan (CISD/CHoCH/BOS).\n"
        f"   - Jika memberikan sinyal baru SEARAH, jelaskan apakah ini revisi zona entry (lebih dekat/lebih valid) atau setup cadangan.\n"
        f"   Jika sinyal lama masih yang terbaik dan belum perlu diubah, nyatakan bahwa sinyal lama tetap menjadi rekomendasi utama!\n"
    )

prompt = kumpulkan_konteks_mt5() + teks_memori_prompt
print("2. Mengirim data ke Gemini (streaming real-time)...")
client = genai.Client(
    api_key=API_KEY,
    http_options=types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=1)),
)

jawaban_awal, model_aktif = panggil_gemini(client, prompt, stream_output=True)
if not jawaban_awal:
    sys.exit("\nSemua model sedang padat antrean di server Google. Silakan coba lagi beberapa saat.")
_tgt_awal = ekstrak_target_ai(jawaban_awal)
sinyal_baru_ai = _tgt_awal.get("sinyal")

if sinyal_baru_ai:
    # 🔒 DIRECTIONAL INTEGRITY CHECK:
    # Jika sinyal sebelumnya masih valid (belum kena SL / belum selesai), DILARANG membalikkan arah!
    if sig_sebelumnya and isinstance(sig_sebelumnya, dict):
        tick_st = mt5.symbol_info_tick(SYMBOL) if pastikan_mt5_aktif() else None
        bid_st = tick_st.bid if tick_st else 0.0
        st_cek, _ = evaluasi_validitas_sinyal(sig_sebelumnya, bid_st)
        if "BATAL" not in st_cek and "SELESAI" not in st_cek and sinyal_baru_ai["arah"] != sig_sebelumnya["arah"]:
            print(f"\n   [🔒 DIRECTIONAL LOCK: AI mengusulkan Sinyal Lawan {sinyal_baru_ai['arah']}, namun ditolak karena sinyal {sig_sebelumnya['arah']} sebelumnya masih valid ({st_cek})! Sinyal {sig_sebelumnya['arah']} tetap dipertahankan.]")
            sinyal_baru_ai = sig_sebelumnya

    SINYAL_AKTIF_DINAMIS = sinyal_baru_ai
    # Jika sinyal baru berbeda angka dari sinyal sebelumnya, simpan sinyal lama sebagai riwayat
    if sig_sebelumnya and (sinyal_baru_ai["lo"] != sig_sebelumnya["lo"] or sinyal_baru_ai["arah"] != sig_sebelumnya["arah"]):
        SINYAL_SEBELUMNYA_DINAMIS = sig_sebelumnya
        print(f"\n   [ℹ️ UPDATE SINYAL: AI menerbitkan Sinyal Baru ({sinyal_baru_ai['arah']} @ {sinyal_baru_ai['lo']:.2f}-{sinyal_baru_ai['hi']:.2f}). Sinyal sebelumnya tetap dicatat di memori riwayat.]")
else:
    SINYAL_AKTIF_DINAMIS = sig_sebelumnya

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
print("- Ketik 'refresh' atau 'update' -> tarik ulang data terbaru dari MT5.")
print("- Ketik 'simpan'                -> simpan hasil analisa & tanya jawab ke file Jurnal (.md).")
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
            is_gagal_lama = "BATAL" in st_lama or "JEBOL" in st_lama or "SL" in st_lama
            tag_lama = "❌ [SETUP GAGAL / KENA SL]" if is_gagal_lama else f"📌 [{lbl_eks_lama}]"
            print("\n" + "─" * 70)
            print(f"1. {tag_lama} SINYAL SEBELUMNYA (SESI LALU):")
            print(f"   • Setup Sinyal   : {sig_lama['arah']} @ {sig_lama['lo']:.2f} - {sig_lama['hi']:.2f}")
            print(f"   • Stop Loss & TP : SL {sig_lama['sl']:.2f} | TP1 {sig_lama['tp1']:.2f} | TP2 {sig_lama.get('tp2', sig_lama['tp1']):.2f}")
            print(f"   • STATUS VALID   : [{st_lama}]" + (" ❌ SUDAH TIDAK BERLAKU!" if is_gagal_lama else " ✅ MASIH VALID"))
            print(f"   • Keterangan     : {al_lama}")
            if is_gagal_lama:
                print("   • PERINGATAN     : ⚠️ Setup ini sudah gugur. JANGAN jadikan acuan entry!")
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
            mm_s = hitung_mm_sinyal(sig_aktif['lo'], sig_aktif['hi'], sig_aktif['sl'], sig_aktif['tp1'], sig_aktif.get('tp2', 0))
            if mm_s:
                print(f"   • Rasio R:R      : TP1 1:{mm_s['rrr_tp1']:.1f} | TP2 1:{mm_s['rrr_tp2']:.1f} (Jarak SL: {mm_s['jarak_sl']:.2f} pt)")
                if kat_aktif == "HIGH RISK":
                    print(f"   • Lot High Risk  : {mm_s.get('lot_025', mm_s['lot_05']):.2f} lot (Risiko mini 0.25%: -${mm_s.get('risk_025_usd', mm_s['risk_05_usd']):.2f}) [DIANJURKAN JIKA AMBIL]")
                print(f"   • Rekomendasi Lot: 0.5% (Disiplin): {mm_s['lot_05']:.2f} lot (-${mm_s['risk_05_usd']:.2f}) | 1.0% (Standar): {mm_s['lot_10']:.2f} lot (-${mm_s['risk_10_usd']:.2f})")
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

    if cmd in ("simpan", "save"):
        nama_file = f"Jurnal_XAUUSD_{datetime.now():%Y%m%d_%H%M%S}.md"
        with open(nama_file, "w", encoding="utf-8") as f:
            f.write(f"# Jurnal Analisis {SYMBOL} — {datetime.now():%Y-%m-%d %H:%M:%S}\n\n")
            for msg in riwayat_chat[1:]:
                role = "AI Analis" if msg["role"] == "model" else "Trader"
                f.write(f"## {role}\n{msg['parts'][0]['text']}\n\n---\n\n")
        print(f"   Jurnal berhasil disimpan ke: {nama_file}")
        continue

    if cmd in ("refresh", "update"):
        # Jika user ketik refresh manual, cek juga apakah zona dari AI sebelumnya sudah ditembus oleh harga terbaru
        teks_ai_lama = next((m["parts"][0]["text"] for m in reversed(riwayat_chat) if m["role"] == "model"), "")
        target_lama = ekstrak_target_ai(teks_ai_lama)
        prompt_baru = kumpulkan_konteks_mt5(is_refresh=True)

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
            sinyal_kandidat = _tgt_baru.get("sinyal")
            # 🔒 DIRECTIONAL INTEGRITY: Jangan timpa sinyal aktif lama dengan sinyal lawan jika sinyal lama masih valid!
            if sinyal_kandidat and SINYAL_AKTIF_DINAMIS and isinstance(SINYAL_AKTIF_DINAMIS, dict):
                tick_ref = mt5.symbol_info_tick(SYMBOL) if pastikan_mt5_aktif() else None
                bid_ref = tick_ref.bid if tick_ref else 0.0
                st_aktif_ref, _ = evaluasi_validitas_sinyal(SINYAL_AKTIF_DINAMIS, bid_ref)
                if "BATAL" not in st_aktif_ref and "SELESAI" not in st_aktif_ref and sinyal_kandidat["arah"] != SINYAL_AKTIF_DINAMIS["arah"]:
                    print(f"\n   [🔒 DIRECTIONAL LOCK: Sinyal baru AI ({sinyal_kandidat['arah']}) ditahan karena sinyal {SINYAL_AKTIF_DINAMIS['arah']} masih berjalan dan belum tersentuh SL!]")
                    sinyal_kandidat = SINYAL_AKTIF_DINAMIS

            SINYAL_AKTIF_DINAMIS = sinyal_kandidat or SINYAL_AKTIF_DINAMIS
            ZONA_ATAS_DINAMIS = _tgt_baru.get("atas")
            ZONA_BAWAH_DINAMIS = _tgt_baru.get("bawah")
            simpan_state_sinyal(SINYAL_AKTIF_DINAMIS, atas=ZONA_ATAS_DINAMIS, bawah=ZONA_BAWAH_DINAMIS, catatan="Refresh Manual")
            riwayat_chat = [
                {"role": "user", "parts": [{"text": prompt_baru}]},
                {"role": "model", "parts": [{"text": jwb_baru}]},
            ]
        else:
            print("\nGagal memperbarui analisa (server padat). Coba lagi sebentar.")
        continue

    live_snap = snapshot_live_qa()
    instruksi_qa = (
        f"Pertanyaan trader: {tanya}\n"
        f"{live_snap}\n"
        "INSTRUKSI JAWABAN (PROTOKOL AKURASI TINGGI):\n"
        "1. Gunakan [UPDATE DATA LIVE MT5 DETIK INI] di atas sebagai kondisi pasar terbaru!\n"
        "2. Jika memberikan setup/sinyal, WAJIB ikuti Protokol Akurasi Tinggi: tentukan Tipe Order (Limit Pullback vs Confirmed Close), "
        "Entry ketat (rentang maks 1.5-2.0 pt), SL terukur di balik ujung wick + buffer ATR (1.5-2.5 pt), TP1 realistis (R:R min 1:1.5 + instruksi amankan 50% & BEP), "
        "TP2 runner likuiditas, Lot rekomendasi dari saldo akun, dan Level Invalidasi (kapan setup batal).\n"
        "3. Jika harga saat ini sedang menyentuh Zona Supply/Demand terdekat dan di M5 ada reaksi (Sweep/CISD/CHoCH/Rejection), "
        "berikan: (a) Setup Eksekusi Reaksi Zona Terdekat (MEDIUM RISK) dan (b) Skenario Cadangan di zona lebih jauh jika jebol.\n"
        "4. Jawab singkat, tajam, profesional, dan selalu sebutkan angka level/waktu candle spesifik dari data.\n"
        "5. KONSISTENSI SINYAL (SIGNAL LOCK): Jika di data live di atas SUDAH ada Sinyal Aktif Dinamis atau Posisi Terbuka MT5 yang sedang berjalan, "
        "JANGAN membuat sinyal baru yang menimpa atau menggeser angka entri tersebut! Tetap evaluasi dan kawal setup yang sedang berjalan tersebut sampai selesai (kena TP/SL). "
        "Hanya berikan setup/sinyal baru jika sinyal lama sudah selesai atau trader secara eksplisit meminta 'sinyal baru' / 'setup baru'."
    )
    # Jaga ukuran riwayat chat tetap ramping (konteks data utama + maks 6 pesan tanya jawab terakhir) agar respon secepat kilat
    if len(riwayat_chat) > 8:
        riwayat_chat = riwayat_chat[:2] + riwayat_chat[-6:]
    riwayat_chat.append({"role": "user", "parts": [{"text": instruksi_qa}]})

    print("   Memilih server tercepat...", end="\r")
    jawaban_qa, mdl = panggil_gemini(client, riwayat_chat, stream_output=True)
    if jawaban_qa:
        sig_qa = ekstrak_target_ai(jawaban_qa).get("sinyal")
        # 🔒 SIGNAL LOCK: Hanya timpa sinyal jika belum ada sinyal aktif atau trader meminta sinyal baru
        tanya_lower = tanya.lower()
        minta_baru = any(k in tanya_lower for k in ("sinyal baru", "setup baru", "ganti sinyal", "refresh"))
        if sig_qa and (SINYAL_AKTIF_DINAMIS is None or minta_baru):
            # 🔒 DIRECTIONAL INTEGRITY: Jangan timpa sinyal aktif dengan sinyal lawan jika belum selesai/kena SL
            if SINYAL_AKTIF_DINAMIS and isinstance(SINYAL_AKTIF_DINAMIS, dict) and sig_qa["arah"] != SINYAL_AKTIF_DINAMIS["arah"] and not any(k in tanya_lower for k in ("balik arah", "reversal", "ganti sell", "ganti buy", "switch")):
                tick_q = mt5.symbol_info_tick(SYMBOL) if pastikan_mt5_aktif() else None
                bid_q = tick_q.bid if tick_q else 0.0
                st_q, _ = evaluasi_validitas_sinyal(SINYAL_AKTIF_DINAMIS, bid_q)
                if "BATAL" not in st_q and "SELESAI" not in st_q:
                    print(f"\n   [🔒 DIRECTIONAL LOCK: Sinyal aktif tetap {SINYAL_AKTIF_DINAMIS['arah']}. Sinyal lawan ({sig_qa['arah']}) tidak diaktifkan karena setup sebelumnya belum selesai.]")
                    sig_qa = SINYAL_AKTIF_DINAMIS
            SINYAL_AKTIF_DINAMIS = sig_qa
            simpan_state_sinyal(SINYAL_AKTIF_DINAMIS, atas=ZONA_ATAS_DINAMIS, bawah=ZONA_BAWAH_DINAMIS, catatan=f"Update QA: {tanya[:30]}")
        riwayat_chat.append({"role": "model", "parts": [{"text": jawaban_qa}]})
    else:
        riwayat_chat.pop()
        print("   Maaf, server AI sedang sibuk. Silakan kirim ulang pertanyaan Anda.")
