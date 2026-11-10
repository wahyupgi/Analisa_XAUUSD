# -*- coding: utf-8 -*-
"""
modules/mt5_conn.py
Koneksi ke MetaTrader 5, penarikan data candle & tick, kalkulasi indikator dasar, dan audit akun/lot/MM.
"""
import math
import time
from datetime import datetime, timezone, timedelta
import MetaTrader5 as mt5
import numpy as np
import pandas as pd

from modules.config import SYMBOL, TIMEFRAMES, MAKS_RISIKO_TERBUKA_PCT, MAKS_RUGI_HARIAN_PCT
from modules.utils import _catat_error

MT5_TERHUBUNG = False


def _tutup_mt5():
    try:
        mt5.shutdown()
    except Exception:
        pass


def _suffix_simbol():
    """Akhiran simbol broker (mis. 'XAUUSDc' -> 'c') untuk mencari pasangan USD / perak dengan akhiran yang sama."""
    return SYMBOL[6:] if SYMBOL.upper().startswith("XAUUSD") else ""


def ambil_candle_berjalan(tf):
    """Candle yang SEDANG berjalan (posisi 0): untuk range hari ini / ATH minggu ini. None jika gagal."""
    r = mt5.copy_rates_from_pos(SYMBOL, tf, 0, 1)
    if r is None or len(r) == 0:
        return None
    return r[0]


def ambil_data(tf, jumlah, percobaan=3):
    # Posisi 0 = candle berjalan (belum selesai), jadi mulai dari 1
    # History MT5 sering belum tersinkron pada panggilan pertama -> beri beberapa kali percobaan.
    rates = None
    for i in range(max(1, percobaan)):
        rates = mt5.copy_rates_from_pos(SYMBOL, tf, 1, jumlah)
        if rates is not None and len(rates) > 0:
            break
        if i < percobaan - 1:
            time.sleep(0.4)
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


def _ambil_m1_sejak(ts_server):
    """Candle M1 (termasuk yang sedang berjalan) setelah ts_server (epoch waktu server). None bila gagal."""
    if not pastikan_mt5_aktif():
        return None
    tick = mt5.symbol_info_tick(SYMBOL)
    if not tick:
        return None
    menit = min(max(int((tick.time - ts_server) / 60) + 3, 3), 50000)
    rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M1, 0, menit)
    if rates is None or len(rates) == 0:
        return None
    return rates[rates["time"] > ts_server]


def _nilai_per_poin(harga_ref, buy=True, contract=100.0):
    """Nilai (dalam mata uang AKUN) per 1.00 poin harga per 1 lot. Memakai order_calc_profit bila ada
    (sudah memperhitungkan kurs & akun cent); fallback ke contract size."""
    try:
        tipe = mt5.ORDER_TYPE_BUY if buy else mt5.ORDER_TYPE_SELL
        sisi = harga_ref - 1.0 if buy else harga_ref + 1.0
        p = mt5.order_calc_profit(tipe, SYMBOL, 1.0, harga_ref, sisi)
        if p is not None and p != 0:
            return abs(float(p))
    except Exception as e:
        _catat_error("_nilai_per_poin", e)
    return float(contract)


def _hitung_lot(risiko_uang, rugi_per_lot, vol_min, vol_step, vol_max):
    """Lot dibulatkan KE BAWAH ke kelipatan step agar risiko tidak melebihi batas.
    Mengembalikan (lot, risiko_nyata, melebihi_batas); melebihi_batas=True bila lot minimum broker sudah > batas risiko."""
    step = vol_step if vol_step and vol_step > 0 else 0.01
    digit = max(0, -int(math.floor(math.log10(step) + 1e-9)))
    if rugi_per_lot <= 0:
        return vol_min, 0.0, False
    lot = math.floor(risiko_uang / rugi_per_lot / step + 1e-9) * step
    melebihi = False
    if lot < vol_min:
        lot = vol_min
        melebihi = (vol_min * rugi_per_lot) > risiko_uang * 1.0001
    lot = round(min(lot, vol_max), digit)
    return lot, lot * rugi_per_lot, melebihi


def status_risiko_akun():
    """Total risiko posisi+pending terbuka (jarak ke SL x lot, semua simbol) & P/L tertutup HARI INI (hari server).
    Mengembalikan dict atau None bila MT5 tidak tersedia."""
    if not pastikan_mt5_aktif():
        return None
    acc = mt5.account_info()
    if not acc:
        return None
    eq = float(acc.equity if acc.equity > 0 else acc.balance)
    total, tanpa_sl = 0.0, 0
    for p in (mt5.positions_get() or ()):
        if p.sl and p.sl > 0:
            v = mt5.order_calc_profit(p.type, p.symbol, p.volume, p.price_open, p.sl)
            if v is not None:
                total += abs(float(v))
        else:
            tanpa_sl += 1
    for o in (mt5.orders_get() or ()):
        if o.sl and o.sl > 0:
            buy = _info_tipe_pending(o.type)[1] == "BUY"
            vol = getattr(o, "volume_current", getattr(o, "volume_initial", 0.0))
            v = mt5.order_calc_profit(mt5.ORDER_TYPE_BUY if buy else mt5.ORDER_TYPE_SELL, o.symbol, vol, o.price_open, o.sl)
            if v is not None:
                total += abs(float(v))
        else:
            tanpa_sl += 1
    pnl_hari = 0.0
    tick = mt5.symbol_info_tick(SYMBOL)
    if tick:
        awal_hari = int(tick.time) - int(tick.time) % 86400
        deals = mt5.history_deals_get(datetime.now() - timedelta(days=2), datetime.now() + timedelta(days=1)) or ()
        for d in deals:
            if d.time >= awal_hari and d.type in (mt5.DEAL_TYPE_BUY, mt5.DEAL_TYPE_SELL):
                pnl_hari += d.profit + d.commission + d.swap + getattr(d, "fee", 0.0)
    saldo_awal = float(acc.balance) - pnl_hari
    return dict(
        equity=eq, risiko_terbuka=total, risiko_terbuka_pct=(total / eq * 100.0) if eq > 0 else 0.0,
        tanpa_sl=tanpa_sl, pnl_hari=pnl_hari,
        rugi_hari_pct=max(0.0, -pnl_hari / saldo_awal * 100.0) if saldo_awal > 0 else 0.0,
        currency=acc.currency,
    )


def analisa_akun_dan_lot(acc_info, sym_info, atr_m5, harga):
    """Membaca saldo MT5 asli & menghitung lot untuk risiko 0.5% (Medium) dan 1% (Low): dibulatkan ke BAWAH,
    nilai per poin dari order_calc_profit, spread dimasukkan ke risiko; plus status risiko agregat & rugi harian."""
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
    vol_max = sym_info.volume_max or 100.0
    tick = mt5.symbol_info_tick(SYMBOL)
    spread = float(tick.ask - tick.bid) if tick else 0.0
    per_poin = _nilai_per_poin(harga, True, contract)

    out.append(f"Saldo (Balance): {bal:.2f} {cur} | Equity: {eq:.2f} {cur} | Contract Size: {contract:g} | Lot Min: {vol_min} | Nilai 1.00 poin/lot: {per_poin:.2f} {cur} | Spread: {spread:.2f} pt")
    if str(cur).upper() != "USD":
        out.append(f"CATATAN: mata uang akun = {cur}. Semua nilai uang di bawah dalam {cur} (bukan USD).")

    risk_low = eq * 0.01      # 1.0% untuk LOW RISK
    risk_med = eq * 0.005     # 0.5% untuk MEDIUM RISK
    out.append(f"Batas Risiko per Trade -> LOW RISK (1.0%): {risk_low:.2f} {cur} | MEDIUM RISK (0.5%): {risk_med:.2f} {cur}")
    out.append("Tabel Acuan Ukuran Lot berdasarkan jarak Stop Loss (poin/dolar emas; sudah termasuk spread, lot dibulatkan ke bawah):")
    for sl_poin in (3.0, 5.0, 8.0, 10.0, 15.0):
        rugi_per_lot = (sl_poin + spread) * per_poin
        lot_low, _, over_low = _hitung_lot(risk_low, rugi_per_lot, vol_min, vol_step, vol_max)
        lot_med, _, over_med = _hitung_lot(risk_med, rugi_per_lot, vol_min, vol_step, vol_max)
        flag = "  [⚠️ lot minimum broker melebihi batas risiko]" if (over_low or over_med) else ""
        out.append(f"  - Jika jarak SL = {sl_poin:4.1f} poin ({sl_poin * 10:.0f} pips) -> Lot LOW RISK (1%): {lot_low:.2f} lot | Lot MEDIUM RISK (0.5%): {lot_med:.2f} lot{flag}")

    st = status_risiko_akun()
    if st:
        out.append(f"Risiko terbuka (posisi+pending, jarak ke SL): {st['risiko_terbuka']:.2f} {cur} = {st['risiko_terbuka_pct']:.2f}% equity (batas {MAKS_RISIKO_TERBUKA_PCT:g}%)"
                   + (f" | {st['tanpa_sl']} order TANPA SL!" if st["tanpa_sl"] else ""))
        out.append(f"P/L tertutup hari ini: {st['pnl_hari']:+.2f} {cur} | Rugi harian: {st['rugi_hari_pct']:.2f}% (batas {MAKS_RUGI_HARIAN_PCT:g}%)")
        if st["rugi_hari_pct"] >= MAKS_RUGI_HARIAN_PCT:
            out.append("🛑 BATAS RUGI HARIAN TERCAPAI: JANGAN beri sinyal baru hari ini (WAIT).")
        if st["risiko_terbuka_pct"] >= MAKS_RISIKO_TERBUKA_PCT:
            out.append("🛑 RISIKO TERBUKA SUDAH MENCAPAI BATAS: JANGAN beri sinyal baru sebelum posisi/pending dikurangi.")
    return "\n".join(out) + "\n"


def hitung_mm_sinyal(entry_lo, entry_hi, sl, tp1=0.0, tp2=0.0):
    """Money Management presisi dari Equity MT5 & jarak SL sinyal. Lot dibulatkan KE BAWAH; nilai/poin dari
    order_calc_profit; spread dimasukkan ke risiko. Kunci hasil lama dipertahankan; ditambah 'melebihi_batas', 'spread', 'per_poin'."""
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

    tick = mt5.symbol_info_tick(SYMBOL)
    spread = float(tick.ask - tick.bid) if tick else 0.0
    per_poin = _nilai_per_poin(entry_mid, entry_mid > float(sl), contract)
    rugi_per_lot = (jarak_sl + spread) * per_poin

    lot_025, real_risk_025, _ = _hitung_lot(eq * 0.0025, rugi_per_lot, vol_min, vol_step, vol_max)
    lot_05, real_risk_05, over_05 = _hitung_lot(eq * 0.005, rugi_per_lot, vol_min, vol_step, vol_max)
    lot_10, real_risk_10, _ = _hitung_lot(eq * 0.010, rugi_per_lot, vol_min, vol_step, vol_max)
    lot_max, real_risk_max, _ = _hitung_lot(eq * 0.020, rugi_per_lot, vol_min, vol_step, vol_max)

    jarak_tp1 = abs(float(tp1) - entry_mid) if tp1 > 0 else 0
    jarak_tp2 = abs(float(tp2) - entry_mid) if tp2 > 0 else 0
    rrr_tp1 = (jarak_tp1 / jarak_sl) if jarak_sl > 0 else 0
    rrr_tp2 = (jarak_tp2 / jarak_sl) if jarak_sl > 0 else 0

    return {
        "equity": eq,
        "balance": float(acc.balance),
        "currency": cur,
        "jarak_sl": jarak_sl,
        "spread": spread,
        "per_poin": per_poin,
        "melebihi_batas": over_05,
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
    """Membaca identitas akun real, posisi trading yang sedang running (floating), Pending Order, & riwayat deal hari ini di MT5."""
    out = ["\n===== FAKTA AKUN REAL MT5, POSISI RUNNING & PENDING ORDER ANDA (DATA ASLI LIVE) ====="]
    acc = mt5.account_info()
    tick = mt5.symbol_info_tick(SYMBOL)
    cur = acc.currency if acc else "USD"
    eq = float(acc.equity if acc and acc.equity > 0 else (acc.balance if acc else 0.0))
    bid_live = float(tick.bid) if tick and tick.bid > 0 else float(harga_skrg)
    ask_live = float(tick.ask) if tick and tick.ask > 0 else float(harga_skrg)

    if acc:
        mode_akun = "REAL ACCOUNT" if getattr(acc, "trade_mode", 0) == getattr(mt5, "ACCOUNT_TRADE_MODE_REAL", 2) else "DEMO/CONTEST"
        out.append(
            f"  [IDENTITAS AKUN MT5]: Login #{acc.login} ({acc.name} | Server: {acc.server} | Tipe: {mode_akun}) | "
            f"Mata Uang: {cur} (JANGAN sebut USD/$ jika tertulis {cur}!) | "
            f"Balance: {acc.balance:,.2f} {cur} | Equity: {acc.equity:,.2f} {cur} | Floating Total: {acc.profit:+,.2f} {cur} | Free Margin: {acc.margin_free:,.2f} {cur}"
        )

    pos = mt5.positions_get(symbol=SYMBOL)
    orders = mt5.orders_get(symbol=SYMBOL)

    if not pos and not orders:
        out.append("  - POSISI & PENDING ORDER: Tidak ada posisi running maupun pending order (Buy/Sell Limit) pada simbol ini saat ini.")
    else:
        if pos:
            out.append(f"  [POSISI RUNNING / FLOATING SAAT INI — TOTAL {len(pos)} POSISI AKTIF (WAJIB DIKAWAL, JANGAN DIABAIKAN!)]:")
            for p in pos:
                arah = "BUY" if p.type == mt5.ORDER_TYPE_BUY else "SELL"
                harga_tutup = bid_live if arah == "BUY" else ask_live
                poin = (harga_tutup - p.price_open) if arah == "BUY" else (p.price_open - harga_tutup)
                pct_eq = (p.profit / eq * 100.0) if eq > 0 else 0.0
                if p.sl and p.sl > 0:
                    sudah_bep = (arah == "BUY" and p.sl >= p.price_open) or (arah == "SELL" and p.sl <= p.price_open)
                    if sudah_bep:
                        kunci_pt = (p.sl - p.price_open) if arah == "BUY" else (p.price_open - p.sl)
                        sl_info = f"{p.sl:.3f} [✅ SUDAH DIKUNCI BEP+ (+{kunci_pt:.2f} pt di area profit) — POSISI 100% BEBAS RISIKO!]"
                    else:
                        jrk_sl = (harga_tutup - p.sl) if arah == "BUY" else (p.sl - harga_tutup)
                        sl_info = f"{p.sl:.3f} (jarak dari harga sekarang ke SL: {jrk_sl:.2f} pt)"
                else:
                    sl_info = "BELUM DISET (⚠️ RAWAN TANPA PENGAMAN!)"
                tp_info = f"{p.tp:.3f}" if (p.tp and p.tp > 0) else "BELUM DISET (Terbuka / Runner)"
                out.append(
                    f"  - Ticket #{p.ticket} | {arah} {p.volume} lot @ {p.price_open:.3f} (Harga sekarang: {harga_tutup:.3f}) | "
                    f"SL: {sl_info} | TP: {tp_info} | "
                    f"Floating: {p.profit:+,.2f} {cur} ({pct_eq:+.2f}% equity | {poin:+.2f} poin)"
                )
        else:
            out.append("  [POSISI RUNNING / FLOATING SAAT INI]: Tidak ada posisi market yang sedang terbuka (0 posisi running).")

        if orders:
            out.append(f"  [PENDING ORDER TERPASANG SAAT INI — TOTAL {len(orders)} ORDER (WAJIB DIEVALUASI VALIDITAS ZONANYA OLEH AI)]:")
            for o in orders:
                nama_ord, arah_ord = _info_tipe_pending(o.type)
                vol_ord = getattr(o, "volume_current", getattr(o, "volume_initial", 0.0))
                jarak_ord = abs(bid_live - o.price_open) if bid_live > 0 else 0.0
                letak_ord = "di atas harga sekarang" if o.price_open > bid_live else "di bawah harga sekarang"
                sl_ord_txt = f"{o.sl:.3f} (jarak SL {abs(o.price_open - o.sl):.2f} pt)" if (o.sl and o.sl > 0) else "BELUM DISET (⚠️ WAJIB INGATKAN PASANG SL!)"
                tp_ord_txt = f"{o.tp:.3f}" if (o.tp and o.tp > 0) else "BELUM DISET"
                out.append(
                    f"  - Ticket #{o.ticket} | {nama_ord} ({arah_ord}) {vol_ord} lot @ {o.price_open:.3f} "
                    f"(Jarak: {jarak_ord:.2f} poin {letak_ord}) | "
                    f"SL: {sl_ord_txt} | TP: {tp_ord_txt}"
                )
        else:
            out.append("  [PENDING ORDER TERPASANG SAAT INI]: Tidak ada pending order (0 limit/stop order).")

    # Riwayat transaksi yang sudah tertutup hari ini (agar AI tahu histori eksekusi hari ini & tidak mengarang)
    if tick:
        try:
            awal_hari = int(tick.time) - int(tick.time) % 86400
            deals = mt5.history_deals_get(datetime.now() - timedelta(days=2), datetime.now() + timedelta(days=1)) or ()
            deals_hari = [
                d for d in deals
                if d.time >= awal_hari and getattr(d, "symbol", "") == SYMBOL
                and getattr(d, "entry", 0) in (getattr(mt5, "DEAL_ENTRY_OUT", 1), getattr(mt5, "DEAL_ENTRY_INOUT", 2))
            ]
            if deals_hari:
                out.append(f"  [RIWAYAT POSISI YANG SUDAH DITUTUP HARI INI ({len(deals_hari)} transaksi selesai, terbaru di bawah)]:")
                for d in deals_hari[-5:]:
                    arah_d = "SELL (tutup BUY)" if d.type == mt5.DEAL_TYPE_SELL else "BUY (tutup SELL)"
                    t_d = pd.Timestamp(int(d.time), unit="s").strftime("%H:%M")
                    net_d = d.profit + d.commission + d.swap + getattr(d, "fee", 0.0)
                    alasan_d = {
                        getattr(mt5, "DEAL_REASON_SL", 4): "kena SL",
                        getattr(mt5, "DEAL_REASON_TP", 5): "kena TP",
                        getattr(mt5, "DEAL_REASON_SO", 6): "Stop Out",
                    }.get(getattr(d, "reason", -1), "tutup manual/script")
                    out.append(
                        f"  - [{t_d} waktu server] Deal #{d.ticket} (Pos #{getattr(d, 'position_id', '-')}) | "
                        f"{arah_d} {d.volume} lot @ {d.price:.3f} | Hasil: {net_d:+,.2f} {cur} ({alasan_d})"
                    )
        except Exception as e_hist:
            _catat_error("cek_posisi_terbuka_mt5_hist", e_hist)

    return "\n".join(out) + "\n"


def pastikan_mt5_aktif():
    """Menjaga koneksi MT5 tetap terbuka (persistent) selama aplikasi berjalan agar tidak perlu buka-tutup berulang kali."""
    global MT5_TERHUBUNG
    if MT5_TERHUBUNG and mt5.terminal_info() is not None:
        return True
    try:
        if mt5.initialize(timeout=3000):
            mt5.symbol_select(SYMBOL, True)
            MT5_TERHUBUNG = True
            return True
    except Exception:
        pass
    MT5_TERHUBUNG = False
    return False
