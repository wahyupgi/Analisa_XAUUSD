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
    # Waktu MT5 = waktu server/epoch MT5 (UTC); tandai attrs['time_tz'] = 'UTC' agar konversi ke WIB tepat satu kali
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.set_index("time")
    out_df = df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                                "close": "Close", "tick_volume": "Volume"})
    out_df.attrs["time_tz"] = "UTC"
    return out_df


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
    """Membaca saldo MT5 asli & menghitung lot untuk alokasi risiko modal 0.5% (konservatif) dan 1.0% (standar / risiko lebih tinggi).
    Lot dibulatkan ke BAWAH, nilai per poin dari order_calc_profit, spread dimasukkan ke risiko."""
    out = ["\n===== MANAJEMEN RISIKO & KALKULATOR LOT OTOMATIS (AKUN MT5) ====="]
    if acc_info is None or sym_info is None or getattr(sym_info, 'trade_contract_size', 0) <= 0:
        out.append("DATA UNAVAILABLE: Info akun atau spesifikasi kontrak simbol MT5 tidak lengkap/tersedia.")
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

    risk_konservatif = eq * 0.005  # 0.5% (risiko lebih konservatif)
    risk_standar = eq * 0.01       # 1.0% (risiko lebih tinggi dibandingkan 0.5%)
    out.append(f"Batas Risiko Modal per Trade:")
    out.append(f"  • 0,5% (risiko lebih konservatif)             : {risk_konservatif:.2f} {cur}")
    out.append(f"  • 1,0% (risiko lebih tinggi dibandingkan 0,5%): {risk_standar:.2f} {cur}")
    out.append("Tabel Acuan Ukuran Lot berdasarkan jarak Stop Loss (poin emas; sudah termasuk spread, lot dibulatkan ke bawah):")
    for sl_poin in (3.0, 5.0, 8.0, 10.0, 15.0):
        rugi_per_lot = (sl_poin + spread) * per_poin
        lot_kons, _, over_kons = _hitung_lot(risk_konservatif, rugi_per_lot, vol_min, vol_step, vol_max)
        lot_std, _, over_std = _hitung_lot(risk_standar, rugi_per_lot, vol_min, vol_step, vol_max)
        flag = "  [⚠️ lot minimum broker melebihi batas risiko]" if (over_kons or over_std) else ""
        out.append(f"  - Jarak SL = {sl_poin:4.1f} pt ({sl_poin * 10:.0f} pips) -> Lot 0,5% (konservatif): {lot_kons:.2f} lot | Lot 1,0% (standar): {lot_std:.2f} lot{flag}")

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
    if isinstance(tipe, str):
        up = tipe.strip().upper().replace("_", " ")
        for label, arah in (
            ("BUY STOP LIMIT", "BUY"),
            ("SELL STOP LIMIT", "SELL"),
            ("BUY LIMIT", "BUY"),
            ("SELL LIMIT", "SELL"),
            ("BUY STOP", "BUY"),
            ("SELL STOP", "SELL"),
        ):
            if label in up:
                return label, arah
        return up, ("SELL" if "SELL" in up else "BUY")
    mapping = {
        mt5.ORDER_TYPE_BUY_LIMIT: ("BUY LIMIT", "BUY"),
        mt5.ORDER_TYPE_SELL_LIMIT: ("SELL LIMIT", "SELL"),
        mt5.ORDER_TYPE_BUY_STOP: ("BUY STOP", "BUY"),
        mt5.ORDER_TYPE_SELL_STOP: ("SELL STOP", "SELL"),
        getattr(mt5, "ORDER_TYPE_BUY_STOP_LIMIT", 6): ("BUY STOP LIMIT", "BUY"),
        getattr(mt5, "ORDER_TYPE_SELL_STOP_LIMIT", 7): ("SELL STOP LIMIT", "SELL"),
    }
    return mapping.get(tipe, (f"PENDING({tipe})", "BUY"))


# Toleransi terukur rekonsiliasi entry & SL/TP terhadap presisi tick_size broker dan pembulatan 2 desimal scanner (0.01 pt)
PENDING_ENTRY_TOLERANCE_MIN_PT = 0.05
PENDING_LEVEL_TOLERANCE_MIN_PT = 0.05


def _acceptable_trade_symbols(target_symbol=None):
    """Menentukan simbol target dan pemetaan broker-symbol terverifikasi (mis. XAUUSD <-> XAUUSDc)."""
    sym = str(target_symbol or SYMBOL or "").strip()
    cfg = str(SYMBOL or "").strip()
    accepted = set()
    if sym:
        accepted.add(sym.upper())
    if cfg:
        if not sym or sym.upper() == cfg.upper():
            accepted.add(cfg.upper())
        elif sym.upper() == "XAUUSD" and cfg.upper().startswith("XAUUSD"):
            accepted.add(cfg.upper())
    return sym or cfg, accepted


def _normalize_mt5_pending_order(raw_order, default_symbol=SYMBOL, retrieved_at=""):
    """Menormalisasi objek/dict order MT5 menjadi struktur read-only yang seragam."""
    if isinstance(raw_order, dict):
        ticket = raw_order.get("ticket")
        sym = str(raw_order.get("symbol") or default_symbol or SYMBOL).strip()
        raw_type = raw_order.get("order_type") if raw_order.get("order_type") is not None else raw_order.get("type", 2)
        entry_val = raw_order.get("price_open", raw_order.get("entry_price", raw_order.get("entry", 0.0)))
        sl_val = raw_order.get("sl", 0.0)
        tp_val = raw_order.get("tp", 0.0)
        vol_val = raw_order.get("volume_current", raw_order.get("volume_initial", raw_order.get("volume", 0.0)))
        setup_ts = raw_order.get("time_setup", 0)
    else:
        ticket = getattr(raw_order, "ticket", None)
        sym = str(getattr(raw_order, "symbol", None) or default_symbol or SYMBOL).strip()
        raw_type = getattr(raw_order, "type", 2)
        entry_val = getattr(raw_order, "price_open", getattr(raw_order, "entry_price", 0.0))
        sl_val = getattr(raw_order, "sl", 0.0)
        tp_val = getattr(raw_order, "tp", 0.0)
        vol_val = getattr(raw_order, "volume_current", getattr(raw_order, "volume_initial", getattr(raw_order, "volume", 0.0)))
        setup_ts = getattr(raw_order, "time_setup", 0)

    order_type_str, direction_str = _info_tipe_pending(raw_type)
    try:
        entry_f = float(entry_val or 0.0)
    except (TypeError, ValueError):
        entry_f = 0.0
    try:
        sl_f = float(sl_val or 0.0)
    except (TypeError, ValueError):
        sl_f = 0.0
    try:
        tp_f = float(tp_val or 0.0)
    except (TypeError, ValueError):
        tp_f = 0.0
    try:
        vol_f = float(vol_val or 0.0)
    except (TypeError, ValueError):
        vol_f = 0.0
    try:
        tkt_val = int(ticket) if ticket is not None else None
    except (TypeError, ValueError):
        tkt_val = ticket

    return {
        "ticket": tkt_val,
        "order_type": order_type_str,
        "direction": direction_str,
        "symbol": sym,
        "entry_price": entry_f if math.isfinite(entry_f) else 0.0,
        "sl": sl_f if (math.isfinite(sl_f) and sl_f > 0) else 0.0,
        "tp": tp_f if (math.isfinite(tp_f) and tp_f > 0) else 0.0,
        "volume": vol_f if (math.isfinite(vol_f) and vol_f > 0) else 0.0,
        "time_setup": int(setup_ts or 0) if isinstance(setup_ts, (int, float)) else 0,
        "retrieved_at": retrieved_at,
    }


def fetch_mt5_pending_orders_snapshot(symbol=SYMBOL, orders_override=None, allow_live_query=True):
    """
    Mengambil snapshot pending order terkini dari MT5 secara READ-ONLY.
    Tidak pernah memanggil order_send() atau memodifikasi/membatalkan order.
    Jika query MT5 gagal atau tidak tersedia, mengembalikan query_ok=False dan status='UNKNOWN'.
    """
    from modules.utils import format_wib_timestamp

    target_sym, accepted_syms = _acceptable_trade_symbols(symbol)
    now_wib = format_wib_timestamp(default_now=True)
    now_ts = time.time()

    if isinstance(orders_override, dict) and "query_ok" in orders_override and "orders" in orders_override:
        return dict(orders_override)

    digits = 2
    point = 0.01
    tick_size = 0.01

    if orders_override is not None and isinstance(orders_override, (list, tuple)):
        try:
            if pastikan_mt5_aktif():
                sym_info = mt5.symbol_info(target_sym) or mt5.symbol_info(SYMBOL)
                if sym_info is not None:
                    digits = int(getattr(sym_info, "digits", 2) or 2)
                    point = float(getattr(sym_info, "point", 0.01) or 0.01)
                    tick_size = float(getattr(sym_info, "trade_tick_size", point) or point)
        except Exception:
            pass
        entry_tol = round(max(10.0 * tick_size, 2.0 * point, PENDING_ENTRY_TOLERANCE_MIN_PT), max(digits, 3))
        level_tol = round(max(10.0 * tick_size, 2.0 * point, PENDING_LEVEL_TOLERANCE_MIN_PT), max(digits, 3))
        norm_orders = []
        for raw_o in orders_override:
            no = _normalize_mt5_pending_order(raw_o, default_symbol=target_sym, retrieved_at=now_wib)
            if no["symbol"].upper() in accepted_syms:
                norm_orders.append(no)
        return {
            "query_ok": True,
            "status": "OK",
            "reason": "",
            "symbol": target_sym,
            "digits": digits,
            "point": point,
            "tick_size": tick_size,
            "entry_tolerance": entry_tol,
            "level_tolerance": level_tol,
            "retrieved_at": now_wib,
            "retrieved_ts": now_ts,
            "total_account_orders": len(orders_override),
            "symbol_orders_count": len(norm_orders),
            "orders": norm_orders,
        }

    if not allow_live_query:
        return {
            "query_ok": False,
            "status": "UNKNOWN",
            "reason": "MT5_QUERY_NOT_EXECUTED",
            "symbol": target_sym,
            "digits": digits,
            "point": point,
            "tick_size": tick_size,
            "entry_tolerance": PENDING_ENTRY_TOLERANCE_MIN_PT,
            "level_tolerance": PENDING_LEVEL_TOLERANCE_MIN_PT,
            "retrieved_at": now_wib,
            "retrieved_ts": now_ts,
            "total_account_orders": 0,
            "symbol_orders_count": 0,
            "orders": [],
        }

    try:
        if not pastikan_mt5_aktif():
            return {
                "query_ok": False,
                "status": "UNKNOWN",
                "reason": "MT5_NOT_CONNECTED",
                "symbol": target_sym,
                "digits": digits,
                "point": point,
                "tick_size": tick_size,
                "entry_tolerance": PENDING_ENTRY_TOLERANCE_MIN_PT,
                "level_tolerance": PENDING_LEVEL_TOLERANCE_MIN_PT,
                "retrieved_at": now_wib,
                "retrieved_ts": now_ts,
                "total_account_orders": 0,
                "symbol_orders_count": 0,
                "orders": [],
            }
        sym_info = mt5.symbol_info(target_sym) or mt5.symbol_info(SYMBOL)
        if sym_info is not None:
            digits = int(getattr(sym_info, "digits", 2) or 2)
            point = float(getattr(sym_info, "point", 0.01) or 0.01)
            tick_size = float(getattr(sym_info, "trade_tick_size", point) or point)

        raw_orders = mt5.orders_get(symbol=target_sym)
        if raw_orders is None and target_sym != SYMBOL:
            raw_orders = mt5.orders_get(symbol=SYMBOL)
        if raw_orders is None:
            raw_orders = mt5.orders_get()
        if raw_orders is None:
            return {
                "query_ok": False,
                "status": "UNKNOWN",
                "reason": "MT5_ORDERS_GET_FAILED",
                "symbol": target_sym,
                "digits": digits,
                "point": point,
                "tick_size": tick_size,
                "entry_tolerance": PENDING_ENTRY_TOLERANCE_MIN_PT,
                "level_tolerance": PENDING_LEVEL_TOLERANCE_MIN_PT,
                "retrieved_at": now_wib,
                "retrieved_ts": now_ts,
                "total_account_orders": 0,
                "symbol_orders_count": 0,
                "orders": [],
            }

        entry_tol = round(max(10.0 * tick_size, 2.0 * point, PENDING_ENTRY_TOLERANCE_MIN_PT), max(digits, 3))
        level_tol = round(max(10.0 * tick_size, 2.0 * point, PENDING_LEVEL_TOLERANCE_MIN_PT), max(digits, 3))
        norm_orders = []
        for raw_o in raw_orders:
            no = _normalize_mt5_pending_order(raw_o, default_symbol=target_sym, retrieved_at=now_wib)
            if no["symbol"].upper() in accepted_syms:
                norm_orders.append(no)
        return {
            "query_ok": True,
            "status": "OK",
            "reason": "",
            "symbol": target_sym,
            "digits": digits,
            "point": point,
            "tick_size": tick_size,
            "entry_tolerance": entry_tol,
            "level_tolerance": level_tol,
            "retrieved_at": now_wib,
            "retrieved_ts": now_ts,
            "total_account_orders": len(raw_orders),
            "symbol_orders_count": len(norm_orders),
            "orders": norm_orders,
        }
    except Exception as exc:
        _catat_error("fetch_mt5_pending_orders_snapshot", exc)
        return {
            "query_ok": False,
            "status": "UNKNOWN",
            "reason": f"MT5_QUERY_EXCEPTION:{type(exc).__name__}",
            "symbol": target_sym,
            "digits": digits,
            "point": point,
            "tick_size": tick_size,
            "entry_tolerance": PENDING_ENTRY_TOLERANCE_MIN_PT,
            "level_tolerance": PENDING_LEVEL_TOLERANCE_MIN_PT,
            "retrieved_at": now_wib,
            "retrieved_ts": now_ts,
            "total_account_orders": 0,
            "symbol_orders_count": 0,
            "orders": [],
        }


def format_pending_compact_status(plan_status, recon, order_type="", entry_price=0.0):
    """
    Memformat status ringkas pending plan yang memisahkan plan_status (READY/WAIT/INVALID/EXPIRED)
    dari broker_order_status (PLACED_VERIFIED/PLACED_MISMATCH/NOT_PLACED/OTHER_ORDER_EXISTS/UNKNOWN).
    """
    p_st = str(plan_status or "WAIT").strip().upper()
    if p_st.startswith("READY"):
        p_st = "READY"
    elif p_st.startswith("EXPIRED"):
        p_st = "EXPIRED"
    elif p_st.startswith("INVALID"):
        p_st = "INVALID"
    elif p_st not in ("READY", "WAIT", "INVALID", "EXPIRED"):
        p_st = "WAIT"

    if p_st != "READY":
        return p_st

    rec = recon if isinstance(recon, dict) else {}
    b_st = str(rec.get("broker_order_status") or "UNKNOWN").strip().upper()
    ord_t = str(order_type or rec.get("plan_order_type") or "").strip().upper()
    if ord_t == "NONE":
        ord_t = ""
    try:
        ent_v = float(entry_price or rec.get("plan_entry") or 0.0)
    except (TypeError, ValueError):
        ent_v = 0.0
    ord_part = f"{ord_t} {ent_v:.2f}".strip() if (ent_v > 0 and ord_t) else (f"{ent_v:.2f}" if ent_v > 0 else ord_t)

    tickets = [t for t in (rec.get("matched_tickets") or []) if t is not None]
    if len(tickets) == 1:
        tkt_tag = f"#{tickets[0]}"
    elif len(tickets) > 1:
        extra_tkts = ", ".join(f"#{t}" for t in tickets[1:])
        tkt_tag = f"#{tickets[0]} (+{len(tickets) - 1}: {extra_tkts})"
    else:
        tkt_tag = ""

    if b_st == "PLACED_VERIFIED":
        base = f"READY | MT5 PLACED {tkt_tag}".strip()
        return f"{base} | {ord_part}" if ord_part else base

    if b_st == "PLACED_MISMATCH":
        if rec.get("sl_missing_high_risk") and not rec.get("tp_match", True):
            mm_tag = "SL/TP MISMATCH (SL MISSING — HIGH RISK)"
        elif rec.get("sl_missing_high_risk"):
            mm_tag = "SL MISMATCH (SL MISSING — HIGH RISK)"
        elif not rec.get("sl_match", True) and not rec.get("tp_match", True):
            mm_tag = "SL/TP MISMATCH"
        elif not rec.get("sl_match", True):
            mm_tag = "SL MISMATCH"
        else:
            mm_tag = "TP MISMATCH"
        base = f"READY | MT5 PLACED {tkt_tag}".strip()
        return f"{base} | {mm_tag} | {ord_part}" if ord_part else f"{base} | {mm_tag}"

    if b_st == "NEARBY_MISMATCH":
        pm = rec.get("primary_match") or {}
        p_tkt = f"#{pm.get('ticket')}" if pm.get("ticket") else (f"#{tickets[0]}" if tickets else "")
        p_diff = float(pm.get("entry_diff_pt") or 0.0)
        p_ent = float(pm.get("entry_price") or 0.0)
        p_type = pm.get("order_type") or ord_t
        sl_warn = " [SL MISSING — HIGH RISK]" if rec.get("sl_missing_high_risk") else ""
        diff_str = f"Δentry {p_diff:+.2f} pt"
        tkt_str = f" {p_tkt}".rstrip()
        base = f"READY | MT5 NOT PLACED (NEARBY MISMATCH{tkt_str} MT5 {p_ent:.2f} vs Plan {ent_v:.2f}, {diff_str}{sl_warn})".strip()
        return base

    if b_st == "WRONG_TYPE":
        near = rec.get("nearest_unmatched_order") or {}
        n_tkt = f"#{near.get('ticket')}" if near.get("ticket") else ""
        n_type = near.get("order_type") or "OPPOSITE TYPE"
        n_ent = float(near.get("entry_price") or 0.0)
        tkt_str = f" {n_tkt}".rstrip()
        return f"READY | MT5 NOT PLACED (WRONG TYPE{tkt_str} {n_type} @ {n_ent:.2f} vs Plan {ord_part})".strip()

    if b_st == "NOT_PLACED":
        return f"READY | MT5 NOT PLACED | {ord_part}" if ord_part else "READY | MT5 NOT PLACED"

    if b_st == "OTHER_ORDER_EXISTS":
        return f"READY | MT5 NOT PLACED | {ord_part} (OTHER_ORDER_EXISTS)" if ord_part else "READY | MT5 NOT PLACED (OTHER_ORDER_EXISTS)"

    return f"READY | MT5 STATUS UNKNOWN | {ord_part}" if ord_part else "READY | MT5 STATUS UNKNOWN"


def format_pending_reconciliation_evidence(recon):
    """Menyusun bukti rekonsiliasi MT5 lengkap beserta detail mismatch SL/TP untuk `pantau detail` dan `audit`."""
    if not isinstance(recon, dict):
        return "MT5 STATUS UNKNOWN (Reconciliation data unavailable)"
    b_st = str(recon.get("broker_order_status") or "UNKNOWN").upper()
    digits = int(recon.get("digits") or 3)
    entry_tol = float(recon.get("entry_tolerance") or PENDING_ENTRY_TOLERANCE_MIN_PT)
    level_tol = float(recon.get("level_tolerance") or PENDING_LEVEL_TOLERANCE_MIN_PT)
    ts_chk = recon.get("retrieved_at") or "N/A"
    sym = recon.get("symbol") or SYMBOL

    if b_st == "UNKNOWN" or not recon.get("query_ok"):
        rsn = recon.get("reason") or "MT5 query failed/unavailable"
        return f"UNKNOWN — Verifikasi MT5 gagal/tidak tersedia ({rsn}) | Checked: {ts_chk}"

    matched = recon.get("matched_orders") or []
    if b_st in ("PLACED_VERIFIED", "PLACED_MISMATCH") and matched:
        m0 = matched[0]
        cnt = len(matched)
        tkts_str = ", ".join(f"#{m.get('ticket')}" for m in matched)
        cnt_lbl = f"{cnt} order ({tkts_str})" if cnt > 1 else f"Ticket #{m0.get('ticket')}"
        o_ent = float(m0.get("entry_price") or 0.0)
        o_sl = float(m0.get("sl") or 0.0)
        o_tp = float(m0.get("tp") or 0.0)
        o_vol = float(m0.get("volume") or 0.0)
        e_diff = float(m0.get("entry_diff_pt") or 0.0)
        p_sl = float(recon.get("plan_sl") or 0.0)
        p_tp1 = float(recon.get("plan_tp1") or 0.0)
        p_ent = float(recon.get("plan_entry") or 0.0)
        match_qual = recon.get("match_quality") or ("EXACT MATCH" if abs(e_diff) <= float(recon.get("tick_size") or 0.01) + 1e-9 else "ACCEPTED TOLERANCE")

        if m0.get("sl_missing"):
            sl_ev = f"SL=0.0 vs Plan {p_sl:.2f} [SL MISSING — HIGH RISK]"
        elif m0.get("sl_match"):
            sl_ev = f"SL={o_sl:.{digits}f} (OK vs Plan {p_sl:.2f}, tol {level_tol:.3f})"
        else:
            sl_ev = f"SL={o_sl:.{digits}f} vs Plan {p_sl:.2f} [SL MISMATCH Δ{float(m0.get('sl_diff_pt') or 0.0):+.{digits}f} pt]"

        if m0.get("tp_missing") and p_tp1 > 0:
            tp_ev = f"TP=0.0 vs Plan {p_tp1:.2f} [TP MISSING]"
        elif m0.get("tp_match"):
            tp_ev = f"TP={o_tp:.{digits}f} (OK vs Plan {p_tp1:.2f})"
        else:
            tp_ev = f"TP={o_tp:.{digits}f} vs Plan {p_tp1:.2f} [TP MISMATCH Δ{float(m0.get('tp_diff_pt') or 0.0):+.{digits}f} pt]"

        tol_note = f"{match_qual} | MT5 {o_ent:.{digits}f} vs Plan {p_ent:.2f}, Δ {e_diff:+.{digits}f} pt <= tol {entry_tol:.3f} pt"
        return (
            f"{b_st} — {cnt_lbl} | {m0.get('symbol', sym)} {m0.get('order_type')} {o_vol:g} lot @ {o_ent:.{digits}f} "
            f"({tol_note}) | {sl_ev} | {tp_ev} | Checked: {ts_chk}"
        )

    if b_st == "NEARBY_MISMATCH":
        pm = recon.get("primary_match") or {}
        p_tkt = f"#{pm.get('ticket')}" if pm.get("ticket") else "N/A"
        o_ent = float(pm.get("entry_price") or 0.0)
        o_vol = float(pm.get("volume") or 0.0)
        e_diff = float(pm.get("entry_diff_pt") or 0.0)
        p_ent = float(recon.get("plan_entry") or 0.0)
        p_ord = str(recon.get("plan_order_type") or "PENDING")
        sl_warn = " [SL MISSING — HIGH RISK]" if recon.get("sl_missing_high_risk") else ""
        return (
            f"NEARBY_MISMATCH — Ticket {p_tkt} | {pm.get('symbol', sym)} {pm.get('order_type', p_ord)} {o_vol:g} lot @ {o_ent:.{digits}f} "
            f"(MT5 {o_ent:.{digits}f} vs Plan {p_ent:.2f}, Δentry {e_diff:+.{digits}f} pt > tol {entry_tol:.3f} pt; NOT PLACED){sl_warn} | Checked: {ts_chk}"
        )

    tot_sym = int(recon.get("symbol_orders_count") or 0)
    p_ent = float(recon.get("plan_entry") or 0.0)
    p_ord = str(recon.get("plan_order_type") or "PENDING")

    if b_st in ("WRONG_TYPE", "TYPE_MISMATCH"):
        near = recon.get("nearest_unmatched_order") or {}
        n_tkt = f"#{near.get('ticket')}" if near.get("ticket") else ""
        n_type = near.get("order_type") or "OPPOSITE TYPE"
        n_ent = float(near.get("entry_price") or 0.0)
        n_diff = float(near.get("entry_diff_pt") or 0.0)
        return (
            f"WRONG_TYPE — Ditemukan order {n_type} {n_tkt} @ {n_ent:.{digits}f} di {sym} (Δ {n_diff:+.{digits}f} pt), "
            f"berbeda tipe dari plan {p_ord} @ {p_ent:.2f} (NOT PLACED) | Checked: {ts_chk}"
        )
    if b_st == "OTHER_ORDER_EXISTS":
        near = recon.get("nearest_unmatched_order")
        if isinstance(near, dict) and near.get("ticket") is not None:
            n_ent = float(near.get("entry_price") or 0.0)
            n_diff = float(near.get("entry_diff_pt") or (n_ent - p_ent))
            n_vol = float(near.get("volume") or 0.0)
            n_sl = float(near.get("sl") or 0.0)
            n_tp = float(near.get("tp") or 0.0)
            z_tag = " | Inside Plan Zone" if near.get("inside_zone") else ""
            sl_warn = " [SL MISSING — HIGH RISK]" if near.get("sl_missing") else ""
            return (
                f"OTHER_ORDER_EXISTS — Tidak ada {p_ord} @ {p_ent:.2f} (tol ±{entry_tol:.3f} pt) dari {tot_sym} pending order {sym} | "
                f"Nearest: #{near.get('ticket')} {near.get('order_type')} {n_vol:g} lot @ {n_ent:.{digits}f} "
                f"(Δentry {n_diff:+.{digits}f} pt > tol {entry_tol:.3f} pt{z_tag} | SL={n_sl:.{digits}f}{sl_warn} | TP={n_tp:.{digits}f}) | Checked: {ts_chk}"
            )
        return f"OTHER_ORDER_EXISTS — {tot_sym} pending order aktif di {sym}, tidak ada yang cocok dengan {p_ord} @ {p_ent:.2f} (tol ±{entry_tol:.3f} pt) | Checked: {ts_chk}"

    return f"NOT_PLACED — 0 pending order di {sym} (Terverifikasi live MT5) | Checked: {ts_chk}"


def reconcile_pending_plan_with_mt5(plan, mt5_snapshot=None, symbol=SYMBOL, excluded_tickets=None):
    """
    Rekonsiliasi READ-ONLY satu rencana pending scanner terhadap pending order aktual di MT5.
    Aturan:
    1. Filter berdasarkan akun yang terhubung dan simbol trading yang tepat.
    2. Wajib cocok tipe order kompatibel (SELL LIMIT == SELL LIMIT; tidak pernah cocok hanya karena arah SELL).
    3. Bandingkan entry price menggunakan tick_size simbol dan toleransi terukur (PENDING_ENTRY_TOLERANCE_MIN_PT).
    4. Bandingkan SL dan TP secara independen.
    5. Tidak mewajibkan magic number / comment khusus scanner (mendukung manual order).
    6. Jika >1 order cocok, ekspos jumlah dan seluruh tiket.
    7. Tidak menggunakan tiket yang sama untuk mengklaim dua rencana berbeda (via excluded_tickets).
    """
    p_dict = plan if isinstance(plan, dict) else {}
    target_sym, accepted_syms = _acceptable_trade_symbols(symbol or p_dict.get("symbol") or SYMBOL)
    snap = (
        mt5_snapshot
        if (isinstance(mt5_snapshot, dict) and "query_ok" in mt5_snapshot and "orders" in mt5_snapshot)
        else fetch_mt5_pending_orders_snapshot(symbol=target_sym, orders_override=mt5_snapshot)
    )

    raw_st = str(p_dict.get("plan_status") or p_dict.get("early_pending_status") or p_dict.get("status") or "WAIT").strip().upper()
    if raw_st.startswith("READY"):
        plan_status = "READY"
    elif raw_st.startswith("EXPIRED") or p_dict.get("is_expired"):
        plan_status = "EXPIRED"
    elif raw_st.startswith("INVALID") or p_dict.get("valid") is False:
        plan_status = "INVALID" if raw_st.startswith("INVALID") else "WAIT"
    else:
        plan_status = "WAIT"

    order_type = str(p_dict.get("order_type") or "").strip().upper()
    if not order_type or order_type == "NONE":
        p_dir = str(p_dict.get("direction") or "").strip().upper()
        p_zt = str(p_dict.get("zone_type") or "").strip().upper()
        if p_dir == "BUY" or p_zt == "DEMAND":
            order_type = "BUY LIMIT"
        elif p_dir == "SELL" or p_zt == "SUPPLY":
            order_type = "SELL LIMIT"
        else:
            order_type = ""

    try:
        plan_entry = float(p_dict.get("entry_price") if p_dict.get("entry_price") is not None else (p_dict.get("entry") or 0.0))
    except (TypeError, ValueError):
        plan_entry = 0.0
    try:
        plan_sl = float(p_dict.get("sl") or 0.0)
    except (TypeError, ValueError):
        plan_sl = 0.0
    try:
        plan_tp1 = float(p_dict.get("tp1") or 0.0)
    except (TypeError, ValueError):
        plan_tp1 = 0.0
    try:
        plan_tp2 = float(p_dict.get("tp2") or 0.0)
    except (TypeError, ValueError):
        plan_tp2 = 0.0

    z_rng = p_dict.get("zone_range") if isinstance(p_dict.get("zone_range"), (list, tuple)) and len(p_dict.get("zone_range")) >= 2 else (0.0, 0.0)
    try:
        zone_low = float(p_dict.get("zone_low") or p_dict.get("price_low") or z_rng[0] or 0.0)
        zone_high = float(p_dict.get("zone_high") or p_dict.get("price_high") or z_rng[1] or 0.0)
    except (TypeError, ValueError):
        zone_low, zone_high = 0.0, 0.0

    zone_id = str(p_dict.get("zone_id") or p_dict.get("source_zone_id") or "").strip()
    digits = int(snap.get("digits") or 2)
    tick_size = float(snap.get("tick_size") or 0.01)
    entry_tol = float(snap.get("entry_tolerance") or PENDING_ENTRY_TOLERANCE_MIN_PT)
    level_tol = float(snap.get("level_tolerance") or PENDING_LEVEL_TOLERANCE_MIN_PT)
    retrieved_at = str(snap.get("retrieved_at") or "")
    orders = list(snap.get("orders") or [])
    excl_set = set(excluded_tickets or ())

    base_res = {
        "zone_id": zone_id,
        "symbol": target_sym,
        "plan_status": plan_status,
        "plan_order_type": order_type,
        "plan_entry": plan_entry,
        "plan_sl": plan_sl,
        "plan_tp1": plan_tp1,
        "plan_tp2": plan_tp2,
        "query_ok": bool(snap.get("query_ok")),
        "reason": str(snap.get("reason") or ""),
        "digits": digits,
        "tick_size": tick_size,
        "entry_tolerance": entry_tol,
        "level_tolerance": level_tol,
        "retrieved_at": retrieved_at,
        "symbol_orders_count": len(orders),
        "broker_order_status": "UNKNOWN",
        "is_placed": False,
        "matched_count": 0,
        "matched_tickets": [],
        "matched_orders": [],
        "primary_match": None,
        "nearest_unmatched_order": None,
        "sl_match": False,
        "tp_match": False,
        "sl_missing_high_risk": False,
    }

    if not snap.get("query_ok"):
        base_res["compact_status"] = format_pending_compact_status(plan_status, base_res, order_type=order_type, entry_price=plan_entry)
        base_res["evidence_summary"] = format_pending_reconciliation_evidence(base_res)
        return base_res

    if plan_status != "READY" or plan_entry <= 0 or not order_type:
        base_res["broker_order_status"] = "OTHER_ORDER_EXISTS" if len(orders) > 0 else "NOT_PLACED"
        base_res["compact_status"] = format_pending_compact_status(plan_status, base_res, order_type=order_type, entry_price=plan_entry)
        base_res["evidence_summary"] = format_pending_reconciliation_evidence(base_res)
        return base_res

    matched_orders = []
    price_mismatch_orders = []
    type_mismatch_orders = []

    # Proximity threshold terukur untuk membedakan nearby orders vs completely unrelated distant orders
    # Sesuai aturan: jika di dalam zona atau jarak <= max(entry_tol, 1.5 pt), dievaluasi sebagai nearby
    manual_entry_tol = max(entry_tol, 1.5)

    for ord_item in orders:
        tkt = ord_item.get("ticket")
        if tkt in excl_set:
            continue
        if str(ord_item.get("symbol") or "").upper() not in accepted_syms:
            continue
        o_type = str(ord_item.get("order_type") or "").upper()
        o_entry = float(ord_item.get("entry_price") or 0.0)
        if o_entry <= 0:
            continue
        o_sl = float(ord_item.get("sl") or 0.0)
        o_tp = float(ord_item.get("tp") or 0.0)
        sl_missing = (o_sl <= 0.0)
        tp_missing = (o_tp <= 0.0)
        sl_diff = round(o_sl - plan_sl, max(digits, 3)) if (not sl_missing and plan_sl > 0) else None
        sl_ok = (not sl_missing) and (plan_sl <= 0 or abs(o_sl - plan_sl) <= level_tol + 1e-9)

        tp1_ok = (not tp_missing) and (plan_tp1 <= 0 or abs(o_tp - plan_tp1) <= level_tol + 1e-9)
        tp2_ok = (not tp_missing) and (plan_tp2 > 0 and abs(o_tp - plan_tp2) <= level_tol + 1e-9)
        tp_ok = tp1_ok or tp2_ok
        tp_diff = round(o_tp - plan_tp1, max(digits, 3)) if (not tp_missing and plan_tp1 > 0) else None

        entry_diff = round(o_entry - plan_entry, max(digits, 3))
        in_zone = bool(zone_low > 0 and zone_high > 0 and (zone_low - 1e-9 <= o_entry <= zone_high + 1e-9))
        is_nearby = in_zone or (abs(entry_diff) <= manual_entry_tol + 1e-9)

        if o_type != order_type:
            # Hanya catat sebagai wrong type jika order berada di sekitar level rencana (in zone / nearby)
            # Jangan pernah gunakan order berlawanan arah yang jauh sebagai match atau type mismatch
            if is_nearby:
                type_mismatch_orders.append({
                    "ticket": tkt,
                    "order_type": o_type,
                    "direction": ord_item.get("direction"),
                    "symbol": ord_item.get("symbol"),
                    "entry_price": o_entry,
                    "sl": o_sl,
                    "tp": o_tp,
                    "volume": float(ord_item.get("volume") or 0.0),
                    "time_setup": ord_item.get("time_setup", 0),
                    "retrieved_at": ord_item.get("retrieved_at") or retrieved_at,
                    "entry_diff_pt": entry_diff,
                    "inside_zone": in_zone,
                    "sl_match": sl_ok,
                    "tp_match": tp_ok,
                    "sl_missing": sl_missing,
                    "tp_missing": tp_missing,
                    "sl_diff_pt": sl_diff,
                    "tp_diff_pt": tp_diff,
                })
            continue

        # Cek apakah exact match atau price mismatch terukur
        mm_reasons = []
        if sl_missing:
            mm_reasons.append("SL MISSING — HIGH RISK")
        elif not sl_ok:
            mm_reasons.append(f"SL MISMATCH (MT5 {o_sl:.{digits}f} vs Plan {plan_sl:.2f}, Δ{sl_diff:+.{digits}f} pt)")
        if tp_missing and plan_tp1 > 0:
            mm_reasons.append("TP MISSING")
        elif not tp_ok:
            mm_reasons.append(f"TP MISMATCH (MT5 {o_tp:.{digits}f} vs Plan {plan_tp1:.2f}, Δ{tp_diff:+.{digits}f} pt)")

        order_data = {
            "ticket": tkt,
            "order_type": o_type,
            "direction": ord_item.get("direction"),
            "symbol": ord_item.get("symbol"),
            "entry_price": o_entry,
            "sl": o_sl,
            "tp": o_tp,
            "volume": float(ord_item.get("volume") or 0.0),
            "time_setup": ord_item.get("time_setup", 0),
            "retrieved_at": ord_item.get("retrieved_at") or retrieved_at,
            "entry_diff_pt": entry_diff,
            "inside_zone": in_zone,
            "sl_match": sl_ok,
            "tp_match": tp_ok,
            "sl_missing": sl_missing,
            "tp_missing": tp_missing,
            "sl_diff_pt": sl_diff,
            "tp_diff_pt": tp_diff,
            "mismatch_reasons": mm_reasons,
        }

        if abs(entry_diff) <= entry_tol + 1e-9:
            order_data["match_quality"] = "EXACT_MATCH" if abs(entry_diff) <= tick_size + 1e-9 else "ACCEPTED_TOLERANCE"
            matched_orders.append(order_data)
        elif is_nearby:
            # Order tipe sama tetapi selisih entry melebihi entry_tol (NEARBY MISMATCH)
            order_data["match_quality"] = "NEARBY_MISMATCH"
            price_mismatch_orders.append(order_data)

    matched_orders.sort(key=lambda m: (abs(m["entry_diff_pt"]), not (m["sl_match"] and m["tp_match"]), m["ticket"] or 0))
    price_mismatch_orders.sort(key=lambda m: (abs(m["entry_diff_pt"]), not (m["sl_match"] and m["tp_match"]), m["ticket"] or 0))
    type_mismatch_orders.sort(key=lambda m: (abs(m["entry_diff_pt"]), m["ticket"] or 0))

    if matched_orders:
        all_verified = all(m["sl_match"] and m["tp_match"] for m in matched_orders)
        b_status = "PLACED_VERIFIED" if all_verified else "PLACED_MISMATCH"
        primary_qual = matched_orders[0].get("match_quality", "ACCEPTED_TOLERANCE")
        base_res.update({
            "broker_order_status": b_status,
            "is_placed": True,
            "match_quality": primary_qual,
            "matched_count": len(matched_orders),
            "matched_tickets": [m["ticket"] for m in matched_orders if m.get("ticket") is not None],
            "matched_orders": matched_orders,
            "primary_match": matched_orders[0],
            "sl_match": all(m["sl_match"] for m in matched_orders),
            "tp_match": all(m["tp_match"] for m in matched_orders),
            "sl_missing_high_risk": any(m["sl_missing"] for m in matched_orders),
        })
    elif price_mismatch_orders:
        p0 = price_mismatch_orders[0]
        # JANGAN label PLACED jika hanya order terdekat di luar toleransi (NEARBY MISMATCH)
        base_res.update({
            "broker_order_status": "NEARBY_MISMATCH",
            "is_placed": False,
            "match_quality": "NEARBY_MISMATCH",
            "matched_count": 0,
            "matched_tickets": [p0["ticket"]] if p0.get("ticket") else [],
            "matched_orders": [],
            "primary_match": p0,
            "sl_match": bool(p0.get("sl_match")),
            "tp_match": bool(p0.get("tp_match")),
            "sl_missing_high_risk": bool(p0.get("sl_missing")),
        })
    elif type_mismatch_orders:
        t0 = type_mismatch_orders[0]
        base_res.update({
            "broker_order_status": "WRONG_TYPE",
            "is_placed": False,
            "match_quality": "WRONG_TYPE",
            "nearest_unmatched_order": t0,
        })
    else:
        avail_orders = [o for o in orders if o.get("ticket") not in excl_set]
        if len(avail_orders) > 0 or len(orders) > 0:
            base_res.update({
                "broker_order_status": "OTHER_ORDER_EXISTS",
                "is_placed": False,
                "match_quality": "NOT_FOUND",
            })
            same_type_pool = [o for o in avail_orders if str(o.get("order_type") or "").upper() == order_type]
            pool = same_type_pool or avail_orders or orders
            if pool:
                nearest = min(pool, key=lambda o: abs(float(o.get("entry_price") or 0.0) - plan_entry))
                n_entry = float(nearest.get("entry_price") or 0.0)
                n_sl = float(nearest.get("sl") or 0.0)
                n_tp = float(nearest.get("tp") or 0.0)
                in_z = bool(zone_low > 0 and zone_high > 0 and (zone_low - 1e-9 <= n_entry <= zone_high + 1e-9))
                base_res["nearest_unmatched_order"] = {
                    "ticket": nearest.get("ticket"),
                    "order_type": nearest.get("order_type"),
                    "symbol": nearest.get("symbol"),
                    "entry_price": n_entry,
                    "sl": n_sl,
                    "tp": n_tp,
                    "volume": float(nearest.get("volume") or 0.0),
                    "retrieved_at": nearest.get("retrieved_at") or retrieved_at,
                    "entry_diff_pt": round(n_entry - plan_entry, max(digits, 3)),
                    "inside_zone": in_z,
                    "sl_missing": (n_sl <= 0.0),
                    "tp_missing": (n_tp <= 0.0),
                }
        else:
            base_res.update({
                "broker_order_status": "NOT_PLACED",
                "is_placed": False,
                "match_quality": "NOT_FOUND",
            })

    base_res["compact_status"] = format_pending_compact_status(plan_status, base_res, order_type=order_type, entry_price=plan_entry)
    base_res["evidence_summary"] = format_pending_reconciliation_evidence(base_res)
    return base_res


def reconcile_pending_plans_batch(plans, mt5_snapshot=None, symbol=SYMBOL):
    """
    Rekonsiliasi sekumpulan pending plan (Selected Pending + Alt Pending) secara independen
    menggunakan 1 snapshot MT5 yang sama. Mencegah 1 tiket order MT5 yang sama diklaim oleh
    dua rencana zona yang berbeda.
    """
    if not plans:
        return []
    target_sym, _ = _acceptable_trade_symbols(symbol)
    snap = (
        mt5_snapshot
        if (isinstance(mt5_snapshot, dict) and "query_ok" in mt5_snapshot and "orders" in mt5_snapshot)
        else fetch_mt5_pending_orders_snapshot(symbol=target_sym, orders_override=mt5_snapshot)
    )
    claimed_tickets = set()
    results = []
    for p in plans:
        if not isinstance(p, dict):
            continue
        recon = reconcile_pending_plan_with_mt5(
            p,
            mt5_snapshot=snap,
            symbol=target_sym,
            excluded_tickets=claimed_tickets,
        )
        for tkt in (recon.get("matched_tickets") or []):
            if tkt is not None:
                claimed_tickets.add(tkt)
        p["plan_status"] = recon["plan_status"]
        p["broker_order_status"] = recon["broker_order_status"]
        p["mt5_execution_state"] = recon["broker_order_status"]
        p["is_placed"] = recon["is_placed"]
        p["sl_missing_high_risk"] = recon["sl_missing_high_risk"]
        p["mt5_reconciliation"] = recon
        p["compact_status"] = recon["compact_status"]
        results.append(recon)
    return results


def cek_posisi_terbuka_mt5(harga_skrg=0.0, pending_plans=None):
    """Membaca identitas akun real, posisi trading yang sedang running (floating), Pending Order, & riwayat deal hari ini di MT5."""
    out = ["\n===== [MT5 ACTUAL ORDER STATUS] FAKTA AKUN MT5, POSISI RUNNING & PENDING ORDER (DATA ASLI LIVE) ====="]
    out.append("  [KATEGORI DATA]: [MT5 ACTUAL ORDER STATUS] (Bukan Rekomendasi AI dan Bukan Setup Plan)")
    if not pastikan_mt5_aktif():
        out.append("DATA UNAVAILABLE: Koneksi MT5 tidak aktif.")
        return "\n".join(out) + "\n"

    acc = mt5.account_info()
    if acc is None:
        out.append("DATA UNAVAILABLE: Data akun MT5 tidak tersedia.")
        return "\n".join(out) + "\n"

    sym = mt5.symbol_info(SYMBOL)
    digits = int(getattr(sym, "digits", 2)) if sym else 2
    tick = mt5.symbol_info_tick(SYMBOL)
    cur = getattr(acc, "currency", "USD")
    eq = float(acc.equity if acc and acc.equity > 0 else (acc.balance if acc else 0.0))
    try:
        fallback_px = float(harga_skrg) if not isinstance(harga_skrg, str) else 0.0
    except (TypeError, ValueError):
        fallback_px = 0.0
    bid_live = float(tick.bid) if tick and tick.bid > 0 else fallback_px
    ask_live = float(tick.ask) if tick and tick.ask > 0 else fallback_px

    mode_akun = "REAL ACCOUNT" if getattr(acc, "trade_mode", 0) == getattr(mt5, "ACCOUNT_TRADE_MODE_REAL", 2) else "DEMO/CONTEST"
    out.append(
        f"  [IDENTITAS AKUN MT5]: Tipe: {mode_akun} | "
        f"Mata Uang: {cur} (JANGAN sebut USD/$ jika tertulis {cur}!) | "
        f"Balance: {acc.balance:,.2f} {cur} | Equity: {acc.equity:,.2f} {cur} | Floating Total: {acc.profit:+,.2f} {cur} | Free Margin: {acc.margin_free:,.2f} {cur}"
    )

    pos = mt5.positions_get(symbol=SYMBOL)
    orders = mt5.orders_get(symbol=SYMBOL)

    if pos is None and orders is None:
        out.append("  [POSISI & PENDING ORDER]: DATA UNAVAILABLE (Gagal membaca data posisi/order dari MT5).")
        return "\n".join(out) + "\n"

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
                    sl_info = f"{p.sl:.{digits}f} [✅ SUDAH DIKUNCI BEP+ (+{kunci_pt:.2f} pt di area profit) — POSISI 100% BEBAS RISIKO!]"
                else:
                    jrk_sl = (harga_tutup - p.sl) if arah == "BUY" else (p.sl - harga_tutup)
                    sl_info = f"{p.sl:.{digits}f} (jarak dari harga sekarang ke SL: {jrk_sl:.2f} pt)"
            else:
                sl_info = f"BELUM DISET (⚠️ RAWAN TANPA PENGAMAN!) [CRITICAL WARNING: Posisi #{p.ticket} TANPA STOP LOSS!]"
            tp_info = f"{p.tp:.{digits}f}" if (p.tp and p.tp > 0) else "BELUM DISET (Terbuka / Runner)"
            out.append(
                f"  - Ticket #{p.ticket} | Simbol: {getattr(p, 'symbol', SYMBOL)} | {arah} {p.volume} lot @ {p.price_open:.{digits}f} (Harga sekarang: {harga_tutup:.{digits}f}) | "
                f"SL: {sl_info} | TP: {tp_info} | "
                f"Floating: {p.profit:+,.2f} {cur} ({pct_eq:+.2f}% equity | {poin:+.2f} poin)"
            )
            if not (p.sl and p.sl > 0):
                out.append(f"    CRITICAL WARNING: Posisi #{p.ticket} tidak memiliki Stop Loss! Segera pasang Stop Loss manual di MT5 untuk membatasi risiko.")
    else:
        out.append("  [POSISI RUNNING / FLOATING SAAT INI]: Total 0 posisi running (Tidak ada posisi market terbuka).")

    if orders:
        out.append(f"  [PENDING ORDER TERPASANG SAAT INI — TOTAL {len(orders)} ORDER (WAJIB DIEVALUASI VALIDITAS ZONANYA OLEH AI)]:")
        for o in orders:
            nama_ord, arah_ord = _info_tipe_pending(o.type)
            vol_ord = getattr(o, "volume_current", getattr(o, "volume_initial", 0.0))
            jarak_ord = abs(bid_live - o.price_open) if bid_live > 0 else 0.0
            letak_ord = "di atas harga sekarang" if o.price_open > bid_live else "di bawah harga sekarang"
            if (o.sl and o.sl > 0):
                sl_ord_txt = f"{o.sl:.{digits}f} (jarak SL {abs(o.price_open - o.sl):.2f} pt)"
            else:
                sl_ord_txt = f"BELUM DISET (⚠️ WAJIB INGATKAN PASANG SL!) [CRITICAL WARNING: Pending Order #{o.ticket} TANPA STOP LOSS!]"
            tp_ord_txt = f"{o.tp:.{digits}f}" if (o.tp and o.tp > 0) else "BELUM DISET"
            exp_time = getattr(o, "time_expiration", 0)
            if isinstance(exp_time, (int, float)) and exp_time > 0:
                try:
                    exp_txt = pd.Timestamp(int(exp_time), unit="s").strftime("%Y-%m-%d %H:%M")
                except Exception:
                    exp_txt = str(exp_time)
            else:
                exp_txt = "GTC (Good till cancelled)"

            out.append(
                f"  - Ticket #{o.ticket} | Simbol: {getattr(o, 'symbol', SYMBOL)} | {nama_ord} ({arah_ord}) {vol_ord} lot @ {o.price_open:.{digits}f} "
                f"(Jarak: {jarak_ord:.2f} poin {letak_ord}) | "
                f"SL: {sl_ord_txt} | TP: {tp_ord_txt} | Expiration: {exp_txt} | Status: Placed in MT5"
            )
            if not (o.sl and o.sl > 0):
                out.append(f"    CRITICAL WARNING: Pending Order #{o.ticket} tidak memiliki Stop Loss! Segera pasang Stop Loss manual di MT5 untuk membatasi risiko.")
    else:
        out.append("  [PENDING ORDER TERPASANG SAAT INI]: Total 0 order (Tidak ada pending order aktif).")

    # Rekonsiliasi Pending Plan Scanner vs MT5 Pending Orders (Read-Only)
    try:
        plans_to_check = list(pending_plans) if isinstance(pending_plans, (list, tuple)) else []
        if not plans_to_check:
            from modules.zone_scanner import get_zone_tracker
            trk = get_zone_tracker(auto_save=False)
            tracked = list((trk.tracked_zones or {}).values()) if trk else []
            ready_recs = [
                r for r in tracked
                if isinstance(r, dict)
                and str(r.get("status") or "").upper() not in ("INVALIDATED", "EXPIRED", "MITIGATED")
                and isinstance(r.get("pending_plan"), dict)
                and r["pending_plan"].get("valid") is True
                and not r["pending_plan"].get("is_expired")
                and str(r["pending_plan"].get("status") or "").upper().startswith("READY")
            ]
            if ready_recs:
                latest_snap = max((str(r.get("snapshot_time") or "") for r in ready_recs), default="")
                if latest_snap:
                    cur_recs = [r for r in ready_recs if str(r.get("snapshot_time") or "") == latest_snap]
                    if cur_recs:
                        ready_recs = cur_recs
                ready_recs.sort(key=lambda r: float((r.get("pending_plan") or {}).get("zone_score") or r.get("score") or 0.0), reverse=True)
                for idx_r, r_item in enumerate(ready_recs):
                    p_copy = dict(r_item["pending_plan"])
                    p_copy["zone_id"] = p_copy.get("zone_id") or r_item.get("zone_id")
                    p_copy["plan_role"] = "SELECTED_PENDING" if idx_r == 0 else "ALT_PENDING"
                    plans_to_check.append(p_copy)
        if plans_to_check and orders is not None:
            ord_snap = fetch_mt5_pending_orders_snapshot(symbol=SYMBOL, orders_override=list(orders))
            reconcile_pending_plans_batch(plans_to_check, mt5_snapshot=ord_snap, symbol=SYMBOL)
            out.append("  [REKONSILIASI PENDING PLAN SCANNER vs MT5 PENDING ORDERS (READ-ONLY)]:")
            for idx_p, pln in enumerate(plans_to_check):
                role_lbl = pln.get("plan_role") or ("SELECTED PENDING" if idx_p == 0 else "ALT PENDING")
                rec_p = pln.get("mt5_reconciliation") or {}
                out.append(
                    f"  - [{role_lbl}] {pln.get('zone_id', 'N/A')} | Plan: {pln.get('order_type')} @ {float(pln.get('entry_price', pln.get('entry', 0.0))):.2f} "
                    f"(SL {float(pln.get('sl', 0.0)):.2f} | TP1 {float(pln.get('tp1', 0.0)):.2f}) -> {pln.get('compact_status')}"
                )
                out.append(f"    Evidence: {rec_p.get('evidence_summary')}")
    except Exception as e_rec:
        _catat_error("cek_posisi_terbuka_mt5_reconcile", e_rec)

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
