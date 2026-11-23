# -*- coding: utf-8 -*-
"""
modules/mt5_conn.py
Koneksi ke MetaTrader 5, penarikan data candle & tick, kalkulasi indikator dasar, dan audit akun/lot/MM.
"""
import math
import threading
import time
from datetime import datetime, timezone, timedelta
import MetaTrader5 as mt5
import numpy as np
import pandas as pd

from modules.config import SYMBOL, TIMEFRAMES, MAKS_RISIKO_TERBUKA_PCT, MAKS_RUGI_HARIAN_PCT, SERVER_UTC_OFFSET_JAM
from modules.utils import (
    _catat_error,
    deteksi_offset,
    get_symbol_unit_spec,
    price_distance_to_mt5_points,
    price_distance_to_pips,
    format_price_distance,
    WIB_TZ,
)

MT5_TERHUBUNG = False
MT5_IPC_LOCK = threading.RLock()
_MT5_SERVER_OFFSET_CACHE = {"offset": None, "ts": 0.0}
_MT5_ITEM_HISTORY_CACHE = {}


def get_mt5_server_utc_offset_hours(tick=None, allow_live_query=True):
    """Deteksi offset UTC server broker MT5 (jam) dan cache agar konversi timestamp posisi/order lama tetap akurat."""
    if SERVER_UTC_OFFSET_JAM is not None:
        try:
            return float(SERVER_UTC_OFFSET_JAM)
        except (TypeError, ValueError):
            pass
    now_s = time.time()
    if tick is not None and getattr(tick, "time", 0):
        off = deteksi_offset(tick)
        if off is not None:
            _MT5_SERVER_OFFSET_CACHE["offset"] = float(off)
            _MT5_SERVER_OFFSET_CACHE["ts"] = now_s
            return float(off)
    cached_off = _MT5_SERVER_OFFSET_CACHE.get("offset")
    if cached_off is not None and (now_s - float(_MT5_SERVER_OFFSET_CACHE.get("ts") or 0.0)) < 300.0:
        return float(cached_off)
    if allow_live_query:
        try:
            if pastikan_mt5_aktif():
                with MT5_IPC_LOCK:
                    t_live = mt5.symbol_info_tick(SYMBOL)
                if t_live is not None and getattr(t_live, "time", 0):
                    off = deteksi_offset(t_live)
                    if off is not None:
                        _MT5_SERVER_OFFSET_CACHE["offset"] = float(off)
                        _MT5_SERVER_OFFSET_CACHE["ts"] = now_s
                        return float(off)
        except Exception:
            pass
    return float(cached_off) if cached_off is not None else 0.0


def mt5_ts_to_utc_epoch(raw_ts, raw_msc=0, server_offset_hours=None):
    """Konversi timestamp MT5 (waktu server broker, detik/milidetik) ke UTC epoch detik; 0.0 bila tidak tersedia."""
    try:
        msc_v = float(raw_msc or 0.0)
    except (TypeError, ValueError):
        msc_v = 0.0
    try:
        sec_v = float(raw_ts or 0.0)
    except (TypeError, ValueError):
        sec_v = 0.0
    ts_val = (msc_v / 1000.0) if msc_v > 1e11 else (sec_v if sec_v > 0 else 0.0)
    if ts_val <= 0.0 or not math.isfinite(ts_val):
        return 0.0
    off_h = server_offset_hours if server_offset_hours is not None else get_mt5_server_utc_offset_hours(allow_live_query=False)
    if off_h:
        ts_val -= float(off_h) * 3600.0
    return float(ts_val)


def format_mt5_wib(utc_ts, short=False):
    """Format UTC epoch ke WIB (Asia/Jakarta, UTC+7). Mengembalikan '' bila timestamp tidak tersedia (jangan mengarang)."""
    try:
        ts_f = float(utc_ts or 0.0)
    except (TypeError, ValueError):
        return ""
    if ts_f <= 0.0 or not math.isfinite(ts_f):
        return ""
    try:
        dt_wib = datetime.fromtimestamp(ts_f, tz=timezone.utc).astimezone(WIB_TZ)
        if short:
            return dt_wib.strftime("%d-%b %H:%M WIB")
        return dt_wib.strftime("%Y-%m-%d %H:%M:%S WIB")
    except Exception:
        return ""


def fetch_mt5_item_history(
    ticket=None,
    position_id=None,
    is_order=False,
    server_offset_hours=None,
    allow_live_query=True,
):
    """
    Ambil metadata histori MT5 secara targeted berdasarkan ticket/position_id (tanpa full history scan).
    Hasil di-cache per ticket/position_id agar tidak membebani loop pantau tiap tick.
    """
    pid = int(position_id) if (position_id is not None and str(position_id).isdigit() and int(position_id) > 0) else None
    tkt = int(ticket) if (ticket is not None and str(ticket).isdigit() and int(ticket) > 0) else None
    cache_key = ("ORD" if is_order else "POS", pid or tkt)
    if cache_key[1] is None:
        return {}

    now_s = time.time()
    cached = _MT5_ITEM_HISTORY_CACHE.get(cache_key)
    if isinstance(cached, dict):
        ttl = 300.0 if cached.get("closed_ts", 0.0) > 0 else 60.0
        if (now_s - float(cached.get("_fetched_at") or 0.0)) < ttl:
            return cached

    if not allow_live_query:
        return cached if isinstance(cached, dict) else {}

    out = {
        "order_id": tkt if is_order else None,
        "deal_id": None,
        "close_deal_id": None,
        "order_created_ts": 0.0,
        "executed_ts": 0.0,
        "closed_ts": 0.0,
        "order_created_wib": "",
        "executed_wib": "",
        "closed_wib": "",
        "order_created_wib_short": "",
        "executed_wib_short": "",
        "closed_wib_short": "",
        "close_price": 0.0,
        "_fetched_at": now_s,
    }
    try:
        if not pastikan_mt5_aktif():
            return cached if isinstance(cached, dict) else out
        off_h = server_offset_hours if server_offset_hours is not None else get_mt5_server_utc_offset_hours(allow_live_query=True)
        h_orders = ()
        h_deals = ()
        with MT5_IPC_LOCK:
            if pid:
                h_orders = mt5.history_orders_get(position=pid) or ()
                h_deals = mt5.history_deals_get(position=pid) or ()
            elif tkt:
                h_orders = mt5.history_orders_get(ticket=tkt) or ()
                if not is_order:
                    h_deals = mt5.history_deals_get(position=tkt) or ()

        if h_orders:
            h_orders_sorted = sorted(h_orders, key=lambda o: int(getattr(o, "time_setup", 0) or 0))
            o0 = h_orders_sorted[0]
            out["order_id"] = getattr(o0, "ticket", out["order_id"])
            c_ts = mt5_ts_to_utc_epoch(getattr(o0, "time_setup", 0), getattr(o0, "time_setup_msc", 0), off_h)
            if c_ts > 0:
                out["order_created_ts"] = c_ts
                out["order_created_wib"] = format_mt5_wib(c_ts, short=False)
                out["order_created_wib_short"] = format_mt5_wib(c_ts, short=True)
            d_ts = mt5_ts_to_utc_epoch(getattr(o0, "time_done", 0), getattr(o0, "time_done_msc", 0), off_h)
            if d_ts > 0 and getattr(o0, "state", None) == getattr(mt5, "ORDER_STATE_FILLED", 4):
                out["executed_ts"] = d_ts
                out["executed_wib"] = format_mt5_wib(d_ts, short=False)
                out["executed_wib_short"] = format_mt5_wib(d_ts, short=True)

        if h_deals:
            h_deals_sorted = sorted(h_deals, key=lambda d: int(getattr(d, "time", 0) or 0))
            in_entries = (getattr(mt5, "DEAL_ENTRY_IN", 0), getattr(mt5, "DEAL_ENTRY_INOUT", 2))
            out_entries = (getattr(mt5, "DEAL_ENTRY_OUT", 1), getattr(mt5, "DEAL_ENTRY_INOUT", 2), getattr(mt5, "DEAL_ENTRY_OUT_BY", 3))
            in_ds = [d for d in h_deals_sorted if getattr(d, "entry", -1) in in_entries]
            out_ds = [d for d in h_deals_sorted if getattr(d, "entry", -1) in out_entries]
            if in_ds:
                d_in = in_ds[0]
                out["deal_id"] = getattr(d_in, "ticket", None)
                if not out["order_id"]:
                    out["order_id"] = getattr(d_in, "order", None)
                ex_ts = mt5_ts_to_utc_epoch(getattr(d_in, "time", 0), getattr(d_in, "time_msc", 0), off_h)
                if ex_ts > 0:
                    out["executed_ts"] = ex_ts
                    out["executed_wib"] = format_mt5_wib(ex_ts, short=False)
                    out["executed_wib_short"] = format_mt5_wib(ex_ts, short=True)
            if out_ds:
                d_out = out_ds[-1]
                out["close_deal_id"] = getattr(d_out, "ticket", None)
                out["close_price"] = float(getattr(d_out, "price", 0.0) or 0.0)
                cl_ts = mt5_ts_to_utc_epoch(getattr(d_out, "time", 0), getattr(d_out, "time_msc", 0), off_h)
                if cl_ts > 0:
                    out["closed_ts"] = cl_ts
                    out["closed_wib"] = format_mt5_wib(cl_ts, short=False)
                    out["closed_wib_short"] = format_mt5_wib(cl_ts, short=True)

        _MT5_ITEM_HISTORY_CACHE[cache_key] = out
        if len(_MT5_ITEM_HISTORY_CACHE) > 256:
            oldest_k = min(_MT5_ITEM_HISTORY_CACHE.keys(), key=lambda k: float(_MT5_ITEM_HISTORY_CACHE[k].get("_fetched_at") or 0.0))
            _MT5_ITEM_HISTORY_CACHE.pop(oldest_k, None)
    except Exception as exc_h:
        _catat_error("fetch_mt5_item_history", exc_h)
    return out


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
        with MT5_IPC_LOCK:
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

    u_spec = get_symbol_unit_spec(symbol=SYMBOL, sym_info=sym_info)
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
    spread_txt = format_price_distance(spread, spec=u_spec, include_mt5_points=True)

    out.append(f"Saldo (Balance): {bal:.2f} {cur} | Equity: {eq:.2f} {cur} | Contract Size: {contract:g} | Lot Min: {vol_min} | Nilai 1.00 USD/lot: {per_poin:.2f} {cur} | Spread: {spread_txt}")
    if str(cur).upper() != "USD":
        out.append(f"CATATAN: mata uang akun = {cur}. Semua nilai uang di bawah dalam {cur} (bukan USD).")

    risk_konservatif = eq * 0.005  # 0.5% (risiko lebih konservatif)
    risk_standar = eq * 0.01       # 1.0% (risiko lebih tinggi dibandingkan 0.5%)
    out.append(f"Batas Risiko Modal per Trade:")
    out.append(f"  • 0,5% (risiko lebih konservatif)             : {risk_konservatif:.2f} {cur}")
    out.append(f"  • 1,0% (risiko lebih tinggi dibandingkan 0,5%): {risk_standar:.2f} {cur}")
    out.append("Tabel Acuan Ukuran Lot berdasarkan jarak Stop Loss (Price Distance USD/oz; sudah termasuk spread, lot dibulatkan ke bawah):")
    for sl_poin in (3.0, 5.0, 8.0, 10.0, 15.0):
        rugi_per_lot = (sl_poin + spread) * per_poin
        lot_kons, _, over_kons = _hitung_lot(risk_konservatif, rugi_per_lot, vol_min, vol_step, vol_max)
        lot_std, _, over_std = _hitung_lot(risk_standar, rugi_per_lot, vol_min, vol_step, vol_max)
        flag = "  [⚠️ lot minimum broker melebihi batas risiko]" if (over_kons or over_std) else ""
        sl_lbl = format_price_distance(sl_poin, spec=u_spec, include_mt5_points=True, include_pips=True)
        out.append(f"  - Jarak SL = {sl_lbl} -> Lot 0,5% (konservatif): {lot_kons:.2f} lot | Lot 1,0% (standar): {lot_std:.2f} lot{flag}")

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

    u_spec = get_symbol_unit_spec(symbol=SYMBOL, sym_info=sym)
    return {
        "equity": eq,
        "balance": float(acc.balance),
        "currency": cur,
        "jarak_sl": jarak_sl,
        "jarak_sl_usd": round(jarak_sl, u_spec["digits"]),
        "jarak_sl_mt5_points": price_distance_to_mt5_points(jarak_sl, spec=u_spec),
        "jarak_sl_pips": price_distance_to_pips(jarak_sl, pip_size=u_spec.get("pip_size")),
        "spread": spread,
        "spread_mt5_points": price_distance_to_mt5_points(spread, spec=u_spec),
        "digits": u_spec["digits"],
        "point": u_spec["point"],
        "trade_tick_size": u_spec["trade_tick_size"],
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
        elif cfg.upper() == "XAUUSD" and sym.upper().startswith("XAUUSD"):
            accepted.add(cfg.upper())
    eff_sym = sym or cfg
    if eff_sym.upper() == "XAUUSD" and cfg and cfg.upper().startswith("XAUUSD"):
        eff_sym = cfg
    return eff_sym, accepted


def _query_mt5_symbol_items(fetch_fn, target_sym, accepted_syms):
    """Membaca order/posisi MT5 secara read-only dengan fallback bebas-filter-simbol agar order manual XAUUSDc tidak terlewat."""
    raw = fetch_fn(symbol=target_sym)
    if raw is None and target_sym != SYMBOL:
        raw = fetch_fn(symbol=SYMBOL)
    if raw is None or len(raw) == 0:
        all_raw = fetch_fn()
        if all_raw is not None:
            raw = all_raw
    if raw is None:
        return None, 0
    total_all = len(raw)
    matched = []
    for item in raw:
        s_val = str(
            item.get("symbol") if isinstance(item, dict) else getattr(item, "symbol", "") or ""
        ).strip().upper()
        if not s_val or s_val in accepted_syms:
            matched.append(item)
    return matched, total_all


def _normalize_mt5_pending_order(raw_order, default_symbol=SYMBOL, retrieved_at="", server_offset_hours=None):
    """Menormalisasi objek/dict order MT5 menjadi struktur read-only lengkap dengan metadata & timestamp WIB asli."""
    if isinstance(raw_order, dict):
        ticket = raw_order.get("ticket")
        pos_id = raw_order.get("position_id") or raw_order.get("identifier")
        sym = str(raw_order.get("symbol") or default_symbol or SYMBOL).strip()
        raw_type = raw_order.get("order_type") if raw_order.get("order_type") is not None else raw_order.get("type", 2)
        entry_val = raw_order.get("price_open", raw_order.get("entry_price", raw_order.get("entry", 0.0)))
        sl_val = raw_order.get("sl", 0.0)
        tp_val = raw_order.get("tp", 0.0)
        vol_val = raw_order.get("volume_current", raw_order.get("volume_initial", raw_order.get("volume", 0.0)))
        setup_ts = raw_order.get("time_setup", 0)
        setup_msc = raw_order.get("time_setup_msc", 0)
        done_ts = raw_order.get("time_done", 0)
        done_msc = raw_order.get("time_done_msc", 0)
        magic_val = raw_order.get("magic", 0)
        comment_val = raw_order.get("comment", "")
        pre_created_ts = raw_order.get("order_created_ts", 0.0)
    else:
        ticket = getattr(raw_order, "ticket", None)
        pos_id = getattr(raw_order, "position_id", None)
        sym = str(getattr(raw_order, "symbol", None) or default_symbol or SYMBOL).strip()
        raw_type = getattr(raw_order, "type", 2)
        entry_val = getattr(raw_order, "price_open", getattr(raw_order, "entry_price", 0.0))
        sl_val = getattr(raw_order, "sl", 0.0)
        tp_val = getattr(raw_order, "tp", 0.0)
        vol_val = getattr(raw_order, "volume_current", getattr(raw_order, "volume_initial", getattr(raw_order, "volume", 0.0)))
        setup_ts = getattr(raw_order, "time_setup", 0)
        setup_msc = getattr(raw_order, "time_setup_msc", 0)
        done_ts = getattr(raw_order, "time_done", 0)
        done_msc = getattr(raw_order, "time_done_msc", 0)
        magic_val = getattr(raw_order, "magic", 0)
        comment_val = getattr(raw_order, "comment", "")
        pre_created_ts = 0.0

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
    try:
        pid_val = int(pos_id) if (pos_id is not None and int(pos_id) > 0) else None
    except (TypeError, ValueError):
        pid_val = None
    try:
        magic_i = int(magic_val or 0)
    except (TypeError, ValueError):
        magic_i = 0

    created_utc = float(pre_created_ts) if (isinstance(pre_created_ts, (int, float)) and pre_created_ts > 0) else mt5_ts_to_utc_epoch(setup_ts, setup_msc, server_offset_hours)
    exec_utc = mt5_ts_to_utc_epoch(done_ts, done_msc, server_offset_hours)

    return {
        "ticket": tkt_val,
        "order_id": tkt_val,
        "position_id": pid_val,
        "deal_id": None,
        "mt5_status": "PENDING",
        "order_type": order_type_str,
        "direction": direction_str,
        "symbol": sym,
        "entry_price": entry_f if math.isfinite(entry_f) else 0.0,
        "price_open": entry_f if math.isfinite(entry_f) else 0.0,
        "sl": sl_f if (math.isfinite(sl_f) and sl_f > 0) else 0.0,
        "tp": tp_f if (math.isfinite(tp_f) and tp_f > 0) else 0.0,
        "volume": vol_f if (math.isfinite(vol_f) and vol_f > 0) else 0.0,
        "volume_current": vol_f if (math.isfinite(vol_f) and vol_f > 0) else 0.0,
        "magic": magic_i,
        "is_manual": magic_i == 0,
        "comment": str(comment_val or ""),
        "time_setup": int(setup_ts or 0) if isinstance(setup_ts, (int, float)) else 0,
        "time_setup_msc": int(setup_msc or 0) if isinstance(setup_msc, (int, float)) else 0,
        "order_created_ts": created_utc,
        "order_created_wib": format_mt5_wib(created_utc, short=False),
        "order_created_wib_short": format_mt5_wib(created_utc, short=True),
        "executed_ts": exec_utc,
        "executed_wib": format_mt5_wib(exec_utc, short=False),
        "executed_wib_short": format_mt5_wib(exec_utc, short=True),
        "position_opened_ts": 0.0,
        "position_opened_wib": "",
        "position_opened_wib_short": "",
        "closed_ts": 0.0,
        "closed_wib": "",
        "closed_wib_short": "",
        "retrieved_at": retrieved_at,
    }


def _normalize_mt5_position(raw_pos, default_symbol=SYMBOL, retrieved_at="", server_offset_hours=None, enrich_history=False):
    """Menormalisasi objek/dict posisi aktif MT5 menjadi struktur read-only lengkap dengan metadata & timestamp WIB asli."""
    if isinstance(raw_pos, dict):
        ticket = raw_pos.get("ticket")
        pos_id = raw_pos.get("position_id") or raw_pos.get("identifier") or ticket
        order_id = raw_pos.get("order_id") or raw_pos.get("order")
        deal_id = raw_pos.get("deal_id")
        sym = str(raw_pos.get("symbol") or default_symbol or SYMBOL).strip()
        raw_type = raw_pos.get("direction") if raw_pos.get("direction") is not None else raw_pos.get("type", 0)
        entry_val = raw_pos.get("price_open", raw_pos.get("entry_price", 0.0))
        sl_val = raw_pos.get("sl", 0.0)
        tp_val = raw_pos.get("tp", 0.0)
        vol_val = raw_pos.get("volume", 0.0)
        prof_val = raw_pos.get("profit", 0.0)
        open_ts = raw_pos.get("time", 0)
        open_msc = raw_pos.get("time_msc", 0)
        magic_val = raw_pos.get("magic", 0)
        comment_val = raw_pos.get("comment", "")
        pre_open_utc = raw_pos.get("position_opened_ts", 0.0)
        pre_created_utc = raw_pos.get("order_created_ts", 0.0)
        pre_exec_utc = raw_pos.get("executed_ts", 0.0)
        pre_closed_utc = raw_pos.get("closed_ts", 0.0)
        raw_mt5_st = str(raw_pos.get("mt5_status") or "OPEN").strip().upper()
    else:
        ticket = getattr(raw_pos, "ticket", None)
        pos_id = getattr(raw_pos, "identifier", None) or ticket
        order_id = None
        deal_id = None
        sym = str(getattr(raw_pos, "symbol", None) or default_symbol or SYMBOL).strip()
        raw_type = getattr(raw_pos, "type", 0)
        entry_val = getattr(raw_pos, "price_open", 0.0)
        sl_val = getattr(raw_pos, "sl", 0.0)
        tp_val = getattr(raw_pos, "tp", 0.0)
        vol_val = getattr(raw_pos, "volume", 0.0)
        prof_val = getattr(raw_pos, "profit", 0.0)
        open_ts = getattr(raw_pos, "time", 0)
        open_msc = getattr(raw_pos, "time_msc", 0)
        magic_val = getattr(raw_pos, "magic", 0)
        comment_val = getattr(raw_pos, "comment", "")
        pre_open_utc = 0.0
        pre_created_utc = 0.0
        pre_exec_utc = 0.0
        pre_closed_utc = 0.0
        raw_mt5_st = "OPEN"

    if isinstance(raw_type, str):
        direction_str = "SELL" if "SELL" in raw_type.strip().upper() else "BUY"
    else:
        direction_str = "SELL" if int(raw_type or 0) == getattr(mt5, "ORDER_TYPE_SELL", 1) else "BUY"
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
        prof_f = float(prof_val or 0.0)
    except (TypeError, ValueError):
        prof_f = 0.0
    try:
        tkt_val = int(ticket) if ticket is not None else None
    except (TypeError, ValueError):
        tkt_val = ticket
    try:
        pid_val = int(pos_id) if pos_id is not None else tkt_val
    except (TypeError, ValueError):
        pid_val = tkt_val
    try:
        magic_i = int(magic_val or 0)
    except (TypeError, ValueError):
        magic_i = 0

    open_utc = float(pre_open_utc) if (isinstance(pre_open_utc, (int, float)) and pre_open_utc > 0) else mt5_ts_to_utc_epoch(open_ts, open_msc, server_offset_hours)
    created_utc = float(pre_created_utc) if (isinstance(pre_created_utc, (int, float)) and pre_created_utc > 0) else 0.0
    exec_utc = float(pre_exec_utc) if (isinstance(pre_exec_utc, (int, float)) and pre_exec_utc > 0) else 0.0
    closed_utc = float(pre_closed_utc) if (isinstance(pre_closed_utc, (int, float)) and pre_closed_utc > 0) else 0.0

    if enrich_history and (pid_val or tkt_val):
        hist = fetch_mt5_item_history(
            ticket=tkt_val,
            position_id=pid_val,
            is_order=False,
            server_offset_hours=server_offset_hours,
            allow_live_query=True,
        )
        if hist:
            if not order_id and hist.get("order_id"):
                order_id = hist.get("order_id")
            if not deal_id and hist.get("deal_id"):
                deal_id = hist.get("deal_id")
            if created_utc <= 0 and float(hist.get("order_created_ts") or 0.0) > 0:
                created_utc = float(hist["order_created_ts"])
            if exec_utc <= 0 and float(hist.get("executed_ts") or 0.0) > 0:
                exec_utc = float(hist["executed_ts"])
            if closed_utc <= 0 and float(hist.get("closed_ts") or 0.0) > 0:
                closed_utc = float(hist["closed_ts"])

    return {
        "ticket": tkt_val,
        "position_id": pid_val,
        "identifier": pid_val,
        "order_id": order_id,
        "deal_id": deal_id,
        "mt5_status": raw_mt5_st if raw_mt5_st in ("PENDING", "OPEN", "CLOSED") else "OPEN",
        "type": 0 if direction_str == "BUY" else 1,
        "direction": direction_str,
        "symbol": sym,
        "entry_price": entry_f if math.isfinite(entry_f) else 0.0,
        "price_open": entry_f if math.isfinite(entry_f) else 0.0,
        "sl": sl_f if (math.isfinite(sl_f) and sl_f > 0) else 0.0,
        "tp": tp_f if (math.isfinite(tp_f) and tp_f > 0) else 0.0,
        "volume": vol_f if (math.isfinite(vol_f) and vol_f > 0) else 0.0,
        "profit": prof_f if math.isfinite(prof_f) else 0.0,
        "magic": magic_i,
        "is_manual": magic_i == 0,
        "comment": str(comment_val or ""),
        "time": int(open_ts or 0) if isinstance(open_ts, (int, float)) else 0,
        "time_msc": int(open_msc or 0) if isinstance(open_msc, (int, float)) else 0,
        "order_created_ts": created_utc,
        "order_created_wib": format_mt5_wib(created_utc, short=False),
        "order_created_wib_short": format_mt5_wib(created_utc, short=True),
        "executed_ts": exec_utc,
        "executed_wib": format_mt5_wib(exec_utc, short=False),
        "executed_wib_short": format_mt5_wib(exec_utc, short=True),
        "position_opened_ts": open_utc,
        "position_opened_wib": format_mt5_wib(open_utc, short=False),
        "position_opened_wib_short": format_mt5_wib(open_utc, short=True),
        "closed_ts": closed_utc,
        "closed_wib": format_mt5_wib(closed_utc, short=False),
        "closed_wib_short": format_mt5_wib(closed_utc, short=True),
        "retrieved_at": retrieved_at,
    }


def evaluate_mt5_setup_match(
    item,
    *,
    direction="",
    zone_low=0.0,
    zone_high=0.0,
    plan_entry=0.0,
    zone_id="",
    setup_version=None,
    setup_created_ts=0.0,
    zone_formation_ts=0.0,
    linked_tickets=None,
    signal_state=None,
    accepted_syms=None,
    entry_tol=0.05,
    is_order=False,
    method="",
    server_offset_hours=None,
):
    """
    Klasifikasi read-only yang memisahkan:
    - `mt5_status`: PENDING | OPEN | CLOSED
    - `setup_match`: CONFIRMED | INFERRED | UNRELATED | UNKNOWN
    - `is_existing_position`: True bila posisi/order dibuka sebelum setup dibuat / zona terbentuk.

    Aturan:
    - CONFIRMED hanya jika ada bukti keterkaitan valid (mapping ticket/position_id/order_id/deal_id ke setup,
      comment menyebut zone_id, atau signal_state terhubung) + symbol & arah cocok.
    - Posisi/order yang dibuka sebelum setup_created_ts atau zone_formation_ts TIDAK boleh dianggap
      eksekusi setup tersebut (is_existing_position=True, setup_match=UNRELATED/UNKNOWN).
    - Kecocokan harga, arah, dan waktu saja (tanpa bukti mapping) hanya INFERRED.
    """
    def _get(obj, key, default=None):
        return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)

    sym = str(_get(item, "symbol", "") or "").strip().upper()
    if accepted_syms and sym and sym not in accepted_syms:
        return {"mt5_status": "PENDING" if is_order else "OPEN", "setup_match": "UNRELATED", "is_existing_position": False, "reason": "Symbol tidak cocok", "price_matches": False, "item_ts": 0.0}

    tkt = _get(item, "ticket")
    pid = _get(item, "position_id") or _get(item, "identifier") or tkt
    oid = _get(item, "order_id") or _get(item, "order")
    did = _get(item, "deal_id")
    mt5_st = str(_get(item, "mt5_status") or ("PENDING" if is_order else "OPEN")).strip().upper()
    if mt5_st not in ("PENDING", "OPEN", "CLOSED"):
        mt5_st = "PENDING" if is_order else "OPEN"

    if is_order:
        raw_t = _get(item, "order_type")
        if raw_t is None:
            raw_t = _get(item, "type")
        _, item_dir = _info_tipe_pending(raw_t)
    else:
        raw_t = _get(item, "direction")
        if raw_t is None:
            raw_t = _get(item, "type")
        if isinstance(raw_t, str):
            item_dir = "SELL" if "SELL" in raw_t.upper() else ("BUY" if "BUY" in raw_t.upper() else "")
        else:
            item_dir = "SELL" if int(raw_t or 0) == getattr(mt5, "ORDER_TYPE_SELL", 1) else "BUY"

    want_dir = str(direction or "").strip().upper()
    if want_dir in ("BUY", "SELL") and item_dir and item_dir != want_dir:
        return {"mt5_status": mt5_st, "setup_match": "UNRELATED", "is_existing_position": False, "reason": "Arah tidak cocok", "price_matches": False, "item_ts": 0.0}

    try:
        px = float(_get(item, "price_open") or _get(item, "entry_price") or _get(item, "price") or 0.0)
    except (TypeError, ValueError):
        px = 0.0
    tol = max(float(entry_tol or 0.0), PENDING_ENTRY_TOLERANCE_MIN_PT)
    in_zone = bool(zone_low > 0 and zone_high >= zone_low and (zone_low - tol - 1e-9 <= px <= zone_high + tol + 1e-9))
    near_ent = bool(plan_entry > 0 and abs(px - plan_entry) <= tol + 1e-9)
    price_matches = bool(px > 0 and (in_zone or near_ent))

    # Ambil timestamp UTC asli MT5 (jangan mengarang jika tidak ada)
    if is_order:
        item_ts = float(_get(item, "order_created_ts") or 0.0)
        if item_ts <= 0:
            item_ts = mt5_ts_to_utc_epoch(_get(item, "time_setup", 0), _get(item, "time_setup_msc", 0), server_offset_hours)
    else:
        item_ts = float(_get(item, "position_opened_ts") or _get(item, "executed_ts") or 0.0)
        if item_ts <= 0:
            item_ts = mt5_ts_to_utc_epoch(_get(item, "time", 0), _get(item, "time_msc", 0), server_offset_hours)

    # Bukti keterkaitan eksplisit (CONFIRMED)
    linked_set = set()
    for lt in (linked_tickets or ()):
        if lt is not None:
            try:
                linked_set.add(int(lt))
            except (TypeError, ValueError):
                linked_set.add(str(lt))
    item_ids = set()
    for cid in (tkt, pid, oid, did):
        if cid is not None:
            try:
                item_ids.add(int(cid))
            except (TypeError, ValueError):
                item_ids.add(str(cid))

    tkt_linked = bool(linked_set and (item_ids & linked_set))
    comment_str = str(_get(item, "comment", "") or "").strip()
    magic_val = 0
    try:
        magic_val = int(_get(item, "magic", 0) or 0)
    except (TypeError, ValueError):
        magic_val = 0
    zid = str(zone_id or "").strip()
    zid_short = zid[-8:] if len(zid) >= 8 else zid
    comment_linked = bool(
        comment_str and (
            (zid and zid.lower() in comment_str.lower())
            or (zid_short and len(zid_short) >= 8 and zid_short.lower() in comment_str.lower())
            or (method and f"{str(method).lower()}:" in comment_str.lower())
        )
    )
    sig_linked = False
    if isinstance(signal_state, dict):
        s_obj = signal_state.get("sinyal") if isinstance(signal_state.get("sinyal"), dict) else signal_state
        if isinstance(s_obj, dict):
            s_tkt = s_obj.get("ticket")
            s_zid = str(s_obj.get("zone_id") or "")
            s_magic = int(s_obj.get("magic") or 0)
            if s_tkt and (s_tkt in item_ids) and (not s_zid or not zid or s_zid == zid):
                sig_linked = True
            elif magic_val > 0 and s_magic > 0 and magic_val == s_magic and str(s_obj.get("arah") or "").upper() == item_dir:
                sig_linked = True

    has_explicit_link = bool(tkt_linked or comment_linked or sig_linked)

    z_ts = float(zone_formation_ts or 0.0)
    s_ts = float(setup_created_ts or 0.0)
    ref_ts = max(z_ts, s_ts) if (z_ts > 0 or s_ts > 0) else 0.0
    opened_before_zone = bool(item_ts > 0 and z_ts > 0 and item_ts < (z_ts - 60.0))
    opened_before_setup = bool(item_ts > 0 and s_ts > 0 and item_ts < (s_ts - 60.0))
    is_existing = bool(opened_before_zone or opened_before_setup)

    if not price_matches and not has_explicit_link:
        return {"mt5_status": mt5_st, "setup_match": "UNRELATED", "is_existing_position": is_existing, "reason": "Harga di luar zona & tidak ada mapping setup", "price_matches": False, "item_ts": item_ts}

    if has_explicit_link and not opened_before_zone:
        return {
            "mt5_status": mt5_st,
            "setup_match": "CONFIRMED",
            "is_existing_position": False,
            "reason": "Mapping order/deal/position ke setup terkonfirmasi",
            "price_matches": price_matches,
            "item_ts": item_ts,
        }

    if is_existing:
        rsn_ex = (
            "Dibuka sebelum zona terbentuk (EXISTING POSITION)"
            if opened_before_zone
            else "Dibuka sebelum setup version dibuat (EXISTING POSITION)"
        )
        return {
            "mt5_status": mt5_st,
            "setup_match": "UNRELATED",
            "is_existing_position": True,
            "reason": rsn_ex,
            "price_matches": price_matches,
            "item_ts": item_ts,
        }

    if item_ts <= 0 or ref_ts <= 0:
        return {
            "mt5_status": mt5_st,
            "setup_match": "UNKNOWN",
            "is_existing_position": True if not is_order else False,
            "reason": "Waktu pembukaan MT5 atau waktu pembentukan setup tidak lengkap",
            "price_matches": price_matches,
            "item_ts": item_ts,
        }

    return {
        "mt5_status": mt5_st,
        "setup_match": "INFERRED",
        "is_existing_position": False,
        "reason": "Harga, arah, dan waktu cocok namun tanpa mapping order/setup (manual)",
        "price_matches": price_matches,
        "item_ts": item_ts,
    }


def format_tracked_position_compact(pos_info, digits=2):
    """
    Format ringkas 1 baris untuk terminal:
    `#60927598 SELL @3624.51 | OPEN 23-Nov 10:25 WIB | EXISTING | MATCH:UNKNOWN`
    Menggunakan timestamp MT5 sebenarnya (bukan contoh).
    """
    if not isinstance(pos_info, dict):
        return ""
    tkt = pos_info.get("ticket")
    if not tkt:
        return ""
    arah = str(pos_info.get("direction") or "").strip().upper()
    try:
        px = float(pos_info.get("open_price") or pos_info.get("price_open") or pos_info.get("entry_price") or 0.0)
    except (TypeError, ValueError):
        px = 0.0
    px_part = f"@{px:.{digits}f}" if px > 0 else ""

    mt5_st = str(pos_info.get("mt5_status") or "OPEN").strip().upper()
    if mt5_st not in ("PENDING", "OPEN", "CLOSED"):
        mt5_st = "OPEN"

    if mt5_st == "CLOSED":
        t_short = str(pos_info.get("closed_wib_short") or format_mt5_wib(pos_info.get("closed_ts"), short=True) or "").strip()
    elif mt5_st == "PENDING":
        t_short = str(pos_info.get("order_created_wib_short") or format_mt5_wib(pos_info.get("order_created_ts"), short=True) or "").strip()
    else:
        t_short = str(
            pos_info.get("position_opened_wib_short")
            or format_mt5_wib(pos_info.get("position_opened_ts"), short=True)
            or pos_info.get("executed_wib_short")
            or format_mt5_wib(pos_info.get("executed_ts"), short=True)
            or ""
        ).strip()

    st_time_part = f"{mt5_st} {t_short}".strip() if t_short else f"{mt5_st} TIME:UNKNOWN"
    raw_match = str(pos_info.get("setup_match") or pos_info.get("match_status") or "UNKNOWN").strip().upper()
    if raw_match == "VERIFIED":
        raw_match = "CONFIRMED"
    elif raw_match == "UNVERIFIED":
        raw_match = "UNKNOWN"
    if raw_match not in ("CONFIRMED", "INFERRED", "UNRELATED", "UNKNOWN"):
        raw_match = "UNKNOWN"

    parts = [f"#{tkt} {arah} {px_part}".strip(), st_time_part]
    if pos_info.get("is_existing_position") or raw_match in ("UNRELATED", "UNKNOWN"):
        parts.append("EXISTING")
    parts.append(f"MATCH:{raw_match}")
    return " | ".join(parts)


def fetch_mt5_pending_orders_snapshot(
    symbol=SYMBOL,
    orders_override=None,
    allow_live_query=True,
    positions_override=None,
):
    """
    Mengambil snapshot pending order dan posisi aktif terkini dari MT5 secara READ-ONLY
    menggunakan `orders_get()` dan `positions_get()` (termasuk order/posisi manual tanpa magic number).
    Tidak pernah memanggil order_send() atau memodifikasi/membatalkan order.
    Jika query MT5 gagal atau tidak tersedia, mengembalikan query_ok=False dan status='UNKNOWN'.
    """
    from modules.utils import format_wib_timestamp

    target_sym, accepted_syms = _acceptable_trade_symbols(symbol)
    now_wib = format_wib_timestamp(default_now=True)
    now_ts = time.time()

    if isinstance(orders_override, dict) and "query_ok" in orders_override and "orders" in orders_override:
        snap_copy = dict(orders_override)
        if positions_override is not None and isinstance(positions_override, (list, tuple)):
            norm_pos_ov = []
            for raw_p in positions_override:
                np_item = _normalize_mt5_position(raw_p, default_symbol=target_sym, retrieved_at=now_wib)
                if np_item["symbol"].upper() in accepted_syms:
                    norm_pos_ov.append(np_item)
            snap_copy["positions"] = norm_pos_ov
            snap_copy["positions_query_ok"] = True
            snap_copy["symbol_positions_count"] = len(norm_pos_ov)
            snap_copy["positions_without_sl"] = [p["ticket"] for p in norm_pos_ov if p["sl"] <= 0 and p.get("ticket") is not None]
        else:
            snap_copy.setdefault("positions", [])
            snap_copy.setdefault("positions_query_ok", False)
            snap_copy.setdefault("symbol_positions_count", len(snap_copy["positions"]))
            snap_copy.setdefault("positions_without_sl", [p.get("ticket") for p in snap_copy["positions"] if float(p.get("sl") or 0.0) <= 0 and p.get("ticket") is not None])
        snap_copy.setdefault("orders_without_sl", [o.get("ticket") for o in (snap_copy.get("orders") or []) if float(o.get("sl") or 0.0) <= 0 and o.get("ticket") is not None])
        return snap_copy

    u_spec = get_symbol_unit_spec(symbol=target_sym)
    digits = u_spec["digits"]
    point = u_spec["point"]
    tick_size = u_spec["trade_tick_size"]

    if orders_override is not None and isinstance(orders_override, (list, tuple)):
        try:
            if pastikan_mt5_aktif():
                sym_info = mt5.symbol_info(target_sym) or mt5.symbol_info(SYMBOL)
                if sym_info is not None:
                    u_spec = get_symbol_unit_spec(symbol=target_sym, sym_info=sym_info)
                    digits = u_spec["digits"]
                    point = u_spec["point"]
                    tick_size = u_spec["trade_tick_size"]
        except Exception:
            pass
        entry_tol = round(max(10.0 * tick_size, 2.0 * point, PENDING_ENTRY_TOLERANCE_MIN_PT), max(digits, 3))
        level_tol = round(max(10.0 * tick_size, 2.0 * point, PENDING_LEVEL_TOLERANCE_MIN_PT), max(digits, 3))
        norm_orders = []
        seen_ord_tkts = set()
        for raw_o in orders_override:
            no = _normalize_mt5_pending_order(raw_o, default_symbol=target_sym, retrieved_at=now_wib)
            if no["symbol"].upper() in accepted_syms:
                t_o = no.get("ticket")
                if t_o is not None:
                    if t_o in seen_ord_tkts:
                        continue
                    seen_ord_tkts.add(t_o)
                norm_orders.append(no)
        norm_positions = []
        seen_pos_tkts = set()
        pos_q_ok = False
        if positions_override is not None and isinstance(positions_override, (list, tuple)):
            pos_q_ok = True
            for raw_p in positions_override:
                np_item = _normalize_mt5_position(raw_p, default_symbol=target_sym, retrieved_at=now_wib)
                if np_item["symbol"].upper() in accepted_syms:
                    t_p = np_item.get("ticket")
                    if t_p is not None:
                        if t_p in seen_pos_tkts:
                            continue
                        seen_pos_tkts.add(t_p)
                    norm_positions.append(np_item)
        pos_no_sl_list = [p["ticket"] for p in norm_positions if p["sl"] <= 0 and p.get("ticket") is not None]
        pos_no_sl_set = set(pos_no_sl_list)
        ord_no_sl_list = [o["ticket"] for o in norm_orders if o["sl"] <= 0 and o.get("ticket") is not None and o.get("ticket") not in pos_no_sl_set]
        return {
            "query_ok": True,
            "positions_query_ok": pos_q_ok,
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
            "orders_without_sl": ord_no_sl_list,
            "symbol_positions_count": len(norm_positions),
            "positions": norm_positions,
            "positions_without_sl": pos_no_sl_list,
        }

    if not allow_live_query:
        return {
            "query_ok": False,
            "positions_query_ok": False,
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
            "orders_without_sl": [],
            "symbol_positions_count": 0,
            "positions": [],
            "positions_without_sl": [],
        }

    try:
        if not pastikan_mt5_aktif():
            return {
                "query_ok": False,
                "positions_query_ok": False,
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
                "orders_without_sl": [],
                "symbol_positions_count": 0,
                "positions": [],
                "positions_without_sl": [],
            }
        with MT5_IPC_LOCK:
            sym_info = mt5.symbol_info(target_sym) or mt5.symbol_info(SYMBOL)
            if sym_info is not None:
                u_spec = get_symbol_unit_spec(symbol=target_sym, sym_info=sym_info)
                digits = u_spec["digits"]
                point = u_spec["point"]
                tick_size = u_spec["trade_tick_size"]

            raw_orders, total_acc_orders = _query_mt5_symbol_items(mt5.orders_get, target_sym, accepted_syms)
        if raw_orders is None:
            return {
                "query_ok": False,
                "positions_query_ok": False,
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
                "orders_without_sl": [],
                "symbol_positions_count": 0,
                "positions": [],
                "positions_without_sl": [],
            }

        if positions_override is not None and isinstance(positions_override, (list, tuple)):
            raw_positions = list(positions_override)
            pos_q_ok = True
        else:
            with MT5_IPC_LOCK:
                raw_positions, _ = _query_mt5_symbol_items(mt5.positions_get, target_sym, accepted_syms)
            pos_q_ok = raw_positions is not None
            raw_positions = raw_positions or []

        server_off_h = get_mt5_server_utc_offset_hours(allow_live_query=True)
        entry_tol = round(max(10.0 * tick_size, 2.0 * point, PENDING_ENTRY_TOLERANCE_MIN_PT), max(digits, 3))
        level_tol = round(max(10.0 * tick_size, 2.0 * point, PENDING_LEVEL_TOLERANCE_MIN_PT), max(digits, 3))
        norm_orders = []
        seen_ord_tkts = set()
        for raw_o in raw_orders:
            no = _normalize_mt5_pending_order(raw_o, default_symbol=target_sym, retrieved_at=now_wib, server_offset_hours=server_off_h)
            if no["symbol"].upper() in accepted_syms:
                t_o = no.get("ticket")
                if t_o is not None:
                    if t_o in seen_ord_tkts:
                        continue
                    seen_ord_tkts.add(t_o)
                norm_orders.append(no)
        norm_positions = []
        seen_pos_tkts = set()
        for raw_p in raw_positions:
            np_item = _normalize_mt5_position(
                raw_p,
                default_symbol=target_sym,
                retrieved_at=now_wib,
                server_offset_hours=server_off_h,
                enrich_history=True,
            )
            if np_item["symbol"].upper() in accepted_syms:
                t_p = np_item.get("ticket")
                if t_p is not None:
                    if t_p in seen_pos_tkts:
                        continue
                    seen_pos_tkts.add(t_p)
                norm_positions.append(np_item)
        pos_no_sl_list = [p["ticket"] for p in norm_positions if p["sl"] <= 0 and p.get("ticket") is not None]
        pos_no_sl_set = set(pos_no_sl_list)
        ord_no_sl_list = [o["ticket"] for o in norm_orders if o["sl"] <= 0 and o.get("ticket") is not None and o.get("ticket") not in pos_no_sl_set]
        return {
            "query_ok": True,
            "positions_query_ok": pos_q_ok,
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
            "total_account_orders": total_acc_orders,
            "symbol_orders_count": len(norm_orders),
            "orders": norm_orders,
            "orders_without_sl": ord_no_sl_list,
            "symbol_positions_count": len(norm_positions),
            "positions": norm_positions,
            "positions_without_sl": pos_no_sl_list,
        }
    except Exception as exc:
        _catat_error("fetch_mt5_pending_orders_snapshot", exc)
        return {
            "query_ok": False,
            "positions_query_ok": False,
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
            "orders_without_sl": [],
            "symbol_positions_count": 0,
            "positions": [],
            "positions_without_sl": [],
        }


def fetch_mt5_live_state(symbol=SYMBOL, orders_override=None, positions_override=None, allow_live_query=True):
    """Alias terpadu read-only untuk membaca pending orders + active positions MT5 di semua command."""
    return fetch_mt5_pending_orders_snapshot(
        symbol=symbol,
        orders_override=orders_override,
        positions_override=positions_override,
        allow_live_query=allow_live_query,
    )


def find_pending_orders_on_invalid_zones(
    invalid_zones,
    mt5_snapshot=None,
    symbol=SYMBOL,
    default_source=None,
    active_valid_zones=None,
):
    """
    Mendeteksi secara READ-ONLY apakah masih ada pending order aktif di MT5 yang benar-benar berada pada zona INVALID/BROKEN.
    Memisahkan order manual yang tidak cocok dengan zona scanner (unmatched) atau berada di zona valid aktif/radar
    dari order yang benar-benar berada di zona broken/invalidated.
    Dideduplikasi per ticket (1 ticket muncul tepat sekali meski menyentuh beberapa zona invalid SMC/SNR) —
    TIDAK pernah membatalkan/mengubah order otomatis.
    """
    if not invalid_zones:
        return []
    target_sym, accepted_syms = _acceptable_trade_symbols(symbol)
    snap = (
        mt5_snapshot
        if (isinstance(mt5_snapshot, dict) and "query_ok" in mt5_snapshot and "orders" in mt5_snapshot)
        else fetch_mt5_pending_orders_snapshot(symbol=target_sym, orders_override=mt5_snapshot)
    )
    if not snap.get("query_ok"):
        return []
    orders = list(snap.get("orders") or [])
    if not orders:
        return []
    entry_tol = float(snap.get("entry_tolerance") or PENDING_ENTRY_TOLERANCE_MIN_PT)
    digits = int(snap.get("digits") or 2)

    valid_zone_ranges = []
    for vz in (active_valid_zones or []):
        vzd = vz.to_dict() if hasattr(vz, "to_dict") else (vz if isinstance(vz, dict) else None)
        if not isinstance(vzd, dict):
            continue
        if vzd.get("is_valid") is False or vzd.get("is_broken") is True or vzd.get("is_invalidated") is True:
            continue
        v_st = str(vzd.get("lifecycle_status") or vzd.get("status") or "ACTIVE").upper()
        v_fr = str(vzd.get("freshness") or "FRESH").upper()
        if v_st in ("INVALIDATED", "BROKEN", "MITIGATED", "EXPIRED", "REJECTED") or v_fr in ("INVALIDATED", "MITIGATED", "EXPIRED"):
            continue
        v_lo = float(vzd.get("price_low", vzd.get("low", 0.0)) or 0.0)
        v_hi = float(vzd.get("price_high", vzd.get("high", 0.0)) or 0.0)
        if v_lo <= 0 or v_hi <= 0:
            continue
        if v_lo > v_hi:
            v_lo, v_hi = v_hi, v_lo
        v_type = str(vzd.get("zone_type") or vzd.get("type") or "").upper()
        v_dir = "BUY" if v_type in ("DEMAND", "SUPPORT") else ("SELL" if v_type in ("SUPPLY", "RESISTANCE") else "")
        valid_zone_ranges.append((v_dir, v_lo, v_hi))

    by_ticket = {}
    ticket_order = []
    for z in invalid_zones:
        zd = z.to_dict() if hasattr(z, "to_dict") else (z if isinstance(z, dict) else None)
        if not isinstance(zd, dict):
            continue
        z_st = str(zd.get("lifecycle_status") or zd.get("status") or "").upper()
        z_fr = str(zd.get("freshness") or "").upper()
        is_genuinely_broken = bool(
            zd.get("is_broken") is True
            or zd.get("is_invalidated") is True
            or z_st in ("INVALIDATED", "BROKEN")
            or z_fr == "INVALIDATED"
        )
        if not is_genuinely_broken:
            continue
        if z_st in ("RECLAIMED", "MITIGATED", "EXPIRED", "REJECTED", "RADAR", "ACTIVE", "TOUCHED", "ZONE_TOUCHED", "APPROACHING", "UNCONFIRMED"):
            continue
        rej_low = str(zd.get("rejection_reason") or "").lower()
        if any(
            kw in rej_low
            for kw in (
                "tidak ada fvg",
                "displacement tidak memadai",
                "bos tipis",
                "bos terlalu jauh",
                "tidak ada candle impuls",
                "zona terlalu lebar",
                "zona terlalu sempit",
                "zona sudah dimitigasi",
                "gagal hard validation",
                "skor ",
            )
        ):
            continue
        bo_info = zd.get("breakout_info") if isinstance(zd.get("breakout_info"), dict) else {}
        if (
            zd.get("is_distant_historical") is True
            or zd.get("lifecycle_is_historical") is True
            or bo_info.get("is_historical_break") is True
            or int(zd.get("lifecycle_bars_ago") or bo_info.get("bars_ago") or 0) > 6
        ):
            continue
        z_lo = float(zd.get("price_low", zd.get("low", 0.0)) or 0.0)
        z_hi = float(zd.get("price_high", zd.get("high", 0.0)) or 0.0)
        if z_lo <= 0 or z_hi <= 0:
            continue
        if z_lo > z_hi:
            z_lo, z_hi = z_hi, z_lo
        z_type = str(zd.get("zone_type") or zd.get("type") or "").upper()
        want_dir = "BUY" if z_type in ("DEMAND", "SUPPORT") else ("SELL" if z_type in ("SUPPLY", "RESISTANCE") else "")
        z_tf = str(zd.get("timeframe") or "").strip()
        z_id = str(zd.get("zone_id") or f"{z_tf}_{z_type}_{z_lo:.2f}_{z_hi:.2f}")
        z_src = str(zd.get("source") or default_source or "").strip().upper()
        if not z_src:
            if z_type in ("SUPPORT", "RESISTANCE") or z_id.upper().startswith("SNR_"):
                z_src = "SNR"
            elif z_type in ("DEMAND", "SUPPLY") or z_id.upper().startswith(("ZONE_", "TF_")):
                z_src = "SMC"
        z_lbl = f"{z_tf} {z_type} {z_lo:.2f}-{z_hi:.2f}".strip()
        if z_src:
            z_lbl = f"[{z_src}] {z_lbl}"
        for ord_item in orders:
            if str(ord_item.get("symbol") or "").upper() not in accepted_syms:
                continue
            o_dir = str(ord_item.get("direction") or "").upper()
            if want_dir and o_dir and o_dir != want_dir:
                continue
            o_ent = float(ord_item.get("entry_price") or 0.0)
            if o_ent <= 0:
                continue
            # Jika order berada pada/dekat zona aktif/radar valid searah, jangan klasifikasikan sebagai zona invalid
            if any(
                (not v_d or not o_dir or v_d == o_dir) and (v_l - entry_tol - 1e-9) <= o_ent <= (v_h + entry_tol + 1e-9)
                for v_d, v_l, v_h in valid_zone_ranges
            ):
                continue
            if (z_lo - entry_tol - 1e-9) <= o_ent <= (z_hi + entry_tol + 1e-9):
                tkt = ord_item.get("ticket")
                tkt_key = tkt if tkt is not None else f"{ord_item.get('order_type', 'PENDING')}@{o_ent:.{digits}f}"
                entry_rec = by_ticket.get(tkt_key)
                if entry_rec is None:
                    o_sl = float(ord_item.get("sl") or 0.0)
                    entry_rec = {
                        "ticket": tkt,
                        "order_type": ord_item.get("order_type", "PENDING"),
                        "direction": o_dir,
                        "entry_price": o_ent,
                        "sl": o_sl,
                        "sl_missing": o_sl <= 0,
                        "zone_id": z_id,
                        "zone_label": z_lbl,
                        "zone_range": (z_lo, z_hi),
                        "zone_ids": [z_id],
                        "zone_labels": [z_lbl],
                        "sources": [z_src] if z_src else [],
                    }
                    by_ticket[tkt_key] = entry_rec
                    ticket_order.append(tkt_key)
                else:
                    if z_id not in entry_rec["zone_ids"]:
                        entry_rec["zone_ids"].append(z_id)
                    if z_lbl not in entry_rec["zone_labels"]:
                        entry_rec["zone_labels"].append(z_lbl)
                    if z_src and z_src not in entry_rec["sources"]:
                        entry_rec["sources"].append(z_src)

    warnings_out = []
    for tkt_key in ticket_order:
        rec = by_ticket[tkt_key]
        tkt = rec["ticket"]
        o_ent = rec["entry_price"]
        o_sl = rec["sl"]
        z_labels = rec["zone_labels"]
        z_cnt = len(z_labels)
        z_desc = z_labels[0] if z_cnt == 1 else f"{z_labels[0]} (+{z_cnt - 1} zona)"
        src_list = sorted(rec.get("sources") or [])
        src_scope = "+".join(src_list) if src_list else ""
        sl_warn = " [SL MISSING — HIGH RISK]" if rec["sl_missing"] else f" | SL={o_sl:.{digits}f}"
        rec["invalid_zone_count"] = z_cnt
        rec["source_scope"] = src_scope
        rec["message"] = (
            f"[WARNING] Pending #{tkt} ({rec['order_type']} @ {o_ent:.{digits}f}{sl_warn}) "
            f"aktif di zona INVALID ({z_desc}); tinjau manual di MT5."
        )
        warnings_out.append(rec)
    return warnings_out


def _format_invalid_zone_scope_tag(invalid_zone_scope, inv_by_tkt=None):
    """Susun label cakupan hitungan invalid order (SMC / SNR / SMC+SNR gabungan) secara eksplisit."""
    if isinstance(invalid_zone_scope, dict):
        scope_name = str(invalid_zone_scope.get("scope") or "SMC+SNR").strip().upper()
        smc_c = int(invalid_zone_scope.get("smc", 0) or 0)
        snr_c = int(invalid_zone_scope.get("snr", 0) or 0)
        comb_c = int(invalid_zone_scope.get("combined", len(inv_by_tkt or {})) or 0)
        if scope_name in ("SMC+SNR", "COMBINED", "UNIFIED"):
            return f" [Gabungan: SMC={smc_c}, SNR={snr_c}, Unik={comb_c}]"
        return f" [{scope_name}]"
    if isinstance(invalid_zone_scope, str) and invalid_zone_scope.strip():
        return f" [{invalid_zone_scope.strip().upper()}]"
    if isinstance(inv_by_tkt, dict) and inv_by_tkt:
        all_srcs = sorted({s for iw in inv_by_tkt.values() for s in (iw.get("sources") or []) if s})
        if all_srcs:
            return f" [{'+'.join(all_srcs)}]"
    return ""


def summarize_mt5_risk_warnings(
    mt5_snapshot,
    invalid_zone_warnings=None,
    matched_recons=None,
    spread_warning=None,
    invalid_zone_scope=None,
):
    """
    Kelompokkan dan deduplikasi seluruh peringatan risiko MT5 berdasarkan ticket & jenis risiko:
    - Satu ticket hanya muncul sekali pada breakdown detail, meski terkait beberapa risiko/zona invalid.
    - Headline ringkas anti-spam (contoh: `RISK: 5 Positions + 17 Pending Without SL [HIGH RISK]`).
    - Detail ticket hanya untuk matched active setup atau saat diminta (mode detail / perubahan status).
    """
    snap = mt5_snapshot if isinstance(mt5_snapshot, dict) else {}
    digits = int(snap.get("digits") or 2)
    orders = list(snap.get("orders") or [])
    positions = list(snap.get("positions") or [])

    ord_by_tkt = {o.get("ticket"): o for o in orders if o.get("ticket") is not None}
    pos_by_tkt = {p.get("ticket"): p for p in positions if p.get("ticket") is not None}

    pos_no_sl = []
    seen_p = set()
    for t in (snap.get("positions_without_sl") or []):
        if t is not None and t not in seen_p:
            seen_p.add(t)
            pos_no_sl.append(t)

    ord_no_sl = []
    seen_o = set()
    for t in (snap.get("orders_without_sl") or []):
        if t is not None and t not in seen_o and t not in seen_p:
            seen_o.add(t)
            ord_no_sl.append(t)

    inv_by_tkt = {}
    inv_tkt_order = []
    for iw in (invalid_zone_warnings or []):
        if not isinstance(iw, dict):
            continue
        tkt = iw.get("ticket")
        key = tkt if tkt is not None else str(iw.get("message") or len(inv_tkt_order))
        if key not in inv_by_tkt:
            inv_by_tkt[key] = {
                "ticket": tkt,
                "order_type": str(iw.get("order_type") or "PENDING"),
                "entry_price": float(iw.get("entry_price") or 0.0),
                "sl_missing": bool(iw.get("sl_missing")),
                "zone_labels": list(iw.get("zone_labels") or ([iw.get("zone_label")] if iw.get("zone_label") else [])),
                "sources": list(iw.get("sources") or []),
            }
            inv_tkt_order.append(key)
        else:
            for zl in (iw.get("zone_labels") or ([iw.get("zone_label")] if iw.get("zone_label") else [])):
                if zl and zl not in inv_by_tkt[key]["zone_labels"]:
                    inv_by_tkt[key]["zone_labels"].append(zl)
            for src_s in (iw.get("sources") or []):
                if src_s and src_s not in inv_by_tkt[key]["sources"]:
                    inv_by_tkt[key]["sources"].append(src_s)

    matched_issues = {}
    matched_issue_order = []
    for rec in (matched_recons or []):
        if not isinstance(rec, dict):
            continue
        for mo in (rec.get("matched_orders") or []):
            if not isinstance(mo, dict):
                continue
            tkt = mo.get("ticket")
            if tkt is None:
                continue
            tags = []
            if mo.get("sl_missing"):
                tags.append("SL MISSING")
            elif not mo.get("sl_match", True):
                tags.append("SL MISMATCH")
            if not mo.get("tp_match", True) and not mo.get("tp_missing"):
                tags.append("TP MISMATCH")
            if tags:
                if tkt not in matched_issues:
                    matched_issues[tkt] = []
                    matched_issue_order.append(tkt)
                for tg in tags:
                    if tg not in matched_issues[tkt]:
                        matched_issues[tkt].append(tg)

    # Kelompokkan per ticket unik (satu ticket tepat 1 entri detail)
    by_ticket = {}
    ticket_seq = []

    def _ensure_tkt(tkt, kind, otype="", ent=0.0):
        k = (kind, tkt)
        if k not in by_ticket:
            by_ticket[k] = {
                "ticket": tkt,
                "kind": kind,
                "order_type": otype,
                "entry_price": ent,
                "risks": [],
            }
            ticket_seq.append(k)
        return by_ticket[k]

    for t in pos_no_sl:
        p = pos_by_tkt.get(t) or {}
        rec = _ensure_tkt(t, "POSITION", str(p.get("direction") or "POS"), float(p.get("entry_price") or p.get("price_open") or 0.0))
        if "WITHOUT SL" not in rec["risks"]:
            rec["risks"].append("WITHOUT SL")

    for t in ord_no_sl:
        o = ord_by_tkt.get(t) or {}
        rec = _ensure_tkt(t, "PENDING", str(o.get("order_type") or "PENDING"), float(o.get("entry_price") or 0.0))
        if "WITHOUT SL" not in rec["risks"]:
            rec["risks"].append("WITHOUT SL")

    for key in inv_tkt_order:
        iw = inv_by_tkt[key]
        t = iw["ticket"]
        if t is None:
            continue
        rec = _ensure_tkt(t, "PENDING", iw["order_type"], iw["entry_price"])
        if iw["sl_missing"] and "WITHOUT SL" not in rec["risks"]:
            rec["risks"].append("WITHOUT SL")
        zls = iw["zone_labels"]
        srcs_t = sorted({s for s in (iw.get("sources") or []) if s})
        src_pfx = f"[{'+'.join(srcs_t)}] " if srcs_t else ""
        inv_tag = f"INVALID ZONE {src_pfx}({zls[0]}{f' +{len(zls)-1}' if len(zls) > 1 else ''})" if zls else f"INVALID ZONE {src_pfx}".strip()
        if inv_tag not in rec["risks"]:
            rec["risks"].append(inv_tag)

    for t in matched_issue_order:
        o = ord_by_tkt.get(t) or {}
        rec = _ensure_tkt(t, "PENDING", str(o.get("order_type") or "PENDING"), float(o.get("entry_price") or 0.0))
        for tg in matched_issues[t]:
            m_tag = f"MATCHED {tg}"
            if m_tag not in rec["risks"]:
                rec["risks"].append(m_tag)

    nosl_parts = []
    if len(pos_no_sl) > 0:
        nosl_parts.append(f"{len(pos_no_sl)} Position{'s' if len(pos_no_sl) != 1 else ''}")
    if len(ord_no_sl) > 0:
        nosl_parts.append(f"{len(ord_no_sl)} Pending")
    without_sl_text = f"{' + '.join(nosl_parts)} Without SL" if nosl_parts else ""

    inv_cnt = len(inv_tkt_order)
    scope_tag = _format_invalid_zone_scope_tag(invalid_zone_scope, inv_by_tkt=inv_by_tkt)
    invalid_zone_text = f"{inv_cnt} Pending on INVALID Zone{scope_tag}" if inv_cnt > 0 else ""

    matched_issue_parts = []
    for t in matched_issue_order[:2]:
        matched_issue_parts.append(f"Matched #{t} {'/'.join(matched_issues[t])}")
    if len(matched_issue_order) > 2:
        matched_issue_parts.append(f"+{len(matched_issue_order) - 2} matched")
    matched_issue_text = "; ".join(matched_issue_parts)

    headline_parts = [p for p in (without_sl_text, invalid_zone_text, matched_issue_text) if p]
    has_high_risk = bool(headline_parts)
    if spread_warning:
        headline_parts.append(str(spread_warning).strip())
    if headline_parts:
        hr_suffix = " [HIGH RISK]" if has_high_risk else ""
        headline = f"RISK: {' | '.join(headline_parts)}{hr_suffix}"
    else:
        headline = ""

    detail_lines = []
    for k in ticket_seq:
        r = by_ticket[k]
        ent_part = f" @ {r['entry_price']:.{digits}f}" if r["entry_price"] > 0 else ""
        detail_lines.append(
            f"  • #{r['ticket']} [{r['kind']} {r['order_type']}{ent_part}] -> {', '.join(r['risks'])}"
        )

    signature = (
        tuple(sorted(str(x) for x in pos_no_sl)),
        tuple(sorted(str(x) for x in ord_no_sl)),
        tuple(sorted(str(x) for x in inv_tkt_order)),
        tuple((str(t), tuple(matched_issues[t])) for t in sorted(matched_issues.keys(), key=str)),
        str(spread_warning or ""),
    )

    return {
        "has_risk": bool(headline_parts),
        "has_high_risk": has_high_risk,
        "positions_without_sl_count": len(pos_no_sl),
        "pending_without_sl_count": len(ord_no_sl),
        "invalid_zone_tickets_count": inv_cnt,
        "matched_issue_tickets": list(matched_issue_order),
        "without_sl_text": without_sl_text,
        "invalid_zone_text": invalid_zone_text,
        "matched_issue_text": matched_issue_text,
        "headline": headline,
        "detail_lines": detail_lines,
        "by_ticket": by_ticket,
        "signature": signature,
    }


def format_pending_compact_status(plan_status, recon, order_type="", entry_price=0.0):
    """
    Memformat status ringkas pending plan yang memisahkan plan_status (READY/PROVISIONAL/WAIT/INVALID/EXPIRED)
    dari broker_order_status (MT5 PLACED / NOT PLACED / NO MATCH | OTHER ORDERS / SL/TP MISMATCH / UNKNOWN).
    Detail ticket hanya ditampilkan untuk order yang cocok dengan active setup atau mismatch langsung pada setup.
    """
    p_st = str(plan_status or "WAIT").strip().upper()
    if p_st.startswith("READY"):
        p_st = "READY"
    elif p_st.startswith("PROVISIONAL"):
        p_st = "PROVISIONAL"
    elif p_st.startswith("EXPIRED"):
        p_st = "EXPIRED"
    elif p_st.startswith("INVALID"):
        p_st = "INVALID"
    elif p_st not in ("READY", "PROVISIONAL", "WAIT", "INVALID", "EXPIRED"):
        p_st = "WAIT"

    rec = recon if isinstance(recon, dict) else {}
    b_st = str(rec.get("broker_order_status") or "UNKNOWN").strip().upper()
    tot_sym = int(rec.get("symbol_orders_count") or 0)

    if p_st not in ("READY", "PROVISIONAL"):
        if b_st == "OTHER_ORDER_EXISTS" or (rec.get("query_ok") and tot_sym > 0):
            return f"{p_st} | MT5: NO MATCH | OTHER ORDERS: {tot_sym}"
        if b_st == "NOT_PLACED":
            return f"{p_st} | MT5: NOT PLACED"
        return p_st

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
        tkt_tag = f"#{tickets[0]} (+{len(tickets) - 1})"
    else:
        tkt_tag = ""

    prefix_lbl = "PENDING READY" if p_st == "READY" else "PROVISIONAL"

    if b_st == "PLACED_VERIFIED":
        base = f"{prefix_lbl} | MT5 PLACED {tkt_tag}".strip()
        return f"{base} | {ord_part}" if ord_part else base

    if b_st == "PLACED_MISMATCH":
        if rec.get("sl_missing_high_risk"):
            mm_tag = "SL/TP MISMATCH (SL MISSING — HIGH RISK)"
        elif not rec.get("sl_match", True) and not rec.get("tp_match", True):
            mm_tag = "SL/TP MISMATCH"
        elif not rec.get("sl_match", True):
            mm_tag = "SL/TP MISMATCH (SL)"
        else:
            mm_tag = "SL/TP MISMATCH (TP)"
        base = f"{prefix_lbl} | MT5 PLACED {tkt_tag}".strip()
        return f"{base} | {mm_tag} | {ord_part}" if ord_part else f"{base} | {mm_tag}"

    if b_st == "NEARBY_MISMATCH":
        pm = rec.get("primary_match") or {}
        p_tkt = f"#{pm.get('ticket')}" if pm.get("ticket") else (f"#{tickets[0]}" if tickets else "")
        p_diff = float(pm.get("entry_diff_pt") or 0.0)
        sl_warn = " [SL MISSING]" if rec.get("sl_missing_high_risk") else ""
        near_note = f"NEARBY {p_tkt} Δ{p_diff:+.2f} USD{sl_warn}".strip()
        return f"{prefix_lbl} | MT5: NO MATCH ({near_note}) | OTHER ORDERS: {tot_sym}"

    if b_st == "WRONG_TYPE":
        near = rec.get("nearest_unmatched_order") or {}
        n_tkt = f"#{near.get('ticket')}" if near.get("ticket") else ""
        sl_warn = " [SL MISSING]" if near.get("sl_missing") else ""
        wt_note = f"WRONG TYPE {n_tkt}{sl_warn}".strip()
        return f"{prefix_lbl} | MT5: NO MATCH ({wt_note}) | OTHER ORDERS: {tot_sym}"

    if b_st == "NOT_PLACED":
        return f"{prefix_lbl} | MT5: NOT PLACED | {ord_part}" if ord_part else f"{prefix_lbl} | MT5: NOT PLACED"

    if b_st == "OTHER_ORDER_EXISTS":
        base = f"{prefix_lbl} | MT5: NO MATCH | OTHER ORDERS: {tot_sym}"
        return f"{base} | {ord_part}" if ord_part else base

    return f"{prefix_lbl} | MT5 STATUS UNKNOWN | {ord_part}" if ord_part else f"{prefix_lbl} | MT5 STATUS UNKNOWN"


def format_pending_reconciliation_evidence(recon):
    """Menyusun bukti rekonsiliasi MT5 lengkap beserta detail mismatch SL/TP untuk `pantau detail`, `scan`, `scan snr`, dan `audit`."""
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
        tkts_str = f"#{m0.get('ticket')} (+{cnt - 1})" if cnt > 2 else ", ".join(f"#{m.get('ticket')}" for m in matched)
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
        status_lbl = "MT5 PLACED" if b_st == "PLACED_VERIFIED" else "MT5 PLACED (SL/TP MISMATCH)"

        if m0.get("sl_missing"):
            sl_ev = f"SL=0.0 vs Plan {p_sl:.2f} [SL MISSING — HIGH RISK]"
        elif m0.get("sl_match"):
            sl_ev = f"SL={o_sl:.{digits}f} (OK vs Plan {p_sl:.2f}, tol {level_tol:.3f} USD)"
        else:
            sl_ev = f"SL={o_sl:.{digits}f} vs Plan {p_sl:.2f} [SL MISMATCH Δ{float(m0.get('sl_diff_pt') or 0.0):+.{digits}f} USD]"

        if m0.get("tp_missing") and p_tp1 > 0:
            tp_ev = f"TP=0.0 vs Plan {p_tp1:.2f} [TP MISSING]"
        elif m0.get("tp_match"):
            tp_ev = f"TP={o_tp:.{digits}f} (OK vs Plan {p_tp1:.2f})"
        else:
            tp_ev = f"TP={o_tp:.{digits}f} vs Plan {p_tp1:.2f} [TP MISMATCH Δ{float(m0.get('tp_diff_pt') or 0.0):+.{digits}f} USD]"

        tol_note = f"{match_qual} | MT5 {o_ent:.{digits}f} vs Plan {p_ent:.2f}, Δ {e_diff:+.{digits}f} USD <= tol {entry_tol:.3f} USD"
        return (
            f"{status_lbl} — {cnt_lbl} | {m0.get('symbol', sym)} {m0.get('order_type')} {o_vol:g} lot @ {o_ent:.{digits}f} "
            f"({tol_note}) | {sl_ev} | {tp_ev} | Checked: {ts_chk}"
        )

    tot_sym = int(recon.get("symbol_orders_count") or 0)
    p_ent = float(recon.get("plan_entry") or 0.0)
    p_ord = str(recon.get("plan_order_type") or "PENDING")

    if b_st == "NEARBY_MISMATCH":
        pm = recon.get("primary_match") or {}
        p_tkt = f"#{pm.get('ticket')}" if pm.get("ticket") else "N/A"
        o_ent = float(pm.get("entry_price") or 0.0)
        o_vol = float(pm.get("volume") or 0.0)
        e_diff = float(pm.get("entry_diff_pt") or 0.0)
        sl_warn = " [SL MISSING — HIGH RISK]" if recon.get("sl_missing_high_risk") else ""
        return (
            f"MT5: NO MATCH | OTHER ORDERS: {tot_sym} — Nearby {p_tkt} {pm.get('order_type', p_ord)} {o_vol:g}lot @ {o_ent:.{digits}f} "
            f"(Δentry {e_diff:+.{digits}f} USD > tol {entry_tol:.3f} USD){sl_warn} | Checked: {ts_chk}"
        )

    if b_st in ("WRONG_TYPE", "TYPE_MISMATCH"):
        near = recon.get("nearest_unmatched_order") or {}
        n_tkt = f"#{near.get('ticket')}" if near.get("ticket") else ""
        n_type = near.get("order_type") or "OPPOSITE TYPE"
        n_ent = float(near.get("entry_price") or 0.0)
        n_diff = float(near.get("entry_diff_pt") or 0.0)
        sl_warn = " [SL MISSING — HIGH RISK]" if near.get("sl_missing") else ""
        return (
            f"MT5: NO MATCH | OTHER ORDERS: {tot_sym} — Wrong Type {n_type} {n_tkt} @ {n_ent:.{digits}f} "
            f"(Δ {n_diff:+.{digits}f} USD{sl_warn}) vs Plan {p_ord} @ {p_ent:.2f} | Checked: {ts_chk}"
        )

    if b_st == "OTHER_ORDER_EXISTS":
        if p_ent > 0:
            return (
                f"MT5: NO MATCH | OTHER ORDERS: {tot_sym} "
                f"(tidak ada {p_ord} @ {p_ent:.2f}, tol ±{entry_tol:.3f} USD) | Checked: {ts_chk}"
            )
        return f"MT5: NO MATCH | OTHER ORDERS: {tot_sym} ({sym}) | Checked: {ts_chk}"

    return f"NOT PLACED — 0 pending order di {sym} (Terverifikasi live MT5) | Checked: {ts_chk}"


def format_mt5_orders_summary_lines(
    mt5_snapshot,
    invalid_zone_warnings=None,
    matched_orders=None,
    matched_positions=None,
    matched_recons=None,
    detailed=False,
    invalid_zone_scope=None,
):
    """
    Menyusun ringkasan read-only pending order & posisi aktif MT5 + warning SL/zona invalid.
    Output utama ringkas (jumlah posisi, pending, matched, tanpa SL) tanpa daftar panjang seluruh ticket;
    detail ticket hanya untuk order yang cocok dengan active setup atau saat detailed=True.
    """
    snap = mt5_snapshot if isinstance(mt5_snapshot, dict) else {}
    sync_ts = str(snap.get("retrieved_at") or "").strip()
    sync_sfx = f" | Sync: {sync_ts}" if sync_ts else ""
    if not snap.get("query_ok"):
        return [f"- MT5 Sync         : UNKNOWN ({snap.get('reason') or 'MT5 tidak terhubung'}){sync_sfx}"]
    digits = int(snap.get("digits") or 2)
    sym = snap.get("symbol") or SYMBOL
    orders = list(snap.get("orders") or [])
    positions = list(snap.get("positions") or [])

    matched_ord_map = {}
    matched_ord_seq = []
    for mo in (matched_orders or []):
        if isinstance(mo, dict):
            tkt = mo.get("ticket")
            k = tkt if tkt is not None else id(mo)
            if k not in matched_ord_map:
                matched_ord_map[k] = mo
                matched_ord_seq.append(k)
    matched_pos_map = {}
    matched_pos_seq = []
    for mp in (matched_positions or []):
        if isinstance(mp, dict):
            tkt = mp.get("ticket")
            k = tkt if tkt is not None else id(mp)
            if k not in matched_pos_map:
                matched_pos_map[k] = mp
                matched_pos_seq.append(k)
    for rec in (matched_recons or []):
        if not isinstance(rec, dict):
            continue
        for mo in (rec.get("matched_orders") or []):
            if isinstance(mo, dict):
                tkt = mo.get("ticket")
                k = tkt if tkt is not None else id(mo)
                if k not in matched_ord_map:
                    matched_ord_map[k] = mo
                    matched_ord_seq.append(k)
        for mp in (rec.get("matched_positions") or []):
            if isinstance(mp, dict):
                tkt = mp.get("ticket")
                k = tkt if tkt is not None else id(mp)
                if k not in matched_pos_map:
                    matched_pos_map[k] = mp
                    matched_pos_seq.append(k)

    risk_sum = summarize_mt5_risk_warnings(
        snap,
        invalid_zone_warnings=invalid_zone_warnings,
        matched_recons=matched_recons,
        invalid_zone_scope=invalid_zone_scope,
    )
    pos_no_sl_cnt = risk_sum["positions_without_sl_count"]
    ord_no_sl_cnt = risk_sum["pending_without_sl_count"]
    total_no_sl = pos_no_sl_cnt + ord_no_sl_cnt
    nosl_breakdown = f" ({pos_no_sl_cnt} Pos + {ord_no_sl_cnt} Pending)" if total_no_sl > 0 else ""
    matched_cnt = len(matched_ord_seq)
    inv_ord_cnt = risk_sum["invalid_zone_tickets_count"]
    unmatched_manual_cnt = max(0, len(orders) - matched_cnt - inv_ord_cnt)
    other_cnt = max(0, len(orders) - matched_cnt)

    lines = [
        f"- MT5 Summary      : Positions: {len(positions)} | Pending: {len(orders)} (Matched: {matched_cnt} | Unmatched Manual: {unmatched_manual_cnt} | On Invalid Zone: {inv_ord_cnt}) | Without SL: {total_no_sl}{nosl_breakdown}{sync_sfx}",
    ]
    if matched_ord_seq:
        m_parts = []
        for k in matched_ord_seq[:3]:
            mo = matched_ord_map[k]
            sl_v = float(mo.get("sl") or 0.0)
            tp_v = float(mo.get("tp") or 0.0)
            sl_tag = "SL MISSING!" if sl_v <= 0 else f"SL {sl_v:.{digits}f}"
            tp_tag = f"TP {tp_v:.{digits}f}" if tp_v > 0 else "TP -"
            m_parts.append(
                f"#{mo.get('ticket')} {mo.get('order_type')} {float(mo.get('volume') or 0.0):g}lot @ {float(mo.get('entry_price') or 0.0):.{digits}f} ({sl_tag} | {tp_tag})"
            )
        extra_m = f" (+{matched_cnt - 3} matched)" if matched_cnt > 3 else ""
        other_note = f" | UNMATCHED MANUAL: {unmatched_manual_cnt}" if unmatched_manual_cnt > 0 else ""
        lines.append(f"- MT5 Matched      : {'; '.join(m_parts)}{extra_m}{other_note}")
    elif len(orders) > 0:
        lines.append(
            f"- MT5 Pending      : MT5: NO MATCH | UNMATCHED MANUAL ORDERS: {unmatched_manual_cnt} (order manual di luar plan aktif scanner — bukan otomatis invalid)"
            + (f" | ON INVALID ZONE: {inv_ord_cnt}" if inv_ord_cnt > 0 else "")
        )
    else:
        lines.append(f"- MT5 Pending      : NOT PLACED (0 pending order aktif pada {sym})")

    if matched_pos_seq:
        p_parts = []
        for k in matched_pos_seq[:2]:
            mp = matched_pos_map[k]
            sl_v = float(mp.get("sl") or 0.0)
            sl_tag = "SL MISSING!" if sl_v <= 0 else f"SL {sl_v:.{digits}f}"
            compact_p = format_tracked_position_compact(mp, digits=digits)
            p_parts.append(f"{compact_p} ({sl_tag})" if compact_p else f"#{mp.get('ticket')} {mp.get('direction')} @ {float(mp.get('entry_price') or mp.get('price_open') or 0.0):.{digits}f} ({sl_tag})")
        extra_p = f" (+{len(matched_pos_seq) - 2})" if len(matched_pos_seq) > 2 else ""
        lines.append(f"- Tracked Position : {'; '.join(p_parts)}{extra_p}")

    if risk_sum["headline"]:
        lines.append(f"- {risk_sum['headline']}")
        if detailed and risk_sum["detail_lines"]:
            for dln in risk_sum["detail_lines"][:12]:
                lines.append(dln)
            if len(risk_sum["detail_lines"]) > 12:
                lines.append(f"  • (+{len(risk_sum['detail_lines']) - 12} ticket risiko lainnya; gunakan 'audit' untuk daftar penuh)")
    return lines


def reconcile_pending_plan_with_mt5(plan, mt5_snapshot=None, symbol=SYMBOL, excluded_tickets=None):
    """
    Rekonsiliasi READ-ONLY satu rencana pending scanner terhadap pending order & posisi aktual di MT5.
    Aturan:
    1. Filter berdasarkan akun yang terhubung dan simbol broker (mis. XAUUSDc).
    2. Wajib cocok direction & tipe order kompatibel (BUY LIMIT == BUY LIMIT, SELL LIMIT == SELL LIMIT).
    3. Bandingkan entry price menggunakan tick_size simbol dan toleransi terukur (PENDING_ENTRY_TOLERANCE_MIN_PT).
    4. Bandingkan SL dan TP secara independen; deteksi SL=0 sebagai SL MISSING — HIGH RISK.
    5. Tidak mewajibkan magic number / comment khusus scanner (mendukung penuh order manual).
    6. Bedakan secara eksplisit: MT5 PLACED, NOT PLACED, OTHER ORDER, dan SL/TP MISMATCH.
    7. Tidak pernah membuat, mengubah, atau membatalkan order MT5.
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
    elif raw_st.startswith("PROVISIONAL"):
        plan_status = "PROVISIONAL"
    elif raw_st.startswith("EXPIRED") or p_dict.get("is_expired"):
        plan_status = "EXPIRED"
    elif raw_st.startswith("INVALID") or p_dict.get("valid") is False:
        plan_status = "INVALID" if raw_st.startswith("INVALID") else "WAIT"
    else:
        plan_status = "WAIT"

    order_type = str(p_dict.get("order_type") or "").strip().upper()
    p_dir = str(p_dict.get("direction") or "").strip().upper()
    if not order_type or order_type == "NONE":
        p_zt = str(p_dict.get("zone_type") or "").strip().upper()
        if p_dir == "BUY" or p_zt in ("DEMAND", "SUPPORT"):
            order_type = "BUY LIMIT"
            p_dir = "BUY"
        elif p_dir == "SELL" or p_zt in ("SUPPLY", "RESISTANCE"):
            order_type = "SELL LIMIT"
            p_dir = "SELL"
        else:
            order_type = ""
    if p_dir not in ("BUY", "SELL") and order_type:
        _, p_dir = _info_tipe_pending(order_type)

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
    plan_ticket = p_dict.get("ticket")
    try:
        plan_ticket = int(plan_ticket) if plan_ticket is not None else None
    except (TypeError, ValueError):
        plan_ticket = None

    z_rng = p_dict.get("zone_range") if isinstance(p_dict.get("zone_range"), (list, tuple)) and len(p_dict.get("zone_range")) >= 2 else (0.0, 0.0)
    try:
        zone_low = float(p_dict.get("zone_low") or p_dict.get("price_low") or p_dict.get("low") or z_rng[0] or 0.0)
        zone_high = float(p_dict.get("zone_high") or p_dict.get("price_high") or p_dict.get("high") or z_rng[1] or 0.0)
    except (TypeError, ValueError):
        zone_low, zone_high = 0.0, 0.0

    zone_id = str(p_dict.get("zone_id") or p_dict.get("source_zone_id") or "").strip()
    digits = int(snap.get("digits") or 2)
    tick_size = float(snap.get("tick_size") or 0.01)
    entry_tol = float(snap.get("entry_tolerance") or PENDING_ENTRY_TOLERANCE_MIN_PT)
    level_tol = float(snap.get("level_tolerance") or PENDING_LEVEL_TOLERANCE_MIN_PT)
    retrieved_at = str(snap.get("retrieved_at") or "")
    orders = list(snap.get("orders") or [])
    positions = list(snap.get("positions") or [])
    excl_set = set(excluded_tickets or ())

    setup_created_ts = float(p_dict.get("created_at_ts") or 0.0)
    zone_form_ts = float(p_dict.get("zone_formation_ts") or 0.0)
    linked_tkts = list(p_dict.get("linked_tickets") or [])
    if plan_ticket is not None:
        linked_tkts.append(plan_ticket)

    matched_positions = []
    confirmed_positions = []
    existing_positions = []
    for pos_item in positions:
        if str(pos_item.get("symbol") or "").upper() not in accepted_syms:
            continue
        pos_dir = str(pos_item.get("direction") or "").upper()
        pos_ent = float(pos_item.get("entry_price") or pos_item.get("price_open") or 0.0)
        in_z_p = bool(zone_low > 0 and zone_high > 0 and (zone_low - entry_tol <= pos_ent <= zone_high + entry_tol))
        near_e_p = bool(plan_entry > 0 and abs(pos_ent - plan_entry) <= max(entry_tol, 1.5))
        if p_dir in ("BUY", "SELL") and pos_dir == p_dir and (in_z_p or near_e_p):
            ev_m = evaluate_mt5_setup_match(
                pos_item,
                direction=p_dir,
                zone_low=zone_low,
                zone_high=zone_high,
                plan_entry=plan_entry,
                zone_id=zone_id,
                setup_created_ts=setup_created_ts,
                zone_formation_ts=zone_form_ts,
                linked_tickets=linked_tkts,
                accepted_syms=accepted_syms,
                entry_tol=max(entry_tol, 1.5),
                is_order=False,
            )
            pos_enriched = dict(pos_item)
            pos_enriched["mt5_status"] = ev_m["mt5_status"]
            pos_enriched["setup_match"] = ev_m["setup_match"]
            pos_enriched["is_existing_position"] = ev_m["is_existing_position"]
            pos_enriched["match_reason"] = ev_m["reason"]
            matched_positions.append(pos_enriched)
            if ev_m["setup_match"] == "CONFIRMED":
                confirmed_positions.append(pos_enriched)
            elif ev_m["is_existing_position"] or ev_m["setup_match"] in ("UNRELATED", "UNKNOWN"):
                existing_positions.append(pos_enriched)

    base_res = {
        "zone_id": zone_id,
        "symbol": target_sym,
        "plan_status": plan_status,
        "plan_direction": p_dir,
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
        "symbol_positions_count": len(positions),
        "broker_order_status": "UNKNOWN",
        "display_status": "UNKNOWN",
        "is_placed": False,
        "matched_count": 0,
        "matched_tickets": [],
        "matched_orders": [],
        "matched_positions": matched_positions,
        "confirmed_positions": confirmed_positions,
        "existing_positions": existing_positions,
        "active_position_tickets": [p.get("ticket") for p in matched_positions if p.get("ticket") is not None],
        "mt5_status": "OPEN" if matched_positions else ("PENDING" if len(orders) > 0 else "NONE"),
        "setup_match": (
            "CONFIRMED" if confirmed_positions
            else (existing_positions[0].get("setup_match", "UNKNOWN") if existing_positions else ("INFERRED" if matched_positions else "UNRELATED"))
        ),
        "orders_without_sl": list(snap.get("orders_without_sl") or []),
        "positions_without_sl": list(snap.get("positions_without_sl") or []),
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

    if plan_status not in ("READY", "PROVISIONAL") or plan_entry <= 0 or not order_type:
        if len(orders) > 0:
            base_res["broker_order_status"] = "OTHER_ORDER_EXISTS"
            base_res["display_status"] = f"NO MATCH | OTHER ORDERS: {len(orders)}"
            base_res["sl_missing_high_risk"] = any(float(o.get("sl") or 0.0) <= 0.0 for o in orders)
            base_res["nearest_unmatched_order"] = None
        else:
            base_res["broker_order_status"] = "NOT_PLACED"
            base_res["display_status"] = "NOT PLACED"
        base_res["compact_status"] = format_pending_compact_status(plan_status, base_res, order_type=order_type, entry_price=plan_entry)
        base_res["evidence_summary"] = format_pending_reconciliation_evidence(base_res)
        return base_res

    matched_orders = []
    price_mismatch_orders = []
    type_mismatch_orders = []

    manual_entry_tol = max(entry_tol, 1.5)

    for ord_item in orders:
        tkt = ord_item.get("ticket")
        if tkt in excl_set:
            continue
        if str(ord_item.get("symbol") or "").upper() not in accepted_syms:
            continue
        o_type = str(ord_item.get("order_type") or "").upper()
        o_dir = str(ord_item.get("direction") or "").upper()
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
        ticket_match = bool(plan_ticket is not None and tkt == plan_ticket)

        if o_type != order_type or (p_dir in ("BUY", "SELL") and o_dir and o_dir != p_dir):
            if is_nearby:
                type_mismatch_orders.append({
                    "ticket": tkt,
                    "order_type": o_type,
                    "direction": o_dir,
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

        mm_reasons = []
        if sl_missing:
            mm_reasons.append("SL MISSING — HIGH RISK")
        elif not sl_ok:
            mm_reasons.append(f"SL MISMATCH (MT5 {o_sl:.{digits}f} vs Plan {plan_sl:.2f}, Δ{sl_diff:+.{digits}f} USD)")
        if tp_missing and plan_tp1 > 0:
            mm_reasons.append("TP MISSING")
        elif not tp_ok:
            mm_reasons.append(f"TP MISMATCH (MT5 {o_tp:.{digits}f} vs Plan {plan_tp1:.2f}, Δ{tp_diff:+.{digits}f} USD)")

        order_data = {
            "ticket": tkt,
            "order_type": o_type,
            "direction": o_dir,
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
            "ticket_match": ticket_match,
        }

        if abs(entry_diff) <= entry_tol + 1e-9 or ticket_match:
            order_data["match_quality"] = "EXACT_MATCH" if abs(entry_diff) <= tick_size + 1e-9 else "ACCEPTED_TOLERANCE"
            matched_orders.append(order_data)
        elif is_nearby:
            order_data["match_quality"] = "NEARBY_MISMATCH"
            price_mismatch_orders.append(order_data)

    matched_orders.sort(key=lambda m: (not m.get("ticket_match"), abs(m["entry_diff_pt"]), not (m["sl_match"] and m["tp_match"]), m["ticket"] or 0))
    price_mismatch_orders.sort(key=lambda m: (abs(m["entry_diff_pt"]), not (m["sl_match"] and m["tp_match"]), m["ticket"] or 0))
    type_mismatch_orders.sort(key=lambda m: (abs(m["entry_diff_pt"]), m["ticket"] or 0))

    if matched_orders:
        all_verified = all(m["sl_match"] and m["tp_match"] for m in matched_orders)
        b_status = "PLACED_VERIFIED" if all_verified else "PLACED_MISMATCH"
        primary_qual = matched_orders[0].get("match_quality", "ACCEPTED_TOLERANCE")
        base_res.update({
            "broker_order_status": b_status,
            "display_status": "MT5 PLACED" if all_verified else "SL/TP MISMATCH",
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
        base_res.update({
            "broker_order_status": "NEARBY_MISMATCH",
            "display_status": "OTHER ORDER",
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
            "display_status": "OTHER ORDER",
            "is_placed": False,
            "match_quality": "WRONG_TYPE",
            "nearest_unmatched_order": t0,
            "sl_missing_high_risk": bool(t0.get("sl_missing")),
        })
    else:
        avail_orders = [o for o in orders if o.get("ticket") not in excl_set]
        if len(avail_orders) > 0 or len(orders) > 0:
            base_res.update({
                "broker_order_status": "OTHER_ORDER_EXISTS",
                "display_status": f"NO MATCH | OTHER ORDERS: {len(orders)}",
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
                    "direction": nearest.get("direction"),
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
                if n_sl <= 0.0:
                    base_res["sl_missing_high_risk"] = True
        else:
            base_res.update({
                "broker_order_status": "NOT_PLACED",
                "display_status": "NOT PLACED",
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
    u_spec = get_symbol_unit_spec(symbol=SYMBOL, sym_info=sym)
    digits = u_spec["digits"]
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

    target_sym, accepted_syms = _acceptable_trade_symbols(SYMBOL)
    pos, _ = _query_mt5_symbol_items(mt5.positions_get, target_sym, accepted_syms)
    orders, _ = _query_mt5_symbol_items(mt5.orders_get, target_sym, accepted_syms)

    if pos is None and orders is None:
        out.append("  [POSISI & PENDING ORDER]: DATA UNAVAILABLE (Gagal membaca data posisi/order dari MT5).")
        return "\n".join(out) + "\n"

    server_off_live = get_mt5_server_utc_offset_hours(tick=tick, allow_live_query=False)
    if pos:
        out.append(f"  [POSISI RUNNING / FLOATING SAAT INI — TOTAL {len(pos)} POSISI AKTIF (WAJIB DIKAWAL, JANGAN DIABAIKAN!)]:")
        for p in pos:
            arah = "BUY" if p.type == mt5.ORDER_TYPE_BUY else "SELL"
            harga_tutup = bid_live if arah == "BUY" else ask_live
            poin = (harga_tutup - p.price_open) if arah == "BUY" else (p.price_open - harga_tutup)
            pct_eq = (p.profit / eq * 100.0) if eq > 0 else 0.0
            open_utc_p = mt5_ts_to_utc_epoch(getattr(p, "time", 0), getattr(p, "time_msc", 0), server_off_live)
            open_wib_p = format_mt5_wib(open_utc_p, short=False) or "TIME_UNKNOWN"
            if p.sl and p.sl > 0:
                sudah_bep = (arah == "BUY" and p.sl >= p.price_open) or (arah == "SELL" and p.sl <= p.price_open)
                if sudah_bep:
                    kunci_pt = (p.sl - p.price_open) if arah == "BUY" else (p.price_open - p.sl)
                    sl_info = f"{p.sl:.{digits}f} [✅ SUDAH DIKUNCI BEP+ ({format_price_distance(kunci_pt, spec=u_spec, include_mt5_points=True, signed=True)} di area profit) — POSISI 100% BEBAS RISIKO!]"
                else:
                    jrk_sl = (harga_tutup - p.sl) if arah == "BUY" else (p.sl - harga_tutup)
                    sl_info = f"{p.sl:.{digits}f} (jarak dari harga sekarang ke SL: {format_price_distance(jrk_sl, spec=u_spec, include_mt5_points=True)})"
            else:
                sl_info = f"BELUM DISET (⚠️ RAWAN TANPA PENGAMAN!) [CRITICAL WARNING: Posisi #{p.ticket} TANPA STOP LOSS!]"
            tp_info = f"{p.tp:.{digits}f}" if (p.tp and p.tp > 0) else "BELUM DISET (Terbuka / Runner)"
            out.append(
                f"  - Ticket #{p.ticket} (PosID #{getattr(p, 'identifier', p.ticket)}) | MT5 STATUS: OPEN | POSITION OPENED: {open_wib_p} | "
                f"Simbol: {getattr(p, 'symbol', SYMBOL)} | {arah} {p.volume} lot @ {p.price_open:.{digits}f} (Harga sekarang: {harga_tutup:.{digits}f}) | "
                f"SL: {sl_info} | TP: {tp_info} | "
                f"Floating: {p.profit:+,.2f} {cur} ({pct_eq:+.2f}% equity | {format_price_distance(poin, spec=u_spec, include_mt5_points=True, signed=True)})"
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
            cr_utc_o = mt5_ts_to_utc_epoch(getattr(o, "time_setup", 0), getattr(o, "time_setup_msc", 0), server_off_live)
            cr_wib_o = format_mt5_wib(cr_utc_o, short=False) or "TIME_UNKNOWN"
            if (o.sl and o.sl > 0):
                sl_ord_txt = f"{o.sl:.{digits}f} (jarak SL {format_price_distance(abs(o.price_open - o.sl), spec=u_spec, include_mt5_points=True)})"
            else:
                sl_ord_txt = f"BELUM DISET (⚠️ WAJIB INGATKAN PASANG SL!) [CRITICAL WARNING: Pending Order #{o.ticket} TANPA STOP LOSS!]"
            tp_ord_txt = f"{o.tp:.{digits}f}" if (o.tp and o.tp > 0) else "BELUM DISET"
            exp_time = getattr(o, "time_expiration", 0)
            if isinstance(exp_time, (int, float)) and exp_time > 0:
                exp_utc = mt5_ts_to_utc_epoch(exp_time, 0, server_off_live)
                exp_txt = format_mt5_wib(exp_utc, short=False) or str(exp_time)
            else:
                exp_txt = "GTC (Good till cancelled)"

            out.append(
                f"  - Ticket #{o.ticket} | MT5 STATUS: PENDING | ORDER CREATED: {cr_wib_o} | Simbol: {getattr(o, 'symbol', SYMBOL)} | {nama_ord} ({arah_ord}) {vol_ord} lot @ {o.price_open:.{digits}f} "
                f"(Jarak: {format_price_distance(jarak_ord, spec=u_spec, include_mt5_points=True)} {letak_ord}) | "
                f"SL: {sl_ord_txt} | TP: {tp_ord_txt} | Expiration: {exp_txt}"
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
                    d_utc = mt5_ts_to_utc_epoch(getattr(d, "time", 0), getattr(d, "time_msc", 0), server_off_live)
                    t_d = format_mt5_wib(d_utc, short=False) or "TIME_UNKNOWN"
                    net_d = d.profit + d.commission + d.swap + getattr(d, "fee", 0.0)
                    alasan_d = {
                        getattr(mt5, "DEAL_REASON_SL", 4): "kena SL",
                        getattr(mt5, "DEAL_REASON_TP", 5): "kena TP",
                        getattr(mt5, "DEAL_REASON_SO", 6): "Stop Out",
                    }.get(getattr(d, "reason", -1), "tutup manual/script")
                    out.append(
                        f"  - [MT5 STATUS: CLOSED | CLOSED: {t_d}] Deal #{d.ticket} (Pos #{getattr(d, 'position_id', '-')}) | "
                        f"{arah_d} {d.volume} lot @ {d.price:.3f} | Hasil: {net_d:+,.2f} {cur} ({alasan_d})"
                    )
        except Exception as e_hist:
            _catat_error("cek_posisi_terbuka_mt5_hist", e_hist)

    return "\n".join(out) + "\n"


def pastikan_mt5_aktif():
    """Menjaga koneksi MT5 tetap terbuka (persistent) selama aplikasi berjalan agar tidak perlu buka-tutup berulang kali."""
    global MT5_TERHUBUNG
    with MT5_IPC_LOCK:
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
