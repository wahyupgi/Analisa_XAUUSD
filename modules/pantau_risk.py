# -*- coding: utf-8 -*-
"""
modules/pantau_risk.py
Audit akun MT5 untuk mode pantau: profit posisi, saran proteksi profit, audit pending order,
dan rekomendasi SL struktural.

Dipindahkan apa adanya dari modules/pantau.py (structural refactoring, tanpa perubahan logika).
Dependency satu arah: modul ini tidak mengimpor modules.pantau; pantau mengimpor kembali
seluruh nama publiknya agar caller lama tetap kompatibel.
"""
import math

import MetaTrader5 as mt5

from modules.config import SYMBOL
from modules.mt5_conn import _info_tipe_pending
from modules.utils import (
    get_symbol_unit_spec,
    price_distance_to_mt5_points,
    mt5_points_to_price_distance,
)


def audit_position_profit(
    pos,
    tick,
    acc_info=None,
    sym_info=None,
    expected_symbol=SYMBOL,
    calc_profit_fn=None,
):
    """Mengaudit floating profit posisi MT5 menggunakan nilai broker `pos.profit` sebagai sumber utama.

    Tidak pernah mengganti nilai broker dengan estimasi manual; jika ditemukan inkonsistensi
    (simbol/volume/harga/tick tidak valid, tick lebih tua dari update posisi, arah sign
    berlawanan jauh di luar spread, atau anomali skala seperti $87,193 pada 0.01 lot USD),
    mengembalikan status `PROFIT_DATA_SUSPECT` sehingga angka mencurigakan tidak disajikan valid.
    """
    ticket = getattr(pos, "ticket", None)
    pos_symbol = str(getattr(pos, "symbol", expected_symbol) or expected_symbol)
    pos_type = getattr(pos, "type", None)
    raw_profit = getattr(pos, "profit", None)
    volume = getattr(pos, "volume", None)
    price_open = getattr(pos, "price_open", None)
    swap = float(getattr(pos, "swap", 0.0) or 0.0)
    raw_comm = getattr(pos, "commission", None)
    has_commission = isinstance(raw_comm, (int, float)) and math.isfinite(raw_comm)
    commission = float(raw_comm) if has_commission else None
    raw_currency = getattr(acc_info, "currency", None) if acc_info is not None else None
    currency = str(raw_currency or "").strip().upper()
    profit_currency = str(getattr(sym_info, "currency_profit", "") or "").strip().upper() if sym_info is not None else ""
    contract_size = float(getattr(sym_info, "trade_contract_size", 0.0) or 0.0) if sym_info is not None else 0.0

    buy_code = getattr(mt5, "ORDER_TYPE_BUY", 0)
    sell_code = getattr(mt5, "ORDER_TYPE_SELL", 1)
    is_buy = pos_type == buy_code
    is_sell = pos_type == sell_code

    def _suspect(reason, points=0.0, close_price=0.0, expected_profit=None, status_code="PROFIT_DATA_SUSPECT"):
        diag = (
            f"MT5_RAW(ticket={ticket}, sym={pos_symbol}, profit={raw_profit}, swap={swap}, "
            f"comm={raw_comm}, acc_cur={currency or 'MISSING'}, sym_cur_prof={profit_currency or 'MISSING'}, "
            f"contract={contract_size}) -> STATUS={status_code} ({reason})"
        )
        return {
            "valid": False,
            "status": status_code,
            "reason": reason,
            "ticket": ticket,
            "symbol": pos_symbol,
            "arah": "BUY" if is_buy else ("SELL" if is_sell else "UNKNOWN"),
            "profit": None,
            "raw_profit": raw_profit,
            "swap": swap,
            "commission": commission,
            "net_profit": None,
            "currency": currency or "UNVERIFIED",
            "profit_currency": profit_currency,
            "contract_size": contract_size,
            "display": status_code,
            "points": points,
            "close_price": close_price,
            "expected_profit": expected_profit,
            "timestamp": getattr(tick, "time", None) if tick is not None else None,
            "diagnostic_log": diag,
        }

    if not currency:
        return _suspect("ACCOUNT_CURRENCY_MISSING", status_code="CURRENCY_OR_PNL_UNVERIFIED")
    if ticket is None or not (is_buy or is_sell):
        return _suspect("INVALID_POSITION_OBJECT")
    if expected_symbol and pos_symbol != expected_symbol:
        return _suspect("SYMBOL_MISMATCH")
    if not isinstance(volume, (int, float)) or not math.isfinite(volume) or volume <= 0:
        return _suspect("INVALID_VOLUME")
    if not isinstance(price_open, (int, float)) or not math.isfinite(price_open) or price_open <= 0:
        return _suspect("INVALID_ENTRY_PRICE")
    if not isinstance(raw_profit, (int, float)) or not math.isfinite(raw_profit):
        return _suspect("NON_FINITE_PROFIT")
    if tick is None:
        return _suspect("MISSING_TICK")

    bid = getattr(tick, "bid", None)
    ask = getattr(tick, "ask", None)
    if (
        not isinstance(bid, (int, float)) or not math.isfinite(bid) or bid <= 0
        or not isinstance(ask, (int, float)) or not math.isfinite(ask) or ask <= 0
    ):
        return _suspect("INVALID_TICK_PRICE")

    tick_time = getattr(tick, "time", None)
    pos_time_upd = getattr(pos, "time_update", getattr(pos, "time", None))
    if (
        isinstance(tick_time, (int, float)) and isinstance(pos_time_upd, (int, float))
        and tick_time > 0 and pos_time_upd > 0 and (pos_time_upd - tick_time) > 60
    ):
        return _suspect("STALE_TICK_OLDER_THAN_POSITION")

    close_price = float(bid) if is_buy else float(ask)
    points = (close_price - float(price_open)) if is_buy else (float(price_open) - close_price)
    spread = max(0.0, float(ask) - float(bid))

    # Hitung referensi pembanding dalam mata uang akun (hanya untuk validasi konsistensi)
    expected_profit = None
    fn_calc = calc_profit_fn if calc_profit_fn is not None else getattr(mt5, "order_calc_profit", None)
    if callable(fn_calc):
        try:
            calc_val = fn_calc(pos_type, pos_symbol, float(volume), float(price_open), close_price)
            if isinstance(calc_val, (int, float)) and math.isfinite(calc_val):
                expected_profit = float(calc_val)
        except Exception:
            expected_profit = None

    if expected_profit is None and sym_info is not None:
        t_size = float(getattr(sym_info, "trade_tick_size", 0.0) or 0.0)
        t_val = float(getattr(sym_info, "trade_tick_value", 0.0) or 0.0)
        if t_size > 0 and t_val > 0:
            expected_profit = (points / t_size) * t_val * float(volume)

    if expected_profit is None and currency in ("USD", "USC"):
        c_size = float(getattr(sym_info, "trade_contract_size", 100.0) or 100.0) if sym_info is not None else 100.0
        mult = 100.0 if currency == "USC" else 1.0
        expected_profit = points * c_size * float(volume) * mult

    broker_profit = float(raw_profit)

    if expected_profit is not None:
        # 1) Sign berlawanan jauh di luar batas spread/komisi
        if abs(points) > max(spread * 2.0, 1.0) and expected_profit != 0.0:
            if (broker_profit * expected_profit) < 0 and abs(broker_profit - expected_profit) > max(5.0, abs(expected_profit) * 0.5):
                return _suspect("SIGN_MISMATCH", points=points, close_price=close_price, expected_profit=expected_profit)

        # 2) Anomali skala besar (mis. $87,193.10 pada 0.01 lot +5.10 pt di akun USD)
        abs_exp = abs(expected_profit)
        abs_brk = abs(broker_profit)
        tol_abs = max(50.0, abs_exp * 2.5)
        if abs_exp >= 0.01 and abs(broker_profit - expected_profit) > tol_abs:
            if (abs_brk / max(abs_exp, 0.01)) > 5.0 or (abs_exp / max(abs_brk, 0.01)) > 5.0:
                return _suspect("SCALE_ANOMALY", points=points, close_price=close_price, expected_profit=expected_profit)
        elif abs_exp < 0.01 and abs_brk > 50.0 and abs(points) < 0.5:
            return _suspect("SCALE_ANOMALY_NEAR_ZERO", points=points, close_price=close_price, expected_profit=expected_profit)
    else:
        # Mata uang non-USD tanpa order_calc_profit / tick_value tidak dapat diverifikasi skalanya
        if currency not in ("USD", "USC"):
            return _suspect(
                "CURRENCY_SCALE_UNVERIFIED",
                points=points,
                close_price=close_price,
                expected_profit=None,
                status_code="CURRENCY_OR_PNL_UNVERIFIED",
            )
        if abs(points) > max(spread * 3.0, 2.0):
            if (points > 0 and broker_profit < -5.0) or (points < 0 and broker_profit > 5.0):
                return _suspect("SIGN_MISMATCH_NO_FX_REF", points=points, close_price=close_price, expected_profit=None)

    display_str = f"{broker_profit:+,.2f} {currency}"
    net_profit = (broker_profit + swap + commission) if has_commission else None
    diag_ok = (
        f"MT5_RAW(ticket={ticket}, sym={pos_symbol}, profit={raw_profit}, swap={swap}, "
        f"comm={raw_comm}, acc_cur={currency}, sym_cur_prof={profit_currency or 'N/A'}, "
        f"contract={contract_size}) -> DISPLAY={display_str}"
    )
    return {
        "valid": True,
        "status": "VALID",
        "reason": "",
        "ticket": ticket,
        "symbol": pos_symbol,
        "arah": "BUY" if is_buy else "SELL",
        "profit": broker_profit,
        "raw_profit": raw_profit,
        "swap": swap,
        "commission": commission,
        "net_profit": net_profit,
        "currency": currency,
        "profit_currency": profit_currency,
        "contract_size": contract_size,
        "display": display_str,
        "points": points,
        "close_price": close_price,
        "expected_profit": expected_profit,
        "timestamp": tick_time,
        "diagnostic_log": diag_ok,
    }


def format_portfolio_profit(
    pos_list,
    tick,
    acc_info=None,
    sym_info=None,
    expected_symbol=SYMBOL,
    compact=False,
    calc_profit_fn=None,
):
    """Mengaudit seluruh posisi terbuka dan memformat total floating profit secara aman.

    Jika salah satu posisi berstatus PROFIT_DATA_SUSPECT, seluruh ringkasan mengembalikan
    `PROFIT_DATA_SUSPECT` agar angka yang korup tidak pernah disajikan sebagai profit valid.
    """
    audits = [
        audit_position_profit(
            p, tick, acc_info=acc_info, sym_info=sym_info,
            expected_symbol=expected_symbol, calc_profit_fn=calc_profit_fn,
        )
        for p in (pos_list or ())
    ]
    if not audits:
        cur = str(getattr(acc_info, "currency", "") or "").strip().upper()
        if not cur:
            return "CURRENCY_OR_PNL_UNVERIFIED", audits
        return (f"0.0 {cur}" if compact else f"0.00 {cur}"), audits
    if any(a["status"] == "PROFIT_DATA_SUSPECT" for a in audits):
        return "PROFIT_DATA_SUSPECT", audits
    if any(a["status"] in ("CURRENCY_OR_PNL_UNVERIFIED", "PROFIT_CURRENCY_UNVERIFIED") for a in audits):
        return "CURRENCY_OR_PNL_UNVERIFIED", audits
    cur = audits[0]["currency"]
    total = sum(float(a["profit"]) for a in audits)
    fmt = f"{total:+,.1f} {cur}" if compact else f"{total:+,.2f} {cur}"
    return fmt, audits


def build_profit_protection_suggestion(pos, tick, sym_info=None, acc_info=None):
    """Menghitung PROFIT PROTECTION SUGGESTION yang valid secara arah & aturan broker (manual, tanpa klaim bebas risiko)."""
    arah = "BUY" if getattr(pos, "type", 0) == getattr(mt5, "ORDER_TYPE_BUY", 0) else "SELL"
    entry = float(getattr(pos, "price_open", 0.0) or 0.0)
    vol = float(getattr(pos, "volume", 0.0) or 0.0)
    raw_swap = getattr(pos, "swap", None)
    raw_comm = getattr(pos, "commission", None)
    has_swap = isinstance(raw_swap, (int, float)) and math.isfinite(raw_swap)
    has_comm = isinstance(raw_comm, (int, float)) and math.isfinite(raw_comm)
    swap = float(raw_swap) if has_swap else 0.0
    comm = float(raw_comm) if has_comm else 0.0
    bid = float(getattr(tick, "bid", 0.0) or 0.0) if tick is not None else 0.0
    ask = float(getattr(tick, "ask", 0.0) or 0.0) if tick is not None else 0.0

    def _unavail(reason):
        return {
            "allowed": False,
            "status": "PROFIT_PROTECTION_UNAVAILABLE",
            "label": "PROFIT PROTECTION SUGGESTION",
            "arah": arah,
            "entry_price": entry,
            "current_bid": bid,
            "current_ask": ask,
            "suggested_sl": None,
            "gross_profit_locked_pts": None,
            "estimated_net_profit": None,
            "buffer_pts": None,
            "calculation_basis": "",
            "reason": reason,
            "disclaimer": "",
        }

    if sym_info is None or tick is None or entry <= 0 or bid <= 0 or ask <= 0:
        return _unavail("PROFIT_PROTECTION_UNAVAILABLE")

    u_spec = get_symbol_unit_spec(symbol=getattr(pos, "symbol", SYMBOL), sym_info=sym_info)
    digits = u_spec["digits"]
    point = u_spec["point"]
    tick_size = u_spec["trade_tick_size"]
    tick_value = float(u_spec["trade_tick_value"] or 0.0)
    contract_size = float(u_spec["trade_contract_size"] or 100.0)
    stops_level_pts = mt5_points_to_price_distance(u_spec["trade_stops_level"] or 0, spec=u_spec)
    freeze_level_pts = mt5_points_to_price_distance(u_spec["trade_freeze_level"] or 0, spec=u_spec)
    spread = max(0.0, ask - bid)
    acc_cur = str(getattr(acc_info, "currency", "") or "").strip().upper()

    # Hitung offset biaya swap+komisi ke dalam poin jika seluruh komponen tersedia
    cost_neg = max(0.0, -(swap + comm))
    cost_pts = 0.0
    can_convert_money = False
    if vol > 0 and tick_value > 0 and tick_size > 0:
        can_convert_money = True
        if cost_neg > 0:
            cost_pts = (cost_neg / (tick_value * vol)) * tick_size
    elif vol > 0 and acc_cur == "USD" and contract_size > 0:
        can_convert_money = True
        if cost_neg > 0:
            cost_pts = cost_neg / (contract_size * vol)

    buffer_pts = max(tick_size, spread * 0.5, cost_pts + tick_size, 0.30)
    raw_sl = (entry + buffer_pts) if arah == "BUY" else (entry - buffer_pts)
    if tick_size > 0:
        raw_sl = round(round(raw_sl / tick_size) * tick_size, digits)
    else:
        raw_sl = round(raw_sl, digits)

    # Validasi arah & batas Stop Level / Freeze Level broker terhadap Bid (BUY) / Ask (SELL)
    if arah == "BUY":
        if raw_sl <= entry:
            return _unavail("INVALID_DIRECTION_BUY")
        dist_broker = bid - raw_sl
        if dist_broker <= 0 or dist_broker < stops_level_pts:
            return _unavail("STOP_LEVEL_RESTRICTION")
        if freeze_level_pts > 0 and dist_broker <= freeze_level_pts:
            return _unavail("FREEZE_LEVEL_RESTRICTION")
        gross_pts = raw_sl - entry
    else:
        if raw_sl >= entry:
            return _unavail("INVALID_DIRECTION_SELL")
        dist_broker = raw_sl - ask
        if dist_broker <= 0 or dist_broker < stops_level_pts:
            return _unavail("STOP_LEVEL_RESTRICTION")
        if freeze_level_pts > 0 and dist_broker <= freeze_level_pts:
            return _unavail("FREEZE_LEVEL_RESTRICTION")
        gross_pts = entry - raw_sl

    est_net_profit = None
    if has_swap and has_comm and can_convert_money:
        if tick_value > 0 and tick_size > 0:
            est_gross_money = (gross_pts / tick_size) * tick_value * vol
        else:
            est_gross_money = gross_pts * contract_size * vol
        est_net_profit = round(est_gross_money + swap + comm, 2)

    basis_str = f"spread({spread:.{digits}f} USD) + cost({cost_pts:.{digits}f} USD) + tick_size({tick_size:.{digits}f} USD)"
    return {
        "allowed": True,
        "status": "AVAILABLE",
        "label": "PROFIT PROTECTION SUGGESTION",
        "arah": arah,
        "entry_price": entry,
        "current_bid": bid,
        "current_ask": ask,
        "suggested_sl": raw_sl,
        "gross_profit_locked_pts": round(gross_pts, digits),
        "gross_profit_locked_usd": round(gross_pts, digits),
        "gross_profit_locked_mt5_points": price_distance_to_mt5_points(gross_pts, spec=u_spec),
        "estimated_net_profit": est_net_profit,
        "currency": acc_cur or None,
        "buffer_pts": round(gross_pts, digits),
        "buffer_usd": round(gross_pts, digits),
        "buffer_mt5_points": price_distance_to_mt5_points(gross_pts, spec=u_spec),
        "cost_buffer_pts": round(buffer_pts, digits),
        "digits": digits,
        "point": point,
        "trade_tick_size": tick_size,
        "calculation_basis": basis_str,
        "reason": "VALID_BROKER_CONSTRAINTS",
        "disclaimer": (
            "SL yang disarankan berpotensi mengunci sebagian keuntungan jika tersentuh dan dieksekusi. "
            "Risiko slippage, biaya transaksi, dan gap harga tetap ada. Tindakan manual oleh trader."
        ),
    }


def audit_pending_orders(orders, tick, radar_zones=None, sym_info=None):
    """Mengaudit seluruh pending order secara READ-ONLY (tanpa modifikasi/hapus order)."""
    bid = float(getattr(tick, "bid", 0.0) or 0.0) if tick is not None else 0.0
    ask = float(getattr(tick, "ask", 0.0) or 0.0) if tick is not None else 0.0
    mid = (bid + ask) / 2.0 if (bid > 0 and ask > 0) else bid
    results = []
    for o in (orders or ()):
        nm_o, arah_o = _info_tipe_pending(getattr(o, "type", 2))
        vol_o = float(getattr(o, "volume_current", getattr(o, "volume_initial", 0.0)) or 0.0)
        p_open = float(getattr(o, "price_open", 0.0) or 0.0)
        sl_o = float(getattr(o, "sl", 0.0) or 0.0)
        tp_o = float(getattr(o, "tp", 0.0) or 0.0)
        ref_px = ask if arah_o == "BUY" else bid
        if ref_px <= 0:
            ref_px = mid
        dist_o = abs(ref_px - p_open) if (ref_px > 0 and p_open > 0) else 0.0

        # Validasi posisi harga limit/stop terhadap harga pasar
        price_valid = p_open > 0 and ref_px > 0
        if price_valid:
            if "BUY LIMIT" in nm_o:
                price_valid = p_open < ask
            elif "SELL LIMIT" in nm_o:
                price_valid = p_open > bid
            elif "BUY STOP" in nm_o:
                price_valid = p_open > ask
            elif "SELL STOP" in nm_o:
                price_valid = p_open < bid

        structural_sl = None
        if sl_o <= 0:
            structural_sl = resolve_structural_sl_recommendation(arah_o, p_open, radar_zones=radar_zones)

        u_spec_o = get_symbol_unit_spec(symbol=str(getattr(o, "symbol", SYMBOL) or SYMBOL), sym_info=sym_info)
        d_o = u_spec_o["digits"]
        dist_usd = round(dist_o, d_o)
        results.append({
            "ticket": getattr(o, "ticket", None),
            "symbol": str(getattr(o, "symbol", SYMBOL) or SYMBOL),
            "order_type": nm_o,
            "arah": arah_o,
            "volume": vol_o,
            "entry_price": p_open,
            "sl": sl_o,
            "tp": tp_o,
            "distance_pt": dist_usd,
            "distance_usd": dist_usd,
            "distance_mt5_points": price_distance_to_mt5_points(dist_usd, spec=u_spec_o),
            "digits": d_o,
            "point": u_spec_o["point"],
            "trade_tick_size": u_spec_o["trade_tick_size"],
            "price_validity": "VALID_PRICE" if price_valid else "PRICE_OUT_OF_RANGE",
            "status": "ACTIVE_PENDING",
            "structural_sl": structural_sl if structural_sl is not None else "STRUCTURAL_SL_UNAVAILABLE",
        })
    return results
    cur = audits[0]["currency"]
    total = sum(float(a["profit"]) for a in audits)
    fmt = f"{total:+,.1f} {cur}" if compact else f"{total:+,.2f} {cur}"
    return fmt, audits


def resolve_structural_sl_recommendation(arah, ref_price, radar_zones=None, active_zones=None):
    """Mencari rekomendasi SL berbasis invalidasi struktur zona aktif; mengembalikan None jika tidak tersedia.

    Mode Pantau tidak pernah mengarang SL berbasis ATR sembarang ketika struktur invalidasi tidak tersedia.
    """
    want_kind = "DEMAND" if str(arah).upper() == "BUY" else "SUPPLY"
    candidates = []
    for src in (radar_zones or ()), (active_zones or ()):
        for z in src:
            if not isinstance(z, dict):
                continue
            z_kind = str(z.get("zone_type", z.get("kind", ""))).upper()
            if z_kind != want_kind:
                continue
            lo = z.get("price_low", z.get("lo"))
            hi = z.get("price_high", z.get("hi"))
            inv = z.get("invalidation", z.get("sl", z.get("jebol")))
            if not isinstance(lo, (int, float)) or not isinstance(hi, (int, float)) or not isinstance(inv, (int, float)):
                continue
            if want_kind == "DEMAND" and inv < ref_price:
                dist = 0.0 if lo <= ref_price <= hi else min(abs(ref_price - lo), abs(ref_price - hi))
                candidates.append((dist, float(inv)))
            elif want_kind == "SUPPLY" and inv > ref_price:
                dist = 0.0 if lo <= ref_price <= hi else min(abs(ref_price - lo), abs(ref_price - hi))
                candidates.append((dist, float(inv)))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return round(candidates[0][1], 2)


