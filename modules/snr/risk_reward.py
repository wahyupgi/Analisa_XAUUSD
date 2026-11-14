# -*- coding: utf-8 -*-
"""
risk_reward.py — Kalkulator Risk/Reward & SL/TP Struktural Khusus SNR Advanced Scanner.

Ketentuan Wajib (Section R):
1. Stop Loss berdasarkan invalidasi zona dan struktur harga (bukan sekadar batas zona).
2. Tambahkan buffer volatilitas yang dapat dikonfigurasi.
3. Take Profit berdasarkan support/resistance berikutnya yang relevan (bukan angka sembarangan).
4. Perhitungkan spread.
5. Perhitungkan tick size dan tick value sesuai spesifikasi simbol MT5.
6. Validasi RR minimum 1:1.5.
7. Jika SL struktural tidak tersedia, entry harus WAIT.
8. Jika TP tidak memberikan RR yang memenuhi syarat, entry harus WAIT.
"""
import math
from modules.snr.config import get_snr_config
from modules.snr.data_provider import safe_float


def round_to_tick(price, tick_size=0.01, digits=2):
    """Membulatkan harga sesuai tick_size dan digits spesifikasi simbol broker."""
    ts = max(1e-6, safe_float(tick_size, 0.01))
    steps = round(float(price) / ts)
    return round(steps * ts, int(digits))


def calculate_snr_risk_reward(
    direction,
    entry_price,
    active_zone,
    opposing_zones=None,
    trigger_candle=None,
    atr=None,
    symbol_spec=None,
    config=None,
):
    """
    Menghitung Stop Loss struktural, Take Profit struktural (berdasarkan zona lawan berikutnya),
    biaya spread, nilai risiko per lot berdasarkan tick_size & tick_value, serta validasi RR >= 1:1.5.
    """
    cfg = config or get_snr_config()
    risk_cfg = cfg.get("risk", {})
    min_rr = safe_float(risk_cfg.get("min_rr_ratio", 1.5), 1.5)
    buf_atr_mult = safe_float(risk_cfg.get("sl_volatility_buffer_atr", 0.25), 0.25)
    min_buf_pts = safe_float(risk_cfg.get("min_sl_buffer_points", 0.60), 0.60)

    spec = symbol_spec or {}
    digits = int(spec.get("digits", risk_cfg.get("default_digits", 2)))
    tick_size = max(1e-5, safe_float(spec.get("tick_size", risk_cfg.get("default_tick_size", 0.01)), 0.01))
    tick_value = max(1e-5, safe_float(spec.get("tick_value", risk_cfg.get("default_tick_value", 1.0)), 1.0))
    contract_size = safe_float(spec.get("contract_size", risk_cfg.get("default_contract_size", 100.0)), 100.0)
    spread_pts = max(0.0, safe_float(spec.get("spread_points", risk_cfg.get("default_spread_points", 0.25)), 0.25))

    dir_up = str(direction or "").upper()
    entry_ref = safe_float(entry_price, 0.0)

    if dir_up not in ("BUY", "SELL") or entry_ref <= 0 or not isinstance(active_zone, dict):
        return {
            "checked": False,
            "valid": False,
            "status": "NOT_CHECKED",
            "direction": dir_up if dir_up in ("BUY", "SELL") else "NONE",
            "entry": None,
            "effective_entry": None,
            "sl": None,
            "tp1": None,
            "tp2": None,
            "rr_tp1": None,
            "rr_tp2": None,
            "rr_display": "NOT CHECKED",
            "structural_sl_available": False,
            "structural_tp_available": False,
            "min_rr_required": min_rr,
            "spread_points": spread_pts,
            "tick_size": tick_size,
            "tick_value": tick_value,
            "reason": "Evaluasi RR belum dijalankan karena trigger entry belum terkonfirmasi.",
        }

    # Cek ketersediaan dasar SL struktural
    if active_zone.get("structural_sl_available") is False:
        return {
            "checked": True,
            "valid": False,
            "status": "WAIT",
            "direction": dir_up,
            "entry": round_to_tick(entry_ref, tick_size, digits),
            "effective_entry": None,
            "sl": None,
            "tp1": None,
            "tp2": None,
            "rr_tp1": 0.0,
            "rr_tp2": 0.0,
            "rr_display": "SL N/A",
            "structural_sl_available": False,
            "structural_tp_available": False,
            "min_rr_required": min_rr,
            "spread_points": spread_pts,
            "tick_size": tick_size,
            "tick_value": tick_value,
            "reason": "SL struktural tidak tersedia; entry wajib WAIT.",
        }

    z_low = safe_float(active_zone.get("low"), 0.0)
    z_high = safe_float(active_zone.get("high"), 0.0)
    ref_atr = max(0.20, safe_float(atr, safe_float(active_zone.get("ref_atr"), 2.0)))

    swings = active_zone.get("source_swings") or []
    origin_piv = active_zone.get("origin_pivot") or (swings[0] if swings else {})

    if z_low <= 0 or z_high <= 0 or z_high < z_low:
        return {
            "checked": True,
            "valid": False,
            "status": "WAIT",
            "direction": dir_up,
            "entry": round_to_tick(entry_ref, tick_size, digits),
            "effective_entry": None,
            "sl": None,
            "tp1": None,
            "tp2": None,
            "rr_tp1": 0.0,
            "rr_tp2": 0.0,
            "rr_display": "SL N/A",
            "structural_sl_available": False,
            "structural_tp_available": False,
            "min_rr_required": min_rr,
            "spread_points": spread_pts,
            "tick_size": tick_size,
            "tick_value": tick_value,
            "reason": "Batas invalidasi zona tidak valid sehingga SL struktural tidak tersedia.",
        }

    vol_buffer = max(min_buf_pts, ref_atr * buf_atr_mult)

    if dir_up == "BUY":
        # Entry BUY dieksekusi pada Ask (entry_ref + spread)
        eff_entry = round_to_tick(entry_ref + spread_pts, tick_size, digits)
        struct_anchors = [z_low]
        piv_low = safe_float(origin_piv.get("low", origin_piv.get("price")), 0.0)
        if piv_low > 0:
            struct_anchors.append(piv_low)
        if isinstance(trigger_candle, dict) and safe_float(trigger_candle.get("low"), 0.0) > 0:
            struct_anchors.append(safe_float(trigger_candle["low"], z_low))

        structural_invalidation = min(struct_anchors)
        if structural_invalidation <= 0 or structural_invalidation >= eff_entry:
            return {
                "checked": True,
                "valid": False,
                "status": "WAIT",
                "direction": "BUY",
                "entry": round_to_tick(entry_ref, tick_size, digits),
                "effective_entry": eff_entry,
                "sl": None,
                "tp1": None,
                "tp2": None,
                "rr_tp1": 0.0,
                "rr_tp2": 0.0,
                "rr_display": "SL INVALID",
                "structural_sl_available": False,
                "structural_tp_available": False,
                "min_rr_required": min_rr,
                "spread_points": spread_pts,
                "tick_size": tick_size,
                "tick_value": tick_value,
                "reason": "Titik invalidasi struktural BUY berada di atas harga entry; SL tidak valid.",
            }

        sl_price = round_to_tick(structural_invalidation - vol_buffer - spread_pts, tick_size, digits)
        risk_pts = eff_entry - sl_price
        if risk_pts <= tick_size:
            return {
                "checked": True,
                "valid": False,
                "status": "WAIT",
                "direction": "BUY",
                "entry": round_to_tick(entry_ref, tick_size, digits),
                "effective_entry": eff_entry,
                "sl": sl_price,
                "tp1": None,
                "tp2": None,
                "rr_tp1": 0.0,
                "rr_tp2": 0.0,
                "rr_display": "SL TOO CLOSE",
                "structural_sl_available": False,
                "structural_tp_available": False,
                "min_rr_required": min_rr,
                "spread_points": spread_pts,
                "tick_size": tick_size,
                "tick_value": tick_value,
                "reason": "Jarak SL struktural terlalu sempit.",
            }

        # Cari Resistance berikutnya di atas eff_entry sebagai TP struktural
        opp_candidates = []
        for oz in opposing_zones or []:
            oz_low = safe_float(oz.get("low"), 0.0)
            oz_high = safe_float(oz.get("high"), oz_low)
            if oz_low > (eff_entry + spread_pts):
                opp_candidates.append((oz_low, oz_high, oz))
        opp_candidates.sort(key=lambda x: x[0])

        if not opp_candidates:
            return {
                "checked": True,
                "valid": False,
                "status": "WAIT",
                "direction": "BUY",
                "entry": round_to_tick(entry_ref, tick_size, digits),
                "effective_entry": eff_entry,
                "sl": sl_price,
                "tp1": None,
                "tp2": None,
                "rr_tp1": 0.0,
                "rr_tp2": 0.0,
                "rr_display": "TP N/A",
                "structural_sl_available": True,
                "structural_tp_available": False,
                "min_rr_required": min_rr,
                "spread_points": spread_pts,
                "tick_size": tick_size,
                "tick_value": tick_value,
                "reason": "Zona Resistance struktural berikutnya untuk Take Profit BUY tidak tersedia.",
            }

        # TP1 adalah batas bawah Resistance relevan terdekat di atas entry
        first_res_low, first_res_high, first_res_obj = opp_candidates[0]
        tp1_price = round_to_tick(first_res_low - spread_pts, tick_size, digits)
        if len(opp_candidates) >= 2:
            tp2_price = round_to_tick(opp_candidates[1][0] - spread_pts, tick_size, digits)
        else:
            tp2_price = round_to_tick(first_res_high, tick_size, digits)

        reward_tp1 = max(0.0, tp1_price - eff_entry)
        reward_tp2 = max(reward_tp1, tp2_price - eff_entry)

    else:  # SELL
        # Entry SELL dieksekusi pada Bid (entry_ref), namun penutupan SL/TP menggunakan Ask (+ spread)
        eff_entry = round_to_tick(entry_ref, tick_size, digits)
        struct_anchors = [z_high]
        piv_high = safe_float(origin_piv.get("high", origin_piv.get("price")), 0.0)
        if piv_high > 0:
            struct_anchors.append(piv_high)
        if isinstance(trigger_candle, dict) and safe_float(trigger_candle.get("high"), 0.0) > 0:
            struct_anchors.append(safe_float(trigger_candle["high"], z_high))

        structural_invalidation = max(struct_anchors)
        if structural_invalidation <= eff_entry:
            return {
                "checked": True,
                "valid": False,
                "status": "WAIT",
                "direction": "SELL",
                "entry": round_to_tick(entry_ref, tick_size, digits),
                "effective_entry": eff_entry,
                "sl": None,
                "tp1": None,
                "tp2": None,
                "rr_tp1": 0.0,
                "rr_tp2": 0.0,
                "rr_display": "SL INVALID",
                "structural_sl_available": False,
                "structural_tp_available": False,
                "min_rr_required": min_rr,
                "spread_points": spread_pts,
                "tick_size": tick_size,
                "tick_value": tick_value,
                "reason": "Titik invalidasi struktural SELL berada di bawah harga entry; SL tidak valid.",
            }

        sl_price = round_to_tick(structural_invalidation + vol_buffer + spread_pts, tick_size, digits)
        risk_pts = sl_price - eff_entry
        if risk_pts <= tick_size:
            return {
                "checked": True,
                "valid": False,
                "status": "WAIT",
                "direction": "SELL",
                "entry": round_to_tick(entry_ref, tick_size, digits),
                "effective_entry": eff_entry,
                "sl": sl_price,
                "tp1": None,
                "tp2": None,
                "rr_tp1": 0.0,
                "rr_tp2": 0.0,
                "rr_display": "SL TOO CLOSE",
                "structural_sl_available": False,
                "structural_tp_available": False,
                "min_rr_required": min_rr,
                "spread_points": spread_pts,
                "tick_size": tick_size,
                "tick_value": tick_value,
                "reason": "Jarak SL struktural terlalu sempit.",
            }

        # Cari Support berikutnya di bawah eff_entry sebagai TP struktural
        opp_candidates = []
        for oz in opposing_zones or []:
            oz_low = safe_float(oz.get("low"), 0.0)
            oz_high = safe_float(oz.get("high"), oz_low)
            if oz_high < (eff_entry - spread_pts):
                opp_candidates.append((oz_high, oz_low, oz))
        opp_candidates.sort(key=lambda x: x[0], reverse=True)

        if not opp_candidates:
            return {
                "checked": True,
                "valid": False,
                "status": "WAIT",
                "direction": "SELL",
                "entry": round_to_tick(entry_ref, tick_size, digits),
                "effective_entry": eff_entry,
                "sl": sl_price,
                "tp1": None,
                "tp2": None,
                "rr_tp1": 0.0,
                "rr_tp2": 0.0,
                "rr_display": "TP N/A",
                "structural_sl_available": True,
                "structural_tp_available": False,
                "min_rr_required": min_rr,
                "spread_points": spread_pts,
                "tick_size": tick_size,
                "tick_value": tick_value,
                "reason": "Zona Support struktural berikutnya untuk Take Profit SELL tidak tersedia.",
            }

        first_sup_high, first_sup_low, first_sup_obj = opp_candidates[0]
        tp1_price = round_to_tick(first_sup_high + spread_pts, tick_size, digits)
        if len(opp_candidates) >= 2:
            tp2_price = round_to_tick(opp_candidates[1][0] + spread_pts, tick_size, digits)
        else:
            tp2_price = round_to_tick(first_sup_low, tick_size, digits)

        reward_tp1 = max(0.0, eff_entry - tp1_price)
        reward_tp2 = max(reward_tp1, eff_entry - tp2_price)

    rr_tp1 = round(reward_tp1 / max(1e-6, risk_pts), 2)
    rr_tp2 = round(reward_tp2 / max(1e-6, risk_pts), 2)

    risk_ticks = risk_pts / tick_size
    reward_ticks_tp1 = reward_tp1 / tick_size
    risk_usd_1lot = round(risk_ticks * tick_value, 2)
    reward_usd_1lot = round(reward_ticks_tp1 * tick_value, 2)

    rr_ok = rr_tp1 >= min_rr
    status_rr = "VALID" if rr_ok else "WAIT"
    reason_rr = (
        f"RR memenuhi syarat (1:{rr_tp1:.2f} >= 1:{min_rr:.2f}, SL={sl_price:.2f}, TP1={tp1_price:.2f})."
        if rr_ok
        else f"RR di bawah standar minimal (1:{rr_tp1:.2f} < 1:{min_rr:.2f}); wajib WAIT."
    )

    return {
        "checked": True,
        "valid": bool(rr_ok),
        "status": status_rr,
        "direction": dir_up,
        "entry": round_to_tick(entry_ref, tick_size, digits),
        "effective_entry": eff_entry,
        "structural_invalidation": round(structural_invalidation, digits),
        "volatility_buffer": round(vol_buffer, digits),
        "sl": sl_price,
        "tp1": tp1_price,
        "tp2": tp2_price,
        "risk_points": round(risk_pts, digits),
        "reward_tp1_points": round(reward_tp1, digits),
        "reward_tp2_points": round(reward_tp2, digits),
        "rr_tp1": rr_tp1,
        "rr_tp2": rr_tp2,
        "rr_display": f"1 : {rr_tp1:.2f}",
        "structural_sl_available": True,
        "structural_tp_available": True,
        "min_rr_required": min_rr,
        "spread_points": spread_pts,
        "tick_size": tick_size,
        "tick_value": tick_value,
        "contract_size": contract_size,
        "risk_usd_per_lot": risk_usd_1lot,
        "reward_usd_per_lot": reward_usd_1lot,
        "reason": reason_rr,
    }

