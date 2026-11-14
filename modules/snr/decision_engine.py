# -*- coding: utf-8 -*-
"""
decision_engine.py — Mesin Pengambilan Keputusan Akhir Independen (BUY / SELL / WAIT / INVALID)
untuk SNR Advanced Scanner.
"""


def build_snr_final_decision(
    entry_validation,
    market_bias=None,
    price_action=None,
    market_status="OPEN",
    snr_analysis="AVAILABLE",
):
    """
    Menyusun objek keputusan akhir SNR dari hasil entry_validator, market_bias, dan price_action.
    Tidak pernah memaksakan sinyal jika syarat tidak terpenuhi penuh.
    Ketika market_status == 'CLOSED', analisis historis tetap tersedia namun sinyal live entry
    dikunci pada 'WAIT — MARKET CLOSED' tanpa mengklaim eksekusi live.
    """
    ev = entry_validation or {}
    mb = market_bias or {}
    pa = price_action or {}

    mkt_st = str(market_status or "OPEN").strip().upper()
    if mkt_st in ("MARKET OPEN", "MARKET_OPEN"):
        mkt_st = "OPEN"
    elif mkt_st in ("MARKET CLOSED", "MARKET_CLOSED"):
        mkt_st = "CLOSED"
    elif mkt_st in ("UNAVAILABLE", "NO_DATA", "DATA UNAVAILABLE"):
        mkt_st = "DATA UNAVAILABLE"

    snr_an = str(snr_analysis or ("DATA UNAVAILABLE" if mkt_st == "DATA UNAVAILABLE" else "AVAILABLE")).strip().upper()

    raw_status = str(ev.get("final_status", "WAIT")).upper()
    if raw_status not in ("BUY", "SELL", "WAIT", "INVALID"):
        raw_status = "WAIT"

    if mkt_st == "DATA UNAVAILABLE" or snr_an == "DATA UNAVAILABLE":
        reason_unavail = "DATA UNAVAILABLE — Data candle historis MT5 tidak tersedia."
        return {
            "method": "SNR",
            "market_status": "DATA UNAVAILABLE",
            "snr_analysis": "DATA UNAVAILABLE",
            "final_status": "DATA UNAVAILABLE",
            "live_entry": "WAIT — DATA UNAVAILABLE",
            "signal": "WAIT",
            "action": "WAIT",
            "direction": "NONE",
            "historical_signal": "DATA UNAVAILABLE",
            "setup_type": "NONE",
            "zone_label": "DATA UNAVAILABLE",
            "target_zone": None,
            "trigger_status": "DATA UNAVAILABLE",
            "buy_status": "DATA UNAVAILABLE",
            "sell_status": "DATA UNAVAILABLE",
            "entry": None,
            "effective_entry": None,
            "sl": None,
            "tp1": None,
            "tp2": None,
            "rr_tp1": None,
            "rr_tp2": None,
            "rr_display": "DATA UNAVAILABLE",
            "market_bias": "DATA UNAVAILABLE",
            "snr_status": "DATA UNAVAILABLE",
            "price_action": pa,
            "reason": reason_unavail,
            "wait_reasons": [reason_unavail],
        }

    if mkt_st == "CLOSED":
        closed_reason = (
            "MARKET CLOSED — Analisis teknikal SNR menggunakan data historis terakhir; "
            "entry baru tetap WAIT sampai market buka dan kondisi harga diperbarui."
        )
        wait_list = [closed_reason]
        for wr in (ev.get("wait_reasons") or []):
            if wr and wr not in wait_list:
                wait_list.append(wr)

        return {
            "method": "SNR",
            "market_status": "CLOSED",
            "snr_analysis": "AVAILABLE",
            "final_status": "MARKET CLOSED",
            "live_entry": "WAIT — MARKET CLOSED",
            "signal": "WAIT",
            "action": "WAIT",
            "direction": "NONE",
            "historical_signal": raw_status,
            "setup_type": ev.get("setup_type", "NONE"),
            "zone_label": ev.get("zone_label", "N/A"),
            "target_zone": ev.get("target_zone"),
            "trigger_status": "WAIT — MARKET CLOSED",
            "buy_status": "WAIT — MARKET CLOSED",
            "sell_status": "WAIT — MARKET CLOSED",
            "entry": None,
            "effective_entry": None,
            "sl": None,
            "tp1": None,
            "tp2": None,
            "rr_tp1": None,
            "rr_tp2": None,
            "rr_display": "NOT CHECKED (MARKET CLOSED)",
            "market_bias": mb.get("overall_bias", "UNCONFIRMED"),
            "snr_status": mb.get("snr_status", "UNCONFIRMED"),
            "price_action": pa,
            "reason": closed_reason,
            "wait_reasons": wait_list,
        }

    status = raw_status
    direction = str(ev.get("direction", "NONE")).upper()
    if status in ("WAIT", "INVALID"):
        direction = "NONE"

    rr_calc = ev.get("rr_calculation") or {}
    is_actionable = status in ("BUY", "SELL") and bool(rr_calc.get("valid", False))

    entry_val = rr_calc.get("entry") if is_actionable else None
    eff_entry = rr_calc.get("effective_entry") if is_actionable else None
    sl_val = rr_calc.get("sl") if is_actionable else None
    tp1_val = rr_calc.get("tp1") if is_actionable else None
    tp2_val = rr_calc.get("tp2") if is_actionable else None
    rr_tp1 = rr_calc.get("rr_tp1") if is_actionable else None
    rr_tp2 = rr_calc.get("rr_tp2") if is_actionable else None
    rr_display = ev.get("rr_status", rr_calc.get("rr_display", "NOT CHECKED"))

    reason = ev.get("reason") or "Waiting for valid SNR confirmation."
    wait_reasons = list(ev.get("wait_reasons") or [])
    if status in ("WAIT", "INVALID") and not wait_reasons:
        wait_reasons = [reason]

    return {
        "method": "SNR",
        "market_status": "OPEN",
        "snr_analysis": "AVAILABLE",
        "final_status": status,
        "live_entry": status,
        "signal": status,
        "action": status,
        "direction": direction,
        "historical_signal": status,
        "setup_type": ev.get("setup_type", "NONE"),
        "zone_label": ev.get("zone_label", "N/A"),
        "target_zone": ev.get("target_zone"),
        "trigger_status": ev.get("trigger_status", "NOT CONFIRMED"),
        "buy_status": ev.get("buy_status", "WAIT"),
        "sell_status": ev.get("sell_status", "WAIT"),
        "entry": entry_val,
        "effective_entry": eff_entry,
        "sl": sl_val,
        "tp1": tp1_val,
        "tp2": tp2_val,
        "rr_tp1": rr_tp1,
        "rr_tp2": rr_tp2,
        "rr_display": rr_display,
        "market_bias": mb.get("overall_bias", "UNCONFIRMED"),
        "snr_status": mb.get("snr_status", "UNCONFIRMED"),
        "price_action": pa,
        "reason": reason,
        "wait_reasons": wait_reasons,
    }

