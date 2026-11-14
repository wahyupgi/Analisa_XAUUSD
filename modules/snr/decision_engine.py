# -*- coding: utf-8 -*-
"""
decision_engine.py — Mesin Pengambilan Keputusan Akhir Independen (BUY / SELL / WAIT / INVALID)
untuk SNR Advanced Scanner.
"""


def build_snr_final_decision(entry_validation, market_bias=None, price_action=None):
    """
    Menyusun objek keputusan akhir SNR dari hasil entry_validator, market_bias, dan price_action.
    Tidak pernah memaksakan sinyal jika syarat tidak terpenuhi penuh.
    """
    ev = entry_validation or {}
    mb = market_bias or {}
    pa = price_action or {}

    status = str(ev.get("final_status", "WAIT")).upper()
    if status not in ("BUY", "SELL", "WAIT", "INVALID"):
        status = "WAIT"

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
        "signal": status,
        "action": status,
        "direction": direction,
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

