# -*- coding: utf-8 -*-
"""
entry_validator.py — Validator Entry Independen untuk SNR Advanced Scanner.

Mengevaluasi 4 skenario setup resmi (Section Q):
1A. BUY — SUPPORT REJECTION
1B. BUY — RESISTANCE BREAKOUT + RETEST
2A. SELL — RESISTANCE REJECTION
2B. SELL — SUPPORT BREAKDOWN + RETEST

Menghasilkan status: BUY, SELL, WAIT, atau INVALID.
"""
from modules.snr.breakout_detection import detect_zone_breakout
from modules.snr.config import get_snr_config
from modules.snr.data_provider import prepare_snr_frame, safe_float
from modules.snr.momentum_analysis import analyze_momentum
from modules.snr.rejection_detection import detect_zone_rejection
from modules.snr.retest_detection import detect_zone_retest
from modules.snr.risk_reward import calculate_snr_risk_reward


def _select_target_zone_for_monitoring(ranked_zones, market_bias, current_price):
    """Memilih zona fokus utama (misal SUPPORT S1 atau RESISTANCE R1) untuk evaluasi entry."""
    active_sups = (ranked_zones or {}).get("active_supports") or []
    active_ress = (ranked_zones or {}).get("active_resistances") or []
    broken_zones = (ranked_zones or {}).get("broken_zones") or []
    invalid_zones = (ranked_zones or {}).get("invalidated_zones") or []

    cp = safe_float(current_price, 0.0)
    bias_str = str((market_bias or {}).get("overall_bias", "SIDEWAYS")).upper()

    s1 = active_sups[0] if active_sups else None
    r1 = active_ress[0] if active_ress else None

    if s1 and r1:
        d_s1 = safe_float(s1.get("distance_points"), abs(cp - safe_float(s1.get("high"), cp)))
        d_r1 = safe_float(r1.get("distance_points"), abs(cp - safe_float(r1.get("low"), cp)))
        if s1.get("price_inside_zone"):
            return s1, f"SUPPORT {s1.get('rank_label', 'S1')}"
        if r1.get("price_inside_zone"):
            return r1, f"RESISTANCE {r1.get('rank_label', 'R1')}"
        if bias_str == "BEARISH" and d_r1 <= (d_s1 * 1.45):
            return r1, f"RESISTANCE {r1.get('rank_label', 'R1')}"
        if bias_str == "BULLISH" and d_s1 <= (d_r1 * 1.45):
            return s1, f"SUPPORT {s1.get('rank_label', 'S1')}"
        if d_r1 < d_s1:
            return r1, f"RESISTANCE {r1.get('rank_label', 'R1')}"
        return s1, f"SUPPORT {s1.get('rank_label', 'S1')}"
    if r1:
        return r1, f"RESISTANCE {r1.get('rank_label', 'R1')}"
    if s1:
        return s1, f"SUPPORT {s1.get('rank_label', 'S1')}"
    if broken_zones:
        bz = broken_zones[0]
        return bz, f"BROKEN {bz.get('zone_type', 'ZONE')}"
    if invalid_zones:
        iz = invalid_zones[0]
        return iz, f"INVALIDATED {iz.get('zone_type', 'ZONE')}"
    return None, "N/A"


def validate_snr_entry(
    ranked_zones,
    m5_df,
    market_bias=None,
    current_price=None,
    symbol_spec=None,
    config=None,
    has_unclosed_last_bar=False,
):
    """
    Memvalidasi seluruh syarat BUY SETUP (1A Support Rejection, 1B Resistance Breakout+Retest)
    dan SELL SETUP (2A Resistance Rejection, 2B Support Breakdown+Retest) pada candle M5 CLOSED.
    """
    cfg = config or get_snr_config()
    risk_cfg = cfg.get("risk", {})
    max_entry_dist_atr = safe_float(risk_cfg.get("max_entry_distance_atr", 0.65), 0.65)

    clean_m5, _ = prepare_snr_frame(m5_df, has_unclosed_last_bar=has_unclosed_last_bar)
    cp = safe_float(current_price, 0.0)
    if cp <= 0 and not clean_m5.empty:
        cp = safe_float(clean_m5["Close"].iloc[-1], 0.0)

    active_sups = (ranked_zones or {}).get("active_supports") or []
    active_ress = (ranked_zones or {}).get("active_resistances") or []
    broken_zones = (ranked_zones or {}).get("broken_zones") or []
    invalid_zones = (ranked_zones or {}).get("invalidated_zones") or []

    focus_zone, focus_label = _select_target_zone_for_monitoring(ranked_zones, market_bias, cp)

    if clean_m5.empty or cp <= 0:
        return {
            "final_status": "WAIT",
            "buy_status": "WAIT",
            "sell_status": "WAIT",
            "setup_type": "NONE",
            "direction": "NONE",
            "zone_label": focus_label,
            "target_zone": focus_zone,
            "trigger_status": "NOT CONFIRMED",
            "rr_status": "NOT CHECKED",
            "rr_calculation": {
                "checked": False,
                "valid": False,
                "rr_display": "NOT CHECKED",
                "reason": "Data M5 closed belum tersedia.",
            },
            "checklist": {},
            "wait_reasons": ["Data candle M5 closed belum tersedia untuk konfirmasi trigger entry."],
            "reason": "Waiting for valid M5 closed candle data.",
        }

    m5_mom = analyze_momentum(clean_m5, timeframe="M5", config=cfg, has_unclosed_last_bar=False)
    m5_atr = max(0.20, safe_float(clean_m5["ATR14"].iloc[-1], 1.5))
    last_m5_open = safe_float(clean_m5["Open"].iloc[-1], cp)
    last_m5_close = safe_float(clean_m5["Close"].iloc[-1], cp)
    last_m5_high = safe_float(clean_m5["High"].iloc[-1], cp)
    last_m5_low = safe_float(clean_m5["Low"].iloc[-1], cp)

    m5_bull_close = last_m5_close > last_m5_open
    m5_bear_close = last_m5_close < last_m5_open
    m5_bull_mom = m5_mom.get("direction") == "BULLISH" and not bool(m5_mom.get("momentum_weakening", False))
    m5_bear_mom = m5_mom.get("direction") == "BEARISH" and not bool(m5_mom.get("momentum_weakening", False))

    wait_reasons = []
    invalid_reasons = []

    # ------------------------------------------------------------------
    # 1A. Evaluasi BUY — SUPPORT REJECTION pada S1 (jika tersedia)
    # ------------------------------------------------------------------
    s1 = active_sups[0] if active_sups else None
    if s1 is not None:
        s1_low = safe_float(s1.get("low"), 0.0)
        s1_high = safe_float(s1.get("high"), s1_low)
        s1_atr = max(m5_atr, safe_float(s1.get("ref_atr"), m5_atr))

        # Harga memasuki / menguji support dalam 4 candle M5 terakhir
        recent_min_low = float(clean_m5["Low"].tail(4).min())
        entered_support = (recent_min_low <= (s1_high + 0.20 * m5_atr)) and (last_m5_close >= s1_low)
        dist_from_s1_atr = max(0.0, last_m5_close - s1_high) / s1_atr
        location_reasonable = entered_support and (dist_from_s1_atr <= max_entry_dist_atr)

        rej_s1 = detect_zone_rejection(s1, clean_m5, config=cfg, has_unclosed_last_bar=False)
        valid_bull_rej = (rej_s1.get("status") == "BULLISH_REJECTION") and bool(rej_s1.get("confirmed", False))

        if rej_s1.get("status") == "FALSE_REJECTION" and rej_s1.get("in_zone_context"):
            invalid_reasons.append(f"Support {s1.get('rank_label', 'S1')} mengalami false rejection / ditembus.")

        if entered_support and valid_bull_rej and m5_bull_close and m5_bull_mom and location_reasonable:
            rr_res = calculate_snr_risk_reward(
                direction="BUY",
                entry_price=cp,
                active_zone=s1,
                opposing_zones=active_ress,
                trigger_candle={"low": recent_min_low, "high": last_m5_high},
                atr=m5_atr,
                symbol_spec=symbol_spec,
                config=cfg,
            )
            checklist = {
                "price_entered_support": True,
                "valid_bullish_rejection": True,
                "m5_bullish_close": True,
                "momentum_confirmed": True,
                "reasonable_entry_location": True,
                "structural_sl_valid": bool(rr_res.get("structural_sl_available", False)),
                "min_rr_met": bool(rr_res.get("valid", False)),
            }
            if rr_res.get("valid", False):
                return {
                    "final_status": "BUY",
                    "buy_status": "CONFIRMED",
                    "sell_status": "WAIT",
                    "setup_type": "SUPPORT_REJECTION",
                    "direction": "BUY",
                    "zone_label": f"SUPPORT {s1.get('rank_label', 'S1')}",
                    "target_zone": s1,
                    "trigger_status": "SUPPORT REJECTION CONFIRMED",
                    "rr_status": rr_res.get("rr_display", "VALID"),
                    "rr_calculation": rr_res,
                    "rejection_detail": rej_s1,
                    "checklist": checklist,
                    "wait_reasons": [],
                    "reason": (
                        f"BUY terkonfirmasi melalui Support Rejection pada {s1.get('rank_label', 'S1')} "
                        f"({s1_low:.2f}-{s1_high:.2f}) dengan RR {rr_res.get('rr_display')}."
                    ),
                }
            else:
                # Trigger sudah terkonfirmasi, namun gagal pada syarat SL struktural atau RR minimal -> WAIT
                wait_reasons.append(rr_res.get("reason", "RR belum memenuhi syarat minimal 1:1.5."))
                return {
                    "final_status": "WAIT",
                    "buy_status": "WAIT",
                    "sell_status": "WAIT",
                    "setup_type": "SUPPORT_REJECTION_BLOCKED_BY_RR",
                    "direction": "NONE",
                    "zone_label": f"SUPPORT {s1.get('rank_label', 'S1')}",
                    "target_zone": s1,
                    "trigger_status": "CONFIRMED (RR/SL BLOCKED)",
                    "rr_status": rr_res.get("rr_display", "FAILED"),
                    "rr_calculation": rr_res,
                    "rejection_detail": rej_s1,
                    "checklist": checklist,
                    "wait_reasons": wait_reasons,
                    "reason": rr_res.get("reason", "Trigger BUY muncul namun RR/SL tidak memenuhi syarat."),
                }
        else:
            if not entered_support:
                wait_reasons.append(f"Harga ({cp:.2f}) belum memasuki area Support {s1.get('rank_label', 'S1')} ({s1_low:.2f}-{s1_high:.2f}).")
            elif not valid_bull_rej:
                wait_reasons.append(f"Belum ada candle rejection bullish terkonfirmasi pada Support {s1.get('rank_label', 'S1')}.")
            elif not m5_bull_close:
                wait_reasons.append("Candle M5 terakhir belum close bullish.")
            elif not m5_bull_mom:
                wait_reasons.append("Momentum M5 belum mendukung kenaikan (atau sedang melemah).")
            elif not location_reasonable:
                wait_reasons.append(f"Jarak harga dari Support terlalu jauh ({dist_from_s1_atr:.2f} ATR > {max_entry_dist_atr:.2f} ATR).")

    # ------------------------------------------------------------------
    # 2A. Evaluasi SELL — RESISTANCE REJECTION pada R1 (jika tersedia)
    # ------------------------------------------------------------------
    r1 = active_ress[0] if active_ress else None
    if r1 is not None:
        r1_low = safe_float(r1.get("low"), 0.0)
        r1_high = safe_float(r1.get("high"), r1_low)
        r1_atr = max(m5_atr, safe_float(r1.get("ref_atr"), m5_atr))

        recent_max_high = float(clean_m5["High"].tail(4).max())
        entered_resistance = (recent_max_high >= (r1_low - 0.20 * m5_atr)) and (last_m5_close <= r1_high)
        dist_from_r1_atr = max(0.0, r1_low - last_m5_close) / r1_atr
        location_reasonable = entered_resistance and (dist_from_r1_atr <= max_entry_dist_atr)

        rej_r1 = detect_zone_rejection(r1, clean_m5, config=cfg, has_unclosed_last_bar=False)
        valid_bear_rej = (rej_r1.get("status") == "BEARISH_REJECTION") and bool(rej_r1.get("confirmed", False))

        if rej_r1.get("status") == "FALSE_REJECTION" and rej_r1.get("in_zone_context"):
            invalid_reasons.append(f"Resistance {r1.get('rank_label', 'R1')} mengalami false rejection / ditembus.")

        if entered_resistance and valid_bear_rej and m5_bear_close and m5_bear_mom and location_reasonable:
            rr_res = calculate_snr_risk_reward(
                direction="SELL",
                entry_price=cp,
                active_zone=r1,
                opposing_zones=active_sups,
                trigger_candle={"low": last_m5_low, "high": recent_max_high},
                atr=m5_atr,
                symbol_spec=symbol_spec,
                config=cfg,
            )
            checklist = {
                "price_entered_resistance": True,
                "valid_bearish_rejection": True,
                "m5_bearish_close": True,
                "momentum_confirmed": True,
                "reasonable_entry_location": True,
                "structural_sl_valid": bool(rr_res.get("structural_sl_available", False)),
                "min_rr_met": bool(rr_res.get("valid", False)),
            }
            if rr_res.get("valid", False):
                return {
                    "final_status": "SELL",
                    "buy_status": "WAIT",
                    "sell_status": "CONFIRMED",
                    "setup_type": "RESISTANCE_REJECTION",
                    "direction": "SELL",
                    "zone_label": f"RESISTANCE {r1.get('rank_label', 'R1')}",
                    "target_zone": r1,
                    "trigger_status": "RESISTANCE REJECTION CONFIRMED",
                    "rr_status": rr_res.get("rr_display", "VALID"),
                    "rr_calculation": rr_res,
                    "rejection_detail": rej_r1,
                    "checklist": checklist,
                    "wait_reasons": [],
                    "reason": (
                        f"SELL terkonfirmasi melalui Resistance Rejection pada {r1.get('rank_label', 'R1')} "
                        f"({r1_low:.2f}-{r1_high:.2f}) dengan RR {rr_res.get('rr_display')}."
                    ),
                }
            else:
                wait_reasons.append(rr_res.get("reason", "RR belum memenuhi syarat minimal 1:1.5."))
                return {
                    "final_status": "WAIT",
                    "buy_status": "WAIT",
                    "sell_status": "WAIT",
                    "setup_type": "RESISTANCE_REJECTION_BLOCKED_BY_RR",
                    "direction": "NONE",
                    "zone_label": f"RESISTANCE {r1.get('rank_label', 'R1')}",
                    "target_zone": r1,
                    "trigger_status": "CONFIRMED (RR/SL BLOCKED)",
                    "rr_status": rr_res.get("rr_display", "FAILED"),
                    "rr_calculation": rr_res,
                    "rejection_detail": rej_r1,
                    "checklist": checklist,
                    "wait_reasons": wait_reasons,
                    "reason": rr_res.get("reason", "Trigger SELL muncul namun RR/SL tidak memenuhi syarat."),
                }
        else:
            if not entered_resistance:
                wait_reasons.append(f"Harga ({cp:.2f}) belum memasuki area Resistance {r1.get('rank_label', 'R1')} ({r1_low:.2f}-{r1_high:.2f}).")
            elif not valid_bear_rej:
                wait_reasons.append(f"Belum ada candle rejection bearish terkonfirmasi pada Resistance {r1.get('rank_label', 'R1')}.")
            elif not m5_bear_close:
                wait_reasons.append("Candle M5 terakhir belum close bearish.")
            elif not m5_bear_mom:
                wait_reasons.append("Momentum M5 belum mendukung penurunan (atau sedang melemah).")
            elif not location_reasonable:
                wait_reasons.append(f"Jarak harga dari Resistance terlalu jauh ({dist_from_r1_atr:.2f} ATR > {max_entry_dist_atr:.2f} ATR).")

    # ------------------------------------------------------------------
    # 1B & 2B. Evaluasi BREAKOUT + RETEST (Role Reversal) pada zona yang ditembus
    # ------------------------------------------------------------------
    retest_candidates = list(broken_zones) + list(active_ress[:1]) + list(active_sups[:1])
    for bz in retest_candidates:
        z_type = str(bz.get("zone_type", "RESISTANCE")).upper()
        bo_res = bz.get("breakout_info") or detect_zone_breakout(bz, clean_m5, momentum_result=m5_mom, config=cfg)
        rt_res = bz.get("retest_info") or detect_zone_retest(bz, clean_m5, breakout_result=bo_res, config=cfg)

        if bo_res.get("status") == "FALSE_BREAKOUT":
            invalid_reasons.append(bo_res.get("reason", f"False breakout pada {z_type}."))
            continue
        if rt_res.get("status") == "RETEST_FAILED":
            invalid_reasons.append(rt_res.get("reason", f"Retest gagal pada {z_type}."))
            continue

        # 1B. BUY — RESISTANCE BREAKOUT + RETEST BERHASIL
        if z_type == "RESISTANCE" and rt_res.get("status") == "RETEST_CONFIRMED" and rt_res.get("direction") == "BULLISH":
            dist_atr = max(0.0, cp - safe_float(bz.get("high"), cp)) / max(m5_atr, safe_float(bz.get("ref_atr"), m5_atr))
            if m5_bull_close and m5_bull_mom and dist_atr <= max_entry_dist_atr:
                rr_res = calculate_snr_risk_reward(
                    direction="BUY",
                    entry_price=cp,
                    active_zone=bz,
                    opposing_zones=active_ress,
                    atr=m5_atr,
                    symbol_spec=symbol_spec,
                    config=cfg,
                )
                if rr_res.get("valid", False):
                    return {
                        "final_status": "BUY",
                        "buy_status": "CONFIRMED",
                        "sell_status": "WAIT",
                        "setup_type": "RESISTANCE_BREAKOUT_RETEST",
                        "direction": "BUY",
                        "zone_label": f"FLIPPED SUPPORT ({bz.get('low', 0):.2f}-{bz.get('high', 0):.2f})",
                        "target_zone": bz,
                        "trigger_status": "BREAKOUT + RETEST CONFIRMED",
                        "rr_status": rr_res.get("rr_display", "VALID"),
                        "rr_calculation": rr_res,
                        "checklist": {
                            "valid_breakout": True,
                            "retest_confirmed": True,
                            "bullish_confirmation": True,
                            "structural_sl_valid": True,
                            "min_rr_met": True,
                        },
                        "wait_reasons": [],
                        "reason": (
                            f"BUY terkonfirmasi melalui Resistance Breakout + Retest pada "
                            f"{bz.get('low', 0):.2f}-{bz.get('high', 0):.2f} dengan RR {rr_res.get('rr_display')}."
                        ),
                    }
                else:
                    wait_reasons.append(rr_res.get("reason", "RR retest BUY di bawah standar."))
                    return {
                        "final_status": "WAIT",
                        "buy_status": "WAIT",
                        "sell_status": "WAIT",
                        "setup_type": "RESISTANCE_BREAKOUT_RETEST_BLOCKED_BY_RR",
                        "direction": "NONE",
                        "zone_label": f"FLIPPED SUPPORT ({bz.get('low', 0):.2f}-{bz.get('high', 0):.2f})",
                        "target_zone": bz,
                        "trigger_status": "CONFIRMED (RR/SL BLOCKED)",
                        "rr_status": rr_res.get("rr_display", "FAILED"),
                        "rr_calculation": rr_res,
                        "checklist": {
                            "valid_breakout": True,
                            "retest_confirmed": True,
                            "bullish_confirmation": True,
                            "structural_sl_valid": bool(rr_res.get("structural_sl_available", False)),
                            "min_rr_met": False,
                        },
                        "wait_reasons": wait_reasons,
                        "reason": rr_res.get("reason", "Retest BUY terkonfirmasi namun RR/SL tidak memenuhi syarat."),
                    }

        # 2B. SELL — SUPPORT BREAKDOWN + RETEST (gagal reclaim / retest bertahan sebagai resistance)
        if z_type == "SUPPORT" and rt_res.get("status") == "RETEST_CONFIRMED" and rt_res.get("direction") == "BEARISH":
            dist_atr = max(0.0, safe_float(bz.get("low"), cp) - cp) / max(m5_atr, safe_float(bz.get("ref_atr"), m5_atr))
            if m5_bear_close and m5_bear_mom and dist_atr <= max_entry_dist_atr:
                rr_res = calculate_snr_risk_reward(
                    direction="SELL",
                    entry_price=cp,
                    active_zone=bz,
                    opposing_zones=active_sups,
                    atr=m5_atr,
                    symbol_spec=symbol_spec,
                    config=cfg,
                )
                if rr_res.get("valid", False):
                    return {
                        "final_status": "SELL",
                        "buy_status": "WAIT",
                        "sell_status": "CONFIRMED",
                        "setup_type": "SUPPORT_BREAKDOWN_RETEST",
                        "direction": "SELL",
                        "zone_label": f"FLIPPED RESISTANCE ({bz.get('low', 0):.2f}-{bz.get('high', 0):.2f})",
                        "target_zone": bz,
                        "trigger_status": "BREAKDOWN + RETEST CONFIRMED",
                        "rr_status": rr_res.get("rr_display", "VALID"),
                        "rr_calculation": rr_res,
                        "checklist": {
                            "valid_breakdown": True,
                            "retest_confirmed": True,
                            "bearish_confirmation": True,
                            "structural_sl_valid": True,
                            "min_rr_met": True,
                        },
                        "wait_reasons": [],
                        "reason": (
                            f"SELL terkonfirmasi melalui Support Breakdown + Retest pada "
                            f"{bz.get('low', 0):.2f}-{bz.get('high', 0):.2f} dengan RR {rr_res.get('rr_display')}."
                        ),
                    }
                else:
                    wait_reasons.append(rr_res.get("reason", "RR retest SELL di bawah standar."))
                    return {
                        "final_status": "WAIT",
                        "buy_status": "WAIT",
                        "sell_status": "WAIT",
                        "setup_type": "SUPPORT_BREAKDOWN_RETEST_BLOCKED_BY_RR",
                        "direction": "NONE",
                        "zone_label": f"FLIPPED RESISTANCE ({bz.get('low', 0):.2f}-{bz.get('high', 0):.2f})",
                        "target_zone": bz,
                        "trigger_status": "CONFIRMED (RR/SL BLOCKED)",
                        "rr_status": rr_res.get("rr_display", "FAILED"),
                        "rr_calculation": rr_res,
                        "checklist": {
                            "valid_breakdown": True,
                            "retest_confirmed": True,
                            "bearish_confirmation": True,
                            "structural_sl_valid": bool(rr_res.get("structural_sl_available", False)),
                            "min_rr_met": False,
                        },
                        "wait_reasons": wait_reasons,
                        "reason": rr_res.get("reason", "Retest SELL terkonfirmasi namun RR/SL tidak memenuhi syarat."),
                    }

    # Cek apakah seluruh zona aktif sudah INVALIDATED atau setup pada zona fokus terinvalidasi
    if (not active_sups and not active_ress and (invalid_zones or invalid_reasons)) or (
        focus_zone is not None and focus_zone.get("lifecycle_status") == "INVALIDATED"
    ):
        inv_msg = invalid_reasons[0] if invalid_reasons else "Zona target telah INVALIDATED."
        return {
            "final_status": "INVALID",
            "buy_status": "INVALID",
            "sell_status": "INVALID",
            "setup_type": "INVALIDATED_SETUP",
            "direction": "NONE",
            "zone_label": focus_label,
            "target_zone": focus_zone,
            "trigger_status": "INVALIDATED",
            "rr_status": "NOT CHECKED",
            "rr_calculation": {
                "checked": False,
                "valid": False,
                "rr_display": "NOT CHECKED",
                "reason": inv_msg,
            },
            "checklist": {},
            "wait_reasons": invalid_reasons or [inv_msg],
            "reason": inv_msg,
        }

    # Default: WAIT (belum ada konfirmasi lengkap; RR ditampilkan NOT CHECKED sesuai Section U)
    if focus_label.startswith("RESISTANCE"):
        default_wait_msg = "Waiting for valid resistance rejection."
    elif focus_label.startswith("SUPPORT"):
        default_wait_msg = "Waiting for valid support rejection."
    else:
        default_wait_msg = "Waiting for valid price reaction at SNR zone."

    return {
        "final_status": "WAIT",
        "buy_status": "WAIT",
        "sell_status": "WAIT",
        "setup_type": "NONE",
        "direction": "NONE",
        "zone_label": focus_label,
        "target_zone": focus_zone,
        "trigger_status": "NOT CONFIRMED",
        "rr_status": "NOT CHECKED",
        "rr_calculation": {
            "checked": False,
            "valid": False,
            "rr_display": "NOT CHECKED",
            "reason": "Trigger entry belum terkonfirmasi sehingga kalkulasi SL/TP/RR belum dieksekusi.",
        },
        "checklist": {},
        "wait_reasons": wait_reasons if wait_reasons else [default_wait_msg],
        "reason": default_wait_msg,
    }

