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
from modules.snr.data_provider import prepare_snr_frame, safe_float, zone_confirmation_start
from modules.snr.momentum_analysis import analyze_momentum
from modules.snr.rejection_detection import detect_zone_rejection
from modules.snr.retest_detection import detect_zone_retest
from modules.snr.risk_reward import calculate_snr_risk_reward, opposing_risk_zones


def _zone_was_tested(zone, clean_m5=None):
    """Memeriksa apakah zona SNR sudah pernah disentuh, diuji, atau dijemput harga."""
    if not isinstance(zone, dict):
        return False
    if int(zone.get("touch_count", 0) or 0) > 0:
        return True
    if int(zone.get("valid_touch_count", 0) or 0) > 0:
        return True
    if int(zone.get("reactions", 0) or 0) > 0:
        return True
    t_status = str(zone.get("touch_status", "")).upper()
    if t_status and t_status != "UNTESTED":
        return True
    event = str(zone.get("lifecycle_event", "")).upper()
    if event in ("TOUCHED", "SWEEP", "HISTORICAL_TOUCH", "TESTED"):
        return True
    st = str(zone.get("lifecycle_status", zone.get("status", ""))).upper()
    if st in ("TOUCHED", "SWEEP", "TESTED", "HISTORICAL_TOUCH"):
        return True
    if clean_m5 is not None and not clean_m5.empty:
        z_low = safe_float(zone.get("low"), 0.0)
        z_high = safe_float(zone.get("high"), z_low)
        z_type = str(zone.get("zone_type", "")).upper()
        z_lbl = str(zone.get("rank_label", "")).upper()
        if z_type == "RESISTANCE" or "R" in z_lbl or "RESISTANCE" in z_lbl:
            if (clean_m5["High"] >= z_low).any():
                return True
        elif z_type == "SUPPORT" or "S" in z_lbl or "SUPPORT" in z_lbl:
            if (clean_m5["Low"] <= z_high).any():
                return True
    return False


def _select_target_zone_for_monitoring(ranked_zones, market_bias, current_price, clean_m5=None):
    """
    Memilih dan memisahkan tiga konsep zona SNR:
      1. NEAREST_SNR_ZONE  : Zona support/resistance terdekat dengan harga saat ini.
      2. TREND_ENTRY_ZONE  : Zona yang mendukung arah tren utama (Resistance jika BEARISH, Support jika BULLISH).
      3. ENTRY_TRIGGER_ZONE: Zona yang sedang digunakan untuk validasi entry.
      4. FOCUS_ZONE        : Zona yang secara aktif berinteraksi atau sedang diuji harga saat ini.
    """
    active_sups = (ranked_zones or {}).get("active_supports") or []
    active_ress = (ranked_zones or {}).get("active_resistances") or []
    broken_zones = (ranked_zones or {}).get("broken_zones") or []
    invalid_zones = (ranked_zones or {}).get("invalidated_zones") or []

    cp = safe_float(current_price, 0.0)
    mb = market_bias if isinstance(market_bias, dict) else {}
    bias_str = str(mb.get("overall_bias") or mb.get("h4", "SIDEWAYS")).upper()
    h4_bias = str(mb.get("h4", "")).upper()

    s1 = active_sups[0] if active_sups else None
    r1 = active_ress[0] if active_ress else None

    # 1. NEAREST_SNR_ZONE
    d_s1 = safe_float(s1.get("distance_points"), abs(cp - safe_float(s1.get("high"), cp))) if s1 else float("inf")
    d_r1 = safe_float(r1.get("distance_points"), abs(cp - safe_float(r1.get("low"), cp))) if r1 else float("inf")
    if s1 and r1:
        if s1.get("price_inside_zone"):
            nearest_zone = s1
            nearest_label = f"SUPPORT {s1.get('rank_label', 'S1')}"
        elif r1.get("price_inside_zone"):
            nearest_zone = r1
            nearest_label = f"RESISTANCE {r1.get('rank_label', 'R1')}"
        elif d_s1 <= d_r1:
            nearest_zone = s1
            nearest_label = f"SUPPORT {s1.get('rank_label', 'S1')}"
        else:
            nearest_zone = r1
            nearest_label = f"RESISTANCE {r1.get('rank_label', 'R1')}"
    elif s1:
        nearest_zone = s1
        nearest_label = f"SUPPORT {s1.get('rank_label', 'S1')}"
    elif r1:
        nearest_zone = r1
        nearest_label = f"RESISTANCE {r1.get('rank_label', 'R1')}"
    elif broken_zones:
        bz = broken_zones[0]
        nearest_zone = bz
        nearest_label = f"BROKEN {bz.get('zone_type', 'ZONE')}"
    elif invalid_zones:
        iz = invalid_zones[0]
        nearest_zone = iz
        nearest_label = f"INVALIDATED {iz.get('zone_type', 'ZONE')}"
    else:
        nearest_zone = None
        nearest_label = "NONE"

    # 2. TREND_ENTRY_ZONE & FOCUS_ZONE
    is_bearish = (h4_bias == "BEARISH") or (not h4_bias and bias_str == "BEARISH")
    is_bullish = (h4_bias == "BULLISH") or (not h4_bias and bias_str == "BULLISH")

    if is_bearish:
        trend_entry_zone = r1
        trend_entry_label = f"RESISTANCE {r1.get('rank_label', 'R1')}" if r1 else "NONE"

        s1_inside = bool(s1 and (s1.get("price_inside_zone") or (safe_float(s1.get("low")) <= cp <= safe_float(s1.get("high")))))
        r1_tested = _zone_was_tested(r1, clean_m5)
        r1_departed = bool(r1 and (cp < safe_float(r1.get("low"), 0.0)) and r1_tested)

        if s1_inside or (r1_departed and d_s1 <= d_r1) or (s1 and d_s1 < d_r1 and r1_tested):
            focus_zone = s1
            focus_label = f"SUPPORT {s1.get('rank_label', 'S1')}" if s1 else nearest_label
            entry_trigger_zone = trend_entry_zone or s1
            entry_trigger_label = trend_entry_label or focus_label
        else:
            focus_zone = trend_entry_zone if trend_entry_zone else nearest_zone
            focus_label = trend_entry_label if trend_entry_zone else nearest_label
            entry_trigger_zone = trend_entry_zone
            entry_trigger_label = trend_entry_label

    elif is_bullish:
        trend_entry_zone = s1
        trend_entry_label = f"SUPPORT {s1.get('rank_label', 'S1')}" if s1 else "NONE"

        r1_inside = bool(r1 and (r1.get("price_inside_zone") or (safe_float(r1.get("low")) <= cp <= safe_float(r1.get("high")))))
        s1_tested = _zone_was_tested(s1, clean_m5)
        s1_departed = bool(s1 and (cp > safe_float(s1.get("high"), 0.0)) and s1_tested)

        if r1_inside or (s1_departed and d_r1 <= d_s1) or (r1 and d_r1 < d_s1 and s1_tested):
            focus_zone = r1
            focus_label = f"RESISTANCE {r1.get('rank_label', 'R1')}" if r1 else nearest_label
            entry_trigger_zone = trend_entry_zone or r1
            entry_trigger_label = trend_entry_label or focus_label
        else:
            focus_zone = trend_entry_zone if trend_entry_zone else nearest_zone
            focus_label = trend_entry_label if trend_entry_zone else nearest_label
            entry_trigger_zone = trend_entry_zone
            entry_trigger_label = trend_entry_label
    else:
        trend_entry_zone = nearest_zone
        trend_entry_label = nearest_label
        entry_trigger_zone = nearest_zone
        entry_trigger_label = nearest_label
        focus_zone = nearest_zone
        focus_label = nearest_label

    return (
        focus_zone,
        focus_label,
        nearest_zone,
        nearest_label,
        trend_entry_zone,
        trend_entry_label,
        entry_trigger_zone,
        entry_trigger_label,
    )


def validate_snr_entry(
    ranked_zones,
    m5_df,
    market_bias=None,
    current_price=None,
    symbol_spec=None,
    config=None,
    has_unclosed_last_bar=False,
    data_freshness="FRESH",
):
    """
    Memvalidasi seluruh syarat BUY SETUP (1A Support Rejection, 1B Resistance Breakout+Retest)
    dan SELL SETUP (2A Resistance Rejection, 2B Support Breakdown+Retest) pada candle M5 CLOSED.
    """
    cfg = config or get_snr_config()
    risk_cfg = cfg.get("risk", {})
    max_entry_dist_atr = safe_float(risk_cfg.get("max_entry_distance_atr", 0.65), 0.65)
    max_spread_points = safe_float(risk_cfg.get("max_spread_points", 1.0), 1.0)

    clean_m5, _ = prepare_snr_frame(m5_df, has_unclosed_last_bar=has_unclosed_last_bar)
    cp = safe_float(current_price, 0.0)
    if cp <= 0 and not clean_m5.empty:
        cp = safe_float(clean_m5["Close"].iloc[-1], 0.0)

    active_sups = (ranked_zones or {}).get("active_supports") or []
    active_ress = (ranked_zones or {}).get("active_resistances") or []
    broken_zones = (ranked_zones or {}).get("broken_zones") or []
    invalid_zones = (ranked_zones or {}).get("invalidated_zones") or []
    risk_resistances = opposing_risk_zones(ranked_zones, "BUY")
    risk_supports = opposing_risk_zones(ranked_zones, "SELL")

    (
        focus_zone,
        focus_label,
        nearest_snr_zone,
        nearest_snr_label,
        trend_entry_zone,
        trend_entry_label,
        entry_trigger_zone,
        entry_trigger_label,
    ) = _select_target_zone_for_monitoring(ranked_zones, market_bias, cp, clean_m5=clean_m5)

    def _blocked_wait(reason_msg, trigger_st="WAIT", rr_st="NOT CHECKED"):
        return {
            "final_status": "WAIT",
            "buy_status": "WAIT",
            "sell_status": "WAIT",
            "setup_type": "NONE",
            "direction": "NONE",
            "zone_label": focus_label,
            "target_zone": focus_zone,
            "nearest_snr_zone": nearest_snr_zone,
            "nearest_snr_label": nearest_snr_label,
            "trend_entry_zone": trend_entry_zone,
            "trend_entry_label": trend_entry_label,
            "entry_trigger_zone": entry_trigger_zone,
            "entry_trigger_label": entry_trigger_label,
            "trigger_status": trigger_st,
            "rr_status": rr_st,
            "rr_calculation": {
                "checked": False,
                "valid": False,
                "rr_display": rr_st,
                "reason": reason_msg,
            },
            "checklist": {},
            "wait_reasons": [reason_msg],
            "reason": reason_msg,
        }

    # Gate 1: Validasi Data Freshness
    df_fresh_st = str(data_freshness or "FRESH").upper()
    if df_fresh_st in ("STALE", "INVALID", "NO DATA"):
        return _blocked_wait(
            f"Data pasar berstatus {df_fresh_st}; sinyal entry SNR diblokir demi integritas data.",
            trigger_st=df_fresh_st,
        )

    # Gate 2: Candle M5 belum close / invalid OHLC
    if (
        getattr(m5_df, "attrs", {}).get("is_closed") is False
        or getattr(clean_m5, "attrs", {}).get("is_closed") is False
    ):
        return _blocked_wait(
            "Candle M5 terakhir belum selesai (forming); entry hanya diputuskan dari candle close."
        )

    if getattr(clean_m5, "attrs", {}).get("invalid_ohlc", False):
        return _blocked_wait("Data OHLC M5 tidak valid (korup / High < Low).")

    if clean_m5.empty or len(clean_m5) < 2 or cp <= 0:
        return _blocked_wait(
            "Data candle M5 closed belum mencukupi untuk konfirmasi trigger entry.",
            trigger_st="NOT CONFIRMED",
        )

    # Gate 3: Validasi Spread
    curr_spread = safe_float((symbol_spec or {}).get("spread_points"), 0.0)
    if curr_spread > max_spread_points:
        return _blocked_wait(
            f"Spread saat ini ({curr_spread:.2f} pt) melebihi batas maksimal ({max_spread_points:.2f} pt)."
        )

    # Gate 4: Market Bias CONFLICT / UNCONFIRMED / M30 CONFLICT
    mb = market_bias if isinstance(market_bias, dict) else {}
    bias_overall = str(mb.get("overall_bias", "")).upper()
    h4_bias = str(mb.get("h4", "")).upper()
    has_conflict = bool(mb.get("has_conflict", False)) or bias_overall == "CONFLICT"
    m30_conflict = bool(mb.get("m30_conflict", False))
    if has_conflict or bias_overall in ("CONFLICT", "UNCONFIRMED", "DATA UNAVAILABLE"):
        return _blocked_wait(
            f"Kondisi bias multi-timeframe ({bias_overall or 'CONFLICT'}) belum mengizinkan sinyal entry."
        )
    if m30_conflict:
        return _blocked_wait("Konflik momentum/struktur M30 terhadap arah struktur utama H4.")

    h4_bear_counter = (h4_bias == "BEARISH") or (not h4_bias and bias_overall == "BEARISH")
    h4_bull_counter = (h4_bias == "BULLISH") or (not h4_bias and bias_overall == "BULLISH")

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
    snr_role_context = ""
    def _rejection_touches_zone(rejection, zone):
        index = rejection.get("bar_index")
        if not isinstance(index, int) or not 0 <= index < len(clean_m5):
            return False
        bar = clean_m5.iloc[index]
        return safe_float(bar["Low"]) <= safe_float(zone.get("high")) and safe_float(bar["High"]) >= safe_float(zone.get("low"))

    def _closed_touch(zone):
        start = max(zone_confirmation_start(zone, clean_m5), len(clean_m5) - 4)
        recent = clean_m5.iloc[start:]
        return bool(((recent["Low"] <= safe_float(zone.get("high"))) & (recent["High"] >= safe_float(zone.get("low")))).any())

    # Context labels describe the live/current price, not historical candle touches.
    def _position_label(zone_kind, zone, low, high):
        inside = low <= cp <= high
        if inside:
            prefix = "PRICE AT"
        elif zone_kind == "SUPPORT" and cp > high:
            prefix = "PRICE ABOVE"
        elif zone_kind == "RESISTANCE" and cp < low:
            prefix = "PRICE BELOW"
        else:
            prefix = "PRICE OUTSIDE"
        return f"{prefix} {zone_kind} {zone.get('rank_label', 'S1' if zone_kind == 'SUPPORT' else 'R1')} ({low:.2f}-{high:.2f})"

    zone_hold_state = "NONE"
    pullback_target_zone = None
    pullback_target_label = "NONE"

    # ------------------------------------------------------------------
    # 1A. Evaluasi BUY — SUPPORT REJECTION pada S1 (jika tersedia)
    # ------------------------------------------------------------------
    s1 = active_sups[0] if active_sups else None
    r1_ref = active_ress[0] if active_ress else None
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
        bo_s1 = detect_zone_breakout(s1, clean_m5, momentum_result=m5_mom, config=cfg, has_unclosed_last_bar=False)
        valid_bull_rej = (rej_s1.get("status") == "BULLISH_REJECTION") and bool(rej_s1.get("confirmed", False)) and _rejection_touches_zone(rej_s1, s1)

        if rej_s1.get("status") == "FALSE_REJECTION" and rej_s1.get("in_zone_context"):
            invalid_reasons.append(f"Support {s1.get('rank_label', 'S1')} mengalami false rejection / ditembus.")

        if h4_bear_counter:
            if s1.get("price_inside_zone") or entered_support:
                if bo_s1.get("status") == "VALID_BREAKOUT":
                    zone_hold_state = "M5_BREAKDOWN"
                    snr_role_context = f"SUPPORT {s1.get('rank_label', 'S1')} ({s1_low:.2f}-{s1_high:.2f}): M5 BREAKDOWN CONFIRMED; {s1.get('timeframe', 'HTF')} {s1.get('lifecycle_status', 'UNCONFIRMED')}"
                elif valid_bull_rej and str(s1.get("lifecycle_status", s1.get("status", ""))).upper() not in ("UNCONFIRMED", "BREAKDOWN_PENDING"):
                    zone_hold_state = "HOLDING_PULLBACK"
                    if r1_ref is not None:
                        pullback_target_zone = r1_ref
                        pullback_target_label = f"{r1_ref.get('rank_label', 'R1')} ({safe_float(r1_ref.get('low')):.2f}-{safe_float(r1_ref.get('high')):.2f})"
                        snr_role_context = (
                            f"{_position_label('SUPPORT', s1, s1_low, s1_high)} [HOLDING PULLBACK] | "
                            f"Next SELL Pullback Area: {pullback_target_label}"
                        )
                    else:
                        snr_role_context = f"{_position_label('SUPPORT', s1, s1_low, s1_high)} [HOLDING PULLBACK]"
                else:
                    zone_hold_state = "TESTING_SUPPORT"
                    if r1_ref is not None:
                        pullback_target_zone = r1_ref
                        pullback_target_label = f"{r1_ref.get('rank_label', 'R1')} ({safe_float(r1_ref.get('low')):.2f}-{safe_float(r1_ref.get('high')):.2f})"
                        snr_role_context = (
                            f"{_position_label('SUPPORT', s1, s1_low, s1_high)} [TESTING] | "
                            f"Next SELL Pullback Area: {pullback_target_label}"
                        )
                    else:
                        snr_role_context = f"{_position_label('SUPPORT', s1, s1_low, s1_high)} [TESTING]"
                wait_reasons.append(
                    f"Harga di Support {s1.get('rank_label', 'S1')} saat tren utama BEARISH ({zone_hold_state}): "
                    "tidak otomatis BUY; pantau apakah pantulan membentuk pullback ke Resistance berikutnya."
                    + (f" Target area SELL berikutnya: {pullback_target_label}." if pullback_target_zone else "")
                )
            else:
                wait_reasons.append(f"Rejection Support {s1.get('rank_label', 'S1')} melawan tren utama BEARISH tanpa role-reversal terkonfirmasi.")
        elif entered_support and valid_bull_rej and m5_bull_close and m5_bull_mom and location_reasonable:
            rr_res = calculate_snr_risk_reward(
                direction="BUY",
                entry_price=cp,
                active_zone=s1,
                opposing_zones=risk_resistances,
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
        bo_r1 = detect_zone_breakout(r1, clean_m5, momentum_result=m5_mom, config=cfg, has_unclosed_last_bar=False)
        valid_bear_rej = (rej_r1.get("status") == "BEARISH_REJECTION") and bool(rej_r1.get("confirmed", False)) and _rejection_touches_zone(rej_r1, r1)

        if rej_r1.get("status") == "FALSE_REJECTION" and rej_r1.get("in_zone_context"):
            invalid_reasons.append(f"Resistance {r1.get('rank_label', 'R1')} mengalami false rejection / ditembus.")

        if h4_bull_counter:
            if r1.get("price_inside_zone") or entered_resistance:
                if bo_r1.get("status") == "VALID_BREAKOUT":
                    zone_hold_state = "M5_BREAKOUT"
                    snr_role_context = f"RESISTANCE {r1.get('rank_label', 'R1')} ({r1_low:.2f}-{r1_high:.2f}): M5 BREAKOUT CONFIRMED; {r1.get('timeframe', 'HTF')} {r1.get('lifecycle_status', 'UNCONFIRMED')}"
                elif valid_bear_rej and str(r1.get("lifecycle_status", r1.get("status", ""))).upper() not in ("UNCONFIRMED", "BREAKDOWN_PENDING"):
                    zone_hold_state = "HOLDING_PULLBACK"
                    if s1 is not None:
                        pullback_target_zone = s1
                        pullback_target_label = f"{s1.get('rank_label', 'S1')} ({safe_float(s1.get('low')):.2f}-{safe_float(s1.get('high')):.2f})"
                        snr_role_context = (
                            f"{_position_label('RESISTANCE', r1, r1_low, r1_high)} [HOLDING PULLBACK] | "
                            f"Next BUY Pullback Area: {pullback_target_label}"
                        )
                    else:
                        snr_role_context = f"{_position_label('RESISTANCE', r1, r1_low, r1_high)} [HOLDING PULLBACK]"
                else:
                    zone_hold_state = "TESTING_RESISTANCE"
                    if s1 is not None:
                        pullback_target_zone = s1
                        pullback_target_label = f"{s1.get('rank_label', 'S1')} ({safe_float(s1.get('low')):.2f}-{safe_float(s1.get('high')):.2f})"
                        snr_role_context = (
                            f"{_position_label('RESISTANCE', r1, r1_low, r1_high)} [TESTING] | "
                            f"Next BUY Pullback Area: {pullback_target_label}"
                        )
                    else:
                        snr_role_context = f"{_position_label('RESISTANCE', r1, r1_low, r1_high)} [TESTING]"
                wait_reasons.append(
                    f"Harga di Resistance {r1.get('rank_label', 'R1')} saat tren utama BULLISH ({zone_hold_state}): "
                    "tidak otomatis SELL; pantau apakah penolakan membentuk pullback ke Support berikutnya."
                    + (f" Target area BUY berikutnya: {pullback_target_label}." if pullback_target_zone else "")
                )
            else:
                wait_reasons.append(f"Rejection Resistance {r1.get('rank_label', 'R1')} melawan tren utama BULLISH tanpa role-reversal terkonfirmasi.")
        elif entered_resistance and valid_bear_rej and m5_bear_close and m5_bear_mom and location_reasonable:
            rr_res = calculate_snr_risk_reward(
                direction="SELL",
                entry_price=cp,
                active_zone=r1,
                opposing_zones=risk_supports,
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
        bo_res = detect_zone_breakout(bz, clean_m5, momentum_result=m5_mom, config=cfg)
        rt_res = detect_zone_retest(bz, clean_m5, breakout_result=bo_res, config=cfg)

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
                    opposing_zones=risk_resistances,
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
                    opposing_zones=risk_supports,
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
            "snr_role_context": snr_role_context,
            "zone_hold_state": zone_hold_state,
            "pullback_target_zone": pullback_target_zone,
            "pullback_target_label": pullback_target_label,
            "wait_reasons": invalid_reasons or [inv_msg],
            "reason": inv_msg,
        }

    # Default: WAIT (belum ada konfirmasi lengkap; RR ditampilkan NOT CHECKED sesuai Section U)
    if h4_bear_counter:
        if s1 and (s1.get("price_inside_zone") or entered_support or cp <= (s1_high + 0.20 * m5_atr)):
            if _zone_was_tested(r1, clean_m5):
                default_wait_msg = (
                    f"WAIT_SUPPORT_REACTION: Harga telah terjemput di Resistance {r1.get('rank_label', 'R1')} "
                    f"({r1_low:.2f}-{r1_high:.2f}) dan running turun menguji Support {s1.get('rank_label', 'S1')} "
                    f"({s1_low:.2f}-{s1_high:.2f}). Menunggu konfirmasi M5 Breakdown {s1.get('rank_label', 'S1')} "
                    f"untuk sell continuation, atau pantau reaksi support."
                )
            else:
                default_wait_msg = (
                    f"WAIT_SUPPORT_REACTION: Harga di Support {s1.get('rank_label', 'S1')} ({s1_low:.2f}-{s1_high:.2f}) "
                    f"saat tren utama BEARISH; pantau potensi M5 Breakdown atau pantulan menuju Resistance {r1.get('rank_label', 'R1')} "
                    f"({r1_low:.2f}-{r1_high:.2f})."
                ) if r1 else f"WAIT_SUPPORT_REACTION: Harga di Support {s1.get('rank_label', 'S1')} ({s1_low:.2f}-{s1_high:.2f}) saat tren utama BEARISH."
        elif r1 is not None and (r1.get("price_inside_zone") or (entered_resistance and cp >= r1_low - 0.20 * m5_atr)):
            default_wait_msg = f"WAITING BEARISH CONFIRMATION at Resistance {r1.get('rank_label', 'R1')} ({r1_low:.2f}-{r1_high:.2f})."
        elif r1:
            if _zone_was_tested(r1, clean_m5) and cp < r1_low:
                default_wait_msg = (
                    f"RUNNING_TO_SUPPORT: Harga telah terjemput di Resistance {r1.get('rank_label', 'R1')} "
                    f"({r1_low:.2f}-{r1_high:.2f}) dan sedang running turun menuju Support {s1.get('rank_label', 'S1') if s1 else 'berikutnya'}."
                )
            else:
                distance = max(r1_low - cp, 0.0)
                approach = safe_float(cfg.get("radar", {}).get("approaching_distance_atr"), 0.60) * max(0.25, safe_float(r1.get("ref_atr"), 2.0))
                code = "WAIT_ZONE_TOUCH" if distance <= approach else "WAIT_ZONE_APPROACH"
                default_wait_msg = f"{code}: Menunggu harga mencapai {trend_entry_label}."
        else:
            default_wait_msg = "WAITING RESISTANCE PULLBACK (No active resistance detected)."
    elif h4_bull_counter:
        if r1 and (r1.get("price_inside_zone") or entered_resistance or cp >= (r1_low - 0.20 * m5_atr)):
            if _zone_was_tested(s1, clean_m5):
                default_wait_msg = (
                    f"WAIT_RESISTANCE_REACTION: Harga telah terjemput di Support {s1.get('rank_label', 'S1')} "
                    f"({s1_low:.2f}-{s1_high:.2f}) dan running naik menguji Resistance {r1.get('rank_label', 'R1')} "
                    f"({r1_low:.2f}-{r1_high:.2f}). Menunggu konfirmasi M5 Breakout {r1.get('rank_label', 'R1')} "
                    f"untuk buy continuation, atau pantau reaksi resistance."
                )
            else:
                default_wait_msg = (
                    f"WAIT_RESISTANCE_REACTION: Harga di Resistance {r1.get('rank_label', 'R1')} ({r1_low:.2f}-{r1_high:.2f}) "
                    f"saat tren utama BULLISH; pantau potensi M5 Breakout atau penolakan menuju Support {s1.get('rank_label', 'S1')} "
                    f"({s1_low:.2f}-{s1_high:.2f})."
                ) if s1 else f"WAIT_RESISTANCE_REACTION: Harga di Resistance {r1.get('rank_label', 'R1')} ({r1_low:.2f}-{r1_high:.2f}) saat tren utama BULLISH."
        elif s1 is not None and (s1.get("price_inside_zone") or (entered_support and cp <= s1_high + 0.20 * m5_atr)):
            default_wait_msg = f"WAITING BULLISH CONFIRMATION at Support {s1.get('rank_label', 'S1')} ({s1_low:.2f}-{s1_high:.2f})."
        elif s1:
            if _zone_was_tested(s1, clean_m5) and cp > s1_high:
                default_wait_msg = (
                    f"RUNNING_TO_RESISTANCE: Harga telah terjemput di Support {s1.get('rank_label', 'S1')} "
                    f"({s1_low:.2f}-{s1_high:.2f}) dan sedang running naik menuju Resistance {r1.get('rank_label', 'R1') if r1 else 'berikutnya'}."
                )
            else:
                distance = max(cp - s1_high, 0.0)
                approach = safe_float(cfg.get("radar", {}).get("approaching_distance_atr"), 0.60) * max(0.25, safe_float(s1.get("ref_atr"), 2.0))
                code = "WAIT_ZONE_TOUCH" if distance <= approach else "WAIT_ZONE_APPROACH"
                default_wait_msg = f"{code}: Menunggu harga mencapai {trend_entry_label}."
        else:
            default_wait_msg = "WAITING SUPPORT PULLBACK (No active support detected)."
    elif focus_label.startswith("RESISTANCE"):
        default_wait_msg = "Waiting for valid resistance rejection."
    elif focus_label.startswith("SUPPORT"):
        default_wait_msg = "Waiting for valid support rejection."
    else:
        default_wait_msg = "Waiting for valid price reaction at SNR zone."

    focus_is_support = bool(s1 and focus_zone and focus_zone.get("zone_id") == s1.get("zone_id"))
    focus_is_resistance = bool(r1 and focus_zone and focus_zone.get("zone_id") == r1.get("zone_id"))
    recent_closed_touch = False
    if (focus_is_support and not h4_bear_counter) or (focus_is_resistance and not h4_bull_counter):
        target = s1 if focus_is_support else r1
        entered = entered_support if focus_is_support else entered_resistance
        rejection_valid = valid_bull_rej if focus_is_support else valid_bear_rej
        directional_close = m5_bull_close if focus_is_support else m5_bear_close
        momentum_valid = m5_bull_mom if focus_is_support else m5_bear_mom
        location_valid = (
            entered_support and dist_from_s1_atr <= max_entry_dist_atr
            if focus_is_support else entered_resistance and dist_from_r1_atr <= max_entry_dist_atr
        )
        inside_now = safe_float(target.get("low")) <= cp <= safe_float(target.get("high"))
        closed_touch = _closed_touch(target)
        recent_closed_touch = closed_touch
        if not closed_touch and inside_now:
            default_wait_msg = "WAIT_M5_CONFIRMATION: Harga di zona; tunggu interaksi candle M5 tertutup."
        elif not closed_touch or not entered:
            if focus_is_resistance and _zone_was_tested(target, clean_m5) and cp < safe_float(target.get("low")):
                default_wait_msg = (
                    f"RUNNING_TO_SUPPORT: Harga telah terjemput di {focus_label} dan sedang running turun "
                    f"menuju Support {s1.get('rank_label', 'S1') if s1 else 'berikutnya'}."
                )
            elif focus_is_support and _zone_was_tested(target, clean_m5) and cp > safe_float(target.get("high")):
                default_wait_msg = (
                    f"RUNNING_TO_RESISTANCE: Harga telah terjemput di {focus_label} dan sedang running naik "
                    f"menuju Resistance {r1.get('rank_label', 'R1') if r1 else 'berikutnya'}."
                )
            else:
                distance = max(safe_float(target.get("low")) - cp, cp - safe_float(target.get("high")), 0.0)
                approach = safe_float(cfg.get("radar", {}).get("approaching_distance_atr"), 0.60) * max(0.25, safe_float(target.get("ref_atr"), 2.0))
                code = "WAIT_ZONE_TOUCH" if distance <= approach else "WAIT_ZONE_APPROACH"
                default_wait_msg = f"{code}: Menunggu harga mencapai {focus_label}."
        elif not rejection_valid:
            default_wait_msg = f"WAIT_M5_CONFIRMATION: Belum ada rejection M5 terkonfirmasi pada {focus_label}."
        elif not directional_close or not momentum_valid:
            default_wait_msg = f"WAIT_M5_CONFIRMATION: Candle atau momentum M5 belum mendukung {focus_label}."
        elif not location_valid:
            default_wait_msg = f"WAIT_ZONE_APPROACH: Lokasi entry terlalu jauh dari {focus_label}."

    return {
        "final_status": "WAIT",
        "buy_status": "WAIT",
        "sell_status": "WAIT",
        "setup_type": "NONE",
        "direction": "NONE",
        "zone_label": focus_label,
        "target_zone": focus_zone,
        "recent_closed_touch": recent_closed_touch,
        "nearest_snr_zone": nearest_snr_zone,
        "nearest_snr_label": nearest_snr_label,
        "trend_entry_zone": trend_entry_zone,
        "trend_entry_label": trend_entry_label,
        "entry_trigger_zone": entry_trigger_zone,
        "entry_trigger_label": entry_trigger_label,
        "trigger_status": "NOT CONFIRMED",
        "rr_status": "NOT CHECKED",
        "rr_calculation": {
            "checked": False,
            "valid": False,
            "rr_display": "NOT CHECKED",
            "reason": "Trigger entry belum terkonfirmasi sehingga kalkulasi SL/TP/RR belum dieksekusi.",
        },
        "checklist": {},
        "snr_role_context": snr_role_context,
        "zone_hold_state": zone_hold_state,
        "pullback_target_zone": pullback_target_zone,
        "pullback_target_label": pullback_target_label,
        "wait_reasons": [default_wait_msg] + [w for w in wait_reasons if w != default_wait_msg],
        "reason": default_wait_msg,
    }

