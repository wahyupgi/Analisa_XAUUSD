# -*- coding: utf-8 -*-
"""
modules/trend_following_context.py
Konteks pasar terpadu (SMC + SNR + trend following) untuk laporan unified.

Dipindahkan apa adanya dari modules/trend_following.py (structural refactoring, tanpa perubahan logika).
Modul ini DIMUAT dari bagian akhir modules.trend_following (compatibility re-export); impor selalu lewat modul tersebut
agar tidak terjadi circular import.
"""
from typing import Any, Dict, Optional
from modules.trend_following import (
    _safe_float,
)


def _norm_dir_bias(val: Any) -> Optional[str]:
    u = str(val or "").strip().upper()
    if "BULL" in u or u == "BUY":
        return "BULLISH"
    if "BEAR" in u or u == "SELL":
        return "BEARISH"
    return None


def _pick_smc_focus_zone_label(smc_scan: Dict[str, Any], primary_trend: Optional[str]) -> str:
    """Ambil label zona SMC relevan (Supply saat Bearish, Demand saat Bullish) secara read-only."""
    want_type = "SUPPLY" if primary_trend == "BEARISH" else ("DEMAND" if primary_trend == "BULLISH" else None)
    zones = (smc_scan.get("zones") or smc_scan.get("active_zones") or []) if isinstance(smc_scan, dict) else []
    cp = _safe_float(smc_scan.get("snapshot_price") or smc_scan.get("current_price"), 0.0) if isinstance(smc_scan, dict) else 0.0
    candidates = []
    for z in zones:
        if not isinstance(z, dict):
            continue
        st = str(z.get("lifecycle_status") or z.get("status") or "ACTIVE").upper()
        if any(bad in st for bad in ("INVALID", "EXPIRED", "MITIGATED_FULL", "BROKEN")):
            continue
        zt = str(z.get("type") or z.get("zone_type") or "").upper()
        if want_type and zt != want_type:
            continue
        lo = _safe_float(z.get("low"), 0.0)
        hi = _safe_float(z.get("high"), 0.0)
        if hi <= 0:
            continue
        dist = 0.0 if (cp > 0 and lo <= cp <= hi) else (min(abs(cp - lo), abs(cp - hi)) if cp > 0 else 999.0)
        score = _safe_float(z.get("score"), 0.0)
        candidates.append((dist, -score, zt, lo, hi))
    if candidates:
        candidates.sort()
        _, _, zt, lo, hi = candidates[0]
        return f"{zt} ({lo:.2f}-{hi:.2f})"
    p_lbl = str(smc_scan.get("primary_zone_label") or smc_scan.get("nearest_zone_label") or "").strip() if isinstance(smc_scan, dict) else ""
    if p_lbl and p_lbl.upper() != "NONE":
        return p_lbl
    return want_type or "ZONE"


def build_unified_market_context(
    smc_scan: Optional[Dict[str, Any]] = None,
    smc_dec: Optional[Dict[str, Any]] = None,
    snr_scan: Optional[Dict[str, Any]] = None,
    snr_dec: Optional[Dict[str, Any]] = None,
    snapshot: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Lapisan integrasi Market Context yang menggabungkan hasil SMC dan SNR secara
    independen tanpa memaksa kedua metodologi identik.

    Memisahkan:
      - Primary Trend: H4
      - Trend Confirmation: H1
      - Setup Context: M30
      - Entry Trigger: M5
      - Short-Term Momentum: M5/M30 momentum lokal
      - SNR Context: Support, Resistance, breakout, retest, rejection
    """
    s_smc = smc_scan if isinstance(smc_scan, dict) else {}
    d_smc = smc_dec if isinstance(smc_dec, dict) else {}
    s_snr = snr_scan if isinstance(snr_scan, dict) else {}
    d_snr = snr_dec if isinstance(snr_dec, dict) else (s_snr.get("decision") if isinstance(s_snr.get("decision"), dict) else {})
    snap = snapshot if isinstance(snapshot, dict) else {}

    mkt_status = str(s_snr.get("market_status") or s_smc.get("market_status") or snap.get("market_status") or "OPEN").strip().upper()
    fresh_smc = str(s_smc.get("data_freshness") or snap.get("data_freshness") or "FRESH").strip().upper()
    fresh_snr = str(s_snr.get("data_freshness") or snap.get("data_freshness") or "FRESH").strip().upper()
    market_closed = mkt_status in ("CLOSED", "MARKET CLOSED", "MARKET_CLOSED")
    hard_data_bad = (
        mkt_status in ("DATA UNAVAILABLE", "NO DATA", "UNAVAILABLE")
        or fresh_smc in ("INVALID", "NO DATA")
        or fresh_snr in ("INVALID", "NO DATA")
    )
    stale_while_open = (not market_closed) and (fresh_smc == "STALE" or fresh_snr == "STALE")
    data_unavailable = hard_data_bad or stale_while_open

    # 1. Ekstrak dimensi bias & struktur independen SMC & SNR
    smc_htf = s_smc.get("htf_context") or {}
    smc_tf_ev = d_smc.get("trend_following") or s_smc.get("trend_following") or {}
    smc_h4_raw = str(smc_htf.get("H4") or s_smc.get("htf_bias") or s_smc.get("h4_bias") or d_smc.get("h4_bias") or (smc_tf_ev.get("h4") or {}).get("direction") or "UNCONFIRMED").upper()
    smc_h1_raw = str(smc_htf.get("H1") or s_smc.get("h1_bias") or d_smc.get("h1_bias") or (smc_tf_ev.get("h1") or {}).get("direction") or "UNCONFIRMED").upper()
    smc_h1_conf = str(smc_htf.get("h1_confirmation") or d_smc.get("h1_confirmation") or (smc_tf_ev.get("h1") or {}).get("status") or "").upper()
    smc_m30_cond = str(smc_htf.get("m30_setup_condition") or d_smc.get("m30_setup_condition") or (smc_tf_ev.get("m30") or {}).get("condition") or d_smc.get("m30_status") or s_smc.get("m30_bias") or "").upper()
    smc_m30_status = str((smc_tf_ev.get("m30") or {}).get("status") or d_smc.get("m30_status") or s_smc.get("m30_bias") or "WAIT").upper()
    smc_m5_raw = str(d_smc.get("m5_status") or s_smc.get("m5_bias") or ("CONFIRMED" if str(d_smc.get("status", s_smc.get("decision_signal", ""))).upper() in ("VALID", "BUY", "SELL") else "WAIT")).upper()
    smc_status_raw = str(d_smc.get("status") or s_smc.get("decision_signal") or "WAIT").upper()
    smc_status = "VALID" if smc_status_raw in ("VALID", "BUY", "SELL") else smc_status_raw
    smc_dir = str(d_smc.get("direction") or s_smc.get("direction") or (smc_status_raw if smc_status_raw in ("BUY", "SELL") else "NONE")).upper()
    smc_setup_type = str(d_smc.get("setup_type") or "NONE").upper()

    snr_mb = s_snr.get("market_bias") or {}
    snr_h4_raw = str(snr_mb.get("h4") or "UNCONFIRMED").upper()
    snr_h1_raw = str(snr_mb.get("h1") or "UNCONFIRMED").upper()
    snr_m30_raw = str(snr_mb.get("m30") or "UNCONFIRMED").upper()
    snr_m5_raw = str(snr_mb.get("m5") or "WAIT").upper()
    snr_overall = str(snr_mb.get("overall_bias") or "UNCONFIRMED").upper()
    snr_pa = s_snr.get("price_action") or {}
    snr_ev = s_snr.get("entry_validation") or {}
    snr_ranked = s_snr.get("ranked_zones") or {}
    snr_moms = s_snr.get("momentum_by_tf") or {}
    snr_sig = str((d_snr or {}).get("signal") or "WAIT").upper()

    d_smc_h4 = _norm_dir_bias(smc_h4_raw)
    d_snr_h4 = _norm_dir_bias(snr_h4_raw)
    d_smc_h1 = _norm_dir_bias(smc_h1_raw)
    d_snr_h1 = _norm_dir_bias(snr_h1_raw)
    d_snr_m30 = _norm_dir_bias(snr_m30_raw)

    if d_smc_h4 and d_snr_h4:
        primary_trend = d_smc_h4 if d_smc_h4 == d_snr_h4 else "CONFLICT"
    elif d_smc_h4 or d_snr_h4:
        primary_trend = d_smc_h4 or d_snr_h4
    elif "SIDEWAYS" in (snr_h4_raw, snr_overall):
        primary_trend = "SIDEWAYS"
    else:
        primary_trend = "UNCONFIRMED"

    # H1 Confirmation state (berbasis struktur H1 SMC & SNR)
    opp_primary = "BEARISH" if primary_trend == "BULLISH" else ("BULLISH" if primary_trend == "BEARISH" else None)
    h1_opposes = bool(opp_primary and ((d_smc_h1 == opp_primary) or (d_snr_h1 == opp_primary)))
    h1_confirmed = bool(
        primary_trend in ("BULLISH", "BEARISH")
        and not h1_opposes
        and ((d_smc_h1 == primary_trend) or (d_snr_h1 == primary_trend) or smc_h1_conf == "ALIGNED")
    )
    if primary_trend == "CONFLICT" or smc_h1_conf == "CONFLICT" or h1_opposes:
        trend_confirmation = f"CONFLICT (SMC H1={smc_h1_raw}, SNR H1={snr_h1_raw})"
    elif smc_h1_raw.replace("NETRAL", "NEUTRAL") != snr_h1_raw.replace("NETRAL", "NEUTRAL"):
        trend_confirmation = f"PARTIAL (SMC H1={smc_h1_raw}, SNR H1={snr_h1_raw})"
    elif h1_confirmed:
        trend_confirmation = "PULLBACK" if smc_h1_conf == "PULLBACK" else "ALIGNED"
    else:
        trend_confirmation = smc_h1_conf or snr_h1_raw or "UNCONFIRMED"

    # Short-term Momentum (M5 & M30 closed candle momentum)
    m5_mom_dir = str((snr_moms.get("M5") or {}).get("direction") or snr_pa.get("momentum_summary") or "NEUTRAL").upper()
    m30_mom_dir = str((snr_moms.get("M30") or {}).get("direction") or "NEUTRAL").upper()
    st_mom_dir = m5_mom_dir if m5_mom_dir in ("BULLISH", "BEARISH") else (m30_mom_dir if m30_mom_dir in ("BULLISH", "BEARISH") else "NEUTRAL")
    if st_mom_dir == "NEUTRAL" and d_snr_m30 in ("BULLISH", "BEARISH"):
        st_mom_dir = d_snr_m30

    # SNR position & breakout/retest/rejection evidence
    s1 = (snr_ranked.get("active_supports") or [None])[0]
    r1 = (snr_ranked.get("active_resistances") or [None])[0]
    zone_lbl_ev = str(snr_ev.get("zone_label") or "").upper()
    price = _safe_float(snap.get("snapshot_price") or s_snr.get("snapshot_price") or s_snr.get("price") or s_smc.get("snapshot_price") or s_smc.get("current_price"), 0.0)
    # A selected focus zone is NOT proof that the current price is inside it.
    at_support = bool(price > 0 and s1 and _safe_float(s1.get("low")) <= price <= _safe_float(s1.get("high")))
    at_resistance = bool(price > 0 and r1 and _safe_float(r1.get("low")) <= price <= _safe_float(r1.get("high")))

    # Bukti pembalikan struktural (H1 + M30 BOS/CHoCH, bukan momentum M5/M30 semata)
    h1_per = (snr_mb.get("per_timeframe") or {}).get("H1") or {}
    m30_per = (snr_mb.get("per_timeframe") or {}).get("M30") or {}
    h1_bo_dir = _norm_dir_bias(h1_per.get("breakout_state")) or _norm_dir_bias(h1_per.get("swing_structure"))
    m30_bo_dir = _norm_dir_bias(m30_per.get("breakout_state")) or _norm_dir_bias(m30_per.get("swing_structure"))
    snr_rt_code = str(snr_pa.get("retest_code") or "NO_RETEST").upper()
    snr_rt_dir = _norm_dir_bias(snr_pa.get("retest_direction"))

    h1_shift_against = bool(opp_primary and (d_smc_h1 == opp_primary or d_snr_h1 == opp_primary or smc_h1_conf in ("CONFLICT", "REVERSAL") or h1_bo_dir == opp_primary))
    m30_shift_against = bool(
        opp_primary
        and (
            m30_bo_dir == opp_primary
            or d_snr_m30 == opp_primary
            or ("REVERSAL" in smc_m30_cond)
            or (snr_rt_code == "RETEST_CONFIRMED" and snr_rt_dir == opp_primary)
            or (d_smc_h1 == opp_primary and smc_m30_status == "CONFIRMED" and _norm_dir_bias(smc_dir) == opp_primary)
        )
    )
    confirmed_reversal = bool(
        (smc_setup_type == "REVERSAL" and smc_status == "VALID")
        or (
            opp_primary
            and h1_opposes
            and (snr_rt_code == "RETEST_CONFIRMED" and snr_rt_dir == opp_primary)
            and (d_smc_h1 == opp_primary or d_snr_h1 == opp_primary)
        )
        or (
            opp_primary
            and d_smc_h1 == opp_primary
            and d_snr_h1 == opp_primary
            and m30_bo_dir == opp_primary
            and (smc_status == "VALID" or snr_sig in ("BUY", "SELL"))
        )
    )

    is_counter_mom = bool(opp_primary and (st_mom_dir == opp_primary or m30_mom_dir == opp_primary))
    is_opposing_sr = bool((primary_trend == "BEARISH" and at_support) or (primary_trend == "BULLISH" and at_resistance))
    is_trend_sr = bool((primary_trend == "BEARISH" and at_resistance) or (primary_trend == "BULLISH" and at_support))
    is_pullback_state = bool(
        is_opposing_sr
        or smc_m30_cond in ("PULLBACK", "WAITING PULLBACK", "TREND_PULLBACK")
        or smc_h1_conf == "PULLBACK"
    )

    # 2. Klasifikasi Market Context berbasis struktur & closed candles
    if data_unavailable:
        classification = "DATA UNAVAILABLE"
        short_term_label = f"{st_mom_dir} (DATA UNAVAILABLE)"
        trade_bias = "NEUTRAL / WAIT"
        entry_status = "DATA UNAVAILABLE"
        reason = "Data MT5 stale/invalid atau tidak lengkap sehingga evaluasi entry dihentikan."
    elif confirmed_reversal:
        classification = "CONFIRMED REVERSAL"
        rev_dir_label = opp_primary or smc_dir or "REVERSAL"
        short_term_label = f"{rev_dir_label} REVERSAL CONFIRMED"
        trade_bias = "BUY PRIORITY" if rev_dir_label in ("BULLISH", "BUY") else "SELL PRIORITY"
        entry_status = "WAIT" if (market_closed or (smc_status != "VALID" and snr_sig not in ("BUY", "SELL"))) else f"READY ({rev_dir_label})"
        reason = f"Struktur H1 & M30 telah berbalik ke {rev_dir_label} melawan tren H4 sebelumnya."
    elif primary_trend in ("BULLISH", "BEARISH") and h1_shift_against and m30_shift_against and (
        "REVERSAL" in smc_m30_cond or smc_setup_type == "REVERSAL" or (snr_rt_code == "RETEST_CONFIRMED" and snr_rt_dir == opp_primary) or (h1_bo_dir == opp_primary and m30_bo_dir == opp_primary)
    ):
        classification = "POTENTIAL REVERSAL"
        short_term_label = f"{opp_primary} STRUCTURE SHIFT"
        trade_bias = "NEUTRAL / WAIT"
        entry_status = "MARKET CLOSED" if market_closed else "WAIT"
        reason = f"H1 & M30 mulai membentuk pergeseran struktur melawan tren utama {primary_trend}; menunggu konfirmasi lanjutan."
    elif primary_trend == "CONFLICT" or h1_opposes or smc_h1_conf == "CONFLICT" or "CONFLICT" in smc_m30_cond:
        classification = "STRUCTURAL CONFLICT"
        short_term_label = f"{st_mom_dir} (CONFLICT)"
        trade_bias = "NEUTRAL / WAIT"
        entry_status = "MARKET CLOSED" if market_closed else "WAIT"
        reason = f"Terjadi konflik struktur antar-timeframe atau metode (H4 SMC={smc_h4_raw}, H4 SNR={snr_h4_raw}, H1 SMC={smc_h1_raw}, H1 SNR={snr_h1_raw})."
    elif primary_trend in ("BULLISH", "BEARISH"):
        if is_opposing_sr or (is_counter_mom and not is_trend_sr) or (is_pullback_state and is_counter_mom):
            classification = "COUNTER-TREND PULLBACK"
            short_term_label = f"{opp_primary} PULLBACK"
            trade_bias = "SELL PRIORITY" if primary_trend == "BEARISH" else "BUY PRIORITY"
            reason = (
                f"Tren utama {primary_trend} (H4 & H1 terjaga); reaksi {opp_primary} saat ini "
                "merupakan counter-trend pullback jangka pendek, bukan pembalikan tren utama."
            )
        elif is_counter_mom and is_trend_sr:
            classification = "MOMENTUM CONFLICT"
            short_term_label = f"{st_mom_dir} MOMENTUM CONFLICT"
            trade_bias = "SELL PRIORITY" if primary_trend == "BEARISH" else "BUY PRIORITY"
            reason = (
                f"Harga berada di area searah tren utama {primary_trend}, tetapi momentum jangka pendek M5 ({st_mom_dir}) "
                "masih bertentangan dengan arah entry."
            )
        else:
            classification = "TREND ALIGNED" if trend_confirmation == "ALIGNED" else "TREND PARTIAL"
            short_term_label = f"{primary_trend} ALIGNED" if st_mom_dir in (primary_trend, "NEUTRAL") else f"{st_mom_dir} PULLBACK"
            trade_bias = "SELL PRIORITY" if primary_trend == "BEARISH" else "BUY PRIORITY"
            reason = f"Struktur multi-timeframe selaras dengan tren utama {primary_trend}."
        if market_closed:
            entry_status = "MARKET CLOSED"
        elif smc_status == "VALID" or snr_sig in ("BUY", "SELL"):
            entry_status = "WAIT" if (smc_status != "VALID" and snr_sig not in ("BUY", "SELL")) else "CONFIRMED"
        else:
            entry_status = "WAIT"
    else:
        if m5_mom_dir in ("BULLISH", "BEARISH") and m30_mom_dir in ("BULLISH", "BEARISH") and m5_mom_dir != m30_mom_dir:
            classification = "MOMENTUM CONFLICT"
            short_term_label = f"M5 {m5_mom_dir} vs M30 {m30_mom_dir}"
        elif snr_overall in ("BULLISH", "BEARISH") and m5_mom_dir == snr_overall:
            classification = "TREND ALIGNED"
            short_term_label = f"{snr_overall} SHORT-TERM"
        else:
            classification = "MOMENTUM CONFLICT" if m5_mom_dir in ("BULLISH", "BEARISH") else "STRUCTURAL CONFLICT"
            short_term_label = st_mom_dir
        trade_bias = "NEUTRAL / WAIT"
        entry_status = "MARKET CLOSED" if market_closed else "WAIT"
        reason = f"Struktur tren utama H4 belum terkonfirmasi ({primary_trend})."

    # 3. Susun SNR CONTEXT & SMC CONTEXT tanpa memaksa hasil identik
    snr_role_ctx = str(snr_ev.get("snr_role_context") or "").strip()
    if snr_role_ctx:
        snr_context_str = snr_role_ctx
    elif s1 and at_support:
        s1_rng = f"{_safe_float(s1.get('low')):.2f}-{_safe_float(s1.get('high')):.2f}"
        pb_tgt = snr_ev.get("pullback_target_zone") or (r1 if primary_trend == "BEARISH" else None)
        if pb_tgt:
            snr_context_str = (
                f"PRICE AT SUPPORT {s1.get('rank_label', 'S1')} ({s1_rng}) | "
                f"Next SELL Pullback: {pb_tgt.get('rank_label', 'R1')} ({_safe_float(pb_tgt.get('low')):.2f}-{_safe_float(pb_tgt.get('high')):.2f})"
            )
        else:
            snr_context_str = f"PRICE AT SUPPORT {s1.get('rank_label', 'S1')} ({s1_rng})"
    elif r1 and at_resistance:
        r1_rng = f"{_safe_float(r1.get('low')):.2f}-{_safe_float(r1.get('high')):.2f}"
        pb_tgt = snr_ev.get("pullback_target_zone") or (s1 if primary_trend == "BULLISH" else None)
        if pb_tgt:
            snr_context_str = (
                f"PRICE AT RESISTANCE {r1.get('rank_label', 'R1')} ({r1_rng}) | "
                f"Next BUY Pullback: {pb_tgt.get('rank_label', 'S1')} ({_safe_float(pb_tgt.get('low')):.2f}-{_safe_float(pb_tgt.get('high')):.2f})"
            )
        else:
            snr_context_str = f"PRICE AT RESISTANCE {r1.get('rank_label', 'R1')} ({r1_rng})"
    else:
        snr_context_str = (
            f"Focus={snr_ev.get('zone_label', 'N/A')} | "
            f"BO={snr_pa.get('breakout_summary', 'NOT CONFIRMED')} | "
            f"Rej={snr_pa.get('rejection_summary', 'NOT CONFIRMED')}"
        )

    if smc_status == "VALID" and smc_dir in ("BUY", "SELL"):
        smc_context_str = f"{smc_dir} CONFIRMED ({smc_setup_type}) @ {s_smc.get('primary_zone_label', 'SMC ZONE')}"
    else:
        smc_z_lbl = _pick_smc_focus_zone_label(s_smc, primary_trend if primary_trend in ("BULLISH", "BEARISH") else d_smc_h4)
        conf_need = "BEARISH CONFIRMATION" if primary_trend == "BEARISH" else ("BULLISH CONFIRMATION" if primary_trend == "BULLISH" else "STRUCTURE CONFIRMATION")
        smc_context_str = f"WAITING FOR {smc_z_lbl} / {conf_need}" if s_smc else "SMC STANDALONE NOT RUN"

    return {
        "classification": classification,
        "primary_trend": primary_trend,
        "trend_confirmation": trend_confirmation,
        "setup_context": f"SMC M30={smc_m30_status} | SNR M30={snr_m30_raw}",
        "entry_trigger": f"SMC M5={smc_m5_raw} | SNR M5={snr_m5_raw}",
        "short_term": short_term_label,
        "short_term_momentum": st_mom_dir,
        "snr_context": snr_context_str,
        "smc_context": smc_context_str,
        "trade_bias": trade_bias,
        "entry_status": entry_status,
        "reason": reason,
        "components": {
            "h4_primary": {"smc": smc_h4_raw, "snr": snr_h4_raw},
            "h1_confirmation": {"smc": smc_h1_raw, "snr": snr_h1_raw, "status": trend_confirmation},
            "m30_setup": {"smc": smc_m30_status, "smc_condition": smc_m30_cond, "snr": snr_m30_raw, "m30_momentum": m30_mom_dir},
            "m5_trigger": {"smc": smc_m5_raw, "snr": snr_m5_raw, "m5_momentum": m5_mom_dir},
            "snr_price_position": "AT_SUPPORT" if at_support else ("AT_RESISTANCE" if at_resistance else "MID_RANGE"),
        },
    }
