# -*- coding: utf-8 -*-
"""
modules/zone_formatter.py
Presentation layer SMC Zone Scanner: formatter output terminal (scan / scan smc / diagnostic / audit).

Dipindahkan apa adanya dari modules/zone_scanner.py (structural refactoring, tanpa perubahan logika).
Catatan dependency: modul ini bergantung pada modules.zone_scanner dan DIMUAT dari bagian akhir
modules.zone_scanner (compatibility re-export). Impor selalu lewat `modules.zone_scanner`
agar tidak terjadi circular import.
"""
import math
from datetime import datetime
from typing import Any, Dict, List, Optional

from modules.config import (
    SYMBOL, MULTI_TIER_MAX_ZONES,
    LOW_RISK_MAX_ATR_RATIO, LOW_RISK_MAX_DISTANCE_PT,
    SIGNAL_STATUS_NO_SETUP, SIGNAL_STATUS_SIGNAL_CONFIRMED,
    INVALIDATION_SETUP_SL_BREACHED, INVALIDATION_ZONE_STRUCTURAL,
)
from modules.mt5_conn import (
    fetch_mt5_pending_orders_snapshot,
    find_pending_orders_on_invalid_zones,
    format_mt5_orders_summary_lines,
    format_pending_compact_status,
    format_pending_reconciliation_evidence,
    reconcile_pending_plan_with_mt5,
    reconcile_pending_plans_batch,
    summarize_mt5_risk_warnings,
)
from modules.utils import (
    get_current_wib,
    to_wib_datetime,
    extract_unit_spec_from_container,
    format_invalid_zone_compact,
    format_mt5_sync_short,
    format_price_distance,
    format_trade_plan_lines,
    format_zone_relative_distance,
    resolve_entry_blocker_code,
    select_relevant_invalid_zone,
    consume_one_shot_invalid_zone,
    zones_represent_same_level,
)
from modules.terminal_ui import get_terminal_width, wrap_dashboard_field
from modules.zone_scanner import (
    SCORE_MIN_ACTIVE,
    _format_radar_status_clause,
    calculate_zone_approach,
    calculate_zone_distance,
    create_market_snapshot,
    detect_early_setups,
    evaluate_smc_snr_confluence,
    evaluate_trade_decision,
    get_zone_tracker,
    is_far_reference_zone,
    prioritize_radar_zones,
    resolve_analysis_direction,
    resolve_radar_state,
    resolve_smc_setup_invalidation_price,
    resolve_smc_structure_invalidation_price,
    sanitize_wait_reason,
    scan_xauusd_zones,
)




def format_diagnostic_report(scan: Optional[Dict[str, Any]] = None) -> str:
    """Format laporan diagnostik komprehensif scanner XAUUSD (TASK Section 5)."""
    if scan is None:
        _diag_snap = create_market_snapshot()
        scan = scan_xauusd_zones(snapshot=_diag_snap)

    if not scan.get("data_ok"):
        return (
            "=" * 70 + "\n"
            "XAUUSD ZONE SCANNER DIAGNOSTIC REPORT\n"
            "=" * 70 + "\n"
            f"Snapshot ID        : {scan.get('snapshot_id', 'N/A')}\n"
            f"Status Data        : {scan.get('snapshot_status') or 'TIDAK VALID / OFFLINE'}\n"
            f"Alasan             : {scan.get('empty_reason', 'Data tidak tersedia')}\n"
            "=" * 70
        )

    diag = scan.get("diagnostics") or {}
    ctx = scan.get("htf_context") or {}
    price = scan.get("current_price", 0.0)
    ts = scan.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    sym = scan.get("symbol", SYMBOL)
    snap_id = scan.get("snapshot_id", "N/A")
    snap_ts = scan.get("snapshot_time") or ts
    snap_price = scan.get("snapshot_price", price)
    r_prio = scan.get("radar_priority") or {}
    prio_ts = r_prio.get("priority_calc_timestamp") or snap_ts
    pz_diag = r_prio.get("primary_directional_zone")
    pz_comp = f"{pz_diag.get('priority_score'):.1f}" if isinstance(pz_diag, dict) and pz_diag.get("priority_score") is not None else "N/A"

    by_tf = diag.get("by_timeframe", {})
    tf_summary_parts = []
    for tf_key in ("H4", "H1", "M30"):
        if tf_key in by_tf:
            tf_summary_parts.append(f"{tf_key}: {by_tf[tf_key].get('bars', 0)} bars")
    tf_bars_str = " (" + ", ".join(tf_summary_parts) + ")" if tf_summary_parts else ""

    lines = [
        "=" * 70,
        "XAUUSD ZONE SCANNER DIAGNOSTIC REPORT",
        "=" * 70,
        f"Timestamp          : {ts}",
        f"Snapshot ID        : {snap_id}",
        f"Snapshot Timestamp : {snap_ts}",
        f"Snapshot Price     : {snap_price:.2f}" if isinstance(snap_price, (int, float)) else f"Snapshot Price     : {snap_price}",
        f"Priority Calc Time : {prio_ts}",
        f"Priority Composite : {pz_comp}",
        f"Broker / Symbol    : {sym}",
        f"Current Price      : {price:.2f}",
        "",
        "1. DATA & STRUKTUR PASAR:",
        f"- Candle Diperiksa : {diag.get('candles_checked', 0)} bars total{tf_bars_str}",
        f"- Swing Ditemukan  : {diag.get('swings_found', 0)} swings ({diag.get('swings_high', 0)} High, {diag.get('swings_low', 0)} Low)",
        f"- BOS / CHoCH      : {diag.get('bos_choch_count', 0)} event struktural teridentifikasi",
        f"- FVG Imbalance    : {diag.get('fvg_count', 0)} FVG terdeteksi",
        "",
        "2. DETEKSI & VALIDASI ZONA:",
        f"- Kandidat OB      : {diag.get('ob_candidates_count', 0)} leg impuls diperiksa",
        f"- Total Terdeteksi : {diag.get('zones_detected', 0)} zona (Radar: {diag.get('zones_radar', 0)}, Valid: {diag.get('zones_valid_active', 0)})",
        f"- Lolos Hard Valid : {diag.get('zones_passed_hard_validation', 0)} zona",
        f"- Lolos Scoring    : {diag.get('zones_passed_scoring', 0)} zona (Skor >= {SCORE_MIN_ACTIVE})",
        f"- Zona Valid Aktif : {diag.get('zones_valid_active', 0)} zona (setelah merge overlap)",
        "",
        "3. BREAKDOWN ZONA PER TIMEFRAME (H4, H1, M30):",
    ]

    if by_tf:
        for tf_k in ("H4", "H1", "M30"):
            if tf_k in by_tf:
                tf_info = by_tf[tf_k]
                lines.append(f"  • {tf_k} : {tf_info.get('ob_candidates', 0)} OB candidates ({tf_info.get('hard_pass', 0)} lolos hard valid, {tf_info.get('hard_reject', 0)} ditolak)")
    else:
        lines.append("  • Belum ada data per timeframe.")

    lines.extend([
        "",
        "4. BREAKDOWN FILTER & PENOLAKAN:",
        f"- Zona Invalidated (Tembus Close) : {diag.get('zones_invalidated', 0)} zona",
        f"- Zona Termitigasi (> 2 Sentuhan) : {diag.get('zones_mitigated', 0)} zona",
        f"- Zona Terlalu Jauh (> 35 ATR)   : {diag.get('zones_too_far', 0)} zona",
        f"- Total Zona Ditolak / Gagal Valid: {diag.get('zones_rejected', 0)} kandidat",
    ])

    reasons = diag.get("rejected_reasons_breakdown", {})
    if reasons:
        lines.append("  Rincian alasan penolakan kandidat:")
        for r_name, r_cnt in sorted(reasons.items(), key=lambda x: -x[1])[:8]:
            lines.append(f"    * {r_name}: {r_cnt}x")

    iz_log = diag.get("inspected_zones_log", [])
    if iz_log:
        u_spec_d = extract_unit_spec_from_container(scan)
        lines.append("")
        lines.append("5. AUDIT LOG ZONA MULTI-TIMEFRAME (H4/H1/M30):")
        for rz in iz_log[:10]:
            st_flag = "LOLOS" if rz["is_valid"] else f"DITOLAK: {rz['rejection_reason']}"
            fvg_st = rz.get("fvg_status", "-")
            dist_lbl = format_price_distance(rz["distance"], spec=u_spec_d, include_mt5_points=True)
            lines.append(f"  • [{rz['source_timeframe']}] {rz['zone_type']} {rz['price_low']:.2f}-{rz['price_high']:.2f} "
                         f"[Grade: {rz['grade']}, Skor: {rz['score']:.1f}, FVG: {fvg_st}, Touch: {rz['touch_count']}x, Jarak: {dist_lbl} ({rz['direction_status']})] -> {st_flag}")

    conflict_lbl = "CONFLICT" if diag.get("htf_conflict") else "ALIGNED"
    lines.extend([
        "",
        "6. STATUS HTF & KONFLIK:",
        f"- H4 Bias          : {ctx.get('H4', 'NETRAL')}",
        f"- H1 Structure     : {ctx.get('H1', 'NETRAL')}",
        f"- M30 Momentum     : {ctx.get('M30', 'NETRAL')}",
        f"- Status Konflik   : {conflict_lbl} ({ctx.get('bias_utama', '-')})",
    ])

    snr_ctx = scan.get("snr_context") if isinstance(scan.get("snr_context"), dict) else {}
    snr_st_diag = snr_ctx.get("status") or ("AVAILABLE" if snr_ctx.get("available") else "UNAVAILABLE")
    snr_sum_diag = snr_ctx.get("summary") or "Belum dievaluasi"
    snr_fresh_obj = snr_ctx.get("freshness") if isinstance(snr_ctx.get("freshness"), dict) else {}
    snr_fresh_diag = str(snr_fresh_obj.get("status") or ("PASS" if snr_ctx.get("available") else "N/A"))
    snr_sups = snr_ctx.get("supports") or []
    snr_ress = snr_ctx.get("resistances") or []
    snr_sups_cnt = len(snr_sups)
    snr_ress_cnt = len(snr_ress)
    snr_brk_cnt = len(snr_ctx.get("broken_zones") or [])
    snr_avail_tf = snr_ctx.get("available_timeframes") or []
    snr_miss_tf = snr_ctx.get("missing_timeframes") or []
    snr_err_rsn = snr_ctx.get("error_reason") or snr_ctx.get("reason")
    snr_warns = snr_ctx.get("warnings") or []
    lines.extend([
        "",
        "7. STATUS INTEGRASI SNR CONTEXT:",
        f"- Status Integrasi : {snr_st_diag}",
        f"- Freshness SNR    : {snr_fresh_diag}" + (f" ({snr_fresh_obj.get('reason')})" if snr_fresh_obj.get("reason") and snr_fresh_diag != "PASS" else ""),
        f"- Ringkasan        : {snr_sum_diag}",
        f"- Zona Terdeteksi  : Support={snr_sups_cnt} | Resistance={snr_ress_cnt} | Broken={snr_brk_cnt}",
    ])
    if snr_st_diag in ("AVAILABLE", "PARTIAL") and (snr_sups or snr_ress):
        s1_d = snr_sups[0] if snr_sups else None
        r1_d = snr_ress[0] if snr_ress else None
        s1_txt = f"{s1_d.get('low', 0.0):.2f}-{s1_d.get('high', 0.0):.2f} (Grade {s1_d.get('grade', '-')})" if isinstance(s1_d, dict) else "NONE"
        r1_txt = f"{r1_d.get('low', 0.0):.2f}-{r1_d.get('high', 0.0):.2f} (Grade {r1_d.get('grade', '-')})" if isinstance(r1_d, dict) else "NONE"
        lines.append(f"- Support / Resist : S1={s1_txt} | R1={r1_txt}")
        lines.append(
            f"- Price Action SNR : Breakout={(snr_ctx.get('price_action') or {}).get('breakout_summary', 'NOT CONFIRMED')} | "
            f"Retest={(snr_ctx.get('price_action') or {}).get('retest_summary', 'NOT CONFIRMED')} | "
            f"Rejection={(snr_ctx.get('price_action') or {}).get('rejection_summary', 'NOT CONFIRMED')}"
        )
    if snr_avail_tf and snr_st_diag == "PARTIAL":
        lines.append(f"- Available TF     : {', '.join(snr_avail_tf)}")
    if snr_miss_tf:
        lines.append(f"- Missing/Short TF : {', '.join(snr_miss_tf)}")
    if snr_st_diag == "UNAVAILABLE" or snr_ctx.get("error_type"):
        err_src = snr_ctx.get("error_source") or "snr"
        lines.append(f"- Error Integrasi  : [{snr_ctx.get('error_type', 'Error')} | {err_src}] {snr_err_rsn or 'Kegagalan integrasi SNR'}")
    elif snr_warns:
        lines.append(f"- Peringatan       : {'; '.join(snr_warns)}")
    confl_diag = scan.get("smc_snr_confluence")
    if confl_diag and confl_diag.get("available") and confl_diag.get("is_aligned"):
        lines.append("")
        lines.append("8. KONFLUENSI SMC + SNR (SUPPORTING CONTEXT):")
        for cl in confl_diag.get("display_block", "").splitlines():
            lines.append(f"  {cl}")
    lines.append("=" * 70)

    return "\n".join(lines)


def format_scanner_audit(dec: Dict[str, Any]) -> str:
    """Format audit log standar [SCANNER] sesuai spesifikasi audit trading (TASK Section E)."""
    status = dec.get("status", "WAIT")
    direction = dec.get("direction")
    stages = dec.get("stages", {})
    plan = dec.get("plan") or {}

    lines = []
    # 1. H4 Bias
    h4_stage = stages.get("H4", {})
    h4_msg = h4_stage.get("msg", "-")
    if h4_stage.get("ok"):
        h4_bias = "BULLISH" if "BULLISH" in h4_msg.upper() else ("BEARISH" if "BEARISH" in h4_msg.upper() else "NETRAL")
    elif "H4" in stages:
        h4_bias = "UNCONFIRMED"
    else:
        h4_bias = "NOT_CHECKED"
    lines.append(f"[SCANNER] H4 Bias: {h4_bias}")

    # 2. H1 Structure
    h1_stage = stages.get("H1", {})
    if h1_stage.get("ok"):
        h1_str = "BULLISH" if direction == "BUY" else ("BEARISH" if direction == "SELL" else "ALIGNED")
    elif "H1" in stages:
        h1_str = "CONFLICT" if ("bertentangan" in h1_stage.get("msg", "").lower() or "konflik" in h1_stage.get("msg", "").lower()) else "UNCONFIRMED"
    else:
        h1_str = "NOT_CHECKED"
    lines.append(f"[SCANNER] H1 Structure: {h1_str}")

    # 3. M30 Momentum
    m30_stage = stages.get("M30", {})
    if "status" in m30_stage:
        m30_status = m30_stage["status"]
    elif m30_stage.get("ok"):
        m30_status = "CONFIRMED"
    elif "M30" in stages:
        m30_status = "UNCONFIRMED"
    else:
        m30_status = "NOT_CHECKED"
    lines.append(f"[SCANNER] M30 Momentum: {m30_status}")

    # 4. Zone Validation
    zv_stage = stages.get("zone_validation", {})
    zv_status = zv_stage.get("status")
    if not zv_status:
        zv_status = "CONFIRMED" if status == "VALID" else ("NOT_CHECKED" if "zone_validation" not in stages else "WAIT")
    lines.append(f"[SCANNER] Zone Validation: {zv_status}")

    # 5. Zone Touch
    zt_stage = stages.get("zone_touch", {})
    zt_status = zt_stage.get("status")
    if not zt_status:
        zt_status = "CONFIRMED" if status == "VALID" else ("NOT_CHECKED" if "zone_touch" not in stages else "WAIT")
    lines.append(f"[SCANNER] Zone Touch: {zt_status}")

    # 6. Liquidity Sweep
    sw_stage = stages.get("liquidity_sweep", {})
    sw_status = sw_stage.get("status")
    if not sw_status:
        if status == "VALID":
            sw_status = "CONFIRMED"
        elif "liquidity_sweep" in stages:
            sw_status = "NOT CONFIRMED"
        else:
            sw_status = "NOT_CHECKED"
    elif sw_status == "WAIT":
        sw_status = "NOT CONFIRMED"
    lines.append(f"[SCANNER] M5 Liquidity Sweep: {sw_status}")

    # 7. CHoCH / BOS / CISD
    tr_stage = stages.get("structure_trigger", {})
    tr_status = tr_stage.get("status")
    if not tr_status:
        tr_status = "CONFIRMED" if status == "VALID" else ("NOT_CHECKED" if "structure_trigger" not in stages else "WAIT")
    lines.append(f"[SCANNER] CHoCH / BOS / CISD: {tr_status}")

    # 8. Displacement
    dp_stage = stages.get("displacement", {})
    dp_status = dp_stage.get("status")
    if not dp_status:
        dp_status = "CONFIRMED" if status == "VALID" else ("NOT_CHECKED" if "displacement" not in stages else "NOT CONFIRMED")
    elif dp_status in ("WAIT", "WEAK"):
        dp_status = "WEAK" if dp_stage.get("status") == "WEAK" else "NOT CONFIRMED"
    lines.append(f"[SCANNER] M5 Displacement: {dp_status}")

    # 9. Entry Validation
    ev_stage = stages.get("entry_validation", {})
    ev_status = ev_stage.get("status")
    if not ev_status:
        ev_status = "PASSED" if status == "VALID" else ("INVALID" if status == "INVALID" else "WAIT")
    lines.append(f"[SCANNER] Entry Validation: {ev_status}")

    # 10. Risk / Reward & Final Decision
    rr_stage = stages.get("risk_reward", {})
    rr_val = plan.get("rr") or rr_stage.get("rr")
    if status == "VALID" and rr_val:
        lines.append(f"[SCANNER] Risk Reward: 1:{rr_val:.1f}")
        lines.append(f"[SCANNER] Final Decision: {direction} VALID")
    else:
        rr_status = rr_stage.get("status", "NOT_CHECKED" if "risk_reward" not in stages else "WAIT")
        lines.append(f"[SCANNER] Risk Reward: {rr_status}")
        lines.append(f"[SCANNER] Final Decision: {status}")
        reasons = [r.replace("[SCANNER] Rejection Reason: ", "") for r in dec.get("reasons", []) if r]
        rej_msg = reasons[0] if reasons else "Persyaratan entry belum terpenuhi"
        lines.append(f"[SCANNER] Rejection Reason: {rej_msg}")

    return "\n".join(lines)


def build_smc_zone_reason(
    zone: Dict[str, Any],
    h4_bias: Optional[str] = None,
    max_words: Optional[int] = None,
    max_keywords: int = 5,
    h1_bias: Optional[str] = None,
) -> str:
    """Concise 2-5 keyword formation reason for an SMC zone using only detected metadata."""
    if not isinstance(zone, dict):
        return ""
    ev = zone.get("evidence") if isinstance(zone.get("evidence"), dict) else {}
    bos = zone.get("origin_bos") if isinstance(zone.get("origin_bos"), dict) else {}
    sb = zone.get("score_breakdown") if isinstance(zone.get("score_breakdown"), dict) else {}

    keywords: List[str] = []

    bos_type = str(bos.get("tipe") or ev.get("bos_type") or "").upper()
    has_ob = bool(zone.get("ob_candle_time") or ev.get("ob_candle_time"))
    snd_pat = str(zone.get("snd_pattern") or ev.get("snd_pattern") or sb.get("snd_pattern") or "").upper()
    snd_qual = str(zone.get("snd_base_quality") or ev.get("snd_base_quality") or sb.get("snd_base_quality") or "").upper()
    if bos_type == "CHOCH":
        keywords.append("CHoCH OB" if has_ob else "CHoCH")
    elif bos_type == "BOS":
        keywords.append("BOS OB" if has_ob else "BOS")
    elif has_ob:
        keywords.append("OB")

    if snd_pat in ("RBR", "DBR", "RBD", "DBD"):
        keywords.append(f"{snd_pat} Tight" if snd_qual == "TIGHT" else snd_pat)

    sw_status = str(zone.get("liquidity_sweep_status") or "").upper()
    has_sweep = bool(
        sw_status.startswith("CONFIRMED")
        or isinstance(zone.get("liquidity_sweep_evidence") or ev.get("sweep"), dict)
        or _num(sb.get("liquidity_sweep") or sb.get("liquidity_context")) > 0
    )
    if has_sweep:
        keywords.append("Liquidity Sweep")

    fvg_status = str((zone.get("fvg_confirmation") or {}).get("status") or ev.get("fvg_status") or "").upper()
    if fvg_status == "UNTOUCHED":
        keywords.append("FVG Untouched")
    elif fvg_status == "PARTIALLY_MITIGATED":
        keywords.append("FVG Partial")
    elif _num(sb.get("fvg_konfirmasi") or sb.get("fvg_quality")) > 0:
        keywords.append("FVG")

    freshness = str(zone.get("freshness") or ev.get("mitigation_status") or "").upper()
    if freshness == "FRESH":
        keywords.append("Fresh")
    elif freshness == "TESTED_1X":
        keywords.append("Tested 1x")
    elif freshness == "TESTED_2X":
        keywords.append("Tested 2x")

    disp_val = _num(zone.get("displacement_strength") or ev.get("displacement_atr"))
    p_disp = _num(sb.get("displacement") or sb.get("zone_strength"))
    if disp_val >= 2.0 or p_disp >= 16.0:
        keywords.append("Strong Disp")
    elif disp_val >= 1.2 or p_disp > 0:
        keywords.append("Displacement")

    align_tag = ""
    h4_u = str(h4_bias or "").upper()
    h1_u = str(h1_bias or "").upper()
    zt_u = str(zone.get("zone_type") or "").upper()
    want_bias = "BULLISH" if zt_u == "DEMAND" else ("BEARISH" if zt_u == "SUPPLY" else "")
    htf_map = zone.get("htf_alignment") if isinstance(zone.get("htf_alignment"), dict) else {}
    h4_map_val = str(htf_map.get("H4") or "").upper()
    h1_map_val = str(htf_map.get("H1") or "").upper()
    h4_al = bool(
        h4_map_val == "ALIGNED"
        or (want_bias and h4_u == want_bias)
        or (zone.get("is_aligned") and not h4_map_val and h4_u not in ("BULLISH", "BEARISH"))
    )
    h4_ct = bool(
        zone.get("is_counter")
        or h4_map_val in ("OPPOSED", "COUNTER")
        or (want_bias and h4_u in ("BULLISH", "BEARISH") and h4_u != want_bias)
    )
    h1_al = bool(h1_map_val == "ALIGNED" or (want_bias and h1_u == want_bias))
    h1_ct = bool(
        h1_map_val in ("OPPOSED", "COUNTER")
        or (want_bias and h1_u in ("BULLISH", "BEARISH") and h1_u != want_bias)
    )
    if h4_ct:
        align_tag = "Counter H4"
    elif h4_al:
        if h1_al:
            align_tag = "HTF Aligned (H4+H1)"
        elif h1_ct:
            align_tag = "H4 Aligned (vs H1)"
        else:
            align_tag = "H4 Aligned"
    elif _num(sb.get("htf_alignment") or sb.get("konteks_htf")) >= 6.0:
        align_tag = "HTF Aligned (H4+H1)" if _num(sb.get("htf_alignment") or sb.get("konteks_htf")) >= 9.0 else "H4 Aligned"

    merged_from = zone.get("merged_from") or ev.get("merged_from")
    if isinstance(merged_from, list) and len(merged_from) > 1 and len(keywords) < max_keywords - (1 if align_tag else 0):
        keywords.append("MTF Merged")

    limit = max(1, int(max_keywords or 5))
    if align_tag:
        if len(keywords) >= limit:
            keywords = keywords[: limit - 1]
        keywords.append(align_tag)
    else:
        keywords = keywords[:limit]

    if max_words is not None and max_words > 0:
        kept: List[str] = []
        words = 0
        for kw in keywords:
            cnt = len(kw.split())
            if kept and words + cnt > max_words:
                continue
            kept.append(kw)
            words += cnt
        keywords = kept

    return ", ".join(keywords)


def build_smc_invalidation_reason(zone: Dict[str, Any]) -> str:
    """Concise invalidation reason label for an INVALID SMC zone from existing metadata."""
    if not isinstance(zone, dict):
        return ""
    inv_type = str(zone.get("invalidation_type") or "").upper()
    inv_lbl = str(zone.get("invalid_label") or "").upper()
    life = str(zone.get("lifecycle_status") or zone.get("status") or "").upper()
    fresh = str(zone.get("freshness") or "").upper()
    rej_raw = str(zone.get("invalidation_reason") or zone.get("rejection_reason") or "").strip()
    rej_low = rej_raw.lower()

    if (
        inv_type == INVALIDATION_SETUP_SL_BREACHED
        or "SL_BREACHED" in inv_lbl
        or ("sl" in rej_low and any(tok in rej_low for tok in ("breach", "tembus", "hit")))
    ):
        return "SETUP SL BREACHED (live price)"
    if (
        inv_type == INVALIDATION_ZONE_STRUCTURAL
        or "STRUCTURALLY_INVALID" in inv_lbl
        or "BROKEN" in inv_lbl
        or zone.get("is_broken")
        or life in ("INVALIDATED", "BROKEN")
        or fresh == "INVALIDATED"
        or any(tok in rej_low for tok in ("tembus", "broken", "breach", "close di luar", "invalid"))
    ):
        return "STRUCTURE BROKEN (closed TF)"
    if life == "MITIGATED" or fresh == "MITIGATED" or "mitiga" in rej_low:
        return "MITIGATED"
    if life == "EXPIRED" or "expir" in rej_low or "kedaluwarsa" in rej_low:
        return "EXPIRED"
    if rej_raw:
        return "ZONE INVALIDATED"
    return ""


def format_smc_invalid_why(
    zone: Dict[str, Any],
    h4_bias: Optional[str] = None,
    h1_bias: Optional[str] = None,
) -> str:
    """One-line invalidation + formation origin distinction for an INVALID SMC zone."""
    if not isinstance(zone, dict):
        return ""
    inv_why = build_smc_invalidation_reason(zone)
    origin = build_smc_zone_reason(zone, h4_bias=h4_bias, h1_bias=h1_bias) or str(zone.get("origin_reason") or "").strip()
    if inv_why and origin:
        return f"INVALID: {inv_why} | Origin: {origin}"
    if inv_why:
        return f"INVALID: {inv_why}"
    if origin:
        return f"Origin: {origin}"
    return ""


_PLAN_NOTES = (
    ("Konflik SMC-SNR", "BLOCKED BY SNR"), ("Konflik SMC", "BLOCKED BY SNR"), ("Terhalang zona S&R", "BLOCKED BY SNR"),
    ("historis/terlewat", "M5 HISTORICAL/MISSED"),
    ("Risk/Reward", "RR BELOW MINIMUM"), ("Jarak SL", "SL INVALID"), ("sudah di atas/di dalam entry", "PRICE AT/INSIDE ENTRY"),
    ("sudah di bawah/di dalam entry", "PRICE AT/INSIDE ENTRY"), ("Bias H4", "HTF FILTER"), ("Konflik struktur", "HTF FILTER"),
    ("M30", "M30 FILTER"), ("Spread", "SPREAD TOO HIGH"), ("Jarak zona", "ZONE TOO FAR"), ("Data MT5", "DATA STALE"),
    ("invalid", "ZONE INVALID"), ("termitigasi", "ZONE MITIGATED"), ("invalidation", "ZONE INVALID"),
)


def format_zone_evidence(zone: Dict[str, Any], detailed: bool = False) -> Any:
    """Format bukti teknikal zona SMC (TF, waktu candle, range OB/FVG, mitigasi, alasan scoring)."""
    if not isinstance(zone, dict):
        return [] if detailed else "N/A"
    ev = zone.get("evidence") if isinstance(zone.get("evidence"), dict) else {}
    tf = str(ev.get("timeframe") or zone.get("timeframe") or "N/A")
    ob_t = str(ev.get("ob_candle_time") or zone.get("ob_candle_time") or "N/A")
    bos_t = str(ev.get("bos_candle_time") or zone.get("bos_candle_time") or "N/A")
    lo = _num(zone.get("price_low"))
    hi = _num(zone.get("price_high"))
    fvg = zone.get("fvg_confirmation") if isinstance(zone.get("fvg_confirmation"), dict) else {}
    fvg_rng = ev.get("fvg_range") or ([_num(fvg.get("price_low")), _num(fvg.get("price_high"))] if fvg else None)
    fvg_st = str(ev.get("fvg_status") or fvg.get("status") or "NONE").upper()
    fvg_t = str(ev.get("fvg_time") or fvg.get("time") or "N/A")
    fvg_txt = f"{_num(fvg_rng[0]):.2f}-{_num(fvg_rng[1]):.2f} ({fvg_st})" if (isinstance(fvg_rng, (list, tuple)) and len(fvg_rng) == 2 and _num(fvg_rng[0]) > 0) else "NONE"
    fresh = str(ev.get("mitigation_status") or zone.get("freshness") or "FRESH").upper()
    touches = int(ev.get("touch_count") if ev.get("touch_count") is not None else (zone.get("touch_count") or 0))
    sc_reason = str(ev.get("score_reason") or zone.get("score_reason") or build_smc_zone_reason(zone) or "N/A")

    if not detailed:
        ob_t_short = ob_t[:16] if len(ob_t) >= 16 else ob_t
        bos_t_short = bos_t[:16] if len(bos_t) >= 16 else bos_t
        snd_p = str(ev.get("snd_pattern") or zone.get("snd_pattern") or "").upper()
        snd_q = str(ev.get("snd_base_quality") or zone.get("snd_base_quality") or "").upper()
        snd_tag = f" | SND {snd_p}/{snd_q or 'STD'}" if snd_p in ("RBR", "DBR", "RBD", "DBD") else ""
        return f"{tf} | OB {ob_t_short} [{lo:.2f}-{hi:.2f}]{snd_tag} | BOS {bos_t_short} | FVG {fvg_txt} | {fresh} ({touches}x)"

    bos = zone.get("origin_bos") if isinstance(zone.get("origin_bos"), dict) else {}
    bos_type = str(ev.get("bos_type") or bos.get("tipe") or "BOS")
    bos_lvl = _num(ev.get("bos_level") or bos.get("level"))
    bos_str = _num(ev.get("bos_strength_atr") or bos.get("strength_atr"))
    ob_bars = ev.get("ob_to_bos_bars")
    bars_txt = f" | Leg {int(ob_bars)} bars" if isinstance(ob_bars, (int, float)) else ""
    closed_txt = "Closed" if ev.get("candle_closed", True) else "Unverified"
    fvg_c3 = str(ev.get("fvg_c3_time") or fvg.get("c3_time") or "")
    fvg_time_txt = f"{fvg_t} -> c3 {fvg_c3}" if (fvg_c3 and fvg_c3 != "N/A") else fvg_t
    disp_atr = _num(ev.get("displacement_atr") or zone.get("displacement_strength"))
    body_atr = _num(ev.get("impulse_max_body_atr") or bos.get("impulse_max_body_atr"))
    sw_ev = ev.get("sweep") or zone.get("liquidity_sweep_evidence")
    sw_txt = (
        f"{sw_ev.get('type')} @ {_num(sw_ev.get('level')):.2f} (wick {_num(sw_ev.get('wick_extreme')):.2f}, {sw_ev.get('time', 'N/A')})"
        if isinstance(sw_ev, dict)
        else str(zone.get("liquidity_sweep_status") or "NONE")
    )
    lines = [
        f"    - Bukti OB & BOS     : TF {tf} ({closed_txt}) | OB {ob_t} [{lo:.2f}-{hi:.2f}] | {bos_type} {bos_t} @ {bos_lvl:.2f} ({bos_str:.2f} ATR{bars_txt})",
        f"    - Bukti FVG & Impuls : FVG {fvg_txt} ({fvg_time_txt}) | Disp {disp_atr:.2f}x ATR | Max Body {body_atr:.2f}x ATR",
        f"    - Mitigasi & Sweep   : {fresh} (Touch {touches}x) | Sweep: {sw_txt}",
        f"    - Alasan Scoring     : {sc_reason}",
    ]
    merged_from = ev.get("merged_from") or zone.get("merged_from")
    if isinstance(merged_from, list) and len(merged_from) > 1:
        lines.append(f"    - Merged Sources     : {', '.join(str(m) for m in merged_from)}")
    return lines


def _num(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else 0.0


def smc_zone_summary(zone: Dict[str, Any], decision: Dict[str, Any], allow_plan: bool = True) -> Dict[str, Any]:
    """Single source for SMC zone reason, risk plan and status; levels always come from existing planners."""
    zone_id = zone.get("zone_id")
    active = bool(zone_id) and (decision.get("zone") or {}).get("zone_id") == zone_id
    expected = "BUY" if zone.get("zone_type") == "DEMAND" else "SELL" if zone.get("zone_type") == "SUPPLY" else ""
    lifecycle = str(zone.get("lifecycle_status") or zone.get("status") or "").upper()
    grd_raw = str(zone.get("grade") or "N/A").upper()
    terminal = lifecycle in ("INVALIDATED", "EXPIRED", "MITIGATED", "REJECTED") or zone.get("is_valid") is False or grd_raw == "REJECTED"
    is_mkt_closed = (
        str(decision.get("market_status") or "").upper() == "CLOSED"
        or str(decision.get("snapshot_status") or "").upper() == "MARKET_CLOSED"
    )
    stages = decision.get("stages") or {}
    if terminal:
        code = lifecycle if lifecycle in ("INVALIDATED", "EXPIRED", "MITIGATED", "REJECTED") else "INVALID"
    elif is_mkt_closed:
        code = "RADAR (MARKET CLOSED — WAIT)"
    elif active and decision.get("status") == "INVALID":
        code = "INVALID"
    elif active and (stages.get("zone_validation") or {}).get("status") in ("WAIT", "DATA_INCOMPLETE"):
        code = "WAIT | " + str((stages.get("zone_validation") or {}).get("msg") or "ZONE VALIDATION")
    elif active and (stages.get("zone_touch") or {}).get("status") not in ("CONFIRMED", "NOT_REQUIRED"):
        code = "WAIT_ZONE_TOUCH"
    elif active and (
        (stages.get("structure_trigger") or {}).get("status") != "CONFIRMED"
        or (stages.get("displacement") or {}).get("status") != "CONFIRMED"
    ):
        code = "WAIT_M5_CONFIRMATION"
    elif active and decision.get("status") != "VALID" and (
        (stages.get("risk_reward") or {}).get("status") != "CONFIRMED"
        or (stages.get("entry_validation") or {}).get("status") in ("WAIT", "FAILED", "FAIL", "INVALID")
    ):
        rr_ev_msg = str(
            (stages.get("risk_reward") or {}).get("msg")
            or (stages.get("entry_validation") or {}).get("msg")
            or ""
        )
        if "Jarak SL" in rr_ev_msg or "melebihi batas" in rr_ev_msg:
            code = "RADAR | SETUP REJECTED (SL > MAX)"
        elif "Konflik SMC" in rr_ev_msg or "S&R" in rr_ev_msg or "SNR" in rr_ev_msg:
            code = "WAIT | M5 CONFIRMED (SNR CONFLICT)"
        elif "historis" in rr_ev_msg or "terlewat" in rr_ev_msg:
            code = "WAIT | M5 HISTORICAL/MISSED"
        elif (stages.get("risk_reward") or {}).get("rr") is not None and "R:R" in rr_ev_msg:
            code = "RR_FAILED"
        else:
            code = "WAIT"
    elif active:
        code = "WAIT"
    else:
        code = "RADAR"

    plan, kind, note = None, "NONE", "PLAN UNAVAILABLE"
    computed = (decision.get("zone_plans") or {}).get(zone_id) or {}
    if allow_plan and not terminal:
        confirmed = decision.get("plan") or {}
        if (
            not is_mkt_closed
            and active and decision.get("status") == "VALID" and confirmed and decision.get("direction") == expected
            and (not confirmed.get("zone_id") or confirmed.get("zone_id") == zone_id)
        ):
            plan, kind = confirmed, "CONFIRMED"
        else:
            ctx = decision.get("context") or {}
            h4_st = str(ctx.get("htf_bias") or ctx.get("H4") or "").upper()
            m30_st_info = stages.get("M30") or {}
            m30_st = str(m30_st_info.get("status") or ctx.get("m30_status") or ("CONFIRMED" if m30_st_info.get("ok") else "")).upper()
            h1_ok = (stages.get("H1") or {}).get("ok", True)
            pending_gate_ok = (
                not is_mkt_closed
                and decision.get("status") != "INVALID"
                and not ctx.get("has_conflict")
                and h1_ok is not False
                and (not h4_st or h4_st in ("BULLISH", "BEARISH"))
                and m30_st not in ("CONFLICT", "INVALID", "WEAK")
            )
            pending = decision.get("early_setup") or {}
            qualified = list(pending.get("all_setups") or []) + ([pending["best_setup"]] if pending.get("best_setup") else [])
            plan = next((p for p in qualified if p.get("zone_id") == zone_id and p.get("valid") and p.get("status") == "READY"), None)
            if plan and pending_gate_ok:
                best_zid_s = str((pending.get("best_setup") or {}).get("zone_id") or "")
                kind = "SELECTED_PENDING" if (best_zid_s and zone_id == best_zid_s) else "ALT_PENDING"
            elif plan:
                kind = "PROVISIONAL"
            elif (computed.get("valid") and computed.get("status") == "READY" and not is_mkt_closed) or (
                computed.get("status") in ("PROVISIONAL", "WATCHLIST") or computed.get("provisional")
            ):
                plan = computed
                kind = "PROVISIONAL" if (computed.get("provisional") or is_mkt_closed) else ("WATCHLIST" if computed.get("status") == "WATCHLIST" else "PLAN")
            else:
                text = str(computed.get("rejection_reason") or zone.get("pending_rejection_reason") or "")
                if "jarak sl" in text.lower() and "melebihi batas" in text.lower():
                    note = f"SETUP REJECTED: {text}"
                else:
                    note = next((label for key, label in _PLAN_NOTES if key.lower() in text.lower()), text[:40].upper() or "PLAN UNAVAILABLE")
    elif terminal:
        note = "ZONE NOT VALID"
    elif not allow_plan:
        rej_txt = str(computed.get("rejection_reason") or zone.get("pending_rejection_reason") or "")
        if "jarak sl" in rej_txt.lower() and "melebihi batas" in rej_txt.lower():
            note = f"SETUP REJECTED: {rej_txt}"
        else:
            note = "MARKET CLOSED (WAIT REVALIDATION)" if is_mkt_closed else "DATA NOT FRESH"

    entry = _num((plan or {}).get("entry_acuan", (plan or {}).get("entry_price")))
    sl, tp1, tp2 = _num((plan or {}).get("sl")), _num((plan or {}).get("tp1")), _num((plan or {}).get("tp2"))
    valid = bool(
        plan and expected and entry > 0 and sl > 0 and tp1 > 0
        and (sl < entry < tp1 if expected == "BUY" else tp1 < entry < sl)
    )
    if plan and not valid:
        plan, kind, note = None, "NONE", "LEVELS INVALID"
    if valid and (tp2 == tp1 or not (tp2 > entry if expected == "BUY" else 0 < tp2 < entry)):
        tp2 = 0.0
    status = code
    plan_status_val = code
    broker_status_val = "UNKNOWN"
    recon_val = None
    if kind == "CONFIRMED":
        status = "CONFIRMED"
        plan_status_val = "CONFIRMED"
    elif kind in ("SELECTED_PENDING", "ALT_PENDING"):
        plan_status_val = "READY"
        if isinstance(plan, dict) and not isinstance(plan.get("mt5_reconciliation"), dict):
            pending_obj = decision.get("early_setup") or {}
            batch_list = []
            seen_b = set()
            for p_cand in ([pending_obj.get("best_setup")] if isinstance(pending_obj.get("best_setup"), dict) else []) + list(pending_obj.get("all_setups") or []):
                if isinstance(p_cand, dict) and p_cand.get("valid") and str(p_cand.get("status") or "").upper().startswith("READY"):
                    pz_id = str(p_cand.get("zone_id") or "")
                    if pz_id not in seen_b:
                        seen_b.add(pz_id)
                        batch_list.append(p_cand)
            if plan not in batch_list:
                batch_list.append(plan)
            reconcile_pending_plans_batch(
                batch_list,
                mt5_snapshot=decision.get("mt5_orders_snapshot"),
                symbol=zone.get("symbol") or decision.get("symbol") or SYMBOL,
            )
        recon_val = (plan or {}).get("mt5_reconciliation")
        broker_status_val = str((plan or {}).get("broker_order_status") or (recon_val or {}).get("broker_order_status") or "UNKNOWN").upper()
        c_st = (plan or {}).get("compact_status") or format_pending_compact_status(
            "READY",
            recon_val,
            order_type=str((plan or {}).get("order_type") or ""),
            entry_price=entry,
        )
        role_tag = "SELECTED PENDING" if kind == "SELECTED_PENDING" else "ALT PENDING"
        status = f"{code} | {role_tag} {c_st}"
    elif kind == "PROVISIONAL":
        status = f"{code} (PROVISIONAL ONLY — WAIT)"
    elif kind == "WATCHLIST":
        status = f"{code} (WATCHLIST — FAR ZONE)"
    elif kind == "PLAN":
        status = f"{code} (VALIDATED PLAN)"
    elif "SETUP REJECTED" in note or zone.get("risk_rejected") or str(zone.get("setup_eligibility") or "").upper().startswith("NOT_ELIGIBLE"):
        status = "RADAR (SETUP NOT ELIGIBLE — MARKET CLOSED)" if is_mkt_closed else "RADAR (SETUP NOT ELIGIBLE)"

    grade_disp = f"{zone.get('grade', 'N/A')} ({_num(zone.get('score')):.2f})" if zone.get("score") is not None else str(zone.get("grade", "N/A"))
    fresh_disp = str(zone.get("freshness") or "FRESH").upper()
    zone_quality_val = zone.get("zone_quality") or (
        f"REJECTED ({zone.get('rejection_reason') or lifecycle})"
        if terminal
        else f"VALID (Grade {grade_disp} | {fresh_disp})"
    )
    if terminal:
        setup_elig_val = f"NOT_ELIGIBLE ({zone.get('rejection_reason') or 'Zona tidak valid'})"
    elif kind in ("CONFIRMED", "SELECTED_PENDING", "ALT_PENDING", "PLAN") and not str(zone.get("setup_eligibility") or "").upper().startswith("NOT_ELIGIBLE"):
        setup_elig_val = "ELIGIBLE"
    elif kind == "PROVISIONAL" and not str(zone.get("setup_eligibility") or "").upper().startswith("NOT_ELIGIBLE"):
        setup_elig_val = "ELIGIBLE (PROVISIONAL — MARKET CLOSED)" if is_mkt_closed else "ELIGIBLE (PROVISIONAL)"
    else:
        rej_s = str(zone.get("pending_rejection_reason") or computed.get("rejection_reason") or note)
        setup_elig_val = zone.get("setup_eligibility") or f"NOT_ELIGIBLE ({rej_s})"

    if is_mkt_closed:
        entry_perm_val = "WAIT (MARKET CLOSED — Revalidasi saat market OPEN)"
    elif kind == "CONFIRMED" and decision.get("status") == "VALID":
        entry_perm_val = "GRANTED (CONFIRMED ENTRY M5)"
    elif not setup_elig_val.upper().startswith("ELIGIBLE"):
        entry_perm_val = "WAIT — BLOCKED (Setup Not Eligible)"
    else:
        entry_perm_val = "WAIT (Menunggu konfirmasi trigger M5)"

    short_note = "SETUP NOT ELIGIBLE" if not setup_elig_val.upper().startswith("ELIGIBLE") else note
    radar_state_val = resolve_radar_state(
        zone,
        decision=decision,
        current_price=_num(decision.get("current_price")) or None,
    )
    return {
        "zone_id": str(zone_id or "N/A"),
        "radar_state": radar_state_val,
        "evidence": format_zone_evidence(zone, detailed=False),
        "label": (
            f"{zone.get('timeframe', '')} {str(zone.get('zone_type', '')).upper()} "
            f"{_num(zone.get('price_low')):.2f}-{_num(zone.get('price_high')):.2f}"
        ),
        "grade": grade_disp,
        "zone_quality": zone_quality_val,
        "setup_eligibility": setup_elig_val,
        "entry_permission": entry_perm_val,
        "reason": build_smc_zone_reason(zone) or "N/A",
        "direction": expected, "kind": kind, "note": short_note, "status": status, "active": active,
        "plan_status": plan_status_val, "broker_order_status": broker_status_val, "mt5_reconciliation": recon_val,
        "entry": entry if valid else 0.0, "sl": sl if valid else 0.0, "tp1": tp1 if valid else 0.0,
        "tp2": tp2 if valid else 0.0,
        "rr": abs(tp1 - entry) / abs(entry - sl) if valid else 0.0,
    }


def format_smc_zone_lines(zone: Dict[str, Any], decision: Dict[str, Any], allow_plan: bool = True, role: str = "SMC ZONE") -> List[str]:
    """Compact multi-line zone block shared by scan, scan gabungan and pantau."""
    if not isinstance(zone, dict) or not zone:
        return [
            f"{role:<17} : NONE",
            "ZONE QUALITY      : NONE",
            "SETUP ELIGIBILITY : NOT_ELIGIBLE (No active SMC zone)",
            "ENTRY PERMISSION  : WAIT",
            "REASON            : no active SMC zone",
            "ENTRY             : N/A",
            "SL / TP1 / TP2    : N/A / N/A / N/A",
            "RR                : NOT CHECKED",
            "STATUS            : WAIT",
        ]
    s = smc_zone_summary(zone, decision, allow_plan)
    if s["kind"] == "CONFIRMED":
        entry_text = f"{s['direction']} {s['entry']:.2f}"
    elif s["kind"] in ("SELECTED_PENDING", "ALT_PENDING", "PENDING", "PROVISIONAL", "WATCHLIST", "PLAN") and s["entry"] > 0:
        entry_text = f"{s['direction']} PLAN {s['entry']:.2f}"
    else:
        entry_text = f"N/A ({s['note']})"
    has_plan = s["entry"] > 0
    tp_text = (
        f"{s['tp1']:.2f} / {s['tp2']:.2f}" if s["tp2"] > 0 else f"{s['tp1']:.2f} / N/A"
    ) if has_plan else "N/A / N/A"
    return [
        f"{role:<17} : {s['label']} | Grade {s['grade']} [ID: {s['zone_id']}]",
        f"ZONE QUALITY      : {s['zone_quality']} | Radar: {s['radar_state']}",
        f"SETUP ELIGIBILITY : {s['setup_eligibility']}",
        f"ENTRY PERMISSION  : {s['entry_permission']}",
        f"EVIDENCE          : {s['evidence']}",
        f"REASON            : {s['reason']}",
        f"ENTRY             : {entry_text}",
        f"SL                : {s['sl']:.2f}" if has_plan else "SL                : N/A",
        f"TP1 / TP2         : {tp_text}",
        f"RR                : 1:{s['rr']:.2f}" if has_plan else "RR                : NOT CHECKED",
        f"STATUS            : {s['status']}",
    ]


def format_smc_zone_plan(zone: Dict[str, Any], decision: Dict[str, Any], allow_plan: bool = True) -> str:
    """One-line variant for non-focus zones."""
    s = smc_zone_summary(zone, decision, allow_plan)
    levels = (
        f"{s['entry']:.2f} / {s['sl']:.2f} / {s['tp1']:.2f} / {s['tp2']:.2f}".replace("/ 0.00", "/ N/A") + f" | RR 1:{s['rr']:.2f}"
        if s["entry"] > 0 else f"N/A ({s['setup_eligibility']})"
    )
    return f"{s['reason']} | ENTRY/SL/TP1/TP2 {levels} | {s['status']}"


def premarket_lines(dec: Optional[Dict[str, Any]]) -> List[str]:
    mkt = str((dec or {}).get("market_status") or "").upper()
    snap_st = str((dec or {}).get("snapshot_status") or "").upper()
    if mkt != "CLOSED" and snap_st != "MARKET_CLOSED":
        return []
    pm = (dec or {}).get("premarket")
    if not isinstance(pm, dict):
        return []
    _u_pm = extract_unit_spec_from_container(dec, symbol=(dec or {}).get("symbol") or SYMBOL)
    _d_pm = _u_pm["digits"]
    ct = pm.get("candle_times") or {}
    ts = " | ".join(f"{k}={ct[k]}" for k in ("H4", "H1", "M30", "M5") if ct.get(k))
    px = pm.get("price")
    cands = pm.get("candidates") or pm.get("plans") or []
    rej_cands = pm.get("rejected_candidates") or []
    far_refs = pm.get("far_references") or []
    elig_status = pm.get("eligibility_status") or (
        "ELIGIBLE_SETUPS_FOUND" if cands else ("NO_ELIGIBLE_SETUPS" if (rej_cands or far_refs) else "NO_ZONES_FOUND")
    )
    L = [
        "PRE-MARKET ANALYSIS (MARKET CLOSED — PROVISIONAL, bukan sinyal live / izin entry)",
        f"- Ref. Harga     : {px:.{_d_pm}f} = LAST CLOSED CANDLE M5 @ {pm.get('price_time') or 'N/A'} (sesi terakhir; BUKAN LIVE)" if px else "- Ref. Harga     : N/A",
        f"- Last Tick     : {pm.get('last_tick_time') or 'N/A (tidak dipakai untuk jarak/entry)'}",
        f"- Candle Time   : {ts or 'N/A'}",
        f"- Symbol Spec   : Digits={_u_pm['digits']} | Point={_u_pm['point']:g} USD | TickSize={_u_pm['trade_tick_size']:g} USD",
        f"- Status Seleksi: {elig_status} | LIVE SIGNAL: WAIT (MARKET CLOSED)",
        "",
        "KANDIDAT SETUP ELIGIBLE PRE-MARKET (Maksimal 3 Zona Layak):",
    ]
    if not cands:
        L.append(f"- {pm.get('reason') or f'{elig_status}: Belum ada kandidat setup yang lolos filter kelayakan'}")
    for i, c in enumerate(cands[:3], 1):
        d = c.get("direction") or ("BUY" if "BUY" in str(c.get("order_type", "")) else "SELL")
        tf = c.get("zone_timeframe") or c.get("timeframe", "H1")
        zt = c.get("zone_type", "ZONE")
        sub_t = c.get("zone_subtype") or c.get("ob_type") or ""
        type_lbl = f"{zt} ({sub_t})" if sub_t and sub_t != zt else zt
        z_lo = float(c.get("zone_low", 0.0))
        z_hi = float(c.get("zone_high", 0.0))
        dist_u = float(c.get("distance", 0.0))
        dist_atr = c.get("distance_atr_ratio")
        atr_str = f" / {dist_atr:.1f}x ATR" if isinstance(dist_atr, (int, float)) else ""
        grd = c.get("zone_grade") or c.get("grade", "-")
        sc = float(c.get("zone_score") or c.get("score") or 0.0)
        st = c.get("status", "PROVISIONAL")
        z_qual = c.get("zone_quality") or f"VALID (Grade {grd} | {sc:.1f})"
        s_elig = c.get("setup_eligibility") or "ELIGIBLE (PROVISIONAL)"
        e_perm = c.get("entry_permission") or "WAIT (MARKET CLOSED)"
        confl = c.get("validity_reason") or c.get("confluence") or c.get("alasan") or "Struktur valid MT5"
        ep = float(c.get("entry_price") or 0.0)
        sl = float(c.get("sl") or 0.0)
        tp1 = float(c.get("tp1") or 0.0)
        rr = float(c.get("rr") or c.get("rr_tp1") or 0.0)
        plan_str = f"Entry {ep:.{_d_pm}f} | SL {sl:.{_d_pm}f} | TP1 {tp1:.{_d_pm}f} | RR 1:{rr:.2f} (WAIT — Revalidasi saat market OPEN)" if (ep > 0 and sl > 0 and tp1 > 0) else "Setup belum lengkap (WAIT)"

        L.append(
            f"- KANDIDAT {i} [{st}]: {d} {tf} {type_lbl} | Range: {z_lo:.{_d_pm}f} - {z_hi:.{_d_pm}f} "
            f"| Jarak: {format_price_distance(dist_u, spec=_u_pm, include_mt5_points=True)}{atr_str} | Grade: {grd} ({sc:.1f})"
        )
        L.append(f"  * Quality/Elig  : {z_qual} | Setup: {s_elig} | Entry: {e_perm} | Conf: {confl}")
        L.append(f"  * Planned Setup : {plan_str}")
    if rej_cands:
        L.append("")
        L.append("KANDIDAT DITOLAK TERDEKAT (Maksimal 3 Zona — Tetap RADAR, Bukan Setup Aktif):")
        for rc in rej_cands[:3]:
            rc_tf = rc.get("timeframe", "")
            rc_type = rc.get("zone_type", "")
            rc_lo = float(rc.get("price_low", 0.0))
            rc_hi = float(rc.get("price_high", 0.0))
            rc_dist = float(rc.get("distance", 0.0))
            rc_atr = rc.get("distance_atr_ratio")
            rc_atr_s = f" / {rc_atr:.1f}x ATR" if isinstance(rc_atr, (int, float)) else ""
            rc_grd = rc.get("grade") or "-"
            rc_sc = float(rc.get("score") or 0.0)
            L.append(
                f"- {rc_tf} {rc_type} {rc_lo:.{_d_pm}f}-{rc_hi:.{_d_pm}f} [Grade {rc_grd} ({rc_sc:.1f}) | "
                f"Jarak {format_price_distance(rc_dist, spec=_u_pm, include_mt5_points=True)}{rc_atr_s}] -> NOT_ELIGIBLE: {rc.get('reason')}"
            )
    if far_refs:
        n_far_intra = sum(
            1 for fr_i in far_refs
            if (fr_i.get("far_category") == "FAR_INTRADAY")
            or (not fr_i.get("far_category") and (not isinstance(fr_i.get("distance_atr_ratio"), (int, float)) or float(fr_i["distance_atr_ratio"]) <= 15.0))
        )
        n_ext_far = len(far_refs) - n_far_intra
        L.append("")
        L.append(
            f"ZONA RADAR JAUH / FAR REFERENCE (Maks 3 Relevan dari Total {len(far_refs)}: "
            f"FAR_INTRADAY={n_far_intra} [> batas TF/poin, <=15x ATR] | EXTREME_FAR={n_ext_far} [>15x ATR]):"
        )
        for fr in far_refs[:3]:
            fr_tf = fr.get("timeframe", "")
            fr_zt = fr.get("zone_type", "")
            fr_lo = float(fr.get("price_low", 0.0))
            fr_hi = float(fr.get("price_high", 0.0))
            fr_dist = float(fr.get("distance", 0.0))
            fr_atr = fr.get("distance_atr_ratio")
            fr_atr_s = f" / {fr_atr:.1f}x ATR" if isinstance(fr_atr, (int, float)) else ""
            fr_grd = fr.get("grade") or "-"
            fr_sc = float(fr.get("score") or 0.0)
            fr_cat = fr.get("far_category") or ("EXTREME_FAR" if (isinstance(fr_atr, (int, float)) and fr_atr > 15.0) else "FAR_INTRADAY")
            fr_lim = fr.get("radar_limit_pt")
            fr_note = f"> 15x ATR" if fr_cat == "EXTREME_FAR" else (f"> batas intraday {fr_tf} {float(fr_lim):.0f} USD" if isinstance(fr_lim, (int, float)) and fr_lim > 0 else "> batas intraday TF")
            L.append(
                f"- [{fr_cat}] {fr_tf} {fr_zt} {fr_lo:.{_d_pm}f}-{fr_hi:.{_d_pm}f} | Grade {fr_grd} ({fr_sc:.1f}) "
                f"| Jarak: {format_price_distance(fr_dist, spec=_u_pm, include_mt5_points=True)}{fr_atr_s} ({fr_note})"
            )
            fr_why = build_smc_zone_reason(fr)
            if fr_why:
                L.append(f"  * WHY           : {fr_why}")
    L.append("")
    L.append("- Saat market OPEN wajib validasi ulang: harga, gap, zona, SL/TP, RR, dan konfirmasi M5 sebelum sinyal aktif.")
    return L


def format_scan_presentation(
    scan: Optional[Dict[str, Any]] = None,
    dec: Optional[Dict[str, Any]] = None,
    early_setups: Optional[Dict[str, Any]] = None,
    tracker: Optional[Any] = None,
    detailed: bool = False,
) -> str:
    """
    Format ringkas visual command 'scan' sesuai aturan Otoritas Keputusan Python Scanner:
    5 Bagian Utama:
      A. KONDISI PASAR (Snapshot harga, sumber tick/close, status multi-TF bias)
      B. RADAR ZONE (Primary Directional Zone + Nearest Radar Zone; kandidat hanya tampil jika detailed=True)
      C. ENTRY PLAN (Early Pending Plan vs Confirmed Entry M5)
      D. FINAL DECISION & REASON (Keputusan resmi Python Scanner + dynamic reason)
      E. RISK WARNING (Peringatan singkat MM, pending order & kewajiban SL)
    """
    if scan is None:
        # Tidak ada scan: buat SATU snapshot baru + scan baru. Keputusan (dec) lama yang diberikan berasal
        # dari snapshot lain, sehingga TIDAK boleh digabung dengan scan baru -> dievaluasi ulang dari snapshot baru.
        _pres_snap = create_market_snapshot()
        scan = scan_xauusd_zones(snapshot=_pres_snap)
        dec = evaluate_trade_decision(snapshot=_pres_snap, scan_result=scan)
    elif dec is None:
        # Ada scan, tanpa dec: evaluasi memakai objek snapshot yang tersimpan pada scan (tanpa fetch baru).
        _pres_snap = scan.get("snapshot") if isinstance(scan.get("snapshot"), dict) else None
        dec = evaluate_trade_decision(snapshot=_pres_snap, scan_result=scan)

    price = scan.get("current_price") or dec.get("current_price") or 0.0
    ctx = scan.get("htf_context") or {}
    h4_bias = ctx.get("H4", "NETRAL")
    h1_struct = ctx.get("H1", "NETRAL")
    m30_val = dec.get("stages", {}).get("M30", {})
    if not m30_val and ctx:
        m30_val = ctx.get("m30_eval") or {}
    m30_dir = ctx.get("M30", "NETRAL")
    m30_status = m30_val.get("status") or ctx.get("m30_status") or ("CONFIRMED" if m30_val.get("ok") else "UNCONFIRMED")

    zone = dec.get("zone")
    zone_dict = zone.to_dict() if hasattr(zone, "to_dict") else (dict(zone) if isinstance(zone, dict) else {})

    stages = dec.get("stages", {})
    status = dec.get("status", "WAIT")
    direction = dec.get("direction")
    plan = dec.get("plan")

    def _cand_label(c_dict):
        if not c_dict:
            return "NONE"
        sc = f"Skor {c_dict.get('score', 0.0):.1f}" if c_dict.get("scoring_status") != "DATA_INCOMPLETE" else "DATA INCOMPLETE"
        return f"{c_dict.get('timeframe')} {c_dict.get('zone_type')} ({c_dict.get('price_low', 0.0):.2f}-{c_dict.get('price_high', 0.0):.2f}) [Grade {c_dict.get('grade')} | {sc}]"

    def _format_multi_tier_zone_item(z: Dict[str, Any], curr_p: float) -> List[str]:
        tf = z.get("timeframe", "H1")
        zt = z.get("zone_type", "")
        lo = float(z.get("price_low", 0.0))
        hi = float(z.get("price_high", 0.0))
        prox = float(z.get("proximal_edge") or (hi if zt == "DEMAND" else lo))
        distal = float(z.get("distal_edge") or (lo if zt == "DEMAND" else hi))
        w = float(z.get("zone_width") or round(abs(hi - lo), 2))
        grd = z.get("grade", "B")
        sc_st = z.get("scoring_status", "VALID")
        sc_val = f"{z.get('score', 0.0):.1f}" if sc_st != "DATA_INCOMPLETE" else "DATA INCOMPLETE"
        frsh = z.get("freshness", "FRESH")
        inval = float(z.get("invalidation_price") or distal)
        dist = float(z.get("distance", 0.0))
        d_st = z.get("direction_status") or ("ABOVE_PRICE" if lo > curr_p else ("BELOW_PRICE" if hi < curr_p else "INSIDE_ZONE"))
        app = z.get("approach", "UNTOUCHED")
        atr_r = z.get("distance_atr_ratio")
        atr_str = f"{atr_r:.1f}x ATR" if (atr_r is not None and isinstance(atr_r, (int, float))) else "DATA INCOMPLETE"

        # Reference price & nearest boundary
        ref_p = z.get("reference_price")
        ref_t = z.get("reference_price_type")
        if ref_p is None or not (isinstance(ref_p, (int, float)) and ref_p > 0):
            if zt == "SUPPLY" and isinstance(scan.get("ask"), (int, float)) and scan.get("ask") > 0:
                ref_p = float(scan["ask"])
                ref_t = "ASK"
            elif zt == "DEMAND" and isinstance(scan.get("bid"), (int, float)) and scan.get("bid") > 0:
                ref_p = float(scan["bid"])
                ref_t = "BID"
            else:
                ref_p = curr_p
                ref_t = "MID"
        elif not ref_t:
            ref_t = "ASK" if zt == "SUPPLY" else "BID"
        ref_str = f" | Ref: {ref_p:.2f} ({ref_t}) -> Proximal: {prox:.2f}"

        # Directional context transparan
        is_al = z.get("is_aligned", False)
        is_ct = z.get("is_counter", False)
        if is_al:
            dir_context = f"PRO-TREND (Searah bias HTF {h4_bias})"
        elif is_ct or (h4_bias in ("BULLISH", "BEARISH") and ((h4_bias == "BULLISH" and zt == "SUPPLY") or (h4_bias == "BEARISH" and zt == "DEMAND"))):
            dir_context = f"COUNTERTREND (Melawan bias HTF {h4_bias} — Pantau reaksi, BUKAN sinyal entry)"
        else:
            dir_context = f"NETRAL (Bias HTF {h4_bias})"

        struct_st = "VALID" if (z.get("is_valid", True) and "invalid" not in str(z.get("rejection_reason", "")).lower()) else "INVALID"
        p_elig = z.get("pending_eligibility", "NOT_QUALIFIED")
        p_rsn = z.get("pending_rejection_reason", "")
        c_elig = z.get("confirmed_entry_eligibility", "WAIT")
        c_rsn = z.get("confirmed_rejection_reason", "")

        radar_st_item = resolve_radar_state(z, decision=dec, current_price=curr_p)
        why_mt = build_smc_zone_reason(z, h4_bias=h4_bias)
        return [
            f"    - Zone ID & Tipe     : {z.get('zone_id', 'N/A')} | {tf} {zt} [{lo:.2f} - {hi:.2f}] (Proximal: {prox:.2f}, Distal: {distal:.2f}, Lebar: {format_price_distance(w, spec=_u_spec_pres)})",
            f"    - Kualitas & Skor    : Grade {grd} | Skor: {sc_val} | {frsh} | Invalidation: {inval:.2f}",
            *([f"    - WHY                : {why_mt}"] if why_mt else []),
            *format_zone_evidence(z, detailed=True),
            f"    - Directional Context: {dir_context}",
            f"    - Jarak & Volatilitas: {format_price_distance(dist, spec=_u_spec_pres, include_mt5_points=True)} ({atr_str}) [{app}] ({d_st}){ref_str}",
            f"    - Status Monitoring  : {radar_st_item} (Raw: {z.get('status', 'RADAR')}) | Relevansi: {z.get('zone_relevance', 'RADAR')}",
            f"    - Alasan Klasifikasi : {z.get('risk_tier_reason') or 'Sesuai ambang batas tier'}",
            f"    - Validitas Struktur : {struct_st}",
            f"    - Pending Eligibility: {p_elig}" + (f" ({p_rsn})" if p_rsn else ""),
            f"    - Confirmed M5       : {c_elig}" + (f" ({c_rsn})" if c_rsn else ""),
            f"    - Rencana Zona       : {format_smc_zone_plan(z, dec)}",
        ]

    # Unified MT5 Orders & Positions Snapshot (Read-Only, termasuk manual order tanpa magic number)
    _sym_pres = scan.get("symbol") or dec.get("symbol") or SYMBOL
    _u_spec_pres = extract_unit_spec_from_container(scan, symbol=_sym_pres)
    _d_pres_px = _u_spec_pres["digits"]
    _ord_snap_pres = scan.get("mt5_orders_snapshot") or dec.get("mt5_orders_snapshot")
    if not (isinstance(_ord_snap_pres, dict) and "query_ok" in _ord_snap_pres and "orders" in _ord_snap_pres):
        _ord_snap_pres = fetch_mt5_pending_orders_snapshot(symbol=_sym_pres, orders_override=_ord_snap_pres)

    # Early Setups (Aggressive Pending)
    if early_setups is None:
        early_setups = dec.get("early_setup")
    if early_setups is None:
        try:
            early_setups = detect_early_setups(current_price=price, scan=scan)
        except Exception:
            early_setups = {"best_setup": None, "all_setups": [], "empty_reason": "Data tidak tersedia"}
    _matched_recons_pres = []
    if isinstance(early_setups, dict):
        _es_plans = []
        _seen_es = set()
        for _ep in ([early_setups.get("best_setup")] if isinstance(early_setups.get("best_setup"), dict) else []) + list(early_setups.get("all_setups") or []):
            if isinstance(_ep, dict) and _ep.get("valid") and str(_ep.get("status") or "READY").upper().startswith("READY"):
                _ezid = str(_ep.get("zone_id") or "")
                _ekey = _ezid or f"_idx_{len(_es_plans)}"
                if _ekey not in _seen_es:
                    _seen_es.add(_ekey)
                    _es_plans.append(_ep)
        if _es_plans:
            reconcile_pending_plans_batch(_es_plans, mt5_snapshot=_ord_snap_pres, symbol=_sym_pres)
            for _ep in _es_plans:
                _ep_rec = _ep.get("mt5_reconciliation")
                if isinstance(_ep_rec, dict):
                    _matched_recons_pres.append(_ep_rec)

    _inv_zones_pres = [
        z_i for z_i in (
            list(scan.get("invalidated_zones") or [])
            + list(dec.get("invalidated_zones") or [])
            + [
                rz_i for rz_i in (scan.get("rejected_zones") or [])
                if isinstance(rz_i, dict)
                and (
                    rz_i.get("is_broken")
                    or str(rz_i.get("status") or "").upper() in ("INVALIDATED", "BROKEN")
                    or str(rz_i.get("freshness") or "").upper() == "INVALIDATED"
                )
            ]
        )
        if isinstance(z_i, dict)
    ]
    _valid_smc_zones_pres = [
        vz_i for vz_i in (list(scan.get("active_zones") or []) + list(scan.get("radar_zones") or []))
        if isinstance(vz_i, dict) or hasattr(vz_i, "to_dict")
    ]
    _invalid_pending_warnings = find_pending_orders_on_invalid_zones(
        _inv_zones_pres,
        mt5_snapshot=_ord_snap_pres,
        symbol=_sym_pres,
        default_source="SMC",
        active_valid_zones=_valid_smc_zones_pres,
    )

    # Zone Tracker
    if tracker is None:
        tracker = dec.get("zone_tracker")
    if tracker is None:
        try:
            tracker = get_zone_tracker(auto_save=False)
            tracker.update(scan.get("active_zones", []), price, htf_bias=h4_bias, m30_status=m30_status, conservative_dec=dec)
        except Exception:
            tracker = None

    # M5 Trigger status summary
    has_htf_conflict_fmt = bool(
        ctx.get("has_conflict")
        or (h4_bias in ("BULLISH", "BEARISH") and h1_struct in ("BULLISH", "BEARISH") and h4_bias != h1_struct)
    )
    has_m30_conflict_fmt = bool(ctx.get("m30_conflict") or m30_status == "CONFLICT")
    if h4_bias in ("BULLISH", "BEARISH") and h1_struct in ("BULLISH", "BEARISH") and h4_bias != h1_struct:
        htf_conflict_phrase = (
            f"Konflik struktur H4 {h4_bias} vs H1/M30 {h1_struct}"
            if m30_dir == h1_struct
            else f"Konflik struktur H4 {h4_bias} vs H1 {h1_struct}"
        )
    elif has_m30_conflict_fmt and h4_bias in ("BULLISH", "BEARISH"):
        htf_conflict_phrase = f"Konflik M30 {m30_dir or 'CONFLICT'} terhadap H4 {h4_bias}"
    else:
        htf_conflict_phrase = ""
    m5_stage_st = str((stages.get("M5") or {}).get("status") or dec.get("m5_eval_status") or "").upper()
    m5_was_eval_fmt = bool(
        dec.get("m5_conflict_evaluated")
        or m5_stage_st in ("CONFIRMED", "UNCONFIRMED", "CONFIRMED_BLOCKED_HTF")
        or any(
            str((stages.get(k) or {}).get("status") or "NOT_CHECKED").upper() not in ("", "NOT_CHECKED")
            for k in ("zone_touch", "liquidity_sweep", "structure_trigger", "displacement")
        )
    )
    no_eligible_setup_fmt = bool(
        status != "VALID"
        and not m5_was_eval_fmt
        and (
            has_htf_conflict_fmt
            or has_m30_conflict_fmt
            or h4_bias == "NETRAL"
            or (not dec.get("zone") and not stages.get("zone_touch"))
        )
    )

    zt = stages.get("zone_touch", {}).get("status", "CONFIRMED" if status == "VALID" else ("NOT_CHECKED" if no_eligible_setup_fmt else "WAIT"))
    sw = stages.get("liquidity_sweep", {}).get("status", "CONFIRMED" if status == "VALID" else ("NOT_CHECKED" if no_eligible_setup_fmt else "WAIT"))
    tr = stages.get("structure_trigger", {}).get("status", "CONFIRMED" if status == "VALID" else ("NOT_CHECKED" if no_eligible_setup_fmt else "WAIT"))
    dp = stages.get("displacement", {}).get("status", "CONFIRMED" if status == "VALID" else ("NOT_CHECKED" if no_eligible_setup_fmt else "WAIT"))
    rr_st = stages.get("risk_reward", {}).get("status", "CONFIRMED" if (status == "VALID" and plan) else ("NOT_CHECKED" if no_eligible_setup_fmt else "WAIT"))

    sw_desc = stages.get("liquidity_sweep", {}).get("msg")
    tr_desc = stages.get("structure_trigger", {}).get("msg")
    dp_desc = stages.get("displacement", {}).get("msg")
    rr_desc = str(
        stages.get("risk_reward", {}).get("msg")
        or stages.get("entry_validation", {}).get("msg")
        or ""
    ).strip()
    m5_tech_confirmed = (status == "VALID") or (tr == "CONFIRMED" and dp == "CONFIRMED") or m5_stage_st in ("CONFIRMED", "CONFIRMED_BLOCKED_HTF")
    if m5_stage_st == "DATA_UNAVAILABLE" or (stages.get("DATA") or {}).get("ok") is False or not scan.get("data_ok", True):
        m5_bias_disp = "DATA_UNAVAILABLE"
    elif status == "VALID":
        m5_bias_disp = "CONFIRMED"
    elif m5_tech_confirmed:
        m5_bias_disp = "CONFIRMED (Blocked: HTF_CONFLICT)" if (has_htf_conflict_fmt or has_m30_conflict_fmt) else "CONFIRMED (Entry WAIT)"
    elif m5_was_eval_fmt:
        m5_bias_disp = "UNCONFIRMED"
    else:
        m5_bias_disp = "NOT_CHECKED"

    if status == "VALID":
        m5_trig_str = f"CONFIRMED ({direction} Trigger Valid)"
    elif m5_tech_confirmed:
        m5_trig_str = f"CONFIRMED ({direction} M5 Valid — Entry WAIT)"
    elif no_eligible_setup_fmt:
        m5_trig_str = (
            f"NOT CHECKED ({htf_conflict_phrase} — belum ada setup eligible, trigger M5 tidak dijalankan)"
            if htf_conflict_phrase
            else "NOT CHECKED (Belum ada setup eligible — trigger M5 tidak dijalankan)"
        )
    elif zt in ("FAIL", "WAIT") and sw in ("NOT_CHECKED", "WAIT"):
        m5_trig_str = "WAIT (Zone Touch belum terpenuhi — trigger zona M5 belum dijalankan)"
    elif sw != "CONFIRMED":
        m5_trig_str = "WAIT (Sweep M5 zona tidak terpenuhi)"
    elif tr != "CONFIRMED":
        m5_trig_str = "WAIT (Menunggu CHoCH/BOS/CISD M5)"
    else:
        m5_trig_str = "WAIT (Menunggu konfirmasi M5)"

    out = []
    out.append("=" * 70)
    out.append("XAUUSD PYTHON SMC SCANNER")
    out.append("=" * 70)
    out.append("")

    h4_diag = ctx.get("h4_diagnostic") or ""
    if h4_bias == "NETRAL" and h4_diag:
        h4_str = f"{h4_bias} ({h4_diag})"
    else:
        h4_str = f"{h4_bias}"

    snapshot_time = scan.get("snapshot_time") or scan.get("timestamp") or datetime.now().strftime("%Y-%m-%d %H:%M:%S WIB")
    bid_val = scan.get("bid")
    ask_val = scan.get("ask")
    mid_val = scan.get("mid")
    spread_val = scan.get("spread")
    tick_age = scan.get("tick_age_seconds")
    if bid_val is not None and ask_val is not None and bid_val > 0:
        if mid_val is None:
            mid_val = round((bid_val + ask_val) / 2.0, _d_pres_px)
        if spread_val is None:
            spread_val = round(ask_val - bid_val, _d_pres_px)
        bid_ask_str = f" | Bid: {bid_val:.{_d_pres_px}f} | Ask: {ask_val:.{_d_pres_px}f}"
    else:
        bid_ask_str = ""
    _sp_src_txt = format_price_distance(spread_val, spec=_u_spec_pres, include_mt5_points=True) if spread_val is not None else "N/A"
    price_src = scan.get("price_source") or (f"MT5 Live Tick (Bid: {bid_val:.{_d_pres_px}f} | Ask: {ask_val:.{_d_pres_px}f} | Mid: {mid_val:.{_d_pres_px}f} | Spread: {_sp_src_txt})" if bid_val else f"Candle Close Snapshot ({price:.{_d_pres_px}f})")

    if str(dec.get("market_status") or scan.get("market_status") or "").upper() == "CLOSED" or dec.get("snapshot_status") == "MARKET_CLOSED":
        price_src = f"HISTORIS — MARKET CLOSED (bukan LIVE; {price_src})"
        bid_ask_str = bid_ask_str.replace("Bid:", "Last Bid:").replace("Ask:", "Last Ask:")
    conflict_tag = " [CONFLICT H4 vs H1]" if ctx.get("has_conflict") else ""
    atr_diag_pres = dec.get("atr_diagnostics") or scan.get("atr_diagnostics")
    raw_radar = scan.get("radar_zones") or scan.get("active_zones", [])
    all_radar = [z.to_dict() if hasattr(z, "to_dict") else dict(z) for z in raw_radar]
    r_prio = dec.get("radar_priority") or scan.get("radar_priority")
    _pres_atr = atr_diag_pres.get("atr_value") if (isinstance(atr_diag_pres, dict) and atr_diag_pres.get("atr_value")) else 2.0
    if not r_prio:
        r_prio = prioritize_radar_zones(
            all_radar, price, htf_bias=h4_bias, h1_structure=h1_struct,
            atr=_pres_atr, m5_stages=stages, bid=scan.get("bid"), ask=scan.get("ask"),
            snapshot_time=scan.get("snapshot_time"), snapshot_id=scan.get("snapshot_id")
        )

    # =========================================================================
    # MODE RINGKAS DEFAULT (`scan`, detailed=False) — SATU BARIS PER INFORMASI
    # =========================================================================
    if not detailed:
        eff_mkt_c = str(dec.get("market_status") or scan.get("market_status") or "OPEN").strip().upper()
        if eff_mkt_c in ("MARKET_CLOSED", "MARKET CLOSED") or str(dec.get("snapshot_status") or "").upper() == "MARKET_CLOSED":
            eff_mkt_c = "CLOSED"
        elif eff_mkt_c in ("", "UNKNOWN"):
            eff_mkt_c = "OPEN" if scan.get("data_ok", True) else "DATA UNAVAILABLE"
        data_ok_c = bool(scan.get("data_ok", True) and (stages.get("DATA") or {}).get("ok") is not False and price > 0)
        if eff_mkt_c == "CLOSED":
            data_ok_c = False

        snap_id_tag = f" [ID:{scan['snapshot_id']}]" if scan.get("snapshot_id") else ""
        px_part = f"{price:.{_d_pres_px}f}" if price > 0 else "N/A"
        if bid_val is not None and ask_val is not None and bid_val > 0 and eff_mkt_c != "CLOSED":
            px_part = f"{px_part} (Bid {bid_val:.{_d_pres_px}f} / Ask {ask_val:.{_d_pres_px}f} | Spr {_sp_src_txt})"
        elif eff_mkt_c == "CLOSED":
            px_part = f"{px_part} (Last Close)"

        dir_analysis = resolve_analysis_direction(
            ctx=ctx, decision=dec, scan=scan,
            market_status=eff_mkt_c, data_ok=data_ok_c,
        )

        def _is_far_zone_disp(z_obj):
            if not isinstance(z_obj, dict):
                return False
            if is_far_reference_zone(z_obj, current_price=price if price > 0 else None, atr=_pres_atr):
                return True
            if str(z_obj.get("radar_eligibility") or "").upper() == "NOT_QUALIFIED" and "DUPLICATE_OVERLAP" not in str(z_obj.get("radar_rejection_reason") or ""):
                return True
            return False

        pz_c = dec.get("primary_zone") or ((r_prio.get("primary_directional_zone") or r_prio.get("primary_zone")) if r_prio else None)
        nz_c = dec.get("nearest_valid_zone") or (r_prio.get("nearest_radar_zone") if r_prio else None)
        raw_main_z = dec.get("zone")
        main_z = raw_main_z if isinstance(raw_main_z, dict) else (pz_c or nz_c)
        far_ref_z = dec.get("far_reference_zone")
        if main_z and _is_far_zone_disp(main_z):
            far_ref_z = far_ref_z or main_z
            main_z = None
        smc_main_z = main_z
        smc_far_ref_z = far_ref_z

        snr_ctx_obj = (dec.get("snr_context") if isinstance(dec.get("snr_context"), dict) else None) or (scan.get("snr_context") if isinstance(scan.get("snr_context"), dict) else None) or {}
        _snr_mb_c = (snr_ctx_obj.get("market_bias") or {}) if isinstance(snr_ctx_obj, dict) else {}
        _snr_h4_c = str(_snr_mb_c.get("h4") or "").upper()
        _snr_h1_c = str(_snr_mb_c.get("h1") or "").upper()
        _snr_m30_c = str(_snr_mb_c.get("m30") or "").upper()
        _eff_h4_snr_c = h4_bias if h4_bias in ("BULLISH", "BEARISH") else _snr_h4_c

        best_tp_c = dec.get("best_trade_plan") if isinstance(dec.get("best_trade_plan"), dict) else None
        if isinstance(best_tp_c, dict):
            from modules.utils import refresh_trade_plan_live_state
            best_tp_c = refresh_trade_plan_live_state(
                best_tp_c,
                live_price=price,
                bid=bid_val,
                ask=ask_val,
                is_closed=(eff_mkt_c == "CLOSED"),
                m5_eval_status=m5_bias_disp,
                mt5_orders_snapshot=_ord_snap_pres,
            )
        _tp_st_c = str((best_tp_c or {}).get("status") or "").strip().upper()
        _tp_src_c = str((best_tp_c or {}).get("source") or "").strip().upper()
        _tp_zid_c = str((best_tp_c or {}).get("zone_id") or "").strip()
        _tp_active_focus = bool(
            best_tp_c
            and _tp_zid_c
            and _tp_st_c not in ("", "NO_ACTIVE_PLAN", "REJECTED", "INVALID", "SL_BREACHED", "EXPIRED", "MISSED")
        )
        selected_snr_main_z = None
        if _tp_active_focus and _tp_src_c == "SNR":
            _snr_search_pool = (
                list(snr_ctx_obj.get("supports") or [])
                + list(snr_ctx_obj.get("resistances") or [])
                + [
                    snr_ctx_obj.get("support"),
                    snr_ctx_obj.get("resistance"),
                    ((snr_ctx_obj.get("entry_validation") or {}).get("target_zone") if isinstance(snr_ctx_obj.get("entry_validation"), dict) else None),
                    snr_ctx_obj.get("breakout_zone"),
                    snr_ctx_obj.get("retest_zone"),
                    snr_ctx_obj.get("rejection_zone"),
                ]
            )
            matched_snr_tp = next(
                (sz for sz in _snr_search_pool if isinstance(sz, dict) and str(sz.get("zone_id") or "").strip() == _tp_zid_c),
                None,
            )
            if isinstance(matched_snr_tp, dict):
                selected_snr_main_z = dict(matched_snr_tp)
            else:
                selected_snr_main_z = {
                    "zone_id": _tp_zid_c,
                    "timeframe": best_tp_c.get("timeframe") or "H1",
                    "zone_type": best_tp_c.get("zone_type") or ("RESISTANCE" if best_tp_c.get("direction") == "SELL" else "SUPPORT"),
                    "low": float(best_tp_c.get("zone_low") or 0.0),
                    "high": float(best_tp_c.get("zone_high") or 0.0),
                    "grade": best_tp_c.get("grade") or "-",
                    "score": best_tp_c.get("score"),
                    "structure_invalidation": float(best_tp_c.get("struct_inv") or 0.0),
                    "setup_sl": float(best_tp_c.get("setup_sl") or 0.0),
                    "setup_version": best_tp_c.get("setup_version"),
                    "zone_relevance": best_tp_c.get("zone_status") or "ACTIVE",
                }
            selected_snr_main_z["source"] = "SNR"
            selected_snr_main_z.setdefault("price_low", float(selected_snr_main_z.get("low") or 0.0))
            selected_snr_main_z.setdefault("price_high", float(selected_snr_main_z.get("high") or 0.0))
        elif _tp_active_focus and _tp_src_c == "SMC":
            if not isinstance(main_z, dict) or str(main_z.get("zone_id") or "").strip() != _tp_zid_c:
                matched_smc_tp = next(
                    (
                        cz for cz in ([raw_main_z, pz_c, nz_c] + list(all_radar))
                        if isinstance(cz, dict) and str(cz.get("zone_id") or "").strip() == _tp_zid_c and not _is_far_zone_disp(cz)
                    ),
                    None,
                )
                if isinstance(matched_smc_tp, dict):
                    main_z = matched_smc_tp
                    smc_main_z = matched_smc_tp

        best_p_c = (early_setups or {}).get("best_setup")
        pending_ready_c = bool(
            eff_mkt_c != "CLOSED"
            and best_p_c
            and best_p_c.get("valid")
            and str(best_p_c.get("status") or "READY").upper() == "READY"
            and not best_p_c.get("is_expired")
            and h4_bias in ("BULLISH", "BEARISH")
            and not ctx.get("has_conflict")
            and m30_status not in ("CONFLICT", "INVALID", "WEAK")
        )
        confirmed_ready_c = bool(eff_mkt_c != "CLOSED" and status == "VALID" and plan and direction in ("BUY", "SELL"))

        def _short_blocker() -> str:
            raw_r = ""
            if dec.get("reasons"):
                raw_r = sanitize_wait_reason(str(dec["reasons"][0]).replace("[SCANNER] Rejection Reason: ", ""))
            elif not scan.get("data_ok", True):
                raw_r = str(scan.get("empty_reason") or "DATA UNAVAILABLE")
            elif rr_st != "CONFIRMED" and rr_desc:
                raw_r = rr_desc
            return resolve_entry_blocker_code(
                raw_r,
                market_status=eff_mkt_c,
                data_ok=bool(scan.get("data_ok", True)),
                htf_conflict=has_htf_conflict_fmt,
                m30_conflict=has_m30_conflict_fmt,
                h4_bias=h4_bias,
                has_active_zone=bool(main_z or selected_snr_main_z),
                zone_touch=zt,
                rr_failed=(rr_st == "FAIL" or ("rr" in (rr_desc or "").lower() and "min" in (rr_desc or "").lower())),
                is_missed=bool(dec.get("is_missed") or (dec.get("m5") or {}).get("is_missed")),
            )

        if confirmed_ready_c:
            tp2_v = plan.get("tp2")
            tp_part = f"TP1 {plan.get('tp1', 0.0):.2f}" + (f" / TP2 {tp2_v:.2f}" if tp2_v and tp2_v != plan.get("tp1") else "")
            entry_status_line = (
                f"{direction} VALID (CONFIRMED) | E {plan.get('entry_acuan', 0.0):.2f} | "
                f"SL {plan.get('sl', 0.0):.2f} | {tp_part} | RR 1:{plan.get('rr', 0.0):.1f}"
            )
        else:
            blk_c = _short_blocker()
            if blk_c == "HTF_CONFLICT" and m5_bias_disp != "NOT_CHECKED":
                m5_tag_e = "CONFIRMED" if m5_tech_confirmed else m5_bias_disp
                entry_status_line = f"NO VALID ENTRY ({blk_c} | M5 {m5_tag_e})"
            else:
                entry_status_line = f"NO VALID ENTRY ({blk_c})"

        def _fmt_zone_1line(z_obj: Optional[Dict[str, Any]], is_far: bool = False) -> str:
            if not isinstance(z_obj, dict) or not z_obj.get("price_low"):
                return "NONE"
            zt_f = str(z_obj.get("timeframe") or "H1")
            zt_t = str(z_obj.get("zone_type") or "ZONE").upper()
            z_lo = float(z_obj.get("price_low", 0.0))
            z_hi = float(z_obj.get("price_high", 0.0))
            z_grd = str(z_obj.get("grade") or "-")
            z_sc = f"/{float(z_obj.get('score', 0.0)):.1f}" if z_obj.get("scoring_status") != "DATA_INCOMPLETE" else ""
            z_fr = str(z_obj.get("freshness") or "FRESH").upper()
            z_struct_inv = resolve_smc_structure_invalidation_price(z_obj) or (z_lo if zt_t == "DEMAND" else z_hi)
            from modules.utils import resolve_stored_setup_sl
            z_setup_sl = 0.0 if is_far else (resolve_stored_setup_sl(
                "SMC",
                str(z_obj.get("zone_id") or ""),
                trade_plans=dec.get("trade_plans") if isinstance(dec.get("trade_plans"), list) else None,
                best_trade_plan=dec.get("best_trade_plan") if isinstance(dec.get("best_trade_plan"), dict) else None,
                timeframe=zt_f,
                setup_version=z_obj.get("setup_version") if isinstance(z_obj.get("setup_version"), int) else None,
            ) or resolve_smc_setup_invalidation_price(z_obj))
            d_txt = format_zone_relative_distance(z_lo, z_hi, price, spec=_u_spec_pres, is_far=is_far)
            inv_segs = []
            if z_struct_inv > 0:
                inv_segs.append(f"StructInv {z_struct_inv:.2f}")
            if z_setup_sl > 0 and abs(z_setup_sl - z_struct_inv) >= 0.01:
                inv_segs.append(f"SetupSL {z_setup_sl:.2f}")
            inv_txt = (" | " + " | ".join(inv_segs)) if inv_segs else ""
            rel_t = str(z_obj.get("zone_relevance") or z_obj.get("status") or "RADAR").upper()
            ct_t = " COUNTER" if z_obj.get("is_counter") else ""
            if is_far:
                return f"[SMC] {zt_f} {zt_t} {z_lo:.2f}-{z_hi:.2f} | {d_txt} | Grade {z_grd}{z_sc} ({z_fr}){inv_txt}"
            return f"[SMC] {zt_f} {zt_t} {z_lo:.2f}-{z_hi:.2f} | {d_txt} | Grade {z_grd}{z_sc} ({z_fr}){inv_txt} | {rel_t}{ct_t}"

        from modules.snr.formatter import build_snr_zone_reason

        def _fmt_snr_zone_1line(sz_obj: Optional[Dict[str, Any]]) -> str:
            if not isinstance(sz_obj, dict):
                return "NONE"
            s_lo = float(sz_obj.get("low", sz_obj.get("price_low", 0.0)) or 0.0)
            s_hi = float(sz_obj.get("high", sz_obj.get("price_high", s_lo)) or s_lo)
            if s_lo <= 0:
                return "NONE"
            s_tf = str(sz_obj.get("timeframe") or "H1").strip().upper()
            s_zt = str(sz_obj.get("zone_type") or "ZONE").strip().upper()
            s_grd = str(sz_obj.get("grade") or "-")
            s_sc = f"/{float(sz_obj['score']):.1f}" if sz_obj.get("score") is not None else ""
            s_sinv = float(sz_obj.get("structure_invalidation") or sz_obj.get("invalidation_price") or (s_lo if s_zt == "SUPPORT" else s_hi))
            from modules.utils import resolve_stored_setup_sl
            s_sl = resolve_stored_setup_sl(
                "SNR",
                str(sz_obj.get("zone_id") or ""),
                trade_plans=dec.get("trade_plans") if isinstance(dec.get("trade_plans"), list) else None,
                best_trade_plan=best_tp_c,
                timeframe=s_tf,
                setup_version=sz_obj.get("setup_version") if isinstance(sz_obj.get("setup_version"), int) else None,
            ) or float(sz_obj.get("setup_sl") or 0.0)
            d_txt_s = format_zone_relative_distance(s_lo, s_hi, price, spec=_u_spec_pres)
            inv_s_segs = []
            if s_sinv > 0:
                inv_s_segs.append(f"StructInv {s_sinv:.2f}")
            if s_sl > 0 and abs(s_sl - s_sinv) >= 0.01:
                inv_s_segs.append(f"SetupSL {s_sl:.2f}")
            inv_s_txt = (" | " + " | ".join(inv_s_segs)) if inv_s_segs else ""
            rel_s = str(sz_obj.get("zone_relevance") or sz_obj.get("lifecycle_status") or sz_obj.get("status") or "ACTIVE").upper()
            return f"[SNR] {s_tf} {s_zt} {s_lo:.2f}-{s_hi:.2f} | {d_txt_s} | Grade {s_grd}{s_sc}{inv_s_txt} | {rel_s}"

        smc_far_ref_line = ""
        if isinstance(selected_snr_main_z, dict) and float(selected_snr_main_z.get("price_low") or 0.0) > 0:
            active_radar_line = _fmt_snr_zone_1line(selected_snr_main_z)
            shown_main_z = selected_snr_main_z
            main_why_c = (
                build_snr_zone_reason(selected_snr_main_z, h4_bias=_eff_h4_snr_c, h1_bias=_snr_h1_c)
                or str((best_tp_c or {}).get("why") or "")
            )
            if isinstance(smc_far_ref_z, dict) and smc_far_ref_z.get("price_low"):
                smc_far_ref_line = _fmt_zone_1line(smc_far_ref_z, is_far=True)
        else:
            active_radar_line = _fmt_zone_1line(main_z)
            shown_main_z = main_z if (isinstance(main_z, dict) and main_z.get("price_low")) else None
            if active_radar_line == "NONE" and isinstance(far_ref_z, dict):
                active_radar_line = f"NONE (Hanya Ref Jauh: {_fmt_zone_1line(far_ref_z, is_far=True)})"
                if far_ref_z.get("price_low"):
                    shown_main_z = far_ref_z
            main_why_c = build_smc_zone_reason(shown_main_z, h4_bias=h4_bias, h1_bias=h1_struct) if shown_main_z else ""

        next_z = smc_main_z if isinstance(selected_snr_main_z, dict) else dec.get("next_radar_zone")
        if not next_z and nz_c and (not shown_main_z or nz_c.get("zone_id") != shown_main_z.get("zone_id")):
            next_z = nz_c

        # Latest INVALID (prioritaskan invalidasi terbaru, relevan terhadap harga & direction, serta kualitas memadai)
        inv_pool_c = list(dec.get("invalidated_zones") or []) + list((r_prio or {}).get("invalidated_zones") or []) + list(scan.get("invalidated_zones") or [])
        if tracker is not None and isinstance(getattr(tracker, "invalidated_zones", None), dict):
            inv_pool_c.extend(list(tracker.invalidated_zones.values()))
        iz0 = consume_one_shot_invalid_zone(
            inv_pool_c,
            current_price=price if price > 0 else None,
            atr=_pres_atr,
            direction=dir_analysis,
            cycle_key=f"scan:{snapshot_time}",
        )
        if isinstance(iz0, dict):
            iz_t = str(iz0.get("invalidation_type") or iz0.get("invalid_label") or INVALIDATION_ZONE_STRUCTURAL).upper()
            invalid_line = format_invalid_zone_compact(iz0, current_price=price if price > 0 else None, unit_spec=_u_spec_pres, status_override=iz_t, include_id=False)
        else:
            invalid_line = "NONE"
        inv_why_c = format_smc_invalid_why(iz0, h4_bias=h4_bias, h1_bias=h1_struct) if isinstance(iz0, dict) else ""

        # SNR 1-line summary
        confl_obj = dec.get("smc_snr_confluence") or scan.get("smc_snr_confluence")
        _confl_zone = smc_main_z or pz_c
        if snr_ctx_obj and _confl_zone and (
            confl_obj is None
            or str(((confl_obj.get("smc_zone") or {}) if isinstance(confl_obj, dict) else {}).get("zone_id") or "")
            != str(_confl_zone.get("zone_id") or "")
        ):
            confl_obj = evaluate_smc_snr_confluence(_confl_zone, snr_ctx_obj, price, htf_bias=h4_bias)
        sup_c = snr_ctx_obj.get("support") or ((snr_ctx_obj.get("supports") or [None])[0]) if isinstance(snr_ctx_obj, dict) else None
        res_c = snr_ctx_obj.get("resistance") or ((snr_ctx_obj.get("resistances") or [None])[0]) if isinstance(snr_ctx_obj, dict) else None
        s1_why_c = ""
        r1_why_c = ""
        snr_parts = []
        if isinstance(sup_c, dict) and sup_c.get("low"):
            s1_same_c = zones_represent_same_level(sup_c, shown_main_z)
            s1_tf_c = str(sup_c.get("timeframe") or "").strip().upper()
            snr_parts.append(f"S1: {(s1_tf_c + ' ') if s1_tf_c else ''}{float(sup_c['low']):.2f}-{float(sup_c.get('high', sup_c['low'])):.2f} (G:{sup_c.get('grade', '-')}{', = RADAR' if s1_same_c else ''})")
            if not s1_same_c:
                s1_why_c = build_snr_zone_reason(sup_c, h4_bias=_eff_h4_snr_c, h1_bias=_snr_h1_c)
        else:
            snr_parts.append("S1: NONE")
        if isinstance(res_c, dict) and res_c.get("low"):
            r1_same_c = zones_represent_same_level(res_c, shown_main_z)
            r1_tf_c = str(res_c.get("timeframe") or "").strip().upper()
            snr_parts.append(f"R1: {(r1_tf_c + ' ') if r1_tf_c else ''}{float(res_c['low']):.2f}-{float(res_c.get('high', res_c['low'])):.2f} (G:{res_c.get('grade', '-')}{', = RADAR' if r1_same_c else ''})")
            if not r1_same_c:
                r1_why_c = build_snr_zone_reason(res_c, h4_bias=_eff_h4_snr_c, h1_bias=_snr_h1_c)
        else:
            snr_parts.append("R1: NONE")
        if isinstance(confl_obj, dict) and confl_obj.get("available"):
            if confl_obj.get("is_aligned") and float(confl_obj.get("overlap_points") or 0.0) > 0:
                snr_parts.append(f"CONFLUENCE ({format_price_distance(float(confl_obj['overlap_points']), spec=_u_spec_pres)})")
            elif confl_obj.get("opposing_conflict"):
                snr_parts.append("SNR CONFLICT")
        snr_line = " | ".join(snr_parts) if (isinstance(sup_c, dict) or isinstance(res_c, dict)) else str(snr_ctx_obj.get("summary") or "UNAVAILABLE")

        # RISK 1-line summary (MT5 read-only + sync timestamp)
        mt5_risk_c = summarize_mt5_risk_warnings(
            _ord_snap_pres,
            invalid_zone_warnings=_invalid_pending_warnings,
            matched_recons=_matched_recons_pres,
            invalid_zone_scope="SMC",
        )
        pos_c = int((_ord_snap_pres or {}).get("symbol_positions_count", 0) or 0)
        ord_c = int((_ord_snap_pres or {}).get("symbol_orders_count", 0) or 0)
        nosl_c = len((_ord_snap_pres or {}).get("positions_without_sl") or []) + len((_ord_snap_pres or {}).get("orders_without_sl") or [])
        sync_c = format_mt5_sync_short(_ord_snap_pres, fallback_ts=snapshot_time)
        risk_head = str(mt5_risk_c.get("headline") or "").strip()
        if (_ord_snap_pres or {}).get("query_ok") is False:
            risk_line = f"{risk_head or 'MT5 DISCONNECTED'} | Sync:{sync_c} | Read-Only"
        elif risk_head:
            risk_line = f"{risk_head} | Pos:{pos_c} Pend:{ord_c} NoSL:{nosl_c} | Sync:{sync_c}"
        else:
            risk_line = f"AMAN | Pos:{pos_c} Pend:{ord_c} NoSL:{nosl_c} | Sync:{sync_c}"

        term_w_c = get_terminal_width(default=100, min_width=64, max_width=120)
        _smc_m30_c = (
            f"{m30_dir} ({m30_status})"
            if (m30_dir and m30_status and str(m30_dir).upper() != str(m30_status).upper())
            else str(m30_dir or m30_status)
        )
        _m30_snr_seg_c = (
            f" | M30(SNR) {_snr_m30_c}"
            if (_snr_m30_c and _snr_m30_c not in ("N/A", "UNKNOWN", "UNCONFIRMED", "NO_DATA", "DATA UNAVAILABLE"))
            else ""
        )
        out.extend(wrap_dashboard_field("MARKET       : ", f"{eff_mkt_c} | {px_part} | {snapshot_time}{snap_id_tag}", term_w_c, max_lines=None))
        out.extend(wrap_dashboard_field("BIAS         : ", f"H4 {h4_bias} | H1(SMC Struct) {h1_struct}{conflict_tag} | M30(SMC) {_smc_m30_c}{_m30_snr_seg_c} | M5 {m5_bias_disp}", term_w_c, max_lines=None))
        out.extend(wrap_dashboard_field("MARKET BIAS  : ", str(dir_analysis), term_w_c, max_lines=None))
        out.extend(wrap_dashboard_field("ENTRY STATUS : ", entry_status_line, term_w_c, max_lines=None))

        bp_same_zone = bool(
            pending_ready_c and not confirmed_ready_c and best_tp_c
            and str(best_tp_c.get("zone_id") or "") == str(best_p_c.get("zone_id") or "")
            and str(best_tp_c.get("status") or "").upper() == "READY"
        )
        _act_zid_c = shown_main_z.get("zone_id") if isinstance(shown_main_z, dict) else None
        if best_tp_c:
            _brk_tag_c = (best_p_c.get("compact_status") or "NOT PLACED") if bp_same_zone else None
            out.extend(format_trade_plan_lines(
                best_tp_c,
                prefix="TRADE PLAN   : ",
                why_prefix="  WHY PLAN   : ",
                width=term_w_c,
                include_source=True,
                broker_tag=_brk_tag_c,
                skip_why_for_zone_id=_act_zid_c,
                include_id=False,
                active_zone_id=_act_zid_c,
            ))
        if pending_ready_c and not confirmed_ready_c and not bp_same_zone:
            bp_c_st = best_p_c.get("compact_status") or "READY | NOT PLACED"
            out.extend(wrap_dashboard_field(
                "PENDING PLAN : ",
                f"{best_p_c['order_type']} {best_p_c['entry_price']:.2f} | "
                f"SetupSL {best_p_c['sl']:.2f} | TP1 {best_p_c['tp1']:.2f} | RR 1:{best_p_c['rr']:.1f} | {bp_c_st}",
                term_w_c,
                max_lines=None,
            ))
        out.extend(wrap_dashboard_field("ACTIVE/RADAR : ", active_radar_line, term_w_c, max_lines=None))
        if main_why_c:
            out.extend(wrap_dashboard_field("  WHY        : ", main_why_c, term_w_c, max_lines=None))
        if isinstance(next_z, dict) and next_z.get("zone_id") and (not shown_main_z or next_z.get("zone_id") != shown_main_z.get("zone_id")):
            out.extend(wrap_dashboard_field("NEXT RADAR   : ", _fmt_zone_1line(next_z), term_w_c, max_lines=None))
            next_why_c = build_smc_zone_reason(next_z, h4_bias=h4_bias, h1_bias=h1_struct)
            if next_why_c:
                out.extend(wrap_dashboard_field("  WHY        : ", next_why_c, term_w_c, max_lines=None))
        out.extend(wrap_dashboard_field("SNR LEVEL    : ", snr_line, term_w_c, max_lines=None))
        if s1_why_c:
            out.extend(wrap_dashboard_field("  WHY S1     : ", s1_why_c, term_w_c, max_lines=None))
        if r1_why_c:
            out.extend(wrap_dashboard_field("  WHY R1     : ", r1_why_c, term_w_c, max_lines=None))
        if smc_far_ref_line:
            out.extend(wrap_dashboard_field("SMC FAR REF  : ", smc_far_ref_line, term_w_c, max_lines=None))
        if invalid_line != "NONE":
            out.extend(wrap_dashboard_field("INVALID      : ", invalid_line, term_w_c, max_lines=None))
        out.extend(wrap_dashboard_field("RISK         : ", risk_line, term_w_c, max_lines=None))
        out.append("=" * min(70, term_w_c))
        return "\n".join(out)

    # =========================================================================
    # MODE DETAIL AUDIT (`scan detail`, detailed=True) — LENGKAP SEPERTI SEBELUMNYA
    # =========================================================================
    # A. KONDISI PASAR (MARKET OVERVIEW & BIAS)
    out.append("A. KONDISI PASAR — 2. MARKET BIAS (MARKET OVERVIEW)")
    out.append(f"- Current Price    : {price:.2f} (Snapshot: {snapshot_time}{bid_ask_str})" if price > 0 else "- Current Price    : N/A")
    out.append(f"- Sumber Harga     : {price_src}")
    m30_msg_d = str(m30_val.get("msg") or "").strip()
    m30_cond_d = str(m30_val.get("condition") or ctx.get("m30_setup_condition") or "").strip()
    if m30_status in ("CONFLICT", "WEAK", "INVALID") and m30_msg_d:
        m30_disp_d = f"{m30_status} ({m30_msg_d})"
    elif m30_status == "CONFIRMED" and m30_cond_d and m30_cond_d != m30_status:
        m30_disp_d = f"{m30_status} (Condition: {m30_cond_d})"
    else:
        m30_disp_d = f"{m30_status}"
    m30_disp_bias_d = f"{m30_dir} ({m30_status})" if (m30_dir and m30_dir in ("BULLISH", "BEARISH") and m30_status != m30_dir) else (m30_dir or m30_status)
    _snr_mb_det = ((dec.get("snr_context") or scan.get("snr_context") or {}).get("market_bias") or {}) if isinstance(dec.get("snr_context") or scan.get("snr_context"), dict) else {}
    _snr_m30_det = str(_snr_mb_det.get("m30") or "").upper()
    _m30_snr_det_seg = f" | M30(SNR): {_snr_m30_det}" if (_snr_m30_det and _snr_m30_det not in ("N/A", "UNKNOWN", "UNCONFIRMED", "NO_DATA", "DATA UNAVAILABLE")) else ""
    out.append(f"- Multi-TF Bias    : H4: {h4_bias} | H1(SMC): {h1_struct}{conflict_tag} | M30(SMC): {m30_disp_bias_d}{_m30_snr_det_seg} | M5: {m5_bias_disp}")
    out.append(f"- H4 Bias          : {h4_str}")
    out.append(f"- H1 Structure     : {h1_struct}")
    out.append(f"- M30 Momentum     : {m30_disp_d}")
    out.append(f"- M5 Trigger       : {m5_trig_str}")
    if atr_diag_pres:
        atr_v_str = format_price_distance(atr_diag_pres['atr_value'], spec=_u_spec_pres, include_mt5_points=True) if isinstance(atr_diag_pres.get("atr_value"), (int, float)) else "DATA_INCOMPLETE"
        out.append(
            f"- ATR Diagnostics  : Source: {atr_diag_pres.get('atr_source_timeframe', 'H1')} | "
            f"Period: {atr_diag_pres.get('atr_period', 14)} ({atr_diag_pres.get('atr_candle_mode', 'CLOSED_CANDLE')}) | "
            f"Value: {atr_v_str} | Candle: {atr_diag_pres.get('atr_candle_timestamp') or 'N/A'} | "
            f"Status: {atr_diag_pres.get('atr_calculation_status', 'N/A')} | Fallback: {atr_diag_pres.get('atr_fallback_status', 'N/A')}"
        )
    out.append("")

    # B. RADAR ZONE (ZONA PEMANTAUAN & ZONE ANALYSIS)
    out.append("B. RADAR ZONE — 3. RADAR ZONE (ZONA PEMANTAUAN & ZONE ANALYSIS)")
    b_hr = r_prio.get("best_high_risk") if r_prio else None
    b_mr = r_prio.get("best_medium_risk") if r_prio else None
    b_lr = r_prio.get("best_low_risk") if r_prio else None
    b_pend = r_prio.get("best_pending_candidate") if r_prio else None
    b_conf = r_prio.get("best_confirmed_candidate") if r_prio else None
    pz = None
    nz = None
    bz = None
    sz = None
    cz = None

    if not all_radar:
        out.append("  NO RADAR ZONE DETECTED.")
    else:
        pz = r_prio.get("primary_directional_zone") or r_prio.get("primary_zone")
        nz = r_prio.get("nearest_radar_zone")
        bz = r_prio.get("best_entry_monitoring_zone")
        sz = r_prio.get("secondary_zone")
        cz = r_prio.get("countertrend_zone")

        # Zone Evaluation Summary
        eval_sum = scan.get("zone_evaluation_summary") or {}
        if not eval_sum and all_radar:
            tot_sup = sum(1 for z in all_radar if z.get("zone_type") == "SUPPLY")
            tot_dem = sum(1 for z in all_radar if z.get("zone_type") == "DEMAND")
            tot_fresh = sum(1 for z in all_radar if z.get("freshness") == "FRESH")
            tot_tested = sum(1 for z in all_radar if str(z.get("freshness", "")).startswith("TESTED"))
            eval_sum = {
                "total_candidates": len(all_radar),
                "total_raw": len(all_radar),
                "total_supply": tot_sup,
                "total_demand": tot_dem,
                "total_fresh": tot_fresh,
                "total_tested": tot_tested,
                "total_mitigated": 0,
                "total_invalidated": 0,
                "initial_radar_count": len(all_radar),
                "radar_qualified": len(all_radar),
                "entry_validated_count": 1 if (status == "VALID") else 0,
                "entry_qualified": 1 if (status == "VALID") else 0,
            }

        if eval_sum:
            tot_cand = eval_sum.get('total_candidates', len(all_radar))
            tot_raw = eval_sum.get('raw_candidates', eval_sum.get('total_raw', tot_cand))
            tot_struct = eval_sum.get('structurally_valid', eval_sum.get('count_struct_valid', tot_cand))
            tot_struct_rej = eval_sum.get('structurally_rejected', eval_sum.get('count_structurally_rejected', 0))
            tot_sup = eval_sum.get('total_supply', eval_sum.get('count_supply', 0))
            tot_dem = eval_sum.get('total_demand', eval_sum.get('count_demand', 0))
            tot_fresh = eval_sum.get('total_fresh', eval_sum.get('count_fresh', 0))
            tot_tested = eval_sum.get('total_tested', eval_sum.get('count_tested', 0))
            tot_mit = eval_sum.get('total_mitigated', eval_sum.get('count_mitigated', 0))
            tot_inval = eval_sum.get('total_invalidated', eval_sum.get('count_invalidated', 0))
            tot_exp = eval_sum.get('total_expired', eval_sum.get('count_expired', 0))
            tot_radar = eval_sum.get('radar_qualified', eval_sum.get('initial_radar_count', len(all_radar)))
            tot_pend = eval_sum.get('pending_qualified', 1 if b_pend else 0)
            tot_conf = eval_sum.get('confirmed_entry_qualified', 1 if (status == "VALID" and b_conf) else 0)

            out.append("  [ZONE EVALUATION SUMMARY]:")
            out.append(f"  • Total Kandidat Zona      : {tot_cand}")
            out.append(f"  • Raw Candidates           : {tot_raw}")
            out.append(f"  • Structurally Valid       : {tot_struct}")
            out.append("  • Definisi Struct. Valid   : Fresh Valid + Tested Valid (Mitigated ditampilkan terpisah)")
            out.append(f"  • Radar Qualified          : {tot_radar} (Zona Aktif Pemantauan)")
            out.append(f"  • Pending Qualified (Limit): {tot_pend} (Seleksi radar awal; bukan status rencana pending final)")
            pend_ready_det = bool(
                ((early_setups or {}).get("best_setup") or {}).get("valid")
                and h4_bias in ("BULLISH", "BEARISH")
                and not ctx.get("has_conflict")
                and m30_status not in ("CONFLICT", "INVALID", "WEAK")
            )
            out.append(f"  • Pending Plan READY       : {1 if pend_ready_det else 0} (Planner agresif terpisah; BELUM order MT5)")
            out.append(f"  • Confirmed Entry Qualified: {tot_conf}")
            out.append(f"  • Jumlah Supply            : {tot_sup}")
            out.append(f"  • Jumlah Demand            : {tot_dem}")
            out.append(f"  • Jumlah Zona Fresh        : {tot_fresh}")
            out.append(f"  • Detail Granular          : Fresh: {tot_fresh} | Tested: {tot_tested} | Mitigated: {tot_mit} | Invalidated: {tot_inval}")
            out.append(f"  • Rekonsiliasi Status      : Fresh Valid ({tot_fresh}) + Tested Valid ({tot_tested}) + Mitigated ({tot_mit}) + Invalidated ({tot_inval}) + Structurally Rejected ({tot_struct_rej}) + Expired ({tot_exp}) = Total Raw ({tot_raw})")
            out.append("")

        # ZONE COUNT RECONCILIATION (Requirement 1 & 2)
        recon = (r_prio.get("zone_count_reconciliation") if r_prio else None) or dec.get("zone_count_reconciliation") or {}
        if recon:
            out.append("  [ZONE COUNT RECONCILIATION — AUDIT DISTRIBUSI KANDIDAT]:")
            out.append(f"  • Structurally Valid                       : {tot_struct} (Lolos validasi struktur awal)")
            out.append(f"  • Radar Qualified (Zona Aktif)             : {recon.get('radar_qualified_count', tot_radar)} (Memenuhi syarat radar aktif)")
            out.append(f"  • Total Ditampilkan di Multi-Tier          : {recon.get('multi_tier_displayed_count', 0)} (High: {recon.get('high_risk_count', 0)}, Medium: {recon.get('medium_risk_count', 0)}, Low: {recon.get('low_risk_count', 0)})")
            out.append(f"  • Di Luar Kategori High/Medium/Low         : {recon.get('unclassified_tier_count', 0)}")
            for uc_r in recon.get("unclassified_tier_reasons", [])[:3]:
                out.append(f"    - {uc_r}")
            out.append(f"  • Cadangan Tier di Luar Batas Kuota Top 3  : {recon.get('backup_tier_count', 0)}")
            for bk_r in recon.get("backup_tier_reasons", [])[:3]:
                out.append(f"    - {bk_r}")
            fz_reasons = recon.get("far_zones_reasons", [])
            fz_count = recon.get("far_zones_count", len(fz_reasons))
            fz_shown = len(fz_reasons)
            fz_hidden = max(0, fz_count - fz_shown)
            fz_meta = f"Detail Displayed: {fz_shown}/{fz_count}" + (f" | Hidden Details: {fz_hidden}" if fz_hidden > 0 else "")
            out.append(f"  • Ditolak Evaluasi Lanjutan (Far Zone)     : {fz_count} ({fz_meta})")
            for fz_r in fz_reasons:
                out.append(f"    - {fz_r}")
            out.append(f"  • Tidak Ditampilkan Batas Tampilan Ranking : {recon.get('hidden_active_ranking_count', 0)}")
            for hr_r in recon.get("hidden_active_ranking_reasons", [])[:3]:
                out.append(f"    - {hr_r}")
            out.append("")

        # MULTI-TIER RADAR ZONE (Three-Tier Risk Classification Engine)
        out.append("  [MULTI-TIER RADAR ZONE — RISK-BASED SELEKSI RESMI PYTHON SCANNER]:")
        hr_zones = r_prio.get("high_risk_zones") or r_prio.get("multi_tier_zones", {}).get("high_risk", [])
        mr_zones = r_prio.get("medium_risk_zones") or r_prio.get("multi_tier_zones", {}).get("medium_risk", [])
        lr_zones = r_prio.get("low_risk_zones") or r_prio.get("multi_tier_zones", {}).get("low_risk", [])

        out.append("  • A. HIGH RISK ZONES (Aggressive — Peluang Dekat / LTF-MTF):")
        if hr_zones:
            for idx, hz in enumerate(hr_zones[:MULTI_TIER_MAX_ZONES], 1):
                out.append(f"    [{idx}] {hz.get('timeframe')} {hz.get('zone_type')} @ {hz.get('price_low', 0.0):.2f}-{hz.get('price_high', 0.0):.2f}:")
                out.extend(_format_multi_tier_zone_item(hz, price))
        else:
            out.append("    NONE (Tidak ada zona High Risk yang memenuhi kriteria)")

        out.append("  • B. MEDIUM RISK ZONES (Balanced — Kualitas Struktur & Jarak Proporsional):")
        if mr_zones:
            for idx, mz in enumerate(mr_zones[:MULTI_TIER_MAX_ZONES], 1):
                out.append(f"    [{idx}] {mz.get('timeframe')} {mz.get('zone_type')} @ {mz.get('price_low', 0.0):.2f}-{mz.get('price_high', 0.0):.2f}:")
                out.extend(_format_multi_tier_zone_item(mz, price))
        else:
            out.append("    NONE (Tidak ada zona Medium Risk yang memenuhi kriteria)")

        out.append("  • C. LOW RISK ZONES (Conservative — Kualitas Tertinggi & Selaras HTF):")
        if lr_zones:
            for idx, lz in enumerate(lr_zones[:MULTI_TIER_MAX_ZONES], 1):
                out.append(f"    [{idx}] {lz.get('timeframe')} {lz.get('zone_type')} @ {lz.get('price_low', 0.0):.2f}-{lz.get('price_high', 0.0):.2f}:")
                out.extend(_format_multi_tier_zone_item(lz, price))
        else:
            out.append("    NONE (Tidak ada zona Low Risk yang memenuhi kriteria)")
        out.append("")

        out.append("  [4 KATEGORI EVALUASI ZONA — SELEKSI RESMI PYTHON SCANNER]:")
        if pz:
            pz_lbl = f"[{pz.get('status', 'RADAR')}]" + (f" [{pz.get('approach')}]" if pz.get('approach') in ("FAR", "INSIDE_ZONE") else "")
            pz_dir = pz.get("direction_status") or ("ABOVE_PRICE" if pz["price_low"] > price else ("BELOW_PRICE" if pz["price_high"] < price else "INSIDE_ZONE"))
            fresh_str = f"Freshness: {pz.get('freshness', 'FRESH')} | " if pz.get('freshness') else ""
            skor_str = f"Skor: {pz.get('score', 0.0):.1f}" if pz.get("scoring_status") != "DATA_INCOMPLETE" else "Skor: DATA INCOMPLETE"
            out.append(f"  • A. PRIMARY DIRECTIONAL ZONE (Primary Radar Zone) : {pz['timeframe']} {pz['zone_type']} {pz['price_low']:.2f} - {pz['price_high']:.2f} [Grade: {pz['grade']} | {skor_str}] | {fresh_str}Jarak: {format_price_distance(pz['distance'], spec=_u_spec_pres, include_mt5_points=True)} ({pz_dir}) | Status: {pz_lbl}")
            pz_why_d = build_smc_zone_reason(pz, h4_bias=h4_bias)
            if pz_why_d:
                out.append(f"    - WHY                                            : {pz_why_d}")
            out.append(f"    - Alasan Directional                             : {r_prio.get('primary_reason', '-')}")
        else:
            out.append(f"  • A. PRIMARY DIRECTIONAL ZONE (Primary Radar Zone) : - (Tidak ada zona yang selaras dengan bias {h4_bias})")

        if nz:
            nz_lbl = f"[{nz.get('status', 'RADAR')}]" + (" [COUNTERTREND]" if nz.get("is_counter") else "") + (f" [{nz.get('approach')}]" if nz.get('approach') in ("FAR", "INSIDE_ZONE") else "")
            nz_dir = nz.get("direction_status") or ("ABOVE_PRICE" if nz["price_low"] > price else ("BELOW_PRICE" if nz["price_high"] < price else "INSIDE_ZONE"))
            nz_role = " (Countertrend/Support - pantau reaksi, bukan sinyal)" if nz.get("is_counter") else ""
            fresh_nz = f"Freshness: {nz.get('freshness', 'FRESH')} | " if nz.get('freshness') else ""
            skor_nz = f"Skor: {nz.get('score', 0.0):.1f}" if nz.get("scoring_status") != "DATA_INCOMPLETE" else "Skor: DATA INCOMPLETE"
            out.append(f"  • B. NEAREST RADAR ZONE                             : {nz['timeframe']} {nz['zone_type']} {nz['price_low']:.2f} - {nz['price_high']:.2f} [Grade: {nz['grade']} | {skor_nz}] | {fresh_nz}Jarak: {format_price_distance(nz['distance'], spec=_u_spec_pres, include_mt5_points=True)} ({nz_dir}) | Status: {nz_lbl}{nz_role}")
            nz_why_d = build_smc_zone_reason(nz, h4_bias=h4_bias)
            if nz_why_d:
                out.append(f"    - WHY                                            : {nz_why_d}")
        else:
            out.append("  • B. NEAREST RADAR ZONE                             : - (Tidak ada zona radar aktif)")

        vz = r_prio.get("best_validated_entry_zone")
        if vz and status == "VALID":
            vz_dir = vz.get("direction_status") or ("ABOVE_PRICE" if vz["price_low"] > price else ("BELOW_PRICE" if vz["price_high"] < price else "INSIDE_ZONE"))
            out.append(f"  • C. BEST VALIDATED ENTRY ZONE                      : {vz['timeframe']} {vz['zone_type']} {vz['price_low']:.2f} - {vz['price_high']:.2f} [Grade: {vz.get('grade', 'A+')}] | Jarak: {format_price_distance(vz.get('distance', 0.0), spec=_u_spec_pres, include_mt5_points=True)} ({vz_dir}) | Status: [VALID]")
            out.append("    - Alasan Validasi                                : Seluruh syarat validasi entry (Zone Touch, Sweep, Trigger M5, Displacement, RR) terpenuhi.")
        else:
            out.append("  • C. BEST VALIDATED ENTRY ZONE                      : NONE")

        az = r_prio.get("alternative_zone")
        if az:
            is_ct = az.get("is_counter") or (cz and (az.get("price_low") == cz.get("price_low") and az.get("zone_type") == cz.get("zone_type")))
            ct_tag = " (COUNTERTREND / REACTION ZONE)" if is_ct else ""
            az_dir = az.get("direction_status") or ("ABOVE_PRICE" if az["price_low"] > price else ("BELOW_PRICE" if az["price_high"] < price else "INSIDE_ZONE"))
            az_skor = f"Skor: {az.get('score', 0.0):.1f}" if az.get("scoring_status") != "DATA_INCOMPLETE" else "Skor: DATA INCOMPLETE"
            out.append(f"  • D. ALTERNATIVE ZONE{ct_tag}                               : {az['timeframe']} {az['zone_type']} {az['price_low']:.2f} - {az['price_high']:.2f} [Grade: {az['grade']} | {az_skor}] | Jarak: {format_price_distance(az.get('distance', 0.0), spec=_u_spec_pres, include_mt5_points=True)} ({az_dir}) | Status: [{az.get('status', 'RADAR')}]{' [COUNTERTREND]' if is_ct else ''}")
            if is_ct:
                opp_act = "BUY" if az['zone_type'] == "DEMAND" else "SELL"
                opp_role = "support/pantau reaksi" if az['zone_type'] == "DEMAND" else "resistance/pantau reaksi"
                out.append(f"    - Catatan Countertrend                            : Area {opp_role} (bukan sinyal {opp_act} karena melawan bias HTF).")
        else:
            out.append("  • D. ALTERNATIVE ZONE                               : NONE")

        if cz and cz != nz and (not az or (az.get("price_low") != cz.get("price_low") or az.get("zone_type") != cz.get("zone_type"))):
            cz_dir = cz.get("direction_status") or ("ABOVE_PRICE" if cz["price_low"] > price else ("BELOW_PRICE" if cz["price_high"] < price else "INSIDE_ZONE"))
            opp_act = "BUY" if cz['zone_type'] == "DEMAND" else "SELL"
            opp_role = "support/pantau reaksi" if cz['zone_type'] == "DEMAND" else "resistance/pantau reaksi"
            out.append(f"  • COUNTERTREND / REACTION ZONE                     : {cz['timeframe']} {cz['zone_type']} {cz['price_low']:.2f} - {cz['price_high']:.2f} [Grade: {cz['grade']}] | Jarak: {format_price_distance(cz['distance'], spec=_u_spec_pres, include_mt5_points=True)} ({cz_dir}) | Status: [RADAR] [COUNTERTREND]")
            out.append(f"    - Catatan Countertrend                            : Area {opp_role} (bukan sinyal {opp_act} karena melawan bias HTF).")
        out.append("")

        # Zone Scoring Summary
        score_sum = r_prio.get("zone_scoring_summary")
        if score_sum:
            out.append("  [ZONE SCORING SUMMARY — DEFINISI RANKING TERPISAH]:")
            act_rank = score_sum.get("active_ranking") or score_sum.get("ranking", [])
            if act_rank:
                out.append("  - A. Active Valid Zone Ranking (Zona Aktif Memenuhi Syarat Radar):")
                for item in act_rank[:5]:
                    st_info = f"Skor {item.get('score', 0.0):.1f}" if item.get("scoring_status") != "DATA_INCOMPLETE" else "DATA INCOMPLETE"
                    grd_info = f"Grade {item.get('grade', '-')}"
                    fresh_info = item.get("freshness", "-")
                    dist_info = f"Jarak {format_price_distance(item.get('distance', 0.0), spec=_u_spec_pres, include_mt5_points=True)}"
                    out.append(f"    {item.get('rank')}. {item.get('zone')} — [{grd_info} | {st_info} | {fresh_info} | {dist_info}]")
                if len(act_rank) > 5:
                    out.append(f"    (Catatan: {len(act_rank) - 5} zona aktif lainnya berada di luar batas tampilan 5 teratas. Ketik 'scan detail' untuk rincian lengkap).")
            else:
                out.append("  - A. Active Valid Zone Ranking: NONE (Tidak ada zona aktif yang valid)")

            raw_rank = score_sum.get("raw_ranking", [])
            if raw_rank:
                out.append("  - B. Raw / Historical Candidate Scoring (Audit Only — Tanpa Otoritas Entry):")
                for item in raw_rank[:5]:
                    st_info = f"{item.get('score', 0.0):.1f}" if item.get("scoring_status") != "DATA_INCOMPLETE" else "DATA INCOMPLETE"
                    v_st = item.get("validity_status", "VALID")
                    rsn = item.get("reason", "")
                    out.append(f"    {item.get('rank')}. {item.get('zone')} (Skor: {st_info}) | Status: {v_st} — {rsn}")
                if len(raw_rank) > 5:
                    out.append(f"    (Catatan: {len(raw_rank) - 5} kandidat mentah lainnya berada di luar batas tampilan 5 teratas).")
                raw_scored_ok = sum(1 for r in raw_rank if r.get("scoring_status") != "DATA_INCOMPLETE")
                raw_excl_inval = sum(1 for r in raw_rank if r.get("validity_status") in ("INVALIDATED", "EXPIRED", "MITIGATED", "REJECTED", "STRUCTURALLY_REJECTED"))
                raw_excl_far = recon.get("far_zones_count", 0) if recon else 0
                out.append(
                    f"    (Audit Raw Scoring: Masuk={len(raw_rank)} | Dinilai={raw_scored_ok} | "
                    f"Top 5 Tampil={min(5, len(raw_rank))} | Di Luar Batas Tampilan={max(0, len(raw_rank) - 5)} | "
                    f"Dikeluarkan dari Active Ranking: Invalid/Exp/Mit/Rej={raw_excl_inval}, Far Zone={raw_excl_far})."
                )

            prio_reasons = score_sum.get("prioritized_reasons", [])
            if prio_reasons:
                out.append("  - Alasan Zona Diprioritaskan:")
                for pr in prio_reasons[:3]:
                    out.append(f"    • {pr}")
            deprio_reasons = score_sum.get("deprioritized_reasons", [])
            if deprio_reasons:
                out.append("  - Alasan Zona Tidak Diprioritaskan:")
                for dr in deprio_reasons[:3]:
                    out.append(f"    • {dr}")
            out.append("")

        # Invalidated / Historical Zones
        inval_list = r_prio.get("invalidated_zones") or dec.get("invalidated_zones") or []
        if inval_list:
            out.append("  [INVALIDATED / HISTORICAL ZONES (Zona historis / tidak aktif untuk entry)]:")
            for iz in inval_list[:5]:
                iz_tf = iz.get("timeframe", "H1")
                iz_t = iz.get("zone_type", "")
                iz_lo = float(iz.get("price_low", 0.0))
                iz_hi = float(iz.get("price_high", 0.0))
                iz_sc = f"{iz.get('score', 0.0):.1f}" if iz.get("scoring_status") != "DATA_INCOMPLETE" else "DATA INCOMPLETE"
                iz_st = iz.get("invalidation_type") or iz.get("invalid_label") or (iz.get("status") if iz.get("status") in ("INVALIDATED", "EXPIRED", "MITIGATED") else "INVALIDATED")
                iz_rsn = iz.get("rejection_reason") or "Struktur invalid / level tertembus (breached)"
                out.append(f"    • {iz_tf} {iz_t} [{iz_lo:.2f} - {iz_hi:.2f}] (Skor: {iz_sc}) — Status: {iz_st} (Alasan: {iz_rsn})")
                iz_why_d = format_smc_invalid_why(iz, h4_bias=h4_bias)
                if iz_why_d:
                    out.append(f"      WHY: {iz_why_d}")
            out.append("")
        elif detailed:
            out.append("  [INVALIDATED / HISTORICAL ZONES (Zona historis / tidak aktif untuk entry)]:")
            out.append("    • NONE (Tidak ada zona yang berstatus invalid / breached)")
            out.append("")

        if detailed:
            if bz and bz != pz and bz != nz and bz != cz:
                bz_lbl = f"[{bz.get('status', 'RADAR')}]" + (f" [{bz.get('approach')}]" if bz.get('approach') in ("FAR", "INSIDE_ZONE") else "")
                bz_dir = bz.get("direction_status") or ("ABOVE_PRICE" if bz["price_low"] > price else ("BELOW_PRICE" if bz["price_high"] < price else "INSIDE_ZONE"))
                bz_d_txt = format_price_distance(bz['distance'], spec=_u_spec_pres, include_mt5_points=True)
                out.append(f"  • Best Entry Monitoring                             : {bz['timeframe']} {bz['zone_type']} {bz['price_low']:.2f} - {bz['price_high']:.2f} [Grade: {bz['grade']}] | Jarak: {bz_d_txt} ({bz_dir}) | Status: {bz_lbl}")
                out.append(f"    - Alasan Monitoring                               : Zona pro-trend optimal: Grade {bz['grade']} (Skor {bz['score']:.1f}), {bz['freshness']}, jarak {bz_d_txt} ({bz_dir}).")

            if sz and sz != bz and sz != pz and sz != nz and sz != cz:
                sz_lbl = f"[{sz.get('status', 'RADAR')}]" + (f" [{sz.get('approach')}]" if sz.get('approach') == "FAR" else "")
                out.append(f"  • Secondary Radar Zone                              : {sz['timeframe']} {sz['zone_type']} {sz['price_low']:.2f} - {sz['price_high']:.2f} [Grade: {sz['grade']}] | Jarak: {format_price_distance(sz['distance'], spec=_u_spec_pres, include_mt5_points=True)} | Status: {sz_lbl}")

            out.append("")
            out.append("  [Kandidat Zona Pemantauan — BUKAN Sinyal Entry / Pending Order Aktif]:")
            demands = [z for z in all_radar if z.get("zone_type") == "DEMAND"]
            supplies = [z for z in all_radar if z.get("zone_type") == "SUPPLY"]
            has_conflict = ctx.get("has_conflict", False)

            if demands:
                for zd in demands[:3]:
                    z_lo, z_hi = float(zd.get("price_low", 0.0)), float(zd.get("price_high", 0.0))
                    d_val, d_st = calculate_zone_distance(z_lo, z_hi, price, digits=_d_pres_px, spec=_u_spec_pres)
                    inval = float(zd.get("invalidation_price", z_lo))
                    tf = zd.get("timeframe", "H1")
                    grd = zd.get("grade", "A")
                    z_st = zd.get("status", "RADAR")
                    fvg_c = zd.get("fvg_confirmation") or {}
                    fvg_str = f"FVG: {fvg_c.get('status', '-')}"
                    ob_c = zd.get("origin_bos") or {}
                    ob_str = f"OB: {ob_c.get('tipe', 'OB')}"
                    is_far = is_far_reference_zone(zd, current_price=price if price > 0 else None, atr=_pres_atr)
                    rel_tag = str(zd.get("zone_relevance") or ("FAR_REFERENCE" if is_far else "RADAR")).upper()
                    far_lbl = " [FAR REFERENCE]" if is_far else f" [{rel_tag}]"
                    ct_lbl = " [COUNTERTREND]" if (h4_bias == "BEARISH") else ""
                    d_txt = format_price_distance(d_val, spec=_u_spec_pres, include_mt5_points=True)
                    st_lbl = f"[{z_st}]{far_lbl}{ct_lbl} " + ("CONFLICT H4 vs H1" if has_conflict else (f"Referensi Jauh {format_price_distance(d_val, spec=_u_spec_pres)} ({d_st})" if is_far else f"Terpantau ({d_st})"))
                    out.append(f"- Kandidat Demand  : {tf} DEMAND {z_lo:.2f} - {z_hi:.2f} [Grade: {grd}] | Jarak: {d_txt} ({d_st}) | Invalidation: {inval:.2f} | {ob_str} | {fvg_str} | Status: {st_lbl}")
                    zd_why = build_smc_zone_reason(zd, h4_bias=h4_bias)
                    if zd_why:
                        out.append(f"  WHY              : {zd_why}")
            else:
                out.append("- Kandidat Demand  : Tidak ada zona Demand aktif yang memenuhi syarat")

            if supplies:
                for zs_z in supplies[:3]:
                    z_lo, z_hi = float(zs_z.get("price_low", 0.0)), float(zs_z.get("price_high", 0.0))
                    d_val, d_st = calculate_zone_distance(z_lo, z_hi, price, digits=_d_pres_px, spec=_u_spec_pres)
                    inval = float(zs_z.get("invalidation_price", z_hi))
                    tf = zs_z.get("timeframe", "H1")
                    grd = zs_z.get("grade", "A")
                    z_st = zs_z.get("status", "RADAR")
                    fvg_c = zs_z.get("fvg_confirmation") or {}
                    fvg_str = f"FVG: {fvg_c.get('status', '-')}"
                    ob_c = zs_z.get("origin_bos") or {}
                    ob_str = f"OB: {ob_c.get('tipe', 'OB')}"
                    is_far = is_far_reference_zone(zs_z, current_price=price if price > 0 else None, atr=_pres_atr)
                    rel_tag = str(zs_z.get("zone_relevance") or ("FAR_REFERENCE" if is_far else "RADAR")).upper()
                    far_lbl = " [FAR REFERENCE]" if is_far else f" [{rel_tag}]"
                    ct_lbl = " [COUNTERTREND]" if (h4_bias == "BULLISH") else ""
                    d_txt = format_price_distance(d_val, spec=_u_spec_pres, include_mt5_points=True)
                    st_lbl = f"[{z_st}]{far_lbl}{ct_lbl} " + ("CONFLICT H4 vs H1" if has_conflict else (f"Referensi Jauh {format_price_distance(d_val, spec=_u_spec_pres)} ({d_st})" if is_far else f"Terpantau ({d_st})"))
                    out.append(f"- Kandidat Supply  : {tf} SUPPLY {z_lo:.2f} - {z_hi:.2f} [Grade: {grd}] | Jarak: {d_txt} ({d_st}) | Invalidation: {inval:.2f} | {ob_str} | {fvg_str} | Status: {st_lbl}")
                    zs_why = build_smc_zone_reason(zs_z, h4_bias=h4_bias)
                    if zs_why:
                        out.append(f"  WHY              : {zs_why}")
            else:
                out.append("- Kandidat Supply  : Tidak ada zona Supply aktif yang memenuhi syarat")
        else:
            out.append("  (Ketik 'scan detail' untuk melihat rincian seluruh kandidat zona)")
    out.append("")

    # C. ENTRY PLAN (ENTRY VALIDATION & OTORITAS SCANNER)
    out.append("C. ENTRY PLAN — 4. ENTRY PLAN (ENTRY VALIDATION & OTORITAS SCANNER)")
    best_p = (early_setups or {}).get("best_setup")

    # HYBRID ENTRY CANDIDATES SUMMARY
    b_hr = r_prio.get("best_high_risk") if r_prio else None
    b_mr = r_prio.get("best_medium_risk") if r_prio else None
    b_lr = r_prio.get("best_low_risk") if r_prio else None
    b_pend = r_prio.get("best_pending_candidate") if r_prio else None
    b_conf = r_prio.get("best_confirmed_candidate") if r_prio else None

    out.append("  [HYBRID ENTRY CANDIDATES SUMMARY]:")
    out.append(f"  • Best High Risk Candidate   : {_cand_label(b_hr)}")
    out.append(f"  • Best Medium Risk Candidate : {_cand_label(b_mr)}")
    out.append(f"  • Best Low Risk Candidate    : {_cand_label(b_lr)}")
    if b_pend:
        p_pln = b_pend.get("pending_plan") or {}
        p_tier = b_pend.get("risk_tier") or "UNTIERED PENDING"
        tier_tag = f" [{p_tier}]" if p_tier in ("UNTIERED PENDING", "HIGH_RISK", "MEDIUM_RISK", "LOW_RISK") else ""
        out.append(f"  • Best Pending Candidate     : {_cand_label(b_pend)}{tier_tag} — Radar Candidate (Entry: {p_pln.get('entry_price', 0.0):.2f}, SL: {p_pln.get('sl', 0.0):.2f}, TP1: {p_pln.get('tp1', 0.0):.2f}, RR: 1:{p_pln.get('rr', 0.0):.1f}); cek EARLY PENDING PLAN untuk validasi planner")
    else:
        out.append("  • Best Pending Candidate     : NONE (Belum ada zona yang memenuhi syarat Pending Limit Grade A)")

    if b_conf and status == "VALID":
        out.append(f"  • Best Confirmed Candidate   : {_cand_label(b_conf)} — Status: CONFIRMED (Trigger M5 Lengkap)")
    else:
        out.append("  • Best Confirmed Candidate   : WAIT (Menunggu konfirmasi M5 Touch/Sweep/CHoCH/Displacement/RR)")
    out.append("")

    # SMC + SNR Confluence (Supporting Context Only)
    confl = dec.get("smc_snr_confluence") or scan.get("smc_snr_confluence")
    target_smc = dec.get("zone") or pz or b_pend
    snr_ctx_det_c = dec.get("snr_context") or scan.get("snr_context")
    if target_smc and snr_ctx_det_c and (
        confl is None
        or str(((confl.get("smc_zone") or {}) if isinstance(confl, dict) else {}).get("zone_id") or "")
        != str(target_smc.get("zone_id") or "")
    ):
        confl = evaluate_smc_snr_confluence(target_smc, snr_ctx_det_c, price, htf_bias=h4_bias)
    if isinstance(confl, dict) and confl.get("available") and (confl.get("is_aligned") or confl.get("opposing_conflict")):
        out.append("  [SMC + SNR CONFLUENCE EVALUATION — SUPPORTING CONTEXT]:")
        for cl in confl.get("display_block", "").splitlines():
            out.append(f"  {cl}")
        out.append("")

    # A. EARLY PENDING PLAN
    out.append("A. EARLY PENDING PLAN (Strategi Agresif - Risiko Lebih Tinggi):")
    if best_p and best_p.get("valid") and h4_bias in ("BULLISH", "BEARISH") and not ctx.get("has_conflict") and m30_status not in ("CONFLICT", "INVALID", "WEAK"):
        bp_det_compact = best_p.get("compact_status") or "READY | MT5 STATUS UNKNOWN"
        bp_det_bst = str(best_p.get("broker_order_status") or "UNKNOWN").upper()
        bp_det_recon = best_p.get("mt5_reconciliation") or {}
        if bp_det_bst in ("PLACED_VERIFIED", "PLACED_MISMATCH", "UNKNOWN"):
            out.append(f"   * Status         : {bp_det_compact}")
        else:
            out.append(f"   * Status         : {bp_det_compact} — PENDING PLAN (BELUM DIPASANG DI MT5)")
        out.append("   * EARLY_PENDING_STATUS : READY")
        out.append(f"   * BROKER_ORDER_STATUS  : {bp_det_bst}")
        out.append(f"   * MT5_RECONCILIATION   : {format_pending_reconciliation_evidence(bp_det_recon)}")
        p_det_src = best_p.get("source_zone_label") or (
            f"{best_p.get('timeframe', 'N/A')} {best_p.get('zone_type', 'ZONE')} "
            f"{float(best_p.get('zone_low', 0.0)):.2f}-{float(best_p.get('zone_high', 0.0)):.2f}"
        )
        pz_det_id = (pz or {}).get("zone_id") if isinstance(pz, dict) else None
        act_det_id = (dec.get("zone") or {}).get("zone_id") if isinstance(dec.get("zone"), dict) else None
        bp_det_id = best_p.get("zone_id")
        if act_det_id and bp_det_id == act_det_id:
            p_det_origin = "SELECTED PENDING | ACTIVE FOCUS"
        elif pz_det_id and bp_det_id == pz_det_id:
            p_det_origin = "SELECTED PENDING | PRIMARY ZONE"
        else:
            p_det_origin = "SELECTED PENDING | NON-FOCUS ZONE"
        out.append(f"   * Source Zone    : {p_det_src} | Grade {best_p.get('zone_grade', 'N/A')} ({float(best_p.get('zone_score', 0.0)):.1f}) [{best_p.get('zone_id', 'N/A')} | {p_det_origin}]")
        out.append(f"   * Order Type     : {best_p['order_type']}")
        out.append(f"   * Entry Price    : {best_p['entry_price']:.2f} ({best_p['entry_reason']})")
        out.append(f"   * Stop Loss      : {best_p['sl']:.2f} (Risk: {format_price_distance(best_p['risk'], spec=_u_spec_pres, include_mt5_points=True, include_pips=True)})")
        out.append(f"   * Take Profit 1  : {best_p['tp1']:.2f} (Reward: {format_price_distance(best_p['reward'], spec=_u_spec_pres, include_mt5_points=True)})")
        out.append(f"   * Take Profit 2  : {best_p['tp2']:.2f}")
        out.append(f"   * Risk Distance  : {format_price_distance(best_p.get('risk_points', best_p.get('risk', 0.0)), spec=_u_spec_pres, include_mt5_points=True)}")
        out.append(f"   * Reward TP1     : {format_price_distance(best_p.get('reward_tp1', best_p.get('reward', 0.0)), spec=_u_spec_pres, include_mt5_points=True)}")
        out.append(f"   * Reward TP2     : {format_price_distance(best_p.get('reward_tp2', 0.0), spec=_u_spec_pres, include_mt5_points=True)}")
        out.append(f"   * EARLY_PENDING_RR (TP1) : PASSED (1:{best_p.get('rr_tp1', best_p.get('rr', 0.0)):.1f})")
        out.append(f"   * EARLY_PENDING_RR (TP2) : 1:{best_p.get('rr_tp2', 0.0):.1f}")
        out.append(f"   * Risk/Reward    : 1:{best_p['rr']:.1f}")
        out.append(f"   * Invalidation   : {best_p['invalidation_price']:.2f}")
        out.append(f"   * Structural Invalidation : {best_p['invalidation_price']:.2f}")
        out.append(f"   * Invalidation Buffer : {format_price_distance(best_p.get('invalidation_buffer', 0.0), spec=_u_spec_pres, include_mt5_points=True)}")
        out.append(f"   * Zone Boundary  : {best_p.get('zone_low', 0.0):.2f} - {best_p.get('zone_high', 0.0):.2f}")
        out.append(f"   * Suggested Exp  : {best_p['suggested_expiration']}")
        if best_p.get("expiration_block"):
            for el in best_p["expiration_block"].splitlines():
                if el.startswith("Pending Plan Expiration:"):
                    out.append(f"   * {el}")
                else:
                    out.append(f"     {el}")
        if bp_det_bst in ("PLACED_VERIFIED", "PLACED_MISMATCH"):
            out.append(f"   * Keterangan     : {bp_det_compact} ({format_pending_reconciliation_evidence(bp_det_recon)}).")
        elif bp_det_bst == "UNKNOWN":
            out.append("   * Keterangan     : MT5 STATUS UNKNOWN (Verifikasi pending order MT5 belum dapat dilakukan).")
        else:
            out.append("   * Keterangan     : BELUM DIPASANG DI MT5 (Rencana pending order agresif; eksekusi manual oleh trader).")
        out.append("   * Peringatan     : Setup agresif sebelum konfirmasi M5; pasang limit manual hanya jika siap menanggung risiko.")
        alt_det_list = [
            ap for ap in ((early_setups or {}).get("all_setups") or [])
            if isinstance(ap, dict)
            and ap.get("valid")
            and str(ap.get("status") or "READY").upper() == "READY"
            and not ap.get("is_expired")
            and ap.get("zone_id")
            and ap.get("zone_id") != best_p.get("zone_id")
        ]
        for idx_ap, ap_det in enumerate(alt_det_list, start=1):
            ap_det_src = ap_det.get("source_zone_label") or (
                f"{ap_det.get('timeframe', 'N/A')} {ap_det.get('zone_type', 'ZONE')} "
                f"{float(ap_det.get('zone_low', 0.0)):.2f}-{float(ap_det.get('zone_high', 0.0)):.2f}"
            )
            ap_det_recon = ap_det.get("mt5_reconciliation") or {}
            out.append(
                f"   * Alt Pending #{idx_ap} : {ap_det_src} | Grade {ap_det.get('zone_grade', 'N/A')} ({float(ap_det.get('zone_score', 0.0)):.1f}) "
                f"[{ap_det.get('zone_id', 'N/A')}] -> {ap_det.get('compact_status', 'READY')} "
                f"(SL {float(ap_det.get('sl', 0.0)):.2f} | TP1 {float(ap_det.get('tp1', 0.0)):.2f} | RR 1:{float(ap_det.get('rr', 0.0)):.1f})"
            )
            out.append(f"     - Alt MT5 Reconcile  : {format_pending_reconciliation_evidence(ap_det_recon)}")
        if detailed:
            out.append("   * Penjelasan Level:")
            out.append(f"     - Zone Boundary           : Area reaksi institusional ({best_p.get('zone_low', 0.0):.2f} - {best_p.get('zone_high', 0.0):.2f}) tempat order berpotensi terisi.")
            out.append(f"     - Structural Invalidation : Level ({best_p['invalidation_price']:.2f}) di mana struktur zona resmi batal jika tertembus.")
            out.append(f"     - Stop Loss               : Level proteksi modal ({best_p['sl']:.2f}) yang diletakkan di luar Invalidation dengan buffer ({format_price_distance(best_p.get('invalidation_buffer', 0.0), spec=_u_spec_pres, include_mt5_points=True)}) untuk menghindari market noise/spread.")
    else:
        rej_early = (early_setups or {}).get("empty_reason") or (best_p.get("rejection_reason") if best_p else "Syarat pending order belum terpenuhi")
        if h4_bias == "NETRAL":
            rej_early = "H4 belum memberikan directional bias yang jelas (NETRAL)"
        out.append(f"   * Status         : TIDAK AKTIF ({rej_early})")
        out.append("   * EARLY_PENDING_STATUS : TIDAK_AKTIF")
        out.append("   * EARLY_PENDING_RR : DATA_UNAVAILABLE")
        out.append("   * Recommendation : NO PENDING SETUP (WAIT)")
        out.append("   * Entry Price    : -")
        out.append("   * Stop Loss      : -")
        out.append("   * Take Profit    : -")
        out.append("   * Risk/Reward    : -")
    for _mt5_det_ln in format_mt5_orders_summary_lines(
        _ord_snap_pres,
        invalid_zone_warnings=_invalid_pending_warnings,
        matched_recons=_matched_recons_pres,
        detailed=True,
        invalid_zone_scope="SMC",
    ):
        out.append(f"   * {_mt5_det_ln.lstrip('- ')}" if not _mt5_det_ln.startswith("  ") else f"     {_mt5_det_ln.strip()}")

    out.append("")
    # B. CONFIRMED ENTRY (CONSERVATIVE M5)
    out.append("B. CONFIRMED ENTRY (Strategi Konservatif - Sinyal Resmi Terkonfirmasi):")
    if status == "VALID" and plan:
        confirmed_rr_val = f"PASSED (1:{plan.get('rr', 0.0):.1f})"
    elif tr == "CONFIRMED" and dp == "CONFIRMED" and rr_st == "CONFIRMED":
        confirmed_rr_val = "PASSED"
    elif tr == "CONFIRMED" and dp == "CONFIRMED" and (rr_st in ("WAIT", "FAILED")):
        confirmed_rr_val = "FAILED"
    else:
        confirmed_rr_val = "NOT_CHECKED (Menunggu trigger M5 lengkap)"

    global_m5_sw = dec.get("global_m5_sweep_status") or scan.get("global_m5_sweep_status") or "NOT_CHECKED"
    out.append(f"   * GLOBAL_M5_SWEEP_STATUS     : {global_m5_sw}")
    out.append(f"   * ZONE_SPECIFIC_SWEEP_STATUS : {sw}" + (f" ({sw_desc})" if sw == "CONFIRMED" and sw_desc else ""))
    out.append(f"   * Zone Touch     : {zt}")
    out.append(f"   * Liquidity Sweep: {sw}" + (f" ({sw_desc})" if sw == "CONFIRMED" and sw_desc else ""))
    out.append(f"   * CHoCH/BOS/CISD : {tr}" + (f" ({tr_desc})" if tr == "CONFIRMED" and tr_desc else ""))
    out.append(f"   * Displacement   : {dp}" + (f" ({dp_desc})" if dp == "CONFIRMED" and dp_desc else ""))
    out.append(f"   * CONFIRMED_ENTRY_RR : {confirmed_rr_val}")
    out.append(
        f"   * Risk/Reward    : {rr_st}"
        + (
            f" (1:{plan['rr']:.1f})"
            if (status == "VALID" and plan and plan.get("rr"))
            else (f" ({rr_desc})" if rr_desc else "")
        )
    )

    if status == "VALID" and plan:
        out.append(f"   * BUY / SELL / WAIT / INVALID: {direction} VALID")
        out.append(f"   * Sinyal Resmi   : {direction} CONFIRMED")
        out.append(f"   * Entry Price    : {plan.get('entry_acuan', 0.0):.2f} (Area: {plan.get('entry_min', 0.0):.2f} - {plan.get('entry_max', 0.0):.2f})")
        out.append(f"   * Stop Loss      : {plan.get('sl', 0.0):.2f}")
        out.append(f"   * Take Profit 1  : {plan.get('tp1', 0.0):.2f}")
        out.append(f"   * Take Profit 2  : {plan.get('tp2', 0.0):.2f}" if plan.get("tp2") else f"   * Take Profit 2  : -")
        out.append(f"   * Risk/Reward    : 1:{plan.get('rr', 0.0):.1f}")
        out.append(f"   * Setup Quality  : HIGH ({zone_dict.get('grade', 'A+') if zone_dict else 'A+'})")
    else:
        out.append(f"   * BUY / SELL / WAIT / INVALID: {status} (NO VALID SETUP)")
        out.append(f"   * Sinyal Resmi   : TIDAK ADA SINYAL AKTIF ({status})")
        out.append("   * Entry Price    : - (Setup belum valid)")
        out.append("   * Stop Loss      : -")
        out.append("   * Take Profit    : -")
        out.append("   * Risk/Reward    : -")
        out.append("   * Setup Quality  : -")
    out.append("")

    # Entry Validation Summary
    val_sum = dec.get("entry_validation_summary")
    if not val_sum:
        mon_zone = (
            f"{zone_dict.get('timeframe', 'H1')} {zone_dict.get('zone_type', '')} {zone_dict.get('price_low', 0.0):.2f}-{zone_dict.get('price_high', 0.0):.2f}"
            if zone_dict else "NONE"
        )
        passed_c = []
        missing_c = []
        for name, val in [("Zone Touch", zt), ("Liquidity Sweep", sw), ("CHoCH/BOS/CISD", tr), ("Displacement", dp), ("Risk/Reward", rr_st)]:
            if val == "CONFIRMED":
                passed_c.append(name)
            else:
                missing_c.append(name)
        val_sum = {
            "monitored_zone": mon_zone,
            "fulfilled_conditions": passed_c,
            "unfulfilled_conditions": missing_c,
            "final_verdict": f"{direction} VALID" if status == "VALID" else status,
        }

    out.append("  [ENTRY VALIDATION SUMMARY]:")
    out.append(f"  • Zona yang Dipantau       : {val_sum.get('monitored_zone', 'NONE')}")
    m_ident = dec.get("monitored_zone_identity")
    if m_ident:
        out.append(f"  • Identitas Zona Terpilih  : ID {m_ident.get('zone_id')} | Arah: {m_ident.get('direction')} | Proximal: {m_ident.get('proximal', 0.0):.2f} | Distal: {m_ident.get('distal', 0.0):.2f} | Status: {m_ident.get('status')}")
    fc = val_sum.get("conditions_met") or val_sum.get("fulfilled_conditions") or val_sum.get("conditions_pass") or []
    out.append(f"  • Kondisi Terpenuhi        : {', '.join(fc) if fc else 'Belum ada'}")
    fail_c = val_sum.get("conditions_unmet") or val_sum.get("unfulfilled_conditions") or val_sum.get("conditions_fail") or []
    if fail_c:
        ufc_text = ', '.join(fail_c)
    elif status == "VALID":
        ufc_text = "None (Semua syarat terpenuhi)"
    elif m_ident or (val_sum.get("monitored_zone") and val_sum.get("monitored_zone") != "NONE"):
        ufc_text = "Zona aktif tersedia dan dipantau; evaluasi trigger M5 menunggu harga menyentuh zona"
    else:
        ufc_text = "Tidak ada zona aktif yang memenuhi syarat untuk dipantau"
    out.append(f"  • Kondisi Belum Terpenuhi  : {ufc_text}")
    nc_c = val_sum.get("conditions_not_checked") or val_sum.get("not_checked_conditions") or []
    if nc_c:
        out.append(f"  • Belum Dijalankan         : {', '.join(nc_c)}")
    inc_c = val_sum.get("conditions_incomplete") or val_sum.get("incomplete_conditions") or []
    if inc_c:
        out.append(f"  • Data Tidak Lengkap       : {', '.join(inc_c)}")
    verdict = sanitize_wait_reason(val_sum.get('final_decision_reason') or val_sum.get('final_verdict') or status)
    out.append(f"  • Alasan Akhir             : {verdict}")
    out.append("")

    # D. FINAL DECISION & REASON
    out.append("D. FINAL DECISION — 1. FINAL DECISION & 5. REASON")
    if status == "VALID" and plan:
        out.append(f"- Status           : {direction} VALID")
        out.append(f"- Recommendation   : {direction} CONFIRMED (Eksekusi Sesuai Rencana)")
        out.append(f"- Direction        : {direction}")
    else:
        out.append(f"- Status           : {status} (NO VALID SETUP)")
        out.append("- Recommendation   : NO ENTRY / WAIT (Dilarang Buka Posisi Baru)")
        out.append("- Direction        : -")

    out.append("  [CANDIDATE SELECTION OVERVIEW]:")
    pz_ov = r_prio.get("primary_directional_zone") if r_prio else None
    nz_ov = r_prio.get("nearest_radar_zone") if r_prio else None
    m_ident_ov = dec.get("monitored_zone_identity")
    mon_ov_label = (
        f"{m_ident_ov.get('timeframe')} {m_ident_ov.get('zone_type')} ({m_ident_ov.get('price_low', 0.0):.2f}-{m_ident_ov.get('price_high', 0.0):.2f}) [ID: {m_ident_ov.get('zone_id')}]"
        if m_ident_ov else _cand_label(pz_ov)
    )
    out.append(f"  • Primary Directional Zone       : {_cand_label(pz_ov)}")
    out.append(f"  • Nearest Radar Zone             : {_cand_label(nz_ov)}")
    out.append(f"  • Confirmed Entry Monitoring Zone: {mon_ov_label}")
    if nz_ov and m_ident_ov and nz_ov.get("zone_id") != m_ident_ov.get("zone_id"):
        if nz_ov.get("is_counter"):
            nz_skip_reason = f"berlawanan arah bias HTF ({h4_bias})"
        elif pz_ov and pz_ov.get("zone_id") != nz_ov.get("zone_id"):
            nz_skip_reason = f"prioritas komposit lebih rendah ({float(nz_ov.get('priority_score') or 0.0):.1f} vs {float(pz_ov.get('priority_score') or 0.0):.1f})"
        else:
            nz_skip_reason = f"evaluator entry memantau kandidat {m_ident_ov.get('timeframe')} {m_ident_ov.get('zone_type')} ({m_ident_ov.get('zone_id')})"
        out.append(f"    - Catatan Nearest vs Monitoring: Nearest Radar Zone tidak dipilih sebagai Confirmed Entry Monitoring Zone karena {nz_skip_reason}.")
    out.append(f"  • Best High Risk Zone        : {_cand_label(b_hr)}")
    out.append(f"  • Best Medium Risk Zone      : {_cand_label(b_mr)}")
    out.append(f"  • Best Low Risk Zone         : {_cand_label(b_lr)}")
    p_tier_ov = b_pend.get("risk_tier") or "UNTIERED PENDING" if b_pend else ""
    tier_tag_ov = f" [{p_tier_ov}]" if p_tier_ov in ("UNTIERED PENDING", "HIGH_RISK", "MEDIUM_RISK", "LOW_RISK") else ""
    out.append(f"  • Best Pending Candidate     : {(_cand_label(b_pend) + tier_tag_ov) if b_pend else 'NONE'}")
    out.append(f"  • Best Confirmed Candidate   : {_cand_label(b_conf) if (b_conf and status == 'VALID') else 'NONE'}")
    out.append("")

    if not r_prio and all_radar:
        r_prio = prioritize_radar_zones(
            all_radar, price, htf_bias=h4_bias, h1_structure=h1_struct,
            atr=_pres_atr, m5_stages=stages, bid=scan.get("bid"), ask=scan.get("ask"),
            snapshot_time=scan.get("snapshot_time")
        )

    setup_t_det = str(dec.get("setup_type") or (plan.get("setup_type") if isinstance(plan, dict) else None) or ("TREND_PULLBACK" if status == "VALID" else "NONE")).upper()
    sig_st_det = str(dec.get("signal_status") or (SIGNAL_STATUS_SIGNAL_CONFIRMED if status == "VALID" else SIGNAL_STATUS_NO_SETUP)).upper()
    snr_obj_det = (dec.get("snr_context") if isinstance(dec.get("snr_context"), dict) else None) or (scan.get("snr_context") if isinstance(scan.get("snr_context"), dict) else None) or {}
    snr_sum_det = snr_obj_det.get("summary") or "Independen"
    snr_st_det = snr_obj_det.get("status")
    snr_fresh_det = (snr_obj_det.get("freshness") or {}).get("status") if isinstance(snr_obj_det.get("freshness"), dict) else None
    out.append(f"- Setup Type               : {setup_t_det}")
    out.append(f"- Signal Lifecycle Status  : {sig_st_det}")
    out.append(f"- SNR Supporting Context   : {snr_sum_det}")
    if snr_fresh_det:
        out.append(f"- SNR Freshness            : {snr_fresh_det}")
    if snr_st_det in ("AVAILABLE", "PARTIAL"):
        s_det = snr_obj_det.get("support") or ((snr_obj_det.get("supports") or [None])[0])
        r_det = snr_obj_det.get("resistance") or ((snr_obj_det.get("resistances") or [None])[0])
        from modules.snr.formatter import format_snr_nearest_support_line
        _eff_h4_det = h4_bias if h4_bias in ("BULLISH", "BEARISH") else str((snr_obj_det.get("market_bias") or {}).get("h4") or "").upper()
        s_det_str = format_snr_nearest_support_line(s_det, current_price=price, h4_bias=_eff_h4_det, unit_spec=_u_spec_pres) if isinstance(s_det, dict) else "NONE"
        r_det_str = f"{r_det.get('low', 0.0):.2f}-{r_det.get('high', 0.0):.2f}" if isinstance(r_det, dict) else "NONE"
        out.append(f"- SNR Support / Resistance : Support={s_det_str} | Resistance={r_det_str}")
        if snr_obj_det.get("available_timeframes") and snr_st_det == "PARTIAL":
            out.append(f"- SNR Available TF         : {', '.join(snr_obj_det.get('available_timeframes') or [])}")
        if snr_obj_det.get("missing_timeframes"):
            out.append(f"- SNR Missing/Short TF     : {', '.join(snr_obj_det.get('missing_timeframes') or [])}")
    if snr_st_det in ("PARTIAL", "UNAVAILABLE") or snr_obj_det.get("error_reason"):
        det_err_info = snr_obj_det.get("error_reason") or snr_obj_det.get("reason") or "; ".join(snr_obj_det.get("warnings") or [])
        if det_err_info:
            out.append(f"- SNR Integration Note     : [{snr_st_det or 'UNAVAILABLE'}] {det_err_info}")
    if status == "VALID" and plan:
        reasons_txt = "; ".join(dec.get("reasons", [])) or "Setup lolos validasi multi-timeframe"
        out.append(f"- Final Decision Reason    : {reasons_txt}")
        out.append(f"- Alasan Keputusan         : {reasons_txt}")
        if r_prio and r_prio.get("primary_reason"):
            out.append(f"- Zone Priority Reason     : {r_prio['primary_reason']}")
        if r_prio and r_prio.get("distance_to_zone_str"):
            out.append(f"- Distance to Zone         : {r_prio['distance_to_zone_str']}")
        out.append("- Validated Conditions     : Seluruh tahap validasi entry M5 & R:R terkonfirmasi.")
    else:
        explicit_reasons = [r.replace("[SCANNER] Rejection Reason: ", "") for r in dec.get("reasons", []) if r]
        dyn_wait = r_prio.get("dynamic_wait_reason") if r_prio else None

        # Jika terdapat konflik struktur H4 vs H1
        if ctx.get("has_conflict") or (explicit_reasons and any("konflik" in r.lower() for r in explicit_reasons)):
            eval_sum_w = scan.get("zone_evaluation_summary") or {}
            recon_w = (r_prio.get("zone_count_reconciliation") if r_prio else None) or dec.get("zone_count_reconciliation") or {}
            rq_cnt = recon_w.get("radar_qualified_count")
            if rq_cnt is None:
                rq_cnt = eval_sum_w.get("radar_qualified")
            if rq_cnt is None:
                rq_cnt = len(r_prio.get("radar_qualified", [])) if r_prio else len(all_radar)
            fz_cnt = int(recon_w.get("far_zones_count", 0) or 0)
            inval_list_w = (r_prio.get("invalidated_zones") if r_prio else None) or dec.get("invalidated_zones") or []
            inval_cnt = int(eval_sum_w.get("total_invalidated", 0) or 0)
            mit_cnt = int(eval_sum_w.get("total_mitigated", 0) or 0)
            exp_cnt = int(eval_sum_w.get("total_expired", 0) or 0)
            if not eval_sum_w and inval_list_w:
                inval_cnt = sum(1 for iz in inval_list_w if str(iz.get("status", "")).upper() == "INVALIDATED")
                mit_cnt = sum(1 for iz in inval_list_w if str(iz.get("status", "")).upper() == "MITIGATED")
                exp_cnt = sum(1 for iz in inval_list_w if str(iz.get("status", "")).upper() == "EXPIRED")
            radar_clause = _format_radar_status_clause(
                radar_qualified_count=int(rq_cnt or 0),
                far_zones_count=fz_cnt,
                invalidated_count=inval_cnt,
                mitigated_count=mit_cnt,
                expired_count=exp_cnt,
            )
            dyn_wait = f"{htf_conflict_phrase or f'Konflik struktur H4 {h4_bias} vs H1 {h1_struct}'}. Belum ada setup eligible; trigger M5 tidak dijalankan. {radar_clause}"
        elif not dyn_wait:
            if h4_bias == "NETRAL":
                h4_diag_str = f" ({h4_diag})" if h4_diag else ""
                dyn_wait = f"H4 belum memberikan directional bias yang jelas (NETRAL){h4_diag_str}. Menunggu breakout struktur."
            else:
                dyn_wait = explicit_reasons[0] if explicit_reasons else (scan.get("empty_reason") or "Persyaratan entry belum terpenuhi")

        # Jika h4_diag spesifik ada dan h4 netral, pastikan termuat
        if h4_bias == "NETRAL" and h4_diag and h4_diag not in dyn_wait:
            dyn_wait = f"Bias H4 NETRAL ({h4_diag}). Belum ada directional bias yang jelas. Zona Supply dan Demand dipantau sebagai batas range reaksi, bukan sinyal entry."

        dyn_wait = sanitize_wait_reason(dyn_wait)
        out.append(f"- Final WAIT Reason        : {dyn_wait}")
        out.append(f"- Alasan Keputusan         : {dyn_wait}")

        wb = (r_prio.get("wait_breakdown") if r_prio else None) or dec.get("wait_breakdown") or {}
        p_block = sanitize_wait_reason(wb.get("primary_blocker") or dyn_wait)
        pend_blocks = [sanitize_wait_reason(pb) for pb in (wb.get("pending_limit_blockers") or []) if pb]
        conf_blocks = [sanitize_wait_reason(cb) for cb in (wb.get("confirmed_entry_blockers") or []) if cb]
        sec_warns = [sanitize_wait_reason(sw) for sw in (wb.get("secondary_warnings") or []) if sw]

        out.append("  [WAIT REASON BREAKDOWN]:")
        out.append(f"  • PRIMARY BLOCKER        : {p_block}")
        out.append("  • PENDING LIMIT BLOCKERS :")
        if pend_blocks:
            for pb in pend_blocks:
                out.append(f"    - {pb}")
        else:
            out.append("    - (Syarat Pending Limit terpenuhi)")

        out.append("  • CONFIRMED ENTRY BLOCKERS:")
        if conf_blocks:
            for cb in conf_blocks:
                out.append(f"    - {cb}")
        elif no_eligible_setup_fmt:
            out.append("    - (Belum ada setup eligible — trigger M5 tidak dijalankan)")
        else:
            out.append("    - (Menunggu konfirmasi M5 lengkap)")

        if sec_warns:
            out.append("  • SECONDARY WARNINGS     :")
            for sw_item in sec_warns:
                out.append(f"    - {sw_item}")

        if r_prio and r_prio.get("primary_reason"):
            out.append(f"- Zone Priority Reason     : {r_prio['primary_reason']}")
        if r_prio and r_prio.get("distance_to_zone_str"):
            out.append(f"- Distance to Zone         : {r_prio['distance_to_zone_str']}")
        if r_prio and r_prio.get("missing_conditions"):
            out.append("- Missing Entry Conditions :")
            for mc in r_prio["missing_conditions"]:
                out.append(f"  * {mc}")
    out.append("")

    _pm = premarket_lines(dec)
    if _pm:
        out.extend(_pm + [""])

    # E. RISK WARNING
    out.append("E. RISK WARNING — 6. RISK WARNING")
    if detailed:
        out.append("- Peringatan Risiko Pasar: Pergerakan XAUUSD memiliki volatilitas tinggi.")
        out.append("  * Klasifikasi Risiko Multi-Tier: Klasifikasi HIGH RISK / MEDIUM RISK / LOW RISK merefleksikan karakteristik teknikal setup, BUKAN pembenaran untuk meningkatkan ukuran lot atau risiko modal!")
        out.append("  * Manajemen Risiko Modal:")
        out.append("    - 0,5%: risiko lebih konservatif (modal terlindungi secara ketat).")
        out.append("    - 1,0%: risiko lebih tinggi dibandingkan 0,5% (eksposur risiko modal standar/maksimal).")
        out.append("  * Perbedaan Persentase Modal vs Ukuran Lot: Persentase risiko adalah toleransi kerugian modal; ukuran lot dihitung secara matematis dari nominal risiko dibagi jarak SL x contract size MT5.")
        out.append("  * Integritas Perhitungan: Jangan mengubah Entry, SL, TP, atau RR resmi yang dihitung oleh Python Scanner.")
        out.append("  * Status Lot: Jika data kontrak MT5 tidak lengkap atau koneksi terputus, kalkulasi lot dinyatakan DATA UNAVAILABLE (tidak mengarang ukuran lot).")
        out.append("  * Status WAIT: Dilarang membuka posisi baru sebelum seluruh syarat validasi terpenuhi.")
        out.append("  * Pending Plan (READY): Bersifat agresif sebelum konfirmasi M5; alokasi risiko disarankan 0,5% (lebih konservatif).")
        out.append("  * Perlindungan Modal: Wajib pasang Stop Loss di balik invalidation level struktur.")
    else:
        out.append("- Peringatan Risiko: Pergerakan XAUUSD memiliki volatilitas tinggi. Disiplin MM (alokasi risiko modal 0,5% konservatif vs 1,0% standar).")
        out.append("- Klasifikasi Risiko Multi-Tier: Klasifikasi HIGH RISK / MEDIUM RISK / LOW RISK merefleksikan karakteristik teknikal setup, BUKAN pembenaran untuk meningkatkan ukuran lot atau risiko modal!")
        out.append("- Status Pending Plan: Early Pending Plan bersifat rencana analisis dan BELUM dipasang di MT5 (eksekusi manual oleh trader).")
        out.append("- Kewajiban Stop Loss: Wajib pasang Stop Loss di balik level invalidation struktur institusional.")
    out.append("=" * 70)
    return "\n".join(out)


def format_decision_text(dec: Dict[str, Any]) -> str:
    audit_txt = format_scanner_audit(dec)
    lines = [f"[ENTRY DECISION] {dec['status']} {dec.get('direction') or ''}".rstrip()]
    for name in ("DATA", "H4", "H1", "M30", "M5", "RISK", "ERROR"):
        if name in dec["stages"]:
            s = dec["stages"][name]
            lines.append(f"  {'OK ' if s['ok'] else 'XX '}{name}: {s['msg']}")
    for r in dec["reasons"]:
        lines.append(f"  - {r}")
    lines.append("-" * 40)
    lines.append(audit_txt)
    return "\n".join(lines)


def _extract_risk_monitor_snapshot(
    pos_list: Optional[Any] = None,
    orders_list: Optional[Any] = None,
    risk_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Mengambil ringkasan fakta risiko akun MT5 (posisi running, pending order, dan status SL/TP)
    tanpa mengubah logika trading maupun mengeksekusi transaksi.
    """
    if isinstance(risk_data, dict):
        rp = int(risk_data.get("running_position", 0) or 0)
        po = int(risk_data.get("pending_orders", 0) or 0)
        p_nosl = int(risk_data.get("position_without_sl", 0) or 0)
        o_nosl = int(risk_data.get("pending_without_sl", 0) or 0)
        p_notp = int(risk_data.get("position_without_tp", 0) or 0)
        o_notp = int(risk_data.get("pending_without_tp", 0) or 0)
        return {
            "available": bool(risk_data.get("available", True)),
            "running_position": rp,
            "pending_orders": po,
            "position_without_sl": p_nosl,
            "pending_without_sl": o_nosl,
            "position_without_tp": p_notp,
            "pending_without_tp": o_notp,
            "positions": list(pos_list or ()),
            "orders": list(orders_list or ()),
        }

    available = (pos_list is not None) or (orders_list is not None)
    orders_query_ok = orders_list is not None
    orders_snapshot = None
    if pos_list is None or orders_list is None:
        try:
            snap_live = fetch_mt5_pending_orders_snapshot(
                symbol=SYMBOL,
                orders_override=orders_list if orders_list is not None else None,
                positions_override=pos_list if pos_list is not None else None,
                allow_live_query=True,
            )
            if snap_live.get("query_ok"):
                orders_snapshot = snap_live
                if orders_list is None:
                    orders_list = snap_live.get("orders") or []
                    orders_query_ok = True
                if pos_list is None and snap_live.get("positions_query_ok"):
                    pos_list = snap_live.get("positions") or []
                available = True
        except Exception:
            pass

    positions = list(pos_list or ())
    orders = list(orders_list or ())
    if orders_snapshot is None:
        orders_snapshot = fetch_mt5_pending_orders_snapshot(
            symbol=SYMBOL,
            orders_override=orders if orders_query_ok else None,
            positions_override=positions if pos_list is not None else None,
            allow_live_query=False,
        )

    def _get_float(obj: Any, key: str) -> float:
        if isinstance(obj, dict):
            val = obj.get(key, 0.0)
        else:
            val = getattr(obj, key, 0.0)
        try:
            return float(val or 0.0)
        except Exception:
            return 0.0

    pos_no_sl = sum(1 for p in positions if _get_float(p, "sl") <= 0.0)
    ord_no_sl = sum(1 for o in orders if _get_float(o, "sl") <= 0.0)
    pos_no_tp = sum(1 for p in positions if _get_float(p, "tp") <= 0.0)
    ord_no_tp = sum(1 for o in orders if _get_float(o, "tp") <= 0.0)

    return {
        "available": available,
        "running_position": len(positions),
        "pending_orders": len(orders),
        "position_without_sl": pos_no_sl,
        "pending_without_sl": ord_no_sl,
        "position_without_tp": pos_no_tp,
        "pending_without_tp": ord_no_tp,
        "positions": positions,
        "orders": orders,
        "orders_snapshot": orders_snapshot,
    }


def format_risk_alert(
    pos_list: Optional[Any] = None,
    orders_list: Optional[Any] = None,
    risk_data: Optional[Dict[str, Any]] = None,
) -> str:
    """Memformat blok RISK MONITOR ringkas beserta peringatan SL/TP kritis."""
    snap_r = _extract_risk_monitor_snapshot(
        pos_list=pos_list, orders_list=orders_list, risk_data=risk_data
    )
    lines = [
        "-" * 56,
        "RISK MONITOR",
        "-" * 56,
    ]
    if not snap_r["available"]:
        lines.extend([
            "RUNNING POSITION : N/A",
            "PENDING ORDERS   : N/A",
            "POSITION WITHOUT SL : N/A",
            "PENDING WITHOUT SL  : N/A",
            "",
            "WARNING:",
            "Data posisi/pending MT5 belum tersedia (MT5 tidak terhubung).",
        ])
        return "\n".join(lines)

    rp = snap_r["running_position"]
    po = snap_r["pending_orders"]
    p_nosl = snap_r["position_without_sl"]
    o_nosl = snap_r["pending_without_sl"]
    p_notp = snap_r["position_without_tp"]
    o_notp = snap_r["pending_without_tp"]

    lines.extend([
        f"RUNNING POSITION : {rp}",
        f"PENDING ORDERS   : {po}",
        f"POSITION WITHOUT SL : {p_nosl}",
        f"PENDING WITHOUT SL  : {o_nosl}",
    ])

    warnings = []
    ord_snap_r = snap_r.get("orders_snapshot") if isinstance(snap_r.get("orders_snapshot"), dict) else {}
    pos_nosl_tkts = [f"#{t}" for t in (ord_snap_r.get("positions_without_sl") or []) if t is not None]
    ord_nosl_tkts = [f"#{t}" for t in (ord_snap_r.get("orders_without_sl") or []) if t is not None]
    pos_tkt_note = f" ({', '.join(pos_nosl_tkts)})" if pos_nosl_tkts else ""
    ord_tkt_note = f" ({', '.join(ord_nosl_tkts)})" if ord_nosl_tkts else ""
    if p_nosl > 0 and o_nosl > 0:
        warnings.append(f"Ada posisi{pos_tkt_note} dan pending order{ord_tkt_note} yang belum memiliki Stop Loss (SL MISSING — HIGH RISK).")
    elif p_nosl > 0:
        warnings.append(f"Ada posisi running{pos_tkt_note} yang belum memiliki Stop Loss (SL MISSING — HIGH RISK).")
    elif o_nosl > 0:
        warnings.append(f"Ada pending order{ord_tkt_note} yang belum memiliki Stop Loss (SL MISSING — HIGH RISK).")

    if p_notp > 0 or o_notp > 0:
        warnings.append("Ada posisi/pending order yang belum memiliki Take Profit.")

    if warnings:
        lines.append("")
        lines.append("WARNING:")
        lines.extend(warnings)

    return "\n".join(lines)


def _norm_stage_label(val: Any, default: str = "NOT CHECKED") -> str:
    if val is None:
        return default
    s = str(val).strip()
    if not s:
        return default
    if s.upper() == "NOT_CHECKED":
        return "NOT CHECKED"
    return s.upper()


def _format_wib_summary_time(raw_ts: Any) -> str:
    dt = to_wib_datetime(raw_ts)
    if dt is None:
        dt = get_current_wib()
    return dt.strftime("%d-%m-%Y %H:%M WIB")


def format_scan_summary(
    scan: Optional[Dict[str, Any]] = None,
    dec: Optional[Dict[str, Any]] = None,
    early_setups: Optional[Dict[str, Any]] = None,
    tracker: Optional[Any] = None,
    pos_list: Optional[Any] = None,
    orders_list: Optional[Any] = None,
    risk_data: Optional[Dict[str, Any]] = None,
) -> str:
    """
    DEFAULT MODE formatter untuk command `scan` (Two-Level Output Display).
    Menampilkan ringkasan singkat, informatif, dan terstruktur tanpa mengubah logika analisis.
    """
    if scan is None:
        _snap = create_market_snapshot()
        scan = scan_xauusd_zones(snapshot=_snap)
        dec = evaluate_trade_decision(snapshot=_snap, scan_result=scan)
    elif dec is None:
        _snap = scan.get("snapshot") if isinstance(scan.get("snapshot"), dict) else None
        dec = evaluate_trade_decision(snapshot=_snap, scan_result=scan)

    price_raw = scan.get("current_price") or dec.get("current_price") or 0.0
    try:
        price = float(price_raw)
    except Exception:
        price = 0.0

    time_str = _format_wib_summary_time(scan.get("snapshot_time") or scan.get("timestamp"))

    ctx = scan.get("htf_context") or scan.get("context") or {}
    stages = dec.get("stages") or {}
    status = str(dec.get("status", "WAIT")).upper()
    direction = dec.get("direction")

    h4_bias = str(ctx.get("H4", "N/A")).upper()
    h1_struct = str(ctx.get("H1", "N/A")).upper()
    m30_stage = stages.get("M30") or {}
    m30_status = str(
        m30_stage.get("status")
        or ("CONFIRMED" if m30_stage.get("ok") else "UNCONFIRMED")
    ).upper()
    m5_status = "CONFIRMED" if status == "VALID" else ("INVALID" if status == "INVALID" else "WAIT")

    if ctx.get("has_conflict"):
        htf_status = "CONFLICT"
    elif h4_bias in ("BULLISH", "BEARISH") and h4_bias == h1_struct:
        htf_status = "ALIGNED"
    elif h4_bias in ("NETRAL", "NEUTRAL"):
        htf_status = "NEUTRAL"
    else:
        htf_status = "UNCONFIRMED"

    raw_radar = scan.get("radar_zones") or scan.get("active_zones") or []
    all_radar = [z.to_dict() if hasattr(z, "to_dict") else dict(z) for z in raw_radar]

    atr_diag_pres = dec.get("atr_diagnostics") or scan.get("atr_diagnostics") or {}
    _pres_atr = atr_diag_pres.get("atr_value") if isinstance(atr_diag_pres, dict) and atr_diag_pres.get("atr_value") else 2.0

    r_prio = dec.get("radar_priority") or scan.get("radar_priority")
    if not r_prio and all_radar:
        r_prio = prioritize_radar_zones(
            all_radar,
            price,
            htf_bias=h4_bias,
            h1_structure=h1_struct,
            atr=_pres_atr,
            m5_stages=stages,
            bid=scan.get("bid"),
            ask=scan.get("ask"),
            snapshot_time=scan.get("snapshot_time"),
            snapshot_id=scan.get("snapshot_id"),
        )
    r_prio = r_prio or {}

    zone_fallback = dec.get("zone")
    if zone_fallback and hasattr(zone_fallback, "to_dict"):
        zone_fallback = zone_fallback.to_dict()
    if isinstance(zone_fallback, dict) and is_far_reference_zone(zone_fallback, current_price=price if price > 0 else None, atr=_pres_atr):
        zone_fallback = None

    pz = None
    for pz_cand in (
        dec.get("primary_zone"),
        r_prio.get("primary_directional_zone"),
        r_prio.get("primary_zone"),
        zone_fallback,
    ):
        if isinstance(pz_cand, dict) and not is_far_reference_zone(pz_cand, current_price=price if price > 0 else None, atr=_pres_atr):
            pz = pz_cand
            break
    cz = r_prio.get("countertrend_zone")
    sz = r_prio.get("secondary_zone")
    az = sz if sz is not None else r_prio.get("alternative_zone")
    if az and cz and az.get("price_low") == cz.get("price_low") and az.get("price_high") == cz.get("price_high") and az.get("zone_type") == cz.get("zone_type"):
        az = None

    def _short_zone_line(z_item: Optional[Dict[str, Any]]) -> str:
        if not z_item:
            return "NONE"
        tf = z_item.get("timeframe", "N/A")
        zt_name = z_item.get("zone_type", "N/A")
        lo = float(z_item.get("price_low", 0.0) or 0.0)
        hi = float(z_item.get("price_high", 0.0) or 0.0)
        return f"{tf} {zt_name} {lo:.2f} - {hi:.2f}"

    # Entry Validation states (gunakan hasil stage scanner secara langsung)
    zt_stage = stages.get("zone_touch") or {}
    zt = _norm_stage_label(zt_stage.get("status"), "CONFIRMED" if status == "VALID" else "WAIT")
    zt_ok = bool(zt_stage.get("ok")) or zt in ("CONFIRMED", "PASS", "PASSED", "TOUCHED", "INSIDE")

    if not zt_ok and status != "VALID":
        sw_raw = (stages.get("liquidity_sweep") or {}).get("status")
        tr_raw = (stages.get("structure_trigger") or {}).get("status")
        dp_raw = (stages.get("displacement") or {}).get("status")
        rr_raw = (stages.get("risk_reward") or {}).get("status")
        sw = _norm_stage_label(sw_raw if sw_raw not in (None, "WAIT") else "NOT CHECKED")
        tr = _norm_stage_label(tr_raw if tr_raw not in (None, "WAIT") else "NOT CHECKED")
        dp = _norm_stage_label(dp_raw if dp_raw not in (None, "WAIT") else "NOT CHECKED")
        rr_val_str = _norm_stage_label(rr_raw if rr_raw not in (None, "WAIT") else "NOT CHECKED")
    else:
        sw = _norm_stage_label((stages.get("liquidity_sweep") or {}).get("status"), "CONFIRMED" if status == "VALID" else "WAIT")
        tr = _norm_stage_label((stages.get("structure_trigger") or {}).get("status"), "CONFIRMED" if status == "VALID" else "WAIT")
        dp = _norm_stage_label((stages.get("displacement") or {}).get("status"), "CONFIRMED" if status == "VALID" else "WAIT")
        rr_stage = stages.get("risk_reward") or {}
        rr_st = _norm_stage_label(rr_stage.get("status"), "CONFIRMED" if status == "VALID" else "NOT CHECKED")
        rr_num = rr_stage.get("rr")
        if rr_st == "CONFIRMED" and isinstance(rr_num, (int, float)) and rr_num > 0:
            rr_val_str = f"CONFIRMED (1:{float(rr_num):.2f})"
        else:
            rr_val_str = rr_st

    plan = dec.get("plan") or dec.get("entry_plan")
    if not direction and isinstance(plan, dict):
        direction = plan.get("direction")

    if status == "VALID" and direction:
        sig_str = f"{direction} VALID"
        dir_str = str(direction).upper()
        if isinstance(plan, dict) and "entry_min" in plan and "entry_max" in plan:
            entry_str = (
                f"{float(plan['entry_min']):.2f} - {float(plan['entry_max']):.2f} "
                f"(SL: {float(plan.get('sl', 0.0)):.2f} | TP1: {float(plan.get('tp1', 0.0)):.2f} | "
                f"RR: 1:{float(plan.get('rr', 0.0)):.2f})"
            )
        else:
            entry_str = "CONFIRMED"
        reason_str = "; ".join(str(r) for r in (dec.get("reasons") or ["M5 CONFIRMED"])[:2])
    else:
        sig_str = status
        dir_str = "NONE"
        entry_str = "NONE"
        wb = (r_prio.get("wait_breakdown") if r_prio else None) or dec.get("wait_breakdown") or {}
        if dec.get("reasons"):
            reason_str = sanitize_wait_reason(str(dec["reasons"][0]).replace("[SCANNER] Rejection Reason: ", ""))
        elif ctx.get("has_conflict") and m5_status != "CONFIRMED":
            reason_str = "HTF CONFLICT + M5 NOT CONFIRMED"
        elif wb.get("primary_blocker"):
            reason_str = sanitize_wait_reason(str(wb["primary_blocker"]))
        elif dec.get("reasons"):
            reason_str = sanitize_wait_reason(str(dec["reasons"][0]).replace("[SCANNER] Rejection Reason: ", ""))
        else:
            reason_str = "WAIT FOR CONFIRMATION"

    # Radar Summary counts (ambil langsung dari scan / reconciliation)
    eval_sum = scan.get("zone_evaluation_summary") or {}
    recon = (r_prio.get("zone_count_reconciliation") if r_prio else None) or dec.get("zone_count_reconciliation") or {}

    active_zones_cnt = recon.get("radar_qualified_count", eval_sum.get("radar_qualified", len(all_radar)))
    fresh_cnt = eval_sum.get("total_fresh", sum(1 for z in all_radar if z.get("freshness") == "FRESH"))
    invalidated_cnt = eval_sum.get("total_invalidated", 0)
    mitigated_cnt = eval_sum.get("total_mitigated", 0)
    far_excluded_cnt = recon.get("far_zones_count", len(recon.get("far_zones_reasons", [])))

    fresh_info = scan.get("data_freshness_info") or dec.get("data_freshness_info") or {}
    conn_disp = str(scan.get("connection_status") or dec.get("connection_status") or fresh_info.get("connection_status") or "UNKNOWN")
    tick_st_disp = str(fresh_info.get("tick_status") or scan.get("data_freshness") or dec.get("data_freshness") or "N/A")
    candle_st_disp = str(fresh_info.get("candle_status") or scan.get("data_freshness") or dec.get("data_freshness") or "N/A")
    last_tick_disp = str(scan.get("tick_time") or dec.get("snapshot_time") or "N/A")
    lct_map = scan.get("last_candle_times") or dec.get("last_candle_times") or fresh_info.get("last_candle_times") or {}
    last_m5_disp = str(lct_map.get("M5") or "N/A")
    fc_map = scan.get("forming_candles") or dec.get("forming_candles") or {}
    curr_m5_disp = str((fc_map.get("M5") or {}).get("time") or "N/A")
    data_val_disp = str(fresh_info.get("data_validation") or ("PASS" if scan.get("data_freshness") == "FRESH" else "FAIL"))

    out = [
        "=" * 56,
        "             XAUUSD PYTHON SMC SCANNER",
        "=" * 56,
        "",
        f"PRICE       : {price:.2f}" if price > 0 else "PRICE       : N/A",
        f"TIME        : {time_str}",
        "",
        "DATA INTEGRITY",
        "-" * 32,
        f"MT5 CONNECTION   : {conn_disp}",
        f"TICK STATUS      : {tick_st_disp}",
        f"CANDLE STATUS    : {candle_st_disp}",
        f"LAST TICK        : {last_tick_disp}",
        f"LAST CLOSED M5   : {last_m5_disp}",
        f"CURRENT CANDLE   : {curr_m5_disp}",
        f"ANALYSIS TIME    : {time_str}",
        f"DATA VALIDATION  : {data_val_disp}",
        "",
        "MARKET BIAS",
        f"H4          : {h4_bias}",
        f"H1          : {h1_struct}",
        f"M30         : {m30_status}",
        f"M5          : {m5_status}",
        f"HTF STATUS  : {htf_status}",
        "",
        "-" * 56,
        "PRIMARY RADAR ZONE",
        "-" * 56,
    ]

    if pz:
        pz_type = f"{pz.get('timeframe', 'N/A')} {pz.get('zone_type', 'N/A')}".strip()
        pz_lo = float(pz.get("price_low", 0.0) or 0.0)
        pz_hi = float(pz.get("price_high", 0.0) or 0.0)
        pz_grd = pz.get("grade", "N/A")
        pz_sc = pz.get("score")
        grade_str = f"{pz_grd} ({float(pz_sc):.1f})" if isinstance(pz_sc, (int, float)) else str(pz_grd)
        pz_fresh = pz.get("freshness", "N/A")
        _u_pz = extract_unit_spec_from_container(scan)
        if price > 0 and pz_lo > 0 and pz_hi >= pz_lo:
            _pz_d = calculate_zone_distance(
                pz_lo, pz_hi, price,
                digits=_u_pz["digits"],
                zone_type=pz.get("zone_type"), bid=scan.get("bid"), ask=scan.get("ask"),
                atr=pz.get("atr_value"),
                spec=_u_pz,
            )
            pz_dist = _pz_d.distance
            pz_app = str(calculate_zone_approach(pz_dist, _pz_d.status, float(pz.get("atr_value") or 2.0)))
            dist_str = format_price_distance(pz_dist, spec=_u_pz, include_mt5_points=True)
            pz_stat = (
                "CONFIRMED"
                if (status == "VALID" and str(pz.get("zone_id") or "") == str((dec.get("zone") or {}).get("zone_id") or ""))
                else ("ZONE_TOUCHED" if _pz_d.status == "INSIDE_ZONE" else ("APPROACHING" if pz_app == "APPROACHING" else ("FAR" if pz_app == "FAR" else "RADAR")))
            )
        else:
            pz_dist = pz.get("distance")
            dist_str = format_price_distance(float(pz_dist), spec=_u_pz, include_mt5_points=True) if isinstance(pz_dist, (int, float)) else "N/A"
            pz_stat = str(pz.get("approach") or pz.get("status") or "RADAR").upper()
        pz_why_sum = build_smc_zone_reason(pz, h4_bias=h4_bias)
        out.extend([
            f"TYPE        : {pz_type}",
            f"ZONE        : {pz_lo:.2f} - {pz_hi:.2f}",
            f"GRADE       : {grade_str}",
            f"FRESHNESS   : {pz_fresh}",
            f"DISTANCE    : {dist_str}",
            f"STATUS      : {pz_stat}",
            *([f"WHY         : {pz_why_sum}"] if pz_why_sum else []),
        ])
    else:
        out.extend([
            "TYPE        : NONE",
            "ZONE        : N/A",
            "GRADE       : N/A",
            "FRESHNESS   : N/A",
            "DISTANCE    : N/A",
            "STATUS      : NONE",
        ])

    out.extend([
        "",
        f"ACTIVE ZONE : {_short_zone_line(dec.get('zone'))}",
        f"NEAREST ZONE: {_short_zone_line(dec.get('nearest_valid_zone'))}",
        f"ALTERNATIVE : {_short_zone_line(az)}",
        f"COUNTERTREND: {_short_zone_line(cz)}",
        "",
        "-" * 56,
        "ENTRY VALIDATION",
        "-" * 56,
        f"ZONE TOUCH      : {zt}",
        f"LIQUIDITY SWEEP : {sw}",
        f"CHoCH / BOS     : {tr}",
        f"DISPLACEMENT    : {dp}",
        f"RR VALIDATION   : {rr_val_str}",
        "",
        "-" * 56,
        "FINAL DECISION",
        "-" * 56,
        f"SIGNAL      : {sig_str}",
        f"SETUP TYPE  : {str(dec.get('setup_type') or (plan.get('setup_type') if isinstance(plan, dict) else None) or ('TREND_PULLBACK' if status == 'VALID' else 'NONE')).upper()}",
        f"LIFECYCLE   : {str(dec.get('signal_status') or (SIGNAL_STATUS_SIGNAL_CONFIRMED if status == 'VALID' else SIGNAL_STATUS_NO_SETUP)).upper()}",
        f"DIRECTION   : {dir_str}",
        f"ENTRY       : {entry_str}",
        f"REASON      : {reason_str}",
        "",
        "-" * 56,
        "RADAR SUMMARY",
        "-" * 56,
        f"ACTIVE ZONES    : {active_zones_cnt}",
        f"FRESH           : {fresh_cnt} (total scanner freshness count)",
        f"INVALIDATED     : {invalidated_cnt}",
        f"MITIGATED       : {mitigated_cnt}",
        f"FAR EXCLUDED    : {far_excluded_cnt}",
        "",
        *(premarket_lines(dec) + [""] if dec.get("premarket") else []),
        format_risk_alert(pos_list=pos_list, orders_list=orders_list, risk_data=risk_data),
        "",
        "=" * 56,
    ])
    return "\n".join(out)


def format_scan_detail(
    scan: Optional[Dict[str, Any]] = None,
    dec: Optional[Dict[str, Any]] = None,
    early_setups: Optional[Dict[str, Any]] = None,
    tracker: Optional[Any] = None,
    pos_list: Optional[Any] = None,
    orders_list: Optional[Any] = None,
    risk_data: Optional[Dict[str, Any]] = None,
    include_mt5_positions: bool = True,
) -> str:
    """
    DETAIL MODE formatter untuk command `scan detail` (Two-Level Output Display).
    Mempertahankan seluruh informasi teknis lengkap dari scanner + rekonsiliasi Smart Radar + detail posisi/pending MT5.
    """
    if scan is None:
        _snap = create_market_snapshot()
        scan = scan_xauusd_zones(snapshot=_snap)
        dec = evaluate_trade_decision(snapshot=_snap, scan_result=scan)
    elif dec is None:
        _snap = scan.get("snapshot") if isinstance(scan.get("snapshot"), dict) else None
        dec = evaluate_trade_decision(snapshot=_snap, scan_result=scan)

    base = format_scan_presentation(
        scan, dec, early_setups=early_setups, tracker=tracker, detailed=True
    )
    sections = [base]

    # Tambahan rekonsiliasi Smart Radar penuh
    try:
        from modules.pantau import collect_radar_watchlist_with_reconciliation
        wl, rec = collect_radar_watchlist_with_reconciliation(scan, dec)
        if rec:
            rec_lines = [
                "",
                "-" * 70,
                "F. SMART RADAR RECONCILIATION & LIVE ACCOUNT RISK DETAIL",
                "-" * 70,
                (
                    f"• Rekonsiliasi Smart Radar : Scanner Qualified={rec.get('scanner_radar_qualified_count', 0)} | "
                    f"Received={rec.get('monitor_received_count', 0)} | Active={rec.get('monitor_active_count', 0)} | "
                    f"Excluded={rec.get('excluded_count', 0)} | Status={rec.get('transfer_status', 'OK')}"
                ),
            ]
            for ex in rec.get("excluded_details", [])[:10]:
                rec_lines.append(
                    f"  - Excluded [{ex.get('zone_id')}]: {ex.get('timeframe')} {ex.get('zone_type')} "
                    f"({ex.get('initial_status')} -> {ex.get('final_status')}) | Reason: {ex.get('reason')}"
                )
            sections.append("\n".join(rec_lines))
    except Exception:
        pass

    # Risk Monitor & Detail Posisi Running / Pending Order MT5
    if include_mt5_positions and pos_list is None and orders_list is None and risk_data is None:
        try:
            from modules.mt5_conn import cek_posisi_terbuka_mt5
            import modules.mt5_conn as _mt5_conn
            if getattr(_mt5_conn, "TERHUBUNG_MT5", False):
                mt5_info = cek_posisi_terbuka_mt5(SYMBOL)
                if mt5_info:
                    sections.append(str(mt5_info).strip())
        except Exception:
            pass

    sections.append(format_risk_alert(pos_list=pos_list, orders_list=orders_list, risk_data=risk_data))
    sections.append("")
    sections.append(format_scanner_audit(dec))
    sections.append("=" * 70)
    return "\n".join(sections)
