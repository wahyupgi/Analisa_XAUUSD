# -*- coding: utf-8 -*-
"""
monitor.py — Mesin Pemantauan (Monitoring Engine) Independen untuk SNR Advanced Scanner
(`pantau snr`, `pantau snr detail`, serta komponen SNR pada Unified Monitor).
"""
import time
from modules.config import SYMBOL as DEFAULT_APP_SYMBOL
from modules.snr.config import get_snr_config
from modules.snr.data_provider import log_snr_error, safe_float
from modules.snr.formatter import (
    build_snr_zone_reason,
    format_snr_detail,
    format_snr_focus_plan,
    format_snr_invalid_why,
    format_snr_lifecycle_status,
    format_snr_summary,
)
from modules.snr.radar import SnrAlertTracker
from modules.snr.scanner import evaluate_snr_decision, run_snr_scan
from modules.terminal_ui import print_terminal


def format_snr_monitor_section(cycle_result, detailed=False):
    """
    Menyusun blok informasi SNR Monitor (baik untuk `pantau snr` maupun bagian
    `### B. SNR MONITOR` pada Unified Monitor).
    Memuat:
    - Market bias H4, H1, M30, M5
    - Support dan resistance aktif
    - Kekuatan dan grade zona
    - Jumlah serta kualitas sentuhan zona
    - Status rejection, breakout, dan retest
    - Status konfirmasi entry
    - Keputusan SNR: BUY / SELL / WAIT / INVALID
    - Entry, SL, TP, dan RR jika sudah valid
    - Alasan utama keputusan
    """
    if not isinstance(cycle_result, dict):
        return "### B. SNR MONITOR\nSTATUS : ERROR (Hasil siklus SNR tidak valid)"

    if cycle_result.get("status") == "ERROR":
        err_msg = cycle_result.get("error", "Unknown SNR error")
        return (
            "### B. SNR MONITOR\n"
            "STATUS          : ERROR\n"
            f"ERROR REASON    : {err_msg}"
        )

    sr = cycle_result.get("scan_result") or {}
    dec = cycle_result.get("decision") or sr.get("decision") or {}
    mb = sr.get("market_bias") or {}
    ranked = sr.get("ranked_zones") or {}
    pa = sr.get("price_action") or {}
    ev = sr.get("entry_validation") or {}
    alerts = cycle_result.get("alerts") or []
    crit = cycle_result.get("critical_warnings") or []

    active_sups = ranked.get("active_supports") or []
    active_ress = ranked.get("active_resistances") or []
    mkt_status = str(sr.get("market_status", dec.get("market_status", "OPEN"))).upper()
    data_source = str(sr.get("data_source", "MT5 HISTORICAL" if mkt_status == "CLOSED" else ("DATA UNAVAILABLE" if mkt_status == "DATA UNAVAILABLE" else "MT5 LIVE")))
    snr_analysis = str(sr.get("snr_analysis", dec.get("snr_analysis", "DATA UNAVAILABLE" if mkt_status == "DATA UNAVAILABLE" else "AVAILABLE")))
    live_entry = str(dec.get("live_entry", sr.get("live_entry", "WAIT — MARKET CLOSED" if mkt_status == "CLOSED" else ("WAIT — DATA UNAVAILABLE" if mkt_status == "DATA UNAVAILABLE" else dec.get("signal", "WAIT")))))
    last_price = safe_float(sr.get("last_price", sr.get("price", 0.0)), 0.0)
    last_price_str = f"{last_price:.2f}" if (last_price > 0 and mkt_status != "DATA UNAVAILABLE") else "DATA UNAVAILABLE"
    last_candle = str(sr.get("last_candle", "DATA UNAVAILABLE" if mkt_status == "DATA UNAVAILABLE" else "N/A"))

    if not detailed:
        from modules.snr.breakout_detection import build_snr_breakout_watch, classify_snr_breakout_state
        from modules.utils import (
            extract_unit_spec_from_container,
            format_invalid_zone_compact,
            format_price_distance,
            resolve_entry_blocker_code,
            consume_one_shot_invalid_zone,
        )
        from modules.zone_scanner import resolve_analysis_direction
        _u_mon = extract_unit_spec_from_container(sr)
        data_ok_m = mkt_status == "OPEN" and last_price > 0
        dir_m = resolve_analysis_direction(decision=dec, snr_bias=mb, market_status=mkt_status, data_ok=data_ok_m)
        sig_m = str(dec.get("signal", "WAIT")).upper()
        tz_m = dec.get("target_zone") or ev.get("target_zone") or (active_sups[0] if active_sups else (active_ress[0] if active_ress else None))
        if mkt_status == "OPEN" and sig_m in ("BUY", "SELL"):
            e_m, sl_m, tp_m = safe_float(dec.get("effective_entry") or dec.get("entry")), safe_float(dec.get("sl")), safe_float(dec.get("tp1"))
            ent_m = f"{sig_m} VALID | E {e_m:.2f} | SL {sl_m:.2f} | TP1 {tp_m:.2f}" if (e_m > 0 and sl_m > 0 and tp_m > 0) else f"{sig_m} VALID"
        else:
            snr_m_conflict = str(mb.get("overall_bias") or "").upper() == "CONFLICT" or (
                str(mb.get("h4") or "").upper() in ("BULLISH", "BEARISH")
                and str(mb.get("h1") or "").upper() in ("BULLISH", "BEARISH")
                and str(mb.get("h4") or "").upper() != str(mb.get("h1") or "").upper()
            )
            blk_m_snr = resolve_entry_blocker_code(
                dec.get("reason") or ev.get("reason"),
                market_status=mkt_status,
                data_ok=data_ok_m,
                htf_conflict=snr_m_conflict,
                h4_bias=mb.get("h4"),
                has_active_zone=isinstance(tz_m, dict),
                rr_failed=bool((ev.get("rr_calculation") or {}).get("rr_failed")),
            )
            ent_m = f"NO VALID ENTRY ({blk_m_snr})"
        s1_m = f"{active_sups[0].get('rank_label', 'S1')} {safe_float(active_sups[0].get('low')):.2f}-{safe_float(active_sups[0].get('high')):.2f}(G:{active_sups[0].get('grade', '-')})" if active_sups else "S1 NONE"
        r1_m = f"{active_ress[0].get('rank_label', 'R1')} {safe_float(active_ress[0].get('low')):.2f}-{safe_float(active_ress[0].get('high')):.2f}(G:{active_ress[0].get('grade', '-')})" if active_ress else "R1 NONE"
        h4_m = mb.get("h4")
        s1_why_m = build_snr_zone_reason(active_sups[0], h4_bias=h4_m) if active_sups else ""
        r1_why_m = build_snr_zone_reason(active_ress[0], h4_bias=h4_m) if active_ress else ""
        bo_w_m = sr.get("breakout_watch") or build_snr_breakout_watch(ranked, price_action=pa)
        inv_pool_m = (
            list(bo_w_m.get("retest_failed") or [])
            + list(bo_w_m.get("breakout_invalidated") or [])
            + list(bo_w_m.get("confirmed") or [])
            + list(bo_w_m.get("broken") or [])
            + list(bo_w_m.get("historical") or [])
            + list(ranked.get("broken_zones") or [])
            + list(ranked.get("reclaimed_zones") or [])
            + list(ranked.get("invalidated_zones") or [])
        )
        h1_atr_m = safe_float(((sr.get("atr_diagnostics") or {}).get("H1") or {}).get("atr14"), 0.0) or None
        iz_m = consume_one_shot_invalid_zone(inv_pool_m, current_price=last_price if last_price > 0 else None, atr=h1_atr_m, cycle_key=f"snr:{last_price_str}")
        if isinstance(iz_m, dict):
            iz_m_z = iz_m.get("zone") if isinstance(iz_m.get("zone"), dict) else iz_m
            iz_m_st = str(iz_m.get("state") or classify_snr_breakout_state(iz_m_z).get("state") or "BROKEN").upper()
            inv_m = format_invalid_zone_compact(iz_m, current_price=last_price if last_price > 0 else None, unit_spec=_u_mon, status_override=iz_m_st, include_id=False)
        else:
            inv_m = "NONE"
        if isinstance(tz_m, dict):
            tz_lo = safe_float(tz_m.get("low"), 0.0)
            tz_hi = safe_float(tz_m.get("high"), tz_lo)
            tz_zt = str(tz_m.get("zone_type") or "").upper()
            tz_sc = f"/{safe_float(tz_m.get('score')):.1f}" if tz_m.get("score") is not None else ""
            tz_d = 0.0 if (last_price > 0 and tz_lo <= last_price <= tz_hi) else (min(abs(last_price - tz_lo), abs(last_price - tz_hi)) if last_price > 0 else safe_float(tz_m.get("distance_points"), 0.0))
            tz_d_txt = "IN ZONE" if tz_d <= 0 else format_price_distance(tz_d, spec=_u_mon)
            tz_inv = safe_float(tz_m.get("invalidation_price"), 0.0) or (tz_lo if tz_zt == "SUPPORT" else (tz_hi if tz_zt == "RESISTANCE" else 0.0))
            tz_inv_t = f" | Inv {tz_inv:.2f}" if tz_inv > 0 else ""
            act_m = f"{tz_m.get('rank_label', 'ZONE')} {tz_m.get('timeframe', '')} {tz_lo:.2f}-{tz_hi:.2f} | G:{tz_m.get('grade', '-')}{tz_sc} | Dist {tz_d_txt}{tz_inv_t} [ID:{tz_m.get('zone_id', 'N/A')}]".replace("  ", " ")
        else:
            act_m = "NONE"
        act_why_m = build_snr_zone_reason(tz_m, h4_bias=h4_m) if isinstance(tz_m, dict) else ""
        risk_m = crit[0] if crit else "AMAN"
        lines = [
            "### B. SNR MONITOR",
            f"MARKET       : {mkt_status} | {last_price_str}",
            f"DIRECTION    : {dir_m} (H4={mb.get('h4', 'N/A')} | H1(SNR Trend)={mb.get('h1', 'N/A')} | M30={mb.get('m30', 'N/A')})",
            f"ENTRY STATUS : {ent_m}",
            f"ACTIVE/RADAR : {act_m}",
            *([f"  WHY        : {act_why_m}"] if act_why_m else []),
            *([f"INVALID      : {inv_m}"] if inv_m != "NONE" else []),
            f"SNR          : {s1_m} | {r1_m}",
            *([f"  WHY S1     : {s1_why_m}"] if s1_why_m else []),
            *([f"  WHY R1     : {r1_why_m}"] if r1_why_m else []),
            f"RISK         : {risk_m}",
        ]
        return "\n".join(lines)

    lines = [
        "### B. SNR MONITOR",
        f"STATUS          : OK ({sr.get('snr_status', mb.get('snr_status', 'ACTIVE'))})",
        f"MARKET STATUS   : {mkt_status} | DATA SOURCE: {data_source} | LAST PRICE: {last_price_str} | LAST CANDLE: {last_candle}",
        f"SNR ANALYSIS    : {snr_analysis} | LIVE ENTRY: {live_entry}",
        f"MARKET BIAS     : H4={mb.get('h4', 'N/A')} | H1(SNR Trend)={mb.get('h1', 'N/A')} | M30={mb.get('m30', 'N/A')} | M5={mb.get('m5', 'WAIT')} -> {mb.get('overall_bias', 'UNCONFIRMED')}",
    ]

    lines.extend(format_snr_lifecycle_status(sr, dec))

    from modules.utils import extract_unit_spec_from_container, format_price_distance
    u_spec = extract_unit_spec_from_container(sr)

    if active_sups:
        lines.append("ACTIVE SUPPORT  :")
        for sz in active_sups[:2]:
            lbl = sz.get("rank_label", "S")
            ta = sz.get("touch_analysis") or {}
            t_cls = ta.get("classification", sz.get("touch_status", "UNTESTED"))
            t_cnt = int(ta.get("valid_touch_count", sz.get("touch_count", 0)))
            t_q = safe_float(ta.get("touch_quality_score_15"), 0.0)
            dist = safe_float(sz.get("distance_points"), 0.0)
            lines.append(
                f"  • {lbl} : {safe_float(sz.get('low')):.2f} - {safe_float(sz.get('high')):.2f} "
                f"| Grade {sz.get('grade', 'D')} (Score {safe_float(sz.get('score')):.2f} - {sz.get('strength_short', 'MODERATE')}) "
                f"| Touch: {t_cls} ({t_cnt}x, Q={t_q:.1f}/15) | Dist: {format_price_distance(dist, spec=u_spec, include_mt5_points=True)}"
            )
            sz_why = build_snr_zone_reason(sz, h4_bias=mb.get("h4"))
            if sz_why:
                lines.append(f"    WHY: {sz_why}")
    else:
        lines.append("ACTIVE SUPPORT  : N/A (Tidak ada support aktif)")

    if active_ress:
        lines.append("ACTIVE RESIST.  :")
        for rz in active_ress[:2]:
            lbl = rz.get("rank_label", "R")
            ta = rz.get("touch_analysis") or {}
            t_cls = ta.get("classification", rz.get("touch_status", "UNTESTED"))
            t_cnt = int(ta.get("valid_touch_count", rz.get("touch_count", 0)))
            t_q = safe_float(ta.get("touch_quality_score_15"), 0.0)
            dist = safe_float(rz.get("distance_points"), 0.0)
            lines.append(
                f"  • {lbl} : {safe_float(rz.get('low')):.2f} - {safe_float(rz.get('high')):.2f} "
                f"| Grade {rz.get('grade', 'D')} (Score {safe_float(rz.get('score')):.2f} - {rz.get('strength_short', 'MODERATE')}) "
                f"| Touch: {t_cls} ({t_cnt}x, Q={t_q:.1f}/15) | Dist: {format_price_distance(dist, spec=u_spec, include_mt5_points=True)}"
            )
            rz_why = build_snr_zone_reason(rz, h4_bias=mb.get("h4"))
            if rz_why:
                lines.append(f"    WHY: {rz_why}")
    else:
        lines.append("ACTIVE RESIST.  : N/A (Tidak ada resistance aktif)")

    mc = sr.get("market_context") or {}
    pa_bo_disp = pa.get("breakout_summary", "NOT CONFIRMED")
    if pa.get("breakout_zone_label") and pa.get("focus_zone_label") and pa.get("breakout_zone_label") != pa.get("focus_zone_label") and pa.get("breakout_code") != "NO_BREAKOUT":
        pa_bo_disp = f"{pa_bo_disp} [{pa.get('breakout_zone_label')}]"
    pa_rt_disp = pa.get("retest_summary", "NOT CONFIRMED")
    if pa.get("retest_zone_label") and pa.get("focus_zone_label") and pa.get("retest_zone_label") != pa.get("focus_zone_label") and pa.get("retest_code") != "NO_RETEST":
        pa_rt_disp = f"{pa_rt_disp} [{pa.get('retest_zone_label')}]"

    # Informational trend-aligned monitoring zone; independent SNR decision logic remains unchanged.
    overall = str(mb.get("overall_bias", "")).upper()
    aligned_zone = active_ress[0] if overall == "BEARISH" and active_ress else (active_sups[0] if overall == "BULLISH" and active_sups else None)
    aligned_label = (f"RESISTANCE {aligned_zone.get('rank_label', 'R1')}" if overall == "BEARISH" else f"SUPPORT {aligned_zone.get('rank_label', 'S1')}") if aligned_zone else "NONE"
    lines.append(f"TREND ENTRY ZONE: {aligned_label} (referensi arah tren, bukan sinyal entry)")

    lines.extend([
        f"PRICE ACTION    : Rejection={pa.get('rejection_summary', 'NOT CONFIRMED')} | Breakout={pa_bo_disp} | Retest={pa_rt_disp} | Momentum={pa.get('momentum_summary', 'N/A')}",
        f"ENTRY CONFIRM.  : Focus={ev.get('zone_label', 'N/A')} | Trigger={ev.get('trigger_status', 'NOT CONFIRMED')} | BUY={ev.get('buy_status', 'WAIT')} | SELL={ev.get('sell_status', 'WAIT')}",
        f"SNR CONTEXT     : {ev.get('snr_role_context') or mc.get('snr_context') or ev.get('zone_label', 'N/A')}",
        f"SNR DECISION    : {dec.get('signal', 'WAIT')} (Direction: {dec.get('direction', 'NONE')})",
    ])

    lines.append(format_snr_focus_plan(dec, ev, market_bias=mb))

    lines.append(f"MAIN REASON     : {dec.get('reason', 'N/A')}")

    if alerts:
        lines.append("SNR ALERTS      :")
        for al in alerts:
            lines.append(f"  ! {al}")
    if crit:
        for cw in crit:
            lines.append(f"  !! {cw}")

    if detailed:
        lines.append("")
        lines.append(format_snr_detail(sr, dec))

    return "\n".join(lines)


class SnrCycleEngine:
    """
    Engine siklus pemantauan SNR yang menyimpan state antar-siklus,
    mendeteksi perubahan status, dan mengisolasi error agar aman dijalankan terus-menerus.
    """

    def __init__(self, symbol=None, config=None):
        self.cfg = get_snr_config(config)
        self.symbol = symbol or self.cfg.get("symbol", DEFAULT_APP_SYMBOL)
        self.alert_tracker = SnrAlertTracker()
        self.cycle_count = 0
        self.last_signature = None

    def execute_cycle(
        self,
        snapshot=None,
        frames=None,
        tick=None,
        live=True,
        detailed=False,
        market_status=None,
    ):
        """Menjalankan satu siklus evaluasi SNR Monitor dengan isolasi error penuh."""
        self.cycle_count += 1
        try:
            scan_res = run_snr_scan(
                symbol=self.symbol,
                snapshot=snapshot,
                frames=frames,
                tick=tick,
                live=live,
                config=self.cfg,
                market_status=market_status,
            )
            dec = evaluate_snr_decision(scan_res)
            radar = scan_res.get("radar") or {}
            pa = scan_res.get("price_action") or {}
            alert_bundle = self.alert_tracker.evaluate_alerts(radar, dec, price_action=pa)

            bo_w_sig = scan_res.get("breakout_watch") if isinstance(scan_res.get("breakout_watch"), dict) else {}
            sig_tuple = (
                dec.get("signal"),
                dec.get("direction"),
                dec.get("zone_label"),
                dec.get("trigger_status"),
                (scan_res.get("market_bias") or {}).get("overall_bias"),
                radar.get("radar_status"),
                bo_w_sig.get("state"),
                bo_w_sig.get("zone_id"),
            )
            state_changed = (self.last_signature is None) or (sig_tuple != self.last_signature) or bool(alert_bundle["alerts"])
            self.last_signature = sig_tuple

            cycle_payload = {
                "method": "SNR",
                "status": "OK",
                "cycle": self.cycle_count,
                "scan_result": scan_res,
                "decision": dec,
                "radar": radar,
                "alerts": alert_bundle["alerts"],
                "historical_events": alert_bundle.get("historical_events") or [],
                "critical_warnings": alert_bundle["critical_warnings"],
                "state_changed": state_changed,
            }
            cycle_payload["formatted_section"] = format_snr_monitor_section(cycle_payload, detailed=detailed)
            cycle_payload["formatted_scanner"] = (
                format_snr_detail(scan_res, dec) if detailed else format_snr_summary(scan_res, dec)
            )
            return cycle_payload
        except Exception as exc:
            log_snr_error("SnrCycleEngine.execute_cycle", exc)
            err_payload = {
                "method": "SNR",
                "status": "ERROR",
                "cycle": self.cycle_count,
                "error": f"{type(exc).__name__}: {exc}",
                "scan_result": None,
                "decision": {
                    "method": "SNR",
                    "signal": "ERROR",
                    "direction": "NONE",
                    "reason": f"SNR Scanner Error: {exc}",
                },
                "radar": {},
                "alerts": [f"[SNR] ERROR: {exc}"],
                "critical_warnings": [],
                "state_changed": True,
            }
            err_payload["formatted_section"] = format_snr_monitor_section(err_payload, detailed=detailed)
            err_payload["formatted_scanner"] = err_payload["formatted_section"]
            return err_payload


def mode_pantau_snr(cmd_str="pantau snr", max_cycles=None, sleep_sec=2.0, snapshot=None, frames=None, live=True):
    """
    Menjalankan mode pemantauan khusus SNR (`pantau snr` atau `pantau snr detail`).
    Hanya mencetak laporan penuh pada siklus pertama atau ketika terjadi perubahan state/alert.
    """
    cmd_clean = str(cmd_str or "pantau snr").strip().lower()
    detailed = "detail" in cmd_clean
    engine = SnrCycleEngine(symbol=DEFAULT_APP_SYMBOL)

    print_terminal("\n" + "=" * 68)
    print_terminal(f"📡 MEMULAI SNR ADVANCED MONITOR ({'DETAIL MODE' if detailed else 'SUMMARY MODE'})")
    print_terminal("   Tekan Ctrl+C untuk menghentikan pemantauan.")
    print_terminal("=" * 68)

    cycles_done = 0
    last_result = None
    try:
        while True:
            res = engine.execute_cycle(snapshot=snapshot, frames=frames, live=live, detailed=detailed)
            last_result = res
            cycles_done += 1

            if cycles_done == 1 or res.get("state_changed"):
                print_terminal("\n" + res["formatted_section"])
            elif res.get("critical_warnings"):
                for cw in res["critical_warnings"]:
                    print_terminal(cw)

            if max_cycles is not None and cycles_done >= int(max_cycles):
                break
            time.sleep(max(0.1, float(sleep_sec)))
    except KeyboardInterrupt:
        print_terminal("\n[SNR MONITOR] Pemantauan SNR dihentikan oleh pengguna.")

    return last_result

