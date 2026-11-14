# -*- coding: utf-8 -*-
"""
monitor.py — Mesin Pemantauan (Monitoring Engine) Independen untuk SNR Advanced Scanner
(`pantau snr`, `pantau snr detail`, serta komponen SNR pada Unified Monitor).
"""
import time
from modules.config import SYMBOL as DEFAULT_APP_SYMBOL
from modules.snr.config import get_snr_config
from modules.snr.data_provider import log_snr_error, safe_float
from modules.snr.formatter import format_snr_detail, format_snr_summary
from modules.snr.radar import SnrAlertTracker
from modules.snr.scanner import evaluate_snr_decision, run_snr_scan


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

    lines = [
        "### B. SNR MONITOR",
        f"STATUS          : OK ({sr.get('snr_status', mb.get('snr_status', 'ACTIVE'))})",
        f"MARKET BIAS     : H4={mb.get('h4', 'N/A')} | H1={mb.get('h1', 'N/A')} | M30={mb.get('m30', 'N/A')} | M5={mb.get('m5', 'WAIT')} -> {mb.get('overall_bias', 'UNCONFIRMED')}",
    ]

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
                f"| Grade {sz.get('grade', 'D')} (Score {int(round(safe_float(sz.get('score'))))} - {sz.get('strength_short', 'MODERATE')}) "
                f"| Touch: {t_cls} ({t_cnt}x, Q={t_q:.1f}/15) | Dist: {dist:.2f} pts"
            )
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
                f"| Grade {rz.get('grade', 'D')} (Score {int(round(safe_float(rz.get('score'))))} - {rz.get('strength_short', 'MODERATE')}) "
                f"| Touch: {t_cls} ({t_cnt}x, Q={t_q:.1f}/15) | Dist: {dist:.2f} pts"
            )
    else:
        lines.append("ACTIVE RESIST.  : N/A (Tidak ada resistance aktif)")

    lines.extend([
        f"PRICE ACTION    : Rejection={pa.get('rejection_summary', 'WAIT')} | Breakout={pa.get('breakout_summary', 'NOT CONFIRMED')} | Retest={pa.get('retest_summary', 'NOT CONFIRMED')} | Momentum={pa.get('momentum_summary', 'N/A')}",
        f"ENTRY CONFIRM.  : Focus={ev.get('zone_label', 'N/A')} | Trigger={ev.get('trigger_status', 'NOT CONFIRMED')} | BUY={ev.get('buy_status', 'WAIT')} | SELL={ev.get('sell_status', 'WAIT')}",
        f"SNR DECISION    : {dec.get('signal', 'WAIT')} (Direction: {dec.get('direction', 'NONE')})",
    ])

    if dec.get("signal") in ("BUY", "SELL") and dec.get("entry") is not None:
        lines.append(
            f"ENTRY / SL / TP : Entry {safe_float(dec.get('entry')):.2f} | SL {safe_float(dec.get('sl')):.2f} "
            f"| TP1 {safe_float(dec.get('tp1')):.2f} | TP2 {safe_float(dec.get('tp2', dec.get('tp1'))):.2f} "
            f"| RR {dec.get('rr_display', 'N/A')}"
        )
    else:
        lines.append(f"ENTRY / SL / TP : N/A (RR: {dec.get('rr_display', 'NOT CHECKED')})")

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
            )
            dec = evaluate_snr_decision(scan_res)
            radar = scan_res.get("radar") or {}
            pa = scan_res.get("price_action") or {}
            alert_bundle = self.alert_tracker.evaluate_alerts(radar, dec, price_action=pa)

            sig_tuple = (
                dec.get("signal"),
                dec.get("direction"),
                dec.get("zone_label"),
                dec.get("trigger_status"),
                (scan_res.get("market_bias") or {}).get("overall_bias"),
                radar.get("radar_status"),
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

    print("\n" + "=" * 68)
    print(f"📡 MEMULAI SNR ADVANCED MONITOR ({'DETAIL MODE' if detailed else 'SUMMARY MODE'})")
    print("   Tekan Ctrl+C untuk menghentikan pemantauan.")
    print("=" * 68)

    cycles_done = 0
    last_result = None
    try:
        while True:
            res = engine.execute_cycle(snapshot=snapshot, frames=frames, live=live, detailed=detailed)
            last_result = res
            cycles_done += 1

            if cycles_done == 1 or res.get("state_changed"):
                print("\n" + res["formatted_section"])
            elif res.get("critical_warnings"):
                for cw in res["critical_warnings"]:
                    print(cw)

            if max_cycles is not None and cycles_done >= int(max_cycles):
                break
            time.sleep(max(0.1, float(sleep_sec)))
    except KeyboardInterrupt:
        print("\n[SNR MONITOR] Pemantauan SNR dihentikan oleh pengguna.")

    return last_result

