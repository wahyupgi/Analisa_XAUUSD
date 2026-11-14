# -*- coding: utf-8 -*-
"""
unified_monitor.py — Mode Pantau Terpadu (Unified Monitor: SMC + SNR + MT5 Account Monitor).

Arsitektur:
Unified Monitor
├── SMC Scanner (SMC Market Analysis, SMC Zone Monitoring, SMC Entry Validation, SMC Signal Decision)
├── SNR Scanner (SNR Market Analysis, SNR Zone Monitoring, SNR Entry Validation, SNR Signal Decision)
└── MT5 Account Monitor (Open Positions, Pending Orders, Spread, Risk Alerts — Read-Only)

Aturan Utama:
- Jangan membandingkan SMC dengan SNR, jangan membuat ranking metode terbaik, jangan menggabungkan skor.
- Jangan membuat satu metode menjadi filter/syarat validasi metode lainnya.
- Jangan memaksa kedua metode menghasilkan arah sinyal yang sama; tampilkan apa adanya tanpa label konflik.
- Gunakan 1 snapshot data MT5 yang konsisten untuk setiap siklus monitoring.
- Isolasi error: kegagalan SMC tidak menghentikan SNR, dan kegagalan SNR tidak menghentikan SMC.
- Alert independen beridentitas metode ([SMC] ... dan [SNR] ...) hanya muncul saat terjadi perubahan status.
"""
from datetime import datetime
import re
import time

try:
    import MetaTrader5 as mt5
except Exception:  # pragma: no cover
    mt5 = None

from modules.config import SYMBOL as DEFAULT_APP_SYMBOL
from modules.mt5_conn import pastikan_mt5_aktif
from modules.pantau import mode_pantau_mt5
from modules.snr.data_provider import safe_float
from modules.snr.monitor import (
    SnrCycleEngine,
    format_snr_monitor_section,
    mode_pantau_snr,
)
from modules.zone_scanner import (
    _extract_risk_monitor_snapshot,
    create_market_snapshot,
    evaluate_trade_decision,
    format_scan_presentation,
    scan_xauusd_zones,
)


class SmcAlertTracker:
    """
    Pelacak perubahan status (State-Change Detector) khusus SMC Monitor di dalam Unified Monitor.
    Menghasilkan alert beridentitas `[SMC]` hanya ketika terjadi perubahan status penting.
    """

    def __init__(self):
        self.last_state = {}

    def reset(self):
        self.last_state = {}

    def evaluate_alerts(self, smc_summary):
        if not isinstance(smc_summary, dict) or smc_summary.get("status") == "ERROR":
            return {"alerts": [], "critical_warnings": []}

        curr_state = {
            "signal": str(smc_summary.get("decision_signal", "WAIT")).upper(),
            "direction": str(smc_summary.get("direction", "NONE")).upper(),
            "primary_zone": str(smc_summary.get("primary_zone_label", "NONE")),
            "zone_touch": str(smc_summary.get("zone_touch", "WAIT")).upper(),
            "sweep": str(smc_summary.get("liquidity_sweep", "WAIT")).upper(),
            "choch_bos": str(smc_summary.get("choch_bos", "WAIT")).upper(),
            "displacement": str(smc_summary.get("displacement", "WAIT")).upper(),
            "invalidated_count": int(smc_summary.get("invalidated_count", 0)),
            "mitigated_count": int(smc_summary.get("mitigated_count", 0)),
        }

        alerts = []
        critical_warnings = []

        # 1. Perubahan keputusan setup SMC
        if curr_state["signal"] != self.last_state.get("signal"):
            if curr_state["signal"] == "BUY":
                alerts.append("[SMC] BUY SETUP TERKONFIRMASI")
            elif curr_state["signal"] == "SELL":
                alerts.append("[SMC] SELL SETUP TERKONFIRMASI")
            elif curr_state["signal"] == "INVALID":
                alerts.append("[SMC] SETUP INVALID")

        # 2. Perubahan tahapan konfirmasi M5 SMC
        if curr_state["zone_touch"] != self.last_state.get("zone_touch") and curr_state["zone_touch"] == "CONFIRMED":
            alerts.append("[SMC] ZONE TOUCH TERKONFIRMASI")

        if curr_state["sweep"] != self.last_state.get("sweep") and curr_state["sweep"] == "CONFIRMED":
            alerts.append("[SMC] LIQUIDITY SWEEP TERKONFIRMASI")

        if curr_state["choch_bos"] != self.last_state.get("choch_bos") and curr_state["choch_bos"] == "CONFIRMED":
            alerts.append("[SMC] CHoCH / BOS M5 TERKONFIRMASI")

        if curr_state["displacement"] != self.last_state.get("displacement") and curr_state["displacement"] == "CONFIRMED":
            alerts.append("[SMC] DISPLACEMENT TERKONFIRMASI")

        # 3. Perubahan zona aktif / invalidasi zona SMC
        prev_inval = self.last_state.get("invalidated_count")
        if prev_inval is not None and curr_state["invalidated_count"] > prev_inval:
            last_inval_type = str(smc_summary.get("last_invalidated_type", "ZONA")).upper()
            alerts.append(f"[SMC] ZONA {last_inval_type} INVALID")

        prev_mit = self.last_state.get("mitigated_count")
        if prev_mit is not None and curr_state["mitigated_count"] > prev_mit:
            alerts.append("[SMC] ZONA TERMITIGASI")

        if (
            self.last_state.get("primary_zone") is not None
            and curr_state["primary_zone"] != self.last_state.get("primary_zone")
            and curr_state["primary_zone"] not in ("NONE", "-", "")
        ):
            alerts.append(f"[SMC] PERUBAHAN ZONA FOKUS -> {curr_state['primary_zone']}")

        self.last_state = curr_state
        return {
            "alerts": alerts,
            "critical_warnings": critical_warnings,
            "state": curr_state,
        }


def extract_smc_monitor_summary(scan_result, decision):
    """
    Mengekstrak ringkasan terstruktur dari output asli `scan_xauusd_zones` dan `evaluate_trade_decision`
    tanpa mengubah satu baris pun algoritma SMC.
    """
    scan = scan_result or {}
    dec = decision or {}

    ctx = dec.get("context") or scan.get("context") or {}
    h4_bias = ctx.get("htf_bias") or scan.get("htf_bias") or "NETRAL"
    h1_struct = ctx.get("h1_structure") or scan.get("h1_structure") or "NETRAL"
    m30_status = ctx.get("m30_status") or scan.get("m30_status") or "NETRAL"

    raw_status = str(dec.get("status", "WAIT")).upper()
    raw_dir = str(dec.get("direction") or "").upper()
    if raw_status == "VALID" and raw_dir in ("BUY", "SELL"):
        decision_signal = raw_dir
        direction = raw_dir
        m5_bias = "CONFIRMED"
    elif raw_status in ("BUY", "SELL"):
        decision_signal = raw_status
        direction = raw_status
        m5_bias = "CONFIRMED"
    elif raw_status == "INVALID":
        decision_signal = "INVALID"
        direction = "NONE"
        m5_bias = "INVALID"
    else:
        decision_signal = "WAIT"
        direction = "NONE"
        m5_bias = "WAIT"

    stages = dec.get("m5_stages") or dec.get("stages") or {}
    zt = (stages.get("zone_touch") or {}).get("status", "CONFIRMED" if decision_signal in ("BUY", "SELL") else "WAIT")
    sw = (stages.get("liquidity_sweep") or {}).get("status", "CONFIRMED" if decision_signal in ("BUY", "SELL") else "WAIT")
    tr = (stages.get("structure_trigger") or {}).get("status", "CONFIRMED" if decision_signal in ("BUY", "SELL") else "WAIT")
    dp = (stages.get("displacement") or {}).get("status", "CONFIRMED" if decision_signal in ("BUY", "SELL") else "WAIT")
    rr_st = (stages.get("risk_reward") or {}).get("status", "CONFIRMED" if decision_signal in ("BUY", "SELL") else "WAIT")

    r_prio = dec.get("radar_priority") or scan.get("radar_priority") or {}
    pz = r_prio.get("primary_directional_zone") or r_prio.get("primary_zone")
    nz = r_prio.get("nearest_radar_zone")

    if isinstance(pz, dict) and pz.get("price_low") is not None:
        pz_label = (
            f"{pz.get('timeframe', 'H1')} {pz.get('zone_type', 'ZONE')} "
            f"{safe_float(pz.get('price_low')):.2f}-{safe_float(pz.get('price_high')):.2f} "
            f"(Grade {pz.get('grade', '-')}, Dist: {safe_float(pz.get('distance')):.2f} pts)"
        )
    else:
        pz_label = "NONE"

    if isinstance(nz, dict) and nz.get("price_low") is not None:
        nz_label = (
            f"{nz.get('timeframe', 'H1')} {nz.get('zone_type', 'ZONE')} "
            f"{safe_float(nz.get('price_low')):.2f}-{safe_float(nz.get('price_high')):.2f} "
            f"(Grade {nz.get('grade', '-')}, Dist: {safe_float(nz.get('distance')):.2f} pts)"
        )
    else:
        nz_label = "NONE"

    eval_sum = scan.get("zone_evaluation_summary") or {}
    inval_list = r_prio.get("invalidated_zones") or dec.get("invalidated_zones") or []
    inval_cnt = int(eval_sum.get("total_invalidated", len(inval_list)) or 0)
    mit_cnt = int(eval_sum.get("total_mitigated", 0) or 0)
    fresh_cnt = int(eval_sum.get("total_fresh", 0) or 0)
    tested_cnt = int(eval_sum.get("total_tested", 0) or 0)

    last_inval_type = "SUPPLY"
    if inval_list and isinstance(inval_list[0], dict):
        last_inval_type = str(inval_list[0].get("zone_type", "SUPPLY")).upper()

    plan = dec.get("plan") or {}
    entry_val = safe_float(plan.get("entry_acuan", plan.get("entry")), 0.0) if decision_signal in ("BUY", "SELL") else None
    sl_val = safe_float(plan.get("sl"), 0.0) if decision_signal in ("BUY", "SELL") else None
    tp1_val = safe_float(plan.get("tp1"), 0.0) if decision_signal in ("BUY", "SELL") else None
    tp2_val = safe_float(plan.get("tp2"), 0.0) if decision_signal in ("BUY", "SELL") else None
    rr_val = safe_float(plan.get("rr"), 0.0) if decision_signal in ("BUY", "SELL") else None

    reasons_list = dec.get("reasons") or []
    if decision_signal in ("BUY", "SELL"):
        main_reason = "; ".join(str(r) for r in reasons_list if r) or f"Setup {decision_signal} SMC terkonfirmasi."
    else:
        dyn_wait = r_prio.get("dynamic_wait_reason")
        if dyn_wait:
            main_reason = str(dyn_wait)
        elif reasons_list:
            main_reason = str(reasons_list[0]).replace("[SCANNER] Rejection Reason: ", "")
        else:
            main_reason = scan.get("empty_reason") or "Menunggu konfirmasi struktur & trigger M5 SMC."

    struct_status = "CONFLICT H4 vs H1" if ctx.get("has_conflict") else f"H4 {h4_bias} / H1 {h1_struct}"

    return {
        "method": "SMC",
        "status": "OK",
        "h4_bias": h4_bias,
        "h1_bias": h1_struct,
        "m30_bias": m30_status,
        "m5_bias": m5_bias,
        "structure_status": struct_status,
        "primary_zone_label": pz_label,
        "nearest_zone_label": nz_label,
        "fresh_count": fresh_cnt,
        "tested_count": tested_cnt,
        "mitigated_count": mit_cnt,
        "invalidated_count": inval_cnt,
        "last_invalidated_type": last_inval_type,
        "zone_touch": zt,
        "liquidity_sweep": sw,
        "choch_bos": tr,
        "displacement": dp,
        "rr_stage": rr_st,
        "decision_signal": decision_signal,
        "direction": direction,
        "entry": entry_val,
        "sl": sl_val,
        "tp1": tp1_val,
        "tp2": tp2_val,
        "rr": rr_val,
        "main_reason": main_reason,
    }


def format_smc_monitor_section(smc_cycle_result, detailed=False):
    """
    Menyusun blok `### A. SMC MONITOR` sesuai spesifikasi Part 2 Section 3:
    - Market bias H4, H1, M30, M5.
    - Status struktur pasar.
    - Zona SMC aktif dan jarak harga.
    - Status mitigasi atau invalidasi zona.
    - Status liquidity sweep, BOS, CHoCH, dan displacement jika tersedia.
    - Status konfirmasi entry.
    - Keputusan SMC: BUY / SELL / WAIT.
    - Entry, SL, TP, dan RR jika sudah valid.
    - Alasan utama keputusan.
    """
    if not isinstance(smc_cycle_result, dict):
        return "### A. SMC MONITOR\nSTATUS          : ERROR (Hasil siklus SMC tidak valid)"

    if smc_cycle_result.get("status") == "ERROR":
        return (
            "### A. SMC MONITOR\n"
            "STATUS          : ERROR\n"
            f"ERROR REASON    : {smc_cycle_result.get('error', 'Unknown SMC error')}"
        )

    s = smc_cycle_result.get("summary") or {}
    alerts = smc_cycle_result.get("alerts") or []

    lines = [
        "### A. SMC MONITOR",
        f"STATUS          : OK ({s.get('structure_status', 'ACTIVE')})",
        f"MARKET BIAS     : H4={s.get('h4_bias', 'N/A')} | H1={s.get('h1_bias', 'N/A')} | M30={s.get('m30_bias', 'N/A')} | M5={s.get('m5_bias', 'WAIT')}",
        f"MARKET STRUCT.  : {s.get('structure_status', 'N/A')}",
        f"ACTIVE SMC ZONE : Primary={s.get('primary_zone_label', 'NONE')}",
        f"NEAREST ZONE    : {s.get('nearest_zone_label', 'NONE')}",
        f"MITIGATION/INV. : Fresh={s.get('fresh_count', 0)} | Tested={s.get('tested_count', 0)} | Mitigated={s.get('mitigated_count', 0)} | Invalidated={s.get('invalidated_count', 0)}",
        f"SMC TRIGGERS    : Sweep={s.get('liquidity_sweep', 'WAIT')} | BOS/CHoCH={s.get('choch_bos', 'WAIT')} | Displacement={s.get('displacement', 'WAIT')}",
        f"ENTRY CONFIRM.  : Touch={s.get('zone_touch', 'WAIT')} | RR_Check={s.get('rr_stage', 'WAIT')}",
        f"SMC DECISION    : {s.get('decision_signal', 'WAIT')} (Direction: {s.get('direction', 'NONE')})",
    ]

    if s.get("decision_signal") in ("BUY", "SELL") and s.get("entry") is not None:
        lines.append(
            f"ENTRY / SL / TP : Entry {safe_float(s.get('entry')):.2f} | SL {safe_float(s.get('sl')):.2f} "
            f"| TP1 {safe_float(s.get('tp1')):.2f} | TP2 {safe_float(s.get('tp2', s.get('tp1'))):.2f} "
            f"| RR 1:{safe_float(s.get('rr')):.2f}"
        )
    else:
        lines.append("ENTRY / SL / TP : N/A (Setup belum terkonfirmasi)")

    lines.append(f"MAIN REASON     : {s.get('main_reason', 'N/A')}")

    if alerts:
        lines.append("SMC ALERTS      :")
        for al in alerts:
            lines.append(f"  ! {al}")

    if detailed and smc_cycle_result.get("scan_result") is not None and smc_cycle_result.get("decision") is not None:
        lines.append("")
        lines.append(
            format_scan_presentation(
                smc_cycle_result["scan_result"],
                smc_cycle_result["decision"],
                detailed=True,
            )
        )

    return "\n".join(lines)


def collect_account_monitor_data(symbol=None, snapshot=None, pos_list=None, orders_list=None, risk_data=None):
    """
    Mengumpulkan fakta akun MT5 secara read-only (posisi aktif, pending order,
    floating P/L, posisi/order tanpa SL, spread, dan peringatan risiko kritis).
    DILARANG membuka, mengubah, atau menutup order secara otomatis.
    """
    sym = symbol or DEFAULT_APP_SYMBOL
    snap_r = _extract_risk_monitor_snapshot(
        pos_list=pos_list,
        orders_list=orders_list,
        risk_data=risk_data,
    )

    positions = snap_r.get("positions") or []
    orders = snap_r.get("orders") or []

    floating_pnl = 0.0
    for p in positions:
        if isinstance(p, dict):
            floating_pnl += safe_float(p.get("profit", 0.0), 0.0)
        else:
            floating_pnl += safe_float(getattr(p, "profit", 0.0), 0.0)

    spread_val = None
    if isinstance(snapshot, dict) and snapshot.get("spread") is not None:
        spread_val = safe_float(snapshot.get("spread"), 0.0)
    elif mt5 is not None:
        try:
            t_info = mt5.symbol_info_tick(sym)
            if t_info is not None and getattr(t_info, "bid", 0.0) > 0:
                spread_val = round(float(t_info.ask - t_info.bid), 2)
        except Exception:
            pass

    risk_warnings = []
    p_nosl = int(snap_r.get("position_without_sl", 0))
    o_nosl = int(snap_r.get("pending_without_sl", 0))
    if p_nosl > 0:
        risk_warnings.append(f"CRITICAL: Terdapat {p_nosl} posisi running TANPA Stop Loss!")
    if o_nosl > 0:
        risk_warnings.append(f"WARNING: Terdapat {o_nosl} pending order TANPA Stop Loss!")
    if spread_val is not None and spread_val >= 0.80:
        risk_warnings.append(f"WARNING: Spread saat ini melebar ({spread_val:.2f} pts).")

    return {
        "symbol": sym,
        "available": bool(snap_r.get("available", False)),
        "open_positions": int(snap_r.get("running_position", 0)),
        "pending_orders": int(snap_r.get("pending_orders", 0)),
        "floating_pnl": round(floating_pnl, 2),
        "position_without_sl": p_nosl,
        "pending_without_sl": o_nosl,
        "spread": spread_val,
        "risk_warnings": risk_warnings,
    }


def format_account_monitor_section(acc_data):
    """
    Menyusun blok `### C. ACCOUNT MONITOR` (Read-Only):
    - Posisi aktif
    - Pending order
    - Profit/loss berjalan
    - Posisi atau pending order tanpa SL
    - Informasi spread
    - Peringatan risiko penting
    """
    acc = acc_data or {}
    spread_str = f"{safe_float(acc.get('spread')):.2f} pts" if acc.get("spread") is not None else "N/A"

    if not acc.get("available", False):
        lines = [
            "### C. ACCOUNT MONITOR (READ-ONLY)",
            "OPEN POSITIONS  : N/A (MT5 Offline / Mock)",
            "PENDING ORDERS  : N/A",
            "FLOATING P/L    : N/A",
            "WITHOUT SL      : N/A",
            f"SPREAD          : {spread_str}",
            "RISK ALERTS     : Data posisi akun MT5 belum tersedia.",
        ]
        return "\n".join(lines)

    warns = acc.get("risk_warnings") or []
    warn_str = "; ".join(warns) if warns else "AMAN (Seluruh posisi/order terproteksi SL)"

    lines = [
        "### C. ACCOUNT MONITOR (READ-ONLY)",
        f"OPEN POSITIONS  : {acc.get('open_positions', 0)}",
        f"PENDING ORDERS  : {acc.get('pending_orders', 0)}",
        f"FLOATING P/L    : ${safe_float(acc.get('floating_pnl')):.2f}",
        f"WITHOUT SL      : Position={acc.get('position_without_sl', 0)} | Pending={acc.get('pending_without_sl', 0)}",
        f"SPREAD          : {spread_str}",
        f"RISK ALERTS     : {warn_str}",
    ]
    return "\n".join(lines)


class UnifiedMonitor:
    """
    Pusat Monitoring Terpadu yang menjalankan SMC Monitor, SNR Monitor, dan MT5 Account Monitor
    secara simultan dalam satu siklus dengan 1x pengambilan data MT5 per siklus,
    independensi sinyal penuh, dan isolasi kegagalan (error isolation).
    """

    def __init__(self, symbol=None, smc_runner=None, snr_runner=None, snapshot_provider=None):
        self.symbol = symbol or DEFAULT_APP_SYMBOL
        self.smc_runner = smc_runner
        self.snr_runner = snr_runner
        self.snapshot_provider = snapshot_provider or create_market_snapshot
        self.smc_alert_tracker = SmcAlertTracker()
        self.snr_engine = SnrCycleEngine(symbol=self.symbol)
        self.cycle_count = 0
        self.last_combined_signature = None

    def _run_smc_isolated(self, shared_snapshot, detailed=False):
        """Menjalankan SMC Scanner dalam blok terisolasi agar error SMC tidak menghentikan SNR."""
        try:
            if callable(self.smc_runner):
                custom_out = self.smc_runner(shared_snapshot)
                if isinstance(custom_out, tuple) and len(custom_out) == 2:
                    smc_scan, smc_dec = custom_out
                elif isinstance(custom_out, dict) and "summary" in custom_out:
                    summary = custom_out["summary"]
                    al_bundle = self.smc_alert_tracker.evaluate_alerts(summary)
                    res = {
                        "method": "SMC",
                        "status": "OK",
                        "scan_result": custom_out.get("scan_result"),
                        "decision": custom_out.get("decision"),
                        "summary": summary,
                        "alerts": al_bundle["alerts"],
                        "critical_warnings": al_bundle["critical_warnings"],
                    }
                    res["formatted_section"] = format_smc_monitor_section(res, detailed=detailed)
                    return res
                else:
                    smc_scan, smc_dec = custom_out, {}
            else:
                smc_scan = scan_xauusd_zones(symbol=self.symbol, snapshot=shared_snapshot)
                smc_dec = evaluate_trade_decision(snapshot=shared_snapshot, scan_result=smc_scan)

            summary = extract_smc_monitor_summary(smc_scan, smc_dec)
            al_bundle = self.smc_alert_tracker.evaluate_alerts(summary)

            res = {
                "method": "SMC",
                "status": "OK",
                "scan_result": smc_scan,
                "decision": smc_dec,
                "summary": summary,
                "alerts": al_bundle["alerts"],
                "critical_warnings": al_bundle["critical_warnings"],
            }
            res["formatted_section"] = format_smc_monitor_section(res, detailed=detailed)
            return res
        except Exception as exc:
            err_res = {
                "method": "SMC",
                "status": "ERROR",
                "error": f"{type(exc).__name__}: {exc}",
                "scan_result": None,
                "decision": {
                    "method": "SMC",
                    "status": "ERROR",
                    "direction": "NONE",
                    "reasons": [f"SMC Error: {exc}"],
                },
                "summary": {
                    "method": "SMC",
                    "status": "ERROR",
                    "decision_signal": "ERROR",
                    "direction": "NONE",
                    "main_reason": f"SMC Error: {exc}",
                },
                "alerts": [f"[SMC] ERROR: {exc}"],
                "critical_warnings": [],
            }
            err_res["formatted_section"] = format_smc_monitor_section(err_res, detailed=detailed)
            return err_res

    def _run_snr_isolated(self, shared_snapshot, detailed=False):
        """Menjalankan SNR Scanner dalam blok terisolasi agar error SNR tidak menghentikan SMC."""
        try:
            if callable(self.snr_runner):
                custom_snr = self.snr_runner(shared_snapshot)
                if isinstance(custom_snr, dict) and custom_snr.get("method") == "SNR" and "formatted_section" in custom_snr:
                    return custom_snr
                scan_res = custom_snr.get("scan_result", custom_snr) if isinstance(custom_snr, dict) else {}
                dec = custom_snr.get("decision", scan_res.get("decision", {})) if isinstance(custom_snr, dict) else {}
                radar = scan_res.get("radar") or {}
                pa = scan_res.get("price_action") or {}
                al_bundle = self.snr_engine.alert_tracker.evaluate_alerts(radar, dec, price_action=pa)
                payload = {
                    "method": "SNR",
                    "status": "OK",
                    "scan_result": scan_res,
                    "decision": dec,
                    "radar": radar,
                    "alerts": al_bundle["alerts"],
                    "critical_warnings": al_bundle["critical_warnings"],
                }
                payload["formatted_section"] = format_snr_monitor_section(payload, detailed=detailed)
                return payload

            live_flag = bool(shared_snapshot.get("live", False)) if isinstance(shared_snapshot, dict) else True
            res = self.snr_engine.execute_cycle(
                snapshot=shared_snapshot,
                live=live_flag,
                detailed=detailed,
            )
            return res
        except Exception as exc:
            err_payload = {
                "method": "SNR",
                "status": "ERROR",
                "error": f"{type(exc).__name__}: {exc}",
                "scan_result": None,
                "decision": {
                    "method": "SNR",
                    "signal": "ERROR",
                    "direction": "NONE",
                    "reason": f"SNR Error: {exc}",
                },
                "radar": {},
                "alerts": [f"[SNR] ERROR: {exc}"],
                "critical_warnings": [],
            }
            err_payload["formatted_section"] = format_snr_monitor_section(err_payload, detailed=detailed)
            return err_payload

    def run_cycle(
        self,
        snapshot=None,
        mode="unified",
        detailed=False,
        pos_list=None,
        orders_list=None,
        risk_data=None,
    ):
        """
        Menjalankan satu siklus Unified Monitor:
        - Mengambil 1 snapshot data MT5 yang konsisten (jika `snapshot` belum diberikan).
        - Menjalankan SMC Monitor dan/atau SNR Monitor sesuai `mode` ("unified", "smc", "snr").
        - Menjalankan MT5 Account Monitor secara read-only.
        - Mengumpulkan alert perubahan state (`[SMC] ...`, `[SNR] ...`) tanpa menggabungkan keputusan.
        """
        self.cycle_count += 1
        mode_norm = str(mode or "unified").strip().lower()

        shared_snap = snapshot
        if shared_snap is None:
            try:
                shared_snap = self.snapshot_provider(symbol=self.symbol)
            except Exception:
                shared_snap = {"symbol": self.symbol, "frames": {}, "live": False}

        smc_res = None
        snr_res = None

        if mode_norm in ("unified", "smc"):
            smc_res = self._run_smc_isolated(shared_snap, detailed=detailed)

        if mode_norm in ("unified", "snr"):
            snr_res = self._run_snr_isolated(shared_snap, detailed=detailed)

        acc_res = collect_account_monitor_data(
            symbol=self.symbol,
            snapshot=shared_snap,
            pos_list=pos_list,
            orders_list=orders_list,
            risk_data=risk_data,
        )
        acc_section = format_account_monitor_section(acc_res)

        combined_alerts = []
        if smc_res and smc_res.get("alerts"):
            combined_alerts.extend(smc_res["alerts"])
        if snr_res and snr_res.get("alerts"):
            combined_alerts.extend(snr_res["alerts"])

        smc_sig = (smc_res.get("summary") or {}).get("decision_signal", "N/A") if smc_res else "OFF"
        snr_sig = (snr_res.get("decision") or {}).get("signal", "N/A") if snr_res else "OFF"

        curr_sig_tuple = (
            smc_sig,
            snr_sig,
            (smc_res.get("summary") or {}).get("primary_zone_label") if smc_res else None,
            (snr_res.get("decision") or {}).get("zone_label") if snr_res else None,
            smc_res.get("status") if smc_res else None,
            snr_res.get("status") if snr_res else None,
        )
        state_changed = (
            self.last_combined_signature is None
            or curr_sig_tuple != self.last_combined_signature
            or bool(combined_alerts)
        )
        self.last_combined_signature = curr_sig_tuple

        price_disp = safe_float(shared_snap.get("snapshot_price", shared_snap.get("bid", 0.0)), 0.0) if isinstance(shared_snap, dict) else 0.0
        if price_disp <= 0 and snr_res and isinstance(snr_res.get("scan_result"), dict):
            price_disp = safe_float(snr_res["scan_result"].get("price"), 0.0)

        ts_str = (
            shared_snap.get("server_time")
            if isinstance(shared_snap, dict) and shared_snap.get("server_time")
            else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

        header_title = (
            "XAUUSD UNIFIED MONITOR — SMC + SNR + ACCOUNT"
            if mode_norm == "unified"
            else (
                "XAUUSD SMC MONITOR"
                if mode_norm == "smc"
                else "XAUUSD SNR MONITOR"
            )
        )

        report_lines = [
            "====================================================",
            f"   {header_title}",
            "====================================================",
            f"SYMBOL : {self.symbol} | PRICE : {price_disp:.2f} | TIME : {ts_str} | CYCLE : #{self.cycle_count}",
        ]
        if mode_norm == "unified":
            report_lines.append(f"INDEPENDENT SIGNALS : SMC = {smc_sig} | SNR = {snr_sig}")
        report_lines.append("----------------------------------------------------")

        if smc_res is not None:
            report_lines.append(smc_res["formatted_section"])
            report_lines.append("----------------------------------------------------")

        if snr_res is not None:
            report_lines.append(snr_res["formatted_section"])
            report_lines.append("----------------------------------------------------")

        report_lines.append(acc_section)

        if combined_alerts:
            report_lines.append("----------------------------------------------------")
            report_lines.append("ACTIVE STATE-CHANGE ALERTS:")
            for al in combined_alerts:
                report_lines.append(f"  {al}")

        report_lines.append("====================================================")
        formatted_report = "\n".join(report_lines)

        return {
            "cycle": self.cycle_count,
            "mode": mode_norm,
            "detailed": bool(detailed),
            "snapshot": shared_snap,
            "snapshot_id": shared_snap.get("snapshot_id") if isinstance(shared_snap, dict) else None,
            "smc": smc_res,
            "snr": snr_res,
            "account": acc_res,
            "smc_signal": smc_sig,
            "snr_signal": snr_sig,
            "alerts": combined_alerts,
            "risk_warnings": acc_res.get("risk_warnings") or [],
            "state_changed": state_changed,
            "formatted_report": formatted_report,
        }


def parse_monitor_command(cmd_str):
    """
    Mem-parsing command pemantauan secara presisi agar tidak salah mengenali command yang mirip:
    - `pantau` -> mode="unified", detailed=False
    - `pantau detail` -> mode="unified", detailed=True
    - `pantau smc` -> mode="smc", detailed=False
    - `pantau smc detail` -> mode="smc", detailed=True
    - `pantau snr` -> mode="snr", detailed=False
    - `pantau snr detail` -> mode="snr", detailed=True
    - `pantau 3900 3905` -> mode="legacy_manual", detailed=False
    """
    raw = str(cmd_str or "").strip().lower()
    tokens = raw.split()
    if not tokens or tokens[0] not in ("pantau", "watch"):
        return {"valid": False, "mode": "none", "detailed": False}

    rest = tokens[1:]
    if not rest:
        return {"valid": True, "mode": "unified", "detailed": False}

    if rest == ["detail"]:
        return {"valid": True, "mode": "unified", "detailed": True}

    if rest == ["smc"]:
        return {"valid": True, "mode": "smc", "detailed": False}

    if rest in (["smc", "detail"], ["detail", "smc"]):
        return {"valid": True, "mode": "smc", "detailed": True}

    if rest == ["snr"]:
        return {"valid": True, "mode": "snr", "detailed": False}

    if rest in (["snr", "detail"], ["detail", "snr"]):
        return {"valid": True, "mode": "snr", "detailed": True}

    # Jika pengguna memasukkan angka harga manual (misal `pantau 3900 3905`), arahkan ke mode_pantau_mt5 lama
    if any(re.search(r"\d", tok) for tok in rest):
        return {"valid": True, "mode": "legacy_manual", "detailed": "detail" in rest}

    return {"valid": True, "mode": "unified", "detailed": "detail" in rest}


def run_unified_pantau(
    cmd_str="pantau",
    teks_ai_terakhir="",
    max_cycles=None,
    sleep_sec=2.0,
    snapshot=None,
    smc_runner=None,
    snr_runner=None,
):
    """
    Entrypoint utama untuk menjalankan command:
    - `pantau`
    - `pantau detail`
    - `pantau smc`
    - `pantau smc detail`
    - `pantau snr`
    - `pantau snr detail`
    """
    parsed = parse_monitor_command(cmd_str)
    mode = parsed["mode"]
    detailed = parsed["detailed"]

    if mode == "legacy_manual":
        return mode_pantau_mt5(cmd_str, teks_ai_terakhir)

    monitor = UnifiedMonitor(
        symbol=DEFAULT_APP_SYMBOL,
        smc_runner=smc_runner,
        snr_runner=snr_runner,
    )

    cycles_done = 0
    last_out = None
    try:
        while True:
            cycle_out = monitor.run_cycle(
                snapshot=snapshot,
                mode=mode,
                detailed=detailed,
            )
            last_out = cycle_out
            cycles_done += 1

            # Cetak laporan penuh pada siklus pertama, atau cetak alert/laporan saat ada perubahan state
            if cycles_done == 1:
                print("\n" + cycle_out["formatted_report"])
            elif cycle_out.get("state_changed"):
                if cycle_out.get("alerts"):
                    for al in cycle_out["alerts"]:
                        print(al)
                print("\n" + cycle_out["formatted_report"])
            elif cycle_out.get("risk_warnings"):
                for rw in cycle_out["risk_warnings"]:
                    print(f"[ACCOUNT RISK] {rw}")

            if max_cycles is not None and cycles_done >= int(max_cycles):
                break
            time.sleep(max(0.1, float(sleep_sec)))
    except KeyboardInterrupt:
        print("\n[UNIFIED MONITOR] Pemantauan dihentikan oleh pengguna.")

    return last_out
