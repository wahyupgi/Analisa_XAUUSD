# -*- coding: utf-8 -*-
"""
modules/smart_monitor_report.py
Presentation layer Smart Monitor / Unified Monitor: konteks smart monitor dan formatter laporan
(`pantau` ringkas & detail) serta blok Account Monitor.

Dipindahkan apa adanya dari modules/unified_monitor.py (structural refactoring, tanpa perubahan logika).
Modul ini tidak bergantung pada unified_monitor (dependency satu arah); unified_monitor
mengimpor kembali seluruh nama publiknya agar caller lama tetap kompatibel.
"""
import re

from modules.config import LOW_RISK_MAX_ATR_RATIO, LOW_RISK_MAX_DISTANCE_PT, SL_JARAK_MAKS, ZONE_APPROACH_DISTANCE_POINTS
from modules.mt5_conn import (
    fetch_mt5_pending_orders_snapshot,
    find_pending_orders_on_invalid_zones,
    format_pending_compact_status,
    format_pending_reconciliation_evidence,
    reconcile_pending_plan_with_mt5,
    reconcile_pending_plans_batch,
    summarize_mt5_risk_warnings,
)
from modules.snr.breakout_detection import build_snr_breakout_watch
from modules.snr.data_provider import safe_float
from modules.snr.entry_validator import _is_live_active_snr_zone, select_live_snr_focus
from modules.terminal_ui import (
    fit_dashboard_text,
    get_terminal_height,
    get_terminal_width,
    render_big_price_digits,
    supports_unicode_glyphs,
    truncate_clean,
    wrap_dashboard_field,
)
from modules.utils import (
    format_wib_timestamp,
    get_symbol_unit_spec,
    extract_unit_spec_from_container,
    format_price_distance,
)
from modules.zone_scanner import format_smc_zone_lines, is_far_reference_zone, smc_zone_summary


def format_account_monitor_section(acc_data):
    """
    Menyusun blok `### C. ACCOUNT MONITOR` (STRICTLY READ-ONLY):
    - Balance, Equity, Margin, Free Margin, Margin Level (%)
    - Posisi aktif & Exposure Lot (Buy lot vs Sell lot)
    - Pending order
    - Profit/loss berjalan
    - Posisi atau pending order tanpa SL (Peringatan Keras)
    - Akumulasi total risiko akun (berdasarkan SL atau UNBOUNDED)
    - Informasi spread
    - Peringatan risiko penting
    """
    acc = acc_data or {}
    _u_acc_r = extract_unit_spec_from_container(acc, symbol=acc.get("symbol"))
    spread_str = format_price_distance(safe_float(acc.get("spread")), spec=_u_acc_r, include_mt5_points=True) if acc.get("spread") is not None else "N/A"

    if not acc.get("available", False):
        lines = [
            "### C. ACCOUNT MONITOR (STRICTLY READ-ONLY)",
            "BALANCE / EQUITY: N/A (MT5 Offline / Mock)",
            "OPEN POSITIONS  : N/A",
            "PENDING ORDERS  : N/A",
            "FLOATING P/L    : N/A",
            "TOTAL RISK (SL) : N/A",
            "WITHOUT SL      : N/A",
            f"SPREAD          : {spread_str}",
            "RISK ALERTS     : Data posisi akun MT5 belum tersedia.",
            "AUDIT MODE      : STRICTLY READ-ONLY (Tidak ada modifikasi order otomatis)",
        ]
        return "\n".join(lines)

    warns = acc.get("risk_warnings") or []
    warn_str = "; ".join(warns) if warns else "AMAN (Seluruh posisi/order terproteksi SL)"

    bal = safe_float(acc.get("balance", 0.0))
    eq = safe_float(acc.get("equity", 0.0))
    cur = acc.get("currency", "USD")
    m_lvl = safe_float(acc.get("margin_level", 0.0))
    m_val = safe_float(acc.get("margin", 0.0))
    fm_val = safe_float(acc.get("free_margin", 0.0))
    b_lots = safe_float(acc.get("buy_lots", 0.0))
    s_lots = safe_float(acc.get("sell_lots", 0.0))
    net_lots = safe_float(acc.get("net_lot_exposure", 0.0))

    lines = [
        "### C. ACCOUNT MONITOR (STRICTLY READ-ONLY)",
        f"BALANCE / EQUITY: {bal:,.2f} {cur} / {eq:,.2f} {cur} | MARGIN LEVEL: {m_lvl:.2f}% (Margin: {m_val:,.2f} {cur} | Free: {fm_val:,.2f} {cur})",
        f"OPEN POSITIONS  : {acc.get('open_positions', 0)} (Buy: {b_lots:.2f} lot | Sell: {s_lots:.2f} lot | Net: {net_lots:+.2f} lot)",
        f"PENDING ORDERS  : {acc.get('pending_orders', 0)}",
        f"FLOATING P/L    : {acc.get('floating_pnl', 0.0):+,.2f} {cur}",
        f"TOTAL RISK (SL) : {acc.get('total_risk_status', 'N/A')}",
        f"WITHOUT SL      : Position={acc.get('position_without_sl', 0)} | Pending={acc.get('pending_without_sl', 0)}",
        f"SPREAD          : {spread_str}",
        f"RISK ALERTS     : {warn_str}",
        "AUDIT MODE      : STRICTLY READ-ONLY (Dilarang auto-close / auto-modify / auto-trade)",
    ]
    return "\n".join(lines)


def _format_pending_compact_tag(item, active_zone_id=None, is_selected=True):
    if not isinstance(item, dict):
        return "WAIT"
    plan_st = str(item.get("plan_status") or item.get("status") or "WAIT").strip()
    if not plan_st.upper().startswith("READY"):
        recon = item.get("mt5_reconciliation")
        b_st = str(item.get("broker_order_status") or (recon or {}).get("broker_order_status") or "").upper()
        if b_st == "OTHER_ORDER_EXISTS":
            if item.get("compact_status"):
                return str(item["compact_status"])
            tot_ord = int((recon or {}).get("symbol_orders_count") or 0)
            return f"{plan_st} | MT5: NO MATCH | OTHER ORDERS: {tot_ord}"
        return plan_st
    if item.get("compact_status"):
        return str(item["compact_status"])
    recon = item.get("mt5_reconciliation")
    if isinstance(recon, dict):
        return format_pending_compact_status(
            "READY",
            recon,
            order_type=str(item.get("order_type") or ""),
            entry_price=safe_float(item.get("entry"), 0.0),
        )
    return str(item.get("status") or "READY | MT5 STATUS UNKNOWN")


def resolve_monitor_pending_plans(
    decision,
    scan,
    active_zone,
    tracker,
    data_invalid,
    completed_zone_id=None,
    zone_usable_fn=None,
    mt5_orders_snapshot=None,
    symbol=None,
):
    """
    Satu sumber otoritatif untuk Selected Pending Plan dan alternatif pending valid dalam 1 siklus monitor.
    - Tidak mengubah ranking/strategi `detect_early_setups`.
    - Memvalidasi setiap kandidat terhadap lifecycle live (`scan` + `ZoneTracker` + `completed_zone_id`).
    - Memisahkan `plan_status` (READY/WAIT/INVALID/EXPIRED) dari `broker_order_status`
      (NOT_PLACED/PLACED_VERIFIED/PLACED_MISMATCH/OTHER_ORDER_EXISTS/UNKNOWN) melalui rekonsiliasi MT5 read-only.
    """
    dec = decision if isinstance(decision, dict) else {}
    sc = scan if isinstance(scan, dict) else {}
    eff_symbol = symbol or sc.get("symbol") or dec.get("symbol")
    ord_snap = (
        mt5_orders_snapshot
        if mt5_orders_snapshot is not None
        else (sc.get("mt5_orders_snapshot") or dec.get("mt5_orders_snapshot"))
    )
    if not (isinstance(ord_snap, dict) and "query_ok" in ord_snap and "orders" in ord_snap):
        ord_snap = fetch_mt5_pending_orders_snapshot(symbol=eff_symbol, orders_override=ord_snap)

    early = dec.get("early_setup") if isinstance(dec.get("early_setup"), dict) else {}
    best_raw = early.get("best_setup") if isinstance(early.get("best_setup"), dict) else None
    all_raw = list(early.get("all_setups") or [])

    raw_list = []
    seen_keys = set()
    for item in ([best_raw] if best_raw else []) + all_raw:
        if not isinstance(item, dict):
            continue
        zid = str(item.get("zone_id") or item.get("source_zone_id") or "").strip()
        key = zid or f"_anon_{len(raw_list)}"
        if key in seen_keys:
            continue
        seen_keys.add(key)
        raw_list.append(item)

    live_by_id = {}
    for z in (sc.get("active_zones") or []) + (sc.get("radar_zones") or []):
        if isinstance(z, dict) and z.get("zone_id") and z["zone_id"] not in live_by_id:
            live_by_id[str(z["zone_id"])] = z
    rejected_ids = {
        str(z.get("zone_id"))
        for z in (sc.get("rejected_zones") or [])
        if isinstance(z, dict) and z.get("zone_id") and str(z.get("zone_id")) not in live_by_id
    }

    r_prio = dec.get("radar_priority") or sc.get("radar_priority") or {}
    prim_ref = r_prio.get("primary_directional_zone") or r_prio.get("primary_zone") or {}
    primary_zid = str(prim_ref.get("zone_id") or "") if isinstance(prim_ref, dict) else ""
    active_zid = str((active_zone or {}).get("zone_id") or "") if isinstance(active_zone, dict) else ""

    valid_items = []
    if not data_invalid:
        for raw_p in raw_list:
            if raw_p.get("valid") is False or raw_p.get("is_expired") or raw_p.get("is_expiration_error"):
                continue
            st_raw = str(raw_p.get("plan_status") or raw_p.get("early_pending_status") or raw_p.get("status") or "").upper()
            if not st_raw.startswith("READY"):
                continue
            p_zid = str(raw_p.get("zone_id") or raw_p.get("source_zone_id") or "").strip()
            live_z = None
            if p_zid:
                if completed_zone_id and p_zid == completed_zone_id:
                    continue
                if p_zid in rejected_ids:
                    continue
                live_z = live_by_id.get(p_zid)
                if live_z is not None:
                    if callable(zone_usable_fn) and not zone_usable_fn(live_z):
                        continue
                    if str(live_z.get("risk_tier") or "").upper() == "FAR_ZONE" or is_far_reference_zone(live_z):
                        continue
                    if str(live_z.get("radar_eligibility") or "QUALIFIED").upper() == "NOT_QUALIFIED":
                        continue
                if tracker is not None and hasattr(tracker, "get_zone"):
                    trk = tracker.get_zone(p_zid)
                    trk_st = str((trk or {}).get("lifecycle_status") or (trk or {}).get("status") or "").upper()
                    if trk_st in ("INVALIDATED", "EXPIRED", "MITIGATED", "BROKEN"):
                        continue

            order_type = str(raw_p.get("order_type") or "").upper()
            p_dir = "BUY" if "BUY" in order_type else ("SELL" if "SELL" in order_type else "NONE")
            entry_v = safe_float(raw_p.get("entry_price", raw_p.get("entry")), 0.0)
            sl_v = safe_float(raw_p.get("sl"), 0.0)
            tp1_v = safe_float(raw_p.get("tp1"), 0.0)
            tp2_v = safe_float(raw_p.get("tp2"), 0.0)
            if p_dir not in ("BUY", "SELL") or entry_v <= 0 or sl_v <= 0 or tp1_v <= 0:
                continue
            if p_dir == "BUY" and not (sl_v < entry_v < tp1_v):
                continue
            if p_dir == "SELL" and not (tp1_v < entry_v < sl_v):
                continue

            p_lo = safe_float(raw_p.get("zone_low", raw_p.get("price_low", (live_z or {}).get("price_low"))), 0.0)
            p_hi = safe_float(raw_p.get("zone_high", raw_p.get("price_high", (live_z or {}).get("price_high"))), 0.0)
            p_tf = str((live_z or {}).get("timeframe") or raw_p.get("timeframe") or raw_p.get("zone_timeframe") or "H1")
            p_zt = str((live_z or {}).get("zone_type") or raw_p.get("zone_type") or ("DEMAND" if p_dir == "BUY" else ("SUPPLY" if p_dir == "SELL" else ""))).upper()
            p_grade = str((live_z or {}).get("grade") or raw_p.get("zone_grade") or "N/A")
            raw_sc = (live_z or {}).get("score") if (live_z and live_z.get("score") is not None) else raw_p.get("zone_score")
            p_score = safe_float(raw_sc) if raw_sc is not None else None
            is_active = bool(p_zid and active_zid and p_zid == active_zid)
            is_prim = bool(p_zid and primary_zid and p_zid == primary_zid)
            src_role = (
                "ACTIVE_FOCUS" if is_active
                else ("PRIMARY_ZONE" if is_prim else "NON_FOCUS_ZONE")
            )
            src_label = raw_p.get("source_zone_label") or (
                f"{p_tf} {p_zt} {p_lo:.2f}-{p_hi:.2f}".strip() if (p_lo > 0 and p_hi > 0) else f"{p_tf} {p_zt}".strip()
            )
            synced_plan = dict(raw_p)
            if live_z is not None:
                synced_plan["zone_grade"] = p_grade
                if p_score is not None:
                    synced_plan["zone_score"] = p_score
            item_struct = {
                "setup_id": f"{p_zid}:{p_dir}:{raw_p.get('setup_type', 'PENDING')}" if p_zid else "N/A",
                "zone_id": p_zid or "N/A",
                "source_zone_id": p_zid or "N/A",
                "source_zone_label": src_label,
                "source_role": src_role,
                "is_active_focus": is_active,
                "is_primary_zone": is_prim,
                "plan_role": "CANDIDATE",
                "direction": p_dir,
                "order_type": order_type or ("BUY LIMIT" if p_dir == "BUY" else ("SELL LIMIT" if p_dir == "SELL" else "NONE")),
                "zone_type": p_zt,
                "zone_range": (p_lo, p_hi),
                "setup_type": str(raw_p.get("setup_type") or "PENDING"),
                "timeframe": p_tf,
                "zone_grade": p_grade,
                "zone_score": p_score,
                "entry": entry_v or None,
                "sl": sl_v or None,
                "tp1": tp1_v or None,
                "tp2": tp2_v or None,
                "rr": safe_float(raw_p.get("rr", raw_p.get("rr_tp1")), 0.0) or None,
                "created_at": raw_p.get("created_at") or None,
                "expires_at": raw_p.get("expires_at") or raw_p.get("suggested_expiration") or None,
                "suggested_expiration": raw_p.get("suggested_expiration") or raw_p.get("expires_at") or None,
                "remaining": raw_p.get("remaining") or None,
                "snapshot_id": raw_p.get("snapshot_id") or sc.get("snapshot_id") or dec.get("snapshot_id"),
                "snapshot_time": raw_p.get("snapshot_time") or sc.get("snapshot_time") or dec.get("snapshot_time"),
                "plan_status": "READY",
                "broker_order_status": "UNKNOWN",
                "mt5_execution_state": "UNKNOWN",
                "status": "READY",
                "is_placed": False,
                "raw_plan": synced_plan,
            }
            valid_items.append(item_struct)

    if valid_items:
        best_zid = str((best_raw or {}).get("zone_id") or (best_raw or {}).get("source_zone_id") or "").strip() if isinstance(best_raw, dict) else ""
        selected = next((it for it in valid_items if best_zid and it.get("zone_id") == best_zid), valid_items[0])
        selected["plan_role"] = "SELECTED_PENDING"
        alternatives = []
        for it in valid_items:
            if it is selected or (selected.get("zone_id") != "N/A" and it.get("zone_id") == selected.get("zone_id")):
                continue
            it["plan_role"] = "ALT_PENDING"
            alternatives.append(it)

        ordered_plans = [selected, *alternatives]
        reconcile_pending_plans_batch(ordered_plans, mt5_snapshot=ord_snap, symbol=eff_symbol)
        for it_p in ordered_plans:
            it_p["status"] = it_p.get("compact_status") or format_pending_compact_status(
                it_p.get("plan_status", "READY"),
                it_p.get("mt5_reconciliation"),
                order_type=it_p.get("order_type", ""),
                entry_price=safe_float(it_p.get("entry"), 0.0),
            )
            raw_ref = it_p.get("raw_plan")
            if isinstance(raw_ref, dict):
                raw_ref["plan_status"] = it_p["plan_status"]
                raw_ref["broker_order_status"] = it_p["broker_order_status"]
                raw_ref["mt5_execution_state"] = it_p["mt5_execution_state"]
                raw_ref["is_placed"] = it_p["is_placed"]
                raw_ref["sl_missing_high_risk"] = it_p.get("sl_missing_high_risk", False)
                raw_ref["mt5_reconciliation"] = it_p.get("mt5_reconciliation")
                raw_ref["compact_status"] = it_p["status"]

        selected_setup = {k: v for k, v in selected.items() if k != "raw_plan"}
        selected_setup["alternatives"] = [{k: v for k, v in alt.items() if k != "raw_plan"} for alt in alternatives]
        return {
            "selected_plan": selected["raw_plan"],
            "pending_setup": selected_setup,
            "alt_pending_setups": selected_setup["alternatives"],
            "pending_status": selected_setup["status"],
        }

    empty_recon = reconcile_pending_plan_with_mt5({"plan_status": "WAIT"}, mt5_snapshot=ord_snap, symbol=eff_symbol)
    empty_setup = {
        "setup_id": "N/A",
        "zone_id": "N/A",
        "source_zone_id": "N/A",
        "source_zone_label": "NONE",
        "source_role": "NONE",
        "direction": "NONE",
        "order_type": "NONE",
        "zone_type": "",
        "zone_range": (0.0, 0.0),
        "setup_type": "PENDING",
        "timeframe": "N/A",
        "zone_grade": "N/A",
        "zone_score": None,
        "entry": None,
        "sl": None,
        "tp1": None,
        "tp2": None,
        "rr": None,
        "created_at": None,
        "expires_at": None,
        "suggested_expiration": None,
        "remaining": None,
        "snapshot_id": sc.get("snapshot_id") or dec.get("snapshot_id"),
        "snapshot_time": sc.get("snapshot_time") or dec.get("snapshot_time"),
        "plan_status": "WAIT",
        "broker_order_status": empty_recon["broker_order_status"],
        "mt5_execution_state": empty_recon["broker_order_status"],
        "mt5_reconciliation": empty_recon,
        "compact_status": empty_recon.get("compact_status") or "WAIT",
        "status": "WAIT",
        "is_placed": False,
        "alternatives": [],
    }
    return {
        "selected_plan": {},
        "pending_setup": empty_setup,
        "alt_pending_setups": [],
        "pending_status": "WAIT",
    }


def classify_monitor_alerts(alerts, risk_warnings=None, default_ts=None):
    """
    Pisahkan alert monitor ke 5 kategori tegas:
    - RISK_ALERT      : critical risk warning (tanpa SL, MT5 disconnected, spread terlalu lebar, SNR critical).
    - MARKET_ALERT    : breakdown, breakout, reclaim, retest, rejection, broken, sweep, touch/approaching aktif.
    - ENTRY_ALERT     : pending ready, confirmed entry (BUY/SELL setup).
    - ACCOUNT_ALERT   : posisi/order MT5 dan status fokus-posisi.
    - HISTORICAL_INFO : event lama/historis, bukan current alert.
    """
    categorized = {
        "risk": [],
        "market": [],
        "entry": [],
        "account": [],
        "historical": [],
        "critical_risk": [],
        "events": [],
    }
    seen = set()
    raw_items = []
    for item in (alerts or []):
        raw_items.append((item, False))
    for item in (risk_warnings or []):
        raw_items.append((item, True))

    for raw, from_risk in raw_items:
        msg = str(raw or "").replace("\n", " ").strip()
        if not msg or msg in seen:
            continue
        seen.add(msg)
        up = msg.upper()
        is_critical = bool(
            from_risk
            or "[RISK WARNING]" in up
            or "TANPA STOP LOSS" in up
            or "[SNR CRITICAL]" in up
            or "MT5 DISCONNECTED" in up
            or "RISK ALERT" in up
            or "SPREAD TOO HIGH" in up
        )

        # 1. HISTORICAL_INFO (event lama — tidak pernah dipromosikan menjadi current alert)
        if any(tok in up for tok in ("(HISTORIS)", "HISTORICAL", "PERNAH DISENTUH", "MARKET CLOSED")):
            cat = "HISTORICAL_INFO"
            prio = 10
        # 2. RISK_ALERT (critical risk warning — wajib selalu dipertahankan)
        elif is_critical:
            cat = "RISK_ALERT"
            prio = 88
        # 3. ACCOUNT_ALERT (posisi/order MT5, fokus-posisi, koneksi/spread normal)
        elif any(tok in up for tok in (
            "[RISK", "[ACCOUNT RISK]", "[DATA]",
            "POSISI MT5", "POSISI DIBUKA", "RUNNING: POSISI", "POSISI AMBIGU",
            "TIDAK LAGI TERLIHAT", "BELUM ADA POSISI", "SEBELUM POSISI TERDETEKSI",
            "HASIL BELUM DIPASTIKAN", "RIWAYAT TICK", "RIWAYAT TIDAK LENGKAP",
            "MT5 RECONNECTED", "SPREAD NORMAL",
            "DATA UNAVAILABLE", "RENCANA SL/TP1", "SL RENCANA", "TP1 RENCANA",
        )):
            cat = "ACCOUNT_ALERT"
            prio = 30
        # 4. ENTRY_ALERT (confirmed entry / pending ready)
        elif any(tok in up for tok in (
            "BUY SETUP", "SELL SETUP", "SETUP TERKONFIRMASI", "PENDING READY",
            "READY UNTUK BUY", "READY UNTUK SELL", "[READY:",
        )):
            cat = "ENTRY_ALERT"
            prio = 100 if any(tok in up for tok in ("CONFIRMED", "TERKONFIRMASI", "BUY SETUP", "SELL SETUP")) else 75
        # 5. MARKET_ALERT (breakdown, breakout, reclaim, retest, rejection, broken, structure, touch)
        elif any(tok in up for tok in (
            "BREAKDOWN", "BREAKOUT", "RECLAIM", "RETEST", "REJECTION", "BROKEN", "DITEMBUS",
            "CHOCH", "BOS M5", "LIQUIDITY SWEEP", "DISPLACEMENT",
            "ZONE TOUCH", "HARGA DI ZONA", "HARGA DI SUPPORT", "HARGA DI RESISTANCE",
            "HARGA DEKAT", "APPROACHING", "PERUBAHAN ZONA", "FOCUS CHANGED", "INVALID", "TERMITIGASI",
        )):
            cat = "MARKET_ALERT"
            if any(tok in up for tok in ("BREAKDOWN", "BREAKOUT", "RECLAIM", "RETEST")):
                prio = 95
            elif any(tok in up for tok in ("REJECTION", "BROKEN", "DITEMBUS", "CHOCH", "BOS M5", "SWEEP", "DISPLACEMENT")):
                prio = 90
            elif any(tok in up for tok in ("INVALID", "TERMITIGASI")):
                prio = 70
            elif any(tok in up for tok in ("PERUBAHAN ZONA", "FOCUS CHANGED")):
                prio = 65
            else:
                prio = 60
        else:
            cat = "ACCOUNT_ALERT"
            prio = 25

        source = (
            "SNR" if up.startswith("[SNR") else
            "SMC" if up.startswith("[SMC") else
            "RISK" if (from_risk or is_critical or up.startswith("[RISK") or up.startswith("[ACCOUNT")) else
            "DATA" if up.startswith("[DATA") else
            "MONITOR"
        )
        ev = {
            "category": cat,
            "priority": prio,
            "is_critical": is_critical,
            "source": source,
            "timestamp": str(default_ts) if default_ts else None,
            "message": msg,
        }
        categorized["events"].append(ev)
        if cat == "RISK_ALERT":
            categorized["risk"].append(msg)
            categorized["critical_risk"].append(msg)
        elif cat == "MARKET_ALERT":
            categorized["market"].append(msg)
        elif cat == "ENTRY_ALERT":
            categorized["entry"].append(msg)
        elif cat == "ACCOUNT_ALERT":
            categorized["account"].append(msg)
        else:
            categorized["historical"].append(msg)

    return categorized


def _as_monitor_zone(zone):
    if isinstance(zone, dict):
        return zone
    to_dict = getattr(zone, "to_dict", None)
    if callable(to_dict):
        converted = to_dict()
        return converted if isinstance(converted, dict) else {}
    return {}


def _m5_stages_match_zone(decision, zone):
    """True hanya jika hasil evaluasi/konfirmasi M5 pada decision benar-benar milik zone target."""
    if not isinstance(decision, dict) or not isinstance(zone, dict):
        return False
    target_zid = str(zone.get("zone_id") or "").strip()
    if not target_zid or target_zid in ("N/A", "NONE"):
        return False
    if str(zone.get("source") or "SMC").strip().upper() == "SNR":
        return False
    stages = decision.get("m5_stages") or decision.get("stages") or {}
    trig = stages.get("structure_trigger") if isinstance(stages.get("structure_trigger"), dict) else {}
    m5_conf = decision.get("m5_confirmation") if isinstance(decision.get("m5_confirmation"), dict) else {}
    m5_obj = decision.get("m5") if isinstance(decision.get("m5"), dict) else {}
    dec_zone = decision.get("zone") if isinstance(decision.get("zone"), dict) else {}
    origin_zid = str(
        decision.get("m5_stages_zone_id")
        or m5_conf.get("zone_id")
        or trig.get("zone_id")
        or m5_obj.get("zone_id")
        or dec_zone.get("zone_id")
        or ""
    ).strip()
    if not origin_zid or origin_zid != target_zid:
        return False
    origin_tf = str(
        m5_conf.get("timeframe")
        or trig.get("timeframe")
        or m5_obj.get("timeframe")
        or dec_zone.get("timeframe")
        or ""
    ).strip().upper()
    target_tf = str(zone.get("timeframe") or "").strip().upper()
    if origin_tf and target_tf and origin_tf != target_tf:
        return False
    return True


def _calc_zone_touch(z, p, stages_dict, trk_rec, life_st):
    z_lo = safe_float(z.get("price_low"), 0.0)
    z_hi = safe_float(z.get("price_high"), 0.0)
    if not z:
        return "NO_ZONE", "NO SMC SETUP"
    if (
        z.get("is_valid") is False
        or z.get("is_broken") is True
        or life_st in ("INVALIDATED", "BROKEN")
        or str(z.get("status") or "").upper() in ("INVALIDATED", "BROKEN")
    ):
        return "INVALIDATED", "INVALID — BROKEN"
    if z_lo > 0 and z_hi > 0 and p > 0 and (z_lo <= p <= z_hi):
        return "IN_ZONE", "IN ZONE"
    t_stage = (stages_dict.get("zone_touch") if isinstance(stages_dict, dict) else {}) or {}
    t_stage_st = str(t_stage.get("status") if isinstance(t_stage, dict) else t_stage or "").upper()
    t_cnt = int(z.get("touch_count", 0))
    app = str(z.get("approach", "")).upper()
    has_touched = (
        t_stage_st in ("CONFIRMED", "PASS", "PASSED")
        or app in ("TOUCHED", "ZONE_TOUCHED")
        or life_st in ("ZONE_TOUCHED", "TOUCHED", "CONFIRMED")
        or t_cnt > 0
    )
    if not has_touched and trk_rec:
        hist = trk_rec.get("history") or []
        has_touched = any(str(h.get("status", "")).upper() in ("ZONE_TOUCHED", "TOUCHED", "CONFIRMED") for h in hist)
        if not has_touched and str(trk_rec.get("touch_status", "")).upper() in ("IN_ZONE", "TOUCHED_PREVIOUSLY"):
            has_touched = True
    if has_touched:
        return "TOUCHED_PREVIOUSLY", "PREVIOUSLY TOUCHED"
    return "NOT_TOUCHED", "NOT TOUCHED"


def _smart_monitor_context(cycle_result):
    cycle = cycle_result if isinstance(cycle_result, dict) else {}
    snapshot = cycle.get("snapshot") if isinstance(cycle.get("snapshot"), dict) else {}
    smc = cycle.get("smc") if isinstance(cycle.get("smc"), dict) else {}
    snr = cycle.get("snr") if isinstance(cycle.get("snr"), dict) else {}
    scan = smc.get("scan_result") if isinstance(smc.get("scan_result"), dict) else {}
    decision = smc.get("decision") if isinstance(smc.get("decision"), dict) else {}
    summary = smc.get("summary") if isinstance(smc.get("summary"), dict) else {}
    snr_scan = snr.get("scan_result") if isinstance(snr.get("scan_result"), dict) else {}
    snr_decision = snr.get("decision") if isinstance(snr.get("decision"), dict) else {}

    def _zone_is_usable(candidate):
        if (
            not isinstance(candidate, dict)
            or candidate.get("is_valid") is False
            or candidate.get("is_active") is False
            or candidate.get("is_broken") is True
            or candidate.get("is_invalidated") is True
            or str(candidate.get("grade") or "").upper() == "REJECTED"
        ):
            return False
        status = str(candidate.get("lifecycle_status") or candidate.get("status") or "ACTIVE").upper()
        freshness_status = str(candidate.get("freshness") or candidate.get("freshness_status") or "FRESH").upper()
        rel_status = str(candidate.get("zone_relevance") or "").upper()
        return (
            rel_status != "INVALID"
            and status not in (
                "INVALIDATED", "EXPIRED", "MITIGATED", "BROKEN",
                "RECLAIMED", "RECLAIM_PENDING", "RETEST",
                "BREAKDOWN_PENDING", "BREAKOUT_PENDING", "HISTORICAL", "UNCONFIRMED",
            )
            and freshness_status not in ("INVALIDATED", "EXPIRED", "MITIGATED", "STALE", "INVALID")
        )

    price = safe_float(snapshot.get("snapshot_price", snapshot.get("bid")), 0.0)
    h4_bias_ctx = str(summary.get("h4_bias") or (scan.get("htf_context") or {}).get("H4") or "").upper()
    h1_bias_ctx = str(summary.get("h1_bias") or (scan.get("htf_context") or {}).get("H1") or "").upper()

    def _is_directional_for_htf(candidate):
        if h4_bias_ctx not in ("BULLISH", "BEARISH"):
            return True
        if candidate.get("is_counter"):
            return False
        zt = str(candidate.get("zone_type") or candidate.get("type") or "").upper()
        want = ("DEMAND", "SUPPORT") if h4_bias_ctx == "BULLISH" else ("SUPPLY", "RESISTANCE")
        return zt in want

    def _is_far_zone_ctx(candidate):
        if not isinstance(candidate, dict):
            return False
        if str(candidate.get("zone_relevance") or "").upper() == "FAR_REFERENCE" or candidate.get("is_far_reference"):
            return True
        if str(candidate.get("source") or "SMC").upper() != "SNR" and is_far_reference_zone(
            candidate, current_price=price if price > 0 else None
        ):
            return True
        if str(candidate.get("risk_tier") or "").upper() == "FAR_ZONE":
            return True
        if str(candidate.get("radar_eligibility") or "").upper() == "NOT_QUALIFIED":
            return True
        if candidate.get("is_distant_historical"):
            return True
        if str(candidate.get("lifecycle_status") or candidate.get("status") or "").upper() == "HISTORICAL":
            return True
        d_atr_snr = candidate.get("distance_atr")
        if isinstance(d_atr_snr, (int, float)) and d_atr_snr > 12.0:
            return True
        return False

    def _is_actionable_monitor_candidate(candidate):
        if not (
            _zone_is_usable(candidate)
            and not _is_far_zone_ctx(candidate)
            and _is_directional_for_htf(candidate)
        ):
            return False
        if candidate.get("risk_rejected") is True:
            return False
        s_elig_c = str(candidate.get("setup_eligibility") or candidate.get("pending_rejection_reason") or "").lower()
        if "jarak sl" in s_elig_c and "melebihi batas" in s_elig_c:
            return False
        return True

    far_reference_zone = None
    zone = _as_monitor_zone(decision.get("zone"))
    if zone and _is_far_zone_ctx(zone):
        far_reference_zone = dict(zone)
        zone = {}
    elif zone and not _is_actionable_monitor_candidate(zone):
        zone = {}
    if not zone or safe_float(zone.get("price_low", zone.get("low")), 0.0) <= 0:
        priority = decision.get("radar_priority") or scan.get("radar_priority") or {}
        smc_candidates = [
            decision.get("primary_zone"),
            priority.get("primary_directional_zone"),
            priority.get("primary_zone"),
            decision.get("best_radar_zone"),
            decision.get("nearest_valid_zone"),
            decision.get("next_radar_zone"),
            scan.get("primary_zone"),
            scan.get("nearest_zone"),
            (decision.get("active_setup") or {}).get("zone"),
            (decision.get("early_setup") or {}).get("best_setup"),
        ]
        for cand in smc_candidates:
            cz = _as_monitor_zone(cand)
            if cz and safe_float(cz.get("price_low", cz.get("low", cz.get("zone_low"))), 0.0) > 0 and _is_actionable_monitor_candidate(cz):
                zone = dict(cz)
                if "price_low" not in zone and "low" in zone:
                    zone["price_low"] = zone["low"]
                elif "price_low" not in zone and "zone_low" in zone:
                    zone["price_low"] = zone["zone_low"]
                if "price_high" not in zone and "high" in zone:
                    zone["price_high"] = zone["high"]
                elif "price_high" not in zone and "zone_high" in zone:
                    zone["price_high"] = zone["zone_high"]
                if "zone_type" not in zone and "type" in zone:
                    zone["zone_type"] = zone["type"]
                break

    lo = safe_float(zone.get("price_low", zone.get("low")), 0.0)
    hi = safe_float(zone.get("price_high", zone.get("high")), 0.0)
    if lo > hi > 0:
        lo, hi = hi, lo
    if price > 0 and lo > 0 and hi >= lo:
        if lo <= price <= hi:
            distance = 0.0
        elif lo > price:
            distance = max(round(lo - price, 2), 0.01)
        else:
            distance = max(round(price - hi, 2), 0.01)
    else:
        distance = None
    zone_type = str(zone.get("zone_type") or zone.get("type") or "").upper()
    direction = str(decision.get("direction") or summary.get("direction") or "").upper()
    if direction not in ("BUY", "SELL"):
        direction = "BUY" if zone_type in ("DEMAND", "SUPPORT") else ("SELL" if zone_type in ("SUPPLY", "RESISTANCE") else "WAIT")

    # Structured Setup Identity for Monitored Active Setup (Single Source of Truth)
    z_id = str(zone.get("zone_id") or "N/A")
    setup_type = str(decision.get("setup_type") or zone.get("setup_type") or "TREND_PULLBACK")
    now_str = format_wib_timestamp(default_now=True)
    active_setup = dict(decision.get("active_setup") or {})
    active_setup.update({
        "setup_id": active_setup.get("setup_id") or (f"{z_id}:{direction}:{setup_type}" if z_id != "N/A" else "N/A"),
        "zone_id": z_id,
        "direction": direction,
        "zone_range": (lo, hi),
        "setup_type": setup_type,
        "timeframe": str(zone.get("timeframe") or "N/A"),
        "grade": str(zone.get("grade") or "N/A"),
        "score": safe_float(zone.get("score")) if zone.get("score") is not None else None,
        "created_at": zone.get("created_at") or zone.get("first_detected") or now_str,
        "last_updated": zone.get("last_updated") or now_str,
        "lifecycle_status": str(zone.get("lifecycle_status") or zone.get("status") or "RADAR").upper(),
    })

    snr_mb = snr_scan.get("market_bias") if isinstance(snr_scan.get("market_bias"), dict) else {}
    snr_h4_bias = str(snr_mb.get("h4") or "").upper()
    snr_overall_bias = str(snr_mb.get("overall_bias") or "").upper()
    snr_bias = snr_h4_bias if snr_h4_bias in ("BULLISH", "BEARISH") else snr_overall_bias
    snr_zones = snr_scan.get("ranked_zones") or {}
    live_snr_sel = select_live_snr_focus(snr_zones, snr_mb, price)
    active_sups = [z for z in live_snr_sel["active_supports"] if _zone_is_usable(z)]
    active_ress = [z for z in live_snr_sel["active_resistances"] if _zone_is_usable(z)]
    raw_snr_target = (
        snr_decision.get("target_zone")
        or (snr_scan.get("entry_validation") or {}).get("target_zone")
        or snr_decision.get("trend_entry_zone")
        or {}
    )
    snr_zone = raw_snr_target
    def _same_snr_identity(z1, z2):
        if not isinstance(z1, dict) or not isinstance(z2, dict):
            return False
        if not z1.get("zone_id") or z2.get("zone_id") != z1.get("zone_id"):
            return False
        s1 = str(z1.get("source") or "SNR").strip().upper()
        s2 = str(z2.get("source") or "SNR").strip().upper()
        if s1 and s2 and s1 != s2:
            return False
        zt1 = str(z1.get("zone_type") or "").strip().upper()
        zt2 = str(z2.get("zone_type") or "").strip().upper()
        if zt1 and zt2 and zt1 != zt2:
            return False
        tf1 = str(z1.get("timeframe") or "").strip().upper()
        tf2 = str(z2.get("timeframe") or "").strip().upper()
        if tf1 and tf2 and tf1 != tf2:
            return False
        lo1 = safe_float(z1.get("low", z1.get("price_low")), 0.0)
        hi1 = safe_float(z1.get("high", z1.get("price_high")), 0.0)
        lo2 = safe_float(z2.get("low", z2.get("price_low")), 0.0)
        hi2 = safe_float(z2.get("high", z2.get("price_high")), 0.0)
        if lo1 > 0 and hi1 >= lo1 and lo2 > 0 and hi2 >= lo2:
            if abs(lo1 - lo2) >= 1e-6 or abs(hi1 - hi2) >= 1e-6:
                return False
        for ts_key in ("confirmed_time", "confirmation_time", "formation_time"):
            v1, v2 = z1.get(ts_key), z2.get(ts_key)
            if v1 and v2 and str(v1) != str(v2):
                return False
        return True

    if isinstance(snr_zone, dict) and snr_zone.get("zone_id"):
        snr_pool = (
            list(snr_zones.get("active_supports") or [])
            + list(snr_zones.get("active_resistances") or [])
            + list(snr_zones.get("pending_break_zones") or [])
            + list(snr_zones.get("broken_zones") or [])
            + list(snr_zones.get("reclaimed_zones") or [])
            + list(snr_zones.get("historical_zones") or [])
            + list(snr_zones.get("all_zones") or [])
        )
        matched_snr = next((cand for cand in snr_pool if _same_snr_identity(snr_zone, cand)), None)
        if matched_snr is not None:
            enriched_snr = dict(snr_zone)
            for meta_key in (
                "grade", "score", "rank_label", "zone_type", "timeframe", "source",
                "confirmed_time", "confirmation_time", "formation_time", "risk_tier", "distance", "distance_points",
                "lifecycle_status", "status", "is_broken", "is_invalidated", "is_active", "breakout_info",
            ):
                if meta_key in matched_snr and matched_snr.get(meta_key) is not None:
                    enriched_snr[meta_key] = matched_snr.get(meta_key)
            snr_zone = enriched_snr

    snr_live_sig_pre = str(snr_decision.get("signal") or "WAIT").upper()
    live_focus_cand = live_snr_sel.get("focus_zone")
    live_focus_usable = bool(isinstance(live_focus_cand, dict) and _zone_is_usable(live_focus_cand))

    def _snr_side_valid_for_live_price(cand_z):
        if not _zone_is_usable(cand_z):
            return False
        rk = str(cand_z.get("rank_label") or "").upper()
        zt = str(
            cand_z.get("zone_type")
            or ("RESISTANCE" if rk.startswith("R") else ("SUPPORT" if rk.startswith("S") else ""))
        ).upper()
        if zt in ("SUPPORT", "RESISTANCE"):
            return _is_live_active_snr_zone(cand_z, zt, price)
        return True

    snr_overridden = False
    # Pertahankan target_zone jika SNR sedang menerbitkan sinyal entry live terkonfirmasi (BUY/SELL) pada zona valid;
    # selain itu gunakan satu sumber kebenaran `select_live_snr_focus` agar konsisten dengan `scan snr` & harga live.
    if snr_live_sig_pre in ("BUY", "SELL") and _snr_side_valid_for_live_price(snr_zone):
        pass
    elif live_focus_usable:
        if not isinstance(snr_zone, dict) or snr_zone.get("zone_id") != live_focus_cand.get("zone_id"):
            snr_overridden = bool(raw_snr_target)
        snr_zone = dict(live_focus_cand)
    elif not _snr_side_valid_for_live_price(snr_zone) or (
        isinstance(snr_zone, dict) and snr_zone.get("zone_id")
        and str(snr_zone.get("zone_id")) not in {
            str(z_a.get("zone_id")) for z_a in (active_sups + active_ress) if isinstance(z_a, dict)
        }
    ):
        # Zona harus ada di daftar aktif live (sama dengan `scan snr`); target keputusan lama bukan S/R aktif.
        snr_zone = {}
        snr_overridden = bool(raw_snr_target)

    # Propagasikan status breakout SNR kanonis (sinkron penuh dengan `scan snr`:
    # BREAKDOWN/BREAKOUT PENDING, CONFIRMED, RETEST WATCH, BROKEN) ke Smart Monitor
    snr_breakout_watch = ""
    snr_historical_watch = ""
    historical_events_meta = []
    if snr.get("status") == "OK":
        bo_watch_snr = snr_scan.get("breakout_watch") or build_snr_breakout_watch(
            snr_zones, price_action=snr_scan.get("price_action")
        )
        snr_breakout_watch = str(bo_watch_snr.get("summary_line") or "").strip()

        recl_pool = [z for z in (snr_zones.get("reclaimed_zones") or []) if isinstance(z, dict)]
        recl_hist = [
            z for z in recl_pool
            if (
                z.get("lifecycle_is_historical")
                or z.get("is_distant_historical")
                or safe_float(z.get("distance_atr"), 0.0) > 12.0
                or (str(z.get("lifecycle_status") or z.get("status") or "").upper() == "RECLAIMED" and int(safe_float(z.get("lifecycle_bars_ago"), 0)) > 1)
                or bool((z.get("breakout_info") or {}).get("is_historical_break"))
            )
        ]
        recl_hist.sort(
            key=lambda item: (
                str(item.get("lifecycle_event_time") or (item.get("breakout_info") or {}).get("breakout_bar_time") or ""),
                str(item.get("zone_id") or ""),
            ),
            reverse=True,
        )
        for rh in recl_hist[:5]:
            rh_st = str(rh.get("lifecycle_status") or "RECLAIMED").upper()
            rh_zt = str(rh.get("zone_type") or "").upper()
            rh_ts = str(rh.get("lifecycle_event_time") or (rh.get("breakout_info") or {}).get("breakout_bar_time") or "").strip()
            rh_ts_short = rh_ts[:16] if len(rh_ts) >= 16 else rh_ts
            rh_ts_part = f" | {rh_ts_short}" if rh_ts_short else ""
            rh_lbl = f"SNR {rh_st.replace('_', ' ')} HISTORICAL ({rh.get('timeframe', '')} {rh_zt} {safe_float(rh.get('low')):.2f}-{safe_float(rh.get('high')):.2f}{rh_ts_part})".replace("  ", " ")
            historical_events_meta.append({
                "zone_id": rh.get("zone_id"),
                "timeframe": rh.get("timeframe"),
                "zone_type": rh_zt,
                "event_type": str(rh.get("lifecycle_event") or rh_st).upper(),
                "lifecycle_status": rh_st,
                "timestamp": rh_ts or None,
                "label": rh_lbl,
            })
        # Pertahankan event historis SNR yang relevan (mis. rejection / breakout / retest lama)
        # terpisah dari current alerts, terdedup dengan (zone_id, event_type, timestamp).
        seen_hist_keys = {
            (
                str(ev.get("zone_id") or "").upper(),
                str(ev.get("event_type") or "").upper(),
                str(ev.get("timestamp") or "").strip(),
            )
            for ev in historical_events_meta
        }
        snr_pa_hist = snr_scan.get("price_action") if isinstance(snr_scan.get("price_action"), dict) else {}
        rej_det_h = snr_pa_hist.get("rejection_detail") if isinstance(snr_pa_hist.get("rejection_detail"), dict) else {}
        if rej_det_h.get("confirmed") and bool(snr_pa_hist.get("rejection_is_historical") or rej_det_h.get("is_historical")):
            r_ev_type = str(rej_det_h.get("status") or snr_pa_hist.get("rejection_code") or "REJECTION").upper()
            r_ts = str(rej_det_h.get("bar_time") or "").strip()
            r_zid = rej_det_h.get("zone_id")
            r_key = (str(r_zid or "").upper(), r_ev_type, r_ts)
            if r_key not in seen_hist_keys:
                seen_hist_keys.add(r_key)
                r_ts_short = r_ts[:16] if len(r_ts) >= 16 else r_ts
                r_z_lbl = str(rej_det_h.get("zone_label") or snr_pa_hist.get("rejection_zone_label") or "SNR").strip()
                r_ts_part = f" | {r_ts_short}" if r_ts_short else ""
                historical_events_meta.append({
                    "zone_id": r_zid,
                    "timeframe": rej_det_h.get("timeframe"),
                    "zone_type": str(rej_det_h.get("zone_type") or "").upper(),
                    "event_type": r_ev_type,
                    "lifecycle_status": "HISTORICAL",
                    "timestamp": r_ts or None,
                    "label": f"SNR {r_ev_type.replace('_', ' ')} HISTORICAL ({r_z_lbl}{r_ts_part})",
                })
        for hev in (snr.get("historical_events") or []):
            if not isinstance(hev, dict):
                continue
            h_zid = hev.get("zone_id")
            h_ev = str(hev.get("event_type") or "HISTORICAL").upper()
            h_ts = str(hev.get("timestamp") or "").strip()
            h_key = (str(h_zid or "").upper(), h_ev, h_ts)
            if h_key in seen_hist_keys:
                continue
            seen_hist_keys.add(h_key)
            h_msg = str(hev.get("message") or "").strip()
            if h_msg:
                historical_events_meta.append({
                    "zone_id": h_zid,
                    "timeframe": None,
                    "zone_type": None,
                    "event_type": h_ev,
                    "lifecycle_status": "HISTORICAL",
                    "timestamp": h_ts or None,
                    "label": h_msg,
                })
        if historical_events_meta:
            snr_historical_watch = historical_events_meta[0]["label"]
    if not snr_zone and snr_bias == "BEARISH":
        if active_ress:
            snr_zone = dict(active_ress[0])
            snr_overridden = True
    elif not snr_zone and snr_bias == "BULLISH":
        if active_sups:
            snr_zone = dict(active_sups[0])
            snr_overridden = True
    if not snr_zone and (active_sups or active_ress):
        pool_fb = active_sups + active_ress
        if price > 0:
            snr_zone = dict(min(
                pool_fb,
                key=lambda z: (
                    0.0 if safe_float(z.get("low"), 0.0) <= price <= safe_float(z.get("high"), 0.0)
                    else min(abs(price - safe_float(z.get("low"), price)), abs(price - safe_float(z.get("high"), price)))
                ),
            ))
        else:
            snr_zone = dict(pool_fb[0])
        snr_overridden = True

    snr_zt_eff = str(snr_zone.get("zone_type") or "").upper()
    snr_rk_eff = str(snr_zone.get("rank_label") or "").upper()
    snr_zone_direction = (
        str(snr_decision.get("signal")).upper() if snr_decision.get("signal") in ("BUY", "SELL") and not snr_overridden
        else "SELL" if snr_zt_eff == "RESISTANCE" or snr_rk_eff.startswith("R")
        else "BUY" if snr_zt_eff == "SUPPORT" or snr_rk_eff.startswith("S")
        else ("SELL" if snr_zone and any(snr_zone.get("zone_id") == r.get("zone_id") and r.get("zone_id") for r in active_ress) or snr_zone in active_ress
              else "BUY" if snr_zone and (any(snr_zone.get("zone_id") == s.get("zone_id") and s.get("zone_id") for s in active_sups) or snr_zone in active_sups)
              else "NONE")
    )

    snr_lo_sel = safe_float(snr_zone.get("low", snr_zone.get("price_low")), 0.0)
    snr_hi_sel = safe_float(snr_zone.get("high", snr_zone.get("price_high")), 0.0)
    snr_inside_sel = bool(price > 0 and snr_lo_sel > 0 and snr_lo_sel <= price <= snr_hi_sel)
    raw_sel_reason = "" if snr_overridden else str(
        snr_decision.get("focus_selection_reason")
        or (snr_scan.get("entry_validation") or {}).get("focus_selection_reason")
        or ""
    ).strip()
    if snr_bias == "BEARISH" and snr_zone_direction == "BUY":
        snr_reason_short = "PRICE AT SUPPORT" if snr_inside_sel else "NO ACTIVE RESISTANCE"
    elif snr_bias == "BULLISH" and snr_zone_direction == "SELL":
        snr_reason_short = "PRICE AT RESISTANCE" if snr_inside_sel else "NO ACTIVE SUPPORT"
    elif snr_zone_direction in ("BUY", "SELL"):
        snr_reason_short = (
            "TREND ALIGNED"
            if (snr_bias == "BEARISH" and snr_zone_direction == "SELL") or (snr_bias == "BULLISH" and snr_zone_direction == "BUY")
            else "NEAREST ACTIVE SNR"
        )
    else:
        snr_reason_short = "NONE"
    snr_selection_reason = raw_sel_reason or (live_snr_sel.get("focus_selection_reason") if live_focus_usable else "") or snr_reason_short

    # Preserve independent SMC candidate reference before any SNR display fallback
    smc_zone_ref = dict(zone) if (isinstance(zone, dict) and safe_float(zone.get("price_low", zone.get("low")), 0.0) > 0 and str(zone.get("source") or "SMC").upper() != "SNR") else {}
    # If SMC zone was not found, fall back to directional SNR zone for active setup display
    if (not zone or safe_float(zone.get("price_low", zone.get("low")), 0.0) <= 0) and snr_zone and _is_actionable_monitor_candidate(snr_zone):
        s_lo = safe_float(snr_zone.get("low", snr_zone.get("price_low")), 0.0)
        s_hi = safe_float(snr_zone.get("high", snr_zone.get("price_high")), 0.0)
        if s_lo > 0 and s_hi > 0:
            s_zt = str(snr_zone.get("zone_type") or ("RESISTANCE" if snr_zone_direction == "SELL" else "SUPPORT")).upper()
            snr_raw_score = snr_zone.get("score")
            snr_raw_dist = snr_zone.get("distance") if snr_zone.get("distance") is not None else snr_zone.get("distance_points")
            zone = {
                "zone_id": snr_zone.get("zone_id") or "N/A",
                "timeframe": snr_zone.get("timeframe") or "N/A",
                "zone_type": s_zt,
                "price_low": s_lo,
                "price_high": s_hi,
                "grade": snr_zone.get("grade") or "N/A",
                "score": safe_float(snr_raw_score) if snr_raw_score not in (None, "") else None,
                "status": str(snr_zone.get("lifecycle_status") or snr_zone.get("status") or "RADAR").upper(),
                "source": "SNR",
                "rank_label": str(snr_zone.get("rank_label") or "").strip(),
                "risk_tier": snr_zone.get("risk_tier") or "N/A",
                "distance": safe_float(snr_raw_dist) if snr_raw_dist not in (None, "") else None,
                "confirmed_time": snr_zone.get("confirmed_time") or snr_zone.get("confirmation_time") or snr_zone.get("formation_time"),
            }
            lo, hi = s_lo, s_hi
            distance = 0.0 if lo <= price <= hi and price > 0 else (
                lo - price if price > 0 and lo > price else (price - hi if price > hi else None)
            )
            zone_type = zone["zone_type"]
            direction = "SELL" if s_zt == "RESISTANCE" else "BUY"
            z_id = str(zone.get("zone_id") or "N/A")
            active_setup.update({
                "setup_id": f"{z_id}:{direction}:{setup_type}",
                "zone_id": z_id,
                "direction": direction,
                "zone_range": (lo, hi),
                "timeframe": str(zone.get("timeframe") or "N/A"),
                "grade": str(zone.get("grade") or "N/A"),
                "score": safe_float(zone.get("score")) if zone.get("score") is not None else None,
            })

    if (not zone or safe_float(zone.get("price_low", zone.get("low")), 0.0) <= 0) and far_reference_zone is None:
        dec_far = decision.get("far_reference_zone")
        if isinstance(dec_far, dict) and safe_float(dec_far.get("price_low", dec_far.get("low")), 0.0) > 0:
            far_reference_zone = dict(dec_far)
        else:
            far_smc_pool = [
                z_f for z_f in (scan.get("radar_zones") or [])
                if isinstance(z_f, dict)
                and _zone_is_usable(z_f)
                and _is_far_zone_ctx(z_f)
                and _is_directional_for_htf(z_f)
            ]
            if far_smc_pool:
                far_reference_zone = dict(min(
                    far_smc_pool,
                    key=lambda z_f: max(safe_float(z_f.get("price_low"), 0.0) - price, price - safe_float(z_f.get("price_high"), 0.0), 0.0) if price > 0 else safe_float(z_f.get("distance"), 9999.0),
                ))
            elif isinstance(snr_zone, dict) and _is_far_zone_ctx(snr_zone) and safe_float(snr_zone.get("low", snr_zone.get("price_low")), 0.0) > 0:
                far_reference_zone = dict(snr_zone)
                far_reference_zone.setdefault("source", "SNR")
            else:
                snr_hist = [
                    z_h for z_h in (
                        list(snr_zones.get("far_reference_zones") or [])
                        + list(snr_zones.get("historical_zones") or [])
                    )
                    if isinstance(z_h, dict)
                ]
                if snr_hist:
                    far_reference_zone = dict(snr_hist[0])
                    far_reference_zone.setdefault("source", "SNR")

    smc_zone_usable = _zone_is_usable(smc_zone_ref) if smc_zone_ref else (zone.get("source") != "SNR" and _zone_is_usable(zone))
    snr_zone_usable = _zone_is_usable(snr_zone)
    smc_comp_lo = safe_float(smc_zone_ref.get("price_low", smc_zone_ref.get("low")), 0.0) if smc_zone_ref else (lo if zone.get("source") != "SNR" else 0.0)
    smc_comp_hi = safe_float(smc_zone_ref.get("price_high", smc_zone_ref.get("high")), 0.0) if smc_zone_ref else (hi if zone.get("source") != "SNR" else 0.0)
    smc_comp_zt = str((smc_zone_ref or zone).get("zone_type") or (smc_zone_ref or zone).get("type") or "").upper()
    smc_comp_dir = "BUY" if smc_comp_zt in ("DEMAND", "SUPPORT") else ("SELL" if smc_comp_zt in ("SUPPLY", "RESISTANCE") else direction)
    priority_ctx = decision.get("radar_priority") or scan.get("radar_priority") or {}
    expected_h4_dir = "BUY" if h4_bias_ctx == "BULLISH" else ("SELL" if h4_bias_ctx == "BEARISH" else "")
    expected_h1_dir = "BUY" if h1_bias_ctx == "BULLISH" else ("SELL" if h1_bias_ctx == "BEARISH" else "")
    is_counter_htf = bool(
        direction in ("BUY", "SELL")
        and (
            (expected_h4_dir and direction != expected_h4_dir)
            or (expected_h1_dir and direction != expected_h1_dir)
        )
    )

    # Pisahkan directional confluence terhadap zona entry SMC vs lokasi harga live di support/resistance terdekat.
    # Pisahkan tegas SNR FOCUS (snr_zone: zona terdekat harga) dari SNR CONFLUENCE SOURCE (zona directional yang benar-benar overlap SMC).
    snr_live_signal = str(snr_decision.get("signal") or "WAIT").upper()
    directional_snr_zone = None
    snr_confluence_source = None
    snr_confluence_overlap_pts = 0.0
    snr_confluence_source_display = "NONE"
    snr_price_proximity = ""
    confluence_display = ""

    def _zones_overlap_points(lo_a, hi_a, lo_b, hi_b):
        if lo_a > hi_a and hi_a > 0:
            lo_a, hi_a = hi_a, lo_a
        if lo_b > hi_b and hi_b > 0:
            lo_b, hi_b = hi_b, lo_b
        if not (lo_a > 0 and hi_a >= lo_a and lo_b > 0 and hi_b >= lo_b):
            return 0.0
        ov = round(min(hi_a, hi_b) - max(lo_a, lo_b), 2)
        return ov if ov > 0.0 else 0.0

    def _snr_pool_key(cand_z):
        zid_c = str(cand_z.get("zone_id") or "").strip().upper()
        lo_c = round(safe_float(cand_z.get("low", cand_z.get("price_low")), 0.0), 4)
        hi_c = round(safe_float(cand_z.get("high", cand_z.get("price_high")), 0.0), 4)
        return (zid_c, lo_c, hi_c)

    if smc_zone_usable and smc_comp_dir in ("BUY", "SELL"):
        dir_pool = []
        seen_dir_keys = set()
        for cand in list(active_ress if smc_comp_dir == "SELL" else active_sups):
            if not _zone_is_usable(cand):
                continue
            ck = _snr_pool_key(cand)
            if ck not in seen_dir_keys:
                seen_dir_keys.add(ck)
                dir_pool.append(cand)
        opp_pool = [z for z in (active_sups if smc_comp_dir == "SELL" else active_ress) if _zone_is_usable(z)]
        trend_z_cand = live_snr_sel.get("trend_entry_zone") or snr_decision.get("trend_entry_zone")
        if isinstance(trend_z_cand, dict) and _snr_side_valid_for_live_price(trend_z_cand):
            tz_type = str(trend_z_cand.get("zone_type") or "").upper()
            want_tz = "RESISTANCE" if smc_comp_dir == "SELL" else "SUPPORT"
            if tz_type == want_tz:
                ck = _snr_pool_key(trend_z_cand)
                if ck not in seen_dir_keys:
                    seen_dir_keys.add(ck)
                    dir_pool.insert(0, trend_z_cand)
        if snr_zone_usable and snr_zone_direction == smc_comp_dir and _snr_side_valid_for_live_price(snr_zone):
            ck = _snr_pool_key(snr_zone)
            if ck not in seen_dir_keys:
                seen_dir_keys.add(ck)
                dir_pool.insert(0, snr_zone)

        overlapping_dir = None
        best_dir_key = None
        for idx_c, cand in enumerate(dir_pool):
            c_lo = safe_float(cand.get("low", cand.get("price_low")), 0.0)
            c_hi = safe_float(cand.get("high", cand.get("price_high")), 0.0)
            ov_pts = _zones_overlap_points(smc_comp_lo, smc_comp_hi, c_lo, c_hi)
            if ov_pts > 0.0:
                rank_key = (ov_pts, safe_float(cand.get("score"), 0.0), -idx_c)
                if best_dir_key is None or rank_key > best_dir_key:
                    best_dir_key = rank_key
                    overlapping_dir = cand
                    snr_confluence_overlap_pts = ov_pts
        snr_confluence_source = overlapping_dir
        directional_snr_zone = snr_confluence_source or (dir_pool[0] if dir_pool else None)

        overlapping_opp = None
        opp_overlap_pts = 0.0
        for cand in opp_pool:
            o_lo_c = safe_float(cand.get("low", cand.get("price_low")), 0.0)
            o_hi_c = safe_float(cand.get("high", cand.get("price_high")), 0.0)
            ov_opp = _zones_overlap_points(smc_comp_lo, smc_comp_hi, o_lo_c, o_hi_c)
            if ov_opp > opp_overlap_pts:
                opp_overlap_pts = ov_opp
                overlapping_opp = cand
        near_opp = opp_pool[0] if opp_pool else None
        price_at_opp = bool(
            near_opp and price > 0
            and safe_float(near_opp.get("low", near_opp.get("price_low")), 0.0) <= price <= safe_float(near_opp.get("high", near_opp.get("price_high")), 0.0)
        )
        if not price_at_opp and snr_zone_usable and snr_zone_direction != smc_comp_dir and snr_inside_sel:
            near_opp = snr_zone
            price_at_opp = True
        if price_at_opp and near_opp:
            opp_lbl = str(near_opp.get("rank_label") or ("S1" if smc_comp_dir == "SELL" else "R1")).strip()
            opp_lo = safe_float(near_opp.get("low", near_opp.get("price_low")), 0.0)
            opp_hi = safe_float(near_opp.get("high", near_opp.get("price_high")), 0.0)
            opp_kind = "Support" if smc_comp_dir == "SELL" else "Resistance"
            snr_price_proximity = f"PRICE AT {opp_lbl}"
            snr_proximity_detail = f"harga saat ini di {opp_kind} terdekat {opp_lbl} ({opp_lo:.2f}-{opp_hi:.2f})"
        else:
            snr_proximity_detail = ""
    else:
        overlapping_opp = None
        opp_overlap_pts = 0.0
        snr_proximity_detail = ""

    if smc.get("status") != "OK" or snr.get("status") != "OK":
        confluence = "UNAVAILABLE"
        confluence_reason = "Salah satu scanner (SMC/SNR) tidak berstatus OK"
    elif not smc_zone_usable and not snr_zone_usable:
        confluence = "UNAVAILABLE"
        confluence_reason = "Belum ada zona aktif SMC maupun SNR"
    elif is_counter_htf:
        confluence = "COUNTERTREND"
        confluence_reason = f"Setup {direction} berlawanan dengan bias HTF (H4={h4_bias_ctx or 'N/A'}, H1={h1_bias_ctx or 'N/A'})"
    elif not smc_zone_usable or smc_comp_dir not in ("BUY", "SELL"):
        confluence = "PARTIAL"
        confluence_reason = "Hanya satu sisi zona aktif valid (SMC tidak lengkap)"
    elif snr_live_signal in ("BUY", "SELL") and snr_live_signal != smc_comp_dir:
        confluence = "CONFLICT"
        confluence_reason = f"Sinyal live SNR ({snr_live_signal}) bertentangan dengan setup SMC ({smc_comp_dir})"
    elif overlapping_opp is not None:
        o_lbl = str(overlapping_opp.get("rank_label") or overlapping_opp.get("zone_type") or "SNR").strip()
        o_lo = safe_float(overlapping_opp.get("low", overlapping_opp.get("price_low")), 0.0)
        o_hi = safe_float(overlapping_opp.get("high", overlapping_opp.get("price_high")), 0.0)
        confluence = "CONFLICT"
        confluence_reason = (
            f"Zona entry SMC {smc_comp_dir} ({smc_comp_lo:.2f}-{smc_comp_hi:.2f}) overlap langsung "
            f"dengan zona SNR lawan {o_lbl} ({o_lo:.2f}-{o_hi:.2f}, {opp_overlap_pts:.2f} USD)"
        )
        snr_confluence_source_display = f"CONFLICT ({o_lbl} {o_lo:.2f}-{o_hi:.2f} overlap lawan {opp_overlap_pts:.2f} USD)"
    elif (
        snr_zone_usable
        and snr_zone_direction in ("BUY", "SELL")
        and snr_zone_direction != smc_comp_dir
        and (
            (smc_comp_dir == "SELL" and snr_bias == "BULLISH")
            or (smc_comp_dir == "BUY" and snr_bias == "BEARISH")
        )
    ):
        confluence = "CONFLICT"
        confluence_reason = f"Bias & fokus SNR ({snr_bias} / {snr_zone_direction}) bertentangan dengan setup SMC ({smc_comp_dir})"
    elif snr_confluence_source is not None:
        d_lo = safe_float(snr_confluence_source.get("low", snr_confluence_source.get("price_low")), 0.0)
        d_hi = safe_float(snr_confluence_source.get("high", snr_confluence_source.get("price_high")), 0.0)
        d_lbl = str(snr_confluence_source.get("rank_label") or snr_confluence_source.get("zone_type") or "SNR").strip()
        d_tf = str(snr_confluence_source.get("timeframe") or "").strip()
        d_grd = str(snr_confluence_source.get("grade") or "N/A").strip()
        d_tag = f"{d_lbl} {d_tf}".strip()
        confluence = "ALIGNED"
        snr_confluence_source_display = f"{d_tag} {d_lo:.2f} - {d_hi:.2f} (Grade {d_grd} | overlap {snr_confluence_overlap_pts:.2f} USD)"
        base_c_reason = (
            f"Directional {smc_comp_dir}: zona SMC {smc_comp_lo:.2f}-{smc_comp_hi:.2f} "
            f"overlap dengan SNR {d_lbl} {d_lo:.2f}-{d_hi:.2f} ({snr_confluence_overlap_pts:.2f} USD)"
        )
        if snr_proximity_detail:
            confluence_reason = f"{base_c_reason}; {snr_proximity_detail} (bukan konflik setup {smc_comp_dir})"
            confluence_display = f"{confluence} ({d_lbl} {d_lo:.2f}-{d_hi:.2f} overlap {snr_confluence_overlap_pts:.2f} USD | {snr_price_proximity})"
        else:
            confluence_reason = base_c_reason
            confluence_display = f"{confluence} ({d_lbl} {d_lo:.2f}-{d_hi:.2f} overlap {snr_confluence_overlap_pts:.2f} USD)"
    elif directional_snr_zone is not None:
        d_lo = safe_float(directional_snr_zone.get("low", directional_snr_zone.get("price_low")), 0.0)
        d_hi = safe_float(directional_snr_zone.get("high", directional_snr_zone.get("price_high")), 0.0)
        d_lbl = str(directional_snr_zone.get("rank_label") or directional_snr_zone.get("zone_type") or "SNR").strip()
        gap_pts = (smc_comp_lo - d_hi) if smc_comp_lo > d_hi else max(0.0, d_lo - smc_comp_hi)
        confluence = "PARTIAL"
        snr_confluence_source_display = f"NONE (searah terdekat {d_lbl} {d_lo:.2f}-{d_hi:.2f} tidak overlap SMC, jarak {gap_pts:.2f} USD)"
        base_c_reason = (
            f"Directional {smc_comp_dir}: searah SNR {d_lbl} {d_lo:.2f}-{d_hi:.2f}, "
            f"rentang SMC {smc_comp_lo:.2f}-{smc_comp_hi:.2f} tidak overlap (jarak {gap_pts:.2f} USD)"
        )
        if snr_proximity_detail:
            confluence_reason = f"{base_c_reason}; {snr_proximity_detail} (bukan konflik setup {smc_comp_dir})"
            confluence_display = f"{confluence} (no overlap vs {d_lbl} {d_lo:.2f}-{d_hi:.2f} | {snr_price_proximity})"
        else:
            confluence_reason = base_c_reason
            confluence_display = f"{confluence} (no overlap vs {d_lbl} {d_lo:.2f}-{d_hi:.2f})"
    elif snr_proximity_detail:
        confluence = "PARTIAL"
        opp_sig = "BUY" if smc_comp_dir == "SELL" else "SELL"
        confluence_reason = (
            f"Tidak ada konflik {smc_comp_dir}: {snr_proximity_detail} tanpa sinyal {opp_sig} terkonfirmasi; "
            f"belum ada zona SNR aktif searah {smc_comp_dir}"
        )
        confluence_display = f"{confluence} (NO DIR. SNR | {snr_price_proximity})"
    else:
        confluence = "PARTIAL"
        confluence_reason = "Hanya satu sisi zona aktif valid (SMC atau SNR tidak lengkap)"

    if not confluence_display:
        confluence_display = confluence

    tracker = decision.get("zone_tracker")
    tracked = tracker.get_zone(str(zone.get("zone_id"))) if tracker and zone.get("zone_id") and hasattr(tracker, "get_zone") else None
    lifecycle = str((tracked or {}).get("lifecycle_status") or (tracked or {}).get("status") or active_setup["lifecycle_status"]).upper()
    if not lifecycle:
        lifecycle = str(zone.get("approach") or zone.get("status") or "RADAR").upper()
    state_map = {
        "RADAR": "IDLE",
        "READY": "IDLE",
        "DETECTED": "IDLE",
        "APPROACHING": "APPROACHING",
        "ZONE_TOUCHED": "IN_ZONE",
        "TOUCHED": "IN_ZONE",
        "CONFIRMED": "CONFIRMED",
        "INVALIDATED": "INVALIDATED",
        "EXPIRED": "EXPIRED",
    }
    monitor_state = state_map.get(lifecycle, lifecycle)
    if lo > 0 and hi >= lo and price > 0 and monitor_state not in ("INVALIDATED", "EXPIRED", "CONFIRMED"):
        if lo <= price <= hi:
            monitor_state = "AWAITING_CONFIRMATION"
        elif distance is not None and distance <= float(ZONE_APPROACH_DISTANCE_POINTS):
            monitor_state = "APPROACHING"
        elif monitor_state in ("APPROACHING", "AWAITING_CONFIRMATION", "IN_ZONE", "ZONE_TOUCHED"):
            monitor_state = "IDLE"
    elif monitor_state == "IN_ZONE":
        # Jika harga sudah berada di luar zona dan tidak ada trigger aktif, jangan paksakan AWAITING_CONFIRMATION
        if lo > 0 and hi >= lo and price > 0 and not (lo <= price <= hi):
            monitor_state = "IDLE"
        else:
            monitor_state = "AWAITING_CONFIRMATION"

    raw_stages = decision.get("m5_stages") or decision.get("stages") or {}
    m5_bound = _m5_stages_match_zone(decision, zone)
    stages = raw_stages if m5_bound else {}
    touch_status, touch_display = _calc_zone_touch(zone, price, stages, tracked, lifecycle)

    if m5_bound:
        trigger_stage = stages.get("structure_trigger") or {}
        trigger = str(
            (trigger_stage.get("status") if isinstance(trigger_stage, dict) else trigger_stage)
            or summary.get("choch_bos") or "WAIT"
        ).upper()
        raw_m5_conf = decision.get("m5_confirmation") if isinstance(decision.get("m5_confirmation"), dict) else None
        if raw_m5_conf and str(raw_m5_conf.get("zone_id") or "") == str(zone.get("zone_id") or ""):
            m5_confirmation = dict(raw_m5_conf)
        elif trigger == "CONFIRMED":
            m5_obj_ref = decision.get("m5") if isinstance(decision.get("m5"), dict) else {}
            m5_confirmation = {
                "zone_id": zone.get("zone_id"),
                "timeframe": zone.get("timeframe"),
                "zone_type": zone.get("zone_type"),
                "zone_range": (lo, hi),
                "timestamp": (
                    (trigger_stage.get("timestamp") if isinstance(trigger_stage, dict) else None)
                    or m5_obj_ref.get("timestamp")
                    or m5_obj_ref.get("confirmed_at")
                ),
                "status": "CONFIRMED",
            }
        else:
            m5_confirmation = None
    else:
        trigger = "NOT_CHECKED"
        m5_confirmation = None

    snapshot_status = str(snapshot.get("snapshot_status") or snapshot.get("data_status") or "UNKNOWN").upper()
    freshness = str(snapshot.get("data_freshness") or (snapshot.get("data_freshness_info") or {}).get("status") or "UNKNOWN").upper()
    data_invalid = (
        snapshot_status in ("DATA_INCOMPLETE", "INVALID", "ERROR", "MARKET_CLOSED")
        or freshness in ("STALE", "INVALID", "NO DATA")
        or snapshot.get("data_ok") is False
        or price <= 0
        or smc.get("status") == "ERROR"
    )
    validation_price = safe_float(decision.get("snapshot_price") or decision.get("current_price"), 0.0)
    is_tick_only = cycle.get("focus_validation_fresh") is False
    needs_revalidation = (not is_tick_only) and validation_price > 0 and abs(validation_price - price) > 1e-8
    if needs_revalidation:
        data_invalid = True

    # 1. Authoritative Pending Plan Resolution (single source of truth per monitor cycle)
    completed_smc_id = ((cycle.get("completed_focus") or {}).get("SMC") or {}).get("zone_id")
    acc_ctx = cycle.get("account") if isinstance(cycle.get("account"), dict) else {}
    mt5_ord_snap = (
        acc_ctx.get("orders_snapshot")
        if acc_ctx.get("orders_snapshot") is not None
        else snapshot.get("mt5_orders_snapshot")
    )
    sym_ctx = snapshot.get("symbol") or acc_ctx.get("symbol") or scan.get("symbol")
    pending_res = resolve_monitor_pending_plans(
        decision=decision,
        scan=scan,
        active_zone=zone,
        tracker=tracker,
        data_invalid=data_invalid,
        completed_zone_id=completed_smc_id,
        zone_usable_fn=_zone_is_usable,
        mt5_orders_snapshot=mt5_ord_snap,
        symbol=sym_ctx,
    )
    plan = pending_res["selected_plan"]
    pending_setup = pending_res["pending_setup"]
    alt_pending_setups = pending_res["alt_pending_setups"]
    p_status = pending_res["pending_status"]
    p_zid = str(pending_setup.get("zone_id") or "")
    if p_zid == "N/A":
        p_zid = ""
    pending_display = _format_pending_compact_tag(pending_setup, z_id, is_selected=True)
    alt_pending_display = ""
    if alt_pending_setups:
        alt_first = _format_pending_compact_tag(alt_pending_setups[0], z_id, is_selected=False)
        if len(alt_pending_setups) > 1:
            alt_first = f"{alt_first} (+{len(alt_pending_setups) - 1})"
        alt_pending_display = alt_first

    # Radar Candidate: alternatif intraday yang BUKAN Active Focus dan BUKAN Selected Pending Plan
    aligned_ctx = sorted(
        (priority_ctx.get("aligned_zones") or []),
        key=lambda item: safe_float((item or {}).get("distance"), 9999.0),
    )
    radar_pool = [
        decision.get("nearest_valid_zone"),
        priority_ctx.get("best_high_risk"),
        priority_ctx.get("nearest_radar_zone"),
        priority_ctx.get("best_medium_risk"),
        decision.get("next_radar_zone"),
        *aligned_ctx,
    ]
    radar_candidate = None
    if not data_invalid:
        for cand in radar_pool:
            cz = _as_monitor_zone(cand)
            if not cz or not _zone_is_usable(cz):
                continue
            r_lo = safe_float(cz.get("price_low", cz.get("low")), 0.0)
            r_hi = safe_float(cz.get("price_high", cz.get("high")), 0.0)
            if r_lo <= 0 or r_hi < r_lo:
                continue
            if str(cz.get("radar_eligibility") or "QUALIFIED").upper() != "QUALIFIED":
                continue
            if str(cz.get("risk_tier") or "").upper() == "FAR_ZONE" or is_far_reference_zone(cz, current_price=price if price > 0 else None):
                continue
            cz_type = str(cz.get("zone_type") or "").upper()
            if h4_bias_ctx in ("BULLISH", "BEARISH") and cz_type != ("DEMAND" if h4_bias_ctx == "BULLISH" else "SUPPLY"):
                continue
            cz_id = str(cz.get("zone_id") or "")
            if smc_zone_usable and cz_id and cz_id == z_id and not ((cycle.get("focus_detail") or {}).get("SMC") or {}).get("historical_touch"):
                continue
            if p_status.startswith("READY") and p_zid and cz_id and cz_id == p_zid:
                continue
            radar_candidate = dict(cz)
            if price > 0 and radar_candidate.get("distance") is None:
                radar_candidate["distance"] = 0.0 if r_lo <= price <= r_hi else (r_lo - price if r_lo > price else price - r_hi)
            break

    # 2. Confirmed Entry validation (Section 5)
    dec_status = str(decision.get("status") or "").upper()
    dec_dir = str(decision.get("direction") or "").upper()
    conf_plan = decision.get("plan") or {}

    is_valid_confirmed = (
        dec_status == "VALID"
        and dec_dir in ("BUY", "SELL")
        and not data_invalid
        and not is_tick_only
        and bool(conf_plan)
        and bool(zone.get("zone_id"))
        and (not conf_plan.get("zone_id") or conf_plan.get("zone_id") == zone.get("zone_id"))
        and lifecycle not in ("INVALIDATED", "EXPIRED")
        and zone.get("is_valid") is not False
    )

    c_entry = safe_float(conf_plan.get("entry_acuan", conf_plan.get("entry")), 0.0)
    is_entry_stale = False

    conf_zid = str(conf_plan.get("zone_id") or (decision.get("zone") or {}).get("zone_id") or "")
    conf_lo = safe_float((decision.get("zone") or {}).get("price_low"), 0.0)
    conf_hi = safe_float((decision.get("zone") or {}).get("price_high"), 0.0)
    confirmed_setup = {
        "setup_id": f"{conf_zid}:{dec_dir}:{decision.get('setup_type', 'CONFIRMED')}" if conf_zid else "N/A",
        "zone_id": conf_zid or "N/A",
        "direction": dec_dir if is_valid_confirmed else "NONE",
        "zone_range": (conf_lo, conf_hi),
        "setup_type": str(decision.get("setup_type") or "TREND_PULLBACK"),
        "timeframe": str((decision.get("zone") or {}).get("timeframe") or "H1"),
        "is_valid": is_valid_confirmed,
        "is_stale": is_entry_stale,
    }

    snr_sig = str(snr_decision.get("signal", "WAIT")).upper()
    snr_confirmed = (
        snr.get("status") == "OK" and snr_sig in ("BUY", "SELL")
        and snr_decision.get("entry") is not None and not data_invalid and not is_tick_only
    )
    if not is_valid_confirmed:
        if snr_confirmed:
            confirmed_status = f"{snr_sig} [SNR]"
        else:
            confirmed_status = "WAIT"
    else:
        method_tag = "[SMC + SNR]" if snr_sig == dec_dir else "[SMC]"
        if is_entry_stale:
            confirmed_status = f"{dec_dir} {method_tag} (ENTRY STALE)"
        else:
            confirmed_status = f"{dec_dir} {method_tag}"

    if confirmed_status != "WAIT" and conf_zid and z_id != "N/A" and conf_zid != z_id:
        c_label = f"[{conf_lo:.2f}-{conf_hi:.2f}]" if (conf_lo > 0 and conf_hi > 0) else f"[{conf_zid}]"
        confirmed_display = f"{confirmed_status} {c_label}"
    else:
        confirmed_display = confirmed_status

    if is_valid_confirmed and not is_entry_stale:
        monitor_state = "CONFIRMED"
    elif monitor_state == "CONFIRMED":
        monitor_state = "AWAITING_CONFIRMATION"

    # State, Reason, Next Action, and Final Signal
    if data_invalid:
        reason = "PRICE CHANGED; VALIDATION SNAPSHOT EXPIRED" if needs_revalidation else "DATA STALE / UNAVAILABLE"
        next_action, final_signal = "WAIT FRESH VALIDATION", "WAIT"
    elif is_valid_confirmed and not is_entry_stale:
        reason, next_action, final_signal = "M5 TRIGGER VALID", "CHECK ENTRY VALIDITY", confirmed_status
    elif snr_confirmed:
        reason = str(snr_decision.get("reason") or "SNR ENTRY VALID")
        next_action, final_signal = "CHECK ENTRY VALIDITY", confirmed_status
    elif is_entry_stale:
        reason = f"Entry stale: harga ({price:.2f}) telah bergerak melewati entry {c_entry:.2f}"
        next_action, final_signal = "WAIT PULLBACK", "WAIT (ENTRY STALE)"
    elif monitor_state == "INVALIDATED" or touch_status == "INVALIDATED":
        reason, next_action, final_signal = "ZONE INVALIDATED", "CANCEL SETUP", "WAIT"
    elif monitor_state == "EXPIRED":
        reason, next_action, final_signal = "SETUP EXPIRED", "WAIT NEW SETUP", "WAIT"
    elif decision.get("is_missed") or (decision.get("m5") or {}).get("is_missed") or decision.get("signal_status") in ("SIGNAL_STATUS_ENTRY_MISSED", "ENTRY_MISSED", "ENTRY MISSED"):
        reason = "Sinyal M5 historis/terlewat; harga meninggalkan zona"
        next_action, final_signal = "WAIT PULLBACK", "WAIT (ENTRY MISSED)"
        monitor_state = "IDLE"
    elif touch_status in ("IN_ZONE", "TOUCHED_PREVIOUSLY") or monitor_state == "AWAITING_CONFIRMATION":
        reason = str((decision.get("reasons") or ["ZONE TOUCHED; M5 TRIGGER PENDING"])[0])
        next_action, final_signal = "CHECK VALIDATION", "WAIT"
    elif monitor_state == "APPROACHING":
        reason, next_action, final_signal = "APPROACHING ENTRY ZONE", "WAIT ZONE TOUCH", "WAIT"
    elif not zone:
        reason, next_action, final_signal = "NO SMC SETUP", "WAIT SETUP", "WAIT"
    else:
        reason, next_action, final_signal = "WAITING FOR ZONE PULLBACK", "WAIT APPROACH", "WAIT"
    historical_event = ""
    focus_status = cycle.get("focus_status") or {}
    focus_detail_all = cycle.get("focus_detail") or {}
    is_mkt_closed_ctx = (
        str(snapshot.get("market_status") or decision.get("market_status") or "").upper() == "CLOSED"
        or snapshot_status == "MARKET_CLOSED"
    )
    tracked_position = None
    for _m_tp, _it_tp in focus_detail_all.items():
        if isinstance(_it_tp, dict) and _it_tp.get("account_status") == "RUNNING" and _it_tp.get("ticket"):
            tracked_position = {
                "method": _m_tp,
                "ticket": _it_tp.get("ticket"),
                "direction": _it_tp.get("direction") or direction,
                "open_price": safe_float(_it_tp.get("position_open_price"), 0.0),
                "zone_id": _it_tp.get("zone_id") or "",
                "zone_low": safe_float(_it_tp.get("low"), 0.0),
                "zone_high": safe_float(_it_tp.get("high"), 0.0),
                "match_status": str(_it_tp.get("match_status") or "INFERRED").upper(),
                "match_reason": str(_it_tp.get("match_reason") or ""),
                "historical_tracking": bool(_it_tp.get("historical_tracking") or is_mkt_closed_ctx),
                "plan_valid": bool(_it_tp.get("plan_valid")),
                "uncertain": bool(_it_tp.get("uncertain")),
            }
            break

    if "RUNNING" in focus_status.values() and not is_valid_confirmed and not snr_confirmed:
        tp_hist = bool(tracked_position and tracked_position.get("historical_tracking")) or is_mkt_closed_ctx
        tp_tkt = tracked_position.get("ticket") if tracked_position else None
        tp_mst = tracked_position.get("match_status", "INFERRED") if tracked_position else "INFERRED"
        monitor_state = "RUNNING (HISTORICAL TRACKING)" if tp_hist else "RUNNING"
        touch_status = "TOUCHED_PREVIOUSLY"
        touch_display = f"TRACKED POS #{tp_tkt} [{tp_mst}]" if tp_tkt else "ZONA TERAMBIL (POSISI COCOK)"
        running_uncertain = any(
            item.get("account_status") == "RUNNING" and item.get("uncertain")
            for item in focus_detail_all.values()
        )
        running_with_plan = any(
            item.get("account_status") == "RUNNING" and item.get("plan_valid")
            for item in focus_detail_all.values()
        )
        if tp_hist:
            reason = f"HISTORICAL TRACKING: Posisi #{tp_tkt} [{tp_mst}] aktif; Entry baru tetap WAIT (MARKET CLOSED)"
            next_action = "WAIT MARKET OPEN | PANTAU POSISI MT5"
            final_signal = "WAIT"
        else:
            reason = f"POSISI #{tp_tkt} [{tp_mst}]; RIWAYAT BELUM PASTI" if running_uncertain else f"POSISI #{tp_tkt} [{tp_mst}]; PERIKSA MT5"
            next_action = "PANTAU SL/TP1 ACUAN" if running_with_plan else "PERIKSA SL/TP POSISI MT5"
            final_signal = "WAIT (MONITORING)"
    elif not is_valid_confirmed and not snr_confirmed:
        focus_detail = focus_detail_all
        active_method = "SNR" if zone.get("source") == "SNR" else "SMC"
        act_item = focus_detail.get(active_method) or {}
        has_active_zone = bool(zone and safe_float(zone.get("price_low", zone.get("low")), 0.0) > 0 and smc_zone_usable)
        historical = next(((m, it) for m, it in focus_detail.items() if it.get("historical_touch")), None)
        if act_item.get("historical_touch"):
            historical_event = f"{active_method} TOUCHED (HISTORICAL — bukan bukti entry)"
        elif historical:
            historical_event = f"{historical[0]} TOUCHED (HISTORICAL — bukan bukti entry)"

        if act_item.get("touched"):
            monitor_state = "ZONE_TOUCHED"
            touch_status = "IN_ZONE" if (lo > 0 and hi >= lo and price > 0 and lo <= price <= hi) else "TOUCHED_PREVIOUSLY"
            touch_display = "IN ZONE" if touch_status == "IN_ZONE" else f"{active_method} ZONE TOUCHED"
            dec_reasons_touch = [str(r).strip() for r in (decision.get("reasons") or []) if str(r).strip()]
            if active_method == "SMC" and m5_bound and trigger == "CONFIRMED" and dec_reasons_touch:
                reason = dec_reasons_touch[0]
                next_action = "PERIKSA RIWAYAT DAN MT5" if act_item.get("uncertain") else "CHECK VALIDATION"
            else:
                reason = f"{active_method} ZONE TOUCHED; ENTRY BELUM DIKONFIRMASI"
                next_action = "PERIKSA RIWAYAT DAN MT5" if act_item.get("uncertain") else "PANTAU KONFIRMASI M5"
            final_signal = "WAIT"
        elif has_active_zone and monitor_state in ("APPROACHING", "AWAITING_CONFIRMATION", "IN_ZONE"):
            # Pertahankan status radar / approaching / inside zone dari active setup
            pass
        elif has_active_zone:
            # Zona aktif sedang menunggu pullback (RADAR)
            monitor_state = "RADAR"
            touch_status = "NOT_TOUCHED"
            z_desc = f"{zone.get('timeframe', '')} {zone.get('zone_type', '')} {safe_float(zone.get('price_low', zone.get('low'))):.2f}-{safe_float(zone.get('price_high', zone.get('high'))):.2f}".strip()
            touch_display = f"UNTOUCHED ({format_price_distance(distance, spec=extract_unit_spec_from_container(snapshot), include_mt5_points=True)})" if distance is not None else "UNTOUCHED"
            reason = f"RADAR: Menunggu pullback ke {z_desc}"
            next_action = "WAIT PULLBACK"
            final_signal = "WAIT"
        elif radar_candidate and not data_invalid:
            # Historical touch tidak boleh menggantikan kandidat radar terbaru yang masih valid
            r_lo = safe_float(radar_candidate.get("price_low", radar_candidate.get("low")), 0.0)
            r_hi = safe_float(radar_candidate.get("price_high", radar_candidate.get("high")), 0.0)
            if price > 0 and r_lo > 0 and r_hi >= r_lo:
                if r_lo <= price <= r_hi:
                    r_dist = 0.0
                elif r_lo > price:
                    r_dist = max(round(r_lo - price, 2), 0.01)
                else:
                    r_dist = max(round(price - r_hi, 2), 0.01)
            else:
                r_dist = safe_float(radar_candidate.get("distance"), 0.0)
            r_desc = f"{radar_candidate.get('timeframe', '')} {radar_candidate.get('zone_type', '')} {r_lo:.2f}-{r_hi:.2f}".strip()
            zone = dict(radar_candidate)
            lo, hi = r_lo, r_hi
            distance = r_dist
            zone_type = str(zone.get("zone_type") or zone.get("type") or "").upper()
            direction = "BUY" if zone_type in ("DEMAND", "SUPPORT") else ("SELL" if zone_type in ("SUPPLY", "RESISTANCE") else direction)
            z_id = str(zone.get("zone_id") or "N/A")
            if pending_setup.get("zone_id") and pending_setup.get("zone_id") != "N/A":
                pending_setup["is_active_focus"] = (pending_setup["zone_id"] == z_id)
                pending_setup["source_role"] = (
                    "ACTIVE_FOCUS" if pending_setup["is_active_focus"]
                    else ("PRIMARY_ZONE" if pending_setup.get("is_primary_zone") else "NON_FOCUS_ZONE")
                )
            for alt_item in alt_pending_setups:
                if alt_item.get("zone_id") and alt_item.get("zone_id") != "N/A":
                    alt_item["is_active_focus"] = (alt_item["zone_id"] == z_id)
                    alt_item["source_role"] = (
                        "ACTIVE_FOCUS" if alt_item["is_active_focus"]
                        else ("PRIMARY_ZONE" if alt_item.get("is_primary_zone") else "NON_FOCUS_ZONE")
                    )
            active_setup.update({
                "setup_id": f"{z_id}:{direction}:{setup_type}" if z_id != "N/A" else "N/A",
                "zone_id": z_id,
                "direction": direction,
                "zone_range": (lo, hi),
                "timeframe": str(zone.get("timeframe") or "N/A"),
                "grade": str(zone.get("grade") or "N/A"),
                "score": safe_float(zone.get("score")) if zone.get("score") is not None else None,
                "lifecycle_status": str(zone.get("lifecycle_status") or zone.get("status") or "RADAR").upper(),
            })
            if price > 0 and r_lo > 0 and r_hi >= r_lo and (r_lo <= price <= r_hi):
                monitor_state = "AWAITING_CONFIRMATION"
                touch_status = "IN_ZONE"
                touch_display = "IN ZONE"
                reason = f"ZONE TOUCHED: Harga berada di {r_desc}"
                next_action = "PANTAU KONFIRMASI M5"
            elif price > 0 and r_lo > 0 and r_hi >= r_lo and r_dist <= float(ZONE_APPROACH_DISTANCE_POINTS):
                monitor_state = "APPROACHING"
                touch_status = "NOT_TOUCHED"
                _r_d_txt = format_price_distance(r_dist, spec=extract_unit_spec_from_container(snapshot), include_mt5_points=True)
                touch_display = f"APPROACHING ({_r_d_txt})"
                reason = f"APPROACHING: Harga mendekati {r_desc} ({_r_d_txt})"
                next_action = "WAIT ZONE TOUCH"
            else:
                monitor_state = "RADAR"
                touch_status = "NOT_TOUCHED"
                touch_display = f"RADAR ({format_price_distance(r_dist, spec=extract_unit_spec_from_container(snapshot), include_mt5_points=True)})" if r_dist > 0 else "UNTOUCHED"
                reason = f"RADAR: Menunggu pullback ke {r_desc}"
                next_action = "WAIT PULLBACK"
            final_signal = "WAIT"
            if not _m5_stages_match_zone(decision, zone):
                trigger = "NOT_CHECKED"
                m5_confirmation = None
            pending_display = _format_pending_compact_tag(pending_setup, z_id, is_selected=True)
            if alt_pending_setups:
                alt_first = _format_pending_compact_tag(alt_pending_setups[0], z_id, is_selected=False)
                if len(alt_pending_setups) > 1:
                    alt_first = f"{alt_first} (+{len(alt_pending_setups) - 1})"
                alt_pending_display = alt_first
            radar_candidate = None
        elif act_item.get("historical_touch"):
            monitor_state = "WAIT_NO_ACTIVE_ZONE"  # riwayat sentuh = info saja, bukan status entry
            touch_status = "TOUCHED_PREVIOUSLY"
            touch_display = f"{active_method} TOUCHED (HISTORICAL)"
            reason = f"{active_method} PERNAH DISENTUH; ENTRY BELUM TERBUKTI"
            next_action = "PERIKSA RIWAYAT DAN MT5" if act_item.get("uncertain") else "PANTAU KUNJUNGAN BARU"
            final_signal = "WAIT"
        elif act_item.get("uncertain"):
            monitor_state = "HISTORY_UNCERTAIN"
            touch_status = "NO_ZONE"
            touch_display = "SENTUHAN BELUM TERBUKTI"
            reason, next_action, final_signal = "FOKUS TERKUNCI: RIWAYAT BELUM PASTI", "PERIKSA RIWAYAT DAN MT5", "WAIT"
        elif not has_active_zone:
            touched = next(((m, it) for m, it in focus_detail.items() if it.get("touched")), None)
            uncertain = next(((m, it) for m, it in focus_detail.items() if it.get("uncertain")), None)
            if touched:
                m_t, it_t = touched
                monitor_state = "ZONE_TOUCHED"
                touch_status = "TOUCHED_PREVIOUSLY"
                touch_display = f"{m_t} ZONE TOUCHED"
                reason = f"{m_t} ZONE TOUCHED; ENTRY BELUM DIKONFIRMASI"
                next_action = "PERIKSA RIWAYAT DAN MT5" if it_t.get("uncertain") else "PANTAU KONFIRMASI M5"
                final_signal = "WAIT"
            elif historical:
                m_hist, it_hist = historical
                monitor_state = "WAIT_NO_ACTIVE_ZONE"  # riwayat sentuh = info saja, bukan status entry
                touch_status = "TOUCHED_PREVIOUSLY"
                touch_display = f"{m_hist} TOUCHED (HISTORICAL)"
                reason = f"{m_hist} PERNAH DISENTUH; ENTRY BELUM TERBUKTI"
                next_action = "PERIKSA RIWAYAT DAN MT5" if it_hist.get("uncertain") else "PANTAU KUNJUNGAN BARU"
                final_signal = "WAIT"
            elif uncertain:
                monitor_state = "HISTORY_UNCERTAIN"
                touch_status = "NO_ZONE"
                touch_display = "SENTUHAN BELUM TERBUKTI"
                reason, next_action, final_signal = "FOKUS TERKUNCI: RIWAYAT BELUM PASTI", "PERIKSA RIWAYAT DAN MT5", "WAIT"

    account = cycle.get("account") if isinstance(cycle.get("account"), dict) else {}
    warnings = []
    for raw_w in list(account.get("risk_warnings") or []) + list(cycle.get("risk_warnings") or []):
        w_str = str(raw_w or "").strip()
        if w_str and w_str not in warnings:
            warnings.append(w_str)
    focus_det_map = cycle.get("focus_detail") or {}
    has_pending_waiting = any(
        isinstance(it, dict) and it.get("plan_status") == "READY" and it.get("is_placed")
        for it in ([pending_setup] + list(alt_pending_setups or []))
    )
    ever_taken_any = any(
        isinstance(fd, dict) and (fd.get("ever_taken") or fd.get("account_status") in ("RUNNING", "POSITION NOT OBSERVED"))
        for fd in focus_det_map.values()
    )
    raw_alerts = [
        al for al in (cycle.get("alerts") or [])
        if not (
            "BELUM ADA POSISI MT5 YANG COCOK DENGAN ZONA" in str(al).upper()
            and (has_pending_waiting or not ever_taken_any)
        )
    ]
    # Deteksi pending order MT5 yang masih aktif pada zona INVALID — BROKEN (SMC, SNR & gabungan terdeduplikasi, read-only)
    inv_smc_pool = [
        {**z_inv, "source": "SMC"} for z_inv in (
            list(scan.get("invalidated_zones") or [])
            + list(decision.get("invalidated_zones") or [])
            + [
                rz for rz in (scan.get("rejected_zones") or [])
                if isinstance(rz, dict)
                and (
                    rz.get("is_broken")
                    or str(rz.get("status") or "").upper() in ("INVALIDATED", "BROKEN")
                    or str(rz.get("freshness") or "").upper() == "INVALIDATED"
                )
            ]
        )
        if isinstance(z_inv, dict)
    ]
    inv_snr_pool = [
        {**z_inv, "source": "SNR"} for z_inv in (
            list(snr_zones.get("broken_zones") or [])
            + list(snr_zones.get("invalidated_zones") or [])
        )
        if isinstance(z_inv, dict)
        and str(z_inv.get("lifecycle_status") or z_inv.get("status") or "").upper() != "RECLAIMED"
    ]
    valid_smc_pool = [
        vz for vz in (list(scan.get("active_zones") or []) + list(scan.get("radar_zones") or []))
        if isinstance(vz, dict) or hasattr(vz, "to_dict")
    ]
    valid_snr_pool = [
        vz for vz in (list(snr_zones.get("active_supports") or []) + list(snr_zones.get("active_resistances") or []))
        if isinstance(vz, dict)
    ]
    inv_smc_warns = find_pending_orders_on_invalid_zones(
        inv_smc_pool,
        mt5_snapshot=mt5_ord_snap,
        symbol=sym_ctx,
        default_source="SMC",
        active_valid_zones=valid_smc_pool,
    )
    inv_snr_warns = find_pending_orders_on_invalid_zones(
        inv_snr_pool,
        mt5_snapshot=mt5_ord_snap,
        symbol=sym_ctx,
        default_source="SNR",
        active_valid_zones=valid_snr_pool,
    )
    invalid_pending_warnings = find_pending_orders_on_invalid_zones(
        inv_smc_pool + inv_snr_pool,
        mt5_snapshot=mt5_ord_snap,
        symbol=sym_ctx,
        active_valid_zones=valid_smc_pool + valid_snr_pool,
    )
    invalid_pending_scope = {
        "scope": "SMC+SNR",
        "smc": len(inv_smc_warns),
        "snr": len(inv_snr_warns),
        "combined": len(invalid_pending_warnings),
    }
    for iw in invalid_pending_warnings:
        iw_msg = iw.get("message") if isinstance(iw, dict) else str(iw)
        if iw_msg and iw_msg not in warnings:
            warnings.append(iw_msg)

    matched_recons_ctx = []
    for p_chk in ([pending_setup] + list(alt_pending_setups or [])):
        if isinstance(p_chk, dict):
            recon_p = p_chk.get("mt5_reconciliation")
            if isinstance(recon_p, dict):
                matched_recons_ctx.append(recon_p)
            if p_chk.get("sl_missing_high_risk") and p_chk.get("is_placed"):
                tkts_p = [f"#{t}" for t in ((recon_p or {}).get("matched_tickets") or []) if t is not None]
                tkt_lbl = ", ".join(tkts_p[:2]) if tkts_p else str(p_chk.get("zone_id") or "PENDING")
                risk_msg = f"[RISK WARNING] Matched order {tkt_lbl} TANPA Stop Loss — SL MISSING HIGH RISK!"
                if risk_msg not in warnings:
                    warnings.append(risk_msg)

    spread_warn_ctx = ""
    acc_spread_v = account.get("spread")
    if acc_spread_v is not None and safe_float(acc_spread_v, 0.0) >= 0.80:
        spread_warn_ctx = f"Spread Wide ({format_price_distance(safe_float(acc_spread_v), spec=extract_unit_spec_from_container(snapshot), include_mt5_points=True)})"
    mt5_risk_summary = summarize_mt5_risk_warnings(
        mt5_ord_snap,
        invalid_zone_warnings=invalid_pending_warnings,
        matched_recons=matched_recons_ctx,
        spread_warning=spread_warn_ctx or None,
        invalid_zone_scope=invalid_pending_scope,
    )
    alerts = list(raw_alerts)
    for w_item in warnings:
        if w_item not in alerts:
            alerts.append(w_item)

    categorized = classify_monitor_alerts(
        raw_alerts,
        risk_warnings=warnings,
        default_ts=cycle.get("snapshot_time") or snapshot.get("snapshot_time"),
    )
    for hev_m in historical_events_meta:
        h_lbl_m = str(hev_m.get("label") or "").strip()
        if h_lbl_m and h_lbl_m not in categorized["historical"]:
            categorized["historical"].append(h_lbl_m)
    if snr_historical_watch and snr_historical_watch not in categorized["historical"]:
        categorized["historical"].append(snr_historical_watch)
    if not historical_event and categorized["historical"]:
        historical_event = categorized["historical"][0]

    def _clean_alert_label(raw_txt):
        txt = str(raw_txt or "").replace("\n", " ").strip()
        for prefix in ("[SMC] ", "[SNR] ", "[ALERT] ", "[ACCOUNT RISK] "):
            if txt.startswith(prefix):
                txt = txt[len(prefix):].strip()
        if "TERKONFIRMASI" in txt:
            txt = txt.replace("TERKONFIRMASI", "CONFIRMED")
        if "PERUBAHAN ZONA FOKUS -> " in txt:
            txt = "FOCUS CHANGED [" + txt.replace("PERUBAHAN ZONA FOKUS -> ", "").strip() + "]"
        return txt

    # Clean priority-based alert display resolution:
    # - Prioritaskan current MARKET_ALERT / ENTRY_ALERT tanpa menghilangkan critical RISK_ALERT.
    # - RISK_ALERT ditampilkan terpisah di baris RISK ALERT agar tidak tertutup maupun terduplikasi.
    # - HISTORICAL_INFO tidak pernah dipromosikan menjadi current ALERT.
    if is_valid_confirmed and not is_entry_stale:
        alert_display = f"{dec_dir} CONFIRMED"
    elif is_entry_stale:
        alert_display = "ENTRY STALE"
    elif monitor_state == "INVALIDATED" or touch_status == "INVALIDATED":
        alert_display = "ZONE INVALIDATED"
    else:
        non_hist_non_risk = [
            ev for ev in categorized["events"]
            if ev.get("category") not in ("HISTORICAL_INFO", "RISK_ALERT")
        ]
        if non_hist_non_risk:
            top_ev = max(non_hist_non_risk, key=lambda ev: int(ev.get("priority", 0)))
            best_al = _clean_alert_label(top_ev.get("message", ""))
            other_important = [
                ev for ev in non_hist_non_risk
                if ev is not top_ev and ev.get("category") in ("MARKET_ALERT", "ENTRY_ALERT")
            ]
            if best_al and other_important:
                best_al = f"{best_al} (+{len(other_important)})"
        else:
            best_al = "NONE"
        alert_display = best_al if best_al else "NONE"

    critical_risk_display = mt5_risk_summary.get("headline") or ""
    if not critical_risk_display:
        p_nosl_cnt = int(safe_float(account.get("position_without_sl"), 0))
        o_nosl_cnt = int(safe_float(account.get("pending_without_sl"), 0))
        if p_nosl_cnt > 0 or o_nosl_cnt > 0:
            parts_nosl = []
            if p_nosl_cnt > 0:
                parts_nosl.append(f"{p_nosl_cnt} Position{'s' if p_nosl_cnt != 1 else ''}")
            if o_nosl_cnt > 0:
                parts_nosl.append(f"{o_nosl_cnt} Pending")
            critical_risk_display = f"RISK: {' + '.join(parts_nosl)} Without SL [HIGH RISK]"
        elif categorized["risk"]:
            first_risk = _clean_alert_label(categorized["risk"][0])
            if first_risk.startswith("[RISK WARNING] "):
                first_risk = first_risk[len("[RISK WARNING] "):].strip()
            critical_risk_display = first_risk

    return {
        "snapshot": snapshot,
        "smc": smc,
        "snr": snr,
        "scan": scan,
        "decision": decision,
        "summary": summary,
        "snr_scan": snr_scan,
        "snr_decision": snr_decision,
        "smc_zone": smc_zone_ref,
        "snr_zone": snr_zone,
        "snr_focus_zone": snr_zone,
        "directional_snr_zone": directional_snr_zone,
        "snr_confluence_source": snr_confluence_source,
        "snr_confluence_overlap_points": round(snr_confluence_overlap_pts, 2),
        "snr_confluence_source_display": snr_confluence_source_display,
        "snr_zone_direction": snr_zone_direction,
        "snr_selection_reason": snr_selection_reason,
        "snr_selection_reason_short": snr_reason_short,
        "snr_price_proximity": snr_price_proximity,
        "zone": zone,
        "far_reference_zone": far_reference_zone,
        "plan": plan,
        "price": price,
        "distance": distance,
        "direction": direction,
        "confluence": confluence,
        "confluence_display": confluence_display,
        "confluence_reason": confluence_reason,
        "lifecycle": lifecycle,
        "monitor_state": monitor_state,
        "touch": touch_display,
        "touch_status": touch_status,
        "touch_display": touch_display,
        "trigger": trigger,
        "m5_confirmation": m5_confirmation,
        "confirmed": confirmed_status,
        "confirmed_display": confirmed_display,
        "pending_status": p_status,
        "pending_display": pending_display,
        "alt_pending_setups": alt_pending_setups,
        "alt_pending_display": alt_pending_display,
        "pending_setup": pending_setup,
        "signal": final_signal,
        "signal_display": final_signal,
        "data_invalid": data_invalid,
        "freshness": freshness,
        "reason": reason,
        "reason_display": reason,
        "next_action": next_action,
        "alerts": alerts,
        "alert_display": alert_display,
        "critical_risk_display": critical_risk_display,
        "mt5_risk_summary": mt5_risk_summary,
        "categorized_alerts": categorized,
        "risk_alerts": categorized["risk"],
        "market_alerts": categorized["market"],
        "entry_alerts": categorized["entry"],
        "account_alerts": categorized["account"],
        "historical_info": categorized["historical"],
        "historical_events_meta": historical_events_meta,
        "account": account,
        "active_setup": active_setup,
        "tracked_position": tracked_position,
        "radar_candidate": radar_candidate,
        "historical_event": historical_event,
        "snr_breakout_watch": snr_breakout_watch,
        "snr_snapshot_id": str(snr_scan.get("snapshot_id") or ""),
        "pending_setup": pending_setup,
        "confirmed_setup": confirmed_setup,
        "invalid_pending_warnings": invalid_pending_warnings,
        "invalid_pending_scope": invalid_pending_scope,
        "mt5_orders_snapshot": mt5_ord_snap,
    }


def _resolve_dashboard_ticker_info(cycle, snapshot, data, acc):
    """Ekstrak data harga live TradingView-style dari live_ticker / snapshot MT5 (tanpa angka perkiraan)."""
    raw_tk = cycle.get("live_ticker") if isinstance(cycle.get("live_ticker"), dict) else {}
    sym = str(raw_tk.get("symbol") or snapshot.get("symbol") or acc.get("symbol") or "XAUUSDc")

    u_spec_tk = extract_unit_spec_from_container(raw_tk if raw_tk.get("point") else snapshot, symbol=sym)
    digits_raw = raw_tk.get("digits", snapshot.get("digits"))
    try:
        digits = max(0, min(8, int(digits_raw))) if digits_raw is not None else u_spec_tk["digits"]
    except (TypeError, ValueError):
        digits = u_spec_tk["digits"]
    u_spec_tk = dict(u_spec_tk)
    u_spec_tk["digits"] = digits

    conn_st = str(raw_tk.get("connection_status") or snapshot.get("connection_status") or "").upper()
    mkt_st = str(raw_tk.get("market_status") or snapshot.get("market_status") or "").upper()
    fresh_st = str(raw_tk.get("data_freshness") or snapshot.get("data_freshness") or "").upper()
    snap_st = str(snapshot.get("snapshot_status") or "").upper()

    feed_st = str(raw_tk.get("feed_status") or "").upper()
    if not feed_st:
        if conn_st == "DISCONNECTED" or fresh_st == "NO DATA":
            feed_st = "DISCONNECTED"
        elif mkt_st == "CLOSED" or snap_st == "MARKET_CLOSED":
            feed_st = "CLOSED"
        elif fresh_st in ("STALE", "INVALID") or snap_st == "DATA_INCOMPLETE" or snapshot.get("data_ok") is False:
            feed_st = "STALE"
        else:
            feed_st = "OPEN"

    prev_d1 = raw_tk.get("prev_d1_close")
    if prev_d1 is None:
        prev_d1 = snapshot.get("prev_d1_close")
    if prev_d1 is None:
        frames = snapshot.get("frames") if isinstance(snapshot.get("frames"), dict) else {}
        df_d1 = frames.get("D1") if isinstance(frames, dict) else None
        if df_d1 is not None and len(df_d1) > 0 and "Close" in df_d1.columns:
            try:
                idx_d1 = -2 if (feed_st == "CLOSED" and len(df_d1) >= 2) else -1
                cand_d1 = float(df_d1["Close"].iloc[idx_d1])
                if cand_d1 > 0:
                    prev_d1 = cand_d1
            except Exception:
                prev_d1 = None
    prev_d1_val = safe_float(prev_d1, 0.0) if prev_d1 is not None else 0.0

    if feed_st in ("DISCONNECTED", "STALE"):
        live_price = 0.0
        bid_v = 0.0
        ask_v = 0.0
        spr_v = None
    elif feed_st == "CLOSED":
        live_price = safe_float(raw_tk.get("live_price", snapshot.get("snapshot_price", data.get("price"))), 0.0)
        bid_v = safe_float(raw_tk.get("bid", snapshot.get("tick_bid_raw", snapshot.get("bid"))), 0.0)
        ask_v = safe_float(raw_tk.get("ask", snapshot.get("tick_ask_raw", snapshot.get("ask"))), 0.0)
        spr_v = raw_tk.get("spread", snapshot.get("spread"))
    else:
        bid_v = safe_float(raw_tk.get("bid", snapshot.get("tick_bid_raw", snapshot.get("bid"))), 0.0)
        ask_v = safe_float(raw_tk.get("ask", snapshot.get("tick_ask_raw", snapshot.get("ask"))), 0.0)
        live_price = safe_float(raw_tk.get("live_price"), 0.0)
        if live_price <= 0:
            live_price = bid_v if bid_v > 0 else safe_float(snapshot.get("snapshot_price", data.get("price")), 0.0)
        if live_price <= 0:
            feed_st = "DISCONNECTED" if conn_st == "DISCONNECTED" else "STALE"
            bid_v = 0.0
            ask_v = 0.0
            spr_v = None
        else:
            spr_v = raw_tk.get("spread")
            if spr_v is None:
                spr_v = (ask_v - bid_v) if (bid_v > 0 and ask_v >= bid_v) else snapshot.get("spread")

    daily_chg = None
    daily_pct = None
    if live_price > 0 and prev_d1_val > 0 and feed_st in ("OPEN", "CLOSED"):
        daily_chg = live_price - prev_d1_val
        daily_pct = (daily_chg / prev_d1_val) * 100.0

    change_dir = "FLAT"
    if daily_chg is not None:
        if daily_chg > 1e-9:
            change_dir = "UP"
        elif daily_chg < -1e-9:
            change_dir = "DOWN"

    price_dir = str(raw_tk.get("price_direction") or "").upper()
    if price_dir not in ("UP", "DOWN", "FLAT"):
        price_dir = change_dir

    tick_delta = raw_tk.get("tick_delta")
    tick_delta_val = safe_float(tick_delta, 0.0) if tick_delta is not None else 0.0
    spr_float = safe_float(spr_v, -1.0) if spr_v is not None else -1.0

    return {
        "symbol": sym,
        "digits": digits,
        "point": u_spec_tk["point"],
        "trade_tick_size": u_spec_tk["trade_tick_size"],
        "unit_spec": u_spec_tk,
        "feed_status": feed_st,
        "live_price": live_price,
        "bid": bid_v,
        "ask": ask_v,
        "spread": spr_float if spr_float >= 0 else None,
        "spread_wide": bool(spr_float >= 0.80),
        "prev_d1_close": prev_d1_val if prev_d1_val > 0 else None,
        "daily_change": daily_chg,
        "daily_change_pct": daily_pct,
        "change_direction": change_dir,
        "price_direction": price_dir,
        "tick_delta": tick_delta_val,
    }


def _build_dashboard_ticker_widget(tk, ts_short, cyc_no, term_w, use_uni):
    """Bangun baris TradingView-style Big Price Widget + metadata pewarnaan (ringan, <0.3 ms/tick)."""
    eq_div = "=" * term_w
    dash_div = "-" * term_w
    digits = tk["digits"]
    feed_st = tk["feed_status"]
    sym_txt = tk["symbol"]
    live_p = tk["live_price"]
    bid_v = tk["bid"]
    ask_v = tk["ask"]
    spr_v = tk["spread"]
    prev_d1 = tk["prev_d1_close"]
    d_chg = tk["daily_change"]
    d_pct = tk["daily_change_pct"]

    dot = "●" if use_uni else "*"
    up_ar = "▲" if use_uni else "^"
    dn_ar = "▼" if use_uni else "v"
    fl_ar = "─" if use_uni else "-"

    status_label_map = {
        "OPEN": f"{dot} MARKET OPEN",
        "CLOSED": f"{dot} MARKET CLOSED",
        "STALE": f"{dot} STALE",
        "DISCONNECTED": f"{dot} DISCONNECTED",
    }
    status_badge = status_label_map.get(feed_st, f"{dot} {feed_st}")

    # Header bar TradingView-style: Symbol + Status Badge di kiri, Timestamp Tick MT5 + Cycle di kanan
    left_hdr = f" {sym_txt}  {status_badge}"
    right_hdr = f"{ts_short} | #{cyc_no} | Ctrl+C "
    gap_hdr = term_w - len(left_hdr) - len(right_hdr)
    if gap_hdr >= 2:
        hdr_line = left_hdr + (" " * gap_hdr) + right_hdr
        time_span = (len(left_hdr) + gap_hdr, len(hdr_line))
    else:
        short_right = f" | {ts_short} | #{cyc_no}"
        hdr_line = truncate_clean(left_hdr + short_right, term_w)
        t_pos = hdr_line.find(" | ")
        time_span = (t_pos, len(hdr_line)) if t_pos >= 0 else None
    sym_span = (1, 1 + len(sym_txt))
    st_start = 1 + len(sym_txt) + 2
    status_span = (st_start, min(len(hdr_line), st_start + len(status_badge)))

    # Susun teks perubahan harian (berdasarkan D1 Close sebelumnya dari MT5)
    if feed_st in ("DISCONNECTED", "STALE"):
        price_str = "---"
        chg_str = "MT5 DISCONNECTED — NO LIVE PRICE" if feed_st == "DISCONNECTED" else "STALE FEED — NO LIVE TICK"
        quote_str = "Bid: --- | Ask: --- | Spr: ---"
        meta_str = "Waiting for live MT5 connection..." if feed_st == "DISCONNECTED" else "Waiting for fresh MT5 live tick..."
    else:
        price_str = f"{live_p:.{digits}f}" if live_p > 0 else "---"
        if d_chg is not None and d_pct is not None and prev_d1 is not None:
            ar = up_ar if d_chg > 1e-9 else (dn_ar if d_chg < -1e-9 else fl_ar)
            chg_prefix = "LAST CLOSE | " if feed_st == "CLOSED" else ""
            chg_str = f"{chg_prefix}{ar} {d_chg:+.{digits}f} ({d_pct:+.2f}%)  D1:{prev_d1:.{digits}f}"
        elif feed_st == "CLOSED":
            chg_str = "LAST CLOSE | D1: N/A"
        else:
            chg_str = "Daily Chg: N/A (D1 Close unavailable)"

        if bid_v > 0 and ask_v > 0:
            _spr_val = spr_v if spr_v is not None else max(0.0, ask_v - bid_v)
            spr_txt = format_price_distance(_spr_val, spec=tk.get("unit_spec"), digits=digits, include_mt5_points=True)
            quote_str = f"Bid: {bid_v:.{digits}f} | Ask: {ask_v:.{digits}f} | Spr: {spr_txt}"
        else:
            quote_str = "Bid: --- | Ask: --- | Spr: ---"

        if feed_st == "CLOSED":
            meta_str = "MARKET CLOSED (Last Session Close)"
        else:
            td = tk["tick_delta"]
            td_ar = up_ar if tk["price_direction"] == "UP" else (dn_ar if tk["price_direction"] == "DOWN" else fl_ar)
            td_txt = f"Tick {td_ar} {td:+.{digits}f}" if abs(td) > 1e-9 else f"Tick {td_ar} LIVE"
            meta_str = f"MT5 LIVE TICK | {td_txt}"

    widget_lines = [eq_div, hdr_line, dash_div]
    widget_meta = {
        "feed_status": feed_st,
        "price_direction": tk["price_direction"],
        "change_direction": tk["change_direction"],
        "spread_wide": tk["spread_wide"],
        "header_idx": 1,
        "sym_span": sym_span,
        "status_span": status_span,
        "time_span": time_span,
        "price_rows": [],
        "compact_price_idx": None,
        "compact_price_span": None,
        "compact_change_span": None,
    }

    big_rows = render_big_price_digits(price_str) if use_uni else []
    big_w = max((len(r) for r in big_rows), default=0)
    use_big_widget = bool(use_uni and big_rows and (1 + big_w + 3 + 26) <= term_w)
    if use_big_widget:
        side_texts = [(chg_str, "change"), (quote_str, "quote"), (meta_str, "meta")]
        split_col = 1 + big_w + 3
        side_avail = max(16, term_w - split_col)
        for r_i in range(3):
            b_row = big_rows[r_i].ljust(big_w)
            s_txt, s_role = side_texts[r_i]
            line_str = f" {b_row}   {truncate_clean(s_txt, side_avail)}"
            line_idx = len(widget_lines)
            widget_lines.append(line_str)
            widget_meta["price_rows"].append((line_idx, split_col, s_role))
    else:
        c_prefix = " PRICE : "
        c_p = price_str
        c_gap = "  "
        c_chg = truncate_clean(chg_str, max(12, term_w - len(c_prefix) - len(c_p) - len(c_gap)))
        c_line = f"{c_prefix}{c_p}{c_gap}{c_chg}"
        c_idx = len(widget_lines)
        widget_lines.append(c_line)
        widget_meta["compact_price_idx"] = c_idx
        widget_meta["compact_price_span"] = (len(c_prefix), len(c_prefix) + len(c_p))
        widget_meta["compact_change_span"] = (len(c_prefix) + len(c_p) + len(c_gap), len(c_line))
        widget_lines.extend(wrap_dashboard_field(" QUOTE : ", quote_str, term_w, max_lines=1))

    return widget_lines, widget_meta


def _format_live_distance_line(lo, hi, price_val, feed_st, monitor_state, touch_display, term_w, unit_spec=None):
    if feed_st in ("DISCONNECTED", "STALE") or price_val <= 0 or lo <= 0 or hi < lo:
        dist_txt = "N/A"
    elif lo <= price_val <= hi:
        dist_txt = f"{format_price_distance(0.0, spec=unit_spec, include_mt5_points=True)} (IN ZONE | {touch_display})"
    else:
        dist_v = (lo - price_val) if lo > price_val else (price_val - hi)
        side_lbl = "ABOVE" if lo > price_val else "BELOW"
        dist_txt = f"{format_price_distance(dist_v, spec=unit_spec, include_mt5_points=True)} ({side_lbl} | {monitor_state})"
    return wrap_dashboard_field(" DISTANCE    : ", dist_txt, term_w, max_lines=1)[0]


def patch_dashboard_live_ticker(base_text, base_meta, live_ticker, snapshot_time=None):
    """
    Update in-place HANYA widget harga real-time + baris jarak Active Focus tanpa menghitung ulang
    SMC/SNR ataupun `_smart_monitor_context()` (<0.3 ms/tick).
    """
    if not isinstance(base_meta, dict) or not isinstance(base_meta.get("_live_patch"), dict):
        return None, None
    lp = base_meta["_live_patch"]
    term_w = get_terminal_width()
    if term_w != lp.get("term_w"):
        return None, None

    raw_tk = live_ticker if isinstance(live_ticker, dict) else {}
    tk = _resolve_dashboard_ticker_info({"live_ticker": raw_tk}, raw_tk, {}, {})
    raw_ts = str(snapshot_time or raw_tk.get("snapshot_time") or lp.get("last_ts") or "NO TICK")
    ts_short = raw_ts[11:19] + " WIB" if len(raw_ts) >= 19 else raw_ts

    widget_lines, new_meta = _build_dashboard_ticker_widget(
        tk,
        ts_short=ts_short,
        cyc_no=lp.get("cycle_no", 1),
        term_w=term_w,
        use_uni=bool(lp.get("use_uni", True)),
    )

    body_lines = list(lp.get("body_lines") or [])
    dist_idx = lp.get("dist_body_idx")
    focus_bounds = lp.get("focus_bounds")
    if dist_idx is not None and 0 <= dist_idx < len(body_lines) and focus_bounds:
        f_lo, f_hi = focus_bounds
        if tk.get("feed_status") == "CLOSED":
            price_val = tk["live_price"] if tk.get("live_price", 0.0) > 0 else (round((tk["bid"] + tk["ask"]) / 2.0, int(tk.get("digits") or 2)) if tk.get("bid", 0) > 0 else 0.0)
        elif tk["bid"] > 0 and tk["ask"] > 0:
            price_val = round((tk["bid"] + tk["ask"]) / 2.0, int(tk.get("digits") or 2))
        else:
            price_val = tk["live_price"] if tk["live_price"] > 0 else 0.0
        eff_mon_state = lp.get("monitor_state", "IDLE")
        if price_val > 0 and f_lo > 0 and f_hi >= f_lo and eff_mon_state not in ("CONFIRMED", "RUNNING", "INVALIDATED", "EXPIRED"):
            if f_lo <= price_val <= f_hi:
                eff_mon_state = "AWAITING_CONFIRMATION"
            else:
                d_live = (f_lo - price_val) if f_lo > price_val else (price_val - f_hi)
                eff_mon_state = "APPROACHING" if d_live <= float(ZONE_APPROACH_DISTANCE_POINTS) else "RADAR"
        body_lines[dist_idx] = _format_live_distance_line(
            f_lo,
            f_hi,
            price_val,
            tk["feed_status"],
            eff_mon_state,
            lp.get("touch_display", "IN ZONE"),
            term_w,
            unit_spec=tk.get("unit_spec"),
        )

    new_lp = dict(lp)
    new_lp["last_ts"] = raw_ts
    new_meta["_live_patch"] = new_lp
    return fit_dashboard_text("\n".join([*widget_lines, *body_lines]), width=term_w), new_meta


def format_loading_live_dashboard(live_ticker, snapshot_time=None, mode="unified"):
    """Dashboard awal real-time agar harga XAUUSD langsung bergerak sambil menunggu siklus scan SMC/SNR pertama."""
    term_w = get_terminal_width()
    use_uni = supports_unicode_glyphs()
    eq_div = "=" * term_w
    dash_div = "-" * term_w

    raw_tk = live_ticker if isinstance(live_ticker, dict) else {}
    tk = _resolve_dashboard_ticker_info({"live_ticker": raw_tk}, raw_tk, {}, {})
    raw_ts = str(snapshot_time or raw_tk.get("snapshot_time") or "NO TICK")
    ts_short = raw_ts[11:19] + " WIB" if len(raw_ts) >= 19 else raw_ts

    widget_lines, widget_meta = _build_dashboard_ticker_widget(
        tk, ts_short=ts_short, cyc_no=0, term_w=term_w, use_uni=use_uni
    )
    mode_lbl = str(mode or "unified").upper()
    body_lines = [
        dash_div,
        *wrap_dashboard_field(" STATUS      : ", f"INITIALIZING {mode_lbl} SCANNER (SMC/SNR running in background...)", term_w, max_lines=1),
        *wrap_dashboard_field(" ACTIVE FOCUS: ", "SCANNING...", term_w, max_lines=1),
        eq_div,
    ]
    widget_meta["_live_patch"] = {
        "body_lines": body_lines,
        "dist_body_idx": None,
        "focus_bounds": None,
        "monitor_state": "IDLE",
        "touch_display": "NOT TOUCHED",
        "cycle_no": 0,
        "term_w": term_w,
        "use_uni": use_uni,
        "last_ts": raw_ts,
    }
    return fit_dashboard_text("\n".join([*widget_lines, *body_lines]), width=term_w), widget_meta


def format_smart_monitor_dashboard_with_meta(cycle_result, recent_events=None):
    """
    Dashboard terpadu TradingView-style Price Widget + Smart Monitor untuk in-place refresh `pantau`.
    Mengembalikan `(plain_text, widget_meta)` agar renderer Rich/ANSI mewarnai widget harga secara presisi.
    """
    data = _smart_monitor_context(cycle_result)
    cycle = cycle_result if isinstance(cycle_result, dict) else {}
    zone = data["zone"]
    far_ref = data.get("far_reference_zone")
    summary = data["summary"]
    snapshot = data["snapshot"]
    acc = data.get("account") or {}
    mt5_snap = data.get("mt5_orders_snapshot") if isinstance(data.get("mt5_orders_snapshot"), dict) else {}
    mt5_risk = data.get("mt5_risk_summary") if isinstance(data.get("mt5_risk_summary"), dict) else {}

    term_w = get_terminal_width()
    term_h = get_terminal_height()
    use_uni = supports_unicode_glyphs()
    max_recent = 3 if term_h <= 28 else 5
    eq_div = "=" * term_w
    dash_div = "-" * term_w

    tk = _resolve_dashboard_ticker_info(cycle, snapshot, data, acc)
    feed_st = tk["feed_status"]
    live_p = tk["live_price"]

    snap_p = safe_float(snapshot.get("snapshot_price", data.get("price")), 0.0)
    price_val = snap_p if snap_p > 0 else (live_p if live_p > 0 else 0.0)
    raw_ts = str(cycle.get("snapshot_time") or snapshot.get("snapshot_time") or format_wib_timestamp(default_now=True))
    ts_short = raw_ts[11:19] + " WIB" if len(raw_ts) >= 19 else raw_ts
    cyc_no = cycle.get("cycle", 1)

    widget_lines, widget_meta = _build_dashboard_ticker_widget(
        tk, ts_short=ts_short, cyc_no=cyc_no, term_w=term_w, use_uni=use_uni
    )

    h4 = str(summary.get("h4_bias") or "N/A").upper()
    h1 = str(summary.get("h1_bias") or "N/A").upper()
    m30 = str(summary.get("m30_bias") or "N/A").upper()
    m5 = str(data.get("trigger") or "WAIT").upper()

    lo = safe_float(zone.get("price_low", zone.get("low")), 0.0)
    hi = safe_float(zone.get("price_high", zone.get("high")), 0.0)
    z_src = str(zone.get("source") or "SMC").upper()
    z_tf = str(zone.get("timeframe") or "").strip()
    z_type = str(zone.get("zone_type") or zone.get("type") or "").upper()
    z_grd = str(zone.get("grade") or "N/A").strip()
    z_sc = zone.get("score")
    z_sc_txt = f"{safe_float(z_sc):.1f}" if z_sc is not None else "N/A"

    u_spec_dash = tk.get("unit_spec") or extract_unit_spec_from_container(snapshot)
    tracked_pos = data.get("tracked_position") if isinstance(data.get("tracked_position"), dict) else None
    radar_cand_d = data.get("radar_candidate") if isinstance(data.get("radar_candidate"), dict) else None
    focus_lines = []
    if feed_st == "CLOSED":
        focus_lines.append(" ENTRY FOCUS : NONE | WAIT (MARKET CLOSED — revalidasi saat OPEN)")
    elif lo > 0 and hi > 0 and not (tracked_pos and tracked_pos.get("zone_id") and tracked_pos.get("zone_id") == zone.get("zone_id")):
        main_zone_txt = f"{data['direction']} | {z_tf} {z_type} {lo:.2f}-{hi:.2f} (G:{z_grd}/{z_sc_txt}) [{z_src}]"
        dist_v = data.get("distance")
        if dist_v is None or feed_st in ("DISCONNECTED", "STALE"):
            dist_txt = "N/A"
        elif dist_v <= 0.0:
            dist_txt = f"{format_price_distance(0.0, spec=u_spec_dash, include_mt5_points=True)} (IN ZONE | {data.get('touch_display', 'IN ZONE')})"
        else:
            side_lbl = "ABOVE" if lo > price_val else "BELOW"
            dist_txt = f"{format_price_distance(dist_v, spec=u_spec_dash, include_mt5_points=True)} ({side_lbl} | {data.get('monitor_state', 'RADAR')})"
        focus_lines.extend(wrap_dashboard_field(" ENTRY FOCUS : ", main_zone_txt, term_w, max_lines=2))
        focus_lines.extend(wrap_dashboard_field(" DISTANCE    : ", dist_txt, term_w, max_lines=1))
    else:
        focus_lines.append(" ENTRY FOCUS : NONE | WAIT")

    if tracked_pos:
        tp_px = f"@ {tracked_pos['open_price']:.2f}" if tracked_pos.get("open_price", 0.0) > 0 else ""
        tp_z = f"{tracked_pos['zone_low']:.2f}-{tracked_pos['zone_high']:.2f}" if tracked_pos.get("zone_low", 0.0) > 0 else "N/A"
        tp_hist_lbl = "RUNNING — HISTORICAL TRACKING" if tracked_pos.get("historical_tracking") else "RUNNING"
        tp_txt = f"#{tracked_pos['ticket']} {tracked_pos['direction']} {tp_px} | Match:{tracked_pos['match_status']} | Zone {tp_z} [{tp_hist_lbl}]".replace("  ", " ")
        focus_lines.extend(wrap_dashboard_field(" TRACKED POS : ", tp_txt, term_w, max_lines=2))

    if isinstance(radar_cand_d, dict) and safe_float(radar_cand_d.get("price_low", radar_cand_d.get("low")), 0.0) > 0:
        rc_lo = safe_float(radar_cand_d.get("price_low", radar_cand_d.get("low")), 0.0)
        rc_hi = safe_float(radar_cand_d.get("price_high", radar_cand_d.get("high")), 0.0)
        rc_tf = str(radar_cand_d.get("timeframe") or "").strip()
        rc_zt = str(radar_cand_d.get("zone_type") or "ZONE").upper()
        rc_dist = safe_float(radar_cand_d.get("distance"), max(rc_lo - price_val, price_val - rc_hi, 0.0) if price_val > 0 else 0.0)
        rc_txt = f"{rc_tf} {rc_zt} {rc_lo:.2f}-{rc_hi:.2f} (G:{radar_cand_d.get('grade', '-')} | {format_price_distance(rc_dist, spec=u_spec_dash, include_mt5_points=True)})".strip()
        focus_lines.extend(wrap_dashboard_field(" RADAR       : ", rc_txt, term_w, max_lines=1))
    elif isinstance(far_ref, dict):
        f_lo = safe_float(far_ref.get("price_low", far_ref.get("low")), 0.0)
        f_hi = safe_float(far_ref.get("price_high", far_ref.get("high")), 0.0)
        f_tf = str(far_ref.get("timeframe") or "").strip()
        f_zt = str(far_ref.get("rank_label") or far_ref.get("zone_type") or far_ref.get("type") or "ZONE").upper()
        f_dist = (
            max(f_lo - price_val, price_val - f_hi, 0.0)
            if (price_val > 0 and f_lo > 0 and f_hi >= f_lo)
            else safe_float(far_ref.get("distance", far_ref.get("distance_points")), 0.0)
        )
        f_atr = far_ref.get("distance_atr_ratio") if far_ref.get("distance_atr_ratio") is not None else far_ref.get("distance_atr")
        f_atr_s = f"/{safe_float(f_atr):.1f}xATR" if f_atr is not None else ""
        far_cat_d = far_ref.get("far_category") or ("EXTREME_FAR" if (f_atr is not None and safe_float(f_atr) > 15.0) else "FAR_INTRADAY")
        far_txt = f"{far_cat_d} | {f_tf} {f_zt} {f_lo:.2f}-{f_hi:.2f} ({format_price_distance(f_dist, spec=u_spec_dash, include_mt5_points=True)}{f_atr_s})".replace("  ", " ")
        focus_lines.extend(wrap_dashboard_field(" RADAR (FAR) : ", far_txt, term_w, max_lines=1))

    # Pisahkan tegas Active Focus vs SNR Reference bila tidak overlap
    snr_z = data.get("snr_focus_zone") or data.get("snr_zone") or {}
    snr_lo = safe_float(snr_z.get("low", snr_z.get("price_low")), 0.0)
    snr_hi = safe_float(snr_z.get("high", snr_z.get("price_high")), 0.0)
    confl_src = data.get("snr_confluence_source")
    snr_is_far = bool(
        isinstance(snr_z, dict)
        and (
            snr_z.get("is_far_reference")
            or str(snr_z.get("zone_relevance") or "").upper() == "FAR_REFERENCE"
            or snr_z.get("is_distant_historical")
            or str(snr_z.get("lifecycle_status") or snr_z.get("status") or "").upper() == "HISTORICAL"
            or safe_float(snr_z.get("distance_atr"), 0.0) > 12.0
        )
    )
    if data.get("confluence") == "ALIGNED" and isinstance(confl_src, dict):
        c_lbl = str(confl_src.get("rank_label") or "SNR").strip()
        c_ov = safe_float(data.get("snr_confluence_overlap_points"), 0.0)
        snr_ref_txt = f"OVERLAP {c_lbl} {safe_float(confl_src.get('low')):.2f}-{safe_float(confl_src.get('high')):.2f} ({format_price_distance(c_ov, spec=u_spec_dash)})"
    elif snr_lo > 0 and snr_hi > 0:
        s_lbl = str(snr_z.get("rank_label") or "SNR").strip()
        s_tf = str(snr_z.get("timeframe") or "").strip()
        s_zt_ref = str(snr_z.get("zone_type") or ("RESISTANCE" if s_lbl.upper().startswith("R") else "SUPPORT")).upper()
        snr_not_live = not _is_live_active_snr_zone(snr_z, s_zt_ref, price_val)
        s_tag = "FAR ZONE / REFERENCE ONLY" if snr_is_far else ("NOT ACTIVE / REFERENCE ONLY" if snr_not_live else "SEPARATED")
        s_ts = str(snr_z.get("confirmed_time") or snr_z.get("confirmation_time") or snr_z.get("formation_time") or "").strip()
        s_ts_part = f" | confirmed {s_ts[:16]}" if s_ts else ""
        if data.get("snr_snapshot_id"):
            s_ts_part += f" | snap {str(data['snr_snapshot_id'])[-12:]}"
        snr_ref_txt = f"{s_lbl} {s_tf} {snr_lo:.2f}-{snr_hi:.2f} ({s_tag}{s_ts_part})".replace("  ", " ")
    else:
        snr_ref_txt = "NONE"

    # Ringkasan MT5 Order Display (jumlah posisi, pending, matched, tanpa SL — TANPA daftar penuh / ticket acak)
    pending_setup_d = data.get("pending_setup") if isinstance(data.get("pending_setup"), dict) else {}
    alt_pend_d = list(data.get("alt_pending_setups") or [])
    matched_tkts_set = set()
    for p_it in [pending_setup_d, *alt_pend_d]:
        if isinstance(p_it, dict) and p_it.get("is_placed"):
            rec_it = p_it.get("mt5_reconciliation") or {}
            for mt in (rec_it.get("matched_tickets") or []):
                if mt is not None:
                    matched_tkts_set.add(mt)
    matched_cnt = len(matched_tkts_set)
    ord_list = list(mt5_snap.get("orders") or acc.get("orders") or [])
    pos_cnt = int(safe_float(acc.get("open_positions"), mt5_snap.get("symbol_positions_count", 0)))
    ord_cnt = int(safe_float(acc.get("pending_orders"), len(ord_list)))
    p_nosl_c = int(mt5_risk.get("positions_without_sl_count", safe_float(acc.get("position_without_sl"), 0)))
    o_nosl_c = int(mt5_risk.get("pending_without_sl_count", safe_float(acc.get("pending_without_sl"), 0)))
    no_sl_tot = p_nosl_c + o_nosl_c
    flt_pnl = safe_float(acc.get("floating_pnl"), 0.0)
    cur_lbl = str(acc.get("currency") or "USD")

    pending_disp = str(data.get("pending_display") or data.get("pending_status") or "WAIT")
    if matched_cnt == 0 and ord_cnt > 0 and "NO MATCH" not in pending_disp.upper():
        if pending_disp.upper() in ("WAIT", "NONE", "NOT PLACED"):
            pending_disp = f"MT5: NO MATCH | OTHER ORDERS: {ord_cnt}"
        else:
            pending_disp = f"{pending_disp} | MT5: NO MATCH | OTHER ORDERS: {ord_cnt}"
    elif matched_cnt == 0 and ord_cnt == 0 and pending_disp.upper() == "WAIT":
        pending_disp = "WAIT | MT5: NOT PLACED"

    if feed_st == "DISCONNECTED":
        pos_txt = "MT5 DISCONNECTED"
    else:
        pos_txt = f"Pos:{pos_cnt} ({flt_pnl:+.2f} {cur_lbl}) | Pend:{ord_cnt} | Match:{matched_cnt} | NoSL:{no_sl_tot}"

    confirmed_disp = str(data.get("confirmed_display") or data.get("confirmed") or "WAIT")
    signal_disp = str(data.get("signal_display") or data.get("signal") or "WAIT")
    reason_disp = str(data.get("reason") or "WAIT").replace("\n", " ").strip()
    next_act = str(data.get("next_action") or "WAIT").strip()

    # Ringkasan risiko singkat & jelas (tanpa daftar panjang ticket / tanpa pesan warning panjang)
    risk_txt = str(data.get("critical_risk_display") or "").strip()
    if feed_st == "DISCONNECTED" and "DISCONNECTED" not in risk_txt.upper():
        risk_txt = f"MT5 DISCONNECTED | {risk_txt}" if risk_txt else "MT5 DISCONNECTED [HIGH RISK]"
    elif not risk_txt:
        risk_txt = "AMAN (No critical risk)"
    news_sum = str(cycle.get("news_summary") or "").strip()

    # Rencana scanner (sumber sama dengan scan): read-only, bukan order aktual MT5
    _plan_txt = "WAIT (tidak ada rencana valid)"
    try:
        _cp = data.get("plan") if isinstance(data.get("plan"), dict) else {}
        _ps = data.get("pending_setup") if isinstance(data.get("pending_setup"), dict) else {}
        _kind = "CONFIRMED"
        _dir = str(_cp.get("arah") or "").upper()
        _e = safe_float(_cp.get("entry_acuan") or _cp.get("entry"), 0.0)
        _s, _t, _r = safe_float(_cp.get("sl"), 0.0), safe_float(_cp.get("tp1"), 0.0), safe_float(_cp.get("rr"), 0.0)
        if not (_dir in ("BUY", "SELL") and _e > 0 and _s > 0 and _t > 0):
            _kind = "PENDING"
            _dir = str(_ps.get("direction") or _ps.get("order_type") or "").upper()
            _dir = "BUY" if "BUY" in _dir else ("SELL" if "SELL" in _dir else "")
            _e = safe_float(_ps.get("entry"), 0.0)
            _s, _t, _r = safe_float(_ps.get("sl"), 0.0), safe_float(_ps.get("tp1"), 0.0), safe_float(_ps.get("rr"), 0.0)
        if _dir and _e > 0 and _s > 0 and _t > 0:
            _plan_txt = f"{_dir} {_kind} | E {_e:.2f} | SL {_s:.2f} | TP {_t:.2f} | RR 1:{_r:.2f} (rencana scanner)"
    except Exception:
        pass

    body_lines = [
        dash_div,
        *wrap_dashboard_field(" MTF BIAS    : ", f"H4={h4} | H1={h1} | M30={m30} | M5={m5}", term_w, max_lines=1),
        *focus_lines,
        *wrap_dashboard_field(" SNR REF     : ", snr_ref_txt, term_w, max_lines=1),
        dash_div,
        *wrap_dashboard_field(" PLAN        : ", _plan_txt, term_w, max_lines=2),
        *wrap_dashboard_field(" MT5 PENDING : ", pending_disp, term_w, max_lines=2),
        *wrap_dashboard_field(" MT5 ACCOUNT : ", pos_txt, term_w, max_lines=1),
        *wrap_dashboard_field(" ENTRY       : ", f"CONF={confirmed_disp} | SIG={signal_disp} | STATE={data.get('monitor_state', 'IDLE')}" + (" | LIVE SIGNAL: WAIT (MARKET CLOSED/DATA HISTORIS — validasi ulang saat OPEN)" if feed_st in ("CLOSED", "STALE") else ""), term_w, max_lines=2),
        *wrap_dashboard_field(" REASON      : ", f"{reason_disp} -> {next_act}", term_w, max_lines=2),
        *wrap_dashboard_field(" RISK WARNING: ", risk_txt, term_w, max_lines=2),
    ]
    if news_sum:
        compact_news = news_sum.replace("NEWS RISK : ", "").replace(" | NEXT EVENT : ", " | ")
        body_lines.extend(wrap_dashboard_field(" US NEWS     : ", compact_news, term_w, max_lines=1))
    if data.get("snr_breakout_watch"):
        body_lines.extend(wrap_dashboard_field(" BREAK WATCH : ", str(data["snr_breakout_watch"]), term_w, max_lines=1))

    ev_list = []
    _seen_ev_norm = set()
    for e_raw in (recent_events or []):
        e_str = str(e_raw).strip()
        if not e_str:
            continue
        e_norm = re.sub(r"^\[[^\]]+\]\s*", "", e_str).strip().upper()
        if e_norm in _seen_ev_norm:
            continue
        _seen_ev_norm.add(e_norm)
        ev_list.append(e_str)
    if ev_list:
        body_lines.append(dash_div)
        body_lines.append(" RECENT EVENTS:")
        for ev_ln in ev_list[-max_recent:]:
            body_lines.append("  * " + truncate_clean(ev_ln, max(20, term_w - 4)))
    body_lines.append(eq_div)

    dist_body_idx = next((idx_b for idx_b, ln_b in enumerate(body_lines) if ln_b.startswith(" DISTANCE    : ")), None)
    widget_meta["_live_patch"] = {
        "body_lines": body_lines,
        "dist_body_idx": dist_body_idx,
        "focus_bounds": (lo, hi) if (lo > 0 and hi >= lo) else None,
        "monitor_state": data.get("monitor_state", "IDLE"),
        "touch_display": data.get("touch_display", "IN ZONE"),
        "cycle_no": cyc_no,
        "term_w": term_w,
        "use_uni": use_uni,
        "last_ts": raw_ts,
    }
    lines = [*widget_lines, *body_lines]
    return fit_dashboard_text("\n".join(lines), width=term_w), widget_meta


def format_smart_monitor_dashboard(cycle_result, recent_events=None):
    """
    Dashboard ringkas anti-spam untuk in-place refresh pada command `pantau` (kompatibel Windows/PowerShell).
    Setiap baris dibatasi sesuai lebar terminal (`shutil.get_terminal_size()`) dengan wrapping/truncation terkontrol.
    """
    text, _ = format_smart_monitor_dashboard_with_meta(cycle_result, recent_events=recent_events)
    return text


def format_smart_monitor_report(cycle_result):
    """Format default `pantau` output in one compact, non-repeating terminal view."""
    data = _smart_monitor_context(cycle_result)
    zone = data["zone"]
    summary = data["summary"]
    snapshot = data["snapshot"]
    quote_fresh = (
        snapshot.get("data_ok") is not False and snapshot.get("live") is True
        and str(snapshot.get("data_freshness") or "").upper() not in ("STALE", "INVALID", "NO DATA")
    )
    price_txt = f"{data['price']:.2f}" if data["price"] > 0 and (not data["data_invalid"] or quote_fresh) else "N/A"
    h4 = str(summary.get("h4_bias") or "N/A").upper()
    h1 = str(summary.get("h1_bias") or "N/A").upper()
    snr_h1 = str((data["snr_scan"].get("market_bias") or {}).get("h1") or "N/A").upper()
    htf_alignment = "ALIGNED" if h1 in ("BULLISH", "BEARISH") and h1 == snr_h1 else "PARTIAL"
    m30 = str(summary.get("m30_bias") or "N/A").upper()
    m5 = data["trigger"]
    lo = safe_float(zone.get("price_low", zone.get("low")), 0.0)
    hi = safe_float(zone.get("price_high", zone.get("high")), 0.0)
    _u_rep = extract_unit_spec_from_container(data.get("snapshot"))
    dist_val_rep = data.get("distance")
    dist_rep_str = (
        f"{format_price_distance(0.0, spec=_u_rep, include_mt5_points=True)} (IN ZONE)"
        if (dist_val_rep is not None and dist_val_rep <= 0.0)
        else (format_price_distance(dist_val_rep, spec=_u_rep, include_mt5_points=True) if dist_val_rep is not None else "N/A")
    )
    raw_score = zone.get("score")
    if raw_score is None:
        score_disp = "N/A"
    else:
        score_val = safe_float(raw_score, 0.0)
        if abs(score_val - round(score_val)) < 1e-4:
            score_disp = f"{int(round(score_val))}"
        elif abs(score_val - round(score_val, 1)) < 1e-4:
            score_disp = f"{score_val:.1f}"
        else:
            score_disp = f"{score_val:.2f}"
    def _format_snr_source_part(rank_label):
        lbl = str(rank_label or "").strip()
        upper = lbl.upper()
        if not lbl or upper in ("SNR", "SNR ZONE", "ZONE"):
            return "SNR ZONE"
        if upper.startswith("SNR "):
            return lbl
        return f"SNR {lbl}"

    snr_z = data.get("snr_focus_zone") or data.get("snr_zone") or {}
    confl_snr_z = data.get("snr_confluence_source")
    z_tf = str(zone.get("timeframe") or "").strip()
    z_type = str(zone.get("zone_type") or zone.get("type") or "").strip()
    z_src = str(zone.get("source") or "SMC").strip().upper()
    z_rank = str(zone.get("rank_label") or "").strip()
    base_tag = f"{z_tf} {z_type}".strip()
    if z_src == "SNR":
        src_part = _format_snr_source_part(z_rank)
        tag_inner = f"{base_tag} | {src_part}".strip(" |")
    elif data.get("confluence") == "ALIGNED" and isinstance(confl_snr_z, dict):
        c_tf = str(confl_snr_z.get("timeframe") or "").strip()
        c_zt = str(confl_snr_z.get("zone_type") or "").strip().upper()
        c_lbl = str(confl_snr_z.get("rank_label") or "").strip()
        snr_desc = " ".join(part for part in (c_tf, c_zt, _format_snr_source_part(c_lbl)) if part)
        tag_inner = f"{base_tag} + {snr_desc}".strip(" +")
    else:
        tag_inner = base_tag
    z_tag = f" ({tag_inner})" if tag_inner else ""

    z_grade = str(zone.get("grade") or "N/A").strip()
    zt_raw = str(zone.get("zone_type") or data.get("zone_type") or ("SUPPLY" if data.get("direction") == "SELL" else ("DEMAND" if data.get("direction") == "BUY" else "ZONE"))).strip().upper()
    zone_header_label = f"SMC ACTIVE {zt_raw}" if (z_src == "SMC" and zt_raw in ("SUPPLY", "DEMAND")) else "ACTIVE SETUP ZONE"
    if lo > 0 and hi > 0:
        zone_lines = [
            f"SETUP DIRECTION     : {data['direction']}",
            f"{zone_header_label:<20}: {lo:.2f} - {hi:.2f}{z_tag}",
            f"ZONE DISTANCE       : {dist_rep_str}",
            f"GRADE               : {z_grade} ({score_disp}) [{z_src}]",
        ]
    else:
        zone_lines = [
            "ACTIVE FOCUS        : NONE | WAIT",
        ]
        far_ref_rep = data.get("far_reference_zone")
        if isinstance(far_ref_rep, dict):
            fr_lo = safe_float(far_ref_rep.get("price_low", far_ref_rep.get("low")), 0.0)
            fr_hi = safe_float(far_ref_rep.get("price_high", far_ref_rep.get("high")), 0.0)
            fr_tf = str(far_ref_rep.get("timeframe") or "").strip()
            fr_zt = str(far_ref_rep.get("rank_label") or far_ref_rep.get("zone_type") or far_ref_rep.get("type") or "ZONE").upper()
            fr_dist = (
                max(fr_lo - data["price"], data["price"] - fr_hi, 0.0)
                if (data["price"] > 0 and fr_lo > 0 and fr_hi >= fr_lo)
                else safe_float(far_ref_rep.get("distance", far_ref_rep.get("distance_points")), 0.0)
            )
            zone_lines.append(
                f"FAR ZONE            : FAR ZONE / REFERENCE ONLY — {fr_tf} {fr_zt} {fr_lo:.2f}-{fr_hi:.2f} ({format_price_distance(fr_dist, spec=_u_rep, include_mt5_points=True)})".replace("  ", " ")
            )

    snr_lo = safe_float(snr_z.get("low", snr_z.get("price_low")), 0.0)
    snr_hi = safe_float(snr_z.get("high", snr_z.get("price_high")), 0.0)
    snr_focus_line = None
    if snr_lo > 0 and snr_hi > 0 and (z_src != "SNR" or abs(snr_lo - lo) > 0.01 or abs(snr_hi - hi) > 0.01):
        s_lbl = str(snr_z.get("rank_label") or "SNR").strip()
        s_tf = str(snr_z.get("timeframe") or "").strip()
        s_grd = str(snr_z.get("grade") or "N/A").strip()
        s_life = str(snr_z.get("lifecycle_status") or snr_z.get("status") or "ACTIVE").upper()
        s_state_tag = f" | {s_life.replace('_', ' ')}" if s_life not in ("ACTIVE", "TOUCHED", "SWEEP", "RADAR", "") else ""
        s_tag = f"{s_lbl} {s_tf}".strip()
        s_rsn_short = str(data.get("snr_selection_reason_short") or "").strip()
        s_rsn_tag = (
            f" | {s_rsn_short}"
            if s_rsn_short in ("PRICE AT SUPPORT", "NO ACTIVE RESISTANCE", "PRICE AT RESISTANCE", "NO ACTIVE SUPPORT")
            else ""
        )
        ref_lbl = "SNR REFERENCE" if (z_src != "SNR" and data.get("confluence") != "ALIGNED") else "SNR FOCUS"
        sep_note = " | SEPARATED (NO OVERLAP)" if (z_src != "SNR" and data.get("confluence") != "ALIGNED") else ""
        snr_focus_line = f"{ref_lbl:<13}: {snr_lo:.2f} - {snr_hi:.2f} ({s_tag} | Grade {s_grd}{s_state_tag}{s_rsn_tag}{sep_note})"

    snr_confl_source_line = None
    if z_src != "SNR" and lo > 0 and hi > 0:
        confl_src_disp = str(data.get("snr_confluence_source_display") or "NONE").strip()
        snr_confl_source_line = f"SNR CONFLUENCE SOURCE : {confl_src_disp}"

    smc_z = data.get("smc_zone") or {}
    smc_lo = safe_float(smc_z.get("price_low", smc_z.get("low")), 0.0)
    smc_hi = safe_float(smc_z.get("price_high", smc_z.get("high")), 0.0)
    smc_zone_line = None
    if z_src == "SNR" and smc_lo > 0 and smc_hi > 0 and (abs(smc_lo - lo) > 0.01 or abs(smc_hi - hi) > 0.01):
        m_tf = str(smc_z.get("timeframe") or "").strip()
        m_zt = str(smc_z.get("zone_type") or "").strip()
        m_grd = str(smc_z.get("grade") or "N/A").strip()
        smc_zone_line = f"SMC ZONE    : {smc_lo:.2f} - {smc_hi:.2f} ({m_tf} {m_zt} | Grade {m_grd})".replace("  ", " ")

    pending_disp = data.get("pending_display", data.get("pending_status", "WAIT"))
    confirmed_disp = data.get("confirmed_display", data.get("confirmed", "WAIT"))
    touch_disp = data.get("touch_display", "NOT TOUCHED")
    trigger_disp = data.get("trigger", "WAIT")
    focused_states = ("RUNNING", "ZONE_TOUCHED", "HISTORICAL_TOUCH", "WAIT_NO_ACTIVE_ZONE", "HISTORY_UNCERTAIN")
    status_text = "WAIT" if data["data_invalid"] and data["monitor_state"] not in focused_states else data["monitor_state"]
    alert_text = data.get("alert_display", "NONE")
    alert_text = str(alert_text).replace("\n", " ").strip()
    signal = data.get("signal_display", data.get("signal", "WAIT"))
    main_reason = str(data["reason"] or summary.get("main_reason") or "N/A").replace("\n", " ").strip()
    if data["reason"] == "WAITING FOR ZONE PULLBACK" and summary.get("main_reason"):
        main_reason = str(summary["main_reason"]).replace("\n", " ").strip()

    is_mkt_closed_rep = (
        str(snapshot.get("market_status") or (data.get("decision") or {}).get("market_status") or "").upper() == "CLOSED"
        or str(snapshot.get("snapshot_status") or "").upper() == "MARKET_CLOSED"
    )
    entry_perm = (
        "WAIT (MARKET CLOSED)"
        if is_mkt_closed_rep
        else ("GRANTED" if (str(signal).upper() in ("BUY", "SELL") or str(data.get("signal_status", "")).upper() == "SIGNAL_CONFIRMED") else "WAIT")
    )
    entry_status_lines = [
        "ENTRY STATUS",
        f"M5 TRIGGER          : {trigger_disp}",
        f"ENTRY PERMISSION    : {entry_perm}",
    ]
    if entry_perm != "GRANTED":
        # Formulate clean dynamic blocking reason
        raw_reasons = (data.get("decision") or {}).get("reasons") or []
        blocker_line = ""
        # Check SL distance limit first if mentioned or present in decision/plan
        plan_dict = data.get("plan") or {}
        p_ent = safe_float(plan_dict.get("entry_acuan") or plan_dict.get("entry"), 0.0)
        p_sl = safe_float(plan_dict.get("sl"), 0.0)
        sl_dist = abs(p_ent - p_sl) if (p_ent > 0 and p_sl > 0) else 0.0
        for r in raw_reasons:
            r_str = str(r)
            if "jarak sl" in r_str.lower() or "melebihi batas" in r_str.lower():
                blocker_line = r_str.upper()
                break
        if not blocker_line and sl_dist > SL_JARAK_MAKS:
            blocker_line = f"SL DISTANCE {format_price_distance(sl_dist, spec=_u_rep, include_mt5_points=True)} > MAX {SL_JARAK_MAKS:g} USD"
        elif not blocker_line and raw_reasons:
            blocker_line = str(raw_reasons[0])
        elif not blocker_line:
            blocker_line = main_reason
        entry_status_lines.append(f"BLOCKING REASON     : {blocker_line}")

    entry_status_lines.extend([
        f"PENDING             : {pending_disp}",
        *( [f"ALT PENDING         : {data['alt_pending_display']}"] if data.get("alt_pending_display") else [] ),
        f"CONFIRMED           : {confirmed_disp}",
        f"TOUCH               : {touch_disp}",
    ])

    smc_focus = zone or data["decision"].get("primary_zone") or {}
    smc_block = format_smc_zone_lines(
        smc_focus, data["decision"], allow_plan=not data["data_invalid"],
        role="SMC ACTIVE" if zone else "SMC PRIMARY",
    ) if smc_focus else []
    if zone:
        smc_block = smc_block[1:]

    radar_cand = data.get("radar_candidate")
    radar_lines = []
    if isinstance(radar_cand, dict) and radar_cand:
        r_lo = safe_float(radar_cand.get("price_low", radar_cand.get("low")), 0.0)
        r_hi = safe_float(radar_cand.get("price_high", radar_cand.get("high")), 0.0)
        r_tf = str(radar_cand.get("timeframe") or "").strip()
        r_zt = str(radar_cand.get("zone_type") or "").strip()
        r_grd = str(radar_cand.get("grade") or "N/A").strip()
        r_raw_sc = radar_cand.get("score")
        r_sc_txt = f"{safe_float(r_raw_sc):.1f}" if r_raw_sc is not None else "N/A"
        r_dist = safe_float(
            radar_cand.get("distance"),
            max(r_lo - data["price"], data["price"] - r_hi, 0.0) if data["price"] > 0 else 0.0,
        )
        radar_lines = [
            "RADAR CANDIDATE",
            f"RADAR       : {r_lo:.2f} - {r_hi:.2f} ({r_tf} {r_zt} | Grade {r_grd} {r_sc_txt} | Dist {format_price_distance(r_dist, spec=_u_rep, include_mt5_points=True)} | RADAR ALT)",
        ]

    tracked_pos_rep = data.get("tracked_position") if isinstance(data.get("tracked_position"), dict) else None
    tracked_pos_lines = []
    if tracked_pos_rep:
        tp_px_r = f"@ {tracked_pos_rep['open_price']:.2f}" if tracked_pos_rep.get("open_price", 0.0) > 0 else ""
        tp_z_r = f"{tracked_pos_rep['zone_low']:.2f}-{tracked_pos_rep['zone_high']:.2f}" if tracked_pos_rep.get("zone_low", 0.0) > 0 else "N/A"
        tp_mode_r = "RUNNING (HISTORICAL TRACKING)" if tracked_pos_rep.get("historical_tracking") else "RUNNING"
        tracked_pos_lines = [
            "TRACKED POSITION (MT5 OPEN)",
            f"POSITION    : #{tracked_pos_rep['ticket']} {tracked_pos_rep['direction']} {tp_px_r} | Match: {tracked_pos_rep['match_status']} | Zone: {tp_z_r} [{tp_mode_r}]".replace("  ", " "),
        ]

    smc_snr_short_disp = (
        data["confluence"]
        if data.get("confluence") == "ALIGNED"
        else f"{data['confluence']} (SEPARATED — BUKAN FULL CONFLUENCE)"
        if data.get("confluence") == "PARTIAL"
        else data["confluence"]
    )
    lines = [
        "==========================================",
        "       XAUUSD SMART MONITOR v2.5",
        "==========================================",
        f"PRICE       : {price_txt}",
        f"SMC H4 / H1 : {h4} / {h1}",
        f"SNR H1      : {snr_h1} | HTF: {htf_alignment}",
        f"M30 / M5    : {m30} / {m5}",
        "------------------------------------------",
        "ENTRY FOCUS",
        "------------------------------------------",
        *zone_lines,
        *( [smc_zone_line] if smc_zone_line else [] ),
        *( [snr_focus_line] if snr_focus_line else [] ),
        *( [snr_confl_source_line] if snr_confl_source_line else [] ),
        f"CONFLUENCE  : {data.get('confluence_display', data['confluence'])}",
        f"SMC + SNR   : {smc_snr_short_disp}",
        *radar_lines,
        *tracked_pos_lines,
        *entry_status_lines,
        "SMART MONITOR",
        f"STATE       : {status_text}",
        *( [f"SNR WATCH   : {data['snr_breakout_watch']}"] if data.get("snr_breakout_watch") else [] ),
        *( [f"HIST. EVENT : {data['historical_event']}"] if data.get("historical_event") and status_text in ("HISTORICAL_TOUCH", "WAIT_NO_ACTIVE_ZONE") else [] ),
        f"NEXT        : {data['next_action']}",
        f"ALERT       : {alert_text}",
        *(
            [f"RISK ALERT  : {data['critical_risk_display']}"]
            if data.get("critical_risk_display") else []
        ),
        "FINAL",
        f"SIGNAL      : {signal}",
        f"REASON      : {main_reason}",
        "==========================================",
    ]
    news_summary = str((cycle_result or {}).get("news_summary") or "").strip() if isinstance(cycle_result, dict) else ""
    if news_summary and len(lines) < 36:
        compact_news = news_summary.replace("NEWS RISK : ", "").replace(" | NEXT EVENT : ", " | ")
        lines.insert(-1, f"US NEWS     : {compact_news}")
    gold_summary = str((cycle_result or {}).get("gold_news_summary") or "").strip() if isinstance(cycle_result, dict) else ""
    if gold_summary and len(lines) < 38:
        compact_gold = gold_summary.replace("GOLD NEWS : ", "")
        lines.insert(-1, f"GOLD NEWS   : {compact_gold}")
    if not snapshot.get("snapshot_status") and data["price"] <= 0:
        lines[3] = "PRICE       : N/A"
    return "\n".join(lines)


def format_smart_monitor_detail(cycle_result):
    """Structured lifecycle, validation, data freshness, and blocker details."""
    data = _smart_monitor_context(cycle_result)
    zone = data["zone"]
    plan = data["plan"]
    snapshot = data["snapshot"]
    decision = data["decision"]
    summary = data["summary"]
    raw_stages = decision.get("m5_stages") or decision.get("stages") or {}
    stages = raw_stages if _m5_stages_match_zone(decision, zone) else {}
    m5_conf_det = data.get("m5_confirmation") if isinstance(data.get("m5_confirmation"), dict) else None
    struct_det_line = data["trigger"]
    if m5_conf_det and data["trigger"] == "CONFIRMED":
        m5_tf_d = str(m5_conf_det.get("timeframe") or "").strip()
        m5_ts_d = str(m5_conf_det.get("timestamp") or "").strip()
        m5_meta_parts = [p for p in (m5_tf_d, m5_ts_d) if p]
        if m5_meta_parts:
            struct_det_line = f"{data['trigger']} ({' @ '.join(m5_meta_parts)})"
    active_setup = data.get("active_setup") or {}
    zone_id = str(active_setup.get("zone_id") or zone.get("zone_id") or "N/A")
    setup_type = str(active_setup.get("setup_type") or decision.get("setup_type") or plan.get("setup_type") or "N/A")
    direction = data["direction"]
    setup_id = active_setup.get("setup_id") or (f"{zone_id}:{direction}:{setup_type}" if zone_id != "N/A" else "N/A")
    blockers = [str(x) for x in (decision.get("reasons") or []) if x]
    if data["data_invalid"]:
        blockers.insert(0, str(snapshot.get("error_reason") or f"Data freshness: {data['freshness']}"))
    if not blockers:
        blockers = [str(summary.get("main_reason") or "N/A")]
    pending_setup = data.get("pending_setup") or {}
    pending_status = str(data.get("pending_status") or "WAIT").upper()
    plan_st_det = str(pending_setup.get("plan_status") or ("READY" if pending_status.startswith("READY") else pending_status)).upper()
    broker_st_det = str(pending_setup.get("broker_order_status") or "UNKNOWN").upper()
    exp = pending_setup.get("suggested_expiration") or pending_setup.get("expires_at") or "N/A"
    p_det_lines = [
        f"PENDING PLAN (PLAN: {plan_st_det} | MT5: {broker_st_det})",
        f"PLAN STATUS : {plan_st_det}",
        f"MT5 STATUS  : {broker_st_det}",
        f"STATUS      : {pending_status}",
    ]
    if pending_status.startswith("READY") and pending_setup.get("zone_id") and pending_setup.get("zone_id") != "N/A":
        p_det_lines.append(
            f"ZONE ID     : {pending_setup['zone_id']} ({pending_setup.get('source_zone_label', 'N/A')} | "
            f"{pending_setup.get('plan_role', 'SELECTED_PENDING')} | {pending_setup.get('source_role', 'N/A')})"
        )
        if pending_setup.get("entry") and pending_setup.get("sl") and pending_setup.get("tp1"):
            rr_det = f"1:{safe_float(pending_setup.get('rr')):.2f}" if pending_setup.get("rr") else "N/A"
            p_det_lines.append(
                f"LEVELS      : {pending_setup.get('order_type', '')} {safe_float(pending_setup['entry']):.2f} | "
                f"SL {safe_float(pending_setup['sl']):.2f} | TP1 {safe_float(pending_setup['tp1']):.2f} | RR {rr_det}"
            )
        p_recon = pending_setup.get("mt5_reconciliation")
        if isinstance(p_recon, dict):
            p_det_lines.append(f"MT5 EVIDENCE: {format_pending_reconciliation_evidence(p_recon)}")
    p_det_lines.append(f"EXPIRATION  : {exp}")
    for alt_p in (data.get("alt_pending_setups") or [])[:2]:
        alt_rr = f"1:{safe_float(alt_p.get('rr')):.2f}" if alt_p.get("rr") else "N/A"
        alt_lvls = ""
        if alt_p.get("entry") and alt_p.get("sl") and alt_p.get("tp1"):
            alt_lvls = (
                f" | {alt_p.get('order_type', '')} {safe_float(alt_p['entry']):.2f} "
                f"/ SL {safe_float(alt_p['sl']):.2f} / TP1 {safe_float(alt_p['tp1']):.2f}"
            )
        alt_plan_st = str(alt_p.get("plan_status") or "READY").upper()
        alt_brk_st = str(alt_p.get("broker_order_status") or "UNKNOWN").upper()
        p_det_lines.append(
            f"ALT PENDING : {alt_p.get('zone_id', 'N/A')} ({alt_p.get('source_zone_label', 'N/A')} | "
            f"{alt_p.get('plan_role', 'ALT_PENDING')} | {alt_p.get('source_role', 'N/A')} | "
            f"PLAN: {alt_plan_st} | MT5: {alt_brk_st} | RR {alt_rr}{alt_lvls})"
        )
        alt_recon = alt_p.get("mt5_reconciliation")
        if isinstance(alt_recon, dict):
            p_det_lines.append(f"ALT MT5     : {format_pending_reconciliation_evidence(alt_recon)}")
    if zone_id and zone_id != "N/A" and str(zone.get("source") or "SMC").upper() != "SNR":
        focus_sum = smc_zone_summary(zone, decision, allow_plan=not data["data_invalid"])
        if focus_sum.get("entry", 0.0) <= 0:
            p_det_lines.append(
                f"FOCUS PLAN  : {zone_id} (ACTIVE_FOCUS | SL/TP1 N/A — {focus_sum.get('note', 'PLAN UNAVAILABLE')})"
            )
    snr_z_det = data.get("snr_focus_zone") or data.get("snr_zone") or {}
    snr_z_det_lo = safe_float(snr_z_det.get("low", snr_z_det.get("price_low")), 0.0)
    snr_z_det_hi = safe_float(snr_z_det.get("high", snr_z_det.get("price_high")), 0.0)
    if snr_z_det_lo > 0 and snr_z_det_hi > 0:
        snr_z_det_sc = f"{safe_float(snr_z_det.get('score')):.2f}" if snr_z_det.get("score") is not None else "N/A"
        snr_det_line = (
            f"{snr_z_det.get('rank_label', 'SNR')} {snr_z_det.get('timeframe', '')} "
            f"{snr_z_det_lo:.2f}-{snr_z_det_hi:.2f} | Grade {snr_z_det.get('grade', 'N/A')} ({snr_z_det_sc}) | "
            f"{data.get('snr_selection_reason', 'N/A')}"
        ).replace("  ", " ")
    else:
        snr_det_line = f"NONE ({data.get('snr_selection_reason', 'N/A')})"
    snr_confl_det = data.get("snr_confluence_source")
    _u_det_sm = extract_unit_spec_from_container(snapshot)
    if isinstance(snr_confl_det, dict):
        c_det_lo = safe_float(snr_confl_det.get("low", snr_confl_det.get("price_low")), 0.0)
        c_det_hi = safe_float(snr_confl_det.get("high", snr_confl_det.get("price_high")), 0.0)
        c_det_sc = f"{safe_float(snr_confl_det.get('score')):.2f}" if snr_confl_det.get("score") is not None else "N/A"
        c_det_ov = safe_float(data.get("snr_confluence_overlap_points"), 0.0)
        snr_confl_det_line = (
            f"{snr_confl_det.get('rank_label', 'SNR')} {snr_confl_det.get('timeframe', '')} "
            f"{c_det_lo:.2f}-{c_det_hi:.2f} | Grade {snr_confl_det.get('grade', 'N/A')} ({c_det_sc}) | "
            f"overlap {format_price_distance(c_det_ov, spec=_u_det_sm)}"
        ).replace("  ", " ")
    else:
        snr_confl_det_line = str(data.get("snr_confluence_source_display") or "NONE")
    lines = [
        format_smart_monitor_report(cycle_result),
        "",
        "LIFECYCLE DETAIL",
        f"SETUP ID    : {setup_id}",
        f"ZONE STATUS : {data['lifecycle']}",
        f"MONITOR     : {data['monitor_state']}",
        f"SIGNAL      : {summary.get('signal_status', 'NO SETUP')}",
        f"SNR FOCUS   : {snr_det_line}",
        f"SNR CONFLUENCE SOURCE : {snr_confl_det_line}",
        f"CONFLUENCE  : {data['confluence']} ({data.get('confluence_reason', 'N/A')})",
        "",
        "VALIDATION",
        f"H4 / H1     : {summary.get('h4_bias', 'N/A')} / {summary.get('h1_bias', 'N/A')}",
        f"M30         : {summary.get('m30_bias', 'N/A')}",
        f"ZONE TOUCH  : {data.get('touch_display', data['touch'])}",
        f"SWEEP       : {_stage_status(stages.get('liquidity_sweep'), 'NOT_CHECKED' if not stages else summary.get('liquidity_sweep', 'WAIT'))}",
        f"STRUCTURE   : {struct_det_line}",
        f"DISPLACEMENT: {_stage_status(stages.get('displacement'), 'NOT_CHECKED' if not stages else summary.get('displacement', 'WAIT'))}",
        f"R:R CHECK   : {_stage_status(stages.get('risk_reward'), 'NOT_CHECKED' if not stages else summary.get('rr_stage', 'WAIT'))}",
        f"SPREAD      : {format_price_distance(safe_float(snapshot.get('spread'), 0.0), spec=_u_det_sm, include_mt5_points=True)}" if snapshot.get("spread") is not None else "SPREAD      : N/A",
        f"FRESHNESS   : {data['freshness']} | CONNECTION: {snapshot.get('connection_status', 'UNKNOWN')}",
        f"LAST M5     : {(snapshot.get('last_candle_times') or {}).get('M5', snapshot.get('last_closed_m5', 'N/A'))}",
        "",
        *p_det_lines,
        "",
        "BLOCKERS",
    ]
    lines.extend(f"- {reason.replace(chr(10), ' ')[:180]}" for reason in blockers)
    lines.extend([
        "",
        f"SMC + SNR   : {data['confluence']} (informational only)",
        f"NEXT ACTION : {data['next_action']}",
    ])
    cat_al = data.get("categorized_alerts") or {}
    if any(cat_al.get(k) for k in ("risk", "market", "entry", "account", "historical")):
        lines.extend(["", "ALERT PRIORITY BREAKDOWN"])
        for cat_key, cat_lbl in (
            ("risk", "RISK ALERT"),
            ("market", "MARKET ALERT"),
            ("entry", "ENTRY ALERT"),
            ("account", "ACCOUNT ALERT"),
            ("historical", "HISTORICAL INFO"),
        ):
            for item_msg in (cat_al.get(cat_key) or []):
                lines.append(f"  [{cat_lbl}] {item_msg}")
    news_sum_det = str((cycle_result or {}).get("news_summary") or "").strip() if isinstance(cycle_result, dict) else ""
    gold_sum_det = str((cycle_result or {}).get("gold_news_summary") or "").strip() if isinstance(cycle_result, dict) else ""
    news_al_det = list((cycle_result or {}).get("news_alerts") or []) if isinstance(cycle_result, dict) else []
    gold_arts_det = list(((cycle_result or {}).get("news") or {}).get("gold_articles") or []) if isinstance(cycle_result, dict) else []
    if news_sum_det or gold_sum_det or news_al_det or gold_arts_det:
        lines.extend(["", "US NEWS RADAR & GOLD MARKET NEWS"])
        if news_sum_det:
            lines.append(news_sum_det)
        if gold_sum_det:
            lines.append(gold_sum_det)
        for ga in gold_arts_det[:3]:
            if isinstance(ga, dict) and ga.get("headline"):
                g_sent = str((ga.get("sentiment") or {}).get("direction") or "UNCONFIRMED")
                g_hl = str(ga.get("headline") or "").strip()
                if len(g_hl) > 90:
                    g_hl = g_hl[:87].rstrip() + "..."
                lines.append(f"  - [{ga.get('topic', 'Gold')} | {g_sent}] {g_hl} ({ga.get('publisher', 'N/A')})")
        for na in news_al_det:
            lines.append(na)
    smc_detail = (data["smc"].get("formatted_section") or "").strip()
    snr_detail = (data["snr"].get("formatted_section") or "").strip()
    if smc_detail:
        lines.extend(["", "SMC SCANNER DETAIL", smc_detail])
    if snr_detail:
        lines.extend(["", "SNR SCANNER DETAIL", snr_detail])
    return "\n".join(lines)


def _stage_status(stage, default="WAIT"):
    if isinstance(stage, dict):
        return stage.get("status") or default
    return stage or default


