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

from modules.config import (
    INVALIDATION_SETUP_SL_BREACHED,
    INVALIDATION_ZONE_STRUCTURAL,
    LOW_RISK_MAX_ATR_RATIO,
    LOW_RISK_MAX_DISTANCE_PT,
    SL_JARAK_MAKS,
    ZONE_APPROACH_DISTANCE_POINTS,
)
from modules.mt5_conn import (
    fetch_mt5_pending_orders_snapshot,
    find_pending_orders_on_invalid_zones,
    format_pending_compact_status,
    format_pending_reconciliation_evidence,
    reconcile_pending_plan_with_mt5,
    reconcile_pending_plans_batch,
    summarize_mt5_risk_warnings,
)
from modules.snr.breakout_detection import build_snr_breakout_watch, classify_snr_breakout_state
from modules.snr.data_provider import safe_float
from modules.snr.entry_validator import _is_live_active_snr_zone, select_live_snr_focus
from modules.snr.formatter import build_snr_zone_reason, format_snr_invalid_why
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
    format_invalid_zone_compact,
    format_mt5_sync_short,
    format_price_distance,
    format_trade_plan_lines,
    refresh_trade_plan_live_state,
    resolve_entry_blocker_code,
    resolve_trade_plan_focus,
    select_best_trade_plan,
    select_relevant_invalid_zone,
)
from modules.zone_scanner import (
    build_smc_zone_reason,
    format_smc_invalid_why,
    format_smc_zone_lines,
    is_far_reference_zone,
    resolve_analysis_direction,
    resolve_smc_setup_invalidation_price,
    resolve_smc_structure_invalidation_price,
    smc_zone_summary,
)


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
    htf_conflict=False,
    m5_eval_status=None,
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

    htf_ctx_p = sc.get("htf_context") if isinstance(sc.get("htf_context"), dict) else {}
    _m5_st_p = str(m5_eval_status or dec.get("m5_eval_status") or "").strip().upper()
    _htf_blk_p = bool(htf_conflict or htf_ctx_p.get("has_conflict") or htf_ctx_p.get("m30_conflict"))
    _early_blocked = bool(_htf_blk_p or _m5_st_p in ("UNCONFIRMED", "CONFLICT", "CONFIRMED_BLOCKED_HTF"))

    valid_items = []
    if not data_invalid and not _early_blocked:
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
                    if hasattr(tracker, "is_zone_invalidated") and tracker.is_zone_invalidated(live_z or {"zone_id": p_zid})[0]:
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
            or str(candidate.get("invalidation_type") or "").upper() in (INVALIDATION_SETUP_SL_BREACHED, INVALIDATION_ZONE_STRUCTURAL)
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
    snr_h1_bias = str(snr_mb.get("h1") or "").upper()
    snr_m30_bias = str(snr_mb.get("m30") or "").upper()
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
                "swing_class", "historical_reaction_count", "bounce_strength_atr", "confluence_info",
                "consolidation_structure", "touch_analysis", "touch_status", "lifecycle_event",
                "retest_info", "rejection_info", "score_breakdown",
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
            zone = dict(snr_zone)
            zone.update({
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
            })
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

    if (not smc_zone_ref or safe_float(smc_zone_ref.get("price_low", smc_zone_ref.get("low")), 0.0) <= 0) and far_reference_zone is None:
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
    _z_plans_ctx = decision.get("zone_plans") if isinstance(decision.get("zone_plans"), dict) else {}
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
        if not data_invalid and smc_zone_usable and cz_id and cz_id == z_id and not ((cycle.get("focus_detail") or {}).get("SMC") or {}).get("historical_touch"):
            continue
        if not data_invalid and p_status.startswith("READY") and p_zid and cz_id and cz_id == p_zid:
            continue
        radar_candidate = dict(cz)
        if price > 0 and radar_candidate.get("distance") is None:
            radar_candidate["distance"] = 0.0 if r_lo <= price <= r_hi else (r_lo - price if r_lo > price else price - r_hi)
        _zp_rc = _z_plans_ctx.get(cz_id) if cz_id else None
        _rc_rej = str(
            radar_candidate.get("pending_rejection_reason")
            or ((_zp_rc or {}).get("rejection_reason") if isinstance(_zp_rc, dict) else "")
            or radar_candidate.get("setup_eligibility")
            or ""
        ).strip()
        if _rc_rej.upper().startswith("NOT_ELIGIBLE"):
            radar_candidate["not_eligible_reason"] = _rc_rej
        elif radar_candidate.get("risk_rejected") or _rc_rej:
            radar_candidate["not_eligible_reason"] = f"NOT_ELIGIBLE ({_rc_rej})" if _rc_rej else "NOT_ELIGIBLE"
        elif data_invalid:
            radar_candidate["not_eligible_reason"] = "NOT_ELIGIBLE (MARKET CLOSED)" if snapshot_status == "MARKET_CLOSED" else "NOT_ELIGIBLE (DATA STALE)"
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

    dec_reasons_all = [str(r).strip() for r in (decision.get("reasons") or []) if str(r).strip()]
    specific_dec_reason = dec_reasons_all[0] if dec_reasons_all else ""
    htf_ctx_sm = scan.get("htf_context") if isinstance(scan.get("htf_context"), dict) else {}
    m30_dir_ctx = str(htf_ctx_sm.get("M30") or summary.get("m30_bias") or "").upper()
    htf_conflict_ctx = bool(
        htf_ctx_sm.get("has_conflict")
        or (h4_bias_ctx in ("BULLISH", "BEARISH") and h1_bias_ctx in ("BULLISH", "BEARISH") and h4_bias_ctx != h1_bias_ctx)
    )
    m30_conflict_ctx = bool(
        htf_ctx_sm.get("m30_conflict")
        or "CONFLICT" in str(summary.get("m30_bias") or "").upper()
        or (h4_bias_ctx in ("BULLISH", "BEARISH") and m30_dir_ctx in ("BULLISH", "BEARISH") and m30_dir_ctx != h4_bias_ctx)
    )
    _smc_m5_st = str(decision.get("m5_eval_status") or (stages.get("M5") or {}).get("status") or "").upper()
    _snr_m5_st = str(snr_decision.get("m5_eval_status") or "").upper()
    _m5_was_evaluated_conflict = bool(
        decision.get("m5_conflict_evaluated")
        or snr_decision.get("m5_conflict_evaluated")
        or _smc_m5_st in ("CONFIRMED", "UNCONFIRMED", "CONFIRMED_BLOCKED_HTF")
        or _snr_m5_st in ("CONFIRMED", "UNCONFIRMED")
    )
    if data_invalid or _smc_m5_st == "DATA_UNAVAILABLE" or _snr_m5_st == "DATA_UNAVAILABLE":
        trigger = "DATA_UNAVAILABLE"
        m5_confirmation = None
    elif _m5_was_evaluated_conflict:
        if _smc_m5_st in ("CONFIRMED", "CONFIRMED_BLOCKED_HTF") or _snr_m5_st == "CONFIRMED":
            trigger = "CONFIRMED"
        else:
            trigger = "UNCONFIRMED"
            m5_confirmation = None
    if htf_conflict_ctx and h4_bias_ctx in ("BULLISH", "BEARISH") and h1_bias_ctx in ("BULLISH", "BEARISH"):
        if m30_dir_ctx == h1_bias_ctx or h1_bias_ctx in m30_dir_ctx:
            canonical_conflict_reason = (
                f"HTF_CONFLICT (H4 {h4_bias_ctx} vs H1/M30 {h1_bias_ctx}) — M5 {trigger}; diblokir filter risiko"
                if _m5_was_evaluated_conflict
                else f"HTF_CONFLICT (H4 {h4_bias_ctx} vs H1/M30 {h1_bias_ctx}) — menunggu harga mendekati zona eligible & alignment"
            )
        else:
            canonical_conflict_reason = (
                f"HTF_CONFLICT (H4 {h4_bias_ctx} vs H1 {h1_bias_ctx}) — M5 {trigger}; diblokir filter risiko"
                if _m5_was_evaluated_conflict
                else f"HTF_CONFLICT (H4 {h4_bias_ctx} vs H1 {h1_bias_ctx}) — menunggu harga mendekati zona eligible & alignment"
            )
    elif m30_conflict_ctx and h4_bias_ctx in ("BULLISH", "BEARISH"):
        canonical_conflict_reason = (
            f"M30_CONFLICT ({m30_dir_ctx or 'CONFLICT'} vs H4 {h4_bias_ctx}) — M5 {trigger}; diblokir filter risiko"
            if _m5_was_evaluated_conflict
            else f"M30_CONFLICT ({m30_dir_ctx or 'CONFLICT'} vs H4 {h4_bias_ctx}) — menunggu harga mendekati zona eligible & alignment"
        )
    else:
        canonical_conflict_reason = ""
    if canonical_conflict_reason and not is_valid_confirmed and not snr_confirmed:
        if not (_m5_was_evaluated_conflict and specific_dec_reason):
            specific_dec_reason = canonical_conflict_reason
        if not _m5_was_evaluated_conflict and trigger != "DATA_UNAVAILABLE":
            trigger = "NOT_CHECKED"
            m5_confirmation = None
    elif (not is_valid_confirmed and not snr_confirmed) and (not zone or not m5_bound) and not _m5_was_evaluated_conflict:
        if trigger != "DATA_UNAVAILABLE":
            trigger = "NOT_CHECKED"
        m5_confirmation = None
    _htf_m30_blocker = bool(
        htf_conflict_ctx
        or m30_conflict_ctx
        or (
            specific_dec_reason
            and any(
                kw in specific_dec_reason.lower()
                for kw in ("konflik", "conflict", "m30", "netral", "choppy", "jarak sl", "melebihi batas", "too far", "far reference", "stale", "invalid", "berita", "spread")
            )
        )
    )

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
    elif _htf_m30_blocker:
        reason, next_action, final_signal = specific_dec_reason, ("WAIT ALIGNMENT" if (htf_conflict_ctx or m30_conflict_ctx) else "WAIT CONDITION"), "WAIT"
    elif touch_status in ("IN_ZONE", "TOUCHED_PREVIOUSLY") or monitor_state == "AWAITING_CONFIRMATION":
        reason = specific_dec_reason or "ZONE TOUCHED; M5 TRIGGER PENDING"
        next_action, final_signal = "CHECK VALIDATION", "WAIT"
    elif monitor_state == "APPROACHING":
        reason, next_action, final_signal = "APPROACHING ENTRY ZONE", "WAIT ZONE TOUCH", "WAIT"
    elif not zone:
        if specific_dec_reason:
            reason = specific_dec_reason
        elif isinstance(far_reference_zone, dict):
            f_tf_r = str(far_reference_zone.get("timeframe") or "").strip()
            f_zt_r = str(far_reference_zone.get("zone_type") or far_reference_zone.get("type") or "ZONE").upper()
            f_lo_r = safe_float(far_reference_zone.get("price_low", far_reference_zone.get("low")), 0.0)
            f_hi_r = safe_float(far_reference_zone.get("price_high", far_reference_zone.get("high")), 0.0)
            reason = f"WAIT (FAR REFERENCE ONLY: {f_tf_r} {f_zt_r} {f_lo_r:.2f}-{f_hi_r:.2f} di luar batas intraday)".strip()
        else:
            reason = str(scan.get("empty_reason") or snr_decision.get("reason") or "NO ACTIVE INTRADAY SETUP")
        next_action, final_signal = "WAIT SETUP", "WAIT"
    else:
        reason = specific_dec_reason or "WAITING FOR ZONE PULLBACK"
        next_action, final_signal = "WAIT APPROACH", "WAIT"
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
            _raw_sm = str(_it_tp.get("setup_match") or _it_tp.get("match_status") or "INFERRED").upper()
            if _raw_sm == "VERIFIED":
                _raw_sm = "CONFIRMED"
            elif _raw_sm == "UNVERIFIED":
                _raw_sm = "UNKNOWN"
            tracked_position = {
                "method": _m_tp,
                "ticket": _it_tp.get("ticket"),
                "position_id": _it_tp.get("position_id") or _it_tp.get("ticket"),
                "direction": _it_tp.get("direction") or direction,
                "open_price": safe_float(_it_tp.get("position_open_price"), 0.0),
                "zone_id": _it_tp.get("zone_id") or "",
                "zone_low": safe_float(_it_tp.get("low"), 0.0),
                "zone_high": safe_float(_it_tp.get("high"), 0.0),
                "mt5_status": str(_it_tp.get("mt5_status") or "OPEN").upper(),
                "setup_match": _raw_sm,
                "match_status": _raw_sm,
                "is_existing_position": bool(_it_tp.get("is_existing_position")),
                "position_opened_ts": safe_float(_it_tp.get("position_opened_ts"), 0.0),
                "position_opened_wib_short": str(_it_tp.get("position_opened_wib_short") or ""),
                "position_opened_wib": str(_it_tp.get("position_opened_wib") or ""),
                "match_reason": str(_it_tp.get("match_reason") or ""),
                "historical_tracking": bool(_it_tp.get("historical_tracking") or is_mkt_closed_ctx),
                "plan_valid": bool(_it_tp.get("plan_valid")),
                "uncertain": bool(_it_tp.get("uncertain")),
            }
            break
    if tracked_position is None:
        for _m_tp, _it_tp in focus_detail_all.items():
            _ex_p = _it_tp.get("existing_position") if isinstance(_it_tp, dict) else None
            if isinstance(_ex_p, dict) and _ex_p.get("ticket"):
                tracked_position = {
                    "method": _m_tp,
                    "ticket": _ex_p.get("ticket"),
                    "position_id": _ex_p.get("position_id") or _ex_p.get("ticket"),
                    "direction": _ex_p.get("direction") or _it_tp.get("direction") or direction,
                    "open_price": safe_float(_ex_p.get("open_price"), 0.0),
                    "zone_id": _it_tp.get("zone_id") or "",
                    "zone_low": safe_float(_it_tp.get("low"), 0.0),
                    "zone_high": safe_float(_it_tp.get("high"), 0.0),
                    "mt5_status": str(_ex_p.get("mt5_status") or "OPEN").upper(),
                    "setup_match": str(_ex_p.get("setup_match") or "UNKNOWN").upper(),
                    "match_status": str(_ex_p.get("setup_match") or "UNKNOWN").upper(),
                    "is_existing_position": True,
                    "position_opened_ts": safe_float(_ex_p.get("position_opened_ts"), 0.0),
                    "position_opened_wib_short": str(_ex_p.get("position_opened_wib_short") or ""),
                    "position_opened_wib": str(_ex_p.get("position_opened_wib") or ""),
                    "match_reason": str(_ex_p.get("match_reason") or ""),
                    "historical_tracking": bool(is_mkt_closed_ctx),
                    "plan_valid": False,
                    "uncertain": False,
                }
                break

    entry_signal_new = "WAIT" if (_htf_m30_blocker or trigger == "UNCONFIRMED") and final_signal in ("BUY", "SELL") else final_signal
    entry_reason_new = reason
    tracked_pos_verified = bool(
        tracked_position
        and tracked_position.get("setup_match") == "CONFIRMED"
        and not tracked_position.get("is_existing_position")
    )
    # Posisi INFERRED hanya tampil pada TRACKED POS dan tidak boleh mengubah keputusan entry baru maupun status aktif.
    if "RUNNING" in focus_status.values() and tracked_pos_verified and not is_valid_confirmed and not snr_confirmed:
        tp_hist = bool(tracked_position and tracked_position.get("historical_tracking")) or is_mkt_closed_ctx
        tp_tkt = tracked_position.get("ticket") if tracked_position else None
        tp_mst = tracked_position.get("match_status", "VERIFIED") if tracked_position else "VERIFIED"
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
            if _htf_m30_blocker and specific_dec_reason:
                reason = specific_dec_reason
                next_action = "WAIT ALIGNMENT" if (htf_conflict_ctx or m30_conflict_ctx) else "WAIT CONDITION"
            elif active_method == "SMC" and m5_bound and trigger == "CONFIRMED" and dec_reasons_touch:
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
            # Zona aktif sedang menunggu pullback (RADAR); jangan menimpa blocker spesifik HTF/M30/risk
            monitor_state = "RADAR"
            touch_status = "NOT_TOUCHED"
            z_desc = f"{zone.get('timeframe', '')} {zone.get('zone_type', '')} {safe_float(zone.get('price_low', zone.get('low'))):.2f}-{safe_float(zone.get('price_high', zone.get('high'))):.2f}".strip()
            touch_display = f"UNTOUCHED ({format_price_distance(distance, spec=extract_unit_spec_from_container(snapshot), include_mt5_points=True)})" if distance is not None else "UNTOUCHED"
            if not _htf_m30_blocker:
                reason = f"RADAR: Menunggu pullback ke {z_desc}"
                next_action = "WAIT PULLBACK"
            final_signal = "WAIT"
        elif radar_candidate and not data_invalid and _is_actionable_monitor_candidate(radar_candidate):
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
                if not _htf_m30_blocker:
                    reason = f"ZONE TOUCHED: Harga berada di {r_desc}"
                    next_action = "PANTAU KONFIRMASI M5"
            elif price > 0 and r_lo > 0 and r_hi >= r_lo and r_dist <= float(ZONE_APPROACH_DISTANCE_POINTS):
                monitor_state = "APPROACHING"
                touch_status = "NOT_TOUCHED"
                _r_d_txt = format_price_distance(r_dist, spec=extract_unit_spec_from_container(snapshot), include_mt5_points=True)
                touch_display = f"APPROACHING ({_r_d_txt})"
                if not _htf_m30_blocker:
                    reason = f"APPROACHING: Harga mendekati {r_desc} ({_r_d_txt})"
                    next_action = "WAIT ZONE TOUCH"
            else:
                monitor_state = "RADAR"
                touch_status = "NOT_TOUCHED"
                touch_display = f"RADAR ({format_price_distance(r_dist, spec=extract_unit_spec_from_container(snapshot), include_mt5_points=True)})" if r_dist > 0 else "UNTOUCHED"
                if not _htf_m30_blocker:
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

    if not tracked_pos_verified:
        entry_signal_new = final_signal
        entry_reason_new = reason

    eff_mkt_an = str(snapshot.get("market_status") or decision.get("market_status") or "OPEN").upper()
    if str(snapshot.get("snapshot_status") or "").upper() == "MARKET_CLOSED":
        eff_mkt_an = "CLOSED"
    analysis_direction = resolve_analysis_direction(
        ctx=scan.get("htf_context") or decision.get("context"),
        decision=decision,
        scan=scan,
        snr_bias=snr_mb,
        market_status=eff_mkt_an,
        data_ok=(not data_invalid and eff_mkt_an != "CLOSED" and price > 0),
    )

    has_live_entry_zone = bool(zone and safe_float(zone.get("price_low", zone.get("low")), 0.0) > 0)
    entry_blocker_code = resolve_entry_blocker_code(
        entry_reason_new,
        market_status=eff_mkt_an,
        data_ok=(not data_invalid),
        htf_conflict=htf_conflict_ctx,
        m30_conflict=m30_conflict_ctx,
        h4_bias=h4_bias_ctx,
        has_active_zone=has_live_entry_zone,
        zone_touch=touch_status,
        monitor_state=monitor_state if not tracked_pos_verified else None,
        rr_failed=(str(((decision.get("stages") or {}).get("risk_reward") or {}).get("status") or "").upper() == "FAIL"),
        is_missed=bool(decision.get("is_missed") or (decision.get("m5") or {}).get("is_missed")),
        is_stale=bool(is_entry_stale),
    )

    latest_invalid_display = "NONE"
    latest_invalid_zone = None
    inv_snr_display_pool = [
        {**z_inv, "source": "SNR"} for z_inv in (
            list(snr_zones.get("broken_zones") or [])
            + list(snr_zones.get("reclaimed_zones") or [])
            + list(snr_zones.get("invalidated_zones") or [])
        )
        if isinstance(z_inv, dict)
    ]
    inv_candidates = list(inv_smc_pool) + list(inv_snr_display_pool)
    trk_ctx = decision.get("zone_tracker")
    if trk_ctx is not None and isinstance(getattr(trk_ctx, "invalidated_zones", None), dict):
        inv_candidates.extend(list(trk_ctx.invalidated_zones.values()))
    eff_atr_inv = (
        safe_float(((decision.get("atr_diagnostics") or scan.get("atr_diagnostics") or {}).get("atr_value")), 0.0)
        or safe_float((((snr_scan.get("atr_diagnostics") or {}).get("H1") or {}).get("atr14")), 0.0)
        or None
    )
    iz_best = select_relevant_invalid_zone(
        inv_candidates,
        current_price=price if price > 0 else None,
        atr=eff_atr_inv,
        direction=analysis_direction,
    )
    if isinstance(iz_best, dict):
        latest_invalid_zone = iz_best
        if _is_snr_zone_obj(iz_best):
            iz_bo_cls = classify_snr_breakout_state(iz_best)
            iz_kind = str(iz_bo_cls.get("state") or iz_best.get("lifecycle_event") or iz_best.get("lifecycle_status") or "BROKEN").upper()
            if iz_kind == "NONE":
                iz_kind = str(iz_best.get("lifecycle_event") or iz_best.get("lifecycle_status") or "INVALIDATED").upper()
        else:
            iz_kind = str(
                iz_best.get("invalidation_type")
                or iz_best.get("invalid_label")
                or iz_best.get("lifecycle_status")
                or INVALIDATION_ZONE_STRUCTURAL
            ).upper()
        latest_invalid_display = format_invalid_zone_compact(
            iz_best,
            current_price=price if price > 0 else None,
            unit_spec=extract_unit_spec_from_container(snapshot),
            status_override=iz_kind,
        )

    tp_pool_mon = []
    smc_tp_list = decision.get("trade_plans") if isinstance(decision.get("trade_plans"), list) else []
    snr_tp_list = snr_scan.get("trade_plans") if isinstance(snr_scan.get("trade_plans"), list) else []
    if smc_tp_list:
        tp_pool_mon.extend([p for p in smc_tp_list if isinstance(p, dict)])
    elif isinstance(decision.get("best_trade_plan"), dict):
        tp_pool_mon.append(decision["best_trade_plan"])
    if snr_tp_list:
        tp_pool_mon.extend([p for p in snr_tp_list if isinstance(p, dict)])
    elif isinstance(snr_scan.get("best_trade_plan"), dict):
        tp_pool_mon.append(snr_scan["best_trade_plan"])
    _mon_closed = (
        str(snapshot.get("market_status") or decision.get("market_status") or "").upper() in ("CLOSED", "MARKET_CLOSED")
        or str(snapshot.get("snapshot_status") or "").upper() == "MARKET_CLOSED"
    )
    _reval_ctx_mon = {
        "snr_zones": snr_zones,
        "snr_price_action": (snr_scan.get("price_action") if isinstance(snr_scan.get("price_action"), dict) else {}),
        "scan_zones": list(scan.get("active_zones") or []) + list(scan.get("radar_zones") or []),
    }
    best_trade_plan_mon = resolve_trade_plan_focus(
        tp_pool_mon,
        direction=analysis_direction,
        scope="UNIFIED",
        live_price=price,
        bid=safe_float(snapshot.get("bid"), 0.0) or None,
        ask=safe_float(snapshot.get("ask"), 0.0) or None,
        is_closed=_mon_closed,
        mt5_orders_snapshot=mt5_ord_snap,
        htf_conflict=htf_conflict_ctx,
        m5_eval_status=trigger,
        revalidation_ctx=_reval_ctx_mon,
        h4_bias=h4_bias_ctx,
    )
    if isinstance(best_trade_plan_mon, dict):
        _tp_m_st = str(best_trade_plan_mon.get("status") or "").strip().upper()
        _tp_m_src = str(best_trade_plan_mon.get("source") or "").strip().upper()
        _tp_m_zid = str(best_trade_plan_mon.get("zone_id") or "").strip()
        if _tp_m_zid and _tp_m_st not in ("", "NO_ACTIVE_PLAN", "REJECTED", "INVALID", "SL_BREACHED", "EXPIRED", "MISSED"):
            if _tp_m_src == "SNR" and str(zone.get("zone_id") or "").strip() != _tp_m_zid:
                _snr_m_cand = next(
                    (
                        sz_m for sz_m in (active_sups + active_ress + list(snr_zones.get("all_zones") or []))
                        if isinstance(sz_m, dict) and str(sz_m.get("zone_id") or "").strip() == _tp_m_zid
                    ),
                    None,
                )
                if not isinstance(_snr_m_cand, dict):
                    _snr_m_cand = {
                        "zone_id": _tp_m_zid,
                        "timeframe": best_trade_plan_mon.get("timeframe") or "H1",
                        "zone_type": best_trade_plan_mon.get("zone_type") or ("RESISTANCE" if best_trade_plan_mon.get("direction") == "SELL" else "SUPPORT"),
                        "low": safe_float(best_trade_plan_mon.get("zone_low"), 0.0),
                        "high": safe_float(best_trade_plan_mon.get("zone_high"), 0.0),
                        "grade": best_trade_plan_mon.get("grade") or "N/A",
                        "score": best_trade_plan_mon.get("score"),
                        "structure_invalidation": safe_float(best_trade_plan_mon.get("struct_inv"), 0.0),
                        "setup_sl": safe_float(best_trade_plan_mon.get("setup_sl"), 0.0),
                        "setup_version": best_trade_plan_mon.get("setup_version"),
                        "lifecycle_status": best_trade_plan_mon.get("zone_status") or "ACTIVE",
                    }
                _s_lo_m = safe_float(_snr_m_cand.get("low", _snr_m_cand.get("price_low")), 0.0)
                _s_hi_m = safe_float(_snr_m_cand.get("high", _snr_m_cand.get("price_high")), 0.0)
                if _s_lo_m > 0 and _s_hi_m >= _s_lo_m:
                    zone = dict(_snr_m_cand)
                    zone.update({
                        "zone_id": _tp_m_zid,
                        "timeframe": _snr_m_cand.get("timeframe") or "H1",
                        "zone_type": str(_snr_m_cand.get("zone_type") or ("RESISTANCE" if best_trade_plan_mon.get("direction") == "SELL" else "SUPPORT")).upper(),
                        "price_low": _s_lo_m,
                        "price_high": _s_hi_m,
                        "grade": _snr_m_cand.get("grade") or "N/A",
                        "score": safe_float(_snr_m_cand.get("score")) if _snr_m_cand.get("score") not in (None, "") else None,
                        "status": str(_snr_m_cand.get("lifecycle_status") or _snr_m_cand.get("status") or "ACTIVE").upper(),
                        "source": "SNR",
                    })
                    lo, hi = _s_lo_m, _s_hi_m
                    distance = 0.0 if (price > 0 and lo <= price <= hi) else (min(abs(price - lo), abs(price - hi)) if price > 0 else None)
                    zone_type = zone["zone_type"]
                    z_id = _tp_m_zid
                    if far_reference_zone is None and isinstance(decision.get("far_reference_zone"), dict):
                        far_reference_zone = dict(decision["far_reference_zone"])

    return {
        "snapshot": snapshot,
        "analysis_direction": analysis_direction,
        "best_trade_plan": best_trade_plan_mon,
        "trade_plans": tp_pool_mon,
        "entry_signal": entry_signal_new,
        "entry_blocker_code": entry_blocker_code,
        "latest_invalid_display": latest_invalid_display,
        "latest_invalid_zone": latest_invalid_zone,
        "snr_h4_bias": snr_h4_bias,
        "snr_h1_bias": snr_h1_bias,
        "snr_m30_bias": snr_m30_bias,
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
        "snr_nearest_support": active_sups[0] if active_sups else None,
        "snr_nearest_resistance": active_ress[0] if active_ress else None,
        "h4_bias": h4_bias_ctx,
        "h1_bias": h1_bias_ctx,
        "m30_bias": m30_dir_ctx,
        "htf_conflict": htf_conflict_ctx,
        "m30_conflict": m30_conflict_ctx,
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
        hdr_line = (left_hdr + short_right)[:term_w]
        t_pos = hdr_line.find(" | ")
        time_span = (t_pos, len(hdr_line)) if t_pos >= 0 else None
    sym_span = (1, 1 + len(sym_txt))
    st_start = 1 + len(sym_txt) + 2
    status_span = (st_start, min(len(hdr_line), st_start + len(status_badge)))

    # Susun teks perubahan harian (berdasarkan D1 Close sebelumnya dari MT5)
    if feed_st in ("DISCONNECTED", "STALE"):
        price_str = "---"
        chg_str = "MT5 DISCONNECTED - NO LIVE PRICE" if feed_st == "DISCONNECTED" else "STALE FEED - NO LIVE TICK"
        quote_str = "Bid: --- | Ask: --- | Spr: ---"
        meta_str = "Waiting for live MT5 connection" if feed_st == "DISCONNECTED" else "Waiting for fresh MT5 live tick"
    else:
        price_str = f"{live_p:.{digits}f}" if live_p > 0 else "---"
        if d_chg is not None and d_pct is not None and prev_d1 is not None:
            ar = up_ar if d_chg > 1e-9 else (dn_ar if d_chg < -1e-9 else fl_ar)
            chg_prefix = "LAST CLOSE | " if feed_st == "CLOSED" else ""
            chg_str = f"{chg_prefix}{ar} {d_chg:+.{digits}f} ({d_pct:+.2f}%) D1:{prev_d1:.{digits}f}"
        elif feed_st == "CLOSED":
            chg_str = "LAST CLOSE | D1: N/A"
        else:
            chg_str = "Daily Chg: N/A"

        if bid_v > 0 and ask_v > 0:
            _spr_val = spr_v if spr_v is not None else max(0.0, ask_v - bid_v)
            spr_txt = format_price_distance(_spr_val, spec=tk.get("unit_spec"), digits=digits, include_mt5_points=False)
            quote_str = f"Bid:{bid_v:.{digits}f} Ask:{ask_v:.{digits}f} Spr:{spr_txt}"
        else:
            quote_str = "Bid: --- | Ask: --- | Spr: ---"

        if feed_st == "CLOSED":
            meta_str = "MARKET CLOSED (Last Close)"
        else:
            td = tk["tick_delta"]
            td_ar = up_ar if tk["price_direction"] == "UP" else (dn_ar if tk["price_direction"] == "DOWN" else fl_ar)
            td_txt = f"Tick {td_ar} {td:+.{digits}f}" if abs(td) > 1e-9 else f"Tick {td_ar} LIVE"
            meta_str = f"MT5 LIVE | {td_txt}"

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
    side_texts = [(chg_str, "change"), (quote_str, "quote"), (meta_str, "meta")]
    max_side_len = max((len(s_t) for s_t, _ in side_texts), default=0)
    use_big_widget = bool(use_uni and big_rows and (1 + big_w + 3 + max_side_len) <= term_w)
    if use_big_widget:
        split_col = 1 + big_w + 3
        for r_i in range(3):
            b_row = big_rows[r_i].ljust(big_w)
            s_txt, s_role = side_texts[r_i]
            line_str = f" {b_row}   {s_txt}"
            line_idx = len(widget_lines)
            widget_lines.append(line_str)
            widget_meta["price_rows"].append((line_idx, split_col, s_role))
    else:
        c_prefix = " PRICE : "
        c_p = price_str
        c_gap = "  "
        c_line = f"{c_prefix}{c_p}{c_gap}{chg_str}"
        if len(c_line) <= term_w:
            c_idx = len(widget_lines)
            widget_lines.append(c_line)
            widget_meta["compact_price_idx"] = c_idx
            widget_meta["compact_price_span"] = (len(c_prefix), len(c_prefix) + len(c_p))
            widget_meta["compact_change_span"] = (len(c_prefix) + len(c_p) + len(c_gap), len(c_line))
        else:
            c_idx = len(widget_lines)
            widget_lines.append(f"{c_prefix}{c_p}")
            widget_meta["compact_price_idx"] = c_idx
            widget_meta["compact_price_span"] = (len(c_prefix), len(c_prefix) + len(c_p))
            widget_lines.extend(wrap_dashboard_field(" CHANGE: ", chg_str, term_w, max_lines=None))
        widget_lines.extend(wrap_dashboard_field(" QUOTE : ", quote_str, term_w, max_lines=None))

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


def _is_snr_zone_obj(z_obj):
    if not isinstance(z_obj, dict):
        return False
    src = str(z_obj.get("source") or "").upper()
    zt = str(z_obj.get("zone_type") or z_obj.get("type") or "").upper()
    return (
        src == "SNR"
        or zt in ("SUPPORT", "RESISTANCE")
        or "swing_class" in z_obj
        or "touch_analysis" in z_obj
        or "breakout_info" in z_obj
    )


def _build_monitor_zone_why(z_obj, smc_h4=None, snr_h4=None, smc_h1=None, snr_h1=None):
    if not isinstance(z_obj, dict):
        return ""
    if _is_snr_zone_obj(z_obj):
        eff_h4 = snr_h4 if str(snr_h4 or "").upper() in ("BULLISH", "BEARISH") else smc_h4
        eff_h1 = snr_h1 if str(snr_h1 or "").upper() in ("BULLISH", "BEARISH") else smc_h1
        return build_snr_zone_reason(z_obj, h4_bias=eff_h4, h1_bias=eff_h1)
    return build_smc_zone_reason(z_obj, h4_bias=smc_h4, h1_bias=smc_h1)


def _build_monitor_invalid_why(z_obj, smc_h4=None, snr_h4=None, smc_h1=None, snr_h1=None):
    if not isinstance(z_obj, dict):
        return ""
    if _is_snr_zone_obj(z_obj):
        eff_h4 = snr_h4 if str(snr_h4 or "").upper() in ("BULLISH", "BEARISH") else smc_h4
        eff_h1 = snr_h1 if str(snr_h1 or "").upper() in ("BULLISH", "BEARISH") else smc_h1
        return format_snr_invalid_why(z_obj, h4_bias=eff_h4, h1_bias=eff_h1)
    return format_smc_invalid_why(z_obj, h4_bias=smc_h4, h1_bias=smc_h1)


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
    has_live_focus = bool(
        feed_st != "CLOSED"
        and lo > 0
        and hi > 0
        and not (
            tracked_pos
            and tracked_pos.get("setup_match") == "CONFIRMED"
            and not tracked_pos.get("is_existing_position")
            and tracked_pos.get("zone_id")
            and tracked_pos.get("zone_id") == zone.get("zone_id")
        )
    )
    dir_dash = str(data.get("analysis_direction") or data.get("direction") or "N/A").upper()
    if feed_st == "CLOSED" or data.get("data_invalid"):
        dir_dash = "N/A"

    # ACTIVE/RADAR zone (pisahkan StructInv dari SetupSL)
    def _fmt_inv_tags(z_item, lo_v, hi_v, zt_u):
        from modules.utils import resolve_stored_setup_sl
        s_inv = resolve_smc_structure_invalidation_price(z_item)
        if s_inv <= 0 and lo_v > 0:
            s_inv = lo_v if zt_u in ("DEMAND", "SUPPORT") else (hi_v if zt_u in ("SUPPLY", "RESISTANCE") else 0.0)
        is_snr_z = _is_snr_zone_obj(z_item)
        z_src_l = "SNR" if is_snr_z else "SMC"
        sl_v = resolve_stored_setup_sl(
            z_src_l,
            str((z_item or {}).get("zone_id") or "") if isinstance(z_item, dict) else "",
            trade_plans=data.get("trade_plans") if isinstance(data.get("trade_plans"), list) else None,
            best_trade_plan=data.get("best_trade_plan") if isinstance(data.get("best_trade_plan"), dict) else None,
            timeframe=str((z_item or {}).get("timeframe") or "").strip().upper() if isinstance(z_item, dict) else None,
            setup_version=(z_item.get("setup_version") if (isinstance(z_item, dict) and isinstance(z_item.get("setup_version"), int)) else None),
        )
        if sl_v <= 0 and not is_snr_z:
            sl_v = resolve_smc_setup_invalidation_price(z_item)
        segs = []
        if s_inv > 0:
            segs.append(f"StructInv {s_inv:.2f}")
        if sl_v > 0 and abs(sl_v - s_inv) >= 0.01:
            segs.append(f"SetupSL {sl_v:.2f}")
        return (" | " + " | ".join(segs)) if segs else ""

    active_radar_dash = "NONE"
    shown_active_z_dash = None
    if feed_st != "CLOSED" and has_live_focus:
        shown_active_z_dash = zone
        z_id_d = str(zone.get("zone_id") or "").strip()
        z_id_tag = f" [ID:{z_id_d}]" if z_id_d and z_id_d != "N/A" else ""
        inv_tag_d = _fmt_inv_tags(zone, lo, hi, z_type)
        dist_v = data.get("distance")
        if dist_v is None or feed_st in ("DISCONNECTED", "STALE"):
            d_short = "N/A"
        elif dist_v <= 0.0:
            d_short = "IN ZONE"
        else:
            d_short = format_price_distance(dist_v, spec=u_spec_dash)
        z_rel_raw = str(zone.get("zone_relevance") or zone.get("lifecycle_status") or zone.get("status") or "ACTIVE").upper()
        mon_st_d = str(data.get("monitor_state") or "").upper()
        if str(zone.get("radar_eligibility") or "").upper() == "NOT_QUALIFIED" or str(zone.get("risk_tier") or "").upper() == "FAR_ZONE":
            excl_rsn_d = str(zone.get("risk_tier_reason") or zone.get("pending_rejection_reason") or "EXCLUDED_FROM_RADAR").strip()
            z_rel_d = f"EXCLUDED ({excl_rsn_d})"
        elif mon_st_d in ("ZONE_TOUCHED", "AWAITING_CONFIRMATION") or z_rel_raw == "TOUCHED":
            z_rel_d = "ACTIVE (TOUCHED)"
        elif z_rel_raw in ("ACTIVE", "RADAR"):
            z_rel_d = z_rel_raw
        else:
            z_rel_d = f"ACTIVE/{z_rel_raw}"
        active_radar_dash = f"{z_tf} {z_type} {lo:.2f}-{hi:.2f} | ZoneGrade:{z_grd}/{z_sc_txt} | Dist {d_short}{inv_tag_d} | {z_rel_d} [{z_src}]{z_id_tag}".replace("  ", " ")
    elif isinstance(radar_cand_d, dict) and safe_float(radar_cand_d.get("price_low", radar_cand_d.get("low")), 0.0) > 0:
        shown_active_z_dash = radar_cand_d
        rc_lo = safe_float(radar_cand_d.get("price_low", radar_cand_d.get("low")), 0.0)
        rc_hi = safe_float(radar_cand_d.get("price_high", radar_cand_d.get("high")), 0.0)
        rc_tf = str(radar_cand_d.get("timeframe") or "").strip()
        rc_zt = str(radar_cand_d.get("zone_type") or "ZONE").upper()
        rc_grd = str(radar_cand_d.get("grade") or "-")
        rc_sc = f"/{safe_float(radar_cand_d.get('score')):.1f}" if radar_cand_d.get("score") is not None else ""
        rc_dist = safe_float(radar_cand_d.get("distance"), max(rc_lo - price_val, price_val - rc_hi, 0.0) if price_val > 0 else 0.0)
        rc_inv_s = _fmt_inv_tags(radar_cand_d, rc_lo, rc_hi, rc_zt)
        rc_id = f" [ID:{radar_cand_d['zone_id']}]" if radar_cand_d.get("zone_id") else ""
        active_radar_dash = f"RADAR {rc_tf} {rc_zt} {rc_lo:.2f}-{rc_hi:.2f} | G:{rc_grd}{rc_sc} | Dist {format_price_distance(rc_dist, spec=u_spec_dash)}{rc_inv_s}{rc_id}".replace("  ", " ")
    elif isinstance(far_ref, dict) and safe_float(far_ref.get("price_low", far_ref.get("low")), 0.0) > 0:
        shown_active_z_dash = far_ref
        f_lo = safe_float(far_ref.get("price_low", far_ref.get("low")), 0.0)
        f_hi = safe_float(far_ref.get("price_high", far_ref.get("high")), 0.0)
        f_tf = str(far_ref.get("timeframe") or "").strip()
        f_zt = str(far_ref.get("rank_label") or far_ref.get("zone_type") or "ZONE").upper()
        f_grd = str(far_ref.get("grade") or "-")
        f_sc = f"/{safe_float(far_ref.get('score')):.1f}" if far_ref.get("score") is not None else ""
        f_dist = max(f_lo - price_val, price_val - f_hi, 0.0) if price_val > 0 else safe_float(far_ref.get("distance", far_ref.get("distance_points")), 0.0)
        f_zt_raw = str(far_ref.get("zone_type") or "").upper()
        f_inv_s = _fmt_inv_tags(far_ref, f_lo, f_hi, f_zt_raw)
        f_id = f" [ID:{far_ref['zone_id']}]" if far_ref.get("zone_id") else ""
        active_radar_dash = f"NONE | FAR REF {f_tf} {f_zt} {f_lo:.2f}-{f_hi:.2f} | G:{f_grd}{f_sc} | Dist {format_price_distance(f_dist, spec=u_spec_dash)}{f_inv_s}{f_id}".replace("  ", " ")

    snr_h4_d = data.get("snr_h4_bias") or h4
    snr_h1_d_pre = str(data.get("snr_h1_bias") or "").upper()
    active_why_dash = _build_monitor_zone_why(shown_active_z_dash, smc_h4=h4, snr_h4=snr_h4_d, smc_h1=h1, snr_h1=snr_h1_d_pre)

    # 1-line SNR summary
    snr_sup_near = data.get("snr_nearest_support")
    snr_res_near = data.get("snr_nearest_resistance")
    snr_dash_parts = []
    s1_why_dash = ""
    r1_why_dash = ""
    if isinstance(snr_sup_near, dict) and safe_float(snr_sup_near.get("low")) > 0:
        snr_dash_parts.append(f"S1 {safe_float(snr_sup_near.get('low')):.2f}-{safe_float(snr_sup_near.get('high')):.2f}(G:{snr_sup_near.get('grade', '-')})")
        s1_why_dash = build_snr_zone_reason(snr_sup_near, h4_bias=snr_h4_d, h1_bias=snr_h1_d_pre)
    if isinstance(snr_res_near, dict) and safe_float(snr_res_near.get("low")) > 0:
        snr_dash_parts.append(f"R1 {safe_float(snr_res_near.get('low')):.2f}-{safe_float(snr_res_near.get('high')):.2f}(G:{snr_res_near.get('grade', '-')})")
        r1_why_dash = build_snr_zone_reason(snr_res_near, h4_bias=snr_h4_d, h1_bias=snr_h1_d_pre)
    if data.get("confluence") == "ALIGNED":
        snr_dash_parts.append("CONFLUENCE")
    snr_1line_dash = " | ".join(snr_dash_parts) if snr_dash_parts else "NONE"

    # 1-line TRACKED POSITION (if open)
    tracked_lines = []
    if tracked_pos:
        from modules.mt5_conn import format_tracked_position_compact
        tp_txt = format_tracked_position_compact(tracked_pos)
        if tp_txt:
            tracked_lines.extend(wrap_dashboard_field(" TRACKED POS : ", tp_txt, term_w, max_lines=None))

    # MT5 risk & order summary + sync timestamp
    ord_list = list(mt5_snap.get("orders") or acc.get("orders") or [])
    pos_cnt = int(safe_float(acc.get("open_positions"), mt5_snap.get("symbol_positions_count", 0)))
    ord_cnt = int(safe_float(acc.get("pending_orders"), len(ord_list)))
    p_nosl_c = int(mt5_risk.get("positions_without_sl_count", safe_float(acc.get("position_without_sl"), 0)))
    o_nosl_c = int(mt5_risk.get("pending_without_sl_count", safe_float(acc.get("pending_without_sl"), 0)))
    no_sl_tot = p_nosl_c + o_nosl_c
    flt_pnl = safe_float(acc.get("floating_pnl"), 0.0)
    cur_lbl = str(acc.get("currency") or "USD")
    sync_dash_t = format_mt5_sync_short(mt5_snap, fallback_ts=raw_ts)
    pos_txt = (
        f"MT5 DISCONNECTED | Sync:{sync_dash_t}"
        if feed_st == "DISCONNECTED"
        else f"Pos:{pos_cnt}({flt_pnl:+.2f} {cur_lbl}) Pend:{ord_cnt} NoSL:{no_sl_tot} | Sync:{sync_dash_t}"
    )

    signal_disp = str(data.get("signal_display") or data.get("signal") or "WAIT").upper()
    reason_disp = str(data.get("reason") or "WAIT").replace("\n", " ").strip()
    risk_head_d = str(data.get("critical_risk_display") or "").strip()
    if feed_st == "DISCONNECTED" and "DISCONNECTED" not in risk_head_d.upper():
        risk_line_dash = f"MT5 DISCONNECTED [HIGH RISK] | Sync:{sync_dash_t}"
    elif risk_head_d:
        risk_line_dash = f"{risk_head_d} | {pos_txt}"
    else:
        risk_line_dash = f"AMAN | {pos_txt}"
    news_sum = str(cycle.get("news_summary") or "").strip()

    # 1-line ENTRY STATUS (terpisah dari TRACKED POS; BUY/SELL VALID hanya jika confirmed; else NO VALID ENTRY + short code)
    entry_sig_dash = str(data.get("entry_signal") or signal_disp).upper()
    _cp = data.get("plan") if isinstance(data.get("plan"), dict) else {}
    _ps = data.get("pending_setup") if isinstance(data.get("pending_setup"), dict) else {}
    _cp_dir = str(_cp.get("arah") or entry_sig_dash).upper()
    _cp_e = safe_float(_cp.get("entry_acuan") or _cp.get("entry"), 0.0)
    _cp_s = safe_float(_cp.get("sl"), 0.0)
    _cp_t = safe_float(_cp.get("tp1"), 0.0)
    _cp_r = safe_float(_cp.get("rr"), 0.0)
    if feed_st != "CLOSED" and entry_sig_dash in ("BUY", "SELL") and _cp_e > 0 and _cp_s > 0 and _cp_t > 0:
        entry_status_dash = f"{entry_sig_dash} VALID | E {_cp_e:.2f} | SetupSL {_cp_s:.2f} | TP1 {_cp_t:.2f} | RR 1:{_cp_r:.1f}"
    elif feed_st != "CLOSED" and entry_sig_dash in ("BUY", "SELL"):
        entry_status_dash = f"{entry_sig_dash} VALID"
    else:
        blk_d = "MARKET_CLOSED" if feed_st == "CLOSED" else str(data.get("entry_blocker_code") or resolve_entry_blocker_code(reason_disp, market_status=feed_st))
        if blk_d == "HTF_CONFLICT" and m5 in ("CONFIRMED", "UNCONFIRMED", "DATA_UNAVAILABLE"):
            entry_status_dash = f"NO VALID ENTRY ({blk_d} | M5 {m5})"
        else:
            entry_status_dash = f"NO VALID ENTRY ({blk_d})"

    _pend_txt_d = ""
    if feed_st != "CLOSED" and str(_ps.get("plan_status") or "").upper() == "READY":
        _p_e = safe_float(_ps.get("entry"), 0.0)
        _p_s = safe_float(_ps.get("sl"), 0.0)
        _p_t = safe_float(_ps.get("tp1"), 0.0)
        _p_r = safe_float(_ps.get("rr"), 0.0)
        if _p_e > 0 and _p_s > 0 and _p_t > 0:
            _pend_txt_d = f"{_ps.get('order_type', 'PENDING')} {_p_e:.2f} | SetupSL {_p_s:.2f} | TP1 {_p_t:.2f} | RR 1:{_p_r:.1f} | {_ps.get('status', 'READY')}"

    invalid_dash = str(data.get("latest_invalid_display") or "NONE")
    invalid_why_dash = (
        _build_monitor_invalid_why(data.get("latest_invalid_zone"), smc_h4=h4, snr_h4=snr_h4_d, smc_h1=h1, snr_h1=snr_h1_d_pre)
        if invalid_dash != "NONE"
        else ""
    )
    smc_far_ref_dash = ""
    if (
        shown_active_z_dash is not None
        and _is_snr_zone_obj(shown_active_z_dash)
        and isinstance(far_ref, dict)
        and not _is_snr_zone_obj(far_ref)
        and safe_float(far_ref.get("price_low", far_ref.get("low")), 0.0) > 0
    ):
        f_lo_s = safe_float(far_ref.get("price_low", far_ref.get("low")), 0.0)
        f_hi_s = safe_float(far_ref.get("price_high", far_ref.get("high")), 0.0)
        f_tf_s = str(far_ref.get("timeframe") or "").strip()
        f_zt_s = str(far_ref.get("zone_type") or "ZONE").upper()
        f_grd_s = str(far_ref.get("grade") or "-")
        f_sc_s = f"/{safe_float(far_ref.get('score')):.1f}" if far_ref.get("score") is not None else ""
        f_fr_s = f" {str(far_ref.get('freshness') or 'FRESH').upper()}" if far_ref.get("freshness") else ""
        f_dist_s = max(f_lo_s - price_val, price_val - f_hi_s, 0.0) if price_val > 0 else safe_float(far_ref.get("distance", far_ref.get("distance_points")), 0.0)
        f_inv_s_d = _fmt_inv_tags(far_ref, f_lo_s, f_hi_s, f_zt_s)
        f_id_s = f" [ID:{far_ref['zone_id']}]" if far_ref.get("zone_id") else ""
        smc_far_ref_dash = f"{f_tf_s} {f_zt_s} {f_lo_s:.2f}-{f_hi_s:.2f} | G:{f_grd_s}{f_sc_s}{f_fr_s} | Dist {format_price_distance(f_dist_s, spec=u_spec_dash)}{f_inv_s_d} | FAR_REFERENCE{f_id_s}".replace("  ", " ")
    snr_h1_d = str(data.get("snr_h1_bias") or "").upper()
    h1_dir_dash = (
        f"H1(SMC)={h1} H1(SNR)={snr_h1_d}"
        if (snr_h1_d and snr_h1_d not in ("N/A", "UNKNOWN", "DATA UNAVAILABLE") and snr_h1_d != h1)
        else f"H1(SMC)={h1}"
    )
    snr_m30_d = str(data.get("snr_m30_bias") or "").upper()
    m30_dir_dash = (
        f"M30(SMC)={m30} M30(SNR)={snr_m30_d}"
        if (snr_m30_d and snr_m30_d not in ("N/A", "UNKNOWN", "UNCONFIRMED", "NO_DATA", "DATA UNAVAILABLE") and snr_m30_d != m30)
        else f"M30(SMC)={m30}"
    )
    best_tp_dash = data.get("best_trade_plan") if isinstance(data.get("best_trade_plan"), dict) else None
    if best_tp_dash:
        best_tp_dash = refresh_trade_plan_live_state(
            best_tp_dash,
            live_price=live_p if live_p > 0 else price_val,
            bid=tk.get("bid"),
            ask=tk.get("ask"),
            is_closed=(str(feed_st or "").upper() in ("CLOSED", "MARKET_CLOSED")),
            m5_eval_status=m5,
            mt5_orders_snapshot=mt5_snap,
            htf_conflict=bool(data.get("htf_conflict")),
            h4_bias=h4,
        )
    _pend_same_dash = bool(
        _pend_txt_d and best_tp_dash
        and str(best_tp_dash.get("zone_id") or "") == str(_ps.get("zone_id") or "")
        and str(best_tp_dash.get("status") or "").upper() == "READY"
    )
    _tp_dash_lines = format_trade_plan_lines(
        best_tp_dash,
        prefix=" TRADE PLAN  : ",
        why_prefix="   WHY PLAN  : ",
        width=term_w,
        include_source=True,
        broker_tag=(str(_ps.get("status") or "") if _pend_same_dash else None),
        skip_why_for_zone_id=(shown_active_z_dash.get("zone_id") if isinstance(shown_active_z_dash, dict) else None),
    ) if best_tp_dash else []
    body_lines = [
        dash_div,
        *wrap_dashboard_field(" MARKET BIAS : ", f"H4={h4} {h1_dir_dash} {m30_dir_dash} M5={m5}", term_w, max_lines=None),
        *wrap_dashboard_field(" ENTRY STATUS: ", entry_status_dash, term_w, max_lines=None),
        *_tp_dash_lines,
        *(wrap_dashboard_field(" PENDING PLAN: ", _pend_txt_d, term_w, max_lines=None) if (_pend_txt_d and not _pend_same_dash) else []),
        *wrap_dashboard_field(" ACTIVE/RADAR: ", active_radar_dash, term_w, max_lines=None),
        *(wrap_dashboard_field("   WHY       : ", active_why_dash, term_w, max_lines=None) if active_why_dash else []),
        *(wrap_dashboard_field(" SMC FAR REF : ", smc_far_ref_dash, term_w, max_lines=None) if smc_far_ref_dash else []),
        *wrap_dashboard_field(" INVALID     : ", invalid_dash, term_w, max_lines=None),
        *(wrap_dashboard_field("   WHY       : ", invalid_why_dash, term_w, max_lines=None) if invalid_why_dash else []),
        *wrap_dashboard_field(" SNR         : ", snr_1line_dash, term_w, max_lines=None),
        *(wrap_dashboard_field("   WHY S1    : ", s1_why_dash, term_w, max_lines=None) if s1_why_dash else []),
        *(wrap_dashboard_field("   WHY R1    : ", r1_why_dash, term_w, max_lines=None) if r1_why_dash else []),
        *wrap_dashboard_field(" RISK        : ", risk_line_dash, term_w, max_lines=None),
        *tracked_lines,
    ]
    if news_sum:
        compact_news = news_sum.replace("NEWS RISK : ", "").replace(" | NEXT EVENT : ", " | ")
        body_lines.extend(wrap_dashboard_field(" US NEWS     : ", compact_news, term_w, max_lines=None))

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
            body_lines.extend(wrap_dashboard_field("  * ", ev_ln, term_w, max_lines=None))
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
    """Format default `pantau` output: maksimal 1 baris per informasi, tanpa paragraf alasan berulang."""
    data = _smart_monitor_context(cycle_result)
    zone = data["zone"]
    summary = data["summary"]
    snapshot = data["snapshot"]
    acc = data.get("account") or {}
    mt5_snap = data.get("mt5_orders_snapshot") if isinstance(data.get("mt5_orders_snapshot"), dict) else {}
    mt5_risk = data.get("mt5_risk_summary") if isinstance(data.get("mt5_risk_summary"), dict) else {}
    _u_rep = extract_unit_spec_from_container(snapshot)

    is_mkt_closed_rep = (
        str(snapshot.get("market_status") or (data.get("decision") or {}).get("market_status") or "").upper() == "CLOSED"
        or str(snapshot.get("snapshot_status") or "").upper() == "MARKET_CLOSED"
    )
    mkt_lbl = "CLOSED" if is_mkt_closed_rep else ("DATA UNAVAILABLE" if data.get("data_invalid") else "OPEN")
    quote_fresh = (
        snapshot.get("data_ok") is not False and snapshot.get("live") is True
        and str(snapshot.get("data_freshness") or "").upper() not in ("STALE", "INVALID", "NO DATA")
    )
    price_txt = f"{data['price']:.2f}" if data["price"] > 0 and (not data["data_invalid"] or quote_fresh) else "N/A"
    bid_r, ask_r = safe_float(snapshot.get("bid"), 0.0), safe_float(snapshot.get("ask"), 0.0)
    if not is_mkt_closed_rep and bid_r > 0 and ask_r > 0:
        price_txt = f"{price_txt} (Bid {bid_r:.2f} / Ask {ask_r:.2f})"
    snap_ts_rep = str(snapshot.get("snapshot_time") or (cycle_result or {}).get("snapshot_time") or "N/A").strip() if isinstance(cycle_result, dict) else str(snapshot.get("snapshot_time") or "N/A").strip()
    snap_id_rep = str(snapshot.get("snapshot_id") or (cycle_result or {}).get("snapshot_id") or "").strip() if isinstance(cycle_result, dict) else str(snapshot.get("snapshot_id") or "").strip()
    id_sfx = f" [ID:{snap_id_rep}]" if snap_id_rep else ""

    h4 = str(summary.get("h4_bias") or "N/A").upper()
    h1 = str(summary.get("h1_bias") or "N/A").upper()
    m30 = str(summary.get("m30_bias") or "N/A").upper()
    m5 = str(data.get("trigger") or "WAIT").upper()
    dir_rep = str(data.get("analysis_direction") or "N/A").upper() if mkt_lbl == "OPEN" else "N/A"

    signal = str(data.get("entry_signal") or data.get("signal_display") or data.get("signal") or "WAIT").upper()
    main_reason = str(data.get("reason") or summary.get("main_reason") or "Menunggu konfirmasi").replace("\n", " ").strip()
    plan_dict = data.get("plan") if isinstance(data.get("plan"), dict) else {}
    p_ent = safe_float(plan_dict.get("entry_acuan") or plan_dict.get("entry"), 0.0)
    p_sl = safe_float(plan_dict.get("sl"), 0.0)
    p_tp1 = safe_float(plan_dict.get("tp1"), 0.0)
    p_rr = safe_float(plan_dict.get("rr"), 0.0)

    if mkt_lbl == "OPEN" and signal in ("BUY", "SELL") and p_ent > 0 and p_sl > 0 and p_tp1 > 0:
        entry_line = f"{signal} VALID | E {p_ent:.2f} | SetupSL {p_sl:.2f} | TP1 {p_tp1:.2f} | RR 1:{p_rr:.1f}"
    elif mkt_lbl == "OPEN" and signal in ("BUY", "SELL"):
        entry_line = f"{signal} VALID"
    else:
        blk = "MARKET_CLOSED" if is_mkt_closed_rep else str(data.get("entry_blocker_code") or resolve_entry_blocker_code(main_reason, market_status=mkt_lbl))
        if blk == "HTF_CONFLICT" and m5 in ("CONFIRMED", "UNCONFIRMED", "DATA_UNAVAILABLE"):
            entry_line = f"NO VALID ENTRY ({blk} | M5 {m5})"
        else:
            entry_line = f"NO VALID ENTRY ({blk})"

    pending_s = data.get("pending_setup") if isinstance(data.get("pending_setup"), dict) else {}
    pend_line = ""
    if mkt_lbl == "OPEN" and str(pending_s.get("plan_status") or "").upper() == "READY":
        pe, psl, ptp, prr = safe_float(pending_s.get("entry")), safe_float(pending_s.get("sl")), safe_float(pending_s.get("tp1")), safe_float(pending_s.get("rr"))
        if pe > 0 and psl > 0 and ptp > 0:
            pend_line = f"{pending_s.get('order_type', 'PENDING')} {pe:.2f} | SetupSL {psl:.2f} | TP1 {ptp:.2f} | RR 1:{prr:.1f} | {pending_s.get('status', 'READY')}"

    def _fmt_rep_inv_tags(z_item, lo_v, hi_v, zt_u):
        from modules.utils import resolve_stored_setup_sl
        s_inv = resolve_smc_structure_invalidation_price(z_item)
        if s_inv <= 0 and lo_v > 0:
            s_inv = lo_v if zt_u in ("DEMAND", "SUPPORT") else (hi_v if zt_u in ("SUPPLY", "RESISTANCE") else 0.0)
        is_snr_z = _is_snr_zone_obj(z_item)
        z_src_l = "SNR" if is_snr_z else "SMC"
        sl_v = resolve_stored_setup_sl(
            z_src_l,
            str((z_item or {}).get("zone_id") or "") if isinstance(z_item, dict) else "",
            trade_plans=data.get("trade_plans") if isinstance(data.get("trade_plans"), list) else None,
            best_trade_plan=data.get("best_trade_plan") if isinstance(data.get("best_trade_plan"), dict) else None,
            timeframe=str((z_item or {}).get("timeframe") or "").strip().upper() if isinstance(z_item, dict) else None,
            setup_version=(z_item.get("setup_version") if (isinstance(z_item, dict) and isinstance(z_item.get("setup_version"), int)) else None),
        )
        if sl_v <= 0 and not is_snr_z:
            sl_v = resolve_smc_setup_invalidation_price(z_item)
        segs = []
        if s_inv > 0:
            segs.append(f"StructInv {s_inv:.2f}")
        if sl_v > 0 and abs(sl_v - s_inv) >= 0.01:
            segs.append(f"SetupSL {sl_v:.2f}")
        return (" | " + " | ".join(segs)) if segs else ""

    lo = safe_float(zone.get("price_low", zone.get("low")), 0.0)
    hi = safe_float(zone.get("price_high", zone.get("high")), 0.0)
    radar_cand = data.get("radar_candidate") if isinstance(data.get("radar_candidate"), dict) else None
    far_ref_rep = data.get("far_reference_zone") if isinstance(data.get("far_reference_zone"), dict) else None
    shown_active_z_rep = None
    if not is_mkt_closed_rep and lo > 0 and hi > 0:
        shown_active_z_rep = zone
        z_tf = str(zone.get("timeframe") or "").strip()
        z_type = str(zone.get("zone_type") or zone.get("type") or "ZONE").upper()
        z_grd = str(zone.get("grade") or "-")
        z_sc_r = f"/{safe_float(zone.get('score')):.1f}" if zone.get("score") is not None else ""
        z_inv_s = _fmt_rep_inv_tags(zone, lo, hi, z_type)
        dist_v = data.get("distance")
        d_s = "IN ZONE" if (dist_v is not None and dist_v <= 0) else (format_price_distance(dist_v, spec=_u_rep) if dist_v is not None else "N/A")
        z_rel_raw_r = str(zone.get("zone_relevance") or zone.get("lifecycle_status") or zone.get("status") or "ACTIVE").upper()
        mon_st_r = str(data.get("monitor_state") or "").upper()
        if str(zone.get("radar_eligibility") or "").upper() == "NOT_QUALIFIED" or str(zone.get("risk_tier") or "").upper() == "FAR_ZONE":
            excl_rsn_r = str(zone.get("risk_tier_reason") or zone.get("pending_rejection_reason") or "EXCLUDED_FROM_RADAR").strip()
            z_rel_r = f"EXCLUDED ({excl_rsn_r})"
        elif mon_st_r in ("ZONE_TOUCHED", "AWAITING_CONFIRMATION") or z_rel_raw_r == "TOUCHED":
            z_rel_r = "ACTIVE (TOUCHED)"
        elif z_rel_raw_r in ("ACTIVE", "RADAR"):
            z_rel_r = z_rel_raw_r
        else:
            z_rel_r = f"ACTIVE/{z_rel_raw_r}"
        z_src_r = str(zone.get("source") or ("SNR" if _is_snr_zone_obj(zone) else "SMC")).upper()
        z_id_s = f" [ID:{zone['zone_id']}]" if zone.get("zone_id") and zone.get("zone_id") != "N/A" else ""
        active_line = f"{z_tf} {z_type} {lo:.2f}-{hi:.2f} | ZoneGrade:{z_grd}{z_sc_r} | Dist {d_s}{z_inv_s} | {z_rel_r} [{z_src_r}]{z_id_s}".strip()
    elif radar_cand and safe_float(radar_cand.get("price_low", radar_cand.get("low")), 0.0) > 0:
        shown_active_z_rep = radar_cand
        r_lo = safe_float(radar_cand.get("price_low", radar_cand.get("low")), 0.0)
        r_hi = safe_float(radar_cand.get("price_high", radar_cand.get("high")), 0.0)
        r_zt = str(radar_cand.get("zone_type") or "").upper()
        r_sc = f"/{safe_float(radar_cand.get('score')):.1f}" if radar_cand.get("score") is not None else ""
        r_d = safe_float(radar_cand.get("distance"), 0.0)
        r_inv_s = _fmt_rep_inv_tags(radar_cand, r_lo, r_hi, r_zt)
        r_id = f" [ID:{radar_cand['zone_id']}]" if radar_cand.get("zone_id") else ""
        active_line = f"RADAR {radar_cand.get('timeframe', '')} {r_zt} {r_lo:.2f}-{r_hi:.2f} | G:{radar_cand.get('grade', '-')}{r_sc} | Dist {format_price_distance(r_d, spec=_u_rep)}{r_inv_s}{r_id}".replace("  ", " ")
    elif far_ref_rep and safe_float(far_ref_rep.get("price_low", far_ref_rep.get("low")), 0.0) > 0:
        shown_active_z_rep = far_ref_rep
        fr_lo = safe_float(far_ref_rep.get("price_low", far_ref_rep.get("low")), 0.0)
        fr_hi = safe_float(far_ref_rep.get("price_high", far_ref_rep.get("high")), 0.0)
        fr_zt = str(far_ref_rep.get("zone_type") or "").upper()
        fr_sc = f"/{safe_float(far_ref_rep.get('score')):.1f}" if far_ref_rep.get("score") is not None else ""
        fr_d = safe_float(far_ref_rep.get("distance", far_ref_rep.get("distance_points")), 0.0)
        fr_inv_s = _fmt_rep_inv_tags(far_ref_rep, fr_lo, fr_hi, fr_zt)
        fr_id = f" [ID:{far_ref_rep['zone_id']}]" if far_ref_rep.get("zone_id") else ""
        active_line = f"NONE | FAR REF {far_ref_rep.get('timeframe', '')} {fr_zt} {fr_lo:.2f}-{fr_hi:.2f} | G:{far_ref_rep.get('grade', '-')}{fr_sc} | Dist {format_price_distance(fr_d, spec=_u_rep)}{fr_inv_s}{fr_id}".replace("  ", " ")
    else:
        active_line = "NONE"

    snr_h4_rep = data.get("snr_h4_bias") or h4
    snr_h1_rep = str(data.get("snr_h1_bias") or "").upper()
    active_why_rep = _build_monitor_zone_why(shown_active_z_rep, smc_h4=h4, snr_h4=snr_h4_rep, smc_h1=h1, snr_h1=snr_h1_rep)
    inv_disp_rep = str(data.get("latest_invalid_display") or "NONE")
    invalid_why_rep = (
        _build_monitor_invalid_why(data.get("latest_invalid_zone"), smc_h4=h4, snr_h4=snr_h4_rep, smc_h1=h1, snr_h1=snr_h1_rep)
        if inv_disp_rep != "NONE"
        else ""
    )

    snr_sup = data.get("snr_nearest_support")
    snr_res = data.get("snr_nearest_resistance")
    snr_parts = []
    s1_why_rep = ""
    r1_why_rep = ""
    if isinstance(snr_sup, dict) and safe_float(snr_sup.get("low")) > 0:
        snr_parts.append(f"S1 {safe_float(snr_sup.get('low')):.2f}-{safe_float(snr_sup.get('high')):.2f}(G:{snr_sup.get('grade', '-')})")
        s1_why_rep = build_snr_zone_reason(snr_sup, h4_bias=snr_h4_rep, h1_bias=snr_h1_rep)
    if isinstance(snr_res, dict) and safe_float(snr_res.get("low")) > 0:
        snr_parts.append(f"R1 {safe_float(snr_res.get('low')):.2f}-{safe_float(snr_res.get('high')):.2f}(G:{snr_res.get('grade', '-')})")
        r1_why_rep = build_snr_zone_reason(snr_res, h4_bias=snr_h4_rep, h1_bias=snr_h1_rep)
    if data.get("confluence") == "ALIGNED":
        snr_parts.append("CONFLUENCE")
    snr_line = " | ".join(snr_parts) if snr_parts else "NONE"

    pos_c = int(safe_float(acc.get("open_positions"), mt5_snap.get("symbol_positions_count", 0)))
    ord_c = int(safe_float(acc.get("pending_orders"), len(mt5_snap.get("orders") or [])))
    nosl_c = int(mt5_risk.get("positions_without_sl_count", safe_float(acc.get("position_without_sl"), 0))) + int(mt5_risk.get("pending_without_sl_count", safe_float(acc.get("pending_without_sl"), 0)))
    sync_rep_t = format_mt5_sync_short(mt5_snap, fallback_ts=snap_ts_rep)
    risk_head = str(data.get("critical_risk_display") or "").strip()
    risk_line = f"{risk_head} | Pos:{pos_c} Pend:{ord_c} NoSL:{nosl_c} | Sync:{sync_rep_t}" if risk_head else f"AMAN | Pos:{pos_c} Pend:{ord_c} NoSL:{nosl_c} | Sync:{sync_rep_t}"

    tracked_pos_rep = data.get("tracked_position") if isinstance(data.get("tracked_position"), dict) else None
    h1_dir_rep = (
        f"H1(SMC)={h1} H1(SNR)={snr_h1_rep}"
        if (snr_h1_rep and snr_h1_rep not in ("N/A", "UNKNOWN", "DATA UNAVAILABLE") and snr_h1_rep != h1)
        else f"H1(SMC)={h1}"
    )
    snr_m30_rep = str(data.get("snr_m30_bias") or "").upper()
    m30_dir_rep = (
        f"M30(SMC)={m30} M30(SNR)={snr_m30_rep}"
        if (snr_m30_rep and snr_m30_rep not in ("N/A", "UNKNOWN", "UNCONFIRMED", "NO_DATA", "DATA UNAVAILABLE") and snr_m30_rep != m30)
        else f"M30(SMC)={m30}"
    )
    best_tp_rep = data.get("best_trade_plan") if isinstance(data.get("best_trade_plan"), dict) else None
    if best_tp_rep:
        best_tp_rep = refresh_trade_plan_live_state(
            best_tp_rep,
            live_price=safe_float(data.get("price"), 0.0),
            bid=bid_r if bid_r > 0 else None,
            ask=ask_r if ask_r > 0 else None,
            is_closed=is_mkt_closed_rep,
            m5_eval_status=m5,
            mt5_orders_snapshot=mt5_snap,
            htf_conflict=bool(data.get("htf_conflict")),
            h4_bias=h4,
        )
    _pend_same_rep = bool(
        pend_line and best_tp_rep
        and str(best_tp_rep.get("zone_id") or "") == str(pending_s.get("zone_id") or "")
        and str(best_tp_rep.get("status") or "").upper() == "READY"
    )
    term_w_rep = get_terminal_width(default=100, min_width=64, max_width=120)
    div_rep = "=" * min(70, term_w_rep)
    _tp_rep_lines = format_trade_plan_lines(
        best_tp_rep,
        prefix="TRADE PLAN   : ",
        why_prefix="  WHY PLAN   : ",
        width=term_w_rep,
        include_source=True,
        broker_tag=(str(pending_s.get("status") or "") if _pend_same_rep else None),
        skip_why_for_zone_id=(shown_active_z_rep.get("zone_id") if isinstance(shown_active_z_rep, dict) else None),
    ) if best_tp_rep else []
    lines = [
        div_rep,
        "XAUUSD SMART MONITOR",
        div_rep,
        *wrap_dashboard_field("MARKET       : ", f"{mkt_lbl} | {price_txt} | {snap_ts_rep}{id_sfx}", term_w_rep, max_lines=None),
        *wrap_dashboard_field("MARKET BIAS  : ", f"H4={h4} {h1_dir_rep} {m30_dir_rep} M5={m5}", term_w_rep, max_lines=None),
        *wrap_dashboard_field("ENTRY STATUS : ", entry_line, term_w_rep, max_lines=None),
        *_tp_rep_lines,
        *(wrap_dashboard_field("PENDING PLAN : ", pend_line, term_w_rep, max_lines=None) if (pend_line and not _pend_same_rep) else []),
        *wrap_dashboard_field("ACTIVE/RADAR : ", active_line, term_w_rep, max_lines=None),
        *(wrap_dashboard_field("  WHY        : ", active_why_rep, term_w_rep, max_lines=None) if active_why_rep else []),
        *wrap_dashboard_field("INVALID      : ", inv_disp_rep, term_w_rep, max_lines=None),
        *(wrap_dashboard_field("  WHY        : ", invalid_why_rep, term_w_rep, max_lines=None) if invalid_why_rep else []),
        *wrap_dashboard_field("SNR          : ", snr_line, term_w_rep, max_lines=None),
        *(wrap_dashboard_field("  WHY S1     : ", s1_why_rep, term_w_rep, max_lines=None) if s1_why_rep else []),
        *(wrap_dashboard_field("  WHY R1     : ", r1_why_rep, term_w_rep, max_lines=None) if r1_why_rep else []),
        *wrap_dashboard_field("RISK         : ", risk_line, term_w_rep, max_lines=None),
    ]
    if tracked_pos_rep:
        from modules.mt5_conn import format_tracked_position_compact
        tp_rep_txt = format_tracked_position_compact(tracked_pos_rep)
        if tp_rep_txt:
            lines.extend(wrap_dashboard_field("TRACKED POS  : ", tp_rep_txt, term_w_rep, max_lines=None))
    news_summary = str((cycle_result or {}).get("news_summary") or "").strip() if isinstance(cycle_result, dict) else ""
    if news_summary:
        lines.extend(wrap_dashboard_field("US NEWS      : ", news_summary.replace('NEWS RISK : ', '').replace(' | NEXT EVENT : ', ' | '), term_w_rep, max_lines=None))
    lines.append(div_rep)
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
        f"H4 / H1     : {summary.get('h4_bias', 'N/A')} / SMC Struct={summary.get('h1_bias', 'N/A')}" + (f" | SNR Trend={data.get('snr_h1_bias')}" if data.get("snr_h1_bias") and data.get("snr_h1_bias") not in ("N/A", "UNKNOWN", "DATA UNAVAILABLE") else ""),
        f"M30         : M30(SMC)={summary.get('m30_bias', 'N/A')}" + (f" | M30(SNR)={data.get('snr_m30_bias')}" if data.get("snr_m30_bias") and data.get("snr_m30_bias") not in ("N/A", "UNKNOWN", "UNCONFIRMED", "NO_DATA", "DATA UNAVAILABLE") else ""),
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


