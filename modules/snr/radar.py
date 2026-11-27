# -*- coding: utf-8 -*-
"""
radar.py — Smart SNR Radar & Detektor Perubahan Status Alert ([SNR]) Independen.

Fitur Wajib (Section S):
- Monitoring support terdekat (S1)
- Monitoring resistance terdekat (R1)
- Monitoring strong zone (STRONG SUPPORT & STRONG RESISTANCE)
- Zone approaching
- Zone touch
- Zone rejection
- Breakout alert
- Retest alert
- Zone broken
- Zone invalidated
- Confirmed BUY/SELL setup
- Deteksi perubahan status agar alert tidak diulang pada siklus berikutnya jika status tetap.
"""
import json
import os
from modules.snr.config import FILE_STATE_SNR_LIFECYCLE, get_snr_config
from modules.snr.data_provider import safe_float


SNR_EMITTED_EVENTS_KEY = "snr_emitted_events"
_MAX_EMITTED_SNR_EVENTS = 300


def _snr_event_dedup_key(zone_id, event_type, timestamp):
    ts = str(timestamp or "").strip()
    if not ts:
        return None
    zid = str(zone_id or "GLOBAL").strip().upper()
    ev = str(event_type or "EVENT").strip().upper()
    return f"{zid}|{ev}|{ts}"


def _load_persisted_snr_emitted_events():
    if not os.path.exists(FILE_STATE_SNR_LIFECYCLE):
        return []
    try:
        from modules.config import read_shared_zone_state
        data = read_shared_zone_state(FILE_STATE_SNR_LIFECYCLE)
        if not isinstance(data, dict):
            return []
        raw = data.get(SNR_EMITTED_EVENTS_KEY)
        if not isinstance(raw, list):
            return []
        return [str(item) for item in raw if isinstance(item, str) and item.strip()][-_MAX_EMITTED_SNR_EVENTS:]
    except Exception:
        return []


def _save_persisted_snr_emitted_events(keys):
    try:
        from modules.config import update_shared_zone_state
        val = list(keys)[-_MAX_EMITTED_SNR_EVENTS:]
        update_shared_zone_state(
            lambda d: d.__setitem__(SNR_EMITTED_EVENTS_KEY, val),
            filepath=FILE_STATE_SNR_LIFECYCLE,
            tag="snr_ev",
        )
    except Exception:
        pass


def _evaluate_zone_proximity_state(zone, current_price, radar_cfg):
    """Menentukan status interaksi harga terhadap satu zona (DISTANT / APPROACHING / TOUCH / REJECTION / BROKEN / INVALIDATED)."""
    if not zone or not isinstance(zone, dict):
        return "NONE"

    lifecycle = str(zone.get("lifecycle_status", zone.get("status", "ACTIVE"))).upper()
    if lifecycle == "INVALIDATED":
        return "INVALIDATED"
    if lifecycle == "BROKEN":
        return "BROKEN"

    rej = zone.get("rejection_info") or {}
    if rej.get("confirmed") and not rej.get("is_historical") and rej.get("status") in ("BULLISH_REJECTION", "BEARISH_REJECTION"):
        return "REJECTION"

    cp = safe_float(current_price, 0.0)
    z_low = safe_float(zone.get("low"), 0.0)
    z_high = safe_float(zone.get("high"), z_low)
    ref_atr = max(0.25, safe_float(zone.get("ref_atr"), 2.0))

    if cp <= 0:
        return "DISTANT"

    touch_atr = safe_float(radar_cfg.get("touch_distance_atr", 0.08), 0.08)
    app_atr = safe_float(radar_cfg.get("approaching_distance_atr", 0.60), 0.60)

    if (z_low - touch_atr * ref_atr) <= cp <= (z_high + touch_atr * ref_atr):
        return "TOUCH"

    dist_pts = min(abs(cp - z_low), abs(cp - z_high))
    dist_atr = dist_pts / ref_atr
    if dist_atr <= app_atr:
        return "APPROACHING"
    return "DISTANT"


def evaluate_snr_radar(ranked_zones, decision, price_action=None, current_price=None, config=None):
    """
    Mengevaluasi status Smart SNR Radar secara independen tanpa membaca/mengubah Radar SMC.
    """
    cfg = config or get_snr_config()
    radar_cfg = cfg.get("radar", {})
    cp = safe_float(current_price, 0.0)

    active_sups = (ranked_zones or {}).get("active_supports") or []
    active_ress = (ranked_zones or {}).get("active_resistances") or []
    broken_zones = (ranked_zones or {}).get("broken_zones") or []
    invalid_zones = (ranked_zones or {}).get("invalidated_zones") or []
    by_cat = (ranked_zones or {}).get("by_category") or {}

    s1 = active_sups[0] if active_sups else None
    r1 = active_ress[0] if active_ress else None

    s1_state = _evaluate_zone_proximity_state(s1, cp, radar_cfg) if s1 else "NONE"
    r1_state = _evaluate_zone_proximity_state(r1, cp, radar_cfg) if r1 else "NONE"

    strong_zones = list(by_cat.get("STRONG SUPPORT", [])) + list(by_cat.get("STRONG RESISTANCE", []))

    pa = price_action or {}
    recent_touch_id = pa.get("recent_touch_zone_id")
    if recent_touch_id and cp > 0:
        if s1 and s1.get("zone_id") == recent_touch_id and s1_state in ("TOUCH", "APPROACHING", "DISTANT"):
            if not safe_float(s1.get("low")) <= cp <= safe_float(s1.get("high")):
                s1_state = "RECENT_TOUCH"
        if r1 and r1.get("zone_id") == recent_touch_id and r1_state in ("TOUCH", "APPROACHING", "DISTANT"):
            if not safe_float(r1.get("low")) <= cp <= safe_float(r1.get("high")):
                r1_state = "RECENT_TOUCH"
    bo_state = str(pa.get("breakout_code", "NO_BREAKOUT")).upper()
    bo_dir = str(pa.get("breakout_direction", "NONE")).upper()
    bo_detail = pa.get("breakout_detail") or {}
    bo_is_hist = bool(pa.get("breakout_is_historical") or bo_detail.get("is_historical"))
    rt_state = str(pa.get("retest_code", "NO_RETEST")).upper()
    rt_detail = pa.get("retest_detail") or {}
    rt_is_hist = bool(pa.get("retest_is_historical") or rt_detail.get("is_historical"))
    rej_state = str(pa.get("rejection_code", "NONE")).upper()
    rej_detail = pa.get("rejection_detail") or {}
    rej_is_hist = bool(pa.get("rejection_is_historical") or rej_detail.get("is_historical"))
    pending_breaks = (ranked_zones or {}).get("pending_break_zones") or []

    sig = str((decision or {}).get("signal", "WAIT")).upper()
    setup_type = str((decision or {}).get("setup_type", "NONE")).upper()
    mkt_status = str((decision or {}).get("market_status", "OPEN")).upper()

    # Tentukan status ringkas radar (hanya event fresh/aktif, bukan event historis)
    if mkt_status == "DATA UNAVAILABLE":
        radar_status = "DATA_UNAVAILABLE"
    elif mkt_status == "CLOSED":
        radar_status = "MARKET_CLOSED"
    elif sig in ("BUY", "SELL"):
        radar_status = f"CONFIRMED_{sig}_SETUP"
    elif rej_state in ("BULLISH_REJECTION", "BEARISH_REJECTION") and not rej_is_hist:
        radar_status = f"ZONE_{rej_state}"
    elif rt_state == "RETEST_CONFIRMED" and not rt_is_hist:
        radar_status = "RETEST_CONFIRMED"
    elif bo_state == "VALID_BREAKOUT" and not bo_is_hist:
        radar_status = "BREAKDOWN_CONFIRMED" if bo_dir == "BEARISH" else "BREAKOUT_CONFIRMED"
    elif bo_state == "BREAKOUT_PENDING_CONFIRMATION" and not bo_is_hist:
        radar_status = "POTENTIAL_BREAKDOWN" if bo_dir == "BEARISH" else "POTENTIAL_BREAKOUT"
    elif s1_state == "TOUCH" or r1_state == "TOUCH":
        radar_status = "ZONE_TOUCH"
    elif s1_state == "RECENT_TOUCH" or r1_state == "RECENT_TOUCH":
        radar_status = "ZONE_RECENT_TOUCH"
    elif s1_state == "APPROACHING" or r1_state == "APPROACHING":
        radar_status = "ZONE_APPROACHING"
    elif sig == "INVALID":
        radar_status = "ZONE_INVALIDATED"
    else:
        radar_status = "STANDBY_MONITORING"

    is_live_open = mkt_status == "OPEN"
    has_pot_breakdown = is_live_open and (
        (bool(pa.get("potential_breakdown")) and bo_state != "VALID_BREAKOUT")
        or (bo_state == "BREAKOUT_PENDING_CONFIRMATION" and bo_dir == "BEARISH" and not bo_is_hist)
        or any(
            str(z.get("lifecycle_status", "")).upper() == "BREAKDOWN_PENDING"
            and not z.get("is_broken")
            and not z.get("is_invalidated")
            for z in pending_breaks
        )
    )
    has_pot_breakout = is_live_open and (
        (bool(pa.get("potential_breakout")) and bo_state != "VALID_BREAKOUT")
        or (bo_state == "BREAKOUT_PENDING_CONFIRMATION" and bo_dir == "BULLISH" and not bo_is_hist)
        or any(
            str(z.get("lifecycle_status", "")).upper() == "BREAKOUT_PENDING"
            and not z.get("is_broken")
            and not z.get("is_invalidated")
            for z in pending_breaks
        )
    )
    reclaimed_zones = (ranked_zones or {}).get("reclaimed_zones") or []
    zone_meta_map = {}
    for z_item in list(pending_breaks) + list(broken_zones) + list(reclaimed_zones) + list(invalid_zones):
        if not isinstance(z_item, dict) or not z_item.get("zone_id"):
            continue
        zid = z_item["zone_id"]
        if zid in zone_meta_map:
            continue
        z_tf = str(z_item.get("timeframe") or "").upper()
        z_tp = str(z_item.get("zone_type") or "").upper()
        z_lo = safe_float(z_item.get("low"), 0.0)
        z_hi = safe_float(z_item.get("high"), z_lo)
        z_lbl = f"{z_tf} {z_tp} {z_lo:.2f}-{z_hi:.2f}".strip()
        bo_i = z_item.get("breakout_info") if isinstance(z_item.get("breakout_info"), dict) else {}
        ev_t = z_item.get("lifecycle_event_time") or bo_i.get("breakout_bar_time") or None
        zone_meta_map[zid] = {
            "zone_id": zid,
            "label": z_lbl,
            "timeframe": z_tf,
            "zone_type": z_tp,
            "low": z_lo,
            "high": z_hi,
            "lifecycle_status": str(z_item.get("lifecycle_status", z_item.get("status", ""))).upper(),
            "event_time": str(ev_t) if ev_t else None,
            "is_historical": bool(z_item.get("lifecycle_is_historical") or z_item.get("is_distant_historical")),
        }

    return {
        "radar_status": radar_status,
        "market_status": mkt_status,
        "nearest_support": {
            "zone_id": s1.get("zone_id") if s1 else None,
            "label": s1.get("rank_label", "S1") if s1 else "N/A",
            "low": s1.get("low") if s1 else None,
            "high": s1.get("high") if s1 else None,
            "grade": s1.get("grade") if s1 else "N/A",
            "score": s1.get("score_int") if s1 else 0,
            "strength": s1.get("strength_short") if s1 else "N/A",
            "distance_points": s1.get("distance_points") if s1 else None,
            "state": s1_state,
            "price_inside": bool(s1 and cp > 0 and safe_float(s1.get("low")) <= cp <= safe_float(s1.get("high"))),
        },
        "nearest_resistance": {
            "zone_id": r1.get("zone_id") if r1 else None,
            "label": r1.get("rank_label", "R1") if r1 else "N/A",
            "low": r1.get("low") if r1 else None,
            "high": r1.get("high") if r1 else None,
            "grade": r1.get("grade") if r1 else "N/A",
            "score": r1.get("score_int") if r1 else 0,
            "strength": r1.get("strength_short") if r1 else "N/A",
            "distance_points": r1.get("distance_points") if r1 else None,
            "state": r1_state,
            "price_inside": bool(r1 and cp > 0 and safe_float(r1.get("low")) <= cp <= safe_float(r1.get("high"))),
        },
        "strong_zone_count": len(strong_zones),
        "broken_zone_count": len(broken_zones),
        "pending_break_count": len(pending_breaks),
        "invalidated_zone_count": len(invalid_zones),
        "broken_zone_ids": [
            z.get("zone_id") for z in broken_zones
            if z.get("zone_id") and str(z.get("lifecycle_status", z.get("status", ""))).upper() == "BROKEN"
        ],
        "pending_break_zone_ids": [
            z.get("zone_id") for z in pending_breaks
            if z.get("zone_id") and str(z.get("lifecycle_status", z.get("status", ""))).upper() in ("BREAKDOWN_PENDING", "BREAKOUT_PENDING")
        ],
        "reclaim_pending_zone_ids": [
            z.get("zone_id") for z in reclaimed_zones
            if z.get("zone_id") and str(z.get("lifecycle_status", z.get("status", ""))).upper() == "RECLAIM_PENDING"
        ],
        "reclaimed_zone_ids": [
            z.get("zone_id") for z in reclaimed_zones
            if z.get("zone_id") and str(z.get("lifecycle_status", z.get("status", ""))).upper() == "RECLAIMED"
        ],
        "invalidated_zone_ids": [
            z.get("zone_id") for z in invalid_zones
            if z.get("zone_id") and str(z.get("lifecycle_status", z.get("status", ""))).upper() == "INVALIDATED"
        ],
        "observed_zone_ids": [z.get("zone_id") for z in ((ranked_zones or {}).get("all_zones") or []) if z.get("zone_id")],
        "zone_meta_map": zone_meta_map,
        "events": {
            "zone_approaching": s1_state == "APPROACHING" or r1_state == "APPROACHING",
            "zone_touch": s1_state == "TOUCH" or r1_state == "TOUCH",
            "zone_recent_touch": s1_state == "RECENT_TOUCH" or r1_state == "RECENT_TOUCH",
            "zone_rejection": is_live_open and (rej_state in ("BULLISH_REJECTION", "BEARISH_REJECTION")) and not rej_is_hist,
            "breakout_alert": is_live_open and (bo_state == "VALID_BREAKOUT") and not bo_is_hist,
            "potential_breakdown": has_pot_breakdown,
            "potential_breakout": has_pot_breakout,
            "reclaim_detected": is_live_open and bool(pa.get("reclaim_detected") or bo_detail.get("reclaim_detected")) and not bo_is_hist,
            "retest_alert": is_live_open and (rt_state in ("RETEST_IN_PROGRESS", "RETEST_CONFIRMED")) and not rt_is_hist,
            "zone_broken": len(broken_zones) > 0,
            "zone_invalidated": len(invalid_zones) > 0 or sig == "INVALID",
            "confirmed_setup": is_live_open and (sig in ("BUY", "SELL")),
            "setup_type": setup_type,
        },
    }


_GLOBAL_LAST_SNR_STATE = {}
_GLOBAL_EMITTED_SNR_EVENTS = []


class SnrAlertTracker:
    """
    Pelacak perubahan status (State-Change Detector) untuk SNR Radar.
    Memastikan alert [SNR] hanya muncul saat terjadi perubahan status penting,
    menduplikasi-filter event berdasarkan (zone_id, event_type, timestamp),
    mempertahankan informasi historis secara terpisah, dan menjaga peringatan kritis tetap terlihat.
    """

    def __init__(self, persist_dedup=True):
        global _GLOBAL_LAST_SNR_STATE, _GLOBAL_EMITTED_SNR_EVENTS
        self.persist_dedup = bool(persist_dedup)
        self.last_state = dict(_GLOBAL_LAST_SNR_STATE)
        if self.persist_dedup and not _GLOBAL_EMITTED_SNR_EVENTS:
            _GLOBAL_EMITTED_SNR_EVENTS = _load_persisted_snr_emitted_events()
        self.emitted_events = list(_GLOBAL_EMITTED_SNR_EVENTS)
        self._emitted_set = set(self.emitted_events)

    def reset(self):
        global _GLOBAL_LAST_SNR_STATE, _GLOBAL_EMITTED_SNR_EVENTS
        self.last_state = {}
        _GLOBAL_LAST_SNR_STATE = {}
        self.emitted_events = []
        self._emitted_set = set()
        _GLOBAL_EMITTED_SNR_EVENTS = []

    def _claim_event(self, zone_id, event_type, timestamp):
        """
        Klaim event ber-timestamp agar (zone_id, event_type, timestamp) tidak pernah terulang
        sebagai alert baru (termasuk saat fokus S1<->R1 berganti atau restart monitor).
        """
        global _GLOBAL_EMITTED_SNR_EVENTS
        key = _snr_event_dedup_key(zone_id, event_type, timestamp)
        if not key:
            return True
        if key in self._emitted_set:
            return False
        self.emitted_events.append(key)
        self._emitted_set.add(key)
        if len(self.emitted_events) > _MAX_EMITTED_SNR_EVENTS:
            overflow = len(self.emitted_events) - _MAX_EMITTED_SNR_EVENTS
            removed = self.emitted_events[:overflow]
            self.emitted_events = self.emitted_events[overflow:]
            self._emitted_set.difference_update(removed)
        _GLOBAL_EMITTED_SNR_EVENTS = list(self.emitted_events)
        if self.persist_dedup:
            _save_persisted_snr_emitted_events(self.emitted_events)
        return True

    def evaluate_alerts(self, radar_result, decision, price_action=None):
        """
        Membandingkan state siklus saat ini dengan `self.last_state` dan registry dedup event.
        Mengembalikan:
        - alerts: list of str (hanya alert baru/berubah pada siklus ini, berawalan `[SNR]`)
        - historical_events: list of dict (informasi event historis yang relevan, bukan alert baru)
        - critical_warnings: list of str (peringatan kritis yang wajib tetap terlihat)
        """
        global _GLOBAL_LAST_SNR_STATE
        radar = radar_result or {}
        dec = decision or {}
        pa = price_action or {}

        s1 = radar.get("nearest_support") or {}
        r1 = radar.get("nearest_resistance") or {}
        mkt_status = str(dec.get("market_status", radar.get("market_status", "OPEN"))).upper()

        bo_zone = str(pa.get("breakout_zone_label") or "").strip()
        bo_is_hist = bool(
            pa.get("breakout_is_historical")
            or (pa.get("breakout_detail") or {}).get("is_historical")
        )
        rt_zone = str(pa.get("retest_zone_label") or "").strip()
        rt_is_hist = bool(
            pa.get("retest_is_historical")
            or (pa.get("retest_detail") or {}).get("is_historical")
        )
        rej_zone = str(pa.get("rejection_zone_label") or "").strip()
        rej_detail = pa.get("rejection_detail") or {}
        rej_is_hist = bool(
            pa.get("rejection_is_historical")
            or rej_detail.get("is_historical")
        )

        zone_meta_map = radar.get("zone_meta_map") if isinstance(radar.get("zone_meta_map"), dict) else {}
        bo_detail_map = pa.get("breakout_detail") or {}
        rt_detail_map = pa.get("retest_detail") or {}

        curr_state = {
            "market_status": mkt_status,
            "signal": str(dec.get("signal", "WAIT")).upper(),
            "setup_type": str(dec.get("setup_type", "NONE")).upper(),
            "setup_zone_id": (dec.get("target_zone") or {}).get("zone_id"),
            "s1_id": s1.get("zone_id"),
            "s1_band": (s1.get("low"), s1.get("high")),
            "s1_state": str(s1.get("state", "NONE")).upper(),
            "s1_inside": bool(s1.get("price_inside")),
            "r1_band": (r1.get("low"), r1.get("high")),
            "r1_id": r1.get("zone_id"),
            "r1_state": str(r1.get("state", "NONE")).upper(),
            "r1_inside": bool(r1.get("price_inside")),
            "breakout_code": str(pa.get("breakout_code", "NO_BREAKOUT")).upper(),
            "breakout_dir": str(pa.get("breakout_direction", "NONE")).upper(),
            "breakout_zone": bo_zone,
            "breakout_zone_id": bo_detail_map.get("zone_id"),
            "breakout_bar_time": str(bo_detail_map.get("breakout_bar_time") or ""),
            "breakout_is_hist": bo_is_hist,
            "retest_code": str(pa.get("retest_code", "NO_RETEST")).upper(),
            "retest_zone": rt_zone,
            "retest_zone_id": rt_detail_map.get("zone_id"),
            "retest_bar_time": str(rt_detail_map.get("retest_bar_time") or ""),
            "retest_is_hist": rt_is_hist,
            "rejection_code": str(pa.get("rejection_code", "NONE")).upper(),
            "rejection_zone": rej_zone,
            "rejection_zone_id": rej_detail.get("zone_id"),
            "rejection_bar_time": str(rej_detail.get("bar_time") or ""),
            "rejection_is_hist": rej_is_hist,
            "broken_count": int(radar.get("broken_zone_count", 0)),
            "invalid_count": int(radar.get("invalidated_zone_count", 0)),
            "broken_ids": radar.get("broken_zone_ids") or [],
            "pending_break_ids": radar.get("pending_break_zone_ids") or [],
            "reclaim_pending_ids": radar.get("reclaim_pending_zone_ids") or [],
            "reclaimed_ids": radar.get("reclaimed_zone_ids") or [],
            "invalid_ids": radar.get("invalidated_zone_ids") or [],
            "observed_ids": radar.get("observed_zone_ids") or [],
        }

        alerts = []
        alert_events = []
        historical_events = []
        critical_warnings = []

        def _emit(msg, category="MARKET_ALERT", zone_id=None, zone_label=None, timestamp=None):
            alerts.append(msg)
            alert_events.append({
                "category": category,
                "source": "SNR",
                "zone_id": zone_id,
                "zone_label": zone_label,
                "timestamp": str(timestamp) if timestamp else None,
                "message": msg,
            })

        def _record_historical(event_type, zone_id, zone_label, timestamp, msg):
            historical_events.append({
                "category": "HISTORICAL_INFO",
                "source": "SNR",
                "zone_id": zone_id,
                "zone_label": zone_label,
                "event_type": event_type,
                "timestamp": str(timestamp) if timestamp else None,
                "message": msg,
            })

        def _ts_tag(ts_val):
            s_ts = str(ts_val or "").strip()
            return f" [{s_ts}]" if s_ts else ""

        # Saat market tutup atau data tidak tersedia, jangan klaim alert live breakout/rejection/setup baru
        if mkt_status == "CLOSED":
            if self.last_state.get("market_status") != "CLOSED":
                _emit("[SNR] MARKET CLOSED — ANALISIS HISTORICAL AKTIF (LIVE ENTRY: WAIT)", category="HISTORICAL_INFO")
            self.last_state = curr_state
            _GLOBAL_LAST_SNR_STATE = dict(curr_state)
            return {
                "alerts": alerts,
                "alert_events": alert_events,
                "historical_events": historical_events,
                "critical_warnings": critical_warnings,
                "state": curr_state,
            }
        if mkt_status == "DATA UNAVAILABLE":
            if self.last_state.get("market_status") != "DATA UNAVAILABLE":
                _emit("[SNR] DATA UNAVAILABLE — HISTORICAL DATA TIDAK TERSEDIA", category="ACCOUNT_ALERT")
            self.last_state = curr_state
            _GLOBAL_LAST_SNR_STATE = dict(curr_state)
            return {
                "alerts": alerts,
                "alert_events": alert_events,
                "historical_events": historical_events,
                "critical_warnings": critical_warnings,
                "state": curr_state,
            }

        # 1. Alert Perubahan Keputusan Setup (BUY / SELL / INVALID)
        tgt_z = dec.get("target_zone") if isinstance(dec.get("target_zone"), dict) else {}
        tgt_lbl = str(dec.get("zone_label") or "").strip()
        if not tgt_lbl or tgt_lbl == "N/A":
            if tgt_z.get("low") is not None:
                tgt_lbl = f"{tgt_z.get('timeframe', '')} {tgt_z.get('zone_type', '')} {safe_float(tgt_z.get('low')):.2f}-{safe_float(tgt_z.get('high')):.2f}".strip()
            else:
                tgt_lbl = ""
        tgt_suffix = f" ({tgt_lbl})" if tgt_lbl else ""
        if (curr_state["signal"], curr_state["setup_zone_id"]) != (
            self.last_state.get("signal"), self.last_state.get("setup_zone_id")
        ):
            if curr_state["signal"] == "BUY":
                _emit(f"[SNR] BUY SETUP TERKONFIRMASI{tgt_suffix}", category="ENTRY_ALERT", zone_id=curr_state["setup_zone_id"], zone_label=tgt_lbl or None)
            elif curr_state["signal"] == "SELL":
                _emit(f"[SNR] SELL SETUP TERKONFIRMASI{tgt_suffix}", category="ENTRY_ALERT", zone_id=curr_state["setup_zone_id"], zone_label=tgt_lbl or None)
            elif curr_state["signal"] == "INVALID":
                _emit(f"[SNR] SETUP / ZONA INVALID{tgt_suffix}", category="MARKET_ALERT", zone_id=curr_state["setup_zone_id"], zone_label=tgt_lbl or None)

        # 2. Alert Perubahan Rejection (dedup ketat berdasarkan zone_id + event_type + timestamp; historis dipertahankan terpisah)
        rej_confirmed = bool(
            rej_detail.get("confirmed")
            or str(pa.get("rejection_summary") or "").upper() == "CONFIRMED"
        )
        if rej_confirmed and curr_state["rejection_code"] in ("BULLISH_REJECTION", "BEARISH_REJECTION"):
            rej_suffix = f" ({rej_zone})" if rej_zone and rej_zone != "NONE" else ""
            rej_ts = rej_detail.get("bar_time")
            rej_zid = curr_state["rejection_zone_id"] or rej_zone
            rej_state_changed = (
                curr_state["rejection_code"] != self.last_state.get("rejection_code")
                or curr_state["rejection_zone_id"] != self.last_state.get("rejection_zone_id")
                or curr_state["rejection_bar_time"] != self.last_state.get("rejection_bar_time")
            )
            rej_base_lbl = (
                "SUPPORT REJECTION" if curr_state["rejection_code"] == "BULLISH_REJECTION"
                else "RESISTANCE REJECTION"
            )
            if rej_is_hist:
                _record_historical(
                    curr_state["rejection_code"],
                    rej_zid,
                    rej_zone or None,
                    rej_ts,
                    f"[SNR] {rej_base_lbl} (HISTORIS){rej_suffix}{_ts_tag(rej_ts)}",
                )
            elif (rej_ts and self._claim_event(rej_zid, curr_state["rejection_code"], rej_ts)) or (
                not rej_ts and rej_state_changed
            ):
                _emit(
                    f"[SNR] {rej_base_lbl} TERKONFIRMASI{rej_suffix}{_ts_tag(rej_ts)}",
                    category="MARKET_ALERT",
                    zone_id=curr_state["rejection_zone_id"],
                    zone_label=rej_zone or None,
                    timestamp=rej_ts,
                )

        # 3. Alert Perubahan Breakout / Breakdown / False Breakout / Reclaim (dedup zone_id + event_type + timestamp)
        bo_reclaim = bool(pa.get("reclaim_detected") or bo_detail_map.get("reclaim_detected"))
        bo_ts = bo_detail_map.get("breakout_bar_time")
        bo_zid = curr_state["breakout_zone_id"] or bo_zone
        bo_ev_key = f"{curr_state['breakout_code']}:{curr_state['breakout_dir']}:{'RECLAIM' if bo_reclaim else 'STD'}"
        bo_state_changed = (
            curr_state["breakout_code"] != self.last_state.get("breakout_code")
            or curr_state["breakout_dir"] != self.last_state.get("breakout_dir")
            or curr_state["breakout_zone_id"] != self.last_state.get("breakout_zone_id")
            or curr_state["breakout_bar_time"] != self.last_state.get("breakout_bar_time")
        )
        bo_suffix = f" ({bo_zone})" if bo_zone and bo_zone != "NONE" else ""
        if bo_is_hist and curr_state["breakout_code"] in ("VALID_BREAKOUT", "FALSE_BREAKOUT"):
            bo_hist_tag = (
                "RECLAIM PASCA-BREAK" if bo_reclaim else (
                    "RESISTANCE BREAKOUT" if curr_state["breakout_dir"] == "BULLISH" else "SUPPORT BREAKDOWN"
                )
            )
            _record_historical(
                bo_ev_key,
                bo_zid,
                bo_zone or None,
                bo_ts,
                f"[SNR] {bo_hist_tag} (HISTORIS){bo_suffix}{_ts_tag(bo_ts)}",
            )
        elif not bo_is_hist and curr_state["breakout_code"] in ("VALID_BREAKOUT", "BREAKOUT_PENDING_CONFIRMATION", "FALSE_BREAKOUT"):
            bo_allow = (bo_ts and self._claim_event(bo_zid, bo_ev_key, bo_ts)) or (not bo_ts and bo_state_changed)
            if bo_allow:
                if curr_state["breakout_code"] == "VALID_BREAKOUT":
                    if curr_state["breakout_dir"] == "BULLISH":
                        _emit(f"[SNR] RESISTANCE BREAKOUT TERKONFIRMASI{bo_suffix}{_ts_tag(bo_ts)}", category="MARKET_ALERT", zone_id=curr_state["breakout_zone_id"], zone_label=bo_zone or None, timestamp=bo_ts)
                    else:
                        _emit(f"[SNR] SUPPORT BREAKDOWN TERKONFIRMASI{bo_suffix}{_ts_tag(bo_ts)}", category="MARKET_ALERT", zone_id=curr_state["breakout_zone_id"], zone_label=bo_zone or None, timestamp=bo_ts)
                elif curr_state["breakout_code"] == "BREAKOUT_PENDING_CONFIRMATION":
                    if curr_state["breakout_dir"] == "BULLISH":
                        _emit(f"[SNR] POTENSI RESISTANCE BREAKOUT{bo_suffix}{_ts_tag(bo_ts)}", category="MARKET_ALERT", zone_id=curr_state["breakout_zone_id"], zone_label=bo_zone or None, timestamp=bo_ts)
                    else:
                        _emit(f"[SNR] POTENSI SUPPORT BREAKDOWN{bo_suffix}{_ts_tag(bo_ts)}", category="MARKET_ALERT", zone_id=curr_state["breakout_zone_id"], zone_label=bo_zone or None, timestamp=bo_ts)
                elif curr_state["breakout_code"] == "FALSE_BREAKOUT":
                    if bo_reclaim:
                        _emit(f"[SNR] RECLAIM PASCA-BREAK TERDETEKSI{bo_suffix}{_ts_tag(bo_ts)}", category="MARKET_ALERT", zone_id=curr_state["breakout_zone_id"], zone_label=bo_zone or None, timestamp=bo_ts)
                    elif curr_state["breakout_dir"] == "BEARISH":
                        _emit(f"[SNR] FALSE BREAKDOWN TERDETEKSI{bo_suffix}{_ts_tag(bo_ts)}", category="MARKET_ALERT", zone_id=curr_state["breakout_zone_id"], zone_label=bo_zone or None, timestamp=bo_ts)
                    else:
                        _emit(f"[SNR] FALSE BREAKOUT TERDETEKSI{bo_suffix}{_ts_tag(bo_ts)}", category="MARKET_ALERT", zone_id=curr_state["breakout_zone_id"], zone_label=bo_zone or None, timestamp=bo_ts)

        # 4. Alert Perubahan Retest (hanya jika aktif fresh, dedup zone_id + event_type + timestamp; hindari duplikasi lintas siklus/section)
        rt_ts = rt_detail_map.get("retest_bar_time") or bo_ts
        rt_zid = curr_state["retest_zone_id"] or rt_zone
        rt_is_bo_inv = bool(rt_detail_map.get("breakout_invalidated"))
        rt_inv_ev_key = "BREAKDOWN_INVALIDATED" if str(rt_detail_map.get("zone_type") or "").upper() == "SUPPORT" else "BREAKOUT_INVALIDATED"
        rt_ev_key = rt_inv_ev_key if (curr_state["retest_code"] == "RETEST_FAILED" and rt_is_bo_inv) else curr_state["retest_code"]
        rt_state_changed = (
            curr_state["retest_code"] != self.last_state.get("retest_code")
            or curr_state["retest_zone_id"] != self.last_state.get("retest_zone_id")
            or curr_state["retest_bar_time"] != self.last_state.get("retest_bar_time")
        )
        rt_suffix = f" ({rt_zone})" if rt_zone and rt_zone != "NONE" else ""
        if rt_is_hist and curr_state["retest_code"] == "RETEST_CONFIRMED":
            _record_historical(
                curr_state["retest_code"],
                rt_zid,
                rt_zone or None,
                rt_ts,
                f"[SNR] RETEST (HISTORIS){rt_suffix}{_ts_tag(rt_ts)}",
            )
        elif not rt_is_hist and curr_state["retest_code"] in ("RETEST_CONFIRMED", "RETEST_IN_PROGRESS", "RETEST_FAILED"):
            rt_allow = (rt_ts and self._claim_event(rt_zid, rt_ev_key, rt_ts)) or (not rt_ts and rt_state_changed)
            if rt_allow:
                if curr_state["retest_code"] == "RETEST_CONFIRMED":
                    _emit(f"[SNR] RETEST TERKONFIRMASI{rt_suffix}{_ts_tag(rt_ts)}", category="MARKET_ALERT", zone_id=curr_state["retest_zone_id"], zone_label=rt_zone or None, timestamp=rt_ts)
                elif curr_state["retest_code"] == "RETEST_IN_PROGRESS":
                    _emit(f"[SNR] RETEST IN PROGRESS{rt_suffix}{_ts_tag(rt_ts)}", category="MARKET_ALERT", zone_id=curr_state["retest_zone_id"], zone_label=rt_zone or None, timestamp=rt_ts)
                elif curr_state["retest_code"] == "RETEST_FAILED":
                    if rt_is_bo_inv:
                        rt_inv_lbl = "BREAKDOWN INVALIDATED" if str(rt_detail_map.get("zone_type") or "").upper() == "SUPPORT" else "BREAKOUT INVALIDATED"
                        _emit(f"[SNR] {rt_inv_lbl}{rt_suffix}{_ts_tag(rt_ts)}", category="MARKET_ALERT", zone_id=curr_state["retest_zone_id"], zone_label=rt_zone or None, timestamp=rt_ts)
                    else:
                        _emit(f"[SNR] RETEST FAILED{rt_suffix}{_ts_tag(rt_ts)}", category="MARKET_ALERT", zone_id=curr_state["retest_zone_id"], zone_label=rt_zone or None, timestamp=rt_ts)

        # 5. Alert Interaksi S1 & R1 (Approaching / Touch — abaikan penuh bila zona sudah ketembus SL / invalid)
        from modules.utils import is_zone_sl_breached_or_invalidated
        s1_invalid_sl = bool(s1 and is_zone_sl_breached_or_invalidated(s1, default_source="SNR"))
        r1_invalid_sl = bool(r1 and is_zone_sl_breached_or_invalidated(r1, default_source="SNR"))

        if (not s1_invalid_sl) and (curr_state["s1_state"], curr_state["s1_id"], curr_state["s1_inside"]) != (
            self.last_state.get("s1_state"), self.last_state.get("s1_id"), self.last_state.get("s1_inside")
        ):
            s1_lbl = f"S1 {s1['low']:.2f}-{s1['high']:.2f}" if s1.get("low") is not None else None
            s1_retouch = int(safe_float(s1.get("touch_count"), 0)) >= 1 or str(s1.get("lifecycle_event") or "").upper() in ("TOUCHED", "SWEEP")
            if curr_state["s1_state"] == "TOUCH" and s1.get("low") is not None:
                label = (
                    ("HARGA DISENTUH KEMBALI SUPPORT S1" if s1_retouch else "HARGA DI SUPPORT S1")
                    if s1.get("price_inside")
                    else "HARGA DEKAT SUPPORT S1"
                )
                _emit(f"[SNR] {label} ({s1['low']:.2f}-{s1['high']:.2f})", category="MARKET_ALERT", zone_id=s1.get("zone_id"), zone_label=s1_lbl)
            elif curr_state["s1_state"] == "RECENT_TOUCH" and s1.get("low") is not None:
                _emit(f"[SNR] SUPPORT S1 DISENTUH M5; HARGA DI LUAR ZONA ({s1['low']:.2f}-{s1['high']:.2f})", category="MARKET_ALERT", zone_id=s1.get("zone_id"), zone_label=s1_lbl)
            elif curr_state["s1_state"] == "APPROACHING" and s1.get("low") is not None:
                _emit(f"[SNR] HARGA DEKAT SUPPORT S1 ({s1['low']:.2f}-{s1['high']:.2f})", category="MARKET_ALERT", zone_id=s1.get("zone_id"), zone_label=s1_lbl)

        if (not r1_invalid_sl) and (curr_state["r1_state"], curr_state["r1_id"], curr_state["r1_inside"]) != (
            self.last_state.get("r1_state"), self.last_state.get("r1_id"), self.last_state.get("r1_inside")
        ):
            r1_lbl = f"R1 {r1['low']:.2f}-{r1['high']:.2f}" if r1.get("low") is not None else None
            r1_retouch = int(safe_float(r1.get("touch_count"), 0)) >= 1 or str(r1.get("lifecycle_event") or "").upper() in ("TOUCHED", "SWEEP")
            if curr_state["r1_state"] == "TOUCH" and r1.get("low") is not None:
                label = (
                    ("HARGA DISENTUH KEMBALI RESISTANCE R1" if r1_retouch else "HARGA DI RESISTANCE R1")
                    if r1.get("price_inside")
                    else "HARGA DEKAT RESISTANCE R1"
                )
                _emit(f"[SNR] {label} ({r1['low']:.2f}-{r1['high']:.2f})", category="MARKET_ALERT", zone_id=r1.get("zone_id"), zone_label=r1_lbl)
            elif curr_state["r1_state"] == "RECENT_TOUCH" and r1.get("low") is not None:
                _emit(f"[SNR] RESISTANCE R1 DISENTUH M5; HARGA DI LUAR ZONA ({r1['low']:.2f}-{r1['high']:.2f})", category="MARKET_ALERT", zone_id=r1.get("zone_id"), zone_label=r1_lbl)
            elif curr_state["r1_state"] == "APPROACHING" and r1.get("low") is not None:
                _emit(f"[SNR] HARGA DEKAT RESISTANCE R1 ({r1['low']:.2f}-{r1['high']:.2f})", category="MARKET_ALERT", zone_id=r1.get("zone_id"), zone_label=r1_lbl)

        # 6. Alert Transisi Lifecycle Zona (Pending Break / Broken / Reclaim Pending / Reclaimed / Invalidated)
        # Hanya terbit pada perubahan state + dedup (zone_id, event_type, timestamp).
        observed_prev = set(self.last_state.get("observed_ids") or [])
        if "pending_break_ids" in self.last_state:
            new_pend = set(curr_state["pending_break_ids"]) - set(self.last_state["pending_break_ids"])
            for zone_id in sorted(new_pend & observed_prev):
                zm = zone_meta_map.get(zone_id) or {}
                if zm.get("is_historical"):
                    continue
                z_tp = str(zm.get("zone_type") or "").upper()
                z_lbl = zm.get("label") or zone_id
                ev_t = zm.get("event_time")
                if not self._claim_event(zone_id, "PENDING_BREAK", ev_t):
                    continue
                tag = "POTENSI SUPPORT BREAKDOWN" if z_tp == "SUPPORT" else "POTENSI RESISTANCE BREAKOUT"
                _emit(f"[SNR] {tag} ({z_lbl}){_ts_tag(ev_t)}", category="MARKET_ALERT", zone_id=zone_id, zone_label=z_lbl, timestamp=ev_t)

        if "broken_ids" in self.last_state:
            new_broken = set(curr_state["broken_ids"]) - set(self.last_state["broken_ids"])
            for zone_id in sorted(new_broken & observed_prev):
                zm = zone_meta_map.get(zone_id) or {}
                if zm.get("is_historical"):
                    continue
                ev_t = zm.get("event_time")
                if not self._claim_event(zone_id, "BROKEN", ev_t):
                    continue
                _emit(f"[SNR] ZONA {zone_id} DITEMBUS (BROKEN){_ts_tag(ev_t)}", category="MARKET_ALERT", zone_id=zone_id, zone_label=zm.get("label") or zone_id, timestamp=ev_t)

        if "reclaim_pending_ids" in self.last_state:
            new_rec_p = set(curr_state["reclaim_pending_ids"]) - set(self.last_state["reclaim_pending_ids"])
            for zone_id in sorted(new_rec_p & observed_prev):
                zm = zone_meta_map.get(zone_id) or {}
                if zm.get("is_historical"):
                    continue
                z_lbl = zm.get("label") or zone_id
                ev_t = zm.get("event_time")
                if not self._claim_event(zone_id, "RETEST_FAILED", ev_t):
                    continue
                _emit(f"[SNR] RETEST FAILED ({z_lbl}){_ts_tag(ev_t)}", category="MARKET_ALERT", zone_id=zone_id, zone_label=z_lbl, timestamp=ev_t)

        if "reclaimed_ids" in self.last_state:
            new_rec_c = set(curr_state["reclaimed_ids"]) - set(self.last_state["reclaimed_ids"])
            for zone_id in sorted(new_rec_c & observed_prev):
                zm = zone_meta_map.get(zone_id) or {}
                if zm.get("is_historical"):
                    continue
                z_lbl = zm.get("label") or zone_id
                ev_t = zm.get("event_time")
                z_tp_rc = str(zm.get("zone_type") or "").upper()
                rc_inv_ev = "BREAKDOWN_INVALIDATED" if z_tp_rc == "SUPPORT" else "BREAKOUT_INVALIDATED"
                if not self._claim_event(zone_id, rc_inv_ev, ev_t):
                    continue
                rc_inv_lbl = "BREAKDOWN INVALIDATED" if z_tp_rc == "SUPPORT" else "BREAKOUT INVALIDATED"
                _emit(f"[SNR] {rc_inv_lbl} ({z_lbl}){_ts_tag(ev_t)}", category="MARKET_ALERT", zone_id=zone_id, zone_label=z_lbl, timestamp=ev_t)

        if "invalid_ids" in self.last_state:
            new_invalid = set(curr_state["invalid_ids"]) - set(self.last_state["invalid_ids"])
            for zone_id in sorted(new_invalid & observed_prev):
                zm = zone_meta_map.get(zone_id) or {}
                if zm.get("is_historical"):
                    continue
                ev_t = zm.get("event_time")
                if not self._claim_event(zone_id, "INVALIDATED", ev_t):
                    continue
                _emit(f"[SNR] ZONA {zone_id} INVALID{_ts_tag(ev_t)}", category="MARKET_ALERT", zone_id=zone_id, zone_label=zm.get("label") or zone_id, timestamp=ev_t)

        # Peringatan kritis yang wajib tetap muncul
        if curr_state["signal"] == "INVALID":
            critical_warnings.append("[SNR CRITICAL] Setup/Zona fokus saat ini telah INVALIDATED.")

        self.last_state = curr_state
        _GLOBAL_LAST_SNR_STATE = dict(curr_state)
        return {
            "alerts": alerts,
            "alert_events": alert_events,
            "historical_events": historical_events,
            "critical_warnings": critical_warnings,
            "state": curr_state,
        }

