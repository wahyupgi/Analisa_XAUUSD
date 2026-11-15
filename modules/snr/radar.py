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
from modules.snr.config import get_snr_config
from modules.snr.data_provider import safe_float


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
    if rej.get("confirmed") and rej.get("status") in ("BULLISH_REJECTION", "BEARISH_REJECTION"):
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
    bo_state = str(pa.get("breakout_code", "NO_BREAKOUT")).upper()
    rt_state = str(pa.get("retest_code", "NO_RETEST")).upper()
    rej_state = str(pa.get("rejection_code", "NONE")).upper()

    sig = str((decision or {}).get("signal", "WAIT")).upper()
    setup_type = str((decision or {}).get("setup_type", "NONE")).upper()
    mkt_status = str((decision or {}).get("market_status", "OPEN")).upper()

    # Tentukan status ringkas radar
    if mkt_status == "DATA UNAVAILABLE":
        radar_status = "DATA_UNAVAILABLE"
    elif mkt_status == "CLOSED":
        radar_status = "MARKET_CLOSED"
    elif sig in ("BUY", "SELL"):
        radar_status = f"CONFIRMED_{sig}_SETUP"
    elif rej_state in ("BULLISH_REJECTION", "BEARISH_REJECTION"):
        radar_status = f"ZONE_{rej_state}"
    elif rt_state == "RETEST_CONFIRMED":
        radar_status = "RETEST_CONFIRMED"
    elif bo_state == "VALID_BREAKOUT":
        radar_status = "BREAKOUT_CONFIRMED"
    elif s1_state == "TOUCH" or r1_state == "TOUCH":
        radar_status = "ZONE_TOUCH"
    elif s1_state == "APPROACHING" or r1_state == "APPROACHING":
        radar_status = "ZONE_APPROACHING"
    elif sig == "INVALID":
        radar_status = "ZONE_INVALIDATED"
    else:
        radar_status = "STANDBY_MONITORING"

    is_live_open = mkt_status == "OPEN"
    return {
        "radar_status": radar_status,
        "market_status": mkt_status,
        "nearest_support": {
            "label": s1.get("rank_label", "S1") if s1 else "N/A",
            "low": s1.get("low") if s1 else None,
            "high": s1.get("high") if s1 else None,
            "grade": s1.get("grade") if s1 else "N/A",
            "score": s1.get("score_int") if s1 else 0,
            "strength": s1.get("strength_short") if s1 else "N/A",
            "distance_points": s1.get("distance_points") if s1 else None,
            "state": s1_state,
        },
        "nearest_resistance": {
            "label": r1.get("rank_label", "R1") if r1 else "N/A",
            "low": r1.get("low") if r1 else None,
            "high": r1.get("high") if r1 else None,
            "grade": r1.get("grade") if r1 else "N/A",
            "score": r1.get("score_int") if r1 else 0,
            "strength": r1.get("strength_short") if r1 else "N/A",
            "distance_points": r1.get("distance_points") if r1 else None,
            "state": r1_state,
        },
        "strong_zone_count": len(strong_zones),
        "broken_zone_count": len(broken_zones),
        "invalidated_zone_count": len(invalid_zones),
        "events": {
            "zone_approaching": s1_state == "APPROACHING" or r1_state == "APPROACHING",
            "zone_touch": s1_state == "TOUCH" or r1_state == "TOUCH",
            "zone_rejection": is_live_open and (rej_state in ("BULLISH_REJECTION", "BEARISH_REJECTION")),
            "breakout_alert": is_live_open and (bo_state == "VALID_BREAKOUT"),
            "retest_alert": is_live_open and (rt_state in ("RETEST_IN_PROGRESS", "RETEST_CONFIRMED")),
            "zone_broken": len(broken_zones) > 0,
            "zone_invalidated": len(invalid_zones) > 0 or sig == "INVALID",
            "confirmed_setup": is_live_open and (sig in ("BUY", "SELL")),
            "setup_type": setup_type,
        },
    }


class SnrAlertTracker:
    """
    Pelacak perubahan status (State-Change Detector) untuk SNR Radar.
    Memastikan alert [SNR] hanya muncul saat terjadi perubahan status penting,
    sementara peringatan risiko kritis tetap disertakan secara terpisah.
    """

    def __init__(self):
        self.last_state = {}

    def reset(self):
        self.last_state = {}

    def evaluate_alerts(self, radar_result, decision, price_action=None):
        """
        Membandingkan state siklus saat ini dengan `self.last_state`.
        Mengembalikan:
        - new_alerts: list of str (hanya alert yang berubah pada siklus ini, berawalan `[SNR]`)
        - critical_warnings: list of str (peringatan kritis yang wajib tetap terlihat)
        """
        radar = radar_result or {}
        dec = decision or {}
        pa = price_action or {}

        s1 = radar.get("nearest_support") or {}
        r1 = radar.get("nearest_resistance") or {}
        mkt_status = str(dec.get("market_status", radar.get("market_status", "OPEN"))).upper()

        curr_state = {
            "market_status": mkt_status,
            "signal": str(dec.get("signal", "WAIT")).upper(),
            "setup_type": str(dec.get("setup_type", "NONE")).upper(),
            "s1_band": (s1.get("low"), s1.get("high")),
            "s1_state": str(s1.get("state", "NONE")).upper(),
            "r1_band": (r1.get("low"), r1.get("high")),
            "r1_state": str(r1.get("state", "NONE")).upper(),
            "breakout_code": str(pa.get("breakout_code", "NO_BREAKOUT")).upper(),
            "breakout_dir": str(pa.get("breakout_direction", "NONE")).upper(),
            "retest_code": str(pa.get("retest_code", "NO_RETEST")).upper(),
            "rejection_code": str(pa.get("rejection_code", "NONE")).upper(),
            "broken_count": int(radar.get("broken_zone_count", 0)),
            "invalid_count": int(radar.get("invalidated_zone_count", 0)),
        }

        alerts = []
        critical_warnings = []

        # Saat market tutup atau data tidak tersedia, jangan klaim alert live breakout/rejection/setup baru
        if mkt_status == "CLOSED":
            if self.last_state.get("market_status") != "CLOSED":
                alerts.append("[SNR] MARKET CLOSED — ANALISIS HISTORICAL AKTIF (LIVE ENTRY: WAIT)")
            self.last_state = curr_state
            return {
                "alerts": alerts,
                "critical_warnings": critical_warnings,
                "state": curr_state,
            }
        if mkt_status == "DATA UNAVAILABLE":
            if self.last_state.get("market_status") != "DATA UNAVAILABLE":
                alerts.append("[SNR] DATA UNAVAILABLE — HISTORICAL DATA TIDAK TERSEDIA")
            self.last_state = curr_state
            return {
                "alerts": alerts,
                "critical_warnings": critical_warnings,
                "state": curr_state,
            }

        # 1. Alert Perubahan Keputusan Setup (BUY / SELL / INVALID)
        if curr_state["signal"] != self.last_state.get("signal"):
            if curr_state["signal"] == "BUY":
                alerts.append("[SNR] BUY SETUP TERKONFIRMASI")
            elif curr_state["signal"] == "SELL":
                alerts.append("[SNR] SELL SETUP TERKONFIRMASI")
            elif curr_state["signal"] == "INVALID":
                alerts.append("[SNR] SETUP / ZONA INVALID")

        # 2. Alert Perubahan Rejection
        if curr_state["rejection_code"] != self.last_state.get("rejection_code"):
            if curr_state["rejection_code"] == "BULLISH_REJECTION":
                alerts.append("[SNR] SUPPORT REJECTION TERKONFIRMASI")
            elif curr_state["rejection_code"] == "BEARISH_REJECTION":
                alerts.append("[SNR] RESISTANCE REJECTION TERKONFIRMASI")

        # 3. Alert Perubahan Breakout
        if (
            curr_state["breakout_code"] != self.last_state.get("breakout_code")
            or curr_state["breakout_dir"] != self.last_state.get("breakout_dir")
        ):
            if curr_state["breakout_code"] == "VALID_BREAKOUT":
                if curr_state["breakout_dir"] == "BULLISH":
                    alerts.append("[SNR] RESISTANCE BREAKOUT TERKONFIRMASI")
                else:
                    alerts.append("[SNR] SUPPORT BREAKDOWN TERKONFIRMASI")
            elif curr_state["breakout_code"] == "FALSE_BREAKOUT":
                alerts.append("[SNR] FALSE BREAKOUT TERDETEKSI")

        # 4. Alert Perubahan Retest
        if curr_state["retest_code"] != self.last_state.get("retest_code"):
            if curr_state["retest_code"] == "RETEST_CONFIRMED":
                alerts.append("[SNR] RETEST TERKONFIRMASI")
            elif curr_state["retest_code"] == "RETEST_IN_PROGRESS":
                alerts.append("[SNR] RETEST SEDANG BERLANGSUNG")
            elif curr_state["retest_code"] == "RETEST_FAILED":
                alerts.append("[SNR] RETEST GAGAL")

        # 5. Alert Interaksi S1 & R1 (Approaching / Touch)
        if curr_state["s1_state"] != self.last_state.get("s1_state") or curr_state["s1_band"] != self.last_state.get("s1_band"):
            if curr_state["s1_state"] == "TOUCH" and s1.get("low") is not None:
                alerts.append(f"[SNR] SUPPORT S1 DISENTUH ({s1['low']:.2f}-{s1['high']:.2f})")
            elif curr_state["s1_state"] == "APPROACHING" and s1.get("low") is not None:
                alerts.append(f"[SNR] HARGA MENDEKATI SUPPORT S1 ({s1['low']:.2f}-{s1['high']:.2f})")

        if curr_state["r1_state"] != self.last_state.get("r1_state") or curr_state["r1_band"] != self.last_state.get("r1_band"):
            if curr_state["r1_state"] == "TOUCH" and r1.get("low") is not None:
                alerts.append(f"[SNR] RESISTANCE R1 DISENTUH ({r1['low']:.2f}-{r1['high']:.2f})")
            elif curr_state["r1_state"] == "APPROACHING" and r1.get("low") is not None:
                alerts.append(f"[SNR] HARGA MENDEKATI RESISTANCE R1 ({r1['low']:.2f}-{r1['high']:.2f})")

        # 6. Alert Zona Broken / Invalidated Baru
        prev_broken = self.last_state.get("broken_count")
        if prev_broken is not None and curr_state["broken_count"] > prev_broken:
            alerts.append("[SNR] ZONA SNR DITEMBUS (BROKEN)")

        prev_invalid = self.last_state.get("invalid_count")
        if prev_invalid is not None and curr_state["invalid_count"] > prev_invalid:
            alerts.append("[SNR] ZONA SNR INVALID")

        # Peringatan kritis yang wajib tetap muncul
        if curr_state["signal"] == "INVALID":
            critical_warnings.append("[SNR CRITICAL] Setup/Zona fokus saat ini telah INVALIDATED.")

        self.last_state = curr_state
        return {
            "alerts": alerts,
            "critical_warnings": critical_warnings,
            "state": curr_state,
        }

