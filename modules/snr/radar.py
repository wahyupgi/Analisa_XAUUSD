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
    recent_touch_id = pa.get("recent_touch_zone_id")
    if recent_touch_id and cp > 0:
        if s1 and s1.get("zone_id") == recent_touch_id and s1_state in ("TOUCH", "APPROACHING", "DISTANT"):
            if not safe_float(s1.get("low")) <= cp <= safe_float(s1.get("high")):
                s1_state = "RECENT_TOUCH"
        if r1 and r1.get("zone_id") == recent_touch_id and r1_state in ("TOUCH", "APPROACHING", "DISTANT"):
            if not safe_float(r1.get("low")) <= cp <= safe_float(r1.get("high")):
                r1_state = "RECENT_TOUCH"
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
    elif s1_state == "RECENT_TOUCH" or r1_state == "RECENT_TOUCH":
        radar_status = "ZONE_RECENT_TOUCH"
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
        "invalidated_zone_count": len(invalid_zones),
        "broken_zone_ids": [
            z.get("zone_id") for z in broken_zones
            if z.get("zone_id") and str(z.get("lifecycle_status", z.get("status", ""))).upper() == "BROKEN"
        ],
        "invalidated_zone_ids": [
            z.get("zone_id") for z in invalid_zones
            if z.get("zone_id") and str(z.get("lifecycle_status", z.get("status", ""))).upper() == "INVALIDATED"
        ],
        "observed_zone_ids": [z.get("zone_id") for z in ((ranked_zones or {}).get("all_zones") or []) if z.get("zone_id")],
        "events": {
            "zone_approaching": s1_state == "APPROACHING" or r1_state == "APPROACHING",
            "zone_touch": s1_state == "TOUCH" or r1_state == "TOUCH",
            "zone_recent_touch": s1_state == "RECENT_TOUCH" or r1_state == "RECENT_TOUCH",
            "zone_rejection": is_live_open and (rej_state in ("BULLISH_REJECTION", "BEARISH_REJECTION")),
            "breakout_alert": is_live_open and (bo_state == "VALID_BREAKOUT"),
            "retest_alert": is_live_open and (rt_state in ("RETEST_IN_PROGRESS", "RETEST_CONFIRMED")),
            "zone_broken": len(broken_zones) > 0,
            "zone_invalidated": len(invalid_zones) > 0 or sig == "INVALID",
            "confirmed_setup": is_live_open and (sig in ("BUY", "SELL")),
            "setup_type": setup_type,
        },
    }


_GLOBAL_LAST_SNR_STATE = {}


class SnrAlertTracker:
    """
    Pelacak perubahan status (State-Change Detector) untuk SNR Radar.
    Memastikan alert [SNR] hanya muncul saat terjadi perubahan status penting,
    sementara peringatan risiko kritis tetap disertakan secara terpisah.
    """

    def __init__(self):
        global _GLOBAL_LAST_SNR_STATE
        self.last_state = dict(_GLOBAL_LAST_SNR_STATE)

    def reset(self):
        global _GLOBAL_LAST_SNR_STATE
        self.last_state = {}
        _GLOBAL_LAST_SNR_STATE = {}

    def evaluate_alerts(self, radar_result, decision, price_action=None):
        """
        Membandingkan state siklus saat ini dengan `self.last_state`.
        Mengembalikan:
        - new_alerts: list of str (hanya alert yang berubah pada siklus ini, berawalan `[SNR]`)
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
            "breakout_zone_id": (pa.get("breakout_detail") or {}).get("zone_id"),
            "breakout_is_hist": bo_is_hist,
            "retest_code": str(pa.get("retest_code", "NO_RETEST")).upper(),
            "retest_zone": rt_zone,
            "retest_zone_id": (pa.get("retest_detail") or {}).get("zone_id"),
            "retest_is_hist": rt_is_hist,
            "rejection_code": str(pa.get("rejection_code", "NONE")).upper(),
            "rejection_zone": rej_zone,
            "rejection_zone_id": rej_detail.get("zone_id"),
            "broken_count": int(radar.get("broken_zone_count", 0)),
            "invalid_count": int(radar.get("invalidated_zone_count", 0)),
            "broken_ids": radar.get("broken_zone_ids") or [],
            "invalid_ids": radar.get("invalidated_zone_ids") or [],
            "observed_ids": radar.get("observed_zone_ids") or [],
        }

        alerts = []
        critical_warnings = []

        # Saat market tutup atau data tidak tersedia, jangan klaim alert live breakout/rejection/setup baru
        if mkt_status == "CLOSED":
            if self.last_state.get("market_status") != "CLOSED":
                alerts.append("[SNR] MARKET CLOSED — ANALISIS HISTORICAL AKTIF (LIVE ENTRY: WAIT)")
            self.last_state = curr_state
            _GLOBAL_LAST_SNR_STATE = dict(curr_state)
            return {
                "alerts": alerts,
                "critical_warnings": critical_warnings,
                "state": curr_state,
            }
        if mkt_status == "DATA UNAVAILABLE":
            if self.last_state.get("market_status") != "DATA UNAVAILABLE":
                alerts.append("[SNR] DATA UNAVAILABLE — HISTORICAL DATA TIDAK TERSEDIA")
            self.last_state = curr_state
            _GLOBAL_LAST_SNR_STATE = dict(curr_state)
            return {
                "alerts": alerts,
                "critical_warnings": critical_warnings,
                "state": curr_state,
            }

        # 1. Alert Perubahan Keputusan Setup (BUY / SELL / INVALID)
        if (curr_state["signal"], curr_state["setup_zone_id"]) != (
            self.last_state.get("signal"), self.last_state.get("setup_zone_id")
        ):
            if curr_state["signal"] == "BUY":
                alerts.append("[SNR] BUY SETUP TERKONFIRMASI")
            elif curr_state["signal"] == "SELL":
                alerts.append("[SNR] SELL SETUP TERKONFIRMASI")
            elif curr_state["signal"] == "INVALID":
                alerts.append("[SNR] SETUP / ZONA INVALID")

        # 2. Alert Perubahan Rejection (dengan identitas zona)
        if rej_detail.get("confirmed") and not rej_detail.get("is_historical") and (
            curr_state["rejection_code"] != self.last_state.get("rejection_code")
            or curr_state["rejection_zone_id"] != self.last_state.get("rejection_zone_id")
        ):
            rej_suffix = f" ({rej_zone})" if rej_zone and rej_zone != "NONE" else ""
            if curr_state["rejection_code"] == "BULLISH_REJECTION":
                alerts.append(f"[SNR] SUPPORT REJECTION TERKONFIRMASI{rej_suffix}")
            elif curr_state["rejection_code"] == "BEARISH_REJECTION":
                alerts.append(f"[SNR] RESISTANCE REJECTION TERKONFIRMASI{rej_suffix}")

        # 3. Alert Perubahan Breakout (hanya untuk breakout aktif fresh, sertakan identitas zona)
        if not bo_is_hist and (
            curr_state["breakout_code"] != self.last_state.get("breakout_code")
            or curr_state["breakout_dir"] != self.last_state.get("breakout_dir")
            or curr_state["breakout_zone_id"] != self.last_state.get("breakout_zone_id")
        ):
            bo_suffix = f" ({bo_zone})" if bo_zone and bo_zone != "NONE" else ""
            if curr_state["breakout_code"] == "VALID_BREAKOUT":
                if curr_state["breakout_dir"] == "BULLISH":
                    alerts.append(f"[SNR] RESISTANCE BREAKOUT TERKONFIRMASI{bo_suffix}")
                else:
                    alerts.append(f"[SNR] SUPPORT BREAKDOWN TERKONFIRMASI{bo_suffix}")
            elif curr_state["breakout_code"] == "FALSE_BREAKOUT":
                alerts.append(f"[SNR] FALSE BREAKOUT TERDETEKSI{bo_suffix}")

        # 4. Alert Perubahan Retest (hanya jika aktif fresh, sertakan identitas zona)
        if not rt_is_hist and (
            curr_state["retest_code"] != self.last_state.get("retest_code")
            or curr_state["retest_zone_id"] != self.last_state.get("retest_zone_id")
        ):
            rt_suffix = f" ({rt_zone})" if rt_zone and rt_zone != "NONE" else ""
            if curr_state["retest_code"] == "RETEST_CONFIRMED":
                alerts.append(f"[SNR] RETEST TERKONFIRMASI{rt_suffix}")
            elif curr_state["retest_code"] == "RETEST_IN_PROGRESS":
                alerts.append(f"[SNR] RETEST SEDANG BERLANGSUNG{rt_suffix}")
            elif curr_state["retest_code"] == "RETEST_FAILED":
                alerts.append(f"[SNR] RETEST GAGAL{rt_suffix}")

        # 5. Alert Interaksi S1 & R1 (Approaching / Touch)
        if (curr_state["s1_state"], curr_state["s1_id"], curr_state["s1_inside"]) != (
            self.last_state.get("s1_state"), self.last_state.get("s1_id"), self.last_state.get("s1_inside")
        ):
            if curr_state["s1_state"] == "TOUCH" and s1.get("low") is not None:
                label = "HARGA DI SUPPORT S1" if s1.get("price_inside") else "HARGA DEKAT SUPPORT S1"
                alerts.append(f"[SNR] {label} ({s1['low']:.2f}-{s1['high']:.2f})")
            elif curr_state["s1_state"] == "RECENT_TOUCH" and s1.get("low") is not None:
                alerts.append(f"[SNR] SUPPORT S1 DISENTUH M5; HARGA DI LUAR ZONA ({s1['low']:.2f}-{s1['high']:.2f})")
            elif curr_state["s1_state"] == "APPROACHING" and s1.get("low") is not None:
                alerts.append(f"[SNR] HARGA DEKAT SUPPORT S1 ({s1['low']:.2f}-{s1['high']:.2f})")

        if (curr_state["r1_state"], curr_state["r1_id"], curr_state["r1_inside"]) != (
            self.last_state.get("r1_state"), self.last_state.get("r1_id"), self.last_state.get("r1_inside")
        ):
            if curr_state["r1_state"] == "TOUCH" and r1.get("low") is not None:
                label = "HARGA DI RESISTANCE R1" if r1.get("price_inside") else "HARGA DEKAT RESISTANCE R1"
                alerts.append(f"[SNR] {label} ({r1['low']:.2f}-{r1['high']:.2f})")
            elif curr_state["r1_state"] == "RECENT_TOUCH" and r1.get("low") is not None:
                alerts.append(f"[SNR] RESISTANCE R1 DISENTUH M5; HARGA DI LUAR ZONA ({r1['low']:.2f}-{r1['high']:.2f})")
            elif curr_state["r1_state"] == "APPROACHING" and r1.get("low") is not None:
                alerts.append(f"[SNR] HARGA DEKAT RESISTANCE R1 ({r1['low']:.2f}-{r1['high']:.2f})")

        # 6. Alert Zona Broken / Invalidated Baru
        if "broken_ids" in self.last_state:
            new_broken = set(curr_state["broken_ids"]) - set(self.last_state["broken_ids"])
            for zone_id in sorted(new_broken & set(self.last_state.get("observed_ids") or [])):
                alerts.append(f"[SNR] ZONA {zone_id} DITEMBUS (BROKEN)")

        if "invalid_ids" in self.last_state:
            new_invalid = set(curr_state["invalid_ids"]) - set(self.last_state["invalid_ids"])
            for zone_id in sorted(new_invalid & set(self.last_state.get("observed_ids") or [])):
                alerts.append(f"[SNR] ZONA {zone_id} INVALID")

        # Peringatan kritis yang wajib tetap muncul
        if curr_state["signal"] == "INVALID":
            critical_warnings.append("[SNR CRITICAL] Setup/Zona fokus saat ini telah INVALIDATED.")

        self.last_state = curr_state
        _GLOBAL_LAST_SNR_STATE = dict(curr_state)
        return {
            "alerts": alerts,
            "critical_warnings": critical_warnings,
            "state": curr_state,
        }

