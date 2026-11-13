"""
Test Suite Audit Multi-Tier Risk Classification Consistency & ATR Diagnostics.
"""

import os
import sys
import unittest
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.zone_scanner import (
    classify_zone_risk_tier,
    extract_atr_diagnostics,
    format_scan_presentation,
    prioritize_radar_zones,
)


def _make_zone(
    zone_id: str,
    tf: str,
    zone_type: str,
    lo: float,
    hi: float,
    score: float = 78.0,
    grade: str = "A",
    freshness: str = "FRESH",
) -> dict:
    return {
        "zone_id": zone_id,
        "timeframe": tf,
        "zone_type": zone_type,
        "price_low": lo,
        "price_high": hi,
        "proximal_edge": hi if zone_type == "DEMAND" else lo,
        "distal_edge": lo if zone_type == "DEMAND" else hi,
        "zone_width": round(abs(hi - lo), 2),
        "invalidation_price": lo if zone_type == "DEMAND" else hi,
        "score": score,
        "grade": grade,
        "freshness": freshness,
        "is_valid": True,
        "touch_count": 0,
        "status": "RADAR",
        "scoring_status": "VALID",
        "origin_bos": {"origin_valid": True},
        "fvg_confirmation": {"status": "UNTOUCHED"},
        "created_time": datetime.now(timezone.utc).isoformat(),
    }


class TestMultiTierM5ConsistencyAudit(unittest.TestCase):
    def test_01_m5_supply_and_m5_demand_countertrend_never_enter_medium_risk(self):
        """M5 Supply & M5 Demand Countertrend di luar jarak High Risk (>18 pt) tidak boleh masuk Medium Risk."""
        z_m5_sup = _make_zone("Z_M5_SUP", "M5", "SUPPLY", 3917.61, 3920.00, score=76.0, grade="A")
        z_m5_dem = _make_zone("Z_M5_DEM", "M5", "DEMAND", 3865.59, 3868.26, score=78.0, grade="A")

        t_sup = classify_zone_risk_tier(z_m5_sup, 3891.60, htf_bias="BEARISH", h1_structure="BEARISH", atr=5.16, bid=3891.48, ask=3891.72)
        t_dem = classify_zone_risk_tier(z_m5_dem, 3891.60, htf_bias="BEARISH", h1_structure="BEARISH", atr=5.16, bid=3891.48, ask=3891.72)

        self.assertEqual(t_sup, "UNCLASSIFIED")
        self.assertEqual(t_dem, "UNCLASSIFIED")
        self.assertIn("Medium Risk butuh TF M15/M30/H1", z_m5_sup["risk_tier_reason"])
        self.assertIn("Medium Risk butuh TF M15/M30/H1", z_m5_dem["risk_tier_reason"])

        # M5 dekat (<= 18 pt & <= 6x ATR) tetap boleh masuk HIGH_RISK
        z_m5_close = _make_zone("Z_M5_CLOSE", "M5", "SUPPLY", 3901.00, 3903.00, score=76.0, grade="A")
        t_close = classify_zone_risk_tier(z_m5_close, 3891.60, htf_bias="BEARISH", h1_structure="BEARISH", atr=5.16, bid=3891.48, ask=3891.72)
        self.assertEqual(t_close, "HIGH_RISK")

    def test_02_m15_m15m5_and_m30_supply_classification(self):
        """M15 Supply, M15+M5 Supply, dan M30 Supply memenuhi syarat TF Medium Risk pada jarak moderat."""
        z_m15 = _make_zone("Z_M15", "M15", "SUPPLY", 3914.61, 3920.00, score=72.0, grade="B")
        z_m15_m5 = _make_zone("Z_M15_M5", "M15+M5", "SUPPLY", 3914.61, 3923.19, score=72.0, grade="B")
        z_m30 = _make_zone("Z_M30", "M30", "SUPPLY", 3920.00, 3925.00, score=74.0, grade="B")
        z_m30_far = _make_zone("Z_M30_FAR", "M30", "SUPPLY", 3986.67, 3990.00, score=78.0, grade="A")

        self.assertEqual(classify_zone_risk_tier(z_m15, 3891.60, "BEARISH", "BEARISH", atr=5.16, bid=3891.48, ask=3891.72), "MEDIUM_RISK")
        self.assertEqual(classify_zone_risk_tier(z_m15_m5, 3891.60, "BEARISH", "BEARISH", atr=5.16, bid=3891.48, ask=3891.72), "MEDIUM_RISK")
        self.assertEqual(classify_zone_risk_tier(z_m30, 3891.60, "BEARISH", "BEARISH", atr=5.16, bid=3891.48, ask=3891.72), "MEDIUM_RISK")
        self.assertEqual(classify_zone_risk_tier(z_m30_far, 3891.60, "BEARISH", "BEARISH", atr=5.16, bid=3891.48, ask=3891.72), "FAR_ZONE")

    def test_03_h1_h4_supply_and_demand_classification(self):
        """H1/H4 pro-trend masuk LOW_RISK; H1 countertrend jarak moderat masuk MEDIUM_RISK."""
        z_h1_sup = _make_zone("Z_H1_SUP", "H1", "SUPPLY", 3914.61, 3923.19, score=80.0, grade="A")
        z_h4_sup = _make_zone("Z_H4_SUP", "H4", "SUPPLY", 3930.00, 3938.00, score=85.0, grade="A+")
        z_h1_dem_ct = _make_zone("Z_H1_DEM_CT", "H1", "DEMAND", 3860.00, 3865.00, score=75.0, grade="A")

        self.assertEqual(classify_zone_risk_tier(z_h1_sup, 3891.60, "BEARISH", "BEARISH", atr=5.16, bid=3891.48, ask=3891.72), "LOW_RISK")
        self.assertEqual(classify_zone_risk_tier(z_h4_sup, 3891.60, "BEARISH", "BEARISH", atr=5.16, bid=3891.48, ask=3891.72), "LOW_RISK")
        self.assertEqual(classify_zone_risk_tier(z_h1_dem_ct, 3891.60, "BEARISH", "BEARISH", atr=5.16, bid=3891.48, ask=3891.72), "MEDIUM_RISK")

    def test_04_distance_pass_fail_timeframe_and_timeframe_pass_fail_distance(self):
        """Zona lolos jarak tapi gagal TF (M5 @ 25pt) -> UNCLASSIFIED; lolos TF tapi gagal jarak (M15 @ 55pt) -> UNCLASSIFIED."""
        z_fail_tf = _make_zone("Z_FAIL_TF", "M5", "DEMAND", 3865.59, 3868.26, score=78.0, grade="A")
        z_fail_dist = _make_zone("Z_FAIL_DIST", "M15", "SUPPLY", 3947.00, 3950.00, score=78.0, grade="A")

        self.assertEqual(classify_zone_risk_tier(z_fail_tf, 3891.60, "BEARISH", "BEARISH", atr=5.16, bid=3891.48, ask=3891.72), "UNCLASSIFIED")
        self.assertEqual(classify_zone_risk_tier(z_fail_dist, 3891.60, "BEARISH", "BEARISH", atr=5.16, bid=3891.48, ask=3891.72), "UNCLASSIFIED")
        self.assertIn("Medium Risk butuh jarak", z_fail_dist["risk_tier_reason"])

    def test_05_atr_valid_unavailable_and_explicit_timezone(self):
        """ATR valid menampilkan zona waktu eksplisit; ATR unavailable -> DATA_INCOMPLETE tanpa fallback diam-diam."""
        idx_naive = pd.date_range(end=datetime(2026, 11, 13, 7, 0), periods=30, freq="1h")
        df_valid = pd.DataFrame({"ATR14": np.full(30, 5.16)}, index=idx_naive)
        diag_ok = extract_atr_diagnostics({"H1": df_valid}, preferred_tf="H1")
        self.assertEqual(diag_ok["atr_calculation_status"], "VALID")
        self.assertEqual(diag_ok["atr_fallback_status"], "NONE (Primary H1)")
        self.assertIn("UTC", diag_ok["atr_candle_timestamp"])

        diag_bad = extract_atr_diagnostics({"H1": pd.DataFrame({"ATR14": [np.nan] * 30}, index=idx_naive)}, preferred_tf="H1")
        self.assertEqual(diag_bad["atr_calculation_status"], "DATA_INCOMPLETE")
        self.assertIsNone(diag_bad["atr_value"])

    def test_06_multi_tier_and_outside_tier_explanation_consistency_and_raw_audit(self):
        """Tidak ada zona yang muncul di Multi-Tier sekaligus di Outside Tier; audit Raw Scoring konsisten."""
        z_m5_dem = _make_zone("Z_M5_DEM", "M5", "DEMAND", 3865.59, 3868.26, score=78.0, grade="A")
        z_h1_sup = _make_zone("Z_H1_SUP", "H1", "SUPPLY", 3914.61, 3923.19, score=80.0, grade="A")
        extra_zones = [_make_zone(f"Z_EX_{i}", "M15", "SUPPLY", 3925.0 + i * 2, 3927.0 + i * 2, score=70.0 - i, grade="B") for i in range(6)]
        all_z = [z_m5_dem, z_h1_sup] + extra_zones

        prio = prioritize_radar_zones(all_z, 3891.60, htf_bias="BEARISH", h1_structure="BEARISH", atr=5.16, bid=3891.48, ask=3891.72)
        mt_ids = {z["zone_id"] for z in (prio["high_risk_zones"] + prio["medium_risk_zones"] + prio["low_risk_zones"])}
        self.assertNotIn("Z_M5_DEM", mt_ids)

        recon = prio["zone_count_reconciliation"]
        self.assertEqual(recon["unclassified_tier_count"], 1)
        self.assertIn("M5 DEMAND (3865.59-3868.26)", recon["unclassified_tier_reasons"][0])
        self.assertIn("Medium Risk butuh TF M15/M30/H1", recon["unclassified_tier_reasons"][0])

        scan_mock = {
            "symbol": "XAUUSD",
            "current_price": 3891.60,
            "snapshot_price": 3891.60,
            "bid": 3891.48,
            "ask": 3891.72,
            "spread": 0.24,
            "snapshot_time": "2026-11-13 14:55:00 WIB",
            "radar_zones": all_z,
            "active_zones": all_z,
            "htf_context": {"H4": "BEARISH", "H1": "BEARISH", "M30": "BEARISH"},
            "radar_priority": prio,
            "atr_diagnostics": {
                "atr_source_timeframe": "H1",
                "atr_period": 14,
                "atr_value": 5.16,
                "atr_candle_timestamp": "2026-11-13 07:00:00 UTC (MT5 Server)",
                "atr_calculation_status": "VALID",
                "atr_fallback_status": "NONE (Primary H1)",
                "atr_candle_mode": "CLOSED_CANDLE",
            },
        }
        out = format_scan_presentation(scan=scan_mock, dec={"status": "WAIT", "stages": {}, "current_price": 3891.60, "radar_priority": prio})
        self.assertIn("Audit Raw Scoring: Masuk=8 | Dinilai=8 | Top 5 Tampil=5 | Di Luar Batas Tampilan=3", out)
        self.assertIn("Source: H1 | Period: 14 (CLOSED_CANDLE)", out)


if __name__ == "__main__":
    unittest.main()

