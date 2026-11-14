"""
Test Suite Final Audit Status Semantics & Cross-Section Consistency.
"""

import os
import sys
import unittest
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.zone_scanner import (
    format_scan_presentation,
    prioritize_radar_zones,
)


def _make_zone(
    zone_id: str,
    tf: str,
    zone_type: str,
    lo: float,
    hi: float,
    score: float = 75.0,
    grade: str = "A",
    freshness: str = "FRESH",
    is_valid: bool = True,
    touch_count: int = 0,
    status: str = "RADAR",
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
        "is_valid": is_valid,
        "touch_count": touch_count,
        "status": status,
        "scoring_status": "VALID",
        "origin_bos": {"origin_valid": True},
        "fvg_confirmation": {"status": "UNTOUCHED"},
        "created_time": datetime.now(timezone.utc).isoformat(),
    }


class TestFinalStatusSemanticsAndAuthority(unittest.TestCase):
    def test_01_structurally_valid_but_far_zone_has_clear_audit_label_and_zero_authority(self):
        """H4 SUPPLY 4133.24-4148.65 (Skor 73.0 / 85.0) yang FAR_ZONE berstatus FAR_ZONE_EXCLUDED dan tidak jadi Radar/Entry."""
        z_far = _make_zone("Z_H4_FAR", "H4", "SUPPLY", 4133.24, 4148.65, score=73.0, grade="B")
        z_far_high = _make_zone("Z_H4_FAR_HI", "H4", "SUPPLY", 4200.00, 4210.00, score=85.0, grade="A+")
        z_active = _make_zone("Z_M15_ACT", "M15", "SUPPLY", 3914.61, 3920.00, score=72.0, grade="B")

        prio = prioritize_radar_zones(
            [z_far, z_far_high, z_active],
            3891.60,
            htf_bias="BEARISH",
            h1_structure="BEARISH",
            atr=5.16,
            bid=3891.48,
            ask=3891.72,
        )

        # Far Zone tidak boleh masuk Radar Qualified, Multi-Tier, Primary, Nearest, Pending, atau Confirmed
        rq_ids = {z["zone_id"] for z in prio["radar_qualified"]}
        self.assertNotIn("Z_H4_FAR", rq_ids)
        self.assertNotIn("Z_H4_FAR_HI", rq_ids)
        self.assertIn("Z_M15_ACT", rq_ids)

        self.assertEqual(prio["primary_directional_zone"]["zone_id"], "Z_M15_ACT")
        self.assertEqual(prio["nearest_radar_zone"]["zone_id"], "Z_M15_ACT")
        self.assertIsNone(prio["best_pending_candidate"])
        self.assertIsNone(prio["best_confirmed_candidate"])

        # Raw scoring wajib memisahkan STRUCTURALLY_VALID vs FAR_ZONE_EXCLUDED
        raw_map = {r["zone_id"]: r for r in prio["ranking_systems"]["raw_scoring_ranking"]}
        self.assertEqual(raw_map["Z_H4_FAR"]["structural_status"], "STRUCTURALLY_VALID")
        self.assertEqual(raw_map["Z_H4_FAR"]["radar_status"], "FAR_ZONE_EXCLUDED")
        self.assertEqual(raw_map["Z_H4_FAR"]["entry_status"], "NO_ENTRY_AUTHORITY")
        self.assertIn("FAR_ZONE_EXCLUDED", raw_map["Z_H4_FAR"]["validity_status"])
        self.assertFalse(raw_map["Z_H4_FAR"]["entry_authority"])

        self.assertEqual(raw_map["Z_M15_ACT"]["structural_status"], "STRUCTURALLY_VALID")
        self.assertEqual(raw_map["Z_M15_ACT"]["radar_status"], "RADAR_QUALIFIED")
        self.assertEqual(raw_map["Z_M15_ACT"]["entry_status"], "NO_ENTRY_AUTHORITY")

    def test_02_invalidated_and_mitigated_status_semantics(self):
        """Invalidated & Mitigated memiliki status eksplisit dan EXCLUDED dari Radar serta Entry."""
        z_inv = _make_zone("Z_INV", "H1", "SUPPLY", 3910.0, 3915.0, score=82.0, grade="A+", freshness="INVALIDATED", is_valid=False, status="INVALIDATED")
        z_mit = _make_zone("Z_MIT", "H1", "SUPPLY", 3920.0, 3925.0, score=80.0, grade="A", freshness="MITIGATED", touch_count=3, status="MITIGATED")

        prio = prioritize_radar_zones([z_inv, z_mit], 3891.60, htf_bias="BEARISH", h1_structure="BEARISH", atr=5.16)
        self.assertEqual(len(prio["radar_qualified"]), 0)

        raw_map = {r["zone_id"]: r for r in prio["ranking_systems"]["raw_scoring_ranking"]}
        self.assertEqual(raw_map["Z_INV"]["structural_status"], "INVALIDATED")
        self.assertEqual(raw_map["Z_INV"]["radar_status"], "EXCLUDED")
        self.assertEqual(raw_map["Z_MIT"]["structural_status"], "MITIGATED")
        self.assertEqual(raw_map["Z_MIT"]["radar_status"], "EXCLUDED")

    def test_03_multi_tier_and_cross_section_report_consistency(self):
        """M5 Demand Countertrend, M5 Supply, M15/M30/H1 Medium, H1/H4 Low, dan konsistensi di seluruh report."""
        z_m5_dem = _make_zone("Z_M5_DEM", "M5", "DEMAND", 3865.59, 3868.26, score=78.0, grade="A")
        z_m5_sup = _make_zone("Z_M5_SUP", "M5", "SUPPLY", 3901.00, 3903.00, score=76.0, grade="A")
        z_m15 = _make_zone("Z_M15_SUP", "M15", "SUPPLY", 3914.61, 3920.00, score=72.0, grade="B")
        z_h1 = _make_zone("Z_H1_SUP", "H1", "SUPPLY", 3925.00, 3930.00, score=82.0, grade="A+")
        z_far = _make_zone("Z_H4_FAR", "H4", "SUPPLY", 4133.24, 4148.65, score=73.0, grade="B")

        all_z = [z_m5_dem, z_m5_sup, z_m15, z_h1, z_far]
        prio = prioritize_radar_zones(all_z, 3891.60, htf_bias="BEARISH", h1_structure="BEARISH", atr=5.16, bid=3891.48, ask=3891.72)

        self.assertEqual([z["zone_id"] for z in prio["high_risk_zones"]], ["Z_M5_SUP"])
        self.assertEqual([z["zone_id"] for z in prio["medium_risk_zones"]], ["Z_M15_SUP"])
        self.assertEqual([z["zone_id"] for z in prio["low_risk_zones"]], ["Z_H1_SUP"])

        scan_mock = {
            "symbol": "XAUUSD",
            "current_price": 3891.60,
            "snapshot_price": 3891.60,
            "bid": 3891.48,
            "ask": 3891.72,
            "spread": 0.24,
            "snapshot_time": "2026-11-13 15:05:00 WIB",
            "radar_zones": all_z,
            "active_zones": [z_m5_dem, z_m5_sup, z_m15, z_h1],
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
        out = format_scan_presentation(scan=scan_mock, dec={"status": "WAIT", "stages": {}, "current_price": 3891.60, "radar_priority": prio}, detailed=True)
        self.assertIn("FAR_ZONE_EXCLUDED (Audit Only)", out)
        self.assertIn("4133.24-4148.65", out)


if __name__ == "__main__":
    unittest.main()

