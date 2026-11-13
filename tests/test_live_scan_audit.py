# -*- coding: utf-8 -*-
"""
Regression tests for Live Scan Audit:
1. Far Zone Threshold (OR rule + exact triggered condition in exclusion reason, incl. M15 Supply 3971.76-3983.28)
2. Priority Composite separation from Technical Score Ranking
3. M5 Trigger State sequential dependencies (Zone Touch FAIL -> downstream NOT_CHECKED, WAIT)
4. Price Reference Consistency (Snapshot Mid, Radar Ask/Bid, Pending BUY/SELL LIMIT Bid/Ask)
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(__file__))

from modules.zone_scanner import (
    Zone,
    _entry_check,
    classify_zone_risk_tier,
    evaluate_trade_decision,
    prioritize_radar_zones,
)
from test_single_snapshot_flow import _make_frames, _make_zone


class TestLiveScanAudit(unittest.TestCase):
    def test_01_far_zone_or_rule_and_exact_reason_m15_supply(self):
        """M15 Supply 3971.76-3983.28 (dist 76.16 pt, ATR ~16.56 -> 4.6x) -> FAR_ZONE with exact pt reason."""
        z_m15 = _make_zone("Z_M15_FAR", "M15", "SUPPLY", 3971.76, 3983.28, score=84.0, grade="A")
        # Ask = 3895.60 -> 3971.76 - 3895.60 = 76.16 pt; ATR = 16.56 -> 4.6x ATR
        tier = classify_zone_risk_tier(
            z_m15, current_price=3895.40, htf_bias="BEARISH", h1_structure="BEARISH",
            atr=16.56, bid=3895.20, ask=3895.60
        )
        self.assertEqual(tier, "FAR_ZONE")
        self.assertIn("76.16 pt", z_m15["risk_tier_reason"])
        self.assertIn("4.6x ATR", z_m15["risk_tier_reason"])
        self.assertIn("jarak absolut maksimal (> 75.0 pt)", z_m15["risk_tier_reason"])
        self.assertNotIn("25.0x ATR", z_m15["risk_tier_reason"])

        # Case 2: Only ATR ratio > 25.0x (dist 60.0 pt <= 75.0, atr=2.0 -> 30.0x ATR)
        z_atr = _make_zone("Z_ATR_FAR", "H1", "SUPPLY", 3960.0, 3965.0, score=80.0, grade="A")
        tier_atr = classify_zone_risk_tier(
            z_atr, current_price=3900.0, htf_bias="BEARISH", h1_structure="BEARISH",
            atr=2.0, bid=3899.80, ask=3900.00
        )
        self.assertEqual(tier_atr, "FAR_ZONE")
        self.assertIn("ATR ratio maksimal (> 25.0x ATR)", z_atr["risk_tier_reason"])
        self.assertNotIn("75.0 pt", z_atr["risk_tier_reason"])

    def test_02_priority_composite_separates_technical_score_from_primary(self):
        """Skor teknikal tertinggi tidak otomatis menjadi Primary Directional Zone."""
        z_counter_high = _make_zone("Z_CTR_HIGH", "H4", "DEMAND", 3890.0, 3895.0, score=92.0, grade="A+")
        z_far_high = _make_zone("Z_FAR_HIGH", "H4", "SUPPLY", 3990.0, 3995.0, score=90.0, grade="A+")
        z_aligned_ok = _make_zone("Z_ALIGNED_OK", "H1", "SUPPLY", 3910.0, 3915.0, score=78.0, grade="A")

        prio = prioritize_radar_zones(
            radar_zones=[z_counter_high, z_far_high, z_aligned_ok],
            current_price=3900.0,
            htf_bias="BEARISH",
            h1_structure="BEARISH",
            atr=5.0,
            bid=3899.80,
            ask=3900.20,
            snapshot_id="SNAP_PRIO_AUDIT",
        )
        # Raw technical ranking places Z_CTR_HIGH (92.0) #1
        self.assertEqual(prio["ranking_systems"]["raw_scoring_ranking"][0]["zone_id"], "Z_CTR_HIGH")
        # Primary Directional Zone must be Z_ALIGNED_OK (not countertrend or far zone)
        self.assertEqual(prio["primary_directional_zone"]["zone_id"], "Z_ALIGNED_OK")
        by_id = {z["zone_id"]: z for z in prio["all_prioritized"]}
        self.assertGreater(
            prio["primary_directional_zone"]["priority_score"],
            by_id["Z_CTR_HIGH"]["priority_score"],
        )

    def test_03_m5_trigger_sequential_not_checked_when_untouched(self):
        """Jika Zone Touch FAIL, stage lanjutan tetap NOT_CHECKED (bukan FAIL) dan keputusan WAIT."""
        frames = _make_frames(base=3900.0)
        z_far_above = Zone(
            zone_id="Z_UNTOUCHED", symbol="XAUUSD", timeframe="H1", zone_type="SUPPLY",
            price_low=3950.0, price_high=3955.0, midpoint=3952.5, score=82.0, grade="A",
            origin_bos={"origin_valid": True}, fvg_confirmation={"status": "UNTOUCHED"},
            displacement_strength=2.0, freshness="FRESH", touch_count=0,
            liquidity_sweep_status="NONE", invalidation_price=3955.0,
            sl_recommended=3957.0, tp1_recommended=3930.0, tp2_recommended=3915.0,
            is_valid=True, rejection_reason="",
        )
        res = _entry_check(z_far_above, price=3900.0, df_m5=frames["M5"], spread=0.3)
        self.assertEqual(res["status"], "WAIT")
        self.assertEqual(res["stages"]["zone_touch"]["status"], "FAIL")
        self.assertEqual(res["stages"]["liquidity_sweep"]["status"], "NOT_CHECKED")
        self.assertEqual(res["stages"]["structure_trigger"]["status"], "NOT_CHECKED")
        self.assertEqual(res["stages"]["displacement"]["status"], "NOT_CHECKED")
        self.assertEqual(res["stages"]["risk_reward"]["status"], "NOT_CHECKED")

        # Verify evaluate_trade_decision puts downstream stages in conditions_not_checked, not conditions_fail
        scan_mock = {
            "data_ok": True,
            "snapshot_id": "SNAP_M5_AUDIT",
            "snapshot_price": 3900.0,
            "current_price": 3900.0,
            "bid": 3899.80,
            "ask": 3900.20,
            "mid": 3900.0,
            "spread": 0.40,
            "snapshot_status": "OK",
            "htf_context": {"H4": "BEARISH", "H1": "BEARISH", "has_conflict": False},
            "active_zones": [z_far_above.to_dict()],
            "radar_zones": [z_far_above.to_dict()],
            "snapshot": {"snapshot_id": "SNAP_M5_AUDIT", "snapshot_status": "OK", "frames": frames, "snapshot_price": 3900.0, "bid": 3899.80, "ask": 3900.20, "mid": 3900.0, "spread": 0.40},
        }
        dec = evaluate_trade_decision(scan_result=scan_mock, live=False)
        self.assertEqual(dec["status"], "WAIT")
        self.assertEqual(dec["monitored_zone_identity"]["zone_id"], "Z_UNTOUCHED")
        evs = dec["entry_validation_summary"]
        self.assertTrue(any("Zone Touch [FAIL" in c for c in evs["conditions_fail"]))
        self.assertTrue(any("Liquidity Sweep M5 [NOT_CHECKED]" in c for c in evs["conditions_not_checked"]))
        self.assertFalse(any("Liquidity Sweep" in c for c in evs["conditions_fail"]))

    def test_04_price_reference_consistency_ask_for_supply_bid_for_demand(self):
        """Radar distance uses Ask for Supply and Bid for Demand explicitly."""
        z_sup = _make_zone("Z_SUP_REF", "H1", "SUPPLY", 3910.0, 3915.0, score=80.0, grade="A")
        z_dem = _make_zone("Z_DEM_REF", "H1", "DEMAND", 3885.0, 3890.0, score=80.0, grade="A")
        prio = prioritize_radar_zones(
            radar_zones=[z_sup, z_dem],
            current_price=3900.0,
            htf_bias="BEARISH",
            h1_structure="BEARISH",
            atr=5.0,
            bid=3899.80,
            ask=3900.20,
            snapshot_id="SNAP_REF_TEST",
        )
        by_id = {z["zone_id"]: z for z in prio["all_prioritized"]}
        self.assertEqual(by_id["Z_SUP_REF"]["reference_price_type"], "ASK")
        self.assertAlmostEqual(by_id["Z_SUP_REF"]["reference_price"], 3900.20, places=2)
        self.assertAlmostEqual(by_id["Z_SUP_REF"]["distance"], 9.80, places=2)

        self.assertEqual(by_id["Z_DEM_REF"]["reference_price_type"], "BID")
        self.assertAlmostEqual(by_id["Z_DEM_REF"]["reference_price"], 3899.80, places=2)
        self.assertAlmostEqual(by_id["Z_DEM_REF"]["distance"], 9.80, places=2)


if __name__ == "__main__":
    unittest.main()
