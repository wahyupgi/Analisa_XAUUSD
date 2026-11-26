import unittest

from modules.snr.entry_validator import _is_live_active_snr_zone
from modules.trade_validator import select_candidate_zones


def _sup(lo=3574.67, hi=3578.05):
    return {"zone_id": "SNR_S", "zone_type": "SUPPORT", "timeframe": "M30", "low": lo, "high": hi,
            "score": 75.0, "grade": "B", "lifecycle_status": "ACTIVE"}


class SnrCandidateTests(unittest.TestCase):
    def test_small_edge_break_keeps_zone_active(self):
        self.assertTrue(_is_live_active_snr_zone(_sup(), "SUPPORT", 3574.40))

    def test_far_break_drops_zone(self):
        self.assertFalse(_is_live_active_snr_zone(_sup(), "SUPPORT", 3560.0))

    def test_dead_support_not_a_candidate(self):
        scan = {"symbol": "XAUUSDc", "current_price": 3560.0}
        snr = {"scan_result": {"symbol": "XAUUSDc", "supports": [_sup()]}, "decision": {"target_zone": _sup()}}
        self.assertEqual(select_candidate_zones(scan, {}, snr_result=snr), [])

    def test_live_support_is_a_candidate(self):
        scan = {"symbol": "XAUUSDc", "current_price": 3580.0}
        snr = {"scan_result": {"symbol": "XAUUSDc", "supports": [_sup()]}, "decision": {}}
        self.assertEqual(len(select_candidate_zones(scan, {}, snr_result=snr)), 1)


if __name__ == "__main__":
    unittest.main()
