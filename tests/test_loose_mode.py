import os
import unittest
from unittest import mock

from modules.config import get_auto_trading_config
from modules.dynamic_lot_manager import select_target_lot
from modules.trade_validator import evaluate_directional_alignment, validate_zone_quality_layer


class LooseModeTests(unittest.TestCase):
    def _loose_env(self):
        return mock.patch.dict(os.environ, {"AUTO_LOOSE_MODE": "true"})

    def test_default_off_keeps_strict_rules(self):
        with mock.patch.dict(os.environ, {"AUTO_LOOSE_MODE": "false"}):
            cfg = get_auto_trading_config()
        self.assertFalse(cfg["AUTO_LOOSE_MODE"])
        self.assertEqual(cfg["MAX_ACTIVE_POSITIONS"], 1)
        self.assertEqual(cfg["MIN_ZONE_SCORE"], 70.0)

    def test_loose_overrides(self):
        with self._loose_env():
            cfg = get_auto_trading_config()
        self.assertTrue(cfg["AUTO_LOOSE_MODE"])
        self.assertEqual(cfg["MAX_ACTIVE_POSITIONS"], 5)
        self.assertEqual(cfg["MAX_ACTIVE_PENDING_ORDERS"], 5)
        self.assertEqual(cfg["TP1_RR"], 2.0)
        self.assertEqual(cfg["MIN_RR"], 2.0)
        self.assertTrue(cfg["REQUIRE_STOP_LOSS"])
        self.assertTrue(cfg["REQUIRE_TAKE_PROFIT"])

    def test_low_grade_zone_gets_min_lot_and_not_blocked(self):
        zone = {"is_valid": True, "status": "ACTIVE", "score": 55.0, "grade": "C", "timeframe": "M30"}
        with self._loose_env():
            cfg = get_auto_trading_config()
        self.assertEqual(select_target_lot(zone, "ADAPTIVE_ENTRY", cfg=cfg), 0.01)
        strict = dict(cfg, AUTO_LOOSE_MODE=False, MIN_ZONE_SCORE=70.0)
        self.assertEqual(select_target_lot(zone, "ADAPTIVE_ENTRY", cfg=strict), 0.0)

    def test_htf_conflict_ignored_only_in_loose(self):
        with self._loose_env():
            cfg = get_auto_trading_config()
        ok, _, _ = evaluate_directional_alignment("SELL", "ADAPTIVE_ENTRY", "BEARISH", "BULLISH", "CONFLICT", True, cfg=cfg)
        self.assertTrue(ok)
        strict = dict(cfg, AUTO_LOOSE_MODE=False)
        ok2, _, _ = evaluate_directional_alignment("SELL", "ADAPTIVE_ENTRY", "BEARISH", "BULLISH", "CONFLICT", True, cfg=strict)
        self.assertFalse(ok2)


if __name__ == "__main__":
    unittest.main()
