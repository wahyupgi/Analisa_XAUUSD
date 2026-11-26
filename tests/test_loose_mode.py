import os
import unittest
from unittest import mock

from modules.adaptive_entry_engine import evaluate_entry_paths_for_zone
from modules.config import compute_zone_limit_entry_price, get_auto_trading_config, resolve_zone_limit_depth
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

    def test_limit_order_uses_midzone_not_front_edge(self):
        # Supply 3600-3608 -> tengah zona 3604.00 (bukan awal zona 3600.00)
        self.assertEqual(compute_zone_limit_entry_price(3600.0, 3608.0, demand=False), 3604.0)
        # Demand 3550-3560 -> tengah zona 3555.00 (bukan awal zona 3560.00)
        self.assertEqual(compute_zone_limit_entry_price(3550.0, 3560.0, demand=True), 3555.0)
        # Bahkan bila depth diset 0.0 tetap di-clamp minimal 25% masuk zona (tidak pernah di awal zona)
        self.assertEqual(compute_zone_limit_entry_price(3600.0, 3608.0, demand=False, depth=0.0), 3602.0)
        self.assertEqual(compute_zone_limit_entry_price(3550.0, 3560.0, demand=True, depth=0.0), 3557.5)

        supply_z = {
            "zone_id": "SNR_M30_RES_3600_3608",
            "symbol": "XAUUSDc",
            "timeframe": "M30",
            "zone_type": "SUPPLY",
            "price_low": 3600.0,
            "price_high": 3608.0,
            "midpoint": 3604.0,
            "score": 68.0,
            "grade": "C",
            "freshness": "FRESH",
            "touch_count": 0,
            "invalidation_price": 3608.0,
            "is_valid": True,
            "status": "ACTIVE",
        }
        with self._loose_env():
            cfg = get_auto_trading_config()
        # 1) Harga belum menyentuh zona (3590) -> SELL LIMIT di tengah zona 3604.00, bukan 3600.00
        res_far = evaluate_entry_paths_for_zone(
            supply_z, frames=None, current_price=3590.0, bid=3589.9, ask=3590.1, spread=0.2,
            htf_bias="BEARISH", h1_bias="BULLISH", m30_status="CONFLICT", has_htf_conflict=True, cfg=cfg,
        )
        self.assertEqual(res_far["order_kind"], "LIMIT")
        self.assertEqual(res_far["order_type"], "SELL LIMIT")
        self.assertEqual(res_far["levels"]["entry_price"], 3604.0)
        self.assertGreater(res_far["levels"]["sl"], 3608.0)

        # 2) Harga baru menyentuh awal zona tipis (3600.40 < 3604.00) -> tetap pasang SELL LIMIT di 3604.00, bukan SELL MARKET di awal zona!
        res_edge = evaluate_entry_paths_for_zone(
            supply_z, frames=None, current_price=3600.5, bid=3600.4, ask=3600.6, spread=0.2,
            htf_bias="BEARISH", h1_bias="BULLISH", m30_status="CONFLICT", has_htf_conflict=True, cfg=cfg,
        )
        self.assertEqual(res_edge["order_kind"], "LIMIT")
        self.assertEqual(res_edge["order_type"], "SELL LIMIT")
        self.assertEqual(res_edge["levels"]["entry_price"], 3604.0)

        # 3) Zona Grade B normal (bukan HIGH_RISK) pada 3600-3608 -> ambil 25% masuk zona = 3602.00!
        supply_b = dict(supply_z, zone_id="SNR_M30_RES_B", score=72.0, grade="B", risk_tier="MEDIUM_RISK")
        self.assertEqual(resolve_zone_limit_depth(zone=supply_b, cfg=cfg), 0.25)
        res_b = evaluate_entry_paths_for_zone(
            supply_b, frames=None, current_price=3590.0, bid=3589.9, ask=3590.1, spread=0.2,
            htf_bias="BEARISH", h1_bias="BULLISH", m30_status="CONFLICT", has_htf_conflict=True, cfg=cfg,
        )
        self.assertEqual(res_b["order_kind"], "LIMIT")
        self.assertEqual(res_b["levels"]["entry_price"], 3602.0)

        # 4) Zona Grade B tapi berstatus HIGH_RISK -> tetap ambil 50% tengah zona = 3604.00!
        supply_b_hr = dict(supply_b, zone_id="SNR_M30_RES_B_HR", risk_tier="HIGH_RISK")
        self.assertEqual(resolve_zone_limit_depth(zone=supply_b_hr, cfg=cfg), 0.50)
        res_b_hr = evaluate_entry_paths_for_zone(
            supply_b_hr, frames=None, current_price=3590.0, bid=3589.9, ask=3590.1, spread=0.2,
            htf_bias="BEARISH", h1_bias="BULLISH", m30_status="CONFLICT", has_htf_conflict=True, cfg=cfg,
        )
        self.assertEqual(res_b_hr["levels"]["entry_price"], 3604.0)

        # 5) Demand Grade B normal pada 3550-3560 -> BUY LIMIT 25% masuk dari atas (3560 - 2.50 = 3557.50)
        demand_b = dict(
            supply_b, zone_id="SNR_H1_SUP_B", zone_type="DEMAND",
            price_low=3550.0, price_high=3560.0, midpoint=3555.0, invalidation_price=3550.0,
        )
        res_dem_b = evaluate_entry_paths_for_zone(
            demand_b, frames=None, current_price=3570.0, bid=3569.9, ask=3570.1, spread=0.2,
            htf_bias="BULLISH", h1_bias="BEARISH", m30_status="CONFLICT", has_htf_conflict=True, cfg=cfg,
        )
        self.assertEqual(res_dem_b["order_type"], "BUY LIMIT")
        self.assertEqual(res_dem_b["levels"]["entry_price"], 3557.5)


if __name__ == "__main__":
    unittest.main()
