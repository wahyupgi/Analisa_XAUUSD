"""
Test Suite Final Audit — Single Immutable Snapshot, Zone Status & Count Reconciliation,
Confirmed Entry Zone Selection, Global vs Zone-Specific Sweep, Status Semantics,
dan Konsistensi Laporan End-to-End dengan Mock MT5 Tick.
"""

import os
import sys
import types
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.zone_scanner import (
    _entry_check,
    _zone_from_dict,
    calculate_distance_atr_ratio,
    calculate_zone_distance,
    format_scan_presentation,
    prioritize_radar_zones,
    scan_xauusd_zones,
)


def _make_ohlcv(n: int = 80, base: float = 3891.60, trend: float = -0.1, atr_val: float = 5.10) -> pd.DataFrame:
    times = pd.date_range(end=datetime(2026, 11, 13, 6, 44, tzinfo=timezone.utc), periods=n, freq="5min")
    closes = np.array([base + trend * i for i in range(n)], dtype=float)
    opens = closes - 0.5 * trend
    highs = np.maximum(opens, closes) + atr_val * 0.4
    lows = np.minimum(opens, closes) - atr_val * 0.4
    return pd.DataFrame(
        {
            "Time": times,
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": np.full(n, 100.0),
            "ATR14": np.full(n, atr_val),
        }
    )


def _make_zone(
    zone_id: str,
    tf: str,
    zone_type: str,
    lo: float,
    hi: float,
    score: float = 76.0,
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
        "touch_count": 0 if freshness == "FRESH" else 1,
        "status": "RADAR",
        "scoring_status": "VALID",
        "origin_bos": {"origin_valid": True},
        "fvg_confirmation": {"status": "UNTOUCHED"},
        "created_time": datetime.now(timezone.utc).isoformat(),
    }


class _DummyTick:
    def __init__(self, bid: float, ask: float, ts: int = 1794552240):
        self.bid = bid
        self.ask = ask
        self.time = ts


class TestFinalAuditSnapshotConsistency(unittest.TestCase):
    def test_01_single_immutable_snapshot_bid_ask_mid_spread_timestamp(self):
        """1. Bid, Ask, Mid, Spread, dan timestamp berasal dari satu snapshot MT5 tanpa panggilan tick berulang."""
        frames = {
            "H4": _make_ohlcv(80, base=3950.0, trend=-0.5, atr_val=8.0),
            "H1": _make_ohlcv(80, base=3930.0, trend=-0.4, atr_val=5.1),
            "M30": _make_ohlcv(80, base=3920.0, trend=-0.3, atr_val=4.0),
            "M15": _make_ohlcv(80, base=3910.0, trend=-0.2, atr_val=3.0),
            "M5": _make_ohlcv(80, base=3895.0, trend=-0.05, atr_val=2.0),
        }
        fake_mt5 = types.SimpleNamespace(symbol_info_tick=lambda sym: _DummyTick(3891.48, 3891.72))
        with patch.dict(sys.modules, {"MetaTrader5": fake_mt5}):
            scan = scan_xauusd_zones(frames=frames)

        self.assertEqual(scan["bid"], 3891.48)
        self.assertEqual(scan["ask"], 3891.72)
        self.assertEqual(scan["mid"], 3891.60)
        self.assertEqual(scan["spread"], 0.24)
        self.assertEqual(scan["snapshot_price"], 3891.60)
        self.assertEqual(scan["snapshot_status"], "OK")
        self.assertTrue(bool(scan["snapshot_time"]))

    def test_02_incomplete_snapshot_returns_data_incomplete(self):
        """1b. Snapshot MT5 yang tidak valid (ask < bid) mengembalikan DATA_INCOMPLETE."""
        frames = {
            "H4": _make_ohlcv(80, base=3950.0, trend=-0.5, atr_val=8.0),
            "H1": _make_ohlcv(80, base=3930.0, trend=-0.4, atr_val=5.1),
            "M5": _make_ohlcv(80, base=3895.0, trend=-0.05, atr_val=2.0),
        }
        fake_mt5 = types.SimpleNamespace(symbol_info_tick=lambda sym: _DummyTick(3891.72, 3891.48))
        with patch.dict(sys.modules, {"MetaTrader5": fake_mt5}):
            scan = scan_xauusd_zones(frames=frames)

        self.assertEqual(scan["snapshot_status"], "DATA_INCOMPLETE")
        self.assertIn("DATA_INCOMPLETE", scan["empty_reason"])

    def test_03_supply_demand_distance_and_atr_ratio_consistency(self):
        """2 & 3. Perhitungan jarak Supply/Demand dan ATR Ratio konsisten terhadap snapshot."""
        sup = calculate_zone_distance(3914.61, 3923.19, 3891.60, zone_type="SUPPLY", bid=3891.48, ask=3891.72, atr=5.10)
        dem = calculate_zone_distance(3871.94, 3875.11, 3891.60, zone_type="DEMAND", bid=3891.48, ask=3891.72, atr=5.10)

        self.assertAlmostEqual(sup.distance, 22.89, places=2)
        self.assertEqual(sup.reference_price_type, "ASK")
        self.assertEqual(sup.status, "ABOVE_PRICE")
        sup_ratio, _ = calculate_distance_atr_ratio(sup.distance, 5.10)
        self.assertAlmostEqual(sup.distance_atr_ratio, sup_ratio, places=2)

        self.assertAlmostEqual(dem.distance, 16.37, places=2)
        self.assertEqual(dem.reference_price_type, "BID")
        self.assertEqual(dem.status, "BELOW_PRICE")
        dem_ratio, _ = calculate_distance_atr_ratio(dem.distance, 5.10)
        self.assertAlmostEqual(dem.distance_atr_ratio, dem_ratio, places=2)

    def test_04_and_05_raw_candidates_reconciliation_and_structurally_valid_definition(self):
        """4 & 5. Rekonsiliasi Raw Candidates dan definisi Structurally Valid = Fresh Valid + Tested Valid (tanpa Mitigated)."""
        frames = {
            "H4": _make_ohlcv(80, base=3950.0, trend=-0.5, atr_val=8.0),
            "H1": _make_ohlcv(80, base=3930.0, trend=-0.4, atr_val=5.1),
            "M5": _make_ohlcv(80, base=3895.0, trend=-0.05, atr_val=2.0),
        }
        fake_mt5 = types.SimpleNamespace(symbol_info_tick=lambda sym: _DummyTick(3891.48, 3891.72))
        with patch.dict(sys.modules, {"MetaTrader5": fake_mt5}):
            scan = scan_xauusd_zones(frames=frames)

        ev = scan["zone_evaluation_summary"]
        total_parts = (
            ev["total_fresh"]
            + ev["total_tested"]
            + ev["total_mitigated"]
            + ev["total_invalidated"]
            + ev["structurally_rejected"]
            + ev["total_expired"]
        )
        self.assertEqual(total_parts, ev["raw_candidates"])
        self.assertEqual(ev["structurally_valid"], ev["total_fresh"] + ev["total_tested"])
        self.assertIn("Fresh Valid + Tested Valid", ev["structurally_valid_definition"])

    def test_06_confirmed_entry_monitoring_zone_separation(self):
        """6. Pemisahan eksplisit 5 istilah zona beserta alasan bila Nearest Radar Zone tidak dipilih sebagai Monitoring Zone."""
        z_primary = _make_zone("Z_M5_3953", "M5", "SUPPLY", 3953.33, 3955.71, score=85.0, grade="A+")
        z_nearest = _make_zone("Z_M15_3914", "M15+M5", "SUPPLY", 3914.61, 3923.19, score=68.0, grade="B")

        scan = {
            "symbol": "XAUUSD",
            "current_price": 3891.60,
            "snapshot_price": 3891.60,
            "bid": 3891.48,
            "ask": 3891.72,
            "spread": 0.24,
            "snapshot_time": "2026-11-13 14:10:00 WIB",
            "radar_zones": [z_primary, z_nearest],
            "active_zones": [z_primary, z_nearest],
            "htf_context": {"H4": "BEARISH", "H1": "BEARISH", "M30": "BEARISH"},
        }
        dec = {
            "status": "WAIT",
            "stages": {
                "zone_touch": {"status": "FAIL", "msg": "Belum tersentuh"},
                "liquidity_sweep": {"status": "NOT_CHECKED", "msg": "Belum diperiksa"},
                "structure_trigger": {"status": "NOT_CHECKED", "msg": "Belum diperiksa"},
                "displacement": {"status": "NOT_CHECKED", "msg": "Belum diperiksa"},
                "risk_reward": {"status": "NOT_CHECKED", "msg": "Belum diperiksa"},
            },
            "monitored_zone_identity": {
                "zone_id": "Z_M5_3953",
                "timeframe": "M5",
                "zone_type": "SUPPLY",
                "direction": "SELL",
                "proximal": 3953.33,
                "distal": 3955.71,
                "price_low": 3953.33,
                "price_high": 3955.71,
                "status": "RADAR",
            },
            "current_price": 3891.60,
        }
        out = format_scan_presentation(scan=scan, dec=dec, detailed=True)
        self.assertIn("Primary Directional Zone", out)
        self.assertIn("Nearest Radar Zone", out)
        self.assertIn("Confirmed Entry Monitoring Zone", out)
        self.assertIn("Best Pending Candidate", out)
        self.assertIn("Best Confirmed Candidate", out)

    def test_07_and_08_global_vs_zone_sweep_and_fail_vs_not_checked(self):
        """7 & 8. Pemisahan GLOBAL_M5_SWEEP_STATUS vs ZONE_SPECIFIC_SWEEP_STATUS dan FAIL vs NOT_CHECKED."""
        d5 = _make_ohlcv(60, base=3954.0, trend=0.0, atr_val=1.5)
        zone = _make_zone("Z_TOUCHED", "M5", "SUPPLY", 3953.0, 3956.0, score=80.0, grade="A")

        # Saat harga sudah menyentuh zona tetapi sweep tidak ditemukan -> liquidity_sweep = FAIL (bukan NOT_CHECKED)
        z_obj = _zone_from_dict(zone)
        res = _entry_check(z_obj, 3954.0, d5, require_sweep=True)
        self.assertEqual(res["stages"]["zone_touch"]["status"], "CONFIRMED")
        self.assertEqual(res["stages"]["liquidity_sweep"]["status"], "FAIL")

        # Saat Zone Touch belum terpenuhi -> Zone Touch = FAIL, sweep zona = NOT_CHECKED
        scan = {
            "symbol": "XAUUSD",
            "current_price": 3891.60,
            "snapshot_price": 3891.60,
            "bid": 3891.48,
            "ask": 3891.72,
            "spread": 0.24,
            "snapshot_time": "2026-11-13 14:10:00 WIB",
            "radar_zones": [zone],
            "active_zones": [zone],
            "htf_context": {"H4": "BEARISH", "H1": "BEARISH", "M30": "BEARISH"},
        }
        dec = {
            "status": "WAIT",
            "global_m5_sweep_status": "FAIL",
            "stages": {
                "zone_touch": {"status": "FAIL", "msg": "Harga belum menyentuh zona"},
                "liquidity_sweep": {"status": "NOT_CHECKED", "msg": "Belum diperiksa"},
                "structure_trigger": {"status": "NOT_CHECKED", "msg": "Belum diperiksa"},
                "displacement": {"status": "NOT_CHECKED", "msg": "Belum diperiksa"},
                "risk_reward": {"status": "NOT_CHECKED", "msg": "Belum diperiksa"},
            },
            "current_price": 3891.60,
        }
        out = format_scan_presentation(scan=scan, dec=dec, detailed=True)
        self.assertIn("GLOBAL_M5_SWEEP_STATUS     : FAIL", out)
        self.assertIn("ZONE_SPECIFIC_SWEEP_STATUS : NOT_CHECKED", out)
        self.assertIn("Zone Touch belum terpenuhi", out)

    def test_09_full_report_consistency_end_to_end(self):
        """9. Konsistensi zona, harga snapshot, dan jarak di seluruh bagian report."""
        z_primary = _make_zone("Z_M5_3953", "M5", "SUPPLY", 3953.33, 3955.71, score=75.3, grade="A")
        scan = {
            "symbol": "XAUUSD",
            "current_price": 3891.60,
            "snapshot_price": 3891.60,
            "bid": 3891.48,
            "ask": 3891.72,
            "spread": 0.24,
            "snapshot_time": "2026-11-13 14:10:00 WIB",
            "radar_zones": [z_primary],
            "active_zones": [z_primary],
            "htf_context": {"H4": "BEARISH", "H1": "BEARISH", "M30": "BEARISH"},
            "atr_diagnostics": {
                "atr_source_timeframe": "H1",
                "atr_period": 14,
                "atr_value": 5.10,
                "atr_candle_timestamp": "2026-11-13 07:00:00 UTC (MT5 Server)",
                "atr_calculation_status": "VALID",
                "atr_fallback_status": "NONE (Primary H1)",
                "atr_candle_mode": "CLOSED_CANDLE",
            },
        }
        out = format_scan_presentation(scan=scan, dec={"status": "WAIT", "stages": {}, "current_price": 3891.60})
        self.assertIn("Bid: 3891.48 | Ask: 3891.72 | Mid: 3891.60 | Spread: 0.24 pt", out)
        self.assertIn("61.61 pt", out)
        self.assertIn("M5 SUPPLY 3953.33 - 3955.71", out)


if __name__ == "__main__":
    unittest.main()
