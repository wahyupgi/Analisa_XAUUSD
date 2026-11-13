"""
Test Suite Audit ATR Consistency & Multi-Tier Classification.
"""

import os
import sys
import unittest
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.mt5_conn import tambah_indikator
from modules.zone_scanner import (
    calculate_distance_atr_ratio,
    classify_zone_risk_tier,
    evaluate_trade_decision,
    extract_atr_diagnostics,
    format_scan_presentation,
    prioritize_radar_zones,
    scan_xauusd_zones,
)


def _make_ohlcv_with_atr(n: int = 80, base: float = 3891.60, trend: float = -0.1, atr_val: float = 5.16, freq: str = "1h") -> pd.DataFrame:
    times = pd.date_range(end=datetime(2026, 11, 13, 7, 0, tzinfo=timezone.utc), periods=n, freq=freq)
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


def _make_zone(zone_id: str, tf: str, zone_type: str, lo: float, hi: float, score: float = 80.0, grade: str = "A") -> dict:
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
        "freshness": "FRESH",
        "is_valid": True,
        "touch_count": 0,
        "status": "RADAR",
        "scoring_status": "VALID",
        "origin_bos": {"origin_valid": True},
        "fvg_confirmation": {"status": "UNTOUCHED"},
        "created_time": datetime.now(timezone.utc).isoformat(),
    }


class TestAtrConsistencyAndMultiTier(unittest.TestCase):
    def test_01_atr_calculation_determinism_same_candles(self):
        """1. tambah_indikator menghasilkan ATR14 yang deterministik untuk data candle yang sama."""
        df = _make_ohlcv_with_atr(50, atr_val=5.0).drop(columns=["ATR14"])
        df1 = tambah_indikator(df.copy())
        df2 = tambah_indikator(df.copy())
        self.assertAlmostEqual(float(df1["ATR14"].iloc[-1]), float(df2["ATR14"].iloc[-1]), places=6)

    def test_02_and_03_atr_timeframe_period_and_candle_timestamp_consistent(self):
        """2 & 3. Timeframe ATR (H1), Period (14), dan Candle Timestamp konsisten di scan, decision, dan report."""
        frames = {
            "H4": _make_ohlcv_with_atr(80, base=3950.0, trend=-0.5, atr_val=17.30, freq="4h"),
            "H1": _make_ohlcv_with_atr(80, base=3930.0, trend=-0.4, atr_val=5.16, freq="1h"),
            "M30": _make_ohlcv_with_atr(80, base=3920.0, trend=-0.3, atr_val=3.80, freq="30min"),
            "M15": _make_ohlcv_with_atr(80, base=3910.0, trend=-0.2, atr_val=2.50, freq="15min"),
            "M5": _make_ohlcv_with_atr(80, base=3895.0, trend=-0.05, atr_val=1.40, freq="5min"),
        }
        scan = scan_xauusd_zones(frames=frames, current_price=3891.60)
        diag = scan["atr_diagnostics"]
        self.assertEqual(diag["atr_source_timeframe"], "H1")
        self.assertEqual(diag["atr_period"], 14)
        self.assertAlmostEqual(diag["atr_value"], 5.16, places=2)
        self.assertEqual(diag["atr_calculation_status"], "VALID")
        self.assertEqual(diag["atr_fallback_status"], "NONE (Primary H1)")
        self.assertIn("UTC", diag["atr_candle_timestamp"])

        dec = evaluate_trade_decision(frames=frames, current_price=3891.60, spread=0.24, live=False)
        self.assertAlmostEqual(dec["atr_diagnostics"]["atr_value"], 5.16, places=2)
        rendered = format_scan_presentation(scan=scan, dec=dec)
        self.assertIn("ATR Diagnostics", rendered)
        self.assertIn("Source: H1 | Period: 14", rendered)
        self.assertIn("5.16 pt", rendered)

    def test_04_atr_ratio_for_supply_and_demand(self):
        """4. Perhitungan ATR Ratio Supply dan Demand sesuai formula distance / atr."""
        r_sup, _ = calculate_distance_atr_ratio(19.60, 5.16)
        r_dem, _ = calculate_distance_atr_ratio(26.51, 5.16)
        self.assertAlmostEqual(r_sup, round(19.60 / 5.16, 2), places=2)
        self.assertAlmostEqual(r_dem, round(26.51 / 5.16, 2), places=2)

    def test_05_and_06_atr_unavailable_returns_data_incomplete_no_silent_fallback(self):
        """5 & 6. Jika H1 ATR tidak tersedia/invalid, status = DATA_INCOMPLETE tanpa fallback diam-diam ke 2.0."""
        df_h1_bad = _make_ohlcv_with_atr(80, atr_val=5.16, freq="1h").drop(columns=["ATR14"])
        diag = extract_atr_diagnostics({"H1": df_h1_bad}, preferred_tf="H1")
        self.assertEqual(diag["atr_calculation_status"], "DATA_INCOMPLETE")
        self.assertIsNone(diag["atr_value"])
        self.assertIn("UNAVAILABLE", diag["atr_fallback_status"])

        z = _make_zone("Z_TEST", "H1", "SUPPLY", 3914.61, 3923.19, score=80.0, grade="A")
        tier = classify_zone_risk_tier(z, 3891.60, htf_bias="BEARISH", h1_structure="BEARISH", atr=None)
        self.assertEqual(tier, "DATA_INCOMPLETE")

    def test_07_and_08_atr_change_between_scans_explains_ratio_and_tier_shift(self):
        """7 & 8. Perubahan ATR H1 (5.16 -> 17.30 seperti 14:25 vs 14:43 WIB) mengubah ATR Ratio dan klasifikasi secara konsisten."""
        z_m15 = _make_zone("Z_M15", "M15", "SUPPLY", 3911.20, 3915.00, score=76.0, grade="A")
        z_m30 = _make_zone("Z_M30", "M30", "SUPPLY", 3986.67, 3990.00, score=78.0, grade="A")

        # Scan 1 (ATR 5.16) -> M15 @ 19.60 pt (3.80x ATR -> MEDIUM_RISK), M30 @ 95.07 pt (>75 pt -> FAR_ZONE)
        prio_1 = prioritize_radar_zones([z_m15, z_m30], 3891.60, htf_bias="BEARISH", h1_structure="BEARISH", atr=5.16)
        by_id_1 = {z["zone_id"]: z for z in prio_1["all_prioritized"]}
        self.assertAlmostEqual(by_id_1["Z_M15"]["distance_atr_ratio"], round(19.60 / 5.16, 2), places=2)
        self.assertEqual(by_id_1["Z_M15"]["risk_tier"], "MEDIUM_RISK")
        self.assertEqual(by_id_1["Z_M30"]["risk_tier"], "FAR_ZONE")

        # Scan 2 (ATR 17.30) -> M15 @ 19.60 pt (1.13x ATR), M30 @ 95.07 pt (5.50x ATR, masih >75 pt -> FAR_ZONE)
        prio_2 = prioritize_radar_zones([z_m15, z_m30], 3891.60, htf_bias="BEARISH", h1_structure="BEARISH", atr=17.30)
        by_id_2 = {z["zone_id"]: z for z in prio_2["all_prioritized"]}
        self.assertAlmostEqual(by_id_2["Z_M15"]["distance_atr_ratio"], round(19.60 / 17.30, 2), places=2)
        self.assertAlmostEqual(by_id_2["Z_M30"]["distance_atr_ratio"], round(95.07 / 17.30, 2), places=2)

    def test_09_identical_inputs_produce_identical_atr_ratio(self):
        """9. Input identik menghasilkan ATR Ratio dan tier yang identik pada pemanggilan berulang."""
        z = _make_zone("Z_EQ", "H1", "SUPPLY", 3914.61, 3923.19, score=80.0, grade="A")
        p1 = prioritize_radar_zones([z], 3891.60, htf_bias="BEARISH", h1_structure="BEARISH", atr=5.16, bid=3891.48, ask=3891.72)
        p2 = prioritize_radar_zones([z], 3891.60, htf_bias="BEARISH", h1_structure="BEARISH", atr=5.16, bid=3891.48, ask=3891.72)
        self.assertEqual(p1["all_prioritized"][0]["distance_atr_ratio"], p2["all_prioritized"][0]["distance_atr_ratio"])
        self.assertEqual(p1["all_prioritized"][0]["risk_tier"], p2["all_prioritized"][0]["risk_tier"])


if __name__ == "__main__":
    unittest.main()

