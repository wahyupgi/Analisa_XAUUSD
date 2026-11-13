"""
Test Suite Audit Priority Composite Single Source of Truth & Far Zone Reconciliation.
"""

import os
import sys
import unittest
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.zone_scanner import (
    evaluate_trade_decision,
    format_scan_presentation,
    prioritize_radar_zones,
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
    score: float = 75.3,
    grade: str = "A",
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
        "freshness": "FRESH",
        "is_valid": True,
        "touch_count": 0,
        "status": "RADAR",
        "scoring_status": "VALID",
        "origin_bos": {"origin_valid": True},
        "fvg_confirmation": {"status": "UNTOUCHED"},
        "created_time": datetime.now(timezone.utc).isoformat(),
    }


class TestPriorityCompositeAndFarZone(unittest.TestCase):
    def test_01_priority_composite_single_source_of_truth_across_calls_and_report(self):
        """Nilai priority_score satu zona tetap identik meski dipanggil ulang dengan ATR berbeda atau dirender ulang."""
        z_primary = _make_zone("Z_PRI", "M15", "SUPPLY", 3914.61, 3920.00, score=75.3, grade="A")
        z_alt = _make_zone("Z_ALT", "M15", "SUPPLY", 3925.00, 3930.00, score=76.0, grade="A")

        # Hitung pertama dengan H1 ATR = 5.10
        prio_1 = prioritize_radar_zones(
            radar_zones=[z_primary, z_alt],
            current_price=3891.60,
            htf_bias="BEARISH",
            h1_structure="BEARISH",
            atr=5.10,
            bid=3891.48,
            ask=3891.72,
        )
        p_score_1 = prio_1["primary_directional_zone"]["priority_score"]
        a_score_1 = prio_1["alternative_zone"]["priority_score"]

        # Panggil ulang pada objek hasil prio_1 dengan ATR berbeda (3.00) -> harus terkunci sama!
        prio_2 = prioritize_radar_zones(
            radar_zones=prio_1["all_prioritized"],
            current_price=3891.60,
            htf_bias="BEARISH",
            h1_structure="BEARISH",
            atr=3.00,
            bid=3891.48,
            ask=3891.72,
        )
        self.assertEqual(prio_2["primary_directional_zone"]["priority_score"], p_score_1)
        self.assertEqual(prio_2["alternative_zone"]["priority_score"], a_score_1)

        # Pastikan alasan tidak diprioritaskan memakai nilai komposit yang sama
        unprio_txt = " ".join(prio_1["zone_scoring_summary"]["unprioritized_reasons"])
        self.assertIn(f"{p_score_1:.1f}", unprio_txt)
        self.assertIn(f"{a_score_1:.1f}", unprio_txt)

    def test_02_decide_uses_h1_atr_matching_scan(self):
        """evaluate_trade_decision menggunakan H1 ATR (sama seperti scan_xauusd_zones), bukan M5 ATR."""
        frames = {
            "H4": _make_ohlcv(80, base=3950.0, trend=-0.5, atr_val=8.0),
            "H1": _make_ohlcv(80, base=3930.0, trend=-0.4, atr_val=5.1),
            "M30": _make_ohlcv(80, base=3920.0, trend=-0.3, atr_val=4.0),
            "M15": _make_ohlcv(80, base=3910.0, trend=-0.2, atr_val=3.0),
            "M5": _make_ohlcv(80, base=3895.0, trend=-0.05, atr_val=1.5),
        }
        dec = evaluate_trade_decision(frames=frames, current_price=3891.60, spread=0.24, live=False)
        rp = dec.get("radar_priority")
        if rp and rp.get("all_prioritized"):
            for z in rp["all_prioritized"]:
                self.assertAlmostEqual(float(z.get("atr_value", 0.0)), 5.1, places=1)

    def test_03_far_zone_5_rejected_displays_all_5_and_excludes_from_active(self):
        """5 Far Zone ditolak -> count=5, 5 detail ditampilkan (5/5), tanpa duplikasi, tidak masuk Radar/Multi-Tier."""
        z_near = _make_zone("Z_NEAR", "H1", "SUPPLY", 3914.61, 3923.19, score=80.0, grade="A")
        far_list = [
            _make_zone(f"Z_FAR_{i}", "H4", "SUPPLY", 4050.0 + i * 20, 4055.0 + i * 20, score=85.0, grade="A+")
            for i in range(1, 6)
        ]
        # Tambahkan duplikat Z_FAR_1 untuk menguji deduplikasi
        all_zones = [z_near] + far_list + [dict(far_list[0])]

        prio = prioritize_radar_zones(
            radar_zones=all_zones,
            current_price=3891.60,
            htf_bias="BEARISH",
            h1_structure="BEARISH",
            atr=5.10,
            bid=3891.48,
            ask=3891.72,
        )

        recon = prio["zone_count_reconciliation"]
        self.assertEqual(recon["far_zones_count"], 5)
        self.assertEqual(len(recon["far_zones_reasons"]), 5)

        # Tidak ada Far Zone yang masuk Radar Qualified atau Multi-Tier
        rq_ids = {z["zone_id"] for z in prio["radar_qualified"]}
        mt_ids = {
            z["zone_id"]
            for z in (prio["high_risk_zones"] + prio["medium_risk_zones"] + prio["low_risk_zones"])
        }
        for i in range(1, 6):
            self.assertNotIn(f"Z_FAR_{i}", rq_ids)
            self.assertNotIn(f"Z_FAR_{i}", mt_ids)

        scan_mock = {
            "symbol": "XAUUSD",
            "current_price": 3891.60,
            "snapshot_price": 3891.60,
            "bid": 3891.48,
            "ask": 3891.72,
            "spread": 0.24,
            "snapshot_time": "2026-11-13 14:25:00 WIB",
            "radar_zones": all_zones,
            "active_zones": [z_near],
            "htf_context": {"H4": "BEARISH", "H1": "BEARISH", "M30": "BEARISH"},
            "radar_priority": prio,
        }
        out = format_scan_presentation(scan=scan_mock, dec={"status": "WAIT", "stages": {}, "current_price": 3891.60, "radar_priority": prio})
        self.assertIn("Ditolak Evaluasi Lanjutan (Far Zone)     : 5 (Detail Displayed: 5/5)", out)
        for i in range(1, 6):
            self.assertIn(f"{4050.0 + i * 20:.2f}-{4055.0 + i * 20:.2f}", out)

    def test_04_far_zone_hidden_count_shown_when_details_truncated(self):
        """Jika detail list < total far_zones_count, format report menampilkan Detail Displayed dan Hidden Details."""
        z_near = _make_zone("Z_NEAR", "H1", "SUPPLY", 3914.61, 3923.19, score=80.0, grade="A")
        prio = prioritize_radar_zones(
            radar_zones=[z_near],
            current_price=3891.60,
            htf_bias="BEARISH",
            h1_structure="BEARISH",
            atr=5.10,
            bid=3891.48,
            ask=3891.72,
        )
        prio["zone_count_reconciliation"]["far_zones_count"] = 5
        prio["zone_count_reconciliation"]["far_zones_reasons"] = ["Far 1", "Far 2", "Far 3"]

        scan_mock = {
            "symbol": "XAUUSD",
            "current_price": 3891.60,
            "snapshot_price": 3891.60,
            "bid": 3891.48,
            "ask": 3891.72,
            "spread": 0.24,
            "snapshot_time": "2026-11-13 14:25:00 WIB",
            "radar_zones": [z_near],
            "active_zones": [z_near],
            "htf_context": {"H4": "BEARISH", "H1": "BEARISH", "M30": "BEARISH"},
            "radar_priority": prio,
        }
        out = format_scan_presentation(scan=scan_mock, dec={"status": "WAIT", "stages": {}, "current_price": 3891.60, "radar_priority": prio})
        self.assertIn("Ditolak Evaluasi Lanjutan (Far Zone)     : 5 (Detail Displayed: 3/5 | Hidden Details: 2)", out)


if __name__ == "__main__":
    unittest.main()

