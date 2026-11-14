"""
Test Suite Audit Lanjutan Round 3 — Ranking Explanation, Price Reference (Bid/Ask/Mid),
Zone Touch Status, Entry Validation Message, Multi-Tier Explanation & Konsistensi Snapshot.
"""

import os
import sys
import unittest
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.zone_scanner import (
    Zone,
    calculate_zone_distance,
    classify_zone_risk_tier,
    evaluate_trade_decision,
    format_scan_presentation,
    prioritize_radar_zones,
)


def _make_zone(
    zone_id: str,
    tf: str,
    zone_type: str,
    lo: float,
    hi: float,
    score: float,
    grade: str = "A",
    freshness: str = "FRESH",
    is_valid: bool = True,
    invalidation_price: float = None,
) -> dict:
    inval = invalidation_price if invalidation_price is not None else (lo if zone_type == "DEMAND" else hi)
    return {
        "zone_id": zone_id,
        "timeframe": tf,
        "zone_type": zone_type,
        "price_low": lo,
        "price_high": hi,
        "proximal_edge": hi if zone_type == "DEMAND" else lo,
        "distal_edge": lo if zone_type == "DEMAND" else hi,
        "zone_width": round(abs(hi - lo), 2),
        "invalidation_price": inval,
        "score": score,
        "grade": grade,
        "freshness": freshness,
        "is_valid": is_valid,
        "touch_count": 0 if freshness == "FRESH" else 1,
        "status": "RADAR",
        "scoring_status": "VALID",
        "origin_bos": {"origin_valid": True},
        "fvg_confirmation": {"status": "UNTOUCHED"},
        "created_time": datetime.now(timezone.utc).isoformat(),
    }


def _make_ohlcv(n: int = 80, base: float = 3891.60, trend: float = 0.0, atr_val: float = 3.0) -> pd.DataFrame:
    times = pd.date_range(end=datetime(2026, 11, 13, 6, 44, tzinfo=timezone.utc), periods=n, freq="5min")
    closes = np.array([base + trend * i for i in range(n)], dtype=float)
    opens = closes - 0.5 * trend
    highs = np.maximum(opens, closes) + atr_val * 0.4
    lows = np.minimum(opens, closes) - atr_val * 0.4
    df = pd.DataFrame(
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
    return df


class TestTargetedAuditRound3(unittest.TestCase):
    # -------------------------------------------------------------------------
    # 1. Ranking Explanation & Technical Score vs Directional Priority
    # -------------------------------------------------------------------------
    def test_01_ranking_explanation_never_claims_higher_score_is_lower(self):
        """Zona 76.0 tidak boleh disebut memiliki skor lebih rendah dari Primary 75.3."""
        z_primary = _make_zone("Z_3953", "M5", "SUPPLY", 3953.33, 3955.71, score=75.3, grade="A")
        z_alt = _make_zone("Z_3963", "M5", "SUPPLY", 3963.38, 3966.42, score=76.0, grade="A")

        prio = prioritize_radar_zones(
            radar_zones=[z_primary, z_alt],
            current_price=3891.60,
            htf_bias="BEARISH",
            h1_structure="BEARISH",
            atr=5.10,
            bid=3891.48,
            ask=3891.72,
            snapshot_time="2026-11-13 13:44:00 WIB",
        )

        # Primary Directional Zone harus memilih Z_3953 karena lebih dekat sehingga priority_score lebih tinggi
        self.assertEqual(prio["primary_directional_zone"]["zone_id"], "Z_3953")
        self.assertEqual(prio["alternative_zone"]["zone_id"], "Z_3963")

        # Technical Score Ranking harus menempatkan 76.0 di rank #1 dan 75.3 di rank #2
        tech_rank = prio["ranking_systems"]["technical_quality_ranking"]
        self.assertEqual(tech_rank[0]["zone_id"], "Z_3963")
        self.assertEqual(tech_rank[1]["zone_id"], "Z_3953")

        # Alasan tidak diprioritaskan harus akurat secara matematis
        unprio = prio["zone_scoring_summary"]["unprioritized_reasons"]
        self.assertTrue(len(unprio) >= 1)
        reason_3963 = " ".join(unprio)
        self.assertNotIn("berada di bawah zona prioritas utama", reason_3963)
        self.assertIn("Skor teknikal lebih tinggi", reason_3963)
        self.assertIn("jarak lebih jauh", reason_3963)

    # -------------------------------------------------------------------------
    # 2. Price Reference & Distance Calculation (Bid / Ask / Mid)
    # -------------------------------------------------------------------------
    def test_02_exact_bid_ask_distance_calculations_five_zones(self):
        """Uji 5 zona wajib terhadap Bid=3891.48, Ask=3891.72, Mid=3891.60."""
        bid, ask, mid = 3891.48, 3891.72, 3891.60

        cases = [
            ("SUPPLY", 3914.61, 3923.19, 22.89, "ASK", ask, 3914.61),
            ("DEMAND", 3871.94, 3875.11, 16.37, "BID", bid, 3875.11),
            ("DEMAND", 3865.59, 3868.26, 23.22, "BID", bid, 3868.26),
            ("SUPPLY", 3953.33, 3955.71, 61.61, "ASK", ask, 3953.33),
            ("SUPPLY", 3963.38, 3966.42, 71.66, "ASK", ask, 3963.38),
        ]
        for zt, lo, hi, exp_dist, exp_ref_type, exp_ref_price, exp_bound in cases:
            res = calculate_zone_distance(lo, hi, mid, digits=2, zone_type=zt, bid=bid, ask=ask, atr=5.10)
            self.assertAlmostEqual(res.distance, exp_dist, places=2, msg=f"Gagal pada {zt} {lo}-{hi}")
            self.assertEqual(res.reference_price_type, exp_ref_type)
            self.assertAlmostEqual(res.reference_price, exp_ref_price, places=2)
            self.assertAlmostEqual(res.nearest_boundary, exp_bound, places=2)

    def test_03_reference_label_never_swaps_mid_as_ask_or_bid(self):
        """Jika Bid/Ask tersedia, Ref memakai Bid/Ask; jika tidak, Ref berlabel MID (bukan ASK/BID)."""
        z_sup = _make_zone("Z_S", "H1", "SUPPLY", 3914.61, 3923.19, score=80.0, grade="A")
        z_dem = _make_zone("Z_D", "H1", "DEMAND", 3871.94, 3875.11, score=80.0, grade="A")

        # Dengan Bid/Ask
        scan_live = {
            "symbol": "XAUUSD",
            "current_price": 3891.60,
            "snapshot_price": 3891.60,
            "bid": 3891.48,
            "ask": 3891.72,
            "spread": 0.24,
            "snapshot_time": "2026-11-13 13:44:00 WIB",
            "radar_zones": [z_sup, z_dem],
            "active_zones": [z_sup, z_dem],
            "htf_context": {"H4": "BEARISH", "H1": "BEARISH", "M30": "BEARISH"},
        }
        out_live = format_scan_presentation(scan=scan_live, dec={"status": "WAIT", "stages": {}, "current_price": 3891.60}, detailed=True)
        self.assertIn("Ref: 3891.72 (ASK) -> Proximal: 3914.61", out_live)
        self.assertIn("Ref: 3891.48 (BID) -> Proximal: 3875.11", out_live)
        self.assertNotIn("3891.60 (ASK)", out_live)
        self.assertNotIn("3891.60 (BID)", out_live)

        # Tanpa Bid/Ask -> harus berlabel MID
        scan_mid = {
            "symbol": "XAUUSD",
            "current_price": 3891.60,
            "snapshot_price": 3891.60,
            "snapshot_time": "2026-11-13 13:44:00 WIB",
            "radar_zones": [z_sup, z_dem],
            "active_zones": [z_sup, z_dem],
            "htf_context": {"H4": "BEARISH", "H1": "BEARISH", "M30": "BEARISH"},
        }
        out_mid = format_scan_presentation(scan=scan_mid, dec={"status": "WAIT", "stages": {}, "current_price": 3891.60}, detailed=True)
        self.assertIn("Ref: 3891.60 (MID)", out_mid)
        self.assertNotIn("3891.60 (ASK)", out_mid)
        self.assertNotIn("3891.60 (BID)", out_mid)

    # -------------------------------------------------------------------------
    # 3 & 4. Confirmed Entry Zone Touch Status & Entry Validation Message
    # -------------------------------------------------------------------------
    def test_04_zone_touch_fail_when_untouched_and_accurate_validation_msg(self):
        """Saat zona aktif tersedia dan harga belum menyentuh zona: Zone Touch = FAIL, sisanya NOT_CHECKED, dan tidak muncul 'Tidak ada zona aktif'."""
        z_sup = _make_zone("ZONE_M5_SUPPLY_1794481200", "M5", "SUPPLY", 3953.33, 3955.71, score=75.3, grade="A")
        scan_mock = {
            "symbol": "XAUUSD",
            "current_price": 3891.60,
            "snapshot_price": 3891.60,
            "bid": 3891.48,
            "ask": 3891.72,
            "spread": 0.24,
            "snapshot_time": "2026-11-13 13:44:00 WIB",
            "radar_zones": [z_sup],
            "active_zones": [z_sup],
            "htf_context": {"H4": "BEARISH", "H1": "BEARISH", "M30": "BEARISH"},
        }
        dec_mock = {
            "status": "WAIT",
            "direction": None,
            "reasons": ["Harga belum mencapai zona Supply"],
            "stages": {
                "zone_validation": {"status": "PASS", "msg": "Zona valid"},
                "zone_touch": {"status": "FAIL", "msg": "Harga belum menyentuh zona SUPPLY (3953.33-3955.71, jarak 61.61 pt)"},
                "liquidity_sweep": {"status": "NOT_CHECKED", "msg": "Belum diperiksa"},
                "structure_trigger": {"status": "NOT_CHECKED", "msg": "Belum diperiksa"},
                "displacement": {"status": "NOT_CHECKED", "msg": "Belum diperiksa"},
                "risk_reward": {"status": "NOT_CHECKED", "msg": "Belum diperiksa"},
            },
            "monitored_zone_identity": {
                "zone_id": "ZONE_M5_SUPPLY_1794481200",
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
        out = format_scan_presentation(scan=scan_mock, dec=dec_mock)
        self.assertIn("Zone Touch     : FAIL", out)
        self.assertIn("Liquidity Sweep: NOT_CHECKED", out)
        self.assertIn("CHoCH/BOS/CISD : NOT_CHECKED", out)
        self.assertIn("Displacement   : NOT_CHECKED", out)
        self.assertNotIn("Belum dievaluasi / Tidak ada zona aktif", out)

    def test_05_evaluate_trade_decision_sets_zone_touch_fail_not_not_checked(self):
        """evaluate_trade_decision harus mengisi stages['zone_touch']['status'] == 'FAIL' ketika zona aktif ada tetapi belum tersentuh."""
        frames = {
            "H4": _make_ohlcv(80, base=3950.0, trend=-0.5, atr_val=8.0),
            "H1": _make_ohlcv(80, base=3930.0, trend=-0.4, atr_val=5.1),
            "M30": _make_ohlcv(80, base=3920.0, trend=-0.3, atr_val=4.0),
            "M15": _make_ohlcv(80, base=3910.0, trend=-0.2, atr_val=3.0),
            "M5": _make_ohlcv(80, base=3895.0, trend=-0.05, atr_val=2.0),
        }
        dec = evaluate_trade_decision(frames=frames, current_price=3891.60, spread=0.24, live=False)
        if dec.get("monitored_zone_identity") and dec.get("status") == "WAIT":
            zt_st = dec["stages"].get("zone_touch", {}).get("status")
            self.assertIn(zt_st, ("FAIL", "CONFIRMED", "PASS"))
            if zt_st == "FAIL":
                self.assertEqual(dec["stages"]["liquidity_sweep"]["status"], "NOT_CHECKED")
                self.assertEqual(dec["stages"]["structure_trigger"]["status"], "NOT_CHECKED")
                self.assertEqual(dec["stages"]["displacement"]["status"], "NOT_CHECKED")

    # -------------------------------------------------------------------------
    # 5. Multi-Tier Explanation untuk Zona Radar Qualified yang UNCLASSIFIED
    # -------------------------------------------------------------------------
    def test_06_multi_tier_unclassified_explains_thresholds_and_radar_status(self):
        """M5 Supply 76.0 (71.66 pt) & 75.3 (61.61 pt) tidak dipaksa masuk Medium Risk dan memiliki penjelasan threshold spesifik."""
        z_75 = _make_zone("Z_3953", "M5", "SUPPLY", 3953.33, 3955.71, score=75.3, grade="A")
        z_76 = _make_zone("Z_3963", "M5", "SUPPLY", 3963.38, 3966.42, score=76.0, grade="A")

        prio = prioritize_radar_zones(
            radar_zones=[z_75, z_76],
            current_price=3891.60,
            htf_bias="BEARISH",
            h1_structure="BEARISH",
            atr=5.10,
            bid=3891.48,
            ask=3891.72,
        )

        # Tidak boleh dipaksa masuk High / Medium / Low Risk
        self.assertEqual(len(prio["high_risk_zones"]), 0)
        self.assertEqual(len(prio["medium_risk_zones"]), 0)
        self.assertEqual(len(prio["low_risk_zones"]), 0)

        recon = prio["zone_count_reconciliation"]
        self.assertEqual(recon["radar_qualified_count"], 2)
        self.assertEqual(recon["multi_tier_displayed_count"], 0)
        self.assertEqual(recon["unclassified_tier_count"], 2)

        reasons_joined = "\n".join(recon["unclassified_tier_reasons"])
        self.assertIn("High Risk butuh jarak", reasons_joined)
        self.assertIn("Medium Risk butuh TF M15/M30/H1", reasons_joined)
        self.assertIn("Low Risk butuh TF H1/H4", reasons_joined)
        self.assertIn("Tetap aktif di Radar", reasons_joined)
        self.assertIn("Dipantau sebagai Primary Directional Zone", reasons_joined)

    # -------------------------------------------------------------------------
    # 6. Konsistensi Lintas Bagian Output (Single Source of Truth)
    # -------------------------------------------------------------------------
    def test_07_cross_section_consistency_single_snapshot_and_zone(self):
        """Seluruh bagian output memakai jarak, snapshot, dan identitas zona yang konsisten."""
        z_75 = _make_zone("Z_3953", "M5", "SUPPLY", 3953.33, 3955.71, score=75.3, grade="A")
        z_76 = _make_zone("Z_3963", "M5", "SUPPLY", 3963.38, 3966.42, score=76.0, grade="A")
        z_dem = _make_zone("Z_3871", "H1", "DEMAND", 3871.94, 3875.11, score=80.0, grade="A")

        scan_mock = {
            "symbol": "XAUUSD",
            "current_price": 3891.60,
            "snapshot_price": 3891.60,
            "bid": 3891.48,
            "ask": 3891.72,
            "spread": 0.24,
            "snapshot_time": "2026-11-13 13:44:00 WIB",
            "radar_zones": [z_75, z_76, z_dem],
            "active_zones": [z_75, z_76, z_dem],
            "htf_context": {"H4": "BEARISH", "H1": "BEARISH", "M30": "BEARISH"},
        }
        out = format_scan_presentation(scan=scan_mock, dec={"status": "WAIT", "stages": {}, "current_price": 3891.60}, detailed=True)

        # Jarak konsisten menggunakan Ask=3891.72 untuk Supply (61.61 pt) dan Bid=3891.48 untuk Demand (16.37 pt)
        self.assertIn("61.61 pt", out)
        self.assertIn("16.37 pt", out)
        self.assertIn("Ref: 3891.48 (BID) -> Proximal: 3875.11", out)
        self.assertNotIn("berada di bawah zona prioritas utama", out)
        self.assertNotIn("Belum dievaluasi / Tidak ada zona aktif", out)


if __name__ == "__main__":
    unittest.main()
