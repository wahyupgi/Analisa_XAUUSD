"""
Automated tests for Priority Composite Audit, Final WAIT Reason Consistency,
Combined Timeframe (M15+M5) vs Pure M5 Classification, and Report Consistency.
"""

import os
import sys
import unittest
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import modules.zone_scanner as zs
from modules.zone_scanner import (
    classify_zone_risk_tier,
    format_scan_presentation,
    prioritize_radar_zones,
    scan_xauusd_zones,
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
        "is_valid": True,
        "touch_count": 0,
        "status": status,
        "scoring_status": "VALID",
        "origin_bos": {"origin_valid": True},
        "fvg_confirmation": {"status": "UNTOUCHED"},
        "created_time": datetime.now(timezone.utc).isoformat(),
    }


def _make_ohlcv(n: int = 80, base: float = 3900.0, trend: float = -0.1, atr_val: float = 5.0) -> pd.DataFrame:
    times = pd.date_range(end=datetime(2026, 11, 13, 8, 0, tzinfo=timezone.utc), periods=n, freq="5min")
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


class TestPriorityCompositeAudit(unittest.TestCase):
    def test_01_priority_composite_when_price_changes_same_zone_bucket_vs_continuous(self):
        """
        1. Priority Composite ketika harga berubah tetapi zona tetap sama:
        - Selama jarak berada di bucket APPROACHING (<= max(3.0, 1.5*ATR)), composite tetap sama (+8.0).
        - Ketika harga bergeser keluar ke UNTOUCHED (> 1.5*ATR), composite berubah sesuai penalti jarak kontinu.
        """
        z_sup = _make_zone("Z_SUP", "H1", "SUPPLY", 3920.00, 3925.00, score=78.4, grade="A")

        # Dengan ATR=14.0, approach_threshold = 1.5 * 14.0 = 21.0 pt
        # Jarak 13.08, 14.14, dan 20.15 semuanya masuk bucket APPROACHING (+8.0) -> composite identik
        prices_approaching = [
            3920.00 - 13.08,  # 3906.92
            3920.00 - 14.14,  # 3905.86
            3920.00 - 20.15,  # 3899.85
        ]
        scores_app = []
        for cp in prices_approaching:
            res = prioritize_radar_zones(
                [dict(z_sup)],
                current_price=cp,
                htf_bias="BEARISH",
                h1_structure="BULLISH",  # konflik H4 vs H1 -> tidak ada +15 H1 alignment
                atr=14.0,
            )
            pz = res["primary_directional_zone"]
            self.assertEqual(pz["approach"], "APPROACHING")
            scores_app.append(pz["priority_score"])

        self.assertEqual(scores_app, [151.4, 151.4, 151.4])

        # Ketika harga menjauh keluar dari bucket APPROACHING (> 21.0 pt, misal 25.0 pt) -> UNTOUCHED
        res_untouched = prioritize_radar_zones(
            [dict(z_sup)],
            current_price=3920.00 - 25.00,
            htf_bias="BEARISH",
            h1_structure="BULLISH",
            atr=14.0,
        )
        pz_un = res_untouched["primary_directional_zone"]
        self.assertEqual(pz_un["approach"], "UNTOUCHED")
        self.assertLess(pz_un["priority_score"], scores_app[0])

    def test_02_priority_composite_when_zone_distance_changes_in_untouched(self):
        """2. Priority Composite ketika jarak zona berubah pada status UNTOUCHED menggunakan jarak aktual."""
        z_near = _make_zone("Z_1", "M15", "SUPPLY", 3913.08, 3918.00, score=78.5, grade="A")
        z_mid = _make_zone("Z_1", "M15", "SUPPLY", 3914.14, 3919.00, score=78.5, grade="A")
        z_far = _make_zone("Z_1", "M15", "SUPPLY", 3920.15, 3925.00, score=78.5, grade="A")

        # ATR = 5.0 -> approach_threshold = 7.5 pt, sehingga 13.08, 14.14, 20.15 semuanya UNTOUCHED
        r1 = prioritize_radar_zones([z_near], current_price=3900.00, htf_bias="BEARISH", h1_structure="BULLISH", atr=5.0)
        r2 = prioritize_radar_zones([z_mid], current_price=3900.00, htf_bias="BEARISH", h1_structure="BULLISH", atr=5.0)
        r3 = prioritize_radar_zones([z_far], current_price=3900.00, htf_bias="BEARISH", h1_structure="BULLISH", atr=5.0)

        p1 = r1["primary_directional_zone"]["priority_score"]
        p2 = r2["primary_directional_zone"]["priority_score"]
        p3 = r3["primary_directional_zone"]["priority_score"]

        self.assertGreater(p1, p2)
        self.assertGreater(p2, p3)
        self.assertEqual(p1, round(78.5 + 50.0 + 10.0 + 5.0 - 13.08 * 0.15, 1))
        self.assertEqual(p2, round(78.5 + 50.0 + 10.0 + 5.0 - 14.14 * 0.15, 1))
        self.assertEqual(p3, round(78.5 + 50.0 + 10.0 + 5.0 - 20.15 * 0.15, 1))

    def test_03_priority_composite_consistency_across_report_sections(self):
        """3. Konsistensi nilai composite antara Primary Directional, Alternative Zone, dan Alasan Tidak Diprioritaskan."""
        z_pri = _make_zone("Z_PRI", "H1", "SUPPLY", 3914.00, 3918.00, score=78.5, grade="A")
        z_alt = _make_zone("Z_ALT", "M15", "SUPPLY", 3924.00, 3928.00, score=80.0, grade="A")

        prio = prioritize_radar_zones(
            [z_pri, z_alt],
            current_price=3900.00,
            htf_bias="BEARISH",
            h1_structure="BEARISH",
            atr=5.0,
        )
        pz = prio["primary_directional_zone"]
        az = prio["alternative_zone"]
        self.assertIsNotNone(pz)
        self.assertIsNotNone(az)

        unprio_txt = " ".join(prio["zone_scoring_summary"]["unprioritized_reasons"])
        deprio_txt = " ".join(prio["zone_scoring_summary"]["deprioritized_reasons"])
        self.assertIn(f"{pz['priority_score']:.1f}", unprio_txt)
        self.assertIn(f"{az['priority_score']:.1f}", unprio_txt)
        self.assertEqual(unprio_txt, deprio_txt)

    def test_04_priority_composite_cache_lock_and_refresh(self):
        """4. Cache & refresh: terkunci untuk snapshot harga sama, diperbarui saat harga berubah atau _SCAN_CACHE kedaluwarsa."""
        z = _make_zone("Z_CACHE", "H1", "SUPPLY", 3915.00, 3920.00, score=78.5, grade="A")
        r1 = prioritize_radar_zones([z], current_price=3900.00, htf_bias="BEARISH", h1_structure="BEARISH", atr=5.0)
        locked_item = r1["all_prioritized"][0]
        p_locked = locked_item["priority_score"]

        # Harga sama persis -> tetap terkunci meski ATR berbeda
        r2 = prioritize_radar_zones([locked_item], current_price=3900.00, htf_bias="BEARISH", h1_structure="BEARISH", atr=3.0)
        self.assertEqual(r2["all_prioritized"][0]["priority_score"], p_locked)

        # Harga berubah (dari 3900.00 ke 3890.00) -> _priority_snapshot_price berbeda -> wajib dihitung ulang!
        r3 = prioritize_radar_zones([locked_item], current_price=3890.00, htf_bias="BEARISH", h1_structure="BEARISH", atr=5.0)
        self.assertNotEqual(r3["all_prioritized"][0]["priority_score"], p_locked)

        # Verifikasi _SCAN_CACHE tidak mengembalikan hasil lama ketika harga berubah > 0.1 pt
        zs._SCAN_CACHE.update(key=None, ts=0.0, data=None)
        frames = {tf: _make_ohlcv(80, base=3900.0, atr_val=5.0) for tf in ("H4", "H1", "M30", "M15", "M5")}
        s1 = scan_xauusd_zones(frames=frames, current_price=3900.00)
        s2 = scan_xauusd_zones(frames=frames, current_price=3895.00)
        self.assertEqual(s1["snapshot_price"], 3900.00)
        self.assertEqual(s2["snapshot_price"], 3895.00)

    def test_05_radar_qualified_vs_far_zone_excluded(self):
        """5. Zona Radar Qualified masuk pemantauan aktif; Far Zone Excluded dikeluarkan dari pemantauan intraday."""
        zones = [
            _make_zone("Z_RQ1", "H1", "SUPPLY", 3915.00, 3920.00, score=80.0, grade="A"),
            _make_zone("Z_RQ2", "M15", "SUPPLY", 3910.00, 3914.00, score=76.0, grade="A"),
            _make_zone("Z_FAR1", "H4", "SUPPLY", 4020.00, 4030.00, score=88.0, grade="A+"),
            _make_zone("Z_FAR2", "H1", "DEMAND", 3780.00, 3790.00, score=85.0, grade="A"),
        ]
        prio = prioritize_radar_zones(zones, current_price=3900.00, htf_bias="BEARISH", h1_structure="BULLISH", atr=5.0)
        recon = prio["zone_count_reconciliation"]

        self.assertEqual(recon["radar_qualified_count"], 2)
        self.assertEqual(recon["far_zones_count"], 2)
        rq_ids = {z["zone_id"] for z in prio["radar_qualified"]}
        self.assertEqual(rq_ids, {"Z_RQ1", "Z_RQ2"})
        for r_item in prio["ranking_systems"]["raw_scoring_ranking"]:
            if r_item["zone_id"] in ("Z_FAR1", "Z_FAR2"):
                self.assertEqual(r_item["radar_status"], "FAR_ZONE_EXCLUDED")

    def test_06_final_wait_reason_consistency_with_radar_and_far_zones(self):
        """
        6. Final WAIT Reason ketika terdapat zona Radar dan Far Zone bersamaan:
        Tidak boleh menyebut 'Seluruh zona dipantau sebagai RADAR.' dan wajib membedakan status aktual.
        """
        zones = [
            _make_zone("Z_RQ1", "H1", "SUPPLY", 3912.00, 3916.00, score=80.0, grade="A"),
            _make_zone("Z_RQ2", "M15", "SUPPLY", 3918.00, 3922.00, score=76.0, grade="A"),
            _make_zone("Z_RQ3", "M30", "DEMAND", 3882.00, 3886.00, score=75.0, grade="A"),
            _make_zone("Z_RQ4", "M15+M5", "SUPPLY", 3924.00, 3927.00, score=72.0, grade="B"),
            _make_zone("Z_FAR1", "H4", "SUPPLY", 4010.00, 4020.00, score=88.0, grade="A+"),
            _make_zone("Z_FAR2", "H4", "DEMAND", 3790.00, 3800.00, score=86.0, grade="A+"),
            _make_zone("Z_FAR3", "H1", "SUPPLY", 4030.00, 4040.00, score=84.0, grade="A"),
            _make_zone("Z_FAR4", "H1", "DEMAND", 3770.00, 3780.00, score=82.0, grade="A"),
            _make_zone("Z_FAR5", "M30", "SUPPLY", 4050.00, 4058.00, score=80.0, grade="A"),
        ]
        prio = prioritize_radar_zones(zones, current_price=3900.00, htf_bias="BEARISH", h1_structure="BULLISH", atr=5.0)
        scan = {
            "snapshot_price": 3900.00,
            "htf_context": {"D1": "BEARISH", "H4": "BEARISH", "H1": "BULLISH", "M30": "BULLISH", "bias_utama": "WAIT", "has_conflict": True},
            "radar_zones": zones,
            "active_zones": [],
            "radar_priority": prio,
            "zone_evaluation_summary": {
                "total_candidates": 12,
                "raw_candidates": 12,
                "structurally_valid": 9,
                "structurally_rejected": 0,
                "radar_qualified": 4,
                "pending_qualified": 0,
                "confirmed_entry_qualified": 0,
                "total_supply": 7,
                "total_demand": 5,
                "total_fresh": 9,
                "total_tested": 0,
                "total_mitigated": 1,
                "total_invalidated": 1,
                "total_expired": 1,
            },
        }
        dec = {
            "status": "WAIT",
            "action": "WAIT",
            "direction": None,
            "plan": None,
            "reasons": ["Konflik struktur H4 BEARISH vs H1 BULLISH"],
            "stages": {},
            "radar_priority": prio,
            "zone_count_reconciliation": prio["zone_count_reconciliation"],
        }
        report = format_scan_presentation(scan, dec, detailed=True)
        self.assertNotIn("Seluruh zona dipantau sebagai RADAR", report)
        self.assertIn("4 zona Radar Qualified aktif dipantau", report)
        self.assertIn("5 zona Far Zone Excluded (dikeluarkan dari pemantauan intraday)", report)
        self.assertIn("1 zona Invalidated (tidak valid)", report)
        self.assertIn("1 zona Mitigated (sudah termitigasi)", report)
        self.assertIn("1 zona Expired (kedaluwarsa)", report)

    def test_07_combined_timeframe_m15_plus_m5_evaluation(self):
        """7. Evaluasi timeframe gabungan M15+M5: masuk Medium Risk karena komponen M15, ditolak Low Risk karena bukan HTF."""
        z_combo = _make_zone("Z_COMBO", "M15+M5", "SUPPLY", 3922.00, 3926.00, score=76.0, grade="A")
        tier = classify_zone_risk_tier(z_combo, current_price=3900.00, htf_bias="BEARISH", h1_structure="BEARISH", atr=5.0)
        self.assertEqual(tier, "MEDIUM_RISK")

        # Meski jarak diperlebar sedikit agar tetap moderate (22 pt) dan grade A+, tetap bukan LOW_RISK karena tidak ada H1/H4
        z_combo_aplus = _make_zone("Z_COMBO_AP", "M15+M5", "SUPPLY", 3922.00, 3926.00, score=88.0, grade="A+")
        tier_ap = classify_zone_risk_tier(z_combo_aplus, current_price=3900.00, htf_bias="BEARISH", h1_structure="BEARISH", atr=5.0)
        self.assertEqual(tier_ap, "MEDIUM_RISK")

    def test_08_pure_m5_never_enters_medium_risk(self):
        """8. M5 murni tidak boleh masuk Medium Risk melalui jalur utama maupun jalur fallback."""
        # Jarak 22 pt (> 18 pt batas High Risk, <= 50 pt batas Medium Risk), skor tinggi 85.0
        z_m5 = _make_zone("Z_M5_PURE", "M5", "SUPPLY", 3922.00, 3924.00, score=85.0, grade="A+")
        tier = classify_zone_risk_tier(z_m5, current_price=3900.00, htf_bias="BEARISH", h1_structure="BEARISH", atr=5.0)
        self.assertEqual(tier, "UNCLASSIFIED")
        self.assertIn("Medium Risk butuh TF M15/M30/H1 (aktual TF: M5)", z_m5["risk_tier_reason"])


if __name__ == "__main__":
    unittest.main()
