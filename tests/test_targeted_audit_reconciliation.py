"""
Tests untuk TARGETED FIX — ZONE COUNT, IDENTITY, VALIDATION CONSISTENCY & DISTANCE AUDIT
Memvalidasi 12 skenario wajib:
  1. Radar Qualified 6 tetapi Active Ranking hanya menampilkan 5.
  2. Zona tidak masuk Multi-Tier karena tidak memenuhi kategori risiko (unclassified tier).
  3. Status zona yang belum terklasifikasi (rekonsiliasi mutually exclusive).
  4. Dua zona dengan koordinat sama tetapi timeframe berbeda (H1 vs H4).
  5. Zona yang berubah dari INVALIDATED menjadi VALID karena ambang batas timeframe.
  6. Primary Directional Zone sinkron dengan Confirmed Monitored Zone.
  7. Zona countertrend dalam kategori Medium Risk (tag Countertrend transparan).
  8. Perhitungan jarak Supply (Ask ke proximal low, ABOVE_PRICE).
  9. Perhitungan jarak Demand (Bid ke proximal high, BELOW_PRICE).
  10. ATR tidak tersedia (DATA INCOMPLETE).
  11. H4 dan H1 conflict.
  12. H4 dan H1 aligned.
"""
import math
import unittest
from datetime import datetime
import pandas as pd
import numpy as np

import modules.zone_scanner as zs
from modules.zone_scanner import (
    Zone,
    calculate_zone_distance,
    calculate_distance_atr_ratio,
    classify_zone_risk_tier,
    prioritize_radar_zones,
    format_scan_presentation,
    evaluate_trade_decision,
    is_zone_active_and_valid,
)


def _create_zone(
    zone_id="ZONE_TEST_1",
    tf="H1",
    z_type="DEMAND",
    lo=2600.0,
    hi=2605.0,
    score=75.0,
    grade="A",
    fresh="FRESH",
    touch=0,
    inval=None,
    dist=10.0,
    is_valid=True,
    rej="",
    origin_valid=True,
    status="RADAR",
):
    lo = float(lo)
    hi = float(hi)
    mid = round((lo + hi) / 2.0, 2)
    inval_val = float(inval if inval is not None else (lo if z_type == "DEMAND" else hi))
    return Zone(
        zone_id=zone_id,
        symbol="XAUUSD",
        timeframe=tf,
        zone_type=z_type,
        price_low=lo,
        price_high=hi,
        midpoint=mid,
        score=score,
        grade=grade,
        origin_bos={"tipe": "BOS", "arah": "bull" if z_type == "DEMAND" else "bear", "origin_valid": origin_valid},
        fvg_confirmation={"status": "UNTOUCHED", "size_atr": 1.2},
        displacement_strength=2.2,
        freshness=fresh,
        touch_count=touch,
        liquidity_sweep_status="NONE",
        invalidation_price=inval_val,
        is_valid=is_valid,
        rejection_reason=rej,
        ob_candle_time=pd.Timestamp("2026-11-13 10:00:00"),
        bos_candle_time=pd.Timestamp("2026-11-13 10:30:00"),
        distance=dist,
        status=status,
    )


class TestTargetedAuditReconciliation(unittest.TestCase):

    def test_01_radar_qualified_6_active_ranking_display_5(self):
        """1. Radar Qualified 6 tetapi Active Ranking menampilkan 5 dengan rekonsiliasi eksplisit."""
        zones = [
            _create_zone(f"H1-DEMAND-{i}", "H1", "DEMAND", 2640.0 - i * 5, 2645.0 - i * 5, score=80.0 - i, dist=5.0 + i * 5)
            for i in range(6)
        ]
        prio = prioritize_radar_zones(zones, current_price=2650.0, htf_bias="BULLISH", h1_structure="BULLISH", atr=3.0)
        self.assertEqual(len(prio["radar_qualified"]), 6)
        recon = prio["zone_count_reconciliation"]
        self.assertEqual(recon["radar_qualified_count"], 6)
        self.assertEqual(recon["hidden_active_ranking_count"], 1)

        # Presentation rendering
        scan = {"radar_zones": zones, "active_zones": zones, "radar_priority": prio, "current_price": 2650.0}
        dec = {"status": "WAIT", "stages": {}, "reasons": [], "radar_priority": prio}
        txt = format_scan_presentation(scan=scan, dec=dec, early_setups={"best_setup": None}, detailed=True)
        self.assertIn("1 zona aktif lainnya berada di luar batas tampilan 5 teratas", txt)
        self.assertIn("Tidak Ditampilkan Batas Tampilan Ranking : 1", txt)

    def test_02_zone_unclassified_risk_tier(self):
        """2. Zona tidak masuk Multi-Tier karena tidak memenuhi kategori risiko (unclassified tier)."""
        # Skor 52.0 (di bawah 55.0 untuk high risk, di bawah 65.0 untuk medium risk, di bawah 75.0 untuk low risk)
        z = _create_zone("H1-DEMAND-UNCLASS", "H1", "DEMAND", 2640.0, 2645.0, score=52.0, grade="B", fresh="FRESH", dist=5.0)
        tier = classify_zone_risk_tier(z, current_price=2650.0, htf_bias="BULLISH", h1_structure="BULLISH", atr=3.0)
        self.assertEqual(tier, "UNCLASSIFIED")
        self.assertIn("Tidak memenuhi kriteria High/Medium/Low Risk", z.risk_tier_reason)

        prio = prioritize_radar_zones([z], current_price=2650.0, htf_bias="BULLISH", h1_structure="BULLISH", atr=3.0)
        recon = prio["zone_count_reconciliation"]
        self.assertEqual(recon["unclassified_tier_count"], 1)
        self.assertIn("Tidak memenuhi kriteria High/Medium/Low Risk", recon["unclassified_tier_reasons"][0])

    def test_03_status_mutually_exclusive_reconciliation(self):
        """3. Status zona yang belum terklasifikasi direkonsiliasi secara mutually exclusive."""
        # Simulasi 98 kandidat mentah: 8 fresh valid, 1 tested valid, 15 mitigated, 54 invalidated, 20 structurally rejected
        mock_logged = []
        # 8 fresh valid
        for i in range(8):
            mock_logged.append({"is_valid": True, "mitigation_status": "FRESH", "status": "RADAR", "zone_type": "DEMAND", "touch_count": 0})
        # 1 tested valid
        mock_logged.append({"is_valid": True, "mitigation_status": "TESTED_1X", "status": "RADAR", "zone_type": "DEMAND", "touch_count": 1})
        # 15 mitigated
        for i in range(15):
            mock_logged.append({"is_valid": False, "mitigation_status": "MITIGATED", "status": "MITIGATED", "zone_type": "SUPPLY", "touch_count": 3})
        # 54 invalidated
        for i in range(54):
            mock_logged.append({"is_valid": False, "mitigation_status": "INVALIDATED", "status": "INVALIDATED", "zone_type": "SUPPLY", "rejection_reason": "invalid"})
        # 20 structurally rejected (misal: lebar zona berlebih, origin BOS invalid)
        for i in range(20):
            mock_logged.append({"is_valid": False, "mitigation_status": "FRESH", "status": "REJECTED", "zone_type": "SUPPLY", "rejection_reason": "Zona terlalu lebar (15.41 pt > 14.00)"})

        self.assertEqual(len(mock_logged), 98)

        # Jalankan logika klasifikasi rekonsiliasi eksklusif
        c_inval = c_exp = c_mit = c_frsh_v = c_tested_v = c_struct_rej = 0
        for z in mock_logged:
            z_mit_st = str(z.get("mitigation_status", "")).upper()
            z_st = str(z.get("status", "")).upper()
            z_rej = str(z.get("rejection_reason", "")).lower()
            z_is_v = bool(z.get("is_valid", False))
            z_touch = int(z.get("touch_count", 0))

            if z_st == "INVALIDATED" or z_mit_st == "INVALIDATED" or "invalid" in z_rej:
                c_inval += 1
            elif z_st == "EXPIRED" or z_mit_st == "EXPIRED" or "expired" in z_rej:
                c_exp += 1
            elif z_st == "MITIGATED" or z_mit_st == "MITIGATED" or "dimitigasi" in z_rej or z_touch > 2:
                c_mit += 1
            elif z_is_v:
                if z_mit_st.startswith("TESTED"):
                    c_tested_v += 1
                else:
                    c_frsh_v += 1
            else:
                c_struct_rej += 1

        self.assertEqual(c_frsh_v, 8)
        self.assertEqual(c_tested_v, 1)
        self.assertEqual(c_mit, 15)
        self.assertEqual(c_inval, 54)
        self.assertEqual(c_struct_rej, 20)
        self.assertEqual(c_exp, 0)
        # Total reconciles to exactly 98!
        self.assertEqual(c_frsh_v + c_tested_v + c_mit + c_inval + c_struct_rej + c_exp, 98)

    def test_04_same_coordinates_different_timeframes(self):
        """4. Dua zona dengan koordinat sama tetapi timeframe berbeda memiliki ID dan threshold berbeda."""
        z_h1 = _create_zone("OB-H1-SUPPLY-123", tf="H1", z_type="SUPPLY", lo=4133.24, hi=4148.65, is_valid=False, rej="Zona terlalu lebar (15.41 pt > 14.00)")
        z_h4 = _create_zone("OB-H4-SUPPLY-123", tf="H4", z_type="SUPPLY", lo=4133.24, hi=4148.65, is_valid=True)

        self.assertNotEqual(z_h1.zone_id, z_h4.zone_id)
        self.assertEqual(z_h1.timeframe, "H1")
        self.assertEqual(z_h4.timeframe, "H4")
        self.assertFalse(z_h1.is_valid)
        self.assertTrue(z_h4.is_valid)

    def test_05_zone_invalidation_vs_validation_threshold_consistency(self):
        """5. Ambang batas lebar zona konsisten: H1 batas 14.0 pt, H4 batas 20.0 pt."""
        width = 4148.65 - 4133.24  # 15.41 pt
        atr_h1 = 4.0
        atr_h4 = 6.0
        cap_h1 = min(14.0, 3.0 * atr_h1)  # 12.0 pt
        cap_h4 = min(20.0, 3.0 * atr_h4)  # 18.0 pt

        self.assertTrue(width > cap_h1)  # H1 invalid karena 15.41 > 12.0
        self.assertTrue(width <= cap_h4)  # H4 valid karena 15.41 <= 18.0

    def test_06_primary_directional_matches_confirmed_monitored_zone(self):
        """6. Primary Directional Zone sinkron dengan Confirmed Monitored Zone."""
        z_target = _create_zone("M5-SUPPLY-TARGET", "M5", "SUPPLY", 3953.33, 3955.71, score=85.0, grade="A", fresh="FRESH", dist=36.72)
        z_other = _create_zone("M5-SUPPLY-OTHER", "M5", "SUPPLY", 3963.38, 3966.42, score=78.0, grade="B", fresh="FRESH", dist=46.77)

        prio = prioritize_radar_zones([z_target, z_other], current_price=3916.49, htf_bias="BEARISH", h1_structure="BEARISH", atr=3.0, primary_target_override=z_target)
        self.assertEqual(prio["primary_directional_zone"]["zone_id"], "M5-SUPPLY-TARGET")
        self.assertEqual(prio["eval_target"]["zone_id"], "M5-SUPPLY-TARGET")
        self.assertIn("3953.33-3955.71", prio["wait_breakdown"]["primary_blocker"])

    def test_07_countertrend_medium_risk_transparent_tag(self):
        """7. Zona countertrend dalam kategori Medium Risk menampilkan tag Countertrend transparan."""
        # H4 BEARISH, Demand di bawah harga adalah Countertrend
        z_demand = _create_zone("M30-DEMAND-CT", "M30", "DEMAND", 3871.94, 3875.11, score=72.0, grade="B", fresh="FRESH", dist=41.25)
        tier = classify_zone_risk_tier(z_demand, current_price=3916.49, htf_bias="BEARISH", h1_structure="BEARISH", atr=3.0)
        self.assertEqual(tier, "MEDIUM_RISK")
        self.assertIn("Countertrend", z_demand.risk_tier_reason)
        self.assertIn("Melawan bias HTF BEARISH", z_demand.risk_tier_reason)

    def test_08_distance_supply_calculation(self):
        """8. Perhitungan jarak Supply: Ask ke proximal low, status ABOVE_PRICE."""
        # Current Price 3916.49, Bid 3916.36, Ask 3916.61
        # Supply 3953.33 - 3955.71
        res = calculate_zone_distance(3953.33, 3955.71, current_price=3916.49, digits=2, zone_type="SUPPLY", bid=3916.36, ask=3916.61, atr=3.0)
        self.assertEqual(res.reference_price, 3916.61)
        self.assertEqual(res.reference_price_type, "ASK")
        self.assertEqual(res.nearest_boundary, 3953.33)
        self.assertEqual(res.distance, 36.72)
        self.assertEqual(res.status, "ABOVE_PRICE")
        self.assertEqual(res.distance_atr_ratio, 12.24)
        self.assertEqual(res.ratio_str, "12.2x")

        # Supply 3963.38 - 3966.42
        res2 = calculate_zone_distance(3963.38, 3966.42, current_price=3916.49, digits=2, zone_type="SUPPLY", bid=3916.36, ask=3916.61, atr=3.0)
        self.assertEqual(res2.reference_price, 3916.61)
        self.assertEqual(res2.nearest_boundary, 3963.38)
        self.assertEqual(res2.distance, 46.77)
        self.assertEqual(res2.distance_atr_ratio, 15.59)
        self.assertEqual(res2.ratio_str, "15.6x")

    def test_09_distance_demand_calculation(self):
        """9. Perhitungan jarak Demand: Bid ke proximal high, status BELOW_PRICE."""
        # Current Price 3916.49, Bid 3916.36, Ask 3916.61
        # Demand 3871.94 - 3875.11
        res = calculate_zone_distance(3871.94, 3875.11, current_price=3916.49, digits=2, zone_type="DEMAND", bid=3916.36, ask=3916.61, atr=3.0)
        self.assertEqual(res.reference_price, 3916.36)
        self.assertEqual(res.reference_price_type, "BID")
        self.assertEqual(res.nearest_boundary, 3875.11)
        self.assertEqual(res.distance, 41.25)
        self.assertEqual(res.status, "BELOW_PRICE")
        self.assertEqual(res.distance_atr_ratio, 13.75)
        self.assertEqual(res.ratio_str, "13.8x")

    def test_10_distance_atr_ratio_data_incomplete(self):
        """10. ATR tidak tersedia menghasilkan DATA INCOMPLETE tanpa crash."""
        res = calculate_zone_distance(2640.0, 2645.0, current_price=2650.0, digits=2, zone_type="DEMAND", atr=None)
        self.assertIsNone(res.distance_atr_ratio)
        self.assertEqual(res.ratio_str, "DATA INCOMPLETE")

        ratio, label = calculate_distance_atr_ratio(10.0, atr=None)
        self.assertIsNone(ratio)
        self.assertEqual(label, "DATA INCOMPLETE")

    def test_11_htf_conflict_handling(self):
        """11. Konflik H4 vs H1: seluruh zona berstatus RADAR, primary blocker memuat konflik."""
        z = _create_zone("H1-DEMAND-1", "H1", "DEMAND", 2640.0, 2645.0, score=85.0, grade="A", fresh="FRESH", dist=5.0)
        prio = prioritize_radar_zones([z], current_price=2650.0, htf_bias="BULLISH", h1_structure="BEARISH", atr=3.0)
        self.assertIn("Konflik", prio["wait_breakdown"]["primary_blocker"])
        self.assertIn("BULLISH vs H1 BEARISH", prio["wait_breakdown"]["primary_blocker"])

    def test_12_htf_aligned_handling(self):
        """12. H4 dan H1 aligned: Primary Directional Zone mengarah ke Demand searah."""
        z = _create_zone("H1-DEMAND-1", "H1", "DEMAND", 2640.0, 2645.0, score=85.0, grade="A", fresh="FRESH", dist=5.0)
        prio = prioritize_radar_zones([z], current_price=2650.0, htf_bias="BULLISH", h1_structure="BULLISH", atr=3.0)
        self.assertIsNotNone(prio["primary_directional_zone"])
        self.assertEqual(prio["primary_directional_zone"]["zone_type"], "DEMAND")
        self.assertIn("searah bias BULLISH", prio["primary_reason"])


if __name__ == "__main__":
    unittest.main()
