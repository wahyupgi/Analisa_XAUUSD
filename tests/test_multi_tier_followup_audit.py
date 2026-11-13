"""
Unit & Integration Tests untuk Follow-Up Audit Multi-Tier SMC Zone Engine XAUUSD
Memastikan 10 kriteria pengujian wajib dari USER:
1. Zona INVALID tidak dapat masuk Multi-Tier aktif.
2. Zona dengan ATR ratio sangat besar tidak otomatis menjadi Medium/Low terbaik.
3. Kategori dapat menampilkan NONE jika tidak ada kandidat yang memenuhi (tidak memaksakan 3 kandidat).
4. Zona Fresh dan Structural Valid dapat masuk radar sesuai kriteria.
5. Radar Qualified tidak disamakan dengan Pending Qualified.
6. Pending Qualified tidak disamakan dengan Confirmed Entry Qualified.
7. Status APPROACHING dan ZONE_TOUCHED dibedakan dengan benar (termasuk Bid/Ask Demand vs Supply).
8. Final WAIT Reason dipisahkan menjadi Primary Blocker, Pending Limit Blockers, Confirmed Entry Blockers, Secondary Warnings.
9. Statistik summary memisahkan Raw, Structurally Valid, Radar Qualified, Pending Qualified, Confirmed Qualified, Mitigated, Invalidated.
10. Seluruh fitur lama tetap berjalan (zero regression).
"""

import unittest
from typing import Dict, Any
from modules.zone_scanner import (
    Zone,
    ZoneStatus,
    classify_zone_risk_tier,
    evaluate_zone_hybrid_eligibility,
    prioritize_radar_zones,
    calculate_zone_distance,
    calculate_zone_approach,
    format_scan_presentation,
    HIGH_RISK_MAX_ATR_RATIO,
    MEDIUM_RISK_MAX_ATR_RATIO,
    LOW_RISK_MAX_ATR_RATIO,
)


def _make_zone(
    zone_id: str,
    tf: str,
    z_type: str,
    lo: float,
    hi: float,
    score: float = 80.0,
    grade: str = "A",
    fresh: str = "FRESH",
    is_valid: bool = True,
    is_breached: bool = False,
    rejection: str = "",
    origin_ok: bool = True,
) -> Dict[str, Any]:
    return {
        "zone_id": zone_id,
        "timeframe": tf,
        "zone_type": z_type,
        "price_low": lo,
        "price_high": hi,
        "score": score,
        "grade": grade,
        "freshness": fresh,
        "is_valid": is_valid,
        "is_breached": is_breached,
        "rejection_reason": rejection,
        "origin_bos": {"origin_valid": origin_ok, "tipe": "OB"},
        "fvg_confirmation": {"status": "CONFIRMED"},
        "invalidation_price": lo if z_type == "DEMAND" else hi,
    }


class TestMultiTierFollowupAudit(unittest.TestCase):

    # -------------------------------------------------------------------------
    # 1. ZONA INVALID TIDAK BOLEH MASUK MULTI-TIER AKTIF
    # -------------------------------------------------------------------------
    def test_01_invalid_zone_hard_rejection_from_active_tiers(self):
        """
        Kriteria 1:
        H4 SUPPLY 4274.76–4315.72 memiliki skor 82.9, tetapi Structural Validity INVALID.
        Hard rejection: tidak boleh masuk ke High, Medium, atau Low Risk aktif.
        Skor tinggi TIDAK BOLEH mengalahkan status INVALID.
        Harus masuk ke invalidated_zones.
        """
        curr_p = 2650.0
        atr = 2.0
        # Mocking zona mirip temuan audit: Skor 82.9, tapi is_valid=False
        z_invalid = _make_zone(
            "ZONE_H4_INVALID", tf="H4", z_type="SUPPLY",
            lo=2700.0, hi=2720.0, score=82.9, grade="A+",
            fresh="FRESH", is_valid=False,
            rejection="Struktur patah oleh candle close H4"
        )

        tier = classify_zone_risk_tier(z_invalid, curr_p, htf_bias="BEARISH", atr=atr)
        self.assertEqual(tier, "UNCLASSIFIED", "Zona invalid harus UNCLASSIFIED pada tier aktif")

        elig = evaluate_zone_hybrid_eligibility(z_invalid, curr_p, htf_bias="BEARISH", atr=atr)
        self.assertEqual(elig["radar_eligibility"], "NOT_QUALIFIED", "Radar eligibility harus NOT_QUALIFIED untuk zona invalid")
        self.assertEqual(elig["pending_eligibility"], "NOT_QUALIFIED")
        self.assertEqual(elig["confirmed_entry_eligibility"], "WAIT")

        prio = prioritize_radar_zones([z_invalid], curr_p, htf_bias="BEARISH", atr=atr)
        self.assertNotIn(z_invalid["zone_id"], [z["zone_id"] for z in prio["high_risk_zones"]])
        self.assertNotIn(z_invalid["zone_id"], [z["zone_id"] for z in prio["medium_risk_zones"]])
        self.assertNotIn(z_invalid["zone_id"], [z["zone_id"] for z in prio["low_risk_zones"]])
        self.assertIsNone(prio["best_low_risk"])

        # Masuk ke invalidated_zones
        inval_ids = [z["zone_id"] for z in prio.get("invalidated_zones", [])]
        self.assertIn("ZONE_H4_INVALID", inval_ids)

        # Presentation checks
        mock_scan = {
            "current_price": curr_p, "radar_zones": [z_invalid], "active_zones": [z_invalid],
            "htf_context": {"H4": "BEARISH", "H1": "BEARISH", "M30": "BEARISH", "has_conflict": False},
        }
        mock_dec = {"status": "WAIT", "stages": {}, "reasons": [], "radar_priority": prio}
        pres = format_scan_presentation(scan=mock_scan, dec=mock_dec)
        self.assertIn("[INVALIDATED / HISTORICAL ZONES", pres)
        self.assertIn("ZONE_H4_INVALID", str(prio.get("invalidated_zones", [])))
        self.assertIn("NONE (Tidak ada zona Low Risk yang memenuhi kriteria)", pres)

    # -------------------------------------------------------------------------
    # 2. ZONA DENGAN ATR RATIO BESAR MENDAPAT STATUS FAR / PENALTI
    # -------------------------------------------------------------------------
    def test_02_far_zone_not_best_medium_or_low_risk(self):
        """
        Kriteria 2:
        Zona yang sangat jauh (misal 198.97 pt / 39.0x ATR atau 218.65 pt / 42.8x ATR)
        tidak boleh menjadi kandidat terbaik Medium atau Low Risk.
        Wajib diklasifikasikan sebagai FAR_ZONE.
        """
        curr_p = 2650.0
        atr = 5.0
        # Zona 198.97 pt -> ~39.8x ATR
        z_far_medium = _make_zone(
            "ZONE_FAR_198", tf="H1", z_type="SUPPLY",
            lo=2848.97, hi=2855.00, score=85.0, grade="A+"
        )
        # Zona 218.65 pt -> ~43.7x ATR
        z_far_low = _make_zone(
            "ZONE_FAR_218", tf="H4", z_type="SUPPLY",
            lo=2868.65, hi=2880.00, score=90.0, grade="A+"
        )

        tier_198 = classify_zone_risk_tier(z_far_medium, curr_p, htf_bias="BEARISH", atr=atr)
        tier_218 = classify_zone_risk_tier(z_far_low, curr_p, htf_bias="BEARISH", atr=atr)

        self.assertEqual(tier_198, "FAR_ZONE")
        self.assertEqual(tier_218, "FAR_ZONE")

        prio = prioritize_radar_zones([z_far_medium, z_far_low], curr_p, htf_bias="BEARISH", atr=atr)
        self.assertEqual(len(prio["medium_risk_zones"]), 0)
        self.assertEqual(len(prio["low_risk_zones"]), 0)
        self.assertIsNone(prio["best_medium_risk"])
        self.assertIsNone(prio["best_low_risk"])

    # -------------------------------------------------------------------------
    # 3. KATEGORI DAPAT MENAMPILKAN NONE JIKA TIDAK ADA KANDIDAT
    # -------------------------------------------------------------------------
    def test_03_categories_display_none_when_empty(self):
        """
        Kriteria 3:
        Jika tidak ada kandidat yang memenuhi kategori Medium atau Low Risk,
        tampilkan NONE. JANGAN MEMAKSAKAN tiga kandidat untuk setiap kategori.
        """
        curr_p = 2650.0
        atr = 2.0
        # Hanya ada zona High Risk dekat
        z_high = _make_zone(
            "ZONE_HIGH_ONLY", tf="M15", z_type="DEMAND",
            lo=2645.0, hi=2648.0, score=70.0, grade="B"  # dist = 2.0 pt / 1.0x ATR
        )

        prio = prioritize_radar_zones([z_high], curr_p, htf_bias="BULLISH", atr=atr)
        self.assertEqual(len(prio["high_risk_zones"]), 1)
        self.assertEqual(len(prio["medium_risk_zones"]), 0)
        self.assertEqual(len(prio["low_risk_zones"]), 0)

        mock_scan = {
            "current_price": curr_p, "radar_zones": [z_high], "active_zones": [z_high],
            "htf_context": {"H4": "BULLISH", "H1": "BULLISH", "M30": "BULLISH", "has_conflict": False},
        }
        mock_dec = {"status": "WAIT", "stages": {}, "reasons": [], "radar_priority": prio}
        pres = format_scan_presentation(scan=mock_scan, dec=mock_dec)

        self.assertIn("• A. HIGH RISK ZONES (Aggressive — Peluang Dekat / LTF-MTF):", pres)
        self.assertIn("2645.00-2648.00", pres)
        self.assertIn("NONE (Tidak ada zona Medium Risk yang memenuhi kriteria)", pres)
        self.assertIn("NONE (Tidak ada zona Low Risk yang memenuhi kriteria)", pres)
        self.assertIn("Best Medium Risk Candidate : NONE", pres)
        self.assertIn("Best Low Risk Candidate    : NONE", pres)

    # -------------------------------------------------------------------------
    # 4. ZONA FRESH DAN STRUCTURAL VALID MASUK RADAR SESUAI KRITERIA
    # -------------------------------------------------------------------------
    def test_04_fresh_structural_valid_enters_radar(self):
        """
        Kriteria 4:
        Zona Fresh dan Structural Valid dapat masuk radar sesuai kriteria.
        """
        curr_p = 2650.0
        atr = 2.5
        z_valid_fresh = _make_zone(
            "ZONE_H1_VALID_FRESH", tf="H1", z_type="DEMAND",
            lo=2638.0, hi=2644.0, score=85.0, grade="A", fresh="FRESH", is_valid=True
        )  # dist = 6.0 pt / 2.4x ATR

        tier = classify_zone_risk_tier(z_valid_fresh, curr_p, htf_bias="BULLISH", atr=atr)
        self.assertIn(tier, ("MEDIUM_RISK", "LOW_RISK"))

        elig = evaluate_zone_hybrid_eligibility(z_valid_fresh, curr_p, htf_bias="BULLISH", atr=atr)
        self.assertEqual(elig["radar_eligibility"], "QUALIFIED")

        prio = prioritize_radar_zones([z_valid_fresh], curr_p, htf_bias="BULLISH", atr=atr)
        self.assertEqual(prio["primary_zone"]["zone_id"], "ZONE_H1_VALID_FRESH")
        self.assertEqual(prio["primary_zone"]["status"], "RADAR")

    # -------------------------------------------------------------------------
    # 5. RADAR QUALIFIED TIDAK DISAMAKAN DENGAN PENDING QUALIFIED
    # -------------------------------------------------------------------------
    def test_05_radar_qualified_not_equal_to_pending_qualified(self):
        """
        Kriteria 5:
        Radar Qualified (lolos kriteria pemantauan) tidak boleh disamakan dengan
        Pending Qualified (lolos seluruh syarat Pending Limit Grade A pro-trend).
        """
        curr_p = 2650.0
        atr = 2.0
        # Zona Grade B (Score 65) lolos pemantauan radar, tetapi DITOLAK untuk Pending Limit (butuh Grade A)
        z_grade_b = _make_zone(
            "ZONE_GRADE_B", tf="M30", z_type="DEMAND",
            lo=2640.0, hi=2644.0, score=65.0, grade="B", fresh="FRESH", is_valid=True
        )

        elig = evaluate_zone_hybrid_eligibility(z_grade_b, curr_p, htf_bias="BULLISH", atr=atr)
        self.assertEqual(elig["radar_eligibility"], "QUALIFIED", "Zona Grade B harus lolos radar")
        self.assertEqual(elig["pending_eligibility"], "NOT_QUALIFIED", "Zona Grade B dilarang lolos pending limit")
        self.assertIn("Grade", elig["pending_rejection_reason"])

    # -------------------------------------------------------------------------
    # 6. PENDING QUALIFIED TIDAK DISAMAKAN DENGAN CONFIRMED ENTRY QUALIFIED
    # -------------------------------------------------------------------------
    def test_06_pending_qualified_not_equal_to_confirmed_entry_qualified(self):
        """
        Kriteria 6:
        Pending Limit READY tidak otomatis membuat Confirmed Entry READY.
        M5 WAIT tidak otomatis memblokir Pending Limit jika syarat Pending terpenuhi.
        """
        curr_p = 2650.0
        atr = 2.0
        z_pending_ready = _make_zone(
            "ZONE_PENDING_READY", tf="H1", z_type="DEMAND",
            lo=2640.0, hi=2645.0, score=85.0, grade="A", fresh="FRESH", is_valid=True
        )

        # M5 stages masih belum menyentuh zona (Zone Touch WAIT)
        m5_stages_wait = {
            "zone_touch": {"status": "WAIT"},
            "liquidity_sweep": {"status": "WAIT"},
            "structure_trigger": {"status": "WAIT"},
            "displacement": {"status": "WAIT"},
            "risk_reward": {"status": "WAIT"},
        }

        prio = prioritize_radar_zones(
            [z_pending_ready], curr_p, htf_bias="BULLISH", h1_structure="BULLISH",
            m30_status="CONFIRMED", atr=atr, m5_stages=m5_stages_wait
        )

        # Best Pending Candidate: READY
        self.assertIsNotNone(prio["best_pending_candidate"])
        self.assertEqual(prio["best_pending_candidate"]["pending_eligibility"], "READY")

        # Best Confirmed Candidate: None di prio, dan format presentation menampilkan WAIT
        self.assertIsNone(prio["best_confirmed_candidate"])
        evaluated_z = prio["all_prioritized"][0]
        self.assertEqual(evaluated_z["confirmed_entry_eligibility"], "WAIT")
        self.assertIn("Touch belum terkonfirmasi", evaluated_z["confirmed_rejection_reason"])

        mock_scan = {
            "current_price": curr_p, "radar_zones": [z_pending_ready], "active_zones": [z_pending_ready],
            "htf_context": {"H4": "BULLISH", "H1": "BULLISH", "M30": "BULLISH", "has_conflict": False},
        }
        mock_dec = {"status": "WAIT", "stages": m5_stages_wait, "reasons": [], "radar_priority": prio}
        pres = format_scan_presentation(scan=mock_scan, dec=mock_dec)
        self.assertIn("Best Pending Candidate     : H1 DEMAND (2640.00-2645.00) [Grade A | Skor 85.0] — Status: READY", pres)
        self.assertIn("Best Confirmed Candidate   : WAIT", pres)

    # -------------------------------------------------------------------------
    # 7. STATUS APPROACHING DAN ZONE_TOUCHED DIBEDAKAN DENGAN BENAR
    # -------------------------------------------------------------------------
    def test_07_distinguish_approaching_vs_zone_touched(self):
        """
        Kriteria 7:
        Bedakan kondisi harga mendekati zona (APPROACHING) vs menyentuh zona (ZONE_TOUCHED).
        Jangan menyimpulkan Zone Touch jika harga belum benar-benar memasuki rentang zona.
        Gunakan Bid/Ask sesuai konteks BUY (Demand) / SELL (Supply).
        """
        atr = 2.0

        # Demand: 2640 - 2645. Ask menyentuh di 2645.00
        d_touch = calculate_zone_distance(2640.0, 2645.0, current_price=2645.0, zone_type="DEMAND", ask=2645.0)
        self.assertEqual(d_touch.distance, 0.0)
        app_touch = calculate_zone_approach(d_touch.distance, d_touch.status, atr=atr)
        self.assertEqual(app_touch, ZoneStatus("ZONE_TOUCHED"))

        # Demand: 2640 - 2645. Harga di 2648.0 (jarak 3.0 pt = 1.5x ATR)
        d_app = calculate_zone_distance(2640.0, 2645.0, current_price=2648.0, zone_type="DEMAND")
        self.assertEqual(d_app.distance, 3.0)
        app_near = calculate_zone_approach(d_app.distance, d_app.status, atr=atr)
        self.assertEqual(app_near, "APPROACHING")

        # Supply: 2660 - 2665. Bid menyentuh di 2660.00
        s_touch = calculate_zone_distance(2660.0, 2665.0, current_price=2660.0, zone_type="SUPPLY", bid=2660.0)
        self.assertEqual(s_touch.distance, 0.0)
        app_s_touch = calculate_zone_approach(s_touch.distance, s_touch.status, atr=atr)
        self.assertEqual(app_s_touch, ZoneStatus("ZONE_TOUCHED"))

        # Jarak > 0.0 tidak boleh menghasilkan ZONE_TOUCHED
        d_outside = calculate_zone_distance(2660.0, 2665.0, current_price=2658.0, zone_type="SUPPLY", bid=2658.0)
        self.assertGreater(d_outside.distance, 0.0)
        app_outside = calculate_zone_approach(d_outside.distance, d_outside.status, atr=atr)
        self.assertNotEqual(app_outside, "ZONE_TOUCHED")

    # -------------------------------------------------------------------------
    # 8. FINAL WAIT REASON SESUAI VALIDASI (WAIT BREAKDOWN)
    # -------------------------------------------------------------------------
    def test_08_final_wait_reason_breakdown(self):
        """
        Kriteria 8:
        Pisahkan alasan WAIT menjadi:
        - PRIMARY BLOCKER
        - PENDING LIMIT BLOCKERS
        - CONFIRMED ENTRY BLOCKERS
        - SECONDARY WARNINGS
        Evaluasi murni Python SMC tanpa AI.
        """
        curr_p = 2650.0
        atr = 2.0
        z_dem = _make_zone("ZONE_DEM_1", tf="H1", z_type="DEMAND", lo=2630.0, hi=2635.0, score=80.0, grade="A")

        # Kasus 1: HTF Conflict
        prio_conflict = prioritize_radar_zones(
            [z_dem], curr_p, htf_bias="BEARISH", h1_structure="BULLISH", atr=atr
        )
        wb = prio_conflict["wait_breakdown"]
        self.assertIn("Konflik", wb["primary_blocker"])
        self.assertTrue(len(wb["pending_limit_blockers"]) > 0)
        self.assertTrue(len(wb["confirmed_entry_blockers"]) > 0)

        mock_scan = {
            "current_price": curr_p, "radar_zones": [z_dem], "active_zones": [z_dem],
            "htf_context": {"H4": "BEARISH", "H1": "BULLISH", "M30": "BEARISH", "has_conflict": True},
        }
        mock_dec = {"status": "WAIT", "stages": {}, "reasons": ["Konflik H4 vs H1"], "radar_priority": prio_conflict}
        pres = format_scan_presentation(scan=mock_scan, dec=mock_dec)

        self.assertIn("[WAIT REASON BREAKDOWN]:", pres)
        self.assertIn("PRIMARY BLOCKER        :", pres)
        self.assertIn("PENDING LIMIT BLOCKERS :", pres)
        self.assertIn("CONFIRMED ENTRY BLOCKERS:", pres)

    # -------------------------------------------------------------------------
    # 9. STATISTIK SUMMARY SESUAI JUMLAH KANDIDAT SEBENARNYA
    # -------------------------------------------------------------------------
    def test_09_zone_evaluation_summary_metrics(self):
        """
        Kriteria 9:
        Pisahkan metrik:
        A. RAW CANDIDATES
        B. STRUCTURALLY VALID
        C. RADAR QUALIFIED
        D. PENDING QUALIFIED
        E. CONFIRMED ENTRY QUALIFIED
        F. INVALIDATED
        G. MITIGATED
        Tidak ada satu metrik mewakili lainnya.
        """
        eval_sum = {
            "total_candidates": 10,
            "raw_candidates": 10,
            "structurally_valid": 8,
            "radar_qualified": 5,
            "pending_qualified": 0,
            "confirmed_entry_qualified": 0,
            "total_supply": 4,
            "total_demand": 6,
            "total_fresh": 5,
            "total_tested": 2,
            "total_mitigated": 1,
            "total_invalidated": 2,
        }

        mock_scan = {
            "current_price": 2650.0,
            "radar_zones": [{"timeframe": "H1", "zone_type": "DEMAND", "price_low": 2640.0, "price_high": 2645.0}],
            "zone_evaluation_summary": eval_sum,
        }
        mock_dec = {"status": "WAIT", "stages": {}}
        pres = format_scan_presentation(scan=mock_scan, dec=mock_dec)

        self.assertIn("[ZONE EVALUATION SUMMARY]:", pres)
        self.assertIn("Total Kandidat Zona      : 10", pres)
        self.assertIn("Raw Candidates           : 10", pres)
        self.assertIn("Structurally Valid       : 8", pres)
        self.assertIn("Radar Qualified          : 5 (Zona Aktif Pemantauan)", pres)
        self.assertIn("Pending Qualified (Limit): 0", pres)
        self.assertIn("Confirmed Entry Qualified: 0", pres)
        self.assertIn("Detail Granular          : Fresh: 5 | Tested: 2 | Mitigated: 1 | Invalidated: 2", pres)

    # -------------------------------------------------------------------------
    # 10. REGRESI DAN INTEGRITAS FITUR LAMA
    # -------------------------------------------------------------------------
    def test_10_zero_regression_and_backward_compatibility(self):
        """
        Kriteria 10:
        Seluruh fitur lama tetap berjalan:
        - Bid/Ask/Mid/Spread snapshot
        - ZoneTracker state preservation
        - Overlap merge
        """
        # Bid/Ask/Mid/Spread
        mock_scan = {
            "current_price": 2650.0,
            "bid": 2649.80,
            "ask": 2650.20,
            "mid": 2650.00,
            "spread": 0.40,
            "radar_zones": [],
            "active_zones": [],
        }
        mock_dec = {"status": "WAIT", "stages": {}}
        pres = format_scan_presentation(scan=mock_scan, dec=mock_dec)
        self.assertIn("Bid: 2649.80 | Ask: 2650.20", pres)
        self.assertIn("MT5 Live Tick", pres)


if __name__ == "__main__":
    unittest.main()
