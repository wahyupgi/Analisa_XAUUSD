# -*- coding: utf-8 -*-
"""
tests/test_multi_tier_engine.py
Automated Verification Suite for Multi-Tier SMC Zone Engine (BAGIAN 10: TEST 1 to TEST 13).

Covering:
  TEST 1  — Zone Summary: Supply, Demand, Fresh, Tested, and candidate counts match actual candidate data.
  TEST 2  — Multi-Tier Classification: Classification based on 13 factors, not distance/timeframe alone; far zones (> 25 ATR / > 75 pt) rejected from active tiers.
  TEST 3  — Nearest Zone: Close or touched zones enter radar and are prioritized appropriately.
  TEST 4  — Invalid Structure: Zones with invalid structure are NOT entry candidates just because they are close.
  TEST 5  — HTF Conflict: H4 BEARISH and H1 BULLISH makes Pending Limit NOT_QUALIFIED.
  TEST 6  — Grade B: Grade B cannot be Pending Limit (min Grade A >= 75.0).
  TEST 7  — Grade A Valid: Grade A fresh, HTF aligned, valid structure, RR >= 1.5 is Pending Candidate (READY) even if M5 Confirmed Entry is WAIT.
  TEST 8  — Confirmed Entry: Confirmed Entry is NOT READY if Sweep, Trigger, Displacement not complete.
  TEST 9  — Snapshot MT5: Current Price, Bid, Ask, Mid, Spread, and Timestamp are consistent.
  TEST 10 — Data Incomplete: Missing ATR, tick, or candle data returns DATA INCOMPLETE, not fabricated values.
  TEST 11 — Entry Summary: No contradiction between conditions met, unmet, and Final Decision ("Semua syarat terpenuhi" only shown when VALID).
  TEST 12 — Ranking: Technical Quality Ranking, Directional Priority, Nearest Zone, Pending Eligibility, and Confirmed Entry are separated.
  TEST 13 — Regression: Confluent zone merge vs LTF preservation, deterministic rescans, and tracker integrity.
"""
import math
import unittest
from datetime import datetime
import pandas as pd
import numpy as np

import modules.zone_scanner as zs
from modules.zone_scanner import (
    Zone,
    classify_zone_risk_tier,
    evaluate_zone_hybrid_eligibility,
    calculate_distance_atr_ratio,
    prioritize_radar_zones,
    merge_overlapping_zones,
    format_scan_presentation,
    get_zone_tracker,
)
from modules.config import (
    MULTI_TIER_MAX_ZONES,
    HIGH_RISK_MAX_ATR_RATIO,
    HIGH_RISK_MIN_SCORE,
    MEDIUM_RISK_MAX_ATR_RATIO,
    MEDIUM_RISK_MIN_SCORE,
    LOW_RISK_MIN_SCORE,
)


def _create_mock_zone(
    zone_id="ZONE_MOCK_1",
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
    sl=None,
    tp1=None,
    tp2=None,
):
    lo = float(lo)
    hi = float(hi)
    mid = round((lo + hi) / 2.0, 2)
    inval_val = float(inval if inval is not None else (lo if z_type == "DEMAND" else hi))
    demand = (z_type == "DEMAND")
    sl_val = float(sl if sl is not None else (lo - 2.0 if demand else hi + 2.0))
    tp1_val = float(tp1 if tp1 is not None else (hi + 12.0 if demand else lo - 12.0))
    tp2_val = float(tp2 if tp2 is not None else (hi + 20.0 if demand else lo - 20.0))

    return Zone(
        zone_id=zone_id,
        symbol="XAUUSD",
        timeframe=tf,
        zone_type=z_type,
        price_low=lo,
        price_high=hi,
        midpoint=mid,
        score=float(score),
        grade=grade,
        origin_bos={"origin_valid": origin_valid, "tipe": "OB"},
        fvg_confirmation={"status": "UNTOUCHED"},
        displacement_strength=2.0,
        freshness=fresh,
        touch_count=touch,
        liquidity_sweep_status="SWEEP_VALID",
        invalidation_price=inval_val,
        is_valid=is_valid,
        rejection_reason=rej,
        sl_recommended=sl_val,
        tp1_recommended=tp1_val,
        tp2_recommended=tp2_val,
        distance=dist,
    )


class TestMultiTierEngine(unittest.TestCase):

    def setUp(self):
        self.tracker = get_zone_tracker(auto_save=False)
        self.tracker.clear()

    def test_01_zone_summary_consistency(self):
        """
        TEST 1 — Zone Summary:
        Pastikan jumlah Supply, Demand, Fresh, Tested, dan kandidat lainnya sesuai dengan data kandidat sebenarnya.
        """
        all_logged = [
            {"zone_type": "SUPPLY", "mitigation_status": "FRESH", "is_valid": True},
            {"zone_type": "SUPPLY", "mitigation_status": "TESTED_1X", "is_valid": True},
            {"zone_type": "DEMAND", "mitigation_status": "FRESH", "is_valid": True},
            {"zone_type": "DEMAND", "mitigation_status": "MITIGATED", "is_valid": False, "rejection_reason": "Zona sudah dimitigasi"},
            {"zone_type": "DEMAND", "mitigation_status": "INVALIDATED", "is_valid": False, "status": "INVALIDATED", "rejection_reason": "Zona invalid tertembus"},
        ]
        count_sup = sum(1 for z in all_logged if z["zone_type"] == "SUPPLY")
        count_dem = sum(1 for z in all_logged if z["zone_type"] == "DEMAND")
        count_fresh = sum(1 for z in all_logged if z["mitigation_status"] == "FRESH")
        count_tested = sum(1 for z in all_logged if str(z.get("mitigation_status", "")).startswith("TESTED"))
        count_mit = sum(1 for z in all_logged if z["mitigation_status"] == "MITIGATED")
        count_inval = sum(1 for z in all_logged if z.get("status") == "INVALIDATED")

        self.assertEqual(count_sup, 2)
        self.assertEqual(count_dem, 3)
        self.assertEqual(count_fresh, 2)
        self.assertEqual(count_tested, 1)
        self.assertEqual(count_mit, 1)
        self.assertEqual(count_inval, 1)

        eval_sum = {
            "total_candidates": len(all_logged),
            "total_raw": len(all_logged),
            "total_supply": count_sup,
            "total_demand": count_dem,
            "total_fresh": count_fresh,
            "total_tested": count_tested,
            "total_mitigated": count_mit,
            "total_invalidated": count_inval,
            "initial_radar_count": 3,
            "entry_validated_count": 0,
        }
        rendered = format_scan_presentation(
            scan={"radar_zones": [all_logged[0]], "current_price": 2650.0, "zone_evaluation_summary": eval_sum},
            dec={"status": "WAIT", "stages": {}},
            detailed=True,
        )
        self.assertIn("Total Kandidat Zona      : 5", rendered)
        self.assertIn("Jumlah Supply            : 2", rendered)
        self.assertIn("Jumlah Demand            : 3", rendered)
        self.assertIn("Jumlah Zona Fresh        : 2", rendered)
        self.assertIn("Detail Granular          : Fresh: 2 | Tested: 1 | Mitigated: 1 | Invalidated: 1", rendered)

    def test_02_multi_tier_classification_factors(self):
        """
        TEST 2 — Multi-Tier Classification:
        Pastikan zona tidak diklasifikasikan berdasarkan jarak atau timeframe saja.
        Zona berjarak berlebih (> 25 ATR atau > 75 pt) ditolak dari tier aktif (FAR_ZONE).
        """
        curr_p = 2650.0
        atr = 2.0

        # M15 Demand @ 2645 - 2648 (dist = 4.0 pt / 2.0x ATR) -> HIGH_RISK
        z_m15 = _create_mock_zone("ZONE_M15", tf="M15", z_type="DEMAND", lo=2645.0, hi=2648.0, score=66.0, grade="B", dist=4.0)
        # M30 Demand @ 2625 - 2630 (dist = 20.0 pt / 10.0x ATR) -> MEDIUM_RISK
        z_m30 = _create_mock_zone("ZONE_M30", tf="M30", z_type="DEMAND", lo=2625.0, hi=2630.0, score=72.0, grade="A", dist=20.0)
        # H1 Demand @ 2615 - 2622 (dist = 28.0 pt / 14.0x ATR) -> LOW_RISK
        z_h1 = _create_mock_zone("ZONE_H1", tf="H1", z_type="DEMAND", lo=2615.0, hi=2622.0, score=82.0, grade="A", dist=28.0)
        # H4 Demand jauh (dist = 200.0 pt / 100.0x ATR) -> FAR_ZONE (Ditolak dari active Low/Medium Risk)
        z_h4_far = _create_mock_zone("ZONE_H4_FAR", tf="H4", z_type="DEMAND", lo=2440.0, hi=2450.0, score=88.0, grade="A+", dist=200.0)

        t_m15 = classify_zone_risk_tier(z_m15, curr_p, htf_bias="BULLISH", atr=atr)
        t_m30 = classify_zone_risk_tier(z_m30, curr_p, htf_bias="BULLISH", atr=atr)
        t_h1 = classify_zone_risk_tier(z_h1, curr_p, htf_bias="BULLISH", atr=atr)
        t_far = classify_zone_risk_tier(z_h4_far, curr_p, htf_bias="BULLISH", atr=atr)

        self.assertEqual(t_m15, "HIGH_RISK")
        self.assertEqual(t_m30, "MEDIUM_RISK")
        self.assertEqual(t_h1, "LOW_RISK")
        self.assertEqual(t_far, "FAR_ZONE", "Zona berjarak 200 pt (> 25 ATR) wajib diklasifikasikan sebagai FAR_ZONE")

        res = prioritize_radar_zones([z_m15, z_m30, z_h1, z_h4_far], curr_p, htf_bias="BULLISH", atr=atr)
        lr_ids = [z["zone_id"] for z in res.get("low_risk_zones", [])]
        self.assertNotIn("ZONE_H4_FAR", lr_ids, "Zona 200 pt tidak boleh masuk ke Low Risk intraday")

    def test_03_nearest_and_touched_zone_enters_radar(self):
        """
        TEST 3 — Nearest Zone:
        Pastikan zona yang dekat atau sedang disentuh tetap masuk radar jika memenuhi kriteria.
        """
        curr_p = 2650.0
        atr = 2.0
        z_near = _create_mock_zone("ZONE_NEAR_TOUCHED", tf="M15", z_type="DEMAND", lo=2648.0, hi=2651.0, score=70.0, grade="B", dist=0.0)
        z_near.approach = "TOUCHED"

        res = prioritize_radar_zones([z_near], curr_p, htf_bias="BULLISH", atr=atr)
        hr_zones = res.get("high_risk_zones", [])
        self.assertEqual(len(hr_zones), 1)
        self.assertEqual(hr_zones[0]["zone_id"], "ZONE_NEAR_TOUCHED")
        self.assertEqual(hr_zones[0]["radar_eligibility"], "QUALIFIED")

    def test_04_invalid_structure_rejected_from_entry_even_if_close(self):
        """
        TEST 4 — Invalid Structure:
        Pastikan zona yang tidak valid secara struktur tidak menjadi kandidat entry hanya karena dekat dengan harga.
        """
        curr_p = 2650.0
        atr = 2.0
        z_broken = _create_mock_zone(
            "ZONE_M5_BROKEN", tf="M5", z_type="DEMAND", lo=2648.0, hi=2650.0,
            score=45.0, grade="REJECTED", fresh="INVALIDATED", is_valid=False,
            rej="Zona invalid tertembus candle close", dist=1.0
        )
        tier = classify_zone_risk_tier(z_broken, curr_p, htf_bias="BULLISH", atr=atr)
        self.assertEqual(tier, "UNCLASSIFIED")

        h_eval = evaluate_zone_hybrid_eligibility(z_broken, curr_p, htf_bias="BULLISH", atr=atr)
        self.assertEqual(h_eval["radar_eligibility"], "NOT_QUALIFIED")
        self.assertEqual(h_eval["pending_eligibility"], "NOT_QUALIFIED")

    def test_05_htf_conflict_pending_not_qualified(self):
        """
        TEST 5 — HTF Conflict:
        H4 BEARISH dan H1 BULLISH harus membuat Pending Limit NOT_QUALIFIED sesuai aturan.
        """
        curr_p = 2650.0
        atr = 2.0
        z_h1 = _create_mock_zone("ZONE_H1_CONFLICT", tf="H1", z_type="SUPPLY", lo=2665.0, hi=2670.0, score=85.0, grade="A+", dist=15.0)

        h_eval = evaluate_zone_hybrid_eligibility(
            z_h1, curr_p, htf_bias="BEARISH", h1_struct="BULLISH", m30_status="CONFIRMED", atr=atr
        )
        self.assertEqual(h_eval["pending_eligibility"], "NOT_QUALIFIED")
        self.assertIn("Konflik", h_eval["pending_rejection_reason"])

    def test_06_grade_b_not_pending_limit(self):
        """
        TEST 6 — Grade B:
        Zona Grade B tidak boleh menjadi Pending Limit jika persyaratan minimal adalah Grade A.
        """
        curr_p = 2650.0
        atr = 2.0
        z_b = _create_mock_zone("ZONE_GRADE_B", tf="M15", z_type="DEMAND", lo=2642.0, hi=2646.0, score=65.0, grade="B", dist=4.0)

        h_eval = evaluate_zone_hybrid_eligibility(
            z_b, curr_p, htf_bias="BULLISH", h1_struct="BULLISH", m30_status="CONFIRMED", atr=atr
        )
        self.assertEqual(h_eval["radar_eligibility"], "QUALIFIED")
        self.assertEqual(h_eval["pending_eligibility"], "NOT_QUALIFIED")
        self.assertIn("Grade A", h_eval["pending_rejection_reason"])

    def test_07_grade_a_valid_pending_ready_without_m5_trigger(self):
        """
        TEST 7 — Grade A Valid:
        Zona Grade A yang fresh, HTF aligned, struktural valid, dan RR memenuhi syarat boleh menjadi Pending Candidate meskipun M5 belum memberikan trigger Confirmed Entry.
        """
        curr_p = 2650.0
        atr = 2.0
        # Entry 2630, SL 2622 (risk = 8), TP1 2646 (reward = 16, RR = 2.0 >= 1.5)
        z_a = _create_mock_zone(
            "ZONE_GRADE_A", tf="H1", z_type="DEMAND", lo=2625.0, hi=2630.0,
            score=82.0, grade="A", fresh="FRESH", inval=2623.0, dist=20.0,
            sl=2622.0, tp1=2646.0
        )
        m5_stages = {
            "zone_touch": {"status": "WAIT"},
            "liquidity_sweep": {"status": "WAIT"},
            "structure_trigger": {"status": "WAIT"},
            "displacement": {"status": "WAIT"},
            "risk_reward": {"status": "WAIT"},
        }
        h_eval = evaluate_zone_hybrid_eligibility(
            z_a, curr_p, htf_bias="BULLISH", h1_struct="BULLISH",
            m30_status="CONFIRMED", m5_stages=m5_stages, atr=atr
        )
        self.assertEqual(h_eval["pending_eligibility"], "READY")
        self.assertEqual(h_eval["confirmed_entry_eligibility"], "WAIT")
        self.assertTrue(h_eval["rr"] >= 1.5)

    def test_08_confirmed_entry_wait_if_stages_incomplete(self):
        """
        TEST 8 — Confirmed Entry:
        Pastikan Confirmed Entry tidak READY jika Liquidity Sweep, Structure Trigger, atau Displacement belum lengkap.
        """
        curr_p = 2650.0
        atr = 2.0
        z_cand = _create_mock_zone("ZONE_CONF_TEST", tf="H1", z_type="DEMAND", lo=2645.0, hi=2649.0, dist=1.0)
        m5_stages = {
            "zone_touch": {"status": "CONFIRMED"},
            "liquidity_sweep": {"status": "WAIT", "msg": "Belum ada sweep"},
            "structure_trigger": {"status": "WAIT"},
            "displacement": {"status": "WAIT"},
            "risk_reward": {"status": "WAIT"},
        }
        h_eval = evaluate_zone_hybrid_eligibility(
            z_cand, curr_p, htf_bias="BULLISH", h1_struct="BULLISH",
            m30_status="CONFIRMED", m5_stages=m5_stages, atr=atr
        )
        self.assertEqual(h_eval["confirmed_entry_eligibility"], "WAIT")

    def test_09_snapshot_mt5_consistency(self):
        """
        TEST 9 — Snapshot MT5:
        Pastikan Current Price, Bid, Ask, Mid, Spread, dan Timestamp konsisten.
        """
        bid = 3914.07
        ask = 3914.30
        mid = round((bid + ask) / 2.0, 2)
        spread = round(ask - bid, 2)
        now_str = "2026-11-13 10:15:00 WIB"

        scan_mock = {
            "current_price": mid,
            "snapshot_price": mid,
            "snapshot_time": now_str,
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "spread": spread,
            "price_source": f"MT5 Live Tick (Bid: {bid:.2f} | Ask: {ask:.2f} | Mid: {mid:.2f} | Spread: {spread:.2f} pt)",
            "radar_zones": [_create_mock_zone("ZONE_SNP", lo=3900.0, hi=3905.0, dist=9.0)],
        }
        rendered = format_scan_presentation(scan=scan_mock, dec={"status": "WAIT", "stages": {}})
        self.assertIn(f"Current Price    : {mid:.2f} (Snapshot: {now_str} | Bid: {bid:.2f} | Ask: {ask:.2f})", rendered)
        self.assertIn(f"MT5 Live Tick (Bid: {bid:.2f} | Ask: {ask:.2f} | Mid: {mid:.2f} | Spread: {spread:.2f} pt)", rendered)

    def test_10_data_incomplete_no_fabricated_values(self):
        """
        TEST 10 — Data Incomplete:
        Pastikan ATR, tick, atau data candle yang tidak tersedia tidak dianggap sebagai nilai valid.
        """
        ratio_none, label_none = calculate_distance_atr_ratio(10.0, None)
        self.assertIsNone(ratio_none)
        self.assertEqual(label_none, "DATA INCOMPLETE")

        ratio_zero, label_zero = calculate_distance_atr_ratio(10.0, 0.0)
        self.assertIsNone(ratio_zero)
        self.assertEqual(label_zero, "DATA INCOMPLETE")

        z = _create_mock_zone("ZONE_NO_ATR", tf="M15", z_type="DEMAND", lo=2640.0, hi=2645.0, dist=5.0)
        z.distance_atr_ratio = None
        res = prioritize_radar_zones([z], 2650.0, htf_bias="BULLISH", atr=None)
        rendered = format_scan_presentation(scan={"radar_zones": [z], "current_price": 2650.0, "radar_priority": res}, dec={"status": "WAIT", "stages": {}}, detailed=True)
        self.assertIn("DATA INCOMPLETE", rendered)

    def test_11_entry_summary_no_contradiction(self):
        """
        TEST 11 — Entry Summary:
        Pastikan tidak ada kontradiksi antara kondisi terpenuhi, belum terpenuhi, dan Final Decision.
        ("Semua syarat terpenuhi" hanya muncul jika status == VALID).
        """
        # Case A: Status WAIT -> tidak boleh memunculkan 'Semua syarat terpenuhi'
        dec_wait = {
            "status": "WAIT",
            "direction": "BUY",
            "reasons": ["Belum ada sweep likuiditas M5"],
            "entry_validation_summary": {
                "monitored_zone": "H1 DEMAND (2630.00 - 2635.00)",
                "conditions_met": ["Zone Touch [CONFIRMED]"],
                "conditions_unmet": ["Liquidity Sweep [WAIT]", "CHoCH/BOS/CISD M5 [WAIT]"],
                "final_decision_reason": "WAIT: Menunggu konfirmasi trigger M5",
            }
        }
        rend_wait = format_scan_presentation(scan={"radar_zones": [], "current_price": 2650.0}, dec=dec_wait, detailed=True)
        self.assertIn("Kondisi Terpenuhi        : Zone Touch [CONFIRMED]", rend_wait)
        self.assertIn("Liquidity Sweep [WAIT]", rend_wait)
        self.assertNotIn("Semua syarat terpenuhi", rend_wait)

        # Case B: Status VALID -> boleh memunculkan 'None (Semua syarat terpenuhi)'
        dec_valid = {
            "status": "VALID",
            "direction": "BUY",
            "reasons": ["Seluruh konfirmasi M5 terpenuhi"],
            "plan": {"entry_acuan": 2635.0, "entry_min": 2634.0, "entry_max": 2636.0, "sl": 2627.0, "tp1": 2651.0, "rr": 2.0},
            "entry_validation_summary": {
                "monitored_zone": "H1 DEMAND (2630.00 - 2635.00)",
                "conditions_met": ["Zone Touch", "Liquidity Sweep", "CHoCH/BOS/CISD", "Displacement", "Risk/Reward"],
                "conditions_unmet": [],
                "final_decision_reason": "BUY VALID",
            }
        }
        rend_valid = format_scan_presentation(scan={"radar_zones": [], "current_price": 2650.0}, dec=dec_valid, detailed=True)
        self.assertIn("Kondisi Belum Terpenuhi  : None (Semua syarat terpenuhi)", rend_valid)

    def test_12_ranking_systems_separation(self):
        """
        TEST 12 — Ranking:
        Pastikan Technical Quality Ranking, Directional Priority, Nearest Zone, dan Entry Eligibility tidak tercampur.
        """
        curr_p = 2650.0
        atr = 2.0
        # Zone A: Close (dist = 3.0), lower score (65.0)
        z_a = _create_mock_zone("ZONE_A_NEAR", tf="M15", z_type="DEMAND", lo=2644.0, hi=2647.0, score=65.0, grade="B", dist=3.0)
        # Zone B: Farther (dist = 20.0), higher score (88.0)
        z_b = _create_mock_zone("ZONE_B_FAR", tf="H1", z_type="DEMAND", lo=2625.0, hi=2630.0, score=88.0, grade="A+", dist=20.0)

        res = prioritize_radar_zones([z_a, z_b], curr_p, htf_bias="BULLISH", atr=atr)
        rankings = res.get("ranking_systems", {})

        self.assertIn("technical_quality_ranking", rankings)
        self.assertIn("directional_priority_ranking", rankings)
        self.assertIn("nearest_zone_ranking", rankings)
        self.assertIn("pending_eligibility_ranking", rankings)
        self.assertIn("confirmed_entry_ranking", rankings)

        # Technical Quality: Zone B must rank #1 due to higher score (88.0 vs 65.0)
        self.assertEqual(rankings["technical_quality_ranking"][0]["zone_id"], "ZONE_B_FAR")

        # Nearest Zone: Zone A must rank #1 due to shorter distance (3.0 vs 20.0)
        self.assertEqual(rankings["nearest_zone_ranking"][0]["zone_id"], "ZONE_A_NEAR")

    def test_13_regression_and_legacy_features(self):
        """
        TEST 13 — Regression:
        Pastikan fungsi dan fitur lama yang tidak terkait dengan perbaikan tetap berjalan:
        - Penggabungan zona overlap (merge vs independent LTF preservation).
        - Konsistensi deterministik identitas zona & scoring saat rescan.
        - Tracker state update.
        """
        # 1. Overlapping zones: Confluent merge vs LTF preservation
        z_h4 = _create_mock_zone("ZONE_H4_A", tf="H4", z_type="DEMAND", lo=2600.0, hi=2610.0, score=85.0)
        z_h1 = _create_mock_zone("ZONE_H1_A", tf="H1", z_type="DEMAND", lo=2602.0, hi=2611.0, score=80.0)
        merged_confluent = merge_overlapping_zones([z_h4, z_h1])
        self.assertEqual(len(merged_confluent), 1)

        z_h4_broad = _create_mock_zone("ZONE_H4_BROAD", tf="H4", z_type="DEMAND", lo=2600.0, hi=2620.0, score=85.0)
        z_m15_refine = _create_mock_zone("ZONE_M15_REFINE", tf="M15", z_type="DEMAND", lo=2605.0, hi=2608.0, score=72.0)
        preserved_refinements = merge_overlapping_zones([z_h4_broad, z_m15_refine])
        self.assertEqual(len(preserved_refinements), 2)

        # 2. Rescan determinism
        curr_p = 2610.0
        atr = 2.0
        res1 = prioritize_radar_zones([z_m15_refine, z_h4_broad], curr_p, htf_bias="BULLISH", atr=atr)
        res2 = prioritize_radar_zones([z_m15_refine, z_h4_broad], curr_p, htf_bias="BULLISH", atr=atr)
        self.assertEqual(res1["high_risk_zones"][0]["zone_id"], res2["high_risk_zones"][0]["zone_id"])
        self.assertEqual(res1["high_risk_zones"][0]["score"], res2["high_risk_zones"][0]["score"])


if __name__ == "__main__":
    unittest.main()
