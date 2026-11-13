# -*- coding: utf-8 -*-
"""
tests/test_active_zone_scoring_wait_audit.py
Automated Verification Suite for Section 7: Testing Wajib (15 Mandatory Scenarios).

Scenarios:
  1. Zona skor tinggi tetapi INVALIDATED.
  2. Zona skor rendah tetapi structurally valid.
  3. Zona dekat tetapi countertrend.
  4. Zona jauh tetapi HTF aligned.
  5. Zona fresh dan valid.
  6. Zona mitigated.
  7. Zona expired.
  8. H4 dan H1 conflict.
  9. H4 dan H1 aligned.
  10. Harga belum menyentuh zona.
  11. Harga sudah menyentuh zona tetapi belum ada sweep.
  12. Sweep ada tetapi belum ada CHoCH/BOS/CISD.
  13. Semua kondisi Confirmed Entry terpenuhi.
  14. Tidak ada kandidat Low Risk.
  15. Data ATR tidak tersedia.
"""
import math
import unittest
from datetime import datetime
import pandas as pd
import numpy as np

import modules.zone_scanner as zs
from modules.zone_scanner import (
    Zone,
    is_zone_active_and_valid,
    classify_zone_risk_tier,
    evaluate_zone_hybrid_eligibility,
    calculate_distance_atr_ratio,
    prioritize_radar_zones,
    format_scan_presentation,
    evaluate_trade_decision,
    sanitize_wait_reason,
    ValidationStatus,
)


def _create_test_zone(
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
    is_expired=False,
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
        liquidity_sweep_status="CONFIRMED_SWEEP_LOW" if z_type == "DEMAND" else "CONFIRMED_SWEEP_HIGH",
        invalidation_price=inval_val,
        is_valid=is_valid,
        rejection_reason=rej,
        distance=dist,
        status=status,
        sl_recommended=lo - 2.0 if z_type == "DEMAND" else hi + 2.0,
        tp1_recommended=mid + 10.0 if z_type == "DEMAND" else mid - 10.0,
        tp2_recommended=mid + 15.0 if z_type == "DEMAND" else mid - 15.0,
    )


class TestActiveZoneScoringWaitAudit(unittest.TestCase):
    """Pengujian komprehensif 15 skenario wajib Section 7."""

    def test_01_high_score_but_invalidated(self):
        """1. Zona skor tinggi tetapi INVALIDATED: Hard rejection dari radar aktif dan multi-tier."""
        curr_p = 2650.0
        # Skor 85.0 (Grade A+), tetapi status INVALIDATED dan distal level breached
        z_inval = _create_test_zone(
            "ZONE_H4_INVAL", tf="H4", z_type="SUPPLY",
            lo=2620.0, hi=2630.0, score=85.0, grade="A+",
            fresh="INVALIDATED", status="INVALIDATED", is_valid=False,
            rej="Zona invalid (close 2650.00 > 2630.00)", inval=2630.0
        )
        self.assertFalse(is_zone_active_and_valid(z_inval, curr_p))

        # Test classify_zone_risk_tier
        tier = classify_zone_risk_tier(z_inval, curr_p, htf_bias="BEARISH", atr=2.0)
        self.assertEqual(tier, "UNCLASSIFIED")
        self.assertIn("invalid", z_inval.risk_tier_reason.lower())

        # Test prioritize_radar_zones
        prio = prioritize_radar_zones([z_inval], curr_p, htf_bias="BEARISH", atr=2.0)
        self.assertEqual(len(prio["radar_qualified"]), 0)
        self.assertIsNone(prio["primary_directional_zone"])
        self.assertIsNone(prio["best_high_risk"])
        self.assertIsNone(prio["best_medium_risk"])
        self.assertIsNone(prio["best_low_risk"])
        self.assertIsNone(prio["best_pending_candidate"])
        self.assertIsNone(prio["best_confirmed_candidate"])
        self.assertIsNone(prio["nearest_radar_zone"])

        # Masuk ke historical / invalidated zones
        self.assertEqual(len(prio["invalidated_zones"]), 1)
        self.assertEqual(prio["invalidated_zones"][0]["zone_id"], "ZONE_H4_INVAL")

    def test_02_low_score_but_structurally_valid(self):
        """2. Zona skor rendah tetapi structurally valid: Tidak dicap invalidated dan dipantau transparan."""
        curr_p = 2640.0
        # Skor 55.0 (Grade B), valid secara struktur, fresh
        z_low = _create_test_zone(
            "ZONE_M15_LOW", tf="M15", z_type="DEMAND",
            lo=2635.0, hi=2638.0, score=55.0, grade="B",
            fresh="FRESH", touch=0, is_valid=True, dist=2.0
        )
        self.assertTrue(is_zone_active_and_valid(z_low, curr_p))

        prio = prioritize_radar_zones([z_low], curr_p, htf_bias="BULLISH", atr=2.0)
        # Boleh masuk radar jika memenuhi ambang minimal
        self.assertEqual(len(prio["radar_qualified"]), 1)
        self.assertEqual(prio["radar_qualified"][0]["zone_id"], "ZONE_M15_LOW")
        # Grade B tidak boleh menjadi Low Risk (min Grade A >= 75.0)
        self.assertIsNone(prio["best_low_risk"])
        # Tidak masuk historical invalidated
        self.assertEqual(len(prio["invalidated_zones"]), 0)

    def test_03_close_zone_but_countertrend(self):
        """3. Zona dekat tetapi countertrend: Berjarak dekat namun melawan bias HTF tidak boleh menjadi pro-trend entry."""
        curr_p = 2650.0
        # HTF BULLISH tapi ada SUPPLY sangat dekat (dist = 1.0 pt)
        z_supply_close = _create_test_zone(
            "ZONE_SUPPLY_CLOSE", tf="M15", z_type="SUPPLY",
            lo=2651.0, hi=2653.0, score=78.0, grade="A",
            fresh="FRESH", touch=0, dist=1.0
        )
        prio = prioritize_radar_zones([z_supply_close], curr_p, htf_bias="BULLISH", atr=2.0)

        # Tidak boleh menjadi Primary Directional Zone
        self.assertIsNone(prio["primary_directional_zone"])
        # Terdeteksi sebagai countertrend
        self.assertIsNotNone(prio["countertrend_zone"])
        self.assertEqual(prio["countertrend_zone"]["zone_id"], "ZONE_SUPPLY_CLOSE")
        # Menghasilkan catatan/peringatan countertrend
        wb = prio.get("wait_breakdown", {})
        self.assertTrue(any("berlawanan" in w.lower() or "countertrend" in w.lower() for w in wb.get("secondary_warnings", [])))

    def test_04_far_zone_but_htf_aligned(self):
        """4. Zona jauh tetapi HTF aligned: Diberi penalti jarak atau status FAR, tidak otomatis terpilih."""
        curr_p = 2650.0
        # HTF BULLISH, demand berkualitas tinggi (skor 88.0, Grade A+) tapi jarak 90 pt (45x ATR)
        z_far = _create_test_zone(
            "ZONE_DEMAND_FAR", tf="H4", z_type="DEMAND",
            lo=2555.0, hi=2560.0, score=88.0, grade="A+",
            fresh="FRESH", touch=0, dist=90.0
        )
        tier = classify_zone_risk_tier(z_far, curr_p, htf_bias="BULLISH", atr=2.0)
        # Melebihi ambang batas intraday (> 25 ATR / > 75 pt) -> FAR_ZONE
        self.assertEqual(tier, "FAR_ZONE")

        prio = prioritize_radar_zones([z_far], curr_p, htf_bias="BULLISH", atr=2.0)
        # FAR_ZONE dikeluarkan dari radar_qualified aktif
        self.assertEqual(len(prio["radar_qualified"]), 0)
        self.assertIsNone(prio["best_low_risk"])
        self.assertIsNone(prio["best_medium_risk"])
        self.assertIsNone(prio["best_high_risk"])

    def test_05_fresh_and_valid_zone(self):
        """5. Zona fresh dan valid: Lolos radar, multi-tier, dan siap dievaluasi."""
        curr_p = 2650.0
        z_fresh = _create_test_zone(
            "ZONE_H1_FRESH", tf="H1", z_type="DEMAND",
            lo=2638.0, hi=2642.0, score=82.0, grade="A",
            fresh="FRESH", touch=0, dist=8.0
        )
        self.assertTrue(is_zone_active_and_valid(z_fresh, curr_p))

        prio = prioritize_radar_zones([z_fresh], curr_p, htf_bias="BULLISH", h1_structure="BULLISH", atr=2.0)
        self.assertEqual(len(prio["radar_qualified"]), 1)
        self.assertIsNotNone(prio["primary_directional_zone"])
        self.assertEqual(prio["primary_directional_zone"]["zone_id"], "ZONE_H1_FRESH")
        # Terklasifikasi ke tier yang valid (Medium atau Low Risk)
        self.assertIn(prio["radar_qualified"][0]["risk_tier"], ("LOW_RISK", "MEDIUM_RISK"))

    def test_06_mitigated_zone(self):
        """6. Zona mitigated: touch_count > 2 atau MITIGATED ditolak dari radar aktif."""
        curr_p = 2650.0
        z_mit = _create_test_zone(
            "ZONE_MITIGATED", tf="H1", z_type="DEMAND",
            lo=2640.0, hi=2645.0, score=80.0, grade="A",
            fresh="MITIGATED", touch=3, is_valid=False,
            rej="Zona sudah dimitigasi (3x > 2x sentuhan)", status="MITIGATED"
        )
        self.assertFalse(is_zone_active_and_valid(z_mit, curr_p))

        prio = prioritize_radar_zones([z_mit], curr_p, htf_bias="BULLISH", atr=2.0)
        self.assertEqual(len(prio["radar_qualified"]), 0)
        self.assertIsNone(prio["best_high_risk"])
        self.assertIsNone(prio["best_medium_risk"])
        self.assertIsNone(prio["best_low_risk"])
        self.assertEqual(len(prio["invalidated_zones"]), 1)
        self.assertEqual(prio["invalidated_zones"][0]["zone_id"], "ZONE_MITIGATED")

    def test_07_expired_zone(self):
        """7. Zona expired: Batas waktu habis ditolak dari radar aktif."""
        curr_p = 2650.0
        z_exp = _create_test_zone(
            "ZONE_EXPIRED", tf="H1", z_type="DEMAND",
            lo=2640.0, hi=2645.0, score=78.0, grade="A",
            fresh="EXPIRED", touch=0, is_valid=False,
            rej="Zona expired", status="EXPIRED", is_expired=True
        )
        self.assertFalse(is_zone_active_and_valid(z_exp, curr_p))

        prio = prioritize_radar_zones([z_exp], curr_p, htf_bias="BULLISH", atr=2.0)
        self.assertEqual(len(prio["radar_qualified"]), 0)
        self.assertEqual(len(prio["invalidated_zones"]), 1)
        self.assertEqual(prio["invalidated_zones"][0]["zone_id"], "ZONE_EXPIRED")

    def test_08_h4_and_h1_conflict(self):
        """8. H4 dan H1 conflict: Konflik bias menghasilkan PRIMARY BLOCKER dan Pending Limit diblokir."""
        curr_p = 2650.0
        z = _create_test_zone("ZONE_A", tf="H1", z_type="DEMAND", lo=2640.0, hi=2645.0, score=80.0, grade="A")
        prio = prioritize_radar_zones([z], curr_p, htf_bias="BULLISH", h1_structure="BEARISH", atr=2.0)

        wb = prio.get("wait_breakdown", {})
        self.assertIn("konflik", wb.get("primary_blocker", "").lower())
        self.assertTrue(any("konflik" in pb.lower() for pb in wb.get("pending_limit_blockers", [])))
        self.assertIn("konflik", prio.get("dynamic_wait_reason", "").lower())

    def test_09_h4_and_h1_aligned(self):
        """9. H4 dan H1 aligned: Tidak ada blocker konflik HTF."""
        curr_p = 2650.0
        z = _create_test_zone("ZONE_A", tf="H1", z_type="DEMAND", lo=2640.0, hi=2645.0, score=80.0, grade="A")
        prio = prioritize_radar_zones([z], curr_p, htf_bias="BULLISH", h1_structure="BULLISH", atr=2.0)

        wb = prio.get("wait_breakdown", {})
        self.assertFalse(any("konflik" in pb.lower() for pb in wb.get("pending_limit_blockers", [])))

    def test_10_price_has_not_touched_zone(self):
        """10. Harga belum menyentuh zona: Hanya memblokir pada [ZONE_TOUCH], tidak memicu false failure trigger."""
        curr_p = 2660.0
        z = _create_test_zone("ZONE_A", tf="H1", z_type="DEMAND", lo=2630.0, hi=2635.0, score=80.0, grade="A", dist=25.0)
        m5_stages = {
            "zone_touch": {"status": "WAIT", "msg": "Harga belum menyentuh zona"},
            "liquidity_sweep": {"status": "NOT_CHECKED", "msg": "Belum diperiksa"},
            "structure_trigger": {"status": "NOT_CHECKED", "msg": "Belum diperiksa"},
            "displacement": {"status": "NOT_CHECKED", "msg": "Belum diperiksa"},
            "risk_reward": {"status": "NOT_CHECKED", "msg": "Belum diperiksa"},
        }
        prio = prioritize_radar_zones([z], curr_p, htf_bias="BULLISH", h1_structure="BULLISH", atr=2.0, m5_stages=m5_stages)

        wb = prio.get("wait_breakdown", {})
        cb = wb.get("confirmed_entry_blockers", [])
        self.assertTrue(any("ZONE_TOUCH" in x for x in cb))
        # Jangan memunculkan trigger sweep / CHoCH gagal secara prematur
        self.assertFalse(any("LIQUIDITY_SWEEP" in x for x in cb))
        self.assertFalse(any("STRUCTURE_TRIGGER" in x for x in cb))

    def test_11_price_touched_zone_but_no_sweep(self):
        """11. Harga sudah menyentuh zona tetapi belum ada sweep: Blocker berada pada LIQUIDITY_SWEEP."""
        curr_p = 2634.0
        # Zona tersentuh (curr_p di dalam zona)
        z = _create_test_zone("ZONE_A", tf="H1", z_type="DEMAND", lo=2630.0, hi=2635.0, score=80.0, grade="A", dist=0.0)
        m5_stages = {
            "zone_touch": {"status": "CONFIRMED", "msg": "Zona tersentuh"},
            "liquidity_sweep": {"status": "WAIT", "msg": "Belum ada sweep di area zona"},
            "structure_trigger": {"status": "NOT_CHECKED", "msg": "Belum diperiksa"},
            "displacement": {"status": "NOT_CHECKED", "msg": "Belum diperiksa"},
            "risk_reward": {"status": "NOT_CHECKED", "msg": "Belum diperiksa"},
        }
        prio = prioritize_radar_zones([z], curr_p, htf_bias="BULLISH", h1_structure="BULLISH", atr=2.0, m5_stages=m5_stages)

        wb = prio.get("wait_breakdown", {})
        self.assertIn("sweep", wb.get("primary_blocker", "").lower())
        cb = wb.get("confirmed_entry_blockers", [])
        self.assertTrue(any("LIQUIDITY_SWEEP" in x for x in cb))
        self.assertFalse(any("STRUCTURE_TRIGGER" in x for x in cb))

    def test_12_sweep_confirmed_but_no_choch_bos(self):
        """12. Sweep ada tetapi belum ada CHoCH/BOS/CISD: Blocker berada pada STRUCTURE_TRIGGER."""
        curr_p = 2634.0
        z = _create_test_zone("ZONE_A", tf="H1", z_type="DEMAND", lo=2630.0, hi=2635.0, score=80.0, grade="A", dist=0.0)
        m5_stages = {
            "zone_touch": {"status": "CONFIRMED", "msg": "Zona tersentuh"},
            "liquidity_sweep": {"status": "CONFIRMED", "msg": "Sweep Low M5 valid"},
            "structure_trigger": {"status": "WAIT", "msg": "Belum ada CHoCH/BOS bullish"},
            "displacement": {"status": "NOT_CHECKED", "msg": "Belum diperiksa"},
            "risk_reward": {"status": "NOT_CHECKED", "msg": "Belum diperiksa"},
        }
        prio = prioritize_radar_zones([z], curr_p, htf_bias="BULLISH", h1_structure="BULLISH", atr=2.0, m5_stages=m5_stages)

        wb = prio.get("wait_breakdown", {})
        self.assertIn("choch/bos", wb.get("primary_blocker", "").lower())
        cb = wb.get("confirmed_entry_blockers", [])
        self.assertTrue(any("STRUCTURE_TRIGGER" in x for x in cb))

    def test_13_all_confirmed_entry_conditions_met(self):
        """13. Semua kondisi Confirmed Entry terpenuhi: Menghasilkan status VALID dan Confirmed Candidate terisi."""
        curr_p = 2634.0
        z = _create_test_zone("ZONE_WIN", tf="H1", z_type="DEMAND", lo=2630.0, hi=2635.0, score=85.0, grade="A+", dist=0.0)
        m5_stages = {
            "zone_touch": {"status": "CONFIRMED", "msg": "Zona tersentuh"},
            "liquidity_sweep": {"status": "CONFIRMED", "msg": "Sweep Low M5 valid"},
            "structure_trigger": {"status": "CONFIRMED", "msg": "BOS bull @ 2634.50"},
            "displacement": {"status": "CONFIRMED", "msg": "Displacement 1.4 ATR"},
            "risk_reward": {"status": "CONFIRMED", "msg": "R:R 1:2.0"},
        }
        prio = prioritize_radar_zones([z], curr_p, htf_bias="BULLISH", h1_structure="BULLISH", atr=2.0, m5_stages=m5_stages, validated_entry_zone=z)

        self.assertIsNotNone(prio["best_validated_entry_zone"])
        self.assertEqual(prio["best_validated_entry_zone"]["zone_id"], "ZONE_WIN")
        self.assertIsNotNone(prio["best_confirmed_candidate"])

    def test_14_no_low_risk_candidate(self):
        """14. Tidak ada kandidat Low Risk: Menampilkan NONE tanpa memaksakan kandidat inferior."""
        curr_p = 2650.0
        # Hanya ada zona Grade B atau LTF M5 dengan skor 60.0
        z_ltf = _create_test_zone("ZONE_M5", tf="M5", z_type="DEMAND", lo=2646.0, hi=2648.0, score=62.0, grade="B", dist=2.0)
        prio = prioritize_radar_zones([z_ltf], curr_p, htf_bias="BULLISH", atr=2.0)

        self.assertIsNone(prio["best_low_risk"])
        self.assertEqual(len(prio["low_risk_zones"]), 0)

        # Test output presentation
        rendered = format_scan_presentation(scan={"radar_zones": [z_ltf], "current_price": curr_p, "radar_priority": prio}, dec={"status": "WAIT", "stages": {}})
        self.assertIn("NONE (Tidak ada zona Low Risk yang memenuhi kriteria)", rendered)

    def test_15_atr_data_unavailable(self):
        """15. Data ATR tidak tersedia: Menggunakan fallback jarak poin secara aman tanpa error."""
        curr_p = 2650.0
        z = _create_test_zone("ZONE_A", tf="H1", z_type="DEMAND", lo=2640.0, hi=2645.0, score=80.0, grade="A", dist=5.0)

        # Passing atr=None
        prio = prioritize_radar_zones([z], curr_p, htf_bias="BULLISH", atr=None)
        self.assertIsNotNone(prio)
        self.assertEqual(len(prio["radar_qualified"]), 1)
        z_res = prio["radar_qualified"][0]
        self.assertIsNone(z_res.get("distance_atr_ratio"))
        self.assertEqual(z_res.get("distance"), 5.0)

        # Test classify_zone_risk_tier dengan atr=None
        tier = classify_zone_risk_tier(z, curr_p, htf_bias="BULLISH", atr=None)
        self.assertEqual(tier, "DATA_INCOMPLETE")


if __name__ == "__main__":
    unittest.main()

