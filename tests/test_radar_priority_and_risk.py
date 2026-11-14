# -*- coding: utf-8 -*-
"""
tests/test_radar_priority_and_risk.py
Audit & Pengujian Komprehensif:
  1. H4/H1 bearish dengan Supply belum tersentuh.
  2. H4/H1 bearish dengan Demand lebih dekat daripada Supply.
  3. Zona Grade A tetapi terlalu jauh (Label FAR, tidak otomatis READY/CONFIRMED).
  4. Zona dekat tetapi berlawanan dengan bias (Countertrend / Support, bukan entry BUY).
  5. Zona sudah tersentuh tetapi belum ada sweep.
  6. Sweep terjadi tetapi tidak relevan dengan zona.
  7. Sweep dan CHoCH ada tetapi displacement belum valid (WEAK).
  8. Seluruh syarat terpenuhi dan RR minimal 1:1.5 (VALID / CONFIRMED).
  9. Tidak ada zona yang ditemukan (NO RADAR ZONE DETECTED).
  10. H4 neutral (Diagnostic reason spesifik, zona pemantauan range).
"""
import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np

import modules.zone_scanner as zs
from modules.zone_scanner import (
    Zone, prioritize_radar_zones, evaluate_trade_decision,
    format_scan_presentation, plan_pending_order
)
from tests.test_zone_scanner_validation import (
    _make_df, _make_h4_bullish, _make_h4_bearish,
    _make_h1_with_high_target, _make_h1_with_low_target,
    _make_m30_bullish, _make_m30_bearish, _make_test_zone,
    _make_m5_valid_sell_trigger
)


class TestRadarPriorityAndRisk(unittest.TestCase):

    def setUp(self):
        # Pastikan MT5 live loader dinonaktifkan untuk determinisme offline
        self._orig_load = zs._load_frames
        zs._load_frames = lambda ttl=10.0: None

    def tearDown(self):
        zs._load_frames = self._orig_load

    # -------------------------------------------------------------------------
    # 1. H4/H1 bearish dengan Supply belum tersentuh
    # -------------------------------------------------------------------------
    def test_01_bearish_supply_untouched(self):
        """H4/H1 bearish dengan Supply belum tersentuh -> WAIT, Supply jadi Primary Radar, alasan dinamis."""
        current_price = 2650.0
        supply = _make_test_zone("SUPPLY", 2670.0, 2675.0, sl=2678.0, score=88.0, grade="A+")
        
        prio = prioritize_radar_zones(
            radar_zones=[supply],
            current_price=current_price,
            htf_bias="BEARISH",
            h1_structure="BEARISH",
            atr=3.0
        )
        
        self.assertIsNotNone(prio["primary_zone"])
        self.assertEqual(prio["primary_zone"]["zone_type"], "SUPPLY")
        self.assertEqual(prio["primary_zone"]["distance"], 20.0)
        self.assertEqual(prio["primary_zone"]["approach"], "UNTOUCHED")
        self.assertIn("belum mencapai", prio["dynamic_wait_reason"].lower())
        self.assertIn("supply", prio["dynamic_wait_reason"].lower())
        
        # Format scan presentation harus memuat Primary Radar Zone & Final WAIT Reason
        mock_scan = {
            "symbol": "XAUUSD", "current_price": current_price,
            "radar_zones": [supply.to_dict()], "active_zones": [supply.to_dict()],
            "htf_context": {"H4": "BEARISH", "H1": "BEARISH", "M30": "BEARISH", "has_conflict": False},
            "data_ok": True,
        }
        mock_dec = {"status": "WAIT", "stages": {}, "reasons": [], "radar_priority": prio}
        pres = format_scan_presentation(scan=mock_scan, dec=mock_dec, detailed=True)
        
        self.assertIn("Primary Radar Zone", pres)
        self.assertIn("2670.00 - 2675.00", pres)
        self.assertIn("Final WAIT Reason", pres)
        self.assertIn("Missing Entry Conditions", pres)

    # -------------------------------------------------------------------------
    # 2. H4/H1 bearish dengan Demand lebih dekat daripada Supply
    # -------------------------------------------------------------------------
    def test_02_bearish_demand_closer_than_supply(self):
        """H4/H1 bearish dengan Demand lebih dekat daripada Supply -> Supply tetap Primary, Demand jadi Countertrend."""
        current_price = 2650.0
        demand_near = _make_test_zone("DEMAND", 2640.0, 2645.0, sl=2637.0, score=85.0, grade="A")  # dist = 5.0 pt
        supply_far = _make_test_zone("SUPPLY", 2670.0, 2675.0, sl=2678.0, score=88.0, grade="A+")  # dist = 20.0 pt

        prio = prioritize_radar_zones(
            radar_zones=[demand_near, supply_far],
            current_price=current_price,
            htf_bias="BEARISH",
            h1_structure="BEARISH",
            atr=3.0
        )

        # Supply tetap diprioritaskan sebagai Primary Radar Zone meskipun lebih jauh!
        self.assertIsNotNone(prio["primary_zone"])
        self.assertEqual(prio["primary_zone"]["zone_type"], "SUPPLY")
        self.assertEqual(prio["primary_zone"]["distance"], 20.0)

        # Demand diklasifikasikan sebagai Countertrend Zone (bukan entry BUY)
        self.assertIsNotNone(prio["countertrend_zone"])
        self.assertEqual(prio["countertrend_zone"]["zone_type"], "DEMAND")
        self.assertEqual(prio["countertrend_zone"]["distance"], 5.0)

        # Alasan dinamis wajib menyebut bahwa Demand di bawah hanya area reaksi/support
        self.assertIn("bukan sinyal buy", prio["dynamic_wait_reason"].lower())

    # -------------------------------------------------------------------------
    # 3. Zona Grade A tetapi terlalu jauh
    # -------------------------------------------------------------------------
    def test_03_grade_a_zone_too_far_labeled_far(self):
        """Zona Grade A tetapi terlalu jauh -> Diberi label FAR, tetap RADAR, tidak otomatis READY/CONFIRMED."""
        current_price = 2600.0
        # Zona 60 pt dari harga berjalan (dengan ATR 2.5, far threshold = 37.5 pt)
        far_zone = _make_test_zone("SUPPLY", 2660.0, 2665.0, sl=2668.0, score=82.0, grade="A")

        prio = prioritize_radar_zones(
            radar_zones=[far_zone],
            current_price=current_price,
            htf_bias="BEARISH",
            h1_structure="BEARISH",
            atr=2.5
        )

        pz = prio["primary_zone"]
        self.assertIsNotNone(pz)
        self.assertEqual(pz["approach"], "FAR")
        self.assertIn("FAR", pz["priority_labels"])
        # Status zona tetap RADAR (tidak otomatis READY atau CONFIRMED)
        self.assertEqual(pz["status"], "RADAR")

    # -------------------------------------------------------------------------
    # 4. Zona dekat tetapi berlawanan dengan bias
    # -------------------------------------------------------------------------
    def test_04_close_countertrend_zone_not_entry(self):
        """Zona dekat tetapi berlawanan dengan bias -> Diberi label COUNTERTREND, bukan sinyal entry."""
        current_price = 2642.0  # Harga berada di dalam zona Demand
        demand_touch = _make_test_zone("DEMAND", 2640.0, 2645.0, sl=2637.0, score=90.0, grade="A+")

        prio = prioritize_radar_zones(
            radar_zones=[demand_touch],
            current_price=current_price,
            htf_bias="BEARISH",
            h1_structure="BEARISH",
            atr=2.0
        )

        # Karena bias BEARISH, Demand ini adalah COUNTERTREND
        self.assertIsNone(prio["primary_zone"])  # Tidak ada zona pro-trend
        self.assertIsNotNone(prio["countertrend_zone"])
        self.assertIn("COUNTERTREND", prio["countertrend_zone"]["priority_labels"])
        self.assertIn("bukan sinyal buy", prio["dynamic_wait_reason"].lower())

    # -------------------------------------------------------------------------
    # 5. Zona sudah tersentuh tetapi belum ada sweep
    # -------------------------------------------------------------------------
    def test_05_zone_touched_but_no_sweep(self):
        """Zona sudah tersentuh tetapi belum ada sweep -> WAIT, missing conditions memuat LIQUIDITY_SWEEP."""
        current_price = 2672.0  # Menyentuh zona Supply 2670-2675
        supply = _make_test_zone("SUPPLY", 2670.0, 2675.0, sl=2678.0, score=85.0, grade="A")

        m5_stages = {
            "zone_touch": {"status": "CONFIRMED", "msg": "Zona tersentuh"},
            "liquidity_sweep": {"status": "WAIT", "msg": "Belum ada sweep"},
            "structure_trigger": {"status": "WAIT", "msg": "Menunggu CHoCH"},
            "displacement": {"status": "NOT_CHECKED", "msg": ""},
            "risk_reward": {"status": "NOT_CHECKED", "msg": ""},
        }

        prio = prioritize_radar_zones(
            radar_zones=[supply],
            current_price=current_price,
            htf_bias="BEARISH",
            h1_structure="BEARISH",
            atr=2.0,
            m5_stages=m5_stages
        )

        self.assertEqual(prio["primary_zone"]["approach"], "TOUCHED")
        self.assertIn("belum ada konfirmasi liquidity sweep", prio["dynamic_wait_reason"].lower())
        missing_text = " ".join(prio["missing_conditions"])
        self.assertIn("[LIQUIDITY_SWEEP]", missing_text)

    # -------------------------------------------------------------------------
    # 6. Sweep terjadi tetapi tidak relevan dengan zona
    # -------------------------------------------------------------------------
    def test_06_sweep_irrelevant_to_zone(self):
        """Sweep terjadi jauh dari zona -> Relevansi sweep gagal, WAIT."""
        from modules.zone_scanner import is_sweep_relevant_to_zone
        supply = _make_test_zone("SUPPLY", 2670.0, 2675.0, sl=2678.0)

        # Sweep low di 2640.0 (jauh di bawah zona supply 2670-2675)
        bad_sweep = {
            "type": "SWEEP_LOW",
            "level": 2640.0,
            "wick_extreme": 2639.0,
        }
        is_rel, msg = is_sweep_relevant_to_zone(bad_sweep, supply, atr=2.0)
        self.assertFalse(is_rel)
        self.assertIn("tidak sesuai", msg.lower())

        # Sweep high tetapi levelnya di 2620.0 (jauh di bawah zona)
        bad_sweep_high = {
            "type": "SWEEP_HIGH",
            "level": 2620.0,
            "wick_extreme": 2621.0,
        }
        is_rel2, msg2 = is_sweep_relevant_to_zone(bad_sweep_high, supply, atr=2.0)
        self.assertFalse(is_rel2)
        self.assertIn("terlalu jauh", msg2.lower())

    # -------------------------------------------------------------------------
    # 7. Sweep dan CHoCH ada tetapi displacement belum valid
    # -------------------------------------------------------------------------
    def test_07_sweep_and_choch_present_but_displacement_weak(self):
        """Sweep dan CHoCH ada tetapi displacement lemah -> status WEAK, WAIT."""
        current_price = 2672.0
        supply = _make_test_zone("SUPPLY", 2670.0, 2675.0, sl=2678.0)

        m5_stages = {
            "zone_touch": {"status": "CONFIRMED", "msg": "Zona tersentuh"},
            "liquidity_sweep": {"status": "CONFIRMED", "msg": "Sweep terkonfirmasi"},
            "structure_trigger": {"status": "CONFIRMED", "msg": "CHoCH bearish terkonfirmasi"},
            "displacement": {"status": "WEAK", "msg": "Displacement M5 lemah (0.60 ATR < 1.20)"},
            "risk_reward": {"status": "NOT_CHECKED", "msg": ""},
        }

        prio = prioritize_radar_zones(
            radar_zones=[supply],
            current_price=current_price,
            htf_bias="BEARISH",
            h1_structure="BEARISH",
            atr=2.0,
            m5_stages=m5_stages
        )

        self.assertIn("displacement m5 masih lemah", prio["dynamic_wait_reason"].lower())
        missing_text = " ".join(prio["missing_conditions"])
        self.assertIn("[DISPLACEMENT]", missing_text)

    # -------------------------------------------------------------------------
    # 8. Seluruh syarat terpenuhi dan RR minimal 1:1.5
    # -------------------------------------------------------------------------
    def test_08_all_conditions_met_and_rr_valid(self):
        """Seluruh 6 syarat berurutan terpenuhi -> Decision VALID, sinyal CONFIRMED."""
        h4 = _make_h4_bearish()
        h1 = _make_h1_with_low_target()
        m30 = _make_m30_bearish()
        m5 = _make_m5_valid_sell_trigger(zone_lo=2020.0, zone_hi=2024.0)

        frames = {"H4": h4, "H1": h1, "M30": m30, "M5": m5}
        price_entry = 2021.0

        # Inject supply zone yang match dengan M5 trigger
        test_supply = _make_test_zone("SUPPLY", 2020.0, 2024.0, sl=2027.0, bos_candle_time=h1.index[20])
        with patch("modules.zone_scanner.scan_xauusd_zones") as mock_scan:
            mock_scan.return_value = {
                "symbol": "XAUUSD", "current_price": price_entry,
                "radar_zones": [test_supply.to_dict()],
                "active_zones": [test_supply.to_dict()],
                "htf_context": {"H4": "BEARISH", "H1": "BEARISH", "M30": "BEARISH", "has_conflict": False},
                "data_ok": True,
            }
            dec = evaluate_trade_decision(frames=frames, current_price=price_entry)

            self.assertEqual(dec["status"], "VALID")
            self.assertEqual(dec["direction"], "SELL")
            self.assertIsNotNone(dec["plan"])
            self.assertGreaterEqual(dec["plan"]["rr"], 1.5)

    # -------------------------------------------------------------------------
    # 9. Tidak ada zona yang ditemukan
    # -------------------------------------------------------------------------
    def test_09_no_zone_detected(self):
        """Tidak ada zona ditemukan -> Menghasilkan NO RADAR ZONE DETECTED."""
        prio = prioritize_radar_zones(
            radar_zones=[],
            current_price=2650.0,
            htf_bias="BEARISH",
            h1_structure="BEARISH"
        )

        self.assertIsNone(prio["primary_zone"])
        self.assertIn("NO RADAR ZONE DETECTED.", prio["dynamic_wait_reason"])
        self.assertIn("NO RADAR ZONE DETECTED.", prio["missing_conditions"][0])

        mock_scan = {
            "symbol": "XAUUSD", "current_price": 2650.0,
            "radar_zones": [], "active_zones": [],
            "htf_context": {"H4": "BEARISH", "H1": "BEARISH", "M30": "BEARISH", "has_conflict": False},
            "data_ok": True,
        }
        pres = format_scan_presentation(scan=mock_scan, dec={"status": "WAIT", "stages": {}, "reasons": []})
        self.assertIn("NO RADAR ZONE DETECTED.", pres)

    # -------------------------------------------------------------------------
    # 10. H4 neutral
    # -------------------------------------------------------------------------
    def test_10_h4_neutral_diagnostic_and_range_boundaries(self):
        """H4 NETRAL -> Status WAIT, penjelasan diagnostic H4 tampil, zona jadi batas range."""
        current_price = 2650.0
        demand = _make_test_zone("DEMAND", 2635.0, 2640.0, sl=2632.0, score=80.0, grade="A")
        supply = _make_test_zone("SUPPLY", 2660.0, 2665.0, sl=2668.0, score=80.0, grade="A")

        prio = prioritize_radar_zones(
            radar_zones=[demand, supply],
            current_price=current_price,
            htf_bias="NETRAL",
            h1_structure="NETRAL",
            atr=2.0
        )

        # Dalam kondisi netral, alasan menyatakan zona dipantau sebagai batas range reaksi, bukan entry
        self.assertIn("bias h4 netral", prio["dynamic_wait_reason"].lower())
        self.assertIn("batas range", prio["dynamic_wait_reason"].lower())

        mock_scan = {
            "symbol": "XAUUSD", "current_price": current_price,
            "radar_zones": [demand.to_dict(), supply.to_dict()], "active_zones": [demand.to_dict(), supply.to_dict()],
            "htf_context": {"H4": "NETRAL", "H1": "NETRAL", "M30": "NETRAL", "h4_diagnostic": "Ranging sempit konsolidasi", "has_conflict": False},
            "data_ok": True,
        }
        dec = {"status": "WAIT", "stages": {}, "reasons": [], "radar_priority": prio}
        pres = format_scan_presentation(scan=mock_scan, dec=dec, detailed=True)

        self.assertIn("- H4 Bias          : NETRAL (Ranging sempit konsolidasi)", pres)
        self.assertIn("Ranging sempit konsolidasi", pres)
        self.assertIn("Final WAIT Reason", pres)


if __name__ == "__main__":
    unittest.main()

