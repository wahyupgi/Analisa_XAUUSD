# -*- coding: utf-8 -*-
"""
tests/test_early_setup_and_tracker.py
Unit tests untuk fitur:
Early Setup Detection, Pending Order Planner (BUY LIMIT / SELL LIMIT), dan Zone Tracking System.

Memvalidasi 10 skenario sesuai spesifikasi TASK:
1. Zona terdeteksi sebelum M5 confirmation.
2. BUY LIMIT berhasil dihitung secara objektif.
3. SELL LIMIT berhasil dihitung secara objektif.
4. Zona berubah menjadi TOUCHED saat harga menyentuh zona.
5. Zona berubah menjadi INVALIDATED saat invalidation price tertembus.
6. Zona kedaluwarsa (EXPIRED) saat TTL terlewati.
7. H4 berubah arah membatalkan pending plan.
8. M30 kehilangan momentum (WEAK/CONFLICT) menolak pending setup.
9. Pending plan tidak direkomendasikan saat zona invalid.
10. Zona yang sama tidak menghasilkan duplikasi setup.
"""
import unittest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

import modules.zone_scanner as zs
from modules.zone_scanner import (
    Zone, ZoneTracker, plan_pending_order, detect_early_setups,
    get_zone_tracker, format_scan_presentation, evaluate_trade_decision,
)
from tests.test_zone_scanner_validation import (
    _make_df, _make_h4_bullish, _make_h4_bearish,
    _make_h1_with_high_target, _make_h1_with_low_target,
    _make_m30_bullish, _make_m30_bearish, _make_test_zone,
)


class TestEarlySetupAndTracker(unittest.TestCase):

    def setUp(self):
        # Reset tracker in-memory untuk setiap test agar terisolasi
        self.tracker = ZoneTracker(filepath=None, auto_save=False)

    def test_01_zone_detected_before_m5_confirmation(self):
        """1. Zona terdeteksi oleh Early Setup sebelum konfirmasi M5 terpenuhi."""
        h1 = _make_h1_with_high_target()
        h4 = _make_h4_bullish()
        m30 = _make_m30_bullish()
        # M5 datar tanpa touch / tanpa sweep / tanpa trigger
        rows_m5 = [(2015.0, 2016.0, 2014.0, 2015.0, 50)] * 70
        m5 = _make_df(rows_m5, freq="5min")

        frames = {"H4": h4, "H1": h1, "M30": m30, "M5": m5}
        zone = _make_test_zone("DEMAND", 2000.0, 2004.0, sl=1997.0, bos_candle_time=h1.index[25])

        # A. Conservative decision: Harga di 2015.0 belum sentuh zona (2000-2004) -> status WAIT
        dec = evaluate_trade_decision(frames=frames, current_price=2015.0)
        self.assertEqual(dec["status"], "WAIT")
        self.assertIsNone(dec["plan"])

        # B. Early Setup detection: Mampu mendeteksi zona lebih awal dan menyusun BUY LIMIT
        early_plan = plan_pending_order(zone, current_price=2015.0, frames=frames, htf_bias="BULLISH", m30_status="CONFIRMED")
        self.assertTrue(early_plan["valid"])
        self.assertEqual(early_plan["order_type"], "BUY LIMIT")
        self.assertEqual(early_plan["status"], "READY")
        self.assertEqual(early_plan["entry_price"], 2004.0)

    def test_02_buy_limit_successfully_calculated(self):
        """2. BUY LIMIT berhasil dihitung dengan entry proximal, SL struktural, TP, dan RR >= 1.5."""
        h1 = _make_h1_with_high_target()  # swing target di 2030.5
        frames = {"H1": h1, "H4": _make_h4_bullish()}
        zone = _make_test_zone("DEMAND", 2000.0, 2004.0, sl=1997.0, bos_candle_time=h1.index[25])

        plan = plan_pending_order(zone, current_price=2012.0, frames=frames, htf_bias="BULLISH", m30_status="CONFIRMED")
        self.assertTrue(plan["valid"])
        self.assertEqual(plan["order_type"], "BUY LIMIT")
        self.assertEqual(plan["entry_price"], 2004.0)  # proximal edge Demand
        self.assertEqual(plan["sl"], 1997.0)
        self.assertEqual(plan["risk"], 7.0)  # 2004 - 1997
        self.assertGreater(plan["tp1"], 2004.0)
        self.assertGreaterEqual(plan["rr"], 1.5)
        self.assertEqual(plan["invalidation_price"], 2000.0)
        self.assertTrue(len(plan["entry_reason"]) > 0)
        self.assertIn("jam", plan["suggested_expiration"])

    def test_03_sell_limit_successfully_calculated(self):
        """3. SELL LIMIT berhasil dihitung secara simetris dengan entry proximal, SL, TP, dan RR >= 1.5."""
        h1 = _make_h1_with_low_target()  # swing target di 1989.5
        frames = {"H1": h1, "H4": _make_h4_bearish()}
        zone = _make_test_zone("SUPPLY", 2020.0, 2024.0, sl=2027.0, bos_candle_time=h1.index[25])

        plan = plan_pending_order(zone, current_price=2010.0, frames=frames, htf_bias="BEARISH", m30_status="CONFIRMED")
        self.assertTrue(plan["valid"])
        self.assertEqual(plan["order_type"], "SELL LIMIT")
        self.assertEqual(plan["entry_price"], 2020.0)  # proximal edge Supply
        self.assertEqual(plan["sl"], 2027.0)
        self.assertEqual(plan["risk"], 7.0)  # 2027 - 2020
        self.assertLess(plan["tp1"], 2020.0)
        self.assertGreaterEqual(plan["rr"], 1.5)
        self.assertEqual(plan["invalidation_price"], 2024.0)
        self.assertTrue(len(plan["entry_reason"]) > 0)

    def test_04_zone_transitions_to_touched(self):
        """4. Status zona berubah menjadi TOUCHED saat harga menyentuh zona tanpa mengklaim konfirmasi final."""
        zone = _make_test_zone("DEMAND", 2000.0, 2004.0, sl=1997.0)
        h1 = _make_h1_with_high_target()
        frames = {"H1": h1, "H4": _make_h4_bullish()}

        # Awal: harga di 2012.0 (di luar zona)
        self.tracker.update([zone], current_price=2012.0, frames=frames, htf_bias="BULLISH", m30_status="CONFIRMED")
        rec = self.tracker.tracked_zones[zone.zone_id]
        self.assertEqual(rec["status"], "READY")

        # Harga masuk ke dalam area zona (2002.50)
        alerts = self.tracker.update([zone], current_price=2002.50, frames=frames, htf_bias="BULLISH", m30_status="CONFIRMED")
        rec = self.tracker.tracked_zones[zone.zone_id]
        self.assertEqual(rec["status"], "TOUCHED")
        self.assertTrue(any("menyentuh zona" in a["msg"] for a in alerts))

    def test_05_zone_transitions_to_invalidated(self):
        """5. Status zona berubah menjadi INVALIDATED saat invalidation price tertembus."""
        zone = _make_test_zone("DEMAND", 2000.0, 2004.0, sl=1997.0)  # invalidation_price = 2000.0
        h1 = _make_h1_with_high_target()
        frames = {"H1": h1, "H4": _make_h4_bullish()}

        self.tracker.update([zone], current_price=2005.0, frames=frames, htf_bias="BULLISH", m30_status="CONFIRMED")
        # Harga jebol di bawah 2000.0 (misal 1998.0)
        alerts = self.tracker.update([zone], current_price=1998.0, frames=frames, htf_bias="BULLISH", m30_status="CONFIRMED")
        rec = self.tracker.tracked_zones[zone.zone_id]
        self.assertEqual(rec["status"], "INVALIDATED")
        self.assertIsNone(rec["pending_plan"])
        self.assertTrue(any("INVALID" in a["msg"] for a in alerts))

    def test_06_zone_expires_when_ttl_passed(self):
        """6. Status zona berubah menjadi EXPIRED saat waktu kedaluwarsa terlampaui."""
        zone = _make_test_zone("DEMAND", 2000.0, 2004.0, sl=1997.0)
        t0 = "2026-11-01 00:00:00"
        self.tracker.update([zone], current_price=2015.0, server_time=t0, htf_bias="BULLISH", m30_status="CONFIRMED")
        self.assertEqual(self.tracker.tracked_zones[zone.zone_id]["status"], "READY")

        # Maju 10 jam (melebihi DEFAULT_PENDING_EXPIRATION_HOURS = 8 jam)
        t_future = "2026-11-01 10:01:00"
        alerts = self.tracker.update([zone], current_price=2015.0, server_time=t_future, htf_bias="BULLISH", m30_status="CONFIRMED")
        rec = self.tracker.tracked_zones[zone.zone_id]
        self.assertEqual(rec["status"], "EXPIRED")
        self.assertIsNone(rec["pending_plan"])
        self.assertTrue(any("EXPIRED" in a["msg"] for a in alerts))

    def test_07_h4_changes_direction_invalidates_plan(self):
        """7. Perubahan arah H4 membatalkan status READY zona demand dan menjadikannya INVALIDATED."""
        zone = _make_test_zone("DEMAND", 2000.0, 2004.0, sl=1997.0)
        h1 = _make_h1_with_high_target()
        frames = {"H1": h1, "H4": _make_h4_bullish()}

        # Awal: H4 Bullish -> READY (harga 2015.0 di luar jarak approach)
        self.tracker.update([zone], current_price=2015.0, frames=frames, htf_bias="BULLISH", m30_status="CONFIRMED")
        self.assertEqual(self.tracker.tracked_zones[zone.zone_id]["status"], "READY")

        # H4 berbalik menjadi BEARISH
        alerts = self.tracker.update([zone], current_price=2015.0, frames=frames, htf_bias="BEARISH", m30_status="CONFIRMED")
        rec = self.tracker.tracked_zones[zone.zone_id]
        self.assertEqual(rec["status"], "INVALIDATED")
        self.assertTrue(any("berbalik menjadi BEARISH" in a["msg"] for a in alerts))

    def test_08_m30_loses_momentum_rejects_pending(self):
        """8. M30 kehilangan momentum (WEAK atau CONFLICT) menolak pembuatan pending plan."""
        zone = _make_test_zone("DEMAND", 2000.0, 2004.0, sl=1997.0)
        h1 = _make_h1_with_high_target()
        frames = {"H1": h1, "H4": _make_h4_bullish()}

        # Uji status WEAK
        plan_weak = plan_pending_order(zone, current_price=2010.0, frames=frames, htf_bias="BULLISH", m30_status="WEAK")
        self.assertFalse(plan_weak["valid"])
        self.assertEqual(plan_weak["status"], "WEAK")
        self.assertIn("lemah", plan_weak["rejection_reason"])

        # Uji status CONFLICT
        plan_conflict = plan_pending_order(zone, current_price=2010.0, frames=frames, htf_bias="BULLISH", m30_status="CONFLICT")
        self.assertFalse(plan_conflict["valid"])
        self.assertEqual(plan_conflict["status"], "CONFLICT")
        self.assertIn("berlawanan", plan_conflict["rejection_reason"])

    def test_09_pending_plan_not_recommended_when_zone_invalid(self):
        """9. Pending plan tidak lagi direkomendasikan ketika zona sudah invalid."""
        zone = _make_test_zone("DEMAND", 2000.0, 2004.0, sl=1997.0)
        zone.is_valid = False
        zone.rejection_reason = "Zone broken by previous impulse"

        plan = plan_pending_order(zone, current_price=2010.0, htf_bias="BULLISH", m30_status="CONFIRMED")
        self.assertFalse(plan["valid"])
        self.assertEqual(plan["status"], "INVALID")
        self.assertIn("tidak valid", plan["rejection_reason"])

        # Verifikasi juga pada format_scan_presentation
        mock_scan = {
            "symbol": "XAUUSD", "current_price": 2010.0, "active_zones": [],
            "rejected_zones": [zone.to_dict()], "best_demand": None, "best_supply": None,
            "htf_context": {"H4": "BULLISH", "H1": "BULLISH", "M30": "BULLISH"},
            "empty_reason": "Tidak ada zona valid", "summary_text": "", "data_ok": True,
        }
        mock_dec = {"status": "WAIT", "stages": {}, "reasons": ["Tidak ada zona"]}
        out_text = format_scan_presentation(scan=mock_scan, dec=mock_dec)
        self.assertIn("NO PENDING SETUP", out_text)
        self.assertIn("Entry Price    : -", out_text)

    def test_10_same_zone_no_duplicate_setups(self):
        """10. Pemindaian berulang terhadap zona yang sama tidak menghasilkan duplikasi setup pada tracker."""
        zone = _make_test_zone("DEMAND", 2000.0, 2004.0, sl=1997.0)
        h1 = _make_h1_with_high_target()
        frames = {"H1": h1, "H4": _make_h4_bullish()}

        # Jalankan update 5 kali berturut-turut dengan zona yang sama
        for p in (2015.0, 2014.0, 2012.0, 2011.0, 2010.0):
            self.tracker.update([zone], current_price=p, frames=frames, htf_bias="BULLISH", m30_status="CONFIRMED")

        # Pastikan tetap hanya ada 1 record zona di tracker
        self.assertEqual(len(self.tracker.tracked_zones), 1)
        self.assertEqual(len(self.tracker.get_active_zones()), 1)
        self.assertIn(zone.zone_id, self.tracker.tracked_zones)


if __name__ == "__main__":
    unittest.main()
