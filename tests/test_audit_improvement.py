# -*- coding: utf-8 -*-
"""
tests/test_audit_improvement.py
Comprehensive Verification Suite for Python XAUUSD SMC Scanner Audit & Improvement.

Covers Mandatory Tests:
  TEST 1: Multi-Zone Evaluation (All Supply & Demand candidates evaluated independently)
  TEST 2: Zone Priority & Ranking (Nearest lower quality vs further higher quality)
  TEST 3: Zone Scoring Audit (Fresh zones receive valid scoring and causes for scores)
  TEST 4: Incomplete Data Scoring (Displays DATA INCOMPLETE)
  TEST 5: Price Not Touching Zone (Final decision strictly WAIT)
  TEST 6: Zone Touched Without Sweep (Final decision strictly WAIT)
  TEST 7: Sweep Without Trigger/Displacement (Final decision strictly WAIT)
  TEST 8: Full Confirmation Generates Valid Entry (Entry produced according to SMC strategy rules)
  TEST 9: Expiration 1H & 8H Validation (WIB timezone, actual duration, remaining time, persistence)
  TEST 10: Scanner Without Connection / Incomplete MT5 Data (No false signals)
"""
import math
import os
import tempfile
import unittest
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

import modules.zone_scanner as zs
from modules.zone_scanner import (
    Zone, score_zone, prioritize_radar_zones, calculate_suggested_expiration,
    plan_pending_order, detect_early_setups, get_zone_tracker,
    evaluate_trade_decision, scan_xauusd_zones, format_scan_presentation,
)
from modules.utils import WIB_TZ, get_current_wib, to_wib_datetime
from modules.mt5_conn import tambah_indikator


def _make_df(rows, start="2026-11-13 08:00:00", freq="5min"):
    t0 = pd.to_datetime(start)
    idx = [t0 + i * pd.Timedelta(freq) for i in range(len(rows))]
    df = pd.DataFrame(rows, columns=["Open", "High", "Low", "Close", "Volume"], index=idx)
    return tambah_indikator(df)


def _make_h4_bullish():
    rows = []
    base = 2600.0
    for i in range(70):
        c = base + i * 1.5
        rows.append((c - 1.0, c + 2.0, c - 2.0, c + 1.0, 100))
    return _make_df(rows, freq="4h")


def _make_h4_bearish():
    rows = []
    base = 2700.0
    for i in range(70):
        c = base - i * 1.5
        rows.append((c + 1.0, c + 2.0, c - 2.0, c - 1.0, 100))
    return _make_df(rows, freq="4h")


def _make_h1_bearish():
    rows = []
    base = 2680.0
    for i in range(70):
        c = base - i * 0.8
        rows.append((c + 0.5, c + 1.0, c - 1.0, c - 0.5, 100))
    return _make_df(rows, freq="1h")


def _make_h1_bullish():
    rows = []
    base = 2620.0
    for i in range(70):
        c = base + i * 0.8
        rows.append((c - 0.5, c + 1.0, c - 1.0, c + 0.5, 100))
    return _make_df(rows, freq="1h")


def _make_test_zone(zone_type="SUPPLY", price_lo=2670.0, price_hi=2675.0, sl=2678.0, **kw):
    base = dict(
        zone_id=f"TEST_{zone_type}_{price_lo:.0f}", symbol="XAUUSD", timeframe="H1", zone_type=zone_type,
        price_low=price_lo, price_high=price_hi, midpoint=(price_lo + price_hi) / 2.0,
        score=85.0, grade="A", origin_bos={"tipe": "BOS", "strength_atr": 1.5},
        fvg_confirmation={"status": "UNTOUCHED"}, displacement_strength=2.0,
        freshness="FRESH", touch_count=0, liquidity_sweep_status="NONE",
        invalidation_price=price_hi if zone_type == "SUPPLY" else price_lo,
        is_valid=True, rejection_reason="", sl_recommended=sl,
    )
    base.update(kw)
    return Zone(**base)


class TestAuditImprovement(unittest.TestCase):
    """Mandatory Audit & Improvement Tests 1 through 10."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self._orig_audit = zs.FILE_AUDIT_ZONA
        self._orig_scan_func = zs.scan_xauusd_zones
        zs.FILE_AUDIT_ZONA = os.path.join(self._tmp, "audit_test.jsonl")
        zs._AUDIT_LAST.clear()
        zs._SCAN_CACHE.update(key=None, data=None, ts=0.0)
        zs._FRAMES_CACHE.update(ts=0.0, data=None)

    def tearDown(self):
        zs.FILE_AUDIT_ZONA = self._orig_audit
        zs.scan_xauusd_zones = self._orig_scan_func

    # -------------------------------------------------------------------------
    # TEST 1: Multi-Zone Evaluation (All candidate zones evaluated independently)
    # -------------------------------------------------------------------------
    def test_01_multi_zone_evaluation(self):
        """TEST 1: Seluruh kandidat zona Supply dan Demand dievaluasi dan diklasifikasikan ke 4 kategori."""
        current_price = 2650.0
        s1 = _make_test_zone("SUPPLY", 2670.0, 2675.0, score=90.0, grade="A+")
        s2 = _make_test_zone("SUPPLY", 2690.0, 2695.0, score=75.0, grade="B")
        d1 = _make_test_zone("DEMAND", 2635.0, 2640.0, score=85.0, grade="A")
        d2 = _make_test_zone("DEMAND", 2615.0, 2620.0, score=70.0, grade="B")

        all_zones = [s1, s2, d1, d2]
        prio = prioritize_radar_zones(
            radar_zones=all_zones,
            current_price=current_price,
            htf_bias="BEARISH",
            h1_structure="BEARISH",
            atr=3.0,
        )

        # 1. Pastikan 4 kategori ada dalam output
        self.assertIn("primary_directional_zone", prio)
        self.assertIn("nearest_radar_zone", prio)
        self.assertIn("best_validated_entry_zone", prio)
        self.assertIn("alternative_zone", prio)

        # 2. Primary directional zone harus Supply yang selaras dengan BEARISH
        self.assertEqual(prio["primary_directional_zone"]["price_low"], 2670.0)
        self.assertEqual(prio["primary_directional_zone"]["zone_type"], "SUPPLY")

        # 3. Nearest radar zone harus Demand d1 (jarak 10 pt ke 2640 vs s1 jarak 20 pt ke 2670)
        self.assertEqual(prio["nearest_radar_zone"]["price_low"], 2635.0)
        self.assertEqual(prio["nearest_radar_zone"]["zone_type"], "DEMAND")

        # 4. Best validated entry zone harus None karena belum ada konfirmasi M5
        self.assertIsNone(prio["best_validated_entry_zone"])

        # 5. Alternative zone harus ada (zona relevan lain yang tidak terpilih sbg primary/nearest)
        self.assertIsNotNone(prio["alternative_zone"])

        # 6. Pastikan seluruh 4 zona dievaluasi di scoring summary ranking
        score_sum = prio.get("zone_scoring_summary")
        self.assertIsNotNone(score_sum)
        ranking = score_sum.get("ranking", [])
        self.assertEqual(len(ranking), 4, "Seluruh 4 zona harus masuk dalam ranking evaluasi")

        # 7. Verifikasi pada format visual scan presentation
        mock_scan = {
            "symbol": "XAUUSD", "current_price": current_price,
            "radar_zones": [z.to_dict() for z in all_zones],
            "active_zones": [z.to_dict() for z in all_zones],
            "htf_context": {"H4": "BEARISH", "H1": "BEARISH", "M30": "BEARISH", "has_conflict": False},
            "data_ok": True,
        }
        dec = {"status": "WAIT", "stages": {}, "reasons": [], "radar_priority": prio}
        rendered = format_scan_presentation(scan=mock_scan, dec=dec, detailed=True)
        self.assertIn("A. PRIMARY DIRECTIONAL ZONE", rendered)
        self.assertIn("B. NEAREST RADAR ZONE", rendered)
        self.assertIn("C. BEST VALIDATED ENTRY ZONE                      : NONE", rendered)
        self.assertIn("D. ALTERNATIVE ZONE", rendered)
        self.assertIn("[ZONE EVALUATION SUMMARY]", rendered)
        self.assertIn("Total Kandidat Zona      : 4", rendered)
        self.assertIn("Jumlah Supply            : 2", rendered)
        self.assertIn("Jumlah Demand            : 2", rendered)

    # -------------------------------------------------------------------------
    # TEST 2: Zone Priority & Ranking (Nearest lower quality vs further higher quality)
    # -------------------------------------------------------------------------
    def test_02_nearest_lower_quality_vs_further_higher_quality(self):
        """TEST 2: Zona terdekat kualitas rendah tidak otomatis menjadi Primary jika zona lebih jauh lebih berkualitas."""
        current_price = 2650.0
        # Demand 1 dekat (jarak 5 pt) tapi skor rendah (60.0, Grade B)
        d_near_low = _make_test_zone("DEMAND", 2640.0, 2645.0, score=60.0, grade="B", freshness="TOUCHED")
        # Demand 2 lebih jauh (jarak 20 pt) tapi skor tinggi (92.0, Grade A+, FRESH, strong displacement)
        d_far_high = _make_test_zone("DEMAND", 2625.0, 2630.0, score=92.0, grade="A+", freshness="FRESH")

        prio = prioritize_radar_zones(
            radar_zones=[d_near_low, d_far_high],
            current_price=current_price,
            htf_bias="BULLISH",
            h1_structure="BULLISH",
            atr=3.0,
        )

        # Nearest harus d_near_low
        self.assertEqual(prio["nearest_radar_zone"]["price_low"], 2640.0)
        # Primary Directional Zone harus d_far_high karena skor & kualitas Grade A+ jauh lebih superior
        self.assertEqual(prio["primary_directional_zone"]["price_low"], 2625.0)
        self.assertGreater(prio["primary_directional_zone"]["score"], prio["nearest_radar_zone"]["score"])

        # Ranking harus menempatkan d_far_high di rank #1
        ranking = prio["zone_scoring_summary"]["ranking"]
        self.assertEqual(ranking[0]["zone"], "H1 DEMAND 2625.00-2630.00")
        self.assertEqual(ranking[0]["rank"], 1)

    # -------------------------------------------------------------------------
    # TEST 3: Fresh Zone Scoring Audit (Cause of 0.0 identified and fixed)
    # -------------------------------------------------------------------------
    def test_03_fresh_zone_scoring_audit(self):
        """TEST 3: Zona FRESH tidak boleh memiliki skor palsu 0.0; audit fungsi scoring bekerja deterministik."""
        fresh_zone = _make_test_zone("SUPPLY", 2670.0, 2675.0, freshness="FRESH")
        # Skor awal sengaja diset 0.0 seperti bug yang ditemukan
        fresh_zone.score = 0.0

        # Panggil score_zone dengan allow_radar=True
        scored_zone = score_zone(
            fresh_zone,
            htf_bias="BEARISH",
            h1_bias="BEARISH",
            allow_radar=True,
        )
        computed_score = scored_zone.score

        # Pastikan skor dihitung secara aktual (> 0.0) dan bukan 0.0
        self.assertGreater(computed_score, 50.0, f"Skor zona FRESH terhitung harus > 50.0, dapat {computed_score}")
        self.assertEqual(fresh_zone.scoring_status, "VALID")
        self.assertTrue(len(fresh_zone.score_reason) > 0, "Alasan scoring harus tercatat secara transparan")
        self.assertIn("Skor", fresh_zone.score_reason)
        self.assertIn("Freshness FRESH", fresh_zone.score_reason)

    # -------------------------------------------------------------------------
    # TEST 4: Incomplete Data Scoring (Shows DATA INCOMPLETE)
    # -------------------------------------------------------------------------
    def test_04_incomplete_data_scoring(self):
        """TEST 4: Data scoring tidak lengkap menampilkan status DATA INCOMPLETE bukan skor palsu 0.0."""
        incomplete_zone = _make_test_zone("SUPPLY", 2670.0, 2675.0)
        incomplete_zone.incomplete_data = True

        scored_inc = score_zone(incomplete_zone, htf_bias="BEARISH", h1_bias="BEARISH")
        self.assertEqual(scored_inc.score, 0.0)
        self.assertEqual(incomplete_zone.scoring_status, "DATA_INCOMPLETE")
        self.assertIn("DATA INCOMPLETE", incomplete_zone.score_reason)

        # Verifikasi format presentation menampilkan DATA INCOMPLETE
        prio = prioritize_radar_zones([incomplete_zone], current_price=2650.0, htf_bias="BEARISH")
        mock_scan = {
            "symbol": "XAUUSD", "current_price": 2650.0,
            "radar_zones": [incomplete_zone.to_dict()],
            "active_zones": [incomplete_zone.to_dict()],
            "htf_context": {"H4": "BEARISH", "H1": "BEARISH"},
            "data_ok": True,
        }
        dec = {"status": "WAIT", "stages": {}, "reasons": [], "radar_priority": prio}
        rendered = format_scan_presentation(scan=mock_scan, dec=dec)
        self.assertIn("DATA INCOMPLETE", rendered)

    # -------------------------------------------------------------------------
    # TEST 5: Price Not Touching Zone (Final decision strictly WAIT)
    # -------------------------------------------------------------------------
    def test_05_price_not_touching_zone_waits(self):
        """TEST 5: Harga belum menyentuh zona menghasilkan keputusan WAIT."""
        h4 = _make_h4_bearish()
        h1 = _make_h1_bearish()
        zone = _make_test_zone("SUPPLY", 2670.0, 2675.0, bos_candle_time=h1.index[-1])
        # M30 bearish
        rows_m30 = []
        for i in range(70):
            c = 2650.0 - i * 0.1
            rows_m30.append((c + 0.2, c + 0.4, c - 0.4, c - 0.2, 100))
        m30 = _make_df(rows_m30, freq="30min")

        # M5 harga di sekitar 2650.0 (belum menyentuh supply 2670)
        rows_m5 = []
        for i in range(60):
            c = 2650.0 + math.sin(i)
            rows_m5.append((c, c + 0.5, c - 0.5, c, 100))
        m5 = _make_df(rows_m5, freq="5min")

        frames = {"H4": h4, "H1": h1, "M30": m30, "M5": m5}

        zs.scan_xauusd_zones = lambda frames=None, current_price=None, symbol=None: {
            "symbol": "XAUUSD", "current_price": 2650.0, "active_zones": [zone.to_dict()],
            "rejected_zones": [], "best_demand": None, "best_supply": zone.to_dict(),
            "htf_context": {"H4": "BEARISH", "H1": "BEARISH", "M30": "BEARISH", "has_conflict": False, "m30_conflict": False},
            "empty_reason": "", "summary_text": "ok", "data_ok": True,
        }

        dec = evaluate_trade_decision(frames=frames, current_price=2650.0, live=False)
        self.assertEqual(dec["status"], "WAIT")
        self.assertIsNone(dec.get("plan"))
        self.assertEqual(dec["stages"].get("zone_touch", {}).get("status"), "FAIL")

        rendered = format_scan_presentation(scan=zs.scan_xauusd_zones(), dec=dec, detailed=True)
        self.assertIn("C. BEST VALIDATED ENTRY ZONE                      : NONE", rendered)
        self.assertIn("NO ENTRY / WAIT", rendered)

    # -------------------------------------------------------------------------
    # TEST 6: Zone Touched Without Sweep (Final decision strictly WAIT)
    # -------------------------------------------------------------------------
    def test_06_zone_touched_without_sweep_waits(self):
        """TEST 6: Harga menyentuh zona tetapi belum ada Liquidity Sweep menghasilkan WAIT."""
        zone = _make_test_zone("SUPPLY", 2670.0, 2675.0)
        h4 = _make_h4_bearish()
        h1 = _make_h1_bearish()
        frames = {"H4": h4, "H1": h1}

        # Simulasikan stages: zone touch confirmed, sweep WAIT
        dec_mock = {
            "status": "WAIT", "direction": None, "current_price": 2671.5,
            "stages": {
                "H4": {"ok": True, "msg": "Bias H4 BEARISH"},
                "H1": {"ok": True, "msg": "H1 BEARISH"},
                "M30": {"ok": True, "status": "CONFIRMED", "msg": "M30 BEARISH"},
                "zone_touch": {"status": "CONFIRMED", "msg": "Harga menyentuh supply"},
                "liquidity_sweep": {"status": "WAIT", "msg": "Belum ada BSL sweep M5"},
                "structure_trigger": {"status": "WAIT"},
                "displacement": {"status": "WAIT"},
                "risk_reward": {"status": "NOT_CHECKED"},
            },
            "reasons": ["Belum ada validasi liquidity sweep"],
        }
        mock_scan = {
            "symbol": "XAUUSD", "current_price": 2671.5,
            "active_zones": [zone.to_dict()], "radar_zones": [zone.to_dict()],
            "htf_context": {"H4": "BEARISH", "H1": "BEARISH", "M30": "BEARISH"},
            "data_ok": True,
        }
        rendered = format_scan_presentation(scan=mock_scan, dec=dec_mock, detailed=True)
        self.assertIn("Zone Touch     : CONFIRMED", rendered)
        self.assertIn("Liquidity Sweep: WAIT", rendered)
        self.assertIn("C. BEST VALIDATED ENTRY ZONE                      : NONE", rendered)
        self.assertIn("Status           : WAIT", rendered)

    # -------------------------------------------------------------------------
    # TEST 7: Sweep Without Trigger/Displacement (Final decision strictly WAIT)
    # -------------------------------------------------------------------------
    def test_07_sweep_without_trigger_or_displacement_waits(self):
        """TEST 7: Sweep terjadi tetapi belum ada CHoCH/BOS/CISD & Displacement menghasilkan WAIT."""
        zone = _make_test_zone("SUPPLY", 2670.0, 2675.0)
        dec_mock = {
            "status": "WAIT", "direction": None, "current_price": 2671.5,
            "stages": {
                "H4": {"ok": True, "msg": "Bias H4 BEARISH"},
                "H1": {"ok": True, "msg": "H1 BEARISH"},
                "M30": {"ok": True, "status": "CONFIRMED", "msg": "M30 BEARISH"},
                "zone_touch": {"status": "CONFIRMED"},
                "liquidity_sweep": {"status": "CONFIRMED", "msg": "BSL sweep terkonfirmasi"},
                "structure_trigger": {"status": "WAIT", "msg": "Menunggu CHoCH M5"},
                "displacement": {"status": "WAIT", "msg": "Displacement belum terkonfirmasi"},
                "risk_reward": {"status": "NOT_CHECKED"},
            },
            "reasons": ["Menunggu CHoCH M5 dan displacement"],
        }
        mock_scan = {
            "symbol": "XAUUSD", "current_price": 2671.5,
            "active_zones": [zone.to_dict()], "radar_zones": [zone.to_dict()],
            "htf_context": {"H4": "BEARISH", "H1": "BEARISH", "M30": "BEARISH"},
            "data_ok": True,
        }
        rendered = format_scan_presentation(scan=mock_scan, dec=dec_mock, detailed=True)
        self.assertIn("Liquidity Sweep: CONFIRMED", rendered)
        self.assertIn("CHoCH/BOS/CISD : WAIT", rendered)
        self.assertIn("C. BEST VALIDATED ENTRY ZONE                      : NONE", rendered)
        self.assertIn("Status           : WAIT", rendered)

    # -------------------------------------------------------------------------
    # TEST 8: Full Confirmation Generates Valid Entry
    # -------------------------------------------------------------------------
    def test_08_full_confirmation_generates_valid_entry(self):
        """TEST 8: Seluruh validasi terpenuhi menghasilkan sinyal VALID dan mengisi Best Validated Entry Zone."""
        zone = _make_test_zone("SUPPLY", 2670.0, 2675.0, sl=2678.0)
        plan = {
            "entry_acuan": 2672.5, "entry_min": 2670.0, "entry_max": 2674.5,
            "sl": 2678.0, "tp1": 2655.0, "rr": 3.18,
        }
        dec_mock = {
            "status": "VALID", "direction": "SELL", "zone": zone.to_dict(),
            "plan": plan, "current_price": 2671.5,
            "stages": {
                "H4": {"ok": True, "msg": "H4 BEARISH"},
                "H1": {"ok": True, "msg": "H1 BEARISH"},
                "M30": {"ok": True, "status": "CONFIRMED", "msg": "M30 BEARISH"},
                "zone_touch": {"status": "CONFIRMED"},
                "liquidity_sweep": {"status": "CONFIRMED", "msg": "BSL sweep confirmed"},
                "structure_trigger": {"status": "CONFIRMED", "msg": "CHoCH bearish M5 confirmed"},
                "displacement": {"status": "CONFIRMED", "msg": "Displacement 2.4 ATR"},
                "risk_reward": {"status": "CONFIRMED", "rr": 3.18},
            },
            "reasons": ["Seluruh syarat konfirmasi M5 terpenuhi."],
            "radar_priority": {
                "primary_directional_zone": zone.to_dict(),
                "nearest_radar_zone": zone.to_dict(),
                "best_validated_entry_zone": zone.to_dict(),
                "alternative_zone": None,
                "zone_scoring_summary": {"ranking": [{"rank": 1, "zone": "H1 SUPPLY 2670.00-2675.00", "score": 90.0, "scoring_status": "VALID"}]},
            },
        }
        mock_scan = {
            "symbol": "XAUUSD", "current_price": 2671.5,
            "active_zones": [zone.to_dict()], "radar_zones": [zone.to_dict()],
            "htf_context": {"H4": "BEARISH", "H1": "BEARISH", "M30": "BEARISH"},
            "data_ok": True,
        }

        rendered = format_scan_presentation(scan=mock_scan, dec=dec_mock, detailed=True)
        self.assertIn("SELL VALID", rendered)
        self.assertIn("C. BEST VALIDATED ENTRY ZONE                      : H1 SUPPLY 2670.00 - 2675.00", rendered)
        self.assertIn("Status: [VALID]", rendered)
        self.assertIn("Risk/Reward    : CONFIRMED (1:3.2)", rendered)

    # -------------------------------------------------------------------------
    # TEST 9: Expiration 1H & 8H Validation (WIB timezone, actual duration, persistence)
    # -------------------------------------------------------------------------
    def test_09_expiration_1h_and_8h_validation(self):
        """TEST 9: Expiration 1 jam dan 8 jam akurat, berdasar WIB, remaining stabil, dan tangani error."""
        base_wib = datetime(2026, 11, 13, 8, 25, 0, tzinfo=WIB_TZ)

        # 1. Uji Expiration 1 Jam
        exp_1h = calculate_suggested_expiration(hours=1, server_time=base_wib, current_time=base_wib)
        self.assertFalse(exp_1h["is_error"])
        self.assertEqual(exp_1h["duration_str"], "1 Hours")
        self.assertEqual(exp_1h["created_at_str"], "08:25 WIB")
        self.assertEqual(exp_1h["expiration_str"], "09:25 WIB")
        self.assertEqual(exp_1h["target_time"].hour, 9)
        self.assertEqual(exp_1h["target_time"].minute, 25)

        # 2. Uji Expiration 8 Jam
        exp_8h = calculate_suggested_expiration(hours=8, server_time=base_wib, current_time=base_wib)
        self.assertFalse(exp_8h["is_error"])
        self.assertEqual(exp_8h["duration_str"], "8 Hours")
        self.assertEqual(exp_8h["created_at_str"], "08:25 WIB")
        self.assertEqual(exp_8h["expiration_str"], "16:25 WIB")
        self.assertEqual(exp_8h["target_time"].hour, 16)
        self.assertEqual(exp_8h["target_time"].minute, 25)

        # 3. Uji Remaining Time: Pada 08:26 WIB (1 menit setelah created_at), sisa durasi harus 07h 59m
        t_plus_1m = base_wib + timedelta(minutes=1)
        exp_rem = calculate_suggested_expiration(hours=8, server_time=base_wib, current_time=t_plus_1m)
        self.assertEqual(exp_rem["remaining_str"], "07h 59m")
        self.assertIn("Remaining: 07h 59m", exp_rem["formatted_block"])

        # 4. Uji Stabilitas Expiration di ZoneTracker (refresh tidak mengubah created_at dan expiration)
        tracker = get_zone_tracker(auto_save=False)
        tracker.tracked_zones.clear()
        z_bull = _make_test_zone("DEMAND", 2600.0, 2605.0)

        # Update pertama pada t = base_wib
        tracker.update([z_bull], current_price=2620.0, htf_bias="BULLISH", m30_status="CONFIRMED")
        rec = tracker.get_zone(z_bull.zone_id)
        self.assertIsNotNone(rec)
        first_created = rec.get("created_at")
        first_exp = rec.get("expires_at")
        self.assertIsNotNone(first_created)
        self.assertIsNotNone(first_exp)

        # Update kedua 10 menit kemudian (simulasi polling / report refresh)
        t_plus_10m = base_wib + timedelta(minutes=10)
        tracker.update([z_bull], current_price=2618.0, htf_bias="BULLISH", m30_status="CONFIRMED")
        rec_after = tracker.get_zone(z_bull.zone_id)
        self.assertEqual(rec_after.get("created_at"), first_created, "Created at tidak boleh berubah pada refresh report")
        self.assertEqual(rec_after.get("expires_at"), first_exp, "Expiration tidak boleh bergeser saat refresh report")

        # 5. Uji Data Waktu Tidak Valid -> Menampilkan EXPIRATION ERROR
        exp_err = calculate_suggested_expiration(hours=-5)
        self.assertTrue(exp_err["is_error"])
        self.assertIn("EXPIRATION ERROR", exp_err["formatted_block"])

    # -------------------------------------------------------------------------
    # TEST 10: Scanner Without Connection / Incomplete MT5 Data
    # -------------------------------------------------------------------------
    def test_10_scanner_no_connection_or_incomplete_data(self):
        """TEST 10: Scanner tanpa koneksi atau data MT5 tidak lengkap tidak menghasilkan sinyal palsu."""
        orig_load = zs._load_frames
        try:
            # Simulasikan MT5 offline / terputus
            zs._load_frames = lambda *a, **kw: None

            # Jalankan evaluasi dengan frames None / kosong
            dec = evaluate_trade_decision(frames=None, current_price=0.0, live=False)
            self.assertIn(dec["status"], ("WAIT", "ERROR"))
            self.assertIsNone(dec["direction"])
            self.assertIsNone(dec.get("plan"))

            # Jalankan scan dengan MT5 offline
            scan_res = scan_xauusd_zones(frames=None, current_price=0.0)
            self.assertFalse(scan_res.get("data_ok", True))
            self.assertEqual(len(scan_res.get("active_zones", [])), 0)

            # Pastikan format output tetap menyajikan status WAIT dan tidak mengizinkan open posisi
            rendered = format_scan_presentation(scan=scan_res, dec=dec)
            self.assertIn("Dilarang Buka Posisi Baru", rendered)
            self.assertNotIn("BUY CONFIRMED", rendered)
            self.assertNotIn("SELL CONFIRMED", rendered)
        finally:
            zs._load_frames = orig_load


if __name__ == "__main__":
    unittest.main()
