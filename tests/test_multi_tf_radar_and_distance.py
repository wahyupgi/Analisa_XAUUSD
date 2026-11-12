# -*- coding: utf-8 -*-
"""
tests/test_multi_tf_radar_and_distance.py
Suite Pengujian Komprehensif Multi-Timeframe Radar, Distance Accuracy, Zone Integrity, & MT5 Data Integrity.

Memenuhi 15 skenario pengujian mandiri:
  1. M30 zone terdeteksi dan ditampilkan.
  2. M30 zone ditolak dengan diagnostic reason yang tepat.
  3. Zona H4 dan H1 tetap terdeteksi normal.
  4. Primary Directional Zone dan Nearest Radar Zone berbeda jika zona terdekat adalah countertrend.
  5. Countertrend zone tidak otomatis menjadi entry.
  6. Distance zona di atas harga: Zone Low - Price.
  7. Distance zona di bawah harga: Price - Zone High.
  8. Distance harga di dalam zona: 0.0 dan status INSIDE_ZONE.
  9. Snapshot harga yang sama digunakan oleh seluruh zona.
  10. Akurasi touch count berdasarkan interaksi harga terhadap batas zona.
  11. Status mitigasi FVG (UNTOUCHED, PARTIALLY_MITIGATED, FULLY_MITIGATED).
  12. Pengaruh riwayat pengujian terhadap Grade dan Score.
  13. Data MT5 tidak tersedia -> DATA UNAVAILABLE.
  14. Gemini tidak dapat mengubah keputusan teknikal Python.
  15. Pending order tidak dimodifikasi atau dieksekusi otomatis.
"""
import math
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd

import modules.zone_scanner as zs
from modules.zone_scanner import (
    Zone, calculate_zone_distance, calculate_zone_approach,
    prioritize_radar_zones, score_zone, detect_fvgs, detect_order_blocks,
    scan_xauusd_zones, evaluate_trade_decision, format_scan_presentation,
    format_diagnostic_report, plan_pending_order
)
from modules.mt5_conn import tambah_indikator, cek_posisi_terbuka_mt5, analisa_akun_dan_lot
from tests.test_zone_scanner_validation import (
    _make_df, _make_h4_bullish, _make_h4_bearish,
    _make_h1_with_high_target, _make_h1_with_low_target,
    _make_m30_bullish, _make_m30_bearish, _make_test_zone
)


def _generate_synthetic_ob_data(bullish: bool = True, n_bars: int = 70, base_price: float = 2000.0) -> pd.DataFrame:
    """Membuat data OHLC sintetis dengan Order Block, FVG, dan BOS yang jelas."""
    rows = []
    # Fase 1: Flat/oscillation
    for i in range(30):
        c = base_price + 0.5 * math.sin(i / 2.0)
        rows.append((c - 0.5, c + 1.0, c - 1.0, c + 0.2, 100))

    if bullish:
        # Swing High di bar 25 (level BOS yang akan ditembus)
        rows[25] = (base_price + 3.0, base_price + 5.0, base_price + 2.5, base_price + 4.5, 150)
        # Swing Low lawan (origin) di bar 28
        rows[28] = (base_price - 1.0, base_price + 0.5, base_price - 3.0, base_price - 2.5, 200)
        # OB Bearish candle lawan sebelum impuls naik (bar 29)
        ob_lo, ob_hi = base_price - 2.8, base_price - 0.5
        rows[29] = (ob_hi, ob_hi + 0.2, ob_lo, ob_lo + 0.3, 300)
        # Impuls Bullish dengan FVG (bar 30, 31, 32)
        # Bar 30: Strong Bullish
        rows.append((ob_lo + 0.5, base_price + 2.0, ob_lo + 0.2, base_price + 1.8, 500))
        # Bar 31: Displacement + FVG gap
        rows.append((base_price + 2.0, base_price + 6.0, base_price + 1.9, base_price + 5.8, 800))
        # Bar 32: BOS close > 2005.0 (level bar 25)
        rows.append((base_price + 5.8, base_price + 9.0, base_price + 5.5, base_price + 8.5, 900))
        # Bar 33..70: Pullback di atas zona
        curr = base_price + 8.5
        for i in range(len(rows), n_bars):
            curr += 0.2 * math.cos(i)
            rows.append((curr - 0.4, curr + 0.8, curr - 0.8, curr + 0.1, 150))
    else:
        # Swing Low di bar 25 (level BOS bearish)
        rows[25] = (base_price - 3.0, base_price - 2.5, base_price - 5.0, base_price - 4.5, 150)
        # Swing High lawan (origin) di bar 28
        rows[28] = (base_price + 1.0, base_price + 3.0, base_price - 0.5, base_price + 2.5, 200)
        # OB Bullish candle lawan sebelum impuls turun (bar 29)
        ob_lo, ob_hi = base_price + 0.5, base_price + 2.8
        rows[29] = (ob_lo, ob_hi, ob_lo - 0.2, ob_hi - 0.3, 300)
        # Impuls Bearish dengan FVG (bar 30, 31, 32)
        rows.append((ob_hi - 0.5, ob_hi - 0.2, base_price - 2.0, base_price - 1.8, 500))
        rows.append((base_price - 1.8, base_price - 1.7, base_price - 6.0, base_price - 5.8, 800))
        rows.append((base_price - 5.8, base_price - 5.5, base_price - 9.0, base_price - 8.5, 900))
        curr = base_price - 8.5
        for i in range(len(rows), n_bars):
            curr -= 0.2 * math.cos(i)
            rows.append((curr + 0.4, curr + 0.8, curr - 0.8, curr - 0.1, 150))

    return _make_df(rows, freq="30min")


class TestMultiTFRadarAndDistance(unittest.TestCase):

    def setUp(self):
        self._orig_load = zs._load_frames
        zs._load_frames = lambda ttl=10.0: None

    def tearDown(self):
        zs._load_frames = self._orig_load

    # -------------------------------------------------------------------------
    # 1. M30 zone terdeteksi dan ditampilkan
    # -------------------------------------------------------------------------
    def test_01_m30_zone_detected_and_displayed(self):
        """1. M30 zone terdeteksi oleh engine SMC Python dan ditampilkan pada radar."""
        m30_df = _generate_synthetic_ob_data(bullish=True, n_bars=70, base_price=2600.0)
        h4_df = _make_h4_bullish()
        h1_df = _make_h1_with_high_target()
        frames = {"H4": h4_df, "H1": h1_df, "M30": m30_df}

        curr_price = float(m30_df["Close"].iloc[-1])
        scan = scan_xauusd_zones(frames=frames, current_price=curr_price)

        self.assertTrue(scan["data_ok"])
        by_tf = scan["diagnostics"].get("by_timeframe", {})
        self.assertIn("M30", by_tf, "M30 harus masuk ke dalam analisis per-timeframe")
        self.assertGreater(by_tf["M30"]["ob_candidates"], 0, "M30 harus memiliki kandidat OB yang diperiksa")

        # Cek apakah zona M30 tercatat di radar_zones atau inspected_zones_log
        iz_log = scan["diagnostics"].get("inspected_zones_log", [])
        m30_records = [r for r in iz_log if r["source_timeframe"] == "M30"]
        self.assertGreater(len(m30_records), 0, "Inspected zones log wajib memuat rekaman zona dari M30")

        # Verifikasi laporan diagnostik memuat M30
        diag_rep = format_diagnostic_report(scan)
        self.assertIn("M30", diag_rep)

    # -------------------------------------------------------------------------
    # 2. M30 zone ditolak dengan diagnostic reason yang tepat
    # -------------------------------------------------------------------------
    def test_02_m30_zone_rejected_with_diagnostic_reason(self):
        """2. M30 zone yang gagal hard validation ditolak dan dicatat beserta alasan spesifik."""
        # Buat data M30 dengan OB yang terdeteksi tetapi kemudian ditembus (invalidated)
        m30_df = _generate_synthetic_ob_data(bullish=True, n_bars=60, base_price=2000.0)
        # Tambahkan candle yang menembus batas bawah zona (ob_lo ~ 1997.2)
        rows_inv = [
            (2005.0, 2005.5, 1990.0, 1992.0, 500) for _ in range(10)
        ]
        t0 = m30_df.index[-1]
        idx_inv = [t0 + (i + 1) * pd.Timedelta("30min") for i in range(len(rows_inv))]
        df_inv = pd.DataFrame(rows_inv, columns=["Open", "High", "Low", "Close", "Volume"], index=idx_inv)
        bad_m30 = tambah_indikator(pd.concat([m30_df, df_inv]))

        frames = {"H4": _make_h4_bullish(), "H1": _make_h1_with_high_target(), "M30": bad_m30}
        scan = scan_xauusd_zones(frames=frames, current_price=1992.0)

        self.assertTrue(scan["data_ok"])
        iz_log = scan["diagnostics"].get("inspected_zones_log", [])
        self.assertGreater(len(iz_log), 0)

        # Seluruh zona yang ditolak wajib memiliki alasan penolakan spesifik
        rejected_cands = [rz for rz in iz_log if not rz["is_valid"]]
        self.assertGreater(len(rejected_cands), 0)
        for rz in rejected_cands:
            self.assertTrue(rz["rejection_reason"])
            self.assertNotEqual(rz["rejection_reason"], "PASSED")

        # Pastikan laporan diagnostik memuat alasan penolakan
        diag_rep = format_diagnostic_report(scan)
        self.assertIn("BREAKDOWN FILTER & PENOLAKAN", diag_rep)

    # -------------------------------------------------------------------------
    # 3. Zona H4 dan H1 tetap terdeteksi normal
    # -------------------------------------------------------------------------
    def test_03_h4_and_h1_zones_detected_normally(self):
        """3. Zona H4 dan H1 tetap terdeteksi secara normal dan dicatat di diagnostics."""
        h4 = _make_h4_bearish()
        h1 = _make_h1_with_low_target()
        frames = {"H4": h4, "H1": h1}
        scan = scan_xauusd_zones(frames=frames, current_price=2000.0)

        self.assertTrue(scan["data_ok"])
        by_tf = scan["diagnostics"].get("by_timeframe", {})
        self.assertIn("H4", by_tf)
        self.assertIn("H1", by_tf)
        self.assertGreater(by_tf["H4"]["bars"], 0)
        self.assertGreater(by_tf["H1"]["bars"], 0)

    # -------------------------------------------------------------------------
    # 4. Primary Directional Zone dan Nearest Radar Zone berbeda
    # -------------------------------------------------------------------------
    def test_04_primary_directional_and_nearest_differ(self):
        """4. Primary Directional Zone dan Nearest Radar Zone terpisah jika zona terdekat adalah countertrend."""
        price = 2650.0
        # Countertrend Demand lebih dekat (jarak 5 pt)
        demand_near = _make_test_zone("DEMAND", 2640.0, 2645.0, sl=2637.0, score=85.0, grade="A")
        # Pro-trend Supply lebih jauh (jarak 20 pt)
        supply_far = _make_test_zone("SUPPLY", 2670.0, 2675.0, sl=2678.0, score=90.0, grade="A+")

        prio = prioritize_radar_zones(
            radar_zones=[demand_near, supply_far],
            current_price=price,
            htf_bias="BEARISH",
            h1_structure="BEARISH",
            atr=2.5
        )

        pz = prio["primary_directional_zone"]
        nz = prio["nearest_radar_zone"]
        bz = prio["best_entry_monitoring_zone"]

        self.assertIsNotNone(pz)
        self.assertIsNotNone(nz)
        self.assertEqual(pz["zone_type"], "SUPPLY")
        self.assertEqual(nz["zone_type"], "DEMAND")
        self.assertNotEqual(pz["zone_type"], nz["zone_type"])
        self.assertEqual(pz["distance"], 20.0)
        self.assertEqual(nz["distance"], 5.0)
        self.assertTrue(nz["is_counter"])
        self.assertFalse(pz["is_counter"])

    # -------------------------------------------------------------------------
    # 5. Countertrend zone tidak otomatis menjadi entry
    # -------------------------------------------------------------------------
    def test_05_countertrend_zone_not_auto_entry(self):
        """5. Countertrend zone tidak otomatis menjadi entry BUY/SELL tanpa konfirmasi struktur."""
        price = 2642.0  # Harga masuk ke dalam demand
        demand = _make_test_zone("DEMAND", 2640.0, 2645.0, sl=2637.0, score=88.0, grade="A+")

        prio = prioritize_radar_zones(
            radar_zones=[demand],
            current_price=price,
            htf_bias="BEARISH",
            h1_structure="BEARISH",
            atr=2.0
        )

        self.assertIsNone(prio["primary_directional_zone"])
        self.assertIsNone(prio["best_entry_monitoring_zone"])
        self.assertIsNotNone(prio["countertrend_zone"])
        self.assertIn("COUNTERTREND", prio["countertrend_zone"]["priority_labels"])
        self.assertIn("bukan sinyal buy", prio["dynamic_wait_reason"].lower())

    # -------------------------------------------------------------------------
    # 6. Distance zona di atas harga: Zone Low - Price
    # -------------------------------------------------------------------------
    def test_06_distance_above_price(self):
        """6. Zona di atas harga: Distance = Zone Low - Current Price dan status = ABOVE_PRICE."""
        z_low = 2670.0
        z_high = 2675.0
        price = 2650.0

        res = calculate_zone_distance(z_low, z_high, price, digits=2)
        expected_dist = round(z_low - price, 2)

        self.assertEqual(res.distance, expected_dist)
        self.assertEqual(res.distance, 20.0)
        self.assertEqual(res.status, "ABOVE_PRICE")

        # Uji juga tuple unpacking
        d, st = calculate_zone_distance(z_low, z_high, price, digits=2)
        self.assertEqual(d, 20.0)
        self.assertEqual(st, "ABOVE_PRICE")

    # -------------------------------------------------------------------------
    # 7. Distance zona di bawah harga: Price - Zone High
    # -------------------------------------------------------------------------
    def test_07_distance_below_price(self):
        """7. Zona di bawah harga: Distance = Current Price - Zone High dan status = BELOW_PRICE."""
        z_low = 2620.0
        z_high = 2635.0
        price = 2650.0

        res = calculate_zone_distance(z_low, z_high, price, digits=2)
        expected_dist = round(price - z_high, 2)

        self.assertEqual(res.distance, expected_dist)
        self.assertEqual(res.distance, 15.0)
        self.assertEqual(res.status, "BELOW_PRICE")

    # -------------------------------------------------------------------------
    # 8. Distance harga di dalam zona: 0.0 dan status INSIDE_ZONE
    # -------------------------------------------------------------------------
    def test_08_distance_inside_zone(self):
        """8. Harga di dalam zona: Distance = 0.0 dan status = INSIDE_ZONE."""
        z_low = 2640.0
        z_high = 2648.0
        price = 2645.0

        res = calculate_zone_distance(z_low, z_high, price, digits=2)

        self.assertEqual(res.distance, 0.0)
        self.assertEqual(res.status, "INSIDE_ZONE")

        # Edge case: tepat di batas low
        res_lo = calculate_zone_distance(z_low, z_high, 2640.0, digits=2)
        self.assertEqual(res_lo.distance, 0.0)
        self.assertEqual(res_lo.status, "INSIDE_ZONE")

        # Edge case: tepat di batas high
        res_hi = calculate_zone_distance(z_low, z_high, 2648.0, digits=2)
        self.assertEqual(res_hi.distance, 0.0)
        self.assertEqual(res_hi.status, "INSIDE_ZONE")

    # -------------------------------------------------------------------------
    # 9. Snapshot harga yang sama digunakan oleh seluruh zona
    # -------------------------------------------------------------------------
    def test_09_single_snapshot_price_across_all_zones(self):
        """9. Seluruh zona menggunakan snapshot harga yang sama dalam satu siklus scanning."""
        test_price = 2650.55
        h4 = _make_h4_bearish()
        h1 = _make_h1_with_low_target()
        frames = {"H4": h4, "H1": h1}

        scan = scan_xauusd_zones(frames=frames, current_price=test_price)

        self.assertEqual(scan["snapshot_price"], 2650.55)
        self.assertEqual(scan["current_price"], 2650.55)
        self.assertTrue(scan.get("snapshot_time"))

        # Verifikasi seluruh rekaman zona menggunakan snapshot harga ini
        prio = scan.get("radar_priority", {})
        for zd in prio.get("all_prioritized", []):
            lo, hi = zd["price_low"], zd["price_high"]
            exp_d, exp_st = calculate_zone_distance(lo, hi, test_price, digits=2)
            self.assertEqual(zd["distance"], exp_d)
            self.assertEqual(zd["direction_status"], exp_st)

    # -------------------------------------------------------------------------
    # 10. Akurasi touch count berdasarkan interaksi harga terhadap batas zona
    # -------------------------------------------------------------------------
    def test_10_touch_count_accuracy(self):
        """10. Touch count dihitung secara akurat berdasarkan interaksi harga terhadap batas zona setelah BOS."""
        rows = [
            # 0..26: flat oscillation
            *[(2000.0, 2002.0, 1999.0, 2000.5, 100) for _ in range(27)],
            # 27: Swing High level 2005.0
            (2003.0, 2005.0, 2002.0, 2004.0, 100),
            # 28: Origin Swing Low & OB candle (1996.0 - 2000.0)
            (2000.0, 2000.0, 1996.0, 1997.0, 150),
            # 29: BOS break candle (close 2006.0 > 2005.0)
            (1997.0, 2006.5, 1997.0, 2006.0, 500),
            # 30: k+1 outside (L=2004.0, H=2006.0)
            (2005.0, 2006.0, 2004.0, 2005.0, 100),
            # 31: k+2 touch 1 (L=1998.0, H=2002.0 -> inside 1996.0-2000.0)
            (2004.0, 2004.0, 1998.0, 1999.0, 100),
            # 32: k+3 still inside (L=1997.0, H=1999.0 -> inside, no new entry)
            (1999.0, 1999.5, 1997.0, 1998.5, 100),
            # 33: k+4 moves outside (L=2002.0, H=2004.0)
            (2001.0, 2004.0, 2002.0, 2003.5, 100),
            # 34: k+5 touch 2 (L=1998.0, H=2001.0 -> inside second entry)
            (2003.0, 2003.0, 1998.0, 1999.0, 100),
        ]
        df = _make_df(rows, freq="30min")
        events = [{"k": 29, "arah": "bull", "origin_idx": 28, "origin_valid": True, "level": 2005.0, "tipe": "BOS"}]
        fvgs = [{"idx": 29, "type": "BULLISH_FVG", "price_low": 2000.0, "price_high": 2003.0, "size_atr": 1.0, "status": "UNTOUCHED"}]
        sweeps = []

        ok, bad = detect_order_blocks(df, events, fvgs, sweeps, "M30")
        all_obs = ok + bad
        self.assertGreater(len(all_obs), 0)
        target_ob = all_obs[0]
        self.assertEqual(target_ob.touch_count, 2)
        self.assertEqual(target_ob.freshness, "TESTED_2X")

    # -------------------------------------------------------------------------
    # 11. Status mitigasi FVG (UNTOUCHED, PARTIALLY_MITIGATED, FULLY_MITIGATED)
    # -------------------------------------------------------------------------
    def test_11_fvg_mitigation_status(self):
        """11. Status mitigasi FVG diverifikasi: UNTOUCHED, PARTIALLY_MITIGATED, FULLY_MITIGATED."""
        # 1. UNTOUCHED: candle berikutnya tidak menyentuh area gap
        rows_untouched = [
            (2000.0, 2002.0, 1999.0, 2001.0, 100),
            (2001.0, 2005.0, 2001.0, 2005.0, 200),
            (2005.0, 2008.0, 2004.5, 2008.0, 200),  # FVG antara 2002.0 dan 2004.5
            (2008.0, 2010.0, 2007.5, 2009.5, 100),  # low 2007.5 > 2004.5
        ]
        fvg_u = detect_fvgs(_make_df(rows_untouched, freq="5min"))
        self.assertGreaterEqual(len(fvg_u), 1)
        self.assertEqual(fvg_u[0]["status"], "UNTOUCHED")

        # 2. PARTIALLY_MITIGATED: candle berikutnya masuk ke dalam gap tetapi belum menembus dasar
        rows_part = [
            (2000.0, 2002.0, 1999.0, 2001.0, 100),
            (2001.0, 2005.0, 2001.0, 2005.0, 200),
            (2005.0, 2008.0, 2004.5, 2008.0, 200),  # FVG 2002.0 - 2004.5
            (2008.0, 2008.5, 2003.0, 2004.0, 100),  # low 2003.0 masuk gap (antara 2002.0 dan 2004.5)
        ]
        fvg_p = detect_fvgs(_make_df(rows_part, freq="5min"))
        self.assertEqual(len(fvg_p), 1)
        self.assertEqual(fvg_p[0]["status"], "PARTIALLY_MITIGATED")

        # 3. FULLY_MITIGATED: candle berikutnya menembus seluruh area gap
        rows_full = [
            (2000.0, 2002.0, 1999.0, 2001.0, 100),
            (2001.0, 2005.0, 2001.0, 2005.0, 200),
            (2005.0, 2008.0, 2004.5, 2008.0, 200),  # FVG 2002.0 - 2004.5
            (2008.0, 2008.5, 2001.0, 2001.5, 100),  # low 2001.0 <= 2002.0 (tembus penuh)
        ]
        fvg_f = detect_fvgs(_make_df(rows_full, freq="5min"))
        self.assertEqual(len(fvg_f), 1)
        self.assertEqual(fvg_f[0]["status"], "FULLY_MITIGATED")

    # -------------------------------------------------------------------------
    # 12. Pengaruh riwayat pengujian terhadap Grade dan Score
    # -------------------------------------------------------------------------
    def test_12_score_affected_by_testing_history(self):
        """12. Skor zona dipengaruhi oleh riwayat pengujian (FRESH=15, TESTED_1X=8, TESTED_2X=3)."""
        z_fresh = _make_test_zone("DEMAND", 2000.0, 2004.0, freshness="FRESH", touch_count=0)
        z_t1 = _make_test_zone("DEMAND", 2000.0, 2004.0, freshness="TESTED_1X", touch_count=1)
        z_t2 = _make_test_zone("DEMAND", 2000.0, 2004.0, freshness="TESTED_2X", touch_count=2)

        score_zone(z_fresh)
        score_zone(z_t1)
        score_zone(z_t2)

        self.assertGreater(z_fresh.score, z_t1.score)
        self.assertGreater(z_t1.score, z_t2.score)
        self.assertEqual(z_fresh.score_breakdown["freshness"], 15.0)
        self.assertEqual(z_t1.score_breakdown["freshness"], 8.0)
        self.assertEqual(z_t2.score_breakdown["freshness"], 3.0)

    # -------------------------------------------------------------------------
    # 13. Data MT5 tidak tersedia -> DATA UNAVAILABLE
    # -------------------------------------------------------------------------
    def test_13_mt5_unavailable_shows_data_unavailable(self):
        """13. Jika MT5 offline atau info akun tidak tersedia, tampilkan pesan standar DATA UNAVAILABLE."""
        with patch("modules.mt5_conn.pastikan_mt5_aktif", return_value=False):
            pos_text = cek_posisi_terbuka_mt5(harga_skrg=2650.0)
            self.assertIn("DATA UNAVAILABLE", pos_text)

            lot_text = analisa_akun_dan_lot(None, None, 2.0, 2650.0)
            self.assertIn("DATA UNAVAILABLE", lot_text)

    # -------------------------------------------------------------------------
    # 14. Gemini tidak dapat mengubah keputusan teknikal Python
    # -------------------------------------------------------------------------
    def test_14_gemini_cannot_alter_python_decision(self):
        """14. Gemini tidak dapat mengubah keputusan teknikal Python (sanitasi output membatalkan BUY/SELL saat Python WAIT)."""
        from modules.ai_gemini import sanitasi_output_gemini
        gemini_raw = """
[SINYAL HIGH RISK] BUY NOW 2650.00
• Rekomendasi : BUY NOW 2650.00
• Area Entry   : 2650.00 - 2652.00
• Stop Loss    : 2640.00
• Target TP    : 2670.00
"""
        dec_wait = {
            "status": "WAIT",
            "direction": None,
            "reasons": ["Harga belum mencapai zona supply"],
        }
        sanitized = sanitasi_output_gemini(gemini_raw, dec=dec_wait)
        self.assertIn("[STATUS TRADING: WAIT — TIDAK ADA REKOMENDASI ENTRY AKTIF]", sanitized)
        self.assertNotIn("BUY NOW", sanitized)
        self.assertIn("• Area Entry   : - (WAIT: Tidak ada rekomendasi entry aktif dari Python Scanner)", sanitized)

    # -------------------------------------------------------------------------
    # 15. Pending order tidak dimodifikasi atau dieksekusi otomatis
    # -------------------------------------------------------------------------
    def test_15_pending_orders_not_auto_executed(self):
        """15. Pending order planner hanya menghasilkan metadata analisis tanpa eksekusi/modifikasi otomatis."""
        z = _make_test_zone("DEMAND", 2000.0, 2004.0, sl=1996.0)
        plan = plan_pending_order(z, current_price=2015.0, frames=None, htf_bias="BULLISH", m30_status="CONFIRMED")

        self.assertTrue(plan["valid"])
        self.assertEqual(plan["order_type"], "BUY LIMIT")
        self.assertGreater(plan["entry_price"], 0.0)
        self.assertGreater(plan["sl"], 0.0)
        self.assertGreaterEqual(plan["rr"], 1.5)

        # Presentation output wajib memuat peringatan bahwa limit dipasang manual oleh user
        mock_scan = {
            "symbol": "XAUUSD", "current_price": 2015.0,
            "radar_zones": [z.to_dict()], "active_zones": [z.to_dict()],
            "htf_context": {"H4": "BULLISH", "H1": "BULLISH", "M30": "BULLISH"},
            "data_ok": True
        }
        mock_dec = {
            "status": "WAIT",
            "stages": {"M30": {"status": "CONFIRMED", "ok": True}},
            "reasons": [],
            "early_setup": {"best_setup": plan}
        }
        pres = format_scan_presentation(scan=mock_scan, dec=mock_dec)
        self.assertIn("pasang limit manual", pres.lower())


if __name__ == "__main__":
    unittest.main()
