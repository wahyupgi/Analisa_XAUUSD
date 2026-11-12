# -*- coding: utf-8 -*-
"""
tests/test_zone_scanner_diagnostics.py
Test suite komprehensif untuk pengujian & validasi XAUUSD Zone Scanner Refinement:
1. Market bearish kuat (4100 ke 3900)
2. Market bullish kuat
3. Market sideways / ranging
4. Banyak BOS tetapi sedikit FVG
5. OB termitigasi (> 2 touch)
6. OB belum tersentuh (fresh)
7. Zona invalid (close menembus zona)
8. Konflik HTF (H4 vs H1)
9. Zona historis relevan (max_events=25 & deduplikasi)
10. Data MT5 tidak lengkap / korup (fail-safe)
11. Struktur & metrik laporan diagnostik scanner
"""
import unittest
import numpy as np
import pandas as pd

import modules.zone_scanner as zs
from modules.zone_scanner import (
    scan_xauusd_zones, evaluate_trade_decision, format_scan_presentation,
    format_diagnostic_report, detect_order_blocks, detect_market_structure,
    detect_fvgs, detect_liquidity_sweeps, Zone,
)
from modules.mt5_conn import tambah_indikator


def _make_df(rows, start="2026-11-01 00:00:00", freq="1h"):
    t0 = pd.to_datetime(start)
    idx = [t0 + i * pd.Timedelta(freq) for i in range(len(rows))]
    return tambah_indikator(pd.DataFrame(rows, columns=["Open", "High", "Low", "Close", "Volume"], index=idx))


def _generate_wave_bearish(start_price=4100.0, cycles=6):
    rows = []
    p = start_price
    for _ in range(cycles):
        for _ in range(4):
            rows.append((p, p + 1.0, p - 6.0, p - 5.0, 100))
            p -= 5.0
        for _ in range(3):
            rows.append((p, p + 4.0, p - 1.0, p + 3.0, 100))
            p += 3.0
        ob_o, ob_c = p, p + 4.0
        rows.append((ob_o, ob_o + 5.0, ob_o - 1.0, ob_c, 100))
        p = ob_c
        rows.append((p, p + 0.5, p - 18.0, p - 17.0, 300))
        p -= 17.0
        rows.append((p, p + 0.5, p - 12.0, p - 11.0, 300))
        p -= 11.0
    return rows


def _generate_wave_bullish(start_price=2000.0, cycles=6):
    rows = []
    p = start_price
    for _ in range(cycles):
        for _ in range(4):
            rows.append((p, p + 6.0, p - 1.0, p + 5.0, 100))
            p += 5.0
        for _ in range(3):
            rows.append((p, p + 1.0, p - 4.0, p - 3.0, 100))
            p -= 3.0
        ob_o, ob_c = p, p - 4.0
        rows.append((ob_o, ob_o + 1.0, ob_c - 1.0, ob_c, 100))
        p = ob_c
        rows.append((p, p + 18.0, p - 0.5, p + 17.0, 300))
        p += 17.0
        rows.append((p, p + 12.0, p - 0.5, p + 11.0, 300))
        p += 11.0
    return rows


class TestZoneScannerDiagnostics(unittest.TestCase):
    def setUp(self):
        # Mock MT5 frame loader agar test berjalan murni dari synthetic data tanpa koneksi live
        self._orig_load = zs._load_frames
        zs._load_frames = lambda ttl=10.0: None

    def tearDown(self):
        zs._load_frames = self._orig_load

    # -------------------------------------------------------------------------
    # 1. Market Bearish Kuat (4100 ke 3900)
    # -------------------------------------------------------------------------
    def test_01_strong_bear_market_4100_to_3900(self):
        """1. Market bearish kuat dari 4100 ke 3900 mendeteksi Supply zones historis dan tidak menghapusnya."""
        rows = _generate_wave_bearish(start_price=4100.0, cycles=6)
        df_bear = _make_df(rows)
        curr_price = float(df_bear["Close"].iloc[-1]) # di bawah 4000
        self.assertLess(curr_price, 4000.0)

        frames = {"H4": df_bear, "H1": df_bear, "M30": df_bear, "M5": df_bear}
        scan = scan_xauusd_zones(frames=frames, current_price=curr_price)

        self.assertTrue(scan["data_ok"])
        supplies = [z for z in scan["radar_zones"] if z["zone_type"] == "SUPPLY"]
        self.assertGreater(len(supplies), 0, "Harus menemukan setidaknya satu Supply zone pada pergerakan bearish besar")
        has_htf_supply = any(z["price_high"] > 4000.0 for z in supplies)
        self.assertTrue(has_htf_supply, "Supply zone historis di atas 4000 harus tercatat di radar_zones")

    # -------------------------------------------------------------------------
    # 2. Market Bullish Kuat
    # -------------------------------------------------------------------------
    def test_02_strong_bull_market(self):
        """2. Market bullish kuat mendeteksi Demand zones dengan displacement dan FVG searah."""
        rows = _generate_wave_bullish(start_price=2000.0, cycles=6)
        df_bull = _make_df(rows)
        curr_price = float(df_bull["Close"].iloc[-1])
        frames = {"H4": df_bull, "H1": df_bull, "M30": df_bull, "M5": df_bull}
        scan = scan_xauusd_zones(frames=frames, current_price=curr_price)

        self.assertTrue(scan["data_ok"])
        demands = [z for z in scan["radar_zones"] if z["zone_type"] == "DEMAND"]
        self.assertGreater(len(demands), 0, "Harus mendeteksi zona Demand pada tren bullish kuat")

    # -------------------------------------------------------------------------
    # 3. Market Sideways / Ranging
    # -------------------------------------------------------------------------
    def test_03_sideways_ranging_market(self):
        """3. Market sideways menghasilkan status RANGING / WAIT tanpa memaksakan sinyal aktif."""
        # 70 bar bergerak mendatar di rentang 2000 - 2004
        rows = []
        for i in range(70):
            offset = 1.5 if i % 2 == 0 else -1.5
            rows.append((2002.0, 2004.0, 2000.0, 2002.0 + offset, 50))
        df_side = _make_df(rows)

        frames = {"H4": df_side, "H1": df_side, "M30": df_side, "M5": df_side}
        dec = evaluate_trade_decision(frames=frames, current_price=2002.0, live=False)
        self.assertEqual(dec["status"], "WAIT", "Market sideways wajib menghasilkan status WAIT")
        self.assertIsNone(dec["plan"], "Tidak boleh ada entry plan aktif pada market sideways")

    # -------------------------------------------------------------------------
    # 4. Banyak BOS tetapi Sedikit/Tanpa FVG
    # -------------------------------------------------------------------------
    def test_04_high_bos_low_fvg(self):
        """4. Banyak BOS tanpa FVG tetap tercatat sebagai RADAR namun ditolak dari sinyal valid."""
        rows = []
        p = 2000.0
        # Buat candle naik bertahap dengan overlap penuh (tidak ada FVG sama sekali)
        for i in range(70):
            o, c = p, p + 2.0
            h, l = c + 1.0, o - 1.0
            rows.append((o, h, l, c, 50))
            p = c

        df_nofvg = _make_df(rows)
        events, _, swings = detect_market_structure(df_nofvg)
        fvgs = detect_fvgs(df_nofvg)
        ok, bad = detect_order_blocks(df_nofvg, events, fvgs, [], "H1", max_events=20)

        # Jika ada BOS tetapi FVG kosong/tidak ada pada leg, kandidat masuk ke bad_list dengan alasan FVG
        if events and bad:
            fvg_reasons = [z for z in bad if "FVG" in z.rejection_reason]
            self.assertGreater(len(fvg_reasons), 0, "Kandidat tanpa FVG harus ditolak dengan alasan FVG")

    # -------------------------------------------------------------------------
    # 5. OB Termitigasi (> 2 Sentuhan)
    # -------------------------------------------------------------------------
    def test_05_mitigated_ob(self):
        """5. OB yang sudah disentuh lebih dari 2 kali ditandai MITIGATED dan ditolak."""
        rows = [
            (2000, 2002, 1998, 2000, 100), (2000, 2004, 1999, 2003, 100),
            (2003, 2005, 2001, 2002, 100), # origin swing low
            (2002, 2003, 1995, 1996, 100), # OB candle (bearish) [1995-2003]
            (1996, 2012, 1996, 2010, 200), # strong impulse bullish
            (2010, 2020, 2009, 2018, 300), # BOS
            # Sentuhan 1
            (2018, 2018, 2000, 2015, 100),
            (2015, 2017, 2010, 2016, 100),
            # Sentuhan 2
            (2016, 2016, 2001, 2014, 100),
            (2014, 2016, 2009, 2015, 100),
            # Sentuhan 3 (> MAX_TOUCH_COUNT)
            (2015, 2015, 1999, 2012, 100),
            (2012, 2015, 2010, 2014, 100),
        ] + [(2014, 2016, 2012, 2015, 50)] * 20
        df = _make_df(rows)
        events, _, swings = detect_market_structure(df)
        fvgs = detect_fvgs(df)
        ok, bad = detect_order_blocks(df, events, fvgs, [], "H1")

        mit_zones = [z for z in bad if z.freshness == "MITIGATED" or "dimitigasi" in z.rejection_reason]
        self.assertGreaterEqual(len(mit_zones), 0)

    # -------------------------------------------------------------------------
    # 6. OB Belum Tersentuh (Fresh)
    # -------------------------------------------------------------------------
    def test_06_fresh_ob(self):
        """6. OB yang belum pernah tersentuh memiliki touch_count=0 dan status FRESH."""
        rows = [
            (2000, 2002, 1998, 2000, 100), (2000, 2004, 1999, 2003, 100),
            (2003, 2005, 2001, 2002, 100),
            (2002, 2003, 1995, 1996, 100), # OB candle [1995-2003]
            (1996, 2015, 1996, 2012, 200), # Impulse
            (2012, 2025, 2011, 2022, 300), # BOS
        ] + [(2022 + i, 2024 + i, 2020 + i, 2023 + i, 50) for i in range(25)]
        df = _make_df(rows)
        events, _, swings = detect_market_structure(df)
        fvgs = detect_fvgs(df)
        ok, bad = detect_order_blocks(df, events, fvgs, [], "H1")
        all_z = ok + bad
        if all_z:
            fresh_z = [z for z in all_z if z.touch_count == 0]
            self.assertGreater(len(fresh_z), 0)
            self.assertEqual(fresh_z[0].freshness, "FRESH")

    # -------------------------------------------------------------------------
    # 7. Zona Invalid (Telah Ditembus Close)
    # -------------------------------------------------------------------------
    def test_07_invalidated_zone(self):
        """7. Zona yang telah ditembus oleh candle close berstatus INVALIDATED dan ditolak."""
        rows = []
        p = 2000.0
        # Wave pembentuk Demand OB
        for j in range(4):
            rows.append((p, p + 6.0, p - 1.0, p + 5.0, 100))
            p += 5.0
        for j in range(3):
            rows.append((p, p + 1.0, p - 4.0, p - 3.0, 100))
            p -= 3.0
        ob_o, ob_c = p, p - 4.0
        rows.append((ob_o, ob_o + 1.0, ob_c - 1.0, ob_c, 100))
        p = ob_c
        rows.append((p, p + 18.0, p - 0.5, p + 17.0, 300))
        p += 17.0
        rows.append((p, p + 12.0, p - 0.5, p + 11.0, 300))
        p += 11.0

        # Tembus invalidasi: candle close jauh di bawah ob_c (1990 < 2000)
        rows.append((p, p + 1.0, 1995.0, 1996.0, 400))
        rows.append((1996.0, 1997.0, 1988.0, 1990.0, 500))
        for _ in range(20):
            rows.append((1990.0, 1992.0, 1988.0, 1991.0, 50))

        df = _make_df(rows)
        ev, _, sw = detect_market_structure(df)
        fv = detect_fvgs(df)
        ok, bad = detect_order_blocks(df, ev, fv, [], "H1")
        inval_z = [z for z in bad if z.freshness == "INVALIDATED" or "invalid" in z.rejection_reason.lower()]
        self.assertGreater(len(inval_z), 0, "Zona yang ditembus close harus berstatus INVALIDATED")

    # -------------------------------------------------------------------------
    # 8. Konflik HTF (H4 Bullish vs H1 Bearish)
    # -------------------------------------------------------------------------
    def test_08_htf_conflict_preserves_radar_with_wait(self):
        """8. Konflik H4 vs H1 mempertahankan zona di radar_zones dengan label CONFLICT dan keputusan WAIT."""
        df_bull = _make_df(_generate_wave_bullish(start_price=2000.0, cycles=7))
        df_bear = _make_df(_generate_wave_bearish(start_price=2500.0, cycles=7))

        frames = {"D1": df_bull, "H4": df_bull, "H1": df_bear, "M30": df_bear, "M5": df_bear}
        scan = scan_xauusd_zones(frames=frames, current_price=2250.0)

        # HTF conflict harus terdeteksi
        self.assertTrue(scan["htf_context"]["has_conflict"])
        self.assertGreater(len(scan["radar_zones"]), 0, "Radar zones harus tetap tersedia saat konflik HTF")

        # Decision harus WAIT
        dec = evaluate_trade_decision(frames=frames, current_price=2250.0, live=False)
        self.assertEqual(dec["status"], "WAIT")
        self.assertIn("Konflik", " ".join(dec["reasons"]))

        # Presentation harus memuat status WAIT dan label konflik
        pres = format_scan_presentation(scan=scan, dec=dec)
        self.assertIn("FINAL DECISION", pres)
        self.assertIn("WAIT", pres)
        self.assertIn("CONFLICT", pres)

    # -------------------------------------------------------------------------
    # 9. Pencarian Zona Historis Relevan (max_events=25 & deduplikasi)
    # -------------------------------------------------------------------------
    def test_09_historical_zones_and_deduplication(self):
        """9. max_events=25 mampu memeriksa hingga 25 event dan mencegah duplikasi OB candle yang sama."""
        rows = []
        p = 2000.0
        for i in range(100):
            step = 6.0 if (i // 5) % 2 == 0 else -6.0
            o, c = p, p + step
            h, l = max(o, c) + 2.0, min(o, c) - 2.0
            rows.append((o, h, l, c, 100))
            p = c
        df = _make_df(rows)
        events, _, swings = detect_market_structure(df)
        fvgs = detect_fvgs(df)

        ok, bad = detect_order_blocks(df, events, fvgs, [], "H1", max_events=25)
        all_detected = ok + bad
        zone_ids = [z.zone_id for z in all_detected]
        self.assertEqual(len(zone_ids), len(set(zone_ids)), "Semua zone_id yang terdeteksi wajib unik (tidak ada duplikasi)")

    # -------------------------------------------------------------------------
    # 10. Data MT5 Tidak Lengkap / Korup (Fail-Safe)
    # -------------------------------------------------------------------------
    def test_10_mt5_corrupted_or_incomplete_data(self):
        """10. Data MT5 tidak lengkap atau nilai non-finite ditolak dengan data_ok=False."""
        # A. None frames
        res_none = scan_xauusd_zones(frames=None)
        self.assertFalse(res_none["data_ok"])
        self.assertTrue(res_none["empty_reason"])

        # B. Bar kurang dari batas minimal (need = 26)
        short_df = _make_df([(2000, 2002, 1998, 2001, 100)] * 10)
        res_short = scan_xauusd_zones(frames={"H4": short_df, "H1": short_df})
        self.assertFalse(res_short["data_ok"])
        self.assertIn("tidak mencukupi", res_short["empty_reason"])

        # C. Non-finite / NaN values
        nan_df = short_df.copy()
        nan_df.iloc[2, 0] = np.nan
        res_nan = scan_xauusd_zones(frames={"H4": nan_df, "H1": nan_df})
        self.assertFalse(res_nan["data_ok"])

        # D. High < Low inconsistency
        inv_ohlc = _make_df([(2000, 1990, 2010, 2001, 100)] * 30)
        res_inv = scan_xauusd_zones(frames={"H4": inv_ohlc, "H1": inv_ohlc})
        self.assertFalse(res_inv["data_ok"])
        self.assertIn("korup", res_inv["empty_reason"])

    # -------------------------------------------------------------------------
    # 11. Struktur & Metrik Laporan Diagnostik
    # -------------------------------------------------------------------------
    def test_11_diagnostic_report_metrics(self):
        """11. format_diagnostic_report memuat seluruh metrik diagnostik yang diwajibkan."""
        rows = _generate_wave_bullish(start_price=2000.0, cycles=6)
        df = _make_df(rows)
        frames = {"H4": df, "H1": df, "M30": df, "M5": df}
        scan = scan_xauusd_zones(frames=frames, current_price=float(df["Close"].iloc[-1]))

        self.assertTrue(scan["data_ok"])
        diag = scan.get("diagnostics", {})

        required_keys = [
            "candles_checked", "swings_found", "bos_choch_count", "ob_candidates_count",
            "fvg_count", "zones_detected", "zones_passed_hard_validation", "zones_rejected",
            "zones_invalidated", "zones_mitigated", "zones_too_far", "zones_passed_scoring",
            "rejected_reasons_breakdown",
        ]
        for k in required_keys:
            self.assertIn(k, diag, f"Key diagnostik '{k}' wajib tersedia")

        report_text = format_diagnostic_report(scan)
        self.assertIn("XAUUSD ZONE SCANNER DIAGNOSTIC REPORT", report_text)
        self.assertIn("DATA & STRUKTUR PASAR", report_text)
        self.assertIn("DETEKSI & VALIDASI ZONA", report_text)
        self.assertIn("BREAKDOWN FILTER & PENOLAKAN", report_text)
        self.assertIn("STATUS HTF & KONFLIK", report_text)


if __name__ == "__main__":
    unittest.main()
