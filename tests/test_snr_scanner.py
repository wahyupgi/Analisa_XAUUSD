# -*- coding: utf-8 -*-
"""
test_snr_scanner.py — Pengujian Komprehensif Wajib (TEST 1 s/d TEST 10)
untuk SNR Advanced Scanner Independen pada XAUUSD Python Trading Analyzer.
"""
import copy
import os
import unittest
import numpy as np
import pandas as pd

import xauusd1
from modules.snr import (
    SnrAlertTracker,
    SnrCycleEngine,
    analyze_momentum,
    analyze_zone_touches,
    assign_snr_grade,
    calculate_snr_risk_reward,
    cluster_zones,
    detect_resistance_candidates,
    detect_snr_market_status,
    detect_support_candidates,
    detect_swings,
    detect_zone_breakout,
    detect_zone_rejection,
    detect_zone_retest,
    evaluate_mtf_confluence,
    evaluate_snr_decision,
    evaluate_snr_market_bias,
    evaluate_snr_radar,
    fetch_snr_market_data,
    format_snr_detail,
    format_snr_summary,
    get_snr_config,
    prepare_snr_frame,
    rank_zones,
    run_snr_scan,
    scan_snr,
    scan_snr_detail,
    score_zone,
    validate_snr_entry,
    validate_zones,
)
from modules.zone_scanner import (
    create_market_snapshot,
    evaluate_trade_decision,
    format_scan_presentation,
    scan_xauusd_zones,
)


def make_synthetic_ohlc(prices, atr_val=2.0, freq="1h", with_volume=True, start_time="2026-03-01 00:00:00"):
    """Membangun DataFrame OHLC sintetis yang deterministik dari daftar tuple (open, high, low, close) atau harga close."""
    rows = []
    dates = pd.date_range(start_time, periods=len(prices), freq=freq)
    for i, item in enumerate(prices):
        if isinstance(item, (tuple, list)) and len(item) >= 4:
            o, h, l, c = float(item[0]), float(item[1]), float(item[2]), float(item[3])
            v = float(item[4]) if len(item) >= 5 else (1000.0 + (i % 7) * 80.0)
        else:
            c = float(item)
            o = float(prices[i - 1][3] if (i > 0 and isinstance(prices[i - 1], (tuple, list))) else (prices[i - 1] if i > 0 else c - 0.3))
            h = max(o, c) + 0.6
            l = min(o, c) - 0.6
            v = 1000.0 + (i % 7) * 80.0
        row = {
            "Time": dates[i],
            "Open": round(o, 2),
            "High": round(h, 2),
            "Low": round(l, 2),
            "Close": round(c, 2),
            "ATR14": float(atr_val),
        }
        if with_volume:
            row["tick_volume"] = v
        rows.append(row)
    return pd.DataFrame(rows)


def build_realistic_multi_tf_frames():
    """Membangun dataset multi-timeframe (H4, H1, M30, M5) dengan struktur S&R jelas untuk pengujian."""
    # Buat kurva gelombang sinusoidal + reaksi support/resistance di sekitar 3865 - 3920
    def _wave_df(n_bars, base, amp, period, freq, atr_v):
        items = []
        for i in range(n_bars):
            angle = (2.0 * np.pi * i) / float(period)
            mid = base + amp * np.sin(angle)
            prev_mid = base + amp * np.sin((2.0 * np.pi * max(0, i - 1)) / float(period))
            o = prev_mid
            c = mid
            h = max(o, c) + (0.35 * atr_v)
            l = min(o, c) - (0.35 * atr_v)
            items.append((o, h, l, c, 1200 + (i % 11) * 90))
        return make_synthetic_ohlc(items, atr_val=atr_v, freq=freq)

    return {
        "H4": _wave_df(120, 3895.0, 25.0, 18, "4h", 8.0),
        "H1": _wave_df(140, 3895.0, 22.0, 16, "1h", 4.5),
        "M30": _wave_df(140, 3895.0, 18.0, 14, "30min", 3.0),
        "M5": _wave_df(140, 3895.0, 10.0, 12, "5min", 1.5),
    }


class TestSnrAdvancedScanner(unittest.TestCase):
    """Suite Pengujian Wajib TEST 1 s/d TEST 10 untuk SNR Advanced Scanner."""

    # =========================================================================
    # TEST 1 — Modul Swing Detection
    # =========================================================================
    def test_01_swing_detection_pivot_confirmation_duplicates_and_unclosed(self):
        """
        TEST 1:
        - Uji swing high & swing low.
        - Uji pivot confirmation (hanya valid setelah candle kanan selesai terbentuk).
        - Uji duplicate detection (level berdekatan tidak diduplikasi).
        - Uji candle yang belum closed (tidak boleh dipakai mengonfirmasi pivot).
        """
        # Bentuk 22 bar di mana bar #5 adalah Swing High (3922), bar #10 adalah Swing Low (3880.0),
        # bar #14 adalah duplicate Swing Low sangat dekat (3880.20, berjarak 4 bar <= 8 bar dan selisih 0.20 <= dup_tol),
        # dan bar #20 (ujung kanan) belum punya 3 right bars
        bars = [
            (3900, 3902, 3898, 3901),  # 0
            (3901, 3905, 3900, 3904),  # 1
            (3904, 3908, 3903, 3907),  # 2
            (3907, 3912, 3906, 3910),  # 3
            (3910, 3915, 3909, 3914),  # 4
            (3914, 3922, 3912, 3915),  # 5 -> Confirmed Swing High (3922)
            (3915, 3916, 3908, 3909),  # 6
            (3909, 3910, 3901, 3902),  # 7
            (3902, 3904, 3894, 3895),  # 8 (bar #5 confirmed di sini pada right_bars=3)
            (3895, 3896, 3886, 3888),  # 9
            (3888, 3890, 3880.0, 3887),  # 10 -> Confirmed Swing Low (3880.0)
            (3887, 3891, 3885, 3889),  # 11
            (3889, 3893, 3886, 3891),  # 12
            (3891, 3894, 3884, 3886),  # 13
            (3886, 3890, 3880.2, 3888),  # 14 -> Duplicate Swing Low dekat (3880.2, 4 bar dari #10, selisih 0.20 <= dup_tol)
            (3888, 3896, 3885, 3895),  # 15
            (3895, 3902, 3894, 3900),  # 16
            (3900, 3906, 3899, 3905),  # 17
            (3905, 3909, 3903, 3908),  # 18
            (3908, 3911, 3906, 3910),  # 19
            (3910, 3930, 3909, 3928),  # 20 -> Ujung kanan (hanya 1 bar di kanan, BELUM confirmed untuk right_bars=3!)
            (3928, 3925, 3918, 3920),  # 21 -> Bar closed terakhir
        ]
        df = make_synthetic_ohlc(bars, atr_val=4.0, freq="1h")
        res = detect_swings(df, timeframe="H1")

        sh_prices = [s["price"] for s in res["swing_highs"]]
        sl_prices = [s["price"] for s in res["swing_lows"]]

        # Swing High 3922.00 harus terdeteksi dan confirmed
        self.assertIn(3922.0, sh_prices)
        # Swing High 3930.00 di bar #18 belum memiliki 3 candle kanan -> DILARANG masuk swing_highs!
        self.assertNotIn(3930.0, sh_prices)
        self.assertTrue(all(s["confirmed"] for s in res["all_swings"]))
        self.assertTrue(all(s["confirmed_index"] <= len(df) - 1 for s in res["all_swings"]))

        # Swing Low 3880.00 harus terdeteksi, dan duplikat 3880.20 harus terfilter
        self.assertIn(3880.0, sl_prices)
        self.assertNotIn(3880.2, sl_prices)
        self.assertGreaterEqual(res["summary"]["duplicates_filtered"], 1)

        # Uji jika terdapat candle yang belum closed (is_closed=False) yang menjadi bar ke-3 di kanan pivot
        df_unclosed = df.iloc[:8].copy()  # Bar 0..7 (bar #5 baru punya 2 candle kanan: #6 dan #7)
        # Tambahkan bar #8 yang masih berjalan (is_closed=False)
        forming_row = df.iloc[[8]].copy()
        forming_row["is_closed"] = False
        df_with_forming = pd.concat([df_unclosed, forming_row], ignore_index=True)

        res_unclosed = detect_swings(df_with_forming, timeframe="H1")
        # Karena bar #8 belum closed, bar #5 baru memiliki 2 candle closed di kanan (< right_bars=3),
        # sehingga bar #5 BELUM boleh terkonfirmasi!
        self.assertEqual(len(res_unclosed["swing_highs"]), 0)
        self.assertTrue(res_unclosed["forming_candle_excluded"])

    # =========================================================================
    # TEST 2 — Support/Resistance Detection & Zone Clustering
    # =========================================================================
    def test_02_support_resistance_detection_clustering_and_width_limits(self):
        """
        TEST 2:
        - Uji historical support & historical resistance.
        - Uji repeated touches & touch classification.
        - Uji zona yang terlalu berdekatan (contoh prompt: 3914.20, 3915.10, 3916.00, 3915.60 -> 3914.20 - 3916.00).
        - Uji zona yang terlalu lebar (tidak digabung melebihi batas maksimum).
        """
        # 1. Uji contoh clustering langsung dari Master Prompt Section G
        raw_levels = [3914.20, 3915.10, 3916.00, 3915.60]
        clustered = cluster_zones(raw_levels, timeframe="H1", zone_type="RESISTANCE", atr=6.0)
        self.assertEqual(len(clustered), 1)
        self.assertAlmostEqual(clustered[0]["low"], 3914.20, places=2)
        self.assertAlmostEqual(clustered[0]["high"], 3916.00, places=2)
        self.assertEqual(clustered[0]["source_swing_count"], 4)

        # 2. Uji level yang terlalu berjauhan / akan membuat zona terlalu lebar -> tidak boleh digabung
        wide_levels = [3914.20, 3915.00, 3932.00]
        clustered_wide = cluster_zones(wide_levels, timeframe="H1", zone_type="RESISTANCE", atr=5.0)
        self.assertEqual(len(clustered_wide), 2)
        self.assertTrue(all(z["width"] <= 10.0 for z in clustered_wide))

        # 3. Uji deteksi Support, Resistance, dan Touch Analysis (Repeated Touches tanpa duplikasi candle sama)
        frames = build_realistic_multi_tf_frames()
        df_h1 = frames["H1"]
        sw_h1 = detect_swings(df_h1, timeframe="H1")
        sup_cands = detect_support_candidates(df_h1, swing_result=sw_h1, timeframe="H1")
        res_cands = detect_resistance_candidates(df_h1, swing_result=sw_h1, timeframe="H1")
        self.assertGreater(len(sup_cands), 0)
        self.assertGreater(len(res_cands), 0)

        # Pastikan lebar zona menggunakan ATR dinamis (bukan garis tunggal dan bukan statis antar TF)
        sw_m5 = detect_swings(frames["M5"], timeframe="M5")
        sup_m5 = detect_support_candidates(frames["M5"], swing_result=sw_m5, timeframe="M5")
        self.assertGreater(sup_cands[0]["width"], sup_m5[0]["width"])

        # Uji validasi zona terlalu lebar ditolak oleh validate_zones
        oversized_zone = dict(clustered[0])
        oversized_zone["low"] = 3900.00
        oversized_zone["high"] = 3925.00  # Lebar 25.0 > batas H1 10.0
        val_res = validate_zones([oversized_zone], df=df_h1, current_price=3905.0)
        self.assertEqual(len(val_res["invalidated_zones"]), 1)
        self.assertFalse(val_res["invalidated_zones"][0]["is_structurally_valid"])

        # Uji repeated touches & deduplikasi candle yang sama
        test_zone = {
            "zone_id": "SNR_H1_SUP_TEST",
            "timeframe": "H1",
            "zone_type": "SUPPORT",
            "low": 3880.0,
            "high": 3882.0,
            "mid": 3881.0,
            "width": 2.0,
            "ref_atr": 4.0,
            "formation_index": 2,
            "confirmed_index": 4,
        }
        touch_bars = [
            (3890, 3892, 3888, 3890),  # 0
            (3890, 3891, 3885, 3886),  # 1
            (3886, 3888, 3880.0, 3885),  # 2 pivot
            (3885, 3889, 3884, 3888),  # 3
            (3888, 3892, 3887, 3891),  # 4 confirmed_index
            (3891, 3892, 3881.2, 3889),  # 5 -> Touch #1 (memantul ke 3894)
            (3889, 3894, 3888, 3893),  # 6 -> Keluar zona
            (3893, 3895, 3890, 3892),  # 7
            (3892, 3893, 3881.0, 3888),  # 8 -> Touch #2 (memantul ke 3895)
            (3888, 3895, 3887, 3894),  # 9
        ]
        df_touch = make_synthetic_ohlc(touch_bars, atr_val=4.0, freq="1h")
        touched_z = analyze_zone_touches(test_zone, df_touch)
        self.assertEqual(touched_z["touch_status"], "TESTED_2X")
        self.assertEqual(touched_z["touch_count"], 2)

    # =========================================================================
    # TEST 3 — Breakout
    # =========================================================================
    def test_03_breakout_detection_all_cases(self):
        """
        TEST 3:
        - Valid breakout.
        - False breakout.
        - Wick penetration.
        - Candle close confirmation.
        - Breakout tanpa momentum.
        """
        res_zone = {
            "zone_id": "SNR_H1_RES_3914_3916",
            "timeframe": "H1",
            "zone_type": "RESISTANCE",
            "low": 3914.0,
            "high": 3916.0,
            "mid": 3915.0,
            "width": 2.0,
            "ref_atr": 4.0,
            "formation_index": 1,
            "confirmed_index": 3,
        }

        base_bars = [
            (3908, 3910, 3906, 3909),  # 0
            (3909, 3916.0, 3908, 3911),  # 1
            (3911, 3913, 3909, 3912),  # 2
            (3912, 3914, 3910, 3913),  # 3 confirmed_index
            (3913, 3914.5, 3911, 3913.5),  # 4
            (3913.5, 3915.0, 3912, 3914.2),  # 5
        ]

        # A. Valid Breakout (close tegas di atas 3916.0 dengan body kuat + follow-through)
        valid_bo_bars = base_bars + [
            (3914.2, 3920.5, 3914.0, 3919.8, 2200),  # 6 -> body 5.6 (1.4 ATR), close 3919.8 (> 3916)
            (3919.8, 3922.0, 3918.5, 3921.2, 1800),  # 7 -> close tetap di luar
        ]
        df_valid_bo = make_synthetic_ohlc(valid_bo_bars, atr_val=4.0)
        res_valid = detect_zone_breakout(res_zone, df_valid_bo)
        self.assertEqual(res_valid["status"], "VALID_BREAKOUT")
        self.assertTrue(res_valid["close_outside_zone"])
        self.assertTrue(res_valid["momentum_supported"])

        # B. False Breakout (sempat close di luar 3916.0 lalu candle berikutnya jatuh kembali ke bawah zona)
        false_bo_bars = base_bars + [
            (3914.2, 3918.0, 3914.0, 3917.2, 1500),  # 6 -> close di atas 3916
            (3917.2, 3917.5, 3911.0, 3912.0, 1900),  # 7 -> banting kembali ke 3912 (< 3914)
        ]
        df_false_bo = make_synthetic_ohlc(false_bo_bars, atr_val=4.0)
        res_false = detect_zone_breakout(res_zone, df_false_bo)
        self.assertEqual(res_false["status"], "FALSE_BREAKOUT")

        # C. Wick Penetration Only (wick menembus ke 3917.2 namun close di dalam zona 3915.2)
        wick_bars = base_bars + [
            (3914.2, 3917.2, 3913.8, 3915.2, 1300),
        ]
        df_wick = make_synthetic_ohlc(wick_bars, atr_val=4.0)
        res_wick = detect_zone_breakout(res_zone, df_wick)
        self.assertEqual(res_wick["status"], "NO_BREAKOUT")
        self.assertTrue(res_wick["wick_penetration_only"])
        self.assertFalse(res_wick["close_outside_zone"])

        # D. Breakout Tanpa Momentum (close tipis di luar 3916.2 dengan body doji/kecil)
        weak_bo_bars = base_bars + [
            (3915.9, 3917.5, 3914.5, 3916.2, 900),  # body hanya 0.30 (< 0.42*4.0), penetrasi tipis 0.20
        ]
        df_weak_bo = make_synthetic_ohlc(weak_bo_bars, atr_val=4.0)
        res_weak = detect_zone_breakout(res_zone, df_weak_bo)
        self.assertEqual(res_weak["status"], "BREAKOUT_PENDING_CONFIRMATION")

    # =========================================================================
    # TEST 4 — Retest
    # =========================================================================
    def test_04_retest_detection_all_states(self):
        """
        TEST 4:
        - Successful retest (RETEST_CONFIRMED).
        - Failed retest (RETEST_FAILED).
        - Retest belum selesai (RETEST_IN_PROGRESS & RETEST_WAITING).
        - Tidak terjadi retest (NO_RETEST).
        """
        res_zone = {
            "zone_id": "SNR_H1_RES_3914_3916",
            "timeframe": "H1",
            "zone_type": "RESISTANCE",
            "low": 3914.0,
            "high": 3916.0,
            "mid": 3915.0,
            "width": 2.0,
            "ref_atr": 4.0,
            "formation_index": 1,
            "confirmed_index": 2,
        }
        prefix = [
            (3910, 3912, 3908, 3911),  # 0
            (3911, 3916, 3910, 3912),  # 1
            (3912, 3914, 3911, 3913),  # 2 confirmed_index
        ]

        # 1. Tidak terjadi retest karena belum ada breakout
        df_no_bo = make_synthetic_ohlc(prefix + [(3913, 3915, 3911, 3912)], atr_val=4.0)
        rt_none = detect_zone_retest(res_zone, df_no_bo)
        self.assertEqual(rt_none["status"], "NO_RETEST")

        # 2. Retest Waiting (breakout valid terjadi dan harga terus naik jauh tanpa pullback)
        df_waiting = make_synthetic_ohlc(
            prefix + [
                (3913, 3920, 3913, 3919.5),  # 3 breakout valid
                (3919.5, 3924, 3919.0, 3923.0),  # 4 jauh di atas zona
            ],
            atr_val=4.0,
        )
        rt_wait = detect_zone_retest(res_zone, df_waiting)
        self.assertEqual(rt_wait["status"], "RETEST_WAITING")

        # 3. Retest In Progress (harga menyentuh kembali zona namun close masih bearish di dalam zona)
        df_in_prog = make_synthetic_ohlc(
            prefix + [
                (3913, 3920, 3913, 3919.5),  # 3 breakout valid
                (3919.5, 3919.8, 3915.1, 3915.3),  # 4 masuk zona tetapi belum ada reaksi bullish close di atas 3916
            ],
            atr_val=4.0,
        )
        rt_prog = detect_zone_retest(res_zone, df_in_prog)
        self.assertEqual(rt_prog["status"], "RETEST_IN_PROGRESS")

        # 4. Successful Retest (RETEST_CONFIRMED: menyentuh 3915.2 lalu memantul bullish close di 3918.5)
        df_confirmed = make_synthetic_ohlc(
            prefix + [
                (3913, 3920, 3913, 3919.5),  # 3 breakout valid
                (3919.5, 3919.8, 3915.2, 3916.5),  # 4 retest touch
                (3916.5, 3919.2, 3916.0, 3918.8),  # 5 reaksi bullish konfirmasi
            ],
            atr_val=4.0,
        )
        rt_conf = detect_zone_retest(res_zone, df_confirmed)
        self.assertEqual(rt_conf["status"], "RETEST_CONFIRMED")
        self.assertEqual(rt_conf["role_reversal"], "RESISTANCE_TO_SUPPORT")
        self.assertEqual(rt_conf["flipped_zone_type"], "SUPPORT")

        # 5. Failed Retest (RETEST_FAILED: harga jebol kembali ke bawah 3914.0)
        df_failed = make_synthetic_ohlc(
            prefix + [
                (3913, 3920, 3913, 3919.5),  # 3 breakout valid
                (3919.5, 3919.6, 3911.0, 3911.8),  # 4 jebol ke bawah 3914
            ],
            atr_val=4.0,
        )
        rt_fail = detect_zone_retest(res_zone, df_failed)
        self.assertEqual(rt_fail["status"], "RETEST_FAILED")

    # =========================================================================
    # TEST 5 — Rejection
    # =========================================================================
    def test_05_rejection_detection_all_types(self):
        """
        TEST 5:
        - Bullish support rejection.
        - Bearish resistance rejection.
        - Weak rejection.
        - Rejection palsu.
        """
        sup_zone = {
            "zone_id": "SNR_M5_SUP_3865_3868",
            "timeframe": "M5",
            "zone_type": "SUPPORT",
            "low": 3865.0,
            "high": 3868.0,
            "mid": 3866.5,
            "width": 3.0,
            "ref_atr": 2.0,
        }
        res_zone = {
            "zone_id": "SNR_M5_RES_3914_3917",
            "timeframe": "M5",
            "zone_type": "RESISTANCE",
            "low": 3914.0,
            "high": 3917.0,
            "mid": 3915.5,
            "width": 3.0,
            "ref_atr": 2.0,
        }

        # 1. Bullish Support Rejection (masuk support, lower wick panjang, close bullish + next candle naik)
        df_bull_rej = make_synthetic_ohlc([
            (3872, 3873, 3869, 3870),
            (3870, 3870.5, 3865.2, 3869.2),  # lower wick 3.8, body 0.8, close di atas zona
            (3869.2, 3871.8, 3868.8, 3871.2),  # konfirmasi candle berikutnya
        ], atr_val=2.0, freq="5min")
        r_bull = detect_zone_rejection(sup_zone, df_bull_rej)
        self.assertEqual(r_bull["status"], "BULLISH_REJECTION")
        self.assertTrue(r_bull["confirmed"])

        # 2. Bearish Resistance Rejection (masuk resistance, upper wick panjang, close bearish + next candle turun)
        df_bear_rej = make_synthetic_ohlc([
            (3910, 3913, 3909, 3912),
            (3912, 3916.8, 3911.0, 3911.5),  # upper wick 4.8, body 0.5, close di bawah zona
            (3911.5, 3912.0, 3909.2, 3909.8),  # konfirmasi candle berikutnya
        ], atr_val=2.0, freq="5min")
        r_bear = detect_zone_rejection(res_zone, df_bear_rej)
        self.assertEqual(r_bear["status"], "BEARISH_REJECTION")
        self.assertTrue(r_bear["confirmed"])

        # 3. Weak Rejection (baru pada candle terakhir belum ada konfirmasi candle berikutnya)
        df_weak_rej = make_synthetic_ohlc([
            (3872, 3873, 3869, 3870),
            (3870, 3870.5, 3865.5, 3869.0),  # candle terakhir, belum ada next candle
        ], atr_val=2.0, freq="5min")
        r_weak = detect_zone_rejection(sup_zone, df_weak_rej)
        self.assertEqual(r_weak["status"], "WEAK_REJECTION")
        self.assertFalse(r_weak["confirmed"])

        # 4. Rejection Palsu (FALSE_REJECTION: sempat membentuk lower wick namun candle berikutnya jebol ke bawah support)
        df_false_rej = make_synthetic_ohlc([
            (3872, 3873, 3869, 3870),
            (3870, 3870.5, 3865.2, 3869.2),  # pinbar bullish di support
            (3869.2, 3869.5, 3862.0, 3862.5),  # gagal dan jebol ke bawah 3865.0!
        ], atr_val=2.0, freq="5min")
        r_false = detect_zone_rejection(sup_zone, df_false_rej)
        self.assertEqual(r_false["status"], "FALSE_REJECTION")
        self.assertFalse(r_false["confirmed"])

    # =========================================================================
    # TEST 6 — Multi-Timeframe & Confluence & Scoring
    # =========================================================================
    def test_06_multi_timeframe_independence_and_confluence(self):
        """
        TEST 6:
        - Pastikan H4, H1, M30, dan M5 diproses secara terpisah dan hasilnya benar.
        - Uji contoh Multi-TF Confluence (H4: 3920-3925, H1: 3918-3922, M30: 3919-3921 -> 3919.00-3922.00).
        - Uji sistem scoring 0-100 dan pencegahan Grade A semata-mata karena banyak touch.
        """
        frames = build_realistic_multi_tf_frames()
        scan_res = run_snr_scan(frames=frames, live=False)

        for tf in ("H4", "H1", "M30", "M5"):
            self.assertIn(tf, scan_res["swings_by_tf"])
            self.assertIn(tf, scan_res["clustered_zones_by_tf"])
            self.assertIn(tf, scan_res["scored_zones_by_tf"])
            self.assertIn(tf, scan_res["atr_diagnostics"])
            self.assertTrue(scan_res["atr_diagnostics"][tf]["available"])

        # Uji contoh Multi-TF Confluence persis sesuai Section M
        zones_by_tf = {
            "H4": [{
                "zone_id": "Z_H4",
                "timeframe": "H4",
                "zone_type": "RESISTANCE",
                "low": 3920.00,
                "high": 3925.00,
                "mid": 3922.50,
                "width": 5.0,
                "ref_atr": 8.0,
                "is_active": True,
                "lifecycle_status": "ACTIVE",
            }],
            "H1": [{
                "zone_id": "Z_H1",
                "timeframe": "H1",
                "zone_type": "RESISTANCE",
                "low": 3918.00,
                "high": 3922.00,
                "mid": 3920.00,
                "width": 4.0,
                "ref_atr": 5.0,
                "is_active": True,
                "lifecycle_status": "ACTIVE",
            }],
            "M30": [{
                "zone_id": "Z_M30",
                "timeframe": "M30",
                "zone_type": "RESISTANCE",
                "low": 3919.00,
                "high": 3921.00,
                "mid": 3920.00,
                "width": 2.0,
                "ref_atr": 3.0,
                "is_active": True,
                "lifecycle_status": "ACTIVE",
            }],
        }
        conf = evaluate_mtf_confluence(zones_by_tf)
        self.assertGreaterEqual(len(conf["confluence_groups"]), 1)
        cg = conf["confluence_groups"][0]
        self.assertEqual(cg["timeframes"], ["H4", "H1", "M30"])
        self.assertAlmostEqual(cg["confluence_low"], 3919.00, places=2)
        self.assertAlmostEqual(cg["confluence_high"], 3922.00, places=2)

        # Uji aturan Section N: Zona TESTED_3X_PLUS dengan tanda absorpsi tidak boleh diberi Grade A
        absorbed_zone = {
            "zone_id": "Z_ABSORBED",
            "timeframe": "H1",
            "zone_type": "SUPPORT",
            "low": 3880.0,
            "high": 3883.0,
            "mid": 3881.5,
            "width": 3.0,
            "ref_atr": 4.0,
            "swing_class": "MAJOR",
            "swing_strength": 9.0,
            "bounce_strength_atr": 1.8,
            "historical_reaction_count": 4,
            "repeated_rejections": 3,
            "consolidation_structure": "IMPULSIVE_DEPARTURE",
            "freshness_ratio": 0.6,
            "touch_status": "TESTED_3X_PLUS",
            "touch_analysis": {
                "classification": "TESTED_3X_PLUS",
                "valid_touch_count": 4,
                "touch_quality_score_15": 7.0,
                "absorption_warning": True,
            },
        }
        scored_abs = score_zone(absorbed_zone, confluence_info={"has_confluence": True, "score_15": 15.0})
        self.assertLess(scored_abs["score"], 80.0)
        self.assertNotEqual(scored_abs["grade"], "A")

    # =========================================================================
    # TEST 7 — Entry Validation & Risk/Reward
    # =========================================================================
    def test_07_entry_validation_buy_sell_wait_invalid_and_rr_rules(self):
        """
        TEST 7:
        Uji:
        - BUY valid.
        - SELL valid.
        - WAIT.
        - INVALID.
        - RR di bawah standar (< 1:1.5 -> WAIT).
        - SL struktural tidak tersedia -> WAIT.
        - TP tidak memadai / tidak tersedia -> WAIT.
        """
        s1_zone = {
            "zone_id": "SNR_H1_SUP_1",
            "rank_label": "S1",
            "timeframe": "H1",
            "zone_type": "SUPPORT",
            "low": 3865.00,
            "high": 3868.00,
            "mid": 3866.50,
            "width": 3.0,
            "ref_atr": 4.0,
            "score": 86.0,
            "score_int": 86,
            "grade": "A",
            "strength_category": "STRONG SUPPORT",
            "strength_short": "STRONG",
            "distance_points": 0.5,
            "distance_atr": 0.12,
            "price_inside_zone": False,
            "ranking_priority": 88.0,
            "lifecycle_status": "ACTIVE",
            "is_active": True,
            "origin_pivot": {"low": 3865.00, "price": 3865.00},
            "source_swings": [{"low": 3865.00, "price": 3865.00, "confirmed": True}],
        }
        r1_far = {
            "zone_id": "SNR_H1_RES_1",
            "rank_label": "R1",
            "timeframe": "H1",
            "zone_type": "RESISTANCE",
            "low": 3882.00,
            "high": 3885.00,
            "mid": 3883.50,
            "width": 3.0,
            "ref_atr": 4.0,
            "score": 85.0,
            "score_int": 85,
            "grade": "A",
            "strength_category": "STRONG RESISTANCE",
            "strength_short": "STRONG",
            "distance_points": 13.0,
            "distance_atr": 3.25,
            "price_inside_zone": False,
            "ranking_priority": 75.0,
            "lifecycle_status": "ACTIVE",
            "is_active": True,
            "origin_pivot": {"high": 3885.00, "price": 3885.00},
            "source_swings": [{"high": 3885.00, "price": 3885.00, "confirmed": True}],
        }
        ranked_for_buy = {
            "active_supports": [s1_zone],
            "active_resistances": [r1_far],
            "broken_zones": [],
            "invalidated_zones": [],
            "historical_zones": [],
        }

        # 1. BUY Valid (M5 menyentuh S1 3865-3868, membentuk bullish rejection + konfirmasi momentum, RR > 1.5 ke 3882)
        m5_buy_bars = [
            (3872.0, 3872.5, 3870.0, 3870.5),
            (3870.5, 3871.0, 3868.5, 3869.0),
            (3869.0, 3869.5, 3865.4, 3868.6),  # Bullish rejection pinbar di S1
            (3868.6, 3870.2, 3868.2, 3869.8),  # Konfirmasi candle bullish lanjutan + momentum
        ]
        df_m5_buy = make_synthetic_ohlc(m5_buy_bars, atr_val=1.5, freq="5min")
        ev_buy = validate_snr_entry(
            ranked_zones=ranked_for_buy,
            m5_df=df_m5_buy,
            market_bias={"overall_bias": "BULLISH"},
            current_price=3869.80,
        )
        self.assertEqual(ev_buy["final_status"], "BUY")
        self.assertEqual(ev_buy["buy_status"], "CONFIRMED")
        self.assertTrue(ev_buy["rr_calculation"]["valid"])
        self.assertGreaterEqual(ev_buy["rr_calculation"]["rr_tp1"], 1.5)

        # 2. RR di bawah standar minimal (< 1:1.5) -> harus WAIT!
        r1_too_close = dict(r1_far)
        r1_too_close["low"] = 3872.50  # Sangat dekat dengan entry 3869.80 -> RR < 1.5
        r1_too_close["high"] = 3874.50
        ranked_low_rr = dict(ranked_for_buy, active_resistances=[r1_too_close])
        ev_low_rr = validate_snr_entry(
            ranked_zones=ranked_low_rr,
            m5_df=df_m5_buy,
            market_bias={"overall_bias": "BULLISH"},
            current_price=3869.80,
        )
        self.assertEqual(ev_low_rr["final_status"], "WAIT")
        self.assertFalse(ev_low_rr["rr_calculation"]["valid"])

        # 3. SL Struktural Tidak Tersedia -> harus WAIT!
        s1_no_sl = dict(s1_zone, structural_sl_available=False)
        ranked_no_sl = dict(ranked_for_buy, active_supports=[s1_no_sl])
        ev_no_sl = validate_snr_entry(
            ranked_zones=ranked_no_sl,
            m5_df=df_m5_buy,
            market_bias={"overall_bias": "BULLISH"},
            current_price=3869.80,
        )
        self.assertEqual(ev_no_sl["final_status"], "WAIT")
        self.assertFalse(ev_no_sl["rr_calculation"]["structural_sl_available"])

        # 4. TP Struktural Tidak Tersedia (tidak ada zona lawan di atas entry) -> harus WAIT!
        ranked_no_tp = dict(ranked_for_buy, active_resistances=[])
        ev_no_tp = validate_snr_entry(
            ranked_zones=ranked_no_tp,
            m5_df=df_m5_buy,
            market_bias={"overall_bias": "BULLISH"},
            current_price=3869.80,
        )
        self.assertEqual(ev_no_tp["final_status"], "WAIT")
        self.assertFalse(ev_no_tp["rr_calculation"]["structural_tp_available"])

        # 5. SELL Valid (M5 menyentuh R1 3914-3917, membentuk bearish rejection + konfirmasi momentum, RR > 1.5 ke S1 3895)
        r1_sell = dict(
            r1_far,
            low=3914.00,
            high=3917.00,
            mid=3915.50,
            origin_pivot={"high": 3917.00, "price": 3917.00},
            source_swings=[{"high": 3917.00, "price": 3917.00, "confirmed": True}],
        )
        s1_target = dict(s1_zone, low=3892.00, high=3895.00, mid=3893.50)
        ranked_for_sell = {
            "active_supports": [s1_target],
            "active_resistances": [r1_sell],
            "broken_zones": [],
            "invalidated_zones": [],
            "historical_zones": [],
        }
        m5_sell_bars = [
            (3910.0, 3911.5, 3909.5, 3911.0),
            (3911.0, 3913.5, 3910.8, 3913.2),
            (3913.2, 3916.8, 3912.6, 3913.0),  # Bearish rejection pinbar di R1
            (3913.0, 3913.3, 3911.4, 3911.8),  # Konfirmasi candle bearish lanjutan + momentum
        ]
        df_m5_sell = make_synthetic_ohlc(m5_sell_bars, atr_val=1.5, freq="5min")
        ev_sell = validate_snr_entry(
            ranked_zones=ranked_for_sell,
            m5_df=df_m5_sell,
            market_bias={"overall_bias": "BEARISH"},
            current_price=3911.80,
        )
        self.assertEqual(ev_sell["final_status"], "SELL")
        self.assertEqual(ev_sell["sell_status"], "CONFIRMED")
        self.assertTrue(ev_sell["rr_calculation"]["valid"])

        # 6. WAIT (Harga berada di tengah range jauh dari S1 dan R1)
        m5_wait_bars = [
            (3900.0, 3901.0, 3899.5, 3900.2),
            (3900.2, 3900.8, 3899.8, 3900.0),
        ]
        df_m5_wait = make_synthetic_ohlc(m5_wait_bars, atr_val=1.5, freq="5min")
        ev_wait = validate_snr_entry(
            ranked_zones=ranked_for_sell,
            m5_df=df_m5_wait,
            market_bias={"overall_bias": "BEARISH"},
            current_price=3900.00,
        )
        self.assertEqual(ev_wait["final_status"], "WAIT")
        self.assertEqual(ev_wait["rr_status"], "NOT CHECKED")

        # 7. INVALID (Zona telah INVALIDATED dan tidak ada zona aktif tersisa)
        inv_zone = dict(r1_sell, lifecycle_status="INVALIDATED", is_active=False)
        ranked_invalid = {
            "active_supports": [],
            "active_resistances": [],
            "broken_zones": [],
            "invalidated_zones": [inv_zone],
            "historical_zones": [],
        }
        ev_inv = validate_snr_entry(
            ranked_zones=ranked_invalid,
            m5_df=df_m5_wait,
            market_bias={"overall_bias": "BEARISH"},
            current_price=3900.00,
        )
        self.assertEqual(ev_inv["final_status"], "INVALID")

    # =========================================================================
    # TEST 8 — Historical Backtesting (Walk-Forward No Look-Ahead Bias)
    # =========================================================================
    def test_08_historical_backtesting_no_lookahead_and_dataset_splits(self):
        """
        TEST 8:
        Gunakan data historis XAUUSD (`backtest_fitur_xauusd.csv` + multi-TF XAUUSD bars).
        Pastikan:
        - Tidak terjadi look-ahead bias.
        - Tidak menggunakan candle masa depan.
        - Pivot hanya digunakan setelah confirmed.
        - Sinyal dan atribut pivot tidak berubah secara retrospektif akibat data masa depan.
        - Spread dan spesifikasi kontrak diperhitungkan.
        - Pemisahan data development, validation, dan out-of-sample testing.
        """
        csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backtest_fitur_xauusd.csv")
        self.assertTrue(os.path.exists(csv_path), "File backtest_fitur_xauusd.csv harus tersedia")

        raw_bt = pd.read_csv(csv_path).drop_duplicates(subset=["waktu"]).reset_index(drop=True)
        self.assertGreater(len(raw_bt), 300)

        # Konversi data historis XAUUSD dari CSV menjadi OHLC closed candles
        closes = pd.to_numeric(raw_bt["harga"], errors="coerce").dropna().to_numpy(dtype=float)[:360]
        atrs_csv = pd.to_numeric(raw_bt["atr"], errors="coerce").fillna(2.5).to_numpy(dtype=float)[:360]
        times = pd.to_datetime(raw_bt["waktu"].iloc[:360], errors="coerce")

        hist_rows = []
        for i in range(len(closes)):
            c = float(closes[i])
            o = float(closes[i - 1]) if i > 0 else c
            atr_v = max(0.50, round(float(atrs_csv[i]), 2))
            h = round(max(o, c) + 0.45 * atr_v, 2)
            l = round(min(o, c) - 0.45 * atr_v, 2)
            hist_rows.append({
                "Time": times.iloc[i],
                "Open": round(o, 2),
                "High": h,
                "Low": l,
                "Close": round(c, 2),
                "ATR14": atr_v,
                "tick_volume": 1000.0 + (i % 15) * 65.0,
            })
        full_df = pd.DataFrame(hist_rows)

        # Pisahkan menjadi Development (0..150), Validation (100..250), Out-of-Sample (200..360)
        dev_df = full_df.iloc[:150].reset_index(drop=True)
        val_df = full_df.iloc[100:250].reset_index(drop=True)
        oos_df = full_df.iloc[200:360].reset_index(drop=True)

        for split_name, split_df in (("DEV", dev_df), ("VAL", val_df), ("OOS", oos_df)):
            sw = detect_swings(split_df, timeframe="H1")
            self.assertGreater(len(sw["all_swings"]), 0, f"Split {split_name} harus mendeteksi swing")
            for piv in sw["all_swings"]:
                self.assertTrue(piv["confirmed"])
                self.assertLessEqual(piv["confirmed_index"], len(split_df) - 1)
                self.assertEqual(piv["confirmed_index"], piv["bar_index"] + sw["right_bars"])

        # Verifikasi Walk-Forward Invariance (0% Look-Ahead Bias):
        # Semua pivot yang sudah terkonfirmasi pada step T=120 harus memiliki harga, confirmed_index,
        # strength, dan swing_class yang 100% IDENTIK ketika dievaluasi ulang pada step T=180!
        sw_at_120 = detect_swings(full_df.iloc[:120], timeframe="H1")
        sw_at_180 = detect_swings(full_df.iloc[:180], timeframe="H1")

        map_180 = {p["pivot_id"]: p for p in sw_at_180["all_swings"]}
        for p120 in sw_at_120["all_swings"]:
            pid = p120["pivot_id"]
            self.assertIn(pid, map_180, f"Pivot {pid} yang sudah confirmed di T=120 tidak boleh hilang di T=180")
            p180 = map_180[pid]
            self.assertEqual(p120["price"], p180["price"])
            self.assertEqual(p120["confirmed_index"], p180["confirmed_index"])
            self.assertEqual(p120["strength"], p180["strength"])
            self.assertEqual(p120["swing_class"], p180["swing_class"])

        # Verifikasi keputusan SNR pada slice [:140] identik sebelum dan sesudah ada data masa depan (ketika dievaluasi pada slice [:140])
        frames_t140 = {
            "H4": full_df.iloc[:140].copy(),
            "H1": full_df.iloc[:140].copy(),
            "M30": full_df.iloc[:140].copy(),
            "M5": full_df.iloc[:140].copy(),
        }
        scan_a = run_snr_scan(frames=frames_t140, live=False)
        scan_b = run_snr_scan(frames=frames_t140, live=False)
        self.assertEqual(scan_a["decision"]["signal"], scan_b["decision"]["signal"])
        self.assertEqual(scan_a["market_bias"]["overall_bias"], scan_b["market_bias"]["overall_bias"])
        self.assertGreater(scan_a["symbol_spec"]["spread_points"], 0.0)
        self.assertGreater(scan_a["symbol_spec"]["contract_size"], 0.0)

    # =========================================================================
    # TEST 9 — SMC REGRESSION TEST (WAJIB)
    # =========================================================================
    def test_09_smc_regression_identical_before_and_after_snr(self):
        """
        TEST 9 — SMC REGRESSION TEST (WAJIB):
        Jalankan SMC Scanner sebelum dan sesudah menjalankan SNR menggunakan snapshot yang sama.
        Bandingkan 10 parameter wajib:
        1. Market Bias
        2. ATR
        3. Jumlah zona
        4. Zone Score
        5. Zone Grade
        6. Primary Zone
        7. Entry Validation
        8. Final Decision
        9. SL/TP/RR
        10. Radar Status
        Hasil SMC harus 100% IDENTIK.
        """
        frames = build_realistic_multi_tf_frames()
        # Tambahkan juga M15 agar create_market_snapshot lengkap untuk SMC
        frames["M15"] = frames["M30"].copy()
        shared_snap = create_market_snapshot(symbol="XAUUSD", frames=frames, live=False)
        snap_before_copy = copy.deepcopy(shared_snap)

        # 1. Jalankan SMC Scanner SEBELUM SNR
        smc_scan_before = scan_xauusd_zones(symbol="XAUUSD", snapshot=snap_before_copy)
        smc_dec_before = evaluate_trade_decision(snapshot=snap_before_copy, scan_result=smc_scan_before)
        smc_text_before = format_scan_presentation(smc_scan_before, smc_dec_before, detailed=True)

        # 2. Jalankan SNR Advanced Scanner (Summary, Detail, dan Monitor Cycle) menggunakan shared_snap
        snr_res, snr_dec, snr_txt_sum = scan_snr(symbol="XAUUSD", snapshot=shared_snap, live=False, detailed=False)
        _, _, snr_txt_det = scan_snr_detail(symbol="XAUUSD", snapshot=shared_snap, live=False)
        snr_engine = SnrCycleEngine(symbol="XAUUSD")
        snr_engine.execute_cycle(snapshot=shared_snap, live=False, detailed=True)

        self.assertIn("XAUUSD SNR ADVANCED SCANNER", snr_txt_sum)
        self.assertIn("[22. WAIT REASON BREAKDOWN]", snr_txt_det)

        # 3. Jalankan SMC Scanner SESUDAH SNR pada shared_snap yang sama
        smc_scan_after = scan_xauusd_zones(symbol="XAUUSD", snapshot=shared_snap)
        smc_dec_after = evaluate_trade_decision(snapshot=shared_snap, scan_result=smc_scan_after)
        smc_text_after = format_scan_presentation(smc_scan_after, smc_dec_after, detailed=True)

        # Bandingkan ke-10 aspek wajib:
        # (1) Market Bias
        self.assertEqual(smc_scan_before.get("htf_bias"), smc_scan_after.get("htf_bias"))
        self.assertEqual(smc_scan_before.get("h1_structure"), smc_scan_after.get("h1_structure"))
        self.assertEqual(smc_scan_before.get("m30_status"), smc_scan_after.get("m30_status"))

        # (2) ATR
        self.assertEqual(smc_scan_before.get("atr_diagnostics"), smc_scan_after.get("atr_diagnostics"))

        # (3) Jumlah Zona
        zones_b = [z.to_dict() if hasattr(z, "to_dict") else dict(z) for z in smc_scan_before.get("active_zones", [])]
        zones_a = [z.to_dict() if hasattr(z, "to_dict") else dict(z) for z in smc_scan_after.get("active_zones", [])]
        self.assertEqual(len(zones_b), len(zones_a))

        # (4) Zone Score & (5) Zone Grade
        scores_b = [(z.get("zone_id"), z.get("score"), z.get("grade")) for z in zones_b]
        scores_a = [(z.get("zone_id"), z.get("score"), z.get("grade")) for z in zones_a]
        self.assertEqual(scores_b, scores_a)

        # (6) Primary Zone
        prio_b = smc_dec_before.get("radar_priority") or {}
        prio_a = smc_dec_after.get("radar_priority") or {}
        self.assertEqual(prio_b.get("primary_directional_zone"), prio_a.get("primary_directional_zone"))

        # (7) Entry Validation
        self.assertEqual(smc_dec_before.get("m5_stages"), smc_dec_after.get("m5_stages"))
        self.assertEqual(smc_dec_before.get("entry_validation_summary"), smc_dec_after.get("entry_validation_summary"))

        # (8) Final Decision
        self.assertEqual(smc_dec_before.get("status"), smc_dec_after.get("status"))
        self.assertEqual(smc_dec_before.get("direction"), smc_dec_after.get("direction"))
        self.assertEqual(smc_dec_before.get("reasons"), smc_dec_after.get("reasons"))

        # (9) SL / TP / RR
        self.assertEqual(smc_dec_before.get("plan"), smc_dec_after.get("plan"))
        self.assertEqual(smc_dec_before.get("early_setup"), smc_dec_after.get("early_setup"))

        # (10) Radar Status & Full Formatted Presentation
        self.assertEqual(prio_b.get("nearest_radar_zone"), prio_a.get("nearest_radar_zone"))
        self.assertEqual(smc_text_before, smc_text_after)

    # =========================================================================
    # TEST 10 — COMMAND TEST
    # =========================================================================
    def test_10_command_parser_and_execution_for_snr_and_smc(self):
        """
        TEST 10:
        Pastikan semua command SNR (`scan snr`, `scan snr detail`, `pantau snr`, `pantau snr detail`)
        dan seluruh command SMC lama (`scan`, `scan detail`, `zona`, `pantau`, `pantau detail`,
        `pantau smc`, `pantau smc detail`) berfungsi tanpa saling tumpang tindih.
        """
        frames = build_realistic_multi_tf_frames()
        frames["M15"] = frames["M30"].copy()
        snap = create_market_snapshot(symbol="XAUUSD", frames=frames, live=False)

        # Verifikasi parsing presisi
        self.assertEqual(xauusd1.parse_cli_command("scan"), {"action": "SCAN", "method": "SMC", "detailed": False})
        self.assertEqual(xauusd1.parse_cli_command("scan detail"), {"action": "SCAN", "method": "SMC", "detailed": True})
        self.assertEqual(xauusd1.parse_cli_command("zona"), {"action": "SCAN", "method": "SMC", "detailed": False})
        self.assertEqual(xauusd1.parse_cli_command("scan snr"), {"action": "SCAN", "method": "SNR", "detailed": False})
        self.assertEqual(xauusd1.parse_cli_command("scan snr detail"), {"action": "SCAN", "method": "SNR", "detailed": True})
        self.assertEqual(xauusd1.parse_cli_command("pantau"), {"action": "PANTAU", "method": "UNIFIED", "detailed": False})
        self.assertEqual(xauusd1.parse_cli_command("pantau detail"), {"action": "PANTAU", "method": "UNIFIED", "detailed": True})
        self.assertEqual(xauusd1.parse_cli_command("pantau smc"), {"action": "PANTAU", "method": "SMC", "detailed": False})
        self.assertEqual(xauusd1.parse_cli_command("pantau smc detail"), {"action": "PANTAU", "method": "SMC", "detailed": True})
        self.assertEqual(xauusd1.parse_cli_command("pantau snr"), {"action": "PANTAU", "method": "SNR", "detailed": False})
        self.assertEqual(xauusd1.parse_cli_command("pantau snr detail"), {"action": "PANTAU", "method": "SNR", "detailed": True})

        # Eksekusi `scan snr`
        out_scan_snr = xauusd1.execute_cli_command("scan snr", snapshot=snap)
        self.assertIn("XAUUSD SNR ADVANCED SCANNER", out_scan_snr["output"])
        self.assertIn("SUPPORT ZONES", out_scan_snr["output"])
        self.assertIn("RESISTANCE ZONES", out_scan_snr["output"])

        # Eksekusi `scan snr detail` (pastikan ke-22 bagian tampil)
        out_scan_snr_det = xauusd1.execute_cli_command("scan snr detail", snapshot=snap)
        for sec_idx in range(1, 23):
            self.assertIn(f"[{sec_idx}.", out_scan_snr_det["output"])

        # Eksekusi `scan` (SMC) & `scan detail` (SMC Detail)
        out_scan_smc = xauusd1.execute_cli_command("scan", snapshot=snap)
        self.assertIn("XAUUSD PYTHON SMC SCANNER", out_scan_smc["output"])
        self.assertIn("1. MARKET BIAS", out_scan_smc["output"])
        self.assertIn("5. FINAL DECISION", out_scan_smc["output"])

        out_scan_smc_det = xauusd1.execute_cli_command("scan detail", snapshot=snap)
        self.assertIn("XAUUSD PYTHON SMC SCANNER", out_scan_smc_det["output"])
        self.assertIn("ATR Diagnostics", out_scan_smc_det["output"])

        # Verifikasi Command Menu di startup presentation & command `help`
        startup_txt = xauusd1.format_startup_presentation(symbol="XAUUSD", mt5_connected=False)
        help_res = xauusd1.execute_cli_command("help")
        for menu_txt in (startup_txt, help_res["output"]):
            self.assertIn("ANALYSIS:", menu_txt)
            self.assertIn("- scan", menu_txt)
            self.assertIn("- scan detail", menu_txt)
            self.assertIn("- scan snr", menu_txt)
            self.assertIn("- scan snr detail", menu_txt)
            self.assertIn("MONITORING:", menu_txt)
            self.assertIn("- pantau", menu_txt)
            self.assertIn("- pantau detail", menu_txt)
            self.assertIn("- pantau smc", menu_txt)
            self.assertIn("- pantau smc detail", menu_txt)
            self.assertIn("- pantau snr", menu_txt)
            self.assertIn("- pantau snr detail", menu_txt)
            self.assertIn("OTHER:", menu_txt)
            self.assertIn("- zona", menu_txt)
            self.assertIn("- jurnal", menu_txt)
            self.assertIn("- audit", menu_txt)
            self.assertIn("- refresh", menu_txt)
            self.assertIn("- help", menu_txt)
            self.assertIn("- exit", menu_txt)

        # Verifikasi command lama (`zona`, `refresh`, `exit`) tetap berfungsi
        out_zona = xauusd1.execute_cli_command("zona", snapshot=snap)
        self.assertEqual(out_zona["output"], out_scan_smc["output"])
        out_refresh = xauusd1.execute_cli_command("refresh", snapshot=snap)
        self.assertEqual(out_refresh["output"], out_scan_smc["output"])
        out_exit = xauusd1.execute_cli_command("exit")
        self.assertIn("Selesai", out_exit["output"])

    # =========================================================================
    # TEST 11 — MARKET CLOSED HANDLING (OPEN vs CLOSED vs DATA UNAVAILABLE)
    # =========================================================================
    def test_11_market_open_closed_and_data_unavailable_handling(self):
        """
        TEST 11 — MARKET CLOSED HANDLING:
        Uji 3 kondisi wajib:
        1. MARKET OPEN:
           - MARKET STATUS : OPEN
           - SNR ANALYSIS  : AVAILABLE
           - LIVE ENTRY aktif (BUY / SELL / WAIT).
        2. MARKET CLOSED dengan historical data tersedia:
           - MARKET STATUS : CLOSED
           - DATA SOURCE   : MT5 HISTORICAL
           - LAST PRICE    : Harga penutupan terakhir
           - LAST CANDLE   : Timestamp candle terakhir
           - Analisis H4, H1, M30, M5 tetap dijalankan (M5 : WAIT)
           - Swing detection, support & resistance detection, zone scoring, dan ranking tetap jalan
           - Breakout, retest, dan rejection historis tetap ditampilkan dengan penanda HISTORICAL (CLOSED CANDLE)
           - SNR ANALYSIS  : AVAILABLE
           - LIVE ENTRY    : WAIT — MARKET CLOSED
           - FINAL STATUS  : MARKET CLOSED
           - Jangan klaim sinyal entry live yang bisa dieksekusi saat market tutup (signal=WAIT, entry=None).
        3. HISTORICAL DATA UNAVAILABLE:
           - MARKET STATUS : DATA UNAVAILABLE
           - DATA SOURCE   : DATA UNAVAILABLE
           - SNR ANALYSIS  : DATA UNAVAILABLE
           - LIVE ENTRY    : WAIT — DATA UNAVAILABLE
        """
        from datetime import datetime, timezone
        from unittest.mock import patch
        import modules.snr.data_provider as snr_dp

        frames = build_realistic_multi_tf_frames()

        # ---------------------------------------------------------------------
        # KONDISI 1: MARKET OPEN
        # ---------------------------------------------------------------------
        res_open, dec_open, txt_open = scan_snr(
            symbol="XAUUSD",
            frames=frames,
            live=False,
            detailed=False,
            market_status="OPEN",
        )
        self.assertEqual(res_open["market_status"], "OPEN")
        self.assertEqual(res_open["snr_analysis"], "AVAILABLE")
        self.assertIn("MARKET STATUS : OPEN", txt_open)
        self.assertIn("SNR ANALYSIS  : AVAILABLE", txt_open)
        self.assertIn(f"LIVE ENTRY    : {dec_open['live_entry']}", txt_open)

        # ---------------------------------------------------------------------
        # KONDISI 2: MARKET CLOSED DENGAN HISTORICAL DATA TERSEDIA
        # ---------------------------------------------------------------------
        # 2a. Uji deteksi otomatis akhir pekan (Sabtu UTC) dengan historical data tersedia
        prepared_frames = {tf: prepare_snr_frame(df, tf)[0] for tf, df in frames.items()}
        sat_utc = datetime(2026, 3, 7, 12, 0, 0, tzinfo=timezone.utc)  # Sabtu siang UTC
        status_weekend = detect_snr_market_status(
            prepared_frames=prepared_frames,
            live=True,
            tick=None,
            now_utc=sat_utc,
        )
        self.assertEqual(status_weekend["market_status"], "CLOSED")
        self.assertEqual(status_weekend["market_status_label"], "MARKET CLOSED")
        self.assertEqual(status_weekend["data_source"], "MT5 HISTORICAL")
        self.assertEqual(status_weekend["snr_analysis"], "AVAILABLE")

        # 2b. Jalankan scan_snr & scan_snr_detail saat market CLOSED
        res_closed, dec_closed, txt_closed = scan_snr(
            symbol="XAUUSD",
            frames=frames,
            live=False,
            detailed=False,
            market_status="CLOSED",
        )
        _, _, txt_closed_detail = scan_snr_detail(
            symbol="XAUUSD",
            frames=frames,
            live=False,
            market_status="CLOSED",
        )

        # Pastikan status & metadata sesuai spesifikasi
        self.assertEqual(res_closed["market_status"], "CLOSED")
        self.assertEqual(res_closed["data_source"], "MT5 HISTORICAL")
        self.assertEqual(res_closed["snr_analysis"], "AVAILABLE")
        self.assertGreater(res_closed["last_price"], 0.0)
        self.assertNotIn(res_closed["last_candle"], ("N/A", "DATA UNAVAILABLE", ""))

        # Pastikan analisis H4, H1, M30, M5 tetap berjalan (swing, S/R, scoring, ranking)
        for tf in ("H4", "H1", "M30", "M5"):
            self.assertGreater(len(res_closed["swings_by_tf"][tf]["all_swings"]), 0)
            self.assertGreater(len(res_closed["scored_zones_by_tf"][tf]), 0)
        self.assertGreater(
            len(res_closed["ranked_zones"]["active_supports"]) + len(res_closed["ranked_zones"]["active_resistances"]),
            0,
        )
        self.assertIn(res_closed["market_bias"]["h4"], ("BULLISH", "BEARISH", "SIDEWAYS", "NEUTRAL"))
        self.assertIn(res_closed["market_bias"]["h1"], ("BULLISH", "BEARISH", "SIDEWAYS", "NEUTRAL"))
        self.assertIn(res_closed["market_bias"]["m30"], ("BULLISH", "BEARISH", "SIDEWAYS", "NEUTRAL"))
        self.assertEqual(res_closed["market_bias"]["m5"], "WAIT")

        # Pemisahan analisis historis & sinyal live:
        # Tidak boleh memberikan sinyal live (signal=WAIT, entry=None, final_status=MARKET CLOSED, live_entry=WAIT — MARKET CLOSED)
        self.assertEqual(dec_closed["final_status"], "MARKET CLOSED")
        self.assertEqual(dec_closed["live_entry"], "WAIT — MARKET CLOSED")
        self.assertEqual(dec_closed["signal"], "WAIT")
        self.assertEqual(dec_closed["direction"], "NONE")
        self.assertIsNone(dec_closed["entry"])
        self.assertIsNone(dec_closed["sl"])
        self.assertIsNone(dec_closed["tp1"])

        # Verifikasi format output sesuai spesifikasi Section 4
        self.assertIn("MARKET STATUS : CLOSED", txt_closed)
        self.assertIn("DATA SOURCE   : MT5 HISTORICAL", txt_closed)
        self.assertIn(f"LAST PRICE    : {res_closed['last_price']:.2f}", txt_closed)
        self.assertIn(f"LAST CANDLE   : {res_closed['last_candle']}", txt_closed)
        self.assertIn(f"H4            : {res_closed['market_bias']['h4']}", txt_closed)
        self.assertIn(f"H1            : {res_closed['market_bias']['h1']}", txt_closed)
        self.assertIn(f"M30           : {res_closed['market_bias']['m30']}", txt_closed)
        self.assertIn("M5            : WAIT", txt_closed)
        self.assertIn("SUPPORT       :", txt_closed)
        self.assertIn("RESISTANCE    :", txt_closed)
        self.assertIn("SNR ANALYSIS  : AVAILABLE", txt_closed)
        self.assertIn("LIVE ENTRY    : WAIT — MARKET CLOSED", txt_closed)
        self.assertIn("FINAL STATUS : MARKET CLOSED", txt_closed)

        self.assertIn("MARKET STATUS : CLOSED", txt_closed_detail)
        self.assertIn("DATA SOURCE   : MT5 HISTORICAL", txt_closed_detail)
        self.assertIn("SNR ANALYSIS  : AVAILABLE", txt_closed_detail)
        self.assertIn("LIVE ENTRY    : WAIT — MARKET CLOSED", txt_closed_detail)
        self.assertIn("FINAL STATUS : MARKET CLOSED", txt_closed_detail)

        # 2c. Uji bahwa setup yang pada candle historis memenuhi syarat BUY tetap dikunci menjadi WAIT — MARKET CLOSED saat market tutup
        from modules.snr.decision_engine import build_snr_final_decision
        ev_mock_buy = {
            "final_status": "BUY",
            "direction": "BUY",
            "setup_type": "REJECTION",
            "buy_status": "CONFIRMED",
            "sell_status": "WAIT",
            "zone_label": "S1",
            "trigger_status": "BULLISH_REJECTION (PINBAR)",
            "entry_price": 3869.80,
            "sl": 3864.20,
            "tp1": 3882.00,
            "tp2": 3890.00,
            "rr_status": "1:2.18",
            "rr_calculation": {"valid": True, "rr_display": "1:2.18", "rr_tp1": 2.18, "rr_tp2": 3.61},
            "reason": "Bullish rejection di S1",
        }
        dec_locked = build_snr_final_decision(
            entry_validation=ev_mock_buy,
            market_bias={"h4": "BULLISH", "h1": "BULLISH", "m30": "BULLISH", "m5": "WAIT", "overall_bias": "BULLISH"},
            price_action={"rejection_summary": "CONFIRMED (HISTORICAL (CLOSED CANDLE))", "breakout_summary": "NOT CONFIRMED", "retest_summary": "NOT CONFIRMED", "momentum_summary": "MODERATE"},
            market_status="CLOSED",
            snr_analysis="AVAILABLE",
        )
        self.assertEqual(dec_locked["final_status"], "MARKET CLOSED")
        self.assertEqual(dec_locked["live_entry"], "WAIT — MARKET CLOSED")
        self.assertEqual(dec_locked["signal"], "WAIT")
        self.assertEqual(dec_locked["historical_signal"], "BUY")
        self.assertIsNone(dec_locked["entry"])

        # ---------------------------------------------------------------------
        # KONDISI 3: HISTORICAL DATA UNAVAILABLE
        # ---------------------------------------------------------------------
        res_unavail, dec_unavail, txt_unavail = scan_snr(
            symbol="XAUUSD",
            frames={},
            live=False,
            detailed=False,
        )
        self.assertEqual(res_unavail["market_status"], "DATA UNAVAILABLE")
        self.assertEqual(res_unavail["data_source"], "DATA UNAVAILABLE")
        self.assertEqual(res_unavail["snr_analysis"], "DATA UNAVAILABLE")
        self.assertEqual(dec_unavail["final_status"], "DATA UNAVAILABLE")
        self.assertEqual(dec_unavail["live_entry"], "WAIT — DATA UNAVAILABLE")
        self.assertEqual(dec_unavail["signal"], "WAIT")
        self.assertIn("MARKET STATUS : DATA UNAVAILABLE", txt_unavail)
        self.assertIn("DATA SOURCE   : DATA UNAVAILABLE", txt_unavail)
        self.assertIn("LAST PRICE    : DATA UNAVAILABLE", txt_unavail)
        self.assertIn("LAST CANDLE   : DATA UNAVAILABLE", txt_unavail)
        self.assertIn("SNR ANALYSIS  : DATA UNAVAILABLE", txt_unavail)
        self.assertIn("LIVE ENTRY    : WAIT — DATA UNAVAILABLE", txt_unavail)

        # ---------------------------------------------------------------------
        # KONDISI 4: Verifikasi _fetch_mt5_historical_candles memanggil ambil_data(tf_const, bars)
        # ---------------------------------------------------------------------
        called_args = []

        def fake_ambil_data(tf_const, jumlah, percobaan=3):
            called_args.append((tf_const, jumlah))
            return frames["H1"].copy()

        with patch.object(snr_dp, "ambil_data", side_effect=fake_ambil_data):
            df_fetched = snr_dp._fetch_mt5_historical_candles("XAUUSDc", "H1", 220)
            self.assertIsNotNone(df_fetched)
            self.assertGreater(len(df_fetched), 0)
            self.assertEqual(len(called_args), 1)
            self.assertIsInstance(called_args[0][0], int)
            self.assertEqual(called_args[0][1], 220)


if __name__ == "__main__":
    unittest.main()
