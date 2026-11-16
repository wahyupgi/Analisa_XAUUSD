# -*- coding: utf-8 -*-
"""
Comprehensive Regression Test Suite:
- Perbaikan extract_snr_supporting_context() & pencatatan error transparan (AVAILABLE / PARTIAL / UNAVAILABLE)
- Klasifikasi Market Context terpadu SMC + SNR (tanpa mencampur/memaksakan hasil identik)
- Breakout / Retest / Rejection PA coherence & S&R role reading
- Shared Snapshot pada scan gabungan (`scan gabungan` / `scan all`) & `UnifiedMonitor`
- Kompatibilitas seluruh command lama (`scan`, `scan smc`, `scan detail`, `scan snr`, `pantau`, `diag`)
"""
import copy
import unittest
from datetime import datetime
from unittest.mock import patch

import pandas as pd

import xauusd1
from modules.snr.breakout_detection import detect_zone_breakout
from modules.snr.config import get_snr_config
from modules.snr.data_provider import log_snr_error
from modules.snr.entry_validator import validate_snr_entry
from modules.snr.formatter import format_snr_detail, format_snr_summary
from modules.snr.market_bias import evaluate_snr_market_bias
from modules.snr.momentum_analysis import analyze_momentum
from modules.snr.rejection_detection import detect_zone_rejection
from modules.snr.retest_detection import detect_zone_retest
from modules.snr.scanner import _build_price_action_summary, run_snr_scan
from modules.snr.zone_ranking import rank_zones
from modules.trend_following import build_unified_market_context, extract_snr_supporting_context
from modules.unified_monitor import UnifiedMonitor, compare_scanner_snapshots, run_unified_scan
from modules.zone_scanner import (
    create_market_snapshot,
    evaluate_trade_decision,
    format_diagnostic_report,
    format_scan_presentation,
    scan_xauusd_zones,
)
from tests.test_snr_scanner import build_realistic_multi_tf_frames


def _make_ohlc(rows, atr_val=2.0, freq_min=5, end_dt=None):
    end_t = end_dt or datetime(2026, 11, 16, 10, 30)
    idx = pd.date_range(end=end_t, periods=len(rows), freq=f"{freq_min}min")
    df = pd.DataFrame(rows, columns=["Open", "High", "Low", "Close", "Volume"], index=idx)
    df = df.reset_index().rename(columns={"index": "Time"})
    df["ATR14"] = float(atr_val)
    df["HasTickVolume"] = True
    return df


def _make_bearish_frames(inside_support_price=3874.5):
    """Membangun data multi-TF konsisten: H4 BEARISH, H1 BEARISH, M5 bounce di S1."""
    frames = build_realistic_multi_tf_frames()
    h4_rows = []
    p = 3980.0
    for i in range(60):
        step = -2.2 if i % 5 != 3 else 1.5
        o = p
        c = p + step
        h = max(o, c) + 2.5
        l = min(o, c) - 2.5
        h4_rows.append((o, h, l, c, 2500))
        p = c
    h1_rows = []
    p1 = 3945.0
    for i in range(80):
        step = -1.2 if i % 4 != 2 else 0.7
        o = p1
        c = p1 + step
        h = max(o, c) + 1.5
        l = min(o, c) - 1.5
        h1_rows.append((o, h, l, c, 1500))
        p1 = c
    m30_rows = []
    p30 = 3900.0
    for i in range(80):
        step = -0.6 if i < 72 else 0.9
        o = p30
        c = p30 + step
        h = max(o, c) + 1.1
        l = min(o, c) - 1.1
        m30_rows.append((o, h, l, c, 1100))
        p30 = c
    m5_rows = []
    pm5 = inside_support_price + 8.0
    for i in range(60):
        if i < 55:
            o = pm5
            c = pm5 - 0.25
        else:
            o = inside_support_price - 1.5 + (i - 55) * 0.4
            c = o + 0.65
        h = max(o, c) + 0.4
        l = min(o, c) - 0.5
        m5_rows.append((o, h, l, c, 900))
        pm5 = c

    end_t = datetime(2026, 11, 16, 10, 30)
    for key, rows_list, f_min, atr_v in (
        ("H4", h4_rows, 240, 10.0),
        ("H1", h1_rows, 60, 6.0),
        ("M30", m30_rows, 30, 4.0),
        ("M5", m5_rows, 5, 1.8),
    ):
        idx = pd.date_range(end=end_t, periods=len(rows_list), freq=f"{f_min}min")
        df_tf = pd.DataFrame(rows_list, columns=["Open", "High", "Low", "Close", "TickVolume"], index=idx)
        df_tf["ATR14"] = atr_v
        df_tf["EMA20"] = df_tf["Close"].rolling(20, min_periods=1).mean()
        df_tf["EMA50"] = df_tf["Close"].rolling(50, min_periods=1).mean()
        df_tf["RSI14"] = 44.0
        df_tf["HasTickVolume"] = True
        frames[key] = df_tf
    frames["M15"] = frames["M30"].copy()
    return frames


class TestSmcSnrContextIntegration(unittest.TestCase):
    """Suite Regresi Wajib Perbaikan Integrasi SNR Context + Shared Snapshot."""

    # 1. Import seluruh fungsi SNR
    def test_01_import_all_verified_snr_apis(self):
        """Pastikan seluruh API resmi SNR dapat diimpor & callable tanpa fungsi hantu lama."""
        for fn in (
            get_snr_config,
            run_snr_scan,
            rank_zones,
            detect_zone_breakout,
            detect_zone_retest,
            detect_zone_rejection,
            validate_snr_entry,
            evaluate_snr_market_bias,
            analyze_momentum,
            log_snr_error,
        ):
            self.assertTrue(callable(fn))

    # 2. SNR context berhasil dihitung & SMC membaca SNR context
    def test_02_snr_context_calculated_and_smc_reads_without_mutation(self):
        """extract_snr_supporting_context() berhasil menghitung SNR context dan SMC memakainya tanpa mutasi frame."""
        frames = build_realistic_multi_tf_frames()
        frames["M15"] = frames["M30"].copy()
        orig_h4 = frames["H4"].copy(deep=True)
        orig_m5 = frames["M5"].copy(deep=True)
        cp = float(frames["M5"]["Close"].iloc[-1])

        snr_ctx = extract_snr_supporting_context(frames, cp)
        self.assertIn(snr_ctx["status"], ("AVAILABLE", "PARTIAL"))
        self.assertTrue(snr_ctx["available"])
        self.assertIsNone(snr_ctx["error_type"])
        self.assertIsNone(snr_ctx["error_reason"])
        self.assertTrue(len(snr_ctx["supports"]) + len(snr_ctx["resistances"]) > 0)
        self.assertIn(snr_ctx["breakout_code"], ("NO_BREAKOUT", "VALID_BREAKOUT", "FALSE_BREAKOUT", "BREAKOUT_PENDING_CONFIRMATION"))
        self.assertIn(snr_ctx["retest_code"], ("NO_RETEST", "RETEST_WAITING", "RETEST_IN_PROGRESS", "RETEST_CONFIRMED", "RETEST_FAILED"))
        self.assertTrue(frames["H4"].equals(orig_h4))
        self.assertTrue(frames["M5"].equals(orig_m5))

        # Pastikan SMC Scanner & Diagnostic membaca SNR context nyata (bukan 'SNR context tidak tersedia')
        snap = create_market_snapshot(symbol="XAUUSD", frames=frames, live=False)
        smc_scan = scan_xauusd_zones(snapshot=snap)
        self.assertIn("snr_context", smc_scan)
        self.assertTrue(smc_scan["snr_context"]["available"])
        smc_dec = evaluate_trade_decision(snapshot=snap, scan_result=smc_scan)
        self.assertIsNotNone(smc_dec.get("snr_context"))
        self.assertTrue(smc_dec["snr_context"]["available"])
        diag_txt = format_diagnostic_report(smc_scan)
        self.assertIn("7. STATUS INTEGRASI SNR CONTEXT:", diag_txt)
        self.assertIn("AVAILABLE", diag_txt)

    # 3. SNR context gagal dan error tercatat
    def test_03_snr_context_failure_logs_error_and_marks_unavailable(self):
        """Kegagalan kalkulasi/API SNR harus tercatat di log, berstatus UNAVAILABLE, dan terlihat di diagnostik."""
        frames = build_realistic_multi_tf_frames()
        cp = float(frames["M5"]["Close"].iloc[-1])

        with patch("modules.snr.scanner.run_snr_scan", side_effect=RuntimeError("Simulasi kegagalan pipeline SNR")), \
             patch("modules.utils._catat_error") as mock_err_log, \
             patch("modules.snr.data_provider.log_snr_error") as mock_snr_log:
            res_err = extract_snr_supporting_context(frames, cp)

        self.assertEqual(res_err["status"], "UNAVAILABLE")
        self.assertFalse(res_err["available"])
        self.assertEqual(res_err["error_type"], "RuntimeError")
        self.assertIn("Simulasi kegagalan pipeline SNR", res_err["error_reason"])
        self.assertIn("[UNAVAILABLE]", res_err["summary"])
        self.assertTrue(mock_err_log.called)
        self.assertTrue(mock_snr_log.called)

        # Pastikan alasan kegagalan terlihat jelas di diagnostic & detailed scan
        fake_scan = {
            "data_ok": True,
            "symbol": "XAUUSD",
            "current_price": cp,
            "snapshot_price": cp,
            "snapshot_id": "SNAP_ERR_TEST",
            "snapshot_time": "2026-11-16 11:30:00 WIB",
            "htf_context": {"H4": "BEARISH", "H1": "BEARISH", "M30": "ALIGNED", "bias_utama": "BEARISH"},
            "diagnostics": {"by_timeframe": {}},
            "active_zones": [],
            "radar_zones": [],
            "rejected_zones": [],
            "snr_context": res_err,
        }
        diag_txt = format_diagnostic_report(fake_scan)
        self.assertIn("UNAVAILABLE", diag_txt)
        self.assertIn("Simulasi kegagalan pipeline SNR", diag_txt)

        det_txt = format_scan_presentation(fake_scan, dec={"status": "WAIT", "direction": None, "reasons": ["Wait"], "stages": {}, "snr_context": res_err}, detailed=True)
        self.assertIn("Simulasi kegagalan pipeline SNR", det_txt)

    # 4. H4/H1 bearish + harga di support + momentum bullish
    def test_04_bearish_trend_at_support_s1_bullish_momentum_pullback(self):
        """H4/H1 BEARISH + harga di Support S1 + momentum bullish -> COUNTER-TREND PULLBACK, SELL PRIORITY, WAIT."""
        smc_scan = {
            "symbol": "XAUUSD",
            "snapshot_price": 3875.50,
            "market_status": "OPEN",
            "data_freshness": "FRESH",
            "htf_context": {"H4": "BEARISH", "H1": "BEARISH", "h1_confirmation": "ALIGNED", "m30_setup_condition": "ALIGNED"},
            "m30_bias": "CONFIRMED",
            "zones": [{"zone_id": "SMC_SUPP_3914", "type": "SUPPLY", "low": 3914.61, "high": 3923.19, "score": 88.0, "lifecycle_status": "FRESH"}],
            "primary_zone_label": "SUPPLY 3914.61-3923.19",
        }
        snr_scan = {
            "symbol": "XAUUSD",
            "price": 3875.50,
            "market_status": "OPEN",
            "data_freshness": "FRESH",
            "market_bias": {"h4": "BEARISH", "h1": "BEARISH", "m30": "SIDEWAYS", "m5": "WAIT", "overall_bias": "BEARISH"},
            "ranked_zones": {
                "active_supports": [{"rank_label": "S1", "low": 3869.93, "high": 3878.88, "price_inside_zone": True, "score": 84.0, "grade": "A"}],
                "active_resistances": [{"rank_label": "R1", "low": 3914.61, "high": 3923.19, "price_inside_zone": False, "score": 87.0, "grade": "A"}],
            },
            "momentum_by_tf": {"M5": {"direction": "BULLISH", "strength": "MODERATE"}, "M30": {"direction": "BULLISH", "strength": "MODERATE"}},
            "price_action": {"breakout_code": "NO_BREAKOUT", "breakout_summary": "NOT CONFIRMED", "retest_code": "NO_RETEST", "retest_summary": "NOT CONFIRMED", "rejection_code": "WEAK_REJECTION", "rejection_summary": "PENDING", "momentum_summary": "BULLISH"},
            "entry_validation": {"zone_label": "SUPPORT S1", "trigger_status": "NOT CONFIRMED", "pullback_target_zone": {"rank_label": "R1", "low": 3914.61, "high": 3923.19}},
            "decision": {"signal": "WAIT", "direction": "NONE"},
        }
        ctx = build_unified_market_context(smc_scan=smc_scan, smc_dec={"status": "WAIT"}, snr_scan=snr_scan)
        self.assertEqual(ctx["classification"], "COUNTER-TREND PULLBACK")
        self.assertEqual(ctx["primary_trend"], "BEARISH")
        self.assertEqual(ctx["short_term"], "BULLISH PULLBACK")
        self.assertIn("PRICE AT SUPPORT S1 (3869.93-3878.88)", ctx["snr_context"])
        self.assertIn("3914.61-3923.19", ctx["smc_context"])
        self.assertIn("SUPPLY", ctx["smc_context"])
        self.assertEqual(ctx["trade_bias"], "SELL PRIORITY")
        self.assertEqual(ctx["entry_status"], "WAIT")
        self.assertNotIn("REVERSAL", ctx["classification"])

        # Entry validator juga menahan auto-BUY di Support S1 & menunjuk R1 berikutnya
        m5_df = _make_ohlc([
            (3879.0, 3879.5, 3874.5, 3875.5, 100),
            (3875.5, 3876.2, 3871.2, 3874.8, 120),
            (3874.8, 3877.5, 3872.0, 3877.2, 150),
            (3877.2, 3878.8, 3876.0, 3878.4, 160),
        ], atr_val=2.0)
        val = validate_snr_entry(
            {"active_supports": snr_scan["ranked_zones"]["active_supports"], "active_resistances": snr_scan["ranked_zones"]["active_resistances"], "broken_zones": [], "invalidated_zones": []},
            m5_df,
            market_bias=snr_scan["market_bias"],
            current_price=3878.4,
        )
        self.assertEqual(val["final_status"], "WAIT")
        self.assertEqual(val["buy_status"], "WAIT")
        self.assertIn("PRICE AT SUPPORT S1", val["snr_role_context"])
        self.assertIn("3914.61-3923.19", val["pullback_target_label"])

    # 5. H4/H1 bearish + harga di resistance
    def test_05_bearish_trend_at_resistance_r1(self):
        """H4/H1 BEARISH + harga di Resistance R1 -> SELL PRIORITY, context ALIGNED atau MOMENTUM CONFLICT."""
        base_smc = {
            "snapshot_price": 3918.0,
            "htf_context": {"H4": "BEARISH", "H1": "BEARISH", "h1_confirmation": "ALIGNED", "m30_setup_condition": "ALIGNED"},
            "zones": [{"type": "SUPPLY", "low": 3914.61, "high": 3923.19, "score": 88.0, "lifecycle_status": "FRESH"}],
        }
        snr_aligned = {
            "market_bias": {"h4": "BEARISH", "h1": "BEARISH", "m30": "BEARISH", "m5": "WAIT", "overall_bias": "BEARISH"},
            "ranked_zones": {"active_supports": [{"rank_label": "S1", "low": 3869.93, "high": 3878.88, "price_inside_zone": False}], "active_resistances": [{"rank_label": "R1", "low": 3914.61, "high": 3923.19, "price_inside_zone": True}]},
            "momentum_by_tf": {"M5": {"direction": "BEARISH"}, "M30": {"direction": "BEARISH"}},
            "price_action": {"momentum_summary": "BEARISH"},
            "entry_validation": {"zone_label": "RESISTANCE R1"},
        }
        ctx_al = build_unified_market_context(smc_scan=base_smc, smc_dec={"status": "WAIT"}, snr_scan=snr_aligned)
        self.assertEqual(ctx_al["classification"], "TREND ALIGNED")
        self.assertEqual(ctx_al["trade_bias"], "SELL PRIORITY")
        self.assertIn("PRICE AT RESISTANCE R1", ctx_al["snr_context"])
        self.assertEqual(ctx_al["entry_status"], "WAIT")

        snr_mom_conf = copy.deepcopy(snr_aligned)
        snr_mom_conf["momentum_by_tf"]["M5"] = {"direction": "BULLISH"}
        ctx_mc = build_unified_market_context(smc_scan=base_smc, smc_dec={"status": "WAIT"}, snr_scan=snr_mom_conf)
        self.assertEqual(ctx_mc["classification"], "MOMENTUM CONFLICT")
        self.assertEqual(ctx_mc["trade_bias"], "SELL PRIORITY")
        self.assertEqual(ctx_mc["entry_status"], "WAIT")

    # 6. H4/H1 bullish + harga di resistance
    def test_06_bullish_trend_at_resistance_r1(self):
        """H4/H1 BULLISH saat harga di Resistance R1 + momentum M5 BEARISH -> BUY PRIORITY, BEARISH PULLBACK."""
        smc_scan = {
            "snapshot_price": 3920.0,
            "htf_context": {"H4": "BULLISH", "H1": "BULLISH", "h1_confirmation": "ALIGNED", "m30_setup_condition": "PULLBACK"},
            "zones": [{"type": "DEMAND", "low": 3880.0, "high": 3888.0, "score": 85.0, "lifecycle_status": "FRESH"}],
        }
        snr_scan = {
            "market_bias": {"h4": "BULLISH", "h1": "BULLISH", "m30": "SIDEWAYS", "m5": "WAIT", "overall_bias": "BULLISH"},
            "ranked_zones": {
                "active_supports": [{"rank_label": "S1", "low": 3880.0, "high": 3888.0, "price_inside_zone": False}],
                "active_resistances": [{"rank_label": "R1", "low": 3918.0, "high": 3923.0, "price_inside_zone": True}],
            },
            "momentum_by_tf": {"M5": {"direction": "BEARISH"}, "M30": {"direction": "BEARISH"}},
            "price_action": {"momentum_summary": "BEARISH"},
            "entry_validation": {"zone_label": "RESISTANCE R1"},
        }
        ctx = build_unified_market_context(smc_scan=smc_scan, smc_dec={"status": "WAIT"}, snr_scan=snr_scan)
        self.assertEqual(ctx["classification"], "COUNTER-TREND PULLBACK")
        self.assertEqual(ctx["primary_trend"], "BULLISH")
        self.assertEqual(ctx["short_term"], "BEARISH PULLBACK")
        self.assertIn("PRICE AT RESISTANCE R1", ctx["snr_context"])
        self.assertIn("DEMAND (3880.00-3888.00)", ctx["smc_context"])
        self.assertEqual(ctx["trade_bias"], "BUY PRIORITY")
        self.assertEqual(ctx["entry_status"], "WAIT")

    # 7 & 8. Breakout valid & False breakout
    def test_07_and_08_breakout_valid_and_false_breakout(self):
        """Verifikasi deteksi VALID_BREAKOUT vs FALSE_BREAKOUT serta pencegahan false-positive dari 1 wick/pullback."""
        res_zone = {"zone_id": "R1", "zone_type": "RESISTANCE", "timeframe": "H1", "low": 3914.0, "high": 3916.0, "ref_atr": 4.0, "confirmed_index": 2}
        prefix = [
            (3910.0, 3912.0, 3908.0, 3911.0, 100),
            (3911.0, 3916.0, 3910.0, 3912.0, 100),
            (3912.0, 3914.0, 3911.0, 3913.0, 100),
        ]
        df_valid_bo = _make_ohlc(prefix + [
            (3913.0, 3921.0, 3912.8, 3920.2, 260),
            (3920.2, 3923.0, 3919.5, 3922.5, 240),
        ], atr_val=4.0, freq_min=60)
        bo_v = detect_zone_breakout(res_zone, df_valid_bo)
        self.assertEqual(bo_v["status"], "VALID_BREAKOUT")
        self.assertEqual(bo_v["direction"], "BULLISH")

        df_false_bo = _make_ohlc(prefix + [
            (3913.0, 3918.0, 3912.8, 3917.2, 180),
            (3917.2, 3917.4, 3911.0, 3911.8, 220),
        ], atr_val=4.0, freq_min=60)
        bo_f = detect_zone_breakout(res_zone, df_false_bo)
        self.assertEqual(bo_f["status"], "FALSE_BREAKOUT")

        # Wick 1 candle tanpa follow-through close tidak boleh langsung FALSE_BREAKOUT
        df_wick_only = _make_ohlc(prefix + [(3913.0, 3918.5, 3912.5, 3913.5, 140)], atr_val=4.0, freq_min=60)
        bo_w = detect_zone_breakout(res_zone, df_wick_only)
        self.assertNotEqual(bo_w["status"], "FALSE_BREAKOUT")

    # 9. Retest pending dan confirmed
    def test_09_retest_pending_and_confirmed(self):
        """Verifikasi RETEST_IN_PROGRESS (pending reaksi) dan RETEST_CONFIRMED serta anti-kontradiksi."""
        res_zone = {"zone_id": "R1", "zone_type": "RESISTANCE", "timeframe": "H1", "low": 3914.0, "high": 3916.0, "ref_atr": 4.0, "confirmed_index": 2}
        prefix = [
            (3910.0, 3912.0, 3908.0, 3911.0, 100),
            (3911.0, 3916.0, 3910.0, 3912.0, 100),
            (3912.0, 3914.0, 3911.0, 3913.0, 100),
        ]
        df_prog = _make_ohlc(prefix + [
            (3913.0, 3920.0, 3913.0, 3919.5, 220),
            (3919.5, 3919.8, 3915.1, 3915.3, 150),
        ], atr_val=4.0, freq_min=60)
        bo_p = detect_zone_breakout(res_zone, df_prog)
        rt_p = detect_zone_retest(res_zone, df_prog, breakout_result=bo_p)
        self.assertNotEqual(bo_p["status"], "FALSE_BREAKOUT")
        self.assertEqual(rt_p["status"], "RETEST_IN_PROGRESS")

        df_conf = _make_ohlc(prefix + [
            (3913.0, 3920.0, 3913.0, 3919.5, 220),
            (3919.5, 3919.8, 3915.2, 3916.5, 160),
            (3916.5, 3919.5, 3916.0, 3918.9, 210),
        ], atr_val=4.0, freq_min=60)
        rt_c = detect_zone_retest(res_zone, df_conf)
        self.assertEqual(rt_c["status"], "RETEST_CONFIRMED")
        self.assertEqual(rt_c["direction"], "BULLISH")

    # 10. Rejection valid dan false rejection
    def test_10_rejection_valid_false_and_inside_zone_safety(self):
        """Verifikasi BULLISH_REJECTION valid, FALSE_REJECTION sejati, dan pencegahan false rejection di dalam S1."""
        sup_zone = {"zone_id": "S1", "zone_type": "SUPPORT", "timeframe": "M5", "low": 3869.93, "high": 3878.88, "ref_atr": 2.0}
        df_valid_rej = _make_ohlc([
            (3880.0, 3880.5, 3878.0, 3879.0, 100),
            (3879.0, 3879.2, 3870.5, 3878.2, 190),
            (3878.2, 3881.5, 3877.8, 3881.0, 220),
        ], atr_val=2.0)
        r_ok = detect_zone_rejection(sup_zone, df_valid_rej)
        self.assertEqual(r_ok["status"], "BULLISH_REJECTION")
        self.assertTrue(r_ok["confirmed"])

        # Jebol jelas melewati batas bawah z_low (3869.93 - 0.15*2) -> FALSE_REJECTION sejati
        df_false_rej = _make_ohlc([
            (3880.0, 3880.5, 3878.0, 3879.0, 100),
            (3879.0, 3879.2, 3870.5, 3878.2, 190),
            (3878.2, 3878.4, 3865.0, 3866.0, 240),
        ], atr_val=2.0)
        r_false = detect_zone_rejection(sup_zone, df_false_rej)
        self.assertEqual(r_false["status"], "FALSE_REJECTION")

        # Masih di dalam S1 (close 3872.8 > 3869.93) tidak boleh FALSE_REJECTION
        df_inside = _make_ohlc([
            (3880.0, 3880.5, 3877.0, 3877.5, 100),
            (3877.5, 3878.0, 3874.5, 3877.6, 120),
            (3877.6, 3878.0, 3872.2, 3872.8, 130),
        ], atr_val=2.0)
        self.assertNotEqual(detect_zone_rejection(sup_zone, df_inside)["status"], "FALSE_REJECTION")

        # Out-of-zone tidak boleh FALSE_REJECTION
        df_out = _make_ohlc([(3895.0, 3896.0, 3894.5, 3895.5, 100), (3895.5, 3895.8, 3890.0, 3895.2, 120)], atr_val=2.0)
        self.assertEqual(detect_zone_rejection(sup_zone, df_out)["status"], "NONE")

    # 11. Data candle tidak lengkap (PARTIAL & UNAVAILABLE)
    def test_11_incomplete_candle_data_partial_and_unavailable(self):
        """Data kurang timeframe menjadi PARTIAL, data terlalu pendek/kosong menjadi UNAVAILABLE."""
        frames = build_realistic_multi_tf_frames()
        cp = float(frames["M5"]["Close"].iloc[-1])

        # Tanpa H4 -> PARTIAL (masih tersedia dari H1/M30/M5, missing_timeframes tercatat eksplisit)
        partial_frames = {"H1": frames["H1"], "M30": frames["M30"], "M5": frames["M5"]}
        res_part = extract_snr_supporting_context(partial_frames, cp)
        self.assertEqual(res_part["status"], "PARTIAL")
        self.assertTrue(res_part["available"])
        self.assertTrue(any("H4" in m for m in res_part["missing_timeframes"]))

        # Data terlalu pendek (< 20 bar di semua TF) -> UNAVAILABLE + error tercatat
        short_frames = {tf: df.tail(5).copy() for tf, df in frames.items() if isinstance(df, pd.DataFrame)}
        with patch("modules.utils._catat_error") as mock_log:
            res_short = extract_snr_supporting_context(short_frames, cp)
        self.assertEqual(res_short["status"], "UNAVAILABLE")
        self.assertFalse(res_short["available"])
        self.assertEqual(res_short["error_type"], "IncompleteDataError")
        self.assertTrue(mock_log.called)

        # Kosong sama sekali -> UNAVAILABLE
        res_empty = extract_snr_supporting_context({}, cp)
        self.assertEqual(res_empty["status"], "UNAVAILABLE")
        self.assertFalse(res_empty["available"])

    # 12. Shared snapshot SMC dan SNR
    def test_12_shared_snapshot_combined_scan_and_verification_headers(self):
        """scan gabungan memakai 1 snapshot untuk SMC & SNR, memverifikasi snapshot ID/time/harga/freshness."""
        frames = _make_bearish_frames(inside_support_price=3874.5)
        shared_snap = create_market_snapshot(symbol="XAUUSD", frames=frames, live=False)
        calls = {"n": 0}
        seen_ids = {}

        def _provider(symbol="XAUUSD"):
            calls["n"] += 1
            return shared_snap

        def _smc_spy(snap):
            seen_ids["smc"] = snap.get("snapshot_id")
            sc = scan_xauusd_zones(symbol="XAUUSD", snapshot=copy.deepcopy(snap))
            dc = evaluate_trade_decision(snapshot=copy.deepcopy(snap), scan_result=sc)
            return sc, dc

        def _snr_spy(snap):
            seen_ids["snr"] = snap.get("snapshot_id")
            sr = run_snr_scan(symbol="XAUUSD", snapshot=copy.deepcopy(snap), live=False)
            return {"scan_result": sr, "decision": sr.get("decision")}

        cyc = run_unified_scan(
            symbol="XAUUSD",
            snapshot=None,
            detailed=False,
            smc_runner=_smc_spy,
            snr_runner=_snr_spy,
            snapshot_provider=_provider,
        )
        self.assertEqual(calls["n"], 1)
        self.assertEqual(seen_ids["smc"], shared_snap["snapshot_id"])
        self.assertEqual(seen_ids["snr"], shared_snap["snapshot_id"])
        self.assertTrue(cyc["comparison"]["same_snapshot"])
        self.assertEqual(cyc["comparison"]["price_diff_points"], 0.0)
        self.assertIn(f"SNAPSHOT ID : {shared_snap['snapshot_id']} | SAME=True", cyc["formatted_report"])
        self.assertIn("FRESHNESS   : SMC=", cyc["formatted_report"])
        self.assertIn("MARKET CONTEXT", cyc["formatted_report"])
        self.assertNotIn("METODE TERBAIK", cyc["formatted_report"].upper())
        self.assertNotIn("KONFLIK METODE", cyc["formatted_report"].upper())

        # CLI `scan gabungan` dan `scan all detail` harus memanggil jalur shared snapshot yang sama
        parsed_g = xauusd1.parse_cli_command("scan gabungan")
        self.assertEqual(parsed_g, {"action": "SCAN", "method": "UNIFIED", "detailed": False})
        res_g = xauusd1.execute_cli_command("scan gabungan", snapshot=copy.deepcopy(shared_snap))
        self.assertEqual(res_g["snapshot_id"], shared_snap["snapshot_id"])
        self.assertIn("MARKET CONTEXT", res_g["output"])

        res_ad = xauusd1.execute_cli_command("scan all detail", snapshot=copy.deepcopy(shared_snap))
        self.assertEqual(res_ad["snapshot_id"], shared_snap["snapshot_id"])
        self.assertIn("XAUUSD SNR ADVANCED", res_ad["output"])

    # 13. Perintah scan lama tetap berfungsi
    def test_13_legacy_scan_commands_preserved(self):
        """Command lama (`scan`, `scan smc`, `scan detail`, `scan snr`, `scan snr detail`, `diag`) tetap berfungsi."""
        frames = build_realistic_multi_tf_frames()
        frames["M15"] = frames["M30"].copy()
        snap = create_market_snapshot(symbol="XAUUSD", frames=frames, live=False)

        for cmd in ("scan", "scan smc", "scan detail", "scan snr", "scan snr detail"):
            out = xauusd1.execute_cli_command(cmd, snapshot=copy.deepcopy(snap))
            self.assertIn("output", out)
            self.assertTrue(len(out["output"]) > 100, f"Output command '{cmd}' kosong")

        smc_scan = scan_xauusd_zones(snapshot=copy.deepcopy(snap))
        diag_out = format_diagnostic_report(smc_scan)
        self.assertIn("7. STATUS INTEGRASI SNR CONTEXT:", diag_out)

    # 14. Unified Monitor tetap berfungsi (+ cross-zone isolation & MT5 probe kondisional)
    def test_14_unified_monitor_and_cross_zone_isolation(self):
        """Unified Monitor berjalan lengkap dan _build_price_action_summary tidak mencampur FALSE_BREAKOUT zona lain."""
        frames = _make_bearish_frames(inside_support_price=3874.5)
        snap = create_market_snapshot(symbol="XAUUSD", frames=frames, live=False)
        for cmd in ("pantau", "pantau detail", "pantau smc", "pantau snr"):
            res = xauusd1.execute_cli_command(cmd, snapshot=copy.deepcopy(snap))
            self.assertIn("cycle_result", res)
            self.assertTrue(len(res["output"]) > 80)

        # Isolasi cross-zone PA
        s1 = {"zone_id": "S1", "rank_label": "S1", "zone_type": "SUPPORT", "low": 3869.93, "high": 3878.88, "ref_atr": 2.0}
        bz_false = {
            "zone_id": "H1_RES_OLD",
            "rank_label": "R2",
            "zone_type": "RESISTANCE",
            "low": 3890.0,
            "high": 3893.0,
            "breakout_info": {"status": "FALSE_BREAKOUT", "summary_status": "FALSE BREAKOUT", "direction": "BULLISH"},
            "retest_info": {"status": "RETEST_IN_PROGRESS", "summary_status": "IN PROGRESS", "direction": "BULLISH"},
        }
        m5 = _make_ohlc([(3879.0, 3879.5, 3876.0, 3877.0, 100), (3877.0, 3877.8, 3874.2, 3877.4, 130)], atr_val=2.0)
        pa = _build_price_action_summary(
            s1, {"broken_zones": [bz_false]}, m5, {"M5": m5},
            {"M5": {"available": True, "direction": "BULLISH", "strength": "MODERATE", "summary_label": "BULLISH"}},
            cfg=None,
        )
        self.assertNotEqual(pa["breakout_code"], "FALSE_BREAKOUT")
        self.assertEqual(pa["rejection_zone_label"], "S1 (3869.93-3878.88)")

        # Market Context fail-safe untuk DATA STALE / CLOSED
        ctx_stale = build_unified_market_context(
            smc_scan={"htf_context": {"H4": "BEARISH", "H1": "BEARISH"}, "data_freshness": "STALE"},
            smc_dec={"status": "WAIT"},
            snr_scan={"market_status": "OPEN", "data_freshness": "STALE"},
        )
        self.assertEqual(ctx_stale["classification"], "DATA UNAVAILABLE")
        self.assertEqual(ctx_stale["entry_status"], "DATA UNAVAILABLE")

        ctx_closed = build_unified_market_context(
            smc_scan={"htf_context": {"H4": "BEARISH", "H1": "BEARISH"}, "market_status": "CLOSED", "data_freshness": "FRESH"},
            smc_dec={"status": "WAIT"},
            snr_scan={"market_status": "CLOSED", "data_freshness": "FRESH", "market_bias": {"h4": "BEARISH", "h1": "BEARISH"}},
        )
        self.assertEqual(ctx_closed["entry_status"], "MARKET CLOSED")

        # Integration test MT5 live (hanya bila MetaTrader5 terhubung di lingkungan saat ini; aman di-skip bila offline)
        try:
            import MetaTrader5 as mt5
            if mt5.terminal_info() is not None and mt5.symbol_info_tick("XAUUSD") is not None:
                live_cyc = run_unified_scan(symbol="XAUUSD", detailed=False)
                self.assertIn("market_context", live_cyc)
        except Exception:
            pass


if __name__ == "__main__":
    unittest.main()
