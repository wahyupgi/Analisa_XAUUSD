# -*- coding: utf-8 -*-
"""
test_unified_monitor.py — Pengujian Komprehensif Wajib untuk Unified Monitor Mode (SMC + SNR + Account).
"""
import copy
import unittest

import xauusd1
from modules.snr.scanner import evaluate_snr_decision, run_snr_scan
from modules.unified_monitor import (
    SmcAlertTracker,
    UnifiedMonitor,
    collect_account_monitor_data,
    format_account_monitor_section,
    parse_monitor_command,
)
from modules.zone_scanner import (
    create_market_snapshot,
    evaluate_trade_decision,
    scan_xauusd_zones,
)
from tests.test_snr_scanner import build_realistic_multi_tf_frames


def _make_mock_smc_runner(signal_str="WAIT", invalid_cnt=0):
    def _runner(snap):
        is_valid = signal_str in ("BUY", "SELL")
        return {
            "summary": {
                "method": "SMC",
                "status": "OK",
                "h4_bias": "BULLISH" if signal_str == "BUY" else ("BEARISH" if signal_str == "SELL" else "NETRAL"),
                "h1_bias": "BULLISH" if signal_str == "BUY" else ("BEARISH" if signal_str == "SELL" else "NETRAL"),
                "m30_bias": "STRONG" if is_valid else "WAIT",
                "m5_bias": "CONFIRMED" if is_valid else "WAIT",
                "structure_status": "ALIGNED",
                "primary_zone_label": "H1 DEMAND 3870.00-3874.00 (Grade A, Dist: 2.00 pts)",
                "nearest_zone_label": "H1 DEMAND 3870.00-3874.00 (Grade A, Dist: 2.00 pts)",
                "fresh_count": 2,
                "tested_count": 1,
                "mitigated_count": 0,
                "invalidated_count": invalid_cnt,
                "last_invalidated_type": "SUPPLY",
                "zone_touch": "CONFIRMED" if is_valid else "WAIT",
                "liquidity_sweep": "CONFIRMED" if is_valid else "WAIT",
                "choch_bos": "CONFIRMED" if is_valid else "WAIT",
                "displacement": "CONFIRMED" if is_valid else "WAIT",
                "rr_stage": "CONFIRMED" if is_valid else "WAIT",
                "decision_signal": signal_str,
                "direction": signal_str if is_valid else "NONE",
                "entry": 3874.00 if is_valid else None,
                "sl": 3868.00 if is_valid else None,
                "tp1": 3888.00 if is_valid else None,
                "tp2": 3895.00 if is_valid else None,
                "rr": 2.33 if is_valid else None,
                "main_reason": f"SMC {signal_str} decision.",
            }
        }
    return _runner


def _make_mock_snr_runner(signal_str="WAIT", breakout_code="NO_BREAKOUT", rejection_code="NONE"):
    def _runner(snap):
        is_valid = signal_str in ("BUY", "SELL")
        scan_res = {
            "price": 3895.00,
            "snr_status": f"{signal_str} BIAS",
            "market_bias": {
                "h4": "BULLISH" if signal_str == "BUY" else "BEARISH",
                "h1": "BULLISH" if signal_str == "BUY" else "BEARISH",
                "m30": "SIDEWAYS",
                "m5": signal_str if is_valid else "WAIT",
                "overall_bias": "BULLISH" if signal_str == "BUY" else "BEARISH",
                "snr_status": "BULLISH BIAS" if signal_str == "BUY" else "BEARISH BIAS",
            },
            "ranked_zones": {
                "active_supports": [{
                    "rank_label": "S1",
                    "low": 3865.00,
                    "high": 3869.00,
                    "grade": "A",
                    "score": 85.0,
                    "strength_short": "STRONG",
                    "touch_status": "TESTED_1X",
                    "touch_count": 1,
                    "distance_points": 26.0,
                    "touch_analysis": {"classification": "TESTED_1X", "valid_touch_count": 1, "touch_quality_score_15": 13.0},
                }],
                "active_resistances": [{
                    "rank_label": "R1",
                    "low": 3914.00,
                    "high": 3922.00,
                    "grade": "A",
                    "score": 88.0,
                    "strength_short": "STRONG",
                    "touch_status": "UNTESTED",
                    "touch_count": 0,
                    "distance_points": 19.0,
                    "touch_analysis": {"classification": "UNTESTED", "valid_touch_count": 0, "touch_quality_score_15": 11.5},
                }],
            },
            "price_action": {
                "breakout_code": breakout_code,
                "breakout_direction": "BULLISH" if breakout_code == "VALID_BREAKOUT" else "NONE",
                "breakout_summary": "CONFIRMED" if breakout_code == "VALID_BREAKOUT" else "NOT CONFIRMED",
                "retest_code": "NO_RETEST",
                "retest_summary": "NOT CONFIRMED",
                "rejection_code": rejection_code,
                "rejection_summary": "CONFIRMED" if rejection_code != "NONE" else "WAIT",
                "momentum_summary": "BULLISH" if signal_str == "BUY" else "BEARISH",
            },
            "entry_validation": {
                "zone_label": "SUPPORT S1" if signal_str == "BUY" else "RESISTANCE R1",
                "trigger_status": "CONFIRMED" if is_valid else "NOT CONFIRMED",
                "buy_status": "CONFIRMED" if signal_str == "BUY" else "WAIT",
                "sell_status": "CONFIRMED" if signal_str == "SELL" else "WAIT",
            },
            "radar": {
                "radar_status": f"CONFIRMED_{signal_str}" if is_valid else "STANDBY",
                "nearest_support": {"label": "S1", "low": 3865.0, "high": 3869.0, "state": "DISTANT"},
                "nearest_resistance": {"label": "R1", "low": 3914.0, "high": 3922.0, "state": "DISTANT"},
                "broken_zone_count": 0,
                "invalidated_zone_count": 0,
            },
        }
        decision = {
            "method": "SNR",
            "signal": signal_str,
            "direction": signal_str if is_valid else "NONE",
            "setup_type": "SUPPORT_REJECTION" if signal_str == "BUY" else ("RESISTANCE_REJECTION" if signal_str == "SELL" else "NONE"),
            "zone_label": "SUPPORT S1" if signal_str == "BUY" else "RESISTANCE R1",
            "entry": 3869.50 if is_valid else None,
            "sl": 3863.80 if is_valid else None,
            "tp1": 3913.50 if is_valid else None,
            "tp2": 3922.00 if is_valid else None,
            "rr_display": "1 : 2.40" if is_valid else "NOT CHECKED",
            "reason": f"SNR {signal_str} decision.",
        }
        return {"scan_result": scan_res, "decision": decision}
    return _runner


class TestUnifiedMonitorMode(unittest.TestCase):
    """Suite Pengujian Wajib untuk Unified Monitor Mode (SMC + SNR + Account)."""

    def test_01_simultaneous_execution_and_three_sections_output(self):
        """Pastikan SMC Monitor, SNR Monitor, dan Account Monitor berjalan bersamaan dalam satu siklus."""
        frames = build_realistic_multi_tf_frames()
        frames["M15"] = frames["M30"].copy()
        snap = create_market_snapshot(symbol="XAUUSD", frames=frames, live=False)

        mon = UnifiedMonitor(symbol="XAUUSD")
        res = mon.run_cycle(
            snapshot=snap,
            mode="unified",
            detailed=False,
            pos_list=[{"ticket": 101, "profit": 42.50, "sl": 3880.0, "tp": 3920.0}],
            orders_list=[{"ticket": 202, "sl": 0.0, "tp": 3930.0}],
        )

        self.assertEqual(res["smc"]["status"], "OK")
        self.assertEqual(res["snr"]["status"], "OK")
        self.assertTrue(res["account"]["available"])
        self.assertEqual(res["account"]["open_positions"], 1)
        self.assertEqual(res["account"]["pending_orders"], 1)
        self.assertEqual(res["account"]["pending_without_sl"], 1)

        report = res["formatted_report"]
        self.assertIn("### A. SMC MONITOR", report)
        self.assertIn("### B. SNR MONITOR", report)
        self.assertIn("### C. ACCOUNT MONITOR", report)
        self.assertIn("WARNING: Terdapat 1 pending order TANPA Stop Loss!", report)

    def test_02_all_signal_combinations_allowed_without_conflict_label(self):
        """
        Pastikan seluruh kombinasi arah sinyal kedua metode diperbolehkan:
        - SMC = BUY, SNR = WAIT
        - SMC = SELL, SNR = BUY
        - SMC = WAIT, SNR = SELL
        - SMC = BUY, SNR = BUY
        - SMC = WAIT, SNR = WAIT
        Tanpa label konflik antar-metode atau klaim satu metode lebih unggul.
        """
        combinations = [
            ("BUY", "WAIT"),
            ("SELL", "BUY"),
            ("WAIT", "SELL"),
            ("BUY", "BUY"),
            ("WAIT", "WAIT"),
        ]
        snap = {"symbol": "XAUUSD", "snapshot_price": 3895.0, "spread": 0.25, "frames": {}, "live": False}

        for smc_sig, snr_sig in combinations:
            mon = UnifiedMonitor(
                symbol="XAUUSD",
                smc_runner=_make_mock_smc_runner(smc_sig),
                snr_runner=_make_mock_snr_runner(snr_sig),
            )
            res = mon.run_cycle(snapshot=snap, mode="unified", detailed=False)
            self.assertEqual(res["smc_signal"], smc_sig)
            self.assertEqual(res["snr_signal"], snr_sig)
            self.assertIn(f"INDEPENDENT SIGNALS : SMC = {smc_sig} | SNR = {snr_sig}", res["formatted_report"])
            # Pastikan tidak ada penghakiman/perbandingan antar metode
            self.assertNotIn("METODE TERBAIK", res["formatted_report"].upper())
            self.assertNotIn("KONFLIK METODE", res["formatted_report"].upper())

    def test_03_smc_error_does_not_stop_snr(self):
        """Jika SMC gagal (melempar exception), tampilkan status ERROR pada SMC dan lanjutkan SNR + Account."""
        def _failing_smc(_snap):
            raise RuntimeError("Simulated SMC internal failure")

        snap = {"symbol": "XAUUSD", "snapshot_price": 3895.0, "spread": 0.25, "frames": {}, "live": False}
        mon = UnifiedMonitor(
            symbol="XAUUSD",
            smc_runner=_failing_smc,
            snr_runner=_make_mock_snr_runner("BUY"),
        )
        res = mon.run_cycle(snapshot=snap, mode="unified", detailed=False)

        self.assertEqual(res["smc"]["status"], "ERROR")
        self.assertEqual(res["snr"]["status"], "OK")
        self.assertEqual(res["snr_signal"], "BUY")
        self.assertIn("### A. SMC MONITOR\nSTATUS          : ERROR", res["formatted_report"])
        self.assertIn("### B. SNR MONITOR\nSTATUS          : OK", res["formatted_report"])

    def test_04_snr_error_does_not_stop_smc(self):
        """Jika SNR gagal (melempar exception), tampilkan status ERROR pada SNR dan lanjutkan SMC + Account."""
        def _failing_snr(_snap):
            raise ValueError("Simulated SNR internal failure")

        snap = {"symbol": "XAUUSD", "snapshot_price": 3895.0, "spread": 0.25, "frames": {}, "live": False}
        mon = UnifiedMonitor(
            symbol="XAUUSD",
            smc_runner=_make_mock_smc_runner("SELL"),
            snr_runner=_failing_snr,
        )
        res = mon.run_cycle(snapshot=snap, mode="unified", detailed=False)

        self.assertEqual(res["smc"]["status"], "OK")
        self.assertEqual(res["smc_signal"], "SELL")
        self.assertEqual(res["snr"]["status"], "ERROR")
        self.assertIn("### A. SMC MONITOR\nSTATUS          : OK", res["formatted_report"])
        self.assertIn("### B. SNR MONITOR\nSTATUS          : ERROR", res["formatted_report"])

    def test_05_mt5_data_consistency_single_fetch_per_cycle(self):
        """Pastikan data MT5 hanya diambil 1x per siklus dan dibagikan secara konsisten ke SMC dan SNR."""
        frames = build_realistic_multi_tf_frames()
        frames["M15"] = frames["M30"].copy()
        base_snap = create_market_snapshot(symbol="XAUUSD", frames=frames, live=False)

        fetch_counter = {"calls": 0}
        received_ids = {"smc": None, "snr": None}

        def _counting_provider(symbol="XAUUSD"):
            fetch_counter["calls"] += 1
            return base_snap

        def _spy_smc(snap):
            received_ids["smc"] = snap.get("snapshot_id")
            return _make_mock_smc_runner("WAIT")(snap)

        def _spy_snr(snap):
            received_ids["snr"] = snap.get("snapshot_id")
            return _make_mock_snr_runner("WAIT")(snap)

        mon = UnifiedMonitor(
            symbol="XAUUSD",
            smc_runner=_spy_smc,
            snr_runner=_spy_snr,
            snapshot_provider=_counting_provider,
        )
        res = mon.run_cycle(snapshot=None, mode="unified", detailed=False)

        self.assertEqual(fetch_counter["calls"], 1, "Snapshot provider harus dipanggil tepat 1 kali per siklus")
        self.assertEqual(received_ids["smc"], base_snap["snapshot_id"])
        self.assertEqual(received_ids["snr"], base_snap["snapshot_id"])
        self.assertEqual(res["snapshot_id"], base_snap["snapshot_id"])

    def test_06_non_repeating_state_change_alerts(self):
        """
        Pastikan alert beridentitas [SMC] dan [SNR] muncul saat terjadi perubahan status,
        dan TIDAK diulang pada siklus berikutnya jika status tidak berubah.
        """
        snap = {"symbol": "XAUUSD", "snapshot_price": 3895.0, "spread": 0.25, "frames": {}, "live": False}
        state_box = {
            "smc_sig": "BUY",
            "smc_inv": 0,
            "snr_sig": "SELL",
            "snr_bo": "VALID_BREAKOUT",
            "snr_rej": "BULLISH_REJECTION",
        }

        def _dyn_smc(s):
            return _make_mock_smc_runner(state_box["smc_sig"], invalid_cnt=state_box["smc_inv"])(s)

        def _dyn_snr(s):
            return _make_mock_snr_runner(
                state_box["snr_sig"],
                breakout_code=state_box["snr_bo"],
                rejection_code=state_box["snr_rej"],
            )(s)

        mon = UnifiedMonitor(symbol="XAUUSD", smc_runner=_dyn_smc, snr_runner=_dyn_snr)

        # Siklus 1: State baru -> harus menghasilkan alert [SMC] dan [SNR]
        c1 = mon.run_cycle(snapshot=snap, mode="unified")
        self.assertIn("[SMC] BUY SETUP TERKONFIRMASI", c1["alerts"])
        self.assertIn("[SNR] SELL SETUP TERKONFIRMASI", c1["alerts"])
        self.assertIn("[SNR] RESISTANCE BREAKOUT TERKONFIRMASI", c1["alerts"])
        self.assertIn("[SNR] SUPPORT REJECTION TERKONFIRMASI", c1["alerts"])
        self.assertTrue(c1["state_changed"])

        # Siklus 2: State tidak berubah -> alert TIDAK BOLEH berulang!
        c2 = mon.run_cycle(snapshot=snap, mode="unified")
        self.assertEqual(c2["alerts"], [])
        self.assertFalse(c2["state_changed"])

        # Siklus 3: Zona Supply SMC berubah menjadi Invalid -> muncul [SMC] ZONA SUPPLY INVALID
        state_box["smc_inv"] = 1
        c3 = mon.run_cycle(snapshot=snap, mode="unified")
        self.assertIn("[SMC] ZONA SUPPLY INVALID", c3["alerts"])
        self.assertTrue(c3["state_changed"])

    def test_07_all_pantau_commands_and_regression(self):
        """
        Uji command:
        - `pantau`
        - `pantau detail`
        - `pantau smc`
        - `pantau smc detail`
        - `pantau snr`
        - `pantau snr detail`
        Serta pastikan hasil analisis masing-masing metode di dalam Unified Monitor
        identik dengan ketika dijalankan secara terpisah (Regression Test).
        """
        frames = build_realistic_multi_tf_frames()
        frames["M15"] = frames["M30"].copy()
        snap = create_market_snapshot(symbol="XAUUSD", frames=frames, live=False)

        # Jalankan SMC & SNR secara mandiri terlebih dahulu
        smc_scan_solo = scan_xauusd_zones(symbol="XAUUSD", snapshot=copy.deepcopy(snap))
        smc_dec_solo = evaluate_trade_decision(snapshot=copy.deepcopy(snap), scan_result=smc_scan_solo)

        snr_scan_solo = run_snr_scan(symbol="XAUUSD", snapshot=copy.deepcopy(snap), live=False)
        snr_dec_solo = evaluate_snr_decision(snr_scan_solo)

        # 1. `pantau` (Unified ringkas)
        r_unified = xauusd1.execute_cli_command("pantau", snapshot=snap)
        cyc_u = r_unified["cycle_result"]
        self.assertIn("### A. SMC MONITOR", r_unified["output"])
        self.assertIn("### B. SNR MONITOR", r_unified["output"])
        self.assertIn("### C. ACCOUNT MONITOR", r_unified["output"])

        # Regression check: hasil SMC & SNR di dalam Unified Monitor harus sama persis dengan eksekusi mandiri
        self.assertEqual(cyc_u["smc"]["decision"]["status"], smc_dec_solo["status"])
        self.assertEqual(cyc_u["smc"]["decision"]["direction"], smc_dec_solo["direction"])
        self.assertEqual(cyc_u["snr"]["decision"]["signal"], snr_dec_solo["signal"])
        self.assertEqual(cyc_u["snr"]["decision"]["direction"], snr_dec_solo["direction"])

        # 2. `pantau detail` (Unified detail)
        r_unified_det = xauusd1.execute_cli_command("pantau detail", snapshot=snap)
        self.assertIn("XAUUSD PYTHON SMC SCANNER", r_unified_det["output"])
        self.assertIn("XAUUSD SNR ADVANCED — FULL AUDIT DETAIL", r_unified_det["output"])

        # 3. `pantau smc` & `pantau smc detail`
        r_smc = xauusd1.execute_cli_command("pantau smc", snapshot=snap)
        self.assertIn("### A. SMC MONITOR", r_smc["output"])
        self.assertNotIn("### B. SNR MONITOR", r_smc["output"])

        r_smc_det = xauusd1.execute_cli_command("pantau smc detail", snapshot=snap)
        self.assertIn("### A. SMC MONITOR", r_smc_det["output"])
        self.assertIn("XAUUSD PYTHON SMC SCANNER", r_smc_det["output"])
        self.assertNotIn("### B. SNR MONITOR", r_smc_det["output"])

        # 4. `pantau snr` & `pantau snr detail`
        r_snr = xauusd1.execute_cli_command("pantau snr", snapshot=snap)
        self.assertIn("### B. SNR MONITOR", r_snr["output"])
        self.assertNotIn("### A. SMC MONITOR", r_snr["output"])

        r_snr_det = xauusd1.execute_cli_command("pantau snr detail", snapshot=snap)
        self.assertIn("### B. SNR MONITOR", r_snr_det["output"])
        self.assertIn("[22. WAIT REASON BREAKDOWN]", r_snr_det["output"])
        self.assertNotIn("### A. SMC MONITOR", r_snr_det["output"])


if __name__ == "__main__":
    unittest.main()
