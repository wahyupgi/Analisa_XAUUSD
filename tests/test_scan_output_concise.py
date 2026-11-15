# -*- coding: utf-8 -*-
"""
Pengujian komprehensif untuk PERBAIKAN OUTPUT SCAN — RINGKAS, JELAS, DAN MUDAH DIBACA:
1. Output Default (`scan` / detailed=False) hanya menampilkan 5 bagian:
   1. MARKET BIAS
   2. ZONA UTAMA
   3. ENTRY PLAN
   4. ENTRY CONFIRMATION
   5. FINAL DECISION
2. Detail audit (Raw candidates, seluruh kandidat, rincian scoring, rekonsiliasi jumlah zona,
   ranking berulang, alasan zona ditolak, diagnostik teknis internal) TIDAK tampil pada `scan`
   biasa, tetapi tetap tersedia lengkap pada `scan detail` (detailed=True).
3. Pemisahan status RADAR, PENDING PLAN READY (BELUM DIPASANG DI MT5), dan CONFIRMED ENTRY.
4. Seluruh nilai harga, skor, status, SL, TP, dan RR identik antara `scan` dan `scan detail`.
5. Command routing CLI (`scan`, `scan detail`, `zona`, `smc`, dll.) tetap berjalan tanpa regresi.
"""

import unittest
from datetime import datetime, timezone
import numpy as np
import pandas as pd

import xauusd1
from modules.zone_scanner import (
    Zone,
    create_market_snapshot,
    detect_early_setups,
    evaluate_trade_decision,
    format_scan_presentation,
    plan_pending_order,
    prioritize_radar_zones,
    scan_xauusd_zones,
)


def _make_sample_frames(n: int = 90) -> dict:
    base_time = datetime(2026, 11, 13, 12, 0, tzinfo=timezone.utc)
    frames = {}
    for tf, freq, step, atr_v in [
        ("D1", "1D", -2.0, 25.0),
        ("H4", "4h", -0.8, 12.0),
        ("H1", "1h", -0.4, 5.16),
        ("M30", "30min", -0.25, 3.60),
        ("M15", "15min", -0.15, 2.40),
        ("M5", "5min", -0.05, 1.40),
    ]:
        times = pd.date_range(end=base_time, periods=n, freq=freq)
        closes = np.array([3950.0 + step * i for i in range(n)], dtype=float)
        opens = closes - step * 0.5
        highs = np.maximum(opens, closes) + atr_v * 0.4
        lows = np.minimum(opens, closes) - atr_v * 0.4
        frames[tf] = pd.DataFrame(
            {
                "Time": times,
                "Open": opens,
                "High": highs,
                "Low": lows,
                "Close": closes,
                "Volume": np.full(n, 150.0),
                "ATR14": np.full(n, atr_v),
            }
        )
    return frames


def _make_zone_dict(
    zone_id: str,
    tf: str,
    zone_type: str,
    lo: float,
    hi: float,
    score: float,
    grade: str = "A",
    freshness: str = "FRESH",
    invalidation_price: float = None,
) -> dict:
    inval = invalidation_price if invalidation_price is not None else (lo if zone_type == "DEMAND" else hi)
    return {
        "zone_id": zone_id,
        "symbol": "XAUUSD",
        "timeframe": tf,
        "zone_type": zone_type,
        "price_low": lo,
        "price_high": hi,
        "midpoint": round((lo + hi) / 2.0, 2),
        "proximal_edge": hi if zone_type == "DEMAND" else lo,
        "distal_edge": lo if zone_type == "DEMAND" else hi,
        "zone_width": round(abs(hi - lo), 2),
        "invalidation_price": inval,
        "score": score,
        "grade": grade,
        "freshness": freshness,
        "is_valid": True,
        "touch_count": 0,
        "status": "RADAR",
        "scoring_status": "VALID",
        "origin_bos": {"tipe": "BOS", "arah": "bear" if zone_type == "SUPPLY" else "bull", "origin_valid": True},
        "fvg_confirmation": {"status": "UNTOUCHED", "size_atr": 1.3},
        "displacement_strength": 2.2,
        "liquidity_sweep_status": "CONFIRMED",
        "sl_recommended": (hi + 2.0) if zone_type == "SUPPLY" else (lo - 2.0),
        "tp1_recommended": (lo - 18.0) if zone_type == "SUPPLY" else (hi + 18.0),
        "tp2_recommended": (lo - 28.0) if zone_type == "SUPPLY" else (hi + 28.0),
    }


class TestScanOutputConcise(unittest.TestCase):
    def setUp(self):
        z_sup_primary = _make_zone_dict("Z_H1_SUP_1", "H1", "SUPPLY", 3914.61, 3923.19, score=84.5, grade="A")
        z_sup_alt = _make_zone_dict("Z_M30_SUP_2", "M30", "SUPPLY", 3930.00, 3934.50, score=76.0, grade="A")
        z_dem_ct = _make_zone_dict("Z_H1_DEM_1", "H1", "DEMAND", 3871.94, 3875.11, score=79.0, grade="A")
        z_far = _make_zone_dict("Z_H4_FAR", "H4", "SUPPLY", 4050.00, 4060.00, score=88.0, grade="A+")

        all_zones = [z_sup_primary, z_sup_alt, z_dem_ct, z_far]
        self.prio = prioritize_radar_zones(
            radar_zones=all_zones,
            current_price=3891.60,
            htf_bias="BEARISH",
            h1_structure="BEARISH",
            atr=5.16,
            bid=3891.48,
            ask=3891.72,
            snapshot_time="2026-11-14 10:30:00 WIB",
            snapshot_id="SNAP_CONCISE_TEST",
        )
        self.mock_scan = {
            "symbol": "XAUUSD",
            "current_price": 3891.60,
            "snapshot_price": 3891.60,
            "bid": 3891.48,
            "ask": 3891.72,
            "mid": 3891.60,
            "spread": 0.24,
            "snapshot_time": "2026-11-14 10:30:00 WIB",
            "snapshot_id": "SNAP_CONCISE_TEST",
            "radar_zones": all_zones,
            "active_zones": [z_sup_primary, z_sup_alt, z_dem_ct],
            "htf_context": {"H4": "BEARISH", "H1": "BEARISH", "M30": "BEARISH", "has_conflict": False},
            "radar_priority": self.prio,
            "atr_diagnostics": {
                "atr_source_timeframe": "H1",
                "atr_period": 14,
                "atr_value": 5.16,
                "atr_candle_timestamp": "2026-11-14 03:00:00 UTC",
                "atr_calculation_status": "VALID",
                "atr_fallback_status": "NONE (Primary H1)",
                "atr_candle_mode": "CLOSED_CANDLE",
            },
            "data_ok": True,
        }
        self.mock_dec_wait = {
            "status": "WAIT",
            "direction": None,
            "plan": None,
            "current_price": 3891.60,
            "snapshot_id": "SNAP_CONCISE_TEST",
            "radar_priority": self.prio,
            "stages": {
                "H4": {"ok": True, "msg": "Bias H4 BEARISH"},
                "H1": {"ok": True, "msg": "Struktur H1 BEARISH"},
                "M30": {"ok": True, "status": "CONFIRMED", "msg": "M30 BEARISH"},
                "zone_touch": {"status": "FAIL", "msg": "Harga belum menyentuh zona SUPPLY"},
                "liquidity_sweep": {"status": "NOT_CHECKED"},
                "structure_trigger": {"status": "NOT_CHECKED"},
                "displacement": {"status": "NOT_CHECKED"},
                "risk_reward": {"status": "NOT_CHECKED"},
            },
            "reasons": ["Harga belum menyentuh zona Supply"],
        }

    def test_01_default_scan_only_5_sections_and_no_audit_clutter(self):
        """1. Output default `scan` (detailed=False) hanya menampilkan 5 bagian dan bebas rincian audit."""
        out_concise = format_scan_presentation(self.mock_scan, self.mock_dec_wait, detailed=False)

        # Wajib memuat 5 bagian utama
        self.assertIn("1. MARKET BIAS", out_concise)
        self.assertIn("2. ZONA UTAMA", out_concise)
        self.assertIn("3. ENTRY PLAN", out_concise)
        self.assertIn("4. ENTRY CONFIRMATION", out_concise)
        self.assertIn("5. FINAL DECISION", out_concise)

        # Wajib TIDAK memuat detail audit yang dipindahkan ke `scan detail`
        forbidden_in_default = [
            "ATR Diagnostics",
            "[ZONE EVALUATION SUMMARY]",
            "[ZONE COUNT RECONCILIATION",
            "[MULTI-TIER RADAR ZONE",
            "[ZONE SCORING SUMMARY",
            "Raw / Historical Candidate Scoring",
            "Alasan Zona Diprioritaskan",
            "Alasan Zona Tidak Diprioritaskan",
            "[INVALIDATED / HISTORICAL ZONES",
            "[Kandidat Zona Pemantauan",
            "[HYBRID ENTRY CANDIDATES SUMMARY]",
            "[ENTRY VALIDATION SUMMARY]",
            "[CANDIDATE SELECTION OVERVIEW]",
            "[WAIT REASON BREAKDOWN]",
            "GLOBAL_M5_SWEEP_STATUS",
        ]
        for item in forbidden_in_default:
            self.assertNotIn(item, out_concise, f"'{item}' seharusnya tidak tampil pada `scan` biasa!")

        # Pastikan ringkas (mudah dibaca dalam beberapa detik, <= 45 baris)
        lines = out_concise.strip().splitlines()
        self.assertLessEqual(len(lines), 45, f"Output `scan` terlalu panjang ({len(lines)} baris)!")

    def test_02_scan_detail_preserves_all_audit_information(self):
        """2. Output `scan detail` (detailed=True) tetap menampilkan seluruh informasi teknis & audit."""
        out_detail = format_scan_presentation(self.mock_scan, self.mock_dec_wait, detailed=True)

        required_in_detail = [
            "ATR Diagnostics",
            "[ZONE EVALUATION SUMMARY]:",
            "[ZONE COUNT RECONCILIATION — AUDIT DISTRIBUSI KANDIDAT]:",
            "[MULTI-TIER RADAR ZONE — RISK-BASED SELEKSI RESMI PYTHON SCANNER]:",
            "[ZONE SCORING SUMMARY — DEFINISI RANKING TERPISAH]:",
            "Raw / Historical Candidate Scoring",
            "[Kandidat Zona Pemantauan — BUKAN Sinyal Entry / Pending Order Aktif]:",
            "[HYBRID ENTRY CANDIDATES SUMMARY]:",
            "[ENTRY VALIDATION SUMMARY]:",
            "[CANDIDATE SELECTION OVERVIEW]:",
            "[WAIT REASON BREAKDOWN]:",
        ]
        for item in required_in_detail:
            self.assertIn(item, out_detail, f"'{item}' wajib tetap tersedia pada `scan detail`!")

    def test_03_status_separation_radar_pending_ready_vs_confirmed_wait(self):
        """3. Pemisahan tegas antara RADAR, PENDING PLAN READY (BELUM DIPASANG DI MT5), dan CONFIRMED ENTRY WAIT."""
        pending_setup = {
            "valid": True,
            "status": "READY",
            "order_type": "SELL LIMIT",
            "entry_price": 3914.61,
            "entry_reason": "Proximal Supply H1",
            "sl": 3925.19,
            "tp1": 3893.00,
            "tp2": 3880.00,
            "risk": 10.58,
            "reward": 21.61,
            "rr": 2.04,
            "invalidation_price": 3923.19,
            "suggested_expiration": "2026-11-14 18:30 WIB (8 jam)",
        }
        out_concise = format_scan_presentation(
            self.mock_scan,
            self.mock_dec_wait,
            early_setups={"best_setup": pending_setup},
            detailed=False,
        )

        # 1. Status Zona: RADAR
        self.assertIn("- Status Zona      : RADAR", out_concise)
        # 2. Status Rencana memisahkan Pending Plan READY dan Confirmed Entry WAIT
        self.assertIn("- Status Rencana   : READY (Pending Plan) | WAIT (Confirmed Entry)", out_concise)
        self.assertIn("- Pending Plan     : READY — PENDING PLAN (BELUM DIPASANG DI MT5) [SELL LIMIT]", out_concise)
        self.assertIn("- Confirmed Entry  : WAIT (Bukan sinyal entry terkonfirmasi; menunggu validasi M5 lengkap)", out_concise)
        # 3. Nilai Entry, SL, TP, RR identik dengan pending_setup
        self.assertIn("3914.61", out_concise)
        self.assertIn("3925.19", out_concise)
        self.assertIn("TP1 3893.00 | TP2 3880.00", out_concise)
        self.assertIn("1:2.0", out_concise)
        # 4. Final Decision tetap WAIT (tidak menampilkan Pending Plan READY seolah sinyal Confirmed Entry)
        self.assertIn("- Status           : WAIT (NO ENTRY / WAIT", out_concise)
        self.assertNotIn("SELL VALID", out_concise)

    def test_04_confirmed_entry_valid_displays_official_signal(self):
        """4. Ketika seluruh validasi M5 terpenuhi, Confirmed Entry dan Final Decision menampilkan sinyal resmi."""
        confirmed_plan = {
            "entry_acuan": 3915.20,
            "entry_min": 3914.61,
            "entry_max": 3916.00,
            "sl": 3925.19,
            "tp1": 3890.00,
            "tp2": 3875.00,
            "rr": 2.52,
        }
        prio_valid = dict(self.prio)
        prio_valid["best_validated_entry_zone"] = self.prio["primary_directional_zone"]
        dec_valid = {
            "status": "VALID",
            "direction": "SELL",
            "zone": self.prio["primary_directional_zone"],
            "plan": confirmed_plan,
            "current_price": 3915.20,
            "radar_priority": prio_valid,
            "stages": {
                "H4": {"ok": True, "msg": "H4 BEARISH"},
                "H1": {"ok": True, "msg": "H1 BEARISH"},
                "M30": {"ok": True, "status": "CONFIRMED", "msg": "M30 BEARISH"},
                "zone_touch": {"status": "CONFIRMED"},
                "liquidity_sweep": {"status": "CONFIRMED", "msg": "BSL sweep confirmed"},
                "structure_trigger": {"status": "CONFIRMED", "msg": "CHoCH bearish M5"},
                "displacement": {"status": "CONFIRMED", "msg": "Displacement 2.1 ATR"},
                "risk_reward": {"status": "CONFIRMED", "rr": 2.52},
            },
            "reasons": ["Seluruh konfirmasi M5 terpenuhi"],
        }
        out_concise = format_scan_presentation(self.mock_scan, dec_valid, detailed=False)
        out_detail = format_scan_presentation(self.mock_scan, dec_valid, detailed=True)

        # Pastikan nilai harga, Entry, SL, TP1, TP2, dan RR identik pada `scan` dan `scan detail`
        for expected_val in ["3915.20", "3914.61", "3925.19", "3890.00", "3875.00", "1:2.5", "SELL VALID"]:
            self.assertIn(expected_val, out_concise)
            self.assertIn(expected_val, out_detail)

    def test_05_cli_commands_scan_vs_scan_detail_identical_metrics(self):
        """5. Command `scan` dan `scan detail` melalui execute_cli_command menggunakan perhitungan yang 100% identik."""
        frames = _make_sample_frames()
        snap = create_market_snapshot(frames=frames, current_price=3891.60, live=False)

        res_scan = xauusd1.execute_cli_command("scan", snapshot=snap)
        res_detail = xauusd1.execute_cli_command("scan detail", snapshot=snap)

        # Objek hasil scan dan decision memiliki metrik yang identik
        self.assertEqual(res_scan["scan_result"]["current_price"], res_detail["scan_result"]["current_price"])
        self.assertEqual(res_scan["decision"]["status"], res_detail["decision"]["status"])
        self.assertEqual(res_scan["decision"]["direction"], res_detail["decision"]["direction"])
        self.assertEqual(res_scan["decision"]["plan"], res_detail["decision"]["plan"])

        # Output `scan` ringkas, output `scan detail` lengkap
        self.assertIn("1. MARKET BIAS", res_scan["output"])
        self.assertIn("5. FINAL DECISION", res_scan["output"])
        self.assertNotIn("ATR Diagnostics", res_scan["output"])
        self.assertNotIn("[HYBRID ENTRY CANDIDATES SUMMARY]:", res_scan["output"])
        self.assertNotIn("[WAIT REASON BREAKDOWN]:", res_scan["output"])
        self.assertIn("ATR Diagnostics", res_detail["output"])
        self.assertIn("[HYBRID ENTRY CANDIDATES SUMMARY]:", res_detail["output"])
        self.assertIn("[WAIT REASON BREAKDOWN]:", res_detail["output"])


if __name__ == "__main__":
    unittest.main()
