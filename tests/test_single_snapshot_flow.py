# -*- coding: utf-8 -*-
"""
Test Suite: Single Snapshot Data Flow — XAUUSD Python SMC Scanner.
Memverifikasi seluruh 12 skenario pengujian wajib (Single Scan, Single Tick,
Shared Scan Result, Shared Snapshot ID, No Hidden Rescan, TypeError Handling,
Price/Distance Consistency, Priority Composite Lock, Entry Validation,
Backward Compatibility, dan Error Handling DATA_INCOMPLETE).
"""

import os
import sys
import types
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import modules.zone_scanner as zs
from modules.zone_scanner import (
    create_market_snapshot,
    evaluate_trade_decision,
    format_diagnostic_report,
    format_scan_presentation,
    prioritize_radar_zones,
    scan_xauusd_zones,
)


def _make_ohlcv(n: int = 80, base: float = 3900.0, trend: float = -0.1, atr_val: float = 5.0,
                freq: str = "5min", end_dt: datetime = None) -> pd.DataFrame:
    if end_dt is None:
        end_dt = datetime(2026, 11, 13, 6, 0)
    times = pd.date_range(end=end_dt, periods=n, freq=freq)
    closes = np.array([base + trend * i for i in range(n)], dtype=float)
    opens = closes - 0.5 * trend
    highs = np.maximum(opens, closes) + atr_val * 0.4
    lows = np.minimum(opens, closes) - atr_val * 0.4
    df = pd.DataFrame(
        {
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": np.full(n, 100.0),
            "ATR14": np.full(n, atr_val),
        },
        index=times,
    )
    return df


def _make_frames(base: float = 3900.0, end_dt: datetime = None) -> dict:
    if end_dt is None:
        end_dt = datetime(2026, 11, 13, 6, 0)
    return {
        "H4": _make_ohlcv(80, base=base + 30.0, trend=-0.4, atr_val=8.0, freq="4h", end_dt=end_dt),
        "H1": _make_ohlcv(80, base=base + 20.0, trend=-0.3, atr_val=5.0, freq="1h", end_dt=end_dt),
        "M30": _make_ohlcv(80, base=base + 10.0, trend=-0.2, atr_val=4.0, freq="30min", end_dt=end_dt),
        "M15": _make_ohlcv(80, base=base + 5.0, trend=-0.1, atr_val=3.0, freq="15min", end_dt=end_dt),
        "M5": _make_ohlcv(80, base=base, trend=-0.05, atr_val=2.0, freq="5min", end_dt=end_dt),
    }


def _make_zone(zone_id: str, tf: str, zone_type: str, lo: float, hi: float,
               score: float = 78.0, grade: str = "A", freshness: str = "FRESH") -> dict:
    return {
        "zone_id": zone_id,
        "timeframe": tf,
        "zone_type": zone_type,
        "price_low": lo,
        "price_high": hi,
        "proximal_edge": hi if zone_type == "DEMAND" else lo,
        "distal_edge": lo if zone_type == "DEMAND" else hi,
        "zone_width": round(abs(hi - lo), 2),
        "invalidation_price": lo if zone_type == "DEMAND" else hi,
        "score": score,
        "grade": grade,
        "freshness": freshness,
        "is_valid": True,
        "touch_count": 0 if freshness == "FRESH" else 1,
        "status": "RADAR",
        "scoring_status": "VALID",
        "origin_bos": {"origin_valid": True},
        "fvg_confirmation": {"status": "UNTOUCHED"},
        "created_time": datetime.now(timezone.utc).isoformat(),
    }


class _DummyTick:
    def __init__(self, bid: float, ask: float, ts: int = 1794552000):
        self.bid = bid
        self.ask = ask
        self.time = ts


class TestSingleSnapshotDataFlow(unittest.TestCase):
    def setUp(self):
        zs._SCAN_CACHE.update(key=None, ts=0.0, data=None)
        zs._FRAMES_CACHE.update(ts=0.0, data=None)

    def test_01_single_scan_execution(self):
        """Test 1 — Dalam satu siklus perintah scan, scan_xauusd_zones() hanya dipanggil 1 kali."""
        frames = _make_frames()
        tick_dt = datetime(2026, 11, 13, 6, 40, tzinfo=timezone.utc)
        fake_mt5 = types.SimpleNamespace(
            symbol_info_tick=lambda sym: _DummyTick(3899.80, 3900.20, ts=int(tick_dt.timestamp()))
        )
        scan_calls = []
        orig_scan = zs.scan_xauusd_zones

        def counting_scan(*args, **kwargs):
            scan_calls.append(kwargs.get("snapshot", {}).get("snapshot_id") if isinstance(kwargs.get("snapshot"), dict) else None)
            return orig_scan(*args, **kwargs)

        with patch.dict(sys.modules, {"MetaTrader5": fake_mt5}), \
             patch.object(zs, "_load_frames", return_value=frames), \
             patch.object(zs, "scan_xauusd_zones", side_effect=counting_scan):
            snap = create_market_snapshot()
            hasil_scan = zs.scan_xauusd_zones(snapshot=snap)
            dec = evaluate_trade_decision(snapshot=snap, scan_result=hasil_scan)
            report = format_scan_presentation(hasil_scan, dec, detailed=False)

        self.assertEqual(len(scan_calls), 1)
        self.assertIn("XAUUSD PYTHON SMC SCANNER", report)

    def test_02_single_tick_snapshot(self):
        """Test 2 — Satu siklus scan hanya mengambil 1 MT5 tick dan 1 kali load frames."""
        frames = _make_frames()
        tick_calls = []
        frame_calls = []
        tick_dt = datetime(2026, 11, 13, 6, 40, tzinfo=timezone.utc)

        def fake_tick(sym):
            tick_calls.append(sym)
            # Jika dipanggil kedua kali, sengaja kembalikan harga berbeda untuk membuktikan single tick
            if len(tick_calls) == 1:
                return _DummyTick(3899.80, 3900.20, ts=int(tick_dt.timestamp()))
            return _DummyTick(3950.00, 3950.50, ts=int(tick_dt.timestamp()) + 10)

        def fake_load_frames(ttl=0.0):
            frame_calls.append(ttl)
            return frames

        fake_mt5 = types.SimpleNamespace(symbol_info_tick=fake_tick)
        with patch.dict(sys.modules, {"MetaTrader5": fake_mt5}), \
             patch.object(zs, "_load_frames", side_effect=fake_load_frames):
            snap = create_market_snapshot()
            hasil_scan = scan_xauusd_zones(snapshot=snap)
            dec = evaluate_trade_decision(snapshot=snap, scan_result=hasil_scan)

        self.assertEqual(len(tick_calls), 1)
        self.assertEqual(len(frame_calls), 1)
        self.assertEqual(hasil_scan["snapshot_price"], 3900.00)
        self.assertEqual(dec["snapshot_price"], 3900.00)

    def test_03_shared_scan_result(self):
        """Test 3 — evaluate_trade_decision() menggunakan hasil scan yang sama dengan laporan."""
        frames = _make_frames()
        snap = create_market_snapshot(frames=frames, current_price=3900.00, live=False)
        mock_zone = _make_zone("ZONE_SHARED_01", "H1", "SUPPLY", 3915.0, 3920.0, score=82.0, grade="A")
        custom_scan = {
            "symbol": "XAUUSD",
            "timestamp": "2026-11-13 13:40:00",
            "snapshot_id": snap["snapshot_id"],
            "snapshot_time": snap["snapshot_time"],
            "snapshot_price": 3900.00,
            "current_price": 3900.00,
            "bid": 3899.80,
            "ask": 3900.20,
            "mid": 3900.00,
            "spread": 0.40,
            "snapshot_status": "OK",
            "data_ok": True,
            "htf_context": {"H4": "BEARISH", "H1": "BEARISH", "has_conflict": False},
            "active_zones": [mock_zone],
            "valid_zones": [mock_zone],
            "radar_zones": [mock_zone],
            "rejected_zones": [],
            "atr_diagnostics": snap["atr_diagnostics"],
            "snapshot": snap,
        }
        dec = evaluate_trade_decision(snapshot=snap, scan_result=custom_scan)
        report = format_scan_presentation(custom_scan, dec, detailed=False)

        self.assertEqual(dec["zone"]["zone_id"], "ZONE_SHARED_01")
        self.assertEqual(dec["monitored_zone_identity"]["zone_id"], "ZONE_SHARED_01")
        self.assertIn("3915.00", report)

    def test_04_shared_snapshot(self):
        """Test 4 — Scanner dan evaluator menggunakan snapshot_id yang identik."""
        frames = _make_frames()
        tick_dt = datetime(2026, 11, 13, 6, 5)
        fake_mt5 = types.SimpleNamespace(
            symbol_info_tick=lambda sym: _DummyTick(3899.80, 3900.20, ts=int(tick_dt.timestamp()))
        )
        with patch.dict(sys.modules, {"MetaTrader5": fake_mt5}), \
             patch.object(zs, "_load_frames", return_value=frames):
            snap = create_market_snapshot()
            hasil_scan = scan_xauusd_zones(snapshot=snap)
            dec = evaluate_trade_decision(snapshot=snap, scan_result=hasil_scan)

        self.assertTrue(snap["snapshot_id"].startswith("SNAP_"))
        self.assertEqual(hasil_scan["snapshot_id"], snap["snapshot_id"])
        self.assertEqual(dec["snapshot_id"], snap["snapshot_id"])
        self.assertEqual(hasil_scan["radar_priority"]["snapshot_id"], snap["snapshot_id"])
        self.assertEqual(dec["radar_priority"]["snapshot_id"], snap["snapshot_id"])

        diag_txt = format_diagnostic_report(hasil_scan)
        self.assertIn(snap["snapshot_id"], diag_txt)

    def test_05_no_hidden_rescan(self):
        """Test 5 — Evaluator tidak melakukan pemindaian ulang ketika scan_result tersedia."""
        frames = _make_frames()
        snap = create_market_snapshot(frames=frames, current_price=3900.00, live=False)
        hasil_scan = scan_xauusd_zones(snapshot=snap)

        def boom(*args, **kwargs):
            raise AssertionError("scan_xauusd_zones tidak boleh dipanggil ulang saat scan_result sudah tersedia!")

        with patch.object(zs, "scan_xauusd_zones", side_effect=boom):
            dec = evaluate_trade_decision(snapshot=snap, scan_result=hasil_scan)

        self.assertEqual(dec["snapshot_id"], snap["snapshot_id"])
        self.assertIn(dec["status"], ("VALID", "WAIT", "INVALID"))

    def test_06_typeerror_handling_no_fallback_rescan(self):
        """Test 6 — TypeError internal scanner tidak memicu fallback pemindaian ulang tanpa snapshot."""
        frames = _make_frames()
        call_snapshots = []

        def internal_typeerror_scan(frames=None, current_price=None, symbol="XAUUSD", snapshot=None):
            call_snapshots.append(snapshot)
            raise TypeError("internal bug inside scanner")

        with patch.object(zs, "scan_xauusd_zones", side_effect=internal_typeerror_scan), \
             patch.object(zs, "_catat_error") as mock_log:
            dec = evaluate_trade_decision(frames=frames, current_price=3900.00, live=False)

        self.assertEqual(len(call_snapshots), 1)
        self.assertEqual(dec["status"], "WAIT")
        self.assertIn("ERROR", dec["stages"])
        self.assertIn("TypeError", dec["reasons"][0])
        self.assertTrue(mock_log.called)

    def test_07_price_consistency(self):
        """Test 7 — Bid, Ask, Mid, Spread, dan Current Price pada laporan berasal dari snapshot yang sama."""
        frames = _make_frames()
        tick_dt = datetime(2026, 11, 13, 6, 40, tzinfo=timezone.utc)
        fake_mt5 = types.SimpleNamespace(
            symbol_info_tick=lambda sym: _DummyTick(3891.48, 3891.72, ts=int(tick_dt.timestamp()))
        )
        with patch.dict(sys.modules, {"MetaTrader5": fake_mt5}), \
             patch.object(zs, "_load_frames", return_value=frames):
            snap = create_market_snapshot()
            hasil_scan = scan_xauusd_zones(snapshot=snap)
            dec = evaluate_trade_decision(snapshot=snap, scan_result=hasil_scan)
            report = format_scan_presentation(hasil_scan, dec, detailed=False)

        self.assertIn("3891.60", report)
        self.assertIn("Bid: 3891.48", report)
        self.assertIn("Ask: 3891.72", report)
        self.assertIn("Mid: 3891.60", report)
        self.assertIn("Spread: 0.24 pt", report)

    def test_08_distance_consistency(self):
        """Test 8 — Jarak Primary, Secondary, dan Countertrend Zone menggunakan referensi harga snapshot yang sama."""
        z_prim = _make_zone("Z_PRIM", "H1", "SUPPLY", 3910.00, 3915.00, score=84.0, grade="A")
        z_sec = _make_zone("Z_SEC", "H4", "SUPPLY", 3925.00, 3930.00, score=80.0, grade="A")
        z_ctr = _make_zone("Z_CTR", "H1", "DEMAND", 3880.00, 3885.00, score=78.0, grade="A")

        prio = prioritize_radar_zones(
            radar_zones=[z_prim, z_sec, z_ctr],
            current_price=3900.00,
            htf_bias="BEARISH",
            h1_structure="BEARISH",
            atr=5.0,
            bid=3899.80,
            ask=3900.20,
            snapshot_id="SNAP_DIST_TEST",
        )
        # SUPPLY uses Ask (3900.20) -> proximal (low): 3910.00 - 3900.20 = 9.80; 3925.00 - 3900.20 = 24.80
        # DEMAND uses Bid (3899.80) -> proximal (high): 3899.80 - 3885.00 = 14.80
        self.assertAlmostEqual(prio["primary_directional_zone"]["distance"], 9.80, places=2)
        self.assertAlmostEqual(prio["secondary_zone"]["distance"], 24.80, places=2)
        self.assertAlmostEqual(prio["countertrend_zone"]["distance"], 14.80, places=2)

    def test_09_priority_composite_snapshot_lock(self):
        """Test 9 — Priority Composite konsisten dalam snapshot sama dan tidak memakai cache snapshot lama saat snapshot_id baru."""
        z = _make_zone("Z_PRIO", "H1", "SUPPLY", 3915.00, 3920.00, score=78.5, grade="A")
        p1 = prioritize_radar_zones(
            [z], current_price=3900.00, htf_bias="BEARISH", h1_structure="BEARISH",
            atr=5.0, snapshot_id="SNAP_1"
        )
        locked_item = p1["all_prioritized"][0]
        score_1 = locked_item["priority_score"]

        # Snapshot baru dengan snapshot_id berbeda tetapi harga kebetulan sama (3900.00) -> wajib hitung ulang
        p2 = prioritize_radar_zones(
            [locked_item], current_price=3900.00, htf_bias="BULLISH", h1_structure="BULLISH",
            atr=5.0, snapshot_id="SNAP_2"
        )
        score_2 = p2["all_prioritized"][0]["priority_score"]
        self.assertNotEqual(score_1, score_2)

    def test_10_entry_validation_same_zone_as_scan(self):
        """Test 10 — Pending Limit dan Confirmed Entry memantau zona yang sama dengan hasil scan."""
        frames = _make_frames()
        snap = create_market_snapshot(frames=frames, current_price=3900.00, live=False)
        z_target = _make_zone("ZONE_M15+M5_SUPPLY_3918", "H1", "SUPPLY", 3918.0, 3922.0, score=86.0, grade="A+")
        z_backup = _make_zone("ZONE_BACKUP", "H4", "SUPPLY", 3940.0, 3945.0, score=76.0, grade="A")

        scan_res = {
            "symbol": "XAUUSD",
            "timestamp": "2026-11-13 13:40:00",
            "snapshot_id": snap["snapshot_id"],
            "snapshot_time": snap["snapshot_time"],
            "snapshot_price": 3900.00,
            "current_price": 3900.00,
            "bid": 3899.80,
            "ask": 3900.20,
            "mid": 3900.00,
            "spread": 0.40,
            "snapshot_status": "OK",
            "data_ok": True,
            "htf_context": {"H4": "BEARISH", "H1": "BEARISH", "has_conflict": False},
            "active_zones": [z_target, z_backup],
            "valid_zones": [z_target, z_backup],
            "radar_zones": [z_target, z_backup],
            "rejected_zones": [],
            "atr_diagnostics": snap["atr_diagnostics"],
            "snapshot": snap,
        }
        scan_res["radar_priority"] = prioritize_radar_zones(
            radar_zones=[z_target, z_backup],
            current_price=3900.00,
            htf_bias="BEARISH",
            h1_structure="BEARISH",
            atr=5.0,
            frames=frames,
            bid=3899.80,
            ask=3900.20,
            snapshot_id=snap["snapshot_id"],
            snapshot_time=snap["snapshot_time"],
        )
        dec = evaluate_trade_decision(snapshot=snap, scan_result=scan_res)

        self.assertEqual(scan_res["radar_priority"]["primary_directional_zone"]["zone_id"], "ZONE_M15+M5_SUPPLY_3918")
        self.assertEqual(dec["monitored_zone_identity"]["zone_id"], "ZONE_M15+M5_SUPPLY_3918")
        self.assertEqual(dec["best_pending_candidate"]["zone_id"], "ZONE_M15+M5_SUPPLY_3918")

    def test_11_backward_compatibility_standalone_evaluate(self):
        """Test 11 — Pemanggilan lama evaluate_trade_decision() secara mandiri tetap kompatibel."""
        frames = _make_frames()
        # Simulasi pemanggil lama dengan mock scan_xauusd_zones 3-argumen tanpa snapshot
        legacy_zone = _make_zone("ZONE_LEGACY", "H1", "SUPPLY", 3915.0, 3920.0, score=80.0, grade="A")

        def legacy_scan(frames=None, current_price=None, symbol=None):
            return {
                "data_ok": True,
                "current_price": current_price,
                "htf_context": {"H4": "BEARISH", "H1": "BEARISH", "has_conflict": False},
                "active_zones": [legacy_zone],
                "radar_zones": [legacy_zone],
            }

        with patch.object(zs, "scan_xauusd_zones", side_effect=legacy_scan):
            dec = evaluate_trade_decision(frames=frames, current_price=3900.00, spread=0.2, live=False)

        self.assertEqual(dec["zone"]["zone_id"], "ZONE_LEGACY")
        self.assertIn(dec["status"], ("VALID", "WAIT", "INVALID"))

    def test_12_error_handling_data_incomplete(self):
        """Test 12 — Kegagalan / data parsial MT5 ditandai DATA_INCOMPLETE dan tidak dianggap valid."""
        frames = _make_frames()
        # Kasus 1: Tick tidak valid (ask < bid)
        fake_mt5_bad_tick = types.SimpleNamespace(
            symbol_info_tick=lambda sym: _DummyTick(3901.00, 3900.00)
        )
        with patch.dict(sys.modules, {"MetaTrader5": fake_mt5_bad_tick}), \
             patch.object(zs, "_load_frames", return_value=frames):
            snap_bad = create_market_snapshot()
            scan_bad = scan_xauusd_zones(snapshot=snap_bad)
            dec_bad = evaluate_trade_decision(snapshot=snap_bad, scan_result=scan_bad)

        self.assertEqual(snap_bad["snapshot_status"], "DATA_INCOMPLETE")
        self.assertFalse(snap_bad["data_ok"])
        self.assertEqual(scan_bad["snapshot_status"], "DATA_INCOMPLETE")
        self.assertFalse(scan_bad["data_ok"])
        self.assertEqual(dec_bad["status"], "WAIT")
        self.assertEqual(dec_bad.get("snapshot_status"), "DATA_INCOMPLETE")

        # Kasus 2: MT5 offline (_load_frames -> None)
        with patch.object(zs, "_load_frames", return_value=None):
            snap_off = create_market_snapshot()
            scan_off = scan_xauusd_zones(snapshot=snap_off)
            dec_off = evaluate_trade_decision(snapshot=snap_off, scan_result=scan_off)

        self.assertEqual(snap_off["snapshot_status"], "DATA_INCOMPLETE")
        self.assertFalse(scan_off["data_ok"])
        self.assertEqual(dec_off["status"], "WAIT")


if __name__ == "__main__":
    unittest.main()
