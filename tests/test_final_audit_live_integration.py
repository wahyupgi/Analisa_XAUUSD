# -*- coding: utf-8 -*-
"""
Test Suite: Final Audit Single Snapshot & MT5 Live Integration (Mocked MT5).
Menguji seluruh skenario Section 5:
- Tick valid dan stale
- Bid/Ask/Mid tidak valid
- Candle berjalan vs candle tertutup
- MT5 disconnected
- Cache fresh dan stale
- Snapshot ID mismatch
- Tidak ada fetch ulang
- Pending Limit menggunakan sisi harga pasar (Bid/Ask) yang tepat
- DATA_INCOMPLETE selalu menghasilkan WAIT (tidak pernah READY/VALID)
"""

import os
import sys
import time
import types
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(__file__))

import modules.zone_scanner as zs
from modules.zone_scanner import (
    SCAN_CACHE_TTL,
    Zone,
    create_market_snapshot,
    evaluate_trade_decision,
    evaluate_zone_hybrid_eligibility,
    format_scan_presentation,
    plan_pending_order,
    scan_xauusd_zones,
)
from test_single_snapshot_flow import _DummyTick, _make_frames, _make_ohlcv


class TestFinalAuditLiveIntegration(unittest.TestCase):
    def setUp(self):
        zs._SCAN_CACHE.update(key=None, ts=0.0, data=None)
        zs._FRAMES_CACHE.update(ts=0.0, data=None)
        self.base_dt = datetime(2026, 11, 13, 6, 0)
        self.frames = {
            "H4": _make_ohlcv(80, base=3930.0, trend=-0.4, atr_val=8.0, freq="4h", end_dt=datetime(2026, 11, 13, 2, 0)),
            "H1": _make_ohlcv(80, base=3920.0, trend=-0.3, atr_val=5.0, freq="1h", end_dt=datetime(2026, 11, 13, 5, 0)),
            "M30": _make_ohlcv(80, base=3910.0, trend=-0.2, atr_val=4.0, freq="30min", end_dt=datetime(2026, 11, 13, 5, 30)),
            "M15": _make_ohlcv(80, base=3905.0, trend=-0.1, atr_val=3.0, freq="15min", end_dt=datetime(2026, 11, 13, 5, 45)),
            "M5": _make_ohlcv(80, base=3900.0, trend=-0.05, atr_val=2.0, freq="5min", end_dt=datetime(2026, 11, 13, 6, 0)),
        }

    def test_01_valid_tick_and_stale_tick(self):
        """Tick valid lolos check data; tick stale (> 20 menit setelah M5 atau lebih lama dari candle) -> WAIT."""
        # Valid tick: 5 menit setelah candle terakhir M5 (06:00 -> 06:05 UTC)
        valid_ts = int(datetime(2026, 11, 13, 6, 5, tzinfo=timezone.utc).timestamp())
        fake_ok = types.SimpleNamespace(symbol_info_tick=lambda s: _DummyTick(3899.80, 3900.20, ts=valid_ts))
        with patch.dict(sys.modules, {"MetaTrader5": fake_ok}), \
             patch.object(zs, "_load_frames", return_value=self.frames):
            snap_ok = create_market_snapshot()
            scan_ok = scan_xauusd_zones(snapshot=snap_ok)
            dec_ok = evaluate_trade_decision(snapshot=snap_ok, scan_result=scan_ok)
        self.assertTrue(dec_ok["stages"]["DATA"]["ok"])

        # Stale tick 1: tick 45 menit setelah candle M5 terakhir (market tutup / feed tertinggal)
        stale_old_candle_ts = int(datetime(2026, 11, 13, 6, 45, tzinfo=timezone.utc).timestamp())
        fake_stale1 = types.SimpleNamespace(symbol_info_tick=lambda s: _DummyTick(3899.80, 3900.20, ts=stale_old_candle_ts))
        with patch.dict(sys.modules, {"MetaTrader5": fake_stale1}), \
             patch.object(zs, "_load_frames", return_value=self.frames):
            snap_s1 = create_market_snapshot()
            scan_s1 = scan_xauusd_zones(snapshot=snap_s1)
            dec_s1 = evaluate_trade_decision(snapshot=snap_s1, scan_result=scan_s1)
        self.assertEqual(dec_s1["status"], "WAIT")
        self.assertFalse(dec_s1["stages"]["DATA"]["ok"])

        # Stale tick 2: tick lebih lama dari candle terakhir (05:45 vs candle 06:00)
        stale_tick_ts = int(datetime(2026, 11, 13, 5, 45, tzinfo=timezone.utc).timestamp())
        fake_stale2 = types.SimpleNamespace(symbol_info_tick=lambda s: _DummyTick(3899.80, 3900.20, ts=stale_tick_ts))
        with patch.dict(sys.modules, {"MetaTrader5": fake_stale2}), \
             patch.object(zs, "_load_frames", return_value=self.frames):
            snap_s2 = create_market_snapshot()
            scan_s2 = scan_xauusd_zones(snapshot=snap_s2)
            dec_s2 = evaluate_trade_decision(snapshot=snap_s2, scan_result=scan_s2)
        self.assertEqual(dec_s2["status"], "WAIT")
        self.assertIn("stale tick", " ".join(dec_s2["reasons"]).lower())

    def test_02_invalid_bid_ask_mid(self):
        """Bid/Ask/Mid tidak valid (ask < bid, bid=0, NaN) -> DATA_INCOMPLETE dan WAIT."""
        bad_cases = [
            (3901.00, 3900.00),          # ask < bid
            (0.0, 3900.00),              # bid <= 0
            (float("nan"), 3900.00),     # NaN
        ]
        for b, a in bad_cases:
            fake_bad = types.SimpleNamespace(symbol_info_tick=lambda s, bb=b, aa=a: _DummyTick(bb, aa))
            with patch.dict(sys.modules, {"MetaTrader5": fake_bad}), \
                 patch.object(zs, "_load_frames", return_value=self.frames):
                snap = create_market_snapshot()
                scan = scan_xauusd_zones(snapshot=snap)
                dec = evaluate_trade_decision(snapshot=snap, scan_result=scan)
            self.assertEqual(snap["snapshot_status"], "DATA_INCOMPLETE")
            self.assertFalse(scan["data_ok"])
            self.assertEqual(dec["status"], "WAIT")
            self.assertIsNone(dec["plan"])

    def test_03_running_vs_closed_candle(self):
        """Candle berjalan (diff < 300s atau is_closed=False) -> WAIT; candle tertutup -> DATA ok."""
        # Candle M5 jam 06:00, server_time jam 06:02 -> masih berjalan (diff=120s < 300s)
        running_ts = int(datetime(2026, 11, 13, 6, 2, tzinfo=timezone.utc).timestamp())
        fake_run = types.SimpleNamespace(symbol_info_tick=lambda s: _DummyTick(3899.80, 3900.20, ts=running_ts))
        with patch.dict(sys.modules, {"MetaTrader5": fake_run}), \
             patch.object(zs, "_load_frames", return_value=self.frames):
            snap_r = create_market_snapshot()
            scan_r = scan_xauusd_zones(snapshot=snap_r)
            dec_r = evaluate_trade_decision(snapshot=snap_r, scan_result=scan_r)
        self.assertEqual(dec_r["status"], "WAIT")
        self.assertIn("belum selesai", " ".join(dec_r["reasons"]).lower())

        # Candle tertutup tetapi ditandai is_closed=False -> juga ditolak
        closed_ts = int(datetime(2026, 11, 13, 6, 5, tzinfo=timezone.utc).timestamp())
        frames_unclosed = dict(self.frames)
        frames_unclosed["M5"] = self.frames["M5"].copy()
        frames_unclosed["M5"].attrs["is_closed"] = False
        fake_ok = types.SimpleNamespace(symbol_info_tick=lambda s: _DummyTick(3899.80, 3900.20, ts=closed_ts))
        with patch.dict(sys.modules, {"MetaTrader5": fake_ok}), \
             patch.object(zs, "_load_frames", return_value=frames_unclosed):
            snap_u = create_market_snapshot()
            scan_u = scan_xauusd_zones(snapshot=snap_u)
            dec_u = evaluate_trade_decision(snapshot=snap_u, scan_result=scan_u)
        self.assertEqual(dec_u["status"], "WAIT")
        self.assertIn("belum selesai", " ".join(dec_u["reasons"]).lower())

    def test_04_mt5_disconnected(self):
        """MT5 disconnected (_load_frames=None atau symbol_info_tick=None) -> DATA_INCOMPLETE & WAIT."""
        with patch.object(zs, "_load_frames", return_value=None):
            snap = create_market_snapshot()
            scan = scan_xauusd_zones(snapshot=snap)
            dec = evaluate_trade_decision(snapshot=snap, scan_result=scan)
        self.assertEqual(snap["snapshot_status"], "DATA_INCOMPLETE")
        self.assertFalse(scan["data_ok"])
        self.assertEqual(dec["status"], "WAIT")

    def test_05_cache_fresh_and_stale(self):
        """Cache jalur lama: fresh ditandai is_cached=True & data_source_mode='CACHE'; stale -> WAIT."""
        # Call 1 (tanpa snapshot): populate cache
        s1 = scan_xauusd_zones(frames=self.frames, current_price=3900.00)
        self.assertFalse(s1["is_cached"])
        self.assertEqual(s1["data_source_mode"], "LIVE_SCAN")

        # Call 2 (tanpa snapshot, dalam TTL): hit cache dengan metadata jelas
        s2 = scan_xauusd_zones(frames=self.frames, current_price=3900.00)
        self.assertTrue(s2["is_cached"])
        self.assertEqual(s2["data_source_mode"], "CACHE")
        self.assertIsNotNone(s2["cache_timestamp"])

        # Simulasikan data cache stale melewati SCAN_CACHE_TTL -> evaluator wajib WAIT
        s2["cache_age_seconds"] = SCAN_CACHE_TTL + 10.0
        dec_stale = evaluate_trade_decision(frames=self.frames, current_price=3900.00, live=False, scan_result=s2)
        self.assertEqual(dec_stale["status"], "WAIT")
        self.assertIn("stale cache", " ".join(dec_stale["reasons"]).lower())

    def test_06_snapshot_id_mismatch(self):
        """Snapshot ID berbeda antara snapshot dan scan_result -> DATA_INCOMPLETE & WAIT."""
        snap1 = create_market_snapshot(frames=self.frames, current_price=3900.0, live=False)
        snap2 = create_market_snapshot(frames=self.frames, current_price=3900.0, live=False)
        scan1 = scan_xauusd_zones(snapshot=snap1)
        dec = evaluate_trade_decision(snapshot=snap2, scan_result=scan1)
        self.assertEqual(dec["status"], "WAIT")
        self.assertEqual(dec.get("snapshot_status"), "DATA_INCOMPLETE")
        self.assertTrue(any("SNAPSHOT_MISMATCH" in r for r in dec["reasons"]))

    def test_07_no_refetch_during_full_cycle(self):
        """Satu siklus penuh hanya 1x tick, 1x load frames, 1x scan."""
        tick_calls, frame_calls = [], []
        ts = int(datetime(2026, 11, 13, 6, 5, tzinfo=timezone.utc).timestamp())

        def fake_tick(sym):
            tick_calls.append(sym)
            return _DummyTick(3899.80, 3900.20, ts=ts)

        def fake_frames(ttl=0.0):
            frame_calls.append(ttl)
            return self.frames

        fake_mt5 = types.SimpleNamespace(symbol_info_tick=fake_tick)
        with patch.dict(sys.modules, {"MetaTrader5": fake_mt5}), \
             patch.object(zs, "_load_frames", side_effect=fake_frames):
            snap = create_market_snapshot()
            scan = scan_xauusd_zones(snapshot=snap)
            dec = evaluate_trade_decision(snapshot=snap, scan_result=scan)
            report = format_scan_presentation(scan, dec, detailed=False)

        self.assertEqual(len(tick_calls), 1)
        self.assertEqual(len(frame_calls), 1)
        self.assertEqual(snap["snapshot_id"], scan["snapshot_id"])
        self.assertEqual(scan["snapshot_id"], dec["snapshot_id"])
        self.assertIn("Bid: 3899.80 | Ask: 3900.20 | Mid: 3900.00 | Spread: 0.40 pt", report)
        print(
            f"\nSAMPLE_SNAPSHOT_CYCLE: id={snap['snapshot_id']} bid={snap['bid']} "
            f"ask={snap['ask']} mid={snap['mid']} spread={snap['spread']} ts={snap['snapshot_time']}"
        )

    def test_08_pending_limit_uses_bid_ask_side(self):
        """BUY LIMIT valid hanya jika entry < Ask; SELL LIMIT valid hanya jika entry > Bid."""
        z_demand = Zone(
            zone_id="Z_DEM", symbol="XAUUSD", timeframe="H1", zone_type="DEMAND",
            price_low=3885.0, price_high=3890.0, midpoint=3887.5, score=82.0, grade="A",
            origin_bos={"origin_valid": True}, fvg_confirmation={"status": "UNTOUCHED"},
            displacement_strength=2.0, freshness="FRESH", touch_count=0,
            liquidity_sweep_status="CONFIRMED", invalidation_price=3885.0,
            sl_recommended=3883.0, tp1_recommended=3910.0, tp2_recommended=3920.0, is_valid=True,
            rejection_reason="",
        )
        # Ask = 3900.20 > entry 3890.00 -> BUY LIMIT valid
        p_buy_ok = plan_pending_order(z_demand, current_price=3900.0, htf_bias="BULLISH",
                                      m30_status="CONFIRMED", bid=3899.80, ask=3900.20)
        self.assertTrue(p_buy_ok["valid"])

        # Ask = 3889.50 <= entry 3890.00 -> tidak bisa pasang BUY LIMIT (harus WAIT / Confirmed)
        p_buy_bad = plan_pending_order(z_demand, current_price=3889.3, htf_bias="BULLISH",
                                       m30_status="CONFIRMED", bid=3889.10, ask=3889.50)
        self.assertFalse(p_buy_bad["valid"])
        self.assertEqual(p_buy_bad["status"], "WAIT")

        z_supply = Zone(
            zone_id="Z_SUP", symbol="XAUUSD", timeframe="H1", zone_type="SUPPLY",
            price_low=3910.0, price_high=3915.0, midpoint=3912.5, score=82.0, grade="A",
            origin_bos={"origin_valid": True}, fvg_confirmation={"status": "UNTOUCHED"},
            displacement_strength=2.0, freshness="FRESH", touch_count=0,
            liquidity_sweep_status="CONFIRMED", invalidation_price=3915.0,
            sl_recommended=3917.0, tp1_recommended=3890.0, tp2_recommended=3880.0, is_valid=True,
            rejection_reason="",
        )
        # Bid = 3899.80 < entry 3910.00 -> SELL LIMIT valid
        p_sell_ok = plan_pending_order(z_supply, current_price=3900.0, htf_bias="BEARISH",
                                       m30_status="CONFIRMED", bid=3899.80, ask=3900.20)
        self.assertTrue(p_sell_ok["valid"])

        # Bid = 3910.50 >= entry 3910.00 -> tidak bisa pasang SELL LIMIT
        p_sell_bad = plan_pending_order(z_supply, current_price=3910.7, htf_bias="BEARISH",
                                        m30_status="CONFIRMED", bid=3910.50, ask=3910.90)
        self.assertFalse(p_sell_bad["valid"])
        self.assertEqual(p_sell_bad["status"], "WAIT")

    def test_09_data_incomplete_never_ready_or_valid(self):
        """DATA_INCOMPLETE selalu menghasilkan WAIT dan tidak pernah READY atau VALID."""
        fake_bad = types.SimpleNamespace(symbol_info_tick=lambda s: _DummyTick(3901.0, 3900.0))
        with patch.dict(sys.modules, {"MetaTrader5": fake_bad}), \
             patch.object(zs, "_load_frames", return_value=self.frames):
            snap = create_market_snapshot()
            scan = scan_xauusd_zones(snapshot=snap)
            dec = evaluate_trade_decision(snapshot=snap, scan_result=scan)
        self.assertEqual(scan["snapshot_status"], "DATA_INCOMPLETE")
        self.assertEqual(dec["status"], "WAIT")
        self.assertIsNone(dec.get("best_pending_candidate"))
        self.assertIsNone(dec.get("best_confirmed_candidate"))


if __name__ == "__main__":
    unittest.main()
