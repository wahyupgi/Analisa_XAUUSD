# -*- coding: utf-8 -*-
"""Finalisasi Single Snapshot: jalur fallback format_scan_presentation & cari_golden_confluence."""
import os
import sys
import types
import unittest
from datetime import datetime
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(__file__))

import modules.zone_scanner as zs
import modules.analisis_teknikal as at
from modules.zone_scanner import (
    create_market_snapshot, evaluate_trade_decision, format_scan_presentation, scan_xauusd_zones,
)
from test_single_snapshot_flow import _make_frames, _DummyTick

TICK_TS = int(datetime(2026, 11, 13, 6, 5).timestamp())


def _fake_mt5(counter, bid=3899.80, ask=3900.20):
    def _t(sym):
        counter.append(sym)
        return _DummyTick(bid, ask, ts=TICK_TS)
    return types.SimpleNamespace(symbol_info_tick=_t)


class TestFinalizeSnapshot(unittest.TestCase):
    def setUp(self):
        zs._SCAN_CACHE.update(key=None, ts=0.0, data=None)
        zs._FRAMES_CACHE.update(ts=0.0, data=None)
        self.frames = _make_frames()

    def _make_scan(self):
        ticks = []
        with patch.dict(sys.modules, {"MetaTrader5": _fake_mt5(ticks)}), \
             patch.object(zs, "_load_frames", return_value=self.frames):
            snap = create_market_snapshot()
            scan = scan_xauusd_zones(snapshot=snap)
        self.assertEqual(len(ticks), 1)
        return snap, scan

    def test_01_presentation_without_dec_fetches_nothing_new(self):
        snap, scan = self._make_scan()
        ticks, loads = [], []
        # Setelah scan dibuat, MT5/frames tidak boleh disentuh lagi
        with patch.dict(sys.modules, {"MetaTrader5": _fake_mt5(ticks, 4000.0, 4000.5)}), \
             patch.object(zs, "_load_frames", side_effect=lambda *a, **k: loads.append(1)), \
             patch.object(zs, "scan_xauusd_zones", side_effect=AssertionError("rescan")):
            txt = format_scan_presentation(scan, None)
        self.assertEqual(ticks, [])
        self.assertEqual(loads, [])
        self.assertIn("3900.00", txt)
        self.assertNotIn("4000.", txt)
        self.assertIs(scan["snapshot"], snap)

    def test_02_old_scan_not_evaluated_with_new_snapshot(self):
        snap, scan = self._make_scan()
        other = create_market_snapshot(frames=self.frames, current_price=3950.0, live=False)
        self.assertNotEqual(other["snapshot_id"], snap["snapshot_id"])
        dec = evaluate_trade_decision(snapshot=other, scan_result=scan)
        self.assertEqual(dec["status"], "WAIT")
        self.assertTrue(any("SNAPSHOT_MISMATCH" in r for r in dec["reasons"]))
        self.assertEqual(dec.get("snapshot_status"), "DATA_INCOMPLETE")

    def test_02b_scan_none_with_old_dec_is_reevaluated_from_new_snapshot(self):
        snap, scan = self._make_scan()
        old_dec = evaluate_trade_decision(snapshot=snap, scan_result=scan)
        ticks = []
        with patch.dict(sys.modules, {"MetaTrader5": _fake_mt5(ticks, 3910.0, 3910.4)}), \
             patch.object(zs, "_load_frames", return_value=self.frames):
            txt = format_scan_presentation(None, old_dec)
        self.assertEqual(len(ticks), 1)          # satu snapshot baru, bukan campur
        self.assertIn("3910.20", txt)
        self.assertNotIn(snap["snapshot_id"], txt)

    def test_03_golden_confluence_consistent_sources(self):
        ticks = []
        seen = {}
        orig_scan, orig_eval = zs.scan_xauusd_zones, zs.evaluate_trade_decision

        def spy_scan(*a, **k):
            seen["scan_snap"] = k.get("snapshot")
            return orig_scan(*a, **k)

        def spy_eval(*a, **k):
            seen["eval_snap"] = k.get("snapshot")
            seen["eval_scan"] = k.get("scan_result")
            return orig_eval(*a, **k)

        tick = _DummyTick(3899.80, 3900.20, ts=TICK_TS)
        with patch.dict(sys.modules, {"MetaTrader5": _fake_mt5(ticks, 4100.0, 4100.4)}), \
             patch.object(zs, "scan_xauusd_zones", side_effect=spy_scan), \
             patch.object(zs, "evaluate_trade_decision", side_effect=spy_eval), \
             patch.object(zs, "_load_frames", side_effect=AssertionError("frames refetch")):
            # jalur dengan tick pemanggil: tidak boleh ada fetch tick baru
            at.cari_golden_confluence(self.frames, 1.0, tick=tick)
        self.assertEqual(ticks, [])
        s = seen["scan_snap"]
        self.assertIs(s, seen["eval_snap"])
        self.assertIs(seen["eval_scan"]["snapshot"], s)
        self.assertEqual((s["bid"], s["ask"], s["mid"], s["spread"]), (3899.8, 3900.2, 3900.0, 0.4))
        self.assertEqual(s["snapshot_price"], 3900.0)        # bukan harga 1.0 dari pemanggil
        self.assertEqual(s["data_sources"]["tick"], "caller_supplied")
        self.assertEqual(s["data_sources"]["frames"], "caller_supplied")

    def test_03b_golden_confluence_with_snapshot_no_fetch(self):
        snap = create_market_snapshot(frames=self.frames, tick=_DummyTick(3899.8, 3900.2, TICK_TS))
        ticks = []
        with patch.dict(sys.modules, {"MetaTrader5": _fake_mt5(ticks)}):
            at.cari_golden_confluence(self.frames, 3899.8, snapshot=snap)
        self.assertEqual(ticks, [])

    def test_04_snapshot_id_consistent_scan_and_evaluator(self):
        snap, scan = self._make_scan()
        dec = evaluate_trade_decision(snapshot=snap, scan_result=scan)
        print("SNAPSHOT_ID_TEST:", snap["snapshot_id"])
        self.assertEqual(snap["snapshot_id"], scan["snapshot_id"])
        self.assertEqual(scan["snapshot_id"], dec["snapshot_id"])
        self.assertEqual(scan["radar_priority"]["snapshot_id"], snap["snapshot_id"])
        self.assertEqual(scan["current_price"], scan["snapshot_price"])
        self.assertEqual(dec["snapshot_price"], scan["snapshot_price"])
        self.assertEqual(dec["snapshot_time"], scan["snapshot_time"])

    def test_05_no_rescan_when_scan_result_given(self):
        snap, scan = self._make_scan()
        with patch.object(zs, "scan_xauusd_zones", side_effect=AssertionError("rescan")), \
             patch.object(zs, "_load_frames", side_effect=AssertionError("refetch")), \
             patch.object(zs, "_live_context", side_effect=AssertionError("retick")):
            dec = evaluate_trade_decision(scan_result=scan)    # tanpa snapshot eksplisit
        self.assertEqual(dec["snapshot_id"], snap["snapshot_id"])

    def test_05b_scan_result_without_ohlc_is_data_incomplete_not_refetch(self):
        bare = {"data_ok": True, "snapshot_id": "SNAP_X", "current_price": 3900.0, "snapshot_price": 3900.0}
        with patch.object(zs, "_load_frames", side_effect=AssertionError("refetch")), \
             patch.object(zs, "_live_context", side_effect=AssertionError("retick")):
            dec = evaluate_trade_decision(scan_result=bare)
        self.assertEqual(dec["status"], "WAIT")
        self.assertEqual(dec.get("snapshot_status"), "DATA_INCOMPLETE")

    def test_06_backward_compat_legacy_calls(self):
        scan = scan_xauusd_zones(frames=self.frames, current_price=3900.0)       # tanpa snapshot
        self.assertTrue(scan["snapshot_id"].startswith("SNAP_"))
        self.assertIs(scan["snapshot"]["frames"], self.frames)
        dec = evaluate_trade_decision(frames=self.frames, current_price=3900.0, live=False)
        self.assertIn(dec["status"], ("VALID", "WAIT", "INVALID"))
        txt = format_scan_presentation(scan, None)      # evaluasi dari snapshot sintetis scan, tanpa fetch
        self.assertIn("XAUUSD PYTHON SMC SCANNER", txt)

    def test_07_data_incomplete_never_valid(self):
        ticks = []
        with patch.dict(sys.modules, {"MetaTrader5": _fake_mt5(ticks, 3901.0, 3900.0)}), \
             patch.object(zs, "_load_frames", return_value=self.frames):
            snap = create_market_snapshot()
            scan = scan_xauusd_zones(snapshot=snap)
            dec = evaluate_trade_decision(snapshot=snap, scan_result=scan)
            gc = at.cari_golden_confluence(self.frames, 3900.0)
        self.assertEqual(snap["snapshot_status"], "DATA_INCOMPLETE")
        self.assertEqual(dec["status"], "WAIT")
        self.assertIsNone(dec["plan"])
        self.assertNotIn("VALID", [dec["status"]])
        self.assertNotIn("BUY VALID", gc)
        self.assertNotIn("SELL VALID", gc)


if __name__ == "__main__":
    unittest.main()

