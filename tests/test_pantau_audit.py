# -*- coding: utf-8 -*-
"""Regression tests for Mode Pantau audit & fixes (Priorities 1-8, mocked MT5)."""
import inspect
import math
import unittest
from types import SimpleNamespace
import pandas as pd

import modules.pantau as pantau


def _make_tick(bid=3899.43, ask=3899.63, t=1763000000):
    return SimpleNamespace(bid=bid, ask=ask, time=t)


def _make_pos(
    ticket=1001,
    symbol=pantau.SYMBOL,
    is_buy=False,
    volume=0.01,
    price_open=3904.53,
    profit=4.90,
    sl=0.0,
    tp=0.0,
    t=1762999000,
):
    return SimpleNamespace(
        ticket=ticket,
        symbol=symbol,
        type=0 if is_buy else 1,
        volume=volume,
        price_open=price_open,
        profit=profit,
        sl=sl,
        tp=tp,
        time=t,
        time_update=t,
    )


def _make_acc(currency="USD", balance=1000.0, equity=1004.90):
    return SimpleNamespace(currency=currency, balance=balance, equity=equity)


def _make_sym(contract=100.0, tick_size=0.01, tick_value=0.0):
    return SimpleNamespace(
        trade_contract_size=contract,
        trade_tick_size=tick_size,
        trade_tick_value=tick_value,
    )


def _make_df_m5(close_price=3899.50, ts="2026-11-13 09:00:00"):
    idx = pd.to_datetime([ts])
    return pd.DataFrame(
        {"Open": [close_price], "High": [close_price + 0.5], "Low": [close_price - 0.5], "Close": [close_price], "Volume": [100]},
        index=idx,
    )


class TestPantauAudit(unittest.TestCase):
    def test_01_sell_001_lot_normal_usd_and_idr(self):
        tick = _make_tick(bid=3899.43, ask=3899.63)
        pos = _make_pos(is_buy=False, volume=0.01, price_open=3904.53, profit=4.90)
        res_usd = pantau.audit_position_profit(
            pos, tick, acc_info=_make_acc("USD"), sym_info=_make_sym(100.0), calc_profit_fn=lambda *a: 4.90
        )
        self.assertTrue(res_usd["valid"])
        self.assertEqual(res_usd["status"], "VALID")
        self.assertEqual(res_usd["display"], "+4.90 USD")

        # Akun IDR: +83,300.00 IDR (~$4.90 * 17,000) dengan order_calc_profit broker IDR
        pos_idr = _make_pos(is_buy=False, volume=0.01, price_open=3904.53, profit=83300.0)
        res_idr = pantau.audit_position_profit(
            pos_idr, tick, acc_info=_make_acc("IDR"), sym_info=_make_sym(100.0), calc_profit_fn=lambda *a: 83300.0
        )
        self.assertTrue(res_idr["valid"])
        self.assertEqual(res_idr["display"], "+83,300.00 IDR")
        self.assertNotIn("$", res_idr["display"])

    def test_02_buy_001_lot_normal(self):
        tick = _make_tick(bid=3905.10, ask=3905.30)
        pos = _make_pos(is_buy=True, volume=0.01, price_open=3900.00, profit=5.10)
        res = pantau.audit_position_profit(
            pos, tick, acc_info=_make_acc("USD"), sym_info=_make_sym(100.0), calc_profit_fn=lambda *a: 5.10
        )
        self.assertTrue(res["valid"])
        self.assertAlmostEqual(res["points"], 5.10, places=2)
        self.assertEqual(res["display"], "+5.10 USD")

    def test_03_multi_lot_scaling(self):
        tick = _make_tick(bid=3902.00, ask=3902.20)
        for lot, exp_prof in ((0.05, 10.0), (0.10, 20.0), (1.00, 200.0)):
            pos = _make_pos(is_buy=True, volume=lot, price_open=3900.00, profit=exp_prof)
            res = pantau.audit_position_profit(
                pos, tick, acc_info=_make_acc("USD"), sym_info=_make_sym(100.0), calc_profit_fn=lambda *a, ep=exp_prof: ep
            )
            self.assertTrue(res["valid"])
            self.assertAlmostEqual(res["profit"], exp_prof, places=2)

    def test_04_profit_changes_with_price(self):
        pos1 = _make_pos(is_buy=True, volume=0.02, price_open=3900.00, profit=4.00)
        r1 = pantau.audit_position_profit(
            pos1, _make_tick(3902.00, 3902.20), acc_info=_make_acc("USD"), sym_info=_make_sym(100.0), calc_profit_fn=lambda *a: 4.00
        )
        pos2 = _make_pos(is_buy=True, volume=0.02, price_open=3900.00, profit=10.00)
        r2 = pantau.audit_position_profit(
            pos2, _make_tick(3905.00, 3905.20), acc_info=_make_acc("USD"), sym_info=_make_sym(100.0), calc_profit_fn=lambda *a: 10.00
        )
        self.assertTrue(r1["valid"] and r2["valid"])
        self.assertGreater(r2["profit"], r1["profit"])

    def test_05_negative_and_positive_profit(self):
        tick = _make_tick(bid=3895.00, ask=3895.20)
        pos_loss = _make_pos(is_buy=True, volume=0.01, price_open=3900.00, profit=-5.00)
        r_loss = pantau.audit_position_profit(
            pos_loss, tick, acc_info=_make_acc("USD"), sym_info=_make_sym(100.0), calc_profit_fn=lambda *a: -5.00
        )
        self.assertTrue(r_loss["valid"])
        self.assertEqual(r_loss["display"], "-5.00 USD")

        pos_win = _make_pos(is_buy=False, volume=0.01, price_open=3900.00, profit=4.80)
        r_win = pantau.audit_position_profit(
            pos_win, tick, acc_info=_make_acc("USD"), sym_info=_make_sym(100.0), calc_profit_fn=lambda *a: 4.80
        )
        self.assertTrue(r_win["valid"])
        self.assertEqual(r_win["display"], "+4.80 USD")

    def test_06_suspect_profit_anomaly_flagged(self):
        # Contoh anomali nyata: SELL 0.01 lot @ 3904.53, Bid 3899.43, Ask 3899.63, profit dilaporkan $87,193.10 (USD)
        tick = _make_tick(bid=3899.43, ask=3899.63)
        pos_bad = _make_pos(is_buy=False, volume=0.01, price_open=3904.53, profit=87193.10)
        res = pantau.audit_position_profit(
            pos_bad, tick, acc_info=_make_acc("USD"), sym_info=_make_sym(100.0), calc_profit_fn=lambda *a: 4.90
        )
        self.assertFalse(res["valid"])
        self.assertEqual(res["status"], "PROFIT_DATA_SUSPECT")
        self.assertEqual(res["display"], "PROFIT_DATA_SUSPECT")

        # Portfolio summary juga wajib menampilkan PROFIT_DATA_SUSPECT
        summary, _ = pantau.format_portfolio_profit(
            [pos_bad], tick, acc_info=_make_acc("USD"), sym_info=_make_sym(100.0), calc_profit_fn=lambda *a: 4.90
        )
        self.assertEqual(summary, "PROFIT_DATA_SUSPECT")

        # NaN profit & sign mismatch juga suspect
        pos_nan = _make_pos(profit=float("nan"))
        self.assertEqual(pantau.audit_position_profit(pos_nan, tick, acc_info=_make_acc("USD"))["status"], "PROFIT_DATA_SUSPECT")
        pos_sign = _make_pos(is_buy=False, volume=0.10, price_open=3904.53, profit=-49.0)
        self.assertEqual(
            pantau.audit_position_profit(pos_sign, tick, acc_info=_make_acc("USD"), calc_profit_fn=lambda *a: 49.0)["status"],
            "PROFIT_DATA_SUSPECT",
        )
        pos_sym = _make_pos(symbol="EURUSD", profit=4.90)
        self.assertEqual(
            pantau.audit_position_profit(pos_sym, tick, acc_info=_make_acc("USD"))["reason"],
            "SYMBOL_MISMATCH",
        )

    def test_07_all_radar_qualified_monitored_with_stable_id_not_ai(self):
        scan_result = {
            "radar_priority": {
                "radar_qualified": [
                    {"zone_id": "M15_SUP_3920", "timeframe": "M15", "zone_type": "SUPPLY", "price_low": 3920.0, "price_high": 3924.0, "invalidation": 3926.0, "risk_tier": "HIGH_RISK", "grade": "B", "score": 60.0, "status": "RADAR", "radar_eligibility": "QUALIFIED"},
                    {"zone_id": "H1_DEM_3880", "timeframe": "H1", "zone_type": "DEMAND", "price_low": 3880.0, "price_high": 3885.0, "invalidation": 3877.0, "risk_tier": "MEDIUM_RISK", "grade": "A", "score": 78.0, "status": "RADAR", "radar_eligibility": "QUALIFIED"},
                    {"zone_id": "H4_DEM_3850", "timeframe": "H4", "zone_type": "DEMAND", "price_low": 3850.0, "price_high": 3856.0, "invalidation": 3845.0, "risk_tier": "LOW_RISK", "grade": "A", "score": 85.0, "status": "RADAR", "radar_eligibility": "QUALIFIED"},
                    {"zone_id": "M30_SUP_3940", "timeframe": "M30", "zone_type": "SUPPLY", "price_low": 3940.0, "price_high": 3944.0, "invalidation": 3947.0, "risk_tier": "", "grade": "B", "score": 58.0, "status": "RADAR", "radar_eligibility": "QUALIFIED"},
                ]
            }
        }
        wl = pantau.collect_radar_watchlist(scan_result)
        ids = [z["zone_id"] for z in wl]
        self.assertEqual(ids, ["M15_SUP_3920", "H1_DEM_3880", "H4_DEM_3850", "M30_SUP_3940"])
        self.assertEqual(wl[3]["risk_tier"], "UNCLASSIFIED")

    def test_08_to_12_zone_transitions_no_dup_no_false_startup(self):
        zones = [
            {"zone_id": "H1_SUP_3910", "timeframe": "H1", "zone_type": "SUPPLY", "price_low": 3910.0, "price_high": 3914.0, "invalidation": 3916.0, "status": "RADAR"},
            {"zone_id": "H1_DEM_3880", "timeframe": "H1", "zone_type": "DEMAND", "price_low": 3880.0, "price_high": 3884.0, "invalidation": 3878.0, "status": "RADAR"},
        ]
        mon = pantau.ZoneMonitorState(symbol="XAUUSD")
        # 12. Startup baseline tidak memicu alert palsu
        snap0 = {"bid": 3899.9, "ask": 3900.1, "mid": 3900.0, "server_time": 1763000000}
        mon.initialize_baseline(zones, snap0, close_m5=3900.0, atr_m5=1.0)
        self.assertEqual(mon.states["H1_SUP_3910"], "RADAR")
        self.assertEqual(mon.states["H1_DEM_3880"], "RADAR")

        # 11. Status sama tidak memicu notifikasi duplikat
        self.assertEqual(mon.update(zones, snap0, close_m5=3900.0, atr_m5=1.0), [])

        # 08. RADAR -> APPROACHING -> ZONE_TOUCHED
        snap_app = {"bid": 3908.4, "ask": 3908.6, "mid": 3908.5, "server_time": 1763000060}
        ev_app = mon.update(zones, snap_app, close_m5=3908.5, atr_m5=1.0)
        self.assertEqual(len(ev_app), 1)
        self.assertEqual((ev_app[0]["prev_status"], ev_app[0]["new_status"]), ("RADAR", "APPROACHING"))
        self.assertIn("WIB", ev_app[0]["timestamp_wib"])
        self.assertEqual(ev_app[0]["zone_id"], "H1_SUP_3910")

        # Tidak duplikat saat masih APPROACHING
        self.assertEqual(mon.update(zones, snap_app, close_m5=3908.5, atr_m5=1.0), [])

        snap_touch = {"bid": 3911.0, "ask": 3911.2, "mid": 3911.1, "server_time": 1763000120}
        ev_touch = mon.update(zones, snap_touch, close_m5=3911.1, atr_m5=1.0)
        self.assertEqual(len(ev_touch), 1)
        self.assertEqual((ev_touch[0]["prev_status"], ev_touch[0]["new_status"]), ("APPROACHING", "ZONE_TOUCHED"))

        # 09. ZONE_TOUCHED -> MITIGATED (memantul meninggalkan Supply tanpa invalidasi)
        snap_mit = {"bid": 3906.0, "ask": 3906.2, "mid": 3906.1, "server_time": 1763000180}
        ev_mit = mon.update(zones, snap_mit, close_m5=3906.1, atr_m5=1.0)
        self.assertEqual(len(ev_mit), 1)
        self.assertEqual((ev_mit[0]["prev_status"], ev_mit[0]["new_status"]), ("ZONE_TOUCHED", "MITIGATED"))

        # 10. ZONE_TOUCHED -> INVALIDATED & RADAR -> EXPIRED
        zones_inv = [
            {"zone_id": "H1_SUP_3910", "timeframe": "H1", "zone_type": "SUPPLY", "price_low": 3910.0, "price_high": 3914.0, "invalidation": 3916.0, "status": "RADAR"},
            {"zone_id": "H1_DEM_3880", "timeframe": "H1", "zone_type": "DEMAND", "price_low": 3880.0, "price_high": 3884.0, "invalidation": 3878.0, "status": "EXPIRED", "expired": True},
        ]
        ev_end = mon.update(zones_inv, {"bid": 3917.0, "ask": 3917.2, "mid": 3917.1, "server_time": 1763000240}, close_m5=3917.1, atr_m5=1.0)
        pairs = {(e["zone_id"], e["new_status"]) for e in ev_end}
        self.assertIn(("H1_SUP_3910", "INVALIDATED"), pairs)
        self.assertIn(("H1_DEM_3880", "EXPIRED"), pairs)

    def test_13_pending_ready_and_confirmed_wait_independent(self):
        eng = pantau.PantauCycleEngine()
        df1 = _make_df_m5(3900.0, "2026-11-13 09:00:00")
        res = eng.step(
            _make_tick(3899.9, 3900.1),
            frames={"M5": df1},
            df_m5=df1,
            snapshot_fn=lambda **kw: {"data_status": "COMPLETE", "bid": 3899.9, "ask": 3900.1, "mid": 3900.0, "snapshot_price": 3900.0, "server_time": 1763000000},
            scan_fn=lambda **kw: {"radar_priority": {"radar_qualified": []}},
            eval_fn=lambda **kw: {"final_decision": "WAIT", "confirmed_status": "WAIT", "pending_plan": {"status": "READY"}},
        )
        self.assertEqual(res["pending_status"], "READY")
        self.assertEqual(res["confirmed_status"], "WAIT")

    def test_14_read_only_and_structural_sl_unavailable(self):
        src = inspect.getsource(pantau)
        self.assertNotIn("order_send", src)
        self.assertNotIn("positions_close", src)
        # Tanpa invalidasi struktur yang cocok, kembalikan None -> STRUCTURAL_SL_UNAVAILABLE
        self.assertIsNone(pantau.resolve_structural_sl_recommendation("BUY", 3900.0, radar_zones=[]))
        # Jika ada zona Demand valid, rekomendasikan invalidasi strukturnya
        sl = pantau.resolve_structural_sl_recommendation(
            "BUY", 3901.0, radar_zones=[{"zone_type": "DEMAND", "price_low": 3899.0, "price_high": 3902.0, "invalidation": 3896.5}]
        )
        self.assertEqual(sl, 3896.5)

    def test_15_disconnect_data_incomplete_and_single_snapshot_no_hidden_rescan(self):
        eng = pantau.PantauCycleEngine()
        df1 = _make_df_m5(3900.0, "2026-11-13 09:00:00")

        # Disconnect -> DATA_INCOMPLETE / WAIT
        disc = eng.step(_make_tick(), df_m5=df1, mt5_connected=False)
        self.assertEqual(disc["data_status"], "DATA_INCOMPLETE")
        self.assertEqual(disc["final_decision"], "WAIT")

        # Initial scan = 1x
        scan_calls = [0]
        def _fake_scan(**kw):
            scan_calls[0] += 1
            return {"radar_priority": {"radar_qualified": []}}

        eng.step(
            _make_tick(3899.9, 3900.1),
            frames={"M5": df1},
            df_m5=df1,
            snapshot_fn=lambda **kw: {"data_status": "COMPLETE", "bid": 3899.9, "ask": 3900.1, "mid": 3900.0, "snapshot_price": 3900.0, "server_time": 1763000000},
            scan_fn=_fake_scan,
            eval_fn=lambda **kw: {"final_decision": "WAIT", "confirmed_status": "WAIT", "pending_plan": {"status": "WAIT"}},
        )
        self.assertEqual(scan_calls[0], 1)

        # Price-only tick pada candle M5 yang sama -> TIDAK memicu rescan!
        eng.step(_make_tick(3900.5, 3900.7), frames={"M5": df1}, df_m5=df1, scan_fn=_fake_scan)
        self.assertEqual(scan_calls[0], 1)

        # Candle M5 baru -> boleh rescan 1x lagi
        df2 = _make_df_m5(3901.0, "2026-11-13 09:05:00")
        eng.step(
            _make_tick(3900.9, 3901.1),
            frames={"M5": df2},
            df_m5=df2,
            snapshot_fn=lambda **kw: {"data_status": "COMPLETE", "bid": 3900.9, "ask": 3901.1, "mid": 3901.0, "snapshot_price": 3901.0, "server_time": 1763000300},
            scan_fn=_fake_scan,
            eval_fn=lambda **kw: {"final_decision": "WAIT", "confirmed_status": "WAIT", "pending_plan": {"status": "WAIT"}},
        )
        self.assertEqual(scan_calls[0], 2)

    def test_16_radar_qualified_5_transferred_with_snapshot_status_ok_and_reconciliation(self):
        # Memastikan format snapshot asli (`snapshot_status="OK"`, `data_ok=True`) meneruskan seluruh 5 zona
        eng = pantau.PantauCycleEngine(symbol=pantau.SYMBOL)
        df1 = _make_df_m5(3900.0, "2026-11-13 09:00:00")
        zones_5 = [
            {"zone_id": f"Z{i}", "timeframe": "M15" if i < 3 else "M5", "zone_type": "SUPPLY", "price_low": 3910.0 + i * 5, "price_high": 3912.0 + i * 5, "invalidation": 3914.0 + i * 5, "risk_tier": "HIGH_RISK" if i == 0 else ("MEDIUM_RISK" if i == 1 else "UNCLASSIFIED"), "grade": "A" if i >= 3 else "B", "score": 73.4 + i, "status": "RADAR", "radar_eligibility": "QUALIFIED"}
            for i in range(5)
        ]
        res = eng.step(
            _make_tick(3899.9, 3900.1),
            frames={"M5": df1},
            df_m5=df1,
            snapshot_fn=lambda **kw: {"snapshot_id": "SNAP_5", "snapshot_status": "OK", "data_ok": True, "bid": 3899.9, "ask": 3900.1, "mid": 3900.0, "snapshot_price": 3900.0, "server_time": 1763000000},
            scan_fn=lambda **kw: {"snapshot_id": "SNAP_5", "radar_priority": {"radar_qualified": zones_5}},
            eval_fn=lambda **kw: {"final_decision": "WAIT", "confirmed_status": "WAIT", "pending_plan": {"status": "WAIT"}},
        )
        self.assertEqual(res["data_status"], "COMPLETE")
        self.assertEqual(len(res["watchlist"]), 5)
        rec = res["reconciliation"]
        self.assertEqual(rec["scanner_radar_qualified_count"], 5)
        self.assertEqual(rec["monitor_received_count"], 5)
        self.assertEqual(rec["monitor_active_count"], 5)
        self.assertEqual(rec["excluded_count"], 0)
        self.assertEqual(rec["snapshot_id"], "SNAP_5")
        self.assertEqual(rec["transfer_status"], "OK")
        self.assertFalse(res["snapshot_mismatch"])

    def test_17_zone_transfer_mismatch_and_snapshot_mismatch(self):
        # Jika scanner punya zona tapi semua invalid -> ZONE_TRANSFER_MISMATCH
        bad_scan = {
            "snapshot_id": "SNAP_SCAN",
            "radar_priority": {
                "radar_qualified": [
                    {"zone_id": "Z_INV", "timeframe": "M5", "zone_type": "SUPPLY", "price_low": 3910.0, "price_high": 3912.0, "invalidation": 3914.0, "status": "INVALIDATED", "invalidated": True}
                ]
            },
        }
        wl, rec = pantau.collect_radar_watchlist_with_reconciliation(bad_scan)
        self.assertEqual(len(wl), 0)
        self.assertEqual(rec["scanner_radar_qualified_count"], 1)
        self.assertEqual(rec["monitor_active_count"], 0)
        self.assertEqual(rec["transfer_status"], "ZONE_TRANSFER_MISMATCH")

        # Snapshot mismatch detection
        eng = pantau.PantauCycleEngine(symbol=pantau.SYMBOL)
        df1 = _make_df_m5(3900.0, "2026-11-13 09:00:00")
        res = eng.step(
            _make_tick(3899.9, 3900.1),
            frames={"M5": df1},
            df_m5=df1,
            snapshot_fn=lambda **kw: {"snapshot_id": "SNAP_MON", "snapshot_status": "OK", "data_ok": True, "bid": 3899.9, "ask": 3900.1, "mid": 3900.0, "snapshot_price": 3900.0, "server_time": 1763000000},
            scan_fn=lambda **kw: {"snapshot_id": "SNAP_OTHER", "radar_priority": {"radar_qualified": []}},
            eval_fn=lambda **kw: {"final_decision": "WAIT"},
        )
        self.assertTrue(res["snapshot_mismatch"])
        self.assertIn("SNAPSHOT_ID_MISMATCH", res["snapshot_mismatch_reason"])

    def test_18_idc_currency_and_unverified_currency(self):
        tick = _make_tick(bid=3898.00, ask=3898.20)
        pos_idc = _make_pos(ticket=42596664, symbol="XAUUSDc", is_buy=False, volume=0.01, price_open=3904.53, profit=98098.26)
        res_idc = pantau.audit_position_profit(
            pos_idc, tick, acc_info=_make_acc("IDC"), sym_info=_make_sym(100.0), expected_symbol="XAUUSDc", calc_profit_fn=lambda *a: 98098.26
        )
        self.assertTrue(res_idc["valid"])
        self.assertEqual(res_idc["display"], "+98,098.26 IDC")
        self.assertNotIn("$", res_idc["display"])

        # Tanpa acc_info / currency kosong -> CURRENCY_OR_PNL_UNVERIFIED (tidak pernah mengasumsikan USD)
        res_no_cur = pantau.audit_position_profit(pos_idc, tick, acc_info=None, sym_info=_make_sym(100.0), expected_symbol="XAUUSDc")
        self.assertFalse(res_no_cur["valid"])
        self.assertEqual(res_no_cur["status"], "CURRENCY_OR_PNL_UNVERIFIED")
        self.assertEqual(res_no_cur["display"], "CURRENCY_OR_PNL_UNVERIFIED")
        self.assertIn("MT5_RAW(", res_no_cur["diagnostic_log"])
        self.assertIn("MT5_RAW(", res_idc["diagnostic_log"])
        # Tanpa commission attribute -> net_profit=None (tidak mengarang 0.0)
        self.assertIsNone(res_idc["net_profit"])

    def test_19_profit_protection_suggestion_directions_stops_freeze_no_risk_free(self):
        src = inspect.getsource(pantau)
        self.assertNotIn("100% BEBAS RISIKO", src)
        self.assertNotIn("Risk-Free", src)

        # SELL: entry 3904.53, ask 3899.63 -> SL profit protection harus < entry & > ask
        pos_sell = _make_pos(is_buy=False, volume=0.01, price_open=3904.53)
        tick = _make_tick(bid=3899.43, ask=3899.63)
        sym_ok = SimpleNamespace(digits=2, point=0.01, trade_tick_size=0.01, trade_tick_value=1.0, trade_contract_size=100.0, trade_stops_level=50, trade_freeze_level=0)
        pp_sell = pantau.build_profit_protection_suggestion(pos_sell, tick, sym_info=sym_ok, acc_info=_make_acc("USD"))
        self.assertTrue(pp_sell["allowed"])
        self.assertEqual(pp_sell["label"], "PROFIT PROTECTION SUGGESTION")
        self.assertLess(pp_sell["suggested_sl"], 3904.53)
        self.assertGreater(pp_sell["suggested_sl"], 3899.63)
        self.assertGreater(pp_sell["gross_profit_locked_pts"], 0.0)
        self.assertIn("spread(", pp_sell["calculation_basis"])
        self.assertIn("Tindakan manual oleh trader", pp_sell["disclaimer"])

        # Tanpa sym_info -> PROFIT_PROTECTION_UNAVAILABLE (tidak mengarang SL)
        pp_unavail = pantau.build_profit_protection_suggestion(pos_sell, tick, sym_info=None, acc_info=_make_acc("USD"))
        self.assertFalse(pp_unavail["allowed"])
        self.assertEqual(pp_unavail["status"], "PROFIT_PROTECTION_UNAVAILABLE")
        self.assertIsNone(pp_unavail["suggested_sl"])

        # BUY: entry 3895.00, bid 3900.00 -> SL profit protection harus > entry & < bid
        pos_buy = _make_pos(is_buy=True, volume=0.01, price_open=3895.00)
        tick_buy = _make_tick(bid=3900.00, ask=3900.20)
        pp_buy = pantau.build_profit_protection_suggestion(pos_buy, tick_buy, sym_info=sym_ok, acc_info=_make_acc("USD"))
        self.assertTrue(pp_buy["allowed"])
        self.assertGreater(pp_buy["suggested_sl"], 3895.00)
        self.assertLess(pp_buy["suggested_sl"], 3900.00)

        # Stop level & freeze level restriction -> tidak mengarang level tidak valid
        sym_freeze = SimpleNamespace(digits=2, point=0.01, trade_tick_size=0.01, trade_tick_value=1.0, trade_contract_size=100.0, trade_stops_level=50, trade_freeze_level=600)
        pp_frz = pantau.build_profit_protection_suggestion(pos_sell, tick, sym_info=sym_freeze, acc_info=_make_acc("USD"))
        self.assertFalse(pp_frz["allowed"])
        self.assertIsNone(pp_frz["suggested_sl"])
        self.assertEqual(pp_frz["reason"], "FREEZE_LEVEL_RESTRICTION")

    def test_20_audit_8_pending_orders_and_structural_sl_unavailable(self):
        orders = [
            SimpleNamespace(ticket=500 + i, symbol=pantau.SYMBOL, type=2, volume_current=0.01, price_open=3890.0 - i, sl=0.0, tp=0.0)
            for i in range(8)
        ]
        tick = _make_tick(bid=3899.43, ask=3899.63)
        audited = pantau.audit_pending_orders(orders, tick, radar_zones=[])
        self.assertEqual(len(audited), 8)
        for oa in audited:
            self.assertEqual(oa["structural_sl"], "STRUCTURAL_SL_UNAVAILABLE")
            self.assertEqual(oa["price_validity"], "VALID_PRICE")
            self.assertEqual(oa["status"], "ACTIVE_PENDING")

    def test_21_reconciliation_scanner_3_received_8_active_3_excluded_5_reasons(self):
        rq3 = [
            {"zone_id": "RQ1", "timeframe": "M15", "zone_type": "SUPPLY", "price_low": 3910.0, "price_high": 3912.0, "invalidation": 3914.0, "risk_tier": "HIGH_RISK", "status": "RADAR", "radar_eligibility": "QUALIFIED"},
            {"zone_id": "RQ2", "timeframe": "M5", "zone_type": "SUPPLY", "price_low": 3920.0, "price_high": 3922.0, "invalidation": 3924.0, "risk_tier": "UNCLASSIFIED", "status": "RADAR", "radar_eligibility": "QUALIFIED"},
            {"zone_id": "RQ3", "timeframe": "H1", "zone_type": "DEMAND", "price_low": 3880.0, "price_high": 3884.0, "invalidation": 3877.0, "risk_tier": "MEDIUM_RISK", "status": "RADAR", "radar_eligibility": "QUALIFIED"},
        ]
        all8 = rq3 + [
            {"zone_id": "EX1", "timeframe": "H4", "zone_type": "SUPPLY", "price_low": 3990.0, "price_high": 3995.0, "invalidation": 4000.0, "risk_tier": "FAR_ZONE", "status": "RADAR", "radar_eligibility": "NOT_QUALIFIED"},
            {"zone_id": "EX2", "timeframe": "M15", "zone_type": "DEMAND", "price_low": 3870.0, "price_high": 3872.0, "invalidation": 3868.0, "risk_tier": "UNCLASSIFIED", "status": "INVALIDATED", "invalidated": True, "radar_eligibility": "NOT_QUALIFIED"},
            {"zone_id": "EX3", "timeframe": "M30", "zone_type": "SUPPLY", "price_low": 3930.0, "price_high": 3932.0, "invalidation": 3935.0, "risk_tier": "UNCLASSIFIED", "status": "MITIGATED", "mitigated": True, "radar_eligibility": "NOT_QUALIFIED"},
            {"zone_id": "EX4", "timeframe": "M5", "zone_type": "DEMAND", "price_low": 3860.0, "price_high": 3862.0, "invalidation": 3858.0, "risk_tier": "UNCLASSIFIED", "status": "EXPIRED", "expired": True, "radar_eligibility": "NOT_QUALIFIED"},
            {"zone_id": "EX5", "timeframe": "M15", "zone_type": "SUPPLY", "price_low": 3940.0, "price_high": 3942.0, "invalidation": 3945.0, "risk_tier": "UNCLASSIFIED", "status": "RADAR", "radar_eligibility": "NOT_QUALIFIED"},
        ]
        wl, rec = pantau.collect_radar_watchlist_with_reconciliation({
            "snapshot_id": "SNAP_3_8",
            "radar_priority": {"radar_qualified": rq3, "all_prioritized": all8},
        })
        self.assertEqual(rec["scanner_radar_qualified_count"], 3)
        self.assertEqual(rec["monitor_received_count"], 8)
        self.assertEqual(rec["monitor_active_count"], 3)
        self.assertEqual(rec["excluded_count"], 5)
        self.assertEqual(rec["excluded_from_radar_qualified_count"], 0)
        self.assertEqual(rec["transfer_status"], "OK")
        self.assertEqual([z["zone_id"] for z in wl], ["RQ1", "RQ2", "RQ3"])
        reasons_by_id = {d["zone_id"]: d["reason"] for d in rec["excluded_details"]}
        self.assertEqual(reasons_by_id, {
            "EX1": "FAR_ZONE_EXCLUDED",
            "EX2": "INVALIDATED",
            "EX3": "MITIGATED",
            "EX4": "EXPIRED",
            "EX5": "NOT_RADAR_QUALIFIED",
        })

    def test_22_global_m5_structure_separated_from_zone_specific_and_not_checked(self):
        dec = {
            "final_decision": "READY",  # Simulasi anomali jika ada yang mencoba READY tanpa Zone Touch
            "m5_stages": {
                "zone_touch": {"status": "FAIL"},
                "liquidity_sweep": {"status": "NOT_CHECKED"},
                "structure_trigger": {"status": "NOT_CHECKED"},
                "displacement": {"status": "NOT_CHECKED"},
            },
        }
        scopes = pantau.extract_m5_validation_scopes("BOS bear 3893.6", dec)
        self.assertEqual(scopes["GLOBAL_M5_STRUCTURE"], "BOS bear 3893.6")
        self.assertEqual(scopes["ZONE_TOUCH"], "FAIL")
        self.assertEqual(scopes["ZONE_SPECIFIC_SWEEP"], "NOT_CHECKED")
        self.assertEqual(scopes["ZONE_SPECIFIC_CHOCH_BOS_CISD"], "NOT_CHECKED")
        self.assertEqual(scopes["ZONE_SPECIFIC_DISPLACEMENT"], "NOT_CHECKED")
        # Global BOS tidak boleh membuat Confirmed Entry menjadi READY saat Zone Touch belum terpenuhi
        self.assertEqual(scopes["CONFIRMED_ENTRY_STATUS"], "WAIT")


if __name__ == "__main__":
    unittest.main()

