import os
import unittest
from unittest import mock

from modules.adaptive_entry_engine import evaluate_entry_paths_for_zone
from modules.config import compute_zone_limit_entry_price, get_auto_trading_config, resolve_zone_limit_depth
from modules.dynamic_lot_manager import select_target_lot
from modules.trade_validator import evaluate_directional_alignment, validate_zone_quality_layer


class LooseModeTests(unittest.TestCase):
    def _loose_env(self):
        return mock.patch.dict(os.environ, {"AUTO_LOOSE_MODE": "true", "LOOSE_MAX_ACTIVE": "5"})

    def test_default_off_keeps_strict_rules(self):
        with mock.patch.dict(os.environ, {"AUTO_LOOSE_MODE": "false"}):
            cfg = get_auto_trading_config()
        self.assertFalse(cfg["AUTO_LOOSE_MODE"])
        self.assertEqual(cfg["MAX_ACTIVE_POSITIONS"], 1)
        self.assertEqual(cfg["MIN_ZONE_SCORE"], 70.0)

    def test_loose_overrides(self):
        with self._loose_env():
            cfg = get_auto_trading_config()
        self.assertTrue(cfg["AUTO_LOOSE_MODE"])
        self.assertEqual(cfg["MAX_ACTIVE_POSITIONS"], 5)
        self.assertEqual(cfg["MAX_ACTIVE_PENDING_ORDERS"], 5)
        self.assertEqual(cfg["TP1_RR"], 2.0)
        self.assertEqual(cfg["MIN_RR"], 2.0)
        self.assertTrue(cfg["REQUIRE_STOP_LOSS"])
        self.assertTrue(cfg["REQUIRE_TAKE_PROFIT"])

    def test_low_grade_zone_gets_min_lot_and_not_blocked(self):
        zone = {"is_valid": True, "status": "ACTIVE", "score": 55.0, "grade": "C", "timeframe": "M30"}
        with self._loose_env():
            cfg = get_auto_trading_config()
        self.assertEqual(select_target_lot(zone, "ADAPTIVE_ENTRY", cfg=cfg), 0.01)
        strict = dict(cfg, AUTO_LOOSE_MODE=False, MIN_ZONE_SCORE=70.0)
        self.assertEqual(select_target_lot(zone, "ADAPTIVE_ENTRY", cfg=strict), 0.0)

    def test_htf_conflict_ignored_only_in_loose(self):
        with self._loose_env():
            cfg = get_auto_trading_config()
        ok, _, _ = evaluate_directional_alignment("SELL", "ADAPTIVE_ENTRY", "BEARISH", "BULLISH", "CONFLICT", True, cfg=cfg)
        self.assertTrue(ok)
        strict = dict(cfg, AUTO_LOOSE_MODE=False)
        ok2, _, _ = evaluate_directional_alignment("SELL", "ADAPTIVE_ENTRY", "BEARISH", "BULLISH", "CONFLICT", True, cfg=strict)
        self.assertFalse(ok2)

    def test_limit_order_uses_midzone_not_front_edge(self):
        # Supply 3600-3608 -> tengah zona 3604.00 (bukan awal zona 3600.00)
        self.assertEqual(compute_zone_limit_entry_price(3600.0, 3608.0, demand=False), 3604.0)
        # Demand 3550-3560 -> tengah zona 3555.00 (bukan awal zona 3560.00)
        self.assertEqual(compute_zone_limit_entry_price(3550.0, 3560.0, demand=True), 3555.0)
        # Bahkan bila depth diset 0.0 tetap di-clamp minimal 25% masuk zona (tidak pernah di awal zona)
        self.assertEqual(compute_zone_limit_entry_price(3600.0, 3608.0, demand=False, depth=0.0), 3602.0)
        self.assertEqual(compute_zone_limit_entry_price(3550.0, 3560.0, demand=True, depth=0.0), 3557.5)

        supply_z = {
            "zone_id": "SNR_M30_RES_3600_3608",
            "symbol": "XAUUSDc",
            "timeframe": "M30",
            "zone_type": "SUPPLY",
            "price_low": 3600.0,
            "price_high": 3608.0,
            "midpoint": 3604.0,
            "score": 68.0,
            "grade": "C",
            "freshness": "FRESH",
            "touch_count": 0,
            "invalidation_price": 3608.0,
            "is_valid": True,
            "status": "ACTIVE",
        }
        with self._loose_env():
            cfg = get_auto_trading_config()
        # 1) Harga belum menyentuh zona (3590) -> SELL LIMIT di tengah zona 3604.00, bukan 3600.00
        res_far = evaluate_entry_paths_for_zone(
            supply_z, frames=None, current_price=3590.0, bid=3589.9, ask=3590.1, spread=0.2,
            htf_bias="BEARISH", h1_bias="BULLISH", m30_status="CONFLICT", has_htf_conflict=True, cfg=cfg,
        )
        self.assertEqual(res_far["order_kind"], "LIMIT")
        self.assertEqual(res_far["order_type"], "SELL LIMIT")
        self.assertEqual(res_far["levels"]["entry_price"], 3604.0)
        self.assertGreater(res_far["levels"]["sl"], 3608.0)

        # 2) Harga baru menyentuh awal zona tipis (3600.40 < 3604.00) -> tetap pasang SELL LIMIT di 3604.00, bukan SELL MARKET di awal zona!
        res_edge = evaluate_entry_paths_for_zone(
            supply_z, frames=None, current_price=3600.5, bid=3600.4, ask=3600.6, spread=0.2,
            htf_bias="BEARISH", h1_bias="BULLISH", m30_status="CONFLICT", has_htf_conflict=True, cfg=cfg,
        )
        self.assertEqual(res_edge["order_kind"], "LIMIT")
        self.assertEqual(res_edge["order_type"], "SELL LIMIT")
        self.assertEqual(res_edge["levels"]["entry_price"], 3604.0)

        # 3) Zona Grade B normal (bukan HIGH_RISK) pada 3600-3608 -> ambil 25% masuk zona = 3602.00!
        supply_b = dict(supply_z, zone_id="SNR_M30_RES_B", score=72.0, grade="B", risk_tier="MEDIUM_RISK")
        self.assertEqual(resolve_zone_limit_depth(zone=supply_b, cfg=cfg), 0.25)
        res_b = evaluate_entry_paths_for_zone(
            supply_b, frames=None, current_price=3590.0, bid=3589.9, ask=3590.1, spread=0.2,
            htf_bias="BEARISH", h1_bias="BULLISH", m30_status="CONFLICT", has_htf_conflict=True, cfg=cfg,
        )
        self.assertEqual(res_b["order_kind"], "LIMIT")
        self.assertEqual(res_b["levels"]["entry_price"], 3602.0)

        # 4) Zona Grade B tapi berstatus HIGH_RISK -> tetap ambil 50% tengah zona = 3604.00!
        supply_b_hr = dict(supply_b, zone_id="SNR_M30_RES_B_HR", risk_tier="HIGH_RISK")
        self.assertEqual(resolve_zone_limit_depth(zone=supply_b_hr, cfg=cfg), 0.50)
        res_b_hr = evaluate_entry_paths_for_zone(
            supply_b_hr, frames=None, current_price=3590.0, bid=3589.9, ask=3590.1, spread=0.2,
            htf_bias="BEARISH", h1_bias="BULLISH", m30_status="CONFLICT", has_htf_conflict=True, cfg=cfg,
        )
        self.assertEqual(res_b_hr["levels"]["entry_price"], 3604.0)

        # 5) Demand Grade B normal pada 3550-3560 -> BUY LIMIT 25% masuk dari atas (3560 - 2.50 = 3557.50)
        demand_b = dict(
            supply_b, zone_id="SNR_H1_SUP_B", zone_type="DEMAND",
            price_low=3550.0, price_high=3560.0, midpoint=3555.0, invalidation_price=3550.0,
        )
        res_dem_b = evaluate_entry_paths_for_zone(
            demand_b, frames=None, current_price=3570.0, bid=3569.9, ask=3570.1, spread=0.2,
            htf_bias="BULLISH", h1_bias="BEARISH", m30_status="CONFLICT", has_htf_conflict=True, cfg=cfg,
        )
        self.assertEqual(res_dem_b["order_type"], "BUY LIMIT")
        self.assertEqual(res_dem_b["levels"]["entry_price"], 3557.5)

    def test_canonical_zone_key_and_stable_continuation_id_prevent_multi_layer_duplicates(self):
        import numpy as np
        import pandas as pd
        from modules.execution_state import ExecutionStateManager, canonical_zone_key, make_setup_id
        from modules.trend_following import build_trend_following_candidates

        # 1) Wrapper TF_PB_SNR_* dan SNR_* wajib menghasilkan canonical key & setup_id identik
        raw_snr = "SNR_M30_RES_2026-11-26T18:30:00_357003"
        wrapped_snr = f"TF_PB_{raw_snr}"
        self.assertEqual(canonical_zone_key(wrapped_snr), raw_snr)
        self.assertEqual(make_setup_id(raw_snr, "SELL"), make_setup_id(wrapped_snr, "SELL"))

        # 2) TF_CONT_* zone_id tetap stabil walau current_price bergeser antar tick (anti-duplikat 11 detik)
        idx_m5 = pd.date_range("2026-11-27 04:00:00", periods=40, freq="5min")
        base_px = np.linspace(3585.0, 3573.0, 40)
        # Buat swing low di bar 20 lalu BOS bearish di bar 32
        lows = base_px - 0.8
        highs = base_px + 0.8
        closes = base_px.copy()
        lows[20] = 3574.0
        closes[20] = 3574.5
        for k in range(21, 31):
            lows[k] = 3575.5
            highs[k] = 3577.5
            closes[k] = 3576.5
        closes[32:] = 3572.8
        lows[32:] = 3572.2
        highs[32:] = 3573.6
        df_m5 = pd.DataFrame({
            "Open": closes + 0.3,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": 100,
            "ATR14": 1.5,
            "EMA20": 3575.0,
            "EMA50": 3578.0,
            "RSI14": 40.0,
        }, index=idx_m5)
        idx_m30 = pd.date_range("2026-11-26 12:00:00", periods=30, freq="30min")
        df_m30 = pd.DataFrame({
            "Open": 3576.0, "High": 3577.5, "Low": 3572.0, "Close": 3573.0,
            "Volume": 500, "ATR14": 3.0, "EMA20": 3575.5, "EMA50": 3580.0, "RSI14": 42.0,
        }, index=idx_m30)
        frames = {"M5": df_m5, "M30": df_m30, "H1": df_m30}
        h4_info = {"bias": "BEARISH", "ok": True}
        h1_info = {"status": "ALIGNED", "ok": True}
        m30_info = {"status": "CONFIRMED", "condition": "ALIGNED", "ok": True}

        c1 = [c for c in build_trend_following_candidates(frames, h4_info, h1_info, m30_info, 3572.71) if c["setup_type"] == "TREND_CONTINUATION"]
        c2 = [c for c in build_trend_following_candidates(frames, h4_info, h1_info, m30_info, 3572.84) if c["setup_type"] == "TREND_CONTINUATION"]
        self.assertTrue(c1 and c2)
        self.assertEqual(c1[0]["zone_id"], c2[0]["zone_id"])

    def test_grade_a_76_and_80plus_two_stage_works_with_strict_mode(self):
        from modules.config import is_scale_in_80plus_zone
        from modules.dynamic_lot_manager import select_target_lot

        cfg_strict = dict(get_auto_trading_config(), AUTO_LOOSE_MODE=False, MAX_ACTIVE_POSITIONS=1, LOT_MEDIUM=0.02, LOT_STANDARD=0.01)

        # Grade A skor 76.5 -> WAJIB masuk 2-Tahap (Tahap 1 Limit 0.01 -> Tahap 2 Market 0.02 setelah valid M5)
        z_grade_a = {"zone_id": "SMC_H1_GA_765", "is_valid": True, "status": "ACTIVE", "score": 76.5, "grade": "A", "timeframe": "H1", "zone_type": "SUPPLY"}
        self.assertTrue(is_scale_in_80plus_zone(z_grade_a, cfg=cfg_strict))
        self.assertEqual(select_target_lot(z_grade_a, "EARLY_PENDING", cfg=cfg_strict), 0.01)
        self.assertEqual(select_target_lot(z_grade_a, "ADAPTIVE_ENTRY", cfg=cfg_strict), 0.02)

        # Skor >= 80 (meski label B) -> WAJIB masuk 2-Tahap
        z_82 = {"zone_id": "SNR_H1_82", "is_valid": True, "status": "ACTIVE", "score": 82.0, "grade": "B", "timeframe": "H1", "zone_type": "SUPPLY"}
        self.assertTrue(is_scale_in_80plus_zone(z_82, cfg=cfg_strict))

        # Grade B skor < 80 (mis. 74.0) -> BUKAN 2-Tahap (hanya 1x entry, maksimal 0.01 lot)
        z_grade_b = {"zone_id": "SNR_H1_74", "is_valid": True, "status": "ACTIVE", "score": 74.0, "grade": "B", "timeframe": "H1", "zone_type": "SUPPLY"}
        self.assertFalse(is_scale_in_80plus_zone(z_grade_b, cfg=cfg_strict))
        self.assertEqual(select_target_lot(z_grade_b, "ADAPTIVE_ENTRY", cfg=cfg_strict), 0.01)

        # End-to-end Grade A (skor 76.5) pada mode strict (AUTO_LOOSE_MODE=False, MAX_ACTIVE_POSITIONS=1):
        # Tahap 1 Limit 0.01 -> kejemput tanpa M5 confirm (WAIT_TRIGGER) -> M5 valid close -> tambah +0.02 lot (maks 1x)!
        import time
        from types import SimpleNamespace
        from modules.config import SCANNER_MAGIC_NUMBER
        from modules.execution_monitor import AutoExecutionMonitor
        from tests.test_adaptive_auto_trading import TestAdaptiveAutoTrading40Scenarios, _make_snapshot_and_scan, _make_supply_zone

        import modules.utils as _u_iso
        _u_iso._load_trade_plan_state_once()
        _saved_inv = dict(_u_iso._INVALIDATED_SETUP_VERSIONS)
        _u_iso._INVALIDATED_SETUP_VERSIONS.clear()
        helper = TestAdaptiveAutoTrading40Scenarios()
        helper.setUp()
        try:
            cfg_e2e = dict(helper.cfg)
            cfg_e2e.update({
                "AUTO_LOOSE_MODE": False,
                "MAX_ACTIVE_POSITIONS": 1,
                "LOT_STANDARD": 0.01,
                "LOT_MEDIUM": 0.02,
                "LOT_STRONG": 0.03,
                "EARLY_PENDING_MAX_LOT": 0.01,
                "MAX_LOT": 0.03,
                "MIN_RR": 1.0,
                "TP1_RR": 1.0,
                "TP1_MODE": "FIXED_RR",
                "FORCE_FIXED_TP1": True,
                "SCALE_IN_80PLUS_ENABLED": True,
                "SCALE_IN_MIN_SCORE": 80.0,
                "GRADE_A_MIN_SCORE": 75.0,
                "RISK_EARLY_PENDING_PERCENT": 2.5,
                "RISK_ADAPTIVE_PERCENT": 2.5,
                "MAX_RISK_PER_TRADE_PERCENT": 2.5,
                "MAX_TOTAL_EXPOSURE_PERCENT": 5.0,
                "SNR_ZONE_MIN_STABLE_SECONDS": 0.0,
            })
            mt5_m, sm_m, jrn_m, _ = helper._new_env(equity_idc=20_000_000.0, bid=3630.00, ask=3630.20)
            mon_a = AutoExecutionMonitor(mt5_api=mt5_m, state_mgr=sm_m, journal=jrn_m, symbol="XAUUSDc", cfg=cfg_e2e, connect_fn=lambda: True)
            df_untouched = helper.df_m5_rej.iloc[:-15].copy()
            df_untouched.attrs["is_closed"] = True

            z_a76 = _make_supply_zone(score=76.5, grade="A", timeframe="H1", lo=3643.00, hi=3647.00)
            z_a76.zone_id = "ZONE_H1_GA_765_E2E"
            z_a76.sl_recommended = 3649.00

            # 1) Sebelum zona kejemput -> pasang SELL LIMIT 0.01 lot dulu
            s1, sc1, d1 = _make_snapshot_and_scan(z_a76, df_untouched, bid=3630.00, ask=3630.20)
            s1["server_time"] = df_untouched.index[-1]
            d1["plan"]["sl"] = 3649.00
            c1_out = mon_a.run_once(snapshot=s1, scan_result=sc1, decision=d1, news_override=helper.news_ok)
            self.assertTrue(c1_out["execution_outcome"]["executed"])
            self.assertEqual(len(mt5_m.orders), 1)
            self.assertAlmostEqual(mt5_m.orders[0].volume_current, 0.01, places=2)

            # 2) Limit 0.01 kejemput, tapi M5 belum konfirmasi -> wajib WAIT_TRIGGER
            ord_1 = mt5_m.orders.pop(0)
            mt5_m.positions.append(SimpleNamespace(
                ticket=ord_1.ticket, symbol="XAUUSDc", type=mt5_m.ORDER_TYPE_SELL, volume=0.01,
                price_open=ord_1.price_open, sl=ord_1.sl, tp=ord_1.tp, profit=0.0,
                swap=0.0, commission=0.0, magic=SCANNER_MAGIC_NUMBER, comment=ord_1.comment, time=int(time.time()),
            ))
            mt5_m.bid, mt5_m.ask = 3644.50, 3644.70
            s2, sc2, d2 = _make_snapshot_and_scan(z_a76, df_untouched, bid=3644.50, ask=3644.70)
            s2["server_time"] = df_untouched.index[-1]
            d2["plan"]["sl"] = 3649.00
            c2_out = mon_a.run_once(snapshot=s2, scan_result=sc2, decision=d2, news_override=helper.news_ok)
            self.assertIsNone(c2_out["execution_outcome"])
            self.assertEqual(len(mt5_m.positions), 1)

            # 3) Candle M5 close valid (Rejection) -> tambah entry ke-2 (+0.02 lot) meskipun MAX_ACTIVE_POSITIONS=1!
            mt5_m.bid, mt5_m.ask = 3644.00, 3644.20
            s3, sc3, d3 = _make_snapshot_and_scan(z_a76, helper.df_m5_rej, bid=3644.00, ask=3644.20)
            s3["server_time"] = helper.df_m5_rej.index[-1]
            d3["plan"]["sl"] = 3649.00
            c3_out = mon_a.run_once(snapshot=s3, scan_result=sc3, decision=d3, news_override=helper.news_ok)
            self.assertTrue(c3_out["execution_outcome"]["executed"])
            self.assertEqual([round(p.volume, 2) for p in mt5_m.positions], [0.01, 0.02])

            # 4) Siklus berikutnya -> terkunci, TIDAK nambah layer ke-3!
            c4_out = mon_a.run_once(snapshot=s3, scan_result=sc3, decision=d3, news_override=helper.news_ok)
            self.assertIsNone(c4_out["execution_outcome"])
            self.assertEqual(len(mt5_m.positions), 2)
        finally:
            _u_iso._INVALIDATED_SETUP_VERSIONS.clear()
            _u_iso._INVALIDATED_SETUP_VERSIONS.update(_saved_inv)
            helper.tearDown()

    def test_snr_active_zone_not_falsely_invalidated_by_historical_ob_rule_and_expired_unfilled_pending_rearms(self):
        import time
        from modules.execution_monitor import AutoExecutionMonitor
        from modules.execution_state import STATE_EXPIRED, STATE_ORDER_PLACED, make_setup_id
        from tests.test_adaptive_auto_trading import TestAdaptiveAutoTrading40Scenarios

        import modules.utils as _u_iso
        _u_iso._load_trade_plan_state_once()
        _saved_inv = dict(_u_iso._INVALIDATED_SETUP_VERSIONS)
        _u_iso._INVALIDATED_SETUP_VERSIONS.clear()
        helper = TestAdaptiveAutoTrading40Scenarios()
        helper.setUp()
        try:
            cfg_live = dict(helper.cfg)
            cfg_live.update({
                "AUTO_LOOSE_MODE": False,
                "CONFLICT_MIDZONE_LIMIT_ENABLED": True,
                "CONFLICT_MIDZONE_MIN_SCORE": 70.0,
                "CONFLICT_MIDZONE_MAX_WIDTH": 16.0,
                "CONFLICT_MIDZONE_SL_BUFFER": 0.60,
                "REAL_LOT_001_FIXED_TP_ENABLED": True,
                "REAL_LOT_001_FIXED_TP_PIPS": 110.0,
                "MIN_RR": 1.0,
                "TP1_RR": 1.0,
                "TP1_MODE": "FIXED_RR",
                "FORCE_FIXED_TP1": True,
                "RISK_EARLY_PENDING_PERCENT": 2.5,
                "MAX_RISK_PER_TRADE_PERCENT": 2.5,
                "MAX_TOTAL_EXPOSURE_PERCENT": 5.0,
            })
            mt5_m, sm_m, jrn_m, _ = helper._new_env(equity_idc=18_300_000.0, bid=3616.14, ask=3616.34)
            mon = AutoExecutionMonitor(mt5_api=mt5_m, state_mgr=sm_m, journal=jrn_m, symbol="XAUUSDc", cfg=cfg_live, connect_fn=lambda: True)

            # Simulasikan pending lama kemarin yang expired tanpa pernah fill -> wajib boleh dipasang ulang!
            zid_r1 = "SNR_M30_RES_2026-11-24T14:30:00_363218"
            sid_r1 = make_setup_id(zid_r1, "SELL")
            sm_m.update_setup_state(sid_r1, STATE_ORDER_PLACED, zone_id=zid_r1, direction="SELL", order_ticket=4544632327)
            sm_m.update_setup_state(sid_r1, STATE_EXPIRED, reason_code="PENDING_CANCELLED_OR_EXPIRED", zone_id=zid_r1, direction="SELL")
            dup, _ = sm_m.is_setup_duplicate_or_locked(sid_r1, zone_id=zid_r1)
            self.assertFalse(dup)

            # Simulasikan bar M30 historis sempat close 3642.00 setelah zona SNR 3634.44-3640.76 terbentuk,
            # namun engine SNR menyatakan zona tetap ACTIVE (reclaimed/valid multi-reaction SNR) & harga live 3616.24!
            df_m30 = helper.df_m5_rej.copy()
            df_m30.iloc[5, df_m30.columns.get_loc("Close")] = 3642.00
            df_m30.attrs["is_closed"] = True
            snap = {
                "symbol": "XAUUSDc", "bid": 3616.14, "ask": 3616.34, "mid": 3616.24,
                "current_price": 3616.24, "spread": 0.20, "market_status": "OPEN",
                "data_freshness": "FRESH", "tick_age": 0.2,
                "server_time": df_m30.index[-1],
                "frames": {"M30": df_m30, "M5": df_m30},
            }
            scan_smc = {
                "symbol": "XAUUSDc", "current_price": 3616.24, "bid": 3616.14, "ask": 3616.34,
                "mid": 3616.24, "spread": 0.20, "market_status": "OPEN", "data_freshness": "FRESH",
                "htf_bias": "BEARISH", "h1_structure": "BULLISH", "m30_status": "NETRAL",
                "htf_context": {"H4": "BEARISH", "H1": "BULLISH", "m30_status": "NETRAL", "has_conflict": True},
                "active_zones": [], "radar_zones": [],
            }
            snr_zone_focus = {
                "zone_id": "SNR_M30_RES_2026-11-24T05:00:00_364076",
                "zone_type": "RESISTANCE", "timeframe": "M30",
                "low": 3634.44, "high": 3640.76, "score": 80.74, "grade": "A",
                "lifecycle_status": "ACTIVE", "confirmed_time": str(df_m30.index[0]), "touch_count": 1,
            }
            snr_res = {
                "scan_result": {"symbol": "XAUUSDc", "resistances": [snr_zone_focus], "supports": []},
                "decision": {"target_zone": snr_zone_focus},
            }
            cyc = mon.run_once(snapshot=snap, scan_result=scan_smc, decision={}, snr_result=snr_res, news_override=helper.news_ok)
            self.assertIsNotNone(cyc["execution_outcome"], msg=str(cyc.get("best_evaluation")))
            self.assertTrue(cyc["execution_outcome"]["executed"])
            self.assertEqual(len(mt5_m.orders), 1)
            self.assertAlmostEqual(mt5_m.orders[0].price_open, 3637.60, places=2)
            self.assertAlmostEqual(mt5_m.orders[0].volume_current, 0.01, places=2)

            # Uji juga skenario AUTO_LOOSE_MODE=True + harga sempat sentuh zona lalu turun ke 3623.09 (left_entry_range)
            # + SL struktural > 5.50 USD dengan TP 110 pip -> WAJIB tetap pasang SELL LIMIT 0.01 lot (bukan PRICE_LEFT_ENTRY_RANGE / RR_BELOW_MINIMUM)!
            mt5_l, sm_l, jrn_l, _ = helper._new_env(equity_idc=18_300_000.0, bid=3622.99, ask=3623.19)
            cfg_loose_live = dict(cfg_live, AUTO_LOOSE_MODE=True, MIN_RR=2.0, TP1_RR=2.0, BASE_MIN_RR=1.0, CONFLICT_MIDZONE_LIMIT_ENABLED=False)
            mon_l = AutoExecutionMonitor(mt5_api=mt5_l, state_mgr=sm_l, journal=jrn_l, symbol="XAUUSDc", cfg=cfg_loose_live, connect_fn=lambda: True)
            df_m5_touched_ago = helper.df_m5_rej.copy()
            # Buat 1 bar M5 beberapa bar lalu sempat menyentuh 3635.0, lalu turun ke 3623.09
            df_m5_touched_ago["High"] = 3625.0
            df_m5_touched_ago["Low"] = 3621.0
            df_m5_touched_ago["Close"] = 3623.0
            df_m5_touched_ago["Open"] = 3623.5
            df_m5_touched_ago.iloc[-6, df_m5_touched_ago.columns.get_loc("High")] = 3635.20
            df_m5_touched_ago.attrs["is_closed"] = True
            snap_l = dict(snap, bid=3622.99, ask=3623.19, mid=3623.09, current_price=3623.09, frames={"M30": df_m5_touched_ago, "M5": df_m5_touched_ago})
            scan_l = dict(scan_smc, bid=3622.99, ask=3623.19, mid=3623.09, current_price=3623.09)
            snr_h1_4680 = {
                "zone_id": "SNR_H1_RES_2026-11-24T12:00:00_364680",
                "zone_type": "RESISTANCE", "timeframe": "H1",
                "low": 3641.05, "high": 3646.80, "score": 80.50, "grade": "A",
                "lifecycle_status": "ACTIVE", "confirmed_time": str(df_m5_touched_ago.index[0]), "touch_count": 0,
            }
            snr_m30_4076 = dict(snr_zone_focus, touch_count=1)
            snr_res_l = {
                "scan_result": {"symbol": "XAUUSDc", "resistances": [snr_h1_4680, snr_m30_4076], "supports": []},
                "decision": {"target_zone": snr_h1_4680},
            }
            cyc_l = mon_l.run_once(snapshot=snap_l, scan_result=scan_l, decision={}, snr_result=snr_res_l, news_override=helper.news_ok)
            self.assertIsNotNone(cyc_l["execution_outcome"], msg=str(cyc_l.get("best_evaluation")))
            self.assertTrue(cyc_l["execution_outcome"]["executed"])
            self.assertEqual(len(mt5_l.orders), 1)
            self.assertAlmostEqual(mt5_l.orders[0].volume_current, 0.01, places=2)

            # Pastikan siklus ke-2 (1 detik kemudian) TIDAK membatalkan pending order baru hanya karena setup pertama terdeteksi kemarin (> 8 jam lalu)!
            sid_placed = make_setup_id("SNR_H1_RES_2026-11-24T12:00:00_364680", "SELL")
            sm_l.state["setups"][sid_placed]["created_epoch"] = time.time() - 86400.0  # 24 jam lalu di radar!
            cyc_l2 = mon_l.run_once(snapshot=snap_l, scan_result=scan_l, decision={}, snr_result=snr_res_l, news_override=helper.news_ok)
            self.assertEqual(len(mt5_l.orders), 1, msg="Pending order baru TIDAK boleh auto-cancel 1 detik kemudian gara-gara created_epoch lama!")
        finally:
            _u_iso._INVALIDATED_SETUP_VERSIONS.clear()
            _u_iso._INVALIDATED_SETUP_VERSIONS.update(_saved_inv)
            helper.tearDown()


if __name__ == "__main__":
    unittest.main()
