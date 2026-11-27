# -*- coding: utf-8 -*-
"""Unit test Session + PO3/AMD Engine (data sintetis; tidak menyentuh MT5 / order apapun)."""
from __future__ import annotations

import unittest
from datetime import datetime, timedelta
from unittest import mock

import numpy as np
import pandas as pd

from modules import session_amd_engine as amd
from modules.session_amd_engine import (
    PHASE_ACCUMULATION, PHASE_DIRECT_EXPANSION, PHASE_DISTRIBUTION, PHASE_MANIPULATION, PHASE_NO_CLEAR,
    analyze_session_amd, evaluate_zone_amd_alignment, session_at, session_bounds_utc,
)


def _bars(spec, start_price, seed=7):
    """spec: list segmen ('range', n, center, sigma) | ('move', n, to_price, wick) | ('bar', o, h, l, c)."""
    rng = np.random.default_rng(seed)
    out, prev, x, k = [], start_price, 0.0, 0
    for seg in spec:
        kind = seg[0]
        if kind == "osc":         # osilasi sinus deterministik (ekstrem berulang persis => tidak ada sweep palsu)
            _, n, center, amp = seg
            for _ in range(n):
                c = center + amp * np.sin(2 * np.pi * k / 24.0)
                k += 1
                out.append((prev, max(prev, c) + 0.2, min(prev, c) - 0.2, c))
                prev = c
        elif kind == "range":       # mean-reverting (AR1) => kompresi dengan efisiensi rendah
            _, n, center, sigma = seg
            for _ in range(n):
                x = 0.9 * x + rng.normal(0.0, sigma)
                c = center + x
                out.append((prev, max(prev, c) + 0.2, min(prev, c) - 0.2, c))
                prev = c
        elif kind == "move":
            _, n, to_price, wick = seg
            step = (to_price - prev) / n
            for _ in range(n):
                c = prev + step
                out.append((prev, max(prev, c) + wick, min(prev, c) - wick, c))
                prev = c
        elif kind == "bar":
            _, o, h, l, c = seg
            out.append((o, h, l, c))
            prev = c
    return out


def _df(bars, end="2026-01-14 14:00", tz="UTC"):
    idx = pd.date_range(end=pd.Timestamp(end) - pd.Timedelta(minutes=5), periods=len(bars), freq="5min")
    df = pd.DataFrame(bars, columns=["Open", "High", "Low", "Close"], index=idx)
    df.attrs["time_tz"] = tz
    return df


def _lead(center=2000.0, n=230):
    return [("osc", n, center, 4.0)]


class SessionTests(unittest.TestCase):
    def test_dst_aware_session_bounds(self):
        w = session_bounds_utc(datetime(2026, 1, 14))
        s = session_bounds_utc(datetime(2026, 7, 14))
        self.assertEqual((w["london_open"], w["ny_open"]), (8.0, 13.0))
        self.assertEqual((s["london_open"], s["ny_open"]), (7.0, 12.0))
        self.assertEqual(session_at(datetime(2026, 1, 14, 3, 0)), "ASIA")
        self.assertEqual(session_at(datetime(2026, 1, 14, 9, 0)), "LONDON")
        self.assertEqual(session_at(datetime(2026, 1, 14, 14, 0)), "LONDON_NY_OVERLAP")
        self.assertEqual(session_at(datetime(2026, 1, 14, 19, 0)), "NEW_YORK")
        self.assertEqual(session_at(datetime(2026, 1, 14, 23, 0)), "OFF_HOURS")


class PhaseTests(unittest.TestCase):
    def test_unavailable_when_data_short(self):
        r = analyze_session_amd(_df(_bars([("range", 50, 2000.0, 2.0)], 2000.0)))
        self.assertFalse(r["available"])
        self.assertEqual(r["phase"], amd.PHASE_UNAVAILABLE)
        self.assertFalse(analyze_session_amd(None)["available"])

    def test_accumulation_active(self):
        r = analyze_session_amd(_df(_bars(_lead(), 2000.0)))
        self.assertTrue(r["available"], r)
        self.assertEqual(r["phase"], PHASE_ACCUMULATION, r)
        self.assertEqual(r["bias"], "NEUTRAL")

    def test_low_sweep_reclaim_is_bullish_manipulation_or_distribution(self):
        spec = _lead() + [
            ("bar", 2000.0, 2000.5, 1992.5, 1999.5),      # sweep low lalu close kembali ke dalam
            ("move", 12, 2018.0, 0.3),                    # displacement naik
        ]
        r = analyze_session_amd(_df(_bars(spec, 2000.0)))
        self.assertIn(r["phase"], (PHASE_MANIPULATION, PHASE_DISTRIBUTION), r)
        self.assertEqual(r["bias"], "BULLISH")
        self.assertIsNotNone(r["event"])

    def test_high_sweep_reclaim_is_bearish(self):
        spec = _lead() + [
            ("bar", 2000.0, 2007.5, 1999.5, 2000.5),
            ("move", 12, 1982.0, 0.3),
        ]
        r = analyze_session_amd(_df(_bars(spec, 2000.0)))
        self.assertIn(r["phase"], (PHASE_MANIPULATION, PHASE_DISTRIBUTION), r)
        self.assertEqual(r["bias"], "BEARISH")

    def test_breakout_that_holds_is_direct_expansion_not_manipulation(self):
        spec = _lead() + [("move", 16, 2020.0, 0.3), ("range", 6, 2021.0, 0.5)]
        r = analyze_session_amd(_df(_bars(spec, 2000.0)))
        self.assertEqual(r["phase"], PHASE_DIRECT_EXPANSION, r)
        self.assertEqual(r["bias"], "BULLISH")

    def test_steady_trend_is_not_forced_into_accumulation_or_manipulation(self):
        spec = [("move", 230, 2200.0, 0.4)]
        r = analyze_session_amd(_df(_bars(spec, 2000.0)))
        self.assertIn(r["phase"], (PHASE_NO_CLEAR, PHASE_DIRECT_EXPANSION), r)
        if r["phase"] == PHASE_DIRECT_EXPANSION:
            self.assertEqual(r["bias"], "BULLISH")

    def test_phase_does_not_depend_on_clock_hours(self):
        spec = _lead() + [("bar", 2000.0, 2000.5, 1992.5, 1999.5), ("move", 12, 2018.0, 0.3)]
        phases, sessions = set(), set()
        for end in ("2026-01-14 05:00", "2026-01-14 10:30", "2026-01-14 18:30"):   # Asia / London / New York
            r = analyze_session_amd(_df(_bars(spec, 2000.0), end=end))
            phases.add((r["phase"], r["bias"]))
            sessions.add(r["session"]["active"])
        self.assertEqual(len(phases), 1, phases)
        self.assertGreaterEqual(len(sessions), 2)

    def test_wib_index_hint_is_converted(self):
        spec = _lead()
        df_utc = _df(_bars(spec, 2000.0), end="2026-01-14 10:00", tz="UTC")
        df_wib = _df(_bars(spec, 2000.0), end="2026-01-14 17:00", tz="WIB")
        self.assertEqual(analyze_session_amd(df_utc)["session"]["as_of_utc"],
                         analyze_session_amd(df_wib)["session"]["as_of_utc"])

    def test_no_lookahead(self):
        spec = _lead() + [("bar", 2000.0, 2000.5, 1992.5, 1999.5), ("move", 12, 2018.0, 0.3)]
        full = _df(_bars(spec, 2000.0))
        cut = len(full) - 13                                    # sebelum bar sweep
        trunc = full.iloc[:cut]
        as_of = trunc.index[-1].to_pydatetime() + timedelta(minutes=5)
        r_full_asof = analyze_session_amd(full, as_of=as_of)
        r_trunc = analyze_session_amd(trunc)
        self.assertEqual(r_full_asof["phase"], r_trunc["phase"])
        self.assertEqual(r_trunc["phase"], PHASE_ACCUMULATION)

    def test_forming_candle_is_ignored(self):
        spec = _lead() + [("bar", 2000.0, 2000.5, 1992.5, 1999.5)]
        full = _df(_bars(spec, 2000.0))
        as_of = full.index[-1].to_pydatetime()                  # bar terakhir masih berjalan
        r = analyze_session_amd(full, as_of=as_of)
        self.assertEqual(r["phase"], PHASE_ACCUMULATION, r)

    def test_engine_never_raises_on_garbage(self):
        bad = pd.DataFrame({"Open": [1, 2], "High": [1, 2], "Low": [1, 2], "Close": [1, 2]})
        self.assertFalse(analyze_session_amd(bad)["available"])
        self.assertFalse(analyze_session_amd(pd.DataFrame({"x": [1]}))["available"])


class AlignmentTests(unittest.TestCase):
    def _amd(self, phase, bias, **kw):
        base = {"available": True, "phase": phase, "bias": bias, "reference": "Asia Low", "atr": 4.0,
                "levels": [], "box": None}
        base.update(kw)
        return base

    def test_unavailable_never_holds(self):
        r = evaluate_zone_amd_alignment({"available": False}, "DEMAND", 1990, 1992)
        self.assertFalse(r["hold_early_pending"])
        self.assertFalse(evaluate_zone_amd_alignment(None, "DEMAND", 1990, 1992)["hold_early_pending"])

    def test_aligned_post_sweep_not_held(self):
        r = evaluate_zone_amd_alignment(self._amd(PHASE_MANIPULATION, "BULLISH"), "DEMAND", 1990, 1992)
        self.assertEqual(r["relation"], amd.REL_ALIGNED_POST_SWEEP)
        self.assertFalse(r["hold_early_pending"])

    def test_counter_expansion_held_with_reason_code(self):
        r = evaluate_zone_amd_alignment(self._amd(PHASE_DIRECT_EXPANSION, "BEARISH"), "DEMAND", 1990, 1992)
        self.assertTrue(r["hold_early_pending"])
        self.assertEqual(r["reason_code"], amd.REASON_AMD_COUNTER_PHASE)
        r2 = evaluate_zone_amd_alignment(self._amd(PHASE_DIRECT_EXPANSION, "BEARISH"), "SUPPLY", 2010, 2012)
        self.assertFalse(r2["hold_early_pending"])

    def test_liquidity_exposed_above_accumulation_low(self):
        a = self._amd(PHASE_ACCUMULATION, "NEUTRAL", box={"lo": 1990.0, "hi": 2010.0})
        r = evaluate_zone_amd_alignment(a, "DEMAND", 1991.5, 1993.0)
        self.assertTrue(r["hold_early_pending"])
        self.assertEqual(r["reason_code"], amd.REASON_AMD_LIQUIDITY_EXPOSED)
        far = evaluate_zone_amd_alignment(a, "DEMAND", 1998.0, 2000.0)         # jauh dari likuiditas -> bebas
        self.assertFalse(far["hold_early_pending"])
        sup = evaluate_zone_amd_alignment(a, "SUPPLY", 2007.0, 2008.5)
        self.assertTrue(sup["hold_early_pending"])

    def test_unswept_session_level_exposes_zone_but_swept_does_not(self):
        lv = [{"name": "Asia Low", "price": 1990.0, "status": "UNTOUCHED", "kind": "L"}]
        a = self._amd(PHASE_NO_CLEAR, "NEUTRAL", levels=lv)
        self.assertTrue(evaluate_zone_amd_alignment(a, "DEMAND", 1991.0, 1992.5)["hold_early_pending"])
        lv[0]["status"] = "SWEPT"
        self.assertFalse(evaluate_zone_amd_alignment(a, "DEMAND", 1991.0, 1992.5)["hold_early_pending"])

    def test_zone_below_liquidity_is_not_exposed(self):
        lv = [{"name": "Asia Low", "price": 1990.0, "status": "UNTOUCHED", "kind": "L"}]
        a = self._amd(PHASE_NO_CLEAR, "NEUTRAL", levels=lv)
        self.assertFalse(evaluate_zone_amd_alignment(a, "DEMAND", 1986.0, 1988.0)["hold_early_pending"])


class EntryEngineIntegrationTests(unittest.TestCase):
    def _run(self, hold):
        from modules import adaptive_entry_engine as ae
        from tests.test_adaptive_auto_trading import _build_m5_df_with_bearish_rejection, _make_supply_zone
        df5 = _build_m5_df_with_bearish_rejection()
        zone = _make_supply_zone(score=80.0, timeframe="H1", lo=3665.0, hi=3669.0)    # belum tersentuh (harga ~3644)
        with mock.patch.object(ae, "_amd_early_pending_hold", return_value=hold):
            return ae.evaluate_entry_paths_for_zone(
                zone, {"M5": df5}, current_price=3644.2, bid=3644.1, ask=3644.3, spread=0.2,
                htf_bias="BEARISH", h1_bias="BEARISH", m30_status="CONFIRMED", cfg=ae.get_auto_trading_config())

    def test_hold_turns_early_pending_into_adaptive_wait_with_reason(self):
        hold = {"reason_code": amd.REASON_AMD_COUNTER_PHASE, "reason": "ditahan", "relation": amd.REL_COUNTER_PHASE}
        r = self._run(hold)
        self.assertEqual(r["entry_mode"], "ADAPTIVE_ENTRY")
        self.assertEqual(r["reason_code"], amd.REASON_AMD_COUNTER_PHASE)
        self.assertFalse(r["trigger_passed"])
        self.assertEqual(r["amd_guard"], hold)

    def test_no_hold_keeps_previous_behaviour(self):
        r = self._run(None)
        self.assertNotIn(r["reason_code"], (amd.REASON_AMD_COUNTER_PHASE, amd.REASON_AMD_LIQUIDITY_EXPOSED))

    def test_guard_helper_is_fail_safe(self):
        from modules import adaptive_entry_engine as ae
        from tests.test_adaptive_auto_trading import _make_supply_zone
        zone = _make_supply_zone()
        with mock.patch.object(amd, "analyze_session_amd", side_effect=RuntimeError("boom")):
            self.assertIsNone(ae._amd_early_pending_hold(zone, None, {}))
        self.assertIsNone(ae._amd_early_pending_hold(zone, None, {"AMD_ENGINE_ENABLED": False}))
        self.assertIsNone(ae._amd_early_pending_hold(zone, None, {}))     # data kosong => tidak memblokir


if __name__ == "__main__":
    unittest.main()


class UserLiveEnvFullAutoTests(unittest.TestCase):
    """Uji langsung memakai parameter aktual .env user (AUTO_LOOSE_MODE=true, MAX_LOT=0.03, TP 110 pips, dll.)
    dalam mode FULL_AUTO mock-MT5 tanpa menyentuh akun/order MT5 asli."""

    def test_user_env_loose_mode_holds_blind_entry_on_counter_amd_until_m5_trigger(self):
        import tempfile
        from modules.config import _build_auto_trading_config, _apply_loose_overrides
        from modules.execution_monitor import AutoExecutionMonitor, format_adaptive_execution_panel
        from modules.execution_journal import ExecutionJournal
        from modules.execution_state import ExecutionStateManager, STATE_ORDER_PLACED, STATE_POSITION_ACTIVE
        from tests.test_adaptive_auto_trading import MockMT5, _make_snapshot_and_scan, _make_supply_zone

        # Paksa baca konfigurasi persis seperti mode live user (AUTO_LOOSE_MODE=true)
        base_cfg = _build_auto_trading_config()
        base_cfg["SNR_ZONE_MIN_STABLE_SECONDS"] = 0.0
        old_env = amd.os.environ.get("AUTO_LOOSE_MODE") if hasattr(amd, "os") else None
        import os as _os
        old_env = _os.environ.get("AUTO_LOOSE_MODE")
        try:
            _os.environ["AUTO_LOOSE_MODE"] = "true"
            user_cfg = _apply_loose_overrides(dict(base_cfg))
        finally:
            if old_env is None:
                _os.environ.pop("AUTO_LOOSE_MODE", None)
            else:
                _os.environ["AUTO_LOOSE_MODE"] = old_env

        # Buat M5 250 bar: kompresi di sekitar 2000 lalu DIRECT_EXPANSION BULLISH kuat ke 2022 (melawan zona SUPPLY!)
        spec_exp = _lead(2000.0, 230) + [("move", 16, 2022.0, 0.3), ("osc", 4, 2022.5, 0.3)]
        df_exp = _df(_bars(spec_exp, 2000.0))
        amd_state = analyze_session_amd(df_exp)
        self.assertEqual(amd_state["phase"], PHASE_DIRECT_EXPANSION)
        self.assertEqual(amd_state["bias"], "BULLISH")

        with tempfile.TemporaryDirectory() as tmp:
            mt5_m = MockMT5(balance=30_000_000.0, equity=30_000_000.0, bid=2022.40, ask=2022.60)
            sm = ExecutionStateManager(state_file=_os.path.join(tmp, "st.json"), auto_load=False)
            sm.authorize_account(login=mt5_m.login, server=mt5_m.server, currency="IDC", account_mode="REAL", real_confirmed=True)
            sm.set_trading_mode("FULL_AUTO", auto_trading=True)
            jrn = ExecutionJournal(journal_path=_os.path.join(tmp, "jrn.csv"))
            mon = AutoExecutionMonitor(mt5_api=mt5_m, state_mgr=sm, journal=jrn, symbol="XAUUSDc", cfg=user_cfg, connect_fn=lambda: True)

            # 1) Zona SUPPLY 2026-2030 belum tersentuh saat ekspansi BULLISH kencang ->
            #    Tanpa AMD guard, AUTO_LOOSE_MODE akan pasang SELL LIMIT buta. Dengan AMD guard -> DITAHAN (0 order)!
            z_sup = _make_supply_zone(score=76.0, grade="B", timeframe="H1", lo=2026.0, hi=2030.0)
            z_sup.sl_recommended = 2031.0
            snap1, scan1, dec1 = _make_snapshot_and_scan(z_sup, df_exp, bid=2022.40, ask=2022.60)
            cyc1 = mon.run_once(snapshot=snap1, scan_result=scan1, decision=dec1, news_override={"blackout": False, "status": "READY", "age_seconds": 5.0})
            self.assertEqual(len(mt5_m.orders_get()), 0)
            self.assertEqual(len(mt5_m.positions), 0)
            self.assertEqual(cyc1["best_evaluation"]["reason_code"], amd.REASON_AMD_COUNTER_PHASE)
            panel_txt = format_adaptive_execution_panel(cyc1, state_mgr=sm)
            self.assertIn("SESSION/AMD", panel_txt)
            self.assertIn("DIRECT_EXPANSION", panel_txt)

            # 2) Harga masuk ke tengah zona SUPPLY (2028.50) TANPA trigger candle M5 ->
            #    Meski AUTO_LOOSE_MODE=true, entry market buta tetap DITAHAN sampai ada konfirmasi M5!
            mt5_m.bid, mt5_m.ask = 2028.50, 2028.70
            snap2, scan2, dec2 = _make_snapshot_and_scan(z_sup, df_exp, bid=2028.50, ask=2028.70)
            cyc2 = mon.run_once(snapshot=snap2, scan_result=scan2, decision=dec2, news_override={"blackout": False, "status": "READY", "age_seconds": 5.0})
            self.assertEqual(len(mt5_m.positions), 0)
            self.assertEqual(cyc2["best_evaluation"]["reason_code"], amd.REASON_AMD_COUNTER_PHASE)

            # 3) Begitu terbentuk 1 candle Bearish Pinbar Rejection valid di dalam zona (M5 close) ->
            #    ADAPTIVE_ENTRY langsung mengeksekusi SELL MARKET di FULL_AUTO sesuai parameter user (lot 0.01, TP 110 pips)!
            spec_rej = spec_exp + [("bar", 2027.20, 2029.80, 2026.60, 2026.80)]
            df_rej = _df(_bars(spec_rej, 2000.0))
            mt5_m.bid, mt5_m.ask = 2026.80, 2027.00
            snap3, scan3, dec3 = _make_snapshot_and_scan(z_sup, df_rej, bid=2026.80, ask=2027.00)
            dec3["plan"]["sl"] = 2031.50
            cyc3 = mon.run_once(snapshot=snap3, scan_result=scan3, decision=dec3, news_override={"blackout": False, "status": "READY", "age_seconds": 5.0})
            self.assertTrue(cyc3["execution_outcome"]["executed"], cyc3["best_evaluation"])
            self.assertEqual(len(mt5_m.positions), 1)
            self.assertIn(cyc3["best_evaluation"]["state"], (STATE_POSITION_ACTIVE, STATE_ORDER_PLACED))

    def test_pantau_dashboard_and_report_auto_display_session_amd(self):
        from modules.smart_monitor_report import (
            format_smart_monitor_dashboard_with_meta,
            format_smart_monitor_detail,
            format_smart_monitor_report,
        )
        from tests.test_adaptive_auto_trading import _make_snapshot_and_scan, _make_supply_zone

        spec_m = _lead(2000.0, 230) + [("bar", 2000.0, 2000.5, 1990.0, 1999.5), ("move", 12, 2018.0, 0.3)]
        df_m = _df(_bars(spec_m, 2000.0))
        z = _make_supply_zone(score=82.0, timeframe="H1", lo=2025.0, hi=2029.0)
        snap, scan, dec = _make_snapshot_and_scan(z, df_m, bid=2018.0, ask=2018.2)
        cyc = {
            "cycle": 1,
            "mode": "unified",
            "snapshot": snap,
            "snapshot_time": "2026-01-14 21:00:00 WIB",
            "smc": {"scan_result": scan, "decision": dec, "summary": {"h4_bias": "BEARISH", "h1_bias": "BEARISH", "m30_bias": "CONFIRMED"}},
            "snr": {},
            "account": {},
        }
        dash_txt, _ = format_smart_monitor_dashboard_with_meta(cyc)
        rep_txt = format_smart_monitor_report(cyc)
        det_txt = format_smart_monitor_detail(cyc)
        self.assertIn("SESSION/AMD", dash_txt)
        self.assertIn("DISTRIBUTION", dash_txt)
        self.assertIn("SESSION/AMD", rep_txt)
        self.assertIn("POWER OF THREE / AMD", det_txt)
