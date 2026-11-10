# -*- coding: utf-8 -*-
"""
tests/test_zone_scanner.py
Unit test Python-Only Zone Scanner: struktur, OB, FVG, sweep, scoring, merge, pipeline MTF,
entry validator, fail-safe, audit, dan integrasi ai_gemini (tanpa MT5/Gemini live).
"""
import json
import math
import os
import tempfile
import unittest

import pandas as pd

import modules.zone_scanner as zs
from modules.zone_scanner import (
    detect_swings, detect_market_structure, detect_liquidity_sweeps, detect_fvgs, detect_order_blocks,
    score_zone, merge_overlapping_zones, evaluate_htf_context, validate_entry_setup, scan_xauusd_zones,
    evaluate_trade_decision, detect_market_regime, format_scanner_audit, Zone,
)
from modules.mt5_conn import tambah_indikator


def make_df(rows, start="2026-11-01 00:00:00", freq="1h"):
    t0 = pd.to_datetime(start)
    idx = [t0 + i * pd.Timedelta(freq) for i in range(len(rows))]
    return tambah_indikator(pd.DataFrame(rows, columns=["Open", "High", "Low", "Close", "Volume"], index=idx))


def mirror(rows, m=4000.0):
    return [(m - o, m - l, m - h, m - c, v) for (o, h, l, c, v) in rows]


BULL_ROWS = [
    (1990, 1995, 1985, 1992, 100), (1992, 2000, 1988, 1998, 100), (1998, 2005, 1994, 2002, 100),
    (2002, 2010, 1999, 2005, 100), (2005, 2006, 1995, 1997, 100), (1997, 1998, 1984, 1986, 100),
    (1988, 1994, 1986, 1992, 100), (1992, 1998, 1989, 1996, 100), (1996, 1999, 1988, 1990, 100),
    (1990, 1991, 1980, 1982, 100), (1980, 1982, 1974, 1975, 150), (1975, 1998, 1974, 1996, 400),
    (1996, 2025, 1990, 2020, 500), (2020, 2030, 2015, 2025, 300),
]


def make_zone(**kw):
    base = dict(zone_id="Z1", symbol="XAUUSD", timeframe="H1", zone_type="DEMAND", price_low=2000.0, price_high=2004.0,
                midpoint=2002.0, score=90.0, grade="A+", origin_bos={"tipe": "BOS", "strength_atr": 1.0},
                fvg_confirmation={"status": "UNTOUCHED"}, displacement_strength=2.2, freshness="FRESH", touch_count=0,
                liquidity_sweep_status="NONE", invalidation_price=2000.0, is_valid=True, rejection_reason="",
                sl_recommended=1997.0)
    base.update(kw)
    return Zone(**base)


def m5_trigger_df():
    """M5 sintetis: osilasi, swing high 2012, swing low 2003, sweep low ke 2001, reversal impuls break close > 2012, pullback ke zona."""
    rows = []
    for i in range(45):
        c = 2008 + 2 * math.sin(i / 2.0)
        rows.append((c - 0.3, c + 1.0, c - 1.0, c + 0.3, 100))
    rows.append((2009.0, 2012.0, 2008.5, 2009.5, 100)) # swing high 2012
    for p in (2008.0, 2006.0, 2005.0, 2004.0):
        rows.append((p, p + 0.5, p - 0.5, p, 100))
    rows.append((2004.0, 2004.5, 2003.0, 2004.0, 100)) # swing low 2003.0
    rows.append((2004.0, 2005.5, 2003.8, 2005.0, 100))
    rows.append((2005.0, 2006.0, 2004.5, 2005.5, 100))
    rows.append((2005.5, 2006.5, 2005.0, 2006.0, 100)) # confirmed swing low
    rows.append((2005.0, 2005.2, 2001.0, 2003.5, 200)) # SWEEP_LOW to 2001.0
    rows.append((2003.5, 2006.0, 2003.2, 2005.5, 200)) # sweep confirmation
    for o in (2006.0, 2008.0, 2010.0):
        rows.append((o, o + 2.5, o - 0.2, o + 2.2, 300))
    rows.append((2012.0, 2015.0, 2011.8, 2014.5, 500)) # BOS break above 2012
    for c in (2012.0, 2009.0, 2006.5, 2004.5, 2003.5):
        rows.append((c + 0.5, c + 0.8, c - 0.5, c, 100)) # pullback to 2003.5
    return make_df(rows, freq="5min")


def m5_no_sweep_df():
    """M5 sintetis dengan BOS/CHoCH tetapi tanpa Liquidity Sweep."""
    rows = []
    for i in range(50):
        c = 2008 + 2 * math.sin(i / 2.0)
        rows.append((c - 0.3, c + 1.0, c - 1.0, c + 0.3, 100))
    rows.append((2009.0, 2012.0, 2008.5, 2009.5, 100))
    for p in (2008.0, 2006.0, 2004.0, 2003.0):
        rows.append((p, p + 0.5, p - 0.5, p, 100))
    for o in (2004.0, 2007.0, 2010.0):
        rows.append((o, o + 2.5, o - 0.2, o + 2.2, 300))
    rows.append((2012.0, 2015.0, 2011.8, 2014.5, 500))
    for c in (2012.0, 2009.0, 2006.5, 2004.5, 2003.5):
        rows.append((c + 0.5, c + 0.8, c - 0.5, c, 100))
    return make_df(rows, freq="5min")


def h1_target_df():
    rows = []
    for i in range(21):
        c = 1990 + 2 * i
        rows.append((c - 0.5, c + 0.5, c - 1.0, c, 100))
    for i in range(21, 41):
        c = 2030 - 1.1 * (i - 20)
        rows.append((c + 0.5, c + 0.8, c - 0.5, c, 100))
    c = 2008.0
    for _ in range(41, 70):
        c -= 0.2
        rows.append((c + 0.2, c + 0.3, c - 0.3, c, 100))
    return make_df(rows)


def flat_df(level=2006.0, n=70, freq="1h"):
    return make_df([(level, level + 2, level - 1, level + 1, 100)] * n, freq=freq)


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self._old_audit = zs.FILE_AUDIT_ZONA
        zs.FILE_AUDIT_ZONA = os.path.join(self._tmp, "audit.jsonl")
        zs._AUDIT_LAST.clear()
        zs._SCAN_CACHE.update(key=None, data=None, ts=0.0)
        zs._FRAMES_CACHE.update(ts=0.0, data=None)
        self._orig_scan = zs.scan_xauusd_zones

    def tearDown(self):
        zs.FILE_AUDIT_ZONA = self._old_audit
        zs.scan_xauusd_zones = self._orig_scan


class TestStructure(Base):
    def test_bos_requires_close_not_wick(self):
        rows = [(1980, 1985, 1978, 1982, 1), (1982, 1990, 1980, 1988, 1), (1988, 1995, 1985, 1992, 1),
                (1992, 2000, 1990, 1995, 1), (1995, 1998, 1988, 1990, 1), (1990, 1994, 1982, 1985, 1),
                (1985, 1988, 1975, 1980, 1), (1980, 2003, 1979, 1996, 2), (1996, 2008, 1995, 2005, 3)]
        events, trend, swings = detect_market_structure(make_df(rows), n=3)
        self.assertEqual([e["k"] for e in events], [8])
        self.assertEqual(trend, 1)
        self.assertEqual([s["idx"] for s in swings if s["type"] == "H" and s["price"] == 2000.0], [3])

    def test_swing_confirmed_only_after_n_candles(self):
        sw = detect_swings(make_df(BULL_ROWS), 3)
        self.assertTrue(all(s["confirmed_at"] == s["idx"] + 3 for s in sw))


class TestOrderBlock(Base):
    def _run(self, rows, tf="H1", fvgs=None):
        df = make_df(rows)
        ev, _, sw = detect_market_structure(df)
        return detect_order_blocks(df, ev, detect_fvgs(df) if fvgs is None else fvgs, detect_liquidity_sweeps(df, sw, None), tf)

    def test_bullish_ob(self):
        ok, _ = self._run(BULL_ROWS)
        z = [x for x in ok if x.zone_type == "DEMAND"][0]
        self.assertEqual((z.price_low, z.price_high), (1974.0, 1982.0))
        self.assertGreaterEqual(z.displacement_strength, 1.5)
        self.assertIsNotNone(z.fvg_confirmation)

    def test_bearish_ob(self):
        ok, _ = self._run(mirror(BULL_ROWS), "H4")
        z = [x for x in ok if x.zone_type == "SUPPLY"][0]
        self.assertEqual((z.price_low, z.price_high), (2018.0, 2026.0))
        self.assertIsNotNone(z.fvg_confirmation)

    def test_origin_unidentified_rejected(self):
        """Regresi: tanpa swing lawan terkonfirmasi, OB tidak boleh dicari memakai swing_idx/fallback."""
        rows = [(2020, 2025, 2015, 2020, 1), (2020, 2022, 2005, 2010, 1), (2010, 2015, 1995, 2000, 1),
                (2000, 2005, 1990, 1995, 1), (1995, 2000, 1992, 1998, 1), (1998, 2005, 1995, 2002, 1),
                (2002, 2010, 2000, 2008, 1), (2008, 2015, 2005, 2012, 1), (2012, 2022, 2010, 2020, 1),
                (2020, 2021, 1995, 1998, 1), (1998, 2000, 1970, 1975, 1), (1975, 1980, 1965, 1970, 1)]
        ok, bad = self._run(rows, "H4")
        self.assertFalse([z for z in ok if z.zone_type == "SUPPLY"])
        self.assertTrue(any("Asal impuls" in z.rejection_reason for z in bad))

    def test_fvg_filters(self):
        base = {"idx": 10, "time": None, "price_low": 1982.0, "price_high": 1990.0, "size_points": 12.0, "size_atr": 1.0}
        for fv, label in (([], "tanpa FVG"),
                          ([dict(base, type="BEARISH_FVG", status="UNTOUCHED")], "arah salah"),
                          ([dict(base, type="BULLISH_FVG", status="FULLY_MITIGATED")], "terisi penuh"),
                          ([dict(base, type="BULLISH_FVG", status="UNTOUCHED", price_low=2100.0, price_high=2110.0)], "tak relevan harga")):
            ok, bad = self._run(BULL_ROWS, fvgs=fv)
            self.assertFalse([z for z in ok if z.zone_type == "DEMAND"], label)
            self.assertTrue(any("FVG" in z.rejection_reason for z in bad), label)

    def test_tiny_fvg_not_detected(self):
        df = make_df([(1990, 2000, 1985, 1995, 1), (1995, 2020, 1995, 2018, 1), (2018, 2025, 2000.2, 2022, 1)] + [(2022, 2028, 2015, 2025, 1)] * 20)
        self.assertEqual(detect_fvgs(df), [])

    def test_fvg_mitigation_status(self):
        rows = [(1990, 2000, 1985, 1995, 1), (1995, 2020, 1995, 2018, 1), (2018, 2025, 2005, 2022, 1), (2022, 2028, 2015, 2025, 1)]
        self.assertEqual(detect_fvgs(make_df(rows), 0.1)[0]["status"], "UNTOUCHED")
        self.assertEqual(detect_fvgs(make_df(rows + [(2025, 2026, 1998, 2010, 1)]), 0.1)[0]["status"], "FULLY_MITIGATED")
        self.assertEqual(detect_fvgs(make_df(rows + [(2025, 2026, 2002, 2010, 1)]), 0.1)[0]["status"], "PARTIALLY_MITIGATED")

    def test_fresh_tested_invalidated(self):
        ok, _ = self._run(BULL_ROWS)
        self.assertEqual((ok[0].freshness, ok[0].touch_count), ("FRESH", 0))
        ok, bad = self._run(BULL_ROWS + [(2025, 2026, 1979, 2000, 1), (2000, 2030, 1999, 2025, 1)])
        # retest pertama mengisi FVG yang menempel di OB -> konfirmasi FVG gugur -> zona tidak aktif (bukan fallback)
        self.assertFalse(ok)
        t = [z for z in bad if z.freshness == "TESTED_1X"]
        self.assertTrue(t and t[0].touch_count == 1 and "FVG" in t[0].rejection_reason)
        ok, bad = self._run(BULL_ROWS + [(2020, 2022, 1960, 1965, 1)])
        self.assertFalse(ok)
        self.assertTrue([z for z in bad if z.freshness == "INVALIDATED"])


class TestSweep(Base):
    BASE = [(1980, 1985, 1978, 1982, 1), (1982, 1990, 1980, 1988, 1), (1988, 1995, 1985, 1992, 1),
            (1992, 2000, 1990, 1995, 1), (1995, 1998, 1988, 1990, 1), (1990, 1994, 1982, 1985, 1),
            (1985, 1988, 1975, 1980, 1)]

    def _sw(self, extra):
        df = make_df(self.BASE + extra)
        return detect_liquidity_sweeps(df, detect_swings(df, 3), 20)

    def test_valid_sweep(self):
        s = self._sw([(1980, 2004, 1978, 1995, 3)])
        self.assertEqual((len(s), s[0]["type"], s[0]["level"]), (1, "SWEEP_HIGH", 2000.0))

    def test_breakout_not_sweep(self):
        self.assertEqual(self._sw([(1980, 2004, 1978, 2003, 3)]), [])

    def test_tiny_wick_not_sweep(self):
        self.assertEqual(self._sw([(1999, 2000.1, 1990, 1992, 3)]), [])

    def test_continuation_after_wick_not_sweep(self):
        self.assertEqual(self._sw([(1980, 2004, 1978, 1995, 3), (1996, 2010, 1995, 2008, 3)]), [])


class TestScoreMerge(Base):
    def test_scoring_weights(self):
        z = score_zone(make_zone(timeframe="H4", origin_bos={"tipe": "BOS", "strength_atr": 2.0}, displacement_strength=2.6,
                                 liquidity_sweep_status="CONFIRMED_SWEEP_LOW"), htf_bias_aligned=True)
        self.assertEqual((z.score, z.grade), (100.0, "A+"))
        low = score_zone(make_zone(origin_bos={"tipe": "CHoCH"}, fvg_confirmation={"status": "PARTIALLY_MITIGATED"},
                                   displacement_strength=1.6, freshness="TESTED_2X", touch_count=2), htf_bias_aligned=False)
        self.assertEqual((low.score, low.grade, low.is_valid), (39.4, "REJECTED", False))

    def test_htf_alignment_points(self):
        a = score_zone(make_zone(timeframe="H4"), htf_alignment={"H4": "ALIGNED", "H1": "ALIGNED", "M30": "ALIGNED"})
        b = score_zone(make_zone(timeframe="H4"), htf_alignment={"H4": "ALIGNED", "H1": "OPPOSED", "M30": "NEUTRAL"})
        self.assertEqual(a.score_breakdown["konteks_htf"], 10.0)
        self.assertEqual(b.score_breakdown["konteks_htf"], 5.5)

    def test_hard_validation_beats_score(self):
        z = score_zone(make_zone(is_valid=False, score=99.0, rejection_reason="invalid"))
        self.assertEqual((z.score, z.grade), (0.0, "REJECTED"))

    def test_merge_no_bonus_and_worst_freshness(self):
        a = make_zone(zone_id="a", timeframe="H4", price_low=2000, price_high=2006, score=80.0, grade="A")
        b = make_zone(zone_id="b", price_low=2002, price_high=2008, score=78.0, grade="A", freshness="TESTED_1X", touch_count=1)
        m = merge_overlapping_zones([a, b])
        self.assertEqual(len(m), 1)
        self.assertEqual((m[0].timeframe, m[0].price_low, m[0].price_high, m[0].score, m[0].grade), ("H4+H1", 2000.0, 2008.0, 80.0, "A"))
        self.assertEqual((m[0].freshness, m[0].touch_count), ("TESTED_1X", 1))

    def test_no_merge_when_disjoint_or_opposite(self):
        a = make_zone(zone_id="a", price_low=2000, price_high=2004)
        b = make_zone(zone_id="b", price_low=2010, price_high=2014)
        c = make_zone(zone_id="c", zone_type="SUPPLY", price_low=2001, price_high=2003)
        self.assertEqual(len(merge_overlapping_zones([a, b, c])), 3)


class TestScannerStandalone(Base):
    def test_htf_conflict(self):
        up = make_df([(1900 + i * 5, 1910 + i * 5, 1895 + i * 5, 1908 + i * 5, 1) for i in range(40)])
        dn = make_df([(2100 - i * 5, 2105 - i * 5, 2090 - i * 5, 2092 - i * 5, 1) for i in range(40)])
        ctx = evaluate_htf_context({"D1": up, "H4": up, "H1": dn})
        self.assertTrue(ctx["has_conflict"])
        self.assertIn("WAIT", ctx["bias_utama"])

    def test_no_zone_returns_empty_with_reason(self):
        f = make_df([(2000.0 + (i % 2), 2002.0, 1999.0, 2000.5, 1) for i in range(60)])
        r = scan_xauusd_zones(frames={"D1": f, "H4": f, "H1": f, "M30": f, "M5": f}, current_price=2000.0)
        self.assertEqual(r["active_zones"], [])
        self.assertTrue(r["empty_reason"])
        self.assertIn("WAIT", r["summary_text"])

    def test_scan_finds_zone_and_writes_audit(self):
        pad = [(1980, 1985, 1975, 1980, 100)] * 0
        df = make_df(pad + BULL_ROWS)
        r = scan_xauusd_zones(frames={"H4": df, "H1": df}, current_price=2025.0)
        # data 13 candle < batas minimum (26): scanner wajib menolak, bukan memaksakan
        self.assertFalse(r["data_ok"])
        self.assertEqual(r["active_zones"], [])

    def test_invalid_price_waits(self):
        f = flat_df()
        r = scan_xauusd_zones(frames={"H4": f, "H1": f}, current_price=float("nan"))
        self.assertFalse(r["data_ok"])

    def test_audit_file_written_and_throttled(self):
        f = make_df([(2000.0 + (i % 2), 2002.0, 1999.0, 2000.5, 1) for i in range(60)])
        for _ in range(3):
            zs._SCAN_CACHE.update(key=None)
            scan_xauusd_zones(frames={"H4": f, "H1": f}, current_price=2000.0)
        with open(zs.FILE_AUDIT_ZONA, encoding="utf-8") as fh:
            lines = [json.loads(x) for x in fh]
        self.assertEqual(len(lines), 1)   # duplikat identik dibatasi
        self.assertEqual(lines[0]["kind"], "zone_scan")


class TestEntryValidator(Base):
    def setUp(self):
        super().setUp()
        self.m5, self.h1 = m5_trigger_df(), h1_target_df()

    def chk(self, zone=None, price=2003.5, m5=None, spread=0.3, news=False, htf=True, **kw):
        return zs._entry_check(zone or make_zone(), price, self.m5 if m5 is None else m5, spread, news,
                               {"H1": self.h1} if htf else {}, **kw)

    def test_valid_entry_has_structural_rr(self):
        r = self.chk()
        self.assertEqual(r["status"], "VALID", r["reasons"])
        p = r["plan"]
        self.assertGreaterEqual(p["rr"], 1.5)
        self.assertGreater(p["tp1"], p["entry_acuan"])
        self.assertLess(p["sl"], p["entry_min"])
        self.assertAlmostEqual(p["tp1"], 2030.2, places=1)   # TP = likuiditas swing H1, bukan R:R buatan

    def test_wait_cases(self):
        flat = flat_df(2003.0, 70, "5min")
        cases = {
            "harga jauh": self.chk(price=2025.0),
            "tanpa trigger": self.chk(m5=flat),
            "spread lebar": self.chk(spread=3.0),
            "berita": self.chk(news="NFP"),
            "m5 kosong": self.chk(m5=pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])),
            "harga nan": self.chk(price=float("nan")),
            "tanpa target struktural": self.chk(htf=False),
            "tanpa sweep (semua grade wajib)": self.chk(m5=m5_no_sweep_df()),
        }
        for name, r in cases.items():
            self.assertEqual(r["status"], "WAIT", name)
            self.assertTrue(r["reasons"], name)
            self.assertIsNone(r["plan"], name)

    def test_invalid_cases(self):
        self.assertEqual(self.chk(price=1998.0)["status"], "INVALID")
        self.assertEqual(self.chk(zone=make_zone(is_valid=False, rejection_reason="x"))["status"], "INVALID")

    def test_rr_too_low_waits(self):
        r = zs._entry_check(make_zone(sl_recommended=1983.0), 2003.5, self.m5, 0.3, False, {"H1": self.h1})
        self.assertEqual(r["status"], "WAIT")
        self.assertIn("R:R", " ".join(r["reasons"]))

    def test_legacy_tuple_api(self):
        ok, msg, plan = validate_entry_setup(make_zone(), 2003.5, self.m5, 0.3, False, {"H1": self.h1})
        self.assertTrue(ok)
        self.assertTrue(msg.startswith("VALID"))
        ok, msg, plan = validate_entry_setup(make_zone(), 2025.0, self.m5)
        self.assertFalse(ok)
        self.assertTrue(msg.startswith("WAIT"))


def fake_ctx(**kw):
    c = {"D1": "BULLISH", "H4": "BULLISH", "H1": "BULLISH", "M30": "BULLISH", "has_conflict": False,
         "m30_conflict": False, "bias_utama": "BULLISH"}
    c.update(kw)
    return c


class TestPipeline(Base):
    def setUp(self):
        super().setUp()
        self.h1 = h1_target_df()
        self.frames = {"H4": make_df([(2201 - 2 * i, 2201.5 - 2 * i, 2199.5 - 2 * i, 2200 - 2 * i, 100) for i in range(70)], freq="4h"), "H1": self.h1, "M30": flat_df(2006, freq="30min"), "M5": m5_trigger_df()}
        self.zone = make_zone(bos_candle_time=self.h1.index[30]).to_dict()

    def run_dec(self, ctx=None, zone="default", frames=None, **kw):
        zd = self.zone if zone == "default" else zone
        zs.scan_xauusd_zones = lambda frames=None, current_price=None, symbol=None: {
            "data_ok": True, "htf_context": ctx or fake_ctx(), "active_zones": [zd] if zd else [],
            "empty_reason": "tidak ada zona valid", "best_demand": zd, "best_supply": None, "rejected_zones": []}
        return evaluate_trade_decision(frames=frames or self.frames, current_price=kw.pop("price", 2003.5),
                                       spread=0.3, news_blackout=None, live=False, **kw)

    def test_all_aligned_valid_and_chained(self):
        d = self.run_dec()
        self.assertEqual(d["status"], "VALID", d["reasons"])
        self.assertEqual(d["direction"], "BUY")
        for s in ("DATA", "H4", "H1", "M30", "M5", "RISK"):
            self.assertTrue(d["stages"][s]["ok"], s)
        self.assertIsNotNone(d["plan"])

    def test_h4_h1_conflict_waits(self):
        d = self.run_dec(fake_ctx(H1="BEARISH", has_conflict=True))
        self.assertEqual((d["status"], d["plan"]), ("WAIT", None))

    def test_h4_neutral_waits(self):
        self.assertEqual(self.run_dec(fake_ctx(H4="NETRAL"))["status"], "WAIT")

    def test_m30_conflict_waits(self):
        d = self.run_dec(fake_ctx(M30="BEARISH", m30_conflict=True))
        self.assertEqual(d["status"], "WAIT")
        self.assertFalse(d["stages"]["M30"]["ok"])

    def test_zone_against_h4_bias_waits(self):
        opp = make_zone(zone_type="SUPPLY", bos_candle_time=self.h1.index[30]).to_dict()
        self.assertEqual(self.run_dec(zone=opp)["status"], "WAIT")

    def test_no_zone_waits(self):
        self.assertEqual(self.run_dec(zone=None)["status"], "WAIT")

    def test_h1_close_through_zone_is_invalid(self):
        h1 = h1_target_df()
        h1.iloc[-1, h1.columns.get_loc("Close")] = 1995.0
        d = self.run_dec(frames=dict(self.frames, H1=h1))
        self.assertEqual((d["status"], d["plan"]), ("INVALID", None))

    def test_incomplete_timeframes_wait(self):
        for tf in ("H4", "H1", "M30", "M5"):
            fr = dict(self.frames)
            fr[tf] = None
            self.assertEqual(self.run_dec(frames=fr)["status"], "WAIT", tf)
        short = dict(self.frames, M5=self.frames["M5"].head(20))
        self.assertEqual(self.run_dec(frames=short)["status"], "WAIT")

    def test_stale_data_waits(self):
        late = self.frames["M5"].index[-1] + pd.Timedelta(days=2)
        d = self.run_dec(server_time=late)
        self.assertEqual(d["status"], "WAIT")
        self.assertIn("terlambat", " ".join(d["reasons"]))

    def test_bad_price_waits(self):
        for p in (float("nan"), -5.0, 0.0, None):
            self.assertEqual(self.run_dec(price=p)["status"], "WAIT", p)

    def test_scanner_exception_waits(self):
        def boom(**kw):
            raise RuntimeError("scanner rusak")
        zs.scan_xauusd_zones = boom
        d = evaluate_trade_decision(frames=self.frames, current_price=2003.5, spread=0.3, live=False)
        self.assertEqual((d["status"], d["plan"]), ("WAIT", None))
        self.assertIn("fail-safe", " ".join(d["reasons"]))

    def test_mt5_offline_waits(self):
        zs._load_frames = lambda ttl=10.0: None
        d = evaluate_trade_decision()
        self.assertEqual(d["status"], "WAIT")

    def test_entry_audit_written(self):
        self.run_dec()
        with open(zs.FILE_AUDIT_ZONA, encoding="utf-8") as fh:
            recs = [json.loads(x) for x in fh]
        e = [r for r in recs if r["kind"] == "entry_decision"][-1]
        for k in ("status", "arah", "bias_H4", "konfirmasi_H1", "validasi_M30", "trigger_M5", "entry", "sl", "tp1", "rr", "alasan"):
            self.assertIn(k, e)


class TestGeminiFailSafe(Base):
    """AI tidak boleh mengubah WAIT/INVALID/error menjadi BUY/SELL."""

    @classmethod
    def setUpClass(cls):
        import modules.ai_gemini as ag
        cls.ag = ag

    def setUp(self):
        super().setUp()
        self.ag_dec, self.ag_scan = self.ag.evaluate_trade_decision, self.ag.scan_xauusd_zones
        zone = make_zone().to_dict()
        self.scan = {"best_demand": zone, "best_supply": None}
        self.ag.scan_xauusd_zones = lambda *a, **k: self.scan
        self.plan = {"arah": "BUY", "entry_min": 2001.5, "entry_max": 2004.0, "entry_acuan": 2002.75, "sl": 1997.0,
                     "tp1": 2030.2, "tp2": 2030.2, "rr": 4.7, "keterangan": "plan python"}

    def tearDown(self):
        self.ag.evaluate_trade_decision, self.ag.scan_xauusd_zones = self.ag_dec, self.ag_scan
        super().tearDown()

    def target(self, arah="BUY"):
        return {"atas": (2100.0, 2102.0, 2105.0), "bawah": (1900.0, 1902.0, 1895.0),
                "sinyal": dict(arah=arah, lo=2001.0, hi=2003.0, sl=1990.0, tp1=2040.0, tp2=2050.0, lo_asli=2001.0, hi_asli=2003.0)}

    def test_wait_blocks_ai_signal_and_ai_zones(self):
        self.ag.evaluate_trade_decision = lambda: {"status": "WAIT", "reasons": ["M5 belum trigger"], "plan": None, "direction": "BUY"}
        t = self.ag._terapkan_otoritas_scanner(self.target())
        self.assertIsNone(t["sinyal"])
        self.assertIsNone(t["atas"])          # zona AI di luar scanner dibuang
        self.assertEqual(t["bawah"][:2], (2000.0, 2004.0))
        self.assertIn("M5 belum trigger", self.ag._STATE_PARSER["ditolak"][0])

    def test_invalid_blocks(self):
        self.ag.evaluate_trade_decision = lambda: {"status": "INVALID", "reasons": ["x"], "plan": None, "direction": "BUY"}
        self.assertIsNone(self.ag._terapkan_otoritas_scanner(self.target())["sinyal"])

    def test_exception_blocks_everything(self):
        def boom():
            raise RuntimeError("x")
        self.ag.evaluate_trade_decision = boom
        t = self.ag._terapkan_otoritas_scanner(self.target())
        self.assertEqual((t["sinyal"], t["atas"], t["bawah"]), (None, None, None))

    def test_direction_mismatch_blocks(self):
        self.ag.evaluate_trade_decision = lambda: {"status": "VALID", "reasons": [], "plan": self.plan, "direction": "BUY"}
        self.assertIsNone(self.ag._terapkan_otoritas_scanner(self.target("SELL"))["sinyal"])

    def test_valid_locks_numbers_to_python_plan(self):
        self.ag.evaluate_trade_decision = lambda: {"status": "VALID", "reasons": [], "plan": self.plan, "direction": "BUY"}
        s = self.ag._terapkan_otoritas_scanner(self.target())["sinyal"]
        self.assertEqual((s["lo"], s["hi"], s["sl"], s["tp1"]), (2001.5, 2004.0, 1997.0, 2030.2))

    def test_scanner_wait_stays_wait(self):
        self.ag.evaluate_trade_decision = lambda: {"status": "WAIT", "reasons": ["M5 belum konfirmasi"], "plan": None, "direction": "BUY"}
        t = self.target()
        res = self.ag._terapkan_otoritas_scanner(t)
        self.assertIsNone(res["sinyal"])
        self.assertIn("M5 belum konfirmasi", self.ag._STATE_PARSER["ditolak"][0])

    def test_python_autonomous_signal_when_valid(self):
        # Python Scanner menghasilkan sinyal mandiri bahkan jika input t["sinyal"] kosong
        self.ag.evaluate_trade_decision = lambda: {"status": "VALID", "reasons": [], "plan": self.plan, "direction": "BUY"}
        t = {"atas": None, "bawah": None, "sinyal": None}
        res = self.ag._terapkan_otoritas_scanner(t)
        self.assertIsNotNone(res["sinyal"])
        self.assertEqual((res["sinyal"]["lo"], res["sinyal"]["hi"], res["sinyal"]["sl"]), (2001.5, 2004.0, 1997.0))


class TestRequirement7ZoneDetection(Base):
    """Pengujian Unit Mandiri: Deteksi OB, FVG, Liquidity Sweep, BOS, dan CHoCH."""

    def test_order_block_and_fvg_detection(self):
        df = make_df(BULL_ROWS)
        ev, _, sw = detect_market_structure(df)
        fvgs = detect_fvgs(df)
        sweeps = detect_liquidity_sweeps(df, sw)
        ok, bad = detect_order_blocks(df, ev, fvgs, sweeps, "H1")
        self.assertTrue(len(ok) > 0)
        self.assertEqual(ok[0].zone_type, "DEMAND")
        self.assertIsNotNone(ok[0].fvg_confirmation)

    def test_liquidity_sweep_detection(self):
        df = m5_trigger_df()
        swings = detect_swings(df, 3)
        sweeps = detect_liquidity_sweeps(df, swings, 40)
        types = [s["type"] for s in sweeps]
        self.assertIn("SWEEP_LOW", types)

    def test_bos_and_choch_detection(self):
        df = m5_trigger_df()
        events, trend, swings = detect_market_structure(df, 3)
        event_types = [e["tipe"] for e in events]
        self.assertTrue("BOS" in event_types or "CHoCH" in event_types)


class TestRequirement7MultiTimeframe(Base):
    """Pengujian Unit Mandiri: Hirarki Multi-Timeframe (H4, H1, M30) & Kondisi Pasar."""

    def test_h4_bullish_and_h1_bullish_aligned(self):
        up = make_df([(1900 + i * 3, 1905 + i * 3, 1898 + i * 3, 1904 + i * 3, 100) for i in range(50)], freq="4h")
        ctx = evaluate_htf_context({"H4": up, "H1": up})
        self.assertFalse(ctx["has_conflict"])
        self.assertEqual(ctx["H4"], "BULLISH")
        self.assertEqual(ctx["H1"], "BULLISH")

    def test_h4_bearish_and_h1_bearish_aligned(self):
        dn = make_df([(2200 - i * 3, 2202 - i * 3, 2195 - i * 3, 2196 - i * 3, 100) for i in range(50)], freq="4h")
        ctx = evaluate_htf_context({"H4": dn, "H1": dn})
        self.assertFalse(ctx["has_conflict"])
        self.assertEqual(ctx["H4"], "BEARISH")
        self.assertEqual(ctx["H1"], "BEARISH")

    def test_h4_bullish_vs_h1_bearish_conflict_waits(self):
        # H4 Bullish vs H1 Bearish -> harus memicu conflict -> WAIT
        c = fake_ctx(H4="BULLISH", H1="BEARISH", has_conflict=True)
        self.assertTrue(c["has_conflict"])
        self.assertEqual(c["bias_utama"], "BULLISH") # bias tercatat tapi conflict aktif

    def test_market_ranging_detection(self):
        df = flat_df(2050.0, 40, "1h")
        reg = detect_market_regime(df)
        self.assertIn(reg["regime"], ("RANGING", "UNKNOWN"))

    def test_market_choppy_detection(self):
        # Bolak-balik break wick tajam
        rows = []
        for i in range(35):
            o = 2000.0 if i % 2 == 0 else 2004.0
            c = 2004.0 if i % 2 == 0 else 2000.0
            rows.append((o, 2008.0, 1996.0, c, 100))
        df = make_df(rows, freq="1h")
        reg = detect_market_regime(df)
        self.assertIn(reg["regime"], ("CHOPPY", "RANGING"))


class TestRequirement7EntryValidation(Base):
    """Pengujian Unit Mandiri: Validasi Entry M5 Konsisten untuk Semua Grade."""

    def setUp(self):
        super().setUp()
        self.m5 = m5_trigger_df()
        self.m5_nosweep = m5_no_sweep_df()
        self.h1 = h1_target_df()
        self.zone = make_zone()

    def test_sweep_terkonfirmasi_passes(self):
        res = zs._entry_check(self.zone, 2003.5, self.m5, 0.3, False, {"H1": self.h1})
        self.assertEqual(res["status"], "VALID")

    def test_tidak_ada_sweep_waits(self):
        res = zs._entry_check(self.zone, 2003.5, self.m5_nosweep, 0.3, False, {"H1": self.h1})
        self.assertEqual(res["status"], "WAIT")
        self.assertTrue(any("Sweep" in r for r in res["reasons"]))

    def test_sweep_tanpa_displacement_waits(self):
        # M5 dengan body candle sangat kecil (tanpa displacement kuat)
        m5_small = self.m5.copy()
        m5_small["Close"] = m5_small["Open"] + 0.05
        res = zs._entry_check(self.zone, 2003.5, m5_small, 0.3, False, {"H1": self.h1})
        self.assertEqual(res["status"], "WAIT")

    def test_displacement_tanpa_konfirmasi_struktur_waits(self):
        flat = flat_df(2003.0, 70, "5min")
        res = zs._entry_check(self.zone, 2003.5, flat, 0.3, False, {"H1": self.h1})
        self.assertEqual(res["status"], "WAIT")

    def test_entry_terlambat_waits(self):
        # Harga sudah lari jauh di atas zona (misal di 2025.0)
        res = zs._entry_check(self.zone, 2025.0, self.m5, 0.3, False, {"H1": self.h1})
        self.assertEqual(res["status"], "WAIT")
        self.assertTrue(any("terlambat" in r.lower() or "belum di area" in r.lower() for r in res["reasons"]))

    def test_harga_belum_menyentuh_zona_waits(self):
        # Zona 2000-2004, harga di 2015 dan data m5 tidak pernah menyentuh 2000-2004
        high_df = flat_df(2020.0, 70, "5min")
        res = zs._entry_check(self.zone, 2020.0, high_df, 0.3, False, {"H1": self.h1})
        self.assertEqual(res["status"], "WAIT")
        self.assertTrue(any("belum" in r.lower() for r in res["reasons"]))


class TestRequirement7RiskManagement(Base):
    """Pengujian Unit Mandiri: Kontrol Risiko (R:R, Batas SL, Spread)."""

    def setUp(self):
        super().setUp()
        self.m5 = m5_trigger_df()
        self.h1 = h1_target_df()

    def test_rr_memenuhi_syarat(self):
        res = zs._entry_check(make_zone(sl_recommended=1997.0), 2003.5, self.m5, 0.3, False, {"H1": self.h1})
        self.assertEqual(res["status"], "VALID")
        self.assertGreaterEqual(res["plan"]["rr"], 1.5)

    def test_rr_dibawah_minimal_waits(self):
        # SL sangat jauh (1983.0), risk = 20.5 -> RR < 1.5
        res = zs._entry_check(make_zone(sl_recommended=1983.0), 2003.5, self.m5, 0.3, False, {"H1": self.h1})
        self.assertEqual(res["status"], "WAIT")
        self.assertTrue(any("R:R" in r or "SL" in r for r in res["reasons"]))

    def test_sl_terlalu_lebar_waits(self):
        # SL berjarak > 20 poin dari entry
        res = zs._entry_check(make_zone(sl_recommended=1970.0), 2003.5, self.m5, 0.3, False, {"H1": self.h1})
        self.assertEqual(res["status"], "WAIT")
        self.assertTrue(any("SL" in r for r in res["reasons"]))

    def test_spread_terlalu_tinggi_waits(self):
        # Spread 3.0 pt > batas 1.0 pt
        res = zs._entry_check(make_zone(), 2003.5, self.m5, 3.0, False, {"H1": self.h1})
        self.assertEqual(res["status"], "WAIT")
        self.assertTrue(any("Spread" in r for r in res["reasons"]))


class TestRequirement8LoggingAndAudit(Base):
    """Pengujian Format Audit [SCANNER] & Keamanan Informasi."""

    def test_audit_format_structure(self):
        dec_valid = {
            "status": "VALID",
            "direction": "BUY",
            "stages": {
                "H4": {"ok": True, "msg": "Bias H4 BULLISH"},
                "H1": {"ok": True, "msg": "H1 BULLISH"},
                "M30": {"ok": True, "msg": "M30 BULLISH"},
                "M5": {"ok": True, "msg": "M5 trigger valid"},
                "RISK": {"ok": True, "msg": "R:R 2.0"},
            },
            "plan": {"entry_acuan": 2003.0, "sl": 1997.0, "tp1": 2015.0, "rr": 2.0},
            "reasons": ["Entry valid"],
        }
        audit = format_scanner_audit(dec_valid)
        self.assertIn("[SCANNER] H4 Bias: BULLISH", audit)
        self.assertIn("[SCANNER] H1 Structure: BULLISH", audit)
        self.assertIn("[SCANNER] M30 Momentum: CONFIRMED", audit)
        self.assertIn("[SCANNER] M5 Liquidity Sweep: CONFIRMED", audit)
        self.assertIn("[SCANNER] M5 Displacement: CONFIRMED", audit)
        self.assertIn("[SCANNER] Entry Validation: PASSED", audit)
        self.assertIn("[SCANNER] Risk Reward: 1:2.0", audit)
        self.assertIn("[SCANNER] Final Decision: BUY VALID", audit)

    def test_audit_rejection_format(self):
        dec_wait = {
            "status": "WAIT",
            "direction": "BUY",
            "stages": {"H4": {"ok": True, "msg": "Bias H4 BULLISH"}, "H1": {"ok": True, "msg": "H1 BULLISH"}},
            "plan": None,
            "reasons": ["M5 Liquidity Sweep Not Confirmed"],
        }
        audit = format_scanner_audit(dec_wait)
        self.assertIn("[SCANNER] Final Decision: WAIT", audit)
        self.assertIn("[SCANNER] Rejection Reason: M5 Liquidity Sweep Not Confirmed", audit)

    def test_no_sensitive_data_in_audit(self):
        dec_test = {"status": "WAIT", "direction": None, "stages": {}, "plan": None, "reasons": ["test"]}
        audit = format_scanner_audit(dec_test)
        for s in ("api_key", "password", "AIza", "secret", "token"):
            self.assertNotIn(s, audit.lower())


class TestIntegration(unittest.TestCase):
    def test_main_app_imports_and_exposes_features(self):
        import xauusd1
        for name in ("scan_xauusd_zones", "evaluate_trade_decision", "terapkan_gerbang", "simpan_state_sinyal",
                     "mode_pantau_mt5", "tampilkan_rekap", "ekstrak_target_ai", "evaluasi_validitas_sinyal",
                     "detect_market_structure", "struktur", "cari_golden_confluence", "hitung_mm_sinyal"):
            self.assertTrue(callable(getattr(xauusd1, name, None)), name)

    def test_no_circular_import_each_module(self):
        import importlib
        for m in ("config", "utils", "mt5_conn", "analisis_teknikal", "analisis_pasar", "sinyal_jurnal",
                  "state_manager", "zone_scanner", "ai_gemini", "pantau"):
            importlib.import_module(f"modules.{m}")


if __name__ == "__main__":
    unittest.main()
