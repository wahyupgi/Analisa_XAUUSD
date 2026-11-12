# -*- coding: utf-8 -*-
"""
tests/test_zone_scanner_validation.py
Pengujian Validasi Mandiri Python-Only XAUUSD SMC Zone Scanner (TASK Section H).

Minimal 14 skenario pengujian mandiri:
  1. H4 bullish dan seluruh timeframe mendukung BUY.
  2. H4 bearish dan seluruh timeframe mendukung SELL.
  3. H4 bullish tetapi H1 bearish (Konflik HTF).
  4. M30 tidak memiliki momentum (WEAK / Ranging).
  5. Liquidity sweep terjadi jauh dari zona (Irrelevant location).
  6. Harga menyentuh zona tanpa liquidity sweep.
  7. Sweep terjadi tetapi tidak ada CHoCH/BOS/CISD.
  8. Struktur terkonfirmasi tetapi tidak ada displacement.
  9. Entry valid dengan RR memenuhi minimum (>= 1.5).
  10. RR di bawah minimum (< 1.5).
  11. Harga belum menyentuh zona.
  12. Candle terakhir belum selesai (running bar).
  13. Data OHLC kosong atau tidak valid.
  14. Tidak ada zona yang memenuhi syarat.

Tidak memerlukan koneksi MT5 dan tidak memanggil API Gemini.
Semua skenario menggunakan data OHLC sintetis terkontrol.
"""
import math
import os
import tempfile
import unittest

import pandas as pd
import numpy as np

import modules.zone_scanner as zs
from modules.zone_scanner import (
    Zone, evaluate_trade_decision, scan_xauusd_zones, evaluate_htf_context,
    evaluate_m30_momentum, is_sweep_relevant_to_zone, validate_entry_setup,
    format_scanner_audit, format_scan_presentation,
)
from modules.mt5_conn import tambah_indikator


def _make_df(rows, start="2026-11-01 00:00:00", freq="5min"):
    t0 = pd.to_datetime(start)
    idx = [t0 + i * pd.Timedelta(freq) for i in range(len(rows))]
    df = pd.DataFrame(rows, columns=["Open", "High", "Low", "Close", "Volume"], index=idx)
    return tambah_indikator(df)


def _make_h4_bullish():
    rows = []
    base = 1950.0
    for i in range(70):
        c = base + i * 2.5
        rows.append((c - 1.0, c + 2.0, c - 2.0, c + 1.0, 100))
    return _make_df(rows, freq="4h")


def _make_h4_bearish():
    rows = []
    base = 2200.0
    for i in range(70):
        c = base - i * 2.5
        rows.append((c + 1.0, c + 2.0, c - 2.0, c - 1.0, 100))
    return _make_df(rows, freq="4h")


def _make_h1_with_high_target():
    # 70 bar H1 dengan swing high di 2030.5 yang belum tertembus untuk target BUY
    rows = []
    for i in range(21):
        c = 1990 + 2 * i
        rows.append((c - 0.5, c + 0.5, c - 1.0, c, 100))
    for i in range(21, 41):
        c = 2030 - 1.1 * (i - 20)
        rows.append((c + 0.5, c + 0.8, c - 0.5, c, 100))
    c = 2008.0
    for _ in range(41, 70):
        c -= 0.1
        rows.append((c + 0.2, c + 0.3, c - 0.3, c, 100))
    return _make_df(rows, freq="1h")


def _make_h1_with_low_target():
    # 70 bar H1 dengan swing low di 1989.5 yang belum tertembus untuk target SELL
    rows = []
    for i in range(70):
        if i < 20:
            c = 2023.0 - 0.5 * i
            rows.append((c, c + 0.5, c - 0.5, c, 100))
        else:
            c = 1990.0 + 0.6 * (i - 20)
            rows.append((c, c + 1.0, 1989.5 if i == 20 else c - 0.5, c, 100))
    return _make_df(rows, freq="1h")


def _make_h1_bearish_simple():
    rows = []
    base = 2100.0
    for i in range(70):
        c = base - i * 1.0
        rows.append((c + 0.5, c + 0.8, c - 1.0, c - 0.5, 100))
    return _make_df(rows, freq="1h")


def _make_m30_bullish():
    rows = []
    base = 2005.0
    for i in range(70):
        c = base + i * 0.2
        rows.append((c - 0.2, c + 0.5, c - 0.3, c + 0.2, 100))
    return _make_df(rows, freq="30min")


def _make_m30_bearish():
    rows = []
    for i in range(70):
        c = 2023.0 - 0.05 * i
        rows.append((c, c + 0.4, c - 0.4, c - 0.03, 100))
    return _make_df(rows, freq="30min")


def _make_m30_ranging_weak():
    # Osilasi sempit horizontal tanpa arah jelas, RSI < 46
    rows = []
    for i in range(70):
        c = 2005.0 + 0.3 * math.sin(i / 1.5)
        rows.append((c, c + 0.5, c - 0.5, c - 0.05, 100))
    df = _make_df(rows, freq="30min")
    df["RSI14"] = 43.0
    return df


def _make_m5_valid_buy_trigger(zone_lo=2000.0, zone_hi=2004.0):
    """
    Sintetis M5 trigger lengkap untuk BUY di zona Demand 2000-2004:
      1. Osilasi sebelum zona
      2. Swing High 2012.0
      3. Penurunan menuju zona Demand (sentuh 2003.0)
      4. Swing Low terbentuk di 2003.0
      5. Liquidity Sweep: SWEEP_LOW wick menembus ke 2001.0, close kembali ke 2003.5 (di area zona)
      6. Impuls reversal dengan displacement bullish kuat (body >= 2.0 ATR)
      7. BOS break close > 2012.0 (k_struct > k_sweep)
      8. Pullback masuk kembali ke zona di 2003.5 untuk entry
    """
    rows = []
    for i in range(45):
        c = 2008.0 + 2.0 * math.sin(i / 2.0)
        rows.append((c - 0.3, c + 1.0, c - 1.0, c + 0.3, 100))
    # Swing High 2012.0
    rows.append((2009.0, 2012.0, 2008.5, 2009.5, 100))
    # Penurunan ke zona Demand
    for p in (2008.0, 2006.0, 2005.0, 2004.0):
        rows.append((p, p + 0.5, p - 0.5, p, 100))
    # Swing low 2003.0 di dalam zona
    rows.append((2004.0, 2004.5, 2003.0, 2004.0, 100))
    rows.append((2004.0, 2005.5, 2003.8, 2005.0, 100))
    rows.append((2005.0, 2006.0, 2004.5, 2005.5, 100))
    rows.append((2005.5, 2006.5, 2005.0, 2006.0, 100))  # konfirmasi swing low
    # SWEEP_LOW ke 2001.0 (wick menyapu swing low 2003.0, close 2003.5 di dalam zona)
    rows.append((2005.0, 2005.2, 2001.0, 2003.5, 200))
    # Candle konfirmasi sweep + displacement bullish
    rows.append((2003.5, 2006.0, 2003.2, 2005.5, 200))
    for o in (2006.0, 2008.0, 2010.0):
        rows.append((o, o + 2.5, o - 0.2, o + 2.2, 300))  # displacement kuat
    # BOS break level 2012.0
    rows.append((2012.0, 2015.0, 2011.8, 2014.5, 500))
    # Pullback kembali ke zona 2003.5
    for c in (2012.0, 2009.0, 2006.5, 2004.5, 2003.5):
        rows.append((c + 0.5, c + 0.8, c - 0.5, c, 100))
    return _make_df(rows, freq="5min")


def _make_m5_valid_sell_trigger(zone_lo=2020.0, zone_hi=2024.0):
    """
    Sintetis M5 trigger lengkap untuk SELL di zona Supply 2020-2024:
      1. Osilasi sebelum zona
      2. Swing Low 2010.0
      3. Kenaikan menuju zona Supply (sentuh 2021.0)
      4. Swing High terbentuk di 2021.0
      5. Liquidity Sweep: SWEEP_HIGH wick menembus ke 2023.5, close kembali ke 2020.5 (di area zona)
      6. Impuls reversal dengan displacement bearish kuat
      7. BOS break close < 2010.0 (k_struct > k_sweep)
      8. Pullback masuk kembali ke zona di 2021.0 untuk entry
    """
    rows = []
    for i in range(45):
        c = 2015.0 + 2.0 * math.sin(i / 2.0)
        rows.append((c - 0.3, c + 1.0, c - 1.0, c + 0.3, 100))
    # Swing Low 2010.0
    rows.append((2014.0, 2014.5, 2010.0, 2013.5, 100))
    # Kenaikan ke zona Supply
    for p in (2015.0, 2017.0, 2019.0, 2020.0):
        rows.append((p, p + 0.5, p - 0.5, p, 100))
    # Swing High 2021.0 di dalam zona
    rows.append((2020.0, 2021.0, 2019.5, 2020.0, 100))
    rows.append((2020.0, 2020.5, 2018.5, 2019.0, 100))
    rows.append((2019.0, 2019.5, 2018.0, 2018.5, 100))
    rows.append((2018.5, 2019.0, 2017.5, 2018.0, 100))  # konfirmasi swing high
    # SWEEP_HIGH ke 2023.5 (wick menyapu swing high 2021.0, close 2020.5 di dalam zona)
    rows.append((2019.0, 2023.5, 2018.8, 2020.5, 200))
    # Candle konfirmasi sweep + displacement bearish
    rows.append((2020.5, 2020.8, 2017.5, 2018.0, 200))
    for o in (2018.0, 2016.0, 2014.0):
        rows.append((o, o + 0.2, o - 2.5, o - 2.2, 300))  # displacement kuat
    # BOS break level 2010.0
    rows.append((2011.0, 2011.5, 2008.0, 2008.5, 500))
    # Pullback kembali ke zona 2021.0
    for c in (2012.0, 2015.0, 2018.0, 2020.0, 2021.0):
        rows.append((c - 0.5, c + 0.5, c - 0.8, c, 100))
    return _make_df(rows, freq="5min")


def _make_test_zone(zone_type="DEMAND", price_lo=2000.0, price_hi=2004.0, sl=1997.0, **kw):
    base = dict(
        zone_id=f"TEST_{zone_type}", symbol="XAUUSD", timeframe="H1", zone_type=zone_type,
        price_low=price_lo, price_high=price_hi, midpoint=(price_lo + price_hi) / 2.0,
        score=90.0, grade="A+", origin_bos={"tipe": "BOS", "strength_atr": 1.2},
        fvg_confirmation={"status": "UNTOUCHED"}, displacement_strength=2.2,
        freshness="FRESH", touch_count=0, liquidity_sweep_status="NONE",
        invalidation_price=price_lo if zone_type == "DEMAND" else price_hi,
        is_valid=True, rejection_reason="", sl_recommended=sl,
    )
    base.update(kw)
    return Zone(**base)


class TestZoneScannerValidation(unittest.TestCase):
    """Pengujian Unit Mandiri 14 Skenario SMC Zone Scanner (TASK Section H)."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self._orig_audit = zs.FILE_AUDIT_ZONA
        self._orig_scan_func = zs.scan_xauusd_zones
        zs.FILE_AUDIT_ZONA = os.path.join(self._tmp, "audit_test.jsonl")
        zs._AUDIT_LAST.clear()
        zs._SCAN_CACHE.update(key=None, data=None, ts=0.0)
        zs._FRAMES_CACHE.update(ts=0.0, data=None)

    def tearDown(self):
        zs.FILE_AUDIT_ZONA = self._orig_audit
        zs.scan_xauusd_zones = self._orig_scan_func

    # -------------------------------------------------------------------------
    # SKENARIO 1: H4 bullish dan seluruh timeframe mendukung BUY
    # -------------------------------------------------------------------------
    def test_01_h4_bullish_all_aligned_buy(self):
        h1 = _make_h1_with_high_target()
        zone = _make_test_zone("DEMAND", 2000.0, 2004.0, sl=1997.0, bos_candle_time=h1.index[30])
        h4 = _make_h4_bullish()
        m30 = _make_m30_bullish()
        m5 = _make_m5_valid_buy_trigger(2000.0, 2004.0)

        frames = {"H4": h4, "H1": h1, "M30": m30, "M5": m5}
        zs.scan_xauusd_zones = lambda frames=None, current_price=None, symbol=None: {
            "symbol": "XAUUSD", "current_price": 2003.5, "active_zones": [zone.to_dict()],
            "rejected_zones": [], "best_demand": zone.to_dict(), "best_supply": None,
            "htf_context": {"H4": "BULLISH", "H1": "BULLISH", "M30": "BULLISH", "has_conflict": False, "m30_conflict": False, "bias_utama": "BULLISH"},
            "empty_reason": "", "summary_text": "Ditemukan zona", "data_ok": True,
        }
        dec = evaluate_trade_decision(frames=frames, current_price=2003.5, spread=0.2, live=False)
        self.assertEqual(dec["status"], "VALID", f"Harus VALID: {dec.get('reasons')}")
        self.assertEqual(dec["direction"], "BUY")
        self.assertIsNotNone(dec["plan"])
        self.assertGreaterEqual(dec["plan"]["rr"], 1.5)
        audit = format_scanner_audit(dec)
        self.assertIn("[SCANNER] Final Decision: BUY VALID", audit)
        self.assertIn("[SCANNER] H4 Bias: BULLISH", audit)
        self.assertIn("[SCANNER] M30 Momentum: CONFIRMED", audit)
        self.assertIn("[SCANNER] M5 Liquidity Sweep: CONFIRMED", audit)
        self.assertIn("[SCANNER] CHoCH / BOS / CISD: CONFIRMED", audit)

    # -------------------------------------------------------------------------
    # SKENARIO 2: H4 bearish dan seluruh timeframe mendukung SELL
    # -------------------------------------------------------------------------
    def test_02_h4_bearish_all_aligned_sell(self):
        h1 = _make_h1_with_low_target()
        zone = _make_test_zone("SUPPLY", 2020.0, 2024.0, sl=2027.0, bos_candle_time=h1.index[25])
        h4 = _make_h4_bearish()
        m30 = _make_m30_bearish()
        m5 = _make_m5_valid_sell_trigger(2020.0, 2024.0)

        frames = {"H4": h4, "H1": h1, "M30": m30, "M5": m5}
        zs.scan_xauusd_zones = lambda frames=None, current_price=None, symbol=None: {
            "symbol": "XAUUSD", "current_price": 2021.0, "active_zones": [zone.to_dict()],
            "rejected_zones": [], "best_demand": None, "best_supply": zone.to_dict(),
            "htf_context": {"H4": "BEARISH", "H1": "BEARISH", "M30": "BEARISH", "has_conflict": False, "m30_conflict": False, "bias_utama": "BEARISH"},
            "empty_reason": "", "summary_text": "Ditemukan zona", "data_ok": True,
        }
        dec = evaluate_trade_decision(frames=frames, current_price=2021.0, spread=0.2, live=False)
        self.assertEqual(dec["status"], "VALID", f"Harus VALID: {dec.get('reasons')}")
        self.assertEqual(dec["direction"], "SELL")
        self.assertIsNotNone(dec["plan"])
        self.assertGreaterEqual(dec["plan"]["rr"], 1.5)
        audit = format_scanner_audit(dec)
        self.assertIn("[SCANNER] Final Decision: SELL VALID", audit)
        self.assertIn("[SCANNER] H4 Bias: BEARISH", audit)
        self.assertIn("[SCANNER] M5 Liquidity Sweep: CONFIRMED", audit)

    # -------------------------------------------------------------------------
    # SKENARIO 3: H4 bullish tetapi H1 bearish (Konflik HTF)
    # -------------------------------------------------------------------------
    def test_03_h4_bullish_h1_bearish_conflict(self):
        zone = _make_test_zone("DEMAND", 2000.0, 2004.0)
        frames = {"H4": _make_h4_bullish(), "H1": _make_h1_bearish_simple(), "M30": _make_m30_bullish(), "M5": _make_m5_valid_buy_trigger()}
        zs.scan_xauusd_zones = lambda frames=None, current_price=None, symbol=None: {
            "symbol": "XAUUSD", "current_price": 2003.5, "active_zones": [zone.to_dict()],
            "rejected_zones": [], "best_demand": zone.to_dict(), "best_supply": None,
            "htf_context": {"H4": "BULLISH", "H1": "BEARISH", "M30": "BULLISH", "has_conflict": True, "m30_conflict": False, "bias_utama": "WAIT (KONFLIK H4 vs H1)"},
            "empty_reason": "Konflik H4 vs H1", "summary_text": "Konflik", "data_ok": True,
        }
        dec = evaluate_trade_decision(frames=frames, current_price=2003.5, live=False)
        self.assertEqual(dec["status"], "WAIT")
        self.assertIn("Konflik", " ".join(dec["reasons"]))
        audit = format_scanner_audit(dec)
        self.assertIn("[SCANNER] Final Decision: WAIT", audit)
        self.assertIn("[SCANNER] H1 Structure: CONFLICT", audit)

    # -------------------------------------------------------------------------
    # SKENARIO 4: M30 tidak memiliki momentum
    # -------------------------------------------------------------------------
    def test_04_m30_no_momentum_weak(self):
        zone = _make_test_zone("DEMAND", 2000.0, 2004.0)
        weak_m30 = _make_m30_ranging_weak()
        m30_eval = evaluate_m30_momentum(weak_m30, "BULLISH")
        self.assertEqual(m30_eval["status"], "WEAK")

        frames = {"H4": _make_h4_bullish(), "H1": _make_h1_with_high_target(), "M30": weak_m30, "M5": _make_m5_valid_buy_trigger()}
        zs.scan_xauusd_zones = lambda frames=None, current_price=None, symbol=None: {
            "symbol": "XAUUSD", "current_price": 2003.5, "active_zones": [zone.to_dict()],
            "rejected_zones": [], "best_demand": zone.to_dict(), "best_supply": None,
            "htf_context": {"H4": "BULLISH", "H1": "BULLISH", "M30": "NETRAL", "has_conflict": False, "m30_conflict": False, "bias_utama": "BULLISH"},
            "empty_reason": "", "summary_text": "ok", "data_ok": True,
        }
        dec = evaluate_trade_decision(frames=frames, current_price=2003.5, live=False)
        self.assertEqual(dec["status"], "WAIT")
        self.assertEqual(dec["stages"]["M30"]["status"], "WEAK")
        self.assertIn("momentum", " ".join(dec["reasons"]).lower())

    # -------------------------------------------------------------------------
    # SKENARIO 5: Liquidity sweep terjadi jauh dari zona
    # -------------------------------------------------------------------------
    def test_05_liquidity_sweep_far_from_zone_rejected(self):
        zone = _make_test_zone("DEMAND", 2000.0, 2004.0)
        sweep_far = {"type": "SWEEP_LOW", "level": 2030.0, "wick_extreme": 2029.0, "k": 30}
        rel, msg = is_sweep_relevant_to_zone(sweep_far, zone, atr=1.5)
        self.assertFalse(rel)
        self.assertIn("jauh", msg.lower())

        # Buat M5 di mana sweep low terjadi di 2030.0 bukan di area zona
        rows = []
        for i in range(50):
            c = 2030.0 + math.sin(i)
            rows.append((c, c + 1.0, c - 1.0, c, 100))
        rows.append((2030.0, 2030.5, 2029.0, 2030.2, 100))
        rows.append((2030.2, 2031.0, 2027.0, 2029.5, 200)) # sweep low di 2027 jauh dari zona 2000
        for _ in range(10):
            rows.append((2003.0, 2003.5, 2002.5, 2003.0, 100)) # harga saat ini masuk zona
        m5_far = _make_df(rows)
        res = zs._entry_check(zone, 2003.0, m5_far)
        self.assertEqual(res["status"], "WAIT")
        self.assertTrue(any("jauh" in r.lower() or "tidak terkonfirmasi" in r.lower() for r in res["reasons"]))

    # -------------------------------------------------------------------------
    # SKENARIO 6: Harga menyentuh zona tanpa liquidity sweep
    # -------------------------------------------------------------------------
    def test_06_zone_touched_without_liquidity_sweep_waits(self):
        zone = _make_test_zone("DEMAND", 2000.0, 2004.0)
        rows = []
        for i in range(55):
            c = 2005.0 - (i * 0.05)
            rows.append((c, c + 0.3, c - 0.3, c, 100))
        for _ in range(10):
            rows.append((2002.0, 2002.5, 2001.8, 2002.0, 100))
        m5_no_sw = _make_df(rows)
        res = zs._entry_check(zone, 2002.0, m5_no_sw)
        self.assertEqual(res["status"], "WAIT")
        self.assertTrue(any("sweep" in r.lower() for r in res["reasons"]))

    # -------------------------------------------------------------------------
    # SKENARIO 7: Sweep terjadi tetapi tidak ada CHoCH/BOS/CISD
    # -------------------------------------------------------------------------
    def test_07_sweep_without_choch_bos_cisd_waits(self):
        zone = _make_test_zone("DEMAND", 2000.0, 2004.0)
        rows = []
        for i in range(50):
            c = 2006.0 + math.sin(i)
            rows.append((c, c + 0.5, c - 0.5, c, 100))
        rows.append((2003.0, 2003.5, 2002.0, 2003.0, 100))
        for _ in range(5):
            rows.append((2003.0, 2003.5, 2002.5, 2003.0, 100))
        rows.append((2003.0, 2003.2, 2000.5, 2002.8, 200))
        for _ in range(8):
            rows.append((2002.8, 2003.2, 2002.6, 2003.0, 100))
        m5_no_bos = _make_df(rows)
        res = zs._entry_check(zone, 2003.0, m5_no_bos)
        self.assertEqual(res["status"], "WAIT")
        self.assertTrue(any("struktur" in r.lower() or "bos/choch" in r.lower() for r in res["reasons"]))

    # -------------------------------------------------------------------------
    # SKENARIO 8: Struktur terkonfirmasi tetapi tidak ada displacement
    # -------------------------------------------------------------------------
    def test_08_structure_confirmed_without_displacement_waits(self):
        zone = _make_test_zone("DEMAND", 2000.0, 2004.0)
        rows = []
        for i in range(45):
            c = 2008.0 + 2.0 * math.sin(i / 2.0)
            rows.append((c - 0.3, c + 1.0, c - 1.0, c + 0.3, 100))
        rows.append((2009.0, 2012.0, 2008.5, 2009.5, 100))
        for p in (2008.0, 2006.0, 2005.0, 2004.0):
            rows.append((p, p + 0.5, p - 0.5, p, 100))
        rows.append((2004.0, 2004.5, 2003.0, 2004.0, 100))
        rows.append((2004.0, 2005.5, 2003.8, 2005.0, 100))
        rows.append((2005.0, 2006.0, 2004.5, 2005.5, 100))
        rows.append((2005.5, 2006.5, 2005.0, 2006.0, 100))
        rows.append((2005.0, 2005.2, 2001.0, 2003.5, 200))
        rows.append((2003.5, 2004.0, 2003.2, 2003.7, 100))
        for o in (2006.0, 2008.0, 2010.0):
            rows.append((o, o + 0.4, o - 0.2, o + 0.25, 100))
        # BOS break level 2012.0 tetapi dengan candle body kecil (0.2 pt < 0.8 ATR)
        rows.append((2012.0, 2013.0, 2011.9, 2012.2, 100))
        for c in (2010.0, 2008.0, 2006.0, 2004.0, 2003.5):
            rows.append((c + 0.2, c + 0.4, c - 0.2, c, 100))
        df_weak_disp = _make_df(rows)
        res = zs._entry_check(zone, 2003.5, df_weak_disp)
        self.assertEqual(res["status"], "WAIT")
        self.assertTrue(any("displacement" in r.lower() for r in res["reasons"]))

    # -------------------------------------------------------------------------
    # SKENARIO 9: Entry valid dengan RR memenuhi minimum
    # -------------------------------------------------------------------------
    def test_09_entry_valid_rr_meets_minimum(self):
        zone = _make_test_zone("DEMAND", 2000.0, 2004.0, sl=1997.0)
        m5 = _make_m5_valid_buy_trigger(2000.0, 2004.0)
        h1 = _make_h1_with_high_target()
        res = zs._entry_check(zone, 2003.5, m5, htf_frames={"H1": h1, "H4": _make_h4_bullish()})
        self.assertEqual(res["status"], "VALID")
        self.assertGreaterEqual(res["plan"]["rr"], 1.5)
        self.assertIn("Entry valid", " ".join(res["reasons"]))

    # -------------------------------------------------------------------------
    # SKENARIO 10: RR di bawah minimum
    # -------------------------------------------------------------------------
    def test_10_rr_below_minimum_waits(self):
        zone = _make_test_zone("DEMAND", 2000.0, 2004.0, sl=1990.0) # SL lebar (Risk 12 pt)
        m5 = _make_m5_valid_buy_trigger(2000.0, 2004.0)
        # H1 dengan target sangat dekat di 2006 (Reward 3.5 pt => RR < 0.5)
        rows_h1 = []
        for i in range(50):
            rows_h1.append((2005.0, 2006.0, 2004.0, 2005.5, 100))
        h1_low_target = _make_df(rows_h1, freq="1h")
        res = zs._entry_check(zone, 2003.5, m5, htf_frames={"H1": h1_low_target})
        self.assertEqual(res["status"], "WAIT")
        self.assertTrue(any("r:r" in r.lower() or "rr" in r.lower() for r in res["reasons"]))

    # -------------------------------------------------------------------------
    # SKENARIO 11: Harga belum menyentuh zona
    # -------------------------------------------------------------------------
    def test_11_price_not_touching_zone_waits(self):
        zone = _make_test_zone("DEMAND", 2000.0, 2004.0)
        rows = []
        for i in range(60):
            c = 2025.0 + math.sin(i)
            rows.append((c, c + 1.0, c - 1.0, c, 100))
        m5_far_price = _make_df(rows)
        res = zs._entry_check(zone, 2025.0, m5_far_price)
        self.assertEqual(res["status"], "WAIT")
        self.assertTrue(any("belum mencapai atau menyentuh" in r.lower() for r in res["reasons"]))

    # -------------------------------------------------------------------------
    # SKENARIO 12: Candle terakhir belum selesai
    # -------------------------------------------------------------------------
    def test_12_last_candle_unclosed_waits(self):
        zone = _make_test_zone("DEMAND", 2000.0, 2004.0)
        m5 = _make_m5_valid_buy_trigger(2000.0, 2004.0)
        running_time = m5.index[-1] + pd.Timedelta(minutes=2)
        res = zs._entry_check(zone, 2003.5, m5, server_time=running_time)
        self.assertEqual(res["status"], "WAIT")
        self.assertTrue(any("belum selesai" in r.lower() for r in res["reasons"]))

    # -------------------------------------------------------------------------
    # SKENARIO 13: Data OHLC kosong atau tidak valid
    # -------------------------------------------------------------------------
    def test_13_ohlc_empty_or_invalid_waits(self):
        zone = _make_test_zone("DEMAND", 2000.0, 2004.0)
        res_none = zs._entry_check(zone, 2003.5, None)
        self.assertEqual(res_none["status"], "WAIT")
        self.assertIn("tidak mencukupi", " ".join(res_none["reasons"]).lower())

        empty_df = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        res_empty = zs._entry_check(zone, 2003.5, empty_df)
        self.assertEqual(res_empty["status"], "WAIT")

        for bad_p in (float("nan"), -10.0, 0.0):
            res_bad_p = zs._entry_check(zone, bad_p, _make_m5_valid_buy_trigger())
            self.assertEqual(res_bad_p["status"], "WAIT")

    # -------------------------------------------------------------------------
    # SKENARIO 14: Tidak ada zona yang memenuhi syarat
    # -------------------------------------------------------------------------
    def test_14_no_eligible_zones_waits(self):
        rows = [(2000.0, 2001.0, 1999.0, 2000.0, 100)] * 70
        flat_f = _make_df(rows, freq="1h")
        frames = {"D1": flat_f, "H4": flat_f, "H1": flat_f, "M30": flat_f, "M5": flat_f}
        scan_res = scan_xauusd_zones(frames=frames, current_price=2000.0)
        self.assertEqual(scan_res["active_zones"], [])
        self.assertTrue(len(scan_res["empty_reason"]) > 0)
        self.assertIn("WAIT", scan_res["summary_text"])

        dec = evaluate_trade_decision(frames=frames, current_price=2000.0, live=False)
        self.assertEqual(dec["status"], "WAIT")

    # -------------------------------------------------------------------------
    # PENGUJIAN PENYAJIAN SCAN FORMAT (TASK Section F)
    # -------------------------------------------------------------------------
    def test_format_scan_presentation_output(self):
        h1 = _make_h1_with_high_target()
        zone = _make_test_zone("DEMAND", 2000.0, 2004.0, sl=1997.0, bos_candle_time=h1.index[30])
        h4 = _make_h4_bullish()
        m30 = _make_m30_bullish()
        m5 = _make_m5_valid_buy_trigger(2000.0, 2004.0)

        scan_mock = {
            "symbol": "XAUUSD", "current_price": 2003.5, "active_zones": [zone.to_dict()],
            "rejected_zones": [], "best_demand": zone.to_dict(), "best_supply": None,
            "htf_context": {"H4": "BULLISH", "H1": "BULLISH", "M30": "BULLISH", "bias_utama": "BULLISH"},
            "empty_reason": "", "summary_text": "ok", "data_ok": True,
        }
        dec_mock = {
            "status": "VALID", "direction": "BUY", "zone": zone.to_dict(),
            "plan": {"entry_acuan": 2002.75, "entry_min": 2001.5, "entry_max": 2004.0, "sl": 1997.0, "tp1": 2030.2, "rr": 4.77},
            "stages": {
                "H4": {"ok": True, "msg": "Bias H4 BULLISH"},
                "H1": {"ok": True, "msg": "H1 BULLISH"},
                "M30": {"ok": True, "status": "CONFIRMED", "msg": "M30 BULLISH"},
                "zone_touch": {"status": "CONFIRMED"},
                "liquidity_sweep": {"status": "CONFIRMED", "msg": "SSL sweep @ 2001.0"},
                "structure_trigger": {"status": "CONFIRMED", "msg": "BOS bull @ 2012.0"},
                "displacement": {"status": "CONFIRMED", "msg": "Displacement M5 2.2 ATR"},
                "risk_reward": {"status": "CONFIRMED", "rr": 4.77},
            },
            "reasons": ["Entry valid: trigger M5 BOS bull"],
        }
        rendered = format_scan_presentation(scan_mock, dec_mock)
        self.assertIn("XAUUSD PYTHON SMC SCANNER", rendered)
        self.assertIn("MARKET OVERVIEW", rendered)
        self.assertIn("ZONE ANALYSIS", rendered)
        self.assertIn("ENTRY VALIDATION", rendered)
        self.assertIn("FINAL DECISION", rendered)
        self.assertIn("BUY VALID", rendered)
        self.assertIn("2002.75", rendered)


if __name__ == "__main__":
    unittest.main()
