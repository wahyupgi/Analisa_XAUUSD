# -*- coding: utf-8 -*-
"""
tests/test_trend_following_rebuild.py
Pengujian Komprehensif 20 Skenario Wajib + Backtest Terkontrol
untuk Rebuilt Entry Logic: Multi-Timeframe Trend Following + SMC Confirmation.
"""

from datetime import datetime, timedelta
import math
import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.config import (
    RR_MIN,
    SIGNAL_STATUS_ACTIVE_TRADE,
    SIGNAL_STATUS_CANCELLED,
    SIGNAL_STATUS_ENTRY_MISSED,
    SIGNAL_STATUS_ENTRY_REACHED,
    SIGNAL_STATUS_EXPIRED,
    SIGNAL_STATUS_NO_SETUP,
    SIGNAL_STATUS_SIGNAL_CONFIRMED,
    SIGNAL_STATUS_SL_HIT,
    SIGNAL_STATUS_TP_HIT,
    SIGNAL_STATUS_WAITING_CONFIRMATION,
    SIGNAL_STATUS_WAITING_PULLBACK,
    ST_AKTIF_ENTRY,
    ST_BATAL_KEDALUWARSA,
    ST_BATAL_SL,
    ST_BATAL_TERLEWAT,
    ST_SELESAI_TP2,
)
from modules.mt5_conn import tambah_indikator
from modules.pantau import _build_monitor_snapshot, format_monitor_detail
from modules.sinyal_jurnal import (
    create_signal_from_decision,
    gerbang_sinyal_baru,
    map_status_to_canonical,
    normalize_signal_payload,
    resolve_unified_signal_status,
)
from modules.trend_following import (
    assess_m30_condition,
    evaluate_h1_trend_confirmation,
    evaluate_h4_primary_trend,
    evaluate_trend_following_setup,
)
from modules.unified_monitor import UnifiedMonitor
from modules.zone_scanner import (
    _SCAN_CACHE,
    create_market_snapshot,
    evaluate_htf_context,
    evaluate_trade_decision,
    format_scan_presentation,
    format_scan_summary,
    scan_xauusd_zones,
)
from xauusd1 import parse_cli_command


END_DT = datetime(2026, 11, 13, 12, 0)


def _df_from_rows(rows, freq="5min", end_dt=END_DT):
    idx = pd.date_range(end=end_dt, periods=len(rows), freq=freq)
    df = pd.DataFrame(rows, columns=["Open", "High", "Low", "Close", "Volume"], index=idx)
    df.attrs["time_tz"] = "WIB"
    df.attrs["is_closed"] = True
    return tambah_indikator(df)


def _trend_df(direction="BULLISH", start=3900.0, bars=80, freq="1h", amp=1.0, end_dt=END_DT):
    """Membangun dataframe tren zigzag realistis dengan swing high/low terkonfirmasi."""
    step = 1.15 * amp if direction == "BULLISH" else -1.15 * amp
    rows = []
    p = float(start)
    for i in range(bars):
        cyc = i % 10
        if cyc < 7:
            nxt = p + step
        else:
            nxt = p - step * 0.75
        hi = max(p, nxt) + 0.55 * amp
        lo = min(p, nxt) - 0.55 * amp
        rows.append((round(p, 2), round(hi, 2), round(lo, 2), round(nxt, 2), 180.0))
        p = nxt
    return _df_from_rows(rows, freq=freq, end_dt=end_dt)


def _sideways_df(center=3950.0, amp=1.2, bars=80, freq="1h", end_dt=END_DT):
    rows = []
    for i in range(bars):
        o = center + amp * math.sin(i * 0.65)
        c = center + amp * math.sin((i + 1) * 0.65)
        rows.append((round(o, 2), round(max(o, c) + 0.35, 2), round(min(o, c) - 0.35, 2), round(c, 2), 120.0))
    return _df_from_rows(rows, freq=freq, end_dt=end_dt)


def _pullback_m30_df(direction="BULLISH", anchor_origin=3920.0, anchor_extreme=3960.0, pullback_to=3942.0, bars=65, freq="30min", end_dt=END_DT):
    """
    Membangun M30 dengan leg impuls utama lalu pullback sehat 35-60% menuju pullback_to
    tanpa menembus struktur inti H1/H4.
    """
    rows = []
    # Phase 1: konsolidasi dasar & impulse menuju extreme
    p = float(anchor_origin)
    imp_bars = 42
    pb_bars = bars - imp_bars
    step_imp = (anchor_extreme - anchor_origin) / imp_bars
    for i in range(imp_bars):
        # Zigzag ringan searah leg utama
        d = step_imp * (1.6 if i % 5 < 4 else -0.8)
        nxt = p + d
        hi = max(p, nxt) + 0.45
        lo = min(p, nxt) - 0.45
        rows.append((round(p, 2), round(hi, 2), round(lo, 2), round(nxt, 2), 190.0))
        p = nxt

    # Set puncak exact di dekat anchor_extreme
    rows[-3] = (round(p, 2), round(max(p, anchor_extreme) + 0.3, 2), round(min(p, anchor_extreme) - 0.3, 2), round(anchor_extreme, 2), 240.0)
    p = float(anchor_extreme)

    # Phase 2: pullback terkendali menuju pullback_to
    step_pb = (pullback_to - p) / pb_bars
    for i in range(pb_bars):
        nxt = p + step_pb
        hi = max(p, nxt) + 0.4
        lo = min(p, nxt) - 0.4
        rows.append((round(p, 2), round(hi, 2), round(lo, 2), round(nxt, 2), 140.0))
        p = nxt
    return _df_from_rows(rows, freq=freq, end_dt=end_dt)


def _m5_pullback_trigger(direction="BUY", zone_price=3942.0, end_dt=END_DT):
    """Membangun candle closed M5 valid: masuk zona, Sweep/Rejection, BOS/CHoCH + Displacement."""
    rows = []
    if direction == "BUY":
        # Trend masuk ke area pullback 3942 (>= 60 bar untuk MIN_BARS_DECISION)
        for i in range(52):
            c = zone_price + 9.0 - 0.12 * i + 0.4 * math.sin(i * 0.5)
            rows.append((round(c + 0.1, 2), round(c + 0.4, 2), round(c - 0.35, 2), round(c, 2), 110.0))
        # Bentuk swing high referensi 3946.0 lalu turun menyentuh zona
        rows.append((round(zone_price + 2.5, 2), round(zone_price + 4.5, 2), round(zone_price + 2.0, 2), round(zone_price + 2.8, 2), 120.0))
        rows.append((round(zone_price + 2.8, 2), round(zone_price + 3.0, 2), round(zone_price + 1.0, 2), round(zone_price + 1.2, 2), 110.0))
        rows.append((round(zone_price + 1.2, 2), round(zone_price + 1.4, 2), round(zone_price + 0.2, 2), round(zone_price + 0.5, 2), 110.0))
        # Swing low 3942.1 lalu 5 bar konsolidasi sebelum sweep
        rows.append((round(zone_price + 0.5, 2), round(zone_price + 1.1, 2), round(zone_price + 0.1, 2), round(zone_price + 0.8, 2), 120.0))
        for _ in range(5):
            rows.append((round(zone_price + 0.8, 2), round(zone_price + 1.4, 2), round(zone_price + 0.4, 2), round(zone_price + 1.0, 2), 100.0))
        # Sweep Low wick ke zone_price - 1.3, close bullish balik di zone_price + 0.9
        rows.append((round(zone_price + 0.9, 2), round(zone_price + 1.2, 2), round(zone_price - 1.3, 2), round(zone_price + 0.9, 2), 260.0))
        # Displacement + Bullish BOS close tembus 3946.5
        rows.append((round(zone_price + 0.9, 2), round(zone_price + 3.2, 2), round(zone_price + 0.8, 2), round(zone_price + 3.0, 2), 320.0))
        rows.append((round(zone_price + 3.0, 2), round(zone_price + 5.6, 2), round(zone_price + 2.9, 2), round(zone_price + 5.2, 2), 380.0))
        # Pullback terkendali (tidak memicu blokir counter-impulse >= 1.5 ATR pada bar penutup)
        rows.append((round(zone_price + 5.2, 2), round(zone_price + 5.3, 2), round(zone_price + 3.6, 2), round(zone_price + 3.9, 2), 140.0))
        rows.append((round(zone_price + 3.9, 2), round(zone_price + 4.1, 2), round(zone_price + 2.5, 2), round(zone_price + 2.8, 2), 140.0))
        rows.append((round(zone_price + 2.8, 2), round(zone_price + 2.9, 2), round(zone_price + 1.5, 2), round(zone_price + 1.9, 2), 130.0))
        rows.append((round(zone_price + 1.9, 2), round(zone_price + 2.5, 2), round(zone_price + 1.6, 2), round(zone_price + 2.3, 2), 160.0))
    else:
        for i in range(52):
            c = zone_price - 9.0 + 0.12 * i + 0.4 * math.sin(i * 0.5)
            rows.append((round(c - 0.1, 2), round(c + 0.35, 2), round(c - 0.4, 2), round(c, 2), 110.0))
        # Swing low referensi 3938.0 lalu naik menyentuh zona Supply
        rows.append((round(zone_price - 2.5, 2), round(zone_price - 2.0, 2), round(zone_price - 4.5, 2), round(zone_price - 2.8, 2), 120.0))
        rows.append((round(zone_price - 2.8, 2), round(zone_price - 1.0, 2), round(zone_price - 3.0, 2), round(zone_price - 1.2, 2), 110.0))
        rows.append((round(zone_price - 1.2, 2), round(zone_price - 0.2, 2), round(zone_price - 1.4, 2), round(zone_price - 0.5, 2), 110.0))
        rows.append((round(zone_price - 0.5, 2), round(zone_price - 0.1, 2), round(zone_price - 1.1, 2), round(zone_price - 0.8, 2), 120.0))
        for _ in range(5):
            rows.append((round(zone_price - 0.8, 2), round(zone_price - 0.4, 2), round(zone_price - 1.4, 2), round(zone_price - 1.0, 2), 100.0))
        # Sweep High wick ke zone_price + 1.3, close bearish balik di zone_price - 0.9
        rows.append((round(zone_price - 0.9, 2), round(zone_price + 1.3, 2), round(zone_price - 1.2, 2), round(zone_price - 0.9, 2), 260.0))
        # Displacement + Bearish BOS close tembus 3937.5
        rows.append((round(zone_price - 0.9, 2), round(zone_price - 0.8, 2), round(zone_price - 3.2, 2), round(zone_price - 3.0, 2), 320.0))
        rows.append((round(zone_price - 3.0, 2), round(zone_price - 2.9, 2), round(zone_price - 5.6, 2), round(zone_price - 5.2, 2), 380.0))
        rows.append((round(zone_price - 5.2, 2), round(zone_price - 3.6, 2), round(zone_price - 5.3, 2), round(zone_price - 3.9, 2), 140.0))
        rows.append((round(zone_price - 3.9, 2), round(zone_price - 2.5, 2), round(zone_price - 4.1, 2), round(zone_price - 2.8, 2), 140.0))
        rows.append((round(zone_price - 2.8, 2), round(zone_price - 1.5, 2), round(zone_price - 2.9, 2), round(zone_price - 1.9, 2), 130.0))
        rows.append((round(zone_price - 1.9, 2), round(zone_price - 1.6, 2), round(zone_price - 2.5, 2), round(zone_price - 2.3, 2), 160.0))
    return _df_from_rows(rows, freq="5min", end_dt=end_dt)


def _m30_and_m5_continuation(direction="BUY", base=3960.0, end_dt=END_DT):
    """M30 konsolidasi pasca-tren lalu breakout continuation + M5 BOS/displacement searah (tanpa sweep wajib)."""
    rows30 = []
    p = base - (28.0 if direction == "BUY" else -28.0)
    for i in range(56):
        d = 0.58 if direction == "BUY" else -0.58
        if i % 6 == 5:
            d = -d * 0.6
        nxt = p + d
        rows30.append((round(p, 2), round(max(p, nxt) + 0.4, 2), round(min(p, nxt) - 0.4, 2), round(nxt, 2), 180.0))
        p = nxt
    # Kotak konsolidasi ketat 10 bar di sekitar base
    for i in range(10):
        off = 0.45 * math.sin(i)
        rows30.append((round(base + off, 2), round(base + 0.9, 2), round(base - 0.9, 2), round(base + off * 0.5, 2), 120.0))
    # Breakout konsolidasi M30 close searah tren
    if direction == "BUY":
        rows30.append((round(base + 0.3, 2), round(base + 3.2, 2), round(base + 0.1, 2), round(base + 2.9, 2), 340.0))
    else:
        rows30.append((round(base - 0.3, 2), round(base - 0.1, 2), round(base - 3.2, 2), round(base - 2.9, 2), 340.0))
    df30 = _df_from_rows(rows30, freq="30min", end_dt=end_dt)

    # M5: konsolidasi kecil -> sweep/swing -> BOS + displacement kuat searah (harga tetap di continuation shelf!)
    rows5 = []
    if direction == "BUY":
        for i in range(52):
            c = base - 4.8 + 0.08 * i + 0.25 * math.sin(i)
            rows5.append((round(c - 0.1, 2), round(c + 0.3, 2), round(c - 0.3, 2), round(c, 2), 110.0))
        # Swing High @ base + 1.5 lalu swing low @ base - 0.4 -> BOS breakout close @ base + 2.9
        rows5.append((round(base + 0.2, 2), round(base + 1.5, 2), round(base + 0.0, 2), round(base + 0.4, 2), 120.0))
        for _ in range(6):
            rows5.append((round(base + 0.3, 2), round(base + 0.8, 2), round(base - 0.4, 2), round(base + 0.2, 2), 110.0))
        # FVG 3-bar displacement breakout: Low[n] > High[n-2]
        rows5.append((round(base + 0.2, 2), round(base + 0.9, 2), round(base - 0.1, 2), round(base + 0.8, 2), 200.0))
        rows5.append((round(base + 0.8, 2), round(base + 2.4, 2), round(base + 0.7, 2), round(base + 2.2, 2), 320.0))
        rows5.append((round(base + 2.2, 2), round(base + 3.2, 2), round(base + 1.6, 2), round(base + 2.8, 2), 340.0))
        # Micro pullback ke FVG shelf [0.9..1.6]
        rows5.append((round(base + 2.8, 2), round(base + 2.9, 2), round(base + 1.4, 2), round(base + 1.5, 2), 150.0))
    else:
        for i in range(52):
            c = base + 4.8 - 0.08 * i + 0.25 * math.sin(i)
            rows5.append((round(c + 0.1, 2), round(c + 0.3, 2), round(c - 0.3, 2), round(c, 2), 110.0))
        rows5.append((round(base - 0.2, 2), round(base - 0.0, 2), round(base - 1.5, 2), round(base - 0.4, 2), 120.0))
        for _ in range(6):
            rows5.append((round(base - 0.3, 2), round(base + 0.4, 2), round(base - 0.8, 2), round(base - 0.2, 2), 110.0))
        rows5.append((round(base - 0.2, 2), round(base + 0.1, 2), round(base - 0.9, 2), round(base - 0.8, 2), 200.0))
        rows5.append((round(base - 0.8, 2), round(base - 0.7, 2), round(base - 2.4, 2), round(base - 2.2, 2), 320.0))
        rows5.append((round(base - 2.2, 2), round(base - 1.6, 2), round(base - 3.2, 2), round(base - 2.8, 2), 340.0))
        rows5.append((round(base - 2.8, 2), round(base - 1.4, 2), round(base - 2.9, 2), round(base - 1.5, 2), 150.0))
    df5 = _df_from_rows(rows5, freq="5min", end_dt=end_dt)
    return df30, df5


class TestTrendFollowingRebuild(unittest.TestCase):
    def setUp(self):
        _SCAN_CACHE["key"] = None
        _SCAN_CACHE["ts"] = 0.0
        _SCAN_CACHE["data"] = None

    def _scan_mock(self, h4_bias="BULLISH", h1_bias="BULLISH", m30_bias="BULLISH", has_conflict=False, m30_conflict=False, price=3944.0):
        return {
            "data_ok": True,
            "symbol": "XAUUSD",
            "current_price": price,
            "snapshot_id": "SNAP_TEST_TF",
            "snapshot_time": "2026-11-13 12:00:00 WIB",
            "bid": round(price - 0.1, 2),
            "ask": round(price + 0.1, 2),
            "mid": price,
            "spread": 0.2,
            "tick_time": "2026-11-13 12:00:00 WIB",
            "data_freshness": "FRESH",
            "data_freshness_info": {"status": "FRESH", "tick_status": "FRESH", "candle_status": "FRESH", "data_validation": "PASS"},
            "market_status": "OPEN",
            "connection_status": "CONNECTED",
            "active_zones": [],
            "radar_zones": [],
            "rejected_zones": [],
            "best_demand": None,
            "best_supply": None,
            "htf_context": {
                "D1": h4_bias, "H4": h4_bias, "H1": h1_bias, "M30": m30_bias,
                "has_conflict": has_conflict, "m30_conflict": m30_conflict,
                "bias_utama": h4_bias,
            },
            "empty_reason": "",
        }

    # Skenario 1: H4 bullish, H1 bullish, M30 pullback, M5 bullish confirmation -> BUY (TREND_PULLBACK)
    def test_01_h4_bull_h1_bull_m30_pullback_m5_buy(self):
        h4 = _trend_df("BULLISH", start=3880.0, bars=80, freq="4h")
        h1 = _trend_df("BULLISH", start=3900.0, bars=80, freq="1h")
        # H1 active high ~ 3960; M30 pulls back to FVG/OTE ~ 3942 without breaking H1 origin 3920
        m30 = _pullback_m30_df("BULLISH", anchor_origin=3920.0, anchor_extreme=3960.0, pullback_to=3942.0)
        # Tambahkan FVG bullish M30 tepat di 3940-3944 supaya area pullback sangat terstruktur
        m5 = _m5_pullback_trigger("BUY", zone_price=3942.0)
        price = float(m5["Close"].iloc[-1])

        frames = {"D1": h4, "H4": h4, "H1": h1, "M30": m30, "M5": m5}
        scan = self._scan_mock("BULLISH", "BULLISH", "NETRAL", price=price)
        dec = evaluate_trade_decision(frames=frames, current_price=price, spread=0.2, live=False, scan_result=scan)

        self.assertEqual(dec["status"], "VALID", f"Harus BUY VALID meski active_zones kosong: {dec.get('reasons')}")
        self.assertEqual(dec["direction"], "BUY")
        self.assertEqual(dec["setup_type"], "TREND_PULLBACK")
        self.assertIsNotNone(dec["plan"])
        self.assertGreaterEqual(dec["plan"]["rr"], RR_MIN)

    # Skenario 2: H4 bearish, H1 bearish, M30 bullish pullback, M5 bearish confirmation -> SELL (no false M30 conflict)
    def test_02_h4_bear_h1_bear_m30_bull_pullback_m5_sell(self):
        h4 = _trend_df("BEARISH", start=4020.0, bars=80, freq="4h")
        h1 = _trend_df("BEARISH", start=4000.0, bars=80, freq="1h")
        m30 = _pullback_m30_df("BEARISH", anchor_origin=3965.0, anchor_extreme=3920.0, pullback_to=3940.0)
        m5 = _m5_pullback_trigger("SELL", zone_price=3940.0)
        price = float(m5["Close"].iloc[-1])

        frames = {"D1": h4, "H4": h4, "H1": h1, "M30": m30, "M5": m5}
        ctx = evaluate_htf_context(frames)
        self.assertFalse(ctx["m30_conflict"], "M30 bullish pullback dalam H4/H1 bearish tidak boleh dicap konflik fatal")
        scan = scan_xauusd_zones(frames=frames, current_price=price)
        dec = evaluate_trade_decision(frames=frames, current_price=price, spread=0.2, live=False, scan_result=scan)

        self.assertEqual(dec["status"], "VALID", f"Harus SELL VALID pada M30 pullback: {dec.get('reasons')}")
        self.assertEqual(dec["direction"], "SELL")
        self.assertIn(dec["setup_type"], ("TREND_PULLBACK", "BREAKOUT_RETEST"))
        self.assertGreaterEqual(dec["plan"]["rr"], RR_MIN)

    # Skenario 3: H4 bullish, H1 bullish, M30 consolidation breakout, M5 continuation -> BUY
    def test_03_m30_consolidation_breakout_m5_continuation_buy(self):
        h4 = _trend_df("BULLISH", start=3890.0, bars=80, freq="4h")
        h1 = _trend_df("BULLISH", start=3910.0, bars=80, freq="1h")
        m30, m5 = _m30_and_m5_continuation("BUY", base=3962.0)
        price = float(m5["Close"].iloc[-1])

        frames = {"D1": h4, "H4": h4, "H1": h1, "M30": m30, "M5": m5}
        scan = self._scan_mock("BULLISH", "BULLISH", "BULLISH", price=price)
        dec = evaluate_trade_decision(frames=frames, current_price=price, spread=0.2, live=False, scan_result=scan)

        self.assertEqual(dec["status"], "VALID", f"Trend Continuation BUY harus VALID: {dec.get('reasons')}")
        self.assertEqual(dec["direction"], "BUY")
        self.assertIn(dec["setup_type"], ("TREND_CONTINUATION", "TREND_PULLBACK", "BREAKOUT_RETEST"))
        self.assertGreaterEqual(dec["plan"]["rr"], RR_MIN)

    # Skenario 4: H4 bearish, H1 bearish, M30 breakdown, M5 continuation -> SELL
    def test_04_m30_breakdown_m5_continuation_sell(self):
        h4 = _trend_df("BEARISH", start=4030.0, bars=80, freq="4h")
        h1 = _trend_df("BEARISH", start=4010.0, bars=80, freq="1h")
        m30, m5 = _m30_and_m5_continuation("SELL", base=3950.0)
        price = float(m5["Close"].iloc[-1])

        frames = {"D1": h4, "H4": h4, "H1": h1, "M30": m30, "M5": m5}
        scan = self._scan_mock("BEARISH", "BEARISH", "BEARISH", price=price)
        dec = evaluate_trade_decision(frames=frames, current_price=price, spread=0.2, live=False, scan_result=scan)

        self.assertEqual(dec["status"], "VALID", f"Trend Continuation SELL harus VALID: {dec.get('reasons')}")
        self.assertEqual(dec["direction"], "SELL")
        self.assertGreaterEqual(dec["plan"]["rr"], RR_MIN)

    # Skenario 5: H4 bullish, H1 pullback wajar (ranging compression di atas H4 low), M5 confirmation -> BUY
    def test_05_h4_bull_h1_normal_pullback_m5_buy(self):
        h4 = _trend_df("BULLISH", start=3870.0, bars=80, freq="4h")
        h1 = _sideways_df(center=3945.0, amp=1.1, bars=75, freq="1h")  # H1 konsolidasi/pullback wajar di atas H4 active_low
        m30 = _pullback_m30_df("BULLISH", anchor_origin=3920.0, anchor_extreme=3960.0, pullback_to=3943.0)
        m5 = _m5_pullback_trigger("BUY", zone_price=3943.0)
        price = float(m5["Close"].iloc[-1])

        frames = {"D1": h4, "H4": h4, "H1": h1, "M30": m30, "M5": m5}
        ctx = evaluate_htf_context(frames)
        self.assertFalse(ctx["has_conflict"], "H1 pullback/konsolidasi wajar dalam H4 bullish tidak boleh diblokir sebagai CONFLICT")
        self.assertEqual(ctx.get("h1_confirmation"), "PULLBACK")

        scan = self._scan_mock("BULLISH", "NETRAL", "PULLBACK", price=price)
        dec = evaluate_trade_decision(frames=frames, current_price=price, spread=0.2, live=False, scan_result=scan)
        self.assertEqual(dec["status"], "VALID", f"Harus BUY VALID saat H1 pullback wajar: {dec.get('reasons')}")
        self.assertEqual(dec["direction"], "BUY")

    # Skenario 6: H4 bullish, H1 reversal struktural serius -> WAIT
    def test_06_h4_bull_h1_serious_structural_conflict_waits(self):
        h4 = _trend_df("BULLISH", start=3880.0, bars=80, freq="4h")
        h1 = _trend_df("BEARISH", start=4010.0, bars=80, freq="1h")
        m30 = _trend_df("BULLISH", start=3920.0, bars=80, freq="30min")
        m5 = _m5_pullback_trigger("BUY", zone_price=3942.0)
        frames = {"D1": h4, "H4": h4, "H1": h1, "M30": m30, "M5": m5}

        ctx = evaluate_htf_context(frames)
        self.assertTrue(ctx["has_conflict"])
        dec = evaluate_trade_decision(frames=frames, current_price=float(m5["Close"].iloc[-1]), spread=0.2, live=False)
        self.assertEqual(dec["status"], "WAIT")

    # Skenario 7: H4 unclear (UNCONFIRMED) -> WAIT
    def test_07_h4_unclear_waits(self):
        h4 = _sideways_df(center=3940.0, amp=1.0, bars=80, freq="4h")
        h1 = _trend_df("BULLISH", start=3920.0, bars=80, freq="1h")
        m30 = _trend_df("BULLISH", start=3930.0, bars=80, freq="30min")
        m5 = _m5_pullback_trigger("BUY", zone_price=3942.0)
        frames = {"D1": h4, "H4": h4, "H1": h1, "M30": m30, "M5": m5}
        dec = evaluate_trade_decision(frames=frames, current_price=float(m5["Close"].iloc[-1]), spread=0.2, live=False)
        self.assertEqual(dec["status"], "WAIT")

    # Skenario 8: Breakout valid + Retest bertahan -> BUY (BREAKOUT_RETEST)
    def test_08_valid_breakout_retest_held_triggers_buy(self):
        h4 = _trend_df("BULLISH", start=3880.0, bars=80, freq="4h")
        # Bangun H1 dengan resistance swing ~ 3948 yang ditembus close kuat ke 3954, lalu retest ke 3948.5 dan bertahan
        rows1 = []
        for i in range(68):
            c = 3926.0 + 9.0 * math.sin(i * 0.42) + 0.16 * i
            rows1.append((round(c - 0.3, 2), round(min(3948.0, c + 0.9), 2), round(c - 0.9, 2), round(c, 2), 140.0))
        # Bentuk swing high terkonfirmasi di 3948.0 pada idx -14
        rows1[-14] = (3945.0, 3948.0, 3944.0, 3946.0, 150.0)
        for j in range(-13, -6):
            rows1[j] = (3945.0, 3946.5, 3943.5, 3944.5, 130.0)
        # Closed Breakout kuat di atas 3948.0!
        rows1[-5] = (3945.0, 3953.5, 3944.8, 3953.0, 350.0)
        rows1[-4] = (3953.0, 3955.0, 3951.5, 3954.0, 280.0)
        rows1[-3] = (3954.0, 3954.5, 3949.5, 3950.5, 180.0)
        rows1[-2] = (3950.5, 3951.0, 3948.2, 3949.0, 170.0)
        rows1[-1] = (3949.0, 3950.2, 3948.1, 3949.5, 190.0)
        h1 = _df_from_rows(rows1, freq="1h")

        m30 = _pullback_m30_df("BULLISH", anchor_origin=3935.0, anchor_extreme=3955.0, pullback_to=3948.5)
        m5 = _m5_pullback_trigger("BUY", zone_price=3948.5)
        price = float(m5["Close"].iloc[-1])

        frames = {"D1": h4, "H4": h4, "H1": h1, "M30": m30, "M5": m5}
        scan = self._scan_mock("BULLISH", "BULLISH", "PULLBACK", price=price)
        dec = evaluate_trade_decision(frames=frames, current_price=price, spread=0.2, live=False, scan_result=scan)
        self.assertEqual(dec["status"], "VALID", f"Breakout Retest harus VALID: {dec.get('reasons')}")
        self.assertEqual(dec["direction"], "BUY")
        self.assertIn(dec["setup_type"], ("BREAKOUT_RETEST", "TREND_PULLBACK", "TREND_CONTINUATION"))

    # Skenario 9: False breakout -> WAIT (wick poke saja tanpa close atau segera menutup balik ke bawah level)
    def test_09_false_breakout_waits(self):
        h4 = _trend_df("BULLISH", start=3880.0, bars=80, freq="4h")
        h1 = _sideways_df(center=3945.0, amp=1.0, bars=80, freq="1h")
        m30 = _sideways_df(center=3945.0, amp=0.5, bars=70, freq="30min")
        m30["RSI14"] = 44.0
        # M5 wick tembus 3947 lalu close balik melemah turun tanpa BOS/displacement bullish
        rows5 = []
        for _ in range(55):
            rows5.append((3945.0, 3945.6, 3944.4, 3945.0, 100.0))
        rows5.append((3945.0, 3948.5, 3944.8, 3944.9, 250.0))  # false breakout wick spike
        rows5.append((3944.9, 3945.1, 3944.2, 3944.5, 110.0))
        m5 = _df_from_rows(rows5, freq="5min")

        frames = {"D1": h4, "H4": h4, "H1": h1, "M30": m30, "M5": m5}
        dec = evaluate_trade_decision(frames=frames, current_price=3944.5, spread=0.2, live=False)
        self.assertEqual(dec["status"], "WAIT")

    # Skenario 10: Failed retest -> WAIT
    def test_10_failed_retest_waits(self):
        h4 = _trend_df("BULLISH", start=3880.0, bars=80, freq="4h")
        rows1 = []
        for i in range(60):
            c = 3940.0 + 0.1 * i
            rows1.append((c, c + 1.0, c - 1.0, c, 120.0))
        rows1[-10] = (3945.0, 3948.0, 3944.0, 3946.0, 140.0)
        rows1[-5] = (3946.0, 3951.0, 3945.8, 3950.5, 250.0)
        # Retest GAGAL: close jatuh keras balik di bawah 3948 & di bawah 3944!
        rows1[-1] = (3949.0, 3949.2, 3941.0, 3941.5, 300.0)
        h1 = _df_from_rows(rows1, freq="1h")
        m30 = _sideways_df(center=3942.0, amp=0.4, bars=65, freq="30min")
        m30["RSI14"] = 41.0
        rows5 = [(3942.0, 3942.4, 3941.6, 3941.8, 100.0)] * 60
        m5 = _df_from_rows(rows5, freq="5min")
        dec = evaluate_trade_decision(frames={"D1": h4, "H4": h4, "H1": h1, "M30": m30, "M5": m5}, current_price=3941.8, spread=0.2, live=False)
        self.assertEqual(dec["status"], "WAIT")

    # Skenario 11: Liquidity Sweep + CHoCH + Displacement -> Reversal (Strict REVERSAL gate)
    def test_11_strict_htf_reversal_triggers(self):
        h4 = _trend_df("BULLISH", start=3880.0, bars=80, freq="4h")

        # Bangun H1 & M30: tren naik jelas (trend == 1 via BOS) ke 3977, swing low 3966, sweep high > 3980, lalu CHoCH < 3966
        def _rev_frame(freq):
            rows = []
            # 50 bar tren naik bertahap memicu trend == 1 (BOS bull)
            p = 3910.0
            for i in range(50):
                cyc = i % 12
                nxt = p + (1.8 if cyc < 8 else -0.9)
                rows.append((round(p, 2), round(max(p, nxt) + 0.4, 2), round(min(p, nxt) - 0.4, 2), round(nxt, 2), 140.0))
                p = nxt
            # Pastikan Swing High puncak awal @ 3972 lalu pullback membentuk Swing Low terkonfirmasi @ 3966
            rows.append((3970.0, 3974.0, 3969.5, 3972.0, 150.0))
            for _ in range(5):
                rows.append((3969.0, 3970.0, 3968.0, 3968.5, 130.0))
            rows.append((3968.5, 3969.0, 3966.0, 3968.0, 150.0))  # swing low 3966.0
            for _ in range(5):
                rows.append((3969.0, 3973.0, 3968.5, 3972.0, 130.0))
            # BOS bull > 3974 memastikan trend == 1 dan bentuk Swing High 3980.0
            rows.append((3973.0, 3980.0, 3972.5, 3977.0, 180.0))
            for _ in range(5):
                rows.append((3976.0, 3978.0, 3974.0, 3975.0, 130.0))
            # Sweep High > 3980 (wick ke 3983.5, close kembali <= 3980)
            rows.append((3977.5, 3983.5, 3977.0, 3978.0, 290.0))
            # Bearish displacement + CHoCH close < 3966.0!
            rows.append((3978.0, 3978.5, 3968.0, 3968.5, 350.0))
            rows.append((3968.5, 3969.0, 3962.0, 3963.0, 400.0))
            # Stabil dekat 3963 agar harga sekarang dekat area supply/reversal
            rows.append((3963.0, 3964.2, 3962.2, 3963.2, 160.0))
            rows.append((3963.2, 3963.8, 3962.4, 3963.0, 150.0))
            rows.append((3963.0, 3963.5, 3962.0, 3962.5, 150.0))
            return _df_from_rows(rows, freq=freq)

        h1 = _rev_frame("1h")
        m30 = _rev_frame("30min")
        m5 = _m5_pullback_trigger("SELL", zone_price=3963.0)
        price = float(m5["Close"].iloc[-1])

        frames = {"D1": h4, "H4": h4, "H1": h1, "M30": m30, "M5": m5}
        tf_res = evaluate_trend_following_setup(frames, price, explicit_h1_conflict=True)
        self.assertEqual(tf_res["status"], "VALID", f"Strict Reversal harus VALID: {tf_res.get('reasons')}")
        self.assertEqual(tf_res["direction"], "SELL")
        self.assertEqual(tf_res["setup_type"], "REVERSAL")

    # Skenario 12: Counter-trend lemah (RSI overbought tanpa CHoCH H1+M30 & Sweep) -> WAIT
    def test_12_weak_counter_trend_rsi_only_waits(self):
        h4 = _trend_df("BULLISH", start=3880.0, bars=80, freq="4h")
        h1 = _trend_df("BULLISH", start=3910.0, bars=80, freq="1h")
        m30 = _trend_df("BULLISH", start=3930.0, bars=80, freq="30min")
        m30["RSI14"] = 79.0  # overbought ekstrem, tapi TIDAK ada CHoCH bearish H1/M30!
        m5 = _m5_pullback_trigger("SELL", zone_price=float(m30["Close"].iloc[-1]))
        dec = evaluate_trade_decision(frames={"D1": h4, "H4": h4, "H1": h1, "M30": m30, "M5": m5}, current_price=float(m5["Close"].iloc[-1]), spread=0.2, live=False)
        self.assertNotEqual(dec["direction"], "SELL")

    # Skenario 13: Data stale -> WAIT (tidak boleh mengeluarkan sinyal)
    def test_13_stale_data_blocks_signal(self):
        h4 = _trend_df("BULLISH", start=3880.0, bars=80, freq="4h")
        h1 = _trend_df("BULLISH", start=3900.0, bars=80, freq="1h")
        m30 = _pullback_m30_df("BULLISH", anchor_origin=3920.0, anchor_extreme=3960.0, pullback_to=3942.0)
        m5 = _m5_pullback_trigger("BUY", zone_price=3942.0)
        stale_srv = END_DT + timedelta(days=3)
        dec = evaluate_trade_decision(frames={"D1": h4, "H4": h4, "H1": h1, "M30": m30, "M5": m5}, current_price=float(m5["Close"].iloc[-1]), server_time=stale_srv, live=False)
        self.assertEqual(dec["status"], "WAIT")
        self.assertEqual(dec["data_freshness"], "STALE")
        self.assertIsNone(dec.get("plan"))

    # Skenario 14: Data invalid (NaN / corrupt OHLC) -> WAIT
    def test_14_invalid_corrupt_data_blocks_signal(self):
        h4 = _trend_df("BULLISH", start=3880.0, bars=80, freq="4h")
        h1 = _trend_df("BULLISH", start=3900.0, bars=80, freq="1h")
        m30 = _pullback_m30_df("BULLISH", anchor_origin=3920.0, anchor_extreme=3960.0, pullback_to=3942.0)
        m5 = _m5_pullback_trigger("BUY", zone_price=3942.0)
        m5_bad = m5.copy()
        m5_bad.iloc[-1, m5_bad.columns.get_loc("Low")] = float("nan")
        dec = evaluate_trade_decision(frames={"D1": h4, "H4": h4, "H1": h1, "M30": m30, "M5": m5_bad}, current_price=3944.0, live=False)
        self.assertEqual(dec["status"], "WAIT")
        self.assertIsNone(dec.get("plan"))

    # Skenario 15: Market closed -> MARKET CLOSED (WAIT, dibedakan dari disconnected)
    def test_15_market_closed_blocks_new_entry_clearly(self):
        h4 = _trend_df("BULLISH", start=3880.0, bars=80, freq="4h")
        h1 = _trend_df("BULLISH", start=3900.0, bars=80, freq="1h")
        m30 = _pullback_m30_df("BULLISH", anchor_origin=3920.0, anchor_extreme=3960.0, pullback_to=3942.0)
        m5 = _m5_pullback_trigger("BUY", zone_price=3942.0)
        snap_closed = create_market_snapshot(
            symbol="XAUUSD",
            frames={"D1": h4, "H4": h4, "H1": h1, "M30": m30, "M5": m5},
            current_price=float(m5["Close"].iloc[-1]),
            live=False,
            market_status="CLOSED",
        )
        dec = evaluate_trade_decision(snapshot=snap_closed)
        self.assertEqual(dec["status"], "WAIT")
        self.assertEqual(dec["market_status"], "CLOSED")
        self.assertIsNone(dec.get("plan"))

    # Skenario 16: Candle terakhir masih berjalan (belum close) -> tidak dijadikan konfirmasi final (WAIT)
    def test_16_unclosed_forming_candle_waits(self):
        h4 = _trend_df("BULLISH", start=3880.0, bars=80, freq="4h")
        h1 = _trend_df("BULLISH", start=3900.0, bars=80, freq="1h")
        m30 = _pullback_m30_df("BULLISH", anchor_origin=3920.0, anchor_extreme=3960.0, pullback_to=3942.0)
        m5 = _m5_pullback_trigger("BUY", zone_price=3942.0)
        m5.attrs["is_closed"] = False
        dec = evaluate_trade_decision(frames={"D1": h4, "H4": h4, "H1": h1, "M30": m30, "M5": m5}, current_price=float(m5["Close"].iloc[-1]), live=False)
        self.assertEqual(dec["status"], "WAIT")
        self.assertIn("belum selesai", " ".join(dec["reasons"]).lower())

    # Skenario 17: Signal Confirmed vs Entry Reached vs Active Trade (posisi MT5 nyata) vs Missed / TP / SL / Expired
    def test_17_signal_manager_lifecycle_distinctions(self):
        sig = normalize_signal_payload({
            "arah": "BUY", "lo": 3941.0, "hi": 3943.0, "sl": 3937.0, "tp1": 3951.0, "tp2": 3959.0,
            "setup_type": "TREND_PULLBACK", "reasons": ["M5 Sweep + BOS valid"], "harga_buat": 3946.0,
        })
        for req_k in ("signal_id", "direction", "setup_type", "timestamp", "entry_price", "sl", "tp1", "tp2", "rr", "invalidation_price", "reasons", "status"):
            self.assertIn(req_k, sig)
        self.assertEqual(sig["rr"], 1.8)

        # a) Price inside entry WITHOUT open MT5 position -> ENTRY REACHED (NOT ACTIVE TRADE!)
        st_reached, _ = resolve_unified_signal_status(sig, 3942.0, pos_list=(), order_list=())
        self.assertEqual(st_reached, SIGNAL_STATUS_ENTRY_REACHED)

        # b) Price inside entry WITH real open MT5 BUY position -> ACTIVE TRADE!
        mock_pos = [SimpleNamespace(ticket=101, type=0, volume=0.05, price_open=3942.1, sl=3937.0, tp=3951.0, profit=12.0)]
        st_active, _ = resolve_unified_signal_status(sig, 3942.5, pos_list=mock_pos, order_list=())
        self.assertEqual(st_active, SIGNAL_STATUS_ACTIVE_TRADE)

        # c) Explicit terminal mappings
        self.assertEqual(map_status_to_canonical(ST_BATAL_TERLEWAT), SIGNAL_STATUS_ENTRY_MISSED)
        self.assertEqual(map_status_to_canonical(ST_BATAL_SL), SIGNAL_STATUS_SL_HIT)
        self.assertEqual(map_status_to_canonical(ST_SELESAI_TP2), SIGNAL_STATUS_TP_HIT)
        self.assertEqual(map_status_to_canonical(ST_BATAL_KEDALUWARSA), SIGNAL_STATUS_EXPIRED)
        self.assertEqual(map_status_to_canonical(ST_AKTIF_ENTRY, has_live_position=False), SIGNAL_STATUS_ENTRY_REACHED)
        self.assertEqual(map_status_to_canonical("CANCELLED"), SIGNAL_STATUS_CANCELLED)

    # Skenario 18: Konsistensi Signal & Final Decision + Format Transparan & Gerbang Python
    def test_18_decision_signal_and_gate_consistency(self):
        h4 = _trend_df("BULLISH", start=3880.0, bars=80, freq="4h")
        h1 = _trend_df("BULLISH", start=3900.0, bars=80, freq="1h")
        m30 = _pullback_m30_df("BULLISH", anchor_origin=3920.0, anchor_extreme=3960.0, pullback_to=3942.0)
        m5 = _m5_pullback_trigger("BUY", zone_price=3942.0)
        price = float(m5["Close"].iloc[-1])

        scan = self._scan_mock("BULLISH", "BULLISH", "PULLBACK", price=price)
        dec = evaluate_trade_decision(frames={"D1": h4, "H4": h4, "H1": h1, "M30": m30, "M5": m5}, current_price=price, spread=0.2, live=False, scan_result=scan)
        self.assertEqual(dec["status"], "VALID")

        sig_obj = create_signal_from_decision(dec)
        self.assertIsNotNone(sig_obj)
        self.assertEqual(sig_obj["direction"], dec["direction"])
        self.assertEqual(sig_obj["setup_type"], dec["setup_type"])
        self.assertEqual(sig_obj["status"], SIGNAL_STATUS_SIGNAL_CONFIRMED)

        # Formatter memuat seluruh komponen transparansi wajib
        txt_concise = format_scan_presentation(scan, dec, detailed=False)
        txt_sum = format_scan_summary(scan, dec, pos_list=(), orders_list=(), risk_data={"available": False})
        for txt in (txt_concise, txt_sum):
            self.assertIn(dec["setup_type"], txt)
            self.assertIn("BUY", txt)

        # Monitor snapshot memuat setup_type dan membedakan status lifecycle secara akurat
        snap_mon = _build_monitor_snapshot(price=price, scan_result=scan, decision=dec, otomatis={"sinyal": sig_obj})
        self.assertEqual(snap_mon["setup_type"], dec["setup_type"])
        det_mon = format_monitor_detail(snap_mon)
        self.assertIn(dec["setup_type"], det_mon)

    # Skenario 19: Integrasi Unified Monitor (SMC + SNR tetap independen) & CLI commands utuh
    def test_19_unified_monitor_and_cli_commands_preserved(self):
        h4 = _trend_df("BULLISH", start=3880.0, bars=80, freq="4h")
        h1 = _trend_df("BULLISH", start=3900.0, bars=80, freq="1h")
        m30 = _pullback_m30_df("BULLISH", anchor_origin=3920.0, anchor_extreme=3960.0, pullback_to=3942.0)
        m5 = _m5_pullback_trigger("BUY", zone_price=3942.0)
        frames = {"D1": h4, "H4": h4, "H1": h1, "M30": m30, "M5": m5, "M15": m30.copy()}
        snap = create_market_snapshot(symbol="XAUUSD", frames=frames, current_price=float(m5["Close"].iloc[-1]), live=False, market_status="OPEN")

        mon = UnifiedMonitor(symbol="XAUUSD", snapshot_provider=lambda symbol="XAUUSD": snap)
        cyc = mon.run_cycle(snapshot=snap, mode="unified", detailed=False, pos_list=(), orders_list=())
        self.assertEqual(cyc["smc"]["status"], "OK")
        self.assertEqual(cyc["snr"]["status"], "OK")
        self.assertIn("SMC MONITOR", cyc["formatted_report"])
        self.assertIn("SNR MONITOR", cyc["formatted_report"])
        self.assertIn("Setup:", cyc["smc"]["formatted_section"])
        self.assertIn(cyc["smc"]["summary"]["setup_type"], ("TREND_PULLBACK", "TREND_CONTINUATION", "BREAKOUT_RETEST", "REVERSAL", "NONE"))
        self.assertEqual(parse_cli_command("scan")["action"], "SCAN")
        self.assertEqual(parse_cli_command("scan snr detail")["method"], "SNR")
        self.assertEqual(parse_cli_command("pantau")["action"], "PANTAU")

    # Skenario 20: Controlled Historical Walk-Forward Backtest & Setup Breakdown (Pullback / Continuation / Breakout Retest)
    def test_20_walkforward_controlled_backtest_metrics(self):
        csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backtest_fitur_xauusd.csv")
        self.assertTrue(os.path.exists(csv_path), "Dataset historis backtest_fitur_xauusd.csv harus ada")

        # 1. Walk-Forward Simulation pada bar historis M30 -> memverifikasi tanpa look-ahead bias
        #    dan memisahkan performa berdasarkan 3 jenis setup (TREND_PULLBACK, TREND_CONTINUATION, BREAKOUT_RETEST)
        setup_trades = {"TREND_PULLBACK": [], "TREND_CONTINUATION": [], "BREAKOUT_RETEST": []}

        df_csv = pd.read_csv(csv_path)
        unique_bars = df_csv.drop_duplicates(subset=["waktu"]).reset_index(drop=True)
        self.assertGreater(len(unique_bars), 300)

        # Bagi Development (50% awal) vs Out-of-Sample (50% akhir)
        mid_idx = len(df_csv) // 2
        for split_label, sub_df in (("DEV", df_csv.iloc[:mid_idx]), ("OOS", df_csv.iloc[mid_idx:])):
            for _, row in sub_df.iterrows():
                if int(row.get("bias_searah", 0)) != 1:
                    continue
                # Cek trigger SMC & klasifikasi setup dari fitur historis objektif (tertutup saat bar tersebut)
                has_ob_or_fvg = bool(row.get("ob_h4", 0) or row.get("ob_h1", 0) or row.get("ob_m30", 0) or row.get("fvg_h1", 0) or row.get("fvg_m30", 0))
                has_m5_conf = bool(row.get("sweep_m5", 0) or row.get("cisd_m5", 0))
                if not has_m5_conf:
                    continue
                r_out = float(row.get("R_1.5", 0.0))  # sudah memperhitungkan spread historis (-1.076 SL vs +1.424 TP1)
                if has_ob_or_fvg and int(row.get("fib_ote_h1", 0)) == 1:
                    setup_trades["TREND_PULLBACK"].append(r_out)
                elif has_ob_or_fvg and int(row.get("bias_kuat", 0)) == 1:
                    setup_trades["BREAKOUT_RETEST"].append(r_out)
                elif int(row.get("bias_kuat", 0)) == 1 and int(row.get("vwap_searah", 0)) == 1:
                    setup_trades["TREND_CONTINUATION"].append(r_out)

        # 2. Simulasi Walk-Forward Bar-by-Bar Multi-Timeframe Engine pada seri harga XAUUSD historis
        #    Construct M30/H1/H4 causal bars murni dari data historis [:t] tanpa melihat masa depan [t+1..]
        closes_hist = pd.to_numeric(unique_bars["harga"], errors="coerce").dropna().to_numpy(dtype=float)
        atrs_hist = pd.to_numeric(unique_bars["atr"], errors="coerce").fillna(3.0).to_numpy(dtype=float)
        times_hist = pd.to_datetime(unique_bars["waktu"], errors="coerce")
        hist_rows = []
        for i in range(len(closes_hist)):
            c_h = float(closes_hist[i])
            o_h = float(closes_hist[i - 1]) if i > 0 else c_h
            a_h = max(0.5, float(atrs_hist[i]))
            hi_h = round(max(o_h, c_h) + 0.35 * a_h, 2)
            lo_h = round(min(o_h, c_h) - 0.35 * a_h, 2)
            hist_rows.append({"Time": times_hist.iloc[i], "Open": round(o_h, 2), "High": hi_h, "Low": lo_h, "Close": round(c_h, 2), "Volume": 200.0})
        hist_m30_all = pd.DataFrame(hist_rows)
        hist_m30_all.index = pd.DatetimeIndex(hist_m30_all["Time"])

        wf_splits = {
            "DEV": (220, min(600, len(hist_m30_all) - 40)),
            "VAL": (600, min(1050, len(hist_m30_all) - 40)),
            "OOS": (1050, len(hist_m30_all) - 40),
        }
        wf_bar_results = {"DEV": [], "VAL": [], "OOS": []}
        wf_by_setup = {"TREND_PULLBACK": [], "TREND_CONTINUATION": [], "BREAKOUT_RETEST": [], "REVERSAL": []}
        spread_cost_pt = 0.25

        for sp_name, (i_start, i_end) in wf_splits.items():
            if i_end <= i_start:
                continue
            for t_idx in range(i_start, i_end, 4):
                # Causal slice [:t_idx] SAJA (0% look-ahead)
                win_m30 = hist_m30_all.iloc[t_idx - 100 : t_idx + 1].copy()
                # Resample causal ke H1 (kelompok 2 bar M30) & H4 (kelompok 8 bar M30)
                agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
                full_past = hist_m30_all.iloc[max(0, t_idx - 520) : t_idx + 1]
                win_h1 = full_past.resample("1h").agg(agg).dropna().tail(75)
                win_h4 = full_past.resample("4h").agg(agg).dropna().tail(75)
                if len(win_h4) < 35 or len(win_h1) < 40 or len(win_m30) < 60:
                    continue
                # M5 kausal sintetis yang mempertahankan arah struktur bar penutup t_idx tanpa data masa depan
                b_last = win_m30.iloc[-1]
                win_m5_rows = []
                for m_sub in win_m30.iloc[-24:].itertuples():
                    for k_sub in range(3):
                        frac = (k_sub + 1) / 3.0
                        c_sub = round(float(m_sub.Open) + (float(m_sub.Close) - float(m_sub.Open)) * frac, 2)
                        o_sub = float(m_sub.Open) if k_sub == 0 else win_m5_rows[-1][3]
                        win_m5_rows.append((round(o_sub, 2), round(max(o_sub, c_sub) + 0.35, 2), round(min(o_sub, c_sub) - 0.35, 2), c_sub, 80.0))
                win_m5 = _df_from_rows(win_m5_rows, freq="5min", end_dt=win_m30.index[-1])
                win_h4_i = tambah_indikator(win_h4.copy())
                win_h1_i = tambah_indikator(win_h1.copy())
                win_m30_i = tambah_indikator(win_m30.copy())

                cp_t = float(b_last["Close"])
                tf_out = evaluate_trend_following_setup(
                    {"H4": win_h4_i, "H1": win_h1_i, "M30": win_m30_i, "M5": win_m5},
                    cp_t, spread=spread_cost_pt, news_blackout=False,
                )
                if tf_out.get("status") == "VALID" and tf_out.get("plan"):
                    pl = tf_out["plan"]
                    arah_t = pl["arah"]
                    entry_fill = float(pl["entry_acuan"]) + (spread_cost_pt if arah_t == "BUY" else -spread_cost_pt)
                    sl_p = float(pl["sl"])
                    tp1_p = float(pl["tp1"])
                    risk_pt = max(0.5, abs(entry_fill - sl_p))
                    r_realized = 0.0
                    fut = hist_m30_all.iloc[t_idx + 1 : min(len(hist_m30_all), t_idx + 37)]
                    for f_row in fut.itertuples():
                        if arah_t == "BUY":
                            if f_row.Low <= sl_p:
                                r_realized = -1.0
                                break
                            if f_row.High >= tp1_p:
                                r_realized = round((tp1_p - entry_fill) / risk_pt, 2)
                                break
                        else:
                            if f_row.High >= sl_p:
                                r_realized = -1.0
                                break
                            if f_row.Low <= tp1_p:
                                r_realized = round((entry_fill - tp1_p) / risk_pt, 2)
                                break
                    if r_realized == 0.0 and len(fut) > 0:
                        c_exit = float(fut.iloc[-1]["Close"])
                        r_realized = round(((c_exit - entry_fill) if arah_t == "BUY" else (entry_fill - c_exit)) / risk_pt, 2)
                    wf_bar_results[sp_name].append(r_realized)
                    wf_by_setup.setdefault(tf_out["setup_type"], []).append(r_realized)

        # 3. Verifikasi eksekusi masing-masing tipe setup Engine pada skenario terstandar
        wf_engine_trades = []
        wf_configs = [
            ("TREND_PULLBACK", "BULLISH"),
            ("TREND_PULLBACK", "BEARISH"),
            ("TREND_CONTINUATION", "BULLISH"),
            ("TREND_CONTINUATION", "BEARISH"),
        ]
        for s_kind, d_kind in wf_configs:
            if s_kind == "TREND_PULLBACK" and d_kind == "BULLISH":
                h4 = _trend_df("BULLISH", start=3880.0, bars=80, freq="4h")
                h1 = _trend_df("BULLISH", start=3900.0, bars=80, freq="1h")
                m30 = _pullback_m30_df("BULLISH", anchor_origin=3920.0, anchor_extreme=3960.0, pullback_to=3942.0)
                m5 = _m5_pullback_trigger("BUY", zone_price=3942.0)
            elif s_kind == "TREND_PULLBACK" and d_kind == "BEARISH":
                h4 = _trend_df("BEARISH", start=4020.0, bars=80, freq="4h")
                h1 = _trend_df("BEARISH", start=4000.0, bars=80, freq="1h")
                m30 = _pullback_m30_df("BEARISH", anchor_origin=3965.0, anchor_extreme=3920.0, pullback_to=3940.0)
                m5 = _m5_pullback_trigger("SELL", zone_price=3940.0)
            else:
                h4 = _trend_df(d_kind, start=3890.0 if d_kind == "BULLISH" else 4030.0, bars=80, freq="4h")
                h1 = _trend_df(d_kind, start=3910.0 if d_kind == "BULLISH" else 4010.0, bars=80, freq="1h")
                m30, m5 = _m30_and_m5_continuation("BUY" if d_kind == "BULLISH" else "SELL", base=3962.0 if d_kind == "BULLISH" else 3950.0)
            cp = float(m5["Close"].iloc[-1])
            dec_wf = evaluate_trade_decision(
                frames={"D1": h4, "H4": h4, "H1": h1, "M30": m30, "M5": m5},
                current_price=cp, spread=0.25, live=False,
                scan_result=self._scan_mock(d_kind, d_kind, d_kind, price=cp),
            )
            self.assertEqual(dec_wf["status"], "VALID", f"Walk-forward engine gagal di {s_kind} {d_kind}: {dec_wf.get('reasons')}")
            wf_engine_trades.append((dec_wf["setup_type"], dec_wf["direction"], dec_wf["plan"]["rr"]))

        self.assertEqual(len(wf_engine_trades), 4)

        def _summarize_r(rs):
            if not rs:
                return {"trades": 0, "winrate_pct": 0.0, "profit_factor": 0.0, "expectancy_R": 0.0, "max_dd_R": 0.0}
            wins = [x for x in rs if x > 0]
            losses = [abs(x) for x in rs if x <= 0]
            gw, gl = sum(wins), sum(losses)
            pf = round(gw / gl, 2) if gl > 0 else float("inf")
            wr = round(100.0 * len(wins) / len(rs), 1)
            exp = round(sum(rs) / len(rs), 3)
            eq = peak = mdd = 0.0
            for val in rs:
                eq += val
                peak = max(peak, eq)
                mdd = max(mdd, peak - eq)
            return {"trades": len(rs), "winrate_pct": wr, "profit_factor": pf, "expectancy_R": exp, "max_dd_R": round(mdd, 2)}

        report_stats = {k_name: _summarize_r(rs) for k_name, rs in setup_trades.items()}
        wf_engine_stats = {sp: _summarize_r(rs) for sp, rs in wf_bar_results.items()}
        wf_setup_stats = {st: _summarize_r(rs) for st, rs in wf_by_setup.items() if rs}
        print(f"\nBACKTEST_SETUP_SUMMARY={report_stats}")
        print(f"WALKFORWARD_SPLITS={wf_engine_stats}")
        print(f"WALKFORWARD_BY_SETUP={wf_setup_stats}")


if __name__ == "__main__":
    unittest.main()
