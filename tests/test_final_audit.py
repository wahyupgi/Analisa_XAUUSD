# -*- coding: utf-8 -*-
"""
tests/test_final_audit.py
Unit test suite untuk FINAL AUDIT:
1. Python memiliki radar zone dan Gemini tidak boleh mengubah harganya.
2. Python tidak memiliki zona tetapi Gemini mencoba membuat zona -> NO RADAR ZONE DETECTED.
3. H4 NETRAL dengan diagnostic reason yang spesifik.
4. H4 berubah bullish.
5. H4 berubah bearish.
6. Pending order MT5 tidak tersedia -> DATA UNAVAILABLE.
7. Pending order memiliki SL kosong -> BELUM DISET (peringatan pasang SL).
8. Jumlah ticket MT5 tidak sesuai dengan jumlah yang dilaporkan dicegah / konsisten.
9. Zona RADAR tidak otomatis menjadi READY.
10. Zona READY belum otomatis menjadi CONFIRMED.
"""
import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

import modules.zone_scanner as zs
from modules.zone_scanner import (
    Zone, ZoneTracker, scan_xauusd_zones, evaluate_trade_decision,
    detect_market_regime, evaluate_htf_context, format_scan_presentation,
    detect_early_setups, plan_pending_order
)
import modules.ai_gemini as ag
from modules.ai_gemini import sanitasi_output_gemini, ekstrak_target_ai
import modules.mt5_conn as mt5_conn
from modules.mt5_conn import cek_posisi_terbuka_mt5, analisa_akun_dan_lot, tambah_indikator
from tests.test_zone_scanner_validation import (
    _make_df, _make_h4_bullish, _make_h4_bearish,
    _make_h1_with_high_target, _make_h1_with_low_target,
    _make_m30_bullish, _make_test_zone
)


class TestFinalAudit(unittest.TestCase):

    def setUp(self):
        self._orig_load = zs._load_frames
        zs._load_frames = lambda ttl=10.0: None

    def tearDown(self):
        zs._load_frames = self._orig_load

    # -------------------------------------------------------------------------
    # 1. Python memiliki radar zone dan Gemini tidak boleh mengubah harganya
    # -------------------------------------------------------------------------
    def test_01_python_radar_zone_gemini_cannot_alter_prices(self):
        """Python memiliki radar zone dan Gemini tidak boleh mengubah harganya."""
        mock_scan = {
            "symbol": "XAUUSD", "current_price": 2020.0,
            "radar_zones": [
                {"zone_id": "H1_DEMAND_1", "zone_type": "DEMAND", "price_low": 2000.0, "price_high": 2004.0, "timeframe": "H1", "grade": "A", "invalidation_price": 1997.0, "status": "RADAR"},
                {"zone_id": "H1_SUPPLY_1", "zone_type": "SUPPLY", "price_low": 2030.0, "price_high": 2034.0, "timeframe": "H1", "grade": "A", "invalidation_price": 2037.0, "status": "RADAR"},
            ],
            "active_zones": [],
            "best_demand": {"zone_id": "H1_DEMAND_1", "zone_type": "DEMAND", "price_low": 2000.0, "price_high": 2004.0, "timeframe": "H1", "grade": "A", "sl_recommended": 1997.0, "invalidation_price": 1997.0},
            "best_supply": {"zone_id": "H1_SUPPLY_1", "zone_type": "SUPPLY", "price_low": 2030.0, "price_high": 2034.0, "timeframe": "H1", "grade": "A", "sl_recommended": 2037.0, "invalidation_price": 2037.0},
            "htf_context": {"H4": "NETRAL", "H1": "BULLISH", "has_conflict": False},
            "data_ok": True,
        }
        dec_wait = {"status": "WAIT", "direction": None, "reasons": ["H4 Netral"]}

        # Gemini berhalusinasi mengubah angka zona (misal demand jadi 2012.00-2015.00 dan supply jadi 2045.00)
        gemini_raw = """
Analisa: Kami melihat zona demand di 2012.00 - 2015.00 dan supply di 2045.00 - 2050.00.
[RADAR_TARGET]
ZONA_ATAS: 2045.00-2050.00 (batas_jebol: 2055.00)
ZONA_BAWAH: 2012.00-2015.00 (batas_jebol: 2008.00)
SINYAL_AKTIF: WAIT | ENTRY: 0.00-0.00 | SL: 0.00 | TP1: 0.00 | TP2: 0.00
[/RADAR_TARGET]
"""
        sanitized = sanitasi_output_gemini(gemini_raw, dec=dec_wait, scan=mock_scan)

        # Angka zona di [RADAR_TARGET] WAJIB dikoreksi dan dikunci ke angka Python (2000.00-2004.00 dan 2030.00-2034.00)
        self.assertIn("ZONA_BAWAH: 2000.00-2004.00", sanitized)
        self.assertIn("ZONA_ATAS: 2030.00-2034.00", sanitized)
        self.assertNotIn("2012.00-2015.00", sanitized)
        self.assertNotIn("2045.00-2050.00", sanitized)

    # -------------------------------------------------------------------------
    # 2. Python tidak memiliki zona tetapi Gemini mencoba membuat zona
    # -------------------------------------------------------------------------
    def test_02_python_no_zone_gemini_tries_to_create_zone(self):
        """Python tidak memiliki zona tetapi Gemini mencoba membuat zona -> NO RADAR ZONE DETECTED."""
        mock_scan_empty = {
            "symbol": "XAUUSD", "current_price": 2020.0,
            "radar_zones": [], "active_zones": [],
            "best_demand": None, "best_supply": None,
            "htf_context": {"H4": "NETRAL", "H1": "NETRAL", "has_conflict": False},
            "empty_reason": "Tidak ada zona lolos validasi", "data_ok": True,
        }
        dec_wait = {"status": "WAIT", "direction": None, "reasons": ["Tidak ada zona"]}

        # Gemini berhalusinasi membuat zona sendiri
        gemini_raw = """
Saya menemukan zona Demand baru yang bagus di 2010.00 - 2012.00.
- Kandidat Demand: H1 DEMAND 2010.00 - 2012.00
[RADAR_TARGET]
ZONA_BAWAH: 2010.00-2012.00 (batas_jebol: 2005.00)
SINYAL_AKTIF: WAIT | ENTRY: 0.00-0.00 | SL: 0.00 | TP1: 0.00 | TP2: 0.00
[/RADAR_TARGET]
"""
        sanitized = sanitasi_output_gemini(gemini_raw, dec=dec_wait, scan=mock_scan_empty)

        # Verifikasi: zona halusinasi dibersihkan dan diganti dengan NO RADAR ZONE DETECTED
        self.assertIn("NO RADAR ZONE DETECTED", sanitized)
        self.assertNotIn("2010.00-2012.00", sanitized)

        # Verifikasi pada format_scan_presentation
        pres = format_scan_presentation(scan=mock_scan_empty, dec=dec_wait)
        self.assertIn("NO RADAR ZONE DETECTED.", pres)

    # -------------------------------------------------------------------------
    # 3. H4 NETRAL dengan diagnostic reason yang spesifik
    # -------------------------------------------------------------------------
    def test_03_h4_netral_with_diagnostic_reason(self):
        """H4 NETRAL menghasilkan diagnostic reason spesifik (bukan generic)."""
        # Buat candle konsolidasi horizontal sempit (lebar < 2.6 ATR)
        rows = []
        for i in range(50):
            c = 2050.0 + (0.5 if i % 2 == 0 else -0.5)
            rows.append((c - 0.2, c + 0.6, c - 0.6, c, 100))
        df_h4_netral = _make_df(rows, freq="4h")

        reg = detect_market_regime(df_h4_netral)
        self.assertEqual(reg["bias"], "NETRAL")
        self.assertIn("diagnostic_reason", reg)
        self.assertTrue(len(reg["diagnostic_reason"]) > 0)
        self.assertIn("Konsolidasi horizontal", reg["diagnostic_reason"])

        ctx = evaluate_htf_context({"D1": df_h4_netral, "H4": df_h4_netral, "H1": df_h4_netral, "M30": df_h4_netral})
        self.assertEqual(ctx["H4"], "NETRAL")
        self.assertIn("h4_diagnostic", ctx)
        self.assertIn("Konsolidasi horizontal", ctx["h4_diagnostic"])

        mock_scan = {"symbol": "XAUUSD", "current_price": 2050.0, "radar_zones": [], "active_zones": [], "htf_context": ctx, "data_ok": True}
        mock_dec = {"status": "WAIT", "stages": {}, "reasons": []}
        pres = format_scan_presentation(scan=mock_scan, dec=mock_dec, detailed=True)
        self.assertIn("- H4 Bias          : NETRAL", pres)
        self.assertIn("Konsolidasi horizontal", pres)
        self.assertIn("5. REASON", pres)
        self.assertIn("Konsolidasi horizontal", pres)

    # -------------------------------------------------------------------------
    # 4. H4 berubah bullish
    # -------------------------------------------------------------------------
    def test_04_h4_becomes_bullish(self):
        """H4 membentuk Higher Highs dan Higher Lows berubah menjadi BULLISH."""
        df_bull = _make_h4_bullish()
        reg = detect_market_regime(df_bull)
        self.assertEqual(reg["bias"], "BULLISH")
        self.assertEqual(reg["regime"], "TRENDING_BULLISH")
        self.assertIn("bullish", reg["detail"].lower())

    # -------------------------------------------------------------------------
    # 5. H4 berubah bearish
    # -------------------------------------------------------------------------
    def test_05_h4_becomes_bearish(self):
        """H4 membentuk Lower Highs dan Lower Lows berubah menjadi BEARISH."""
        df_bear = _make_h4_bearish()
        reg = detect_market_regime(df_bear)
        self.assertEqual(reg["bias"], "BEARISH")
        self.assertEqual(reg["regime"], "TRENDING_BEARISH")
        self.assertIn("bearish", reg["detail"].lower())

    # -------------------------------------------------------------------------
    # 6. Pending order MT5 tidak tersedia
    # -------------------------------------------------------------------------
    def test_06_mt5_pending_orders_unavailable(self):
        """Pending order MT5 tidak tersedia menghasilkan DATA UNAVAILABLE."""
        with patch("modules.mt5_conn.pastikan_mt5_aktif", return_value=False):
            txt = cek_posisi_terbuka_mt5(2000.0)
            self.assertIn("DATA UNAVAILABLE", txt)

        with patch("modules.mt5_conn.pastikan_mt5_aktif", return_value=True), \
             patch("modules.mt5_conn.mt5.account_info", return_value=None):
            txt = cek_posisi_terbuka_mt5(2000.0)
            self.assertIn("DATA UNAVAILABLE", txt)

        # Analisa akun & lot juga wajib mengembalikan DATA UNAVAILABLE jika info None
        txt_lot = analisa_akun_dan_lot(None, None, 2.0, 2000.0)
        self.assertIn("DATA UNAVAILABLE", txt_lot)

    # -------------------------------------------------------------------------
    # 7. Pending order memiliki SL kosong
    # -------------------------------------------------------------------------
    def test_07_mt5_pending_order_empty_sl(self):
        """Pending order yang memiliki SL kosong (0.0) ditandai dengan peringatan belum diset."""
        mock_acc = MagicMock()
        mock_acc.login = 123456
        mock_acc.name = "Trader Test"
        mock_acc.server = "Demo-Server"
        mock_acc.currency = "USD"
        mock_acc.balance = 10000.0
        mock_acc.equity = 10000.0
        mock_acc.profit = 0.0
        mock_acc.margin_free = 10000.0
        mock_acc.trade_mode = 0

        mock_sym = MagicMock()
        mock_sym.digits = 2
        mock_sym.point = 0.01

        mock_tick = MagicMock()
        mock_tick.bid = 2020.0
        mock_tick.ask = 2020.3
        mock_tick.time = 1700000000

        mock_order = MagicMock()
        mock_order.ticket = 998877
        mock_order.symbol = "XAUUSD"
        mock_order.type = 2  # ORDER_TYPE_BUY_LIMIT
        mock_order.price_open = 2005.00
        mock_order.sl = 0.0  # SL KOSONG!
        mock_order.tp = 2030.00
        mock_order.volume_initial = 0.50
        mock_order.volume_current = 0.50

        with patch("modules.mt5_conn.pastikan_mt5_aktif", return_value=True), \
             patch("modules.mt5_conn.mt5.account_info", return_value=mock_acc), \
             patch("modules.mt5_conn.mt5.symbol_info", return_value=mock_sym), \
             patch("modules.mt5_conn.mt5.symbol_info_tick", return_value=mock_tick), \
             patch("modules.mt5_conn.mt5.positions_get", return_value=[]), \
             patch("modules.mt5_conn.mt5.orders_get", return_value=[mock_order]), \
             patch("modules.mt5_conn.mt5.history_deals_get", return_value=[]):
            txt = cek_posisi_terbuka_mt5(2020.0)

            self.assertIn("Ticket #998877", txt)
            self.assertIn("TOTAL 1 ORDER", txt)
            self.assertIn("BELUM DISET (⚠️ WAJIB INGATKAN PASANG SL!)", txt)
            self.assertIn("2005.00", txt)

    # -------------------------------------------------------------------------
    # 8. Jumlah ticket MT5 tidak sesuai dengan jumlah yang dilaporkan dicegah
    # -------------------------------------------------------------------------
    def test_08_mt5_ticket_count_consistency(self):
        """Jumlah ticket MT5 yang dilaporkan wajib 100% konsisten dengan data MT5."""
        mock_acc = MagicMock()
        mock_acc.login = 123456
        mock_acc.name = "Trader Test"
        mock_acc.server = "Demo-Server"
        mock_acc.currency = "USD"
        mock_acc.balance = 5000.0
        mock_acc.equity = 5000.0
        mock_acc.profit = 0.0
        mock_acc.margin_free = 5000.0
        mock_acc.trade_mode = 0

        mock_sym = MagicMock()
        mock_sym.digits = 2

        mock_tick = MagicMock()
        mock_tick.bid = 2020.0
        mock_tick.ask = 2020.3
        mock_tick.time = 1700000000

        # Ada 2 pending order di MT5
        o1 = MagicMock(ticket=101, symbol="XAUUSD", type=2, price_open=2000.0, sl=1995.0, tp=2020.0, volume_initial=0.1, volume_current=0.1)
        o2 = MagicMock(ticket=102, symbol="XAUUSD", type=3, price_open=2040.0, sl=2045.0, tp=2025.0, volume_initial=0.2, volume_current=0.2)

        with patch("modules.mt5_conn.pastikan_mt5_aktif", return_value=True), \
             patch("modules.mt5_conn.mt5.account_info", return_value=mock_acc), \
             patch("modules.mt5_conn.mt5.symbol_info", return_value=mock_sym), \
             patch("modules.mt5_conn.mt5.symbol_info_tick", return_value=mock_tick), \
             patch("modules.mt5_conn.mt5.positions_get", return_value=[]), \
             patch("modules.mt5_conn.mt5.orders_get", return_value=[o1, o2]), \
             patch("modules.mt5_conn.mt5.history_deals_get", return_value=[]):
            txt = cek_posisi_terbuka_mt5(2020.0)

            # Wajib melaporkan TOTAL 2 ORDER dan menampilkan kedua tiket
            self.assertIn("TOTAL 2 ORDER", txt)
            self.assertIn("Ticket #101", txt)
            self.assertIn("Ticket #102", txt)

    # -------------------------------------------------------------------------
    # 9. Zona RADAR tidak otomatis menjadi READY
    # -------------------------------------------------------------------------
    def test_09_radar_zone_not_automatically_ready(self):
        """Zona RADAR tidak otomatis menjadi READY jika syarat HTF/kualitas belum lolos."""
        zone = _make_test_zone("DEMAND", 2000.0, 2004.0, sl=1997.0)
        self.assertEqual(zone.status, "RADAR")

        tracker = ZoneTracker(filepath=None, auto_save=False)
        # H4 NETRAL -> Pending plan tidak boleh valid (Status tetap RADAR, BUKAN READY)
        tracker.update([zone], current_price=2015.0, htf_bias="NETRAL", m30_status="CONFIRMED")
        rec = tracker.tracked_zones[zone.zone_id]
        self.assertEqual(rec["status"], "RADAR", "Zona pada HTF Netral harus tetap status RADAR, tidak boleh READY")
        self.assertIsNone(rec["pending_plan"])

    # -------------------------------------------------------------------------
    # 10. Zona READY belum otomatis menjadi CONFIRMED
    # -------------------------------------------------------------------------
    def test_10_ready_zone_not_automatically_confirmed(self):
        """Zona READY belum otomatis menjadi CONFIRMED sebelum lolos validasi M5."""
        zone = _make_test_zone("DEMAND", 2000.0, 2004.0, sl=1997.0)
        h1 = _make_h1_with_high_target()
        frames = {"H1": h1, "H4": _make_h4_bullish()}

        tracker = ZoneTracker(filepath=None, auto_save=False)
        # H4 Bullish & M30 Confirmed -> lolos pending plan -> status READY
        tracker.update([zone], current_price=2015.0, frames=frames, htf_bias="BULLISH", m30_status="CONFIRMED")
        rec = tracker.tracked_zones[zone.zone_id]
        self.assertEqual(rec["status"], "READY")
        self.assertIsNotNone(rec["pending_plan"])

        # Evaluasi keputusan konservatif M5: harga di 2015 belum sentuh zona dan belum ada sweep/trigger M5 -> WAIT
        dec_conservative = {"status": "WAIT", "zone": zone.to_dict(), "direction": None}
        tracker.update([zone], current_price=2015.0, frames=frames, htf_bias="BULLISH", m30_status="CONFIRMED", conservative_dec=dec_conservative)
        rec = tracker.tracked_zones[zone.zone_id]
        self.assertNotEqual(rec["status"], "CONFIRMED", "Zona READY tidak boleh menjadi CONFIRMED jika M5 masih WAIT")
        self.assertEqual(rec["status"], "READY")


if __name__ == "__main__":
    unittest.main()
