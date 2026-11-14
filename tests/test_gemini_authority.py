# -*- coding: utf-8 -*-
"""
tests/test_gemini_authority.py
Unit test suite untuk memastikan Otoritas Keputusan Python Scanner dan Sanitasi Output Gemini:
1. Gemini SELL, Python WAIT -> dinetralisir ke WAIT, tidak ada angka entry aktif.
2. Gemini BUY, Python INVALID -> dinetralisir ke INVALID, tidak ada rekomendasi entry aktif.
3. Gemini BUY, Python VALID -> sinyal dikunci ke rencana BUY resmi Python.
4. Gemini SELL, Python VALID -> sinyal dikunci ke rencana SELL resmi Python.
5. H4 NETRAL -> status WAIT, alasan menyebut H4 NETRAL, pending plan tidak aktif, radar zone tampil.
6. H4 BULLISH & H1 BEARISH (Konflik HTF) -> status WAIT, alasan menyebut konflik, pending plan tidak aktif.
7. Radar zone tampil sebagai kandidat pemantauan tanpa sinyal resmi.
8. Pending plan tidak muncul jika HTF belum memenuhi syarat (H4 NETRAL / M30 CONFLICT).
9. Gemini memberikan Entry, SL, TP saat Python WAIT -> seluruh angka entry/SL/TP dinetralisir.
"""
import unittest
import pandas as pd
import numpy as np

import modules.ai_gemini as ag
import modules.zone_scanner as zs
from modules.zone_scanner import (
    evaluate_trade_decision, scan_xauusd_zones, format_scan_presentation,
    detect_early_setups, Zone
)
from modules.ai_gemini import sanitasi_output_gemini, ekstrak_target_ai


def _buat_df_dummy(n: int, arah: str = "BULLISH", close_start: float = 2000.0) -> pd.DataFrame:
    records = []
    c = close_start
    step = 0.5 if arah == "BULLISH" else (-0.5 if arah == "BEARISH" else 0.0)
    for i in range(n):
        o = c
        if arah == "BULLISH":
            c = o + step + (0.1 if i % 2 == 0 else -0.05)
            h = max(o, c) + 0.4
            l = min(o, c) - 0.2
        elif arah == "BEARISH":
            c = o + step - (0.1 if i % 2 == 0 else -0.05)
            h = max(o, c) + 0.2
            l = min(o, c) - 0.4
        else:
            c = o + (0.3 if i % 2 == 0 else -0.3)
            h = max(o, c) + 0.2
            l = min(o, c) - 0.2
        records.append({
            "Open": o, "High": h, "Low": l, "Close": c, "Volume": 1000 + i * 10
        })
    df = pd.DataFrame(records)
    df.index = pd.date_range("2026-01-01", periods=n, freq="5min")
    return df


class TestGeminiAuthority(unittest.TestCase):

    def setUp(self):
        self._orig_ag_dec = ag.evaluate_trade_decision
        self._orig_ag_scan = ag.scan_xauusd_zones
        self._orig_load = zs._load_frames
        zs._load_frames = lambda ttl=10.0: None

    def tearDown(self):
        ag.evaluate_trade_decision = self._orig_ag_dec
        ag.scan_xauusd_zones = self._orig_ag_scan
        zs._load_frames = self._orig_load

    def test_01_gemini_sell_python_wait(self):
        """1. Gemini SELL, Python WAIT -> dinetralisir ke WAIT, tidak ada rekomendasi entry aktif."""
        gemini_raw = """
[SINYAL MEDIUM RISK] SELL LIMIT 3986.15
• Rekomendasi : SELL LIMIT 3986.15
• Area Entry   : 3986.15 - 3988.00 (Jarak dari harga sekarang: 1.5 pt)
• Stop Loss    : 3991.50 (Jarak risiko: 5.35 pt)
• Target TP    :
  - TP1        : 3978.00 (R:R 1:1.5)
  - TP2        : 3968.00 (R:R 1:3.0)
• Manajemen Lot: 0.10 lot
• Note Limit (Anti Kena Kentut) : Pasang limit di 3987.00
[RADAR_TARGET]
ZONA_ATAS: 3986.15-3988.00 (batas_jebol: 3991.50)
ZONA_BAWAH: 3940.00-3942.00 (batas_jebol: 3935.00)
SINYAL_AKTIF: SELL | ENTRY: 3986.15-3988.00 | SL: 3991.50 | TP1: 3978.00 | TP2: 3968.00
[/RADAR_TARGET]
"""
        dec_wait = {
            "status": "WAIT",
            "direction": None,
            "reasons": ["H4 NETRAL, tidak ada directional bias yang jelas"]
        }
        sanitized = sanitasi_output_gemini(gemini_raw, dec=dec_wait)

        # Verifikasi larangan terpenuhi
        self.assertIn("[STATUS TRADING: WAIT — TIDAK ADA REKOMENDASI ENTRY AKTIF]", sanitized)
        self.assertNotIn("[SINYAL MEDIUM RISK]", sanitized)
        self.assertIn("• Area Entry   : - (WAIT: Tidak ada rekomendasi entry aktif dari Python Scanner)", sanitized)
        self.assertIn("• Stop Loss    : - (Tidak ada posisi aktif)", sanitized)
        self.assertIn("SINYAL_AKTIF: WAIT | ENTRY: 0.00-0.00 | SL: 0.00 | TP1: 0.00 | TP2: 0.00", sanitized)
        self.assertNotIn("SINYAL_AKTIF: SELL", sanitized)

        # Verifikasi parser target tidak mengambil sinyal
        tgt = ekstrak_target_ai(sanitized)
        self.assertIsNone(tgt.get("sinyal"))

    def test_02_gemini_buy_python_invalid(self):
        """2. Gemini BUY, Python INVALID -> dinetralisir ke INVALID, setup ditolak."""
        gemini_raw = """
[SINYAL LOW RISK] BUY XAUUSD
• Rekomendasi : BUY LIMIT 2000.00
• Area Entry   : 2000.00 - 2004.00
• Stop Loss    : 1995.00
• Target TP    :
  - TP1        : 2005.00
  - TP2        : 2010.00
• Manajemen Lot: 0.20 lot
[RADAR_TARGET]
ZONA_BAWAH: 2000.00-2004.00 (batas_jebol: 1995.00)
SINYAL_AKTIF: BUY | ENTRY: 2000.00-2004.00 | SL: 1995.00 | TP1: 2005.00 | TP2: 2010.00
[/RADAR_TARGET]
"""
        dec_invalid = {
            "status": "INVALID",
            "direction": None,
            "reasons": ["Risk/Reward tidak memenuhi batas minimum 1:1.5"]
        }
        sanitized = sanitasi_output_gemini(gemini_raw, dec=dec_invalid)

        self.assertIn("[STATUS TRADING: INVALID — TIDAK ADA REKOMENDASI ENTRY AKTIF]", sanitized)
        self.assertNotIn("[SINYAL LOW RISK]", sanitized)
        self.assertIn("• Area Entry   : - (WAIT: Tidak ada rekomendasi entry aktif dari Python Scanner)", sanitized)
        self.assertIn("SINYAL_AKTIF: WAIT", sanitized)
        self.assertNotIn("SINYAL_AKTIF: BUY", sanitized)

        tgt = ekstrak_target_ai(sanitized)
        self.assertIsNone(tgt.get("sinyal"))

    def test_03_gemini_buy_python_valid(self):
        """3. Gemini BUY, Python VALID -> sinyal dikunci ke rencana BUY resmi Python."""
        gemini_raw = """
Analisis pasar menunjukkan tren Bullish yang kuat.
[SINYAL LOW RISK] BUY XAUUSD
• Area Entry   : 2000.00 - 2004.00
• Stop Loss    : 1995.00
• Target TP    :
  - TP1        : 2012.00
  - TP2        : 2025.00
[RADAR_TARGET]
ZONA_BAWAH: 2000.00-2004.00 (batas_jebol: 1995.00)
SINYAL_AKTIF: BUY | ENTRY: 2000.00-2004.00 | SL: 1995.00 | TP1: 2012.00 | TP2: 2025.00
[/RADAR_TARGET]
"""
        dec_valid_buy = {
            "status": "VALID",
            "direction": "BUY",
            "plan": {
                "arah": "BUY", "entry_min": 2000.0, "entry_max": 2004.0, "entry_acuan": 2002.0,
                "sl": 1995.0, "tp1": 2012.0, "tp2": 2025.0, "rr": 2.0, "keterangan": "Valid Demand H1"
            },
            "reasons": ["Setup BUY lolos validasi multi-timeframe"]
        }
        sanitized = sanitasi_output_gemini(gemini_raw, dec=dec_valid_buy)

        # Sinyal resmi dipertahankan
        self.assertIn("SINYAL_AKTIF: BUY | ENTRY: 2000.00-2004.00 | SL: 1995.00 | TP1: 2012.00 | TP2: 2025.00", sanitized)

    def test_04_gemini_sell_python_valid(self):
        """4. Gemini SELL, Python VALID -> sinyal dikunci ke rencana SELL resmi Python."""
        gemini_raw = """
Analisis pasar menunjukkan tekanan Bearish institusi.
[SINYAL LOW RISK] SELL XAUUSD
• Area Entry   : 2050.00 - 2052.00
• Stop Loss    : 2058.00
• Target TP    :
  - TP1        : 2040.00
  - TP2        : 2030.00
[RADAR_TARGET]
ZONA_ATAS: 2050.00-2052.00 (batas_jebol: 2058.00)
SINYAL_AKTIF: SELL | ENTRY: 2050.00-2052.00 | SL: 2058.00 | TP1: 2040.00 | TP2: 2030.00
[/RADAR_TARGET]
"""
        dec_valid_sell = {
            "status": "VALID",
            "direction": "SELL",
            "plan": {
                "arah": "SELL", "entry_min": 2050.0, "entry_max": 2052.0, "entry_acuan": 2051.0,
                "sl": 2058.0, "tp1": 2040.0, "tp2": 2030.0, "rr": 2.0, "keterangan": "Valid Supply H1"
            },
            "reasons": ["Setup SELL lolos validasi multi-timeframe"]
        }
        sanitized = sanitasi_output_gemini(gemini_raw, dec=dec_valid_sell)

        self.assertIn("SINYAL_AKTIF: SELL | ENTRY: 2050.00-2052.00 | SL: 2058.00 | TP1: 2040.00 | TP2: 2030.00", sanitized)

    def test_05_h4_netral_decision_wait(self):
        """5. H4 NETRAL -> status WAIT, alasan menyebut H4 NETRAL, pending plan tidak aktif, radar zone tampil."""
        # Simulasi keputusan scanner dengan H4 netral
        mock_scan = {
            "symbol": "XAUUSD", "current_price": 2020.0,
            "active_zones": [
                {"zone_id": "H1_DEMAND_1", "zone_type": "DEMAND", "price_low": 2000.0, "price_high": 2004.0, "timeframe": "H1", "grade": "A", "invalidation_price": 1997.0},
                {"zone_id": "H1_SUPPLY_1", "zone_type": "SUPPLY", "price_low": 2030.0, "price_high": 2034.0, "timeframe": "H1", "grade": "A", "invalidation_price": 2037.0},
            ],
            "best_demand": None, "best_supply": None,
            "htf_context": {"H4": "NETRAL", "H1": "BULLISH", "M30": "NETRAL"},
            "empty_reason": "Bias H4 netral / tidak ada directional bias (NETRAL)", "data_ok": True,
        }
        mock_dec = {
            "status": "WAIT", "direction": None, "plan": None,
            "stages": {"H4": {"ok": False, "msg": "Bias H4 NETRAL"}},
            "reasons": ["Bias H4 tidak jelas (NETRAL)"]
        }

        output_ui = format_scan_presentation(scan=mock_scan, dec=mock_dec, detailed=True)
        output_concise = format_scan_presentation(scan=mock_scan, dec=mock_dec, detailed=False)
        self.assertNotIn("Kandidat Demand  : H1 DEMAND 2000.00 - 2004.00", output_concise)
        self.assertIn("1. MARKET BIAS", output_concise)
        self.assertIn("2. ZONA UTAMA", output_concise)
        self.assertIn("3. ENTRY PLAN", output_concise)
        self.assertIn("4. ENTRY CONFIRMATION", output_concise)
        self.assertIn("5. FINAL DECISION", output_concise)

        # 1. FINAL DECISION: WAIT
        self.assertIn("1. FINAL DECISION", output_ui)
        self.assertIn("- Status           : WAIT (NO VALID SETUP)", output_ui)
        self.assertIn("- Recommendation   : NO ENTRY / WAIT", output_ui)

        # 2. MARKET BIAS: H4 Bias NETRAL
        self.assertIn("2. MARKET BIAS (MARKET OVERVIEW)", output_ui)
        self.assertIn("- H4 Bias          : NETRAL", output_ui)

        # 3. RADAR ZONE: Tampil sebagai kandidat pemantauan
        self.assertIn("3. RADAR ZONE (ZONA PEMANTAUAN & ZONE ANALYSIS)", output_ui)
        self.assertIn("Kandidat Demand  : H1 DEMAND 2000.00 - 2004.00", output_ui)
        self.assertIn("Kandidat Supply  : H1 SUPPLY 2030.00 - 2034.00", output_ui)

        # 4. ENTRY PLAN: Early Pending Plan TIDAK AKTIF
        self.assertIn("4. ENTRY PLAN (ENTRY VALIDATION & OTORITAS SCANNER)", output_ui)
        self.assertIn("Status         : TIDAK AKTIF (H4 belum memberikan directional bias yang jelas (NETRAL))", output_ui)
        self.assertIn("Recommendation : NO PENDING SETUP (WAIT)", output_ui)

        # 5. REASON: Menjelaskan H4 NETRAL
        self.assertIn("5. REASON", output_ui)
        self.assertIn("H4 belum memberikan directional bias yang jelas (NETRAL)", output_ui)

    def test_06_h4_bullish_h1_bearish_conflict(self):
        """6. H4 BULLISH & H1 BEARISH (Konflik HTF) -> status WAIT, alasan menyebut konflik, pending plan tidak aktif."""
        mock_scan = {
            "symbol": "XAUUSD", "current_price": 2010.0,
            "active_zones": [
                {"zone_id": "H1_DEMAND_1", "zone_type": "DEMAND", "price_low": 2000.0, "price_high": 2004.0, "timeframe": "H1", "grade": "A", "invalidation_price": 1997.0}
            ],
            "best_demand": None, "best_supply": None,
            "htf_context": {"H4": "BULLISH", "H1": "BEARISH", "M30": "BEARISH", "has_conflict": True},
            "empty_reason": "Konflik struktur H4 (BULLISH) vs H1 (BEARISH)", "data_ok": True,
        }
        mock_dec = {
            "status": "WAIT", "direction": None, "plan": None,
            "stages": {
                "H4": {"ok": True, "msg": "Bias H4 BULLISH"},
                "H1": {"ok": False, "msg": "H1 BEARISH bertentangan dengan H4 BULLISH"}
            },
            "reasons": ["Konflik struktur H4 BULLISH vs H1 BEARISH"]
        }

        # Early setup detection wajib menolak karena konflik
        early = detect_early_setups(current_price=2010.0, scan=mock_scan)
        self.assertIsNone(early["best_setup"])
        self.assertIn("Konflik struktur H4 (BULLISH) vs H1 (BEARISH)", early["empty_reason"])

        output_ui = format_scan_presentation(scan=mock_scan, dec=mock_dec, early_setups=early, detailed=True)
        self.assertIn("1. FINAL DECISION", output_ui)
        self.assertIn("WAIT (NO VALID SETUP)", output_ui)
        self.assertIn("Konflik struktur H4 BULLISH vs H1 BEARISH", output_ui)

    def test_07_radar_zone_available_without_signal(self):
        """7. Radar zone tampil sebagai kandidat pemantauan tanpa sinyal resmi terkonfirmasi."""
        mock_scan = {
            "symbol": "XAUUSD", "current_price": 2015.0,
            "active_zones": [
                {"zone_id": "H1_DEMAND_1", "zone_type": "DEMAND", "price_low": 2000.0, "price_high": 2004.0, "timeframe": "H1", "grade": "A+", "invalidation_price": 1997.0},
                {"zone_id": "H1_SUPPLY_1", "zone_type": "SUPPLY", "price_low": 2025.0, "price_high": 2028.0, "timeframe": "H1", "grade": "A", "invalidation_price": 2031.0},
            ],
            "best_demand": None, "best_supply": None,
            "htf_context": {"H4": "BULLISH", "H1": "BULLISH", "M30": "BULLISH"},
            "empty_reason": "Menunggu konfirmasi trigger M5", "data_ok": True,
        }
        mock_dec = {
            "status": "WAIT", "direction": "BUY", "plan": None,
            "stages": {
                "zone_touch": {"status": "WAIT", "msg": "Harga belum menyentuh zona"},
                "liquidity_sweep": {"status": "WAIT", "msg": "Belum ada sweep M5"},
                "structure_trigger": {"status": "WAIT", "msg": "Belum ada CHoCH"},
            },
            "reasons": ["Menunggu harga menyentuh zona Demand dan konfirmasi Sweep M5"]
        }

        output_ui = format_scan_presentation(scan=mock_scan, dec=mock_dec, detailed=True)
        output_concise = format_scan_presentation(scan=mock_scan, dec=mock_dec, detailed=False)
        self.assertNotIn("Kandidat Demand  : H1 DEMAND 2000.00 - 2004.00", output_concise)
        self.assertIn("1. MARKET BIAS", output_concise)
        self.assertIn("2. ZONA UTAMA", output_concise)
        self.assertIn("3. ENTRY PLAN", output_concise)
        self.assertIn("4. ENTRY CONFIRMATION", output_concise)
        self.assertIn("5. FINAL DECISION", output_concise)

        # Radar zone ada
        self.assertIn("3. RADAR ZONE (ZONA PEMANTAUAN & ZONE ANALYSIS)", output_ui)
        self.assertIn("[Kandidat Zona Pemantauan — BUKAN Sinyal Entry / Pending Order Aktif]", output_ui)
        self.assertIn("Kandidat Demand  : H1 DEMAND 2000.00 - 2004.00 [Grade: A+]", output_ui)

        # Sinyal resmi konservatif belum ada (WAIT)
        self.assertIn("B. CONFIRMED ENTRY (Strategi Konservatif - Sinyal Resmi Terkonfirmasi):", output_ui)
        self.assertIn("Sinyal Resmi   : TIDAK ADA SINYAL AKTIF (WAIT)", output_ui)

    def test_08_pending_plan_inactive_when_htf_unqualified(self):
        """8. Pending plan tidak muncul jika HTF belum memenuhi syarat (H4 NETRAL atau M30 CONFLICT)."""
        # A. Kasus H4 NETRAL
        scan_neutral = {
            "current_price": 2010.0, "active_zones": [
                {"zone_id": "Z1", "zone_type": "DEMAND", "price_low": 2000.0, "price_high": 2004.0, "grade": "A", "invalidation_price": 1997.0}
            ],
            "htf_context": {"H4": "NETRAL", "H1": "NETRAL", "M30": "NETRAL"}
        }
        early_neutral = detect_early_setups(current_price=2010.0, scan=scan_neutral)
        self.assertIsNone(early_neutral["best_setup"])
        self.assertIn("Bias H4 netral", early_neutral["empty_reason"])

        # B. Kasus M30 CONFLICT
        # DataFrame M30 dengan close < EMA20 < EMA50 dan RSI rendah saat bias BULLISH
        df_m30_conflict = pd.DataFrame([{
            "Open": 2005.0, "High": 2006.0, "Low": 1990.0, "Close": 1991.0,
            "EMA20": 2000.0, "EMA50": 2010.0, "RSI14": 30.0, "ATR14": 3.0
        } for _ in range(25)])
        df_m30_conflict.index = pd.date_range("2026-01-01", periods=25, freq="30min")

        frames_conflict = {"M30": df_m30_conflict}
        scan_conflict = {
            "current_price": 2010.0, "active_zones": [
                {"zone_id": "Z1", "zone_type": "DEMAND", "price_low": 2000.0, "price_high": 2004.0, "grade": "A", "invalidation_price": 1997.0}
            ],
            "htf_context": {"H4": "BULLISH", "H1": "BULLISH", "M30": "BEARISH"}
        }
        early_m30 = detect_early_setups(frames=frames_conflict, current_price=2010.0, scan=scan_conflict)
        self.assertIsNone(early_m30["best_setup"])
        self.assertIn("Momentum M30", early_m30["empty_reason"])

    def test_09_gemini_entry_sl_tp_neutralized_during_python_wait(self):
        """9. Gemini memberikan Entry, SL, TP saat Python WAIT -> seluruh angka entry dinetralisir."""
        gemini_hallucination = """
Pasar emas menunjukkan potensi penurunan dari supply atas terdekat.
[SINYAL MEDIUM RISK] SELL LIMIT 3986.15
• Rekomendasi : SELL LIMIT 3986.15
• Area Entry   : 3986.15 - 3988.00 (Jarak dari harga sekarang: 1.5 pt)
• Stop Loss    : 3991.50 (Jarak risiko: 5.35 pt | Alasan: di balik wick)
• Target TP    :
  - TP1        : 3978.00 (R:R 1:1.5 | Amankan 50% lot)
  - TP2        : 3968.00 (R:R 1:3.0 | Runner)
• Manajemen Lot: 0.15 lot (Maksimum risiko 0.5% modal)
• Pemicu (Trigger) : Sweep M5 di atas 3988.00
• Note Limit (Anti Kena Kentut) : Pasang limit di 3987.00
• Invalidasi   : Setup BATAL jika candle M5 close di atas 3991.50
[RADAR_TARGET]
ZONA_ATAS: 3986.15-3988.00 (batas_jebol: 3991.50)
ZONA_BAWAH: 3940.00-3942.00 (batas_jebol: 3935.00)
SINYAL_AKTIF: SELL | ENTRY: 3986.15-3988.00 | SL: 3991.50 | TP1: 3978.00 | TP2: 3968.00
[/RADAR_TARGET]
"""
        dec_wait = {
            "status": "WAIT",
            "direction": None,
            "reasons": ["H4 NETRAL. Menunggu konfirmasi breakout struktur."]
        }
        sanitized = sanitasi_output_gemini(gemini_hallucination, dec=dec_wait)

        # Pastikan tidak ada angka entry atau SL aktif yang bocor
        self.assertNotIn("• Area Entry   : 3986.15 - 3988.00", sanitized)
        self.assertNotIn("• Stop Loss    : 3991.50", sanitized)
        self.assertNotIn("• Rekomendasi : SELL LIMIT 3986.15", sanitized)
        self.assertNotIn("• Manajemen Lot: 0.15 lot", sanitized)
        self.assertIn("• Area Entry   : - (WAIT: Tidak ada rekomendasi entry aktif dari Python Scanner)", sanitized)
        self.assertIn("• Stop Loss    : - (Tidak ada posisi aktif)", sanitized)
        self.assertIn("• Target TP    : -", sanitized)
        self.assertIn("• Manajemen Lot: 0.00 lot (Menunggu sinyal valid)", sanitized)
        self.assertIn("Jangan memasang pending limit", sanitized)
        self.assertIn("SINYAL_AKTIF: WAIT | ENTRY: 0.00-0.00 | SL: 0.00 | TP1: 0.00 | TP2: 0.00", sanitized)
        self.assertIn("[OTORITAS PYTHON SCANNER — KEPUTUSAN FINAL: WAIT]", sanitized)


if __name__ == "__main__":
    unittest.main()
