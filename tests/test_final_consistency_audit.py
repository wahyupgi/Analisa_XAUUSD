"""
Unit Test: Final Consistency Audit — XAUUSD SMC Scanner
Memvalidasi 14 Skenario Audit:
1. Distance zona di atas harga (zone_low - current_price)
2. Distance zona di bawah harga (current_price - zone_high)
3. Harga berada di dalam zona (distance = 0.0, status = INSIDE_ZONE)
4. Konsistensi snapshot tunggal harga di seluruh bagian output
5. Early Pending RR (RR 1:4.1 menghasilkan EARLY_PENDING_RR = PASSED)
6. Confirmed Entry RR sebelum M5 (CONFIRMED_ENTRY_RR = NOT_CHECKED)
7. Koeksistensi harmonis Early Pending READY dan Confirmed Entry WAIT tanpa kontradiksi
8. Pending Plan READY berlabel jelas BELUM DIPASANG DI MT5
9. Suggested Expiration 1 jam dinamis ("1 jam")
10. Suggested Expiration 8 jam dinamis ("8 jam")
11. Stop Loss berada di luar invalidation level dengan buffer valid (Demand SL < Inval, Supply SL > Inval)
12. Posisi/Order MT5 tanpa SL memunculkan CRITICAL WARNING
13. Gemini Output Integrity (Gemini tidak dapat mengubah keputusan/harga/order)
14. Pengujian berjalan 100% offline tanpa live MT5 / network / API key
"""

import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
import pandas as pd

from modules.zone_scanner import (
    calculate_zone_distance,
    calculate_zone_approach,
    calculate_suggested_expiration,
    plan_pending_order,
    prioritize_radar_zones,
    format_scan_presentation,
    Zone,
)
from modules.ai_gemini import sanitasi_output_gemini
from modules.mt5_conn import cek_posisi_terbuka_mt5
from modules.utils import format_startup_presentation
from modules.config import APP_VERSION
from tests.test_zone_scanner_validation import (
    _make_df,
    _make_h4_bullish,
    _make_h4_bearish,
    _make_h1_with_high_target,
    _make_h1_with_low_target,
    _make_test_zone,
)


class TestFinalConsistencyAudit(unittest.TestCase):
    """Pengujian komprehensif 14 skenario audit konsistensi scanner."""

    # -------------------------------------------------------------------------
    # 1. Distance Zona di Atas Harga
    # -------------------------------------------------------------------------
    def test_01_distance_zone_above_price(self):
        """1. Distance zona di atas harga: distance = zone_low - current_price."""
        z_low = 2030.00
        z_high = 2035.00
        price = 2020.00
        expected_dist = round(z_low - price, 2)  # 10.00

        res = calculate_zone_distance(z_low, z_high, price, digits=2)
        self.assertEqual(res.status, "ABOVE_PRICE")
        self.assertEqual(res.distance, expected_dist)
        self.assertEqual(res.distance, 10.00)

    # -------------------------------------------------------------------------
    # 2. Distance Zona di Bawah Harga
    # -------------------------------------------------------------------------
    def test_02_distance_zone_below_price(self):
        """2. Distance zona di bawah harga: distance = current_price - zone_high."""
        z_low = 2000.00
        z_high = 2005.00
        price = 2020.00
        expected_dist = round(price - z_high, 2)  # 15.00

        res = calculate_zone_distance(z_low, z_high, price, digits=2)
        self.assertEqual(res.status, "BELOW_PRICE")
        self.assertEqual(res.distance, expected_dist)
        self.assertEqual(res.distance, 15.00)

    # -------------------------------------------------------------------------
    # 3. Harga Berada di Dalam Zona
    # -------------------------------------------------------------------------
    def test_03_price_inside_zone_distance_zero(self):
        """3. Harga berada di dalam zona: distance = 0.0 dan status = INSIDE_ZONE."""
        z_low = 2015.00
        z_high = 2025.00
        price = 2020.00

        res = calculate_zone_distance(z_low, z_high, price, digits=2)
        self.assertEqual(res.status, "INSIDE_ZONE")
        self.assertEqual(res.distance, 0.0)

        app = calculate_zone_approach(res.distance, res.status, atr=2.0)
        self.assertEqual(app, "TOUCHED")

    # -------------------------------------------------------------------------
    # 4. Seluruh Output Menggunakan Snapshot Harga yang Sama
    # -------------------------------------------------------------------------
    def test_04_single_price_snapshot_across_all_sections(self):
        """4. Seluruh bagian output menggunakan satu snapshot harga dan timestamp yang sama."""
        fixed_price = 2020.00
        fixed_time = "2026-11-12 14:46:00"
        z_supply = _make_test_zone("SUPPLY", 2030.0, 2035.0, sl=2037.0)
        z_demand = _make_test_zone("DEMAND", 2000.0, 2005.0, sl=1997.0)

        scan_mock = {
            "symbol": "XAUUSD",
            "current_price": fixed_price,
            "snapshot_price": fixed_price,
            "snapshot_time": fixed_time,
            "bid": 2019.80,
            "ask": 2020.20,
            "radar_zones": [z_supply.to_dict(), z_demand.to_dict()],
            "active_zones": [z_supply.to_dict(), z_demand.to_dict()],
            "rejected_zones": [],
            "best_demand": z_demand.to_dict(),
            "best_supply": z_supply.to_dict(),
            "htf_context": {"H4": "BEARISH", "H1": "BEARISH", "M30": "BEARISH"},
            "empty_reason": "",
            "summary_text": "",
            "data_ok": True,
        }
        dec_mock = {
            "status": "WAIT",
            "direction": None,
            "reasons": ["Menunggu mitigasi zona Supply"],
            "stages": {
                "zone_touch": {"status": "WAIT"},
                "liquidity_sweep": {"status": "WAIT"},
                "structure_trigger": {"status": "WAIT"},
                "displacement": {"status": "WAIT"},
                "risk_reward": {"status": "NOT_CHECKED"},
            },
            "current_price": fixed_price,
        }

        output_text = format_scan_presentation(scan=scan_mock, dec=dec_mock)

        # Periksa Section 2 memuat snapshot waktu & bid/ask konsisten
        self.assertIn("Current Price    : 2020.00 (Snapshot: 2026-11-12 14:46:00 | Bid: 2019.80 | Ask: 2020.20)", output_text)

        # Periksa konsistensi distance pada Primary Zone (Supply @ 2030: 2030 - 2020 = 10.00 pt)
        self.assertIn("Jarak: 10.00 pt (ABOVE_PRICE)", output_text)
        # Periksa konsistensi distance pada Kandidat Demand (Demand @ 2005: 2020 - 2005 = 15.00 pt)
        self.assertIn("Jarak: 15.00 pt (BELOW_PRICE)", output_text)
        # Periksa konsistensi distance pada Distance to Zone
        self.assertIn("10.00 pt", output_text)

    # -------------------------------------------------------------------------
    # 5. Early Pending RR: 1:4.1 Menghasilkan Status PASSED
    # -------------------------------------------------------------------------
    def test_05_early_pending_rr_passed(self):
        """5. Early Pending RR >= 1.5 menghasilkan EARLY_PENDING_RR = PASSED."""
        h1 = _make_h1_with_high_target()
        frames = {"H1": h1, "H4": _make_h4_bullish()}
        # Zone Demand: 2000 - 2004, SL: 1997 (risk = 7.0 pt), target di 2030.5 -> reward ~26.2 pt (RR ~3.74)
        zone = _make_test_zone("DEMAND", 2000.0, 2004.0, sl=1997.0, bos_candle_time=h1.index[25])

        plan = plan_pending_order(zone, current_price=2015.0, frames=frames, htf_bias="BULLISH", m30_status="CONFIRMED")
        self.assertTrue(plan["valid"])
        self.assertEqual(plan["status"], "READY")
        self.assertEqual(plan["early_pending_status"], "READY")
        self.assertGreaterEqual(plan["rr_tp1"], 1.5)
        self.assertTrue(plan["early_pending_rr"].startswith("PASSED"))
        self.assertIn("PASSED", plan["early_pending_rr"])

    # -------------------------------------------------------------------------
    # 6. Confirmed Entry RR Sebelum M5 Trigger Lengkap = NOT_CHECKED
    # -------------------------------------------------------------------------
    def test_06_confirmed_entry_rr_not_checked_before_m5(self):
        """6. Confirmed Entry RR berstatus NOT_CHECKED sebelum pemicu M5 lengkap, tanpa kontradiksi."""
        zone = _make_test_zone("SUPPLY", 2030.0, 2035.0, sl=2037.0)
        stages_m5_untriggered = {
            "zone_touch": {"status": "WAIT"},
            "liquidity_sweep": {"status": "WAIT"},
            "structure_trigger": {"status": "WAIT"},
            "displacement": {"status": "WAIT"},
            "risk_reward": {"status": "NOT_CHECKED"},
        }

        prio = prioritize_radar_zones(
            [zone.to_dict()],
            current_price=2020.0,
            htf_bias="BEARISH",
            h1_structure="BEARISH",
            m5_stages=stages_m5_untriggered,
        )

        self.assertEqual(prio.get("confirmed_entry_rr"), "NOT_CHECKED")
        # Pastikan tidak ada pesan palsu [RISK_REWARD]: Belum memenuhi Risk/Reward...
        missing_str = "; ".join(prio.get("missing_conditions", []))
        self.assertNotIn("[RISK_REWARD]: Belum memenuhi Risk/Reward", missing_str)
        self.assertNotIn("[CONFIRMED_ENTRY_RISK_REWARD]", missing_str)

    # -------------------------------------------------------------------------
    # 7. Early Pending READY dan Confirmed Entry WAIT Koeksistensi Harmonis
    # -------------------------------------------------------------------------
    def test_07_early_pending_ready_and_confirmed_entry_wait_coexist(self):
        """7. Early Pending READY dan Confirmed Entry WAIT dapat berdampingan tanpa kontradiksi status."""
        h1 = _make_h1_with_high_target()
        frames = {"H1": h1, "H4": _make_h4_bullish()}
        zone = _make_test_zone("DEMAND", 2000.0, 2004.0, sl=1997.0, bos_candle_time=h1.index[25])

        plan = plan_pending_order(zone, current_price=2012.0, frames=frames, htf_bias="BULLISH", m30_status="CONFIRMED")
        self.assertTrue(plan["valid"])

        scan_mock = {
            "symbol": "XAUUSD", "current_price": 2012.0, "radar_zones": [zone.to_dict()],
            "active_zones": [zone.to_dict()], "htf_context": {"H4": "BULLISH", "H1": "BULLISH"},
        }
        dec_wait = {
            "status": "WAIT", "direction": None, "plan": None,
            "stages": {
                "zone_touch": {"status": "WAIT"},
                "liquidity_sweep": {"status": "WAIT"},
                "structure_trigger": {"status": "WAIT"},
                "displacement": {"status": "WAIT"},
                "risk_reward": {"status": "NOT_CHECKED"},
            },
        }

        rendered = format_scan_presentation(scan=scan_mock, dec=dec_wait, early_setups={"best_setup": plan})

        # 1. Final Decision adalah WAIT
        self.assertIn("1. FINAL DECISION", rendered)
        self.assertIn("- Status           : WAIT (NO VALID SETUP)", rendered)

        # 2. Early Pending Plan adalah READY
        self.assertIn("READY — PENDING PLAN (BELUM DIPASANG DI MT5)", rendered)
        self.assertIn("EARLY_PENDING_STATUS : READY", rendered)
        self.assertIn("EARLY_PENDING_RR (TP1) : PASSED", rendered)

        # 3. Confirmed Entry adalah WAIT & NOT_CHECKED
        self.assertIn("CONFIRMED_ENTRY_RR : NOT_CHECKED", rendered)
        self.assertIn("TIDAK ADA SINYAL AKTIF (WAIT)", rendered)

        # 4. Tidak ada kontradiksi RR
        self.assertNotIn("[RISK_REWARD]: Belum memenuhi Risk/Reward", rendered)

    # -------------------------------------------------------------------------
    # 8. Pending Plan READY Tidak Berarti Order Sudah Terpasang di MT5
    # -------------------------------------------------------------------------
    def test_08_pending_plan_ready_not_interpreted_as_placed_in_mt5(self):
        """8. Pending Plan READY berlabel jelas belum dipasang di MT5 dan order MT5 berstatus terpisah."""
        h1 = _make_h1_with_high_target()
        frames = {"H1": h1, "H4": _make_h4_bullish()}
        zone = _make_test_zone("DEMAND", 2000.0, 2004.0, sl=1997.0, bos_candle_time=h1.index[25])
        plan = plan_pending_order(zone, current_price=2012.0, frames=frames, htf_bias="BULLISH", m30_status="CONFIRMED")

        scan_mock = {"symbol": "XAUUSD", "current_price": 2012.0, "radar_zones": [zone.to_dict()], "htf_context": {"H4": "BULLISH"}}
        dec_mock = {"status": "WAIT", "stages": {}}
        rendered = format_scan_presentation(scan=scan_mock, dec=dec_mock, early_setups={"best_setup": plan})

        # Wajib menyatakan secara eksplisit: BELUM DIPASANG DI MT5
        self.assertIn("READY — PENDING PLAN (BELUM DIPASANG DI MT5)", rendered)
        self.assertIn("pasang limit manual hanya jika siap menanggung risiko", rendered)

    # -------------------------------------------------------------------------
    # 9. Suggested Expiration 1 Jam Dinamis
    # -------------------------------------------------------------------------
    def test_09_suggested_expiration_one_hour(self):
        """9. Selisih 1 jam menghasilkan teks '1 jam' (Snapshot 14:46 -> Expiry 15:46)."""
        base_t = "2026-11-12 14:46"
        target_t = "2026-11-12 15:46"

        res = calculate_suggested_expiration(base_time=base_t, target_time=target_t)
        self.assertEqual(res["duration_text"], "1 jam")
        self.assertEqual(res["duration_hours"], 1.0)
        self.assertIn("2026-11-12 15:46 (1 jam)", res["suggested_expiration"])

    # -------------------------------------------------------------------------
    # 10. Suggested Expiration 8 Jam Dinamis
    # -------------------------------------------------------------------------
    def test_10_suggested_expiration_eight_hours(self):
        """10. Selisih 8 jam menghasilkan teks '8 jam' (Snapshot 14:46 -> Expiry 22:46)."""
        base_t = "2026-11-12 14:46"
        target_t = "2026-11-12 22:46"

        res = calculate_suggested_expiration(base_time=base_t, target_time=target_t)
        self.assertEqual(res["duration_text"], "8 jam")
        self.assertEqual(res["duration_hours"], 8.0)
        self.assertIn("2026-11-12 22:46 (8 jam)", res["suggested_expiration"])

        # Uji juga dengan duration_hours numerik 8
        res2 = calculate_suggested_expiration(base_time=base_t, duration_hours=8)
        self.assertEqual(res2["duration_text"], "8 jam")
        self.assertIn("(8 jam)", res2["suggested_expiration"])
        self.assertGreaterEqual(res2["expires_at"], base_t)

    # -------------------------------------------------------------------------
    # 11. Stop Loss dan Invalidation Level Memiliki Buffer Valid
    # -------------------------------------------------------------------------
    def test_11_sl_different_from_invalidation_with_buffer(self):
        """11. SL tidak sama dengan invalidation; Demand: SL < Invalidation; Supply: SL > Invalidation."""
        # A. Demand Setup
        h1_bull = _make_h1_with_high_target()
        frames_bull = {"H1": h1_bull, "H4": _make_h4_bullish()}
        zone_d = _make_test_zone("DEMAND", 2000.0, 2005.0, sl=0.0, bos_candle_time=h1_bull.index[25])
        plan_d = plan_pending_order(zone_d, current_price=2015.0, frames=frames_bull, htf_bias="BULLISH", m30_status="CONFIRMED")

        self.assertTrue(plan_d["valid"])
        inval_d = plan_d["structural_invalidation"]
        sl_d = plan_d["sl"]
        buf_d = plan_d["invalidation_buffer"]
        self.assertEqual(inval_d, 2000.0)
        self.assertLess(sl_d, inval_d)  # SL < Invalidation
        self.assertGreater(buf_d, 0.0)
        self.assertEqual(round(inval_d - sl_d, 2), buf_d)

        # B. Supply Setup
        h1_bear = _make_h1_with_low_target()
        frames_bear = {"H1": h1_bear, "H4": _make_h4_bearish()}
        zone_s = _make_test_zone("SUPPLY", 2030.0, 2035.0, sl=0.0, bos_candle_time=h1_bear.index[25])
        plan_s = plan_pending_order(zone_s, current_price=2015.0, frames=frames_bear, htf_bias="BEARISH", m30_status="CONFIRMED")

        self.assertTrue(plan_s["valid"])
        inval_s = plan_s["structural_invalidation"]
        sl_s = plan_s["sl"]
        buf_s = plan_s["invalidation_buffer"]
        self.assertEqual(inval_s, 2035.0)
        self.assertGreater(sl_s, inval_s)  # SL > Invalidation
        self.assertGreater(buf_s, 0.0)
        self.assertEqual(round(sl_s - inval_s, 2), buf_s)

    # -------------------------------------------------------------------------
    # 12. Posisi dan Order MT5 Tanpa SL Menghasilkan CRITICAL WARNING
    # -------------------------------------------------------------------------
    def test_12_mt5_orders_without_sl_trigger_critical_warning(self):
        """12. Posisi atau order MT5 tanpa Stop Loss memunculkan CRITICAL WARNING eksplisit."""
        mock_acc = MagicMock(login=112233, name="Trader Real", server="Broker-Server", currency="USD",
                             balance=50000.0, equity=50000.0, profit=0.0, margin_free=50000.0, trade_mode=2)
        mock_sym = MagicMock(digits=2, point=0.01)
        mock_tick = MagicMock(bid=2020.0, ask=2020.3, time=1700000000)

        # 1 Running position tanpa SL
        mock_pos = MagicMock(ticket=101010, symbol="XAUUSD", type=0, price_open=2015.0, sl=0.0, tp=0.0, volume=1.0, profit=500.0)
        # 1 Pending order tanpa SL
        mock_ord = MagicMock(ticket=202020, symbol="XAUUSD", type=2, price_open=2000.0, sl=0.0, tp=0.0,
                             volume_initial=0.5, volume_current=0.5, time_expiration=0)

        with patch("modules.mt5_conn.pastikan_mt5_aktif", return_value=True), \
             patch("modules.mt5_conn.mt5.account_info", return_value=mock_acc), \
             patch("modules.mt5_conn.mt5.symbol_info", return_value=mock_sym), \
             patch("modules.mt5_conn.mt5.symbol_info_tick", return_value=mock_tick), \
             patch("modules.mt5_conn.mt5.positions_get", return_value=[mock_pos]), \
             patch("modules.mt5_conn.mt5.orders_get", return_value=[mock_ord]), \
             patch("modules.mt5_conn.mt5.history_deals_get", return_value=[]):
            txt = cek_posisi_terbuka_mt5(2020.0)

            # Header [MT5 ACTUAL ORDER STATUS]
            self.assertIn("[MT5 ACTUAL ORDER STATUS]", txt)
            # Critical Warning untuk Posisi
            self.assertIn("CRITICAL WARNING: Posisi #101010 tidak memiliki Stop Loss!", txt)
            # Critical Warning untuk Pending Order
            self.assertIn("CRITICAL WARNING: Pending Order #202020 tidak memiliki Stop Loss!", txt)

    # -------------------------------------------------------------------------
    # 13. Gemini Output Integrity (Otoritas Python Tidak Dapat Diubah)
    # -------------------------------------------------------------------------
    def test_13_gemini_cannot_alter_decision_prices_or_orders(self):
        """13. Gemini tidak dapat mengubah keputusan WAIT menjadi sinyal, atau mengarang harga."""
        dec_wait = {"status": "WAIT", "direction": None, "reasons": ["Zona belum tercapai"]}
        hallucinated_ai_text = """
### [SINYAL 1.0%] BUY SEKARANG
- Rekomendasi Utama: BUY LIMIT 2010.00
- Area Entry: 2010.00 - 2012.00
- Stop Loss: 2000.00
- Target TP: TP1: 2030.00, TP2: 2040.00
[RADAR_TARGET]
ZONA_ATAS: 2035.00-2040.00
ZONA_BAWAH: 2000.00-2005.00
SINYAL_AKTIF: BUY | ENTRY: 2010.00-2012.00 | SL: 2000.00 | TP1: 2030.00 | TP2: 2040.00
[/RADAR_TARGET]
"""
        sanitized = sanitasi_output_gemini(hallucinated_ai_text, dec=dec_wait)

        # Verifikasi sinyal BUY dinetralisir
        self.assertNotIn("[SINYAL 1.0%] BUY", sanitized)
        self.assertIn("[STATUS TRADING: WAIT", sanitized)
        self.assertIn("SINYAL_AKTIF: WAIT", sanitized)
        self.assertIn("[OTORITAS PYTHON SCANNER — KEPUTUSAN FINAL: WAIT]", sanitized)

    # -------------------------------------------------------------------------
    # 14. Unit Test Berjalan Otomatis & Offline
    # -------------------------------------------------------------------------
    def test_14_all_unit_tests_run_offline_no_live_mt5_or_gemini(self):
        """14. Seluruh modul dan kalkulator berfungsi secara deterministik tanpa live network atau MT5."""
        res_dist = calculate_zone_distance(2000.0, 2010.0, 2025.0)
        self.assertEqual(res_dist.status, "BELOW_PRICE")
        self.assertEqual(res_dist.distance, 15.0)

        exp = calculate_suggested_expiration("2026-11-12 10:00", 4)
        self.assertEqual(exp["duration_text"], "4 jam")
        self.assertEqual(exp["expires_at"], "2026-11-12 14:00")

    # -------------------------------------------------------------------------
    # 15. Format Output Scan Ringkas (5 Bagian Utama)
    # -------------------------------------------------------------------------
    def test_15_scan_presentation_concise_5_parts(self):
        """15. Format scan default (detailed=False) menyusun 5 bagian ringkas, tanpa kandidat zona default."""
        mock_scan = {
            "symbol": "XAUUSD", "current_price": 2020.0, "snapshot_price": 2020.0,
            "snapshot_time": "2026-11-12 16:00:00 WIB",
            "bid": 2019.8, "ask": 2020.2,
            "price_source": "MT5 Live Tick (Bid: 2019.80 | Ask: 2020.20) — Presisi Jarak terhadap Proximal Edge",
            "active_zones": [
                {"zone_id": "H1_SUPPLY_1", "zone_type": "SUPPLY", "price_low": 2030.0, "price_high": 2035.0, "timeframe": "H1", "grade": "A+", "score": 90.0, "freshness": "FRESH", "invalidation_price": 2035.0},
                {"zone_id": "H1_DEMAND_1", "zone_type": "DEMAND", "price_low": 2000.0, "price_high": 2005.0, "timeframe": "H1", "grade": "A", "score": 80.0, "freshness": "FRESH", "invalidation_price": 1998.0},
            ],
            "htf_context": {"H4": "BEARISH", "H1": "BEARISH", "M30": "BEARISH", "has_conflict": False},
            "data_ok": True,
        }
        mock_dec = {
            "status": "WAIT", "direction": None, "reasons": ["Menunggu harga mencapai supply"],
            "stages": {
                "zone_touch": {"status": "WAIT"}, "liquidity_sweep": {"status": "WAIT"},
                "structure_trigger": {"status": "WAIT"}, "displacement": {"status": "WAIT"},
                "risk_reward": {"status": "NOT_CHECKED"},
            },
            "current_price": 2020.0,
        }
        res_concise = format_scan_presentation(scan=mock_scan, dec=mock_dec, detailed=False)

        # 5 Bagian Utama
        self.assertIn("A. KONDISI PASAR", res_concise)
        self.assertIn("B. RADAR ZONE", res_concise)
        self.assertIn("C. ENTRY PLAN", res_concise)
        self.assertIn("D. FINAL DECISION", res_concise)
        self.assertIn("E. RISK WARNING", res_concise)

        # Kondisi pasar ringkas
        self.assertIn("Multi-TF Bias    : H4: BEARISH | H1: BEARISH | M30: UNCONFIRMED | M5: WAIT", res_concise)
        self.assertIn("Sumber Harga     : MT5 Live Tick", res_concise)

        # Kandidat zona TIDAK ditampilkan pada default ringkas
        self.assertNotIn("[Kandidat Zona Pemantauan — BUKAN Sinyal Entry / Pending Order Aktif]", res_concise)
        self.assertNotIn("Kandidat Demand  :", res_concise)
        self.assertNotIn("Kandidat Supply  :", res_concise)

        # Primary dan Countertrend zone tetap tampil
        self.assertIn("PRIMARY DIRECTIONAL ZONE", res_concise)
        self.assertIn("Primary Radar Zone", res_concise)
        self.assertIn("H1 SUPPLY 2030.00 - 2035.00", res_concise)
        self.assertIn("COUNTERTREND / REACTION ZONE", res_concise)
        self.assertIn("H1 DEMAND 2000.00 - 2005.00", res_concise)

        # Entry plan jelas memisahkan Early Pending dan Confirmed
        self.assertIn("A. EARLY PENDING PLAN", res_concise)
        self.assertIn("B. CONFIRMED ENTRY", res_concise)

        # Peringatan risiko ringkas
        self.assertIn("Peringatan Risiko: Pergerakan XAUUSD memiliki volatilitas tinggi.", res_concise)

    # -------------------------------------------------------------------------
    # 16. Format Output Scan Detail (detailed=True)
    # -------------------------------------------------------------------------
    def test_16_scan_presentation_detailed_mode(self):
        """16. Format scan detail (detailed=True) menampilkan seluruh kandidat zona, breakdown skor & level invalidasi."""
        mock_scan = {
            "symbol": "XAUUSD", "current_price": 2020.0, "snapshot_price": 2020.0,
            "snapshot_time": "2026-11-12 16:00:00 WIB",
            "active_zones": [
                {"zone_id": "H1_SUPPLY_1", "zone_type": "SUPPLY", "price_low": 2030.0, "price_high": 2035.0, "timeframe": "H1", "grade": "A+", "score": 90.0, "freshness": "FRESH", "invalidation_price": 2035.0},
                {"zone_id": "H1_DEMAND_1", "zone_type": "DEMAND", "price_low": 2000.0, "price_high": 2005.0, "timeframe": "H1", "grade": "A", "score": 80.0, "freshness": "FRESH", "invalidation_price": 1998.0},
            ],
            "htf_context": {"H4": "BEARISH", "H1": "BEARISH", "M30": "BEARISH", "has_conflict": False},
            "data_ok": True,
        }
        mock_dec = {
            "status": "WAIT", "direction": None, "reasons": ["Menunggu harga mencapai supply"],
            "stages": {
                "zone_touch": {"status": "WAIT"}, "liquidity_sweep": {"status": "WAIT"},
                "structure_trigger": {"status": "WAIT"}, "displacement": {"status": "WAIT"},
                "risk_reward": {"status": "NOT_CHECKED"},
            },
            "current_price": 2020.0,
        }
        res_detail = format_scan_presentation(scan=mock_scan, dec=mock_dec, detailed=True)

        # Memuat seluruh kandidat zona
        self.assertIn("[Kandidat Zona Pemantauan — BUKAN Sinyal Entry / Pending Order Aktif]", res_detail)
        self.assertIn("Kandidat Demand  : H1 DEMAND 2000.00 - 2005.00", res_detail)
        self.assertIn("Kandidat Supply  : H1 SUPPLY 2030.00 - 2035.00", res_detail)

        # Extended risk warning
        self.assertIn("Manajemen Risiko Modal:", res_detail)
        self.assertIn("0,5%: risiko lebih konservatif", res_detail)
        self.assertIn("1,0%: risiko lebih tinggi", res_detail)

    # -------------------------------------------------------------------------
    # 17. Startup Presentation - All Systems Ready
    # -------------------------------------------------------------------------
    def test_17_startup_presentation_all_systems_ready(self):
        """17. Format startup awal: 4 bagian (Header, System Status, Market Overview, Command Menu) dengan status READY."""
        class MockTick:
            bid = 2020.15
            ask = 2020.45

        out = format_startup_presentation(
            symbol="XAUUSD",
            mt5_connected=True,
            tick=MockTick(),
            engine_ready=True,
            ai_status="READY",
            version=APP_VERSION,
        )

        # 1. HEADER
        self.assertIn("1. HEADER", out)
        self.assertIn("XAUUSD SMC SCANNER", out)
        self.assertIn("SMC Multi-Timeframe Analysis & Institutional Risk Management", out)
        self.assertIn(APP_VERSION, out)

        # 2. SYSTEM STATUS
        self.assertIn("2. SYSTEM STATUS", out)
        self.assertIn("• MT5 Connection       : CONNECTED", out)
        self.assertIn("• Market Data          : READY (XAUUSD)", out)
        self.assertIn("• Python Scanner Engine: READY", out)
        self.assertIn("• Gemini AI            : READY", out)

        # 3. MARKET OVERVIEW
        self.assertIn("3. MARKET OVERVIEW", out)
        self.assertIn("• Simbol               : XAUUSD", out)
        self.assertIn("• Harga Terkini        : Bid 2020.15 | Ask 2020.45 (Spread: 0.30 pt)", out)
        self.assertIn("• Waktu Update         :", out)
        self.assertIn("WIB", out)

        # 4. COMMAND MENU
        self.assertIn("4. COMMAND MENU", out)
        self.assertIn("• scan       :", out)
        self.assertIn("• scan detail:", out)
        self.assertIn("• zona       :", out)
        self.assertIn("• pantau     :", out)
        self.assertIn("• jurnal     :", out)
        self.assertIn("• audit      :", out)
        self.assertIn("• refresh    :", out)
        self.assertIn("• help       :", out)
        self.assertIn("• exit       :", out)

        # Bebas dari analisa teknikal panjang, info saldo akun, order ticket, dsb
        self.assertNotIn("Balance", out)
        self.assertNotIn("Equity", out)
        self.assertNotIn("Ticket", out)
        self.assertNotIn("Order Block M5", out)
        self.assertNotIn("CHoCH", out)

    # -------------------------------------------------------------------------
    # 18. Startup Presentation - Offline / Disconnected Fallback
    # -------------------------------------------------------------------------
    def test_18_startup_presentation_offline_fallback(self):
        """18. Format startup awal ketika offline: status DISCONNECTED & DATA UNAVAILABLE."""
        out = format_startup_presentation(
            symbol="XAUUSD",
            mt5_connected=False,
            tick=None,
            engine_ready=True,
            ai_status="DISCONNECTED",
            version=APP_VERSION,
        )

        self.assertIn("1. HEADER", out)
        self.assertIn("2. SYSTEM STATUS", out)
        self.assertIn("• MT5 Connection       : DISCONNECTED", out)
        self.assertIn("• Market Data          : DISCONNECTED", out)
        self.assertIn("• Python Scanner Engine: READY", out)
        self.assertIn("• Gemini AI            : DISCONNECTED", out)

        self.assertIn("3. MARKET OVERVIEW", out)
        self.assertIn("• Harga Terkini        : DATA UNAVAILABLE", out)
        self.assertIn("• Waktu Update         : DATA UNAVAILABLE", out)

        self.assertIn("4. COMMAND MENU", out)

    # -------------------------------------------------------------------------
    # 19. Startup Presentation - Error States
    # -------------------------------------------------------------------------
    def test_19_startup_presentation_error_states(self):
        """19. Format startup awal ketika komponen sistem mengalami ERROR."""
        out = format_startup_presentation(
            symbol="XAUUSD",
            mt5_connected=False,
            tick=None,
            engine_ready=False,
            ai_status="ERROR",
            version=APP_VERSION,
        )

        self.assertIn("• Python Scanner Engine: ERROR", out)
        self.assertIn("• Gemini AI            : ERROR", out)


if __name__ == "__main__":
    unittest.main()

