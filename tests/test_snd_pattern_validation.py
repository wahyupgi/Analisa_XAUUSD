# -*- coding: utf-8 -*-
"""
tests/test_snd_pattern_validation.py
Unit & Integration Tests untuk deteksi & validasi kualitas pola SND (RBR, DBR, RBD, DBD)
pada SMC Scanner, Dynamic Lot Manager, Analisis Teknikal, dan SNR Advanced Scanner.
100% Offline / Deterministik — tidak pernah menyentuh koneksi atau order MT5 live!
"""
from __future__ import annotations

import math
import os
import sys
import unittest

import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from modules.analisis_teknikal import supply_demand
from modules.config import get_auto_trading_config, get_snd_pattern_config
from modules.dynamic_lot_manager import select_target_lot, validate_strong_zone
from modules.mt5_conn import tambah_indikator
from modules.snr.config import get_snr_config
from modules.snr.formatter import build_snr_zone_reason
from modules.snr.resistance_detection import detect_resistance_candidates
from modules.snr.support_detection import detect_support_candidates
from modules.snr.swing_detection import detect_swings as snr_detect_swings
from modules.snr.zone_clustering import cluster_zones
from modules.snr.zone_scoring import score_zone as snr_score_zone
from modules.trade_validator import _snr_zone_to_candidate_dict
from modules.utils import evaluate_snd_base_pattern
from modules.zone_formatter import build_smc_zone_reason, format_zone_evidence
from modules.zone_scanner import (
    Zone,
    _zone_from_dict,
    detect_fvgs,
    detect_liquidity_sweeps,
    detect_market_structure,
    detect_order_blocks,
    merge_overlapping_zones,
    score_zone as smc_score_zone,
)


def _make_df(ohlc_rows, freq="1h", start="2026-10-01 00:00:00"):
    idx = pd.date_range(start=start, periods=len(ohlc_rows), freq=freq)
    arr = np.asarray(ohlc_rows, dtype=float)
    df = pd.DataFrame(
        {
            "Open": arr[:, 0],
            "High": arr[:, 1],
            "Low": arr[:, 2],
            "Close": arr[:, 3],
            "Volume": np.full(len(ohlc_rows), 1000.0),
        },
        index=idx,
    )
    return tambah_indikator(df)


class TestSndBasePatternCore(unittest.TestCase):
    def test_rbr_dbr_rbd_dbd_classification_and_tight_quality(self):
        cfg = get_snd_pattern_config()
        atrs = np.full(20, 2.0)

        # 1. RBR (Rally -> Tight Base -> Rally)
        rbr_rows = [
            (3600.0, 3601.5, 3599.8, 3601.2),
            (3601.2, 3603.5, 3601.0, 3603.2),
            (3603.2, 3605.8, 3603.0, 3605.5),
            (3605.5, 3607.5, 3605.2, 3607.0),
            # Base (2 tight pause bars around 3606.5 - 3607.6)
            (3607.0, 3607.5, 3606.6, 3606.8),
            (3606.8, 3607.4, 3606.5, 3607.1),  # anchor_idx = 5
            # Leg-Out Rally
            (3607.1, 3610.5, 3607.0, 3610.2),
            (3610.2, 3613.5, 3609.8, 3613.0),
        ]
        arr = np.asarray(rbr_rows, dtype=float)
        res_rbr = evaluate_snd_base_pattern(
            arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], atrs[: len(arr)],
            anchor_idx=5, zone_direction="DEMAND", max_eval_idx=7, params=cfg,
        )
        self.assertEqual(res_rbr["pattern"], "RBR")
        self.assertEqual(res_rbr["pattern_family"], "CONTINUATION")
        self.assertEqual(res_rbr["base_quality"], "TIGHT")
        self.assertEqual(res_rbr["base_candles"], 2)
        self.assertGreaterEqual(res_rbr["leg_out_to_base_ratio"], 2.0)

        # 2. DBR (Drop -> Tight Base -> Rally)
        dbr_rows = [
            (3620.0, 3620.4, 3617.5, 3617.8),
            (3617.8, 3618.0, 3614.8, 3615.0),
            (3615.0, 3615.3, 3612.2, 3612.5),
            # Base (2 tight bars at bottom 3611.8 - 3612.8)
            (3612.5, 3612.8, 3611.9, 3612.1),
            (3612.1, 3612.7, 3611.8, 3612.4),  # anchor_idx = 4
            # Leg-Out Rally
            (3612.4, 3616.2, 3612.2, 3615.9),
            (3615.9, 3619.5, 3615.5, 3619.0),
        ]
        arr = np.asarray(dbr_rows, dtype=float)
        res_dbr = evaluate_snd_base_pattern(
            arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], atrs[: len(arr)],
            anchor_idx=4, zone_direction="DEMAND", max_eval_idx=6, params=cfg,
        )
        self.assertEqual(res_dbr["pattern"], "DBR")
        self.assertEqual(res_dbr["pattern_family"], "REVERSAL")
        self.assertEqual(res_dbr["base_quality"], "TIGHT")

        # 3. RBD (Rally -> Tight Base -> Drop)
        rbd_rows = [
            (3600.0, 3603.0, 3599.8, 3602.7),
            (3602.7, 3605.8, 3602.5, 3605.5),
            (3605.5, 3608.2, 3605.2, 3607.9),
            # Base at top 3607.4 - 3608.4
            (3607.9, 3608.3, 3607.5, 3607.7),
            (3607.7, 3608.4, 3607.4, 3607.6),  # anchor_idx = 4
            # Leg-Out Drop
            (3607.6, 3607.8, 3604.0, 3604.2),
            (3604.2, 3604.5, 3600.8, 3601.0),
        ]
        arr = np.asarray(rbd_rows, dtype=float)
        res_rbd = evaluate_snd_base_pattern(
            arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], atrs[: len(arr)],
            anchor_idx=4, zone_direction="SUPPLY", max_eval_idx=6, params=cfg,
        )
        self.assertEqual(res_rbd["pattern"], "RBD")
        self.assertEqual(res_rbd["pattern_family"], "REVERSAL")
        self.assertEqual(res_rbd["base_quality"], "TIGHT")

        # 4. DBD (Drop -> Tight Base -> Drop)
        dbd_rows = [
            (3630.0, 3630.3, 3627.2, 3627.5),
            (3627.5, 3627.8, 3624.8, 3625.0),
            (3625.0, 3625.2, 3622.5, 3622.8),
            # Base around 3622.2 - 3623.2
            (3622.8, 3623.2, 3622.3, 3623.0),
            (3623.0, 3623.1, 3622.2, 3622.5),  # anchor_idx = 4
            # Leg-Out Drop
            (3622.5, 3622.7, 3619.0, 3619.2),
            (3619.2, 3619.5, 3616.0, 3616.3),
        ]
        arr = np.asarray(dbd_rows, dtype=float)
        res_dbd = evaluate_snd_base_pattern(
            arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], atrs[: len(arr)],
            anchor_idx=4, zone_direction="SUPPLY", max_eval_idx=6, params=cfg,
        )
        self.assertEqual(res_dbd["pattern"], "DBD")
        self.assertEqual(res_dbd["pattern_family"], "CONTINUATION")
        self.assertEqual(res_dbd["base_quality"], "TIGHT")

    def test_no_lookahead_and_choppy_detection_and_failsafe_inputs(self):
        cfg = get_snd_pattern_config()
        atrs = np.full(30, 2.0)

        base_rows = [
            (3620.0, 3620.3, 3617.0, 3617.2),
            (3617.2, 3617.5, 3614.0, 3614.3),
            (3614.3, 3614.8, 3613.9, 3614.5),  # anchor = 2
            (3614.5, 3618.5, 3614.3, 3618.2),
            (3618.2, 3621.5, 3618.0, 3621.0),  # max_eval_idx = 4
        ]
        arr1 = np.asarray(base_rows, dtype=float)
        r1 = evaluate_snd_base_pattern(
            arr1[:, 0], arr1[:, 1], arr1[:, 2], arr1[:, 3], atrs[: len(arr1)],
            anchor_idx=2, zone_direction="DEMAND", max_eval_idx=4, params=cfg,
        )
        # Tambahkan candle ekstrem setelah max_eval_idx -> tidak boleh mengubah hasil sedikit pun!
        future_rows = base_rows + [
            (3621.0, 3622.0, 3550.0, 3555.0),
            (3555.0, 3700.0, 3540.0, 3690.0),
        ]
        arr2 = np.asarray(future_rows, dtype=float)
        r2 = evaluate_snd_base_pattern(
            arr2[:, 0], arr2[:, 1], arr2[:, 2], arr2[:, 3], atrs[: len(arr2)],
            anchor_idx=2, zone_direction="DEMAND", max_eval_idx=4, params=cfg,
        )
        self.assertEqual(r1, r2)

        # Kongesti berkepanjangan (> max_base_bars + 1 candle sideways tumpang-tindih) -> WIDE_CHOPPY
        choppy_rows = [(3610.0, 3611.2, 3609.5, 3610.3)] * 8 + [
            (3610.3, 3614.0, 3610.1, 3613.8),
        ]
        arr_c = np.asarray(choppy_rows, dtype=float)
        rc = evaluate_snd_base_pattern(
            arr_c[:, 0], arr_c[:, 1], arr_c[:, 2], arr_c[:, 3], atrs[: len(arr_c)],
            anchor_idx=7, zone_direction="DEMAND", max_eval_idx=8, params=cfg,
        )
        self.assertEqual(rc["base_quality"], "WIDE_CHOPPY")

        # Fail-safe pada input rusak / kosong / NaN / out-of-bounds (anti-crash untuk auto real)
        for bad_args in (
            ([], [], [], [], [], 0, "DEMAND"),
            ([math.nan] * 6, [math.nan] * 6, [math.nan] * 6, [math.nan] * 6, [0.0] * 6, 2, "SUPPLY"),
            (arr1[:, 0], arr1[:, 1], arr1[:, 2], arr1[:, 3], atrs[: len(arr1)], 99, "DEMAND"),
            (arr1[:, 0], arr1[:, 1], arr1[:, 2], arr1[:, 3], atrs[: len(arr1)], -5, "DEMAND"),
        ):
            safe_res = evaluate_snd_base_pattern(*bad_args, params=cfg)
            self.assertEqual(safe_res["pattern"], "UNCLASSIFIED")
            self.assertFalse(safe_res["is_valid_pattern"])


class TestSmcAndDynamicLotSndIntegration(unittest.TestCase):
    def _make_zone(self, score=80.0, disp=1.6, snd_pat="", snd_qual="", bos_type="BOS", sweep="NONE"):
        return Zone(
            zone_id="ZONE_H1_DEMAND_TEST",
            symbol="XAUUSDc",
            timeframe="H1",
            zone_type="DEMAND",
            price_low=3600.0,
            price_high=3602.0,
            midpoint=3601.0,
            score=score,
            grade="A",
            origin_bos={"tipe": bos_type, "arah": "bull", "strength_atr": 0.5, "origin_valid": True},
            fvg_confirmation={"status": "UNTOUCHED", "price_low": 3602.0, "price_high": 3604.0},
            displacement_strength=disp,
            freshness="FRESH",
            touch_count=0,
            liquidity_sweep_status=sweep,
            invalidation_price=3600.0,
            is_valid=True,
            rejection_reason="",
            snd_pattern=snd_pat,
            snd_base_quality=snd_qual,
        )

    def test_score_zone_preserves_legacy_and_bounds_0_to_100(self):
        # 1. Legacy zone (tanpa snd_pattern) -> 0.0 adj, tidak mengubah skor lama
        z_legacy = self._make_zone(disp=1.5, snd_pat="", snd_qual="")
        smc_score_zone(z_legacy, htf_alignment={"H4": "ALIGNED", "H1": "ALIGNED", "M30": "ALIGNED"})
        self.assertEqual(z_legacy.score_breakdown["snd_quality_adj"], 0.0)
        base_score = z_legacy.score

        # 2. RBR Tight Base searah BOS + HTF -> mendapat validasi positif terukur
        z_rbr = self._make_zone(disp=1.5, snd_pat="RBR", snd_qual="TIGHT", bos_type="BOS")
        smc_score_zone(z_rbr, htf_alignment={"H4": "ALIGNED", "H1": "ALIGNED", "M30": "ALIGNED"})
        self.assertGreater(z_rbr.score, base_score)
        self.assertLessEqual(z_rbr.score, 100.0)
        self.assertLessEqual(z_rbr.score_breakdown["zone_strength"], 20.0)
        self.assertIn("RBR Tight", build_smc_zone_reason(z_rbr.to_dict(), h4_bias="BULLISH", h1_bias="BULLISH"))
        self.assertIn("SND RBR/TIGHT", format_zone_evidence(z_rbr.to_dict(), detailed=False))

        # 3. Plafon ketat <= 100 meskipun semua komponen maksimal
        z_max = self._make_zone(disp=3.0, snd_pat="DBR", snd_qual="TIGHT", bos_type="BOS", sweep="CONFIRMED_SWEEP_LOW")
        z_max.timeframe = "H4"
        z_max.origin_bos["strength_atr"] = 3.0
        smc_score_zone(z_max, htf_alignment={"H4": "ALIGNED", "H1": "ALIGNED", "M30": "ALIGNED"})
        self.assertLessEqual(z_max.score, 100.0)
        self.assertLessEqual(z_max.score_breakdown["zone_strength"], 20.0)

        # 4. WIDE_CHOPPY base dan DBR lawan tren tanpa Sweep/CHoCH -> dipangkas proporsional
        z_choppy = self._make_zone(disp=1.5, snd_pat="RBR", snd_qual="WIDE_CHOPPY", bos_type="BOS")
        smc_score_zone(z_choppy, htf_alignment={"H4": "ALIGNED", "H1": "ALIGNED", "M30": "ALIGNED"})
        self.assertLess(z_choppy.score, base_score)

        z_counter_dbr = self._make_zone(disp=1.5, snd_pat="DBR", snd_qual="TIGHT", bos_type="BOS", sweep="NONE")
        smc_score_zone(z_counter_dbr, htf_alignment={"H4": "OPPOSED", "H1": "OPPOSED", "M30": "OPPOSED"})
        self.assertLess(z_counter_dbr.score_breakdown["snd_quality_adj"], 0.0)

    def test_strong_zone_validation_blocks_choppy_base_from_005_lot(self):
        cfg = dict(
            get_auto_trading_config(),
            AUTO_LOOSE_MODE=False,
            STRONG_ZONE_SCORE=90.0,
            MEDIUM_ZONE_SCORE=80.0,
            MIN_ZONE_SCORE=70.0,
            LOT_STRONG=0.05,
            LOT_MEDIUM=0.02,
            LOT_STANDARD=0.01,
            MAX_LOT=0.05,
            HTF_STRONG_MAX_LOT=0.05,
            MIN_RR=1.5,
        )
        z_clean = self._make_zone(score=93.0, disp=2.2, snd_pat="RBR", snd_qual="TIGHT")
        val_clean = validate_strong_zone(
            z_clean, entry_mode="ADAPTIVE_ENTRY", htf_aligned=True, has_htf_conflict=False, actual_rr=2.0, cfg=cfg
        )
        self.assertTrue(val_clean["passed"])
        self.assertEqual(select_target_lot(z_clean, "ADAPTIVE_ENTRY", strong_zone_validated=val_clean["passed"], cfg=cfg), 0.05)

        z_choppy = self._make_zone(score=93.0, disp=2.2, snd_pat="RBR", snd_qual="WIDE_CHOPPY")
        val_choppy = validate_strong_zone(
            z_choppy, entry_mode="ADAPTIVE_ENTRY", htf_aligned=True, has_htf_conflict=False, actual_rr=2.0, cfg=cfg
        )
        self.assertFalse(val_choppy["passed"])
        # Tanpa strong_zone_validated, lot otomatis ditahan maksimal 0.02 (tidak pernah bocor ke 0.05)
        self.assertEqual(select_target_lot(z_choppy, "ADAPTIVE_ENTRY", strong_zone_validated=val_choppy["passed"], cfg=cfg), 0.02)

    def test_merge_and_serialization_preserve_snd_fields(self):
        z1 = self._make_zone(score=84.0, disp=2.0, snd_pat="DBR", snd_qual="TIGHT")
        z1.zone_id = "Z_H4"
        z1.timeframe = "H4"
        z1.snd_pattern_family = "REVERSAL"
        z1.snd_base_candles = 2
        z1.snd_leg_out_ratio = 2.8

        z2 = self._make_zone(score=78.0, disp=1.6, snd_pat="UNCLASSIFIED", snd_qual="STANDARD")
        z2.zone_id = "Z_H1"
        z2.timeframe = "H1"
        z2.price_low = 3600.2
        z2.price_high = 3601.8

        merged = merge_overlapping_zones([z1, z2])
        self.assertEqual(len(merged), 1)
        m = merged[0]
        self.assertEqual(m.snd_pattern, "DBR")
        self.assertEqual(m.snd_base_quality, "TIGHT")
        restored = _zone_from_dict(m.to_dict())
        self.assertEqual(restored.snd_pattern, "DBR")
        self.assertEqual(restored.snd_base_quality, "TIGHT")
        self.assertEqual(restored.snd_base_candles, 2)

    def test_detect_order_blocks_and_analisis_teknikal_supply_demand_end_to_end(self):
        # Bangun histori OHLCV uptrend -> swing high -> pullback DBR -> impuls BOS + FVG
        rows = []
        px = 3600.0
        for i in range(25):
            step = 0.8 if (i % 4 != 3) else -0.6
            o = px
            c = px + step
            h = max(o, c) + 0.5
            l = min(o, c) - 0.5
            rows.append((o, h, l, c))
            px = c
        # Bentuk swing high jelas lalu drop ke base lalu impuls kuat menembus swing high (DBR + BOS + FVG)
        rows.extend([
            (px, px + 2.5, px - 0.3, px + 2.2),
            (px + 2.2, px + 4.5, px + 2.0, px + 4.0),  # Swing high ~ px + 4.5
            (px + 4.0, px + 4.2, px + 1.8, px + 2.0),
            (px + 2.0, px + 2.2, px + 0.1, px + 0.3),
            (px + 0.3, px + 0.5, px - 1.5, px - 1.2),
            # Tight base OB lawan (bearish kecil)
            (px - 1.2, px - 0.9, px - 1.8, px - 1.6),
            # Impuls besar naik menciptakan FVG & menembus swing high
            (px - 1.6, px + 2.8, px - 1.6, px + 2.6),
            (px + 2.6, px + 6.8, px + 2.4, px + 6.5),
            (px + 6.5, px + 8.5, px + 5.8, px + 8.0),
            (px + 8.0, px + 9.0, px + 7.5, px + 8.4),
        ])
        df = _make_df(rows, freq="1h")
        events, _, swings = detect_market_structure(df)
        sweeps = detect_liquidity_sweeps(df, swings, lookback=None)
        fvgs = detect_fvgs(df)
        ok, bad = detect_order_blocks(df, events, fvgs, sweeps, "H1")
        all_z = ok + bad
        self.assertTrue(len(all_z) > 0)
        self.assertTrue(any(z.snd_pattern in ("DBR", "RBR") for z in all_z))

        sd_list = supply_demand(df, df["ATR14"].values, impuls=1.2)
        self.assertTrue(len(sd_list) > 0)
        for item in sd_list:
            self.assertIn("pola", item)
            self.assertIn(item["pola"], ("RBR", "DBR", "RBD", "DBD"))


class TestSnrScannerSndIntegration(unittest.TestCase):
    def test_snr_support_resistance_clustering_scoring_and_hybrid_conversion(self):
        cfg = get_snr_config()
        # Buat kurva zig-zag bersih agar menghasilkan Swing High (RBD) & Swing Low (DBR) terkonfirmasi
        prices = [
            3610, 3613, 3616, 3619, 3622,  # Rally ke puncak 3622
            3619, 3615, 3611, 3608, 3605,  # Drop ke lembah 3605
            3609, 3613, 3617, 3620, 3624,  # Rally ke puncak 3624
            3620, 3616, 3612, 3610, 3614,
        ]
        rows = []
        for k, c_val in enumerate(prices):
            o_val = prices[k - 1] if k > 0 else c_val - 1.0
            lo_b, hi_b = min(o_val, c_val), max(o_val, c_val)
            # Buat candle pivot kecil/rapi di puncak (idx 4, 14) & lembah (idx 9)
            if k in (4, 14):
                rows.append((c_val - 0.3, c_val + 0.4, c_val - 0.5, c_val - 0.1))
            elif k == 9:
                rows.append((c_val + 0.3, c_val + 0.5, c_val - 0.4, c_val + 0.2))
            else:
                rows.append((o_val, hi_b + 0.4, lo_b - 0.4, float(c_val)))

        df = _make_df(rows, freq="1h")
        df["Time"] = df.index.strftime("%Y-%m-%d %H:%M:%S")

        sw = snr_detect_swings(df, timeframe="H1", config=cfg)
        sups = detect_support_candidates(df, swing_result=sw, timeframe="H1", config=cfg)
        ress = detect_resistance_candidates(df, swing_result=sw, timeframe="H1", config=cfg)
        self.assertTrue(len(sups) >= 1)
        self.assertTrue(len(ress) >= 1)
        self.assertIn(sups[0]["snd_pattern"], ("DBR", "RBR"))
        self.assertIn(ress[0]["snd_pattern"], ("RBD", "DBD"))

        clust_sup = cluster_zones(sups, timeframe="H1", zone_type="SUPPORT", atr=3.0, config=cfg)
        self.assertTrue(len(clust_sup) >= 1)
        cz = clust_sup[0]
        self.assertTrue(cz["zone_id"].startswith("SNR_H1_SUP_"))
        self.assertIn(cz["snd_pattern"], ("DBR", "RBR"))

        scored = snr_score_zone(cz, config=cfg)
        self.assertLessEqual(scored["score"], 100.0)
        self.assertLessEqual(scored["score_breakdown"]["swing_structural_strength"]["score"], 20.0)
        cz.update(scored)
        reason_txt = build_snr_zone_reason(cz, h4_bias="BULLISH", h1_bias="BULLISH")
        self.assertTrue(any(p in reason_txt for p in ("DBR", "RBR")))

        # Verifikasi konversi kandidat hybrid ke Trade Validator membawa atribut SND utuh
        cand_dict = _snr_zone_to_candidate_dict(cz, symbol="XAUUSDc")
        self.assertIsNotNone(cand_dict)
        self.assertEqual(cand_dict["snd_pattern"], cz["snd_pattern"])
        self.assertEqual(cand_dict["snd_base_quality"], cz["snd_base_quality"])


if __name__ == "__main__":
    unittest.main()
