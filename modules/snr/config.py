# -*- coding: utf-8 -*-
"""
config.py — Konfigurasi Independen SNR Advanced Scanner (Support & Resistance Advanced).
Terpisah sepenuhnya dari konfigurasi SMC Scanner agar perubahan parameter SNR
tidak pernah memengaruhi perhitungan SMC.
"""
import copy
import os

_DIR_SNR = os.path.dirname(os.path.abspath(__file__))
_DIR_MODULES = os.path.dirname(_DIR_SNR)
_DIR_PROJECT = os.path.dirname(_DIR_MODULES)

SNR_SYMBOL_DEFAULT = "XAUUSD"
SNR_TIMEFRAMES = ("H4", "H1", "M30", "M5")
SNR_STRUCTURAL_TIMEFRAMES = ("H4", "H1", "M30")
SNR_ENTRY_TIMEFRAME = "M5"

FILE_LOG_SNR_ERROR = os.path.join(_DIR_PROJECT, "snr_scanner_error.log")
FILE_LOG_SNR_AUDIT = os.path.join(_DIR_PROJECT, "snr_scanner_audit.log")

SNR_TIMEFRAME_BARS = {
    "H4": 220,
    "H1": 240,
    "M30": 240,
    "M5": 240,
}

SNR_SWING_PARAMS = {
    "H4": {
        "left_bars": 3,
        "right_bars": 3,
        "major_atr_mult": 1.20,
        "min_pivot_distance_atr": 0.30,
        "duplicate_tolerance_atr": 0.22,
        "reaction_lookahead_bars": 8,
    },
    "H1": {
        "left_bars": 3,
        "right_bars": 3,
        "major_atr_mult": 1.10,
        "min_pivot_distance_atr": 0.28,
        "duplicate_tolerance_atr": 0.20,
        "reaction_lookahead_bars": 8,
    },
    "M30": {
        "left_bars": 2,
        "right_bars": 2,
        "major_atr_mult": 1.00,
        "min_pivot_distance_atr": 0.25,
        "duplicate_tolerance_atr": 0.18,
        "reaction_lookahead_bars": 8,
    },
    "M5": {
        "left_bars": 2,
        "right_bars": 2,
        "major_atr_mult": 0.90,
        "min_pivot_distance_atr": 0.22,
        "duplicate_tolerance_atr": 0.16,
        "reaction_lookahead_bars": 6,
    },
}

SNR_ZONE_PARAMS = {
    "H4": {
        "zone_half_width_atr": 0.20,
        "min_zone_width_atr": 0.16,
        "max_zone_width_atr": 0.85,
        "max_zone_width_points": 15.0,
        "cluster_tolerance_atr": 0.35,
        "tf_weight": 1.00,
    },
    "H1": {
        "zone_half_width_atr": 0.18,
        "min_zone_width_atr": 0.15,
        "max_zone_width_atr": 0.78,
        "max_zone_width_points": 10.0,
        "cluster_tolerance_atr": 0.32,
        "tf_weight": 0.88,
    },
    "M30": {
        "zone_half_width_atr": 0.17,
        "min_zone_width_atr": 0.14,
        "max_zone_width_atr": 0.72,
        "max_zone_width_points": 7.0,
        "cluster_tolerance_atr": 0.28,
        "tf_weight": 0.78,
    },
    "M5": {
        "zone_half_width_atr": 0.15,
        "min_zone_width_atr": 0.12,
        "max_zone_width_atr": 0.65,
        "max_zone_width_points": 3.8,
        "cluster_tolerance_atr": 0.25,
        "tf_weight": 0.60,
    },
}

SNR_TOUCH_PARAMS = {
    "min_touch_separation_bars": 2,
    "valid_rejection_bounce_atr": 0.32,
    "weak_touch_bounce_atr": 0.15,
    "max_valid_penetration_atr": 0.45,
    "broken_close_beyond_atr": 0.15,
    "invalidation_close_beyond_atr": 0.42,
    "consecutive_closes_for_invalidation": 2,
}

SNR_BREAKOUT_PARAMS = {
    "valid_close_penetration_atr": 0.15,
    "strong_close_penetration_atr": 0.28,
    "min_breakout_body_atr": 0.42,
    "min_breakout_body_ratio": 0.48,
    "volume_confirm_mult": 1.05,
    "lookback_bars": 12,
}

SNR_RETEST_PARAMS = {
    "retest_proximity_atr": 0.28,
    "retest_max_lookback_bars": 28,
    "retest_confirm_bounce_atr": 0.18,
    "retest_fail_close_inside_atr": 0.18,
}

SNR_REJECTION_PARAMS = {
    "min_rejection_wick_to_body": 1.10,
    "min_rejection_wick_to_range": 0.36,
    "weak_rejection_wick_to_range": 0.22,
    "rejection_zone_proximity_atr": 0.25,
    "min_candle_range_atr": 0.25,
    "lookback_bars": 5,
}

SNR_MOMENTUM_PARAMS = {
    "strong_body_atr": 0.52,
    "moderate_body_atr": 0.30,
    "strong_range_atr": 0.80,
    "min_consecutive_bars": 2,
    "velocity_lookback_bars": 4,
    "volume_lookback_bars": 20,
}

SNR_CONFLUENCE_PARAMS = {
    "confluence_gap_tolerance_atr": 0.22,
    "max_confluence_zone_width_atr": 0.90,
    "max_confluence_zone_width_points": 14.0,
}

SNR_SCORING_WEIGHTS = {
    "swing_structural_strength": 20.0,
    "historical_price_reaction": 20.0,
    "touch_quality": 15.0,
    "multi_tf_confluence": 15.0,
    "rejection_breakout_quality": 15.0,
    "freshness": 10.0,
    "momentum_confirmation": 5.0,
}

SNR_GRADE_THRESHOLDS = {
    "A": 80.0,
    "B": 70.0,
    "C": 60.0,
}

SNR_RISK_PARAMS = {
    "min_rr_ratio": 1.5,
    "sl_volatility_buffer_atr": 0.25,
    "min_sl_buffer_points": 0.60,
    "max_entry_distance_atr": 0.65,
    "default_spread_points": 0.25,
    "default_tick_size": 0.01,
    "default_tick_value": 1.0,
    "default_contract_size": 100.0,
    "default_digits": 2,
}

SNR_RADAR_PARAMS = {
    "approaching_distance_atr": 0.60,
    "touch_distance_atr": 0.08,
}

DEFAULT_SNR_CONFIG = {
    "symbol": SNR_SYMBOL_DEFAULT,
    "timeframes": SNR_TIMEFRAMES,
    "structural_timeframes": SNR_STRUCTURAL_TIMEFRAMES,
    "entry_timeframe": SNR_ENTRY_TIMEFRAME,
    "timeframe_bars": SNR_TIMEFRAME_BARS,
    "swing": SNR_SWING_PARAMS,
    "zone": SNR_ZONE_PARAMS,
    "touch": SNR_TOUCH_PARAMS,
    "breakout": SNR_BREAKOUT_PARAMS,
    "retest": SNR_RETEST_PARAMS,
    "rejection": SNR_REJECTION_PARAMS,
    "momentum": SNR_MOMENTUM_PARAMS,
    "confluence": SNR_CONFLUENCE_PARAMS,
    "scoring_weights": SNR_SCORING_WEIGHTS,
    "grades": SNR_GRADE_THRESHOLDS,
    "risk": SNR_RISK_PARAMS,
    "radar": SNR_RADAR_PARAMS,
}


def get_snr_config(overrides=None):
    """Mengembalikan salinan konfigurasi SNR dengan opsi override aman untuk pengujian."""
    cfg = copy.deepcopy(DEFAULT_SNR_CONFIG)
    if overrides and isinstance(overrides, dict):
        for k, v in overrides.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                for sub_k, sub_v in v.items():
                    if isinstance(sub_v, dict) and isinstance(cfg[k].get(sub_k), dict):
                        cfg[k][sub_k].update(sub_v)
                    else:
                        cfg[k][sub_k] = copy.deepcopy(sub_v)
            else:
                cfg[k] = copy.deepcopy(v)
    return cfg

