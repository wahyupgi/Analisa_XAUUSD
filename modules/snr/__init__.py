# -*- coding: utf-8 -*-
"""
modules.snr — Paket Independen SNR Advanced Scanner (Support & Resistance Advanced)
untuk XAUUSD Python Trading Analyzer.
"""
from modules.snr.breakout_detection import detect_zone_breakout
from modules.snr.config import (
    DEFAULT_SNR_CONFIG,
    SNR_STRUCTURAL_TIMEFRAMES,
    SNR_SYMBOL_DEFAULT,
    SNR_TIMEFRAMES,
    get_snr_config,
)
from modules.snr.data_provider import (
    compute_atr14_series,
    compute_atr_diagnostics,
    detect_snr_market_status,
    fetch_snr_market_data,
    get_symbol_spec,
    log_snr_audit,
    log_snr_error,
    prepare_snr_frame,
)
from modules.snr.decision_engine import build_snr_final_decision
from modules.snr.entry_validator import validate_snr_entry
from modules.snr.formatter import (
    format_snr_detail,
    format_snr_presentation,
    format_snr_summary,
)
from modules.snr.market_bias import evaluate_single_tf_snr_bias, evaluate_snr_market_bias
from modules.snr.momentum_analysis import analyze_momentum
from modules.snr.monitor import (
    SnrCycleEngine,
    format_snr_monitor_section,
    mode_pantau_snr,
)
from modules.snr.radar import SnrAlertTracker, evaluate_snr_radar
from modules.snr.rejection_detection import detect_zone_rejection
from modules.snr.resistance_detection import detect_resistance_candidates
from modules.snr.retest_detection import detect_zone_retest
from modules.snr.risk_reward import calculate_snr_risk_reward, round_to_tick
from modules.snr.scanner import (
    evaluate_snr_decision,
    run_snr_scan,
    scan_snr,
    scan_snr_detail,
)
from modules.snr.support_detection import detect_support_candidates
from modules.snr.swing_detection import detect_swings
from modules.snr.touch_analysis import analyze_zone_touches
from modules.snr.zone_clustering import cluster_zones
from modules.snr.zone_ranking import (
    classify_zone_strength_category,
    compute_zone_distance,
    rank_zones,
)
from modules.snr.zone_scoring import (
    assign_snr_grade,
    evaluate_mtf_confluence,
    score_zone,
)
from modules.snr.zone_validation import validate_single_zone, validate_zones

__all__ = [
    "DEFAULT_SNR_CONFIG",
    "SNR_STRUCTURAL_TIMEFRAMES",
    "SNR_SYMBOL_DEFAULT",
    "SNR_TIMEFRAMES",
    "get_snr_config",
    "compute_atr14_series",
    "compute_atr_diagnostics",
    "detect_snr_market_status",
    "fetch_snr_market_data",
    "get_symbol_spec",
    "log_snr_audit",
    "log_snr_error",
    "prepare_snr_frame",
    "detect_swings",
    "detect_support_candidates",
    "detect_resistance_candidates",
    "cluster_zones",
    "validate_single_zone",
    "validate_zones",
    "analyze_zone_touches",
    "detect_zone_breakout",
    "detect_zone_retest",
    "detect_zone_rejection",
    "analyze_momentum",
    "evaluate_mtf_confluence",
    "assign_snr_grade",
    "score_zone",
    "classify_zone_strength_category",
    "compute_zone_distance",
    "rank_zones",
    "evaluate_single_tf_snr_bias",
    "evaluate_snr_market_bias",
    "calculate_snr_risk_reward",
    "round_to_tick",
    "validate_snr_entry",
    "build_snr_final_decision",
    "evaluate_snr_radar",
    "SnrAlertTracker",
    "format_snr_summary",
    "format_snr_detail",
    "format_snr_presentation",
    "run_snr_scan",
    "evaluate_snr_decision",
    "scan_snr",
    "scan_snr_detail",
    "SnrCycleEngine",
    "format_snr_monitor_section",
    "mode_pantau_snr",
]

