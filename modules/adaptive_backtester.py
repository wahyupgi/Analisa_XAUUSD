# -*- coding: utf-8 -*-
"""
modules/adaptive_backtester.py
No-Lookahead Event-Driven Backtester untuk XAUUSD SMC Adaptive Auto Trading System (Section 22):
- Mensimulasikan bar-by-bar pada candle M5 yang SUDAH CLOSE (tanpa melihat bar masa depan).
- Hanya memotong data H4, H1, M30, M15, M5 yang memiliki timestamp close <= waktu evaluasi bar saat ini
  sehingga mustahil terjadi look-ahead bias atau repainting!
- Membandingkan secara objektif:
  1. EARLY_PENDING vs ADAPTIVE_ENTRY vs CONFIRMED_ENTRY
  2. Zona H4/H1 vs Zona Lower Timeframe (M30/M15/M5)
  3. Score Threshold 70 vs 80 vs 90
  4. Dynamic Lot vs Fixed Lot (0.01)
  5. Missed entry rate, Invalid entry rate, Max Drawdown, Win Rate, Profit Factor, dan Expectancy.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from modules.adaptive_entry_engine import evaluate_entry_paths_for_zone
from modules.config import get_auto_trading_config
from modules.dynamic_lot_manager import (
    has_htf_supply_demand_origin,
    select_final_lot,
    select_target_lot,
    validate_strong_zone,
)
from modules.mt5_conn import tambah_indikator
from modules.risk_manager import validate_sl_tp_and_rr
from modules.zone_scanner import (
    TF_MINUTES,
    Zone,
    detect_fvgs,
    detect_liquidity_sweeps,
    detect_market_structure,
    detect_order_blocks,
    evaluate_htf_context,
    score_zone,
)


def _slice_closed_frames_up_to(
    full_frames: Dict[str, pd.DataFrame],
    eval_ts: pd.Timestamp,
) -> Dict[str, pd.DataFrame]:
    """
    Potong seluruh timeframe sehingga HANYA memuat candle yang sudah benar-benar CLOSED
    pada atau sebelum `eval_ts` (candle open_time + tf_duration <= eval_ts).
    Menjamin 0% look-ahead bias!
    """
    sliced: Dict[str, pd.DataFrame] = {}
    for tf, df in (full_frames or {}).items():
        if df is None or len(df) == 0:
            continue
        dur_min = TF_MINUTES.get(tf, 5)
        cutoff_open = eval_ts - pd.Timedelta(minutes=dur_min)
        sub = df.loc[df.index <= cutoff_open]
        if len(sub) > 0:
            sliced[tf] = sub
    return sliced


def _extract_point_in_time_zones(
    sliced_frames: Dict[str, pd.DataFrame],
    current_price: float,
    tfs: Tuple[str, ...] = ("H4", "H1", "M30", "M15", "M5"),
) -> Tuple[List[Zone], Dict[str, Any]]:
    """Deteksi zona aktif point-in-time hanya dari histori masa lalu yang sudah close."""
    ctx = evaluate_htf_context(sliced_frames)
    zones: List[Zone] = []
    for tf in tfs:
        df = sliced_frames.get(tf)
        if df is None or len(df) < 30:
            continue
        d_ind = df if "ATR14" in df.columns else tambah_indikator(df.copy())
        events, _, swings = detect_market_structure(d_ind)
        if not events:
            continue
        sweeps = detect_liquidity_sweeps(d_ind, swings, lookback=None)
        fvgs = detect_fvgs(d_ind)
        ok_z, _ = detect_order_blocks(d_ind, events, fvgs, sweeps, tf, max_events=15)
        for z in ok_z:
            h4_b = str(ctx.get("H4") or "NETRAL").upper()
            h1_b = str(ctx.get("H1") or "NETRAL").upper()
            m30_b = str(ctx.get("m30_status") or "CONFIRMED").upper()
            want = "BULLISH" if z.zone_type == "DEMAND" else "BEARISH"
            align_map = {
                "H4": "ALIGNED" if h4_b == want else ("NEUTRAL" if h4_b == "NETRAL" else "OPPOSING"),
                "H1": "ALIGNED" if want in h1_b else ("NEUTRAL" if "NETRAL" in h1_b else "OPPOSING"),
                "M30": "ALIGNED" if ("CONFIRMED" in m30_b or "ALIGNED" in m30_b) else "NEUTRAL",
            }
            score_zone(z, htf_alignment=align_map, current_price=current_price)
            if z.is_valid and z.freshness not in ("INVALIDATED", "MITIGATED"):
                zones.append(z)
    zones.sort(key=lambda z: (-float(z.score), abs(current_price - z.midpoint)))
    return zones, ctx


def simulate_strategy_variant(
    full_frames: Dict[str, pd.DataFrame],
    mode: str = "ADAPTIVE_ENTRY",
    min_score_threshold: float = 70.0,
    lot_scheme: str = "DYNAMIC",         # DYNAMIC | FIXED_001
    tf_filter: str = "ALL",              # ALL | HTF_ONLY | LTF_ONLY
    initial_equity: float = 10000.0,
    spread: float = 0.25,
    step_bars: int = 2,
) -> Dict[str, Any]:
    """
    Jalankan simulasi satu varian strategi bar-by-bar tanpa look-ahead bias pada M5.
    """
    conf = get_auto_trading_config()
    conf["MIN_ZONE_SCORE"] = float(min_score_threshold)
    if mode == "EARLY_PENDING":
        conf["EARLY_PENDING_ENABLED"] = True
        conf["ADAPTIVE_ENTRY_ENABLED"] = False
        conf["CONFIRMED_ENTRY_ENABLED"] = False
        pref_mode = "EARLY_PENDING"
    elif mode == "CONFIRMED_ENTRY":
        conf["EARLY_PENDING_ENABLED"] = False
        conf["ADAPTIVE_ENTRY_ENABLED"] = False
        conf["CONFIRMED_ENTRY_ENABLED"] = True
        pref_mode = "CONFIRMED_ENTRY"
    else:
        conf["EARLY_PENDING_ENABLED"] = False
        conf["ADAPTIVE_ENTRY_ENABLED"] = True
        conf["CONFIRMED_ENTRY_ENABLED"] = False
        pref_mode = "ADAPTIVE"

    df_m5 = full_frames.get("M5")
    if df_m5 is None or len(df_m5) < 80:
        return {"error": "Data M5 kurang dari 80 candle"}

    # Pre-compute indikator penuh per TF hanya untuk kolom EMA/ATR kausal (ewm adjust=False murni kausal tanpa look-ahead!)
    causal_frames = {
        tf: ( tambah_indikator(df.copy()) if (df is not None and len(df) > 0 and "ATR14" not in df.columns) else df )
        for tf, df in full_frames.items()
        if df is not None and len(df) > 0
    }

    equity = float(initial_equity)
    peak_equity = equity
    max_dd = 0.0

    active_pos: Optional[Dict[str, Any]] = None
    pending_ord: Optional[Dict[str, Any]] = None
    used_zone_ids: set = set()
    missed_zones: set = set()
    invalid_blocked_count = 0
    closed_trades: List[Dict[str, Any]] = []

    m5_idx = df_m5.index
    H5 = df_m5["High"].values
    L5 = df_m5["Low"].values
    C5 = df_m5["Close"].values

    start_i = max(65, len(df_m5) // 4)
    cached_zones: List[Zone] = []
    cached_ctx: Dict[str, Any] = {}
    last_zone_scan_i = -999

    for i in range(start_i, len(df_m5)):
        bar_open_ts = m5_idx[i]
        eval_ts = bar_open_ts + pd.Timedelta(minutes=5)
        bar_h, bar_l, bar_c = float(H5[i]), float(L5[i]), float(C5[i])
        bid = round(bar_c - spread * 0.5, 2)
        ask = round(bar_c + spread * 0.5, 2)

        # 1. Evaluasi fill pending order pada bar berjalan (konservatif: cek SL pada bar yang sama jika tersentuh)
        if pending_ord is not None and active_pos is None:
            p_dir = pending_ord["direction"]
            p_ent = pending_ord["entry"]
            p_inval = pending_ord["invalidation"]
            if (p_dir == "BUY" and bar_c < p_inval) or (p_dir == "SELL" and bar_c > p_inval):
                pending_ord = None
            elif (p_dir == "BUY" and bar_l <= p_ent) or (p_dir == "SELL" and bar_h >= p_ent):
                active_pos = dict(pending_ord)
                pending_ord = None

        # 2. Evaluasi SL / TP posisi aktif pada bar berjalan (konservatif: jika High & Low keduanya kena di 1 bar -> anggap SL kena duluan!)
        if active_pos is not None:
            d_pos = active_pos["direction"]
            sl_p = active_pos["sl"]
            tp_p = active_pos["tp"]
            ent_p = active_pos["entry"]
            lot_p = active_pos["lot"]
            hit_sl = (bar_l <= sl_p) if d_pos == "BUY" else (bar_h >= sl_p)
            hit_tp = (bar_h >= tp_p) if d_pos == "BUY" else (bar_l <= tp_p)

            if hit_sl or hit_tp:
                exit_px = sl_p if hit_sl else tp_p
                pts = (exit_px - ent_p) if d_pos == "BUY" else (ent_p - exit_px)
                # 1 lot XAUUSD = 100 oz -> P/L USD = pts * 100 * lot - komisi $7/lot
                net_pnl = round(pts * 100.0 * lot_p - (7.0 * lot_p), 2)
                equity += net_pnl
                if equity > peak_equity:
                    peak_equity = equity
                max_dd = max(max_dd, peak_equity - equity)
                closed_trades.append({
                    "zone_id": active_pos["zone_id"],
                    "timeframe": active_pos["timeframe"],
                    "score": active_pos["score"],
                    "entry_mode": active_pos["entry_mode"],
                    "lot": lot_p,
                    "rr": active_pos["rr"],
                    "pnl": net_pnl,
                    "outcome": "SL" if hit_sl else "TP",
                })
                active_pos = None

        if active_pos is not None or (i - start_i) % max(1, step_bars) != 0:
            continue

        # 3. Refresh struktur zona point-in-time (tanpa look-ahead!)
        sliced = _slice_closed_frames_up_to(causal_frames, eval_ts)
        if (i - last_zone_scan_i) >= 6 or not cached_zones:
            tfs_scan = (
                ("H4", "H1") if tf_filter == "HTF_ONLY"
                else (("M30", "M15", "M5") if tf_filter == "LTF_ONLY" else ("H4", "H1", "M30", "M15", "M5"))
            )
            cached_zones, cached_ctx = _extract_point_in_time_zones(sliced, bar_c, tfs=tfs_scan)
            last_zone_scan_i = i

        h4_b = str(cached_ctx.get("H4") or "NETRAL").upper()
        h1_b = str(cached_ctx.get("H1") or "NETRAL").upper()
        m30_s = str(cached_ctx.get("m30_status") or "CONFIRMED").upper()

        for z in cached_zones[:4]:
            if z.zone_id in used_zone_ids:
                continue
            if z.score < min_score_threshold:
                invalid_blocked_count += 1
                continue
            has_htf, _ = has_htf_supply_demand_origin(z)
            if tf_filter == "HTF_ONLY" and not has_htf:
                continue
            if tf_filter == "LTF_ONLY" and has_htf:
                continue

            # Cek arah selaras H4 (atau jika H4 netral di sampel pendek, gunakan H1 searah)
            want_b = "BULLISH" if z.zone_type == "DEMAND" else "BEARISH"
            opp_b = "BEARISH" if z.zone_type == "DEMAND" else "BULLISH"
            if h4_b == opp_b or opp_b in h1_b:
                invalid_blocked_count += 1
                continue
            if mode == "EARLY_PENDING" and not (h4_b == want_b and want_b in h1_b):
                continue
            if h4_b != want_b and want_b not in h1_b:
                continue

            path = evaluate_entry_paths_for_zone(
                z,
                frames=sliced,
                current_price=bar_c,
                bid=bid,
                ask=ask,
                spread=spread,
                htf_bias=want_b if h4_b == "NETRAL" else h4_b,
                h1_bias=h1_b,
                m30_status=m30_s,
                preferred_mode=pref_mode,
                cfg=conf,
            )
            if path["reason_code"] == "PRICE_LEFT_ENTRY_RANGE":
                missed_zones.add(z.zone_id)
                continue
            if not path["trigger_passed"]:
                continue

            e_mode = path["entry_mode"]
            if mode == "EARLY_PENDING" and e_mode != "EARLY_PENDING":
                continue
            if mode in ("ADAPTIVE_ENTRY", "CONFIRMED_ENTRY") and e_mode == "EARLY_PENDING":
                continue

            lv = path["levels"]
            ent_px = float(lv.get("entry_price") or path["touch"]["exec_price"])
            sl_tp = validate_sl_tp_and_rr(
                direction=path["direction"],
                entry_price=ent_px,
                sl_price=float(lv.get("sl") or 0.0),
                tp1_price=float(lv.get("tp1") or 0.0),
                structural_invalidation=float(z.invalidation_price or 0.0),
                bid=bid,
                ask=ask,
                is_market_order=(path["order_kind"] == "MARKET"),
                cfg=conf,
            )
            if not sl_tp.get("valid"):
                invalid_blocked_count += 1
                continue

            if lot_scheme == "FIXED_001":
                final_lot = 0.01
            else:
                strong_v = validate_strong_zone(
                    z, entry_mode=e_mode, htf_aligned=True, has_htf_conflict=False,
                    actual_rr=sl_tp["actual_rr"], trigger_valid=True, cfg=conf,
                )
                t_lot = select_target_lot(z, e_mode, strong_zone_validated=strong_v["passed"], cfg=conf)
                risk_pct = 0.25 if e_mode == "EARLY_PENDING" else 0.50
                budget = equity * (risk_pct / 100.0)
                sl_dist = abs(sl_tp["entry"] - sl_tp["sl"])
                final_lot = select_final_lot(t_lot, budget, lambda l, d=sl_dist: (d * 100.0 + 7.0) * l)
                if final_lot <= 0:
                    invalid_blocked_count += 1
                    continue

            order_rec = {
                "zone_id": z.zone_id,
                "timeframe": z.timeframe,
                "score": z.score,
                "entry_mode": e_mode,
                "direction": path["direction"],
                "entry": sl_tp["entry"],
                "sl": sl_tp["sl"],
                "tp": sl_tp["tp"],
                "rr": sl_tp["actual_rr"],
                "lot": final_lot,
                "invalidation": float(z.invalidation_price or sl_tp["sl"]),
            }
            used_zone_ids.add(z.zone_id)
            if path["order_kind"] == "LIMIT":
                pending_ord = order_rec
            else:
                active_pos = order_rec
            break

    pnls = [t["pnl"] for t in closed_trades]
    n_tr = len(pnls)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    wr = (len(wins) / n_tr * 100.0) if n_tr > 0 else 0.0
    exp = (sum(pnls) / n_tr) if n_tr > 0 else 0.0
    pf = (sum(wins) / abs(sum(losses))) if losses else (float("inf") if wins else 0.0)
    tot_opps = n_tr + len(missed_zones)
    missed_rate = (len(missed_zones) / tot_opps * 100.0) if tot_opps > 0 else 0.0

    return {
        "mode": mode,
        "min_score": min_score_threshold,
        "lot_scheme": lot_scheme,
        "tf_filter": tf_filter,
        "trades": n_tr,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": round(wr, 2),
        "net_pnl_usd": round(sum(pnls), 2),
        "expectancy_usd": round(exp, 2),
        "profit_factor": round(pf, 2) if math.isfinite(pf) else "INF",
        "max_drawdown_usd": round(max_dd, 2),
        "missed_entries": len(missed_zones),
        "missed_entry_rate_pct": round(missed_rate, 2),
        "blocked_invalid_setups": invalid_blocked_count,
    }


def run_comprehensive_backtest_comparison(
    full_frames: Dict[str, pd.DataFrame],
    initial_equity: float = 10000.0,
    step_bars: int = 3,
) -> Dict[str, Any]:
    """
    Jalankan matriks perbandingan lengkap sesuai Section 22:
    - Early Pending vs Adaptive Entry vs Confirmed Entry
    - Zona H4/H1 vs Lower Timeframe
    - Score threshold 70 / 80 / 90
    - Dynamic Lot vs Fixed Lot (0.01)
    """
    return {
        "entry_modes": {
            "EARLY_PENDING": simulate_strategy_variant(full_frames, mode="EARLY_PENDING", initial_equity=initial_equity, step_bars=step_bars),
            "ADAPTIVE_ENTRY": simulate_strategy_variant(full_frames, mode="ADAPTIVE_ENTRY", initial_equity=initial_equity, step_bars=step_bars),
            "CONFIRMED_ENTRY": simulate_strategy_variant(full_frames, mode="CONFIRMED_ENTRY", initial_equity=initial_equity, step_bars=step_bars),
        },
        "timeframes": {
            "HTF_H4_H1": simulate_strategy_variant(full_frames, mode="ADAPTIVE_ENTRY", tf_filter="HTF_ONLY", initial_equity=initial_equity, step_bars=step_bars),
            "LTF_M30_M15_M5": simulate_strategy_variant(full_frames, mode="ADAPTIVE_ENTRY", tf_filter="LTF_ONLY", initial_equity=initial_equity, step_bars=step_bars),
        },
        "score_thresholds": {
            "SCORE_70": simulate_strategy_variant(full_frames, mode="ADAPTIVE_ENTRY", min_score_threshold=70.0, initial_equity=initial_equity, step_bars=step_bars),
            "SCORE_80": simulate_strategy_variant(full_frames, mode="ADAPTIVE_ENTRY", min_score_threshold=80.0, initial_equity=initial_equity, step_bars=step_bars),
            "SCORE_90": simulate_strategy_variant(full_frames, mode="ADAPTIVE_ENTRY", min_score_threshold=90.0, initial_equity=initial_equity, step_bars=step_bars),
        },
        "lot_sizing": {
            "DYNAMIC_LOT": simulate_strategy_variant(full_frames, mode="ADAPTIVE_ENTRY", lot_scheme="DYNAMIC", initial_equity=initial_equity, step_bars=step_bars),
            "FIXED_001": simulate_strategy_variant(full_frames, mode="ADAPTIVE_ENTRY", lot_scheme="FIXED_001", initial_equity=initial_equity, step_bars=step_bars),
        },
    }
