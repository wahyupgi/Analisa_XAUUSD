# -*- coding: utf-8 -*-
"""
modules/risk_manager.py
Manajemen risiko terpusat untuk akun Real IDC / Cent / USD:
- Deteksi identitas akun, mata uang aktual, equity, free margin, leverage, dan izin algo trading.
- Perhitungan eksposur total (posisi aktif + pending order, baik milik scanner maupun manual user).
- Perhitungan batas rugi harian (realized + floating + komisi + swap) dengan kebijakan timezone eksplisit.
- Perhitungan risk budget efektif (minimum dari seluruh batas risiko yang berlaku).
- Validasi ketat arah & level SL/TP struktural serta perhitungan ulang RR aktual terhadap harga eksekusi.
"""
from __future__ import annotations

import math
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from modules.config import SYMBOL, get_auto_trading_config
from modules.execution_state import (
    ExecutionStateManager,
    REASON_ACCOUNT_CHANGED,
    REASON_ALGO_TRADING_DISABLED,
    REASON_BROKER_CONSTRAINTS,
    REASON_DAILY_LOSS_LIMIT,
    REASON_DEMO_ONLY_RESTRICTION,
    REASON_INSUFFICIENT_MARGIN,
    REASON_INVALID_STOP_LOSS,
    REASON_INVALID_TAKE_PROFIT,
    REASON_MARKET_CLOSED,
    REASON_MAX_ACTIVE_PENDING_ORDERS,
    REASON_MAX_ACTIVE_POSITIONS,
    REASON_MAX_EXPOSURE_EXCEEDED,
    REASON_MAX_TRADES_PER_DAY,
    REASON_MT5_DISCONNECTED,
    REASON_OK,
    REASON_REAL_ACCOUNT_UNCONFIRMED,
    REASON_RR_BELOW_MINIMUM,
    REASON_UNPROTECTED_POSITION_NO_SL,
    trading_day_start_epoch,
)
from modules.utils import get_symbol_unit_spec, mt5_points_to_price_distance


def resolve_account_mode_label(trade_mode: Any, mt5_api: Any = None) -> str:
    """Terjemahkan trade_mode MT5 menjadi REAL / DEMO / CONTEST."""
    real_code = getattr(mt5_api, "ACCOUNT_TRADE_MODE_REAL", 2) if mt5_api is not None else 2
    contest_code = getattr(mt5_api, "ACCOUNT_TRADE_MODE_CONTEST", 1) if mt5_api is not None else 1
    try:
        tm = int(trade_mode)
    except (TypeError, ValueError):
        return "UNKNOWN"
    if tm == real_code:
        return "REAL"
    if tm == contest_code:
        return "CONTEST"
    return "DEMO"


def inspect_mt5_account_and_permissions(
    mt5_api: Any,
    symbol: str = SYMBOL,
    state_mgr: Optional[ExecutionStateManager] = None,
    require_exec_permission: bool = False,
    trading_mode: str = "SCANNER_ONLY",
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Inspeksi menyeluruh koneksi terminal MT5, identitas akun (Real IDC/Demo), izin algo trading,
    dan ketersediaan simbol broker tanpa melakukan perubahan akun atau hardcode kredensial.
    """
    conf = cfg or get_auto_trading_config()
    eff_sym = str(symbol or conf.get("SYMBOL") or SYMBOL)
    out: Dict[str, Any] = {
        "connected": False,
        "market_open": False,
        "algo_allowed": False,
        "account_valid": False,
        "authorized": False,
        "reason_code": REASON_OK,
        "reason_text": "",
        "symbol": eff_sym,
        "login": None,
        "server": "",
        "account_mode": "UNKNOWN",
        "currency": "",
        "balance": 0.0,
        "equity": 0.0,
        "margin": 0.0,
        "free_margin": 0.0,
        "margin_level": 0.0,
        "leverage": 0,
        "floating_profit": 0.0,
        "terminal_info": None,
        "account_info": None,
        "symbol_info": None,
        "tick": None,
    }

    if mt5_api is None:
        out["reason_code"] = REASON_MT5_DISCONNECTED
        out["reason_text"] = "Modul MetaTrader5 tidak tersedia"
        return out

    term = mt5_api.terminal_info()
    if term is None or getattr(term, "connected", True) is False:
        out["reason_code"] = REASON_MT5_DISCONNECTED
        out["reason_text"] = "Terminal MT5 tidak terhubung ke server broker"
        return out

    acc = mt5_api.account_info()
    if acc is None:
        out["reason_code"] = REASON_MT5_DISCONNECTED
        out["reason_text"] = "Informasi akun MT5 tidak tersedia"
        return out

    sym_info = mt5_api.symbol_info(eff_sym)
    if sym_info is None:
        # Coba aktifkan symbol_select pada nama simbol terkonfigurasi
        try:
            mt5_api.symbol_select(eff_sym, True)
            sym_info = mt5_api.symbol_info(eff_sym)
        except Exception:
            sym_info = None

    tick = mt5_api.symbol_info_tick(eff_sym) if sym_info is not None else None

    login = getattr(acc, "login", None)
    server = str(getattr(acc, "server", "") or "")
    currency = str(getattr(acc, "currency", "") or "").strip().upper()
    acc_mode = resolve_account_mode_label(getattr(acc, "trade_mode", 0), mt5_api=mt5_api)
    balance = float(getattr(acc, "balance", 0.0) or 0.0)
    equity = float(getattr(acc, "equity", 0.0) or balance)
    margin = float(getattr(acc, "margin", 0.0) or 0.0)
    free_margin = float(getattr(acc, "margin_free", 0.0) or 0.0)
    margin_level = float(getattr(acc, "margin_level", 0.0) or 0.0)
    leverage = int(getattr(acc, "leverage", 0) or 0)
    floating_profit = float(getattr(acc, "profit", 0.0) or 0.0)

    term_trade_allowed = bool(getattr(term, "trade_allowed", False))
    term_api_disabled = bool(getattr(term, "tradeapi_disabled", False))
    acc_trade_allowed = bool(getattr(acc, "trade_allowed", False))
    acc_expert_allowed = bool(getattr(acc, "trade_expert", True))
    algo_ok = term_trade_allowed and (not term_api_disabled) and acc_trade_allowed and acc_expert_allowed

    sym_disabled_code = getattr(mt5_api, "SYMBOL_TRADE_MODE_DISABLED", 0)
    sym_closeonly_code = getattr(mt5_api, "SYMBOL_TRADE_MODE_CLOSEONLY", 3)
    sym_trade_mode = getattr(sym_info, "trade_mode", sym_disabled_code) if sym_info is not None else sym_disabled_code
    market_open = (sym_info is not None) and (sym_trade_mode not in (sym_disabled_code, sym_closeonly_code))

    out.update({
        "connected": True,
        "market_open": market_open,
        "algo_allowed": algo_ok,
        "account_valid": bool(login is not None and currency and equity > 0),
        "login": login,
        "server": server,
        "account_mode": acc_mode,
        "currency": currency,
        "balance": balance,
        "equity": equity,
        "margin": margin,
        "free_margin": free_margin,
        "margin_level": margin_level,
        "leverage": leverage,
        "floating_profit": floating_profit,
        "terminal_info": term,
        "account_info": acc,
        "symbol_info": sym_info,
        "tick": tick,
    })

    if not out["account_valid"]:
        out["reason_code"] = REASON_MT5_DISCONNECTED
        out["reason_text"] = "Identitas akun, mata uang, atau equity akun tidak valid"
        return out

    # Verifikasi konsistensi akun MT5 terhadap akun yang sedang diotorisasi
    if state_mgr is not None:
        same_acc, acc_code = state_mgr.verify_bound_account(login, server)
        if not same_acc:
            out["reason_code"] = acc_code
            out["reason_text"] = state_mgr.get_mode_info().get("account_mismatch_reason") or "Akun MT5 berubah dari akun terotorisasi"
            return out

    t_mode_u = str(trading_mode or "SCANNER_ONLY").upper()
    is_live_exec_mode = require_exec_permission or t_mode_u in ("FULL_AUTO", "SEMI_AUTO")

    if is_live_exec_mode:
        if conf.get("DEMO_ONLY", False) and acc_mode == "REAL":
            out["reason_code"] = REASON_DEMO_ONLY_RESTRICTION
            out["reason_text"] = "Konfigurasi DEMO_ONLY=true memblokir transaksi pada akun REAL"
            return out
        if acc_mode == "REAL":
            if not conf.get("ALLOW_REAL_ACCOUNT", True):
                out["reason_code"] = REASON_REAL_ACCOUNT_UNCONFIRMED
                out["reason_text"] = "ALLOW_REAL_ACCOUNT=false memblokir transaksi akun REAL"
                return out
            if conf.get("REQUIRE_REAL_ACCOUNT_CONFIRMATION", True) and t_mode_u == "FULL_AUTO":
                auth_info = state_mgr.get_mode_info().get("authorized_account", {}) if state_mgr else {}
                is_confirmed = (
                    bool(auth_info.get("real_confirmed"))
                    and auth_info.get("login") is not None
                    and int(auth_info.get("login")) == int(login)
                )
                if not is_confirmed:
                    out["reason_code"] = REASON_REAL_ACCOUNT_UNCONFIRMED
                    out["reason_text"] = "Akun REAL belum dikonfirmasi eksplisit melalui command 'auto real'"
                    return out

        if not market_open:
            out["reason_code"] = REASON_MARKET_CLOSED
            out["reason_text"] = f"Market simbol {eff_sym} sedang tutup atau close-only (trade_mode={sym_trade_mode})"
            return out

        if not algo_ok:
            out["reason_code"] = REASON_ALGO_TRADING_DISABLED
            out["reason_text"] = "Algorithmic Trading / External Python API dinonaktifkan pada terminal atau akun MT5"
            return out

    out["authorized"] = True
    return out


def _estimate_single_item_risk(
    mt5_api: Any,
    order_type: int,
    symbol: str,
    volume: float,
    open_price: float,
    sl_price: float,
    fallback_sl_distance: float = 20.0,
) -> Tuple[float, bool]:
    """
    Hitung risiko nominal ke SL dalam mata uang akun aktual menggunakan mt5.order_calc_profit().
    Jika posisi terkunci BEP+ (profit pada level SL >= 0), risiko terbuka = 0.0.
    Jika posisi manual tanpa SL (sl_price <= 0), estimasikan secara konservatif menggunakan fallback_sl_distance
    dan tandai unbounded=True agar risiko akun keseluruhan tidak diabaikan.
    """
    buy_types = {
        getattr(mt5_api, "ORDER_TYPE_BUY", 0),
        getattr(mt5_api, "ORDER_TYPE_BUY_LIMIT", 2),
        getattr(mt5_api, "ORDER_TYPE_BUY_STOP", 4),
        getattr(mt5_api, "ORDER_TYPE_BUY_STOP_LIMIT", 6),
    }
    is_buy = int(order_type) in buy_types
    calc_type = getattr(mt5_api, "ORDER_TYPE_BUY", 0) if is_buy else getattr(mt5_api, "ORDER_TYPE_SELL", 1)

    unbounded = False
    eff_sl = float(sl_price or 0.0)
    if eff_sl <= 0.0:
        unbounded = True
        eff_sl = max(0.01, open_price - fallback_sl_distance) if is_buy else (open_price + fallback_sl_distance)

    try:
        pnl_at_sl = mt5_api.order_calc_profit(calc_type, symbol, float(volume), float(open_price), float(eff_sl))
        if pnl_at_sl is not None and math.isfinite(float(pnl_at_sl)):
            # Jika SL sudah di zona profit (BEP+), pnl_at_sl >= 0 -> risiko kerugian = 0.0
            if float(pnl_at_sl) >= 0.0 and not unbounded:
                return 0.0, False
            return abs(min(0.0, float(pnl_at_sl))) if not unbounded else abs(float(pnl_at_sl)), unbounded
    except Exception:
        pass

    # Fallback proporsional melalui sym_info tick_value/tick_size bila order_calc_profit gagal sementara
    sym_info = mt5_api.symbol_info(symbol) if hasattr(mt5_api, "symbol_info") else None
    if sym_info is not None:
        t_size = float(getattr(sym_info, "trade_tick_size", 0.0) or 0.0)
        t_val = float(getattr(sym_info, "trade_tick_value", 0.0) or 0.0)
        if t_size > 0 and t_val > 0:
            pts_loss = (open_price - eff_sl) if is_buy else (eff_sl - open_price)
            if pts_loss <= 0 and not unbounded:
                return 0.0, False
            return abs(pts_loss / t_size) * t_val * float(volume), unbounded

    pts_loss = (open_price - eff_sl) if is_buy else (eff_sl - open_price)
    if pts_loss <= 0 and not unbounded:
        return 0.0, False
    return abs(pts_loss) * 100.0 * float(volume), unbounded


def evaluate_portfolio_risk_and_daily_limits(
    mt5_api: Any,
    state_mgr: Optional[ExecutionStateManager] = None,
    symbol: str = SYMBOL,
    cfg: Optional[Dict[str, Any]] = None,
    now_utc: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Evaluasi lengkap eksposur akun (scanner + manual), posisi aktif, pending order,
    realized P/L harian, floating P/L, dan trade counter harian.
    """
    conf = cfg or get_auto_trading_config()
    magic = int(conf.get("SCANNER_MAGIC_NUMBER", 20260991))
    sl_fallback = float(conf.get("SL_JARAK_MAKS", 20.0))
    reset_tz = str(conf.get("DAILY_RESET_TIMEZONE", "WIB"))

    acc = mt5_api.account_info() if mt5_api is not None else None
    balance = float(getattr(acc, "balance", 0.0) or 0.0) if acc else 0.0
    equity = float(getattr(acc, "equity", 0.0) or balance) if acc else 0.0
    currency = str(getattr(acc, "currency", "") or "").strip().upper() if acc else ""

    positions = list(mt5_api.positions_get() or ()) if mt5_api is not None else []
    orders = list(mt5_api.orders_get() or ()) if mt5_api is not None else []

    scanner_positions: List[Any] = []
    manual_positions: List[Any] = []
    scanner_orders: List[Any] = []
    manual_orders: List[Any] = []

    total_open_risk = 0.0
    scanner_open_risk = 0.0
    manual_open_risk = 0.0
    unbounded_manual_count = 0
    scanner_without_sl_tickets: List[int] = []
    total_floating_net = 0.0

    for p in positions:
        p_magic = int(getattr(p, "magic", 0) or 0)
        p_cmt = str(getattr(p, "comment", "") or "")
        is_scanner = (p_magic == magic) or p_cmt.startswith("SMC_")
        p_sym = str(getattr(p, "symbol", symbol) or symbol)
        p_vol = float(getattr(p, "volume", 0.0) or 0.0)
        p_open = float(getattr(p, "price_open", 0.0) or 0.0)
        p_sl = float(getattr(p, "sl", 0.0) or 0.0)
        p_type = int(getattr(p, "type", 0) or 0)
        p_tkt = int(getattr(p, "ticket", 0) or 0)
        p_prof = float(getattr(p, "profit", 0.0) or 0.0)
        p_swap = float(getattr(p, "swap", 0.0) or 0.0)
        p_comm = float(getattr(p, "commission", 0.0) or 0.0)
        total_floating_net += (p_prof + p_swap + p_comm)

        r_nom, unb = _estimate_single_item_risk(
            mt5_api, p_type, p_sym, p_vol, p_open, p_sl, fallback_sl_distance=sl_fallback
        )
        total_open_risk += r_nom
        if is_scanner:
            scanner_positions.append(p)
            scanner_open_risk += r_nom
            if p_sl <= 0.0 and p_tkt > 0:
                scanner_without_sl_tickets.append(p_tkt)
        else:
            manual_positions.append(p)
            manual_open_risk += r_nom
            if unb:
                unbounded_manual_count += 1

    for o in orders:
        o_magic = int(getattr(o, "magic", 0) or 0)
        o_cmt = str(getattr(o, "comment", "") or "")
        is_scanner = (o_magic == magic) or o_cmt.startswith("SMC_")
        o_sym = str(getattr(o, "symbol", symbol) or symbol)
        o_vol = float(getattr(o, "volume_current", getattr(o, "volume_initial", 0.0)) or 0.0)
        o_open = float(getattr(o, "price_open", 0.0) or 0.0)
        o_sl = float(getattr(o, "sl", 0.0) or 0.0)
        o_type = int(getattr(o, "type", 2) or 2)

        r_nom, _ = _estimate_single_item_risk(
            mt5_api, o_type, o_sym, o_vol, o_open, o_sl, fallback_sl_distance=sl_fallback
        )
        total_open_risk += r_nom
        if is_scanner:
            scanner_orders.append(o)
            scanner_open_risk += r_nom
        else:
            manual_orders.append(o)
            manual_open_risk += r_nom

    # Hitung realized P/L & trade count hari ini dari histori MT5 + state persisten
    day_start_epoch = trading_day_start_epoch(reset_tz=reset_tz)
    mt5_realized_today = 0.0
    mt5_deal_ids: List[Any] = []
    mt5_scanner_entries_today = 0

    if mt5_api is not None and hasattr(mt5_api, "history_deals_get"):
        try:
            dt_from = datetime.fromtimestamp(day_start_epoch - 86400, tz=timezone.utc)
            dt_to = (now_utc or datetime.now(timezone.utc)) + timedelta(days=1)
            deals = mt5_api.history_deals_get(dt_from, dt_to) or ()
            deal_buy = getattr(mt5_api, "DEAL_TYPE_BUY", 0)
            deal_sell = getattr(mt5_api, "DEAL_TYPE_SELL", 1)
            entry_in = getattr(mt5_api, "DEAL_ENTRY_IN", 0)
            entry_out_set = {
                getattr(mt5_api, "DEAL_ENTRY_OUT", 1),
                getattr(mt5_api, "DEAL_ENTRY_INOUT", 2),
                getattr(mt5_api, "DEAL_ENTRY_OUT_BY", 3),
            }
            seen_in_pos = set()
            for d in deals:
                d_time = float(getattr(d, "time", 0) or 0)
                if d_time < day_start_epoch:
                    continue
                d_type = int(getattr(d, "type", -1))
                if d_type not in (deal_buy, deal_sell):
                    continue
                d_entry = int(getattr(d, "entry", -1))
                d_magic = int(getattr(d, "magic", 0) or 0)
                d_cmt = str(getattr(d, "comment", "") or "")
                if d_entry == entry_in and (d_magic == magic or d_cmt.startswith("SMC_")):
                    pid = int(getattr(d, "position_id", getattr(d, "ticket", 0)) or 0)
                    if pid not in seen_in_pos:
                        seen_in_pos.add(pid)
                        mt5_scanner_entries_today += 1
                if d_entry in entry_out_set or d_entry == -1:
                    net_d = (
                        float(getattr(d, "profit", 0.0) or 0.0)
                        + float(getattr(d, "commission", 0.0) or 0.0)
                        + float(getattr(d, "swap", 0.0) or 0.0)
                        + float(getattr(d, "fee", 0.0) or 0.0)
                    )
                    mt5_realized_today += net_d
                    mt5_deal_ids.append(getattr(d, "ticket", ""))
        except Exception:
            pass

    if state_mgr is not None:
        daily_state = state_mgr.sync_daily_metrics_from_mt5(
            mt5_realized_pnl=mt5_realized_today if mt5_deal_ids else None,
            mt5_floating_pnl=total_floating_net,
            mt5_deal_ids=mt5_deal_ids,
            mt5_trades_opened_today=mt5_scanner_entries_today,
            current_balance=balance,
            current_equity=equity,
        )
        eff_realized_pnl = float(daily_state.get("realized_pnl") or 0.0)
        trades_today = int(daily_state.get("trades_count") or 0)
        ref_base = float(daily_state.get("starting_equity") or daily_state.get("starting_balance") or equity or balance)
    else:
        eff_realized_pnl = mt5_realized_today
        trades_today = mt5_scanner_entries_today
        ref_base = max(equity, balance - mt5_realized_today)

    if ref_base <= 0:
        ref_base = equity if equity > 0 else balance

    # Daily loss mencakup realized loss + floating loss aktual (termasuk komisi & swap)
    floating_loss_component = min(0.0, total_floating_net)
    net_daily_pnl = eff_realized_pnl + floating_loss_component
    daily_loss_nominal = abs(min(0.0, net_daily_pnl))
    daily_loss_pct = (daily_loss_nominal / ref_base * 100.0) if ref_base > 0 else 0.0
    open_exposure_pct = (total_open_risk / equity * 100.0) if equity > 0 else 0.0

    max_daily_loss_pct = float(conf.get("MAX_DAILY_LOSS_PERCENT", 2.0))
    max_total_exp_pct = float(conf.get("MAX_TOTAL_EXPOSURE_PERCENT", 1.0))
    max_pos = int(conf.get("MAX_ACTIVE_POSITIONS", 1))
    max_ord = int(conf.get("MAX_ACTIVE_PENDING_ORDERS", 1))
    max_trades = int(conf.get("MAX_TRADES_PER_DAY", 3))

    return {
        "currency": currency,
        "balance": balance,
        "equity": equity,
        "reference_daily_base": ref_base,
        "scanner_positions": scanner_positions,
        "manual_positions": manual_positions,
        "scanner_orders": scanner_orders,
        "manual_orders": manual_orders,
        "scanner_positions_count": len(scanner_positions),
        "manual_positions_count": len(manual_positions),
        "scanner_orders_count": len(scanner_orders),
        "manual_orders_count": len(manual_orders),
        "scanner_without_sl_tickets": scanner_without_sl_tickets,
        "unbounded_manual_count": unbounded_manual_count,
        "total_open_risk_nominal": round(total_open_risk, 4),
        "scanner_open_risk_nominal": round(scanner_open_risk, 4),
        "manual_open_risk_nominal": round(manual_open_risk, 4),
        "open_exposure_pct": round(open_exposure_pct, 4),
        "realized_pnl_today": round(eff_realized_pnl, 4),
        "floating_pnl_net": round(total_floating_net, 4),
        "daily_loss_nominal": round(daily_loss_nominal, 4),
        "daily_loss_pct": round(daily_loss_pct, 4),
        "trades_today": trades_today,
        "daily_loss_limit_reached": daily_loss_pct >= max_daily_loss_pct - 1e-9,
        "exposure_limit_reached": open_exposure_pct >= max_total_exp_pct - 1e-9,
        "max_positions_reached": len(scanner_positions) >= max_pos,
        "max_pending_reached": len(scanner_orders) >= max_ord,
        "max_trades_reached": trades_today >= max_trades,
    }


def compute_effective_risk_budget(
    entry_mode: str,
    portfolio_risk: Dict[str, Any],
    cfg: Optional[Dict[str, Any]] = None,
    is_pending_order: bool = False,
) -> Dict[str, Any]:
    """
    Hitung risk budget efektif berdasarkan batas TERENDAH dari:
    1. Risk per entry mode (EARLY_PENDING 0.25%, ADAPTIVE 0.5%, CONFIRMED 0.5%)
    2. MAX_RISK_PER_TRADE_PERCENT (1.0%)
    3. Sisa total exposure akun (MAX_TOTAL_EXPOSURE_PERCENT - open_exposure_pct)
    4. Sisa kuota daily loss (MAX_DAILY_LOSS_PERCENT - daily_loss_pct)
    """
    conf = cfg or get_auto_trading_config()
    equity = float(portfolio_risk.get("equity") or 0.0)
    ref_base = float(portfolio_risk.get("reference_daily_base") or equity)

    # Blokir order baru apabila masih terdapat posisi terbuka tanpa Stop Loss (risiko unbounded)
    if (portfolio_risk.get("scanner_without_sl_tickets") or []) or int(portfolio_risk.get("unbounded_manual_count") or 0) > 0:
        return {
            "allowed": False,
            "reason_code": REASON_UNPROTECTED_POSITION_NO_SL,
            "reason_text": "Terdapat posisi aktif tanpa Stop Loss (risiko tidak terbatas); pasang SL terlebih dahulu sebelum order baru",
            "risk_budget_nominal": 0.0,
            "effective_risk_pct": 0.0,
            "mode_risk_pct": 0.0,
        }

    if portfolio_risk.get("daily_loss_limit_reached"):
        return {
            "allowed": False,
            "reason_code": REASON_DAILY_LOSS_LIMIT,
            "reason_text": (
                f"Daily loss limit tercapai ({portfolio_risk.get('daily_loss_pct', 0.0):.2f}% "
                f">= {conf.get('MAX_DAILY_LOSS_PERCENT', 2.0):.2f}%)"
            ),
            "risk_budget_nominal": 0.0,
            "effective_risk_pct": 0.0,
            "mode_risk_pct": 0.0,
        }

    if portfolio_risk.get("max_trades_reached"):
        return {
            "allowed": False,
            "reason_code": REASON_MAX_TRADES_PER_DAY,
            "reason_text": (
                f"Batas maksimal transaksi harian tercapai ({portfolio_risk.get('trades_today', 0)} "
                f"/ {conf.get('MAX_TRADES_PER_DAY', 3)})"
            ),
            "risk_budget_nominal": 0.0,
            "effective_risk_pct": 0.0,
            "mode_risk_pct": 0.0,
        }

    if portfolio_risk.get("max_positions_reached"):
        return {
            "allowed": False,
            "reason_code": REASON_MAX_ACTIVE_POSITIONS,
            "reason_text": (
                f"Posisi scanner aktif sudah mencapai batas ({portfolio_risk.get('scanner_positions_count', 0)} "
                f"/ {conf.get('MAX_ACTIVE_POSITIONS', 1)})"
            ),
            "risk_budget_nominal": 0.0,
            "effective_risk_pct": 0.0,
            "mode_risk_pct": 0.0,
        }

    if is_pending_order and portfolio_risk.get("max_pending_reached"):
        return {
            "allowed": False,
            "reason_code": REASON_MAX_ACTIVE_PENDING_ORDERS,
            "reason_text": (
                f"Pending order scanner aktif sudah mencapai batas ({portfolio_risk.get('scanner_orders_count', 0)} "
                f"/ {conf.get('MAX_ACTIVE_PENDING_ORDERS', 1)})"
            ),
            "risk_budget_nominal": 0.0,
            "effective_risk_pct": 0.0,
            "mode_risk_pct": 0.0,
        }

    if portfolio_risk.get("exposure_limit_reached"):
        return {
            "allowed": False,
            "reason_code": REASON_MAX_EXPOSURE_EXCEEDED,
            "reason_text": (
                f"Total eksposur akun ({portfolio_risk.get('open_exposure_pct', 0.0):.2f}%) "
                f"mencapai/melebihi batas {conf.get('MAX_TOTAL_EXPOSURE_PERCENT', 1.0):.2f}%"
            ),
            "risk_budget_nominal": 0.0,
            "effective_risk_pct": 0.0,
            "mode_risk_pct": 0.0,
        }

    if equity <= 0:
        return {
            "allowed": False,
            "reason_code": REASON_INSUFFICIENT_MARGIN,
            "reason_text": "Equity akun tidak valid (<= 0)",
            "risk_budget_nominal": 0.0,
            "effective_risk_pct": 0.0,
            "mode_risk_pct": 0.0,
        }

    mode_u = str(entry_mode or "ADAPTIVE_ENTRY").upper()
    if mode_u == "EARLY_PENDING":
        mode_risk_pct = float(conf.get("RISK_EARLY_PENDING_PERCENT", 0.25))
    elif mode_u in ("CONFIRMED", "CONFIRMED_ENTRY"):
        mode_risk_pct = float(conf.get("RISK_CONFIRMED_PERCENT", 0.50))
    else:
        mode_risk_pct = float(conf.get("RISK_ADAPTIVE_PERCENT", 0.50))

    max_trade_pct = float(conf.get("MAX_RISK_PER_TRADE_PERCENT", 1.0))
    max_exp_pct = float(conf.get("MAX_TOTAL_EXPOSURE_PERCENT", 1.0))
    max_daily_pct = float(conf.get("MAX_DAILY_LOSS_PERCENT", 2.0))

    rem_exp_nominal = max(0.0, (equity * max_exp_pct / 100.0) - float(portfolio_risk.get("total_open_risk_nominal") or 0.0))
    rem_daily_nominal = max(0.0, (ref_base * max_daily_pct / 100.0) - float(portfolio_risk.get("daily_loss_nominal") or 0.0))
    mode_nominal = equity * min(mode_risk_pct, max_trade_pct) / 100.0

    candidates = [
        ("ENTRY_MODE_LIMIT", mode_nominal),
        ("TOTAL_EXPOSURE_REMAINING", rem_exp_nominal),
        ("DAILY_LOSS_REMAINING", rem_daily_nominal),
    ]
    limiting_factor, budget_nominal = min(candidates, key=lambda x: x[1])
    eff_pct = (budget_nominal / equity * 100.0) if equity > 0 else 0.0

    if budget_nominal <= 0.0:
        code = REASON_DAILY_LOSS_LIMIT if limiting_factor == "DAILY_LOSS_REMAINING" else REASON_MAX_EXPOSURE_EXCEEDED
        return {
            "allowed": False,
            "reason_code": code,
            "reason_text": f"Risk budget tersisa habis ({limiting_factor})",
            "risk_budget_nominal": 0.0,
            "effective_risk_pct": 0.0,
            "mode_risk_pct": mode_risk_pct,
        }

    return {
        "allowed": True,
        "reason_code": REASON_OK,
        "reason_text": "Risk budget tersedia",
        "risk_budget_nominal": round(budget_nominal, 4),
        "effective_risk_pct": round(eff_pct, 4),
        "mode_risk_pct": mode_risk_pct,
        "limiting_factor": limiting_factor,
        "remaining_exposure_nominal": round(rem_exp_nominal, 4),
        "remaining_daily_loss_nominal": round(rem_daily_nominal, 4),
    }


def validate_sl_tp_and_rr(
    direction: str,
    entry_price: float,
    sl_price: float,
    tp1_price: float,
    tp2_price: float = 0.0,
    structural_invalidation: float = 0.0,
    sym_info: Any = None,
    bid: Optional[float] = None,
    ask: Optional[float] = None,
    is_market_order: bool = True,
    symbol: str = SYMBOL,
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Validasi ketat arah SL/TP, posisi SL di balik structural invalidation, stops_level broker,
    dan perhitungan ulang RR aktual berdasarkan harga eksekusi aktual:
    - Default menggunakan TP1 sebagai target eksekusi utama.
    - DILARANG menggunakan TP2 hanya untuk membuat RR terlihat memenuhi syarat apabila TP1 gagal RR minimum!
    """
    conf = cfg or get_auto_trading_config()
    dir_u = str(direction or "").strip().upper()
    u_spec = get_symbol_unit_spec(symbol=symbol, sym_info=sym_info)
    digits = int(u_spec["digits"])
    tick_size = float(u_spec["trade_tick_size"] or u_spec["point"] or 0.01)
    stops_pts = mt5_points_to_price_distance(u_spec["trade_stops_level"] or 0, spec=u_spec)
    freeze_pts = mt5_points_to_price_distance(u_spec["trade_freeze_level"] or 0, spec=u_spec)

    def _norm_px(px: float) -> float:
        if px <= 0 or not math.isfinite(px):
            return 0.0
        if tick_size > 0:
            return round(round(px / tick_size) * tick_size, digits)
        return round(px, digits)

    entry = _norm_px(float(entry_price or 0.0))
    sl = _norm_px(float(sl_price or 0.0))
    tp1 = _norm_px(float(tp1_price or 0.0))
    tp2 = _norm_px(float(tp2_price or 0.0))

    if dir_u not in ("BUY", "SELL") or entry <= 0:
        return {
            "valid": False,
            "reason_code": REASON_INVALID_STOP_LOSS,
            "reason_text": "Arah transaksi atau harga entry tidak valid",
        }

    if conf.get("REQUIRE_STOP_LOSS", True) and sl <= 0:
        return {
            "valid": False,
            "reason_code": REASON_INVALID_STOP_LOSS,
            "reason_text": "Stop Loss wajib tersedia dan > 0",
        }

    if conf.get("REQUIRE_TAKE_PROFIT", True) and tp1 <= 0:
        return {
            "valid": False,
            "reason_code": REASON_INVALID_TAKE_PROFIT,
            "reason_text": "Take Profit (TP1) wajib tersedia dan > 0",
        }

    # Validasi arah SL & TP (Section 11)
    if dir_u == "BUY":
        if sl >= entry:
            return {
                "valid": False,
                "reason_code": REASON_INVALID_STOP_LOSS,
                "reason_text": f"BUY wajib memiliki SL ({sl:.{digits}f}) di bawah Entry ({entry:.{digits}f})",
            }
        if tp1 <= entry:
            return {
                "valid": False,
                "reason_code": REASON_INVALID_TAKE_PROFIT,
                "reason_text": f"BUY wajib memiliki TP ({tp1:.{digits}f}) di atas Entry ({entry:.{digits}f})",
            }
        if structural_invalidation > 0 and sl > structural_invalidation + 1e-9:
            return {
                "valid": False,
                "reason_code": REASON_INVALID_STOP_LOSS,
                "reason_text": (
                    f"SL BUY ({sl:.{digits}f}) tidak berada di balik structural invalidation "
                    f"({structural_invalidation:.{digits}f})"
                ),
            }
        risk_dist = entry - sl
        reward_tp1 = tp1 - entry
        reward_tp2 = (tp2 - entry) if tp2 > entry else 0.0
    else:
        if sl <= entry:
            return {
                "valid": False,
                "reason_code": REASON_INVALID_STOP_LOSS,
                "reason_text": f"SELL wajib memiliki SL ({sl:.{digits}f}) di atas Entry ({entry:.{digits}f})",
            }
        if tp1 >= entry:
            return {
                "valid": False,
                "reason_code": REASON_INVALID_TAKE_PROFIT,
                "reason_text": f"SELL wajib memiliki TP ({tp1:.{digits}f}) di bawah Entry ({entry:.{digits}f})",
            }
        if structural_invalidation > 0 and sl < structural_invalidation - 1e-9:
            return {
                "valid": False,
                "reason_code": REASON_INVALID_STOP_LOSS,
                "reason_text": (
                    f"SL SELL ({sl:.{digits}f}) tidak berada di balik structural invalidation "
                    f"({structural_invalidation:.{digits}f})"
                ),
            }
        risk_dist = sl - entry
        reward_tp1 = entry - tp1
        reward_tp2 = (entry - tp2) if (0 < tp2 < entry) else 0.0

    max_sl_dist = float(conf.get("SL_JARAK_MAKS", 20.0))
    if risk_dist <= max(tick_size * 2.0, 0.30):
        return {
            "valid": False,
            "reason_code": REASON_INVALID_STOP_LOSS,
            "reason_text": f"Jarak SL terlalu sempit ({risk_dist:.{digits}f} USD)",
        }
    if risk_dist > max_sl_dist + 1e-6:
        return {
            "valid": False,
            "reason_code": REASON_INVALID_STOP_LOSS,
            "reason_text": f"Jarak SL ({risk_dist:.{digits}f} USD) melebihi batas {max_sl_dist:.2f} USD",
        }

    # Validasi jarak minimum broker stops_level & freeze_level
    min_broker_dist = max(stops_pts, freeze_pts)
    if min_broker_dist > 0:
        if is_market_order and bid and ask and bid > 0 and ask > 0:
            ref_sl_px = float(bid) if dir_u == "BUY" else float(ask)
            sl_brk_dist = (ref_sl_px - sl) if dir_u == "BUY" else (sl - ref_sl_px)
            tp_brk_dist = (tp1 - ref_sl_px) if dir_u == "BUY" else (ref_sl_px - tp1)
        else:
            sl_brk_dist = risk_dist
            tp_brk_dist = reward_tp1

        if sl_brk_dist < min_broker_dist - 1e-9 or tp_brk_dist < min_broker_dist - 1e-9:
            return {
                "valid": False,
                "reason_code": REASON_BROKER_CONSTRAINTS,
                "reason_text": (
                    f"Jarak SL/TP melanggar stops/freeze level broker "
                    f"(min {min_broker_dist:.{digits}f}, SL dist={sl_brk_dist:.{digits}f}, TP dist={tp_brk_dist:.{digits}f})"
                ),
            }

    min_rr = float(conf.get("MIN_RR", 1.5))
    actual_rr = round(reward_tp1 / risk_dist, 2) if risk_dist > 0 else 0.0
    rr_tp2 = round(reward_tp2 / risk_dist, 2) if (risk_dist > 0 and reward_tp2 > 0) else 0.0

    # TP1 wajib memenuhi RR minimum (DILARANG memakai TP2 untuk menutupi TP1 yang < MIN_RR)
    if reward_tp1 / risk_dist < min_rr - 1e-9:
        return {
            "valid": False,
            "reason_code": REASON_RR_BELOW_MINIMUM,
            "reason_text": f"RR aktual ke TP1 ({actual_rr:.2f}) di bawah minimum 1:{min_rr:.2f}",
            "entry": entry,
            "sl": sl,
            "tp": tp1,
            "tp1": tp1,
            "tp2": tp2,
            "actual_rr": actual_rr,
            "rr_tp2": rr_tp2,
            "risk_distance": round(risk_dist, digits),
        }

    return {
        "valid": True,
        "reason_code": REASON_OK,
        "reason_text": f"SL/TP valid (RR aktual 1:{actual_rr:.2f})",
        "entry": entry,
        "sl": sl,
        "tp": tp1,        # TP1 sebagai target eksekusi utama eksplisit
        "tp1": tp1,
        "tp2": tp2,
        "actual_rr": actual_rr,
        "rr_tp2": rr_tp2,
        "risk_distance": round(risk_dist, digits),
        "reward_distance": round(reward_tp1, digits),
        "digits": digits,
        "tick_size": tick_size,
    }
