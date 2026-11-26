# -*- coding: utf-8 -*-
"""
tests/test_adaptive_auto_trading.py
Unit & Integration Test Suite (40 Mandatory Scenarios + CLI + Backtest No-Lookahead)
menggunakan Mock MT5 terisolasi — 100% aman, tidak pernah menyentuh akun/order MT5 live!
"""
from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from modules.adaptive_backtester import run_comprehensive_backtest_comparison
from modules.config import SCANNER_MAGIC_NUMBER, get_auto_trading_config
from modules.dynamic_lot_manager import resolve_dynamic_lot_for_trade, select_target_lot, validate_strong_zone
from modules.execution_engine import (
    MT5ExecutionEngine,
    RETCODE_DONE,
    RETCODE_DONE_PARTIAL,
    RETCODE_INVALID_STOPS,
    RETCODE_PLACED,
    RETCODE_TIMEOUT,
)
from modules.execution_journal import ExecutionJournal
from modules.execution_monitor import AutoExecutionMonitor, format_adaptive_execution_panel, handle_auto_cli_command
from modules.execution_state import (
    ExecutionStateManager,
    REASON_ACCOUNT_CHANGED,
    REASON_DAILY_LOSS_LIMIT,
    REASON_DUPLICATE_SETUP,
    REASON_HTF_CONFLICT,
    REASON_INVALID_STOP_LOSS,
    REASON_INVALID_TAKE_PROFIT,
    REASON_INVALID_VOLUME,
    REASON_MARKET_CLOSED,
    REASON_MIN_LOT_EXCEEDS_RISK,
    REASON_PRICE_LEFT_ENTRY_RANGE,
    REASON_REAL_ACCOUNT_UNCONFIRMED,
    REASON_RR_BELOW_MINIMUM,
    REASON_SPREAD_TOO_HIGH,
    REASON_STALE_NEWS_DATA,
    STATE_BLOCKED,
    STATE_ENTRY_READY,
    STATE_INVALIDATED,
    STATE_ORDER_PLACED,
    STATE_POSITION_ACTIVE,
    make_setup_id,
)
from modules.mt5_conn import tambah_indikator
from modules.order_manager import PendingOrderManager
from modules.position_manager import PositionManager
from modules.risk_manager import (
    compute_effective_risk_budget,
    evaluate_portfolio_risk_and_daily_limits,
    inspect_mt5_account_and_permissions,
    validate_sl_tp_and_rr,
)
from modules.trade_validator import validate_trade_candidate
from modules.zone_scanner import Zone


# =============================================================================
# DETERMINISTIC MOCK MT5 ENGINE (Supports USD & Real IDC Cent Accounts!)
# =============================================================================
class MockMT5:
    ACCOUNT_TRADE_MODE_DEMO = 0
    ACCOUNT_TRADE_MODE_CONTEST = 1
    ACCOUNT_TRADE_MODE_REAL = 2

    SYMBOL_TRADE_MODE_DISABLED = 0
    SYMBOL_TRADE_MODE_CLOSEONLY = 3
    SYMBOL_TRADE_MODE_FULL = 4

    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    ORDER_TYPE_BUY_LIMIT = 2
    ORDER_TYPE_SELL_LIMIT = 3

    TRADE_ACTION_DEAL = 1
    TRADE_ACTION_PENDING = 5
    TRADE_ACTION_SLTP = 6
    TRADE_ACTION_REMOVE = 8

    ORDER_TIME_GTC = 0
    ORDER_TIME_SPECIFIED = 2
    ORDER_FILLING_FOK = 0
    ORDER_FILLING_IOC = 1
    ORDER_FILLING_RETURN = 2

    DEAL_TYPE_BUY = 0
    DEAL_TYPE_SELL = 1
    DEAL_ENTRY_IN = 0
    DEAL_ENTRY_OUT = 1

    def __init__(
        self,
        login: int = 88001122,
        server: str = "Broker-Real",
        currency: str = "IDC",
        trade_mode: int = 2,             # REAL by default
        balance: float = 15_000_000.0,   # e.g. 15,000,000 IDC (~$1,000 / cent scale)
        equity: Optional[float] = None,
        bid: float = 3645.00,
        ask: float = 3645.30,
        fx_idc_per_usd: float = 15000.0, # Digunakan internal oleh mock broker seperti terminal MT5 asli
    ):
        self._lock = threading.Lock()
        self.connected = True
        self.algo_allowed = True
        self.market_open = True
        self.login = login
        self.server = server
        self.currency = currency
        self.trade_mode = trade_mode
        self.balance = balance
        self.equity = equity if equity is not None else balance
        self.free_margin = self.equity * 0.90
        self.bid = bid
        self.ask = ask
        self.fx_mult = fx_idc_per_usd if currency == "IDC" else (100.0 if currency == "USC" else 1.0)

        self.volume_min = 0.01
        self.volume_max = 50.0
        self.volume_step = 0.01
        self.stops_level = 10   # 0.10 USD

        self.positions: List[SimpleNamespace] = []
        self.orders: List[SimpleNamespace] = []
        self.deals: List[SimpleNamespace] = []
        self.send_calls: List[Dict[str, Any]] = []
        self._next_ticket = 50001

        # Fault-injection flags untuk skenario uji spesifik
        self.simulate_timeout_but_fill = False
        self.simulate_partial_fill_ratio: Optional[float] = None
        self.simulate_missing_sl_on_fill = False
        self.reject_sl_modify = False
        self.fill_pending_during_cancel = False
        self.pre_send_spread_spike: Optional[float] = None

    def terminal_info(self):
        if not self.connected:
            return None
        return SimpleNamespace(
            connected=True,
            trade_allowed=self.algo_allowed,
            tradeapi_disabled=(not self.algo_allowed),
        )

    def account_info(self):
        if not self.connected:
            return None
        floating = sum(float(getattr(p, "profit", 0.0)) for p in self.positions)
        return SimpleNamespace(
            login=self.login,
            server=self.server,
            currency=self.currency,
            trade_mode=self.trade_mode,
            balance=self.balance,
            equity=self.equity,
            margin=max(0.0, self.equity - self.free_margin),
            margin_free=self.free_margin,
            margin_level=1200.0,
            leverage=500,
            profit=floating,
            trade_allowed=self.algo_allowed,
            trade_expert=True,
        )

    def symbol_info(self, symbol: str):
        if not self.connected:
            return None
        return SimpleNamespace(
            name=symbol,
            digits=2,
            point=0.01,
            trade_tick_size=0.01,
            trade_tick_value=1.0 * self.fx_mult,
            trade_contract_size=100.0,
            trade_stops_level=self.stops_level,
            trade_freeze_level=0,
            volume_min=self.volume_min,
            volume_max=self.volume_max,
            volume_step=self.volume_step,
            filling_mode=2,     # IOC
            expiration_mode=7,  # Supports SPECIFIED
            trade_mode=(self.SYMBOL_TRADE_MODE_FULL if self.market_open else self.SYMBOL_TRADE_MODE_DISABLED),
            spread=int(round((self.ask - self.bid) / 0.01)),
        )

    def symbol_select(self, symbol: str, enable: bool = True):
        return True

    def symbol_info_tick(self, symbol: str):
        if not self.connected:
            return None
        if self.pre_send_spread_spike is not None:
            return SimpleNamespace(bid=self.bid, ask=round(self.bid + self.pre_send_spread_spike, 2), time=int(time.time()))
        return SimpleNamespace(bid=self.bid, ask=self.ask, time=int(time.time()))

    def order_calc_profit(self, action: int, symbol: str, volume: float, price_open: float, price_close: float):
        is_buy = (action == self.ORDER_TYPE_BUY)
        pts = (price_close - price_open) if is_buy else (price_open - price_close)
        return round(pts * 100.0 * float(volume) * self.fx_mult, 2)

    def order_calc_margin(self, action: int, symbol: str, volume: float, price: float):
        return round((price * 100.0 * float(volume) / 500.0) * self.fx_mult, 2)

    def positions_get(self, symbol: Optional[str] = None, ticket: Optional[int] = None):
        with self._lock:
            out = list(self.positions)
        if ticket is not None:
            out = [p for p in out if p.ticket == int(ticket)]
        elif symbol is not None:
            out = [p for p in out if p.symbol == symbol]
        return tuple(out)

    def orders_get(self, symbol: Optional[str] = None, ticket: Optional[int] = None):
        with self._lock:
            out = list(self.orders)
        if ticket is not None:
            out = [o for o in out if o.ticket == int(ticket)]
        elif symbol is not None:
            out = [o for o in out if o.symbol == symbol]
        return tuple(out)

    def history_deals_get(self, date_from=None, date_to=None, position=None):
        with self._lock:
            out = list(self.deals)
        if position is not None:
            out = [d for d in out if d.position_id == int(position)]
        return tuple(out)

    def order_check(self, request: Dict[str, Any]):
        if not self.market_open:
            return SimpleNamespace(retcode=10018, comment="Market closed")
        return SimpleNamespace(retcode=0, comment="Done")

    def order_send(self, request: Dict[str, Any]):
        with self._lock:
            self.send_calls.append(dict(request))
            act = request.get("action")

            # 1. CANCEL PENDING ORDER
            if act == self.TRADE_ACTION_REMOVE:
                tkt = int(request["order"])
                ord_item = next((o for o in self.orders if o.ticket == tkt), None)
                if ord_item is None:
                    return SimpleNamespace(retcode=10013, comment="Not found")
                self.orders = [o for o in self.orders if o.ticket != tkt]
                if self.fill_pending_during_cancel:
                    # Simulasi race condition: order tepat terisi menjadi posisi saat request cancel tiba!
                    pos_type = self.ORDER_TYPE_BUY if ord_item.type == self.ORDER_TYPE_BUY_LIMIT else self.ORDER_TYPE_SELL
                    self.positions.append(SimpleNamespace(
                        ticket=tkt,
                        symbol=ord_item.symbol,
                        type=pos_type,
                        volume=ord_item.volume_current,
                        price_open=ord_item.price_open,
                        sl=ord_item.sl,
                        tp=ord_item.tp,
                        profit=0.0,
                        swap=0.0,
                        commission=0.0,
                        magic=ord_item.magic,
                        comment=ord_item.comment,
                        time=int(time.time()),
                    ))
                    return SimpleNamespace(retcode=10013, comment="Already filled")
                return SimpleNamespace(retcode=RETCODE_DONE, order=tkt, comment="Cancelled")

            # 2. MODIFY SL/TP
            if act == self.TRADE_ACTION_SLTP:
                if self.reject_sl_modify:
                    return SimpleNamespace(retcode=RETCODE_INVALID_STOPS, comment="Invalid stops")
                pt = int(request["position"])
                pos = next((p for p in self.positions if p.ticket == pt), None)
                if pos is None:
                    return SimpleNamespace(retcode=10013, comment="Pos not found")
                pos.sl = float(request.get("sl", pos.sl))
                pos.tp = float(request.get("tp", pos.tp))
                return SimpleNamespace(retcode=RETCODE_DONE, order=pt, comment="Modified")

            # 3. CLOSE / PARTIAL CLOSE EXISTING POSITION
            if act == self.TRADE_ACTION_DEAL and request.get("position"):
                pt = int(request["position"])
                pos = next((p for p in self.positions if p.ticket == pt), None)
                if pos is None:
                    return SimpleNamespace(retcode=10013, comment="Pos not found")
                req_v = round(float(request.get("volume", pos.volume) or pos.volume), 2)
                if req_v < float(pos.volume) - 1e-9:
                    pos.volume = round(float(pos.volume) - req_v, 2)
                    return SimpleNamespace(retcode=RETCODE_DONE, order=pt, deal=pt + 1000, price=request["price"], volume=req_v)
                self.positions = [p for p in self.positions if p.ticket != pt]
                return SimpleNamespace(retcode=RETCODE_DONE, order=pt, deal=pt + 1000, price=request["price"], volume=pos.volume)

            tkt = self._next_ticket
            self._next_ticket += 1

            # 4. PLACE PENDING ORDER
            if act == self.TRADE_ACTION_PENDING:
                self.orders.append(SimpleNamespace(
                    ticket=tkt,
                    symbol=request["symbol"],
                    type=request["type"],
                    volume_initial=float(request["volume"]),
                    volume_current=float(request["volume"]),
                    price_open=float(request["price"]),
                    sl=float(request.get("sl", 0.0)),
                    tp=float(request.get("tp", 0.0)),
                    time_setup=int(time.time()),
                    time_expiration=int(request.get("expiration", 0)),
                    magic=int(request.get("magic", 0)),
                    comment=str(request.get("comment", "")),
                ))
                return SimpleNamespace(retcode=RETCODE_PLACED, order=tkt, deal=0, price=request["price"], volume=request["volume"])

            # 5. MARKET ORDER
            req_vol = float(request["volume"])
            fill_vol = round(req_vol * self.simulate_partial_fill_ratio, 2) if self.simulate_partial_fill_ratio else req_vol
            sl_val = 0.0 if self.simulate_missing_sl_on_fill else float(request.get("sl", 0.0))
            tp_val = float(request.get("tp", 0.0))

            self.positions.append(SimpleNamespace(
                ticket=tkt,
                symbol=request["symbol"],
                type=request["type"],
                volume=fill_vol,
                price_open=float(request["price"]),
                sl=sl_val,
                tp=tp_val,
                profit=0.0,
                swap=0.0,
                commission=0.0,
                magic=int(request.get("magic", 0)),
                comment=str(request.get("comment", "")),
                time=int(time.time()),
            ))
            if self.simulate_timeout_but_fill:
                return SimpleNamespace(retcode=RETCODE_TIMEOUT, order=0, deal=0, price=0.0, volume=0.0, comment="Timeout")
            rc = RETCODE_DONE_PARTIAL if self.simulate_partial_fill_ratio else RETCODE_DONE
            return SimpleNamespace(retcode=rc, order=tkt, deal=tkt + 500, price=request["price"], volume=fill_vol, comment="Done")


# =============================================================================
# SYNTHETIC OHLC & ZONE BUILDERS FOR PRECISION TESTING
# =============================================================================
def _build_m5_df_with_bearish_rejection(zone_low: float = 3643.0, zone_high: float = 3647.0, bars: int = 80) -> pd.DataFrame:
    """
    Buat DataFrame M5 CLOSED di mana harga naik menyentuh Supply Zone (3643-3647)
    lalu membentuk candle Bearish Rejection objektif (wick atas panjang >= 55% range, close di 20% bawah)
    TANPA membentuk CHoCH terlebih dahulu!
    """
    now_floor = pd.Timestamp("2026-11-24 14:00:00")
    idx = pd.date_range(end=now_floor - pd.Timedelta(minutes=5), periods=bars, freq="5min")
    opens, highs, lows, closes = [], [], [], []
    base = 3630.0
    for i in range(bars - 1):
        # Naik bertahap halus tanpa membuat swing low yang ditembus ke bawah (sehingga CHoCH = False!)
        p = base + (i / float(bars)) * 13.5
        opens.append(round(p - 0.3, 2))
        closes.append(round(p + 0.3, 2))
        highs.append(round(p + 0.8, 2))
        lows.append(round(p - 0.7, 2))

    # Candle terakhir (closed): menyentuh zona Supply 3643-3647, wick tajam ke 3646.60, close tegas di 3644.20
    # (Close 3644.20 berada di atas local high 6 bar sebelumnya ~3644.13 sehingga murni VALID REJECTION tanpa sweep+reclaim & tanpa CHoCH!)
    opens.append(3644.40)
    highs.append(3646.60)
    lows.append(3643.80)
    closes.append(3644.20)

    df = pd.DataFrame({
        "Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": [150] * bars
    }, index=idx)
    df.attrs["time_tz"] = "UTC"
    df.attrs["is_closed"] = True
    return tambah_indikator(df)


def _make_supply_zone(
    score: float = 94.0,
    grade: str = "A+",
    timeframe: str = "H4+H1",
    lo: float = 3643.0,
    hi: float = 3647.0,
    is_valid: bool = True,
    disp: float = 2.4,
    freshness: str = "FRESH",
    touch_count: int = 0,
) -> Zone:
    return Zone(
        zone_id=f"ZONE_{timeframe}_SUPPLY_17000001",
        symbol="XAUUSDc",
        timeframe=timeframe,
        zone_type="SUPPLY",
        price_low=lo,
        price_high=hi,
        midpoint=round((lo + hi) / 2.0, 2),
        score=score,
        grade=grade,
        origin_bos={"tipe": "BOS", "arah": "bear", "level": 3640.0, "strength_atr": 0.5, "origin_valid": True},
        fvg_confirmation={"status": "UNTOUCHED", "price_low": 3640.0, "price_high": 3643.0, "size_atr": 0.8},
        displacement_strength=disp,
        freshness=freshness,
        touch_count=touch_count,
        liquidity_sweep_status="CONFIRMED_SWEEP_HIGH",
        invalidation_price=hi,
        is_valid=is_valid,
        rejection_reason="" if is_valid else "Zona ditembus close",
        ob_candle_time=pd.Timestamp("2026-11-24 04:00:00"),
        bos_candle_time=pd.Timestamp("2026-11-24 08:00:00"),
        bos_timeframe="H1",
        sl_recommended=3650.0,
        tp1_recommended=3625.0,
        tp2_recommended=3615.0,
    )


def _make_snapshot_and_scan(
    zone: Zone,
    df_m5: pd.DataFrame,
    bid: float = 3644.00,
    ask: float = 3644.20,
    h4_bias: str = "BEARISH",
    h1_bias: str = "BEARISH",
    m30_status: str = "CONFIRMED",
    has_conflict: bool = False,
    market_status: str = "OPEN",
    freshness: str = "FRESH",
):
    mid = round((bid + ask) / 2.0, 2)
    spread = round(ask - bid, 2)
    frames = {"M5": df_m5}
    snap = {
        "snapshot_id": "SNAP_TEST_01",
        "symbol": "XAUUSDc",
        "bid": bid,
        "ask": ask,
        "mid": mid,
        "current_price": mid,
        "spread": spread,
        "frames": frames,
        "market_status": market_status,
        "data_freshness": freshness,
        "tick_age": 0.5,
        "server_time": pd.Timestamp("2026-11-24 14:00:05"),
    }
    scan = {
        "snapshot_id": "SNAP_TEST_01",
        "symbol": "XAUUSDc",
        "current_price": mid,
        "bid": bid,
        "ask": ask,
        "mid": mid,
        "spread": spread,
        "market_status": market_status,
        "data_freshness": freshness,
        "htf_bias": h4_bias,
        "h1_structure": h1_bias,
        "m30_status": m30_status,
        "htf_context": {
            "H4": h4_bias,
            "H1": h1_bias,
            "m30_status": m30_status,
            "has_conflict": has_conflict,
        },
        "active_zones": [zone.to_dict()],
        "radar_zones": [zone.to_dict()],
        "rejected_zones": [],
    }
    dec = {
        "zone": zone.to_dict(),
        "primary_zone": zone.to_dict(),
        "plan": {
            "zone_id": zone.zone_id,
            "arah": "SELL" if zone.zone_type == "SUPPLY" else "BUY",
            "sl": 3650.0 if zone.zone_type == "SUPPLY" else 3638.0,
            "tp1": 3625.0 if zone.zone_type == "SUPPLY" else 3665.0,
            "tp2": 3615.0 if zone.zone_type == "SUPPLY" else 3675.0,
        },
        "stages": {"M30": {"status": m30_status}},
    }
    return snap, scan, dec


class TestAdaptiveAutoTrading40Scenarios(unittest.TestCase):
    """Pengujian deterministik 40 skenario wajib Section 22 + Backtest No-Lookahead."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.state_file = os.path.join(self.tmpdir.name, "state_exec_test.json")
        self.journal_file = os.path.join(self.tmpdir.name, "journal_exec_test.csv")
        self.cfg = get_auto_trading_config()
        self.cfg["SYMBOL"] = "XAUUSDc"
        self.cfg.update({
            "SNR_ZONE_MIN_STABLE_SECONDS": 0.0,
            "LOT_STANDARD": 0.01,
            "LOT_MEDIUM": 0.02,
            "LOT_STRONG": 0.05,
            "MAX_LOT": 0.05,
            "EARLY_PENDING_MAX_LOT": 0.01,
            "LTF_MAX_LOT": 0.02,
            "HTF_STRONG_MAX_LOT": 0.05,
            "MIN_RR": 1.5,
            "TP1_RR": 1.5,
            "TP1_MODE": "STRUCTURAL",
            "FORCE_FIXED_TP1": False,
            "RISK_EARLY_PENDING_PERCENT": 0.25,
            "RISK_ADAPTIVE_PERCENT": 0.50,
            "RISK_CONFIRMED_PERCENT": 0.50,
            "MAX_RISK_PER_TRADE_PERCENT": 1.0,
            "MAX_TOTAL_EXPOSURE_PERCENT": 1.0,
            "MAX_DAILY_LOSS_PERCENT": 2.0,
            "SL_JARAK_MAKS": 20.0,
            "CONFLICT_MIDZONE_LIMIT_ENABLED": False,
            "PARTIAL_CLOSE_ENABLED": False,
            "GRADE_A_WAIT_CONFIRMATION_ENABLED": False,
            "AUTO_CANCEL_GRADE_A_PENDING_LIMIT": False,
            "SCALE_IN_80PLUS_ENABLED": False,
        })
        self.news_ok = {"blackout": False, "status": "READY", "age_seconds": 10.0}
        self.df_m5_rej = _build_m5_df_with_bearish_rejection()

    def tearDown(self):
        self.tmpdir.cleanup()

    def _new_env(self, real_confirmed: bool = True, equity_idc: float = 30_000_000.0, bid: float = 3644.0, ask: float = 3644.2):
        mt5 = MockMT5(balance=equity_idc, equity=equity_idc, bid=bid, ask=ask)
        sm = ExecutionStateManager(state_file=self.state_file, auto_load=False)
        sm.authorize_account(
            login=mt5.login, server=mt5.server, currency=mt5.currency,
            account_mode="REAL", real_confirmed=real_confirmed,
            balance=equity_idc, equity=equity_idc,
        )
        sm.set_trading_mode("FULL_AUTO", auto_trading=True)
        jrn = ExecutionJournal(journal_path=self.journal_file)
        eng = MT5ExecutionEngine(mt5_api=mt5, state_mgr=sm, journal=jrn, cfg=self.cfg)
        return mt5, sm, jrn, eng

    # -------------------------------------------------------------------------
    # SKENARIO 1 & 2 & 17: Zona valid + Rejection valid + CHoCH belum muncul
    # -> ADAPTIVE_ENTRY (OR logic, tidak menunggu semua trigger, eksekusi tanpa WAIT!)
    # -------------------------------------------------------------------------
    def test_01_02_17_valid_rejection_without_choch_executes_immediately(self):
        mt5, sm, _, eng = self._new_env()
        zone = _make_supply_zone(score=94.0, timeframe="H4+H1")
        snap, scan, dec = _make_snapshot_and_scan(zone, self.df_m5_rej)

        val = validate_trade_candidate(
            zone, mt5, sm, snapshot=snap, scan_result=scan, decision=dec,
            news_override=self.news_ok, cfg=self.cfg,
        )
        self.assertEqual(val["entry_mode"], "ADAPTIVE_ENTRY")
        self.assertTrue(val["triggers"]["valid_rejection"])
        self.assertFalse(val["triggers"]["valid_choch"])  # CHoCH belum muncul!
        self.assertEqual(val["state"], STATE_ENTRY_READY)
        self.assertTrue(val["can_execute"])

        ex = eng.execute_validated_setup(val, trading_mode="FULL_AUTO")
        self.assertTrue(ex["executed"])
        self.assertEqual(ex["state"], STATE_POSITION_ACTIVE)
        self.assertEqual(len(mt5.positions), 1)

    # -------------------------------------------------------------------------
    # SKENARIO 3: Zona invalid + trigger valid -> BLOCK / INVALIDATED
    # -------------------------------------------------------------------------
    def test_03_invalid_zone_with_valid_trigger_blocked(self):
        mt5, sm, _, _ = self._new_env()
        zone = _make_supply_zone(score=92.0, is_valid=False, freshness="INVALIDATED")
        snap, scan, dec = _make_snapshot_and_scan(zone, self.df_m5_rej)
        val = validate_trade_candidate(zone, mt5, sm, snapshot=snap, scan_result=scan, decision=dec, news_override=self.news_ok, cfg=self.cfg)
        self.assertEqual(val["state"], STATE_INVALIDATED)
        self.assertFalse(val["can_execute"])

    # -------------------------------------------------------------------------
    # SKENARIO 4: H1 neutral + H4/M30 aligned + trigger valid -> ADAPTIVE_ENTRY
    # -------------------------------------------------------------------------
    def test_04_h1_neutral_with_h4_m30_aligned_allowed(self):
        mt5, sm, _, _ = self._new_env()
        zone = _make_supply_zone(score=85.0, timeframe="H1")
        snap, scan, dec = _make_snapshot_and_scan(zone, self.df_m5_rej, h4_bias="BEARISH", h1_bias="NETRAL", m30_status="CONFIRMED")
        val = validate_trade_candidate(zone, mt5, sm, snapshot=snap, scan_result=scan, decision=dec, news_override=self.news_ok, cfg=self.cfg)
        self.assertEqual(val["state"], STATE_ENTRY_READY)
        self.assertEqual(val["entry_mode"], "ADAPTIVE_ENTRY")

    # -------------------------------------------------------------------------
    # SKENARIO 5: H4/H1 conflict -> EARLY_PENDING diblokir
    # -------------------------------------------------------------------------
    def test_05_h4_h1_conflict_blocks_early_pending(self):
        mt5, sm, _, _ = self._new_env(bid=3630.0, ask=3630.2)
        df_untouched = self.df_m5_rej.iloc[:-15].copy()
        df_untouched.attrs["is_closed"] = True
        zone = _make_supply_zone(score=92.0, timeframe="H4+H1", lo=3643.0, hi=3647.0)
        snap, scan, dec = _make_snapshot_and_scan(
            zone, df_untouched, bid=3630.0, ask=3630.2,
            h4_bias="BEARISH", h1_bias="BULLISH", has_conflict=True,
        )
        val = validate_trade_candidate(
            zone, mt5, sm, snapshot=snap, scan_result=scan, decision=dec,
            preferred_entry_mode="EARLY_PENDING", news_override=self.news_ok, cfg=self.cfg,
        )
        self.assertEqual(val["state"], STATE_BLOCKED)
        self.assertEqual(val["reason_code"], REASON_HTF_CONFLICT)

    # -------------------------------------------------------------------------
    # SKENARIO 6: Harga meninggalkan zona -> jangan chase (PRICE_LEFT_ENTRY_RANGE)
    # -------------------------------------------------------------------------
    def test_06_price_left_zone_no_chase(self):
        # Zona 3643-3647 sudah tersentuh candle M5, tapi harga live sudah terjun jauh ke 3635.0
        mt5, sm, _, _ = self._new_env(bid=3635.0, ask=3635.2)
        zone = _make_supply_zone(score=90.0)
        snap, scan, dec = _make_snapshot_and_scan(zone, self.df_m5_rej, bid=3635.0, ask=3635.2)
        val = validate_trade_candidate(zone, mt5, sm, snapshot=snap, scan_result=scan, decision=dec, news_override=self.news_ok, cfg=self.cfg)
        self.assertEqual(val["state"], STATE_BLOCKED)
        self.assertEqual(val["reason_code"], REASON_PRICE_LEFT_ENTRY_RANGE)

    # -------------------------------------------------------------------------
    # SKENARIO 7 & 16: RR kurang dari minimum -> BLOCK; SL/TP salah arah -> BLOCK
    # -------------------------------------------------------------------------
    def test_07_16_rr_below_minimum_and_wrong_sl_tp_direction_blocked(self):
        mt5, sm, _, _ = self._new_env()
        zone = _make_supply_zone(score=88.0)
        snap, scan, dec = _make_snapshot_and_scan(zone, self.df_m5_rej)
        # RR = (3644 - 3640) / (3650 - 3644) = 4 / 6 = 0.67 < 1.5 -> BLOCK!
        dec["plan"]["tp1"] = 3640.0
        dec["plan"]["tp2"] = 3610.0  # TP2 jauh tidak boleh dipakai memanipulasi TP1 yang gagal RR!
        val_rr = validate_trade_candidate(zone, mt5, sm, snapshot=snap, scan_result=scan, decision=dec, news_override=self.news_ok, cfg=self.cfg)
        self.assertEqual(val_rr["state"], STATE_BLOCKED)
        self.assertEqual(val_rr["reason_code"], REASON_RR_BELOW_MINIMUM)

        # Skenario 16: SL salah arah untuk SELL (SL di bawah entry) & TP di atas entry
        bad_sl = validate_sl_tp_and_rr("SELL", entry_price=3644.0, sl_price=3640.0, tp1_price=3625.0, cfg=self.cfg)
        self.assertFalse(bad_sl["valid"])
        self.assertEqual(bad_sl["reason_code"], REASON_INVALID_STOP_LOSS)

        bad_tp = validate_sl_tp_and_rr("SELL", entry_price=3644.0, sl_price=3650.0, tp1_price=3652.0, cfg=self.cfg)
        self.assertFalse(bad_tp["valid"])
        self.assertEqual(bad_tp["reason_code"], REASON_INVALID_TAKE_PROFIT)

    # -------------------------------------------------------------------------
    # SKENARIO 8 & 15: Duplicate setup -> BLOCK & Restart script -> tidak duplicate
    # -------------------------------------------------------------------------
    def test_08_15_duplicate_setup_and_restart_persistence_blocked(self):
        mt5, sm, jrn, eng = self._new_env()
        zone = _make_supply_zone(score=90.0)
        snap, scan, dec = _make_snapshot_and_scan(zone, self.df_m5_rej)

        val1 = validate_trade_candidate(zone, mt5, sm, snapshot=snap, scan_result=scan, decision=dec, news_override=self.news_ok, cfg=self.cfg)
        ex1 = eng.execute_validated_setup(val1, trading_mode="FULL_AUTO")
        self.assertTrue(ex1["executed"])

        # Percobaan ke-2 pada setup yang sama -> BLOCKED DUPLICATE_SETUP
        val2 = validate_trade_candidate(zone, mt5, sm, snapshot=snap, scan_result=scan, decision=dec, news_override=self.news_ok, cfg=self.cfg)
        self.assertFalse(val2["can_execute"])
        self.assertEqual(val2["reason_code"], REASON_DUPLICATE_SETUP)

        # Skenario 15: Simulasi Restart Script (buat instance ExecutionStateManager baru dari file disk!)
        sm_restarted = ExecutionStateManager(state_file=self.state_file, auto_load=True)
        sm_restarted.reconcile_with_mt5(mt5.positions_get(), mt5.orders_get(), mt5.history_deals_get())
        # Akun real juga otomatis kembali butuh konfirmasi atau jika dikonfirmasi ulang tetap mendeteksi posisi aktif!
        sm_restarted.authorize_account(mt5.login, mt5.server, mt5.currency, "REAL", real_confirmed=True)
        sm_restarted.set_trading_mode("FULL_AUTO", auto_trading=True)

        val_after_restart = validate_trade_candidate(
            zone, mt5, sm_restarted, snapshot=snap, scan_result=scan, decision=dec, news_override=self.news_ok, cfg=self.cfg
        )
        self.assertFalse(val_after_restart["can_execute"])
        self.assertEqual(val_after_restart["reason_code"], REASON_DUPLICATE_SETUP)
        self.assertEqual(len(mt5.positions), 1)

    # -------------------------------------------------------------------------
    # SKENARIO 9 & 40: Market closed -> tidak kirim order, tetap hasilkan pre-market plan
    # -------------------------------------------------------------------------
    def test_09_40_market_closed_preserves_plan_without_executing(self):
        mt5, sm, _, eng = self._new_env()
        mt5.market_open = False
        zone = _make_supply_zone(score=94.0)
        snap, scan, dec = _make_snapshot_and_scan(zone, self.df_m5_rej, market_status="CLOSED")

        val = validate_trade_candidate(zone, mt5, sm, snapshot=snap, scan_result=scan, decision=dec, news_override=self.news_ok, cfg=self.cfg)
        self.assertEqual(val["state"], STATE_BLOCKED)
        self.assertEqual(val["reason_code"], REASON_MARKET_CLOSED)
        # Pre-market trade plan tetap utuh!
        self.assertEqual(val["trade_plan"]["direction"], "SELL")
        self.assertGreater(val["trade_plan"]["sl"], val["trade_plan"]["entry"])
        self.assertLess(val["trade_plan"]["tp1"], val["trade_plan"]["entry"])

        ex = eng.execute_validated_setup(val, trading_mode="FULL_AUTO")
        self.assertFalse(ex["executed"])
        self.assertEqual(len(mt5.send_calls), 0)

    # -------------------------------------------------------------------------
    # SKENARIO 10 & 27 & 39: Pending order management, anti-duplicate, & fill-during-cancel
    # -------------------------------------------------------------------------
    def test_10_27_39_pending_order_invalidation_duplicate_and_race_fill(self):
        mt5, sm, _, eng = self._new_env(bid=3635.0, ask=3635.2)
        ord_mgr = PendingOrderManager(eng, sm, cfg=self.cfg)
        zone = _make_supply_zone(score=92.0, timeframe="H4+H1", lo=3643.0, hi=3647.0)
        sid = make_setup_id(zone.zone_id, "SELL", symbol="XAUUSDc")

        # Pasang pending order scanner
        res_pend = mt5.order_send({
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": "XAUUSDc",
            "volume": 0.01,
            "type": mt5.ORDER_TYPE_SELL_LIMIT,
            "price": 3643.0,
            "sl": 3650.0,
            "tp": 3625.0,
            "magic": SCANNER_MAGIC_NUMBER,
            "comment": f"SMC_EP_{sid[-16:]}",
        })
        tkt = res_pend.order
        sm.update_setup_state(sid, STATE_ORDER_PLACED, zone_id=zone.zone_id, direction="SELL", entry_mode="EARLY_PENDING", order_ticket=tkt)

        # Skenario 27: Pending order aktif + muncul adaptive trigger pada setup yang sama -> TIDAK buka posisi baru!
        mt5.bid, mt5.ask = 3644.0, 3644.2
        snap, scan, dec = _make_snapshot_and_scan(zone, self.df_m5_rej)
        val_dup = validate_trade_candidate(zone, mt5, sm, snapshot=snap, scan_result=scan, decision=dec, news_override=self.news_ok, cfg=self.cfg)
        self.assertFalse(val_dup["can_execute"])
        self.assertEqual(val_dup["reason_code"], REASON_DUPLICATE_SETUP)

        # Skenario 10: Zona menjadi INVALIDATED -> otomatis batalkan pending order!
        inv_zone = _make_supply_zone(score=0.0, is_valid=False, freshness="INVALIDATED")
        inv_snap, inv_scan, _ = _make_snapshot_and_scan(inv_zone, self.df_m5_rej, bid=3648.5, ask=3648.7)
        audit_res = ord_mgr.audit_and_manage_pending_orders(snapshot=inv_snap, scan_result=inv_scan, news_override=self.news_ok, symbol="XAUUSDc")
        self.assertEqual(len(audit_res["cancelled"]), 1)
        self.assertEqual(len(mt5.orders_get()), 0)

        # Skenario 39: Pending order terisi tepat saat cancellation sedang berlangsung -> rekonsiliasi tanpa duplicate!
        res_pend2 = mt5.order_send({
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": "XAUUSDc",
            "volume": 0.01,
            "type": mt5.ORDER_TYPE_SELL_LIMIT,
            "price": 3643.0,
            "sl": 3650.0,
            "tp": 3625.0,
            "magic": SCANNER_MAGIC_NUMBER,
            "comment": "SMC_EP_RACE39",
        })
        tkt2 = res_pend2.order
        sid2 = "SETUP_XAUUSDC_SELL_RACE39"
        sm.update_setup_state(sid2, STATE_ORDER_PLACED, zone_id="Z_RACE", direction="SELL", order_ticket=tkt2)
        mt5.fill_pending_during_cancel = True

        sw_res = ord_mgr.switch_pending_to_market_safely(
            setup_id=sid2,
            order_ticket=tkt2,
            revalidate_fn=lambda: {"can_execute": True, "state": STATE_ENTRY_READY},
            symbol="XAUUSDc",
        )
        self.assertTrue(sw_res["already_filled"])
        self.assertFalse(sw_res["executed"])  # Market order baru DITAHAN!
        self.assertEqual(len(mt5.positions_get()), 1)  # Hanya 1 posisi dari limit yang terisi!

    # -------------------------------------------------------------------------
    # SKENARIO 11 & 30: MT5 timeout -> rekonsiliasi tanpa blind retry; Partial fill -> reconcile volume
    # -------------------------------------------------------------------------
    def test_11_30_timeout_reconciliation_and_partial_fill(self):
        # Skenario 11: Broker mengisi order tetapi respons mengalami TIMEOUT
        mt5, sm, _, eng = self._new_env()
        mt5.simulate_timeout_but_fill = True
        zone = _make_supply_zone(score=90.0)
        snap, scan, dec = _make_snapshot_and_scan(zone, self.df_m5_rej)

        val = validate_trade_candidate(zone, mt5, sm, snapshot=snap, scan_result=scan, decision=dec, news_override=self.news_ok, cfg=self.cfg)
        ex = eng.execute_validated_setup(val, trading_mode="FULL_AUTO")
        self.assertTrue(ex["executed"])
        self.assertTrue(ex.get("reconciled"))
        self.assertEqual(ex["state"], STATE_POSITION_ACTIVE)
        self.assertEqual(len(mt5.send_calls), 1)  # Tidak melakukan blind retry ke-2!

        # Skenario 30: Broker menerima Partial Fill (mis. minta 0.05 lot, terisi 0.03 lot)
        mt5_p, sm_p, _, eng_p = self._new_env(equity_idc=150_000_000.0)
        mt5_p.simulate_partial_fill_ratio = 0.60  # 0.05 * 0.60 = 0.03 lot
        zone_p = _make_supply_zone(score=95.0, timeframe="H4+H1")
        snap_p, scan_p, dec_p = _make_snapshot_and_scan(zone_p, self.df_m5_rej)
        val_p = validate_trade_candidate(zone_p, mt5_p, sm_p, snapshot=snap_p, scan_result=scan_p, decision=dec_p, news_override=self.news_ok, cfg=self.cfg)
        self.assertEqual(val_p["final_lot"], 0.05)

        ex_p = eng_p.execute_validated_setup(val_p, trading_mode="FULL_AUTO")
        self.assertTrue(ex_p["executed"])
        self.assertTrue(ex_p["partial_fill"])
        self.assertAlmostEqual(ex_p["actual_lot"], 0.03, places=2)

    # -------------------------------------------------------------------------
    # SKENARIO 12, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28:
    # Dynamic Lot Tiers, Strong Zone Caps, Step-down Downgrades, & Real IDC Currency
    # -------------------------------------------------------------------------
    def test_12_18_to_26_28_dynamic_lot_tiers_downgrades_and_idc_currency(self):
        # 18: Score 75 -> target lot 0.01
        z75 = _make_supply_zone(score=75.0, grade="A", timeframe="H1")
        self.assertEqual(select_target_lot(z75, "ADAPTIVE_ENTRY", strong_zone_validated=False, cfg=self.cfg), 0.01)

        # 19: Score 85 -> target lot 0.02
        z85 = _make_supply_zone(score=85.0, grade="A+", timeframe="H1")
        self.assertEqual(select_target_lot(z85, "ADAPTIVE_ENTRY", strong_zone_validated=False, cfg=self.cfg), 0.02)

        # 20: Strong Supply H4/H1 + score 94 + validated -> target lot 0.05
        z94_htf = _make_supply_zone(score=94.0, grade="A+", timeframe="H4+H1")
        sv_ok = validate_strong_zone(z94_htf, "ADAPTIVE_ENTRY", htf_aligned=True, has_htf_conflict=False, actual_rr=2.5, trigger_valid=True, cfg=self.cfg)
        self.assertTrue(sv_ok["passed"])
        self.assertEqual(select_target_lot(z94_htf, "ADAPTIVE_ENTRY", strong_zone_validated=sv_ok["passed"], cfg=self.cfg), 0.05)

        # 21: Score 94 tetapi zona M5 -> maksimal lot 0.02
        z94_m5 = _make_supply_zone(score=94.0, grade="A+", timeframe="M5")
        sv_m5 = validate_strong_zone(z94_m5, "ADAPTIVE_ENTRY", htf_aligned=True, has_htf_conflict=False, actual_rr=2.5, trigger_valid=True, cfg=self.cfg)
        self.assertFalse(sv_m5["passed"])
        self.assertEqual(select_target_lot(z94_m5, "ADAPTIVE_ENTRY", strong_zone_validated=sv_m5["passed"], cfg=self.cfg), 0.02)

        # 22: Score 94 di H1 tetapi Strong Zone Validation gagal (mis. displacement lemah 1.0 ATR) -> maksimal lot 0.02
        z94_weak_disp = _make_supply_zone(score=94.0, grade="A+", timeframe="H1", disp=1.0)
        z94_weak_disp.origin_bos = {}
        sv_fail = validate_strong_zone(z94_weak_disp, "ADAPTIVE_ENTRY", htf_aligned=True, has_htf_conflict=False, actual_rr=2.5, trigger_valid=True, cfg=self.cfg)
        self.assertFalse(sv_fail["passed"])
        self.assertEqual(select_target_lot(z94_weak_disp, "ADAPTIVE_ENTRY", strong_zone_validated=sv_fail["passed"], cfg=self.cfg), 0.02)

        # 23: EARLY_PENDING pada Strong Zone H4/H1 score 96 -> maksimal lot 0.01
        self.assertEqual(select_target_lot(z94_htf, "EARLY_PENDING", strong_zone_validated=True, cfg=self.cfg), 0.01)

        # Skenario 28 & 24, 25, 26, 12: Kalkulasi dalam mata uang IDC aktual!
        # Entry 3644.0, SL 3650.0 (jarak 6.0 USD + slippage 0.2 = 6.2 USD; + komisi $7/lot -> ~627 USD/lot = 9,405,000 IDC per 1.0 lot)
        # Maka:
        #   0.01 lot -> rugi ~94,050 IDC
        #   0.02 lot -> rugi ~188,100 IDC
        #   0.05 lot -> rugi ~470,250 IDC
        # Adaptive risk budget = 0.5% dari equity IDC!

        # 24: Equity = 50,000,000 IDC -> Budget 0.5% = 250,000 IDC (0.05 butuh 470k -> gagal, downgrade ke 0.02 = 188.1k <= 250k!)
        mt5_24, sm_24, _, _ = self._new_env(equity_idc=50_000_000.0)
        snap, scan, dec = _make_snapshot_and_scan(z94_htf, self.df_m5_rej)
        val_24 = validate_trade_candidate(z94_htf, mt5_24, sm_24, snapshot=snap, scan_result=scan, decision=dec, news_override=self.news_ok, cfg=self.cfg)
        self.assertEqual(val_24["account"]["currency"], "IDC")
        self.assertEqual(val_24["target_lot"], 0.05)
        self.assertEqual(val_24["final_lot"], 0.02)
        self.assertEqual(val_24["lot_reason"], "RISK_LIMIT_ADJUSTMENT")

        # 25: Equity = 25,000,000 IDC -> Budget 0.5% = 125,000 IDC (0.05 & 0.02 gagal -> downgrade ke 0.01 = 94,050 <= 125,000!)
        mt5_25, sm_25, _, _ = self._new_env(equity_idc=25_000_000.0)
        val_25 = validate_trade_candidate(z94_htf, mt5_25, sm_25, snapshot=snap, scan_result=scan, decision=dec, news_override=self.news_ok, cfg=self.cfg)
        self.assertEqual(val_25["target_lot"], 0.05)
        self.assertEqual(val_25["final_lot"], 0.01)
        self.assertEqual(val_25["lot_reason"], "RISK_LIMIT_ADJUSTMENT")

        # 12 & 26: Equity = 10,000,000 IDC -> Budget 0.5% = 50,000 IDC (0.01 butuh 94,050 IDC > 50,000 IDC -> BLOCK MIN_LOT_EXCEEDS_RISK!)
        mt5_26, sm_26, _, _ = self._new_env(equity_idc=10_000_000.0)
        val_26 = validate_trade_candidate(z94_htf, mt5_26, sm_26, snapshot=snap, scan_result=scan, decision=dec, news_override=self.news_ok, cfg=self.cfg)
        self.assertEqual(val_26["state"], STATE_BLOCKED)
        self.assertEqual(val_26["final_lot"], 0.0)
        self.assertEqual(val_26["reason_code"], REASON_MIN_LOT_EXCEEDS_RISK)

    # -------------------------------------------------------------------------
    # SKENARIO 13 & 38: Daily loss limit tercapai -> BLOCK, dan tetap BLOCK setelah restart!
    # -------------------------------------------------------------------------
    def test_13_38_daily_loss_limit_blocks_and_persists_across_restart(self):
        mt5, sm, _, _ = self._new_env(equity_idc=30_000_000.0)
        # Catat kerugian realized hari ini sebesar -650,000 IDC (> 2% dari 30,000,000 = 600,000 IDC)
        sm.record_realized_deal("DEAL_LOSS_1", -650_000.0, setup_id="OLD_S1", zone_id="OLD_Z1", hit_sl=True)

        zone = _make_supply_zone(score=92.0)
        snap, scan, dec = _make_snapshot_and_scan(zone, self.df_m5_rej)
        val = validate_trade_candidate(zone, mt5, sm, snapshot=snap, scan_result=scan, decision=dec, news_override=self.news_ok, cfg=self.cfg)
        self.assertEqual(val["state"], STATE_BLOCKED)
        self.assertEqual(val["reason_code"], REASON_DAILY_LOSS_LIMIT)

        # Skenario 38: Restart script -> counter daily loss TIDAK ter-reset, tetap BLOCK!
        sm_reloaded = ExecutionStateManager(state_file=self.state_file, auto_load=True)
        sm_reloaded.authorize_account(mt5.login, mt5.server, mt5.currency, "REAL", real_confirmed=True, balance=29_350_000.0, equity=29_350_000.0)
        sm_reloaded.set_trading_mode("FULL_AUTO", auto_trading=True)
        val_after = validate_trade_candidate(zone, mt5, sm_reloaded, snapshot=snap, scan_result=scan, decision=dec, news_override=self.news_ok, cfg=self.cfg)
        self.assertEqual(val_after["state"], STATE_BLOCKED)
        self.assertEqual(val_after["reason_code"], REASON_DAILY_LOSS_LIMIT)

    # -------------------------------------------------------------------------
    # SKENARIO 14 & 29: Real account tanpa konfirmasi eksplisit -> BLOCK; Akun berubah -> HALT!
    # -------------------------------------------------------------------------
    def test_14_29_real_account_confirmation_and_account_change_halt(self):
        # 14: Akun REAL belum dikonfirmasi eksplisit (`real_confirmed=False`)
        mt5, sm, _, _ = self._new_env(real_confirmed=False)
        zone = _make_supply_zone(score=90.0)
        snap, scan, dec = _make_snapshot_and_scan(zone, self.df_m5_rej)
        val = validate_trade_candidate(zone, mt5, sm, snapshot=snap, scan_result=scan, decision=dec, news_override=self.news_ok, cfg=self.cfg)
        self.assertEqual(val["state"], STATE_BLOCKED)
        self.assertEqual(val["reason_code"], REASON_REAL_ACCOUNT_UNCONFIRMED)

        # 29: Akun telah terotorisasi, kemudian login MT5 di terminal diganti user ke akun lain!
        mt5_ok, sm_ok, _, _ = self._new_env(real_confirmed=True)
        mt5_ok.login = 99998888  # Berubah!
        val_chg = validate_trade_candidate(zone, mt5_ok, sm_ok, snapshot=snap, scan_result=scan, decision=dec, news_override=self.news_ok, cfg=self.cfg)
        self.assertEqual(val_chg["state"], STATE_BLOCKED)
        self.assertEqual(val_chg["reason_code"], REASON_ACCOUNT_CHANGED)
        self.assertFalse(sm_ok.get_mode_info()["auto_trading"])  # Auto trading otomatis dimatikan!

    # -------------------------------------------------------------------------
    # SKENARIO 31 & 32: News data stale -> BLOCK; Trigger lama dari zona berbeda -> BLOCK / WAIT_TRIGGER
    # -------------------------------------------------------------------------
    def test_31_32_stale_news_and_old_trigger_from_different_zone_blocked(self):
        mt5, sm, _, _ = self._new_env()
        zone = _make_supply_zone(score=90.0)
        snap, scan, dec = _make_snapshot_and_scan(zone, self.df_m5_rej)

        # 31: News data STALE
        val_news = validate_trade_candidate(
            zone, mt5, sm, snapshot=snap, scan_result=scan, decision=dec,
            news_override={"blackout": False, "status": "STALE", "age_seconds": 50000.0}, cfg=self.cfg,
        )
        self.assertEqual(val_news["state"], STATE_BLOCKED)
        self.assertEqual(val_news["reason_code"], REASON_STALE_NEWS_DATA)

        # 32: Trigger rejection lama berada di area 3643-3647, sekarang harga hanya menyentuh tipis zona BARU di 3655-3659 dengan candle kecil biasa!
        df_other = self.df_m5_rej.copy()
        new_ts = df_other.index[-1] + pd.Timedelta(minutes=5)
        # Candle baru menyentuh zona tinggi 3655.0-3659.0 tetapi hanya candle bullish/doji kecil tanpa rejection valid di zona baru
        df_other.loc[new_ts] = {"Open": 3654.8, "High": 3655.3, "Low": 3654.6, "Close": 3655.2, "Volume": 120,
                                "EMA20": 3645.0, "EMA50": 3640.0, "EMA200": 3635.0, "RSI14": 60.0, "ATR14": 2.0}
        zone_high_new = _make_supply_zone(score=90.0, lo=3655.0, hi=3659.0)
        mt5.bid, mt5.ask = 3655.1, 3655.3
        snap2, scan2, dec2 = _make_snapshot_and_scan(zone_high_new, df_other, bid=3655.1, ask=3655.3)
        dec2["plan"]["sl"] = 3662.0
        dec2["plan"]["tp1"] = 3635.0

        val_old_trig = validate_trade_candidate(
            zone_high_new, mt5, sm, snapshot=snap2, scan_result=scan2, decision=dec2, news_override=self.news_ok, cfg=self.cfg
        )
        # Trigger lama dari zona 3643-3647 TIDAK BOLEH dipakai untuk zona 3655-3659!
        self.assertFalse(val_old_trig["can_execute"])
        self.assertEqual(val_old_trig["reason_code"], "WAIT_TRIGGER")

    # -------------------------------------------------------------------------
    # SKENARIO 33 & 35: SL gagal terpasang -> Emergency Protection; Posisi manual tidak disentuh!
    # -------------------------------------------------------------------------
    def test_33_35_emergency_sl_protection_and_manual_position_protection(self):
        mt5, sm, _, eng = self._new_env(equity_idc=60_000_000.0)
        pos_mgr = PositionManager(eng, sm, cfg=self.cfg)

        # Skenario 35: Tambahkan posisi MANUAL milik user (magic=0) -> exposure manual (300,000 IDC = 0.50% dari 60jt) dihitung,
        # tetapi posisi manual TIDAK PERNAH dimodifikasi/ditutup!
        mt5.positions.append(SimpleNamespace(
            ticket=11111, symbol="XAUUSDc", type=mt5.ORDER_TYPE_BUY, volume=0.02,
            price_open=3640.0, sl=3630.0, tp=3660.0, profit=-10000.0, swap=0.0, commission=0.0,
            magic=0, comment="Manual trade user", time=int(time.time()),
        ))
        port = evaluate_portfolio_risk_and_daily_limits(mt5, sm, symbol="XAUUSDc", cfg=self.cfg)
        self.assertEqual(port["manual_positions_count"], 1)
        self.assertGreater(port["manual_open_risk_nominal"], 0.0)  # Exposure manual diperhitungkan!

        # Skenario 33a: Posisi scanner terbuka tanpa SL -> PositionManager otomatis memasang SL struktural darurat!
        mt5.simulate_missing_sl_on_fill = True
        zone = _make_supply_zone(score=85.0, timeframe="H1")
        snap, scan, dec = _make_snapshot_and_scan(zone, self.df_m5_rej)
        val = validate_trade_candidate(zone, mt5, sm, snapshot=snap, scan_result=scan, decision=dec, news_override=self.news_ok, cfg=self.cfg)
        ex = eng.execute_validated_setup(val, trading_mode="FULL_AUTO")
        self.assertTrue(ex["executed"])
        sc_pos = next(p for p in mt5.positions if p.magic == SCANNER_MAGIC_NUMBER)
        self.assertEqual(sc_pos.sl, 0.0)  # Awalnya gagal terpasang

        audit1 = pos_mgr.audit_and_manage_positions(symbol="XAUUSDc")
        self.assertEqual(audit1["emergency_actions"][0]["action"], "EMERGENCY_SL_ATTACHED")
        self.assertAlmostEqual(sc_pos.sl, 3650.0, places=2)
        # Pastikan posisi manual #11111 sama sekali tidak berubah!
        man_pos = next(p for p in mt5.positions if p.ticket == 11111)
        self.assertEqual(man_pos.sl, 3630.0)

        # Skenario 33b: Jika SL hilang DAN broker menolak modifikasi SL -> tutup posisi scanner secara darurat!
        sc_pos.sl = 0.0
        mt5.reject_sl_modify = True
        audit2 = pos_mgr.audit_and_manage_positions(symbol="XAUUSDc")
        self.assertEqual(audit2["emergency_actions"][0]["action"], "EMERGENCY_POSITION_CLOSED")
        # Posisi scanner tertutup, sedangkan posisi manual #11111 tetap utuh!
        self.assertEqual([p.ticket for p in mt5.positions], [11111])

    # -------------------------------------------------------------------------
    # SKENARIO 34: Dua monitoring loop mendeteksi setup sama secara paralel -> hanya 1 eksekusi!
    # -------------------------------------------------------------------------
    def test_34_concurrent_monitoring_loops_execute_only_once(self):
        mt5, sm, _, eng = self._new_env(equity_idc=50_000_000.0)
        zone = _make_supply_zone(score=90.0)
        snap, scan, dec = _make_snapshot_and_scan(zone, self.df_m5_rej)
        val = validate_trade_candidate(zone, mt5, sm, snapshot=snap, scan_result=scan, decision=dec, news_override=self.news_ok, cfg=self.cfg)

        barrier = threading.Barrier(2)
        outcomes = []

        def _worker():
            barrier.wait()
            res = eng.execute_validated_setup(dict(val), trading_mode="FULL_AUTO")
            outcomes.append(res)

        t1 = threading.Thread(target=_worker)
        t2 = threading.Thread(target=_worker)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        executed_cnt = sum(1 for r in outcomes if r.get("executed"))
        self.assertEqual(executed_cnt, 1)
        self.assertEqual(len(mt5.positions), 1)

    # -------------------------------------------------------------------------
    # SKENARIO 36 & 37: Spread melonjak sebelum order_send -> BLOCK; Volume step invalid -> BLOCK
    # -------------------------------------------------------------------------
    def test_36_37_pre_send_spread_spike_and_invalid_volume_step_blocked(self):
        # 36: Saat validasi awal spread normal 0.20, tepat sebelum order_send spread melonjak ke 2.50 USD!
        mt5, sm, _, eng = self._new_env()
        zone = _make_supply_zone(score=90.0)
        snap, scan, dec = _make_snapshot_and_scan(zone, self.df_m5_rej)
        val = validate_trade_candidate(zone, mt5, sm, snapshot=snap, scan_result=scan, decision=dec, news_override=self.news_ok, cfg=self.cfg)
        self.assertTrue(val["can_execute"])

        mt5.pre_send_spread_spike = 2.50  # Melebihi SPREAD_MAKS_POIN (1.0 USD)
        ex = eng.execute_validated_setup(val, trading_mode="FULL_AUTO")
        self.assertFalse(ex["executed"])
        self.assertEqual(ex["reason_code"], REASON_SPREAD_TOO_HIGH)
        self.assertEqual(len(mt5.send_calls), 0)

        # 37: Broker mengharuskan volume_step = 0.10 dan volume_min = 0.10 -> lot 0.01/0.02/0.05 ditolak tanpa kirim order!
        mt5_v, sm_v, _, _ = self._new_env(equity_idc=100_000_000.0)
        mt5_v.volume_min = 0.10
        mt5_v.volume_step = 0.10
        val_v = validate_trade_candidate(zone, mt5_v, sm_v, snapshot=snap, scan_result=scan, decision=dec, news_override=self.news_ok, cfg=self.cfg)
        self.assertEqual(val_v["state"], STATE_BLOCKED)
        self.assertEqual(val_v["reason_code"], REASON_INVALID_VOLUME)

    # -------------------------------------------------------------------------
    # CLI COMMANDS & PANTAU PANEL INTEGRATION TEST
    # -------------------------------------------------------------------------
    def test_cli_auto_commands_and_pantau_panel(self):
        mt5, sm, jrn, _ = self._new_env(real_confirmed=False)
        mon = AutoExecutionMonitor(
            mt5_api=mt5, state_mgr=sm, journal=jrn, symbol="XAUUSDc", cfg=self.cfg, connect_fn=lambda: True
        )

        # auto real dengan konfirmasi eksplisit 'YES'
        r_real = handle_auto_cli_command("auto real", input_fn=lambda _: "YES", monitor=mon)
        self.assertTrue(r_real["ok"])
        self.assertTrue(sm.get_mode_info()["authorized_account"]["real_confirmed"])
        mon.stop_background()

        # auto risk, auto positions, auto orders, auto journal, auto dryrun, auto off, auto emergency
        for sub in ("risk", "positions", "orders", "journal", "dryrun", "semi", "off"):
            r_sub = handle_auto_cli_command(f"auto {sub}", input_fn=lambda _: "", monitor=mon)
            self.assertTrue(r_sub["ok"], msg=f"Command 'auto {sub}' gagal: {r_sub}")
        mon.stop_background()

        r_emg = handle_auto_cli_command("auto emergency", input_fn=lambda _: "", monitor=mon)
        self.assertTrue(r_emg["ok"])
        self.assertTrue(sm.get_mode_info()["emergency_halt"])

        # Format Pantau Panel (Section 18)
        zone = _make_supply_zone(score=94.0, timeframe="H4+H1")
        snap, scan, dec = _make_snapshot_and_scan(zone, self.df_m5_rej)
        sm.set_trading_mode("DRY_RUN", auto_trading=True, clear_emergency=True)
        cyc = mon.run_once(snapshot=snap, scan_result=scan, decision=dec, news_override=self.news_ok)
        panel_txt = format_adaptive_execution_panel(cyc, state_mgr=sm)
        self.assertIn("XAUUSD SMC ADAPTIVE AUTO TRADING", panel_txt)
        self.assertIn("CURRENCY     : IDC", panel_txt)
        self.assertIn("TRIGGER      : VALID REJECTION", panel_txt)
        self.assertIn("DECISION     : ENTRY_READY", panel_txt)

    # -------------------------------------------------------------------------
    # NO-LOOKAHEAD BACKTEST ENGINE TEST
    # -------------------------------------------------------------------------
    def test_no_lookahead_backtest_comparison_runs_cleanly(self):
        # Bangun histori multi-timeframe sintetis realistis tanpa NaN untuk menguji pipeline backtest kausal
        rng = np.random.default_rng(42)
        n_m5 = 180
        end_ts = pd.Timestamp("2026-11-24 16:00:00")
        idx_m5 = pd.date_range(end=end_ts, periods=n_m5, freq="5min")
        # Gelombang tren turun + pullback ke supply agar menghasilkan OB, FVG, touch, rejection, & CHoCH nyata
        t = np.linspace(0, 8 * np.pi, n_m5)
        mid_series = 3680.0 - np.linspace(0, 35.0, n_m5) + 6.5 * np.sin(t) + rng.normal(0, 0.35, n_m5)
        o5 = mid_series + rng.normal(0, 0.25, n_m5)
        c5 = np.roll(o5, -1)
        c5[-1] = o5[-1] - 0.5
        h5 = np.maximum(o5, c5) + np.abs(rng.normal(0.8, 0.4, n_m5))
        l5 = np.minimum(o5, c5) - np.abs(rng.normal(0.8, 0.4, n_m5))
        df_m5 = pd.DataFrame({"Open": o5, "High": h5, "Low": l5, "Close": c5, "Volume": 200}, index=idx_m5)
        df_m5.attrs["time_tz"] = "UTC"

        def _resample_ohlc(df_src: pd.DataFrame, rule: str, pad_bars: int = 120) -> pd.DataFrame:
            rs = df_src.resample(rule).agg({
                "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"
            }).dropna()
            if len(rs) < pad_bars:
                need_p = pad_bars - len(rs)
                step_td = pd.Timedelta(rule)
                pre_idx = pd.date_range(end=rs.index[0] - step_td, periods=need_p, freq=rule)
                p0 = float(rs["Open"].iloc[0])
                pre_p = p0 + np.linspace(15.0, 0.5, need_p) + 2.5 * np.sin(np.linspace(0, 4 * np.pi, need_p))
                pre_df = pd.DataFrame({
                    "Open": pre_p + 0.3, "High": pre_p + 1.5, "Low": pre_p - 1.5, "Close": pre_p - 0.3, "Volume": 500
                }, index=pre_idx)
                rs = pd.concat([pre_df, rs])
            rs.attrs["time_tz"] = "UTC"
            return tambah_indikator(rs)

        frames = {
            "M5": tambah_indikator(df_m5),
            "M15": _resample_ohlc(df_m5, "15min", 120),
            "M30": _resample_ohlc(df_m5, "30min", 120),
            "H1": _resample_ohlc(df_m5, "1h", 120),
            "H4": _resample_ohlc(df_m5, "4h", 120),
        }

        comp = run_comprehensive_backtest_comparison(frames, initial_equity=10000.0)
        self.assertIn("entry_modes", comp)
        self.assertIn("ADAPTIVE_ENTRY", comp["entry_modes"])
        self.assertIn("CONFIRMED_ENTRY", comp["entry_modes"])
        self.assertIn("EARLY_PENDING", comp["entry_modes"])
        self.assertIn("score_thresholds", comp)
        self.assertIn("lot_sizing", comp)

    # -------------------------------------------------------------------------
    # AUDIT REQUIREMENTS: NO-SL POSITION BLOCK, EARLY PENDING >=75, CONFIRMED STRICT,
    # REQUOTE RETRY WITHOUT DUPLICATE, HYBRID SMC+SNR FALLBACK, & DEMO ACCOUNT FLOW
    # -------------------------------------------------------------------------
    def test_audit_unprotected_no_sl_blocks_new_orders(self):
        mt5, sm, _, _ = self._new_env(equity_idc=60_000_000.0)
        # Posisi manual tanpa SL (unbounded risk) -> wajib blokir order baru!
        mt5.positions.append(SimpleNamespace(
            ticket=22222, symbol="XAUUSDc", type=mt5.ORDER_TYPE_BUY, volume=0.01,
            price_open=3640.0, sl=0.0, tp=0.0, profit=0.0, swap=0.0, commission=0.0,
            magic=0, comment="Manual without SL", time=int(time.time()),
        ))
        zone = _make_supply_zone(score=92.0)
        snap, scan, dec = _make_snapshot_and_scan(zone, self.df_m5_rej)
        val = validate_trade_candidate(zone, mt5, sm, snapshot=snap, scan_result=scan, decision=dec, news_override=self.news_ok, cfg=self.cfg)
        self.assertEqual(val["state"], STATE_BLOCKED)
        self.assertEqual(val["reason_code"], "UNPROTECTED_POSITION_NO_SL")

    def test_audit_early_pending_requires_grade_a_75_and_confirmed_requires_sweep(self):
        mt5, sm, _, _ = self._new_env(bid=3630.0, ask=3630.2)
        df_untouched = self.df_m5_rej.iloc[:-15].copy()
        df_untouched.attrs["is_closed"] = True
        # Score 72 (< 75) -> EARLY_PENDING tidak aktif (tunggu sentuhan zona)
        z72 = _make_supply_zone(score=72.0, grade="B", timeframe="H1", lo=3643.0, hi=3647.0)
        snap, scan, dec = _make_snapshot_and_scan(z72, df_untouched, bid=3630.0, ask=3630.2)
        val_ep = validate_trade_candidate(
            z72, mt5, sm, snapshot=snap, scan_result=scan, decision=dec,
            preferred_entry_mode="EARLY_PENDING", news_override=self.news_ok, cfg=self.cfg,
        )
        self.assertFalse(val_ep["can_execute"])

        # CONFIRMED_ENTRY dengan rejection saja (tanpa Sweep + CHoCH/CISD + Displacement) -> WAIT_TRIGGER
        mt5.bid, mt5.ask = 3644.0, 3644.2
        z90 = _make_supply_zone(score=90.0, grade="A+", timeframe="H1")
        snap_c, scan_c, dec_c = _make_snapshot_and_scan(z90, self.df_m5_rej)
        val_cf = validate_trade_candidate(
            z90, mt5, sm, snapshot=snap_c, scan_result=scan_c, decision=dec_c,
            preferred_entry_mode="CONFIRMED_ENTRY", news_override=self.news_ok, cfg=self.cfg,
        )
        self.assertFalse(val_cf["can_execute"])
        self.assertEqual(val_cf["entry_mode"], "CONFIRMED_ENTRY")

    def test_audit_bounded_retry_on_requote_without_duplicate_and_hybrid_snr_replacement(self):
        mt5, sm, jrn, eng = self._new_env(equity_idc=50_000_000.0)
        orig_send = mt5.order_send
        call_counter = {"n": 0}

        def _requote_once_send(req):
            if req.get("action") == mt5.TRADE_ACTION_DEAL and not req.get("position"):
                call_counter["n"] += 1
                if call_counter["n"] == 1:
                    return SimpleNamespace(retcode=10004, order=0, deal=0, price=0.0, volume=0.0, comment="Requote")
            return orig_send(req)

        mt5.order_send = _requote_once_send
        zone = _make_supply_zone(score=90.0)
        snap, scan, dec = _make_snapshot_and_scan(zone, self.df_m5_rej)
        val = validate_trade_candidate(zone, mt5, sm, snapshot=snap, scan_result=scan, decision=dec, news_override=self.news_ok, cfg=self.cfg)
        ex = eng.execute_validated_setup(val, trading_mode="FULL_AUTO")
        self.assertTrue(ex["executed"])
        self.assertEqual(ex["attempts"], 2)
        self.assertEqual(len(mt5.positions), 1)

        # Hybrid SMC + SNR: SMC zona invalid/none, SNR punya resistance aktif tervalidasi -> AutoExecutionMonitor otomatis mengambil alih!
        mt5_h = MockMT5(trade_mode=MockMT5.ACCOUNT_TRADE_MODE_DEMO, currency="USD", balance=5000.0, equity=5000.0, bid=3644.0, ask=3644.2)
        sm_h = ExecutionStateManager(state_file=os.path.join(self.tmpdir.name, "state_hybrid.json"), auto_load=False)
        mon_h = AutoExecutionMonitor(mt5_api=mt5_h, state_mgr=sm_h, journal=jrn, symbol="XAUUSDc", cfg=self.cfg, connect_fn=lambda: True)
        # Aktivasi lewat CLI `auto demo` (uji akun demo sebelum akun real)
        r_demo = handle_auto_cli_command("auto demo", input_fn=None, monitor=mon_h)
        self.assertTrue(r_demo["ok"])

        bad_smc = _make_supply_zone(score=90.0, is_valid=False, freshness="INVALIDATED")
        snap_h, scan_h, dec_h = _make_snapshot_and_scan(bad_smc, self.df_m5_rej)
        scan_h["active_zones"] = []
        dec_h["zone"] = None
        snr_h = {
            "scan_result": {
                "symbol": "XAUUSDc",
                "resistances": [{
                    "zone_id": "SNR_H1_RES_20261124_3643",
                    "zone_type": "RESISTANCE",
                    "timeframe": "H1",
                    "low": 3643.0,
                    "high": 3647.0,
                    "score": 84.0,
                    "grade": "A",
                    "lifecycle_status": "ACTIVE",
                    "confirmed_time": "2026-11-24 08:00:00",
                    "touch_count": 0,
                }],
            },
            "decision": {},
        }
        cyc_h = mon_h.run_once(snapshot=snap_h, scan_result=scan_h, decision=dec_h, snr_result=snr_h, news_override=self.news_ok)
        self.assertTrue(cyc_h["execution_outcome"]["executed"])
        self.assertEqual(len(mt5_h.positions), 1)
        self.assertEqual(cyc_h["best_evaluation"]["zone_id"], "SNR_H1_RES_20261124_3643")

    def test_env_max_lot_001_cap_enforced(self):
        cfg_cap = {**get_auto_trading_config(), "MAX_LOT": 0.01}
        z94 = _make_supply_zone(score=94.0, grade="A+", timeframe="H4+H1")
        self.assertEqual(select_target_lot(z94, "ADAPTIVE_ENTRY", strong_zone_validated=True, cfg=cfg_cap), 0.01)

    def test_live_1to1_structural_sl_tp_and_wide_zone_execution_not_blocked(self):
        cfg_live = get_auto_trading_config()
        self.assertTrue(cfg_live["FORCE_FIXED_TP1"])
        self.assertEqual(cfg_live["TP1_RR"], 1.0)
        self.assertEqual(cfg_live["MIN_RR"], 1.0)

        # Akun Real IDC 18.280.000 IDC:
        # 1) Dengan konfigurasi ketat (SL_JARAK_MAKS=12.0), zona lebar (SL 26.34 USD) wajib di-skip (REASON_INVALID_STOP_LOSS)
        # 2) Jika SL_JARAK_MAKS dilonggarkan secara eksplisit (35.0), TP1 1:1 dihitung presisi sesuai jarak SL struktural
        mt5, sm, jrn, eng = self._new_env(equity_idc=18_280_000.0, bid=3520.0, ask=3520.2)
        wide_zone = Zone(
            zone_id="SNR_H4_SUP_WIDE_3502",
            symbol="XAUUSDc",
            timeframe="H4",
            zone_type="DEMAND",
            price_low=3502.66,
            price_high=3521.00,
            midpoint=3511.83,
            score=82.0,
            grade="A",
            origin_bos={"tipe": "SNR_STRUCTURE", "arah": "bull", "origin_valid": True},
            fvg_confirmation=None,
            displacement_strength=1.8,
            freshness="FRESH",
            touch_count=0,
            liquidity_sweep_status="CONFIRMED_SWEEP_LOW",
            invalidation_price=3502.66,
            is_valid=True,
            rejection_reason="",
            sl_recommended=3493.86,   # Jarak SL dari Ask 3520.20 = 26.34 USD!
            tp1_recommended=3580.00,  # TP struktural lama jauh (tidak 1:1), harus di-override jadi 1:1!
        )
        # Buat M5 bullish rejection di dalam zona
        idx = pd.date_range("2026-11-24 10:00:00", periods=40, freq="5min")
        base = np.linspace(3535.0, 3521.5, 40)
        df_bull = pd.DataFrame({
            "Open": base + 0.3, "High": base + 1.0, "Low": base - 0.8, "Close": base - 0.1, "Volume": 400
        }, index=idx)
        # Bar closed terakhir (-2) membuat bullish pinbar rejection kuat di dalam zona
        df_bull.iloc[-2, df_bull.columns.get_loc("Open")] = 3520.2
        df_bull.iloc[-2, df_bull.columns.get_loc("Low")] = 3517.0
        df_bull.iloc[-2, df_bull.columns.get_loc("High")] = 3520.9
        df_bull.iloc[-2, df_bull.columns.get_loc("Close")] = 3520.7
        df_bull = tambah_indikator(df_bull)

        snap, scan, dec = _make_snapshot_and_scan(
            wide_zone, df_bull, bid=3520.0, ask=3520.2, h4_bias="BULLISH", h1_bias="BULLISH", m30_status="CONFIRMED"
        )
        dec["plan"]["sl"] = 3493.86
        dec["plan"]["tp1"] = 3580.00

        cfg_strict = {**cfg_live, "SL_JARAK_MAKS": 12.0}
        val_strict = validate_trade_candidate(
            wide_zone, mt5, sm, snapshot=snap, scan_result=scan, decision=dec,
            news_override=self.news_ok, cfg=cfg_strict,
        )
        self.assertEqual(val_strict["reason_code"], REASON_INVALID_STOP_LOSS)

        cfg_wide = {**cfg_live, "SL_JARAK_MAKS": 35.0}
        eng.cfg = cfg_wide
        val = validate_trade_candidate(
            wide_zone, mt5, sm, snapshot=snap, scan_result=scan, decision=dec,
            news_override=self.news_ok, cfg=cfg_wide,
        )
        self.assertEqual(val["state"], STATE_ENTRY_READY, msg=f"Harus ENTRY_READY, bukan {val.get('reason_code')}: {val.get('reason_text')}")
        self.assertEqual(val["final_lot"], 0.01)
        self.assertAlmostEqual(val["actual_rr"], 1.0, places=2)

        ex = eng.execute_validated_setup(val, trading_mode="FULL_AUTO")
        self.assertTrue(ex["executed"])
        pos = mt5.positions[0]
        # Pastikan SL struktural tetap di luar zona (3493.86) dan TP1 = Entry + (Entry - SL) = 1:1 presisi!
        sl_dist = round(pos.price_open - pos.sl, 2)
        tp_dist = round(pos.tp - pos.price_open, 2)
        self.assertAlmostEqual(sl_dist, 26.34, places=2)
        self.assertAlmostEqual(tp_dist, sl_dist, places=2)

    def test_one_shot_invalid_zone_displayed_once_then_hidden(self):
        from unittest.mock import patch
        import modules.utils as u
        old_ready = u._INVALID_ZONES_BASELINE_READY
        old_dismissed = list(u._DISMISSED_INVALID_ZONES)
        old_ck = u._LAST_INVALID_ONCE_CYCLE_KEY
        old_lz = u._LAST_INVALID_ONCE_ZONE
        try:
            u._INVALID_ZONES_BASELINE_READY = False
            u._DISMISSED_INVALID_ZONES.clear()
            u._LAST_INVALID_ONCE_CYCLE_KEY = None
            u._LAST_INVALID_ONCE_ZONE = None
            with patch.object(u, "_save_trade_plan_state", lambda: None), \
                 patch.object(u, "_load_trade_plan_state_once", lambda: None):
                hist_inv = {"zone_id": "OLD_INV_1", "timeframe": "M30", "zone_type": "SUPPLY", "price_low": 3558.37, "price_high": 3560.79}
                # Siklus awal: baseline zona invalid lama langsung diabaikan (tidak dimunculkan)
                self.assertIsNone(u.consume_one_shot_invalid_zone([hist_inv], current_price=3575.0, cycle_key="c1"))

                # Siklus #2: ada zona baru terkena SL -> tampilkan INVALID tepat 1 kali pada siklus #2
                new_sl_hit = {"zone_id": "NEW_SL_HIT_2", "timeframe": "M30", "zone_type": "SUPPLY", "price_low": 3586.0, "price_high": 3589.0, "invalidation_type": "SETUP_INVALID_SL_BREACHED"}
                res_c2_a = u.consume_one_shot_invalid_zone([hist_inv, new_sl_hit], current_price=3590.0, cycle_key="c2")
                res_c2_b = u.consume_one_shot_invalid_zone([hist_inv, new_sl_hit], current_price=3590.0, cycle_key="c2")
                self.assertIsNotNone(res_c2_a)
                self.assertEqual(res_c2_a["zone_id"], "NEW_SL_HIT_2")
                self.assertEqual(res_c2_b["zone_id"], "NEW_SL_HIT_2")

                # Siklus #3 dan seterusnya: zona invalid tadi langsung hilang dan tidak pernah ditampilkan lagi
                self.assertIsNone(u.consume_one_shot_invalid_zone([hist_inv, new_sl_hit], current_price=3590.0, cycle_key="c3"))
                self.assertIsNone(u.consume_one_shot_invalid_zone([hist_inv, new_sl_hit], current_price=3587.0, cycle_key="c4"))
        finally:
            u._INVALID_ZONES_BASELINE_READY = old_ready
            u._DISMISSED_INVALID_ZONES[:] = old_dismissed
            u._LAST_INVALID_ONCE_CYCLE_KEY = old_ck
            u._LAST_INVALID_ONCE_ZONE = old_lz

    def test_conflict_midzone_50pct_limit_sl_tp_outside_zone(self):
        # Skenario live user: Market HTF_CONFLICT (H4=BEARISH, H1=BULLISH, M30=CONFLICT),
        # harga di 3602.31/3602.51, ada zona [SNR] M30 RESISTANCE 3625.18-3632.18 (Grade B / 71.8)
        cfg_mz = {
            **get_auto_trading_config(),
            "SNR_ZONE_MIN_STABLE_SECONDS": 0.0,
            "CONFLICT_MIDZONE_LIMIT_ENABLED": True,
            "CONFLICT_MIDZONE_MIN_SCORE": 70.0,
            "CONFLICT_MIDZONE_MAX_WIDTH": 16.0,
            "CONFLICT_MIDZONE_SL_BUFFER": 0.60,
            "SL_JARAK_MAKS": 12.0,
            "MIN_RR": 1.0,
            "TP1_RR": 1.0,
            "TP1_MODE": "FIXED_RR",
            "FORCE_FIXED_TP1": True,
            "REAL_LOT_001_FIXED_TP_ENABLED": False,
        }
        mt5, sm, jrn, eng = self._new_env(equity_idc=18_280_000.0, bid=3602.31, ask=3602.51)
        mon = AutoExecutionMonitor(mt5_api=mt5, state_mgr=sm, journal=jrn, symbol="XAUUSDc", cfg=cfg_mz, connect_fn=lambda: True)

        df_m5 = self.df_m5_rej.iloc[:-5].copy()
        snap = {
            "symbol": "XAUUSDc",
            "bid": 3602.31,
            "ask": 3602.51,
            "mid": 3602.41,
            "current_price": 3602.41,
            "spread": 0.20,
            "market_status": "OPEN",
            "data_freshness": "FRESH",
            "tick_age": 1.0,
            "frames": {"M5": df_m5},
        }
        scan_empty_smc = {
            "symbol": "XAUUSDc",
            "current_price": 3602.41,
            "bid": 3602.31,
            "ask": 3602.51,
            "spread": 0.20,
            "market_status": "OPEN",
            "data_freshness": "FRESH",
            "htf_bias": "BEARISH",
            "h1_structure": "BULLISH",
            "m30_status": "CONFLICT",
            "htf_context": {"H4": "BEARISH", "H1": "BULLISH", "m30_status": "CONFLICT", "has_conflict": True},
            "active_zones": [],
            "radar_zones": [],
        }
        snr_res = {
            "scan_result": {
                "symbol": "XAUUSDc",
                "ranked_zones": {
                    "active_supports": [],
                    "active_resistances": [{
                        "zone_id": "SNR_M30_RES_362518_363218",
                        "timeframe": "M30",
                        "zone_type": "RESISTANCE",
                        "low": 3625.18,
                        "high": 3632.18,
                        "score": 71.8,
                        "grade": "B",
                        "freshness": "FRESH",
                        "touch_count": 0,
                        "invalidation_price": 3632.18,
                        "is_active": True,
                    }],
                },
            },
            "decision": {"signal": "WAIT"},
        }

        cyc = mon.run_once(snapshot=snap, scan_result=scan_empty_smc, decision={}, snr_result=snr_res, news_override=self.news_ok)
        self.assertTrue(cyc["execution_outcome"]["executed"], msg=str(cyc.get("best_evaluation")))
        self.assertEqual(len(mt5.orders), 1)
        ord_placed = mt5.orders[0]

        # 1. Entry wajib tepat di 50% tengah zona: (3625.18 + 3632.18) / 2 = 3628.68 (bukan 3625.18, bukan 3632.18)
        self.assertAlmostEqual(ord_placed.price_open, 3628.68, places=2)
        # 2. SL tipis tepat di luar ujung atas zona (3632.18 + 0.60 = 3632.78)
        self.assertAlmostEqual(ord_placed.sl, 3632.78, places=2)
        # 3. TP tipis tepat di luar ujung bawah zona (3625.18 - 0.60 = 3624.58)
        self.assertAlmostEqual(ord_placed.tp, 3624.58, places=2)

        # 4. Siklus berikutnya saat masih HTF_CONFLICT -> pending order TIDAK dibatalkan!
        cyc_next = mon.run_once(snapshot=snap, scan_result=scan_empty_smc, decision={}, snr_result=snr_res, news_override=self.news_ok)
        self.assertEqual(len(mt5.orders), 1)

    def test_stepwise_partial_close_002_and_003_lot_with_auto_be_plus(self):
        from modules.position_manager import PositionManager

        cfg_p = {
            **self.cfg,
            "PARTIAL_CLOSE_ENABLED": True,
            "PARTIAL_STEP_POINTS": 5.0,       # 50 pips = 5.00 USD
            "PARTIAL_CLOSE_LOT_STEP": 0.01,
            "PARTIAL_MIN_REMAINING_LOT": 0.01,
            "BE_AFTER_PARTIAL_ENABLED": True,
            "BE_PLUS_LOCK_POINTS": 0.30,
        }
        mt5, sm, jrn, eng = self._new_env(bid=3600.0, ask=3600.2)
        eng.cfg = cfg_p
        pm = PositionManager(exec_engine=eng, state_mgr=sm, cfg=cfg_p)
        magic = int(cfg_p["SCANNER_MAGIC_NUMBER"])

        # A) SKENARIO 0.02 LOT BUY @ 3600.00 (SL 3592.00, TP1 3612.00) + posisi manual user 0.03 lot (magic=0)
        mt5.positions = [
            SimpleNamespace(ticket=501, symbol="XAUUSDc", type=mt5.ORDER_TYPE_BUY, volume=0.02, price_open=3600.00, sl=3592.00, tp=3612.00, profit=0.0, swap=0.0, commission=0.0, magic=magic, comment="SCAN_002", time=1000),
            SimpleNamespace(ticket=999, symbol="XAUUSDc", type=mt5.ORDER_TYPE_BUY, volume=0.03, price_open=3600.00, sl=3590.00, tp=3620.00, profit=0.0, swap=0.0, commission=0.0, magic=0, comment="MANUAL_USER", time=1000),
        ]
        sm.update_setup_state("SETUP_BUY_002", "POSITION_ACTIVE", position_ticket=501, actual_entry=3600.00, sl=3592.00, tp=3612.00, final_lot=0.02, actual_lot=0.02, extra={"planned_sl": 3592.00, "initial_lot": 0.02})

        # Running +4.50 USD (45 pips) -> belum partial, SL masih 3592.00
        mt5.bid, mt5.ask = 3604.50, 3604.70
        res_45 = pm.audit_and_manage_positions("XAUUSDc")
        self.assertEqual(len(res_45["partial_actions"]), 0)
        self.assertAlmostEqual(mt5.positions[0].volume, 0.02, places=2)
        self.assertAlmostEqual(mt5.positions[0].sl, 3592.00, places=2)

        # Running +5.00 USD (50 pips) -> partial 0.01 lot (sisa 0.01 ke TP1) + SL langsung pindah ke BE+ (3600.30)!
        mt5.bid, mt5.ask = 3605.00, 3605.20
        res_50 = pm.audit_and_manage_positions("XAUUSDc")
        self.assertEqual(len(res_50["partial_actions"]), 1)
        self.assertAlmostEqual(mt5.positions[0].volume, 0.01, places=2)
        self.assertAlmostEqual(mt5.positions[0].sl, 3600.30, places=2)
        self.assertAlmostEqual(mt5.positions[0].tp, 3612.00, places=2)
        # Posisi manual user (ticket 999) sama sekali tidak disentuh!
        self.assertAlmostEqual(mt5.positions[1].volume, 0.03, places=2)
        self.assertAlmostEqual(mt5.positions[1].sl, 3590.00, places=2)

        # Running +10.50 USD -> sisa 0.01 lot TIDAK di-partial lagi, tetap dibiarkan sampai TP1!
        mt5.bid, mt5.ask = 3610.50, 3610.70
        res_105 = pm.audit_and_manage_positions("XAUUSDc")
        self.assertEqual(len(res_105["partial_actions"]), 0)
        self.assertAlmostEqual(mt5.positions[0].volume, 0.01, places=2)

        # B) SKENARIO 0.03 LOT SELL @ 3650.00 (SL 3658.00, TP1 3635.00) -> berturut-turut +50 pips & +100 pips!
        mt5.positions = [
            SimpleNamespace(ticket=601, symbol="XAUUSDc", type=mt5.ORDER_TYPE_SELL, volume=0.03, price_open=3650.00, sl=3658.00, tp=3635.00, profit=0.0, swap=0.0, commission=0.0, magic=magic, comment="SCAN_003", time=2000),
        ]
        sm.update_setup_state("SETUP_SELL_003", "POSITION_ACTIVE", position_ticket=601, actual_entry=3650.00, sl=3658.00, tp=3635.00, final_lot=0.03, actual_lot=0.03, extra={"planned_sl": 3658.00, "initial_lot": 0.03})

        # Tahap 1: Running +50 pips SELL (Ask turun ke 3645.00) -> tutup 0.01 (sisa 0.02) + BE+ di 3649.70!
        mt5.bid, mt5.ask = 3644.80, 3645.00
        r_s1 = pm.audit_and_manage_positions("XAUUSDc")
        self.assertEqual(len(r_s1["partial_actions"]), 1)
        self.assertAlmostEqual(mt5.positions[0].volume, 0.02, places=2)
        self.assertAlmostEqual(mt5.positions[0].sl, 3649.70, places=2)

        # Tick berikutnya di +55 pips (Ask 3644.50) -> JANGAN partial tahap 2 dulu sebelum +100 pips!
        mt5.bid, mt5.ask = 3644.30, 3644.50
        r_mid = pm.audit_and_manage_positions("XAUUSDc")
        self.assertEqual(len(r_mid["partial_actions"]), 0)
        self.assertAlmostEqual(mt5.positions[0].volume, 0.02, places=2)

        # Tahap 2: Running +100 pips SELL (Ask turun ke 3640.00) -> tutup 0.01 lagi (sisa 0.01 biarkan ke TP1 3635.00)!
        mt5.bid, mt5.ask = 3639.80, 3640.00
        r_s2 = pm.audit_and_manage_positions("XAUUSDc")
        self.assertEqual(len(r_s2["partial_actions"]), 1)
        self.assertAlmostEqual(mt5.positions[0].volume, 0.01, places=2)
        self.assertAlmostEqual(mt5.positions[0].sl, 3649.70, places=2)
        self.assertAlmostEqual(mt5.positions[0].tp, 3635.00, places=2)

    def test_stale_adopted_pending_order_auto_cancelled_and_valid_adopted_relinked(self):
        """
        Memastikan pending order scanner lama/ADOPTED (tanpa zone_id) yang TP-nya sudah terlewati
        atau tidak lagi cocok dengan zona aktif otomatis dibatalkan sehingga tidak memblokir
        MAX_ACTIVE_PENDING_ORDERS, sementara order ADOPTED yang masih cocok dengan zona aktif
        dipertahankan & di-relink, serta order manual tidak pernah disentuh.
        """
        mt5, sm, _, eng = self._new_env(bid=3522.71, ask=3522.93)
        ord_mgr = PendingOrderManager(eng, sm, cfg=self.cfg)
        active_zone = _make_supply_zone(
            score=78.6, timeframe="M30", lo=3529.39, hi=3540.16,
        )
        snap, scan, _ = _make_snapshot_and_scan(
            active_zone, self.df_m5_rej, bid=3522.71, ask=3522.93,
            h4_bias="BEARISH", h1_bias="BEARISH", m30_status="CONFIRMED",
        )

        # 1) Order scanner lama ADOPTED (TP 3554.05 sudah terlewati live price 3522.71 + di luar zona aktif)
        res_stale = mt5.order_send({
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": "XAUUSDc",
            "volume": 0.01,
            "type": mt5.ORDER_TYPE_SELL_LIMIT,
            "price": 3565.27,
            "sl": 3576.49,
            "tp": 3554.05,
            "magic": SCANNER_MAGIC_NUMBER,
            "comment": "SMC_EP_STALE",
        })
        tkt_stale = res_stale.order
        sm.update_setup_state(
            f"ADOPTED_ORD_{tkt_stale}", STATE_ORDER_PLACED,
            reason_code="ADOPTED_SCANNER_ORDER", direction="SELL",
            entry_mode="EARLY_PENDING", order_ticket=tkt_stale, planned_entry=3565.27,
        )

        # 2) Order manual user di harga jauh yang sama -> WAJIB tetap aman / tidak dibatalkan!
        res_manual = mt5.order_send({
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": "XAUUSDc",
            "volume": 0.01,
            "type": mt5.ORDER_TYPE_SELL_LIMIT,
            "price": 3565.27,
            "sl": 3576.49,
            "tp": 3554.05,
            "magic": 0,
            "comment": "manual_user_limit",
        })
        tkt_manual = res_manual.order

        audit_1 = ord_mgr.audit_and_manage_pending_orders(
            snapshot=snap, scan_result=scan, news_override=self.news_ok, symbol="XAUUSDc"
        )
        self.assertEqual(len(audit_1["cancelled"]), 1)
        self.assertEqual(audit_1["cancelled"][0]["ticket"], tkt_stale)
        rem_tickets = [o.ticket for o in mt5.orders_get()]
        self.assertNotIn(tkt_stale, rem_tickets)
        self.assertIn(tkt_manual, rem_tickets)

        # 3) Order scanner ADOPTED tanpa zone_id, TP belum terlewati, tetapi tidak cocok dengan zona aktif manapun -> BATAL!
        res_orphan = mt5.order_send({
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": "XAUUSDc",
            "volume": 0.01,
            "type": mt5.ORDER_TYPE_SELL_LIMIT,
            "price": 3565.27,
            "sl": 3576.49,
            "tp": 3480.00,  # TP belum terlewati oleh 3522.71, tapi 3565.27 tidak cocok dengan zona aktif 3529.39-3540.16
            "magic": SCANNER_MAGIC_NUMBER,
            "comment": "SMC_EP_ORPHAN",
        })
        tkt_orphan = res_orphan.order
        sm.update_setup_state(
            f"ADOPTED_ORD_{tkt_orphan}", STATE_ORDER_PLACED,
            reason_code="ADOPTED_SCANNER_ORDER", direction="SELL",
            entry_mode="EARLY_PENDING", order_ticket=tkt_orphan, planned_entry=3565.27,
        )
        audit_2 = ord_mgr.audit_and_manage_pending_orders(
            snapshot=snap, scan_result=scan, news_override=self.news_ok, symbol="XAUUSDc"
        )
        self.assertEqual(len(audit_2["cancelled"]), 1)
        self.assertEqual(audit_2["cancelled"][0]["ticket"], tkt_orphan)

        # 4) Order scanner ADOPTED tanpa zone_id yang COCOK dengan zona aktif 3529.39-3540.16 -> DIPERTAHANKAN & DI-RELINK!
        res_valid = mt5.order_send({
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": "XAUUSDc",
            "volume": 0.01,
            "type": mt5.ORDER_TYPE_SELL_LIMIT,
            "price": 3529.39,
            "sl": 3546.68,
            "tp": 3484.62,
            "magic": SCANNER_MAGIC_NUMBER,
            "comment": "SMC_EP_VALID_ADOPTED",
        })
        tkt_valid = res_valid.order
        sid_valid = f"ADOPTED_ORD_{tkt_valid}"
        sm.update_setup_state(
            sid_valid, STATE_ORDER_PLACED,
            reason_code="ADOPTED_SCANNER_ORDER", direction="SELL",
            entry_mode="EARLY_PENDING", order_ticket=tkt_valid, planned_entry=3529.39,
        )
        audit_3 = ord_mgr.audit_and_manage_pending_orders(
            snapshot=snap, scan_result=scan, news_override=self.news_ok, symbol="XAUUSDc"
        )
        self.assertEqual(len(audit_3["cancelled"]), 0)
        self.assertIn(tkt_valid, [o.ticket for o in mt5.orders_get()])
        self.assertEqual(sm.get_setup(sid_valid).get("zone_id"), active_zone.zone_id)

    def test_multi_zone_unlimited_pending_and_auto_cancel_invalid_or_far(self):
        """
        1. MAX_ACTIVE_PENDING_ORDERS=0 memasang pending order limit untuk setiap zona valid dalam jangkauan
           tanpa duplikasi di zona yang sama.
        2. Ketika salah satu zona menjadi invalid, pending order zona tersebut otomatis dibatalkan.
        3. Ketika harga menjauh melebihi batas jarak pending, pending order otomatis dibatalkan tanpa loop
           dan dapat dipasang kembali apabila harga kembali masuk jangkauan selama zona masih fresh & valid.
        """
        cfg_multi = dict(self.cfg)
        cfg_multi.update({
            "MAX_ACTIVE_PENDING_ORDERS": 0,
            "MAX_TOTAL_EXPOSURE_PERCENT": 10.0,
            "RISK_EARLY_PENDING_PERCENT": 2.5,
            "MAX_RISK_PER_TRADE_PERCENT": 2.5,
            "TP1_MODE": "FIXED_RR",
            "FORCE_FIXED_TP1": True,
            "TP1_RR": 1.0,
            "MIN_RR": 1.0,
            "PENDING_MAX_DISTANCE_USD": 35.0,
            "PENDING_CANCEL_HYSTERESIS_USD": 3.0,
            "AUTO_CANCEL_FAR_PENDING": True,
            "AUTO_CANCEL_INVALID_ZONE": True,
        })
        _, _, jrn_m, _ = self._new_env(bid=3520.0, ask=3520.2)
        mt5_m = MockMT5(trade_mode=MockMT5.ACCOUNT_TRADE_MODE_DEMO, currency="USD", balance=10000.0, equity=10000.0, bid=3520.0, ask=3520.2)
        sm_m = ExecutionStateManager(state_file=os.path.join(self.tmpdir.name, "state_multi_pend.json"), auto_load=False)
        sm_m.authorize_account(login=mt5_m.login, server=mt5_m.server, currency="USD", account_mode="DEMO", real_confirmed=False)
        sm_m.set_trading_mode("FULL_AUTO", auto_trading=True)
        mon_m = AutoExecutionMonitor(mt5_api=mt5_m, state_mgr=sm_m, journal=jrn_m, symbol="XAUUSDc", cfg=cfg_multi, connect_fn=lambda: True)

        z1 = _make_supply_zone(score=82.0, timeframe="M30", lo=3530.0, hi=3534.0)
        z1.zone_id = "ZONE_M30_SUPPLY_Z1"
        z2 = _make_supply_zone(score=86.0, timeframe="H1", lo=3545.0, hi=3549.0)
        z2.zone_id = "ZONE_H1_SUPPLY_Z2"

        snap, scan, dec = _make_snapshot_and_scan(z1, self.df_m5_rej, bid=3520.0, ask=3520.2)
        scan["active_zones"] = [z1.to_dict()]
        scan["radar_zones"] = [z2.to_dict()]

        # Siklus 1: Kedua zona valid (jarak 10 USD & 25 USD <= 35 USD) -> KEDUANYA dipasang SELL LIMIT!
        cyc1 = mon_m.run_once(snapshot=snap, scan_result=scan, decision=dec, news_override=self.news_ok)
        self.assertTrue(cyc1["execution_outcome"]["executed"])
        self.assertEqual(len(mt5_m.orders_get()), 2)
        ord_prices = sorted(round(o.price_open, 2) for o in mt5_m.orders_get())
        self.assertEqual(ord_prices, [3531.0, 3546.0])

        # Siklus 2: Tidak ada duplikasi order pada zona yang sama
        mon_m.run_once(snapshot=snap, scan_result=scan, decision=dec, news_override=self.news_ok)
        self.assertEqual(len(mt5_m.orders_get()), 2)

        # Siklus 3: Zona Z1 menjadi INVALIDATED -> order Z1 (3531.0) batal otomatis, Z2 (3546.0) tetap aktif!
        z1_inv = _make_supply_zone(score=0.0, timeframe="M30", lo=3530.0, hi=3534.0, is_valid=False, freshness="INVALIDATED")
        z1_inv.zone_id = "ZONE_M30_SUPPLY_Z1"
        scan_inv = dict(scan)
        scan_inv["active_zones"] = [z2.to_dict()]
        scan_inv["radar_zones"] = []
        scan_inv["rejected_zones"] = [z1_inv.to_dict()]
        cyc3 = mon_m.run_once(snapshot=snap, scan_result=scan_inv, decision=dec, news_override=self.news_ok)
        self.assertEqual(len(cyc3["order_audit"]["cancelled"]), 1)
        self.assertEqual([round(o.price_open, 2) for o in mt5_m.orders_get()], [3546.0])

        # Siklus 4: Harga turun menjauh ke 3502.0 (jarak ke 3546.0 = 44.0 > batas 38.0 USD, namun di atas TP)
        mt5_m.orders[0].tp = 3480.0
        mt5_m.bid, mt5_m.ask = 3502.0, 3502.2
        snap_far, scan_far, dec_far = _make_snapshot_and_scan(z2, self.df_m5_rej, bid=3502.0, ask=3502.2)
        scan_far["active_zones"] = [z2.to_dict()]
        scan_far["radar_zones"] = []
        cyc4 = mon_m.run_once(snapshot=snap_far, scan_result=scan_far, decision=dec_far, news_override=self.news_ok)
        self.assertEqual(len(cyc4["order_audit"]["cancelled"]), 1)
        self.assertEqual(cyc4["order_audit"]["cancelled"][0]["reason_code"], "ZONE_TOO_FAR")
        # Dan TIDAK dipasang ulang selama masih jauh (0 active orders)!
        self.assertEqual(len(mt5_m.orders_get()), 0)

        # Siklus 5: Harga naik kembali ke 3520.0 (jarak 26 USD <= 35 USD) dan Z2 masih fresh -> otomatis dipasang kembali!
        mt5_m.bid, mt5_m.ask = 3520.0, 3520.2
        snap_back, scan_back, dec_back = _make_snapshot_and_scan(z2, self.df_m5_rej, bid=3520.0, ask=3520.2)
        scan_back["active_zones"] = [z2.to_dict()]
        cyc5 = mon_m.run_once(snapshot=snap_back, scan_result=scan_back, decision=dec_back, news_override=self.news_ok)
        self.assertTrue(cyc5["execution_outcome"]["executed"])
        self.assertEqual([round(o.price_open, 2) for o in mt5_m.orders_get()], [3546.0])

    def test_single_lot_001_auto_be_plus_after_80_pips(self):
        """
        Posisi 0.01 lot tidak bisa partial close, sehingga:
        - Di +79 pips (7.9 USD) belum digeser ke BE+.
        - Begitu menyentuh +80 pips (8.0 USD), SL otomatis digeser ke BE+ (entry +/- BE_PLUS_LOCK_POINTS).
        """
        from modules.position_manager import PositionManager
        mt5, sm, _, eng = self._new_env(bid=3530.0, ask=3530.2)
        cfg_be = dict(self.cfg)
        cfg_be.update({
            "PARTIAL_CLOSE_ENABLED": True,
            "PARTIAL_CLOSE_LOT_STEP": 0.01,
            "PARTIAL_MIN_REMAINING_LOT": 0.01,
            "BE_AFTER_PARTIAL_ENABLED": True,
            "BE_PLUS_LOCK_POINTS": 0.30,
            "BE_SINGLE_LOT_ENABLED": True,
            "BE_SINGLE_LOT_TRIGGER_POINTS": 8.0,
            "BREAK_EVEN_ENABLED": False,
        })
        pm = PositionManager(exec_engine=eng, state_mgr=sm, cfg=cfg_be)

        # Buka posisi SELL 0.01 lot @ 3530.00, SL 3545.00, TP 3515.00
        res = mt5.order_send({
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": "XAUUSDc",
            "volume": 0.01,
            "type": mt5.ORDER_TYPE_SELL,
            "price": 3530.00,
            "sl": 3545.00,
            "tp": 3515.00,
            "magic": SCANNER_MAGIC_NUMBER,
            "comment": "SMC_EP_BE001",
        })
        tkt = res.order
        sm.update_setup_state(
            "SETUP_XAUUSDC_SELL_BE001", STATE_POSITION_ACTIVE,
            direction="SELL", position_ticket=tkt, actual_entry=3530.00,
            actual_lot=0.01, final_lot=0.01, sl=3545.00, tp=3515.00,
        )

        # +79 pips (Ask 3522.10 -> profit 7.90 USD < 8.00 USD) -> BELUM BE+!
        mt5.bid, mt5.ask = 3521.90, 3522.10
        r_before = pm.audit_and_manage_positions("XAUUSDc")
        self.assertEqual(len(r_before["protection_updates"]), 0)
        self.assertAlmostEqual(mt5.positions[0].sl, 3545.00, places=2)

        # +80 pips (Ask 3522.00 -> profit 8.00 USD) -> BE+ AKTIF (3530.00 - 0.30 = 3529.70)!
        mt5.bid, mt5.ask = 3521.80, 3522.00
        r_after = pm.audit_and_manage_positions("XAUUSDc")
        self.assertEqual(len(r_after["protection_updates"]), 1)
        self.assertAlmostEqual(mt5.positions[0].sl, 3529.70, places=2)
        self.assertAlmostEqual(mt5.positions[0].volume, 0.01, places=2)

    def test_real_lot_001_fixed_110_pips_tp_and_min_rr_enforcement(self):
        """
        Verifikasi aturan Auto Real (`FULL_AUTO` pada akun `REAL`) untuk `final_lot == 0.01`:
        1. Jika SL <= 5.50 USD pada MIN_RR/LOOSE_RR = 2.0 (mis. SL 5.00 USD -> RR = 11.00 / 5.00 = 2.20 >= 2.0),
           setup ENTRY_READY & tereksekusi dengan TP tepat 110 pips (11.00 USD) dari harga entry.
        2. Jika SL > 5.50 USD pada MIN_RR/LOOSE_RR = 2.0 (mis. SL 6.00 USD -> RR = 11.00 / 6.00 = 1.83 < 2.0),
           setup otomatis BLOCKED (`RR_BELOW_MINIMUM`).
        3. Untuk lot >= 0.02 pada akun REAL, atau untuk akun DEMO / mode DRY_RUN pada lot 0.01,
           aturan TP 110 pips TIDAK menimpa TP standar.
        """
        cfg_110 = dict(self.cfg)
        cfg_110.update({
            "AUTO_LOOSE_MODE": True,
            "LOOSE_LOT": 0.01,
            "MIN_RR": 2.0,
            "TP1_RR": 2.0,
            "TP1_MODE": "FIXED_RR",
            "FORCE_FIXED_TP1": True,
            "REAL_LOT_001_FIXED_TP_ENABLED": True,
            "REAL_LOT_001_FIXED_TP_PIPS": 110.0,
            "RISK_ADAPTIVE_PERCENT": 2.5,
            "MAX_RISK_PER_TRADE_PERCENT": 2.5,
            "MAX_TOTAL_EXPOSURE_PERCENT": 5.0,
        })

        # 1) REAL + FULL_AUTO + lot 0.01 + SL jarak 5.00 USD (SELL @ 3644.00, SL 3649.00) -> TP = 3633.00 (110 pips), RR = 2.20 >= 2.0!
        mt5_ok, sm_ok, _, eng_ok = self._new_env(equity_idc=30_000_000.0, bid=3644.00, ask=3644.20)
        eng_ok.cfg = cfg_110
        z_001_ok = _make_supply_zone(score=74.0, grade="B", timeframe="H1", lo=3643.00, hi=3647.00)
        snap_ok, scan_ok, dec_ok = _make_snapshot_and_scan(z_001_ok, self.df_m5_rej, bid=3644.00, ask=3644.20)
        dec_ok["plan"]["sl"] = 3649.00  # jarak SL = 5.00 USD -> RR 11/5 = 2.20

        val_ok = validate_trade_candidate(
            z_001_ok, mt5_ok, sm_ok, snapshot=snap_ok, scan_result=scan_ok, decision=dec_ok,
            news_override=self.news_ok, cfg=cfg_110,
        )
        self.assertEqual(val_ok["state"], STATE_ENTRY_READY, msg=val_ok.get("reason_text"))
        self.assertAlmostEqual(val_ok["final_lot"], 0.01, places=2)
        self.assertAlmostEqual(val_ok["trade_plan"]["tp"], 3633.00, places=2)  # 3644.00 - 11.00
        self.assertAlmostEqual(val_ok["actual_rr"], 2.20, places=2)

        ex_ok = eng_ok.execute_validated_setup(val_ok, trading_mode="FULL_AUTO")
        self.assertTrue(ex_ok["executed"])
        self.assertAlmostEqual(mt5_ok.positions[0].tp, 3633.00, places=2)
        self.assertAlmostEqual(mt5_ok.positions[0].volume, 0.01, places=2)

        # 2) REAL + FULL_AUTO + lot 0.01 + SL jarak 6.00 USD (SELL @ 3644.00, SL 3650.00) -> RR = 11.00 / 6.00 = 1.83 < MIN_RR 2.0 -> BLOCK!
        mt5_blk, sm_blk, _, _ = self._new_env(equity_idc=30_000_000.0, bid=3644.00, ask=3644.20)
        z_001_blk = _make_supply_zone(score=74.0, grade="B", timeframe="H1", lo=3643.00, hi=3647.00)
        z_001_blk.zone_id = "ZONE_BLK_RR_183"
        snap_blk, scan_blk, dec_blk = _make_snapshot_and_scan(z_001_blk, self.df_m5_rej, bid=3644.00, ask=3644.20)
        dec_blk["plan"]["sl"] = 3650.00  # jarak SL = 6.00 USD -> RR 1.83 < 2.0

        val_blk = validate_trade_candidate(
            z_001_blk, mt5_blk, sm_blk, snapshot=snap_blk, scan_result=scan_blk, decision=dec_blk,
            news_override=self.news_ok, cfg=cfg_110,
        )
        self.assertEqual(val_blk["state"], STATE_BLOCKED)
        self.assertEqual(val_blk["reason_code"], REASON_RR_BELOW_MINIMUM)

        # 3) REAL + FULL_AUTO + lot 0.02 (score 85 Grade A) -> TIDAK dikunci 110 pips, tetap ikut TP1_RR=2.0 (6.00 * 2.0 = 12.00 -> TP 3632.00)!
        mt5_002, sm_002, _, eng_002 = self._new_env(equity_idc=50_000_000.0, bid=3644.00, ask=3644.20)
        eng_002.cfg = cfg_110
        z_002 = _make_supply_zone(score=85.0, grade="A", timeframe="H1", lo=3643.00, hi=3647.00)
        z_002.zone_id = "ZONE_LOT_002_STD_TP"
        snap_2, scan_2, dec_2 = _make_snapshot_and_scan(z_002, self.df_m5_rej, bid=3644.00, ask=3644.20)
        dec_2["plan"]["sl"] = 3650.00  # jarak SL = 6.00 USD -> TP 2R = 3632.00

        val_002 = validate_trade_candidate(
            z_002, mt5_002, sm_002, snapshot=snap_2, scan_result=scan_2, decision=dec_2,
            news_override=self.news_ok, cfg=cfg_110,
        )
        self.assertEqual(val_002["state"], STATE_ENTRY_READY)
        self.assertAlmostEqual(val_002["final_lot"], 0.02, places=2)
        self.assertAlmostEqual(val_002["trade_plan"]["tp"], 3632.00, places=2)

        # 4) Mode DRY_RUN (meskipun akun REAL & lot 0.01) -> TIDAK dikunci 110 pips (tetap ikut TP1_RR=2.0 -> TP 3632.00)!
        mt5_dry, sm_dry, _, _ = self._new_env(equity_idc=30_000_000.0, bid=3644.00, ask=3644.20)
        sm_dry.set_trading_mode("DRY_RUN", auto_trading=True)
        z_dry = _make_supply_zone(score=74.0, grade="B", timeframe="H1", lo=3643.00, hi=3647.00)
        z_dry.zone_id = "ZONE_DRY_001_STD_TP"
        snap_d, scan_d, dec_d = _make_snapshot_and_scan(z_dry, self.df_m5_rej, bid=3644.00, ask=3644.20)
        dec_d["plan"]["sl"] = 3650.00
        val_dry = validate_trade_candidate(
            z_dry, mt5_dry, sm_dry, snapshot=snap_d, scan_result=scan_d, decision=dec_d,
            trading_mode="DRY_RUN", news_override=self.news_ok, cfg=cfg_110,
        )
        self.assertEqual(val_dry["state"], STATE_ENTRY_READY)
        self.assertAlmostEqual(val_dry["final_lot"], 0.01, places=2)
        self.assertAlmostEqual(val_dry["trade_plan"]["tp"], 3632.00, places=2)

    def test_grade_a_wait_confirmation_no_limit_002_003_lot_and_auto_cancel_pending(self):
        """
        Verifikasi aturan Zona Grade A / A+ / 80+ & 90+:
        1. Pending limit 0.01 yang terlanjur aktif di zona Grade A otomatis dibatalkan (GRADE_A_WAIT_CONFIRM),
           sedangkan pending limit di zona Grade B (mis. SNR G:B 77.7) dan order manual user tetap utuh.
        2. Zona Grade A (SMC 76.5 maupun SNR 86.1) dan Strong HTF 94+ TIDAK dipasang Order Limit saat jauh
           (termasuk saat HTF_CONFLICT + CONFLICT_MIDZONE_LIMIT_ENABLED=True).
        3. Saat harga masuk zona Grade A TANPA konfirmasi M5 (meskipun AUTO_LOOSE_MODE=True), robot TIDAK
           eksekusi buta — tetap WAIT_TRIGGER menunggu candle M5 close terkonfirmasi.
        4. Begitu konfirmasi M5 close valid muncul:
           - Grade A (75-89) dieksekusi Market 0.02 lot.
           - Strong HTF 90+ (tervalidasi) dieksekusi Market 0.03 lot.
        """
        cfg_ga = dict(self.cfg)
        cfg_ga.update({
            "AUTO_LOOSE_MODE": True,
            "LOOSE_LOT": 0.01,
            "LOT_STANDARD": 0.01,
            "LOT_MEDIUM": 0.02,
            "LOT_STRONG": 0.03,
            "HTF_STRONG_MAX_LOT": 0.03,
            "MAX_LOT": 0.03,
            "MIN_RR": 2.0,
            "TP1_RR": 2.0,
            "TP1_MODE": "FIXED_RR",
            "FORCE_FIXED_TP1": True,
            "CONFLICT_MIDZONE_LIMIT_ENABLED": True,
            "GRADE_A_WAIT_CONFIRMATION_ENABLED": True,
            "GRADE_A_MIN_SCORE": 75.0,
            "AUTO_CANCEL_GRADE_A_PENDING_LIMIT": True,
            "REAL_LOT_001_FIXED_TP_ENABLED": True,
            "REAL_LOT_001_FIXED_TP_PIPS": 110.0,
            "RISK_EARLY_PENDING_PERCENT": 2.5,
            "RISK_ADAPTIVE_PERCENT": 2.5,
            "MAX_RISK_PER_TRADE_PERCENT": 2.5,
            "MAX_TOTAL_EXPOSURE_PERCENT": 10.0,
            "MAX_ACTIVE_POSITIONS": 5,
            "SNR_ZONE_MIN_STABLE_SECONDS": 0.0,
        })

        # 1) Auto-cancel pending limit 0.01 di R1 Grade A (86.1), pertahankan S1 SNR Grade B (77.7) & manual user!
        mt5, sm, jrn, eng = self._new_env(equity_idc=30_000_000.0, bid=3572.00, ask=3572.20)
        eng.cfg = cfg_ga
        ord_mgr = PendingOrderManager(eng, sm, cfg=cfg_ga)

        # Pasang 3 pending order: (a) R1 Grade A 86.1, (b) S1 SNR Grade B 77.7, (c) Manual user
        r_ga = mt5.order_send({
            "action": mt5.TRADE_ACTION_PENDING, "symbol": "XAUUSDc", "volume": 0.01,
            "type": mt5.ORDER_TYPE_SELL_LIMIT, "price": 3602.07, "sl": 3608.52, "tp": 3589.17,
            "magic": SCANNER_MAGIC_NUMBER, "comment": "SMC_EP_R1_GA",
        })
        sm.update_setup_state(
            f"ADOPTED_ORD_{r_ga.order}", STATE_ORDER_PLACED,
            zone_id="SNR_M30_RES_360732", direction="SELL", entry_mode="EARLY_PENDING",
            order_ticket=r_ga.order, planned_entry=3602.07,
        )
        sm.state.setdefault("executed_setup_ids", []).append(f"ADOPTED_ORD_{r_ga.order}")

        r_gb = mt5.order_send({
            "action": mt5.TRADE_ACTION_PENDING, "symbol": "XAUUSDc", "volume": 0.01,
            "type": mt5.ORDER_TYPE_BUY_LIMIT, "price": 3558.67, "sl": 3549.97, "tp": 3576.07,
            "magic": SCANNER_MAGIC_NUMBER, "comment": "SMC_EP_S1_GB",
        })
        sm.update_setup_state(
            f"ADOPTED_ORD_{r_gb.order}", STATE_ORDER_PLACED,
            zone_id="SNR_H1_SUP_355117", direction="BUY", entry_mode="EARLY_PENDING",
            order_ticket=r_gb.order, planned_entry=3558.67,
        )

        r_man = mt5.order_send({
            "action": mt5.TRADE_ACTION_PENDING, "symbol": "XAUUSDc", "volume": 0.01,
            "type": mt5.ORDER_TYPE_SELL_LIMIT, "price": 3602.50, "sl": 3610.00, "tp": 3585.00,
            "magic": 0, "comment": "manual_user",
        })

        df_untouched = self.df_m5_rej.iloc[:-15].copy()
        df_untouched.attrs["is_closed"] = True
        snap_p = {
            "symbol": "XAUUSDc", "bid": 3572.00, "ask": 3572.20, "mid": 3572.10,
            "current_price": 3572.10, "spread": 0.20, "market_status": "OPEN",
            "data_freshness": "FRESH", "tick_age": 0.5, "frames": {"M5": df_untouched},
        }
        scan_p = {
            "symbol": "XAUUSDc", "current_price": 3572.10, "bid": 3572.00, "ask": 3572.20,
            "mid": 3572.10, "spread": 0.20, "market_status": "OPEN", "data_freshness": "FRESH",
            "htf_bias": "BEARISH", "h1_structure": "BEARISH", "m30_status": "CONFIRMED",
            "htf_context": {"H4": "BEARISH", "H1": "BEARISH", "m30_status": "CONFIRMED", "has_conflict": False},
            "active_zones": [], "radar_zones": [],
        }
        snr_p = {
            "scan_result": {
                "symbol": "XAUUSDc",
                "supports": [{
                    "zone_id": "SNR_H1_SUP_355117", "zone_type": "SUPPORT", "timeframe": "H1",
                    "low": 3551.17, "high": 3561.17, "score": 77.73, "grade": "B",
                    "lifecycle_status": "ACTIVE", "confirmed_time": "2026-11-26 04:00:00", "touch_count": 0,
                }],
                "resistances": [{
                    "zone_id": "SNR_M30_RES_360732", "zone_type": "RESISTANCE", "timeframe": "M30",
                    "low": 3600.32, "high": 3607.32, "score": 86.10, "grade": "A",
                    "lifecycle_status": "ACTIVE", "confirmed_time": "2026-11-26 02:30:00", "touch_count": 1,
                }],
            },
            "decision": {},
        }

        audit_ga = ord_mgr.audit_and_manage_pending_orders(
            snapshot=snap_p, scan_result=scan_p, news_override=self.news_ok, symbol="XAUUSDc", snr_result=snr_p
        )
        self.assertEqual(len(audit_ga["cancelled"]), 1)
        self.assertEqual(audit_ga["cancelled"][0]["ticket"], r_ga.order)
        self.assertEqual(audit_ga["cancelled"][0]["reason_code"], "GRADE_A_WAIT_CONFIRM")
        rem_tkts = [o.ticket for o in mt5.orders_get()]
        self.assertNotIn(r_ga.order, rem_tkts)
        self.assertIn(r_gb.order, rem_tkts)   # S1 SNR Grade B tetap aktif!
        self.assertIn(r_man.order, rem_tkts)  # Manual user tetap utuh!

        # 2) Zona SMC Grade A (76.5) saat harga belum menyentuh -> TIDAK pasang Limit (WAIT_ZONE_TOUCH)
        z_smc_a = _make_supply_zone(score=76.5, grade="A", timeframe="H1", lo=3643.00, hi=3647.00)
        z_smc_a.zone_id = "SMC_H1_GA_765"
        mt5.bid, mt5.ask = 3630.00, 3630.20
        snap_far, scan_far, dec_far = _make_snapshot_and_scan(
            z_smc_a, df_untouched, bid=3630.00, ask=3630.20,
            h4_bias="BEARISH", h1_bias="BULLISH", has_conflict=True,
        )
        val_wait_touch = validate_trade_candidate(
            z_smc_a, mt5, sm, snapshot=snap_far, scan_result=scan_far, decision=dec_far,
            news_override=self.news_ok, cfg=cfg_ga,
        )
        self.assertFalse(val_wait_touch["can_execute"])
        self.assertEqual(val_wait_touch["entry_mode"], "ADAPTIVE_ENTRY")
        self.assertEqual(val_wait_touch["reason_code"], "WAIT_ZONE_TOUCH")

        # 3) Harga masuk ke dalam zona Grade A (3645.50 > 25% kedalaman zona), tapi candle M5 BELUM ada konfirmasi ->
        #    meskipun AUTO_LOOSE_MODE=True, robot TIDAK boleh tembak buta (wajib WAIT_TRIGGER)!
        df_touch_no_trig = df_untouched.copy()
        ts_t = df_touch_no_trig.index[-1] + pd.Timedelta(minutes=5)
        # Candle bullish biasa naik masuk zona tanpa wick rejection / CHoCH bearish
        df_touch_no_trig.loc[ts_t] = {
            "Open": 3641.0, "High": 3645.6, "Low": 3640.8, "Close": 3645.4, "Volume": 200,
            "EMA20": 3638.0, "EMA50": 3636.0, "EMA200": 3634.0, "RSI14": 62.0, "ATR14": 2.0,
        }
        df_touch_no_trig.attrs["is_closed"] = True
        mt5.bid, mt5.ask = 3645.50, 3645.70
        snap_nt, scan_nt, dec_nt = _make_snapshot_and_scan(z_smc_a, df_touch_no_trig, bid=3645.50, ask=3645.70)
        val_nt = validate_trade_candidate(
            z_smc_a, mt5, sm, snapshot=snap_nt, scan_result=scan_nt, decision=dec_nt,
            news_override=self.news_ok, cfg=cfg_ga,
        )
        self.assertFalse(val_nt["can_execute"])
        self.assertEqual(val_nt["state"], "WAIT_TRIGGER")
        self.assertEqual(val_nt["reason_code"], "WAIT_TRIGGER")

        # 4) Begitu muncul konfirmasi M5 close valid (Rejection):
        #    4a. SMC Grade A (76.5) -> ENTRY_READY & eksekusi Market 0.02 lot!
        mt5.bid, mt5.ask = 3644.00, 3644.20
        snap_c, scan_c, dec_c = _make_snapshot_and_scan(z_smc_a, self.df_m5_rej, bid=3644.00, ask=3644.20)
        dec_c["plan"]["sl"] = 3649.00
        val_ga_ok = validate_trade_candidate(
            z_smc_a, mt5, sm, snapshot=snap_c, scan_result=scan_c, decision=dec_c,
            news_override=self.news_ok, cfg=cfg_ga,
        )
        self.assertEqual(val_ga_ok["state"], STATE_ENTRY_READY, msg=val_ga_ok.get("reason_text"))
        self.assertAlmostEqual(val_ga_ok["final_lot"], 0.02, places=2)
        ex_ga = eng.execute_validated_setup(val_ga_ok, trading_mode="FULL_AUTO")
        self.assertTrue(ex_ga["executed"])
        self.assertAlmostEqual(ex_ga["actual_lot"], 0.02, places=2)

        #    4b. Strong H4+H1 (94.0) dengan konfirmasi M5 valid -> ENTRY_READY & eksekusi Market 0.03 lot!
        z_str_94 = _make_supply_zone(score=94.0, grade="A+", timeframe="H4+H1", lo=3643.00, hi=3647.00)
        z_str_94.zone_id = "SMC_H4H1_STR_940"
        snap_s, scan_s, dec_s = _make_snapshot_and_scan(z_str_94, self.df_m5_rej, bid=3644.00, ask=3644.20)
        dec_s["plan"]["sl"] = 3649.00
        val_94 = validate_trade_candidate(
            z_str_94, mt5, sm, snapshot=snap_s, scan_result=scan_s, decision=dec_s,
            news_override=self.news_ok, cfg=cfg_ga,
        )
        self.assertEqual(val_94["state"], STATE_ENTRY_READY, msg=val_94.get("reason_text"))
        self.assertAlmostEqual(val_94["final_lot"], 0.03, places=2)

    def test_two_stage_80plus_scout_limit_001_then_confirmed_scale_in_002_003(self):
        """
        Verifikasi penuh strategi 2-Tahap khusus zona skor >= 80.0 pada Auto Real:
        1. Tahap 1: Pasang Pending Limit 0.01 lot lebih dulu (TP 110 pip / 11.00 USD).
        2. Selama pending belum terjemput -> tidak ada duplikasi & Tahap 2 belum jalan.
        3. Saat zona & pending 0.01 terjemput (jadi posisi aktif) tetapi M5 belum konfirmasi ->
           Tahap 2 wajib WAIT_TRIGGER (tidak tembak buta meskipun AUTO_LOOSE_MODE=True).
        4. Saat konfirmasi M5 close valid muncul -> Tahap 2 langsung tembak Market Order:
           - Skor 80-89 -> tambah +0.02 lot (total 0.03 lot; tiket 0.02 ikut TP 2R & Partial +50 pip).
           - Skor 90+ Strong HTF -> tambah +0.03 lot (total 0.04 lot).
           - Jika risk budget sempit hanya cukup 0.01 lot -> Tahap 2 turun ke +0.01 lot & ikut TP 110 pip.
        5. Setelah Tahap 2 tereksekusi -> terkunci permanen (maksimal 1x Scale-In per zona).
        6. Zona skor < 80 (mis. 74.0) -> hanya Tahap 1 Limit 0.01 lot, TIDAK ada Scale-In Tahap 2.
        """
        cfg_2st = dict(self.cfg)
        cfg_2st.update({
            "AUTO_LOOSE_MODE": True,
            "LOOSE_LOT": 0.01,
            "LOT_STANDARD": 0.01,
            "LOT_MEDIUM": 0.02,
            "LOT_STRONG": 0.03,
            "HTF_STRONG_MAX_LOT": 0.03,
            "LTF_MAX_LOT": 0.02,
            "EARLY_PENDING_MAX_LOT": 0.01,
            "MAX_LOT": 0.04,
            "MIN_RR": 2.0,
            "TP1_RR": 2.0,
            "TP1_MODE": "FIXED_RR",
            "FORCE_FIXED_TP1": True,
            "SCALE_IN_80PLUS_ENABLED": True,
            "SCALE_IN_MIN_SCORE": 80.0,
            "GRADE_A_WAIT_CONFIRMATION_ENABLED": False,
            "AUTO_CANCEL_GRADE_A_PENDING_LIMIT": False,
            "REAL_LOT_001_FIXED_TP_ENABLED": True,
            "REAL_LOT_001_FIXED_TP_PIPS": 110.0,
            "RISK_EARLY_PENDING_PERCENT": 2.5,
            "RISK_ADAPTIVE_PERCENT": 2.5,
            "MAX_RISK_PER_TRADE_PERCENT": 2.5,
            "MAX_TOTAL_EXPOSURE_PERCENT": 5.0,
            "MAX_ACTIVE_POSITIONS": 5,
            "MAX_TRADES_PER_DAY": 10,
            "SNR_ZONE_MIN_STABLE_SECONDS": 0.0,
        })

        # --- A. SKENARIO ZONA 86.0 (80-89): Limit 0.01 (TP 110 pip) -> Terjemput -> Confirm M5 -> +0.02 lot (Total 0.03) ---
        mt5, sm, jrn, _ = self._new_env(equity_idc=30_000_000.0, bid=3630.00, ask=3630.20)
        mon = AutoExecutionMonitor(mt5_api=mt5, state_mgr=sm, journal=jrn, symbol="XAUUSDc", cfg=cfg_2st, connect_fn=lambda: True)

        df_untouched = self.df_m5_rej.iloc[:-15].copy()
        df_untouched.attrs["is_closed"] = True

        z86 = _make_supply_zone(score=86.0, grade="A", timeframe="H1", lo=3643.00, hi=3647.00)
        z86.zone_id = "ZONE_H1_86_2STAGE"
        z86.sl_recommended = 3649.00  # jarak SL dari limit 3644.00 = 5.00 USD -> RR 11/5 = 2.20 >= 2.0

        # Siklus 1: Harga di 3630 -> Pasang Tahap 1 SELL LIMIT 0.01 lot @ 3644.00, TP = 3633.00 (110 pip)!
        snap1, scan1, dec1 = _make_snapshot_and_scan(z86, df_untouched, bid=3630.00, ask=3630.20)
        dec1["plan"]["sl"] = 3649.00
        cyc1 = mon.run_once(snapshot=snap1, scan_result=scan1, decision=dec1, news_override=self.news_ok)
        self.assertTrue(cyc1["execution_outcome"]["executed"])
        self.assertEqual(len(mt5.orders), 1)
        self.assertAlmostEqual(mt5.orders[0].volume_current, 0.01, places=2)
        self.assertAlmostEqual(mt5.orders[0].price_open, 3644.00, places=2)
        self.assertAlmostEqual(mt5.orders[0].tp, 3633.00, places=2)  # 110 pips!

        # Siklus 2: Selama pending belum terjemput -> tidak ada order baru & Tahap 2 belum jalan
        cyc2 = mon.run_once(snapshot=snap1, scan_result=scan1, decision=dec1, news_override=self.news_ok)
        self.assertIsNone(cyc2["execution_outcome"])
        self.assertEqual(len(mt5.orders), 1)
        self.assertEqual(len(mt5.positions), 0)

        # Siklus 3: Harga menjemput 3644.50 -> Simulasikan broker mengisi pending order Stage-1 menjadi posisi aktif 0.01 lot,
        #           NAMUN candle M5 baru naik masuk zona tanpa konfirmasi rejection/CHoCH -> Tahap 2 wajib WAIT_TRIGGER!
        ord_s1 = mt5.orders.pop(0)
        mt5.positions.append(SimpleNamespace(
            ticket=ord_s1.ticket, symbol="XAUUSDc", type=mt5.ORDER_TYPE_SELL, volume=0.01,
            price_open=ord_s1.price_open, sl=ord_s1.sl, tp=ord_s1.tp, profit=-5000.0,
            swap=0.0, commission=0.0, magic=SCANNER_MAGIC_NUMBER, comment=ord_s1.comment, time=int(time.time()),
        ))
        df_touch_no_conf = df_untouched.copy()
        ts_nc = df_touch_no_conf.index[-1] + pd.Timedelta(minutes=5)
        df_touch_no_conf.loc[ts_nc] = {
            "Open": 3641.0, "High": 3645.2, "Low": 3640.8, "Close": 3645.0, "Volume": 180,
            "EMA20": 3638.0, "EMA50": 3636.0, "EMA200": 3634.0, "RSI14": 62.0, "ATR14": 2.0,
        }
        df_touch_no_conf.attrs["is_closed"] = True
        mt5.bid, mt5.ask = 3644.80, 3645.00
        snap3, scan3, dec3 = _make_snapshot_and_scan(z86, df_touch_no_conf, bid=3644.80, ask=3645.00)
        dec3["plan"]["sl"] = 3649.00
        cyc3 = mon.run_once(snapshot=snap3, scan_result=scan3, decision=dec3, news_override=self.news_ok)
        self.assertIsNone(cyc3["execution_outcome"])
        self.assertTrue(cyc3["best_evaluation"]["is_scale_in"])
        self.assertEqual(cyc3["best_evaluation"]["state"], "WAIT_TRIGGER")
        self.assertEqual(len(mt5.positions), 1)  # Masih 1 posisi (0.01 lot)!

        # Siklus 4: Candle M5 close mencetak VALID REJECTION di zona -> Tahap 2 langsung tembak SELL MARKET +0.02 lot (TP 2R = 3634.00)!
        mt5.bid, mt5.ask = 3644.00, 3644.20
        snap4, scan4, dec4 = _make_snapshot_and_scan(z86, self.df_m5_rej, bid=3644.00, ask=3644.20)
        dec4["plan"]["sl"] = 3649.00  # jarak SL = 5.00 USD -> TP 2R = 3634.00
        cyc4 = mon.run_once(snapshot=snap4, scan_result=scan4, decision=dec4, news_override=self.news_ok)
        self.assertTrue(cyc4["execution_outcome"]["executed"])
        self.assertEqual(len(mt5.positions), 2)
        self.assertEqual([round(p.volume, 2) for p in mt5.positions], [0.01, 0.02])  # Total 0.03 lot!
        self.assertAlmostEqual(mt5.positions[0].tp, 3633.00, places=2)  # Stage-1 (0.01) TP = 110 pip
        self.assertAlmostEqual(mt5.positions[1].tp, 3634.00, places=2)  # Stage-2 (0.02) TP = 2R

        # Siklus 5: Siklus berikutnya -> Stage-2 sudah terkunci, TIDAK menembak order ke-3!
        cyc5 = mon.run_once(snapshot=snap4, scan_result=scan4, decision=dec4, news_override=self.news_ok)
        self.assertIsNone(cyc5["execution_outcome"])
        self.assertEqual(len(mt5.positions), 2)

        # --- B. SKENARIO ZONA 94.0 STRONG H4+H1: Stage-1 (0.01) + Stage-2 (+0.03) = Total 0.04 lot! ---
        mt5_94, sm_94, jrn_94, _ = self._new_env(equity_idc=40_000_000.0, bid=3630.00, ask=3630.20)
        mon_94 = AutoExecutionMonitor(mt5_api=mt5_94, state_mgr=sm_94, journal=jrn_94, symbol="XAUUSDc", cfg=cfg_2st, connect_fn=lambda: True)
        z94 = _make_supply_zone(score=94.0, grade="A+", timeframe="H4+H1", lo=3643.00, hi=3647.00)
        z94.zone_id = "ZONE_H4H1_94_2STAGE"
        z94.sl_recommended = 3649.00

        s94_1, sc94_1, d94_1 = _make_snapshot_and_scan(z94, df_untouched, bid=3630.00, ask=3630.20)
        d94_1["plan"]["sl"] = 3649.00
        mon_94.run_once(snapshot=s94_1, scan_result=sc94_1, decision=d94_1, news_override=self.news_ok)
        self.assertEqual(len(mt5_94.orders), 1)
        ord_94 = mt5_94.orders.pop(0)
        mt5_94.positions.append(SimpleNamespace(
            ticket=ord_94.ticket, symbol="XAUUSDc", type=mt5_94.ORDER_TYPE_SELL, volume=0.01,
            price_open=ord_94.price_open, sl=ord_94.sl, tp=ord_94.tp, profit=0.0,
            swap=0.0, commission=0.0, magic=SCANNER_MAGIC_NUMBER, comment=ord_94.comment, time=int(time.time()),
        ))
        mt5_94.bid, mt5_94.ask = 3644.00, 3644.20
        s94_2, sc94_2, d94_2 = _make_snapshot_and_scan(z94, self.df_m5_rej, bid=3644.00, ask=3644.20)
        d94_2["plan"]["sl"] = 3649.00
        cyc_94 = mon_94.run_once(snapshot=s94_2, scan_result=sc94_2, decision=d94_2, news_override=self.news_ok)
        self.assertTrue(cyc_94["execution_outcome"]["executed"])
        self.assertEqual([round(p.volume, 2) for p in mt5_94.positions], [0.01, 0.03])  # Total 0.04 lot!

        # --- C. SKENARIO STEP-DOWN LOT TAHAP 2 -> Jika sisa budget hanya muat +0.01 lot, TP ikut 110 pip! ---
        # Equity 6,500,000 IDC * 2.5% = 162,500 IDC total budget; Stage-1 (0.01 @ SL 5.0 USD) pakai 75,000 IDC ->
        # sisa budget ~87,500 IDC (hanya cukup +0.01 lot ~79,050 IDC, tidak cukup +0.02 lot ~158,100 IDC)!
        mt5_sd, sm_sd, jrn_sd, _ = self._new_env(equity_idc=6_500_000.0, bid=3630.00, ask=3630.20)
        mon_sd = AutoExecutionMonitor(mt5_api=mt5_sd, state_mgr=sm_sd, journal=jrn_sd, symbol="XAUUSDc", cfg=cfg_2st, connect_fn=lambda: True)
        z86_sd = _make_supply_zone(score=86.0, grade="A", timeframe="H1", lo=3643.00, hi=3647.00)
        z86_sd.zone_id = "ZONE_H1_86_STEPDOWN"
        z86_sd.sl_recommended = 3649.00
        s_sd1, sc_sd1, d_sd1 = _make_snapshot_and_scan(z86_sd, df_untouched, bid=3630.00, ask=3630.20)
        d_sd1["plan"]["sl"] = 3649.00
        mon_sd.run_once(snapshot=s_sd1, scan_result=sc_sd1, decision=d_sd1, news_override=self.news_ok)
        ord_sd = mt5_sd.orders.pop(0)
        mt5_sd.positions.append(SimpleNamespace(
            ticket=ord_sd.ticket, symbol="XAUUSDc", type=mt5_sd.ORDER_TYPE_SELL, volume=0.01,
            price_open=ord_sd.price_open, sl=ord_sd.sl, tp=ord_sd.tp, profit=0.0,
            swap=0.0, commission=0.0, magic=SCANNER_MAGIC_NUMBER, comment=ord_sd.comment, time=int(time.time()),
        ))
        mt5_sd.bid, mt5_sd.ask = 3644.00, 3644.20
        s_sd2, sc_sd2, d_sd2 = _make_snapshot_and_scan(z86_sd, self.df_m5_rej, bid=3644.00, ask=3644.20)
        d_sd2["plan"]["sl"] = 3649.00
        cyc_sd = mon_sd.run_once(snapshot=s_sd2, scan_result=sc_sd2, decision=d_sd2, news_override=self.news_ok)
        self.assertTrue(cyc_sd["execution_outcome"]["executed"])
        self.assertAlmostEqual(mt5_sd.positions[1].volume, 0.01, places=2)  # Disesuaikan jadi +0.01 lot!
        self.assertAlmostEqual(mt5_sd.positions[1].tp, 3633.00, places=2)      # Tiket 0.01 ikut TP 110 pip!

        # --- D. SKENARIO ZONA < 80 (skor 74.0): Hanya Tahap 1 Limit 0.01 lot, TIDAK ada Scale-In Tahap 2! ---
        mt5_74, sm_74, jrn_74, _ = self._new_env(equity_idc=30_000_000.0, bid=3630.00, ask=3630.20)
        mon_74 = AutoExecutionMonitor(mt5_api=mt5_74, state_mgr=sm_74, journal=jrn_74, symbol="XAUUSDc", cfg=cfg_2st, connect_fn=lambda: True)
        z74 = _make_supply_zone(score=74.0, grade="B", timeframe="H1", lo=3643.00, hi=3647.00)
        z74.zone_id = "ZONE_H1_74_NO_SCALEIN"
        z74.sl_recommended = 3649.00
        s74_1, sc74_1, d74_1 = _make_snapshot_and_scan(z74, df_untouched, bid=3630.00, ask=3630.20)
        d74_1["plan"]["sl"] = 3649.00
        mon_74.run_once(snapshot=s74_1, scan_result=sc74_1, decision=d74_1, news_override=self.news_ok)
        ord_74 = mt5_74.orders.pop(0)
        mt5_74.positions.append(SimpleNamespace(
            ticket=ord_74.ticket, symbol="XAUUSDc", type=mt5_74.ORDER_TYPE_SELL, volume=0.01,
            price_open=ord_74.price_open, sl=ord_74.sl, tp=ord_74.tp, profit=0.0,
            swap=0.0, commission=0.0, magic=SCANNER_MAGIC_NUMBER, comment=ord_74.comment, time=int(time.time()),
        ))
        mt5_74.bid, mt5_74.ask = 3644.00, 3644.20
        s74_2, sc74_2, d74_2 = _make_snapshot_and_scan(z74, self.df_m5_rej, bid=3644.00, ask=3644.20)
        d74_2["plan"]["sl"] = 3649.00
        cyc_74 = mon_74.run_once(snapshot=s74_2, scan_result=sc74_2, decision=d74_2, news_override=self.news_ok)
        self.assertIsNone(cyc_74["execution_outcome"])
        self.assertEqual(len(mt5_74.positions), 1)  # Tetap 1 posisi (0.01 lot) saja!


if __name__ == "__main__":
    unittest.main(verbosity=2)
