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

            # 3. CLOSE EXISTING POSITION
            if act == self.TRADE_ACTION_DEAL and request.get("position"):
                pt = int(request["position"])
                pos = next((p for p in self.positions if p.ticket == pt), None)
                if pos is None:
                    return SimpleNamespace(retcode=10013, comment="Pos not found")
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
            "LOT_STANDARD": 0.01,
            "LOT_MEDIUM": 0.02,
            "LOT_STRONG": 0.05,
            "MAX_LOT": 0.05,
            "EARLY_PENDING_MAX_LOT": 0.01,
            "LTF_MAX_LOT": 0.02,
            "HTF_STRONG_MAX_LOT": 0.05,
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
        df_untouched = self.df_m5_rej.iloc[:-5].copy()
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
        df_untouched = self.df_m5_rej.iloc[:-5].copy()
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
        cfg_live = get_auto_trading_config()
        self.assertEqual(cfg_live["MAX_LOT"], 0.01)
        z94 = _make_supply_zone(score=94.0, grade="A+", timeframe="H4+H1")
        self.assertEqual(select_target_lot(z94, "ADAPTIVE_ENTRY", strong_zone_validated=True, cfg=cfg_live), 0.01)


if __name__ == "__main__":
    unittest.main(verbosity=2)
