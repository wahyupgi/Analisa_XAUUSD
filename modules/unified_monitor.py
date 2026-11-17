# -*- coding: utf-8 -*-
"""
unified_monitor.py — Mode Pantau Terpadu (Unified Monitor: SMC + SNR + MT5 Account Monitor).

Arsitektur:
Unified Monitor
├── SMC Scanner (SMC Market Analysis, SMC Zone Monitoring, SMC Entry Validation, SMC Signal Decision)
├── SNR Scanner (SNR Market Analysis, SNR Zone Monitoring, SNR Entry Validation, SNR Signal Decision)
└── MT5 Account Monitor (Open Positions, Pending Orders, Spread, Risk Alerts — Read-Only)

Aturan Utama:
- Jangan membandingkan SMC dengan SNR, jangan membuat ranking metode terbaik, jangan menggabungkan skor.
- Jangan membuat satu metode menjadi filter/syarat validasi metode lainnya.
- Jangan memaksa kedua metode menghasilkan arah sinyal yang sama; tampilkan apa adanya tanpa label konflik.
- Gunakan 1 snapshot data MT5 yang konsisten untuk setiap siklus monitoring.
- Isolasi error: kegagalan SMC tidak menghentikan SNR, dan kegagalan SNR tidak menghentikan SMC.
- Alert independen beridentitas metode ([SMC] ... dan [SNR] ...) hanya muncul saat terjadi perubahan status.
"""
from datetime import datetime, timezone
import json
import math
import os
import re
import shutil
import time

try:
    import MetaTrader5 as mt5
except Exception:  # pragma: no cover
    mt5 = None

from modules.config import SYMBOL as DEFAULT_APP_SYMBOL, FILE_STATE_ZONE_TRACKER
from modules.config import PANTAU_FULL_REFRESH_SECONDS, PANTAU_INTERVALS_SECONDS
from modules.utils import format_wib_timestamp
from modules.mt5_conn import pastikan_mt5_aktif
from modules.pantau import (
    ZoneMonitorState,
    collect_radar_watchlist_with_reconciliation,
    mode_pantau_mt5,
)
from modules.snr.data_provider import safe_float
from modules.snr.formatter import format_snr_focus_plan, format_snr_lifecycle_status
from modules.snr.monitor import (
    SnrCycleEngine,
    format_snr_monitor_section,
    mode_pantau_snr,
)
from modules.zone_scanner import (
    _extract_risk_monitor_snapshot,
    create_market_snapshot,
    evaluate_trade_decision,
    format_smc_zone_lines,
    format_scan_presentation,
    scan_xauusd_zones,
    smc_zone_summary,
)


_GLOBAL_LAST_SMC_STATE = {}


def _position_value(position, name, default=None):
    return position.get(name, default) if isinstance(position, dict) else getattr(position, name, default)


class PantauFocusTracker:
    """Track independently published SMC/SNR zones without placing or modifying orders."""

    def __init__(self, symbol):
        self.symbol = symbol
        self.focus = {"SMC": None, "SNR": None}
        self.completed = {"SMC": None, "SNR": None}
        self.active_methods = set()
        self.refresh_required = False
        self._saved_state = None
        self._last_tick = None
        self._history_error_active = False
        self._load()

    def _load(self):
        if not os.path.exists(FILE_STATE_ZONE_TRACKER):
            return
        try:
            with open(FILE_STATE_ZONE_TRACKER, "r", encoding="utf-8") as state_file:
                data = json.load(state_file)
            if not isinstance(data, dict):
                raise ValueError("Shared zone state harus berupa object.")
            namespaces = data.get("pantau_focus_state", {})
            if not isinstance(namespaces, dict):
                raise ValueError("State fokus pantau harus berupa object.")
            saved = namespaces.get(self.symbol)
            if saved is None:
                return
            if not isinstance(saved, dict) or not isinstance(saved.get("focus"), dict) or not isinstance(saved.get("completed"), dict):
                raise ValueError("State fokus pantau tidak valid.")
            for method in self.focus:
                record = saved["focus"].get(method)
                completed = saved["completed"].get(method)
                if record is not None:
                    if not isinstance(record, dict) or not isinstance(record.get("zone"), dict) or not all(
                        key in record for key in ("zone_id", "low", "high", "direction", "issued_at", "plan", "touched", "ticket", "ever_taken", "account_status", "uncertain")
                    ):
                        raise ValueError(f"State fokus {method} tidak lengkap.")
                    record["uncertain"] = True
                    record["account_status"] = ""
                if completed is not None and (not isinstance(completed, dict) or "zone_id" not in completed or "left" not in completed):
                    raise ValueError(f"State zona selesai {method} tidak valid.")
                self.focus[method] = record
                self.completed[method] = completed
        except (OSError, ValueError, TypeError) as exc:
            from modules.utils import _catat_error
            _catat_error("pantau_focus_load", exc)
            raise RuntimeError("State fokus pantau tidak dapat dibaca; pemantauan dihentikan.") from exc

    def _save(self):
        focus = {}
        for method, record in self.focus.items():
            if record is None:
                focus[method] = None
                continue
            saved = dict(record)
            zone = record["zone"]
            saved["zone"] = {
                key: zone.get(key) for key in ("zone_id", "zone_type", "timeframe", "rank_label", "grade")
                if zone.get(key) is None or isinstance(zone.get(key), (str, int, float, bool))
            }
            saved["zone"].update(
                {"price_low": record["low"], "price_high": record["high"]}
                if method == "SMC" else {"low": record["low"], "high": record["high"]}
            )
            focus[method] = saved
        state = {"focus": focus, "completed": self.completed}
        temporary_path = FILE_STATE_ZONE_TRACKER + f".pantau.{os.getpid()}.tmp"
        try:
            serialized = json.dumps(state, sort_keys=True, ensure_ascii=True)
            if serialized == self._saved_state:
                return
            if os.path.exists(FILE_STATE_ZONE_TRACKER):
                with open(FILE_STATE_ZONE_TRACKER, "r", encoding="utf-8") as state_file:
                    data = json.load(state_file)
                if not isinstance(data, dict):
                    raise ValueError("Shared zone state harus berupa object.")
            else:
                data = {}
            namespaces = data.setdefault("pantau_focus_state", {})
            if not isinstance(namespaces, dict):
                raise ValueError("State fokus pantau harus berupa object.")
            namespaces[self.symbol] = state
            with open(temporary_path, "w", encoding="utf-8") as state_file:
                json.dump(data, state_file, indent=2, ensure_ascii=True)
            os.replace(temporary_path, FILE_STATE_ZONE_TRACKER)
            self._saved_state = serialized
        except (OSError, TypeError, ValueError) as exc:
            from modules.utils import _catat_error
            _catat_error("pantau_focus_save", exc)
            raise RuntimeError("State fokus pantau tidak dapat disimpan; pemantauan dihentikan.") from exc

    @staticmethod
    def _candidate(method, result):
        if not isinstance(result, dict) or result.get("status") != "OK":
            return None, None
        dec = result.get("decision") or {}
        scan = result.get("scan_result") or {}
        if method == "SMC":
            zone = dec.get("zone") or dec.get("primary_zone") or {}
            if not zone:
                priority = dec.get("radar_priority") or scan.get("radar_priority") or {}
                zone = priority.get("primary_directional_zone") or priority.get("primary_zone") or {}
            summary = smc_zone_summary(zone, dec) if zone else {}
            plan = {
                "direction": summary.get("direction"),
                "entry": summary.get("entry"),
                "sl": summary.get("sl"),
                "tp1": summary.get("tp1"),
            } if summary.get("kind") in ("CONFIRMED", "PENDING", "PROVISIONAL") else None
        else:
            zone = dec.get("target_zone") or {}
            ev = scan.get("entry_validation") or {}
            provisional = ev.get("provisional_plan") or {}
            if dec.get("signal") in ("BUY", "SELL"):
                plan = {
                    "direction": dec.get("signal"), "entry": dec.get("effective_entry"),
                    "sl": dec.get("sl"), "tp1": dec.get("tp1"),
                }
            elif provisional.get("status") == "PROVISIONAL":
                plan = {key: provisional.get(key) for key in ("direction", "entry", "sl", "tp1")}
            else:
                plan = None
        return zone if isinstance(zone, dict) else None, plan

    @staticmethod
    def _bounds(method, zone):
        if method == "SMC":
            return safe_float(zone.get("price_low")), safe_float(zone.get("price_high"))
        return safe_float(zone.get("low")), safe_float(zone.get("high"))

    @staticmethod
    def _valid_plan(plan):
        if not isinstance(plan, dict):
            return False
        entry = safe_float(plan.get("entry"))
        sl = safe_float(plan.get("sl"))
        tp = safe_float(plan.get("tp1"))
        direction = str(plan.get("direction") or "").upper()
        return entry > 0 and sl > 0 and tp > 0 and (
            sl < entry < tp if direction == "BUY" else tp < entry < sl if direction == "SELL" else False
        )

    @staticmethod
    def _current_zone(method, result, zone_id):
        scan = (result or {}).get("scan_result") or {}
        if method == "SMC":
            zones = (scan.get("radar_zones") or []) + (scan.get("active_zones") or []) + (scan.get("rejected_zones") or [])
        else:
            zones = ((scan.get("ranked_zones") or {}).get("all_zones") or [])
        return next((z for z in zones if isinstance(z, dict) and z.get("zone_id") == zone_id), None)

    @staticmethod
    def _next_zone(method, result, previous, current_zone):
        scan = (result or {}).get("scan_result") or {}
        if method == "SMC":
            priority = (result.get("decision") or {}).get("radar_priority") or scan.get("radar_priority") or {}
            zones = priority.get("aligned_zones") or []
        else:
            ranked = scan.get("ranked_zones") or {}
            zones = (ranked.get("active_supports") or []) + (ranked.get("active_resistances") or [])
        for zone in zones:
            if not isinstance(zone, dict) or not zone.get("zone_id") or zone["zone_id"] == previous["zone_id"]:
                continue
            if zone.get("zone_type") != current_zone.get("zone_type"):
                continue
            lifecycle = str(zone.get("lifecycle_status") or zone.get("status") or "ACTIVE").upper()
            if zone.get("is_valid") is False or lifecycle in ("INVALIDATED", "EXPIRED", "MITIGATED", "BROKEN", "UNCONFIRMED"):
                continue
            return zone
        return None

    def _observations(self, snapshot):
        stamp = int(safe_float(snapshot.get("tick_time_msc")))
        bid = safe_float(snapshot.get("tick_bid_raw"))
        ask = safe_float(snapshot.get("tick_ask_raw"))
        if stamp <= 0 or bid <= 0 or ask < bid:
            raise ValueError("Timestamp milidetik atau Bid/Ask tick tidak tersedia.")
        current = (stamp, bid, ask)
        previous = self._last_tick
        if previous is None:
            self._last_tick = current
            return [current]
        if stamp < previous[0]:
            raise ValueError("Waktu tick MT5 bergerak mundur.")
        ticks = mt5.copy_ticks_range(
            self.symbol,
            datetime.fromtimestamp(previous[0] // 1000, tz=timezone.utc),
            datetime.fromtimestamp(stamp // 1000 + 1, tz=timezone.utc),
            mt5.COPY_TICKS_ALL,
        )
        if ticks is None:
            raise RuntimeError(f"Riwayat tick MT5 tidak tersedia: {mt5.last_error()}")
        history = [
            (int(tick["time_msc"]), float(tick["bid"]), float(tick["ask"]))
            for tick in ticks if previous[0] <= int(tick["time_msc"]) <= stamp
        ]
        if any(not math.isfinite(t[1]) or not math.isfinite(t[2]) or t[1] <= 0 or t[2] < t[1] for t in history):
            raise ValueError("Riwayat tick berisi Bid/Ask tidak valid.")
        starts = [i for i, tick in enumerate(history) if tick == previous]
        ends = [i for i, tick in enumerate(history) if tick == current]
        if not starts or not ends or any(history[i][0] > history[i + 1][0] for i in range(len(history) - 1)):
            raise ValueError("Batas riwayat tick tidak lengkap atau urutannya tidak valid.")
        start, end = starts[-1], ends[0]
        if start > end:
            if previous == current and all(tick == previous for tick in history):
                start = end
            else:
                raise ValueError("Urutan tick pada milidetik yang sama tidak dapat dipastikan.")
        if any(tick != previous for tick in history[:start] if tick[0] == previous[0]):
            raise ValueError("Tick awal pada milidetik yang sama ambigu.")
        if any(tick != current for tick in history[end + 1:ends[-1] + 1]):
            raise ValueError("Tick akhir pada milidetik yang sama ambigu.")
        self._last_tick = current
        return history[start + 1:end + 1] or [current]

    def update(self, results, snapshot, positions, full_refresh):
        alerts = []
        self.refresh_required = False
        self.active_methods = {method for method in ("SMC", "SNR") if results.get(method) is not None}
        snap = snapshot if isinstance(snapshot, dict) else {}
        bid, ask = safe_float(snap.get("bid")), safe_float(snap.get("ask"))
        price = safe_float(snap.get("snapshot_price"))
        fresh = (
            snap.get("data_ok") is not False and snap.get("live") is True
            and str(snap.get("market_status") or "").upper() == "OPEN"
            and str(snap.get("data_freshness") or "").upper() not in ("STALE", "INVALID", "NO DATA")
            and 0 < bid <= ask and price > 0
        )
        observations = []
        history_complete = True
        if fresh:
            try:
                observations = self._observations(snap)
                self._history_error_active = False
            except (AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError, OverflowError) as exc:
                history_complete = False
                if not self._history_error_active:
                    from modules.utils import _catat_error
                    _catat_error("pantau_focus_tick_history", exc)
                    alerts.append("[DATA] RIWAYAT TICK TIDAK LENGKAP; URUTAN HARGA TIDAK PASTI")
                self._history_error_active = True
                stamp = int(safe_float(snap.get("tick_time_msc")))
                tick_bid = safe_float(snap.get("tick_bid_raw"))
                tick_ask = safe_float(snap.get("tick_ask_raw"))
                if stamp > 0 and 0 < tick_bid <= tick_ask:
                    observations = [(stamp, tick_bid, tick_ask)]

        candidates = {method: self._candidate(method, results.get(method)) for method in ("SMC", "SNR")}
        new_focus = set()
        plan_updates = {}
        for method in ("SMC", "SNR"):
            if method not in self.active_methods:
                continue
            zone, plan = candidates[method]
            record = self.focus[method]
            completed = self.completed[method]
            if record is None and full_refresh and completed and zone and zone.get("zone_id") == completed["zone_id"]:
                alternative = self._next_zone(method, results[method], completed, zone)
                if alternative is not None:
                    zone, plan = alternative, None
            if record is None and full_refresh and fresh and isinstance(zone, dict) and zone.get("zone_id"):
                low, high = self._bounds(method, zone)
                state = str(zone.get("lifecycle_status") or zone.get("status") or "ACTIVE").upper()
                kind = str(zone.get("zone_type") or "").upper()
                direction = "BUY" if kind in ("DEMAND", "SUPPORT") else "SELL" if kind in ("SUPPLY", "RESISTANCE") else ""
                completed = self.completed[method]
                if completed and completed["zone_id"] == zone["zone_id"]:
                    if not low <= price <= high:
                        completed["left"] = True
                    if not (completed["left"] and low <= price <= high and (
                        (results[method].get("decision") or {}).get("status") == "VALID"
                        if method == "SMC" else (results[method].get("decision") or {}).get("signal") in ("BUY", "SELL")
                    )):
                        continue
                if direction and 0 < low <= high and state not in ("INVALIDATED", "EXPIRED", "MITIGATED", "BROKEN", "UNCONFIRMED") and zone.get("is_valid") is not False:
                    record = {
                        "zone": dict(zone), "zone_id": zone["zone_id"], "low": low, "high": high,
                        "direction": direction, "issued_at": time.time(), "plan": None, "touched": False,
                        "ticket": None, "ever_taken": False, "account_status": "", "uncertain": False,
                    }
                    self.focus[method] = record
                    new_focus.add(method)
                    self.completed[method] = None
                    if completed:
                        alerts.append(
                            f"[{method}] ZONA BERIKUTNYA {low:.2f}-{high:.2f}; "
                            "KANDIDAT RANKING, MENUNGGU VALIDASI ENTRY"
                        )
            if record and full_refresh and zone and zone.get("zone_id") == record["zone_id"]:
                if not record["touched"]:
                    plan_updates[method] = (
                        {"direction": plan["direction"], **{key: safe_float(plan[key]) for key in ("entry", "sl", "tp1")}}
                        if self._valid_plan(plan) and plan["direction"] == record["direction"] else None
                    )
                elif not record["plan"] and not record["ever_taken"] and self._valid_plan(plan) and plan["direction"] == record["direction"]:
                    plan_updates[method] = {"direction": plan["direction"], **{key: safe_float(plan[key]) for key in ("entry", "sl", "tp1")}}
                if method in new_focus and method in plan_updates:
                    record["plan"] = plan_updates.pop(method)

        matches = {}
        for method, record in self.focus.items():
            matches[method] = []
            if not record or positions is None:
                continue
            for position in positions:
                symbol = str(_position_value(position, "symbol", ""))
                kind = _position_value(position, "type")
                direction = "BUY" if kind == 0 or str(kind).upper() == "BUY" else "SELL" if kind == 1 or str(kind).upper() == "SELL" else ""
                opened = safe_float(_position_value(position, "price_open"))
                opened_at = safe_float(_position_value(position, "time"))
                opened_msc = safe_float(_position_value(position, "time_msc"))
                if opened_msc > 0:
                    opened_at = opened_msc / 1000.0
                ticket = _position_value(position, "ticket")
                if (
                    symbol == self.symbol and direction == record["direction"] and ticket
                    and record["low"] <= opened <= record["high"]
                    and opened_at >= record["issued_at"]
                    and (not record["ticket"] or ticket == record["ticket"])
                ):
                    matches[method].append(position)

        for method, record in list(self.focus.items()):
            if not record or method not in self.active_methods:
                continue
            if not fresh or positions is None or not history_complete:
                if not record["uncertain"]:
                    alerts.append(f"[{method}] DATA ATAU RIWAYAT TIDAK LENGKAP; FOKUS DIPERTAHANKAN")
                record["uncertain"] = True
            other = "SNR" if method == "SMC" else "SMC"
            match = matches[method]
            shared = bool(match and any(
                _position_value(p, "ticket") == _position_value(match[0], "ticket")
                for p in matches[other]
            ))
            account_status = "POSITIONS UNAVAILABLE" if positions is None else "AMBIGUOUS" if len(match) > 1 or shared else "RUNNING" if len(match) == 1 else (
                "POSITION NOT OBSERVED" if record["ever_taken"] else "NO MATCHING POSITION"
            )
            if account_status != record["account_status"]:
                if account_status == "RUNNING":
                    record["ticket"] = _position_value(match[0], "ticket")
                    record["ever_taken"] = True
                    if not record["touched"]:
                        alerts.append(f"[{method}] POSISI DIBUKA DI ZONA {record['low']:.2f}-{record['high']:.2f}")
                    record["touched"] = True
                    alerts.append(f"[{method}] RUNNING: POSISI MT5 COCOK #{record['ticket']} (ASOSIASI OTOMATIS)")
                elif account_status == "AMBIGUOUS":
                    alerts.append(f"[{method}] POSISI MT5 AMBIGU; ENTRY BELUM DAPAT DIPASTIKAN")
                elif account_status == "POSITION NOT OBSERVED":
                    alerts.append(f"[{method}] POSISI #{record['ticket']} TIDAK LAGI TERLIHAT; HASIL BELUM DIPASTIKAN")
                elif account_status == "POSITIONS UNAVAILABLE":
                    alerts.append(f"[{method}] POSISI MT5 TIDAK DAPAT DIBACA")
                elif record["touched"]:
                    alerts.append(f"[{method}] BELUM ADA POSISI MT5 YANG COCOK DENGAN ZONA")
                record["account_status"] = account_status

            current = self._current_zone(method, results.get(method), record["zone_id"]) if full_refresh else None
            if method == "SNR" and current is not None and (
                safe_float(current.get("touch_count")) > 0
                or str(current.get("lifecycle_event") or "").upper() in ("TOUCHED", "SWEEP")
            ):
                if not record.get("historical_touch"):
                    record["historical_touch"] = True
                    alerts.append("[SNR] ZONA PERNAH DISENTUH (HISTORIS); BUKAN BUKTI ENTRY")
            lifecycle = str((current or {}).get("lifecycle_status") or (current or {}).get("status") or "").upper()
            invalid = current is not None and lifecycle in ("INVALIDATED", "BROKEN", "EXPIRED", "MITIGATED")
            plan = record.get("plan") or {}
            sl, tp = safe_float(plan.get("sl")), safe_float(plan.get("tp1"))
            outcome = ""
            ticks = observations[-1:] if method in new_focus else observations
            for tick_stamp, tick_bid, tick_ask in ticks:
                if tick_stamp < record["issued_at"] * 1000:
                    continue
                tick_price = (tick_bid + tick_ask) / 2.0
                previous_price = safe_float(record.get("last_price"))
                if not record["touched"] and previous_price > 0 and (
                    previous_price < record["low"] and tick_price > record["high"]
                    or previous_price > record["high"] and tick_price < record["low"]
                ):
                    if not record["uncertain"]:
                        alerts.append(f"[{method}] HARGA MELEWATI ZONA TANPA TICK DI DALAMNYA; HASIL TIDAK PASTI")
                    record["uncertain"] = True
                record["last_price"] = tick_price
                new_touch = False
                if record["low"] <= tick_price <= record["high"] and not record["touched"]:
                    record["touched"] = True
                    new_touch = True
                    alerts.append(f"[{method}] HARGA DI ZONA {record['low']:.2f}-{record['high']:.2f}; ENTRY BELUM DIPASTIKAN")
                    if account_status == "NO MATCHING POSITION":
                        alerts.append(f"[{method}] BELUM ADA POSISI MT5 YANG COCOK DENGAN ZONA")
                    self.refresh_required = True
                exit_price = tick_bid if record["direction"] == "BUY" else tick_ask
                sl_hit = sl > 0 and (exit_price <= sl if record["direction"] == "BUY" else exit_price >= sl)
                tp_hit = tp > 0 and (exit_price >= tp if record["direction"] == "BUY" else exit_price <= tp)
                if new_touch and (sl_hit or tp_hit) and account_status != "RUNNING":
                    record["uncertain"] = True
                    alerts.append(f"[{method}] ZONA DAN BATAS RENCANA DALAM TICK SAMA; URUTAN TIDAK PASTI")
                opened_msc = safe_float(_position_value(match[0], "time_msc")) if account_status == "RUNNING" else 0
                if not opened_msc and account_status == "RUNNING":
                    opened_msc = safe_float(_position_value(match[0], "time")) * 1000
                if not record["uncertain"] and account_status != "AMBIGUOUS" and record["touched"] and tick_stamp >= opened_msc:
                    if sl_hit:
                        outcome = f"HARGA TERPANTAU DI SL RENCANA {sl:.2f}"
                    elif tp_hit:
                        outcome = f"HARGA TERPANTAU DI TP1 RENCANA {tp:.2f}"
                if outcome:
                    break
            if not outcome and not record["uncertain"] and account_status == "NO MATCHING POSITION" and invalid and not record["ever_taken"]:
                outcome = "ZONA INVALID SEBELUM POSISI TERDETEKSI"
            if outcome:
                note = "POSISI BELUM TERDETEKSI" if not record["ever_taken"] else "STATUS PENUTUPAN MT5 BELUM DIPASTIKAN"
                alerts.append(f"[{method}] {outcome}; {note}")
                self.completed[method] = {"zone_id": record["zone_id"], "left": not record["low"] <= price <= record["high"]}
                self.focus[method] = None
                self.refresh_required = True
            elif method in plan_updates and (not record["touched"] or not record["plan"] and not record["ever_taken"]):
                old_plan = record["plan"]
                record["plan"] = plan_updates[method]
                if record["touched"] and not old_plan and record["plan"]:
                    alerts.append(f"[{method}] RENCANA SL/TP1 TERSEDIA; BUKAN KONFIRMASI ENTRY")
                elif not record["touched"] and old_plan != record["plan"]:
                    alerts.append(f"[{method}] RENCANA SL/TP1 {'DIPERBARUI' if record['plan'] else 'TIDAK TERSEDIA'}; BUKAN KONFIRMASI ENTRY")
        self._save()
        return alerts

    def gate(self, method, result, detailed=False):
        if not isinstance(result, dict) or result.get("status") != "OK":
            return result
        record = self.focus[method]
        completed = self.completed[method]
        zone, current_plan = self._candidate(method, result)
        candidate_id = zone.get("zone_id") if isinstance(zone, dict) else None
        dec = result.get("decision") or {}
        signal_direction = (
            str(dec.get("direction") or "").upper() if method == "SMC" and dec.get("status") == "VALID"
            else str(dec.get("signal") or "").upper() if method == "SNR"
            else ""
        )
        blocked = (
            bool(record and candidate_id != record["zone_id"])
            or bool(record and record["uncertain"])
            or bool(record and (record["ever_taken"] or record["account_status"] == "AMBIGUOUS"))
            or bool(record and signal_direction in ("BUY", "SELL") and signal_direction != record["direction"])
            or bool(record and record["touched"] and self._valid_plan(current_plan) and (
                not self._valid_plan(record["plan"])
                or any(safe_float(current_plan[key]) != safe_float(record["plan"][key]) for key in ("entry", "sl", "tp1"))
            ))
            or bool(not record and completed and candidate_id == completed["zone_id"])
        )
        if not blocked:
            return result
        result = dict(result)
        result["alerts"] = []
        result["critical_warnings"] = []
        dec = dict(result.get("decision") or {})
        reason = (
            "WAIT_HISTORY_RECONCILIATION: Urutan harga atau posisi tidak pasti."
            if record and record["uncertain"] else
            "WAIT_FOCUS_RESOLUTION: Menunggu SL/TP1 atau invalidasi zona sebelum entry."
            if record else "MENUNGGU ZONA ATAU KUNJUNGAN BARU."
        )
        if method == "SMC":
            dec.update(
                status="WAIT", direction=None, zone=None, primary_zone=None, plan=None,
                nearest_valid_zone=None, next_radar_zone=None, signal_status="WAITING CONFIRMATION",
                reasons=[reason], zone_plans={}, early_setup={"best_setup": None, "all_setups": []},
                radar_priority={"primary_zone": None},
            )
            dec["stages"] = {
                **{key: {"status": "NOT_CHECKED", "msg": reason} for key in (
                    "zone_validation", "zone_touch", "liquidity_sweep", "structure_trigger",
                    "displacement", "entry_validation", "risk_reward",
                )},
                "DATA": {"ok": False, "msg": reason},
            }
            dec["active_setup"] = {"final_decision": "WAIT", "blocker": reason}
            scan = dict(result.get("scan_result") or {})
            scan["radar_priority"] = {"primary_zone": None}
            result["scan_result"] = scan
            result["decision"] = dec
            result["summary"] = extract_smc_monitor_summary(scan, dec)
            result["formatted_section"] = format_smc_monitor_section(result, detailed=detailed)
        else:
            same_zone = bool(record and candidate_id == record["zone_id"])
            same_bounds = bool(
                same_zone
                and abs(safe_float(zone.get("low")) - record["low"]) < 1e-8
                and abs(safe_float(zone.get("high")) - record["high"]) < 1e-8
            )
            old_zone = zone if same_bounds else record["zone"] if record else None
            focus = {
                key: old_zone.get(key) for key in (
                    "zone_id", "zone_type", "timeframe", "rank_label", "low", "high", "grade", "score",
                    "swing_class", "confluence_info", "historical_reaction_count", "touch_analysis", "lifecycle_event",
                    "lifecycle_status", "status",
                )
            } if old_zone else None
            if focus and not same_bounds:
                focus["lifecycle_status"] = "UNCONFIRMED"
            scan = dict(result.get("scan_result") or {})
            original_ev = scan.get("entry_validation") or {}
            original_plan = original_ev.get("provisional_plan") or {}
            plan_rr = original_plan.get("rr_calculation") or {}
            entry_rr = original_ev.get("rr_calculation") or {}
            if same_zone and original_ev.get("final_status") == "WAIT":
                validation_reason = str(original_ev.get("reason") or "")
                if validation_reason and not same_bounds:
                    validation_reason = (
                        f"ZONA SCAN {safe_float(zone.get('low')):.2f}-{safe_float(zone.get('high')):.2f}: "
                        + validation_reason
                    )
                result["focus_validation_reason"] = validation_reason
            if not same_bounds or original_ev.get("final_status") != "WAIT":
                entry_rr = {}
            failed_plan = (
                original_plan if same_bounds and plan_rr.get("rr_failed")
                else {"status": "N/A", "note": "FOKUS BELUM DIVALIDASI ULANG"}
            )
            dec.update(
                signal="WAIT", direction="NONE", live_entry="WAIT", target_zone=focus,
                entry=None, effective_entry=None, sl=None, tp1=None, tp2=None, reason=reason,
                trigger_status="NOT CONFIRMED", rr_display="NOT CHECKED",
            )
            ev = dict(original_ev)
            ev.update(
                final_status="WAIT", target_zone=focus, reason=reason, trigger_status="NOT CONFIRMED",
                provisional_plan=failed_plan,
                rr_calculation=entry_rr if entry_rr.get("checked") else {"checked": False, "valid": False, "rr_display": "NOT CHECKED"},
                recent_closed_touch=bool(original_ev.get("recent_closed_touch")) if same_bounds else False,
                focus_change=original_ev.get("focus_change") if same_bounds else None,
            )
            scan["entry_validation"] = ev
            scan["decision"] = dec
            scan["price_action"] = {
                "rejection_summary": "NOT CHECKED",
                "breakout_summary": "NOT CHECKED",
                "retest_summary": "NOT CHECKED",
                "momentum_summary": "NOT CHECKED",
            }
            result["scan_result"] = scan
            result["decision"] = dec
            result["formatted_section"] = format_snr_monitor_section(result, detailed=detailed)
        return result

    def scoped_alerts(self, method, result, detailed=False):
        if not isinstance(result, dict) or result.get("status") != "OK":
            return result
        record = self.focus[method]
        zone_id = record["zone_id"] if record else None
        alerts = []
        scan = result.get("scan_result") or {}
        radar = scan.get("radar") or result.get("radar") or {}
        action = scan.get("price_action") or {}
        for alert in result.get("alerts") or []:
            text = str(alert).upper()
            keep = "MARKET CLOSED" in text or "DATA UNAVAILABLE" in text
            if zone_id and not record["uncertain"]:
                if method == "SMC":
                    keep = not any(word in text for word in ("PERUBAHAN ZONA", "ZONA TERMITIGASI", "ZONA SUPPLY INVALID", "ZONA DEMAND INVALID"))
                elif "SUPPORT S1" in text:
                    keep = (radar.get("nearest_support") or {}).get("zone_id") == zone_id
                elif "RESISTANCE R1" in text:
                    keep = (radar.get("nearest_resistance") or {}).get("zone_id") == zone_id
                elif "REJECTION" in text:
                    keep = (action.get("rejection_detail") or {}).get("zone_id") == zone_id
                elif "BREAKOUT" in text or "BREAKDOWN" in text:
                    keep = (action.get("breakout_detail") or {}).get("zone_id") == zone_id
                elif "RETEST" in text:
                    keep = (action.get("retest_detail") or {}).get("zone_id") == zone_id
                elif "ZONA " in text and ("BROKEN" in text or "INVALID" in text):
                    keep = str(zone_id) in str(alert)
                elif "SETUP" in text:
                    keep = ((result.get("decision") or {}).get("target_zone") or {}).get("zone_id") == zone_id
            if keep:
                alerts.append(alert)
        result = dict(result)
        result["alerts"] = alerts
        if method == "SMC":
            result["formatted_section"] = format_smc_monitor_section(result, detailed=detailed)
        else:
            result["formatted_section"] = format_snr_monitor_section(result, detailed=detailed)
        return result

    def lines(self):
        lines = []
        for method, record in self.focus.items():
            if method not in self.active_methods:
                continue
            if not record:
                if self.completed[method]:
                    lines.append(f"{method} FOCUS: MENUNGGU ZONA ATAU KUNJUNGAN BARU")
                continue
            plan = record.get("plan") or {}
            status = (
                "ZONA DISENTUH" if record["touched"] else
                "ZONA PERNAH DISENTUH (HISTORIS)" if record.get("historical_touch") else
                "MENUNGGU ZONA"
            )
            position = (
                "RUNNING; POSISI COCOK OTOMATIS" if record["account_status"] == "RUNNING" else
                "POSISI AMBIGU" if record["account_status"] == "AMBIGUOUS" else
                "POSISI MT5 TIDAK DAPAT DIBACA" if record["account_status"] == "POSITIONS UNAVAILABLE" else
                "POSISI TIDAK TERLIHAT" if record["ever_taken"] else
                "POSISI BELUM TERDETEKSI" if record["touched"] or record.get("historical_touch") else ""
            )
            if position:
                status += f"; {position}"
            if record["uncertain"]:
                if status == "MENUNGGU ZONA":
                    status = "SENTUHAN BELUM TERBUKTI"
                status += " | RIWAYAT / HASIL TIDAK PASTI"
            levels = (
                f" | SL/TP1 ACUAN {plan['sl']:.2f}/{plan['tp1']:.2f}"
                if self._valid_plan(plan) else " | SL/TP1 N/A"
            )
            lines.append(f"{method} FOCUS: {record['direction']} {record['low']:.2f}-{record['high']:.2f} | {status}{levels}")
        return lines


class SmcAlertTracker:
    """
    Pelacak perubahan status (State-Change Detector) khusus SMC Monitor di dalam Unified Monitor.
    Menghasilkan alert beridentitas `[SMC]` hanya ketika terjadi perubahan status penting.
    """

    def __init__(self):
        global _GLOBAL_LAST_SMC_STATE
        self.last_state = dict(_GLOBAL_LAST_SMC_STATE)

    def reset(self):
        global _GLOBAL_LAST_SMC_STATE
        self.last_state = {}
        _GLOBAL_LAST_SMC_STATE = {}

    def evaluate_alerts(self, smc_summary):
        if not isinstance(smc_summary, dict) or smc_summary.get("status") == "ERROR":
            return {"alerts": [], "critical_warnings": []}

        curr_state = {
            "signal": str(smc_summary.get("decision_signal", "WAIT")).upper(),
            "direction": str(smc_summary.get("direction", "NONE")).upper(),
            "primary_zone": str(smc_summary.get("primary_zone_label", "NONE")),
            "primary_zone_id": str(smc_summary.get("primary_zone_id", "NONE")),
            "primary_setup_id": str(smc_summary.get("primary_setup_id", "NONE")),
            "zone_touch": str(smc_summary.get("zone_touch", "WAIT")).upper(),
            "sweep": str(smc_summary.get("liquidity_sweep", "WAIT")).upper(),
            "choch_bos": str(smc_summary.get("choch_bos", "WAIT")).upper(),
            "displacement": str(smc_summary.get("displacement", "WAIT")).upper(),
            "invalidated_count": int(smc_summary.get("invalidated_count", 0)),
            "mitigated_count": int(smc_summary.get("mitigated_count", 0)),
        }

        alerts = []
        critical_warnings = []

        # 1. Perubahan keputusan setup SMC
        same_setup = curr_state["primary_setup_id"] == self.last_state.get("primary_setup_id")
        if curr_state["signal"] != self.last_state.get("signal") or not same_setup:
            if curr_state["signal"] == "BUY":
                alerts.append("[SMC] BUY SETUP TERKONFIRMASI")
            elif curr_state["signal"] == "SELL":
                alerts.append("[SMC] SELL SETUP TERKONFIRMASI")
            elif curr_state["signal"] == "INVALID":
                alerts.append("[SMC] SETUP INVALID")

        # 2. Perubahan tahapan konfirmasi M5 SMC
        if (not same_setup or curr_state["zone_touch"] != self.last_state.get("zone_touch")) and curr_state["zone_touch"] == "IN_ZONE":
            alerts.append("[SMC] ZONE TOUCH TERKONFIRMASI")

        if (not same_setup or curr_state["sweep"] != self.last_state.get("sweep")) and curr_state["sweep"] == "CONFIRMED":
            alerts.append("[SMC] LIQUIDITY SWEEP TERKONFIRMASI")

        if (not same_setup or curr_state["choch_bos"] != self.last_state.get("choch_bos")) and curr_state["choch_bos"] == "CONFIRMED":
            alerts.append("[SMC] CHoCH / BOS M5 TERKONFIRMASI")

        if (not same_setup or curr_state["displacement"] != self.last_state.get("displacement")) and curr_state["displacement"] == "CONFIRMED":
            alerts.append("[SMC] DISPLACEMENT TERKONFIRMASI")

        # 3. Perubahan zona aktif / invalidasi zona SMC
        prev_inval = self.last_state.get("invalidated_count")
        if prev_inval is not None and curr_state["invalidated_count"] > prev_inval:
            last_inval_type = str(smc_summary.get("last_invalidated_type", "ZONA")).upper()
            alerts.append(f"[SMC] ZONA {last_inval_type} INVALID")

        prev_mit = self.last_state.get("mitigated_count")
        if prev_mit is not None and curr_state["mitigated_count"] > prev_mit:
            alerts.append("[SMC] ZONA TERMITIGASI")

        if (
            self.last_state.get("primary_zone_id") is not None
            and curr_state["primary_zone_id"] != self.last_state.get("primary_zone_id")
            and curr_state["primary_zone_id"] not in ("NONE", "-", "")
        ):
            z_short = curr_state.get("primary_zone", "").split(" (")[0].strip()
            alerts.append(f"[SMC] PERUBAHAN ZONA FOKUS -> {z_short}")

        self.last_state = curr_state
        _GLOBAL_LAST_SMC_STATE = dict(curr_state)
        return {
            "alerts": alerts,
            "critical_warnings": critical_warnings,
            "state": curr_state,
            "setup_id": curr_state.get("primary_setup_id"),
            "zone_id": curr_state.get("primary_zone_id"),
        }


def extract_smc_monitor_summary(scan_result, decision):
    """
    Mengekstrak ringkasan terstruktur dari output asli `scan_xauusd_zones` dan `evaluate_trade_decision`
    tanpa mengubah satu baris pun algoritma SMC.
    """
    scan = scan_result or {}
    dec = decision or {}

    ctx = dec.get("context") or scan.get("context") or {}
    htf_ctx = scan.get("htf_context") or {}
    raw_h4 = htf_ctx.get("H4")
    raw_h1 = htf_ctx.get("H1")
    raw_m30 = htf_ctx.get("M30")
    m30_stage = (dec.get("stages") or {}).get("M30") or {}
    h4_bias = ctx.get("htf_bias") or scan.get("htf_bias") or (raw_h4.get("bias") if isinstance(raw_h4, dict) else raw_h4) or "NETRAL"
    h1_struct = ctx.get("h1_structure") or scan.get("h1_structure") or (raw_h1.get("bias") if isinstance(raw_h1, dict) else raw_h1) or "NETRAL"
    m30_status = (
        ctx.get("m30_status")
        or scan.get("m30_status")
        or htf_ctx.get("m30_status")
        or m30_stage.get("status")
        or (raw_m30.get("bias") if isinstance(raw_m30, dict) else raw_m30)
        or "NETRAL"
    )
    has_conflict = bool(ctx.get("has_conflict") or htf_ctx.get("has_conflict"))

    raw_status = str(dec.get("status", "WAIT")).upper()
    raw_dir = str(dec.get("direction") or "").upper()
    if raw_status == "VALID" and raw_dir in ("BUY", "SELL"):
        decision_signal = raw_dir
        direction = raw_dir
        m5_bias = "CONFIRMED"
    elif raw_status in ("BUY", "SELL"):
        decision_signal = raw_status
        direction = raw_status
        m5_bias = "CONFIRMED"
    elif raw_status == "INVALID":
        decision_signal = "INVALID"
        direction = "NONE"
        m5_bias = "INVALID"
    else:
        decision_signal = "WAIT"
        direction = "NONE"
        m5_bias = "WAIT"

    stages = dec.get("m5_stages") or dec.get("stages") or {}
    zt = (stages.get("zone_touch") or {}).get("status", "NOT_CHECKED")
    sw = (stages.get("liquidity_sweep") or {}).get("status", "NOT_CHECKED")
    tr = (stages.get("structure_trigger") or {}).get("status", "NOT_CHECKED")
    dp = (stages.get("displacement") or {}).get("status", "NOT_CHECKED")
    rr_st = (stages.get("risk_reward") or {}).get("status", "NOT_CHECKED")
    m5_bias = tr

    active_mon_zone = dec.get("zone")
    r_prio = dec.get("radar_priority") or scan.get("radar_priority") or {}
    pz = active_mon_zone
    primary = dec.get("primary_zone") or r_prio.get("primary_directional_zone") or r_prio.get("primary_zone")
    nz = dec.get("nearest_valid_zone")
    current_price = safe_float(dec.get("current_price") or scan.get("current_price"), 0.0)
    if isinstance(pz, dict) and current_price > 0 and scan.get("data_ok") is not False and str(dec.get("data_freshness") or "").upper() not in ("STALE", "INVALID", "NO DATA"):
        lo, hi = safe_float(pz.get("price_low")), safe_float(pz.get("price_high"))
        if lo <= current_price <= hi:
            zt = "IN_ZONE"
        elif str(zt).upper() in ("CONFIRMED", "PASS", "PASSED"):
            zt = "TOUCHED_PREVIOUSLY"

    if isinstance(pz, dict) and pz.get("price_low") is not None:
        pz_label = (
            f"{pz.get('timeframe', 'H1')} {pz.get('zone_type', 'ZONE')} "
            f"{safe_float(pz.get('price_low')):.2f}-{safe_float(pz.get('price_high')):.2f} "
            f"(Grade {pz.get('grade', '-')}, Score {safe_float(pz.get('score')):.2f}, Dist: {max(safe_float(pz.get('price_low')) - current_price, current_price - safe_float(pz.get('price_high')), 0.0):.2f} pts)"
        )
        pz_id = str(pz.get("zone_id") or "NONE")
    else:
        pz_label = "NONE"
        pz_id = "NONE"

    if isinstance(nz, dict) and nz.get("price_low") is not None:
        nz_label = (
            f"{nz.get('timeframe', 'H1')} {nz.get('zone_type', 'ZONE')} "
            f"{safe_float(nz.get('price_low')):.2f}-{safe_float(nz.get('price_high')):.2f} "
            f"(Grade {nz.get('grade', '-')}, Score {safe_float(nz.get('score')):.2f}, Dist: {max(safe_float(nz.get('price_low')) - current_price, current_price - safe_float(nz.get('price_high')), 0.0):.2f} pts)"
        )
    else:
        nz_label = "NONE"

    eval_sum = scan.get("zone_evaluation_summary") or {}
    inval_list = r_prio.get("invalidated_zones") or dec.get("invalidated_zones") or []
    inval_cnt = int(eval_sum.get("total_invalidated", len(inval_list)) or 0)
    mit_cnt = int(eval_sum.get("total_mitigated", 0) or 0)
    fresh_cnt = int(eval_sum.get("total_fresh", 0) or 0)
    tested_cnt = int(eval_sum.get("total_tested", 0) or 0)

    last_inval_type = "SUPPLY"
    if inval_list and isinstance(inval_list[0], dict):
        last_inval_type = str(inval_list[0].get("zone_type", "SUPPLY")).upper()

    plan = dec.get("plan") or {}
    entry_val = safe_float(plan.get("entry_acuan", plan.get("entry")), 0.0) if decision_signal in ("BUY", "SELL") else None
    sl_val = safe_float(plan.get("sl"), 0.0) if decision_signal in ("BUY", "SELL") else None
    tp1_val = safe_float(plan.get("tp1"), 0.0) if decision_signal in ("BUY", "SELL") else None
    tp2_val = safe_float(plan.get("tp2"), 0.0) if decision_signal in ("BUY", "SELL") else None
    rr_val = safe_float(plan.get("rr"), 0.0) if decision_signal in ("BUY", "SELL") else None

    reasons_list = dec.get("reasons") or []
    if decision_signal in ("BUY", "SELL"):
        main_reason = "; ".join(str(r) for r in reasons_list if r) or f"Setup {decision_signal} SMC terkonfirmasi."
    else:
        if reasons_list:
            main_reason = str(reasons_list[0]).replace("[SCANNER] Rejection Reason: ", "")
        else:
            main_reason = scan.get("empty_reason") or "Menunggu konfirmasi struktur & trigger M5 SMC."

    struct_status = "CONFLICT H4 vs H1" if has_conflict else f"H4 {h4_bias} / H1 {h1_struct}"

    return {
        "method": "SMC",
        "status": "OK",
        "h4_bias": h4_bias,
        "h1_bias": h1_struct,
        "m30_bias": m30_status,
        "m5_bias": m5_bias,
        "structure_status": struct_status,
        "primary_zone_label": pz_label,
        "primary_zone_id": pz_id,
        "primary_setup_id": (dec.get("active_setup") or {}).get("setup_id") or "NONE",
        "scored_primary_zone": primary,
        "scored_primary_zone_label": (
            f"{primary.get('timeframe', 'N/A')} {primary.get('zone_type', 'ZONE')} "
            f"{safe_float(primary.get('price_low')):.2f}-{safe_float(primary.get('price_high')):.2f} "
            f"(Grade {primary.get('grade', '-')}, Score {safe_float(primary.get('score')):.2f})"
            if isinstance(primary, dict) else "NONE"
        ),
        "nearest_zone_label": nz_label,
        "next_radar_zone": dec.get("next_radar_zone"),
        "focus_change_reason": dec.get("focus_change_reason"),
        "fresh_count": fresh_cnt,
        "tested_count": tested_cnt,
        "mitigated_count": mit_cnt,
        "invalidated_count": inval_cnt,
        "last_invalidated_type": last_inval_type,
        "zone_touch": zt,
        "liquidity_sweep": sw,
        "choch_bos": tr,
        "displacement": dp,
        "rr_stage": rr_st,
        "decision_signal": decision_signal,
        "setup_type": str(dec.get("setup_type") or plan.get("setup_type") or ("TREND_PULLBACK" if decision_signal in ("BUY", "SELL") else "NONE")).upper(),
        "signal_status": str(dec.get("signal_status") or ("SIGNAL CONFIRMED" if decision_signal in ("BUY", "SELL") else "NO SETUP")).upper(),
        "snr_context_summary": (dec.get("snr_context") or {}).get("summary") or "Independen",
        "direction": direction,
        "entry": entry_val,
        "sl": sl_val,
        "tp1": tp1_val,
        "tp2": tp2_val,
        "rr": rr_val,
        "main_reason": main_reason,
    }


def format_smc_monitor_section(smc_cycle_result, detailed=False):
    """
    Menyusun blok `### A. SMC MONITOR` sesuai spesifikasi Part 2 Section 3:
    - Market bias H4, H1, M30, M5.
    - Status struktur pasar.
    - Zona SMC aktif dan jarak harga.
    - Status mitigasi atau invalidasi zona.
    - Status liquidity sweep, BOS, CHoCH, dan displacement jika tersedia.
    - Status konfirmasi entry.
    - Keputusan SMC: BUY / SELL / WAIT.
    - Entry, SL, TP, dan RR jika sudah valid.
    - Alasan utama keputusan.
    """
    if not isinstance(smc_cycle_result, dict):
        return "### A. SMC MONITOR\nSTATUS          : ERROR (Hasil siklus SMC tidak valid)"

    if smc_cycle_result.get("status") == "ERROR":
        return (
            "### A. SMC MONITOR\n"
            "STATUS          : ERROR\n"
            f"ERROR REASON    : {smc_cycle_result.get('error', 'Unknown SMC error')}"
        )

    s = smc_cycle_result.get("summary") or {}
    alerts = smc_cycle_result.get("alerts") or []

    lines = [
        "### A. SMC MONITOR",
        f"MARKET BIAS     : H4={s.get('h4_bias', 'N/A')} | H1={s.get('h1_bias', 'N/A')} | M30={s.get('m30_bias', 'N/A')} | M5={s.get('m5_bias', 'WAIT')}",
        f"PRIMARY ZONE    : {s.get('scored_primary_zone_label', 'NONE')}",
        f"ACTIVE SMC ZONE : {s.get('primary_zone_label', 'NONE')}",
        f"NEAREST ZONE    : {s.get('nearest_zone_label', 'NONE')}",
        f"SMC TRIGGERS    : Sweep={s.get('liquidity_sweep', 'WAIT')} | BOS/CHoCH={s.get('choch_bos', 'WAIT')} | Displacement={s.get('displacement', 'WAIT')}",
        f"ENTRY CONFIRM.  : Touch={s.get('zone_touch', 'WAIT')} | RR_Check={s.get('rr_stage', 'WAIT')}",
        f"SMC DECISION    : {s.get('decision_signal', 'WAIT')} (Direction: {s.get('direction', 'NONE')} | Setup: {s.get('setup_type', 'NONE')} | Status: {s.get('signal_status', 'NO SETUP')})",
    ]
    if s.get("focus_change_reason"):
        lines.append(f"FOCUS CHANGE    : {s['focus_change_reason']}")
    next_zone = s.get("next_radar_zone")
    if next_zone:
        lines.append(f"NEXT RADAR ZONE : {safe_float(next_zone.get('price_low')):.2f}-{safe_float(next_zone.get('price_high')):.2f}")

    decision = smc_cycle_result.get("decision") or {}
    focus = decision.get("zone") or decision.get("primary_zone") or {}
    if focus:
        allow_plan = (decision.get("stages", {}).get("DATA") or {}).get("ok") is not False
        lines.extend(format_smc_zone_lines(
            focus, decision, allow_plan=allow_plan, role="SMC ACTIVE" if decision.get("zone") else "SMC PRIMARY",
        ))
    else:
        lines.append("ENTRY / SL / TP : N/A (Tidak ada zona SMC)")

    lines.append(f"MAIN REASON     : {s.get('main_reason', 'N/A')}")

    if alerts:
        lines.append("SMC ALERTS      :")
        for al in alerts:
            lines.append(f"  ! {al}")

    if detailed and smc_cycle_result.get("scan_result") is not None and smc_cycle_result.get("decision") is not None:
        lines.append("")
        lines.append(
            format_scan_presentation(
                smc_cycle_result["scan_result"],
                smc_cycle_result["decision"],
                detailed=True,
            )
        )

    return "\n".join(lines)


def collect_account_monitor_data(symbol=None, snapshot=None, pos_list=None, orders_list=None, risk_data=None):
    """
    Mengumpulkan fakta akun MT5 secara read-only (posisi aktif, pending order,
    floating P/L, posisi/order tanpa SL, spread, kalkulasi risiko SL, dan peringatan risiko kritis).
    STRICTLY READ-ONLY: DILARANG membuka, mengubah, atau menutup order secara otomatis.
    """
    sym = symbol or DEFAULT_APP_SYMBOL
    snap_r = _extract_risk_monitor_snapshot(
        pos_list=pos_list,
        orders_list=orders_list,
        risk_data=risk_data,
    )

    positions = list(snap_r.get("positions") or [])
    orders = list(snap_r.get("orders") or [])

    # Ekstrak data akun finansial dari MT5 jika tersedia
    balance = 0.0
    equity = 0.0
    margin = 0.0
    margin_free = 0.0
    margin_level = 0.0
    cur = "USD"

    if mt5 is not None:
        try:
            acc = mt5.account_info()
            if acc is not None:
                balance = safe_float(getattr(acc, "balance", 0.0))
                equity = safe_float(getattr(acc, "equity", balance))
                margin = safe_float(getattr(acc, "margin", 0.0))
                margin_free = safe_float(getattr(acc, "margin_free", equity))
                margin_level = safe_float(getattr(acc, "margin_level", 0.0))
                if margin > 0 and margin_level <= 0.0:
                    margin_level = round((equity / margin) * 100.0, 2)
                cur = str(getattr(acc, "currency", "USD"))
        except Exception:
            pass

    floating_pnl = 0.0
    buy_lots = 0.0
    sell_lots = 0.0
    total_sl_risk_usd = 0.0
    total_sl_risk_pts = 0.0
    pos_no_sl_count = 0

    for p in positions:
        if isinstance(p, dict):
            p_prof = safe_float(p.get("profit", 0.0), 0.0)
            p_vol = safe_float(p.get("volume", 0.0), 0.0)
            p_type = p.get("type", 0)
            p_sl = safe_float(p.get("sl", 0.0), 0.0)
            p_open = safe_float(p.get("price_open", 0.0), 0.0)
        else:
            p_prof = safe_float(getattr(p, "profit", 0.0), 0.0)
            p_vol = safe_float(getattr(p, "volume", 0.0), 0.0)
            p_type = getattr(p, "type", 0)
            p_sl = safe_float(getattr(p, "sl", 0.0), 0.0)
            p_open = safe_float(getattr(p, "price_open", 0.0), 0.0)

        floating_pnl += p_prof
        is_buy = (p_type == 0 or str(p_type).upper() == "BUY")
        if is_buy:
            buy_lots += p_vol
        else:
            sell_lots += p_vol

        if p_sl > 0 and p_open > 0:
            r_pts = max(0.0, p_open - p_sl) if is_buy else max(0.0, p_sl - p_open)
            r_usd = r_pts * p_vol * 100.0  # 1 lot XAUUSD = 100 oz
            total_sl_risk_pts += r_pts
            total_sl_risk_usd += r_usd
        else:
            pos_no_sl_count += 1

    ord_no_sl_count = 0
    for o in orders:
        if isinstance(o, dict):
            o_sl = safe_float(o.get("sl", 0.0), 0.0)
            o_vol = safe_float(o.get("volume_current", o.get("volume_initial", 0.0)), 0.0)
            o_open = safe_float(o.get("price_open", 0.0), 0.0)
            o_type = o.get("type", 0)
        else:
            o_sl = safe_float(getattr(o, "sl", 0.0), 0.0)
            o_vol = safe_float(getattr(o, "volume_current", getattr(o, "volume_initial", 0.0)), 0.0)
            o_open = safe_float(getattr(o, "price_open", 0.0), 0.0)
            o_type = getattr(o, "type", 0)

        if o_sl > 0 and o_open > 0:
            is_buy_o = o_type in (0, 2, 4) or "BUY" in str(o_type).upper()
            r_pts_o = max(0.0, o_open - o_sl) if is_buy_o else max(0.0, o_sl - o_open)
            total_sl_risk_usd += r_pts_o * o_vol * 100.0
        else:
            ord_no_sl_count += 1

    spread_val = None
    if isinstance(snapshot, dict) and snapshot.get("spread") is not None:
        spread_val = safe_float(snapshot.get("spread"), 0.0)
    elif mt5 is not None:
        try:
            t_info = mt5.symbol_info_tick(sym)
            if t_info is not None and getattr(t_info, "bid", 0.0) > 0:
                spread_val = round(float(t_info.ask - t_info.bid), 2)
        except Exception:
            pass

    p_nosl = max(int(snap_r.get("position_without_sl", 0)), pos_no_sl_count)
    o_nosl = max(int(snap_r.get("pending_without_sl", 0)), ord_no_sl_count)

    risk_warnings = []
    has_unprotected = (p_nosl > 0 or o_nosl > 0)
    if has_unprotected:
        risk_warnings.append(f"[RISK WARNING] {p_nosl} posisi / {o_nosl} pending order tanpa Stop Loss! Risiko akun tidak dapat dibatasi.")
        total_risk_status = "UNBOUNDED / TIDAK DAPAT DIBATASI (Terdapat posisi/order tanpa Stop Loss)"
    elif len(positions) == 0 and len(orders) == 0:
        total_risk_status = "NO EXPOSURE (Tidak ada posisi atau pending order aktif)"
    else:
        risk_pct = (total_sl_risk_usd / equity * 100.0) if equity > 0 else 0.0
        total_risk_status = f"${total_sl_risk_usd:,.2f} ({risk_pct:.2f}% Equity | {total_sl_risk_pts:.2f} pts)"

    if spread_val is not None and spread_val >= 0.80:
        risk_warnings.append(f"WARNING: Spread saat ini melebar ({spread_val:.2f} pts).")

    return {
        "symbol": sym,
        "available": bool(snap_r.get("available", False)),
        "open_positions": len(positions),
        "pending_orders": len(orders),
        "floating_pnl": round(floating_pnl, 2),
        "balance": balance,
        "equity": equity,
        "margin": margin,
        "free_margin": margin_free,
        "margin_level": margin_level,
        "currency": cur,
        "buy_lots": round(buy_lots, 2),
        "sell_lots": round(sell_lots, 2),
        "net_lot_exposure": round(buy_lots - sell_lots, 2),
        "position_without_sl": p_nosl,
        "pending_without_sl": o_nosl,
        "total_risk_status": total_risk_status,
        "total_sl_risk_usd": round(total_sl_risk_usd, 2) if not has_unprotected else None,
        "has_unprotected": has_unprotected,
        "spread": spread_val,
        "risk_warnings": risk_warnings,
        "audit_mode": "STRICTLY_READ_ONLY",
    }


def format_account_monitor_section(acc_data):
    """
    Menyusun blok `### C. ACCOUNT MONITOR` (STRICTLY READ-ONLY):
    - Balance, Equity, Margin, Free Margin, Margin Level (%)
    - Posisi aktif & Exposure Lot (Buy lot vs Sell lot)
    - Pending order
    - Profit/loss berjalan
    - Posisi atau pending order tanpa SL (Peringatan Keras)
    - Akumulasi total risiko akun (berdasarkan SL atau UNBOUNDED)
    - Informasi spread
    - Peringatan risiko penting
    """
    acc = acc_data or {}
    spread_str = f"{safe_float(acc.get('spread')):.2f} pts" if acc.get("spread") is not None else "N/A"

    if not acc.get("available", False):
        lines = [
            "### C. ACCOUNT MONITOR (STRICTLY READ-ONLY)",
            "BALANCE / EQUITY: N/A (MT5 Offline / Mock)",
            "OPEN POSITIONS  : N/A",
            "PENDING ORDERS  : N/A",
            "FLOATING P/L    : N/A",
            "TOTAL RISK (SL) : N/A",
            "WITHOUT SL      : N/A",
            f"SPREAD          : {spread_str}",
            "RISK ALERTS     : Data posisi akun MT5 belum tersedia.",
            "AUDIT MODE      : STRICTLY READ-ONLY (Tidak ada modifikasi order otomatis)",
        ]
        return "\n".join(lines)

    warns = acc.get("risk_warnings") or []
    warn_str = "; ".join(warns) if warns else "AMAN (Seluruh posisi/order terproteksi SL)"

    bal = safe_float(acc.get("balance", 0.0))
    eq = safe_float(acc.get("equity", 0.0))
    cur = acc.get("currency", "USD")
    m_lvl = safe_float(acc.get("margin_level", 0.0))
    m_val = safe_float(acc.get("margin", 0.0))
    fm_val = safe_float(acc.get("free_margin", 0.0))
    b_lots = safe_float(acc.get("buy_lots", 0.0))
    s_lots = safe_float(acc.get("sell_lots", 0.0))
    net_lots = safe_float(acc.get("net_lot_exposure", 0.0))

    lines = [
        "### C. ACCOUNT MONITOR (STRICTLY READ-ONLY)",
        f"BALANCE / EQUITY: {bal:,.2f} {cur} / {eq:,.2f} {cur} | MARGIN LEVEL: {m_lvl:.2f}% (Margin: {m_val:,.2f} {cur} | Free: {fm_val:,.2f} {cur})",
        f"OPEN POSITIONS  : {acc.get('open_positions', 0)} (Buy: {b_lots:.2f} lot | Sell: {s_lots:.2f} lot | Net: {net_lots:+.2f} lot)",
        f"PENDING ORDERS  : {acc.get('pending_orders', 0)}",
        f"FLOATING P/L    : {acc.get('floating_pnl', 0.0):+,.2f} {cur}",
        f"TOTAL RISK (SL) : {acc.get('total_risk_status', 'N/A')}",
        f"WITHOUT SL      : Position={acc.get('position_without_sl', 0)} | Pending={acc.get('pending_without_sl', 0)}",
        f"SPREAD          : {spread_str}",
        f"RISK ALERTS     : {warn_str}",
        "AUDIT MODE      : STRICTLY READ-ONLY (Dilarang auto-close / auto-modify / auto-trade)",
    ]
    return "\n".join(lines)
    return "\n".join(lines)


def _as_monitor_zone(zone):
    if isinstance(zone, dict):
        return zone
    to_dict = getattr(zone, "to_dict", None)
    if callable(to_dict):
        converted = to_dict()
        return converted if isinstance(converted, dict) else {}
    return {}


def _calc_zone_touch(z, p, stages_dict, trk_rec, life_st):
    z_lo = safe_float(z.get("price_low"), 0.0)
    z_hi = safe_float(z.get("price_high"), 0.0)
    if not z:
        return "NO_ZONE", "NO SMC SETUP"
    if z.get("is_valid") is False or life_st in ("INVALIDATED", "BROKEN") or z.get("status") == "INVALIDATED":
        return "INVALIDATED", "INVALIDATED"
    inv_p = safe_float(z.get("invalidation_price"), 0.0)
    z_t = str(z.get("zone_type", "")).upper()
    if inv_p > 0 and p > 0:
        if z_t == "DEMAND" and p < inv_p:
            return "INVALIDATED", "INVALIDATED"
        if z_t == "SUPPLY" and p > inv_p:
            return "INVALIDATED", "INVALIDATED"
    if z_lo > 0 and z_hi > 0 and p > 0 and (z_lo <= p <= z_hi):
        return "IN_ZONE", "IN ZONE"
    t_stage = (stages_dict.get("zone_touch") if isinstance(stages_dict, dict) else {}) or {}
    t_stage_st = str(t_stage.get("status") if isinstance(t_stage, dict) else t_stage or "").upper()
    t_cnt = int(z.get("touch_count", 0))
    app = str(z.get("approach", "")).upper()
    has_touched = (
        t_stage_st in ("CONFIRMED", "PASS", "PASSED")
        or app in ("TOUCHED", "ZONE_TOUCHED")
        or life_st in ("ZONE_TOUCHED", "TOUCHED", "CONFIRMED")
        or t_cnt > 0
    )
    if not has_touched and trk_rec:
        hist = trk_rec.get("history") or []
        has_touched = any(str(h.get("status", "")).upper() in ("ZONE_TOUCHED", "TOUCHED", "CONFIRMED") for h in hist)
        if not has_touched and str(trk_rec.get("touch_status", "")).upper() in ("IN_ZONE", "TOUCHED_PREVIOUSLY"):
            has_touched = True
    if has_touched:
        return "TOUCHED_PREVIOUSLY", "PREVIOUSLY TOUCHED"
    return "NOT_TOUCHED", "NOT TOUCHED"


def _smart_monitor_context(cycle_result):
    cycle = cycle_result if isinstance(cycle_result, dict) else {}
    snapshot = cycle.get("snapshot") if isinstance(cycle.get("snapshot"), dict) else {}
    smc = cycle.get("smc") if isinstance(cycle.get("smc"), dict) else {}
    snr = cycle.get("snr") if isinstance(cycle.get("snr"), dict) else {}
    scan = smc.get("scan_result") if isinstance(smc.get("scan_result"), dict) else {}
    decision = smc.get("decision") if isinstance(smc.get("decision"), dict) else {}
    summary = smc.get("summary") if isinstance(smc.get("summary"), dict) else {}
    snr_scan = snr.get("scan_result") if isinstance(snr.get("scan_result"), dict) else {}
    snr_decision = snr.get("decision") if isinstance(snr.get("decision"), dict) else {}

    zone = _as_monitor_zone(decision.get("zone"))
    pending = decision.get("early_setup") or {}
    if not isinstance(pending, dict):
        pending = {}
    plan = pending.get("best_setup") or {}
    if not isinstance(plan, dict):
        plan = {}

    price = safe_float(snapshot.get("snapshot_price", snapshot.get("bid")), 0.0)
    lo = safe_float(zone.get("price_low"), 0.0)
    hi = safe_float(zone.get("price_high"), 0.0)
    distance = 0.0 if lo <= price <= hi and price > 0 else (
        lo - price if price > 0 and lo > price else (price - hi if price > hi else None)
    )
    zone_type = str(zone.get("zone_type") or "").upper()
    direction = str(decision.get("direction") or summary.get("direction") or "").upper()
    if direction not in ("BUY", "SELL"):
        direction = "BUY" if zone_type == "DEMAND" else ("SELL" if zone_type == "SUPPLY" else "WAIT")

    # Structured Setup Identity for Monitored Active Setup (Single Source of Truth)
    z_id = str(zone.get("zone_id") or "N/A")
    setup_type = str(decision.get("setup_type") or zone.get("setup_type") or "TREND_PULLBACK")
    now_str = format_wib_timestamp(default_now=True)
    active_setup = dict(decision.get("active_setup") or {})
    active_setup.update({
        "setup_id": active_setup.get("setup_id") or (f"{z_id}:{direction}:{setup_type}" if z_id != "N/A" else "N/A"),
        "zone_id": z_id,
        "direction": direction,
        "zone_range": (lo, hi),
        "setup_type": setup_type,
        "timeframe": str(zone.get("timeframe") or "H1"),
        "grade": str(zone.get("grade") or "N/A"),
        "score": safe_float(zone.get("score"), 0.0),
        "created_at": zone.get("created_at") or zone.get("first_detected") or now_str,
        "last_updated": zone.get("last_updated") or now_str,
        "lifecycle_status": str(zone.get("lifecycle_status") or zone.get("status") or "RADAR").upper(),
    })

    snr_bias = str((snr_scan.get("market_bias") or {}).get("overall_bias") or "").upper()
    snr_zones = snr_scan.get("ranked_zones") or {}
    snr_zone = snr_decision.get("target_zone") or {}
    snr_zone_direction = (
        str(snr_decision.get("signal")).upper() if snr_decision.get("signal") in ("BUY", "SELL")
        else "SELL" if snr_zone.get("zone_type") == "RESISTANCE"
        else "BUY" if snr_zone.get("zone_type") == "SUPPORT" else "NONE"
    )
    if not snr_zone and snr_bias == "BEARISH":
        choices = snr_zones.get("active_resistances") or []
        if choices:
            snr_zone, snr_zone_direction = choices[0], "SELL"
    elif not snr_zone and snr_bias == "BULLISH":
        choices = snr_zones.get("active_supports") or []
        if choices:
            snr_zone, snr_zone_direction = choices[0], "BUY"
    if not snr_zone:
        choices = snr_zones.get("active_supports") or []
        snr_zone = choices[0] if choices else {}
        snr_zone_direction = "BUY" if snr_zone else "NONE"
    if not snr_zone:
        choices = snr_zones.get("active_resistances") or []
        snr_zone = choices[0] if choices else {}
        snr_zone_direction = "SELL" if snr_zone else "NONE"

    def _zone_is_usable(candidate):
        if not isinstance(candidate, dict) or candidate.get("is_valid") is False or candidate.get("is_active") is False:
            return False
        status = str(candidate.get("lifecycle_status") or candidate.get("status") or "ACTIVE").upper()
        freshness_status = str(candidate.get("freshness") or candidate.get("freshness_status") or "FRESH").upper()
        return (
            status not in ("INVALIDATED", "EXPIRED", "MITIGATED", "BROKEN")
            and freshness_status not in ("INVALIDATED", "EXPIRED", "MITIGATED", "STALE", "INVALID")
        )

    smc_zone_usable = _zone_is_usable(zone)
    snr_zone_usable = _zone_is_usable(snr_zone)
    if smc.get("status") != "OK" or snr.get("status") != "OK":
        confluence = "UNAVAILABLE"
    elif not smc_zone_usable and not snr_zone_usable:
        confluence = "UNAVAILABLE"
    elif not smc_zone_usable or not snr_zone_usable or direction not in ("BUY", "SELL") or snr_zone_direction == "NONE":
        confluence = "PARTIAL"
    elif direction != snr_zone_direction:
        confluence = "CONFLICT"
    else:
        snr_lo = safe_float(snr_zone.get("low"), 0.0)
        snr_hi = safe_float(snr_zone.get("high"), 0.0)
        confluence = "ALIGNED" if max(lo, snr_lo) <= min(hi, snr_hi) else "PARTIAL"

    tracker = decision.get("zone_tracker")
    tracked = tracker.get_zone(str(zone.get("zone_id"))) if tracker and zone.get("zone_id") and hasattr(tracker, "get_zone") else None
    lifecycle = str((tracked or {}).get("lifecycle_status") or (tracked or {}).get("status") or active_setup["lifecycle_status"]).upper()
    if not lifecycle:
        lifecycle = str(zone.get("approach") or zone.get("status") or "RADAR").upper()
    state_map = {
        "RADAR": "IDLE",
        "READY": "IDLE",
        "DETECTED": "IDLE",
        "APPROACHING": "APPROACHING",
        "ZONE_TOUCHED": "IN_ZONE",
        "TOUCHED": "IN_ZONE",
        "CONFIRMED": "CONFIRMED",
        "INVALIDATED": "INVALIDATED",
        "EXPIRED": "EXPIRED",
    }
    monitor_state = state_map.get(lifecycle, lifecycle)
    if monitor_state == "IN_ZONE":
        monitor_state = "AWAITING_CONFIRMATION"

    stages = decision.get("m5_stages") or decision.get("stages") or {}
    touch_status, touch_display = _calc_zone_touch(zone, price, stages, tracked, lifecycle)

    trigger_stage = stages.get("structure_trigger") or {}
    trigger = str(
        (trigger_stage.get("status") if isinstance(trigger_stage, dict) else trigger_stage)
        or summary.get("choch_bos") or "WAIT"
    ).upper()

    snapshot_status = str(snapshot.get("snapshot_status") or snapshot.get("data_status") or "UNKNOWN").upper()
    freshness = str(snapshot.get("data_freshness") or (snapshot.get("data_freshness_info") or {}).get("status") or "UNKNOWN").upper()
    data_invalid = (
        snapshot_status in ("DATA_INCOMPLETE", "INVALID", "ERROR", "MARKET_CLOSED")
        or freshness in ("STALE", "INVALID", "NO DATA")
        or snapshot.get("data_ok") is False
        or price <= 0
        or smc.get("status") == "ERROR"
    )
    validation_price = safe_float(decision.get("snapshot_price") or decision.get("current_price"), 0.0)
    needs_revalidation = validation_price > 0 and abs(validation_price - price) > 1e-8
    if needs_revalidation:
        data_invalid = True

    # 1. Pending Plan evaluation (Section 6)
    p_status_raw = str(plan.get("status") or "WAIT").upper()
    if data_invalid:
        p_status = "WAIT"
    elif p_status_raw in ("NONE", "NOT ACTIVE"):
        p_status = "WAIT"
    elif p_status_raw == "READY":
        p_status = "READY (NOT PLACED)"
    else:
        p_status = p_status_raw

    p_zid = str(plan.get("zone_id") or "")
    p_lo = safe_float(plan.get("zone_low", plan.get("price_low")), 0.0)
    p_hi = safe_float(plan.get("zone_high", plan.get("price_high")), 0.0)
    p_dir = "BUY" if "BUY" in str(plan.get("order_type", "")).upper() else ("SELL" if "SELL" in str(plan.get("order_type", "")).upper() else "NONE")
    pending_setup = {
        "setup_id": f"{p_zid}:{p_dir}:{plan.get('setup_type', 'PENDING')}" if p_zid else "N/A",
        "zone_id": p_zid or "N/A",
        "direction": p_dir,
        "zone_range": (p_lo, p_hi),
        "setup_type": str(plan.get("setup_type") or "PENDING"),
        "timeframe": str(plan.get("timeframe") or "H1"),
        "status": p_status,
        "is_placed": False,
    }
    if p_status.startswith("READY") and p_zid and z_id != "N/A" and p_zid != z_id:
        p_label = f"[{p_lo:.2f}-{p_hi:.2f}]" if (p_lo > 0 and p_hi > 0) else f"[{p_zid}]"
        pending_display = f"{p_status} {p_label}"
    else:
        pending_display = p_status

    # 2. Confirmed Entry validation (Section 5)
    dec_status = str(decision.get("status") or "").upper()
    dec_dir = str(decision.get("direction") or "").upper()
    conf_plan = decision.get("plan") or {}

    is_valid_confirmed = (
        dec_status == "VALID"
        and dec_dir in ("BUY", "SELL")
        and not data_invalid
        and bool(conf_plan)
        and bool(zone.get("zone_id"))
        and (not conf_plan.get("zone_id") or conf_plan.get("zone_id") == zone.get("zone_id"))
        and lifecycle not in ("INVALIDATED", "EXPIRED")
        and zone.get("is_valid") is not False
    )

    c_entry = safe_float(conf_plan.get("entry_acuan", conf_plan.get("entry")), 0.0)
    is_entry_stale = False

    conf_zid = str(conf_plan.get("zone_id") or (decision.get("zone") or {}).get("zone_id") or "")
    conf_lo = safe_float((decision.get("zone") or {}).get("price_low"), 0.0)
    conf_hi = safe_float((decision.get("zone") or {}).get("price_high"), 0.0)
    confirmed_setup = {
        "setup_id": f"{conf_zid}:{dec_dir}:{decision.get('setup_type', 'CONFIRMED')}" if conf_zid else "N/A",
        "zone_id": conf_zid or "N/A",
        "direction": dec_dir if is_valid_confirmed else "NONE",
        "zone_range": (conf_lo, conf_hi),
        "setup_type": str(decision.get("setup_type") or "TREND_PULLBACK"),
        "timeframe": str((decision.get("zone") or {}).get("timeframe") or "H1"),
        "is_valid": is_valid_confirmed,
        "is_stale": is_entry_stale,
    }

    snr_sig = str(snr_decision.get("signal", "WAIT")).upper()
    snr_confirmed = (
        snr.get("status") == "OK" and snr_sig in ("BUY", "SELL")
        and snr_decision.get("entry") is not None and not data_invalid
    )
    if not is_valid_confirmed:
        if snr_confirmed:
            confirmed_status = f"{snr_sig} [SNR]"
        else:
            confirmed_status = "WAIT"
    else:
        method_tag = "[SMC + SNR]" if snr_sig == dec_dir else "[SMC]"
        if is_entry_stale:
            confirmed_status = f"{dec_dir} {method_tag} (ENTRY STALE)"
        else:
            confirmed_status = f"{dec_dir} {method_tag}"

    if confirmed_status != "WAIT" and conf_zid and z_id != "N/A" and conf_zid != z_id:
        c_label = f"[{conf_lo:.2f}-{conf_hi:.2f}]" if (conf_lo > 0 and conf_hi > 0) else f"[{conf_zid}]"
        confirmed_display = f"{confirmed_status} {c_label}"
    else:
        confirmed_display = confirmed_status

    if is_valid_confirmed and not is_entry_stale:
        monitor_state = "CONFIRMED"
    elif monitor_state == "CONFIRMED":
        monitor_state = "AWAITING_CONFIRMATION"

    # State, Reason, Next Action, and Final Signal
    if data_invalid:
        reason = "PRICE CHANGED; VALIDATION SNAPSHOT EXPIRED" if needs_revalidation else "DATA STALE / UNAVAILABLE"
        next_action, final_signal = "WAIT FRESH VALIDATION", "WAIT"
    elif is_valid_confirmed and not is_entry_stale:
        reason, next_action, final_signal = "M5 TRIGGER VALID", "CHECK ENTRY VALIDITY", confirmed_status
    elif snr_confirmed:
        reason = str(snr_decision.get("reason") or "SNR ENTRY VALID")
        next_action, final_signal = "CHECK ENTRY VALIDITY", confirmed_status
    elif is_entry_stale:
        reason = f"Entry stale: harga ({price:.2f}) telah bergerak melewati entry {c_entry:.2f}"
        next_action, final_signal = "WAIT PULLBACK", "WAIT (ENTRY STALE)"
    elif monitor_state == "INVALIDATED" or touch_status == "INVALIDATED":
        reason, next_action, final_signal = "ZONE INVALIDATED", "CANCEL SETUP", "WAIT"
    elif monitor_state == "EXPIRED":
        reason, next_action, final_signal = "SETUP EXPIRED", "WAIT NEW SETUP", "WAIT"
    elif touch_status in ("IN_ZONE", "TOUCHED_PREVIOUSLY") or monitor_state == "AWAITING_CONFIRMATION":
        reason = str((decision.get("reasons") or ["ZONE TOUCHED; M5 TRIGGER PENDING"])[0])
        next_action, final_signal = "CHECK VALIDATION", "WAIT"
    elif monitor_state == "APPROACHING":
        reason, next_action, final_signal = "APPROACHING ENTRY ZONE", "WAIT ZONE TOUCH", "WAIT"
    elif not zone:
        reason, next_action, final_signal = "NO SMC SETUP", "WAIT SETUP", "WAIT"
    else:
        reason, next_action, final_signal = "WAITING FOR ZONE PULLBACK", "WAIT APPROACH", "WAIT"
    focus_status = cycle.get("focus_status") or {}
    if "RUNNING" in focus_status.values() and not is_valid_confirmed and not snr_confirmed:
        monitor_state = "RUNNING"
        touch_status = "TOUCHED_PREVIOUSLY"
        touch_display = "ZONA TERAMBIL (POSISI COCOK)"
        focus_detail = cycle.get("focus_detail") or {}
        running_uncertain = any(
            item.get("account_status") == "RUNNING" and item.get("uncertain")
            for item in focus_detail.values()
        )
        running_with_plan = any(
            item.get("account_status") == "RUNNING" and item.get("plan_valid")
            for item in focus_detail.values()
        )
        reason = "POSISI COCOK; RIWAYAT BELUM PASTI" if running_uncertain else "POSISI COCOK OTOMATIS; PERIKSA MT5"
        next_action = "PANTAU SL/TP1 ACUAN" if running_with_plan else "PERIKSA SL/TP POSISI MT5"
        final_signal = "WAIT (MONITORING)"
    elif not is_valid_confirmed and not snr_confirmed:
        focus_detail = cycle.get("focus_detail") or {}
        touched = next(((method, item) for method, item in focus_detail.items() if item.get("touched")), None)
        historical = next(((method, item) for method, item in focus_detail.items() if item.get("historical_touch")), None)
        uncertain = next(((method, item) for method, item in focus_detail.items() if item.get("uncertain")), None)
        if touched:
            method, item = touched
            monitor_state = "ZONE_TOUCHED"
            touch_status = "TOUCHED_PREVIOUSLY"
            touch_display = f"{method} ZONE TOUCHED"
            reason = f"{method} ZONE TOUCHED; ENTRY BELUM DIKONFIRMASI"
            next_action = "PERIKSA RIWAYAT DAN MT5" if item.get("uncertain") else "PANTAU KONFIRMASI M5"
            final_signal = "WAIT"
        elif historical:
            method, item = historical
            monitor_state = "HISTORICAL_TOUCH"
            touch_status = "TOUCHED_PREVIOUSLY"
            touch_display = f"{method} TOUCHED (HISTORICAL)"
            reason = f"{method} PERNAH DISENTUH; ENTRY BELUM TERBUKTI"
            next_action = "PERIKSA RIWAYAT DAN MT5" if item.get("uncertain") else "PANTAU KUNJUNGAN BARU"
            final_signal = "WAIT"
        elif uncertain:
            monitor_state = "HISTORY_UNCERTAIN"
            touch_status = "NO_ZONE"
            touch_display = "SENTUHAN BELUM TERBUKTI"
            reason, next_action, final_signal = "FOKUS TERKUNCI: RIWAYAT BELUM PASTI", "PERIKSA RIWAYAT DAN MT5", "WAIT"

    account = cycle.get("account") if isinstance(cycle.get("account"), dict) else {}
    warnings = list(account.get("risk_warnings") or [])
    warnings.extend(cycle.get("risk_warnings") or [])
    alerts = list(cycle.get("alerts") or [])
    if warnings:
        alerts.extend(warnings)

    # Clean alert display resolution:
    if is_valid_confirmed and not is_entry_stale:
        alert_display = f"{dec_dir} CONFIRMED"
    elif is_entry_stale:
        alert_display = "ENTRY STALE"
    elif monitor_state == "INVALIDATED" or touch_status == "INVALIDATED":
        alert_display = "ZONE INVALIDATED"
    else:
        best_al = "NONE"
        for al in alerts:
            al_str = str(al).replace("\n", " ").strip()
            if any(k in al_str for k in ("CONFIRMED", "TERKONFIRMASI")):
                best_al = al_str
                break
            elif any(k in al_str for k in ("INVALID", "BATAL")):
                best_al = al_str
                break
            elif any(k in al_str for k in ("FOCUS CHANGED", "PERUBAHAN ZONA")):
                best_al = al_str
                break
            elif any(k in al_str for k in ("TOUCH", "MENYENTUH", "DISENTUH", "HARGA DEKAT", "APPROACHING")):
                best_al = al_str
                break
        if best_al == "NONE" and alerts:
            best_al = str(alerts[0]).replace("\n", " ").strip()
        if best_al.startswith("[SMC] "):
            best_al = best_al[6:].strip()
        elif best_al.startswith("[SNR] "):
            best_al = best_al[6:].strip()
        elif best_al.startswith("[ALERT] "):
            best_al = best_al[8:].strip()
        if "TERKONFIRMASI" in best_al:
            best_al = best_al.replace("TERKONFIRMASI", "CONFIRMED")
        if "PERUBAHAN ZONA FOKUS -> " in best_al:
            best_al = "FOCUS CHANGED [" + best_al.replace("PERUBAHAN ZONA FOKUS -> ", "").strip() + "]"
        alert_display = best_al if best_al else "NONE"
    return {
        "snapshot": snapshot,
        "smc": smc,
        "snr": snr,
        "scan": scan,
        "decision": decision,
        "summary": summary,
        "snr_scan": snr_scan,
        "snr_decision": snr_decision,
        "zone": zone,
        "plan": plan,
        "price": price,
        "distance": distance,
        "direction": direction,
        "confluence": confluence,
        "lifecycle": lifecycle,
        "monitor_state": monitor_state,
        "touch": touch_display,
        "touch_status": touch_status,
        "touch_display": touch_display,
        "trigger": trigger,
        "confirmed": confirmed_status,
        "confirmed_display": confirmed_display,
        "pending_status": p_status,
        "pending_display": pending_display,
        "signal": final_signal,
        "signal_display": final_signal,
        "data_invalid": data_invalid,
        "freshness": freshness,
        "reason": reason,
        "reason_display": reason,
        "next_action": next_action,
        "alerts": alerts,
        "alert_display": alert_display,
        "account": account,
        "active_setup": active_setup,
        "pending_setup": pending_setup,
        "confirmed_setup": confirmed_setup,
    }


def format_smart_monitor_report(cycle_result):
    """Format default `pantau` output in one compact, non-repeating terminal view."""
    data = _smart_monitor_context(cycle_result)
    zone = data["zone"]
    summary = data["summary"]
    snapshot = data["snapshot"]
    quote_fresh = (
        snapshot.get("data_ok") is not False and snapshot.get("live") is True
        and str(snapshot.get("data_freshness") or "").upper() not in ("STALE", "INVALID", "NO DATA")
    )
    price_txt = f"{data['price']:.2f}" if data["price"] > 0 and (not data["data_invalid"] or quote_fresh) else "N/A"
    h4 = str(summary.get("h4_bias") or "N/A").upper()
    h1 = str(summary.get("h1_bias") or "N/A").upper()
    snr_h1 = str((data["snr_scan"].get("market_bias") or {}).get("h1") or "N/A").upper()
    htf_alignment = "ALIGNED" if h1 in ("BULLISH", "BEARISH") and h1 == snr_h1 else "PARTIAL"
    m30 = str(summary.get("m30_bias") or "N/A").upper()
    m5 = data["trigger"]
    if not zone:
        zone_lines = ["DIRECTION   : N/A", "ZONE        : NO ACTIVE SMC SETUP", "GRADE       : N/A"]
    else:
        lo = safe_float(zone.get("price_low"), 0.0)
        hi = safe_float(zone.get("price_high"), 0.0)
        score_val = safe_float(zone.get("score"), 0.0)
        score_disp = f"{int(round(score_val))}" if abs(score_val - round(score_val)) < 1e-4 else f"{score_val:.1f}"
        zone_lines = [
            f"DIRECTION   : {data['direction']}",
            f"ZONE        : {lo:.2f} - {hi:.2f}",
            f"GRADE       : {zone.get('grade', 'N/A')} ({score_disp})",
        ]
    pending_disp = data.get("pending_display", data.get("pending_status", "WAIT"))
    confirmed_disp = data.get("confirmed_display", data.get("confirmed", "WAIT"))
    touch_disp = data.get("touch_display", "NOT TOUCHED")
    trigger_disp = data.get("trigger", "WAIT")
    focused_states = ("RUNNING", "ZONE_TOUCHED", "HISTORICAL_TOUCH", "HISTORY_UNCERTAIN")
    status_text = "WAIT" if data["data_invalid"] and data["monitor_state"] not in focused_states else data["monitor_state"]
    alert_text = data.get("alert_display", "NONE")
    alert_text = str(alert_text).replace("\n", " ").strip()
    if len(alert_text) > 60:
        alert_text = alert_text[:57].rstrip() + "..."
    signal = data.get("signal_display", data.get("signal", "WAIT"))
    main_reason = str(data["reason"] or summary.get("main_reason") or "N/A").replace("\n", " ").strip()
    if data["reason"] == "WAITING FOR ZONE PULLBACK" and summary.get("main_reason"):
        main_reason = str(summary["main_reason"]).replace("\n", " ").strip()
    if len(main_reason) > 60:
        main_reason = main_reason[:57].rstrip() + "..."

    smc_focus = zone or data["decision"].get("primary_zone") or {}
    smc_block = format_smc_zone_lines(
        smc_focus, data["decision"], allow_plan=not data["data_invalid"],
        role="SMC ACTIVE" if zone else "SMC PRIMARY",
    ) if smc_focus else []
    if zone:
        smc_block = smc_block[1:]

    lines = [
        "==========================================",
        "       XAUUSD SMART MONITOR v2.5",
        "==========================================",
        f"PRICE       : {price_txt}",
        f"SMC H4 / H1 : {h4} / {h1}",
        f"SNR H1      : {snr_h1} | HTF: {htf_alignment}",
        f"M30 / M5    : {m30} / {m5}",
        "------------------------------------------",
        "ACTIVE SETUP",
        "------------------------------------------",
        *zone_lines,
        *smc_block,
        f"CONFLUENCE  : {data['confluence']}",
        "ENTRY STATUS",
        f"PENDING     : {pending_disp}",
        f"CONFIRMED   : {confirmed_disp}",
        f"TOUCH       : {touch_disp}",
        f"TRIGGER     : {trigger_disp}",
        "SMART MONITOR",
        f"STATE       : {status_text}",
        f"NEXT        : {data['next_action']}",
        f"ALERT       : {alert_text}",
        "FINAL",
        f"SIGNAL      : {signal}",
        f"REASON      : {main_reason}",
        "==========================================",
    ]
    overlaps = (data["snr_scan"].get("ranked_zones") or {}).get("overlaps") or []
    snr_decision = data.get("snr_decision") or {}
    if snr_decision:
        visible_snr = {**snr_decision, "signal": "WAIT"} if data["data_invalid"] else snr_decision
        snr_entry = dict(data["snr_scan"].get("entry_validation") or {})
        if data["data_invalid"]:
            snr_entry["provisional_plan"] = {"status": "N/A", "note": "DATA NOT FRESH"}
            snr_entry["rr_calculation"] = {"checked": False, "valid": False, "rr_display": "NOT CHECKED"}
        lines.insert(lines.index("ENTRY STATUS"), format_snr_focus_plan(
            visible_snr, snr_entry, market_bias=data["snr_scan"].get("market_bias"),
        ))
        focus_validation_reason = data["snr"].get("focus_validation_reason")
        if focus_validation_reason and cycle_result.get("focus_validation_fresh"):
            lines.insert(lines.index("ENTRY STATUS"), f"SNR CHECK   : {focus_validation_reason}")
    for focus_line in cycle_result.get("focus_tracking") or []:
        lines.insert(lines.index("ENTRY STATUS"), focus_line)
    if overlaps:
        overlap = next((item for item in overlaps if item.get("entry_conflict")), overlaps[0])
        overlap_state = "CONFLICT" if overlap.get("entry_conflict") else "GEOMETRIC"
        lines.insert(lines.index("ENTRY STATUS"), f"SNR OVERLAP : {overlap['support_label']} / {overlap['resistance_label']} {overlap_state}")
    if not snapshot.get("snapshot_status") and data["price"] <= 0:
        lines[3] = "PRICE       : N/A"
    return "\n".join(lines)


def format_smart_monitor_detail(cycle_result):
    """Structured lifecycle, validation, data freshness, and blocker details."""
    data = _smart_monitor_context(cycle_result)
    zone = data["zone"]
    plan = data["plan"]
    snapshot = data["snapshot"]
    decision = data["decision"]
    summary = data["summary"]
    stages = decision.get("m5_stages") or decision.get("stages") or {}
    active_setup = data.get("active_setup") or {}
    zone_id = str(active_setup.get("zone_id") or zone.get("zone_id") or "N/A")
    setup_type = str(active_setup.get("setup_type") or decision.get("setup_type") or plan.get("setup_type") or "N/A")
    direction = data["direction"]
    setup_id = active_setup.get("setup_id") or (f"{zone_id}:{direction}:{setup_type}" if zone_id != "N/A" else "N/A")
    blockers = [str(x) for x in (decision.get("reasons") or []) if x]
    if data["data_invalid"]:
        blockers.insert(0, str(snapshot.get("error_reason") or f"Data freshness: {data['freshness']}"))
    if not blockers:
        blockers = [str(summary.get("main_reason") or "N/A")]
    pending_status = str(plan.get("status") or "WAIT").upper()
    exp = plan.get("suggested_expiration") or plan.get("expires_at") or "N/A"
    lines = [
        format_smart_monitor_report(cycle_result),
        "",
        "LIFECYCLE DETAIL",
        f"SETUP ID    : {setup_id}",
        f"ZONE STATUS : {data['lifecycle']}",
        f"MONITOR     : {data['monitor_state']}",
        f"SIGNAL      : {summary.get('signal_status', 'NO SETUP')}",
        "",
        "VALIDATION",
        f"H4 / H1     : {summary.get('h4_bias', 'N/A')} / {summary.get('h1_bias', 'N/A')}",
        f"M30         : {summary.get('m30_bias', 'N/A')}",
        f"ZONE TOUCH  : {data.get('touch_display', data['touch'])}",
        f"SWEEP       : {_stage_status(stages.get('liquidity_sweep'), summary.get('liquidity_sweep', 'WAIT'))}",
        f"STRUCTURE   : {data['trigger']}",
        f"DISPLACEMENT: {_stage_status(stages.get('displacement'), summary.get('displacement', 'WAIT'))}",
        f"R:R CHECK   : {_stage_status(stages.get('risk_reward'), summary.get('rr_stage', 'WAIT'))}",
        f"SPREAD      : {safe_float(snapshot.get('spread'), 0.0):.2f}" if snapshot.get("spread") is not None else "SPREAD      : N/A",
        f"FRESHNESS   : {data['freshness']} | CONNECTION: {snapshot.get('connection_status', 'UNKNOWN')}",
        f"LAST M5     : {(snapshot.get('last_candle_times') or {}).get('M5', snapshot.get('last_closed_m5', 'N/A'))}",
        "",
        "PENDING PLAN (NOT PLACED)",
        f"STATUS      : {pending_status}",
        f"EXPIRATION  : {exp}",
        "",
        "BLOCKERS",
    ]
    lines.extend(f"- {reason.replace(chr(10), ' ')[:180]}" for reason in blockers)
    lines.extend([
        "",
        f"SMC + SNR   : {data['confluence']} (informational only)",
        f"NEXT ACTION : {data['next_action']}",
    ])
    smc_detail = (data["smc"].get("formatted_section") or "").strip()
    snr_detail = (data["snr"].get("formatted_section") or "").strip()
    if smc_detail:
        lines.extend(["", "SMC SCANNER DETAIL", smc_detail])
    if snr_detail:
        lines.extend(["", "SNR SCANNER DETAIL", snr_detail])
    return "\n".join(lines)


def _stage_status(stage, default="WAIT"):
    if isinstance(stage, dict):
        return stage.get("status") or default
    return stage or default


def _get_m5_tick_bucket(snapshot):
    snap = snapshot if isinstance(snapshot, dict) else {}
    timestamp = snap.get("tick_time") or snap.get("server_time")
    if isinstance(timestamp, (int, float)) and timestamp > 0:
        return str(int(float(timestamp) // 300))
    if hasattr(timestamp, "timestamp"):
        try:
            return str(int(timestamp.timestamp() // 300))
        except (OverflowError, OSError, ValueError):
            pass
    if timestamp:
        raw_timestamp = str(timestamp).replace(" WIB", "").strip()
        try:
            parsed_timestamp = datetime.fromisoformat(raw_timestamp.replace("Z", "+00:00"))
            if parsed_timestamp.tzinfo is None:
                parsed_timestamp = parsed_timestamp.replace(tzinfo=timezone.utc)
            return str(int(parsed_timestamp.timestamp() // 300))
        except ValueError:
            pass
    candle_id = snap.get("current_candle_m5") or snap.get("last_closed_m5")
    if candle_id:
        raw = str(candle_id).replace(" WIB", "").strip()
        try:
            candle_time = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if candle_time.tzinfo is None:
                candle_time = candle_time.replace(tzinfo=timezone.utc)
            return str(int(candle_time.timestamp() // 300))
        except ValueError:
            return raw[:16]
    return None


def _refresh_tick_snapshot(snapshot, symbol):
    """Refresh only MT5 tick fields; closed OHLC frames stay cached between M5 closes."""
    updated = dict(snapshot) if isinstance(snapshot, dict) else {}
    try:
        tick = mt5.symbol_info_tick(symbol) if mt5 is not None else None
    except Exception as exc:
        from modules.utils import _catat_error
        _catat_error("unified_monitor_tick", exc)
        tick = None
    if tick is None:
        updated.update({
            "snapshot_status": "DATA_INCOMPLETE",
            "data_ok": False,
            "data_freshness": "NO DATA",
            "connection_status": "DISCONNECTED",
            "bid": None,
            "ask": None,
            "mid": None,
            "snapshot_price": None,
            "current_price": None,
            "error_reason": "MT5 tick tidak tersedia; menunggu koneksi/data baru",
        })
        return updated, None
    bid = safe_float(getattr(tick, "bid", None), 0.0)
    ask = safe_float(getattr(tick, "ask", None), 0.0)
    tick_time = safe_float(getattr(tick, "time", None), 0.0)
    if bid <= 0 or ask < bid or ask <= 0:
        updated.update({
            "snapshot_status": "DATA_INCOMPLETE",
            "data_ok": False,
            "data_freshness": "INVALID",
            "connection_status": "CONNECTED",
            "bid": None,
            "ask": None,
            "mid": None,
            "snapshot_price": None,
            "current_price": None,
            "error_reason": "MT5 tick Bid/Ask tidak valid",
        })
        return updated, str(int(tick_time // 300)) if tick_time > 0 else None
    mid = (bid + ask) / 2.0
    updated.update({
        "bid": bid,
        "ask": ask,
        "mid": mid,
        "spread": round(ask - bid, 4),
        "snapshot_price": mid,
        "current_price": mid,
        "tick_time": tick_time or None,
        "tick_time_msc": int(safe_float(getattr(tick, "time_msc", None))),
        "tick_bid_raw": bid,
        "tick_ask_raw": ask,
        "server_time": tick_time or updated.get("server_time"),
        "snapshot_time": format_wib_timestamp(tick_time, default_now=True, source_tz="UTC"),
        "snapshot_status": "OK",
        "data_ok": True,
        "connection_status": "CONNECTED",
        "error_reason": "",
    })
    return updated, str(int(tick_time // 300)) if tick_time > 0 else None


class UnifiedMonitor:
    """
    Pusat Monitoring Terpadu yang menjalankan SMC Monitor, SNR Monitor, dan MT5 Account Monitor
    secara simultan dalam satu siklus dengan 1x pengambilan data MT5 per siklus,
    independensi sinyal penuh, dan isolasi kegagalan (error isolation).
    """

    def __init__(self, symbol=None, smc_runner=None, snr_runner=None, snapshot_provider=None, focus_tracking=False):
        self.symbol = symbol or DEFAULT_APP_SYMBOL
        self.smc_runner = smc_runner
        self.snr_runner = snr_runner
        self.snapshot_provider = snapshot_provider or create_market_snapshot
        self.smc_alert_tracker = SmcAlertTracker()
        self.snr_engine = SnrCycleEngine(symbol=self.symbol)
        self.cycle_count = 0
        self.last_combined_signature = None
        self.zone_monitor = ZoneMonitorState(symbol=self.symbol)
        self.last_cycle_result = None
        self.last_connection_state = None
        self.last_spread_state = None
        self.last_snr_focus_touch = None
        self.focus_tracker = PantauFocusTracker(self.symbol) if focus_tracking else None

    def _run_smc_isolated(self, shared_snapshot, detailed=False):
        """Menjalankan SMC Scanner dalam blok terisolasi agar error SMC tidak menghentikan SNR."""
        try:
            if callable(self.smc_runner):
                custom_out = self.smc_runner(shared_snapshot)
                if isinstance(custom_out, tuple) and len(custom_out) == 2:
                    smc_scan, smc_dec = custom_out
                elif isinstance(custom_out, dict) and "summary" in custom_out:
                    summary = custom_out["summary"]
                    al_bundle = self.smc_alert_tracker.evaluate_alerts(summary)
                    res = {
                        "method": "SMC",
                        "status": "OK",
                        "scan_result": custom_out.get("scan_result"),
                        "decision": custom_out.get("decision"),
                        "summary": summary,
                        "alerts": al_bundle["alerts"],
                        "critical_warnings": al_bundle["critical_warnings"],
                    }
                    res["formatted_section"] = format_smc_monitor_section(res, detailed=detailed)
                    return res
                else:
                    smc_scan, smc_dec = custom_out, {}
            else:
                smc_scan = scan_xauusd_zones(symbol=self.symbol, snapshot=shared_snapshot)
                smc_dec = evaluate_trade_decision(snapshot=shared_snapshot, scan_result=smc_scan)

            summary = extract_smc_monitor_summary(smc_scan, smc_dec)
            al_bundle = self.smc_alert_tracker.evaluate_alerts(summary)

            res = {
                "method": "SMC",
                "status": "OK",
                "scan_result": smc_scan,
                "decision": smc_dec,
                "summary": summary,
                "alerts": al_bundle["alerts"],
                "critical_warnings": al_bundle["critical_warnings"],
            }
            res["formatted_section"] = format_smc_monitor_section(res, detailed=detailed)
            return res
        except Exception as exc:
            err_res = {
                "method": "SMC",
                "status": "ERROR",
                "error": f"{type(exc).__name__}: {exc}",
                "scan_result": None,
                "decision": {
                    "method": "SMC",
                    "status": "ERROR",
                    "direction": "NONE",
                    "reasons": [f"SMC Error: {exc}"],
                },
                "summary": {
                    "method": "SMC",
                    "status": "ERROR",
                    "decision_signal": "ERROR",
                    "direction": "NONE",
                    "main_reason": f"SMC Error: {exc}",
                },
                "alerts": [f"[SMC] ERROR: {exc}"],
                "critical_warnings": [],
            }
            err_res["formatted_section"] = format_smc_monitor_section(err_res, detailed=detailed)
            return err_res

    def _run_snr_isolated(self, shared_snapshot, detailed=False):
        """Menjalankan SNR Scanner dalam blok terisolasi agar error SNR tidak menghentikan SMC."""
        try:
            if callable(self.snr_runner):
                custom_snr = self.snr_runner(shared_snapshot)
                if isinstance(custom_snr, dict) and custom_snr.get("method") == "SNR" and "formatted_section" in custom_snr:
                    return custom_snr
                scan_res = custom_snr.get("scan_result", custom_snr) if isinstance(custom_snr, dict) else {}
                dec = custom_snr.get("decision", scan_res.get("decision", {})) if isinstance(custom_snr, dict) else {}
                radar = scan_res.get("radar") or {}
                pa = scan_res.get("price_action") or {}
                al_bundle = self.snr_engine.alert_tracker.evaluate_alerts(radar, dec, price_action=pa)
                payload = {
                    "method": "SNR",
                    "status": "OK",
                    "scan_result": scan_res,
                    "decision": dec,
                    "radar": radar,
                    "alerts": al_bundle["alerts"],
                    "critical_warnings": al_bundle["critical_warnings"],
                }
                payload["formatted_section"] = format_snr_monitor_section(payload, detailed=detailed)
                return payload

            live_flag = bool(shared_snapshot.get("live", False)) if isinstance(shared_snapshot, dict) else True
            res = self.snr_engine.execute_cycle(
                snapshot=shared_snapshot,
                live=live_flag,
                detailed=detailed,
            )
            return res
        except Exception as exc:
            err_payload = {
                "method": "SNR",
                "status": "ERROR",
                "error": f"{type(exc).__name__}: {exc}",
                "scan_result": None,
                "decision": {
                    "method": "SNR",
                    "signal": "ERROR",
                    "direction": "NONE",
                    "reason": f"SNR Error: {exc}",
                },
                "radar": {},
                "alerts": [f"[SNR] ERROR: {exc}"],
                "critical_warnings": [],
            }
            err_payload["formatted_section"] = format_snr_monitor_section(err_payload, detailed=detailed)
            return err_payload

    def run_cycle(
        self,
        snapshot=None,
        mode="unified",
        detailed=False,
        pos_list=None,
        orders_list=None,
        risk_data=None,
        tick_only=False,
    ):
        """
        Menjalankan satu siklus Unified Monitor:
        - Mengambil 1 snapshot data MT5 yang konsisten (jika `snapshot` belum diberikan).
        - Menjalankan SMC Monitor dan/atau SNR Monitor sesuai `mode` ("unified", "smc", "snr").
        - Menjalankan MT5 Account Monitor secara read-only.
        - Mengumpulkan alert perubahan state (`[SMC] ...`, `[SNR] ...`) tanpa menggabungkan keputusan.
        """
        self.cycle_count += 1
        mode_norm = str(mode or "unified").strip().lower()

        shared_snap = snapshot
        if shared_snap is None:
            try:
                shared_snap = self.snapshot_provider(symbol=self.symbol)
            except Exception as snap_exc:
                shared_snap = {
                    "symbol": self.symbol,
                    "snapshot_id": f"SNAP-ERR-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    "snapshot_time": format_wib_timestamp(default_now=True),
                    "frames": None,
                    "live": False,
                    "data_freshness": "NO DATA",
                    "snapshot_status": "DATA_INCOMPLETE",
                    "error_reason": f"Gagal mengambil snapshot ({type(snap_exc).__name__}: {snap_exc})",
                }

        smc_res = None
        snr_res = None
        if tick_only and self.last_cycle_result is not None:
            smc_res = self.last_cycle_result.get("smc")
            snr_res = self.last_cycle_result.get("snr")
            smc_res = dict(smc_res) if isinstance(smc_res, dict) else smc_res
            snr_res = dict(snr_res) if isinstance(snr_res, dict) else snr_res
            if isinstance(smc_res, dict):
                smc_res["alerts"] = []
            if isinstance(snr_res, dict):
                snr_res["alerts"] = []
            previous_snapshot = self.last_cycle_result.get("snapshot") or {}
            new_price = safe_float((shared_snap or {}).get("snapshot_price"), 0.0)
            quote_changed = any(
                previous_snapshot.get(key) != (shared_snap or {}).get(key)
                for key in ("snapshot_price", "bid", "ask", "spread")
            )
            if quote_changed or (shared_snap or {}).get("data_ok") is False:
                reason = (
                    str((shared_snap or {}).get("error_reason") or "Data tick tidak tersedia.")
                    if (shared_snap or {}).get("data_ok") is False
                    else "Harga atau spread berubah; menunggu validasi ulang snapshot."
                )
                if isinstance(smc_res, dict) and isinstance(smc_res.get("decision"), dict):
                    dec = dict(smc_res["decision"])
                    dec.update(status="WAIT", plan=None, signal_status="WAITING CONFIRMATION", reasons=[reason], current_price=new_price)
                    dec["early_setup"] = {"best_setup": None, "all_setups": [], "empty_reason": reason}
                    dec["zone_plans"] = {}
                    dec["stages"] = {
                        "DATA": {"ok": False, "msg": reason},
                        **{key: {"status": "NOT_CHECKED", "msg": reason} for key in (
                            "zone_validation", "zone_touch", "liquidity_sweep", "structure_trigger",
                            "displacement", "entry_validation", "risk_reward",
                        )},
                    }
                    active_setup = dict(dec.get("active_setup") or {})
                    active_setup.update(
                        zone_status="NOT_CHECKED", touch_status="NOT_CHECKED",
                        trigger_status="NOT_CHECKED", entry_status="NOT_CHECKED",
                        final_decision="WAIT", blocker=reason,
                    )
                    dec["active_setup"] = active_setup
                    smc_res["decision"] = dec
                    smc_res["summary"] = extract_smc_monitor_summary(smc_res.get("scan_result"), dec)
                    smc_res["formatted_section"] = format_smc_monitor_section(smc_res, detailed=detailed)
                if isinstance(snr_res, dict) and isinstance(snr_res.get("decision"), dict):
                    dec = dict(snr_res["decision"])
                    dec.update(signal="WAIT", direction="NONE", live_entry="WAIT", reason=reason,
                               entry=None, effective_entry=None, sl=None, tp1=None, tp2=None, rr_tp1=None, rr_tp2=None,
                               rr_display="NOT CHECKED")
                    snr_res["decision"] = dec
                    if isinstance(snr_res.get("scan_result"), dict):
                        snr_scan = dict(snr_res["scan_result"])
                        entry = dict(snr_scan.get("entry_validation") or {})
                        entry.update(
                            final_status="WAIT", buy_status="WAIT", sell_status="WAIT", trigger_status="NOT CHECKED",
                            reason=reason, rr_status="NOT CHECKED",
                            rr_calculation={"checked": False, "valid": False, "rr_display": "NOT CHECKED", "reason": reason},
                            provisional_plan={"status": "N/A", "note": "REVALIDATING"},
                        )
                        snr_scan["entry_validation"] = entry
                        snr_scan["decision"] = dec
                        snr_scan["live_entry"] = "WAIT"
                        action = dict(snr_scan.get("price_action") or {})
                        action.update(rejection_summary="CACHED", breakout_summary="CACHED", retest_summary="CACHED")
                        snr_scan["price_action"] = action
                        snr_res["scan_result"] = snr_scan
                    snr_res["formatted_section"] = format_snr_monitor_section(snr_res, detailed=detailed)
        elif mode_norm == "unified" and not callable(self.snr_runner):
            # Jalankan SNR sekali pada shared_snap lebih dulu agar hasil scan valid bisa di-share ke konteks SMC pada snapshot yang sama
            snr_res = self._run_snr_isolated(shared_snap, detailed=detailed)
            smc_snap = shared_snap
            if (
                not callable(self.smc_runner)
                and isinstance(shared_snap, dict)
                and isinstance(snr_res, dict)
                and snr_res.get("status") == "OK"
                and isinstance(snr_res.get("scan_result"), dict)
            ):
                snr_scan_obj = snr_res["scan_result"]
                snap_id_ref = shared_snap.get("snapshot_id")
                if snap_id_ref and snr_scan_obj.get("snapshot_id") == snap_id_ref and isinstance(shared_snap.get("frames"), dict):
                    try:
                        from modules.trend_following import _snr_frames_signature
                        smc_snap = dict(shared_snap)
                        smc_snap["_shared_snr_scan"] = snr_scan_obj
                        smc_snap["_shared_snr_sig"] = _snr_frames_signature(shared_snap.get("frames"))
                        smc_snap["_shared_snr_quote"] = tuple(shared_snap.get(key) for key in (
                            "snapshot_price", "bid", "ask", "spread", "tick_time", "server_time",
                            "snapshot_time", "market_status", "data_freshness",
                        ))
                    except Exception:
                        smc_snap = shared_snap
            smc_res = self._run_smc_isolated(smc_snap, detailed=detailed)
        else:
            if mode_norm in ("unified", "smc"):
                smc_res = self._run_smc_isolated(shared_snap, detailed=detailed)

            if mode_norm in ("unified", "snr"):
                snr_res = self._run_snr_isolated(shared_snap, detailed=detailed)

        if self.focus_tracker is None and isinstance(snr_res, dict) and snr_res.get("status") == "OK" and isinstance(shared_snap, dict):
            snr_decision = snr_res.get("decision") or {}
            snr_zone = snr_decision.get("target_zone") or {}
            zone_id = snr_zone.get("zone_id")
            low = safe_float(snr_zone.get("low"), 0.0)
            high = safe_float(snr_zone.get("high"), 0.0)
            price = safe_float(shared_snap.get("snapshot_price"), 0.0)
            freshness = str(shared_snap.get("data_freshness") or "").upper()
            market_status = str(shared_snap.get("market_status") or "").upper()
            lifecycle = str(snr_zone.get("lifecycle_status") or snr_zone.get("status") or "").upper()
            valid_observation = (
                zone_id and 0 < low <= high and price > 0
                and shared_snap.get("data_ok") is not False
                and freshness not in ("STALE", "INVALID", "NO DATA")
                and market_status == "OPEN"
                and lifecycle in ("ACTIVE", "TOUCHED", "SWEEP")
            )
            if valid_observation:
                inside = low <= price <= high
                previous = self.last_snr_focus_touch
                if tick_only and inside and (
                    previous is None or previous["zone_id"] != zone_id or not previous["inside"]
                ):
                    label = snr_zone.get("rank_label") or snr_zone.get("zone_type") or "ZONE"
                    snr_res["alerts"] = list(snr_res.get("alerts") or []) + [
                        f"[SNR] HARGA DI ZONA {label} ({low:.2f}-{high:.2f}); KONFIRMASI ENTRY MENUNGGU"
                    ]
                self.last_snr_focus_touch = {"zone_id": zone_id, "inside": inside}
            else:
                self.last_snr_focus_touch = None

        focus_alerts = []
        if self.focus_tracker is not None:
            positions = pos_list
            if positions is None and mt5 is not None and isinstance(shared_snap, dict) and shared_snap.get("live") is True:
                try:
                    positions = mt5.positions_get(symbol=self.symbol)
                    if positions is None:
                        from modules.utils import _catat_error
                        _catat_error("pantau_focus_positions", RuntimeError(f"MT5 positions_get gagal: {mt5.last_error()}"))
                except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                    from modules.utils import _catat_error
                    _catat_error("pantau_focus_positions", exc)
                    positions = None
            focus_alerts = self.focus_tracker.update(
                {"SMC": smc_res, "SNR": snr_res}, shared_snap, positions, full_refresh=not tick_only,
            )
            smc_res = self.focus_tracker.gate("SMC", smc_res, detailed=detailed)
            snr_res = self.focus_tracker.gate("SNR", snr_res, detailed=detailed)
            smc_res = self.focus_tracker.scoped_alerts("SMC", smc_res, detailed=detailed)
            snr_res = self.focus_tracker.scoped_alerts("SNR", snr_res, detailed=detailed)

        acc_res = collect_account_monitor_data(
            symbol=self.symbol,
            snapshot=shared_snap,
            pos_list=positions if self.focus_tracker is not None else pos_list,
            orders_list=orders_list,
            risk_data=risk_data,
        )
        acc_section = format_account_monitor_section(acc_res)

        combined_alerts = []
        combined_alerts.extend(focus_alerts)
        if smc_res and smc_res.get("alerts"):
            combined_alerts.extend(smc_res["alerts"])
        if snr_res and snr_res.get("alerts"):
            combined_alerts.extend(snr_res["alerts"])

        zone_transitions = []
        if mode_norm in ("unified", "smc") and isinstance(smc_res, dict) and smc_res.get("status") == "OK":
            scan_data = smc_res.get("scan_result") or {}
            decision_data = smc_res.get("decision") or {}
            zones, _ = collect_radar_watchlist_with_reconciliation(scan_data, decision_data)
            if not self.zone_monitor.initialized:
                self.zone_monitor.initialize_baseline(zones, shared_snap)
            elif isinstance(shared_snap, dict):
                zone_transitions = self.zone_monitor.update(zones, shared_snap)
            for transition in zone_transitions:
                focus = self.focus_tracker.focus["SMC"] if self.focus_tracker is not None else None
                if self.focus_tracker is not None and (not focus or transition.get("zone_id") != focus["zone_id"]):
                    continue
                combined_alerts.append(
                    f"[ZONE:{transition['zone_id']}] {transition['prev_status']} -> {transition['new_status']}"
                )
            if shared_snap.get("live") is True:
                from modules.zone_scanner import get_zone_tracker
                tracker = get_zone_tracker(auto_save=True)
                if not tick_only:
                    tracker.save()
            if tick_only and self.last_cycle_result is not None:
                zone_tracker = decision_data.get("zone_tracker")
                if zone_tracker is not None and isinstance(shared_snap, dict):
                    try:
                        frames = shared_snap.get("frames") or {}
                        dec = decision_data
                        ctx = dec.get("context") or scan_data.get("htf_context") or {}
                        tracker_alerts = zone_tracker.update(
                            zones,
                            safe_float(shared_snap.get("snapshot_price"), 0.0),
                            frames=frames,
                            htf_bias=ctx.get("H4") if isinstance(ctx, dict) else None,
                            m30_status=(dec.get("stages") or {}).get("M30", {}).get("status"),
                            server_time=shared_snap.get("server_time"),
                            conservative_dec=None,
                        )
                        combined_alerts.extend(
                            f"[{str(alert.get('type') or 'ZONE')}:{alert.get('zone_id', 'N/A')}] {alert.get('msg', '')}"
                            for alert in tracker_alerts
                            if self.focus_tracker is None or (
                                self.focus_tracker.focus["SMC"]
                                and alert.get("zone_id") == self.focus_tracker.focus["SMC"]["zone_id"]
                            )
                        )
                    except (AttributeError, TypeError, ValueError, KeyError) as tracker_exc:
                        from modules.utils import _catat_error
                        _catat_error("unified_monitor_zone_tracker", tracker_exc)

        snap_status = str((shared_snap or {}).get("snapshot_status") or "").upper()
        conn_status = str((shared_snap or {}).get("connection_status") or "").upper()
        current_connection = (
            "DISCONNECTED"
            if snap_status in ("DATA_INCOMPLETE", "INVALID") or conn_status == "DISCONNECTED"
            else ("CONNECTED" if (shared_snap or {}).get("live") is True or conn_status == "CONNECTED" else "UNKNOWN")
        )
        if current_connection != self.last_connection_state and current_connection != "UNKNOWN":
            if current_connection == "DISCONNECTED":
                combined_alerts.append("[DATA] MT5 DISCONNECTED")
            elif self.last_connection_state == "DISCONNECTED":
                combined_alerts.append("[DATA] MT5 RECONNECTED")
        self.last_connection_state = current_connection
        spread = safe_float((shared_snap or {}).get("spread"), -1.0)
        spread_state = spread >= 0.80 if spread >= 0 else None
        if spread_state is True and self.last_spread_state is not True:
            combined_alerts.append(f"[RISK] SPREAD TOO HIGH ({spread:.2f})")
        elif spread_state is False and self.last_spread_state is True:
            combined_alerts.append("[RISK] SPREAD NORMAL")
        if spread_state is not None:
            self.last_spread_state = spread_state

        deduped_alerts = []
        seen_alerts = set()
        for al in combined_alerts:
            if al not in seen_alerts:
                seen_alerts.add(al)
                deduped_alerts.append(al)
        combined_alerts = deduped_alerts

        smc_sig = (smc_res.get("summary") or {}).get("decision_signal", "N/A") if smc_res else "OFF"
        snr_sig = (snr_res.get("decision") or {}).get("signal", "N/A") if snr_res else "OFF"

        curr_sig_tuple = (
            smc_sig,
            snr_sig,
            (smc_res.get("summary") or {}).get("primary_zone_label") if smc_res else None,
            (snr_res.get("decision") or {}).get("zone_label") if snr_res else None,
            smc_res.get("status") if smc_res else None,
            snr_res.get("status") if snr_res else None,
        )
        state_changed = (
            self.last_combined_signature is None
            or curr_sig_tuple != self.last_combined_signature
            or bool(combined_alerts)
        )
        self.last_combined_signature = curr_sig_tuple

        price_disp = safe_float(shared_snap.get("snapshot_price", shared_snap.get("bid", 0.0)), 0.0) if isinstance(shared_snap, dict) else 0.0
        if price_disp <= 0 and snr_res and isinstance(snr_res.get("scan_result"), dict):
            price_disp = safe_float(snr_res["scan_result"].get("price"), 0.0)

        ts_str = (
            shared_snap.get("snapshot_time")
            or (format_wib_timestamp(shared_snap.get("server_time"), default_now=False) if shared_snap.get("server_time") else "")
            if isinstance(shared_snap, dict)
            else ""
        ) or format_wib_timestamp(default_now=True)

        header_title = (
            "XAUUSD UNIFIED MONITOR — SMC + SNR + ACCOUNT"
            if mode_norm == "unified"
            else (
                "XAUUSD SMC MONITOR"
                if mode_norm == "smc"
                else "XAUUSD SNR MONITOR"
            )
        )

        comparison = compare_scanner_snapshots(
            smc_scan=(smc_res.get("scan_result") or smc_res.get("summary")) if isinstance(smc_res, dict) else None,
            smc_dec=smc_res.get("decision") if isinstance(smc_res, dict) else None,
            snr_scan=snr_res.get("scan_result") if isinstance(snr_res, dict) else None,
            shared_snap=shared_snap,
            snr_dec=snr_res.get("decision") if isinstance(snr_res, dict) else None,
        )
        mkt_ctx = comparison.get("market_context") or {}
        snap_id_disp = comparison.get("snapshot_id") or (shared_snap.get("snapshot_id") if isinstance(shared_snap, dict) else None) or "N/A"
        same_snap_disp = comparison.get("same_snapshot", False)
        p_diff_disp = safe_float(comparison.get("price_diff_points"), 0.0)
        fresh_smc_disp = comparison.get("smc_data_freshness", "UNKNOWN")
        fresh_snr_disp = comparison.get("snr_data_freshness", "UNKNOWN")
        bias_agree_disp = comparison.get("bias_agreement", "N/A")
        sig_agree_disp = comparison.get("signal_agreement", "N/A")

        price_txt = f"{price_disp:.2f}" if price_disp > 0 else "DATA UNAVAILABLE"
        report_lines = [
            "====================================================",
            f"   {header_title}",
            "====================================================",
            f"SYMBOL : {self.symbol} | PRICE : {price_txt} | TIME : {ts_str} | CYCLE : #{self.cycle_count}",
            f"SNAPSHOT ID : {snap_id_disp} | SAME={same_snap_disp} | DIFF={p_diff_disp:.2f} pts",
            f"FRESHNESS   : SMC={fresh_smc_disp} | SNR={fresh_snr_disp}",
        ]
        if mode_norm == "unified":
            report_lines.append(f"INDEPENDENT SIGNALS : SMC = {smc_sig} | SNR = {snr_sig}")
            report_lines.append(
                f"SIGNAL COMPARISON   : Overall Bias={comparison.get('smc_bias', 'N/A')} vs {comparison.get('snr_bias', 'N/A')} ({bias_agree_disp}) | "
                f"Signal={smc_sig} vs {snr_sig} ({sig_agree_disp})"
            )
            report_lines.append(
                f"H1 STRUCTURE        : SMC={comparison.get('smc_h1', 'N/A')} | "
                f"SNR={comparison.get('snr_h1', 'N/A')} ({comparison.get('h1_agreement', 'UNCONFIRMED')})"
            )
            if mkt_ctx and detailed:
                report_lines.extend([
                    f"MARKET CONTEXT      : {mkt_ctx.get('classification', 'N/A')}",
                    f"  PRIMARY TREND     : {mkt_ctx.get('primary_trend', 'UNCONFIRMED')} (H1={mkt_ctx.get('trend_confirmation', 'UNCONFIRMED')})",
                    f"  SHORT TERM        : {mkt_ctx.get('short_term', 'NEUTRAL')}",
                    f"  SNR CONTEXT       : {mkt_ctx.get('snr_context', 'N/A')}",
                    f"  SMC CONTEXT       : {mkt_ctx.get('smc_context', 'N/A')}",
                    f"  TRADE BIAS        : {mkt_ctx.get('trade_bias', 'NEUTRAL / WAIT')}",
                    f"  ENTRY STATUS      : {mkt_ctx.get('entry_status', 'WAIT')}",
                ])
        report_lines.append("----------------------------------------------------")

        if smc_res is not None:
            report_lines.append(smc_res["formatted_section"])
            report_lines.append("----------------------------------------------------")

        if snr_res is not None:
            report_lines.append(snr_res["formatted_section"])
            report_lines.append("----------------------------------------------------")

        if self.focus_tracker is not None:
            report_lines.extend(self.focus_tracker.lines())
        report_lines.append(acc_section)

        if combined_alerts:
            report_lines.append("----------------------------------------------------")
            report_lines.append("ACTIVE STATE-CHANGE ALERTS:")
            for al in combined_alerts:
                report_lines.append(f"  {al}")

        report_lines.append("====================================================")
        formatted_report = "\n".join(report_lines)

        cycle_result = {
            "cycle": self.cycle_count,
            "mode": mode_norm,
            "detailed": bool(detailed),
            "snapshot": shared_snap,
            "snapshot_id": shared_snap.get("snapshot_id") if isinstance(shared_snap, dict) else None,
            "snapshot_time": ts_str,
            "smc": smc_res,
            "snr": snr_res,
            "account": acc_res,
            "smc_signal": smc_sig,
            "snr_signal": snr_sig,
            "market_context": mkt_ctx,
            "comparison": comparison,
            "alerts": combined_alerts,
            "risk_warnings": acc_res.get("risk_warnings") or [],
            "state_changed": state_changed,
            "formatted_report": formatted_report,
            "focus_tracking": self.focus_tracker.lines() if self.focus_tracker is not None else [],
            "focus_status": {
                method: record["account_status"]
                for method, record in self.focus_tracker.focus.items()
                if record and method in self.focus_tracker.active_methods
            } if self.focus_tracker is not None else {},
            "focus_detail": {
                method: {
                    "touched": record["touched"],
                    "historical_touch": bool(record.get("historical_touch")),
                    "account_status": record["account_status"],
                    "uncertain": record["uncertain"],
                    "plan_valid": self.focus_tracker._valid_plan(record.get("plan")),
                }
                for method, record in self.focus_tracker.focus.items()
                if record and method in self.focus_tracker.active_methods
            } if self.focus_tracker is not None else {},
            "focus_validation_fresh": not tick_only,
        }
        cycle_result["compact_report"] = format_smart_monitor_report(cycle_result)
        cycle_result["detail_report"] = format_smart_monitor_detail(cycle_result)
        self.last_cycle_result = cycle_result
        return cycle_result


def compare_scanner_snapshots(smc_scan, smc_dec, snr_scan, shared_snap=None, snr_dec=None):
    """
    Membandingkan output SMC dan SNR pada snapshot pasar yang sama tanpa menggabungkan
    skor, tanpa ranking pemenang, dan tanpa memaksa kedua metodologi memberi sinyal identik.
    Juga membangun lapisan Market Context terpadu dari kedua hasil independen.
    """
    snap = shared_snap if isinstance(shared_snap, dict) else {}
    s_smc = smc_scan if isinstance(smc_scan, dict) else {}
    d_smc = smc_dec if isinstance(smc_dec, dict) else {}
    s_snr = snr_scan if isinstance(snr_scan, dict) else {}
    d_snr = snr_dec if isinstance(snr_dec, dict) else (s_snr.get("decision") if isinstance(s_snr.get("decision"), dict) else {})

    snap_id = snap.get("snapshot_id") or s_smc.get("snapshot_id") or s_snr.get("snapshot_id")
    snap_time = snap.get("snapshot_time") or s_smc.get("snapshot_time") or s_snr.get("snapshot_time") or ""

    smc_price = safe_float(s_smc.get("snapshot_price") or s_smc.get("current_price") or snap.get("snapshot_price"), 0.0)
    snr_price = safe_float(s_snr.get("snapshot_price") or s_snr.get("price") or snap.get("snapshot_price"), 0.0)
    price_diff = round(abs(smc_price - snr_price), 4) if (smc_price > 0 and snr_price > 0) else 0.0

    smc_bias = str(
        (s_smc.get("htf_context") or {}).get("H4")
        or s_smc.get("htf_bias")
        or "NETRAL"
    ).upper()
    snr_mb = s_snr.get("market_bias") or {}
    snr_bias = str(snr_mb.get("overall_bias") or "UNCONFIRMED").upper()
    smc_h1 = str((s_smc.get("htf_context") or {}).get("H1") or "UNCONFIRMED").upper().replace("NETRAL", "NEUTRAL")
    snr_h1 = str(snr_mb.get("h1") or "UNCONFIRMED").upper().replace("NETRAL", "NEUTRAL")
    h1_agreement = (
        "ALIGNED" if smc_h1 == snr_h1 and smc_h1 in ("BULLISH", "BEARISH")
        else "DIVERGENT" if smc_h1 in ("BULLISH", "BEARISH", "NEUTRAL") and snr_h1 in ("BULLISH", "BEARISH", "NEUTRAL") and smc_h1 != snr_h1
        else "UNCONFIRMED"
    )

    smc_raw_st = str(d_smc.get("status") or "WAIT").upper()
    smc_dir = str(d_smc.get("direction") or "").upper()
    smc_sig = smc_dir if (smc_raw_st == "VALID" and smc_dir in ("BUY", "SELL")) else (smc_raw_st if smc_raw_st in ("BUY", "SELL") else "WAIT")
    snr_sig = str((s_snr.get("decision") or {}).get("signal") or s_snr.get("live_entry") or "WAIT").upper()

    bias_same = (smc_bias == snr_bias) or (smc_bias == "NETRAL" and snr_bias == "SIDEWAYS")
    sig_same = smc_sig == snr_sig

    if bias_same:
        bias_note = f"Bias selaras ({smc_bias})."
    else:
        bias_note = (
            f"Perbedaan metodologi wajar: SMC H4 berbasis struktur BOS/CHoCH/OB ({smc_bias}) "
            f"sedangkan SNR berbasis swing/dominasi/reaksi level S&R ({snr_bias})."
        )

    try:
        from modules.trend_following import build_unified_market_context
        mkt_ctx = build_unified_market_context(
            smc_scan=s_smc,
            smc_dec=d_smc,
            snr_scan=s_snr,
            snr_dec=d_snr,
            snapshot=snap,
        )
    except Exception:
        mkt_ctx = {}

    smc_snap_id = s_smc.get("snapshot_id")
    snr_snap_id = s_snr.get("snapshot_id")
    smc_time = str(s_smc.get("snapshot_time") or "").strip()
    snr_time = str(s_snr.get("snapshot_time") or "").strip()
    base_time = str(snap.get("snapshot_time") or "").strip()

    def _same_snap_time(ta, tb):
        if not ta or not tb:
            return True
        ca = ta.replace(" WIB", "").strip()
        cb = tb.replace(" WIB", "").strip()
        if ca == cb:
            return True
        # Izinkan kecocokan menit apabila salah satu string snapshot_time tanpa detik (YYYY-MM-DD HH:MM vs HH:MM:SS)
        if len(ca) == 16 or len(cb) == 16:
            return ca[:16] == cb[:16]
        return False

    time_consistent = (
        _same_snap_time(base_time, smc_time)
        and _same_snap_time(base_time, snr_time)
        and _same_snap_time(smc_time, snr_time)
    )

    same_snapshot = bool(
        snap_id
        and smc_snap_id == snap_id
        and snr_snap_id == snap_id
        and price_diff <= 0.01
        and time_consistent
    )

    return {
        "snapshot_id": snap_id,
        "smc_snapshot_id": smc_snap_id,
        "snr_snapshot_id": snr_snap_id,
        "snapshot_time": snap_time,
        "same_snapshot": same_snapshot,
        "smc_price": smc_price,
        "snr_price": snr_price,
        "price_diff_points": price_diff,
        "smc_data_freshness": s_smc.get("data_freshness") or snap.get("data_freshness") or "UNKNOWN",
        "snr_data_freshness": s_snr.get("data_freshness") or snap.get("data_freshness") or "UNKNOWN",
        "smc_bias": smc_bias,
        "snr_bias": snr_bias,
        "bias_agreement": "SAME" if bias_same else "DIFFERENT",
        "smc_h1": smc_h1,
        "snr_h1": snr_h1,
        "h1_agreement": h1_agreement,
        "smc_signal": smc_sig,
        "snr_signal": snr_sig,
        "signal_agreement": "SAME" if sig_same else "DIFFERENT",
        "methodology_note": bias_note,
        "market_context": mkt_ctx,
    }


def parse_monitor_command(cmd_str):
    """
    Mem-parsing command pemantauan secara presisi agar tidak salah mengenali command yang mirip:
    - `pantau` -> mode="unified", detailed=False
    - `pantau detail` -> mode="unified", detailed=True
    - `pantau smc` -> mode="smc", detailed=False
    - `pantau smc detail` -> mode="smc", detailed=True
    - `pantau snr` -> mode="snr", detailed=False
    - `pantau snr detail` -> mode="snr", detailed=True
    - `pantau 3900 3905` -> mode="legacy_manual", detailed=False
    """
    raw = str(cmd_str or "").strip().lower()
    tokens = raw.split()
    if not tokens or tokens[0] not in ("pantau", "watch"):
        return {"valid": False, "mode": "none", "detailed": False}

    rest = tokens[1:]
    if not rest:
        return {"valid": True, "mode": "unified", "detailed": False}

    if rest == ["detail"]:
        return {"valid": True, "mode": "unified", "detailed": True}

    if rest == ["smc"]:
        return {"valid": True, "mode": "smc", "detailed": False}

    if rest in (["smc", "detail"], ["detail", "smc"]):
        return {"valid": True, "mode": "smc", "detailed": True}

    if rest == ["snr"]:
        return {"valid": True, "mode": "snr", "detailed": False}

    if rest in (["snr", "detail"], ["detail", "snr"]):
        return {"valid": True, "mode": "snr", "detailed": True}

    # Jika pengguna memasukkan angka harga manual (misal `pantau 3900 3905`), arahkan ke mode_pantau_mt5 lama
    if any(re.search(r"\d", tok) for tok in rest):
        return {"valid": True, "mode": "legacy_manual", "detailed": "detail" in rest}

    return {"valid": True, "mode": "unified", "detailed": "detail" in rest}


def run_unified_pantau(
    cmd_str="pantau",
    teks_ai_terakhir="",
    max_cycles=None,
    sleep_sec=None,
    snapshot=None,
    smc_runner=None,
    snr_runner=None,
):
    """
    Entrypoint utama untuk menjalankan command:
    - `pantau`
    - `pantau detail`
    - `pantau smc`
    - `pantau smc detail`
    - `pantau snr`
    - `pantau snr detail`
    """
    parsed = parse_monitor_command(cmd_str)
    mode = parsed["mode"]
    detailed = parsed["detailed"]

    if mode == "legacy_manual":
        return mode_pantau_mt5(cmd_str, teks_ai_terakhir)

    monitor = UnifiedMonitor(
        symbol=DEFAULT_APP_SYMBOL,
        smc_runner=smc_runner,
        snr_runner=snr_runner,
        focus_tracking=True,
    )

    cycles_done = 0
    last_out = None
    last_full_refresh = 0.0
    last_m5_bucket = None
    last_risk_signature = None
    last_risk_emit = 0.0
    status_bar_active = False
    try:
        while True:
            tick_only = False
            cycle_snapshot = snapshot
            now_mono = time.monotonic()
            if snapshot is None and cycles_done and last_out:
                refreshed, m5_bucket = _refresh_tick_snapshot(last_out.get("snapshot"), monitor.symbol)
                refresh_due = (
                    m5_bucket is not None
                    and m5_bucket != last_m5_bucket
                ) or (now_mono - last_full_refresh >= PANTAU_FULL_REFRESH_SECONDS)
                if refresh_due:
                    cycle_snapshot = None
                else:
                    cycle_snapshot = refreshed
                    tick_only = True
            cycle_out = monitor.run_cycle(
                snapshot=cycle_snapshot,
                mode=mode,
                detailed=detailed,
                tick_only=tick_only,
            )
            if not tick_only:
                last_full_refresh = now_mono
                last_m5_bucket = _get_m5_tick_bucket(cycle_out.get("snapshot"))
            if monitor.focus_tracker is not None and monitor.focus_tracker.refresh_required:
                last_full_refresh = 0.0
            last_out = cycle_out
            cycles_done += 1

            # Cetak laporan penuh pada siklus pertama, atau cetak alert/laporan saat ada perubahan state
            output = cycle_out.get("detail_report") if detailed and mode == "unified" else (
                cycle_out.get("compact_report") if mode == "unified" else cycle_out["formatted_report"]
            )
            if cycles_done == 1:
                for al in cycle_out.get("alerts") or []:
                    print(al)
                print("\n" + output)
                if cycle_out.get("risk_warnings"):
                    last_risk_signature = tuple(cycle_out["risk_warnings"])
                    last_risk_emit = time.monotonic()
            elif cycle_out.get("state_changed"):
                if mode == "unified" and status_bar_active:
                    print()
                    status_bar_active = False
                if cycle_out.get("alerts"):
                    for al in cycle_out["alerts"]:
                        print(al)
                print("\n" + output)
                if cycle_out.get("risk_warnings"):
                    last_risk_signature = tuple(cycle_out["risk_warnings"])
                    last_risk_emit = time.monotonic()
            elif cycle_out.get("risk_warnings"):
                risk_signature = tuple(cycle_out["risk_warnings"])
                if risk_signature != last_risk_signature or time.monotonic() - last_risk_emit >= 60.0:
                    if mode == "unified" and status_bar_active:
                        print()
                        status_bar_active = False
                    for rw in cycle_out["risk_warnings"]:
                        print(f"[ACCOUNT RISK] {rw}")
                    last_risk_signature = risk_signature
                    last_risk_emit = time.monotonic()
            elif mode == "unified":
                view = _smart_monitor_context(cycle_out)
                price_bar = f"{view['price']:.2f}" if view["price"] > 0 and not view["data_invalid"] else "N/A"
                status_bar = (
                    f"[SMART MONITOR] {price_bar} | {view['monitor_state']} | "
                    f"NEXT {view['next_action']} | SIGNAL {view['confirmed']}"
                )
                bar_width = max(40, shutil.get_terminal_size((100, 20)).columns - 1)
                print("\r" + status_bar[:bar_width].ljust(bar_width), end="", flush=True)
                status_bar_active = True

            if max_cycles is not None and cycles_done >= int(max_cycles):
                break
            if sleep_sec is None:
                monitor_state = _smart_monitor_context(cycle_out).get("monitor_state", "IDLE")
                wait_seconds = PANTAU_INTERVALS_SECONDS.get(monitor_state, 5.0)
            else:
                monitor_state = _smart_monitor_context(cycle_out).get("monitor_state", "IDLE")
                wait_seconds = min(float(sleep_sec), PANTAU_INTERVALS_SECONDS.get(monitor_state, 5.0))
            time.sleep(max(0.1, float(wait_seconds)))
    except KeyboardInterrupt:
        if status_bar_active:
            print()
        print("\n[UNIFIED MONITOR] Pemantauan dihentikan oleh pengguna.")

    if status_bar_active and max_cycles is not None and cycles_done >= int(max_cycles):
        print()
    return last_out


def run_unified_scan(
    symbol=None,
    snapshot=None,
    detailed=False,
    smc_runner=None,
    snr_runner=None,
    snapshot_provider=None,
):
    """
    Menjalankan scan gabungan SMC + SNR 1 siklus (one-shot) menggunakan mekanisme
    shared snapshot milik UnifiedMonitor tanpa duplikasi pengambilan data.
    """
    mon = UnifiedMonitor(
        symbol=symbol or DEFAULT_APP_SYMBOL,
        smc_runner=smc_runner,
        snr_runner=snr_runner,
        snapshot_provider=snapshot_provider,
    )
    return mon.run_cycle(snapshot=snapshot, mode="unified", detailed=detailed)
