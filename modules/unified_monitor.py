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
from modules.terminal_ui import print_terminal
from modules.utils import format_wib_timestamp
from modules.mt5_conn import _acceptable_trade_symbols, _info_tipe_pending, pastikan_mt5_aktif
from modules.pantau import (
    ZoneMonitorState,
    collect_radar_watchlist_with_reconciliation,
    mode_pantau_mt5,
)
from modules.snr.data_provider import safe_float
from modules.snr.entry_validator import select_live_snr_focus
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
        self._recently_completed = {"SMC": None, "SNR": None}
        self._last_lifecycle_event = {}
        self.active_methods = set()
        self.refresh_required = False
        self._saved_state = None
        self._last_tick = None
        self._history_error_active = False
        self._load()

    def _emit_lifecycle_alert(self, alerts, method, kind, key, msg):
        sig = (method, kind)
        if self._last_lifecycle_event.get(sig) == key:
            return False
        self._last_lifecycle_event[sig] = key
        alerts.append(msg)
        return True

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
                    if not record.get("ever_taken") and not self._valid_plan(record.get("plan")):
                        record["touched"] = False
                    record["uncertain"] = bool(record.get("ever_taken") or record.get("touched"))
                    record["account_status"] = str(record.get("account_status") or "")
                    record.setdefault("pending_ticket", None)
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
                key: zone.get(key) for key in (
                    "zone_id", "zone_type", "timeframe", "rank_label", "grade", "score", "source",
                    "freshness", "snapshot_id", "snapshot_time",
                    "confirmed_time", "confirmation_time", "formation_time", "formation_timeframe",
                    "risk_tier", "distance", "distance_points",
                    "invalidation_price", "setup_type", "source_kind", "source_zone_id",
                    "lifecycle_status", "status",
                )
                if key in zone and (zone.get(key) is None or isinstance(zone.get(key), (str, int, float, bool)))
            }
            if isinstance(zone.get("merged_from"), list):
                saved["zone"]["merged_from"] = [str(m) for m in zone["merged_from"] if m]
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
            zone = dec.get("zone") or {}
            if not zone:
                pz = dec.get("primary_zone")
                if isinstance(pz, dict) and pz.get("risk_tier") != "FAR_ZONE":
                    zone = pz
            if not zone:
                priority = dec.get("radar_priority") or scan.get("radar_priority") or {}
                zone = priority.get("primary_directional_zone") or priority.get("primary_zone") or {}
                if not zone:
                    near_cand = priority.get("nearest_radar_zone")
                    h4_bias = str((scan.get("htf_context") or {}).get("H4") or "").upper()
                    if isinstance(near_cand, dict) and near_cand.get("risk_tier") != "FAR_ZONE":
                        zt = str(near_cand.get("zone_type") or "").upper()
                        aligned_ok = (
                            h4_bias not in ("BULLISH", "BEARISH")
                            or (not near_cand.get("is_counter") and zt == ("DEMAND" if h4_bias == "BULLISH" else "SUPPLY"))
                        )
                        if aligned_ok:
                            zone = near_cand
            summary = smc_zone_summary(zone, dec) if zone else {}
            plan = {
                "zone_id": zone.get("zone_id"),
                "direction": summary.get("direction"),
                "entry": summary.get("entry"),
                "sl": summary.get("sl"),
                "tp1": summary.get("tp1"),
                "tp2": summary.get("tp2"),
                "rr": summary.get("rr"),
                "kind": summary.get("kind"),
                "note": summary.get("note"),
            } if summary.get("kind") in ("CONFIRMED", "SELECTED_PENDING", "ALT_PENDING", "PENDING", "PROVISIONAL") else None
        else:
            zone = dec.get("target_zone") or {}
            ev = scan.get("entry_validation") or {}
            provisional = ev.get("provisional_plan") or {}
            if dec.get("signal") in ("BUY", "SELL"):
                plan = {
                    "zone_id": zone.get("zone_id"),
                    "direction": dec.get("signal"), "entry": dec.get("effective_entry"),
                    "sl": dec.get("sl"), "tp1": dec.get("tp1"),
                }
            elif provisional.get("status") == "PROVISIONAL":
                plan = {
                    "zone_id": zone.get("zone_id"),
                    **{key: provisional.get(key) for key in ("direction", "entry", "sl", "tp1")},
                }
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
    def _current_zone(method, result, zone_id, record=None):
        res = result or {}
        scan = res.get("scan_result") or {}
        dec = res.get("decision") or {}
        if method == "SMC":
            tf_block = dec.get("trend_following") if isinstance(dec.get("trend_following"), dict) else {}
            r_prio = dec.get("radar_priority") or scan.get("radar_priority") or {}
            live_pool = []
            for single in (
                dec.get("zone"),
                tf_block.get("best_candidate"),
                dec.get("primary_zone"),
                dec.get("nearest_valid_zone"),
                dec.get("next_radar_zone"),
            ):
                if isinstance(single, dict) and single.get("zone_id"):
                    live_pool.append(single)
            for group in (
                scan.get("active_zones") or [],
                scan.get("radar_zones") or [],
                tf_block.get("candidates") or [],
                r_prio.get("aligned_zones") or [],
            ):
                for item in group:
                    if isinstance(item, dict) and item.get("zone_id"):
                        live_pool.append(item)

            exact = next((z for z in live_pool if z.get("zone_id") == zone_id), None)
            if exact is not None:
                return exact

            # Preserve stable SMC merged-zone identity if constituents rebuilt with a new hash ID
            if isinstance(record, dict):
                rec_zone = record.get("zone") or {}
                rec_zt = str(rec_zone.get("zone_type") or "").upper()
                rec_lo, rec_hi = safe_float(record.get("low")), safe_float(record.get("high"))
                old_sources = {
                    str(s) for s in (
                        list(rec_zone.get("merged_from") or [])
                        + ([rec_zone.get("source_zone_id")] if rec_zone.get("source_zone_id") else [])
                        + ([zone_id] if zone_id else [])
                    ) if s
                }
                if old_sources and rec_zt and rec_lo > 0 and rec_hi >= rec_lo:
                    seen_ids = set()
                    rebuilt_matches = []
                    for cand in live_pool:
                        cz_id = cand.get("zone_id")
                        if not cz_id or cz_id in seen_ids:
                            continue
                        seen_ids.add(cz_id)
                        if str(cand.get("zone_type") or "").upper() != rec_zt:
                            continue
                        c_lo, c_hi = safe_float(cand.get("price_low")), safe_float(cand.get("price_high"))
                        if not (c_lo > 0 and c_hi >= c_lo and min(rec_hi, c_hi) >= max(rec_lo, c_lo)):
                            continue
                        cand_sources = {
                            str(s) for s in (
                                list(cand.get("merged_from") or [])
                                + ([cand.get("source_zone_id")] if cand.get("source_zone_id") else [])
                                + [cz_id]
                            ) if s
                        }
                        if old_sources.intersection(cand_sources):
                            rebuilt_matches.append(cand)
                    if len(rebuilt_matches) == 1:
                        return rebuilt_matches[0]

            rejected_pool = [z for z in (scan.get("rejected_zones") or []) if isinstance(z, dict)]
            return next((z for z in rejected_pool if z.get("zone_id") == zone_id), None)

        ranked = scan.get("ranked_zones") or {}
        zones = (
            list(ranked.get("active_supports") or [])
            + list(ranked.get("active_resistances") or [])
            + list(ranked.get("pending_break_zones") or [])
            + list(ranked.get("broken_zones") or [])
            + list(ranked.get("reclaimed_zones") or [])
            + list(ranked.get("invalidated_zones") or [])
            + list(ranked.get("historical_zones") or [])
            + list(ranked.get("all_zones") or [])
        )
        return next((z for z in zones if isinstance(z, dict) and z.get("zone_id") == zone_id), None)

    @classmethod
    def _is_valid_focus_candidate(cls, method, zone):
        if not isinstance(zone, dict) or not zone.get("zone_id"):
            return False
        low, high = cls._bounds(method, zone)
        if not (0 < low <= high):
            return False
        kind = str(zone.get("zone_type") or "").upper()
        if kind not in ("DEMAND", "SUPPORT", "SUPPLY", "RESISTANCE"):
            return False
        state = str(zone.get("lifecycle_status") or zone.get("status") or "ACTIVE").upper()
        if state in (
            "INVALIDATED", "EXPIRED", "MITIGATED", "BROKEN", "RECLAIMED", "RECLAIM_PENDING",
            "RETEST", "BREAKDOWN_PENDING", "BREAKOUT_PENDING", "UNCONFIRMED", "HISTORICAL",
        ):
            return False
        if (
            zone.get("is_valid") is False
            or zone.get("is_broken") is True
            or zone.get("is_invalidated") is True
            or zone.get("is_active") is False
            or str(zone.get("grade") or "").upper() == "REJECTED"
        ):
            return False
        if str(zone.get("risk_tier") or "").upper() == "FAR_ZONE":
            return False
        if str(zone.get("radar_eligibility") or "QUALIFIED").upper() == "NOT_QUALIFIED":
            return False
        return True

    @classmethod
    def _same_zone_identity(cls, method, record, candidate):
        if not isinstance(record, dict) or not isinstance(candidate, dict):
            return False
        rec_id = record.get("zone_id")
        if not rec_id or candidate.get("zone_id") != rec_id:
            return False
        rec_zone = record.get("zone") or {}
        rec_src = str(rec_zone.get("source") or method or "").strip().upper()
        cand_src = str(candidate.get("source") or method or "").strip().upper()
        if rec_src and cand_src and rec_src != cand_src:
            return False
        rec_zt = str(rec_zone.get("zone_type") or rec_zone.get("type") or "").strip().upper()
        cand_zt = str(candidate.get("zone_type") or candidate.get("type") or "").strip().upper()
        if rec_zt and cand_zt and rec_zt != cand_zt:
            return False
        rec_tf = str(rec_zone.get("timeframe") or "").strip().upper()
        cand_tf = str(candidate.get("timeframe") or "").strip().upper()
        if rec_tf and cand_tf and rec_tf != cand_tf:
            return False
        rec_lo, rec_hi = safe_float(record.get("low")), safe_float(record.get("high"))
        cand_lo, cand_hi = cls._bounds(method, candidate)
        if not (cand_lo > 0 and cand_hi >= cand_lo):
            return False
        if rec_lo > 0 and rec_hi >= rec_lo and (abs(rec_lo - cand_lo) >= 1e-6 or abs(rec_hi - cand_hi) >= 1e-6):
            return False
        for ts_key in ("confirmed_time", "confirmation_time", "formation_time"):
            rec_ts = rec_zone.get(ts_key)
            cand_ts = candidate.get(ts_key)
            if rec_ts and cand_ts and str(rec_ts) != str(cand_ts):
                return False
        return True

    @staticmethod
    def _merged_zone_metadata(method, record, live_zone):
        refreshed = dict(live_zone)
        if method == "SMC":
            refreshed["price_low"] = record["low"]
            refreshed["price_high"] = record["high"]
        else:
            refreshed["low"] = record["low"]
            refreshed["high"] = record["high"]
        if not refreshed.get("source"):
            refreshed["source"] = (record.get("zone") or {}).get("source") or method
        return refreshed

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
            if (
                zone.get("is_valid") is False
                or lifecycle in ("INVALIDATED", "EXPIRED", "MITIGATED", "BROKEN", "UNCONFIRMED")
                or str(zone.get("risk_tier") or "").upper() == "FAR_ZONE"
                or str(zone.get("radar_eligibility") or "QUALIFIED").upper() == "NOT_QUALIFIED"
            ):
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

    def update(self, results, snapshot, positions, full_refresh, orders=None):
        alerts = []
        self.refresh_required = False
        self.active_methods = {method for method in ("SMC", "SNR") if results.get(method) is not None}
        snap = snapshot if isinstance(snapshot, dict) else {}
        bid, ask = safe_float(snap.get("bid")), safe_float(snap.get("ask"))
        price = safe_float(snap.get("snapshot_price"))
        snapshot_valid = (
            snap.get("data_ok") is not False
            and str(snap.get("data_freshness") or "").upper() not in ("STALE", "INVALID", "NO DATA")
            and price > 0
        )
        fresh = (
            snapshot_valid and snap.get("live") is True
            and str(snap.get("market_status") or "").upper() == "OPEN"
            and 0 < bid <= ask
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
            recent_comp = self._recently_completed.get(method)
            if isinstance(recent_comp, dict) and price > 0:
                rc_lo = safe_float(recent_comp.get("low"), 0.0)
                rc_hi = safe_float(recent_comp.get("high"), 0.0)
                if rc_lo > 0 and rc_hi >= rc_lo and not (rc_lo <= price <= rc_hi):
                    recent_comp["left"] = True
            method_ok = isinstance(results.get(method), dict) and results[method].get("status") == "OK"
            snr_live_switch = (method == "SNR" and not full_refresh)
            if (
                record is not None
                and (full_refresh or snr_live_switch)
                and snapshot_valid
                and method_ok
                and not record["ever_taken"]
                and record.get("account_status") not in ("RUNNING", "AMBIGUOUS")
            ):
                cur_chk = self._current_zone(method, results.get(method), record["zone_id"], record=record)
                if (
                    method == "SMC"
                    and isinstance(cur_chk, dict)
                    and cur_chk.get("zone_id")
                    and cur_chk.get("zone_id") != record["zone_id"]
                    and self._is_valid_focus_candidate(method, cur_chk)
                ):
                    new_lo, new_hi = self._bounds(method, cur_chk)
                    record["zone_id"] = cur_chk["zone_id"]
                    record["low"] = new_lo
                    record["high"] = new_hi
                    record["zone"] = dict(cur_chk)
                cur_life = str((cur_chk or {}).get("lifecycle_status") or (cur_chk or {}).get("status") or "").upper()
                cur_tier = str((cur_chk or {}).get("risk_tier") or "").upper()
                cur_elig = str((cur_chk or {}).get("radar_eligibility") or "QUALIFIED").upper()
                trk_obj = ((results.get(method) or {}).get("decision") or {}).get("zone_tracker") if method == "SMC" else None
                trk_rec = trk_obj.get_zone(record["zone_id"]) if (trk_obj and hasattr(trk_obj, "get_zone")) else None
                trk_life = str((trk_rec or {}).get("lifecycle_status") or (trk_rec or {}).get("status") or "").upper()
                smc_inval = safe_float((cur_chk or {}).get("invalidation_price"), 0.0) if method == "SMC" else 0.0
                smc_inval_breached = bool(
                    method == "SMC" and price > 0 and smc_inval > 0 and (
                        (record["direction"] == "BUY" and price < smc_inval)
                        or (record["direction"] == "SELL" and price > smc_inval)
                    )
                )
                snr_side_breached = bool(
                    method == "SNR" and price > 0 and (
                        (record["direction"] == "BUY" and price < record["low"])
                        or (record["direction"] == "SELL" and price > record["high"])
                    )
                )
                stale_zone = (
                    cur_chk is None
                    or smc_inval_breached
                    or snr_side_breached
                    or cur_life in (
                        "INVALIDATED", "BROKEN", "RECLAIMED", "RECLAIM_PENDING", "RETEST",
                        "BREAKDOWN_PENDING", "BREAKOUT_PENDING", "EXPIRED", "MITIGATED", "UNCONFIRMED", "HISTORICAL",
                    )
                    or trk_life in ("INVALIDATED", "EXPIRED", "MITIGATED", "BROKEN")
                    or cur_chk.get("is_valid") is False
                    or cur_chk.get("is_broken") is True
                    or cur_chk.get("is_invalidated") is True
                    or cur_chk.get("is_active") is False
                    or str(cur_chk.get("grade") or "").upper() == "REJECTED"
                    or cur_tier == "FAR_ZONE"
                    or cur_elig == "NOT_QUALIFIED"
                )
                cand_valid = self._is_valid_focus_candidate(method, zone)
                cand_is_recent_comp = bool(
                    isinstance(recent_comp, dict)
                    and isinstance(zone, dict)
                    and zone.get("zone_id") == recent_comp.get("zone_id")
                    and not (
                        recent_comp.get("left")
                        and (
                            (results[method].get("decision") or {}).get("status") == "VALID"
                            if method == "SMC" else (results[method].get("decision") or {}).get("signal") in ("BUY", "SELL")
                        )
                    )
                )
                can_supersede = (
                    method == "SNR"
                    or not record["touched"]
                    or not (record["low"] <= price <= record["high"])
                )
                superseded_candidate = bool(
                    cand_valid
                    and not cand_is_recent_comp
                    and (
                        zone.get("zone_id") != record["zone_id"]
                        or not self._same_zone_identity(method, record, zone)
                    )
                    and can_supersede
                )
                if stale_zone or superseded_candidate:
                    self.focus[method] = None
                    record = None
                    if snr_live_switch:
                        self.refresh_required = True
                elif full_refresh and fresh and positions is not None and history_complete:
                    record["uncertain"] = False
            if (
                record is not None
                and full_refresh
                and method_ok
                and snap.get("data_ok") is not False
                and str(snap.get("data_freshness") or "").upper() not in ("STALE", "INVALID", "NO DATA")
            ):
                live_zone = self._current_zone(method, results[method], record["zone_id"], record=record)
                if live_zone is None and isinstance(zone, dict) and zone.get("zone_id") == record["zone_id"]:
                    live_zone = zone
                if self._same_zone_identity(method, record, live_zone):
                    record["zone"] = self._merged_zone_metadata(method, record, live_zone)
            switched_from_completed = False
            if record is None and full_refresh and completed and zone and zone.get("zone_id") == completed["zone_id"]:
                alternative = self._next_zone(method, results[method], completed, zone)
                if alternative is not None:
                    zone, plan = alternative, None
                    switched_from_completed = True
            if record is None and (full_refresh or snr_live_switch) and snapshot_valid and isinstance(zone, dict) and zone.get("zone_id"):
                low, high = self._bounds(method, zone)
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
                if direction and self._is_valid_focus_candidate(method, zone):
                    record = {
                        "zone": dict(zone), "zone_id": zone["zone_id"], "low": low, "high": high,
                        "direction": direction, "issued_at": time.time(), "plan": None, "touched": False,
                        "ticket": None, "pending_ticket": None, "ever_taken": False, "account_status": "", "uncertain": False,
                    }
                    self.focus[method] = record
                    new_focus.add(method)
                    if switched_from_completed and isinstance(completed, dict):
                        self._recently_completed[method] = dict(completed)
                    elif isinstance(recent_comp, dict) and zone["zone_id"] == recent_comp.get("zone_id"):
                        self._recently_completed[method] = None
                    self.completed[method] = None
                    if completed:
                        self._emit_lifecycle_alert(
                            alerts,
                            method,
                            "NEXT_ZONE",
                            zone["zone_id"],
                            f"[{method}] ZONA BERIKUTNYA {low:.2f}-{high:.2f}; "
                            "KANDIDAT RANKING, MENUNGGU VALIDASI ENTRY",
                        )
            if record and full_refresh and self._same_zone_identity(method, record, zone):
                if not record["touched"]:
                    plan_updates[method] = (
                        {"direction": plan["direction"], **{key: round(safe_float(plan[key]), 2) for key in ("entry", "sl", "tp1")}}
                        if self._valid_plan(plan) and plan["direction"] == record["direction"] else None
                    )
                elif not record["plan"] and not record["ever_taken"] and self._valid_plan(plan) and plan["direction"] == record["direction"]:
                    plan_updates[method] = {"direction": plan["direction"], **{key: round(safe_float(plan[key]), 2) for key in ("entry", "sl", "tp1")}}
                if method in new_focus and method in plan_updates:
                    record["plan"] = plan_updates.pop(method)

        _, accepted_syms = _acceptable_trade_symbols(self.symbol)
        pending_matches = {}
        for method, record in self.focus.items():
            pending_matches[method] = []
            if not record or orders is None:
                continue
            plan_entry = safe_float((record.get("plan") or {}).get("entry"), 0.0)
            for order in orders:
                ord_sym = str(_position_value(order, "symbol", "") or "").strip()
                if ord_sym.upper() not in accepted_syms:
                    continue
                raw_ord_type = _position_value(order, "order_type")
                if raw_ord_type is None:
                    raw_ord_type = _position_value(order, "type")
                if raw_ord_type is None:
                    continue
                ord_label, ord_dir = _info_tipe_pending(raw_ord_type)
                if ord_dir != record["direction"] or not any(tok in ord_label for tok in ("LIMIT", "STOP")):
                    continue
                ord_price = safe_float(_position_value(order, "price_open", _position_value(order, "entry_price")), 0.0)
                ord_ticket = _position_value(order, "ticket")
                if ord_price <= 0 or not ord_ticket:
                    continue
                in_bounds = (record["low"] - 0.05 - 1e-9 <= ord_price <= record["high"] + 0.05 + 1e-9)
                near_plan = (plan_entry > 0 and abs(ord_price - plan_entry) <= 0.05 + 1e-9)
                if in_bounds or near_plan:
                    pending_matches[method].append(order)

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
            record["pending_ticket"] = _position_value(pending_matches[method][0], "ticket") if pending_matches.get(method) else None
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
                record["account_status"] = account_status

            method_ok = isinstance(results.get(method), dict) and results[method].get("status") == "OK"
            current = self._current_zone(method, results.get(method), record["zone_id"], record=record) if full_refresh and method_ok else None
            if method == "SNR" and current is not None and (
                safe_float(current.get("touch_count")) > 0
                or str(current.get("lifecycle_event") or "").upper() in ("TOUCHED", "SWEEP")
            ):
                if not record.get("historical_touch"):
                    record["historical_touch"] = True
                    alerts.append("[SNR] ZONA PERNAH DISENTUH (HISTORIS); BUKAN BUKTI ENTRY")
            lifecycle = str((current or {}).get("lifecycle_status") or (current or {}).get("status") or "").upper()
            if method == "SNR" and current is not None and lifecycle in ("BREAKDOWN_PENDING", "BREAKOUT_PENDING"):
                if record.get("break_watch") != lifecycle:
                    record["break_watch"] = lifecycle
                    bw_lbl = "POTENSI SUPPORT BREAKDOWN" if lifecycle == "BREAKDOWN_PENDING" else "POTENSI RESISTANCE BREAKOUT"
                    alerts.append(f"[SNR] {bw_lbl} ({record['low']:.2f}-{record['high']:.2f}); MENUNGGU KONFIRMASI CANDLE")
            elif record.get("break_watch"):
                record["break_watch"] = ""
            broken_or_reclaimed = bool(
                current is not None and (
                    lifecycle in ("BROKEN", "RECLAIMED", "RECLAIM_PENDING", "RETEST")
                    or current.get("is_broken") is True
                )
            )
            vanished = bool(full_refresh and fresh and method_ok and current is None and method not in new_focus)
            trk_obj_inv = ((results.get(method) or {}).get("decision") or {}).get("zone_tracker") if (method == "SMC" and full_refresh and method_ok) else None
            trk_rec_inv = trk_obj_inv.get_zone(record["zone_id"]) if (trk_obj_inv and hasattr(trk_obj_inv, "get_zone")) else None
            trk_life_inv = str((trk_rec_inv or {}).get("lifecycle_status") or (trk_rec_inv or {}).get("status") or "").upper()
            cur_smc_inval = safe_float((current or {}).get("invalidation_price"), 0.0) if method == "SMC" else 0.0
            smc_inval_breached_inv = bool(
                method == "SMC" and fresh and price > 0 and cur_smc_inval > 0 and (
                    (record["direction"] == "BUY" and price < cur_smc_inval)
                    or (record["direction"] == "SELL" and price > cur_smc_inval)
                )
            )
            invalid = bool(
                vanished
                or smc_inval_breached_inv
                or trk_life_inv in ("INVALIDATED", "EXPIRED", "MITIGATED", "BROKEN")
                or (
                    current is not None and (
                        lifecycle in (
                            "INVALIDATED", "BROKEN", "RECLAIMED", "RECLAIM_PENDING", "RETEST",
                            "EXPIRED", "MITIGATED", "HISTORICAL",
                        )
                        or current.get("is_valid") is False
                        or current.get("is_broken") is True
                        or current.get("is_invalidated") is True
                        or str(current.get("grade") or "").upper() == "REJECTED"
                    )
                )
            )
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
                    self._emit_lifecycle_alert(
                        alerts,
                        method,
                        "TOUCH",
                        record["zone_id"],
                        f"[{method}] HARGA DI ZONA {record['low']:.2f}-{record['high']:.2f}; ENTRY BELUM DIPASTIKAN",
                    )
                    if not full_refresh:
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
            if not outcome and account_status == "NO MATCHING POSITION" and invalid and not record["ever_taken"]:
                unc_tag = " (RIWAYAT SEBELUMNYA TIDAK PASTI)" if record["uncertain"] else ""
                if broken_or_reclaimed:
                    outcome = f"ZONA {lifecycle or 'BROKEN'} SEBELUM POSISI TERDETEKSI{unc_tag}"
                elif vanished:
                    outcome = f"ZONA TIDAK LAGI TERDETEKSI SEBELUM POSISI TERDETEKSI{unc_tag}"
                else:
                    outcome = f"ZONA INVALID SEBELUM POSISI TERDETEKSI{unc_tag}"
            if outcome:
                note = "POSISI BELUM TERDETEKSI" if not record["ever_taken"] else "STATUS PENUTUPAN MT5 BELUM DIPASTIKAN"
                self._emit_lifecycle_alert(
                    alerts,
                    method,
                    "OUTCOME",
                    (record["zone_id"], outcome),
                    f"[{method}] {outcome}; {note}",
                )
                self.completed[method] = {
                    "zone_id": record["zone_id"],
                    "low": record["low"],
                    "high": record["high"],
                    "left": not record["low"] <= price <= record["high"],
                }
                self.focus[method] = None
                self.refresh_required = True
            elif method in plan_updates and (not record["touched"] or not record["plan"] and not record["ever_taken"]):
                old_plan = record["plan"]
                record["plan"] = plan_updates[method]
                rec_zid = record.get("zone_id") or "N/A"
                plan_key = (
                    rec_zid,
                    tuple(sorted(record["plan"].items())) if isinstance(record["plan"], dict) else None,
                )
                if record["touched"] and not old_plan and record["plan"]:
                    self._emit_lifecycle_alert(
                        alerts,
                        method,
                        "PLAN",
                        plan_key,
                        f"[{method}] ACTIVE FOCUS [{rec_zid}] RENCANA SL/TP1 TERSEDIA; BUKAN KONFIRMASI ENTRY",
                    )
                elif not record["touched"] and old_plan != record["plan"]:
                    self._emit_lifecycle_alert(
                        alerts,
                        method,
                        "PLAN",
                        plan_key,
                        f"[{method}] ACTIVE FOCUS [{rec_zid}] RENCANA SL/TP1 "
                        f"{'DIPERBARUI' if record['plan'] else 'TIDAK TERSEDIA'}; BUKAN KONFIRMASI ENTRY",
                    )
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
        orig_alerts = list(result.get("alerts") or [])
        orig_critical = list(result.get("critical_warnings") or [])
        result = dict(result)
        if method == "SNR":
            result["alerts"] = [
                al for al in orig_alerts
                if any(tok in str(al).upper() for tok in ("BREAKDOWN", "BREAKOUT", "RECLAIM", "RETEST", "BROKEN", "INVALID", "REJECTION"))
            ]
            result["critical_warnings"] = orig_critical
        else:
            result["alerts"] = []
            result["critical_warnings"] = orig_critical
        dec = dict(result.get("decision") or {})
        reason = (
            "WAIT_HISTORY_RECONCILIATION: Urutan harga atau posisi tidak pasti."
            if record and record["uncertain"] else
            "WAIT_FOCUS_RESOLUTION: Menunggu SL/TP1 atau invalidasi zona sebelum entry."
            if record else "MENUNGGU ZONA ATAU KUNJUNGAN BARU."
        )
        if method == "SMC":
            same_smc_zone = bool(record and candidate_id == record["zone_id"])
            same_smc_id = bool(same_smc_zone and self._same_zone_identity(method, record, zone))
            live_smc = (
                zone if same_smc_id
                else self._current_zone(method, result, record["zone_id"]) if record
                else None
            )
            if record:
                if same_smc_id and isinstance(zone, dict):
                    focus_smc = dict(zone)
                elif self._same_zone_identity(method, record, live_smc):
                    focus_smc = self._merged_zone_metadata(method, record, live_smc)
                else:
                    focus_smc = dict(record["zone"])
            else:
                focus_smc = None

            orig_primary = dec.get("primary_zone")
            orig_prio = dec.get("radar_priority")
            orig_early = dec.get("early_setup") if isinstance(dec.get("early_setup"), dict) else {}
            orig_best = orig_early.get("best_setup") if isinstance(orig_early.get("best_setup"), dict) else None
            orig_plans = dec.get("zone_plans") if isinstance(dec.get("zone_plans"), dict) else {}

            # Preserve valid SMC Pending READY plans independently when no live position was taken yet
            # (Confirmed Entry remains strictly WAIT while gated). Validate every candidate against live lifecycle.
            completed_id = completed.get("zone_id") if isinstance(completed, dict) else None
            trk_gate = dec.get("zone_tracker")
            raw_early_list = []
            seen_early_ids = set()
            for cand_p in ([orig_best] if orig_best else []) + list(orig_early.get("all_setups") or []):
                if not isinstance(cand_p, dict):
                    continue
                cz_id = str(cand_p.get("zone_id") or cand_p.get("source_zone_id") or "").strip()
                key_p = cz_id or f"_anon_{len(raw_early_list)}"
                if key_p in seen_early_ids:
                    continue
                seen_early_ids.add(key_p)
                raw_early_list.append(cand_p)

            valid_early = []
            if not (record and record.get("ever_taken")):
                for cand_p in raw_early_list:
                    if (
                        cand_p.get("valid") is False
                        or str(cand_p.get("status") or "").upper() != "READY"
                        or cand_p.get("is_expired")
                        or cand_p.get("is_expiration_error")
                    ):
                        continue
                    c_ord = str(cand_p.get("order_type") or "").upper()
                    c_dir = "BUY" if "BUY" in c_ord else ("SELL" if "SELL" in c_ord else "")
                    c_ent = safe_float(cand_p.get("entry_price", cand_p.get("entry")), 0.0)
                    c_sl = safe_float(cand_p.get("sl"), 0.0)
                    c_tp1 = safe_float(cand_p.get("tp1"), 0.0)
                    if not (
                        c_dir in ("BUY", "SELL")
                        and c_ent > 0 and c_sl > 0 and c_tp1 > 0
                        and (c_sl < c_ent < c_tp1 if c_dir == "BUY" else c_tp1 < c_ent < c_sl)
                    ):
                        continue
                    cz_id = str(cand_p.get("zone_id") or cand_p.get("source_zone_id") or "").strip()
                    if cz_id:
                        if completed_id and cz_id == completed_id:
                            continue
                        live_cand = self._current_zone("SMC", result, cz_id)
                        if live_cand is not None and not self._is_valid_focus_candidate("SMC", live_cand):
                            continue
                        trk_c = trk_gate.get_zone(cz_id) if (trk_gate and hasattr(trk_gate, "get_zone")) else None
                        trk_c_st = str((trk_c or {}).get("lifecycle_status") or (trk_c or {}).get("status") or "").upper()
                        if trk_c_st in ("INVALIDATED", "EXPIRED", "MITIGATED", "BROKEN"):
                            continue
                    valid_early.append(cand_p)

            preserved_early = {
                **orig_early,
                "best_setup": valid_early[0] if valid_early else None,
                "all_setups": valid_early,
            } if valid_early else {"best_setup": None, "all_setups": []}
            valid_early_ids = {
                str(p.get("zone_id") or p.get("source_zone_id") or "")
                for p in valid_early
                if p.get("zone_id") or p.get("source_zone_id")
            }
            keep_plan_ids = set(valid_early_ids)
            if focus_smc and focus_smc.get("zone_id"):
                keep_plan_ids.add(focus_smc.get("zone_id"))
            preserved_plans = {k: v for k, v in orig_plans.items() if k in keep_plan_ids} if keep_plan_ids else {}

            orig_stages = dict(dec.get("stages") or {})
            orig_data_ok = (orig_stages.get("DATA") or {}).get("ok") is not False
            preserve_bound_m5 = bool(
                same_smc_id
                and focus_smc
                and orig_data_ok
                and _m5_stages_match_zone(dec, focus_smc)
            )
            dec.update(
                status="WAIT", direction=None, zone=focus_smc,
                primary_zone=orig_primary if isinstance(orig_primary, dict) else focus_smc,
                plan=None,
                nearest_valid_zone=dec.get("nearest_valid_zone") if focus_smc else None,
                next_radar_zone=dec.get("next_radar_zone") if focus_smc else None,
                signal_status="WAITING CONFIRMATION",
                reasons=[reason],
                zone_plans=preserved_plans,
                early_setup=preserved_early,
                radar_priority=orig_prio if isinstance(orig_prio, dict) else {"primary_zone": focus_smc},
                entry_permission="WAIT",
            )
            if preserve_bound_m5:
                kept_stages = dict(orig_stages)
                kept_stages["entry_validation"] = {"status": "WAIT", "msg": reason}
                dec["stages"] = kept_stages
                dec["m5_stages_zone_id"] = focus_smc.get("zone_id")
            else:
                dec["stages"] = {
                    **{key: {"status": "NOT_CHECKED", "msg": reason} for key in (
                        "zone_validation", "zone_touch", "liquidity_sweep", "structure_trigger",
                        "displacement", "entry_validation", "risk_reward",
                    )},
                    "DATA": {"ok": False, "msg": reason},
                }
                dec["m5"] = {}
                dec["m5_confirmation"] = None
                dec["m5_stages_zone_id"] = focus_smc.get("zone_id") if isinstance(focus_smc, dict) else None
                dec["smc_technical_status"] = "WAIT"
            dec["active_setup"] = {
                **((dec.get("active_setup") or {}) if isinstance(dec.get("active_setup"), dict) else {}),
                "zone_id": focus_smc.get("zone_id") if isinstance(focus_smc, dict) else None,
                "trigger_status": (dec["stages"].get("structure_trigger") or {}).get("status", "NOT_CHECKED"),
                "entry_status": (dec["stages"].get("entry_validation") or {}).get("status", "WAIT"),
                "m5_confirmation": dec.get("m5_confirmation") if preserve_bound_m5 else None,
                "final_decision": "WAIT",
                "blocker": reason,
            }
            scan = dict(result.get("scan_result") or {})
            if not isinstance(scan.get("radar_priority"), dict):
                scan["radar_priority"] = {"primary_zone": focus_smc}
            result["scan_result"] = scan
            result["decision"] = dec
            result["summary"] = extract_smc_monitor_summary(scan, dec)
            result["formatted_section"] = format_smc_monitor_section(result, detailed=detailed)
        else:
            use_live_snr_focus = bool(
                record
                and not record.get("ever_taken")
                and record.get("account_status") not in ("RUNNING", "AMBIGUOUS")
                and self._is_valid_focus_candidate("SNR", zone)
            )
            if use_live_snr_focus:
                same_zone = True
                same_identity = True
                same_bounds = True
                old_zone = zone
            else:
                same_zone = bool(record and candidate_id == record["zone_id"])
                same_identity = bool(same_zone and self._same_zone_identity(method, record, zone))
                same_bounds = bool(
                    same_identity
                    and abs(safe_float(zone.get("low")) - record["low"]) < 1e-8
                    and abs(safe_float(zone.get("high")) - record["high"]) < 1e-8
                )
                live_zone = (
                    zone if same_identity
                    else self._current_zone(method, result, record["zone_id"]) if record
                    else None
                )
                if same_bounds:
                    old_zone = zone
                elif self._same_zone_identity(method, record, live_zone):
                    old_zone = self._merged_zone_metadata(method, record, live_zone)
                else:
                    old_zone = record["zone"] if record else None
            focus = {
                key: old_zone.get(key) for key in (
                    "zone_id", "zone_type", "timeframe", "rank_label", "low", "high", "grade", "score", "source",
                    "confirmed_time", "confirmation_time", "formation_time", "risk_tier", "distance", "distance_points",
                    "swing_class", "confluence_info", "historical_reaction_count", "touch_analysis", "lifecycle_event",
                    "lifecycle_status", "status",
                )
                if key in old_zone
            } if old_zone else None
            if focus and not same_bounds:
                cur_focus_life = str(focus.get("lifecycle_status") or focus.get("status") or "").upper()
                if cur_focus_life not in (
                    "BROKEN", "RECLAIMED", "RECLAIM_PENDING", "RETEST",
                    "BREAKDOWN_PENDING", "BREAKOUT_PENDING", "INVALIDATED", "HISTORICAL",
                ):
                    focus["lifecycle_status"] = "UNCONFIRMED"
            scan = dict(result.get("scan_result") or {})
            original_pa = scan.get("price_action") if isinstance(scan.get("price_action"), dict) else {}
            original_ev = scan.get("entry_validation") or {}
            original_plan = original_ev.get("provisional_plan") or {}
            plan_rr = original_plan.get("rr_calculation") or {}
            entry_rr = original_ev.get("rr_calculation") or {}
            preserved_trend_zone = dec.get("trend_entry_zone") or original_ev.get("trend_entry_zone")
            preserved_focus_reason = dec.get("focus_selection_reason") or original_ev.get("focus_selection_reason")
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
                trend_entry_zone=preserved_trend_zone,
                focus_selection_reason=preserved_focus_reason,
                entry=None, effective_entry=None, sl=None, tp1=None, tp2=None, reason=reason,
                trigger_status="NOT CONFIRMED", rr_display="NOT CHECKED",
            )
            ev = dict(original_ev)
            ev.update(
                final_status="WAIT", target_zone=focus,
                trend_entry_zone=preserved_trend_zone,
                focus_selection_reason=preserved_focus_reason,
                reason=reason, trigger_status="NOT CONFIRMED",
                provisional_plan=failed_plan,
                rr_calculation=entry_rr if entry_rr.get("checked") else {"checked": False, "valid": False, "rr_display": "NOT CHECKED"},
                recent_closed_touch=bool(original_ev.get("recent_closed_touch")) if same_bounds else False,
                focus_change=original_ev.get("focus_change") if same_bounds else None,
            )
            scan["entry_validation"] = ev
            scan["decision"] = dec
            preserved_pa = {
                "rejection_summary": "NOT CHECKED",
                "breakout_summary": original_pa.get("breakout_summary", "NOT CHECKED"),
                "retest_summary": "NOT CHECKED",
                "momentum_summary": "NOT CHECKED",
            }
            for pa_key in (
                "breakout_code", "break_type", "potential_breakdown", "potential_breakout",
                "reclaim_detected", "zone_failed", "breakout_direction", "breakout_is_historical",
                "breakout_zone_label", "breakout_detail",
            ):
                if pa_key in original_pa:
                    preserved_pa[pa_key] = original_pa[pa_key]
            scan["price_action"] = preserved_pa
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
            if method == "SNR" and any(tok in text for tok in ("BREAKDOWN", "BREAKOUT", "RECLAIM", "RETEST", "BROKEN", "INVALID")):
                keep = True
            elif zone_id and not record["uncertain"]:
                if method == "SMC":
                    keep = not any(word in text for word in ("PERUBAHAN ZONA", "ZONA TERMITIGASI", "ZONA SUPPLY INVALID", "ZONA DEMAND INVALID"))
                elif "SUPPORT S1" in text:
                    keep = (radar.get("nearest_support") or {}).get("zone_id") == zone_id
                elif "RESISTANCE R1" in text:
                    keep = (radar.get("nearest_resistance") or {}).get("zone_id") == zone_id
                elif "REJECTION" in text:
                    keep = (action.get("rejection_detail") or {}).get("zone_id") == zone_id
                elif "RETEST" in text:
                    keep = (action.get("retest_detail") or {}).get("zone_id") == zone_id
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
                f"PENDING ORDER #{record['pending_ticket']} MENUNGGU FILL (BELUM ADA POSISI TERBUKA)" if record.get("pending_ticket") else
                "BELUM ADA POSISI TERBUKA MT5 (TRACKING POSISI AKTIF MEMBUTUHKAN POSISI TERBUKA)" if record["touched"] else
                "POSISI BELUM TERDETEKSI" if record.get("historical_touch") else ""
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
        self._emitted_stage_events = set()

    def reset(self):
        global _GLOBAL_LAST_SMC_STATE
        self.last_state = {}
        self._emitted_stage_events = set()
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

        z_short_cur = curr_state.get("primary_zone", "").split(" (")[0].strip()
        z_suffix = f" ({z_short_cur})" if z_short_cur and z_short_cur not in ("NONE", "-") else ""
        ev_ts = str(smc_summary.get("snapshot_time") or "").strip()
        ts_suffix = f" [{ev_ts}]" if ev_ts else ""

        # 1. Perubahan keputusan setup SMC
        same_setup = curr_state["primary_setup_id"] == self.last_state.get("primary_setup_id")
        if not same_setup:
            self._emitted_stage_events = {
                ev_k for ev_k in self._emitted_stage_events
                if ev_k[0] == curr_state["primary_setup_id"]
            }
        if curr_state["signal"] != self.last_state.get("signal") or not same_setup:
            if curr_state["signal"] == "BUY":
                alerts.append(f"[SMC] BUY SETUP TERKONFIRMASI{z_suffix}{ts_suffix}")
            elif curr_state["signal"] == "SELL":
                alerts.append(f"[SMC] SELL SETUP TERKONFIRMASI{z_suffix}{ts_suffix}")
            elif curr_state["signal"] == "INVALID":
                alerts.append(f"[SMC] SETUP INVALID{z_suffix}{ts_suffix}")

        # 2. Perubahan tahapan konfirmasi M5 SMC (terdedup per setup/zona agar tidak berulang)
        setup_key = curr_state["primary_setup_id"]
        if (
            (not same_setup or curr_state["zone_touch"] != self.last_state.get("zone_touch"))
            and curr_state["zone_touch"] == "IN_ZONE"
            and (setup_key, "ZONE_TOUCH") not in self._emitted_stage_events
        ):
            self._emitted_stage_events.add((setup_key, "ZONE_TOUCH"))
            alerts.append(f"[SMC] ZONE TOUCH TERKONFIRMASI{z_suffix}{ts_suffix}")

        if (
            (not same_setup or curr_state["sweep"] != self.last_state.get("sweep"))
            and curr_state["sweep"] == "CONFIRMED"
            and (setup_key, "SWEEP") not in self._emitted_stage_events
        ):
            self._emitted_stage_events.add((setup_key, "SWEEP"))
            alerts.append(f"[SMC] LIQUIDITY SWEEP TERKONFIRMASI{z_suffix}{ts_suffix}")

        if (
            (not same_setup or curr_state["choch_bos"] != self.last_state.get("choch_bos"))
            and curr_state["choch_bos"] == "CONFIRMED"
            and (setup_key, "CHOCH_BOS") not in self._emitted_stage_events
        ):
            self._emitted_stage_events.add((setup_key, "CHOCH_BOS"))
            alerts.append(f"[SMC] CHoCH / BOS M5 TERKONFIRMASI{z_suffix}{ts_suffix}")

        if (
            (not same_setup or curr_state["displacement"] != self.last_state.get("displacement"))
            and curr_state["displacement"] == "CONFIRMED"
            and (setup_key, "DISPLACEMENT") not in self._emitted_stage_events
        ):
            self._emitted_stage_events.add((setup_key, "DISPLACEMENT"))
            alerts.append(f"[SMC] DISPLACEMENT TERKONFIRMASI{z_suffix}{ts_suffix}")

        # 3. Perubahan zona aktif / invalidasi zona SMC
        prev_inval = self.last_state.get("invalidated_count")
        if prev_inval is not None and curr_state["invalidated_count"] > prev_inval:
            last_inval_type = str(smc_summary.get("last_invalidated_type", "ZONA")).upper()
            alerts.append(f"[SMC] ZONA {last_inval_type} INVALID{ts_suffix}")

        prev_mit = self.last_state.get("mitigated_count")
        if prev_mit is not None and curr_state["mitigated_count"] > prev_mit:
            alerts.append(f"[SMC] ZONA TERMITIGASI{z_suffix}{ts_suffix}")

        if (
            self.last_state.get("primary_zone_id") is not None
            and curr_state["primary_zone_id"] != self.last_state.get("primary_zone_id")
            and curr_state["primary_zone_id"] not in ("NONE", "-", "")
        ):
            alerts.append(f"[SMC] PERUBAHAN ZONA FOKUS -> {z_short_cur}{ts_suffix}")

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

    active_mon_zone = dec.get("zone")
    raw_stages = dec.get("m5_stages") or dec.get("stages") or {}
    m5_stages_bound = _m5_stages_match_zone(dec, active_mon_zone) if isinstance(active_mon_zone, dict) else False
    stages = raw_stages if m5_stages_bound else {}
    zt = (stages.get("zone_touch") or {}).get("status", "NOT_CHECKED")
    sw = (stages.get("liquidity_sweep") or {}).get("status", "NOT_CHECKED")
    tr = (stages.get("structure_trigger") or {}).get("status", "NOT_CHECKED")
    dp = (stages.get("displacement") or {}).get("status", "NOT_CHECKED")
    rr_st = (stages.get("risk_reward") or {}).get("status", "NOT_CHECKED")
    m5_bias = tr
    m5_conf_sum = (
        dict(dec["m5_confirmation"])
        if m5_stages_bound and isinstance(dec.get("m5_confirmation"), dict)
        else None
    )

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
        "m5_confirmation": m5_conf_sum,
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
        "snapshot_time": str(dec.get("snapshot_time") or scan.get("snapshot_time") or ""),
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
        "STATUS          : OK",
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
        unp_parts = []
        if p_nosl > 0:
            unp_parts.append(f"{p_nosl} posisi")
        if o_nosl > 0:
            unp_parts.append(f"{o_nosl} pending order")
        risk_warnings.append(
            f"[RISK WARNING] {' / '.join(unp_parts)} tanpa Stop Loss (SL MISSING)! Risiko akun tidak dapat dibatasi."
        )
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
        "orders": orders,
        "orders_snapshot": snap_r.get("orders_snapshot"),
        "audit_mode": "STRICTLY_READ_ONLY",
    }


from modules.smart_monitor_report import (  # noqa: E402,F401  (compat re-export)
    _as_monitor_zone,
    _calc_zone_touch,
    _m5_stages_match_zone,
    _smart_monitor_context,
    _stage_status,
    classify_monitor_alerts,
    format_account_monitor_section,
    format_smart_monitor_detail,
    format_smart_monitor_report,
    resolve_monitor_pending_plans,
)


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


def _normalize_risk_warning_sig(msg):
    """Normalisasi teks peringatan risiko agar jitter angka spread >= 0.80 tidak memicu cetak ulang dashboard."""
    txt = str(msg or "").strip()
    if not txt:
        return ""
    up = txt.upper()
    if "SPREAD" in up and ("MELEBAR" in up or "TOO HIGH" in up):
        return "SPREAD_HIGH"
    return txt


def _is_snr_touch_or_exit_alert(msg, active_is_smc=False):
    """
    True hanya untuk event sentuh/keluar/kedekatan SNR yang tidak mengubah keputusan,
    sehingga cukup ditampilkan satu baris tanpa mencetak ulang dashboard penuh.
    """
    up = str(msg or "").strip().upper()
    if not up.startswith("[SNR]"):
        return False
    # Event keputusan, lifecycle penting, risiko, MT5, error, atau historis tidak pernah diturunkan ke 1 baris sentuh/keluar.
    if any(tok in up for tok in (
        "BUY SETUP", "SELL SETUP", "CONFIRMED", "TERKONFIRMASI", "PENDING READY",
        "BREAKDOWN", "BREAKOUT", "RECLAIM", "RETEST", "REJECTION", "BROKEN", "DITEMBUS", "INVALID",
        "POSISI", "RUNNING", "AMBIGU", "SL RENCANA", "TP1 RENCANA", "RISK", "CRITICAL", "ERROR",
        "DATA UNAVAILABLE", "(HISTORIS)", "HISTORICAL", "PERNAH DISENTUH", "MARKET CLOSED",
    )):
        return False
    if any(tok in up for tok in (
        "HARGA DI SUPPORT", "HARGA DEKAT SUPPORT",
        "HARGA DI RESISTANCE", "HARGA DEKAT RESISTANCE",
        "HARGA DI ZONA", "HARGA SENTUH", "HARGA KELUAR", "HARGA MENINGGALKAN",
        "HARGA DI LUAR ZONA", "DISENTUH M5; HARGA DI LUAR ZONA",
    )):
        return True
    if active_is_smc and any(tok in up for tok in ("ZONA BERIKUTNYA", "BUKAN KONFIRMASI ENTRY")):
        return True
    return False


def _extract_mt5_items_sig(items, is_order=False):
    """Ringkasan struktural posisi/order MT5 tanpa floating P/L atau harga pasar live."""
    if not items:
        return ()
    rows = []
    for it in items:
        if isinstance(it, dict):
            tkt = it.get("ticket")
            typ = str(it.get("order_type", it.get("type", ""))).upper()
            vol = round(safe_float(it.get("volume_current", it.get("volume_initial", it.get("volume", 0.0))), 0.0), 2)
            p_open = round(safe_float(it.get("price_open", it.get("entry_price", 0.0)), 0.0), 2)
            sl = round(safe_float(it.get("sl", 0.0), 0.0), 2)
            tp = round(safe_float(it.get("tp", 0.0), 0.0), 2)
        else:
            tkt = getattr(it, "ticket", None)
            typ = str(getattr(it, "type", "")).upper()
            vol = round(safe_float(
                getattr(it, "volume_current", getattr(it, "volume_initial", getattr(it, "volume", 0.0))), 0.0
            ), 2)
            p_open = round(safe_float(getattr(it, "price_open", 0.0), 0.0), 2)
            sl = round(safe_float(getattr(it, "sl", 0.0), 0.0), 2)
            tp = round(safe_float(getattr(it, "tp", 0.0), 0.0), 2)
        rows.append((tkt, typ, vol, p_open, sl, tp))
    rows.sort(key=lambda r: (str(r[0] or ""), r[1], r[3]))
    return tuple(rows)


def _build_plan_signature(pending_setup, alt_pending_setups=None):
    def _one(ps):
        if not isinstance(ps, dict):
            return None
        return (
            str(ps.get("plan_status") or "").upper(),
            str(ps.get("broker_order_status") or "").upper(),
            str(ps.get("zone_id") or "N/A"),
            round(safe_float(ps.get("entry"), 0.0), 2),
            round(safe_float(ps.get("sl"), 0.0), 2),
            round(safe_float(ps.get("tp1"), 0.0), 2),
            bool(ps.get("sl_missing_high_risk")),
        )
    alts = tuple(_one(a) for a in (alt_pending_setups or []) if isinstance(a, dict))
    return (_one(pending_setup), alts)


class UnifiedMonitor:
    """
    Pusat Monitoring Terpadu yang menjalankan SMC Monitor, SNR Monitor, dan MT5 Account Monitor
    secara simultan dalam satu siklus dengan 1x pengambilan data MT5 per siklus,
    independensi sinyal penuh, dan isolasi kegagalan (error isolation).
    """

    def __init__(self, symbol=None, smc_runner=None, snr_runner=None, snapshot_provider=None, focus_tracking=False, news_alert_engine=None):
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
        self.last_snr_live_zone_state = None
        self.last_snr_inline_event = None
        self.last_mt5_status_sig = None
        self._last_zone_transition = {}
        self.focus_tracker = PantauFocusTracker(self.symbol) if focus_tracking else None
        if news_alert_engine is not None:
            self.news_alert_engine = news_alert_engine
        else:
            try:
                from modules.us_news_radar import NewsAlertEngine
                self.news_alert_engine = NewsAlertEngine()
            except Exception:
                self.news_alert_engine = None

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
                    "historical_events": al_bundle.get("historical_events") or [],
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
        tick_refresh_needed = False
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
            data_broken = (shared_snap or {}).get("data_ok") is False
            if data_broken:
                reason = str((shared_snap or {}).get("error_reason") or "Data tick tidak tersedia.")
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
            elif quote_changed and new_price > 0:
                reval_reason = "Harga atau spread berubah; menunggu validasi ulang snapshot."
                # SMC: Jangan menghapus pending plan / rekonsiliasi MT5 saat perubahan harga biasa dalam kondisi WAIT.
                # Hanya blokir sinyal live CONFIRMED (VALID/BUY/SELL) sampai validasi penuh berikutnya.
                if isinstance(smc_res, dict) and isinstance(smc_res.get("decision"), dict):
                    dec = dict(smc_res["decision"])
                    prev_smc_st = str(dec.get("status") or "WAIT").upper()
                    dec["current_price"] = new_price
                    if prev_smc_st in ("VALID", "BUY", "SELL"):
                        tick_refresh_needed = True
                        dec.update(status="WAIT", plan=None, signal_status="WAITING CONFIRMATION", reasons=[reval_reason])
                        active_setup = dict(dec.get("active_setup") or {})
                        active_setup.update(final_decision="WAIT", blocker=reval_reason)
                        dec["active_setup"] = active_setup
                    else:
                        act_z = dec.get("zone") or dec.get("primary_zone") or {}
                        if isinstance(act_z, dict):
                            z_lo = safe_float(act_z.get("price_low", act_z.get("low")), 0.0)
                            z_hi = safe_float(act_z.get("price_high", act_z.get("high")), 0.0)
                            inv_p = safe_float(act_z.get("invalidation_price"), 0.0)
                            z_tp = str(act_z.get("zone_type") or "").upper()
                            prev_p = safe_float(previous_snapshot.get("snapshot_price"), 0.0)
                            was_in = bool(z_lo > 0 and prev_p > 0 and z_lo <= prev_p <= z_hi)
                            now_in = bool(z_lo > 0 and z_lo <= new_price <= z_hi)
                            inv_hit = bool(
                                inv_p > 0 and (
                                    (z_tp in ("DEMAND", "SUPPORT") and new_price < inv_p)
                                    or (z_tp in ("SUPPLY", "RESISTANCE") and new_price > inv_p)
                                )
                            )
                            if (was_in != now_in) or inv_hit:
                                tick_refresh_needed = True
                    smc_res["decision"] = dec
                    smc_res["summary"] = extract_smc_monitor_summary(smc_res.get("scan_result"), dec)
                    smc_res["formatted_section"] = format_smc_monitor_section(smc_res, detailed=detailed)
                # SNR: Sinkronkan fokus R1/S1 berdasarkan harga live; hanya turunkan sinyal CONFIRMED ke WAIT.
                if isinstance(snr_res, dict) and isinstance(snr_res.get("decision"), dict):
                    dec = dict(snr_res["decision"])
                    prev_snr_sig = str(dec.get("signal") or "WAIT").upper()
                    snr_scan_orig = snr_res.get("scan_result") if isinstance(snr_res.get("scan_result"), dict) else {}
                    snr_scan = dict(snr_scan_orig)
                    snr_scan["price"] = new_price
                    snr_scan["snapshot_price"] = new_price
                    snr_scan["last_price"] = new_price
                    if prev_snr_sig in ("BUY", "SELL"):
                        tick_refresh_needed = True
                        dec.update(
                            signal="WAIT", direction="NONE", live_entry="WAIT", reason=reval_reason,
                            entry=None, effective_entry=None, sl=None, tp1=None, tp2=None,
                            rr_tp1=None, rr_tp2=None, rr_display="NOT CHECKED",
                        )
                    ranked_cached = snr_scan.get("ranked_zones") or {}
                    mb_cached = snr_scan.get("market_bias") or {}
                    if ranked_cached:
                        live_sel = select_live_snr_focus(ranked_cached, mb_cached, new_price)
                        prev_tz_id = (dec.get("target_zone") or {}).get("zone_id")
                        new_tz = live_sel.get("focus_zone")
                        new_tz_id = new_tz.get("zone_id") if isinstance(new_tz, dict) else None
                        if new_tz_id != prev_tz_id:
                            tick_refresh_needed = True
                            dec["target_zone"] = new_tz
                            dec["zone_label"] = live_sel.get("focus_label", "NONE")
                            dec["focus_selection_reason"] = live_sel.get("focus_selection_reason")
                            dec["nearest_snr_zone"] = live_sel.get("nearest_snr_zone")
                            dec["nearest_snr_label"] = live_sel.get("nearest_snr_label")
                            dec["trend_entry_zone"] = live_sel.get("trend_entry_zone")
                            dec["trend_entry_label"] = live_sel.get("trend_entry_label")
                            ev_copy = dict(snr_scan.get("entry_validation") or {})
                            ev_copy.update(
                                target_zone=new_tz,
                                zone_label=live_sel.get("focus_label", "NONE"),
                                focus_selection_reason=live_sel.get("focus_selection_reason"),
                                nearest_snr_zone=live_sel.get("nearest_snr_zone"),
                                nearest_snr_label=live_sel.get("nearest_snr_label"),
                                trend_entry_zone=live_sel.get("trend_entry_zone"),
                                trend_entry_label=live_sel.get("trend_entry_label"),
                            )
                            snr_scan["entry_validation"] = ev_copy
                    snr_scan["decision"] = dec
                    snr_res["decision"] = dec
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

        positions = pos_list
        if self.focus_tracker is not None and positions is None and mt5 is not None and isinstance(shared_snap, dict) and shared_snap.get("live") is True:
            try:
                positions = mt5.positions_get(symbol=self.symbol)
                if positions is None:
                    from modules.utils import _catat_error
                    _catat_error("pantau_focus_positions", RuntimeError(f"MT5 positions_get gagal: {mt5.last_error()}"))
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                from modules.utils import _catat_error
                _catat_error("pantau_focus_positions", exc)
                positions = None

        acc_res = collect_account_monitor_data(
            symbol=self.symbol,
            snapshot=shared_snap,
            pos_list=positions if self.focus_tracker is not None else pos_list,
            orders_list=orders_list,
            risk_data=risk_data,
        )
        if isinstance(shared_snap, dict) and acc_res.get("orders_snapshot") is not None:
            shared_snap["mt5_orders_snapshot"] = acc_res["orders_snapshot"]
        if isinstance(smc_res, dict) and acc_res.get("orders_snapshot") is not None:
            if isinstance(smc_res.get("decision"), dict):
                smc_res["decision"]["mt5_orders_snapshot"] = acc_res["orders_snapshot"]
            if isinstance(smc_res.get("scan_result"), dict):
                smc_res["scan_result"]["mt5_orders_snapshot"] = acc_res["orders_snapshot"]

        focus_alerts = []
        if self.focus_tracker is not None:
            focus_alerts = self.focus_tracker.update(
                {"SMC": smc_res, "SNR": snr_res},
                shared_snap,
                positions,
                full_refresh=not tick_only,
                orders=acc_res.get("orders"),
            )
            smc_res = self.focus_tracker.gate("SMC", smc_res, detailed=detailed)
            snr_res = self.focus_tracker.gate("SNR", snr_res, detailed=detailed)
            smc_res = self.focus_tracker.scoped_alerts("SMC", smc_res, detailed=detailed)
            snr_res = self.focus_tracker.scoped_alerts("SNR", snr_res, detailed=detailed)
            if isinstance(smc_res, dict) and acc_res.get("orders_snapshot") is not None:
                if isinstance(smc_res.get("decision"), dict):
                    smc_res["decision"]["mt5_orders_snapshot"] = acc_res["orders_snapshot"]
                if isinstance(smc_res.get("scan_result"), dict):
                    smc_res["scan_result"]["mt5_orders_snapshot"] = acc_res["orders_snapshot"]

        acc_section = format_account_monitor_section(acc_res)

        combined_alerts = []
        if smc_res and smc_res.get("alerts"):
            combined_alerts.extend(smc_res["alerts"])
        if snr_res and snr_res.get("alerts"):
            combined_alerts.extend(snr_res["alerts"])
        combined_alerts.extend(focus_alerts)

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
                tr_zid = transition.get("zone_id")
                tr_prev = str(transition.get("prev_status") or "").upper()
                tr_new = str(transition.get("new_status") or "").upper()
                if tr_prev == "APPROACHING" and tr_new == "RADAR":
                    continue
                if self._last_zone_transition.get(tr_zid) == (tr_prev, tr_new):
                    continue
                self._last_zone_transition[tr_zid] = (tr_prev, tr_new)
                combined_alerts.append(
                    f"[ZONE:{tr_zid}] {tr_prev} -> {tr_new}"
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

        news_eval = {
            "status": "UNAVAILABLE",
            "risk": "UNKNOWN",
            "summary": "NEWS RISK : UNKNOWN | NEXT EVENT : UNAVAILABLE",
            "gold_news_status": "UNAVAILABLE",
            "gold_news_summary": "GOLD NEWS : UNAVAILABLE",
            "gold_articles": [],
            "next_event": None,
            "alerts": [],
            "active_alerts": [],
        }
        if self.news_alert_engine is not None:
            try:
                news_eval = self.news_alert_engine.evaluate()
            except Exception:
                pass
        news_summary = str(news_eval.get("summary") or "NEWS RISK : UNKNOWN | NEXT EVENT : UNAVAILABLE")
        gold_news_summary = str(news_eval.get("gold_news_summary") or "").strip()
        news_alerts = list(news_eval.get("alerts") or [])

        price_txt = f"{price_disp:.2f}" if price_disp > 0 else "DATA UNAVAILABLE"
        report_lines = [
            "====================================================",
            f"   {header_title}",
            "====================================================",
            f"SYMBOL : {self.symbol} | PRICE : {price_txt} | TIME : {ts_str} | CYCLE : #{self.cycle_count}",
            f"SNAPSHOT ID : {snap_id_disp} | SAME={same_snap_disp} | DIFF={p_diff_disp:.2f} pts",
            f"FRESHNESS   : SMC={fresh_smc_disp} | SNR={fresh_snr_disp}",
            news_summary,
        ]
        if gold_news_summary:
            report_lines.append(gold_news_summary)
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

        if news_alerts:
            report_lines.append("----------------------------------------------------")
            for na in news_alerts:
                report_lines.append(na)

        report_lines.append("====================================================")
        formatted_report = "\n".join(report_lines)

        categorized_alerts = classify_monitor_alerts(
            combined_alerts,
            risk_warnings=acc_res.get("risk_warnings") or [],
            default_ts=ts_str,
        )

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
            "categorized_alerts": categorized_alerts,
            "news": news_eval,
            "news_alerts": news_alerts,
            "news_summary": news_summary,
            "gold_news_summary": gold_news_summary,
            "risk_warnings": acc_res.get("risk_warnings") or [],
            "state_changed": False,
            "refresh_required": bool(
                tick_refresh_needed
                or (self.focus_tracker is not None and self.focus_tracker.refresh_required)
            ),
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
                    "ever_taken": bool(record.get("ever_taken")),
                    "pending_ticket": record.get("pending_ticket"),
                    "account_status": record["account_status"],
                    "uncertain": record["uncertain"],
                    "plan_valid": self.focus_tracker._valid_plan(record.get("plan")),
                }
                for method, record in self.focus_tracker.focus.items()
                if record and method in self.focus_tracker.active_methods
            } if self.focus_tracker is not None else {},
            "focus_validation_fresh": not tick_only,
            "completed_focus": {
                method: dict(comp)
                for method, comp in self.focus_tracker.completed.items()
                if isinstance(comp, dict)
            } if self.focus_tracker is not None else {},
        }
        smart_view = _smart_monitor_context(cycle_result)
        smc_zone_id = str((smc_res.get("summary") or {}).get("primary_zone_id") or "NONE") if smc_res else None
        snr_zone_obj = smart_view.get("snr_focus_zone") or smart_view.get("snr_zone") or ((snr_res.get("decision") or {}).get("target_zone") if snr_res else None) or {}
        snr_zone_id = str(snr_zone_obj.get("zone_id") or "NONE") if snr_res else None
        snr_confl_src = smart_view.get("snr_confluence_source")
        snr_confl_src_id = str(snr_confl_src.get("zone_id") or "NONE") if isinstance(snr_confl_src, dict) else "NONE"

        active_zone = smart_view.get("zone") or {}
        active_zone_src = str(active_zone.get("source") or ("SMC" if active_zone else "NONE")).upper()
        active_zone_id = str(active_zone.get("zone_id") or "NONE")
        active_zone_lo = round(safe_float(active_zone.get("price_low", active_zone.get("low")), 0.0), 2)
        active_zone_hi = round(safe_float(active_zone.get("price_high", active_zone.get("high")), 0.0), 2)
        active_is_smc = (mode_norm == "unified" and active_zone_src == "SMC" and active_zone_lo > 0)

        raw_non_hist_alerts = (
            list(categorized_alerts.get("entry") or [])
            + list(categorized_alerts.get("market") or [])
            + list(categorized_alerts.get("account") or [])
        )
        material_alerts = []
        inline_alerts = []
        for al_msg in raw_non_hist_alerts:
            if _is_snr_touch_or_exit_alert(al_msg, active_is_smc=active_is_smc):
                if al_msg not in inline_alerts:
                    inline_alerts.append(al_msg)
            else:
                if al_msg not in material_alerts:
                    material_alerts.append(al_msg)

        # Deteksi satu-baris sentuh/keluar zona SNR live (termasuk pada siklus tick_only) tanpa mencetak ulang dashboard.
        snap_fresh_up = str((shared_snap or {}).get("data_freshness") or "").upper()
        snap_mkt_up = str((shared_snap or {}).get("market_status") or "").upper()
        live_snr_trackable = (
            mode_norm in ("unified", "snr")
            and isinstance(snr_res, dict)
            and snr_res.get("status") == "OK"
            and isinstance(shared_snap, dict)
            and shared_snap.get("data_ok") is not False
            and snap_fresh_up not in ("STALE", "INVALID", "NO DATA")
            and snap_mkt_up not in ("CLOSED", "DATA UNAVAILABLE")
            and price_disp > 0
        )
        if live_snr_trackable:
            prev_snr_live = self.last_snr_live_zone_state
            s_lo_live = safe_float(snr_zone_obj.get("low", snr_zone_obj.get("price_low")), 0.0)
            s_hi_live = safe_float(snr_zone_obj.get("high", snr_zone_obj.get("price_high")), 0.0)
            s_lbl_live = str(snr_zone_obj.get("rank_label") or snr_zone_obj.get("zone_type") or "SNR").strip()
            s_zid_live = str(snr_zone_obj.get("zone_id") or (f"{s_lbl_live}:{s_lo_live:.2f}-{s_hi_live:.2f}" if s_lo_live > 0 else "NONE"))

            # Deduplikasi echo antara tick-level event dan full-refresh alert untuk zona & arah transisi yang sama
            if inline_alerts:
                dedup_inline = []
                for ial in inline_alerts:
                    ial_up = str(ial).upper()
                    ev_kind = "EXIT" if ("DI LUAR ZONA" in ial_up or "KELUAR" in ial_up) else (
                        "TOUCH" if ("HARGA DI " in ial_up or "HARGA SENTUH" in ial_up) else None
                    )
                    if ev_kind and self.last_snr_inline_event == (s_zid_live, ev_kind):
                        continue
                    dedup_inline.append(ial)
                    if ev_kind:
                        self.last_snr_inline_event = (s_zid_live, ev_kind)
                inline_alerts = dedup_inline

            has_inline_exit = any(
                "KELUAR" in str(x).upper() or "DI LUAR ZONA" in str(x).upper() for x in inline_alerts
            )
            has_inline_touch = any(
                "HARGA DI " in str(x).upper() or "HARGA SENTUH" in str(x).upper()
                for x in (inline_alerts + material_alerts)
            )

            if prev_snr_live and prev_snr_live.get("inside"):
                p_lo_l, p_hi_l = prev_snr_live["low"], prev_snr_live["high"]
                if price_disp < p_lo_l or price_disp > p_hi_l:
                    exit_key = (prev_snr_live["zone_id"], "EXIT")
                    if not has_inline_exit and self.last_snr_inline_event != exit_key:
                        inline_alerts.append(
                            f"[SNR] HARGA KELUAR {prev_snr_live['label']} ({p_lo_l:.2f}-{p_hi_l:.2f}) | {price_disp:.2f}"
                        )
                        self.last_snr_inline_event = exit_key

            if s_lo_live > 0 and s_hi_live >= s_lo_live:
                now_inside_snr = (s_lo_live <= price_disp <= s_hi_live)
                was_same_inside = bool(
                    prev_snr_live and prev_snr_live.get("zone_id") == s_zid_live and prev_snr_live.get("inside")
                )
                if now_inside_snr and not was_same_inside and self.cycle_count > 1:
                    touch_key = (s_zid_live, "TOUCH")
                    if not has_inline_touch and self.last_snr_inline_event != touch_key:
                        inline_alerts.append(
                            f"[SNR] HARGA SENTUH {s_lbl_live} ({s_lo_live:.2f}-{s_hi_live:.2f}) | {price_disp:.2f}"
                        )
                        self.last_snr_inline_event = touch_key
                self.last_snr_live_zone_state = {
                    "zone_id": s_zid_live,
                    "label": s_lbl_live,
                    "low": s_lo_live,
                    "high": s_hi_live,
                    "inside": now_inside_snr,
                }
            else:
                self.last_snr_live_zone_state = None
        else:
            self.last_snr_live_zone_state = None

        norm_risk_warns = []
        for rw_item in list(acc_res.get("risk_warnings") or []) + list(smart_view.get("risk_alerts") or []):
            rw_norm = _normalize_risk_warning_sig(rw_item)
            if rw_norm and rw_norm not in norm_risk_warns:
                norm_risk_warns.append(rw_norm)
        risk_sig = (
            _normalize_risk_warning_sig(smart_view.get("critical_risk_display") or ""),
            tuple(norm_risk_warns),
            int(safe_float(acc_res.get("position_without_sl"), 0)),
            int(safe_float(acc_res.get("pending_without_sl"), 0)),
            bool(spread_state is True),
        )

        active_focus_mt5_sig = tuple(
            sorted(
                (m_k, m_v)
                for m_k, m_v in (cycle_result.get("focus_status") or {}).items()
                if str(m_v or "").upper() not in ("", "NO MATCHING POSITION")
            )
        )
        mt5_status_sig = (
            current_connection,
            bool(acc_res.get("available", False)),
            int(safe_float(acc_res.get("open_positions"), 0)),
            int(safe_float(acc_res.get("pending_orders"), 0)),
            round(safe_float(acc_res.get("buy_lots"), 0.0), 2),
            round(safe_float(acc_res.get("sell_lots"), 0.0), 2),
            _extract_mt5_items_sig(positions, is_order=False),
            _extract_mt5_items_sig(acc_res.get("orders"), is_order=True),
            active_focus_mt5_sig,
        )
        if self.last_mt5_status_sig is not None and mt5_status_sig != self.last_mt5_status_sig:
            has_mt5_msg = any(
                tok in str(al).upper()
                for al in material_alerts
                for tok in ("MT5", "POSISI", "RUNNING", "PENDING")
            )
            if not has_mt5_msg:
                nosl_total = int(safe_float(acc_res.get("position_without_sl"), 0)) + int(safe_float(acc_res.get("pending_without_sl"), 0))
                material_alerts.append(
                    f"[MT5] STATUS AKUN/ORDER BERUBAH (Conn={current_connection} | "
                    f"Posisi={acc_res.get('open_positions', 0)} | Pending={acc_res.get('pending_orders', 0)} | "
                    f"Tanpa SL={nosl_total})"
                )
        self.last_mt5_status_sig = mt5_status_sig

        plan_sig = _build_plan_signature(
            smart_view.get("pending_setup"),
            smart_view.get("alt_pending_setups"),
        )
        curr_sig_tuple = (
            mode_norm,
            smc_sig,
            snr_sig,
            smart_view.get("signal") if mode_norm == "unified" else None,
            smart_view.get("confirmed") if mode_norm == "unified" else None,
            smc_zone_id if mode_norm == "smc" else None,
            snr_zone_id if (mode_norm == "snr" or not active_is_smc) else None,
            (active_zone_src, active_zone_id, smart_view.get("direction"), active_zone_lo, active_zone_hi) if mode_norm == "unified" else None,
            smc_res.get("status") if smc_res else None,
            snr_res.get("status") if snr_res else None,
            smart_view.get("monitor_state") if mode_norm == "unified" else None,
            smart_view.get("trigger") if mode_norm == "unified" else None,
            smart_view.get("confluence") if mode_norm == "unified" else None,
            snr_confl_src_id if mode_norm == "unified" else None,
            smart_view.get("pending_status") if mode_norm == "unified" else None,
            plan_sig if mode_norm == "unified" else None,
            smart_view.get("snr_breakout_watch") if mode_norm == "unified" else None,
            risk_sig,
            mt5_status_sig,
        )
        state_changed = (
            self.last_combined_signature is None
            or curr_sig_tuple != self.last_combined_signature
            or bool(material_alerts)
        )
        self.last_combined_signature = curr_sig_tuple
        cycle_result["state_changed"] = state_changed
        cycle_result["material_alerts"] = material_alerts
        cycle_result["inline_alerts"] = inline_alerts
        cycle_result["risk_signature"] = risk_sig
        cycle_result["smart_view"] = smart_view
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
            if cycle_out.get("refresh_required") or (monitor.focus_tracker is not None and monitor.focus_tracker.refresh_required):
                last_full_refresh = 0.0
            last_out = cycle_out
            cycles_done += 1

            # Cetak laporan penuh pada siklus pertama, atau ketika terjadi perubahan state material (SIGNAL/ZONE/STATE/alert baru)
            output = cycle_out.get("detail_report") if detailed and mode == "unified" else (
                cycle_out.get("compact_report") if mode == "unified" else cycle_out["formatted_report"]
            )
            news_alerts_cycle = cycle_out.get("news_alerts") or []
            view = cycle_out.get("smart_view") or _smart_monitor_context(cycle_out)
            eff_risk_warnings = []
            for rw_item in list(cycle_out.get("risk_warnings") or []) + list(view.get("risk_alerts") or []):
                rw_txt = str(rw_item or "").strip()
                if rw_txt and rw_txt not in eff_risk_warnings:
                    eff_risk_warnings.append(rw_txt)
            risk_signature = cycle_out.get("risk_signature")
            if risk_signature is None:
                risk_signature = tuple(_normalize_risk_warning_sig(rw) for rw in eff_risk_warnings)
            material_alerts_cycle = cycle_out.get("material_alerts")
            if material_alerts_cycle is None:
                material_alerts_cycle = list(cycle_out.get("alerts") or [])
            inline_alerts_cycle = list(cycle_out.get("inline_alerts") or [])

            if cycles_done == 1:
                for al in material_alerts_cycle:
                    print_terminal(al)
                print_terminal("\n" + output)
                if mode == "unified" and not detailed and news_alerts_cycle:
                    for na in news_alerts_cycle:
                        print_terminal(na)
                last_risk_signature = risk_signature
                if eff_risk_warnings:
                    last_risk_emit = time.monotonic()
            elif cycle_out.get("state_changed"):
                if status_bar_active:
                    print_terminal()
                    status_bar_active = False
                for al in material_alerts_cycle:
                    print_terminal(al)
                if risk_signature != last_risk_signature and eff_risk_warnings:
                    for rw in eff_risk_warnings:
                        if rw not in material_alerts_cycle:
                            print_terminal(f"[ACCOUNT RISK] {rw}")
                print_terminal("\n" + output)
                if mode == "unified" and not detailed and news_alerts_cycle:
                    for na in news_alerts_cycle:
                        print_terminal(na)
                last_risk_signature = risk_signature
                if eff_risk_warnings:
                    last_risk_emit = time.monotonic()
            else:
                if inline_alerts_cycle:
                    if status_bar_active:
                        print_terminal()
                        status_bar_active = False
                    snap_ts_raw = str(cycle_out.get("snapshot_time") or "").strip()
                    ts_short = snap_ts_raw[11:19] + " WIB" if len(snap_ts_raw) >= 19 else datetime.now().strftime("%H:%M:%S WIB")
                    for ial in inline_alerts_cycle:
                        print_terminal(f"[{ts_short}] {ial}")
                if news_alerts_cycle:
                    if status_bar_active:
                        print_terminal()
                        status_bar_active = False
                    for na in news_alerts_cycle:
                        print_terminal(na)
                if risk_signature != last_risk_signature:
                    if eff_risk_warnings:
                        if status_bar_active:
                            print_terminal()
                            status_bar_active = False
                        for rw in eff_risk_warnings:
                            print_terminal(f"[ACCOUNT RISK] {rw}")
                        last_risk_emit = time.monotonic()
                    last_risk_signature = risk_signature

                snap_cur = cycle_out.get("snapshot") if isinstance(cycle_out.get("snapshot"), dict) else {}
                quote_ok = (
                    snap_cur.get("data_ok") is not False
                    and str(snap_cur.get("data_freshness") or "").upper() not in ("STALE", "INVALID", "NO DATA")
                )
                price_num = safe_float(view.get("price"), safe_float(snap_cur.get("snapshot_price"), 0.0))
                price_bar = f"{price_num:.2f}" if price_num > 0 and (quote_ok or not view["data_invalid"]) else "N/A"
                has_sl_missing = bool(
                    int(safe_float((cycle_out.get("account") or {}).get("position_without_sl"), 0)) > 0
                    or int(safe_float((cycle_out.get("account") or {}).get("pending_without_sl"), 0)) > 0
                    or any("SL MISSING" in str(rw).upper() or "TANPA STOP LOSS" in str(rw).upper() for rw in eff_risk_warnings)
                )
                risk_tag = " | [RISK WARNING] SL MISSING" if has_sl_missing else (" | [RISK WARNING]" if eff_risk_warnings else "")
                news_risk_lbl = str((cycle_out.get("news") or {}).get("risk") or "")
                news_tag = f" | NEWS {news_risk_lbl}" if news_risk_lbl in ("HIGH", "MEDIUM") else ""

                if mode == "unified":
                    z_bar = view.get("zone") or {}
                    z_lo_b = safe_float(z_bar.get("price_low", z_bar.get("low")), 0.0)
                    z_hi_b = safe_float(z_bar.get("price_high", z_bar.get("high")), 0.0)
                    zone_tag = f" | {view.get('direction', '')} {z_lo_b:.2f}-{z_hi_b:.2f}".strip() if (z_lo_b > 0 and z_hi_b > 0) else ""
                    if zone_tag:
                        zone_tag = " | " + zone_tag.lstrip("| ")
                    status_bar = (
                        f"[SMART MONITOR] {price_bar} | {view['monitor_state']} | "
                        f"SIG {view['confirmed']}{risk_tag}{zone_tag}{news_tag}"
                    )
                elif mode == "smc":
                    status_bar = (
                        f"[SMART MONITOR] SMC {price_bar} | SIG {cycle_out.get('smc_signal', 'WAIT')}{risk_tag}{news_tag}"
                    )
                else:
                    snr_tz = ((cycle_out.get("snr") or {}).get("decision") or {}).get("target_zone") or {}
                    snr_lo_b = safe_float(snr_tz.get("low"), 0.0)
                    snr_hi_b = safe_float(snr_tz.get("high"), 0.0)
                    snr_z_tag = f" | {snr_tz.get('rank_label', 'SNR')} {snr_lo_b:.2f}-{snr_hi_b:.2f}" if (snr_lo_b > 0 and snr_hi_b > 0) else ""
                    status_bar = (
                        f"[SNR MONITOR] {price_bar} | SIG {cycle_out.get('snr_signal', 'WAIT')}{snr_z_tag}{risk_tag}{news_tag}"
                    )
                bar_width = max(40, shutil.get_terminal_size((100, 20)).columns - 1)
                if len(status_bar) <= bar_width:
                    line_out = status_bar.ljust(bar_width)
                else:
                    # Jangan pernah memotong tag risiko kritis jika lebar terminal sempit
                    compact_core = (
                        f"[SMART MONITOR] {price_bar} | {view['monitor_state']} | SIG {view['confirmed']}{risk_tag}"
                        if mode == "unified" else status_bar
                    )
                    line_out = compact_core[:bar_width].ljust(bar_width) if len(compact_core) > bar_width else compact_core.ljust(bar_width)
                print_terminal("\r" + line_out, end="", flush=True)
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
            print_terminal()
        print_terminal("\n[UNIFIED MONITOR] Pemantauan dihentikan oleh pengguna.")

    if status_bar_active and max_cycles is not None and cycles_done >= int(max_cycles):
        print_terminal()
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
