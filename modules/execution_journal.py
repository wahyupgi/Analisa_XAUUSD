# -*- coding: utf-8 -*-
"""
modules/execution_journal.py
Execution Journal & Strategy Performance Analytics (Section 19):
- Mencatat setiap keputusan eksekusi, order terpasang, posisi terbuka/tertutup, missed entry,
  blocked entry, rejection broker, dan execution latency ke file CSV terstruktur.
- Menyediakan analitik lengkap: Win rate, Expectancy, Profit factor, Max drawdown, Average RR,
  performa per Entry Mode / Zone Timeframe / Zone Score / Lot Classification, Missed Entry,
  Blocked Entry, Rejection Rate, dan rata-rata Execution Latency.
"""
from __future__ import annotations

import csv
import math
import os
import threading
from typing import Any, Dict, List, Optional

from modules.config import FILE_JURNAL_AUTO_EXECUTION
from modules.utils import _catat_error, format_wib_timestamp


AUTO_JOURNAL_COLUMNS = [
    "timestamp",
    "event_type",
    "trading_mode",
    "account_mode",
    "currency",
    "setup_id",
    "zone_id",
    "symbol",
    "direction",
    "zone_timeframe",
    "entry_mode",
    "h4_bias",
    "h1_bias",
    "m30_bias",
    "m5_bias",
    "zone_score",
    "zone_grade",
    "zone_quality",
    "trigger",
    "planned_entry",
    "actual_entry",
    "sl",
    "tp",
    "target_lot",
    "final_lot",
    "actual_lot",
    "lot_adjustment_reason",
    "risk_nominal",
    "risk_percent",
    "planned_rr",
    "actual_rr",
    "spread",
    "slippage",
    "commission",
    "swap",
    "mt5_retcode",
    "order_ticket",
    "position_ticket",
    "decision_reason",
    "realized_pnl",
    "exit_reason",
    "execution_latency_ms",
]


class ExecutionJournal:
    """Pencatat jurnal eksekusi otomatis CSV thread-safe beserta evaluator metrik kuantitatif."""

    def __init__(self, journal_path: Optional[str] = None):
        self.journal_path = journal_path or FILE_JURNAL_AUTO_EXECUTION
        self._lock = threading.RLock()
        self._dedup_recent: Dict[str, float] = {}

    def _ensure_header_locked(self) -> None:
        parent = os.path.dirname(os.path.abspath(self.journal_path))
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)
        if not os.path.exists(self.journal_path) or os.path.getsize(self.journal_path) == 0:
            with open(self.journal_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=AUTO_JOURNAL_COLUMNS)
                writer.writeheader()

    def append_row(self, row_dict: Dict[str, Any]) -> bool:
        with self._lock:
            try:
                self._ensure_header_locked()
                clean_row = {col: row_dict.get(col, "") for col in AUTO_JOURNAL_COLUMNS}
                if not clean_row["timestamp"]:
                    clean_row["timestamp"] = format_wib_timestamp(default_now=True)
                with open(self.journal_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=AUTO_JOURNAL_COLUMNS)
                    writer.writerow(clean_row)
                return True
            except Exception as exc:
                _catat_error("execution_journal_append", exc)
                return False

    def record_execution_event(
        self,
        event_type: str,
        validation_result: Dict[str, Any],
        trading_mode: str = "",
        account_info: Optional[Dict[str, Any]] = None,
        actual_entry: float = 0.0,
        actual_lot: float = 0.0,
        actual_sl: float = 0.0,
        actual_tp: float = 0.0,
        actual_rr: float = 0.0,
        spread: float = 0.0,
        slippage: float = 0.0,
        commission: float = 0.0,
        swap: float = 0.0,
        retcode: int = 0,
        order_ticket: int = 0,
        position_ticket: int = 0,
        decision_reason: str = "",
        realized_pnl: Optional[float] = None,
        exit_reason: str = "",
        latency_ms: float = 0.0,
    ) -> bool:
        """Catat satu baris event eksekusi lengkap ke jurnal CSV (anti-spam untuk event BLOCKED berulang)."""
        import time as _t

        setup_id = str(validation_result.get("setup_id") or "")
        ev_u = str(event_type or "EVENT").upper()
        rsn_str = str(decision_reason or validation_result.get("reason_code") or "")

        # Cegah penulisan berulang tiap detik untuk setup BLOCKED / MISSED yang identik
        if ev_u in ("BLOCKED", "BLOCKED_ENTRY", "MISSED_ENTRY"):
            dedup_key = f"{ev_u}|{setup_id}|{rsn_str}"
            now_s = _t.time()
            with self._lock:
                last_s = self._dedup_recent.get(dedup_key, 0.0)
                if (now_s - last_s) < 300.0:
                    return False
                self._dedup_recent[dedup_key] = now_s
                if len(self._dedup_recent) > 300:
                    oldest_k = min(self._dedup_recent, key=self._dedup_recent.get)
                    self._dedup_recent.pop(oldest_k, None)

        acc = account_info or validation_result.get("account") or {}
        plan = validation_result.get("trade_plan") or {}
        bias = validation_result.get("market_bias") or {}
        zone = validation_result.get("zone") or {}
        triggers = validation_result.get("triggers") or {}

        row = {
            "timestamp": format_wib_timestamp(default_now=True),
            "event_type": ev_u,
            "trading_mode": trading_mode or validation_result.get("trading_mode", ""),
            "account_mode": acc.get("account_mode", ""),
            "currency": acc.get("currency", ""),
            "setup_id": setup_id,
            "zone_id": validation_result.get("zone_id") or plan.get("zone_id", ""),
            "symbol": validation_result.get("symbol") or plan.get("symbol", ""),
            "direction": plan.get("direction", ""),
            "zone_timeframe": zone.get("timeframe") or plan.get("zone_timeframe", ""),
            "entry_mode": validation_result.get("entry_mode") or plan.get("entry_mode", ""),
            "h4_bias": bias.get("H4", ""),
            "h1_bias": bias.get("H1", ""),
            "m30_bias": bias.get("M30", ""),
            "m5_bias": bias.get("M5", ""),
            "zone_score": f"{float(zone.get('score') or plan.get('zone_score') or 0.0):.1f}",
            "zone_grade": zone.get("grade") or plan.get("zone_grade", ""),
            "zone_quality": validation_result.get("zone_quality", ""),
            "trigger": triggers.get("primary_trigger", ""),
            "planned_entry": f"{float(plan.get('entry') or 0.0):.2f}",
            "actual_entry": f"{float(actual_entry):.2f}" if actual_entry > 0 else "",
            "sl": f"{float(actual_sl if actual_sl > 0 else (plan.get('sl') or 0.0)):.2f}",
            "tp": f"{float(actual_tp if actual_tp > 0 else (plan.get('tp') or 0.0)):.2f}",
            "target_lot": f"{float(validation_result.get('target_lot') or 0.0):.2f}",
            "final_lot": f"{float(validation_result.get('final_lot') or 0.0):.2f}",
            "actual_lot": f"{float(actual_lot):.2f}" if actual_lot > 0 else "",
            "lot_adjustment_reason": validation_result.get("lot_reason", ""),
            "risk_nominal": f"{float(validation_result.get('estimated_loss') or 0.0):.2f}",
            "risk_percent": f"{float(validation_result.get('risk_percent') or 0.0):.4f}",
            "planned_rr": f"{float(plan.get('planned_rr') or 0.0):.2f}",
            "actual_rr": f"{float(actual_rr if actual_rr > 0 else (validation_result.get('actual_rr') or 0.0)):.2f}",
            "spread": f"{float(spread if spread > 0 else (validation_result.get('spread') or 0.0)):.3f}",
            "slippage": f"{float(slippage):.3f}",
            "commission": f"{float(commission):.2f}",
            "swap": f"{float(swap):.2f}",
            "mt5_retcode": str(retcode) if retcode else "",
            "order_ticket": str(order_ticket) if order_ticket else "",
            "position_ticket": str(position_ticket) if position_ticket else "",
            "decision_reason": rsn_str,
            "realized_pnl": f"{float(realized_pnl):.2f}" if realized_pnl is not None else "",
            "exit_reason": exit_reason,
            "execution_latency_ms": f"{float(latency_ms):.2f}" if latency_ms > 0 else "",
        }
        return self.append_row(row)

    def record_trade_closed(
        self,
        setup_rec: Dict[str, Any],
        realized_pnl: float,
        exit_reason: str,
        commission: float = 0.0,
        swap: float = 0.0,
        account_mode: str = "",
        currency: str = "",
    ) -> bool:
        """Catat penutupan posisi scanner ke jurnal agar evaluasi Win Rate / Expectancy / Drawdown akurat."""
        row = {
            "timestamp": format_wib_timestamp(default_now=True),
            "event_type": "POSITION_CLOSED",
            "trading_mode": setup_rec.get("trading_mode", "FULL_AUTO"),
            "account_mode": account_mode,
            "currency": currency,
            "setup_id": setup_rec.get("setup_id", ""),
            "zone_id": setup_rec.get("zone_id", ""),
            "symbol": setup_rec.get("symbol", ""),
            "direction": setup_rec.get("direction", ""),
            "zone_timeframe": setup_rec.get("zone_timeframe", ""),
            "entry_mode": setup_rec.get("entry_mode", ""),
            "zone_score": f"{float(setup_rec.get('zone_score') or 0.0):.1f}",
            "zone_grade": setup_rec.get("zone_grade", ""),
            "zone_quality": setup_rec.get("zone_quality", ""),
            "trigger": setup_rec.get("trigger_name", ""),
            "planned_entry": f"{float(setup_rec.get('planned_entry') or 0.0):.2f}",
            "actual_entry": f"{float(setup_rec.get('actual_entry') or 0.0):.2f}",
            "sl": f"{float(setup_rec.get('sl') or 0.0):.2f}",
            "tp": f"{float(setup_rec.get('tp') or 0.0):.2f}",
            "target_lot": f"{float(setup_rec.get('target_lot') or 0.0):.2f}",
            "final_lot": f"{float(setup_rec.get('final_lot') or 0.0):.2f}",
            "actual_lot": f"{float(setup_rec.get('actual_lot') or setup_rec.get('final_lot') or 0.0):.2f}",
            "commission": f"{float(commission):.2f}",
            "swap": f"{float(swap):.2f}",
            "order_ticket": str(setup_rec.get("order_ticket") or ""),
            "position_ticket": str(setup_rec.get("position_ticket") or ""),
            "decision_reason": exit_reason,
            "realized_pnl": f"{float(realized_pnl):.2f}",
            "exit_reason": exit_reason,
        }
        return self.append_row(row)

    def read_rows(self) -> List[Dict[str, str]]:
        with self._lock:
            if not os.path.exists(self.journal_path):
                return []
            try:
                with open(self.journal_path, "r", encoding="utf-8") as f:
                    return list(csv.DictReader(f))
            except Exception as exc:
                _catat_error("execution_journal_read", exc)
                return []

    def export_summary(self, period: str = "weekly", output_path: str | None = None) -> None:
        """Ekspor ringkasan statistik jurnal ke file CSV.

        * ``period``: ``"weekly"`` atau ``"monthly"`` menentukan cara mengelompokkan
          baris berdasarkan kolom ``timestamp``.
        * ``output_path``: Jika diberikan, hasil disimpan ke file tersebut; bila ``None``
          (default) akan dibuat file ``jurnal_eksekusi_{period}_summary.csv`` di direktori yang sama.

        Ringkasan mencakup:
        - total trade (baris ``ORDER_PLACED`` atau ``POSITION_CLOSED``)
        - net P&L (kolom ``realized_pnl``)
        - rata‑rata ``actual_rr``
        - win rate (jumlah baris dengan ``actual_rr`` > 0)
        """
        import csv, datetime, collections, math
        from dateutil import parser as dt_parser

        # Pastikan header ada
        with self._lock:
            if not os.path.exists(self.journal_path):
                print("[WARNING] Jurnal tidak ditemukan, tidak dapat membuat ringkasan.")
                return

            rows = self.read_rows()

        # Filter hanya baris eksekusi
        exec_rows = [r for r in rows if r.get("event_type") in ("ORDER_PLACED", "POSITION_CLOSED")]
        if not exec_rows:
            print("[INFO] Tidak ada trade yang dieksekusi untuk diringkas.")
            return

        # Fungsi bantu untuk mengubah timestamp ke period
        def period_key(ts: str) -> str:
            dt = dt_parser.parse(ts)
            if period.lower() == "weekly":
                # ISO week number
                return f"{dt.isocalendar()[0]}-W{dt.isocalendar()[1]:02d}"
            elif period.lower() == "monthly":
                return f"{dt.year}-{dt.month:02d}"
            else:  # daily
                return dt.date().isoformat()

            dt = dt_parser.parse(ts)
            if period.lower() == "weekly":
                # ISO week number
                return f"{dt.isocalendar()[0]}-W{dt.isocalendar()[1]:02d}"
            else:  # monthly
                return f"{dt.year}-{dt.month:02d}"

        groups = collections.defaultdict(list)
        for r in exec_rows:
            key = period_key(r.get("timestamp", ""))
            groups[key].append(r)

        # Tentukan path output
        if output_path:
            out_path = output_path
        else:
            base_dir = os.path.dirname(self.journal_path)
            out_path = os.path.join(base_dir, f"jurnal_eksekusi_{period}_summary.csv")

        # Tulis ringkasan
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["period", "trades", "net_pnl", "avg_rr", "win_rate_pct"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for key, rows in sorted(groups.items()):
                trades = len(rows)
                pnl_vals = []
                rr_vals = []
                win_cnt = 0
                for r in rows:
                    # realized_pnl may be empty
                    try:
                        pnl = float(r.get("realized_pnl", "0") or 0)
                    except ValueError:
                        pnl = 0.0
                    pnl_vals.append(pnl)
                    try:
                        rr = float(r.get("actual_rr", "0") or 0)
                    except ValueError:
                        rr = 0.0
                    rr_vals.append(rr)
                    if rr > 0:
                        win_cnt += 1
                net_pnl = sum(pnl_vals)
                avg_rr = sum(rr_vals) / len(rr_vals) if rr_vals else 0.0
                win_rate = (win_cnt / trades) * 100 if trades else 0.0
                writer.writerow({
                    "period": key,
                    "trades": trades,
                    "net_pnl": round(net_pnl, 2),
                    "avg_rr": round(avg_rr, 2),
                    "win_rate_pct": round(win_rate, 1),
                })
        print(f"[INFO] Ringkasan {period} selesai, disimpan ke: {out_path}")

        """Menulis ulang CSV jurnal eksekusi sehingga hanya menyimpan baris yang
        mewakili trade yang sebenarnya dieksekusi (ORDER_PLACED atau
        POSITION_CLOSED).

        Jika ``output_path`` diberikan, hasil akan ditulis ke file baru dengan
        nama tersebut; bila ``None`` (default) file jurnal yang asli akan
        ditimpa.

        Metode ini memperoleh ``self._lock`` untuk menghindari konflik dengan
        proses auto‑trading yang mungkin sedang menulis ke file. Jika file
        sedang terkunci, akan ditangkap ``PermissionError`` dan fungsi akan
        menghentikan penulisan serta mencatat error.
        """
        import csv
        with self._lock:
            if not os.path.exists(self.journal_path):
                return
            # Baca semua baris yang ada
            with open(self.journal_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = [row for row in reader if row.get("event_type") in ("ORDER_PLACED", "POSITION_CLOSED")]
            # Tentukan path tujuan
            target_path = output_path or self.journal_path
            # Tulis kembali hanya baris yang terfilter
            try:
                with open(target_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=AUTO_JOURNAL_COLUMNS)
                    writer.writeheader()
                    for row in rows:
                        clean_row = {col: row.get(col, "") for col in AUTO_JOURNAL_COLUMNS}
                        writer.writerow(clean_row)
            except PermissionError as e:
                # File mungkin masih dibuka oleh auto‑trading; catat dan beri tahu pengguna.
                _catat_error("compact_executed_journal_permission", e)
                print("[WARNING] Tidak dapat menulis ke jurnal karena file sedang dikunci."
                      " Harap hentikan auto‑trading (auto off) dan jalankan kembali skrip.")
                return
        """Menulis ulang CSV jurnal eksekusi sehingga hanya menyimpan baris yang
        mewakili trade yang sebenarnya dieksekusi (ORDER_PLACED atau
        POSITION_CLOSED).

        Metode ini memperoleh ``self._lock`` untuk menghindari konflik dengan
        proses auto‑trading yang mungkin sedang menulis ke file. Jika file
        sedang terkunci oleh proses lain, akan ditangkap ``PermissionError``
        dan fungsi akan menghentikan penulisan serta mencatat error.
        """
        import csv
        with self._lock:
            if not os.path.exists(self.journal_path):
                return
            # Baca semua baris yang ada
            with open(self.journal_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = [row for row in reader if row.get("event_type") in ("ORDER_PLACED", "POSITION_CLOSED")]
            # Tulis kembali hanya baris yang terfilter
            try:
                with open(self.journal_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=AUTO_JOURNAL_COLUMNS)
                    writer.writeheader()
                    for row in rows:
                        clean_row = {col: row.get(col, "") for col in AUTO_JOURNAL_COLUMNS}
                        writer.writerow(clean_row)
            except PermissionError as e:
                # File mungkin masih dibuka oleh auto‑trading; catat dan beri tahu pengguna.
                _catat_error("compact_executed_journal_permission", e)
                print("[WARNING] Tidak dapat menulis ke jurnal karena file sedang dikunci."
                      " Harap hentikan auto‑trading (auto off) dan jalankan kembali skrip.")
                return
        """Rewrite the execution journal CSV to keep only rows representing
        actual executed trades.

        Executed rows are those with ``event_type`` equal to ``ORDER_PLACED``
        or ``POSITION_CLOSED``. The method acquires the internal lock to avoid
        race conditions with the running auto‑trading process.
        """
        import csv
        with self._lock:
            if not os.path.exists(self.journal_path):
                return
            # Read all rows
            with open(self.journal_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = [row for row in reader if row.get("event_type") in ("ORDER_PLACED", "POSITION_CLOSED")]
            # Write back filtered rows
            with open(self.journal_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=AUTO_JOURNAL_COLUMNS)
                writer.writeheader()
                for row in rows:
                    clean_row = {col: row.get(col, "") for col in AUTO_JOURNAL_COLUMNS}
                    writer.writerow(clean_row)

    def compute_analytics(self) -> Dict[str, Any]:
        """
        Evaluasi kuantitatif menyeluruh (Section 19):
        - Win rate, Expectancy, Profit Factor, Maximum Drawdown, Average RR
        - Breakdown performa per Entry Mode, Zone Timeframe, Zone Score Band, dan Lot Classification
        - Missed entries, Blocked/Invalid entries, Rejection rate, dan Average Execution Latency.
        """
        rows = self.read_rows()
        closed_pnls: List[float] = []
        actual_rrs: List[float] = []
        latencies: List[float] = []

        by_entry_mode: Dict[str, List[float]] = {}
        by_timeframe: Dict[str, List[float]] = {}
        by_score_band: Dict[str, List[float]] = {}
        by_lot_tier: Dict[str, List[float]] = {}

        missed_count = 0
        blocked_count = 0
        rejected_count = 0
        executed_count = 0

        def _fval(v: Any) -> Optional[float]:
            try:
                f = float(str(v).strip())
                return f if math.isfinite(f) else None
            except (TypeError, ValueError):
                return None

        for r in rows:
            ev = str(r.get("event_type") or "").upper()
            if ev == "MISSED_ENTRY":
                missed_count += 1
            elif ev in ("BLOCKED", "BLOCKED_ENTRY"):
                blocked_count += 1
            elif ev == "REJECTED":
                rejected_count += 1
            elif ev in ("POSITION_OPENED", "ORDER_PLACED", "PARTIAL_FILL", "EXECUTED_RECONCILED"):
                executed_count += 1
                rr_v = _fval(r.get("actual_rr"))
                if rr_v is not None and rr_v > 0:
                    actual_rrs.append(rr_v)

            lat_v = _fval(r.get("execution_latency_ms"))
            if lat_v is not None and lat_v > 0:
                latencies.append(lat_v)

            if ev == "POSITION_CLOSED":
                pnl = _fval(r.get("realized_pnl"))
                if pnl is None:
                    continue
                closed_pnls.append(pnl)

                em = str(r.get("entry_mode") or "UNKNOWN")
                by_entry_mode.setdefault(em, []).append(pnl)

                tf = str(r.get("zone_timeframe") or "UNKNOWN")
                by_timeframe.setdefault(tf, []).append(pnl)

                sc_v = _fval(r.get("zone_score")) or 0.0
                s_band = "90-100" if sc_v >= 90 else ("80-89" if sc_v >= 80 else ("70-79" if sc_v >= 70 else "<70"))
                by_score_band.setdefault(s_band, []).append(pnl)

                lot_s = str(r.get("actual_lot") or r.get("final_lot") or "0.01")
                by_lot_tier.setdefault(lot_s, []).append(pnl)

        total_closed = len(closed_pnls)
        wins = [x for x in closed_pnls if x > 0]
        losses = [x for x in closed_pnls if x < 0]
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        net_profit = sum(closed_pnls)

        win_rate = (len(wins) / total_closed * 100.0) if total_closed > 0 else 0.0
        expectancy = (net_profit / total_closed) if total_closed > 0 else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)

        # Maximum Drawdown dari kurva kumulatif P/L tertutup
        peak = 0.0
        cum = 0.0
        max_dd = 0.0
        for p in closed_pnls:
            cum += p
            if cum > peak:
                peak = cum
            dd = peak - cum
            if dd > max_dd:
                max_dd = dd

        avg_rr = (sum(actual_rrs) / len(actual_rrs)) if actual_rrs else 0.0
        avg_latency = (sum(latencies) / len(latencies)) if latencies else 0.0
        total_attempts = executed_count + rejected_count
        rejection_rate = (rejected_count / total_attempts * 100.0) if total_attempts > 0 else 0.0

        def _summarize_group(grp: Dict[str, List[float]]) -> Dict[str, Dict[str, Any]]:
            out_g = {}
            for k, vals in sorted(grp.items()):
                n_g = len(vals)
                w_g = sum(1 for v in vals if v > 0)
                out_g[k] = {
                    "trades": n_g,
                    "win_rate_pct": round(w_g / n_g * 100.0, 1) if n_g else 0.0,
                    "net_pnl": round(sum(vals), 2),
                    "expectancy": round(sum(vals) / n_g, 2) if n_g else 0.0,
                }
            return out_g

        return {
            "total_events": len(rows),
            "executed_count": executed_count,
            "closed_trades": total_closed,
            "wins": len(wins),
            "losses": len(losses),
            "win_rate_pct": round(win_rate, 2),
            "net_pnl": round(net_profit, 2),
            "gross_profit": round(gross_profit, 2),
            "gross_loss": round(gross_loss, 2),
            "profit_factor": round(profit_factor, 2) if math.isfinite(profit_factor) else "INF",
            "expectancy": round(expectancy, 2),
            "max_drawdown": round(max_dd, 2),
            "average_rr": round(avg_rr, 2),
            "missed_entries": missed_count,
            "blocked_entries": blocked_count,
            "rejected_orders": rejected_count,
            "rejection_rate_pct": round(rejection_rate, 2),
            "avg_latency_ms": round(avg_latency, 2),
            "by_entry_mode": _summarize_group(by_entry_mode),
            "by_zone_timeframe": _summarize_group(by_timeframe),
            "by_zone_score": _summarize_group(by_score_band),
            "by_lot_classification": _summarize_group(by_lot_tier),
        }


_GLOBAL_EXEC_JOURNAL: Optional[ExecutionJournal] = None


def get_execution_journal(journal_path: Optional[str] = None) -> ExecutionJournal:
    global _GLOBAL_EXEC_JOURNAL
    if journal_path:
        return ExecutionJournal(journal_path=journal_path)
    if _GLOBAL_EXEC_JOURNAL is None:
        _GLOBAL_EXEC_JOURNAL = ExecutionJournal()
    return _GLOBAL_EXEC_JOURNAL
