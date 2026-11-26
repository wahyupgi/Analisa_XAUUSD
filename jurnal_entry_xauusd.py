#!/usr/bin/env python
"""Generate a single Excel file with trade detail and summary (daily, weekly, monthly).

File output: c:\Users\ADMIN\Documents\Analisa_XAUUSD\jurnal_entry_xauusd.xlsx

- Sheet "Detail": semua baris yang merupakan trade dieksekusi (ORDER_PLACED atau POSITION_CLOSED).
- Sheet "Daily", "Weekly", "Monthly": ringkasan jumlah trade, net P&L, rata‑rata RR, win rate.

Pastikan auto‑trading dimatikan (auto off) sebelum menjalankan agar tidak terjadi
PermissionError pada file jurnal.
"""
import os, sys

# Tambahkan root proyek ke PYTHONPATH
root = os.path.abspath(os.path.dirname(__file__))
if root not in sys.path:
    sys.path.insert(0, root)

from modules.execution_journal import ExecutionJournal

def _make_summary(df, period):
    """Return DataFrame summary for given period (daily/weekly/monthly)."""
    if period == "daily":
        grp = df.groupby(df["timestamp"].dt.date)
    elif period == "weekly":
        grp = df.groupby(df["timestamp"].dt.isocalendar().week)
    else:  # monthly
        grp = df.groupby([df["timestamp"].dt.year, df["timestamp"].dt.month])

    rows = []
    for key, sub in grp:
        trades = len(sub)
        net_pnl = sub["realized_pnl"].sum()
        avg_rr = sub["actual_rr"].mean()
        win_rate = (sub["actual_rr"] > 0).mean() * 100
        if period == "daily":
            period_label = str(key)
        elif period == "weekly":
            # key is week number, need year for proper label
            year = sub["timestamp"].dt.isocalendar().year.iloc[0]
            period_label = f"{year}-W{int(key):02d}"
        else:  # monthly
            year, month = key
            period_label = f"{year}-{int(month):02d}"
        rows.append({
            "period": period_label,
            "trades": trades,
            "net_pnl": round(net_pnl, 2),
            "avg_rr": round(avg_rr, 2),
            "win_rate_pct": round(win_rate, 1),
        })
    return rows

def main():
    ej = ExecutionJournal()
    all_rows = ej.read_rows()
    # Filter hanya baris eksekusi
    exec_rows = [r for r in all_rows if r.get("event_type") in ("ORDER_PLACED", "POSITION_CLOSED")]
    if not exec_rows:
        print("[INFO] Tidak ada trade yang dieksekusi.")
        return

    # Gunakan pandas jika tersedia
    try:
        import pandas as pd
        from dateutil import parser as dt_parser
        # Buat DataFrame
        df = pd.DataFrame(exec_rows)
        # Konversi kolom numerik
        numeric_cols = ["realized_pnl", "actual_rr"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        # Parse timestamp
        df["timestamp"] = df["timestamp"].apply(dt_parser.parse)
        # Sheet Detail
        detail_path = os.path.join(root, "jurnal_entry_xauusd.xlsx")
        with pd.ExcelWriter(detail_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Detail", index=False)
            # Summary sheets
            for period in ("daily", "weekly", "monthly"):
                summary_rows = _make_summary(df, period)
                summary_df = pd.DataFrame(summary_rows)
                sheet_name = period.capitalize()
                summary_df.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"[INFO] Excel selesai: {detail_path}")
    except Exception as exc:
        # Fallback ke openpyxl tanpa pandas (lebih simpel, tapi akan tetap menulis detail hanya)
        print(f"[WARNING] Pandas tidak tersedia atau error: {exc}. Menggunakan fallback sederhana.")
        try:
            from openpyxl import Workbook
            wb = Workbook()
            ws = wb.active
            ws.title = "Detail"
            # Header
            header = list(exec_rows[0].keys())
            ws.append(header)
            for r in exec_rows:
                ws.append([r.get(col, "") for col in header])
            out_path = os.path.join(root, "jurnal_entry_xauusd.xlsx")
            wb.save(out_path)
            print(f"[INFO] Excel (fallback) selesai: {out_path}")
        except Exception as e2:
            print(f"[ERROR] Gagal menulis Excel: {e2}")

if __name__ == "__main__":
    main()
