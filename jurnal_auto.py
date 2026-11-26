#!/usr/bin/env python
import os, sys

# Pastikan root proyek ada di sys.path
root = os.path.abspath(os.path.dirname(__file__))
if root not in sys.path:
    sys.path.insert(0, root)

from modules.execution_journal import ExecutionJournal

def main() -> None:
    ej = ExecutionJournal()
    # Hasilkan ringkasan harian, mingguan, bulanan
    periods = [
        ("daily", "jurnal_harian_summary.csv"),
        ("weekly", "jurnal_mingguan_summary.csv"),
        ("monthly", "jurnal_bulanan_summary.csv"),
    ]
    for period, filename in periods:
        out_path = os.path.join(root, filename)
        ej.export_summary(period=period, output_path=out_path)
        print(f"[INFO] Ringkasan {period} disimpan ke: {out_path}")

    print("\nSemua ringkasan selesai. Buka file CSV di Excel untuk melihat hasil yang rapi.")

if __name__ == "__main__":
    main()
