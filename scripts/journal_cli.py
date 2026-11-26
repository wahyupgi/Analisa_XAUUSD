import argparse, os, sys

# Ensure project root is in sys.path so we can import the module
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root not in sys.path:
    sys.path.insert(0, root)

from modules.execution_journal import ExecutionJournal

def create_new_journal(path: str | None = None) -> None:
    """Buat file jurnal baru (atau timpa jika sudah ada) dengan header standar.
    Jika *path* None, gunakan path default yang didefinisikan di config.
    """
    ej = ExecutionJournal(journal_path=path) if path else ExecutionJournal()
    # Memanggil method privat untuk memastikan header ditulis.
    # _ensure_header_locked bersifat internal, tapi kami panggil via name‑mangling.
    ej._ensure_header_locked()
    print(f"[INFO] Jurnal baru dibuat: {ej.journal_path}")

def compact_journal(output: str | None = None) -> None:
    """Tulis ulang jurnal sehingga hanya menyimpan baris eksekusi.
    Jika *output* diberikan, file hasil akan disimpan ke path tersebut.
    """
    ej = ExecutionJournal()
    ej.compact_executed_journal(output_path=output)
    target = output or ej.journal_path
    print(f"[INFO] Jurnal terkompak disimpan ke: {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description='Utility CLI untuk jurnal eksekusi auto‑trading.')
    subparsers = parser.add_subparsers(dest='command', required=True)

    # subcommand: new
    parser_new = subparsers.add_parser('new', help='Buat jurnal baru (reset)')
    parser_new.add_argument('--path', type=str, help='Path file jurnal baru (default: config path)')

    # subcommand: compact
    parser_compact = subparsers.add_parser('compact', help='Kompak jurnal, hanya simpan trade yang dieksekusi')
    parser_compact.add_argument('--output', type=str, help='File tujuan (default: timpa file asli)')

    args = parser.parse_args()
    if args.command == 'new':
        create_new_journal(path=args.path)
    elif args.command == 'compact':
        compact_journal(output=args.output)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
