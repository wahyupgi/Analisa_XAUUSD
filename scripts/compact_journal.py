import sys, os
# Ensure project root is in path
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root not in sys.path:
    sys.path.insert(0, root)
from modules.execution_journal import ExecutionJournal

if __name__ == "__main__":
    ej = ExecutionJournal()
    ej.compact_executed_journal()
    print("Journal compacted successfully.")
