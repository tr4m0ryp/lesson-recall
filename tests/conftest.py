"""Point the store and the session root at throwaway directories before the package is imported."""
import os
import sys
import tempfile
from pathlib import Path

os.environ["RECALL_HOME"] = tempfile.mkdtemp(prefix="recall-test-")
os.environ["CLAUDE_CONFIG_DIR"] = tempfile.mkdtemp(prefix="recall-test-claude-")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
