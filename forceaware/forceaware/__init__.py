"""Forceaware application package."""
from pathlib import Path
import sys

# Standalone stage entry points also use the shared source-tree package.
_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
