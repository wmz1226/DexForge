"""Disposable runtime caches live outside the source tree."""
import os
from pathlib import Path

_user_cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
CACHE_ROOT = Path(os.environ.get("DEXFORGE_CACHE_DIR", _user_cache / "dexforge")) / "contactaware"
