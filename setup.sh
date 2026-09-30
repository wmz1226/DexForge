#!/usr/bin/env bash
# One Python 3.10 environment for both stages and the simulator.
set -euo pipefail
DEXFORGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEXFORGE_VENV="${DEXFORGE_VENV:-$DEXFORGE_ROOT/.venv}"
DEXFORGE_BOOTSTRAP="${DEXFORGE_PYTHON:-python3.10}"
"$DEXFORGE_BOOTSTRAP" -c 'import sys; assert sys.version_info[:2] == (3, 10), "Python 3.10 is required"'
if [[ ! -x "$DEXFORGE_VENV/bin/python" ]]; then
    "$DEXFORGE_BOOTSTRAP" -m venv "$DEXFORGE_VENV"
fi
DEXFORGE_RUNTIME="$DEXFORGE_VENV/bin/python"
"$DEXFORGE_RUNTIME" -c 'import sys; assert sys.version_info[:2] == (3, 10), "Existing environment must use Python 3.10"'
"$DEXFORGE_RUNTIME" -m pip install 'pip<26' 'setuptools<81' wheel numpy==1.26.4 scipy==1.13.1 six
# Chumpy's legacy build imports its dependencies before declaring them.
"$DEXFORGE_RUNTIME" -m pip install --no-build-isolation chumpy==0.70
"$DEXFORGE_RUNTIME" -m pip install -r "$DEXFORGE_ROOT/requirements.txt"
"$DEXFORGE_RUNTIME" - "$DEXFORGE_ROOT" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
if not (root / 'contactaware/third_party/manopth/manopth/manolayer.py').is_file():
    raise FileNotFoundError('Bundled manopth is missing; restore contactaware/third_party/manopth')
print('Setup complete. See assets/mano/README.md for model files.')
PY
