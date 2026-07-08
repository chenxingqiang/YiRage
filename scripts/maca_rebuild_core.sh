#!/usr/bin/env bash
# Rebuild yirage.core on MetaX VM so C++ maca::MAX_SMEM_SIZE (64 KB) is baked in.
# Closes runtime plan_stensor_memory warnings that still cite 98304 (Volta) on stale builds.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export MACA_PATH="${MACA_PATH:-/opt/maca}"
export YIRAGE_BACKEND=maca
export LD_LIBRARY_PATH="${ROOT}/build/abstract_subexpr/release:${ROOT}/build/formal_verifier/release:${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${LD_LIBRARY_PATH:-}"

echo "[maca_rebuild_core] MACA_PATH=${MACA_PATH}"
echo "[maca_rebuild_core] Rebuilding editable yirage.core (YIRAGE_BACKEND=maca)..."

python3 -m pip install -e . --no-build-isolation

python3 - <<'PY'
import os

os.environ.setdefault("YIRAGE_BACKEND", "maca")
from yirage.utils.common import get_shared_memory_capacity

cap = get_shared_memory_capacity(70)
assert cap == 65536, f"expected 65536, got {cap}"
print(f"[maca_rebuild_core] get_shared_memory_capacity(70)={cap} OK")
PY

echo "[maca_rebuild_core] PASS — rerun demo/maca_superopt_test.py to confirm smem warnings gone"
