#!/usr/bin/env bash
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
# One-shot setup for Serving Loop yirage.core tier on Cloud CPU VM.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-python3}"
CARGO="${CARGO:-/usr/local/cargo/bin/cargo}"

echo "[setup] YiRage serving yirage.core tier"

if [ ! -d deps/json/include/nlohmann ]; then
  echo "[setup] cloning nlohmann/json..."
  mkdir -p deps/json
  git clone --depth 1 https://github.com/nlohmann/json.git deps/json/_src
  mkdir -p deps/json/include
  cp -a deps/json/_src/include/nlohmann deps/json/include/
fi

if [ ! -d deps/cutlass/include/cutlass ]; then
  echo "[setup] cloning NVIDIA/cutlass..."
  git clone --depth 1 https://github.com/NVIDIA/cutlass.git deps/cutlass/_src
  mkdir -p deps/cutlass/include
  cp -a deps/cutlass/_src/include/cutlass deps/cutlass/include/
  cp -a deps/cutlass/_src/include/cute deps/cutlass/include/ 2>/dev/null || true
fi

echo "[setup] building Rust search helpers..."
"$CARGO" +stable build --release --manifest-path src/search/abstract_expr/abstract_subexpr/Cargo.toml \
  --target-dir "$ROOT/build/abstract_subexpr"
"$CARGO" +stable build --release --manifest-path src/search/verification/formal_verifier_equiv/Cargo.toml \
  --target-dir "$ROOT/build/formal_verifier"

export LD_LIBRARY_PATH="$ROOT/build/abstract_subexpr/release:$ROOT/build/formal_verifier/release:${LD_LIBRARY_PATH:-}"
export YIRAGE_BACKEND=cpu
export PYTHONPATH="$ROOT"

echo "[setup] pip install -e . (CPU backend)..."
"$PY" -m pip install -e . --no-build-isolation --break-system-packages

echo "[setup] verify import..."
"$PY" -c "import yirage as yr; print('yirage', yr.__version__, yr.get_available_backends())"

echo "[setup] serving yirage.core pytest smoke..."
"$PY" -m pytest tests/python/test_runtime_fusion_yirage_core.py -q --tb=short

echo "[setup] done"
