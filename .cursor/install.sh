#!/usr/bin/env bash
# Cloud Agent install for YiRage (CPU backend).
#
# Builds the native runtime (yirage.core / libyirage_runtime) for the CPU
# backend and installs the real-torch / real-vLLM stack used by the
# RuntimeFusion serving verification gate. Idempotent: safe to re-run.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# The base image marks the system Python as PEP 668 externally-managed once
# python3-dev is present; installs target the user site via this override.
PIP=(python3 -m pip install --break-system-packages)

echo "[1/6] System packages (Python headers + protobuf compiler)"
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  python3-dev protobuf-compiler

echo "[2/6] Rust toolchain (edition2024 support for vendored search crates)"
# The vendored search/verifier crates resolve dependencies that require the
# cargo edition2024 feature (stable >= 1.85).
rustup update stable
rustup default stable

echo "[3/6] Header-only C++ deps (cutlass, nlohmann/json)"
# .gitmodules references these but they are not committed as gitlinks in this
# tree, so fetch them directly. deps/ (except llvm-project) is not gitignored;
# these clones are build inputs only and must not be committed.
if [ ! -f deps/cutlass/include/cutlass/cutlass.h ]; then
  rm -rf deps/cutlass
  git clone --depth 1 https://github.com/NVIDIA/cutlass.git deps/cutlass
fi
if [ ! -f deps/json/CMakeLists.txt ]; then
  rm -rf deps/json
  git clone --depth 1 https://github.com/nlohmann/json.git deps/json
fi

echo "[4/6] Python dependencies (search / RL / profiling)"
"${PIP[@]}" \
  "cmake>=3.24" "cython>=0.28" "accelforge>=1.0.355" "z3-solver>=4.12.0" \
  numpy graphviz tqdm "protobuf>=3.20.3" "ray>=2.55.0" pytest tg4perfetto

echo "[5/6] vLLM CPU + transformers (serving RF real-engine gate)"
# Provided by the repo; installs the vLLM CPU wheel plus a compatible
# torch/transformers used by the RuntimeFusion serving cert.
bash scripts/setup_serving_vllm_cpu.sh

echo "[6/6] Build native CPU runtime (yirage.core, editable)"
YIRAGE_BACKEND=cpu CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-4}" \
  "${PIP[@]}" -e . --no-build-isolation

echo "Verifying import ..."
python3 -c "import yirage as yr; print('yirage', yr.__version__, yr.get_available_backends())"
echo "YiRage CPU environment ready."
