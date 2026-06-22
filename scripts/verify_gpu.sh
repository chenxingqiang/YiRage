#!/usr/bin/env bash
# =============================================================================
# YiRage GPU Verification Harness
# =============================================================================
#
# Automates the 7-step verification plan for validating YiRage on a remote
# NVIDIA GPU machine. Designed to be copied (or ``rsync``ed) onto the target
# host and executed there.
#
# Steps performed (each can be skipped with a flag):
#   1. Environment check          (--skip-env)
#   2. (Skipped here -- code-sync is done by the operator before running)
#   3. Build / install             (--skip-build)
#   4. CPU-only smoke tests        (--skip-smoke)
#   5. GPU end-to-end demo         (--skip-demo)
#   6. Persistent kernel / MoE     (--skip-pk)
#   7. Result collection           (always; output dir is --output-dir)
#
# Usage:
#   scripts/verify_gpu.sh [options]
#
# Options:
#   --output-dir=<path>    Where to write logs (default: ./verify_gpu_results)
#   --gpu-id=<int>         CUDA device id used by demo (default: 0)
#   --skip-env             Skip step 1
#   --skip-build           Skip step 3
#   --skip-smoke           Skip step 4
#   --skip-demo            Skip step 5
#   --skip-pk              Skip step 6
#   --build-cmd=<cmd>      Override the install command for step 3
#                          (default: ``python3 -m pip install -e . --no-build-isolation``)
#   -h, --help             Show this help
#
# Exit code is 0 only when every executed step succeeded.
# =============================================================================

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

OUTPUT_DIR="${PROJECT_ROOT}/verify_gpu_results"
GPU_ID="0"
SKIP_ENV=false
SKIP_BUILD=false
SKIP_SMOKE=false
SKIP_DEMO=false
SKIP_PK=false
BUILD_CMD="python3 -m pip install -e . --no-build-isolation"

# ---- arg parsing ------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir=*) OUTPUT_DIR="${1#*=}" ;;
        --gpu-id=*)     GPU_ID="${1#*=}" ;;
        --build-cmd=*)  BUILD_CMD="${1#*=}" ;;
        --skip-env)     SKIP_ENV=true ;;
        --skip-build)   SKIP_BUILD=true ;;
        --skip-smoke)   SKIP_SMOKE=true ;;
        --skip-demo)    SKIP_DEMO=true ;;
        --skip-pk)      SKIP_PK=true ;;
        -h|--help)
            awk '/^#/ {print; next} {exit}' "${BASH_SOURCE[0]}"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 2
            ;;
    esac
    shift
done

mkdir -p "${OUTPUT_DIR}"
SUMMARY="${OUTPUT_DIR}/SUMMARY.txt"
: > "${SUMMARY}"

# ---- helpers ----------------------------------------------------------------
declare -a STEP_NAMES=()
declare -a STEP_STATUS=()

log()    { printf '\n[%s] %s\n' "$(date -u +%H:%M:%SZ)" "$*"; }
banner() { printf '\n============================================================\n  %s\n============================================================\n' "$*"; }

# run_step <name> <log-file-relative> <command...>
run_step() {
    local name="$1"; shift
    local logfile="${OUTPUT_DIR}/$1"; shift
    banner "${name}"
    echo "Log: ${logfile}"
    local rc=0
    if "$@" > "${logfile}" 2>&1; then
        STEP_STATUS+=("PASS")
        echo "[OK]   ${name}"
    else
        rc=$?
        STEP_STATUS+=("FAIL(rc=${rc})")
        echo "[FAIL] ${name} (rc=${rc}) -- see ${logfile}"
    fi
    STEP_NAMES+=("${name}")
    return 0   # never abort the harness; final summary decides exit code
}

# ---- step 1: environment check ---------------------------------------------
step_env() {
    {
        echo "## host"
        uname -a
        echo
        echo "## nvidia-smi"
        if command -v nvidia-smi >/dev/null 2>&1; then
            nvidia-smi || true
        else
            echo "nvidia-smi NOT FOUND"
        fi
        echo
        echo "## nvcc"
        if command -v nvcc >/dev/null 2>&1; then
            nvcc --version
        else
            echo "nvcc NOT FOUND"
        fi
        echo
        echo "## python / build tools"
        python3 --version 2>&1 || echo "python3 NOT FOUND"
        cmake --version 2>&1 | head -1 || echo "cmake NOT FOUND"
        ninja --version 2>&1 || echo "ninja NOT FOUND"
        echo
        echo "## torch"
        python3 - <<'PY' 2>&1 || true
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device count:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f"  [{i}] {torch.cuda.get_device_name(i)}")
except Exception as e:
    print("torch import failed:", e)
PY
    }
}

# ---- step 3: build / install -----------------------------------------------
step_build() {
    cd "${PROJECT_ROOT}" || return 1
    echo "## install command: ${BUILD_CMD}"
    # shellcheck disable=SC2086
    eval ${BUILD_CMD}
    local rc=$?
    if [[ ${rc} -ne 0 ]]; then return "${rc}"; fi
    echo
    echo "## import smoke"
    python3 -c "import yirage; print('yirage version:', getattr(yirage, '__version__', '?'))"
}

# ---- step 4: CPU-only smoke tests ------------------------------------------
step_smoke() {
    cd "${PROJECT_ROOT}" || return 1
    local fail=0

    echo "### tools/generate_pypi_readme.py --check"
    python3 tools/generate_pypi_readme.py --check || fail=$?

    echo
    echo "### focused pytest (no-GPU modules)"
    PYTHONPATH="tests/python:${PYTHONPATH:-}" python3 -m pytest -q \
        tests/python/test_hardware_registry.py \
        tests/python/test_backends.py \
        tests/python/test_compiler.py \
        tests/python/test_global_config.py \
        tests/python/test_utils_common.py \
        || fail=$?

    echo
    echo "### tests/python/test_rl"
    python3 -m pytest -q tests/python/test_rl || fail=$?

    return "${fail}"
}

# ---- step 5: GPU end-to-end demo -------------------------------------------
step_demo() {
    cd "${PROJECT_ROOT}" || return 1

    # Confirm CUDA is actually visible from python before invoking demos.
    python3 - <<'PY'
import sys, torch
if not torch.cuda.is_available():
    print("CUDA is not available to torch -- aborting demo step.")
    sys.exit(2)
print("CUDA OK, device 0 =", torch.cuda.get_device_name(0))
PY

    # Pick the smallest, fastest demo: rms_norm. Patch the hard-coded GPU id
    # (cuda:7 in the source) onto the user-selected GPU via env-var override.
    echo "### demo_rms_norm.py on cuda:${GPU_ID}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" YIRAGE_DEMO_GPU_ID=0 python3 - <<'PY'
import yirage as yr
import torch

yr.set_gpu_device_id(0)
dt = yr.bfloat16
tdt = yr.convert_dtype_to_torch_type(dt)

graph = yr.new_kernel_graph()
X = graph.new_input(dims=(1, 7168), dtype=dt)
W = graph.new_input(dims=(7168, 16384), dtype=dt)
D = graph.rms_norm(X, normalized_shape=(7168,))
O = graph.matmul(D, W)
graph.mark_output(O)

opt = graph.superoptimize(config="mlp")

inputs = [
    torch.randn(1, 1, 7168, dtype=tdt, device="cuda:0"),
    torch.randn(7168, 16384, dtype=tdt, device="cuda:0"),
]
out = opt(inputs=inputs)
print("output shape:", out[0].shape, "dtype:", out[0].dtype, "device:", out[0].device)
for _ in range(8):
    opt(inputs=inputs)
torch.cuda.synchronize()
print("demo_rms_norm: OK")
PY
}

# ---- step 6: persistent kernel / MoE ---------------------------------------
step_pk() {
    cd "${PROJECT_ROOT}" || return 1
    local fail=0
    echo "### tests/python/test_moe_cpu.py"
    python3 -m pytest -q tests/python/test_moe_cpu.py || fail=$?
    echo
    echo "### tests/python/test_persistent_kernel.py"
    python3 -m pytest -q tests/python/test_persistent_kernel.py || fail=$?
    return "${fail}"
}

# ---- main -------------------------------------------------------------------
banner "YiRage GPU Verification"
echo "Project root: ${PROJECT_ROOT}"
echo "Output dir  : ${OUTPUT_DIR}"
echo "GPU id      : ${GPU_ID}"

[[ "${SKIP_ENV}"   == false ]] && run_step "1. environment check"        "01_env.log"   step_env
[[ "${SKIP_BUILD}" == false ]] && run_step "3. build / install"          "03_build.log" step_build
[[ "${SKIP_SMOKE}" == false ]] && run_step "4. CPU smoke tests"          "04_smoke.log" step_smoke
[[ "${SKIP_DEMO}"  == false ]] && run_step "5. GPU end-to-end demo"      "05_demo.log"  step_demo
[[ "${SKIP_PK}"    == false ]] && run_step "6. persistent kernel / MoE" "06_pk.log"    step_pk

# Optional: capture a brief GPU utilisation sample for the report.
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi dmon -c 5 > "${OUTPUT_DIR}/07_nvidia_smi_dmon.log" 2>&1 || true
fi

# ---- summary ----------------------------------------------------------------
banner "SUMMARY"
overall=0
for status in "${STEP_STATUS[@]}"; do
    [[ "${status}" == PASS ]] || overall=1
done

{
    printf '%-40s %s\n' "STEP" "RESULT"
    printf '%-40s %s\n' "----" "------"
    for i in "${!STEP_NAMES[@]}"; do
        printf '%-40s %s\n' "${STEP_NAMES[$i]}" "${STEP_STATUS[$i]}"
    done
    echo
    echo "Logs: ${OUTPUT_DIR}"
} | tee "${SUMMARY}"

exit "${overall}"
