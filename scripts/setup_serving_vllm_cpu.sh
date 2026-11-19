#!/usr/bin/env bash
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
# Install vLLM CPU wheel + transformers for Serving Loop fork e2e.
set -euo pipefail

VLLM_VERSION="${VLLM_VERSION:-0.26.0}"
WHEEL_URL="https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cpu-cp38-abi3-manylinux_2_34_x86_64.whl"

echo "Installing vLLM CPU wheel ${VLLM_VERSION} ..."
# ubuntu-22.04's /usr/bin/pip is 22.0.2 and rejects --break-system-packages.
# Drive the interpreter's pip, and upgrade it first when the flag is missing.
python3 -m pip install --upgrade "pip>=23.1" --break-system-packages \
  || python3 -m pip install --upgrade "pip>=23.1"
python3 -m pip install --break-system-packages --force-reinstall \
  "${WHEEL_URL}" \
  --extra-index-url https://download.pytorch.org/whl/cpu

python3 -m pip install --break-system-packages transformers

python3 - <<'PY'
import vllm
from vllm.platforms import current_platform
print("vllm", vllm.__version__, "platform", current_platform.device_type)
assert current_platform.device_type == "cpu", "expected CpuPlatform on headless CI"
PY

echo "vLLM CPU setup OK"
