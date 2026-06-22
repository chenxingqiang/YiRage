# Verifying YiRage on a Remote NVIDIA GPU Machine

This document describes the validation workflow encoded in
[`scripts/verify_gpu.sh`](./verify_gpu.sh). The harness automates steps **1, 3,
4, 5, 6, 7** of the plan; step **2 (sync code)** is performed by the operator
before the script is run.

> **Security note.** Do **not** pass an SSH password on the command line or
> embed it in scripts. Use SSH keys (`ssh-copy-id`) or, at minimum, store
> credentials outside source control. If a password has been pasted into a
> chat, ticket or commit, rotate it immediately on the target host.

## 1. Environment check

The script captures `nvidia-smi`, `nvcc --version`, Python / CMake / Ninja
versions, and the `torch.cuda` view of the GPUs into `01_env.log`.

## 2. Sync the code (operator action, not automated)

Recommended (uses SSH keys, no plaintext password). Drop the `-p` flag if your
host listens on the default SSH port 22; the example below uses a non-default
port (32222) only because that was the operator's environment:

```bash
# from your workstation
ssh-copy-id [-p 32222] user@<remote-host>          # one-time
rsync -avz -e "ssh [-p 32222]" \
      --exclude '.git/' --exclude 'build/' \
      ./ user@<remote-host>:/path/to/YiRage/
```

Or, if the host has internet access, just clone:

```bash
git clone https://github.com/chenxingqiang/YiRage.git
```

## 3. Build / install

The harness runs (override with `--build-cmd=...` if you maintain a custom
build flow):

```bash
python3 -m pip install -e . --no-build-isolation
python3 -c "import yirage; print(yirage.__version__)"
```

This is equivalent to `make install` (see [`Makefile`](../Makefile)).

## 4. CPU-only smoke tests (no GPU required)

```bash
python tools/generate_pypi_readme.py --check
PYTHONPATH=tests/python pytest -q \
    tests/python/test_hardware_registry.py \
    tests/python/test_backends.py \
    tests/python/test_compiler.py \
    tests/python/test_global_config.py \
    tests/python/test_utils_common.py
pytest -q tests/python/test_rl
```

These tests bypass `yirage.__init__` via `importlib` and so run even when the
native `yirage.core` extension is unavailable.

## 5. GPU end-to-end demo

Runs an inline reproduction of [`demo/demo_rms_norm.py`](../demo/demo_rms_norm.py)
that pins `CUDA_VISIBLE_DEVICES` to the GPU you select with `--gpu-id`. The
demo builds a `rms_norm + matmul` kernel graph, calls `superoptimize`, then
executes the generated kernel a few times to confirm the JIT path is healthy.
A short `nvidia-smi dmon -c 5` sample is captured alongside the demo log.

## 6. Persistent kernel / MoE

```bash
pytest -q tests/python/test_moe_cpu.py
pytest -q tests/python/test_persistent_kernel.py
```

These cover the persistent-kernel runtime as well as the CPU MoE kernels
(`cpu_moe_linear` / `cpu_moe_silu_linear` defined in
`src/persistent_kernel/pk_cpu_kernels.cc`).

## 7. Result collection

Every step writes its log under the output directory (default
`./verify_gpu_results/`):

| File                           | Source                            |
|--------------------------------|-----------------------------------|
| `01_env.log`                   | step 1 environment dump           |
| `03_build.log`                 | step 3 install + import           |
| `04_smoke.log`                 | step 4 pytest output              |
| `05_demo.log`                  | step 5 GPU demo output            |
| `06_pk.log`                    | step 6 persistent kernel / MoE    |
| `07_nvidia_smi_dmon.log`       | 5-sample GPU utilisation snapshot |
| `SUMMARY.txt`                  | pass/fail table for every step    |

The harness exits non-zero if any executed step failed.

## Running the harness

```bash
# default: run every step, output to ./verify_gpu_results
scripts/verify_gpu.sh

# choose GPU 3 and a custom output directory
scripts/verify_gpu.sh --gpu-id=3 --output-dir=/tmp/yirage_verify

# only re-run smoke + demo (assume already built)
scripts/verify_gpu.sh --skip-env --skip-build --skip-pk

# inspect available options
scripts/verify_gpu.sh --help
```
