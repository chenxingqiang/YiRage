# Copyright 2026 YiRage team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# =============================================================================
# YiRage Setup - Multi-Backend Installation
# =============================================================================
#
# Backend Selection via Environment Variables:
#   YIRAGE_BACKEND=cuda          # NVIDIA GPU
#   YIRAGE_BACKEND=rocm          # AMD GPU
#   YIRAGE_BACKEND=mps           # Apple Silicon
#   YIRAGE_BACKEND=ascend        # Huawei NPU
#   YIRAGE_BACKEND=maca          # MetaX GPU
#   YIRAGE_BACKEND=xpu           # Intel GPU
#   YIRAGE_BACKEND=tpu           # Google TPU
#   YIRAGE_BACKEND=cpu           # CPU only
#   YIRAGE_BACKEND=auto          # Auto-detect (default)
#
# Multiple backends (comma-separated):
#   YIRAGE_BACKEND=cuda,cpu
#
# Individual backend flags (override YIRAGE_BACKEND):
#   USE_CUDA=ON/OFF
#   USE_ROCM=ON/OFF
#   USE_MPS=ON/OFF
#   USE_ASCEND=ON/OFF
#   USE_MACA=ON/OFF
#   USE_XPU=ON/OFF
#   USE_TPU=ON/OFF
#   USE_CPU=ON/OFF
#
# pip install examples:
#   pip install .                                    # Auto-detect
#   YIRAGE_BACKEND=cuda pip install .               # CUDA only
#   YIRAGE_BACKEND=mps pip install .                # MPS only
#   USE_CUDA=ON USE_CPU=ON pip install .            # CUDA + CPU
#   YIRAGE_BACKEND=cpu USE_MLIR=1 pip install -e .  # CPU + MLIR
# Without deps/llvm-project: uses system LLVM if found, or set MLIR_DIR or YIRAGE_LLVM_SOURCE=fetch|prebuilt
#   pip install . --config-settings=cmake.args=-DUSE_CUDA=ON
#
# =============================================================================
import importlib.util
import os
import shutil
from os import path
from pathlib import Path
import sys
import sysconfig
from setuptools import find_packages, setup, Command
from contextlib import contextmanager

sys.path.insert(0, str(Path(__file__).parent / "tools"))
from setup_backend_config import (
    cmake_mlir_extra_definitions,
    cython_mlir_extra_link_args,
    env_to_cmake_onoff as _env_to_cmake_onoff,
    merge_extra_use_flags_from_env as _merge_extra_use_flags_from_env,
    should_regenerate_config_cmake as _should_regenerate_config_cmake,
)
import subprocess
import re
import platform

# need to use distutils.core for correct placement of cython dll
if "--inplace" in sys.argv:
    from distutils.core import setup
    from distutils.extension import Extension
else:
    from setuptools import setup
    from setuptools.extension import Extension

import z3


def _get_homebrew_omp_lib_dir():
    """Return the Homebrew libomp library directory on macOS, or None."""
    if sys.platform != "darwin":
        return None

    # 1) Ask Homebrew for the actual prefix.
    try:
        brew_prefix = subprocess.check_output(
            ["brew", "--prefix", "libomp"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
        if brew_prefix:
            lib_dir = os.path.join(brew_prefix, "lib")
            if os.path.isdir(lib_dir) and os.path.exists(os.path.join(lib_dir, "libomp.dylib")):
                return lib_dir
    except (FileNotFoundError, subprocess.CalledProcessError, OSError):
        pass

    # 2) Fallback: well-known Homebrew prefixes.
    for prefix in [
        "/opt/homebrew/opt/libomp/lib",   # Apple Silicon
        "/usr/local/opt/libomp/lib",       # Intel
    ]:
        if os.path.isdir(prefix) and os.path.exists(os.path.join(prefix, "libomp.dylib")):
            return prefix

    return None


# =============================================================================
# Hardware Detection and Environment Setup
# =============================================================================

def auto_setup_environment():
    """
    Auto-detect hardware and set required environment variables.
    This is called automatically during pip install.
    """
    env_changes = {}

    # =========================================================================
    # CUDA Environment
    # =========================================================================
    if not os.environ.get("CUDA_HOME"):
        cuda_paths = [
            "/usr/local/cuda",
            "/opt/cuda",
            "/usr/lib/cuda",
        ]
        # Also check versioned paths
        for base in ["/usr/local", "/opt"]:
            if os.path.exists(base):
                for d in os.listdir(base):
                    if d.startswith("cuda-"):
                        cuda_paths.append(os.path.join(base, d))

        for cuda_path in cuda_paths:
            if os.path.exists(os.path.join(cuda_path, "bin", "nvcc")):
                os.environ["CUDA_HOME"] = cuda_path
                env_changes["CUDA_HOME"] = cuda_path
                break

    # =========================================================================
    # Ascend Environment (Huawei NPU)
    # =========================================================================
    if not os.environ.get("ASCEND_HOME_PATH"):
        ascend_paths = [
            "/usr/local/Ascend/ascend-toolkit/latest",
            "/usr/local/Ascend/nnrt/latest",
            "/opt/Ascend/ascend-toolkit/latest",
        ]
        for ascend_path in ascend_paths:
            if os.path.exists(ascend_path):
                os.environ["ASCEND_HOME_PATH"] = ascend_path
                os.environ["ASCEND_HOME"] = ascend_path
                env_changes["ASCEND_HOME_PATH"] = ascend_path
                env_changes["ASCEND_HOME"] = ascend_path

                # Set OPP path
                opp_path = os.path.join(ascend_path, "opp")
                if os.path.exists(opp_path):
                    os.environ["ASCEND_OPP_PATH"] = opp_path
                    env_changes["ASCEND_OPP_PATH"] = opp_path

                # Add to LD_LIBRARY_PATH
                lib_paths = [
                    os.path.join(ascend_path, "lib64"),
                    os.path.join(ascend_path, "aarch64-linux", "lib64"),
                    "/usr/local/Ascend/driver/lib64",
                    "/usr/local/Ascend/driver/lib64/driver",
                ]
                existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
                new_paths = [p for p in lib_paths if os.path.exists(p) and p not in existing_ld]
                if new_paths:
                    os.environ["LD_LIBRARY_PATH"] = ":".join(new_paths) + ":" + existing_ld
                    env_changes["LD_LIBRARY_PATH"] = os.environ["LD_LIBRARY_PATH"]

                # Add to PATH
                bin_path = os.path.join(ascend_path, "bin")
                if os.path.exists(bin_path):
                    existing_path = os.environ.get("PATH", "")
                    if bin_path not in existing_path:
                        os.environ["PATH"] = bin_path + ":" + existing_path
                        env_changes["PATH"] = os.environ["PATH"]
                break

    # =========================================================================
    # ROCm Environment (AMD GPU)
    # =========================================================================
    if not os.environ.get("ROCM_PATH"):
        rocm_paths = ["/opt/rocm", "/opt/rocm-5.7.0", "/opt/rocm-6.0.0"]
        for rocm_path in rocm_paths:
            if os.path.exists(rocm_path):
                os.environ["ROCM_PATH"] = rocm_path
                os.environ["HIP_PATH"] = rocm_path
                env_changes["ROCM_PATH"] = rocm_path
                env_changes["HIP_PATH"] = rocm_path

                # Add to LD_LIBRARY_PATH
                lib_path = os.path.join(rocm_path, "lib")
                existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
                if os.path.exists(lib_path) and lib_path not in existing_ld:
                    os.environ["LD_LIBRARY_PATH"] = lib_path + ":" + existing_ld
                    env_changes["LD_LIBRARY_PATH"] = os.environ["LD_LIBRARY_PATH"]
                break

    # =========================================================================
    # MACA Environment (MetaX GPU)
    # =========================================================================
    if not os.environ.get("MACA_PATH"):
        maca_paths = ["/opt/maca", "/usr/local/maca"]
        for maca_path in maca_paths:
            if os.path.exists(maca_path):
                os.environ["MACA_PATH"] = maca_path
                env_changes["MACA_PATH"] = maca_path
                break

    # =========================================================================
    # Intel oneAPI (XPU)
    # =========================================================================
    if not os.environ.get("ONEAPI_ROOT"):
        oneapi_paths = ["/opt/intel/oneapi", "/usr/local/intel/oneapi"]
        for oneapi_path in oneapi_paths:
            if os.path.exists(oneapi_path):
                os.environ["ONEAPI_ROOT"] = oneapi_path
                env_changes["ONEAPI_ROOT"] = oneapi_path
                # Source the setvars.sh would be ideal but can't in Python
                break

    # =========================================================================
    # TPU Environment
    # =========================================================================
    if os.path.exists("/usr/share/tpu") and not os.environ.get("TPU_NAME"):
        # On TPU VMs, TPU_NAME is usually set automatically
        pass

    # Print detected environment changes
    if env_changes:
        print("  Auto-configured environment variables:")
        for key, value in env_changes.items():
            if key == "LD_LIBRARY_PATH" or key == "PATH":
                # Truncate long paths
                print(f"    {key}=...{value[-60:] if len(value) > 60 else value}")
            else:
                print(f"    {key}={value}")

    return env_changes


def check_torch_npu_dependency():
    """
    Check torch_npu availability and compatibility for Ascend backend.
    Returns tuple: (is_available, version, torch_version, message)
    """
    torch_npu_available = False
    torch_npu_version = None
    torch_version = None
    message = ""

    # First check if torch is available
    try:
        import torch
        torch_version = torch.__version__.split('+')[0]
    except ImportError:
        return False, None, None, "PyTorch not installed"

    # Check torch_npu
    try:
        import torch_npu
        torch_npu_available = True
        torch_npu_version = torch_npu.__version__

        # Check if NPU is actually available
        if hasattr(torch, 'npu') and torch.npu.is_available():
            device_count = torch.npu.device_count()
            device_name = torch.npu.get_device_name(0) if device_count > 0 else "Unknown"
            message = f"torch_npu {torch_npu_version} ready ({device_count} NPU: {device_name})"
        else:
            message = f"torch_npu {torch_npu_version} installed but no NPU detected"

    except ImportError:
        message = "torch_npu not installed"
    except Exception as e:
        message = f"torch_npu error: {str(e)}"

    return torch_npu_available, torch_npu_version, torch_version, message


def print_torch_npu_install_guide(torch_version=None):
    """Print installation guide for torch_npu."""
    print("\n  torch_npu Installation Guide:")
    print("  " + "=" * 50)

    if torch_version:
        print(f"\n  Your PyTorch version: {torch_version}")
        print(f"  torch_npu version should match PyTorch version.")

    print("""
  Option 1: From Huawei Ascend Repository (Recommended)
    pip install torch-npu -i https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/ascend-repo/simple/

  Option 2: Direct pip install (if available)
    pip install torch-npu

  Option 3: Version-matched install
    pip install torch-npu=={torch_version}

  Option 4: From source (Gitee)
    git clone https://gitee.com/ascend/pytorch.git
    cd pytorch && pip install -e .

  Prerequisites:
    - CANN toolkit installed (check with: npu-smi info)
    - PyTorch installed (matching version)
    - ASCEND_HOME_PATH set correctly

  Verify installation:
    python -c "import torch_npu; print(torch.npu.is_available())"
  """.format(torch_version=torch_version or "X.X.X"))


def detect_hardware():
    """Detect available hardware backends."""
    # First, auto-setup environment
    auto_setup_environment()

    detected = {"cpu": True}  # CPU always available

    # NVIDIA CUDA
    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home and os.path.exists(cuda_home):
        detected["cuda"] = True
        print(f"  Detected: NVIDIA GPU (CUDA_HOME={cuda_home})")
    elif shutil.which("nvidia-smi"):
        try:
            result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
            if result.returncode == 0 and "GPU" in result.stdout:
                detected["cuda"] = True
                print(f"  Detected: NVIDIA GPU")
        except Exception:
            pass

    # AMD ROCm
    rocm_path = os.environ.get("ROCM_PATH")
    if rocm_path and os.path.exists(rocm_path):
        detected["rocm"] = True
        print(f"  Detected: AMD GPU (ROCM_PATH={rocm_path})")
    elif shutil.which("rocm-smi") or os.path.exists("/opt/rocm"):
        detected["rocm"] = True
        print(f"  Detected: AMD GPU (ROCm)")

    # Apple MPS
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        detected["mps"] = True
        print(f"  Detected: Apple Silicon (MPS)")

    # Huawei Ascend
    ascend_home = os.environ.get("ASCEND_HOME_PATH") or os.environ.get("ASCEND_HOME")
    if ascend_home and os.path.exists(ascend_home):
        detected["ascend"] = True
        print(f"  Detected: Huawei Ascend NPU (ASCEND_HOME={ascend_home})")
        # Check torch_npu status
        npu_ok, npu_ver, torch_ver, npu_msg = check_torch_npu_dependency()
        if npu_ok:
            print(f"    torch_npu: {npu_msg}")
        else:
            print(f"    Warning: {npu_msg}")
            if os.environ.get("YIRAGE_SHOW_NPU_GUIDE", "0") == "1":
                print_torch_npu_install_guide(torch_ver)
    elif shutil.which("npu-smi") or os.path.exists("/usr/local/Ascend"):
        detected["ascend"] = True
        print(f"  Detected: Huawei Ascend NPU")
        # Check torch_npu status
        npu_ok, npu_ver, torch_ver, npu_msg = check_torch_npu_dependency()
        if npu_ok:
            print(f"    torch_npu: {npu_msg}")
        else:
            print(f"    Warning: {npu_msg}")
            print(f"    (Set YIRAGE_SHOW_NPU_GUIDE=1 for installation guide)")

    # MetaX MACA
    maca_path = os.environ.get("MACA_PATH", "/opt/maca")
    if os.path.exists(maca_path):
        detected["maca"] = True
        print(f"  Detected: MetaX MACA GPU")

    # Intel XPU
    oneapi_root = os.environ.get("ONEAPI_ROOT")
    if oneapi_root and os.path.exists(oneapi_root):
        detected["xpu"] = True
        print(f"  Detected: Intel XPU (ONEAPI_ROOT={oneapi_root})")
    elif shutil.which("xpu-smi") or os.path.exists("/opt/intel/oneapi"):
        detected["xpu"] = True
        print(f"  Detected: Intel XPU")

    # Google TPU
    if os.environ.get("TPU_NAME") or os.path.exists("/usr/share/tpu"):
        detected["tpu"] = True
        print(f"  Detected: Google TPU")

    # FPGA (Xilinx/Intel)
    if (os.path.exists("/opt/xilinx") or os.path.exists("/opt/Xilinx") or
        os.path.exists("/tools/Xilinx") or shutil.which("vivado") or
        os.path.exists("/opt/intel/oneapi/compiler") or shutil.which("aocl")):
        detected["fpga"] = True
        print(f"  Detected: FPGA (Xilinx/Intel)")

    return detected


def get_backends_from_env():
    """Get backend configuration from environment variables."""
    # Check YIRAGE_BACKEND first
    backend_str = os.environ.get("YIRAGE_BACKEND", "auto").lower()

    # Map of backend names to USE_* flags
    backend_map = {
        "cuda": "USE_CUDA",
        "rocm": "USE_ROCM",
        "mps": "USE_MPS",
        "ascend": "USE_ASCEND",
        "maca": "USE_MACA",
        "xpu": "USE_XPU",
        "tpu": "USE_TPU",
        "fpga": "USE_FPGA",
        "cpu": "USE_CPU",
        "cudnn": "USE_CUDNN",
        "mkl": "USE_MKL",
        "triton": "USE_TRITON",
        "nki": "USE_NKI",
        "mlir": "USE_MLIR",
    }

    backends = {}

    if backend_str == "auto":
        # Auto-detect hardware
        print("Auto-detecting hardware backends...")
        detected = detect_hardware()
        for backend in detected:
            if backend in backend_map:
                backends[backend_map[backend]] = "ON"
    else:
        # Parse comma-separated backends
        for backend in backend_str.split(","):
            backend = backend.strip()
            if backend in backend_map:
                backends[backend_map[backend]] = "ON"

    # Override with individual USE_* environment variables
    for backend, flag in backend_map.items():
        val = os.environ.get(flag)
        if val is not None and str(val).strip() != "":
            coerced = _env_to_cmake_onoff(val)
            if coerced is not None:
                backends[flag] = coerced

    # Ensure at least CPU is enabled
    if not any(v == "ON" for v in backends.values()):
        backends["USE_CPU"] = "ON"

    # yirage_runtime always compiles src/kernel/cpu/*.cc. Headers such as
    # cpu_kernel_config.h (SIMDType, CPUKernelConfig) are guarded by
    # YIRAGE_BACKEND_CPU_ENABLED, which CMake defines only when USE_CPU is ON.
    # Enabling MPS/CUDA/... alone used to leave USE_CPU OFF and break the build.
    # Match cmake/backends/mps.cmake (CPU fallback + shared kernel objects).
    accelerator_flags = (
        "USE_CUDA",
        "USE_ROCM",
        "USE_MPS",
        "USE_XPU",
        "USE_ASCEND",
        "USE_MACA",
        "USE_TPU",
        "USE_FPGA",
        "USE_CUDNN",
        "USE_MKL",
        "USE_MKLDNN",
    )
    codegen_host_flags = (
        "USE_MLIR",
        "USE_STABLEHLO",
        "USE_TVM",
        "USE_TRITON",
        "USE_IREE",
    )
    if any(backends.get(f) == "ON" for f in accelerator_flags) or any(
        backends.get(f) == "ON" for f in codegen_host_flags
    ):
        if "USE_CPU" not in os.environ:
            backends["USE_CPU"] = "ON"

    _merge_extra_use_flags_from_env(backends)

    return backends


def generate_config_cmake(backends, output_path="config.cmake"):
    """Generate config.cmake based on detected/specified backends."""
    # Default all backends to OFF
    all_flags = {
        # GPU
        "USE_CUDA": "OFF",
        "USE_CUDNN": "OFF",
        "USE_CUSPARSELT": "OFF",
        "USE_CUTLASS": "OFF",
        "USE_ROCM": "OFF",
        "USE_MPS": "OFF",
        "USE_XPU": "OFF",
        "USE_ASCEND": "OFF",
        "USE_MACA": "OFF",
        "USE_TPU": "OFF",
        "USE_FPGA": "OFF",
        # CPU
        "USE_CPU": "OFF",
        "USE_MKL": "OFF",
        "USE_MKLDNN": "OFF",
        "USE_OPENMP": "ON",  # Default ON for parallel search
        "USE_XEON": "OFF",
        # DSL
        "USE_NKI": "OFF",
        "USE_TRITON": "OFF",
        # MLIR
        "USE_MLIR": "OFF",
        "USE_STABLEHLO": "OFF",
        "USE_TVM": "OFF",
        "USE_IREE": "OFF",
        # Others
        "USE_MHA": "OFF",
        "USE_NNPACK": "OFF",
        "USE_OPT_EINSUM": "OFF",
        # Build
        "BUILD_CPP_EXAMPLES": "OFF",
        "USE_FORMAL_VERIFIER": "OFF",
        "YIRAGE_BUILD_UNIT_TEST": "OFF",
    }

    # Update with specified backends
    for flag, val in backends.items():
        if flag in all_flags:
            all_flags[flag] = val

    # Auto-enable related backends
    if all_flags["USE_CUDA"] == "ON":
        all_flags["USE_CUTLASS"] = "ON"

    enabled = [k.replace("USE_", "") for k, v in all_flags.items() if v == "ON" and k.startswith("USE_")]
    print(f"Enabled backends: {', '.join(enabled)}")

    # Generate config content
    content = f"""# =============================================================================
# YiRage Build Configuration (Auto-generated)
# =============================================================================
# Generated by: pip install with YIRAGE_BACKEND={os.environ.get('YIRAGE_BACKEND', 'auto')}
# Enabled: {', '.join(enabled)}
# =============================================================================

# GPU Backends
set(USE_CUDA {all_flags['USE_CUDA']})
set(USE_CUDNN {all_flags['USE_CUDNN']})
set(USE_CUSPARSELT {all_flags['USE_CUSPARSELT']})
set(USE_CUTLASS {all_flags['USE_CUTLASS']})
set(USE_ROCM {all_flags['USE_ROCM']})
set(USE_MPS {all_flags['USE_MPS']})
set(USE_XPU {all_flags['USE_XPU']})
set(USE_ASCEND {all_flags['USE_ASCEND']})
set(USE_MACA {all_flags['USE_MACA']})
set(USE_TPU {all_flags['USE_TPU']})
set(USE_FPGA {all_flags['USE_FPGA']})

# CPU Backends
set(USE_CPU {all_flags['USE_CPU']})
set(USE_MKL {all_flags['USE_MKL']})
set(USE_MKLDNN {all_flags['USE_MKLDNN']})
set(USE_OPENMP {all_flags['USE_OPENMP']})
set(USE_XEON {all_flags['USE_XEON']})

# DSL Backends
set(USE_NKI {all_flags['USE_NKI']})
set(USE_TRITON {all_flags['USE_TRITON']})

# MLIR Ecosystem
set(USE_MLIR {all_flags['USE_MLIR']})
set(USE_STABLEHLO {all_flags['USE_STABLEHLO']})
set(USE_TVM {all_flags['USE_TVM']})
set(USE_IREE {all_flags['USE_IREE']})

# Specialized
set(USE_MHA {all_flags['USE_MHA']})
set(USE_NNPACK {all_flags['USE_NNPACK']})
set(USE_OPT_EINSUM {all_flags['USE_OPT_EINSUM']})

# Build Options
set(BUILD_CPP_EXAMPLES {all_flags['BUILD_CPP_EXAMPLES']})
set(USE_FORMAL_VERIFIER {all_flags['USE_FORMAL_VERIFIER']})
set(YIRAGE_BUILD_UNIT_TEST {all_flags['YIRAGE_BUILD_UNIT_TEST']})
"""

    with open(output_path, "w") as f:
        f.write(content)

    return all_flags


# =============================================================================
# Path Configuration
# =============================================================================

nvcc_path = shutil.which("nvcc")
if nvcc_path:
    cuda_home = os.path.dirname(os.path.dirname(nvcc_path))
else:
    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")

cuda_include_dir = os.path.join(cuda_home, "include")
cuda_library_dirs = [
    os.path.join(cuda_home, "lib"),
    os.path.join(cuda_home, "lib64"),
    os.path.join(cuda_home, "lib64", "stubs"),
]

# MACA SDK paths (MetaX GPU)
maca_home = os.environ.get("MACA_PATH") or os.environ.get("MACA_HOME") or "/opt/maca"
maca_include_dir = os.path.join(maca_home, "include")
maca_library_dirs = [
    os.path.join(maca_home, "lib"),
    os.path.join(maca_home, "lib64"),
]

# ROCm paths (AMD GPU)
rocm_home = os.environ.get("ROCM_PATH", "/opt/rocm")
rocm_include_dir = os.path.join(rocm_home, "include")
rocm_library_dirs = [
    os.path.join(rocm_home, "lib"),
    os.path.join(rocm_home, "lib64"),
]

# Ascend paths (Huawei NPU)
ascend_home = os.environ.get("ASCEND_HOME", "/usr/local/Ascend/ascend-toolkit/latest")
ascend_include_dir = os.path.join(ascend_home, "include")
ascend_library_dirs = [
    os.path.join(ascend_home, "lib64"),
    os.path.join(ascend_home, "aarch64-linux", "lib64"),
]

z3_path = path.dirname(z3.__file__)

# Use version.py to get package version
version_file = os.path.join(os.path.dirname(__file__), "python/yirage/version.py")
with open(version_file, "r") as f:
    exec(f.read())  # This will define __version__

def _load_cmake_macros_module():
    """Load python/yirage/cmake_macros.py without importing the yirage package.

    During setuptools configuration, ``yirage.core`` may not be built yet;
    importing ``yirage`` would run ``__init__.py`` and require the native runtime.
    """
    root = path.dirname(path.abspath(__file__))
    mod_path = path.join(root, "python", "yirage", "cmake_macros.py")
    spec = importlib.util.spec_from_file_location("_yirage_cmake_macros", mod_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load cmake macros from {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def get_backend_macros(config_file):
    """Read config.cmake; return define_macros aligned with CMake yirage_runtime."""
    cm = _load_cmake_macros_module()
    macros = cm.macros_from_config(config_file)
    flags = cm.parse_config_cmake(config_file)
    enabled = sorted(
        k.replace("USE_", "")
        for k, v in flags.items()
        if k.startswith("USE_") and v
    )
    print(f"Enabled backends (Cython macros): {', '.join(enabled)}")
    return macros


def _relativize_extension_sources(setup_dir, extensions):
    """Editable installs (PEP 660) require Extension.sources relative to setup.py dir."""
    root = path.abspath(setup_dir)
    prefix = root + path.sep
    for ext in extensions:
        rel_sources = []
        for src in ext.sources:
            if path.isabs(src):
                abs_src = path.abspath(src)
                if abs_src.startswith(prefix):
                    rel = path.relpath(abs_src, root)
                    rel_sources.append(rel.replace("\\", "/"))
                else:
                    rel_sources.append(src.replace("\\", "/"))
            else:
                rel_sources.append(src.replace("\\", "/"))
        ext.sources = rel_sources
    return extensions


def _linux_cpu_blas_link_args(yirage_path: str) -> list:
    """Link libblas when static runtime references cblas (fused rms_matmul)."""
    if sys.platform == "darwin":
        return []
    runtime_lib = path.join(yirage_path, "build", "libyirage_runtime.a")
    if not path.isfile(runtime_lib):
        return []
    try:
        import subprocess

        out = subprocess.check_output(
            ["nm", "-u", runtime_lib], text=True, errors="ignore"
        )
        if "cblas_sgemm" in out:
            return ["-lblas"]
    except Exception:
        pass
    return []


def config_cython():
    sys_cflags = sysconfig.get_config_var("CFLAGS")
    try:
        from Cython.Build import cythonize

        ret = []
        yirage_path = path.dirname(path.abspath(__file__))
        cpu_blas_link = _linux_cpu_blas_link_args(yirage_path)
        config_path = path.join(yirage_path, "config.cmake")
        mlir_link = cython_mlir_extra_link_args(
            config_path, os.environ, platform=sys.platform
        )
        macros = get_backend_macros(config_path)
        cython_path = path.join(yirage_path, "python", "yirage", "_cython")
        # Skip problematic modules on macOS for now
        skip_modules = []
        for fn in os.listdir(cython_path):
            if not fn.endswith(".pyx"):
                continue
            if fn in skip_modules:
                print(f"Skipping {fn} on macOS (requires additional fixes)")
                continue
            pyx_src = path.join("python", "yirage", "_cython", fn).replace("\\", "/")
            ret.append(
                Extension(
                    "yirage.%s" % fn[:-4],
                    [pyx_src],
                    include_dirs=[
                        path.join(yirage_path, "include"),
                        path.join(yirage_path, "deps", "json", "include"),
                        path.join(yirage_path, "deps", "cutlass", "include"),
                        path.join(yirage_path, "deps", "cutlass", "tools", "util", "include"),
                        path.join(yirage_path, "build", "abstract_subexpr", "release"),
                        path.join(yirage_path, "build", "formal_verifier", "release"),
                        path.join(z3_path, "include"),
                        cuda_include_dir,
                        maca_include_dir,
                    ],
                    libraries=[
                        # Note: Core libraries linked via extra_link_args for proper order
                        "z3",
                    ] + (["omp"] if sys.platform == "darwin" else ["gomp"]) + (  # OpenMP library for parallel search
                        ["mcruntime"] if macros and any("MACA" in str(m) for m in macros) else []
                    ),
                    library_dirs=[
                        path.join(yirage_path, "build"),
                        path.join(z3_path, "lib"),
                        path.join(yirage_path, "build", "abstract_subexpr", "release"),
                        path.join(yirage_path, "build", "formal_verifier", "release"),
                    ]
                    + ([_get_homebrew_omp_lib_dir()] if _get_homebrew_omp_lib_dir() else [])
                    + cuda_library_dirs
                    + maca_library_dirs,
                    define_macros=macros,
                    extra_compile_args=["-std=c++17"] + (["-Xpreprocessor", "-fopenmp"] if sys.platform == "darwin" else ["-fopenmp"]),
                    extra_link_args=(
                        # macOS specific link args
                        [
                            "-fPIC",
                            f"-L{path.join(z3_path, 'lib')}",
                            f"-lz3",
                            f"-Wl,-rpath,{path.join(z3_path, 'lib')}",
                            f"-L{path.join(yirage_path, 'build')}",
                            "-lyirage_runtime",
                            f"-L{path.join(yirage_path, 'build', 'abstract_subexpr', 'release')}",
                            "-labstract_subexpr",
                            f"-L{path.join(yirage_path, 'build', 'formal_verifier', 'release')}",
                            "-lformal_verifier",
                            f"-Wl,-rpath,@loader_path/../../build/abstract_subexpr/release",
                            f"-Wl,-rpath,@loader_path/../../build/formal_verifier/release",
                        ]
                        + mlir_link
                        if sys.platform == "darwin" else
                        # Linux specific link args
                        [
                            "-fPIC",
                            "-fopenmp",
                            f"-Wl,--no-as-needed",
                            f"-L{path.join(z3_path, 'lib')}",
                            f"-lz3",
                            f"-Wl,--as-needed",
                            f"-Wl,-rpath,{path.join(z3_path, 'lib')}",
                            f"-Wl,--whole-archive",
                            f"-L{path.join(yirage_path, 'build')}",
                            f"-lyirage_runtime",
                            f"-Wl,--no-whole-archive",
                            f"-L{path.join(yirage_path, 'build', 'abstract_subexpr', 'release')}",
                            f"-labstract_subexpr",
                            f"-L{path.join(yirage_path, 'build', 'formal_verifier', 'release')}",
                            f"-lformal_verifier",
                            f"-Wl,-rpath,{path.join('$ORIGIN', '..', '..', 'build', 'abstract_subexpr', 'release')}",
                            f"-Wl,-rpath,{path.join('$ORIGIN', '..', '..', 'build', 'formal_verifier', 'release')}",
                        ]
                        + cpu_blas_link
                        + mlir_link
                    )
                    # Add CUDA linking only if CUDA backend is enabled
                    + ([
                        "-L/usr/local/cuda/lib64",
                        "-L/usr/local/cuda-12.1/lib64",
                        "-lcudart",
                        "-Wl,-rpath,/usr/local/cuda/lib64",
                        "-Wl,-rpath,/usr/local/cuda-12.1/lib64",
                    ] if macros and any("CUDA" in str(m) for m in macros) else [])
                    # Add MACA linking only if MACA backend is enabled
                    + ([f"-Wl,-rpath,{maca_home}/lib"] if macros and any("MACA" in str(m) for m in macros) else [])
                    # Add Ascend linking only if Ascend backend is enabled
                    + ([
                        "-L/usr/local/Ascend/ascend-toolkit/latest/lib64",
                        "-L/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/lib64",
                        "-lascendcl",
                        "-Wl,-rpath,/usr/local/Ascend/ascend-toolkit/latest/lib64",
                        "-Wl,-rpath,/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/lib64",
                    ] if macros and any("ASCEND" in str(m) for m in macros) else []),
                    language="c++",
                )
            )
        exts = cythonize(ret, compiler_directives={"language_level": 3})
        return _relativize_extension_sources(yirage_path, exts)
    except ImportError:
        print("WARNING: cython is not installed!!!")
        raise SystemExit(1)

yirage_path = path.dirname(__file__)
if yirage_path == '':
    yirage_path = '.'

# Skip Rust build if SKIP_BUILD is set and libraries already exist
skip_rust_build = os.environ.get("SKIP_BUILD") and (
    os.path.exists(os.path.join(yirage_path, 'build', 'abstract_subexpr', 'release', 'libabstract_subexpr.so')) and
    os.path.exists(os.path.join(yirage_path, 'build', 'formal_verifier', 'release', 'libformal_verifier.so'))
)

if skip_rust_build:
    print("Skipping Rust library builds (SKIP_BUILD set and libraries exist)")
else:
    # Install Rust if not yet available
    try:
        # Attempt to run a Rust command to check if Rust is installed
        subprocess.check_output(['cargo', '--version'])
    except FileNotFoundError:
        print("Rust/Cargo not found, installing it...")
        # Rust is not installed, so install it using rustup
        try:
            subprocess.run("curl https://sh.rustup.rs -sSf | sh -s -- -y", shell=True, check=True)
            print("Rust and Cargo installed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")
        # Add the cargo binary directory to the PATH
        os.environ["PATH"] = f"{os.path.join(os.environ.get('HOME', '/root'), '.cargo', 'bin')}:{os.environ.get('PATH', '')}"

    try:
        subprocess.check_output(['cargo', 'build', '--release', '--target-dir', '../../../../build/abstract_subexpr'], cwd='src/search/abstract_expr/abstract_subexpr')
    except subprocess.CalledProcessError as e:
        print("Failed to build abstract_subexpr Rust library, building it ...")
        try:
            subprocess.run(['cargo', 'build', '--release', '--target-dir', '../../../../build/abstract_subexpr'], cwd='src/search/abstract_expr/abstract_subexpr', check=True)
            print("Abstract_subexpr Rust library built successfully.")
        except subprocess.CalledProcessError as e:
            print("Failed to build abstract_subexpr Rust library.")
        os.environ['ABSTRACT_SUBEXPR_LIB'] = os.path.join(yirage_path,'build', 'abstract_subexpr', 'release', 'libabstract_subexpr.so')

    try:
        subprocess.check_output(['cargo', 'build', '--release', '--target-dir', '../../../../build/formal_verifier'], cwd='src/search/verification/formal_verifier_equiv')
    except subprocess.CalledProcessError as e:
        print("Failed to build formal_verifier Rust library, building it ...")
        try:
            subprocess.run(['cargo', 'build', '--release', '--target-dir', '../../../../build/formal_verifier'], cwd='src/search/verification/formal_verifier_equiv', check=True)
            print("formal_verifier Rust library built successfully.")
        except subprocess.CalledProcessError as e:
            print("Failed to build formal_verifier Rust library.")
        os.environ['FORMAL_VERIFIER_LIB'] = os.path.join(yirage_path,'build', 'formal_verifier', 'release', 'libformal_verifier.so')


# =============================================================================
# Generate config.cmake from environment
# =============================================================================
yirage_path = path.dirname(__file__)
if yirage_path == "":
    yirage_path = "."

# Check if we should auto-generate config.cmake
if _should_regenerate_config_cmake():
    print("\n" + "=" * 60)
    print("YiRage Backend Configuration")
    print("=" * 60)
    backends = get_backends_from_env()
    config_path = os.path.join(yirage_path, "config.cmake")
    generate_config_cmake(backends, config_path)
    print("=" * 60 + "\n")

# build YiRage runtime library
try:
    os.environ["CUDACXX"] = nvcc_path if nvcc_path else os.path.join(
        cuda_home, "bin", "nvcc"
    )
    # z3_path = os.path.join(yirage_path, 'deps', 'z3', 'build')
    # os.environ['Z3_DIR'] = z3_path
    os.makedirs(yirage_path, exist_ok=True)
    os.chdir(yirage_path)
    build_dir = os.path.join(yirage_path, "build")

    # Check if library already exists and skip rebuild if SKIP_BUILD is set
    runtime_lib = os.path.join(build_dir, "libyirage_runtime.a")
    skip_cmake_build = os.environ.get("SKIP_BUILD") and os.path.exists(runtime_lib)
    if skip_cmake_build:
        print(f"Found existing runtime library at {runtime_lib}, skipping cmake build...")

    # Detect compiler
    if platform.system() == "Darwin":
        # macOS: prefer clang
        cc_path = shutil.which("clang") or shutil.which("gcc")
        cxx_path = shutil.which("clang++") or shutil.which("g++")
    else:
        cc_path = shutil.which("gcc")
        cxx_path = shutil.which("g++")

    os.environ["CC"] = cc_path if cc_path else "/usr/bin/gcc"
    os.environ["CXX"] = cxx_path if cxx_path else "/usr/bin/g++"
    print(f"CC: {os.environ['CC']}, CXX: {os.environ['CXX']}", flush=True)

    # Create the build directory if it does not exist
    os.makedirs(build_dir, exist_ok=True)

    # Determine Z3 library extension based on platform
    if platform.system() == "Darwin":
        z3_lib_name = "libz3.dylib"
    else:
        z3_lib_name = "libz3.so"

    # Build cmake command with backend options from environment
    cmake_args = [
        "cmake",
        "..",
        "-DCMAKE_BUILD_TYPE=Debug",
        "-DZ3_CXX_INCLUDE_DIRS=" + z3_path + "/include/",
        "-DZ3_LIBRARIES=" + path.join(z3_path, "lib", z3_lib_name),
        '-DABSTRACT_SUBEXPR_LIB=' + path.join(yirage_path, 'build', 'abstract_subexpr', 'release'),
        '-DABSTRACT_SUBEXPR_LIBRARIES=' + path.join(yirage_path, 'build', 'abstract_subexpr', 'release', 'libabstract_subexpr.so'),
        '-DFORMAL_VERIFIER_LIB=' + path.join(yirage_path, 'build', 'formal_verifier', 'release'),
        '-DFORMAL_VERIFIER_LIBRARIES=' + path.join(yirage_path, 'build', 'formal_verifier', 'release', 'libformal_verifier.so'),
        "-DCMAKE_C_COMPILER=" + os.environ["CC"],
        "-DCMAKE_CXX_COMPILER=" + os.environ["CXX"],
    ]

    # Add all backend options from environment variables
    backend_env_vars = [
        "USE_CUDA", "USE_CUDNN", "USE_CUTLASS", "USE_CUSPARSELT",
        "USE_ROCM", "USE_MPS", "USE_XPU", "USE_ASCEND", "USE_MACA",
        "USE_TPU", "USE_FPGA", "USE_CPU", "USE_MKL", "USE_MKLDNN",
        "USE_OPENMP", "USE_NKI", "USE_TRITON", "USE_MLIR",
    ]
    for var in backend_env_vars:
        val = os.environ.get(var)
        if val is not None and str(val).strip() != "":
            coerced = _env_to_cmake_onoff(val)
            cmake_args.append(f"-D{var}={coerced if coerced is not None else val}")

    # Add hardware-specific paths
    if os.environ.get("USE_CUDA") == "ON" or os.environ.get("YIRAGE_BACKEND", "").find("cuda") >= 0:
        cmake_args.append(f"-DCUDA_TOOLKIT_ROOT_DIR={cuda_home}")

    if os.environ.get("USE_ROCM") == "ON" or os.environ.get("YIRAGE_BACKEND", "").find("rocm") >= 0:
        cmake_args.append(f"-DROCM_PATH={rocm_home}")

    if os.environ.get("USE_ASCEND") == "ON" or os.environ.get("YIRAGE_BACKEND", "").find("ascend") >= 0:
        cmake_args.append(f"-DASCEND_HOME={ascend_home}")

    if os.environ.get("USE_MACA") == "ON" or os.environ.get("YIRAGE_BACKEND", "").find("maca") >= 0:
        cmake_args.append(f"-DMACA_PATH={maca_home}")

    # Add Z3_DIR from environment if set
    z3_dir = os.environ.get("Z3_DIR")
    if z3_dir:
        cmake_args.append(f"-DZ3_DIR={z3_dir}")

    for arg in cmake_mlir_extra_definitions(yirage_path, os.environ):
        cmake_args.append(arg)

    # Add extra CXX flags if needed (e.g., for Z3 include)
    cxxflags = os.environ.get("CXXFLAGS", "")
    if cxxflags:
        cmake_args.append(f"-DCMAKE_CXX_FLAGS={cxxflags}")

    if not skip_cmake_build:
        subprocess.check_call(
            cmake_args,
            cwd=build_dir,
            env=os.environ.copy(),
        )
        subprocess.check_call(["make", "-j8"], cwd=build_dir, env=os.environ.copy())
        print("YiRage runtime library built successfully.")
    else:
        print("Using pre-built runtime library.")
except subprocess.CalledProcessError as e:
    print("Failed to build runtime library.")
    raise SystemExit(e.returncode)

setup_args = {}

# Runtime dependencies should match pyproject.toml [project.dependencies].
# requirements.txt intentionally describes the larger local dev/CI environment.
requirements = [
    "accelforge>=1.0.355",
    "numpy>=1.21.0",
    "z3-solver>=4.12.0",
    "graphviz>=0.20.0",
    "tqdm>=4.65.0",
    "ray>=2.55.0",
]
print(f"Runtime requirements: {requirements}")

INCLUDE_BASE = "python/yirage/include"


@contextmanager
def copy_include():
    if not path.exists(INCLUDE_BASE):
        src_dirs = ["deps/cutlass/include", "deps/json/include"]
        for src_dir in src_dirs:
            shutil.copytree(src_dir, path.join(INCLUDE_BASE, src_dir))
        # copy include/transpiler/runtime/*
        # to python/yirage/include/transpiler/runtime/*
        # instead of python/yirage/include/include/transpiler/runtime/*
        include_yirage_dirs = [
            "include/transpiler/runtime",
            "include/triton_transpiler/runtime",
            "include/persistent_kernel",
        ]
        include_yirage_dsts = [
            path.join(INCLUDE_BASE, "transpiler/runtime"),
            path.join(INCLUDE_BASE, "triton_transpiler/runtime"),
            path.join(INCLUDE_BASE, "persistent_kernel"),
        ]
        for include_yirage_dir, include_yirage_dst in zip(
            include_yirage_dirs, include_yirage_dsts
        ):
            shutil.copytree(include_yirage_dir, include_yirage_dst)

        config_h_src = path.join(
            yirage_path, "include/config.h"
        )  # Needed by transpiler/runtime/threadblock/utils.h
        config_h_dst = path.join(INCLUDE_BASE, "config.h")
        shutil.copy(config_h_src, config_h_dst)
        yield True
    else:
        yield False
    shutil.rmtree(INCLUDE_BASE)


with copy_include() as copied:
    if not copied:
        print(
            "WARNING: include directory already exists. Not copying again. "
            f"This may cause issues. Please remove {INCLUDE_BASE} and rerun setup.py",
            flush=True,
        )

    setup(
        name="yirage",
        version=__version__,
        description=(
            "YiRage - Multi-Backend LLM Inference Optimization Engine "
            "(requires built libyirage_runtime / yirage.core at install time)"
        ),
        zip_safe=False,
        install_requires=requirements,
        python_requires=">=3.8",
        packages=find_packages(where="python"),
        package_dir={"": "python"},
        url="https://github.com/chenxingqiang/YiRage",
        license="Apache-2.0",
        ext_modules=config_cython(),
        include_package_data=True,
        # **setup_args,
    )
