#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""
YiRage Test Configuration and Fixtures

Centralized pytest configuration for all Python tests.
Run with: pytest tests/python/ -v
"""

import sys
import os
import json
import importlib.util
from pathlib import Path
from typing import Optional, Dict, Any
import pytest

from tests.python._yirage_test_support import ensure_native_library_path, restore_real_yirage_if_shimmed

# =============================================================================
# Path Setup
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
PYTHON_ROOT = PROJECT_ROOT / "python"
MLIR_ROOT = PROJECT_ROOT / "mlir" / "python"

# Add paths
sys.path.insert(0, str(PYTHON_ROOT))
if MLIR_ROOT.exists():
    sys.path.insert(0, str(MLIR_ROOT))


# =============================================================================
# Availability Detection
# =============================================================================

def _check_torch():
    """Check PyTorch availability."""
    try:
        import torch
        return True, torch.cuda.is_available(), (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
    except ImportError:
        return False, False, False


def _check_ray():
    """Check Ray availability."""
    try:
        import ray
        return True
    except ImportError:
        return False


def _check_numpy():
    """Check NumPy availability."""
    try:
        import numpy
        return True
    except ImportError:
        return False


TORCH_AVAILABLE, CUDA_AVAILABLE, MPS_AVAILABLE = _check_torch()
RAY_AVAILABLE = _check_ray()
NUMPY_AVAILABLE = _check_numpy()


# =============================================================================
# Module Loading Utilities
# =============================================================================

def load_module(name: str, path: Path) -> Optional[Any]:
    """Load Python module from path without raising ImportError.

    Prefer safe_import() for installed packages; this is kept for
    files that are not part of the installed yirage package.
    """
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        return module
    except Exception:
        if name in sys.modules:
            del sys.modules[name]
        return None


def _is_yirage_test_shim(module: Optional[Any]) -> bool:
    """Return True if ``module`` is the bare yirage namespace stub installed by
    ``tests/python/test_rl/conftest.py`` (used so that ``import yirage.rl.*``
    works without the native runtime). Such a stub has no real package
    attributes (``__version__``, ``new_kernel_graph``, ``HardwareRegistry``,
    ``get_available_backends`` …), so tests that depend on the real ``yirage``
    package should treat it as "not available".
    """
    return getattr(module, "_is_test_shim", False) is True


def safe_import(dotted_name: str) -> Optional[Any]:
    """Import a module by its fully-qualified dotted name, returning None on failure."""
    try:
        module = importlib.import_module(dotted_name)
    except Exception:
        return None
    # The top-level ``yirage`` shim is not a real package — callers asking
    # specifically for ``yirage`` should see it as unavailable. Submodule
    # imports (e.g. ``yirage.rl``) still return the real module.
    if dotted_name == "yirage" and _is_yirage_test_shim(module):
        return None
    return module


def check_module_syntax(path: Path) -> tuple[bool, str]:
    """Check if a Python file has valid syntax."""
    try:
        with open(path, "r") as f:
            source = f.read()
        compile(source, str(path), "exec")
        return True, ""
    except SyntaxError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)


# =============================================================================
# Pytest Configuration
# =============================================================================

def pytest_sessionstart(session):
    """Ensure native helper libraries are on LD_LIBRARY_PATH for yirage.core."""
    ensure_native_library_path()


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "cuda: tests requiring CUDA")
    config.addinivalue_line("markers", "mps: tests requiring MPS (Apple Silicon)")
    config.addinivalue_line("markers", "ray: tests requiring Ray")
    config.addinivalue_line("markers", "torch: tests requiring PyTorch")
    config.addinivalue_line("markers", "slow: slow tests")
    config.addinivalue_line("markers", "integration: integration tests")
    config.addinivalue_line("markers", "coverage: module coverage tests")


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests based on available dependencies."""
    skip_cuda = pytest.mark.skip(reason="CUDA not available")
    skip_mps = pytest.mark.skip(reason="MPS not available")
    skip_ray = pytest.mark.skip(reason="Ray not available")
    skip_torch = pytest.mark.skip(reason="PyTorch not available")

    # Use explicit markers only — parametrize values (e.g. backend="cuda") also
    # appear in item.keywords and must not trigger hardware skips.
    for item in items:
        if item.get_closest_marker("cuda") and not CUDA_AVAILABLE:
            item.add_marker(skip_cuda)
        if item.get_closest_marker("mps") and not MPS_AVAILABLE:
            item.add_marker(skip_mps)
        if item.get_closest_marker("ray") and not RAY_AVAILABLE:
            item.add_marker(skip_ray)
        if item.get_closest_marker("torch") and not TORCH_AVAILABLE:
            item.add_marker(skip_torch)


# =============================================================================
# Core Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def _prefer_real_yirage_over_rl_shim(request):
    """Drop the RL namespace shim before non-RL tests when native core is built."""
    nodeid = request.node.nodeid.replace("\\", "/")
    if "/test_rl/" in nodeid:
        yield
        return
    restore_real_yirage_if_shimmed()
    yield


@pytest.fixture(scope="session")
def project_root():
    """Get project root directory."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def python_root():
    """Get Python source root directory."""
    return PYTHON_ROOT


@pytest.fixture(scope="session")
def device():
    """Get best available device."""
    if CUDA_AVAILABLE:
        return "cuda:0"
    elif MPS_AVAILABLE:
        return "mps"
    return "cpu"


# =============================================================================
# Ray Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def ray_session():
    """Initialize Ray for the test session."""
    if not RAY_AVAILABLE:
        pytest.skip("Ray not available")

    import ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, logging_level="ERROR")
    yield ray
    # Don't shutdown - might be used by other processes


# =============================================================================
# Module Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def yirage_module():
    """Load yirage main module."""
    try:
        import yirage
    except ImportError:
        return None
    if _is_yirage_test_shim(yirage):
        return None
    return yirage


@pytest.fixture(scope="module")
def backend_api_module():
    """Load backend API module."""
    return safe_import("yirage.backends.api")


@pytest.fixture(scope="module")
def compiler_module():
    """Load compiler module."""
    return safe_import("yirage.compiler")


@pytest.fixture(scope="module")
def rl_features_module():
    """Load RL features module."""
    return safe_import("yirage.rl.features.mugraph_features")


@pytest.fixture(scope="module")
def rl_processor_module():
    """Load RL processor module."""
    return safe_import("yirage.rl.features.processor")


@pytest.fixture(scope="module")
def rl_reward_module():
    """Load RL reward module."""
    return safe_import("yirage.rl.env.reward")


@pytest.fixture(scope="module")
def rl_observation_module():
    """Load RL observation module."""
    return safe_import("yirage.rl.env.observation")


@pytest.fixture(scope="module")
def rl_search_config_module():
    """Load RL search config module."""
    return safe_import("yirage.rl.search.config_space")


@pytest.fixture(scope="module")
def rl_search_graph_module():
    """Load RL search graph module."""
    return safe_import("yirage.rl.search.graph_space")


@pytest.fixture(scope="module")
def rl_policy_module():
    """Load RL policy network module."""
    return safe_import("yirage.rl.models.search_policy")


@pytest.fixture(scope="module")
def rl_verifier_module():
    """Load RL GPU verifier module."""
    return safe_import("yirage.rl.verifier.gpu_verifier")


@pytest.fixture(scope="module")
def storage_module():
    """Load storage module."""
    return safe_import("yirage.storage.mugraph_store")


@pytest.fixture(scope="module")
def profiler_module():
    """Load profiler module."""
    return safe_import("yirage.profiler.hardware")


# =============================================================================
# Test Data Fixtures
# =============================================================================

@pytest.fixture
def sample_mugraph_json():
    """Sample µGraph JSON for testing."""
    return json.dumps({
        "operators": [
            {
                "op_id": 0,
                "op_type": "matmul",
                "op_type_id": 0,
                "num_inputs": 2,
                "num_outputs": 1,
                "flops": 2097152.0,
                "memory_read_bytes": 16384,
                "memory_write_bytes": 8192,
                "input_tensor_ids": [0, 1],
                "output_tensor_ids": [2],
            },
            {
                "op_id": 1,
                "op_type": "silu",
                "op_type_id": 5,
                "num_inputs": 1,
                "num_outputs": 1,
                "flops": 2048.0,
                "memory_read_bytes": 8192,
                "memory_write_bytes": 8192,
                "input_tensor_ids": [2],
                "output_tensor_ids": [3],
            },
        ],
        "tensors": [
            {"tensor_id": 0, "dims": [64, 128], "dtype": "float16", "dtype_id": 0,
             "size_bytes": 16384, "memory_level": 1, "is_input": True, "is_output": False},
            {"tensor_id": 1, "dims": [128, 64], "dtype": "float16", "dtype_id": 0,
             "size_bytes": 16384, "memory_level": 1, "is_input": True, "is_output": False},
            {"tensor_id": 2, "dims": [64, 64], "dtype": "float16", "dtype_id": 0,
             "size_bytes": 8192, "memory_level": 0, "is_input": False, "is_output": False},
            {"tensor_id": 3, "dims": [64, 64], "dtype": "float16", "dtype_id": 0,
             "size_bytes": 8192, "memory_level": 0, "is_input": False, "is_output": True},
        ],
        "edges": [[0, 1]],
        "num_operators": 2,
        "num_tensors": 4,
        "graph_depth": 2,
        "graph_width": 1,
        "critical_path_length": 2,
        "parallelism_degree": 0.5,
        "grid_dim": {"x": 4, "y": 1, "z": 1},
        "block_dim": {"x": 128, "y": 1, "z": 1},
        "forloop_range": 8,
        "reduction_dimx": 16,
        "occupancy": 0.75,
        "shared_mem_usage": 8192,
        "register_usage": 32,
        "theoretical_flops": 2099200.0,
        "memory_bandwidth_utilization": 0.6,
        "arithmetic_intensity": 128.0,
        "estimated_latency_ms": 0.05,
        "search_level": 2,
        "search_depth": 5,
    })


# =============================================================================
# Backend Config Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def backend_configs():
    """Load all backend config modules."""
    configs = {}
    backends = ["cuda", "mps", "rocm", "cpu", "ascend", "maca", "tpu", "xpu", "fpga"]

    for backend in backends:
        path = PYTHON_ROOT / "yirage" / "backends" / backend / "config.py"
        module = load_module(f"{backend}_config", path)
        if module:
            configs[backend] = module

    return configs


# =============================================================================
# Helper Functions (Exposed as fixtures)
# =============================================================================

@pytest.fixture
def module_loader():
    """Return module loader function."""
    return load_module


@pytest.fixture
def syntax_checker():
    """Return syntax checker function."""
    return check_module_syntax
