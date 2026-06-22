#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""
Focused AccelForge bridge tests loaded without importing ``yirage.__init__``.
"""

import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

_PYTHON_ROOT = Path(__file__).parent.parent.parent.parent / "python"
EXPECTED_MOCK_CONFIDENCE = 0.90


def _install_namespace_package(name: str, path: Path) -> None:
    """Install a namespace-package shim so direct file loads can resolve relative imports."""
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    module.__package__ = name
    sys.modules[name] = module


def _load_module(module_name: str, path: Path) -> types.ModuleType:
    """Load a module from a source file and register it in sys.modules."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        pytest.skip(f"{module_name} module not found")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_accelforge_bridge():
    """Load the AccelForge bridge module without importing top-level yirage."""
    _install_namespace_package("yirage", _PYTHON_ROOT / "yirage")
    _install_namespace_package("yirage.rl", _PYTHON_ROOT / "yirage" / "rl")
    _install_namespace_package("yirage.rl.hardware", _PYTHON_ROOT / "yirage" / "rl" / "hardware")
    return _load_module(
        "yirage.rl.hardware.accelforge_bridge",
        _PYTHON_ROOT / "yirage" / "rl" / "hardware" / "accelforge_bridge.py",
    )


def _load_accelforge_verifier():
    """Load AccelForgeVerifier and its lightweight dependencies directly."""
    _install_namespace_package("yirage", _PYTHON_ROOT / "yirage")
    _install_namespace_package("yirage.rl", _PYTHON_ROOT / "yirage" / "rl")
    _install_namespace_package("yirage.rl.verifier", _PYTHON_ROOT / "yirage" / "rl" / "verifier")
    _load_module(
        "yirage.rl.verifier.gpu_verifier",
        _PYTHON_ROOT / "yirage" / "rl" / "verifier" / "gpu_verifier.py",
    )
    return _load_module(
        "yirage.rl.verifier.accelforge_verifier",
        _PYTHON_ROOT / "yirage" / "rl" / "verifier" / "accelforge_verifier.py",
    )


def test_availability_diagnostics_shape():
    bridge_mod = _load_accelforge_bridge()

    status = bridge_mod.get_accelforge_availability()

    assert set(status) >= {
        "available",
        "installed",
        "version",
        "minimum_version",
        "maximum_version_exclusive",
        "supported_version",
        "reason",
    }
    assert isinstance(status["available"], bool)


@pytest.mark.parametrize(
    ("expected_op_type", "tensors", "operators", "expected"),
    [
        (
            "matmul",
            [
                {"id": 0, "shape": [4, 16, 32]},
                {"id": 1, "shape": [4, 32, 64]},
            ],
            [{"type": "matmul", "input_tensor_ids": [0, 1]}],
            {"m_dim": 16, "k_dim": 32, "n_dim": 64, "batch_dim": 4},
        ),
        (
            "attention",
            [{"id": 0, "shape": [2, 8, 128, 64]}],
            [{"type": "attention", "input_tensor_ids": [0]}],
            {"m_dim": 128, "k_dim": 64, "n_dim": 128, "batch_dim": 16},
        ),
        (
            "conv",
            [
                {"id": 0, "shape": [1, 3, 32, 32]},
                {"id": 1, "shape": [16, 3, 3, 3]},
            ],
            [{"type": "conv", "input_tensor_ids": [0, 1]}],
            {"m_dim": 900, "k_dim": 27, "n_dim": 16, "batch_dim": 1},
        ),
        (
            "reduction",
            [{"id": 0, "shape": [16, 32]}],
            [{"type": "reduction", "input_tensor_ids": [0]}],
            {"m_dim": 16, "k_dim": 32, "n_dim": 1, "batch_dim": 1},
        ),
        (
            "relu",
            [{"id": 0, "shape": [8, 16]}],
            [{"type": "relu", "input_tensor_ids": [0]}],
            {"m_dim": 1, "k_dim": 128, "n_dim": 1, "batch_dim": 1},
        ),
    ],
)
def test_mugraph_to_workload_operator_shapes(expected_op_type, tensors, operators, expected):
    bridge_mod = _load_accelforge_bridge()
    graph_json = json.dumps({"tensors": tensors, "operators": operators})

    workload = bridge_mod.mugraph_to_workload(graph_json)

    assert workload["dominant_op_type"] == expected_op_type
    for key, value in expected.items():
        assert workload[key] == value
    assert workload["num_operators"] == 1
    assert workload["estimated_flops"] > 0


def test_multi_op_workload_preserves_total_flops_and_counts():
    bridge_mod = _load_accelforge_bridge()
    graph_json = json.dumps(
        {
            "tensors": [
                {"id": 0, "shape": [16, 32]},
                {"id": 1, "shape": [32, 64]},
                {"id": 2, "shape": [16, 64]},
            ],
            "operators": [
                {"type": "matmul", "input_tensor_ids": [0, 1]},
                {"type": "relu", "input_tensor_ids": [2]},
            ],
        }
    )

    workload = bridge_mod.mugraph_to_workload(graph_json)

    assert workload["dominant_op_type"] == "matmul"
    assert workload["num_operators"] == 2
    assert workload["operator_counts"]["matmul"] == 1
    assert workload["operator_counts"]["relu"] == 1
    assert workload["estimated_flops"] > 2 * 16 * 32 * 64


def test_evaluate_with_mock_accelforge_spec():
    bridge_mod = _load_accelforge_bridge()

    class FakeResults:
        def energy(self, per_component=False):
            if per_component:
                return {"MAC": 100.0, "L1Buffer": 20.0, "L2Buffer": 10.0, "MainMemory": 5.0}
            return 135.0

        def latency(self):
            return 1000

        def n_computes(self):
            return 256

        def resource_usage(self):
            return {"L1Buffer": 0.25}

    class FakeSpec:
        @classmethod
        def from_yaml(cls, arch_path, workload_path):
            return cls()

        def map_workload_to_arch(self, print_progress=False):
            return FakeResults()

        def calculate_component_area_energy_latency_leak(self):
            return SimpleNamespace(
                arch=SimpleNamespace(
                    nodes=[
                        SimpleNamespace(
                            name="MAC",
                            total_area=1.0,
                            area=0.0,
                            total_leak_power=2.0,
                            leak_power=0.0,
                        ),
                        SimpleNamespace(
                            name="L1Buffer",
                            total_area=3.0,
                            area=0.0,
                            total_leak_power=4.0,
                            leak_power=0.0,
                        ),
                    ]
                )
            )

    bridge = bridge_mod.AccelForgeBridge()
    bridge._af_model = FakeSpec
    metrics = bridge.evaluate(
        bridge_mod.AccelForgeDesignPoint(pe_array_rows=4, pe_array_cols=4),
        {"m_dim": 16, "k_dim": 32, "n_dim": 64},
    )

    assert metrics.confidence == EXPECTED_MOCK_CONFIDENCE
    assert metrics.area_mm2 == 4.0
    assert metrics.energy_per_op_pj == pytest.approx(135.0 / 256.0)


def test_yirage_mugraph_workload_drives_accelforge_yaml_dimensions():
    bridge_mod = _load_accelforge_bridge()
    captured = {}

    class FakeResults:
        def energy(self, per_component=False):
            if per_component:
                return {"MAC": 1.0}
            return 1.0

        def latency(self):
            return 1

        def n_computes(self):
            return 1

        def resource_usage(self):
            return {}

    class CapturingSpec:
        @classmethod
        def from_yaml(cls, arch_path, workload_path):
            captured["workload"] = Path(workload_path).read_text()
            return cls()

        def map_workload_to_arch(self, print_progress=False):
            return FakeResults()

        def calculate_component_area_energy_latency_leak(self):
            return SimpleNamespace(arch=SimpleNamespace(nodes=[]))

    graph_json = json.dumps(
        {
            "tensors": [
                {"id": 0, "shape": [2, 8, 16]},
                {"id": 1, "shape": [2, 16, 32]},
            ],
            "operators": [{"type": "batch_matmul", "input_tensor_ids": [0, 1]}],
        }
    )

    workload = bridge_mod.mugraph_to_workload(graph_json)
    bridge = bridge_mod.AccelForgeBridge()
    bridge._af_model = CapturingSpec
    bridge.evaluate(
        bridge_mod.AccelForgeDesignPoint(data_precision="fp32"),
        workload,
    )

    assert workload["effective_m_dim"] == 16
    assert "    M: 16\n" in captured["workload"]
    assert "    K: 16\n" in captured["workload"]
    assert "    N: 32\n" in captured["workload"]
    assert "  bits_per_value: {All: 32}\n" in captured["workload"]


def test_evaluate_passes_yirage_workload_yaml_to_accelforge_spec():
    bridge_mod = _load_accelforge_bridge()
    captured = {}

    class FakeResults:
        def energy(self, per_component=False):
            if per_component:
                return {"MAC": 80.0, "L1Buffer": 10.0, "L2Buffer": 5.0, "MainMemory": 5.0}
            return 100.0

        def latency(self):
            return 512

        def n_computes(self):
            return 128

        def resource_usage(self):
            return {"L1Buffer": 0.5}

    class CapturingSpec:
        @classmethod
        def from_yaml(cls, arch_path, workload_path):
            captured["arch"] = Path(arch_path).read_text()
            captured["workload"] = Path(workload_path).read_text()
            return cls()

        def map_workload_to_arch(self, print_progress=False):
            assert print_progress is False
            return FakeResults()

        def calculate_component_area_energy_latency_leak(self):
            return SimpleNamespace(
                arch=SimpleNamespace(
                    nodes=[
                        SimpleNamespace(
                            name="MAC",
                            total_area=2.0,
                            area=0.0,
                            total_leak_power=1.0,
                            leak_power=0.0,
                        ),
                        SimpleNamespace(
                            name="L1Buffer",
                            total_area=1.0,
                            area=0.0,
                            total_leak_power=1.0,
                            leak_power=0.0,
                        ),
                    ]
                )
            )

    bridge = bridge_mod.AccelForgeBridge()
    bridge._af_model = CapturingSpec
    workload = bridge_mod.mugraph_to_workload(
        json.dumps(
            {
                "tensors": [{"id": 0, "shape": [4, 64]}, {"id": 1, "shape": [64, 128]}],
                "operators": [{"type": "matmul", "input_tensor_ids": [0, 1]}],
            }
        )
    )

    metrics = bridge.evaluate(
        bridge_mod.AccelForgeDesignPoint(pe_array_rows=8, pe_array_cols=4),
        workload,
    )

    assert "n_parallel_instances: 32" in captured["arch"]
    assert "    M: 4\n" in captured["workload"]
    assert "    K: 64\n" in captured["workload"]
    assert "    N: 128\n" in captured["workload"]
    assert metrics.confidence == EXPECTED_MOCK_CONFIDENCE


def test_verifier_profiles_yirage_mugraph_with_accelforge_workload(monkeypatch):
    bridge_mod = _load_accelforge_bridge()
    verifier_mod = _load_accelforge_verifier()
    captured_workloads = []

    class CapturingBridge:
        def evaluate(self, design, workload):
            captured_workloads.append(dict(workload))
            return SimpleNamespace(latency_ms=0.125)

    monkeypatch.setattr(bridge_mod, "AccelForgeBridge", CapturingBridge)

    verifier = verifier_mod.AccelForgeVerifier(
        design_point={"pe_array_rows": 8, "pe_array_cols": 8}
    )
    result = verifier.profile_kernel(
        json.dumps(
            {
                "tensors": [
                    {"id": 0, "shape": [2, 8, 16]},
                    {"id": 1, "shape": [2, 16, 32]},
                ],
                "operators": [{"type": "batch_matmul", "input_tensor_ids": [0, 1]}],
            }
        ),
        input_shapes=[[2, 8, 16], [2, 16, 32]],
    )

    assert result.latency_ms == 0.125
    assert result.memory_bytes == 2560
    assert result.flops == captured_workloads[0]["estimated_flops"]
    assert captured_workloads[0]["m_dim"] == 8
    assert captured_workloads[0]["k_dim"] == 16
    assert captured_workloads[0]["n_dim"] == 32
    assert captured_workloads[0]["batch_dim"] == 2


CY_MATMUL_GRAPH = [
    {
        "op_type": "kn_input_op",
        "input_tensors": [],
        "output_tensors": [
            {"guid": 10000003, "num_dims": 2, "dim": [8, 64, 0, 0]},
        ],
    },
    {
        "op_type": "kn_input_op",
        "input_tensors": [],
        "output_tensors": [
            {"guid": 10000004, "num_dims": 2, "dim": [64, 64, 0, 0]},
        ],
    },
    {
        "op_type": "kn_matmul_op",
        "input_tensors": [
            {"guid": 10000003, "num_dims": 2, "dim": [8, 64, 0, 0]},
            {"guid": 10000004, "num_dims": 2, "dim": [64, 64, 0, 0]},
        ],
        "output_tensors": [
            {"guid": 10000005, "num_dims": 2, "dim": [8, 64, 0, 0]},
        ],
    },
    {
        "op_type": "kn_output_op",
        "input_tensors": [
            {"guid": 10000005, "num_dims": 2, "dim": [8, 64, 0, 0]},
        ],
        "output_tensors": [],
    },
]


def test_mugraph_to_workload_accepts_cy_to_json_matmul_list():
    bridge_mod = _load_accelforge_bridge()
    workload = bridge_mod.mugraph_to_workload(json.dumps(CY_MATMUL_GRAPH))

    assert workload["op_type"] == "matmul"
    assert workload["m_dim"] == 8
    assert workload["k_dim"] == 64
    assert workload["n_dim"] == 64
    assert workload["estimated_flops"] == 2 * 8 * 64 * 64


def test_mugraph_to_workload_accepts_mugraph_cache_variants():
    bridge_mod = _load_accelforge_bridge()
    relu_variant = [
        {
            "op_type": "kn_relu_op",
            "input_tensors": [
                {"guid": 1, "num_dims": 2, "dim": [4, 4, 0, 0]},
            ],
            "output_tensors": [
                {"guid": 2, "num_dims": 2, "dim": [4, 4, 0, 0]},
            ],
        }
    ]
    cache_entry = [relu_variant, CY_MATMUL_GRAPH]
    workload = bridge_mod.mugraph_to_workload(json.dumps(cache_entry))

    assert workload["op_type"] == "matmul"
    assert workload["m_dim"] == 8
    assert workload["n_dim"] == 64


def test_prescreen_accepts_cy_to_json_kernel_graph(monkeypatch):
    bridge_mod = _load_accelforge_bridge()
    verifier_mod = _load_accelforge_verifier()

    class CapturingBridge:
        def evaluate(self, design, workload):
            return SimpleNamespace(
                latency_ms=0.5,
                energy_per_op_pj=1.0,
                area_mm2=1.0,
                total_power_mw=1.0,
                to_dict=lambda: {
                    "latency_ms": 0.5,
                    "energy_per_op_pj": 1.0,
                    "area_mm2": 1.0,
                    "total_power_mw": 1.0,
                },
            )

    monkeypatch.setattr(bridge_mod, "AccelForgeBridge", CapturingBridge)

    verifier = verifier_mod.AccelForgeVerifier()
    result = verifier.prescreen_kernel(json.dumps(CY_MATMUL_GRAPH))

    assert result["verified"] is True
    assert result["accepted"] is True
    assert result["metrics"]["latency_ms"] == 0.5
