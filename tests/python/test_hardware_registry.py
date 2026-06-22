#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""
Hardware Registry Unit Tests

Tests for yirage/hardware/ module.
Run with: pytest tests/python/test_hardware_registry.py -v
"""

import json
import pytest
from pathlib import Path

from conftest import PYTHON_ROOT, load_module, safe_import


# =============================================================================
# Module Loading
# =============================================================================


@pytest.fixture(scope="module")
def hw_module():
    """Import the hardware module."""
    return safe_import("yirage.hardware")


@pytest.fixture(scope="module")
def chip_arch_mod():
    return safe_import("yirage.hardware.chip_arch")


@pytest.fixture(scope="module")
def registry_mod():
    return safe_import("yirage.hardware.registry")


@pytest.fixture(scope="module")
def builtin_mod():
    return safe_import("yirage.hardware.builtin_chips")


@pytest.fixture()
def fresh_registry(registry_mod):
    """Provide a clean registry for each test that mutates state."""
    if registry_mod is None:
        pytest.skip("registry module not available")
    registry_mod.HardwareRegistry.reset()
    reg = registry_mod.HardwareRegistry.instance()
    yield reg
    registry_mod.HardwareRegistry.reset()


# =============================================================================
# Syntax & Import Tests
# =============================================================================


class TestModuleStructure:
    """Verify the module files exist and have valid syntax."""

    @pytest.mark.parametrize("filename", [
        "__init__.py",
        "chip_arch.py",
        "registry.py",
        "builtin_chips.py",
        "detector.py",
    ])
    def test_file_exists(self, filename):
        path = PYTHON_ROOT / "yirage" / "hardware" / filename
        assert path.exists(), f"Missing {path}"

    @pytest.mark.parametrize("filename", [
        "__init__.py",
        "chip_arch.py",
        "registry.py",
        "builtin_chips.py",
        "detector.py",
    ])
    def test_syntax_valid(self, filename):
        path = PYTHON_ROOT / "yirage" / "hardware" / filename
        if not path.exists():
            pytest.skip(f"{filename} not found")
        with open(path) as f:
            compile(f.read(), str(path), "exec")

    def test_import_hardware(self, hw_module):
        if hw_module is None:
            pytest.skip("hardware module not available (native runtime not built)")

    def test_public_api(self, hw_module):
        if hw_module is None:
            pytest.skip("hardware module not available")
        assert hasattr(hw_module, "HardwareRegistry")
        assert hasattr(hw_module, "ChipArchitecture")
        assert hasattr(hw_module, "detect_current_chip")


# =============================================================================
# ChipArchitecture Tests
# =============================================================================


class TestChipArchitecture:
    """Tests for the ChipArchitecture dataclass."""

    def test_create_minimal(self, chip_arch_mod):
        if chip_arch_mod is None:
            pytest.skip("chip_arch module not available")
        chip = chip_arch_mod.ChipArchitecture(chip_id="test_1", chip_name="Test Chip")
        assert chip.chip_id == "test_1"
        assert chip.chip_name == "Test Chip"

    def test_to_dict_roundtrip(self, chip_arch_mod):
        if chip_arch_mod is None:
            pytest.skip("chip_arch module not available")
        chip = chip_arch_mod.ChipArchitecture(
            chip_id="roundtrip",
            chip_name="Roundtrip Test",
            vendor=chip_arch_mod.ChipVendor.NVIDIA,
            category=chip_arch_mod.ChipCategory.GPU,
            arch_name="TestArch",
            arch_code="sm_999",
            backend="cuda",
            memory=chip_arch_mod.MemorySpec(capacity_gb=80, bandwidth_gbps=3000, memory_type=chip_arch_mod.MemoryType.HBM3),
            compute=chip_arch_mod.ComputeSpec(warp_size=32, num_compute_units=128, peak_tflops_fp16=1000),
            features=chip_arch_mod.FeatureFlags(tensor_cores=True, fp8=True),
        )
        d = chip.to_dict()
        assert isinstance(d, dict)
        assert d["chip_id"] == "roundtrip"
        assert d["vendor"] == "nvidia"
        # JSON serialisable
        json_str = json.dumps(d)
        assert "roundtrip" in json_str

    def test_from_dict(self, chip_arch_mod):
        if chip_arch_mod is None:
            pytest.skip("chip_arch module not available")
        data = {
            "chip_id": "from_dict",
            "chip_name": "From Dict",
            "vendor": "nvidia",
            "category": "gpu",
            "backend": "cuda",
            "memory": {"capacity_gb": 80, "memory_type": "hbm3"},
            "compute": {"warp_size": 32},
            "features": {"tensor_cores": True},
        }
        chip = chip_arch_mod.ChipArchitecture.from_dict(data)
        assert chip.chip_id == "from_dict"
        assert chip.vendor == chip_arch_mod.ChipVendor.NVIDIA

    def test_summary(self, chip_arch_mod):
        if chip_arch_mod is None:
            pytest.skip("chip_arch module not available")
        chip = chip_arch_mod.ChipArchitecture(
            chip_id="s",
            chip_name="Summary Chip",
            compute=chip_arch_mod.ComputeSpec(num_compute_units=80, peak_tflops_fp16=100),
            memory=chip_arch_mod.MemorySpec(capacity_gb=40, memory_type=chip_arch_mod.MemoryType.HBM2),
        )
        s = chip.summary()
        assert "Summary Chip" in s
        assert "80 CUs" in s


# =============================================================================
# HardwareRegistry Tests
# =============================================================================


class TestHardwareRegistry:
    """Tests for the singleton registry."""

    def test_singleton(self, registry_mod):
        if registry_mod is None:
            pytest.skip("registry module not available")
        a = registry_mod.HardwareRegistry.instance()
        b = registry_mod.HardwareRegistry.instance()
        assert a is b

    def test_register_and_get(self, fresh_registry, chip_arch_mod):
        if chip_arch_mod is None:
            pytest.skip("chip_arch module not available")
        chip = chip_arch_mod.ChipArchitecture(chip_id="reg_test", chip_name="Reg Test", backend="cuda")
        assert fresh_registry.register(chip) is True
        got = fresh_registry.get("reg_test")
        assert got is not None
        assert got.chip_name == "Reg Test"

    def test_register_duplicate_returns_false(self, fresh_registry, chip_arch_mod):
        if chip_arch_mod is None:
            pytest.skip("chip_arch module not available")
        chip = chip_arch_mod.ChipArchitecture(chip_id="dup", backend="cuda")
        fresh_registry.register(chip)
        assert fresh_registry.register(chip) is False

    def test_register_overwrite(self, fresh_registry, chip_arch_mod):
        if chip_arch_mod is None:
            pytest.skip("chip_arch module not available")
        chip1 = chip_arch_mod.ChipArchitecture(chip_id="ow", chip_name="V1", backend="cuda")
        chip2 = chip_arch_mod.ChipArchitecture(chip_id="ow", chip_name="V2", backend="cuda")
        fresh_registry.register(chip1)
        assert fresh_registry.register(chip2, overwrite=True) is True
        assert fresh_registry.get("ow").chip_name == "V2"

    def test_unregister(self, fresh_registry, chip_arch_mod):
        if chip_arch_mod is None:
            pytest.skip("chip_arch module not available")
        chip = chip_arch_mod.ChipArchitecture(chip_id="unreg", backend="cuda")
        fresh_registry.register(chip)
        assert fresh_registry.unregister("unreg") is True
        assert fresh_registry.get("unreg") is None
        assert fresh_registry.unregister("unreg") is False

    def test_empty_chip_id_raises(self, fresh_registry, chip_arch_mod):
        if chip_arch_mod is None:
            pytest.skip("chip_arch module not available")
        chip = chip_arch_mod.ChipArchitecture(chip_id="", backend="cuda")
        with pytest.raises(ValueError):
            fresh_registry.register(chip)

    def test_list_all(self, fresh_registry, chip_arch_mod):
        if chip_arch_mod is None:
            pytest.skip("chip_arch module not available")
        fresh_registry.register(chip_arch_mod.ChipArchitecture(chip_id="a", backend="cuda"))
        fresh_registry.register(chip_arch_mod.ChipArchitecture(chip_id="b", backend="rocm"))
        assert len(fresh_registry.list_all()) == 2

    def test_list_by_backend(self, fresh_registry, chip_arch_mod):
        if chip_arch_mod is None:
            pytest.skip("chip_arch module not available")
        fresh_registry.register(chip_arch_mod.ChipArchitecture(chip_id="c1", backend="cuda"))
        fresh_registry.register(chip_arch_mod.ChipArchitecture(chip_id="c2", backend="cuda"))
        fresh_registry.register(chip_arch_mod.ChipArchitecture(chip_id="r1", backend="rocm"))
        assert len(fresh_registry.list_by_backend("cuda")) == 2
        assert len(fresh_registry.list_by_backend("rocm")) == 1

    def test_list_by_vendor(self, fresh_registry, chip_arch_mod):
        if chip_arch_mod is None:
            pytest.skip("chip_arch module not available")
        fresh_registry.register(chip_arch_mod.ChipArchitecture(
            chip_id="nv1", vendor=chip_arch_mod.ChipVendor.NVIDIA, backend="cuda",
        ))
        fresh_registry.register(chip_arch_mod.ChipArchitecture(
            chip_id="nv2", vendor=chip_arch_mod.ChipVendor.NVIDIA, backend="cuda",
        ))
        fresh_registry.register(chip_arch_mod.ChipArchitecture(
            chip_id="amd1", vendor=chip_arch_mod.ChipVendor.AMD, backend="rocm",
        ))
        assert len(fresh_registry.list_by_vendor("nvidia")) == 2

    def test_list_by_vendor_shadow_enum_class(self, fresh_registry, chip_arch_mod):
        """Index by vendor string must use Enum.value, not str(member) (differs on Py<3.11)."""
        if chip_arch_mod is None:
            pytest.skip("chip_arch module not available")
        from enum import Enum

        class ShadowVendor(str, Enum):
            NVIDIA = "nvidia"

        assert not isinstance(ShadowVendor.NVIDIA, chip_arch_mod.ChipVendor)
        chip = chip_arch_mod.ChipArchitecture(
            chip_id="shadow_nv", vendor=ShadowVendor.NVIDIA, backend="cuda"
        )
        assert fresh_registry.register(chip) is True
        assert len(fresh_registry.list_by_vendor("nvidia")) == 1

    def test_contains(self, fresh_registry, chip_arch_mod):
        if chip_arch_mod is None:
            pytest.skip("chip_arch module not available")
        fresh_registry.register(chip_arch_mod.ChipArchitecture(chip_id="x", backend="cuda"))
        assert "x" in fresh_registry
        assert "y" not in fresh_registry

    def test_size(self, fresh_registry, chip_arch_mod):
        if chip_arch_mod is None:
            pytest.skip("chip_arch module not available")
        assert fresh_registry.size == 0
        fresh_registry.register(chip_arch_mod.ChipArchitecture(chip_id="sz", backend="cuda"))
        assert fresh_registry.size == 1

    def test_clear(self, fresh_registry, chip_arch_mod):
        if chip_arch_mod is None:
            pytest.skip("chip_arch module not available")
        fresh_registry.register(chip_arch_mod.ChipArchitecture(chip_id="cl", backend="cuda"))
        fresh_registry.clear()
        assert fresh_registry.size == 0

    def test_callback_on_register(self, fresh_registry, chip_arch_mod):
        if chip_arch_mod is None:
            pytest.skip("chip_arch module not available")
        events = []
        fresh_registry.on_register(lambda c: events.append(c.chip_id))
        fresh_registry.register(chip_arch_mod.ChipArchitecture(chip_id="cb", backend="cuda"))
        assert events == ["cb"]


# =============================================================================
# JSON Import / Export Tests
# =============================================================================


class TestRegistryIO:
    """Test JSON import / export."""

    def test_export_json(self, fresh_registry, chip_arch_mod):
        if chip_arch_mod is None:
            pytest.skip("chip_arch module not available")
        fresh_registry.register(chip_arch_mod.ChipArchitecture(
            chip_id="ex1", chip_name="Export1", backend="cuda",
        ))
        text = fresh_registry.export_json()
        data = json.loads(text)
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["chip_id"] == "ex1"

    def test_export_to_file(self, fresh_registry, chip_arch_mod, tmp_path):
        if chip_arch_mod is None:
            pytest.skip("chip_arch module not available")
        fresh_registry.register(chip_arch_mod.ChipArchitecture(
            chip_id="f1", backend="cuda",
        ))
        out = tmp_path / "chips.json"
        fresh_registry.export_json(str(out))
        assert out.exists()
        data = json.loads(out.read_text())
        assert data[0]["chip_id"] == "f1"

    def test_import_json_string(self, fresh_registry, chip_arch_mod):
        if chip_arch_mod is None:
            pytest.skip("chip_arch module not available")
        payload = json.dumps([
            {"chip_id": "imp1", "chip_name": "Imported 1", "backend": "cuda"},
            {"chip_id": "imp2", "chip_name": "Imported 2", "backend": "rocm"},
        ])
        count = fresh_registry.import_json(payload)
        assert count == 2
        assert fresh_registry.get("imp1") is not None
        assert fresh_registry.get("imp2") is not None

    def test_import_json_file(self, fresh_registry, chip_arch_mod, tmp_path):
        if chip_arch_mod is None:
            pytest.skip("chip_arch module not available")
        f = tmp_path / "in.json"
        f.write_text(json.dumps([{"chip_id": "ff", "backend": "maca"}]))
        count = fresh_registry.import_json(str(f))
        assert count == 1
        assert "ff" in fresh_registry


# =============================================================================
# Built-in Chips Tests
# =============================================================================


class TestBuiltinChips:
    """Verify built-in chip registrations."""

    def test_builtin_count(self, builtin_mod, fresh_registry):
        if builtin_mod is None:
            pytest.skip("builtin module not available")
        n = builtin_mod.register_builtin_chips(fresh_registry)
        assert n >= 15  # we ship at least 15 chips

    def test_known_chips_present(self, builtin_mod, fresh_registry):
        if builtin_mod is None:
            pytest.skip("builtin module not available")
        builtin_mod.register_builtin_chips(fresh_registry)
        expected = [
            "nvidia_h100",
            "nvidia_a100",
            "nvidia_b200",
            "amd_mi300x",
            "ascend_910b",
            "metax_c500",
            "apple_m3_max",
            "google_tpu_v5e",
        ]
        for cid in expected:
            assert cid in fresh_registry, f"{cid} missing from builtin chips"

    def test_nvidia_h100_specs(self, builtin_mod, fresh_registry):
        if builtin_mod is None:
            pytest.skip("builtin module not available")
        builtin_mod.register_builtin_chips(fresh_registry)
        h100 = fresh_registry.get("nvidia_h100")
        assert h100 is not None
        assert h100.compute.warp_size == 32
        assert h100.features.tensor_cores is True
        assert h100.features.tma is True
        assert h100.memory.capacity_gb == 80

    def test_metax_c500_warp_size(self, builtin_mod, fresh_registry):
        if builtin_mod is None:
            pytest.skip("builtin module not available")
        builtin_mod.register_builtin_chips(fresh_registry)
        c500 = fresh_registry.get("metax_c500")
        assert c500 is not None
        assert c500.compute.warp_size == 64

    def test_apple_m3_max_specs(self, builtin_mod, fresh_registry):
        if builtin_mod is None:
            pytest.skip("builtin module not available")
        builtin_mod.register_builtin_chips(fresh_registry)
        m3m = fresh_registry.get("apple_m3_max")
        assert m3m is not None
        assert m3m.vendor.name == "APPLE"
        assert m3m.backend == "mps"
        assert m3m.compute.warp_size == 32
        assert m3m.compute.num_compute_units == 40
        assert m3m.compute.peak_tflops_fp16 == 14.2
        assert m3m.memory.memory_type.name == "UNIFIED"
        assert m3m.memory.capacity_gb == 128
        assert m3m.memory.bandwidth_gbps == 400

    def test_all_chips_have_backend(self, builtin_mod, fresh_registry):
        if builtin_mod is None:
            pytest.skip("builtin module not available")
        builtin_mod.register_builtin_chips(fresh_registry)
        for chip in fresh_registry.list_all():
            assert chip.backend, f"{chip.chip_id} has no backend"

    def test_list_by_backend_mps(self, builtin_mod, fresh_registry):
        if builtin_mod is None:
            pytest.skip("builtin module not available")
        builtin_mod.register_builtin_chips(fresh_registry)
        mps_chips = fresh_registry.list_by_backend("mps")
        assert len(mps_chips) >= 3  # M1/M2/M3/M4/M5 families

    def test_list_by_backend_cuda(self, builtin_mod, fresh_registry):
        if builtin_mod is None:
            pytest.skip("builtin module not available")
        builtin_mod.register_builtin_chips(fresh_registry)
        cuda_chips = fresh_registry.list_by_backend("cuda")
        assert len(cuda_chips) >= 5  # V100, T4, A100, RTX3090, RTX4090, H100, B200


# =============================================================================
# Integration — import from top-level yirage
# =============================================================================


class TestTopLevelIntegration:
    """Test that the hardware module is accessible from yirage."""

    def test_import_from_yirage(self):
        mod = safe_import("yirage")
        if mod is None:
            pytest.skip("yirage not importable")
        assert hasattr(mod, "HardwareRegistry")
        assert hasattr(mod, "ChipArchitecture")

    def test_registry_populated_on_import(self):
        mod = safe_import("yirage.hardware")
        if mod is None:
            pytest.skip("hardware module not available")
        # Re-register builtins (singleton may have been reset by other tests)
        mod.HardwareRegistry.reset()
        reg = mod.HardwareRegistry.instance()
        from yirage.hardware.builtin_chips import register_builtin_chips
        register_builtin_chips(reg)
        assert reg.size > 0, "Built-in chips should be registerable"
