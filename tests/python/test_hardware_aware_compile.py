#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""
Hardware-Aware Compile Pipeline Tests

Tests for the end-to-end hardware → search → compile pipeline:
  - chip_arch_to_search_config() in search_space.py
  - Search config respects chip overrides, derives from compute specs, and falls back
  - hardware_aware_compile() wires detection → search → compile correctly

Run with:  pytest tests/python/test_hardware_aware_compile.py -v
"""

from __future__ import annotations

import sys
import importlib.util
import warnings
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "python"))

# ---------------------------------------------------------------------------
# Load pure-Python modules directly (no torch / C++ dependency)
# ---------------------------------------------------------------------------

def _load_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_HW_ROOT = PROJECT_ROOT / "python" / "yirage" / "hardware"
_COMP_ROOT = PROJECT_ROOT / "python" / "yirage" / "compiler"

chip_arch_mod = _load_file("yirage.hardware.chip_arch", _HW_ROOT / "chip_arch.py")
search_space_mod = _load_file(
    "yirage.compiler.search_space", _COMP_ROOT / "search_space.py"
)

ChipArchitecture = chip_arch_mod.ChipArchitecture
ChipVendor       = chip_arch_mod.ChipVendor
ChipCategory     = chip_arch_mod.ChipCategory
ComputeSpec      = chip_arch_mod.ComputeSpec
FeatureFlags     = chip_arch_mod.FeatureFlags
MemorySpec       = chip_arch_mod.MemorySpec
MemoryType       = chip_arch_mod.MemoryType

chip_arch_to_search_config = search_space_mod.chip_arch_to_search_config
MODE_FAST         = search_space_mod.MODE_FAST
MODE_SUPEROPTIMIZE = search_space_mod.MODE_SUPEROPTIMIZE
MODE_AGGRESSIVE   = search_space_mod.MODE_AGGRESSIVE


# ---------------------------------------------------------------------------
# Fixtures: synthetic chip architectures
# ---------------------------------------------------------------------------

def _make_h100() -> ChipArchitecture:
    return ChipArchitecture(
        chip_id="test_h100", chip_name="Test H100",
        vendor=ChipVendor.NVIDIA, category=ChipCategory.GPU,
        arch_code="sm_90", backend="cuda",
        compute=ComputeSpec(
            warp_size=32, max_threads_per_block=1024,
            shared_mem_per_block_kb=228, num_compute_units=132,
            peak_tflops_fp16=989.0,
        ),
        features=FeatureFlags(tensor_cores=True, fp8=True, tma=True),
    )


def _make_a100() -> ChipArchitecture:
    return ChipArchitecture(
        chip_id="test_a100", chip_name="Test A100",
        vendor=ChipVendor.NVIDIA, category=ChipCategory.GPU,
        arch_code="sm_80", backend="cuda",
        compute=ComputeSpec(
            warp_size=32, max_threads_per_block=1024,
            shared_mem_per_block_kb=96, num_compute_units=108,
            peak_tflops_fp16=312.0,
        ),
    )


def _make_t4() -> ChipArchitecture:
    return ChipArchitecture(
        chip_id="test_t4", chip_name="Test T4",
        vendor=ChipVendor.NVIDIA, category=ChipCategory.GPU,
        arch_code="sm_75", backend="cuda",
        compute=ComputeSpec(
            warp_size=32, max_threads_per_block=1024,
            shared_mem_per_block_kb=64, num_compute_units=40,
            peak_tflops_fp16=65.0,
        ),
    )


def _make_chip_with_overrides() -> ChipArchitecture:
    return ChipArchitecture(
        chip_id="test_custom", chip_name="Custom Chip",
        vendor=ChipVendor.OTHER, category=ChipCategory.GPU, backend="cuda",
        compute=ComputeSpec(
            warp_size=64, max_threads_per_block=512,
            shared_mem_per_block_kb=32, num_compute_units=50,
        ),
        search_config_overrides={
            "griddims":  [(4, 1, 1), (8, 1, 1)],
            "blockdims": [(64, 1, 1)],
            "franges":   [2, 4],
            "fmaps":     [1],
        },
    )


def _make_cpu_chip() -> ChipArchitecture:
    return ChipArchitecture(
        chip_id="test_cpu", chip_name="Generic CPU",
        vendor=ChipVendor.INTEL, category=ChipCategory.CPU, backend="cpu",
        compute=ComputeSpec(
            warp_size=1, max_threads_per_block=64,
            shared_mem_per_block_kb=32, num_compute_units=16,
        ),
    )


def _make_apple_m1() -> ChipArchitecture:
    return ChipArchitecture(
        chip_id="apple_m1", chip_name="Apple M1",
        vendor=ChipVendor.APPLE, category=ChipCategory.GPU,
        arch_code="apple_g13g", backend="mps",
        compute=ComputeSpec(
            warp_size=32, max_threads_per_block=1024,
            shared_mem_per_block_kb=32, num_compute_units=8,
            peak_tflops_fp16=2.6,
        ),
        memory=MemorySpec(capacity_gb=16, bandwidth_gbps=68.25,
                          memory_type=MemoryType.UNIFIED),
    )


def _make_apple_m3_max() -> ChipArchitecture:
    return ChipArchitecture(
        chip_id="apple_m3_max", chip_name="Apple M3 Max",
        vendor=ChipVendor.APPLE, category=ChipCategory.GPU,
        arch_code="apple_g15p", backend="mps",
        compute=ComputeSpec(
            warp_size=32, max_threads_per_block=1024,
            shared_mem_per_block_kb=32, num_compute_units=40,
            peak_tflops_fp16=14.2,
        ),
        memory=MemorySpec(capacity_gb=128, bandwidth_gbps=400,
                          memory_type=MemoryType.UNIFIED),
    )


def _make_apple_m4_max() -> ChipArchitecture:
    return ChipArchitecture(
        chip_id="apple_m4_max", chip_name="Apple M4 Max",
        vendor=ChipVendor.APPLE, category=ChipCategory.GPU,
        arch_code="apple_g16p", backend="mps",
        compute=ComputeSpec(
            warp_size=32, max_threads_per_block=1024,
            shared_mem_per_block_kb=32, num_compute_units=40,
            peak_tflops_fp16=18.0,
        ),
        memory=MemorySpec(capacity_gb=128, bandwidth_gbps=546,
                          memory_type=MemoryType.UNIFIED),
    )


# ===========================================================================
# Tests: no-chip fallback behaviour
# ===========================================================================

class TestSearchConfigNoChip:
    def test_superoptimize_has_required_keys(self):
        cfg = chip_arch_to_search_config(None, MODE_SUPEROPTIMIZE)
        assert "griddims" in cfg
        assert "blockdims" in cfg

    def test_fast_has_single_grid(self):
        cfg = chip_arch_to_search_config(None, MODE_FAST)
        assert cfg["griddims"] == [(1, 1, 1)]

    def test_aggressive_has_more_blockdims_than_superoptimize(self):
        cfg_agg = chip_arch_to_search_config(None, MODE_AGGRESSIVE)
        cfg_sup = chip_arch_to_search_config(None, MODE_SUPEROPTIMIZE)
        assert len(cfg_agg["blockdims"]) >= len(cfg_sup["blockdims"])

    def test_all_griddims_are_3_tuples(self):
        for mode in [MODE_FAST, MODE_SUPEROPTIMIZE, MODE_AGGRESSIVE]:
            for gd in chip_arch_to_search_config(None, mode)["griddims"]:
                assert len(gd) == 3

    def test_all_blockdims_are_3_tuples(self):
        for mode in [MODE_FAST, MODE_SUPEROPTIMIZE, MODE_AGGRESSIVE]:
            for bd in chip_arch_to_search_config(None, mode)["blockdims"]:
                assert len(bd) == 3

    def test_fast_no_franges(self):
        cfg = chip_arch_to_search_config(None, MODE_FAST)
        assert "franges" not in cfg


# ===========================================================================
# Tests: hardware-aware derivation
# ===========================================================================

class TestSearchConfigWithChip:
    def test_blockdims_respect_max_threads(self):
        for chip in [_make_h100(), _make_t4()]:
            cfg = chip_arch_to_search_config(chip, MODE_SUPEROPTIMIZE)
            max_t = chip.compute.max_threads_per_block
            for bx, _, _ in cfg["blockdims"]:
                assert bx <= max_t, (
                    f"blockdim {bx} > max_threads_per_block {max_t} for {chip.chip_id}"
                )

    def test_blockdims_are_warp_multiples(self):
        chip = _make_h100()
        cfg = chip_arch_to_search_config(chip, MODE_SUPEROPTIMIZE)
        warp = chip.compute.warp_size
        for bx, _, _ in cfg["blockdims"]:
            assert bx % warp == 0

    def test_griddims_scale_with_sm_count(self):
        """H100 (132 SMs) must have at least as large a max grid as T4 (40 SMs)."""
        cfg_h100 = chip_arch_to_search_config(_make_h100(), MODE_SUPEROPTIMIZE)
        cfg_t4   = chip_arch_to_search_config(_make_t4(),   MODE_SUPEROPTIMIZE)
        assert max(gd[0] for gd in cfg_h100["griddims"]) >= max(
            gd[0] for gd in cfg_t4["griddims"]
        )

    def test_larger_smem_gives_larger_franges(self):
        """H100 (228 KB smem) explores larger forloop ranges than T4 (64 KB)."""
        cfg_h100 = chip_arch_to_search_config(_make_h100(), MODE_SUPEROPTIMIZE)
        cfg_t4   = chip_arch_to_search_config(_make_t4(),   MODE_SUPEROPTIMIZE)
        assert max(cfg_h100["franges"]) >= max(cfg_t4["franges"])

    def test_chip_overrides_take_priority(self):
        chip = _make_chip_with_overrides()
        cfg = chip_arch_to_search_config(chip, MODE_SUPEROPTIMIZE)
        assert cfg["griddims"]  == [(4, 1, 1), (8, 1, 1)]
        assert cfg["blockdims"] == [(64, 1, 1)]
        assert cfg["franges"]   == [2, 4]
        assert cfg["fmaps"]     == [1]

    def test_chip_overrides_respected_in_fast_mode(self):
        chip = _make_chip_with_overrides()
        cfg = chip_arch_to_search_config(chip, MODE_FAST)
        assert cfg["griddims"]  == [(4, 1, 1), (8, 1, 1)]
        assert cfg["blockdims"] == [(64, 1, 1)]

    def test_chip_overrides_respected_in_aggressive_mode(self):
        chip = _make_chip_with_overrides()
        cfg = chip_arch_to_search_config(chip, MODE_AGGRESSIVE)
        assert cfg["griddims"]  == [(4, 1, 1), (8, 1, 1)]
        assert cfg["blockdims"] == [(64, 1, 1)]

    def test_partial_override_fills_rest_from_hardware(self):
        chip = ChipArchitecture(
            chip_id="partial", chip_name="Partial Override", backend="cuda",
            compute=ComputeSpec(
                warp_size=32, max_threads_per_block=1024,
                shared_mem_per_block_kb=96, num_compute_units=80,
            ),
            search_config_overrides={"franges": [8, 64]},
        )
        cfg = chip_arch_to_search_config(chip, MODE_SUPEROPTIMIZE)
        assert cfg["franges"] == [8, 64]
        assert len(cfg["blockdims"]) > 0

    def test_cpu_chip_blockdims_within_limit(self):
        chip = _make_cpu_chip()
        cfg = chip_arch_to_search_config(chip, MODE_SUPEROPTIMIZE)
        max_t = chip.compute.max_threads_per_block
        for bx, _, _ in cfg["blockdims"]:
            assert bx <= max_t


# ===========================================================================
# Tests: edge cases
# ===========================================================================

class TestEdgeCases:
    def test_unknown_mode_returns_non_empty_config(self):
        cfg = chip_arch_to_search_config(None, "UNKNOWN_MODE")
        assert "griddims" in cfg
        assert len(cfg["griddims"]) > 0

    def test_chip_with_zero_sms_falls_back_gracefully(self):
        chip = ChipArchitecture(
            chip_id="no_sms", backend="cuda",
            compute=ComputeSpec(num_compute_units=0, max_threads_per_block=1024, warp_size=32),
        )
        cfg = chip_arch_to_search_config(chip, MODE_SUPEROPTIMIZE)
        assert len(cfg["griddims"]) > 0

    def test_chip_with_zero_smem_falls_back_gracefully(self):
        chip = ChipArchitecture(
            chip_id="no_smem", backend="cuda",
            compute=ComputeSpec(
                num_compute_units=80, max_threads_per_block=1024,
                warp_size=32, shared_mem_per_block_kb=0,
            ),
        )
        cfg = chip_arch_to_search_config(chip, MODE_SUPEROPTIMIZE)
        assert len(cfg["franges"]) > 0

    def test_empty_overrides_uses_hardware_derivation(self):
        chip = ChipArchitecture(
            chip_id="empty_ov", backend="cuda",
            compute=ComputeSpec(
                warp_size=32, max_threads_per_block=1024,
                shared_mem_per_block_kb=96, num_compute_units=108,
            ),
            search_config_overrides={},
        )
        cfg = chip_arch_to_search_config(chip, MODE_SUPEROPTIMIZE)
        assert len(cfg["griddims"]) > 0
        assert len(cfg["blockdims"]) > 0

    def test_none_chip_aggressive_has_franges(self):
        cfg = chip_arch_to_search_config(None, MODE_AGGRESSIVE)
        assert "franges" in cfg
        assert len(cfg["franges"]) > 0


# ===========================================================================
# Tests: parametric — all chip families × all modes
# ===========================================================================

class TestBuiltinChipFamilies:
    @pytest.mark.parametrize(
        "chip_factory",
        [_make_h100, _make_a100, _make_t4, _make_chip_with_overrides, _make_cpu_chip,
         _make_apple_m1, _make_apple_m3_max, _make_apple_m4_max],
        ids=["h100", "a100", "t4", "custom_overrides", "cpu",
             "apple_m1", "apple_m3_max", "apple_m4_max"],
    )
    @pytest.mark.parametrize("mode", [MODE_FAST, MODE_SUPEROPTIMIZE, MODE_AGGRESSIVE])
    def test_config_non_empty(self, chip_factory, mode):
        cfg = chip_arch_to_search_config(chip_factory(), mode)
        assert len(cfg["griddims"]) > 0
        assert len(cfg["blockdims"]) > 0

    @pytest.mark.parametrize(
        "chip_factory",
        [_make_h100, _make_a100, _make_t4, _make_chip_with_overrides, _make_cpu_chip,
         _make_apple_m1, _make_apple_m3_max, _make_apple_m4_max],
        ids=["h100", "a100", "t4", "custom_overrides", "cpu",
             "apple_m1", "apple_m3_max", "apple_m4_max"],
    )
    @pytest.mark.parametrize("mode", [MODE_FAST, MODE_SUPEROPTIMIZE, MODE_AGGRESSIVE])
    def test_all_dims_positive(self, chip_factory, mode):
        cfg = chip_arch_to_search_config(chip_factory(), mode)
        for gd in cfg["griddims"]:
            assert all(d >= 1 for d in gd)
        for bd in cfg["blockdims"]:
            assert all(d >= 1 for d in bd)


# ===========================================================================
# Integration: hardware_aware_compile()
# Loaded with proper package context so relative imports work
# ===========================================================================

class TestHardwareAwareCompileFunction:
    @pytest.fixture(autouse=True)
    def _load_hac(self):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                from yirage.compiler.unified import hardware_aware_compile, CompileMode
            self.hardware_aware_compile = hardware_aware_compile
            self.CompileMode = CompileMode
        except (ImportError, ModuleNotFoundError) as exc:
            pytest.skip(f"Cannot import yirage.compiler.unified: {exc}")

    def test_function_is_callable(self):
        assert callable(self.hardware_aware_compile)

    def test_accepts_dict_graph(self):
        chip = _make_cpu_chip()
        result = self.hardware_aware_compile(
            {"type": "test_graph"}, chip_arch=chip,
            mode=self.CompileMode.FAST,
        )
        assert hasattr(result, "success")

    def test_accepts_string_graph(self, tmp_path):
        graph_file = tmp_path / "graph.json"
        # Same payload as test_accepts_dict_graph; trivial JSON can abort in cy_from_json.
        graph_file.write_text('{"type": "test_graph"}')
        chip = _make_cpu_chip()
        result = self.hardware_aware_compile(
            str(graph_file), chip_arch=chip, mode=self.CompileMode.FAST,
        )
        assert hasattr(result, "success")

    def test_verbose_prints_pipeline_header(self, capsys):
        chip = _make_h100()
        self.hardware_aware_compile(
            {"type": "test"}, chip_arch=chip,
            mode=self.CompileMode.FAST, verbose=True,
        )
        out = capsys.readouterr().out
        assert "hardware_aware_compile" in out

    def test_chip_name_appears_in_verbose_output(self, capsys):
        chip = _make_h100()
        self.hardware_aware_compile(
            {"type": "test"}, chip_arch=chip,
            mode=self.CompileMode.FAST, verbose=True,
        )
        out = capsys.readouterr().out
        assert "Test H100" in out

    def test_fallback_message_when_no_chip(self, capsys):
        self.hardware_aware_compile(
            {"type": "test"}, chip_arch=None,
            mode=self.CompileMode.FAST, verbose=True,
        )
        out = capsys.readouterr().out
        lower = out.lower()
        # When chip_arch=None, the system either falls back to a generic CPU
        # path or auto-detects the local hardware.  On Apple Silicon macOS
        # machines, detect_current_chip() will find an M-series chip via MPS,
        # so the output reports the detected chip rather than a fallback.
        if sys.platform == "darwin":
            assert (
                "fallback" in lower
                or "no chip" in out
                or "detected chip" in lower
                or "apple_m" in lower  # M-series auto-detect (M1-M5)
            )
        else:
            assert "fallback" in lower or "No chip" in out
