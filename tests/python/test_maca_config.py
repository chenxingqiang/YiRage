"""Tests for MACA config (loads config module without yirage.core)."""

import importlib.util
import sys
from pathlib import Path


def _load_maca_backend_config():
    root = Path(__file__).resolve().parents[2]
    pkg_root = root / "python"
    if str(pkg_root) not in sys.path:
        sys.path.insert(0, str(pkg_root))
    path = pkg_root / "yirage" / "backends" / "maca" / "config.py"
    spec = importlib.util.spec_from_file_location("maca_backend_config", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_maca_config_shim_file_reexports_backend():
    shim = Path(__file__).resolve().parents[2] / "python" / "yirage" / "maca_config.py"
    text = shim.read_text(encoding="utf-8")
    assert "from yirage.backends.maca.config import" in text


def test_maca_search_config_quick_smaller_than_full():
    cfg = _load_maca_backend_config()
    full = cfg.get_maca_search_config()
    quick = cfg.get_maca_search_config_quick()
    assert cfg.MACA_WARP_SIZE == 64
    assert len(full["grid_dims_to_explore"]) > len(quick["grid_dims_to_explore"])
    assert len(full["block_dims_to_explore"]) > len(quick["block_dims_to_explore"])
    assert cfg.resolve_maca_search_config(quick=True) == quick
    assert cfg.resolve_maca_search_config(quick=False) == full


def test_resolve_maca_search_config_env(monkeypatch):
    cfg = _load_maca_backend_config()
    monkeypatch.setenv("YIRAGE_MACA_SEARCH_QUICK", "0")
    assert cfg.resolve_maca_search_config() == cfg.get_maca_search_config()
    monkeypatch.setenv("YIRAGE_MACA_SEARCH_QUICK", "1")
    assert cfg.resolve_maca_search_config() == cfg.get_maca_search_config_quick()


def test_maca_shared_memory_capacity_matches_device_limit(monkeypatch):
    """Transpiler smem gate must use 64 KB on MACA, not Volta 96 KB."""
    root = Path(__file__).resolve().parents[2]
    pkg_root = root / "python"
    path = pkg_root / "yirage" / "utils" / "common.py"
    spec = importlib.util.spec_from_file_location("yirage_utils_common", path)
    common = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(common)

    monkeypatch.setenv("YIRAGE_BACKEND", "maca")
    assert common.get_shared_memory_capacity(70) == 65536
    assert common.get_shared_memory_capacity(80) == 65536


def test_resolve_maca_use_ray_env(monkeypatch):
    cfg = _load_maca_backend_config()
    monkeypatch.delenv("YIRAGE_MACA_USE_RAY", raising=False)
    assert cfg.resolve_maca_use_ray() is False
    assert cfg.maca_superoptimize_ray_kwargs() == {"use_ray": False}
    monkeypatch.setenv("YIRAGE_MACA_USE_RAY", "1")
    assert cfg.resolve_maca_use_ray() is True
    assert cfg.maca_superoptimize_ray_kwargs() == {"use_ray": True}


def test_resolve_maca_gpus_per_worker_env(monkeypatch):
    cfg = _load_maca_backend_config()
    monkeypatch.delenv("YIRAGE_MACA_INTEGRATION", raising=False)
    monkeypatch.delenv("YIRAGE_MACA_ALLOW_NON_METAX", raising=False)
    monkeypatch.delenv("MACA_PATH", raising=False)
    monkeypatch.delenv("MACA_HOME", raising=False)
    assert cfg.resolve_maca_gpus_per_worker() == 0.0
    assert cfg.maca_ray_gpu_placement_kwargs()["gpus_per_worker"] == 0.0
    monkeypatch.setenv("YIRAGE_MACA_INTEGRATION", "1")
    assert cfg.resolve_maca_gpus_per_worker(requested=1.0) == 1.0
    assert cfg.maca_ray_gpu_placement_kwargs(gpus_per_worker=1.0)["gpus_per_worker"] == 1.0
    monkeypatch.delenv("YIRAGE_MACA_INTEGRATION", raising=False)
    assert cfg.resolve_maca_gpus_per_worker(requested=0.0) == 0.0


def test_config_h_maca_smem_64kb():
    """C++ transpiler must use maca::MAX_SMEM_SIZE = 64 KB (C500 per-block limit)."""
    config_h = Path(__file__).resolve().parents[2] / "include" / "config.h"
    text = config_h.read_text(encoding="utf-8")
    assert "namespace maca {" in text
    assert "MAX_SMEM_SIZE = 64 * 1024" in text
    assert "64 KB (C500 per-block limit)" in text


def test_mxcc_cmd_uses_software_mma_not_hardware_ptx():
    """mxcc must not pass CUTE_ARCH_MMA_SM70_ENABLED (mma.sync invalid on xcore1000)."""
    graph_py = Path(__file__).resolve().parents[2] / "python" / "yirage" / "kernel" / "graph.py"
    text = graph_py.read_text(encoding="utf-8")
    assert "YIRAGE_MACA_SOFTWARE_MMA=1" in text
    assert "CUTE_ARCH_MMA_SM70_ENABLED" not in text
    assert "CUTE_ARCH_LDSM_SM75_ACTIVATED" not in text
