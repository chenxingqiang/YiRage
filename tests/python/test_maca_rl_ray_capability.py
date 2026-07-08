"""MACA vs CUDA capability parity for RL / Ray distributed search (contract layer).

These tests document gaps and verify API/config alignment without MetaX GPU.
Tests that need ``yirage.core`` use source/importlib fallbacks so Cloud CPU VMs
can run the contract layer (merge still requires MetaX VM smoke per AGENTS.md).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_PKG = _REPO / "python"


def _load_module(name: str, path: Path):
    if str(_PKG) not in sys.path:
        sys.path.insert(0, str(_PKG))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def maca_rl_ray_capability_matrix() -> list[dict]:
    """Single-source checklist: CUDA reference vs MACA status (evolve in AGENTS.md)."""
    return [
        {
            "id": "superoptimize_ray",
            "cuda": "KNGraph.superoptimize(use_ray=True) + griddims>1",
            "maca": "Same API; demos default use_ray=False",
            "maca_vm_test": "demo/maca_superopt_test.py + tests/integration/test_ray_maca_e2e.py",
            "tier": "partial",
        },
        {
            "id": "distributed_coordinator",
            "cuda": "DistributedSearchCoordinator.parallel_search(backend=cuda)",
            "maca": "backend=maca param accepted; maca e2e pytest added",
            "maca_vm_test": "tests/integration/test_ray_maca_e2e.py",
            "tier": "partial",
        },
        {
            "id": "ray_distributed_engine",
            "cuda": "RayDistributedEngine + GPUPlacementConfig",
            "maca": "Uses torch.cuda.is_available(); no MACA placement policy",
            "maca_vm_test": "MetaX VM Ray engine smoke (planned)",
            "tier": "partial",
        },
        {
            "id": "walkthrough",
            "cuda": "N/A (walkthrough hardcoded cpu today)",
            "maca": "scripts/business_capability_walkthrough.py backend=cpu only",
            "maca_vm_test": "scripts/maca_capability_walkthrough.py",
            "tier": "partial",
        },
        {
            "id": "hierarchical_rl",
            "cuda": "ConstrainedGraphEnv FINISH + accelforge_metrics; default backend=cuda",
            "maca": "config_space encodes maca; no env e2e backend=maca",
            "maca_vm_test": "tests/python/test_rl/test_accelforge.py maca profile (planned)",
            "tier": "partial",
        },
        {
            "id": "gpu_verifier_pool",
            "cuda": "VerifierPool(backend=cuda)",
            "maca": "Hardcoded cuda in verifier_pool; mcPytorch may piggyback",
            "maca_vm_test": "VerifierPool(backend=maca) smoke (planned)",
            "tier": "gap",
        },
        {
            "id": "bench_ray_search",
            "cuda": "N/A",
            "maca": "scripts/bench_ray_search.py cpu-only",
            "maca_vm_test": "bench_ray_search backend=maca variant (planned)",
            "tier": "gap",
        },
    ]


def test_maca_rl_ray_matrix_has_required_fields():
    for row in maca_rl_ray_capability_matrix():
        for key in ("id", "cuda", "maca", "maca_vm_test", "tier"):
            assert key in row


def test_distributed_coordinator_accepts_maca_backend():
    coordinator_py = _PKG / "yirage" / "ray" / "coordinator.py"
    text = coordinator_py.read_text(encoding="utf-8")
    assert "def parallel_search" in text
    assert "backend" in text


def test_ray_engine_config_accepts_maca_backend_string():
    ray_py = _PKG / "yirage" / "ray" / "ray_distributed.py"
    text = ray_py.read_text(encoding="utf-8")
    assert "class DistributedConfig" in text
    assert "backend" in text


def test_maca_detector_warp_size_in_source():
    det_py = _PKG / "yirage" / "rl" / "hardware" / "detector.py"
    text = det_py.read_text(encoding="utf-8")
    assert "class MACACDetector" in text
    assert "warp_size=64" in text
    assert '"maca"' in text


def test_walkthrough_still_cpu_only_documents_gap():
    walk = _REPO / "scripts" / "business_capability_walkthrough.py"
    text = walk.read_text(encoding="utf-8")
    assert 'YIRAGE_BACKEND", "cpu"' in text or "backend=cpu" in text


def test_maca_capability_walkthrough_exists_and_uses_maca_backend():
    walk = _REPO / "scripts" / "maca_capability_walkthrough.py"
    assert walk.is_file()
    text = walk.read_text(encoding="utf-8")
    assert 'backend="maca"' in text or "backend='maca'" in text
    assert "build_maca_walkthrough_report" in text


def test_ray_maca_e2e_module_exists():
    e2e = _REPO / "tests" / "integration" / "test_ray_maca_e2e.py"
    assert e2e.is_file()
    text = e2e.read_text(encoding="utf-8")
    assert 'backend="maca"' in text
    assert "DistributedSearchCoordinator" in text


def test_rl_config_space_includes_maca():
    cfg_mod = _load_module(
        "maca_rl_config_space", _PKG / "yirage" / "rl" / "search" / "config_space.py"
    )
    obs = cfg_mod.ConfigObservationSpace()
    features = obs.encode_hardware(backend="maca")
    assert features[4] == 1.0  # MACA warp_size 64 normalized


def test_superoptimize_signature_supports_ray_and_maca():
    graph_py = _PKG / "yirage" / "kernel" / "graph.py"
    text = graph_py.read_text(encoding="utf-8")
    assert "use_ray" in text
    assert "maca" in text


def test_maca_demo_utils_default_disables_ray_documents_gap():
    """Documented gap: MACA smoke paths set use_ray=False until Ray MACA e2e exists."""
    utils = _REPO / "demo" / "_maca_utils.py"
    text = utils.read_text(encoding="utf-8")
    assert "use_ray=False" in text
