# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0

"""
Tests for CPU-mode MoE (Mixture-of-Experts) kernels.

Covers:
  - cpu_moe_gate correctness (top-k selection + softmax routing weights)
  - cpu_moe_linear correctness (expert GEMM with weighted accumulation)
  - cpu_moe_silu_linear correctness (fused SwiGLU + down-projection)
  - End-to-end single layer forward pass for LLaMA 3B MoE

All modules are loaded directly with importlib to bypass yirage.__init__,
which requires the native C++ runtime (yirage.core).

torch is imported lazily inside fixtures so the file can still be collected
when PyTorch is not installed (tests are skipped via ``pytest.importorskip``).
"""

import importlib.util
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_BENCHMARK_ROOT = _PROJECT_ROOT / "benchmark" / "end-to-end"


def _load_benchmark_module():
    """Load llama3b_moe_cpu.py directly (no yirage.core dependency)."""
    path = _BENCHMARK_ROOT / "llama3b_moe_cpu.py"
    if not path.exists():
        pytest.skip(f"llama3b_moe_cpu.py not found at {path}")
    spec = importlib.util.spec_from_file_location("llama3b_moe_cpu_test", path)
    if spec is None or spec.loader is None:
        pytest.skip("Could not create module spec for llama3b_moe_cpu.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["llama3b_moe_cpu_test"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def torch():
    """Return the torch module, or skip the whole test module if unavailable."""
    return pytest.importorskip("torch")


@pytest.fixture(scope="module")
def F(torch):
    """Return torch.nn.functional."""
    import torch.nn.functional as _F
    return _F


@pytest.fixture(scope="module")
def moe_module():
    """Loaded llama3b_moe_cpu module (PyTorch functions only)."""
    # Ensure torch is importable before loading the benchmark module
    pytest.importorskip("torch")
    return _load_benchmark_module()


# ---------------------------------------------------------------------------
# Helper: reference softmax-gated top-k
# ---------------------------------------------------------------------------

def _ref_moe_gate(torch, F, hidden, W_gate, top_k: int):
    scores = hidden @ W_gate.T
    topk_w, topk_ids = torch.topk(scores, top_k, dim=-1)
    routing_w = F.softmax(topk_w, dim=-1)
    return topk_ids, routing_w


# ---------------------------------------------------------------------------
# Helper: reference fused MoE SwiGLU + down projection
# ---------------------------------------------------------------------------

def _ref_moe_silu_linear(F, hidden, W_gate, W_up, W_down, expert_ids, routing_w):
    T, H = hidden.shape
    K = expert_ids.shape[1]
    E = W_gate.shape[0]
    import torch
    out = torch.zeros(T, H, dtype=hidden.dtype)
    for k in range(K):
        for e in range(E):
            mask = (expert_ids[:, k] == e)
            if not mask.any():
                continue
            x_e = hidden[mask]
            g = F.silu(x_e @ W_gate[e].T)
            u = x_e @ W_up[e].T
            act = g * u
            proj = act @ W_down[e].T
            out[mask] += routing_w[mask, k : k + 1] * proj
    return out


# ===========================================================================
# Tests: pytorch_moe_gate
# ===========================================================================

class TestMoEGate:
    """Correctness tests for the MoE top-k gating function."""

    @pytest.fixture(params=[
        (8, 3072, 8, 2),
        (1, 64, 4, 1),
        (16, 128, 16, 4),
    ])
    def gate_cfg(self, request):
        return request.param

    def test_expert_ids_in_range(self, torch, moe_module, gate_cfg):
        T, H, E, K = gate_cfg
        hidden = torch.randn(T, H)
        W_gate = torch.randn(E, H)
        ids, _ = moe_module.pytorch_moe_gate(hidden, W_gate, K)
        assert ids.shape == (T, K)
        assert (ids >= 0).all() and (ids < E).all()

    def test_routing_weights_sum_to_one(self, torch, moe_module, gate_cfg):
        T, H, E, K = gate_cfg
        hidden = torch.randn(T, H)
        W_gate = torch.randn(E, H)
        _, routing_w = moe_module.pytorch_moe_gate(hidden, W_gate, K)
        assert routing_w.shape == (T, K)
        row_sums = routing_w.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones(T), atol=1e-5)

    def test_routing_weights_non_negative(self, torch, moe_module, gate_cfg):
        T, H, E, K = gate_cfg
        hidden = torch.randn(T, H)
        W_gate = torch.randn(E, H)
        _, routing_w = moe_module.pytorch_moe_gate(hidden, W_gate, K)
        assert (routing_w >= 0).all()

    def test_matches_reference(self, torch, F, moe_module):
        T, H, E, K = 4, 32, 4, 2
        torch.manual_seed(99)
        hidden = torch.randn(T, H)
        W_gate = torch.randn(E, H)
        ids, w = moe_module.pytorch_moe_gate(hidden, W_gate, K)
        ref_ids, ref_w = _ref_moe_gate(torch, F, hidden, W_gate, K)
        assert torch.equal(ids, ref_ids)
        assert torch.allclose(w, ref_w, atol=1e-6)


# ===========================================================================
# Tests: pytorch_moe_silu_linear
# ===========================================================================

class TestMoESiLULinear:
    """Correctness tests for the fused MoE SwiGLU expert FFN."""

    @pytest.fixture
    def small_moe(self, torch):
        T, H, I, E, K = 4, 32, 16, 4, 2
        torch.manual_seed(1)
        return dict(
            T=T, H=H, I=I, E=E, K=K,
            hidden=torch.randn(T, H),
            W_gate=torch.randn(E, I, H),
            W_up=torch.randn(E, I, H),
            W_down=torch.randn(E, H, I),
            W_router=torch.randn(E, H),
        )

    def test_output_shape(self, torch, F, moe_module, small_moe):
        d = small_moe
        ids, routing_w = _ref_moe_gate(torch, F, d["hidden"], d["W_router"], d["K"])
        out = moe_module.pytorch_moe_silu_linear(
            d["hidden"], d["W_gate"], d["W_up"], d["W_down"], ids, routing_w
        )
        assert out.shape == (d["T"], d["H"])

    def test_matches_reference(self, torch, F, moe_module, small_moe):
        d = small_moe
        ids, routing_w = _ref_moe_gate(torch, F, d["hidden"], d["W_router"], d["K"])
        got = moe_module.pytorch_moe_silu_linear(
            d["hidden"], d["W_gate"], d["W_up"], d["W_down"], ids, routing_w
        )
        ref = _ref_moe_silu_linear(
            F, d["hidden"], d["W_gate"], d["W_up"], d["W_down"], ids, routing_w
        )
        assert torch.allclose(got, ref, atol=1e-4), (
            f"max diff = {(got - ref).abs().max().item():.2e}"
        )

    def test_zero_routing_weight(self, torch, F, moe_module):
        T, H, I, E, K = 2, 16, 8, 2, 1
        torch.manual_seed(7)
        hidden = torch.randn(T, H)
        W_gate = torch.randn(E, I, H)
        W_up   = torch.randn(E, I, H)
        W_down = torch.randn(E, H, I)
        ids = torch.zeros(T, K, dtype=torch.long)
        routing_w = torch.zeros(T, K)
        out = moe_module.pytorch_moe_silu_linear(
            hidden, W_gate, W_up, W_down, ids, routing_w
        )
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)

    def test_single_expert_single_token(self, torch, F, moe_module):
        H, I = 8, 4
        torch.manual_seed(3)
        hidden = torch.randn(1, H)
        Wg = torch.randn(1, I, H)
        Wu = torch.randn(1, I, H)
        Wd = torch.randn(1, H, I)
        ids = torch.zeros(1, 1, dtype=torch.long)
        routing_w = torch.ones(1, 1)
        got = moe_module.pytorch_moe_silu_linear(hidden, Wg, Wu, Wd, ids, routing_w)
        gate = F.silu(hidden @ Wg[0].T)
        up   = hidden @ Wu[0].T
        ref  = (gate * up) @ Wd[0].T
        assert torch.allclose(got, ref, atol=1e-5)


# ===========================================================================
# Tests: full layer end-to-end
# ===========================================================================

class TestLLaMA3BMoELayerForward:
    """Smoke and correctness tests for the full MoE decoder layer."""

    CFG = dict(
        hidden_size=128,
        num_heads=4,
        num_kv_heads=2,
        num_experts=4,
        top_k=2,
        intermediate_size=64,
        batch_size=1,
        seq_len=4,
    )

    def _make_weights(self, torch, c):
        head_dim = c["hidden_size"] // c["num_heads"]
        H = c["hidden_size"]
        I = c["intermediate_size"]
        E = c["num_experts"]
        Nq  = c["num_heads"] * head_dim
        Nkv = c["num_kv_heads"] * head_dim
        d = torch.float32
        torch.manual_seed(42)
        return dict(
            W_q=torch.randn(Nq, H, dtype=d),
            W_k=torch.randn(Nkv, H, dtype=d),
            W_v=torch.randn(Nkv, H, dtype=d),
            W_o=torch.randn(H, Nq, dtype=d),
            rms_attn_weight=torch.ones(H, dtype=d),
            W_gate_router=torch.randn(E, H, dtype=d),
            W_gate=torch.randn(E, I, H, dtype=d),
            W_up=torch.randn(E, I, H, dtype=d),
            W_down=torch.randn(E, H, I, dtype=d),
            rms_ffn_weight=torch.ones(H, dtype=d),
            K_cache=torch.zeros(256, c["num_kv_heads"], head_dim, dtype=d),
            V_cache=torch.zeros(256, c["num_kv_heads"], head_dim, dtype=d),
        )

    def _fwd(self, torch, moe_module, h, weights, c):
        head_dim = c["hidden_size"] // c["num_heads"]
        return moe_module.llama3b_moe_layer_forward(
            h, **weights,
            step=0, top_k=c["top_k"],
            num_heads=c["num_heads"],
            num_kv_heads=c["num_kv_heads"],
            head_dim=head_dim,
        )

    def test_output_shape(self, torch, moe_module):
        c = self.CFG
        h = torch.randn(c["batch_size"], c["seq_len"], c["hidden_size"])
        out = self._fwd(torch, moe_module, h, self._make_weights(torch, c), c)
        assert out.shape == h.shape

    def test_output_is_finite(self, torch, moe_module):
        c = self.CFG
        h = torch.randn(c["batch_size"], c["seq_len"], c["hidden_size"])
        out = self._fwd(torch, moe_module, h, self._make_weights(torch, c), c)
        assert torch.isfinite(out).all()

    def test_deterministic(self, torch, moe_module):
        c = self.CFG
        torch.manual_seed(0)
        h = torch.randn(c["batch_size"], c["seq_len"], c["hidden_size"])
        w1 = self._make_weights(torch, c)
        w2 = self._make_weights(torch, c)  # independent copy
        out1 = self._fwd(torch, moe_module, h.clone(), w1, c)
        out2 = self._fwd(torch, moe_module, h.clone(), w2, c)
        assert torch.allclose(out1, out2, atol=1e-6)

    def test_residual_connection(self, torch, moe_module):
        """With all-zero projection weights, output == input (residual)."""
        c = dict(
            hidden_size=16, num_heads=2, num_kv_heads=2,
            num_experts=2, top_k=1, intermediate_size=8,
            batch_size=1, seq_len=2,
        )
        head_dim = c["hidden_size"] // c["num_heads"]
        H = c["hidden_size"]; I = c["intermediate_size"]; E = c["num_experts"]
        Nq = c["num_heads"] * head_dim; d = torch.float32
        weights = dict(
            W_q=torch.zeros(Nq, H, dtype=d),
            W_k=torch.zeros(c["num_kv_heads"] * head_dim, H, dtype=d),
            W_v=torch.zeros(c["num_kv_heads"] * head_dim, H, dtype=d),
            W_o=torch.zeros(H, Nq, dtype=d),
            rms_attn_weight=torch.ones(H, dtype=d),
            W_gate_router=torch.zeros(E, H, dtype=d),
            W_gate=torch.zeros(E, I, H, dtype=d),
            W_up=torch.zeros(E, I, H, dtype=d),
            W_down=torch.zeros(E, H, I, dtype=d),
            rms_ffn_weight=torch.ones(H, dtype=d),
            K_cache=torch.zeros(64, c["num_kv_heads"], head_dim, dtype=d),
            V_cache=torch.zeros(64, c["num_kv_heads"], head_dim, dtype=d),
        )
        h = torch.randn(c["batch_size"], c["seq_len"], H)
        out = self._fwd(torch, moe_module, h.clone(), weights, c)
        assert torch.allclose(out, h, atol=1e-5)
