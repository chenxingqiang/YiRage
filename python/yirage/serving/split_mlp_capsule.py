# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S7: split MLP as a 2-Capsule pipeline (gate_up → down), Qwen-kernel aligned."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from .capsule import FusionCapsule
from .capsule_orchestration import split_mlp_down_name, split_mlp_gate_up_name
from .exec_backend import BACKEND_NUMPY_REF, BACKEND_TORCH, default_serving_backend
from .mlp_capsule import _rms_norm, _silu, mlp_eager_numpy
from .plan import FusionPlan
from .torch_exec import mlp_torch, require_torch, rms_norm_torch, to_torch


def _gate_up_numpy(hidden, *, rms_weight, w_gate, w_up, eps=1e-6):
    h = _rms_norm(hidden, rms_weight, eps=eps)
    gate = h @ w_gate
    up = h @ w_up
    return gate, up


def _down_numpy(hidden, gate, up, *, w_down):
    mid = _silu(gate) * up
    return hidden + mid @ w_down


class MlpGateUpFusionCapsule(FusionCapsule):
    """Stage 1: RMSNorm + gate/up projections (outputs ``gate``, ``up``)."""

    def __init__(
        self,
        plan: FusionPlan,
        *,
        rms_weight: Any,
        w_gate: Any,
        w_up: Any,
        eps: float = 1e-6,
        device: Optional[str] = None,
    ):
        if plan.kind != "mlp_gate_up":
            raise ValueError(f"expected kind mlp_gate_up, got {plan.kind!r}")
        super().__init__(plan)
        self.eps = float(eps)
        self._device = device
        if plan.backend == BACKEND_TORCH:
            require_torch()
            import torch

            dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._device = dev
            self.rms_weight = to_torch(rms_weight, device=dev, dtype=torch.float32)
            self.w_gate = to_torch(w_gate, device=dev, dtype=torch.float32)
            self.w_up = to_torch(w_up, device=dev, dtype=torch.float32)
        else:
            self.rms_weight = np.asarray(rms_weight)
            self.w_gate = np.asarray(w_gate)
            self.w_up = np.asarray(w_up)

    def execute(
        self,
        inputs: Mapping[str, Any],
        meta: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        hidden = inputs["hidden"]
        radix = None
        if meta:
            from .radix_meta import infer_batch_size_from_hidden, parse_radix_hit_mask

            mask = meta.get("radix_hit_mask")
            if mask is None and meta.get("radix_hit"):
                mask = meta["radix_hit"].get("radix_hit_mask")
            if mask is not None:
                bs = infer_batch_size_from_hidden(hidden)
                radix = parse_radix_hit_mask(mask, batch_size=bs)

        if self.plan.backend == BACKEND_TORCH:
            import torch

            h = to_torch(hidden, device=self._device)

            def _run(active):
                normed = rms_norm_torch(active, self.rms_weight, eps=self.eps)
                return normed @ self.w_gate, normed @ self.w_up

            if radix is not None and radix.any_hit:
                hidden_out = h
                gate = torch.zeros(h.shape[0], self.w_gate.shape[1], device=h.device, dtype=h.dtype)
                up = torch.zeros_like(gate)
                active = torch.as_tensor(radix.active_row_indices(), device=h.device)
                g, u = _run(h.index_select(0, active))
                gate.index_copy_(0, active, g)
                up.index_copy_(0, active, u)
                return {"hidden": hidden_out, "gate": gate, "up": up}
            normed = rms_norm_torch(h, self.rms_weight, eps=self.eps)
            return {"hidden": h, "gate": normed @ self.w_gate, "up": normed @ self.w_up}

        hidden_np = np.asarray(hidden)
        if radix is not None and radix.any_hit:
            active = radix.active_row_indices()
            gate = np.zeros((hidden_np.shape[0], self.w_gate.shape[1]), dtype=hidden_np.dtype)
            up = np.zeros_like(gate)
            g, u = _gate_up_numpy(
                hidden_np[active],
                rms_weight=self.rms_weight,
                w_gate=self.w_gate,
                w_up=self.w_up,
                eps=self.eps,
            )
            gate[active] = g
            up[active] = u
            return {"hidden": hidden_np, "gate": gate, "up": up}
        gate, up = _gate_up_numpy(
            hidden_np,
            rms_weight=self.rms_weight,
            w_gate=self.w_gate,
            w_up=self.w_up,
            eps=self.eps,
        )
        return {"hidden": hidden_np, "gate": gate, "up": up}


class MlpDownFusionCapsule(FusionCapsule):
    """Stage 2: SiLU(gate)*up + down + residual."""

    def __init__(
        self,
        plan: FusionPlan,
        *,
        w_down: Any,
        device: Optional[str] = None,
    ):
        if plan.kind != "mlp_down":
            raise ValueError(f"expected kind mlp_down, got {plan.kind!r}")
        super().__init__(plan)
        self._device = device
        if plan.backend == BACKEND_TORCH:
            require_torch()
            import torch

            dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._device = dev
            self.w_down = to_torch(w_down, device=dev, dtype=torch.float32)
        else:
            self.w_down = np.asarray(w_down)

    def execute(
        self,
        inputs: Mapping[str, Any],
        meta: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        if "gate" not in inputs or "up" not in inputs:
            raise KeyError("MlpDownFusionCapsule requires inputs['gate'] and inputs['up']")
        hidden = inputs["hidden"]
        gate = inputs["gate"]
        up = inputs["up"]
        radix = None
        if meta:
            from .radix_meta import infer_batch_size_from_hidden, parse_radix_hit_mask

            mask = meta.get("radix_hit_mask")
            if mask is None and meta.get("radix_hit"):
                mask = meta["radix_hit"].get("radix_hit_mask")
            if mask is not None:
                bs = infer_batch_size_from_hidden(hidden)
                radix = parse_radix_hit_mask(mask, batch_size=bs)

        if self.plan.backend == BACKEND_TORCH:
            import torch.nn.functional as F

            h = to_torch(hidden, device=self._device)
            g = to_torch(gate, device=self._device)
            u = to_torch(up, device=self._device)

            def _run(active_h):
                active_t = torch.as_tensor(radix.active_row_indices(), device=h.device)
                mid = F.silu(g.index_select(0, active_t)) * u.index_select(0, active_t)
                return active_h + mid @ self.w_down

            if radix is not None and radix.any_hit:
                from .radix_meta import apply_radix_shrink

                out = apply_radix_shrink(h, radix, _run)
                return {"hidden": out}
            mid = F.silu(g) * u
            return {"hidden": h + mid @ self.w_down}

        hidden_np = np.asarray(hidden)
        gate_np = np.asarray(gate)
        up_np = np.asarray(up)

        def _run_np(active_h):
            idx = radix.active_row_indices() if radix is not None else slice(None)
            return _down_numpy(active_h, gate_np[idx], up_np[idx], w_down=self.w_down)

        if radix is not None and radix.any_hit:
            from .radix_meta import apply_radix_shrink

            out = apply_radix_shrink(hidden_np, radix, _run_np)
            return {"hidden": out}
        out = _down_numpy(hidden_np, gate_np, up_np, w_down=self.w_down)
        return {"hidden": out}


def build_layer_split_mlp_capsules(
    layer,
    *,
    backend: Optional[str] = None,
) -> Tuple[MlpGateUpFusionCapsule, MlpDownFusionCapsule]:
    from .exec_backend import BACKEND_NUMPY_REF, default_serving_backend
    from .engine_stub import EngineDecoderLayerStub

    be = backend or (
        BACKEND_NUMPY_REF
        if isinstance(layer, EngineDecoderLayerStub)
        else default_serving_backend()
    )
    lid = layer.layer_id
    h, i = layer.hidden_size, layer.intermediate_size
    gate_plan = FusionPlan(
        name=split_mlp_gate_up_name(lid),
        kind="mlp_gate_up",
        hidden_size=h,
        intermediate_size=i,
        dtype="float32",
        backend=be,
        version="s7",
        notes=("rmsnorm + gate/up matmul", "S7 split MLP stage 1"),
    )
    down_plan = FusionPlan(
        name=split_mlp_down_name(lid),
        kind="mlp_down",
        hidden_size=h,
        intermediate_size=i,
        dtype="float32",
        backend=be,
        version="s7",
        notes=("silu(gate)*up + down + residual", "S7 split MLP stage 2"),
    )
    gate_up = MlpGateUpFusionCapsule(
        gate_plan,
        rms_weight=layer.rms_weight,
        w_gate=layer.w_gate,
        w_up=layer.w_up,
        device=getattr(layer, "device", None),
    )
    down = MlpDownFusionCapsule(
        down_plan,
        w_down=layer.w_down,
        device=getattr(layer, "device", None),
    )
    return gate_up, down


def split_mlp_parity_numpy(
    hidden: np.ndarray,
    *,
    rms_weight: np.ndarray,
    w_gate: np.ndarray,
    w_up: np.ndarray,
    w_down: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Oracle: gate_up capsule → down capsule equals fused MLP."""
    gate, up = _gate_up_numpy(
        hidden, rms_weight=rms_weight, w_gate=w_gate, w_up=w_up, eps=eps
    )
    return _down_numpy(hidden, gate, up, w_down=w_down)


def split_mlp_matches_fused(
    hidden: Any,
    *,
    rms_weight,
    w_gate,
    w_up,
    w_down,
    backend: str,
    eps: float = 1e-6,
) -> bool:
    if backend == BACKEND_TORCH:
        require_torch()
        import torch

        ref = mlp_torch(hidden, rms_weight=rms_weight, w_gate=w_gate, w_up=w_up, w_down=w_down, eps=eps)
        gate, down = build_layer_split_mlp_capsules(
            type("_L", (), {
                "layer_id": 0,
                "hidden_size": w_gate.shape[0],
                "intermediate_size": w_gate.shape[1],
                "rms_weight": rms_weight,
                "w_gate": w_gate,
                "w_up": w_up,
                "w_down": w_down,
            })(),
            backend=BACKEND_TORCH,
        )
        from .runtime_fusion import RuntimeFusion

        rf = RuntimeFusion([gate, down])
        out = rf.step(
            {"hidden": hidden},
            meta={"pipeline": [gate.name, down.name]},
        )
        return bool(torch.allclose(out.outputs["hidden"], ref, rtol=1e-5, atol=1e-5))
    ref = mlp_eager_numpy(
        np.asarray(hidden),
        rms_weight=np.asarray(rms_weight),
        w_gate=np.asarray(w_gate),
        w_up=np.asarray(w_up),
        w_down=np.asarray(w_down),
        eps=eps,
    )
    got = split_mlp_parity_numpy(
        np.asarray(hidden),
        rms_weight=np.asarray(rms_weight),
        w_gate=np.asarray(w_gate),
        w_up=np.asarray(w_up),
        w_down=np.asarray(w_down),
        eps=eps,
    )
    return bool(np.allclose(got, ref, rtol=1e-5, atol=1e-6))
