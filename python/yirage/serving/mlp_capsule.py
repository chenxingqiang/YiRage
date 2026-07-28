# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""MLP FusionCapsule — first RF-selectable block (S1).

Semantics (Qwen-style gated MLP with pre-RMSNorm + residual)::

    h = rmsnorm(x) * rms_weight
    mid = silu(h @ W_gate) * (h @ W_up)
    y = x + mid @ W_down

S1 default backend is ``torch`` when PyTorch is available (real execution).
``numpy_ref`` remains for offline reference parity only.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple, Union

import numpy as np

from .capsule import FusionCapsule
from .exec_backend import (
    BACKEND_NUMPY_REF,
    BACKEND_TORCH,
    BACKEND_YIRAGE_CPU,
    BACKEND_YIRAGE_MACA,
    default_serving_backend,
)
from .plan import FusionPlan
from .torch_exec import mlp_torch, require_torch, to_numpy, to_torch


def _silu(x: np.ndarray) -> np.ndarray:
    x32 = x.astype(np.float32, copy=False)
    return (x32 * (1.0 / (1.0 + np.exp(-x32)))).astype(x.dtype, copy=False)


def _rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x32 = x.astype(np.float32, copy=False)
    var = np.mean(np.square(x32), axis=-1, keepdims=True)
    y = x32 * np.reciprocal(np.sqrt(var + eps))
    return (y * weight.astype(np.float32, copy=False)).astype(x.dtype, copy=False)


def mlp_eager_numpy(
    hidden: np.ndarray,
    *,
    rms_weight: np.ndarray,
    w_gate: np.ndarray,
    w_up: np.ndarray,
    w_down: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Reference fused MLP used by :class:`MlpFusionCapsule` (S1)."""
    if hidden.ndim != 2:
        raise ValueError(f"hidden must be rank-2 [B,H], got shape={hidden.shape}")
    h = _rms_norm(hidden, rms_weight, eps=eps)
    gate = h @ w_gate
    up = h @ w_up
    mid = _silu(gate) * up
    return hidden + mid @ w_down


def mlp_unfused_numpy(
    hidden: np.ndarray,
    *,
    rms_weight: np.ndarray,
    w_gate: np.ndarray,
    w_up: np.ndarray,
    w_down: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Same math as :func:`mlp_eager_numpy`, written as staged ops (parity oracle)."""
    return mlp_eager_numpy(
        hidden,
        rms_weight=rms_weight,
        w_gate=w_gate,
        w_up=w_up,
        w_down=w_down,
        eps=eps,
    )


class MlpFusionCapsule(FusionCapsule):
    """RF-selectable MLP FusionCapsule (S1)."""

    def __init__(
        self,
        plan: FusionPlan,
        *,
        rms_weight: Any,
        w_gate: Any,
        w_up: Any,
        w_down: Any,
        eps: float = 1e-6,
        device: Optional[str] = None,
    ):
        if plan.kind != "mlp":
            raise ValueError(f"MlpFusionCapsule requires plan.kind=='mlp', got {plan.kind!r}")
        super().__init__(plan)
        self.eps = float(eps)
        self._device = device
        backend = plan.backend
        if backend == BACKEND_YIRAGE_CPU:
            from .yirage_exec import YirageServingMlpRunner, require_yirage_core

            require_yirage_core()
            require_torch()
            self._yirage_runner = YirageServingMlpRunner(
                rms_weight=rms_weight,
                w_gate=w_gate,
                w_up=w_up,
                w_down=w_down,
                eps=eps,
                device=device,
                dtype_name=plan.dtype or "float32",
                quick_superopt=True,
            )
            self._device = self._yirage_runner._device
            self.rms_weight = self._yirage_runner.rms_weight
            self.w_gate = self._yirage_runner.w_gate
            self.w_up = self._yirage_runner.w_up
            self.w_down = self._yirage_runner.w_down
        elif backend == BACKEND_YIRAGE_MACA:
            from .maca_exec import YirageMacaServingMlpRunner, require_yirage_maca

            require_yirage_maca()
            require_torch()
            self._maca_runner = YirageMacaServingMlpRunner(
                rms_weight=rms_weight,
                w_gate=w_gate,
                w_up=w_up,
                w_down=w_down,
                eps=eps,
                device=device,
                dtype_name=plan.dtype or "float32",
                quick_superopt=True,
            )
            self._device = self._maca_runner._device
            self.rms_weight = self._maca_runner.rms_weight
            self.w_gate = self._maca_runner.w_gate
            self.w_up = self._maca_runner.w_up
            self.w_down = self._maca_runner.w_down
        elif backend == BACKEND_TORCH:
            require_torch()
            import torch

            dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._device = dev
            self.rms_weight = to_torch(rms_weight, device=dev, dtype=torch.float32)
            self.w_gate = to_torch(w_gate, device=dev, dtype=torch.float32)
            self.w_up = to_torch(w_up, device=dev, dtype=torch.float32)
            self.w_down = to_torch(w_down, device=dev, dtype=torch.float32)
        else:
            self.rms_weight = np.asarray(rms_weight)
            self.w_gate = np.asarray(w_gate)
            self.w_up = np.asarray(w_up)
            self.w_down = np.asarray(w_down)
        self._validate_shapes()

    def _validate_shapes(self) -> None:
        h = self.plan.hidden_size
        i = self.plan.intermediate_size
        if i is None:
            raise ValueError("MLP FusionPlan requires intermediate_size")
        if self.plan.backend in (BACKEND_TORCH, BACKEND_YIRAGE_CPU, BACKEND_YIRAGE_MACA):
            import torch

            def _shape(t):
                return tuple(t.shape)

            rw, wg, wu, wd = self.rms_weight, self.w_gate, self.w_up, self.w_down
            assert isinstance(rw, torch.Tensor)
        else:
            _shape = lambda t: t.shape  # noqa: E731
            rw, wg, wu, wd = self.rms_weight, self.w_gate, self.w_up, self.w_down
        if _shape(rw) != (h,):
            raise ValueError(f"rms_weight shape {_shape(rw)} != ({h},)")
        if _shape(wg) != (h, i):
            raise ValueError(f"w_gate shape {_shape(wg)} != ({h}, {i})")
        if _shape(wu) != (h, i):
            raise ValueError(f"w_up shape {_shape(wu)} != ({h}, {i})")
        if _shape(wd) != (i, h):
            raise ValueError(f"w_down shape {_shape(wd)} != ({i}, {h})")

    @classmethod
    def from_random(
        cls,
        *,
        hidden_size: int = 64,
        intermediate_size: int = 128,
        seed: int = 0,
        name: str = "mlp_rms_gated_residual",
        dtype=np.float32,
        plan: Optional[FusionPlan] = None,
        backend: Optional[str] = None,
        device: Optional[str] = None,
    ) -> "MlpFusionCapsule":
        rng = np.random.default_rng(seed)
        be = backend or default_serving_backend()
        if plan is None:
            plan = FusionPlan.mlp(
                name=name,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                dtype=np.dtype(dtype).name,
                backend=be,
            )
        scale = 0.02
        return cls(
            plan,
            rms_weight=np.ones((hidden_size,), dtype=dtype),
            w_gate=rng.normal(0.0, scale, size=(hidden_size, intermediate_size)).astype(dtype),
            w_up=rng.normal(0.0, scale, size=(hidden_size, intermediate_size)).astype(dtype),
            w_down=rng.normal(0.0, scale, size=(intermediate_size, hidden_size)).astype(dtype),
            device=device,
        )

    def execute(
        self,
        inputs: Mapping[str, Any],
        meta: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        if "hidden" not in inputs:
            raise KeyError("MlpFusionCapsule.execute requires inputs['hidden']")
        hidden = inputs["hidden"]
        radix = None
        if meta:
            from .radix_meta import infer_batch_size_from_hidden, parse_radix_hit_mask

            mask = meta.get("radix_hit_mask")
            if mask is not None:
                bs = infer_batch_size_from_hidden(hidden)
                radix = parse_radix_hit_mask(mask, batch_size=bs)

        if self.plan.backend == BACKEND_YIRAGE_CPU:
            from .radix_meta import apply_radix_shrink

            h = to_torch(hidden, device=self._device)
            if h.ndim == 2 and h.shape[0] != 1 and radix is not None:
                raise ValueError(
                    "yirage_cpu MLP with radix shrink requires batch=1 decode today"
                )
            if radix is not None and radix.any_hit:
                out = apply_radix_shrink(h, radix, self._yirage_runner.forward)
            else:
                out = self._yirage_runner.forward(h)
            return {"hidden": out}
        if self.plan.backend == BACKEND_YIRAGE_MACA:
            from .radix_meta import apply_radix_shrink

            h = to_torch(hidden, device=self._device)
            if h.ndim == 2 and h.shape[0] != 1 and radix is not None:
                raise ValueError(
                    "yirage_maca MLP with radix shrink requires batch=1 decode today"
                )
            if radix is not None and radix.any_hit:
                out = apply_radix_shrink(h, radix, self._maca_runner.forward)
            else:
                out = self._maca_runner.forward(h)
            return {"hidden": out}
        if self.plan.backend == BACKEND_TORCH:
            h = to_torch(hidden, device=self._device)

            def _run(active):
                return mlp_torch(
                    active,
                    rms_weight=self.rms_weight,
                    w_gate=self.w_gate,
                    w_up=self.w_up,
                    w_down=self.w_down,
                    eps=self.eps,
                )

            if radix is not None and radix.any_hit:
                from .radix_meta import apply_radix_shrink

                out = apply_radix_shrink(h, radix, _run)
            else:
                out = _run(h)
            return {"hidden": out}
        hidden_np = np.asarray(hidden)

        def _run_np(active):
            return mlp_eager_numpy(
                active,
                rms_weight=self.rms_weight,
                w_gate=self.w_gate,
                w_up=self.w_up,
                w_down=self.w_down,
                eps=self.eps,
            )

        if radix is not None and radix.any_hit:
            from .radix_meta import apply_radix_shrink

            out = apply_radix_shrink(hidden_np, radix, _run_np)
        else:
            out = _run_np(hidden_np)
        return {"hidden": out}

    def weights(self) -> Tuple[Any, Any, Any, Any]:
        return self.rms_weight, self.w_gate, self.w_up, self.w_down
