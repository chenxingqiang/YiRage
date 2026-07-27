# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""FusionPlan: search-time / cacheable local execution plan."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class FusionPlan:
    """YiRage standard name for a local fused execution plan.

    Replaces legacy product wording ``µGraph`` / ``MuGraph`` in serving narrative.
    Implementation may still be stored under legacy mugraph paths (chore rename later).
    """

    name: str
    kind: str  # e.g. "mlp", "attn_tile", "decoder_fragment"
    hidden_size: int
    intermediate_size: Optional[int] = None
    dtype: str = "float32"
    backend: str = "eager_numpy"
    version: str = "s1"
    notes: Tuple[str, ...] = ()
    legacy_aliases: Tuple[str, ...] = ()
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def mlp(
        cls,
        *,
        name: str = "mlp_rms_gated_residual",
        hidden_size: int,
        intermediate_size: int,
        dtype: str = "float32",
        backend: str = "eager_numpy",
    ) -> "FusionPlan":
        return cls(
            name=name,
            kind="mlp",
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
            backend=backend,
            notes=(
                "rmsnorm + silu(gate)*up + down + residual",
                "RF-selectable FusionCapsule (S1)",
            ),
            legacy_aliases=("mugraph_mlp", "pk_silu_mul_linear_fragment"),
        )
