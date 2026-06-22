# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""Search-time equivalence verification configuration.

YiRage search filters µGraph candidates with a **fast** probabilistic fingerprint
verifier by default (``ProbabilisticVerifier`` in C++). An optional **slow**
formal path (``FormalVerifier`` + Rust ``libformal_verifier``) is enabled via
``is_formal_verified`` / ``YIRAGE_FORMAL_VERIFY``.

Runtime execution should still be checked with full-precision torch reference
ops (see ``runtime_verify_mugraph``).
"""

from __future__ import annotations

import os
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def env_truthy(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUTHY


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def formal_verifier_library_path() -> Optional[Path]:
    """Return ``libformal_verifier.so`` when the Rust helper is built."""
    for base in (_repo_root() / "build" / "formal_verifier" / "release",):
        for name in ("libformal_verifier.so", "libformal_verifier.dylib"):
            p = base / name
            if p.is_file():
                return p
    return None


def is_formal_verifier_built() -> bool:
    """True when CMake linked formal verification or the Rust .so exists."""
    try:
        from yirage.cmake_macros import parse_config_cmake

        cfg = _repo_root() / "config.cmake"
        if cfg.is_file() and parse_config_cmake(str(cfg)).get("USE_FORMAL_VERIFIER"):
            return True
    except Exception:
        pass
    return formal_verifier_library_path() is not None


@dataclass(frozen=True)
class VerifierConfig:
    """Resolved search-time verifier settings."""

    is_formal_verified: bool
    verifier_type: str  # "probabilistic" | "formal"
    formal_available: bool
    env_formal_verify: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def resolve_verifier_config(
    *,
    formal_verify: Optional[bool] = None,
    is_formal_verified: Optional[bool] = None,
    warn_on_fallback: bool = True,
) -> VerifierConfig:
    """Resolve search verifier from explicit args and ``YIRAGE_FORMAL_VERIFY``.

    Precedence: ``is_formal_verified`` > ``formal_verify`` > env
    ``YIRAGE_FORMAL_VERIFY``. Default is fast probabilistic fingerprinting.
    """
    env_formal = env_truthy("YIRAGE_FORMAL_VERIFY")
    if is_formal_verified is not None:
        want_formal = bool(is_formal_verified)
    elif formal_verify is not None:
        want_formal = bool(formal_verify)
    else:
        want_formal = env_formal

    formal_available = is_formal_verifier_built()
    if want_formal and not formal_available:
        if warn_on_fallback:
            warnings.warn(
                "YIRAGE_FORMAL_VERIFY requested but formal verifier is not built "
                "(USE_FORMAL_VERIFIER=OFF and libformal_verifier missing). "
                "Falling back to probabilistic fingerprint verification.",
                RuntimeWarning,
                stacklevel=2,
            )
        want_formal = False

    verifier_type = "formal" if want_formal else "probabilistic"
    return VerifierConfig(
        is_formal_verified=want_formal,
        verifier_type=verifier_type,
        formal_available=formal_available,
        env_formal_verify=env_formal,
    )


def runtime_verify_mugraph(
    runner: Any,
    inputs: List[torch.Tensor],
    reference: torch.Tensor,
    *,
    max_abs_tol: float = 0.08,
) -> Tuple[bool, float, Optional[str]]:
    """Full-precision torch reference check after search-time fingerprint pass."""
    try:
        out = runner(inputs=inputs)[0]
        max_err = (out.float() - reference.float()).abs().max().item()
        return max_err < max_abs_tol, max_err, None
    except Exception as exc:
        return False, float("inf"), str(exc)
