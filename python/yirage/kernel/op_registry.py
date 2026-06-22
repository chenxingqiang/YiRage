# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0

"""
Custom Operator Registry for YiRage.

Allows users to register custom operators as named, reusable builders that
are backed by YiRage's existing ``KN_CUSTOMIZED_OP`` / ``KNGraph.customized()``
mechanism.  No C++ changes are needed — the registry lives entirely in Python.

Quick-start
-----------
**Decorator API** (recommended)::

    import yirage as mi

    @mi.custom_op("softmax", n_inputs=1, description="Row-wise softmax")
    def build_softmax(kgraph, inputs, **kwargs):
        A = inputs[0]
        seq = A.dim[A.num_dims - 1]
        bgraph = mi.new_threadblock_graph(
            grid_dim=kwargs.get("grid_dim", (1, 1, 1)),
            block_dim=kwargs.get("block_dim", (128, 1, 1)),
            forloop_range=kwargs.get("forloop_range", 1),
            reduction_dimx=kwargs.get("reduction_dimx", seq),
        )
        a_smem  = bgraph.new_input(A, input_map=(0, -1, -1), forloop_dim=-1)
        exp_a   = bgraph.exp(a_smem)
        sum_a   = bgraph.reduction(exp_a, dim=1)
        out     = bgraph.div(exp_a, sum_a)
        bgraph.new_output(out, output_map=(0, -1, -1))
        return kgraph.customized(inputs, bgraph)

**Functional API**::

    mi.register_op("softmax", build_softmax, n_inputs=1)

**Using a registered operator inside a kernel graph**::

    kgraph = mi.new_kernel_graph()
    A = kgraph.new_input([seq, d], dtype=mi.float16)
    (out,) = kgraph.call_op("softmax", [A],
                             grid_dim=(1, 1, 1), block_dim=(128, 1, 1))
    kgraph.mark_output(out)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

__all__ = [
    "CustomOpSpec",
    "OpRegistry",
    "global_registry",
    "register_op",
    "custom_op",
    "list_ops",
    "get_op",
]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class CustomOpSpec:
    """Metadata for a single registered custom operator.

    Attributes
    ----------
    name:
        Unique string identifier used to call this operator via
        :py:meth:`KNGraph.call_op`.
    builder:
        Callable with signature ``(kgraph, inputs: list, **kwargs) -> list``.
        It should create a :class:`~yirage.kernel.threadblock.TBGraph`,
        populate it, and return the result of ``kgraph.customized(inputs, bgraph)``.
    n_inputs:
        Expected number of input tensors.  Pass ``-1`` (default) to skip the
        arity check.
    description:
        Human-readable documentation string.
    """

    name: str
    builder: Callable
    n_inputs: int = -1
    description: str = ""
    tags: List[str] = field(default_factory=list)

    def __call__(self, kgraph, inputs: list, **kwargs) -> list:
        if self.n_inputs >= 0 and len(inputs) != self.n_inputs:
            raise ValueError(
                f"Custom op '{self.name}' expects {self.n_inputs} input(s), "
                f"got {len(inputs)}."
            )
        result = self.builder(kgraph, inputs, **kwargs)
        # Normalise to list so callers can always unpack
        if result is None:
            return []
        if not isinstance(result, (list, tuple)):
            return [result]
        return list(result)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class OpRegistry:
    """A registry that maps operator names to :class:`CustomOpSpec` objects.

    The module-level :data:`global_registry` is the singleton used by
    :func:`register_op`, :func:`custom_op`, and :func:`list_ops`.
    A fresh :class:`OpRegistry` can also be created for isolated use-cases
    (e.g. unit tests).
    """

    def __init__(self) -> None:
        self._ops: Dict[str, CustomOpSpec] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        builder: Callable,
        *,
        n_inputs: int = -1,
        description: str = "",
        tags: Optional[List[str]] = None,
        overwrite: bool = False,
    ) -> CustomOpSpec:
        """Register a custom operator builder.

        Parameters
        ----------
        name:
            Unique name.  Raises :class:`ValueError` if the name is already
            registered and *overwrite* is ``False``.
        builder:
            Callable ``(kgraph, inputs, **kwargs) -> list[DTensor]``.
        n_inputs:
            Expected input count (``-1`` = unchecked).
        description:
            Human-readable doc string.
        tags:
            Optional list of metadata tags (e.g. ``["attention", "fused"]``).
        overwrite:
            When ``True``, silently replace an existing registration.
            When ``False`` (default), emit a :class:`UserWarning` and keep
            the existing one.

        Returns
        -------
        CustomOpSpec
            The spec that was stored (or the pre-existing one if not overwritten).
        """
        if not isinstance(name, str) or not name:
            raise ValueError("Operator name must be a non-empty string.")
        if not callable(builder):
            raise TypeError(f"builder for '{name}' must be callable, got {type(builder)}.")

        if name in self._ops:
            if overwrite:
                pass  # fall through to re-register
            else:
                warnings.warn(
                    f"Custom op '{name}' is already registered. "
                    "Pass overwrite=True to replace it.",
                    UserWarning,
                    stacklevel=3,
                )
                return self._ops[name]

        spec = CustomOpSpec(
            name=name,
            builder=builder,
            n_inputs=n_inputs,
            description=description,
            tags=list(tags or []),
        )
        self._ops[name] = spec
        return spec

    def decorator(
        self,
        name: str,
        *,
        n_inputs: int = -1,
        description: str = "",
        tags: Optional[List[str]] = None,
        overwrite: bool = False,
    ) -> Callable:
        """Return a decorator that registers the decorated function.

        Usage::

            registry = OpRegistry()

            @registry.decorator("my_op", n_inputs=2)
            def build_my_op(kgraph, inputs, **kwargs):
                ...
        """

        def _decorator(fn: Callable) -> Callable:
            self.register(
                name,
                fn,
                n_inputs=n_inputs,
                description=description or (fn.__doc__ or ""),
                tags=tags,
                overwrite=overwrite,
            )
            return fn  # preserve the original function unchanged

        return _decorator

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, name: str) -> CustomOpSpec:
        """Look up a registered op by name.

        Raises :class:`KeyError` if not found.
        """
        if name not in self._ops:
            registered = list(self._ops.keys())
            raise KeyError(
                f"Custom op '{name}' is not registered. "
                f"Registered ops: {registered}"
            )
        return self._ops[name]

    def __contains__(self, name: str) -> bool:
        return name in self._ops

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def list(self) -> List[CustomOpSpec]:
        """Return all registered :class:`CustomOpSpec` objects."""
        return list(self._ops.values())

    def names(self) -> List[str]:
        """Return all registered operator names."""
        return list(self._ops.keys())

    def __len__(self) -> int:
        return len(self._ops)

    def __repr__(self) -> str:
        return f"OpRegistry({self.names()})"

    # ------------------------------------------------------------------
    # Unregistration (useful for tests / hot-reload)
    # ------------------------------------------------------------------

    def unregister(self, name: str) -> None:
        """Remove a registered operator.  Raises :class:`KeyError` if absent."""
        if name not in self._ops:
            raise KeyError(f"Custom op '{name}' is not registered.")
        del self._ops[name]

    def clear(self) -> None:
        """Remove all registered operators."""
        self._ops.clear()


# ---------------------------------------------------------------------------
# Module-level singleton & convenience helpers
# ---------------------------------------------------------------------------

#: The global :class:`OpRegistry` instance used by :func:`register_op`,
#: :func:`custom_op`, and :func:`list_ops`.
global_registry: OpRegistry = OpRegistry()


def register_op(
    name: str,
    builder: Callable,
    *,
    n_inputs: int = -1,
    description: str = "",
    tags: Optional[List[str]] = None,
    overwrite: bool = False,
    registry: Optional[OpRegistry] = None,
) -> CustomOpSpec:
    """Register a custom operator in the global (or provided) registry.

    Parameters
    ----------
    name:
        Unique operator name.
    builder:
        Builder function ``(kgraph, inputs: list, **kwargs) -> list[DTensor]``.
    n_inputs:
        Expected number of inputs (``-1`` = unchecked).
    description:
        Human-readable documentation.
    tags:
        Optional metadata tags.
    overwrite:
        Replace existing registration without warning.
    registry:
        Target :class:`OpRegistry` (defaults to :data:`global_registry`).

    Returns
    -------
    CustomOpSpec
        The stored spec object.

    Example
    -------
    ::

        import yirage as mi

        def build_softmax(kgraph, inputs, **kwargs):
            ...
            return kgraph.customized(inputs, bgraph)

        mi.register_op("softmax", build_softmax, n_inputs=1)
    """
    reg = registry if registry is not None else global_registry
    return reg.register(
        name,
        builder,
        n_inputs=n_inputs,
        description=description,
        tags=tags,
        overwrite=overwrite,
    )


def custom_op(
    name: str,
    *,
    n_inputs: int = -1,
    description: str = "",
    tags: Optional[List[str]] = None,
    overwrite: bool = False,
    registry: Optional[OpRegistry] = None,
) -> Callable:
    """Decorator that registers the decorated function as a custom operator.

    Parameters
    ----------
    name:
        Unique operator name.
    n_inputs:
        Expected input count (``-1`` = unchecked).
    description:
        Human-readable doc.  Defaults to the function's docstring.
    tags:
        Optional metadata tags.
    overwrite:
        Replace an existing registration without warning.
    registry:
        Target :class:`OpRegistry` (defaults to :data:`global_registry`).

    Example
    -------
    ::

        @mi.custom_op("softmax", n_inputs=1)
        def build_softmax(kgraph, inputs, **kwargs):
            \"\"\"Row-wise softmax.\"\"\"
            ...
            return kgraph.customized(inputs, bgraph)
    """
    reg = registry if registry is not None else global_registry
    return reg.decorator(
        name,
        n_inputs=n_inputs,
        description=description,
        tags=tags,
        overwrite=overwrite,
    )


def list_ops(registry: Optional[OpRegistry] = None) -> List[CustomOpSpec]:
    """Return all registered :class:`CustomOpSpec` objects.

    Parameters
    ----------
    registry:
        Source registry (defaults to :data:`global_registry`).

    Returns
    -------
    list[CustomOpSpec]
    """
    reg = registry if registry is not None else global_registry
    return reg.list()


def get_op(name: str, registry: Optional[OpRegistry] = None) -> CustomOpSpec:
    """Look up a registered operator by name.

    Parameters
    ----------
    name:
        Operator name.
    registry:
        Source registry (defaults to :data:`global_registry`).

    Raises
    ------
    KeyError
        If the operator is not found.
    """
    reg = registry if registry is not None else global_registry
    return reg.get(name)
