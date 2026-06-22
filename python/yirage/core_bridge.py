# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0

"""
Core Bridge Module - Unified Interface to C++ Core

This module provides a clean, unified interface to all C++ core functionality:
- muGraph search and optimization
- RL closed-loop interface
- MLIR code generation
- GPU verification and profiling

Architecture:
    
    ┌────────────────────────────────────────────────────────────────────────┐
    │                          Python Layer                                   │
    │                                                                         │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
    │  │   Kernel    │  │     RL      │  │   MLIR      │  │  Compiler   │    │
    │  │   Graph     │  │   Search    │  │  Codegen    │  │  Pipeline   │    │
    │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘    │
    │         │                │                │                │            │
    │         └────────────────┴────────────────┴────────────────┘            │
    │                                  │                                      │
    │                        ┌─────────▼─────────┐                           │
    │                        │    CoreBridge     │                           │
    │                        └─────────┬─────────┘                           │
    └──────────────────────────────────┼──────────────────────────────────────┘
                                       │
    ┌──────────────────────────────────┼──────────────────────────────────────┐
    │                          C++ Layer                                      │
    │                        ┌─────────▼─────────┐                           │
    │                        │   Cython Core     │                           │
    │                        └─────────┬─────────┘                           │
    │         ┌────────────────┬───────┴───────┬────────────────┐            │
    │  ┌──────▼──────┐  ┌──────▼──────┐ ┌──────▼──────┐ ┌───────▼──────┐    │
    │  │   Search    │  │  RL Core    │ │   Graph     │ │  Transpiler  │    │
    │  │   Engine    │  │  Context    │ │  Features   │ │   Runtime    │    │
    │  └─────────────┘  └─────────────┘ └─────────────┘ └──────────────┘    │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘
"""

from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
import json
import os


# ============================================================
# Core Status Tracking
# ============================================================


class CoreStatus(Enum):
    """Status of C++ core components."""

    AVAILABLE = auto()
    UNAVAILABLE = auto()
    PARTIAL = auto()


@dataclass
class CoreCapabilities:
    """Tracks available C++ core capabilities."""

    core_available: bool = False
    kernel_available: bool = False
    rl_available: bool = False
    mlir_available: bool = False
    profiler_available: bool = False

    def get_status(self) -> CoreStatus:
        if all([self.core_available, self.kernel_available]):
            if self.rl_available and self.mlir_available:
                return CoreStatus.AVAILABLE
            return CoreStatus.PARTIAL
        return CoreStatus.UNAVAILABLE

    def __str__(self) -> str:
        components = []
        if self.core_available:
            components.append("core")
        if self.kernel_available:
            components.append("kernel")
        if self.rl_available:
            components.append("rl")
        if self.mlir_available:
            components.append("mlir")
        if self.profiler_available:
            components.append("profiler")
        return f"CoreCapabilities({', '.join(components) or 'none'})"


# ============================================================
# Core Bridge Class
# ============================================================


class CoreBridge:
    """
    Unified bridge to C++ core functionality.

    Usage:
        bridge = CoreBridge()

        # Check capabilities
        if bridge.capabilities.rl_available:
            context = bridge.create_rl_context(graph)
            result = context.verify()

        # Optimize kernel
        optimized = bridge.optimize(graph, backend='cuda')

        # Generate code
        code = bridge.generate_code(optimized, target='cuda')
    """

    _instance: Optional["CoreBridge"] = None

    def __init__(self):
        self._capabilities = CoreCapabilities()
        self._init_components()

    @classmethod
    def get_instance(cls) -> "CoreBridge":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _init_components(self):
        """Initialize and detect C++ components."""
        # Core module
        try:
            from . import core

            self._core = core
            self._capabilities.core_available = True
        except ImportError:
            self._core = None

        # Kernel module
        try:
            from .kernel.graph import KNGraph

            self._kn_graph_class = KNGraph
            self._capabilities.kernel_available = True
        except ImportError:
            self._kn_graph_class = None

        # RL module
        try:
            from ._cython.rl_core import RLSearchContext

            self._rl_context_class = RLSearchContext
            self._capabilities.rl_available = True
        except ImportError:
            self._rl_context_class = None

        # MLIR module
        try:
            import sys
            from pathlib import Path

            mlir_path = Path(__file__).parent.parent.parent / "mlir" / "python"
            sys.path.insert(0, str(mlir_path))
            from mugraph_to_mlir import MuGraphToMLIR

            self._mlir_converter = MuGraphToMLIR
            self._capabilities.mlir_available = True
        except ImportError:
            self._mlir_converter = None

        # Profiler
        try:
            from .profiler.hardware import HardwareProfiler

            self._profiler_class = HardwareProfiler
            self._capabilities.profiler_available = True
        except ImportError:
            self._profiler_class = None

    @property
    def capabilities(self) -> CoreCapabilities:
        """Get current core capabilities."""
        return self._capabilities

    # ============================================================
    # Graph Operations
    # ============================================================

    def create_graph(self) -> Any:
        """Create a new kernel graph."""
        if not self._capabilities.kernel_available:
            raise RuntimeError("Kernel module not available")

        try:
            from . import new_kernel_graph

            return new_kernel_graph()
        except Exception as e:
            raise RuntimeError(f"Failed to create graph: {e}")

    def load_graph(self, path: str) -> Any:
        """Load graph from JSON file."""
        if not self._capabilities.core_available:
            raise RuntimeError("Core module not available")

        try:
            return self._core.cy_from_json(path)
        except Exception as e:
            raise RuntimeError(f"Failed to load graph: {e}")

    def save_graph(self, graph: Any, path: str):
        """Save graph to JSON file."""
        if not self._capabilities.core_available:
            raise RuntimeError("Core module not available")

        try:
            self._core.cy_to_json(graph, path)
        except Exception as e:
            raise RuntimeError(f"Failed to save graph: {e}")

    # ============================================================
    # Optimization Operations
    # ============================================================

    def optimize(
        self, graph: Any, backend: str = "auto", use_ray: bool = True, **kwargs
    ) -> List[Any]:
        """
        Run superoptimization on graph.

        Args:
            graph: Kernel graph to optimize
            backend: Target backend
            use_ray: Use Ray for distributed search
            **kwargs: Additional search parameters

        Returns:
            List of optimized graphs (best first)
        """
        if not self._capabilities.kernel_available:
            raise RuntimeError("Kernel module not available")

        if hasattr(graph, "superoptimize"):
            return graph.superoptimize(backend=backend, use_ray=use_ray, **kwargs)
        else:
            raise TypeError("Graph does not support superoptimize")

    # ============================================================
    # RL Operations
    # ============================================================

    def create_rl_context(
        self,
        target_graph: Union[Any, str, Dict],
        backend: str = "cuda",
        gpu_id: int = 0,
    ) -> Any:
        """
        Create RL search context.

        Args:
            target_graph: Target computation graph
            backend: Target backend for verification
            gpu_id: GPU to use for verification

        Returns:
            RLSearchContext instance
        """
        if not self._capabilities.rl_available:
            raise RuntimeError("RL module not available")

        # Convert to JSON if needed
        if isinstance(target_graph, str):
            if os.path.exists(target_graph):
                with open(target_graph) as f:
                    graph_json = f.read()
            else:
                graph_json = target_graph
        elif isinstance(target_graph, dict):
            graph_json = json.dumps(target_graph)
        else:
            # Assume it's a graph object with to_json
            if hasattr(target_graph, "to_json"):
                graph_json = target_graph.to_json()
            else:
                raise TypeError(f"Unsupported target_graph type: {type(target_graph)}")

        return self._rl_context_class(graph_json, backend, gpu_id)

    def run_rl_episode(
        self,
        context: Any,
        policy: Callable[[Dict], int],
        max_steps: int = 100,
    ) -> Dict[str, Any]:
        """
        Run a single RL episode.

        Args:
            context: RL search context
            policy: Function that takes state dict and returns action
            max_steps: Maximum steps per episode

        Returns:
            Episode results including best kernel found
        """
        if not self._capabilities.rl_available:
            raise RuntimeError("RL module not available")

        context.reset()

        results = {
            "steps": 0,
            "verified": False,
            "best_latency_ms": float("inf"),
            "kernels": [],
        }

        for step in range(max_steps):
            # Get state
            state = context.get_state()

            if context.is_done():
                break

            # Get action from policy
            action = policy(state)

            # Apply action
            config = self._decode_action(action, state)
            success = context.apply_action(action, config)

            if not success:
                continue

            # Verify
            verify_result = context.verify()

            if verify_result.get("verified", False):
                results["verified"] = True

                # Profile
                profile_result = context.profile()
                latency = profile_result.get("latency_ms", float("inf"))

                if latency < results["best_latency_ms"]:
                    results["best_latency_ms"] = latency

                results["kernels"].append(
                    {
                        "kernel": context.get_kernel_graph(),
                        "latency_ms": latency,
                    }
                )

            results["steps"] = step + 1

        return results

    def _decode_action(self, action: int, state: Dict) -> Dict:
        """Decode action integer to configuration dict."""
        # Default configuration
        return {
            "grid_dim_x": 1,
            "grid_dim_y": 1,
            "grid_dim_z": 1,
            "block_dim_x": 128,
            "block_dim_y": 1,
            "block_dim_z": 1,
        }

    # ============================================================
    # MLIR Operations
    # ============================================================

    def to_mlir(self, graph: Any) -> str:
        """
        Convert graph to MLIR representation.

        Args:
            graph: Kernel graph

        Returns:
            MLIR code string
        """
        if not self._capabilities.mlir_available:
            raise RuntimeError("MLIR module not available")

        converter = self._mlir_converter()
        return converter.convert(graph)

    # ============================================================
    # Code Generation
    # ============================================================

    def generate_code(
        self,
        graph: Any,
        target: str = "cuda",
        profiling: bool = False,
    ) -> str:
        """
        Generate executable code from graph.

        Args:
            graph: Optimized kernel graph
            target: Target backend (cuda, metal, cpu, etc.)
            profiling: Include profiling instrumentation

        Returns:
            Generated code string
        """
        if hasattr(graph, "generate_code"):
            return graph.generate_code(target=target, profiling=profiling)
        else:
            raise TypeError("Graph does not support code generation")

    # ============================================================
    # Profiling Operations
    # ============================================================

    def profile(
        self,
        code: Union[str, Callable],
        inputs: List[Any],
        warmup: int = 10,
        iterations: int = 100,
    ) -> Dict[str, float]:
        """
        Profile kernel execution.

        Args:
            code: Compiled code or callable
            inputs: Input tensors
            warmup: Warmup iterations
            iterations: Profile iterations

        Returns:
            Profiling results
        """
        if not self._capabilities.profiler_available:
            # Fallback: simple timing
            import time

            if callable(code):
                # Warmup
                for _ in range(warmup):
                    code(*inputs)

                # Measure
                start = time.perf_counter()
                for _ in range(iterations):
                    code(*inputs)
                elapsed = time.perf_counter() - start

                return {
                    "latency_ms": (elapsed / iterations) * 1000,
                    "total_time_ms": elapsed * 1000,
                    "iterations": iterations,
                }

            raise RuntimeError("Cannot profile non-callable code without profiler")

        profiler = self._profiler_class()
        return profiler.profile(code, inputs, warmup, iterations)


# ============================================================
# Convenience Functions
# ============================================================


def get_core_bridge() -> CoreBridge:
    """Get the global CoreBridge instance."""
    return CoreBridge.get_instance()


def get_capabilities() -> CoreCapabilities:
    """Get current core capabilities."""
    return get_core_bridge().capabilities


def is_core_available() -> bool:
    """Check if C++ core is available."""
    return get_capabilities().core_available


def is_rl_available() -> bool:
    """Check if RL core is available."""
    return get_capabilities().rl_available


def is_mlir_available() -> bool:
    """Check if MLIR is available."""
    return get_capabilities().mlir_available
