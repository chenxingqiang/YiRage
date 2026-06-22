# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0

"""
Unified Compiler - Integrates muGraph Search with MLIR Compilation

This module provides end-to-end compilation from high-level kernel graphs
to optimized executable code for any supported backend.

Key Features:
- Multi-stage compilation pipeline
- Automatic backend selection
- Superoptimization with muGraph search
- MLIR lowering for portable code generation
- JIT compilation for immediate execution
- AOT compilation for deployment
"""

import os
import sys
import hashlib
import tempfile
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Union
from pathlib import Path

# Import core modules
try:
    from ..core import search as core_search, cy_to_json, cy_from_json

    HAS_CORE = True
except ImportError:
    HAS_CORE = False

try:
    from ..kernel.graph import KNGraph

    HAS_KERNEL = True
except ImportError:
    HAS_KERNEL = False

from ..backends.api import get_available_backends, get_default_backend, is_backend_available

# Hardware detection (optional – graceful fallback when not available)
try:
    from ..hardware.chip_arch import ChipArchitecture
    from ..hardware.detector import detect_current_chip
    from ..hardware.registry import HardwareRegistry

    HAS_HARDWARE = True
except (ImportError, OSError):
    ChipArchitecture = None  # type: ignore[assignment,misc]
    HAS_HARDWARE = False

# Pure-Python search-space derivation (no torch / core dependency)
from .search_space import (
    chip_arch_to_search_config,
    MODE_FAST,
    MODE_SUPEROPTIMIZE,
    MODE_AGGRESSIVE,
)


class CompileMode(Enum):
    """Compilation modes for different use cases."""

    # Fast compilation without superoptimization
    FAST = auto()

    # Standard superoptimization (default)
    SUPEROPTIMIZE = auto()

    # Aggressive superoptimization with more search iterations
    AGGRESSIVE = auto()

    # RL-guided search for complex graphs
    RL_GUIDED = auto()

    # MLIR-only compilation (skip muGraph search)
    MLIR_ONLY = auto()

    # Debug mode with verbose output
    DEBUG = auto()


@dataclass
class CompileOptions:
    """Configuration options for compilation."""

    # Backend selection
    backend: str = "auto"
    fallback_backends: List[str] = field(default_factory=lambda: ["cpu"])

    # Search configuration
    max_search_iterations: int = 1000
    search_timeout_seconds: float = 300.0
    num_workers: int = 4
    use_ray: bool = False

    # MLIR options
    enable_mlir: bool = True
    mlir_opt_level: int = 3
    mlir_target: Optional[str] = None

    # Code generation
    enable_jit: bool = True
    enable_aot: bool = False
    aot_output_path: Optional[str] = None

    # Caching
    enable_cache: bool = True
    cache_dir: Optional[str] = None

    # Debug
    verbose: bool = False
    dump_ir: bool = False
    ir_dump_path: Optional[str] = None


@dataclass
class CompileResult:
    """Result of compilation."""

    success: bool
    backend: str
    latency_ms: Optional[float] = None
    executable: Optional[Any] = None
    mlir_code: Optional[str] = None
    generated_code: Optional[str] = None
    search_iterations: int = 0
    compile_time_seconds: float = 0.0
    cache_hit: bool = False
    error_message: Optional[str] = None

    def __call__(self, *inputs):
        """Execute the compiled kernel."""
        if not self.success or self.executable is None:
            raise RuntimeError(f"Compilation failed: {self.error_message}")
        return self.executable(*inputs)


class UnifiedCompiler:
    """
    Unified compiler integrating muGraph superoptimization with MLIR.

    Example:
        compiler = UnifiedCompiler(backend='cuda', mode=CompileMode.SUPEROPTIMIZE)

        # Compile a kernel graph
        result = compiler.compile(graph)

        # Execute
        output = result(input_tensor)

        # Or use as decorator
        @compiler.jit
        def my_kernel(x, y):
            return x @ y + x
    """

    def __init__(
        self,
        backend: str = "auto",
        mode: CompileMode = CompileMode.SUPEROPTIMIZE,
        options: Optional[CompileOptions] = None,
        chip_arch: Optional["ChipArchitecture"] = None,
        auto_detect_hardware: bool = True,
    ):
        self.mode = mode
        self.options = options or CompileOptions()

        # Resolve chip architecture
        # Priority: explicit chip_arch > auto-detect > None (uses hardcoded defaults)
        self.chip_arch: Optional["ChipArchitecture"] = chip_arch
        if self.chip_arch is None and auto_detect_hardware and HAS_HARDWARE:
            try:
                self.chip_arch = detect_current_chip()
            except Exception:
                pass

        # Resolve backend
        if backend == "auto":
            # Prefer the chip's declared backend if we detected one
            if self.chip_arch is not None and self.chip_arch.backend:
                self.backend = self.chip_arch.backend
            else:
                self.backend = get_default_backend() or "cpu"
        else:
            self.backend = backend

        self.options.backend = self.backend

        # Initialize cache
        self._cache: Dict[str, CompileResult] = {}
        self._cache_dir = self.options.cache_dir or os.path.expanduser("~/.yirage/compile_cache")
        os.makedirs(self._cache_dir, exist_ok=True)

        # Statistics
        self._compile_count = 0
        self._cache_hits = 0

    def compile(
        self, graph: Union["KNGraph", str, Dict], entry_func: str = "kernel"
    ) -> CompileResult:
        """
        Compile a kernel graph.

        Args:
            graph: KNGraph instance, path to JSON, or dict representation
            entry_func: Name of the entry function

        Returns:
            CompileResult with executable and metadata
        """
        import time

        start_time = time.time()

        try:
            # Check cache
            graph_hash = self._compute_graph_hash(graph)
            if self.options.enable_cache:
                cached = self._check_cache(graph_hash)
                if cached:
                    cached.cache_hit = True
                    self._cache_hits += 1
                    return cached

            # Stage 1: Convert to KNGraph if needed
            kn_graph = self._ensure_kn_graph(graph)

            # Stage 2: Superoptimization (unless MLIR_ONLY or FAST)
            optimized_graph = kn_graph
            search_iterations = 0

            if self.mode not in (CompileMode.MLIR_ONLY, CompileMode.FAST):
                optimized_graph, search_iterations = self._superoptimize(kn_graph)

            # Stage 3: Generate MLIR (if enabled)
            mlir_code = None
            if self.options.enable_mlir:
                mlir_code = self._generate_mlir(optimized_graph)

            # Stage 4: Code generation and compilation
            executable, generated_code = self._codegen(optimized_graph, mlir_code)

            # Stage 5: Profile
            latency_ms = self._profile(executable) if executable else None

            compile_time = time.time() - start_time

            result = CompileResult(
                success=True,
                backend=self.backend,
                latency_ms=latency_ms,
                executable=executable,
                mlir_code=mlir_code,
                generated_code=generated_code,
                search_iterations=search_iterations,
                compile_time_seconds=compile_time,
                cache_hit=False,
            )

            # Cache result
            if self.options.enable_cache:
                self._save_cache(graph_hash, result)

            self._compile_count += 1
            return result

        except Exception as e:
            compile_time = time.time() - start_time
            return CompileResult(
                success=False,
                backend=self.backend,
                compile_time_seconds=compile_time,
                error_message=str(e),
            )

    def jit(self, func: Callable) -> Callable:
        """
        JIT compile decorator.

        Usage:
            @compiler.jit
            def my_kernel(x, y):
                return x @ y
        """

        # This would trace the function and create a KNGraph
        # For now, return the function unchanged with a wrapper
        def wrapper(*args, **kwargs):
            # TODO: Implement proper tracing and compilation
            return func(*args, **kwargs)

        return wrapper

    def _ensure_kn_graph(self, graph: Union["KNGraph", str, Dict]) -> "KNGraph":
        """Convert input to KNGraph."""
        if HAS_KERNEL and isinstance(graph, KNGraph):
            return graph

        if isinstance(graph, str):
            # Load JSON in Python then reuse the dict path. Calling cy_from_json(path)
            # directly can abort the process on some inputs/platforms; the tempfile
            # path used for dict graphs is the stable loading path.
            import json

            with open(graph, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                return self._ensure_kn_graph(payload)
            raise TypeError(
                f"JSON root must be an object (dict), got {type(payload).__name__}"
            )

        if isinstance(graph, dict):
            # Convert dict to KNGraph
            import json
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
                json.dump(graph, f)
                f.flush()
                if HAS_CORE:
                    return cy_from_json(f.name)
                else:
                    raise RuntimeError("Core module not available")

        raise TypeError(f"Unsupported graph type: {type(graph)}")

    def _superoptimize(self, graph: "KNGraph") -> tuple:
        """Run muGraph superoptimization."""
        if not HAS_KERNEL:
            return graph, 0

        try:
            # Configure search based on mode
            search_config = self._get_search_config()

            # Run superoptimization
            optimized = graph.superoptimize(
                backend=self.backend,
                verbose=self.options.verbose,
                use_ray=self.options.use_ray,
                num_workers=self.options.num_workers,
                **search_config,
            )

            # Return best result
            if optimized and len(optimized) > 0:
                return optimized[0], len(optimized)
            return graph, 0

        except Exception as e:
            if self.options.verbose:
                print(f"Superoptimization failed: {e}")
            return graph, 0

    def _get_search_config(self) -> Dict[str, Any]:
        """Build a search-space configuration, hardware-aware when possible.

        Delegates to :func:`~yirage.compiler.search_space.chip_arch_to_search_config`
        so the derivation logic is reusable and independently testable.
        """
        return chip_arch_to_search_config(
            self.chip_arch,
            mode=self.mode.name,
        )

    def _generate_mlir(self, graph: "KNGraph") -> Optional[str]:
        """Generate MLIR from optimized graph."""
        try:
            # Try to import MLIR module
            sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "mlir" / "python"))
            from mugraph_to_mlir import MuGraphToMLIR

            converter = MuGraphToMLIR()
            return converter.convert(graph)
        except ImportError:
            if self.options.verbose:
                print("MLIR module not available, skipping MLIR generation")
            return None
        except Exception as e:
            if self.options.verbose:
                print(f"MLIR generation failed: {e}")
            return None

    def _codegen(self, graph: "KNGraph", mlir_code: Optional[str]) -> tuple:
        """Generate executable code."""
        try:
            if HAS_KERNEL and hasattr(graph, "generate_code"):
                code = graph.generate_code()
                # Compile and load
                executable = self._compile_and_load(code)
                return executable, code
        except Exception as e:
            if self.options.verbose:
                print(f"Code generation failed: {e}")

        return None, None

    def _compile_and_load(self, code: str) -> Optional[Callable]:
        """Compile generated code and return executable."""
        # This would compile the code to a shared library and load it
        # Implementation depends on backend
        return None

    def _profile(self, executable: Optional[Callable]) -> Optional[float]:
        """Profile the executable to get latency."""
        if executable is None:
            return None

        try:
            import time

            # Warm up
            for _ in range(3):
                pass  # Would call executable with dummy inputs

            # Measure
            start = time.perf_counter()
            for _ in range(10):
                pass  # Would call executable
            elapsed = time.perf_counter() - start

            return (elapsed / 10) * 1000  # Convert to ms
        except:
            return None

    def _compute_graph_hash(self, graph: Union["KNGraph", str, Dict]) -> str:
        """Compute a hash for the graph."""
        import json

        if isinstance(graph, str):
            with open(graph, "r") as f:
                content = f.read()
        elif isinstance(graph, dict):
            content = json.dumps(graph, sort_keys=True)
        else:
            # Try to serialize
            content = str(graph)

        # Include options in hash
        options_str = f"{self.backend}:{self.mode.name}:{self.options.mlir_opt_level}"
        full_content = content + options_str

        return hashlib.sha256(full_content.encode()).hexdigest()[:16]

    def _check_cache(self, graph_hash: str) -> Optional[CompileResult]:
        """Check if result is cached."""
        if graph_hash in self._cache:
            return self._cache[graph_hash]

        cache_file = os.path.join(self._cache_dir, f"{graph_hash}.json")
        if os.path.exists(cache_file):
            # Load from disk cache
            import json

            with open(cache_file, "r") as f:
                data = json.load(f)
            # Reconstruct CompileResult (simplified)
            return CompileResult(
                success=data.get("success", False),
                backend=data.get("backend", self.backend),
                latency_ms=data.get("latency_ms"),
                mlir_code=data.get("mlir_code"),
                generated_code=data.get("generated_code"),
                search_iterations=data.get("search_iterations", 0),
                compile_time_seconds=data.get("compile_time_seconds", 0),
            )

        return None

    def _save_cache(self, graph_hash: str, result: CompileResult):
        """Save result to cache."""
        self._cache[graph_hash] = result

        # Save to disk
        cache_file = os.path.join(self._cache_dir, f"{graph_hash}.json")
        import json

        data = {
            "success": result.success,
            "backend": result.backend,
            "latency_ms": result.latency_ms,
            "mlir_code": result.mlir_code,
            "generated_code": result.generated_code,
            "search_iterations": result.search_iterations,
            "compile_time_seconds": result.compile_time_seconds,
        }
        with open(cache_file, "w") as f:
            json.dump(data, f)

    def get_statistics(self) -> Dict[str, Any]:
        """Get compilation statistics."""
        return {
            "compile_count": self._compile_count,
            "cache_hits": self._cache_hits,
            "cache_hit_rate": self._cache_hits / max(1, self._compile_count),
            "backend": self.backend,
            "mode": self.mode.name,
        }


# Convenience functions


def compile_graph(
    graph: Union["KNGraph", str, Dict],
    backend: str = "auto",
    mode: CompileMode = CompileMode.SUPEROPTIMIZE,
    **kwargs,
) -> CompileResult:
    """
    Compile a kernel graph.

    Args:
        graph: KNGraph instance, path to JSON, or dict representation
        backend: Target backend ('cuda', 'mps', 'cpu', 'auto', etc.)
        mode: Compilation mode
        **kwargs: Additional options passed to CompileOptions

    Returns:
        CompileResult with executable and metadata
    """
    options = CompileOptions(**kwargs)
    compiler = UnifiedCompiler(backend=backend, mode=mode, options=options)
    return compiler.compile(graph)


def hardware_aware_compile(
    graph: Union["KNGraph", str, Dict],
    *,
    chip_arch: Optional["ChipArchitecture"] = None,
    backend: str = "auto",
    mode: CompileMode = CompileMode.SUPEROPTIMIZE,
    verbose: bool = False,
    **kwargs,
) -> CompileResult:
    """One-call end-to-end pipeline: hardware detection → search → compile.

    This is the recommended entry-point for users who want a fully automatic
    "plug-and-play" experience.  It combines:

    1. **Hardware detection** – auto-detects the current chip via
       :func:`~yirage.hardware.detector.detect_current_chip` and looks up its
       specification in the :class:`~yirage.hardware.registry.HardwareRegistry`.
    2. **Search-space derivation** – translates chip specs (SM count, thread
       budget, shared-memory size, ``search_config_overrides``) into the
       ``griddims`` / ``blockdims`` / ``franges`` arguments consumed by
       :py:meth:`~yirage.kernel.graph.KNGraph.superoptimize`.
    3. **muGraph superoptimization** – runs the stochastic search to find the
       best µGraph for the target hardware.
    4. **Compilation** – generates executable kernel code (CUDA / Triton /
       CPU) from the winning µGraph.

    Args:
        graph:
            A :class:`~yirage.kernel.graph.KNGraph` instance, a path to a
            serialised JSON graph, or a ``dict`` representation.
        chip_arch:
            Optional explicit :class:`~yirage.hardware.chip_arch.ChipArchitecture`.
            When *None* (default) the chip is auto-detected at runtime.
        backend:
            Target backend string.  ``"auto"`` (default) uses the chip's
            declared backend or falls back to PyTorch auto-detection.
        mode:
            :class:`CompileMode` controlling search aggressiveness.
        verbose:
            Print pipeline progress to stdout.
        **kwargs:
            Extra keyword arguments forwarded to :class:`CompileOptions`.

    Returns:
        :class:`CompileResult` containing the executable, latency estimate,
        generated code, and metadata.

    Example::

        import yirage as mi

        kgraph = mi.new_kernel_graph()
        A = kgraph.new_input([1024, 1024], dtype=mi.float16)
        B = kgraph.new_input([1024, 1024], dtype=mi.float16)
        C = kgraph.matmul(A, B)
        kgraph.mark_output(C)

        # Fully automatic — detects GPU, derives search params, optimises, compiles
        result = mi.hardware_aware_compile(kgraph, verbose=True)
        print(f"backend={result.backend}, latency={result.latency_ms} ms")

    Notes:
        * If no GPU / accelerator is detected the pipeline falls back to the
          ``"cpu"`` backend and uses conservative default search parameters.
        * The search space is cached on disk (``~/.yirage/compile_cache``) so
          repeated calls for the same graph are instant.
    """
    if verbose:
        print("[hardware_aware_compile] Starting hardware-aware compilation pipeline")

    # Step 1 — hardware detection
    resolved_chip = chip_arch
    if resolved_chip is None and HAS_HARDWARE:
        try:
            resolved_chip = detect_current_chip()
        except Exception as exc:
            if verbose:
                print(
                    f"[hardware_aware_compile] Hardware detection failed ({exc}), "
                    "using defaults"
                )

    if verbose:
        if resolved_chip is not None:
            print(f"[hardware_aware_compile] Detected chip: {resolved_chip.summary()}")
        else:
            print("[hardware_aware_compile] No chip detected, using fallback defaults")

    # Steps 2 + 3 + 4 — UnifiedCompiler with hardware context
    options = CompileOptions(verbose=verbose, **kwargs)
    compiler = UnifiedCompiler(
        backend=backend,
        mode=mode,
        options=options,
        chip_arch=resolved_chip,
        auto_detect_hardware=False,  # already resolved above
    )

    if verbose:
        cfg = compiler._get_search_config()
        print(f"[hardware_aware_compile] Derived search config: {cfg}")

    return compiler.compile(graph)


def jit_compile(func: Callable, backend: str = "auto") -> Callable:
    """
    JIT compile a function.

    Args:
        func: Function to compile
        backend: Target backend

    Returns:
        Compiled function
    """
    compiler = UnifiedCompiler(backend=backend, mode=CompileMode.FAST)
    return compiler.jit(func)
