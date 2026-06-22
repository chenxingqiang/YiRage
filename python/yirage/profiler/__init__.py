# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0

"""
YiRage Profiler Module.

Provides performance profiling for kernels and persistent kernels.

Profiler Types:
- HardwareProfiler: Multi-backend performance measurement (CUDA, MPS, CPU)
- TritonProfiler: Triton kernel profiling with autotuning
- Persistent kernel profiling: Perfetto trace export

Usage:
    from yirage.profiler import HardwareProfiler, ProfileConfig
    
    profiler = HardwareProfiler(backend="cuda")
    result = profiler.benchmark(kernel_func)
    print(f"Latency: {result.mean_ms:.4f} ± {result.std_ms:.4f} ms")
"""

from .hardware import (
    ProfilerBackend,
    HardwareProfiler,
    ProfileConfig,
    TimingResult,
    HardwareCounters,
    TrainingDataCollector,
    TrainingBenchmarkResult,
    check_google_benchmark_available,
    get_google_benchmark_install_instructions,
)

# Perfetto trace export (optional, requires torch and tg4perfetto)
try:
    from .base import (
        export_to_perfetto_trace,
        EventType,
        decode_tag,
        event_name_list,
    )

    PERFETTO_AVAILABLE = True
except (ImportError, TypeError):
    PERFETTO_AVAILABLE = False
    export_to_perfetto_trace = None
    EventType = None
    decode_tag = None
    event_name_list = None

# Persistent kernel trace export (optional)
try:
    from .persistent import (
        export_to_perfetto_trace as export_pk_trace,
    )
except (ImportError, TypeError):
    export_pk_trace = None

# Triton profiler (optional, requires triton)
try:
    from .triton import (
        TritonProfiler,
        profile_and_select_best_graph,
    )

    TRITON_PROFILER_AVAILABLE = True
except ImportError:
    TRITON_PROFILER_AVAILABLE = False
    TritonProfiler = None
    profile_and_select_best_graph = None


def create_profiler(
    backend: str = "auto",
    config: ProfileConfig = None,
) -> HardwareProfiler:
    """
    Create a hardware profiler for the specified backend.

    Args:
        backend: Backend type ("cuda", "mps", "cpu", "ascend", "maca", "auto")
        config: Optional profiler configuration

    Returns:
        HardwareProfiler instance
    """
    return HardwareProfiler(backend=backend, config=config)


__all__ = [
    # Hardware profiler (always available)
    "ProfilerBackend",
    "HardwareProfiler",
    "ProfileConfig",
    "TimingResult",
    "HardwareCounters",
    "create_profiler",
    # Training data collection
    "TrainingDataCollector",
    "TrainingBenchmarkResult",
    # Google Benchmark integration
    "check_google_benchmark_available",
    "get_google_benchmark_install_instructions",
    # Perfetto trace export (optional)
    "export_to_perfetto_trace",
    "export_pk_trace",
    "EventType",
    "decode_tag",
    "event_name_list",
    "PERFETTO_AVAILABLE",
    # Triton profiler (optional)
    "TritonProfiler",
    "profile_and_select_best_graph",
    "TRITON_PROFILER_AVAILABLE",
]
