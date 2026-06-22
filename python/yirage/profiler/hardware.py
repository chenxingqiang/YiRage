"""
Hardware Profiler for Training Data Collection

Provides precise performance measurements for different hardware backends.
Integrates with Google Benchmark (C++) for CPU measurements and uses
hardware-specific APIs for GPU/accelerator measurements.

Key Features:
- Multi-backend support (CUDA, MPS, CPU, Ascend, MACA)
- Statistical analysis (mean, std, min, max, percentiles)
- Hardware counter collection (where available)
- Training-ready data output format
"""

import time
import statistics
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Callable, Tuple
from enum import Enum
import logging
import os

logger = logging.getLogger(__name__)


class ProfilerBackend(str, Enum):
    """Profiler backend types."""

    CUDA = "cuda"
    MPS = "mps"
    CPU = "cpu"
    ASCEND = "ascend"
    MACA = "maca"


@dataclass
class TimingResult:
    """
    Comprehensive timing result with statistical analysis.

    This provides training-quality performance data.
    """

    # Primary metrics
    mean_ms: float = 0.0
    std_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0

    # Percentiles
    p50_ms: float = 0.0
    p90_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0

    # Iteration info
    num_iterations: int = 0
    num_warmup: int = 0

    # All measurements (for training)
    all_latencies_ms: List[float] = field(default_factory=list)

    # Coefficient of variation
    cv: float = 0.0  # std / mean

    # Hardware counters (where available)
    cpu_cycles: int = 0
    cache_misses: int = 0
    instructions: int = 0

    # Throughput
    throughput_ops_per_sec: float = 0.0

    # Validity
    is_valid: bool = True
    error_message: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_latencies(cls, latencies_ms: List[float], num_warmup: int = 0) -> "TimingResult":
        """Create TimingResult from a list of latency measurements.

        The first ``num_warmup`` measurements (in original order) are excluded
        from all statistics but are still stored in ``all_latencies_ms``.
        """
        if not latencies_ms:
            return cls(is_valid=False, error_message="No measurements")

        measured = latencies_ms[num_warmup:] if num_warmup > 0 else latencies_ms
        if not measured:
            return cls(is_valid=False, error_message="All measurements are warmup")

        sorted_lat = sorted(measured)
        n = len(sorted_lat)

        result = cls(
            mean_ms=statistics.mean(sorted_lat),
            std_ms=statistics.stdev(sorted_lat) if n > 1 else 0.0,
            min_ms=sorted_lat[0],
            max_ms=sorted_lat[-1],
            p50_ms=sorted_lat[n // 2],
            p90_ms=sorted_lat[int(n * 0.9)],
            p95_ms=sorted_lat[int(n * 0.95)],
            p99_ms=sorted_lat[int(n * 0.99)] if n >= 100 else sorted_lat[-1],
            num_iterations=n,
            num_warmup=num_warmup,
            all_latencies_ms=latencies_ms,
            is_valid=True,
        )

        if result.mean_ms > 0:
            result.cv = result.std_ms / result.mean_ms
            result.throughput_ops_per_sec = 1000.0 / result.mean_ms

        return result


@dataclass
class HardwareCounters:
    """Hardware performance counters."""

    # CPU counters
    cpu_cycles: int = 0
    instructions: int = 0
    cache_references: int = 0
    cache_misses: int = 0
    branch_instructions: int = 0
    branch_misses: int = 0

    # GPU counters (CUDA)
    sm_efficiency: float = 0.0
    achieved_occupancy: float = 0.0
    dram_read_throughput_gbps: float = 0.0
    dram_write_throughput_gbps: float = 0.0

    # Memory
    peak_memory_mb: float = 0.0
    memory_allocated_mb: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ProfileConfig:
    """Profiler configuration."""

    # Iteration settings
    num_warmup: int = 10
    num_iterations: int = 100

    # Adaptive iteration (like Google Benchmark)
    min_time_ms: float = 100.0  # Minimum total time to run
    adaptive_iterations: bool = True
    max_iterations: int = 10000

    # Statistical requirements
    target_cv: float = 0.05  # Target coefficient of variation
    max_cv_iterations: int = 5  # Max attempts to achieve target CV

    # Hardware counters
    collect_hw_counters: bool = False

    # Output
    save_all_latencies: bool = True

    def to_dict(self) -> Dict:
        return asdict(self)


class HardwareProfiler:
    """
    Multi-backend hardware profiler for training data collection.

    Provides Google Benchmark-quality measurements across different
    hardware platforms.

    Example:
        profiler = HardwareProfiler(backend="mps")

        def kernel_func():
            # Code to benchmark
            result = model(input_tensor)
            return result

        timing = profiler.benchmark(kernel_func)
        print(f"Mean: {timing.mean_ms:.4f} ± {timing.std_ms:.4f} ms")
    """

    def __init__(
        self,
        backend: str = "auto",
        config: Optional[ProfileConfig] = None,
    ):
        """
        Initialize the hardware profiler.

        Args:
            backend: Hardware backend (cuda, mps, cpu, ascend, maca, auto)
            config: Profiler configuration
        """
        self.config = config or ProfileConfig()
        self.backend = self._detect_backend(backend)
        self._setup_backend()

    def _detect_backend(self, backend: str) -> ProfilerBackend:
        """Detect and validate the backend."""
        if backend == "auto":
            try:
                import torch

                if torch.cuda.is_available():
                    return ProfilerBackend.CUDA
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    return ProfilerBackend.MPS
            except ImportError:
                pass
            return ProfilerBackend.CPU

        return ProfilerBackend(backend.lower())

    def _setup_backend(self):
        """Setup backend-specific resources."""
        if self.backend == ProfilerBackend.CUDA:
            import torch

            self._cuda_start_event = torch.cuda.Event(enable_timing=True)
            self._cuda_end_event = torch.cuda.Event(enable_timing=True)
        elif self.backend == ProfilerBackend.ASCEND:
            try:
                import torch_npu
                import torch

                self._npu_start_event = torch.npu.Event(enable_timing=True)
                self._npu_end_event = torch.npu.Event(enable_timing=True)
            except ImportError:
                logger.warning("torch_npu not available, falling back to CPU timing")
                self.backend = ProfilerBackend.CPU

    def _get_time_ns(self) -> int:
        """Get current time in nanoseconds."""
        return time.perf_counter_ns()

    def _synchronize(self):
        """Synchronize the device."""
        if self.backend == ProfilerBackend.CUDA:
            import torch

            torch.cuda.synchronize()
        elif self.backend == ProfilerBackend.MPS:
            import torch

            if hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()
        elif self.backend == ProfilerBackend.ASCEND:
            import torch

            torch.npu.synchronize()

    def _time_single_cuda(self, func: Callable) -> float:
        """Time a single execution using CUDA events."""
        import torch

        self._cuda_start_event.record()
        func()
        self._cuda_end_event.record()
        torch.cuda.synchronize()

        return self._cuda_start_event.elapsed_time(self._cuda_end_event)

    def _time_single_npu(self, func: Callable) -> float:
        """Time a single execution using NPU events."""
        import torch

        self._npu_start_event.record()
        func()
        self._npu_end_event.record()
        torch.npu.synchronize()

        return self._npu_start_event.elapsed_time(self._npu_end_event)

    def _time_single_cpu(self, func: Callable) -> float:
        """Time a single execution using high-resolution CPU timer."""
        self._synchronize()

        start_ns = self._get_time_ns()
        func()
        self._synchronize()
        end_ns = self._get_time_ns()

        return (end_ns - start_ns) / 1_000_000  # Convert to ms

    def _time_single(self, func: Callable) -> float:
        """Time a single execution using the appropriate method."""
        if self.backend == ProfilerBackend.CUDA:
            return self._time_single_cuda(func)
        elif self.backend == ProfilerBackend.ASCEND:
            return self._time_single_npu(func)
        else:
            return self._time_single_cpu(func)

    def _determine_iterations(self, func: Callable) -> int:
        """
        Determine optimal number of iterations (Google Benchmark style).

        Runs enough iterations to achieve min_time_ms total runtime.
        """
        if not self.config.adaptive_iterations:
            return self.config.num_iterations

        # Warmup first
        for _ in range(min(3, self.config.num_warmup)):
            func()
        self._synchronize()

        # Time a few iterations to estimate
        start = self._get_time_ns()
        test_iters = 5
        for _ in range(test_iters):
            func()
        self._synchronize()
        elapsed_ms = (self._get_time_ns() - start) / 1_000_000

        # Calculate needed iterations
        per_iter_ms = elapsed_ms / test_iters
        if per_iter_ms > 0:
            needed = int(self.config.min_time_ms / per_iter_ms)
            return min(max(needed, self.config.num_iterations), self.config.max_iterations)

        return self.config.num_iterations

    def benchmark(
        self,
        func: Callable,
        name: str = "benchmark",
    ) -> TimingResult:
        """
        Benchmark a function with statistical analysis.

        Args:
            func: Function to benchmark (no arguments, captures state via closure)
            name: Name for logging

        Returns:
            TimingResult with comprehensive statistics
        """
        try:
            # Determine number of iterations
            num_iters = self._determine_iterations(func)

            # Warmup
            for _ in range(self.config.num_warmup):
                func()
            self._synchronize()

            # Collect measurements
            latencies = []

            for _ in range(num_iters):
                lat = self._time_single(func)
                latencies.append(lat)

            # Check if we need more iterations for stability
            result = TimingResult.from_latencies(latencies, self.config.num_warmup)

            # Adaptive refinement if CV is too high
            if (
                self.config.adaptive_iterations
                and result.cv > self.config.target_cv
                and num_iters < self.config.max_iterations
            ):

                for attempt in range(self.config.max_cv_iterations):
                    # Add more iterations
                    additional = min(num_iters, self.config.max_iterations - len(latencies))
                    for _ in range(additional):
                        lat = self._time_single(func)
                        latencies.append(lat)

                    result = TimingResult.from_latencies(latencies, self.config.num_warmup)

                    if result.cv <= self.config.target_cv:
                        break

            # Optionally clear all latencies to save memory
            if not self.config.save_all_latencies:
                result.all_latencies_ms = []

            logger.debug(
                f"{name}: {result.mean_ms:.4f} ± {result.std_ms:.4f} ms "
                f"(n={result.num_iterations}, CV={result.cv:.2%})"
            )

            return result

        except Exception as e:
            return TimingResult(
                is_valid=False,
                error_message=str(e),
            )

    def benchmark_multiple(
        self,
        funcs: Dict[str, Callable],
    ) -> Dict[str, TimingResult]:
        """
        Benchmark multiple functions.

        Args:
            funcs: Dictionary mapping names to functions

        Returns:
            Dictionary mapping names to TimingResults
        """
        results = {}
        for name, func in funcs.items():
            results[name] = self.benchmark(func, name)
        return results

    def collect_hw_counters(self) -> HardwareCounters:
        """
        Collect hardware performance counters (where available).

        Note: Requires appropriate permissions and hardware support.
        """
        counters = HardwareCounters()

        if self.backend == ProfilerBackend.CUDA:
            try:
                import torch

                counters.peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
                counters.memory_allocated_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            except:
                pass

        elif self.backend == ProfilerBackend.MPS:
            try:
                import torch

                if hasattr(torch.mps, "current_allocated_memory"):
                    counters.memory_allocated_mb = torch.mps.current_allocated_memory() / (
                        1024 * 1024
                    )
            except:
                pass

        return counters


@dataclass
class TrainingBenchmarkResult:
    """
    Complete benchmark result for training data.

    Includes timing, hardware info, and configuration context.
    """

    # Identification
    benchmark_id: str = ""
    name: str = ""

    # Timing
    timing: TimingResult = field(default_factory=TimingResult)

    # Hardware context
    backend: str = ""
    device_name: str = ""
    hardware_counters: HardwareCounters = field(default_factory=HardwareCounters)

    # Configuration context
    config: Dict = field(default_factory=dict)

    # Metadata
    timestamp: str = ""
    yirage_version: str = ""

    def to_dict(self) -> Dict:
        return {
            "benchmark_id": self.benchmark_id,
            "name": self.name,
            "timing": self.timing.to_dict(),
            "backend": self.backend,
            "device_name": self.device_name,
            "hardware_counters": self.hardware_counters.to_dict(),
            "config": self.config,
            "timestamp": self.timestamp,
            "yirage_version": self.yirage_version,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class TrainingDataCollector:
    """
    Collects high-quality benchmark data for ML model training.

    Example:
        collector = TrainingDataCollector(backend="mps")

        # Benchmark a kernel configuration
        result = collector.benchmark_kernel(
            kernel_func=lambda: kernel(inputs),
            config={"grid_dim": (8, 1, 1), "block_dim": (64, 1, 1)},
            name="matmul_config_1",
        )

        # Export all data
        collector.export("training_data.jsonl")
    """

    def __init__(
        self,
        backend: str = "auto",
        profile_config: Optional[ProfileConfig] = None,
    ):
        self.profiler = HardwareProfiler(backend, profile_config)
        self.results: List[TrainingBenchmarkResult] = []
        self._result_counter = 0

    def benchmark_kernel(
        self,
        kernel_func: Callable,
        config: Dict,
        name: str = "",
    ) -> TrainingBenchmarkResult:
        """
        Benchmark a kernel with full training data collection.

        Args:
            kernel_func: Kernel function to benchmark
            config: Configuration dict (grid_dim, block_dim, etc.)
            name: Benchmark name

        Returns:
            Complete benchmark result for training
        """
        import time
        from datetime import datetime

        self._result_counter += 1

        # Perform benchmark
        timing = self.profiler.benchmark(kernel_func, name)

        # Collect hardware counters
        hw_counters = self.profiler.collect_hw_counters()

        # Get device name
        device_name = self._get_device_name()

        # Get version
        yirage_version = ""
        try:
            import yirage

            yirage_version = getattr(yirage, "__version__", "")
        except:
            pass

        result = TrainingBenchmarkResult(
            benchmark_id=f"bench_{self._result_counter}_{int(time.time())}",
            name=name or f"kernel_{self._result_counter}",
            timing=timing,
            backend=self.profiler.backend.value,
            device_name=device_name,
            hardware_counters=hw_counters,
            config=config,
            timestamp=datetime.now().isoformat(),
            yirage_version=yirage_version,
        )

        self.results.append(result)
        return result

    def _get_device_name(self) -> str:
        """Get the device name."""
        backend = self.profiler.backend

        if backend == ProfilerBackend.CUDA:
            try:
                import torch

                return torch.cuda.get_device_name(0)
            except:
                pass
        elif backend == ProfilerBackend.MPS:
            return "Apple Silicon MPS"
        elif backend == ProfilerBackend.CPU:
            import platform

            return platform.processor() or "CPU"

        return ""

    def benchmark_candidates(
        self,
        kernel_factory: Callable[[Dict], Callable],
        candidates: List[Dict],
        name_prefix: str = "candidate",
    ) -> List[TrainingBenchmarkResult]:
        """
        Benchmark multiple candidate configurations.

        Args:
            kernel_factory: Function that takes config and returns kernel callable
            candidates: List of configuration dicts
            name_prefix: Prefix for benchmark names

        Returns:
            List of benchmark results
        """
        results = []

        for i, config in enumerate(candidates):
            name = f"{name_prefix}_{i}"

            try:
                kernel_func = kernel_factory(config)
                result = self.benchmark_kernel(kernel_func, config, name)
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to benchmark {name}: {e}")
                # Add failed result
                results.append(
                    TrainingBenchmarkResult(
                        name=name,
                        config=config,
                        timing=TimingResult(is_valid=False, error_message=str(e)),
                        backend=self.profiler.backend.value,
                    )
                )

        return results

    def export(
        self,
        output_path: str,
        format: str = "jsonl",
    ):
        """
        Export all collected results.

        Args:
            output_path: Output file path
            format: Output format (jsonl, json)
        """
        with open(output_path, "w") as f:
            if format == "jsonl":
                for result in self.results:
                    f.write(json.dumps(result.to_dict()) + "\n")
            else:
                json.dump([r.to_dict() for r in self.results], f, indent=2)

        logger.info(f"Exported {len(self.results)} benchmark results to {output_path}")

    def get_training_dataset(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Get features and labels for training.

        Returns:
            Tuple of (features_list, labels_list)
        """
        features = []
        labels = []

        for result in self.results:
            if not result.timing.is_valid:
                continue

            # Extract features from config
            feat = {
                "backend": result.backend,
                "device": result.device_name,
                **result.config,
            }

            # Labels are the timing statistics
            label = {
                "mean_ms": result.timing.mean_ms,
                "std_ms": result.timing.std_ms,
                "min_ms": result.timing.min_ms,
                "p50_ms": result.timing.p50_ms,
                "p95_ms": result.timing.p95_ms,
                "p99_ms": result.timing.p99_ms,
            }

            features.append(feat)
            labels.append(label)

        return features, labels


# =============================================================================
# Google Benchmark Integration (C++ side)
# =============================================================================


def check_google_benchmark_available() -> bool:
    """Check if Google Benchmark C++ library is available."""
    try:
        # Try to import Cython bindings if available
        from yirage._cython import benchmark_core

        return True
    except ImportError:
        pass

    # Check for library file
    import platform

    lib_ext = ".dylib" if platform.system() == "Darwin" else ".so"

    possible_paths = [
        "/usr/local/lib/libbenchmark.a",
        "/usr/local/lib/libbenchmark" + lib_ext,
        "/opt/homebrew/lib/libbenchmark.a",
        os.path.expanduser("~/.local/lib/libbenchmark.a"),
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return True

    return False


def get_google_benchmark_install_instructions() -> str:
    """Get installation instructions for Google Benchmark."""
    import platform

    system = platform.system()

    if system == "Darwin":
        return """
# macOS Installation (Homebrew)
brew install google-benchmark

# Or from source:
git clone https://github.com/google/benchmark.git
cd benchmark
cmake -E make_directory "build"
cmake -DBENCHMARK_DOWNLOAD_DEPENDENCIES=on -DCMAKE_BUILD_TYPE=Release -S . -B "build"
cmake --build "build" --config Release
sudo cmake --build "build" --config Release --target install
"""
    elif system == "Linux":
        return """
# Ubuntu/Debian
sudo apt-get install libbenchmark-dev

# Or from source:
git clone https://github.com/google/benchmark.git
cd benchmark
cmake -E make_directory "build"
cmake -DBENCHMARK_DOWNLOAD_DEPENDENCIES=on -DCMAKE_BUILD_TYPE=Release -S . -B "build"
cmake --build "build" --config Release
sudo cmake --build "build" --config Release --target install
"""
    else:
        return """
# Build from source:
git clone https://github.com/google/benchmark.git
cd benchmark
cmake -E make_directory "build"
cmake -DBENCHMARK_DOWNLOAD_DEPENDENCIES=on -DCMAKE_BUILD_TYPE=Release -S . -B "build"
cmake --build "build" --config Release
cmake --build "build" --config Release --target install
"""
