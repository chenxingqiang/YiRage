"""
YiRage Persistent Kernel Runtime Python Interface

This module provides Python bindings for the multi-backend persistent kernel
runtime, enabling LLM inference on CUDA, CPU, Ascend, MACA, and MPS backends.

Example usage:
    from yirage.pk_runtime import (
        PKRuntime, PKBackendType, PKMode,
        create_runtime, get_available_backends
    )
    
    # Get available backends
    backends = get_available_backends()
    
    # Create runtime with best backend
    runtime = create_runtime(
        backend=PKBackendType.AUTO,
        num_workers=4,
        num_schedulers=1
    )
    
    # Build and run task graph
    with runtime:
        runtime.add_task("embedding", inputs=[tokens], outputs=[embeddings])
        runtime.add_task("rms_norm", inputs=[embeddings], outputs=[normed])
        runtime.run()
"""

from enum import IntEnum
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
import threading
import queue
import time

# =============================================================================
# Enumerations (matching C++ pk_runtime_core.h)
# =============================================================================


class PKBackendType(IntEnum):
    """Supported backend types."""

    CUDA = 0
    CPU = 1
    MPS = 2
    ASCEND = 3
    MACA = 4
    TRITON = 5
    NKI = 6
    AUTO = 99  # Automatic selection

    @classmethod
    def from_name(cls, name: str) -> "PKBackendType":
        """Convert string name to backend type."""
        mapping = {
            "cuda": cls.CUDA,
            "cpu": cls.CPU,
            "mps": cls.MPS,
            "ascend": cls.ASCEND,
            "maca": cls.MACA,
            "triton": cls.TRITON,
            "nki": cls.NKI,
            "auto": cls.AUTO,
        }
        return mapping.get(name.lower(), cls.CPU)

    def to_name(self) -> str:
        """Convert backend type to string name."""
        return self.name.lower()


class PKMode(IntEnum):
    """Execution modes."""

    OFFLINE = 0  # Pre-compiled kernel graphs
    ONLINE = 1  # Persistent kernel loop for LLM serving
    ONEPASS = 2  # Single-pass execution
    EAGER = 3  # Immediate execution
    GRAPH = 4  # Execution graph capture and replay
    STREAMING = 5  # Multi-node streaming

    @classmethod
    def from_name(cls, name: str) -> "PKMode":
        """Convert string name to mode."""
        mapping = {
            "offline": cls.OFFLINE,
            "online": cls.ONLINE,
            "onepass": cls.ONEPASS,
            "eager": cls.EAGER,
            "graph": cls.GRAPH,
            "streaming": cls.STREAMING,
        }
        return mapping.get(name.lower(), cls.OFFLINE)


class PKTaskType(IntEnum):
    """Task types for persistent kernel."""

    TERMINATE = 0
    BEGIN_TASK_GRAPH = 10
    EMBEDDING = 101
    RMS_NORM = 119
    RMS_NORM_LINEAR = 102
    ATTENTION = 103
    ATTENTION_2 = 104
    LINEAR = 120
    LINEAR_RESIDUAL = 108
    SILU_MUL = 118
    SILU_MUL_LINEAR = 105
    ARGMAX = 109
    PAGED_ATTENTION_1 = 116
    PAGED_ATTENTION_2 = 117
    ALLREDUCE = 106
    REDUCE = 107


class PKEventType(IntEnum):
    """Event types for task synchronization."""

    EMPTY = 900
    LAUNCH_TASKS = 901
    LAUNCH_MASSIVE_TASKS = 902
    LAUNCH_DEPENDENT_TASKS = 903
    END_OF_TASK_GRAPH = 910
    TERMINATION = 911
    INVALID = 999


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class PKCapabilities:
    """Backend capabilities."""

    supports_tma: bool = False
    supports_tensor_cores: bool = False
    supports_async_copy: bool = False
    supports_nvshmem: bool = False
    supports_fp8: bool = False
    max_shared_memory: int = 0
    max_global_memory: int = 0
    compute_major: int = 0
    compute_minor: int = 0
    supported_modes: List[PKMode] = field(default_factory=list)


@dataclass
class PKRuntimeConfig:
    """Runtime configuration."""

    backend: PKBackendType = PKBackendType.CPU
    mode: PKMode = PKMode.EAGER
    device_id: int = 0
    num_workers: int = 4
    num_local_schedulers: int = 1
    num_remote_schedulers: int = 0
    per_worker_queue_len: int = 1024
    per_sched_queue_len: int = 1024
    max_seq_length: int = 4096
    max_batch_size: int = 16
    max_tokens: int = 64
    eos_token_id: int = 2
    profiling_enabled: bool = False


@dataclass
class PKTaskDesc:
    """Task descriptor."""

    task_type: PKTaskType
    variant_id: int = 0
    trigger_event: int = -1
    dependent_event: int = -1
    input_ptrs: List[Any] = field(default_factory=list)
    output_ptrs: List[Any] = field(default_factory=list)
    request_id: int = -1


@dataclass
class PKEventDesc:
    """Event descriptor."""

    event_type: PKEventType
    num_triggers: int = 1
    first_task_id: int = 0
    last_task_id: int = 0


# =============================================================================
# Backend Capability Matrix
# =============================================================================

BACKEND_CAPABILITIES = {
    PKBackendType.CUDA: PKCapabilities(
        supports_tma=True,
        supports_tensor_cores=True,
        supports_async_copy=True,
        supports_nvshmem=True,
        max_shared_memory=228 * 1024,
        supported_modes=[PKMode.OFFLINE, PKMode.ONLINE, PKMode.ONEPASS, PKMode.GRAPH],
    ),
    PKBackendType.CPU: PKCapabilities(
        supports_tma=False,
        supports_tensor_cores=False,
        supports_async_copy=True,
        max_shared_memory=0,
        supported_modes=[PKMode.EAGER, PKMode.GRAPH, PKMode.OFFLINE],
    ),
    PKBackendType.ASCEND: PKCapabilities(
        supports_tma=False,
        supports_tensor_cores=True,  # AI Core
        supports_async_copy=True,
        max_shared_memory=512 * 1024,
        supported_modes=[PKMode.OFFLINE, PKMode.ONLINE, PKMode.GRAPH],
    ),
    PKBackendType.MACA: PKCapabilities(
        supports_tma=False,
        supports_tensor_cores=True,
        supports_async_copy=True,
        max_shared_memory=64 * 1024,  # MetaX C500 64 KB/block (see maca::MAX_SMEM_SIZE)
        supported_modes=[PKMode.OFFLINE, PKMode.ONLINE, PKMode.ONEPASS],
    ),
    PKBackendType.MPS: PKCapabilities(
        supports_tma=False,
        supports_tensor_cores=False,
        supports_async_copy=True,
        max_shared_memory=32 * 1024,
        supported_modes=[PKMode.EAGER, PKMode.GRAPH],
    ),
}


# =============================================================================
# Worker/Scheduler Implementation (Python simulation)
# =============================================================================


class PKWorker(threading.Thread):
    """Worker thread for task execution."""

    def __init__(
        self, worker_id: int, task_queue: queue.Queue, event_counters: Dict[int, int], executor
    ):
        super().__init__(daemon=True)
        self.worker_id = worker_id
        self.task_queue = task_queue
        self.event_counters = event_counters
        self.executor = executor
        self.running = True
        self.lock = threading.Lock()

    def run(self):
        while self.running:
            try:
                task = self.task_queue.get(timeout=0.1)
                if task is None:
                    break

                # Wait for dependent event
                if task.dependent_event >= 0:
                    while self.event_counters.get(task.dependent_event, 0) < 1:
                        if not self.running:
                            return
                        time.sleep(0.0001)

                # Execute task
                if task.task_type == PKTaskType.TERMINATE:
                    break

                self.executor.execute(task)

                # Trigger completion event
                if task.trigger_event >= 0:
                    with self.lock:
                        self.event_counters[task.trigger_event] = (
                            self.event_counters.get(task.trigger_event, 0) + 1
                        )

                self.task_queue.task_done()
            except queue.Empty:
                continue

    def stop(self):
        self.running = False


class PKScheduler(threading.Thread):
    """Scheduler thread for task dispatch."""

    def __init__(
        self,
        sched_id: int,
        workers: List[PKWorker],
        event_queue: queue.Queue,
        task_descs: List[PKTaskDesc],
    ):
        super().__init__(daemon=True)
        self.sched_id = sched_id
        self.workers = workers
        self.event_queue = event_queue
        self.task_descs = task_descs
        self.running = True
        self.next_worker = 0

    def run(self):
        while self.running:
            try:
                event = self.event_queue.get(timeout=0.1)
                if event is None:
                    break

                if event.event_type == PKEventType.TERMINATION:
                    for worker in self.workers:
                        worker.task_queue.put(PKTaskDesc(PKTaskType.TERMINATE))
                    break

                # Dispatch tasks
                for task_id in range(event.first_task_id, event.last_task_id):
                    if task_id < len(self.task_descs):
                        task = self.task_descs[task_id]
                        worker = self.workers[self.next_worker]
                        worker.task_queue.put(task)
                        self.next_worker = (self.next_worker + 1) % len(self.workers)

                self.event_queue.task_done()
            except queue.Empty:
                continue

    def stop(self):
        self.running = False


# =============================================================================
# Task Executor
# =============================================================================


class PKTaskExecutor:
    """Task executor for various backends."""

    def __init__(self, backend: PKBackendType):
        self.backend = backend
        self.task_handlers = {
            PKTaskType.EMBEDDING: self._execute_embedding,
            PKTaskType.RMS_NORM: self._execute_rms_norm,
            PKTaskType.LINEAR: self._execute_linear,
            PKTaskType.ATTENTION: self._execute_attention,
            PKTaskType.SILU_MUL: self._execute_silu_mul,
            PKTaskType.ARGMAX: self._execute_argmax,
        }

    def execute(self, task: PKTaskDesc):
        """Execute a task."""
        handler = self.task_handlers.get(task.task_type)
        if handler:
            handler(task)

    def _execute_embedding(self, task: PKTaskDesc):
        """Execute embedding lookup."""
        pass

    def _execute_rms_norm(self, task: PKTaskDesc):
        """Execute RMS normalization."""
        pass

    def _execute_linear(self, task: PKTaskDesc):
        """Execute linear projection."""
        pass

    def _execute_attention(self, task: PKTaskDesc):
        """Execute attention."""
        pass

    def _execute_silu_mul(self, task: PKTaskDesc):
        """Execute SiLU activation."""
        pass

    def _execute_argmax(self, task: PKTaskDesc):
        """Execute argmax."""
        pass


# =============================================================================
# Persistent Kernel Runtime
# =============================================================================


class PKRuntime:
    """
    Multi-backend Persistent Kernel Runtime.

    This class manages the worker-scheduler execution model for
    LLM inference across different hardware backends.
    """

    def __init__(self, config: PKRuntimeConfig):
        """Initialize the runtime."""
        self.config = config
        self.tasks: List[PKTaskDesc] = []
        self.events: List[PKEventDesc] = []
        self.workers: List[PKWorker] = []
        self.schedulers: List[PKScheduler] = []
        self.event_counters: Dict[int, int] = {}
        self.task_queue = queue.Queue()
        self.event_queue = queue.Queue()
        self.executor = PKTaskExecutor(config.backend)
        self.initialized = False

    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finalize()
        return False

    def initialize(self) -> bool:
        """Initialize the runtime."""
        if self.initialized:
            return True

        # Create workers
        for i in range(self.config.num_workers):
            worker = PKWorker(i, self.task_queue, self.event_counters, self.executor)
            self.workers.append(worker)

        # Create schedulers
        for i in range(self.config.num_local_schedulers):
            scheduler = PKScheduler(i, self.workers, self.event_queue, self.tasks)
            self.schedulers.append(scheduler)

        self.initialized = True
        return True

    def add_task(
        self,
        task_type: PKTaskType,
        inputs: List[Any] = None,
        outputs: List[Any] = None,
        dependent_event: int = -1,
    ) -> int:
        """Add a task to the graph."""
        task_id = len(self.tasks)

        task = PKTaskDesc(
            task_type=task_type,
            trigger_event=task_id,
            dependent_event=dependent_event,
            input_ptrs=inputs or [],
            output_ptrs=outputs or [],
        )
        self.tasks.append(task)

        # Create corresponding event
        event = PKEventDesc(
            event_type=PKEventType.LAUNCH_TASKS,
            num_triggers=1,
            first_task_id=task_id,
            last_task_id=task_id + 1,
        )
        self.events.append(event)

        return task_id

    def run(self):
        """Execute the task graph."""
        if not self.initialized:
            self.initialize()

        # Start workers
        for worker in self.workers:
            worker.start()

        # Start schedulers
        for scheduler in self.schedulers:
            scheduler.start()

        # Queue initial event
        if self.events:
            self.event_queue.put(self.events[0])

        # Wait for completion
        for scheduler in self.schedulers:
            scheduler.join(timeout=30)

        for worker in self.workers:
            worker.join(timeout=30)

    def synchronize(self):
        """Wait for all tasks to complete."""
        self.task_queue.join()

    def finalize(self):
        """Cleanup and release resources."""
        # Stop workers
        for worker in self.workers:
            worker.stop()
            worker.task_queue.put(None)

        # Stop schedulers
        for scheduler in self.schedulers:
            scheduler.stop()
            scheduler.event_queue.put(None)

        self.workers.clear()
        self.schedulers.clear()
        self.initialized = False

    def get_capabilities(self) -> PKCapabilities:
        """Get backend capabilities."""
        return BACKEND_CAPABILITIES.get(self.config.backend, PKCapabilities())

    def get_supported_modes(self) -> List[PKMode]:
        """Get supported execution modes."""
        caps = self.get_capabilities()
        return caps.supported_modes


# =============================================================================
# Factory Functions
# =============================================================================


def get_available_backends() -> List[PKBackendType]:
    """Get list of available backends on this system."""
    available = [PKBackendType.CPU]  # CPU always available

    # Check CUDA / MACA (mcPytorch exposes torch.cuda API on MetaX)
    try:
        import os

        import torch

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            if "MetaX" in device_name or os.environ.get("YIRAGE_BACKEND", "").lower() == "maca":
                available.append(PKBackendType.MACA)
            else:
                available.append(PKBackendType.CUDA)
    except ImportError:
        pass

    # Check MPS (Apple Silicon)
    try:
        import torch

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            available.append(PKBackendType.MPS)
    except ImportError:
        pass

    return available


def get_best_backend() -> PKBackendType:
    """Get the best available backend."""
    available = get_available_backends()

    priority = [
        PKBackendType.CUDA,
        PKBackendType.MACA,
        PKBackendType.ASCEND,
        PKBackendType.MPS,
        PKBackendType.CPU,
    ]

    for backend in priority:
        if backend in available:
            return backend

    return PKBackendType.CPU


def create_runtime(
    backend: PKBackendType = PKBackendType.AUTO,
    mode: PKMode = None,
    device_id: int = 0,
    num_workers: int = 4,
    num_schedulers: int = 1,
    **kwargs,
) -> PKRuntime:
    """
    Create a persistent kernel runtime.

    Args:
        backend: Backend type (or AUTO for automatic selection)
        mode: Execution mode (None for backend default)
        device_id: Device ID for GPU backends
        num_workers: Number of worker threads/blocks
        num_schedulers: Number of scheduler threads
        **kwargs: Additional configuration

    Returns:
        PKRuntime instance
    """
    # Resolve AUTO backend
    if backend == PKBackendType.AUTO:
        backend = get_best_backend()

    # Get default mode for backend
    if mode is None:
        caps = BACKEND_CAPABILITIES.get(backend, PKCapabilities())
        mode = caps.supported_modes[0] if caps.supported_modes else PKMode.EAGER

    # Create configuration
    config = PKRuntimeConfig(
        backend=backend,
        mode=mode,
        device_id=device_id,
        num_workers=num_workers,
        num_local_schedulers=num_schedulers,
        **kwargs,
    )

    return PKRuntime(config)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "PKBackendType",
    "PKMode",
    "PKTaskType",
    "PKEventType",
    "PKCapabilities",
    "PKRuntimeConfig",
    "PKTaskDesc",
    "PKEventDesc",
    "PKRuntime",
    "PKWorker",
    "PKScheduler",
    "PKTaskExecutor",
    "get_available_backends",
    "get_best_backend",
    "create_runtime",
    "BACKEND_CAPABILITIES",
]
