# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0

"""
Persistent Kernel Module Tests

Maps to C++ tests:
  - test_pk_backend_interface_gtest.cc
  - test_pk_task_gtest.cc
  - test_pk_utils_gtest.cc
  - test_pk_runtime_gtest.cc
"""

import pytest
from typing import List

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def pk_runtime_module():
    """Import the persistent kernel runtime module."""
    try:
        from yirage.persistent_kernel import runtime
        return runtime
    except ImportError:
        pytest.skip("persistent_kernel module not available")


@pytest.fixture
def pk_kernel_module():
    """Import the persistent kernel module."""
    try:
        from yirage.persistent_kernel import kernel
        return kernel
    except ImportError:
        pytest.skip("persistent_kernel.kernel module not available")


# =============================================================================
# PKBackendType Tests (maps to test_pk_backend_interface_gtest.cc)
# =============================================================================


class TestPKBackendType:
    """Tests for PKBackendType enum."""

    def test_backend_type_values(self, pk_runtime_module):
        """Test that backend type enum values are defined."""
        PKBackendType = pk_runtime_module.PKBackendType
        
        assert hasattr(PKBackendType, "CUDA")
        assert hasattr(PKBackendType, "CPU")
        assert hasattr(PKBackendType, "MPS")
        assert hasattr(PKBackendType, "ASCEND")
        assert hasattr(PKBackendType, "MACA")

    def test_backend_type_from_name(self, pk_runtime_module):
        """Test string to backend type conversion."""
        PKBackendType = pk_runtime_module.PKBackendType
        
        assert PKBackendType.from_name("cuda") == PKBackendType.CUDA
        assert PKBackendType.from_name("cpu") == PKBackendType.CPU
        assert PKBackendType.from_name("mps") == PKBackendType.MPS
        assert PKBackendType.from_name("CUDA") == PKBackendType.CUDA  # Case insensitive

    def test_backend_type_to_name(self, pk_runtime_module):
        """Test backend type to string conversion."""
        PKBackendType = pk_runtime_module.PKBackendType
        
        assert PKBackendType.CUDA.to_name() == "cuda"
        assert PKBackendType.CPU.to_name() == "cpu"


# =============================================================================
# PKMode Tests (maps to test_pk_backend_interface_gtest.cc)
# =============================================================================


class TestPKMode:
    """Tests for PKMode enum."""

    def test_mode_values(self, pk_runtime_module):
        """Test that mode enum values are defined."""
        PKMode = pk_runtime_module.PKMode
        
        assert hasattr(PKMode, "OFFLINE")
        assert hasattr(PKMode, "ONLINE")
        assert hasattr(PKMode, "ONEPASS")
        assert hasattr(PKMode, "EAGER")
        assert hasattr(PKMode, "GRAPH")
        assert hasattr(PKMode, "STREAMING")

    def test_mode_from_name(self, pk_runtime_module):
        """Test string to mode conversion."""
        PKMode = pk_runtime_module.PKMode
        
        assert PKMode.from_name("offline") == PKMode.OFFLINE
        assert PKMode.from_name("online") == PKMode.ONLINE
        assert PKMode.from_name("eager") == PKMode.EAGER


# =============================================================================
# PKTaskType Tests (maps to test_pk_task_gtest.cc)
# =============================================================================


class TestPKTaskType:
    """Tests for PKTaskType enum."""

    def test_task_type_values(self, pk_runtime_module):
        """Test that task type enum values are defined."""
        PKTaskType = pk_runtime_module.PKTaskType
        
        assert hasattr(PKTaskType, "TERMINATE")
        assert hasattr(PKTaskType, "EMBEDDING")
        assert hasattr(PKTaskType, "RMS_NORM")
        assert hasattr(PKTaskType, "LINEAR")
        assert hasattr(PKTaskType, "ATTENTION")

    def test_task_type_categories(self, pk_runtime_module):
        """Test task type categorization."""
        PKTaskType = pk_runtime_module.PKTaskType
        
        # Compute tasks
        compute_tasks = [PKTaskType.EMBEDDING, PKTaskType.RMS_NORM, PKTaskType.LINEAR]
        for task in compute_tasks:
            assert task.value > 100  # Compute tasks have values > 100


# =============================================================================
# PKCapabilities Tests (maps to test_pk_backend_interface_gtest.cc)
# =============================================================================


class TestPKCapabilities:
    """Tests for PKCapabilities dataclass."""

    def test_capabilities_structure(self, pk_runtime_module):
        """Test capabilities dataclass structure."""
        PKCapabilities = pk_runtime_module.PKCapabilities
        
        caps = PKCapabilities()
        
        assert hasattr(caps, "supports_tma")
        assert hasattr(caps, "supports_tensor_cores")
        assert hasattr(caps, "supports_async_copy")
        assert hasattr(caps, "max_shared_memory")
        assert hasattr(caps, "supported_modes")

    def test_capabilities_default_values(self, pk_runtime_module):
        """Test default capability values."""
        PKCapabilities = pk_runtime_module.PKCapabilities
        
        caps = PKCapabilities()
        
        assert caps.supports_tma is False
        assert caps.supports_tensor_cores is False
        assert caps.max_shared_memory == 0

    def test_backend_capabilities_matrix(self, pk_runtime_module):
        """Test backend capability matrix."""
        BACKEND_CAPABILITIES = pk_runtime_module.BACKEND_CAPABILITIES
        PKBackendType = pk_runtime_module.PKBackendType
        
        # CUDA should support tensor cores
        if PKBackendType.CUDA in BACKEND_CAPABILITIES:
            cuda_caps = BACKEND_CAPABILITIES[PKBackendType.CUDA]
            assert cuda_caps.supports_tensor_cores is True
        
        # CPU should not require TMA
        if PKBackendType.CPU in BACKEND_CAPABILITIES:
            cpu_caps = BACKEND_CAPABILITIES[PKBackendType.CPU]
            assert cpu_caps.supports_tma is False


# =============================================================================
# PKRuntimeConfig Tests (maps to test_pk_utils_gtest.cc)
# =============================================================================


class TestPKRuntimeConfig:
    """Tests for PKRuntimeConfig dataclass."""

    def test_config_structure(self, pk_runtime_module):
        """Test config dataclass structure."""
        PKRuntimeConfig = pk_runtime_module.PKRuntimeConfig
        
        config = PKRuntimeConfig()
        
        assert hasattr(config, "backend")
        assert hasattr(config, "mode")
        assert hasattr(config, "num_workers")
        assert hasattr(config, "num_local_schedulers")

    def test_config_default_values(self, pk_runtime_module):
        """Test default configuration values."""
        PKRuntimeConfig = pk_runtime_module.PKRuntimeConfig
        PKBackendType = pk_runtime_module.PKBackendType
        PKMode = pk_runtime_module.PKMode
        
        config = PKRuntimeConfig()
        
        assert config.backend == PKBackendType.CPU
        assert config.mode == PKMode.EAGER
        assert config.num_workers == 4
        assert config.num_local_schedulers == 1


# =============================================================================
# PKTaskDesc Tests (maps to test_pk_task_gtest.cc)
# =============================================================================


class TestPKTaskDesc:
    """Tests for PKTaskDesc dataclass."""

    def test_task_desc_structure(self, pk_runtime_module):
        """Test task descriptor structure."""
        PKTaskDesc = pk_runtime_module.PKTaskDesc
        PKTaskType = pk_runtime_module.PKTaskType
        
        task = PKTaskDesc(task_type=PKTaskType.LINEAR)
        
        assert hasattr(task, "task_type")
        assert hasattr(task, "trigger_event")
        assert hasattr(task, "dependent_event")
        assert hasattr(task, "input_ptrs")
        assert hasattr(task, "output_ptrs")

    def test_task_desc_default_values(self, pk_runtime_module):
        """Test task descriptor defaults."""
        PKTaskDesc = pk_runtime_module.PKTaskDesc
        PKTaskType = pk_runtime_module.PKTaskType
        
        task = PKTaskDesc(task_type=PKTaskType.EMBEDDING)
        
        assert task.trigger_event == -1
        assert task.dependent_event == -1
        assert task.input_ptrs == []
        assert task.output_ptrs == []


# =============================================================================
# PKRuntime Tests (maps to test_pk_runtime_gtest.cc)
# =============================================================================


class TestPKRuntime:
    """Tests for PKRuntime class."""

    def test_runtime_creation(self, pk_runtime_module):
        """Test runtime can be created."""
        PKRuntime = pk_runtime_module.PKRuntime
        PKRuntimeConfig = pk_runtime_module.PKRuntimeConfig
        
        config = PKRuntimeConfig()
        runtime = PKRuntime(config)
        
        assert runtime is not None
        assert runtime.initialized is False

    def test_runtime_context_manager(self, pk_runtime_module):
        """Test runtime context manager."""
        PKRuntime = pk_runtime_module.PKRuntime
        PKRuntimeConfig = pk_runtime_module.PKRuntimeConfig
        
        config = PKRuntimeConfig(num_workers=2)
        
        with PKRuntime(config) as runtime:
            assert runtime.initialized is True
        
        assert runtime.initialized is False

    def test_runtime_add_task(self, pk_runtime_module):
        """Test adding tasks to runtime."""
        PKRuntime = pk_runtime_module.PKRuntime
        PKRuntimeConfig = pk_runtime_module.PKRuntimeConfig
        PKTaskType = pk_runtime_module.PKTaskType
        
        config = PKRuntimeConfig()
        runtime = PKRuntime(config)
        
        task_id = runtime.add_task(PKTaskType.EMBEDDING)
        
        assert task_id == 0
        assert len(runtime.tasks) == 1

    def test_runtime_get_capabilities(self, pk_runtime_module):
        """Test getting runtime capabilities."""
        PKRuntime = pk_runtime_module.PKRuntime
        PKRuntimeConfig = pk_runtime_module.PKRuntimeConfig
        PKBackendType = pk_runtime_module.PKBackendType
        
        config = PKRuntimeConfig(backend=PKBackendType.CPU)
        runtime = PKRuntime(config)
        
        caps = runtime.get_capabilities()
        assert caps is not None

    def test_runtime_get_supported_modes(self, pk_runtime_module):
        """Test getting supported modes."""
        PKRuntime = pk_runtime_module.PKRuntime
        PKRuntimeConfig = pk_runtime_module.PKRuntimeConfig
        
        config = PKRuntimeConfig()
        runtime = PKRuntime(config)
        
        modes = runtime.get_supported_modes()
        assert isinstance(modes, list)


# =============================================================================
# Factory Function Tests (maps to test_pk_utils_gtest.cc)
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_get_available_backends(self, pk_runtime_module):
        """Test getting available backends."""
        get_available_backends = pk_runtime_module.get_available_backends
        PKBackendType = pk_runtime_module.PKBackendType
        
        backends = get_available_backends()
        
        assert isinstance(backends, list)
        assert PKBackendType.CPU in backends  # CPU always available

    def test_get_best_backend(self, pk_runtime_module):
        """Test getting best backend."""
        get_best_backend = pk_runtime_module.get_best_backend
        PKBackendType = pk_runtime_module.PKBackendType

        backend = get_best_backend()

        assert isinstance(backend, PKBackendType)

    def test_create_runtime(self, pk_runtime_module):
        """Test runtime factory function."""
        create_runtime = pk_runtime_module.create_runtime
        PKBackendType = pk_runtime_module.PKBackendType
        PKRuntime = pk_runtime_module.PKRuntime

        runtime = create_runtime(backend=PKBackendType.CPU, num_workers=2)

        assert isinstance(runtime, PKRuntime)
        assert runtime.config.num_workers == 2

    def test_create_runtime_mps(self, pk_runtime_module):
        """Test creating an MPS runtime on Apple Silicon."""
        create_runtime = pk_runtime_module.create_runtime
        PKBackendType = pk_runtime_module.PKBackendType
        PKRuntime = pk_runtime_module.PKRuntime
        get_available_backends = pk_runtime_module.get_available_backends

        if PKBackendType.MPS not in get_available_backends():
            pytest.skip("MPS backend not available on this system")

        runtime = create_runtime(backend=PKBackendType.MPS, num_workers=1)
        assert isinstance(runtime, PKRuntime)
        assert runtime.config.backend == PKBackendType.MPS

    def test_create_runtime_auto(self, pk_runtime_module):
        """Test runtime creation with auto backend."""
        create_runtime = pk_runtime_module.create_runtime
        PKBackendType = pk_runtime_module.PKBackendType
        
        runtime = create_runtime(backend=PKBackendType.AUTO)
        
        # Should resolve to a concrete backend
        assert runtime.config.backend != PKBackendType.AUTO


# =============================================================================
# PKWorker Tests (maps to test_pk_task_gtest.cc)
# =============================================================================


class TestPKWorker:
    """Tests for PKWorker thread class."""

    def test_worker_creation(self, pk_runtime_module):
        """Test worker can be created."""
        PKWorker = pk_runtime_module.PKWorker
        PKTaskExecutor = pk_runtime_module.PKTaskExecutor
        PKBackendType = pk_runtime_module.PKBackendType
        
        import queue
        
        task_queue = queue.Queue()
        event_counters = {}
        executor = PKTaskExecutor(PKBackendType.CPU)
        
        worker = PKWorker(0, task_queue, event_counters, executor)
        
        assert worker.worker_id == 0
        assert worker.running is True


# =============================================================================
# PKTaskExecutor Tests (maps to test_pk_task_gtest.cc)
# =============================================================================


class TestPKTaskExecutor:
    """Tests for PKTaskExecutor class."""

    def test_executor_creation(self, pk_runtime_module):
        """Test executor can be created."""
        PKTaskExecutor = pk_runtime_module.PKTaskExecutor
        PKBackendType = pk_runtime_module.PKBackendType
        
        executor = PKTaskExecutor(PKBackendType.CPU)
        
        assert executor.backend == PKBackendType.CPU

    def test_executor_has_handlers(self, pk_runtime_module):
        """Test executor has task handlers."""
        PKTaskExecutor = pk_runtime_module.PKTaskExecutor
        PKBackendType = pk_runtime_module.PKBackendType
        PKTaskType = pk_runtime_module.PKTaskType
        
        executor = PKTaskExecutor(PKBackendType.CPU)
        
        # Check that handlers are registered
        assert PKTaskType.EMBEDDING in executor.task_handlers
        assert PKTaskType.RMS_NORM in executor.task_handlers
        assert PKTaskType.LINEAR in executor.task_handlers


# =============================================================================
# Parameterized Tests
# =============================================================================


@pytest.mark.parametrize("backend_name,expected_type", [
    ("cuda", "CUDA"),
    ("cpu", "CPU"),
    ("mps", "MPS"),
    ("ascend", "ASCEND"),
    ("triton", "TRITON"),
])
def test_backend_name_mapping(pk_runtime_module, backend_name, expected_type):
    """Test backend name to type mapping."""
    PKBackendType = pk_runtime_module.PKBackendType
    
    result = PKBackendType.from_name(backend_name)
    assert result.name == expected_type


@pytest.mark.parametrize("mode_name,expected_type", [
    ("offline", "OFFLINE"),
    ("online", "ONLINE"),
    ("eager", "EAGER"),
    ("graph", "GRAPH"),
])
def test_mode_name_mapping(pk_runtime_module, mode_name, expected_type):
    """Test mode name to type mapping."""
    PKMode = pk_runtime_module.PKMode
    
    result = PKMode.from_name(mode_name)
    assert result.name == expected_type


# =============================================================================
# Integration Tests
# =============================================================================


class TestPKIntegration:
    """Integration tests for persistent kernel module."""

    def test_full_task_graph_workflow(self, pk_runtime_module):
        """Test complete task graph workflow."""
        create_runtime = pk_runtime_module.create_runtime
        PKBackendType = pk_runtime_module.PKBackendType
        PKTaskType = pk_runtime_module.PKTaskType
        
        runtime = create_runtime(backend=PKBackendType.CPU, num_workers=2)
        
        # Build task graph
        embed_id = runtime.add_task(PKTaskType.EMBEDDING)
        norm_id = runtime.add_task(PKTaskType.RMS_NORM, dependent_event=embed_id)
        linear_id = runtime.add_task(PKTaskType.LINEAR, dependent_event=norm_id)
        
        assert len(runtime.tasks) == 3
        assert runtime.tasks[1].dependent_event == embed_id
        assert runtime.tasks[2].dependent_event == norm_id

    def test_full_task_graph_workflow_mps(self, pk_runtime_module):
        """Test complete task graph workflow with MPS backend."""
        create_runtime = pk_runtime_module.create_runtime
        PKBackendType = pk_runtime_module.PKBackendType
        PKTaskType = pk_runtime_module.PKTaskType
        get_available_backends = pk_runtime_module.get_available_backends

        if PKBackendType.MPS not in get_available_backends():
            pytest.skip("MPS backend not available on this system")

        runtime = create_runtime(backend=PKBackendType.MPS, num_workers=1)

        embed_id = runtime.add_task(PKTaskType.EMBEDDING)
        norm_id = runtime.add_task(PKTaskType.RMS_NORM, dependent_event=embed_id)
        linear_id = runtime.add_task(PKTaskType.LINEAR, dependent_event=norm_id)

        assert len(runtime.tasks) == 3
        assert runtime.tasks[1].dependent_event == embed_id
        assert runtime.tasks[2].dependent_event == norm_id

    def test_runtime_initialization_and_finalization(self, pk_runtime_module):
        """Test runtime lifecycle."""
        PKRuntime = pk_runtime_module.PKRuntime
        PKRuntimeConfig = pk_runtime_module.PKRuntimeConfig
        
        config = PKRuntimeConfig(num_workers=2, num_local_schedulers=1)
        runtime = PKRuntime(config)
        
        # Initialize
        assert runtime.initialize() is True
        assert runtime.initialized is True
        assert len(runtime.workers) == 2
        assert len(runtime.schedulers) == 1
        
        # Finalize
        runtime.finalize()
        assert runtime.initialized is False
        assert len(runtime.workers) == 0
