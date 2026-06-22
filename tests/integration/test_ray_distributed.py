#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""
Ray Distributed Integration Tests

Tests for Ray-based distributed search functionality.
Run with: pytest tests/integration/test_ray_distributed.py -v
"""

import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
PYTHON_ROOT = PROJECT_ROOT / "python"
sys.path.insert(0, str(PYTHON_ROOT))

from conftest import RAY_AVAILABLE, load_module


# =============================================================================
# Ray Integration Tests
# =============================================================================

@pytest.mark.ray
class TestRayDistributed:
    """Tests for Ray distributed computing."""

    def test_ray_worker_initialization(self, ray_session):
        """Test Ray worker can be initialized."""
        assert ray_session.is_initialized()

    def test_scatter_gather_pattern(self, ray_session):
        """Test scatter-gather search pattern."""
        @ray_session.remote
        def search_partition(partition_id, config):
            # Simulate search on partition
            return {
                "partition_id": partition_id,
                "best_latency": 0.1 + partition_id * 0.01,
                "num_searched": 100,
            }
        
        # Scatter work across partitions
        configs = [{"grid_x": i} for i in range(4)]
        futures = [
            search_partition.remote(i, config)
            for i, config in enumerate(configs)
        ]
        
        # Gather results
        results = ray_session.get(futures)
        
        assert len(results) == 4
        assert all("best_latency" in r for r in results)

    def test_result_aggregation(self, ray_session):
        """Test aggregation of distributed results."""
        @ray_session.remote
        def search_worker(worker_id, search_space):
            # Simulate finding best result
            return {
                "worker_id": worker_id,
                "best_config": {"grid_x": worker_id + 1},
                "best_latency": 1.0 / (worker_id + 1),
            }
        
        # Run workers
        search_spaces = [{"start": i * 100, "end": (i + 1) * 100} for i in range(4)]
        futures = [
            search_worker.remote(i, space)
            for i, space in enumerate(search_spaces)
        ]
        results = ray_session.get(futures)
        
        # Aggregate: find best overall
        best_result = min(results, key=lambda r: r["best_latency"])
        
        assert best_result["best_latency"] == 0.25  # Worker 3 has 1/4

    def test_worker_state_isolation(self, ray_session):
        """Test that worker states are isolated."""
        @ray_session.remote
        class StatefulWorker:
            def __init__(self, worker_id):
                self.worker_id = worker_id
                self.state = 0
            
            def increment(self):
                self.state += 1
                return self.state
            
            def get_state(self):
                return self.state
        
        # Create two workers
        worker1 = StatefulWorker.remote(1)
        worker2 = StatefulWorker.remote(2)
        
        # Increment worker1 multiple times
        ray_session.get(worker1.increment.remote())
        ray_session.get(worker1.increment.remote())
        ray_session.get(worker1.increment.remote())
        
        # worker2 should still be at 0
        state1 = ray_session.get(worker1.get_state.remote())
        state2 = ray_session.get(worker2.get_state.remote())
        
        assert state1 == 3
        assert state2 == 0


# =============================================================================
# Distributed Search Tests
# =============================================================================

@pytest.mark.ray
class TestDistributedSearch:
    """Tests for distributed kernel search."""

    def test_parallel_config_exploration(self, ray_session):
        """Test parallel exploration of config space."""
        @ray_session.remote
        def evaluate_config(config):
            # Simulate config evaluation
            grid_product = config["grid_x"] * config["grid_y"]
            latency = 1.0 / (grid_product + 1)  # Better for larger grids
            return {
                "config": config,
                "latency": latency,
                "valid": True,
            }
        
        # Generate configs to evaluate
        configs = [
            {"grid_x": x, "grid_y": y, "block_x": 128}
            for x in [1, 2, 4]
            for y in [1, 2]
        ]
        
        # Evaluate in parallel
        futures = [evaluate_config.remote(c) for c in configs]
        results = ray_session.get(futures)
        
        # Find best
        valid_results = [r for r in results if r["valid"]]
        best = min(valid_results, key=lambda r: r["latency"])
        
        assert best["config"]["grid_x"] == 4
        assert best["config"]["grid_y"] == 2

    def test_distributed_verification(self, ray_session):
        """Test distributed kernel verification."""
        @ray_session.remote
        def verify_kernel(kernel_id, expected_output):
            # Simulate verification
            actual_output = expected_output  # Assume correct
            return {
                "kernel_id": kernel_id,
                "verified": actual_output == expected_output,
                "verification_time_ms": 0.5,
            }
        
        # Verify multiple kernels
        kernels = [
            {"kernel_id": i, "expected": f"output_{i}"}
            for i in range(10)
        ]
        
        futures = [
            verify_kernel.remote(k["kernel_id"], k["expected"])
            for k in kernels
        ]
        results = ray_session.get(futures)
        
        # All should verify
        assert all(r["verified"] for r in results)


# =============================================================================
# Error Handling Tests
# =============================================================================

@pytest.mark.ray
class TestRayErrorHandling:
    """Tests for Ray error handling."""

    def test_worker_exception_handling(self, ray_session):
        """Test handling of worker exceptions."""
        @ray_session.remote
        def failing_task(should_fail):
            if should_fail:
                raise ValueError("Intentional failure")
            return "success"
        
        # One successful, one failing
        futures = [
            failing_task.remote(False),
            failing_task.remote(True),
        ]
        
        # Get successful one
        result = ray_session.get(futures[0])
        assert result == "success"
        
        # Failing one should raise
        with pytest.raises(ray_session.exceptions.RayTaskError):
            ray_session.get(futures[1])

    def test_timeout_handling(self, ray_session):
        """Test handling of task timeouts."""
        import time
        
        @ray_session.remote
        def slow_task():
            time.sleep(0.5)
            return "done"
        
        future = slow_task.remote()
        
        # Should complete within reasonable time
        result = ray_session.get(future, timeout=5.0)
        assert result == "done"
