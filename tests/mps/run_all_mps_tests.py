#!/usr/bin/env python3
"""
YiRage MPS Full Test Suite - Ray Mode

This script runs ALL MPS tests with Ray distributed search enabled.

Test Categories:
1. Core muGraph Search - superoptimize with Ray
2. MuGraph Execution and Correctness
3. Custom Threadblock Graph
4. Ray Integration
5. MuGraph Storage (persistent cache)
6. Performance Benchmark

Requirements:
- Apple Silicon Mac with MPS
- Ray installed
- YiRage compiled for MPS

Usage:
    python tests/mps/run_all_mps_tests.py
"""

import sys
import os
import time
import json
import platform
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# Setup path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "python"))


@dataclass
class TestResult:
    """Test result container."""
    name: str
    passed: bool
    duration_s: float
    detail: str = ""
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


class MPSTestSuite:
    """Complete MPS test suite with Ray mode."""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.start_time = None
        self.ray_initialized = False
        
    def setup(self) -> bool:
        """Setup test environment."""
        print("=" * 70)
        print("  YiRage MPS Test Suite - Ray Mode")
        print("=" * 70)
        print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Platform: {platform.system()} {platform.release()}")
        print(f"  Machine: {platform.machine()}")
        
        # Check PyTorch and MPS
        try:
            import torch
            print(f"  PyTorch: {torch.__version__}")
            
            if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
                print("  ❌ MPS not available!")
                return False
            print(f"  MPS: Available ✓")
        except ImportError:
            print("  ❌ PyTorch not installed!")
            return False
        
        # Check YiRage
        try:
            import yirage as yr
            print(f"  YiRage: {yr.__version__}")
            print(f"  Backends: {yr.get_available_backends()}")
            
            if 'mps' not in yr.get_available_backends():
                print("  ❌ MPS backend not available in YiRage!")
                return False
        except ImportError as e:
            print(f"  ❌ YiRage import failed: {e}")
            return False
        
        # Initialize Ray
        try:
            import ray
            if not ray.is_initialized():
                num_cpus = os.cpu_count()
                ray.init(num_cpus=num_cpus, ignore_reinit_error=True)
                print(f"  Ray: Initialized with {num_cpus} CPUs ✓")
            else:
                print(f"  Ray: Already initialized ✓")
            self.ray_initialized = True
        except ImportError:
            print("  ❌ Ray not installed!")
            return False
        except Exception as e:
            print(f"  ❌ Ray init failed: {e}")
            return False
        
        # Check MuGraph storage
        try:
            from yirage.storage.mugraph_store import MuGraphStore
            store = MuGraphStore()
            stats = store.get_stats()
            print(f"  MuGraph Store: {stats['total_entries']} entries")
        except Exception as e:
            print(f"  MuGraph Store: Warning - {e}")
        
        self.start_time = time.perf_counter()
        return True
    
    def run_test(self, name: str, test_func) -> TestResult:
        """Run a single test."""
        print(f"\n{'='*70}")
        print(f"  TEST: {name}")
        print(f"{'='*70}")
        
        start = time.perf_counter()
        try:
            passed, detail, metrics = test_func()
            duration = time.perf_counter() - start
            
            result = TestResult(
                name=name,
                passed=passed,
                duration_s=duration,
                detail=detail,
                metrics=metrics
            )
        except Exception as e:
            duration = time.perf_counter() - start
            import traceback
            result = TestResult(
                name=name,
                passed=False,
                duration_s=duration,
                detail="Exception",
                error=str(e) + "\n" + traceback.format_exc()
            )
        
        status = "✅ PASS" if result.passed else "❌ FAIL"
        print(f"\n  Result: {status} ({duration:.2f}s)")
        if result.error:
            print(f"  Error: {result.error[:200]}...")
        
        self.results.append(result)
        return result
    
    # =========================================================================
    # Test 1: Core muGraph Search with Ray
    # =========================================================================
    def test_matmul_ray_search(self) -> Tuple[bool, str, Dict]:
        """Test MatMul muGraph search using Ray distributed mode."""
        import torch
        import yirage as yr
        from yirage.kernel import KNGraph
        from yirage.backends.mps.config import get_mps_search_config
        import os
        
        print("  Testing: MatMul muGraph search with Ray PARALLEL")
        
        # Problem size - larger for better optimization potential
        M, K, N = 128, 1024, 1024
        print(f"  Problem: ({M}, {K}) @ ({K}, {N}) FP16")
        
        # Create kernel graph using KNGraph class (supports Ray)
        graph = KNGraph(backend='mps')
        X = graph.new_input(dims=(M, K), dtype=yr.float16)
        W = graph.new_input(dims=(K, N), dtype=yr.float16)
        O = graph.matmul(X, W)
        graph.mark_output(O)
        print(f"  ✓ Kernel graph created")
        
        # Use FULL MPS search config from Apple Silicon specs
        mps_config = get_mps_search_config()
        griddims = mps_config["grid_dims_to_explore"]
        blockdims = mps_config["block_dims_to_explore"]
        fmaps = mps_config["fmaps_to_explore"]
        franges = mps_config["franges_to_explore"]
        
        num_workers = os.cpu_count()
        print(f"  Using FULL MPS search space with Ray ({num_workers} workers):")
        print(f"    - Grid dims: {len(griddims)} configs")
        print(f"    - Block dims: {len(blockdims)} configs")
        print(f"    - Fmaps: {fmaps}")
        print(f"    - Franges: {franges}")
        
        print(f"  Running Ray PARALLEL search...")
        start_search = time.perf_counter()
        
        # Use superoptimize() with Ray enabled (default)
        optimized = graph.superoptimize(
            griddims=griddims,
            blockdims=blockdims,
            fmaps=fmaps,
            franges=franges,
            verbose=True,
            use_ray=True,  # Enable Ray parallel search
            num_workers=num_workers,
            is_formal_verified=False,
        )
        
        search_time = time.perf_counter() - start_search
        print(f"  Search completed in {search_time:.2f}s")
        
        if optimized is None:
            return False, "No muGraph found", {"search_time": search_time}
        
        # Create input tensors
        input_x = torch.randn(M, K, dtype=torch.float16, device="mps")
        input_w = torch.randn(K, N, dtype=torch.float16, device="mps")
        
        # PyTorch baseline
        torch.mps.synchronize()
        ref = torch.matmul(input_x, input_w)
        
        # Warmup
        for _ in range(10):
            _ = torch.matmul(input_x, input_w)
        torch.mps.synchronize()
        
        # Benchmark PyTorch
        start = time.perf_counter()
        for _ in range(100):
            _ = torch.matmul(input_x, input_w)
        torch.mps.synchronize()
        pytorch_time = (time.perf_counter() - start) / 100 * 1000
        
        # Benchmark YiRage
        for _ in range(10):
            outputs = optimized(inputs=[input_x, input_w])
        torch.mps.synchronize()
        
        start = time.perf_counter()
        for _ in range(100):
            outputs = optimized(inputs=[input_x, input_w])
        torch.mps.synchronize()
        yirage_time = (time.perf_counter() - start) / 100 * 1000
        
        # Verify correctness
        max_diff = (outputs[0].cpu() - ref.cpu()).abs().max().item()
        correct = max_diff < 0.1
        
        speedup = pytorch_time / yirage_time
        
        print(f"  PyTorch: {pytorch_time:.4f} ms")
        print(f"  YiRage:  {yirage_time:.4f} ms")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Max diff: {max_diff:.6f}")
        
        metrics = {
            "search_time": search_time,
            "pytorch_ms": pytorch_time,
            "yirage_ms": yirage_time,
            "speedup": speedup,
            "max_diff": max_diff,
        }
        
        return correct, f"speedup={speedup:.2f}x, diff={max_diff:.6f}", metrics
    
    # =========================================================================
    # Test 2: muGraph Execution and Profiling
    # =========================================================================
    def test_mugraph_execution(self) -> Tuple[bool, str, Dict]:
        """Test muGraph execution and profiling on MPS."""
        import torch
        import yirage as yr
        from yirage.core import search
        from yirage.kernel import KNGraph
        from yirage.backends.mps.config import get_mps_search_config
        
        print("  Testing: muGraph Execution and Profiling")
        
        # Larger problem for better optimization
        M, K, N = 64, 512, 512
        print(f"  Problem: MatMul ({M}, {K}) @ ({K}, {N})")
        
        # Create kernel graph
        graph = yr.new_kernel_graph()
        X = graph.new_input(dims=(M, K), dtype=yr.float16)
        W = graph.new_input(dims=(K, N), dtype=yr.float16)
        O = graph.matmul(X, W)
        graph.mark_output(O)
        print(f"  ✓ Kernel graph created")
        
        # Use FULL MPS search config
        mps_config = get_mps_search_config()
        griddims = mps_config["grid_dims_to_explore"]
        blockdims = mps_config["block_dims_to_explore"]
        fmaps = mps_config["fmaps_to_explore"]
        franges = mps_config["franges_to_explore"]
        
        print(f"  Using full MPS config: {len(griddims)} grids x {len(blockdims)} blocks")
        
        # Search for muGraphs
        cygraphs = search(
            graph.cygraph,
            backend='mps',
            griddims=griddims,
            blockdims=blockdims,
            fmaps=fmaps,
            franges=franges,
            verbose=False,
            is_formal_verified=False,
        )
        
        print(f"  Found {len(cygraphs)} muGraphs")
        
        if len(cygraphs) == 0:
            return False, "No muGraph found", {}
        
        # Create tensors
        input_x = torch.randn(M, K, dtype=torch.float16, device="mps")
        input_w = torch.randn(K, N, dtype=torch.float16, device="mps")
        
        # PyTorch reference
        ref = torch.matmul(input_x, input_w)
        
        # Profile each muGraph
        best_time = float('inf')
        best_diff = float('inf')
        
        for idx, cygraph in enumerate(cygraphs[:5]):
            try:
                g = KNGraph(cygraph, backend='mps')
                
                # Warmup
                for _ in range(10):
                    outputs = g(inputs=[input_x, input_w])
                torch.mps.synchronize()
                
                # Profile
                start = time.perf_counter()
                for _ in range(100):
                    outputs = g(inputs=[input_x, input_w])
                torch.mps.synchronize()
                elapsed = (time.perf_counter() - start) / 100 * 1000
                
                # Verify
                max_diff = (outputs[0].cpu() - ref.cpu()).abs().max().item()
                
                print(f"    muGraph[{idx}]: {elapsed:.4f}ms, diff={max_diff:.6f}")
                
                if elapsed < best_time:
                    best_time = elapsed
                    best_diff = max_diff
                    
            except Exception as e:
                print(f"    muGraph[{idx}]: Error - {e}")
        
        # PyTorch benchmark
        start = time.perf_counter()
        for _ in range(100):
            _ = torch.matmul(input_x, input_w)
        torch.mps.synchronize()
        pytorch_time = (time.perf_counter() - start) / 100 * 1000
        
        speedup = pytorch_time / best_time if best_time < float('inf') else 0
        
        print(f"  PyTorch: {pytorch_time:.4f} ms")
        print(f"  Best YiRage: {best_time:.4f} ms")
        print(f"  Speedup: {speedup:.2f}x")
        
        correct = best_diff < 0.1
        
        metrics = {
            "pytorch_ms": pytorch_time,
            "yirage_ms": best_time,
            "speedup": speedup,
            "max_diff": best_diff,
            "num_mugraphs": len(cygraphs),
        }
        
        return correct, f"speedup={speedup:.2f}x", metrics
    
    # =========================================================================
    # Test 3: MuGraph Storage Verification
    # =========================================================================
    def test_mugraph_storage(self) -> Tuple[bool, str, Dict]:
        """Test MuGraph persistent storage."""
        from yirage.storage.mugraph_store import MuGraphStore
        
        print("  Testing: MuGraph Storage")
        
        store = MuGraphStore()
        stats = store.get_stats()
        
        print(f"  Total entries: {stats.get('total_entries', 0)}")
        by_backend = stats.get('by_backend', {})
        print(f"  MPS entries: {by_backend.get('mps', {}).get('count', 0)}")
        
        # List MPS entries
        try:
            entries = store.list_all(backend='mps', limit=10)
            print(f"\n  Stored MuGraphs:")
            for e in entries[:5]:
                if hasattr(e, 'metadata'):
                    print(f"    - {e.metadata.graph_hash}: {e.metadata.latency_ms:.4f}ms")
                else:
                    print(f"    - {e}")
        except Exception as e:
            print(f"  List entries: {e}")
        
        mps_count = by_backend.get('mps', {}).get('count', 0)
        total = stats.get('total_entries', 0)
        
        metrics = {
            "total_entries": total,
            "mps_entries": mps_count,
        }
        
        # Pass if store initialized successfully (even if empty)
        return True, f"{mps_count} MPS muGraphs, {total} total", metrics
    
    # =========================================================================
    # Test 4: Ray Parallel Task Execution
    # =========================================================================
    def test_ray_parallel_tasks(self) -> Tuple[bool, str, Dict]:
        """Test Ray parallel task execution for search coordination."""
        import ray
        
        print("  Testing: Ray Parallel Task Execution")
        
        # Define simple search task
        @ray.remote(num_cpus=1)
        def search_task(worker_id: int, grid_dims: List[Tuple], config: Dict) -> Dict:
            """Simulate a search task on a partition."""
            import time
            
            # Simulate search work
            start = time.perf_counter()
            results = []
            for grid in grid_dims:
                # Simulate evaluation
                latency = 1.0 / (grid[0] + 1)  # Fake metric
                results.append({
                    "grid": grid,
                    "latency_ms": latency,
                    "valid": True,
                })
            elapsed = time.perf_counter() - start
            
            return {
                "worker_id": worker_id,
                "num_candidates": len(results),
                "best_latency": min(r["latency_ms"] for r in results) if results else None,
                "search_time": elapsed,
            }
        
        # Create partitions
        num_workers = 4
        all_grids = [(1, 1, 1), (2, 1, 1), (4, 1, 1), (8, 1, 1),
                     (16, 1, 1), (32, 1, 1), (64, 1, 1), (128, 1, 1)]
        
        partitions = [all_grids[i::num_workers] for i in range(num_workers)]
        
        print(f"  Workers: {num_workers}")
        print(f"  Total grid configs: {len(all_grids)}")
        print(f"  Partitions: {[len(p) for p in partitions]}")
        
        # Launch parallel tasks
        config = {"backend": "mps", "fmaps": [-1], "franges": [4]}
        
        start = time.perf_counter()
        futures = [
            search_task.remote(i, partitions[i], config)
            for i in range(num_workers)
        ]
        
        results = ray.get(futures)
        search_time = time.perf_counter() - start
        
        # Aggregate results
        total_candidates = sum(r["num_candidates"] for r in results)
        best_latency = min(r["best_latency"] for r in results if r["best_latency"])
        
        print(f"  Results from {len(results)} workers:")
        for r in results:
            print(f"    Worker {r['worker_id']}: {r['num_candidates']} candidates, "
                  f"best={r['best_latency']:.4f}ms")
        
        print(f"  Total candidates: {total_candidates}")
        print(f"  Global best: {best_latency:.4f}ms")
        print(f"  Total time: {search_time:.4f}s")
        
        metrics = {
            "num_workers": num_workers,
            "total_candidates": total_candidates,
            "best_latency": best_latency,
            "search_time": search_time,
        }
        
        passed = total_candidates == len(all_grids) and len(results) == num_workers
        
        return passed, f"{total_candidates} configs searched by {num_workers} workers", metrics
    
    # =========================================================================
    # Test 5: MPS Config Detection
    # =========================================================================
    def test_mps_config(self) -> Tuple[bool, str, Dict]:
        """Test MPS configuration detection."""
        from yirage.backends.mps.config import detect_apple_silicon, get_mps_search_config
        
        print("  Testing: MPS Config Detection")
        
        # Detect chip
        chip_family, specs = detect_apple_silicon()
        print(f"  Chip: {chip_family.value} - {specs.chip_name}")
        print(f"  GPU Cores: {specs.gpu_cores}")
        print(f"  Memory BW: {specs.memory_bandwidth_gbps} GB/s")
        
        # Get search config
        config = get_mps_search_config()
        print(f"  Grid dims: {len(config['grid_dims_to_explore'])} configs")
        print(f"  Block dims: {len(config['block_dims_to_explore'])} configs")
        print(f"  Fmaps: {config['fmaps_to_explore']}")
        print(f"  Franges: {config['franges_to_explore']}")
        
        metrics = {
            "chip_name": specs.chip_name,
            "gpu_cores": specs.gpu_cores,
            "memory_bw": specs.memory_bandwidth_gbps,
            "grid_configs": len(config['grid_dims_to_explore']),
            "block_configs": len(config['block_dims_to_explore']),
        }
        
        return specs.gpu_cores > 0, f"{specs.chip_name}, {specs.gpu_cores} cores", metrics
    
    # =========================================================================
    # Run All Tests
    # =========================================================================
    def run_all(self) -> int:
        """Run all tests."""
        if not self.setup():
            return 1
        
        # Test list
        tests = [
            ("MPS Config Detection", self.test_mps_config),
            ("MuGraph Storage", self.test_mugraph_storage),
            ("MatMul Search", self.test_matmul_ray_search),
            ("MuGraph Execution", self.test_mugraph_execution),
            ("Ray Parallel Tasks", self.test_ray_parallel_tasks),
        ]
        
        # Run tests
        for name, test_func in tests:
            self.run_test(name, test_func)
        
        # Summary
        total_time = time.perf_counter() - self.start_time
        
        print(f"\n{'='*70}")
        print("  SUMMARY")
        print(f"{'='*70}")
        
        passed = 0
        for result in self.results:
            status = "✅" if result.passed else "❌"
            print(f"  {status} {result.name}: {result.detail} ({result.duration_s:.2f}s)")
            if result.passed:
                passed += 1
        
        print(f"\n  Total: {passed}/{len(self.results)} passed")
        print(f"  Time: {total_time:.2f}s")
        
        # Save results
        self.save_results()
        
        if passed == len(self.results):
            print(f"\n  🎉 ALL TESTS PASSED!")
            return 0
        else:
            print(f"\n  ⚠️ Some tests failed")
            return 1
    
    def save_results(self):
        """Save test results to JSON."""
        results_file = PROJECT_ROOT / "test_results_mps.json"
        
        data = {
            "timestamp": datetime.now().isoformat(),
            "platform": platform.platform(),
            "total_tests": len(self.results),
            "passed": sum(1 for r in self.results if r.passed),
            "failed": sum(1 for r in self.results if not r.passed),
            "results": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "duration_s": r.duration_s,
                    "detail": r.detail,
                    "error": r.error,
                    "metrics": r.metrics,
                }
                for r in self.results
            ]
        }
        
        with open(results_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n  Results saved to: {results_file}")


def main():
    suite = MPSTestSuite()
    return suite.run_all()


if __name__ == "__main__":
    sys.exit(main())
