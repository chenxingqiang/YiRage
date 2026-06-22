#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
YPK (YiRage Persistent Kernel) Ray Optimization Demo

This demo shows how to use Ray distributed optimization to find
optimal configurations for YPK persistent kernels.

Features demonstrated:
1. Distributed kernel configuration search
2. Multi-worker parallel evaluation
3. Optimal configuration selection
4. Distributed profiling
5. Kernel compilation metadata generation

Usage:
    python ypk_ray_optimization_demo.py
"""

import sys
import os
import json
import time
from pathlib import Path

# Add project path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "python"))


def load_module_from_path(name: str, path: str):
    """Load module from file path."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Load YPK integration module
YPK_PATH = PROJECT_ROOT / "python" / "yirage" / "distributed" / "ypk_integration.py"
ypk = load_module_from_path("yirage.distributed.ypk_integration", str(YPK_PATH))


def print_header(title: str):
    """Print section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def demo_basic_optimization():
    """Demo 1: Basic kernel optimization."""
    print_header("Demo 1: Basic YPK Kernel Optimization")

    # Define kernel to optimize
    kernel_graph = {
        "type": "attention",
        "batch_size": 32,
        "seq_length": 2048,
        "num_heads": 32,
        "head_dim": 128,
        "estimated_flops": 32 * 2048 * 32 * 128 * 2048 * 4,
    }

    print(f"\n  Kernel: {kernel_graph['type']}")
    print(f"  Batch: {kernel_graph['batch_size']}, Seq: {kernel_graph['seq_length']}")
    print(f"  Heads: {kernel_graph['num_heads']}, Dim: {kernel_graph['head_dim']}")

    # Define search space
    search_space = ypk.KernelSearchSpace(
        grid_dims=[
            (1, 1, 1),
            (2, 1, 1),
            (4, 1, 1),
            (8, 1, 1),
            (2, 2, 1),
            (4, 2, 1),
        ],
        block_dims=[
            (64, 1, 1),
            (128, 1, 1),
            (256, 1, 1),
            (512, 1, 1),
        ],
        modes=[ypk.YPKMode.ONLINE],
    )

    print(f"\n  Search Space:")
    print(f"  - Grid dims: {len(search_space.grid_dims)} options")
    print(f"  - Block dims: {len(search_space.block_dims)} options")
    print(f"  - Total configs: {search_space.get_total_configs()}")

    # Run optimization
    print(f"\n  Running distributed optimization with 4 workers...")

    result = ypk.optimize_ypk_kernel(
        kernel_graph,
        num_workers=4,
        search_space=search_space,
    )

    print(f"\n  Results:")
    print(f"  - Configs evaluated: {result.num_configs_evaluated}")
    print(f"  - Search time: {result.search_time_s:.2f}s")
    print(f"  - Best latency: {result.best_latency_ms:.4f} ms")

    print(f"\n  Best Configuration:")
    print(f"  - Grid dim: {result.best_config.grid_dim}")
    print(f"  - Block dim: {result.best_config.block_dim}")
    print(f"  - Mode: {result.best_config.mode.value}")

    return result


def demo_multi_backend():
    """Demo 2: Multi-backend configuration."""
    print_header("Demo 2: Multi-Backend Configuration")

    backends = [
        ypk.YPKBackend.CUDA,
        ypk.YPKBackend.MPS,
        ypk.YPKBackend.ASCEND,
        ypk.YPKBackend.MACA,
    ]

    print(f"\n  Supported YPK Backends:")
    for backend in backends:
        config = ypk.YPKConfig(backend=backend)
        print(f"  - {backend.name}: target_cc={config.target_cc}")

    # Create configurations for different backends
    cuda_config = ypk.YPKConfig(
        backend=ypk.YPKBackend.CUDA,
        target_cc=90,  # Hopper
        mode=ypk.YPKMode.ONLINE,
    )

    print(f"\n  CUDA Hopper Configuration:")
    print(f"  - Backend: {cuda_config.backend.name}")
    print(f"  - Target CC: {cuda_config.target_cc}")
    print(f"  - Mode: {cuda_config.mode.value}")

    return True


def demo_distributed_profiling():
    """Demo 3: Distributed profiling."""
    print_header("Demo 3: Distributed Kernel Profiling")

    optimizer = ypk.YPKRayOptimizer(num_workers=4)

    try:
        # Configuration to profile
        config = ypk.YPKConfig(
            grid_dim=(8, 1, 1),
            block_dim=(256, 1, 1),
            mode=ypk.YPKMode.ONLINE,
        )

        kernel_graph = {
            "type": "rmsnorm",
            "hidden_size": 4096,
            "batch_size": 32,
            "estimated_flops": 32 * 4096 * 5,
        }

        print(f"\n  Profiling: {kernel_graph['type']}")
        print(f"  Config: grid={config.grid_dim}, block={config.block_dim}")

        # Run distributed profiling
        profile_result = optimizer.profile_distributed(config, kernel_graph, num_iterations=100)

        print(f"\n  Profile Results:")
        print(f"  - Workers: {profile_result['num_workers']}")
        print(f"  - Mean latency: {profile_result['mean_latency_ms']:.4f} ms")
        print(f"  - Min latency: {profile_result['min_latency_ms']:.4f} ms")
        print(f"  - Max latency: {profile_result['max_latency_ms']:.4f} ms")

        print(f"\n  Per-Worker Latencies:")
        for i, lat in enumerate(profile_result["worker_latencies"]):
            print(f"    Worker {i}: {lat:.4f} ms")

    finally:
        optimizer.shutdown()

    return True


def demo_kernel_compilation():
    """Demo 4: Kernel compilation metadata."""
    print_header("Demo 4: Kernel Compilation Metadata")

    import tempfile

    optimizer = ypk.YPKRayOptimizer(num_workers=1)

    # Optimized configuration
    config = ypk.YPKConfig(
        mode=ypk.YPKMode.ONLINE,
        backend=ypk.YPKBackend.CUDA,
        target_cc=90,
        grid_dim=(8, 1, 1),
        block_dim=(256, 1, 1),
        max_seq_length=8192,
        use_cutlass_kernel=True,
    )

    kernel_graph = {
        "type": "transformer_layer",
        "hidden_size": 4096,
        "num_heads": 32,
        "head_dim": 128,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "optimized_kernel.so")

        result_path = optimizer.compile_kernel(config, kernel_graph, output_path)

        print(f"\n  Compilation Output:")
        print(f"  - Kernel path: {result_path}")

        # Read metadata
        meta_path = result_path + ".meta.json"
        with open(meta_path) as f:
            meta = json.load(f)

        print(f"\n  Kernel Metadata:")
        print(f"  - Mode: {meta['config']['mode']}")
        print(f"  - Backend: {meta['config']['backend']}")
        print(f"  - Target CC: {meta['config']['target_cc']}")
        print(f"  - Grid: {meta['config']['grid_dim']}")
        print(f"  - Block: {meta['config']['block_dim']}")
        print(f"  - CUTLASS: {meta['config']['use_cutlass_kernel']}")

    return True


def demo_llm_optimization():
    """Demo 5: LLM layer optimization."""
    print_header("Demo 5: LLM Layer Optimization")

    # Define LLM layers to optimize
    layers = [
        {
            "name": "attention",
            "graph": {
                "type": "attention",
                "batch_size": 1,
                "seq_length": 8192,
                "num_heads": 32,
                "head_dim": 128,
                "estimated_flops": 1 * 8192 * 32 * 128 * 8192 * 4,
            },
        },
        {
            "name": "mlp",
            "graph": {
                "type": "gated_mlp",
                "batch_size": 1,
                "seq_length": 8192,
                "hidden_size": 4096,
                "intermediate_size": 11008,
                "estimated_flops": 1 * 8192 * 4096 * 11008 * 3,
            },
        },
        {
            "name": "rmsnorm",
            "graph": {
                "type": "rmsnorm",
                "batch_size": 1,
                "seq_length": 8192,
                "hidden_size": 4096,
                "estimated_flops": 1 * 8192 * 4096 * 5,
            },
        },
    ]

    # Compact search space for demo
    search_space = ypk.KernelSearchSpace(
        grid_dims=[(1, 1, 1), (2, 1, 1), (4, 1, 1)],
        block_dims=[(128, 1, 1), (256, 1, 1)],
    )

    print(f"\n  Optimizing {len(layers)} LLM layers...")

    results = {}
    for layer in layers:
        result = ypk.optimize_ypk_kernel(
            layer["graph"],
            num_workers=2,
            search_space=search_space,
        )
        results[layer["name"]] = result

        print(f"\n  {layer['name']}:")
        print(f"    Best grid: {result.best_config.grid_dim}")
        print(f"    Best block: {result.best_config.block_dim}")
        print(f"    Latency: {result.best_latency_ms:.4f} ms")

    # Summary
    print(f"\n  Summary:")
    total_latency = sum(r.best_latency_ms for r in results.values())
    print(f"  - Total layer latency: {total_latency:.4f} ms")
    print(f"  - Estimated layer throughput: {1000/total_latency:.1f} layers/sec")

    return results


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("  YPK (YiRage Persistent Kernel) Ray Optimization Demo")
    print("=" * 70)

    demos = [
        ("Basic Optimization", demo_basic_optimization),
        ("Multi-Backend Config", demo_multi_backend),
        ("Distributed Profiling", demo_distributed_profiling),
        ("Kernel Compilation", demo_kernel_compilation),
        ("LLM Layer Optimization", demo_llm_optimization),
    ]

    results = []
    for name, demo_func in demos:
        try:
            result = demo_func()
            results.append((name, True))
        except Exception as e:
            print(f"\n  ✗ Error: {e}")
            import traceback

            traceback.print_exc()
            results.append((name, False))

    # Summary
    print_header("Summary")
    print("\n  Demo Results:")
    for name, success in results:
        status = "✓" if success else "✗"
        print(f"    {status} {name}")

    all_passed = all(s for _, s in results)
    print(f"\n  Overall: {'All demos passed!' if all_passed else 'Some demos failed'}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
