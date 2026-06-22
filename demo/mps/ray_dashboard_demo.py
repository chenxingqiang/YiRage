#!/usr/bin/env python3
"""
Ray Dashboard Visualization Demo for YiRage MPS Search.

This script demonstrates Ray distributed search with visual monitoring.

Usage:
    python ray_dashboard_demo.py

Then open http://127.0.0.1:8265 in your browser to observe:
- Active Workers (Actors)
- Task execution timeline
- Resource utilization
- Object store memory
"""

import ray
import time
import sys
from typing import List, Dict, Any

# =============================================================================
# Ray Task Definitions (visible in Dashboard)
# =============================================================================


@ray.remote(num_cpus=1)
class SearchWorkerActor:
    """
    Search worker actor - visible in Ray Dashboard as an Actor.

    Each actor handles a partition of the search space.
    """

    def __init__(self, worker_id: int, backend: str = "mps"):
        self.worker_id = worker_id
        self.backend = backend
        self.tasks_completed = 0
        self.total_candidates = 0

        # Try to import YiRage search
        self._search_available = False
        try:
            from yirage.core import search

            self._search_available = True
        except ImportError:
            pass

    def get_info(self) -> Dict[str, Any]:
        """Get worker info - visible in Dashboard Actor detail."""
        return {
            "worker_id": self.worker_id,
            "backend": self.backend,
            "tasks_completed": self.tasks_completed,
            "total_candidates": self.total_candidates,
            "search_available": self._search_available,
        }

    def search_partition(
        self,
        graph_data: Dict,
        partition: Dict,
        config: Dict,
    ) -> Dict[str, Any]:
        """
        Execute search on a partition.

        This task is visible in Ray Dashboard timeline.
        """
        start_time = time.time()

        grid_range = partition.get("grid_dim_range", [(1, 1, 1)])
        block_range = partition.get("block_dim_range", [(128, 1, 1)])

        # Simulate search with some real work
        candidates = []
        for grid in grid_range:
            for block in block_range:
                # Simulate GPU verification time
                time.sleep(0.1)  # 100ms per candidate

                # Simple performance model
                if isinstance(grid, (list, tuple)):
                    parallelism = grid[0] * grid[1] * grid[2]
                else:
                    parallelism = 1

                if isinstance(block, (list, tuple)):
                    threads = block[0] * block[1] * block[2]
                else:
                    threads = 128

                latency = 10.0 / (parallelism * (threads / 128))

                candidates.append(
                    {
                        "grid_dim": grid,
                        "block_dim": block,
                        "latency_ms": latency,
                        "verified": True,
                    }
                )

        self.tasks_completed += 1
        self.total_candidates += len(candidates)

        best = min(candidates, key=lambda x: x["latency_ms"]) if candidates else None

        return {
            "worker_id": self.worker_id,
            "partition_id": partition.get("partition_id", 0),
            "num_candidates": len(candidates),
            "best_latency_ms": best["latency_ms"] if best else float("inf"),
            "best_config": best,
            "elapsed_s": time.time() - start_time,
        }


@ray.remote
def aggregate_results(results: List[Dict]) -> Dict[str, Any]:
    """
    Aggregate results from all workers.

    This is a separate task visible in Dashboard.
    """
    all_candidates = 0
    best_latency = float("inf")
    best_config = None

    for r in results:
        all_candidates += r.get("num_candidates", 0)
        if r.get("best_latency_ms", float("inf")) < best_latency:
            best_latency = r["best_latency_ms"]
            best_config = r.get("best_config")

    return {
        "total_candidates": all_candidates,
        "best_latency_ms": best_latency,
        "best_config": best_config,
        "num_workers": len(results),
    }


# =============================================================================
# Main Demo
# =============================================================================


def create_search_space(num_grids: int = 8, num_blocks: int = 4) -> Dict:
    """Create a search space configuration."""
    grid_dims = [(2**i, 1, 1) for i in range(num_grids)]
    block_dims = [(32 * (i + 1), 1, 1) for i in range(num_blocks)]

    return {
        "grid_dims": grid_dims,
        "block_dims": block_dims,
    }


def partition_search_space(search_space: Dict, num_partitions: int) -> List[Dict]:
    """Partition search space across workers."""
    grid_dims = search_space["grid_dims"]
    block_dims = search_space["block_dims"]

    grids_per_partition = max(1, len(grid_dims) // num_partitions)

    partitions = []
    for i in range(num_partitions):
        start = i * grids_per_partition
        end = start + grids_per_partition if i < num_partitions - 1 else len(grid_dims)

        partitions.append(
            {
                "partition_id": i,
                "total_partitions": num_partitions,
                "grid_dim_range": grid_dims[start:end],
                "block_dim_range": block_dims,
            }
        )

    return partitions


def run_distributed_search(
    num_workers: int = 4,
    num_rounds: int = 3,
    verbose: bool = True,
) -> List[Dict]:
    """
    Run multiple rounds of distributed search.

    Each round is visible in Ray Dashboard.
    """
    results = []

    # Create workers (Actors)
    if verbose:
        print(f"\n创建 {num_workers} 个 Worker Actors...")

    workers = [SearchWorkerActor.remote(worker_id=i, backend="mps") for i in range(num_workers)]

    # Wait for workers to be ready
    worker_infos = ray.get([w.get_info.remote() for w in workers])
    if verbose:
        print(f"Workers 已就绪: {[w['worker_id'] for w in worker_infos]}")

    # Run multiple search rounds
    for round_idx in range(num_rounds):
        if verbose:
            print(f"\n{'='*60}")
            print(f"  第 {round_idx + 1}/{num_rounds} 轮搜索")
            print(f"{'='*60}")

        # Create search space (vary size per round)
        num_grids = 4 * (round_idx + 1)
        num_blocks = 3
        search_space = create_search_space(num_grids, num_blocks)

        if verbose:
            print(
                f"搜索空间: {len(search_space['grid_dims'])} grids × {len(search_space['block_dims'])} blocks"
            )

        # Create partitions
        partitions = partition_search_space(search_space, num_workers)

        # Create graph data (put in object store)
        graph_data = ray.put(
            {
                "type": "matmul",
                "shape": [1, 256 * (round_idx + 1), 256],
                "round": round_idx,
            }
        )

        config = {"backend": "mps", "round": round_idx}

        # Launch parallel search tasks
        if verbose:
            print(f"启动 {num_workers} 个并行搜索任务...")

        start_time = time.time()

        search_futures = [
            workers[i].search_partition.remote(graph_data, partitions[i], config)
            for i in range(num_workers)
        ]

        # Wait for all search tasks
        search_results = ray.get(search_futures)

        # Aggregate results (separate task)
        final_result = ray.get(aggregate_results.remote(search_results))

        elapsed = time.time() - start_time
        final_result["round"] = round_idx
        final_result["elapsed_s"] = elapsed
        results.append(final_result)

        if verbose:
            print(f"\n第 {round_idx + 1} 轮结果:")
            print(f"  总候选数: {final_result['total_candidates']}")
            print(f"  最佳延迟: {final_result['best_latency_ms']:.4f} ms")
            print(f"  总耗时: {elapsed:.2f}s")

            for sr in search_results:
                print(
                    f"    Worker {sr['worker_id']}: {sr['num_candidates']} candidates, {sr['elapsed_s']:.2f}s"
                )

    # Get final worker stats
    if verbose:
        print(f"\n{'='*60}")
        print("  Worker 最终统计")
        print(f"{'='*60}")

        final_infos = ray.get([w.get_info.remote() for w in workers])
        for info in final_infos:
            print(
                f"  Worker {info['worker_id']}: "
                f"tasks={info['tasks_completed']}, "
                f"candidates={info['total_candidates']}"
            )

    return results


def main():
    print("=" * 70)
    print("  Ray Dashboard 可视化演示")
    print("  YiRage MPS 分布式搜索")
    print("=" * 70)

    # Initialize Ray with Dashboard
    if not ray.is_initialized():
        ray.init(
            dashboard_host="127.0.0.1",
            dashboard_port=8265,
            include_dashboard=True,
            logging_level="WARNING",
        )

    print()
    print("📊 Ray Dashboard: http://127.0.0.1:8265")
    print()
    print("请在浏览器中打开 Dashboard 查看:")
    print("  - Actors: 查看 SearchWorkerActor 实例")
    print("  - Tasks: 查看任务执行时间线")
    print("  - Metrics: 查看资源使用情况")
    print()

    # Give user time to open dashboard
    print("5 秒后开始执行任务...")
    for i in range(5, 0, -1):
        print(f"  {i}...")
        time.sleep(1)

    # Run distributed search
    try:
        results = run_distributed_search(
            num_workers=4,
            num_rounds=3,
            verbose=True,
        )

        print()
        print("=" * 70)
        print("  ✅ 演示完成!")
        print("=" * 70)
        print()
        print("Dashboard 将保持运行 30 秒，供您查看任务历史...")
        print("按 Ctrl+C 退出")

        time.sleep(30)

    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        ray.shutdown()
        print("Ray 已关闭")


if __name__ == "__main__":
    main()
