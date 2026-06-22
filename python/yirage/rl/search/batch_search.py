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
Batch Search API (Problem 3)

Eliminates per-step JSON serialization overhead by batching multiple
search configurations into a single C++ call.

Instead of:
    for config in configs:
        result = cpp_search(json.dumps(config))  # 2x JSON per step

We do:
    results = batch_search(configs)  # 1x JSON for N configs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import numpy as np
import time


@dataclass
class BatchSearchConfig:
    """Configuration for batch search."""

    # Batch size for parallel search
    batch_size: int = 16
    # Maximum threads for C++ search
    max_threads: int = 4
    # Use shared memory for large graphs (vs JSON)
    use_shared_memory: bool = False
    # Timeout per search (seconds)
    timeout_seconds: float = 30.0


@dataclass
class KernelSearchResult:
    """Result from a single kernel search."""

    config_id: int = 0
    verified: bool = False
    latency_ms: float = float("inf")
    energy_pj: float = 0.0
    kernel_graph_json: str = "{}"
    search_time_ms: float = 0.0
    num_graphs_explored: int = 0
    rejection_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config_id": self.config_id,
            "verified": self.verified,
            "latency_ms": self.latency_ms,
            "energy_pj": self.energy_pj,
            "search_time_ms": self.search_time_ms,
            "num_graphs_explored": self.num_graphs_explored,
            "rejection_reason": self.rejection_reason,
        }


class BatchSearchAPI:
    """
    Batch search interface for high-throughput kernel search.

    Sends N configurations in one call, reducing Python↔C++ boundary
    crossings from 2N to 2 (one serialize, one deserialize).

    Supports three modes:
    1. Sequential: Process configs one at a time (baseline)
    2. Threaded: Use Python ThreadPoolExecutor
    3. Native: Use C++ internal parallelism (when available)
    """

    def __init__(self, config: Optional[BatchSearchConfig] = None):
        self.config = config or BatchSearchConfig()
        self._cpp_available = False
        self._init_cpp()

    def _init_cpp(self):
        """Try to initialize C++ batch search backend."""
        try:
            from yirage import core
            if hasattr(core, "batch_search"):
                self._cpp_available = True
        except ImportError:
            pass

    def search_batch(
        self,
        target_graph_json: str,
        configs: List[Dict[str, Any]],
    ) -> List[KernelSearchResult]:
        """
        Search for kernels across multiple configurations in batch.

        Args:
            target_graph_json: Target computation graph
            configs: List of HardwareConfig dicts

        Returns:
            List of KernelSearchResult, one per config
        """
        if not configs:
            return []

        if self._cpp_available:
            return self._native_batch_search(target_graph_json, configs)
        else:
            return self._threaded_batch_search(target_graph_json, configs)

    def _native_batch_search(
        self,
        target_graph_json: str,
        configs: List[Dict[str, Any]],
    ) -> List[KernelSearchResult]:
        """
        Use C++ native batch search for maximum throughput.

        Single JSON payload → C++ parallel DFS → single JSON result.
        """
        try:
            from yirage import core

            # Pack all configs into single JSON
            batch_payload = json.dumps({
                "target_graph": target_graph_json,
                "configs": configs,
                "max_threads": self.config.max_threads,
                "timeout_seconds": self.config.timeout_seconds,
            })

            # Single C++ call
            result_json = core.batch_search(batch_payload)
            results_data = json.loads(result_json)

            return [
                KernelSearchResult(
                    config_id=i,
                    verified=r.get("verified", False),
                    latency_ms=r.get("latency_ms", float("inf")),
                    energy_pj=r.get("energy_pj", 0.0),
                    kernel_graph_json=r.get("kernel_graph_json", "{}"),
                    search_time_ms=r.get("search_time_ms", 0.0),
                    num_graphs_explored=r.get("num_graphs_explored", 0),
                )
                for i, r in enumerate(results_data.get("results", []))
            ]
        except Exception:
            # Fallback to threaded
            return self._threaded_batch_search(target_graph_json, configs)

    def _threaded_batch_search(
        self,
        target_graph_json: str,
        configs: List[Dict[str, Any]],
    ) -> List[KernelSearchResult]:
        """
        Use Python ThreadPoolExecutor for parallel search.

        Falls back when C++ batch search is not available.
        """
        results: List[Optional[KernelSearchResult]] = [None] * len(configs)

        with ThreadPoolExecutor(max_workers=self.config.max_threads) as executor:
            futures = {}
            for i, config in enumerate(configs):
                future = executor.submit(
                    self._search_single, target_graph_json, config, i
                )
                futures[future] = i

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result(
                        timeout=self.config.timeout_seconds
                    )
                except Exception as e:
                    results[idx] = KernelSearchResult(
                        config_id=idx,
                        verified=False,
                        rejection_reason=str(e),
                    )

        return [r if r is not None else KernelSearchResult(config_id=i)
                for i, r in enumerate(results)]

    def _search_single(
        self,
        target_graph_json: str,
        config: Dict[str, Any],
        config_id: int,
    ) -> KernelSearchResult:
        """Search a single configuration (used in threaded mode)."""
        start = time.monotonic()

        try:
            from yirage import core
            result_json = core.search(
                json.dumps({
                    "target_graph": target_graph_json,
                    "config": config,
                })
            )
            result = json.loads(result_json)
            elapsed = (time.monotonic() - start) * 1000

            return KernelSearchResult(
                config_id=config_id,
                verified=result.get("verified", False),
                latency_ms=result.get("latency_ms", float("inf")),
                kernel_graph_json=result.get("kernel_graph_json", "{}"),
                search_time_ms=elapsed,
                num_graphs_explored=result.get("num_graphs_explored", 0),
            )
        except ImportError:
            # C++ not available — simulate
            elapsed = (time.monotonic() - start) * 1000
            return KernelSearchResult(
                config_id=config_id,
                verified=False,
                search_time_ms=elapsed,
                rejection_reason="cpp_not_available",
            )

    def search_with_expert_warmstart(
        self,
        target_graph_json: str,
        configs: List[Dict[str, Any]],
        expert_results: Optional[List[Dict[str, Any]]] = None,
    ) -> List[KernelSearchResult]:
        """
        Batch search with expert demonstrations as warm start.

        Uses C++ DFS results (expert_results) to seed the search
        and focus on nearby configurations (Problem 6b).
        """
        if expert_results:
            # Add expert-adjacent configs to the batch
            augmented_configs = list(configs)
            for expert in expert_results:
                expert_config = expert.get("config", {})
                # Generate nearby configs by perturbing expert config
                nearby = self._perturb_config(expert_config)
                augmented_configs.extend(nearby)
            configs = augmented_configs

        return self.search_batch(target_graph_json, configs)

    @staticmethod
    def _perturb_config(config: Dict[str, Any], n_perturbations: int = 3) -> List[Dict[str, Any]]:
        """Generate nearby configurations by small perturbations."""
        perturbations = []
        rng = np.random.default_rng()

        for _ in range(n_perturbations):
            perturbed = dict(config)
            # Perturb one dimension at a time
            keys_to_perturb = ["grid_dim_x", "block_dim_x", "forloop_range"]
            for key in keys_to_perturb:
                if key in perturbed and isinstance(perturbed[key], (int, float)):
                    val = perturbed[key]
                    # ±1 step in the discrete space
                    delta = rng.choice([-1, 0, 1])
                    perturbed[key] = max(1, int(val) + delta)
            perturbations.append(perturbed)

        return perturbations
