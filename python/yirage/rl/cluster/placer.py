"""
Device Placement Strategies

Assigns sub-tasks to devices for optimal execution.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from abc import ABC, abstractmethod
import numpy as np

from .topology import ClusterTopology, DeviceSpec
from .task import TaskGraph, SubTask, ComputeTask


class PlacementStrategy(ABC):
    """Base class for placement strategies."""

    @abstractmethod
    def place(
        self,
        task_graph: TaskGraph,
        cluster: ClusterTopology,
    ) -> Dict[str, str]:
        """
        Assign subtasks to devices.

        Returns:
            Mapping from subtask_id to device_id
        """
        pass


@dataclass
class GreedyPlacer(PlacementStrategy):
    """
    Greedy placement: assign each subtask to the fastest available device.
    """

    def place(
        self,
        task_graph: TaskGraph,
        cluster: ClusterTopology,
    ) -> Dict[str, str]:
        """Greedy placement by estimated load."""

        devices = cluster.all_devices()
        device_load = {d[0]: 0.0 for d in devices}

        placement = {}

        for subtask in task_graph.subtasks:
            # Find device with lowest load
            best_device = min(device_load.keys(), key=lambda d: device_load[d])

            placement[subtask.subtask_id] = best_device

            # Update load estimate
            device_load[best_device] += subtask.estimated_time_ms

        return placement


@dataclass
class DPPlacer(PlacementStrategy):
    """
    Dynamic Programming based placer for optimal placement.
    Considers both compute and communication costs.
    """

    def place(
        self,
        task_graph: TaskGraph,
        cluster: ClusterTopology,
    ) -> Dict[str, str]:
        """DP-based optimal placement."""

        subtasks = task_graph.subtasks
        devices = cluster.all_devices()

        if not subtasks or not devices:
            return {}

        n_subtasks = len(subtasks)
        n_devices = len(devices)

        # DP table: cost[i][d] = min cost to execute subtasks 0..i with subtask i on device d
        INF = float("inf")
        cost = [[INF] * n_devices for _ in range(n_subtasks)]
        parent = [[-1] * n_devices for _ in range(n_subtasks)]

        # Base case: first subtask
        for d in range(n_devices):
            device_id, device_spec = devices[d]
            cost[0][d] = self._estimate_cost(subtasks[0], device_spec, cluster)

        # Fill DP table
        for i in range(1, n_subtasks):
            for d in range(n_devices):
                device_id, device_spec = devices[d]
                subtask_cost = self._estimate_cost(subtasks[i], device_spec, cluster)

                for prev_d in range(n_devices):
                    prev_device_id = devices[prev_d][0]

                    # Communication cost if different devices
                    comm_cost = 0.0
                    if d != prev_d:
                        comm_cost = self._estimate_comm_cost(
                            subtasks[i - 1],
                            subtasks[i],
                            prev_device_id,
                            device_id,
                            cluster,
                            task_graph.original_task,
                        )

                    total = cost[i - 1][prev_d] + subtask_cost + comm_cost

                    if total < cost[i][d]:
                        cost[i][d] = total
                        parent[i][d] = prev_d

        # Backtrack to find optimal placement
        placement = {}

        # Find best final device
        best_d = min(range(n_devices), key=lambda d: cost[n_subtasks - 1][d])

        # Backtrack
        d = best_d
        for i in range(n_subtasks - 1, -1, -1):
            placement[subtasks[i].subtask_id] = devices[d][0]
            if i > 0:
                d = parent[i][d]

        return placement

    def _estimate_cost(
        self,
        subtask: SubTask,
        device: DeviceSpec,
        cluster: ClusterTopology,
    ) -> float:
        """Estimate execution cost on a device."""

        if subtask.estimated_time_ms > 0:
            # Scale by device performance relative to baseline
            baseline_tflops = 100.0
            scale = baseline_tflops / max(device.peak_compute("fp16"), 1.0)
            return subtask.estimated_time_ms * scale

        # Estimate from FLOPs
        if subtask.estimated_flops > 0:
            peak = device.peak_compute("fp16") * 1e12
            return (subtask.estimated_flops / peak) * 1000 * 1.2

        return 1.0  # Default

    def _estimate_comm_cost(
        self,
        src_subtask: SubTask,
        dst_subtask: SubTask,
        src_device: str,
        dst_device: str,
        cluster: ClusterTopology,
        task: ComputeTask,
    ) -> float:
        """Estimate communication cost between subtasks."""

        if src_device == dst_device:
            return 0.0

        # Estimate data transfer size
        size = 0
        for out in src_subtask.external_outputs:
            if out in task.tensors:
                size += task.tensors[out].size_bytes()

        if size == 0:
            return 0.0

        return cluster.transfer_time_ms(src_device, dst_device, size)


@dataclass
class LearnedPlacer(PlacementStrategy):
    """
    ML-based placement using learned policy.
    """

    # Feature dimension
    feature_dim: int = 64

    # Policy weights (simple linear model, can be extended to NN)
    weights: Optional[np.ndarray] = None

    def __post_init__(self):
        """Initialize weights."""
        if self.weights is None:
            self.weights = np.zeros((self.feature_dim, 1))

    def place(
        self,
        task_graph: TaskGraph,
        cluster: ClusterTopology,
    ) -> Dict[str, str]:
        """Learned placement."""

        subtasks = task_graph.subtasks
        devices = cluster.all_devices()

        if not subtasks or not devices:
            return {}

        placement = {}

        for subtask in subtasks:
            # Compute scores for each device
            scores = []
            for device_id, device_spec in devices:
                features = self._compute_features(subtask, device_spec, cluster, task_graph)
                score = np.dot(features, self.weights).item()
                scores.append((device_id, score))

            # Select best device
            best_device = max(scores, key=lambda x: x[1])[0]
            placement[subtask.subtask_id] = best_device

        return placement

    def _compute_features(
        self,
        subtask: SubTask,
        device: DeviceSpec,
        cluster: ClusterTopology,
        task_graph: TaskGraph,
    ) -> np.ndarray:
        """Compute features for placement decision."""

        features = np.zeros(self.feature_dim)

        # Subtask features
        features[0] = subtask.estimated_flops / 1e12
        features[1] = subtask.estimated_memory_bytes / 1e9
        features[2] = len(subtask.operators) / 10
        features[3] = len(subtask.external_inputs) / 5
        features[4] = len(subtask.external_outputs) / 5

        # Device features
        device_feats = device.to_feature_vector()
        features[10 : 10 + len(device_feats)] = device_feats

        return features

    def update_weights(
        self,
        task_graph: TaskGraph,
        placement: Dict[str, str],
        reward: float,
        cluster: ClusterTopology,
        learning_rate: float = 0.01,
    ):
        """Update weights based on feedback."""

        devices = {d[0]: d[1] for d in cluster.all_devices()}

        for subtask in task_graph.subtasks:
            device_id = placement.get(subtask.subtask_id)
            if device_id and device_id in devices:
                device = devices[device_id]
                features = self._compute_features(subtask, device, cluster, task_graph)

                # Simple gradient update
                gradient = features.reshape(-1, 1) * reward
                self.weights += learning_rate * gradient


class DevicePlacer:
    """
    Main interface for device placement.
    Combines multiple strategies.
    """

    def __init__(self, default_strategy: str = "dp"):
        """Initialize with default strategy."""
        self.strategies = {
            "greedy": GreedyPlacer(),
            "dp": DPPlacer(),
            "learned": LearnedPlacer(),
        }
        self.default_strategy = default_strategy

    def place(
        self,
        task_graph: TaskGraph,
        cluster: ClusterTopology,
        strategy: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Place subtasks on devices.

        Args:
            task_graph: The task graph to place
            cluster: Target cluster topology
            strategy: Placement strategy (greedy, dp, learned)

        Returns:
            Mapping from subtask_id to device_id
        """

        strategy = strategy or self.default_strategy
        placer = self.strategies.get(strategy, self.strategies["greedy"])

        return placer.place(task_graph, cluster)

    def find_best_placement(
        self,
        task_graph: TaskGraph,
        cluster: ClusterTopology,
    ) -> Tuple[Dict[str, str], str]:
        """
        Try all strategies and return the best placement.

        Returns:
            (placement, strategy_name)
        """

        from .simulator import ClusterSimulator

        simulator = ClusterSimulator(cluster)

        best_placement = None
        best_strategy = None
        best_time = float("inf")

        for name, strategy in self.strategies.items():
            try:
                placement = strategy.place(task_graph, cluster)
                sim = simulator.simulate_execution(task_graph, placement)

                if sim.total_time_ms < best_time:
                    best_time = sim.total_time_ms
                    best_placement = placement
                    best_strategy = name
            except Exception:
                continue

        return best_placement, best_strategy
