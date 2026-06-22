"""
Cluster Simulator

Simulates multi-device cluster execution on a single GPU.
Models communication costs accurately for optimization.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
import numpy as np
import time
import json

from .topology import ClusterTopology, DeviceSpec, DeviceType
from .task import ComputeTask, TaskGraph, SubTask, Operator, TensorSpec


class CommunicationType(Enum):
    """Types of collective communication."""

    P2P = "p2p"  # Point-to-point
    ALL_REDUCE = "all_reduce"  # Sum across all devices
    ALL_GATHER = "all_gather"  # Gather tensors from all devices
    REDUCE_SCATTER = "reduce_scatter"
    ALL_TO_ALL = "all_to_all"
    BROADCAST = "broadcast"


@dataclass
class CommunicationModel:
    """
    Base class for communication cost models.
    Estimates time for different collective operations.
    """

    def p2p_time_ms(
        self,
        size_bytes: int,
        bandwidth_gbps: float,
        latency_us: float,
    ) -> float:
        """Point-to-point transfer time."""
        size_gb = size_bytes / (1024**3)
        transfer_ms = (size_gb / bandwidth_gbps) * 1000
        latency_ms = latency_us / 1000
        return transfer_ms + latency_ms

    def all_reduce_time_ms(
        self,
        size_bytes: int,
        num_devices: int,
        bandwidth_gbps: float,
        latency_us: float,
        algorithm: str = "ring",
    ) -> float:
        """AllReduce collective time."""
        if algorithm == "ring":
            # Ring all-reduce: 2*(n-1)/n * size / bandwidth
            factor = 2 * (num_devices - 1) / num_devices
            size_gb = size_bytes / (1024**3)
            transfer_ms = (factor * size_gb / bandwidth_gbps) * 1000
            latency_ms = 2 * (num_devices - 1) * latency_us / 1000
            return transfer_ms + latency_ms

        elif algorithm == "tree":
            # Tree all-reduce: 2*log(n) * size / bandwidth
            factor = 2 * np.log2(num_devices)
            size_gb = size_bytes / (1024**3)
            transfer_ms = (factor * size_gb / bandwidth_gbps) * 1000
            latency_ms = 2 * np.log2(num_devices) * latency_us / 1000
            return transfer_ms + latency_ms

        return self.p2p_time_ms(size_bytes, bandwidth_gbps, latency_us) * num_devices

    def all_gather_time_ms(
        self,
        size_bytes: int,
        num_devices: int,
        bandwidth_gbps: float,
        latency_us: float,
    ) -> float:
        """AllGather collective time."""
        # Ring: (n-1)/n * n * size / bandwidth = (n-1) * size / bandwidth
        factor = num_devices - 1
        size_gb = size_bytes / (1024**3)
        transfer_ms = (factor * size_gb / bandwidth_gbps) * 1000
        latency_ms = (num_devices - 1) * latency_us / 1000
        return transfer_ms + latency_ms

    def reduce_scatter_time_ms(
        self,
        size_bytes: int,
        num_devices: int,
        bandwidth_gbps: float,
        latency_us: float,
    ) -> float:
        """ReduceScatter collective time."""
        # Same as AllGather
        return self.all_gather_time_ms(size_bytes, num_devices, bandwidth_gbps, latency_us)


@dataclass
class NVLinkModel(CommunicationModel):
    """Communication model for NVLink interconnect."""

    base_bandwidth_gbps: float = 300.0  # NVLink 3.0
    base_latency_us: float = 1.0

    def p2p_time_ms(self, size_bytes: int, **kwargs) -> float:
        return super().p2p_time_ms(
            size_bytes,
            kwargs.get("bandwidth_gbps", self.base_bandwidth_gbps),
            kwargs.get("latency_us", self.base_latency_us),
        )


@dataclass
class PCIeModel(CommunicationModel):
    """Communication model for PCIe interconnect."""

    base_bandwidth_gbps: float = 32.0  # PCIe 4.0 x16
    base_latency_us: float = 5.0


@dataclass
class InfiniBandModel(CommunicationModel):
    """Communication model for InfiniBand network."""

    base_bandwidth_gbps: float = 200.0  # HDR
    base_latency_us: float = 2.0


@dataclass
class EthernetModel(CommunicationModel):
    """Communication model for Ethernet network."""

    base_bandwidth_gbps: float = 100.0  # 100GbE
    base_latency_us: float = 10.0


@dataclass
class ComputeEvent:
    """A compute event in the simulation timeline."""

    event_id: str
    device_id: str
    operator_id: str

    start_time_ms: float
    end_time_ms: float

    flops: int = 0
    memory_bytes: int = 0


@dataclass
class CommunicationEvent:
    """A communication event in the simulation timeline."""

    event_id: str
    comm_type: CommunicationType

    src_devices: List[str]
    dst_devices: List[str]

    tensor_name: str
    size_bytes: int

    start_time_ms: float
    end_time_ms: float


@dataclass
class SimulatedExecution:
    """Result of a simulated execution."""

    # Timeline
    compute_events: List[ComputeEvent] = field(default_factory=list)
    comm_events: List[CommunicationEvent] = field(default_factory=list)

    # Metrics
    total_time_ms: float = 0.0
    compute_time_ms: float = 0.0
    comm_time_ms: float = 0.0

    # Per-device utilization
    device_utilization: Dict[str, float] = field(default_factory=dict)

    # Breakdown
    comm_breakdown: Dict[str, float] = field(default_factory=dict)

    def compute_efficiency(self) -> float:
        """Compute efficiency (compute time / total time)."""
        if self.total_time_ms > 0:
            return self.compute_time_ms / self.total_time_ms
        return 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "total_time_ms": self.total_time_ms,
            "compute_time_ms": self.compute_time_ms,
            "comm_time_ms": self.comm_time_ms,
            "compute_efficiency": self.compute_efficiency(),
            "device_utilization": self.device_utilization,
            "comm_breakdown": self.comm_breakdown,
            "num_compute_events": len(self.compute_events),
            "num_comm_events": len(self.comm_events),
        }


@dataclass
class ClusterSimulator:
    """
    Simulates cluster execution on a single device.

    Key features:
    - Accurate communication modeling
    - Overlap of compute and communication
    - Multiple scheduling strategies
    """

    topology: ClusterTopology
    comm_model: CommunicationModel = field(default_factory=CommunicationModel)

    # Performance model cache
    _perf_cache: Dict[str, float] = field(default_factory=dict, repr=False)

    def estimate_operator_time_ms(
        self,
        operator: Operator,
        device: DeviceSpec,
        tensor_specs: Dict[str, TensorSpec],
    ) -> float:
        """Estimate execution time for an operator on a device."""

        flops = operator.estimate_flops(tensor_specs)
        memory_bytes = operator.estimate_memory_bytes(tensor_specs)

        # Get device capabilities
        peak_tflops = device.peak_compute("fp16")
        memory_bw_gbps = device.memory_bandwidth_gbps

        # Roofline model
        compute_time_ms = (flops / (peak_tflops * 1e12)) * 1000
        memory_time_ms = (memory_bytes / (memory_bw_gbps * 1e9)) * 1000

        # Take max (roofline bound)
        base_time = max(compute_time_ms, memory_time_ms)

        # Add overhead factor (kernel launch, etc.)
        overhead_factor = 1.1

        return base_time * overhead_factor

    def simulate_execution(
        self,
        task_graph: TaskGraph,
        placement: Dict[str, str],  # subtask_id -> device_id
        schedule: Optional[List[str]] = None,  # Execution order
    ) -> SimulatedExecution:
        """
        Simulate execution of a task graph with given placement.

        Args:
            task_graph: The decomposed task graph
            placement: Mapping from subtask to device
            schedule: Optional execution order

        Returns:
            SimulatedExecution with timing information
        """

        result = SimulatedExecution()

        # Device availability times
        device_available = {d[0]: 0.0 for d in self.topology.all_devices()}

        # Tensor locations and availability
        tensor_ready = {}  # tensor_name -> (device, time)

        # Get execution order
        if schedule is None:
            schedule = [st.subtask_id for st in task_graph.subtasks]

        # Build operator lookup
        op_lookup = {op.op_id: op for op in task_graph.original_task.operators}
        tensor_specs = task_graph.original_task.tensors

        # Simulate each subtask
        for subtask_id in schedule:
            subtask = next((s for s in task_graph.subtasks if s.subtask_id == subtask_id), None)
            if not subtask:
                continue

            device_id = placement.get(subtask_id, "node0/gpu0")
            device = None
            for d_id, d_spec in self.topology.all_devices():
                if d_id == device_id:
                    device = d_spec
                    break

            if device is None:
                continue

            # Start time: max of device availability and input availability
            start_time = device_available.get(device_id, 0.0)

            # Check input dependencies
            for inp in subtask.external_inputs:
                if inp in tensor_ready:
                    src_device, ready_time = tensor_ready[inp]
                    if src_device != device_id:
                        # Need to transfer
                        tensor_spec = tensor_specs.get(inp)
                        if tensor_spec:
                            comm_time = self.topology.transfer_time_ms(
                                src_device, device_id, tensor_spec.size_bytes()
                            )

                            result.comm_events.append(
                                CommunicationEvent(
                                    event_id=f"p2p_{inp}",
                                    comm_type=CommunicationType.P2P,
                                    src_devices=[src_device],
                                    dst_devices=[device_id],
                                    tensor_name=inp,
                                    size_bytes=tensor_spec.size_bytes(),
                                    start_time_ms=ready_time,
                                    end_time_ms=ready_time + comm_time,
                                )
                            )

                            start_time = max(start_time, ready_time + comm_time)
                    else:
                        start_time = max(start_time, ready_time)

            # Execute operators
            current_time = start_time
            for op_id in subtask.operators:
                op = op_lookup.get(op_id)
                if op is None:
                    continue

                op_time = self.estimate_operator_time_ms(op, device, tensor_specs)

                result.compute_events.append(
                    ComputeEvent(
                        event_id=f"compute_{op_id}",
                        device_id=device_id,
                        operator_id=op_id,
                        start_time_ms=current_time,
                        end_time_ms=current_time + op_time,
                        flops=op.estimate_flops(tensor_specs),
                        memory_bytes=op.estimate_memory_bytes(tensor_specs),
                    )
                )

                current_time += op_time

                # Update tensor locations
                for out in op.outputs:
                    tensor_ready[out] = (device_id, current_time)

            # Update device availability
            device_available[device_id] = current_time

            # Update output tensor locations
            for out in subtask.external_outputs:
                tensor_ready[out] = (device_id, current_time)

        # Calculate metrics
        if result.compute_events:
            result.total_time_ms = max(e.end_time_ms for e in result.compute_events)
            result.compute_time_ms = sum(
                e.end_time_ms - e.start_time_ms for e in result.compute_events
            )

        if result.comm_events:
            result.comm_time_ms = sum(e.end_time_ms - e.start_time_ms for e in result.comm_events)

        # Device utilization
        for device_id, _ in self.topology.all_devices():
            device_compute = sum(
                e.end_time_ms - e.start_time_ms
                for e in result.compute_events
                if e.device_id == device_id
            )
            if result.total_time_ms > 0:
                result.device_utilization[device_id] = device_compute / result.total_time_ms
            else:
                result.device_utilization[device_id] = 0.0

        return result

    def simulate_data_parallel(
        self,
        task: ComputeTask,
        num_devices: int,
        batch_size: int,
    ) -> SimulatedExecution:
        """
        Simulate data-parallel execution.

        Splits batch across devices, performs forward pass,
        then all-reduce gradients.
        """

        devices = self.topology.all_devices()[:num_devices]

        result = SimulatedExecution()

        # Batch per device
        batch_per_device = batch_size // num_devices

        # Forward pass (parallel on all devices)
        forward_times = []
        for device_id, device_spec in devices:
            # Scale task for this batch size
            total_flops = task.total_flops() * batch_per_device
            peak_tflops = device_spec.peak_compute("fp16")
            forward_time = (total_flops / (peak_tflops * 1e12)) * 1000 * 1.2  # overhead
            forward_times.append(forward_time)

            result.compute_events.append(
                ComputeEvent(
                    event_id=f"forward_{device_id}",
                    device_id=device_id,
                    operator_id="forward",
                    start_time_ms=0.0,
                    end_time_ms=forward_time,
                    flops=total_flops,
                )
            )

        forward_end = max(forward_times)

        # All-reduce gradients
        gradient_size = task.total_memory_bytes()
        bw_matrix = self.topology.get_bandwidth_matrix()
        avg_bandwidth = np.mean(bw_matrix[bw_matrix > 0]) if np.any(bw_matrix > 0) else 100.0

        allreduce_time = self.comm_model.all_reduce_time_ms(
            gradient_size,
            num_devices,
            avg_bandwidth,
            latency_us=1.0,
        )

        result.comm_events.append(
            CommunicationEvent(
                event_id="allreduce_grads",
                comm_type=CommunicationType.ALL_REDUCE,
                src_devices=[d[0] for d in devices],
                dst_devices=[d[0] for d in devices],
                tensor_name="gradients",
                size_bytes=gradient_size,
                start_time_ms=forward_end,
                end_time_ms=forward_end + allreduce_time,
            )
        )

        result.total_time_ms = forward_end + allreduce_time
        result.compute_time_ms = sum(forward_times)
        result.comm_time_ms = allreduce_time
        result.comm_breakdown["all_reduce"] = allreduce_time

        return result

    def simulate_tensor_parallel(
        self,
        task: ComputeTask,
        num_devices: int,
        partition_dim: int = -1,
    ) -> SimulatedExecution:
        """
        Simulate tensor-parallel execution.

        Partitions tensors across devices, with all-gather for dependencies.
        """

        devices = self.topology.all_devices()[:num_devices]

        result = SimulatedExecution()

        # Each device handles 1/num_devices of the computation
        total_flops = task.total_flops() // num_devices

        current_time = 0.0

        for i, op in enumerate(task.operators):
            # Compute
            device_id, device_spec = devices[0]  # Use first device as reference
            peak_tflops = device_spec.peak_compute("fp16")
            op_flops = op.estimate_flops(task.tensors) // num_devices
            op_time = (op_flops / (peak_tflops * 1e12)) * 1000 * 1.1

            for d_id, _ in devices:
                result.compute_events.append(
                    ComputeEvent(
                        event_id=f"compute_{op.op_id}_{d_id}",
                        device_id=d_id,
                        operator_id=op.op_id,
                        start_time_ms=current_time,
                        end_time_ms=current_time + op_time,
                        flops=op_flops,
                    )
                )

            current_time += op_time

            # All-gather after each matmul for tensor parallel
            if op.op_type.value in ("matmul", "batch_matmul"):
                out_tensor = task.tensors.get(op.outputs[0]) if op.outputs else None
                if out_tensor:
                    # Each device has 1/N of output, need to gather
                    gather_size = out_tensor.size_bytes() // num_devices

                    bw_matrix = self.topology.get_bandwidth_matrix()
                    avg_bandwidth = (
                        np.mean(bw_matrix[bw_matrix > 0]) if np.any(bw_matrix > 0) else 100.0
                    )

                    gather_time = self.comm_model.all_gather_time_ms(
                        gather_size, num_devices, avg_bandwidth, 1.0
                    )

                    result.comm_events.append(
                        CommunicationEvent(
                            event_id=f"allgather_{op.op_id}",
                            comm_type=CommunicationType.ALL_GATHER,
                            src_devices=[d[0] for d in devices],
                            dst_devices=[d[0] for d in devices],
                            tensor_name=op.outputs[0],
                            size_bytes=out_tensor.size_bytes(),
                            start_time_ms=current_time,
                            end_time_ms=current_time + gather_time,
                        )
                    )

                    current_time += gather_time

        result.total_time_ms = current_time
        result.compute_time_ms = (
            sum(e.end_time_ms - e.start_time_ms for e in result.compute_events) / num_devices
        )
        result.comm_time_ms = sum(e.end_time_ms - e.start_time_ms for e in result.comm_events)

        return result

    def simulate_pipeline_parallel(
        self,
        task: ComputeTask,
        num_stages: int,
        num_micro_batches: int,
    ) -> SimulatedExecution:
        """
        Simulate pipeline-parallel execution.

        Partitions operators across stages, overlaps micro-batches.
        """

        devices = self.topology.all_devices()[:num_stages]

        result = SimulatedExecution()

        # Divide operators into stages
        ops = task.operators
        ops_per_stage = len(ops) // num_stages

        stages = []
        for i in range(num_stages):
            start = i * ops_per_stage
            end = start + ops_per_stage if i < num_stages - 1 else len(ops)
            stages.append(ops[start:end])

        # Stage compute times
        stage_times = []
        for stage_idx, stage_ops in enumerate(stages):
            device_id, device_spec = devices[stage_idx]
            peak_tflops = device_spec.peak_compute("fp16")

            stage_flops = sum(op.estimate_flops(task.tensors) for op in stage_ops)
            stage_time = (stage_flops / (peak_tflops * 1e12)) * 1000 * 1.1
            stage_times.append(stage_time)

        # Simulate 1F1B schedule
        # Timeline: [stage_idx] -> [(start, end), ...]
        timelines = {i: [] for i in range(num_stages)}

        for mb in range(num_micro_batches):
            for stage_idx in range(num_stages):
                # Wait for previous stage and previous micro-batch
                prev_mb_end = timelines[stage_idx][-1][1] if timelines[stage_idx] else 0.0
                prev_stage_end = (
                    timelines[stage_idx - 1][-1][1]
                    if stage_idx > 0 and len(timelines[stage_idx - 1]) > mb
                    else 0.0
                )

                start = max(prev_mb_end, prev_stage_end)
                end = start + stage_times[stage_idx]

                timelines[stage_idx].append((start, end))

                device_id = devices[stage_idx][0]
                result.compute_events.append(
                    ComputeEvent(
                        event_id=f"stage{stage_idx}_mb{mb}",
                        device_id=device_id,
                        operator_id=f"stage_{stage_idx}",
                        start_time_ms=start,
                        end_time_ms=end,
                    )
                )

        # Add p2p communication between stages
        bw_matrix = self.topology.get_bandwidth_matrix()
        for mb in range(num_micro_batches):
            for stage_idx in range(num_stages - 1):
                # Get activation size between stages
                stage_ops = stages[stage_idx]
                if stage_ops and stage_ops[-1].outputs:
                    out_name = stage_ops[-1].outputs[0]
                    out_tensor = task.tensors.get(out_name)
                    if out_tensor:
                        src_id = devices[stage_idx][0]
                        dst_id = devices[stage_idx + 1][0]

                        p2p_time = self.topology.transfer_time_ms(
                            src_id, dst_id, out_tensor.size_bytes()
                        )

                        start = timelines[stage_idx][mb][1]

                        result.comm_events.append(
                            CommunicationEvent(
                                event_id=f"p2p_stage{stage_idx}_{stage_idx+1}_mb{mb}",
                                comm_type=CommunicationType.P2P,
                                src_devices=[src_id],
                                dst_devices=[dst_id],
                                tensor_name=out_name,
                                size_bytes=out_tensor.size_bytes(),
                                start_time_ms=start,
                                end_time_ms=start + p2p_time,
                            )
                        )

        # Calculate metrics
        all_events = result.compute_events
        if all_events:
            result.total_time_ms = max(e.end_time_ms for e in all_events)
            result.compute_time_ms = sum(e.end_time_ms - e.start_time_ms for e in all_events)

        if result.comm_events:
            result.comm_time_ms = sum(e.end_time_ms - e.start_time_ms for e in result.comm_events)

        return result

    def find_optimal_parallelism(
        self,
        task: ComputeTask,
        batch_size: int,
        max_devices: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Find optimal parallelism strategy for a task.

        Tries different combinations and returns the best one.
        """

        if max_devices is None:
            max_devices = self.topology.num_devices()

        results = []

        # Try data parallel
        for dp in range(1, max_devices + 1):
            if batch_size % dp == 0:
                sim = self.simulate_data_parallel(task, dp, batch_size)
                results.append(
                    {
                        "strategy": "data_parallel",
                        "num_devices": dp,
                        "time_ms": sim.total_time_ms,
                        "efficiency": sim.compute_efficiency(),
                        "simulation": sim,
                    }
                )

        # Try tensor parallel
        for tp in [2, 4, 8]:
            if tp <= max_devices:
                sim = self.simulate_tensor_parallel(task, tp)
                results.append(
                    {
                        "strategy": "tensor_parallel",
                        "num_devices": tp,
                        "time_ms": sim.total_time_ms,
                        "efficiency": sim.compute_efficiency(),
                        "simulation": sim,
                    }
                )

        # Try pipeline parallel
        for pp in [2, 4]:
            if pp <= max_devices and pp <= len(task.operators):
                for mb in [4, 8, 16]:
                    sim = self.simulate_pipeline_parallel(task, pp, mb)
                    results.append(
                        {
                            "strategy": "pipeline_parallel",
                            "num_stages": pp,
                            "micro_batches": mb,
                            "time_ms": sim.total_time_ms,
                            "efficiency": sim.compute_efficiency(),
                            "simulation": sim,
                        }
                    )

        # Find best
        if results:
            best = min(results, key=lambda x: x["time_ms"])
            return {
                "best": best,
                "all_results": results,
            }

        return {"best": None, "all_results": []}


# =============================================================================
# COMET: Compound Operation Cost Model with Explicit Collectives
# =============================================================================
# Reference: COMET paper (Negi et al.)
# "A Framework for Modeling Compound Operation Dataflows with Explicit Collectives"


class SchedulingStrategy(Enum):
    """
    Scheduling strategies for compound operations (COMET Fig. 1d).
    
    - SEQUENTIAL: Operations execute one after another
    - PIPELINED: Operations overlap in pipeline stages
    - PARALLEL: Operations execute concurrently on different units
    """
    SEQUENTIAL = "sequential"
    PIPELINED = "pipelined"
    PARALLEL = "parallel"


class DataStagingState(Enum):
    """
    Data staging states for COMET cost model (Section IV-B).
    
    Models the ramp-up/steady/ramp-down phases of data movement.
    """
    IDLE = "idle"
    RAMP_UP = "ramp_up"      # Filling buffer (compute waiting)
    STEADY = "steady"        # Compute and memory overlap
    RAMP_DOWN = "ramp_down"  # Draining buffer (memory waiting)


class MemoryLevel(Enum):
    """
    Memory hierarchy levels (COMET Fig. 2b).
    
    DRAM -> Global Buffer (GB) -> Input/Weight/Output Buffer -> Compute
    """
    DRAM = "dram"
    GLOBAL_BUFFER = "global_buffer"
    INPUT_BUFFER = "input_buffer"
    WEIGHT_BUFFER = "weight_buffer"
    OUTPUT_BUFFER = "output_buffer"
    REGISTER = "register"
    SHARED_MEMORY = "shared_memory"


@dataclass
class COMETHardwareConfig:
    """
    Hardware configuration for COMET cost model.
    
    Captures multi-level memory hierarchy and compute capabilities.
    """
    # Memory bandwidths (GB/s) at each level
    dram_bandwidth_gbps: float = 900.0      # HBM2e: ~900 GB/s
    global_buffer_bandwidth_gbps: float = 3000.0  # On-chip: ~3 TB/s
    local_buffer_bandwidth_gbps: float = 10000.0  # Register-like access
    
    # Memory sizes (bytes)
    dram_size_bytes: int = 80 * 1024**3     # 80 GB HBM
    global_buffer_size_bytes: int = 32 * 1024**2  # 32 MB L2
    local_buffer_size_bytes: int = 256 * 1024     # 256 KB per SM
    
    # Memory access energy (pJ/bit)
    dram_energy_pj_per_bit: float = 10.0
    global_buffer_energy_pj_per_bit: float = 1.0
    local_buffer_energy_pj_per_bit: float = 0.1
    
    # Network parameters
    noc_bandwidth_gbps: float = 1000.0  # On-chip NoC
    noc_latency_ns: float = 10.0        # Per-hop latency
    noc_hops: int = 4                   # Average hops
    
    # Compute parameters
    num_compute_units: int = 108        # SMs on A100
    peak_tflops_fp16: float = 312.0     # FP16 tensor core
    peak_tflops_fp32: float = 156.0     # FP32


@dataclass
class COMETLatencyBreakdown:
    """
    Detailed latency breakdown following COMET equations.
    
    Total latency = compute + memory + collective + scheduling_overhead
    
    Memory latency includes:
    - Ramp-up: Initial buffer fill time
    - Steady-state: Overlapped compute/memory
    - Ramp-down: Final buffer drain time
    
    Scheduling overhead includes:
    - CS (Compulsory Stall): Data dependency waits
    - OS (Optional Stall): Resource blocking
    - CF (Conflict): Resource contention
    """
    # Compute latency
    compute_latency_ms: float = 0.0
    
    # Memory latency components (COMET Eq. 1-2)
    mem_ramp_up_ms: float = 0.0
    mem_steady_state_ms: float = 0.0
    mem_ramp_down_ms: float = 0.0
    
    # Collective latency (COMET Eq. 3-4)
    collective_latency_ms: float = 0.0
    
    # NoC latency
    noc_latency_ms: float = 0.0
    
    # Scheduling overhead (COMET Eq. 5-7)
    compulsory_stall_ms: float = 0.0  # CS: Must wait for data
    optional_stall_ms: float = 0.0     # OS: Resource blocked
    conflict_stall_ms: float = 0.0     # CF: Resource contention
    
    @property
    def total_memory_latency_ms(self) -> float:
        """Total memory latency (COMET Eq. 2)."""
        return self.mem_ramp_up_ms + self.mem_steady_state_ms + self.mem_ramp_down_ms
    
    @property
    def total_scheduling_overhead_ms(self) -> float:
        """Total scheduling overhead."""
        return self.compulsory_stall_ms + self.optional_stall_ms + self.conflict_stall_ms
    
    @property
    def total_latency_ms(self) -> float:
        """Total operation latency."""
        # Compute is typically overlapped with memory in steady state
        effective_compute = max(0, self.compute_latency_ms - self.mem_steady_state_ms)
        return (
            self.mem_ramp_up_ms
            + max(self.compute_latency_ms, self.mem_steady_state_ms)
            + self.mem_ramp_down_ms
            + self.collective_latency_ms
            + self.noc_latency_ms
            + self.total_scheduling_overhead_ms
        )
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "compute_latency_ms": self.compute_latency_ms,
            "mem_ramp_up_ms": self.mem_ramp_up_ms,
            "mem_steady_state_ms": self.mem_steady_state_ms,
            "mem_ramp_down_ms": self.mem_ramp_down_ms,
            "total_memory_latency_ms": self.total_memory_latency_ms,
            "collective_latency_ms": self.collective_latency_ms,
            "noc_latency_ms": self.noc_latency_ms,
            "compulsory_stall_ms": self.compulsory_stall_ms,
            "optional_stall_ms": self.optional_stall_ms,
            "conflict_stall_ms": self.conflict_stall_ms,
            "total_scheduling_overhead_ms": self.total_scheduling_overhead_ms,
            "total_latency_ms": self.total_latency_ms,
        }


@dataclass
class COMETEnergyBreakdown:
    """
    Energy breakdown for COMET cost model.
    
    Total energy = compute + memory_access + noc + collective
    """
    compute_energy_mj: float = 0.0
    
    # Memory access energy at each level
    dram_energy_mj: float = 0.0
    global_buffer_energy_mj: float = 0.0
    local_buffer_energy_mj: float = 0.0
    
    # Network energy
    noc_energy_mj: float = 0.0
    collective_energy_mj: float = 0.0
    
    @property
    def total_memory_energy_mj(self) -> float:
        """Total memory access energy."""
        return self.dram_energy_mj + self.global_buffer_energy_mj + self.local_buffer_energy_mj
    
    @property
    def total_energy_mj(self) -> float:
        """Total energy."""
        return (
            self.compute_energy_mj
            + self.total_memory_energy_mj
            + self.noc_energy_mj
            + self.collective_energy_mj
        )
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "compute_energy_mj": self.compute_energy_mj,
            "dram_energy_mj": self.dram_energy_mj,
            "global_buffer_energy_mj": self.global_buffer_energy_mj,
            "local_buffer_energy_mj": self.local_buffer_energy_mj,
            "total_memory_energy_mj": self.total_memory_energy_mj,
            "noc_energy_mj": self.noc_energy_mj,
            "collective_energy_mj": self.collective_energy_mj,
            "total_energy_mj": self.total_energy_mj,
        }


@dataclass
class COMETCostModel:
    """
    COMET cost model for compound operations with explicit collectives.
    
    Implements the cost equations from the COMET paper:
    - Eq. 1: Memory transaction latency: MemLat(T_i^j) = DV / BW
    - Eq. 2: Total memory latency with data staging
    - Eq. 3-4: Collective operation latency
    - Eq. 5-7: Scheduling-aware latency (CS, OS, CF)
    
    Reference: Negi et al., "COMET: A Framework for Modeling Compound
    Operation Dataflows with Explicit Collectives"
    """
    
    hw_config: COMETHardwareConfig = field(default_factory=COMETHardwareConfig)
    
    def memory_transaction_latency_ms(
        self,
        data_volume_bytes: int,
        src_level: MemoryLevel,
        dst_level: MemoryLevel,
    ) -> float:
        """
        Calculate memory transaction latency (COMET Eq. 1).
        
        MemLat(T_i^j) = DV(T_i^j) / BW(src, dst)
        
        Args:
            data_volume_bytes: Data volume to transfer
            src_level: Source memory level
            dst_level: Destination memory level
        
        Returns:
            Transfer latency in milliseconds
        """
        # Get bandwidth for the transfer
        bandwidth_gbps = self._get_bandwidth(src_level, dst_level)
        
        # MemLat = DV / BW
        data_volume_gb = data_volume_bytes / (1024**3)
        latency_ms = (data_volume_gb / bandwidth_gbps) * 1000
        
        return latency_ms
    
    def _get_bandwidth(
        self,
        src_level: MemoryLevel,
        dst_level: MemoryLevel,
    ) -> float:
        """Get bandwidth between two memory levels."""
        # DRAM transfers
        if src_level == MemoryLevel.DRAM or dst_level == MemoryLevel.DRAM:
            return self.hw_config.dram_bandwidth_gbps
        
        # Global buffer transfers
        if src_level == MemoryLevel.GLOBAL_BUFFER or dst_level == MemoryLevel.GLOBAL_BUFFER:
            return self.hw_config.global_buffer_bandwidth_gbps
        
        # Local buffer transfers
        return self.hw_config.local_buffer_bandwidth_gbps
    
    def total_memory_latency_ms(
        self,
        data_volume_bytes: int,
        src_level: MemoryLevel,
        dst_level: MemoryLevel,
        tile_count: int,
        compute_time_per_tile_ms: float,
    ) -> Tuple[float, float, float]:
        """
        Calculate total memory latency with data staging (COMET Eq. 2).
        
        Models ramp-up, steady-state, and ramp-down phases.
        
        Args:
            data_volume_bytes: Total data to transfer
            src_level: Source memory level
            dst_level: Destination memory level
            tile_count: Number of tiles in the computation
            compute_time_per_tile_ms: Compute time for each tile
        
        Returns:
            Tuple of (ramp_up_ms, steady_state_ms, ramp_down_ms)
        """
        if tile_count <= 0:
            return (0.0, 0.0, 0.0)
        
        # Per-tile memory latency
        tile_data_bytes = data_volume_bytes / tile_count
        mem_time_per_tile = self.memory_transaction_latency_ms(
            int(tile_data_bytes), src_level, dst_level
        )
        
        # Ramp-up: First tile must fully load before compute starts
        ramp_up_ms = mem_time_per_tile
        
        # Steady-state: Memory and compute overlap for middle tiles
        # Number of middle tiles
        middle_tiles = max(0, tile_count - 2)
        steady_state_ms = middle_tiles * max(mem_time_per_tile, compute_time_per_tile_ms)
        
        # Ramp-down: Last tile compute after memory completes
        ramp_down_ms = compute_time_per_tile_ms if tile_count > 1 else 0.0
        
        return (ramp_up_ms, steady_state_ms, ramp_down_ms)
    
    def collective_latency_ms(
        self,
        data_volume_bytes: int,
        collective_type: CommunicationType,
        num_participants: int,
        bandwidth_gbps: Optional[float] = None,
        latency_us: float = 1.0,
    ) -> float:
        """
        Calculate collective operation latency (COMET Eq. 3-4).
        
        For Ring algorithm:
        - AllReduce: 2(n-1)/n * size / bandwidth
        - AllGather: (n-1)/n * size / bandwidth * n = (n-1) * size / bandwidth
        - ReduceScatter: Same as AllGather
        - Broadcast: size / bandwidth + (n-1) * latency
        
        Args:
            data_volume_bytes: Data volume for the collective
            collective_type: Type of collective operation
            num_participants: Number of participating devices
            bandwidth_gbps: Inter-device bandwidth (uses NoC if None)
            latency_us: Per-message latency
        
        Returns:
            Collective latency in milliseconds
        """
        if num_participants <= 1:
            return 0.0
        
        bw = bandwidth_gbps or self.hw_config.noc_bandwidth_gbps
        n = num_participants
        size_gb = data_volume_bytes / (1024**3)
        
        if collective_type == CommunicationType.ALL_REDUCE:
            # Ring AllReduce: 2(n-1)/n * size / bandwidth
            factor = 2 * (n - 1) / n
            transfer_ms = (factor * size_gb / bw) * 1000
            latency_ms = 2 * (n - 1) * latency_us / 1000
            return transfer_ms + latency_ms
        
        elif collective_type == CommunicationType.ALL_GATHER:
            # Ring AllGather: (n-1) * size / bandwidth
            factor = n - 1
            transfer_ms = (factor * size_gb / bw) * 1000
            latency_ms = (n - 1) * latency_us / 1000
            return transfer_ms + latency_ms
        
        elif collective_type == CommunicationType.REDUCE_SCATTER:
            # Same as AllGather
            factor = n - 1
            transfer_ms = (factor * size_gb / bw) * 1000
            latency_ms = (n - 1) * latency_us / 1000
            return transfer_ms + latency_ms
        
        elif collective_type == CommunicationType.BROADCAST:
            # Broadcast: size / bandwidth + (n-1) * latency
            transfer_ms = (size_gb / bw) * 1000
            latency_ms = (n - 1) * latency_us / 1000
            return transfer_ms + latency_ms
        
        else:  # P2P
            transfer_ms = (size_gb / bw) * 1000
            return transfer_ms + latency_us / 1000
    
    def noc_latency_ms(
        self,
        data_volume_bytes: int,
        num_hops: Optional[int] = None,
    ) -> float:
        """
        Calculate Network-on-Chip latency.
        
        Args:
            data_volume_bytes: Data volume to transfer
            num_hops: Number of hops (uses default if None)
        
        Returns:
            NoC latency in milliseconds
        """
        hops = num_hops or self.hw_config.noc_hops
        
        # Transfer time
        size_gb = data_volume_bytes / (1024**3)
        transfer_ms = (size_gb / self.hw_config.noc_bandwidth_gbps) * 1000
        
        # Per-hop latency
        hop_latency_ms = hops * self.hw_config.noc_latency_ns / 1e6
        
        return transfer_ms + hop_latency_ms
    
    def scheduling_latency_ms(
        self,
        strategy: SchedulingStrategy,
        op_latencies_ms: List[float],
        dependencies: Optional[List[Tuple[int, int]]] = None,
    ) -> Tuple[float, float, float]:
        """
        Calculate scheduling-related latencies (COMET Eq. 5-7).
        
        Returns (compulsory_stall, optional_stall, conflict_stall).
        
        Args:
            strategy: Scheduling strategy
            op_latencies_ms: Latency of each operation
            dependencies: (src_op, dst_op) dependency edges
        
        Returns:
            Tuple of (CS, OS, CF) in milliseconds
        """
        if not op_latencies_ms:
            return (0.0, 0.0, 0.0)
        
        deps = dependencies or []
        n_ops = len(op_latencies_ms)
        
        if strategy == SchedulingStrategy.SEQUENTIAL:
            # Sequential: No overlap, all ops wait for previous
            # CS = sum of all dependency waits
            compulsory_stall = 0.0
            for src, dst in deps:
                if 0 <= src < n_ops and 0 <= dst < n_ops:
                    compulsory_stall += op_latencies_ms[src]
            return (compulsory_stall, 0.0, 0.0)
        
        elif strategy == SchedulingStrategy.PIPELINED:
            # Pipelined: Ops overlap but have stage dependencies
            # OS = pipeline bubble time
            max_latency = max(op_latencies_ms)
            pipeline_overhead = max_latency * (n_ops - 1) * 0.1  # 10% bubble
            return (0.0, pipeline_overhead, 0.0)
        
        else:  # PARALLEL
            # Parallel: Ops can conflict for resources
            # CF = resource contention time
            # Assume some contention when many ops run together
            if n_ops > 1:
                avg_latency = sum(op_latencies_ms) / n_ops
                conflict_factor = 0.05 * (n_ops - 1)  # 5% per additional op
                conflict_stall = avg_latency * conflict_factor
                return (0.0, 0.0, conflict_stall)
            return (0.0, 0.0, 0.0)
    
    def estimate_compound_operation(
        self,
        op_name: str,
        input_shapes: List[Tuple[int, ...]],
        dtype_bytes: int = 2,  # FP16
        tile_size: int = 128,
        num_devices: int = 1,
        strategy: SchedulingStrategy = SchedulingStrategy.PIPELINED,
    ) -> Tuple[COMETLatencyBreakdown, COMETEnergyBreakdown]:
        """
        Estimate latency and energy for a compound operation.
        
        Supported operations:
        - gemm_softmax: GEMM followed by row-wise softmax
        - gemm_layernorm: GEMM followed by layer normalization
        - self_attention: Q@K^T -> softmax -> @V
        - gated_mlp: gate * up_proj(x) pattern
        
        Args:
            op_name: Name of the compound operation
            input_shapes: Input tensor shapes
            dtype_bytes: Bytes per element
            tile_size: Tile size for blocking
            num_devices: Number of devices for distribution
            strategy: Scheduling strategy
        
        Returns:
            Tuple of (latency_breakdown, energy_breakdown)
        """
        latency = COMETLatencyBreakdown()
        energy = COMETEnergyBreakdown()
        
        # Calculate total data volume and FLOPs based on operation
        if op_name == "gemm_softmax" and len(input_shapes) >= 2:
            # GEMM: C = A @ B, then softmax(C)
            M, K = input_shapes[0] if len(input_shapes[0]) == 2 else (input_shapes[0][-2], input_shapes[0][-1])
            K2, N = input_shapes[1] if len(input_shapes[1]) == 2 else (input_shapes[1][-2], input_shapes[1][-1])
            
            gemm_flops = 2 * M * N * K
            softmax_flops = 5 * M * N  # max, sub, exp, sum, div
            total_flops = gemm_flops + softmax_flops
            
            # Memory: read A, B; write C; read/write for softmax (in-place)
            input_bytes = (M * K + K * N) * dtype_bytes
            output_bytes = M * N * dtype_bytes
            
        elif op_name == "gemm_layernorm" and len(input_shapes) >= 2:
            M, K = input_shapes[0] if len(input_shapes[0]) == 2 else (input_shapes[0][-2], input_shapes[0][-1])
            K2, N = input_shapes[1] if len(input_shapes[1]) == 2 else (input_shapes[1][-2], input_shapes[1][-1])
            
            gemm_flops = 2 * M * N * K
            ln_flops = 7 * M * N  # mean, sub, var, sqrt, div, scale, bias
            total_flops = gemm_flops + ln_flops
            
            input_bytes = (M * K + K * N) * dtype_bytes
            output_bytes = M * N * dtype_bytes
            
        elif op_name == "self_attention" and len(input_shapes) >= 3:
            # Q, K, V shapes
            B, H, S, D = input_shapes[0] if len(input_shapes[0]) == 4 else (1, *input_shapes[0])
            
            # Q @ K^T: B*H*S*S*D FLOPs
            qk_flops = 2 * B * H * S * S * D
            # Softmax: 5 * B*H*S*S
            softmax_flops = 5 * B * H * S * S
            # Attn @ V: B*H*S*D*S FLOPs
            av_flops = 2 * B * H * S * D * S
            total_flops = qk_flops + softmax_flops + av_flops
            
            # Memory for Q, K, V, output
            input_bytes = 3 * B * H * S * D * dtype_bytes
            output_bytes = B * H * S * D * dtype_bytes
            
        else:
            # Generic estimate
            total_elements = sum(np.prod(s) for s in input_shapes)
            total_flops = total_elements * 2  # Assume 2 FLOPs per element
            input_bytes = total_elements * dtype_bytes
            output_bytes = input_bytes // 2
        
        # Calculate tile count
        total_bytes = input_bytes + output_bytes
        tile_bytes = tile_size * tile_size * dtype_bytes
        tile_count = max(1, total_bytes // tile_bytes)
        
        # Compute latency
        compute_time_ms = (total_flops / (self.hw_config.peak_tflops_fp16 * 1e12)) * 1000
        compute_time_per_tile = compute_time_ms / tile_count
        latency.compute_latency_ms = compute_time_ms
        
        # Memory latency with data staging
        ramp_up, steady, ramp_down = self.total_memory_latency_ms(
            total_bytes,
            MemoryLevel.DRAM,
            MemoryLevel.GLOBAL_BUFFER,
            tile_count,
            compute_time_per_tile,
        )
        latency.mem_ramp_up_ms = ramp_up
        latency.mem_steady_state_ms = steady
        latency.mem_ramp_down_ms = ramp_down
        
        # Collective latency (if distributed)
        if num_devices > 1:
            # Assume AllReduce at the end
            latency.collective_latency_ms = self.collective_latency_ms(
                output_bytes,
                CommunicationType.ALL_REDUCE,
                num_devices,
            )
        
        # NoC latency (for on-chip data movement)
        latency.noc_latency_ms = self.noc_latency_ms(output_bytes)
        
        # Scheduling overhead
        # Create synthetic op latencies for the compound operation
        if op_name == "self_attention":
            op_latencies = [compute_time_ms * 0.4, compute_time_ms * 0.1, compute_time_ms * 0.4]
        else:
            op_latencies = [compute_time_ms * 0.7, compute_time_ms * 0.3]
        
        cs, os, cf = self.scheduling_latency_ms(strategy, op_latencies)
        latency.compulsory_stall_ms = cs
        latency.optional_stall_ms = os
        latency.conflict_stall_ms = cf
        
        # Energy estimation
        energy.compute_energy_mj = total_flops * 1e-15  # ~1 pJ per FLOP
        energy.dram_energy_mj = (
            total_bytes * 8 * self.hw_config.dram_energy_pj_per_bit * 1e-9
        )
        energy.global_buffer_energy_mj = (
            total_bytes * 8 * self.hw_config.global_buffer_energy_pj_per_bit * 1e-9
        )
        
        if num_devices > 1:
            energy.collective_energy_mj = output_bytes * 8 * 0.5e-12 * 1e3  # ~0.5 pJ/bit
        
        return latency, energy
    
    def compare_distributed_variants(
        self,
        op_name: str,
        input_shapes: List[Tuple[int, ...]],
        num_devices: int = 4,
    ) -> Dict[str, Dict]:
        """
        Compare different distributed variants of a compound operation.
        
        For GEMM-Softmax, compares:
        - Local SM: Full result computed locally, then distributed
        - distSM: Partial results computed, then AllReduce for softmax
        
        For GEMM-LayerNorm, compares:
        - Local LN: Full result computed locally
        - distLN: Distributed LayerNorm with AllReduce for statistics
        
        Args:
            op_name: Name of the compound operation
            input_shapes: Input tensor shapes
            num_devices: Number of devices
        
        Returns:
            Dictionary with results for each variant
        """
        results = {}
        
        # Local variant (no distribution)
        local_latency, local_energy = self.estimate_compound_operation(
            op_name, input_shapes, num_devices=1,
            strategy=SchedulingStrategy.PIPELINED,
        )
        results["local"] = {
            "latency_ms": local_latency.total_latency_ms,
            "energy_mj": local_energy.total_energy_mj,
            "breakdown": local_latency.to_dict(),
        }
        
        # Distributed variant
        dist_latency, dist_energy = self.estimate_compound_operation(
            op_name, input_shapes, num_devices=num_devices,
            strategy=SchedulingStrategy.PIPELINED,
        )
        results["distributed"] = {
            "latency_ms": dist_latency.total_latency_ms,
            "energy_mj": dist_energy.total_energy_mj,
            "breakdown": dist_latency.to_dict(),
        }
        
        # Calculate speedup
        if local_latency.total_latency_ms > 0:
            results["speedup"] = local_latency.total_latency_ms / dist_latency.total_latency_ms
        else:
            results["speedup"] = 1.0
        
        return results
