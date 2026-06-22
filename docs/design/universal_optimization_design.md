# Universal Compute Optimization System Design

## Vision

Build a system that can automatically optimize **any compute task** on **any hardware** at **any cluster scale**.

## Implementation Status: ✅ Complete

The system is fully implemented and tested. Key modules:

- `python/yirage/rl/cluster/topology.py` - Cluster topology definition
- `python/yirage/rl/cluster/task.py` - Universal task representation
- `python/yirage/rl/cluster/simulator.py` - Communication simulation
- `python/yirage/rl/cluster/auto_optimizer.py` - Automatic optimization
- `python/yirage/rl/cluster/e2e_optimizer.py` - End-to-end pipeline

### Quick Start

```python
from yirage.rl.cluster import optimize_any_task

# Optimize any task with one function call
result = optimize_any_task(
    {"type": "attention", "batch": 32, "seq_len": 2048, "num_heads": 32},
    cluster_spec={"type": "multi_node", "num_nodes": 4, "gpus_per_node": 8}
)

print(f"Strategy: {result.result.parallelism_strategy}")
print(f"Latency: {result.result.estimated_latency_ms:.2f} ms")
print(f"Throughput: {result.result.estimated_throughput_tps:.1f} samples/sec")
```

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Universal Compute Optimization System                     │
│                                                                              │
│  ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐        │
│  │   任意计算任务    │──▶│   任意硬件集群    │──▶│   最优计算图实现  │        │
│  │   (Any Task)     │   │   (Any Cluster)   │   │ (Optimal µGraph) │        │
│  └──────────────────┘   └──────────────────┘   └──────────────────┘        │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                      Automatic Optimization Pipeline                     ││
│  │                                                                          ││
│  │  [Task Input] ─▶ [Analyze] ─▶ [Decompose] ─▶ [Place] ─▶ [Optimize]     ││
│  │       │              │            │            │            │            ││
│  │       ▼              ▼            ▼            ▼            ▼            ││
│  │   Workload       Hardware     Sub-Tasks    Placement    Optimized       ││
│  │   Analysis       Profiling    DAG          Strategy     µGraphs         ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
```

## Architecture

### 1. Universal Task Representation

```python
@dataclass
class ComputeTask:
    """Hardware-agnostic compute task definition."""
    
    # Input specification
    inputs: List[TensorSpec]          # Input tensor shapes and dtypes
    outputs: List[TensorSpec]         # Expected output shapes
    
    # Compute specification (one of):
    operation: str                    # High-level op: "attention", "mlp", "conv"
    custom_graph: Optional[Graph]     # Or custom computation graph
    pytorch_module: Optional[Module]  # Or PyTorch module to optimize
    
    # Constraints
    latency_budget_ms: Optional[float]
    memory_budget_gb: Optional[float]
    throughput_target: Optional[float]
    
    # Hints
    batch_sizes: List[int]            # Expected batch size range
    is_training: bool = False
    precision: str = "auto"           # auto, fp16, bf16, fp32, int8
```

### 2. Cluster Abstraction

```python
@dataclass
class ClusterTopology:
    """Heterogeneous cluster representation."""
    
    # Nodes
    nodes: List[ComputeNode]
    
    # Network topology
    network: NetworkTopology
    
    # Resource management
    scheduler: ResourceScheduler
    
class ComputeNode:
    """Single compute node with multiple devices."""
    
    node_id: str
    devices: List[HardwareProfile]    # GPUs, NPUs, CPUs on this node
    local_memory_gb: float
    numa_topology: Optional[NumaInfo]
    
class NetworkTopology:
    """Network connectivity between nodes."""
    
    bandwidth_matrix: np.ndarray      # NxN bandwidth in GB/s
    latency_matrix: np.ndarray        # NxN latency in ms
    topology_type: str                # "ring", "tree", "full_mesh", "custom"
```

### 3. Automatic Optimization Pipeline

```
Input Task ─────────────────────────────────────────────────────────▶ Optimized Execution
     │                                                                       ▲
     ▼                                                                       │
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │
│   Analyze   │───▶│  Decompose  │───▶│    Place    │───▶│  Optimize   │───┘
│   Workload  │    │   to DAG    │    │   on HW     │    │   Kernels   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
     │                   │                  │                  │
     ▼                   ▼                  ▼                  ▼
 - Compute       - Operator         - Device          - Per-device
   intensity       fusion             assignment         µGraph search
 - Memory        - Parallelism      - Data             - Cross-device
   pattern         strategy           partitioning       coordination
 - Data          - Pipeline         - Communication    - Collective
   dependencies    stages             optimization       operations
```

## Implementation

### Phase 1: Task Analysis & Decomposition

1. **Workload Analysis**
   - Detect compute patterns (attention, MLP, conv, etc.)
   - Estimate FLOPs, memory, arithmetic intensity
   - Identify parallelization opportunities

2. **Automatic Decomposition**
   - Split large tasks into sub-graphs
   - Apply fusion rules for efficiency
   - Generate execution DAG

### Phase 2: Cluster-Level Placement

1. **Device Assignment**
   - Match sub-tasks to best-fit devices
   - Consider memory constraints
   - Minimize communication overhead

2. **Data Partitioning**
   - Tensor parallelism across devices
   - Pipeline parallelism for sequences
   - Expert parallelism for MoE

### Phase 3: Per-Device Optimization

1. **Automatic Kernel Search**
   - Use learned policy for search
   - Hardware-specific optimization
   - Performance verification

2. **Cross-Device Coordination**
   - Overlap compute and communication
   - Collective operation optimization
   - Synchronization minimization

## Key Innovations

### 1. Zero-Shot Hardware Adaptation

The system learns hardware-agnostic representations that transfer to new hardware:

```python
class UniversalOptimizer:
    def optimize(self, task: ComputeTask, cluster: Cluster) -> ExecutionPlan:
        # 1. Analyze task (hardware-agnostic)
        workload = self.analyze_workload(task)
        
        # 2. Profile cluster capabilities
        capabilities = self.profile_cluster(cluster)
        
        # 3. Generate placement strategy
        placement = self.learned_placer.place(workload, capabilities)
        
        # 4. Optimize each sub-task for assigned device
        optimized_kernels = {}
        for subtask, device in placement.items():
            optimized_kernels[subtask] = self.kernel_optimizer.optimize(
                subtask, device, transfer_from_similar=True
            )
        
        return ExecutionPlan(placement, optimized_kernels)
```

### 2. Learned Search with Transfer

The kernel optimizer learns patterns that transfer across:
- Similar operators on different hardware
- Similar hardware for different operators
- Similar problem sizes

### 3. Online Adaptation

Continuously improve based on execution feedback:

```python
def execute_and_learn(self, plan: ExecutionPlan, task: ComputeTask):
    # Execute
    result, metrics = self.execute(plan)
    
    # Learn from execution
    self.update_cost_model(plan, metrics)
    self.update_policy(plan, metrics)
    
    # Store for future reference
    self.experience_buffer.add(task, plan, metrics)
```
