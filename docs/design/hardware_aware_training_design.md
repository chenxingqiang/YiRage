# Hardware-Aware Training Design for µGraph Search

## Overview

This document describes the design of a hardware-aware training system that:
1. **Detects and characterizes hardware** across heterogeneous environments
2. **Couples hardware features with search configuration**
3. **Supports PyTorch-based model training** with extracted µGraph features
4. **Implements GRPO and fine-tuning strategies** for large model optimization
5. **Provides unified abstraction** for any hardware platform

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Hardware-Aware Training System                            │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                     Hardware Detection Layer                             │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐      │ │
│  │  │  CUDA    │ │  MACA    │ │  Ascend  │ │   CPU    │ │   MPS    │      │ │
│  │  │ Detector │ │ Detector │ │ Detector │ │ Detector │ │ Detector │      │ │
│  │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘      │ │
│  │       └────────────┴────────────┴────────────┴────────────┘             │ │
│  │                              │                                           │ │
│  │                    ┌─────────▼─────────┐                                 │ │
│  │                    │ HardwareRegistry  │                                 │ │
│  │                    │ (Unified Profile) │                                 │ │
│  │                    └─────────┬─────────┘                                 │ │
│  └──────────────────────────────┼───────────────────────────────────────────┘ │
│                                 │                                             │
│  ┌──────────────────────────────▼───────────────────────────────────────────┐ │
│  │                    Config Coupling Layer                                  │ │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐          │ │
│  │  │ HardwareProfile │──│ ConfigGenerator │──│ SearchConstraint│          │ │
│  │  │   (features)    │  │ (auto-config)   │  │   (coupling)    │          │ │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘          │ │
│  └──────────────────────────────┬───────────────────────────────────────────┘ │
│                                 │                                             │
│  ┌──────────────────────────────▼───────────────────────────────────────────┐ │
│  │                    PyTorch Training Layer                                 │ │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐          │ │
│  │  │ FeatureDataset  │  │ SearchPolicy    │  │  TrainingLoop   │          │ │
│  │  │ (µGraph + HW)   │  │ (Transformer)   │  │  (distributed)  │          │ │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘          │ │
│  └──────────────────────────────┬───────────────────────────────────────────┘ │
│                                 │                                             │
│  ┌──────────────────────────────▼───────────────────────────────────────────┐ │
│  │                    RL Strategy Layer                                      │ │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐          │ │
│  │  │      PPO        │  │     GRPO        │  │   Fine-tuning   │          │ │
│  │  │   (baseline)    │  │ (group-relative)│  │   (LoRA/QLoRA)  │          │ │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘          │ │
│  └──────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 1. Hardware Detection Layer

### 1.1 Hardware Profile Structure

```python
@dataclass
class HardwareProfile:
    # Basic info
    backend: str              # cuda, maca, ascend, cpu, mps
    device_name: str          # e.g., "NVIDIA A100"
    device_count: int         # Number of devices
    
    # Compute capabilities
    compute_capability: Tuple[int, int]  # e.g., (8, 0) for SM80
    total_cores: int          # CUDA cores / AI cores
    tensor_core_count: int    # Tensor cores (if available)
    warp_size: int            # 32 for CUDA, 64 for MACA
    
    # Memory
    global_memory_gb: float
    shared_memory_kb: float
    l2_cache_mb: float
    memory_bandwidth_gbps: float
    
    # Execution limits
    max_threads_per_block: int
    max_blocks_per_sm: int
    max_shared_memory_per_block: int
    max_registers_per_thread: int
    
    # Performance characteristics
    peak_tflops_fp16: float
    peak_tflops_fp32: float
    memory_clock_ghz: float
```

### 1.2 Detector Interface

Each hardware backend implements:

```python
class HardwareDetector(ABC):
    @abstractmethod
    def detect(self) -> Optional[HardwareProfile]: ...
    
    @abstractmethod
    def is_available(self) -> bool: ...
    
    @abstractmethod
    def get_optimal_config(self, workload: WorkloadSpec) -> HardwareConfig: ...
```

## 2. Config Coupling Layer

### 2.1 Automatic Config Generation

```python
class ConfigGenerator:
    def generate(
        self,
        hardware: HardwareProfile,
        workload: WorkloadSpec,
        optimization_target: str = "latency"
    ) -> HardwareConfig:
        """
        Generate optimal hardware config based on:
        - Hardware capabilities (memory, compute)
        - Workload characteristics (tensor sizes, ops)
        - Optimization target (latency, throughput, memory)
        """
```

### 2.2 Hardware-Search Coupling

```python
class HardwareSearchCoupling:
    def get_valid_configs(self, hardware: HardwareProfile) -> List[HardwareConfig]:
        """Return all valid configurations for this hardware."""
    
    def get_constraints(
        self, 
        hardware: HardwareProfile,
        config: HardwareConfig
    ) -> SearchSpaceConstraints:
        """Compute search constraints from hardware + config."""
    
    def estimate_performance(
        self,
        hardware: HardwareProfile,
        config: HardwareConfig,
        graph: MuGraphFeature
    ) -> PerformanceEstimate:
        """Estimate kernel performance without execution."""
```

## 3. PyTorch Training Layer

### 3.1 Feature Dataset

```python
class MuGraphDataset(torch.utils.data.Dataset):
    """
    Dataset of (µGraph features, hardware profile, optimal config) tuples.
    
    Supports:
    - On-the-fly feature extraction
    - Hardware profile embedding
    - Multi-hardware training
    """
    
    def __getitem__(self, idx):
        return {
            "graph_features": self.graphs[idx],
            "hardware_features": self.hardware_embeddings[idx],
            "config_target": self.optimal_configs[idx],
            "performance_label": self.performance_labels[idx],
        }
```

### 3.2 Search Policy Model

```python
class SearchPolicyTransformer(nn.Module):
    """
    Transformer-based policy for µGraph search.
    
    Architecture:
    - Graph Encoder: GNN/Transformer for µGraph structure
    - Hardware Encoder: MLP for hardware features
    - Cross-Attention: Graph-Hardware interaction
    - Policy Head: Action distribution
    - Value Head: State value estimate
    """
```

### 3.3 Training Loop

```python
class DistributedTrainer:
    """
    Distributed training with:
    - Data parallelism across GPUs
    - Mixed precision (FP16/BF16)
    - Gradient accumulation
    - Checkpointing
    """
```

## 4. RL Strategy Layer

### 4.1 GRPO (Group Relative Policy Optimization)

GRPO improves upon PPO by:
- Computing advantages relative to group of samples
- Better credit assignment for sparse rewards
- More stable training for large action spaces

```python
class GRPOTrainer:
    def compute_group_advantages(
        self,
        rewards: Tensor,
        group_size: int = 8
    ) -> Tensor:
        """
        Compute advantages relative to group:
        A_i = r_i - mean(r_group)
        """
    
    def update(self, batch):
        # Sample multiple completions per prompt
        # Rank by reward within group
        # Update policy to increase probability of best
```

### 4.2 Fine-tuning Strategies

```python
class FineTuningConfig:
    method: str  # "full", "lora", "qlora"
    lora_rank: int = 8
    lora_alpha: float = 32
    target_modules: List[str] = ["q_proj", "v_proj"]
    quantization_bits: int = 4  # for QLoRA
```

## 5. Implementation Plan

### Phase 1: Hardware Detection (Week 1)
- [ ] Implement CUDA detector
- [ ] Implement CPU detector
- [ ] Implement MACA detector
- [ ] Implement Ascend detector
- [ ] Hardware registry and caching

### Phase 2: Config Coupling (Week 2)
- [ ] ConfigGenerator implementation
- [ ] SearchSpaceConstraints from hardware
- [ ] Performance estimation model

### Phase 3: PyTorch Training (Week 3)
- [ ] MuGraphDataset implementation
- [ ] SearchPolicyTransformer model
- [ ] Distributed training loop

### Phase 4: RL Strategies (Week 4)
- [ ] GRPO implementation
- [ ] LoRA/QLoRA integration
- [ ] Multi-hardware fine-tuning

## 6. API Design

### 6.1 High-Level API

```python
from yirage.rl.hardware import detect_hardware, HardwareRegistry
from yirage.rl.training import SearchPolicyTrainer, GRPOConfig

# Detect available hardware
hardware = detect_hardware()
print(f"Detected: {hardware.backend} - {hardware.device_name}")

# Create trainer with GRPO
trainer = SearchPolicyTrainer(
    model="transformer-base",
    hardware=hardware,
    strategy=GRPOConfig(
        group_size=8,
        learning_rate=1e-4,
    ),
    fine_tuning=FineTuningConfig(
        method="lora",
        lora_rank=16,
    ),
)

# Train on dataset
trainer.train(
    train_dataset=train_data,
    val_dataset=val_data,
    num_epochs=100,
)

# Export for deployment
trainer.export("checkpoints/policy_v1.onnx")
```
