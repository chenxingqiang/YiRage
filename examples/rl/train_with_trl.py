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
Example: Training µGraph Search Policy with TRL

This example demonstrates how to train a language model to generate
optimal µGraph configurations using various fine-tuning strategies.

Supported strategies:
- SFT (Supervised Fine-Tuning): Learn from expert demonstrations
- DPO (Direct Preference Optimization): Learn from preference pairs
- GRPO (Group Relative Policy Optimization): Learn from grouped samples

Usage:
    # SFT training with LoRA
    python train_with_trl.py --strategy sft --use-lora --use-4bit
    
    # DPO training
    python train_with_trl.py --strategy dpo --model meta-llama/Llama-2-7b-hf
    
    # Inference
    python train_with_trl.py --mode inference --checkpoint outputs/mugraph_policy
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Add yirage to path
WORKSPACE_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT / "python"))


def create_sample_training_data() -> List[Dict[str, Any]]:
    """Create sample training data for demonstration."""

    # Sample target graphs
    target_graphs = [
        {
            "name": "matmul_64x128",
            "operators": [{"type": "matmul", "inputs": [0, 1], "outputs": [2]}],
            "tensors": [
                {"id": 0, "dims": [64, 128], "dtype": "float16", "is_input": True},
                {"id": 1, "dims": [128, 64], "dtype": "float16", "is_input": True},
                {"id": 2, "dims": [64, 64], "dtype": "float16", "is_output": True},
            ],
        },
        {
            "name": "mlp_block",
            "operators": [
                {"type": "matmul", "inputs": [0, 1], "outputs": [2]},
                {"type": "silu", "inputs": [2], "outputs": [3]},
                {"type": "matmul", "inputs": [3, 4], "outputs": [5]},
            ],
            "tensors": [
                {"id": 0, "dims": [8, 4096], "dtype": "float16", "is_input": True},
                {"id": 1, "dims": [4096, 11008], "dtype": "float16", "is_input": True},
                {"id": 2, "dims": [8, 11008], "dtype": "float16"},
                {"id": 3, "dims": [8, 11008], "dtype": "float16"},
                {"id": 4, "dims": [11008, 4096], "dtype": "float16", "is_input": True},
                {"id": 5, "dims": [8, 4096], "dtype": "float16", "is_output": True},
            ],
        },
        {
            "name": "attention_head",
            "operators": [
                {"type": "matmul", "inputs": [0, 1], "outputs": [2]},  # Q
                {"type": "matmul", "inputs": [0, 3], "outputs": [4]},  # K
                {"type": "matmul", "inputs": [0, 5], "outputs": [6]},  # V
                {"type": "matmul", "inputs": [2, 4], "outputs": [7]},  # QK^T
                {"type": "softmax", "inputs": [7], "outputs": [8]},
                {"type": "matmul", "inputs": [8, 6], "outputs": [9]},  # Attention
            ],
            "tensors": [
                {"id": 0, "dims": [8, 128, 4096], "dtype": "float16", "is_input": True},
                {"id": 1, "dims": [4096, 4096], "dtype": "float16", "is_input": True},
                {"id": 9, "dims": [8, 128, 4096], "dtype": "float16", "is_output": True},
            ],
        },
    ]

    # Sample hardware profiles
    hardware_profiles = [
        {
            "backend": "cuda",
            "device_name": "NVIDIA A100",
            "compute_capability": [8, 0],
            "total_cores": 6912,
            "tensor_core_count": 432,
            "global_memory_gb": 40.0,
            "peak_tflops_fp16": 312.0,
            "memory_bandwidth_gbps": 2039.0,
        },
        {
            "backend": "cuda",
            "device_name": "NVIDIA V100",
            "compute_capability": [7, 0],
            "total_cores": 5120,
            "tensor_core_count": 640,
            "global_memory_gb": 16.0,
            "peak_tflops_fp16": 125.0,
            "memory_bandwidth_gbps": 900.0,
        },
        {
            "backend": "maca",
            "device_name": "MetaX C500",
            "compute_capability": [8, 0],
            "total_cores": 8192,
            "tensor_core_count": 512,
            "global_memory_gb": 32.0,
            "peak_tflops_fp16": 100.0,
            "memory_bandwidth_gbps": 1200.0,
        },
    ]

    # Generate training examples
    examples = []

    for graph in target_graphs:
        for hw in hardware_profiles:
            # Generate optimal config (simplified heuristic)
            optimal_config = _generate_optimal_config(graph, hw)
            optimal_graph = _generate_optimal_graph(graph, optimal_config)

            examples.append(
                {
                    "target_graph": graph,
                    "hardware": hw,
                    "optimal_config": optimal_config,
                    "optimal_graph": optimal_graph,
                }
            )

    return examples


def create_sample_preference_data() -> List[Dict[str, Any]]:
    """Create sample preference data for DPO training."""

    examples = []
    base_examples = create_sample_training_data()

    for ex in base_examples:
        # Create a suboptimal config
        suboptimal_config = {
            "grid_dim": {"x": 1, "y": 1, "z": 1},
            "block_dim": {"x": 64, "y": 1, "z": 1},
            "forloop_range": 1,
            "shared_memory_size": 16384,
        }

        optimal_config = ex["optimal_config"]

        # Simulate latencies
        optimal_latency = 0.1 + 0.05 * len(ex["target_graph"]["operators"])
        suboptimal_latency = optimal_latency * 2.5

        examples.append(
            {
                "target_graph": ex["target_graph"],
                "hardware": ex["hardware"],
                "chosen_config": optimal_config,
                "rejected_config": suboptimal_config,
                "chosen_latency": optimal_latency,
                "rejected_latency": suboptimal_latency,
            }
        )

    return examples


def _generate_optimal_config(graph: Dict, hw: Dict) -> Dict[str, Any]:
    """Generate optimal config based on graph and hardware."""
    num_ops = len(graph.get("operators", []))

    # Heuristic: larger graphs need more parallelism
    if num_ops >= 4:
        grid_x, block_x = 16, 256
    elif num_ops >= 2:
        grid_x, block_x = 8, 128
    else:
        grid_x, block_x = 4, 128

    # Adjust for hardware
    if hw.get("backend") == "maca":
        # MACA uses 64-thread warps
        block_x = min(block_x, 512)

    return {
        "grid_dim": {"x": grid_x, "y": 1, "z": 1},
        "block_dim": {"x": block_x, "y": 1, "z": 1},
        "forloop_range": 8 if num_ops > 2 else 4,
        "shared_memory_size": 49152,
    }


def _generate_optimal_graph(graph: Dict, config: Dict) -> Dict[str, Any]:
    """Generate optimized µGraph."""
    # Copy and add optimization hints
    optimized = {**graph}
    optimized["optimization_hints"] = {
        "use_tensor_cores": True,
        "fused_ops": True,
        "tile_size": config["block_dim"]["x"],
    }
    return optimized


def train_sft(args):
    """Train with SFT strategy."""
    print("=" * 60)
    print("Training with SFT (Supervised Fine-Tuning)")
    print("=" * 60)

    try:
        from yirage.rl.training.trl_integration import (
            FineTuningConfig,
            MuGraphPolicyTrainer,
        )
    except ImportError as e:
        print(f"TRL not available: {e}")
        print("Install with: pip install trl transformers peft bitsandbytes")
        return

    # Create training data
    train_data = create_sample_training_data()
    print(f"Created {len(train_data)} training examples")

    # Configure training
    config = FineTuningConfig(
        strategy="sft",
        model_name_or_path=args.model,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        use_4bit=args.use_4bit,
        use_8bit=args.use_8bit,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
    )

    # Create trainer
    trainer = MuGraphPolicyTrainer(config)

    print(f"\nModel: {args.model}")
    print(f"LoRA: {args.use_lora} (r={args.lora_r})")
    print(f"Quantization: {'4-bit' if args.use_4bit else '8-bit' if args.use_8bit else 'None'}")
    print()

    # Train
    trainer.train(train_data)

    # Save
    trainer.save(args.output_dir)
    print(f"\nModel saved to: {args.output_dir}")


def train_dpo(args):
    """Train with DPO strategy."""
    print("=" * 60)
    print("Training with DPO (Direct Preference Optimization)")
    print("=" * 60)

    try:
        from yirage.rl.training.trl_integration import (
            FineTuningConfig,
            MuGraphPolicyTrainer,
        )
    except ImportError as e:
        print(f"TRL not available: {e}")
        return

    # Create preference data
    train_data = create_sample_preference_data()
    print(f"Created {len(train_data)} preference pairs")

    # Configure training
    config = FineTuningConfig(
        strategy="dpo",
        model_name_or_path=args.model,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        use_4bit=args.use_4bit,
        beta=args.dpo_beta,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        output_dir=args.output_dir,
    )

    trainer = MuGraphPolicyTrainer(config)
    trainer.train(train_data)
    trainer.save(args.output_dir)

    print(f"\nModel saved to: {args.output_dir}")


def train_grpo(args):
    """Train with GRPO strategy."""
    print("=" * 60)
    print("Training with GRPO (Group Relative Policy Optimization)")
    print("=" * 60)

    try:
        from yirage.rl.training.grpo import GRPOConfig, GRPOTrainer
    except ImportError as e:
        print(f"GRPO not available: {e}")
        return

    # Note: GRPO requires online training with environment
    print("GRPO requires online training with environment interaction.")
    print("See scripts/train_hierarchical_search.py for example.")


def inference(args):
    """Run inference with trained model."""
    print("=" * 60)
    print("Running Inference")
    print("=" * 60)

    try:
        from yirage.rl.training.trl_integration import MuGraphPolicyTrainer
    except ImportError as e:
        print(f"TRL not available: {e}")
        return

    # Load model
    trainer = MuGraphPolicyTrainer.__new__(MuGraphPolicyTrainer)
    trainer.load(args.checkpoint)

    # Sample input
    target_graph = {
        "operators": [{"type": "matmul", "inputs": [0, 1], "outputs": [2]}],
        "tensors": [
            {"id": 0, "dims": [64, 256], "dtype": "float16"},
            {"id": 1, "dims": [256, 128], "dtype": "float16"},
        ],
    }

    hardware = {
        "backend": "cuda",
        "device_name": "NVIDIA A100",
        "compute_capability": [8, 0],
        "total_cores": 6912,
        "peak_tflops_fp16": 312.0,
    }

    # Generate configurations
    print("\nGenerating configurations...")
    configs = trainer.generate_config(target_graph, hardware, num_samples=3)

    print(f"\nGenerated {len(configs)} configurations:")
    for i, config in enumerate(configs):
        print(f"\n[{i + 1}] {json.dumps(config, indent=2)}")


def demo_hardware_detection(args):
    """Demonstrate hardware detection."""
    print("=" * 60)
    print("Hardware Detection Demo")
    print("=" * 60)

    import types
    import importlib.util

    # Setup fake package structure for isolated loading
    def setup_and_load_module(name: str, path: Path, deps: dict = None):
        """Load module with dependencies."""
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)

        # Inject dependencies
        if deps:
            for dep_name, dep_mod in deps.items():
                # Create a fake relative import context
                parts = name.rsplit(".", 1)
                if len(parts) > 1:
                    parent = parts[0]
                    sys.modules[f"{parent}.{dep_name.split('.')[-1]}"] = dep_mod

        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    # Create fake yirage.rl.hardware package
    hw_path = WORKSPACE_ROOT / "python" / "yirage" / "rl" / "hardware"

    fake_yirage = types.ModuleType("yirage")
    fake_yirage.__path__ = [str(WORKSPACE_ROOT / "python" / "yirage")]
    sys.modules["yirage"] = fake_yirage

    fake_rl = types.ModuleType("yirage.rl")
    fake_rl.__path__ = [str(WORKSPACE_ROOT / "python" / "yirage" / "rl")]
    sys.modules["yirage.rl"] = fake_rl

    fake_hw = types.ModuleType("yirage.rl.hardware")
    fake_hw.__path__ = [str(hw_path)]
    sys.modules["yirage.rl.hardware"] = fake_hw

    # Load profile first (no dependencies)
    profile_mod = setup_and_load_module("yirage.rl.hardware.profile", hw_path / "profile.py")

    # Load detector (depends on profile)
    detector_mod = setup_and_load_module(
        "yirage.rl.hardware.detector", hw_path / "detector.py", {"profile": profile_mod}
    )

    # Detect hardware
    print("\nDetecting available hardware...")

    detectors = [
        ("CUDA", detector_mod.CUDADetector),
        ("CPU", detector_mod.CPUDetector),
        ("MACA", detector_mod.MACACDetector),
        ("Ascend", detector_mod.AscendDetector),
        ("MPS", detector_mod.MPSDetector),
    ]

    for backend, detector_class in detectors:
        try:
            detector = detector_class()
            if detector.is_available():
                profile = detector.detect()
                if profile:
                    print(f"\n✓ {backend}:")
                    print(f"  Device: {profile.device_name}")
                    print(f"  Cores: {profile.total_cores}")
                    print(f"  Memory: {profile.global_memory_gb:.1f} GB")
                    print(f"  Peak FP16: {profile.peak_tflops_fp16:.1f} TFLOPS")

                    # Show feature vector
                    features = profile.to_feature_vector()
                    print(f"  Feature dim: {features.shape}")
            else:
                print(f"\n✗ {backend}: Not available")
        except Exception as e:
            print(f"\n✗ {backend}: Error - {e}")


def main():
    parser = argparse.ArgumentParser(description="µGraph Policy Training with TRL")

    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "inference", "hardware"],
        help="Mode: train, inference, or hardware detection",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="sft",
        choices=["sft", "dpo", "grpo"],
        help="Training strategy",
    )

    # Model args
    parser.add_argument(
        "--model", type=str, default="meta-llama/Llama-2-7b-hf", help="Base model name or path"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/mugraph_policy",
        help="Checkpoint path for inference",
    )

    # LoRA args
    parser.add_argument(
        "--use-lora", action="store_true", default=True, help="Use LoRA for efficient fine-tuning"
    )
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")

    # Quantization
    parser.add_argument("--use-4bit", action="store_true", help="Use 4-bit quantization")
    parser.add_argument("--use-8bit", action="store_true", help="Use 8-bit quantization")

    # Training args
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument(
        "--output-dir", type=str, default="outputs/mugraph_policy", help="Output directory"
    )

    # DPO args
    parser.add_argument("--dpo-beta", type=float, default=0.1, help="DPO beta parameter")

    args = parser.parse_args()

    if args.mode == "hardware":
        demo_hardware_detection(args)
    elif args.mode == "inference":
        inference(args)
    elif args.strategy == "sft":
        train_sft(args)
    elif args.strategy == "dpo":
        train_dpo(args)
    elif args.strategy == "grpo":
        train_grpo(args)
    else:
        parser.print_help()

    return 0


if __name__ == "__main__":
    sys.exit(main())
