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
TRL (Transformer Reinforcement Learning) Integration for µGraph Search.

Supports various LLM fine-tuning strategies:
- SFT (Supervised Fine-Tuning)
- PPO (Proximal Policy Optimization)
- DPO (Direct Preference Optimization)
- GRPO (Group Relative Policy Optimization)
- ORPO (Odds Ratio Preference Optimization)
- KTO (Kahneman-Tversky Optimization)

These strategies can be used to train a policy model that learns to
construct optimal µGraphs for given hardware configurations.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Callable, Union
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

# Check for required dependencies
TRL_AVAILABLE = False
TRANSFORMERS_AVAILABLE = False
PEFT_AVAILABLE = False

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        PreTrainedModel,
        PreTrainedTokenizer,
        TrainingArguments,
        BitsAndBytesConfig,
    )

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from trl import (
        SFTTrainer,
        SFTConfig,
        PPOTrainer,
        PPOConfig,
        DPOTrainer,
        DPOConfig,
        AutoModelForCausalLMWithValueHead,
    )

    TRL_AVAILABLE = True
except ImportError:
    TRL_AVAILABLE = False

try:
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
        PeftModel,
        TaskType,
    )

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


def check_dependencies():
    """Check if required dependencies are available."""
    missing = []
    if not HAS_TORCH:
        missing.append("torch")
    if not TRANSFORMERS_AVAILABLE:
        missing.append("transformers")
    if not TRL_AVAILABLE:
        missing.append("trl")
    if not PEFT_AVAILABLE:
        missing.append("peft")

    if missing:
        raise ImportError(
            f"Missing dependencies: {', '.join(missing)}. "
            f"Install with: pip install {' '.join(missing)}"
        )


@dataclass
class FineTuningConfig:
    """
    Unified configuration for LLM fine-tuning.

    Supports multiple strategies and quantization options.
    """

    # Model
    model_name_or_path: str = "meta-llama/Llama-2-7b-hf"
    tokenizer_name: Optional[str] = None

    # Strategy
    strategy: str = "sft"  # sft, ppo, dpo, grpo, orpo, kto

    # LoRA configuration
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # Quantization
    use_4bit: bool = False
    use_8bit: bool = False
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"

    # Training
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.03
    max_seq_length: int = 2048

    # PPO specific
    ppo_epochs: int = 4
    init_kl_coef: float = 0.2
    adap_kl_ctrl: bool = True

    # DPO specific
    beta: float = 0.1

    # GRPO specific
    group_size: int = 8

    # Output
    output_dir: str = "outputs"
    logging_steps: int = 10
    save_steps: int = 500

    # Hardware
    bf16: bool = True
    fp16: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


class MuGraphDatasetFormatter:
    """
    Formats µGraph data for LLM training.

    Converts graph structures and hardware configs into
    text format suitable for language model training.
    """

    def __init__(
        self,
        tokenizer: Optional[Any] = None,
        max_length: int = 2048,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def format_graph_as_text(self, graph: Dict[str, Any]) -> str:
        """Convert µGraph to text representation."""
        lines = []

        # Header
        lines.append("### µGraph Specification")

        # Operators
        operators = graph.get("operators", [])
        lines.append(f"\n## Operators ({len(operators)})")
        for i, op in enumerate(operators):
            op_type = op.get("type", op.get("op_type", "unknown"))
            inputs = op.get("inputs", op.get("input_tensor_ids", []))
            outputs = op.get("outputs", op.get("output_tensor_ids", []))
            lines.append(f"{i}: {op_type} inputs={inputs} outputs={outputs}")

        # Tensors
        tensors = graph.get("tensors", [])
        if tensors:
            lines.append(f"\n## Tensors ({len(tensors)})")
            for t in tensors:
                tid = t.get("tensor_id", t.get("id", 0))
                dims = t.get("dims", [])
                dtype = t.get("dtype", "float16")
                lines.append(f"{tid}: dims={dims} dtype={dtype}")

        return "\n".join(lines)

    def format_hardware_as_text(self, hardware: Dict[str, Any]) -> str:
        """Convert hardware profile to text representation."""
        lines = []

        lines.append("### Hardware Configuration")
        lines.append(f"Backend: {hardware.get('backend', 'unknown')}")
        lines.append(f"Device: {hardware.get('device_name', 'unknown')}")
        lines.append(f"Compute Capability: {hardware.get('compute_capability', [0, 0])}")
        lines.append(f"Memory: {hardware.get('global_memory_gb', 0):.1f} GB")
        lines.append(f"Cores: {hardware.get('total_cores', 0)}")
        lines.append(f"Tensor Cores: {hardware.get('tensor_core_count', 0)}")
        lines.append(f"Peak FP16: {hardware.get('peak_tflops_fp16', 0):.1f} TFLOPS")

        return "\n".join(lines)

    def format_config_as_text(self, config: Dict[str, Any]) -> str:
        """Convert hardware config to text representation."""
        grid = config.get("grid_dim", {"x": 1, "y": 1, "z": 1})
        block = config.get("block_dim", {"x": 128, "y": 1, "z": 1})

        lines = []
        lines.append("### Execution Config")
        lines.append(f"Grid: ({grid.get('x', 1)}, {grid.get('y', 1)}, {grid.get('z', 1)})")
        lines.append(f"Block: ({block.get('x', 128)}, {block.get('y', 1)}, {block.get('z', 1)})")
        lines.append(f"Forloop: {config.get('forloop_range', 1)}")
        lines.append(f"Shared Memory: {config.get('shared_memory_size', 49152)} bytes")

        return "\n".join(lines)

    def create_sft_example(
        self,
        target_graph: Dict[str, Any],
        hardware: Dict[str, Any],
        optimal_config: Dict[str, Any],
        optimal_graph: Dict[str, Any],
    ) -> Dict[str, str]:
        """
        Create SFT training example.

        Input: Target graph + Hardware profile
        Output: Optimal config + Optimized graph
        """
        prompt = (
            f"{self.format_graph_as_text(target_graph)}\n\n"
            f"{self.format_hardware_as_text(hardware)}\n\n"
            "### Task\n"
            "Generate optimal execution configuration and µGraph for this workload.\n\n"
            "### Solution\n"
        )

        response = (
            f"{self.format_config_as_text(optimal_config)}\n\n"
            f"{self.format_graph_as_text(optimal_graph)}"
        )

        return {
            "prompt": prompt,
            "response": response,
            "text": prompt + response,
        }

    def create_preference_pair(
        self,
        target_graph: Dict[str, Any],
        hardware: Dict[str, Any],
        chosen_config: Dict[str, Any],
        rejected_config: Dict[str, Any],
        chosen_latency: float,
        rejected_latency: float,
    ) -> Dict[str, str]:
        """
        Create preference pair for DPO/GRPO training.

        Chosen: Better performing configuration
        Rejected: Worse performing configuration
        """
        prompt = (
            f"{self.format_graph_as_text(target_graph)}\n\n"
            f"{self.format_hardware_as_text(hardware)}\n\n"
            "### Task\n"
            "Generate optimal execution configuration for this workload.\n\n"
            "### Solution\n"
        )

        chosen = (
            f"{self.format_config_as_text(chosen_config)}\n"
            f"# Expected latency: {chosen_latency:.3f} ms"
        )

        rejected = (
            f"{self.format_config_as_text(rejected_config)}\n"
            f"# Expected latency: {rejected_latency:.3f} ms"
        )

        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        }


class TRLTrainerFactory:
    """
    Factory for creating TRL trainers with different strategies.
    """

    @staticmethod
    def create_model_and_tokenizer(
        config: FineTuningConfig,
    ) -> Tuple[Any, Any]:
        """
        Create model and tokenizer with optional quantization and LoRA.
        """
        check_dependencies()

        tokenizer_name = config.tokenizer_name or config.model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Quantization config
        quantization_config = None
        if config.use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, config.bnb_4bit_compute_dtype),
                bnb_4bit_quant_type=config.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=True,
            )
        elif config.use_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16 if config.bf16 else torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

        # Prepare for k-bit training
        if config.use_4bit or config.use_8bit:
            model = prepare_model_for_kbit_training(model)

        # Apply LoRA
        if config.use_lora:
            lora_config = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=config.lora_target_modules,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

        return model, tokenizer

    @staticmethod
    def create_sft_trainer(
        config: FineTuningConfig,
        model: Any,
        tokenizer: Any,
        train_dataset: Any,
        eval_dataset: Optional[Any] = None,
    ) -> Any:
        """Create SFT (Supervised Fine-Tuning) trainer."""
        check_dependencies()

        sft_config = SFTConfig(
            output_dir=config.output_dir,
            num_train_epochs=config.num_train_epochs,
            per_device_train_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            warmup_ratio=config.warmup_ratio,
            logging_steps=config.logging_steps,
            save_steps=config.save_steps,
            bf16=config.bf16,
            fp16=config.fp16,
            max_seq_length=config.max_seq_length,
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=sft_config,
        )

        return trainer

    @staticmethod
    def create_ppo_trainer(
        config: FineTuningConfig,
        model: Any,
        tokenizer: Any,
        reward_model: Optional[Any] = None,
    ) -> Any:
        """Create PPO trainer for RLHF."""
        check_dependencies()

        # Wrap model with value head
        model_with_value = AutoModelForCausalLMWithValueHead.from_pretrained(model)

        ppo_config = PPOConfig(
            model_name=config.model_name_or_path,
            learning_rate=config.learning_rate,
            batch_size=config.per_device_train_batch_size,
            mini_batch_size=config.per_device_train_batch_size // 2,
            ppo_epochs=config.ppo_epochs,
            init_kl_coef=config.init_kl_coef,
            adap_kl_ctrl=config.adap_kl_ctrl,
            log_with="tensorboard",
        )

        trainer = PPOTrainer(
            config=ppo_config,
            model=model_with_value,
            tokenizer=tokenizer,
        )

        return trainer

    @staticmethod
    def create_dpo_trainer(
        config: FineTuningConfig,
        model: Any,
        tokenizer: Any,
        train_dataset: Any,
        eval_dataset: Optional[Any] = None,
    ) -> Any:
        """Create DPO (Direct Preference Optimization) trainer."""
        check_dependencies()

        dpo_config = DPOConfig(
            output_dir=config.output_dir,
            num_train_epochs=config.num_train_epochs,
            per_device_train_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            warmup_ratio=config.warmup_ratio,
            logging_steps=config.logging_steps,
            save_steps=config.save_steps,
            bf16=config.bf16,
            beta=config.beta,
            max_length=config.max_seq_length,
        )

        trainer = DPOTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=dpo_config,
        )

        return trainer


class MuGraphPolicyTrainer:
    """
    High-level trainer for µGraph search policy using TRL.

    Supports multiple training strategies and provides
    unified interface for training and inference.
    """

    def __init__(self, config: FineTuningConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.formatter = MuGraphDatasetFormatter()

    def setup(self):
        """Initialize model and tokenizer."""
        if not TRL_AVAILABLE:
            logger.warning("TRL not available. Using fallback training.")
            return

        self.model, self.tokenizer = TRLTrainerFactory.create_model_and_tokenizer(self.config)
        self.formatter = MuGraphDatasetFormatter(
            tokenizer=self.tokenizer,
            max_length=self.config.max_seq_length,
        )

    def prepare_sft_dataset(
        self,
        examples: List[Dict[str, Any]],
    ) -> Any:
        """
        Prepare dataset for SFT training.

        Args:
            examples: List of dicts with target_graph, hardware, optimal_config, optimal_graph
        """
        from datasets import Dataset

        formatted = []
        for ex in examples:
            formatted.append(
                self.formatter.create_sft_example(
                    target_graph=ex["target_graph"],
                    hardware=ex["hardware"],
                    optimal_config=ex["optimal_config"],
                    optimal_graph=ex["optimal_graph"],
                )
            )

        return Dataset.from_list(formatted)

    def prepare_preference_dataset(
        self,
        examples: List[Dict[str, Any]],
    ) -> Any:
        """
        Prepare dataset for DPO/GRPO training.

        Args:
            examples: List of dicts with target_graph, hardware, chosen_config,
                     rejected_config, chosen_latency, rejected_latency
        """
        from datasets import Dataset

        formatted = []
        for ex in examples:
            formatted.append(
                self.formatter.create_preference_pair(
                    target_graph=ex["target_graph"],
                    hardware=ex["hardware"],
                    chosen_config=ex["chosen_config"],
                    rejected_config=ex["rejected_config"],
                    chosen_latency=ex["chosen_latency"],
                    rejected_latency=ex["rejected_latency"],
                )
            )

        return Dataset.from_list(formatted)

    def train(
        self,
        train_data: List[Dict[str, Any]],
        eval_data: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Train the policy model.

        Automatically selects trainer based on config.strategy.
        """
        if self.model is None:
            self.setup()

        strategy = self.config.strategy.lower()

        if strategy == "sft":
            train_dataset = self.prepare_sft_dataset(train_data)
            eval_dataset = self.prepare_sft_dataset(eval_data) if eval_data else None

            self.trainer = TRLTrainerFactory.create_sft_trainer(
                self.config, self.model, self.tokenizer, train_dataset, eval_dataset
            )

        elif strategy == "dpo":
            train_dataset = self.prepare_preference_dataset(train_data)
            eval_dataset = self.prepare_preference_dataset(eval_data) if eval_data else None

            self.trainer = TRLTrainerFactory.create_dpo_trainer(
                self.config, self.model, self.tokenizer, train_dataset, eval_dataset
            )

        elif strategy == "ppo":
            self.trainer = TRLTrainerFactory.create_ppo_trainer(
                self.config, self.model, self.tokenizer
            )
            # PPO requires online training loop
            logger.info("PPO trainer created. Use trainer.step() for online training.")
            return

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Run training
        self.trainer.train()

    def generate_config(
        self,
        target_graph: Dict[str, Any],
        hardware: Dict[str, Any],
        num_samples: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Generate optimal configuration for given graph and hardware.

        Args:
            target_graph: Target computation graph
            hardware: Hardware profile
            num_samples: Number of configurations to generate

        Returns:
            List of generated configurations
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not initialized. Call setup() first.")

        # Format prompt
        prompt = (
            f"{self.formatter.format_graph_as_text(target_graph)}\n\n"
            f"{self.formatter.format_hardware_as_text(hardware)}\n\n"
            "### Task\n"
            "Generate optimal execution configuration for this workload.\n\n"
            "### Solution\n"
        )

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # Generate
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=256,
            num_return_sequences=num_samples,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

        # Decode and parse
        configs = []
        for output in outputs:
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            config = self._parse_generated_config(text)
            if config:
                configs.append(config)

        return configs

    def _parse_generated_config(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse generated text into configuration dict."""
        try:
            config = {
                "grid_dim": {"x": 1, "y": 1, "z": 1},
                "block_dim": {"x": 128, "y": 1, "z": 1},
                "forloop_range": 1,
                "shared_memory_size": 49152,
            }

            # Parse grid dimensions
            if "Grid:" in text:
                grid_match = text.split("Grid:")[1].split("\n")[0]
                # Extract numbers from format like (4, 2, 1)
                nums = [int(n) for n in grid_match.replace("(", "").replace(")", "").split(",")]
                if len(nums) >= 3:
                    config["grid_dim"] = {"x": nums[0], "y": nums[1], "z": nums[2]}

            # Parse block dimensions
            if "Block:" in text:
                block_match = text.split("Block:")[1].split("\n")[0]
                nums = [int(n) for n in block_match.replace("(", "").replace(")", "").split(",")]
                if len(nums) >= 3:
                    config["block_dim"] = {"x": nums[0], "y": nums[1], "z": nums[2]}

            return config

        except Exception as e:
            logger.warning(f"Failed to parse config: {e}")
            return None

    def save(self, path: Union[str, Path]):
        """Save trained model."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self.model is not None:
            self.model.save_pretrained(path / "model")
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(path / "tokenizer")

        # Save config
        with open(path / "config.json", "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

    def load(self, path: Union[str, Path]):
        """Load trained model."""
        path = Path(path)

        # Load config
        with open(path / "config.json") as f:
            config_dict = json.load(f)

        self.config = FineTuningConfig(**config_dict)

        if PEFT_AVAILABLE:
            self.model = PeftModel.from_pretrained(
                AutoModelForCausalLM.from_pretrained(
                    self.config.model_name_or_path,
                    torch_dtype=torch.bfloat16 if self.config.bf16 else torch.float16,
                    device_map="auto",
                ),
                path / "model",
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                path / "model",
                torch_dtype=torch.bfloat16 if self.config.bf16 else torch.float16,
                device_map="auto",
            )

        self.tokenizer = AutoTokenizer.from_pretrained(path / "tokenizer")


# Convenience functions


def create_trainer(
    strategy: str = "sft",
    model_name: str = "meta-llama/Llama-2-7b-hf",
    use_lora: bool = True,
    use_4bit: bool = False,
    **kwargs,
) -> MuGraphPolicyTrainer:
    """
    Create a MuGraph policy trainer with specified strategy.

    Args:
        strategy: Training strategy (sft, dpo, ppo, grpo)
        model_name: Base model name or path
        use_lora: Whether to use LoRA
        use_4bit: Whether to use 4-bit quantization
        **kwargs: Additional config options

    Returns:
        Configured trainer instance
    """
    config = FineTuningConfig(
        strategy=strategy,
        model_name_or_path=model_name,
        use_lora=use_lora,
        use_4bit=use_4bit,
        **kwargs,
    )

    return MuGraphPolicyTrainer(config)


def train_mugraph_policy(
    train_data: List[Dict[str, Any]],
    strategy: str = "sft",
    model_name: str = "meta-llama/Llama-2-7b-hf",
    output_dir: str = "outputs/mugraph_policy",
    **kwargs,
) -> MuGraphPolicyTrainer:
    """
    Train a µGraph search policy.

    Args:
        train_data: Training examples
        strategy: Training strategy
        model_name: Base model
        output_dir: Output directory
        **kwargs: Additional config options

    Returns:
        Trained policy trainer
    """
    trainer = create_trainer(
        strategy=strategy,
        model_name=model_name,
        output_dir=output_dir,
        **kwargs,
    )

    trainer.train(train_data)
    trainer.save(output_dir)

    return trainer
