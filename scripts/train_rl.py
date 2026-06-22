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
RL-Guided Kernel Search Training Script

This script demonstrates the complete closed-loop training process:

    ┌─────────────────────────────────────────────────────────────┐
    │                     Closed Loop                              │
    │                                                              │
    │  RL Policy ──action──> YiRage Env ──verify(GPU)──> reward   │
    │      │                     │                        │        │
    │      └──────── obs <───────┴───── feedback <────────┘        │
    └─────────────────────────────────────────────────────────────┘

Usage:
    # Basic training
    python scripts/train_rl_kernel_search.py

    # Training with custom config
    python scripts/train_rl_kernel_search.py \
        --algorithm PPO \
        --num-workers 8 \
        --max-iterations 1000 \
        --target-graph examples/matmul.json

    # Resume from checkpoint
    python scripts/train_rl_kernel_search.py \
        --resume /path/to/checkpoint
"""

import argparse
import json
import sys
import types
import importlib.util
from pathlib import Path

# Add yirage to path
YIRAGE_PYTHON_PATH = Path(__file__).parent.parent / "python"
sys.path.insert(0, str(YIRAGE_PYTHON_PATH))


def _load_module_from_file(module_name: str, file_path: Path):
    """Load a module directly from file path, bypassing package structure."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _setup_rl_modules():
    """Set up RL modules without going through yirage.__init__."""
    rl_base = YIRAGE_PYTHON_PATH / "yirage" / "rl"

    # Create fake yirage package structure
    yirage_pkg = types.ModuleType("yirage")
    yirage_pkg.__path__ = [str(YIRAGE_PYTHON_PATH / "yirage")]
    sys.modules["yirage"] = yirage_pkg

    # Create fake subpackages
    for pkg_name in ["_cython", "rl", "rl.env", "rl.verifier", "rl.training", "rl.models"]:
        full_name = f"yirage.{pkg_name}"
        pkg = types.ModuleType(full_name)
        pkg.__path__ = [str(YIRAGE_PYTHON_PATH / "yirage" / pkg_name.replace(".", "/"))]
        sys.modules[full_name] = pkg

        # Set as attribute on parent
        parts = full_name.split(".")
        parent = sys.modules[".".join(parts[:-1])]
        setattr(parent, parts[-1], pkg)

    # Load actual modules in order (dependencies first)
    modules_to_load = [
        # Verifier modules
        ("yirage.rl.verifier.gpu_verifier", rl_base / "verifier" / "gpu_verifier.py"),
        ("yirage.rl.verifier.verifier_pool", rl_base / "verifier" / "verifier_pool.py"),
    ]

    for mod_name, mod_path in modules_to_load:
        if mod_path.exists():
            _load_module_from_file(mod_name, mod_path)

    # Update verifier __init__ exports
    verifier_pkg = sys.modules["yirage.rl.verifier"]
    from yirage.rl.verifier.gpu_verifier import (
        GPUVerifier,
        VerifyResult,
        ProfileResult,
        LocalGPUVerifier,
    )
    from yirage.rl.verifier.verifier_pool import VerifierPool

    verifier_pkg.GPUVerifier = GPUVerifier
    verifier_pkg.VerifyResult = VerifyResult
    verifier_pkg.ProfileResult = ProfileResult
    verifier_pkg.LocalGPUVerifier = LocalGPUVerifier
    verifier_pkg.VerifierPool = VerifierPool

    # Load env modules
    modules_to_load = [
        ("yirage.rl.env.action_space", rl_base / "env" / "action_space.py"),
        ("yirage.rl.env.observation", rl_base / "env" / "observation.py"),
        ("yirage.rl.env.reward", rl_base / "env" / "reward.py"),
        ("yirage.rl.env.yirage_env", rl_base / "env" / "yirage_env.py"),
    ]

    for mod_name, mod_path in modules_to_load:
        if mod_path.exists():
            _load_module_from_file(mod_name, mod_path)

    # Update env __init__ exports
    env_pkg = sys.modules["yirage.rl.env"]
    from yirage.rl.env.yirage_env import YiRageSearchEnv, EnvConfig
    from yirage.rl.env.action_space import ActionSpace, ActionDecoder
    from yirage.rl.env.observation import ObservationSpace, ObservationEncoder

    env_pkg.YiRageSearchEnv = YiRageSearchEnv
    env_pkg.EnvConfig = EnvConfig
    env_pkg.ActionSpace = ActionSpace
    env_pkg.ActionDecoder = ActionDecoder
    env_pkg.ObservationSpace = ObservationSpace
    env_pkg.ObservationEncoder = ObservationEncoder

    # Load model and training modules
    modules_to_load = [
        ("yirage.rl.models.graph_encoder", rl_base / "models" / "graph_encoder.py"),
        ("yirage.rl.training.callbacks", rl_base / "training" / "callbacks.py"),
        ("yirage.rl.training.trainer", rl_base / "training" / "trainer.py"),
    ]

    for mod_name, mod_path in modules_to_load:
        if mod_path.exists():
            _load_module_from_file(mod_name, mod_path)


# Initialize RL modules
_setup_rl_modules()


def create_sample_target_graph():
    """Create a sample target graph for training."""
    return json.dumps(
        {
            "name": "matmul_add",
            "inputs": [
                {"name": "A", "dims": [8, 4096], "dtype": "float16"},
                {"name": "B", "dims": [4096, 4096], "dtype": "float16"},
                {"name": "C", "dims": [8, 4096], "dtype": "float16"},
            ],
            "operators": [
                {
                    "type": "matmul",
                    "inputs": [0, 1],
                    "outputs": [3],
                },
                {
                    "type": "add",
                    "inputs": [3, 2],
                    "outputs": [4],
                },
            ],
            "outputs": [4],
        }
    )


def train_local(args):
    """
    Local training without Ray (for testing).

    Runs a few episodes to verify the environment works.
    """
    print("=== Local Training Mode ===")
    print("Testing environment without Ray/RLlib...")

    # Import environment components
    from yirage.rl.env.yirage_env import YiRageSearchEnv, EnvConfig
    from yirage.rl.env.reward import RewardConfig

    # Create config
    env_config = EnvConfig(
        target_graph_json=create_sample_target_graph(),
        backend=args.backend,
        num_gpus=args.num_gpus,
        max_search_depth=args.max_depth,
        reward_config=RewardConfig(
            validity_weight=1.0,
            performance_weight=0.5,
        ),
    )

    # Create environment
    print("\nCreating YiRageSearchEnv...")
    env = YiRageSearchEnv(vars(env_config))

    print(f"Action space: {env.action_space}")
    print(f"Observation space keys: {list(env.observation_space.spaces.keys())}")

    # Run a few episodes
    total_reward = 0
    total_valid = 0

    for episode in range(args.test_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        step = 0

        while not done and step < args.max_depth:
            # Random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            done = terminated or truncated
            step += 1

        if info.get("num_valid_found", 0) > 0:
            total_valid += 1

        print(
            f"Episode {episode + 1}: "
            f"reward={episode_reward:.3f}, "
            f"steps={step}, "
            f"valid_found={info.get('num_valid_found', 0)}, "
            f"best_latency={info.get('best_latency_ms', 'N/A')}"
        )

        total_reward += episode_reward

    print(f"\nAverage reward: {total_reward / args.test_episodes:.3f}")
    print(f"Episodes with valid kernels: {total_valid}/{args.test_episodes}")

    env.close()
    print("\nLocal training test completed!")


def train_with_ray(args):
    """
    Full training with Ray/RLlib.
    """
    print("=== Ray/RLlib Training Mode ===")

    try:
        import ray
        from ray import tune
    except ImportError:
        print("Error: Ray is not installed. Install with: pip install 'ray[rllib]'")
        sys.exit(1)

    from yirage.rl.env.yirage_env import YiRageSearchEnv, EnvConfig
    from yirage.rl.training.trainer import train_rl_search, TrainingConfig
    from yirage.rl.env.reward import RewardConfig

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(num_cpus=args.num_workers + 2)

    # Create training config
    env_config = EnvConfig(
        target_graph_json=args.target_graph or create_sample_target_graph(),
        backend=args.backend,
        num_gpus=args.num_gpus,
        max_search_depth=args.max_depth,
    )

    training_config = TrainingConfig(
        algorithm=args.algorithm,
        env_config=env_config,
        num_workers=args.num_workers,
        train_batch_size=args.batch_size,
        max_iterations=args.max_iterations,
        checkpoint_freq=args.checkpoint_freq,
        checkpoint_dir=args.checkpoint_dir,
    )

    # Train
    print(f"\nStarting training with {args.algorithm}...")
    print(f"Workers: {args.num_workers}")
    print(f"Max iterations: {args.max_iterations}")

    results = train_rl_search(
        config=training_config,
        resume_checkpoint=args.resume,
    )

    print("\n=== Training Complete ===")
    print(f"Best checkpoint: {results['best_checkpoint']}")

    if results.get("best_metrics"):
        metrics = results["best_metrics"]
        print(f"Final metrics:")
        print(
            f"  - Valid kernels found (mean): {metrics.get('custom_metrics', {}).get('total_valid_mean', 'N/A')}"
        )
        print(
            f"  - Best latency (mean): {metrics.get('custom_metrics', {}).get('best_latency_ms_mean', 'N/A')}"
        )

    ray.shutdown()
    return results


def search_with_trained_policy(args):
    """
    Use a trained policy to search for optimal kernels.
    """
    print("=== Inference Mode ===")

    if not args.checkpoint:
        print("Error: --checkpoint required for inference mode")
        sys.exit(1)

    try:
        import ray
    except ImportError:
        print("Error: Ray is required. Install with: pip install 'ray[rllib]'")
        sys.exit(1)

    from yirage.rl.training.trainer import load_trained_policy, search_with_policy
    from yirage.rl.env.yirage_env import EnvConfig

    # Initialize Ray
    if not ray.is_initialized():
        ray.init()

    # Load policy
    print(f"Loading policy from {args.checkpoint}...")
    policy = load_trained_policy(args.checkpoint)

    # Create env config
    env_config = EnvConfig(
        backend=args.backend,
        num_gpus=args.num_gpus,
        max_search_depth=args.max_depth,
    )

    # Search
    target_graph = args.target_graph or create_sample_target_graph()
    print(f"\nSearching for optimal kernels...")

    results = search_with_policy(
        policy=policy,
        target_graph_json=target_graph,
        env_config=env_config,
        max_episodes=args.search_episodes,
    )

    print(f"\n=== Search Results ===")
    print(f"Best latency: {results['best_latency_ms']:.3f}ms")
    print(f"Top kernels found: {len(results['best_kernels'])}")

    for i, kernel in enumerate(results["best_kernels"][:5]):
        print(f"  {i+1}. Latency: {kernel['latency_ms']:.3f}ms (episode {kernel['episode']})")

    ray.shutdown()
    return results


def main():
    parser = argparse.ArgumentParser(
        description="RL-Guided Kernel Search Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Mode
    parser.add_argument(
        "--mode",
        choices=["local", "train", "search"],
        default="local",
        help="Running mode: local (test), train (full training), search (inference)",
    )

    # Algorithm
    parser.add_argument(
        "--algorithm",
        choices=["PPO", "SAC"],
        default="PPO",
        help="RL algorithm to use",
    )

    # Resources
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of rollout workers",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs for verification",
    )
    parser.add_argument(
        "--backend",
        default="cuda",
        help="Target backend (cuda, maca, etc.)",
    )

    # Training
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=100,
        help="Maximum training iterations",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Training batch size",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=50,
        help="Maximum search depth per episode",
    )

    # Checkpointing
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=10,
        help="Checkpoint frequency",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="./checkpoints",
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--resume",
        help="Resume from checkpoint",
    )
    parser.add_argument(
        "--checkpoint",
        help="Checkpoint to use for inference",
    )

    # Target graph
    parser.add_argument(
        "--target-graph",
        help="Path to target graph JSON file",
    )

    # Local testing
    parser.add_argument(
        "--test-episodes",
        type=int,
        default=5,
        help="Number of episodes for local testing",
    )

    # Search
    parser.add_argument(
        "--search-episodes",
        type=int,
        default=100,
        help="Number of episodes for search",
    )

    args = parser.parse_args()

    # Load target graph from file if provided
    if args.target_graph and Path(args.target_graph).exists():
        with open(args.target_graph) as f:
            args.target_graph = f.read()

    # Run
    if args.mode == "local":
        train_local(args)
    elif args.mode == "train":
        train_with_ray(args)
    elif args.mode == "search":
        search_with_trained_policy(args)


if __name__ == "__main__":
    main()
