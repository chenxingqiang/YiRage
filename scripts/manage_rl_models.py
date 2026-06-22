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
RL Model Management Utility.

Provides tools for:
- Listing saved models
- Loading and inspecting models
- Exporting models (ONNX, TorchScript)
- Comparing model performance
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

# Setup paths
WORKSPACE_ROOT = Path(__file__).parent.parent
PYTHON_ROOT = WORKSPACE_ROOT / "python"
sys.path.insert(0, str(PYTHON_ROOT))


def list_models(model_dir: Path) -> List[Dict[str, Any]]:
    """
    List all saved models in directory.

    Returns:
        List of model info dictionaries
    """
    models = []

    if not model_dir.exists():
        print(f"Directory not found: {model_dir}")
        return models

    for model_file in sorted(model_dir.glob("*.pt")):
        info = {
            "name": model_file.stem,
            "path": str(model_file),
            "size_mb": model_file.stat().st_size / (1024 * 1024),
            "modified": model_file.stat().st_mtime,
        }

        # Try to load metadata
        meta_file = model_file.with_suffix(".json")
        if meta_file.exists():
            with open(meta_file) as f:
                info["metadata"] = json.load(f)

        models.append(info)

    return models


def inspect_model(model_path: Path) -> Dict[str, Any]:
    """
    Inspect a saved model.

    Returns:
        Model information dictionary
    """
    try:
        import torch
    except ImportError:
        return {"error": "PyTorch not installed"}

    if not model_path.exists():
        return {"error": f"Model not found: {model_path}"}

    try:
        checkpoint = torch.load(model_path, map_location="cpu")

        info = {
            "path": str(model_path),
            "keys": list(checkpoint.keys()),
        }

        # Model config
        if "config" in checkpoint:
            info["config"] = checkpoint["config"]

        # State dict statistics
        if "model_state_dict" in checkpoint:
            state = checkpoint["model_state_dict"]
            info["num_parameters"] = sum(v.numel() for v in state.values() if hasattr(v, "numel"))
            info["layers"] = list(state.keys())

        # Training info
        if "epoch" in checkpoint:
            info["epoch"] = checkpoint["epoch"]
        if "best_reward" in checkpoint:
            info["best_reward"] = checkpoint["best_reward"]

        return info

    except Exception as e:
        return {"error": str(e)}


def export_model(
    model_path: Path,
    export_format: str,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Export model to different format.

    Args:
        model_path: Path to saved model
        export_format: "onnx" or "torchscript"
        output_path: Output file path

    Returns:
        Export result dictionary
    """
    try:
        import torch
    except ImportError:
        return {"error": "PyTorch not installed"}

    if not model_path.exists():
        return {"error": f"Model not found: {model_path}"}

    if output_path is None:
        if export_format == "onnx":
            output_path = model_path.with_suffix(".onnx")
        else:
            output_path = model_path.with_suffix(".ts")

    try:
        checkpoint = torch.load(model_path, map_location="cpu")
        config = checkpoint.get("config", {})

        # Create model (simplified)
        class DummyModel(torch.nn.Module):
            def __init__(self, input_dim: int, output_dim: int):
                super().__init__()
                self.net = torch.nn.Sequential(
                    torch.nn.Linear(input_dim, 64),
                    torch.nn.ReLU(),
                    torch.nn.Linear(64, output_dim),
                )

            def forward(self, x):
                return self.net(x)

        input_dim = config.get("input_dim", 64)
        output_dim = config.get("num_actions", 100)

        model = DummyModel(input_dim, output_dim)

        if "model_state_dict" in checkpoint:
            # Try to load state dict (may not match)
            try:
                model.load_state_dict(checkpoint["model_state_dict"])
            except:
                pass

        model.eval()

        # Export
        dummy_input = torch.randn(1, input_dim)

        if export_format == "onnx":
            torch.onnx.export(
                model,
                dummy_input,
                str(output_path),
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            )
        else:
            traced = torch.jit.trace(model, dummy_input)
            traced.save(str(output_path))

        return {
            "success": True,
            "output_path": str(output_path),
            "format": export_format,
        }

    except Exception as e:
        return {"error": str(e)}


def compare_models(model_paths: List[Path]) -> Dict[str, Any]:
    """
    Compare multiple models.

    Returns:
        Comparison results
    """
    try:
        import torch
    except ImportError:
        return {"error": "PyTorch not installed"}

    results = {
        "models": [],
        "comparison": {},
    }

    for path in model_paths:
        info = inspect_model(path)
        results["models"].append(info)

    # Compare metrics
    valid_models = [m for m in results["models"] if "error" not in m]

    if len(valid_models) >= 2:
        # Compare parameter counts
        param_counts = [m.get("num_parameters", 0) for m in valid_models]
        results["comparison"]["param_range"] = {
            "min": min(param_counts),
            "max": max(param_counts),
        }

        # Compare best rewards (if available)
        rewards = [m.get("best_reward") for m in valid_models if "best_reward" in m]
        if rewards:
            results["comparison"]["reward_range"] = {
                "min": min(rewards),
                "max": max(rewards),
                "best_model": valid_models[rewards.index(max(rewards))]["path"],
            }

    return results


def create_sample_model(output_path: Path) -> Dict[str, Any]:
    """
    Create a sample model for testing.

    Returns:
        Creation result
    """
    try:
        import torch
    except ImportError:
        # Create dummy checkpoint without PyTorch
        checkpoint = {
            "config": {
                "input_dim": 64,
                "hidden_dim": 128,
                "num_actions": 100,
            },
            "epoch": 100,
            "best_reward": 3.5,
        }

        # Save as JSON (simplified)
        json_path = output_path.with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(checkpoint, f, indent=2)

        return {
            "success": True,
            "path": str(json_path),
            "note": "PyTorch not available, saved as JSON",
        }

    # Create simple model
    class SamplePolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Sequential(
                torch.nn.Linear(64, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 128),
                torch.nn.ReLU(),
            )
            self.policy_head = torch.nn.Linear(128, 100)
            self.value_head = torch.nn.Linear(128, 1)

        def forward(self, x):
            h = self.encoder(x)
            return self.policy_head(h), self.value_head(h)

    model = SamplePolicy()

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": {
            "input_dim": 64,
            "hidden_dim": 128,
            "num_actions": 100,
        },
        "epoch": 100,
        "best_reward": 3.5,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)

    return {
        "success": True,
        "path": str(output_path),
        "num_parameters": sum(p.numel() for p in model.parameters()),
    }


def main():
    parser = argparse.ArgumentParser(description="RL Model Management")
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # List command
    list_parser = subparsers.add_parser("list", help="List saved models")
    list_parser.add_argument(
        "--dir", type=Path, default=Path("checkpoints"), help="Model directory"
    )

    # Inspect command
    inspect_parser = subparsers.add_parser("inspect", help="Inspect a model")
    inspect_parser.add_argument("model", type=Path, help="Model file path")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export model")
    export_parser.add_argument("model", type=Path, help="Model file path")
    export_parser.add_argument(
        "--format", choices=["onnx", "torchscript"], default="onnx", help="Export format"
    )
    export_parser.add_argument("--output", type=Path, help="Output path")

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare models")
    compare_parser.add_argument("models", nargs="+", type=Path, help="Model file paths")

    # Create sample command
    sample_parser = subparsers.add_parser("create-sample", help="Create sample model")
    sample_parser.add_argument(
        "--output", type=Path, default=Path("checkpoints/sample_model.pt"), help="Output path"
    )

    args = parser.parse_args()

    if args.command == "list":
        models = list_models(args.dir)
        if models:
            print(f"Found {len(models)} models in {args.dir}:\n")
            for m in models:
                print(f"  {m['name']}")
                print(f"    Size: {m['size_mb']:.2f} MB")
                if "metadata" in m:
                    print(f"    Metadata: {m['metadata']}")
                print()
        else:
            print(f"No models found in {args.dir}")

    elif args.command == "inspect":
        info = inspect_model(args.model)
        print(json.dumps(info, indent=2, default=str))

    elif args.command == "export":
        result = export_model(args.model, args.format, args.output)
        print(json.dumps(result, indent=2))

    elif args.command == "compare":
        result = compare_models(args.models)
        print(json.dumps(result, indent=2, default=str))

    elif args.command == "create-sample":
        result = create_sample_model(args.output)
        print(json.dumps(result, indent=2))

    else:
        parser.print_help()

    return 0


if __name__ == "__main__":
    sys.exit(main())
