#!/usr/bin/env python3
"""
Run all MACA end-to-end benchmarks.

Usage:
    python run_all.py [--batch-size 8] [--skip-search]
"""

import subprocess
import sys
import argparse
import os

BENCHMARKS = [
    ("chameleon_maca.py", "Chameleon"),
    ("llama_maca.py", "LLaMA"),
    ("lora_maca.py", "LoRA"),
    ("ngpt_maca.py", "nGPT"),
]


def run_benchmark(script, name, args):
    """Run a single benchmark"""
    print()
    print("=" * 70)
    print(f"Running {name} Benchmark")
    print("=" * 70)
    
    script_path = os.path.join(os.path.dirname(__file__), script)
    
    cmd = [sys.executable, script_path]
    cmd.extend(["--batch-size", str(args.batch_size)])
    cmd.extend(["--warmup", str(args.warmup)])
    cmd.extend(["--repeat", str(args.repeat)])
    
    if args.skip_search:
        cmd.append("--skip-search")
    
    # Add lora-specific args
    if "lora" in script.lower():
        cmd.extend(["--lora-rank", str(args.lora_rank)])
    
    try:
        result = subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {name}: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\nInterrupted while running {name}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run all MACA benchmarks")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--warmup", type=int, default=16, help="Warmup iterations")
    parser.add_argument("--repeat", type=int, default=1000, help="Benchmark repetitions")
    parser.add_argument("--skip-search", action="store_true", help="Skip optimization search")
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--benchmark", type=str, default=None,
                       choices=["chameleon", "llama", "lora", "ngpt"],
                       help="Run specific benchmark only")
    args = parser.parse_args()
    
    print("=" * 70)
    print("MACA End-to-End Benchmarks")
    print("=" * 70)
    
    # Check environment
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"Device: {torch.cuda.get_device_name(0)}")
        else:
            print("Warning: CUDA not available")
    except ImportError:
        print("Error: PyTorch not installed")
        return 1
    
    try:
        import yirage
        print(f"YiRage: loaded")
    except ImportError:
        print("Error: YiRage not installed")
        return 1
    
    # Filter benchmarks if specific one requested
    benchmarks = BENCHMARKS
    if args.benchmark:
        benchmarks = [(s, n) for s, n in BENCHMARKS if args.benchmark in s.lower()]
    
    # Run benchmarks
    results = {}
    for script, name in benchmarks:
        success = run_benchmark(script, name, args)
        results[name] = success
    
    # Summary
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    for name, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {name}")
    
    passed = sum(1 for s in results.values() if s)
    print(f"\nPassed: {passed}/{len(results)}")
    
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())

