#!/usr/bin/env python3
"""Qwen MACA e2e bench — delegates to ``demo/maca/qwen_inference_demo.py`` (CUDA qwen2.5 aligned)."""

import argparse
import os
import subprocess
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_DEMO = os.path.join(_REPO, "demo", "maca", "qwen_inference_demo.py")


def main() -> int:
    parser = argparse.ArgumentParser(description="Qwen MACA e2e (full-chain inference smoke)")
    parser.add_argument("--quick", action="store_true", default=True, help="Tractable search (default)")
    parser.add_argument("--full-search", action="store_true", help="Full MACA search grid")
    parser.add_argument("--decode-steps", type=int, default=4)
    parser.add_argument("--skip-search", action="store_true", help="Skip (exit 0 without running demo)")
    args = parser.parse_args()

    if args.skip_search:
        print("Skipping Qwen MACA search (--skip-search)")
        return 0

    cmd = [sys.executable, _DEMO, "--decode-steps", str(args.decode_steps)]
    if args.full_search:
        cmd.append("--full-search")
    else:
        cmd.append("--quick")

    print("Running:", " ".join(cmd))
    return subprocess.call(cmd, cwd=_REPO)


if __name__ == "__main__":
    raise SystemExit(main())
