#!/usr/bin/env python3
"""
MACA Qwen3 PersistentKernel scaffold — aligned to ``demo/qwen3/demo.py --use-yirage``.

CUDA reference builds an offline ``yirage.PersistentKernel`` task graph, compiles with
nvcc, and runs LLM serving. This MACA demo closes the **smoke/contract** gap:

  - ``--inspect-only``: Cloud/CI scaffold report (no GPU, no weights).
  - Default / ``--quick``: MetaX mcPytorch device + MACA ``PKRuntime`` offline init +
    worker/scheduler counts from ``get_configurations_from_gpu``.

Full ``ypk.compile()`` → mxcc task-graph generation remains **experimental backlog**
(see ``inspect_qwen3_pk_scaffold()`` compile_note).

MetaX VM:
  export MACA_PATH=/opt/maca YIRAGE_BACKEND=maca PYTHONPATH=.
  python3 demo/maca/qwen3_persistent_kernel_demo.py --inspect-only
  python3 demo/maca/qwen3_persistent_kernel_demo.py --quick
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from demo.maca.qwen3_pk_utils import (  # noqa: E402
    Qwen3PKScaffold,
    inspect_maca_pk_compile_contract,
    inspect_qwen3_pk_scaffold,
    maca_pk_runtime_smoke,
    resolve_maca_pk_workers_schedulers,
)


def _apply_maca_env() -> None:
    os.environ.setdefault("YIRAGE_MACA_SEARCH_QUICK", "1")
    os.environ.setdefault("MACA_PATH", "/opt/maca")
    os.environ.setdefault("YIRAGE_BACKEND", "maca")


def _is_maca_device() -> bool:
    import torch

    if not torch.cuda.is_available():
        return False
    name = torch.cuda.get_device_name(0)
    if "MetaX" in name:
        return True
    return os.environ.get("YIRAGE_MACA_ALLOW_NON_METAX", "") == "1"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Qwen3 PersistentKernel MACA scaffold (CUDA demo/qwen3/demo.py --use-yirage)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=Qwen3PKScaffold.model,
        help="HuggingFace model id (config shapes only unless full PK lands)",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=Qwen3PKScaffold.max_num_batched_tokens,
    )
    parser.add_argument(
        "--max-num-batched-requests",
        type=int,
        default=Qwen3PKScaffold.max_num_batched_requests,
    )
    parser.add_argument("--page-size", type=int, default=Qwen3PKScaffold.page_size)
    parser.add_argument("--max-num-pages", type=int, default=Qwen3PKScaffold.max_num_pages)
    parser.add_argument("--max-seq-length", type=int, default=Qwen3PKScaffold.max_seq_length)
    parser.add_argument(
        "--compile-inspect",
        action="store_true",
        help="Validate mxcc PK compile contract (no GPU required)",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Print scaffold JSON and exit (no GPU required)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        default=True,
        help="Tractable smoke: PK runtime init only (default on)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON report",
    )
    args = parser.parse_args()

    _apply_maca_env()

    scaffold = Qwen3PKScaffold(
        model=args.model,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_batched_requests=args.max_num_batched_requests,
        page_size=args.page_size,
        max_num_pages=args.max_num_pages,
        max_seq_length=args.max_seq_length,
    )

    report: dict = {"scaffold": inspect_qwen3_pk_scaffold(scaffold)}

    if args.compile_inspect:
        report["compile_contract"] = inspect_maca_pk_compile_contract(scaffold)
        report["status"] = "compile_inspect"
        if not report["compile_contract"]["compile_ready"]:
            print("✗ PK compile contract failed", file=sys.stderr)
            if args.json:
                print(json.dumps(report, indent=2))
            return 1
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print("=" * 70)
            print("MACA Qwen3 PK mxcc compile contract")
            print("=" * 70)
            for key, val in report["compile_contract"].items():
                print(f"  {key}: {val}")
            print()
            print("PASS — compile-inspect (no MetaX GPU required)")
        return 0

    if args.inspect_only:
        report["status"] = "inspect_only"
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print("=" * 70)
            print("MACA Qwen3 PersistentKernel scaffold (inspect-only)")
            print("=" * 70)
            for key, val in report["scaffold"].items():
                print(f"  {key}: {val}")
            print()
            print("PASS — inspect-only (no MetaX GPU required)")
        return 0

    if not _is_maca_device():
        print("✗ MetaX MACA GPU not detected; use --inspect-only on Cloud VM", file=sys.stderr)
        return 1

    import torch

    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    print(f"✓ Device: {torch.cuda.get_device_name(0)}")

    num_workers, num_schedulers = resolve_maca_pk_workers_schedulers(0)
    report["gpu"] = {
        "device_name": torch.cuda.get_device_name(0),
        "sm_count": torch.cuda.get_device_properties(0).multi_processor_count,
        "num_workers": num_workers,
        "num_schedulers": num_schedulers,
    }
    report["pk_runtime"] = maca_pk_runtime_smoke(
        num_workers=num_workers,
        num_schedulers=num_schedulers,
        device_id=0,
    )

    smem: int | None = None
    try:
        from yirage.utils.common import get_shared_memory_capacity

        smem = get_shared_memory_capacity(70)
    except ImportError:
        pass

    if smem is not None:
        report["smem_capacity"] = smem
        if smem != 65536:
            print(
                f"WARNING: get_shared_memory_capacity(70)={smem}, expected 65536; "
                "run bash scripts/maca_rebuild_core.sh on MetaX VM",
                file=sys.stderr,
            )

    report["status"] = "pass"
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("=" * 70)
        print("MACA Qwen3 PersistentKernel smoke")
        print("=" * 70)
        print(f"  workers/schedulers: {num_workers}/{num_schedulers}")
        print(f"  PK runtime: {report['pk_runtime']}")
        if smem is not None:
            print(f"  smem capacity: {smem}")
        print()
        print("PASS — MACA PK runtime smoke (full ypk.compile mxcc backlog)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
