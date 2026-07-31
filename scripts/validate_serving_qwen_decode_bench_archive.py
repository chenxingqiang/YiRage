#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""Validate a Qwen decode-step bench archive JSON (S31)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "python") not in sys.path:
    sys.path.insert(0, str(_REPO / "python"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Path to decode bench archive JSON")
    parser.add_argument(
        "--metadata-output",
        metavar="PATH",
        help="Write artifact sidecar metadata JSON after successful validation",
    )
    parser.add_argument("--quick", action="store_true", help="Record quick=true in metadata")
    args = parser.parse_args()

    from yirage.serving.decode_bench_archive import (
        load_serving_qwen_decode_bench_archive,
        serving_qwen_decode_bench_archive_metadata,
        validate_serving_qwen_decode_bench_archive,
    )

    payload = load_serving_qwen_decode_bench_archive(args.path)
    errors = validate_serving_qwen_decode_bench_archive(payload)
    if errors:
        print(json.dumps({"ok": False, "errors": errors}, indent=2))
        return 1

    summary: dict = {"ok": True, "parity_ok": payload.get("parity_ok")}
    if args.metadata_output:
        metadata = serving_qwen_decode_bench_archive_metadata(
            payload,
            archive_path=args.path,
            validation_ok=True,
            quick=args.quick,
        )
        Path(args.metadata_output).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        summary["metadata_output"] = args.metadata_output
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
