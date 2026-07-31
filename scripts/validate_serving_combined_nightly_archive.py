#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""Validate a combined Serving nightly archive JSON (S34)."""

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
    parser.add_argument("path", help="Path to combined nightly archive JSON")
    parser.add_argument(
        "--metadata-output",
        metavar="PATH",
        help="Write artifact sidecar metadata JSON after successful validation",
    )
    parser.add_argument("--quick", action="store_true", help="Record quick=true in metadata")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow decode/multistep=null (G1-only smoke archives)",
    )
    args = parser.parse_args()

    from yirage.serving.combined_nightly_archive import (
        serving_combined_nightly_archive_metadata,
        validate_serving_combined_nightly_archive,
    )

    payload = json.loads(Path(args.path).read_text(encoding="utf-8"))
    errors = validate_serving_combined_nightly_archive(payload)
    if args.allow_partial:
        errors = [e for e in errors if not e.endswith("subsection must be a dict")]
        if payload.get("decode") is None:
            errors = [e for e in errors if not e.startswith("decode.")]
        if payload.get("multistep") is None:
            errors = [e for e in errors if not e.startswith("multistep.")]
        if payload.get("engine_multistep") is None:
            errors = [e for e in errors if not e.startswith("engine_multistep.")]
    if errors:
        print(json.dumps({"ok": False, "errors": errors}, indent=2))
        return 1

    summary: dict = {"ok": True, "parity_ok": payload.get("parity_ok")}
    if args.metadata_output:
        metadata = serving_combined_nightly_archive_metadata(
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
