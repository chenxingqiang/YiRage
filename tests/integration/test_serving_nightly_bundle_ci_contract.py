# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""Integration: Serving nightly bundle CI contract manifest."""

from __future__ import annotations

import sys
import types
from pathlib import Path


def _bootstrap():
    root = Path(__file__).resolve().parents[2]
    pkg_root = root / "python"
    if str(pkg_root) not in sys.path:
        sys.path.insert(0, str(pkg_root))
    if str(root / "tests" / "python") not in sys.path:
        sys.path.insert(0, str(root / "tests" / "python"))
    if "yirage" not in sys.modules or not hasattr(sys.modules.get("yirage"), "core"):
        stub = types.ModuleType("yirage")
        stub.__path__ = [str(pkg_root / "yirage")]  # type: ignore[attr-defined]
        sys.modules["yirage"] = stub
    from yirage.serving.serving_nightly_bundle import (
        serving_nightly_bundle_ci_contract,
        validate_serving_nightly_bundle_ci_contract,
    )

    return serving_nightly_bundle_ci_contract, validate_serving_nightly_bundle_ci_contract


def test_serving_nightly_bundle_ci_contract_paths_exist():
    contract_fn, validate_fn = _bootstrap()
    contract = contract_fn()
    assert validate_fn(contract) == []
    root = Path(__file__).resolve().parents[2]
    assert (root / contract["validate_cli"]).is_file()
    assert (root / contract["workflow_path"]).is_file()
    for rel in contract["contract_pytests"]:
        assert (root / rel).is_file(), rel
    for rel in contract["bench_smokes"]:
        assert (root / rel).is_file(), rel
