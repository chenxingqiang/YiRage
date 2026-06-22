# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""Static registry sizes for CPU value-verify inventory (no yirage import)."""

from __future__ import annotations

import re
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_KN_BUILDERS = _REPO / "tests/integration/cpu_op_builders.py"
_TB_BUILDERS = _REPO / "tests/integration/cpu_tb_op_builders.py"

_REGISTRY_NAMES = (
    "KN_OP_BUILDERS",
    "CUSTOMIZED_OP_BUILDERS",
    "FAST_PATH_BUILDERS",
    "LAYOUT_EXPLORE_BUILDERS",
    "TB_OP_BUILDERS",
    "TB_LAYOUT_EXPLORE_BUILDERS",
    "TB_UNSUPPORTED_BUILDERS",
)


def _registry_key_count(path: Path, registry: str) -> int:
    text = path.read_text(encoding="utf-8")
    match = re.search(rf"{registry}\s*=\s*\{{", text)
    if not match:
        return 0
    start = match.end()
    depth = 1
    i = start
    while i < len(text) and depth:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    body = text[start : i - 1]
    return len(re.findall(r'"([a-zA-Z0-9_]+)"\s*:', body))


def registry_sizes() -> dict[str, int]:
    kn_path = _KN_BUILDERS
    tb_path = _TB_BUILDERS
    return {
        "kn_op_builders": _registry_key_count(kn_path, "KN_OP_BUILDERS"),
        "customized_op_builders": _registry_key_count(kn_path, "CUSTOMIZED_OP_BUILDERS"),
        "fast_path_builders": _registry_key_count(kn_path, "FAST_PATH_BUILDERS"),
        "layout_explore_builders": _registry_key_count(kn_path, "LAYOUT_EXPLORE_BUILDERS"),
        "tb_op_builders": _registry_key_count(tb_path, "TB_OP_BUILDERS"),
        "tb_layout_explore_builders": _registry_key_count(
            tb_path, "TB_LAYOUT_EXPLORE_BUILDERS"
        ),
        "tb_unsupported_builders": _registry_key_count(tb_path, "TB_UNSUPPORTED_BUILDERS"),
    }


def planned_value_verify_count() -> int:
    """Matches test_cpu_full_value_verify parametrized cases + four fixed extras."""
    s = registry_sizes()
    fixed_extras = 4  # blas toggle, native primitives, matrix KN/TB coverage audits
    return sum(s.values()) + fixed_extras
