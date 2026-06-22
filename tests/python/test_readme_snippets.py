#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""CPU validation for README Python snippets."""

import re
import sys
import types
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
README = PROJECT_ROOT / "README.md"
PYTHON_ROOT = PROJECT_ROOT / "python"


def _readme_python_snippets() -> list[tuple[int, str]]:
    text = README.read_text(encoding="utf-8")
    snippets: list[tuple[int, str]] = []
    for match in re.finditer(r"```python\n(.*?)```", text, flags=re.DOTALL):
        line_no = text[: match.start()].count("\n") + 1
        snippets.append((line_no, match.group(1)))
    return snippets


def _snippet_containing(marker: str) -> tuple[int, str]:
    for line_no, snippet in _readme_python_snippets():
        if marker in snippet:
            return line_no, snippet
    raise AssertionError(f"README Python snippet containing {marker!r} was not found")


@pytest.fixture()
def yirage_namespace(monkeypatch):
    """Import pure-Python yirage subpackages without requiring the native runtime."""
    for name in list(sys.modules):
        if name == "yirage" or name.startswith("yirage."):
            monkeypatch.delitem(sys.modules, name, raising=False)

    module = types.ModuleType("yirage")
    module.__path__ = [str(PYTHON_ROOT / "yirage")]
    module.__package__ = "yirage"
    monkeypatch.setitem(sys.modules, "yirage", module)
    return module


def test_readme_python_snippets_compile():
    snippets = _readme_python_snippets()
    assert snippets, "README.md should contain Python snippets"
    for line_no, snippet in snippets:
        compile(snippet, f"{README}:{line_no}", "exec")


@pytest.mark.parametrize(
    "marker",
    [
        "from yirage.backends.cuda.config import CUDAArch, get_cuda_search_config",
        "from yirage.backends.cpu.config import get_cpu_search_config",
        "from yirage.hardware import HardwareRegistry",
        'chip_id="myvendor_x1"',
        "callback_demo",
        "from yirage.rl.search import",
        "COMETCostModel, COMETHardwareConfig",
        "from yirage.rl.cluster.simulator import CommunicationModel",
        "from yirage.search import",
    ],
)
def test_cpu_safe_readme_python_snippets_execute(yirage_namespace, marker):
    line_no, snippet = _snippet_containing(marker)
    globals_dict = {"__name__": "__readme_snippet__"}
    exec(compile(snippet, f"{README}:{line_no}", "exec"), globals_dict)
