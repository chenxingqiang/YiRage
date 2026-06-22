# Copyright 2025 YiRage Project
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for ``[project.scripts]`` entries in ``pyproject.toml``.

The package previously declared a ``yirage`` console script pointing at
``yirage.cli:main``, but no such module existed.  Installing the wheel created
a broken ``yirage`` command that always failed with ``ModuleNotFoundError``.

These tests validate that any console script declared today references a real
module path and a real attribute, so the same class of bug cannot regress.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYPROJECT = _REPO_ROOT / "pyproject.toml"
_PYTHON_SRC = _REPO_ROOT / "python"


def _load_project_scripts() -> dict[str, str]:
    with _PYPROJECT.open("rb") as fh:
        data = tomllib.load(fh)
    return dict(data.get("project", {}).get("scripts", {}) or {})


def test_pyproject_is_loadable():
    """pyproject.toml must be syntactically valid TOML for build backends."""
    assert _load_project_scripts() is not None


@pytest.mark.parametrize("name,target", list(_load_project_scripts().items()))
def test_project_script_target_resolves(name: str, target: str) -> None:
    """Every declared console script must reference a real module:attr path.

    We resolve the module to its on-disk source under ``python/`` (rather than
    importing it) so this test does not require the native ``yirage.core``
    runtime to be installed.
    """
    assert ":" in target, f"script {name!r} must use 'module:attr' form, got {target!r}"
    module_path, _, attr = target.partition(":")
    assert module_path, f"script {name!r} has empty module path: {target!r}"
    assert attr, f"script {name!r} has empty attribute: {target!r}"

    parts = module_path.split(".")
    module_file = _PYTHON_SRC.joinpath(*parts).with_suffix(".py")
    package_init = _PYTHON_SRC.joinpath(*parts, "__init__.py")
    assert module_file.is_file() or package_init.is_file(), (
        f"script {name!r} points at {target!r} but neither "
        f"{module_file.relative_to(_REPO_ROOT)} nor "
        f"{package_init.relative_to(_REPO_ROOT)} exists"
    )

    source_path = module_file if module_file.is_file() else package_init
    source = source_path.read_text(encoding="utf-8")
    # Cheap textual check that avoids importing the package: the attribute must
    # appear as a top-level definition or assignment in the target module.
    needles = (f"def {attr}(", f"async def {attr}(", f"{attr} =", f"class {attr}")
    assert any(n in source for n in needles), (
        f"script {name!r} expects attribute {attr!r} in "
        f"{source_path.relative_to(_REPO_ROOT)}, but no top-level definition was found"
    )


def test_pyproject_has_no_dangling_yirage_cli_reference():
    """Guard against the specific regression: ``yirage.cli`` does not exist."""
    scripts = _load_project_scripts()
    for name, target in scripts.items():
        assert not target.startswith("yirage.cli:"), (
            f"script {name!r} still points at non-existent module 'yirage.cli' "
            f"(target={target!r}); add python/yirage/cli.py before re-declaring it."
        )


# Keep the unused-import warning quiet on older interpreters where the fallback
# import path might not be exercised.
del sys
