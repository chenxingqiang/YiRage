"""Contract tests for MACA persistent-kernel smem + mxcc compile path."""

from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]


def test_maca_pk_backend_uses_64kb_smem_not_volta_96kb():
    text = (_REPO / "src/persistent_kernel/maca_pk_backend.cc").read_text(encoding="utf-8")
    assert "maca::MAX_SMEM_SIZE" in text
    assert "return 96 * 1024" not in text


def test_persistent_kernel_compile_has_mxcc_path():
    text = (_REPO / "python/yirage/persistent_kernel/kernel.py").read_text(encoding="utf-8")
    assert "get_maca_pk_compile_command" in text
    assert "_resolve_persistent_kernel_compiler" in text
    assert "YIRAGE_BACKEND_MACA_ENABLED" in text
    assert "compiler_kind == \"mxcc\"" in text


def test_maca_rebuild_core_script_checks_pk_smem_source():
    text = (_REPO / "scripts/maca_rebuild_core.sh").read_text(encoding="utf-8")
    assert "maca_pk_backend.cc" in text
    assert "96 * 1024" in text or "96 \\* 1024" in text
