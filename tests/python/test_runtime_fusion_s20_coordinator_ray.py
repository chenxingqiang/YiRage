# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S20: Ray cluster coordinator e2e tier for serving down matmul."""

from __future__ import annotations

import os

import pytest

from serving_test_utils import serving


pytestmark = pytest.mark.skipif(
    os.environ.get("YIRAGE_SKIP_NATIVE") == "1",
    reason="native yirage.core not built",
)


def test_runtime_fusion_version_s20(serving):
    assert serving.RuntimeFusion([]).inspect()["version"] == "s21"


def test_s20_integration_module_exists():
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[1]
        / "integration"
        / "test_serving_coordinator_ray_e2e.py"
    )
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert "test_serving_coordinator_ray_down_matmul" in text
    assert "test_serving_superoptimize_ray_blockdim_partition" in text


def test_cpu_cert_manifest_s20_contract():
    from yirage.serving.cpu_cert import serving_cpu_cert_manifest

    names = [s.name for s in serving_cpu_cert_manifest(quick=True)]
    assert "s20_contract" in names
