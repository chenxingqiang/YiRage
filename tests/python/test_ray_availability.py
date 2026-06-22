# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for Ray availability detection (Ray 2.55+)."""

import pytest


def test_yirage_top_level_ray_available():
    import yirage

    assert yirage.is_ray_available() is True


def test_yirage_ray_subpackage_available():
    from yirage import ray as yirage_ray

    assert yirage_ray.is_ray_available() is True


def test_ray_distributed_module_ray_flag():
    from yirage.ray import ray_distributed as rd

    assert rd.RAY_AVAILABLE is True
