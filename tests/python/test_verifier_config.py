# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for search verification configuration."""

from __future__ import annotations

import os
import warnings

import pytest

from yirage.search.verifier_config import (
    env_truthy,
    resolve_verifier_config,
)


def test_env_truthy():
    assert env_truthy("YIRAGE_TEST_TRUTHY", default=False) is False
    os.environ["YIRAGE_TEST_TRUTHY"] = "yes"
    try:
        assert env_truthy("YIRAGE_TEST_TRUTHY") is True
        os.environ["YIRAGE_TEST_TRUTHY"] = "0"
        assert env_truthy("YIRAGE_TEST_TRUTHY") is False
    finally:
        os.environ.pop("YIRAGE_TEST_TRUTHY", None)


def test_resolve_defaults_to_probabilistic(monkeypatch):
    monkeypatch.delenv("YIRAGE_FORMAL_VERIFY", raising=False)
    cfg = resolve_verifier_config(warn_on_fallback=False)
    assert cfg.verifier_type == "probabilistic"
    assert cfg.is_formal_verified is False


def test_resolve_env_formal_falls_back_when_unbuilt(monkeypatch):
    monkeypatch.setenv("YIRAGE_FORMAL_VERIFY", "1")
    monkeypatch.setattr(
        "yirage.search.verifier_config.is_formal_verifier_built",
        lambda: False,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = resolve_verifier_config()
    assert cfg.verifier_type == "probabilistic"
    assert any("Falling back" in str(w.message) for w in caught)


def test_resolve_explicit_formal_when_built(monkeypatch):
    monkeypatch.delenv("YIRAGE_FORMAL_VERIFY", raising=False)
    monkeypatch.setattr(
        "yirage.search.verifier_config.is_formal_verifier_built",
        lambda: True,
    )
    cfg = resolve_verifier_config(formal_verify=True, warn_on_fallback=False)
    assert cfg.verifier_type == "formal"
    assert cfg.is_formal_verified is True


def test_is_formal_verified_kwarg_overrides_env(monkeypatch):
    monkeypatch.setenv("YIRAGE_FORMAL_VERIFY", "1")
    monkeypatch.setattr(
        "yirage.search.verifier_config.is_formal_verifier_built",
        lambda: True,
    )
    cfg = resolve_verifier_config(is_formal_verified=False, warn_on_fallback=False)
    assert cfg.verifier_type == "probabilistic"
