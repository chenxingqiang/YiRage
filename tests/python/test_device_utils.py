# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0

import pytest

from demo import _device_utils as du


def test_resolve_device_cpu():
    assert du.resolve_device("cpu") == "cpu"


def test_backend_for_device_mapping():
    assert du.backend_for_device("cpu") == "cpu"
    assert du.backend_for_device("mps") == "mps"
    assert du.backend_for_device("cuda:0") == "cuda"


def test_require_mps_exits_when_unavailable(monkeypatch):
    monkeypatch.setattr(du, "mps_available", lambda: False)
    with pytest.raises(SystemExit) as exc:
        du.require_mps()
    assert exc.value.code == 1
