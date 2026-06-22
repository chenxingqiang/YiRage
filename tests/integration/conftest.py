#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""
Integration Test Configuration

Shared fixtures for integration tests.
"""

import sys
from pathlib import Path

# Add paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
PYTHON_ROOT = PROJECT_ROOT / "python"
sys.path.insert(0, str(PYTHON_ROOT))

# Re-export from main conftest
from tests.python.conftest import *
