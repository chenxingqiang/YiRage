# Copyright 2025 Chen Xingqiang (YiRage Project)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Hardware detection and profiling module.

Provides unified hardware abstraction for heterogeneous computing environments.

Enhanced features:
- SurrogateModel: Learned surrogate for AccelForge (Problem 5)
- CalibrationPoint: Calibration data for surrogate model
"""

from .profile import HardwareProfile, WorkloadSpec, PerformanceEstimate
from .detector import (
    HardwareDetector,
    CUDADetector,
    CPUDetector,
    MACACDetector,
    AscendDetector,
    MPSDetector,
    AccelForgeDetector,
)
from .registry import HardwareRegistry, detect_hardware, get_hardware_features
from .config_coupling import (
    ConfigGenerator,
    HardwareSearchCoupling,
    get_optimal_config,
)
from .accelforge_bridge import (
    AccelForgeBridge,
    AccelForgeDesignPoint,
    AccelForgeMetrics,
    get_accelforge_availability,
    is_accelforge_available,
)
from .surrogate_model import (
    SurrogateModel,
    CalibrationPoint,
)

__all__ = [
    # Profile
    "HardwareProfile",
    "WorkloadSpec",
    "PerformanceEstimate",
    # Detectors
    "HardwareDetector",
    "CUDADetector",
    "CPUDetector",
    "MACACDetector",
    "AscendDetector",
    "MPSDetector",
    "AccelForgeDetector",
    # Registry
    "HardwareRegistry",
    "detect_hardware",
    "get_hardware_features",
    # Config
    "ConfigGenerator",
    "HardwareSearchCoupling",
    "get_optimal_config",
    # AccelForge Bridge
    "AccelForgeBridge",
    "AccelForgeDesignPoint",
    "AccelForgeMetrics",
    "get_accelforge_availability",
    "is_accelforge_available",
    # Surrogate Model (Problem 5)
    "SurrogateModel",
    "CalibrationPoint",
]
