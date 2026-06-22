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
Hardware registry for managing detected hardware profiles.

Provides caching and unified access to hardware information.
"""

from typing import Optional, List, Dict, Any
import threading
import json
import os
import numpy as np

from .profile import HardwareProfile
from .detector import (
    HardwareDetector,
    CUDADetector,
    CPUDetector,
    MACACDetector,
    AscendDetector,
    MPSDetector,
    AccelForgeDetector,
    get_detector,
)


class HardwareRegistry:
    """
    Singleton registry for hardware profiles.

    Detects available hardware on first access and caches results.
    Thread-safe for multi-threaded applications.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._profiles: Dict[str, List[HardwareProfile]] = {}
        self._primary_backend: Optional[str] = None
        self._detected = False
        self._cache_file = os.path.expanduser("~/.cache/yirage/hardware_cache.json")
        self._initialized = True

    def detect_all(self, use_cache: bool = True) -> Dict[str, List[HardwareProfile]]:
        """
        Detect all available hardware.

        Args:
            use_cache: Whether to use cached results if available

        Returns:
            Dictionary mapping backend name to list of profiles
        """
        if self._detected and use_cache:
            return self._profiles

        # Try to load from cache
        if use_cache and self._load_cache():
            self._detected = True
            return self._profiles

        # Detect all backends
        backends = ["cuda", "maca", "ascend", "mps", "cpu"]

        for backend in backends:
            detector = get_detector(backend)
            if detector and detector.is_available():
                profiles = detector.detect_all()
                if profiles:
                    self._profiles[backend] = profiles

                    # Set primary backend (first GPU-like backend found)
                    if self._primary_backend is None and backend != "cpu":
                        self._primary_backend = backend

        # CPU is always fallback
        if "cpu" not in self._profiles:
            cpu_detector = CPUDetector()
            self._profiles["cpu"] = cpu_detector.detect_all()

        if self._primary_backend is None:
            self._primary_backend = "cpu"

        self._detected = True

        # Save to cache
        self._save_cache()

        return self._profiles

    def get_primary(self) -> HardwareProfile:
        """
        Get primary (best) available hardware profile.

        Priority: CUDA > MACA > Ascend > MPS > CPU
        """
        if not self._detected:
            self.detect_all()

        if self._primary_backend and self._primary_backend in self._profiles:
            return self._profiles[self._primary_backend][0]

        return self._profiles.get("cpu", [HardwareProfile()])[0]

    def get_by_backend(self, backend: str) -> Optional[List[HardwareProfile]]:
        """Get all profiles for a specific backend."""
        if not self._detected:
            self.detect_all()

        return self._profiles.get(backend)

    def get_all_profiles(self) -> List[HardwareProfile]:
        """Get all detected hardware profiles."""
        if not self._detected:
            self.detect_all()

        all_profiles = []
        for profiles in self._profiles.values():
            all_profiles.extend(profiles)
        return all_profiles

    def get_available_backends(self) -> List[str]:
        """Get list of available backends."""
        if not self._detected:
            self.detect_all()

        return list(self._profiles.keys())

    def _load_cache(self) -> bool:
        """Load cached hardware profiles."""
        try:
            if os.path.exists(self._cache_file):
                with open(self._cache_file) as f:
                    data = json.load(f)

                for backend, profiles_data in data.get("profiles", {}).items():
                    self._profiles[backend] = [HardwareProfile.from_dict(p) for p in profiles_data]

                self._primary_backend = data.get("primary_backend")
                return True
        except Exception:
            pass

        return False

    def _save_cache(self):
        """Save hardware profiles to cache."""
        try:
            os.makedirs(os.path.dirname(self._cache_file), exist_ok=True)

            data = {
                "profiles": {
                    backend: [p.to_dict() for p in profiles]
                    for backend, profiles in self._profiles.items()
                },
                "primary_backend": self._primary_backend,
            }

            with open(self._cache_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def clear_cache(self):
        """Clear cached profiles."""
        self._profiles.clear()
        self._primary_backend = None
        self._detected = False

        try:
            if os.path.exists(self._cache_file):
                os.remove(self._cache_file)
        except Exception:
            pass


# Convenience functions


def detect_hardware(backend: Optional[str] = None) -> HardwareProfile:
    """
    Detect available hardware and return profile.

    Args:
        backend: Specific backend to detect, or None for best available

    Returns:
        HardwareProfile for detected hardware
    """
    registry = HardwareRegistry()

    if backend:
        profiles = registry.get_by_backend(backend)
        if profiles:
            return profiles[0]

    return registry.get_primary()


def get_hardware_features(backend: Optional[str] = None) -> np.ndarray:
    """
    Get hardware features as numpy array for RL model input.

    Args:
        backend: Specific backend, or None for best available

    Returns:
        Numpy array of shape (32,) with normalized hardware features
    """
    profile = detect_hardware(backend)
    return profile.to_feature_vector()


def get_all_hardware() -> List[HardwareProfile]:
    """Get all detected hardware profiles."""
    registry = HardwareRegistry()
    return registry.get_all_profiles()


def get_available_backends() -> List[str]:
    """Get list of available hardware backends."""
    registry = HardwareRegistry()
    return registry.get_available_backends()
