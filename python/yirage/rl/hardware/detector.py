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
Hardware detectors for different backends.

Each detector probes the system for available hardware and returns
a HardwareProfile with detailed specifications.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
import os
import platform
import subprocess

from .profile import HardwareProfile, WorkloadSpec


class HardwareDetector(ABC):
    """Base class for hardware detectors."""

    @abstractmethod
    def detect(self, device_id: int = 0) -> Optional[HardwareProfile]:
        """
        Detect hardware and return profile.

        Args:
            device_id: Device index to query

        Returns:
            HardwareProfile if detection successful, None otherwise
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this hardware backend is available."""
        pass

    @abstractmethod
    def get_device_count(self) -> int:
        """Return number of available devices."""
        pass

    def detect_all(self) -> List[HardwareProfile]:
        """Detect all available devices of this type."""
        profiles = []
        for i in range(self.get_device_count()):
            profile = self.detect(i)
            if profile:
                profiles.append(profile)
        return profiles


class CUDADetector(HardwareDetector):
    """Detector for NVIDIA CUDA GPUs."""

    def __init__(self):
        self._torch = None
        self._pynvml = None
        self._initialized = False

    def _init(self):
        if self._initialized:
            return

        try:
            import torch

            self._torch = torch
        except ImportError:
            pass

        try:
            import pynvml

            pynvml.nvmlInit()
            self._pynvml = pynvml
        except (ImportError, Exception):
            pass

        self._initialized = True

    def is_available(self) -> bool:
        self._init()
        if self._torch is not None:
            return self._torch.cuda.is_available()
        return False

    def get_device_count(self) -> int:
        self._init()
        if self._torch is not None and self._torch.cuda.is_available():
            return self._torch.cuda.device_count()
        return 0

    def detect(self, device_id: int = 0) -> Optional[HardwareProfile]:
        self._init()

        if not self.is_available():
            return None

        if device_id >= self.get_device_count():
            return None

        torch = self._torch
        props = torch.cuda.get_device_properties(device_id)

        # Compute tensor core count based on architecture
        tensor_cores = self._estimate_tensor_cores(
            props.major, props.minor, props.multi_processor_count
        )

        # Estimate peak TFLOPS
        peak_fp32, peak_fp16 = self._estimate_peak_tflops(props)

        profile = HardwareProfile(
            backend="cuda",
            device_name=props.name,
            device_id=device_id,
            device_count=self.get_device_count(),
            driver_version=self._get_driver_version(),
            compute_capability=(props.major, props.minor),
            total_cores=props.multi_processor_count * self._cores_per_sm(props.major, props.minor),
            tensor_core_count=tensor_cores,
            warp_size=props.warp_size,
            global_memory_gb=props.total_memory / (1024**3),
            shared_memory_kb=props.max_shared_memory_per_block / 1024,
            l2_cache_mb=props.l2_cache_size / (1024**2) if hasattr(props, "l2_cache_size") else 0,
            max_threads_per_block=props.max_threads_per_block,
            max_blocks_per_sm=props.max_threads_per_multi_processor // props.max_threads_per_block,
            max_shared_memory_per_block=props.max_shared_memory_per_block,
            max_registers_per_thread=65536 // props.max_threads_per_block,
            max_grid_dim=(props.max_grid_size[0], props.max_grid_size[1], props.max_grid_size[2]),
            max_block_dim=(props.max_block_dim[0], props.max_block_dim[1], props.max_block_dim[2]),
            peak_tflops_fp32=peak_fp32,
            peak_tflops_fp16=peak_fp16,
            supports_tensor_cores=props.major >= 7,
            supports_async_copy=props.major >= 8,
            supports_cooperative_groups=props.major >= 6,
            supports_unified_memory=True,
        )

        return profile

    def _cores_per_sm(self, major: int, minor: int) -> int:
        """Get CUDA cores per SM based on compute capability."""
        cores_map = {
            (6, 0): 64,  # Pascal
            (6, 1): 128,
            (7, 0): 64,  # Volta
            (7, 5): 64,  # Turing
            (8, 0): 64,  # Ampere A100
            (8, 6): 128,  # Ampere GA10x
            (8, 9): 128,  # Ada Lovelace
            (9, 0): 128,  # Hopper
        }
        return cores_map.get((major, minor), 64)

    def _estimate_tensor_cores(self, major: int, minor: int, sm_count: int) -> int:
        """Estimate tensor core count."""
        if major < 7:
            return 0

        # Tensor cores per SM
        tc_per_sm = {
            (7, 0): 8,  # Volta
            (7, 5): 8,  # Turing
            (8, 0): 4,  # Ampere (4th gen)
            (8, 6): 4,
            (8, 9): 4,  # Ada
            (9, 0): 4,  # Hopper
        }

        return sm_count * tc_per_sm.get((major, minor), 4)

    def _estimate_peak_tflops(self, props) -> tuple:
        """Estimate peak TFLOPS for FP32 and FP16."""
        cores = props.multi_processor_count * self._cores_per_sm(props.major, props.minor)

        # Clock speed (in GHz, approximated)
        clock_ghz = 1.5  # Conservative estimate

        # FP32: 2 ops per cycle per core
        fp32_tflops = cores * clock_ghz * 2 / 1000

        # FP16: Usually 2x FP32, 4x with tensor cores
        if props.major >= 7:
            fp16_tflops = fp32_tflops * 4  # Tensor core boost
        else:
            fp16_tflops = fp32_tflops * 2

        return fp32_tflops, fp16_tflops

    def _get_driver_version(self) -> str:
        if self._pynvml:
            try:
                return self._pynvml.nvmlSystemGetDriverVersion()
            except:
                pass
        return ""


class CPUDetector(HardwareDetector):
    """Detector for CPU."""

    def is_available(self) -> bool:
        return True  # CPU is always available

    def get_device_count(self) -> int:
        return 1  # Single CPU (multi-core)

    def detect(self, device_id: int = 0) -> Optional[HardwareProfile]:
        import multiprocessing

        # Get CPU info
        cpu_count = multiprocessing.cpu_count()
        cpu_name = platform.processor() or "Unknown CPU"

        # Detect SIMD capabilities
        simd_width = self._detect_simd_width()

        # Estimate cache sizes
        l1_cache, l2_cache, l3_cache = self._detect_cache_sizes()

        # Estimate memory bandwidth
        memory_bw = self._estimate_memory_bandwidth()

        profile = HardwareProfile(
            backend="cpu",
            device_name=cpu_name,
            device_id=0,
            device_count=1,
            compute_capability=(0, 0),
            total_cores=cpu_count,
            tensor_core_count=0,
            warp_size=1,  # No warp on CPU
            global_memory_gb=self._get_system_memory_gb(),
            shared_memory_kb=l1_cache,
            l1_cache_kb=l1_cache,
            l2_cache_mb=l2_cache / 1024,
            memory_bandwidth_gbps=memory_bw,
            max_threads_per_block=cpu_count,
            max_blocks_per_sm=1,
            max_shared_memory_per_block=int(l1_cache * 1024),
            max_registers_per_thread=16,  # General purpose registers
            peak_tflops_fp32=self._estimate_cpu_tflops(cpu_count, simd_width),
            peak_tflops_fp16=self._estimate_cpu_tflops(cpu_count, simd_width) * 2,
            supports_tensor_cores=False,
            supports_unified_memory=True,
            extensions={
                "simd_width": simd_width,
                "arch": platform.machine(),
            },
        )

        return profile

    def _detect_simd_width(self) -> int:
        """Detect SIMD width (AVX-512, AVX2, SSE, NEON)."""
        try:
            # Check for AVX-512
            result = subprocess.run(
                ["grep", "-o", "avx512", "/proc/cpuinfo"], capture_output=True, text=True
            )
            if result.stdout:
                return 512

            # Check for AVX2
            result = subprocess.run(
                ["grep", "-o", "avx2", "/proc/cpuinfo"], capture_output=True, text=True
            )
            if result.stdout:
                return 256

            # Check for AVX
            result = subprocess.run(
                ["grep", "-o", "avx", "/proc/cpuinfo"], capture_output=True, text=True
            )
            if result.stdout:
                return 256

        except Exception:
            pass

        # Default to SSE (128-bit)
        return 128

    def _detect_cache_sizes(self) -> tuple:
        """Detect cache sizes in KB."""
        l1 = 32  # Default 32KB L1
        l2 = 256  # Default 256KB L2
        l3 = 8192  # Default 8MB L3

        try:
            # Try to read from /sys/devices/system/cpu
            for cache_level, default in [(1, l1), (2, l2), (3, l3)]:
                path = f"/sys/devices/system/cpu/cpu0/cache/index{cache_level}/size"
                if os.path.exists(path):
                    with open(path) as f:
                        size_str = f.read().strip()
                        if size_str.endswith("K"):
                            size = int(size_str[:-1])
                        elif size_str.endswith("M"):
                            size = int(size_str[:-1]) * 1024
                        else:
                            size = default

                        if cache_level == 1:
                            l1 = size
                        elif cache_level == 2:
                            l2 = size
                        else:
                            l3 = size
        except Exception:
            pass

        return l1, l2, l3

    def _estimate_memory_bandwidth(self) -> float:
        """Estimate memory bandwidth in GB/s."""
        # Typical DDR4/DDR5 bandwidth
        return 50.0  # Conservative estimate

    def _get_system_memory_gb(self) -> float:
        """Get total system memory in GB."""
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return kb / (1024 * 1024)
        except Exception:
            pass
        return 16.0  # Default

    def _estimate_cpu_tflops(self, cores: int, simd_width: int) -> float:
        """Estimate CPU TFLOPS."""
        # Assume 3GHz clock, 2 FMA ops per cycle
        clock_ghz = 3.0
        ops_per_cycle = simd_width // 32 * 2  # FMA = 2 ops

        tflops = cores * clock_ghz * ops_per_cycle / 1000
        return tflops


class MACACDetector(HardwareDetector):
    """Detector for MetaX MACA GPUs."""

    def is_available(self) -> bool:
        # Check for MACA environment
        maca_home = os.environ.get("MACA_HOME")
        return maca_home is not None and os.path.exists(maca_home)

    def get_device_count(self) -> int:
        if not self.is_available():
            return 0

        try:
            # Try to detect via mx-smi or similar
            result = subprocess.run(["mx-smi", "-L"], capture_output=True, text=True)
            return result.stdout.count("GPU")
        except Exception:
            pass

        return 1 if self.is_available() else 0

    def detect(self, device_id: int = 0) -> Optional[HardwareProfile]:
        if not self.is_available():
            return None

        # MACA C500 specifications (based on known data)
        profile = HardwareProfile(
            backend="maca",
            device_name="MetaX C500",
            device_id=device_id,
            device_count=self.get_device_count(),
            compute_capability=(8, 0),  # CUDA-compatible level
            total_cores=8192,  # Estimated
            tensor_core_count=512,
            warp_size=64,  # MACA uses 64-thread warps
            global_memory_gb=32.0,
            shared_memory_kb=128.0,
            memory_bandwidth_gbps=1200.0,
            max_threads_per_block=1024,
            max_blocks_per_sm=32,
            max_shared_memory_per_block=131072,
            max_registers_per_thread=255,
            peak_tflops_fp32=25.0,
            peak_tflops_fp16=100.0,
            supports_tensor_cores=True,
            supports_async_copy=True,
            extensions={
                "maca_version": os.environ.get("MACA_VERSION", "unknown"),
            },
        )

        return profile


class AscendDetector(HardwareDetector):
    """Detector for Huawei Ascend NPUs."""

    def is_available(self) -> bool:
        # Check for Ascend environment
        ascend_home = os.environ.get("ASCEND_HOME")
        if ascend_home and os.path.exists(ascend_home):
            return True

        # Check for CANN installation
        cann_path = "/usr/local/Ascend/ascend-toolkit"
        return os.path.exists(cann_path)

    def get_device_count(self) -> int:
        if not self.is_available():
            return 0

        try:
            # Try to detect via npu-smi
            result = subprocess.run(["npu-smi", "info", "-l"], capture_output=True, text=True)
            return result.stdout.count("NPU ID")
        except Exception:
            pass

        return 1 if self.is_available() else 0

    def detect(self, device_id: int = 0) -> Optional[HardwareProfile]:
        if not self.is_available():
            return None

        # Ascend 910 specifications
        profile = HardwareProfile(
            backend="ascend",
            device_name="Ascend 910",
            device_id=device_id,
            device_count=self.get_device_count(),
            compute_capability=(9, 10),  # Ascend specific
            total_cores=32,  # AI Cores
            tensor_core_count=32,  # Cube units
            warp_size=16,  # Vector unit width
            global_memory_gb=32.0,
            shared_memory_kb=1024.0,  # L1 buffer
            l2_cache_mb=32.0,
            memory_bandwidth_gbps=1200.0,
            max_threads_per_block=32,
            max_blocks_per_sm=1,
            max_shared_memory_per_block=1048576,
            peak_tflops_fp32=160.0,
            peak_tflops_fp16=320.0,
            supports_tensor_cores=True,  # Cube Unit
            supports_async_copy=True,
            extensions={
                "cann_version": self._get_cann_version(),
                "ai_core_count": 32,
            },
        )

        return profile

    def _get_cann_version(self) -> str:
        try:
            result = subprocess.run(
                ["cat", "/usr/local/Ascend/ascend-toolkit/latest/version.info"],
                capture_output=True,
                text=True,
            )
            return result.stdout.strip()
        except Exception:
            return "unknown"


class MPSDetector(HardwareDetector):
    """Detector for Apple Metal Performance Shaders (MPS)."""

    def is_available(self) -> bool:
        if platform.system() != "Darwin":
            return False

        try:
            import torch

            return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        except ImportError:
            pass

        return False

    def get_device_count(self) -> int:
        return 1 if self.is_available() else 0

    def detect(self, device_id: int = 0) -> Optional[HardwareProfile]:
        if not self.is_available():
            return None

        # Detect Apple Silicon model
        device_name = self._detect_apple_silicon()
        specs = self._get_specs(device_name)

        profile = HardwareProfile(
            backend="mps",
            device_name=device_name,
            device_id=0,
            device_count=1,
            compute_capability=(1, 0),  # MPS version
            total_cores=specs["gpu_cores"],
            tensor_core_count=0,  # No tensor cores on Apple Silicon
            warp_size=32,  # Metal SIMD-group size
            global_memory_gb=specs["memory_gb"],
            shared_memory_kb=32.0,
            memory_bandwidth_gbps=specs["memory_bw"],
            max_threads_per_block=1024,
            max_blocks_per_sm=32,
            max_shared_memory_per_block=32768,
            peak_tflops_fp32=specs["tflops_fp32"],
            peak_tflops_fp16=specs["tflops_fp16"],
            supports_tensor_cores=False,
            supports_unified_memory=True,  # Unified memory architecture
            extensions={
                "metal_version": "3.0",
                "apple_silicon": True,
            },
        )

        return profile

    def _detect_apple_silicon(self) -> str:
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
            )
            return result.stdout.strip()
        except Exception:
            return "Apple Silicon"

    def _get_specs(self, device_name: str) -> Dict[str, Any]:
        """Get specs based on Apple Silicon model."""
        specs = {
            "M1": {
                "gpu_cores": 8,
                "memory_gb": 8,
                "memory_bw": 68.25,
                "tflops_fp32": 2.6,
                "tflops_fp16": 5.2,
            },
            "M1 Pro": {
                "gpu_cores": 16,
                "memory_gb": 16,
                "memory_bw": 200,
                "tflops_fp32": 5.2,
                "tflops_fp16": 10.4,
            },
            "M1 Max": {
                "gpu_cores": 32,
                "memory_gb": 32,
                "memory_bw": 400,
                "tflops_fp32": 10.4,
                "tflops_fp16": 20.8,
            },
            "M2": {
                "gpu_cores": 10,
                "memory_gb": 8,
                "memory_bw": 100,
                "tflops_fp32": 3.6,
                "tflops_fp16": 7.2,
            },
            "M2 Pro": {
                "gpu_cores": 19,
                "memory_gb": 16,
                "memory_bw": 200,
                "tflops_fp32": 6.8,
                "tflops_fp16": 13.6,
            },
            "M2 Max": {
                "gpu_cores": 38,
                "memory_gb": 32,
                "memory_bw": 400,
                "tflops_fp32": 13.6,
                "tflops_fp16": 27.2,
            },
            "M3": {
                "gpu_cores": 10,
                "memory_gb": 8,
                "memory_bw": 100,
                "tflops_fp32": 4.0,
                "tflops_fp16": 8.0,
            },
            "M3 Pro": {
                "gpu_cores": 18,
                "memory_gb": 18,
                "memory_bw": 150,
                "tflops_fp32": 7.2,
                "tflops_fp16": 14.4,
            },
            "M3 Max": {
                "gpu_cores": 40,
                "memory_gb": 48,
                "memory_bw": 400,
                "tflops_fp32": 16.0,
                "tflops_fp16": 32.0,
            },
        }

        for model, spec in specs.items():
            if model in device_name:
                return spec

        # Default specs
        return {
            "gpu_cores": 8,
            "memory_gb": 8,
            "memory_bw": 68.25,
            "tflops_fp32": 2.6,
            "tflops_fp16": 5.2,
        }


class AccelForgeDetector(HardwareDetector):
    """
    Detector for AccelForge-modeled virtual accelerators.

    Unlike physical hardware detectors, this creates hardware profiles
    from AccelForge design specifications. Used for hardware-software
    co-design where the accelerator architecture itself is being optimized.
    """

    def __init__(self, design_point: Optional[Dict[str, Any]] = None):
        """
        Args:
            design_point: AccelForge design parameters.
                If None, uses a default design.
        """
        self._design_point = design_point

    def is_available(self) -> bool:
        """AccelForge detector is always available (uses analytical model if needed)."""
        return True

    def get_device_count(self) -> int:
        return 1

    def detect(self, device_id: int = 0) -> Optional[HardwareProfile]:
        from .accelforge_bridge import AccelForgeBridge, AccelForgeDesignPoint

        bridge = AccelForgeBridge()

        if self._design_point:
            design = AccelForgeDesignPoint.from_dict(self._design_point)
        else:
            design = AccelForgeDesignPoint()  # Default design

        return bridge.to_hardware_profile(design)


# Factory function
def get_detector(backend: str) -> Optional[HardwareDetector]:
    """Get detector for specified backend."""
    detectors = {
        "cuda": CUDADetector,
        "maca": MACACDetector,
        "ascend": AscendDetector,
        "cpu": CPUDetector,
        "mps": MPSDetector,
        "accelforge": AccelForgeDetector,
    }

    detector_class = detectors.get(backend)
    if detector_class:
        return detector_class()
    return None
