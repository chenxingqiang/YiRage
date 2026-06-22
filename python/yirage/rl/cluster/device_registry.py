"""
Device Registry

Pre-defined device specifications for common hardware.
Supports all PyTorch-compatible devices plus custom hardware.
"""

from dataclasses import dataclass
from typing import Dict, Optional, List
from .topology import DeviceSpec, DeviceType


# ============================================================================
# Pre-defined Device Specifications
# ============================================================================

DEVICE_SPECS = {
    # =========================================================================
    # NVIDIA GPUs (CUDA)
    # =========================================================================
    # Hopper Architecture
    "H100_SXM": DeviceSpec(
        device_id="h100_sxm",
        device_type=DeviceType.CUDA,
        compute_units=132,
        clock_mhz=1830,
        peak_tflops_fp16=989.0,
        peak_tflops_fp32=67.0,
        peak_tflops_bf16=989.0,
        peak_tflops_int8=1979.0,
        peak_tflops_fp64=33.5,
        memory_gb=80.0,
        memory_bandwidth_gbps=3350.0,
        tensor_cores=True,
        supports_bf16=True,
        supports_fp8=True,
        supports_sparsity=True,
        compute_capability="9.0",
        architecture="Hopper",
        tdp_watts=700.0,
    ),
    "H100_PCIe": DeviceSpec(
        device_id="h100_pcie",
        device_type=DeviceType.CUDA,
        compute_units=114,
        clock_mhz=1620,
        peak_tflops_fp16=756.0,
        peak_tflops_fp32=51.0,
        peak_tflops_bf16=756.0,
        peak_tflops_int8=1513.0,
        memory_gb=80.0,
        memory_bandwidth_gbps=2000.0,
        tensor_cores=True,
        supports_bf16=True,
        supports_fp8=True,
        compute_capability="9.0",
        architecture="Hopper",
        tdp_watts=350.0,
    ),
    # Ampere Architecture
    "A100_SXM": DeviceSpec(
        device_id="a100_sxm",
        device_type=DeviceType.CUDA,
        compute_units=108,
        clock_mhz=1410,
        peak_tflops_fp16=312.0,
        peak_tflops_fp32=19.5,
        peak_tflops_bf16=312.0,
        peak_tflops_int8=624.0,
        peak_tflops_fp64=9.7,
        memory_gb=80.0,
        memory_bandwidth_gbps=2039.0,
        tensor_cores=True,
        supports_bf16=True,
        compute_capability="8.0",
        architecture="Ampere",
        tdp_watts=400.0,
    ),
    "A100": DeviceSpec(
        device_id="a100",
        device_type=DeviceType.CUDA,
        compute_units=108,
        clock_mhz=1410,
        peak_tflops_fp16=312.0,
        peak_tflops_fp32=19.5,
        peak_tflops_bf16=312.0,
        peak_tflops_int8=624.0,
        memory_gb=80.0,
        memory_bandwidth_gbps=2039.0,
        tensor_cores=True,
        supports_bf16=True,
        compute_capability="8.0",
        architecture="Ampere",
        tdp_watts=400.0,
    ),
    "A100_40GB": DeviceSpec(
        device_id="a100_40gb",
        device_type=DeviceType.CUDA,
        compute_units=108,
        clock_mhz=1410,
        peak_tflops_fp16=312.0,
        peak_tflops_fp32=19.5,
        memory_gb=40.0,
        memory_bandwidth_gbps=1555.0,
        tensor_cores=True,
        supports_bf16=True,
        compute_capability="8.0",
        architecture="Ampere",
        tdp_watts=400.0,
    ),
    # Volta Architecture
    "V100_SXM2": DeviceSpec(
        device_id="v100_sxm2",
        device_type=DeviceType.CUDA,
        compute_units=80,
        clock_mhz=1530,
        peak_tflops_fp16=125.0,
        peak_tflops_fp32=15.7,
        peak_tflops_fp64=7.8,
        memory_gb=32.0,
        memory_bandwidth_gbps=900.0,
        tensor_cores=True,
        compute_capability="7.0",
        architecture="Volta",
        tdp_watts=300.0,
    ),
    "V100": DeviceSpec(
        device_id="v100",
        device_type=DeviceType.CUDA,
        compute_units=80,
        clock_mhz=1380,
        peak_tflops_fp16=125.0,
        peak_tflops_fp32=15.7,
        memory_gb=32.0,
        memory_bandwidth_gbps=900.0,
        tensor_cores=True,
        compute_capability="7.0",
        architecture="Volta",
        tdp_watts=300.0,
    ),
    # Consumer GPUs
    "RTX4090": DeviceSpec(
        device_id="rtx4090",
        device_type=DeviceType.CUDA,
        compute_units=128,
        clock_mhz=2520,
        peak_tflops_fp16=330.0,
        peak_tflops_fp32=82.6,
        peak_tflops_int8=660.0,
        memory_gb=24.0,
        memory_bandwidth_gbps=1008.0,
        tensor_cores=True,
        supports_bf16=True,
        supports_fp8=True,
        compute_capability="8.9",
        architecture="Ada Lovelace",
        tdp_watts=450.0,
    ),
    "RTX3090": DeviceSpec(
        device_id="rtx3090",
        device_type=DeviceType.CUDA,
        compute_units=82,
        clock_mhz=1695,
        peak_tflops_fp16=142.0,
        peak_tflops_fp32=35.6,
        memory_gb=24.0,
        memory_bandwidth_gbps=936.0,
        tensor_cores=True,
        supports_bf16=True,
        compute_capability="8.6",
        architecture="Ampere",
        tdp_watts=350.0,
    ),
    # =========================================================================
    # AMD GPUs (ROCm)
    # =========================================================================
    "MI300X": DeviceSpec(
        device_id="mi300x",
        device_type=DeviceType.ROCM,
        compute_units=304,
        clock_mhz=2100,
        peak_tflops_fp16=1307.0,
        peak_tflops_fp32=653.0,
        peak_tflops_bf16=1307.0,
        peak_tflops_int8=2614.0,
        peak_tflops_fp64=163.0,
        memory_gb=192.0,
        memory_bandwidth_gbps=5300.0,
        matrix_units=True,
        supports_bf16=True,
        supports_fp8=True,
        architecture="CDNA3",
        tdp_watts=750.0,
    ),
    "MI250X": DeviceSpec(
        device_id="mi250x",
        device_type=DeviceType.ROCM,
        compute_units=220,
        clock_mhz=1700,
        peak_tflops_fp16=383.0,
        peak_tflops_fp32=47.9,
        peak_tflops_bf16=383.0,
        peak_tflops_fp64=47.9,
        memory_gb=128.0,
        memory_bandwidth_gbps=3276.0,
        matrix_units=True,
        supports_bf16=True,
        architecture="CDNA2",
        tdp_watts=560.0,
    ),
    # =========================================================================
    # Intel GPUs (XPU)
    # =========================================================================
    "Max_1550": DeviceSpec(
        device_id="max_1550",
        device_type=DeviceType.XPU,
        compute_units=128,
        clock_mhz=1600,
        peak_tflops_fp16=839.0,
        peak_tflops_fp32=52.0,
        peak_tflops_bf16=839.0,
        peak_tflops_int8=1678.0,
        memory_gb=128.0,
        memory_bandwidth_gbps=3276.0,
        matrix_units=True,
        supports_bf16=True,
        architecture="Ponte Vecchio",
        tdp_watts=600.0,
    ),
    # =========================================================================
    # Google TPU
    # =========================================================================
    "TPUv4": DeviceSpec(
        device_id="tpuv4",
        device_type=DeviceType.TPU,
        compute_units=2,  # 2 TensorCores per chip
        clock_mhz=1050,
        peak_tflops_fp16=275.0,
        peak_tflops_fp32=275.0,
        peak_tflops_bf16=275.0,
        peak_tflops_int8=550.0,
        memory_gb=32.0,
        memory_bandwidth_gbps=1200.0,
        supports_bf16=True,
        architecture="TPU v4",
        tdp_watts=192.0,
    ),
    "TPUv5e": DeviceSpec(
        device_id="tpuv5e",
        device_type=DeviceType.TPU,
        compute_units=2,
        clock_mhz=1050,
        peak_tflops_fp16=197.0,
        peak_tflops_fp32=197.0,
        peak_tflops_bf16=197.0,
        peak_tflops_int8=394.0,
        memory_gb=16.0,
        memory_bandwidth_gbps=820.0,
        supports_bf16=True,
        supports_int4=True,
        architecture="TPU v5e",
        tdp_watts=100.0,
    ),
    # =========================================================================
    # Huawei Ascend NPU
    # =========================================================================
    "Ascend910B": DeviceSpec(
        device_id="ascend910b",
        device_type=DeviceType.ASCEND,
        compute_units=32,  # AI Cores
        clock_mhz=1800,
        peak_tflops_fp16=320.0,
        peak_tflops_fp32=160.0,
        peak_tflops_bf16=320.0,
        peak_tflops_int8=640.0,
        memory_gb=64.0,
        memory_bandwidth_gbps=1200.0,
        supports_bf16=True,
        architecture="Da Vinci",
        tdp_watts=310.0,
    ),
    "Ascend910": DeviceSpec(
        device_id="ascend910",
        device_type=DeviceType.ASCEND,
        compute_units=32,
        clock_mhz=1500,
        peak_tflops_fp16=256.0,
        peak_tflops_fp32=128.0,
        peak_tflops_int8=512.0,
        memory_gb=32.0,
        memory_bandwidth_gbps=1200.0,
        architecture="Da Vinci",
        tdp_watts=310.0,
    ),
    "Ascend310": DeviceSpec(
        device_id="ascend310",
        device_type=DeviceType.ASCEND,
        compute_units=8,
        clock_mhz=1000,
        peak_tflops_fp16=16.0,
        peak_tflops_fp32=8.0,
        peak_tflops_int8=32.0,
        memory_gb=8.0,
        memory_bandwidth_gbps=100.0,
        architecture="Da Vinci",
        tdp_watts=8.0,
    ),
    # =========================================================================
    # AWS Neuron (Inferentia/Trainium)
    # =========================================================================
    "Trainium2": DeviceSpec(
        device_id="trainium2",
        device_type=DeviceType.NEURON,
        compute_units=2,
        clock_mhz=1400,
        peak_tflops_fp16=380.0,
        peak_tflops_fp32=190.0,
        peak_tflops_bf16=380.0,
        memory_gb=96.0,
        memory_bandwidth_gbps=1638.0,
        supports_bf16=True,
        architecture="Trainium2",
        tdp_watts=175.0,
    ),
    "Inferentia2": DeviceSpec(
        device_id="inferentia2",
        device_type=DeviceType.NEURON,
        compute_units=2,
        clock_mhz=1400,
        peak_tflops_fp16=190.0,
        peak_tflops_fp32=95.0,
        peak_tflops_bf16=190.0,
        peak_tflops_int8=380.0,
        memory_gb=32.0,
        memory_bandwidth_gbps=820.0,
        supports_bf16=True,
        architecture="Inferentia2",
        tdp_watts=75.0,
    ),
    # =========================================================================
    # MetaX MACA
    # =========================================================================
    "C500": DeviceSpec(
        device_id="c500",
        device_type=DeviceType.MACA,
        compute_units=64,
        clock_mhz=1800,
        peak_tflops_fp16=200.0,
        peak_tflops_fp32=50.0,
        peak_tflops_bf16=200.0,
        peak_tflops_int8=400.0,
        memory_gb=64.0,
        memory_bandwidth_gbps=1600.0,
        supports_bf16=True,
        architecture="MACA",
        tdp_watts=300.0,
    ),
    # =========================================================================
    # Apple Silicon (MPS)
    # =========================================================================
    "M2_Ultra": DeviceSpec(
        device_id="m2_ultra",
        device_type=DeviceType.MPS,
        compute_units=76,  # GPU cores
        clock_mhz=1398,
        peak_tflops_fp16=27.2,
        peak_tflops_fp32=13.6,
        memory_gb=192.0,  # Unified memory
        memory_bandwidth_gbps=800.0,
        supports_bf16=True,
        architecture="Apple M2 Ultra",
        tdp_watts=60.0,
    ),
    "M3_Max": DeviceSpec(
        device_id="m3_max",
        device_type=DeviceType.MPS,
        compute_units=40,
        clock_mhz=1398,
        peak_tflops_fp16=14.2,
        peak_tflops_fp32=7.1,
        memory_gb=128.0,
        memory_bandwidth_gbps=400.0,
        supports_bf16=True,
        architecture="Apple M3 Max",
        tdp_watts=30.0,
    ),
    # =========================================================================
    # CPUs
    # =========================================================================
    "EPYC_9654": DeviceSpec(
        device_id="epyc_9654",
        device_type=DeviceType.CPU,
        compute_units=96,  # cores
        clock_mhz=3700,
        peak_tflops_fp16=2.0,
        peak_tflops_fp32=3.0,
        peak_tflops_int8=4.0,
        memory_gb=768.0,
        memory_bandwidth_gbps=460.0,
        supports_bf16=True,
        architecture="AMD Zen 4",
        tdp_watts=360.0,
    ),
    "Xeon_8480": DeviceSpec(
        device_id="xeon_8480",
        device_type=DeviceType.CPU,
        compute_units=56,
        clock_mhz=3800,
        peak_tflops_fp16=1.5,
        peak_tflops_fp32=2.0,
        peak_tflops_int8=3.0,
        memory_gb=512.0,
        memory_bandwidth_gbps=307.0,
        matrix_units=True,  # AMX
        supports_bf16=True,
        architecture="Intel Sapphire Rapids",
        tdp_watts=350.0,
    ),
    # =========================================================================
    # FPGAs
    # =========================================================================
    "Alveo_U280": DeviceSpec(
        device_id="alveo_u280",
        device_type=DeviceType.FPGA,
        compute_units=1,
        clock_mhz=300,
        peak_tflops_fp16=2.0,
        peak_tflops_fp32=1.0,
        peak_tflops_int8=4.0,
        memory_gb=8.0,
        memory_bandwidth_gbps=460.0,
        supports_int4=True,
        architecture="Xilinx Alveo",
        tdp_watts=225.0,
    ),
}


class DeviceRegistry:
    """
    Registry for device specifications.

    Supports:
    - Pre-defined device specs
    - Custom user-defined devices
    - Auto-detection from PyTorch
    """

    _instance = None
    _custom_devices: Dict[str, DeviceSpec] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_device_spec(cls, device_name: str) -> Optional[DeviceSpec]:
        """
        Get device specification by name.

        Args:
            device_name: Device name (e.g., "A100", "H100_SXM", "MI300X")

        Returns:
            DeviceSpec if found, None otherwise
        """
        # Exact match first
        if device_name in DEVICE_SPECS:
            return DEVICE_SPECS[device_name]

        # Case-insensitive match
        name_lower = device_name.lower().replace("-", "_").replace(" ", "_")
        for key, spec in DEVICE_SPECS.items():
            if key.lower() == name_lower:
                return spec

        # Check custom devices (case-insensitive)
        if device_name in cls._custom_devices:
            return cls._custom_devices[device_name]
        for key, spec in cls._custom_devices.items():
            if key.lower() == name_lower:
                return spec

        # Try common aliases
        aliases = {
            "a100": "A100",
            "a100-80gb": "A100",
            "a100_80gb": "A100",
            "a100-40gb": "A100_40GB",
            "a100_40gb": "A100_40GB",
            "h100": "H100_SXM",
            "v100": "V100",
            "rtx4090": "RTX4090",
            "rtx 4090": "RTX4090",
            "rtx_4090": "RTX4090",
            "rtx3090": "RTX3090",
            "rtx 3090": "RTX3090",
            "rtx_3090": "RTX3090",
            "mi300": "MI300X",
            "mi300x": "MI300X",
            "mi250": "MI250X",
            "mi250x": "MI250X",
            "910b": "Ascend910B",
            "ascend_910b": "Ascend910B",
            "910": "Ascend910",
            "ascend_910": "Ascend910",
            "tpu_v4": "TPUv4",
            "tpu v4": "TPUv4",
            "tpuv4": "TPUv4",
            "tpu_v5": "TPUv5e",
            "tpu v5": "TPUv5e",
            "tpuv5": "TPUv5e",
            "tpuv5e": "TPUv5e",
            "c500": "C500",
            "maca_c500": "C500",
            "m2_ultra": "M2_Ultra",
            "m2ultra": "M2_Ultra",
            "m3_max": "M3_Max",
            "m3max": "M3_Max",
            "epyc": "EPYC_9654",
            "xeon": "Xeon_8480",
            "alveo": "Alveo_U280",
            "alveo_u280": "Alveo_U280",
            "trainium": "Trainium2",
            "trainium2": "Trainium2",
            "inferentia": "Inferentia2",
            "inferentia2": "Inferentia2",
            "max_1550": "Max_1550",
            "intel_max": "Max_1550",
        }

        if name_lower in aliases:
            return DEVICE_SPECS.get(aliases[name_lower])

        return None

    @classmethod
    def register_custom_device(cls, name: str, spec: DeviceSpec):
        """
        Register a custom device specification.

        Args:
            name: Device name
            spec: Device specification
        """
        cls._custom_devices[name] = spec

    @classmethod
    def register_custom_device_from_dict(cls, name: str, spec_dict: Dict):
        """
        Register a custom device from dictionary.

        Args:
            name: Device name
            spec_dict: Device specification as dictionary
        """
        spec = DeviceSpec.from_dict(spec_dict)
        spec.device_id = name
        cls._custom_devices[name] = spec

    @classmethod
    def list_devices(cls) -> List[str]:
        """List all available device names."""
        return list(DEVICE_SPECS.keys()) + list(cls._custom_devices.keys())

    @classmethod
    def list_by_type(cls, device_type: DeviceType) -> List[str]:
        """List devices by type."""
        result = []
        for name, spec in DEVICE_SPECS.items():
            if spec.device_type == device_type:
                result.append(name)
        for name, spec in cls._custom_devices.items():
            if spec.device_type == device_type:
                result.append(name)
        return result

    @classmethod
    def detect_pytorch_devices(cls) -> List[DeviceSpec]:
        """Detect devices from PyTorch."""
        specs = []

        try:
            import torch

            # CPU
            specs.append(cls._create_cpu_spec())

            # CUDA
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    spec = cls._create_cuda_spec(i)
                    if spec:
                        specs.append(spec)

            # MPS
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                specs.append(cls._create_mps_spec())

        except ImportError:
            pass

        return specs

    @classmethod
    def _create_cpu_spec(cls) -> DeviceSpec:
        """Create CPU spec from system info."""
        import os

        cores = os.cpu_count() or 4

        return DeviceSpec(
            device_id="cpu",
            device_type=DeviceType.CPU,
            compute_units=cores,
            clock_mhz=3000,  # Estimate
            peak_tflops_fp16=0.5,
            peak_tflops_fp32=1.0,
            memory_gb=64.0,  # Estimate
            memory_bandwidth_gbps=100.0,
            architecture="CPU",
        )

    @classmethod
    def _create_cuda_spec(cls, device_id: int) -> Optional[DeviceSpec]:
        """Create CUDA device spec."""
        try:
            import torch

            props = torch.cuda.get_device_properties(device_id)
            name = props.name.upper().replace(" ", "_")

            # Try to match to known device
            for key in DEVICE_SPECS:
                if key in name or name in key:
                    spec = DEVICE_SPECS[key]
                    # Clone with correct device_id
                    return DeviceSpec(
                        device_id=f"cuda:{device_id}",
                        device_type=spec.device_type,
                        compute_units=props.multi_processor_count,
                        clock_mhz=props.clock_rate // 1000,
                        peak_tflops_fp16=spec.peak_tflops_fp16,
                        peak_tflops_fp32=spec.peak_tflops_fp32,
                        memory_gb=props.total_memory / (1024**3),
                        memory_bandwidth_gbps=spec.memory_bandwidth_gbps,
                        tensor_cores=spec.tensor_cores,
                        supports_bf16=spec.supports_bf16,
                        compute_capability=f"{props.major}.{props.minor}",
                        architecture=spec.architecture,
                    )

            # Generic CUDA device
            return DeviceSpec(
                device_id=f"cuda:{device_id}",
                device_type=DeviceType.CUDA,
                compute_units=props.multi_processor_count,
                clock_mhz=props.clock_rate // 1000,
                peak_tflops_fp16=10.0,  # Conservative estimate
                peak_tflops_fp32=5.0,
                memory_gb=props.total_memory / (1024**3),
                memory_bandwidth_gbps=500.0,
                compute_capability=f"{props.major}.{props.minor}",
            )

        except Exception:
            return None

    @classmethod
    def _create_mps_spec(cls) -> DeviceSpec:
        """Create MPS (Apple Silicon) spec."""
        # Try to detect M-series chip
        return DeviceSpec(
            device_id="mps",
            device_type=DeviceType.MPS,
            compute_units=32,  # Estimate
            clock_mhz=1000,
            peak_tflops_fp16=10.0,
            peak_tflops_fp32=5.0,
            memory_gb=16.0,  # Unified memory estimate
            memory_bandwidth_gbps=200.0,
            supports_bf16=True,
            architecture="Apple Silicon",
        )


def get_device_spec(device_name: str) -> Optional[DeviceSpec]:
    """Convenience function to get device spec."""
    return DeviceRegistry.get_device_spec(device_name)


def register_custom_device(name: str, spec: Dict):
    """Convenience function to register custom device."""
    DeviceRegistry.register_custom_device_from_dict(name, spec)


def list_supported_devices() -> List[str]:
    """List all supported devices."""
    return DeviceRegistry.list_devices()
