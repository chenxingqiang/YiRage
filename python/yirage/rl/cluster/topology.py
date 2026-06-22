"""
Cluster Topology Definition

Defines cluster structure with nodes, devices, and network connectivity.
All simulated on single device for development and optimization.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple
import numpy as np


class TopologyType(Enum):
    """Standard cluster topology types."""

    SINGLE_NODE = "single_node"  # Single machine, multiple GPUs
    RING = "ring"  # Ring topology
    TREE = "tree"  # Tree/hierarchical topology
    FULL_MESH = "full_mesh"  # Fully connected
    TORUS = "torus"  # 2D/3D torus
    FAT_TREE = "fat_tree"  # Data center fat-tree
    CUSTOM = "custom"  # User-defined


class DeviceType(Enum):
    """
    Hardware device types.

    Supports all PyTorch-compatible devices plus custom hardware:
    - Standard: CPU, CUDA (NVIDIA GPU)
    - Apple: MPS (Metal Performance Shaders)
    - Accelerators: TPU (Google), NPU (various), FPGA
    - Custom: MACA (MetaX), Ascend (Huawei), XPU (Intel), etc.
    """

    # Standard devices
    CPU = "cpu"
    CUDA = "cuda"  # NVIDIA GPU

    # Apple Silicon
    MPS = "mps"  # Apple Metal

    # Google TPU
    TPU = "tpu"  # Google TPU (via PyTorch XLA)

    # NPU variants
    NPU = "npu"  # Generic NPU
    ASCEND = "ascend"  # Huawei Ascend NPU

    # Intel
    XPU = "xpu"  # Intel GPU/Accelerator

    # AMD
    ROCM = "rocm"  # AMD GPU (ROCm)

    # Custom accelerators
    MACA = "maca"  # MetaX MACA GPU
    FPGA = "fpga"  # FPGA accelerator

    # AWS
    NEURON = "neuron"  # AWS Inferentia/Trainium

    # Qualcomm
    HEXAGON = "hexagon"  # Qualcomm Hexagon DSP

    # Generic/Custom
    CUSTOM = "custom"  # User-defined custom device

    @classmethod
    def from_string(cls, s: str) -> "DeviceType":
        """Create DeviceType from string, with fallback to CUSTOM."""
        s = s.lower().strip()
        for member in cls:
            if member.value == s:
                return member
        # Check common aliases
        aliases = {
            "gpu": cls.CUDA,
            "nvidia": cls.CUDA,
            "amd": cls.ROCM,
            "intel": cls.XPU,
            "apple": cls.MPS,
            "metal": cls.MPS,
            "huawei": cls.ASCEND,
            "google": cls.TPU,
            "aws": cls.NEURON,
            "inferentia": cls.NEURON,
            "trainium": cls.NEURON,
        }
        if s in aliases:
            return aliases[s]
        return cls.CUSTOM

    @classmethod
    def detect_from_pytorch(cls) -> List["DeviceType"]:
        """Detect available device types from PyTorch."""
        available = [cls.CPU]  # CPU always available

        try:
            import torch

            # CUDA
            if torch.cuda.is_available():
                available.append(cls.CUDA)

            # MPS (Apple Silicon)
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                available.append(cls.MPS)

            # XPU (Intel)
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                available.append(cls.XPU)

            # Check for custom backends
            try:
                import torch_npu

                if torch_npu.npu.is_available():
                    available.append(cls.ASCEND)
            except ImportError:
                pass

            try:
                import torch_maca

                available.append(cls.MACA)
            except ImportError:
                pass

            try:
                import torch_xla

                available.append(cls.TPU)
            except ImportError:
                pass

        except ImportError:
            pass

        return available


@dataclass
class DeviceSpec:
    """
    Specification of a single compute device.

    Supports all device types including:
    - GPU: CUDA (NVIDIA), ROCm (AMD), MPS (Apple), XPU (Intel), MACA (MetaX)
    - NPU: Ascend (Huawei), Neuron (AWS), Hexagon (Qualcomm)
    - TPU: Google Cloud TPU
    - CPU: x86, ARM
    - FPGA: Custom FPGA accelerators
    - Custom: User-defined devices
    """

    device_id: str  # Unique identifier
    device_type: DeviceType

    # Compute capability
    compute_units: int  # SMs, CUs, cores, AI cores
    clock_mhz: int
    peak_tflops_fp16: float
    peak_tflops_fp32: float

    # Memory
    memory_gb: float
    memory_bandwidth_gbps: float

    # Optional compute capabilities
    peak_tflops_int8: float = 0.0
    peak_tflops_bf16: float = 0.0
    peak_tflops_fp64: float = 0.0
    peak_tops_int4: float = 0.0  # INT4 TOPS for quantized models

    # Features
    tensor_cores: bool = False  # NVIDIA Tensor Cores
    matrix_units: bool = False  # AMD Matrix Cores, Intel XMX
    supports_bf16: bool = False
    supports_fp8: bool = False
    supports_int4: bool = False
    supports_sparsity: bool = False  # Structured sparsity support

    # Device-specific info
    compute_capability: str = ""  # e.g., "8.0" for A100
    architecture: str = ""  # e.g., "Ampere", "Hopper", "CDNA2"
    driver_version: str = ""

    # Power and thermal
    tdp_watts: float = 0.0
    max_power_watts: float = 0.0

    # Custom properties for user-defined devices
    custom_properties: Dict = field(default_factory=dict)

    def peak_compute(self, dtype: str = "fp16") -> float:
        """Get peak compute in TFLOPS for given dtype."""
        dtype = dtype.lower()
        if dtype == "fp16":
            return self.peak_tflops_fp16
        elif dtype == "bf16":
            return self.peak_tflops_bf16 if self.peak_tflops_bf16 > 0 else self.peak_tflops_fp16
        elif dtype == "fp32":
            return self.peak_tflops_fp32
        elif dtype == "fp64":
            return self.peak_tflops_fp64
        elif dtype == "int8":
            return self.peak_tflops_int8
        elif dtype == "int4":
            return self.peak_tops_int4
        return self.peak_tflops_fp32

    def to_feature_vector(self) -> np.ndarray:
        """Convert to feature vector for ML models."""
        # Device type one-hot encoding (first 16 values)
        device_type_features = np.zeros(16, dtype=np.float32)
        device_types = list(DeviceType)
        if self.device_type in device_types:
            idx = device_types.index(self.device_type)
            if idx < 16:
                device_type_features[idx] = 1.0

        # Compute features
        compute_features = np.array(
            [
                self.compute_units / 128.0,  # Normalized
                self.clock_mhz / 2000.0,
                self.peak_tflops_fp16 / 500.0,
                self.peak_tflops_fp32 / 100.0,
                self.peak_tflops_bf16 / 500.0,
                self.peak_tflops_int8 / 1000.0,
                self.memory_gb / 80.0,
                self.memory_bandwidth_gbps / 3000.0,
                float(self.tensor_cores),
                float(self.matrix_units),
                float(self.supports_bf16),
                float(self.supports_fp8),
                float(self.supports_int4),
                float(self.supports_sparsity),
                self.tdp_watts / 500.0,
            ],
            dtype=np.float32,
        )

        # Concatenate all features
        return np.concatenate([device_type_features, compute_features])

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "device_id": self.device_id,
            "device_type": self.device_type.value,
            "compute_units": self.compute_units,
            "clock_mhz": self.clock_mhz,
            "peak_tflops_fp16": self.peak_tflops_fp16,
            "peak_tflops_fp32": self.peak_tflops_fp32,
            "peak_tflops_bf16": self.peak_tflops_bf16,
            "peak_tflops_int8": self.peak_tflops_int8,
            "memory_gb": self.memory_gb,
            "memory_bandwidth_gbps": self.memory_bandwidth_gbps,
            "tensor_cores": self.tensor_cores,
            "supports_bf16": self.supports_bf16,
            "supports_fp8": self.supports_fp8,
            "architecture": self.architecture,
            "custom_properties": self.custom_properties,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "DeviceSpec":
        """Create from dictionary."""
        device_type = DeviceType.from_string(d.get("device_type", "cuda"))
        return cls(
            device_id=d.get("device_id", "device0"),
            device_type=device_type,
            compute_units=d.get("compute_units", 64),
            clock_mhz=d.get("clock_mhz", 1500),
            peak_tflops_fp16=d.get("peak_tflops_fp16", 100.0),
            peak_tflops_fp32=d.get("peak_tflops_fp32", 20.0),
            peak_tflops_bf16=d.get("peak_tflops_bf16", 0.0),
            peak_tflops_int8=d.get("peak_tflops_int8", 0.0),
            memory_gb=d.get("memory_gb", 32.0),
            memory_bandwidth_gbps=d.get("memory_bandwidth_gbps", 1000.0),
            tensor_cores=d.get("tensor_cores", False),
            supports_bf16=d.get("supports_bf16", False),
            supports_fp8=d.get("supports_fp8", False),
            architecture=d.get("architecture", ""),
            custom_properties=d.get("custom_properties", {}),
        )


@dataclass
class NetworkLink:
    """Network link between two devices or nodes."""

    src_id: str
    dst_id: str

    # Performance characteristics
    bandwidth_gbps: float  # Unidirectional bandwidth
    latency_us: float  # One-way latency

    # Link type
    link_type: str = "pcie"  # pcie, nvlink, ib, ethernet
    bidirectional: bool = True

    def transfer_time_ms(self, size_bytes: int) -> float:
        """Estimate transfer time in milliseconds."""
        size_gb = size_bytes / (1024**3)
        transfer_ms = (size_gb / self.bandwidth_gbps) * 1000
        latency_ms = self.latency_us / 1000
        return transfer_ms + latency_ms


@dataclass
class ComputeNode:
    """A compute node with multiple devices."""

    node_id: str
    devices: List[DeviceSpec] = field(default_factory=list)

    # Intra-node connectivity
    intra_node_links: List[NetworkLink] = field(default_factory=list)

    # Node-level resources
    host_memory_gb: float = 256.0
    numa_nodes: int = 2

    def total_memory_gb(self) -> float:
        """Total device memory across all devices."""
        return sum(d.memory_gb for d in self.devices)

    def total_compute_tflops(self, dtype: str = "fp16") -> float:
        """Total compute capability."""
        return sum(d.peak_compute(dtype) for d in self.devices)

    def get_device(self, device_id: str) -> Optional[DeviceSpec]:
        """Get device by ID."""
        for d in self.devices:
            if d.device_id == device_id:
                return d
        return None


@dataclass
class ClusterTopology:
    """Complete cluster topology definition."""

    name: str
    topology_type: TopologyType

    # Nodes
    nodes: List[ComputeNode] = field(default_factory=list)

    # Inter-node connectivity
    inter_node_links: List[NetworkLink] = field(default_factory=list)

    # Cached matrices (computed on demand)
    _bandwidth_matrix: Optional[np.ndarray] = field(default=None, repr=False)
    _latency_matrix: Optional[np.ndarray] = field(default=None, repr=False)
    _device_list: Optional[List[str]] = field(default=None, repr=False)

    def all_devices(self) -> List[Tuple[str, DeviceSpec]]:
        """Get all devices with their full IDs (node_id/device_id)."""
        devices = []
        for node in self.nodes:
            for device in node.devices:
                full_id = f"{node.node_id}/{device.device_id}"
                devices.append((full_id, device))
        return devices

    def num_devices(self) -> int:
        """Total number of devices."""
        return sum(len(n.devices) for n in self.nodes)

    def total_memory_gb(self) -> float:
        """Total memory across cluster."""
        return sum(n.total_memory_gb() for n in self.nodes)

    def total_compute_tflops(self, dtype: str = "fp16") -> float:
        """Total compute across cluster."""
        return sum(n.total_compute_tflops(dtype) for n in self.nodes)

    def get_bandwidth_matrix(self) -> np.ndarray:
        """Get NxN bandwidth matrix between all devices."""
        if self._bandwidth_matrix is not None:
            return self._bandwidth_matrix

        devices = self.all_devices()
        n = len(devices)
        matrix = np.zeros((n, n), dtype=np.float32)

        # Build device index map
        device_to_idx = {d[0]: i for i, d in enumerate(devices)}

        # Fill intra-node bandwidths
        for node in self.nodes:
            for link in node.intra_node_links:
                src_full = f"{node.node_id}/{link.src_id}"
                dst_full = f"{node.node_id}/{link.dst_id}"
                if src_full in device_to_idx and dst_full in device_to_idx:
                    i, j = device_to_idx[src_full], device_to_idx[dst_full]
                    matrix[i, j] = link.bandwidth_gbps
                    if link.bidirectional:
                        matrix[j, i] = link.bandwidth_gbps

        # Fill inter-node bandwidths
        for link in self.inter_node_links:
            # For inter-node, src/dst are node IDs
            src_node = next((n for n in self.nodes if n.node_id == link.src_id), None)
            dst_node = next((n for n in self.nodes if n.node_id == link.dst_id), None)

            if src_node and dst_node:
                for src_dev in src_node.devices:
                    for dst_dev in dst_node.devices:
                        src_full = f"{src_node.node_id}/{src_dev.device_id}"
                        dst_full = f"{dst_node.node_id}/{dst_dev.device_id}"
                        i, j = device_to_idx[src_full], device_to_idx[dst_full]
                        matrix[i, j] = link.bandwidth_gbps
                        if link.bidirectional:
                            matrix[j, i] = link.bandwidth_gbps

        self._bandwidth_matrix = matrix
        self._device_list = [d[0] for d in devices]
        return matrix

    def get_latency_matrix(self) -> np.ndarray:
        """Get NxN latency matrix between all devices (microseconds)."""
        if self._latency_matrix is not None:
            return self._latency_matrix

        devices = self.all_devices()
        n = len(devices)
        matrix = np.full((n, n), np.inf, dtype=np.float32)
        np.fill_diagonal(matrix, 0)

        device_to_idx = {d[0]: i for i, d in enumerate(devices)}

        # Fill intra-node latencies
        for node in self.nodes:
            for link in node.intra_node_links:
                src_full = f"{node.node_id}/{link.src_id}"
                dst_full = f"{node.node_id}/{link.dst_id}"
                if src_full in device_to_idx and dst_full in device_to_idx:
                    i, j = device_to_idx[src_full], device_to_idx[dst_full]
                    matrix[i, j] = link.latency_us
                    if link.bidirectional:
                        matrix[j, i] = link.latency_us

        # Fill inter-node latencies
        for link in self.inter_node_links:
            src_node = next((n for n in self.nodes if n.node_id == link.src_id), None)
            dst_node = next((n for n in self.nodes if n.node_id == link.dst_id), None)

            if src_node and dst_node:
                for src_dev in src_node.devices:
                    for dst_dev in dst_node.devices:
                        src_full = f"{src_node.node_id}/{src_dev.device_id}"
                        dst_full = f"{dst_node.node_id}/{dst_dev.device_id}"
                        i, j = device_to_idx[src_full], device_to_idx[dst_full]
                        matrix[i, j] = link.latency_us
                        if link.bidirectional:
                            matrix[j, i] = link.latency_us

        self._latency_matrix = matrix
        return matrix

    def transfer_time_ms(self, src_device: str, dst_device: str, size_bytes: int) -> float:
        """Estimate transfer time between two devices."""
        if src_device == dst_device:
            return 0.0

        bw_matrix = self.get_bandwidth_matrix()
        lat_matrix = self.get_latency_matrix()

        if self._device_list is None:
            return float("inf")

        try:
            i = self._device_list.index(src_device)
            j = self._device_list.index(dst_device)
        except ValueError:
            return float("inf")

        bandwidth_gbps = bw_matrix[i, j]
        latency_us = lat_matrix[i, j]

        if bandwidth_gbps == 0:
            return float("inf")

        size_gb = size_bytes / (1024**3)
        transfer_ms = (size_gb / bandwidth_gbps) * 1000
        latency_ms = latency_us / 1000

        return transfer_ms + latency_ms

    @classmethod
    def create_single_node(
        cls,
        num_gpus: int = 8,
        gpu_type: str = "A100",
        nvlink: bool = True,
    ) -> "ClusterTopology":
        """Create a single-node multi-GPU topology."""

        # GPU specifications
        gpu_specs = {
            "A100": DeviceSpec(
                device_id="",
                device_type=DeviceType.CUDA,
                compute_units=108,
                clock_mhz=1410,
                peak_tflops_fp16=312.0,
                peak_tflops_fp32=19.5,
                peak_tflops_int8=624.0,
                memory_gb=80.0,
                memory_bandwidth_gbps=2039.0,
                tensor_cores=True,
                supports_bf16=True,
            ),
            "H100": DeviceSpec(
                device_id="",
                device_type=DeviceType.CUDA,
                compute_units=132,
                clock_mhz=1830,
                peak_tflops_fp16=989.0,
                peak_tflops_fp32=67.0,
                peak_tflops_int8=1979.0,
                memory_gb=80.0,
                memory_bandwidth_gbps=3350.0,
                tensor_cores=True,
                supports_bf16=True,
                supports_fp8=True,
            ),
            "V100": DeviceSpec(
                device_id="",
                device_type=DeviceType.CUDA,
                compute_units=80,
                clock_mhz=1380,
                peak_tflops_fp16=125.0,
                peak_tflops_fp32=15.7,
                memory_gb=32.0,
                memory_bandwidth_gbps=900.0,
                tensor_cores=True,
            ),
        }

        base_spec = gpu_specs.get(gpu_type, gpu_specs["A100"])

        # Create devices
        devices = []
        for i in range(num_gpus):
            spec = DeviceSpec(
                device_id=f"gpu{i}",
                device_type=base_spec.device_type,
                compute_units=base_spec.compute_units,
                clock_mhz=base_spec.clock_mhz,
                peak_tflops_fp16=base_spec.peak_tflops_fp16,
                peak_tflops_fp32=base_spec.peak_tflops_fp32,
                peak_tflops_int8=base_spec.peak_tflops_int8,
                memory_gb=base_spec.memory_gb,
                memory_bandwidth_gbps=base_spec.memory_bandwidth_gbps,
                tensor_cores=base_spec.tensor_cores,
                supports_bf16=base_spec.supports_bf16,
                supports_fp8=base_spec.supports_fp8,
            )
            devices.append(spec)

        # Create intra-node links
        links = []
        for i in range(num_gpus):
            for j in range(num_gpus):
                if i != j:
                    if nvlink:
                        # NVLink: high bandwidth, low latency
                        links.append(
                            NetworkLink(
                                src_id=f"gpu{i}",
                                dst_id=f"gpu{j}",
                                bandwidth_gbps=600.0 if gpu_type == "H100" else 300.0,
                                latency_us=1.0,
                                link_type="nvlink",
                            )
                        )
                    else:
                        # PCIe: lower bandwidth, higher latency
                        links.append(
                            NetworkLink(
                                src_id=f"gpu{i}",
                                dst_id=f"gpu{j}",
                                bandwidth_gbps=32.0,
                                latency_us=5.0,
                                link_type="pcie",
                            )
                        )

        node = ComputeNode(
            node_id="node0",
            devices=devices,
            intra_node_links=links,
        )

        return cls(
            name=f"single_{gpu_type}x{num_gpus}",
            topology_type=TopologyType.SINGLE_NODE,
            nodes=[node],
        )

    @classmethod
    def create_multi_node(
        cls,
        num_nodes: int = 4,
        gpus_per_node: int = 8,
        gpu_type: str = "A100",
        inter_node_bandwidth_gbps: float = 100.0,  # InfiniBand
        inter_node_latency_us: float = 2.0,
    ) -> "ClusterTopology":
        """Create multi-node cluster topology."""

        nodes = []
        for n in range(num_nodes):
            # Create single-node topology and extract the node
            single = cls.create_single_node(gpus_per_node, gpu_type, nvlink=True)
            node = single.nodes[0]
            node.node_id = f"node{n}"
            # Update device IDs in links
            for link in node.intra_node_links:
                pass  # IDs are already correct
            nodes.append(node)

        # Create inter-node links (full mesh)
        inter_links = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    inter_links.append(
                        NetworkLink(
                            src_id=f"node{i}",
                            dst_id=f"node{j}",
                            bandwidth_gbps=inter_node_bandwidth_gbps,
                            latency_us=inter_node_latency_us,
                            link_type="infiniband",
                        )
                    )

        return cls(
            name=f"cluster_{num_nodes}x{gpus_per_node}_{gpu_type}",
            topology_type=TopologyType.FULL_MESH,
            nodes=nodes,
            inter_node_links=inter_links,
        )

    @classmethod
    def create_heterogeneous(
        cls,
        device_configs: List[Dict],
    ) -> "ClusterTopology":
        """Create heterogeneous cluster with mixed device types.

        Args:
            device_configs: List of device configurations, each with:
                - device_type: "cuda", "maca", "ascend", etc.
                - count: number of devices
                - specs: device specifications dict
        """
        devices = []
        device_idx = 0

        for config in device_configs:
            device_type = DeviceType(config.get("device_type", "cuda"))
            count = config.get("count", 1)
            specs = config.get("specs", {})

            for i in range(count):
                device = DeviceSpec(
                    device_id=f"dev{device_idx}",
                    device_type=device_type,
                    compute_units=specs.get("compute_units", 64),
                    clock_mhz=specs.get("clock_mhz", 1500),
                    peak_tflops_fp16=specs.get("peak_tflops_fp16", 100.0),
                    peak_tflops_fp32=specs.get("peak_tflops_fp32", 20.0),
                    memory_gb=specs.get("memory_gb", 32.0),
                    memory_bandwidth_gbps=specs.get("memory_bandwidth_gbps", 1000.0),
                    tensor_cores=specs.get("tensor_cores", False),
                    supports_bf16=specs.get("supports_bf16", False),
                )
                devices.append(device)
                device_idx += 1

        # Create links (assume PCIe for heterogeneous)
        links = []
        for i in range(len(devices)):
            for j in range(len(devices)):
                if i != j:
                    links.append(
                        NetworkLink(
                            src_id=devices[i].device_id,
                            dst_id=devices[j].device_id,
                            bandwidth_gbps=32.0,
                            latency_us=5.0,
                            link_type="pcie",
                        )
                    )

        node = ComputeNode(
            node_id="node0",
            devices=devices,
            intra_node_links=links,
        )

        return cls(
            name="heterogeneous_cluster",
            topology_type=TopologyType.SINGLE_NODE,
            nodes=[node],
        )

    @classmethod
    def create_from_registry(
        cls,
        device_list: List[str],
        link_type: str = "pcie",
        name: Optional[str] = None,
    ) -> "ClusterTopology":
        """Create cluster topology using device specs from the registry.

        This allows easy creation of heterogeneous clusters with known devices.

        Args:
            device_list: List of device names (e.g., ["A100", "A100", "V100", "Ascend910B"])
                        Each name is looked up in the DeviceRegistry.
                        Can use format "name:count" (e.g., "A100:4") for multiple devices.
            link_type: Connection type between devices ("nvlink", "pcie", "infiniband")
            name: Optional cluster name

        Returns:
            ClusterTopology with devices from the registry

        Example:
            # Create 4x A100 + 2x Ascend910B cluster
            cluster = ClusterTopology.create_from_registry(
                ["A100:4", "Ascend910B:2"],
                link_type="pcie"
            )
        """
        # Import device registry here to avoid circular import
        from .device_registry import get_device_spec

        devices = []
        device_idx = 0
        device_type_counts = {}

        for item in device_list:
            # Parse "name:count" format
            if ":" in item:
                device_name, count_str = item.split(":", 1)
                count = int(count_str)
            else:
                device_name = item
                count = 1

            # Get spec from registry
            base_spec = get_device_spec(device_name)
            if base_spec is None:
                raise ValueError(
                    f"Device '{device_name}' not found in registry. "
                    f"Use DeviceRegistry.list_devices() to see available devices."
                )

            # Track count by device type
            type_key = base_spec.device_type.value
            if type_key not in device_type_counts:
                device_type_counts[type_key] = 0

            for i in range(count):
                # Clone the spec with a unique device_id
                device = DeviceSpec(
                    device_id=f"{type_key}{device_type_counts[type_key]}",
                    device_type=base_spec.device_type,
                    compute_units=base_spec.compute_units,
                    clock_mhz=base_spec.clock_mhz,
                    peak_tflops_fp16=base_spec.peak_tflops_fp16,
                    peak_tflops_fp32=base_spec.peak_tflops_fp32,
                    peak_tflops_bf16=base_spec.peak_tflops_bf16,
                    peak_tflops_int8=base_spec.peak_tflops_int8,
                    peak_tflops_fp64=base_spec.peak_tflops_fp64,
                    peak_tops_int4=base_spec.peak_tops_int4,
                    memory_gb=base_spec.memory_gb,
                    memory_bandwidth_gbps=base_spec.memory_bandwidth_gbps,
                    tensor_cores=base_spec.tensor_cores,
                    matrix_units=base_spec.matrix_units,
                    supports_bf16=base_spec.supports_bf16,
                    supports_fp8=base_spec.supports_fp8,
                    supports_int4=base_spec.supports_int4,
                    supports_sparsity=base_spec.supports_sparsity,
                    compute_capability=base_spec.compute_capability,
                    architecture=base_spec.architecture,
                    tdp_watts=base_spec.tdp_watts,
                    max_power_watts=base_spec.max_power_watts,
                    custom_properties=base_spec.custom_properties.copy(),
                )
                devices.append(device)
                device_type_counts[type_key] += 1
                device_idx += 1

        # Determine link parameters based on link_type
        link_params = {
            "nvlink": {"bandwidth_gbps": 300.0, "latency_us": 1.0},
            "pcie": {"bandwidth_gbps": 32.0, "latency_us": 5.0},
            "infiniband": {"bandwidth_gbps": 100.0, "latency_us": 2.0},
            "ethernet": {"bandwidth_gbps": 25.0, "latency_us": 10.0},
        }
        params = link_params.get(link_type, link_params["pcie"])

        # Create links between all device pairs
        links = []
        for i in range(len(devices)):
            for j in range(len(devices)):
                if i != j:
                    links.append(
                        NetworkLink(
                            src_id=devices[i].device_id,
                            dst_id=devices[j].device_id,
                            bandwidth_gbps=params["bandwidth_gbps"],
                            latency_us=params["latency_us"],
                            link_type=link_type,
                        )
                    )

        node = ComputeNode(
            node_id="node0",
            devices=devices,
            intra_node_links=links,
        )

        # Generate name if not provided
        if name is None:
            name = "cluster_" + "_".join(device_list).replace(":", "x")

        return cls(
            name=name,
            topology_type=TopologyType.SINGLE_NODE,
            nodes=[node],
        )

    @classmethod
    def create_multi_node_from_registry(
        cls,
        device_per_node: List[str],
        num_nodes: int = 4,
        intra_node_link: str = "nvlink",
        inter_node_link: str = "infiniband",
        name: Optional[str] = None,
    ) -> "ClusterTopology":
        """Create multi-node cluster using device specs from the registry.

        Args:
            device_per_node: Device specification for each node (e.g., ["A100:8"])
            num_nodes: Number of nodes in the cluster
            intra_node_link: Link type within a node
            inter_node_link: Link type between nodes
            name: Optional cluster name

        Example:
            # Create 4-node cluster with 8x H100 per node
            cluster = ClusterTopology.create_multi_node_from_registry(
                device_per_node=["H100:8"],
                num_nodes=4,
            )
        """
        from .device_registry import get_device_spec

        nodes = []

        # Link parameters
        link_params = {
            "nvlink": {"bandwidth_gbps": 300.0, "latency_us": 1.0},
            "pcie": {"bandwidth_gbps": 32.0, "latency_us": 5.0},
            "infiniband": {"bandwidth_gbps": 200.0, "latency_us": 2.0},
            "ethernet": {"bandwidth_gbps": 100.0, "latency_us": 10.0},
        }
        intra_params = link_params.get(intra_node_link, link_params["pcie"])
        inter_params = link_params.get(inter_node_link, link_params["infiniband"])

        for n in range(num_nodes):
            devices = []
            device_idx = 0

            for item in device_per_node:
                if ":" in item:
                    device_name, count_str = item.split(":", 1)
                    count = int(count_str)
                else:
                    device_name = item
                    count = 1

                base_spec = get_device_spec(device_name)
                if base_spec is None:
                    raise ValueError(f"Device '{device_name}' not found in registry.")

                for i in range(count):
                    device = DeviceSpec(
                        device_id=f"n{n}_gpu{device_idx}",
                        device_type=base_spec.device_type,
                        compute_units=base_spec.compute_units,
                        clock_mhz=base_spec.clock_mhz,
                        peak_tflops_fp16=base_spec.peak_tflops_fp16,
                        peak_tflops_fp32=base_spec.peak_tflops_fp32,
                        peak_tflops_bf16=base_spec.peak_tflops_bf16,
                        peak_tflops_int8=base_spec.peak_tflops_int8,
                        memory_gb=base_spec.memory_gb,
                        memory_bandwidth_gbps=base_spec.memory_bandwidth_gbps,
                        tensor_cores=base_spec.tensor_cores,
                        supports_bf16=base_spec.supports_bf16,
                        supports_fp8=base_spec.supports_fp8,
                        architecture=base_spec.architecture,
                    )
                    devices.append(device)
                    device_idx += 1

            # Create intra-node links
            links = []
            for i in range(len(devices)):
                for j in range(len(devices)):
                    if i != j:
                        links.append(
                            NetworkLink(
                                src_id=devices[i].device_id,
                                dst_id=devices[j].device_id,
                                bandwidth_gbps=intra_params["bandwidth_gbps"],
                                latency_us=intra_params["latency_us"],
                                link_type=intra_node_link,
                            )
                        )

            nodes.append(
                ComputeNode(
                    node_id=f"node{n}",
                    devices=devices,
                    intra_node_links=links,
                )
            )

        # Create inter-node links (full mesh)
        inter_links = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    inter_links.append(
                        NetworkLink(
                            src_id=f"node{i}",
                            dst_id=f"node{j}",
                            bandwidth_gbps=inter_params["bandwidth_gbps"],
                            latency_us=inter_params["latency_us"],
                            link_type=inter_node_link,
                        )
                    )

        if name is None:
            name = f"multinode_{num_nodes}x{'_'.join(device_per_node)}"

        return cls(
            name=name,
            topology_type=TopologyType.FULL_MESH,
            nodes=nodes,
            inter_node_links=inter_links,
        )

    @classmethod
    def detect_local_devices(cls, name: str = "local_cluster") -> "ClusterTopology":
        """Auto-detect local devices using PyTorch and create a topology.

        This method attempts to detect:
        - CUDA devices (NVIDIA GPUs)
        - MPS devices (Apple Silicon)
        - XPU devices (Intel)
        - CPU

        Returns:
            ClusterTopology with detected devices
        """
        from .device_registry import DeviceRegistry

        devices = DeviceRegistry.detect_pytorch_devices()

        if not devices:
            # Fallback to CPU
            devices = [
                DeviceSpec(
                    device_id="cpu",
                    device_type=DeviceType.CPU,
                    compute_units=4,
                    clock_mhz=3000,
                    peak_tflops_fp16=0.5,
                    peak_tflops_fp32=1.0,
                    memory_gb=32.0,
                    memory_bandwidth_gbps=100.0,
                )
            ]

        # Create links between all device pairs
        links = []
        for i in range(len(devices)):
            for j in range(len(devices)):
                if i != j:
                    links.append(
                        NetworkLink(
                            src_id=devices[i].device_id,
                            dst_id=devices[j].device_id,
                            bandwidth_gbps=32.0,  # PCIe
                            latency_us=5.0,
                            link_type="pcie",
                        )
                    )

        node = ComputeNode(
            node_id="local",
            devices=devices,
            intra_node_links=links,
        )

        return cls(
            name=name,
            topology_type=TopologyType.SINGLE_NODE,
            nodes=[node],
        )

    def to_feature_dict(self) -> Dict:
        """Convert topology to feature dictionary for ML models."""
        devices = self.all_devices()

        return {
            "num_nodes": len(self.nodes),
            "num_devices": len(devices),
            "total_memory_gb": self.total_memory_gb(),
            "total_compute_tflops_fp16": self.total_compute_tflops("fp16"),
            "total_compute_tflops_fp32": self.total_compute_tflops("fp32"),
            "device_features": np.stack([d[1].to_feature_vector() for d in devices]),
            "bandwidth_matrix": self.get_bandwidth_matrix(),
            "latency_matrix": self.get_latency_matrix(),
            "topology_type": self.topology_type.value,
        }
