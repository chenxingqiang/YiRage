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
AccelForge Verifier — Hardware-model based kernel verification.

Uses AccelForge's analytical model to verify and profile kernels on
virtual accelerator designs, without requiring physical hardware.

This enables:
1. Fast pre-screening of kernel candidates (µs vs ms)
2. Energy/area/power-aware profiling
3. Design space exploration for custom accelerators
"""

import time
from typing import Any, Dict, List, Optional

from .gpu_verifier import ProfileResult, VerifyResult


class AccelForgeVerifier:
    """
    AccelForge-based kernel verifier.

    Uses AccelForge's analytical model instead of physical GPU execution.
    Provides faster verification with multi-objective metrics (latency,
    energy, area, power).
    """

    def __init__(
        self,
        design_point: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            design_point: AccelForge design parameters. If None, uses default.
        """
        from ..hardware.accelforge_bridge import (
            AccelForgeBridge,
            AccelForgeDesignPoint,
        )

        self._bridge = AccelForgeBridge()

        if design_point:
            self._design = AccelForgeDesignPoint.from_dict(design_point)
        else:
            self._design = AccelForgeDesignPoint()

    def verify_fingerprint(
        self,
        kernel_graph_json: str,
        target_graph_json: str,
    ) -> VerifyResult:
        """
        Verify kernel correctness using structural analysis.

        For AccelForge targets, verification checks that the kernel
        can execute on the target accelerator design (PE fit, buffer fit, etc.)

        Args:
            kernel_graph_json: Candidate kernel graph (JSON)
            target_graph_json: Target computation graph (JSON)

        Returns:
            VerifyResult with verification outcome
        """
        import json

        start_time = time.perf_counter()

        from ..hardware.accelforge_bridge import normalize_mugraph_json

        try:
            kernel_raw = json.loads(kernel_graph_json) if kernel_graph_json else {}
            target_raw = json.loads(target_graph_json) if target_graph_json else {}
            kernel = normalize_mugraph_json(kernel_raw)
            target = normalize_mugraph_json(target_raw)
        except json.JSONDecodeError:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return VerifyResult(
                verified=False,
                fingerprint_time_ms=elapsed_ms,
                rejection_reason="invalid_json",
            )

        # Check structural compatibility
        rejection = self._check_compatibility(kernel, target)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return VerifyResult(
            verified=(rejection == ""),
            fingerprint_time_ms=elapsed_ms,
            rejection_reason=rejection,
        )

    def profile_kernel(
        self,
        kernel_graph_json: str,
        input_shapes: Optional[List[List[int]]] = None,
        warmup_iters: int = 0,
        profile_iters: int = 1,
    ) -> ProfileResult:
        """
        Profile kernel using AccelForge analytical model.

        Translates the YiRage µGraph into a proper AccelForge Einsum workload
        using ``mugraph_to_workload()`` which extracts real M×K×N dimensions
        from the actual operator types and tensor shapes in the graph.

        Args:
            kernel_graph_json: Kernel graph (JSON)
            input_shapes: Shapes of input tensors (used when graph has no shapes)
            warmup_iters: Ignored (no physical warmup needed)
            profile_iters: Ignored (analytical model)

        Returns:
            ProfileResult with estimated metrics
        """
        from ..hardware.accelforge_bridge import mugraph_to_workload

        try:
            workload = mugraph_to_workload(kernel_graph_json)
        except Exception:
            workload = {"estimated_flops": self._estimate_flops({}, input_shapes)}

        # Supplement with input_shapes-based flops if graph gave no operators
        if workload.get("estimated_flops", 0) <= 1.0 and input_shapes:
            workload["estimated_flops"] = self._estimate_flops({}, input_shapes)

        workload["memory_bytes"] = self._estimate_memory(
            {}, input_shapes
        )

        # Evaluate with AccelForge using the real workload dimensions
        metrics = self._bridge.evaluate(self._design, workload)

        return ProfileResult(
            latency_ms=metrics.latency_ms,
            memory_bytes=int(workload.get("memory_bytes", 0)),
            flops=workload.get("estimated_flops", 1.0),
            compile_time_ms=0.0,  # No compilation needed
        )

    def get_full_metrics(
        self,
        kernel_graph_json: str,
        input_shapes: Optional[List[List[int]]] = None,
    ) -> Dict[str, Any]:
        """
        Get full AccelForge metrics including energy, area, power.

        Uses ``mugraph_to_workload()`` to extract real M×K×N from the µGraph
        so AccelForge models the actual kernel structure.
        """
        from ..hardware.accelforge_bridge import mugraph_to_workload

        try:
            workload = mugraph_to_workload(kernel_graph_json)
        except Exception:
            workload = {"estimated_flops": self._estimate_flops({}, input_shapes)}

        workload["memory_bytes"] = self._estimate_memory({}, input_shapes)

        metrics = self._bridge.evaluate(self._design, workload)
        return metrics.to_dict()

    def prescreen_kernel(
        self,
        kernel_graph_json: str,
        target_graph_json: str = "{}",
        input_shapes: Optional[List[List[int]]] = None,
        latency_budget_ms: Optional[float] = None,
        energy_budget_pj: Optional[float] = None,
        area_budget_mm2: Optional[float] = None,
        power_budget_mw: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Pre-screen a kernel candidate with AccelForge before physical profiling.

        This is the intended end-to-end entry point for using AccelForge as a
        virtual-hardware oracle: reject candidates that violate structural or
        multi-objective budgets, then pass accepted candidates to real
        GPU/CPU/backend profilers for final validation.
        """
        verify = self.verify_fingerprint(kernel_graph_json, target_graph_json)
        metrics = self.get_full_metrics(kernel_graph_json, input_shapes)

        rejections = []
        if not verify.verified:
            rejections.append(verify.rejection_reason or "verification_failed")
        if latency_budget_ms is not None and metrics["latency_ms"] > latency_budget_ms:
            rejections.append("latency_budget_exceeded")
        if energy_budget_pj is not None and metrics["energy_per_op_pj"] > energy_budget_pj:
            rejections.append("energy_budget_exceeded")
        if area_budget_mm2 is not None and metrics["area_mm2"] > area_budget_mm2:
            rejections.append("area_budget_exceeded")
        if power_budget_mw is not None and metrics["total_power_mw"] > power_budget_mw:
            rejections.append("power_budget_exceeded")

        return {
            "accepted": not rejections,
            "rejections": rejections,
            "verified": verify.verified,
            "metrics": metrics,
            "next_step": "physical_profile" if not rejections else "reject_candidate",
        }

    def _check_compatibility(
        self, kernel: Dict, target: Dict
    ) -> str:
        """Check if kernel is compatible with accelerator design."""
        # Check operator count fits PE array
        num_ops = len(kernel.get("operators", []))
        max_parallel = self._design.total_pes

        if num_ops > max_parallel * 10:  # Reasonable limit
            return f"too_many_ops:{num_ops}_max:{max_parallel * 10}"

        # Check memory requirements
        total_tensor_bytes = 0
        for tensor in kernel.get("tensors", []):
            total_tensor_bytes += tensor.get("size_bytes", 0)

        max_buffer = self._design.total_buffer_kb * 1024
        if total_tensor_bytes > max_buffer * 10:  # Allow some spilling
            return f"memory_overflow:{total_tensor_bytes}_max:{int(max_buffer * 10)}"

        return ""  # Compatible

    def _estimate_flops(
        self, kernel: Dict, input_shapes: Optional[List[List[int]]]
    ) -> float:
        """Estimate FLOPs from kernel graph."""
        total_flops = 0.0
        for op in kernel.get("operators", []):
            total_flops += op.get("flops", 0.0)

        # Fallback: estimate from input shapes
        if total_flops == 0 and input_shapes:
            for shape in input_shapes:
                elements = 1
                for dim in shape:
                    elements *= dim
                total_flops += elements * 2  # Rough estimate

        return max(total_flops, 1.0)

    def _estimate_memory(
        self, kernel: Dict, input_shapes: Optional[List[List[int]]]
    ) -> float:
        """Estimate memory bytes from kernel graph."""
        total_bytes = 0.0
        for tensor in kernel.get("tensors", []):
            total_bytes += tensor.get("size_bytes", 0)

        if total_bytes == 0 and input_shapes:
            for shape in input_shapes:
                elements = 1
                for dim in shape:
                    elements *= dim
                total_bytes += elements * 2  # FP16

        return max(total_bytes, 1.0)

    @property
    def design(self):
        """Get current design point."""
        return self._design

    @design.setter
    def design(self, design_point):
        """Update design point."""
        from ..hardware.accelforge_bridge import AccelForgeDesignPoint

        if isinstance(design_point, dict):
            self._design = AccelForgeDesignPoint.from_dict(design_point)
        else:
            self._design = design_point
