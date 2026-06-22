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
GPU Verifier - Ray Actor for GPU-based kernel verification.

This is the critical component that closes the RL loop by providing:
1. Fingerprint verification (correctness check)
2. Performance profiling (reward signal)
"""

from dataclasses import dataclass
from typing import Optional, List, Any
import time


@dataclass
class VerifyResult:
    """Result from fingerprint verification."""

    verified: bool
    fingerprint_time_ms: float
    rejection_reason: str = ""
    kernel_graph_hash: int = 0


@dataclass
class ProfileResult:
    """Result from performance profiling."""

    latency_ms: float
    memory_bytes: int = 0
    flops: float = 0.0
    compile_time_ms: float = 0.0


def create_gpu_verifier(gpu_fraction: float = 0.5):
    """
    Factory function to create GPU Verifier Ray Actor.

    Args:
        gpu_fraction: Fraction of GPU memory to reserve

    Returns:
        Ray Actor class
    """
    import ray

    @ray.remote(num_gpus=gpu_fraction)
    class GPUVerifier:
        """
        GPU-based kernel verification Actor.

        Runs on GPU, provides:
        1. Fast fingerprint verification (O(ms))
        2. Kernel compilation (O(100ms))
        3. Accurate performance profiling (O(10ms))

        This Actor is the GPU-side of the RL closed loop.
        """

        def __init__(self, gpu_id: int = 0, backend: str = "cuda"):
            self.gpu_id = gpu_id
            self.backend = backend
            self._initialized = False

            # Lazy initialization to avoid import issues
            self._yirage_core = None
            self._torch = None

        def _ensure_initialized(self):
            """Lazy initialization of GPU context."""
            if self._initialized:
                return

            try:
                import torch

                self._torch = torch

                # Set GPU device
                if torch.cuda.is_available():
                    torch.cuda.set_device(self.gpu_id)
                    # Warm up CUDA context
                    _ = torch.zeros(1, device=f"cuda:{self.gpu_id}")

                # Import YiRage core
                try:
                    from yirage import core as yirage_core

                    self._yirage_core = yirage_core
                except ImportError:
                    self._yirage_core = None

                self._initialized = True

            except Exception as e:
                print(f"GPUVerifier init error: {e}")
                self._initialized = False

        def verify_fingerprint(
            self,
            kernel_graph_json: str,
            target_graph_json: str,
        ) -> VerifyResult:
            """
            Verify kernel correctness using fingerprint.

            This is the fast verification path:
            - Computes fingerprints for candidate and target
            - Compares fingerprints for equality

            Args:
                kernel_graph_json: Candidate kernel graph (JSON)
                target_graph_json: Target computation graph (JSON)

            Returns:
                VerifyResult with verification outcome
            """
            self._ensure_initialized()

            start_time = time.perf_counter()

            try:
                if self._yirage_core is not None:
                    # Call C++ fingerprint verification
                    result = self._yirage_core.verify_fingerprint(
                        kernel_graph_json,
                        target_graph_json,
                    )
                    verified = result.get("verified", False)
                    rejection = result.get("rejection_reason", "")
                else:
                    # Fallback: simulate verification
                    import hashlib

                    hash1 = hashlib.md5(kernel_graph_json.encode()).hexdigest()
                    hash2 = hashlib.md5(target_graph_json.encode()).hexdigest()
                    verified = hash1 == hash2
                    rejection = "" if verified else "fingerprint_mismatch"

            except Exception as e:
                verified = False
                rejection = str(e)

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            return VerifyResult(
                verified=verified,
                fingerprint_time_ms=elapsed_ms,
                rejection_reason=rejection,
            )

        def compile_kernel(
            self,
            kernel_graph_json: str,
            target_cc: int = 80,
        ) -> Optional[Any]:
            """
            Compile kernel graph to executable.

            Args:
                kernel_graph_json: Kernel graph (JSON)
                target_cc: CUDA compute capability

            Returns:
                Compiled kernel object or None if failed
            """
            self._ensure_initialized()

            try:
                if self._yirage_core is not None:
                    compiled = self._yirage_core.compile_kernel(
                        kernel_graph_json,
                        target_cc=target_cc,
                        gpu_id=self.gpu_id,
                    )
                    return compiled
                return None
            except Exception as e:
                print(f"Compile error: {e}")
                return None

        def profile_kernel(
            self,
            kernel_graph_json: str,
            input_shapes: List[List[int]],
            warmup_iters: int = 10,
            profile_iters: int = 100,
        ) -> ProfileResult:
            """
            Profile kernel performance.

            This provides the performance signal for RL reward.

            Args:
                kernel_graph_json: Kernel graph (JSON)
                input_shapes: Shapes of input tensors
                warmup_iters: Warmup iterations
                profile_iters: Profiling iterations

            Returns:
                ProfileResult with latency measurements
            """
            self._ensure_initialized()

            torch = self._torch
            if torch is None or not torch.cuda.is_available():
                return ProfileResult(latency_ms=float("inf"))

            compile_start = time.perf_counter()

            try:
                # Compile kernel
                compiled = self.compile_kernel(kernel_graph_json)
                if compiled is None:
                    return ProfileResult(
                        latency_ms=float("inf"),
                        compile_time_ms=(time.perf_counter() - compile_start) * 1000,
                    )

                compile_time_ms = (time.perf_counter() - compile_start) * 1000

                # Create input tensors
                device = f"cuda:{self.gpu_id}"
                inputs = [
                    torch.randn(shape, dtype=torch.float16, device=device) for shape in input_shapes
                ]

                # Warmup
                for _ in range(warmup_iters):
                    _ = compiled.run(inputs)

                torch.cuda.synchronize()

                # Profile
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                start_event.record()
                for _ in range(profile_iters):
                    _ = compiled.run(inputs)
                end_event.record()

                torch.cuda.synchronize()
                latency_ms = start_event.elapsed_time(end_event) / profile_iters

                # Estimate memory
                memory_bytes = sum(t.element_size() * t.nelement() for t in inputs)

                return ProfileResult(
                    latency_ms=latency_ms,
                    memory_bytes=memory_bytes,
                    compile_time_ms=compile_time_ms,
                )

            except Exception as e:
                print(f"Profile error: {e}")
                return ProfileResult(latency_ms=float("inf"))

        def is_ready(self) -> bool:
            """Check if verifier is ready."""
            self._ensure_initialized()
            return self._initialized

        def get_gpu_id(self) -> int:
            """Get assigned GPU ID."""
            return self.gpu_id

    return GPUVerifier


# Non-Ray version for testing
class LocalGPUVerifier:
    """
    Local GPU verifier (non-Ray) for testing.
    """

    def __init__(self, gpu_id: int = 0, backend: str = "cuda"):
        self.gpu_id = gpu_id
        self.backend = backend

    def verify_fingerprint(
        self,
        kernel_graph_json: str,
        target_graph_json: str,
    ) -> VerifyResult:
        """Local verification (simulated)."""
        import hashlib

        start = time.perf_counter()

        # Simulate fingerprint check
        h1 = hashlib.sha256(kernel_graph_json.encode()).hexdigest()[:16]
        h2 = hashlib.sha256(target_graph_json.encode()).hexdigest()[:16]
        verified = h1 == h2

        elapsed_ms = (time.perf_counter() - start) * 1000

        return VerifyResult(
            verified=verified,
            fingerprint_time_ms=elapsed_ms,
            rejection_reason="" if verified else "simulated_mismatch",
        )

    def profile_kernel(self, *args, **kwargs) -> ProfileResult:
        """Local profiling (simulated)."""
        return ProfileResult(latency_ms=1.0, memory_bytes=0)


# Export for convenience
GPUVerifier = LocalGPUVerifier  # Default to local, replace with Ray version when available
