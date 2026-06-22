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
Verifier Pool - Manages multiple GPU Verifier actors.

Provides:
1. Load balancing across GPUs
2. Async batch verification
3. Fault tolerance
"""

from typing import List, Optional, Any, Union
from dataclasses import dataclass
import threading
import queue

from .gpu_verifier import VerifyResult, ProfileResult, LocalGPUVerifier


@dataclass
class VerifyRequest:
    """Verification request."""

    request_id: int
    kernel_graph_json: str
    target_graph_json: str
    callback: Optional[callable] = None


class VerifierPool:
    """
    Pool of GPU Verifiers for distributed verification.

    Features:
    - Round-robin load balancing
    - Async verification with futures
    - Batch processing for efficiency
    - Automatic fallback to local if Ray unavailable
    """

    def __init__(
        self,
        num_gpus: int = 1,
        verifiers_per_gpu: int = 2,
        gpu_fraction: float = 0.5,
        use_ray: bool = True,
    ):
        """
        Initialize verifier pool.

        Args:
            num_gpus: Number of GPUs to use
            verifiers_per_gpu: Verifiers per GPU (for async)
            gpu_fraction: GPU memory fraction per verifier
            use_ray: Whether to use Ray actors
        """
        self.num_gpus = num_gpus
        self.verifiers_per_gpu = verifiers_per_gpu
        self.gpu_fraction = gpu_fraction
        self.use_ray = use_ray

        self.verifiers: List[Any] = []
        self.current_idx = 0
        self._lock = threading.Lock()

        self._initialized = False

    def _ensure_initialized(self):
        """Lazy initialization of verifiers."""
        if self._initialized:
            return

        if self.use_ray:
            try:
                import ray
                from .gpu_verifier import create_gpu_verifier

                if not ray.is_initialized():
                    ray.init()

                GPUVerifierActor = create_gpu_verifier(self.gpu_fraction)

                for gpu_id in range(self.num_gpus):
                    for _ in range(self.verifiers_per_gpu):
                        verifier = GPUVerifierActor.remote(
                            gpu_id=gpu_id,
                            backend="cuda",
                        )
                        self.verifiers.append(verifier)

                # Wait for all verifiers to be ready
                ray.get([v.is_ready.remote() for v in self.verifiers])

                self._initialized = True
                print(f"VerifierPool: {len(self.verifiers)} Ray verifiers ready")

            except ImportError:
                print("Ray not available, using local verifiers")
                self.use_ray = False

        if not self.use_ray:
            # Local verifiers (no Ray)
            for gpu_id in range(self.num_gpus):
                for _ in range(self.verifiers_per_gpu):
                    verifier = LocalGPUVerifier(gpu_id=gpu_id)
                    self.verifiers.append(verifier)

            self._initialized = True
            print(f"VerifierPool: {len(self.verifiers)} local verifiers ready")

    def _get_next_verifier(self) -> Any:
        """Get next verifier (round-robin)."""
        with self._lock:
            verifier = self.verifiers[self.current_idx]
            self.current_idx = (self.current_idx + 1) % len(self.verifiers)
            return verifier

    def verify(
        self,
        kernel_graph_json: str,
        target_graph_json: str,
    ) -> VerifyResult:
        """
        Synchronous verification.

        Args:
            kernel_graph_json: Candidate kernel graph
            target_graph_json: Target computation graph

        Returns:
            VerifyResult
        """
        self._ensure_initialized()

        verifier = self._get_next_verifier()

        if self.use_ray:
            import ray

            future = verifier.verify_fingerprint.remote(
                kernel_graph_json,
                target_graph_json,
            )
            return ray.get(future)
        else:
            return verifier.verify_fingerprint(
                kernel_graph_json,
                target_graph_json,
            )

    def verify_async(
        self,
        kernel_graph_json: str,
        target_graph_json: str,
    ) -> Any:
        """
        Async verification - returns future.

        Args:
            kernel_graph_json: Candidate kernel graph
            target_graph_json: Target computation graph

        Returns:
            Ray ObjectRef or local result
        """
        self._ensure_initialized()

        verifier = self._get_next_verifier()

        if self.use_ray:
            return verifier.verify_fingerprint.remote(
                kernel_graph_json,
                target_graph_json,
            )
        else:
            # For local, just return result directly
            return verifier.verify_fingerprint(
                kernel_graph_json,
                target_graph_json,
            )

    def verify_batch(
        self,
        requests: List[tuple],  # List of (kernel_json, target_json)
    ) -> List[VerifyResult]:
        """
        Batch verification.

        Args:
            requests: List of (kernel_graph_json, target_graph_json) tuples

        Returns:
            List of VerifyResult
        """
        self._ensure_initialized()

        if self.use_ray:
            import ray

            futures = []
            for kernel_json, target_json in requests:
                future = self.verify_async(kernel_json, target_json)
                futures.append(future)

            return ray.get(futures)
        else:
            results = []
            for kernel_json, target_json in requests:
                result = self.verify(kernel_json, target_json)
                results.append(result)
            return results

    def profile(
        self,
        kernel_graph_json: str,
        input_shapes: List[List[int]],
        warmup_iters: int = 10,
        profile_iters: int = 100,
    ) -> ProfileResult:
        """
        Profile kernel performance.

        Args:
            kernel_graph_json: Kernel graph
            input_shapes: Input tensor shapes
            warmup_iters: Warmup iterations
            profile_iters: Profile iterations

        Returns:
            ProfileResult
        """
        self._ensure_initialized()

        verifier = self._get_next_verifier()

        if self.use_ray:
            import ray

            future = verifier.profile_kernel.remote(
                kernel_graph_json,
                input_shapes,
                warmup_iters,
                profile_iters,
            )
            return ray.get(future)
        else:
            return verifier.profile_kernel(
                kernel_graph_json,
                input_shapes,
                warmup_iters,
                profile_iters,
            )

    def shutdown(self):
        """Shutdown all verifiers."""
        if self.use_ray:
            import ray

            for verifier in self.verifiers:
                ray.kill(verifier)

        self.verifiers = []
        self._initialized = False

    def __len__(self) -> int:
        return len(self.verifiers)

    def __del__(self):
        self.shutdown()
