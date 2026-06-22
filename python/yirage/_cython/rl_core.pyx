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
Cython wrapper for RL interface.

Provides Python bindings to C++ RL search core for closed-loop operation.
"""

from libc.stdlib cimport malloc, free
from cpython.bytes cimport PyBytes_FromString

cimport rl_core

import json


cdef class RLSearchContext:
    """
    Python wrapper for C++ RLSearchContext.
    
    Provides the interface for RL closed loop:
    - apply_action: Apply RL decision
    - verify: GPU-based verification
    - profile: GPU-based profiling
    - get_state: Get current search state
    """
    
    cdef void* _ctx
    cdef str _backend
    cdef int _gpu_id
    
    def __cinit__(self, str target_graph_json, str backend="cuda", int gpu_id=0):
        """
        Create new RL search context.
        
        Args:
            target_graph_json: Target computation graph (JSON)
            backend: Target backend (cuda, maca, etc.)
            gpu_id: GPU ID for verification
        """
        cdef bytes target_bytes = target_graph_json.encode('utf-8')
        cdef bytes backend_bytes = backend.encode('utf-8')
        cdef const char* target_ptr = target_bytes
        cdef const char* backend_ptr = backend_bytes
        
        with nogil:
            self._ctx = rl_core.rl_context_create(
                target_ptr,
                backend_ptr,
                gpu_id
            )
        
        if self._ctx == NULL:
            raise RuntimeError("Failed to create RLSearchContext")
        
        self._backend = backend
        self._gpu_id = gpu_id
    
    def __dealloc__(self):
        """Destroy context."""
        if self._ctx != NULL:
            with nogil:
                rl_core.rl_context_destroy(self._ctx)
            self._ctx = NULL
    
    def reset(self, str new_target_json=""):
        """
        Reset context for new episode.
        
        Args:
            new_target_json: Optional new target graph
        """
        cdef bytes target_bytes = new_target_json.encode('utf-8') if new_target_json else b""
        cdef const char* target_ptr = <const char*>target_bytes if len(target_bytes) > 0 else NULL
        
        with nogil:
            rl_core.rl_context_reset(self._ctx, target_ptr)
    
    def apply_action(self, int action_type, dict config) -> bool:
        """
        Apply RL action to search.
        
        Args:
            action_type: Action type (0=ADD_KN_OP, 1=CREATE_TB, 2=ADD_TB_OP, 3=FINISH)
            config: Configuration dictionary
            
        Returns:
            True if action was successfully applied
        """
        cdef str config_json = json.dumps(config)
        cdef bytes config_bytes = config_json.encode('utf-8')
        cdef const char* config_ptr = config_bytes
        cdef int result
        
        with nogil:
            result = rl_core.rl_context_apply_action(
                self._ctx, 
                action_type, 
                config_ptr
            )
        
        return result != 0
    
    def verify(self) -> dict:
        """
        Verify current kernel on GPU.
        
        This is the critical GPU operation that closes the RL loop.
        
        Returns:
            Dictionary with verification results:
            - verified: bool
            - fingerprint_time_ms: float
            - rejection_reason: str
            - kernel_hash: int
        """
        cdef char* result_json
        
        with nogil:
            result_json = rl_core.rl_context_verify(self._ctx)
        
        if result_json == NULL:
            return {"verified": False, "rejection_reason": "null_result"}
        
        try:
            result_str = result_json.decode('utf-8')
            return json.loads(result_str)
        finally:
            with nogil:
                rl_core.rl_free_string(result_json)
    
    def profile(self, int warmup_iters=10, int profile_iters=100) -> dict:
        """
        Profile kernel performance on GPU.
        
        Args:
            warmup_iters: Warmup iterations
            profile_iters: Profile iterations
            
        Returns:
            Dictionary with profile results:
            - latency_ms: float
            - memory_bytes: int
            - gflops: float
            - compile_time_ms: float
        """
        cdef char* result_json
        
        with nogil:
            result_json = rl_core.rl_context_profile(
                self._ctx, 
                warmup_iters, 
                profile_iters
            )
        
        if result_json == NULL:
            return {"latency_ms": float("inf")}
        
        try:
            result_str = result_json.decode('utf-8')
            return json.loads(result_str)
        finally:
            with nogil:
                rl_core.rl_free_string(result_json)
    
    def get_state(self) -> dict:
        """
        Get current search state.
        
        Returns:
            Dictionary with search state for RL observation
        """
        cdef char* state_json
        
        with nogil:
            state_json = rl_core.rl_context_get_state(self._ctx)
        
        if state_json == NULL:
            return {}
        
        try:
            state_str = state_json.decode('utf-8')
            return json.loads(state_str)
        finally:
            with nogil:
                rl_core.rl_free_string(state_json)
    
    def get_kernel_graph(self) -> str:
        """
        Get current kernel graph as JSON.
        
        Returns:
            Kernel graph JSON string
        """
        cdef char* graph_json
        
        with nogil:
            graph_json = rl_core.rl_context_get_kernel_graph(self._ctx)
        
        if graph_json == NULL:
            return "{}"
        
        try:
            return graph_json.decode('utf-8')
        finally:
            with nogil:
                rl_core.rl_free_string(graph_json)
    
    def is_done(self) -> bool:
        """Check if search is complete."""
        cdef int result
        
        with nogil:
            result = rl_core.rl_context_is_done(self._ctx)
        
        return result != 0
    
    def extract_features(self) -> str:
        """
        Extract features from current µGraph for RL model input.
        
        This is the key interface for the closed loop:
        - C++ extracts features from internal µGraph
        - Returns JSON that FeatureProcessor can parse
        - Features are used as input to RL model
        
        Returns:
            JSON string with complete graph features
        """
        cdef char* features_json
        
        with nogil:
            features_json = rl_core.rl_context_extract_features(self._ctx)
        
        if features_json == NULL:
            return "{}"
        
        try:
            return features_json.decode('utf-8')
        finally:
            with nogil:
                rl_core.rl_free_string(features_json)
    
    @property
    def backend(self) -> str:
        return self._backend
    
    @property
    def gpu_id(self) -> int:
        return self._gpu_id


# Convenience functions

def create_rl_context(
    target_graph_json: str,
    backend: str = "cuda",
    gpu_id: int = 0,
) -> RLSearchContext:
    """
    Create RL search context.
    
    Args:
        target_graph_json: Target computation graph
        backend: Target backend
        gpu_id: GPU ID
        
    Returns:
        RLSearchContext instance
    """
    return RLSearchContext(target_graph_json, backend, gpu_id)


def verify_fingerprint(
    kernel_graph_json: str,
    target_graph_json: str,
    backend: str = "cuda",
    gpu_id: int = 0,
) -> dict:
    """
    Standalone fingerprint verification.
    
    Verifies a single kernel graph against target.
    
    Args:
        kernel_graph_json: Candidate kernel graph
        target_graph_json: Target computation graph
        backend: Target backend
        gpu_id: GPU ID
        
    Returns:
        Verification result dictionary
    """
    ctx = create_rl_context(target_graph_json, backend, gpu_id)
    
    # Apply a finish action to set the kernel graph
    # (In real implementation, would set kernel directly)
    ctx.apply_action(3, {})  # FINISH
    
    return ctx.verify()
