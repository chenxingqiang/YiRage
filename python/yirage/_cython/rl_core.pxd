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
Cython declaration for RL interface C functions.
"""

cdef extern from "search/rl_interface.h" namespace "yirage::rl_interface":
    # C interface functions
    void* rl_context_create(const char* target_graph_json,
                            const char* backend,
                            int gpu_id) nogil
    
    void rl_context_destroy(void* ctx) nogil
    
    void rl_context_reset(void* ctx, const char* new_target_json) nogil
    
    int rl_context_apply_action(void* ctx, int action_type, 
                                 const char* config_json) nogil
    
    char* rl_context_verify(void* ctx) nogil
    
    char* rl_context_profile(void* ctx, int warmup_iters, 
                              int profile_iters) nogil
    
    char* rl_context_get_state(void* ctx) nogil
    
    char* rl_context_get_kernel_graph(void* ctx) nogil
    
    int rl_context_is_done(void* ctx) nogil
    
    char* rl_context_extract_features(void* ctx) nogil
    
    void rl_free_string(char* str) nogil
