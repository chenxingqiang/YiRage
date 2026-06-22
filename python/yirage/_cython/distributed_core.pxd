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
Cython declarations for distributed search C++ API.

This file declares the C++ interfaces for:
1. SearchPartition - Search space partitioning
2. SearchFeedback - RL training data collection  
3. Partitioned search C API - Direct C++ search calls
"""

from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool
from libc.stddef cimport size_t

from CCore cimport CppKNGraph, dim3, int3


# =============================================================================
# Search Partition Structures
# =============================================================================

cdef extern from "search/distributed/search_partition.h" namespace "yirage::search":
    
    cdef enum PartitionStrategy:
        BY_GRID_DIM
        BY_BLOCK_DIM
        BY_CONFIG_HASH
        ROUND_ROBIN
    
    cdef cppclass PartitionConfig:
        int num_partitions
        PartitionStrategy strategy
        bool balance_by_estimate
        
        string to_json() const
        
        @staticmethod
        PartitionConfig from_json(const string& j)
    
    cdef cppclass SearchPartition:
        int partition_id
        int total_partitions
        vector[dim3] grid_dim_range
        vector[dim3] block_dim_range
        vector[int3] imap_range
        vector[int3] omap_range
        vector[int] fmap_range
        vector[int] frange_range
        size_t estimated_candidates
        
        bool contains(dim3 grid_dim, dim3 block_dim) const
        string to_json() const
        
        @staticmethod
        SearchPartition from_json(const string& j)


# =============================================================================
# Search Feedback Structures
# =============================================================================

cdef extern from "search/distributed/search_feedback.h" namespace "yirage::search":
    
    cdef cppclass CandidateInfo:
        int candidate_id
        dim3 grid_dim
        dim3 block_dim
        vector[int3] imaps
        int3 omap
        int frange
        int search_depth
        int operator_count
        int kernel_level_ops
        int threadblock_level_ops
        bool verified
        double fingerprint_time_ms
        double estimated_performance_ms
        string rejection_reason
        double evaluation_time_ms
        
        string to_json() const
        
        @staticmethod
        CandidateInfo from_json(const string& j)
    
    cdef cppclass SearchFeedback:
        int partition_id
        int total_partitions
        vector[CandidateInfo] candidates
        vector[int] valid_candidate_ids
        int total_states_explored
        int valid_graphs_found
        int candidates_generated
        int candidates_verified
        int candidates_rejected
        double search_time_seconds
        double verification_time_seconds
        double generation_time_seconds
        double best_performance_ms
        int best_candidate_id
        
        void add_candidate(const CandidateInfo& info)
        void mark_verified(int candidate_id, double performance_ms)
        string to_json() const
        
        @staticmethod
        SearchFeedback from_json(const string& j)
        
        @staticmethod
        SearchFeedback merge(const vector[SearchFeedback]& feedbacks)
        
        string get_summary() const
    
    cdef cppclass TrainingSample:
        double reward
        bool done
        bool has_next_state
        
        string to_json() const
    
    # Training sample extraction
    vector[TrainingSample] extract_training_samples(
        const SearchFeedback& feedback,
        double validity_reward,
        double invalid_penalty,
        double depth_penalty
    )


# =============================================================================
# Partitioned Search C API
# =============================================================================

cdef extern from "search/distributed/partitioned_generator.h" namespace "yirage::search::partitioned_search_c":
    
    # Create search partitions from configuration
    # Returns JSON string (caller must free with free_json_string)
    char* create_partitions(int num_partitions, const char* config_json) nogil
    
    # Execute search on a single partition
    # Returns number of graphs generated
    int search_partition(
        const CppKNGraph* input_graph,
        const char* partition_json,
        const char* config_json,
        bool collect_feedback,
        int max_num_graphs,
        CppKNGraph** new_graphs,
        char** feedback_json
    ) nogil
    
    # Free JSON string allocated by C++ functions
    void free_json_string(char* json_str) nogil


# =============================================================================
# RL Interface (from rl_interface.h)
# =============================================================================

cdef extern from "search/rl_interface.h" namespace "yirage::rl_interface":
    
    # Context creation/destruction
    void* rl_context_create(
        const char* target_graph_json,
        const char* backend,
        int gpu_id
    ) nogil
    
    void rl_context_destroy(void* ctx) nogil
    
    void rl_context_reset(void* ctx, const char* new_target_json) nogil
    
    # Action application
    int rl_context_apply_action(
        void* ctx, 
        int action_type, 
        const char* config_json
    ) nogil
    
    # GPU verification (critical for RL closed loop)
    char* rl_context_verify(void* ctx) nogil
    
    # GPU profiling
    char* rl_context_profile(
        void* ctx, 
        int warmup_iters, 
        int profile_iters
    ) nogil
    
    # State access
    char* rl_context_get_state(void* ctx) nogil
    char* rl_context_get_kernel_graph(void* ctx) nogil
    int rl_context_is_done(void* ctx) nogil
    char* rl_context_extract_features(void* ctx) nogil
    
    # Memory management
    void rl_free_string(char* str) nogil
