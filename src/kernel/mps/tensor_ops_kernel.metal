/* Copyright 2025 Chen Xingqiang (YiRage Project)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Metal Tensor Operations for Apple Silicon
 */

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// Fill Operations
// =============================================================================

kernel void fill_f32(
    device float* data [[buffer(0)]],
    constant float& value [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < size) {
        data[gid] = value;
    }
}

kernel void fill_f16(
    device half* data [[buffer(0)]],
    constant float& value [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < size) {
        data[gid] = half(value);
    }
}

kernel void fill_f16_vec4(
    device half4* data [[buffer(0)]],
    constant float& value [[buffer(1)]],
    constant uint& size_vec4 [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < size_vec4) {
        data[gid] = half4(value);
    }
}

// =============================================================================
// Copy Operations
// =============================================================================

kernel void copy_f32(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < size) {
        dst[gid] = src[gid];
    }
}

kernel void copy_f16(
    device const half* src [[buffer(0)]],
    device half* dst [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < size) {
        dst[gid] = src[gid];
    }
}

// =============================================================================
// Transpose Operations
// =============================================================================

struct TransposeParams {
    uint rows;
    uint cols;
};

kernel void transpose_2d_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant TransposeParams& params [[buffer(2)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tg_id [[threadgroup_position_in_grid]]
) {
    constexpr uint TILE_SIZE = 32;
    
    threadgroup float tile[TILE_SIZE][TILE_SIZE + 1];  // +1 to avoid bank conflicts
    
    uint x = tg_id.x * TILE_SIZE + tid.x;
    uint y = tg_id.y * TILE_SIZE + tid.y;
    
    // Load tile (coalesced read)
    if (x < params.cols && y < params.rows) {
        tile[tid.y][tid.x] = input[y * params.cols + x];
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Transposed position
    x = tg_id.y * TILE_SIZE + tid.x;
    y = tg_id.x * TILE_SIZE + tid.y;
    
    // Store transposed (coalesced write)
    if (x < params.rows && y < params.cols) {
        output[y * params.rows + x] = tile[tid.x][tid.y];
    }
}

kernel void transpose_2d_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant TransposeParams& params [[buffer(2)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tg_id [[threadgroup_position_in_grid]]
) {
    constexpr uint TILE_SIZE = 32;
    
    threadgroup half tile[TILE_SIZE][TILE_SIZE + 2];
    
    uint x = tg_id.x * TILE_SIZE + tid.x;
    uint y = tg_id.y * TILE_SIZE + tid.y;
    
    if (x < params.cols && y < params.rows) {
        tile[tid.y][tid.x] = input[y * params.cols + x];
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    x = tg_id.y * TILE_SIZE + tid.x;
    y = tg_id.x * TILE_SIZE + tid.y;
    
    if (x < params.rows && y < params.cols) {
        output[y * params.rows + x] = tile[tid.x][tid.y];
    }
}

// =============================================================================
// Batch Transpose (for attention)
// =============================================================================

struct BatchTransposeParams {
    uint batch_size;
    uint rows;
    uint cols;
};

kernel void batch_transpose_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant BatchTransposeParams& params [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint batch = gid.z;
    uint row = gid.y;
    uint col = gid.x;
    
    if (batch >= params.batch_size || row >= params.rows || col >= params.cols) return;
    
    uint in_idx = batch * params.rows * params.cols + row * params.cols + col;
    uint out_idx = batch * params.cols * params.rows + col * params.rows + row;
    
    output[out_idx] = input[in_idx];
}

// =============================================================================
// Reshape/View Operations
// =============================================================================

struct PermuteParams {
    uint d0, d1, d2, d3;
    uint s0, s1, s2, s3;  // Output strides
    uint perm0, perm1, perm2, perm3;
};

kernel void permute_4d_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant PermuteParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = params.d0 * params.d1 * params.d2 * params.d3;
    if (gid >= total) return;
    
    // Compute input indices
    uint i0 = gid / (params.d1 * params.d2 * params.d3);
    uint rem = gid % (params.d1 * params.d2 * params.d3);
    uint i1 = rem / (params.d2 * params.d3);
    rem = rem % (params.d2 * params.d3);
    uint i2 = rem / params.d3;
    uint i3 = rem % params.d3;
    
    // Map to output indices based on permutation
    uint indices[4] = {i0, i1, i2, i3};
    uint out_idx = indices[params.perm0] * params.s0 + 
                   indices[params.perm1] * params.s1 +
                   indices[params.perm2] * params.s2 + 
                   indices[params.perm3] * params.s3;
    
    output[out_idx] = input[gid];
}

// =============================================================================
// Concatenate
// =============================================================================

struct ConcatParams {
    uint dim0_a, dim1, total_dim0;
    uint offset;
};

kernel void concat_dim0_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant ConcatParams& params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;
    
    if (row >= params.dim0_a || col >= params.dim1) return;
    
    uint in_idx = row * params.dim1 + col;
    uint out_idx = (row + params.offset) * params.dim1 + col;
    
    output[out_idx] = input[in_idx];
}

// =============================================================================
// Slice/Gather
// =============================================================================

struct SliceParams {
    uint start0, end0;
    uint start1, end1;
    uint in_stride0, in_stride1;
    uint out_stride0, out_stride1;
};

kernel void slice_2d_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant SliceParams& params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint out_row = gid.y;
    uint out_col = gid.x;
    
    uint height = params.end0 - params.start0;
    uint width = params.end1 - params.start1;
    
    if (out_row >= height || out_col >= width) return;
    
    uint in_row = params.start0 + out_row;
    uint in_col = params.start1 + out_col;
    
    uint in_idx = in_row * params.in_stride0 + in_col * params.in_stride1;
    uint out_idx = out_row * params.out_stride0 + out_col * params.out_stride1;
    
    output[out_idx] = input[in_idx];
}

kernel void gather_f16(
    device const half* input [[buffer(0)]],
    device const int* indices [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& num_indices [[buffer(3)]],
    constant uint& dim [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint idx_pos = gid.y;
    uint d = gid.x;
    
    if (idx_pos >= num_indices || d >= dim) return;
    
    int src_idx = indices[idx_pos];
    if (src_idx < 0) return;
    
    output[idx_pos * dim + d] = input[src_idx * dim + d];
}
