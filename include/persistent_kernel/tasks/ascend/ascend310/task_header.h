/* Copyright 2025 YiRage Team
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
 */

#pragma once

/**
 * @file task_header.h
 * @brief Ascend 310 Optimized TBE Kernels (Edge Inference)
 *
 * Ascend 310 (2019) characteristics:
 * - 2 AI Cores
 * - 128KB L1 buffer per AI Core
 * - 16x16 Cube unit
 * - 8GB HBM
 * - Focus: Edge inference, low power
 *
 * Optimization strategy:
 * - Small tiles for limited L1
 * - INT8/FP16 preferred
 * - Minimize memory traffic
 */

#include "../common/ascend_common.h"

namespace yirage {
namespace persistent_kernel {
namespace ascend {
namespace ascend310 {

constexpr int ASCEND310_AI_CORES = 2;
constexpr int ASCEND310_L1_KB = 128;
constexpr int ASCEND310_CUBE_SIZE = 16;

/**
 * TBE (Tensor Boost Engine) kernel source for Ascend 310
 * Uses CANN/TBE DSL-like pseudo-code representation
 */
constexpr const char* ASCEND310_KERNEL_SOURCE = R"(
// =============================================================================
// Ascend 310 Optimized Kernels (TBE/CANN)
// Edge inference focused - 2 AI Cores, 128KB L1, 16x16 Cube
// =============================================================================

// Ascend TBE includes
#include "kernel_operator.h"
using namespace AscendC;

// =============================================================================
// Ascend 310 RMSNorm - Single AI Core optimized
// =============================================================================
class RmsNorm310 {
public:
    __aicore__ inline RmsNorm310() {}
    
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR weight, GM_ADDR output,
                                 int num_tokens, int hidden_dim, float eps) {
        this->num_tokens = num_tokens;
        this->hidden_dim = hidden_dim;
        this->eps = eps;
        
        // Initialize Global Memory tensors
        inputGm.SetGlobalBuffer((__gm__ half*)input);
        weightGm.SetGlobalBuffer((__gm__ half*)weight);
        outputGm.SetGlobalBuffer((__gm__ half*)output);
        
        // Calculate tile sizes for 128KB L1
        // Need: input tile + weight + output tile
        // Max tile: ~32KB each = 16K elements in FP16
        tileLength = 4096;  // Conservative for 310
        
        // Allocate L1 buffers
        pipe.InitBuffer(inputQueue, 1, tileLength * sizeof(half));
        pipe.InitBuffer(weightQueue, 1, hidden_dim * sizeof(half));
        pipe.InitBuffer(outputQueue, 1, tileLength * sizeof(half));
    }
    
    __aicore__ inline void Process() {
        int tokenIdx = GetBlockIdx();
        if (tokenIdx >= num_tokens) return;
        
        // Copy weight to L1 once
        LocalTensor<half> weightLocal = weightQueue.AllocTensor<half>();
        DataCopy(weightLocal, weightGm, hidden_dim);
        weightQueue.EnQue(weightLocal);
        
        // Process token
        LocalTensor<half> inputLocal = inputQueue.AllocTensor<half>();
        LocalTensor<half> outputLocal = outputQueue.AllocTensor<half>();
        
        int offset = tokenIdx * hidden_dim;
        DataCopy(inputLocal, inputGm[offset], hidden_dim);
        inputQueue.EnQue(inputLocal);
        
        // Compute sum of squares
        inputLocal = inputQueue.DeQue<half>();
        weightLocal = weightQueue.DeQue<half>();
        
        // Use Vector unit for element-wise operations
        LocalTensor<float> sumSq = pipe.AllocTensor<float>();
        
        // Square and reduce
        Mul(sumSq, inputLocal, inputLocal, hidden_dim);
        float totalSum = 0.0f;
        for (int i = 0; i < hidden_dim; i++) {
            totalSum += (float)sumSq.GetValue(i);
        }
        
        float invRms = 1.0f / sqrtf(totalSum / hidden_dim + eps);
        
        // Apply normalization: out = in * inv_rms * weight
        for (int i = 0; i < hidden_dim; i++) {
            float val = (float)inputLocal.GetValue(i) * invRms * (float)weightLocal.GetValue(i);
            outputLocal.SetValue(i, (half)val);
        }
        
        outputQueue.EnQue(outputLocal);
        outputLocal = outputQueue.DeQue<half>();
        DataCopy(outputGm[offset], outputLocal, hidden_dim);
        
        pipe.FreeTensor(sumSq);
    }
    
private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inputQueue;
    TQue<QuePosition::VECIN, 1> weightQueue;
    TQue<QuePosition::VECOUT, 1> outputQueue;
    GlobalTensor<half> inputGm;
    GlobalTensor<half> weightGm;
    GlobalTensor<half> outputGm;
    
    int num_tokens;
    int hidden_dim;
    float eps;
    int tileLength;
};

// =============================================================================
// Ascend 310 MatMul - 16x16 Cube optimized
// =============================================================================
class MatMul310 {
public:
    __aicore__ inline MatMul310() {}
    
    __aicore__ inline void Init(GM_ADDR A, GM_ADDR B, GM_ADDR C,
                                 int M, int N, int K) {
        this->M = M;
        this->N = N;
        this->K = K;
        
        AGm.SetGlobalBuffer((__gm__ half*)A);
        BGm.SetGlobalBuffer((__gm__ half*)B);
        CGm.SetGlobalBuffer((__gm__ half*)C);
        
        // 310: Small tiles due to limited L1 (128KB)
        // 32x32 tile for A, 32x32 for B, 32x32 for C
        tileM = 32;
        tileN = 32;
        tileK = 16;  // Match Cube size
        
        pipe.InitBuffer(aQueue, 2, tileM * tileK * sizeof(half));
        pipe.InitBuffer(bQueue, 2, tileK * tileN * sizeof(half));
        pipe.InitBuffer(cQueue, 2, tileM * tileN * sizeof(float));
    }
    
    __aicore__ inline void Process() {
        int blockM = GetBlockIdx() / ((N + tileN - 1) / tileN);
        int blockN = GetBlockIdx() % ((N + tileN - 1) / tileN);
        
        int rowStart = blockM * tileM;
        int colStart = blockN * tileN;
        
        if (rowStart >= M || colStart >= N) return;
        
        // Initialize accumulator
        LocalTensor<float> accum = cQueue.AllocTensor<float>();
        for (int i = 0; i < tileM * tileN; i++) {
            accum.SetValue(i, 0.0f);
        }
        
        // Tile over K dimension
        for (int k = 0; k < K; k += tileK) {
            // Load A tile
            LocalTensor<half> aLocal = aQueue.AllocTensor<half>();
            for (int i = 0; i < tileM && rowStart + i < M; i++) {
                for (int j = 0; j < tileK && k + j < K; j++) {
                    aLocal.SetValue(i * tileK + j, 
                        AGm.GetValue((rowStart + i) * K + k + j));
                }
            }
            aQueue.EnQue(aLocal);
            
            // Load B tile
            LocalTensor<half> bLocal = bQueue.AllocTensor<half>();
            for (int i = 0; i < tileK && k + i < K; i++) {
                for (int j = 0; j < tileN && colStart + j < N; j++) {
                    bLocal.SetValue(i * tileN + j,
                        BGm.GetValue((k + i) * N + colStart + j));
                }
            }
            bQueue.EnQue(bLocal);
            
            // Cube matmul: C += A * B
            aLocal = aQueue.DeQue<half>();
            bLocal = bQueue.DeQue<half>();
            
            // Use Cube unit (16x16 native)
            Mmad(accum, aLocal, bLocal, tileM, tileN, tileK);
            
            aQueue.FreeTensor(aLocal);
            bQueue.FreeTensor(bLocal);
        }
        
        // Store result
        cQueue.EnQue(accum);
        accum = cQueue.DeQue<float>();
        
        for (int i = 0; i < tileM && rowStart + i < M; i++) {
            for (int j = 0; j < tileN && colStart + j < N; j++) {
                CGm.SetValue((rowStart + i) * N + colStart + j,
                    (half)accum.GetValue(i * tileN + j));
            }
        }
        
        cQueue.FreeTensor(accum);
    }
    
private:
    TPipe pipe;
    TQue<QuePosition::A, 2> aQueue;
    TQue<QuePosition::B, 2> bQueue;
    TQue<QuePosition::CO1, 2> cQueue;
    GlobalTensor<half> AGm;
    GlobalTensor<half> BGm;
    GlobalTensor<half> CGm;
    
    int M, N, K;
    int tileM, tileN, tileK;
};

// =============================================================================
// Ascend 310 SiLU - Vector unit optimized
// =============================================================================
class SiluMul310 {
public:
    __aicore__ inline SiluMul310() {}
    
    __aicore__ inline void Init(GM_ADDR gate, GM_ADDR up, GM_ADDR output, int size) {
        gateGm.SetGlobalBuffer((__gm__ half*)gate);
        upGm.SetGlobalBuffer((__gm__ half*)up);
        outputGm.SetGlobalBuffer((__gm__ half*)output);
        this->size = size;
        
        // Tile size: 128 elements at a time (vector width)
        tileSize = 128;
        pipe.InitBuffer(gateQueue, 2, tileSize * sizeof(half));
        pipe.InitBuffer(upQueue, 2, tileSize * sizeof(half));
        pipe.InitBuffer(outQueue, 2, tileSize * sizeof(half));
    }
    
    __aicore__ inline void Process() {
        int tileIdx = GetBlockIdx();
        int offset = tileIdx * tileSize;
        
        if (offset >= size) return;
        
        int actualSize = min(tileSize, size - offset);
        
        // Load gate and up
        LocalTensor<half> gateLocal = gateQueue.AllocTensor<half>();
        LocalTensor<half> upLocal = upQueue.AllocTensor<half>();
        LocalTensor<half> outLocal = outQueue.AllocTensor<half>();
        
        DataCopy(gateLocal, gateGm[offset], actualSize);
        DataCopy(upLocal, upGm[offset], actualSize);
        
        gateQueue.EnQue(gateLocal);
        upQueue.EnQue(upLocal);
        
        gateLocal = gateQueue.DeQue<half>();
        upLocal = upQueue.DeQue<half>();
        
        // SiLU(gate) * up
        for (int i = 0; i < actualSize; i++) {
            float g = (float)gateLocal.GetValue(i);
            float u = (float)upLocal.GetValue(i);
            float sigmoid = 1.0f / (1.0f + expf(-g));
            float result = g * sigmoid * u;
            outLocal.SetValue(i, (half)result);
        }
        
        outQueue.EnQue(outLocal);
        outLocal = outQueue.DeQue<half>();
        DataCopy(outputGm[offset], outLocal, actualSize);
    }
    
private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 2> gateQueue;
    TQue<QuePosition::VECIN, 2> upQueue;
    TQue<QuePosition::VECOUT, 2> outQueue;
    GlobalTensor<half> gateGm;
    GlobalTensor<half> upGm;
    GlobalTensor<half> outputGm;
    
    int size;
    int tileSize;
};

// Kernel entry points
extern "C" __global__ __aicore__ void rms_norm_310(GM_ADDR input, GM_ADDR weight,
                                                    GM_ADDR output, int num_tokens,
                                                    int hidden_dim, float eps) {
    RmsNorm310 op;
    op.Init(input, weight, output, num_tokens, hidden_dim, eps);
    op.Process();
}

extern "C" __global__ __aicore__ void matmul_310(GM_ADDR A, GM_ADDR B, GM_ADDR C,
                                                  int M, int N, int K) {
    MatMul310 op;
    op.Init(A, B, C, M, N, K);
    op.Process();
}

extern "C" __global__ __aicore__ void silu_mul_310(GM_ADDR gate, GM_ADDR up,
                                                    GM_ADDR output, int size) {
    SiluMul310 op;
    op.Init(gate, up, output, size);
    op.Process();
}
)";

}  // namespace ascend310
}  // namespace ascend
}  // namespace persistent_kernel
}  // namespace yirage
