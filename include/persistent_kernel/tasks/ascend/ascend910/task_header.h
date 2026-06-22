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
 * @brief Ascend 910 Optimized TBE Kernels (First Training Chip)
 *
 * Ascend 910/910A (2019-2020) characteristics:
 * - 32 AI Cores
 * - 256KB L1 buffer per AI Core
 * - 16x16 Cube unit with FP32 support
 * - 32GB HBM
 * - 256 TFLOPS FP16
 *
 * Optimization strategy:
 * - Full AI Core parallelization
 * - FP32 accumulation for training
 * - Double buffering with async DMA
 */

#include "../common/ascend_common.h"

namespace yirage {
namespace persistent_kernel {
namespace ascend {
namespace ascend910 {

constexpr int ASCEND910_AI_CORES = 32;
constexpr int ASCEND910_L1_KB = 256;
constexpr int ASCEND910_CUBE_SIZE = 16;

constexpr const char* ASCEND910_KERNEL_SOURCE = R"(
// =============================================================================
// Ascend 910 Optimized Kernels (TBE/CANN)
// First training chip - 32 AI Cores, 256KB L1, FP32 Cube
// =============================================================================

#include "kernel_operator.h"
using namespace AscendC;

// =============================================================================
// Ascend 910 RMSNorm - 32 AI Core parallel
// =============================================================================
class RmsNorm910 {
public:
    __aicore__ inline RmsNorm910() {}
    
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR weight, GM_ADDR output,
                                 int num_tokens, int hidden_dim, float eps) {
        this->num_tokens = num_tokens;
        this->hidden_dim = hidden_dim;
        this->eps = eps;
        
        inputGm.SetGlobalBuffer((__gm__ half*)input);
        weightGm.SetGlobalBuffer((__gm__ half*)weight);
        outputGm.SetGlobalBuffer((__gm__ half*)output);
        
        // 910: Medium-large tiles with double buffering
        tileLength = 8192;
        
        pipe.InitBuffer(inputQueue, 2, tileLength * sizeof(half));
        pipe.InitBuffer(weightQueue, 1, hidden_dim * sizeof(half));
        pipe.InitBuffer(outputQueue, 2, tileLength * sizeof(half));
        pipe.InitBuffer(workQueue, 1, hidden_dim * sizeof(float));
    }
    
    __aicore__ inline void Process() {
        // Each AI Core processes one token
        int tokenIdx = GetBlockIdx();
        if (tokenIdx >= num_tokens) return;
        
        int offset = tokenIdx * hidden_dim;
        
        // Load weight (shared)
        LocalTensor<half> weightLocal = weightQueue.AllocTensor<half>();
        DataCopyPad(weightLocal, weightGm, hidden_dim);
        weightQueue.EnQue(weightLocal);
        
        // Load input with async DMA
        LocalTensor<half> inputLocal = inputQueue.AllocTensor<half>();
        DataCopyPad(inputLocal, inputGm[offset], hidden_dim);
        inputQueue.EnQue(inputLocal);
        
        inputLocal = inputQueue.DeQue<half>();
        weightLocal = weightQueue.DeQue<half>();
        
        // FP32 accumulation for numerical stability
        LocalTensor<float> work = workQueue.AllocTensor<float>();
        
        // Vectorized square with 256-wide vector unit
        Cast(work, inputLocal, RoundMode::CAST_NONE, hidden_dim);
        Mul(work, work, work, hidden_dim);
        
        // Parallel reduction within AI Core
        float totalSum = 0.0f;
        int vecWidth = 256;
        for (int i = 0; i < hidden_dim; i += vecWidth) {
            int width = min(vecWidth, hidden_dim - i);
            float partialSum = 0.0f;
            ReduceSum(partialSum, work[i], width);
            totalSum += partialSum;
        }
        
        float invRms = rsqrtf(totalSum / hidden_dim + eps);
        
        // Vectorized normalization
        LocalTensor<half> outputLocal = outputQueue.AllocTensor<half>();
        
        // out = in * inv_rms * weight
        for (int i = 0; i < hidden_dim; i += vecWidth) {
            int width = min(vecWidth, hidden_dim - i);
            
            LocalTensor<float> tempIn = pipe.AllocTensor<float>();
            LocalTensor<float> tempW = pipe.AllocTensor<float>();
            LocalTensor<float> tempOut = pipe.AllocTensor<float>();
            
            Cast(tempIn, inputLocal[i], RoundMode::CAST_NONE, width);
            Cast(tempW, weightLocal[i], RoundMode::CAST_NONE, width);
            
            Muls(tempIn, tempIn, invRms, width);
            Mul(tempOut, tempIn, tempW, width);
            
            Cast(outputLocal[i], tempOut, RoundMode::CAST_ROUND, width);
            
            pipe.FreeTensor(tempIn);
            pipe.FreeTensor(tempW);
            pipe.FreeTensor(tempOut);
        }
        
        outputQueue.EnQue(outputLocal);
        outputLocal = outputQueue.DeQue<half>();
        DataCopyPad(outputGm[offset], outputLocal, hidden_dim);
        
        workQueue.FreeTensor(work);
    }
    
private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 2> inputQueue;
    TQue<QuePosition::VECIN, 1> weightQueue;
    TQue<QuePosition::VECOUT, 2> outputQueue;
    TQue<QuePosition::VECIN, 1> workQueue;
    GlobalTensor<half> inputGm;
    GlobalTensor<half> weightGm;
    GlobalTensor<half> outputGm;
    
    int num_tokens;
    int hidden_dim;
    float eps;
    int tileLength;
};

// =============================================================================
// Ascend 910 GEMM - 16x16 Cube with FP32 accumulation
// =============================================================================
class MatMul910 {
public:
    __aicore__ inline MatMul910() {}
    
    __aicore__ inline void Init(GM_ADDR A, GM_ADDR B, GM_ADDR C,
                                 int M, int N, int K, bool accumulate = false) {
        this->M = M;
        this->N = N;
        this->K = K;
        this->accumulate = accumulate;
        
        AGm.SetGlobalBuffer((__gm__ half*)A);
        BGm.SetGlobalBuffer((__gm__ half*)B);
        CGm.SetGlobalBuffer((__gm__ half*)C);
        
        // 910: 64x64 output tiles with 16x16 Cube
        tileM = 64;
        tileN = 64;
        tileK = 16;
        
        // Double buffer for pipelining
        pipe.InitBuffer(aQueue, 2, tileM * tileK * sizeof(half));
        pipe.InitBuffer(bQueue, 2, tileK * tileN * sizeof(half));
        pipe.InitBuffer(cQueue, 2, tileM * tileN * sizeof(float));
    }
    
    __aicore__ inline void Process() {
        int numBlocksN = (N + tileN - 1) / tileN;
        int blockM = GetBlockIdx() / numBlocksN;
        int blockN = GetBlockIdx() % numBlocksN;
        
        int rowStart = blockM * tileM;
        int colStart = blockN * tileN;
        
        if (rowStart >= M || colStart >= N) return;
        
        // Initialize or load accumulator
        LocalTensor<float> accum = cQueue.AllocTensor<float>();
        
        if (accumulate) {
            // Load existing C for accumulation
            for (int m = 0; m < tileM && rowStart + m < M; m++) {
                for (int n = 0; n < tileN && colStart + n < N; n++) {
                    accum.SetValue(m * tileN + n,
                        (float)CGm.GetValue((rowStart + m) * N + colStart + n));
                }
            }
        } else {
            for (int i = 0; i < tileM * tileN; i++) {
                accum.SetValue(i, 0.0f);
            }
        }
        
        int numKTiles = (K + tileK - 1) / tileK;
        
        for (int kTile = 0; kTile < numKTiles; kTile++) {
            int kStart = kTile * tileK;
            int actualK = min(tileK, K - kStart);
            
            // Load A tile
            LocalTensor<half> aLocal = aQueue.AllocTensor<half>();
            for (int m = 0; m < tileM && rowStart + m < M; m++) {
                DataCopyPad(aLocal[m * tileK], AGm[(rowStart + m) * K + kStart], actualK);
            }
            aQueue.EnQue(aLocal);
            
            // Load B tile
            LocalTensor<half> bLocal = bQueue.AllocTensor<half>();
            for (int k = 0; k < actualK; k++) {
                int actualN = min(tileN, N - colStart);
                DataCopyPad(bLocal[k * tileN], BGm[(kStart + k) * N + colStart], actualN);
            }
            bQueue.EnQue(bLocal);
            
            aLocal = aQueue.DeQue<half>();
            bLocal = bQueue.DeQue<half>();
            
            // Cube matmul with FP32 accumulation
            // 910 supports FP32 Cube operations
            Mmad(accum, aLocal, bLocal, tileM, tileN, actualK, MmadMode::FP32);
            
            aQueue.FreeTensor(aLocal);
            bQueue.FreeTensor(bLocal);
        }
        
        // Store result (convert to FP16)
        cQueue.EnQue(accum);
        accum = cQueue.DeQue<float>();
        
        for (int m = 0; m < tileM && rowStart + m < M; m++) {
            int actualN = min(tileN, N - colStart);
            for (int n = 0; n < actualN; n++) {
                CGm.SetValue((rowStart + m) * N + colStart + n,
                    (half)accum.GetValue(m * tileN + n));
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
    bool accumulate;
};

// =============================================================================
// Ascend 910 SiLU+Mul (SwiGLU) - Vectorized
// =============================================================================
class SiluMul910 {
public:
    __aicore__ inline SiluMul910() {}
    
    __aicore__ inline void Init(GM_ADDR gate, GM_ADDR up, GM_ADDR output, int size) {
        gateGm.SetGlobalBuffer((__gm__ half*)gate);
        upGm.SetGlobalBuffer((__gm__ half*)up);
        outputGm.SetGlobalBuffer((__gm__ half*)output);
        this->size = size;
        
        // 256-element tiles for vector unit
        tileSize = 256;
        pipe.InitBuffer(gateQueue, 2, tileSize * sizeof(half));
        pipe.InitBuffer(upQueue, 2, tileSize * sizeof(half));
        pipe.InitBuffer(outQueue, 2, tileSize * sizeof(half));
        pipe.InitBuffer(workQueue, 1, tileSize * sizeof(float));
    }
    
    __aicore__ inline void Process() {
        int tileIdx = GetBlockIdx();
        int offset = tileIdx * tileSize;
        
        if (offset >= size) return;
        
        int actualSize = min(tileSize, size - offset);
        
        // Load gate and up
        LocalTensor<half> gateLocal = gateQueue.AllocTensor<half>();
        LocalTensor<half> upLocal = upQueue.AllocTensor<half>();
        
        DataCopyPad(gateLocal, gateGm[offset], actualSize);
        DataCopyPad(upLocal, upGm[offset], actualSize);
        
        gateQueue.EnQue(gateLocal);
        upQueue.EnQue(upLocal);
        
        gateLocal = gateQueue.DeQue<half>();
        upLocal = upQueue.DeQue<half>();
        
        // Compute SiLU(gate) * up in FP32 for accuracy
        LocalTensor<float> work = workQueue.AllocTensor<float>();
        LocalTensor<float> gateF = pipe.AllocTensor<float>();
        LocalTensor<float> upF = pipe.AllocTensor<float>();
        
        Cast(gateF, gateLocal, RoundMode::CAST_NONE, actualSize);
        Cast(upF, upLocal, RoundMode::CAST_NONE, actualSize);
        
        // SiLU: x * sigmoid(x) = x / (1 + exp(-x))
        // Vectorized sigmoid
        Neg(work, gateF, actualSize);
        Exp(work, work, actualSize);
        Adds(work, work, 1.0f, actualSize);
        Div(work, gateF, work, actualSize);
        
        // Multiply with up
        Mul(work, work, upF, actualSize);
        
        // Convert back to FP16 and store
        LocalTensor<half> outLocal = outQueue.AllocTensor<half>();
        Cast(outLocal, work, RoundMode::CAST_ROUND, actualSize);
        
        outQueue.EnQue(outLocal);
        outLocal = outQueue.DeQue<half>();
        DataCopyPad(outputGm[offset], outLocal, actualSize);
        
        pipe.FreeTensor(gateF);
        pipe.FreeTensor(upF);
        workQueue.FreeTensor(work);
    }
    
private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 2> gateQueue;
    TQue<QuePosition::VECIN, 2> upQueue;
    TQue<QuePosition::VECOUT, 2> outQueue;
    TQue<QuePosition::VECIN, 1> workQueue;
    GlobalTensor<half> gateGm;
    GlobalTensor<half> upGm;
    GlobalTensor<half> outputGm;
    
    int size;
    int tileSize;
};

// Kernel entry points
extern "C" __global__ __aicore__ void rms_norm_910(GM_ADDR input, GM_ADDR weight,
                                                    GM_ADDR output, int num_tokens,
                                                    int hidden_dim, float eps) {
    RmsNorm910 op;
    op.Init(input, weight, output, num_tokens, hidden_dim, eps);
    op.Process();
}

extern "C" __global__ __aicore__ void matmul_910(GM_ADDR A, GM_ADDR B, GM_ADDR C,
                                                  int M, int N, int K) {
    MatMul910 op;
    op.Init(A, B, C, M, N, K);
    op.Process();
}

extern "C" __global__ __aicore__ void silu_mul_910(GM_ADDR gate, GM_ADDR up,
                                                    GM_ADDR output, int size) {
    SiluMul910 op;
    op.Init(gate, up, output, size);
    op.Process();
}
)";

}  // namespace ascend910
}  // namespace ascend
}  // namespace persistent_kernel
}  // namespace yirage
