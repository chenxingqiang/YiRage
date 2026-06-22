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
 * @brief Ascend 910B Optimized TBE Kernels (Enhanced Training)
 *
 * Ascend 910B (2022) characteristics:
 * - 32 AI Cores
 * - 512KB L1 buffer per AI Core (doubled!)
 * - 16x16 Cube with BF16 support
 * - 64GB HBM (doubled!)
 * - 320 TFLOPS FP16
 * - INT4 quantization support
 *
 * Optimization strategy:
 * - Larger tiles with 512KB L1
 * - BF16 for training stability
 * - INT4 for inference quantization
 */

#include "../common/ascend_common.h"

namespace yirage {
namespace persistent_kernel {
namespace ascend {
namespace ascend910b {

constexpr int ASCEND910B_AI_CORES = 32;
constexpr int ASCEND910B_L1_KB = 512;
constexpr int ASCEND910B_CUBE_SIZE = 16;

constexpr const char* ASCEND910B_KERNEL_SOURCE = R"(
// =============================================================================
// Ascend 910B Optimized Kernels (TBE/CANN)
// Enhanced training - 32 AI Cores, 512KB L1, BF16, INT4
// =============================================================================

#include "kernel_operator.h"
using namespace AscendC;

// =============================================================================
// Ascend 910B RMSNorm with BF16
// =============================================================================
class RmsNorm910B {
public:
    __aicore__ inline RmsNorm910B() {}
    
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR weight, GM_ADDR output,
                                 int num_tokens, int hidden_dim, float eps) {
        this->num_tokens = num_tokens;
        this->hidden_dim = hidden_dim;
        this->eps = eps;
        
        inputGm.SetGlobalBuffer((__gm__ bfloat16_t*)input);
        weightGm.SetGlobalBuffer((__gm__ bfloat16_t*)weight);
        outputGm.SetGlobalBuffer((__gm__ bfloat16_t*)output);
        
        // 910B: Larger tiles with 512KB L1
        tileLength = 16384;
        
        pipe.InitBuffer(inputQueue, 2, tileLength * sizeof(bfloat16_t));
        pipe.InitBuffer(weightQueue, 1, hidden_dim * sizeof(bfloat16_t));
        pipe.InitBuffer(outputQueue, 2, tileLength * sizeof(bfloat16_t));
        pipe.InitBuffer(workQueue, 2, hidden_dim * sizeof(float));
    }
    
    __aicore__ inline void Process() {
        int tokenIdx = GetBlockIdx();
        if (tokenIdx >= num_tokens) return;
        
        int offset = tokenIdx * hidden_dim;
        
        // Async weight load
        LocalTensor<bfloat16_t> weightLocal = weightQueue.AllocTensor<bfloat16_t>();
        DataCopyAsync(weightLocal, weightGm, hidden_dim);
        weightQueue.EnQue(weightLocal);
        
        // Async input load
        LocalTensor<bfloat16_t> inputLocal = inputQueue.AllocTensor<bfloat16_t>();
        DataCopyAsync(inputLocal, inputGm[offset], hidden_dim);
        inputQueue.EnQue(inputLocal);
        
        inputLocal = inputQueue.DeQue<bfloat16_t>();
        weightLocal = weightQueue.DeQue<bfloat16_t>();
        
        // FP32 computation for accuracy
        LocalTensor<float> work = workQueue.AllocTensor<float>();
        
        // Vectorized square and sum with 512-wide vector
        Cast(work, inputLocal, RoundMode::CAST_NONE, hidden_dim);
        Mul(work, work, work, hidden_dim);
        
        // Hierarchical reduction for large hidden_dim
        float totalSum = 0.0f;
        int vecWidth = 512;  // 910B vector width
        
        for (int i = 0; i < hidden_dim; i += vecWidth) {
            int width = min(vecWidth, hidden_dim - i);
            float partialSum = 0.0f;
            ReduceSum(partialSum, work[i], width);
            totalSum += partialSum;
        }
        
        float invRms = rsqrtf(totalSum / hidden_dim + eps);
        
        // Vectorized normalization
        LocalTensor<bfloat16_t> outputLocal = outputQueue.AllocTensor<bfloat16_t>();
        LocalTensor<float> work2 = workQueue.AllocTensor<float>();
        
        Cast(work, inputLocal, RoundMode::CAST_NONE, hidden_dim);
        Cast(work2, weightLocal, RoundMode::CAST_NONE, hidden_dim);
        
        Muls(work, work, invRms, hidden_dim);
        Mul(work, work, work2, hidden_dim);
        
        Cast(outputLocal, work, RoundMode::CAST_ROUND, hidden_dim);
        
        outputQueue.EnQue(outputLocal);
        outputLocal = outputQueue.DeQue<bfloat16_t>();
        DataCopyAsync(outputGm[offset], outputLocal, hidden_dim);
        
        workQueue.FreeTensor(work);
        workQueue.FreeTensor(work2);
    }
    
private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 2> inputQueue;
    TQue<QuePosition::VECIN, 1> weightQueue;
    TQue<QuePosition::VECOUT, 2> outputQueue;
    TQue<QuePosition::VECIN, 2> workQueue;
    GlobalTensor<bfloat16_t> inputGm;
    GlobalTensor<bfloat16_t> weightGm;
    GlobalTensor<bfloat16_t> outputGm;
    
    int num_tokens;
    int hidden_dim;
    float eps;
    int tileLength;
};

// =============================================================================
// Ascend 910B GEMM - 32x32 tiles with BF16
// =============================================================================
class MatMul910B {
public:
    __aicore__ inline MatMul910B() {}
    
    __aicore__ inline void Init(GM_ADDR A, GM_ADDR B, GM_ADDR C,
                                 int M, int N, int K) {
        this->M = M;
        this->N = N;
        this->K = K;
        
        AGm.SetGlobalBuffer((__gm__ bfloat16_t*)A);
        BGm.SetGlobalBuffer((__gm__ bfloat16_t*)B);
        CGm.SetGlobalBuffer((__gm__ bfloat16_t*)C);
        
        // 910B: 128x128 output tiles with 512KB L1
        tileM = 128;
        tileN = 128;
        tileK = 32;
        
        // Triple buffer for maximum throughput
        pipe.InitBuffer(aQueue, 3, tileM * tileK * sizeof(bfloat16_t));
        pipe.InitBuffer(bQueue, 3, tileK * tileN * sizeof(bfloat16_t));
        pipe.InitBuffer(cQueue, 2, tileM * tileN * sizeof(float));
    }
    
    __aicore__ inline void Process() {
        int numBlocksN = (N + tileN - 1) / tileN;
        int blockM = GetBlockIdx() / numBlocksN;
        int blockN = GetBlockIdx() % numBlocksN;
        
        int rowStart = blockM * tileM;
        int colStart = blockN * tileN;
        
        if (rowStart >= M || colStart >= N) return;
        
        int actualM = min(tileM, M - rowStart);
        int actualN = min(tileN, N - colStart);
        
        // Initialize accumulator
        LocalTensor<float> accum = cQueue.AllocTensor<float>();
        for (int i = 0; i < tileM * tileN; i++) {
            accum.SetValue(i, 0.0f);
        }
        
        int numKTiles = (K + tileK - 1) / tileK;
        
        // Pipeline: Load-Compute-Store
        // Stage 0: Load first tiles
        LocalTensor<bfloat16_t> aLocal[3];
        LocalTensor<bfloat16_t> bLocal[3];
        
        for (int stage = 0; stage < 2 && stage < numKTiles; stage++) {
            aLocal[stage] = aQueue.AllocTensor<bfloat16_t>();
            bLocal[stage] = bQueue.AllocTensor<bfloat16_t>();
            
            LoadTileAsync(aLocal[stage], bLocal[stage], rowStart, colStart, 
                         stage * tileK, actualM, actualN);
        }
        
        for (int kTile = 0; kTile < numKTiles; kTile++) {
            int loadStage = (kTile + 2) % 3;
            int computeStage = kTile % 3;
            
            // Start loading next tile
            if (kTile + 2 < numKTiles) {
                aLocal[loadStage] = aQueue.AllocTensor<bfloat16_t>();
                bLocal[loadStage] = bQueue.AllocTensor<bfloat16_t>();
                LoadTileAsync(aLocal[loadStage], bLocal[loadStage], 
                             rowStart, colStart, (kTile + 2) * tileK, 
                             actualM, actualN);
            }
            
            // Compute on current tile
            int actualK = min(tileK, K - kTile * tileK);
            
            // Cube matmul with BF16
            Mmad(accum, aLocal[computeStage], bLocal[computeStage], 
                 actualM, actualN, actualK, MmadMode::BF16);
            
            aQueue.FreeTensor(aLocal[computeStage]);
            bQueue.FreeTensor(bLocal[computeStage]);
        }
        
        // Store result
        cQueue.EnQue(accum);
        accum = cQueue.DeQue<float>();
        
        LocalTensor<bfloat16_t> outLocal = pipe.AllocTensor<bfloat16_t>();
        Cast(outLocal, accum, RoundMode::CAST_ROUND, actualM * actualN);
        
        for (int m = 0; m < actualM; m++) {
            DataCopyAsync(CGm[(rowStart + m) * N + colStart], 
                         outLocal[m * tileN], actualN);
        }
        
        pipe.FreeTensor(outLocal);
        cQueue.FreeTensor(accum);
    }
    
private:
    inline void LoadTileAsync(LocalTensor<bfloat16_t>& a, LocalTensor<bfloat16_t>& b,
                              int rowStart, int colStart, int kStart, 
                              int actualM, int actualN) {
        int actualK = min(tileK, K - kStart);
        
        // Load A tile
        for (int m = 0; m < actualM; m++) {
            DataCopyAsync(a[m * tileK], AGm[(rowStart + m) * K + kStart], actualK);
        }
        aQueue.EnQue(a);
        
        // Load B tile
        for (int k = 0; k < actualK; k++) {
            DataCopyAsync(b[k * tileN], BGm[(kStart + k) * N + colStart], actualN);
        }
        bQueue.EnQue(b);
    }
    
    TPipe pipe;
    TQue<QuePosition::A, 3> aQueue;
    TQue<QuePosition::B, 3> bQueue;
    TQue<QuePosition::CO1, 2> cQueue;
    GlobalTensor<bfloat16_t> AGm;
    GlobalTensor<bfloat16_t> BGm;
    GlobalTensor<bfloat16_t> CGm;
    
    int M, N, K;
    int tileM, tileN, tileK;
};

// =============================================================================
// Ascend 910B Flash Attention - 256 token tiles
// =============================================================================
class FlashAttn910B {
public:
    __aicore__ inline FlashAttn910B() {}
    
    __aicore__ inline void Init(GM_ADDR Q, GM_ADDR K, GM_ADDR V, GM_ADDR output,
                                 int batch, int heads, int seqLen, int headDim,
                                 float scale) {
        this->batch = batch;
        this->heads = heads;
        this->seqLen = seqLen;
        this->headDim = headDim;
        this->scale = scale;
        
        QGm.SetGlobalBuffer((__gm__ bfloat16_t*)Q);
        KGm.SetGlobalBuffer((__gm__ bfloat16_t*)K);
        VGm.SetGlobalBuffer((__gm__ bfloat16_t*)V);
        outGm.SetGlobalBuffer((__gm__ bfloat16_t*)output);
        
        // 910B: Large KV tiles with 512KB L1
        kvTileSize = 256;
        qTileSize = 32;
        
        pipe.InitBuffer(qQueue, 2, qTileSize * headDim * sizeof(bfloat16_t));
        pipe.InitBuffer(kQueue, 2, kvTileSize * headDim * sizeof(bfloat16_t));
        pipe.InitBuffer(vQueue, 2, kvTileSize * headDim * sizeof(bfloat16_t));
        pipe.InitBuffer(scoreQueue, 1, qTileSize * kvTileSize * sizeof(float));
        pipe.InitBuffer(accumQueue, 1, qTileSize * headDim * sizeof(float));
    }
    
    __aicore__ inline void Process() {
        int blockIdx = GetBlockIdx();
        int b = blockIdx / heads;
        int h = blockIdx % heads;
        
        if (b >= batch) return;
        
        int baseOffset = (b * heads + h) * seqLen * headDim;
        
        // Initialize accumulator and softmax state
        LocalTensor<float> accum = accumQueue.AllocTensor<float>();
        float rowMax[32];
        float rowSum[32];
        
        for (int i = 0; i < qTileSize; i++) {
            rowMax[i] = -1e30f;
            rowSum[i] = 0.0f;
        }
        for (int i = 0; i < qTileSize * headDim; i++) {
            accum.SetValue(i, 0.0f);
        }
        
        // Process query tiles
        for (int qStart = 0; qStart < seqLen; qStart += qTileSize) {
            int numQ = min(qTileSize, seqLen - qStart);
            
            // Load Q tile
            LocalTensor<bfloat16_t> qLocal = qQueue.AllocTensor<bfloat16_t>();
            for (int qi = 0; qi < numQ; qi++) {
                DataCopyAsync(qLocal[qi * headDim], 
                             QGm[baseOffset + (qStart + qi) * headDim], headDim);
            }
            qQueue.EnQue(qLocal);
            qLocal = qQueue.DeQue<bfloat16_t>();
            
            // Reset state for new query tile
            for (int i = 0; i < numQ; i++) {
                rowMax[i] = -1e30f;
                rowSum[i] = 0.0f;
            }
            for (int i = 0; i < numQ * headDim; i++) {
                accum.SetValue(i, 0.0f);
            }
            
            // Process KV tiles
            for (int kvStart = 0; kvStart < seqLen; kvStart += kvTileSize) {
                int numKV = min(kvTileSize, seqLen - kvStart);
                
                // Load K and V tiles (pipelined)
                LocalTensor<bfloat16_t> kLocal = kQueue.AllocTensor<bfloat16_t>();
                LocalTensor<bfloat16_t> vLocal = vQueue.AllocTensor<bfloat16_t>();
                
                for (int ki = 0; ki < numKV; ki++) {
                    DataCopyAsync(kLocal[ki * headDim], 
                                 KGm[baseOffset + (kvStart + ki) * headDim], headDim);
                    DataCopyAsync(vLocal[ki * headDim], 
                                 VGm[baseOffset + (kvStart + ki) * headDim], headDim);
                }
                kQueue.EnQue(kLocal);
                vQueue.EnQue(vLocal);
                
                kLocal = kQueue.DeQue<bfloat16_t>();
                vLocal = vQueue.DeQue<bfloat16_t>();
                
                // Compute QK^T scores
                LocalTensor<float> scores = scoreQueue.AllocTensor<float>();
                
                for (int qi = 0; qi < numQ; qi++) {
                    int qPos = qStart + qi;
                    
                    for (int ki = 0; ki < numKV; ki++) {
                        int kPos = kvStart + ki;
                        
                        // Causal mask
                        if (kPos > qPos) {
                            scores.SetValue(qi * kvTileSize + ki, -1e30f);
                            continue;
                        }
                        
                        // Dot product with Cube unit
                        float score = 0.0f;
                        for (int d = 0; d < headDim; d++) {
                            score += (float)qLocal.GetValue(qi * headDim + d) *
                                     (float)kLocal.GetValue(ki * headDim + d);
                        }
                        scores.SetValue(qi * kvTileSize + ki, score * scale);
                    }
                    
                    // Online softmax update
                    float oldMax = rowMax[qi];
                    for (int ki = 0; ki < numKV && kvStart + ki <= qPos; ki++) {
                        rowMax[qi] = fmaxf(rowMax[qi], scores.GetValue(qi * kvTileSize + ki));
                    }
                    
                    float expDiff = expf(oldMax - rowMax[qi]);
                    rowSum[qi] *= expDiff;
                    
                    // Scale existing accumulator
                    for (int d = 0; d < headDim; d++) {
                        accum.SetValue(qi * headDim + d, 
                            accum.GetValue(qi * headDim + d) * expDiff);
                    }
                    
                    // Add weighted V
                    for (int ki = 0; ki < numKV && kvStart + ki <= qPos; ki++) {
                        float weight = expf(scores.GetValue(qi * kvTileSize + ki) - rowMax[qi]);
                        rowSum[qi] += weight;
                        
                        for (int d = 0; d < headDim; d++) {
                            accum.SetValue(qi * headDim + d,
                                accum.GetValue(qi * headDim + d) + 
                                weight * (float)vLocal.GetValue(ki * headDim + d));
                        }
                    }
                }
                
                scoreQueue.FreeTensor(scores);
                kQueue.FreeTensor(kLocal);
                vQueue.FreeTensor(vLocal);
            }
            
            // Normalize and output
            for (int qi = 0; qi < numQ; qi++) {
                float invSum = 1.0f / rowSum[qi];
                for (int d = 0; d < headDim; d++) {
                    outGm.SetValue(baseOffset + (qStart + qi) * headDim + d,
                        (bfloat16_t)(accum.GetValue(qi * headDim + d) * invSum));
                }
            }
            
            qQueue.FreeTensor(qLocal);
        }
        
        accumQueue.FreeTensor(accum);
    }
    
private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 2> qQueue;
    TQue<QuePosition::VECIN, 2> kQueue;
    TQue<QuePosition::VECIN, 2> vQueue;
    TQue<QuePosition::VECIN, 1> scoreQueue;
    TQue<QuePosition::VECIN, 1> accumQueue;
    GlobalTensor<bfloat16_t> QGm;
    GlobalTensor<bfloat16_t> KGm;
    GlobalTensor<bfloat16_t> VGm;
    GlobalTensor<bfloat16_t> outGm;
    
    int batch, heads, seqLen, headDim;
    int kvTileSize, qTileSize;
    float scale;
};

// Kernel entry points
extern "C" __global__ __aicore__ void rms_norm_910b(GM_ADDR input, GM_ADDR weight,
                                                     GM_ADDR output, int num_tokens,
                                                     int hidden_dim, float eps) {
    RmsNorm910B op;
    op.Init(input, weight, output, num_tokens, hidden_dim, eps);
    op.Process();
}

extern "C" __global__ __aicore__ void matmul_910b(GM_ADDR A, GM_ADDR B, GM_ADDR C,
                                                   int M, int N, int K) {
    MatMul910B op;
    op.Init(A, B, C, M, N, K);
    op.Process();
}

extern "C" __global__ __aicore__ void flash_attn_910b(GM_ADDR Q, GM_ADDR K, GM_ADDR V,
                                                       GM_ADDR output, int batch, int heads,
                                                       int seqLen, int headDim, float scale) {
    FlashAttn910B op;
    op.Init(Q, K, V, output, batch, heads, seqLen, headDim, scale);
    op.Process();
}
)";

}  // namespace ascend910b
}  // namespace ascend
}  // namespace persistent_kernel
}  // namespace yirage
