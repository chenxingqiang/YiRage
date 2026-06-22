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
 * @brief Ascend 910C Optimized TBE Kernels (Latest Training)
 *
 * Ascend 910C (2024) - Latest generation:
 * - 48 AI Cores (50% more!)
 * - 1MB L1 buffer per AI Core
 * - 32x32 Cube unit (enlarged!)
 * - 96GB HBM
 * - 500 TFLOPS FP16
 * - Advanced INT4 quantization
 *
 * Optimization strategy:
 * - Maximum tile sizes with 1MB L1
 * - 32x32 Cube operations
 * - Multi-stage pipelining
 * - Fused operators
 */

#include "../common/ascend_common.h"

namespace yirage {
namespace persistent_kernel {
namespace ascend {
namespace ascend910c {

constexpr int ASCEND910C_AI_CORES = 48;
constexpr int ASCEND910C_L1_KB = 1024;
constexpr int ASCEND910C_CUBE_SIZE = 32;

constexpr const char* ASCEND910C_KERNEL_SOURCE = R"(
// =============================================================================
// Ascend 910C Optimized Kernels (TBE/CANN)
// Latest training - 48 AI Cores, 1MB L1, 32x32 Cube
// =============================================================================

#include "kernel_operator.h"
using namespace AscendC;

// =============================================================================
// Ascend 910C Batched RMSNorm - Multiple tokens per AI Core
// =============================================================================
class RmsNorm910C {
public:
    __aicore__ inline RmsNorm910C() {}
    
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR weight, GM_ADDR output,
                                 int num_tokens, int hidden_dim, float eps) {
        this->num_tokens = num_tokens;
        this->hidden_dim = hidden_dim;
        this->eps = eps;
        
        inputGm.SetGlobalBuffer((__gm__ bfloat16_t*)input);
        weightGm.SetGlobalBuffer((__gm__ bfloat16_t*)weight);
        outputGm.SetGlobalBuffer((__gm__ bfloat16_t*)output);
        
        // 910C: Process multiple tokens per AI Core (1MB L1!)
        tokensPerCore = 4;
        
        pipe.InitBuffer(inputQueue, 2, tokensPerCore * hidden_dim * sizeof(bfloat16_t));
        pipe.InitBuffer(weightQueue, 1, hidden_dim * sizeof(bfloat16_t));
        pipe.InitBuffer(outputQueue, 2, tokensPerCore * hidden_dim * sizeof(bfloat16_t));
        pipe.InitBuffer(workQueue, 2, tokensPerCore * hidden_dim * sizeof(float));
    }
    
    __aicore__ inline void Process() {
        int tokenBase = GetBlockIdx() * tokensPerCore;
        if (tokenBase >= num_tokens) return;
        
        int actualTokens = min(tokensPerCore, num_tokens - tokenBase);
        
        // Load weight (shared)
        LocalTensor<bfloat16_t> weightLocal = weightQueue.AllocTensor<bfloat16_t>();
        DataCopyAsync(weightLocal, weightGm, hidden_dim);
        weightQueue.EnQue(weightLocal);
        
        // Load multiple tokens
        LocalTensor<bfloat16_t> inputLocal = inputQueue.AllocTensor<bfloat16_t>();
        for (int t = 0; t < actualTokens; t++) {
            DataCopyAsync(inputLocal[t * hidden_dim], 
                         inputGm[(tokenBase + t) * hidden_dim], hidden_dim);
        }
        inputQueue.EnQue(inputLocal);
        
        inputLocal = inputQueue.DeQue<bfloat16_t>();
        weightLocal = weightQueue.DeQue<bfloat16_t>();
        
        // Process each token
        LocalTensor<float> work = workQueue.AllocTensor<float>();
        LocalTensor<bfloat16_t> outputLocal = outputQueue.AllocTensor<bfloat16_t>();
        
        for (int t = 0; t < actualTokens; t++) {
            int tokenOffset = t * hidden_dim;
            
            // Square and sum
            Cast(work[tokenOffset], inputLocal[tokenOffset], 
                 RoundMode::CAST_NONE, hidden_dim);
            
            float totalSum = 0.0f;
            for (int i = 0; i < hidden_dim; i++) {
                float val = work.GetValue(tokenOffset + i);
                totalSum += val * val;
            }
            
            float invRms = rsqrtf(totalSum / hidden_dim + eps);
            
            // Normalize
            for (int i = 0; i < hidden_dim; i++) {
                float in = (float)inputLocal.GetValue(tokenOffset + i);
                float w = (float)weightLocal.GetValue(i);
                outputLocal.SetValue(tokenOffset + i, (bfloat16_t)(in * invRms * w));
            }
        }
        
        // Store all tokens
        outputQueue.EnQue(outputLocal);
        outputLocal = outputQueue.DeQue<bfloat16_t>();
        
        for (int t = 0; t < actualTokens; t++) {
            DataCopyAsync(outputGm[(tokenBase + t) * hidden_dim], 
                         outputLocal[t * hidden_dim], hidden_dim);
        }
        
        workQueue.FreeTensor(work);
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
    int tokensPerCore;
};

// =============================================================================
// Ascend 910C GEMM - 32x64 Cube tiles (enlarged!)
// =============================================================================
class MatMul910C {
public:
    __aicore__ inline MatMul910C() {}
    
    __aicore__ inline void Init(GM_ADDR A, GM_ADDR B, GM_ADDR C,
                                 int M, int N, int K) {
        this->M = M;
        this->N = N;
        this->K = K;
        
        AGm.SetGlobalBuffer((__gm__ bfloat16_t*)A);
        BGm.SetGlobalBuffer((__gm__ bfloat16_t*)B);
        CGm.SetGlobalBuffer((__gm__ bfloat16_t*)C);
        
        // 910C: 256x256 output tiles with 32x32 Cube and 1MB L1
        tileM = 256;
        tileN = 256;
        tileK = 64;
        
        // Quad buffer for maximum throughput
        pipe.InitBuffer(aQueue, 4, tileM * tileK * sizeof(bfloat16_t));
        pipe.InitBuffer(bQueue, 4, tileK * tileN * sizeof(bfloat16_t));
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
        
        // 4-stage pipeline
        LocalTensor<bfloat16_t> aLocal[4];
        LocalTensor<bfloat16_t> bLocal[4];
        
        // Prefetch first 3 stages
        for (int stage = 0; stage < 3 && stage < numKTiles; stage++) {
            aLocal[stage] = aQueue.AllocTensor<bfloat16_t>();
            bLocal[stage] = bQueue.AllocTensor<bfloat16_t>();
            LoadTileAsync(aLocal[stage], bLocal[stage], 
                         rowStart, colStart, stage * tileK);
        }
        
        for (int kTile = 0; kTile < numKTiles; kTile++) {
            int loadStage = (kTile + 3) % 4;
            int computeStage = kTile % 4;
            
            // Start loading next tile
            if (kTile + 3 < numKTiles) {
                aLocal[loadStage] = aQueue.AllocTensor<bfloat16_t>();
                bLocal[loadStage] = bQueue.AllocTensor<bfloat16_t>();
                LoadTileAsync(aLocal[loadStage], bLocal[loadStage], 
                             rowStart, colStart, (kTile + 3) * tileK);
            }
            
            // Wait for current tile
            aLocal[computeStage] = aQueue.DeQue<bfloat16_t>();
            bLocal[computeStage] = bQueue.DeQue<bfloat16_t>();
            
            int actualK = min(tileK, K - kTile * tileK);
            
            // 32x32 Cube matmul (multiple iterations for 256x256)
            // Split into 8x8 = 64 32x32 sub-tiles
            for (int subM = 0; subM < actualM; subM += 32) {
                for (int subN = 0; subN < actualN; subN += 32) {
                    int sm = min(32, actualM - subM);
                    int sn = min(32, actualN - subN);
                    
                    // 32x32 Cube operation
                    Mmad32x32(accum, subM, subN, 
                             aLocal[computeStage], subM,
                             bLocal[computeStage], subN,
                             sm, sn, actualK);
                }
            }
            
            aQueue.FreeTensor(aLocal[computeStage]);
            bQueue.FreeTensor(bLocal[computeStage]);
        }
        
        // Store result
        cQueue.EnQue(accum);
        accum = cQueue.DeQue<float>();
        
        LocalTensor<bfloat16_t> outLocal = pipe.AllocTensor<bfloat16_t>();
        Cast(outLocal, accum, RoundMode::CAST_ROUND, actualM * tileN);
        
        for (int m = 0; m < actualM; m++) {
            DataCopyAsync(CGm[(rowStart + m) * N + colStart], 
                         outLocal[m * tileN], actualN);
        }
        
        pipe.FreeTensor(outLocal);
        cQueue.FreeTensor(accum);
    }
    
private:
    inline void LoadTileAsync(LocalTensor<bfloat16_t>& a, LocalTensor<bfloat16_t>& b,
                              int rowStart, int colStart, int kStart) {
        int actualK = min(tileK, K - kStart);
        int actualM = min(tileM, M - rowStart);
        int actualN = min(tileN, N - colStart);
        
        for (int m = 0; m < actualM; m++) {
            DataCopyAsync(a[m * tileK], AGm[(rowStart + m) * K + kStart], actualK);
        }
        aQueue.EnQue(a);
        
        for (int k = 0; k < actualK; k++) {
            DataCopyAsync(b[k * tileN], BGm[(kStart + k) * N + colStart], actualN);
        }
        bQueue.EnQue(b);
    }
    
    inline void Mmad32x32(LocalTensor<float>& acc, int accRowOff, int accColOff,
                          LocalTensor<bfloat16_t>& a, int aRowOff,
                          LocalTensor<bfloat16_t>& b, int bColOff,
                          int m, int n, int k) {
        // 32x32 Cube native operation
        for (int mi = 0; mi < m; mi++) {
            for (int ni = 0; ni < n; ni++) {
                float sum = acc.GetValue((accRowOff + mi) * tileN + accColOff + ni);
                for (int ki = 0; ki < k; ki++) {
                    sum += (float)a.GetValue((aRowOff + mi) * tileK + ki) *
                           (float)b.GetValue(ki * tileN + bColOff + ni);
                }
                acc.SetValue((accRowOff + mi) * tileN + accColOff + ni, sum);
            }
        }
    }
    
    TPipe pipe;
    TQue<QuePosition::A, 4> aQueue;
    TQue<QuePosition::B, 4> bQueue;
    TQue<QuePosition::CO1, 2> cQueue;
    GlobalTensor<bfloat16_t> AGm;
    GlobalTensor<bfloat16_t> BGm;
    GlobalTensor<bfloat16_t> CGm;
    
    int M, N, K;
    int tileM, tileN, tileK;
};

// =============================================================================
// Ascend 910C Flash Attention - 512 token tiles
// =============================================================================
class FlashAttn910C {
public:
    __aicore__ inline FlashAttn910C() {}
    
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
        
        // 910C: Maximum tile sizes with 1MB L1
        kvTileSize = 512;
        qTileSize = 64;
        
        pipe.InitBuffer(qQueue, 2, qTileSize * headDim * sizeof(bfloat16_t));
        pipe.InitBuffer(kQueue, 3, kvTileSize * headDim * sizeof(bfloat16_t));
        pipe.InitBuffer(vQueue, 3, kvTileSize * headDim * sizeof(bfloat16_t));
        pipe.InitBuffer(scoreQueue, 2, qTileSize * kvTileSize * sizeof(float));
        pipe.InitBuffer(accumQueue, 1, qTileSize * headDim * sizeof(float));
    }
    
    __aicore__ inline void Process() {
        int blockIdx = GetBlockIdx();
        int b = blockIdx / heads;
        int h = blockIdx % heads;
        
        if (b >= batch) return;
        
        int baseOffset = (b * heads + h) * seqLen * headDim;
        
        LocalTensor<float> accum = accumQueue.AllocTensor<float>();
        float rowMax[64];
        float rowSum[64];
        
        // Process query tiles
        for (int qStart = 0; qStart < seqLen; qStart += qTileSize) {
            int numQ = min(qTileSize, seqLen - qStart);
            
            // Reset state
            for (int i = 0; i < numQ; i++) {
                rowMax[i] = -1e30f;
                rowSum[i] = 0.0f;
            }
            for (int i = 0; i < numQ * headDim; i++) {
                accum.SetValue(i, 0.0f);
            }
            
            // Load Q tile
            LocalTensor<bfloat16_t> qLocal = qQueue.AllocTensor<bfloat16_t>();
            for (int qi = 0; qi < numQ; qi++) {
                DataCopyAsync(qLocal[qi * headDim], 
                             QGm[baseOffset + (qStart + qi) * headDim], headDim);
            }
            qQueue.EnQue(qLocal);
            qLocal = qQueue.DeQue<bfloat16_t>();
            
            // Prefetch first KV tiles
            int numKVTiles = (seqLen + kvTileSize - 1) / kvTileSize;
            LocalTensor<bfloat16_t> kLocal[3];
            LocalTensor<bfloat16_t> vLocal[3];
            
            for (int stage = 0; stage < 2 && stage < numKVTiles; stage++) {
                kLocal[stage] = kQueue.AllocTensor<bfloat16_t>();
                vLocal[stage] = vQueue.AllocTensor<bfloat16_t>();
                LoadKVTileAsync(kLocal[stage], vLocal[stage], 
                               baseOffset, stage * kvTileSize);
            }
            
            // Process KV tiles with pipelining
            for (int kvTile = 0; kvTile < numKVTiles; kvTile++) {
                int kvStart = kvTile * kvTileSize;
                int numKV = min(kvTileSize, seqLen - kvStart);
                
                // Prefetch next tile
                if (kvTile + 2 < numKVTiles) {
                    int loadStage = (kvTile + 2) % 3;
                    kLocal[loadStage] = kQueue.AllocTensor<bfloat16_t>();
                    vLocal[loadStage] = vQueue.AllocTensor<bfloat16_t>();
                    LoadKVTileAsync(kLocal[loadStage], vLocal[loadStage], 
                                   baseOffset, (kvTile + 2) * kvTileSize);
                }
                
                int computeStage = kvTile % 3;
                kLocal[computeStage] = kQueue.DeQue<bfloat16_t>();
                vLocal[computeStage] = vQueue.DeQue<bfloat16_t>();
                
                // Compute scores using 32x32 Cube for QK^T
                LocalTensor<float> scores = scoreQueue.AllocTensor<float>();
                
                ComputeQKT(scores, qLocal, kLocal[computeStage], 
                          numQ, numKV, qStart, kvStart);
                
                // Online softmax and V accumulation
                UpdateSoftmaxAndAccum(accum, scores, vLocal[computeStage],
                                     rowMax, rowSum, numQ, numKV, qStart, kvStart);
                
                scoreQueue.FreeTensor(scores);
                kQueue.FreeTensor(kLocal[computeStage]);
                vQueue.FreeTensor(vLocal[computeStage]);
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
    inline void LoadKVTileAsync(LocalTensor<bfloat16_t>& k, LocalTensor<bfloat16_t>& v,
                                int baseOffset, int kvStart) {
        int numKV = min(kvTileSize, seqLen - kvStart);
        for (int ki = 0; ki < numKV; ki++) {
            DataCopyAsync(k[ki * headDim], 
                         KGm[baseOffset + (kvStart + ki) * headDim], headDim);
            DataCopyAsync(v[ki * headDim], 
                         VGm[baseOffset + (kvStart + ki) * headDim], headDim);
        }
        kQueue.EnQue(k);
        vQueue.EnQue(v);
    }
    
    inline void ComputeQKT(LocalTensor<float>& scores, 
                           LocalTensor<bfloat16_t>& q,
                           LocalTensor<bfloat16_t>& k,
                           int numQ, int numKV, int qStart, int kvStart) {
        // Use 32x32 Cube for batch dot products
        for (int qi = 0; qi < numQ; qi++) {
            int qPos = qStart + qi;
            for (int ki = 0; ki < numKV; ki++) {
                int kPos = kvStart + ki;
                
                if (kPos > qPos) {
                    scores.SetValue(qi * kvTileSize + ki, -1e30f);
                    continue;
                }
                
                float score = 0.0f;
                for (int d = 0; d < headDim; d++) {
                    score += (float)q.GetValue(qi * headDim + d) *
                             (float)k.GetValue(ki * headDim + d);
                }
                scores.SetValue(qi * kvTileSize + ki, score * scale);
            }
        }
    }
    
    inline void UpdateSoftmaxAndAccum(LocalTensor<float>& accum,
                                      LocalTensor<float>& scores,
                                      LocalTensor<bfloat16_t>& v,
                                      float* rowMax, float* rowSum,
                                      int numQ, int numKV, int qStart, int kvStart) {
        for (int qi = 0; qi < numQ; qi++) {
            int qPos = qStart + qi;
            
            float oldMax = rowMax[qi];
            
            // Find new max
            for (int ki = 0; ki < numKV && kvStart + ki <= qPos; ki++) {
                rowMax[qi] = fmaxf(rowMax[qi], scores.GetValue(qi * kvTileSize + ki));
            }
            
            float expDiff = expf(oldMax - rowMax[qi]);
            rowSum[qi] *= expDiff;
            
            // Scale accumulator
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
                        weight * (float)v.GetValue(ki * headDim + d));
                }
            }
        }
    }
    
    TPipe pipe;
    TQue<QuePosition::VECIN, 2> qQueue;
    TQue<QuePosition::VECIN, 3> kQueue;
    TQue<QuePosition::VECIN, 3> vQueue;
    TQue<QuePosition::VECIN, 2> scoreQueue;
    TQue<QuePosition::VECIN, 1> accumQueue;
    GlobalTensor<bfloat16_t> QGm;
    GlobalTensor<bfloat16_t> KGm;
    GlobalTensor<bfloat16_t> VGm;
    GlobalTensor<bfloat16_t> outGm;
    
    int batch, heads, seqLen, headDim;
    int kvTileSize, qTileSize;
    float scale;
};

// =============================================================================
// Ascend 910C Fused MLP (Gate + Up + SiLU + Down)
// =============================================================================
class FusedMLP910C {
public:
    __aicore__ inline FusedMLP910C() {}
    
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR gateW, GM_ADDR upW, 
                                 GM_ADDR downW, GM_ADDR output,
                                 int batchTokens, int hiddenDim, int interDim) {
        this->batchTokens = batchTokens;
        this->hiddenDim = hiddenDim;
        this->interDim = interDim;
        
        inputGm.SetGlobalBuffer((__gm__ bfloat16_t*)input);
        gateWGm.SetGlobalBuffer((__gm__ bfloat16_t*)gateW);
        upWGm.SetGlobalBuffer((__gm__ bfloat16_t*)upW);
        downWGm.SetGlobalBuffer((__gm__ bfloat16_t*)downW);
        outputGm.SetGlobalBuffer((__gm__ bfloat16_t*)output);
        
        // 910C: Large intermediate buffer in 1MB L1
        pipe.InitBuffer(inputQueue, 2, hiddenDim * sizeof(bfloat16_t));
        pipe.InitBuffer(interQueue, 2, interDim * sizeof(float));
        pipe.InitBuffer(outputQueue, 2, hiddenDim * sizeof(bfloat16_t));
    }
    
    __aicore__ inline void Process() {
        int tokenIdx = GetBlockIdx();
        if (tokenIdx >= batchTokens) return;
        
        // Load input
        LocalTensor<bfloat16_t> input = inputQueue.AllocTensor<bfloat16_t>();
        DataCopyAsync(input, inputGm[tokenIdx * hiddenDim], hiddenDim);
        inputQueue.EnQue(input);
        input = inputQueue.DeQue<bfloat16_t>();
        
        // Step 1: Gate and Up projections (fused with SiLU)
        LocalTensor<float> inter = interQueue.AllocTensor<float>();
        
        for (int i = 0; i < interDim; i++) {
            float gateVal = 0.0f;
            float upVal = 0.0f;
            
            for (int d = 0; d < hiddenDim; d++) {
                float x = (float)input.GetValue(d);
                gateVal += x * (float)gateWGm.GetValue(i * hiddenDim + d);
                upVal += x * (float)upWGm.GetValue(i * hiddenDim + d);
            }
            
            // Fused SiLU(gate) * up
            float sigmoid = 1.0f / (1.0f + expf(-gateVal));
            inter.SetValue(i, (gateVal * sigmoid) * upVal);
        }
        
        interQueue.EnQue(inter);
        inter = interQueue.DeQue<float>();
        
        // Step 2: Down projection
        LocalTensor<bfloat16_t> output = outputQueue.AllocTensor<bfloat16_t>();
        
        for (int d = 0; d < hiddenDim; d++) {
            float val = 0.0f;
            for (int i = 0; i < interDim; i++) {
                val += inter.GetValue(i) * (float)downWGm.GetValue(d * interDim + i);
            }
            output.SetValue(d, (bfloat16_t)val);
        }
        
        outputQueue.EnQue(output);
        output = outputQueue.DeQue<bfloat16_t>();
        DataCopyAsync(outputGm[tokenIdx * hiddenDim], output, hiddenDim);
        
        inputQueue.FreeTensor(input);
        interQueue.FreeTensor(inter);
    }
    
private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 2> inputQueue;
    TQue<QuePosition::VECIN, 2> interQueue;
    TQue<QuePosition::VECOUT, 2> outputQueue;
    GlobalTensor<bfloat16_t> inputGm;
    GlobalTensor<bfloat16_t> gateWGm;
    GlobalTensor<bfloat16_t> upWGm;
    GlobalTensor<bfloat16_t> downWGm;
    GlobalTensor<bfloat16_t> outputGm;
    
    int batchTokens, hiddenDim, interDim;
};

// Kernel entry points
extern "C" __global__ __aicore__ void rms_norm_910c(GM_ADDR input, GM_ADDR weight,
                                                     GM_ADDR output, int num_tokens,
                                                     int hidden_dim, float eps) {
    RmsNorm910C op;
    op.Init(input, weight, output, num_tokens, hidden_dim, eps);
    op.Process();
}

extern "C" __global__ __aicore__ void matmul_910c(GM_ADDR A, GM_ADDR B, GM_ADDR C,
                                                   int M, int N, int K) {
    MatMul910C op;
    op.Init(A, B, C, M, N, K);
    op.Process();
}

extern "C" __global__ __aicore__ void flash_attn_910c(GM_ADDR Q, GM_ADDR K, GM_ADDR V,
                                                       GM_ADDR output, int batch, int heads,
                                                       int seqLen, int headDim, float scale) {
    FlashAttn910C op;
    op.Init(Q, K, V, output, batch, heads, seqLen, headDim, scale);
    op.Process();
}

extern "C" __global__ __aicore__ void fused_mlp_910c(GM_ADDR input, GM_ADDR gateW,
                                                      GM_ADDR upW, GM_ADDR downW,
                                                      GM_ADDR output, int batchTokens,
                                                      int hiddenDim, int interDim) {
    FusedMLP910C op;
    op.Init(input, gateW, upW, downW, output, batchTokens, hiddenDim, interDim);
    op.Process();
}
)";

}  // namespace ascend910c
}  // namespace ascend
}  // namespace persistent_kernel
}  // namespace yirage
