/* Copyright 2025 YiRage Team
 * FPGA Matrix Multiplication Kernels (Vitis HLS)
 */

#ifndef __SYNTHESIS__
#include <iostream>
#endif

#include <ap_fixed.h>
#include <hls_stream.h>

// Fixed-point type for FPGA efficiency
typedef ap_fixed<16, 8> fp16_t;
typedef ap_fixed<32, 16> fp32_t;

// =============================================================================
// Tiled GEMM Kernel
// =============================================================================

template<int TILE_M = 64, int TILE_N = 64, int TILE_K = 64, int PARALLEL = 8>
void gemm_tiled(
    hls::stream<fp16_t>& a_stream,
    hls::stream<fp16_t>& b_stream,
    hls::stream<fp16_t>& c_stream,
    int M, int N, int K
) {
#pragma HLS INTERFACE axis port=a_stream
#pragma HLS INTERFACE axis port=b_stream
#pragma HLS INTERFACE axis port=c_stream
#pragma HLS INTERFACE s_axilite port=M
#pragma HLS INTERFACE s_axilite port=N
#pragma HLS INTERFACE s_axilite port=K
#pragma HLS INTERFACE s_axilite port=return

    // Local BRAM tiles
    fp16_t a_tile[TILE_M][TILE_K];
    fp16_t b_tile[TILE_K][TILE_N];
    fp32_t c_tile[TILE_M][TILE_N];

#pragma HLS ARRAY_PARTITION variable=a_tile cyclic factor=PARALLEL dim=2
#pragma HLS ARRAY_PARTITION variable=b_tile cyclic factor=PARALLEL dim=1
#pragma HLS ARRAY_PARTITION variable=c_tile cyclic factor=PARALLEL dim=2

    // Initialize output
    init_c: for (int i = 0; i < TILE_M; i++) {
        for (int j = 0; j < TILE_N; j++) {
#pragma HLS PIPELINE II=1
            c_tile[i][j] = 0;
        }
    }

    // Load A tile
    load_a: for (int i = 0; i < TILE_M; i++) {
        for (int k = 0; k < TILE_K; k++) {
#pragma HLS PIPELINE II=1
            a_tile[i][k] = a_stream.read();
        }
    }

    // Load B tile
    load_b: for (int k = 0; k < TILE_K; k++) {
        for (int j = 0; j < TILE_N; j++) {
#pragma HLS PIPELINE II=1
            b_tile[k][j] = b_stream.read();
        }
    }

    // Compute GEMM
    compute: for (int i = 0; i < TILE_M; i++) {
        for (int j = 0; j < TILE_N; j++) {
#pragma HLS PIPELINE II=1
            fp32_t sum = c_tile[i][j];
            for (int k = 0; k < TILE_K; k++) {
#pragma HLS UNROLL factor=PARALLEL
                sum += fp32_t(a_tile[i][k]) * fp32_t(b_tile[k][j]);
            }
            c_tile[i][j] = sum;
        }
    }

    // Store C tile
    store_c: for (int i = 0; i < TILE_M; i++) {
        for (int j = 0; j < TILE_N; j++) {
#pragma HLS PIPELINE II=1
            c_stream.write(fp16_t(c_tile[i][j]));
        }
    }
}

// =============================================================================
// Systolic Array GEMM (for high throughput)
// =============================================================================

template<int SIZE = 8>
void systolic_gemm(
    fp16_t A[SIZE][SIZE],
    fp16_t B[SIZE][SIZE],
    fp32_t C[SIZE][SIZE]
) {
#pragma HLS ARRAY_PARTITION variable=A complete dim=0
#pragma HLS ARRAY_PARTITION variable=B complete dim=0
#pragma HLS ARRAY_PARTITION variable=C complete dim=0
#pragma HLS PIPELINE II=1

    systolic_loop: for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
#pragma HLS UNROLL
            fp32_t sum = 0;
            for (int k = 0; k < SIZE; k++) {
#pragma HLS UNROLL
                sum += fp32_t(A[i][k]) * fp32_t(B[k][j]);
            }
            C[i][j] = sum;
        }
    }
}

// =============================================================================
// DDR-based GEMM with burst access
// =============================================================================

void gemm_ddr(
    fp16_t* A,
    fp16_t* B,
    fp16_t* C,
    int M, int N, int K
) {
#pragma HLS INTERFACE m_axi port=A bundle=gmem0 depth=4096
#pragma HLS INTERFACE m_axi port=B bundle=gmem1 depth=4096
#pragma HLS INTERFACE m_axi port=C bundle=gmem2 depth=4096
#pragma HLS INTERFACE s_axilite port=M
#pragma HLS INTERFACE s_axilite port=N
#pragma HLS INTERFACE s_axilite port=K
#pragma HLS INTERFACE s_axilite port=return

    constexpr int TILE = 32;
    
    fp16_t a_local[TILE][TILE];
    fp16_t b_local[TILE][TILE];
    fp32_t c_local[TILE][TILE];

#pragma HLS ARRAY_PARTITION variable=a_local cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=b_local cyclic factor=8 dim=1

    for (int m = 0; m < M; m += TILE) {
        for (int n = 0; n < N; n += TILE) {
            // Initialize accumulator
            for (int i = 0; i < TILE; i++) {
                for (int j = 0; j < TILE; j++) {
#pragma HLS PIPELINE II=1
                    c_local[i][j] = 0;
                }
            }
            
            for (int k = 0; k < K; k += TILE) {
                // Burst load A tile
                for (int i = 0; i < TILE; i++) {
                    for (int kk = 0; kk < TILE; kk++) {
#pragma HLS PIPELINE II=1
                        a_local[i][kk] = A[(m + i) * K + k + kk];
                    }
                }
                
                // Burst load B tile
                for (int kk = 0; kk < TILE; kk++) {
                    for (int j = 0; j < TILE; j++) {
#pragma HLS PIPELINE II=1
                        b_local[kk][j] = B[(k + kk) * N + n + j];
                    }
                }
                
                // Compute
                for (int i = 0; i < TILE; i++) {
                    for (int j = 0; j < TILE; j++) {
#pragma HLS PIPELINE II=1
                        fp32_t sum = c_local[i][j];
                        for (int kk = 0; kk < TILE; kk++) {
#pragma HLS UNROLL factor=8
                            sum += fp32_t(a_local[i][kk]) * fp32_t(b_local[kk][j]);
                        }
                        c_local[i][j] = sum;
                    }
                }
            }
            
            // Burst store C tile
            for (int i = 0; i < TILE; i++) {
                for (int j = 0; j < TILE; j++) {
#pragma HLS PIPELINE II=1
                    C[(m + i) * N + n + j] = fp16_t(c_local[i][j]);
                }
            }
        }
    }
}
