/* Copyright 2025 YiRage Team
 * FPGA Attention Kernels (Vitis HLS)
 */

#include <ap_fixed.h>
#include <hls_stream.h>
#include <hls_math.h>

typedef ap_fixed<16, 8> fp16_t;
typedef ap_fixed<32, 16> fp32_t;

// =============================================================================
// Scaled Dot-Product Attention
// =============================================================================

template<int HEAD_DIM = 64, int SEQ_TILE = 32>
void attention_tile(
    hls::stream<fp16_t>& q_stream,    // [SEQ_TILE, HEAD_DIM]
    hls::stream<fp16_t>& k_stream,    // [SEQ_TILE, HEAD_DIM]
    hls::stream<fp16_t>& v_stream,    // [SEQ_TILE, HEAD_DIM]
    hls::stream<fp16_t>& o_stream,    // [SEQ_TILE, HEAD_DIM]
    fp16_t scale
) {
#pragma HLS INTERFACE axis port=q_stream
#pragma HLS INTERFACE axis port=k_stream
#pragma HLS INTERFACE axis port=v_stream
#pragma HLS INTERFACE axis port=o_stream
#pragma HLS INTERFACE s_axilite port=scale
#pragma HLS INTERFACE s_axilite port=return

    // Local buffers
    fp16_t Q[SEQ_TILE][HEAD_DIM];
    fp16_t K[SEQ_TILE][HEAD_DIM];
    fp16_t V[SEQ_TILE][HEAD_DIM];
    fp32_t QK[SEQ_TILE][SEQ_TILE];
    fp32_t attn[SEQ_TILE][SEQ_TILE];
    fp32_t O[SEQ_TILE][HEAD_DIM];

#pragma HLS ARRAY_PARTITION variable=Q cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=K cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=V cyclic factor=8 dim=2

    // Load Q, K, V
    load_qkv: for (int i = 0; i < SEQ_TILE; i++) {
        for (int d = 0; d < HEAD_DIM; d++) {
#pragma HLS PIPELINE II=1
            Q[i][d] = q_stream.read();
            K[i][d] = k_stream.read();
            V[i][d] = v_stream.read();
        }
    }

    // Compute QK^T
    compute_qk: for (int i = 0; i < SEQ_TILE; i++) {
        for (int j = 0; j < SEQ_TILE; j++) {
#pragma HLS PIPELINE II=1
            fp32_t sum = 0;
            for (int d = 0; d < HEAD_DIM; d++) {
#pragma HLS UNROLL factor=8
                sum += fp32_t(Q[i][d]) * fp32_t(K[j][d]);
            }
            QK[i][j] = sum * fp32_t(scale);
        }
    }

    // Softmax per row
    softmax: for (int i = 0; i < SEQ_TILE; i++) {
        // Find max
        fp32_t max_val = QK[i][0];
        for (int j = 1; j < SEQ_TILE; j++) {
#pragma HLS PIPELINE II=1
            if (QK[i][j] > max_val) max_val = QK[i][j];
        }
        
        // Exp and sum
        fp32_t sum_exp = 0;
        for (int j = 0; j < SEQ_TILE; j++) {
#pragma HLS PIPELINE II=1
            fp32_t exp_val = hls::exp(QK[i][j] - max_val);
            attn[i][j] = exp_val;
            sum_exp += exp_val;
        }
        
        // Normalize
        for (int j = 0; j < SEQ_TILE; j++) {
#pragma HLS PIPELINE II=1
            attn[i][j] = attn[i][j] / sum_exp;
        }
    }

    // Compute attention @ V
    compute_output: for (int i = 0; i < SEQ_TILE; i++) {
        for (int d = 0; d < HEAD_DIM; d++) {
#pragma HLS PIPELINE II=1
            fp32_t sum = 0;
            for (int j = 0; j < SEQ_TILE; j++) {
#pragma HLS UNROLL factor=4
                sum += attn[i][j] * fp32_t(V[j][d]);
            }
            O[i][d] = sum;
        }
    }

    // Store output
    store_output: for (int i = 0; i < SEQ_TILE; i++) {
        for (int d = 0; d < HEAD_DIM; d++) {
#pragma HLS PIPELINE II=1
            o_stream.write(fp16_t(O[i][d]));
        }
    }
}

// =============================================================================
// Multi-Head Attention Wrapper
// =============================================================================

void multi_head_attention(
    fp16_t* Q,          // [batch, heads, seq, head_dim]
    fp16_t* K,
    fp16_t* V,
    fp16_t* O,
    int batch, int heads, int seq, int head_dim
) {
#pragma HLS INTERFACE m_axi port=Q bundle=gmem0
#pragma HLS INTERFACE m_axi port=K bundle=gmem1
#pragma HLS INTERFACE m_axi port=V bundle=gmem2
#pragma HLS INTERFACE m_axi port=O bundle=gmem3
#pragma HLS INTERFACE s_axilite port=batch
#pragma HLS INTERFACE s_axilite port=heads
#pragma HLS INTERFACE s_axilite port=seq
#pragma HLS INTERFACE s_axilite port=head_dim
#pragma HLS INTERFACE s_axilite port=return

    fp16_t scale = fp16_t(1.0f / hls::sqrt(float(head_dim)));
    
    // Process each batch and head
    for (int b = 0; b < batch; b++) {
        for (int h = 0; h < heads; h++) {
            // Create streams for tiled processing
            hls::stream<fp16_t> q_stream, k_stream, v_stream, o_stream;
#pragma HLS STREAM variable=q_stream depth=4096
#pragma HLS STREAM variable=k_stream depth=4096
#pragma HLS STREAM variable=v_stream depth=4096
#pragma HLS STREAM variable=o_stream depth=4096

            // Load and process tiles
            // ... (tiled attention implementation)
        }
    }
}
