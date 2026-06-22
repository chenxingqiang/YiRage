/* Copyright 2025 YiRage Team
 * FPGA Embedding Kernels (Vitis HLS)
 */

#include <ap_fixed.h>
#include <hls_stream.h>
#include <hls_math.h>

typedef ap_fixed<16, 8> fp16_t;
typedef ap_fixed<32, 16> fp32_t;

// =============================================================================
// Embedding Lookup
// =============================================================================

void embedding_lookup(
    fp16_t* embedding_table,    // [vocab_size, hidden_dim]
    int* indices,               // [batch, seq_len]
    fp16_t* output,             // [batch, seq_len, hidden_dim]
    int batch, int seq_len, int vocab_size, int hidden_dim
) {
#pragma HLS INTERFACE m_axi port=embedding_table bundle=gmem0
#pragma HLS INTERFACE m_axi port=indices bundle=gmem1
#pragma HLS INTERFACE m_axi port=output bundle=gmem2
#pragma HLS INTERFACE s_axilite port=batch
#pragma HLS INTERFACE s_axilite port=seq_len
#pragma HLS INTERFACE s_axilite port=vocab_size
#pragma HLS INTERFACE s_axilite port=hidden_dim
#pragma HLS INTERFACE s_axilite port=return

    for (int b = 0; b < batch; b++) {
        for (int s = 0; s < seq_len; s++) {
            int token_id = indices[b * seq_len + s];
            
            // Burst read embedding vector
            for (int d = 0; d < hidden_dim; d++) {
#pragma HLS PIPELINE II=1
                output[(b * seq_len + s) * hidden_dim + d] = 
                    embedding_table[token_id * hidden_dim + d];
            }
        }
    }
}

// =============================================================================
// Rotary Position Embedding (RoPE)
// =============================================================================

void apply_rope(
    fp16_t* x,                  // [batch, heads, seq, head_dim]
    fp16_t* cos_cache,          // [seq, head_dim/2]
    fp16_t* sin_cache,          // [seq, head_dim/2]
    int batch, int heads, int seq, int head_dim
) {
#pragma HLS INTERFACE m_axi port=x bundle=gmem0
#pragma HLS INTERFACE m_axi port=cos_cache bundle=gmem1
#pragma HLS INTERFACE m_axi port=sin_cache bundle=gmem2
#pragma HLS INTERFACE s_axilite port=batch
#pragma HLS INTERFACE s_axilite port=heads
#pragma HLS INTERFACE s_axilite port=seq
#pragma HLS INTERFACE s_axilite port=head_dim
#pragma HLS INTERFACE s_axilite port=return

    int half_dim = head_dim / 2;

    for (int b = 0; b < batch; b++) {
        for (int h = 0; h < heads; h++) {
            for (int s = 0; s < seq; s++) {
                for (int d = 0; d < half_dim; d++) {
#pragma HLS PIPELINE II=1
                    int base_idx = ((b * heads + h) * seq + s) * head_dim;
                    
                    fp32_t x1 = fp32_t(x[base_idx + d]);
                    fp32_t x2 = fp32_t(x[base_idx + half_dim + d]);
                    fp32_t cos_val = fp32_t(cos_cache[s * half_dim + d]);
                    fp32_t sin_val = fp32_t(sin_cache[s * half_dim + d]);
                    
                    x[base_idx + d] = fp16_t(x1 * cos_val - x2 * sin_val);
                    x[base_idx + half_dim + d] = fp16_t(x1 * sin_val + x2 * cos_val);
                }
            }
        }
    }
}

// =============================================================================
// LM Head (Output Projection) - Streaming
// =============================================================================

template<int HIDDEN_DIM = 4096, int PARALLEL = 16>
void lm_head_stream(
    hls::stream<fp16_t>& hidden_stream,   // [hidden_dim]
    fp16_t* weight,                        // [vocab_size, hidden_dim]
    hls::stream<fp16_t>& logits_stream,   // [vocab_size]
    int vocab_size
) {
#pragma HLS INTERFACE axis port=hidden_stream
#pragma HLS INTERFACE m_axi port=weight bundle=gmem0
#pragma HLS INTERFACE axis port=logits_stream
#pragma HLS INTERFACE s_axilite port=vocab_size
#pragma HLS INTERFACE s_axilite port=return

    fp16_t hidden[HIDDEN_DIM];
#pragma HLS ARRAY_PARTITION variable=hidden cyclic factor=PARALLEL

    // Read hidden state
    for (int d = 0; d < HIDDEN_DIM; d++) {
#pragma HLS PIPELINE II=1
        hidden[d] = hidden_stream.read();
    }

    // Compute logits for each vocab token
    for (int v = 0; v < vocab_size; v++) {
        fp32_t sum = 0;
        for (int d = 0; d < HIDDEN_DIM; d++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=PARALLEL
            sum += fp32_t(hidden[d]) * fp32_t(weight[v * HIDDEN_DIM + d]);
        }
        logits_stream.write(fp16_t(sum));
    }
}

// =============================================================================
// Top-K Selection
// =============================================================================

template<int K = 50, int VOCAB_SIZE = 32000>
void topk(
    hls::stream<fp16_t>& logits_stream,
    fp16_t* values,
    int* indices
) {
#pragma HLS INTERFACE axis port=logits_stream
#pragma HLS INTERFACE m_axi port=values bundle=gmem0
#pragma HLS INTERFACE m_axi port=indices bundle=gmem1
#pragma HLS INTERFACE s_axilite port=return

    fp16_t top_values[K];
    int top_indices[K];
#pragma HLS ARRAY_PARTITION variable=top_values complete
#pragma HLS ARRAY_PARTITION variable=top_indices complete

    // Initialize with minimum values
    for (int i = 0; i < K; i++) {
#pragma HLS UNROLL
        top_values[i] = fp16_t(-1e10);
        top_indices[i] = -1;
    }

    // Process each logit
    for (int v = 0; v < VOCAB_SIZE; v++) {
#pragma HLS PIPELINE II=1
        fp16_t val = logits_stream.read();
        
        // Find insertion point
        int insert_pos = -1;
        for (int i = K - 1; i >= 0; i--) {
#pragma HLS UNROLL
            if (val > top_values[i]) {
                insert_pos = i;
            }
        }
        
        // Insert if in top-K
        if (insert_pos >= 0) {
            // Shift down
            for (int i = K - 1; i > insert_pos; i--) {
#pragma HLS UNROLL
                top_values[i] = top_values[i - 1];
                top_indices[i] = top_indices[i - 1];
            }
            top_values[insert_pos] = val;
            top_indices[insert_pos] = v;
        }
    }

    // Write results
    for (int i = 0; i < K; i++) {
#pragma HLS PIPELINE II=1
        values[i] = top_values[i];
        indices[i] = top_indices[i];
    }
}
