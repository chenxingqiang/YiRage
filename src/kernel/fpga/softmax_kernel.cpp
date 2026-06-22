/* Copyright 2025 YiRage Team
 * FPGA Softmax Kernels (Vitis HLS)
 */

#include <ap_fixed.h>
#include <hls_stream.h>
#include <hls_math.h>

typedef ap_fixed<16, 8> fp16_t;
typedef ap_fixed<32, 16> fp32_t;

// =============================================================================
// Row-wise Softmax
// =============================================================================

template<int MAX_COLS = 1024>
void softmax_row(
    hls::stream<fp16_t>& input,
    hls::stream<fp16_t>& output,
    int cols
) {
#pragma HLS INTERFACE axis port=input
#pragma HLS INTERFACE axis port=output
#pragma HLS INTERFACE s_axilite port=cols
#pragma HLS INTERFACE s_axilite port=return

    fp32_t buffer[MAX_COLS];
#pragma HLS ARRAY_PARTITION variable=buffer cyclic factor=8

    // Read and find max
    fp32_t max_val = fp32_t(-1e10);
    read_max: for (int i = 0; i < cols; i++) {
#pragma HLS PIPELINE II=1
        fp32_t val = fp32_t(input.read());
        buffer[i] = val;
        if (val > max_val) max_val = val;
    }

    // Compute exp and sum
    fp32_t sum_exp = 0;
    compute_exp: for (int i = 0; i < cols; i++) {
#pragma HLS PIPELINE II=1
        fp32_t exp_val = hls::exp(buffer[i] - max_val);
        buffer[i] = exp_val;
        sum_exp += exp_val;
    }

    // Normalize and write
    normalize: for (int i = 0; i < cols; i++) {
#pragma HLS PIPELINE II=1
        output.write(fp16_t(buffer[i] / sum_exp));
    }
}

// =============================================================================
// Batch Softmax
// =============================================================================

void softmax_batch(
    fp16_t* input,
    fp16_t* output,
    int rows, int cols
) {
#pragma HLS INTERFACE m_axi port=input bundle=gmem0 depth=1048576
#pragma HLS INTERFACE m_axi port=output bundle=gmem1 depth=1048576
#pragma HLS INTERFACE s_axilite port=rows
#pragma HLS INTERFACE s_axilite port=cols
#pragma HLS INTERFACE s_axilite port=return

    constexpr int TILE = 256;
    fp32_t buffer[TILE];
#pragma HLS ARRAY_PARTITION variable=buffer cyclic factor=8

    for (int r = 0; r < rows; r++) {
        // Find max
        fp32_t max_val = fp32_t(-1e10);
        for (int c = 0; c < cols; c++) {
#pragma HLS PIPELINE II=1
            fp32_t val = fp32_t(input[r * cols + c]);
            buffer[c % TILE] = val;
            if (val > max_val) max_val = val;
        }

        // Compute exp and sum
        fp32_t sum_exp = 0;
        for (int c = 0; c < cols; c++) {
#pragma HLS PIPELINE II=1
            fp32_t exp_val = hls::exp(fp32_t(input[r * cols + c]) - max_val);
            buffer[c % TILE] = exp_val;
            sum_exp += exp_val;
        }

        // Normalize and write
        for (int c = 0; c < cols; c++) {
#pragma HLS PIPELINE II=1
            fp32_t val = hls::exp(fp32_t(input[r * cols + c]) - max_val) / sum_exp;
            output[r * cols + c] = fp16_t(val);
        }
    }
}

// =============================================================================
// Online Softmax (streaming)
// =============================================================================

template<int TILE_SIZE = 64>
void online_softmax_tile(
    hls::stream<fp16_t>& input,
    hls::stream<fp16_t>& output,
    fp32_t& running_max,
    fp32_t& running_sum
) {
#pragma HLS INTERFACE axis port=input
#pragma HLS INTERFACE axis port=output
#pragma HLS PIPELINE II=1

    fp32_t tile[TILE_SIZE];
#pragma HLS ARRAY_PARTITION variable=tile complete

    // Read tile and update max
    fp32_t tile_max = running_max;
    for (int i = 0; i < TILE_SIZE; i++) {
#pragma HLS UNROLL
        tile[i] = fp32_t(input.read());
        if (tile[i] > tile_max) tile_max = tile[i];
    }

    // Update running stats
    fp32_t scale = hls::exp(running_max - tile_max);
    fp32_t new_sum = running_sum * scale;
    
    for (int i = 0; i < TILE_SIZE; i++) {
#pragma HLS UNROLL
        new_sum += hls::exp(tile[i] - tile_max);
    }

    running_max = tile_max;
    running_sum = new_sum;

    // Output normalized values (will need correction later)
    for (int i = 0; i < TILE_SIZE; i++) {
#pragma HLS PIPELINE II=1
        output.write(fp16_t(hls::exp(tile[i] - tile_max)));
    }
}
