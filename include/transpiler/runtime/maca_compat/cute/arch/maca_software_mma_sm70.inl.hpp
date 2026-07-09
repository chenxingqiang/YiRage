/* MetaX mxcc: software Volta m8n8k4 MMA (no mma.sync PTX on xcore1000). */
#pragma once

#include <cstdint>

#if defined(__CUDACC__) || defined(__MACA__) || defined(__MACACC__)
#include <cuda_fp16.h>
#endif

#ifndef CUTE_HOST_DEVICE
#if defined(__CUDACC__) || defined(__MACA__) || defined(__MACACC__)
#define CUTE_HOST_DEVICE __host__ __device__
#else
#define CUTE_HOST_DEVICE
#endif
#endif

namespace yirage_maca {
namespace sm70_sw {

#if defined(__CUDA_ARCH__) || defined(__MACA__) || defined(__MACACC__)

CUTE_HOST_DEVICE inline int subgroup_base_lane() {
  return static_cast<int>(threadIdx.x) & ~7;
}

CUTE_HOST_DEVICE inline int subgroup_lane() {
  return static_cast<int>(threadIdx.x) & 7;
}

CUTE_HOST_DEVICE inline unsigned subgroup_mask() {
  int base = subgroup_base_lane() & 31;
  return 0xFFu << base;
}

CUTE_HOST_DEVICE inline void unpack_half2(uint32_t x, half out[2]) {
  half2 const *p = reinterpret_cast<half2 const *>(&x);
  out[0] = p[0].x;
  out[1] = p[0].y;
}

CUTE_HOST_DEVICE inline uint32_t pack_half2(half a, half b) {
  uint32_t out;
  reinterpret_cast<half2 *>(&out)[0] = __halves2half2(a, b);
  return out;
}

// Volta m8n8k4 f16 TN: A row-major 8x4, B col-major 4x8, one output row per lane.
CUTE_HOST_DEVICE inline void mma_m8n8k4_f16_tn(
    uint32_t &d0, uint32_t &d1, uint32_t &d2, uint32_t &d3,
    uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1,
    uint32_t c0, uint32_t c1, uint32_t c2, uint32_t c3) {
  unsigned mask = subgroup_mask();
  int base = subgroup_base_lane();
  int lane = subgroup_lane();

  half A[8][4];
  half B[4][8];
  half C[8][8];

  for (int i = 0; i < 8; ++i) {
    uint32_t ra0 = __shfl_sync(mask, a0, base + i);
    uint32_t ra1 = __shfl_sync(mask, a1, base + i);
    half row_a[4];
    unpack_half2(ra0, row_a);
    unpack_half2(ra1, row_a + 2);
    for (int k = 0; k < 4; ++k) {
      A[i][k] = row_a[k];
    }
  }

  for (int j = 0; j < 8; ++j) {
    uint32_t rb0 = __shfl_sync(mask, b0, base + j);
    uint32_t rb1 = __shfl_sync(mask, b1, base + j);
    half col_b[4];
    unpack_half2(rb0, col_b);
    unpack_half2(rb1, col_b + 2);
    for (int k = 0; k < 4; ++k) {
      B[k][j] = col_b[k];
    }
  }

  for (int i = 0; i < 8; ++i) {
    uint32_t rc0 = __shfl_sync(mask, c0, base + i);
    uint32_t rc1 = __shfl_sync(mask, c1, base + i);
    uint32_t rc2 = __shfl_sync(mask, c2, base + i);
    uint32_t rc3 = __shfl_sync(mask, c3, base + i);
    half row_c[8];
    unpack_half2(rc0, row_c);
    unpack_half2(rc1, row_c + 2);
    unpack_half2(rc2, row_c + 4);
    unpack_half2(rc3, row_c + 6);
    for (int n = 0; n < 8; ++n) {
      C[i][n] = row_c[n];
    }
  }

  for (int m = 0; m < 8; ++m) {
    for (int n = 0; n < 8; ++n) {
      half acc = C[m][n];
      for (int k = 0; k < 4; ++k) {
        acc = __hfma(A[m][k], B[k][n], acc);
      }
      C[m][n] = acc;
    }
  }

  half row_out[8];
  for (int n = 0; n < 8; ++n) {
    row_out[n] = C[lane][n];
  }
  d0 = pack_half2(row_out[0], row_out[1]);
  d1 = pack_half2(row_out[2], row_out[3]);
  d2 = pack_half2(row_out[4], row_out[5]);
  d3 = pack_half2(row_out[6], row_out[7]);
}

CUTE_HOST_DEVICE inline void mma_m8n8k4_f32f16f16f32_tn(
    float &d0, float &d1, float &d2, float &d3,
    float &d4, float &d5, float &d6, float &d7,
    uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3,
    float c4, float c5, float c6, float c7) {
  unsigned mask = subgroup_mask();
  int base = subgroup_base_lane();
  int lane = subgroup_lane();

  half A[8][4];
  half B[4][8];
  float C[8][8];

  for (int i = 0; i < 8; ++i) {
    uint32_t ra0 = __shfl_sync(mask, a0, base + i);
    uint32_t ra1 = __shfl_sync(mask, a1, base + i);
    half row_a[4];
    unpack_half2(ra0, row_a);
    unpack_half2(ra1, row_a + 2);
    for (int k = 0; k < 4; ++k) {
      A[i][k] = row_a[k];
    }
  }

  for (int j = 0; j < 8; ++j) {
    uint32_t rb0 = __shfl_sync(mask, b0, base + j);
    uint32_t rb1 = __shfl_sync(mask, b1, base + j);
    half col_b[4];
    unpack_half2(rb0, col_b);
    unpack_half2(rb1, col_b + 2);
    for (int k = 0; k < 4; ++k) {
      B[k][j] = col_b[k];
    }
  }

  for (int i = 0; i < 8; ++i) {
    float rc[8];
    rc[0] = __shfl_sync(mask, c0, base + i);
    rc[1] = __shfl_sync(mask, c1, base + i);
    rc[2] = __shfl_sync(mask, c2, base + i);
    rc[3] = __shfl_sync(mask, c3, base + i);
    rc[4] = __shfl_sync(mask, c4, base + i);
    rc[5] = __shfl_sync(mask, c5, base + i);
    rc[6] = __shfl_sync(mask, c6, base + i);
    rc[7] = __shfl_sync(mask, c7, base + i);
    for (int n = 0; n < 8; ++n) {
      C[i][n] = rc[n];
    }
  }

  for (int m = 0; m < 8; ++m) {
    for (int n = 0; n < 8; ++n) {
      float acc = C[m][n];
      for (int k = 0; k < 4; ++k) {
        acc = __fmaf_rn(__half2float(A[m][k]), __half2float(B[k][n]), acc);
      }
      C[m][n] = acc;
    }
  }

  d0 = C[lane][0];
  d1 = C[lane][1];
  d2 = C[lane][2];
  d3 = C[lane][3];
  d4 = C[lane][4];
  d5 = C[lane][5];
  d6 = C[lane][6];
  d7 = C[lane][7];
}

#endif // device compile

} // namespace sm70_sw
} // namespace yirage_maca
