/***************************************************************************************************
 * MetaX mxcc shadow for cute/arch/mma_sm70.hpp
 *
 * When YIRAGE_MACA_SOFTWARE_MMA=1, replaces PTX mma.sync with warp-shuffle software MMA so
 * transpiled kernels compile on xcore1000. Otherwise delegates to CUTLASS via include_next.
 **************************************************************************************************/
#pragma once

#if defined(YIRAGE_MACA_SOFTWARE_MMA)

#include <cute/config.hpp>
#include <cstdint>

#include "maca_software_mma_sm70.inl.hpp"

#define CUTE_ARCH_MMA_SM70_SUPPORTED

namespace cute {

struct SM70_8x8x4_F16F16F16F16_TN {
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  fma(uint32_t &d0, uint32_t &d1, uint32_t &d2, uint32_t &d3,
      uint32_t const &a0, uint32_t const &a1, uint32_t const &b0, uint32_t const &b1,
      uint32_t const &c0, uint32_t const &c1, uint32_t const &c2, uint32_t const &c3) {
    yirage_maca::sm70_sw::mma_m8n8k4_f16_tn(d0, d1, d2, d3, a0, a1, b0, b1, c0, c1, c2, c3);
  }
};

struct SM70_8x8x4_F16F16F16F16_NT {
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  fma(uint32_t &d0, uint32_t &d1, uint32_t &d2, uint32_t &d3,
      uint32_t const &a0, uint32_t const &a1, uint32_t const &b0, uint32_t const &b1,
      uint32_t const &c0, uint32_t const &c1, uint32_t const &c2, uint32_t const &c3) {
    yirage_maca::sm70_sw::mma_m8n8k4_f16_tn(d0, d1, d2, d3, a0, a1, b0, b1, c0, c1, c2, c3);
  }
};

struct SM70_8x8x4_F16F16F16F16_NN {
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  fma(uint32_t &d0, uint32_t &d1, uint32_t &d2, uint32_t &d3,
      uint32_t const &a0, uint32_t const &a1, uint32_t const &b0, uint32_t const &b1,
      uint32_t const &c0, uint32_t const &c1, uint32_t const &c2, uint32_t const &c3) {
    yirage_maca::sm70_sw::mma_m8n8k4_f16_tn(d0, d1, d2, d3, a0, a1, b0, b1, c0, c1, c2, c3);
  }
};

struct SM70_8x8x4_F16F16F16F16_TT {
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  fma(uint32_t &d0, uint32_t &d1, uint32_t &d2, uint32_t &d3,
      uint32_t const &a0, uint32_t const &a1, uint32_t const &b0, uint32_t const &b1,
      uint32_t const &c0, uint32_t const &c1, uint32_t const &c2, uint32_t const &c3) {
    yirage_maca::sm70_sw::mma_m8n8k4_f16_tn(d0, d1, d2, d3, a0, a1, b0, b1, c0, c1, c2, c3);
  }
};

struct SM70_8x8x4_F32F16F16F32_TN {
  using DRegisters = float[8];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[2];
  using CRegisters = float[8];

  CUTE_HOST_DEVICE static void
  fma(float &d0, float &d1, float &d2, float &d3, float &d4, float &d5, float &d6, float &d7,
      uint32_t const &a0, uint32_t const &a1, uint32_t const &b0, uint32_t const &b1,
      float const &c0, float const &c1, float const &c2, float const &c3, float const &c4,
      float const &c5, float const &c6, float const &c7) {
    yirage_maca::sm70_sw::mma_m8n8k4_f32f16f16f32_tn(
        d0, d1, d2, d3, d4, d5, d6, d7, a0, a1, b0, b1, c0, c1, c2, c3, c4, c5, c6, c7);
  }
};

struct SM70_8x8x4_F32F16F16F32_NT {
  using DRegisters = float[8];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[2];
  using CRegisters = float[8];

  CUTE_HOST_DEVICE static void
  fma(float &d0, float &d1, float &d2, float &d3, float &d4, float &d5, float &d6, float &d7,
      uint32_t const &a0, uint32_t const &a1, uint32_t const &b0, uint32_t const &b1,
      float const &c0, float const &c1, float const &c2, float const &c3, float const &c4,
      float const &c5, float const &c6, float const &c7) {
    yirage_maca::sm70_sw::mma_m8n8k4_f32f16f16f32_tn(
        d0, d1, d2, d3, d4, d5, d6, d7, a0, a1, b0, b1, c0, c1, c2, c3, c4, c5, c6, c7);
  }
};

struct SM70_8x8x4_F32F16F16F32_NN {
  using DRegisters = float[8];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[2];
  using CRegisters = float[8];

  CUTE_HOST_DEVICE static void
  fma(float &d0, float &d1, float &d2, float &d3, float &d4, float &d5, float &d6, float &d7,
      uint32_t const &a0, uint32_t const &a1, uint32_t const &b0, uint32_t const &b1,
      float const &c0, float const &c1, float const &c2, float const &c3, float const &c4,
      float const &c5, float const &c6, float const &c7) {
    yirage_maca::sm70_sw::mma_m8n8k4_f32f16f16f32_tn(
        d0, d1, d2, d3, d4, d5, d6, d7, a0, a1, b0, b1, c0, c1, c2, c3, c4, c5, c6, c7);
  }
};

struct SM70_8x8x4_F32F16F16F32_TT {
  using DRegisters = float[8];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[2];
  using CRegisters = float[8];

  CUTE_HOST_DEVICE static void
  fma(float &d0, float &d1, float &d2, float &d3, float &d4, float &d5, float &d6, float &d7,
      uint32_t const &a0, uint32_t const &a1, uint32_t const &b0, uint32_t const &b1,
      float const &c0, float const &c1, float const &c2, float const &c3, float const &c4,
      float const &c5, float const &c6, float const &c7) {
    yirage_maca::sm70_sw::mma_m8n8k4_f32f16f16f32_tn(
        d0, d1, d2, d3, d4, d5, d6, d7, a0, a1, b0, b1, c0, c1, c2, c3, c4, c5, c6, c7);
  }
};

} // namespace cute

#else

#include_next <cute/arch/mma_sm70.hpp>

#endif
