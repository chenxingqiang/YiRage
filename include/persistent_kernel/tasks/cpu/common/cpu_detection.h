/* Copyright 2025 YiRage Team */
#pragma once

#include "cpu_common.h"

#if defined(__x86_64__) || defined(_M_X64)
#include <cpuid.h>
#endif

namespace yirage {
namespace persistent_kernel {
namespace cpu {

/**
 * @brief Detect CPU SIMD architecture at runtime
 */
inline CPUArch detect_cpu_arch() {
#if defined(__x86_64__) || defined(_M_X64)
    // x86-64 detection via CPUID
    unsigned int eax, ebx, ecx, edx;
    
    // Check for AVX-512
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        if (ebx & (1 << 16)) {  // AVX-512F
            // Check for AMX
            if (edx & (1 << 22)) {  // AMX-TILE
                return CPUArch::X86_AMX;
            }
            // Check for BF16
            if (eax & (1 << 5)) {  // AVX512_BF16
                return CPUArch::X86_AVX512_BF16;
            }
            return CPUArch::X86_AVX512;
        }
    }
    
    // Check for AVX2
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        if (ebx & (1 << 5)) {  // AVX2
            return CPUArch::X86_AVX2;
        }
    }
    
    // Check for AVX
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        if (ecx & (1 << 28)) {  // AVX
            return CPUArch::X86_AVX;
        }
    }
    
    return CPUArch::X86_SSE;
    
#elif defined(__aarch64__) || defined(_M_ARM64)
    // ARM detection
    #if defined(__ARM_FEATURE_SME)
        return CPUArch::ARM_SME;
    #elif defined(__ARM_FEATURE_SVE)
        return CPUArch::ARM_SVE;
    #else
        return CPUArch::ARM_NEON;
    #endif
    
#else
    return CPUArch::UNKNOWN;
#endif
}

/**
 * @brief Detect CPU vendor
 */
inline CPUVendor detect_cpu_vendor() {
#if defined(__x86_64__) || defined(_M_X64)
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid(0, &eax, &ebx, &ecx, &edx)) {
        char vendor[13];
        *reinterpret_cast<unsigned int*>(&vendor[0]) = ebx;
        *reinterpret_cast<unsigned int*>(&vendor[4]) = edx;
        *reinterpret_cast<unsigned int*>(&vendor[8]) = ecx;
        vendor[12] = '\0';
        
        if (ebx == 0x756e6547) return CPUVendor::INTEL;  // "GenuineIntel"
        if (ebx == 0x68747541) return CPUVendor::AMD;    // "AuthenticAMD"
    }
    return CPUVendor::UNKNOWN;
    
#elif defined(__aarch64__) || defined(_M_ARM64)
    #if defined(__APPLE__)
        return CPUVendor::APPLE;
    #else
        return CPUVendor::ARM;
    #endif
#else
    return CPUVendor::UNKNOWN;
#endif
}

/**
 * @brief Get number of physical CPU cores
 */
inline int get_cpu_core_count() {
#if defined(_OPENMP)
    #include <omp.h>
    return omp_get_max_threads();
#else
    return 1;
#endif
}

/**
 * @brief Check if specific feature is available
 */
inline bool has_avx2() {
    return detect_cpu_arch() >= CPUArch::X86_AVX2;
}

inline bool has_avx512() {
    CPUArch arch = detect_cpu_arch();
    return arch == CPUArch::X86_AVX512 || 
           arch == CPUArch::X86_AVX512_BF16 || 
           arch == CPUArch::X86_AMX;
}

inline bool has_neon() {
    CPUArch arch = detect_cpu_arch();
    return arch == CPUArch::ARM_NEON || 
           arch == CPUArch::ARM_SVE || 
           arch == CPUArch::ARM_SVE2 ||
           arch == CPUArch::ARM_SME;
}

}  // namespace cpu
}  // namespace persistent_kernel
}  // namespace yirage
