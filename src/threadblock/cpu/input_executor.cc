/* Copyright 2025 YiRage Team */

#include "threadblock/cpu/input_loader.h"

#ifdef YIRAGE_USE_CPU

#include <cstring>

namespace yirage {
namespace threadblock {
namespace cpu {

void memcpy_parallel(void* dst, const void* src, size_t size) {
    // For large copies, use parallel memcpy
    if (size > 1024 * 1024) {  // > 1MB
        #pragma omp parallel
        {
            int num_threads = omp_get_num_threads();
            int thread_id = omp_get_thread_num();
            size_t chunk = size / num_threads;
            size_t start = thread_id * chunk;
            size_t end = (thread_id == num_threads - 1) ? size : start + chunk;
            
            std::memcpy(
                static_cast<char*>(dst) + start,
                static_cast<const char*>(src) + start,
                end - start
            );
        }
    } else {
        std::memcpy(dst, src, size);
    }
}

void prefetch_data(const void* ptr, size_t size) {
#if defined(__x86_64__) || defined(_M_X64)
    const char* p = static_cast<const char*>(ptr);
    for (size_t i = 0; i < size; i += 64) {
        _mm_prefetch(p + i, _MM_HINT_T0);
    }
#elif defined(__aarch64__) || defined(_M_ARM64)
    const char* p = static_cast<const char*>(ptr);
    for (size_t i = 0; i < size; i += 64) {
        __builtin_prefetch(p + i, 0, 3);
    }
#endif
}

}  // namespace cpu
}  // namespace threadblock
}  // namespace yirage

#endif
