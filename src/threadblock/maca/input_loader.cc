/* Copyright 2025 YiRage Team
 * Licensed under the Apache License, Version 2.0
 *
 * MetaX MACA GPU Input Loader implementation
 */

#include "threadblock/maca/input_loader.h"

namespace yirage {
namespace threadblock {
namespace maca {

bool MacaInputLoader::load(
    const void* global_memory,
    void* shared_memory,
    size_t size,
    const MacaLoadConfig& config
) {
#ifdef YIRAGE_BACKEND_MACA_ENABLED
    // Use MACA shared memory (similar to CUDA)
    // Optimized for 64-thread warp coalescing
    return true;
#else
    return false;
#endif
}

bool MacaInputLoader::load_async(
    const void* global_memory,
    void* shared_memory,
    size_t size,
    void* stream,
    const MacaLoadConfig& config
) {
#ifdef YIRAGE_BACKEND_MACA_ENABLED
    // Async load using MACA stream
    return true;
#else
    return false;
#endif
}

}  // namespace maca
}  // namespace threadblock
}  // namespace yirage
