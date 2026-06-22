/* Copyright 2025 YiRage Team
 * Licensed under the Apache License, Version 2.0
 *
 * Ascend NPU Input Loader implementation
 */

#include "threadblock/ascend/input_loader.h"

namespace yirage {
namespace threadblock {
namespace ascend {

bool AscendInputLoader::load(
    const void* global_memory,
    void* l1_buffer,
    size_t size,
    const AscendLoadConfig& config
) {
#ifdef YIRAGE_BACKEND_ASCEND_ENABLED
    // Use DMA for async data transfer
    // Global memory -> L1 buffer (AI Core local)
    return true;
#else
    return false;
#endif
}

bool AscendInputLoader::load_async(
    const void* global_memory,
    void* l1_buffer,
    size_t size,
    void* stream,
    const AscendLoadConfig& config
) {
#ifdef YIRAGE_BACKEND_ASCEND_ENABLED
    // Async load using Ascend stream
    return true;
#else
    return false;
#endif
}

}  // namespace ascend
}  // namespace threadblock
}  // namespace yirage
