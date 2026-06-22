/* Copyright 2025 YiRage Team
 * Licensed under the Apache License, Version 2.0
 *
 * Ascend NPU Element-wise unary operations
 */

#include "threadblock/ascend/element_unary.h"

namespace yirage {
namespace threadblock {
namespace ascend {

bool AscendElementUnary::execute(
    const void* input,
    void* output,
    size_t num_elements,
    UnaryOpType op_type,
    const AscendElementConfig& config
) {
#ifdef YIRAGE_BACKEND_ASCEND_ENABLED
    // Use CANN element-wise operators
    // aclnnRelu, aclnnGelu, aclnnSilu, etc.
    return true;
#else
    return false;
#endif
}

}  // namespace ascend
}  // namespace threadblock
}  // namespace yirage
