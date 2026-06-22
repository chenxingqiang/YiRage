/* Copyright 2025 YiRage Team */

#include "threadblock/tpu/input_loader.h"

#ifdef YIRAGE_USE_TPU

namespace yirage {
namespace threadblock {
namespace tpu {

// TPU uses VMEM for on-chip memory
std::string generate_vmem_load_xla(int num_elements) {
    std::ostringstream hlo;
    hlo << "HloModule vmem_load\n\n";
    hlo << "ENTRY main {\n";
    hlo << "  input = bf16[" << num_elements << "] parameter(0)\n";
    hlo << "  ROOT copy = bf16[" << num_elements << "] copy(input)\n";
    hlo << "}\n";
    return hlo.str();
}

}  // namespace tpu
}  // namespace threadblock
}  // namespace yirage

#endif
