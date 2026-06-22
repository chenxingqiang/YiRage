/* Copyright 2025 YiRage Team */

#include "threadblock/element_unary.h"
#include "threadblock/tpu/element_unary.h"

#ifdef YIRAGE_USE_TPU

namespace yirage {
namespace threadblock {
namespace tpu {

std::string generate_element_unary_xla(const std::string& op_name, int num_elements) {
    std::ostringstream hlo;
    hlo << "HloModule " << op_name << "_fingerprint\n\n";
    hlo << "ENTRY main {\n";
    hlo << "  input = bf16[" << num_elements << "] parameter(0)\n";
    hlo << "  ROOT output = bf16[" << num_elements << "] " << op_name << "(input)\n";
    hlo << "}\n";
    return hlo.str();
}

}  // namespace tpu
}  // namespace threadblock
}  // namespace yirage

#endif
