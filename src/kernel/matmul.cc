/* Copyright 2023-2024 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "kernel/matmul.h"
#include "kernel/device_memory_manager.h"
#include "kernel/graph.h"
#include "layout.h"
#include "utils/fingerprint_functions.h"
#include "utils/hash_utils.h"
#include <cassert>
#include <iostream>
#include <stdexcept>

namespace yirage {
namespace kernel {

namespace {

bool kn_matmul_output_shape(DTensor const &A, DTensor const &B, DTensor &C) {
  int const nd_a = A.num_dims;
  int const nd_b = B.num_dims;
  if (nd_a < 2 || nd_b < 2) {
    return false;
  }
  if (A.dim[nd_a - 1] != B.dim[nd_b - 2]) {
    return false;
  }
  if (nd_a == nd_b) {
    for (int i = 0; i < nd_a - 2; ++i) {
      if (A.dim[i] != B.dim[i]) {
        return false;
      }
    }
    C.num_dims = nd_a;
    for (int i = 0; i < nd_a; ++i) {
      C.dim[i] = A.dim[i];
    }
    C.dim[nd_a - 1] = B.dim[nd_b - 1];
    return true;
  }
  if (nd_a == nd_b + 1) {
    for (int i = 0; i < nd_b - 2; ++i) {
      if (A.dim[i] != B.dim[i]) {
        return false;
      }
    }
    C.num_dims = nd_a;
    for (int i = 0; i < nd_a - 1; ++i) {
      C.dim[i] = A.dim[i];
    }
    C.dim[nd_a - 1] = B.dim[nd_b - 1];
    return true;
  }
  return false;
}

} // namespace

DTensor Graph::matmul(DTensor const &A, DTensor const &B) {
  KNOperator *op = create_matmul_op(A, B);
  if (op == nullptr) {
    throw std::invalid_argument(
        "kernel::Graph::matmul: incompatible operands (inner dimension or batch "
        "shape mismatch, dtype/layout issue, or output allocation failed)");
  }
  operators.push_back(op);
  DTensor output = op->output_tensors[0];
  return output;
}

DTensor *Graph::matmul(DTensor const *A, DTensor const *B) {
  KNOperator *op = create_matmul_op(*A, *B);
  if (op == nullptr) {
    throw std::invalid_argument(
        "kernel::Graph::matmul: incompatible operands (inner dimension or batch "
        "shape mismatch, dtype/layout issue, or output allocation failed)");
  }
  operators.push_back(op);
  return &op->output_tensors[0];
}

KNOperator *Graph::create_matmul_op(DTensor const &A, DTensor const &B) {
  DTensor C;
  if (!kn_matmul_output_shape(A, B, C)) {
    return nullptr;
  }
  C.data_type = A.data_type;

  if (!can_allocate(C)) {
    return nullptr;
  }

  KNMatmulOp *op = new KNMatmulOp(this, A, B);
  return op;
}

KNMatmulOp::KNMatmulOp(Graph *_kgraph, DTensor const &A, DTensor const &B)
    : KNOperator(_kgraph, yirage::type::KN_MATMUL_OP, A, B) {
  DTensor C;
  assert(kn_matmul_output_shape(A, B, C));
  // Currently only support row-major output
  // to be consistent with cutlass
  C.layout = yirage::layout::DmemRowMajor;
  C.data_type = A.data_type;
  C.owner_op = this;
  C.owner_ts_idx = 0;
  C.guid = DTensor::next_guid++;
  kgraph->allocate(C);
  assert(output_tensors.size() == 0);
  output_tensors.push_back(C);
}

KNMatmulOp::~KNMatmulOp() {
  for (int i = output_tensors.size() - 1; i >= 0; i--) {
    kgraph->free(output_tensors[i]);
  }
}

KNMatmulOp::operator json() const {
  return json{{"op_type", op_type},
              {"input_tensors", input_tensors},
              {"output_tensors", output_tensors}};
}

void from_json(json const &j, KNMatmulOp &op) {
  j.at("op_type").get_to(op.op_type);
  j.at("input_tensors").get_to(op.input_tensors);
  j.at("output_tensors").get_to(op.output_tensors);
}

#if defined(YIRAGE_FINGERPRINT_USE_CPU) || defined(YIRAGE_FINGERPRINT_USE_ASCEND)
bool KNMatmulOp::fingerprint(void) {
  DeviceMemoryManager *dmm = DeviceMemoryManager::get_instance();
  
  // Safety check: ensure fp_offset is valid for all tensors
  if (input_tensors[0].fp_offset < 0 || input_tensors[1].fp_offset < 0 || 
      output_tensors[0].fp_offset < 0) {
    std::cerr << "[ERROR] KNMatmulOp::fingerprint: Invalid fp_offset detected "
              << "(A=" << input_tensors[0].fp_offset 
              << ", B=" << input_tensors[1].fp_offset 
              << ", C=" << output_tensors[0].fp_offset << ")" << std::endl;
    return false;
  }

  int B = 1;
  for (int i = 0; i < input_tensors[0].num_dims - 2; ++i) {
    B *= input_tensors[0].dim[i];
  }
  int M = input_tensors[0].dim[input_tensors[0].num_dims - 2];
  int N = input_tensors[1].dim[input_tensors[1].num_dims - 1];
  int K = input_tensors[0].dim[input_tensors[0].num_dims - 1];

  for (int device_id = 0; device_id < kgraph->gpu_dim.x; ++device_id) {
    type::FPType *A_ptr = reinterpret_cast<type::FPType *>(
        dmm->fp_base_ptr[device_id] + input_tensors[0].fp_offset);
    type::FPType *B_ptr = reinterpret_cast<type::FPType *>(
        dmm->fp_base_ptr[device_id] + input_tensors[1].fp_offset);
    type::FPType *C_ptr = reinterpret_cast<type::FPType *>(
        dmm->fp_base_ptr[device_id] + output_tensors[0].fp_offset);
    utils::compute_matmul_fingerprint(A_ptr, B_ptr, C_ptr, B, M, N, K);
  }

  return true;
}
#endif // YIRAGE_FINGERPRINT_USE_CPU || YIRAGE_FINGERPRINT_USE_ASCEND

} // namespace kernel
} // namespace yirage
