/* Copyright 2025 Chen Xingqiang (YiRage Project)
 * SPDX-License-Identifier: Apache-2.0
 */

#include "kernel/transpose.h"
#include "kernel/device_memory_manager.h"
#include "kernel/graph.h"
#include "utils/hash_utils.h"
#include <cassert>

namespace yirage {
namespace kernel {

DTensor Graph::transpose01(DTensor const &input) {
  KNOperator *op = create_transpose01_op(input);
  assert(op != nullptr);
  operators.push_back(op);
  assert(op->output_tensors.size() == 1);
  return op->output_tensors[0];
}

DTensor *Graph::transpose01(DTensor const *input) {
  KNOperator *op = create_transpose01_op(*input);
  assert(op != nullptr);
  operators.push_back(op);
  assert(op->output_tensors.size() == 1);
  return &op->output_tensors[0];
}

KNOperator *Graph::create_transpose01_op(DTensor const &input) {
  if (input.num_dims < 2) {
    return nullptr;
  }
  if (!this->can_allocate(input)) {
    return nullptr;
  }
  return new KNTranspose01Op(this, input);
}

KNTranspose01Op::KNTranspose01Op(Graph *_graph, DTensor const &input)
    : KNOperator(_graph, type::KN_TRANSPOSE_01_OP, input) {
  assert(input.num_dims >= 2);
  DTensor output = input;
  std::swap(output.dim[0], output.dim[1]);
  output.owner_op = this;
  output.owner_ts_idx = 0;
  output.guid = DTensor::next_guid++;
  kgraph->allocate(output);
  output_tensors.push_back(output);
}

KNTranspose01Op::~KNTranspose01Op() {
  for (auto &output : output_tensors) {
    kgraph->free(output);
  }
}

KNTranspose01Op::operator json() const {
  return {{"op_type", op_type},
          {"input_tensors", input_tensors},
          {"output_tensors", output_tensors}};
}

void from_json(json const &j, KNTranspose01Op &op) {
  j.at("op_type").get_to(op.op_type);
  j.at("input_tensors").get_to(op.input_tensors);
  j.at("output_tensors").get_to(op.output_tensors);
}

#if defined(YIRAGE_FINGERPRINT_USE_CPU) || defined(YIRAGE_FINGERPRINT_USE_ASCEND)
bool KNTranspose01Op::fingerprint(void) {
  return true;
}
#endif

} // namespace kernel
} // namespace yirage
