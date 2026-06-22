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

#include "kernel/concat.h"
#include "kernel/device_memory_manager.h"
#include "kernel/graph.h"
#include "utils/hash_utils.h"
#include <cassert>

namespace yirage {
namespace kernel {

DTensor Graph::concat(DTensor const &input1,
                      DTensor const &input2,
                      int concat_dim) {
  KNOperator *op = create_concat_op(input1, input2, concat_dim);
  assert(op != nullptr);
  operators.push_back(op);
  assert(op->output_tensors.size() == 1);
  return op->output_tensors[0];
}

DTensor *Graph::concat(DTensor const *input1,
                       DTensor const *input2,
                       int concat_dim) {
  KNOperator *op = create_concat_op(*input1, *input2, concat_dim);
  assert(op != nullptr);
  operators.push_back(op);
  return &op->output_tensors[0];
}

KNOperator *Graph::create_concat_op(DTensor const &input1,
                                    DTensor const &input2,
                                    int concat_dim) {
  if (input1.num_dims != input2.num_dims) {
    return nullptr;
  }
  if (input1.num_dims <= concat_dim || concat_dim < 0) {
    return nullptr;
  }
  for (int i = 0; i < input1.num_dims; i++) {
    if (i != concat_dim && input1.dim[i] != input2.dim[i]) {
      return nullptr;
    }
  }
  DTensor output = input1;
  output.dim[concat_dim] = input1.dim[concat_dim] + input2.dim[concat_dim];
  if (!can_allocate(output)) {
    return nullptr;
  }

  KNConcatOp *op = new KNConcatOp(this, input1, input2, concat_dim);
  return op;
}

KNConcatOp::KNConcatOp(Graph *_graph,
                       DTensor const &input1,
                       DTensor const &input2,
                       int dim)
    : KNOperator(_graph,
                 (type::KNOperatorType)(type::KN_CONCAT_0_OP + dim),
                 input1,
                 input2),
      concat_dim(dim) {
  assert(input1.num_dims > concat_dim);
  DTensor output = input1;
  output.dim[concat_dim] = input1.dim[concat_dim] + input2.dim[concat_dim];
  output.owner_op = this;
  output.owner_ts_idx = 0;
  output.guid = DTensor::next_guid++;
  kgraph->allocate(output);
  output_tensors.push_back(output);
}

KNConcatOp::~KNConcatOp() {
  for (auto &output : output_tensors) {
    kgraph->free(output);
  }
}

KNConcatOp::operator json() const {
  return {{"op_type", op_type},
          {"input_tensors", input_tensors},
          {"output_tensors", output_tensors},
          {"concat_dim", concat_dim}};
}

void from_json(json const &j, KNConcatOp &op) {
  j.at("op_type").get_to(op.op_type);
  j.at("input_tensors").get_to(op.input_tensors);
  j.at("output_tensors").get_to(op.output_tensors);
  j.at("concat_dim").get_to(op.concat_dim);
}

#if defined(YIRAGE_FINGERPRINT_USE_CPU) || defined(YIRAGE_FINGERPRINT_USE_ASCEND)
bool KNConcatOp::fingerprint(void) {
  return true;
}
#endif

} // namespace kernel
} // namespace yirage
