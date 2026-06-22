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

#include "kernel/split.h"
#include "kernel/device_memory_manager.h"
#include "kernel/graph.h"
#include "utils/hash_utils.h"
#include <cassert>

namespace yirage {
namespace kernel {

std::vector<DTensor>
    Graph::split(DTensor const &input, int split_size, int dim) {
  KNOperator *op = create_split_op(input, split_size, dim);
  assert(op != nullptr);
  operators.push_back(op);
  assert(op->output_tensors.size() == 2);
  return op->output_tensors;
}

int Graph::split(DTensor const *input, int split_size, int dim) {
  return (int)split(*input, split_size, dim).size();
}

KNOperator *
    Graph::create_split_op(DTensor const &input, int split_size, int dim) {
  if (dim < 0 || dim >= input.num_dims) {
    return nullptr;
  }
  int axis = input.dim[dim];
  if (split_size <= 0 || split_size >= axis) {
    return nullptr;
  }
  if (!this->can_allocate(input)) {
    return nullptr;
  }

  KNSplitOp *op = new KNSplitOp(this, input, split_size, dim);
  return op;
}

KNSplitOp::KNSplitOp(Graph *_graph,
                     DTensor const &input,
                     int size,
                     int dim)
    : KNOperator(
          _graph, (type::KNOperatorType)(type::KN_SPLIT_0_OP + dim), input),
      split_size(size), split_dim(dim) {
  int axis = input.dim[dim];
  assert(split_size > 0 && split_size < axis);

  for (int i = 0; i < 2; ++i) {
    DTensor output_i = input;
    output_i.dim[dim] = (i == 0) ? split_size : (axis - split_size);
    output_i.owner_op = this;
    output_i.owner_ts_idx = i;
    output_i.guid = DTensor::next_guid++;
    kgraph->allocate(output_i);
    output_tensors.push_back(output_i);
  }
}

KNSplitOp::~KNSplitOp() {
  for (auto &output : output_tensors) {
    kgraph->free(output);
  }
}

KNSplitOp::operator json() const {
  return {{"op_type", op_type},
          {"input_tensors", input_tensors},
          {"output_tensors", output_tensors},
          {"split_size", split_size},
          {"split_dim", split_dim}};
}

void from_json(json const &j, KNSplitOp &op) {
  j.at("op_type").get_to(op.op_type);
  j.at("input_tensors").get_to(op.input_tensors);
  j.at("output_tensors").get_to(op.output_tensors);
  j.at("split_size").get_to(op.split_size);
  j.at("split_dim").get_to(op.split_dim);
}

#if defined(YIRAGE_FINGERPRINT_USE_CPU) || defined(YIRAGE_FINGERPRINT_USE_ASCEND)
bool KNSplitOp::fingerprint(void) {
  return true;
}
#endif

} // namespace kernel
} // namespace yirage
