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

#include "threadblock/split.h"
#include "threadblock/graph.h"
#include <cassert>

namespace yirage {
namespace threadblock {

std::vector<STensor> Graph::split(STensor const &input,
                                  int split_size,
                                  int split_dim) {
  TBOperator *op = create_split_op(input, split_size, split_dim);
  assert(op != nullptr);
  operators.push_back(op);
  return op->output_tensors;
}

std::vector<STensor *> Graph::split(STensor const *input,
                                    int split_size,
                                    int split_dim) {
  TBOperator *op = create_split_op(*input, split_size, split_dim);
  assert(op != nullptr);
  operators.push_back(op);
  return std::vector<STensor *>{&op->output_tensors[0], &op->output_tensors[1]};
}

TBOperator *Graph::create_split_op(STensor const &input,
                                   int split_size,
                                   int split_dim) {
  if (input.num_dims <= split_dim || split_dim < 0) {
    return nullptr;
  }
  int axis = input.dim[split_dim];
  if (split_size <= 0 || split_size >= axis) {
    return nullptr;
  }

  TBOperator *op = new TBSplitOp(this, input, split_size, split_dim);
  size_t smem_usage = calculate_shared_memory_usage(op);
  if (smem_usage > yirage::config::MAX_SMEM_SIZE) {
    delete op;
    return nullptr;
  }
  return op;
}

TBSplitOp::TBSplitOp(Graph *bgraph,
                     STensor const &input,
                     int size,
                     int dim)
    : TBOperator(
          bgraph,
          (yirage::type::TBOperatorType)(yirage::type::TB_SPLIT_0_OP + dim),
          input),
      split_size(size), split_dim(dim) {
  assert(input.num_dims > split_dim);
  assert(input.layout == yirage::layout::SmemRowMajor);
  int axis = input.dim[dim];
  assert(split_size > 0 && split_size < axis);

  for (int i = 0; i < 2; ++i) {
    STensor output = input;
    output.dim[dim] = (i == 0) ? split_size : (axis - split_size);
    output.owner_op = this;
    output.owner_ts_idx = i;
    output.guid = STensor::next_guid++;
    output.after_accum = input.after_accum;
    output.smem_offset = bgraph->allocate_fingerprint(output);
    output_tensors.push_back(output);
  }
}

TBSplitOp::~TBSplitOp() {
  bgraph->free_fingerprint(output_tensors);
}

TBSplitOp::operator json() const {
  return json{{"op_type", op_type},
              {"input_tensors", input_tensors},
              {"output_tensors", output_tensors},
              {"split_size", split_size},
              {"split_dim", split_dim}};
}

} // namespace threadblock
} // namespace yirage
