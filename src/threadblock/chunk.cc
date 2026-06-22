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

#include "threadblock/chunk.h"
#include "threadblock/graph.h"
#include <cassert>

namespace yirage {
namespace threadblock {

std::vector<STensor> Graph::chunk(STensor const &input,
                                  int chunk_size,
                                  int chunk_dim) {
  TBOperator *op = create_chunk_op(input, chunk_size, chunk_dim);
  assert(op != nullptr);
  operators.push_back(op);
  return op->output_tensors;
}

std::vector<STensor *> Graph::chunk(STensor const *input,
                                    int chunk_size,
                                    int chunk_dim) {
  TBOperator *op = create_chunk_op(*input, chunk_size, chunk_dim);
  assert(op != nullptr);
  operators.push_back(op);
  std::vector<STensor *> outputs;
  for (STensor &tensor : op->output_tensors) {
    outputs.push_back(&tensor);
  }
  return outputs;
}

TBOperator *Graph::create_chunk_op(STensor const &input,
                                   int chunk_size,
                                   int chunk_dim) {
  if (input.num_dims <= chunk_dim || chunk_dim < 0 || chunk_size <= 0) {
    return nullptr;
  }
  int axis = input.dim[chunk_dim];
  if (axis % chunk_size != 0) {
    return nullptr;
  }

  TBOperator *op = new TBChunkOp(this, input, chunk_size, chunk_dim);
  size_t smem_usage = calculate_shared_memory_usage(op);
  if (smem_usage > yirage::config::MAX_SMEM_SIZE) {
    delete op;
    return nullptr;
  }
  return op;
}

TBChunkOp::TBChunkOp(Graph *bgraph,
                     STensor const &input,
                     int num_chunks,
                     int dim)
    : TBOperator(
          bgraph,
          (yirage::type::TBOperatorType)(yirage::type::TB_CHUNK_0_OP + dim),
          input),
      chunk_size(num_chunks), chunk_dim(dim) {
  assert(input.num_dims > dim);
  assert(input.layout == yirage::layout::SmemRowMajor);
  assert(input.dim[dim] % num_chunks == 0);

  int part_size = input.dim[dim] / num_chunks;
  for (int i = 0; i < num_chunks; ++i) {
    STensor output = input;
    output.dim[dim] = part_size;
    output.owner_op = this;
    output.owner_ts_idx = i;
    output.guid = STensor::next_guid++;
    output.after_accum = input.after_accum;
    output.smem_offset = bgraph->allocate_fingerprint(output);
    output_tensors.push_back(output);
  }
}

TBChunkOp::~TBChunkOp() {
  bgraph->free_fingerprint(output_tensors);
}

TBChunkOp::operator json() const {
  return json{{"op_type", op_type},
              {"input_tensors", input_tensors},
              {"output_tensors", output_tensors},
              {"chunk_size", chunk_size},
              {"chunk_dim", chunk_dim}};
}

} // namespace threadblock
} // namespace yirage
