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

#include "kernel/element_unary.h"
#include "kernel/device_memory_manager.h"
#include "kernel/graph.h"
#include "layout.h"
#include "utils/hash_utils.h"
#include <cassert>

namespace yirage {
namespace kernel {

DTensor Graph::exp(DTensor const &input) {
  return elementunary(input, yirage::type::KN_EXP_OP);
}

DTensor *Graph::exp(DTensor const *input) {
  return elementunary(input, yirage::type::KN_EXP_OP);
}

DTensor Graph::square(DTensor const &input) {
  return elementunary(input, yirage::type::KN_SQUARE_OP);
}

DTensor *Graph::square(DTensor const *input) {
  return elementunary(input, yirage::type::KN_SQUARE_OP);
}

DTensor Graph::sqrt(DTensor const &input) {
  return elementunary(input, yirage::type::KN_SQRT_OP);
}

DTensor *Graph::sqrt(DTensor const *input) {
  return elementunary(input, yirage::type::KN_SQRT_OP);
}

DTensor Graph::silu(DTensor const &input) {
  return elementunary(input, yirage::type::KN_SILU_OP);
}

DTensor *Graph::silu(DTensor const *input) {
  return elementunary(input, yirage::type::KN_SILU_OP);
}

DTensor Graph::gelu(DTensor const &input) {
  return elementunary(input, yirage::type::KN_GELU_OP);
}

DTensor *Graph::gelu(DTensor const *input) {
  return elementunary(input, yirage::type::KN_GELU_OP);
}

DTensor Graph::relu(DTensor const &input) {
  return elementunary(input, yirage::type::KN_RELU_OP);
}

DTensor *Graph::relu(DTensor const *input) {
  return elementunary(input, yirage::type::KN_RELU_OP);
}

DTensor Graph::sigmoid(DTensor const &input) {
  return elementunary(input, yirage::type::KN_SIGMOID_OP);
}

DTensor *Graph::sigmoid(DTensor const *input) {
  return elementunary(input, yirage::type::KN_SIGMOID_OP);
}

DTensor Graph::log(DTensor const &input) {
  return elementunary(input, yirage::type::KN_LOG_OP);
}

DTensor *Graph::log(DTensor const *input) {
  return elementunary(input, yirage::type::KN_LOG_OP);
}

DTensor Graph::clamp(DTensor const &input,
                     float const &min_val,
                     float const &max_val) {
  type::CLAMP_MIN_MAX["min_val"] = min_val;
  type::CLAMP_MIN_MAX["max_val"] = max_val;
  return elementunary_clamp(input, min_val, max_val);
}

DTensor *Graph::clamp(DTensor const *input,
                      float const &min_val,
                      float const &max_val) {
  type::CLAMP_MIN_MAX["min_val"] = min_val;
  type::CLAMP_MIN_MAX["max_val"] = max_val;
  return elementunary_clamp(input, min_val, max_val);
}

DTensor Graph::mul_scalar(DTensor const &input, float const &scalar) {
  KNOperator *op = create_elementunary_op(input, yirage::type::KN_MUL_SCALAR_OP, scalar);
  assert(op != nullptr);
  operators.push_back(op);
  assert(op->output_tensors.size() == 1);
  return op->output_tensors[0];
}

DTensor *Graph::mul_scalar(DTensor const *input, float const &scalar) {
  KNOperator *op =
      create_elementunary_op(*input, yirage::type::KN_MUL_SCALAR_OP, scalar);
  assert(op != nullptr);
  operators.push_back(op);
  assert(op->output_tensors.size() == 1);
  return &op->output_tensors[0];
}

DTensor Graph::elementunary_clamp(DTensor const &input,
                                  float const &min_val,
                                  float const &max_val) {
  KNOperator *op = create_elementunary_clamp_op(input, min_val, max_val);
  assert(op != nullptr);
  operators.push_back(op);
  assert(op->output_tensors.size() == 1);
  DTensor output = op->output_tensors[0];
  return output;
}

DTensor *Graph::elementunary_clamp(DTensor const *input,
                                   float const &min_val,
                                   float const &max_val) {
  KNOperator *op = create_elementunary_clamp_op(*input, min_val, max_val);
  assert(op != nullptr);
  operators.push_back(op);
  assert(op->output_tensors.size() == 1);
  return &op->output_tensors[0];
}

KNOperator *Graph::create_elementunary_clamp_op(DTensor const &input,
                                                float const &min_val,
                                                float const &max_val) {
  if (!can_allocate(input)) {
    return nullptr;
  }

  KNElementUnaryOp *op = new KNClampUnaryOp(this, input, min_val, max_val);

  return op;
}

DTensor Graph::elementunary(DTensor const &input,
                            yirage::type::KNOperatorType type) {
  KNOperator *op = create_elementunary_op(input, type);
  assert(op != nullptr);
  operators.push_back(op);
  assert(op->output_tensors.size() == 1);
  DTensor output = op->output_tensors[0];
  return output;
}

DTensor *Graph::elementunary(DTensor const *input,
                             yirage::type::KNOperatorType type) {
  KNOperator *op = create_elementunary_op(*input, type);
  assert(op != nullptr);
  operators.push_back(op);
  assert(op->output_tensors.size() == 1);
  return &op->output_tensors[0];
}

KNOperator *Graph::create_elementunary_op(DTensor const &input,
                                          yirage::type::KNOperatorType type,
                                          float scalar) {
  if (!can_allocate(input)) {
    return nullptr;
  }

  KNElementUnaryOp *op = new KNElementUnaryOp(this, input, type, scalar);

  return op;
}

KNClampUnaryOp::KNClampUnaryOp(Graph *_kgraph,
                               DTensor const &input,
                               float min_val,
                               float max_val)
    : KNElementUnaryOp(_kgraph, input, yirage::type::KN_CLAMP_OP, 0.0f),
      min_val(min_val), max_val(max_val) {}

KNClampUnaryOp::operator json() const {
  return json{{"op_type", op_type},
              {"input_tensors", input_tensors},
              {"output_tensors", output_tensors},
              {"min_val", min_val},
              {"max_val", max_val}};
}

bool kn_operator_clamp_bounds(KNOperator const *op,
                              float *min_val,
                              float *max_val) {
  if (op == nullptr || op->op_type != yirage::type::KN_CLAMP_OP) {
    return false;
  }
  auto const *cop = static_cast<KNClampUnaryOp const *>(op);
  *min_val = cop->min_val;
  *max_val = cop->max_val;
  return true;
}

bool kn_operator_mul_scalar_value(KNOperator const *op, float *scalar) {
  if (op == nullptr || op->op_type != yirage::type::KN_MUL_SCALAR_OP) {
    return false;
  }
  *scalar = static_cast<KNElementUnaryOp const *>(op)->scalar;
  return true;
}

KNElementUnaryOp::KNElementUnaryOp(Graph *_kgraph,
                                   DTensor const &input,
                                   yirage::type::KNOperatorType type,
                                   float scalar)
    : yirage::kernel::KNOperator(_kgraph, type, input), scalar(scalar) {
  DTensor output = input;
  output.owner_op = this;
  output.owner_ts_idx = 0;
  output.guid = DTensor::next_guid++;
  kgraph->allocate(output);
  assert(output_tensors.size() == 0);
  output_tensors.push_back(output);
}

KNElementUnaryOp::~KNElementUnaryOp() {
  for (int i = output_tensors.size() - 1; i >= 0; i--) {
    kgraph->free(output_tensors[i]);
  }
}

KNElementUnaryOp::operator json() const {
  json j{{"op_type", op_type},
         {"input_tensors", input_tensors},
         {"output_tensors", output_tensors}};
  if (op_type == yirage::type::KN_MUL_SCALAR_OP) {
    j["scalar"] = scalar;
  }
  return j;
}

#if defined(YIRAGE_FINGERPRINT_USE_CPU) || defined(YIRAGE_FINGERPRINT_USE_ASCEND)
bool KNElementUnaryOp::fingerprint(void) {
  // CPU/Ascend fingerprint - simplified implementation
  return true;
}
#endif

} // namespace kernel
} // namespace yirage
