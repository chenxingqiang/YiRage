/* Copyright 2025 Chen Xingqiang (YiRage Project)
 * SPDX-License-Identifier: Apache-2.0
 */

#include "kernel/conv2d.h"
#include "kernel/device_memory_manager.h"
#include "kernel/graph.h"
#include "utils/hash_utils.h"
#include <cassert>

namespace yirage {
namespace kernel {

namespace {

int conv2d_out_dim(int in,
                   int pad,
                   int dilation,
                   int kernel,
                   int stride) {
  return (in + 2 * pad - dilation * (kernel - 1) - 1) / stride + 1;
}

} // namespace

DTensor Graph::conv2d(DTensor const &input,
                      DTensor const &weight,
                      int stride_h,
                      int stride_w,
                      int padding_h,
                      int padding_w,
                      int dilation_h,
                      int dilation_w) {
  KNOperator *op = create_conv2d_op(input,
                                    weight,
                                    stride_h,
                                    stride_w,
                                    padding_h,
                                    padding_w,
                                    dilation_h,
                                    dilation_w);
  assert(op != nullptr);
  operators.push_back(op);
  assert(op->output_tensors.size() == 1);
  return op->output_tensors[0];
}

DTensor *Graph::conv2d(DTensor const *input,
                       DTensor const *weight,
                       int stride_h,
                       int stride_w,
                       int padding_h,
                       int padding_w,
                       int dilation_h,
                       int dilation_w) {
  KNOperator *op = create_conv2d_op(*input,
                                    *weight,
                                    stride_h,
                                    stride_w,
                                    padding_h,
                                    padding_w,
                                    dilation_h,
                                    dilation_w);
  assert(op != nullptr);
  operators.push_back(op);
  assert(op->output_tensors.size() == 1);
  return &op->output_tensors[0];
}

KNOperator *Graph::create_conv2d_op(DTensor const &input,
                                    DTensor const &weight,
                                    int stride_h,
                                    int stride_w,
                                    int padding_h,
                                    int padding_w,
                                    int dilation_h,
                                    int dilation_w) {
  if (input.num_dims != 4 || weight.num_dims != 4) {
    return nullptr;
  }
  if (input.dim[1] != weight.dim[1]) {
    return nullptr;
  }
  if (stride_h <= 0 || stride_w <= 0 || dilation_h <= 0 || dilation_w <= 0) {
    return nullptr;
  }
  int h_out = conv2d_out_dim(input.dim[2],
                             padding_h,
                             dilation_h,
                             weight.dim[2],
                             stride_h);
  int w_out = conv2d_out_dim(input.dim[3],
                             padding_w,
                             dilation_w,
                             weight.dim[3],
                             stride_w);
  if (h_out <= 0 || w_out <= 0) {
    return nullptr;
  }
  DTensor output;
  output.num_dims = 4;
  output.dim[0] = input.dim[0];
  output.dim[1] = weight.dim[0];
  output.dim[2] = h_out;
  output.dim[3] = w_out;
  output.data_type = input.data_type;
  output.layout = input.layout;
  if (!this->can_allocate(output)) {
    return nullptr;
  }
  return new KNConv2dOp(this,
                        input,
                        weight,
                        stride_h,
                        stride_w,
                        padding_h,
                        padding_w,
                        dilation_h,
                        dilation_w);
}

KNConv2dOp::KNConv2dOp(Graph *_graph,
                       DTensor const &input,
                       DTensor const &weight,
                       int stride_h_,
                       int stride_w_,
                       int padding_h_,
                       int padding_w_,
                       int dilation_h_,
                       int dilation_w_)
    : KNOperator(_graph, type::KN_CONV2D_OP, input, weight),
      stride_h(stride_h_),
      stride_w(stride_w_),
      padding_h(padding_h_),
      padding_w(padding_w_),
      dilation_h(dilation_h_),
      dilation_w(dilation_w_) {
  assert(input.num_dims == 4);
  assert(weight.num_dims == 4);
  assert(input.dim[1] == weight.dim[1]);
  DTensor output;
  output.num_dims = 4;
  output.dim[0] = input.dim[0];
  output.dim[1] = weight.dim[0];
  output.dim[2] = conv2d_out_dim(input.dim[2],
                               padding_h,
                               dilation_h,
                               weight.dim[2],
                               stride_h);
  output.dim[3] = conv2d_out_dim(input.dim[3],
                               padding_w,
                               dilation_w,
                               weight.dim[3],
                               stride_w);
  output.data_type = input.data_type;
  output.layout = input.layout;
  output.owner_op = this;
  output.owner_ts_idx = 0;
  output.guid = DTensor::next_guid++;
  kgraph->allocate(output);
  output_tensors.push_back(output);
}

KNConv2dOp::~KNConv2dOp() {
  for (auto &output : output_tensors) {
    kgraph->free(output);
  }
}

KNConv2dOp::operator json() const {
  return json{{"op_type", op_type},
              {"input_tensors", input_tensors},
              {"output_tensors", output_tensors},
              {"stride_h", stride_h},
              {"stride_w", stride_w},
              {"padding_h", padding_h},
              {"padding_w", padding_w},
              {"dilation_h", dilation_h},
              {"dilation_w", dilation_w}};
}

void from_json(json const &j, KNConv2dOp &op) {
  j.at("op_type").get_to(op.op_type);
  j.at("input_tensors").get_to(op.input_tensors);
  j.at("output_tensors").get_to(op.output_tensors);
  j.at("stride_h").get_to(op.stride_h);
  j.at("stride_w").get_to(op.stride_w);
  j.at("padding_h").get_to(op.padding_h);
  j.at("padding_w").get_to(op.padding_w);
  j.at("dilation_h").get_to(op.dilation_h);
  j.at("dilation_w").get_to(op.dilation_w);
}

bool kn_operator_conv2d_params(KNOperator const *op,
                               int *stride_h,
                               int *stride_w,
                               int *padding_h,
                               int *padding_w,
                               int *dilation_h,
                               int *dilation_w) {
  if (op == nullptr || op->op_type != yirage::type::KN_CONV2D_OP) {
    return false;
  }
  auto const *cop = static_cast<KNConv2dOp const *>(op);
  *stride_h = cop->stride_h;
  *stride_w = cop->stride_w;
  *padding_h = cop->padding_h;
  *padding_w = cop->padding_w;
  *dilation_h = cop->dilation_h;
  *dilation_w = cop->dilation_w;
  return true;
}

#if defined(YIRAGE_FINGERPRINT_USE_CPU) || defined(YIRAGE_FINGERPRINT_USE_ASCEND)
bool KNConv2dOp::fingerprint(void) {
  return true;
}
#endif

} // namespace kernel
} // namespace yirage
