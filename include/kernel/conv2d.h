/* Copyright 2025 Chen Xingqiang (YiRage Project)
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "kernel/operator.h"

namespace yirage {
namespace kernel {

class KNConv2dOp : public yirage::kernel::KNOperator {
public:
  KNConv2dOp(Graph *_graph,
             DTensor const &input,
             DTensor const &weight,
             int stride_h,
             int stride_w,
             int padding_h,
             int padding_w,
             int dilation_h,
             int dilation_w,
             int groups = 1);
  ~KNConv2dOp();
  bool fingerprint(void) override;

  operator json() const override;

public:
  int stride_h;
  int stride_w;
  int padding_h;
  int padding_w;
  int dilation_h;
  int dilation_w;
  int groups;
};

void from_json(json const &j, KNConv2dOp &op);

bool kn_operator_conv2d_params(KNOperator const *op,
                               int *stride_h,
                               int *stride_w,
                               int *padding_h,
                               int *padding_w,
                               int *dilation_h,
                               int *dilation_w,
                               int *groups);

} // namespace kernel
} // namespace yirage
