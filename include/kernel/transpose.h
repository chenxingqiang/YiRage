/* Copyright 2025 Chen Xingqiang (YiRage Project)
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "kernel/operator.h"

namespace yirage {
namespace kernel {

class KNTranspose01Op : public yirage::kernel::KNOperator {
public:
  KNTranspose01Op(Graph *_graph, DTensor const &input);
  ~KNTranspose01Op();
  bool fingerprint(void) override;

  operator json() const override;
};

void from_json(json const &j, KNTranspose01Op &op);

} // namespace kernel
} // namespace yirage
