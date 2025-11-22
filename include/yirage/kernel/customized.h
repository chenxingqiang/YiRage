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

#pragma once

#include "yirage/kernel/device_tensor.h"
#include "yirage/kernel/operator.h"
#include "yirage/threadblock/graph.h"
#include "yirage/threadblock/operator.h"
#include "yirage/vector_types.h"
#include <tuple>

namespace yirage {
namespace kernel {

class KNCustomizedOp : public yirage::kernel::KNOperator {
public:
  KNCustomizedOp(Graph *_kgraph,
                 std::vector<DTensor> const &inputs,
                 yirage::threadblock::Graph const &_graph);
  virtual ~KNCustomizedOp();
  bool fingerprint(void);
  size_t get_owner_independent_hash() const override;

  operator json() const override;

public:
  yirage::threadblock::Graph bgraph;
  void get_bgraph(yirage::threadblock::Graph **bgraph);
};

} // namespace kernel
} // namespace yirage
