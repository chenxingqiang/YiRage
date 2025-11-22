#pragma once

#include "yirage/search/symbolic_graph/symbolic_tensor_dim.h"

#include <vector>

namespace yirage {
namespace search {

class SymbolicDTensor {
public:
  SymbolicDTensor(std::vector<SymbolicTensorDim> dim_templates);

  std::vector<SymbolicTensorDim> dims;

  operator json() const;
};

class SymbolicSTensor {
public:
  SymbolicSTensor(std::vector<SymbolicTensorDim> dim_templates,
                  bool after_accum);

  std::vector<SymbolicTensorDim> dims;
  bool after_accum;

  operator json() const;
};

} // namespace search
} // namespace yirage
