#pragma once

#include "yirage/search/symbolic_graph/symbolic_tensor_dim.h"
#include "yirage/utils/hash_utils.h"
#include "yirage/vector_types.h"
#include <unordered_map>
#include <vector>

namespace yirage {
namespace search {

class SymbolicMap {
public:
  std::vector<SymbolicTensorDim> device_dims;
  size_t num_tensor_dims;
  std::unordered_map<std::pair<SymbolicTensorDim, size_t>,
                     std::shared_ptr<TensorDimExpr const>>
      map_mat;

  SymbolicMap(std::vector<SymbolicTensorDim> const &device_dims,
              size_t num_tensor_dims,
              tensor_dim_var_index_t &index_counter);
  SymbolicMap(std::vector<SymbolicTensorDim> const &device_dims,
              size_t num_tensor_dims,
              std::unordered_map<SymbolicTensorDim, int> const &mapped_dims);

  operator json() const;
};

} // namespace search
} // namespace yirage
