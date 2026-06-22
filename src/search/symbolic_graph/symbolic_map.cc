#include "search/symbolic_graph/symbolic_map.h"

namespace yirage {
namespace search {

SymbolicMap::SymbolicMap(std::vector<SymbolicTensorDim> const &device_dims,
                         size_t num_tensor_dims,
                         tensor_dim_var_index_t &index_counter)
    : device_dims(device_dims), num_tensor_dims(num_tensor_dims) {
  for (SymbolicTensorDim const &dim : device_dims) {
    for (size_t j = 0; j < num_tensor_dims; ++j) {
      map_mat[{dim, j}] =
          dim_expr_make_var(index_counter++, TensorDimVarType::INT);
    }
  }
}

SymbolicMap::SymbolicMap(
    std::vector<SymbolicTensorDim> const &device_dims,
    size_t num_tensor_dims,
    std::unordered_map<SymbolicTensorDim, int> const &mapped_dims)
    : device_dims(device_dims), num_tensor_dims(num_tensor_dims) {
  for (SymbolicTensorDim const &dim : device_dims) {
    for (size_t j = 0; j < num_tensor_dims; ++j) {
      if (mapped_dims.at(dim) == j) {
        map_mat[{dim, j}] = dim_expr_make_const(1);
      } else {
        map_mat[{dim, j}] = dim_expr_make_const(0);
      }
    }
  }
}

SymbolicMap::operator json() const {
  json j;
  j["num_tensor_dims"] = num_tensor_dims;
  j["num_device_dims"] = device_dims.size();
  
  // Serialize the map matrix entries
  json map_entries = json::array();
  for (auto const& [key, expr] : map_mat) {
    json entry;
    // key.first is SymbolicTensorDim - use its JSON conversion
    entry["device_dim"] = json(key.first);  // Uses operator json() const
    entry["tensor_dim"] = key.second;
    // Serialize expression as string representation
    if (expr->is_const()) {
      entry["value"] = std::static_pointer_cast<TensorDimConst const>(expr)->value;
      entry["type"] = "const";
    } else {
      entry["type"] = "var";
    }
    map_entries.push_back(entry);
  }
  j["map_matrix"] = map_entries;
  
  return j;
}

} // namespace search
} // namespace yirage
