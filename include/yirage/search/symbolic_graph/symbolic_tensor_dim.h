#pragma once

#include "yirage/search/symbolic_graph/tensor_dim_expr.h"

namespace yirage {
namespace search {

class SymbolicTensorDim {
public:
  SymbolicTensorDim(std::shared_ptr<TensorDimExpr const> dim_expr);

  std::shared_ptr<TensorDimExpr const> dim_expr;

  operator json() const;
  bool operator==(SymbolicTensorDim const &other) const;

  SymbolicTensorDim operator+(SymbolicTensorDim const &other) const;
  SymbolicTensorDim operator*(SymbolicTensorDim const &other) const;
  SymbolicTensorDim operator/(SymbolicTensorDim const &other) const;
  SymbolicTensorDim operator^(SymbolicTensorDim const &other) const;
};

} // namespace search
} // namespace yirage

namespace std {

template <>
struct hash<yirage::search::SymbolicTensorDim> {
  size_t operator()(yirage::search::SymbolicTensorDim const &dim) const;
};

} // namespace std
