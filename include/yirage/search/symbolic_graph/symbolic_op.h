#pragma once

#include "yirage/search/symbolic_graph/symbolic_tensor.h"
#include "yirage/type.h"

namespace yirage {
namespace search {

class OpArgs;

class SymbolicKNOp {
public:
  SymbolicKNOp(type::KNOperatorType op_type,
               std::shared_ptr<OpArgs const> args);
  SymbolicKNOp(type::KNOperatorType op_type);

  type::KNOperatorType op_type;
  std::shared_ptr<OpArgs const> args;

  operator json() const;
};

class SymbolicTBOp {
public:
  SymbolicTBOp(type::TBOperatorType op_type,
               std::shared_ptr<OpArgs const> args);
  SymbolicTBOp(type::TBOperatorType op_type);

  type::TBOperatorType op_type;
  std::shared_ptr<OpArgs const> args;

  operator json() const;
};

} // namespace search
} // namespace yirage
