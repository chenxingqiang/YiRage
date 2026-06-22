#pragma once

#include "kernel/graph.h"
#include "search/abstract_expr/abstract_expr.h"
#include "search/symbolic_graph/symbolic_graph.h"
#include "threadblock/graph.h"

namespace yirage {
namespace search {

std::shared_ptr<AbstractExpr const> get_abstract_expr(
    type::KNOperatorType op,
    std::vector<kernel::DTensor> const &tensors,
    std::vector<std::shared_ptr<AbstractExpr const>> const &opds);
std::shared_ptr<AbstractExpr const> get_abstract_expr(
    type::TBOperatorType op,
    std::vector<threadblock::STensor> const &tensors,
    std::vector<std::shared_ptr<AbstractExpr const>> const &opds);

std::shared_ptr<AbstractExpr const> get_abstract_expr(
    type::KNOperatorType op,
    std::vector<SymbolicDTensor> const &tensors,
    std::vector<std::shared_ptr<AbstractExpr const>> const &opds,
    SymbolicKNGraph const &g);

std::shared_ptr<AbstractExpr const> get_abstract_expr(
    type::TBOperatorType op,
    std::vector<SymbolicSTensor> const &tensors,
    std::vector<std::shared_ptr<AbstractExpr const>> const &opds,
    SymbolicTBGraph const &g);

} // namespace search
} // namespace yirage