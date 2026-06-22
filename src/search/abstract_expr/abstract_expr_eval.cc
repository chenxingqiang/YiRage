#include "search/abstract_expr/abstract_expr_eval.h"
#include "kernel/chunk.h"
#include "kernel/concat.h"
#include "kernel/rms_norm.h"
#include "kernel/split.h"
#include "search/op_utils.h"
#include "search/symbolic_graph/op_args.h"
#include "threadblock/concat.h"
#include "threadblock/split.h"
#include "threadblock/chunk.h"

namespace yirage {
namespace search {

namespace {

bool try_register_tb_concat_then_matmul(
    threadblock::Graph const &g,
    size_t i,
    std::unordered_map<type::GuidType, std::shared_ptr<AbstractExpr const>>
        &exprs) {
  if (i + 2 >= g.operators.size()) {
    return false;
  }
  threadblock::TBOperator *op0 = g.operators[i];
  threadblock::TBOperator *op1 = g.operators[i + 1];
  threadblock::TBOperator *matmul = g.operators[i + 2];
  if (matmul->op_type != type::TBOperatorType::TB_MATMUL_OP) {
    return false;
  }
  if (op0->op_type < type::TBOperatorType::TB_CONCAT_0_OP ||
      op0->op_type > type::TBOperatorType::TB_CONCAT_2_OP ||
      op1->op_type < type::TBOperatorType::TB_CONCAT_0_OP ||
      op1->op_type > type::TBOperatorType::TB_CONCAT_2_OP) {
    return false;
  }
  auto *concat0 = static_cast<threadblock::TBConcatOp *>(op0);
  auto *concat1 = static_cast<threadblock::TBConcatOp *>(op1);
  int num_dims = concat0->input_tensors[0].num_dims;
  if (num_dims < 2 || concat1->input_tensors[0].num_dims != num_dims) {
    return false;
  }
  int hi_dim = num_dims - 1;
  int lo_dim = num_dims - 2;
  threadblock::TBConcatOp *row_concat = nullptr;
  threadblock::TBConcatOp *col_concat = nullptr;
  if (concat0->concat_dim == hi_dim && concat1->concat_dim == lo_dim) {
    row_concat = concat0;
    col_concat = concat1;
  } else if (concat0->concat_dim == lo_dim && concat1->concat_dim == hi_dim) {
    row_concat = concat1;
    col_concat = concat0;
  } else {
    return false;
  }
  type::GuidType row_out = row_concat->output_tensors[0].guid;
  type::GuidType col_out = col_concat->output_tensors[0].guid;
  if (!((matmul->input_tensors[0].guid == row_out &&
         matmul->input_tensors[1].guid == col_out) ||
        (matmul->input_tensors[0].guid == col_out &&
         matmul->input_tensors[1].guid == row_out))) {
    return false;
  }

  std::vector<threadblock::STensor> input_tensors;
  input_tensors.push_back(row_concat->input_tensors[0]);
  input_tensors.push_back(row_concat->input_tensors[1]);
  input_tensors.push_back(col_concat->input_tensors[0]);
  input_tensors.push_back(col_concat->input_tensors[1]);
  std::vector<std::shared_ptr<AbstractExpr const>> input_exprs;
  for (auto const &input_tensor : input_tensors) {
    if (!contains_key(exprs, input_tensor.guid)) {
      return false;
    }
    input_exprs.push_back(exprs.at(input_tensor.guid));
  }
  exprs.insert({row_concat->output_tensors[0].guid, nullptr});
  exprs.insert({col_concat->output_tensors[0].guid, nullptr});
  exprs.insert(
      {matmul->output_tensors[0].guid,
       get_abstract_expr(type::TBOperatorType::TB_CONCAT_THEN_MATMUL_OP,
                         input_tensors,
                         input_exprs)});
  return true;
}

kernel::KNOperator *find_kn_producer(kernel::Graph const &g, DTensor const &t) {
  for (auto *op : g.operators) {
    for (auto const &out : op->output_tensors) {
      if (out.guid == t.guid) {
        return op;
      }
    }
  }
  return nullptr;
}

bool try_register_kn_concat_matmul_fused_expr(
    kernel::Graph const &g,
    kernel::KNOperator *matmul_op,
    std::unordered_map<type::GuidType, std::shared_ptr<AbstractExpr const>>
        &exprs) {
  if (matmul_op->op_type != type::KNOperatorType::KN_MATMUL_OP) {
    return false;
  }
  kernel::KNOperator *p0 = find_kn_producer(g, matmul_op->input_tensors[0]);
  kernel::KNOperator *p1 = find_kn_producer(g, matmul_op->input_tensors[1]);
  if (p0 == nullptr || p1 == nullptr) {
    return false;
  }
  if (p0->op_type < type::KNOperatorType::KN_CONCAT_0_OP ||
      p0->op_type > type::KNOperatorType::KN_CONCAT_2_OP ||
      p1->op_type < type::KNOperatorType::KN_CONCAT_0_OP ||
      p1->op_type > type::KNOperatorType::KN_CONCAT_2_OP) {
    return false;
  }
  auto *concat0 = static_cast<kernel::KNConcatOp *>(p0);
  auto *concat1 = static_cast<kernel::KNConcatOp *>(p1);
  int num_dims = concat0->input_tensors[0].num_dims;
  if (num_dims < 2 || concat1->input_tensors[0].num_dims != num_dims) {
    return false;
  }
  int hi_dim = num_dims - 1;
  int lo_dim = num_dims - 2;
  kernel::KNConcatOp *row_concat = nullptr;
  kernel::KNConcatOp *col_concat = nullptr;
  if (concat0->concat_dim == hi_dim && concat1->concat_dim == lo_dim) {
    row_concat = concat0;
    col_concat = concat1;
  } else if (concat0->concat_dim == lo_dim && concat1->concat_dim == hi_dim) {
    row_concat = concat1;
    col_concat = concat0;
  } else {
    return false;
  }

  std::vector<DTensor> input_tensors;
  input_tensors.push_back(row_concat->input_tensors[0]);
  input_tensors.push_back(row_concat->input_tensors[1]);
  input_tensors.push_back(col_concat->input_tensors[0]);
  input_tensors.push_back(col_concat->input_tensors[1]);
  std::vector<std::shared_ptr<AbstractExpr const>> input_exprs;
  for (auto const &input_tensor : input_tensors) {
    if (!contains_key(exprs, input_tensor.guid)) {
      return false;
    }
    input_exprs.push_back(exprs.at(input_tensor.guid));
  }
  exprs.insert(
      {matmul_op->output_tensors[0].guid,
       get_abstract_expr(type::KNOperatorType::KN_CONCAT_THEN_MATMUL_OP,
                         input_tensors,
                         input_exprs)});
  return true;
}

} // namespace

void abstract_expr_eval(
    threadblock::Graph const &g,
    std::unordered_map<type::GuidType, std::shared_ptr<AbstractExpr const>>
        &exprs) {
  for (size_t i = 0; i < g.operators.size(); ++i) {
    auto const &op = g.operators[i];
    if (op->output_tensors.size() > 0) {
      bool all_outputs_present = true;
      for (auto const &output_tensor : op->output_tensors) {
        if (!contains_key(exprs, output_tensor.guid)) {
          all_outputs_present = false;
          break;
        }
      }
      if (all_outputs_present) {
        continue;
      }
    }
    if (op->op_type == type::TBOperatorType::TB_INPUT_OP) {
      exprs.insert(
          {op->output_tensors[0].guid,
           exprs.at(static_cast<threadblock::TBInputOp *>(op)->dtensor.guid)});
    } else if (op->op_type == type::TBOperatorType::TB_OUTPUT_OP) {
      exprs.insert({static_cast<threadblock::TBOutputOp *>(op)->dtensor.guid,
                    exprs.at(op->input_tensors[0].guid)});
    } else if (try_register_tb_concat_then_matmul(g, i, exprs)) {
      // Fused blocked GEMM (concat along last two dims + matmul); skip triple.
    } else {
      std::vector<std::shared_ptr<AbstractExpr const>> input_exprs;
      for (auto const &input_tensor : op->input_tensors) {
        input_exprs.push_back(exprs.at(input_tensor.guid));
      }
      std::shared_ptr<AbstractExpr const> output_expr =
          get_abstract_expr(op->op_type, op->input_tensors, input_exprs);
      if (op->op_type >= type::TBOperatorType::TB_REDUCTION_0_MAX_OP &&
          op->op_type <= type::TBOperatorType::TB_REDUCTION_2_MAX_OP) {
        exprs.insert({op->output_tensors[0].guid, output_expr});
        exprs.insert({op->output_tensors[1].guid, output_expr});
      } else if (op->op_type >= type::TBOperatorType::TB_SPLIT_0_OP &&
                 op->op_type <= type::TBOperatorType::TB_SPLIT_2_OP) {
        threadblock::TBSplitOp const *split_op =
            static_cast<threadblock::TBSplitOp const *>(op);
        int dim = (int)op->op_type - (int)type::TBOperatorType::TB_SPLIT_0_OP;
        std::shared_ptr<AbstractExpr const> input_expr = input_exprs[0];
        exprs.insert({op->output_tensors[0].guid,
                      abstract_expr_make_split(dim,
                                               split_op->split_size,
                                               0,
                                               input_expr)});
        exprs.insert({op->output_tensors[1].guid,
                      abstract_expr_make_split(dim,
                                               split_op->split_size,
                                               1,
                                               input_expr)});
      } else if (op->op_type >= type::TBOperatorType::TB_CHUNK_0_OP &&
                 op->op_type <= type::TBOperatorType::TB_CHUNK_2_OP) {
        threadblock::TBChunkOp const *chunk_op =
            static_cast<threadblock::TBChunkOp const *>(op);
        int dim = (int)op->op_type - (int)type::TBOperatorType::TB_CHUNK_0_OP;
        std::shared_ptr<AbstractExpr const> input_expr = input_exprs[0];
        int part_width =
            op->input_tensors[0].dim[dim] / chunk_op->chunk_size;
        for (size_t i = 0; i < op->output_tensors.size(); ++i) {
          exprs.insert({op->output_tensors[i].guid,
                        abstract_expr_make_split(dim, part_width, (int)i,
                                                 input_expr)});
        }
      } else {
        exprs.insert({op->output_tensors[0].guid, output_expr});
      }
    }
  }
}

void abstract_expr_eval(
    kernel::Graph const &g,
    std::unordered_map<type::GuidType, std::shared_ptr<AbstractExpr const>>
        &exprs) {
  int input_id = 0;
  for (auto const &op : g.operators) {
    if (op->op_type == type::KNOperatorType::KN_OUTPUT_OP) {
      continue;
    } else if (op->op_type == type::KNOperatorType::KN_INPUT_OP) {
      exprs.insert({op->output_tensors[0].guid,
                    std::make_shared<Var>("v_" + std::to_string(input_id))});
      input_id++;
    } else if (op->op_type == type::KNOperatorType::KN_RMS_NORM_OP) {
      std::shared_ptr<AbstractExpr const> input_expr =
          exprs.at(op->input_tensors[0].guid);
      std::shared_ptr<AbstractExpr const> denominator_expr =
          abstract_expr_make_rms(
              static_cast<kernel::KNRMSNormOp *>(op)->normalized_size,
              input_expr);
      std::shared_ptr<AbstractExpr const> output_expr =
          abstract_expr_make_div(input_expr, denominator_expr);
      exprs.insert({op->output_tensors[0].guid, output_expr});
    } else if (op->op_type != type::KNOperatorType::KN_CUSTOMIZED_OP) {
      std::vector<std::shared_ptr<AbstractExpr const>> input_exprs;
      for (auto const &input_tensor : op->input_tensors) {
        assert(contains_key(exprs, input_tensor.guid));
        input_exprs.push_back(exprs.at(input_tensor.guid));
      }
      if (op->op_type >= type::KNOperatorType::KN_SPLIT_0_OP &&
          op->op_type <= type::KNOperatorType::KN_SPLIT_2_OP) {
        kernel::KNSplitOp const *split_op =
            static_cast<kernel::KNSplitOp const *>(op);
        int dim = (int)op->op_type - (int)type::KNOperatorType::KN_SPLIT_0_OP;
        std::shared_ptr<AbstractExpr const> input_expr = input_exprs[0];
        exprs.insert({op->output_tensors[0].guid,
                      abstract_expr_make_split(dim,
                                               split_op->split_size,
                                               0,
                                               input_expr)});
        exprs.insert({op->output_tensors[1].guid,
                      abstract_expr_make_split(dim,
                                               split_op->split_size,
                                               1,
                                               input_expr)});
      } else if (op->op_type >= type::KNOperatorType::KN_CHUNK_0_OP &&
                 op->op_type <= type::KNOperatorType::KN_CHUNK_2_OP) {
        kernel::KNChunkOp const *chunk_op =
            static_cast<kernel::KNChunkOp const *>(op);
        int dim = (int)op->op_type - (int)type::KNOperatorType::KN_CHUNK_0_OP;
        int part_width =
            op->input_tensors[0].dim[dim] / chunk_op->chunk_size;
        std::shared_ptr<AbstractExpr const> input_expr = input_exprs[0];
        for (size_t i = 0; i < op->output_tensors.size(); ++i) {
          exprs.insert({op->output_tensors[i].guid,
                        abstract_expr_make_split(dim,
                                                 part_width,
                                                 (int)i,
                                                 input_expr)});
        }
      } else if (op->op_type == type::KNOperatorType::KN_MATMUL_OP &&
                 try_register_kn_concat_matmul_fused_expr(g, op, exprs)) {
        // LoRA blocked GEMM: match TB/KN concat_then_matmul search candidates.
      } else {
        exprs.insert(
            {op->output_tensors[0].guid,
             get_abstract_expr(op->op_type, op->input_tensors, input_exprs)});
      }
    } else {
      assert(op->op_type == type::KNOperatorType::KN_CUSTOMIZED_OP);
      abstract_expr_eval(static_cast<kernel::KNCustomizedOp *>(op)->bgraph,
                         exprs);
    }
  }
}

void abstract_expr_eval(
    SymbolicKNGraph const &kn_graph,
    std::vector<std::shared_ptr<AbstractExpr const>> &exprs) {
  int input_id = 0;
  for (size_t i = 0; i < kn_graph.operators.size(); ++i) {
    if (kn_graph.operators[i].op_type == type::KN_OUTPUT_OP) {
      // Skip the output operator
      continue;
    } else if (kn_graph.operators[i].op_type == type::KN_INPUT_OP) {
      // Create a new variable for each input operator
      exprs.push_back(std::make_shared<Var>("v_" + std::to_string(input_id)));
      input_id++;
    } else if (kn_graph.operators[i].op_type != type::KN_CUSTOMIZED_OP) {
      // Evaluate the expression for pre-defined operators
      std::vector<SymbolicDTensor> input_tensors =
          vector_map(kn_graph.input_indices[i],
                     [&](int i) { return kn_graph.tensors[i]; });
      std::vector<std::shared_ptr<AbstractExpr const>> input_exprs = vector_map(
          kn_graph.input_indices[i], [&](int i) { return exprs[i]; });
      type::KNOperatorType op_type = kn_graph.operators[i].op_type;
      if (op_type >= type::KNOperatorType::KN_SPLIT_0_OP &&
          op_type <= type::KNOperatorType::KN_SPLIT_2_OP) {
        int dim = (int)op_type - (int)type::KNOperatorType::KN_SPLIT_0_OP;
        std::shared_ptr<KNSplitOpArgs const> args =
            std::static_pointer_cast<KNSplitOpArgs const>(
                kn_graph.operators[i].args);
        std::shared_ptr<AbstractExpr const> input_expr = input_exprs[0];
        exprs.push_back(abstract_expr_make_split(
            dim, args->split_size, 0, input_expr));
        exprs.push_back(abstract_expr_make_split(
            dim, args->split_size, 1, input_expr));
      } else if (op_type >= type::KNOperatorType::KN_CHUNK_0_OP &&
                 op_type <= type::KNOperatorType::KN_CHUNK_2_OP) {
        int dim = (int)op_type - (int)type::KNOperatorType::KN_CHUNK_0_OP;
        std::shared_ptr<KNChunkOpArgs const> args =
            std::static_pointer_cast<KNChunkOpArgs const>(
                kn_graph.operators[i].args);
        DimVarAssignments empty;
        int axis = input_tensors[0].dims[dim].dim_expr->get_value(empty);
        int part_width = axis / args->num_chunks;
        std::shared_ptr<AbstractExpr const> input_expr = input_exprs[0];
        for (int part = 0; part < args->num_chunks; ++part) {
          exprs.push_back(abstract_expr_make_split(
              dim, part_width, part, input_expr));
        }
      } else {
        exprs.push_back(get_abstract_expr(
            op_type, input_tensors, input_exprs, kn_graph));
      }
    } else {
      // Evaluate the expression for customized operators
      assert(kn_graph.operators[i].op_type == type::KN_CUSTOMIZED_OP);
      std::vector<std::shared_ptr<AbstractExpr const>> input_exprs = vector_map(
          kn_graph.input_indices[i], [&](int i) { return exprs[i]; });
      std::vector<std::shared_ptr<AbstractExpr const>> tb_graph_exprs,
          output_exprs;
      SymbolicTBGraph const &tb_graph =
          std::static_pointer_cast<KNCustomizedOpArgs const>(
              kn_graph.operators[i].args)
              ->tb_graph_template;
      abstract_expr_eval(tb_graph, input_exprs, tb_graph_exprs, output_exprs);
      exprs.insert(exprs.end(), output_exprs.begin(), output_exprs.end());
    }
  }
}

void abstract_expr_eval(
    SymbolicTBGraph const &tb_graph,
    std::vector<std::shared_ptr<AbstractExpr const>> const &input_exprs,
    std::vector<std::shared_ptr<AbstractExpr const>> &exprs,
    std::vector<std::shared_ptr<AbstractExpr const>> &output_exprs) {
  for (size_t i = 0; i < tb_graph.operators.size(); ++i) {
    if (tb_graph.operators[i].op_type == type::TBOperatorType::TB_INPUT_OP) {
      exprs.push_back(input_exprs[i]);
    } else if (tb_graph.operators[i].op_type ==
               type::TBOperatorType::TB_OUTPUT_OP) {
      output_exprs.push_back(exprs[tb_graph.input_indices[i][0]]);
    } else {
      std::vector<SymbolicSTensor> input_tensors =
          vector_map(tb_graph.input_indices[i],
                     [&](int i) { return tb_graph.tensors[i]; });
      std::vector<std::shared_ptr<AbstractExpr const>> input_exprs = vector_map(
          tb_graph.input_indices[i], [&](int i) { return exprs[i]; });
      type::TBOperatorType op_type = tb_graph.operators[i].op_type;
      std::shared_ptr<AbstractExpr const> expr = get_abstract_expr(
          op_type, input_tensors, input_exprs, tb_graph);
      if (op_type >= type::TBOperatorType::TB_REDUCTION_0_MAX_OP &&
          op_type <= type::TBOperatorType::TB_REDUCTION_2_MAX_OP) {
        exprs.push_back(expr);
        exprs.push_back(expr);
      } else if (op_type >= type::TBOperatorType::TB_SPLIT_0_OP &&
                 op_type <= type::TBOperatorType::TB_SPLIT_2_OP) {
        int dim = (int)op_type - (int)type::TBOperatorType::TB_SPLIT_0_OP;
        std::shared_ptr<TBSplitOpArgs const> args =
            std::static_pointer_cast<TBSplitOpArgs const>(
                tb_graph.operators[i].args);
        std::shared_ptr<AbstractExpr const> input_expr = input_exprs[0];
        exprs.push_back(abstract_expr_make_split(
            dim, args->split_size, 0, input_expr));
        exprs.push_back(abstract_expr_make_split(
            dim, args->split_size, 1, input_expr));
      } else if (op_type >= type::TBOperatorType::TB_CHUNK_0_OP &&
                 op_type <= type::TBOperatorType::TB_CHUNK_2_OP) {
        int dim = (int)op_type - (int)type::TBOperatorType::TB_CHUNK_0_OP;
        std::shared_ptr<TBChunkOpArgs const> args =
            std::static_pointer_cast<TBChunkOpArgs const>(
                tb_graph.operators[i].args);
        std::shared_ptr<AbstractExpr const> input_expr = input_exprs[0];
        DimVarAssignments empty;
        int axis = input_tensors[0].dims[dim].dim_expr->get_value(empty);
        int part_width = axis / args->num_chunks;
        for (int part = 0; part < args->num_chunks; ++part) {
          exprs.push_back(abstract_expr_make_split(
              dim, part_width, part, input_expr));
        }
      } else {
        exprs.push_back(expr);
      }
    }
  }
}

} // namespace search
} // namespace yirage
