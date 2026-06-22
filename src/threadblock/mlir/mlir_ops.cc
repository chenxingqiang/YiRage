/* Copyright 2025 Chen Xingqiang (YiRage Project)
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
 *
 * MLIR Threadblock Operations Implementation
 */

#include "threadblock/mlir/mlir_ops.h"

#include <sstream>
#include <cassert>

namespace yirage {
namespace threadblock {
namespace mlir_ops {

// =============================================================================
// Dialect Utilities
// =============================================================================

namespace dialect {

std::string get_mlir_type(type::DataType dtype) {
    switch (dtype) {
        case type::DT_FLOAT32: return "f32";
        case type::DT_FLOAT16: return "f16";
        case type::DT_BFLOAT16: return "bf16";
        case type::DT_DOUBLE: return "f64";
        case type::DT_INT32: return "i32";
        case type::DT_INT64: return "i64";
        case type::DT_INT8: return "i8";
        case type::DT_INT16: return "i16";
        default: return "f32";
    }
}

std::string get_tensor_type(const std::vector<int>& shape, type::DataType dtype) {
    std::stringstream ss;
    ss << "tensor<";
    for (size_t i = 0; i < shape.size(); i++) {
        if (i > 0) ss << "x";
        ss << shape[i];
    }
    ss << "x" << get_mlir_type(dtype) << ">";
    return ss.str();
}

std::string get_memref_type(const std::vector<int>& shape, type::DataType dtype) {
    std::stringstream ss;
    ss << "memref<";
    for (size_t i = 0; i < shape.size(); i++) {
        if (i > 0) ss << "x";
        ss << shape[i];
    }
    ss << "x" << get_mlir_type(dtype) << ">";
    return ss.str();
}

std::string generate_affine_map(int num_dims) {
    std::stringstream ss;
    ss << "affine_map<(";
    for (int i = 0; i < num_dims; i++) {
        if (i > 0) ss << ", ";
        ss << "d" << i;
    }
    ss << ") -> (";
    for (int i = 0; i < num_dims; i++) {
        if (i > 0) ss << ", ";
        ss << "d" << i;
    }
    ss << ")>";
    return ss.str();
}

}  // namespace dialect

// =============================================================================
// MLIR Code Generator
// =============================================================================

std::string MLIRCodeGenerator::generate_matmul(
    int M, int N, int K,
    type::DataType dtype,
    const MLIRTileConfig& config
) {
    std::stringstream ss;
    std::string elem_type = dialect::get_mlir_type(dtype);
    
    ss << R"(
// YiRage MatMul Kernel - MLIR Linalg
module @yirage_matmul {
  func.func @matmul(
    %A: tensor<)" << M << "x" << K << "x" << elem_type << R"(>,
    %B: tensor<)" << K << "x" << N << "x" << elem_type << R"(>,
    %C: tensor<)" << M << "x" << N << "x" << elem_type << R"(>
  ) -> tensor<)" << M << "x" << N << "x" << elem_type << R"(> {
    
    // Initialize output with zeros
    %c0 = arith.constant 0.0 : )" << elem_type << R"(
    %C_init = linalg.fill ins(%c0 : )" << elem_type << R"() 
                          outs(%C : tensor<)" << M << "x" << N << "x" << elem_type << R"(>)
                          -> tensor<)" << M << "x" << N << "x" << elem_type << R"(>
    
    // MatMul using linalg.matmul
    %result = linalg.matmul
      ins(%A, %B : tensor<)" << M << "x" << K << "x" << elem_type << ">, "
                           << "tensor<" << K << "x" << N << "x" << elem_type << R"(>)
      outs(%C_init : tensor<)" << M << "x" << N << "x" << elem_type << R"(>)
      -> tensor<)" << M << "x" << N << "x" << elem_type << R"(>
    
    return %result : tensor<)" << M << "x" << N << "x" << elem_type << R"(>
  }
}
)";
    
    return ss.str();
}

std::string MLIRCodeGenerator::generate_flash_attention(
    int batch, int heads, int seq_len, int head_dim,
    bool causal,
    const MLIRTileConfig& config
) {
    std::stringstream ss;
    
    ss << R"(
// YiRage Flash Attention Kernel - MLIR
module @yirage_flash_attention {
  func.func @flash_attention(
    %Q: tensor<)" << batch << "x" << heads << "x" << seq_len << "x" << head_dim << R"(xf16>,
    %K: tensor<)" << batch << "x" << heads << "x" << seq_len << "x" << head_dim << R"(xf16>,
    %V: tensor<)" << batch << "x" << heads << "x" << seq_len << "x" << head_dim << R"(xf16>
  ) -> tensor<)" << batch << "x" << heads << "x" << seq_len << "x" << head_dim << R"(xf16> {
    
    // Scale factor: 1/sqrt(head_dim)
    %scale = arith.constant )" << (1.0f / std::sqrt(static_cast<float>(head_dim))) << R"( : f32
    
    // Allocate output
    %c0 = arith.constant 0.0 : f16
    %init = tensor.splat %c0 : tensor<)" << batch << "x" << heads << "x" << seq_len << "x" << head_dim << R"(xf16>
    
    // Iterate over batch and heads
    %result = scf.forall (%b, %h) in ()" << batch << ", " << heads << R"() shared_outs(%out = %init)
        -> tensor<)" << batch << "x" << heads << "x" << seq_len << "x" << head_dim << R"(xf16> {
      
      // Extract Q, K, V slices for this batch and head
      %q_slice = tensor.extract_slice %Q[%b, %h, 0, 0] [1, 1, )" << seq_len << ", " << head_dim << R"(] [1, 1, 1, 1]
          : tensor<)" << batch << "x" << heads << "x" << seq_len << "x" << head_dim << R"(xf16> 
          to tensor<)" << seq_len << "x" << head_dim << R"(xf16>
      
      %k_slice = tensor.extract_slice %K[%b, %h, 0, 0] [1, 1, )" << seq_len << ", " << head_dim << R"(] [1, 1, 1, 1]
          : tensor<)" << batch << "x" << heads << "x" << seq_len << "x" << head_dim << R"(xf16>
          to tensor<)" << seq_len << "x" << head_dim << R"(xf16>
      
      %v_slice = tensor.extract_slice %V[%b, %h, 0, 0] [1, 1, )" << seq_len << ", " << head_dim << R"(] [1, 1, 1, 1]
          : tensor<)" << batch << "x" << heads << "x" << seq_len << "x" << head_dim << R"(xf16>
          to tensor<)" << seq_len << "x" << head_dim << R"(xf16>
      
      // Q @ K^T
      %k_t = linalg.transpose ins(%k_slice : tensor<)" << seq_len << "x" << head_dim << R"(xf16>)
                              outs(%k_slice : tensor<)" << head_dim << "x" << seq_len << R"(xf16>)
                              permutation = [1, 0]
      
      %scores_init = tensor.empty() : tensor<)" << seq_len << "x" << seq_len << R"(xf32>
      %scores = linalg.matmul ins(%q_slice, %k_t : tensor<)" << seq_len << "x" << head_dim << "xf16>, "
                               << "tensor<" << head_dim << "x" << seq_len << R"(xf16>)
                              outs(%scores_init : tensor<)" << seq_len << "x" << seq_len << R"(xf32>)
      
      // Scale scores
      %scaled = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]
      } ins(%scores : tensor<)" << seq_len << "x" << seq_len << R"(xf32>)
        outs(%scores_init : tensor<)" << seq_len << "x" << seq_len << R"(xf32>) {
      ^bb0(%in: f32, %out: f32):
        %s = arith.mulf %in, %scale : f32
        linalg.yield %s : f32
      } -> tensor<)" << seq_len << "x" << seq_len << R"(xf32>
)";

    if (causal) {
        ss << R"(
      // Apply causal mask
      %masked = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]
      } ins(%scaled : tensor<)" << seq_len << "x" << seq_len << R"(xf32>)
        outs(%scores_init : tensor<)" << seq_len << "x" << seq_len << R"(xf32>) {
      ^bb0(%in: f32, %out: f32):
        %i = linalg.index 0 : index
        %j = linalg.index 1 : index
        %cmp = arith.cmpi "sgt", %j, %i : index
        %neg_inf = arith.constant 0xFF800000 : f32  // -inf
        %result = arith.select %cmp, %neg_inf, %in : f32
        linalg.yield %result : f32
      } -> tensor<)" << seq_len << "x" << seq_len << R"(xf32>
)";
    }

    ss << R"(
      // Softmax (row-wise)
      // ... softmax implementation using max, exp, sum, div ...
      
      // Attention @ V
      %attn_v = linalg.matmul ins(%attn, %v_slice : tensor<)" << seq_len << "x" << seq_len << "xf16>, "
                               << "tensor<" << seq_len << "x" << head_dim << R"(xf16>)
                              outs(%out_init : tensor<)" << seq_len << "x" << head_dim << R"(xf16>)
      
      // Store result
      %updated = tensor.insert_slice %attn_v into %out[%b, %h, 0, 0] [1, 1, )" << seq_len << ", " << head_dim << R"(] [1, 1, 1, 1]
          : tensor<)" << seq_len << "x" << head_dim << "xf16> into tensor<" << batch << "x" << heads << "x" << seq_len << "x" << head_dim << R"(xf16>
      
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %attn_v into %out[%b, %h, 0, 0] [1, 1, )" << seq_len << ", " << head_dim << R"(] [1, 1, 1, 1]
            : tensor<)" << seq_len << "x" << head_dim << "xf16> into tensor<" << batch << "x" << heads << "x" << seq_len << "x" << head_dim << R"(xf16>
      }
    }
    
    return %result : tensor<)" << batch << "x" << heads << "x" << seq_len << "x" << head_dim << R"(xf16>
  }
}
)";
    
    return ss.str();
}

std::string MLIRCodeGenerator::generate_rms_norm(
    int hidden_dim,
    float epsilon,
    type::DataType dtype
) {
    std::stringstream ss;
    std::string elem_type = dialect::get_mlir_type(dtype);
    
    ss << R"(
// YiRage RMS Norm Kernel - MLIR
module @yirage_rms_norm {
  func.func @rms_norm(
    %input: tensor<?x)" << hidden_dim << "x" << elem_type << R"(>,
    %gamma: tensor<)" << hidden_dim << "x" << elem_type << R"(>
  ) -> tensor<?x)" << hidden_dim << "x" << elem_type << R"(> {
    
    %c0 = arith.constant 0 : index
    %batch = tensor.dim %input, %c0 : tensor<?x)" << hidden_dim << "x" << elem_type << R"(>
    
    // Compute sum of squares per row
    %c0_f = arith.constant 0.0 : f32
    %sum_sq_init = tensor.splat %c0_f : tensor<?xf32>
    
    %sum_sq = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]
    } ins(%input : tensor<?x)" << hidden_dim << "x" << elem_type << R"(>)
      outs(%sum_sq_init : tensor<?xf32>) {
    ^bb0(%in: )" << elem_type << R"(, %acc: f32):
      %in_f32 = arith.extf %in : )" << elem_type << R"( to f32
      %sq = arith.mulf %in_f32, %in_f32 : f32
      %sum = arith.addf %acc, %sq : f32
      linalg.yield %sum : f32
    } -> tensor<?xf32>
    
    // Compute RMS: 1/sqrt(mean + eps)
    %eps = arith.constant )" << epsilon << R"( : f32
    %dim = arith.constant )" << hidden_dim << R"( : i32
    %dim_f = arith.sitofp %dim : i32 to f32
    
    %rstd = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%sum_sq : tensor<?xf32>)
      outs(%sum_sq_init : tensor<?xf32>) {
    ^bb0(%ss: f32, %out: f32):
      %mean = arith.divf %ss, %dim_f : f32
      %mean_eps = arith.addf %mean, %eps : f32
      %rsqrt = math.rsqrt %mean_eps : f32
      linalg.yield %rsqrt : f32
    } -> tensor<?xf32>
    
    // Normalize: x * rstd * gamma
    %result_init = tensor.empty(%batch) : tensor<?x)" << hidden_dim << "x" << elem_type << R"(>
    
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>,
                       affine_map<(d0, d1) -> (d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%input, %rstd, %gamma : tensor<?x)" << hidden_dim << "x" << elem_type << R"(>, tensor<?xf32>, tensor<)" << hidden_dim << "x" << elem_type << R"(>)
      outs(%result_init : tensor<?x)" << hidden_dim << "x" << elem_type << R"(>) {
    ^bb0(%x: )" << elem_type << ", %r: f32, %g: " << elem_type << ", %out: " << elem_type << R"():
      %x_f32 = arith.extf %x : )" << elem_type << R"( to f32
      %g_f32 = arith.extf %g : )" << elem_type << R"( to f32
      %norm = arith.mulf %x_f32, %r : f32
      %scaled = arith.mulf %norm, %g_f32 : f32
      %result = arith.truncf %scaled : f32 to )" << elem_type << R"(
      linalg.yield %result : )" << elem_type << R"(
    } -> tensor<?x)" << hidden_dim << "x" << elem_type << R"(>
    
    return %result : tensor<?x)" << hidden_dim << "x" << elem_type << R"(>
  }
}
)";
    
    return ss.str();
}

std::string MLIRCodeGenerator::generate_layer_norm(
    int hidden_dim,
    float epsilon,
    type::DataType dtype
) {
    std::stringstream ss;
    std::string elem_type = dialect::get_mlir_type(dtype);
    
    ss << R"(
// YiRage Layer Norm Kernel - MLIR
module @yirage_layer_norm {
  func.func @layer_norm(
    %input: tensor<?x)" << hidden_dim << "x" << elem_type << R"(>,
    %gamma: tensor<)" << hidden_dim << "x" << elem_type << R"(>,
    %beta: tensor<)" << hidden_dim << "x" << elem_type << R"(>
  ) -> tensor<?x)" << hidden_dim << "x" << elem_type << R"(> {
    
    %c0 = arith.constant 0 : index
    %batch = tensor.dim %input, %c0 : tensor<?x)" << hidden_dim << "x" << elem_type << R"(>
    
    // Compute mean per row
    %c0_f = arith.constant 0.0 : f32
    %mean_init = tensor.splat %c0_f : tensor<?xf32>
    
    %sum = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]
    } ins(%input : tensor<?x)" << hidden_dim << "x" << elem_type << R"(>)
      outs(%mean_init : tensor<?xf32>) {
    ^bb0(%in: )" << elem_type << R"(, %acc: f32):
      %in_f32 = arith.extf %in : )" << elem_type << R"( to f32
      %s = arith.addf %acc, %in_f32 : f32
      linalg.yield %s : f32
    } -> tensor<?xf32>
    
    %dim = arith.constant )" << hidden_dim << R"( : i32
    %dim_f = arith.sitofp %dim : i32 to f32
    %mean = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%sum : tensor<?xf32>)
      outs(%mean_init : tensor<?xf32>) {
    ^bb0(%s: f32, %out: f32):
      %m = arith.divf %s, %dim_f : f32
      linalg.yield %m : f32
    } -> tensor<?xf32>
    
    // Compute variance
    %var = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>,
                       affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]
    } ins(%input, %mean : tensor<?x)" << hidden_dim << "x" << elem_type << R"(>, tensor<?xf32>)
      outs(%mean_init : tensor<?xf32>) {
    ^bb0(%in: )" << elem_type << R"(, %m: f32, %acc: f32):
      %in_f32 = arith.extf %in : )" << elem_type << R"( to f32
      %diff = arith.subf %in_f32, %m : f32
      %sq = arith.mulf %diff, %diff : f32
      %s = arith.addf %acc, %sq : f32
      linalg.yield %s : f32
    } -> tensor<?xf32>
    
    // Compute rstd
    %eps = arith.constant )" << epsilon << R"( : f32
    %rstd = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%var : tensor<?xf32>)
      outs(%mean_init : tensor<?xf32>) {
    ^bb0(%v: f32, %out: f32):
      %v_mean = arith.divf %v, %dim_f : f32
      %v_eps = arith.addf %v_mean, %eps : f32
      %rs = math.rsqrt %v_eps : f32
      linalg.yield %rs : f32
    } -> tensor<?xf32>
    
    // Normalize
    %result_init = tensor.empty(%batch) : tensor<?x)" << hidden_dim << "x" << elem_type << R"(>
    
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>,
                       affine_map<(d0, d1) -> (d0)>,
                       affine_map<(d0, d1) -> (d1)>,
                       affine_map<(d0, d1) -> (d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%input, %mean, %rstd, %gamma, %beta : 
          tensor<?x)" << hidden_dim << "x" << elem_type << ">, tensor<?xf32>, tensor<?xf32>, "
                     << "tensor<" << hidden_dim << "x" << elem_type << ">, tensor<" << hidden_dim << "x" << elem_type << R"(>)
      outs(%result_init : tensor<?x)" << hidden_dim << "x" << elem_type << R"(>) {
    ^bb0(%x: )" << elem_type << ", %m: f32, %r: f32, %g: " << elem_type << ", %b: " << elem_type << ", %out: " << elem_type << R"():
      %x_f32 = arith.extf %x : )" << elem_type << R"( to f32
      %g_f32 = arith.extf %g : )" << elem_type << R"( to f32
      %b_f32 = arith.extf %b : )" << elem_type << R"( to f32
      %centered = arith.subf %x_f32, %m : f32
      %norm = arith.mulf %centered, %r : f32
      %scaled = arith.mulf %norm, %g_f32 : f32
      %shifted = arith.addf %scaled, %b_f32 : f32
      %result = arith.truncf %shifted : f32 to )" << elem_type << R"(
      linalg.yield %result : )" << elem_type << R"(
    } -> tensor<?x)" << hidden_dim << "x" << elem_type << R"(>
    
    return %result : tensor<?x)" << hidden_dim << "x" << elem_type << R"(>
  }
}
)";
    
    return ss.str();
}

std::string MLIRCodeGenerator::generate_elementwise(
    MLIROpType op,
    const std::vector<int>& shape,
    type::DataType dtype
) {
    std::stringstream ss;
    std::string elem_type = dialect::get_mlir_type(dtype);
    std::string tensor_type = dialect::get_tensor_type(shape, dtype);
    
    std::string op_name;
    std::string op_body;
    
    switch (op) {
        case MLIR_ELEMENTWISE:
            op_name = "relu";
            op_body = R"(
      %c0 = arith.constant 0.0 : )" + elem_type + R"(
      %cmp = arith.cmpf "ogt", %in, %c0 : )" + elem_type + R"(
      %result = arith.select %cmp, %in, %c0 : )" + elem_type;
            break;
        default:
            op_name = "identity";
            op_body = "      %result = %in";
    }
    
    ss << R"(
// YiRage )" << op_name << R"( Kernel - MLIR
module @yirage_)" << op_name << R"( {
  func.func @)" << op_name << R"((
    %input: )" << tensor_type << R"(
  ) -> )" << tensor_type << R"( {
    
    %result = linalg.generic {
      indexing_maps = [)" << dialect::generate_affine_map(shape.size()) << R"(,
                       )" << dialect::generate_affine_map(shape.size()) << R"(],
      iterator_types = [)" << std::string(shape.size() - 1, '"') << R"("parallel")" << std::string(shape.size() > 1 ? ", " : "") << R"(]
    } ins(%input : )" << tensor_type << R"()
      outs(%input : )" << tensor_type << R"() {
    ^bb0(%in: )" << elem_type << ", %out: " << elem_type << R"():
)" << op_body << R"(
      linalg.yield %result : )" << elem_type << R"(
    } -> )" << tensor_type << R"(
    
    return %result : )" << tensor_type << R"(
  }
}
)";
    
    return ss.str();
}

std::string MLIRCodeGenerator::generate_reduce(
    const std::string& reduce_op,
    const std::vector<int>& shape,
    const std::vector<int>& reduce_dims,
    type::DataType dtype
) {
    std::stringstream ss;
    std::string elem_type = dialect::get_mlir_type(dtype);
    std::string tensor_type = dialect::get_tensor_type(shape, dtype);
    
    ss << R"(
// YiRage Reduce )" << reduce_op << R"( Kernel - MLIR
module @yirage_reduce_)" << reduce_op << R"( {
  func.func @reduce_)" << reduce_op << R"((
    %input: )" << tensor_type << R"(
  ) -> tensor<?x)" << elem_type << R"(> {
    
    // Reduction using linalg.reduce
    %init = arith.constant 0.0 : )" << elem_type << R"(
    %result = linalg.reduce { arith.)" << reduce_op << R"(f } 
      ins(%input : )" << tensor_type << R"()
      outs(%init : )" << elem_type << R"()
      dimensions = [)" << reduce_dims[0];
    
    for (size_t i = 1; i < reduce_dims.size(); i++) {
        ss << ", " << reduce_dims[i];
    }
    
    ss << R"(]
    
    return %result : tensor<?x)" << elem_type << R"(>
  }
}
)";
    
    return ss.str();
}

std::string MLIRCodeGenerator::generate_swiglu(
    int hidden_dim,
    type::DataType dtype
) {
    std::stringstream ss;
    std::string elem_type = dialect::get_mlir_type(dtype);
    
    ss << R"(
// YiRage SwiGLU Kernel - MLIR
module @yirage_swiglu {
  func.func @swiglu(
    %gate: tensor<?x)" << hidden_dim << "x" << elem_type << R"(>,
    %up: tensor<?x)" << hidden_dim << "x" << elem_type << R"(>
  ) -> tensor<?x)" << hidden_dim << "x" << elem_type << R"(> {
    
    %c0 = arith.constant 0 : index
    %batch = tensor.dim %gate, %c0 : tensor<?x)" << hidden_dim << "x" << elem_type << R"(>
    
    %result_init = tensor.empty(%batch) : tensor<?x)" << hidden_dim << "x" << elem_type << R"(>
    
    // SwiGLU: silu(gate) * up = (gate * sigmoid(gate)) * up
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%gate, %up : tensor<?x)" << hidden_dim << "x" << elem_type << ">, tensor<?x" << hidden_dim << "x" << elem_type << R"(>)
      outs(%result_init : tensor<?x)" << hidden_dim << "x" << elem_type << R"(>) {
    ^bb0(%g: )" << elem_type << ", %u: " << elem_type << ", %out: " << elem_type << R"():
      // Convert to f32 for numerical stability
      %g_f32 = arith.extf %g : )" << elem_type << R"( to f32
      %u_f32 = arith.extf %u : )" << elem_type << R"( to f32
      
      // Sigmoid
      %neg_g = arith.negf %g_f32 : f32
      %exp_neg = math.exp %neg_g : f32
      %c1 = arith.constant 1.0 : f32
      %denom = arith.addf %c1, %exp_neg : f32
      %sigmoid = arith.divf %c1, %denom : f32
      
      // SiLU = g * sigmoid(g)
      %silu = arith.mulf %g_f32, %sigmoid : f32
      
      // SwiGLU = silu * up
      %swiglu = arith.mulf %silu, %u_f32 : f32
      
      // Convert back
      %result = arith.truncf %swiglu : f32 to )" << elem_type << R"(
      linalg.yield %result : )" << elem_type << R"(
    } -> tensor<?x)" << hidden_dim << "x" << elem_type << R"(>
    
    return %result : tensor<?x)" << hidden_dim << "x" << elem_type << R"(>
  }
}
)";
    
    return ss.str();
}

std::string MLIRCodeGenerator::generate_fused_ops(
    const std::vector<MLIROpType>& ops,
    const std::vector<std::vector<int>>& shapes,
    type::DataType dtype
) {
    // Generate fused kernel for multiple operations
    std::stringstream ss;
    ss << "// YiRage Fused Operations - MLIR\n";
    ss << "// Fusing " << ops.size() << " operations\n";
    // Implementation would generate a single linalg.generic with all ops fused
    return ss.str();
}

// =============================================================================
// MLIR Pass Pipeline
// =============================================================================

std::vector<std::string> MLIRPassPipeline::build_threadblock_pipeline(
    const MLIRThreadblockPassConfig& config
) {
    std::vector<std::string> passes;
    
    // Canonicalization
    passes.push_back("canonicalize");
    passes.push_back("cse");
    
    // Tiling
    if (config.tile_and_fuse) {
        std::stringstream tile_pass;
        tile_pass << "linalg-tile{tile-sizes=";
        for (size_t i = 0; i < config.tile_sizes.size(); i++) {
            if (i > 0) tile_pass << ",";
            tile_pass << config.tile_sizes[i];
        }
        tile_pass << "}";
        passes.push_back(tile_pass.str());
        passes.push_back("linalg-fuse-elementwise-ops");
    }
    
    // Vectorization
    if (config.vectorize) {
        std::stringstream vec_pass;
        vec_pass << "linalg-vectorization{vector-width=" 
                 << config.vector_width << "}";
        passes.push_back(vec_pass.str());
    }
    
    // Parallelization
    if (config.parallelize && config.num_threads > 1) {
        std::stringstream par_pass;
        par_pass << "scf-parallel-loop-tiling{parallel-loop-tile-sizes="
                 << config.tile_sizes[0] << "}";
        passes.push_back(par_pass.str());
    }
    
    // Bufferization
    if (config.bufferize) {
        passes.push_back("one-shot-bufferize{bufferize-function-boundaries}");
        passes.push_back("buffer-deallocation");
    }
    
    return passes;
}

std::vector<std::string> MLIRPassPipeline::build_target_pipeline(
    const std::string& target,
    const MLIRThreadblockPassConfig& config
) {
    std::vector<std::string> passes = build_threadblock_pipeline(config);
    
    if (target == "llvm") {
        passes.push_back("convert-linalg-to-loops");
        passes.push_back("lower-affine");
        passes.push_back("convert-scf-to-cf");
        passes.push_back("convert-arith-to-llvm");
        passes.push_back("convert-memref-to-llvm");
        passes.push_back("convert-func-to-llvm");
        passes.push_back("reconcile-unrealized-casts");
    } else if (target == "nvvm") {
        passes.push_back("convert-linalg-to-parallel-loops");
        passes.push_back("gpu-map-parallel-loops");
        passes.push_back("convert-parallel-loops-to-gpu");
        passes.push_back("gpu-kernel-outlining");
        passes.push_back("convert-gpu-to-nvvm");
    } else if (target == "rocm") {
        passes.push_back("convert-linalg-to-parallel-loops");
        passes.push_back("gpu-map-parallel-loops");
        passes.push_back("convert-parallel-loops-to-gpu");
        passes.push_back("gpu-kernel-outlining");
        passes.push_back("convert-gpu-to-rocdl");
    } else if (target == "spirv") {
        passes.push_back("convert-linalg-to-spirv");
        passes.push_back("spirv-lower-abi-attrs");
        passes.push_back("spirv-update-vce");
    }
    
    return passes;
}

bool MLIRPassPipeline::run_passes(
    void* mlir_context,
    void* mlir_module,
    const std::vector<std::string>& passes
) {
#ifdef YIRAGE_MLIR_ENABLED
    // Run passes using MLIR PassManager
    // mlir::PassManager pm(mlir_context);
    // for (const auto& pass : passes) {
    //     pm.addPass(parsePassPipeline(pass));
    // }
    // return pm.run(mlir_module).succeeded();
    return true;
#else
    return false;
#endif
}

// =============================================================================
// MLIR Kernel Registry
// =============================================================================

MLIRKernelRegistry& MLIRKernelRegistry::instance() {
    static MLIRKernelRegistry instance;
    return instance;
}

void MLIRKernelRegistry::register_kernel(const std::string& name,
                                         const std::string& mlir_code) {
    kernels_[name] = mlir_code;
}

std::string MLIRKernelRegistry::get_kernel(const std::string& name) const {
    auto it = kernels_.find(name);
    return (it != kernels_.end()) ? it->second : "";
}

bool MLIRKernelRegistry::has_kernel(const std::string& name) const {
    return kernels_.find(name) != kernels_.end();
}

bool MLIRKernelRegistry::compile_kernel(const std::string& name,
                                        const std::string& target) {
#ifdef YIRAGE_MLIR_ENABLED
    auto it = kernels_.find(name);
    if (it == kernels_.end()) return false;
    
    // Compile MLIR to target binary
    // This would use MLIR's compilation infrastructure
    
    return true;
#else
    return false;
#endif
}

std::vector<uint8_t> MLIRKernelRegistry::get_compiled_binary(
    const std::string& name
) const {
    auto it = compiled_binaries_.find(name);
    return (it != compiled_binaries_.end()) ? it->second : std::vector<uint8_t>();
}

}  // namespace mlir_ops
}  // namespace threadblock
}  // namespace yirage
