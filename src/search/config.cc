#include "search/config.h"

namespace yirage {
namespace search {

GeneratorConfig GeneratorConfig::get_cpu_search_config() {
  // Search uses global clamp bounds when exploring clamp ops.
  type::CLAMP_MIN_MAX["min_val"] = -1.0f;
  type::CLAMP_MIN_MAX["max_val"] = 1.0f;

  GeneratorConfig config = get_default_config();
  // Dual-output TB ops (e.g. reduction_max) may expose max + diff as graph leaves.
  config.max_num_threadblock_graph_outputs = 3;
  // Keep in sync with docs/cpu_support_matrix.yaml search_cpu_default_explore.
  config.knop_to_explore = {
      type::KN_MATMUL_OP,
      type::KN_EXP_OP,
      type::KN_SQUARE_OP,
      type::KN_SQRT_OP,
      type::KN_SILU_OP,
      type::KN_GELU_OP,
      type::KN_RELU_OP,
      type::KN_SIGMOID_OP,
      type::KN_LOG_OP,
      type::KN_CLAMP_OP,
      type::KN_MUL_SCALAR_OP,
      type::KN_REDUCTION_0_OP,
      type::KN_REDUCTION_1_OP,
      type::KN_REDUCTION_2_OP,
      type::KN_ADD_OP,
      type::KN_MUL_OP,
      type::KN_DIV_OP,
      type::KN_POW_OP,
      type::KN_CONCAT_0_OP,
      type::KN_CONCAT_1_OP,
      type::KN_CONCAT_2_OP,
      type::KN_SPLIT_0_OP,
      type::KN_SPLIT_1_OP,
      type::KN_SPLIT_2_OP,
      type::KN_CHUNK_0_OP,
      type::KN_CHUNK_1_OP,
      type::KN_CHUNK_2_OP,
      type::KN_CUSTOMIZED_OP,
  };
  config.tbop_to_explore = {
      type::TB_MATMUL_OP,
      type::TB_EXP_OP,
      type::TB_SQUARE_OP,
      type::TB_SQRT_OP,
      type::TB_SILU_OP,
      type::TB_GELU_OP,
      type::TB_RELU_OP,
      type::TB_SIGMOID_OP,
      type::TB_LOG_OP,
      type::TB_CLAMP_OP,
      type::TB_MUL_SCALAR_OP,
      type::TB_ADD_OP,
      type::TB_SUB_OP,
      type::TB_MUL_OP,
      type::TB_DIV_OP,
      type::TB_POW_OP,
      type::TB_RMS_NORM_OP,
      type::TB_CONCAT_0_OP,
      type::TB_CONCAT_1_OP,
      type::TB_CONCAT_2_OP,
      type::TB_SPLIT_0_OP,
      type::TB_SPLIT_1_OP,
      type::TB_SPLIT_2_OP,
      type::TB_CHUNK_0_OP,
      type::TB_CHUNK_1_OP,
      type::TB_CHUNK_2_OP,
      type::TB_REDUCTION_0_OP,
      type::TB_REDUCTION_1_OP,
      type::TB_REDUCTION_2_OP,
      type::TB_REDUCTION_0_TO_DIMX_OP,
      type::TB_REDUCTION_1_TO_DIMX_OP,
      type::TB_REDUCTION_2_TO_DIMX_OP,
      type::TB_REDUCTION_0_MAX_OP,
      type::TB_REDUCTION_1_MAX_OP,
      type::TB_REDUCTION_2_MAX_OP,
      type::TB_FORLOOP_ACCUM_NO_RED_OP,
      type::TB_FORLOOP_ACCUM_RED_LD_SUM_OP,
      type::TB_FORLOOP_ACCUM_RED_LD_MEAN_OP,
      type::TB_FORLOOP_ACCUM_RED_LD_RMS_OP,
      type::TB_FORLOOP_ACCUM_REDTOX_LD_SUM_OP,
      type::TB_FORLOOP_ACCUM_MAX_OP,
  };
  // Expands to concat+concat+matmul (all CPU-supported after Loop R14/R15).
  config.enable_concat_matmul_transformation();
  return config;
}

GeneratorConfig GeneratorConfig::get_default_config() {
  return {
      9 /* max_num_threadblock_graph_op */,
      5 /* max_num_kernel_graph_op */,
      1 /* max_num_threadblock_graphs */,
      3 /* max_num_threadblock_graph_inputs */,
      2 /* max_num_threadblock_graph_outputs */,
      16 /* search_thread */,
      VerifierType::PROBABILISTIC_VERIFIER,
      type::BT_CUDA /* backend_type - default to CUDA */,
      32 /* warp_size - default to CUDA's 32 */,
      {
          type::KN_MATMUL_OP,
          type::KN_EXP_OP,
          type::KN_SQUARE_OP,
          type::KN_SQRT_OP,
          type::KN_SILU_OP,
          type::KN_GELU_OP,
          type::KN_RELU_OP,
          type::KN_CLAMP_OP,
          type::KN_ADD_OP,
          type::KN_MUL_OP,
          type::KN_DIV_OP,
          type::KN_POW_OP,
          // type::KN_REDUCTION_2_OP,
          type::KN_CUSTOMIZED_OP,
      } /* knop_to_explore */,
      {
          type::TB_MATMUL_OP,
          type::TB_EXP_OP,
          type::TB_SQUARE_OP,
          type::TB_SQRT_OP,
          type::TB_SILU_OP,
          type::TB_GELU_OP,
          type::TB_RELU_OP,
          type::TB_CLAMP_OP,
          type::TB_ADD_OP,
          type::TB_MUL_OP,
          type::TB_DIV_OP,
          type::TB_POW_OP,
          type::TB_RMS_NORM_OP,
          type::TB_FORLOOP_ACCUM_NO_RED_OP,
          type::TB_FORLOOP_ACCUM_RED_LD_SUM_OP,
          // type::TB_FORLOOP_ACCUM_RED_LD_MEAN_OP,
          // type::TB_FORLOOP_ACCUM_REDTOX_LD_SUM_OP,
          type::TB_FORLOOP_ACCUM_RED_LD_RMS_OP,
      } /* tbop_to_explore */,
      {} /* imap_to_explore*/,
      {} /* imap_comb_to_explore */,
      {} /* omap_to_explore */,
      {} /* grid_dim_to_explore*/,
      {} /* block_dim_to_explore */,
      {} /* fmap_to_explore */,
      {
          4,
          16,
          64,
      } /* frange_to_explore */,
      64 /* reduction_dimx */,
      {} /* grid_dim_candidates - empty = use defaults */,
      {} /* frange_candidates - empty = use defaults */,
      false /* randomized_branches */,
      false /* _enable_attention_specific_optimization */,
      false /* _enable_concat_matmul_transformation */,
  };
}

void GeneratorConfig::enable_attention_specific_optimization() {
  _enable_attention_specific_optimization = true;
  max_num_threadblock_graphs = 2;
  tbop_to_explore.push_back(type::TB_FORLOOP_ACCUM_REDTOX_LD_SUM_OP);
  deduplicate(tbop_to_explore);
}

void GeneratorConfig::enable_concat_matmul_transformation() {
  _enable_concat_matmul_transformation = true;
}

void GeneratorConfig::show() const {
  printf("========== Search Configuration ==========\n");
  
  // Show backend-specific information
  // Map backend_type enum to name
  const char* backend_name = "UNKNOWN";
  switch (backend_type) {
    case type::BT_CUDA:     backend_name = "CUDA"; break;
    case type::BT_MPS:      backend_name = "MPS"; break;
    case type::BT_CUDNN:    backend_name = "CUDNN"; break;
    case type::BT_CUSPARSELT: backend_name = "CUSPARSELT"; break;
    case type::BT_ASCEND:   backend_name = "ASCEND"; break;
    case type::BT_MACA:     backend_name = "MACA"; break;
    case type::BT_ROCM:     backend_name = "ROCM"; break;
    case type::BT_TPU:      backend_name = "TPU"; break;
    case type::BT_FPGA:     backend_name = "FPGA"; break;
    case type::BT_XPU:      backend_name = "XPU"; break;
    case type::BT_CPU:      backend_name = "CPU"; break;
    default: backend_name = "UNKNOWN"; break;
  }
  printf("  backend_type: %s (%d)\n", backend_name, static_cast<int>(backend_type));
  
  if (backend_type == type::BT_ASCEND) {
    printf("  architecture: Huawei Ascend NPU (AI Core based)\n");
    printf("  parallelism: AI Core blocks (no warp concept)\n");
    printf("  cube_unit: 16x16 matrix tiles\n");
  } else if (backend_type == type::BT_MACA) {
    printf("  warp_size: %d (MetaX 64-thread warps)\n", warp_size);
  } else if (backend_type == type::BT_ROCM) {
    printf("  wavefront_size: %d (AMD 64-thread wavefronts)\n", warp_size);
  } else if (backend_type == type::BT_CUDA) {
    printf("  warp_size: %d (NVIDIA 32-thread warps)\n", warp_size);
  } else if (backend_type == type::BT_MPS) {
    printf("  simd_width: %d (Apple Metal SIMD groups)\n", warp_size);
    printf("  threadgroup_memory: 32KB\n");
    printf("  max_threads_per_threadgroup: 1024\n");
  } else if (backend_type == type::BT_TPU) {
    printf("  architecture: Google TPU (MXU 128x128)\n");
    printf("  parallelism: VMEM-based, no warp concept\n");
  } else if (backend_type == type::BT_FPGA) {
    printf("  architecture: FPGA (HLS-based)\n");
    printf("  parallelism: DSP slices, configurable\n");
  } else if (backend_type == type::BT_XPU) {
    printf("  architecture: Intel XPU (XMX 8x16)\n");
    printf("  simd_width: 16 (sub-group based)\n");
  } else {
    printf("  warp_size: %d\n", warp_size);
  }
  printf("  max num threadblock graph op: %zu\n", max_num_threadblock_graph_op);
  printf("  max num kernel_graph op: %zu\n", max_num_kernel_graph_op);
  printf("  max num threadblock graphs: %zu\n", max_num_threadblock_graphs);
  printf("  max num threadblock graph inputs: %zu\n",
         max_num_threadblock_graph_inputs);
  printf("  max num threadblock graph outputs: %zu\n",
         max_num_threadblock_graph_outputs);
  printf("  search_thread: %zu\n", search_thread);
  printf("  imaps to explore:\n");
  for (auto const &imap : imap_to_explore) {
    printf("    (%d, %d, %d)\n", imap.x, imap.y, imap.z);
  }
  printf("  imap combs to explore:\n");
  for (auto const &imap_comb : imap_comb_to_explore) {
    for (auto const &imap : imap_comb) {
      printf("    (%d, %d, %d), ", imap.x, imap.y, imap.z);
    }
    printf("\n");
  }
  printf("  omaps to explore:\n");
  for (auto const &omap : omap_to_explore) {
    printf("    (%d, %d, %d)\n", omap.x, omap.y, omap.z);
  }
  printf("  grid dims to explore:\n");
  for (auto const &griddim : grid_dim_to_explore) {
    printf("    (%d, %d, %d)\n", griddim.x, griddim.y, griddim.z);
  }
  printf("  block dims to explore:\n");
  for (auto const &blockdim : block_dim_to_explore) {
    printf("    (%d, %d, %d)\n", blockdim.x, blockdim.y, blockdim.z);
  }
  printf("  fmaps to explore:");
  for (auto const &fmap : fmap_to_explore) {
    printf("%d ", fmap);
  }
  printf("\n");
  printf("  franges to explore:");
  for (auto const &frange : frange_to_explore) {
    printf("%d ", frange);
  }
  printf("\n");
}

bool TBGraphConfig::operator==(TBGraphConfig const &other) const {
  return grid_dim == other.grid_dim && block_dim == other.block_dim &&
         imaps == other.imaps && fmaps == other.fmaps && frange == other.frange;
}

void TBGraphConfig::show() const {
  printf("========== Threadblock Graph Configuration ==========\n");
  printf("  grid dim: (%d, %d, %d)\n", grid_dim.x, grid_dim.y, grid_dim.z);
  printf("  block dim: (%d, %d, %d)\n", block_dim.x, block_dim.y, block_dim.z);
  printf("  imaps:\n");
  for (auto const &imap : imaps) {
    printf("    (%d, %d, %d)\n", imap.x, imap.y, imap.z);
  }
  printf("  fmaps:");
  for (auto const &fmap : fmaps) {
    printf("%d ", fmap);
  }
  printf("\n");
  printf("  frange: %d\n", frange);
}

} // namespace search
} // namespace yirage

namespace std {

size_t hash<yirage::search::TBGraphConfig>::operator()(
    yirage::search::TBGraphConfig const &config) const {
  size_t hash = 0;
  hash_combine(hash, config.grid_dim);
  hash_combine(hash, config.block_dim);
  hash_combine(hash, config.imaps);
  hash_combine(hash, config.fmaps);
  hash_combine(hash, config.frange);
  return hash;
}

} // namespace std