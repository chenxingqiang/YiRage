#include "search/search_c.h"
#include "kernel/customized.h"
#include "kernel/graph.h"
#include "search/dim_strategy.h"
#include "search/op_utils.h"
#include "search/search.h"
#include "utils/containers.h"
#include "type.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <string>
#include <fstream>
#include <iostream>
#include <thread>

namespace yirage {
namespace search_c {

int cython_search(yirage::kernel::Graph const *input_graph,
                  char const *backend,
                  int max_num_graphs,
                  yirage::kernel::Graph **new_graphs,
                  std::vector<MInt3> imap_to_explore,
                  std::vector<MInt3> omap_to_explore,
                  std::vector<MDim3> grid_dim_to_explore,
                  std::vector<MDim3> block_dim_to_explore,
                  std::vector<int> fmap_to_explore,
                  std::vector<int> frange_to_explore,
                  char const *filename,
                  bool verbose,
                  char const *default_config,
                  bool is_formal_verified) {
  if (filename) {
    std::ifstream generated_graphs_file(filename, std::ifstream::binary);
    if (generated_graphs_file) {
      json j;
      generated_graphs_file >> j;
      int num = 0;
      for (json const &graph : j) {
        assert(num < max_num_graphs);
        new_graphs[num] = new kernel::Graph();
        from_json(graph, *new_graphs[num]);
        num++;
      }
      return num;
    }
  }
  {
    search::GeneratorConfig config =
        search::GeneratorConfig::get_default_config();
    
    // Set backend type from parameter
    if (backend != nullptr) {
      std::string backend_str(backend);
      config.backend_type = type::string_to_backend_type(backend_str);
      
      // Set backend-specific configurations
      if (config.backend_type == type::BT_MACA) {
        config.warp_size = 64;  // MetaX MACA uses 64-thread warps
        std::cout << "[Search] Using MACA backend (warpSize=64)" << std::endl;
        bool maca_quick = true;
        if (const char *qs = std::getenv("YIRAGE_MACA_SEARCH_QUICK")) {
          std::string s(qs);
          for (auto &c : s) {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
          }
          maca_quick = !(s == "0" || s == "false" || s == "no" || s == "off");
        }
        if (maca_quick) {
          std::cout << "[Search] MACA quick search (YIRAGE_MACA_SEARCH_QUICK)"
                    << std::endl;
          config.max_num_threadblock_graph_op = 6;
          config.max_num_kernel_graph_op = 3;
          config.search_thread =
              std::max<size_t>(1, std::min(config.search_thread, size_t{4}));
          // Plain matmul / fused GEMM smoke: avoid full default explore grid.
          config.knop_to_explore = {
              type::KN_MATMUL_OP,
              type::KN_CUSTOMIZED_OP,
          };
          config.tbop_to_explore = {
              type::TB_MATMUL_OP,
              type::TB_RMS_NORM_OP,
              type::TB_FORLOOP_ACCUM_NO_RED_OP,
              type::TB_FORLOOP_ACCUM_RED_LD_SUM_OP,
              type::TB_FORLOOP_ACCUM_RED_LD_RMS_OP,
              type::TB_ADD_OP,
              type::TB_MUL_OP,
          };
        }
      } else if (config.backend_type == type::BT_ROCM) {
        config.warp_size = 64;  // AMD ROCm uses 64-thread wavefronts
        std::cout << "[Search] Using ROCm backend (wavefrontSize=64)" << std::endl;
        std::cout << "  - LDS (Local Data Share) for shared memory" << std::endl;
        std::cout << "  - MFMA instructions for matrix ops" << std::endl;
      } else if (config.backend_type == type::BT_CUDA) {
        config.warp_size = 32;  // NVIDIA CUDA uses 32-thread warps
        std::cout << "[Search] Using CUDA backend (warpSize=32)" << std::endl;
      } else if (config.backend_type == type::BT_MPS) {
        config.warp_size = 32;  // Apple Metal SIMD width is 32 threads
        std::cout << "[Search] Using MPS backend (simd_width=32)" << std::endl;
        std::cout << "  - Apple Silicon unified memory" << std::endl;
        std::cout << "  - 32KB threadgroup memory" << std::endl;
        std::cout << "  - Max 1024 threads per threadgroup" << std::endl;
      } else if (config.backend_type == type::BT_ASCEND) {
        // Ascend NPU uses AI Cores, not warps
        config.warp_size = 1;
        std::cout << "[Search] Using Ascend NPU backend (AI Core based)" << std::endl;
        std::cout << "  - No warp concept, using AI Core parallelism" << std::endl;
        std::cout << "  - Cube operations: 16x16 matrix tiles" << std::endl;
      } else if (config.backend_type == type::BT_TPU) {
        config.warp_size = 1;  // TPU uses MXU, no warp concept
        std::cout << "[Search] Using TPU backend (MXU 128x128)" << std::endl;
        std::cout << "  - VMEM-based memory model" << std::endl;
        std::cout << "  - XLA/Pallas kernel generation" << std::endl;
      } else if (config.backend_type == type::BT_FPGA) {
        config.warp_size = 1;  // FPGA is configurable
        std::cout << "[Search] Using FPGA backend (HLS-based)" << std::endl;
        std::cout << "  - DSP slices for compute" << std::endl;
        std::cout << "  - BRAM/URAM for on-chip memory" << std::endl;
      } else if (config.backend_type == type::BT_XPU) {
        config.warp_size = 16;  // XPU sub-group is 16
        std::cout << "[Search] Using Intel XPU backend (simd_width=16)" << std::endl;
        std::cout << "  - XMX (Xe Matrix eXtensions)" << std::endl;
        std::cout << "  - DPAS (Dot Product Accumulate Systolic)" << std::endl;
      } else if (config.backend_type == type::BT_CPU) {
        {
          search::GeneratorConfig cpu_cfg =
              search::GeneratorConfig::get_cpu_search_config();
          config.knop_to_explore = cpu_cfg.knop_to_explore;
          config.tbop_to_explore = cpu_cfg.tbop_to_explore;
        }
        config.warp_size = 1;
        std::cout << "[Search] Using CPU backend (SIMD + OpenMP tiling)" << std::endl;
        if (const char *st = std::getenv("YIRAGE_CPU_SEARCH_THREADS")) {
          config.search_thread = static_cast<size_t>(std::max(1, std::atoi(st)));
        } else {
          config.search_thread = std::max<size_t>(
              1, std::thread::hardware_concurrency());
        }
        if (const char *tb = std::getenv("YIRAGE_CPU_MAX_TB_GRAPH_OP")) {
          config.max_num_threadblock_graph_op =
              static_cast<int>(std::max(1, std::atoi(tb)));
        } else {
          config.max_num_threadblock_graph_op = 6;
        }
        if (const char *kn = std::getenv("YIRAGE_CPU_MAX_KN_GRAPH_OP")) {
          config.max_num_kernel_graph_op =
              static_cast<int>(std::max(1, std::atoi(kn)));
        } else {
          config.max_num_kernel_graph_op = 4;
        }
        if (const char *ti = std::getenv("YIRAGE_CPU_MAX_TB_GRAPH_INPUTS")) {
          config.max_num_threadblock_graph_inputs =
              static_cast<size_t>(std::max(1, std::atoi(ti)));
        }
        if (const char *bm = std::getenv("YIRAGE_CPU_BENCH_MINIMAL_EXPLORE")) {
          std::string s(bm);
          for (auto &c : s) {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
          }
          if (s == "1" || s == "true" || s == "yes" || s == "on") {
            // Fusion bench: GEMM + rms_matmul patterns only (skip layout explore).
            config.knop_to_explore = {
                type::KN_MATMUL_OP,
                type::KN_CUSTOMIZED_OP,
            };
            config.tbop_to_explore = {
                type::TB_MATMUL_OP,
                type::TB_RMS_NORM_OP,
                type::TB_FORLOOP_ACCUM_NO_RED_OP,
                type::TB_FORLOOP_ACCUM_RED_LD_SUM_OP,
                type::TB_FORLOOP_ACCUM_RED_LD_RMS_OP,
                type::TB_ADD_OP,
                type::TB_MUL_OP,
            };
          }
        }
        if (const char *kn = std::getenv("YIRAGE_SERVING_KN_MATMUL_ONLY")) {
          std::string s(kn);
          for (auto &c : s) {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
          }
          if (s == "1" || s == "true" || s == "yes" || s == "on") {
            // Serving down-proj: prefer plain KN matmul; allow 1 TB matmul tile if needed.
            config.max_num_threadblock_graphs = 1;
            config.max_num_kernel_graph_op = 2;
            config.max_num_threadblock_graph_op = 1;
            config.knop_to_explore = {type::KN_MATMUL_OP, type::KN_CUSTOMIZED_OP};
            config.tbop_to_explore = {type::TB_MATMUL_OP};
          }
        }
      } else {
        config.warp_size = 1;  // other backends
      }
    }
    
    if (default_config != nullptr) {
      if (!strcmp(default_config, "attention")) {
        config.enable_attention_specific_optimization();
      } else if (!strcmp(default_config, "lora")) {
        config.enable_concat_matmul_transformation();
      } else if (!strcmp(default_config, "mlp")) {
      }
    }
    bool use_formal = is_formal_verified;
    if (!use_formal) {
      if (const char *fv = std::getenv("YIRAGE_FORMAL_VERIFY")) {
        std::string s(fv);
        for (auto &c : s) {
          c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
        use_formal = (s == "1" || s == "true" || s == "yes" || s == "on");
      }
    }
    if (use_formal) {
      config.verifier_type = search::VerifierType::FORMAL_VERIFIER;
      std::cout << "[Search] Verifier: formal (YIRAGE_FORMAL_VERIFY)" << std::endl;
    } else {
      std::cout << "[Search] Verifier: probabilistic fingerprint (default)"
                << std::endl;
    }
    // Customized imaps
    if (imap_to_explore.size() > 0) {
      config.imap_to_explore.clear();
      for (auto const &imap : imap_to_explore) {
        config.imap_to_explore.push_back({imap.x, imap.y, imap.z});
      }
    }
    // Customized omaps
    if (omap_to_explore.size() > 0) {
      config.omap_to_explore.clear();
      for (auto const &omap : omap_to_explore) {
        config.omap_to_explore.push_back({omap.x, omap.y, omap.z});
      }
    }
    // Customized griddims
    if (grid_dim_to_explore.size() > 0) {
      config.grid_dim_to_explore.clear();
      for (auto const &griddim : grid_dim_to_explore) {
        config.grid_dim_to_explore.push_back({griddim.x, griddim.y, griddim.z});
      }
    }
    // Customized blockdims
    if (block_dim_to_explore.size() > 0) {
      config.block_dim_to_explore.clear();
      for (auto const &blockdim : block_dim_to_explore) {
        config.block_dim_to_explore.push_back(
            {blockdim.x, blockdim.y, blockdim.z});
      }
    }
    // Customized fmap
    if (fmap_to_explore.size() > 0) {
      config.fmap_to_explore.clear();
      for (auto const &fmap : fmap_to_explore) {
        config.fmap_to_explore.push_back(fmap);
      }
    }
    // Customized frange
    if (frange_to_explore.size() > 0) {
      config.frange_to_explore.clear();
      for (auto const &frange : frange_to_explore) {
        config.frange_to_explore.push_back(frange);
      }
    }
    char const *result_filename =
        filename ? filename : "yirage_search_checkpoint.json";
    search::KernelGraphGenerator gen(
        *input_graph, config, result_filename, verbose);
    gen.config.show();
    gen.generate_kernel_graphs();
    int num = 0;
    for (json const &j : gen.generated_graphs) {
      assert(num < max_num_graphs);
      new_graphs[num] = new kernel::Graph();
      from_json(j, *new_graphs[num]);
      num++;
    }
    return num;
  }
}

void cython_to_json(yirage::kernel::Graph const *input_graph,
                    char const *filename) {
  json j;
  to_json(j, *input_graph);
  std::ofstream ofs(filename);
  ofs << j;
}

yirage::kernel::Graph *cython_from_json(char const *filename) {
  std::ifstream graph_file(filename, std::ifstream::binary);
  json j;
  graph_file >> j;
  yirage::kernel::Graph *new_graph = new yirage::kernel::Graph();
  from_json(j, *new_graph);
  return new_graph;
}

} // namespace search_c
} // namespace yirage
