#include "yirage/search/search_c.h"
#include "yirage/kernel/customized.h"
#include "yirage/kernel/graph.h"
#include "yirage/search/dim_strategy.h"
#include "yirage/search/op_utils.h"
#include "yirage/search/search.h"
#include "yirage/utils/containers.h"
#include "yirage/type.h"

#include <cstring>
#include <fstream>
#include <iostream>

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
      } else if (config.backend_type == type::BT_CUDA) {
        config.warp_size = 32;  // NVIDIA CUDA uses 32-thread warps
        std::cout << "[Search] Using CUDA backend (warpSize=32)" << std::endl;
      } else if (config.backend_type == type::BT_ASCEND) {
        // Ascend NPU uses AI Cores, not warps
        // Set warp_size to 1 since Ascend doesn't have warp concept
        config.warp_size = 1;
        std::cout << "[Search] Using Ascend NPU backend (AI Core based)" << std::endl;
        std::cout << "  - No warp concept, using AI Core parallelism" << std::endl;
        std::cout << "  - Cube operations: 16x16 matrix tiles" << std::endl;
      } else {
        config.warp_size = 1;  // CPU/other backends
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
    if (is_formal_verified) {
      config.verifier_type = search::VerifierType::FORMAL_VERIFIER;
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
