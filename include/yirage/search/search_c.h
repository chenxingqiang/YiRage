#pragma once
#include "yirage/kernel/graph.h"
#include "yirage/vector_types.h"

namespace yirage {
namespace search_c {

struct MInt3 {
  int x, y, z;
};

struct MDim3 {
  unsigned int x, y, z;
};

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
                  bool is_formal_verified);

void cython_to_json(yirage::kernel::Graph const *input_graph,
                    char const *filename);
yirage::kernel::Graph *cython_from_json(char const *filename);
} // namespace search_c
} // namespace yirage
