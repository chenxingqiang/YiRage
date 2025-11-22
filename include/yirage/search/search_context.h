#pragma once

#include <memory>

#include <atomic>

#include "yirage/kernel/graph.h"
#include "yirage/threadblock/graph.h"

namespace yirage {
namespace search {

enum class SearchLevel {
  LV_KERNEL,
  LV_THREADBLOCK,
};

struct SearchContext {
  std::shared_ptr<kernel::Graph> kn_graph;
  std::shared_ptr<threadblock::Graph> tb_graph;
  SearchLevel level;
};

void from_json(json const &j, SearchContext &c);
void to_json(json &j, SearchContext const &c);

class SerializedSearchContext {
public:
  SerializedSearchContext(SearchContext const &c);
  SearchContext deserialize() const;

private:
  json data;
};

} // namespace search
} // namespace yirage
