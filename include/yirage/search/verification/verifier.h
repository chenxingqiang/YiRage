#pragma once

#include "yirage/kernel/graph.h"
#include "yirage/search/verification/output_match.h"

namespace yirage {
namespace search {

class Verifier {
public:
  Verifier() = default;
  virtual OutputMatch verify(kernel::Graph const &graph) = 0;
  virtual ~Verifier() = default;
};

} // namespace search
} // namespace yirage
