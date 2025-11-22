// common.h - Some common definitions for the transpiler
#pragma once

#include "yirage/kernel/device_tensor.h"
#include "yirage/kernel/graph.h"
#include "yirage/threadblock/graph.h"
#include "yirage/threadblock/smem_tensor.h"

namespace yirage {
namespace transpiler {

namespace kn = yirage::kernel;
namespace tb = yirage::threadblock;
using dguid_t = decltype(kn::DTensor::guid); // Guid of a DTensor
using sguid_t = decltype(tb::STensor::guid); // Guid of a STensor

} // namespace transpiler
} // namespace yirage
