/* libcudacxx cuda::std forwarders for MetaX mxcc + CUTLASS. */
#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <tuple>
#include <type_traits>
#include <utility>

namespace cuda {
namespace std {
using ::std::swap;
using ::std::tuple;
using ::std::make_tuple;
using ::std::get;
using ::std::tuple_size;
using ::std::tuple_size_v;
using ::std::tuple_element;
using ::std::tuple_element_t;
using ::std::is_same;
using ::std::is_same_v;
using ::std::is_integral;
using ::std::is_integral_v;
using ::std::integral_constant;
using ::std::enable_if;
using ::std::enable_if_t;
using ::std::declval;
using ::std::remove_cv;
using ::std::remove_cv_t;
using ::std::remove_reference;
using ::std::remove_reference_t;
using ::std::numeric_limits;
} // namespace std
} // namespace cuda
