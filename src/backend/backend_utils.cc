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
 * This file is part of YiRage (Yi Revolutionary AGile Engine),
 * a derivative work based on Mirage by CMU.
 * Original Mirage Copyright 2023-2024 CMU.
 */


#include "yirage/type.h"
#include <unordered_map>

namespace yirage {
namespace type {

std::string backend_type_to_string(BackendType type) {
  static std::unordered_map<BackendType, std::string> const type_to_string = {
      // GPU Backends
      {BT_CUDA, "cuda"},
      {BT_MPS, "mps"},
      {BT_CUDNN, "cudnn"},
      {BT_CUSPARSELT, "cusparselt"},
      {BT_ASCEND, "ascend"},
      {BT_MACA, "maca"},
      // CPU Backends
      {BT_CPU, "cpu"},
      {BT_MKL, "mkl"},
      {BT_MKLDNN, "mkldnn"},
      {BT_OPENMP, "openmp"},
      {BT_XEON, "xeon"},
      // Specialized Backends
      {BT_NKI, "nki"},
      {BT_TRITON, "triton"},
      {BT_MHA, "mha"},
      {BT_NNPACK, "nnpack"},
      {BT_OPT_EINSUM, "opt_einsum"},
      {BT_UNKNOWN, "unknown"},
  };

  auto it = type_to_string.find(type);
  if (it != type_to_string.end()) {
    return it->second;
  }
  return "unknown";
}

BackendType string_to_backend_type(std::string const &name) {
  static std::unordered_map<std::string, BackendType> const string_to_type = {
      // GPU Backends
      {"cuda", BT_CUDA},
      {"mps", BT_MPS},
      {"cudnn", BT_CUDNN},
      {"cusparselt", BT_CUSPARSELT},
      {"ascend", BT_ASCEND},
      {"maca", BT_MACA},
      // CPU Backends
      {"cpu", BT_CPU},
      {"mkl", BT_MKL},
      {"mkldnn", BT_MKLDNN},
      {"openmp", BT_OPENMP},
      {"xeon", BT_XEON},
      // Specialized Backends
      {"nki", BT_NKI},
      {"triton", BT_TRITON},
      {"mha", BT_MHA},
      {"nnpack", BT_NNPACK},
      {"opt_einsum", BT_OPT_EINSUM},
      {"unknown", BT_UNKNOWN},
  };

  auto it = string_to_type.find(name);
  if (it != string_to_type.end()) {
    return it->second;
  }
  return BT_UNKNOWN;
}

} // namespace type
} // namespace yirage

