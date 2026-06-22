# Copyright 2025 Chen Xingqiang (YiRage Project)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# =============================================================================
# Persistent Kernel Multi-Backend Build Configuration
# =============================================================================

cmake_minimum_required(VERSION 3.18)

# =============================================================================
# Source Files
# =============================================================================

set(PK_BACKEND_SOURCES
    pk_backend_factory.cc
    pk_runtime_adapter.cc
    pk_cpu_kernels.cc
    cpu_pk_backend.cc
)

set(PK_BACKEND_HEADERS
    ${PROJECT_SOURCE_DIR}/include/persistent_kernel/pk_backend_interface.h
    ${PROJECT_SOURCE_DIR}/include/persistent_kernel/pk_runtime_adapter.h
    ${PROJECT_SOURCE_DIR}/include/persistent_kernel/pk_runtime_core.h
    ${PROJECT_SOURCE_DIR}/include/persistent_kernel/pk_utils.h
    ${PROJECT_SOURCE_DIR}/include/persistent_kernel/pk_task_kernels.h
    ${PROJECT_SOURCE_DIR}/include/persistent_kernel/backends/pk_backends.h
    ${PROJECT_SOURCE_DIR}/include/persistent_kernel/backends/cpu_pk_backend.h
    ${PROJECT_SOURCE_DIR}/include/persistent_kernel/backends/cpu_pk_runtime.h
)

# =============================================================================
# CUDA Backend (if CUDA enabled)
# =============================================================================

if(YIRAGE_BACKEND_CUDA_ENABLED)
    list(APPEND PK_BACKEND_SOURCES
        cuda_pk_backend.cc
    )
    list(APPEND PK_BACKEND_HEADERS
        ${PROJECT_SOURCE_DIR}/include/persistent_kernel/backends/cuda_pk_backend.h
    )
    add_definitions(-DYIRAGE_BACKEND_CUDA_ENABLED)
endif()

# =============================================================================
# Ascend Backend (if Ascend enabled)
# =============================================================================

if(YIRAGE_BACKEND_ASCEND_ENABLED)
    list(APPEND PK_BACKEND_SOURCES
        ascend_pk_backend.cc
    )
    list(APPEND PK_BACKEND_HEADERS
        ${PROJECT_SOURCE_DIR}/include/persistent_kernel/backends/ascend_pk_backend.h
        ${PROJECT_SOURCE_DIR}/include/persistent_kernel/backends/ascend_pk_runtime.h
    )
    add_definitions(-DYIRAGE_BACKEND_ASCEND_ENABLED)
endif()

# =============================================================================
# MACA Backend (if MACA enabled)
# =============================================================================

if(YIRAGE_BACKEND_MACA_ENABLED)
    list(APPEND PK_BACKEND_SOURCES
        maca_pk_backend.cc
    )
    list(APPEND PK_BACKEND_HEADERS
        ${PROJECT_SOURCE_DIR}/include/persistent_kernel/backends/maca_pk_backend.h
        ${PROJECT_SOURCE_DIR}/include/persistent_kernel/backends/maca_pk_runtime.h
    )
    add_definitions(-DYIRAGE_BACKEND_MACA_ENABLED)
endif()

# =============================================================================
# ROCm Backend (if ROCm enabled)
# =============================================================================

if(YIRAGE_BACKEND_ROCM_ENABLED)
    list(APPEND PK_BACKEND_SOURCES
        rocm_pk_backend.cc
    )
    list(APPEND PK_BACKEND_HEADERS
        ${PROJECT_SOURCE_DIR}/include/persistent_kernel/backends/rocm_pk_backend.h
        ${PROJECT_SOURCE_DIR}/include/persistent_kernel/backends/rocm_pk_runtime.h
    )
    add_definitions(-DYIRAGE_BACKEND_ROCM_ENABLED)
endif()

# =============================================================================
# MPS Backend (if Apple and MPS enabled)
# =============================================================================

if(APPLE AND YIRAGE_BACKEND_MPS_ENABLED)
    list(APPEND PK_BACKEND_SOURCES
        mps_pk_backend.cc
    )
    list(APPEND PK_BACKEND_HEADERS
        ${PROJECT_SOURCE_DIR}/include/persistent_kernel/backends/mps_pk_backend.h
        ${PROJECT_SOURCE_DIR}/include/persistent_kernel/backends/mps_pk_runtime.h
    )
    add_definitions(-DYIRAGE_BACKEND_MPS_ENABLED)
endif()

# =============================================================================
# Create Library
# =============================================================================

add_library(yirage_pk_backends STATIC
    ${PK_BACKEND_SOURCES}
)

target_include_directories(yirage_pk_backends PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

# C++17 for std::optional, std::variant, etc.
target_compile_features(yirage_pk_backends PUBLIC cxx_std_17)

# Thread support
find_package(Threads REQUIRED)
target_link_libraries(yirage_pk_backends PUBLIC Threads::Threads)

# OpenMP support (optional, for CPU backend)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(yirage_pk_backends PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(yirage_pk_backends PRIVATE _OPENMP)
endif()

# =============================================================================
# Backend-specific linking
# =============================================================================

if(YIRAGE_BACKEND_CUDA_ENABLED)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(yirage_pk_backends PUBLIC
        CUDA::cudart
    )
endif()

if(YIRAGE_BACKEND_ASCEND_ENABLED)
    # Link Ascend ACL libraries
    find_library(ACL_LIBRARY ascendcl HINTS $ENV{ASCEND_HOME}/lib64)
    if(ACL_LIBRARY)
        target_link_libraries(yirage_pk_backends PUBLIC ${ACL_LIBRARY})
    endif()
endif()

if(YIRAGE_BACKEND_MACA_ENABLED)
    # Link MACA runtime libraries
    find_library(MACA_RUNTIME mc_runtime HINTS $ENV{MACA_HOME}/lib)
    if(MACA_RUNTIME)
        target_link_libraries(yirage_pk_backends PUBLIC ${MACA_RUNTIME})
    endif()
endif()

if(YIRAGE_BACKEND_ROCM_ENABLED)
    # Link ROCm/HIP runtime libraries
    find_package(hip HINTS $ENV{ROCM_PATH}/lib/cmake/hip)
    if(hip_FOUND)
        target_link_libraries(yirage_pk_backends PUBLIC hip::host)
    else()
        find_library(HIP_RUNTIME amdhip64 HINTS $ENV{ROCM_PATH}/lib)
        if(HIP_RUNTIME)
            target_link_libraries(yirage_pk_backends PUBLIC ${HIP_RUNTIME})
        endif()
    endif()
endif()

if(APPLE AND YIRAGE_BACKEND_MPS_ENABLED)
    # Link Metal and MPS frameworks
    find_library(METAL_FRAMEWORK Metal)
    find_library(MPS_FRAMEWORK MetalPerformanceShaders)
    find_library(FOUNDATION_FRAMEWORK Foundation)
    if(METAL_FRAMEWORK AND MPS_FRAMEWORK)
        target_link_libraries(yirage_pk_backends PUBLIC
            ${METAL_FRAMEWORK}
            ${MPS_FRAMEWORK}
            ${FOUNDATION_FRAMEWORK}
        )
    endif()
endif()

# =============================================================================
# Tests
# =============================================================================

if(BUILD_TESTING)
    add_executable(test_pk_backends
        ${PROJECT_SOURCE_DIR}/tests/persistent_kernel/test_pk_backends.cc
    )
    target_link_libraries(test_pk_backends PRIVATE yirage_pk_backends)
    add_test(NAME test_pk_backends COMMAND test_pk_backends)
    
    if(APPLE AND YIRAGE_BACKEND_MPS_ENABLED)
        add_executable(test_mps_pk_runtime
            ${PROJECT_SOURCE_DIR}/tests/persistent_kernel/test_mps_pk_runtime.cc
        )
        target_link_libraries(test_mps_pk_runtime PRIVATE yirage_pk_backends)
        add_test(NAME test_mps_pk_runtime COMMAND test_mps_pk_runtime)
    endif()
endif()

# =============================================================================
# Examples
# =============================================================================

if(BUILD_EXAMPLES)
    add_executable(pk_multi_backend_demo
        ${PROJECT_SOURCE_DIR}/examples/pk_multi_backend_demo.cc
    )
    target_link_libraries(pk_multi_backend_demo PRIVATE yirage_pk_backends)
endif()

# =============================================================================
# Install
# =============================================================================

install(TARGETS yirage_pk_backends
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/persistent_kernel
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)
