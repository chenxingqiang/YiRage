# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.14)
project(YiRageTests CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# =============================================================================
# Options
# =============================================================================

option(YIRAGE_USE_GTEST "Use Google Test instead of built-in framework" ON)
option(YIRAGE_TEST_COVERAGE "Enable code coverage" OFF)

# =============================================================================
# Google Test Setup
# =============================================================================

if(YIRAGE_USE_GTEST)
    include(FetchContent)
    
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.15.2
    )
    
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    # Ensure C++17 is used for googletest
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    # Disable gmock - only use gtest
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    # Force non-system include paths for gtest to avoid clang issues with -isystem
    set(CMAKE_NO_SYSTEM_FROM_IMPORTED ON CACHE BOOL "" FORCE)
    
    FetchContent_MakeAvailable(googletest)
    
    # Override the SYSTEM property to use regular includes instead of system includes
    # This fixes issues with clang and parameterized tests
    get_target_property(GTEST_INC_DIRS gtest INTERFACE_INCLUDE_DIRECTORIES)
    set_target_properties(gtest PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "")
    set_target_properties(gtest_main PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "")
    
    enable_testing()
    
    message(STATUS "Using Google Test framework")
else()
    message(STATUS "Using built-in YiRage test framework")
endif()

# =============================================================================
# Include Paths
# =============================================================================

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../include)

# =============================================================================
# Test Executables
# =============================================================================

# Backend tests (all backends)
if(YIRAGE_USE_GTEST)
    add_executable(test_backends_gtest
        test_backends_gtest.cc
    )
    target_link_libraries(test_backends_gtest
        GTest::gtest_main
    )
    include(GoogleTest)
    gtest_discover_tests(test_backends_gtest)

    # Search engine tests
    add_executable(test_search_gtest
        test_search_gtest.cc
    )
    target_link_libraries(test_search_gtest
        GTest::gtest_main
    )
    gtest_discover_tests(test_search_gtest)

    # Strategy tests
    add_executable(test_strategies_gtest
        test_strategies_gtest.cc
    )
    target_link_libraries(test_strategies_gtest
        GTest::gtest_main
    )
    gtest_discover_tests(test_strategies_gtest)

    # Transpiler tests
    add_executable(test_transpiler_gtest
        test_transpiler_gtest.cc
    )
    target_link_libraries(test_transpiler_gtest
        GTest::gtest_main
    )
    gtest_discover_tests(test_transpiler_gtest)

    # PK Runtime tests
    add_executable(test_pk_runtime_gtest
        test_pk_runtime_gtest.cc
    )
    target_link_libraries(test_pk_runtime_gtest
        GTest::gtest_main
    )
    gtest_discover_tests(test_pk_runtime_gtest)

    # MLIR dialect tests
    add_executable(test_mlir_gtest
        test_mlir_gtest.cc
    )
    target_link_libraries(test_mlir_gtest
        GTest::gtest_main
    )
    gtest_discover_tests(test_mlir_gtest)

    # Backend implementation tests (comprehensive)
    add_executable(test_backend_impl_gtest
        test_backend_impl_gtest.cc
    )
    target_link_libraries(test_backend_impl_gtest
        GTest::gtest_main
    )
    gtest_discover_tests(test_backend_impl_gtest)

    # Base module tests (type.h, layout.h, data_type.cc, layout.cc)
    add_executable(test_base_gtest
        test_base_gtest.cc
    )
    target_link_libraries(test_base_gtest
        GTest::gtest_main
    )
    gtest_discover_tests(test_base_gtest)

    # NKI transpiler tests
    add_executable(test_nki_transpiler_gtest
        test_nki_transpiler_gtest.cc
    )
    target_link_libraries(test_nki_transpiler_gtest
        GTest::gtest_main
    )
    gtest_discover_tests(test_nki_transpiler_gtest)

    # Kernel module tests (DTensor, Graph, Operators)
    add_executable(test_kernel_gtest
        test_kernel_gtest.cc
    )
    target_link_libraries(test_kernel_gtest
        GTest::gtest_main
    )
    gtest_discover_tests(test_kernel_gtest)

    # Ascend kernel module tests
    add_executable(test_kernel_ascend_gtest
        test_kernel_ascend_gtest.cc
    )
    target_link_libraries(test_kernel_ascend_gtest
        GTest::gtest_main
    )
    gtest_discover_tests(test_kernel_ascend_gtest)

    # CPU kernel backend tests
    add_executable(test_kernel_cpu_gtest
        test_kernel_cpu_gtest.cc
    )
    target_link_libraries(test_kernel_cpu_gtest
        GTest::gtest_main
    )
    gtest_discover_tests(test_kernel_cpu_gtest)

    # CUDA kernel backend tests
    add_executable(test_kernel_cuda_gtest
        test_kernel_cuda_gtest.cc
    )
    target_link_libraries(test_kernel_cuda_gtest
        GTest::gtest_main
    )
    gtest_discover_tests(test_kernel_cuda_gtest)

    # ROCm kernel backend tests
    add_executable(test_kernel_rocm_gtest
        test_kernel_rocm_gtest.cc
    )
    target_link_libraries(test_kernel_rocm_gtest
        GTest::gtest_main
    )
    gtest_discover_tests(test_kernel_rocm_gtest)

    # MPS (Apple Metal) kernel backend tests
    add_executable(test_kernel_mps_gtest
        test_kernel_mps_gtest.cc
    )
    target_link_libraries(test_kernel_mps_gtest
        GTest::gtest_main
    )
    gtest_discover_tests(test_kernel_mps_gtest)

    # Other kernel backends tests (Triton, MKL, NKI, TPU, XPU, FPGA, MACA)
    add_executable(test_kernel_other_backends_gtest
        test_kernel_other_backends_gtest.cc
    )
    target_link_libraries(test_kernel_other_backends_gtest
        GTest::gtest_main
    )
    gtest_discover_tests(test_kernel_other_backends_gtest)

    # =========================================================================
    # Search Module Tests
    # =========================================================================

    # Abstract expression tests
    add_executable(test_search_abstract_expr_gtest
        test_search_abstract_expr_gtest.cc
    )
    target_link_libraries(test_search_abstract_expr_gtest
        GTest::gtest_main
    )
    gtest_discover_tests(test_search_abstract_expr_gtest)

    # Range propagation tests
    add_executable(test_search_range_gtest
        test_search_range_gtest.cc
    )
    target_link_libraries(test_search_range_gtest
        GTest::gtest_main
    )
    gtest_discover_tests(test_search_range_gtest)

    # Symbolic graph tests
    add_executable(test_search_symbolic_graph_gtest
        test_search_symbolic_graph_gtest.cc
    )
    target_link_libraries(test_search_symbolic_graph_gtest
        GTest::gtest_main
    )
    gtest_discover_tests(test_search_symbolic_graph_gtest)

    # Verification module tests
    add_executable(test_search_verification_gtest
        test_search_verification_gtest.cc
    )
    target_link_libraries(test_search_verification_gtest
        GTest::gtest_main
    )
    gtest_discover_tests(test_search_verification_gtest)

    # Search config and strategy tests
    add_executable(test_search_config_gtest
        test_search_config_gtest.cc
    )
    target_link_libraries(test_search_config_gtest
        GTest::gtest_main
    )
    gtest_discover_tests(test_search_config_gtest)

    # Distributed search tests
    add_executable(test_search_distributed_gtest
        test_search_distributed_gtest.cc
    )
    target_link_libraries(test_search_distributed_gtest
        GTest::gtest_main
    )
    gtest_discover_tests(test_search_distributed_gtest)

    # Graph features tests
    add_executable(test_search_features_gtest
        test_search_features_gtest.cc
    )
    target_link_libraries(test_search_features_gtest
        GTest::gtest_main
    )
    gtest_discover_tests(test_search_features_gtest)

    # =========================================================================
    # Persistent Kernel Module Tests
    # =========================================================================

    # PK backend interface tests
    add_executable(test_pk_backend_interface_gtest
        test_pk_backend_interface_gtest.cc
    )
    target_link_libraries(test_pk_backend_interface_gtest
        GTest::gtest_main
    )
    gtest_discover_tests(test_pk_backend_interface_gtest)

    # PK task and memory tests
    add_executable(test_pk_task_gtest
        test_pk_task_gtest.cc
    )
    target_link_libraries(test_pk_task_gtest
        GTest::gtest_main
    )
    gtest_discover_tests(test_pk_task_gtest)

    # PK utilities tests
    add_executable(test_pk_utils_gtest
        test_pk_utils_gtest.cc
    )
    target_link_libraries(test_pk_utils_gtest
        GTest::gtest_main
    )
    gtest_discover_tests(test_pk_utils_gtest)

    # =========================================================================
    # Threadblock Module Tests
    # =========================================================================

    # Threadblock shared memory tensor tests
    add_executable(test_tb_smem_tensor_gtest
        test_tb_smem_tensor_gtest.cc
    )
    target_link_libraries(test_tb_smem_tensor_gtest
        GTest::gtest_main
    )
    gtest_discover_tests(test_tb_smem_tensor_gtest)

    # Threadblock operator tests
    add_executable(test_tb_operator_gtest
        test_tb_operator_gtest.cc
    )
    target_link_libraries(test_tb_operator_gtest
        GTest::gtest_main
    )
    gtest_discover_tests(test_tb_operator_gtest)

    # Threadblock graph tests
    add_executable(test_tb_graph_gtest
        test_tb_graph_gtest.cc
    )
    target_link_libraries(test_tb_graph_gtest
        GTest::gtest_main
    )
    gtest_discover_tests(test_tb_graph_gtest)

    # Threadblock matmul tests
    add_executable(test_tb_matmul_gtest
        test_tb_matmul_gtest.cc
    )
    target_link_libraries(test_tb_matmul_gtest
        GTest::gtest_main
    )
    gtest_discover_tests(test_tb_matmul_gtest)

    # Threadblock element operations tests
    add_executable(test_tb_element_ops_gtest
        test_tb_element_ops_gtest.cc
    )
    target_link_libraries(test_tb_element_ops_gtest
        GTest::gtest_main
    )
    gtest_discover_tests(test_tb_element_ops_gtest)

    # Threadblock reduction tests
    add_executable(test_tb_reduction_gtest
        test_tb_reduction_gtest.cc
    )
    target_link_libraries(test_tb_reduction_gtest
        GTest::gtest_main
    )
    gtest_discover_tests(test_tb_reduction_gtest)

    # =========================================================================
    # Transpiler Module Tests (detailed)
    # =========================================================================

    # Transpiler config tests
    add_executable(test_transpiler_config_gtest
        test_transpiler_config_gtest.cc
    )
    target_link_libraries(test_transpiler_config_gtest
        GTest::gtest_main
    )
    gtest_discover_tests(test_transpiler_config_gtest)

    # Transpiler structs tests
    add_executable(test_transpiler_structs_gtest
        test_transpiler_structs_gtest.cc
    )
    target_link_libraries(test_transpiler_structs_gtest
        GTest::gtest_main
    )
    gtest_discover_tests(test_transpiler_structs_gtest)

    # Transpiler sched tests
    add_executable(test_transpiler_sched_gtest
        test_transpiler_sched_gtest.cc
    )
    target_link_libraries(test_transpiler_sched_gtest
        GTest::gtest_main
    )
    gtest_discover_tests(test_transpiler_sched_gtest)

    # Transpiler utils tests
    add_executable(test_transpiler_utils_gtest
        test_transpiler_utils_gtest.cc
    )
    target_link_libraries(test_transpiler_utils_gtest
        GTest::gtest_main
    )
    gtest_discover_tests(test_transpiler_utils_gtest)

    # Transpiler memory tests
    add_executable(test_transpiler_memory_gtest
        test_transpiler_memory_gtest.cc
    )
    target_link_libraries(test_transpiler_memory_gtest
        GTest::gtest_main
    )
    gtest_discover_tests(test_transpiler_memory_gtest)

    # Transpiler Ascend tests
    add_executable(test_transpiler_ascend_gtest
        test_transpiler_ascend_gtest.cc
    )
    target_link_libraries(test_transpiler_ascend_gtest
        GTest::gtest_main
    )
    gtest_discover_tests(test_transpiler_ascend_gtest)

    # Transpiler layout and swizzle tests
    add_executable(test_transpiler_layout_gtest
        test_transpiler_layout_gtest.cc
    )
    target_link_libraries(test_transpiler_layout_gtest
        GTest::gtest_main
    )
    gtest_discover_tests(test_transpiler_layout_gtest)

else()
    # Built-in framework tests
    add_executable(test_search
        test_search.cc
    )

    add_executable(test_strategies
        test_strategies.cc
    )

    add_executable(test_layer1_backend_api
        layer1_backend_api_test.cc
    )
endif()

# =============================================================================
# Custom Target: Run All Tests
# =============================================================================

if(YIRAGE_USE_GTEST)
    add_custom_target(run_all_cpp_tests
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
        DEPENDS test_backends_gtest test_search_gtest test_strategies_gtest
                test_transpiler_gtest test_pk_runtime_gtest test_mlir_gtest
                test_backend_impl_gtest test_base_gtest test_nki_transpiler_gtest
                test_kernel_gtest test_kernel_ascend_gtest
                test_kernel_cpu_gtest test_kernel_cuda_gtest
                test_kernel_rocm_gtest test_kernel_mps_gtest
                test_kernel_other_backends_gtest
                test_search_abstract_expr_gtest test_search_range_gtest
                test_search_symbolic_graph_gtest test_search_verification_gtest
                test_search_config_gtest test_search_distributed_gtest
                test_search_features_gtest
                test_pk_backend_interface_gtest test_pk_task_gtest
                test_pk_utils_gtest
                test_tb_smem_tensor_gtest test_tb_operator_gtest
                test_tb_graph_gtest test_tb_matmul_gtest
                test_tb_element_ops_gtest test_tb_reduction_gtest
                test_transpiler_config_gtest test_transpiler_structs_gtest
                test_transpiler_sched_gtest test_transpiler_utils_gtest
                test_transpiler_memory_gtest test_transpiler_ascend_gtest
                test_transpiler_layout_gtest
        COMMENT "Running all C++ GTest tests"
    )
else()
    add_custom_target(run_all_cpp_tests
        COMMAND $<TARGET_FILE:test_search>
        COMMAND $<TARGET_FILE:test_strategies>
        COMMAND $<TARGET_FILE:test_layer1_backend_api>
        COMMENT "Running all C++ tests with built-in framework"
    )
endif()

# =============================================================================
# Coverage (optional)
# =============================================================================

if(YIRAGE_TEST_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    message(STATUS "Enabling code coverage")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
endif()
