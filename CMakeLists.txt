cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
message(STATUS "CMake Version: ${CMAKE_VERSION}")

if (EXISTS ${CMAKE_CURRENT_BINARY_DIR}/config.cmake)
  include (${CMAKE_CURRENT_BINARY_DIR}/config.cmake)
else()
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/config.cmake)
    include(${CMAKE_CURRENT_SOURCE_DIR}/config.cmake)
  endif()
endif()

# Determine which languages to enable based on backends
set(PROJECT_LANGUAGES C CXX)
if (NOT "${USE_CUDA}" STREQUAL "OFF")
  list(APPEND PROJECT_LANGUAGES CUDA)
  message(STATUS "CUDA backend enabled")
endif()
# Note: MACA does NOT use CMake's CUDA language - it uses mxcc compiler directly
if (NOT "${USE_MACA}" STREQUAL "OFF")
  message(STATUS "MACA backend enabled (uses mxcc compiler, not CMake CUDA)")
endif()

project(YIRAGE LANGUAGES ${PROJECT_LANGUAGES})

#include directories
include_directories(${CMAKE_INCLUDE_PATH})
include_directories(${CMAKE_BINARY_DIR}) # to include protobuf headeer files
include_directories("include")

# Add cutlass include directory (required for cmem_tensor.h and other headers)
# For MACA builds, this uses the MACA SDK's cute/cub libraries via symlinks
include_directories("deps/cutlass/include")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_LIST_DIR}/cmake)

#add_subdirectory(deps/cutlass)

set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90")
#set(CUTLASS_NVCC_ARCHS "80;86")

file(GLOB_RECURSE YIRAGE_SRCS
  src/*.cc
)

# Collect backend-specific sources
file(GLOB BACKEND_SRCS src/backend/*.cc)
list(APPEND YIRAGE_SRCS ${BACKEND_SRCS})

# Collect kernel optimizer sources
file(GLOB_RECURSE KERNEL_OPT_SRCS
  src/kernel/common/*.cc
  src/kernel/cuda/*.cc
  src/kernel/cpu/*.cc
  src/kernel/mps/*.cc
  src/kernel/triton/*.cc
  src/kernel/nki/*.cc
  src/kernel/cudnn/*.cc
  src/kernel/mkl/*.cc
  src/kernel/ascend/*.cc
  src/kernel/maca/*.cc
  # Note: src/kernel/maca/*.cpp files contain GPU kernel code that requires
  # mxcc compiler. They are excluded from standard host builds.
  # For full MACA kernel support, these should be compiled with mxcc.
)
list(APPEND YIRAGE_SRCS ${KERNEL_OPT_SRCS})

# Collect search strategy sources
file(GLOB SEARCH_COMMON_SRCS src/search/common/*.cc)
file(GLOB SEARCH_STRATEGY_SRCS src/search/backend_strategies/*.cc)
file(GLOB SEARCH_PROFILER_SRCS src/search/mps_profiler.cc)
list(APPEND YIRAGE_SRCS ${SEARCH_COMMON_SRCS})
list(APPEND YIRAGE_SRCS ${SEARCH_STRATEGY_SRCS})
list(APPEND YIRAGE_SRCS ${SEARCH_PROFILER_SRCS})

# Handle CUDA-specific sources
if (NOT "${USE_CUDA}" STREQUAL "OFF")
  # CUDA enabled: include .cu files and transpiler
  file(GLOB_RECURSE YIRAGE_CUDA_SRCS
    src/*.cu
  )
  list(APPEND YIRAGE_SRCS ${YIRAGE_CUDA_SRCS})
  message(STATUS "CUDA backend enabled, including .cu files and transpiler")
else()
  # CUDA disabled: exclude transpiler, triton_transpiler, and nki_transpiler
  file(GLOB_RECURSE YIRAGE_EXCLUDE_SRCS
    src/transpiler/*.cc
    src/triton_transpiler/*.cc
    src/nki_transpiler/*.cc
  )
  list(REMOVE_ITEM YIRAGE_SRCS ${YIRAGE_EXCLUDE_SRCS})
  
  # But keep transpiler_stub.cc for non-CUDA builds
  list(APPEND YIRAGE_SRCS src/transpiler/transpiler_stub.cc)
  
  # Keep ascend_transpiler_stub.cc for Ascend backend
  if (NOT "${USE_ASCEND}" STREQUAL "OFF")
    list(APPEND YIRAGE_SRCS src/transpiler/ascend_transpiler_stub.cc)
    message(STATUS "Including Ascend transpiler stub")
  endif()
  
  message(STATUS "CUDA backend disabled, excluding transpilers (except stubs)")
  
  # Keep all kernel/*.cc files as they contain backend-agnostic code
  # The CUDA-specific parts are protected by #ifdef
endif()

# Handle CPU backend
if (NOT "${USE_CPU}" STREQUAL "OFF")
  message(STATUS "CPU backend enabled")
endif()

# Handle other backends
if (NOT "${USE_MPS}" STREQUAL "OFF")
  message(STATUS "MPS backend enabled")
endif()

if (NOT "${USE_NKI}" STREQUAL "OFF")
  message(STATUS "NKI backend enabled")
endif()

if (NOT "${USE_ASCEND}" STREQUAL "OFF")
  message(STATUS "Ascend backend enabled (requires CANN + torch_npu at runtime)")
endif()

if (NOT "${USE_MACA}" STREQUAL "OFF")
  message(STATUS "MACA backend enabled (requires MetaX MACA SDK)")
endif()

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++17" SUPPORT_CXX17)

#set CUDA version
if (NOT "${USE_CUDA}" STREQUAL "OFF")
  include(deps/cutlass/CUDA.cmake)
  string(REPLACE "." ";" CUDA_VERSION_PARTS ${CUDA_VERSION})
  list(GET CUDA_VERSION_PARTS 0 CUDA_VERSION_MAJOR)
  list(GET CUDA_VERSION_PARTS 1 CUDA_VERSION_MINOR)
  list(APPEND CMAKE_CUDA_FLAGS "-D__CUDACC_VER_MAJOR__=${CUDA_VERSION_MAJOR} -D__CUDACC_VER_MINOR__=${CUDA_VERSION_MINOR}")
endif()

# TODO: Currently disable CUTLASS_ARCH_MMA_SM80_ENABLED flag
# since we target Triton to perform codegen
#list(APPEND CMAKE_CUDA_FLAGS "-DCUTLASS_ARCH_MMA_SM80_ENABLED")

if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  message("Build in Debug mode")
  set(CMAKE_CUDA_FLAGS "-O0 -g -Xcompiler=-fPIC ${CMAKE_CUDA_FLAGS}")
  set(CMAKE_CXX_FLAGS "-O0 -g -Wall -fPIC ${CMAKE_CXX_FLAGS}")
else()
  set(CMAKE_CUDA_FLAGS "-O2 -Xcompiler=-fPIC ${CMAKE_CUDA_FLAGS}")
  set(CMAKE_CXX_FLAGS "-O2 -Wall -fPIC ${CMAKE_CXX_FLAGS}")
endif()

# Add OpenMP flags only if enabled and available
if (NOT "${USE_OPENMP}" STREQUAL "OFF")
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  endif()
endif()

#set CUDA
if (NOT "${USE_CUDA}" STREQUAL "OFF")
  include(cmake/cuda.cmake)
  find_cuda(${USE_CUDA})
  if (CUDA_FOUND)
    list(APPEND YIRAGE_SRCS ${YIRAGE_CUDA_SRCS})
    include_directories(${CUDA_INCLUDE_DIRS})
    message(STATUS "CUDA_INCLUDE_DIR=" ${CUDA_INCLUDE_DIRS})
    list(APPEND YIRAGE_LINK_LIBS ${CUDA_CUDART_LIBRARY})
    list(APPEND YIRAGE_LINK_LIBS ${CUDA_CUDA_LIBRARY})
  else()
    message(FATAL_ERROR "Cannot find CUDA, USE_CUDA=" ${USE_CUDA})
  endif(CUDA_FOUND)
endif()

#set CUTLASS
set(CUTLASS_DIR ${PROJECT_SOURCE_DIR}/deps/cutlass)
include_directories(deps/cutlass/include)
include_directories(deps/cutlass/tools/util/include)

#include_directories(${YIRAGE_INCLUDE_DIRS})

#set Z3
if(Z3_CXX_INCLUDE_DIRS AND Z3_LIBRARIES)
  message(STATUS "Z3_CXX_INCLUDE_DIRS: ${Z3_CXX_INCLUDE_DIRS}")
  message(STATUS "Z3_LIBRARIES: ${Z3_LIBRARIES}")
else()
  find_package(Z3 REQUIRED)
  message(STATUS "Z3_FOUND: ${Z3_FOUND}")
  message(STATUS "Found Z3 ${Z3_VERSION_STRING}")
  message(STATUS "Z3_DIR: ${Z3_DIR}")
endif()
include_directories(${Z3_CXX_INCLUDE_DIRS})
#target_link_libraries(yirage_runtime PUBLIC ${Z3_LIBRARIES})
list(APPEND YIRAGE_LINK_LIBS ${Z3_LIBRARIES})

add_subdirectory(deps/json)
list(APPEND YIRAGE_LINK_LIBS nlohmann_json::nlohmann_json)

link_directories(${CMAKE_SOURCE_DIR}/build/release)
list(APPEND YIRAGE_LINK_LIBS abstract_subexpr)

#add_subdirectory(deps/z3 z3 EXCLUDE_FROM_ALL)
#include_directories(deps/z3/src/api/c++)
#include_directories(deps/z3/src/api)
#list(APPEND YIRAGE_LINK_LIBS z3)

add_library(yirage_runtime ${YIRAGE_SRCS})

# Note(zhihao): CUDA_SEPARABLE_COMPILATION is non-compatible with
# cython's installation, since enabling separable compilation
# will let cmake generate a seperate object file called
# cmake_device_link.o, and cython's installation cannot
# automatically link to that file, resulting in:
# undefined symbol: __cudaRegisterLinkedBinary_...
#set_target_properties(yirage_runtime
#  PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Ensure Rust is installed
execute_process(COMMAND rustc --version
  RESULT_VARIABLE RUST_COMMAND_RESULT
  OUTPUT_VARIABLE RUSTC_OUTPUT
  ERROR_QUIET)

if(NOT RUST_COMMAND_RESULT EQUAL 0)
  message(FATAL_ERROR
    "Rust is not installed on the system. Please install it by running: \n"
    "'curl https://sh.rustup.rs -sSf | sh -s -- -y' \n"
    "and follow the instructions on the screen.")
endif()

# Ensure Cargo is installed
execute_process(COMMAND cargo --version
  RESULT_VARIABLE CARGO_RESULT
  OUTPUT_QUIET ERROR_QUIET)

if(NOT CARGO_RESULT EQUAL 0)
  message(FATAL_ERROR
    "Rust is installed, but cargo is not. Please install it by running: \n"
    "'curl https://sh.rustup.rs -sSf | sh -s -- -y' \n"
    "and follow the instructions on the screen.")
endif()

set(ABSTRACT_SUBEXPR_LIBRARIES ${PROJECT_SOURCE_DIR}/build/abstract_subexpr/release/libabstract_subexpr.so)
set(FORMAL_VERIFIER_LIBRARIES ${PROJECT_SOURCE_DIR}/build/formal_verifier/release/libformal_verifier.so)
list(APPEND YIRAGE_LINK_LIBS "-lstdc++fs")

#set RUST
if(ABSTRACT_SUBEXPR_LIB AND ABSTRACT_SUBEXPR_LIBRARIES)
  message(STATUS "ABSTRACT_SUBEXPR_LIB: ${ABSTRACT_SUBEXPR_LIB}")
  message(STATUS "ABSTRACT_SUBEXPR_LIBRARIES: ${ABSTRACT_SUBEXPR_LIBRARIES}")
else()
  message(STATUS "ABSTRACT_SUBEXPR_LIB not set, will build it")
endif()
include_directories(${ABSTRACT_SUBEXPR_LIB})
list(APPEND YIRAGE_LINK_LIBS ${ABSTRACT_SUBEXPR_LIBRARIES})

if(FORMAL_VERIFIER_LIB AND FORMAL_VERIFIER_LIBRARIES)
  message(STATUS "FORMAL_VERIFIER_LIB: ${FORMAL_VERIFIER_LIB}")
  message(STATUS "FORMAL_VERIFIER_LIBRARIES: ${FORMAL_VERIFIER_LIBRARIES}")
else()
  message(STATUS "FORMAL_VERIFIER_LIB not set, will build it")
endif()
include_directories(${FORMAL_VERIFIER_LIB})
list(APPEND YIRAGE_LINK_LIBS ${FORMAL_VERIFIER_LIBRARIES})

set_target_properties(yirage_runtime PROPERTIES CUDA_ARCHITECTURES "75;80;86;89;90")

target_compile_features(yirage_runtime PUBLIC cxx_std_17)

# Backend compile definitions - multiple backends can be enabled
if (NOT "${USE_CUDA}" STREQUAL "OFF")
  target_compile_definitions(yirage_runtime PUBLIC YIRAGE_BACKEND_CUDA_ENABLED)
  target_compile_definitions(yirage_runtime PUBLIC YIRAGE_BACKEND_USE_CUDA)  # Backward compatibility
  target_compile_definitions(yirage_runtime PUBLIC YIRAGE_FINGERPRINT_USE_CUDA)
  
  # Enable OpenMP for parallel search with CUDA
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(yirage_runtime PUBLIC OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP enabled for CUDA parallel search")
  endif()
elseif (NOT "${USE_MACA}" STREQUAL "OFF")
  # MACA is CUDA-compatible, use MACA fingerprint (enables parallel search)
  # Don't define YIRAGE_FINGERPRINT_USE_CPU here - it disables OpenMP
  message(STATUS "Using MACA fingerprint (parallel search enabled)")
elseif (NOT "${USE_ASCEND}" STREQUAL "OFF")
  # Ascend NPU - don't define CPU fingerprint, Ascend has its own
  message(STATUS "Using Ascend fingerprint (parallel search enabled)")
else()
  # CPU fingerprint - still enable parallel search with OpenMP
  target_compile_definitions(yirage_runtime PUBLIC YIRAGE_FINGERPRINT_USE_CPU)
  
  # Enable OpenMP for parallel search even with CPU fingerprint
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(yirage_runtime PUBLIC OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP enabled for CPU parallel search")
  else()
    message(WARNING "OpenMP not found - search will be single-threaded")
  endif()
endif()

if (NOT "${USE_CPU}" STREQUAL "OFF")
  target_compile_definitions(yirage_runtime PUBLIC YIRAGE_BACKEND_CPU_ENABLED)
endif()

if (NOT "${USE_MPS}" STREQUAL "OFF")
  target_compile_definitions(yirage_runtime PUBLIC YIRAGE_BACKEND_MPS_ENABLED)
  
  # Enable OpenMP for parallel search with MPS (Apple Metal)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(yirage_runtime PUBLIC OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP enabled for MPS parallel search")
  else()
    # On macOS, OpenMP might need libomp from Homebrew
    message(STATUS "OpenMP not found for MPS - try: brew install libomp")
  endif()
endif()

if (NOT "${USE_CUDNN}" STREQUAL "OFF")
  target_compile_definitions(yirage_runtime PUBLIC YIRAGE_BACKEND_CUDNN_ENABLED)
  # CUDNN uses CUDA's OpenMP, already linked above
endif()

if (NOT "${USE_MKL}" STREQUAL "OFF")
  target_compile_definitions(yirage_runtime PUBLIC YIRAGE_BACKEND_MKL_ENABLED)
  # MKL has its own OpenMP, enable parallel search
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(yirage_runtime PUBLIC OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP enabled for MKL parallel search")
  endif()
endif()

if (NOT "${USE_MKLDNN}" STREQUAL "OFF")
  target_compile_definitions(yirage_runtime PUBLIC YIRAGE_BACKEND_MKLDNN_ENABLED)
  # MKLDNN/oneDNN uses OpenMP internally
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(yirage_runtime PUBLIC OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP enabled for MKLDNN parallel search")
  endif()
endif()

if (NOT "${USE_OPENMP}" STREQUAL "OFF")
  target_compile_definitions(yirage_runtime PUBLIC YIRAGE_BACKEND_OPENMP_ENABLED)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(yirage_runtime PUBLIC OpenMP::OpenMP_CXX)
  endif()
endif()

if (NOT "${USE_XEON}" STREQUAL "OFF")
  target_compile_definitions(yirage_runtime PUBLIC YIRAGE_BACKEND_XEON_ENABLED)
  # Intel Xeon - enable parallel search with OpenMP
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(yirage_runtime PUBLIC OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP enabled for Xeon parallel search")
  endif()
endif()

if (NOT "${USE_NKI}" STREQUAL "OFF")
  target_compile_definitions(yirage_runtime PUBLIC YIRAGE_BACKEND_NKI_ENABLED)
  target_compile_definitions(yirage_runtime PUBLIC YIRAGE_BACKEND_USE_NKI)  # Backward compatibility
  # NKI (AWS Trainium/Inferentia) - use CPU fingerprint for development
  target_compile_definitions(yirage_runtime PUBLIC YIRAGE_FINGERPRINT_USE_CPU)
  
  # Enable OpenMP for parallel search with NKI
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(yirage_runtime PUBLIC OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP enabled for NKI parallel search")
  endif()
endif()

if (NOT "${USE_TRITON}" STREQUAL "OFF")
  target_compile_definitions(yirage_runtime PUBLIC YIRAGE_BACKEND_TRITON_ENABLED)
endif()

if (NOT "${USE_ASCEND}" STREQUAL "OFF")
  target_compile_definitions(yirage_runtime PUBLIC YIRAGE_BACKEND_ASCEND_ENABLED)
  # Ascend NPU fingerprint (enables parallel search)
  target_compile_definitions(yirage_runtime PUBLIC YIRAGE_FINGERPRINT_USE_ASCEND)
  
  # Enable OpenMP for parallel search with Ascend
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(yirage_runtime PUBLIC OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP enabled for Ascend parallel search")
  endif()
endif()

# MACA Backend (MetaX GPU)
if (NOT "${USE_MACA}" STREQUAL "OFF")
  # Include MACA configuration
  include(${CMAKE_CURRENT_SOURCE_DIR}/config.maca.cmake)
  target_compile_definitions(yirage_runtime PUBLIC YIRAGE_BACKEND_MACA_ENABLED)
  # MACA uses GPU fingerprint for verification (compiled with mxcc)
  target_compile_definitions(yirage_runtime PUBLIC YIRAGE_FINGERPRINT_USE_MACA)
  
  # Enable OpenMP for parallel search with MACA
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(yirage_runtime PUBLIC OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP enabled for MACA parallel search")
  else()
    message(WARNING "OpenMP not found - MACA search will be single-threaded")
  endif()
  
  # Set MACA paths
  if(DEFINED ENV{MACA_PATH})
    set(MACA_SDK_PATH $ENV{MACA_PATH})
  elseif(DEFINED ENV{MACA_HOME})
    set(MACA_SDK_PATH $ENV{MACA_HOME})
  else()
    set(MACA_SDK_PATH "/opt/maca")
  endif()
  
  # Add MACA SDK include directories
  target_include_directories(yirage_runtime PUBLIC ${MACA_SDK_PATH}/include)
  target_link_directories(yirage_runtime PUBLIC ${MACA_SDK_PATH}/lib)
  
  # Find mxcc compiler for MACA kernel compilation
  find_program(MXCC_COMPILER mxcc
    PATHS 
      ${MACA_SDK_PATH}/mxgpu_llvm/bin
      ${MACA_SDK_PATH}/bin
      /opt/maca/mxgpu_llvm/bin
  )
  
  if(MXCC_COMPILER)
    message(STATUS "Found MACA compiler: ${MXCC_COMPILER}")
    
    # Collect MACA GPU kernel source files (.maca files)
    file(GLOB MACA_GPU_KERNEL_SRCS src/kernel/maca/*.maca)
    
    # Compile each MACA GPU kernel file with mxcc
    set(MACA_KERNEL_OBJS "")
    foreach(MACA_SRC ${MACA_GPU_KERNEL_SRCS})
      get_filename_component(MACA_SRC_NAME ${MACA_SRC} NAME_WE)
      set(MACA_OBJ ${CMAKE_CURRENT_BINARY_DIR}/maca_kernels/${MACA_SRC_NAME}.o)
      
      add_custom_command(
        OUTPUT ${MACA_OBJ}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/maca_kernels
        COMMAND ${MXCC_COMPILER} -x maca
          -c ${MACA_SRC}
          -o ${MACA_OBJ}
          -I${PROJECT_SOURCE_DIR}/include
          -I${MACA_SDK_PATH}/include
          -I${MACA_SDK_PATH}/include/mcr
          -I${PROJECT_SOURCE_DIR}/deps/cutlass/include
          -I${PROJECT_SOURCE_DIR}/deps/json/include
          -DYIRAGE_BACKEND_MACA_ENABLED
          -DYIRAGE_FINGERPRINT_USE_MACA
          -std=c++17
          -fPIC
          -O2
          --maca-path=${MACA_SDK_PATH}
        DEPENDS ${MACA_SRC}
        COMMENT "Compiling MACA GPU kernel: ${MACA_SRC_NAME}"
      )
      list(APPEND MACA_KERNEL_OBJS ${MACA_OBJ})
    endforeach()
    
    # Create a custom target for MACA kernels
    if(MACA_KERNEL_OBJS)
      add_custom_target(maca_kernels DEPENDS ${MACA_KERNEL_OBJS})
      add_dependencies(yirage_runtime maca_kernels)
      
      # Link MACA kernel objects to the runtime library  
      target_sources(yirage_runtime PRIVATE ${MACA_KERNEL_OBJS})
      message(STATUS "MACA GPU kernels: ${MACA_GPU_KERNEL_SRCS}")
    else()
      message(STATUS "No MACA GPU kernel files (.maca) found")
    endif()
    
    # Link MACA runtime library
    list(APPEND YIRAGE_LINK_LIBS mcruntime)
  else()
    message(WARNING "MACA compiler (mxcc) not found - falling back to CPU fingerprint")
    # Fall back to CPU fingerprint
    target_compile_definitions(yirage_runtime PUBLIC YIRAGE_FINGERPRINT_USE_CPU)
  endif()
endif()

if (NOT "${USE_MHA}" STREQUAL "OFF")
  target_compile_definitions(yirage_runtime PUBLIC YIRAGE_BACKEND_MHA_ENABLED)
endif()

if (NOT "${USE_NNPACK}" STREQUAL "OFF")
  target_compile_definitions(yirage_runtime PUBLIC YIRAGE_BACKEND_NNPACK_ENABLED)
endif()

if (NOT "${USE_OPT_EINSUM}" STREQUAL "OFF")
  target_compile_definitions(yirage_runtime PUBLIC YIRAGE_BACKEND_OPT_EINSUM_ENABLED)
endif()

if (NOT "${USE_CUSPARSELT}" STREQUAL "OFF")
  target_compile_definitions(yirage_runtime PUBLIC YIRAGE_BACKEND_CUSPARSELT_ENABLED)
endif()

if (NOT "${USE_FORMAL_VERIFIER}" STREQUAL "OFF")
  target_compile_definitions(yirage_runtime PUBLIC YIRAGE_USE_FORMAL_VERIFIER)
endif()

target_link_libraries(yirage_runtime PUBLIC ${YIRAGE_LINK_LIBS})

target_include_directories(yirage_runtime
  PUBLIC ${PROJECT_SOURCE_DIR}/include)

install(TARGETS yirage_runtime
    LIBRARY DESTINATION lib)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include
    DESTINATION .)

install(FILES ${ABSTRACT_SUBEXPR_LIBRARIES} 
    DESTINATION lib)

if ("${BUILD_CPP_EXAMPLES}" STREQUAL "ON")
  set(CPP_EXAMPLES_DIR cpp_examples)
  set(CPP_EXAMPLES_BIN dnn)
  file(GLOB CPP_EXAMPLES_SRCS ${CPP_EXAMPLES_DIR}/${CPP_EXAMPLES_BIN}.cc)
  add_executable(${CPP_EXAMPLES_BIN} ${CPP_EXAMPLES_SRCS})
  set_target_properties(${CPP_EXAMPLES_BIN}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CPP_EXAMPLES_DIR})
  target_link_libraries(${CPP_EXAMPLES_BIN} yirage_runtime)
endif()

# Multi-Query Attention Incremental Decoding
if ("${BUILD_CPP_EXAMPLES}" STREQUAL "ON")
  set(CPP_EXAMPLES_DIR cpp_examples)
  set(CPP_EXAMPLES_BIN multi_query_attn_inc_decode)
  file(GLOB CPP_EXAMPLES_SRCS ${CPP_EXAMPLES_DIR}/${CPP_EXAMPLES_BIN}.cc)
  add_executable(${CPP_EXAMPLES_BIN} ${CPP_EXAMPLES_SRCS})
  set_target_properties(${CPP_EXAMPLES_BIN}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CPP_EXAMPLES_DIR})
  target_link_libraries(${CPP_EXAMPLES_BIN} yirage_runtime)
endif()

# Multi-Query Attention Speculative Decoding
if ("${BUILD_CPP_EXAMPLES}" STREQUAL "ON")
  set(CPP_EXAMPLES_DIR cpp_examples)
  set(CPP_EXAMPLES_BIN multi_query_attn_spec_decode)
  file(GLOB CPP_EXAMPLES_SRCS ${CPP_EXAMPLES_DIR}/${CPP_EXAMPLES_BIN}.cc)
  add_executable(${CPP_EXAMPLES_BIN} ${CPP_EXAMPLES_SRCS})
  set_target_properties(${CPP_EXAMPLES_BIN}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CPP_EXAMPLES_DIR})
  target_link_libraries(${CPP_EXAMPLES_BIN} yirage_runtime)
endif()

# Multi-Query Attention Prefilling
if ("${BUILD_CPP_EXAMPLES}" STREQUAL "ON")
  set(CPP_EXAMPLES_DIR cpp_examples)
  set(CPP_EXAMPLES_BIN multi_query_attn_prefill)
  file(GLOB CPP_EXAMPLES_SRCS ${CPP_EXAMPLES_DIR}/${CPP_EXAMPLES_BIN}.cc)
  add_executable(${CPP_EXAMPLES_BIN} ${CPP_EXAMPLES_SRCS})
  set_target_properties(${CPP_EXAMPLES_BIN}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CPP_EXAMPLES_DIR})
  target_link_libraries(${CPP_EXAMPLES_BIN} yirage_runtime)
endif()

# Group-Query Attention Incremental Decoding
if ("${BUILD_CPP_EXAMPLES}" STREQUAL "ON")
  set(CPP_EXAMPLES_DIR cpp_examples)
  set(CPP_EXAMPLES_BIN group_query_attn_inc_decode)
  file(GLOB CPP_EXAMPLES_SRCS ${CPP_EXAMPLES_DIR}/${CPP_EXAMPLES_BIN}.cc)
  add_executable(${CPP_EXAMPLES_BIN} ${CPP_EXAMPLES_SRCS})
  set_target_properties(${CPP_EXAMPLES_BIN}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CPP_EXAMPLES_DIR})
  target_link_libraries(${CPP_EXAMPLES_BIN} yirage_runtime)
endif()

# Group-Query Attention Spec Decoding
if ("${BUILD_CPP_EXAMPLES}" STREQUAL "ON")
  set(CPP_EXAMPLES_DIR cpp_examples)
  set(CPP_EXAMPLES_BIN group_query_attn_spec_decode)
  file(GLOB CPP_EXAMPLES_SRCS ${CPP_EXAMPLES_DIR}/${CPP_EXAMPLES_BIN}.cc)
  add_executable(${CPP_EXAMPLES_BIN} ${CPP_EXAMPLES_SRCS})
  set_target_properties(${CPP_EXAMPLES_BIN}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CPP_EXAMPLES_DIR})
  target_link_libraries(${CPP_EXAMPLES_BIN} yirage_runtime)
endif()

# Group-Query Attention Prefilling
if ("${BUILD_CPP_EXAMPLES}" STREQUAL "ON")
  set(CPP_EXAMPLES_DIR cpp_examples)
  set(CPP_EXAMPLES_BIN group_query_attn_prefill)
  file(GLOB CPP_EXAMPLES_SRCS ${CPP_EXAMPLES_DIR}/${CPP_EXAMPLES_BIN}.cc)
  add_executable(${CPP_EXAMPLES_BIN} ${CPP_EXAMPLES_SRCS})
  set_target_properties(${CPP_EXAMPLES_BIN}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CPP_EXAMPLES_DIR})
  target_link_libraries(${CPP_EXAMPLES_BIN} yirage_runtime)
endif()

# Multi-Head Attention Incremental Decoding
if ("${BUILD_CPP_EXAMPLES}" STREQUAL "ON")
  set(CPP_EXAMPLES_DIR cpp_examples)
  set(CPP_EXAMPLES_BIN multi_head_attn_inc_decode)
  file(GLOB CPP_EXAMPLES_SRCS ${CPP_EXAMPLES_DIR}/${CPP_EXAMPLES_BIN}.cc)
  add_executable(${CPP_EXAMPLES_BIN} ${CPP_EXAMPLES_SRCS})
  set_target_properties(${CPP_EXAMPLES_BIN}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CPP_EXAMPLES_DIR})
  target_link_libraries(${CPP_EXAMPLES_BIN} yirage_runtime)
endif()

# Multi-Head Attention Speculative Decoding
if ("${BUILD_CPP_EXAMPLES}" STREQUAL "ON")
  set(CPP_EXAMPLES_DIR cpp_examples)
  set(CPP_EXAMPLES_BIN multi_head_attn_spec_decode)
  file(GLOB CPP_EXAMPLES_SRCS ${CPP_EXAMPLES_DIR}/${CPP_EXAMPLES_BIN}.cc)
  add_executable(${CPP_EXAMPLES_BIN} ${CPP_EXAMPLES_SRCS})
  set_target_properties(${CPP_EXAMPLES_BIN}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CPP_EXAMPLES_DIR})
  target_link_libraries(${CPP_EXAMPLES_BIN} yirage_runtime)
endif()

# Multi-Head Attention Prefilling
if ("${BUILD_CPP_EXAMPLES}" STREQUAL "ON")
  set(CPP_EXAMPLES_DIR cpp_examples)
  set(CPP_EXAMPLES_BIN multi_head_attn_prefill)
  file(GLOB CPP_EXAMPLES_SRCS ${CPP_EXAMPLES_DIR}/${CPP_EXAMPLES_BIN}.cc)
  add_executable(${CPP_EXAMPLES_BIN} ${CPP_EXAMPLES_SRCS})
  set_target_properties(${CPP_EXAMPLES_BIN}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CPP_EXAMPLES_DIR})
  target_link_libraries(${CPP_EXAMPLES_BIN} yirage_runtime)
endif()

# LoRA
if ("${BUILD_CPP_EXAMPLES}" STREQUAL "ON")
  set(CPP_EXAMPLES_DIR cpp_examples)
  set(CPP_EXAMPLES_BIN lora)
  file(GLOB CPP_EXAMPLES_SRCS ${CPP_EXAMPLES_DIR}/${CPP_EXAMPLES_BIN}.cc)
  add_executable(${CPP_EXAMPLES_BIN} ${CPP_EXAMPLES_SRCS})
  set_target_properties(${CPP_EXAMPLES_BIN}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CPP_EXAMPLES_DIR})
  target_link_libraries(${CPP_EXAMPLES_BIN} yirage_runtime)
endif()

# MLP
if ("${BUILD_CPP_EXAMPLES}" STREQUAL "ON")
  set(CPP_EXAMPLES_DIR cpp_examples)
  set(CPP_EXAMPLES_BIN mlp)
  file(GLOB CPP_EXAMPLES_SRCS ${CPP_EXAMPLES_DIR}/${CPP_EXAMPLES_BIN}.cc)
  add_executable(${CPP_EXAMPLES_BIN} ${CPP_EXAMPLES_SRCS})
  set_target_properties(${CPP_EXAMPLES_BIN}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CPP_EXAMPLES_DIR})
  target_link_libraries(${CPP_EXAMPLES_BIN} yirage_runtime)
endif()

# MoE
if ("${BUILD_CPP_EXAMPLES}" STREQUAL "ON")
  set(CPP_EXAMPLES_DIR cpp_examples)
  set(CPP_EXAMPLES_BIN moe)
  file(GLOB CPP_EXAMPLES_SRCS ${CPP_EXAMPLES_DIR}/${CPP_EXAMPLES_BIN}.cc)
  add_executable(${CPP_EXAMPLES_BIN} ${CPP_EXAMPLES_SRCS})
  set_target_properties(${CPP_EXAMPLES_BIN}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CPP_EXAMPLES_DIR})
  target_link_libraries(${CPP_EXAMPLES_BIN} yirage_runtime)
endif()

# RMS
if ("${BUILD_CPP_EXAMPLES}" STREQUAL "ON")
  set(CPP_EXAMPLES_DIR cpp_examples)
  set(CPP_EXAMPLES_BIN rms)
  file(GLOB CPP_EXAMPLES_SRCS ${CPP_EXAMPLES_DIR}/${CPP_EXAMPLES_BIN}.cc)
  add_executable(${CPP_EXAMPLES_BIN} ${CPP_EXAMPLES_SRCS})
  set_target_properties(${CPP_EXAMPLES_BIN}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CPP_EXAMPLES_DIR})
  target_link_libraries(${CPP_EXAMPLES_BIN} yirage_runtime)
endif()

# GatedMLP
if ("${BUILD_CPP_EXAMPLES}" STREQUAL "ON")
  set(CPP_EXAMPLES_DIR cpp_examples)
  set(CPP_EXAMPLES_BIN gated_mlp)
  file(GLOB CPP_EXAMPLES_SRCS ${CPP_EXAMPLES_DIR}/${CPP_EXAMPLES_BIN}.cc)
  add_executable(${CPP_EXAMPLES_BIN} ${CPP_EXAMPLES_SRCS})
  set_target_properties(${CPP_EXAMPLES_BIN}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CPP_EXAMPLES_DIR})
  target_link_libraries(${CPP_EXAMPLES_BIN} yirage_runtime)
endif()

# Chameleon
if ("${BUILD_CPP_EXAMPLES}" STREQUAL "ON")
  set(CPP_EXAMPLES_DIR cpp_examples)
  set(CPP_EXAMPLES_BIN chameleon)
  file(GLOB CPP_EXAMPLES_SRCS ${CPP_EXAMPLES_DIR}/${CPP_EXAMPLES_BIN}.cc)
  add_executable(${CPP_EXAMPLES_BIN} ${CPP_EXAMPLES_SRCS})
  set_target_properties(${CPP_EXAMPLES_BIN}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CPP_EXAMPLES_DIR})
  target_link_libraries(${CPP_EXAMPLES_BIN} yirage_runtime)
endif()

# egg_tests
if("${BUILD_CPP_EXAMPLES}" STREQUAL "ON")
  set(CPP_EXAMPLES_DIR cpp_examples)
  set(CPP_EXAMPLES_BIN egg_tests)
  file(GLOB CPP_EXAMPLES_SRCS ${CPP_EXAMPLES_DIR}/${CPP_EXAMPLES_BIN}.cc)
  add_executable(${CPP_EXAMPLES_BIN} ${CPP_EXAMPLES_SRCS})
  set_target_properties(${CPP_EXAMPLES_BIN}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CPP_EXAMPLES_DIR})
  target_link_libraries(${CPP_EXAMPLES_BIN} yirage_runtime)
endif()

option(YIRAGE_BUILD_UNIT_TEST "build unit tests" OFF)

if (YIRAGE_BUILD_UNIT_TEST)
  enable_testing()
  add_subdirectory(tests)
  if(CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
    target_link_libraries(yirage_runtime PUBLIC stdc++fs)
  endif()
endif()
