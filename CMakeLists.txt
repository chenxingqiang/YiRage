cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
message(STATUS "CMake Version: ${CMAKE_VERSION}")

# =============================================================================
# Auto-detect Backend Environment Variables
# =============================================================================
# This section automatically detects and configures paths for various backends
# if they are not explicitly set via environment variables or config.cmake

# CUDA Auto-detection
if(NOT DEFINED ENV{CUDA_HOME})
  foreach(CUDA_PATH "/usr/local/cuda" "/opt/cuda" "/usr/lib/cuda")
    if(EXISTS "${CUDA_PATH}/bin/nvcc")
      set(ENV{CUDA_HOME} "${CUDA_PATH}")
      message(STATUS "Auto-detected CUDA_HOME: ${CUDA_PATH}")
      break()
    endif()
  endforeach()
endif()

# Ascend Auto-detection
if(NOT DEFINED ENV{ASCEND_HOME} AND NOT DEFINED ENV{ASCEND_HOME_PATH})
  foreach(ASCEND_PATH 
      "/usr/local/Ascend/ascend-toolkit/latest"
      "/usr/local/Ascend/nnrt/latest"
      "/opt/Ascend/ascend-toolkit/latest")
    if(EXISTS "${ASCEND_PATH}")
      set(ENV{ASCEND_HOME} "${ASCEND_PATH}")
      set(ENV{ASCEND_HOME_PATH} "${ASCEND_PATH}")
      message(STATUS "Auto-detected ASCEND_HOME: ${ASCEND_PATH}")
      break()
    endif()
  endforeach()
endif()

# ROCm Auto-detection
if(NOT DEFINED ENV{ROCM_PATH})
  foreach(ROCM_PATH "/opt/rocm" "/opt/rocm-6.0.0" "/opt/rocm-5.7.0")
    if(EXISTS "${ROCM_PATH}")
      set(ENV{ROCM_PATH} "${ROCM_PATH}")
      message(STATUS "Auto-detected ROCM_PATH: ${ROCM_PATH}")
      break()
    endif()
  endforeach()
endif()

# MACA Auto-detection
if(NOT DEFINED ENV{MACA_PATH})
  foreach(MACA_PATH "/opt/maca" "/usr/local/maca")
    if(EXISTS "${MACA_PATH}")
      set(ENV{MACA_PATH} "${MACA_PATH}")
      message(STATUS "Auto-detected MACA_PATH: ${MACA_PATH}")
      break()
    endif()
  endforeach()
endif()

# Intel oneAPI Auto-detection
if(NOT DEFINED ENV{ONEAPI_ROOT})
  foreach(ONEAPI_PATH "/opt/intel/oneapi" "/usr/local/intel/oneapi")
    if(EXISTS "${ONEAPI_PATH}")
      set(ENV{ONEAPI_ROOT} "${ONEAPI_PATH}")
      message(STATUS "Auto-detected ONEAPI_ROOT: ${ONEAPI_PATH}")
      break()
    endif()
  endforeach()
endif()

# =============================================================================
# Load Config
# =============================================================================

if (EXISTS ${CMAKE_CURRENT_BINARY_DIR}/config.cmake)
  include (${CMAKE_CURRENT_BINARY_DIR}/config.cmake)
else()
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/config.cmake)
    include(${CMAKE_CURRENT_SOURCE_DIR}/config.cmake)
  endif()
endif()

# Determine which languages to enable based on backends
set(PROJECT_LANGUAGES C CXX)
if ("${USE_CUDA}" STREQUAL "ON")
  list(APPEND PROJECT_LANGUAGES CUDA)
  message(STATUS "CUDA backend enabled")
endif()
# Note: MACA does NOT use CMake's CUDA language - it uses mxcc compiler directly
if ("${USE_MACA}" STREQUAL "ON")
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

# Exclude problematic persistent kernel backends (incomplete implementations)
# These will be re-enabled as they are properly implemented
file(GLOB PK_EXCLUDE_SRCS
  src/persistent_kernel/rocm_pk_backend.cc
  src/persistent_kernel/fpga_pk_backend.cc
  src/persistent_kernel/tpu_pk_backend.cc
  src/persistent_kernel/xpu_pk_backend.cc
  src/persistent_kernel/nki_pk_backend.cc
  src/persistent_kernel/triton_pk_backend.cc
  src/persistent_kernel/mlir_pk_backend.cc
  src/persistent_kernel/maca_pk_backend.cc
)
if(PK_EXCLUDE_SRCS)
  list(REMOVE_ITEM YIRAGE_SRCS ${PK_EXCLUDE_SRCS})
  message(STATUS "Excluding incomplete PK backends: ${PK_EXCLUDE_SRCS}")
endif()

# Exclude vendor-specific threadblock files (incomplete implementations)
# These require vendor-specific compilers or have incomplete interfaces
file(GLOB TB_EXCLUDE_SRCS
  src/threadblock/ascend/*.cc
  src/threadblock/maca/*.cc
  src/threadblock/fpga/*.cc
  src/threadblock/rocm/*.cc
  src/threadblock/xpu/*.cc
  src/threadblock/tpu/*.cc
  src/threadblock/nki/*.cc
  src/threadblock/triton/*.cc
)
if(TB_EXCLUDE_SRCS)
  list(REMOVE_ITEM YIRAGE_SRCS ${TB_EXCLUDE_SRCS})
  message(STATUS "Excluding vendor-specific threadblock implementations")
endif()

# Exclude duplicate search partition file (already defined in partitioned_generator.cc)
list(REMOVE_ITEM YIRAGE_SRCS ${CMAKE_SOURCE_DIR}/src/search/distributed/search_partition.cc)
message(STATUS "Excluding duplicate search_partition.cc")

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
  src/kernel/rocm/*.cc
  src/kernel/tpu/*.cc
  src/kernel/fpga/*.cc
  src/kernel/xpu/*.cc
  # Note: Backend-specific GPU kernels (.maca, .hip, .sycl, etc.) require
  # their respective compilers and are compiled separately.
)
list(APPEND YIRAGE_SRCS ${KERNEL_OPT_SRCS})

# Collect threadblock sources (host-side, compiled with standard compiler)
file(GLOB_RECURSE THREADBLOCK_SRCS
  src/threadblock/common/*.cc
  src/threadblock/cpu/*.cc
  src/threadblock/tpu/*.cc
  src/threadblock/fpga/*.cc
  src/threadblock/xpu/*.cc
)
list(APPEND YIRAGE_SRCS ${THREADBLOCK_SRCS})

# Collect search strategy sources
file(GLOB SEARCH_COMMON_SRCS src/search/common/*.cc)
file(GLOB SEARCH_STRATEGY_SRCS src/search/backend_strategies/*.cc)
file(GLOB SEARCH_PROFILER_SRCS src/search/mps_profiler.cc)
list(APPEND YIRAGE_SRCS ${SEARCH_COMMON_SRCS})
list(APPEND YIRAGE_SRCS ${SEARCH_STRATEGY_SRCS})
list(APPEND YIRAGE_SRCS ${SEARCH_PROFILER_SRCS})

# Handle CUDA-specific sources
if ("${USE_CUDA}" STREQUAL "ON")
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
  
  # Keep ascend_transpiler.cc for Ascend backend
  if (NOT "${USE_ASCEND}" STREQUAL "OFF")
    list(APPEND YIRAGE_SRCS src/transpiler/ascend_transpiler.cc)
    message(STATUS "Including Ascend transpiler")
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

if ("${USE_MACA}" STREQUAL "ON")
  message(STATUS "MACA backend enabled (requires MetaX MACA SDK)")
  include(${CMAKE_CURRENT_SOURCE_DIR}/config.maca.cmake)
  if(MACA_COMPILE_KERNELS)
    list(REMOVE_ITEM YIRAGE_SRCS ${CMAKE_SOURCE_DIR}/src/kernel/device_memory_manager.cc)
  endif()
elseif (NOT "${USE_ASCEND}" STREQUAL "OFF")
  # Ascend implementation is src/kernel/ascend/device_memory_manager.cc (already in KERNEL_OPT_SRCS).
  # Omit generic CPU stub .cc before add_library (post-add_library list edits do not change the target).
  list(REMOVE_ITEM YIRAGE_SRCS ${CMAKE_SOURCE_DIR}/src/kernel/device_memory_manager.cc)
endif()

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++17" SUPPORT_CXX17)

#set CUDA version
if ("${USE_CUDA}" STREQUAL "ON")
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
if ("${USE_CUDA}" STREQUAL "ON")
  include(cmake/find_cuda.cmake)
  find_cuda(${USE_CUDA})
  if (CUDA_FOUND)
    # .cu sources already appended when USE_CUDA=ON (see block above)
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
if ("${USE_CUDA}" STREQUAL "ON")
  target_compile_definitions(yirage_runtime PUBLIC YIRAGE_BACKEND_CUDA_ENABLED)
  target_compile_definitions(yirage_runtime PUBLIC YIRAGE_BACKEND_USE_CUDA)  # Backward compatibility
  target_compile_definitions(yirage_runtime PUBLIC YIRAGE_FINGERPRINT_USE_CUDA)
  
  # Enable OpenMP for parallel search with CUDA
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(yirage_runtime PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(yirage_runtime PUBLIC YIRAGE_ENABLE_PARALLEL_SEARCH)
    message(STATUS "OpenMP enabled for CUDA parallel search (YIRAGE_ENABLE_PARALLEL_SEARCH defined)")
  endif()
elseif ("${USE_MACA}" STREQUAL "ON")
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
    target_compile_definitions(yirage_runtime PUBLIC YIRAGE_ENABLE_PARALLEL_SEARCH)
    message(STATUS "OpenMP enabled for CPU parallel search (YIRAGE_ENABLE_PARALLEL_SEARCH defined)")
  else()
    message(WARNING "OpenMP not found - search will be single-threaded")
  endif()
endif()

if (NOT "${USE_CPU}" STREQUAL "OFF")
  target_compile_definitions(yirage_runtime PUBLIC YIRAGE_BACKEND_CPU_ENABLED)
  set(BLA_VENDOR OpenBLAS)
  find_package(BLAS)
  if(NOT BLAS_FOUND)
    unset(BLA_VENDOR)
    find_package(BLAS)
  endif()
  if(BLAS_FOUND)
    target_link_libraries(yirage_runtime PUBLIC ${BLAS_LIBRARIES})
    if(BLAS_INCLUDE_DIRS)
      target_include_directories(yirage_runtime PRIVATE ${BLAS_INCLUDE_DIRS})
    endif()
    target_compile_definitions(yirage_runtime PUBLIC YIRAGE_CPU_BLAS_ENABLED)
    if("${BLAS_LIBRARIES}" MATCHES "[Oo]pen[Bb]las" OR BLA_VENDOR STREQUAL "OpenBLAS")
      target_compile_definitions(yirage_runtime PRIVATE YIRAGE_OPENBLAS_BLAS)
    endif()
    message(STATUS "CPU BLAS enabled for fused rms_matmul (cblas_sgemm)")
  else()
    message(STATUS "BLAS not found — fused rms_matmul uses internal SIMD GEMM")
  endif()
endif()

if (NOT "${USE_MPS}" STREQUAL "OFF")
  target_compile_definitions(yirage_runtime PUBLIC YIRAGE_BACKEND_MPS_ENABLED)
  
  # Add Metal wrapper source (Objective-C++)
  if(APPLE)
    enable_language(OBJCXX)
    target_sources(yirage_runtime PRIVATE src/persistent_kernel/mtl_wrapper.mm)
    # Enable ARC for Metal wrapper
    set_source_files_properties(src/persistent_kernel/mtl_wrapper.mm PROPERTIES
      COMPILE_FLAGS "-fobjc-arc"
    )
    target_link_libraries(yirage_runtime PUBLIC
      "-framework Metal"
      "-framework Foundation"
    )
    message(STATUS "Metal wrapper enabled for MPS backend")
  endif()
  
  # Enable OpenMP for parallel search with MPS (Apple Metal)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(yirage_runtime PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(yirage_runtime PUBLIC YIRAGE_ENABLE_PARALLEL_SEARCH)
    message(STATUS "OpenMP enabled for MPS parallel search (YIRAGE_ENABLE_PARALLEL_SEARCH defined)")
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
  # Enable Ascend ACL runtime detection
  target_compile_definitions(yirage_runtime PUBLIC __ASCEND__)
  # Ascend NPU fingerprint (enables parallel search)
  target_compile_definitions(yirage_runtime PUBLIC YIRAGE_FINGERPRINT_USE_ASCEND)
  
  # Enable OpenMP for parallel search with Ascend
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(yirage_runtime PUBLIC OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP enabled for Ascend parallel search")
  endif()
  
  # Set CANN/Ascend paths
  if(DEFINED ENV{ASCEND_HOME})
    set(ASCEND_SDK_PATH $ENV{ASCEND_HOME})
  elseif(DEFINED ENV{CANN_HOME})
    set(ASCEND_SDK_PATH $ENV{CANN_HOME})
  else()
    set(ASCEND_SDK_PATH "/usr/local/Ascend/ascend-toolkit/latest")
  endif()
  
  # Add Ascend SDK include directories
  target_include_directories(yirage_runtime PUBLIC 
    ${ASCEND_SDK_PATH}/include
    ${ASCEND_SDK_PATH}/include/acl
  )
  target_link_directories(yirage_runtime PUBLIC ${ASCEND_SDK_PATH}/lib64)
  
  # Find AscendC compiler (ascendc or bisheng)
  find_program(ASCENDC_COMPILER ascendc
    PATHS 
      ${ASCEND_SDK_PATH}/compiler/bin
      ${ASCEND_SDK_PATH}/bin
      /usr/local/Ascend/ascend-toolkit/latest/compiler/bin
  )
  
  if(NOT ASCENDC_COMPILER)
    # Try bisheng compiler as fallback
    find_program(ASCENDC_COMPILER bisheng
      PATHS 
        ${ASCEND_SDK_PATH}/compiler/bin
        /usr/local/Ascend/ascend-toolkit/latest/compiler/bin
    )
  endif()
  
  if(ASCENDC_COMPILER AND ASCEND_COMPILE_KERNELS)
    message(STATUS "Found Ascend compiler: ${ASCENDC_COMPILER}")
    message(STATUS "Ascend kernel compilation enabled (ASCEND_COMPILE_KERNELS=ON)")
    
    # Collect Ascend NPU kernel source files (.ascend files)
    # Note: .cc files in ascend/ are Host-side code, compiled with g++
    # .ascend files are Device-side NPU kernels, compiled with bisheng
    file(GLOB ASCEND_GPU_KERNEL_SRCS src/kernel/ascend/*.ascend)
    
    # Compile each Ascend NPU kernel file
    set(ASCEND_KERNEL_OBJS "")
    foreach(ASCEND_SRC ${ASCEND_GPU_KERNEL_SRCS})
      get_filename_component(ASCEND_SRC_NAME ${ASCEND_SRC} NAME_WE)
      set(ASCEND_OBJ ${CMAKE_CURRENT_BINARY_DIR}/ascend_kernels/${ASCEND_SRC_NAME}.o)
      
      add_custom_command(
        OUTPUT ${ASCEND_OBJ}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/ascend_kernels
        COMMAND ${ASCENDC_COMPILER}
          --cce-aicore-lang
          -c ${ASCEND_SRC}
          -o ${ASCEND_OBJ}
          -I${PROJECT_SOURCE_DIR}/include
          -I${ASCEND_SDK_PATH}/include
          -I${ASCEND_SDK_PATH}/include/acl
          -I${PROJECT_SOURCE_DIR}/deps/cutlass/include
          -I${PROJECT_SOURCE_DIR}/deps/json/include
          -DYIRAGE_BACKEND_ASCEND_ENABLED
          -DYIRAGE_FINGERPRINT_USE_ASCEND
          -D__ASCEND__
          -std=c++17
          -fPIC
          -O2
        DEPENDS ${ASCEND_SRC}
        COMMENT "Compiling Ascend NPU kernel: ${ASCEND_SRC_NAME}"
      )
      list(APPEND ASCEND_KERNEL_OBJS ${ASCEND_OBJ})
    endforeach()
    
    # Create a custom target for Ascend kernels
    if(ASCEND_KERNEL_OBJS)
      add_custom_target(ascend_kernels DEPENDS ${ASCEND_KERNEL_OBJS})
      add_dependencies(yirage_runtime ascend_kernels)
      
      # Link Ascend kernel objects to the runtime library  
      target_sources(yirage_runtime PRIVATE ${ASCEND_KERNEL_OBJS})
      message(STATUS "Ascend NPU kernels: ${ASCEND_GPU_KERNEL_SRCS}")
    else()
      message(STATUS "No Ascend NPU kernel files (.ascend) found")
    endif()
    
    # Link Ascend runtime libraries
    list(APPEND YIRAGE_LINK_LIBS ascendcl acl_op_compiler)
  else()
    if(ASCENDC_COMPILER)
      message(STATUS "Found Ascend compiler: ${ASCENDC_COMPILER}")
      message(STATUS "Ascend kernel compilation disabled (set ASCEND_COMPILE_KERNELS=ON to enable)")
    else()
      message(STATUS "Ascend compiler not found")
    endif()
    message(STATUS "Using triton-ascend for runtime kernel generation")
    # Note: We still use YIRAGE_FINGERPRINT_USE_ASCEND for the host-side memory manager
    # Kernels are generated at runtime via triton-ascend
  endif()
endif()

# MACA Backend (MetaX GPU)
if ("${USE_MACA}" STREQUAL "ON")
  # Include MACA configuration
  include(${CMAKE_CURRENT_SOURCE_DIR}/config.maca.cmake)
  target_compile_definitions(yirage_runtime PUBLIC YIRAGE_BACKEND_MACA_ENABLED)
  
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

    if(MACA_COMPILE_KERNELS)
      target_compile_definitions(yirage_runtime PUBLIC YIRAGE_FINGERPRINT_USE_MACA)
      target_compile_definitions(yirage_runtime PUBLIC YIRAGE_PK_MACA_BACKEND_ENABLED)

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
      add_custom_target(maca_kernels ALL DEPENDS ${MACA_KERNEL_OBJS})
      add_dependencies(yirage_runtime maca_kernels)
      
      # Link MACA kernel objects by adding them to link libraries
      # Using OBJECT_LIBRARY approach for proper linking
      set_source_files_properties(${MACA_KERNEL_OBJS} PROPERTIES 
        EXTERNAL_OBJECT true
        GENERATED true
      )
      target_link_libraries(yirage_runtime PUBLIC ${MACA_KERNEL_OBJS})
      message(STATUS "MACA GPU kernels: ${MACA_GPU_KERNEL_SRCS}")
    else()
      message(STATUS "No MACA GPU kernel files (.maca) found")
    endif()
    else()
      message(STATUS "MACA native .maca kernels skipped (MACA_COMPILE_KERNELS=OFF)")
      target_compile_definitions(yirage_runtime PUBLIC YIRAGE_FINGERPRINT_USE_CPU)
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

if (NOT "${USE_CUTLASS}" STREQUAL "OFF")
  target_compile_definitions(yirage_runtime PUBLIC YIRAGE_BACKEND_CUTLASS_ENABLED)
endif()

# =============================================================================
# ROCm Backend (AMD GPU)
# =============================================================================
if ("${USE_ROCM}" STREQUAL "ON")
  target_compile_definitions(yirage_runtime PUBLIC YIRAGE_BACKEND_ROCM_ENABLED)
  target_compile_definitions(yirage_runtime PUBLIC YIRAGE_FINGERPRINT_USE_ROCM)
  
  # Set ROCm paths
  if(DEFINED ENV{ROCM_PATH})
    set(ROCM_PATH $ENV{ROCM_PATH})
  else()
    set(ROCM_PATH "/opt/rocm")
  endif()
  
  target_include_directories(yirage_runtime PUBLIC ${ROCM_PATH}/include)
  target_link_directories(yirage_runtime PUBLIC ${ROCM_PATH}/lib)
  
  # Find hipcc compiler
  find_program(HIPCC_COMPILER hipcc PATHS ${ROCM_PATH}/bin)
  
  if(HIPCC_COMPILER)
    message(STATUS "Found HIP compiler: ${HIPCC_COMPILER}")
    
    # Collect ROCm HIP kernel sources
    file(GLOB ROCM_KERNEL_SRCS src/kernel/rocm/*.hip src/threadblock/rocm/*.hip)
    
    set(ROCM_KERNEL_OBJS "")
    foreach(HIP_SRC ${ROCM_KERNEL_SRCS})
      get_filename_component(HIP_SRC_NAME ${HIP_SRC} NAME_WE)
      set(HIP_OBJ ${CMAKE_CURRENT_BINARY_DIR}/rocm_kernels/${HIP_SRC_NAME}.o)
      
      add_custom_command(
        OUTPUT ${HIP_OBJ}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/rocm_kernels
        COMMAND ${HIPCC_COMPILER}
          -c ${HIP_SRC}
          -o ${HIP_OBJ}
          -I${PROJECT_SOURCE_DIR}/include
          -I${ROCM_PATH}/include
          -I${PROJECT_SOURCE_DIR}/deps/cutlass/include
          -DYIRAGE_BACKEND_ROCM_ENABLED
          -std=c++17
          -fPIC
          -O2
        DEPENDS ${HIP_SRC}
        COMMENT "Compiling ROCm HIP kernel: ${HIP_SRC_NAME}"
      )
      list(APPEND ROCM_KERNEL_OBJS ${HIP_OBJ})
    endforeach()
    
    if(ROCM_KERNEL_OBJS)
      add_custom_target(rocm_kernels ALL DEPENDS ${ROCM_KERNEL_OBJS})
      add_dependencies(yirage_runtime rocm_kernels)
      target_link_libraries(yirage_runtime PUBLIC ${ROCM_KERNEL_OBJS})
      message(STATUS "ROCm HIP kernels: ${ROCM_KERNEL_SRCS}")
    endif()
    
    list(APPEND YIRAGE_LINK_LIBS amdhip64 rocblas)
  else()
    message(WARNING "HIP compiler not found")
  endif()
  
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(yirage_runtime PUBLIC OpenMP::OpenMP_CXX)
  endif()
endif()

# =============================================================================
# TPU Backend (Google TPU via JAX)
# =============================================================================
if ("${USE_TPU}" STREQUAL "ON")
  target_compile_definitions(yirage_runtime PUBLIC YIRAGE_BACKEND_TPU_ENABLED)
  message(STATUS "TPU backend enabled (requires JAX/Pallas at runtime)")
  
  # TPU kernels are Python/JAX-based, compiled at runtime
  # Only enable the C++ search strategy and type definitions
  
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(yirage_runtime PUBLIC OpenMP::OpenMP_CXX)
  endif()
endif()

# =============================================================================
# FPGA Backend (Xilinx/Intel via HLS)
# =============================================================================
if ("${USE_FPGA}" STREQUAL "ON")
  target_compile_definitions(yirage_runtime PUBLIC YIRAGE_BACKEND_FPGA_ENABLED)
  message(STATUS "FPGA backend enabled (requires Vitis HLS or Intel HLS at build time)")
  
  # Check for Xilinx Vitis
  if(DEFINED ENV{XILINX_VITIS})
    set(VITIS_PATH $ENV{XILINX_VITIS})
    target_include_directories(yirage_runtime PUBLIC ${VITIS_PATH}/include)
    message(STATUS "Found Xilinx Vitis: ${VITIS_PATH}")
  endif()
  
  # Check for Intel FPGA
  if(DEFINED ENV{INTELFPGAOCLSDKROOT})
    set(INTEL_FPGA_PATH $ENV{INTELFPGAOCLSDKROOT})
    target_include_directories(yirage_runtime PUBLIC ${INTEL_FPGA_PATH}/include)
    message(STATUS "Found Intel FPGA SDK: ${INTEL_FPGA_PATH}")
  endif()
  
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(yirage_runtime PUBLIC OpenMP::OpenMP_CXX)
  endif()
endif()

# =============================================================================
# XPU Backend (Intel Data Center GPU via SYCL)
# =============================================================================
if ("${USE_XPU}" STREQUAL "ON")
  target_compile_definitions(yirage_runtime PUBLIC YIRAGE_BACKEND_XPU_ENABLED)
  message(STATUS "XPU backend enabled (requires Intel oneAPI at runtime)")
  
  # Check for oneAPI
  if(DEFINED ENV{ONEAPI_ROOT})
    set(ONEAPI_PATH $ENV{ONEAPI_ROOT})
  else()
    set(ONEAPI_PATH "/opt/intel/oneapi")
  endif()
  
  if(EXISTS ${ONEAPI_PATH})
    target_include_directories(yirage_runtime PUBLIC 
      ${ONEAPI_PATH}/compiler/latest/include
      ${ONEAPI_PATH}/compiler/latest/include/sycl
    )
    target_link_directories(yirage_runtime PUBLIC ${ONEAPI_PATH}/compiler/latest/lib)
    message(STATUS "Found Intel oneAPI: ${ONEAPI_PATH}")
    
    # Find icpx compiler for SYCL kernels
    find_program(ICPX_COMPILER icpx PATHS ${ONEAPI_PATH}/compiler/latest/bin)
    
    if(ICPX_COMPILER)
      message(STATUS "Found SYCL compiler: ${ICPX_COMPILER}")
      
      # Collect XPU SYCL kernel sources
      file(GLOB XPU_KERNEL_SRCS src/kernel/xpu/*.sycl src/threadblock/xpu/*.sycl)
      
      set(XPU_KERNEL_OBJS "")
      foreach(SYCL_SRC ${XPU_KERNEL_SRCS})
        get_filename_component(SYCL_SRC_NAME ${SYCL_SRC} NAME_WE)
        set(SYCL_OBJ ${CMAKE_CURRENT_BINARY_DIR}/xpu_kernels/${SYCL_SRC_NAME}.o)
        
        add_custom_command(
          OUTPUT ${SYCL_OBJ}
          COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/xpu_kernels
          COMMAND ${ICPX_COMPILER}
            -fsycl
            -c ${SYCL_SRC}
            -o ${SYCL_OBJ}
            -I${PROJECT_SOURCE_DIR}/include
            -I${ONEAPI_PATH}/compiler/latest/include
            -DYIRAGE_BACKEND_XPU_ENABLED
            -std=c++17
            -fPIC
            -O2
          DEPENDS ${SYCL_SRC}
          COMMENT "Compiling XPU SYCL kernel: ${SYCL_SRC_NAME}"
        )
        list(APPEND XPU_KERNEL_OBJS ${SYCL_OBJ})
      endforeach()
      
      if(XPU_KERNEL_OBJS)
        add_custom_target(xpu_kernels ALL DEPENDS ${XPU_KERNEL_OBJS})
        add_dependencies(yirage_runtime xpu_kernels)
        target_link_libraries(yirage_runtime PUBLIC ${XPU_KERNEL_OBJS})
        message(STATUS "XPU SYCL kernels: ${XPU_KERNEL_SRCS}")
      endif()
      
      list(APPEND YIRAGE_LINK_LIBS sycl)
    endif()
  endif()
  
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(yirage_runtime PUBLIC OpenMP::OpenMP_CXX)
  endif()
endif()

# =============================================================================
# MLIR Backend (Full LLVM/MLIR Integration)
# =============================================================================
if ("${USE_MLIR}" STREQUAL "ON")
  message(STATUS "============================================")
  message(STATUS "  MLIR Backend Configuration")
  message(STATUS "============================================")
  
  target_compile_definitions(yirage_runtime PUBLIC YIRAGE_BACKEND_MLIR_ENABLED)
  target_compile_definitions(yirage_runtime PUBLIC YIRAGE_MLIR_ENABLED)
  
  # Include LLVM/MLIR build configuration
  include(${CMAKE_SOURCE_DIR}/cmake/build_llvm.cmake)
  
  # Setup LLVM/MLIR (handles submodule, prebuilt, or system installation)
  setup_llvm_mlir()
  
  # Find MLIR package
  include(${CMAKE_SOURCE_DIR}/cmake/FindMLIR.cmake)
  
  if(MLIR_FOUND)
    message(STATUS "Found MLIR ${MLIR_VERSION}")
    message(STATUS "  MLIR_DIR: ${MLIR_DIR}")
    message(STATUS "  LLVM_DIR: ${LLVM_DIR}")
    
    # Include directories
    target_include_directories(yirage_runtime SYSTEM PUBLIC 
      ${MLIR_INCLUDE_DIRS}
      ${LLVM_INCLUDE_DIRS}
    )
    
    # Also include YiRage MLIR dialect headers
    target_include_directories(yirage_runtime PUBLIC
      ${CMAKE_SOURCE_DIR}/mlir/include
      ${CMAKE_BINARY_DIR}/mlir/include  # For generated headers
    )
    
    # Compile definitions
    target_compile_definitions(yirage_runtime PUBLIC ${LLVM_DEFINITIONS})
    
    # Link MLIR core libraries
    list(APPEND YIRAGE_LINK_LIBS 
      MLIRAnalysis
      MLIRIR
      MLIRParser
      MLIRPass
      MLIRTransforms
      MLIRSupport
    )
    
    # Link MLIR dialect libraries
    list(APPEND YIRAGE_LINK_LIBS
      MLIRArithDialect
      MLIRFuncDialect
      MLIRLinalgDialect
      MLIRTensorDialect
      MLIRMemRefDialect
      MLIRSCFDialect
      MLIRAffineDialect
      MLIRVectorDialect
    )
    
    # Link MLIR conversion libraries
    list(APPEND YIRAGE_LINK_LIBS
      MLIRArithToLLVM
      MLIRFuncToLLVM
      MLIRMemRefToLLVM
      MLIRSCFToControlFlow
      MLIRAffineToStandard
      MLIRLinalgTransforms
    )
    
    # GPU target support
    if(YIRAGE_MLIR_ENABLE_NVVM OR USE_CUDA)
      list(APPEND YIRAGE_LINK_LIBS MLIRNVVMDialect MLIRGPUToNVVMTransforms)
      message(STATUS "  MLIR NVVM target enabled")
    endif()
    
    if(YIRAGE_MLIR_ENABLE_ROCDL OR USE_ROCM)
      list(APPEND YIRAGE_LINK_LIBS MLIRROCDLDialect MLIRGPUToROCDLTransforms)
      message(STATUS "  MLIR ROCDL target enabled")
    endif()
    
    if(YIRAGE_MLIR_ENABLE_SPIRV OR USE_XPU)
      list(APPEND YIRAGE_LINK_LIBS MLIRSPIRVDialect MLIRGPUToSPIRVTransforms)
      message(STATUS "  MLIR SPIR-V target enabled")
    endif()
    
    # Execution engine for JIT
    list(APPEND YIRAGE_LINK_LIBS
      MLIRExecutionEngine
      MLIRTargetLLVMIRExport
    )
    
    # ==========================================================================
    # Build YiRage MLIR Dialect (from mlir/ subdirectory)
    # ==========================================================================
    if(YIRAGE_BUILD_MLIR_TOOLS OR EXISTS "${CMAKE_SOURCE_DIR}/mlir/CMakeLists.txt")
      message(STATUS "Building YiRage MLIR dialect...")
      
      # Set variables for the mlir subdirectory
      set(MLIR_DIR ${MLIR_DIR})
      set(LLVM_DIR ${LLVM_DIR})
      
      # Add the mlir subdirectory
      add_subdirectory(mlir)
      
      # Link YiRage dialect libraries
      if(TARGET YirageDialect)
        list(APPEND YIRAGE_LINK_LIBS YirageDialect)
        message(STATUS "  YirageDialect library linked")
      endif()
      
      if(TARGET YirageTransforms)
        list(APPEND YIRAGE_LINK_LIBS YirageTransforms)
        message(STATUS "  YirageTransforms library linked")
      endif()
    endif()
    
    message(STATUS "============================================")
  else()
    message(WARNING "MLIR not found - MLIR backend will use stub implementation")
    message(STATUS "  To enable full MLIR support:")
    message(STATUS "    1. Set MLIR_DIR=/path/to/mlir/lib/cmake/mlir")
    message(STATUS "    2. Or run: ./scripts/install_llvm.sh")
    message(STATUS "    3. Or use: cmake -DYIRAGE_LLVM_SOURCE=submodule ..")
  endif()
  
  # Enable OpenMP for parallel compilation
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(yirage_runtime PUBLIC OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP enabled for MLIR parallel search")
  endif()
endif()

# StableHLO/MHLO support (for TPU/XLA compatibility)
if ("${USE_STABLEHLO}" STREQUAL "ON")
  target_compile_definitions(yirage_runtime PUBLIC YIRAGE_BACKEND_STABLEHLO_ENABLED)
  message(STATUS "StableHLO backend enabled")
endif()

# TVM support
if ("${USE_TVM}" STREQUAL "ON")
  target_compile_definitions(yirage_runtime PUBLIC YIRAGE_BACKEND_TVM_ENABLED)
  message(STATUS "TVM backend enabled")
  
  if(DEFINED ENV{TVM_HOME})
    set(TVM_PATH $ENV{TVM_HOME})
    target_include_directories(yirage_runtime PUBLIC ${TVM_PATH}/include)
    target_link_directories(yirage_runtime PUBLIC ${TVM_PATH}/build)
    list(APPEND YIRAGE_LINK_LIBS tvm)
    message(STATUS "Found TVM: ${TVM_PATH}")
  endif()
endif()

# IREE support
if ("${USE_IREE}" STREQUAL "ON")
  target_compile_definitions(yirage_runtime PUBLIC YIRAGE_BACKEND_IREE_ENABLED)
  message(STATUS "IREE backend enabled")
  
  if(DEFINED ENV{IREE_DIR})
    set(IREE_PATH $ENV{IREE_DIR})
    target_include_directories(yirage_runtime PUBLIC ${IREE_PATH}/include)
    message(STATUS "Found IREE: ${IREE_PATH}")
  endif()
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
