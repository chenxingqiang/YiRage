#===----------------------------------------------------------------------===//
#
# YiRage MLIR Dialect Build Configuration
#
# This CMakeLists.txt builds the YiRage MLIR dialect and its transformations.
#
# Requirements:
#   - LLVM/MLIR 17+ (with development headers)
#   - CMake 3.20+
#
# Build Instructions:
#   mkdir build && cd build
#   cmake .. -DMLIR_DIR=/path/to/mlir/lib/cmake/mlir
#   make -j$(nproc)
#
#===----------------------------------------------------------------------===//

cmake_minimum_required(VERSION 3.20)
project(YirageMLIR VERSION 1.0.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#===----------------------------------------------------------------------===//
# MLIR/LLVM Configuration
#===----------------------------------------------------------------------===//

# Debian/Ubuntu MLIR is built against libLLVM-17.so; avoid mixing static LLVM .a.
if(NOT DEFINED LLVM_LINK_LLVM_DYLIB)
  set(LLVM_LINK_LLVM_DYLIB ON CACHE BOOL "Link against libLLVM.so" FORCE)
endif()

find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

# When LLVM is a shared library, link libMLIR.so instead of individual static
# libMLIR*.a archives. Mixing static MLIR with libLLVM.so causes tensor/yirage
# IR parser segfaults (duplicate MLIR globals + mismatched allocators).
set(_YIRAGE_USE_MLIR_DYLIB OFF)
if(LLVM_LINK_LLVM_DYLIB AND TARGET MLIR)
  set(_YIRAGE_USE_MLIR_DYLIB ON)
  message(STATUS "YiRage MLIR: using libMLIR.so (paired with LLVM dylib)")
endif()

if(_YIRAGE_USE_MLIR_DYLIB)
  set(_YIRAGE_MLIR_DIALECT_LINK_LIBS MLIR)
  set(_YIRAGE_MLIR_TRANSFORM_LINK_LIBS MLIR)
else()
  set(_YIRAGE_MLIR_DIALECT_LINK_LIBS
    MLIRIR MLIRDialect MLIRArithDialect MLIRFuncDialect MLIRTensorDialect
    MLIRMemRefDialect MLIRLinalgDialect MLIRSupport)
  set(_YIRAGE_MLIR_TRANSFORM_LINK_LIBS
    MLIRIR MLIRPass MLIRTransforms MLIRArithDialect MLIRMathDialect
    MLIRLinalgDialect MLIRTensorDialect MLIRMemRefDialect MLIRSCFDialect
    MLIRAffineDialect MLIRGPUDialect MLIRVectorDialect MLIRBufferizationDialect
    MLIRBufferizationTransforms MLIRLinalgTransforms MLIRSCFToControlFlow
    MLIRAffineToStandard MLIRArithToLLVM MLIRMemRefToLLVM MLIRFuncToLLVM MLIRAnalysis)
endif()

message(STATUS "Found MLIR ${MLIR_PACKAGE_VERSION}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# Set up LLVM/MLIR include and lib paths
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)

link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

# TableGen outputs (parallel builds may race without pre-created dirs).
file(MAKE_DIRECTORY
  "${PROJECT_BINARY_DIR}/include/yirage-mlir/Dialect/Yirage/IR"
  "${PROJECT_BINARY_DIR}/include/yirage-mlir/Dialect/Yirage/Transforms"
)

#===----------------------------------------------------------------------===//
# TableGen Configuration
#===----------------------------------------------------------------------===//

set(LLVM_TARGET_DEFINITIONS include/yirage-mlir/Dialect/Yirage/IR/YirageDialect.td)

# Generate Dialect
mlir_tablegen(include/yirage-mlir/Dialect/Yirage/IR/YirageDialect.h.inc -gen-dialect-decls)
mlir_tablegen(include/yirage-mlir/Dialect/Yirage/IR/YirageDialect.cpp.inc -gen-dialect-defs)

add_public_tablegen_target(YirageDialectIncGen)

# Generate Operations
set(LLVM_TARGET_DEFINITIONS include/yirage-mlir/Dialect/Yirage/IR/YirageOps.td)
mlir_tablegen(include/yirage-mlir/Dialect/Yirage/IR/YirageOps.h.inc -gen-op-decls)
mlir_tablegen(include/yirage-mlir/Dialect/Yirage/IR/YirageOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(YirageOpsIncGen)

# Generate Passes
set(LLVM_TARGET_DEFINITIONS include/yirage-mlir/Dialect/Yirage/Transforms/Passes.td)
mlir_tablegen(include/yirage-mlir/Dialect/Yirage/Transforms/Passes.h.inc -gen-pass-decls -name Yirage)
add_public_tablegen_target(YiragePassesIncGen)

#===----------------------------------------------------------------------===//
# Yirage Dialect Library
#===----------------------------------------------------------------------===//

add_mlir_dialect_library(YirageDialect
  lib/Dialect/Yirage/IR/YirageDialect.cpp
  lib/Dialect/Yirage/IR/YirageOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/yirage-mlir/Dialect/Yirage/IR

  DEPENDS
  YirageDialectIncGen
  YirageOpsIncGen

  LINK_LIBS PUBLIC
  ${_YIRAGE_MLIR_DIALECT_LINK_LIBS}
)

#===----------------------------------------------------------------------===//
# Yirage Transforms Library
#===----------------------------------------------------------------------===//

add_mlir_library(YirageTransforms
  lib/Dialect/Yirage/Transforms/YirageToLinalg.cpp
  lib/Dialect/Yirage/Transforms/FlashAttention.cpp
  lib/Dialect/Yirage/Transforms/LinalgToGPU.cpp
  lib/Dialect/Yirage/Transforms/LinalgToCPU.cpp
  lib/Dialect/Yirage/Transforms/MemoryPlanning.cpp
  lib/Dialect/Yirage/Transforms/PassRegistration.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/yirage-mlir/Dialect/Yirage/Transforms

  DEPENDS
  YiragePassesIncGen

  LINK_LIBS PUBLIC
  YirageDialect
  ${_YIRAGE_MLIR_TRANSFORM_LINK_LIBS}
)

#===----------------------------------------------------------------------===//
# Yirage Execution Library (GPU Code Generation)
#===----------------------------------------------------------------------===//

if(_YIRAGE_USE_MLIR_DYLIB)
  set(_YIRAGE_EXEC_LINK_LIBS
    YirageDialect
    YirageTransforms
    MLIR
    LLVMTarget
  )
else()
  set(_YIRAGE_EXEC_LINK_LIBS
    YirageDialect
    YirageTransforms
    MLIRIR
    MLIRPass
    MLIRParser
    MLIRTransforms
    MLIRBufferizationTransforms
    MLIRLinalgTransforms
    MLIRMathDialect
    MLIRLLVMDialect
    MLIRArithToLLVM
    MLIRFuncToLLVM
    MLIRMemRefToLLVM
    MLIRSCFToControlFlow
    MLIRTargetLLVMIRExport
    MLIRExecutionEngine
    LLVMCore
    LLVMSupport
    LLVMTarget
  )
endif()
# Optional GPU codegen libs (not present in minimal distro MLIR dev packages).
foreach(_opt_lib IN ITEMS
    MLIRGPUDialect MLIRGPUToNVVMTransforms MLIRGPUToROCDLTransforms
    MLIRGPUToSPIRVTransforms MLIRNVVMDialect MLIRROCDLDialect
    MLIRSPIRVDialect MLIRSPIRVTransforms LLVMNVPTXCodeGen LLVMAMDGPUCodeGen)
  if(TARGET ${_opt_lib})
    list(APPEND _YIRAGE_EXEC_LINK_LIBS ${_opt_lib})
  endif()
endforeach()

if(_YIRAGE_USE_MLIR_DYLIB)
  set(_YIRAGE_CPU_EXEC_LINK_LIBS
    YirageDialect
    YirageTransforms
    MLIR
  )
else()
  set(_YIRAGE_CPU_EXEC_LINK_LIBS
    YirageDialect
    YirageTransforms
    MLIRIR
    MLIRPass
    MLIRParser
    MLIRTransforms
    MLIRBufferizationTransforms
    MLIRLinalgTransforms
    MLIRFuncDialect
    MLIRArithDialect
    MLIRLinalgDialect
    MLIRTensorDialect
    MLIRMemRefDialect
    MLIRSCFDialect
    MLIRMathDialect
    MLIRLLVMDialect
    MLIRArithToLLVM
    MLIRMathToLLVM
    MLIRFuncToLLVM
    MLIRMemRefToLLVM
    MLIRControlFlowToLLVM
    MLIRSCFToControlFlow
    MLIRReconcileUnrealizedCasts
    MLIRTargetLLVMIRExport
    MLIRExecutionEngine
    MLIRBuiltinToLLVMIRTranslation
    MLIRLLVMToLLVMIRTranslation
  )
endif()

add_mlir_library(YirageCPUExecution
  lib/Execution/JITRunner.cpp
  lib/Execution/CPUJITKernel.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/yirage-mlir/Execution

  LINK_LIBS PUBLIC
  ${_YIRAGE_CPU_EXEC_LINK_LIBS}
)

add_mlir_library(YirageExecution
  lib/Execution/GPUCodeGen.cpp
  lib/Execution/BinaryGen.cpp
  lib/Execution/BackendMLIRConfig.cpp
  # KernelExtractor.cpp: GPU-only; pending MLIR 17 attr API port (KernelDim3Attr).

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/yirage-mlir/Execution

  LINK_LIBS PUBLIC
  ${_YIRAGE_EXEC_LINK_LIBS}
  YirageCPUExecution
)

#===----------------------------------------------------------------------===//
# Yirage Opt Tool
#===----------------------------------------------------------------------===//

add_llvm_executable(yirage-opt
  tools/yirage-opt.cpp

  DEPENDS
  YirageDialect
  YirageTransforms
  YirageExecution
)

add_llvm_executable(yirage-cpu-opt
  tools/yirage-cpu-opt.cpp

  DEPENDS
  YirageDialect
  YirageTransforms
  YirageCPUExecution
)

if(_YIRAGE_USE_MLIR_DYLIB)
  set(_YIRAGE_OPT_LINK_LIBS MLIROptLib MLIR)
else()
  set(_YIRAGE_OPT_LINK_LIBS
    MLIROptLib MLIRIR MLIRParser MLIRPass MLIRTransforms MLIRSupport)
endif()

target_link_libraries(yirage-opt PRIVATE
  YirageDialect
  YirageTransforms
  YirageExecution
  ${_YIRAGE_OPT_LINK_LIBS}
)

mlir_check_all_link_libraries(yirage-opt)

target_link_libraries(yirage-cpu-opt PRIVATE
  YirageDialect
  YirageTransforms
  YirageCPUExecution
  ${_YIRAGE_OPT_LINK_LIBS}
)

mlir_check_all_link_libraries(yirage-cpu-opt)

#===----------------------------------------------------------------------===//
# Python Bindings (Optional)
#===----------------------------------------------------------------------===//

option(YIRAGE_ENABLE_PYTHON "Enable Python bindings" ON)

if(YIRAGE_ENABLE_PYTHON)
  find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
  
  # Use pybind11 or MLIR Python bindings
  if(MLIR_ENABLE_BINDINGS_PYTHON)
    include(MLIRDetectPythonEnv)
    mlir_configure_python_dev_packages()
    
    declare_mlir_python_sources(YiragePythonSources)
    declare_mlir_python_sources(YiragePythonSources.Dialects
      ADD_TO_PARENT YiragePythonSources
    )
    
    # Add Python extension module
    declare_mlir_dialect_python_bindings(
      ADD_TO_PARENT YiragePythonSources.Dialects
      ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/python"
      TD_FILE include/yirage-mlir/Dialect/Yirage/IR/YirageOps.td
      SOURCES
        python/YirageDialect.cpp
      DIALECT_NAME yirage
    )
    
    add_mlir_python_modules(YiragePythonModules
      ROOT_PREFIX "${CMAKE_BINARY_DIR}/python/yirage_mlir"
      INSTALL_PREFIX "python/yirage_mlir"
      DECLARED_SOURCES YiragePythonSources
    )
    # CPUJITKernel / JITRunner live in YirageExecution (used by YirageDialect.cpp).
    mlir_target_link_libraries(YiragePythonModules PUBLIC YirageExecution)
  else()
    # Debian/Ubuntu llvm-*-dev ships MLIR_ENABLE_BINDINGS_PYTHON=OFF; use pybind11 directly.
    message(STATUS "MLIR Python bindings disabled in system MLIR; building _yirage_mlir via pybind11")
    execute_process(
      COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
      OUTPUT_VARIABLE _pybind11_cmake_dir
      OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE _pybind11_rc
    )
    if(_pybind11_rc EQUAL 0)
      list(APPEND CMAKE_PREFIX_PATH "${_pybind11_cmake_dir}")
    endif()
    find_package(pybind11 CONFIG REQUIRED)

    pybind11_add_module(_yirage_mlir MODULE
      ${CMAKE_CURRENT_SOURCE_DIR}/python/YirageDialect.cpp
    )

    target_include_directories(_yirage_mlir PRIVATE
      ${PROJECT_SOURCE_DIR}/include
      ${PROJECT_BINARY_DIR}/include
      ${MLIR_INCLUDE_DIRS}
      ${LLVM_INCLUDE_DIRS}
    )

    # CPU JIT: YirageCPUExecution + MLIR dylib (no extra static MLIR*.a).
    if(_YIRAGE_USE_MLIR_DYLIB)
      set(_YIRAGE_PYMLIR_LINK_LIBS YirageCPUExecution MLIR)
    else()
      set(_YIRAGE_PYMLIR_LINK_LIBS
        YirageCPUExecution YirageTransforms YirageDialect
        MLIRFuncDialect MLIRTensorDialect MLIRParser)
    endif()
    target_link_libraries(_yirage_mlir PRIVATE ${_YIRAGE_PYMLIR_LINK_LIBS})
    mlir_check_all_link_libraries(_yirage_mlir)

    set(_yirage_pymlir_out "${CMAKE_CURRENT_SOURCE_DIR}/../python/yirage")
    set_target_properties(_yirage_mlir PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY "${_yirage_pymlir_out}"
      PREFIX ""
      OUTPUT_NAME "_yirage_mlir"
    )
  endif()
endif()

#===----------------------------------------------------------------------===//
# Installation
#===----------------------------------------------------------------------===//

install(TARGETS yirage-opt
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/yirage-mlir
  DESTINATION include
  FILES_MATCHING PATTERN "*.h" PATTERN "*.td"
)

install(DIRECTORY ${PROJECT_BINARY_DIR}/include/yirage-mlir
  DESTINATION include
  FILES_MATCHING PATTERN "*.h.inc"
)
