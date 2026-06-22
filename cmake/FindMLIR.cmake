# =============================================================================
# FindMLIR.cmake - CMake module to find MLIR
# =============================================================================
# This module finds MLIR and sets up the required variables for using MLIR
# in the YiRage project.
#
# Input Variables:
#   MLIR_DIR         - Path to MLIRConfig.cmake directory
#   LLVM_DIR         - Path to LLVMConfig.cmake directory
#   YIRAGE_LLVM_ROOT - Root directory of LLVM installation
#
# Output Variables:
#   MLIR_FOUND       - True if MLIR was found
#   MLIR_VERSION     - MLIR version string
#   MLIR_INCLUDE_DIRS - MLIR include directories
#   MLIR_LIBRARY_DIRS - MLIR library directories
#   MLIR_LIBRARIES   - MLIR libraries to link
#   MLIR_CMAKE_DIR   - MLIR CMake directory
#   LLVM_CMAKE_DIR   - LLVM CMake directory
#
# Targets:
#   MLIR::MLIR       - Imported target for MLIR
#
# =============================================================================

# Try to find MLIR config
if(NOT MLIR_DIR)
    # Search common paths
    set(_MLIR_SEARCH_PATHS
        # Environment variable
        "$ENV{MLIR_DIR}"
        "$ENV{LLVM_DIR}/../mlir"
        
        # YiRage build directory
        "${CMAKE_BINARY_DIR}/llvm-install/lib/cmake/mlir"
        "${CMAKE_BINARY_DIR}/llvm-prebuilt/lib/cmake/mlir"
        
        # System paths - Ubuntu/Debian
        "/usr/lib/llvm-17/lib/cmake/mlir"
        "/usr/lib/llvm-18/lib/cmake/mlir"
        "/usr/lib/llvm-16/lib/cmake/mlir"
        
        # System paths - Generic Linux
        "/usr/local/lib/cmake/mlir"
        "/usr/lib/cmake/mlir"
        "/usr/lib64/cmake/mlir"
        
        # Homebrew paths - macOS
        "/opt/homebrew/opt/llvm/lib/cmake/mlir"
        "/opt/homebrew/opt/llvm@17/lib/cmake/mlir"
        "/usr/local/opt/llvm/lib/cmake/mlir"
        "/usr/local/opt/llvm@17/lib/cmake/mlir"
        
        # Windows paths
        "C:/Program Files/LLVM/lib/cmake/mlir"
        "C:/LLVM/lib/cmake/mlir"
        
        # Conda paths
        "$ENV{CONDA_PREFIX}/lib/cmake/mlir"
    )
    
    foreach(_path ${_MLIR_SEARCH_PATHS})
        if(EXISTS "${_path}/MLIRConfig.cmake")
            set(MLIR_DIR "${_path}" CACHE PATH "Path to MLIRConfig.cmake")
            message(STATUS "Found MLIR at: ${MLIR_DIR}")
            break()
        endif()
    endforeach()
endif()

# Find MLIR package
if(MLIR_DIR)
    find_package(MLIR CONFIG QUIET PATHS "${MLIR_DIR}" NO_DEFAULT_PATH)
endif()

if(NOT MLIR_FOUND)
    find_package(MLIR CONFIG QUIET)
endif()

# Find LLVM if not already found
if(NOT LLVM_DIR)
    if(MLIR_DIR)
        get_filename_component(_LLVM_DIR "${MLIR_DIR}/../llvm" ABSOLUTE)
        if(EXISTS "${_LLVM_DIR}/LLVMConfig.cmake")
            set(LLVM_DIR "${_LLVM_DIR}" CACHE PATH "Path to LLVMConfig.cmake")
        endif()
    endif()
endif()

if(LLVM_DIR)
    find_package(LLVM CONFIG QUIET PATHS "${LLVM_DIR}" NO_DEFAULT_PATH)
endif()

if(NOT LLVM_FOUND)
    find_package(LLVM CONFIG QUIET)
endif()

# Set output variables
if(MLIR_FOUND AND LLVM_FOUND)
    set(MLIR_VERSION "${MLIR_PACKAGE_VERSION}")
    set(MLIR_CMAKE_DIR "${MLIR_DIR}")
    set(LLVM_CMAKE_DIR "${LLVM_DIR}")
    
    # Include directories
    set(MLIR_INCLUDE_DIRS "${MLIR_INCLUDE_DIRS}")
    set(LLVM_INCLUDE_DIRS "${LLVM_INCLUDE_DIRS}")
    
    # Library directories
    get_filename_component(MLIR_LIBRARY_DIRS "${MLIR_DIR}/../../" ABSOLUTE)
    
    # Setup module paths
    list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
    list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
    
    # Include MLIR/LLVM CMake utilities
    include(TableGen OPTIONAL)
    include(AddLLVM OPTIONAL)
    include(AddMLIR OPTIONAL)
    include(HandleLLVMOptions OPTIONAL)
    
    # Core MLIR libraries
    set(MLIR_LIBRARIES
        MLIRIR
        MLIRParser
        MLIRPass
        MLIRTransforms
        MLIRSupport
        MLIRAnalysis
    )
    
    # Dialect libraries
    set(MLIR_DIALECT_LIBRARIES
        MLIRArithDialect
        MLIRFuncDialect
        MLIRLinalgDialect
        MLIRTensorDialect
        MLIRMemRefDialect
        MLIRSCFDialect
        MLIRAffineDialect
        MLIRVectorDialect
        MLIRGPUDialect
        MLIRLLVMDialect
    )
    
    # Conversion libraries
    set(MLIR_CONVERSION_LIBRARIES
        MLIRArithToLLVM
        MLIRFuncToLLVM
        MLIRMemRefToLLVM
        MLIRSCFToControlFlow
        MLIRAffineToStandard
        MLIRLinalgTransforms
    )
    
    # GPU target libraries
    set(MLIR_GPU_LIBRARIES
        MLIRNVVMDialect
        MLIRROCDLDialect
        MLIRSPIRVDialect
        MLIRGPUToNVVMTransforms
        MLIRGPUToROCDLTransforms
        MLIRGPUToSPIRVTransforms
    )
    
    # Execution engine libraries
    set(MLIR_EXECUTION_LIBRARIES
        MLIRExecutionEngine
        MLIRTargetLLVMIRExport
    )
    
    message(STATUS "Found MLIR ${MLIR_VERSION}")
    message(STATUS "  MLIR_DIR: ${MLIR_DIR}")
    message(STATUS "  LLVM_DIR: ${LLVM_DIR}")
    message(STATUS "  MLIR includes: ${MLIR_INCLUDE_DIRS}")
    
else()
    set(MLIR_FOUND FALSE)
    
    if(MLIR_FIND_REQUIRED)
        message(FATAL_ERROR 
            "MLIR not found. Please set MLIR_DIR to the directory containing MLIRConfig.cmake\n"
            "  cmake -DMLIR_DIR=/path/to/mlir/lib/cmake/mlir ..\n"
            "Or install LLVM/MLIR:\n"
            "  ./scripts/install_llvm.sh\n"
            "Or build from submodule:\n"
            "  cmake -DYIRAGE_LLVM_SOURCE=submodule .."
        )
    else()
        message(WARNING "MLIR not found. MLIR support will be disabled.")
    endif()
endif()

# =============================================================================
# Helper Functions
# =============================================================================

# Function to add MLIR includes and libraries to a target
function(target_use_mlir TARGET)
    if(NOT MLIR_FOUND)
        message(FATAL_ERROR "Cannot use MLIR - MLIR not found")
    endif()
    
    target_include_directories(${TARGET} SYSTEM PUBLIC
        ${MLIR_INCLUDE_DIRS}
        ${LLVM_INCLUDE_DIRS}
    )
    
    target_compile_definitions(${TARGET} PUBLIC
        ${LLVM_DEFINITIONS}
        YIRAGE_MLIR_ENABLED=1
    )
    
    target_link_directories(${TARGET} PUBLIC
        ${MLIR_LIBRARY_DIRS}
    )
endfunction()

# Function to link MLIR core libraries
function(target_link_mlir_core TARGET)
    target_link_libraries(${TARGET} PUBLIC
        ${MLIR_LIBRARIES}
    )
endfunction()

# Function to link MLIR dialect libraries
function(target_link_mlir_dialects TARGET)
    target_link_libraries(${TARGET} PUBLIC
        ${MLIR_DIALECT_LIBRARIES}
    )
endfunction()

# Function to link all MLIR libraries
function(target_link_mlir_all TARGET)
    target_use_mlir(${TARGET})
    target_link_libraries(${TARGET} PUBLIC
        ${MLIR_LIBRARIES}
        ${MLIR_DIALECT_LIBRARIES}
        ${MLIR_CONVERSION_LIBRARIES}
    )
endfunction()

# Function to link MLIR GPU libraries
function(target_link_mlir_gpu TARGET)
    target_link_libraries(${TARGET} PUBLIC
        ${MLIR_GPU_LIBRARIES}
    )
endfunction()

# Function to link MLIR execution libraries
function(target_link_mlir_execution TARGET)
    target_link_libraries(${TARGET} PUBLIC
        ${MLIR_EXECUTION_LIBRARIES}
    )
endfunction()

# =============================================================================
# Provide imported target
# =============================================================================

if(MLIR_FOUND AND NOT TARGET MLIR::MLIR)
    add_library(MLIR::MLIR INTERFACE IMPORTED)
    set_target_properties(MLIR::MLIR PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${MLIR_INCLUDE_DIRS};${LLVM_INCLUDE_DIRS}"
        INTERFACE_COMPILE_DEFINITIONS "${LLVM_DEFINITIONS};YIRAGE_MLIR_ENABLED=1"
    )
endif()

# Mark as found
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MLIR
    REQUIRED_VARS MLIR_DIR MLIR_INCLUDE_DIRS
    VERSION_VAR MLIR_VERSION
)
