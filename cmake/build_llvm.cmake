# =============================================================================
# YiRage LLVM/MLIR Build Configuration
# =============================================================================
# This CMake module handles building LLVM/MLIR from source as part of the
# YiRage build process. It supports:
#   - Automatic download via FetchContent or Git submodule
#   - Minimal LLVM build with only required components
#   - Prebuilt binary download from GitHub Releases
#   - System LLVM detection as fallback
#
# Usage:
#   include(cmake/build_llvm.cmake)
#   setup_llvm_mlir()
#
# Options:
#   YIRAGE_LLVM_SOURCE: "submodule" | "fetch" | "prebuilt" | "system"
#   YIRAGE_LLVM_VERSION: LLVM version (default: 17)
#   YIRAGE_LLVM_BUILD_TYPE: Release | Debug | RelWithDebInfo
#
# =============================================================================

cmake_minimum_required(VERSION 3.20)

# =============================================================================
# Configuration Options
# =============================================================================

set(YIRAGE_LLVM_VERSION "17" CACHE STRING "LLVM version to use")
set(YIRAGE_LLVM_SOURCE "submodule" CACHE STRING "LLVM source: submodule, fetch, prebuilt, system")
set_property(CACHE YIRAGE_LLVM_SOURCE PROPERTY STRINGS "submodule" "fetch" "prebuilt" "system")
set(YIRAGE_LLVM_BUILD_TYPE "Release" CACHE STRING "LLVM build type")

# Prebuilt binary URL pattern
set(YIRAGE_LLVM_PREBUILT_URL_BASE 
    "https://github.com/chenxingqiang/YiRage/releases/download/llvm-prebuilt-v${YIRAGE_LLVM_VERSION}"
    CACHE STRING "Base URL for prebuilt LLVM binaries")

# LLVM build options - minimal build for MLIR
set(YIRAGE_LLVM_TARGETS "host;NVPTX;AMDGPU" CACHE STRING "LLVM targets to build")
set(YIRAGE_LLVM_PROJECTS "mlir" CACHE STRING "LLVM projects to enable")

# =============================================================================
# Helper Functions
# =============================================================================

# Detect system architecture
function(detect_system_arch OUT_ARCH)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
        set(${OUT_ARCH} "x86_64" PARENT_SCOPE)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64|arm64")
        set(${OUT_ARCH} "aarch64" PARENT_SCOPE)
    else()
        set(${OUT_ARCH} "${CMAKE_SYSTEM_PROCESSOR}" PARENT_SCOPE)
    endif()
endfunction()

# Detect operating system
function(detect_system_os OUT_OS)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        set(${OUT_OS} "linux" PARENT_SCOPE)
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
        set(${OUT_OS} "macos" PARENT_SCOPE)
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
        set(${OUT_OS} "windows" PARENT_SCOPE)
    else()
        set(${OUT_OS} "${CMAKE_SYSTEM_NAME}" PARENT_SCOPE)
    endif()
endfunction()

# =============================================================================
# System LLVM Detection
# =============================================================================

function(find_system_llvm OUT_FOUND)
    set(${OUT_FOUND} FALSE PARENT_SCOPE)
    
    # Try to find MLIR package
    find_package(MLIR CONFIG QUIET)
    
    if(MLIR_FOUND)
        message(STATUS "Found system MLIR: ${MLIR_DIR}")
        set(${OUT_FOUND} TRUE PARENT_SCOPE)
        return()
    endif()
    
    # Try common paths
    set(LLVM_SEARCH_PATHS
        "/usr/lib/llvm-${YIRAGE_LLVM_VERSION}"
        "/usr/lib/llvm-17"
        "/usr/lib/llvm-18"
        "/opt/homebrew/opt/llvm@17"
        "/opt/homebrew/opt/llvm@18"
        "/opt/homebrew/opt/llvm"
        "/usr/local/opt/llvm@17"
        "/usr/local/opt/llvm@18"
        "/usr/local/opt/llvm"
        "/opt/llvm"
        "C:/Program Files/LLVM"
    )
    
    foreach(LLVM_PATH ${LLVM_SEARCH_PATHS})
        if(EXISTS "${LLVM_PATH}/lib/cmake/mlir/MLIRConfig.cmake")
            set(MLIR_DIR "${LLVM_PATH}/lib/cmake/mlir" CACHE PATH "MLIR CMake directory" FORCE)
            set(LLVM_DIR "${LLVM_PATH}/lib/cmake/llvm" CACHE PATH "LLVM CMake directory" FORCE)
            message(STATUS "Found LLVM/MLIR at: ${LLVM_PATH}")
            set(${OUT_FOUND} TRUE PARENT_SCOPE)
            return()
        endif()
    endforeach()
endfunction()

# =============================================================================
# Download Prebuilt LLVM
# =============================================================================

function(download_prebuilt_llvm)
    detect_system_arch(ARCH)
    detect_system_os(OS)
    
    set(PREBUILT_FILENAME "llvm-${YIRAGE_LLVM_VERSION}-${OS}-${ARCH}.tar.gz")
    set(PREBUILT_URL "${YIRAGE_LLVM_PREBUILT_URL_BASE}/${PREBUILT_FILENAME}")
    set(PREBUILT_DIR "${CMAKE_BINARY_DIR}/llvm-prebuilt")
    set(PREBUILT_ARCHIVE "${CMAKE_BINARY_DIR}/${PREBUILT_FILENAME}")
    
    message(STATUS "Downloading prebuilt LLVM from: ${PREBUILT_URL}")
    
    # Download if not exists
    if(NOT EXISTS "${PREBUILT_ARCHIVE}")
        file(DOWNLOAD 
            "${PREBUILT_URL}" 
            "${PREBUILT_ARCHIVE}"
            STATUS DOWNLOAD_STATUS
            SHOW_PROGRESS
        )
        list(GET DOWNLOAD_STATUS 0 DOWNLOAD_ERROR)
        if(DOWNLOAD_ERROR)
            message(WARNING "Failed to download prebuilt LLVM: ${DOWNLOAD_STATUS}")
            return()
        endif()
    endif()
    
    # Extract if not exists
    if(NOT EXISTS "${PREBUILT_DIR}/lib/cmake/mlir")
        message(STATUS "Extracting prebuilt LLVM...")
        file(ARCHIVE_EXTRACT 
            INPUT "${PREBUILT_ARCHIVE}"
            DESTINATION "${PREBUILT_DIR}"
        )
    endif()
    
    # Set paths
    set(MLIR_DIR "${PREBUILT_DIR}/lib/cmake/mlir" CACHE PATH "MLIR CMake directory" FORCE)
    set(LLVM_DIR "${PREBUILT_DIR}/lib/cmake/llvm" CACHE PATH "LLVM CMake directory" FORCE)
    set(YIRAGE_LLVM_ROOT "${PREBUILT_DIR}" CACHE PATH "LLVM root directory" FORCE)
endfunction()

# =============================================================================
# Build LLVM from Submodule
# =============================================================================

function(build_llvm_from_submodule)
    set(LLVM_SOURCE_DIR "${CMAKE_SOURCE_DIR}/deps/llvm-project/llvm")
    set(LLVM_BUILD_DIR "${CMAKE_BINARY_DIR}/llvm-build")
    set(LLVM_INSTALL_DIR "${CMAKE_BINARY_DIR}/llvm-install")
    
    if(NOT EXISTS "${LLVM_SOURCE_DIR}/CMakeLists.txt")
        message(FATAL_ERROR 
            "LLVM submodule not found at ${LLVM_SOURCE_DIR}\n"
            "Please initialize the submodule:\n"
            "  git submodule update --init --depth 1 deps/llvm-project\n"
            "Or use a different LLVM source:\n"
            "  cmake -DYIRAGE_LLVM_SOURCE=system .."
        )
    endif()
    
    # Check if already built
    if(EXISTS "${LLVM_INSTALL_DIR}/lib/cmake/mlir/MLIRConfig.cmake")
        message(STATUS "Using pre-built LLVM from: ${LLVM_INSTALL_DIR}")
        set(MLIR_DIR "${LLVM_INSTALL_DIR}/lib/cmake/mlir" CACHE PATH "MLIR CMake directory" FORCE)
        set(LLVM_DIR "${LLVM_INSTALL_DIR}/lib/cmake/llvm" CACHE PATH "LLVM CMake directory" FORCE)
        set(YIRAGE_LLVM_ROOT "${LLVM_INSTALL_DIR}" CACHE PATH "LLVM root directory" FORCE)
        return()
    endif()
    
    message(STATUS "Building LLVM from submodule...")
    message(STATUS "  Source: ${LLVM_SOURCE_DIR}")
    message(STATUS "  Build:  ${LLVM_BUILD_DIR}")
    message(STATUS "  Install: ${LLVM_INSTALL_DIR}")
    
    # Configure LLVM
    file(MAKE_DIRECTORY "${LLVM_BUILD_DIR}")
    
    execute_process(
        COMMAND ${CMAKE_COMMAND}
            -G "${CMAKE_GENERATOR}"
            -DCMAKE_BUILD_TYPE=${YIRAGE_LLVM_BUILD_TYPE}
            -DCMAKE_INSTALL_PREFIX=${LLVM_INSTALL_DIR}
            -DLLVM_ENABLE_PROJECTS=${YIRAGE_LLVM_PROJECTS}
            -DLLVM_TARGETS_TO_BUILD=${YIRAGE_LLVM_TARGETS}
            -DLLVM_ENABLE_ASSERTIONS=ON
            -DLLVM_ENABLE_RTTI=ON
            -DLLVM_ENABLE_EH=ON
            -DLLVM_BUILD_EXAMPLES=OFF
            -DLLVM_BUILD_TESTS=OFF
            -DLLVM_BUILD_BENCHMARKS=OFF
            -DLLVM_INCLUDE_EXAMPLES=OFF
            -DLLVM_INCLUDE_TESTS=OFF
            -DLLVM_INCLUDE_BENCHMARKS=OFF
            -DMLIR_ENABLE_BINDINGS_PYTHON=OFF
            -DLLVM_ENABLE_ZLIB=OFF
            -DLLVM_ENABLE_ZSTD=OFF
            -DLLVM_ENABLE_LIBXML2=OFF
            -DLLVM_ENABLE_TERMINFO=OFF
            ${LLVM_SOURCE_DIR}
        WORKING_DIRECTORY "${LLVM_BUILD_DIR}"
        RESULT_VARIABLE LLVM_CONFIG_RESULT
    )
    
    if(NOT LLVM_CONFIG_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to configure LLVM")
    endif()
    
    # Build LLVM
    include(ProcessorCount)
    ProcessorCount(NPROC)
    if(NPROC EQUAL 0)
        set(NPROC 4)
    endif()
    
    execute_process(
        COMMAND ${CMAKE_COMMAND} --build . --parallel ${NPROC}
        WORKING_DIRECTORY "${LLVM_BUILD_DIR}"
        RESULT_VARIABLE LLVM_BUILD_RESULT
    )
    
    if(NOT LLVM_BUILD_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to build LLVM")
    endif()
    
    # Install LLVM
    execute_process(
        COMMAND ${CMAKE_COMMAND} --install .
        WORKING_DIRECTORY "${LLVM_BUILD_DIR}"
        RESULT_VARIABLE LLVM_INSTALL_RESULT
    )
    
    if(NOT LLVM_INSTALL_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to install LLVM")
    endif()
    
    # Set paths
    set(MLIR_DIR "${LLVM_INSTALL_DIR}/lib/cmake/mlir" CACHE PATH "MLIR CMake directory" FORCE)
    set(LLVM_DIR "${LLVM_INSTALL_DIR}/lib/cmake/llvm" CACHE PATH "LLVM CMake directory" FORCE)
    set(YIRAGE_LLVM_ROOT "${LLVM_INSTALL_DIR}" CACHE PATH "LLVM root directory" FORCE)
    
    message(STATUS "LLVM built and installed successfully")
endfunction()

# =============================================================================
# Fetch LLVM via FetchContent
# =============================================================================

function(fetch_llvm_content)
    include(FetchContent)
    
    message(STATUS "Fetching LLVM ${YIRAGE_LLVM_VERSION} via FetchContent...")
    
    FetchContent_Declare(
        llvm-project
        GIT_REPOSITORY https://github.com/llvm/llvm-project.git
        GIT_TAG llvmorg-${YIRAGE_LLVM_VERSION}.0.0
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE
        SOURCE_SUBDIR llvm
    )
    
    # Set LLVM build options before FetchContent_MakeAvailable
    set(LLVM_ENABLE_PROJECTS "mlir" CACHE STRING "" FORCE)
    set(LLVM_TARGETS_TO_BUILD "${YIRAGE_LLVM_TARGETS}" CACHE STRING "" FORCE)
    set(LLVM_ENABLE_ASSERTIONS ON CACHE BOOL "" FORCE)
    set(LLVM_ENABLE_RTTI ON CACHE BOOL "" FORCE)
    set(LLVM_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(LLVM_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(LLVM_INCLUDE_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(LLVM_INCLUDE_TESTS OFF CACHE BOOL "" FORCE)
    set(LLVM_INCLUDE_BENCHMARKS OFF CACHE BOOL "" FORCE)
    
    FetchContent_MakeAvailable(llvm-project)
    
    # Set paths for the fetched LLVM
    set(MLIR_DIR "${llvm-project_BINARY_DIR}/lib/cmake/mlir" CACHE PATH "MLIR CMake directory" FORCE)
    set(LLVM_DIR "${llvm-project_BINARY_DIR}/lib/cmake/llvm" CACHE PATH "LLVM CMake directory" FORCE)
endfunction()

# =============================================================================
# Main Setup Function
# =============================================================================

function(setup_llvm_mlir)
    message(STATUS "============================================")
    message(STATUS "  YiRage LLVM/MLIR Setup")
    message(STATUS "============================================")
    message(STATUS "LLVM Source: ${YIRAGE_LLVM_SOURCE}")
    message(STATUS "LLVM Version: ${YIRAGE_LLVM_VERSION}")
    
    # Check if MLIR_DIR is already set
    if(DEFINED MLIR_DIR AND EXISTS "${MLIR_DIR}/MLIRConfig.cmake")
        message(STATUS "Using pre-configured MLIR: ${MLIR_DIR}")
        return()
    endif()
    
    # Setup based on source type
    if(YIRAGE_LLVM_SOURCE STREQUAL "system")
        find_system_llvm(SYSTEM_LLVM_FOUND)
        if(NOT SYSTEM_LLVM_FOUND)
            message(FATAL_ERROR 
                "System LLVM not found. Please install LLVM/MLIR or use a different source:\n"
                "  cmake -DYIRAGE_LLVM_SOURCE=submodule ..\n"
                "  cmake -DYIRAGE_LLVM_SOURCE=prebuilt .."
            )
        endif()
        
    elseif(YIRAGE_LLVM_SOURCE STREQUAL "prebuilt")
        download_prebuilt_llvm()
        
    elseif(YIRAGE_LLVM_SOURCE STREQUAL "submodule")
        build_llvm_from_submodule()
        
    elseif(YIRAGE_LLVM_SOURCE STREQUAL "fetch")
        fetch_llvm_content()
        
    else()
        message(FATAL_ERROR "Unknown YIRAGE_LLVM_SOURCE: ${YIRAGE_LLVM_SOURCE}")
    endif()
    
    # Verify MLIR is available
    if(NOT EXISTS "${MLIR_DIR}/MLIRConfig.cmake")
        message(FATAL_ERROR "MLIR CMake config not found at: ${MLIR_DIR}")
    endif()
    
    message(STATUS "MLIR_DIR: ${MLIR_DIR}")
    message(STATUS "LLVM_DIR: ${LLVM_DIR}")
    message(STATUS "============================================")
endfunction()

# =============================================================================
# MLIR Integration Macros
# =============================================================================

# Setup MLIR includes and libraries for a target
macro(target_link_mlir TARGET)
    if(NOT DEFINED MLIR_DIR)
        message(FATAL_ERROR "MLIR_DIR is not set. Call setup_llvm_mlir() first.")
    endif()
    
    find_package(MLIR REQUIRED CONFIG)
    find_package(LLVM REQUIRED CONFIG)
    
    list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
    list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
    
    include(TableGen)
    include(AddLLVM)
    include(AddMLIR)
    include(HandleLLVMOptions)
    
    target_include_directories(${TARGET} SYSTEM PUBLIC
        ${LLVM_INCLUDE_DIRS}
        ${MLIR_INCLUDE_DIRS}
    )
    
    target_compile_definitions(${TARGET} PUBLIC
        ${LLVM_DEFINITIONS}
        YIRAGE_MLIR_ENABLED
    )
    
    # Link core MLIR libraries
    target_link_libraries(${TARGET} PUBLIC
        MLIRIR
        MLIRParser
        MLIRPass
        MLIRTransforms
        MLIRSupport
    )
endmacro()

# Add MLIR dialect library
macro(add_yirage_mlir_dialect_library NAME)
    add_mlir_dialect_library(${NAME} ${ARGN})
endmacro()

# =============================================================================
# Export Variables
# =============================================================================

# Make variables available to parent scope
set(YIRAGE_LLVM_AVAILABLE TRUE CACHE BOOL "LLVM/MLIR is available")
