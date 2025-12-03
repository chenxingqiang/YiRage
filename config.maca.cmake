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
#
# MetaX MACA SDK Configuration for YiRage
#
# This file handles detection and configuration of the MetaX MACA SDK.
# MACA is a CUDA-compatible software stack for MetaX GPUs.
#
# Environment Variables:
#   MACA_HOME - Primary path to MACA SDK
#   MACA_PATH - Alternative path to MACA SDK
#
# For more info: https://developer.metax-tech.com/

# Option to enable MACA backend
option(YIRAGE_USE_MACA "Enable MetaX MACA backend" OFF)

# Find MACA SDK
macro(find_maca)
    # Check environment variables first
    if(DEFINED ENV{MACA_HOME})
        set(MACA_ROOT $ENV{MACA_HOME})
    elseif(DEFINED ENV{MACA_PATH})
        set(MACA_ROOT $ENV{MACA_PATH})
    else()
        # Check standard installation paths
        set(MACA_SEARCH_PATHS
            /opt/maca
            /usr/local/maca
            /opt/metax/maca
            /usr/local/metax/maca
        )
        
        foreach(path ${MACA_SEARCH_PATHS})
            if(EXISTS "${path}/include/cuda_runtime.h")
                set(MACA_ROOT ${path})
                break()
            endif()
        endforeach()
    endif()
    
    if(MACA_ROOT)
        message(STATUS "Found MACA SDK at: ${MACA_ROOT}")
        set(MACA_FOUND TRUE)
        
        # Set MACA paths
        set(MACA_INCLUDE_DIR "${MACA_ROOT}/include")
        set(MACA_LIBRARY_DIR "${MACA_ROOT}/lib64")
        
        if(NOT EXISTS "${MACA_LIBRARY_DIR}")
            set(MACA_LIBRARY_DIR "${MACA_ROOT}/lib")
        endif()
        
        # Find mxcc compiler (MACA's CUDA-compatible compiler)
        find_program(MACA_COMPILER mxcc
            PATHS 
                ${MACA_ROOT}/bin
                /opt/maca/bin
                /usr/local/maca/bin
        )
        
        if(MACA_COMPILER)
            message(STATUS "Found MACA compiler: ${MACA_COMPILER}")
        else()
            message(STATUS "MACA compiler (mxcc) not found, using nvcc-compatible mode")
        endif()
        
        # Find MACA libraries
        find_library(MACA_RUNTIME_LIB 
            NAMES maca_runtime cudart
            PATHS ${MACA_LIBRARY_DIR}
            NO_DEFAULT_PATH
        )
        
        find_library(MACA_BLAS_LIB
            NAMES mcblas cublas
            PATHS ${MACA_LIBRARY_DIR}
            NO_DEFAULT_PATH
        )
        
        find_library(MACA_CCL_LIB
            NAMES mccl nccl
            PATHS ${MACA_LIBRARY_DIR}
            NO_DEFAULT_PATH
        )
        
        if(MACA_RUNTIME_LIB)
            message(STATUS "Found MACA runtime: ${MACA_RUNTIME_LIB}")
        endif()
        
        if(MACA_BLAS_LIB)
            message(STATUS "Found MACA BLAS: ${MACA_BLAS_LIB}")
        endif()
        
    else()
        set(MACA_FOUND FALSE)
        message(STATUS "MACA SDK not found")
    endif()
endmacro()

# Configure MACA for build
macro(configure_maca)
    if(NOT MACA_FOUND)
        message(FATAL_ERROR "MACA SDK not found but YIRAGE_USE_MACA is enabled")
    endif()
    
    # Add MACA include directories
    include_directories(${MACA_INCLUDE_DIR})
    link_directories(${MACA_LIBRARY_DIR})
    
    # Add MACA compile definitions
    add_definitions(-DYIRAGE_BACKEND_MACA_ENABLED)
    
    # Set MACA compiler flags
    # MACA is CUDA-compatible, so we use similar flags
    set(MACA_NVCC_FLAGS
        -std=c++17
        -O2
        -Xcompiler=-fPIC
        --expt-relaxed-constexpr
        --expt-extended-lambda
    )
    
    # GPU architecture flags
    # MetaX GPUs typically support SM75 compatible code
    set(MACA_ARCH_FLAGS
        -gencode=arch=compute_75,code=sm_75
    )
    
    # For newer MetaX hardware
    if(MACA_ARCH_80)
        list(APPEND MACA_ARCH_FLAGS
            -gencode=arch=compute_80,code=sm_80
        )
    endif()
    
    # Export MACA configuration
    set(MACA_CONFIGURED TRUE PARENT_SCOPE)
    
    message(STATUS "MACA backend configured successfully")
    message(STATUS "  Include dir: ${MACA_INCLUDE_DIR}")
    message(STATUS "  Library dir: ${MACA_LIBRARY_DIR}")
    
endmacro()

# MACA source file compilation
macro(maca_add_library target_name)
    set(MACA_SOURCES ${ARGN})
    
    if(MACA_COMPILER)
        # Use mxcc for compilation
        set(CMAKE_CUDA_COMPILER ${MACA_COMPILER})
    endif()
    
    # Create library with CUDA language enabled
    add_library(${target_name} ${MACA_SOURCES})
    
    target_include_directories(${target_name} PUBLIC
        ${MACA_INCLUDE_DIR}
    )
    
    target_link_directories(${target_name} PUBLIC
        ${MACA_LIBRARY_DIR}
    )
    
    # Link MACA libraries
    target_link_libraries(${target_name}
        ${MACA_RUNTIME_LIB}
        ${MACA_BLAS_LIB}
    )
    
    if(MACA_CCL_LIB)
        target_link_libraries(${target_name} ${MACA_CCL_LIB})
    endif()
    
    target_compile_definitions(${target_name} PUBLIC
        YIRAGE_BACKEND_MACA_ENABLED
    )
    
endmacro()

# Get MACA device information at configure time
macro(get_maca_device_info)
    if(MACA_FOUND)
        # Try to get device info using mcdevicequery or similar tool
        find_program(MACA_DEVICE_QUERY mcdevicequery deviceQuery
            PATHS 
                ${MACA_ROOT}/bin
                ${MACA_ROOT}/samples/bin
        )
        
        if(MACA_DEVICE_QUERY)
            execute_process(
                COMMAND ${MACA_DEVICE_QUERY}
                OUTPUT_VARIABLE MACA_DEVICE_OUTPUT
                ERROR_QUIET
                OUTPUT_STRIP_TRAILING_WHITESPACE
            )
            message(STATUS "MACA device info available")
        endif()
    endif()
endmacro()

# Main configuration entry point
if(YIRAGE_USE_MACA)
    find_maca()
    
    if(MACA_FOUND)
        configure_maca()
        get_maca_device_info()
    else()
        message(WARNING "YIRAGE_USE_MACA enabled but MACA SDK not found")
        set(YIRAGE_USE_MACA OFF)
    endif()
endif()

