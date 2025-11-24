#!/bin/bash

# Multi-Backend Performance Profiling Script for YiRage
# Supports CUDA, MPS (Apple Silicon), and CPU backends
#
# Usage:
# ./analyze_multibackend.sh <backend> <report_file> <command_to_profile...>
#
# Examples:
# ./analyze_multibackend.sh cuda matmul.ncu-rep python benchmark/gated_mlp.py --backend cuda
# ./analyze_multibackend.sh mps matmul.trace python benchmark/gated_mlp.py --backend mps
# ./analyze_multibackend.sh cpu matmul.perf python benchmark/gated_mlp.py --backend cpu

set -e
export YIRAGE_HOME=$(pwd)

# Parse arguments
if [ $# -lt 3 ]; then
    echo "Usage: $0 <backend> <report_file> <command_to_profile...>"
    echo ""
    echo "Backends:"
    echo "  cuda  - NVIDIA GPU (requires Nsight Compute)"
    echo "  mps   - Apple Silicon (requires Instruments)"
    echo "  cpu   - CPU profiling (requires perf or Instruments)"
    echo ""
    echo "Examples:"
    echo "  $0 cuda output.ncu-rep python benchmark/gated_mlp.py --backend cuda"
    echo "  $0 mps output.trace python benchmark/gated_mlp.py --backend mps"
    echo "  $0 cpu output.perf python benchmark/gated_mlp.py --backend cpu"
    exit 1
fi

BACKEND=$1; shift
REPORT_FILE=$1; shift
TARGET_CMD=( "$@" )

echo "========================================"
echo "YiRage Multi-Backend Performance Profiling"
echo "========================================"
echo "Backend: $BACKEND"
echo "Report: $REPORT_FILE"
echo "Command: ${TARGET_CMD[@]}"
echo "========================================"

# Activate virtual environment if present
if [[ -n "$VIRTUAL_ENV" ]]; then
    source "$VIRTUAL_ENV/bin/activate"
fi

# Profile based on backend
case "$BACKEND" in
    cuda)
        # CUDA profiling with Nsight Compute
        echo "Using NVIDIA Nsight Compute for CUDA profiling..."
        
        ncu_path=$(which ncu 2>/dev/null)
        if [ -z "$ncu_path" ]; then
            echo "Error: ncu (NVIDIA Nsight Compute) not found"
            echo "Install from: https://developer.nvidia.com/nsight-compute"
            exit 1
        fi
        
        sudo env \
            YIRAGE_HOME="$YIRAGE_HOME" \
            TMPDIR="./ncu_tmp" \
            VIRTUAL_ENV="$VIRTUAL_ENV" \
            PATH="$VIRTUAL_ENV/bin:$PATH" \
            LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
            "$ncu_path" \
                --set full \
                --force-overwrite \
                --target-processes all \
                --kill no \
                --filter-mode global \
                --cache-control all \
                --clock-control base \
                --profile-from-start yes \
                --launch-count 1 \
                --apply-rules yes \
                --import-source no \
                --check-exit-code yes \
                --section MemoryWorkloadAnalysis_Chart \
                --section MemoryWorkloadAnalysis_Tables \
                --metrics "group:memory__chart,group:memory__shared_table" \
                --export "$REPORT_FILE" \
                -- "${TARGET_CMD[@]}"
        
        echo "CUDA profiling complete. Report: $REPORT_FILE"
        echo "View with: ncu-ui $REPORT_FILE"
        ;;
        
    mps)
        # MPS profiling with Instruments (macOS)
        echo "Using Instruments for MPS (Apple Silicon) profiling..."
        
        if [[ "$(uname)" != "Darwin" ]]; then
            echo "Error: MPS profiling requires macOS"
            exit 1
        fi
        
        # Check for xctrace (part of Xcode Command Line Tools)
        xctrace_path=$(which xctrace 2>/dev/null)
        if [ -z "$xctrace_path" ]; then
            echo "Error: xctrace not found. Install Xcode Command Line Tools:"
            echo "  xcode-select --install"
            exit 1
        fi
        
        # Use xctrace for automated profiling
        echo "Recording Metal/GPU activity..."
        "$xctrace_path" record \
            --template 'Metal Application' \
            --output "$REPORT_FILE" \
            --time-limit 30s \
            --launch -- "${TARGET_CMD[@]}"
        
        echo "MPS profiling complete. Report: $REPORT_FILE"
        echo "View with: open $REPORT_FILE"
        echo ""
        echo "Or use Instruments manually:"
        echo "  1. Open Instruments.app"
        echo "  2. Choose 'Metal Application' template"
        echo "  3. Run: ${TARGET_CMD[@]}"
        ;;
        
    cpu)
        # CPU profiling
        echo "CPU performance profiling..."
        
        if [[ "$(uname)" == "Darwin" ]]; then
            # macOS: Use Instruments
            echo "Using Instruments for CPU profiling (macOS)..."
            
            xctrace_path=$(which xctrace 2>/dev/null)
            if [ -z "$xctrace_path" ]; then
                echo "Warning: xctrace not found. Using Python timing fallback..."
                # Fallback: Simple timing
                /usr/bin/time -p "${TARGET_CMD[@]}" 2>&1 | tee "$REPORT_FILE.txt"
            else
                "$xctrace_path" record \
                    --template 'CPU Profiler' \
                    --output "$REPORT_FILE" \
                    --time-limit 30s \
                    --launch -- "${TARGET_CMD[@]}"
                
                echo "CPU profiling complete. Report: $REPORT_FILE"
                echo "View with: open $REPORT_FILE"
            fi
        else
            # Linux: Use perf
            echo "Using perf for CPU profiling (Linux)..."
            
            perf_path=$(which perf 2>/dev/null)
            if [ -z "$perf_path" ]; then
                echo "Error: perf not found. Install with:"
                echo "  sudo apt-get install linux-tools-common"
                exit 1
            fi
            
            # Run perf record
            "$perf_path" record \
                -g \
                --call-graph dwarf \
                -o "$REPORT_FILE" \
                -- "${TARGET_CMD[@]}"
            
            # Generate perf report
            "$perf_path" report -i "$REPORT_FILE" > "${REPORT_FILE}.txt"
            
            echo "CPU profiling complete."
            echo "  Raw data: $REPORT_FILE"
            echo "  Report: ${REPORT_FILE}.txt"
            echo "View with: perf report -i $REPORT_FILE"
        fi
        ;;
        
    *)
        echo "Error: Unknown backend '$BACKEND'"
        echo "Supported backends: cuda, mps, cpu"
        exit 1
        ;;
esac

echo "========================================"
echo "Profiling complete!"
echo "========================================"

