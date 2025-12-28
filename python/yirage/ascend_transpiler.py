"""
Ascend Transpiler for YiRage

This module provides code generation for Huawei Ascend NPUs via multiple paths:
1. Triton path (RECOMMENDED): Uses BiSheng compiler for Triton code
2. Ascend C path: Native Ascend C code for 910B+ devices
3. TBE path: TensorBoost Engine for Ascend 910 compatibility

Design rationale:
- BiSheng compiler natively supports Triton, allowing reuse of existing transpiler
- This reduces maintenance burden and ensures parity with CUDA/Triton features
"""

import os
import sys
import tempfile
import subprocess
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Try to import C++ transpiler via Cython bindings
# When available, this provides faster transpilation
_CPP_TRANSPILER_AVAILABLE = False
_cpp_ascend_transpile = None
try:
    from .core import ascend_transpile as _cpp_ascend_transpile
    _CPP_TRANSPILER_AVAILABLE = True
except ImportError:
    pass


def use_cpp_transpiler() -> bool:
    """Check if C++ transpiler is available and should be used."""
    return _CPP_TRANSPILER_AVAILABLE and os.environ.get("YIRAGE_USE_PYTHON_TRANSPILER", "0") != "1"


class CodeGenPath(Enum):
    """Code generation paths for Ascend NPU."""
    TRITON = "triton"      # Preferred: Uses BiSheng compiler
    ASCEND_C = "ascend_c"  # Native Ascend C (910B+)
    TBE = "tbe"            # TensorBoost Engine (910)


class AscendDeviceType(Enum):
    """Ascend NPU device types."""
    ASCEND_910 = 0
    ASCEND_910B = 1
    ASCEND_310P = 2


@dataclass
class AscendTranspileConfig:
    """Configuration for Ascend transpilation."""
    device_type: AscendDeviceType = AscendDeviceType.ASCEND_910B
    codegen_path: CodeGenPath = CodeGenPath.TRITON
    use_cube_ops: bool = True
    enable_fusion: bool = True
    ai_cores_per_block: int = 8
    optimization_level: int = 3
    enable_fp16: bool = True
    enable_bf16: bool = False
    verbose: bool = False


@dataclass
class AscendTranspileResult:
    """Result of Ascend transpilation."""
    success: bool
    code: str
    compile_command: str
    path_used: CodeGenPath
    output_shapes: List[Tuple[int, ...]] = field(default_factory=list)
    error_message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


def detect_ascend_environment() -> Dict[str, Any]:
    """
    Detect Ascend NPU environment and available tools.
    
    Returns:
        Dictionary with detected environment information
    """
    env = {
        "cann_available": False,
        "cann_path": None,
        "bisheng_available": False,
        "bisheng_path": None,
        "ascendc_available": False,
        "ascendc_path": None,
        "torch_npu_available": False,
        "device_count": 0,
        "device_type": None,
    }
    
    # Check CANN installation
    cann_path = os.environ.get("CANN_HOME") or os.environ.get("ASCEND_HOME")
    if not cann_path:
        cann_path = "/usr/local/Ascend/ascend-toolkit/latest"
    
    if os.path.exists(cann_path):
        env["cann_available"] = True
        env["cann_path"] = cann_path
        
        # Check for BiSheng compiler (multiple possible locations)
        bisheng_paths = [
            os.path.join(cann_path, "ccec_compiler/bin/bisheng"),
            os.path.join(cann_path, "aarch64-linux/ccec_compiler/bin/bisheng"),
            os.path.join(cann_path, "x86_64-linux/ccec_compiler/bin/bisheng"),
            os.path.join(cann_path, "compiler/bin/bisheng"),
        ]
        for bisheng_path in bisheng_paths:
            if os.path.exists(bisheng_path):
                env["bisheng_available"] = True
                env["bisheng_path"] = bisheng_path
                break
        
        # Check for Ascend C compiler (opc = operator compiler)
        ascendc_paths = [
            os.path.join(cann_path, "compiler/bin/opc"),
            os.path.join(cann_path, "aarch64-linux/bin/opc"),
            os.path.join(cann_path, "x86_64-linux/bin/opc"),
            os.path.join(cann_path, "compiler/bin/ascendc"),
        ]
        for ascendc_path in ascendc_paths:
            if os.path.exists(ascendc_path):
                env["ascendc_available"] = True
                env["ascendc_path"] = ascendc_path
                break
    
    # Check torch_npu (skip if YIRAGE_SKIP_TORCH_NPU is set to avoid initialization issues)
    if os.environ.get("YIRAGE_SKIP_TORCH_NPU", "0") != "1":
        try:
            import torch_npu
            env["torch_npu_available"] = True
            if torch_npu.npu.is_available():
                env["device_count"] = torch_npu.npu.device_count()
                # Detect device type from SOC name
                try:
                    soc_name = torch_npu.npu.get_device_name(0)
                    if "910B" in soc_name:
                        env["device_type"] = AscendDeviceType.ASCEND_910B
                    elif "910" in soc_name:
                        env["device_type"] = AscendDeviceType.ASCEND_910
                    elif "310P" in soc_name:
                        env["device_type"] = AscendDeviceType.ASCEND_310P
                except Exception:
                    # Device name query may fail in some environments
                    pass
        except ImportError:
            pass
        except Exception:
            # torch_npu may be installed but fail to initialize
            env["torch_npu_available"] = False
    
    return env


def get_recommended_config() -> AscendTranspileConfig:
    """
    Get recommended transpilation configuration based on environment.
    
    Returns:
        AscendTranspileConfig with optimal settings
    """
    env = detect_ascend_environment()
    
    config = AscendTranspileConfig()
    
    # Set device type
    if env["device_type"]:
        config.device_type = env["device_type"]
    
    # Choose code generation path
    if env["bisheng_available"]:
        # BiSheng supports Triton - use Triton path
        config.codegen_path = CodeGenPath.TRITON
    elif env["ascendc_available"]:
        # Use Ascend C for 910B+
        if config.device_type in [AscendDeviceType.ASCEND_910B, AscendDeviceType.ASCEND_310P]:
            config.codegen_path = CodeGenPath.ASCEND_C
        else:
            config.codegen_path = CodeGenPath.TBE
    else:
        # Fallback to TBE
        config.codegen_path = CodeGenPath.TBE
    
    # Enable BF16 for 910B+
    if config.device_type == AscendDeviceType.ASCEND_910B:
        config.enable_bf16 = True
    
    return config


def generate_triton_kernel_for_ascend(
    kernel_spec: Dict[str, Any],
    config: AscendTranspileConfig
) -> str:
    """
    Generate Triton kernel code for Ascend NPU.
    
    The generated code uses standard Triton syntax but with
    Ascend-specific optimizations hints.
    
    Args:
        kernel_spec: Kernel specification from YiRage graph
        config: Transpilation configuration
    
    Returns:
        Triton Python code as string
    """
    code_lines = [
        "# Generated by YiRage Ascend Transpiler",
        "# Target: Huawei Ascend NPU via BiSheng Compiler",
        f"# Device: {config.device_type.name}",
        f"# Code Generation Path: {config.codegen_path.value}",
        "",
        "import triton",
        "import triton.language as tl",
        "",
    ]
    
    # Add Ascend-specific meta-parameters
    code_lines.extend([
        "# Ascend NPU Optimization Hints",
        f"ASCEND_AI_CORES = {config.ai_cores_per_block}",
        f"ASCEND_USE_CUBE = {config.use_cube_ops}",
        f"ASCEND_OPT_LEVEL = {config.optimization_level}",
        "",
    ])
    
    # Generate kernel based on specification
    kernel_type = kernel_spec.get("type", "matmul")
    
    if kernel_type == "matmul":
        code_lines.extend(_generate_triton_matmul_kernel(kernel_spec, config))
    elif kernel_type == "elementwise":
        code_lines.extend(_generate_triton_elementwise_kernel(kernel_spec, config))
    elif kernel_type == "reduction":
        code_lines.extend(_generate_triton_reduction_kernel(kernel_spec, config))
    elif kernel_type == "attention":
        code_lines.extend(_generate_triton_attention_kernel(kernel_spec, config))
    else:
        # Generic kernel template
        code_lines.extend(_generate_triton_generic_kernel(kernel_spec, config))
    
    return "\n".join(code_lines)


def _generate_triton_matmul_kernel(spec: Dict, config: AscendTranspileConfig) -> List[str]:
    """Generate Triton matmul kernel for Ascend."""
    M = spec.get("M", "M")
    N = spec.get("N", "N")
    K = spec.get("K", "K")
    
    # Block sizes optimized for Ascend Cube unit (16x16 native)
    block_m = 64 if config.use_cube_ops else 32
    block_n = 64 if config.use_cube_ops else 32
    block_k = 32
    
    return [
        "@triton.jit",
        "def matmul_kernel(",
        "    A_ptr, B_ptr, C_ptr,",
        f"    M, N, K,",
        "    stride_am, stride_ak,",
        "    stride_bk, stride_bn,",
        "    stride_cm, stride_cn,",
        f"    BLOCK_M: tl.constexpr = {block_m},",
        f"    BLOCK_N: tl.constexpr = {block_n},",
        f"    BLOCK_K: tl.constexpr = {block_k},",
        "):",
        "    # Ascend Cube-optimized matmul",
        "    pid_m = tl.program_id(0)",
        "    pid_n = tl.program_id(1)",
        "",
        "    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)",
        "    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)",
        "    offs_k = tl.arange(0, BLOCK_K)",
        "",
        "    # Initialize accumulator",
        "    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)",
        "",
        "    # Main loop over K dimension",
        "    for k in range(0, K, BLOCK_K):",
        "        # Load A and B tiles",
        "        a_ptrs = A_ptr + offs_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak",
        "        b_ptrs = B_ptr + (k + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn",
        "        ",
        "        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & ((k + offs_k[None, :]) < K))",
        "        b = tl.load(b_ptrs, mask=((k + offs_k[:, None]) < K) & (offs_n[None, :] < N))",
        "        ",
        "        # Accumulate (Cube unit handles this efficiently)",
        "        acc += tl.dot(a, b)",
        "",
        "    # Store result",
        "    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn",
        "    tl.store(c_ptrs, acc.to(tl.float16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))",
        "",
    ]


def _generate_triton_elementwise_kernel(spec: Dict, config: AscendTranspileConfig) -> List[str]:
    """Generate Triton elementwise kernel for Ascend."""
    op = spec.get("op", "add")
    
    op_map = {
        "add": "x + y",
        "mul": "x * y",
        "div": "x / y",
        "sub": "x - y",
        "silu": "x * tl.sigmoid(x)",
        "gelu": "x * 0.5 * (1.0 + tl.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))",
        "relu": "tl.maximum(x, 0)",
    }
    
    op_expr = op_map.get(op, "x")
    is_binary = op in ["add", "mul", "div", "sub"]
    
    lines = [
        "@triton.jit",
        "def elementwise_kernel(",
        "    X_ptr,",
    ]
    
    if is_binary:
        lines.append("    Y_ptr,")
    
    lines.extend([
        "    OUT_ptr,",
        "    n_elements,",
        "    BLOCK_SIZE: tl.constexpr = 1024,",
        "):",
        f"    # Ascend Vector-optimized {op} kernel",
        "    pid = tl.program_id(0)",
        "    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)",
        "    mask = offs < n_elements",
        "",
        "    x = tl.load(X_ptr + offs, mask=mask)",
    ])
    
    if is_binary:
        lines.extend([
            "    y = tl.load(Y_ptr + offs, mask=mask)",
            f"    out = {op_expr}",
        ])
    else:
        lines.append(f"    out = {op_expr}")
    
    lines.extend([
        "    tl.store(OUT_ptr + offs, out, mask=mask)",
        "",
    ])
    
    return lines


def _generate_triton_reduction_kernel(spec: Dict, config: AscendTranspileConfig) -> List[str]:
    """Generate Triton reduction kernel for Ascend."""
    return [
        "@triton.jit",
        "def reduction_kernel(",
        "    X_ptr, OUT_ptr,",
        "    n_rows, n_cols,",
        "    stride_x,",
        "    BLOCK_SIZE: tl.constexpr = 1024,",
        "):",
        "    # Ascend Vector reduction kernel",
        "    row_idx = tl.program_id(0)",
        "    offs = tl.arange(0, BLOCK_SIZE)",
        "    ",
        "    # Load row and sum",
        "    acc = tl.zeros([1], dtype=tl.float32)",
        "    for i in range(0, n_cols, BLOCK_SIZE):",
        "        mask = (i + offs) < n_cols",
        "        x = tl.load(X_ptr + row_idx * stride_x + i + offs, mask=mask)",
        "        acc += tl.sum(x, axis=0)",
        "    ",
        "    tl.store(OUT_ptr + row_idx, acc)",
        "",
    ]


def _generate_triton_attention_kernel(spec: Dict, config: AscendTranspileConfig) -> List[str]:
    """Generate Triton attention kernel for Ascend."""
    return [
        "@triton.jit",
        "def attention_kernel(",
        "    Q_ptr, K_ptr, V_ptr, OUT_ptr,",
        "    batch_size, num_heads, seq_len, head_dim,",
        "    stride_qb, stride_qh, stride_qs, stride_qd,",
        "    stride_kb, stride_kh, stride_ks, stride_kd,",
        "    stride_vb, stride_vh, stride_vs, stride_vd,",
        "    stride_ob, stride_oh, stride_os, stride_od,",
        "    BLOCK_M: tl.constexpr = 64,",
        "    BLOCK_N: tl.constexpr = 64,",
        "    BLOCK_D: tl.constexpr = 64,",
        "):",
        "    # Ascend Cube-optimized attention kernel",
        "    pid_b = tl.program_id(0)",
        "    pid_h = tl.program_id(1)",
        "    pid_m = tl.program_id(2)",
        "    ",
        "    # Compute attention for this block",
        "    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)",
        "    offs_d = tl.arange(0, BLOCK_D)",
        "    ",
        "    # Initialize output accumulator",
        "    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)",
        "    l = tl.zeros((BLOCK_M,), dtype=tl.float32)",
        "    m = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)",
        "    ",
        "    # Scale factor",
        "    scale = 1.0 / tl.sqrt(float(head_dim))",
        "    ",
        "    # Load Q tile (stays in registers)",
        "    q_ptrs = Q_ptr + pid_b * stride_qb + pid_h * stride_qh + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd",
        "    q = tl.load(q_ptrs, mask=(offs_m[:, None] < seq_len) & (offs_d[None, :] < head_dim))",
        "    ",
        "    # Iterate over K/V blocks",
        "    for n in range(0, seq_len, BLOCK_N):",
        "        offs_n = n + tl.arange(0, BLOCK_N)",
        "        ",
        "        # Load K tile",
        "        k_ptrs = K_ptr + pid_b * stride_kb + pid_h * stride_kh + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd",
        "        k = tl.load(k_ptrs, mask=(offs_n[:, None] < seq_len) & (offs_d[None, :] < head_dim))",
        "        ",
        "        # Compute QK^T (Cube unit)",
        "        qk = tl.dot(q, tl.trans(k)) * scale",
        "        ",
        "        # Online softmax",
        "        m_new = tl.maximum(m, tl.max(qk, axis=1))",
        "        p = tl.exp(qk - m_new[:, None])",
        "        l_new = tl.exp(m - m_new) * l + tl.sum(p, axis=1)",
        "        ",
        "        # Load V tile",
        "        v_ptrs = V_ptr + pid_b * stride_vb + pid_h * stride_vh + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd",
        "        v = tl.load(v_ptrs, mask=(offs_n[:, None] < seq_len) & (offs_d[None, :] < head_dim))",
        "        ",
        "        # Update accumulator",
        "        acc = acc * (tl.exp(m - m_new)[:, None]) + tl.dot(p.to(v.dtype), v)",
        "        m = m_new",
        "        l = l_new",
        "    ",
        "    # Normalize and store",
        "    acc = acc / l[:, None]",
        "    out_ptrs = OUT_ptr + pid_b * stride_ob + pid_h * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od",
        "    tl.store(out_ptrs, acc.to(tl.float16), mask=(offs_m[:, None] < seq_len) & (offs_d[None, :] < head_dim))",
        "",
    ]


def _generate_triton_generic_kernel(spec: Dict, config: AscendTranspileConfig) -> List[str]:
    """Generate generic Triton kernel placeholder."""
    return [
        "@triton.jit",
        "def generic_kernel(",
        "    # TODO: Auto-generate from kernel specification",
        "    pass",
        "):",
        "    pid = tl.program_id(0)",
        "    # Generic kernel body",
        "    pass",
        "",
    ]


def transpile_to_ascend(
    kernel_spec: Dict[str, Any],
    config: Optional[AscendTranspileConfig] = None
) -> AscendTranspileResult:
    """
    Transpile YiRage kernel specification to Ascend code.
    
    This function first attempts to use the C++ transpiler (via Cython bindings)
    for optimal performance. If C++ is not available, it falls back to the
    pure-Python implementation.
    
    Args:
        kernel_spec: Kernel specification from YiRage graph
        config: Optional transpilation configuration
    
    Returns:
        AscendTranspileResult with generated code
    """
    if config is None:
        config = get_recommended_config()
    
    # Try C++ transpiler first (faster, more optimized)
    if use_cpp_transpiler() and _cpp_ascend_transpile is not None:
        try:
            cpp_result = _cpp_ascend_transpile(
                kernel_spec,
                device_type=config.device_type.value,
                codegen_path=config.codegen_path.value,
                opt_level=config.optimization_level
            )
            return AscendTranspileResult(
                code=cpp_result.code,
                compile_command=cpp_result.compile_command,
                path_used=CodeGenPath(cpp_result.path_used),
                success=cpp_result.success,
                error_message=cpp_result.error_message if hasattr(cpp_result, 'error_message') else ""
            )
        except Exception as e:
            if config.verbose:
                print(f"C++ transpiler failed, falling back to Python: {e}", file=sys.stderr)
    
    # Python fallback implementation
    env = detect_ascend_environment()
    
    # Generate code based on path
    if config.codegen_path == CodeGenPath.TRITON:
        code = generate_triton_kernel_for_ascend(kernel_spec, config)
        
        # Generate compilation command
        if env["bisheng_available"]:
            soc_version = {
                AscendDeviceType.ASCEND_910: "Ascend910",
                AscendDeviceType.ASCEND_910B: "Ascend910B",
                AscendDeviceType.ASCEND_310P: "Ascend310P",
            }.get(config.device_type, "Ascend910B")
            
            compile_cmd = (
                f"{env['bisheng_path']} "
                f"--target={soc_version} "
                f"--opt-level={config.optimization_level} "
                f"{'--enable-fp16' if config.enable_fp16 else ''} "
                f"{'--enable-bf16' if config.enable_bf16 else ''} "
                "-o kernel.so"
            )
        else:
            compile_cmd = "# BiSheng compiler not available"
        
        return AscendTranspileResult(
            success=True,
            code=code,
            compile_command=compile_cmd,
            path_used=CodeGenPath.TRITON,
            metadata={"env": env}
        )
    
    else:
        # TODO: Implement Ascend C and TBE paths
        return AscendTranspileResult(
            success=False,
            code="",
            compile_command="",
            path_used=config.codegen_path,
            error_message=f"Code generation path {config.codegen_path.value} not yet implemented"
        )


def compile_ascend_kernel(
    result: AscendTranspileResult,
    output_dir: str,
) -> Tuple[bool, str]:
    """
    Compile transpiled Ascend kernel.
    
    Args:
        result: Transpilation result
        output_dir: Directory for output files
    
    Returns:
        Tuple of (success, output_path_or_error)
    """
    if not result.success:
        return False, result.error_message
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Write source file
    source_ext = ".py" if result.path_used == CodeGenPath.TRITON else ".cpp"
    source_path = os.path.join(output_dir, f"kernel{source_ext}")
    
    with open(source_path, 'w') as f:
        f.write(result.code)
    
    # For Triton path, we use JIT compilation
    if result.path_used == CodeGenPath.TRITON:
        return True, source_path
    
    # For other paths, invoke compiler
    try:
        subprocess.check_call(
            result.compile_command.split(),
            cwd=output_dir,
            timeout=300
        )
        return True, os.path.join(output_dir, "kernel.so")
    except subprocess.CalledProcessError as e:
        return False, str(e)
    except FileNotFoundError:
        return False, "Compiler not found"
