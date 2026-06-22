# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0

"""
YiRage MLIR JIT Compiler

High-level Python API for JIT compilation of kernels to GPU code.

Example:
    import yirage
    from yirage.mlir_jit import jit, Target
    
    @jit(target=Target.CUDA_H100)
    def my_kernel(x, y):
        return yirage.matmul(x, y)
    
    # Or with explicit compilation
    from yirage.mlir_jit import MLIRCompiler
    
    compiler = MLIRCompiler(target=Target.CUDA_H100)
    ptx = compiler.to_ptx(module_text)
    cubin = compiler.to_cubin(module_text)
"""

from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
import hashlib
import os
import sys
from pathlib import Path

# Project root for finding build artifacts
PROJECT_ROOT = Path(__file__).parents[3] if len(Path(__file__).parents) > 3 else Path(__file__).parent

# Import MLIR components
_NATIVE_MLIR = False
try:
    from yirage._yirage_mlir import (
        GPUBackend,
        GPUTargetConfig,
        MLIRContext,
        parseMLIR,
        printMLIR,
        generatePTX,
        generateROCm,
        generateSPIRV,
        generateMetal,
        generateCubin,
        generateHSACO,
        generateSPIRVBinary,
        runGPUPipeline,
        runCUDAPipeline,
        runROCmPipeline,
        runCPUPipeline,
        runCustomPipeline,
    )
    _NATIVE_MLIR = True
except ImportError:
    pass

# Import pure Python MLIR API as fallback
_PYTHON_MLIR = False
try:
    _mlir_path = Path(__file__).parent.parent.parent / "mlir" / "python"
    if _mlir_path.exists():
        sys.path.insert(0, str(_mlir_path))
    from yirage_mlir import YirageModule, Target as MLIRTarget, CompiledKernel
    _PYTHON_MLIR = True
except ImportError:
    pass


#==============================================================================
# Target Configuration
#==============================================================================

class Target(Enum):
    """Compilation targets."""
    # NVIDIA CUDA
    CUDA_V100 = ("cuda", "sm_70", 70)
    CUDA_A100 = ("cuda", "sm_80", 80)
    CUDA_H100 = ("cuda", "sm_90", 90)
    CUDA = ("cuda", "sm_80", 80)
    
    # AMD ROCm
    ROCM_MI100 = ("rocm", "gfx908", None)
    ROCM_MI250 = ("rocm", "gfx90a", None)
    ROCM_MI300X = ("rocm", "gfx942", None)
    ROCM = ("rocm", "gfx908", None)
    
    # Intel XPU
    XPU = ("spirv", "spirv64", None)
    
    # Apple Metal
    METAL_M1 = ("metal", "apple-m1", None)
    METAL_M2 = ("metal", "apple-m2", None)
    METAL_M3 = ("metal", "apple-m3", None)
    METAL_M4 = ("metal", "apple-m4", None)
    METAL_M5 = ("metal", "apple-m5", None)
    METAL = ("metal", "apple-m1", None)
    
    # CPU
    CPU_AVX2 = ("cpu", "x86-64-v3", None)
    CPU_AVX512 = ("cpu", "x86-64-v4", None)
    CPU_NEON = ("cpu", "aarch64", None)
    CPU_SVE = ("cpu", "aarch64+sve", None)
    CPU = ("cpu", "x86-64", None)
    
    @property
    def backend(self) -> str:
        return self.value[0]
    
    @property
    def arch(self) -> str:
        return self.value[1]
    
    @property
    def compute_capability(self) -> Optional[int]:
        return self.value[2]


@dataclass
class CompileOptions:
    """Compilation options."""
    opt_level: int = 3
    fast_math: bool = True
    use_fma: bool = True
    debug: bool = False
    
    # Memory optimization
    max_shared_memory: int = 0  # 0 = auto
    max_registers: int = 0
    
    # Code generation
    unroll_loops: bool = True
    vectorize: bool = True
    
    # Caching
    enable_cache: bool = True
    cache_dir: Optional[str] = None


#==============================================================================
# Compilation Result
#==============================================================================

@dataclass
class CompilationResult:
    """Result of MLIR compilation."""
    success: bool
    target: Target
    
    # Generated code
    mlir_source: str = ""
    lowered_mlir: str = ""
    target_code: str = ""
    binary: bytes = b""
    
    # Metadata
    kernel_names: List[str] = field(default_factory=list)
    register_usage: int = 0
    shared_memory_usage: int = 0
    
    # Error handling
    error_message: str = ""
    
    def __bool__(self) -> bool:
        return self.success


#==============================================================================
# Compilation Cache
#==============================================================================

class CompilationCache:
    """Cache for compiled kernels."""
    
    _instance: Optional['CompilationCache'] = None
    
    def __init__(self, cache_dir: Optional[str] = None):
        if cache_dir is None:
            cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "yirage", "mlir")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: Dict[str, CompilationResult] = {}
    
    @classmethod
    def get_instance(cls) -> 'CompilationCache':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def _compute_key(self, source: str, target: Target, options: CompileOptions) -> str:
        key_data = f"{source}:{target.name}:{options.opt_level}:{options.fast_math}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]
    
    def get(self, source: str, target: Target, options: CompileOptions) -> Optional[CompilationResult]:
        key = self._compute_key(source, target, options)
        
        # Check memory cache first
        if key in self._memory_cache:
            return self._memory_cache[key]
        
        # Check disk cache
        cache_file = self.cache_dir / f"{key}.bin"
        if cache_file.exists():
            try:
                import pickle
                with open(cache_file, "rb") as f:
                    result = pickle.load(f)
                self._memory_cache[key] = result
                return result
            except Exception:
                pass
        
        return None
    
    def put(self, source: str, target: Target, options: CompileOptions, result: CompilationResult):
        key = self._compute_key(source, target, options)
        self._memory_cache[key] = result
        
        # Save to disk
        cache_file = self.cache_dir / f"{key}.bin"
        try:
            import pickle
            with open(cache_file, "wb") as f:
                pickle.dump(result, f)
        except Exception:
            pass
    
    def clear(self):
        self._memory_cache.clear()
        import shutil
        shutil.rmtree(self.cache_dir, ignore_errors=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)


#==============================================================================
# MLIR Compiler
#==============================================================================

class MLIRCompiler:
    """
    High-level MLIR compiler for YiRage.
    
    Example:
        compiler = MLIRCompiler(target=Target.CUDA_H100)
        
        mlir = '''
        func.func @matmul(%a: tensor<1024x1024xf16>, %b: tensor<1024x1024xf16>) 
            -> tensor<1024x1024xf16> {
            %c = yirage.matmul %a, %b : tensor<1024x1024xf16>, tensor<1024x1024xf16> 
                -> tensor<1024x1024xf16>
            return %c : tensor<1024x1024xf16>
        }
        '''
        
        result = compiler.compile(mlir)
        print(result.target_code)  # PTX
    """
    
    def __init__(self, 
                 target: Target = Target.CUDA,
                 options: Optional[CompileOptions] = None):
        self.target = target
        self.options = options or CompileOptions()
        self._cache = CompilationCache.get_instance() if self.options.enable_cache else None
        
        # Initialize native MLIR context if available
        self._context = None
        if _NATIVE_MLIR:
            self._context = MLIRContext()
            self._context.loadAllDialects()
    
    def compile(self, mlir_source: str) -> CompilationResult:
        """Compile MLIR source to target code."""
        
        # Check cache
        if self._cache:
            cached = self._cache.get(mlir_source, self.target, self.options)
            if cached:
                return cached
        
        result = self._compile_impl(mlir_source)
        
        # Cache result
        if self._cache and result.success:
            self._cache.put(mlir_source, self.target, self.options, result)
        
        return result
    
    def _compile_impl(self, mlir_source: str) -> CompilationResult:
        """Implementation of compilation."""
        result = CompilationResult(
            success=False,
            target=self.target,
            mlir_source=mlir_source,
        )
        
        if _NATIVE_MLIR and self._context:
            return self._compile_native(mlir_source, result)
        elif _PYTHON_MLIR:
            return self._compile_python(mlir_source, result)
        else:
            result.error_message = "MLIR not available. Build with USE_MLIR=ON"
            return result
    
    def _compile_native(self, mlir_source: str, result: CompilationResult) -> CompilationResult:
        """Compile using native C++ bindings."""
        try:
            # Parse MLIR
            ctx_module = parseMLIR(mlir_source)
            if ctx_module is None:
                result.error_message = "Failed to parse MLIR"
                return result
            
            context, module = ctx_module
            
            # Run appropriate pipeline
            backend = self.target.backend
            if backend == "cuda":
                success = runCUDAPipeline(context, module)
                if success:
                    cc = self.target.compute_capability or 80
                    result.target_code = generatePTX(context, module, cc)
                    result.binary = generateCubin(context, module, cc)
            elif backend == "rocm":
                success = runROCmPipeline(context, module)
                if success:
                    arch = self.target.arch
                    result.target_code = generateROCm(context, module, arch)
                    result.binary = generateHSACO(context, module, arch)
            elif backend == "spirv":
                success = runGPUPipeline(context, module)
                if success:
                    result.target_code = generateSPIRV(context, module)
                    result.binary = generateSPIRVBinary(context, module)
            elif backend == "metal":
                success = runGPUPipeline(context, module)
                if success:
                    result.target_code = generateMetal(context, module)
            elif backend == "cpu":
                success = runCPUPipeline(context, module)
                if success:
                    result.target_code = printMLIR(module)
            else:
                result.error_message = f"Unsupported backend: {backend}"
                return result
            
            result.success = success
            if not success:
                result.error_message = "Pipeline execution failed"
            
        except Exception as e:
            result.error_message = str(e)
        
        return result
    
    def _compile_python(self, mlir_source: str, result: CompilationResult) -> CompilationResult:
        """Compile using pure Python MLIR API."""
        try:
            # Use yirage-opt tool if available
            import subprocess
            import tempfile
            
            # Write MLIR to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.mlir', delete=False) as f:
                f.write(mlir_source)
                input_path = f.name
            
            # Run yirage-opt with appropriate pipeline
            # Pipeline naming convention: yirage-{backend}-pipeline
            # Special case: 'spirv' uses generic GPU pipeline
            backend = self.target.backend
            pipeline_mapping = {
                "cuda": "yirage-cuda-pipeline",
                "rocm": "yirage-rocm-pipeline",
                "spirv": "yirage-gpu-pipeline",
                "metal": "yirage-mps-pipeline",
                "cpu": "yirage-cpu-pipeline",
            }
            pipeline = pipeline_mapping.get(backend, f"yirage-{backend}-pipeline")
            
            yirage_opt = self._find_yirage_opt()
            if yirage_opt:
                cmd = [yirage_opt, input_path, f"--{pipeline}"]
                proc = subprocess.run(cmd, capture_output=True, text=True)
                
                if proc.returncode == 0:
                    result.success = True
                    result.lowered_mlir = proc.stdout
                    result.target_code = proc.stdout  # Lowered MLIR
                else:
                    result.error_message = proc.stderr
            else:
                result.error_message = "yirage-opt not found. Build the MLIR tools."
            
            os.unlink(input_path)
            
        except Exception as e:
            result.error_message = str(e)
        
        return result
    
    def _find_yirage_opt(self) -> Optional[str]:
        """Find yirage-opt executable."""
        import shutil
        
        # Check PATH
        path = shutil.which("yirage-opt")
        if path:
            return path
        
        # Check build directory using PROJECT_ROOT
        build_paths = [
            PROJECT_ROOT / "build" / "mlir" / "yirage-opt",
            PROJECT_ROOT / "build" / "bin" / "yirage-opt",
            PROJECT_ROOT / "build-mlir" / "mlir" / "yirage-opt",
        ]
        for p in build_paths:
            if p.exists():
                return str(p)
        
        return None
    
    def to_ptx(self, mlir_source: str) -> str:
        """Compile to PTX (CUDA only)."""
        if self.target.backend != "cuda":
            raise ValueError("PTX is only available for CUDA targets")
        result = self.compile(mlir_source)
        if not result.success:
            raise RuntimeError(f"Compilation failed: {result.error_message}")
        return result.target_code
    
    def to_cubin(self, mlir_source: str) -> bytes:
        """Compile to cubin binary (CUDA only)."""
        if self.target.backend != "cuda":
            raise ValueError("cubin is only available for CUDA targets")
        result = self.compile(mlir_source)
        if not result.success:
            raise RuntimeError(f"Compilation failed: {result.error_message}")
        return result.binary
    
    def to_rocm(self, mlir_source: str) -> str:
        """Compile to GCN assembly (ROCm only)."""
        if self.target.backend != "rocm":
            raise ValueError("GCN is only available for ROCm targets")
        result = self.compile(mlir_source)
        if not result.success:
            raise RuntimeError(f"Compilation failed: {result.error_message}")
        return result.target_code
    
    def to_spirv(self, mlir_source: str) -> str:
        """Compile to SPIR-V text."""
        result = self.compile(mlir_source)
        if not result.success:
            raise RuntimeError(f"Compilation failed: {result.error_message}")
        return result.target_code


#==============================================================================
# JIT Decorator
#==============================================================================

def jit(target: Target = Target.CUDA,
        options: Optional[CompileOptions] = None) -> Callable:
    """
    JIT compilation decorator for YiRage kernels.
    
    Example:
        @jit(target=Target.CUDA_H100)
        def matmul(x, y):
            return yirage.matmul(x, y)
        
        result = matmul(a, b)
    """
    compiler = MLIRCompiler(target=target, options=options)
    
    def decorator(func: Callable) -> Callable:
        _compiled_kernel = None
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal _compiled_kernel
            
            # For now, just call the original function
            # Full JIT would trace the function and compile
            return func(*args, **kwargs)
        
        # Attach compiler for inspection
        wrapper._compiler = compiler
        wrapper._target = target
        
        return wrapper
    
    return decorator


#==============================================================================
# Convenience Functions
#==============================================================================

def compile_mlir(mlir_source: str, 
                 target: Target = Target.CUDA) -> CompilationResult:
    """Compile MLIR source to target code."""
    compiler = MLIRCompiler(target=target)
    return compiler.compile(mlir_source)


def clear_compile_cache():
    """Clear the compilation cache."""
    CompilationCache.get_instance().clear()


def is_mlir_available() -> bool:
    """Check if MLIR compilation is available."""
    return _NATIVE_MLIR or _PYTHON_MLIR


__all__ = [
    'Target',
    'CompileOptions',
    'CompilationResult',
    'MLIRCompiler',
    'CompilationCache',
    'jit',
    'compile_mlir',
    'clear_compile_cache',
    'is_mlir_available',
]
