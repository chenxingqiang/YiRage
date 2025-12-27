"""
Multi-Backend Persistent Kernel for YiRage

This module provides a unified interface for persistent kernels across
multiple hardware backends (CUDA, MACA, Ascend, CPU).

Design Principles:
1. Common API regardless of backend
2. Automatic backend selection based on hardware
3. Fallback chain for graceful degradation
4. JIT compilation with caching
"""

import os
import sys
import torch
import tempfile
import importlib.util
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Callable
from enum import Enum

from .backend_compiler import (
    CompilerBackend,
    CompileConfig,
    CompileResult,
    CompilerFactory,
    compile_kernel,
    get_target_arch_from_device,
)
from .backend_api import get_available_backends, get_default_backend


class KernelBackend(Enum):
    """Supported kernel execution backends."""
    CUDA = "cuda"
    MACA = "maca"
    ASCEND = "ascend"
    CPU = "cpu"
    TRITON = "triton"


@dataclass
class KernelConfig:
    """Configuration for multi-backend kernel."""
    backend: KernelBackend = KernelBackend.CUDA
    fallback_backends: List[KernelBackend] = field(default_factory=list)
    target_arch: str = "native"
    optimization_level: int = 3
    enable_profiling: bool = False
    cache_compiled: bool = True
    cache_dir: Optional[str] = None
    world_size: int = 1
    rank: int = 0


@dataclass
class KernelExecutionContext:
    """Context for kernel execution."""
    input_tensors: List[torch.Tensor]
    output_tensors: List[torch.Tensor]
    stream: Optional[Any] = None
    profiler_buffer: Optional[torch.Tensor] = None


class BackendKernel(ABC):
    """Abstract base class for backend-specific kernels."""
    
    @property
    @abstractmethod
    def backend(self) -> KernelBackend:
        """Return the kernel backend type."""
        pass
    
    @abstractmethod
    def compile(self, source_code: str, config: KernelConfig) -> CompileResult:
        """Compile kernel source code."""
        pass
    
    @abstractmethod
    def execute(self, ctx: KernelExecutionContext) -> None:
        """Execute the compiled kernel."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available."""
        pass


class CUDAKernel(BackendKernel):
    """CUDA backend kernel implementation."""
    
    def __init__(self):
        self._launcher = None
        self._buffer = None
    
    @property
    def backend(self) -> KernelBackend:
        return KernelBackend.CUDA
    
    def is_available(self) -> bool:
        return torch.cuda.is_available()
    
    def compile(self, source_code: str, config: KernelConfig) -> CompileResult:
        from .kernel import get_key_paths, HARD_CODE
        
        YIRAGE_ROOT, INCLUDE_PATH, DEPS_PATH = get_key_paths()
        
        # Determine target architecture
        if config.target_arch == "native":
            target_arch = get_target_arch_from_device()
        else:
            target_arch = config.target_arch
        
        # Create compilation configuration
        compile_config = CompileConfig(
            source_code=source_code + HARD_CODE,
            output_dir=config.cache_dir or tempfile.mkdtemp(),
            target_arch=target_arch,
            optimization_level=config.optimization_level,
            debug_mode=False,
            profiling=config.enable_profiling,
            include_dirs=[
                os.path.join(INCLUDE_PATH, 'yirage/transpiler/runtime'),
                os.path.join(DEPS_PATH, 'cutlass/include'),
            ],
            defines={"YIRAGE_BACKEND_USE_CUDA": "1"},
            libraries=["cublas"],
        )
        
        # Get CUDA compiler
        compiler = CompilerFactory.get_compiler(CompilerBackend.NVCC)
        result = compiler.compile(compile_config)
        
        if result.success:
            # Load the compiled module
            spec = importlib.util.spec_from_file_location(
                "__yirage_launcher", result.output_path
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self._launcher = getattr(mod, "launch")
        
        return result
    
    def execute(self, ctx: KernelExecutionContext) -> None:
        if self._launcher is None:
            raise RuntimeError("Kernel not compiled")
        
        # Ensure buffer is allocated
        if self._buffer is None:
            # Default buffer size (can be made configurable)
            self._buffer = torch.empty(
                1024 * 1024,  # 1MB buffer
                dtype=torch.uint8,
                device=ctx.input_tensors[0].device
            )
        
        # Get stream
        stream = ctx.stream
        if stream is None:
            stream = torch.cuda.default_stream()
        
        # Prepare pointers
        input_ptrs = [t.data_ptr() for t in ctx.input_tensors]
        output_ptrs = [t.data_ptr() for t in ctx.output_tensors]
        buffer_ptr = self._buffer.data_ptr()
        profiler_ptr = ctx.profiler_buffer.data_ptr() if ctx.profiler_buffer else 0
        
        # Execute
        self._launcher(
            input_ptrs,
            output_ptrs,
            buffer_ptr,
            stream.cuda_stream,
            profiler_ptr
        )


class MACAKernel(BackendKernel):
    """MACA backend kernel implementation (MetaX GPU)."""
    
    def __init__(self):
        self._launcher = None
        self._buffer = None
    
    @property
    def backend(self) -> KernelBackend:
        return KernelBackend.MACA
    
    def is_available(self) -> bool:
        if not torch.cuda.is_available():
            return False
        try:
            device_name = torch.cuda.get_device_name(0)
            return any(x in device_name for x in ["MetaX", "C500", "MACA"])
        except:
            return False
    
    def compile(self, source_code: str, config: KernelConfig) -> CompileResult:
        # MACA uses CUDA-compatible API, can reuse CUDA compilation
        # with MACA-specific compiler (mxcc)
        compiler = CompilerFactory.get_compiler(CompilerBackend.MXCC)
        
        if not compiler.is_available():
            # Fall back to NVCC-style compilation (MACA is CUDA-compatible)
            compiler = CompilerFactory.get_compiler(CompilerBackend.NVCC)
        
        compile_config = CompileConfig(
            source_code=source_code,
            output_dir=config.cache_dir or tempfile.mkdtemp(),
            target_arch=config.target_arch,
            optimization_level=config.optimization_level,
            defines={
                "YIRAGE_BACKEND_MACA_ENABLED": "1",
                "MACA_WARP_SIZE": "64",  # MACA uses 64-thread warps
            },
        )
        
        result = compiler.compile(compile_config)
        
        if result.success:
            spec = importlib.util.spec_from_file_location(
                "__yirage_launcher", result.output_path
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self._launcher = getattr(mod, "launch", None)
        
        return result
    
    def execute(self, ctx: KernelExecutionContext) -> None:
        # MACA uses CUDA-compatible runtime
        if self._launcher is None:
            raise RuntimeError("Kernel not compiled")
        
        if self._buffer is None:
            self._buffer = torch.empty(
                1024 * 1024,
                dtype=torch.uint8,
                device=ctx.input_tensors[0].device
            )
        
        stream = ctx.stream
        if stream is None:
            stream = torch.cuda.default_stream()
        
        input_ptrs = [t.data_ptr() for t in ctx.input_tensors]
        output_ptrs = [t.data_ptr() for t in ctx.output_tensors]
        
        self._launcher(
            input_ptrs,
            output_ptrs,
            self._buffer.data_ptr(),
            stream.cuda_stream,
            0
        )


class AscendKernel(BackendKernel):
    """Ascend NPU backend kernel implementation."""
    
    def __init__(self):
        self._launcher = None
        self._triton_kernel = None
    
    @property
    def backend(self) -> KernelBackend:
        return KernelBackend.ASCEND
    
    def is_available(self) -> bool:
        try:
            import torch_npu
            return torch.npu.is_available()
        except ImportError:
            return False
    
    def compile(self, source_code: str, config: KernelConfig) -> CompileResult:
        from .ascend_transpiler import (
            transpile_to_ascend,
            AscendTranspileConfig,
            CodeGenPath,
        )
        
        # For Ascend, we prefer Triton path
        ascend_config = AscendTranspileConfig(
            codegen_path=CodeGenPath.TRITON,
            optimization_level=config.optimization_level,
        )
        
        # Generate kernel specification from source code
        kernel_spec = self._parse_kernel_spec(source_code)
        
        # Transpile to Triton for Ascend
        transpile_result = transpile_to_ascend(kernel_spec, ascend_config)
        
        if transpile_result.success:
            # For Triton path, we use JIT compilation
            # Save the Triton code and load it
            output_dir = config.cache_dir or tempfile.mkdtemp()
            kernel_path = os.path.join(output_dir, "kernel_ascend.py")
            
            with open(kernel_path, 'w') as f:
                f.write(transpile_result.code)
            
            # Load the Triton kernel
            spec = importlib.util.spec_from_file_location("kernel_ascend", kernel_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            
            # Get the kernel function
            for name in dir(mod):
                attr = getattr(mod, name)
                if hasattr(attr, '__triton_jit__'):
                    self._triton_kernel = attr
                    break
            
            return CompileResult(
                success=True,
                output_path=kernel_path,
                metadata={"triton_jit": True}
            )
        
        return CompileResult(
            success=False,
            output_path="",
            error_message=transpile_result.error_message
        )
    
    def _parse_kernel_spec(self, source_code: str) -> Dict[str, Any]:
        """Parse kernel specification from source code."""
        # Simple heuristic-based parsing
        spec = {"type": "generic"}
        
        if "matmul" in source_code.lower():
            spec["type"] = "matmul"
        elif "attention" in source_code.lower():
            spec["type"] = "attention"
        elif "silu" in source_code.lower() or "gelu" in source_code.lower():
            spec["type"] = "elementwise"
        
        return spec
    
    def execute(self, ctx: KernelExecutionContext) -> None:
        if self._triton_kernel is None:
            raise RuntimeError("Kernel not compiled")
        
        # Execute Triton kernel on Ascend NPU
        # The BiSheng compiler handles the NPU execution
        self._triton_kernel(*ctx.input_tensors, *ctx.output_tensors)


class CPUKernel(BackendKernel):
    """CPU backend kernel implementation."""
    
    def __init__(self):
        self._launcher = None
    
    @property
    def backend(self) -> KernelBackend:
        return KernelBackend.CPU
    
    def is_available(self) -> bool:
        return True  # CPU is always available
    
    def compile(self, source_code: str, config: KernelConfig) -> CompileResult:
        compiler = CompilerFactory.get_best_compiler_for_device("cpu")
        
        if compiler is None:
            return CompileResult(
                success=False,
                output_path="",
                error_message="No CPU compiler available"
            )
        
        compile_config = CompileConfig(
            source_code=source_code,
            output_dir=config.cache_dir or tempfile.mkdtemp(),
            target_arch=config.target_arch,
            optimization_level=config.optimization_level,
            defines={"YIRAGE_BACKEND_CPU_ENABLED": "1"},
        )
        
        result = compiler.compile(compile_config)
        
        if result.success:
            spec = importlib.util.spec_from_file_location(
                "__yirage_launcher", result.output_path
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self._launcher = getattr(mod, "launch", None)
        
        return result
    
    def execute(self, ctx: KernelExecutionContext) -> None:
        if self._launcher is None:
            raise RuntimeError("Kernel not compiled")
        
        input_ptrs = [t.data_ptr() for t in ctx.input_tensors]
        output_ptrs = [t.data_ptr() for t in ctx.output_tensors]
        
        self._launcher(input_ptrs, output_ptrs, 0, 0, 0)


class MultiBackendKernel:
    """
    Multi-backend kernel with automatic backend selection and fallback.
    
    Example usage:
        kernel = MultiBackendKernel(
            source_code=cuda_code,
            config=KernelConfig(
                backend=KernelBackend.CUDA,
                fallback_backends=[KernelBackend.MACA, KernelBackend.CPU],
            )
        )
        kernel.compile()
        kernel.execute(inputs, outputs)
    """
    
    # Registry of available backend implementations
    _backend_classes = {
        KernelBackend.CUDA: CUDAKernel,
        KernelBackend.MACA: MACAKernel,
        KernelBackend.ASCEND: AscendKernel,
        KernelBackend.CPU: CPUKernel,
    }
    
    def __init__(self, source_code: str, config: Optional[KernelConfig] = None):
        """
        Initialize multi-backend kernel.
        
        Args:
            source_code: Kernel source code
            config: Kernel configuration (auto-detected if not provided)
        """
        self.source_code = source_code
        self.config = config or self._auto_detect_config()
        self._active_backend: Optional[BackendKernel] = None
        self._compiled = False
    
    def _auto_detect_config(self) -> KernelConfig:
        """Automatically detect best configuration based on hardware."""
        config = KernelConfig()
        
        # Determine primary backend
        if torch.cuda.is_available():
            try:
                device_name = torch.cuda.get_device_name(0)
                if any(x in device_name for x in ["MetaX", "C500", "MACA"]):
                    config.backend = KernelBackend.MACA
                else:
                    config.backend = KernelBackend.CUDA
            except:
                config.backend = KernelBackend.CUDA
        else:
            try:
                import torch_npu
                if torch.npu.is_available():
                    config.backend = KernelBackend.ASCEND
                else:
                    config.backend = KernelBackend.CPU
            except ImportError:
                config.backend = KernelBackend.CPU
        
        # Set fallback chain
        fallback_priority = [
            KernelBackend.CUDA,
            KernelBackend.MACA,
            KernelBackend.ASCEND,
            KernelBackend.CPU,
        ]
        config.fallback_backends = [
            b for b in fallback_priority 
            if b != config.backend
        ]
        
        # Auto-detect target architecture
        config.target_arch = get_target_arch_from_device()
        
        return config
    
    def compile(self) -> bool:
        """
        Compile the kernel for the configured backend.
        
        Returns:
            True if compilation succeeded, False otherwise
        """
        backends_to_try = [self.config.backend] + self.config.fallback_backends
        
        for backend in backends_to_try:
            if backend not in self._backend_classes:
                continue
            
            kernel_class = self._backend_classes[backend]
            kernel = kernel_class()
            
            if not kernel.is_available():
                print(f"Backend {backend.value} not available, trying next...")
                continue
            
            print(f"Compiling for {backend.value} backend...")
            result = kernel.compile(self.source_code, self.config)
            
            if result.success:
                self._active_backend = kernel
                self._compiled = True
                print(f"Successfully compiled for {backend.value}")
                return True
            else:
                print(f"Compilation failed for {backend.value}: {result.error_message}")
        
        print("All backends failed to compile")
        return False
    
    def execute(
        self,
        input_tensors: List[torch.Tensor],
        output_tensors: List[torch.Tensor],
        stream: Optional[Any] = None,
    ) -> None:
        """
        Execute the compiled kernel.
        
        Args:
            input_tensors: List of input tensors
            output_tensors: List of output tensors
            stream: Optional execution stream
        """
        if not self._compiled or self._active_backend is None:
            raise RuntimeError("Kernel not compiled. Call compile() first.")
        
        ctx = KernelExecutionContext(
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            stream=stream,
        )
        
        self._active_backend.execute(ctx)
    
    @property
    def active_backend(self) -> Optional[KernelBackend]:
        """Return the currently active backend."""
        if self._active_backend:
            return self._active_backend.backend
        return None
    
    @property
    def is_compiled(self) -> bool:
        """Check if kernel is compiled."""
        return self._compiled


def create_kernel(
    source_code: str,
    backend: str = "auto",
    **kwargs
) -> MultiBackendKernel:
    """
    Factory function to create a multi-backend kernel.
    
    Args:
        source_code: Kernel source code
        backend: Backend type ("cuda", "maca", "ascend", "cpu", "auto")
        **kwargs: Additional configuration options
    
    Returns:
        MultiBackendKernel instance
    """
    config = KernelConfig()
    
    if backend != "auto":
        config.backend = KernelBackend(backend)
    
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return MultiBackendKernel(source_code, config)
