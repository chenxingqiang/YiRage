"""
Backend Compiler Abstraction for YiRage

This module provides an abstraction layer for compiling kernels across different
hardware backends (CUDA, MACA, Ascend, Triton, CPU).

Design Principles:
1. Common interface for all backends
2. Backend-specific optimization flags
3. Automatic compiler detection
4. Runtime compilation support
"""

import os
import sys
import shutil
import subprocess
import sysconfig
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum


class CompilerBackend(Enum):
    """Supported compiler backends."""
    NVCC = "nvcc"           # NVIDIA CUDA Compiler
    MXCC = "mxcc"           # MetaX MACA Compiler
    ASCENDC = "ascendc"     # Huawei Ascend Compiler
    BISHENG = "bisheng"     # Huawei BiSheng Compiler (Triton path)
    TRITON = "triton"       # OpenAI Triton
    GCC = "gcc"             # GNU C++ Compiler (CPU)
    CLANG = "clang"         # LLVM Clang (CPU/Metal)


@dataclass
class CompileResult:
    """Result of a compilation operation."""
    success: bool
    output_path: str
    error_message: str = ""
    compile_time_ms: float = 0.0
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompileConfig:
    """Configuration for kernel compilation."""
    source_code: str
    output_dir: str
    target_arch: str = "native"
    optimization_level: int = 3
    debug_mode: bool = False
    profiling: bool = False
    include_dirs: List[str] = field(default_factory=list)
    library_dirs: List[str] = field(default_factory=list)
    libraries: List[str] = field(default_factory=list)
    defines: Dict[str, str] = field(default_factory=dict)
    extra_flags: List[str] = field(default_factory=list)


class BaseCompiler(ABC):
    """Abstract base class for backend compilers."""
    
    def __init__(self):
        self._compiler_path: Optional[str] = None
        self._version: Optional[str] = None
    
    @property
    @abstractmethod
    def backend(self) -> CompilerBackend:
        """Return the compiler backend type."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return human-readable compiler name."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this compiler is available on the system."""
        pass
    
    @abstractmethod
    def get_compile_command(self, config: CompileConfig, output_path: str) -> List[str]:
        """Generate the compile command for the given configuration."""
        pass
    
    @abstractmethod
    def get_supported_archs(self) -> List[str]:
        """Return list of supported target architectures."""
        pass
    
    def compile(self, config: CompileConfig) -> CompileResult:
        """Compile the source code with the given configuration."""
        import time
        
        if not self.is_available():
            return CompileResult(
                success=False,
                output_path="",
                error_message=f"{self.name} compiler not available"
            )
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Determine output file name
        output_name = self._get_output_name(config)
        output_path = os.path.join(config.output_dir, output_name)
        
        # Write source code to temporary file
        source_ext = self._get_source_extension()
        with tempfile.NamedTemporaryFile(
            mode='w', 
            suffix=source_ext, 
            delete=False,
            dir=config.output_dir
        ) as f:
            f.write(config.source_code)
            source_path = f.name
        
        try:
            # Generate compile command
            cmd = self.get_compile_command(config, output_path)
            
            # Replace source code placeholder
            cmd = [source_path if arg == "__SOURCE__" else arg for arg in cmd]
            
            # Execute compilation
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            compile_time_ms = (time.time() - start_time) * 1000
            
            if result.returncode != 0:
                return CompileResult(
                    success=False,
                    output_path="",
                    error_message=result.stderr,
                    compile_time_ms=compile_time_ms
                )
            
            # Parse warnings from stdout/stderr
            warnings = self._parse_warnings(result.stdout + result.stderr)
            
            return CompileResult(
                success=True,
                output_path=output_path,
                compile_time_ms=compile_time_ms,
                warnings=warnings,
                metadata={"source_path": source_path}
            )
            
        except subprocess.TimeoutExpired:
            return CompileResult(
                success=False,
                output_path="",
                error_message="Compilation timed out"
            )
        except Exception as e:
            return CompileResult(
                success=False,
                output_path="",
                error_message=str(e)
            )
    
    def _get_output_name(self, config: CompileConfig) -> str:
        """Generate output file name."""
        return f"kernel.cpython-{sys.version_info.major}{sys.version_info.minor}-x86_64-linux-gnu.so"
    
    def _get_source_extension(self) -> str:
        """Get source file extension for this compiler."""
        return ".cpp"
    
    def _parse_warnings(self, output: str) -> List[str]:
        """Parse warnings from compiler output."""
        warnings = []
        for line in output.split('\n'):
            if 'warning' in line.lower():
                warnings.append(line.strip())
        return warnings
    
    def get_python_include_dir(self) -> str:
        """Get Python include directory."""
        if hasattr(sysconfig, "get_default_scheme"):
            scheme = sysconfig.get_default_scheme()
        else:
            scheme = sysconfig._get_default_scheme()
        if scheme == "posix_local":
            scheme = "posix_prefix"
        return sysconfig.get_paths(scheme=scheme)["include"]


class NVCCCompiler(BaseCompiler):
    """NVIDIA CUDA Compiler (nvcc)."""
    
    @property
    def backend(self) -> CompilerBackend:
        return CompilerBackend.NVCC
    
    @property
    def name(self) -> str:
        return "NVIDIA CUDA Compiler (nvcc)"
    
    def is_available(self) -> bool:
        if self._compiler_path is None:
            self._compiler_path = shutil.which("nvcc")
        return self._compiler_path is not None
    
    def get_supported_archs(self) -> List[str]:
        return ["sm_75", "sm_80", "sm_86", "sm_89", "sm_90", "sm_90a", "sm_100", "sm_100a"]
    
    def get_compile_command(self, config: CompileConfig, output_path: str) -> List[str]:
        cmd = [
            self._compiler_path,
            "__SOURCE__",
            f"-O{config.optimization_level}",
            f"-I{self.get_python_include_dir()}",
            "-DYIRAGE_BACKEND_USE_CUDA",
            "-shared",
            "-std=c++17",
            "-use_fast_math",
            "-lcublas",
            "-Xcompiler=-fPIC",
            "--expt-relaxed-constexpr",
            "-o", output_path,
        ]
        
        # Add architecture flags
        target = config.target_arch
        if target == "native":
            cmd.append("-arch=native")
        elif target in ["sm_90", "sm_90a"]:
            cmd.extend(["-arch=sm_90a", "-gencode=arch=compute_90a,code=sm_90a"])
            cmd.append("-DYPK_ENABLE_TMA")
            cmd.append("-DYIRAGE_GRACE_HOPPER")
        elif target in ["sm_100", "sm_100a"]:
            cmd.extend(["-arch=sm_100a", "-gencode=arch=compute_100a,code=sm_100a"])
            cmd.append("-DYPK_ENABLE_TMA")
            cmd.append("-DYIRAGE_GRACE_BLACKWELL")
        else:
            cmd.append(f"-arch={target}")
        
        # Debug flags
        if config.debug_mode:
            cmd.extend(["-g", "-G", "-lineinfo"])
        
        # Profiling
        if config.profiling:
            cmd.append("-DYIRAGE_ENABLE_PROFILER")
        
        # Include directories
        for inc_dir in config.include_dirs:
            cmd.append(f"-I{inc_dir}")
        
        # Library directories
        for lib_dir in config.library_dirs:
            cmd.append(f"-L{lib_dir}")
        
        # Libraries
        for lib in config.libraries:
            cmd.append(f"-l{lib}")
        
        # Defines
        for key, value in config.defines.items():
            if value:
                cmd.append(f"-D{key}={value}")
            else:
                cmd.append(f"-D{key}")
        
        # Extra flags
        cmd.extend(config.extra_flags)
        
        return cmd
    
    def _get_source_extension(self) -> str:
        return ".cu"


class MXCCCompiler(BaseCompiler):
    """MetaX MACA Compiler (mxcc)."""
    
    def __init__(self):
        super().__init__()
        self._maca_path = self._detect_maca_path()
    
    @property
    def backend(self) -> CompilerBackend:
        return CompilerBackend.MXCC
    
    @property
    def name(self) -> str:
        return "MetaX MACA Compiler (mxcc)"
    
    def _detect_maca_path(self) -> Optional[str]:
        """Detect MACA SDK installation path."""
        # Check environment variables
        for var in ["MACA_PATH", "MACA_HOME"]:
            path = os.environ.get(var)
            if path and os.path.exists(path):
                return path
        
        # Check standard paths
        for path in ["/opt/maca", "/usr/local/maca", "/opt/metax/maca"]:
            if os.path.exists(os.path.join(path, "include")):
                return path
        
        return None
    
    def is_available(self) -> bool:
        if self._compiler_path is None:
            if self._maca_path:
                self._compiler_path = shutil.which("mxcc") or \
                    os.path.join(self._maca_path, "mxgpu_llvm/bin/mxcc")
                if not os.path.exists(self._compiler_path):
                    self._compiler_path = None
        return self._compiler_path is not None and os.path.exists(self._compiler_path)
    
    def get_supported_archs(self) -> List[str]:
        return ["maca_c500", "maca_c600"]
    
    def get_compile_command(self, config: CompileConfig, output_path: str) -> List[str]:
        cmd = [
            self._compiler_path,
            "-x", "maca",
            "__SOURCE__",
            f"-O{config.optimization_level}",
            f"-I{self.get_python_include_dir()}",
            "-DYIRAGE_BACKEND_MACA_ENABLED",
            "-DYIRAGE_FINGERPRINT_USE_MACA",
            "-shared",
            "-std=c++17",
            "-fPIC",
            "-o", output_path,
        ]
        
        # Add MACA SDK path
        if self._maca_path:
            cmd.append(f"--maca-path={self._maca_path}")
            cmd.append(f"-I{self._maca_path}/include")
            cmd.append(f"-I{self._maca_path}/include/mcr")
            cmd.append(f"-L{self._maca_path}/lib")
        
        # Add include directories
        for inc_dir in config.include_dirs:
            cmd.append(f"-I{inc_dir}")
        
        # Add libraries
        cmd.extend(["-lmcruntime", "-lmcblas"])
        
        # Extra flags
        cmd.extend(config.extra_flags)
        
        return cmd
    
    def _get_source_extension(self) -> str:
        return ".maca"


class AscendCompiler(BaseCompiler):
    """Huawei Ascend Compiler (ascendc/bisheng)."""
    
    def __init__(self):
        super().__init__()
        self._ascend_path = self._detect_ascend_path()
        self._use_triton = True  # Prefer Triton path
    
    @property
    def backend(self) -> CompilerBackend:
        return CompilerBackend.ASCENDC
    
    @property
    def name(self) -> str:
        return "Huawei Ascend Compiler"
    
    def _detect_ascend_path(self) -> Optional[str]:
        """Detect Ascend SDK installation path."""
        for var in ["ASCEND_HOME", "CANN_HOME"]:
            path = os.environ.get(var)
            if path and os.path.exists(path):
                return path
        
        default_path = "/usr/local/Ascend/ascend-toolkit/latest"
        if os.path.exists(default_path):
            return default_path
        
        return None
    
    def is_available(self) -> bool:
        if self._compiler_path is None:
            # Try ascendc first
            self._compiler_path = shutil.which("ascendc")
            if not self._compiler_path and self._ascend_path:
                self._compiler_path = os.path.join(
                    self._ascend_path, "compiler/bin/ascendc"
                )
            
            # Try bisheng as fallback
            if not self._compiler_path or not os.path.exists(self._compiler_path):
                self._compiler_path = shutil.which("bisheng")
            
            if self._compiler_path and not os.path.exists(self._compiler_path):
                self._compiler_path = None
        
        return self._compiler_path is not None
    
    def get_supported_archs(self) -> List[str]:
        return ["ascend910", "ascend910b", "ascend310p"]
    
    def get_compile_command(self, config: CompileConfig, output_path: str) -> List[str]:
        cmd = [
            self._compiler_path,
            "-c", "__SOURCE__",
            "-o", output_path,
            f"-I{self.get_python_include_dir()}",
            "-DYIRAGE_BACKEND_ASCEND_ENABLED",
            "-DYIRAGE_FINGERPRINT_USE_ASCEND",
            "-D__ASCEND__",
            "-std=c++17",
            "-fPIC",
            f"-O{config.optimization_level}",
        ]
        
        # Add Ascend SDK paths
        if self._ascend_path:
            cmd.append(f"-I{self._ascend_path}/include")
            cmd.append(f"-I{self._ascend_path}/include/acl")
        
        # Include directories
        for inc_dir in config.include_dirs:
            cmd.append(f"-I{inc_dir}")
        
        # Extra flags
        cmd.extend(config.extra_flags)
        
        return cmd
    
    def _get_source_extension(self) -> str:
        return ".ascend"


class TritonCompiler(BaseCompiler):
    """OpenAI Triton Compiler."""
    
    @property
    def backend(self) -> CompilerBackend:
        return CompilerBackend.TRITON
    
    @property
    def name(self) -> str:
        return "OpenAI Triton"
    
    def is_available(self) -> bool:
        try:
            import triton
            return True
        except ImportError:
            return False
    
    def get_supported_archs(self) -> List[str]:
        archs = ["cuda"]
        try:
            # Check for Ascend Triton support
            import torch_npu
            archs.append("ascend")
        except ImportError:
            pass
        return archs
    
    def get_compile_command(self, config: CompileConfig, output_path: str) -> List[str]:
        # Triton uses JIT compilation, return Python invocation
        return ["python3", "-c", f"import triton; print('Triton JIT compilation')"]
    
    def compile(self, config: CompileConfig) -> CompileResult:
        """Triton uses JIT compilation - just validate the source."""
        if not self.is_available():
            return CompileResult(
                success=False,
                output_path="",
                error_message="Triton not available"
            )
        
        # For Triton, we just save the source as a .py file
        os.makedirs(config.output_dir, exist_ok=True)
        output_path = os.path.join(config.output_dir, "kernel_triton.py")
        
        with open(output_path, 'w') as f:
            f.write(config.source_code)
        
        return CompileResult(
            success=True,
            output_path=output_path,
            metadata={"jit": True}
        )
    
    def _get_source_extension(self) -> str:
        return ".py"


class CPUCompiler(BaseCompiler):
    """CPU Compiler (gcc/clang)."""
    
    def __init__(self):
        super().__init__()
        self._use_clang = False
    
    @property
    def backend(self) -> CompilerBackend:
        return CompilerBackend.CLANG if self._use_clang else CompilerBackend.GCC
    
    @property
    def name(self) -> str:
        return "Clang" if self._use_clang else "GCC"
    
    def is_available(self) -> bool:
        if self._compiler_path is None:
            # Prefer clang on macOS
            if sys.platform == "darwin":
                self._compiler_path = shutil.which("clang++")
                self._use_clang = True
            
            if not self._compiler_path:
                self._compiler_path = shutil.which("g++")
                self._use_clang = False
            
            if not self._compiler_path:
                self._compiler_path = shutil.which("clang++")
                self._use_clang = True
        
        return self._compiler_path is not None
    
    def get_supported_archs(self) -> List[str]:
        return ["native", "x86-64", "arm64", "avx2", "avx512"]
    
    def get_compile_command(self, config: CompileConfig, output_path: str) -> List[str]:
        cmd = [
            self._compiler_path,
            "__SOURCE__",
            f"-O{config.optimization_level}",
            f"-I{self.get_python_include_dir()}",
            "-DYIRAGE_BACKEND_CPU_ENABLED",
            "-shared",
            "-std=c++17",
            "-fPIC",
            "-o", output_path,
        ]
        
        # Architecture-specific flags
        if config.target_arch == "avx512":
            cmd.append("-mavx512f")
        elif config.target_arch == "avx2":
            cmd.append("-mavx2")
        elif config.target_arch == "native":
            cmd.append("-march=native")
        
        # OpenMP for parallelism
        if not self._use_clang:
            cmd.append("-fopenmp")
        else:
            cmd.append("-Xpreprocessor")
            cmd.append("-fopenmp")
        
        # Include directories
        for inc_dir in config.include_dirs:
            cmd.append(f"-I{inc_dir}")
        
        # Library directories
        for lib_dir in config.library_dirs:
            cmd.append(f"-L{lib_dir}")
        
        # Libraries
        for lib in config.libraries:
            cmd.append(f"-l{lib}")
        
        # Extra flags
        cmd.extend(config.extra_flags)
        
        return cmd


# ============================================================================
# Compiler Factory
# ============================================================================

class CompilerFactory:
    """Factory for creating backend compilers."""
    
    _compilers: Dict[CompilerBackend, BaseCompiler] = {}
    
    @classmethod
    def get_compiler(cls, backend: CompilerBackend) -> BaseCompiler:
        """Get a compiler instance for the specified backend."""
        if backend not in cls._compilers:
            if backend == CompilerBackend.NVCC:
                cls._compilers[backend] = NVCCCompiler()
            elif backend == CompilerBackend.MXCC:
                cls._compilers[backend] = MXCCCompiler()
            elif backend == CompilerBackend.ASCENDC:
                cls._compilers[backend] = AscendCompiler()
            elif backend == CompilerBackend.TRITON:
                cls._compilers[backend] = TritonCompiler()
            elif backend == CompilerBackend.BISHENG:
                # BiSheng is a Triton-compatible compiler for Ascend
                cls._compilers[backend] = TritonCompiler()
            elif backend in [CompilerBackend.GCC, CompilerBackend.CLANG]:
                cls._compilers[backend] = CPUCompiler()
            else:
                raise ValueError(f"Unknown compiler backend: {backend}")
        
        return cls._compilers[backend]
    
    @classmethod
    def get_available_compilers(cls) -> List[CompilerBackend]:
        """Get list of available compiler backends."""
        available = []
        for backend in CompilerBackend:
            try:
                compiler = cls.get_compiler(backend)
                if compiler.is_available():
                    available.append(backend)
            except ValueError:
                pass
        return available
    
    @classmethod
    def get_best_compiler_for_device(cls, device: str) -> Optional[BaseCompiler]:
        """
        Get the best available compiler for a device type.
        
        Args:
            device: Device type ("cuda", "maca", "ascend", "cpu", "mps")
        
        Returns:
            Best available compiler, or None if none available
        """
        device_to_backends = {
            "cuda": [CompilerBackend.NVCC, CompilerBackend.TRITON],
            "maca": [CompilerBackend.MXCC, CompilerBackend.NVCC],
            "ascend": [CompilerBackend.ASCENDC, CompilerBackend.TRITON],
            "cpu": [CompilerBackend.GCC, CompilerBackend.CLANG],
            "mps": [CompilerBackend.CLANG],
        }
        
        backends = device_to_backends.get(device, [CompilerBackend.GCC])
        
        for backend in backends:
            try:
                compiler = cls.get_compiler(backend)
                if compiler.is_available():
                    return compiler
            except ValueError:
                continue
        
        return None


# ============================================================================
# Utility Functions
# ============================================================================

def compile_kernel(
    source_code: str,
    output_dir: str,
    backend: str = "auto",
    target_arch: str = "native",
    **kwargs
) -> CompileResult:
    """
    Compile a kernel with the appropriate backend compiler.
    
    Args:
        source_code: Kernel source code
        output_dir: Directory for output files
        backend: Backend type ("cuda", "maca", "ascend", "cpu", "auto")
        target_arch: Target architecture
        **kwargs: Additional compilation options
    
    Returns:
        CompileResult with compilation status and output path
    """
    # Get the appropriate compiler
    if backend == "auto":
        # Try to detect from source code or environment
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            if "MetaX" in device_name or "C500" in device_name:
                backend = "maca"
            else:
                backend = "cuda"
        else:
            backend = "cpu"
    
    compiler = CompilerFactory.get_best_compiler_for_device(backend)
    
    if compiler is None:
        return CompileResult(
            success=False,
            output_path="",
            error_message=f"No compiler available for {backend}"
        )
    
    # Create configuration
    config = CompileConfig(
        source_code=source_code,
        output_dir=output_dir,
        target_arch=target_arch,
        optimization_level=kwargs.get("optimization_level", 3),
        debug_mode=kwargs.get("debug_mode", False),
        profiling=kwargs.get("profiling", False),
        include_dirs=kwargs.get("include_dirs", []),
        library_dirs=kwargs.get("library_dirs", []),
        libraries=kwargs.get("libraries", []),
        defines=kwargs.get("defines", {}),
        extra_flags=kwargs.get("extra_flags", []),
    )
    
    return compiler.compile(config)


def get_target_arch_from_device() -> str:
    """Detect target architecture from current device."""
    import torch
    
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        cc = props.major * 10 + props.minor
        if cc >= 100:
            return "sm_100a"
        elif cc >= 90:
            return "sm_90a"
        elif cc >= 86:
            return "sm_86"
        elif cc >= 80:
            return "sm_80"
        else:
            return "sm_75"
    
    return "native"
