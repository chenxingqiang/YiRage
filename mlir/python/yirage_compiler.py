#!/usr/bin/env python3
"""
YiRage MLIR Compiler Python Interface

End-to-end compilation pipeline:
  muGraph → MLIR → Optimized MLIR → Executable

Usage:
    from mlir.python.yirage_compiler import YirageCompiler
    
    compiler = YirageCompiler(target='cuda')
    
    # From muGraph
    compiled_fn = compiler.compile(graph)
    result = compiled_fn(inputs)
    
    # From MLIR file
    compiled_fn = compiler.compile_mlir_file('kernel.mlir')
"""

import os
import subprocess
import tempfile
from typing import Dict, List, Optional, Tuple, Any, Callable
from pathlib import Path

try:
    from .mugraph_to_mlir import MuGraphToMLIR, convert_mugraph_to_mlir
except ImportError:
    from mugraph_to_mlir import MuGraphToMLIR, convert_mugraph_to_mlir


class YirageCompiler:
    """End-to-end YiRage compiler from muGraph to executable code."""
    
    # Available target backends
    TARGETS = {
        'cuda': 'yirage-cuda-pipeline',
        'rocm': 'yirage-rocm-pipeline',
        'cpu': 'yirage-cpu-pipeline',
        'mps': 'yirage-mps-pipeline',
        'ascend': 'yirage-ascend-pipeline',
        'tpu': 'yirage-tpu-pipeline',
        'fpga': 'yirage-fpga-pipeline',
        'gpu': 'yirage-gpu-pipeline',
    }
    
    def __init__(self, target: str = 'cuda', opt_level: int = 3,
                 yirage_opt_path: Optional[str] = None):
        """Initialize the compiler.
        
        Args:
            target: Target backend ('cuda', 'rocm', 'cpu', 'mps', etc.)
            opt_level: Optimization level (0-3)
            yirage_opt_path: Path to yirage-opt binary
        """
        self.target = target
        self.opt_level = opt_level
        
        # Find yirage-opt binary
        if yirage_opt_path:
            self.yirage_opt = yirage_opt_path
        else:
            self.yirage_opt = self._find_yirage_opt()
        
        self.converter = MuGraphToMLIR()
    
    def _find_yirage_opt(self) -> str:
        """Find the yirage-opt binary."""
        # Check common locations
        locations = [
            # Relative to this file
            Path(__file__).parent.parent / 'build' / 'yirage-opt',
            # In PATH
            'yirage-opt',
            # Standard install locations
            '/usr/local/bin/yirage-opt',
            '/opt/yirage/bin/yirage-opt',
        ]
        
        for loc in locations:
            loc = Path(loc) if isinstance(loc, str) else loc
            if loc.exists():
                return str(loc)
            # Try finding in PATH
            if not loc.is_absolute():
                import shutil
                found = shutil.which(str(loc))
                if found:
                    return found
        
        # Default - will fail if not found
        return str(Path(__file__).parent.parent / 'build' / 'yirage-opt')
    
    def compile(self, graph, entry_func: str = 'mugraph') -> 'CompiledKernel':
        """Compile a muGraph to executable code.
        
        Args:
            graph: KNGraph instance or path to JSON file
            entry_func: Name of the entry function
            
        Returns:
            CompiledKernel that can be invoked
        """
        # Convert to MLIR
        if isinstance(graph, str):
            mlir_text = self.converter.convert_from_json(graph)
        else:
            mlir_text = self.converter.convert(graph)
        
        return self._compile_mlir(mlir_text, entry_func)
    
    def compile_mlir_file(self, mlir_path: str, 
                          entry_func: str = 'mugraph') -> 'CompiledKernel':
        """Compile an MLIR file.
        
        Args:
            mlir_path: Path to .mlir file
            entry_func: Name of the entry function
            
        Returns:
            CompiledKernel that can be invoked
        """
        with open(mlir_path, 'r') as f:
            mlir_text = f.read()
        
        return self._compile_mlir(mlir_text, entry_func)
    
    def compile_mlir_text(self, mlir_text: str,
                          entry_func: str = 'mugraph') -> 'CompiledKernel':
        """Compile MLIR source text.
        
        Args:
            mlir_text: MLIR source code
            entry_func: Name of the entry function
            
        Returns:
            CompiledKernel that can be invoked
        """
        return self._compile_mlir(mlir_text, entry_func)
    
    def _compile_mlir(self, mlir_text: str, entry_func: str) -> 'CompiledKernel':
        """Internal compilation implementation."""
        # Get the pipeline for target
        pipeline = self.TARGETS.get(self.target)
        if not pipeline:
            raise ValueError(f"Unknown target: {self.target}. "
                           f"Available: {list(self.TARGETS.keys())}")
        
        # Write to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.mlir', 
                                         delete=False) as f:
            f.write(mlir_text)
            input_path = f.name
        
        output_path = input_path.replace('.mlir', '.lowered.mlir')
        
        try:
            # Run yirage-opt with the appropriate pipeline
            cmd = [
                self.yirage_opt,
                input_path,
                f'-{pipeline}',
                '-o', output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"Compilation failed:\n{result.stderr}")
            
            # Read lowered MLIR
            with open(output_path, 'r') as f:
                lowered_mlir = f.read()
            
            return CompiledKernel(
                mlir_source=mlir_text,
                lowered_mlir=lowered_mlir,
                target=self.target,
                entry_func=entry_func,
                opt_level=self.opt_level
            )
        finally:
            # Cleanup temp files
            try:
                os.unlink(input_path)
                if os.path.exists(output_path):
                    os.unlink(output_path)
            except:
                pass
    
    def get_pipeline_info(self) -> Dict[str, str]:
        """Get information about available pipelines."""
        return {
            name: f"yirage-opt -{pipeline}"
            for name, pipeline in self.TARGETS.items()
        }


class CompiledKernel:
    """Represents a compiled kernel ready for execution."""
    
    def __init__(self, mlir_source: str, lowered_mlir: str,
                 target: str, entry_func: str, opt_level: int):
        self.mlir_source = mlir_source
        self.lowered_mlir = lowered_mlir
        self.target = target
        self.entry_func = entry_func
        self.opt_level = opt_level
        self._native_fn = None
    
    def get_source(self) -> str:
        """Get the original MLIR source."""
        return self.mlir_source
    
    def get_lowered(self) -> str:
        """Get the lowered MLIR after compilation."""
        return self.lowered_mlir
    
    def __call__(self, *args, **kwargs):
        """Execute the compiled kernel.
        
        Note: Full native execution requires the C++ JIT engine.
        This Python interface currently falls back to PyTorch.
        """
        # For now, we execute using PyTorch as the backend
        # Full native execution would require:
        # 1. LLVM IR generation from lowered MLIR
        # 2. JIT compilation via MLIR ExecutionEngine
        # 3. Direct native function invocation
        
        raise NotImplementedError(
            "Native execution requires the C++ JIT engine. "
            "Use the lowered MLIR with yirage-opt or integrate with "
            "the C++ execution engine."
        )
    
    def save(self, path: str):
        """Save the compiled kernel to a file."""
        with open(path, 'w') as f:
            f.write(f"// Target: {self.target}\n")
            f.write(f"// Entry: {self.entry_func}\n")
            f.write(f"// Opt Level: {self.opt_level}\n")
            f.write("\n// === Original MLIR ===\n")
            f.write("// " + self.mlir_source.replace("\n", "\n// ") + "\n")
            f.write("\n// === Lowered MLIR ===\n")
            f.write(self.lowered_mlir)
    
    def __repr__(self):
        return (f"CompiledKernel(target='{self.target}', "
                f"entry='{self.entry_func}', opt={self.opt_level})")


def compile_graph(graph, target: str = 'cuda') -> CompiledKernel:
    """Convenience function to compile a graph.
    
    Args:
        graph: KNGraph or path to JSON file
        target: Target backend
        
    Returns:
        CompiledKernel
    """
    compiler = YirageCompiler(target=target)
    return compiler.compile(graph)


def compile_mlir(mlir_path: str, target: str = 'cuda') -> CompiledKernel:
    """Convenience function to compile an MLIR file.
    
    Args:
        mlir_path: Path to .mlir file
        target: Target backend
        
    Returns:
        CompiledKernel
    """
    compiler = YirageCompiler(target=target)
    return compiler.compile_mlir_file(mlir_path)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python yirage_compiler.py <input.mlir|input.json> [target]")
        print("\nTargets:", ', '.join(YirageCompiler.TARGETS.keys()))
        sys.exit(1)
    
    input_path = sys.argv[1]
    target = sys.argv[2] if len(sys.argv) > 2 else 'cpu'
    
    compiler = YirageCompiler(target=target)
    
    if input_path.endswith('.json'):
        kernel = compiler.compile(input_path)
    else:
        kernel = compiler.compile_mlir_file(input_path)
    
    print(f"Compiled: {kernel}")
    print("\n=== Lowered MLIR ===")
    print(kernel.get_lowered())
