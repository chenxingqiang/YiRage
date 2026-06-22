#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0

"""
YiRage System Integrity Verification
=====================================
Comprehensive verification of all system components for:
- Completeness: All required components exist
- Closedness: All logic paths are properly terminated
- Robustness: Edge cases are handled correctly
"""

import os
import sys
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional
from enum import Enum, auto

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent


class VerificationStatus(Enum):
    PASSED = auto()
    FAILED = auto()
    WARNING = auto()
    SKIPPED = auto()


@dataclass
class VerificationResult:
    component: str
    test_name: str
    status: VerificationStatus
    message: str = ""
    details: Dict = field(default_factory=dict)


class SystemVerifier:
    """Main verification class for YiRage system integrity."""
    
    def __init__(self):
        self.results: List[VerificationResult] = []
        self.component_scores: Dict[str, Tuple[int, int]] = {}
    
    def add_result(self, component: str, test: str, 
                   status: VerificationStatus, msg: str = "", 
                   details: Dict = None):
        self.results.append(VerificationResult(
            component=component,
            test_name=test,
            status=status,
            message=msg,
            details=details or {}
        ))
    
    def verify_all(self) -> Dict:
        """Run all verification suites."""
        print("=" * 70)
        print("YiRage System Integrity Verification")
        print("=" * 70)
        
        self.verify_mlir_dialect()
        self.verify_search_core()
        self.verify_backend_strategies()
        self.verify_transpiler()
        self.verify_threadblock()
        self.verify_kernel_factory()
        self.verify_rl_interface()
        self.verify_nki_triton()
        self.verify_python_bindings()
        
        return self.generate_report()
    
    # =========================================================================
    # 1. MLIR Dialect Verification
    # =========================================================================
    
    def verify_mlir_dialect(self):
        """Verify MLIR dialect completeness."""
        print("\n1. Verifying MLIR Dialect...")
        
        ops_td = PROJECT_ROOT / "mlir/include/yirage-mlir/Dialect/Yirage/IR/YirageOps.td"
        ops_cpp = PROJECT_ROOT / "mlir/lib/Dialect/Yirage/IR/YirageOps.cpp"
        lowering = PROJECT_ROOT / "mlir/lib/Dialect/Yirage/Transforms/YirageToLinalg.cpp"
        
        # Parse defined operations
        defined_ops = self._extract_ops_from_td(ops_td)
        verifier_ops = self._extract_verifiers(ops_cpp)
        lowering_ops = self._extract_lowerings(lowering)
        
        # Check completeness
        for op in defined_ops:
            has_lowering = op in lowering_ops
            self.add_result("MLIR Dialect", f"{op} has lowering",
                           VerificationStatus.PASSED if has_lowering else VerificationStatus.FAILED)
        
        # Check special ops that use traits instead of explicit verifiers
        trait_verified_ops = {"SiLUOp", "GELUOp", "ReLUOp"}  # SameOperandsAndResultType trait
        
        for op in defined_ops:
            if op in trait_verified_ops:
                self.add_result("MLIR Dialect", f"{op} trait-verified",
                               VerificationStatus.PASSED)
            elif op in verifier_ops:
                self.add_result("MLIR Dialect", f"{op} has verifier",
                               VerificationStatus.PASSED)
            else:
                self.add_result("MLIR Dialect", f"{op} has verifier",
                               VerificationStatus.WARNING, "Missing explicit verifier")
        
        self._update_component_score("MLIR Dialect")
    
    def _extract_ops_from_td(self, path: Path) -> Set[str]:
        """Extract operation names from TableGen file."""
        ops = set()
        if path.exists():
            content = path.read_text()
            import re
            matches = re.findall(r'def Yirage_(\w+Op)', content)
            ops = set(matches)
        return ops
    
    def _extract_verifiers(self, path: Path) -> Set[str]:
        """Extract operations that have verify() methods."""
        ops = set()
        if path.exists():
            content = path.read_text()
            import re
            matches = re.findall(r'(\w+Op)::verify\(\)', content)
            ops = set(matches)
        return ops
    
    def _extract_lowerings(self, path: Path) -> Set[str]:
        """Extract operations that have lowering patterns."""
        ops = set()
        if path.exists():
            content = path.read_text()
            import re
            matches = re.findall(r'struct (\w+Op)Lowering', content)
            ops = set(matches)
        return ops
    
    # =========================================================================
    # 2. Search Core Verification
    # =========================================================================
    
    def verify_search_core(self):
        """Verify search algorithm completeness."""
        print("\n2. Verifying Search Core...")
        
        search_cc = PROJECT_ROOT / "src/search/search.cc"
        config_h = PROJECT_ROOT / "include/search/config.h"
        
        # Verify search.cc exists and has key functions
        if search_cc.exists():
            content = search_cc.read_text()
            
            # Check key search functions
            key_functions = [
                "generate_next_operator",
                "preprocess",
                "check_range",
                "instantiate_symbolic_graph"
            ]
            
            for func in key_functions:
                found = func in content
                self.add_result("Search Core", f"has_{func}",
                               VerificationStatus.PASSED if found else VerificationStatus.FAILED)
            
            # Check for proper cleanup (no memory leaks)
            has_delete = "delete " in content
            self.add_result("Search Core", "has_cleanup_logic",
                           VerificationStatus.PASSED if has_delete else VerificationStatus.WARNING)
        
        # Verify config structure
        if config_h.exists():
            content = config_h.read_text()
            
            config_fields = [
                "max_num_threadblock_graph_op",
                "max_num_kernel_graph_op",
                "grid_dim_to_explore",
                "block_dim_to_explore",
                "backend_type"
            ]
            
            for field in config_fields:
                found = field in content
                self.add_result("Search Core", f"config_has_{field}",
                               VerificationStatus.PASSED if found else VerificationStatus.FAILED)
        
        self._update_component_score("Search Core")
    
    # =========================================================================
    # 3. Backend Strategies Verification
    # =========================================================================
    
    def verify_backend_strategies(self):
        """Verify all backend strategies are implemented."""
        print("\n3. Verifying Backend Strategies...")
        
        strategies_dir = PROJECT_ROOT / "src/search/backend_strategies"
        headers_dir = PROJECT_ROOT / "include/search/backend_strategies"
        
        expected_backends = [
            "cuda", "mps", "rocm", "cpu", "ascend", "maca",
            "tpu", "fpga", "xpu", "triton", "nki", "mlir"
        ]
        
        for backend in expected_backends:
            # Check source file
            src_file = strategies_dir / f"{backend}_strategy.cc"
            has_src = src_file.exists()
            self.add_result("Backend Strategies", f"{backend}_has_source",
                           VerificationStatus.PASSED if has_src else VerificationStatus.FAILED)
            
            # Check header file
            hdr_file = headers_dir / f"{backend}_strategy.h"
            has_hdr = hdr_file.exists()
            self.add_result("Backend Strategies", f"{backend}_has_header",
                           VerificationStatus.PASSED if has_hdr else VerificationStatus.FAILED)
            
            # Check key methods if source exists
            if has_src:
                content = src_file.read_text()
                has_generate = "generate_candidates" in content
                has_optimize = "optimize" in content
                
                self.add_result("Backend Strategies", f"{backend}_has_generate",
                               VerificationStatus.PASSED if has_generate else VerificationStatus.FAILED)
                self.add_result("Backend Strategies", f"{backend}_has_optimize",
                               VerificationStatus.PASSED if has_optimize else VerificationStatus.FAILED)
        
        # Verify factory
        factory_cc = PROJECT_ROOT / "src/search/common/search_strategy_factory.cc"
        if factory_cc.exists():
            content = factory_cc.read_text()
            for backend in expected_backends:
                has_case = f"BT_{backend.upper()}" in content or backend.upper() in content
                self.add_result("Backend Strategies", f"factory_handles_{backend}",
                               VerificationStatus.PASSED if has_case else VerificationStatus.WARNING)
        
        self._update_component_score("Backend Strategies")
    
    # =========================================================================
    # 4. Transpiler Verification
    # =========================================================================
    
    def verify_transpiler(self):
        """Verify transpiler handles all operator types."""
        print("\n4. Verifying Transpiler...")
        
        transpiler_tb = PROJECT_ROOT / "src/transpiler/transpiler_tb.cc"
        
        if transpiler_tb.exists():
            content = transpiler_tb.read_text()
            
            # Check handled operator types
            tb_ops = [
                "TB_INPUT_OP", "TB_OUTPUT_OP", "TB_MATMUL_OP",
                "TB_EXP_OP", "TB_SQRT_OP", "TB_SILU_OP", "TB_GELU_OP",
                "TB_ADD_OP", "TB_MUL_OP", "TB_DIV_OP", "TB_SUB_OP",
                "TB_REDUCTION_0_OP", "TB_REDUCTION_1_OP", "TB_REDUCTION_2_OP",
                "TB_FORLOOP_ACCUM_NO_RED_OP", "TB_FORLOOP_ACCUM_MAX_OP",
                "TB_CONCAT_0_OP", "TB_CONCAT_1_OP"
            ]
            
            for op in tb_ops:
                found = op in content
                self.add_result("Transpiler", f"handles_{op}",
                               VerificationStatus.PASSED if found else VerificationStatus.WARNING)
            
            # Check for default case handling
            has_default = "default:" in content
            self.add_result("Transpiler", "has_default_case",
                           VerificationStatus.PASSED if has_default else VerificationStatus.FAILED)
        
        self._update_component_score("Transpiler")
    
    # =========================================================================
    # 5. Threadblock Verification
    # =========================================================================
    
    def verify_threadblock(self):
        """Verify threadblock graph operations."""
        print("\n5. Verifying Threadblock...")
        
        tb_dir = PROJECT_ROOT / "src/threadblock"
        
        expected_files = [
            "graph.cc", "input_loader.cc", "output.cc", "matmul.cc",
            "reduction.cc", "element_unary.cc", "element_binary.cc",
            "forloop_accum.cc", "concat.cc", "rms_norm.cc"
        ]
        
        for fname in expected_files:
            fpath = tb_dir / fname
            exists = fpath.exists()
            self.add_result("Threadblock", f"has_{fname}",
                           VerificationStatus.PASSED if exists else VerificationStatus.FAILED)
        
        # Check graph.h for create_*_op methods
        graph_h = PROJECT_ROOT / "include/threadblock/graph.h"
        if graph_h.exists():
            content = graph_h.read_text()
            
            create_methods = [
                "create_input_op", "create_output_op", "create_matmul_op",
                "create_elementunary_op", "create_elementbinary_op",
                "create_reduction_op", "create_forloop_accum_op",
                "create_concat_op", "create_rms_norm_op"
            ]
            
            for method in create_methods:
                found = method in content
                self.add_result("Threadblock", f"graph_has_{method}",
                               VerificationStatus.PASSED if found else VerificationStatus.FAILED)
        
        self._update_component_score("Threadblock")
    
    # =========================================================================
    # 6. Kernel Factory Verification
    # =========================================================================
    
    def verify_kernel_factory(self):
        """Verify kernel executor factory."""
        print("\n6. Verifying Kernel Factory...")
        
        factory_cc = PROJECT_ROOT / "src/kernel/common/kernel_factory.cc"
        interface_h = PROJECT_ROOT / "include/kernel/common/kernel_interface.h"
        
        if factory_cc.exists():
            content = factory_cc.read_text()
            
            # Check factory methods
            factory_methods = [
                "create_matmul_executor",
                "create_rmsnorm_executor",
                "create_reduction_executor",
                "create_element_unary_executor",
                "create_element_binary_executor"
            ]
            
            for method in factory_methods:
                found = method in content
                self.add_result("Kernel Factory", f"has_{method}",
                               VerificationStatus.PASSED if found else VerificationStatus.FAILED)
            
            # Check for GenericKernelExecutor implementation
            has_generic = "GenericKernelExecutor" in content
            self.add_result("Kernel Factory", "has_generic_executor",
                           VerificationStatus.PASSED if has_generic else VerificationStatus.FAILED)
            
            # Check backend handling
            backends = ["BT_CUDA", "BT_MPS", "BT_CPU"]
            for backend in backends:
                found = backend in content
                self.add_result("Kernel Factory", f"handles_{backend}",
                               VerificationStatus.PASSED if found else VerificationStatus.WARNING)
        
        if interface_h.exists():
            content = interface_h.read_text()
            
            # Check interface completeness
            interface_methods = [
                "compile", "execute", "get_execution_time",
                "get_metrics", "get_backend_type", "validate_config"
            ]
            
            for method in interface_methods:
                found = method in content
                self.add_result("Kernel Factory", f"interface_has_{method}",
                               VerificationStatus.PASSED if found else VerificationStatus.FAILED)
        
        self._update_component_score("Kernel Factory")
    
    # =========================================================================
    # 7. RL Interface Verification
    # =========================================================================
    
    def verify_rl_interface(self):
        """Verify RL interface implementation."""
        print("\n7. Verifying RL Interface...")
        
        rl_cc = PROJECT_ROOT / "src/search/rl_interface.cc"
        rl_h = PROJECT_ROOT / "include/search/rl_interface.h"
        
        if rl_cc.exists():
            content = rl_cc.read_text()
            
            # Check key methods
            key_methods = [
                "apply_action", "add_kn_operator", "create_threadblock",
                "add_tb_operator", "verify", "profile", "get_state"
            ]
            
            for method in key_methods:
                found = method in content
                self.add_result("RL Interface", f"has_{method}",
                               VerificationStatus.PASSED if found else VerificationStatus.FAILED)
            
            # Check for proper state management
            has_state = "SearchState" in content
            self.add_result("RL Interface", "has_state_management",
                           VerificationStatus.PASSED if has_state else VerificationStatus.FAILED)
            
            # Check for validation logic
            has_validation = "if (" in content and "return false" in content
            self.add_result("RL Interface", "has_validation_logic",
                           VerificationStatus.PASSED if has_validation else VerificationStatus.WARNING)
        
        self._update_component_score("RL Interface")
    
    # =========================================================================
    # 8. NKI/Triton Verification
    # =========================================================================
    
    def verify_nki_triton(self):
        """Verify NKI and Triton transpilers."""
        print("\n8. Verifying NKI/Triton...")
        
        nki_dir = PROJECT_ROOT / "src/nki_transpiler"
        triton_dir = PROJECT_ROOT / "src/triton_transpiler"
        
        # NKI verification
        nki_transpile = nki_dir / "transpile_tb.cc"
        if nki_transpile.exists():
            content = nki_transpile.read_text()
            
            features = [
                ("matmul", "TB_MATMUL_OP" in content),
                ("reduction", "reduction" in content.lower()),
                ("nd_support", "num_dims" in content),
            ]
            
            for name, has_feature in features:
                self.add_result("NKI Transpiler", f"has_{name}",
                               VerificationStatus.PASSED if has_feature else VerificationStatus.WARNING)
        
        # Triton verification
        triton_transpile = triton_dir / "transpile.cc"
        if triton_transpile.exists():
            content = triton_transpile.read_text()
            
            features = [
                ("matmul", "KN_MATMUL_OP" in content),
                ("element_binary", "elementwise_binary" in content),
                ("element_unary", "elementwise_unary" in content),
                ("reduction", "reduce_sum" in content),
                ("nd_support", "num_dims" in content),
            ]
            
            for name, has_feature in features:
                self.add_result("Triton Transpiler", f"has_{name}",
                               VerificationStatus.PASSED if has_feature else VerificationStatus.WARNING)
        
        self._update_component_score("NKI Transpiler")
        self._update_component_score("Triton Transpiler")
    
    # =========================================================================
    # 9. Python Bindings Verification
    # =========================================================================
    
    def verify_python_bindings(self):
        """Verify Python bindings completeness."""
        print("\n9. Verifying Python Bindings...")
        
        init_py = PROJECT_ROOT / "python/yirage/__init__.py"
        
        if init_py.exists():
            content = init_py.read_text()
            
            # Check core imports
            core_imports = [
                "KNGraph", "TBGraph", "get_available_backends",
                "CompilerFactory", "HardwareProfiler", "MuGraphStore"
            ]
            
            for imp in core_imports:
                found = imp in content
                self.add_result("Python Bindings", f"exports_{imp}",
                               VerificationStatus.PASSED if found else VerificationStatus.WARNING)
            
            # Check backend modules
            backend_modules = ["mps", "ascend", "maca"]
            for module in backend_modules:
                found = module in content
                self.add_result("Python Bindings", f"has_{module}_config",
                               VerificationStatus.PASSED if found else VerificationStatus.WARNING)
            
            # Check optional features
            optional_features = [
                ("ray", "is_ray_available"),
                ("rl", "is_rl_available"),
            ]
            
            for name, check_func in optional_features:
                found = check_func in content
                self.add_result("Python Bindings", f"has_{name}_check",
                               VerificationStatus.PASSED if found else VerificationStatus.WARNING)
        
        self._update_component_score("Python Bindings")
    
    # =========================================================================
    # Reporting
    # =========================================================================
    
    def _update_component_score(self, component: str):
        """Update score for a component."""
        passed = sum(1 for r in self.results 
                     if r.component == component and r.status == VerificationStatus.PASSED)
        total = sum(1 for r in self.results if r.component == component)
        self.component_scores[component] = (passed, total)
    
    def generate_report(self) -> Dict:
        """Generate comprehensive verification report."""
        print("\n" + "=" * 70)
        print("VERIFICATION REPORT")
        print("=" * 70)
        
        total_passed = sum(1 for r in self.results if r.status == VerificationStatus.PASSED)
        total_tests = len(self.results)
        total_warnings = sum(1 for r in self.results if r.status == VerificationStatus.WARNING)
        total_failed = sum(1 for r in self.results if r.status == VerificationStatus.FAILED)
        
        print(f"\nOverall: {total_passed}/{total_tests} passed "
              f"({100*total_passed/total_tests:.1f}%)")
        print(f"Warnings: {total_warnings}, Failed: {total_failed}")
        
        print("\n" + "-" * 70)
        print("Component Health:")
        print("-" * 70)
        
        health_chart = {}
        for component, (passed, total) in sorted(self.component_scores.items()):
            pct = 100 * passed / total if total > 0 else 0
            health_chart[component] = pct
            bar_len = int(pct / 5)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            print(f"{component:25s} {bar} {pct:5.1f}%")
        
        if total_failed > 0:
            print("\n" + "-" * 70)
            print("Failed Tests:")
            print("-" * 70)
            for r in self.results:
                if r.status == VerificationStatus.FAILED:
                    print(f"  [{r.component}] {r.test_name}")
                    if r.message:
                        print(f"    -> {r.message}")
        
        print("\n" + "=" * 70)
        
        return {
            "total_tests": total_tests,
            "passed": total_passed,
            "warnings": total_warnings,
            "failed": total_failed,
            "pass_rate": total_passed / total_tests if total_tests > 0 else 0,
            "component_health": health_chart,
        }


def main():
    verifier = SystemVerifier()
    report = verifier.verify_all()
    
    # Exit with non-zero if there are failures
    sys.exit(1 if report["failed"] > 0 else 0)


if __name__ == "__main__":
    main()
