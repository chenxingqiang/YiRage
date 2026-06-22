# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0

"""
Compilation Pipeline Stages

Provides modular, extensible pipeline stages for the compilation process.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Type
from enum import Enum, auto


class StageStatus(Enum):
    """Status of a pipeline stage."""

    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    SKIPPED = auto()


@dataclass
class StageResult:
    """Result from a pipeline stage."""

    status: StageStatus
    output: Any = None
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0


class PipelineStage(ABC):
    """Base class for compilation pipeline stages."""

    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled
        self._next_stage: Optional["PipelineStage"] = None

    @abstractmethod
    def run(self, input_data: Any, context: Dict[str, Any]) -> StageResult:
        """Execute this stage."""
        pass

    def chain(self, next_stage: "PipelineStage") -> "PipelineStage":
        """Chain another stage after this one."""
        self._next_stage = next_stage
        return next_stage

    def run_chain(self, input_data: Any, context: Dict[str, Any]) -> List[StageResult]:
        """Run this stage and all chained stages."""
        results = []

        if self.enabled:
            result = self.run(input_data, context)
            results.append(result)

            if result.status == StageStatus.FAILED:
                return results

            next_input = result.output
        else:
            results.append(StageResult(status=StageStatus.SKIPPED))
            next_input = input_data

        if self._next_stage:
            results.extend(self._next_stage.run_chain(next_input, context))

        return results


class SuperoptimizeStage(PipelineStage):
    """
    Stage 1: muGraph Superoptimization

    Searches for optimal kernel configurations using the muGraph search algorithm.
    """

    def __init__(
        self,
        backend: str = "cuda",
        max_iterations: int = 1000,
        use_ray: bool = False,
        num_workers: int = 4,
    ):
        super().__init__("superoptimize")
        self.backend = backend
        self.max_iterations = max_iterations
        self.use_ray = use_ray
        self.num_workers = num_workers

    def run(self, input_data: Any, context: Dict[str, Any]) -> StageResult:
        """Run superoptimization on the input graph."""
        import time

        start_time = time.time()

        try:
            graph = input_data

            # Get search configuration from context
            search_config = context.get("search_config", {})

            # Run superoptimization
            if hasattr(graph, "superoptimize"):
                results = graph.superoptimize(
                    backend=self.backend,
                    use_ray=self.use_ray,
                    num_workers=self.num_workers,
                    **search_config,
                )

                if results and len(results) > 0:
                    best_graph = results[0]
                    duration = time.time() - start_time

                    return StageResult(
                        status=StageStatus.COMPLETED,
                        output=best_graph,
                        metrics={
                            "num_candidates": len(results),
                            "backend": self.backend,
                        },
                        duration_seconds=duration,
                    )

            # Fallback: return original graph
            return StageResult(
                status=StageStatus.COMPLETED,
                output=graph,
                metrics={"num_candidates": 0},
                duration_seconds=time.time() - start_time,
            )

        except Exception as e:
            return StageResult(
                status=StageStatus.FAILED,
                error=str(e),
                duration_seconds=time.time() - start_time,
            )


class MLIRLoweringStage(PipelineStage):
    """
    Stage 2: MLIR Lowering

    Converts the optimized muGraph to MLIR representation for target-independent
    optimizations and code generation.
    """

    def __init__(
        self,
        target_dialect: str = "linalg",
        opt_level: int = 3,
    ):
        super().__init__("mlir_lowering")
        self.target_dialect = target_dialect
        self.opt_level = opt_level

    def run(self, input_data: Any, context: Dict[str, Any]) -> StageResult:
        """Convert graph to MLIR."""
        import time

        start_time = time.time()

        try:
            graph = input_data

            # Try to import MLIR converter
            import sys
            from pathlib import Path

            mlir_path = Path(__file__).parent.parent.parent.parent / "mlir" / "python"
            sys.path.insert(0, str(mlir_path))

            from mugraph_to_mlir import MuGraphToMLIR

            converter = MuGraphToMLIR()
            mlir_code = converter.convert(graph)

            duration = time.time() - start_time

            return StageResult(
                status=StageStatus.COMPLETED,
                output={"graph": graph, "mlir": mlir_code},
                metrics={
                    "mlir_lines": len(mlir_code.split("\n")),
                    "target_dialect": self.target_dialect,
                },
                duration_seconds=duration,
            )

        except ImportError:
            # MLIR not available, skip
            return StageResult(
                status=StageStatus.SKIPPED,
                output={"graph": input_data, "mlir": None},
                duration_seconds=time.time() - start_time,
            )
        except Exception as e:
            return StageResult(
                status=StageStatus.FAILED,
                error=str(e),
                duration_seconds=time.time() - start_time,
            )


class CodeGenStage(PipelineStage):
    """
    Stage 3: Code Generation

    Generates target-specific code (CUDA, Metal, CPU, etc.) from the optimized
    graph and/or MLIR representation.
    """

    def __init__(
        self,
        backend: str = "cuda",
        generate_profiling: bool = False,
    ):
        super().__init__("codegen")
        self.backend = backend
        self.generate_profiling = generate_profiling

    def run(self, input_data: Any, context: Dict[str, Any]) -> StageResult:
        """Generate target code."""
        import time

        start_time = time.time()

        try:
            if isinstance(input_data, dict):
                graph = input_data.get("graph")
                mlir_code = input_data.get("mlir")
            else:
                graph = input_data
                mlir_code = None

            generated_code = None

            # Try graph-based code generation
            if graph and hasattr(graph, "generate_code"):
                generated_code = graph.generate_code()

            duration = time.time() - start_time

            return StageResult(
                status=StageStatus.COMPLETED,
                output={
                    "graph": graph,
                    "mlir": mlir_code,
                    "code": generated_code,
                },
                metrics={
                    "backend": self.backend,
                    "code_lines": len(generated_code.split("\n")) if generated_code else 0,
                },
                duration_seconds=duration,
            )

        except Exception as e:
            return StageResult(
                status=StageStatus.FAILED,
                error=str(e),
                duration_seconds=time.time() - start_time,
            )


class CompilePipeline:
    """
    Full compilation pipeline combining all stages.

    Example:
        pipeline = CompilePipeline(backend='cuda')
        results = pipeline.run(graph)

        for stage_name, result in results.items():
            print(f"{stage_name}: {result.status.name}")
    """

    def __init__(
        self,
        backend: str = "cuda",
        enable_superoptimize: bool = True,
        enable_mlir: bool = True,
        search_config: Optional[Dict] = None,
        **kwargs,
    ):
        self.backend = backend
        self.search_config = search_config or {}

        # Create stages
        self.stages: List[PipelineStage] = []

        if enable_superoptimize:
            self.stages.append(
                SuperoptimizeStage(
                    backend=backend,
                    use_ray=kwargs.get("use_ray", False),
                    num_workers=kwargs.get("num_workers", 4),
                )
            )

        if enable_mlir:
            self.stages.append(
                MLIRLoweringStage(
                    opt_level=kwargs.get("mlir_opt_level", 3),
                )
            )

        self.stages.append(CodeGenStage(backend=backend))

        # Chain stages
        for i in range(len(self.stages) - 1):
            self.stages[i].chain(self.stages[i + 1])

    def run(self, graph: Any) -> Dict[str, StageResult]:
        """Run the full pipeline."""
        context = {
            "backend": self.backend,
            "search_config": self.search_config,
        }

        if self.stages:
            results_list = self.stages[0].run_chain(graph, context)
        else:
            results_list = []

        # Map results to stage names
        results = {}
        for i, stage in enumerate(self.stages):
            if i < len(results_list):
                results[stage.name] = results_list[i]

        return results

    def get_final_output(self, results: Dict[str, StageResult]) -> Any:
        """Get the final output from pipeline results."""
        # Get last successful stage output
        for stage in reversed(self.stages):
            if stage.name in results:
                result = results[stage.name]
                if result.status == StageStatus.COMPLETED:
                    return result.output
        return None
