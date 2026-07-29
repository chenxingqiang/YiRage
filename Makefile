# YiRage Makefile
# Common development tasks

# Pip post-install hits PyPI (and any extra-index-url) to compare versions; that
# often fails with SSLEOFError behind proxies. Disabling the check is harmless.
export PIP_DISABLE_PIP_VERSION_CHECK ?= 1

.PHONY: help install install-cpu install-mps install-dev format lint test test-cov test-e2e test-e2e-fast test-py-extended clean build docs pre-commit readme-pypi

# Default target
help:
	@echo "YiRage Development Commands"
	@echo "=========================="
	@echo ""
	@echo "Setup:"
	@echo "  make install        Install YiRage in development mode (editable)"
	@echo "  make install-cpu      Editable install with YIRAGE_BACKEND=cpu (native build)"
	@echo "  make install-mps      Editable install with YIRAGE_BACKEND=mps (Apple Silicon)"
	@echo "  make install-dev    Install with development dependencies"
	@echo "  make pre-commit     Setup pre-commit hooks"
	@echo ""
	@echo "Code Quality:"
	@echo "  make format         Format code (black, isort, clang-format)"
	@echo "  make lint           Run linters (flake8, mypy, clang-tidy)"
	@echo "  make check          Run pre-commit on all files"
	@echo ""
	@echo "Testing:"
	@echo "  make test           Default pytest (tests/python + tests/integration, per pyproject)"
	@echo "  make test-cov       Same as test with coverage HTML + terminal report"
	@echo "  make test-e2e       All E2E tests (includes slow superoptimize; may take long)"
	@echo "  make test-e2e-fast  E2E excluding slow + benchmark markers"
	@echo "  make test-cpp       Run C++ tests (ctest in build/)"
	@echo "  make test-py-extended  Default pytest + fast E2E (no superoptimize suite)"
	@echo "  make test-all       Python default pytest + C++ ctest"
	@echo ""
	@echo "Building:"
	@echo "  make build          Build the project"
	@echo "  make build-release  Build release version"
	@echo "  make clean          Clean build artifacts"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs           Build documentation"
	@echo "  make readme-pypi    Regenerate README.pypi.md from README.md"

# =============================================================================
# Setup
# =============================================================================

install:
	python3 -m pip install -e . --no-build-isolation

# Match AGENTS.md: native extensions need --no-build-isolation; PIP_DISABLE_* is exported above.
install-cpu:
	YIRAGE_BACKEND=cpu python3 -m pip install -e . --no-build-isolation

install-mps:
	YIRAGE_BACKEND=mps python3 -m pip install -e . --no-build-isolation

install-dev:
	python3 -m pip install -e ".[dev]"
	python3 -m pip install pre-commit black isort flake8 mypy pytest pytest-cov bandit
	$(MAKE) pre-commit

pre-commit:
	python3 -m pip install pre-commit
	pre-commit install
	pre-commit install --hook-type commit-msg
	@echo "Pre-commit hooks installed!"

# =============================================================================
# Code Quality
# =============================================================================

format:
	@echo "Formatting Python code..."
	black python/ tests/python/ scripts/ benchmark/ demo/ examples/ --line-length 100
	isort python/ tests/python/ scripts/ benchmark/ demo/ examples/ --profile black --line-length 100
	@echo "Formatting C++ code..."
	find src include tests/cpp -name '*.cc' -o -name '*.h' -o -name '*.cuh' -o -name '*.cu' | \
		xargs clang-format -i --style=file 2>/dev/null || true
	@echo "Done!"

format-check:
	black --check python/ tests/python/ scripts/ --line-length 100
	isort --check python/ tests/python/ scripts/ --profile black --line-length 100

lint:
	@echo "Running Python linters..."
	flake8 python/ --max-line-length=120 --ignore=E203,E501,W503,E731,E402 || true
	mypy python/yirage --ignore-missing-imports || true
	@echo "Done!"

check:
	pre-commit run --all-files

security:
	bandit -r python/yirage -ll

# =============================================================================
# Testing
# =============================================================================

test:
	python3 -m pytest -v --tb=short

test-cov:
	python3 -m pytest -v --tb=short --cov=python/yirage --cov-report=html --cov-report=term

# Full E2E (superoptimize, etc.); use test-e2e-fast for CI-style runs.
test-e2e:
	python3 -m pytest tests/e2e -v --tb=short

test-e2e-fast:
	python3 -m pytest tests/e2e -v --tb=short -m "not slow and not benchmark"

test-cpu-cert:
	python3 -m pytest tests/integration/test_cpu_op_contract.py tests/integration/test_cpu_search_explore_sync.py tests/integration/test_cpu_superoptimize_value.py -v --tb=short

test-cpu-cert-quick:
	python3 -m pytest tests/integration/test_cpu_op_contract.py tests/integration/test_cpu_search_explore_sync.py tests/integration/test_cpu_inventory_planned.py tests/integration/test_cpu_certification_profile.py -v --tb=short

test-cpu-cert-profile:
	PYTHONPATH=. python3 scripts/cpu_certification.py --json --quick

test-cpu-cert-full-profile:
	PYTHONPATH=. python3 scripts/cpu_certification.py --json --skip-walkthrough

test-cpu-demos:
	python3 -m pytest tests/integration/test_cpu_demos.py tests/integration/test_cpu_demo_loop.py tests/integration/test_cpu_loop_close.py -v --tb=short

test-serving-cpu-cert:
	PYTHONPATH=python python3 scripts/serving_cpu_cert.py

test-serving-yirage-core:
	LD_LIBRARY_PATH=build/abstract_subexpr/release:build/formal_verifier/release:$$LD_LIBRARY_PATH \
	YIRAGE_BACKEND=cpu PYTHONPATH=python:. python3 scripts/serving_cpu_cert.py --yirage-core

test-serving-yirage-core-pytest:
	LD_LIBRARY_PATH=build/abstract_subexpr/release:build/formal_verifier/release:$$LD_LIBRARY_PATH \
	YIRAGE_BACKEND=cpu PYTHONPATH=python:. python3 -m pytest tests/python/test_runtime_fusion_yirage_core.py -v --tb=short

test-serving-cpu-cert-full:
	PYTHONPATH=python python3 scripts/serving_cpu_cert.py --full

test-serving-cpu-cert-pytest:
	PYTHONPATH=python:tests/python python3 -m pytest tests/python/test_runtime_fusion_s1.py tests/python/test_runtime_fusion_s2_s3.py tests/python/test_runtime_fusion_s4_kv.py tests/python/test_runtime_fusion_s5_sm.py tests/python/test_runtime_fusion_s6_radix.py tests/python/test_runtime_fusion_s7_multi_capsule.py tests/python/test_runtime_fusion_s8_vllm_bench.py tests/python/test_runtime_fusion_s9_sglang_meta.py tests/python/test_runtime_fusion_s10_sglang_plugin.py tests/python/test_runtime_fusion_s11_vllm_e2e.py tests/python/test_runtime_fusion_s12_sglang_e2e.py tests/python/test_runtime_fusion_s13_vllm_paged_e2e.py tests/python/test_runtime_fusion_s14_yirage_core_e2e.py tests/python/test_runtime_fusion_s15_maca_serving.py tests/python/test_runtime_fusion_s16_metax_tiers.py tests/python/test_runtime_fusion_s17_maca_generation.py tests/python/test_runtime_fusion_s18_maca_baseline.py tests/python/test_runtime_fusion_s19_yirage_cpu_search.py tests/python/test_runtime_fusion_s20_coordinator_ray.py tests/python/test_runtime_fusion_torch.py tests/python/test_runtime_fusion_qwen05b_cpu_e2e.py tests/python/test_serving_ray_search.py tests/integration/test_serving_cpu_cert.py -v --tb=short

test-cpu-cert-walkthrough-profile:
	PYTHONPATH=. python3 scripts/cpu_certification.py --json --walkthrough-profile

test-cpu-cert-e2e-profile:
	PYTHONPATH=. python3 scripts/cpu_certification.py --json

test-cpu-loop-close: test-cpu-demos test-cpu-mlir-bench-contract test-cpu-cert-e2e-profile

test-cpu-loop-close-profile:
	PYTHONPATH=. python3 scripts/cpu_loop_close.py --json --quick

test-cpu-loop-close-profile-validate:
	@mkdir -p artifacts
	PYTHONPATH=. python3 scripts/cpu_loop_close.py --json --quick \
		--output artifacts/cpu-loop-close-quick.json
	PYTHONPATH=. python3 scripts/validate_loop_close_archive.py \
		artifacts/cpu-loop-close-quick.json \
		--metadata-output artifacts/cpu-loop-close-quick.meta.json

test-cpu-loop-close-archive:
	PYTHONPATH=. python3 scripts/cpu_loop_close.py --json

validate-cpu-loop-close-archive:
	@test -n "$(ARCHIVE)" || (echo "Usage: make validate-cpu-loop-close-archive ARCHIVE=path/to/cpu-loop-close.json" && exit 1)
	PYTHONPATH=. python3 scripts/validate_loop_close_archive.py "$(ARCHIVE)"

validate-cpu-loop-close-metadata:
	@test -n "$(META)" || (echo "Usage: make validate-cpu-loop-close-metadata META=path/to/meta.json [ARCHIVE=path]" && exit 1)
	PYTHONPATH=. python3 scripts/validate_loop_close_archive_metadata.py "$(META)" \
		$(if $(ARCHIVE),--archive "$(ARCHIVE)",) \
		$(if $(CHECK_STAGE_TIMEOUTS),--check-stage-timeouts,)

validate-mlir-ci-bundle:
	@test -n "$(BUNDLE)" || (echo "Usage: make validate-mlir-ci-bundle BUNDLE=path/to/mlir-ci-dir" && exit 1)
	PYTHONPATH=. python3 scripts/validate_mlir_ci_bundle.py "$(BUNDLE)"

regression-validate-mlir-ci-bundle:
	@test -n "$(SRC)" || (echo "Usage: make regression-validate-mlir-ci-bundle SRC=path DEST=path" && exit 1)
	@test -n "$(DEST)" || (echo "Usage: make regression-validate-mlir-ci-bundle SRC=path DEST=path" && exit 1)
	PYTHONPATH=. python3 scripts/regression_validate_mlir_ci_bundle.py \
		--source "$(SRC)" --dest "$(DEST)"

validate-mlir-ci-metadata-download:
	@test -n "$(SRC)" || (echo "Usage: make validate-mlir-ci-metadata-download SRC=path DEST=path" && exit 1)
	@test -n "$(DEST)" || (echo "Usage: make validate-mlir-ci-metadata-download SRC=path DEST=path" && exit 1)
	$(MAKE) regression-validate-mlir-ci-bundle SRC="$(SRC)" DEST="$(DEST)"

build-mlir-ci-bundle:
	@test -n "$(BUNDLE)" || (echo "Usage: make build-mlir-ci-bundle BUNDLE=dir WORKFLOW=cpu-mlir-jit-contract|cpu-mlir-ci-nightly [RUN_ID=] [SHA=] [WITH_DIALECT_SMOKE=1] [MANIFEST_ONLY=1]" && exit 1)
	@test -n "$(WORKFLOW)" || (echo "Usage: make build-mlir-ci-bundle BUNDLE=dir WORKFLOW=..." && exit 1)
	PYTHONPATH=. python3 scripts/build_mlir_ci_bundle.py \
		--bundle-dir "$(BUNDLE)" --workflow "$(WORKFLOW)" \
		$(if $(RUN_ID),--run-id "$(RUN_ID)",) \
		$(if $(SHA),--sha "$(SHA)",) \
		$(if $(WITH_DIALECT_SMOKE),--with-dialect-smoke,) \
		$(if $(MANIFEST_ONLY),--manifest-only,)

smoke-build-mlir-ci-bundle-manifest:
	@test -n "$(BUNDLE)" || (echo "Usage: make smoke-build-mlir-ci-bundle-manifest BUNDLE=dir WORKFLOW=..." && exit 1)
	@test -n "$(WORKFLOW)" || (echo "Usage: make smoke-build-mlir-ci-bundle-manifest BUNDLE=dir WORKFLOW=..." && exit 1)
	$(MAKE) build-mlir-ci-bundle BUNDLE="$(BUNDLE)" WORKFLOW="$(WORKFLOW)" MANIFEST_ONLY=1

regression-validate-loop-close-archive:
	@test -n "$(ARCHIVE)" || (echo "Usage: make regression-validate-loop-close-archive ARCHIVE=path META=path DEST=dir [CHECK_STAGE_TIMEOUTS=1] [REQUIRE_ALERT_ANNOTATION=1]" && exit 1)
	@test -n "$(META)" || (echo "Usage: make regression-validate-loop-close-archive ARCHIVE=path META=path DEST=dir" && exit 1)
	@test -n "$(DEST)" || (echo "Usage: make regression-validate-loop-close-archive ARCHIVE=path META=path DEST=dir" && exit 1)
	PYTHONPATH=. python3 scripts/regression_validate_loop_close_archive.py \
		--source-archive "$(ARCHIVE)" --source-meta "$(META)" --dest-dir "$(DEST)" \
		$(if $(CHECK_STAGE_TIMEOUTS),--check-stage-timeouts,) \
		$(if $(REQUIRE_ALERT_ANNOTATION),--require-alert-annotation,)

validate-loop-close-metadata-post-alert:
	@test -n "$(ARCHIVE)" || (echo "Usage: make validate-loop-close-metadata-post-alert ARCHIVE=path META=path DEST=dir [CHECK_STAGE_TIMEOUTS=1]" && exit 1)
	@test -n "$(META)" || (echo "Usage: make validate-loop-close-metadata-post-alert ARCHIVE=path META=path DEST=dir" && exit 1)
	@test -n "$(DEST)" || (echo "Usage: make validate-loop-close-metadata-post-alert ARCHIVE=path META=path DEST=dir" && exit 1)
	$(MAKE) regression-validate-loop-close-archive \
		ARCHIVE="$(ARCHIVE)" META="$(META)" DEST="$(DEST)" \
		$(if $(CHECK_STAGE_TIMEOUTS),CHECK_STAGE_TIMEOUTS=1,) \
		REQUIRE_ALERT_ANNOTATION=1

validate-loop-close-metadata-pre-alert:
	@test -n "$(ARCHIVE)" || (echo "Usage: make validate-loop-close-metadata-pre-alert ARCHIVE=path META=path DEST=dir" && exit 1)
	@test -n "$(META)" || (echo "Usage: make validate-loop-close-metadata-pre-alert ARCHIVE=path META=path DEST=dir" && exit 1)
	@test -n "$(DEST)" || (echo "Usage: make validate-loop-close-metadata-pre-alert ARCHIVE=path META=path DEST=dir" && exit 1)
	$(MAKE) regression-validate-loop-close-archive \
		ARCHIVE="$(ARCHIVE)" META="$(META)" DEST="$(DEST)" \
		CHECK_STAGE_TIMEOUTS=1

test-cpu-mlir-ci-bundle:
	python3 -m pytest tests/python/test_cpu_mlir_ci_bundle.py \
		tests/python/test_hardware_optimization_timing_contract.py \
		tests/python/test_loop_close_metadata_doc_contract.py -v --tb=short

emit-cpu-loop-close-timeout-alert:
	@test -n "$(META)" || (echo "Usage: make emit-cpu-loop-close-timeout-alert META=path/to/meta.json" && exit 1)
	PYTHONPATH=. python3 scripts/emit_loop_close_timeout_alert.py "$(META)"

check-loop-close-timing-doc:
	PYTHONPATH=. python3 scripts/render_loop_close_timing_doc.py --check

render-loop-close-timing-doc:
	PYTHONPATH=. python3 scripts/render_loop_close_timing_doc.py --write

check-loop-close-metadata-doc:
	PYTHONPATH=. python3 scripts/render_loop_close_metadata_doc.py --check

render-loop-close-metadata-doc:
	PYTHONPATH=. python3 scripts/render_loop_close_metadata_doc.py --write

check-loop-close-ci-artifact-doc:
	PYTHONPATH=. python3 scripts/render_loop_close_ci_artifact_doc.py --check

render-loop-close-ci-artifact-doc:
	PYTHONPATH=. python3 scripts/render_loop_close_ci_artifact_doc.py --write

check-loop-close-docs: check-loop-close-timing-doc check-loop-close-metadata-doc check-loop-close-ci-artifact-doc

smoke-check-loop-close-docs: check-loop-close-docs

render-loop-close-docs: render-loop-close-timing-doc render-loop-close-metadata-doc render-loop-close-ci-artifact-doc

test-cpu-mlir-dialect-smoke:
	python3 -m pytest tests/integration/test_fused_vs_mkl_baseline.py -v --tb=short -k "dialect_emit_path_smoke"

test-cpu-mlir-bench-contract:
	python3 -m pytest tests/python/test_bench_fusion_search_skip.py -v --tb=short -k "mlir_jit or concat_matmul or parse_bench or mlir_bench_profile or run_mlir_bench"

test-cpu-mlir-bench-profile:
	PYTHONPATH=. python3 scripts/cpu_mlir_bench_profile.py --json

test-cpu-value-verify:
	python3 -m pytest tests/integration/test_cpu_full_value_verify.py -v --tb=short

test-cpu-cert-full: test-cpu-value-verify test-cpu-cert

test-py-extended: test test-e2e-fast

test-cpp:
	@if [ -d "build" ]; then \
		cd build && ctest --output-on-failure; \
	else \
		echo "Build directory not found. Run 'make build' first."; \
	fi

test-all: test test-cpp

test-mps:
	python tests/mps/run_all_mps_tests.py

test-cuda:
	pytest tests/ -v -m cuda

test-ascend:
	python tests/ascend/test_superoptimize.py

# =============================================================================
# Building
# =============================================================================

build:
	mkdir -p build
	cd build && cmake .. -DCMAKE_BUILD_TYPE=Debug
	cd build && make -j$$(nproc)

build-release:
	mkdir -p build
	cd build && cmake .. -DCMAKE_BUILD_TYPE=Release
	cd build && make -j$$(nproc)

build-cuda:
	mkdir -p build
	cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DYIRAGE_BACKEND_CUDA_ENABLED=ON
	cd build && make -j$$(nproc)

build-mps:
	mkdir -p build
	cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DYIRAGE_BACKEND_MPS_ENABLED=ON
	cd build && make -j$$(nproc)

build-ascend:
	cp cmake/backends/ascend.cmake config.cmake
	mkdir -p build
	cd build && cmake .. -DCMAKE_BUILD_TYPE=Release
	cd build && make -j$$(nproc)

clean:
	rm -rf build/
	rm -rf *.egg-info
	rm -rf dist/
	rm -rf .eggs/
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -delete
	find . -name "*.so" -delete
	find . -name "*.o" -delete

clean-all: clean
	rm -rf .mypy_cache/
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage

# =============================================================================
# Documentation
# =============================================================================

docs:
	cd docs && make html

docs-clean:
	cd docs && make clean

readme-pypi:
	python3 tools/generate_pypi_readme.py

# =============================================================================
# Development Utilities
# =============================================================================

update-deps:
	python3 -m pip install --upgrade pip setuptools wheel
	python3 -m pip install --upgrade pre-commit black isort flake8 mypy pytest

check-env:
	@echo "Python: $$(python --version)"
	@echo "Pip: $$(pip --version)"
	@echo "CMake: $$(cmake --version | head -1)"
	@echo "Clang-format: $$(clang-format --version 2>/dev/null || echo 'Not installed')"
	@echo ""
	@echo "Pre-commit: $$(pre-commit --version 2>/dev/null || echo 'Not installed')"
	@echo "Black: $$(black --version 2>/dev/null || echo 'Not installed')"
	@echo "Flake8: $$(flake8 --version 2>/dev/null || echo 'Not installed')"

# Show available backends
backends:
	@python -c "import yirage as yr; print('Available backends:', yr.get_available_backends())" 2>/dev/null || \
		echo "YiRage not installed. Run 'make install' first."

# Generate compile_commands.json for IDE
compile-commands:
	mkdir -p build
	cd build && cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
	ln -sf build/compile_commands.json .
