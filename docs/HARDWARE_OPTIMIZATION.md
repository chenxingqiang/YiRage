# Same-Backend Hardware Optimization

YiRage optimizes computation **for the hardware that is actually installed in the current
environment**. Search, profiling, caching, and execution all use the **same backend** end to
end. The framework does **not** target cross-backend workflows such as searching on CPU and
executing on CUDA.

## Design principle

| Stage | What happens |
|-------|----------------|
| **Detect** | `superoptimize()` picks `backend` from `YIRAGE_BACKEND`, `get_default_backend()`, or PyTorch device probes (`cuda`, `mps`, `ascend`, …, else `cpu`). |
| **Configure** | Backend-specific search space from `python/yirage/backends/<backend>/config.py` (e.g. CPU: core count, SIMD from `/proc/cpuinfo`, grid/block tiling). |
| **Search** | µGraph enumeration + **fast fingerprint** verification (`ProbabilisticVerifier`, default). Optional **formal** verifier via `is_formal_verified=True` or `YIRAGE_FORMAL_VERIFY=1`. |
| **Profile** | Candidates are timed on the **target device** for that backend (CPU tensors on CPU, CUDA events on GPU, …). |
| **Cache** | MuGraph entries are keyed by `(graph_hash, backend, search_config)` under `~/.yirage/mugraphs/`. |
| **Execute** | `KNGraph.__call__` dispatches to `cpu_call`, `cuda_call`, `mps_call`, etc. on the same backend. |

```
Current environment  →  backend B
        ↓
Search config for B  →  µGraph candidates
        ↓
Verify equivalence  →  profile on B  →  pick fastest
        ↓
Save / reuse (backend=B)  →  run on B
```

## How capabilities fit

| Capability | Role (same-backend) |
|------------|---------------------|
| **µGraph search** | Explore equivalent graphs in a backend-specific search space. |
| **Ray** | Parallelize search partitions on the **same** backend (e.g. split `griddims` across CPU workers). |
| **MuGraph persistence** | Skip repeat search when the same graph + backend + config was optimized before. |
| **AccelForge** | Hardware co-design **prescreen** and RL reward signals for a **target** backend or accelerator model—not a bridge to run on a different device. |
| **RL (`ConstrainedGraphEnv`)** | Learn search policies under constraints; FINISH metrics come from the configured target graph / AccelForge design. |

## Evaluating success

Judge optimization on the **same backend** you deployed:

- **CPU host**: correctness after search; plain matmul within ~2× of `torch.matmul` (MKL); fused/customized graphs faster than an MKL baseline that runs unfused ops; search config reflecting local SIMD and core count.
- **CUDA host**: compile success, kernel latency via CUDA profiling, compute capability in saved metadata.
- **MPS / Ascend / MACA**: same pattern with each backend’s profile and execution path.

Do **not** use cross-backend comparisons (e.g. “CPU search value = CUDA speedup”) as the primary metric.

## CPU-specific notes

### Primitive layer: host BLAS (MKL)

On CPU, **plain `matmul` primitives delegate to `torch.matmul`**, which uses the host BLAS
(MKL on typical PyTorch Linux builds). YiRage does **not** re-search MKL micro-kernels in
µGraph space; it uses the same machine’s fastest GEMM for `kn_matmul_op` and TB matmul tiles.

Search value on CPU is **graph-level**: fusion, customized tiling, and memory traffic vs a
**MKL baseline** — not beating `torch.matmul` on an isolated plain GEMM.

Set `YIRAGE_CPU_NATIVE=1` only for experimental YiRage C++ SIMD GEMM (not the default).

### Search space

`get_cpu_search_config()` / `resolve_cpu_search_space()` detect AVX2 / AVX-512 / NEON, core
count, and problem shape to set `search_thread`, grid/block, and frange. `superoptimize(backend="cpu")`
profiles on CPU tensors through `_interpret_mugraph_on_cpu` (same path as `cpu_call`).

For an end-to-end demo on CPU:

```bash
export LD_LIBRARY_PATH=build/abstract_subexpr/release:build/formal_verifier/release:$LD_LIBRARY_PATH
export YIRAGE_BACKEND=cpu
PYTHONPATH=. python3 scripts/business_capability_walkthrough.py
PYTHONPATH=. pytest tests/integration/test_cpu_superoptimize_value.py -v
```

### Two-layer verification

Search and runtime use different checks; both matter on CPU:

| Layer | When | Mechanism | Python API |
|-------|------|-----------|------------|
| **Search (fast)** | Every µGraph candidate during `superoptimize` | C++ fingerprint (`ProbabilisticVerifier`) or optional formal (`FormalVerifier` + Rust) | `resolve_verifier_config()` → `is_formal_verified` |
| **Runtime (full)** | After search, before trusting benchmarks | `torch` reference vs `cpu_call` / interpreter | `runtime_verify_mugraph()` |

Default is **probabilistic fingerprint** (no extra flags). Enable formal search verification:

```bash
# env or explicit kwarg (requires USE_FORMAL_VERIFIER=ON at build time)
export YIRAGE_FORMAL_VERIFY=1
PYTHONPATH=. python3 scripts/bench_fused_vs_mkl_baseline.py --quick --formal-verify --json
```

Bench JSON reports `verification`, `search_verified` (candidate passed search verifier), and
`runtime_verified` (torch numeric check on the executed graph).

#### Bench: skip fusion search on P0 paths (``--quick`` and ``--full``)

``scripts/bench_fused_vs_mkl_baseline.py`` may skip ``superoptimize`` when the unfused seed
graph already matches a production host-BLAS fast path. This measures **runtime**
(``cpu_matmul``, ``cpu_rms_matmul``, etc.) vs MKL, not search time. ``--full`` uses larger
tensor shapes and more benchmark iterations. Skip avoids redundant search on P0 seeds. When
search runs (``YIRAGE_CPU_BENCH_SKIP_FUSION_SEARCH=0``), grid explore is capped to a single
``griddims``/``franges`` point so full benches finish in CI.

| Env | Default | Effect |
|-----|---------|--------|
| ``YIRAGE_CPU_BENCH_SKIP_FUSION_SEARCH`` | ``1`` | When ``1``, quick and full benches skip search for P0-eligible seeds |
| ``--full`` | off | Larger shapes / iters; when search runs, explore grid is still CI-capped |

JSON field ``fusion_search_skipped: true`` and ``search_time_s: 0`` indicate skip. Opt-in slow
workloads: ``--workloads matmul_chain`` or ``concat_matmul``. Force search:
``YIRAGE_CPU_BENCH_SKIP_FUSION_SEARCH=0``.

### CPU certification (op contract)

Single source of truth: ``docs/cpu_support_matrix.yaml`` (loaded by
``python/yirage/backends/cpu/support_matrix.py``).

| Tier | Meaning |
|------|---------|
| **supported** / **fast_path** | Must pass ``runtime_verify_mugraph`` vs torch in ``tests/integration/test_cpu_op_contract.py`` |
| **experimental** | Interpreter may work; no Python graph builder or incomplete semantics |
| **unsupported** | ``cpu_call`` / TB interpreter raises ``NotImplementedError`` (no silent clone) |

```bash
make test-cpu-cert
make test-cpu-cert-quick          # contract + inventory + profile helpers (~1s)
make test-cpu-cert-profile        # --quick JSON with value_verify_aligned (~32s)
make test-cpu-cert-full-profile   # full cert JSON (~35s)
make test-cpu-demos               # CPU demo smoke (optimize loop)
make test-cpu-cert-walkthrough-profile  # walkthrough substage timing JSON
make test-cpu-cert-e2e-profile   # full cert + walkthrough quick + mlir bench profile JSON
make test-cpu-loop-close         # demos + mlir contract + cert e2e (infinite loop close)
make test-cpu-loop-close-profile   # loop-close JSON archive (--quick, ~40s)
make test-cpu-loop-close-profile-validate  # quick archive + validate + metadata (~50s)
make test-cpu-loop-close-archive   # full loop-close JSON (includes cert e2e, ~130s)
make test-cpu-mlir-dialect-smoke   # dialect_lowered bench emit path (USE_MLIR=1)
make test-cpu-mlir-bench-contract  # MLIR JIT + concat deferred JSON contract unit tests
make test-cpu-mlir-bench-profile   # archive bench JSON + contract validation
# Profile JSON file: add --output artifacts/mlir-bench-profile.json
PYTHONPATH=. python3 scripts/cpu_certification.py --json
PYTHONPATH=. python3 scripts/cpu_certification.py --json --quick
PYTHONPATH=. python3 scripts/cpu_certification.py --json --skip-walkthrough
```

Full JSON reports include ``profile.stage_elapsed_s`` per stage and
``profile.value_verify_aligned`` (passed vs planned **110**).

**Cert timing baseline (VM reference, tractability on):**

| Target | Stages | Typical elapsed |
|--------|--------|-----------------|
| `test-cpu-cert-quick` | op contract + explore sync + inventory | ~1s |
| `test-cpu-cert-profile` | value verify (110) + op contract + native_gemm | ~32s |
| `test-cpu-cert-full-profile` | full cert + superoptimize smoke (no walkthrough) | ~35s |
| `cpu_certification.py --json` (full) | + business_capability_walkthrough | minutes (Ray/RL) |
| `test-cpu-cert-walkthrough-profile` | walkthrough quick + `walkthrough_substage_elapsed_s` | ~27s |
| `test-cpu-cert-e2e-profile` | full cert + walkthrough quick + mlir bench profile | ~70s |
| `test-cpu-mlir-bench-contract` | MLIR JIT + concat deferred JSON validator unit tests | ~1s |
| `test-cpu-mlir-bench-profile` | bench `--mlir-jit` JSON archive + concat contract | ~15s |
| `test-cpu-loop-close` | demos + mlir contract + cert e2e profile | ~100s |
| `test-cpu-loop-close-profile` | loop-close JSON archive (quick, no cert e2e) | ~40s |
| `test-cpu-loop-close-profile-validate` | quick archive + validate + metadata sidecar | ~50s |
| `test-cpu-loop-close-archive` | full loop-close JSON (nightly / manual) | ~130s |

Loop-close and MLIR CI Makefile helpers (Loop R84–R136; ``make smoke-check-loop-close-docs`` render check ``loop_close_ci_artifact_doc_bundle_sync_gate_check()``; render ``--check`` scripts support ``--doc-path``; test hook ``YIRAGE_LOOP_CLOSE_DOC_FORCE_SYNC_GATE_FAIL``=1 must be unset on smoke):

<!-- LOOP_CLOSE_MAKEFILE_HELPERS_TABLE_BEGIN -->
| Target | Purpose |
|--------|---------|
| `make check-loop-close-docs` | Verify timing + metadata + CI artifact/workflow-make tables and intro lines (``loop_close_doc_intro_line_doc_row_count()`` = 9 rows; sync gate ``cpu_mlir_ci_bundle_contract_doc_sync_gate_ok()``; manifest/blocks ``cpu_mlir_ci_bundle_contract_manifest_and_blocks_summary_sync_ok()``; ci_artifact test hook ``YIRAGE_LOOP_CLOSE_DOC_FORCE_SYNC_GATE_FAIL``=1 must be unset on smoke; ``loop_close_ci_artifact_doc_bundle_sync_gate_check_force_fail_env_doc_crossref()`` for sync gate stderr contract) |
| `make smoke-check-loop-close-docs` | Single smoke entry for all three render ``--check`` scripts (single source: ``loop_close_docs_smoke_make_target()``; bundle sync gate ``loop_close_ci_artifact_doc_bundle_sync_gate_check()`` via ci_artifact render ``--check`` (stderr ``loop_close_ci_artifact_doc_bundle_sync_gate_check_failure_snippet()``; test hook ``YIRAGE_LOOP_CLOSE_DOC_FORCE_SYNC_GATE_FAIL``=1 must be unset); cross-ref ``loop_close_doc_render_check_write_crossref_rows()`` → ``loop_close_doc_render_write_specs()`` + ``apply_loop_close_doc_render_write_block_replace``) |
| `make render-loop-close-docs` | Regenerate all loop-close doc tables + intro lines from single source |
| `make render-loop-close-timing-doc` | Regenerate timing table + ``loop_close_timing_table_doc_intro_line()`` (``render_loop_close_timing_doc.py`` ``--check``/``--write`` + ``--doc-path``) |
| `make render-loop-close-metadata-doc` | Regenerate metadata field table + ``loop_close_metadata_table_doc_intro_line()`` (``render_loop_close_metadata_doc.py`` ``--check``/``--write`` + ``--doc-path``) |
| `make render-loop-close-ci-artifact-doc` | Regenerate CI artifact/workflow-make tables + ``cpu_ci_*_doc_intro_line()`` (``render_loop_close_ci_artifact_doc.py`` ``--check``/``--write`` + ``--doc-path``) |
| `make build-mlir-ci-bundle BUNDLE=... WORKFLOW=...` | Build MLIR CI bundle (`MANIFEST_ONLY=1` writes manifest only) |
| `make smoke-build-mlir-ci-bundle-manifest ...` | Alias for `build-mlir-ci-bundle ... MANIFEST_ONLY=1` |
| `make regression-validate-loop-close-archive ARCHIVE=... META=... DEST=...` | Simulate download validate (`CHECK_STAGE_TIMEOUTS=1`, optional `REQUIRE_ALERT_ANNOTATION=1`) |
| `make validate-loop-close-metadata-pre-alert ARCHIVE=... META=... DEST=...` | Pre-alert download validate (`CHECK_STAGE_TIMEOUTS=1`) |
| `make validate-loop-close-metadata-post-alert ARCHIVE=... META=... DEST=...` | Post-alert download validate (`REQUIRE_ALERT_ANNOTATION=1`; optional `CHECK_STAGE_TIMEOUTS=1`) |
| `make regression-validate-mlir-ci-bundle SRC=... DEST=...` | Simulate MLIR bundle download validate |
| `make validate-mlir-ci-metadata-download SRC=... DEST=...` | Alias for MLIR bundle download validate |
| `make test-cpu-mlir-ci-bundle` | MLIR bundle + timing + metadata docs + **workflow artifact/smoke contract** tests (``cpu_mlir_ci_bundle_test_contract_manifest_row_count()`` = 56 rows; manifest/blocks ``cpu_mlir_ci_bundle_contract_manifest_and_blocks_summary_sync_ok()``; sync gate ``cpu_mlir_ci_bundle_contract_doc_sync_gate_ok()`` incl. intro ``loop_close_doc_intro_line_doc_row_count()`` = 9 rows; manifest/helpers parity ``loop_close_doc_makefile_helpers_manifest_helpers_parity_ok()`` / ``loop_close_doc_manifest_helpers_and_bundle_intro_parity_ok()``; bundle intro cross-ref manifest/helpers parity ``loop_close_doc_makefile_helpers_manifest_helpers_parity_ok()`` / ``loop_close_doc_manifest_helpers_and_bundle_intro_parity_ok()``; force-fail parity ``loop_close_doc_force_fail_crossref_and_check_row_parity_ok()``; subprocess ``loop_close_doc_force_fail_env_stripped_subprocess_env()``; mixed parse ``loop_close_doc_render_check_write_crossref_mixed_legacy_suffix_markdown_table()``; three-way intro ``loop_close_doc_intro_line_three_way_parity_ok()``; check/smoke ``loop_close_doc_mixed_parse_patched_doc_and_manifest_parity_subprocess_plan()`` / ``loop_close_doc_check_loop_close_docs_make_subprocess_argv()`` / ``loop_close_doc_mixed_parse_full_smoke_and_check_subprocess_argv_batches()``; manifest parity three-way ``loop_close_doc_intro_registry_and_bundle_manifest_parity_three_way_ok()`` / ``loop_close_doc_manifest_parity_three_way_ok()``) |
<!-- LOOP_CLOSE_MAKEFILE_HELPERS_TABLE_END -->

``make test-cpu-mlir-ci-bundle`` contract helpers (Loop R136; single source: ``cpu_mlir_ci_bundle_test_contract_manifest()``; 56 rows; intro registry ``loop_close_doc_intro_line_doc_row_count()`` = 9 rows; sync gate ``cpu_mlir_ci_bundle_contract_doc_sync_gate_ok()``; render check ``loop_close_ci_artifact_doc_bundle_sync_gate_check()``; force-fail parity ``loop_close_doc_force_fail_crossref_and_check_row_parity_ok()`` / ``loop_close_doc_force_fail_three_way_intro_parity_ok()``; manifest/helpers parity ``loop_close_doc_makefile_helpers_manifest_helpers_parity_ok()`` / ``loop_close_doc_manifest_helpers_and_bundle_intro_parity_ok()``; manifest parity three-way ``loop_close_doc_intro_registry_and_bundle_manifest_parity_three_way_ok()`` / ``loop_close_doc_manifest_parity_three_way_ok()``; render write blocks from ``loop_close_doc_render_write_block_specs()``; regenerate with ``render_loop_close_ci_artifact_doc.py`` or ``make check-loop-close-docs``):

<!-- LOOP_CLOSE_MLIR_CI_BUNDLE_CONTRACT_TABLE_BEGIN -->
| Test module | Single-source helper | Contract |
|-------------|----------------------|----------|
| `tests/python/test_cpu_mlir_ci_bundle.py` | `build_mlir_ci_bundle_manifest()` | MLIR CI bundle schema, manifest-only smoke, download regression |
| `tests/python/test_hardware_optimization_timing_contract.py` | `loop_close_timing_contract()` | Timing threshold table doc sync |
| `tests/python/test_loop_close_metadata_doc_contract.py` | `loop_close_doc_render_write_specs()` | Render intro+table idempotence + ``apply_loop_close_doc_render_write_block_replace`` dispatch per write spec (timing/metadata/ci_artifact) |
| `tests/python/test_loop_close_metadata_doc_contract.py` | `loop_close_doc_render_check_write_crossref_rows()` | Render ``--check`` specs cross-ref ``--write`` specs and ``replace_fn`` dispatch (``loop_close_doc_render_check_specs`` ↔ ``loop_close_doc_render_write_block_specs``) |
| `tests/python/test_loop_close_metadata_doc_contract.py` | `loop_close_doc_render_check_script_doc_crossref()` | Check/write cross-ref Check script column (``--check`` + ``--doc-path``; ``normalize_loop_close_doc_render_check_script_doc_label()`` backward compat) |
| `tests/python/test_loop_close_metadata_doc_contract.py` | `parse_hardware_optimization_doc_render_check_write_crossref_table()` | Check/write cross-ref table parse sync (``LOOP_CLOSE_DOC_RENDER_CHECK_WRITE_CROSSREF_TABLE``; 5/6-column + Check script suffix backward compat via ``normalize_loop_close_doc_render_check_script_doc_label()``) |
| `tests/python/test_loop_close_metadata_doc_contract.py` | `loop_close_ci_doc_render_path_triggers_crossref_scripts()` | CI path triggers cover all scripts in ``loop_close_doc_render_check_write_crossref_rows()`` |
| `tests/python/test_loop_close_metadata_doc_contract.py` | `loop_close_doc_render_write_block_specs()` | Render write marker sub-blocks incl. makefile helpers, intro line registry, and paired ``replace_fn`` column (ci_artifact → ``replace_loop_close_doc_marker_block``) |
| `tests/python/test_loop_close_metadata_doc_contract.py` | `loop_close_doc_intro_line_specs()` | Doc intro lines sync with marker blocks (incl. ``loop_close_doc_makefile_helpers_loop_range()``) |
| `tests/python/test_loop_close_metadata_doc_contract.py` | `loop_close_doc_intro_line_doc_rows()` | Doc intro marker ↔ intro_fn ↔ loop label ↔ schema ↔ marker_section cross-ref (``LOOP_CLOSE_DOC_INTRO_LINE_TABLE``; metadata → ``loop_close_metadata_doc_marker_section()``) |
| `tests/python/test_loop_close_metadata_doc_contract.py` | `loop_close_doc_intro_line_doc_row_count()` | Intro line registry row count + ``cpu_mlir_ci_bundle_contract_doc_sync_gate_ok()`` (``LOOP_CLOSE_DOC_INTRO_LINE_TABLE``; combined with manifest/crossref parity) |
| `tests/python/test_loop_close_metadata_doc_contract.py` | `parse_hardware_optimization_doc_intro_line_table()` | Intro line table parse incl. Schema + Marker section columns (metadata row ``marker_section_fn`` → ``loop_close_metadata_doc_marker_section()``) |
| `tests/python/test_loop_close_metadata_doc_contract.py` | `apply_loop_close_doc_render_write_block_replace()` | Render write block ``replace_fn`` dispatch per ``loop_close_doc_render_write_block_specs()`` (timing/metadata render modules; ci_artifact → ``replace_loop_close_doc_marker_block``) |
| `tests/python/test_loop_close_metadata_doc_contract.py` | `parse_hardware_optimization_doc_render_write_block_table()` | Render write sub-block table parse incl. Replace function column (``LOOP_CLOSE_DOC_RENDER_WRITE_BLOCK_TABLE``; 3/4-column backward compat) |
| `tests/python/test_loop_close_metadata_doc_contract.py` | `loop_close_doc_render_check_write_crossref_block_count_parity()` | Cross-ref ``block_count`` column matches render write sub-block counts per ``loop_close_doc_render_write_block_specs()`` write spec |
| `tests/python/test_loop_close_metadata_doc_contract.py` | `cpu_mlir_ci_bundle_contract_doc_sync_gate_ok()` | Bundle manifest row count + crossref ``block_count`` parity + ``loop_close_doc_intro_line_doc_row_count()`` combined sync gate |
| `tests/python/test_loop_close_metadata_doc_contract.py` | `loop_close_doc_render_check_write_crossref_blocks_summary_parity()` | Cross-ref intro ``block_counts_summary`` matches doc ``Blocks`` column per row; combined with manifest row count via ``cpu_mlir_ci_bundle_contract_manifest_and_blocks_summary_sync_ok()`` |
| `tests/python/test_loop_close_metadata_doc_contract.py` | `cpu_mlir_ci_bundle_contract_manifest_and_blocks_summary_sync_ok()` | Bundle manifest row count + crossref blocks summary parity combined sync gate (Makefile helpers cross-ref on ``check-loop-close-docs`` / ``test-cpu-mlir-ci-bundle``) |
| `tests/python/test_loop_close_metadata_doc_contract.py` | `loop_close_ci_artifact_doc_bundle_sync_gate_check()` | Render ``--check`` hook for bundle manifest doc row count + combined sync gates (stderr ``loop_close_ci_artifact_doc_bundle_sync_gate_check_failure_snippet()``) |
| `tests/python/test_loop_close_metadata_doc_contract.py` | `loop_close_ci_artifact_doc_bundle_sync_gate_check_failure_snippet()` | Render ``--check`` stderr fragment when bundle sync gate fails (``render_loop_close_ci_artifact_doc.py``; cross-ref ``loop_close_ci_artifact_doc_bundle_sync_gate_check()``; test hook ``loop_close_ci_artifact_doc_bundle_sync_gate_check_force_fail_env()``) |
| `tests/python/test_loop_close_metadata_doc_contract.py` | `loop_close_ci_artifact_doc_bundle_sync_gate_check_force_fail_env()` | Env var to force sync gate failure stderr in render ``--check`` (``make check-loop-close-docs`` contract tests; doc cross-ref ``loop_close_ci_artifact_doc_bundle_sync_gate_check_force_fail_env_doc_crossref()``) |
| `tests/python/test_loop_close_metadata_doc_contract.py` | `loop_close_ci_artifact_doc_bundle_sync_gate_check_force_fail_env_doc_crossref()` | Doc cross-ref label for force-fail env contract tests (Makefile helpers smoke row ``must be unset`` parity) |
| `tests/python/test_loop_close_metadata_doc_contract.py` | `loop_close_ci_artifact_doc_bundle_sync_gate_check_force_fail_enabled()` | Returns True when ``loop_close_ci_artifact_doc_bundle_sync_gate_check_force_fail_env()`` is ``1`` (must be unset during normal ``make smoke-check-loop-close-docs``) |
| `tests/python/test_loop_close_metadata_doc_contract.py` | `loop_close_doc_force_fail_crossref_and_check_row_parity_ok()` | Crossref intro and ``make check-loop-close-docs`` helpers row share force-fail doc cross-ref fragments (``loop_close_doc_makefile_helpers_check_row_force_fail_purpose_fragment()`` + ``loop_close_doc_render_check_write_crossref_force_fail_intro_fragment()``) |
| `tests/python/test_loop_close_metadata_doc_contract.py` | `loop_close_doc_force_fail_env_stripped_subprocess_env()` | Subprocess env with force-fail hook unset + ``PYTHONPATH`` (``make smoke-check-loop-close-docs`` / ``make check-loop-close-docs`` success tests) |
| `tests/python/test_loop_close_metadata_doc_contract.py` | `loop_close_doc_render_check_write_crossref_mixed_legacy_suffix_markdown_table()` | Mixed legacy/suffix Check script labels for crossref parse backward-compat (``parse_hardware_optimization_doc_render_check_write_crossref_table()`` + sync gate) |
| `tests/python/test_loop_close_metadata_doc_contract.py` | `loop_close_doc_makefile_helpers_manifest_new_helpers_crossref_ok()` | Makefile helpers ``make test-cpu-mlir-ci-bundle`` row cross-ref's R129 manifest helpers (``loop_close_doc_makefile_helpers_manifest_new_helpers_crossref()``) |
| `tests/python/test_loop_close_metadata_doc_contract.py` | `loop_close_doc_force_fail_three_way_intro_parity_ok()` | Helpers/bundle/crossref intro lines share render check hook + force-fail parity (``loop_close_doc_render_check_write_crossref_force_fail_intro_fragment()``) |
| `tests/python/test_loop_close_metadata_doc_contract.py` | `loop_close_doc_render_check_subprocess_argv_chain()` | Render ``--check`` argv chain mirroring ``make check-loop-close-docs`` (``loop_close_doc_render_check_subprocess_argv()`` + ``--doc-path``) |
| `tests/python/test_loop_close_metadata_doc_contract.py` | `loop_close_doc_makefile_helpers_manifest_helpers_parity_ok()` | Makefile helpers ``make test-cpu-mlir-ci-bundle`` row cross-ref's all manifest doc-contract helpers (``loop_close_doc_makefile_helpers_manifest_helpers_crossref()``) |
| `tests/python/test_loop_close_metadata_doc_contract.py` | `loop_close_doc_intro_line_three_way_parity_ok()` | Intro line registry rows mark three-way force-fail parity gate (``loop_close_doc_force_fail_three_way_intro_parity_ok()`` on helpers/bundle/crossref) |
| `tests/python/test_loop_close_metadata_doc_contract.py` | `loop_close_docs_smoke_check_make_subprocess_argv()` | Argv for ``make smoke-check-loop-close-docs`` subprocess mixed-parse chain tests (``loop_close_docs_smoke_make_target()``; see ``loop_close_doc_mixed_parse_check_and_smoke_subprocess_argv()``) |
| `tests/python/test_loop_close_metadata_doc_contract.py` | `loop_close_doc_manifest_helpers_and_bundle_intro_parity_ok()` | Makefile helpers test row + bundle intro share manifest/helpers parity cross-ref (``loop_close_doc_bundle_intro_manifest_helpers_parity_fragment()``) |
| `tests/python/test_loop_close_metadata_doc_contract.py` | `loop_close_doc_mixed_parse_check_and_smoke_subprocess_argv()` | Mixed-parse subprocess argv: ``loop_close_doc_render_check_subprocess_argv_chain()`` + ``loop_close_docs_smoke_check_make_subprocess_argv()``; doc patch via ``loop_close_doc_render_check_write_crossref_mixed_legacy_suffix_patched_doc_text()`` |
| `tests/python/test_loop_close_metadata_doc_contract.py` | `loop_close_doc_render_check_write_crossref_mixed_legacy_suffix_patched_doc_text()` | Mixed legacy/suffix cross-ref table patched into doc text for subprocess tests (``replace_loop_close_doc_marker_block`` + mixed table helper) |
| `tests/python/test_loop_close_metadata_doc_contract.py` | `loop_close_doc_intro_line_bundle_manifest_parity_ok()` | Bundle intro registry row marks merged manifest parity three-way gate (``loop_close_doc_intro_registry_and_bundle_manifest_parity_three_way_ok()``; ``loop_close_doc_bundle_intro_manifest_helpers_parity_fragment()``) |
| `tests/python/test_loop_close_metadata_doc_contract.py` | `loop_close_doc_makefile_helpers_test_row_manifest_parity_doc_parity_ok()` | Doc Makefile helpers ``make test-cpu-mlir-ci-bundle`` row matches single source purpose fragment (``loop_close_doc_makefile_helpers_manifest_helpers_parity_purpose_fragment()``; see ``loop_close_doc_intro_registry_and_bundle_manifest_parity_three_way_ok()``) |
| `tests/python/test_loop_close_metadata_doc_contract.py` | `loop_close_doc_intro_registry_and_bundle_manifest_parity_three_way_ok()` | Intro registry manifest gate + bundle intro + helpers test row three-way (``loop_close_doc_bundle_intro_manifest_helpers_parity_fragment()``; see ``loop_close_doc_manifest_parity_three_way_ok()``) |
| `tests/python/test_loop_close_metadata_doc_contract.py` | `loop_close_doc_manifest_parity_three_way_ok()` | Alias for merged intro registry + bundle + helpers manifest parity gate (``loop_close_doc_intro_registry_and_bundle_manifest_parity_three_way_ok()``) |
| `tests/python/test_loop_close_metadata_doc_contract.py` | `loop_close_doc_check_loop_close_docs_make_subprocess_argv()` | Argv for ``make check-loop-close-docs`` canonical doc subprocess mixed-parse plan (``loop_close_doc_mixed_parse_patched_doc_and_manifest_parity_subprocess_plan()``) |
| `tests/python/test_loop_close_metadata_doc_contract.py` | `loop_close_doc_mixed_parse_full_smoke_and_check_subprocess_argv_batches()` | Full patched smoke+check argv batches ending with canonical ``make check-loop-close-docs`` (``loop_close_doc_mixed_parse_patched_doc_and_manifest_parity_subprocess_plan()``) |
| `tests/python/test_loop_close_metadata_doc_contract.py` | `loop_close_doc_mixed_parse_patched_doc_and_manifest_parity_subprocess_plan()` | Mixed-parse patched doc subprocess plan: patched + full smoke/check argv batches + manifest parity (``loop_close_doc_mixed_parse_patched_doc_and_manifest_parity_subprocess_python_snippet()``; ``loop_close_doc_mixed_parse_full_smoke_and_check_subprocess_argv_batches()``) |
| `tests/python/test_loop_close_metadata_doc_contract.py` | `loop_close_doc_render_write_block_counts_by_write_spec()` | Render write sub-block counts per write spec (cross-ref ``Blocks`` column source) |
| `tests/python/test_loop_close_metadata_doc_contract.py` | `cpu_mlir_ci_bundle_test_contract_manifest_row_count()` | Bundle contract doc table row count matches manifest (``LOOP_CLOSE_MLIR_CI_BUNDLE_CONTRACT_TABLE``; ``cpu_mlir_ci_bundle_contract_manifest_and_blocks_summary_sync_ok()``) |
| `tests/python/test_loop_close_metadata_doc_contract.py` | `loop_close_doc_bundle_loop_revision()` | Loop revision for all doc intro lines + bundle/render marker tables |
| `tests/python/test_loop_close_metadata_doc_contract.py` | `parse_hardware_optimization_makefile_helpers_table()` | Makefile helpers marker table parse sync (``LOOP_CLOSE_MAKEFILE_HELPERS_TABLE``) |
| `tests/python/test_loop_close_metadata_doc_contract.py` | `parse_hardware_optimization_metadata_doc_fields()` | Metadata field names from marker block section only (no full-doc backtick scan) |
| `tests/python/test_loop_close_metadata_doc_contract.py` | `loop_close_metadata_doc_marker_section()` | Bounded metadata intro + marker block for field/schema parse (``_metadata_doc_marker_section``) |
| `tests/python/test_loop_close_metadata_doc_contract.py` | `loop_close_doc_makefile_helpers_loop_range()` | Makefile helpers section Loop range label (``loop_close_doc_makefile_helpers_doc_intro_line()``) |
| `tests/python/test_loop_close_metadata_doc_contract.py` | `loop_close_ci_docs_gate_workflows()` | Workflows whose docs sync gate invokes ``loop_close_docs_smoke_make_target()`` |
| `tests/python/test_loop_close_metadata_doc_contract.py` | `loop_close_docs_smoke_make_target()` | CI docs sync gate Makefile target (``make smoke-check-loop-close-docs``) |
| `tests/python/test_loop_close_metadata_doc_contract.py` | `loop_close_ci_doc_render_path_triggers()` | Unified Makefile + render check/write script CI path triggers |
| `tests/python/test_loop_close_metadata_doc_contract.py` | `loop_close_ci_docs_gate_step_names()` | Docs sync gate human step names in workflow YAML |
| `tests/python/test_loop_close_metadata_doc_contract.py` | `cpu_ci_workflow_path_symmetry_doc_rows()` | Loop-close + MLIR workflow path-filter symmetry (render marker block) |
| `tests/python/test_loop_close_metadata_doc_contract.py` | `cpu_ci_artifact_manifest()` | CI artifact names documented and workflow-aligned |
| `tests/python/test_loop_close_metadata_doc_contract.py` | `cpu_ci_workflow_make_target_manifest()` | Workflow step ↔ Makefile target mapping |
<!-- LOOP_CLOSE_MLIR_CI_BUNDLE_CONTRACT_TABLE_END -->

Loop-close doc render write marker sub-blocks (Loop R136; single source: ``loop_close_doc_render_write_block_specs()``; check/write cross-ref table at ``LOOP_CLOSE_DOC_RENDER_CHECK_WRITE_CROSSREF_TABLE``; ci_artifact blocks use paired ``replace_loop_close_doc_marker_block`` via ``render_loop_close_ci_artifact_doc``; regenerate with ``render_loop_close_ci_artifact_doc.py`` or ``make check-loop-close-docs``):

<!-- LOOP_CLOSE_DOC_RENDER_WRITE_BLOCK_TABLE_BEGIN -->
| Write spec | Marker begin | Table function | Replace function |
|------------|--------------|----------------|------------------|
| `timing` | `LOOP_CLOSE_TIMING_TABLE_BEGIN` | `loop_close_timing_markdown_table()` | `replace_timing_table_markers()` |
| `metadata` | `LOOP_CLOSE_METADATA_FIELDS_BEGIN` | `loop_close_metadata_doc_markdown_table()` | `replace_metadata_table_markers()` |
| `ci_artifact` | `LOOP_CLOSE_MAKEFILE_HELPERS_TABLE_BEGIN` | `loop_close_doc_makefile_helpers_doc_markdown_table()` | `replace_loop_close_doc_marker_block()` |
| `ci_artifact` | `LOOP_CLOSE_MLIR_CI_BUNDLE_CONTRACT_TABLE_BEGIN` | `cpu_mlir_ci_bundle_contract_doc_markdown_table()` | `replace_loop_close_doc_marker_block()` |
| `ci_artifact` | `LOOP_CLOSE_DOC_RENDER_WRITE_BLOCK_TABLE_BEGIN` | `loop_close_doc_render_write_block_doc_markdown_table()` | `replace_loop_close_doc_marker_block()` |
| `ci_artifact` | `LOOP_CLOSE_DOC_RENDER_CHECK_WRITE_CROSSREF_TABLE_BEGIN` | `loop_close_doc_render_check_write_crossref_doc_markdown_table()` | `replace_loop_close_doc_marker_block()` |
| `ci_artifact` | `LOOP_CLOSE_DOC_INTRO_LINE_TABLE_BEGIN` | `loop_close_doc_intro_line_doc_markdown_table()` | `replace_loop_close_doc_marker_block()` |
| `ci_artifact` | `LOOP_CLOSE_CI_PATH_SYMMETRY_TABLE_BEGIN` | `cpu_ci_path_symmetry_doc_markdown_table()` | `replace_loop_close_doc_marker_block()` |
| `ci_artifact` | `LOOP_CLOSE_CI_ARTIFACT_TABLE_BEGIN` | `cpu_ci_artifact_doc_markdown_table()` | `replace_loop_close_doc_marker_block()` |
| `ci_artifact` | `LOOP_CLOSE_CI_WORKFLOW_MAKE_TABLE_BEGIN` | `cpu_ci_workflow_make_step_doc_markdown_table()` | `replace_loop_close_doc_marker_block()` |
<!-- LOOP_CLOSE_DOC_RENDER_WRITE_BLOCK_TABLE_END -->

Loop-close render check/write cross-reference (Loop R136; single source: ``loop_close_doc_render_check_write_crossref_rows()``; 3 rows; block counts via ``loop_close_doc_render_write_block_counts_by_write_spec()`` (ci_artifact=8, metadata=1, timing=1); ci_artifact bundle sync gate ``loop_close_ci_artifact_doc_bundle_sync_gate_check()`` (stderr ``loop_close_ci_artifact_doc_bundle_sync_gate_check_failure_snippet()``; doc cross-ref ``loop_close_ci_artifact_doc_bundle_sync_gate_check_force_fail_env_doc_crossref()`` = ``YIRAGE_LOOP_CLOSE_DOC_FORCE_SYNC_GATE_FAIL``=1); Check script column includes ``--check`` + ``--doc-path``; ``Blocks`` parity via ``loop_close_doc_render_check_write_crossref_blocks_summary_parity()``; regenerate with ``render_loop_close_ci_artifact_doc.py`` or ``make check-loop-close-docs``):

<!-- LOOP_CLOSE_DOC_RENDER_CHECK_WRITE_CROSSREF_TABLE_BEGIN -->
| Check name | Check script | Write module | Write fn | Replace fns | Blocks |
|------------|--------------|--------------|----------|-------------|--------|
| `timing` | `scripts/render_loop_close_timing_doc.py --check --doc-path` | `scripts.render_loop_close_timing_doc` | `write_timing_table_to_doc()` | `replace_timing_table_markers` | `1` |
| `metadata` | `scripts/render_loop_close_metadata_doc.py --check --doc-path` | `scripts.render_loop_close_metadata_doc` | `write_metadata_table_to_doc()` | `replace_metadata_table_markers` | `1` |
| `ci_artifact` | `scripts/render_loop_close_ci_artifact_doc.py --check --doc-path` | `scripts.render_loop_close_ci_artifact_doc` | `write_ci_artifact_tables_to_doc()` | `replace_loop_close_doc_marker_block` | `8` |
<!-- LOOP_CLOSE_DOC_RENDER_CHECK_WRITE_CROSSREF_TABLE_END -->

Loop-close doc intro line registry (Loop R136; single source: ``loop_close_doc_intro_line_doc_rows()``; makefile helpers range via ``loop_close_doc_makefile_helpers_loop_range()``; metadata schema + ``loop_close_metadata_doc_marker_section()`` cross-ref on metadata row; three-way intro parity via ``loop_close_doc_intro_line_three_way_parity_ok()``; bundle manifest parity gate via ``loop_close_doc_intro_registry_and_bundle_manifest_parity_three_way_ok()`` / ``loop_close_doc_intro_line_bundle_manifest_parity_ok()`` / ``loop_close_doc_manifest_parity_three_way_ok()``; regenerate with ``render_loop_close_ci_artifact_doc.py`` or ``make check-loop-close-docs``):

<!-- LOOP_CLOSE_DOC_INTRO_LINE_TABLE_BEGIN -->
| Marker begin | Intro function | Loop label | Schema | Marker section | Intro parity gate | Manifest parity gate |
|--------------|----------------|------------|--------|----------------|-------------------|----------------------|
| `LOOP_CLOSE_MAKEFILE_HELPERS_TABLE_BEGIN` | `loop_close_doc_makefile_helpers_doc_intro_line()` | `R84–R136` | `-` | `-` | `loop_close_doc_force_fail_three_way_intro_parity_ok()` | `-` |
| `LOOP_CLOSE_TIMING_TABLE_BEGIN` | `loop_close_timing_table_doc_intro_line()` | `R136` | `-` | `-` | `-` | `-` |
| `LOOP_CLOSE_METADATA_FIELDS_BEGIN` | `loop_close_metadata_table_doc_intro_line()` | `R136` | `loop_close_artifact_metadata_v2` | `loop_close_metadata_doc_marker_section()` | `-` | `-` |
| `LOOP_CLOSE_MLIR_CI_BUNDLE_CONTRACT_TABLE_BEGIN` | `cpu_mlir_ci_bundle_contract_doc_intro_line()` | `R136` | `-` | `-` | `loop_close_doc_force_fail_three_way_intro_parity_ok()` | `loop_close_doc_intro_registry_and_bundle_manifest_parity_three_way_ok()` |
| `LOOP_CLOSE_DOC_RENDER_WRITE_BLOCK_TABLE_BEGIN` | `loop_close_doc_render_write_block_doc_intro_line()` | `R136` | `-` | `-` | `-` | `-` |
| `LOOP_CLOSE_DOC_RENDER_CHECK_WRITE_CROSSREF_TABLE_BEGIN` | `loop_close_doc_render_check_write_crossref_doc_intro_line()` | `R136` | `-` | `-` | `loop_close_doc_force_fail_three_way_intro_parity_ok()` | `-` |
| `LOOP_CLOSE_CI_PATH_SYMMETRY_TABLE_BEGIN` | `cpu_ci_path_symmetry_doc_intro_line()` | `R136` | `-` | `-` | `-` | `-` |
| `LOOP_CLOSE_CI_ARTIFACT_TABLE_BEGIN` | `cpu_ci_artifact_doc_intro_line()` | `R136` | `-` | `-` | `-` | `-` |
| `LOOP_CLOSE_CI_WORKFLOW_MAKE_TABLE_BEGIN` | `cpu_ci_workflow_make_doc_intro_line()` | `R136` | `-` | `-` | `-` | `-` |
<!-- LOOP_CLOSE_DOC_INTRO_LINE_TABLE_END -->

CI workflow path-filter symmetry (Loop R136; single source: ``cpu_ci_workflow_path_symmetry_doc_rows()``; regenerate with ``render_loop_close_ci_artifact_doc.py`` or ``make check-loop-close-docs``):

<!-- LOOP_CLOSE_CI_PATH_SYMMETRY_TABLE_BEGIN -->
| Left | Right | Contract |
|------|-------|----------|
| `cpu-loop-close-nightly.yml pull_request` | `cpu-loop-close-pr.yml pull_request` | Identical path filters |
| `cpu-loop-close-pr.yml push` | `cpu-loop-close-nightly.yml pull_request` | Identical path filters; asserted by ``test_loop_close_pr_push_paths_symmetric_with_nightly_pull_request`` |
| `cpu-loop-close-nightly.yml push` | `cpu-loop-close-pr.yml pull_request` | Identical path filters; asserted by ``test_loop_close_nightly_push_paths_symmetric_with_pr_pull_request`` |
| `cpu-loop-close-nightly.yml push` | `cpu-loop-close-nightly.yml pull_request` | Identical path filters |
| `cpu-loop-close-pr.yml push` | `cpu-loop-close-pr.yml pull_request` | Identical path filters |
| `cpu-mlir-jit-contract.yml push` | `cpu-mlir-ci-nightly.yml pull_request` | Identical path filters |
| `cpu-mlir-jit-contract.yml pull_request` | `cpu-mlir-ci-nightly.yml pull_request` | Identical path filters |
| `cpu-mlir-ci-nightly.yml push` | `cpu-mlir-jit-contract.yml pull_request` | Identical path filters |
<!-- LOOP_CLOSE_CI_PATH_SYMMETRY_TABLE_END -->

CI workflow artifact names (Loop R136; single source: ``cpu_ci_artifact_manifest()``; regenerate with ``render_loop_close_ci_artifact_doc.py`` or ``make check-loop-close-docs``):

<!-- LOOP_CLOSE_CI_ARTIFACT_TABLE_BEGIN -->
| Workflow | Artifact name | Upload paths (under `artifacts/`) |
|----------|---------------|-----------------------------------|
| `cpu-loop-close-pr.yml` | `cpu-loop-close-quick-pr-${{ github.run_id }}` | quick JSON/meta, timeout alert, `downloaded-regression/`, `downloaded-regression-post-alert/` |
| `cpu-loop-close-nightly.yml` | `cpu-loop-close-json-${{ github.run_id }}` | full JSON/meta, timeout alert, `downloaded-regression/`, `downloaded-regression-post-alert/` |
| `cpu-mlir-jit-contract.yml` | `mlir-ci-bundle-${{ github.run_id }}` | full MLIR CI bundle directory |
| `cpu-mlir-jit-contract.yml` | `mlir-downloaded-regression-${{ github.run_id }}` | `downloaded-regression-mlir/` |
| `cpu-mlir-ci-nightly.yml` | `mlir-ci-nightly-${{ github.run_id }}` | full MLIR CI bundle directory |
| `cpu-mlir-ci-nightly.yml` | `mlir-downloaded-regression-nightly-${{ github.run_id }}` | `downloaded-regression-mlir/` |
<!-- LOOP_CLOSE_CI_ARTIFACT_TABLE_END -->

CI workflow step ↔ Makefile target mapping (Loop R136; single source: ``cpu_ci_workflow_make_target_manifest()``; docs sync gate step names from ``loop_close_ci_docs_gate_step_names()``; regenerate with ``render_loop_close_ci_artifact_doc.py`` or ``make check-loop-close-docs``):

<!-- LOOP_CLOSE_CI_WORKFLOW_MAKE_TABLE_BEGIN -->
| Workflow | Step (human name) | Makefile target |
|----------|-------------------|-----------------|
| `cpu-loop-close-pr.yml` | Loop-close docs sync gate (PR) | `make smoke-check-loop-close-docs` |
| `cpu-loop-close-pr.yml` | Pre-alert metadata validate (PR) | `make validate-loop-close-metadata-pre-alert` |
| `cpu-loop-close-pr.yml` | Post-alert metadata validate (PR) | `make validate-loop-close-metadata-post-alert` |
| `cpu-loop-close-pr.yml` | MLIR CI bundle regression tests | `make test-cpu-mlir-ci-bundle` |
| `cpu-loop-close-nightly.yml` | Loop-close docs sync gate (nightly) | `make smoke-check-loop-close-docs` |
| `cpu-loop-close-nightly.yml` | Pre-alert metadata validate (nightly) | `make validate-loop-close-metadata-pre-alert` |
| `cpu-loop-close-nightly.yml` | Post-alert metadata validate (nightly) | `make validate-loop-close-metadata-post-alert` |
| `cpu-mlir-jit-contract.yml` | Loop-close docs sync gate | `make smoke-check-loop-close-docs` |
| `cpu-mlir-jit-contract.yml` | MLIR CI artifact bundle (dialect smoke log + bench profile JSON) | `make build-mlir-ci-bundle` |
| `cpu-mlir-jit-contract.yml` | MLIR CI bundle manifest-only smoke (same bundle dir) | `make smoke-build-mlir-ci-bundle-manifest` |
| `cpu-mlir-jit-contract.yml` | Simulate downloaded MLIR bundle regression validate | `make validate-mlir-ci-metadata-download` |
| `cpu-mlir-ci-nightly.yml` | Loop-close docs sync gate | `make smoke-check-loop-close-docs` |
| `cpu-mlir-ci-nightly.yml` | MLIR CI artifact bundle (nightly) | `make build-mlir-ci-bundle` |
| `cpu-mlir-ci-nightly.yml` | MLIR CI bundle manifest-only smoke (same bundle dir) | `make smoke-build-mlir-ci-bundle-manifest` |
| `cpu-mlir-ci-nightly.yml` | Simulate downloaded MLIR bundle regression validate (nightly) | `make validate-mlir-ci-metadata-download` |
<!-- LOOP_CLOSE_CI_WORKFLOW_MAKE_TABLE_END -->

**Loop-close archive stage breakdown (VM reference, USE_MLIR=0):**

| Stage | quick archive | full archive |
|-------|---------------|--------------|
| `demos` | ~35–40s | ~35–40s |
| `mlir_bench_profile` | ~2–3s (`bench_quick=true`, `--quick` bench) | ~15–30s (`bench_quick=false`, `--full` bench) |
| `mlir_bench_contract` | ~5s | ~5s |
| `cert_e2e` | skipped | ~65–75s (value verify + walkthrough quick + mlir profile) |

Soft warning thresholds (metadata ``stage_timeout_warnings``) and hard CI ceilings are defined in
``scripts/cpu_cert_utils.py`` via ``loop_close_timing_contract()`` — single source for docs and nightly alerts.

Loop-close archive timing thresholds (Loop R136; single source: ``loop_close_timing_contract()``; regenerate with ``render_loop_close_timing_doc.py`` or ``make check-loop-close-docs``):

<!-- LOOP_CLOSE_TIMING_TABLE_BEGIN -->
| Mode | Stage | Soft limit (s) | Hard ceiling (s) |
|------|-------|----------------|------------------|
| quick | demos | 45 | 90 |
| quick | mlir_bench_profile | 8 | 60 |
| quick | mlir_bench_contract | 12 | 20 |
| quick | total | 50 | 220 |
| full | demos | 50 | 90 |
| full | mlir_bench_profile | 35 | 60 |
| full | mlir_bench_contract | 12 | 20 |
| full | cert_e2e | 70 | 150 |
| full | total | 130 | 220 |
<!-- LOOP_CLOSE_TIMING_TABLE_END -->

Nightly runs ``emit_loop_close_timeout_alert.py`` when soft limits are exceeded (Slack/issue placeholder JSON).

Nightly and PR CI upload a **metadata sidecar** (``loop_close_artifact_metadata_v2``) alongside the JSON archive.
Validate with ``validate_loop_close_archive.py --metadata-output PATH`` and
``validate_loop_close_archive_metadata.py META --archive ARCHIVE --check-stage-timeouts`` (full nightly).

Loop-close archive metadata sidecar fields (Loop R136; schema ``loop_close_artifact_metadata_v2``; single source: ``loop_close_metadata_doc_rows()``; regenerate with ``render_loop_close_metadata_doc.py`` or ``make check-loop-close-docs``):

<!-- LOOP_CLOSE_METADATA_FIELDS_BEGIN -->
| Field | Description |
|-------|-------------|
| bench_quick | Quick vs full archive mode for embedded MLIR bench profile |
| stage_elapsed_s | Per-stage elapsed seconds copied from archive profile |
| validation_ok | True when archive JSON passed validate_loop_close_archive |
| stage_timeout_warning_count | Count of soft stage timeout warnings |
| timeout_alert_pending | True when warnings exist and alert not yet emitted |
| timeout_alert_emitted | Set by emit_loop_close_timeout_alert --annotate-metadata |
| stage_timeout_warnings | Soft warnings when elapsed exceeds doc baseline below hard ceiling |
| archive_sha256 | SHA-256 digest of archive file when archive_path exists on disk |
<!-- LOOP_CLOSE_METADATA_FIELDS_END -->

| `test-cpu-mlir-dialect-smoke` | MLIR dialect_lowered bench emit unfused+fused (USE_MLIR=1) | ~30s |

Fused RMS ``native_gemm`` superoptimize uses ``apply_rms_matmul_search_tractability()``
from ``scripts/cpu_cert_utils.py`` (``YIRAGE_CPU_MAX_KN/TB`` + ``BENCH_MINIMAL_EXPLORE``);
without caps, search can explore 14M+ states and fail CI.

Search explore list gaps (e.g. ``kn_clamp_op`` explored but unsupported) are tracked in
``cpu_search_explore_not_supported()`` for future C++ search alignment.

KN/TB layout chunk symmetry (both supported + explored) is tracked via
``cpu_layout_explore_gap_table()``; guarded by
``tests/integration/test_cpu_search_explore_sync.py`` and ``test_cpu_tb_chunk_deferred.py``.

### CPU execution path (P0 production)

`cpu_call` prioritizes **host BLAS** primitives, then the **TB interpreter** for fused graphs.
MLIR LLVM JIT is **opt-in research** only.

| Priority | Pattern | Path |
|----------|---------|------|
| 1 | Plain `kn_matmul_op` | `cpu_matmul` → `torch.matmul` (MKL) |
| 2 | Unfused or fused semantic `rms_norm @ matmul` | `cpu_rms_matmul` (fp32 RMS + deferred scale + host BLAS) |
| 3 | Other fused `kn_customized_op` | TB interpreter (`_interpret_mugraph_on_cpu_impl`) |
| — | LLVM JIT (experimental) | `YIRAGE_CPU_MLIR_JIT=1` **and** `YIRAGE_CPU_MLIR_JIT_EXPERIMENTAL=1` |

RMS reductions use **fp32 accumulators**. Semantic fused `kn_customized_op` rms+matmul graphs
(bgraph: RMS accum / norm + matmul [+ div]) bypass the TB interpreter and call `cpu_rms_matmul`,
which applies **deferred row scaling** after BLAS GEMM (no full ``[M,K]`` normed buffer).

Disable with ``YIRAGE_CPU_RMS_MATMUL_FAST=0`` to force the TB interpreter for debugging.

### P2: native fused rms_matmul (OpenMP + cblas, opt-in)

C++ entry point ``yirage_cpu_rms_matmul_f32`` (``src/kernel/cpu/rms_matmul_kernel.cc``):
fp32 row RMS via OpenMP, ``cblas_sgemm`` (OpenBLAS when available), row scale.

Build (one-time apt: ``libopenblas-dev`` or ``libblas-dev``):

```bash
YIRAGE_BACKEND=cpu pip install -e . --no-build-isolation
```

Runtime (default **off** — PyTorch/MKL is usually faster on Linux):

```bash
export YIRAGE_CPU_RMS_MATMUL_NATIVE=1   # force native path
# export YIRAGE_CPU_RMS_MATMUL_NATIVE=auto  # enable when m*k*n >= YIRAGE_CPU_RMS_MATMUL_NATIVE_ELEMS (default 1M)
```

Cython: ``yirage.core.cpu_rms_matmul_f32``. Python: ``cpu_native.cpu_rms_matmul``.

### MLIR → LLVM JIT on CPU (experimental)

Design path: **µGraph → MLIR → LLVM → native JIT**. Not the default `cpu_call` path (see P0 table above).

| Component | Path |
|-----------|------|
| MLIR emit (rms+matmul) | `python/yirage/kernel/cpu_mlir_jit.py` |
| JIT prep pipeline | `yirage-cpu-jit-pipeline` (Linalg + one-shot bufferize + loops) |
| LLVM JIT | `mlir/lib/Execution/JITRunner.cpp`, `CPUJITKernel.cpp` |
| `cpu_call` hook | `try_rms_matmul_jit(..., require_experimental=True)` — skipped unless both env vars are set |

Build with MLIR and run JIT tests / benches:

```bash
YIRAGE_BACKEND=cpu USE_MLIR=1 pip install -e . --no-build-isolation
export YIRAGE_CPU_MLIR_JIT=1 YIRAGE_CPU_MLIR_JIT_EXPERIMENTAL=1
PYTHONPATH=. pytest tests/python/test_cpu_mlir_jit.py -v
```

Fused **bgraph** `grid_dim` / `forloop_range` can sink into ``scf.for`` + ``tensor.extract_slice``
when `grid_m > 1`. On compile/invoke failure, JIT callers fall back to the Python TB interpreter.

Benchmark with JIT columns:

```bash
PYTHONPATH=. python3 scripts/bench_fused_vs_mkl_baseline.py --workloads rms_norm_matmul --mlir-jit --json
```

(`--mlir-jit` sets both `YIRAGE_CPU_MLIR_JIT=1` and `YIRAGE_CPU_MLIR_JIT_EXPERIMENTAL=1`.)

JSON fields for MLIR JIT runs (`rms_norm_matmul` only):

| Field | Description |
|-------|-------------|
| `mugraph_source` | Runtime path: `mlir_jit`, `fused_search`, `interpreter_unfused`, `interpreter_fallback` |
| `mlir_jit_emit_path` | Winning MLIR emit label (`dialect_lowered`, `hand_bgrid_tiled`, `hand_tiled`, `hand_flat`, …) |
| `mlir_jit_fused_seed` | Set when `--mlir-jit-fused` superoptimized with fixed bgrid tiling |
| `mlir_jit_ms` / `interpreter_ms` | Latency on the cygraph used for JIT (fused graph when search ran) |
| `hand_mlir_jit_ms` / `dialect_lowered_jit_ms` | Hand vs dialect lowered JIT latency (R42/R63 contract) |
| `speedup_hand_over_dialect_lowered` | Ratio dialect/hand when both paths benchmarked |
| `mlir_hand_dialect_aligned` | Numerical alignment between hand and dialect JIT paths |

Documented combinations (see ``mlir_jit_bench_json_field_guide()`` and
``validate_mlir_jit_bench_row()`` in `scripts/bench_fused_vs_mkl_baseline.py`):

| Seed | CLI / env | Typical `mlir_jit_emit_path` |
|------|-----------|--------------------------------|
| P0 unfused (default quick) | `--mlir-jit` | `hand_tiled` or `hand_flat` |
| P0 unfused | `--mlir-jit` + `YIRAGE_CPU_MLIR_JIT_DIALECT=1` | `dialect_lowered` |
| Fused bgrid | `--mlir-jit --mlir-jit-fused` | `hand_bgrid_tiled` |
| Fused bgrid | above + `YIRAGE_CPU_MLIR_JIT_DIALECT=1` | `dialect_lowered` |

When JIT correctness passes, `mugraph_source` becomes `mlir_jit`; otherwise timing fields still
report `mlir_jit_emit_path` from the fused or seed cygraph while `mugraph_source` may stay
`fused_search` or `interpreter_unfused`.

## Related docs

- [INSTALLATION.md](INSTALLATION.md) — per-backend build and install
- [accelforge_quick_start.md](accelforge_quick_start.md) — AccelForge prescreen and RL coupling
- [README.md](../README.md) — project overview
