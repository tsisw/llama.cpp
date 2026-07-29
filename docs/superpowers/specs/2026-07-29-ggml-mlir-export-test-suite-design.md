# ggml → MLIR export test suite (macOS host build)

Date: 2026-07-29
Status: approved, not yet implemented

## Problem

`examples/mlir-export/` can export a ggml forward graph to linalg MLIR and, on a TSI box, compile
and run it. Nothing verifies the exporter itself. [exporter.h](../../../examples/mlir-export/exporter.h)
names two test programs, `mlir-export-matmul.cpp` and `mlir-export-matmul-add.cpp`; neither exists.
The only host-buildable targets today (`ref_check`, `recon_cpu_check`, `decode_cpu_check`) are
pure-CPU reconstructions and never invoke the MLIR path.

Goal: a unit test suite that takes a ggml graph all the way through export → compile → execute →
numeric comparison, runnable on a macOS host build against the TSI compiler at `~/repo/mlir-compiler`.

## Established facts

Verified on this machine (macOS 26.5, arm64, 2026-07-29), not assumed:

| Fact | Evidence |
|---|---|
| The compiler is built and importable | `~/repo/mlir-compiler/venv/bin/python` (3.11.15) imports `tsavorite`, `tsi_mlir`; torch 2.9.1, numpy 1.24.4 |
| `tsi-opt` runs | reports LLVM 22.0.0git |
| Raw linalg MLIR is a supported entry point | `RawGraphBackend(TXEBackend)` overriding `convert_to_linalg` with `Module.parse`, in `compile_graph_fpga.py:35` |
| A raw linalg module JITs and executes on macOS | probe: hand-written `func @forward` add, full pipeline (`linalg-transforms → tile → bufferize → vector → txe → blobs → host LLVM`), `TXERunner`, **max abs err 0.0** |
| Default target is FFM (host-native) | `TXECompilerConfig().txe_target == "FFM"` |
| **Ten / TXE-sim cannot run on macOS** | `for_ten` blob compile invokes Cadence `xt-clang` at `/proj/vendors/cadence/xtensa/…`; `/proj` does not exist, `xt-clang` absent. Fails at "Compiling blobs" after all MLIR stages succeed. |
| Exporter op coverage | `MUL_MAT, ADD, MUL, SCALE, RMS_NORM, SOFT_MAX, ROPE, PERMUTE, RESHAPE, CONT, GET_ROWS, CONCAT, UNARY(SILU)` (`exporter.h:1490-1534`) |

Consequence of the Ten finding: execution target is a **parameter**, defaulting to FFM. `--target ten`
is wired and skips with an explicit reason when `xt-clang` is unavailable, so the same suite is
meaningful on an SDK box without edits.

## Architecture

Two stages joined by a case directory on disk, plus ctest wiring.

```
[C++] mlir-export-cases              [Python] pytest runner
  build ggml_cgraph                    read case dir
  seeded deterministic inputs          RawGraphBackend → compile(jit)
  ggml CPU reference        ─cases/─▶  runtime_init / forward / runtime_finalize
  exporter.h → forward.mlir            compare vs expected_0.bin
  write inputs + expected              per-case pass/fail/xfail
```

The split mirrors a boundary the repo already has: C++ owns ggml graph construction and the CPU
reference (as `recon_cpu_check` does), Python owns the compiler driver (as `compile_graph_fpga.py`
does). Each stage runs standalone, which is what makes failures diagnosable:

```
mlir-export-cases --emit matmul /tmp/case      # stage 1 alone
pytest ... --case-dir /tmp/case                # stage 2 alone
```

### Case directory format

The stable interface between stages. Adding a case touches only C++.

| File | Contents |
|---|---|
| `forward.mlir` | one `func.func @forward(...) attributes {llvm.emit_c_interface}` |
| `input_<i>.bin` | raw f32, C order, one per runtime arg, in `%arg` order |
| `expected_0.bin` | raw f32 ggml CPU reference for the single output |
| `case.json` | `{name, arg_shapes, out_shape, dtype, rtol, atol, expect: "pass"\|"unsupported"\|"mismatch"}` |

`expect` drives the runner's assertion: `pass` → must match, `unsupported` → xfail with the
exporter's reason, `mismatch` → must **not** match (the vacuous-suite guard, below).

## Components

### `examples/mlir-export/mlir_export_cases.cpp` (new)

Links `ggml` only — not `llama`. This matches `recon_cpu_check` and avoids the documented TSI link
problem where linking `ggml` in a TSI build pulls in `libggml-tsavorite`.

- One `static case_fn` per case: allocates a `ggml_context`, builds the graph, fills leaf tensors
  from a fixed-seed PRNG, runs `ggml_graph_compute` for the reference, calls the exporter, writes
  the case dir.
- Fixed seed is required: the reference and any rerun must agree bit-for-bit.
- Subcommands: `--list`, `--emit <case> <dir>`, `--emit-all <dir>`.
- Catches `mlir_export_error` and writes `expect: "unsupported"` with the reason instead of dying,
  so an exporter gap surfaces as an xfail rather than a build break.

### `examples/mlir-export/tsi_raw_backend.py` (new, extracted)

`RawGraphBackend` currently lives inside `compile_graph_fpga.py`. The runner needs the same class.
Extract it to one module imported by both rather than duplicating it. `compile_graph_fpga.py` keeps
its CLI and behavior unchanged; only the class definition moves. This is the sole edit to existing
code, and it exists because the new work needs it — not as general cleanup.

### `examples/mlir-export/tests/test_mlir_export.py` (new)

pytest, parameterized over case dirs found under a root passed by ctest.

- `--target ffm` (default) → `TXECompilerConfig(log_mlir=True)`.
- `--target ten` → `TXECompilerConfig.for_ten(...)`; skips with reason if `xt-clang` is missing.
- Per case: parse `case.json`, load inputs, compile JIT, `runtime_init()` / `forward()` /
  `runtime_finalize()`, compare with `np.allclose(rtol, atol)`.
- On mismatch, report max abs and max rel error and the argmax index — a bare "not close" is not
  actionable.

### CMake wiring — `examples/mlir-export/CMakeLists.txt`

Already guarded by `if (NOT DEFINED GGML_TSAVORITE)` in `examples/CMakeLists.txt`, so this is
host-build only, which is what we want.

- `add_executable(mlir-export-cases mlir_export_cases.cpp)`, `target_link_libraries(... ggml)`.
- `MLIR_COMPILER_DIR` cache variable, default `$ENV{HOME}/repo/mlir-compiler`.
- Venv interpreter `${MLIR_COMPILER_DIR}/venv/bin/python`.
- If that interpreter exists: `add_test(NAME mlir-export-suite COMMAND <venv python> -m pytest ...)`.
  If not: register a test that reports skipped, and `message(STATUS ...)`. A plain host build must
  **not** start requiring the compiler repo to configure.

## Cases

| Case | Graph | Path exercised | Tolerance |
|---|---|---|---|
| `add` | `ADD(a,b)` | TVU elementwise | exact |
| `mul` | `MUL(a,b)` | TVU elementwise | exact |
| `scale` | `SCALE(a,k)` | TVU elementwise | exact |
| `silu` | `UNARY(SILU,a)` | unary | rtol 1e-5 |
| `rms_norm` | `RMS_NORM(a)` | reduction | rtol 1e-5 |
| `soft_max` | `SOFT_MAX(a)` | reduction | rtol 1e-5 |
| `matmul` | `MUL_MAT(a,b)`, K multiple of 32 | TMU | rtol 1e-5 |
| `matmul_add` | `ADD(MUL_MAT(a,b),c)` | composition | rtol 1e-5 |
| `add_negative` | `ADD(a,b)` with corrupted reference | the harness itself | must mismatch |

`matmul_add` is the composite for the initial suite. An attention-shaped subgraph
(`MUL_MAT → SCALE → SOFT_MAX → MUL_MAT`) was specified and then deferred by request; see Out of scope.

Tolerances are a starting point. fp32 reassociation in lowered reductions may require loosening.
Rule: report the observed per-case error and loosen deliberately with a recorded reason; never widen
silently to get green.

## Error handling

| Condition | Behavior |
|---|---|
| Exporter rejects the graph | case marked `unsupported`, test **xfails with the reason** — gaps stay visible |
| Compile stage fails | test fails naming the failing stage; `log_mlir` artifacts retained in the case dir |
| Numeric mismatch | fail with max abs err, max rel err, argmax index |
| mlir-compiler venv missing | suite skips, actionable message; configure still succeeds |
| `--target ten` without `xt-clang` | skip with reason |

## Guarding against a vacuous suite

One case (`add_negative`) is emitted with a deliberately corrupted `expected_0.bin` and
`expect: "mismatch"`; the runner asserts it **fails** comparison. Without this, a harness bug that
compares nothing is indistinguishable from a fully passing suite. This is the single most important
test in the suite.

## Out of scope

- `attn_small`, the attention-shaped composite (`MUL_MAT → SCALE → SOFT_MAX → MUL_MAT`) — deferred by
  request 2026-07-29. `matmul_add` remains as the composite covering op composition. The builder is
  straightforward to add later: q/k as `(D,T)`, v as `(T,D)`, `mul_mat(k,q) → scale(1/√D) →
  soft_max → mul_mat(v,·)`.
- `ROPE`, `GET_ROWS`, `CONCAT`, `PERMUTE`, `RESHAPE`, `CONT` cases — deferred; they need fiddly
  reference setup and are likely to surface exporter gaps that would turn this into an
  exporter-fixing project.
- f16 cases. The exporter is f32/i32 only (`mlir_element_type`).
- Whole-model / decode graphs. `wholegraph.sh` and `decode.sh` already cover that on a TSI box.
- Any change to the TSI backend or `tsi_wholegraph.cpp`.

## Risks

| Risk | Mitigation |
|---|---|
| `matmul` may not lower cleanly at small shapes on FFM. The FPGA config needed a `tmu_mma_shape` override (product must be 8 for f32, `compile_graph_fpga.py:71`); FFM may need similar. | Probe `matmul` first during implementation, before writing the other cases. If it needs a config override, record why in `case.json`. |
| Tolerances may be wrong in either direction | Report observed error per case; calibrate once, deliberately. |
| Ten path is unexercisable locally | Accepted. Wired, auto-skipped, verifiable only on an SDK box. Do not claim it works. |
| Suite runtime grows with case count | Probe measured a single small case at a few seconds. If total exceeds ~2 min, mark the composites as a separate ctest label. |
