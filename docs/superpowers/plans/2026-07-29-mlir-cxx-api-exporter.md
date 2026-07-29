# MLIR C++ API exporter implementation plan

> **For agentic workers:** execute inline, task by task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Replace the exporter's hand-built IR strings with in-memory MLIR construction via the C++
API, and restructure `examples/mlir-export/` into `include/` `src/` `tools/`.

**Architecture:** A static library `tsi-mlir-export` builds a verified `mlir::ModuleOp` and prints it.
Its public header carries no MLIR types, so consumers only link MLIR. One `exportGraph` entry point
covers single and multiple outputs.

**Tech Stack:** MLIR 22 from `~/repo/mlir-compiler/build/_deps/llvm-build`, clang-17, C++17 for the
library, CMake + ctest, pytest.

## Global Constraints

- MLIR comes from `MLIR_COMPILER_DIR`, never Homebrew llvm@17.
- Library compiles at C++17 to match LLVM. Tools stay C++20.
- Never `include(HandleLLVMOptions)`: it adds `-fno-exceptions` and breaks `throw mlir_export_error`.
- Enclosing CMake project must have C enabled (llama.cpp already does).
- Use `Op::create(builder, ...)`, not the deprecated `builder.create<Op>()`.
- Minimal link set: `MLIRIR MLIRParser MLIRFuncDialect MLIRLinalgDialect MLIRTensorDialect
  MLIRArithDialect MLIRMathDialect MLIRSupport`.
- No MLIR type in any file under `include/`.
- `find_package(MLIR CONFIG)` without `REQUIRED`; absent means skip, not fail.

---

### Task 1: Case builders for the 11 uncovered emitters

**Files:** Modify `examples/mlir-export/mlir_export_cases.cpp`

Add one `build_fn` plus one `CASES` entry each, still against the text emitter. Shapes must satisfy
the constraints already encoded in the emitters:

- `matmul_vec`: `ggml_mul_mat(a[32,32], b[32])`, b rank-1, hits `emit_mul_mat_2d_vec`.
- `matmul_3d`: a,b rank-3 with equal `ne[2]`, hits `emit_mul_mat_batched_3d`.
- `matmul_gqa`: `b->ne[2] % a->ne[2] == 0`, unequal, hits `_gqa` and `emit_repeat_heads_3d`.
- `permute`: rank-3 permute with a real data move, rank preserving, axes within rank.
- `permute_size1`: permute that only reshuffles size-1 dims, hits `emit_size1_reshape`.
- `reshape_split`: 2D to 3D where `ne[2]==x->ne[1]` and `ne[0]*ne[1]==x->ne[0]`.
- `reshape_merge`: 3D to 2D, the inverse.
- `concat`: `ggml_concat` along a dim inside the rank.
- `get_rows`: 2D F32 table, 1D I32 ids, `n_tokens > 1`.
- `get_rows_1tok`: `n_tokens == 1` so the rank-reducing branch runs.
- `rope`: `ggml_rope` on rank-3, plus a rank-2 variant for `emit_rope_rank2`.

`get_rows` needs I32 index data, so `fill_seeded` must not be used on it. Add a companion
`fill_seeded_i32(t, seed, hi)` producing ids in `[0, hi)`. The runner loads inputs as f32, so
`case.json` needs a per-arg `dtype` field; default `"f32"` keeps existing cases unchanged.

- [ ] Add `fill_seeded_i32`, a `dtype` field on `case_spec` args, and the 11 builders
- [ ] `cmake --build build --target mlir-export-cases && ./build/bin/mlir-export-cases --list`
- [ ] `./build/bin/mlir-export-cases --emit-all /tmp/cases-golden` and confirm every case emits
- [ ] Teach `conftest.py`/`test_mlir_export.py` the `dtype` field; run the suite, record which new
      cases pass end to end and which the pipeline rejects
- [ ] Commit

### Task 2: Golden IR capture

**Files:** Create `examples/mlir-export/tests/golden/<case>.mlir`, `tests/test_ir_golden.py`

- [ ] Copy each `forward.mlir` from `/tmp/cases-golden/<case>/` into `tests/golden/<case>.mlir`
- [ ] Write `test_ir_golden.py`: for each case, parse the golden and the freshly emitted IR with
      `tsi_mlir.ir.Module.parse`, print both, assert the strings are equal
- [ ] Run it against the current text emitter: must pass trivially (it is comparing to itself)
- [ ] Temporarily corrupt one golden, confirm the test FAILS, restore. A golden test never observed
      failing is not a test
- [ ] Register `mlir-export-golden` in CMake alongside `mlir-export-suite`
- [ ] Commit

### Task 3: Directory restructure, text emitter still intact

**Files:** many moves; modify `examples/mlir-export/CMakeLists.txt`,
`ggml/src/ggml-tsavorite/CMakeLists.txt`, `ggml/src/ggml-tsavorite/tsi_wholegraph.cpp`

Pure `git mv` plus include-path updates. No logic changes, so the suite must stay green throughout.

- [ ] `git mv` headers to `include/tsi/export/` and `include/tsi/graph/`, tools to `tools/<name>/`,
      python to `python/`
- [ ] Update every `#include` and the two CMakeLists include paths
- [ ] `./llama-cpp-build.sh` then run both test suites: still green
- [ ] Commit

### Task 4: CMake MLIR discovery and library skeleton

**Files:** Modify `examples/mlir-export/CMakeLists.txt`; create `src/export/Exporter.cpp`

- [ ] `find_package(MLIR CONFIG)` with the default `MLIR_DIR`, status message when absent
- [ ] `add_library(tsi-mlir-export STATIC ...)` at C++17, linking the 8 MLIR libs
- [ ] Seed `Exporter.cpp` with a `ModuleOp` that builds an empty `func @forward`, verifies, prints
- [ ] Confirm it compiles and links; confirm configure with `-DMLIR_DIR=/nonexistent` still succeeds
      and skips
- [ ] Commit

### Task 5: Builder core and the single entry point

**Files:** Create `include/tsi/export/Exporter.h`, `src/export/Builder.h`,
`src/export/Exporter.cpp`, `src/export/GraphWalk.cpp`

- [ ] Public header with `ExportOptions`, `exportGraph`, `discoverLeafs`, `mlir_export_error`
- [ ] `Builder.h` with the type helpers and constant/empty/fill helpers
- [ ] `Exporter.cpp`: context, dialect loading, func with `txe.name` arg/result attrs and
      `llvm.emit_c_interface`, constant leafs via `DenseElementsAttr` walking `t->nb[]` strides,
      `return`, `verify()`, print
- [ ] `GraphWalk.cpp`: `discoverLeafs` and the op switch, throwing `mlir_export_error` on unsupported
- [ ] Commit

### Task 6: Port elementwise and unary

**Files:** Create `src/export/OpsElementwise.cpp`

- [ ] Port `emit_elementwise_binop`, `add`, `mul`, `scale`, `silu`
- [ ] Golden test for add, mul, scale, silu passes; numeric suite for those passes
- [ ] Commit

### Task 7: Port norms

**Files:** Create `src/export/OpsNorm.cpp`

- [ ] Port `rms_norm` and `soft_max` including the optional scale/mask stage 0
- [ ] Golden and numeric tests for both pass
- [ ] Commit

### Task 8: Port matmul family

**Files:** Create `src/export/OpsMatmul.cpp`

- [ ] Port `mul_mat` dispatch, `2d`, `2d_vec`, `batched_3d`, `batched_3d_gqa`, `repeat_heads_3d`,
      `batched_3d_core`
- [ ] Golden and numeric tests for matmul, matmul_add, matmul_vec, matmul_3d, matmul_gqa pass
- [ ] Commit

### Task 9: Port shape ops

**Files:** Create `src/export/OpsShape.cpp`

- [ ] Port `size1_reshape`, `permute`, `reshape_like`, `concat`, `get_rows`
- [ ] Golden tests for permute, permute_size1, reshape_split, reshape_merge, concat, get_rows,
      get_rows_1tok pass
- [ ] Commit

### Task 10: Port rope

**Files:** Create `src/export/OpsRope.cpp`

- [ ] Port `rope`, `rope_rank2`, `rope_rank3`
- [ ] Golden tests for rope cases pass
- [ ] Commit

### Task 11: Delete the text emitter, full green

**Files:** Delete `examples/mlir-export/exporter.h`

- [ ] Delete `exporter.h`; confirm nothing references it
- [ ] Full build; both suites green; `add_negative` still fails-to-match as designed
- [ ] Confirm the fpga CMake warning text is present and accurate
- [ ] Commit

### Task 12: Docs and build script

**Files:** Modify `examples/mlir-export/README.md`, `~/repo/llama-cpp-build.sh`, `CLAUDE.md` (local)

- [ ] Document the new layout, `MLIR_DIR`, the golden workflow and how to regenerate goldens
- [ ] Add the golden suite to `llama-cpp-build.sh --test`
- [ ] Full `./llama-cpp-build.sh --clean --test` verification run
- [ ] Commit and push
