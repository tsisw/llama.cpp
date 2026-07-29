# ggml to MLIR export via the MLIR C++ API: design

Replace the exporter's hand-built IR text with in-memory MLIR construction through the C++ API,
and give `examples/mlir-export/` a conventional `include/` `src/` `tools/` layout.

## Goal

`examples/mlir-export/exporter.h` is 1694 lines that build linalg MLIR by concatenating strings into
an `ostringstream`. Types, affine maps, iterator-type lists and SSA names are all formatted by hand.
Nothing validates the result until the TSI compiler parses it in Python, several stages downstream,
where a malformed affine map surfaces as an opaque parse error with no connection to the emitter that
produced it.

Replacing this with `mlir::OpBuilder` moves validation to the point of construction: types become
`RankedTensorType`, affine maps become `AffineMap`, SSA values become `mlir::Value`, and
`mlir::verify()` runs before anything is written out.

## Established facts

Every row was measured on this machine, not assumed.

| Fact | Value | How established |
|---|---|---|
| mlir-compiler LLVM version | 22.0.0git | `third-party/llvm-project-private/cmake/Modules/LLVMVersion.cmake` |
| Built with | `/opt/homebrew/opt/llvm@17/bin/clang++`, C++17 | `build/_deps/llvm-build/CMakeCache.txt` |
| `LLVM_ENABLE_RTTI` / `LLVM_ENABLE_EH` | ON / OFF | same cache |
| Library form | static, 412 `libMLIR*.a` | `ls build/_deps/llvm-build/lib` |
| `MLIRConfig.cmake` | `${MLIR_COMPILER_DIR}/build/_deps/llvm-build/lib/cmake/mlir` | `find` |
| clang-17 compiles + links MLIR 22 | yes, exit 0 | probe built and ran |
| Minimal link set | `MLIRIR MLIRParser MLIRFuncDialect MLIRLinalgDialect MLIRTensorDialect MLIRArithDialect MLIRMathDialect MLIRSupport` | probe linked with these 8, not all 412 |
| Compiles under pinned `MacOSX15.4.sdk` | yes | probe rebuilt with `CMAKE_OSX_SYSROOT` pinned |
| Exceptions usable despite `LLVM_ENABLE_EH=OFF` | yes | probe throws and catches, as long as `HandleLLVMOptions` is NOT included |
| `find_package(MLIR)` needs C enabled | yes | `FindLibEdit` runs a C `check_include_file`; fails in a CXX-only project |
| `OpBuilder::create<Op>()` | deprecated in LLVM 22 | build warnings: "Use OpTy::create instead" |
| Printer hoists affine maps to `#map` aliases | yes | probe output; aliases resolve at parse, so the pipeline sees identical IR |
| Static binary size | 27 MB | probe |
| Duplicate SSA names inside `linalg.generic` regions | legal, region-scoped | parsed a module with two bodies both naming `%sq` |
| Consumers of `exporter.h` | 5 | `mlir_export_cases.cpp`, `live_graph_builder.h`, `decode_run.cpp`, `decode_cpu_check.cpp`, `ggml/src/ggml-tsavorite/tsi_wholegraph.cpp` |
| Emitters to port | 22 | `grep 'std::string emit_'` |
| Emitters with automated coverage today | 8 of 22 | suite covers add, mul, scale, silu, rms_norm, soft_max, matmul, matmul_add |

## Decisions

| Decision | Rationale |
|---|---|
| Remove text emission entirely; no dual path | User directive. A second implementation would drift, as `build_func_text_baked` and its `_multi` variant already have |
| `forward.mlir` still written, by printing a verified `ModuleOp` | `RawGraphBackend` and `compile_graph_fpga.py` parse that text. The file is the process boundary; only the *hand-built strings* go away |
| MLIR from mlir-compiler, not Homebrew llvm@17 | User directive, and it keeps the exporter ABI-consistent with the compiler that consumes its output |
| MLIR types stay out of `include/` | Consumers compile without MLIR headers and only link MLIR. Keeps `tsi_wholegraph.cpp` and `decode_run.cpp` as plain C++20 and keeps compile times sane |
| One `exportGraph` entry point | User directive. Single output is the N=1 case of multiple outputs |
| No cross-compilation support | User directive. aarch64 MLIR libs do not exist here |
| fpga build fails loudly with an explanatory CMake warning | User accepted broken paths. Silently dropping documented on-board capture is worse than a clear error |
| Golden IR diff before deleting the text emitter | 8 of 22 emitters have coverage. Without goldens, "tests pass" would not justify deletion |

## Amendment: a `ggml` MLIR dialect

Direction changed after the elementwise family landed. Instead of building linalg directly from the
ggml graph, the exporter now runs in two stages:

```
ggml_cgraph -> [import] -> `ggml` dialect module -> [convert-ggml-to-linalg] -> linalg -> print
```

I argued against this on the grounds that ggml graphs arrive fully specialized (static shapes, no
control flow, one block, already sorted) so there is little for a source-level dialect to enable,
and that TableGen is a real cost. That was overruled; this records what the dialect buys and what it
costs, and the design proceeds with it.

What it genuinely improves:

| Gain | Detail |
|---|---|
| Per-op lowering tests | `tsi-ggml-opt --convert-ggml-to-linalg %s \| FileCheck %s`, the native idiom, instead of end-to-end pytest goldens |
| Better home for constraints | ggml's own invariants (mul_mat operand compatibility, concat dim range) become ODS verifiers. Our *lowering's* limits (rank <= 3, `mode == NORMAL`, `freq_scale == 1`) become pattern match failures. Today both are the same scattered `if (...) unsupported(...)` |
| Separable failure modes | The intermediate is dumpable, so "did we read the graph right" and "did we lower it right" stop being one question |
| Shape normalization as passes | GQA head repeat, size-1 permute collapse and rank dispatch can become canonicalizations rather than branches inside emitters |

Costs accepted: `mlir-tblgen` becomes a build-time dependency of llama.cpp, and the shape math
(ne reversal, GQA grouping, rope pair interleaving, reassociation indices) is relocated rather than
reduced.

### Additional established facts

| Fact | Value | How established |
|---|---|---|
| `mlir-tblgen`, `FileCheck`, `llvm-lit`, `mlir-opt`, `count`, `not` | all present in mlir-compiler's LLVM build | direct check |
| `MLIROptLib`, `MLIRPass`, `MLIRTransforms`, `MLIRTransformUtils` | present | direct check |
| TableGen from a downstream CMake project | works | probe generated all four `.inc` files and linked |
| `include(TableGen)`/`include(AddLLVM)`/`include(AddMLIR)` | do NOT break exceptions | probe throws and catches after including all three |
| `mlir_tablegen` include paths | needs **directory-scope** `include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})`; target-scope is not enough | probe failed with "could not find include file 'mlir/IR/OpBase.td'" until moved |
| Dialect header needs | `Bytecode/BytecodeOpInterface.h`, `IR/BuiltinAttributes.h`, `IR/BuiltinTypes.h`, `IR/Dialect.h`, `IR/OpDefinition.h`, `IR/OpImplementation.h`, `Interfaces/SideEffectInterfaces.h` | probe, one error at a time |
| Dialect implementation needs | `IR/Builders.h` | probe |

### Dialect scope

Thirteen ops mirroring the op switch: `add`, `mul`, `scale`, `silu`, `rms_norm`, `soft_max`,
`mul_mat`, `rope`, `permute`, `reshape`, `cont`, `get_rows`, `concat`. `reshape` and `cont` stay
distinct so the import is faithful to ggml even though both lower through one pattern.

**Shape convention.** Dialect ops carry MLIR-ordered tensor types (ggml `ne` reversed), reversed once
at import, so a single convention holds across the whole pipeline and the existing goldens stay
valid. Attributes that *name* dimensions (`permute` axes, `concat` dim) stay in ggml dim space,
verbatim from `op_params`; translating them to MLIR dims is part of the lowering, which is where
that logic belongs anyway.

The dialect lives entirely under `src/dialect/`, not `include/`, so the public header stays
MLIR-free.

### Revised file layout

| Path | Contents |
|---|---|
| `src/dialect/` | `GgmlOps.td`, `GgmlDialect.h`, `GgmlDialect.cpp` |
| `src/import/` | `Importer.cpp`: ggml_cgraph -> ggml dialect |
| `src/convert/` | `GgmlToLinalg.cpp` (the pass) plus `Patterns*.cpp`, one per op family |
| `src/Exporter.cpp` | orchestrates import, convert, verify, print |
| `tools/tsi-ggml-opt/` | `mlir-opt`-style driver registering the dialect and pass, for lit tests |
| `tests/lit/` | per-op FileCheck tests |

## Architecture

### Public API

`include/tsi/export/Exporter.h`, free of MLIR types:

```cpp
namespace tsi::mlir_export {

struct mlir_export_error : std::runtime_error { using std::runtime_error::runtime_error; };

struct ExportOptions {
    std::string                      func_name = "forward";
    std::vector<const ggml_tensor *> runtime_args;   // -> %argN {txe.name = "input_N"}
    std::vector<const ggml_tensor *> const_leafs;    // -> arith.constant dense<...>
    std::vector<const ggml_tensor *> outputs;        // empty = infer the single graph output
};

// Builds a module, verifies it, returns its printed text. Throws mlir_export_error.
std::string exportGraph(ggml_cgraph * gf, const ExportOptions & opts);

std::vector<const ggml_tensor *> discoverLeafs(ggml_cgraph * gf);
}
```

`exportGraph` replaces both `build_func_text_baked` and `build_func_text_baked_multi`. Callers that
want the single graph output leave `outputs` empty.

### Internal builder

`src/export/Builder.h` is private to the library and is the only place MLIR types appear in a header:

```cpp
struct GraphBuilder {
    mlir::OpBuilder &                            b;
    mlir::Location                               loc;
    std::map<const ggml_tensor *, mlir::Value> & values;

    mlir::RankedTensorType tensorType(const ggml_tensor * t) const;
    mlir::RankedTensorType tensorTypeRanked(const ggml_tensor * t, int rank) const;
    mlir::RankedTensorType transposedType(const ggml_tensor * t) const;
    mlir::RankedTensorType reducedType(const ggml_tensor * t) const;
    mlir::Value            zeroF32();          // the shared %cst
    mlir::Value            constantF32(float);
    mlir::Value            emptyLike(mlir::RankedTensorType);
    mlir::Value            filledZero(mlir::RankedTensorType);
};
```

`ensure_cst`, `new_id` and every `mlir_*`/`affine_map_*`/`iterator_types_*` string helper are deleted.
Ops are built with `Op::create(builder, ...)`, not the deprecated `builder.create<Op>()`.

### File layout

| Path | Contents |
|---|---|
| `include/tsi/export/Exporter.h` | public API above |
| `include/tsi/graph/` | `LiveGraphBuilder.h`, `ModelLayer.h`, `DecodeModel.h`, `DecodeLayer.h` |
| `src/export/Builder.h` | internal, MLIR-visible |
| `src/export/Exporter.cpp` | module and func assembly, constant leafs, dispatch, print |
| `src/export/GraphWalk.cpp` | `discoverLeafs`, node dispatch |
| `src/export/OpsElementwise.cpp` | add, mul, scale, silu |
| `src/export/OpsMatmul.cpp` | mul_mat 2d, 2d_vec, batched_3d, batched_3d_gqa, repeat_heads_3d, batched_3d_core |
| `src/export/OpsNorm.cpp` | rms_norm, soft_max |
| `src/export/OpsRope.cpp` | rope, rope_rank2, rope_rank3 |
| `src/export/OpsShape.cpp` | permute, reshape_like, size1_reshape, concat, get_rows |
| `tools/<name>/` | one subdir per executable |
| `python/` | `compile_graph_fpga.py`, `tsi_raw_backend.py` |
| `tests/` | `conftest.py`, `test_mlir_export.py`, `test_ir_golden.py`, `golden/` |

Static library `tsi-mlir-export` at C++17 to match LLVM. Tools stay C++20; they include only the
MLIR-free public header. `ref-check` and `recon-cpu-check` never touch the exporter and keep building
with no MLIR at all.

### CMake

```
MLIR_DIR defaults to ${MLIR_COMPILER_DIR}/build/_deps/llvm-build/lib/cmake/mlir
find_package(MLIR CONFIG)          # not REQUIRED
  found     -> build tsi-mlir-export and its dependent tools
  not found -> status message, skip them, host build still succeeds
```

Two hard rules from the probe: the enclosing project must enable C, and `HandleLLVMOptions` must not
be included because it would add `-fno-exceptions` and break `throw mlir_export_error`.

`MLIR_COMPILER_DIR/build` is a build tree, not an install tree. An install prefix is also accepted;
neither present is a clear message, not a confusing failure.

## Verification

Passing the current suite proves 8 of 22 emitters. So, in order:

1. Add case builders for the 11 uncovered emitters: `mul_mat_2d_vec`, `batched_3d`, `batched_3d_gqa`,
   `repeat_heads_3d`, `rope_rank2`, `rope_rank3`, `size1_reshape`, `permute`, `reshape_like`,
   `concat`, `get_rows`.
2. Emit IR for every case from the **current text emitter** and commit it under `tests/golden/`.
3. Port all 22 emitters.
4. `test_ir_golden.py` normalizes both sides through `Module.parse` then print, so affine-map
   aliases, SSA numbering and whitespace cancel, and asserts equality.
5. Delete the text emitter.

New cases are also attempted end to end. Any the TSI pipeline rejects are marked golden-only with the
reason recorded in `case.json`, not quietly dropped.

`add_negative` keeps its role: it must fail to match, proving the numeric comparison is not vacuous.

## Failure states

| # | Failure | Handling |
|---|---|---|
| 1 | fpga/aarch64 link fails, no aarch64 MLIR libs | Accepted. CMake warning explains it rather than a bare linker error |
| 2 | A new case does not survive the TSI pipeline | Golden-only, reason in `case.json` |
| 3 | Golden text drifts on an LLVM bump | Goldens pinned to this LLVM; regeneration documented |
| 4 | `find_package(MLIR)` in a CXX-only project | C must be enabled. Verified |
| 5 | `HandleLLVMOptions` disables exceptions | Not included. Verified by probe |
| 6 | mlir-compiler `build/` cleaned, `MLIR_DIR` stale | Clear configure-time message |
| 7 | 27 MB binaries, slower links | Accepted for host tooling |
| 8 | `verify()` fails on a ported emitter | Throws `mlir_export_error` naming the op, which is strictly better than today's silent bad text |

## Out of scope

- Running MLIR passes (canonicalize, CSE) before handing off. Would change what the TSI pipeline
  sees; no reason to couple that to this change.
- Emitting bytecode instead of text.
- Passing the in-memory module to Python without a text round trip.
- aarch64 or any cross build.
- `attn_small`, still deferred from the previous effort.
