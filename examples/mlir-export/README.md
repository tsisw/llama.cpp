# Whole-graph TinyLlama on Tsavorite — build & run

Compile the **entire** ggml forward pass as one MLIR `func @forward` with the TSI mlir-compiler, run
it on tsisim/FPGA, and verify the next token against llama.cpp's own per-op output.

There are **no wrapper scripts**. The hook is compiled into `libllama`, so plain `llama-cli` drives
it and the flow is three steps: capture → compile → verify/run.

```sh
RT=~/repo/mlir-compiler/build/_deps/runtime-build/lib

# 1. capture: export the prefill graph as linalg MLIR
TSI_WHOLEGRAPH=capture TSI_WG_DIR=. \
  llama-cli -m tinyllama-f32.gguf -p "hello world" -n 1 -no-cnv     # -> ./forward.mlirbc

# 2. compile: MLIR -> host.so  (--target posix also covers host FFM, despite the file name)
USER_DRAM_SIZE=16348 TSI_RT_LIB_DIR=$RT \
  ~/repo/mlir-compiler/venv/bin/python compile_graph_fpga.py forward.mlirbc out_ffm

# 3. verify: run the compiled forward, diff its next-token argmax against llama's per-op result
USER_DRAM_SIZE=16348 DYLD_LIBRARY_PATH=$RT TSI_WHOLEGRAPH=verify TSI_WG_DIR=. \
TSI_WG_LIB=$PWD/out_ffm/host/host.so \
  llama-cli -m tinyllama-f32.gguf -p "hello world" -n 1 -no-cnv
```

| env var | meaning |
|---|---|
| `TSI_WHOLEGRAPH` | `capture` \| `dump` \| `verify` \| `run`; unset = the hook is a no-op |
| `TSI_WG_DIR` | where `forward.mlirbc` / `forward.manifest` are written and read |
| `TSI_WG_LIB` | compiled `host.so` to load (`verify`, `run`) |
| `TSI_WG_SKIP` | skip N graphs before capturing (skips llama's warmup graphs) |
| `TSI_WG_CTX_MB` | override the reconstruction context size (default: sized from weights seen) |
| `TSI_DUMP_GGML_IR` | also dump the ggml dialect before lowering to linalg |
| `USER_DRAM_SIZE` | simulated device DRAM budget; a 1.1B f32 model needs `16348` |
| `TSI_RT_LIB_DIR` | where `libTsavRTShimCAPI` lives, for linking `host.o` -> `host.so` |

Verified end to end on SmolLM2-135M f32 (30 layers, tied embeddings): 874 nodes, and

```
VERIFY recon-CPU vs per-op:   rel_sq_err=1.09e-07  argmax 504 vs 504 -> MATCH
VERIFY compiled vs recon-CPU: rel_sq_err=2.25e-12  argmax 504 vs 504 -> MATCH
VERIFY compiled vs per-op:    rel_sq_err=1.10e-07  argmax 504 vs 504 -> MATCH
```

Note that the graph llama runs first is its **warmup** graph (BOS+EOS), not your prompt. Use
`TSI_WG_SKIP` to reach the real prompt graph.

Capture is **prefill-from-scratch only**, and enforces it: the reconstruction rebuilds positions as
`0..n-1` and attends over the current tokens with no cache, so it checks the live graph's real
positions and refuses anything else rather than emitting valid-looking MLIR for a different function.
Skipping too far lands on a decode graph and reports:

```
capture SKIPPED: live graph is not a prefill-from-scratch: position[0] is 2, expected 0.
```

---

## Build (x86 build box)

```
SDK_VERSION=0.4.17 source tsi-pkg-build.sh triton all build-fpga package
# produces tsi-ggml-0.4.17.tz
```

## Deploy to tsisim

```
scp tsi-ggml-0.4.17.tz <tsisim>:/root/
# on tsisim:
tar -zxvf tsi-ggml-0.4.17.tz            # e.g. -> /root/tsi-ggml
```

tsisim ships with a bundled `tsi-ggml` symlink; repoint it at the package you just built:

```
cd /usr/bin/tsi/bin
ls -lrt                                 # tsi-ggml -> /opt/... (the bundled build)
rm -rf tsi-ggml
ln -s /root/tsi-ggml tsi-ggml           # source = any untarred path
```

## Run (on tsisim)

```
cd /root/tsi-ggml
# 1. activate the mlir-compiler venv first (compiler wheels used by the compile step) -
#    follow the standard mlir-compiler venv activation.
# 2. make the runtime libs visible:
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/bin/tsi/bin/tsi-ggml
# 3. copy the TXE blobs into the runtime's load path:
./ggml.sh
# 4. capture, compile, verify - same three commands as the host flow above, with
#    TXE_FPGA_CONFIG=txe_arm.json and --target fpga on the compile step.
```

Expected tail:

```
[tsi-wholegraph] VERIFY compiled vs per-op:  argmax <N> vs <N>  -> MATCH
```

### Notes

- **Activate the compiler venv before running** — the compile step (`compile_graph_fpga.py`) imports
  the mlir-compiler wheel packages.
- **`./ggml.sh` must run once after deploy** so the runtime finds the TXE blobs.
- The SDK/Xtensa paths are read from the environment (`MLIR_SDK_VERSION`, `XT_TOOLS_DIR`,
  `XT_SYSTEM_DIR`, `TSI_RT_LIB_DIR`, `TXE_FPGA_CONFIG`); `txe_arm.json` is the tsisim/arm config.
- Host-only CPU checks (no device): build `ref_check` + `recon_cpu_check` and compare argmax
  (`ref_check <gguf> "<prompt>"` prints the ids + reference token; `recon_cpu_check <gguf> <ids…>`
  prints the reconstruction's argmax). These cover `build_layer` in `ModelLayer.h`, which is what the
  capture path reconstructs with.

---

## KV-cache decode

The `llama-cli` hook covers **prefill only**: `LiveGraphBuilder` rebuilds a *cache-free* graph,
because llama's in-place KV cache cannot be expressed as a pure tensor function.

The decode graph is a separate, fixed-length graph — one MLIR func returning logits plus per-layer
`k_new`/`v_new`, reused for every token with the cache held on the host. Two tools share it, both
building it with `build_decode` from `DecodeModel.h`, so the checked graph and the compiled graph
cannot drift apart: `decode_cpu_check` emits it and checks it on CPU, `decode_run` executes the
compiled result.

```sh
cmake --build build --target decode_cpu_check decode_run
ids=$(./build/bin/ref_check smollm2-135m-f32.gguf "hello world" | sed -n 's/^ids: //p')

# CPU check + emit: decode step k (cache = tokens 0..k-1) must equal prefill(0..k) at the last position
./build/bin/decode_cpu_check smollm2-135m-f32.gguf $ids --L 6 --emit decode.mlir

# compile, exactly as for prefill
USER_DRAM_SIZE=16348 TSI_RT_LIB_DIR=$RT \
  ~/repo/mlir-compiler/venv/bin/python compile_graph_fpga.py decode.mlir out_decode

# run the compiled decode graph, diffing every step against a CPU prefill of the same prefix
USER_DRAM_SIZE=16348 DYLD_LIBRARY_PATH=$RT ./build/bin/decode_run smollm2-135m-f32.gguf \
    --lib out_decode/host/host.so --prompt "hello world" --L 6 --gen 2 --verify
```

`decode_cpu_check <gguf> <id0> [id1 …]` takes `--L` (cache cap, ≥ number of ids), `--emit <file>` and
`--dump-io <dir>` (every input/output of step 0 plus a `manifest.json`). `decode_run` takes
`--lib`/`--emit`, `--prompt` or raw ids, `--L`, `--gen` and `--verify`.

Verified on SmolLM2-135M f32: 336 args, 61 outputs, `_mlir_ciface_forward` 397 pointer args, and
`compiled-decode vs prefill: 3/3 MATCH` with rel_sq_err ≤ 1.2e-12.

---

## Notes on the host runtime

`tsi_initialize()` must be called before the first `tsi_alloc()`, and `tsi_finalize()` before exit.
On a TSI build the ggml-tsavorite backend does both during `llama_backend_init` / `ggml_backend_free`;
a plain host/FFM build has no such backend, so `WholeGraphHook.cpp` and `decode_run.cpp` do it
themselves. Skipping the init segfaults on the first allocation; skipping the finalize makes the
process **hang at exit**, idling long after it has printed its results — which looks like slow compute
but is not.

Host-only check (no FPGA): `decode_cpu_check <gguf> <ids…> --L <n>` runs the same fixed-L decode
against a CPU prefill and prints the per-step MATCH.

---

## How the exporter works: ggml dialect then linalg

The exporter is a two-stage MLIR pipeline built with the MLIR C++ API. There is no IR text
generation anywhere; MLIR builds and verifies the module, and only the final print produces text.

```
ggml_cgraph ──[import]──► `ggml` dialect ──[convert-ggml-to-linalg]──► linalg ──► forward.mlir
```

| Stage | Path | Role |
|---|---|---|
| dialect | `src/dialect/GgmlOps.td` | 13 ops mirroring ggml, plus ODS verifiers |
| import | `src/import/Importer.cpp` | 1:1, one dialect op per graph node, `op_params` as attributes |
| convert | `src/convert/Patterns*.cpp` | one file per op family, lowering to linalg |
| entry | `src/export/Exporter.cpp` | import, verify, lower, verify, print |

Dump the intermediate to see what was read from the graph, separately from how it was lowered:

```
TSI_DUMP_GGML_IR=1 ./build/bin/test-mlir-export-cases --emit silu /tmp/x
```
```mlir
%0 = ggml.silu %arg0 : tensor<128xf32> -> tensor<128xf32>
```

**Where constraints live.** ggml's *own* invariants are dialect verifiers: `mul_mat` reduction-dim
agreement and GQA divisibility, rope position count, reshape element-count preservation, concat dim
bounds, `get_rows` row count. Our *lowering's* limits are `notifyMatchFailure` in the pattern that
cannot proceed: ALiBi softmax, permute outside rank 2-3, an unhandled reshape pair, NEOX rope,
partial rotation, YaRN scaling. So the dialect accepts anything ggml can express, and only the
lowering admits what it cannot handle. Adding an op means one `.td` entry plus one pattern.

**Shape convention.** Dialect ops carry MLIR-ordered tensor types (ggml `ne` reversed), reversed
once in the importer. Attributes that *name* dims (`permute` axes, `concat` dim) stay in ggml dim
space, verbatim from `op_params`, because translating them is part of lowering.

**One entry point.** `exportGraph(gf, ExportOptions)` handles single and multiple outputs; an empty
`outputs` means the graph's last node. It returns a **complete module**, so callers must not wrap it
in `module { ... }`.

### Building it

Needs MLIR from the mlir-compiler checkout, never a system or Homebrew LLVM (it would be a different
version than the compiler consuming the output). `MLIR_DIR` is derived from `MLIR_COMPILER_DIR`
automatically; if MLIR is absent the library is skipped with a message and a plain host build still
succeeds. Two non-obvious build rules, both learned the hard way:

- The enclosing CMake project must enable **C**. `find_package(MLIR)` reaches `FindLibEdit`, which
  runs a C `check_include_file` and errors out in a CXX-only project.
- Never `include(HandleLLVMOptions)`. It adds `-fno-exceptions`, and the exporter throws.

`mlir_tablegen` also takes its `-I` flags from *directory-scope* `include_directories`, not target
properties. And `find_package(MLIR ...)` is called with `NO_DEFAULT_PATH`: without it, an
unresolvable `MLIR_DIR` is ignored and CMake falls back to its default search paths, which on a
machine with Homebrew LLVM silently builds against MLIR 17.

The fpga/aarch64 target **cannot** link this, since MLIR is host-only here. That is accepted; the
tsavorite CMakeLists emits a warning naming the reason rather than leaving a bare linker error.

## Export unit test suite (no FPGA, no model)

Verifies the exporter itself end to end on small graphs: ggml graph → linalg MLIR → TSI compiler →
JIT execute → compare against ggml's own CPU result. Runs on a plain host build (macOS included) in
under 10 s; needs only a built [mlir-compiler](https://github.com/tsisw/mlir-compiler) checkout, no
model file and no hardware.

```
ctest --test-dir build -L mlir-export --output-on-failure
```

Two tests, deliberately separate so a failure names its own cause:

| Test | What it proves | Needs | Time |
|---|---|---|---|
| `mlir-export-suite` | the compiled graph matches ggml numerically | TSI compiler venv | ~12 s |
| `mlir-export-lit` | each ggml op lowers to the expected linalg | `llvm-lit`, `FileCheck` | ~0.2 s |

`MLIR_COMPILER_DIR` defaults to `~/repo/mlir-compiler`; point it elsewhere with
`cmake -B build -DMLIR_COMPILER_DIR=<path>`. If its `venv/bin/python` is missing the suite reports
**skipped** and configure still succeeds — a host build never starts requiring the compiler repo.

23 cases, covering every one of the exporter's op lowerings: `add`, `mul`, `scale`, `silu`,
`rms_norm`, `soft_max`, `matmul`, `matmul_add`, `matmul_vec`, `matmul_3d`, `matmul_gqa`, `permute`,
`permute_size1`, `reshape_split`, `reshape_merge`, `concat`, `get_rows`, `get_rows_1tok`, `rope_2d`,
`rope_3d`, plus `matmul_const_w` / `get_rows_const_w` (baked-constant weights) and `add_negative`.

The seven pure data-movement cases are held to **bit-exact** equality (`rtol=atol=0`); they measure
0.0 error, so that is a real constraint rather than an accident. rope measures 1.5e-07/1.8e-07 max
abs and the matmul variants 2.4e-07 to 9.5e-07. Tolerances are measured, never guessed.

### Weights as constants

`ExportOptions::runtime_args` is the whole rule: those leafs become `%arg`s, and every other leaf the
exporter discovers is baked into the IR as a constant. There is no flag and no name heuristic - a
leaf is either a per-step input the caller declares, or it is a constant. Baking lets the compiler
see the weight values and fold, pre-tile and place them, and the compiled `host.so` then no longer
needs a matching weight buffer at run time.

Two things make that affordable:

- **`dense_resource`, not inline `dense<>`.** The data lives in a named blob outside the op. A
  contiguous, 4-byte-aligned tensor is referenced in place with no copy, so exporting does not need a
  second copy of the model in memory. A strided view is gathered through `nb[]` into a temporary.
- **`Format::Bytecode`.** Text prints a blob as hex, exactly 2 characters per byte. Bytecode keeps it
  raw, so the module is the weight bytes plus a few KB. Measured on the `test-bytecode-export`
  fixture: text 17223 bytes, bytecode 8889, for an 8192-byte weight.

The two `*_const_w` cases in the end-to-end suite check the values survive the round trip;
`bytecode-export-compile` checks the compiler accepts a bytecode module with a blob in it.

Measured on SmolLM2-135M f32: **275 leafs become 3 args and 272 baked constants**, a 513.24 MiB
bytecode module written in 1.93 s at 2.64 GiB peak RSS. That is the weight bytes plus a rounding
error, which is the point. **Compiling a module that size is not yet proven** - a 64 MiB constant took
35.6 s at 3.42 GiB peak RSS, and extrapolating 8x from one point is not a prediction. Prefer f16
weights and measure before assuming a whole model compiles.

### Per-op lowering tests (lit + FileCheck)

`tests/lit/` checks what a single ggml op lowers to, which the end-to-end suite cannot say without
running a whole graph through the TSI compiler. No Python, no compiler, milliseconds.

```
./build/bin/tsi-ggml-opt --convert-ggml-to-linalg tests/lit/matmul.mlir
```

Run the suite by hand:

```
TSI_LIT_TOOLS_DIR=$PWD/build/bin \
TSI_LIT_LLVM_TOOLS_DIR=~/repo/mlir-compiler/build/_deps/llvm-build/bin \
  ~/repo/mlir-compiler/build/_deps/llvm-build/bin/llvm-lit -sv examples/mlir-export/tests/lit
```

`errors.mlir` is the interesting one. It pins down *which layer* rejects what: a violated ggml
invariant must fail in the dialect verifier, and a limit of our lowering must fail as `failed to
legalize operation 'ggml.<op>'`. If those ever swap places, the separation between the two has broken.

### Two stages, each runnable alone

`test-mlir-export-cases` (C++, links ggml only) writes a self-contained case directory —
`forward.mlir`, `input_<i>.bin`, `expected_0.bin`, `case.json`. The pytest runner consumes it and
never touches ggml. So when a case misbehaves you can bisect the stages:

```
./build/bin/test-mlir-export-cases --list
./build/bin/test-mlir-export-cases --emit matmul /tmp/c          # stage 1: export + CPU reference
~/repo/mlir-compiler/venv/bin/python -m pytest \
    examples/mlir-export/tests/test_mlir_export.py \
    --cases-root /tmp --target ffm -v -k matmul             # stage 2: compile + run + compare
```

Compilation runs with `log_mlir=True`, so every lowering stage (`linalg.mlir`, `tile.mlir`,
`bufferize.mlir`, `vector.mlir`, `txe.mlir`, `host.mlir`) is left in the output dir for debugging.

### Adding a case

One `build_fn` plus one `CASES` entry in `tests/test-mlir-export-cases.cpp`. No Python change — the case
directory is the interface. Inputs are filled from a fixed `mt19937` seed and the reference is
computed with `ggml_graph_compute_with_ctx(ctx, gf, 1)`, so cases are bit-for-bit reproducible.

### Targets

`--target ffm` (default) is the host-native functional model. `--target ten` targets the TXE
simulator and **only works on an SDK box**: TXE blob compilation shells out to Cadence `xt-clang`
under `/proj/vendors/cadence/…`, so everywhere else all cases skip with that reason. It is wired but
unexercised off-box — do not read a green `ffm` run as evidence about `ten`.

### `add_negative`

Emits the `add` graph with element 0 of `expected_0.bin` offset by 1000.0, and asserts the
comparison **fails**. It exists because a harness bug that compared nothing would leave every other
case green and indistinguishable from a working suite. If `add_negative` ever fails, the comparison
logic is broken and the rest of the suite's green means nothing.
