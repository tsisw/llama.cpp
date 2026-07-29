# Whole-graph TinyLlama on Tsavorite — build & run

Compile the **entire** ggml forward pass as one MLIR `func @forward` with the TSI mlir-compiler, run
it on tsisim/FPGA, and verify the next token against llama.cpp's own per-op output.

The flow has three steps (capture → compile → verify/run), wrapped by `wholegraph.sh`:

```
./wholegraph.sh -m <model.gguf> -p "<prompt>" -n <ntokens> --mode verify
```

| flag | meaning |
|---|---|
| `-m` | model gguf path |
| `-p` | prompt |
| `-n` | number of tokens (used by `gen`) |
| `--mode` | `capture` \| `dump` \| `verify` \| `run` \| `gen` |

Modes: `capture` exports `forward.mlir`; `dump` writes `graph.txt`; `verify` runs the compiled
forward and diffs its next-token argmax against the per-op reference; `run` also samples the
compiled token; `gen` generates `-n` tokens, each from the compiled forward (prefill-only:
re-capture + recompile over the growing sequence per token — a correctness demo, not the fast path).

Example — generate 16 tokens on the compiled forward:
```
./wholegraph.sh -m /root/tinyllama-v0-f32.gguf -p "hello world" -n 16 --mode gen
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
# 4. run:
./wholegraph.sh -m /root/tinyllama-v0-f32.gguf -p "hello world" -n 1 --mode verify
```

Expected tail:

```
[tsi-wholegraph] VERIFY compiled vs per-op:  argmax <N> vs <N>  -> MATCH
```

### Notes

- **Activate the compiler venv before running** — the compile step (`compile_graph_fpga.py`) imports
  the mlir-compiler wheel packages.
- **`./ggml.sh` must run once after deploy** so the runtime finds the TXE blobs.
- `wholegraph.sh` bakes the SDK/Xtensa environment; override any path with the matching env var
  (`MLIR_SDK_VERSION`, `XT_TOOLS_DIR`, `XT_SYSTEM_DIR`, `TSI_RT_LIB_DIR`, `TXE_FPGA_CONFIG`, …) or the
  `--host tsisim|x86`, `--sdk`, `-c`, `-d` flags. `--host tsisim` (default) uses `txe_arm.json`.
- Host-only CPU checks (no FPGA): build `ref_check` + `recon_cpu_check` and compare argmax
  (`ref_check <gguf> "<prompt>"` prints the ids + reference token; `recon_cpu_check <gguf> <ids…>`
  prints the reconstruction's argmax).

---

## KV-cache decode (fast multi-token)

`wholegraph.sh --mode gen` recompiles per token (O(n²)). `decode.sh` is the fast path: it compiles
**one** fixed-length decode graph and reuses it for every token, with the KV cache held on the host
(each step's `k_new`/`v_new` are read back and appended). Design details are in the
`WHOLEGRAPH-TSI-FLOW.md` doc (§13).

`decode_run` + `decode.sh` are built and bundled into the `.tz` by the same package build, so on
tsisim there's no extra build step — same deploy as above (untar, repoint symlink, `./ggml.sh`),
then:

```
./decode.sh -m /root/tinyllama-v0-f32.gguf -p "hello world" --gen 16 --verify
```

| flag | meaning |
|---|---|
| `-m` | model gguf path |
| `-p` | prompt (tokenized by llama); or `--ids "id0 id1 …"` for raw token ids |
| `--gen` | number of tokens to generate |
| `--L` | cache cap (default `n_prompt + gen + 2`); must be ≥ prompt + gen |
| `--verify` | also run a CPU prefill each step and diff the argmax |

It prints the generated ids and the detokenized text; `--verify` adds a per-step `MATCH` line and a
`compiled-decode vs prefill: k/k MATCH` tail. Use the 1.1B gguf for coherent text — `tinyllama-v0` is
a toy model (the decode is numerically exact regardless).

Host-only check (no FPGA): `decode_cpu_check <gguf> <ids…> --L <n>` runs the same fixed-L decode
against a CPU prefill and prints the per-step MATCH.

---

## Export unit test suite (no FPGA, no model)

Verifies the exporter itself end to end on small graphs: ggml graph → linalg MLIR → TSI compiler →
JIT execute → compare against ggml's own CPU result. Runs on a plain host build (macOS included) in
under 10 s; needs only a built [mlir-compiler](https://github.com/tsisw/mlir-compiler) checkout, no
model file and no hardware.

```
ctest --test-dir build -R mlir-export-suite --output-on-failure
```

`MLIR_COMPILER_DIR` defaults to `~/repo/mlir-compiler`; point it elsewhere with
`cmake -B build -DMLIR_COMPILER_DIR=<path>`. If its `venv/bin/python` is missing the suite reports
**skipped** and configure still succeeds — a host build never starts requiring the compiler repo.

Cases: `add`, `mul`, `scale`, `silu`, `rms_norm`, `soft_max`, `matmul`, `matmul_add`, plus
`add_negative`.

### Two stages, each runnable alone

`mlir-export-cases` (C++, links ggml only) writes a self-contained case directory —
`forward.mlir`, `input_<i>.bin`, `expected_0.bin`, `case.json`. The pytest runner consumes it and
never touches ggml. So when a case misbehaves you can bisect the stages:

```
./build/bin/mlir-export-cases --list
./build/bin/mlir-export-cases --emit matmul /tmp/c          # stage 1: export + CPU reference
~/repo/mlir-compiler/venv/bin/python -m pytest \
    examples/mlir-export/tests/test_mlir_export.py \
    --cases-root /tmp --target ffm -v -k matmul             # stage 2: compile + run + compare
```

Compilation runs with `log_mlir=True`, so every lowering stage (`linalg.mlir`, `tile.mlir`,
`bufferize.mlir`, `vector.mlir`, `txe.mlir`, `host.mlir`) is left in the output dir for debugging.

### Adding a case

One `build_fn` plus one `CASES` entry in `mlir_export_cases.cpp`. No Python change — the case
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
