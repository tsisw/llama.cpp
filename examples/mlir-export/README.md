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
| `-n` | number of tokens |
| `--mode` | `capture` \| `dump` \| `verify` \| `run` |

Modes: `capture` exports `forward.mlir`; `dump` writes `graph.txt`; `verify` runs the compiled
forward and diffs its next-token argmax against the per-op reference; `run` also samples the
compiled token.

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
