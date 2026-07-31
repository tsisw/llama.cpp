# Verifying the compiled path against native llama.cpp

Three runs. They prove different things, and only one of them produces numbers. All of them need a
fixed `--seed`, or every run decodes different tokens and nothing is comparable.

```sh
RT=~/repo/mlir-compiler/build/_deps/runtime-build/lib
M=~/models/smollm2-135m-f32.gguf
```

`DYLD_LIBRARY_PATH` (`LD_LIBRARY_PATH` on Linux), `TSI_RT_LIB_DIR` and `USER_DRAM_SIZE` are mandatory
for anything on the TSI path. Without them you get a segfault inside the allocator, or a link failure
when the driver builds `host.so`.

---

## 1. Native llama.cpp — the independent baseline

```sh
./build/bin/llama-cli -m $M -p "hello world" -n 8 -no-cnv --seed 42
```

No TSI environment at all. Every hook is a no-op, the KV cache is an ordinary CPU buffer, nothing is
exported or compiled. Keep the generated text; runs 2 and 3 are compared against it.

## 2. Compiled path with llama as the oracle — where the numbers come from

```sh
USER_DRAM_SIZE=16348 DYLD_LIBRARY_PATH=$RT TSI_RT_LIB_DIR=$RT \
TSI_MLIR_EXPORT=1 TSI_MLIR_VERIFY=1 TSI_MLIR_WEIGHT_ARGS=1 \
  ./build/bin/llama-cli -m $M -p "hello world" -n 8 -no-cnv --seed 42
```

Under `TSI_MLIR_VERIFY` llama **does** compute its own forward pass — it cannot be a reference
otherwise — and the driver writes no KV cache cells, so llama's cache stays entirely its own. Expect:

```
[tsi-mlir] compiled vs llama:       rel_sq_err=9.57154e-09 max_abs=0.00596428 argmax 30 vs 30 -> MATCH
[tsi-mlir] decode compiled vs llama: rel_sq_err=1.84377e-07 ... argmax 260 vs 260 -> MATCH
[tsi-mlir] decode compiled vs llama: ...                                              (one per token)
```

**Judge on `MATCH`, not on the error value.** Prefill's figure is stable and reproducible. Per-token
decode error is not: llama's CPU backend reduces in multi-threaded order and its own KV cache
accumulates that variation, so the same command with the same seed reports `rel_sq_err` around `1e-07`
on some runs and around `1e-02` on others — while the compiled side is bit-identical throughout. Five
runs measured with `TSI_MLIR_CACHE_SUM=1` had byte-identical fingerprints and logits, and still
disagreed with llama by five orders of magnitude between runs. The reference moves, not the result.

So the meaningful assertions here are **argmax agreement on every token** and a stable prefill figure.

## 3. Compiled path solo — the production configuration

```sh
USER_DRAM_SIZE=16348 DYLD_LIBRARY_PATH=$RT TSI_RT_LIB_DIR=$RT \
TSI_MLIR_EXPORT=1 TSI_MLIR_WEIGHT_ARGS=1 \
  ./build/bin/llama-cli -m $M -p "hello world" -n 8 -no-cnv --seed 42
```

Drop `TSI_MLIR_VERIFY` and llama runs **no forward pass at all**: it allocates and maintains the KV
cache and does the sampling, while the compiled graph computes everything and authors every K/V value
in that cache. Confirm with two lines:

```
[tsi-mlir] prefill wrote 2 cells x 30 layers into llama's cache
[tsi-mlir] llama's forward pass is skipped from here; the compiled graph drives generation and authors the cache
```

And confirm that neither of these appears:

```
[tsi-mlir] could not update llama's cache; falling back to llama for this token
[tsi-mlir] decode SKIPPED: ...
```

Either one means llama computed after all, and the run proves nothing about the compiled path.

**The verification is that run 3's text matches run 2's and run 1's.** Solo mode prints no error
figures, so text equality is the end-to-end check — and it is a strong one: llama's `SET_ROWS` is
inside the skipped graph, so an incorrect cache write degrades the output rather than reproducing it.

```sh
diff <(run_1_output) <(run_3_output)     # must be empty
```

One legitimate reason they can differ: our logits differ from llama's at ~1e-07, so a sampling
decision sitting on a near-tie can resolve the other way and the continuations diverge from that token
on. Before calling it a defect, check that run 2 reported `MATCH` on every token.

---

## Flags

| flag | what it does |
|---|---|
| `TSI_MLIR_EXPORT=1` | turns the driver on. Unset: every hook is a no-op and you get run 1 |
| `TSI_MLIR_VERIFY=1` | llama computes too, and each phase is diffed against it. Also suppresses the driver's cache writes so llama stays independent |
| `TSI_MLIR_WEIGHT_ARGS=1` | **required for 1B+ models.** TinyLlama's 4.2 GB constant pool cannot be emitted — `llc` fails with `cannot encode offset of relocations; object file too large`. Optional for small models, but much faster: the module drops from 513 MiB to 0.1 MiB |
| `TSI_MLIR_CACHE_SUM=1` | fingerprints the device KV cache and the logits after every decode step. Compare fingerprints across runs to check *our* determinism — this is what separates a real bug from llama's run-to-run variance |
| `TSI_MLIR_DIR=<dir>` | artifact and compile cache (default `./tsi-mlir`). A whole-model compile writes ~13 GB of intermediates per phase, so point it somewhere disposable |
| `TSI_MLIR_CPU_REF=1` | adds a CPU reconstruction diff. **First decode step only** — the cache it would need now lives on the device and nothing updates the host-side snapshot |
| `TSI_MLIR_DUMP_GRAPH=1` | writes each intercepted graph's nodes to `<dir>/graph-<phase>.txt` |
| `TSI_MLIR_SKIP=<n>` | graphs to skip before acting. Default 1, because llama's first graph is its warmup (BOS+EOS), not your prompt |
| `--seed <n>` | pins sampling. Mandatory for any comparison between runs |
| `USER_DRAM_SIZE=<MiB>` | simulated device DRAM. `16348` covers a 1.1B f32 model with weights passed as arguments |

## What is refused rather than approximated

These make the driver decline the graph and leave llama's own result in place, so a run that hits one
is still correct — just not exercising the compiled path:

- a quantized KV cache (`-ctk`/`-ctv q8_0`): a q8_0 block has no memref element type
- a transposed V cache (`-fa off`)
- `n_stream > 1` (multi-sequence contexts)
- multi-token decode batches, as used by server slots or speculative decoding

Quantized **weights** are refused earlier still: the exporter has no element type for a q8_0 block, so
a quantized GGUF never exports at all. Only f32/f16/bf16 models reach this path.
