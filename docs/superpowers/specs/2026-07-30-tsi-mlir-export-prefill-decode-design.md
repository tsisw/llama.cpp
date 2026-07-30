# TSI_MLIR_EXPORT: compiled prefill + decode from llama-cli

**Principle:** offload as much as possible to the compiler so it can optimize. Where work can live on the
host or in the graph, it goes in the graph.

Example: **SmolLM2-135M** — 30 layers, hidden 576, vocab 49152, 9 query / 3 KV heads, head_dim 64, FFN
1536, tied embeddings, cache capacity `L = n_ctx = 4096`. Replaces `TSI_WHOLEGRAPH=*`.

## What happens when you run the command

```sh
TSI_MLIR_EXPORT=1 llama-cli -m smollm2-135m-f32.gguf -p "hello world" -n 4
```

1. **llama starts normally** — loads the GGUF, allocates its own KV cache, builds and runs graphs as
   always. Our hook sits at `graph_compute()`, which sees each full forward graph before the backend
   splits it. llama's behaviour is unchanged; it stays the reference.
2. **We allocate our cache once** — two `tsi_alloc` buffers in device DRAM, 45 MiB each, zeroed:
   `cache_k` and `cache_v`, shaped `[30, 64, 3, 4096]` = `[layer, head_dim, n_head_kv, cell]`.
3. **llama's first graph is a warmup** (BOS+EOS, not your prompt). Skipped.
4. **The prefill graph arrives** — 2 tokens, positions 0 and 1. Pre-compute we snapshot the ids and
   weights; llama computes it and that becomes our reference; post-compute we rebuild an equivalent
   graph, lower to linalg, emit bytecode, compile, run, compare. Prefill **fills** cells 0-1, all layers.
5. **llama samples and calls decode** — 1 token, position 2. Same interception, but we also snapshot
   llama's KV cache *before* compute, when it holds exactly tokens 0-1: the correct decode input. Decode
   **reads** cells 0-1 and **appends** cell 2.
6. **The compiled logits become the result** — handed back to llama, which samples from them and
   continues. By default nothing is compared against anything: the point is to run the model through the
   MLIR path, not to check it.

Prefill fills the cache, decode consumes it. Together they are a generation loop in which the host passes
two pointers and reads logits — **no K/V ever crosses the host boundary.**

Comparison is layered, each level costing more than the last:

| flag | adds | cost |
|---|---|---|
| `TSI_MLIR_EXPORT=1` | export, compile, run; compiled logits are the output | the compiled forward |
| `TSI_MLIR_VERIFY=1` | diff compiled vs llama's own logits | reads llama's result |
| `TSI_MLIR_CPU_REF=1` | also compute the reconstruction on CPU, for a 3-way split that separates a reconstruction bug from a compilation bug | a full extra CPU forward pass |

The reconstructed graph is always *built* — that is where the MLIR comes from — but only *computed* under
`TSI_MLIR_CPU_REF`. Note that llama still runs its own per-op graph today regardless, because the hook
fires after compute; `TSI_MLIR_VERIFY` only decides whether we look at the result. Skipping llama's
compute entirely, so the compiled path is the only one that runs, is a follow-up.

## The two signatures

`L = n_ctx = 4096`, so one binary serves any prompt length; the mask decides which cells are live.

```mlir
// PREFILL — whole prompt at once, N = 2. From-scratch by definition, so slots are 0..N-1.
func.func @forward(
    %ids     : tensor<2xi32>               {txe.name = "input_ids"},
    %cache_k : memref<30x64x3x4096xf16, 1> {txe.name = "cache_k"},
    %cache_v : memref<30x64x3x4096xf16, 1> {txe.name = "cache_v"}
) -> (tensor<2x49152xf32> {txe.name = "res_0"})
   attributes {llvm.emit_c_interface}        // writes cells 0..N-1, in place

// DECODE — one token. pos/slot/mask come from llama, so cell ordering is irrelevant.
func.func @forward(
    %id      : tensor<1xi32>               {txe.name = "input_ids"},
    %pos     : tensor<1xi32>               {txe.name = "pos"},    // RoPE position
    %slot    : index                       {txe.name = "slot"},   // cell to write
    %mask    : tensor<4096xf32>            {txe.name = "mask"},   // cell validity
    %cache_k : memref<30x64x3x4096xf16, 1> {txe.name = "cache_k"},
    %cache_v : memref<30x64x3x4096xf16, 1> {txe.name = "cache_v"}
) -> (tensor<49152xf32> {txe.name = "res_0"})
   attributes {llvm.emit_c_interface}        // writes cell %slot, in place
```

Prefill **4 pointer args**, decode **7** — down from 397 as built today, via three reductions:

| | removes |
|---|---|
| weights are `dense_resource` constants in the IR; no flag to disable | 272 args |
| cache is a DRAM memref written in place, so argument-only, never a result | 60 results |
| one buffer per kind, not per layer (layers unroll, so the layer index is constant) | 58 args |

Cache rank is 4 because `GGML_MAX_DIMS` is 4, so a rank-5
`[layer, K/V, …]` buffer could not be a ggml tensor. Weights ship as bytecode, where blobs are raw binary
(269 MB f16) not hex (2×). Cache is f16 to match llama bit for bit, with **f32 accumulation in every
reduction** — an f16 sum over 2048 elements loses most of its significance.

**Reconstructed, not translated.** llama's graph uses `FLASH_ATTN_EXT` (one fused op per layer, nothing
equivalent in our lowering) and mutates its cache via `SET_ROWS`. So we rebuild the math with unfused
attention. Measured on SmolLM2 prefill: reconstruction vs llama `1.09e-07`, compiled vs reconstruction
`2.25e-12`, compiled vs llama `1.10e-07`. So `TSI_MLIR_VERIFY` should expect `~1.1e-07`, essentially all
of it unfused-vs-flash — compilation itself is near-exact. That is also why `TSI_MLIR_CPU_REF` is a
separate level: the 3-way split rarely locates anything the 2-way diff did not.

## Work required

Outside the exporter: `compile_graph_fpga.py:89` uses `read_text()` and must read bytes to carry bytecode;
`build_layer` must expose per-layer K/V so prefill can fill the cache (`build_decode_layer` already does);
both `host.so` define `@forward` at different arities, so load `RTLD_LOCAL` with separate handles; delete
`TSI_WG_BAKE_WEIGHTS` and `partitionWeights()`.

**Weights as constants is done, and a whole model compiles.** `runtime_args` is now the whole rule:
whatever the caller does not declare becomes a `dense_resource` constant, with no flag and no name
heuristic. On SmolLM2-135M f32 that turns 275 leafs into **3 args and 272 constants** - the 272 the table
above predicted - a 513.24 MiB bytecode module written in 1.93 s at 2.64 GiB peak RSS. That module then
**compiles to a working `host.so` in 140 s at 5.50 GiB peak RSS**, reporting
`_mlir_ciface_forward takes 4 pointer args`. Scaling is sublinear in memory: 8x the 64 MiB probe's payload
cost 3.9x the time and 1.6x the RSS.

The real limit is disk. The pipeline writes 3.46 GiB of `host.ll` and 1.08 GiB of `host.llvm.mlir` for a
513 MiB module, about 9x the weight bytes, because LLVM's textual IR spells out every constant. That ratio,
not memory, is why f16 weights matter for anything at 1B scale.

Two implementation rules, both learned by getting them wrong:

- **Every memref must carry memory space 1** (DRAM). That single omission accounts for every compile
  failure hit while validating the shapes above, including one inside a hand-written
  `bufferization.to_buffer`.
- **In `argv`, memrefs pass by pointer but scalars pass by value.** The generated shim declares every
  parameter `void *` and forwards `a[i]` straight into `_mlir_ciface_forward(ptr, i64, ptr)`, so an
  `index` argument reads the slot itself, not what it points at. Passing `&slot` makes the *address* the
  cell number, which faults far away from the cause (SIGBUS inside the compiled function).

**Cache persistence is now verified** — `dram-cache-persist` allocates one DRAM buffer, appends to two
different cells in two calls, and asserts the first cell survives the second call and untouched cells
stay zero. It was the assumption the whole design rested on.

**Bytecode input works** — `compile_graph_fpga.py` now reads bytes and detects the `ML\xefR` magic, so a
`.mlirbc` module compiles unchanged. A 64 MiB baked constant (`4096x4096xf32` matmul weight) compiled in
**35.6 s at 3.42 GiB peak RSS**. Extrapolating from a single point is unsafe, but the ratio is a warning:
SmolLM2 at f16 is ~4× that payload, and f32 ~8×. Measure at two or three sizes before attempting a full
model, and prefer f16 weights for the first attempt.

Still unverified: the op bodies themselves.

## Status

| piece | state |
|---|---|
| DRAM cache persistence, bytecode input | done, verified |
| `build_layer` exposes per-layer K/V | done |
| cache as a DRAM memref, read + append | done, compiles |
| weights as `dense_resource`, bytecode | done; whole model compiles in 140 s |
| f16/bf16 | done, via one `promote-ggml-to-f32` pass |
| `TSI_MLIR_EXPORT=1` driver | done for **prefill**; decode is classified and reported, not executed |
| decode execution | **not started** |

f16/bf16 turned out smaller than planned, not larger. Promoting the whole graph to f32 between import
and lowering meant no lowering pattern changed at all, and f32 accumulation came for free rather than
needing five separate accumulator widenings. The estimate that this was the largest piece was wrong.

### Decode: cache reading is done, graph reconstruction is not

`LiveCache.h` reads llama's cache and the driver self-checks it on every real decode graph. Measured on
SmolLM2-135M, so these are facts rather than plans:

| | |
|---|---|
| `cache_k_l0` | f16 `[192, 4096]`; `ne[0]` = head_dim x n_head_kv, cells are `ne[1]` |
| decode's `VIEW` | f16 `[64, 3, 256]`; `ne[2]` = `n_kv`, the live window |
| geometry | 30 layers, head_dim 64, n_head_kv 3, capacity 4096, window 256 |

Two questions this design left open are now answered:

- **The layout already matches.** The first L cells reinterpreted as `[head_dim, n_head_kv, L]` are
  exactly what `build_decode` consumes. No shuffling, no per-cell gather.
- **V is not transposed.** `attn_v_trans = !flash_attn`, flash attention is on, so `cache_v` has
  `cache_k`'s shape. The extractor refuses a transposed V (`-fa off`) rather than misreading it.

**The snapshot must be taken before compute.** `after_compute` runs once `SET_ROWS` has written the
current step's cell, so a decode graph built from that state reads its own answer. Positions are a graph
input and are available before compute, so the driver reads them there and snapshots the cache there.
Verified: `last written cell 1, pos 2 -> layout CONFIRMED`, i.e. cells `0..pos-1`.

Remaining, in order:

1. **Factor out the live-model probe.** Dimension, RoPE, eps and kq_scale discovery currently lives
   inside `build_cachefree_from_live`. Decode needs the same values, and two copies that can disagree
   between prefill and decode is exactly the drift to avoid, so extract it before duplicating it.
2. **`build_decode_from_live`.** Assemble a `DecodeModel` from the live graph plus `g_wcap` instead of
   from a GGUF (`bind_w` already does the weight materialization), build id/pos/mask, pass the extracted
   cK/cV, and export. Verify against llama's logits the way prefill is.
3. **Then the DRAM memref cache.** This collapses decode's ~62 host-cache arguments to the 7 above, and
   it also requires **prefill to fill the cache** rather than discard its K/V - which is why `build_layer`
   was taught to expose per-layer K/V early, and why that exposure is still inert.

Note that steps 1 and 2 produce a working compiled decode with the cache passed as host tensors. That is
not the design's signature, but it is the shortest path to a verified decode, and step 3 is a
signature change on a path that already computes the right answer.

## llama-server compatibility

`llama-cli` is single-user and single-sequence. `llama-server` is the concurrent one — `server_slot`,
`n_slots = n_parallel`, `n_ctx_slot = n_ctx / n_parallel`. **This design does not work there**, for two
reasons:

1. **Shape bucketing instead of two fixed graphs.** A server ubatch mixes tokens from several requests, so
   `n_tokens` varies per call and each value is a distinct graph. Compile a fixed set (1, 8, 64, 512), pad
   up to the nearest, and key compiled artifacts by shape rather than by phase.
2. **Per-token sequence routing.** One ubatch carries tokens for different slots at different positions, so
   `slot`/`pos`/`mask` must become per-token vectors rather than the scalars decode uses here, and the
   cache append becomes a scatter rather than one subview.

Carries over: weights-as-constants, the DRAM in-place cache, and the mask-as-input decision, which already
made the graph agnostic to cache layout — the prerequisite for multi-sequence. Breaks: the two-latch
capture is llama-cli-shaped, `L = n_ctx` conflicts with the per-slot context split, and the clean
prefill/decode dichotomy does not exist there at all.

Item 2 is a different reconstruction, not an extension of this one: this design builds a graph for a single
token stream, whereas a server graph is inherently ragged. Treat llama-cli as the deliberate first target —
it isolates the compiler work from the serving work.
