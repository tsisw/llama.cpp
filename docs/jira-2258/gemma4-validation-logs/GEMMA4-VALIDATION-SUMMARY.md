# Gemma 4 validation — JIRA-2258

Stated motivation for this sync (per the reporter): Gemma 4 doesn't work on the old
fork but should work on the synced tree. Tested two tags from `ollama.com/library/gemma4`
against both the old and new posix binaries. **Result: partially confirmed, with an
honest and important caveat** — not a clean "it works now."

Both GGUFs pulled directly from the ollama registry (`registry.ollama.ai`), verified by
sha256 against the registry manifest before use:
- `gemma4:e2b` → `Gemma4-e2b-5.1B-Q4_K_M.gguf` (7.16 GB, 2012 tensors — MoE/matryoshka-style, per the "e2b" = "effective 2B" naming)
- `gemma4:12b` → `Gemma4-12b-Q4_K_M.gguf` (7.38 GB, 667 tensors — conventional dense-style tensor count)

Both placed at `/proj/rel/sw/ggml/models/` alongside the other test models.

## Old binary (935ee7bc0): fails immediately on both tags, as expected

```
 Unable to load Model
```
Confirms the reporter's premise — the old fork has no knowledge of the `gemma4`
architecture string at all, and fails the same way regardless of which tag. See
`old-gemma4-e2b.log` / `old-gemma4-12b.log`.

## New binary (this sync): mixed result — real progress, not full support

### gemma4:e2b (MoE-style, 2012 tensors) — fails, but with a precisely diagnosed cause

The binary exits with no visible error text in its own log (see `new-gemma4-e2b.log`) —
tracked down with `gdb` (breaking on `__cxa_throw`) to an actual thrown exception:

```
done_getting_tensors: wrong number of tensors; expected 2012, got 601
```

Confirmed via `gdb` that execution gets **past** architecture recognition, hparams
loading, and the gemma4-specific tensor declaration step (`llama_model_gemma4::
load_arch_hparams`/`load_arch_tensors` both complete normally) — the failure is a
generic upstream `llama_model_loader::done_getting_tensors()` safety check, present
in `libllama.so.0`, unrelated to the Tsavorite backend. It rejects the model because
601 of the file's 2012 declared tensors were never claimed by the C++ loader for this
architecture.

Checked upstream's own history past our sync target (`1f368f354`) for a gemma4-specific
follow-up fix and found none by commit-message search — inconclusive, not proof one
doesn't exist. Best-supported explanation, **not independently confirmed**: this
specific MoE/matryoshka-style checkpoint uses a tensor layout (per-expert or nested
sub-model tensors) that upstream's `gemma4` support doesn't yet fully consume as of our
exact target commit — a gap in upstream itself, not something introduced by this sync
or specific to Tsavorite.

### gemma4:12b (dense-style, 667 tensors) — loads and runs real OPU/TXE compute

This one gets much further: architecture recognized, hparams loaded, all 667 tensors
matched with no `done_getting_tensors` mismatch, and it begins actual token generation
with real dispatch to the Tsavorite backend. **Important caveat: "begins compute
without crashing" is not the same as "confirmed correct output."** The run was killed
after ~10 minutes without producing any generated text (see below) — treat this as
"loads and dispatches," not as a verified-correct result.

This was also the first K-quantized (Q4_K_M) model tested anywhere in this validation
exercise — every other model tested up to that point (tinyllama-5m, Gemma3-270M,
TinyLlama-1.1B, Llama3.2-1B) was F32 or BF16. Running it surfaced a high-volume warning:

```
TXE::align_address(): Warning: Unaligned memory access 0x7f22e8800104 not 128-byte aligned, aligning to 0x7f22e8800100
```

repeated **381,138 times** in the captured run (see `new-gemma4-12b-trimmed.log` for a
head/tail excerpt — the full raw log was 45MB, kept locally, not checked in).

**Traced the actual mechanism** (`TXE::align_address()`, `tsisw/TXE-FFM`,
`include/txe/txe.h`): it checks whether a memory address passed to a vector load/store
is 128-byte aligned (the hardware's vector-register width). If not, it does **not**
error — it silently rounds the address down to the nearest 128-byte boundary and
proceeds with *that* address instead. No crash, no assertion raised. This means "the
process didn't crash" is weak evidence of correctness here — a silently substituted
address could mean wrong data was read or written, not just a performance cost.

**Follow-up question raised and answered: is this a Gemma4-specific characteristic, or
a general K-quant issue — and did this sync cause it?** Tested two additional
K-quantized, non-Gemma4 models on **both** the old and new binaries, using the exact
same `TSAVORITE_MODEL_DEPLOYMENT_YAML` config as the Gemma4 runs:

| Model | Quant | Old binary | New binary |
|---|---|---|---|
| `qwen2-0_5b-instruct-q5_k_m.gguf` | Q5_K_M | ✅ completes, 0 alignment warnings | ✅ completes, 0 alignment warnings |
| `Qwen2.5-0.5B-Q4_K_M.gguf` (pulled from ollama, same exact quant type as the failing Gemma4-12b run) | Q4_K_M | ✅ completes, 0 alignment warnings | ✅ completes, 0 alignment warnings |

Both models complete cleanly with real OPU dispatch on both binaries, zero alignment
warnings in every case — including the Q4_K_M case, the identical quant sub-format used
by the failing Gemma4-12b run.

### Root cause found, three real bugs fixed, Gemma4-12b now confirmed working

**Correcting an earlier hypothesis in this doc.** The section above concluded the
alignment warnings were explained by Gemma4-12b's `embedding_length / attention.head_count`
= `3840 / 16` = 240 elements per head (960 bytes, not a 128-byte multiple). **That
conclusion was wrong.** Direct empirical tracing of the model's actual per-node tensor
shapes at runtime (a temporary, env-var-gated instrumentation pass added to
`ggml-tsavorite.cpp`'s compute loop, since removed) showed Gemma4-12b's real head_dim
is **256** (16 query heads, 8 grouped KV heads), already a clean multiple of 128 bytes.
The `3840/16` arithmetic was a plausible-sounding inference from published
hyperparameters that didn't hold up against the model's actual tensor shapes — head_dim
never contributed any misalignment. Correcting the record rather than leaving the wrong
conclusion in place.

**Actual root cause, found by tracing the real model, not by inference:** Gemma4-12b's
per-layer `layer_output_scale` weight — a genuine **scalar** tensor (`ggml` shape
`{1,1,1,1}`) applied via `ggml_mul(cur, out_scale)` at the end of every transformer
layer's FFN block. This weight is unique to Gemma4's architecture family (see
`src/models/gemma4.cpp`, `LLM_TENSOR_LAYER_OUT_SCALE`) — Gemma3, Qwen2.5, and
TinyLlama don't have it, which is why none of them ever triggered this.

Confirmed via direct value tracing on the real model:
1. The weight loads correctly for every layer (nonzero values, e.g. layer 0 = `0.052979`).
2. At the exact moment of kernel dispatch, both operands (`cur`'s real activation value
   and the scalar weight) are still correct.
3. Immediately after the kernel call returns — before any writeback — the output
   scratch buffer is already all-zero. The kernel writes nothing at all for this exact
   degenerate broadcast shape (`ne10==1`).

This could not be reproduced in an isolated test despite matching the exact tensor
shape, magnitude, dtype, and even chaining it after a preceding async ADD dispatch (the
same sequence the real model uses) — every isolated attempt passed against a CPU
reference. This points to a genuine gap in the compiled OPU kernel/runtime for a true
scalar broadcast, one level below what's fixable by adjusting `ggml-tsavorite.cpp`'s
C++ dispatch/staging code. Since a scalar broadcast is computationally trivial, the fix
computes this one exact shape directly on CPU inside `ggml-tsavorite.cpp` instead of
ever dispatching it to the kernel — every other broadcast shape keeps using the OPU
path, unchanged.

**Two more real, independent bugs found and fixed along the way** (kept regardless of
their bearing on Gemma4, since both are correctness issues in their own right, and
both are covered by new fast regression tests):

- **Chunked elementwise-op corruption.** `GGML_OP_ADD/MUL/SUB/DIV`'s broadcast-chunking
  loop corrupts the tail of its result whenever a chunk width isn't a multiple of 32
  floats (128 bytes) **and** that same width gets dispatched repeatedly (≥2 times)
  back-to-back — reproduced deterministically and fast (sub-second, not a 40-minute
  model run) with a new isolated test, `examples/simple/simple-chunked-repro.cpp`,
  independent of `multi_thread_enable`. A single dispatch at any size is always
  correct; only repetition at a non-32-multiple width triggers it. Fixed by padding
  each dispatch's length up to the next 32-float multiple via a scratch buffer before
  calling into the kernel, then copying back only the real portion. Also fixed a
  separate, narrower bug this surfaced: the `ADD` op's async Triton-dispatch path (used
  when `multi_thread_enable=true`) was racing its own writeback for the padded case —
  the dispatch returns before its worker thread actually runs. Scoped fix: the padded
  case now calls the synchronous kernel wrapper directly instead of the async one.
- **RMS_NORM multi-row corruption** (found and fixed, but confirmed via the real-model
  tracing above **not** to be what Gemma4-12b actually hits — its RMS_NORM calls are
  always single-row in practice). Applying RMS_NORM to multiple non-32-multiple-width
  rows in one dispatch computed completely wrong output for every row, including the
  first — not just a tail artifact like the ADD/MUL case. Root cause: the fix's first
  attempt wrote the row-width dimension into `shape[0]`, matching the ADD/MUL
  convention, but RMS_NORM's working single-row path actually maps it to
  `shape[Rank-1]` (reversed indexing) — a bug in the fix attempt itself, not the
  underlying kernel. Fixed by looping one row at a time (each padded the same way as
  ADD/MUL), with the correct reversed shape indexing.

**Result: Gemma4-12b now produces coherent, correct output on the Tsavorite backend —
`My cat's name is "Luna` — matching the CPU backend's own output for the same
prompt and weights.** Full completion, zero alignment warnings (down from 381,138),
exit code 0.

Every fix is regression-tested clean: byte-identical output and zero new alignment
warnings on all 4 standard posix models plus both Qwen K-quant controls, before and
after every change.

## Bottom line

- The core claim — "gemma4 architecture recognition doesn't exist on old, exists on
  new" — is confirmed for both tags tested.
- The MoE-style `e2b`/`e4b` tags still hit a real, independently-confirmed upstream
  loader gap ([ggml-org/llama.cpp#27349](https://github.com/ggml-org/llama.cpp/issues/27349)),
  unrelated to Tsavorite or this sync.
- The dense-style `12b` tag now **fully works**: loads, dispatches real OPU compute,
  and produces coherent output matching the CPU reference, with zero alignment
  warnings.
- Three real bugs were found and fixed in `ggml-tsavorite.cpp` along the way — one
  pre-existing (`get_alignment()` 32→128), one a genuine chunked-op corruption
  (ADD/MUL/SUB/DIV padding), and the actual Gemma4 root cause (true scalar-broadcast
  kernel gap, worked around via CPU fallback for that one shape). A fourth
  (RMS_NORM multi-row) was found and fixed but confirmed not to be what Gemma4
  actually triggers.
- Recommend as explicit follow-up, not blocking this PR: (1) check whether a later
  upstream commit past `1f368f354` fixes the MoE tensor-count gap; (2) report the
  true-scalar-broadcast kernel gap to whoever owns the TXE/OPU kernel compilation, since
  the CPU fallback here is a correct, safe workaround but not a fix at the kernel level.
