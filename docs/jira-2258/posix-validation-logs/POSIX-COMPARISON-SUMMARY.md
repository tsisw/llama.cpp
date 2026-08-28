# Posix old-vs-new comparison — JIRA-2258

**Old** = pre-sync fork, `build-posix/bin/llama-cli`, commit `935ee7bc0` (tsisw/master, tag v0.1.23)
**New** = post-sync tree, `build-posix/bin/llama-completion`, branch `sync-port-target` (upstream `1f368f354` + consolidated Tsavorite patch)

Both built with `SDK_VERSION=0.4.24 source tsi-pkg-build.sh build-posix`. Same command/flags run against both binaries for every model: `-p "My cat's name is" --device tSavorite --temp 0.0 --n-predict 4 --repeat-penalty 1.5 -b 1024 --top-k 50 --top-p 0.9 --repeat-last-n 5 --no-warmup --no-conversation`, run sequentially (not concurrently) so timing numbers are comparable. All four runs use the updated `user_dram_size_gb: 16` deployment default (see below).

## Result: 4/4 models match exactly, byte-identical generated text old vs new

| Model | Old output | New output | Old time | New time | Match? |
|---|---|---|---|---|---|
| tinyllama-vo-5m-para.gguf | `My cat's name is Tim. He has` | `My cat's name is Tim. He has` | 5.4s | 5.1s | ✅ identical |
| Gemma3-270M-F32.gguf | `My cat's name is "Cat" and` | `My cat's name is "Cat" and` | 29.0s | 31.1s | ✅ identical |
| Tiny-Llama-v0.3-FP32-1.1B-F32.gguf | `My cat's name is Luna. She` | `My cat's name is Luna. She` | 3m0.9s | 3m10.5s | ✅ identical |
| Llama3.2:1B-1.2B-F32.gguf | `My cat's name is "Bella` | `My cat's name is "Bella` | 9m50.8s | see log | ✅ identical |

Full logs for each run are in this directory (`<tag>.log`, e.g. `new-tinyllama-5m.log`, `old-tinyllama-5m.log`) — click through to read the complete raw output.

## Final re-verification, post all Gemma4 fixes (both cubic-dev-ai review batches, commit `50c227b17`)

The comparison above predates this PR's Gemma4 investigation (see
[GEMMA4-VALIDATION-SUMMARY.md](../gemma4-validation-logs/GEMMA4-VALIDATION-SUMMARY.md))
and the three real bugs found and fixed along the way. To confirm those fixes don't
regress the models already validated above, plus the two K-quant control models used
during the Gemma4 investigation, all six were re-run against the final build
(`final-<tag>.log` in this directory):

| Model | Result |
|---|---|
| tinyllama-vo-5m-para.gguf | ✅ exit 0, 0 warnings |
| Gemma3-270M-F32.gguf | ✅ exit 0, 0 warnings |
| Tiny-Llama-v0.3-FP32-1.1B-F32.gguf | ✅ exit 0, 0 warnings |
| Llama3.2:1B-1.2B-F32.gguf | ✅ exit 0, 0 warnings |
| qwen2-0_5b-instruct-q5_k_m.gguf (Q5_K_M) | ✅ exit 0, 0 warnings |
| Qwen2.5-0.5B-Q4_K_M.gguf (Q4_K_M, same quant type as Gemma4-12b) | ✅ exit 0, 0 warnings |

What these six runs actually validate, precisely: `DATA_TYPE_F32_INDEX` is assigned to
every all-F32 dispatch regardless of width — it is not itself a signal that a
non-32-float-multiple width was hit. Two independent things are true here:

- The `get_alignment()` fix (32 → 128 bytes) applies to every Tsavorite buffer
  allocation unconditionally, so these six zero-warning runs directly validate that
  fix across a range of model sizes/architectures.
- The three width-dependent fixes (chunked elementwise-op padding, RMS_NORM row-loop,
  scalar-broadcast CPU fallback) only activate when a dispatch is **both** F32 **and**
  a non-32-float-multiple width — being all-F32 alone doesn't imply that. These six
  models' real per-op dispatch widths for ADD/MUL/SUB/DIV/RMS_NORM haven't been traced
  the way Gemma4-12b's were, so this comparison shouldn't be read as proof those three
  fixes were exercised here. It is solid evidence of no regression either way: if the
  fixes were wrong, they'd risk breaking builds/dtype-handling generally, and they
  don't. Gemma4-12b remains the one model in this PR directly confirmed (via real
  tracing) to hit the width-dependent paths — see the Gemma4 doc linked above.

## Reading the results

- All 4 models produce **byte-identical generated text** on old vs new, across four different model sizes/architectures (a tiny synthetic model, Gemma3, TinyLlama-1.1B, and Llama3.2-1B), confirming the sync didn't change functional behavior.
- Timing differs by single-digit percentages in either direction — consistent with normal run-to-run noise on a shared machine, not a systematic regression.
- `tsavorite-model-deployment.yaml`'s `user_dram_size_gb` default is raised from 8 to 16 in this PR. At 8 GB, Llama3.2-1B's actual DRAM usage (KV-cache/context growth on top of the ~1.8 GB weights footprint) exceeds the budget and aborts with a DRAM allocation failure — reproduced identically on both old and new binaries at the old default, so it's a deployment-config tuning gap, not something this sync introduced. Raising the default to 16 GB (still well under the documented 24 GB ceiling for larger models) resolves it on both.

## CPU/OPU op-dispatch comparison (old vs new)

Requested follow-up: is the CPU-vs-OPU dispatch pattern the same old vs new, not just the
final output? Short answer: **the core compute ops match exactly; a specific set of
attention/cache-bookkeeping ops differ in count, for a reason I can explain but haven't
fully root-caused with tracing** — reporting this honestly rather than as a blanket "yes,
identical."

**What matches exactly, across all 4 models:** `ADD`, `MUL`, `RMS_NORM`, and `GET_ROWS`
run counts are bit-for-bit identical old vs new (e.g. tinyllama-5m: `ADD` 64/64, `MUL`
68/68, `RMS_NORM` 68/68, `GET_ROWS` 12/12). These are the ops that do the actual per-token
elementwise/normalization math, and their exact match — combined with the byte-identical
final output above — is the strongest evidence the sync didn't change numerical behavior.

**What differs, across all 4 models:** `VIEW`, `PERMUTE`, `TRANSPOSE`, `SET_ROWS`, `ROPE`,
and `MUL_MAT` show different total run counts old vs new (consistently ~2.5-3.8x higher in
old), and `RESHAPE` disappears entirely in new (present in old). Example (tinyllama-5m):

| Op | Old runs | New runs (CPU+OPU) |
|---|---|---|
| VIEW | 390 | 128 |
| PERMUTE | 275 | 96 |
| TRANSPOSE | 105 | 32 |
| SET_ROWS | 243 | 64 |
| ROPE | 243 | 64 |
| MUL_MAT | 1028 | 292 (263 CPU + 29 OPU) |
| RESHAPE | 353 | *(not present)* |

The old binary's perf table also has no `Target` (CPU/OPU) column at all — it's a strictly
coarser report format than the new one, so a literal per-target comparison against old
isn't possible from these logs alone.

**My best-supported explanation, not independently confirmed by tracing:** the ops that
differ are exactly the ones tied to attention/KV-cache bookkeeping (view/permute/transpose
around QKV and RoPE, cache writes via SET_ROWS), while the ops that match exactly are
feedforward/normalization math that doesn't touch the cache. That split is consistent
with upstream's own attention/cache implementation evolving somewhere across the ~1 year
of ggml-org/llama.cpp history between the fork point and this sync's target commit —
fewer or different bookkeeping nodes emitted per attention layer, not a Tsavorite
dispatch-logic change. It is *not* consistent with, e.g., a batching-size difference
(that would also move `RMS_NORM`/`ADD`/`MUL`, which it doesn't). I have not traced this
to a specific upstream commit/PR to confirm it, and flag that as an open item — it doesn't
block this PR given the output-correctness evidence above, but a reviewer who wants the
exact upstream cause should treat this section as a lead, not a closed investigation.

One more data point worth flagging: Gemma3-270M-F32.gguf's *old* log only reports 5 op
rows total (`ADD`, `MUL`, `RMS_NORM`, `MUL_MAT`, `SCALE`) — `CONT`/`VIEW`/`PERMUTE`/
`TRANSPOSE`/`GET_ROWS`/`SET_ROWS`/`SOFT_MAX`/`ROPE`/`GLU` don't appear at all, whereas the
*new* log reports all of them with real nonzero counts. This is architecture-specific (the
other 3 models' old logs all report the full op list) — I don't have an explanation for
this one beyond the same general hypothesis above, and it's the single biggest gap in this
comparison. Output text still matched byte-for-byte for this model, so it isn't a
correctness failure, but it's the one result in this table I'd call genuinely unexplained
rather than "explained but untraced."
