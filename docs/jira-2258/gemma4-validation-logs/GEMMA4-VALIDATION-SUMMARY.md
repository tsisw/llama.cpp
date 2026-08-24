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

**Conclusion, now evidence-based rather than inferred — with one causality gap flagged
by review that's worth stating precisely:**
1. **Not reproduced in either Qwen control, on either binary.** Old and new behave
   identically for every non-Gemma4 K-quantized model tested (Q5_K_M and Q4_K_M, the
   latter being the *exact* format that triggered warnings on Gemma4-12b). This rules
   out "any K-quantized model triggers this" and "old vs new differ for K-quant in
   general."
2. **Gemma4 causality specifically is still open, not ruled out.** The old binary can't
   load Gemma4 at all — it fails before reaching this code path — so there is no
   old-binary-running-Gemma4 data point to compare against directly. `align_address()`
   itself lives in a separate, untouched repository (`tsisw/TXE-FFM`), but that doesn't
   rule out this sync changing *what addresses get passed into it* for Gemma4
   specifically (e.g. via how its tensors/graph get built). The Qwen controls narrow the
   likely explanation away from "general K-quant issue" — they do not prove "not caused
   by this sync" for Gemma4 itself. Stating it this way rather than the stronger claim.
3. **Still not fully root-caused to the exact line of code inside Gemma4's handling**
   that produces the misaligned addresses — that would require tracing through an SDK
   layer between `ggml-tsavorite.cpp` and the TXE simulator that isn't in this repo.
   That's real follow-up work; scope is narrowed (not a general K-quant issue, doesn't
   threaten any other model in this PR) but Gemma4-specific sync causality is not
   closed out, per point 2 above.

## Bottom line

- The core claim — "gemma4 architecture recognition doesn't exist on old, exists on new"
  — is confirmed for both tags tested.
- Whether gemma4 is *usably fast and fully correct* end-to-end depends on which
  checkpoint variant: the MoE-style `e2b`/`e4b` tags hit a real upstream loader gap;
  the dense-style `12b`/`26b`/`31b` tags load and dispatch real compute, but output
  correctness is unverified (the run was killed before producing text) and the
  alignment-warning finding above means "didn't crash" isn't strong evidence of
  correctness on its own.
- The alignment-warning finding is **not reproduced by either non-Gemma4 K-quant
  control, on either binary** — ruling out "general K-quant issue" and "any K-quant
  model regresses on new." Whether this sync specifically changed Gemma4's own
  memory-access pattern is **not** ruled out by these controls (the old binary can't
  run Gemma4 at all, so there's no direct old-vs-new comparison for Gemma4 itself) —
  this does not block the PR, since it doesn't threaten any other model covered here,
  but should not be described as "confirmed unrelated to this sync."
- Recommend as explicit follow-up work, not blocking this PR: (1) check whether a later
  upstream commit past `1f368f354` fixes the MoE tensor-count gap, worth checking now
  that `regenerate-patch`/`UPSTREAM_BASE_COMMIT` make re-targeting cheaper than before
  this sync; (2) trace why Gemma4 specifically produces misaligned addresses in the
  Tsavorite backend, and confirm whether the silent address-realignment in
  `TXE::align_address()` actually corrupts output for this model, and whether this
  sync changed the inputs to it, before trusting any Gemma4 dense-variant result.
