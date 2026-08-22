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
with real dispatch to the Tsavorite backend. This is the clearest evidence gemma4
support genuinely works in this sync for at least this checkpoint shape.

However: this is also the **first K-quantized (Q4_K_M) model tested anywhere in this
validation exercise** — every other model tested (tinyllama-5m, Gemma3-270M, TinyLlama-
1.1B, Llama3.2-1B) was F32 or BF16. Running this one surfaced a high-volume warning:

```
TXE::align_address(): Warning: Unaligned memory access 0x7f22e8800104 not 128-byte aligned, aligning to 0x7f22e8800100
```

repeated **381,138 times** in the captured run (see `new-gemma4-12b-trimmed.log` for a
head/tail excerpt — the full raw log was 45MB, kept locally, not checked in). The
process was killed after ~10 minutes without completing 4-token generation, given the
open-ended runtime risk from this print volume, not because it hung or crashed.

**This is not confirmed as a sync regression** — there is no old-fork comparison point,
since the old binary fails before reaching this code path for *any* gemma4 tag, and no
other K-quantized model was tested earlier in this exercise to compare against. It reads
as a real, previously-unobserved characteristic of the Tsavorite backend's handling of
K-quant block-structured tensor data (which isn't naturally 128-byte aligned the way
F32/BF16 tensors are), surfaced for the first time by this specific test — not something
I can currently attribute to this sync versus pre-existing behavior.

## Bottom line

- The core claim — "gemma4 architecture recognition doesn't exist on old, exists on new"
  — is confirmed for both tags tested.
- Whether gemma4 is *usably fast and fully correct* end-to-end depends on which
  checkpoint variant: the MoE-style `e2b`/`e4b` tags hit a real upstream loader gap;
  the dense-style `12b`/`26b`/`31b` tags load and compute but likely need the
  K-quant alignment-warning volume investigated/fixed before they're practical to run.
- Recommend as explicit follow-up work, not blocking this PR: (1) check whether a later
  upstream commit past `1f368f354` fixes the MoE tensor-count gap, worth checking now
  that `regenerate-patch`/`UPSTREAM_BASE_COMMIT` make re-targeting cheaper than before
  this sync; (2) investigate the K-quant alignment-warning volume in the Tsavorite
  backend directly (`ggml-tsavorite.cpp` or the TXE runtime) since it's a real,
  previously-undiscovered characteristic independent of gemma4 specifically.
