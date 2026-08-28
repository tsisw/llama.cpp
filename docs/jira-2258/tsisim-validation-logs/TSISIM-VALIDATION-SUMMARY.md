# tsisim validation — JIRA-2258

Every `*.log` file named below is attached to PR #162's description, not committed to
this directory — they're evidence for review, not permanent reference material. This
doc quotes the relevant excerpts inline; the full raw logs are in the PR.

Real TSI simulator (tsisim) runs of this sync's package (`tsi-ggml-0.4.24.tz`), via the
production harness `run_llama_cli.sh`. Unlike the posix comparison, these are **not yet**
a side-by-side old-vs-new capture — the reporter (akapoor) ran these against the new
package directly and will attach old-tree tsisim logs for the same models separately;
this file will be updated with a full comparison table once those land.

## Confirmed so far (new tree, real tsisim hardware)

| Model | Result | txe_count (auto-detected) | user_dram_size_gb | Notes |
|---|---|---|---|---|
| Tiny-Llama-v0.3-FP32-1.1B-F32.gguf (run 1) | ✅ completes, generates `is Luna.` | 4 | 8 | see `new-tinyllama-1.1b-tsisim.log` |
| Tiny-Llama-v0.3-FP32-1.1B-F32.gguf (run 2) | ✅ completes, generates `is Luna.` | 4 | 16 | see `new-tinyllama-1.1b-tsisim-run2.log` |
| tinyllama-vo-5m-para.gguf | ✅ completes, generates `was Tim. He loved` | 4 | 16 | see `new-tinyllama-5m-tsisim.log` |
| Gemma3-270M-F32.gguf | ✅ completes, generates `is "cat" and` | 4 | 16 | see `new-gemma3-270m-tsisim.log` |
| Llama3.2:1B-1.2B-F32.gguf | ✅ completes, generates `is molly and she` | 4 | 16 | see `new-llama3.2-1b-tsisim.log` |

Both runs show real OPU dispatch (`MUL_MAT`, `ADD`, `MUL`, `RMS_NORM`, `GLU`, etc. hitting
`OPU` in the `=== GGML Perf Summary ===` table, with `MUL_MAT` correctly split between a
`CPU` fallback path and `OPU` offload), and both show `ggml.sh`'s deployment-yaml
regeneration correctly reporting `txe_count` auto-detected from the simulator's actual
TXE count (4), matching the pattern documented in `tsi-pkg-build.sh`.

**Reproducibility check (the two Tiny-Llama-1.1B runs above):** verified programmatically
— the `Op`/`Target`/`Runs`/`TSI_KERNEL-RUN` columns are byte-for-byte identical between
run 1 and run 2 (e.g. `MUL_MAT` splits `1109 CPU / 85 OPU` in both), despite run 2 using
the updated 16GB DRAM default and landing ~15 minutes later in wall-clock time. Only the
timing columns differ, by normal run-to-run noise. This is good independent evidence of
deterministic op-dispatch behavior on real hardware, not just on posix.

**Cross-environment consistency check (Gemma3-270M, Llama3.2-1B):** both tsisim runs'
op-dispatch *structure* matches their posix counterpart — same op set, same split
pattern (`MUL_MAT` split CPU/OPU in both models; `ROPE` also split for Gemma3-270M,
CPU-only for Llama3.2-1B, matching posix exactly), and `MUL_MAT`'s OPU run count
matches exactly in both cases (Gemma3-270M: 69/69; Llama3.2-1B: 61/61). Most other
counts scale by a consistent ~1.5x on tsisim vs posix — expected, not a discrepancy:
tsisim auto-detects `txe_count=4` here vs posix's `txe_count=1`, and `run_llama_cli.sh`
isn't necessarily using identical flags (including sampling temperature — the generated
text differs from the posix run's deterministic `--temp 0.0` output, e.g. Llama3.2-1B
gives `molly` on tsisim vs `Bella` on posix; not a discrepancy, just non-deterministic
sampling in the production harness vs the greedy posix comparison script). Not a strict
numeric match by design, but the qualitative dispatch pattern being identical across
posix and real hardware, for two different models now, is a good consistency signal.

**Llama3.2-1B DRAM fix confirmed on real hardware:** this is the model that crashed on
posix at the old `user_dram_size_gb: 8` default (see POSIX-COMPARISON-SUMMARY.md). On
tsisim, with the fixed 16GB default, it completes cleanly with no allocation failure —
direct hardware confirmation of the fix, not just posix.

## Update 2026-08-26: Gemma4-12b tested on real tsisim hardware, plus full reconfirmation

Full raw session log:
`new-tsisim-full-session-2026-08-26.log`.
Package built from this PR's final commit (`c56d67765`, both cubic-dev-ai review
batches addressed).

**Gemma4-12b-Q4_K_M.gguf, the model this whole investigation is about, now confirmed on
real tsisim hardware (not just posix):**

| Backend | Output | Notes |
|---|---|---|
| Tsavorite (`--device tSavorite`) | `<\|channel>thought` / `<channel\|>The` | real OPU dispatch, `MUL_MAT` 144 runs/2880 kernel-calls |
| CPU (`--device none`, tsisim) | `<\|channel>thought` / `<channel\|>It` | control run, same hardware |
| CPU (`--device none`, posix) | `<\|channel>thought` / `<channel\|>It` | cross-checked separately on posix |

Both `<|channel>thought` lines are the chat template's own "thinking channel" preamble
(this GGUF's `--jinja` template, not a bug) — 5 tokens isn't enough to get past it on
either backend, on either platform. **tsisim CPU matches posix CPU byte-for-byte**,
confirming CPU execution is identical across platforms as expected. Tsavorite's
one-token divergence from CPU (`The` vs `It`) is the same class of minor
greedy-decoding variance already documented in
[GEMMA4-VALIDATION-SUMMARY.md](../gemma4-validation-logs/GEMMA4-VALIDATION-SUMMARY.md)
(`**"Luna0` vs `"Luna"`) — not corruption. Op/kernel counts on the Tsavorite run look
structurally sound throughout.

**Gemma4-12b DRAM requirement on tsisim is higher than posix's:** this PR's new
`user_dram_size_gb: 16` default (bumped from 8, proven sufficient on posix for all
models including Gemma4-12b) was **not** sufficient on tsisim — with `txe_count: 20`,
16GB produced `TXE manager returned no TXEs (response_count=0)` / `Failed to initialize
runtime`, not the loud SIGABRT the old 8GB default gave. Raising to 30GB (the value
already documented in the pre-existing tsisim validation notes for this exact reason)
resolved it cleanly. **This is a tsisim deployment-tuning value, not a code defect in
this PR** — the same 16GB default is correct and sufficient on posix; tsisim's real
per-TXE memory footprint at `txe_count: 20` is simply higher. Worth calling out
explicitly in the PR/review guide so nobody mistakes this for a regression.

**Full reconfirmation, all 4 previously-tested models, byte-identical output to the
original pre-Gemma4-investigation tsisim baseline above, despite this PR's fixes
landing in between:**

| Model | Output (matches original baseline exactly) |
|---|---|
| tinyllama-vo-5m-para.gguf | `was Tim. He loved` |
| Tiny-Llama-v0.3-FP32-1.1B-F32.gguf | `is Luna.` (MUL_MAT 1109 CPU/85 OPU split also matches exactly) |
| Llama3.2:1B-1.2B-F32.gguf | `is molly and she` |
| Gemma3-270M-F32.gguf | `is "cat" and` (reconfirmed on CPU too) |

This is strong evidence of zero regression on real hardware: every fix in this PR is
scoped to non-32-float-multiple F32 dispatch widths (see
[POSIX-COMPARISON-SUMMARY.md](../posix-validation-logs/POSIX-COMPARISON-SUMMARY.md)),
and none of these four models hit that condition — so byte-identical output across two
separate tsisim sessions, with this PR's code changes landing in between, is exactly
what correct scoping predicts.

Per the reporter, results match what's currently deployed for these models on tsisim
(same generated output, same op-dispatch pattern) — full side-by-side old-tree logs
still to follow.

## What these runs do and don't confirm about JIRA-2258's fixes

- They confirm the sync runs successfully end to end on real tsisim hardware for two
  models, including real OPU offload — not just posix.
- They do **not** specifically exercise the `multi_thread_enable` preservation fix
  (commit `646ba3fab`): both runs use the default `true`, which would behave identically
  whether or not that fix were present. That fix was verified separately with a
  standalone unit test of `update_one_tsavorite_deployment_yaml()` (seeding
  `multi_thread_enable: false` and confirming it survives regeneration) — see the PR
  description. Confirming it live on tsisim would require deliberately setting
  `multi_thread_enable: false` in the deployed yaml before a run.
- Run 1 used the previous `user_dram_size_gb: 8` default (predates the reporter picking
  up the repackaged `.tz`); run 2 used the updated 16GB default. Tiny-Llama-1.1B
  completes fine at *both* values — consistent with the posix finding that only
  Llama3.2-1B specifically needed the larger default, not every model.
