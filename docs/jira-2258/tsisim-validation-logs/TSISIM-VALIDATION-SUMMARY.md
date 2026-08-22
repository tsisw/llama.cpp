# tsisim validation — JIRA-2258

Real TSI simulator (tsisim) runs of this sync's package (`tsi-ggml-0.4.24.tz`), via the
production harness `run_llama_cli.sh`. Unlike the posix comparison, these are **not yet**
a side-by-side old-vs-new capture — the reporter (akapoor) ran these against the new
package directly and will attach old-tree tsisim logs for the same models separately;
this file will be updated with a full comparison table once those land.

## Confirmed so far (new tree, real tsisim hardware)

| Model | Result | txe_count (auto-detected) | user_dram_size_gb | Notes |
|---|---|---|---|---|
| Tiny-Llama-v0.3-FP32-1.1B-F32.gguf | ✅ completes, generates `is Luna.` | 4 | 8 | see [new-tinyllama-1.1b-tsisim.log](new-tinyllama-1.1b-tsisim.log) |
| tinyllama-vo-5m-para.gguf | ✅ completes, generates `was Tim. He loved` | 4 | 16 | see [new-tinyllama-5m-tsisim.log](new-tinyllama-5m-tsisim.log) |

Both runs show real OPU dispatch (`MUL_MAT`, `ADD`, `MUL`, `RMS_NORM`, `GLU`, etc. hitting
`OPU` in the `=== GGML Perf Summary ===` table, with `MUL_MAT` correctly split between a
`CPU` fallback path and `OPU` offload), and both show `ggml.sh`'s deployment-yaml
regeneration correctly reporting `txe_count` auto-detected from the simulator's actual
TXE count (4), matching the pattern documented in `tsi-pkg-build.sh`.

Per the reporter, both results match what's currently deployed for these models on
tsisim (same generated output, same op-dispatch pattern) — full side-by-side logs to
follow.

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
- The Tiny-Llama-1.1B run above used the previous `user_dram_size_gb: 8` default (this
  log predates the reporter picking up the repackaged `.tz`); it completed fine at 8 GB
  for this model, consistent with the posix finding that only Llama3.2-1B needed the
  larger default.
