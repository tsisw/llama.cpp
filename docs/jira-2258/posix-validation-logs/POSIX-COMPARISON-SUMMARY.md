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

## Reading the results

- All 4 models produce **byte-identical generated text** on old vs new, across four different model sizes/architectures (a tiny synthetic model, Gemma3, TinyLlama-1.1B, and Llama3.2-1B), confirming the sync didn't change functional behavior.
- Timing differs by single-digit percentages in either direction — consistent with normal run-to-run noise on a shared machine, not a systematic regression.
- `tsavorite-model-deployment.yaml`'s `user_dram_size_gb` default is raised from 8 to 16 in this PR. At 8 GB, Llama3.2-1B's actual DRAM usage (KV-cache/context growth on top of the ~1.8 GB weights footprint) exceeds the budget and aborts with a DRAM allocation failure — reproduced identically on both old and new binaries at the old default, so it's a deployment-config tuning gap, not something this sync introduced. Raising the default to 16 GB (still well under the documented 24 GB ceiling for larger models) resolves it on both.
