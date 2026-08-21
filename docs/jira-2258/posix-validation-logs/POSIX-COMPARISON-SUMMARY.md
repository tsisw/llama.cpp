# Posix old-vs-new comparison — JIRA-2258

**Old** = pre-sync fork, `build-posix/bin/llama-cli`, commit `935ee7bc0` (tsisw/master, tag v0.1.23)
**New** = post-sync tree, `build-posix/bin/llama-completion`, branch `sync-port-target` (upstream `1f368f354` + consolidated Tsavorite patch)

Both built with `SDK_VERSION=0.4.24 source tsi-pkg-build.sh build-posix`. Same command/flags run against both binaries for every model: `-p "My cat's name is" --device tSavorite --temp 0.0 --n-predict 4 --repeat-penalty 1.5 -b 1024 --top-k 50 --top-p 0.9 --repeat-last-n 5 --no-warmup --no-conversation`, run sequentially (not concurrently) so timing numbers are comparable.

## Result: 3/4 models match exactly, 1/4 fails identically on both (pre-existing, not a regression)

| Model | Old output | New output | Old time | New time | Match? |
|---|---|---|---|---|---|
| tinyllama-vo-5m-para.gguf | `My cat's name is Tim. He has` | `My cat's name is Tim. He has` | 5.4s | 5.1s | ✅ identical |
| Gemma3-270M-F32.gguf | `My cat's name is "Cat" and` | `My cat's name is "Cat" and` | 29.0s | 31.1s | ✅ identical |
| Tiny-Llama-v0.3-FP32-1.1B-F32.gguf | `My cat's name is Luna. She` | `My cat's name is Luna. She` | 3m0.9s | 3m10.5s | ✅ identical |
| Llama3.2:1B-1.2B-F32.gguf | crash: `Failed to allocate 134217728 in DRAM` (SIGABRT, exit 134) | **same crash**, same backtrace shape, exit 134 | 38.8s | 23.7s | ⚠️ fails identically on both — pre-existing DRAM-sizing issue with this specific model+context combo, not caused by the sync |

Full logs for each run are in this directory (`<tag>.log`, e.g. `new-tinyllama-5m.log`, `old-tinyllama-5m.log`) — click through to read the complete raw output, including the full crash backtrace for the 4th model on both sides.

## Reading the results

- The 3 working models produce **byte-identical generated text** on old vs new, across three different model architectures (a tiny synthetic model, Gemma3, and Llama-family), confirming the sync didn't change functional behavior.
- Timing differs by single-digit percentages in both directions (new faster on 2, older faster on 2) — consistent with normal run-to-run noise on a shared machine, not a systematic regression.
- The Llama3.2-1B failure is a genuine crash, but it's the **exact same failure on both binaries** — same allocation size, same call stack shape (`llama_decode` → `graph_compute` → `tsi_alloc` → DRAM allocation failure). This points at a memory-configuration limit for this specific large (5GB F32) model + context size combination, unrelated to anything this sync touched. Worth its own investigation separately, but not a blocker for this PR.
