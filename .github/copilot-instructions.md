# Instructions for AI code review on this repository

This is a fork of `ggml-org/llama.cpp` maintained by Tsavorite (`tsisw`). Most PRs on
this repo fall into two very different categories that need different review scope:

## Upstream-sync PRs (large file counts, e.g. "Sync ... to upstream commit ...")

These PRs vendor a newer upstream `ggml-org/llama.cpp` commit forward and apply
`consolidated-patch.patch` (repo root) on top. They typically show thousands of changed
files, but the vast majority of that diff is upstream's own already-reviewed commit
history landing via the version bump — **do not review it file-by-file**.

For this PR type, scope review to only:
- `consolidated-patch.patch` at the repo root — every Tsavorite-specific change, as one
  patch file. This is the actual diff to read.
- `UPSTREAM_BASE_COMMIT` at the repo root — the upstream SHA this sync vendored forward
  to and that `consolidated-patch.patch` was regenerated against. Check it matches the
  commit named in the PR title/description; a mismatch here means the patch may not
  apply cleanly against the base it claims.
- `tsi-pkg-build.sh` — the build/packaging script, tracked directly (excluded from the
  patch above on purpose).
- Any one-line config default changes called out explicitly in the PR description
  (e.g. `tsavorite-model-deployment.yaml`).
- The linked testing-evidence documents in the PR description (posix/tsisim/model
  validation summaries under `docs/`) — these are the correctness evidence, review them
  as evidence, not as code to critique line-by-line.

Everything else in an upstream-sync PR's diff is upstream's own code landing via the
version bump and does not need comment-by-comment review.

## Normal PRs (everything else)

Regular feature/bugfix PRs — review normally, full diff, standard scrutiny.

## How to tell which kind of PR this is

Check the PR title/description for an explicit "How to review this PR" section. If
present, follow it. If a PR's title mentions "sync" or "upstream" and touches an
unusually large number of files relative to its description, treat it as an
upstream-sync PR per the rules above unless the description says otherwise.
