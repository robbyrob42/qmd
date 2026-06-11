# Embedding corpus invalidation & full rebuild — 2026-06

> **Fork-owned doc.** A `docs/FORK-*.md` filename upstream QMD does not use, so it
> survives `git merge upstream/main` without conflicts (the existing `!docs/*.md`
> rule in `.gitignore` tracks it — no `.gitignore` edit needed, no new conflict
> surface). Pair with the conflict-isolated module `src/legacy-embed-adoption.ts`.
> Do not move this into `CHANGELOG.md` / `CLAUDE.md` / `README.md` — those are
> upstream-tracked and will conflict.

## TL;DR

After merging upstream into this fork, **the entire stored embedding corpus
(~1.9M chunks) no longer reproduces under the current embedding pipeline** and must
be rebuilt with `qmd embed --force`. This is a one-time, expected cost of adopting
upstream's path/chunking changes. There is **no shortcut** — the old vectors are
genuinely incompatible, proven empirically (below). The legacy-vector *adoption*
optimization (relabel-instead-of-recompute) correctly **refused** to run, because
the vectors really did change.

## How we got here

1. `qmd embed` was dying every run at exactly 30m 0s ("session expired"). Root
   cause was a hardcoded 30-minute `maxDuration` watchdog, not a real LLM expiry.
   A ~1.9M-chunk corpus needs ~18–22h, so it could never finish. Fixed on branch
   `fix/embed-legacy-adoption-and-throughput` (commit removing the self-abort;
   watchdog now defaults to disabled, env-gated via `QMD_EMBED_MAX_DURATION_MS`).
2. Separately, a model-URI rename (`embeddinggemma` → fully-qualified
   `hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf`) plus the
   introduction of an `embed_fingerprint` orphaned ~1,723,188 legacy chunks
   (15,496 docs) as `model="embeddinggemma", embed_fingerprint=''`. They showed up
   as "pending", implying a full multi-hour re-embed.
3. Hypothesis at the time: those legacy vectors were *identical* to what the
   current pipeline produces (same model file, same chunking) and could simply be
   **relabeled** (model + fingerprint) instead of recomputed. We built
   `src/legacy-embed-adoption.ts` + an adoption step in `qmd doctor` to do exactly
   that — but **only after a sample re-embed reproduces the stored vector within a
   tight threshold (0.0001 cosine)**. Fail-safe by design.

## What `qmd doctor` actually reported (2026-06-10)

```
⚠ legacy fingerprint adoption: legacy sample differs from current fingerprint
  (distance 0.059469)                      # 0.0001 threshold → REFUSED (correct)
⚠ embedding freshness: 15,720 active documents need embeddings
⚠ embedding fingerprints: 1,528 docs current / 15,496 docs legacy-stale
  (current: 182,023 chunks; legacy: 1,723,188 chunks)
⚠ embedding vector sample: 2/3 sampled chunks differ from stored vectors
  (distance 0.046732)                      # even CURRENT-labeled vectors drifted
```

Two findings, not one:
- **Legacy vectors don't reproduce** (0.059 ≫ 0.0001) → adoption refused. The
  fail-safe worked; relabeling would have pinned mismatched vectors to the current
  fingerprint and silently poisoned vector search.
- **Even current-labeled vectors don't reproduce** (0.047). So this is not just a
  stale label — the current pipeline can't regenerate what's on disk.

## Determinism probe (the decision-maker)

Before committing ~18h of GPU to a `--force` rebuild, we ran a read-only probe:
for several currently-embedded chunks, embed the *same formatted input twice in one
session* (`d_self`) and compare a fresh embedding to the *stored* vector
(`d_stored`). Mirrors `doctor`'s `checkEmbeddingVectorSamples` exactly
(`formatDocForEmbedding` / `extractTitle` / `chunkDocumentByTokens`).

```
chunk                d_self   d_stored   path
56bbded517647668   0.000000   0.077689   …JPM-2025…
2edc0bf4c0db8110   0.000000   0.305614   …Algorithmic-Trading…
071e122f27f7f414   0.000000   0.533940   …Amazon-hybrid-reasoner…
46fec3d355207979   0.000000   0.131857   …AI-for-Drug-Development…
fe1fc2a898d3b902   0.000000   0.084445   …Grok-3…
max d_self = 0.000000   max d_stored = 0.533940
```

Interpretation:
- **`d_self = 0.000000` on every sample → the pipeline is fully deterministic.**
  No GPU floating-point drift. Therefore `qmd embed --force` **converges** to a
  self-consistent corpus that reproduces on the next `qmd doctor` — it is **not** a
  treadmill where rebuilt vectors fail again.
- **`d_stored` is large AND highly variable (0.08 → 0.53).** Far too big for a mere
  title/path string change on a 900-token chunk. The most likely dominant cause is
  that **chunk boundaries shifted** in the upstream merge: the probe re-chunks the
  current body and reads `chunks[seq]`, so if boundaries moved, `chunks[seq]` is now
  *different text* than what produced the stored vector at that `seq` — which
  explains both the magnitude and the per-doc variance. The
  handelize→literal-path title change (`extractTitle` falls back to the path
  basename, which feeds `formatDocForEmbedding`) is a secondary contributor.

> Earlier framing of "it's only the path/title change" was too narrow. The evidence
> says the embedding *input* changed more broadly (chunking + title), which makes a
> full rebuild even more clearly correct.

### Reproducing the probe later

The probe was a throwaway (not committed). To recreate: a Bun script importing
`createStore` (call with an explicit DB path — `createStore` refuses the implicit
global path), `resolveModels`, `setDefaultLlamaCpp(new LlamaCpp({...}))`,
`withLLMSession`, `formatDocForEmbedding`, `extractTitle`, `chunkDocumentByTokens`,
`getEmbeddingFingerprint`. Sample current-fingerprint rows with the same SQL
`doctor` uses, embed each input twice for `d_self`, and compare to
`vectors_vec.embedding` (decode as `Float32Array`) for `d_stored`. Run read-only;
it mutates nothing.

## Action

```sh
qmd embed --force
```

- `--force` (not plain `qmd embed`): plain `embed` rebuilds only stale-fingerprint
  docs and would **skip** the 1,528 "current" docs — but some of those drifted too
  (a chunking/title change does not bump the fingerprint, so qmd cannot tell which
  current docs are stale without re-embedding them). `--force` rebuilds all ~1.9M
  chunks into one homogeneous vector space.
- **Env vars: none required.** The perf fix is what makes this finish in one pass.
  Optional throughput tuning once stable:
  `QMD_EMBED_CONCURRENCY=48 QMD_EMBED_BATCH_SIZE=4096 qmd embed --force`.
- After it finishes, rerun `qmd doctor` — `embedding vector sample` and
  `embedding freshness` should both go green.

## Why the adoption code stays (even though it didn't help here)

`src/legacy-embed-adoption.ts` + the `doctor` adoption step are still correct and
valuable: when a model-URI rename happens *without* a pipeline change, they reclaim
the corpus by relabeling instead of recomputing. Here the pipeline genuinely
changed, so the fail-safe correctly declined. Keep both — they cost nothing when
they don't fire and protect the index when they do.

## Embedding tuning knobs (fork defaults)

| Env var | Default | Effect |
|---|---|---|
| `QMD_EMBED_MAX_DURATION_MS` | `0` (disabled) | embed-session watchdog; set positive to re-arm for unattended jobs |
| `QMD_EMBED_CONCURRENCY` | `32` | `getEmbeddingFor` calls in flight per embedding context |
| `QMD_EMBED_BATCH_SIZE` | `2048` (= context size) | embedding context batch size in tokens |
