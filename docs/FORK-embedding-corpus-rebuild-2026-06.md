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

## Run status & operational rules (2026-06-11)

- The full re-embed is **in progress**, resumed as plain `qmd embed` (NOT `--force`).
  At last check ~30% by input bytes, ~33 KB/s, ETA ~8–9h. GPU was pinned ~96–100%
  (compute-bound — do not bother raising concurrency/batch; the 300M model on the
  M1 Max is the ceiling). Cooling (off-desk VESA mount + barrel fan) holds clocks.
- **RESUME RULE (critical):** if the run stops (sleep/crash/Ctrl-C), restart with
  **plain `qmd embed`** — it skips already-written chunks and continues. **Never
  re-run `qmd embed --force`** to resume: `--force` calls `clearAllEmbeddings()`
  once at start (store.ts:1611) and wipes all progress, restarting the ~1.9M from
  zero. `--force` is a one-time kickoff only.
- When complete: rerun `qmd doctor`. `embedding vector sample` and
  `embedding freshness` should both go green. `qmd` alias points directly at the
  repo binary; `dist/` was built 2026-06-10 19:41 from this branch (contains the
  perf fixes + the 900-token-cap chunker), so it reflects current source.

## Open investigation: "Batch text truncated to fit embedding context (2048 tokens)" flood

**Symptom:** during the run, a bounded region of ~20–30k chunks flooded this warning
(~1 per chunk), then stopped — a contiguous *cluster of documents*, not corpus-wide.
~1.5% of chunks got their tail past token 2044 truncated before embedding. Harmless
to the run (deterministic, doctor still passes); a quality nick on that cluster.

**Where it fires:** `LlamaCpp.embedTextsWithContext` (llm.ts ~1350) calls
`truncateToContextSize` per text; limit = `min(QMD_EMBED_CONTEXT_SIZE=2048,
embedModel.trainContextSize)` = 2048 for embeddinggemma. The embed input is
`formatDocForEmbedding(chunk.text, title)` = `title: {title} | text: {chunk.text}`
(llm.ts:107). It is **pre-existing** (truncateToContextSize present at v2.5.3 base),
i.e. the prior working corpus had the same behavior.

**Proven so far (all read-only, model = tokenizer only, no GPU forward pass):**
- Running `dist/` is current and contains `pushChunkWithinTokenLimit` + `CHUNK_SIZE_TOKENS = 900`.
- `chunkDocumentByTokens` (store.ts:2712) sizes with **embeddinggemma's own tokenizer**
  (`llm.tokenize` → `embedModel.tokenize`, llm.ts:1230) and is *supposed* to cap every
  chunk at 900 tokens (recursive re-split + a hard fallback at store.ts:2776–2788 that
  detokenizes `tokens.slice(0, 900)`).
- 50 largest docs: **0** chunks >2048 formatted. 250 random docs: **0**. Corpus is
  **100% `.md`** (17,239 active docs).
- Title scan (pure JS, all docs): only **1** doc has a title >4000 chars (10,177 chars:
  `Electric-furnace-steel-making-...WeLib-org-c8c20930.md`); 3 docs >500 chars.
- **Arithmetic proof the cap is failing:** with a true 900-token chunk cap, `formatted`
  can exceed 2048 only when the title alone exceeds ~1138 tokens (~4000 chars) — and only
  **1 doc** qualifies. A 20–30k-chunk flood therefore *cannot* be title-driven; the
  cluster must contain docs where `chunkDocumentByTokens` returns chunks **>900 tokens**,
  defeating its own cap. Random sampling missed the cluster (localized ~hundreds of docs
  in 17k).

**Hypothesis to confirm:** a specific markdown structure defeats the recursive
token-split — most likely a near-unsplittable token-dense span where `chunkDocument`
returns subchunks that don't shrink and the fallback clamp doesn't engage as expected.

**Next steps (pick up here after context clear):**
1. Read the full-corpus cap-failure scan result. It was launched as a background Bash:
   script `probe-cap-fail.ts`, task id `b1nag178b`, output at
   `/private/tmp/claude-501/.../tasks/b1nag178b.output` (re-run the script if the temp
   file is gone — it's read-only, ~10–15 min, model = tokenizer only). It lists every
   doc with a chunk whose `chunk.tokens > 900`, worst-first.
2. Open ONE named offender, find the markdown span that produces the oversize chunk,
   and trace why `pushChunkWithinTokenLimit` (store.ts:2741) fails to clamp it to 900.
3. Fix is **fork-side, fingerprint-changing** (so bundle into a future final rebuild,
   not the current run): either fix the re-split/fallback so no chunk exceeds 900, or
   add a hard pre-embed token clamp at chunk time. Also consider clamping `title` length
   in `formatDocForEmbedding`/`extractTitle` (kills the 1 giant-title doc cheaply).
4. Throwaway probe scripts left in repo root (untracked, safe to delete after the scan):
   `probe-cap-fail.ts`, `probe-chunk-tokenlen.ts`, `probe-title-len.ts`, `caperr.log`,
   `err2.log`. They all open `createStore($HOME/.cache/qmd/index.sqlite)` read-only.

**Decision already made:** do NOT stop the current run for this. It reproduces the
known-good quality of the prior corpus; the proper fix is a deliberate, separate
rebuild.
