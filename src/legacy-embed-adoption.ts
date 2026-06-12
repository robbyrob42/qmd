// =============================================================================
// Legacy embedding adoption (fork addition)
// =============================================================================
//
// Background: QMD identifies stored embeddings by a (model, embed_fingerprint)
// pair. `getPendingEmbeddingDocs` only treats a document as embedded when its
// vectors match the CURRENT pair. When the default embed-model URI changed from
// the short alias "embeddinggemma" to the fully-qualified Hugging Face URI
// ("hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf") and
// fingerprinting was introduced, every previously-embedded vector kept the old
// `model` string and an empty fingerprint. Those rows stopped matching, so an
// entire corpus reads as "pending" and triggers a full, multi-hour re-embed —
// even though the underlying model file, document formatting, and chunk sizing
// are unchanged, so the vectors are bit-for-bit reproducible.
//
// This module isolates the fork-owned pieces needed to ADOPT (relabel) those
// legacy vectors instead of recomputing them. Keeping them in a separate file
// minimizes the diff footprint on upstream-tracked files, reducing merge
// conflicts when pulling QMD updates. The actual safety-gated adoption flow
// (re-embed one sample, confirm it reproduces the stored vector within a tight
// cosine-distance threshold, then relabel) lives in store.ts alongside the
// embedding helpers it depends on; this module provides the pure, LLM-free
// building blocks that flow can compose and that tests can exercise directly.

import type { Database } from "./db.js";
// Import the model constant from llm.ts (its source of truth) rather than from
// store.ts. store.ts imports THIS module, so importing back from store.ts would
// create a circular dependency; llm.ts has no dependency on this module.
import { DEFAULT_EMBED_MODEL_URI as DEFAULT_EMBED_MODEL } from "./llm.js";

// Known prior `model` identifiers for the current default embed model. Maps a
// current model URI to the historical strings that referred to the same model
// file. Add an entry here if a future rename orphans another generation of
// vectors. Keep this conservative: only list aliases that are genuinely the
// same model+quantization, because adoption relabels without recomputing.
export const LEGACY_EMBED_MODEL_ALIASES: Record<string, string[]> = {
  [DEFAULT_EMBED_MODEL]: ["embeddinggemma"],
};

/**
 * Return the subset of known legacy aliases for `currentModel` that actually
 * have orphaned (empty-fingerprint) vectors present in the database. Pure DB
 * read — no LLM. Used to decide which candidate model strings are worth running
 * the (expensive) sample-verification step against.
 *
 * @param db - Open database handle
 * @param currentModel - The current/default embed model URI
 * @returns Alias model strings that have at least one empty-fingerprint row
 */
export function legacyAliasesWithEmptyFingerprint(db: Database, currentModel: string): string[] {
  const aliases = LEGACY_EMBED_MODEL_ALIASES[currentModel] ?? [];
  const probe = db.prepare(
    `SELECT 1 FROM content_vectors WHERE model = ? AND embed_fingerprint = '' LIMIT 1`,
  );
  return aliases.filter((alias) => !!probe.get(alias));
}

/**
 * Relabel orphaned legacy vectors onto the current (model, fingerprint) pair.
 *
 * Only rows stored under `fromModel` with an EMPTY fingerprint are touched —
 * these are the orphaned legacy vectors. Rows already carrying a fingerprint
 * (current or otherwise) are left untouched, so this is safe to call even when
 * `fromModel === toModel`. content_vectors' primary key is (hash, seq), so each
 * chunk has exactly one row and relabeling can never create a duplicate.
 *
 * This performs NO verification that the vectors are semantically valid — the
 * caller is responsible for gating on a sample-reproduction check first. It is
 * isolated here purely so the SQL effect is unit-testable without an LLM.
 *
 * @param db - Open database handle
 * @param fromModel - The stored (legacy) model string to migrate away from
 * @param toModel - The current model string to migrate to
 * @param toFingerprint - The current embedding fingerprint to stamp on the rows
 * @returns Number of content_vectors rows relabeled
 */
export function relabelLegacyVectors(
  db: Database,
  fromModel: string,
  toModel: string,
  toFingerprint: string,
): number {
  const result = db
    .prepare(
      `UPDATE content_vectors SET model = ?, embed_fingerprint = ? WHERE model = ? AND embed_fingerprint = ''`,
    )
    .run(toModel, toFingerprint, fromModel);
  return result.changes;
}
