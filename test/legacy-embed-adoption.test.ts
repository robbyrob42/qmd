// Unit tests for the fork-owned legacy embedding adoption building blocks.
// These exercise the pure, LLM-free SQL helpers that the safety-gated adoption
// flow in store.ts composes. The full flow (sample re-embed + distance check)
// is covered by an LLM-gated integration test in store.test.ts.
import { describe, test, expect, beforeEach, afterEach } from "vitest";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { mkdtemp, rm } from "node:fs/promises";
import { openDatabase } from "../src/db.js";
import type { Database } from "../src/db.js";
import {
  createStore,
  getEmbeddingFingerprint,
  maybeAdoptLegacyEmbeddingFingerprint,
  DEFAULT_EMBED_MODEL,
  type Store,
} from "../src/store.js";
import { setDefaultLlamaCpp, type LlamaCpp } from "../src/llm.js";
import {
  LEGACY_EMBED_MODEL_ALIASES,
  legacyAliasesWithEmptyFingerprint,
  relabelLegacyVectors,
} from "../src/legacy-embed-adoption.js";

// Minimal content_vectors table — these helpers only touch this table.
function createContentVectorsTable(db: Database): void {
  db.exec(`
    CREATE TABLE content_vectors (
      hash TEXT NOT NULL,
      seq INTEGER NOT NULL DEFAULT 0,
      pos INTEGER NOT NULL DEFAULT 0,
      model TEXT NOT NULL,
      embed_fingerprint TEXT NOT NULL DEFAULT '',
      embedded_at TEXT NOT NULL,
      total_chunks INTEGER NOT NULL DEFAULT 1,
      PRIMARY KEY (hash, seq)
    )
  `);
}

function insertVector(
  db: Database,
  hash: string,
  seq: number,
  model: string,
  fingerprint: string,
): void {
  db.prepare(
    `INSERT INTO content_vectors (hash, seq, pos, model, embed_fingerprint, embedded_at)
     VALUES (?, ?, 0, ?, ?, '2026-01-01T00:00:00Z')`,
  ).run(hash, seq, model, fingerprint);
}

function rowsFor(db: Database, model: string, fingerprint: string): number {
  return (
    db
      .prepare(
        `SELECT COUNT(*) AS n FROM content_vectors WHERE model = ? AND embed_fingerprint = ?`,
      )
      .get(model, fingerprint) as { n: number }
  ).n;
}

describe("legacy embedding adoption helpers", () => {
  let db: Database;

  beforeEach(() => {
    db = openDatabase(":memory:");
    createContentVectorsTable(db);
  });

  afterEach(() => {
    db.close();
  });

  describe("LEGACY_EMBED_MODEL_ALIASES", () => {
    test("maps the current default model to the historical 'embeddinggemma' alias", () => {
      expect(LEGACY_EMBED_MODEL_ALIASES[DEFAULT_EMBED_MODEL]).toContain("embeddinggemma");
    });
  });

  describe("legacyAliasesWithEmptyFingerprint", () => {
    test("returns aliases that have orphaned empty-fingerprint rows", () => {
      insertVector(db, "h1", 0, "embeddinggemma", "");
      const found = legacyAliasesWithEmptyFingerprint(db, DEFAULT_EMBED_MODEL);
      expect(found).toContain("embeddinggemma");
    });

    test("returns empty when the alias has no empty-fingerprint rows", () => {
      // Alias rows exist but already carry a fingerprint → nothing to adopt.
      insertVector(db, "h1", 0, "embeddinggemma", "abc123");
      expect(legacyAliasesWithEmptyFingerprint(db, DEFAULT_EMBED_MODEL)).toEqual([]);
    });

    test("returns empty when there are no alias rows at all", () => {
      insertVector(db, "h1", 0, DEFAULT_EMBED_MODEL, "abc123");
      expect(legacyAliasesWithEmptyFingerprint(db, DEFAULT_EMBED_MODEL)).toEqual([]);
    });

    test("returns empty for a model with no known aliases", () => {
      insertVector(db, "h1", 0, "some-other-model", "");
      expect(legacyAliasesWithEmptyFingerprint(db, "some-other-model")).toEqual([]);
    });
  });

  describe("relabelLegacyVectors", () => {
    test("relabels only empty-fingerprint rows of the source model", () => {
      insertVector(db, "h1", 0, "embeddinggemma", "");
      insertVector(db, "h1", 1, "embeddinggemma", "");
      insertVector(db, "h2", 0, "embeddinggemma", "");

      const changed = relabelLegacyVectors(db, "embeddinggemma", DEFAULT_EMBED_MODEL, "c37385");

      expect(changed).toBe(3);
      expect(rowsFor(db, DEFAULT_EMBED_MODEL, "c37385")).toBe(3);
      expect(rowsFor(db, "embeddinggemma", "")).toBe(0);
    });

    test("leaves rows under the source model that already have a fingerprint untouched", () => {
      insertVector(db, "h1", 0, "embeddinggemma", ""); // orphaned → adopt
      insertVector(db, "h2", 0, "embeddinggemma", "old99"); // already labeled → keep

      const changed = relabelLegacyVectors(db, "embeddinggemma", DEFAULT_EMBED_MODEL, "c37385");

      expect(changed).toBe(1);
      expect(rowsFor(db, "embeddinggemma", "old99")).toBe(1);
      expect(rowsFor(db, DEFAULT_EMBED_MODEL, "c37385")).toBe(1);
    });

    test("does not touch rows belonging to other models", () => {
      insertVector(db, "h1", 0, "embeddinggemma", "");
      insertVector(db, "h2", 0, "unrelated-model", "");

      const changed = relabelLegacyVectors(db, "embeddinggemma", DEFAULT_EMBED_MODEL, "c37385");

      expect(changed).toBe(1);
      expect(rowsFor(db, "unrelated-model", "")).toBe(1);
    });

    test("is a no-op when there are no matching empty-fingerprint rows", () => {
      insertVector(db, "h1", 0, DEFAULT_EMBED_MODEL, "c37385");
      const changed = relabelLegacyVectors(db, "embeddinggemma", DEFAULT_EMBED_MODEL, "c37385");
      expect(changed).toBe(0);
      expect(rowsFor(db, DEFAULT_EMBED_MODEL, "c37385")).toBe(1);
    });

    test("same-model relabel stamps the fingerprint onto empty-fingerprint rows", () => {
      // Mirrors the original same-model adoption case (empty fp → current fp).
      insertVector(db, "h1", 0, DEFAULT_EMBED_MODEL, "");
      const changed = relabelLegacyVectors(db, DEFAULT_EMBED_MODEL, DEFAULT_EMBED_MODEL, "c37385");
      expect(changed).toBe(1);
      expect(rowsFor(db, DEFAULT_EMBED_MODEL, "c37385")).toBe(1);
      expect(rowsFor(db, DEFAULT_EMBED_MODEL, "")).toBe(0);
    });
  });
});

// =============================================================================
// End-to-end adoption flow (hermetic — fake LLM, no model download)
// =============================================================================
//
// Exercises maybeAdoptLegacyEmbeddingFingerprint over the model-alias path: a
// corpus stored under the legacy "embeddinggemma" model string with empty
// fingerprints is verified (one sample re-embedded reproduces its stored
// vector) and bulk-relabeled onto the current (model, fingerprint) pair WITHOUT
// recomputing. A stub LlamaCpp returns a deterministic vector so the test is
// fast and offline; the relabel itself is the real production code path.

// Small fixed embedding dimension keeps the test cheap and readable.
const DIM = 8;

// A stub LlamaCpp returning a fixed embedding regardless of input. tokenize()
// returns one id per whitespace-separated word so chunkDocumentByTokens yields a
// single chunk (seq 0) for the short fixtures below.
function makeFakeLlm(vector: Float32Array): LlamaCpp {
  return {
    embedModelName: DEFAULT_EMBED_MODEL,
    async embed(_text: string) {
      return { embedding: Array.from(vector), model: DEFAULT_EMBED_MODEL };
    },
    async embedBatch(texts: string[]) {
      return texts.map(() => ({ embedding: Array.from(vector), model: DEFAULT_EMBED_MODEL }));
    },
    async tokenize(text: string) {
      return text.split(/\s+/).filter(Boolean).map((_, i) => i);
    },
  } as unknown as LlamaCpp;
}

function seedDoc(db: Database, hash: string, path: string, body: string): void {
  const now = "2026-01-01T00:00:00Z";
  db.prepare(`INSERT OR IGNORE INTO content (hash, doc, created_at) VALUES (?, ?, ?)`).run(hash, body, now);
  db.prepare(
    `INSERT INTO documents (collection, path, title, hash, created_at, modified_at, active)
     VALUES ('docs', ?, ?, ?, ?, ?, 1)`,
  ).run(path, path, hash, now, now);
}

function seedLegacyVector(db: Database, hash: string, vec: Float32Array): void {
  db.prepare(
    `INSERT INTO content_vectors (hash, seq, pos, model, embed_fingerprint, embedded_at, total_chunks)
     VALUES (?, 0, 0, 'embeddinggemma', '', '2026-01-01T00:00:00Z', 1)`,
  ).run(hash);
  db.prepare(`INSERT INTO vectors_vec (hash_seq, embedding) VALUES (?, ?)`).run(`${hash}_0`, vec);
}

describe("maybeAdoptLegacyEmbeddingFingerprint (alias path, hermetic)", () => {
  let testDir: string;
  let store: Store;

  beforeEach(async () => {
    testDir = await mkdtemp(join(tmpdir(), "qmd-legacy-adopt-"));
    store = createStore(join(testDir, "index.sqlite"));
    store.ensureVecTable(DIM);
  });

  afterEach(async () => {
    setDefaultLlamaCpp(null);
    store.close();
    await rm(testDir, { recursive: true, force: true });
  });

  test("adopts legacy 'embeddinggemma' vectors onto the current model+fingerprint when a sample reproduces", async () => {
    // Sample vector the fake LLM will reproduce exactly (cosine distance 0).
    const sampleVec = Float32Array.from([1, 0, 0, 0, 0, 0, 0, 0]);
    // A second legacy doc with a different stored vector — proves the bulk
    // relabel covers the whole alias cohort after a single sample passes.
    const otherVec = Float32Array.from([0, 1, 0, 0, 0, 0, 0, 0]);

    // "aaa" sorts first, so the adoption query picks it as the sample.
    seedDoc(store.db, "aaa", "a.md", "alpha document body");
    seedDoc(store.db, "bbb", "b.md", "beta document body");
    seedLegacyVector(store.db, "aaa", sampleVec);
    seedLegacyVector(store.db, "bbb", otherVec);

    const fake = makeFakeLlm(sampleVec);
    setDefaultLlamaCpp(fake); // used by chunkDocumentByTokens
    store.llm = fake; // used by getLlm(store) for the sample embed

    const result = await maybeAdoptLegacyEmbeddingFingerprint(store, DEFAULT_EMBED_MODEL);

    const fingerprint = getEmbeddingFingerprint(DEFAULT_EMBED_MODEL);
    expect(result.adopted).toBe(2);
    expect(result.reason).toContain("model alias 'embeddinggemma'");

    // Both rows now carry the current (model, fingerprint) pair; none remain legacy.
    const adopted = (
      store.db
        .prepare(
          `SELECT COUNT(*) AS n FROM content_vectors WHERE model = ? AND embed_fingerprint = ?`,
        )
        .get(DEFAULT_EMBED_MODEL, fingerprint) as { n: number }
    ).n;
    const remainingLegacy = (
      store.db
        .prepare(`SELECT COUNT(*) AS n FROM content_vectors WHERE model = 'embeddinggemma'`)
        .get() as { n: number }
    ).n;
    expect(adopted).toBe(2);
    expect(remainingLegacy).toBe(0);
  });

  test("refuses to adopt when the sample does not reproduce (fails safe)", async () => {
    seedDoc(store.db, "aaa", "a.md", "alpha document body");
    // Stored vector points one way; the fake LLM will return a different vector.
    seedLegacyVector(store.db, "aaa", Float32Array.from([1, 0, 0, 0, 0, 0, 0, 0]));

    const fake = makeFakeLlm(Float32Array.from([0, 0, 0, 0, 0, 0, 0, 1]));
    setDefaultLlamaCpp(fake);
    store.llm = fake;

    const result = await maybeAdoptLegacyEmbeddingFingerprint(store, DEFAULT_EMBED_MODEL);

    expect(result.adopted).toBe(0);
    // Legacy rows untouched.
    const remainingLegacy = (
      store.db
        .prepare(`SELECT COUNT(*) AS n FROM content_vectors WHERE model = 'embeddinggemma' AND embed_fingerprint = ''`)
        .get() as { n: number }
    ).n;
    expect(remainingLegacy).toBe(1);
  });
});
