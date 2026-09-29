import path from 'node:path';
import { Document } from '@langchain/core/documents';
import { z } from 'zod';
import { ReferenceSource, REFERENCE_SOURCE_TYPES } from '../models/ReferenceSource';
import { analyzePdf } from '../documents/pdfAnalyzer';
import { detectDelimitedTables, labRowToText, type ExtractedTable } from '../documents/tables';
import { chunkText, getChunkOptions, normalizeText, ChunkingError } from '../rag/chunking';
import { getEmbeddings, EmbeddingError } from '../rag/embeddings';
import {
  getVectorStore,
  REFERENCE_DOMAIN,
  REFERENCE_NAMESPACE,
  VectorStoreError,
  type MetadataValue,
  type VectorMatch,
} from '../rag/vectorStore';
import type { EvidenceInput } from '../rag/evidence';
import { recordFailure, type FailureCategory } from '../utils/failures';
import { logger, errorMeta } from '../utils/logger';
import { StageTimer } from '../utils/timing';
import { sha256 } from '../utils/hash';

/* ───────────────────────── Source metadata (operator-supplied, never inferred) ───────────────────────── */

const optional = (max: number) => z.string().trim().min(1).max(max).optional();

export const ReferenceMetadataSchema = z
  .object({
    sourceId: z.string().regex(/^[a-z0-9][a-z0-9._-]{1,99}$/, 'sourceId must be a lowercase slug'),
    title: z.string().trim().min(1).max(500),
    organization: z.string().trim().min(1).max(300),
    sourceType: z.enum(REFERENCE_SOURCE_TYPES),
    /** Licence / permission basis for ingesting this document (required). */
    authorization: z.string().trim().min(3).max(1000),
    publicationDate: optional(50),
    version: optional(100),
    url: z.string().trim().url().max(2000).optional(),
    medicalTopics: z.array(z.string().trim().min(1).max(100)).max(20).default([]),
  })
  .strict();

export type ReferenceMetadata = z.infer<typeof ReferenceMetadataSchema>;

/* ───────────────────────── Loading + section-aware segmentation ───────────────────────── */

export interface ReferenceSegment {
  text: string;
  page?: number;
  section?: string;
  contentType: 'text' | 'table';
}

const HEADING = /^(#{1,6}\s+.+|(\d+(\.\d+)*\.?)\s+[A-Z][^.!?]{2,80}|[A-Z][A-Z0-9 ,/&()-]{3,80})$/;

/**
 * Guards the loose HEADING pattern against look-alikes found in real guideline PDFs: numbered sentence
 * fragments ("3. FCS is a rare monogenic disorder characterized by"), numbered bibliography entries
 * ("6. Schwartz GG, ... et al"), hyphen-broken lines and short table codes ("B-NR"). Markdown headings pass as-is.
 */
export const looksLikeHeading = (line: string): boolean => {
  if (/^#{1,6}\s+/.test(line)) return true;
  if (/-$/.test(line) || /\bet al\b/i.test(line) || /\S {2,}\S/.test(line)) return false; // double spaces = justified body text
  if (/^[A-Z][A-Z0-9 ,/&()-]+$/.test(line)) return line.length >= 8 && !/^(\S+) \1$/.test(line);
  const title = line.replace(/^\d+(\.\d+)*\.?\s+/, '');
  const words = title.split(/\s+/).filter((w) => /[A-Za-z]/.test(w));
  const capitalised = words.filter((w) => /^[A-Z(]/.test(w)).length;
  if (title.length < 6 || words.length === 0 || words.length > 12) return false; // "B-NR", "B-R" recommendation codes
  if (capitalised / words.length >= 0.6) return true; // Title Case (ACC/AHA)
  // Sentence case (WHO): short, and not a sentence fragment that stops on a connector word.
  return words.length <= 9 && /^[A-Z]/.test(title) && !/\b(a|an|the|and|or|of|in|on|to|by|for|with|is|are|that|which)$/i.test(title);
};

/** Splits text into sections by headings (markdown, numbered, or short ALL-CAPS lines). */
export const splitSections = (text: string, initialSection?: string): { section?: string; text: string }[] => {
  const out: { section?: string; text: string }[] = [];
  let section = initialSection;
  let buffer: string[] = [];
  const flush = () => {
    const body = buffer.join('\n').trim();
    if (body) out.push({ section, text: body });
    buffer = [];
  };
  for (const line of text.split('\n')) {
    const trimmed = line.trim();
    if (trimmed && trimmed.length <= 90 && HEADING.test(trimmed) && looksLikeHeading(trimmed) && !trimmed.includes('|')) {
      flush();
      section = trimmed.replace(/^#{1,6}\s+/, '');
    } else {
      buffer.push(line);
    }
  }
  flush();
  return out;
};

const tableSegments = (tables: ExtractedTable[], section?: string): ReferenceSegment[] =>
  tables.flatMap((t) =>
    t.labRows.length > 0
      ? t.labRows.map((row) => ({ text: labRowToText(row), page: row.page, section, contentType: 'table' as const }))
      : [{ text: [t.headers.join(' | '), ...t.rows.map((r) => r.join(' | '))].join('\n'), page: t.page, section, contentType: 'table' as const }]
  );

/** Removes table lines from free text so table content is indexed once, row-intact. */
const stripTableLines = (text: string): string =>
  text
    .split('\n')
    .filter((line) => (line.match(/\|/g)?.length ?? 0) < 2)
    .join('\n');

/**
 * Running headers/footers ("Circulation. 2026;153:e1154...", "Downloaded from ...", journal banners) repeat on
 * most pages, would be mistaken for section headings and pollute every chunk. Lines whose text (digits ignored)
 * occurs on >= 30% of the pages of a longer document are dropped. Body text is never affected: it does not repeat.
 */
export const removeRunningLines = (pageTexts: string[]): string[] => {
  if (pageTexts.length < 6) return pageTexts;
  const key = (line: string): string => line.replace(/\d+/g, '#').replace(/\s+/g, ' ').trim();
  const pageCount = new Map<string, number>();
  for (const text of pageTexts) for (const k of new Set(text.split('\n').map(key).filter(Boolean))) pageCount.set(k, (pageCount.get(k) ?? 0) + 1);
  const isRunning = (line: string): boolean => {
    const k = key(line);
    return k.length > 0 && k.length < 200 && ((pageCount.get(k) ?? 0) / pageTexts.length >= 0.3 || /^Downloaded from /i.test(k));
  };
  return pageTexts.map((text) => text.split('\n').filter((line) => !isRunning(line)).join('\n'));
};

export const loadReferenceSegments = async (buffer: Buffer, filename: string): Promise<{ segments: ReferenceSegment[]; pageCount?: number }> => {
  const ext = path.extname(filename).toLowerCase();
  if (ext === '.pdf') {
    const analysis = await analyzePdf(buffer);
    const segments: ReferenceSegment[] = [];
    let section: string | undefined;
    const cleaned = removeRunningLines(analysis.pages.map((p) => p.text));
    for (const [pageIndex, page] of analysis.pages.entries()) {
      if (page.status === 'scanned' || page.status === 'empty') {
        logger.warn('reference.page_without_text', { page: page.page, status: page.status });
        continue; // reference ingestion is text-only; such pages are reported, not guessed
      }
      for (const part of splitSections(stripTableLines(cleaned[pageIndex]), section)) {
        section = part.section;
        segments.push({ text: part.text, page: page.page, section: part.section, contentType: 'text' });
      }
      segments.push(...tableSegments(page.tables, section));
    }
    return { segments, pageCount: analysis.pageCount };
  }
  if (ext === '.txt' || ext === '.md') {
    const text = normalizeText(buffer.toString('utf8'));
    const segments: ReferenceSegment[] = [];
    for (const part of splitSections(stripTableLines(text))) segments.push({ text: part.text, section: part.section, contentType: 'text' });
    segments.push(...tableSegments(detectDelimitedTables(text, undefined, 'ref')));
    return { segments };
  }
  throw new ChunkingError(`Unsupported reference file type: ${ext}`);
};

/**
 * Bibliography text ("Vodnala D, ... Am J Cardiol. 2012;110:...") is not guidance: it names studies but states no
 * recommendation, yet it scores high for topical queries and could be cited as if it supported a claim.
 * A chunk with three or more journal citations (year;volume:pages) is treated as a reference list and not indexed.
 */
export const isCitationDense = (text: string): boolean => (text.match(/\b(?:19|20)\d{2};\s?\d+\s?(?::|\()/g) ?? []).length >= 3;

/** Converts segments into LangChain Documents (one per chunk) carrying full provenance. */
export const buildReferenceDocuments = (meta: ReferenceMetadata, segments: ReferenceSegment[], embeddingSpace: string): Document[] => {
  const options = getChunkOptions();
  const docs: Document[] = [];
  for (const segment of segments) {
    // Table rows are already atomic; never split them.
    const pieces = segment.contentType === 'table' ? [segment.text] : chunkText(segment.text, options).map((c) => c.text);
    for (const text of pieces) {
      if (segment.contentType === 'text' && isCitationDense(text)) continue;
      const chunkIndex = docs.length;
      const metadata: Record<string, MetadataValue> = {
        domain: REFERENCE_DOMAIN,
        sourceId: meta.sourceId,
        documentId: meta.sourceId,
        chunkId: `${meta.sourceId}#${chunkIndex}`,
        chunkIndex,
        title: meta.title,
        organization: meta.organization,
        sourceType: meta.sourceType,
        contentType: segment.contentType,
        embeddingSpace,
        text,
        ...(meta.publicationDate ? { publicationDate: meta.publicationDate } : {}),
        ...(meta.version ? { version: meta.version } : {}),
        ...(meta.url ? { url: meta.url } : {}),
        ...(meta.medicalTopics.length ? { medicalTopic: meta.medicalTopics[0], medicalTopics: meta.medicalTopics } : {}),
        ...(segment.page !== undefined ? { page: segment.page } : {}),
        ...(segment.section ? { section: segment.section } : {}),
      };
      docs.push(new Document({ pageContent: text, metadata }));
    }
  }
  return docs;
};

/* ───────────────────────── Ingestion ───────────────────────── */

export interface ReferenceIngestResult {
  sourceId: string;
  status: 'indexed' | 'unchanged' | 'failed';
  chunkCount: number;
  failureCategory?: FailureCategory;
}

/**
 * Reference document → loader → cleaning → section-aware chunking → provenance
 * metadata → local HF embedding → shared reference namespace → registry.
 * Re-ingesting identical content under the same embedding space is a no-op.
 */
export const ingestReferenceSource = async (
  rawMeta: unknown,
  buffer: Buffer,
  filename: string,
  timer = new StageTimer(),
  options: { force?: boolean } = {}
): Promise<ReferenceIngestResult> => {
  const meta = ReferenceMetadataSchema.parse(rawMeta);
  const embeddings = getEmbeddings();
  const hash = sha256(buffer);

  const existing = await ReferenceSource.findOne({ sourceId: meta.sourceId }).lean().exec();
  if (!options.force && existing && existing.sha256 === hash && existing.status === 'indexed' && existing.embeddingSpace === embeddings.spaceId) {
    return { sourceId: meta.sourceId, status: 'unchanged', chunkCount: existing.chunkCount };
  }

  const store = getVectorStore();
  try {
    const { segments, pageCount } = await timer.time('reference_load', () => loadReferenceSegments(buffer, filename));
    const docs = buildReferenceDocuments(meta, segments, embeddings.spaceId);
    if (docs.length === 0) throw new ChunkingError('No text could be extracted from the reference document');

    const vectors = await timer.time('embedding', () => embeddings.embedDocuments(docs.map((d) => d.pageContent)));
    // Replace any previous version's vectors for this source.
    if (existing?.chunkCount) {
      await store.deleteIds(REFERENCE_NAMESPACE, Array.from({ length: existing.chunkCount }, (_, i) => `${meta.sourceId}#${i}`));
    }
    await timer.time('vector_upsert', () =>
      store.upsert(
        REFERENCE_NAMESPACE,
        docs.map((d, i) => ({ id: d.metadata.chunkId as string, values: vectors[i], metadata: d.metadata as Record<string, MetadataValue> }))
      )
    );

    const record = {
      ...meta,
      filename: path.basename(filename),
      sha256: hash,
      status: 'indexed' as const,
      chunkCount: docs.length,
      ...(pageCount !== undefined ? { pageCount } : {}),
      embeddingSpace: embeddings.spaceId,
    };
    if (existing) await ReferenceSource.findOneAndUpdate({ sourceId: meta.sourceId }, { $set: record, $unset: { failure: 1 } }).exec();
    else await ReferenceSource.create(record);

    invalidateReferenceRegistry();
    logger.info('reference.indexed', { sourceId: meta.sourceId, chunkCount: docs.length, ...timer.toLogMeta() });
    return { sourceId: meta.sourceId, status: 'indexed', chunkCount: docs.length };
  } catch (error) {
    const category: FailureCategory =
      error instanceof EmbeddingError || error instanceof VectorStoreError
        ? error.category
        : error instanceof ChunkingError
          ? 'chunking_failure'
          : 'extraction_failure';
    recordFailure(category, { operation: 'ingest_reference', sourceId: meta.sourceId });
    logger.error('reference.ingest_failed', { sourceId: meta.sourceId, category, ...errorMeta(error) });
    return { sourceId: meta.sourceId, status: 'failed', chunkCount: 0, failureCategory: category };
  }
};

/* ───────────────────────── Registry (cached briefly) ───────────────────────── */

export interface ActiveReference {
  sourceId: string;
  embeddingSpace: string;
}

let registryCache: { at: number; sources: Map<string, ActiveReference> } | null = null;
const REGISTRY_TTL_MS = 60_000;

export const invalidateReferenceRegistry = (): void => {
  registryCache = null;
};

export const getActiveReferences = async (): Promise<Map<string, ActiveReference>> => {
  if (registryCache && Date.now() - registryCache.at < REGISTRY_TTL_MS) return registryCache.sources;
  const rows = await ReferenceSource.find({ status: 'indexed' }).select('sourceId embeddingSpace sourceType').lean().exec();
  // Synthetic test sources are never served as "verified" references unless explicitly enabled (tests/eval).
  const allowSynthetic = process.env.RAG_INCLUDE_SYNTHETIC_REFERENCES === 'true';
  const sources = new Map(
    rows
      .filter((r) => allowSynthetic || r.sourceType !== 'synthetic_test')
      .map((r) => [r.sourceId, { sourceId: r.sourceId, embeddingSpace: r.embeddingSpace }])
  );
  registryCache = { at: Date.now(), sources };
  return sources;
};

export const retireReferenceSource = async (sourceId: string): Promise<boolean> => {
  const updated = await ReferenceSource.findOneAndUpdate({ sourceId }, { $set: { status: 'retired' } }).exec();
  invalidateReferenceRegistry();
  return Boolean(updated);
};

/* ───────────────────────── Retrieval ───────────────────────── */

const envNumber = (name: string, fallback: number, min: number, max: number): number => {
  const value = Number(process.env[name]);
  return Number.isFinite(value) && value >= min && value <= max ? value : fallback;
};

/**
 * Reference retrieval parameters. The minimum similarity is specific to the embedding
 * space (bge-small cosine); it is configurable and evaluated by `npm run eval -- --embedder=hf`.
 */
export const getReferenceRetrievalConfig = () => ({
  topKPerQuery: Math.round(envNumber('RAG_REF_TOP_K', 4, 1, 20)),
  // Set from `npm run eval -- --embedder=hf` (bge-small, synthetic reference corpus): answerable
  // top-1 min 0.7673, unanswerable max 0.5608; 0.60-0.76 had zero FP/FN. 0.66 = near the gap midpoint.
  // Small synthetic sample — re-evaluate once real authorised references are ingested.
  minScore: envNumber('RAG_REF_MIN_SCORE', 0.66, -1, 1),
  maxQueries: Math.round(envNumber('RAG_REF_MAX_QUERIES', 6, 1, 20)),
});

const str = (v: unknown): string | undefined => (typeof v === 'string' && v.trim() ? v : undefined);

/** Validates metadata of a reference match; anything malformed or unregistered is dropped. */
const toReferenceEvidence = (match: VectorMatch, active: Map<string, ActiveReference>, spaceId: string): EvidenceInput | null => {
  const m = match.metadata ?? {};
  const sourceId = str(m.sourceId);
  const registered = sourceId ? active.get(sourceId) : undefined;
  if (
    m.domain !== REFERENCE_DOMAIN ||
    !registered ||
    registered.embeddingSpace !== spaceId ||
    m.embeddingSpace !== spaceId ||
    str(m.chunkId) !== match.id ||
    !str(m.text) ||
    !str(m.title) ||
    !str(m.organization)
  ) {
    return null;
  }
  return {
    chunkId: match.id,
    similarity: match.score,
    content: m.text as string,
    sourceId,
    documentId: sourceId,
    title: m.title as string,
    organization: m.organization as string,
    ...(str(m.publicationDate) ? { publicationDate: m.publicationDate as string } : {}),
    ...(str(m.version) ? { version: m.version as string } : {}),
    ...(str(m.url) ? { url: m.url as string } : {}),
    ...(str(m.section) ? { section: m.section as string } : {}),
    ...(str(m.medicalTopic) ? { medicalTopic: m.medicalTopic as string } : {}),
    ...(str(m.sourceType) ? { sourceType: m.sourceType as string } : {}),
    ...(str(m.contentType) ? { contentType: m.contentType as string } : {}),
    ...(typeof m.page === 'number' ? { page: m.page } : {}),
  };
};

export interface ReferenceRetrieval {
  evidence: EvidenceInput[];
  /** Best score per query (for evaluation / observability), before thresholding. */
  topScores: number[];
  status: 'ok' | 'no_sources' | 'below_threshold';
}

/**
 * Multi-query, multi-source reference retrieval: every query is embedded locally (in parallel), each retrieves topK chunks from the shared reference namespace, results are
 * validated against the registry, thresholded per chunk and de-duplicated. Chunks from
 * any number of different sources may be returned.
 */
export const retrieveReferenceEvidence = async (
  queries: string[],
  options: { topKPerQuery?: number; minScore?: number } = {},
  timer = new StageTimer()
): Promise<ReferenceRetrieval> => {
  const cleaned = queries.map((q) => q.trim()).filter(Boolean);
  if (cleaned.length === 0) return { evidence: [], topScores: [], status: 'no_sources' };
  const active = await getActiveReferences();
  return retrieveReferenceFromStore(cleaned, active, options, timer);
};

/** Vector-side reference retrieval over an explicit registry (exported for the evaluation harness). */
export const retrieveReferenceFromStore = async (
  queries: string[],
  active: Map<string, ActiveReference>,
  options: { topKPerQuery?: number; minScore?: number } = {},
  timer = new StageTimer()
): Promise<ReferenceRetrieval> => {
  const config = getReferenceRetrievalConfig();
  const topK = options.topKPerQuery ?? config.topKPerQuery;
  const minScore = options.minScore ?? config.minScore;
  const cleaned = [...new Set(queries.map((q) => q.trim().slice(0, 500)).filter(Boolean))].slice(0, config.maxQueries);
  if (cleaned.length === 0 || active.size === 0) return { evidence: [], topScores: [], status: 'no_sources' };

  const embeddings = getEmbeddings();
  const store = getVectorStore();
  const vectors = await timer.time('reference_embedding', () => Promise.all(cleaned.map((q) => embeddings.embedQuery(q))));
  const perQuery = await timer.time('reference_retrieval', () =>
    Promise.all(
      vectors.map((vector) =>
        store.query(REFERENCE_NAMESPACE, {
          vector,
          topK,
          filter: { domain: { $eq: REFERENCE_DOMAIN }, embeddingSpace: { $eq: embeddings.spaceId } },
        })
      )
    )
  );

  const best = new Map<string, EvidenceInput>();
  const topScores: number[] = [];
  for (const matches of perQuery) {
    topScores.push(matches[0]?.score ?? 0);
    for (const match of matches) {
      const evidence = toReferenceEvidence(match, active, embeddings.spaceId);
      if (!evidence) {
        recordFailure('invalid_metadata', { operation: 'retrieve_reference' });
        continue;
      }
      if (evidence.similarity < minScore) continue;
      const previous = best.get(evidence.chunkId);
      if (!previous || previous.similarity < evidence.similarity) best.set(evidence.chunkId, evidence);
    }
  }

  const evidence = [...best.values()].sort((a, b) => b.similarity - a.similarity || a.chunkId.localeCompare(b.chunkId));
  if (evidence.length === 0) recordFailure('insufficient_context', { operation: 'retrieve_reference', reason: 'below_threshold' });
  return { evidence, topScores, status: evidence.length ? 'ok' : 'below_threshold' };
};
