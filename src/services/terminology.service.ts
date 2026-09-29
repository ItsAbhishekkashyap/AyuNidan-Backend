import { getEmbeddings, EmbeddingError, type EmbeddingModel } from '../rag/embeddings';
import { getVectorStore, TERMINOLOGY_DOMAIN, TERMINOLOGY_NAMESPACE, VectorStoreError, type MetadataValue, type VectorMatch, type VectorRecord, type VectorStore } from '../rag/vectorStore';
import type { EvidenceInput, TerminologyMatchType } from '../rag/evidence';
import { normalizeTermKey } from '../terminology/text';
import type { DatasetInfo, TerminologyRecord } from '../terminology/types';
import { sha256 } from '../utils/hash';
import { recordFailure } from '../utils/failures';
import { StageTimer } from '../utils/timing';

/**
 * Medical TERMINOLOGY knowledge domain: definitions and synonyms from the official NLM datasets
 * (MedlinePlus Health Topics, MeSH descriptors), embedded LOCALLY (bge-small) into their own
 * namespace. This domain says what a term means. It is not a clinical guideline and is never used
 * to justify a risk level.
 *
 *   ingestion:  parsed records → embedding text (term + synonyms + start of definition) → vectors +
 *               provenance metadata (deterministic ids → re-ingestion is idempotent)
 *   retrieval:  EXACT term/synonym match (metadata key filter) first, then SEMANTIC matches above a
 *               conservative similarity floor; every hit carries dataset provenance and match type.
 */

/* ───────────────────────── Configuration ───────────────────────── */

const envNumber = (name: string, fallback: number, min: number, max: number): number => {
  const value = Number(process.env[name]);
  return Number.isFinite(value) && value >= min && value <= max ? value : fallback;
};

export const getTerminologyConfig = () => ({
  topK: Math.round(envNumber('RAG_TERM_TOP_K', 4, 1, 10)),
  // Semantic floor for bge-small cosine. Measured with `npm run terminology:eval -- --mesh-limit=2000` (3,014 real records):
  // semantic-only definition questions scored 0.73-0.74, off-topic questions peaked at 0.59. 0.70 sits in that gap,
  // erring toward "no match". Exact term/synonym matches bypass the floor. Re-measure after any dataset refresh.
  minScore: envNumber('RAG_TERM_MIN_SCORE', 0.7, -1, 1),
  maxKeys: 4,
});

const EMBED_DEFINITION_CHARS = 700;
const MAX_ALIAS_KEYS = 60;
const MAX_ALIASES_STORED = 30;

/* ───────────────────────── Ingestion ───────────────────────── */

/** Text that is embedded: the term, its synonyms and the start of its definition. */
export const buildEmbeddingText = (record: TerminologyRecord): string => {
  const aliases = record.aliases.slice(0, 6);
  const head = aliases.length ? `${record.term} (also called ${aliases.join(', ')})` : record.term;
  return `${head}: ${record.definition.slice(0, EMBED_DEFINITION_CHARS)}`;
};

export const buildAliasKeys = (record: Pick<TerminologyRecord, 'term' | 'aliases'>): string[] =>
  [...new Set([record.term, ...record.aliases].map(normalizeTermKey).filter(Boolean))].slice(0, MAX_ALIAS_KEYS);

export const toVectorRecord = (record: TerminologyRecord, info: DatasetInfo, embeddingSpace: string, values: number[]): VectorRecord => {
  const metadata: Record<string, MetadataValue> = {
    domain: TERMINOLOGY_DOMAIN,
    chunkId: record.recordId,
    dataset: record.dataset,
    datasetName: info.sourceName,
    organization: info.organization,
    version: info.version,
    sourceFile: info.sourceFile,
    sourceRecordId: record.sourceRecordId,
    term: record.term,
    termKey: normalizeTermKey(record.term),
    aliasKeys: buildAliasKeys(record),
    aliases: record.aliases.slice(0, MAX_ALIASES_STORED),
    text: record.definition,
    contentHash: sha256(`${record.term}\n${record.aliases.join('|')}\n${record.definition}`).slice(0, 16),
    embeddingSpace,
    ...(info.publicationDate ? { publicationDate: info.publicationDate } : {}),
    ...(record.url ? { url: record.url } : {}),
    ...(record.meshDescriptor ? { meshId: record.meshDescriptor.id } : {}),
    ...(record.dataset === 'mesh' ? { meshId: record.sourceRecordId } : {}),
    ...(record.dateCreated ? { dateCreated: record.dateCreated } : {}),
    ...(record.dateRevised ? { dateRevised: record.dateRevised } : {}),
  };
  return { id: record.recordId, values, metadata };
};

export interface TerminologyIngestOptions {
  store?: VectorStore;
  embeddings?: EmbeddingModel;
  batchSize?: number;
  /** Ids written by the previous ingestion of the same dataset — those no longer present are deleted. */
  previousIds?: string[];
  onProgress?: (done: number, total: number) => void;
}

export interface TerminologyIngestResult {
  dataset: string;
  records: number;
  upserted: number;
  deletedStale: number;
  recordIds: string[];
  embeddingSpace: string;
}

/**
 * Embeds and upserts records in deterministic (id-sorted) order. Ids are dataset-native, so running it
 * again over the same records overwrites identical vectors — it never creates duplicates.
 */
export const ingestTerminologyRecords = async (info: DatasetInfo, records: TerminologyRecord[], options: TerminologyIngestOptions = {}): Promise<TerminologyIngestResult> => {
  const store = options.store ?? getVectorStore();
  const embeddings = options.embeddings ?? getEmbeddings();
  const batchSize = options.batchSize ?? 64;
  const unique = [...new Map(records.map((r) => [r.recordId, r])).values()].sort((a, b) => a.recordId.localeCompare(b.recordId));

  let upserted = 0;
  for (let i = 0; i < unique.length; i += batchSize) {
    const batch = unique.slice(i, i + batchSize);
    const vectors = await embeddings.embedDocuments(batch.map(buildEmbeddingText));
    await store.upsert(
      TERMINOLOGY_NAMESPACE,
      batch.map((record, j) => toVectorRecord(record, info, embeddings.spaceId, vectors[j]))
    );
    upserted += batch.length;
    options.onProgress?.(upserted, unique.length);
  }

  const current = new Set(unique.map((r) => r.recordId));
  const stale = (options.previousIds ?? []).filter((id) => !current.has(id));
  if (stale.length) await store.deleteIds(TERMINOLOGY_NAMESPACE, stale);

  return { dataset: info.dataset, records: unique.length, upserted, deletedStale: stale.length, recordIds: unique.map((r) => r.recordId), embeddingSpace: embeddings.spaceId };
};

/* ───────────────────────── Term extraction (deterministic) ───────────────────────── */

const LEADING_SCAFFOLD: RegExp[] = [
  /^(?:please\s+)?(?:can|could|would|will)\s+you\s+(?:please\s+)?/,
  /^(?:please\s+)?(?:tell\s+me|explain|describe|define|clarify)\s+(?:to\s+me\s+)?(?:what\s+|about\s+)?/,
  /^what(?:'s|s)?\s+(?:is|are|was|were)\s+(?:the\s+)?(?:meaning|definition)\s+of\s+/,
  /^what(?:'s|s)?\s+(?:is|are|was|were)\s+/,
  /^what\s+(?:does|do)\s+/,
  /^(?:the\s+)?(?:meaning|definition)\s+of\s+/,
  /^(?:an?|the)\s+/,
];
const TRAILING_SCAFFOLD: RegExp[] = [
  /\s+(?:mean|means|meaning)$/,
  /\s+(?:stand|stands)\s+for$/,
  /\s+in\s+(?:simple|plain)\s+(?:terms|words|english)$/,
  /\s+in\s+medical\s+terms$/,
  /\s+medically$/,
  /\s+(?:is|are)$/,
  /\s+(?:medical\s+)?(?:term|definition)$/,
];

/**
 * Definition-shaped question → the term being asked about ("What does HbA1c mean?" → "hba1c").
 * Returns null unless the question is a pure definition request.
 */
export const extractDefinitionTerm = (question: string): string | null => {
  let text = question.trim().toLowerCase().replace(/["“”‘’]/g, '').replace(/[?!.\s]+$/, '').replace(/\s+/g, ' ');
  if (!text) return null;
  const isDefinitional = /^(?:please\s+)?(?:can|could|would|will|tell|explain|describe|define|clarify|what|the\s+(?:meaning|definition)|meaning|definition)\b/.test(text) || /\s(?:mean|means|meaning)$/.test(text) || /\s+stands?\s+for$/.test(text);
  if (!isDefinitional) return null;
  // "what does <X> mean" is a definition; "what does the guideline say about X" / "what does X do" are not.
  if (/^what\s+(?:does|do)\b/.test(text) && !/\s(?:mean|means|stand|stands)(?:\s+for)?$/.test(text)) return null;
  for (let pass = 0; pass < 3; pass++) {
    const before = text;
    for (const pattern of LEADING_SCAFFOLD) text = text.replace(pattern, '');
    for (const pattern of TRAILING_SCAFFOLD) text = text.replace(pattern, '');
    if (text === before) break;
  }
  text = text.trim();
  const words = text.split(' ').filter(Boolean);
  if (words.length === 0 || words.length > 8 || text.length > 100) return null;
  // Bare scaffolding ("what is that?") leaves nothing meaningful.
  if (/^(?:it|that|this|these|those|there|here|more)$/.test(text)) return null;
  return text;
};

/** Lookup keys for an exact match: the cleaned phrase and, for compact phrases, its individual content. */
export const buildLookupKeys = (queryOrTerm: string): string[] => {
  const term = extractDefinitionTerm(queryOrTerm) ?? queryOrTerm.trim().toLowerCase().replace(/[?!.\s]+$/, '');
  const words = term.split(/\s+/).filter(Boolean);
  if (words.length === 0 || words.length > 8) return [];
  const keys = new Set<string>([normalizeTermKey(term)]);
  const withoutArticle = term.replace(/^(?:an?|the|my)\s+/, '');
  keys.add(normalizeTermKey(withoutArticle));
  // Trailing "test"/"level(s)" variants are common in lab language ("hba1c test" ↔ "hba1c").
  keys.add(normalizeTermKey(withoutArticle.replace(/\s+(?:test|tests|level|levels)$/, '')));
  return [...keys].filter(Boolean).slice(0, getTerminologyConfig().maxKeys);
};

/* ───────────────────────── Retrieval ───────────────────────── */

const str = (v: unknown): string | undefined => (typeof v === 'string' && v.trim() ? v : undefined);

const toTerminologyEvidence = (match: VectorMatch, spaceId: string, keys: string[]): (EvidenceInput & { rank: number }) | null => {
  const m = match.metadata ?? {};
  if (
    m.domain !== TERMINOLOGY_DOMAIN ||
    m.embeddingSpace !== spaceId ||
    str(m.chunkId) !== match.id ||
    !str(m.text) ||
    !str(m.term) ||
    !str(m.dataset) ||
    !str(m.datasetName) ||
    !str(m.organization)
  ) {
    return null;
  }
  const termKey = str(m.termKey);
  const aliasKeys = Array.isArray(m.aliasKeys) ? (m.aliasKeys as unknown[]).filter((k): k is string => typeof k === 'string') : [];
  let matchType: TerminologyMatchType = 'semantic';
  let rank = 2;
  if (termKey && keys.includes(termKey)) {
    matchType = 'exact_term';
    rank = 0;
  } else if (aliasKeys.some((k) => keys.includes(k))) {
    matchType = 'exact_alias';
    rank = 1;
  }
  return {
    rank,
    chunkId: match.id,
    similarity: match.score,
    content: m.text as string,
    sourceId: m.dataset as string,
    dataset: m.dataset as string,
    datasetName: m.datasetName as string,
    title: m.term as string,
    organization: m.organization as string,
    sourceType: 'terminology',
    matchType,
    ...(str(m.version) ? { version: m.version as string } : {}),
    ...(str(m.publicationDate) ? { publicationDate: m.publicationDate as string } : {}),
    ...(str(m.url) ? { url: m.url as string } : {}),
    ...(str(m.meshId) ? { meshId: m.meshId as string } : {}),
    ...(Array.isArray(m.aliases) ? { aliases: (m.aliases as unknown[]).filter((a): a is string => typeof a === 'string').slice(0, 12) } : {}),
  };
};

export interface TerminologyRetrieval {
  evidence: EvidenceInput[];
  status: 'ok' | 'no_match' | 'unavailable';
  exactMatches: number;
  semanticMatches: number;
  topScore?: number;
}

/**
 * Exact term/synonym matches first (by normalised key), then semantic matches that clear the floor.
 * Hits are validated (domain, embedding space, provenance fields) and de-duplicated by record id.
 */
export const retrieveTerminologyEvidence = async (
  query: string,
  options: { topK?: number; minScore?: number; store?: VectorStore; embeddings?: EmbeddingModel } = {},
  timer = new StageTimer()
): Promise<TerminologyRetrieval> => {
  const cleaned = query.trim().slice(0, 300);
  if (!cleaned) return { evidence: [], status: 'no_match', exactMatches: 0, semanticMatches: 0 };
  const config = getTerminologyConfig();
  const topK = options.topK ?? config.topK;
  const minScore = options.minScore ?? config.minScore;
  const embeddings = options.embeddings ?? getEmbeddings();
  const store = options.store ?? getVectorStore();
  const keys = buildLookupKeys(cleaned);

  try {
    const vector = await timer.time('terminology_embedding', () => embeddings.embedQuery(extractDefinitionTerm(cleaned) ?? cleaned));
    const [exact, semantic] = await timer.time('terminology_retrieval', () =>
      Promise.all([
        keys.length
          ? store.query(TERMINOLOGY_NAMESPACE, { vector, topK: 6, filter: { aliasKeys: { $in: keys }, embeddingSpace: { $eq: embeddings.spaceId } } })
          : Promise.resolve([] as VectorMatch[]),
        store.query(TERMINOLOGY_NAMESPACE, { vector, topK, filter: { embeddingSpace: { $eq: embeddings.spaceId } } }),
      ])
    );

    const best = new Map<string, EvidenceInput & { rank: number }>();
    let topScore = 0;
    for (const [matches, exactPass] of [
      [exact, true],
      [semantic, false],
    ] as const) {
      for (const match of matches) {
        const evidence = toTerminologyEvidence(match, embeddings.spaceId, keys);
        if (!evidence) {
          recordFailure('invalid_metadata', { operation: 'retrieve_terminology' });
          continue;
        }
        topScore = Math.max(topScore, match.score);
        const accepted = evidence.matchType !== 'semantic' || (!exactPass && match.score >= minScore);
        if (!accepted) continue;
        const previous = best.get(evidence.chunkId);
        if (!previous || previous.rank > evidence.rank || (previous.rank === evidence.rank && previous.similarity < evidence.similarity)) best.set(evidence.chunkId, evidence);
      }
    }
    // Order: exact term → exact synonym → semantic; MedlinePlus (consumer wording) before MeSH; then similarity.
    const datasetOrder = (d?: string) => (d === 'medlineplus' ? 0 : 1);
    const ordered = [...best.values()].sort(
      (a, b) => a.rank - b.rank || datasetOrder(a.dataset) - datasetOrder(b.dataset) || b.similarity - a.similarity || a.chunkId.localeCompare(b.chunkId)
    );
    const evidence: EvidenceInput[] = ordered.map(({ rank: _rank, ...rest }) => rest);
    if (evidence.length === 0) recordFailure('insufficient_context', { operation: 'retrieve_terminology', reason: 'no_match' });
    return {
      evidence,
      status: evidence.length ? 'ok' : 'no_match',
      exactMatches: ordered.filter((e) => e.rank < 2).length,
      semanticMatches: ordered.filter((e) => e.rank === 2).length,
      ...(topScore ? { topScore: Number(topScore.toFixed(4)) } : {}),
    };
  } catch (error) {
    if (error instanceof VectorStoreError || error instanceof EmbeddingError) recordFailure(error.category, { operation: 'retrieve_terminology' });
    throw error;
  }
};
