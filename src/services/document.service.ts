import { z } from 'zod';
import { ClinicalDocument, type DocumentSource, type StoredTableRow } from '../models/Document';
import { Consultation } from '../models/Consultation';
import { chunkText, ChunkingError, getChunkOptions, normalizeText } from '../rag/chunking';
import { getEmbeddings, EmbeddingError } from '../rag/embeddings';
import { getVectorStore, PATIENT_DOMAIN, VectorStoreError, type MetadataValue, type VectorMatch, type VectorRecord } from '../rag/vectorStore';
import {
  assembleEvidence,
  formatPatientEvidence,
  formatReferenceEvidence,
  inlineCitationIds,
  stripInvalidInlineCitations,
  toCitation,
  validateCitations,
  type Citation,
  type Evidence,
  type EvidenceInput,
} from '../rag/evidence';
import { labRowToText, type LabTableRow } from '../documents/tables';
import { generateStructured, type NormalizedDocument, type ProviderName } from './ai.service';
import { retrieveReferenceEvidence } from './reference.service';
import { explainTerminology } from './terminologyAnswer.service';
import { classifyQuestion, type QuestionKind } from '../rag/routing';
import { MEDICAL_QA_PROMPT, DATA_HANDLING_RULES, escapeForPrompt, renderPrompt } from '../prompts';
import { recordFailure, type FailureCategory } from '../utils/failures';
import { logger, errorMeta } from '../utils/logger';
import { StageTimer } from '../utils/timing';
import { sha256 } from '../utils/hash';

export { sha256 };

/* ───────────────────────── Config ───────────────────────── */

const envNumber = (name: string, fallback: number, min: number, max: number): number => {
  const value = Number(process.env[name]);
  return Number.isFinite(value) && value >= min && value <= max ? value : fallback;
};

/**
 * Patient-document retrieval parameters. The similarity threshold is specific to the
 * embedding space (local bge-small, cosine); it is configurable and set from the
 * evaluation harness (`npm run eval -- --embedder=hf`), not by intuition.
 */
export const getRagConfig = () => ({
  topK: Math.round(envNumber('RAG_DOC_TOP_K', 5, 1, 10)),
  // A RELEVANCE FLOOR, not an answerability test. Measured with bge-small on a real-format 12-page lab
  // report and 14 natural questions: answerable questions scored 0.58-0.76, an in-domain but absent
  // fact ("blood group") scored 0.70, and unrelated questions scored 0.43. Similarity cannot tell "related
  // but absent" from "present", so the floor (0.50) only removes unrelated text; the grounded model
  // (insufficient_context) plus server-side citation validation decide whether the answer exists.
  // An earlier 0.68 (tuned on a toy synthetic set) wrongly rejected 6 of the 10 real answerable questions.
  minScore: envNumber('RAG_DOC_MIN_SCORE', 0.5, -1, 1),
  maxContextChars: Math.round(envNumber('RAG_MAX_CONTEXT_CHARS', 8000, 500, 50_000)),
  maxChunksPerDocument: Math.round(envNumber('RAG_MAX_CHUNKS_PER_DOCUMENT', 200, 1, 5000)),
  maxQuestionChars: 500,
  tableRowsPerChunk: Math.round(envNumber('RAG_TABLE_ROWS_PER_CHUNK', 8, 1, 50)),
});

/** Each user's vectors live in their own namespace (first isolation layer). */
export const userNamespace = (userId: string): string => `user-${userId}`;
export const vectorId = (documentId: string, chunkIndex: number): string => `${documentId}#${chunkIndex}`;

/* ───────────────────────── Indexing ───────────────────────── */

export type ChunkContentType = 'text' | 'table' | 'transcription';

export interface IndexSegment {
  text: string;
  /** 1-based page number, only when genuinely known (PDF text layer). */
  page?: number;
  contentType?: ChunkContentType;
  tableId?: string;
}

export interface IndexDocumentInput {
  userId: string;
  filename: string;
  mimeType: string;
  sizeBytes: number;
  sha256: string;
  source: DocumentSource;
  segments: IndexSegment[];
  tables?: StoredTableRow[];
  pages?: { page: number; status: string; imageCount: number; tableCount: number }[];
  notes?: string[];
}

export interface IndexResult {
  documentId?: string;
  filename: string;
  status: 'indexed' | 'failed';
  chunkCount: number;
  duplicate?: boolean;
  truncated?: boolean;
  failureCategory?: FailureCategory;
}

export interface PreparedChunk {
  id: string;
  chunkIndex: number;
  text: string;
  page?: number;
  contentType: ChunkContentType;
  tableId?: string;
}

/**
 * Deterministically chunks each segment. Text is split with overlap; table segments
 * are pre-grouped whole rows and are never split. Chunks never span pages.
 */
export const prepareChunks = (
  documentId: string,
  segments: IndexSegment[],
  maxChunks = getRagConfig().maxChunksPerDocument
): { chunks: PreparedChunk[]; truncated: boolean } => {
  const chunks: PreparedChunk[] = [];
  const options = getChunkOptions();
  for (const segment of segments) {
    const contentType = segment.contentType ?? 'text';
    const pieces = contentType === 'table' ? [normalizeText(segment.text)] : chunkText(segment.text, options).map((c) => c.text);
    for (const text of pieces) {
      if (!text) continue;
      if (chunks.length >= maxChunks) return { chunks, truncated: true };
      chunks.push({
        id: vectorId(documentId, chunks.length),
        chunkIndex: chunks.length,
        text,
        contentType,
        ...(segment.page !== undefined ? { page: segment.page } : {}),
        ...(segment.tableId ? { tableId: segment.tableId } : {}),
      });
    }
  }
  return { chunks, truncated: false };
};

/** Metadata stored with each vector: everything needed for secure retrieval + citation. */
export const buildChunkMetadata = (
  input: Pick<IndexDocumentInput, 'userId' | 'source' | 'mimeType'>,
  documentId: string,
  chunk: Pick<PreparedChunk, 'id' | 'chunkIndex' | 'text' | 'page'> & Partial<Pick<PreparedChunk, 'contentType' | 'tableId'>>,
  embeddingSpace = getEmbeddings().spaceId
): Record<string, MetadataValue> => ({
  domain: PATIENT_DOMAIN,
  userId: input.userId,
  documentId,
  chunkId: chunk.id,
  chunkIndex: chunk.chunkIndex,
  source: input.source,
  mimeType: input.mimeType,
  contentType: chunk.contentType ?? 'text',
  embeddingSpace,
  text: chunk.text,
  ...(chunk.page !== undefined ? { page: chunk.page } : {}),
  ...(chunk.tableId ? { tableId: chunk.tableId } : {}),
});

/** Groups table rows (per table) into row-intact segments with the table's page. */
export const tableSegments = (rows: LabTableRow[], rowsPerChunk = getRagConfig().tableRowsPerChunk): IndexSegment[] => {
  const byTable = new Map<string, LabTableRow[]>();
  for (const row of rows) byTable.set(row.tableId, [...(byTable.get(row.tableId) ?? []), row]);
  const segments: IndexSegment[] = [];
  for (const [tableId, tableRows] of byTable) {
    for (let i = 0; i < tableRows.length; i += rowsPerChunk) {
      const group = tableRows.slice(i, i + rowsPerChunk);
      segments.push({
        text: `Lab results table (${tableId})\n\n${group.map(labRowToText).join('\n\n')}`,
        contentType: 'table',
        tableId,
        ...(group[0].page !== undefined ? { page: group[0].page } : {}),
      });
    }
  }
  return segments;
};

/**
 * Ingestion: text → normalisation → chunking → local HF embedding → per-user
 * vector namespace → status. Never throws for pipeline failures; returns a failed
 * result with a category instead, so indexing never breaks the extraction workflow.
 */
export const indexDocument = async (input: IndexDocumentInput, timer = new StageTimer()): Promise<IndexResult> => {
  const store = getVectorStore();
  const embeddings = getEmbeddings();
  const base = { filename: input.filename };

  const duplicate = await ClinicalDocument.findOne({
    userId: input.userId,
    sha256: input.sha256,
    source: input.source,
    status: 'indexed',
    embeddingModel: embeddings.spaceId,
  })
    .lean()
    .exec();
  if (duplicate) {
    return { ...base, documentId: String(duplicate._id), status: 'indexed', chunkCount: duplicate.chunkCount, duplicate: true };
  }

  const segments = input.segments
    .map((s) => ({ ...s, text: normalizeText(s.text) }))
    .filter((s) => s.text.length > 0);
  const textChars = segments.reduce((sum, s) => sum + s.text.length, 0);
  const pageCount = new Set(segments.filter((s) => s.page !== undefined).map((s) => s.page)).size;

  const record = await ClinicalDocument.create({
    userId: input.userId,
    filename: input.filename,
    mimeType: input.mimeType,
    sizeBytes: input.sizeBytes,
    sha256: input.sha256,
    source: input.source,
    status: textChars === 0 ? 'failed' : 'indexing',
    textChars,
    ...(pageCount > 0 ? { pageCount } : {}),
    embeddingModel: embeddings.spaceId,
    tables: (input.tables ?? []).slice(0, 500),
    pages: input.pages ?? [],
    notes: (input.notes ?? []).slice(0, 50),
    ...(textChars === 0 ? { failure: { category: 'no_extractable_text', occurredAt: new Date() } } : {}),
  });
  const documentId = String(record._id);

  if (textChars === 0) {
    recordFailure('no_extractable_text', { operation: 'index_document', documentId });
    return { ...base, documentId, status: 'failed', chunkCount: 0, failureCategory: 'no_extractable_text' };
  }

  let written: string[] = [];
  try {
    const { chunks, truncated } = await timer.time('chunking', async () => prepareChunks(documentId, segments));
    if (chunks.length === 0) throw new ChunkingError('No chunks produced');

    const vectors = await timer.time('embedding', () => embeddings.embedDocuments(chunks.map((c) => c.text)));
    const records: VectorRecord[] = chunks.map((chunk, i) => ({
      id: chunk.id,
      values: vectors[i],
      metadata: buildChunkMetadata(input, documentId, chunk, embeddings.spaceId),
    }));

    written = records.map((r) => r.id);
    await timer.time('vector_upsert', () => store.upsert(userNamespace(input.userId), records));

    await ClinicalDocument.findOneAndUpdate(
      { _id: documentId, userId: input.userId },
      { $set: { status: 'indexed', chunkCount: chunks.length, truncated } }
    ).exec();

    logger.info('document.indexed', { documentId, chunkCount: chunks.length, truncated, source: input.source, ...timer.toLogMeta() });
    return { ...base, documentId, status: 'indexed', chunkCount: chunks.length, ...(truncated ? { truncated } : {}) };
  } catch (error) {
    const category: FailureCategory =
      error instanceof VectorStoreError || error instanceof EmbeddingError
        ? error.category
        : error instanceof ChunkingError
          ? 'chunking_failure'
          : 'vector_db_failure';
    recordFailure(category, { operation: 'index_document', documentId });
    logger.error('document.index_failed', { documentId, category, ...errorMeta(error) });

    await ClinicalDocument.findOneAndUpdate(
      { _id: documentId, userId: input.userId },
      { $set: { status: 'failed', failure: { category, occurredAt: new Date() } } }
    )
      .exec()
      .catch(() => undefined);
    if (written.length > 0) await store.deleteIds(userNamespace(input.userId), written).catch(() => undefined);

    return { ...base, documentId, status: 'failed', chunkCount: 0, failureCategory: category };
  }
};

/* ───────────────────────── Ownership / lifecycle ───────────────────────── */

export const listDocuments = (userId: string, consultationId?: string) =>
  ClinicalDocument.find({ userId, ...(consultationId ? { consultationId } : {}) })
    .sort({ createdAt: -1 })
    .lean()
    .exec();

/** Links the caller's own documents to a consultation; ids the caller does not own are ignored. */
export const linkDocumentsToConsultation = async (userId: string, consultationId: string, documentIds: string[]): Promise<string[]> => {
  if (documentIds.length === 0) return [];
  const owned = await ClinicalDocument.find({ _id: { $in: documentIds }, userId }).select('_id').lean().exec();
  const ownedIds = owned.map((d) => String(d._id));
  if (ownedIds.length !== documentIds.length) {
    recordFailure('invalid_metadata', { operation: 'link_documents', dropped: documentIds.length - ownedIds.length });
  }
  if (ownedIds.length > 0) {
    await ClinicalDocument.updateMany({ _id: { $in: ownedIds }, userId }, { $set: { consultationId } }).exec();
  }
  return ownedIds;
};

/** Deletes the record (authoritative) then its vectors (best effort — orphans are never retrievable). */
export const deleteDocument = async (userId: string, documentId: string): Promise<boolean> => {
  const deleted = await ClinicalDocument.findOneAndDelete({ _id: documentId, userId }).lean().exec();
  if (!deleted) return false;
  const ids = Array.from({ length: deleted.chunkCount ?? 0 }, (_, i) => vectorId(documentId, i));
  try {
    await getVectorStore().deleteIds(userNamespace(userId), ids);
  } catch (error) {
    recordFailure('vector_db_failure', { operation: 'delete_document_vectors', documentId });
    logger.warn('document.vector_delete_failed', { documentId, ...errorMeta(error) });
  }
  logger.info('document.deleted', { documentId });
  return true;
};

export const deleteDocumentsForConsultation = async (userId: string, consultationId: string): Promise<number> => {
  const docs = await ClinicalDocument.find({ userId, consultationId }).select('_id').lean().exec();
  let count = 0;
  for (const doc of docs) if (await deleteDocument(userId, String(doc._id))) count++;
  return count;
};

/* ───────────────────────── Patient retrieval ───────────────────────── */

export class RetrievalError extends Error {
  constructor(
    message: string,
    readonly category: FailureCategory
  ) {
    super(message);
    this.name = 'RetrievalError';
  }
}

export interface RetrievedChunk {
  documentId: string;
  chunkId: string;
  chunkIndex: number;
  page?: number;
  source: string;
  contentType: string;
  tableId?: string;
  filename?: string;
  score: number;
  text: string;
}

export type RetrievalStatus = 'ok' | 'no_documents' | 'no_matches' | 'below_threshold';

export interface RetrievalResult {
  status: RetrievalStatus;
  chunks: RetrievedChunk[];
  /** Matches returned by the vector store before threshold filtering. */
  candidates: number;
  /** All valid candidates with scores (for evaluation), before threshold/budget. */
  scored: { chunkId: string; documentId: string; score: number }[];
}

export interface RetrieveOptions {
  userId: string;
  question: string;
  documentIds?: string[];
  consultationId?: string;
  topK?: number;
  minScore?: number;
}

/** Validates vector metadata; anything malformed, not owned, or from another vector space is discarded. */
const toChunk = (match: VectorMatch, userId: string, allowed: Map<string, string | undefined>, spaceId: string): RetrievedChunk | null => {
  const m = match.metadata ?? {};
  if (
    m.domain !== PATIENT_DOMAIN ||
    m.userId !== userId ||
    m.embeddingSpace !== spaceId ||
    typeof m.documentId !== 'string' ||
    !allowed.has(m.documentId) ||
    typeof m.chunkId !== 'string' ||
    m.chunkId !== match.id ||
    typeof m.chunkIndex !== 'number' ||
    typeof m.text !== 'string' ||
    !m.text.trim()
  ) {
    return null;
  }
  return {
    documentId: m.documentId,
    chunkId: m.chunkId,
    chunkIndex: m.chunkIndex,
    ...(typeof m.page === 'number' ? { page: m.page } : {}),
    source: typeof m.source === 'string' ? m.source : 'unknown',
    contentType: typeof m.contentType === 'string' ? m.contentType : 'text',
    ...(typeof m.tableId === 'string' ? { tableId: m.tableId } : {}),
    filename: allowed.get(m.documentId),
    score: match.score,
    text: m.text,
  };
};

/**
 * Tenant-isolated retrieval. Independent layers:
 *  1. per-user namespace, 2. metadata filter on domain + userId + active documentIds,
 *  3. post-query validation against the caller's document records in MongoDB
 *     (drops vectors of deleted/failed documents and of other embedding spaces).
 */
export const retrieveDocumentChunks = async (options: RetrieveOptions, timer = new StageTimer()): Promise<RetrievalResult> => {
  const config = getRagConfig();
  const topK = Math.max(1, Math.min(10, options.topK ?? config.topK));
  const minScore = options.minScore ?? config.minScore;

  const docs = await ClinicalDocument.find({
    userId: options.userId,
    status: 'indexed',
    embeddingModel: getEmbeddings().spaceId,
    ...(options.documentIds?.length ? { _id: { $in: options.documentIds } } : {}),
    ...(options.consultationId ? { consultationId: options.consultationId } : {}),
  })
    .select('_id filename')
    .lean()
    .exec();
  if (docs.length === 0) return { status: 'no_documents', chunks: [], candidates: 0, scored: [] };

  const allowed = new Map(docs.map((d) => [String(d._id), d.filename]));
  return retrieveFromVectorStore({ userId: options.userId, question: options.question, allowed, topK, minScore }, timer);
};

/**
 * Vector-side retrieval core (namespace + metadata filter + metadata validation +
 * threshold + context budget). `allowed` maps the caller's verified documentIds to
 * filenames. Exported so the evaluation harness measures exactly this ranking logic.
 */
export const retrieveFromVectorStore = async (
  params: { userId: string; question: string; allowed: Map<string, string | undefined>; topK: number; minScore: number },
  timer = new StageTimer()
): Promise<RetrievalResult> => {
  const config = getRagConfig();
  const { allowed, topK, minScore, userId } = params;
  const empty = (status: RetrievalStatus, candidates = 0): RetrievalResult => ({ status, chunks: [], candidates, scored: [] });
  if (allowed.size === 0) return empty('no_documents');

  const embeddings = getEmbeddings();
  const store = getVectorStore();
  let matches: VectorMatch[];
  try {
    const vector = await timer.time('embedding', () => embeddings.embedQuery(params.question.slice(0, config.maxQuestionChars)));
    matches = await timer.time('retrieval', () =>
      store.query(userNamespace(userId), {
        vector,
        topK,
        filter: { domain: { $eq: PATIENT_DOMAIN }, userId: { $eq: userId }, documentId: { $in: [...allowed.keys()] } },
      })
    );
  } catch (error) {
    const category: FailureCategory = error instanceof VectorStoreError || error instanceof EmbeddingError ? error.category : 'vector_db_failure';
    recordFailure(category, { operation: 'retrieve_document_chunks' });
    throw new RetrievalError('Document retrieval is temporarily unavailable', category);
  }

  const valid: RetrievedChunk[] = [];
  for (const match of matches) {
    const chunk = toChunk(match, userId, allowed, embeddings.spaceId);
    if (chunk) valid.push(chunk);
    else recordFailure('invalid_metadata', { operation: 'retrieve_document_chunks' });
  }
  valid.sort((a, b) => b.score - a.score || a.chunkId.localeCompare(b.chunkId));
  const scored = valid.map((c) => ({ chunkId: c.chunkId, documentId: c.documentId, score: c.score }));

  if (valid.length === 0) {
    recordFailure('retrieval_miss', { operation: 'retrieve_document_chunks' });
    return { ...empty('no_matches', matches.length), scored };
  }

  const relevant = valid.filter((c) => c.score >= minScore);
  if (relevant.length === 0) {
    recordFailure('insufficient_context', { operation: 'retrieve_document_chunks', reason: 'below_threshold' });
    return { status: 'below_threshold', chunks: [], candidates: matches.length, scored };
  }

  // Context budget: keep the highest-scoring chunks that fit.
  const chunks: RetrievedChunk[] = [];
  let used = 0;
  for (const chunk of relevant) {
    if (used + chunk.text.length > config.maxContextChars && chunks.length > 0) break;
    chunks.push(chunk);
    used += chunk.text.length;
  }
  return { status: 'ok', chunks, candidates: matches.length, scored };
};

/** The caller's indexed documents in the current embedding space (ownership + vector-space gate). */
export const getAllowedDocuments = async (
  userId: string,
  scope: { documentIds?: string[]; consultationId?: string } = {}
): Promise<Map<string, string | undefined>> => {
  const docs = await ClinicalDocument.find({
    userId,
    status: 'indexed',
    embeddingModel: getEmbeddings().spaceId,
    ...(scope.documentIds?.length ? { _id: { $in: scope.documentIds } } : {}),
    ...(scope.consultationId ? { consultationId: scope.consultationId } : {}),
  })
    .select('_id filename')
    .lean()
    .exec();
  return new Map(docs.map((d) => [String(d._id), d.filename]));
};

/**
 * Multi-query patient retrieval over an already-authorised document set; results are
 * de-duplicated by chunk (best score kept). Used by the risk assessment.
 */
export const retrievePatientEvidence = async (
  userId: string,
  queries: string[],
  allowed: Map<string, string | undefined>,
  options: { topK?: number; minScore?: number } = {},
  timer = new StageTimer()
): Promise<{ evidence: EvidenceInput[]; status: RetrievalStatus }> => {
  if (allowed.size === 0 || queries.length === 0) return { evidence: [], status: 'no_documents' };
  const config = getRagConfig();
  const best = new Map<string, RetrievedChunk>();
  let anyCandidates = false;
  for (const question of queries) {
    const result = await retrieveFromVectorStore(
      { userId, question, allowed, topK: options.topK ?? 3, minScore: options.minScore ?? config.minScore },
      timer
    );
    anyCandidates ||= result.scored.length > 0;
    for (const chunk of result.chunks) {
      const previous = best.get(chunk.chunkId);
      if (!previous || previous.score < chunk.score) best.set(chunk.chunkId, chunk);
    }
  }
  const evidence = [...best.values()].map(chunkToEvidence);
  return { evidence, status: evidence.length ? 'ok' : anyCandidates ? 'below_threshold' : 'no_matches' };
};

export const chunkToEvidence = (c: RetrievedChunk): EvidenceInput => ({
  chunkId: c.chunkId,
  similarity: c.score,
  content: c.text,
  documentId: c.documentId,
  contentType: c.contentType,
  ...(c.filename ? { filename: c.filename } : {}),
  ...(c.page !== undefined ? { page: c.page } : {}),
  ...(c.tableId ? { tableId: c.tableId } : {}),
});

/* ───────────────────────── Grounded medical Q&A (patient + verified reference) ───────────────────────── */

export const INSUFFICIENT_CONTEXT_ANSWER = 'The available evidence is insufficient to answer this question.';
export const NO_REFERENCE_UNCERTAINTY =
  'This question asks for a clinical interpretation, which requires a verified medical reference. No verified reference supported an answer, so none is given. Please ask your clinician.';

const AnswerSchema = z.object({
  answer: z.string().max(4000),
  patientCitations: z.array(z.string().max(10)).max(20),
  referenceCitations: z.array(z.string().max(10)).max(20),
  uncertainty: z.string().max(1000),
  insufficientContext: z.boolean(),
});

export interface DocumentAnswer {
  status: 'answered' | 'insufficient_context';
  /** How the question was routed: a term definition, a clinical interpretation, or a lookup in the report. */
  kind: QuestionKind;
  answer: string;
  uncertainty?: string;
  /** Built only from retrieved metadata — never from model-supplied locations. */
  citations: Citation[];
  /** The verified medical references actually cited (subset of citations). */
  referencesUsed: Citation[];
  /** The medical terminology entries actually cited (terminology route only). */
  terminologyUsed: Citation[];
  retrieval: {
    status: RetrievalStatus;
    candidates: number;
    used: number;
    referenceStatus: 'ok' | 'no_sources' | 'below_threshold' | 'unavailable';
    referencesRetrieved: number;
    terminologyRetrieved?: number;
  };
  model?: { provider: ProviderName; model: string; fallbackUsed: boolean };
}

export const buildMedicalQAPrompt = (question: string, evidence: Evidence[]) =>
  renderPrompt(MEDICAL_QA_PROMPT, {
    data_rules: DATA_HANDLING_RULES,
    user_query: escapeForPrompt(question),
    patient_evidence: formatPatientEvidence(evidence),
    medical_evidence: formatReferenceEvidence(evidence),
  });

/**
 * Answers from the supplied evidence with ONE model call. Citations are validated
 * against the evidence ids; invented ids are dropped and an answer that cannot be
 * attributed to any supplied evidence is replaced by an explicit insufficiency.
 */
export const answerFromEvidence = async (
  question: string,
  evidence: Evidence[],
  retrievalInfo: DocumentAnswer['retrieval'],
  timer = new StageTimer(),
  kind: QuestionKind = 'report'
): Promise<DocumentAnswer> => {
  const insufficient = (model?: DocumentAnswer['model'], uncertainty?: string): DocumentAnswer => ({
    status: 'insufficient_context',
    kind,
    answer: INSUFFICIENT_CONTEXT_ANSWER,
    ...(uncertainty ? { uncertainty } : {}),
    citations: [],
    referencesUsed: [],
    terminologyUsed: [],
    retrieval: retrievalInfo,
    ...(model ? { model } : {}),
  });
  if (evidence.length === 0) return insufficient();
  // A clinical interpretation must rest on verified reference evidence; without any there is nothing to ground it.
  if (kind === 'clinical' && !evidence.some((e) => e.domain === 'reference')) {
    recordFailure('insufficient_context', { operation: 'answer_medical_question', reason: 'no_verified_reference' });
    return insufficient(undefined, NO_REFERENCE_UNCERTAINTY);
  }

  const { system, prompt } = await buildMedicalQAPrompt(question, evidence);
  const result = await timer.time('generation', () =>
    generateStructured({ operation: 'answer_medical_question', schema: AnswerSchema, system, prompt, temperature: 0.1, postValidate: (o) => o })
  );
  const output = result.value;
  const model = { provider: result.provider, model: result.model, fallbackUsed: result.fallbackUsed };

  // Structured ids AND ids written inline in the prose count as citations, each checked against its own domain
  // (P# only patient evidence, R# only verified references). Terminology/finding ids are not valid here.
  const inline = inlineCitationIds(output.answer);
  const patient = validateCitations([...output.patientCitations, ...inline.filter((id) => id.startsWith('P'))], evidence, 'patient');
  const reference = validateCitations([...output.referenceCitations, ...inline.filter((id) => id.startsWith('R'))], evidence, 'reference');
  const invalid = [...patient.invalid, ...reference.invalid, ...inline.filter((id) => !/^[PR]/.test(id))];
  if (invalid.length > 0) recordFailure('citation_failure', { operation: 'answer_medical_question', unknownCitations: invalid.length });

  if (output.insufficientContext) {
    recordFailure('insufficient_context', { operation: 'answer_medical_question', reason: 'model_reported' });
    return insufficient(model, output.uncertainty.trim() || undefined);
  }
  const cited = [...patient.valid, ...reference.valid];
  if (cited.length === 0) {
    recordFailure('citation_failure', { operation: 'answer_medical_question', reason: 'uncited_answer' });
    return insufficient(model);
  }
  if (kind === 'clinical' && reference.valid.length === 0) {
    recordFailure('citation_failure', { operation: 'answer_medical_question', reason: 'clinical_without_reference' });
    return insufficient(model, NO_REFERENCE_UNCERTAINTY);
  }
  const citations = cited.map(toCitation);
  const allowed = new Set(cited.map((e) => e.evidenceId));
  return {
    status: 'answered',
    kind,
    answer: stripInvalidInlineCitations(output.answer, allowed).trim(),
    ...(output.uncertainty.trim() ? { uncertainty: output.uncertainty.trim() } : {}),
    citations,
    referencesUsed: citations.filter((c) => c.domain === 'reference'),
    terminologyUsed: [],
    retrieval: retrievalInfo,
    model,
  };
};

/**
 * Query pipeline: local HF query embedding → patient vectors (isolated) + verified
 * reference vectors (shared) → thresholds/metadata validation → evidence budget →
 * one LangChain-prompted model call → validated, cited answer.
 */
const flagText = (l: { flag?: string; isAbnormal?: boolean }): string => l.flag ?? (l.isAbnormal ? 'abnormal' : 'normal');
const pagesText = (s?: { page?: number; pages?: number[] }): string =>
  s?.page !== undefined ? ` [p.${s.page}]` : s?.pages?.length ? ` [pp.${s.pages.join(', ')}]` : '';

/**
 * The consultation's structured report as one compact evidence item, so broad questions ("what is
 * abnormal?", "what did the doctor conclude?") can be answered even when no single chunk is similar
 * to the question. Contains no patient name. Built only from data extracted from the user's own document.
 */
export const buildStructuredEvidence = (c: {
  _id: unknown;
  patientDetails?: { age?: number; gender?: string };
  labValues?: { name: string; value: string; unit?: string; normalRange?: string; flag?: string; isAbnormal?: boolean; source?: { page?: number; pages?: number[] } }[];
  symptoms?: string[];
  medicines?: string[];
  riskLevel?: string;
  riskScore?: number;
  report?: { diagnosesMentioned?: string[]; reportRiskScores?: { name: string; result: string }[]; measurements?: { name: string; value: string; unit?: string }[]; dates?: string[] };
  assessment?: { keyFindings?: { finding: string }[] };
}): EvidenceInput | null => {
  const labs = c.labValues ?? [];
  const out = labs.filter((l) => (l.flag ? l.flag !== 'Normal' : l.isAbnormal));
  const inRange = labs.filter((l) => !out.includes(l));
  const lines: string[] = ['STRUCTURED REPORT extracted from the uploaded document'];
  const demo = [c.patientDetails?.age ? `${c.patientDetails.age} years` : '', c.patientDetails?.gender ?? ''].filter(Boolean).join(', ');
  if (demo) lines.push(`Patient: ${demo}`);
  if (labs.length) lines.push(`Lab values outside the range printed in the report (${out.length} of ${labs.length}):`);
  for (const l of out) lines.push(`- ${l.name}: ${l.value}${l.unit ? ` ${l.unit}` : ''}${l.normalRange ? ` (printed range ${l.normalRange})` : ''} — ${flagText(l)}${pagesText(l.source)}`);
  if (inRange.length) lines.push(`Lab values within range: ${inRange.map((l) => `${l.name} ${l.value}${l.unit ? ` ${l.unit}` : ''}`).join('; ')}`);
  if (c.report?.diagnosesMentioned?.length) lines.push(`Impressions written in the report: ${c.report.diagnosesMentioned.join('; ')}`);
  if (c.report?.reportRiskScores?.length) lines.push(`Risk scores printed in the report: ${c.report.reportRiskScores.map((r) => `${r.name}: ${r.result}`).join('; ')}`);
  if (c.report?.measurements?.length) lines.push(`Measurements: ${c.report.measurements.map((m) => `${m.name} ${m.value}${m.unit ? ` ${m.unit}` : ''}`).join('; ')}`);
  if (c.report?.dates?.length) lines.push(`Report dates: ${c.report.dates.join('; ')}`);
  if (c.symptoms?.length) lines.push(`Symptoms: ${c.symptoms.join('; ')}`);
  if (c.medicines?.length) lines.push(`Medicines: ${c.medicines.join('; ')}`);
  if (c.riskLevel) {
    const findings = (c.assessment?.keyFindings ?? []).map((k) => k.finding).join('; ');
    lines.push(`AyuNidan's own AI assessment (for clinician review, not part of the original report): ${c.riskLevel}${c.riskScore !== undefined ? ` (score ${c.riskScore})` : ''}${findings ? `. Key findings: ${findings}` : ''}`);
  }
  if (lines.length <= 1) return null;
  return {
    chunkId: `structured:${String(c._id)}`,
    similarity: 1,
    content: lines.join('\n').slice(0, 6000),
    filename: 'Structured report (extracted from your document)',
    contentType: 'structured_report',
  };
};

export const askDocuments = async (options: RetrieveOptions, timer = new StageTimer()): Promise<DocumentAnswer> => {
  let route = classifyQuestion(options.question);

  // Definition questions ("What does HbA1c mean?") are answered from the medical terminology domain only:
  // no patient data is needed, and terminology is never presented as a clinical reference.
  if (route.kind === 'terminology') {
    const answer = await explainTerminology(options.question, timer);
    // "What is <something specific to this report>?" — nothing in the terminology domain matched at all, so it is
    // a lookup in the patient's own documents after all.
    if (answer.status === 'insufficient_context' && answer.retrieval.retrieved === 0) {
      route = { kind: 'report' };
    } else {
      return {
        status: answer.status,
        kind: 'terminology',
        answer: answer.answer,
        ...(answer.uncertainty ? { uncertainty: answer.uncertainty } : {}),
        citations: answer.citations,
        referencesUsed: [],
        terminologyUsed: answer.citations,
        retrieval: {
          status: 'no_matches',
          candidates: 0,
          used: 0,
          referenceStatus: 'no_sources',
          referencesRetrieved: 0,
          terminologyRetrieved: answer.retrieval.retrieved,
        },
        ...(answer.model ? { model: answer.model } : {}),
      };
    }
  }

  const patient = await retrieveDocumentChunks(options, timer);

  // Questions scoped to a consultation also see its structured report (owner-checked).
  let structured: EvidenceInput | null = null;
  if (options.consultationId) {
    const consultation = await Consultation.findOne({ _id: options.consultationId, userId: options.userId }).lean().exec();
    if (consultation) structured = buildStructuredEvidence(consultation as unknown as Parameters<typeof buildStructuredEvidence>[0]);
  }

  let referenceStatus: DocumentAnswer['retrieval']['referenceStatus'] = 'no_sources';
  let referenceEvidence: EvidenceInput[] = [];
  try {
    const reference = await retrieveReferenceEvidence([options.question], {}, timer);
    referenceStatus = reference.status;
    referenceEvidence = reference.evidence;
  } catch (error) {
    // Reference KB outage degrades to patient-only evidence, stated explicitly in the response.
    referenceStatus = 'unavailable';
    logger.warn('document.reference_retrieval_failed', errorMeta(error));
  }

  const evidence = assembleEvidence([...(structured ? [structured] : []), ...patient.chunks.map(chunkToEvidence)], referenceEvidence);
  const retrievalInfo = {
    status: patient.status,
    candidates: patient.candidates,
    used: evidence.filter((e) => e.domain === 'patient').length,
    referenceStatus,
    referencesRetrieved: evidence.filter((e) => e.domain === 'reference').length,
  };
  return answerFromEvidence(options.question, evidence, retrievalInfo, timer, route.kind);
};

/* ───────────────────────── Upload integration ───────────────────────── */

const stripTableLines = (text: string): string =>
  text
    .split('\n')
    .filter((line) => (line.match(/\|/g)?.length ?? 0) < 2)
    .join('\n');

const toStoredRows = (rows: LabTableRow[]): StoredTableRow[] =>
  rows.map((r) => ({
    test: r.test,
    result: r.result,
    ...(r.unit ? { unit: r.unit } : {}),
    ...(r.referenceRange ? { referenceRange: r.referenceRange } : {}),
    ...(r.flag ? { flag: r.flag } : {}),
    ...(r.date ? { date: r.date } : {}),
    ...(r.page !== undefined ? { page: r.page } : {}),
    tableId: r.tableId,
    rowIndex: r.rowIndex,
  }));

/**
 * Indexes the normalised documents of one upload into the caller's patient namespace:
 * - PDFs with a text layer / text files → one document each: text chunks per page plus
 *   row-intact table chunks (with page + table provenance).
 * - Content only the model could read (images, scanned pages, audio) → one document
 *   built from the model's transcription (source "ai_transcription", no invented pages).
 * Never throws: failures are returned per document with a category.
 */
export const indexUploadedFiles = async (
  userId: string,
  files: Express.Multer.File[],
  documents: NormalizedDocument[],
  transcription: string,
  timer = new StageTimer()
): Promise<IndexResult[]> => {
  const results: IndexResult[] = [];
  const safeIndex = async (input: IndexDocumentInput): Promise<void> => {
    try {
      results.push(await indexDocument(input, timer));
    } catch (error) {
      recordFailure('persistence_failure', { operation: 'index_document' });
      logger.error('document.index_record_failed', errorMeta(error));
      results.push({ filename: input.filename, status: 'failed', chunkCount: 0, failureCategory: 'persistence_failure' });
    }
  };

  const modelRead: Express.Multer.File[] = [];
  for (const doc of documents) {
    const file = files[doc.fileIndex];
    if (!file) continue;
    const tableRows = doc.tables.flatMap((t) => t.labRows);
    const textSegments: IndexSegment[] =
      doc.sourceType === 'pdf'
        ? doc.pages.filter((p) => p.text.trim()).map((p) => ({ text: stripTableLines(p.text), page: p.page, contentType: 'text' as const }))
        : doc.sourceType === 'text' && doc.text
          ? [{ text: tableRows.length ? stripTableLines(doc.text) : doc.text, contentType: 'text' as const }]
          : [];
    const segments = [...textSegments, ...tableSegments(tableRows)];
    if (doc.needsModelReading) modelRead.push(file);
    if (segments.length === 0) continue;

    await safeIndex({
      userId,
      filename: doc.filename,
      mimeType: doc.mimeType,
      sizeBytes: file.size,
      sha256: sha256(file.buffer),
      source: doc.sourceType === 'text' ? 'text_file' : 'pdf_text',
      segments,
      tables: toStoredRows(tableRows),
      pages: doc.pages.map((p) => ({ page: p.page, status: p.status, imageCount: p.imageCount, tableCount: p.tables.length })),
      notes: doc.notes,
    });
  }

  if (modelRead.length > 0 && transcription.trim()) {
    const single = modelRead.length === 1 ? modelRead[0] : undefined;
    await safeIndex({
      userId,
      filename: single ? single.originalname : `${modelRead.length} files (AI transcription)`,
      mimeType: single ? single.mimetype : 'multipart/mixed',
      sizeBytes: modelRead.reduce((sum, f) => sum + f.size, 0),
      sha256: sha256(Buffer.concat(modelRead.map((f) => f.buffer))),
      source: 'ai_transcription',
      segments: [{ text: transcription, contentType: 'transcription' }],
      notes: ['Text transcribed by the multimodal model from content without a local text layer; not independently verified.'],
    });
  }
  return results;
};
