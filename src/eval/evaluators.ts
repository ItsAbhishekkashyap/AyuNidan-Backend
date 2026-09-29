import type { Dataset } from './dataset';
import { fuzzyMatch, hitAtK, mean, microPRF, normalize, normalizeGender, rate, reciprocalRank, round, setPRF, type PRF } from './metrics';
import { buildPdf, LAB_COLUMNS } from './pdfBuilder';
import {
  extractMedicalData,
  withModelFallback,
  OutputValidationError,
  AIProcessingError,
  AIConfigurationError,
  type ExtractionResult,
  type ModelCandidate,
} from '../services/ai.service';
import {
  answerFromEvidence,
  buildChunkMetadata,
  chunkToEvidence,
  prepareChunks,
  retrieveFromVectorStore,
  userNamespace,
  RetrievalError,
  type RetrievalResult,
} from '../services/document.service';
import { buildReferenceDocuments, loadReferenceSegments, retrieveReferenceFromStore, type ActiveReference } from '../services/reference.service';
import { assessRisk, RISK_LEVELS } from '../services/assessment.service';
import { assembleEvidence, validateCitations } from '../rag/evidence';
import { chunkText, ChunkingError } from '../rag/chunking';
import { getEmbeddings, EmbeddingError, HashingEmbeddings } from '../rag/embeddings';
import { getVectorStore, REFERENCE_NAMESPACE, VectorStoreError, type MetadataValue } from '../rag/vectorStore';
import { InMemoryVectorStore } from '../rag/inMemoryVectorStore';
import { analyzePdf } from '../documents/pdfAnalyzer';
import { detectDelimitedTables, type LabTableRow } from '../documents/tables';
import { validateAudio, AudioValidationError } from '../documents/audio';
import { type FailureCategory } from '../utils/failures';
import { percentile } from '../utils/metrics';

type ExtractionGold = Dataset['extraction'][number]['expected'];

export const categoryOf = (error: unknown): FailureCategory =>
  error instanceof AIProcessingError ||
  error instanceof AIConfigurationError ||
  error instanceof RetrievalError ||
  error instanceof VectorStoreError ||
  error instanceof EmbeddingError ||
  error instanceof AudioValidationError
    ? error.category
    : error instanceof ChunkingError
      ? 'chunking_failure'
      : 'extraction_failure';

const distribution = (values: number[]) =>
  values.length === 0
    ? undefined
    : {
        n: values.length,
        min: round(Math.min(...values), 4),
        p25: round(percentile(values, 25), 4),
        median: round(percentile(values, 50), 4),
        p75: round(percentile(values, 75), 4),
        max: round(Math.max(...values), 4),
      };

/* ───────────────────────── A. Extraction (live) ───────────────────────── */

export interface ExtractionCaseScore {
  id: string;
  nameMatch: boolean;
  ageMatch: boolean;
  genderMatch: boolean;
  symptoms: PRF;
  medicines: PRF;
  labNames: PRF;
  labValueMatches: boolean[];
  labFlagMatches: boolean[];
}

export const scoreExtraction = (id: string, predicted: ExtractionResult, gold: ExtractionGold): ExtractionCaseScore => {
  const labs = setPRF(predicted.labValues, gold.labValues, (p, g) => fuzzyMatch(p.name, g.name));
  const strip = ({ pairs: _pairs, ...rest }: PRF & { pairs: unknown }) => rest;
  return {
    id,
    nameMatch: normalize(predicted.patientDetails.name) === normalize(gold.name),
    ageMatch: (predicted.patientDetails.age ?? null) === gold.age,
    genderMatch: normalizeGender(predicted.patientDetails.gender ?? '') === normalizeGender(gold.gender),
    symptoms: strip(setPRF(predicted.symptoms, gold.symptoms, fuzzyMatch)),
    medicines: strip(setPRF(predicted.medicines, gold.medicines, fuzzyMatch)),
    labNames: strip(labs),
    labValueMatches: labs.pairs.map(([p, g]) => normalize(p.value).replace(/[^0-9.]/g, '') === normalize(g.value).replace(/[^0-9.]/g, '')),
    labFlagMatches: labs.pairs.map(([p, g]) => p.isAbnormal === g.isAbnormal),
  };
};

export interface ExtractionReport {
  model: string;
  cases: number;
  succeeded: number;
  failureRate: number;
  schemaValidationFailureRate: number;
  failureCategories: Record<string, number>;
  fieldAccuracy: { name?: number; age?: number; gender?: number; labValue?: number; labAbnormalFlag?: number };
  /** Undefined when no case succeeded (metrics over zero predictions are not meaningful). */
  symptoms?: PRF;
  medicines?: PRF;
  labNames?: PRF;
  latencyMs: { p50?: number; p95?: number };
  perCase: (ExtractionCaseScore | { id: string; failureCategory: FailureCategory })[];
}

export const aggregateExtraction = (
  model: string,
  scores: ExtractionCaseScore[],
  failures: { id: string; failureCategory: FailureCategory }[],
  latencies: number[]
): ExtractionReport => {
  const total = scores.length + failures.length;
  const failureCategories: Record<string, number> = {};
  for (const f of failures) failureCategories[f.failureCategory] = (failureCategories[f.failureCategory] ?? 0) + 1;
  const r = (v?: number) => round(v);
  const roundPRF = (p: PRF): PRF => ({ ...p, precision: r(p.precision)!, recall: r(p.recall)!, f1: r(p.f1)! });
  return {
    model,
    cases: total,
    succeeded: scores.length,
    failureRate: r(total === 0 ? 0 : failures.length / total)!,
    schemaValidationFailureRate: r(total === 0 ? 0 : (failureCategories.schema_validation_failure ?? 0) / total)!,
    failureCategories,
    fieldAccuracy: {
      name: r(rate(scores.map((s) => s.nameMatch))),
      age: r(rate(scores.map((s) => s.ageMatch))),
      gender: r(rate(scores.map((s) => s.genderMatch))),
      labValue: r(rate(scores.flatMap((s) => s.labValueMatches))),
      labAbnormalFlag: r(rate(scores.flatMap((s) => s.labFlagMatches))),
    },
    symptoms: scores.length ? roundPRF(microPRF(scores.map((s) => s.symptoms))) : undefined,
    medicines: scores.length ? roundPRF(microPRF(scores.map((s) => s.medicines))) : undefined,
    labNames: scores.length ? roundPRF(microPRF(scores.map((s) => s.labNames))) : undefined,
    latencyMs: { p50: percentile(latencies, 50), p95: percentile(latencies, 95) },
    perCase: [...scores, ...failures],
  };
};

/** Runs extraction for every case against ONE model (no fallback, for a fair comparison). */
export const runExtractionEval = async (dataset: Dataset, candidate: ModelCandidate): Promise<ExtractionReport> => {
  const scores: ExtractionCaseScore[] = [];
  const failures: { id: string; failureCategory: FailureCategory }[] = [];
  const latencies: number[] = [];
  for (const c of dataset.extraction) {
    const start = Date.now();
    try {
      const predicted = await extractMedicalData(c.input, [], { candidates: [candidate] });
      latencies.push(Date.now() - start);
      scores.push(scoreExtraction(c.id, predicted, c.expected));
    } catch (error) {
      failures.push({ id: c.id, failureCategory: categoryOf(error) });
    }
  }
  return aggregateExtraction(`${candidate.provider}:${candidate.modelId}`, scores, failures, latencies);
};

/* ───────────────────────── B. Table extraction (offline, deterministic) ───────────────────────── */

export interface TableReport {
  cases: number;
  rows: PRF;
  fieldAccuracy: { result?: number; unit?: number; referenceRange?: number; flag?: number; page?: number };
  perCase: { id: string; expected: number; extracted: number; matched: number }[];
}

export const runTableEval = async (dataset: Dataset): Promise<TableReport> => {
  const rowCounts: Pick<PRF, 'tp' | 'fp' | 'fn'>[] = [];
  const fields: Record<'result' | 'unit' | 'referenceRange' | 'flag' | 'page', boolean[]> = { result: [], unit: [], referenceRange: [], flag: [], page: [] };
  const perCase: TableReport['perCase'] = [];
  for (const c of dataset.tables) {
    let rows: LabTableRow[] = [];
    if (c.format === 'pdf') {
      const pages = Array.from({ length: c.page }, (_, i) =>
        i === c.page - 1 ? { lines: ['FICTIONAL REPORT'], table: { columns: LAB_COLUMNS, rows: [c.header, ...c.rows] } } : { lines: ['Fictional cover page'] }
      );
      const analysis = await analyzePdf(buildPdf(pages));
      rows = analysis.pages.flatMap((p) => p.tables.flatMap((t) => t.labRows));
    } else {
      rows = detectDelimitedTables(c.text).flatMap((t) => t.labRows);
    }
    const matched = setPRF(rows, c.expected, (p, g) => normalize(p.test) === normalize(g.test));
    rowCounts.push(matched);
    for (const [p, g] of matched.pairs) {
      fields.result.push(p.result === g.result);
      if (g.unit !== undefined) fields.unit.push(p.unit === g.unit);
      if (g.referenceRange !== undefined) fields.referenceRange.push(p.referenceRange === g.referenceRange);
      if (g.flag !== undefined) fields.flag.push(p.flag === g.flag);
      if (g.page !== undefined) fields.page.push(p.page === g.page);
    }
    perCase.push({ id: c.id, expected: c.expected.length, extracted: rows.length, matched: matched.pairs.length });
  }
  const prf = microPRF(rowCounts);
  return {
    cases: dataset.tables.length,
    rows: { ...prf, precision: round(prf.precision)!, recall: round(prf.recall)!, f1: round(prf.f1)! },
    fieldAccuracy: Object.fromEntries(Object.entries(fields).map(([k, v]) => [k, round(rate(v))])) as TableReport['fieldAccuracy'],
    perCase,
  };
};

/* ───────────────────────── D. Patient retrieval ───────────────────────── */

export interface IndexedCorpus {
  userId: string;
  ids: Map<string, string>;
  allowed: Map<string, string | undefined>;
  vectorIds: string[];
}

/** Indexes the synthetic patient corpus using the production chunking + metadata code. */
export const indexCorpus = async (dataset: Dataset, userId: string): Promise<IndexedCorpus> => {
  const store = getVectorStore();
  const embeddings = getEmbeddings();
  const ids = new Map<string, string>();
  const allowed = new Map<string, string | undefined>();
  const vectorIds: string[] = [];
  for (const doc of dataset.rag.documents) {
    const documentId = `eval-${doc.id}`;
    ids.set(doc.id, documentId);
    allowed.set(documentId, doc.filename);
    const segments = doc.pages.map((text, i) => ({ text, ...(doc.source === 'pdf_text' ? { page: i + 1 } : {}) }));
    const { chunks } = prepareChunks(documentId, segments);
    const vectors = await embeddings.embedDocuments(chunks.map((c) => c.text));
    const mimeType = doc.source === 'pdf_text' ? 'application/pdf' : 'text/plain';
    const records = chunks.map((chunk, i) => ({
      id: chunk.id,
      values: vectors[i],
      metadata: buildChunkMetadata({ userId, source: doc.source, mimeType }, documentId, chunk, embeddings.spaceId),
    }));
    await store.upsert(userNamespace(userId), records);
    vectorIds.push(...records.map((r) => r.id));
  }
  return { userId, ids, allowed, vectorIds };
};

export interface SweepPoint {
  minScore: number;
  retrievalSuccessRate?: number;
  unanswerableRejectionRate?: number;
  falseNegatives: number;
  falsePositives: number;
}

export interface RetrievalReport {
  embedder: string;
  topK: number;
  minScore: number;
  answerable: number;
  unanswerable: number;
  documentHitAt1?: number;
  documentHitAt3?: number;
  documentHitAtK?: number;
  documentMRR?: number;
  pageHitAtK?: number;
  retrievalSuccessRate?: number;
  unanswerableRejectionRate?: number;
  similarity: { answerableTop?: ReturnType<typeof distribution>; unanswerableTop?: ReturnType<typeof distribution> };
  thresholdSweep: SweepPoint[];
  latencyMs: { p50?: number; p95?: number };
  perQuestion: { id: string; answerable: boolean; status: string; topDocuments: string[]; topScore?: number }[];
}

const sweepPoint = (t: number, answerableOk: (t: number) => boolean[], unanswerableRejected: (t: number) => boolean[]): SweepPoint => {
  const ok = answerableOk(t);
  const rejected = unanswerableRejected(t);
  return {
    minScore: t,
    retrievalSuccessRate: round(rate(ok)),
    unanswerableRejectionRate: round(rate(rejected)),
    falseNegatives: ok.filter((v) => !v).length,
    falsePositives: rejected.filter((v) => !v).length,
  };
};

export const runRetrievalEval = async (
  dataset: Dataset,
  corpus: IndexedCorpus,
  options: { topK: number; minScore: number; sweep?: number[]; embedder: string }
): Promise<{ report: RetrievalReport; results: Map<string, RetrievalResult> }> => {
  const results = new Map<string, RetrievalResult>();
  const latencies: number[] = [];
  const perQuestion: RetrievalReport['perQuestion'] = [];
  const hits1: number[] = [];
  const hits3: number[] = [];
  const hitsK: number[] = [];
  const rr: number[] = [];
  const pageHits: number[] = [];
  const pageOf = new Map<string, number | undefined>();

  for (const q of dataset.rag.questions) {
    const start = Date.now();
    const full = await retrieveFromVectorStore({ userId: corpus.userId, question: q.question, allowed: corpus.allowed, topK: options.topK, minScore: -1 });
    latencies.push(Date.now() - start);
    for (const c of full.chunks) pageOf.set(c.chunkId, c.page);
    results.set(q.id, full);
    const rankedDocs = full.scored.map((s) => s.documentId);
    perQuestion.push({
      id: q.id,
      answerable: q.answerable,
      status: full.scored.some((s) => s.score >= options.minScore) ? 'ok' : 'below_threshold',
      topDocuments: [...new Set(rankedDocs)].slice(0, 3),
      topScore: round(full.scored[0]?.score, 4),
    });
    if (!q.answerable || !q.expectedDocumentId) continue;
    const relevant = new Set([corpus.ids.get(q.expectedDocumentId)!]);
    hits1.push(hitAtK(rankedDocs, relevant, 1));
    hits3.push(hitAtK(rankedDocs, relevant, 3));
    hitsK.push(hitAtK(rankedDocs, relevant, options.topK));
    rr.push(reciprocalRank(rankedDocs, relevant));
    if (q.expectedPage) pageHits.push(full.scored.some((s) => relevant.has(s.documentId) && pageOf.get(s.chunkId) === q.expectedPage) ? 1 : 0);
  }

  const answerableOk = (t: number) =>
    dataset.rag.questions.filter((q) => q.answerable).map((q) => results.get(q.id)!.scored.some((s) => s.documentId === corpus.ids.get(q.expectedDocumentId!) && s.score >= t));
  const unanswerableRejected = (t: number) =>
    dataset.rag.questions.filter((q) => !q.answerable).map((q) => !results.get(q.id)!.scored.some((s) => s.score >= t));
  const topOf = (answerable: boolean) => dataset.rag.questions.filter((q) => q.answerable === answerable).map((q) => results.get(q.id)!.scored[0]?.score ?? 0);

  return {
    results,
    report: {
      embedder: options.embedder,
      topK: options.topK,
      minScore: options.minScore,
      answerable: dataset.rag.questions.filter((q) => q.answerable).length,
      unanswerable: dataset.rag.questions.filter((q) => !q.answerable).length,
      documentHitAt1: round(mean(hits1)),
      documentHitAt3: round(mean(hits3)),
      documentHitAtK: round(mean(hitsK)),
      documentMRR: round(mean(rr)),
      pageHitAtK: round(mean(pageHits)),
      retrievalSuccessRate: round(rate(answerableOk(options.minScore))),
      unanswerableRejectionRate: round(rate(unanswerableRejected(options.minScore))),
      similarity: { answerableTop: distribution(topOf(true)), unanswerableTop: distribution(topOf(false)) },
      thresholdSweep: (options.sweep ?? []).map((t) => sweepPoint(t, answerableOk, unanswerableRejected)),
      latencyMs: { p50: percentile(latencies, 50), p95: percentile(latencies, 95) },
      perQuestion,
    },
  };
};

/* ───────────────────────── D/E. Reference retrieval + source attribution ───────────────────────── */

export interface ReferenceCorpusIndex {
  active: Map<string, ActiveReference>;
  vectorIds: string[];
}

/** Indexes the SYNTHETIC reference corpus with the production loader/chunker/metadata. */
export const indexReferenceCorpus = async (dataset: Dataset): Promise<ReferenceCorpusIndex> => {
  const corpus = dataset.referenceCorpus;
  if (!corpus) return { active: new Map(), vectorIds: [] };
  const store = getVectorStore();
  const embeddings = getEmbeddings();
  const active = new Map<string, ActiveReference>();
  const vectorIds: string[] = [];
  for (const source of corpus.sources) {
    const meta = {
      sourceId: source.sourceId,
      title: source.title,
      organization: 'AyuNidan Synthetic Test Corpus (fictional)',
      sourceType: 'synthetic_test' as const,
      authorization: 'Synthetic evaluation fixture',
      medicalTopics: [],
    };
    const { segments } = await loadReferenceSegments(Buffer.from(source.text), `${source.sourceId}.md`);
    const docs = buildReferenceDocuments(meta, segments, embeddings.spaceId);
    const vectors = await embeddings.embedDocuments(docs.map((d) => d.pageContent));
    await store.upsert(
      REFERENCE_NAMESPACE,
      docs.map((d, i) => ({ id: d.metadata.chunkId as string, values: vectors[i], metadata: d.metadata as Record<string, MetadataValue> }))
    );
    vectorIds.push(...docs.map((d) => d.metadata.chunkId as string));
    active.set(source.sourceId, { sourceId: source.sourceId, embeddingSpace: embeddings.spaceId });
  }
  return { active, vectorIds };
};

export interface ReferenceRetrievalReport {
  questions: number;
  topKPerQuery: number;
  minScore: number;
  sourceHitAtK?: number;
  sourceMRR?: number;
  /** Fraction of expected sources retrieved above the threshold (multi-source questions). */
  expectedSourceRecall?: number;
  multiSourceQuestions: number;
  multiSourceFullRecallRate?: number;
  noResultAccuracy?: number;
  similarity: { answerableTop?: ReturnType<typeof distribution>; unanswerableTop?: ReturnType<typeof distribution> };
  thresholdSweep: SweepPoint[];
}

export const runReferenceRetrievalEval = async (
  dataset: Dataset,
  index: ReferenceCorpusIndex,
  options: { topKPerQuery: number; minScore: number; sweep?: number[] }
): Promise<ReferenceRetrievalReport | undefined> => {
  const corpus = dataset.referenceCorpus;
  if (!corpus) return undefined;
  const ranked = new Map<string, { sourceId: string; score: number }[]>();
  for (const q of corpus.questions) {
    const r = await retrieveReferenceFromStore([q.query], index.active, { topKPerQuery: options.topKPerQuery, minScore: -1 });
    ranked.set(q.id, r.evidence.map((e) => ({ sourceId: e.sourceId!, score: e.similarity })));
  }
  const answerable = corpus.questions.filter((q) => q.answerable);
  const hits = answerable.map((q) => hitAtK(ranked.get(q.id)!.map((r) => r.sourceId), new Set(q.expectedSourceIds), options.topKPerQuery));
  const rr = answerable.map((q) => reciprocalRank(ranked.get(q.id)!.map((r) => r.sourceId), new Set(q.expectedSourceIds)));
  const recallAt = (q: (typeof answerable)[number], t: number) => {
    const found = new Set(ranked.get(q.id)!.filter((r) => r.score >= t).map((r) => r.sourceId));
    return (q.expectedSourceIds ?? []).filter((s) => found.has(s)).length / Math.max(1, q.expectedSourceIds?.length ?? 1);
  };
  const multi = answerable.filter((q) => (q.expectedSourceIds?.length ?? 0) > 1);
  const answerableOk = (t: number) => answerable.map((q) => recallAt(q, t) > 0);
  const unanswerableRejected = (t: number) => corpus.questions.filter((q) => !q.answerable).map((q) => !ranked.get(q.id)!.some((r) => r.score >= t));
  const topOf = (a: boolean) => corpus.questions.filter((q) => q.answerable === a).map((q) => ranked.get(q.id)![0]?.score ?? 0);
  return {
    questions: corpus.questions.length,
    topKPerQuery: options.topKPerQuery,
    minScore: options.minScore,
    sourceHitAtK: round(mean(hits)),
    sourceMRR: round(mean(rr)),
    expectedSourceRecall: round(mean(answerable.map((q) => recallAt(q, options.minScore)))),
    multiSourceQuestions: multi.length,
    multiSourceFullRecallRate: round(rate(multi.map((q) => recallAt(q, options.minScore) === 1))),
    noResultAccuracy: round(rate(unanswerableRejected(options.minScore))),
    similarity: { answerableTop: distribution(topOf(true)), unanswerableTop: distribution(topOf(false)) },
    thresholdSweep: (options.sweep ?? []).map((t) => sweepPoint(t, answerableOk, unanswerableRejected)),
  };
};

/* ───────────────────────── F/H. Grounded answers (live) ───────────────────────── */

export interface AnswerReport {
  questions: number;
  expectedSourceCitedRate?: number;
  answerContainsExpectedRate?: number;
  unanswerableHandledRate?: number;
  /** Answered responses with ≥1 validated citation (1 by construction: uncited answers are rejected). */
  answeredWithCitationRate?: number;
  /** Model-proposed citation ids that did not exist in the supplied evidence (rejected). */
  invalidCitationAttempts: number;
  failures: Record<string, number>;
  note: string;
}

export const runAnswerEval = async (
  dataset: Dataset,
  corpus: IndexedCorpus,
  reference: ReferenceCorpusIndex,
  options: { topK: number; minScore: number; refMinScore: number }
): Promise<AnswerReport> => {
  const cited: boolean[] = [];
  const contains: boolean[] = [];
  const handled: boolean[] = [];
  const withCitations: boolean[] = [];
  const failures: Record<string, number> = {};
  let invalid = 0;
  for (const q of dataset.rag.questions) {
    try {
      const patient = await retrieveFromVectorStore({ userId: corpus.userId, question: q.question, allowed: corpus.allowed, topK: options.topK, minScore: options.minScore });
      const refs = await retrieveReferenceFromStore([q.question], reference.active, { minScore: options.refMinScore });
      const evidence = assembleEvidence(patient.chunks.map(chunkToEvidence), refs.evidence);
      const answer = await answerFromEvidence(q.question, evidence, {
        status: patient.status,
        candidates: patient.candidates,
        used: evidence.filter((e) => e.domain === 'patient').length,
        referenceStatus: refs.status,
        referencesRetrieved: refs.evidence.length,
      });
      if (answer.status === 'answered') withCitations.push(answer.citations.length > 0);
      if (q.answerable) {
        const expectedDoc = corpus.ids.get(q.expectedDocumentId!);
        cited.push(answer.status === 'answered' && answer.citations.some((c) => c.documentId === expectedDoc));
        contains.push(answer.status === 'answered' && (q.answerContains ?? []).every((f) => normalize(answer.answer).includes(normalize(f))));
      } else {
        handled.push(answer.status === 'insufficient_context');
      }
    } catch (error) {
      const category = categoryOf(error);
      failures[category] = (failures[category] ?? 0) + 1;
    }
  }
  const { getFailureCounts } = await import('../utils/failures');
  invalid = getFailureCounts().citation_failure ?? 0;
  return {
    questions: dataset.rag.questions.length,
    expectedSourceCitedRate: round(rate(cited)),
    answerContainsExpectedRate: round(rate(contains)),
    unanswerableHandledRate: round(rate(handled)),
    answeredWithCitationRate: round(rate(withCitations)),
    invalidCitationAttempts: invalid,
    failures,
    note: 'String-containment and citation checks on synthetic questions. Not a faithfulness or clinical-accuracy measure.',
  };
};

/* ───────────────────────── G. Risk classification (live) ───────────────────────── */

export interface AssessmentReport {
  model: string;
  cases: number;
  accuracy?: number;
  confusion: Record<string, Record<string, number>>;
  perClass: Record<string, { precision?: number; recall?: number; support: number }>;
  predictedDistribution: Record<string, number>;
  droppedCitations: number;
  downgradedToInsufficient: number;
  failureRate: number;
  failureCategories: Record<string, number>;
  note: string;
}

export const runAssessmentEval = async (dataset: Dataset, candidate: ModelCandidate): Promise<AssessmentReport> => {
  const confusion: Record<string, Record<string, number>> = {};
  const pairs: { expected: string; predicted: string }[] = [];
  const failureCategories: Record<string, number> = {};
  let dropped = 0;
  let downgraded = 0;
  for (const c of dataset.assessment) {
    try {
      const result = await assessRisk(c.findings, { userId: 'eval-assessment', candidates: [candidate] });
      pairs.push({ expected: c.expectedRiskLevel, predicted: result.riskLevel });
      confusion[c.expectedRiskLevel] ??= {};
      confusion[c.expectedRiskLevel][result.riskLevel] = (confusion[c.expectedRiskLevel][result.riskLevel] ?? 0) + 1;
      dropped += result.validation.droppedCitations;
      if (result.validation.downgradedToInsufficient) downgraded++;
    } catch (error) {
      const category = categoryOf(error);
      failureCategories[category] = (failureCategories[category] ?? 0) + 1;
    }
  }
  const perClass: AssessmentReport['perClass'] = {};
  const predictedDistribution: Record<string, number> = {};
  for (const p of pairs) predictedDistribution[p.predicted] = (predictedDistribution[p.predicted] ?? 0) + 1;
  for (const level of RISK_LEVELS) {
    const tp = pairs.filter((p) => p.expected === level && p.predicted === level).length;
    const predicted = pairs.filter((p) => p.predicted === level).length;
    const support = pairs.filter((p) => p.expected === level).length;
    perClass[level] = { precision: predicted ? round(tp / predicted) : undefined, recall: support ? round(tp / support) : undefined, support };
  }
  const failed = Object.values(failureCategories).reduce((s, n) => s + n, 0);
  return {
    model: `${candidate.provider}:${candidate.modelId}`,
    cases: dataset.assessment.length,
    accuracy: round(rate(pairs.map((p) => p.expected === p.predicted))),
    confusion,
    perClass,
    predictedDistribution,
    droppedCitations: dropped,
    downgradedToInsufficient: downgraded,
    failureRate: round(dataset.assessment.length ? failed / dataset.assessment.length : 0)!,
    failureCategories,
    note: 'Agreement with synthetic engineering labels only (n is tiny). This is NOT clinical validation.',
  };
};

/* ───────────────────────── I. Failure categorisation (offline) ───────────────────────── */

const fakeCandidate = (id: string): ModelCandidate => ({ provider: 'google', modelId: id, model: {} as ModelCandidate['model'] });
const named = (name: string, extra: object = {}) => Object.assign(new Error(name), { name, ...extra });
const tryCategory = async (fn: () => Promise<unknown> | unknown): Promise<FailureCategory | 'none'> => {
  try {
    await fn();
    return 'none';
  } catch (e) {
    return categoryOf(e);
  }
};

/** Each scenario reproduces a failure through production code and returns the observed category. */
const SCENARIOS: Record<string, () => Promise<FailureCategory | 'none'>> = {
  all_providers_down: () => tryCategory(() => withModelFallback('eval', async () => { throw named('AI_APICallError'); }, [fakeCandidate('a1'), fakeCandidate('b1')])),
  all_providers_timeout: () => tryCategory(() => withModelFallback('eval', async () => { throw named('TimeoutError'); }, [fakeCandidate('a2'), fakeCandidate('b2')])),
  timeout_then_malformed_output: async () => {
    let call = 0;
    return tryCategory(() =>
      withModelFallback('eval', async () => { throw call++ === 0 ? named('TimeoutError') : new OutputValidationError('bad'); }, [fakeCandidate('a3'), fakeCandidate('b3')])
    );
  },
  malformed_summary_output: () => tryCategory(() => withModelFallback('eval', async () => { throw new OutputValidationError('schema'); }, [fakeCandidate('a4')])),
  empty_extraction_input: () => tryCategory(() => extractMedicalData('   ', [], { candidates: [fakeCandidate('a5')] })),
  vector_query_outage: async () => {
    const store = new InMemoryVectorStore();
    store.failQuery = true;
    return tryCategory(() => store.query('ns', { vector: [], topK: 1 }));
  },
  embedding_outage: async () => {
    const e = new HashingEmbeddings();
    e.failNext = true;
    return tryCategory(() => e.embedQuery('x'));
  },
  invalid_chunking_options: () => tryCategory(() => chunkText('text', { chunkSize: 10, overlap: 0 })),
  unanswerable_question: async () => {
    const result = await retrieveFromVectorStore({ userId: 'eval', question: 'q', allowed: new Map(), topK: 3, minScore: 0.5 });
    return result.chunks.length === 0 ? 'insufficient_context' : 'none';
  },
  forged_vector_metadata: async () => {
    const { setVectorStore, getVectorStore: current } = await import('../rag/vectorStore');
    const previous = current();
    const store = new InMemoryVectorStore();
    setVectorStore(store);
    try {
      const vector = await getEmbeddings().embedQuery('potassium');
      store.inject(userNamespace('victim'), { id: 'd#0', values: vector, metadata: { domain: 'patient', userId: 'victim', documentId: 'd', embeddingSpace: getEmbeddings().spaceId, text: 'x' } });
      const r = await retrieveFromVectorStore({ userId: 'victim', question: 'potassium', allowed: new Map([['d', 'f']]), topK: 3, minScore: -1 });
      return r.chunks.length === 0 && r.status === 'no_matches' ? 'invalid_metadata' : 'none';
    } finally {
      setVectorStore(previous);
    }
  },
  hallucinated_citation: async () => {
    const evidence = assembleEvidence([], [{ chunkId: 'r#0', similarity: 0.9, content: 'x', sourceId: 'r', title: 't', organization: 'o' }]);
    return validateCitations(['R9'], evidence, 'reference').invalid.length === 1 ? 'citation_failure' : 'none';
  },
  rate_limited_provider: () => tryCategory(() => withModelFallback('eval', async () => { throw named('AI_APICallError', { statusCode: 429 }); }, [fakeCandidate('rl')])),
  retired_model: () => tryCategory(() => withModelFallback('eval', async () => { throw named('AI_APICallError', { statusCode: 404 }); }, [fakeCandidate('retired')])),
  corrupted_pdf: () => tryCategory(() => analyzePdf(Buffer.from('%PDF-1.4 corrupted fictional bytes'))),
  unmappable_table: async () => {
    const pdf = buildPdf([{ table: { columns: LAB_COLUMNS, rows: [['Test', 'Result', 'Unit', 'Reference Range', 'Flag'], ['', '', 'g/dL', '13-17', '']] } }]);
    const tables = (await analyzePdf(pdf)).pages.flatMap((p) => p.tables);
    return tables.length > 0 && tables.every((t) => t.labRows.length === 0) ? 'table_extraction_failure' : 'none';
  },
  invalid_audio: () => tryCategory(() => validateAudio({ buffer: Buffer.from('RIFF....WAVEjunk'), mimetype: 'audio/wav', size: 16 })),
};

export interface FailureReport {
  scenarios: number;
  correctlyCategorised: number;
  accuracy?: number;
  results: { id: string; scenario: string; expected: string; observed: string; ok: boolean }[];
}

export const runFailureEval = async (dataset: Dataset): Promise<FailureReport> => {
  const results: FailureReport['results'] = [];
  for (const f of dataset.failures) {
    const run = SCENARIOS[f.scenario];
    const observed = run ? await run() : 'unknown_scenario';
    results.push({ id: f.id, scenario: f.scenario, expected: f.expectedCategory, observed, ok: observed === f.expectedCategory });
  }
  const ok = results.filter((r) => r.ok).length;
  return { scenarios: results.length, correctlyCategorised: ok, accuracy: round(results.length ? ok / results.length : undefined), results };
};
