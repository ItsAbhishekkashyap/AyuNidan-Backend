/**
 * Evaluation metrics/harness, failure categorisation, observability, timeouts,
 * bounded retries and prompt/context budgets. Synthetic data only.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import request from 'supertest';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const generateText = vi.fn();
vi.mock('ai', async (importOriginal) => ({
  ...(await importOriginal<typeof import('ai')>()),
  generateText: (...args: unknown[]) => generateText(...args),
}));

import { createApp } from '../src/app';
import { cache } from '../src/middleware/cache';
import { resetRateLimits } from '../src/middleware/rateLimiter';
import { fuzzyMatch, hitAtK, microPRF, normalizeGender, prfFromCounts, reciprocalRank, setPRF } from '../src/eval/metrics';
import { loadDataset } from '../src/eval/dataset';
import {
  aggregateExtraction,
  indexCorpus,
  indexReferenceCorpus,
  runFailureEval,
  runReferenceRetrievalEval,
  runRetrievalEval,
  runTableEval,
  scoreExtraction,
} from '../src/eval/evaluators';
import { AIConfigurationError, AIProcessingError, extractMedicalData, getAIPolicy, withModelFallback, type ModelCandidate } from '../src/services/ai.service';
import { retrieveFromVectorStore, userNamespace, buildChunkMetadata } from '../src/services/document.service';
import { PineconeVectorStore, getVectorStore, VectorStoreError } from '../src/rag/vectorStore';
import { getEmbeddings, HashingEmbeddings } from '../src/rag/embeddings';
import { InMemoryVectorStore } from '../src/rag/inMemoryVectorStore';
import { getAICallStats, percentile, resetAICallStats } from '../src/utils/metrics';
import { resetFailureCounts } from '../src/utils/failures';
import { resetDb } from './helpers/fakeModels';
import { createTestUser } from './helpers/auth';
import { assessmentOutput, extractionOutput, isQACall } from './helpers/aiOutputs';

const fake = (id: string): ModelCandidate => ({ provider: 'google', modelId: id, model: {} as ModelCandidate['model'] });
const ENV = [
  'AI_MAX_MODEL_ATTEMPTS', 'AI_MAX_RETRIES', 'AI_TOTAL_BUDGET_MS', 'AI_TIMEOUT_MS', 'AI_MAX_EXTRACTION_CHARS', 'RAG_MAX_CONTEXT_CHARS',
  'PINECONE_TIMEOUT_MS', 'PINECONE_MAX_RETRIES', 'PINECONE_API_KEY', 'GEMINI_API_KEY', 'RAG_DOC_MIN_SCORE',
];

beforeEach(() => {
  for (const key of ENV) delete process.env[key];
  generateText.mockReset();
  resetAICallStats();
  resetFailureCounts();
  resetDb();
  cache.clear();
  resetRateLimits();
});
afterEach(() => {
  for (const key of ENV) delete process.env[key];
  vi.unstubAllGlobals();
});

/* ───────────── Metrics ───────────── */

describe('evaluation metrics', () => {
  it('computes precision/recall/F1 with greedy fuzzy matching', () => {
    const result = setPRF(['Chest pain', 'fever', 'nausea'], ['chest pain', 'sweating', 'fever'], fuzzyMatch);
    expect(result).toMatchObject({ tp: 2, fp: 1, fn: 1 });
    expect(result.f1).toBeCloseTo(2 / 3);
  });

  it('treats empty-vs-empty as perfect and invention against empty gold as a precision error', () => {
    expect(prfFromCounts(0, 0, 0)).toMatchObject({ precision: 1, recall: 1, f1: 1 });
    expect(prfFromCounts(0, 2, 0)).toMatchObject({ precision: 0, recall: 1 });
    expect(microPRF([{ tp: 1, fp: 0, fn: 1 }, { tp: 1, fp: 1, fn: 0 }])).toMatchObject({ tp: 2, fp: 1, fn: 1 });
  });

  it('computes ranking metrics', () => {
    const relevant = new Set(['d2']);
    expect(hitAtK(['d1', 'd2', 'd3'], relevant, 1)).toBe(0);
    expect(hitAtK(['d1', 'd2', 'd3'], relevant, 2)).toBe(1);
    expect(reciprocalRank(['d1', 'd2'], relevant)).toBe(0.5);
    expect(reciprocalRank(['d1'], relevant)).toBe(0);
    expect(percentile([5, 1, 3, 2, 4], 50)).toBe(3);
    expect(percentile([], 95)).toBeUndefined();
  });

  it('normalises strings and genders for matching', () => {
    expect(fuzzyMatch('Aspirin 75 mg', 'aspirin')).toBe(true);
    expect(fuzzyMatch('ab', 'abc')).toBe(false);
    expect(normalizeGender('F')).toBe('female');
  });

  it('scores an extraction against synthetic gold', () => {
    const score = scoreExtraction(
      'x',
      {
        patientDetails: { name: 'Arlo Pemberwick', age: 58, gender: 'M' },
        symptoms: ['chest pain'],
        medicines: ['aspirin 75 mg', 'atorvastatin'],
        labValues: [{ name: 'Troponin I', value: '2.4 ng/mL', unit: 'ng/mL', isAbnormal: false }],
        rawText: '',
        report: { documents: [], pages: [], tables: [], dates: [], diagnosesMentioned: [], measurements: [], reportRiskScores: [], imagingFindings: [], unverifiedFromImages: [], doctorNotes: '', extractionLimitations: [] },
      },
      {
        name: 'Arlo Pemberwick', age: 58, gender: 'male',
        symptoms: ['chest pain', 'sweating'], medicines: ['aspirin', 'atorvastatin'],
        labValues: [{ name: 'Troponin I', value: '2.4', isAbnormal: true }],
      }
    );
    expect(score).toMatchObject({ nameMatch: true, ageMatch: true, genderMatch: true, labValueMatches: [true], labFlagMatches: [false] });
    expect(score.symptoms).toMatchObject({ tp: 1, fn: 1, fp: 0 });
  });

  it('reports no P/R/F1 when every case failed (never a misleading perfect score)', () => {
    const report = aggregateExtraction('google:x', [], [{ id: 'a', failureCategory: 'provider_failure' }], []);
    expect(report.failureRate).toBe(1);
    expect(report.symptoms).toBeUndefined();
  });
});

/* ───────────── Dataset + harness ───────────── */

describe('synthetic dataset and harness', () => {
  it('loads and validates the synthetic dataset', () => {
    const dataset = loadDataset();
    expect(dataset._notice).toMatch(/SYNTHETIC/);
    expect(dataset.referenceCorpus?._notice).toMatch(/SYNTHETIC/);
    expect(dataset.tables.length).toBeGreaterThanOrEqual(4);
    expect(dataset.assessment.some((a) => a.expectedRiskLevel === 'insufficient_evidence')).toBe(true);
  });

  it('rejects datasets with broken references', () => {
    const dataset = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'eval', 'synthetic-dataset.json'), 'utf8'));
    dataset.rag.questions[0].expectedDocumentId = 'does-not-exist';
    const file = path.join(os.tmpdir(), `bad-dataset-${Date.now()}.json`);
    fs.writeFileSync(file, JSON.stringify(dataset));
    expect(() => loadDataset(file)).toThrow(/unknown document/);
    fs.unlinkSync(file);
  });

  it('categorises every failure scenario correctly through production code', async () => {
    const report = await runFailureEval(loadDataset());
    expect(report.results.filter((r) => !r.ok)).toEqual([]);
    expect(report.accuracy).toBe(1);
  });

  it('evaluates deterministic table extraction (PDF layout + delimited)', async () => {
    const report = await runTableEval(loadDataset());
    expect(report.rows).toMatchObject({ precision: 1, recall: 1 });
    expect(report.fieldAccuracy).toEqual({ result: 1, unit: 1, referenceRange: 1, flag: 1, page: 1 });
  });

  it('runs patient + reference retrieval evaluation deterministically on the offline embedder', async () => {
    const dataset = loadDataset();
    const corpus = await indexCorpus(dataset, 'eval-user');
    const run = () => runRetrievalEval(dataset, corpus, { topK: 5, minScore: 0.2, sweep: [0, 0.3], embedder: 'offline' });
    const first = await run();
    const second = await run();
    expect(first.report.documentHitAt3).toBe(1);
    expect(first.report.thresholdSweep[0]).toHaveProperty('falsePositives');
    const { latencyMs: _a, ...stableFirst } = first.report;
    const { latencyMs: _b, ...stableSecond } = second.report;
    expect(stableFirst).toEqual(stableSecond);

    const reference = await indexReferenceCorpus(dataset);
    const refReport = await runReferenceRetrievalEval(dataset, reference, { topKPerQuery: 4, minScore: 0.2, sweep: [0.2] });
    expect(refReport).toMatchObject({ sourceHitAtK: 1, multiSourceQuestions: 1 });
    expect(refReport?.similarity.answerableTop?.n).toBe(6);
  });
});

/* ───────────── Bounded retries / fallback ───────────── */

describe('bounded retries and fallback', () => {
  it('caps the number of models tried', async () => {
    process.env.AI_MAX_MODEL_ATTEMPTS = '2';
    const run = vi.fn(async () => { throw Object.assign(new Error('x'), { name: 'AI_APICallError' }); });
    await expect(withModelFallback('op', run, [fake('a'), fake('b'), fake('c'), fake('d')])).rejects.toBeInstanceOf(AIProcessingError);
    expect(run).toHaveBeenCalledTimes(2);
  });

  it('passes a bounded SDK retry count (default 1, max 3)', async () => {
    expect(getAIPolicy().maxRetries).toBe(1);
    process.env.AI_MAX_RETRIES = '9';
    expect(getAIPolicy().maxRetries).toBe(3);
    process.env.AI_MAX_RETRIES = '0';
    const result = await withModelFallback('op', vi.fn(async (ctx: { maxRetries: number }) => ({ value: ctx.maxRetries })), [fake('a')]);
    expect(result.value).toBe(0);
  });

  it('stops trying further models once the total budget is spent', async () => {
    process.env.AI_TOTAL_BUDGET_MS = '40';
    const run = vi.fn(async () => {
      await new Promise((r) => setTimeout(r, 60));
      throw new Error('slow failure');
    });
    await expect(withModelFallback('op', run, [fake('a'), fake('b'), fake('c')])).rejects.toBeInstanceOf(AIProcessingError);
    expect(run).toHaveBeenCalledTimes(1);
  });

  it('does not fall back on configuration errors', async () => {
    const run = vi.fn(async () => { throw new AIConfigurationError('no key'); });
    await expect(withModelFallback('op', run, [fake('a'), fake('b')])).rejects.toBeInstanceOf(AIConfigurationError);
    expect(run).toHaveBeenCalledTimes(1);
  });

  it('tries each model once for schema failures (no same-model retry)', async () => {
    const run = vi.fn(async () => { throw Object.assign(new Error('bad json'), { name: 'AI_NoObjectGeneratedError' }); });
    const error = await withModelFallback('op', run, [fake('a'), fake('b')]).catch((e) => e);
    expect(run).toHaveBeenCalledTimes(2);
    expect(error.category).toBe('schema_validation_failure');
  });
});

/* ───────────── Timeouts ───────────── */

describe('timeouts on external calls', () => {
  it('aborts hanging AI calls at the per-call timeout and falls back', async () => {
    process.env.AI_TIMEOUT_MS = '30';
    let calls = 0;
    const run = async (ctx: { abortSignal: AbortSignal }) => {
      calls++;
      if (calls === 1) await new Promise((_r, reject) => ctx.abortSignal.addEventListener('abort', () => reject(ctx.abortSignal.reason)));
      return { value: 'ok' };
    };
    const started = Date.now();
    const result = await withModelFallback('op', run, [fake('a'), fake('b')]);
    expect(result).toMatchObject({ value: 'ok', fallbackUsed: true });
    expect(Date.now() - started).toBeLessThan(1_000);
  });

  it('aborts hanging Pinecone requests via the fetch timeout', async () => {
    process.env.PINECONE_API_KEY = 'test-key';
    process.env.PINECONE_TIMEOUT_MS = '50';
    process.env.PINECONE_MAX_RETRIES = '0';
    vi.stubGlobal('fetch', (_url: unknown, init?: RequestInit) =>
      new Promise((_resolve, reject) => init?.signal?.addEventListener('abort', () => reject(init.signal!.reason)))
    );
    const started = Date.now();
    const error = await new PineconeVectorStore('synthetic-index').query('ns', { vector: [0.1, 0.2], topK: 1 }).catch((e) => e);
    expect(error).toBeInstanceOf(VectorStoreError);
    expect(error.category).toBe('vector_db_failure');
    expect(Date.now() - started).toBeLessThan(3_000);
  });
});

/* ───────────── Prompt / context budgets ───────────── */

describe('prompt and context budgets', () => {
  it('truncates oversized extraction input explicitly and flags it', async () => {
    process.env.GEMINI_API_KEY = 'test';
    process.env.AI_MAX_EXTRACTION_CHARS = '200';
    generateText.mockResolvedValue({ output: extractionOutput() });
    const result = await extractMedicalData('synthetic '.repeat(500));
    expect(result.truncated).toBe(true);
    expect(result.report.extractionLimitations.join(' ')).toMatch(/truncated/);
    const sent = generateText.mock.calls[0][0].messages[0].content[0].text as string;
    expect(sent.length).toBeLessThan(500);
    expect(sent).toContain('input truncated');
  });

  it('bounds retrieved context by the character budget', async () => {
    process.env.RAG_MAX_CONTEXT_CHARS = '500';
    const store = getVectorStore() as InMemoryVectorStore;
    const embedder = getEmbeddings() as HashingEmbeddings;
    const allowed = new Map([['d', 'f.txt']]);
    for (let i = 0; i < 6; i++) {
      const text = `synthetic potassium chunk ${i} `.repeat(10);
      store.inject(userNamespace('u'), {
        id: `d#${i}`,
        values: embedder.embedOne(text),
        metadata: buildChunkMetadata({ userId: 'u', source: 'text_file', mimeType: 'text/plain' }, 'd', { id: `d#${i}`, chunkIndex: i, text }),
      });
    }
    const result = await retrieveFromVectorStore({ userId: 'u', question: 'synthetic potassium chunk', allowed, topK: 6, minScore: -1 });
    expect(result.status).toBe('ok');
    expect(result.chunks.length).toBe(1);
    expect(result.chunks.reduce((s, c) => s + c.text.length, 0)).toBeLessThanOrEqual(500);
  });

  it('bounds the question length sent for embedding', async () => {
    const spy = vi.spyOn(getEmbeddings(), 'embedQuery');
    await retrieveFromVectorStore({ userId: 'u', question: 'x'.repeat(5000), allowed: new Map([['d', 'f']]), topK: 3, minScore: 0 });
    expect(spy.mock.calls[0][0].length).toBe(500);
  });
});

/* ───────────── Observability ───────────── */

describe('AI observability', () => {
  const captureLogs = () => {
    const lines: string[] = [];
    for (const m of ['log', 'warn', 'error'] as const) vi.spyOn(console, m).mockImplementation((line: unknown) => { lines.push(String(line)); });
    return lines;
  };

  it('logs safe structured metadata for AI calls, tied to the request id, with no PHI', async () => {
    process.env.GEMINI_API_KEY = 'test';
    const original = process.env.NODE_ENV;
    process.env.NODE_ENV = 'development';
    const lines = captureLogs();

    generateText
      .mockRejectedValueOnce(Object.assign(new Error('Wanda Fictitia has chest pain'), { name: 'AI_APICallError' }))
      .mockResolvedValueOnce({
        output: assessmentOutput({ summary: 'Wanda Fictitia synthetic summary text.', riskLevel: 'medium', riskScore: 50 }),
        usage: { inputTokens: 120, outputTokens: 40, totalTokens: 160 },
      });

    const user = createTestUser();
    const res = await request(createApp())
      .post('/api/consultations')
      .set('Authorization', user.auth)
      .set('X-Request-Id', 'req-observability-1')
      .send({ rawText: 'Wanda Fictitia reports chest pain', symptoms: ['chest pain'] });
    process.env.NODE_ENV = original;

    expect(res.status).toBe(201);
    const aiLines = lines.filter((l) => l.includes('"event":"ai.call"')).map((l) => JSON.parse(l));
    expect(aiLines).toHaveLength(2);
    expect(aiLines[0]).toMatchObject({ requestId: 'req-observability-1', operation: 'assess_clinical_risk', status: 'error', failureCategory: 'provider_failure', attempt: 1 });
    expect(aiLines[1]).toMatchObject({
      requestId: 'req-observability-1', status: 'ok', provider: 'google', model: 'gemini-2.5-flash-lite',
      fallbackUsed: true, inputTokens: 120, outputTokens: 40, totalTokens: 160,
    });
    const completed = lines.map((l) => JSON.parse(l)).find((l) => l.event === 'consultation.completed');
    expect(completed).toMatchObject({ requestId: 'req-observability-1', referencesUsed: 0, droppedCitations: 0 });
    expect(completed.t_assessment_ms).toBeTypeOf('number');

    const all = lines.join('\n');
    expect(all).not.toContain('Wanda');
    expect(all).not.toContain('chest pain');

    const stats = getAICallStats().find((s) => s.operation === 'assess_clinical_risk' && s.model === 'google:gemini-2.5-flash-lite');
    expect(stats).toMatchObject({ calls: 1, errors: 0, fallbackCalls: 1, inputTokens: 120, outputTokens: 40 });
  });

  it('never logs document text or questions during RAG', async () => {
    process.env.GEMINI_API_KEY = 'test';
    process.env.RAG_DOC_MIN_SCORE = '0.01';
    const original = process.env.NODE_ENV;
    process.env.NODE_ENV = 'development';
    const lines = captureLogs();
    generateText.mockImplementation(async (opts: { system?: string }) =>
      isQACall(opts)
        ? { output: { answer: 'Potassium 7.1 [P1]', patientCitations: ['P1'], referenceCitations: [], uncertainty: '', insufficientContext: false } }
        : { output: extractionOutput() }
    );

    const app = createApp();
    const user = createTestUser();
    await request(app).post('/api/uploads').set('Authorization', user.auth)
      .attach('files', Buffer.from('Quillon Fictivus potassium 7.1 mmol/L'), { filename: 'n.txt', contentType: 'text/plain' });
    const res = await request(app).post('/api/documents/query').set('Authorization', user.auth).send({ question: 'What is Quillon Fictivus potassium?' });
    process.env.NODE_ENV = original;

    expect(res.body.data.status).toBe('answered');
    const all = lines.join('\n');
    expect(all).toContain('document.query');
    expect(all).not.toContain('Quillon');
    expect(all).not.toContain('7.1');
  });
});
