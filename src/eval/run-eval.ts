/**
 * Evaluation harness — ENGINEERING evaluation on SYNTHETIC data. It is not clinical validation.
 *
 *   npm run eval                          offline, no network: failure categorisation, table
 *                                          extraction, patient + reference retrieval (hashing embedder)
 *   npm run eval -- --embedder=hf         same, but retrieval uses the LOCAL Hugging Face model
 *                                          (bge-small) — free, no API calls; use this to set thresholds
 *   npm run eval -- --live                adds paid LLM evaluations (extraction, risk classification,
 *                                          grounded answers) and runs retrieval on Pinecone
 *   npm run eval -- --live --models=google:gemini-2.5-flash-lite --only=assessment
 *
 * Writes a JSON report to eval/results/.
 */
import 'dotenv/config';
import fs from 'node:fs';
import path from 'node:path';
import { loadDataset, DEFAULT_DATASET_PATH } from './dataset';
import {
  indexCorpus,
  indexReferenceCorpus,
  runAnswerEval,
  runAssessmentEval,
  runExtractionEval,
  runFailureEval,
  runReferenceRetrievalEval,
  runRetrievalEval,
  runTableEval,
  type RetrievalReport,
} from './evaluators';
import { createModelCandidate, getModelCandidates, type ModelCandidate, type ProviderName } from '../services/ai.service';
import { getRagConfig } from '../services/document.service';
import { getReferenceRetrievalConfig } from '../services/reference.service';
import { getVectorStore, setVectorStore, REFERENCE_NAMESPACE } from '../rag/vectorStore';
import { InMemoryVectorStore } from '../rag/inMemoryVectorStore';
import { getEmbeddings, HashingEmbeddings, setEmbeddings } from '../rag/embeddings';
import { getFailureCounts, resetFailureCounts } from '../utils/failures';
import { getAICallStats, resetAICallStats } from '../utils/metrics';
import { chunkText } from '../rag/chunking';

const args = new Map(
  process.argv.slice(2).map((arg) => {
    const [key, value] = arg.replace(/^--/, '').split('=');
    return [key, value ?? 'true'] as const;
  })
);
const live = args.get('live') === 'true';
const embedderArg = live ? 'hf' : (args.get('embedder') ?? 'hash');
const only = args.get('only')?.split(',');
const wants = (part: string) => !only || only.includes(part);

const parseModels = (): ModelCandidate[] => {
  const spec = args.get('models');
  if (!spec) return [getModelCandidates()[0]];
  return spec.split(',').map((s) => {
    const [provider, ...rest] = s.split(':');
    return createModelCandidate(provider as ProviderName, rest.join(':'));
  });
};

/** Pinecone is eventually consistent: wait until all upserted vectors are queryable. */
const waitForVectors = async (namespace: string, expected: number): Promise<void> => {
  const probe = await getEmbeddings().embedQuery('probe');
  for (let i = 0; i < 30; i++) {
    const matches = await getVectorStore().query(namespace, { vector: probe, topK: Math.min(100, expected) });
    if (matches.length >= Math.min(100, expected)) return;
    await new Promise((r) => setTimeout(r, 1000));
  }
  throw new Error('Timed out waiting for indexed vectors to become queryable');
};

const main = async (): Promise<void> => {
  process.env.NODE_ENV ??= 'test';
  process.env.RAG_INCLUDE_SYNTHETIC_REFERENCES = 'true'; // synthetic reference corpus is evaluation-only
  const dataset = loadDataset(args.get('dataset') ?? DEFAULT_DATASET_PATH);
  resetFailureCounts();
  resetAICallStats();

  if (!live) setVectorStore(new InMemoryVectorStore());
  if (embedderArg === 'hash') setEmbeddings(new HashingEmbeddings());
  const embeddings = getEmbeddings();

  const report: Record<string, unknown> = {
    notice: 'ENGINEERING evaluation on synthetic fixtures. Results are not clinical validation and do not describe real-world clinical performance.',
    mode: live ? 'live' : 'offline',
    embedder: embeddings.spaceId,
    startedAt: new Date().toISOString(),
    dataset: {
      extractionCases: dataset.extraction.length,
      tableCases: dataset.tables.length,
      ragDocuments: dataset.rag.documents.length,
      ragQuestions: dataset.rag.questions.length,
      referenceSources: dataset.referenceCorpus?.sources.length ?? 0,
      referenceQuestions: dataset.referenceCorpus?.questions.length ?? 0,
      assessmentCases: dataset.assessment.length,
      failureScenarios: dataset.failures.length,
    },
    chunking: dataset.rag.documents.map((d) => ({ id: d.id, pages: d.pages.length, chunks: d.pages.flatMap((p) => chunkText(p)).length })),
  };

  if (wants('failures')) report.failureCategorisation = await runFailureEval(dataset);
  if (wants('tables')) report.tableExtraction = await runTableEval(dataset);

  if (live && (wants('extraction') || wants('assessment'))) {
    const models = parseModels();
    if (wants('extraction')) report.extraction = [];
    if (wants('assessment')) report.riskClassification = [];
    for (const model of models) {
      if (wants('extraction')) (report.extraction as unknown[]).push(await runExtractionEval(dataset, model));
      if (wants('assessment')) (report.riskClassification as unknown[]).push(await runAssessmentEval(dataset, model));
    }
  }

  if (wants('retrieval') || (live && wants('answers'))) {
    const docConfig = getRagConfig();
    const refConfig = getReferenceRetrievalConfig();
    const hashScale = embedderArg === 'hash';
    const docMinScore = hashScale ? Number(args.get('offline-min-score') ?? 0.2) : docConfig.minScore;
    const refMinScore = hashScale ? Number(args.get('offline-min-score') ?? 0.2) : refConfig.minScore;
    const sweep = hashScale ? [0, 0.1, 0.2, 0.3, 0.4] : [0.6, 0.62, 0.64, 0.66, 0.68, 0.7, 0.72, 0.74, 0.76, 0.78, 0.8];
    const userId = `eval-${Date.now()}`;

    const corpus = await indexCorpus(dataset, userId);
    const reference = await indexReferenceCorpus(dataset);
    try {
      if (live) {
        await waitForVectors(`user-${userId}`, corpus.vectorIds.length);
        await waitForVectors(REFERENCE_NAMESPACE, reference.vectorIds.length);
      }
      if (wants('retrieval')) {
        report.patientRetrieval = (await runRetrievalEval(dataset, corpus, { topK: docConfig.topK, minScore: docMinScore, sweep, embedder: embeddings.spaceId })).report;
        report.referenceRetrieval = await runReferenceRetrievalEval(dataset, reference, { topKPerQuery: refConfig.topKPerQuery, minScore: refMinScore, sweep });
      }
      if (live && wants('answers')) {
        report.groundedAnswers = await runAnswerEval(dataset, corpus, reference, { topK: docConfig.topK, minScore: docMinScore, refMinScore });
      }
    } finally {
      await getVectorStore().deleteIds(`user-${userId}`, corpus.vectorIds).catch(() => undefined);
      if (live) await getVectorStore().deleteIds(REFERENCE_NAMESPACE, reference.vectorIds).catch(() => undefined);
    }
  }

  setVectorStore(null);
  setEmbeddings(null);
  report.failureCounts = getFailureCounts();
  report.aiCalls = getAICallStats();
  report.finishedAt = new Date().toISOString();

  const outDir = path.resolve(__dirname, '..', '..', 'eval', 'results');
  fs.mkdirSync(outDir, { recursive: true });
  const outFile = path.join(outDir, `eval-${report.mode}-${embedderArg}-${new Date().toISOString().replace(/[:.]/g, '-')}.json`);
  fs.writeFileSync(outFile, JSON.stringify(report, null, 2));

  const headline = (r?: RetrievalReport) => (r ? (({ perQuestion: _p, latencyMs: _l, ...rest }: RetrievalReport) => rest)(r) : undefined);
  console.log(
    JSON.stringify(
      {
        mode: report.mode,
        embedder: report.embedder,
        failureCategorisation: (report.failureCategorisation as { accuracy?: number } | undefined)?.accuracy,
        tableExtraction: report.tableExtraction,
        patientRetrieval: headline(report.patientRetrieval as RetrievalReport | undefined),
        referenceRetrieval: report.referenceRetrieval,
        extraction: report.extraction,
        riskClassification: report.riskClassification,
        groundedAnswers: report.groundedAnswers,
        report: path.relative(process.cwd(), outFile),
      },
      (key, value) => (key === 'perCase' ? undefined : value),
      2
    )
  );
};

main().catch((error: unknown) => {
  console.error('Evaluation failed:', error instanceof Error ? `${error.name}: ${error.message}` : 'unknown error');
  process.exit(1);
});
