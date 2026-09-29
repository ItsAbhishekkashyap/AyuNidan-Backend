import fs from 'node:fs';
import path from 'node:path';
import { z } from 'zod';
import { FAILURE_CATEGORIES } from '../utils/failures';

/** Schema for the SYNTHETIC evaluation dataset (eval/synthetic-dataset.json). */

const LabGold = z.object({ name: z.string(), value: z.string(), isAbnormal: z.boolean() });

export const ExtractionCase = z.object({
  id: z.string(),
  input: z.string().min(1),
  expected: z.object({
    name: z.string(),
    age: z.number().nullable(),
    gender: z.string(),
    symptoms: z.array(z.string()),
    medicines: z.array(z.string()),
    labValues: z.array(LabGold),
  }),
});

export const SummaryCase = z.object({
  id: z.string(),
  expectedRiskLevel: z.enum(['low', 'medium', 'high']),
  input: z.record(z.string(), z.unknown()),
});

export const RagDocument = z.object({
  id: z.string(),
  filename: z.string(),
  source: z.enum(['pdf_text', 'text_file', 'ai_transcription']),
  pages: z.array(z.string().min(1)).min(1),
});

export const RagQuestion = z
  .object({
    id: z.string(),
    question: z.string().min(3),
    answerable: z.boolean(),
    expectedDocumentId: z.string().optional(),
    expectedPage: z.number().int().positive().optional(),
    answerContains: z.array(z.string()).optional(),
  })
  .refine((q) => !q.answerable || q.expectedDocumentId, { message: 'answerable questions need expectedDocumentId' });

export const FailureCase = z.object({
  id: z.string(),
  scenario: z.string(),
  expectedCategory: z.enum(FAILURE_CATEGORIES),
});

const TableRowGold = z.object({
  test: z.string(),
  result: z.string(),
  unit: z.string().optional(),
  referenceRange: z.string().optional(),
  flag: z.string().optional(),
  page: z.number().int().optional(),
});

export const TableCase = z.discriminatedUnion('format', [
  z.object({ id: z.string(), format: z.literal('pdf'), page: z.number().int().min(1), header: z.array(z.string()), rows: z.array(z.array(z.string())), expected: z.array(TableRowGold) }),
  z.object({ id: z.string(), format: z.literal('delimited'), text: z.string(), expected: z.array(TableRowGold) }),
]);

export const ReferenceCorpus = z.object({
  _notice: z.string().regex(/SYNTHETIC/),
  sources: z.array(z.object({ sourceId: z.string(), title: z.string(), text: z.string().min(1) })).min(1),
  questions: z
    .array(z.object({ id: z.string(), query: z.string().min(3), answerable: z.boolean(), expectedSourceIds: z.array(z.string()).optional() }))
    .min(1),
});

export const AssessmentCase = z.object({
  id: z.string(),
  expectedRiskLevel: z.enum(['low', 'medium', 'high', 'insufficient_evidence']),
  findings: z.object({
    symptoms: z.array(z.string()),
    medicines: z.array(z.string()),
    labValues: z.array(
      z.object({ name: z.string(), value: z.string(), unit: z.string(), normalRange: z.string().optional(), isAbnormal: z.boolean(), flag: z.string().optional() })
    ),
    rawText: z.string().optional(),
  }),
});

export const Dataset = z
  .object({
    _notice: z.string().regex(/SYNTHETIC/),
    version: z.number(),
    extraction: z.array(ExtractionCase).min(1),
    summary: z.array(SummaryCase).min(1),
    rag: z.object({ documents: z.array(RagDocument).min(1), questions: z.array(RagQuestion).min(1) }),
    tables: z.array(TableCase).default([]),
    referenceCorpus: ReferenceCorpus.optional(),
    assessment: z.array(AssessmentCase).default([]),
    failures: z.array(FailureCase),
  })
  .superRefine((data, ctx) => {
    const docIds = new Set(data.rag.documents.map((d) => d.id));
    for (const q of data.rag.questions) {
      if (q.expectedDocumentId && !docIds.has(q.expectedDocumentId)) {
        ctx.addIssue({ code: 'custom', message: `question ${q.id} references unknown document ${q.expectedDocumentId}` });
      }
      const doc = data.rag.documents.find((d) => d.id === q.expectedDocumentId);
      if (doc && q.expectedPage && (doc.source !== 'pdf_text' || q.expectedPage > doc.pages.length)) {
        ctx.addIssue({ code: 'custom', message: `question ${q.id} has an invalid expectedPage` });
      }
    }
  });

export type Dataset = z.infer<typeof Dataset>;

export const DEFAULT_DATASET_PATH = path.resolve(__dirname, '..', '..', 'eval', 'synthetic-dataset.json');

export const loadDataset = (file = DEFAULT_DATASET_PATH): Dataset => {
  const raw = JSON.parse(fs.readFileSync(file, 'utf8')) as unknown;
  const parsed = Dataset.safeParse(raw);
  if (!parsed.success) {
    throw new Error(`Invalid evaluation dataset: ${parsed.error.issues.map((i) => `${i.path.join('.')}: ${i.message}`).join('; ')}`);
  }
  return parsed.data;
};
