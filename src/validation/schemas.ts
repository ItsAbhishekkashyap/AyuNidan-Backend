import { z } from 'zod';

export const LIMITS = {
  text: 50_000,
  /** Full verbatim text of uploaded reports (all pages) that the clinician reviews/edits before submitting. */
  reportText: 300_000,
  pageText: 60_000,
  listItems: 100,
  itemText: 500,
  labValues: 200,
  searchTerm: 100,
  explainTerm: 100,
  pageSizeMax: 50,
  ragQuestion: 500,
} as const;

const trimmed = (max: number) => z.string().trim().max(max);

/* ── Auth ── */

const email = z.string().trim().toLowerCase().pipe(z.email('A valid email is required').max(254));

export const registerSchema = z.object({
  name: trimmed(100).min(1, 'Name is required'),
  email,
  password: z.string().min(8, 'Password must be at least 8 characters').max(128),
});

export const loginSchema = z.object({
  email,
  password: z.string().min(1, 'Password is required').max(128),
});

export const googleLoginSchema = z.object({
  idToken: z.string().min(1, 'Google ID Token is missing').max(4096),
});

/* ── Common ── */

const objectId = z.string().regex(/^[a-f\d]{24}$/i, 'Invalid ID format');

export const objectIdParamSchema = z.object({ id: objectId });

/* ── Consultations ── */

export const listConsultationsQuerySchema = z.object({
  page: z.coerce.number().int().min(1).max(10_000).default(1),
  limit: z.coerce.number().int().min(1).max(LIMITS.pageSizeMax).default(10),
  search: trimmed(LIMITS.searchTerm).optional(),
});

export const explainQuerySchema = z.object({
  term: trimmed(LIMITS.explainTerm).min(1, 'A medical term must be provided'),
});

const labValueSchema = z.object({
  name: trimmed(200).min(1),
  value: z.union([z.string(), z.number()]).transform(String).pipe(trimmed(100).min(1)),
  unit: trimmed(50).default(''),
  normalRange: trimmed(100).optional(),
  isAbnormal: z.boolean().default(false),
  flag: trimmed(30).optional(),
  flagSource: z.enum(['printed_range', 'table', 'model']).optional(),
  date: trimmed(50).optional(),
  source: z
    .object({
      method: z.enum(['table', 'text_match', 'model']),
      filename: trimmed(255).optional(),
      page: z.number().int().min(1).max(10_000).optional(),
      pages: z.array(z.number().int().min(1).max(10_000)).max(50).optional(),
      tableId: trimmed(50).optional(),
    })
    .optional(),
});

const tableRowSchema = z.object({
  test: trimmed(200),
  result: trimmed(100),
  unit: trimmed(50).optional(),
  referenceRange: trimmed(100).optional(),
  flag: trimmed(30).optional(),
  date: trimmed(50).optional(),
  page: z.number().int().min(1).max(10_000).optional(),
  tableId: trimmed(50),
  rowIndex: z.number().int().min(0).max(10_000),
  fileIndex: z.number().int().min(0).max(20),
  filename: trimmed(255),
  numericResult: z.number().optional(),
});

/** Structured report produced by /uploads (clinician-reviewed); bounded, provenance fields only. */
const reportSchema = z
  .object({
    documents: z
      .array(
        z.object({
          fileIndex: z.number().int().min(0).max(20),
          filename: trimmed(255),
          mimeType: trimmed(100),
          sourceType: z.enum(['pdf', 'image', 'text', 'audio']),
          pageCount: z.number().int().min(0).max(10_000).optional(),
          pages: z
            .array(z.object({ page: z.number().int().min(1), status: z.enum(['text', 'mixed', 'scanned', 'empty']), imageCount: z.number().int().min(0), tableCount: z.number().int().min(0) }))
            .max(500)
            .optional(),
          tableCount: z.number().int().min(0).max(1000),
          notes: z.array(trimmed(300)).max(50),
        })
      )
      .max(10)
      .default([]),
    tables: z.array(tableRowSchema).max(500).default([]),
    /** Verbatim document text per page (as extracted from the file). */
    pages: z
      .array(
        z.object({
          filename: trimmed(255),
          page: z.number().int().min(1).max(10_000).optional(),
          status: z.enum(['text', 'mixed', 'scanned', 'empty']).optional(),
          text: z.string().max(LIMITS.pageText),
        })
      )
      .max(300)
      .default([]),
    reportRiskScores: z.array(z.object({ name: trimmed(200), result: trimmed(300) })).max(20).default([]),
    unverifiedFromImages: z.array(trimmed(400)).max(100).default([]),
    dates: z.array(trimmed(100)).max(20).default([]),
    diagnosesMentioned: z.array(trimmed(300)).max(50).default([]),
    measurements: z.array(z.object({ name: trimmed(100), value: trimmed(100), unit: trimmed(50) })).max(50).default([]),
    imagingFindings: z.array(trimmed(1000)).max(30).default([]),
    doctorNotes: trimmed(10_000).default(''),
    extractionLimitations: z.array(trimmed(300)).max(100).default([]),
  })
  .optional();

const stringList = z.array(trimmed(LIMITS.itemText)).max(LIMITS.listItems).default([])
  .transform((items) => items.filter(Boolean));

export const createConsultationSchema = z
  .object({
    rawText: trimmed(LIMITS.reportText).default(''),
    voiceTranscript: trimmed(LIMITS.text).optional(),
    patientDetails: z
      .object({
        name: trimmed(200).default(''),
        age: z.number().int().min(0).max(150).nullish().transform((v) => (v ? v : undefined)),
        gender: trimmed(50).optional(),
      })
      .optional(),
    symptoms: stringList,
    medicines: stringList,
    labValues: z.array(labValueSchema).max(LIMITS.labValues).default([]),
    /** Documents returned by /uploads to link to this consultation (ownership verified server-side). */
    documentIds: z.array(objectId).max(10).default([]),
    report: reportSchema,
  })
  .refine(
    (body) =>
      body.rawText.length > 0 ||
      Boolean(body.voiceTranscript) ||
      body.symptoms.length > 0 ||
      body.medicines.length > 0 ||
      body.labValues.length > 0,
    { message: 'Provide rawText, a voice transcript or extracted clinical data' }
  );

/* ── Uploads ── */

export const uploadBodySchema = z.object({
  text: trimmed(LIMITS.text).optional(),
  rawText: trimmed(LIMITS.text).optional(),
  symptoms: trimmed(LIMITS.text).optional(),
});

/* ── Documents (RAG) ── */

export const listDocumentsQuerySchema = z.object({
  consultationId: objectId.optional(),
});

export const documentQuerySchema = z.object({
  question: trimmed(LIMITS.ragQuestion).min(3, 'Question must be at least 3 characters'),
  documentIds: z.array(objectId).min(1).max(10).optional(),
  consultationId: objectId.optional(),
  topK: z.number().int().min(1).max(10).optional(),
});
