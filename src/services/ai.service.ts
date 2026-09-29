import { cleanBiomarkerName } from '../documents/labName';
import { generateText, Output, type LanguageModel, type UserContent } from 'ai';
import { createGoogleGenerativeAI } from '@ai-sdk/google';
import { createOpenAI } from '@ai-sdk/openai';
import { createAnthropic } from '@ai-sdk/anthropic';
import { z } from 'zod';
import { ExtractedEntities, LabValue, LabValueSource, PatientDetails } from '../types';
import { analyzePdf, type AnalyzedPage, type PageStatus } from '../documents/pdfAnalyzer';
import { detectDelimitedTables, type ExtractedTable, type LabTableRow } from '../documents/tables';
import { flagFromRange } from '../documents/ranges';
import { extractIdentity } from '../documents/identity';
import { findLabPages, findText, findTextPages, foldText, type SourcePageText } from '../documents/grounding';
import { EXTRACTION_PROMPT, DATA_HANDLING_RULES, escapeForPrompt, renderPrompt } from '../prompts';
import { logger, errorMeta } from '../utils/logger';
import { recordFailure, type FailureCategory } from '../utils/failures';
import { recordAICall } from '../utils/metrics';

/* ───────────────────────── Errors ───────────────────────── */

export class AIConfigurationError extends Error {
  readonly category: FailureCategory = 'configuration_error';
  constructor(message: string) {
    super(message);
    this.name = 'AIConfigurationError';
  }
}

export interface AttemptRecord {
  provider: ProviderName;
  model: string;
  category: FailureCategory;
  latencyMs: number;
}

/**
 * An AI operation that could not produce a validated result. Callers must treat
 * this as "no clinical conclusion" — never substitute a default risk/summary.
 */
export class AIProcessingError extends Error {
  constructor(
    message: string,
    readonly category: FailureCategory,
    readonly attempts: AttemptRecord[] = []
  ) {
    super(message);
    this.name = 'AIProcessingError';
  }
}

/** Raised by post-generation validation (output parsed but violates clinical/output rules). */
export class OutputValidationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'OutputValidationError';
  }
}

/* ───────────────────────── Providers ───────────────────────── */

export type ProviderName = 'google' | 'openai' | 'anthropic';

export interface ModelCandidate {
  provider: ProviderName;
  modelId: string;
  model: LanguageModel;
}

const envValue = (name: string): string | undefined => {
  const value = process.env[name]?.trim();
  return value ? value : undefined;
};

const envInt = (name: string, fallback: number): number => {
  const value = Number.parseInt(process.env[name] ?? '', 10);
  return Number.isFinite(value) && value > 0 ? value : fallback;
};

const providerKey = (provider: ProviderName): string | undefined =>
  provider === 'google'
    ? envValue('GEMINI_API_KEY') ?? envValue('GOOGLE_API_KEY')
    : provider === 'openai'
      ? envValue('OPENAI_API_KEY')
      : envValue('ANTHROPIC_API_KEY');

/** Builds a single model candidate, e.g. for evaluation runs comparing specific models. */
export const createModelCandidate = (provider: ProviderName, modelId: string): ModelCandidate => {
  const apiKey = providerKey(provider);
  if (!apiKey) throw new AIConfigurationError(`No API key configured for provider ${provider}.`);
  const factory =
    provider === 'google'
      ? createGoogleGenerativeAI({ apiKey })
      : provider === 'openai'
        ? createOpenAI({ apiKey })
        : createAnthropic({ apiKey });
  return { provider, modelId, model: factory(modelId) };
};

/**
 * Ordered list of models to try, built from whichever direct provider keys are configured.
 * Order: Gemini (primary + fallback model) → OpenAI → Anthropic.
 */
export const getModelCandidates = (): ModelCandidate[] => {
  const candidates: ModelCandidate[] = [];

  if (providerKey('google')) {
    // Primary is configurable (preferred: gemini-3.5-flash-lite). The fallback, gemini-2.5-flash-lite,
    // was verified available on this project's account; unavailable models are circuit-broken below.
    const ids = [envValue('GEMINI_MODEL') ?? 'gemini-3.5-flash-lite', envValue('GEMINI_FALLBACK_MODEL') ?? 'gemini-2.5-flash-lite'];
    for (const modelId of [...new Set(ids)]) candidates.push(createModelCandidate('google', modelId));
  }
  if (providerKey('openai')) candidates.push(createModelCandidate('openai', envValue('OPENAI_MODEL') ?? 'gpt-4o-mini'));
  if (providerKey('anthropic')) {
    candidates.push(createModelCandidate('anthropic', envValue('ANTHROPIC_MODEL') ?? 'claude-sonnet-5-5'));
  }

  if (candidates.length === 0) {
    throw new AIConfigurationError('No AI provider configured. Set GEMINI_API_KEY, OPENAI_API_KEY or ANTHROPIC_API_KEY.');
  }
  return candidates;
};

/* ───────────────────────── Circuit breaker ───────────────────────── */

/**
 * Stops hammering a model that is known to be failing:
 *  - 404 (model retired/unknown)   → skipped for AI_BREAKER_UNAVAILABLE_MS (default 30 min)
 *  - 401/403 (key/permission)      → skipped for AI_BREAKER_AUTH_MS (default 10 min)
 *  - 429 (quota/rate limit)        → skipped for Retry-After or AI_BREAKER_RATE_LIMIT_MS (default 60 s)
 * In-process only; resets on restart.
 */
const breaker = new Map<string, { openUntil: number; category: FailureCategory }>();
const breakerKey = (c: Pick<ModelCandidate, 'provider' | 'modelId'>) => `${c.provider}:${c.modelId}`;

export const resetCircuitBreakers = (): void => breaker.clear();

export const getOpenCircuits = (): { model: string; category: FailureCategory; openForMs: number }[] =>
  [...breaker].filter(([, v]) => v.openUntil > Date.now()).map(([model, v]) => ({ model, category: v.category, openForMs: v.openUntil - Date.now() }));

const statusOf = (error: unknown): { status?: number; retryAfterMs?: number } => {
  const e = error as { statusCode?: number; lastError?: unknown; responseHeaders?: Record<string, string> } | undefined;
  if (!e || typeof e !== 'object') return {};
  if (typeof e.statusCode === 'number') {
    const retryAfter = Number(e.responseHeaders?.['retry-after']);
    return { status: e.statusCode, ...(Number.isFinite(retryAfter) && retryAfter > 0 ? { retryAfterMs: retryAfter * 1000 } : {}) };
  }
  return e.lastError ? statusOf(e.lastError) : {};
};

const tripBreaker = (candidate: ModelCandidate, error: unknown): void => {
  const { status, retryAfterMs } = statusOf(error);
  let ms: number | undefined;
  let category: FailureCategory | undefined;
  if (status === 404) [ms, category] = [envInt('AI_BREAKER_UNAVAILABLE_MS', 30 * 60_000), 'model_unavailable'];
  else if (status === 401 || status === 403) [ms, category] = [envInt('AI_BREAKER_AUTH_MS', 10 * 60_000), 'provider_failure'];
  else if (status === 429) [ms, category] = [retryAfterMs ?? envInt('AI_BREAKER_RATE_LIMIT_MS', 60_000), 'provider_rate_limited'];
  if (ms !== undefined && category) {
    breaker.set(breakerKey(candidate), { openUntil: Date.now() + ms, category });
    logger.warn('ai.circuit_open', { provider: candidate.provider, model: candidate.modelId, category, openForMs: ms });
  }
};

const isOpen = (candidate: ModelCandidate): boolean => (breaker.get(breakerKey(candidate))?.openUntil ?? 0) > Date.now();

/* ───────────────────────── Fallback + retry policy ───────────────────────── */

/**
 * Retry policy (single layer per concern, bounded):
 * - Transient provider errors (429/5xx/network) are retried by the AI SDK at most
 *   AI_MAX_RETRIES times (default 1) per model.
 * - Timeouts, schema/validation failures and non-retryable 4xx errors are never
 *   retried on the same model; the next fallback model is tried instead.
 * - At most AI_MAX_MODEL_ATTEMPTS models are tried, within an overall
 *   AI_TOTAL_BUDGET_MS deadline. Configuration errors stop immediately.
 */
export const getAIPolicy = () => ({
  perCallTimeoutMs: envInt('AI_TIMEOUT_MS', 45_000),
  totalBudgetMs: envInt('AI_TOTAL_BUDGET_MS', 120_000),
  maxModelAttempts: envInt('AI_MAX_MODEL_ATTEMPTS', 3),
  maxRetries: Number.isFinite(Number.parseInt(process.env.AI_MAX_RETRIES ?? '', 10))
    ? Math.max(0, Math.min(3, Number.parseInt(process.env.AI_MAX_RETRIES as string, 10)))
    : 1,
});

const SCHEMA_ERROR_NAMES = new Set([
  'AI_NoObjectGeneratedError',
  'AI_NoOutputGeneratedError',
  'AI_TypeValidationError',
  'AI_JSONParseError',
  'NoObjectGeneratedError',
  'NoOutputGeneratedError',
  'TypeValidationError',
  'JSONParseError',
  'ZodError',
  'OutputValidationError',
]);

/** Maps an error from a model call onto the failure taxonomy. */
export const classifyAIError = (error: unknown): FailureCategory => {
  if (error instanceof AIConfigurationError) return 'configuration_error';
  if (!(error instanceof Error)) return 'provider_failure';
  const withCause = error as Error & { lastError?: unknown; cause?: unknown };
  if (error.name === 'AI_RetryError' || error.name === 'RetryError') {
    return withCause.lastError ? classifyAIError(withCause.lastError) : 'provider_failure';
  }
  if (error.name === 'AbortError' || error.name === 'TimeoutError' || /aborted|timed? ?out/i.test(error.name)) {
    return 'provider_timeout';
  }
  if (SCHEMA_ERROR_NAMES.has(error.name)) return 'schema_validation_failure';
  const { status } = statusOf(error);
  if (status === 429) return 'provider_rate_limited';
  if (status === 404) return 'model_unavailable';
  if (withCause.cause instanceof Error && withCause.cause !== error) {
    const nested = classifyAIError(withCause.cause);
    if (nested !== 'provider_failure') return nested;
  }
  return 'provider_failure';
};

export interface CallContext {
  candidate: ModelCandidate;
  abortSignal: AbortSignal;
  maxRetries: number;
}

export interface TokenUsage {
  inputTokens?: number;
  outputTokens?: number;
  totalTokens?: number;
}

export interface ModelResult<T> {
  value: T;
  provider: ProviderName;
  model: string;
  attempts: number;
  fallbackUsed: boolean;
  usage?: TokenUsage;
}

/**
 * Runs `run` against candidates in order until one produces a validated result.
 * Logs metadata only (operation, provider, model, latency, category, token usage).
 */
export const withModelFallback = async <T>(
  operation: string,
  run: (ctx: CallContext) => Promise<{ value: T; usage?: TokenUsage }>,
  candidates: ModelCandidate[] = getModelCandidates()
): Promise<ModelResult<T>> => {
  const policy = getAIPolicy();
  if (candidates.length === 0) {
    throw new AIConfigurationError(`No compatible AI model configured for ${operation}.`);
  }
  const selected = candidates.filter((c) => !isOpen(c)).slice(0, policy.maxModelAttempts);
  if (selected.length === 0) {
    const category = breaker.get(breakerKey(candidates[0]))?.category ?? 'provider_failure';
    recordFailure(category, { operation, reason: 'all_circuits_open' });
    throw new AIProcessingError(`All configured AI models are temporarily unavailable for ${operation}.`, category, []);
  }

  const deadline = Date.now() + policy.totalBudgetMs;
  const attempts: AttemptRecord[] = [];

  for (const [index, candidate] of selected.entries()) {
    const remaining = deadline - Date.now();
    if (remaining <= 0) break;

    const startedAt = Date.now();
    try {
      const { value, usage } = await run({
        candidate,
        abortSignal: AbortSignal.timeout(Math.min(policy.perCallTimeoutMs, remaining)),
        maxRetries: policy.maxRetries,
      });
      recordAICall({
        operation,
        provider: candidate.provider,
        model: candidate.modelId,
        status: 'ok',
        latencyMs: Date.now() - startedAt,
        fallbackUsed: index > 0,
        inputTokens: usage?.inputTokens,
        outputTokens: usage?.outputTokens,
      });
      logger.info('ai.call', {
        operation,
        provider: candidate.provider,
        model: candidate.modelId,
        attempt: index + 1,
        fallbackUsed: index > 0,
        latencyMs: Date.now() - startedAt,
        status: 'ok',
        inputTokens: usage?.inputTokens,
        outputTokens: usage?.outputTokens,
        totalTokens: usage?.totalTokens,
      });
      return {
        value,
        provider: candidate.provider,
        model: candidate.modelId,
        attempts: index + 1,
        fallbackUsed: index > 0,
        usage,
      };
    } catch (error) {
      const category = classifyAIError(error);
      const latencyMs = Date.now() - startedAt;
      tripBreaker(candidate, error);
      attempts.push({ provider: candidate.provider, model: candidate.modelId, category, latencyMs });
      recordAICall({
        operation,
        provider: candidate.provider,
        model: candidate.modelId,
        status: 'error',
        latencyMs,
        fallbackUsed: index > 0,
        failureCategory: category,
      });
      logger.warn('ai.call', {
        operation,
        provider: candidate.provider,
        model: candidate.modelId,
        attempt: index + 1,
        fallbackUsed: index > 0,
        latencyMs,
        status: 'error',
        failureCategory: category,
        ...errorMeta(error),
      });
      if (category === 'configuration_error') throw error;
    }
  }

  // Report the dominant cause when every attempt failed the same way.
  const categories = new Set(attempts.map((a) => a.category));
  const category: FailureCategory =
    attempts.length === 0
      ? 'provider_timeout'
      : categories.size === 1
        ? attempts[0].category
        : 'fallback_exhausted';
  recordFailure(category, { operation, attempts: attempts.length });
  throw new AIProcessingError(`All configured AI models failed for ${operation}.`, category, attempts);
};

const toUsage = (usage: { inputTokens?: number; outputTokens?: number; totalTokens?: number } | undefined): TokenUsage | undefined =>
  usage ? { inputTokens: usage.inputTokens, outputTokens: usage.outputTokens, totalTokens: usage.totalTokens } : undefined;

/** Plain-text generation with model fallback. */
export const generateTextWithFallback = async (
  operation: string,
  prompt: string,
  temperature = 0.2
): Promise<string> => {
  const result = await withModelFallback(operation, async ({ candidate, abortSignal, maxRetries }) => {
    const { text, usage } = await generateText({ model: candidate.model, prompt, temperature, abortSignal, maxRetries });
    if (!text.trim()) throw new OutputValidationError('Empty model response');
    return { value: text, usage: toUsage(usage) };
  });
  return result.value;
};

/** Structured generation: output is parsed against `schema` by the SDK, then by `postValidate`. */
export const generateStructured = async <S extends z.ZodType, R>(options: {
  operation: string;
  schema: S;
  system: string;
  messages?: { role: 'user'; content: UserContent }[];
  prompt?: string;
  temperature: number;
  postValidate: (output: z.infer<S>) => R;
  candidates?: ModelCandidate[];
}): Promise<ModelResult<R>> =>
  withModelFallback(
    options.operation,
    async ({ candidate, abortSignal, maxRetries }) => {
      const common = {
        model: candidate.model,
        system: options.system,
        output: Output.object({ schema: options.schema }),
        temperature: options.temperature,
        abortSignal,
        maxRetries,
      };
      const result = options.messages
        ? await generateText({ ...common, messages: options.messages })
        : await generateText({ ...common, prompt: options.prompt ?? '' });
      // Re-validate defensively: never trust unvalidated output past this point.
      const parsed = options.schema.safeParse(result.output);
      if (!parsed.success) throw new OutputValidationError('Model output failed schema validation');
      return { value: options.postValidate(parsed.data), usage: toUsage(result.usage) };
    },
    options.candidates
  );

/* ───────────────────────── Extraction (JOB A: document understanding) ───────────────────────── */

export const ExtractionSchema = z.object({
  patientDetails: z.object({
    name: z.string().max(200),
    age: z.number().min(0).max(150).nullable(),
    gender: z.string().max(50),
  }),
  symptoms: z.array(z.string().max(500)).max(100),
  medicines: z.array(z.string().max(500)).max(100),
  labValues: z
    .array(
      z.object({
        name: z.string().max(200),
        value: z.string().max(100),
        unit: z.string().max(50),
        normalRange: z.string().max(100),
        isAbnormal: z.boolean(),
        date: z.string().max(50),
      })
    )
    .max(200),
  dates: z.array(z.string().max(100)).max(20),
  diagnosesMentioned: z.array(z.string().max(300)).max(50),
  measurements: z.array(z.object({ name: z.string().max(100), value: z.string().max(100), unit: z.string().max(50) })).max(50),
  /** Information visible only in images/charts/graphics of attached files, each prefixed with its page. */
  imagingFindings: z.array(z.string().max(1000)).max(30),
  /** Risk scores/categories printed in the document itself, copied verbatim. */
  reportRiskScores: z.array(z.object({ name: z.string().max(200), result: z.string().max(300) })).max(20),
  doctorNotes: z.string().max(10_000),
  extractionLimitations: z.array(z.string().max(300)).max(20),
  /**
   * ONLY content that is not in the document's text layer (read from images, scans or audio).
   * The application supplies the document's own text verbatim; the model must not rewrite it.
   */
  fullNarrative: z.string(),
});

type RawExtraction = z.infer<typeof ExtractionSchema>;

export interface ReportDocumentSummary {
  fileIndex: number;
  filename: string;
  mimeType: string;
  sourceType: SourceType;
  pageCount?: number;
  pages?: { page: number; status: PageStatus; imageCount: number; tableCount: number }[];
  tableCount: number;
  notes: string[];
}

export interface ReportTableRow extends LabTableRow {
  fileIndex: number;
  filename: string;
}

/** The document's own text for one page/file, exactly as extracted from the file (never AI-written). */
export interface ReportPage {
  filename: string;
  page?: number;
  status?: PageStatus;
  text: string;
}

/** Comprehensive structured report (Job A output) shown on the dashboard. */
export interface StructuredReport {
  documents: ReportDocumentSummary[];
  /** Verbatim document text, page by page. */
  pages: ReportPage[];
  tables: ReportTableRow[];
  dates: string[];
  diagnosesMentioned: string[];
  measurements: { name: string; value: string; unit: string }[];
  /** Risk scores/categories the document itself prints (verified against the text). */
  reportRiskScores: { name: string; result: string }[];
  imagingFindings: string[];
  /**
   * Items the model read from images/scans that could NOT be found in the document's text layer.
   * Shown separately, clearly labelled, and excluded from the verified data and the assessment.
   */
  unverifiedFromImages: string[];
  doctorNotes: string;
  extractionLimitations: string[];
}

export interface ExtractionResult extends ExtractedEntities {
  patientDetails: PatientDetails;
  report: StructuredReport;
  /** True when the source text exceeded the prompt budget and was truncated. */
  truncated?: boolean;
  /** Normalised per-file representation (internal; used for indexing, never returned to clients). */
  sourceDocuments?: NormalizedDocument[];
  /**
   * Text that ONLY the model produced (read from images/scans/audio). Internal: this — not the document's
   * own verbatim text — is what gets indexed as an "AI transcription" document.
   */
  modelReadText?: string;
  model?: { provider: ProviderName; model: string; fallbackUsed: boolean };
}

export type SourceType = 'pdf' | 'image' | 'text' | 'audio';

/** Common internal representation every input type converges to before extraction/indexing. */
export interface NormalizedDocument {
  fileIndex: number;
  filename: string;
  mimeType: string;
  sourceType: SourceType;
  pages: AnalyzedPage[];
  /** Plain text for text inputs (PDF text lives in pages). */
  text?: string;
  tables: ExtractedTable[];
  /** Content that can only be read by the multimodal model (images, scanned pages, audio). */
  needsModelReading: boolean;
  /** Explicit limitations — nothing is silently dropped. */
  notes: string[];
}

const cleanStrings = (values: unknown[]): string[] =>
  values.filter((v): v is string => typeof v === 'string').map((v) => v.trim()).filter(Boolean);

const matchFirst = (text: string, patterns: RegExp[]): string | undefined => {
  for (const pattern of patterns) {
    const value = text.match(pattern)?.[1]?.trim();
    if (value) return value;
  }
  return undefined;
};

const ABNORMAL_FLAGS = new Set(['High', 'Low', 'Critical', 'Abnormal']);

const sameTest = (a: string, b: string): boolean => {
  const n = (s: string) => s.toLowerCase().replace(/[^a-z0-9]/g, '');
  const x = n(a);
  const y = n(b);
  return Boolean(x && y) && (x === y || (Math.min(x.length, y.length) >= 3 && (x.includes(y) || y.includes(x))));
};

/** Numbers printed in a reference range must all appear on the page for the range to be trusted. */
const rangeIsPrinted = (range: string, pageTexts: SourcePageText[]): boolean => {
  const numbers = range.match(/-?\d+(?:[.,]\d+)?/g)?.map((n) => n.replace(',', '.')) ?? [];
  if (numbers.length === 0) return false;
  return pageTexts.some((p) => {
    const folded = ` ${foldText(p.text)} `;
    return numbers.every((n) => folded.includes(` ${foldText(n)} `));
  });
};

/** Local (non-model) text available for verification: PDF page text, text files, and the clinician's own input. */
const collectSourcePages = (docs: NormalizedDocument[], typedText: string): SourcePageText[] => {
  const pages: SourcePageText[] = [];
  for (const d of docs) {
    for (const p of d.pages) if (p.text.trim()) pages.push({ filename: d.filename, page: p.page, text: p.text });
    if (d.text?.trim()) pages.push({ filename: d.filename, text: d.text });
  }
  if (typedText.trim()) pages.push({ filename: 'typed input', text: typedText });
  return pages;
};

/** The document's own text, page by page, exactly as extracted from the files. */
const buildVerbatimPages = (docs: NormalizedDocument[]): ReportPage[] => {
  const out: ReportPage[] = [];
  for (const d of docs) {
    for (const p of d.pages) out.push({ filename: d.filename, page: p.page, status: p.status, text: p.text });
    if (d.text?.trim()) out.push({ filename: d.filename, text: d.text });
  }
  return out;
};

const verbatimText = (pages: ReportPage[]): string =>
  pages
    .filter((p) => p.text.trim())
    .map((p) => `--- ${p.filename}${p.page !== undefined ? ` · page ${p.page}` : ''} ---\n${p.text.trim()}`)
    .join('\n\n');

const pagesLabel = (hits: SourcePageText[]): LabValueSource | undefined => {
  if (hits.length === 0) return undefined;
  const files = new Set(hits.map((h) => h.filename));
  const first = hits[0];
  const pages = [...new Set(hits.map((h) => h.page).filter((p): p is number => p !== undefined))].sort((a, b) => a - b);
  // `page` is claimed only when the evidence points to exactly one page; otherwise all matching pages are listed.
  return {
    method: 'text_match',
    filename: first.filename,
    ...(pages.length === 1 && files.size === 1 ? { page: pages[0] } : {}),
    ...(pages.length > 1 && files.size === 1 ? { pages } : {}),
  };
};

/**
 * Turns the model's structured output into VERIFIED data.
 *
 *  - The document's text is never AI-written: `rawText`/`report.pages` are the file's own text.
 *  - Patient name/age/gender come from labelled fields in the text when present (authoritative).
 *  - Every model-extracted item must be found in the document text. Ungrounded items are dropped
 *    (text-only inputs) or, when the model may legitimately have read images, moved to
 *    `report.unverifiedFromImages` — never merged into verified data or used for the assessment.
 *  - Abnormal flags are computed from the range PRINTED in the report, not taken from the model.
 *  - Deterministic table rows are authoritative and carry exact page/table provenance.
 */
export const normalizeExtraction = (
  raw: RawExtraction,
  sourceText: string,
  docs: NormalizedDocument[] = [],
  typedText = ''
): ExtractionResult => {
  let pages = collectSourcePages(docs, typedText);
  // Callers without documents (unit tests / plain text) verify against the supplied source text.
  if (pages.length === 0 && docs.length === 0 && sourceText.trim()) pages = [{ filename: 'input', text: sourceText }];
  const hasLocalText = pages.length > 0;
  const hasModelOnly = docs.some((d) => d.needsModelReading);
  const dropped: string[] = [];
  const unverified: string[] = [];

  /** Keeps an item only if it is in the document text; otherwise drops it or files it as unverified image content. */
  const verify = <T>(items: T[], label: string, text: (item: T) => string, find: (item: T) => SourcePageText[]): T[] => {
    if (!hasLocalText) return items;
    const kept: T[] = [];
    for (const item of items) {
      if (find(item).length > 0) kept.push(item);
      else if (hasModelOnly) unverified.push(`${label}: ${text(item)}`);
      else dropped.push(`${label}: ${text(item)}`);
    }
    return kept;
  };

  /* ── identity: labelled fields in the text are authoritative ── */
  const identity = extractIdentity(pages.map((p) => ({ page: p.page, text: p.text })));
  const modelName = raw.patientDetails.name.trim();
  let name = identity.name ?? '';
  if (!name && modelName) {
    if (!hasLocalText || findText(modelName, pages)) name = modelName;
    else if (hasModelOnly) unverified.push(`Patient name: ${modelName}`);
    else dropped.push('Patient name');
  }
  const modelAge = typeof raw.patientDetails.age === 'number' && raw.patientDetails.age > 0 ? Math.round(raw.patientDetails.age) : undefined;
  const ageInText = (n: number) => pages.some((p) => new RegExp(`\\b${n}\\s*(?:y|yr|yrs|years?)?\\b`, 'i').test(p.text));
  const age = identity.age ?? (modelAge !== undefined && (!hasLocalText || ageInText(modelAge)) ? modelAge : undefined);
  const modelGender = raw.patientDetails.gender.trim();
  const gender =
    identity.gender ??
    (modelGender && (!hasLocalText || pages.some((p) => new RegExp(`\\b${modelGender.replace(/[^a-z]/gi, '')}\\b`, 'i').test(p.text))) ? modelGender : '');

  const limitations: string[] = [];
  for (const other of identity.otherNames) {
    limitations.push(
      `Another name appears in the document ("${other.name}"${other.page !== undefined ? `, page ${other.page}` : ''}); the patient name was taken from the "Patient Name" field.`
    );
  }

  /* ── deterministic table rows (authoritative) ── */
  const tableRows: ReportTableRow[] = docs.flatMap((d) =>
    d.tables.flatMap((t) => t.labRows.map((row) => ({ ...row, fileIndex: d.fileIndex, filename: d.filename })))
  );
  const labValues: LabValue[] = tableRows.map((row) => {
    const computed = row.flag ? undefined : flagFromRange(row.result, row.referenceRange);
    const flag = row.flag ?? computed;
    return {
      name: cleanBiomarkerName(row.test),
      value: row.result,
      unit: row.unit ?? '',
      ...(row.referenceRange ? { normalRange: row.referenceRange } : {}),
      isAbnormal: flag ? ABNORMAL_FLAGS.has(flag) : false,
      ...(flag ? { flag } : {}),
      flagSource: row.flag ? ('table' as const) : computed ? ('printed_range' as const) : ('model' as const),
      ...(row.date ? { date: row.date } : {}),
      source: { method: 'table' as const, filename: row.filename, tableId: row.tableId, ...(row.page !== undefined ? { page: row.page } : {}) },
    };
  });

  /* ── model-extracted labs: verified against the text, flags computed from the printed range ── */
  const candidateLabs = raw.labValues
    .map((lab) => ({ ...lab, name: cleanBiomarkerName(lab.name), value: lab.value.trim(), unit: lab.unit.trim(), normalRange: lab.normalRange.trim(), date: lab.date.trim() }))
    .filter((lab) => lab.name && lab.value && !tableRows.some((row) => sameTest(row.test, lab.name)));
  const verifiedLabs = verify(
    candidateLabs,
    'Lab',
    (l) => `${l.name} ${l.value}${l.unit ? ` ${l.unit}` : ''}`,
    (l) => findLabPages(l.name, l.value, pages)
  );
  for (const lab of verifiedLabs) {
    const hits = hasLocalText ? findLabPages(lab.name, lab.value, pages) : [];
    const rangeOk = lab.normalRange && (!hasLocalText || rangeIsPrinted(lab.normalRange, hits.length ? hits : pages));
    const range = rangeOk ? lab.normalRange : '';
    const computed = flagFromRange(lab.value, range);
    labValues.push({
      name: lab.name,
      value: lab.value,
      unit: lab.unit,
      ...(range ? { normalRange: range } : {}),
      isAbnormal: computed ? computed !== 'Normal' : lab.isAbnormal,
      ...(computed ? { flag: computed } : {}),
      flagSource: computed ? 'printed_range' : 'model',
      ...(lab.date ? { date: lab.date } : {}),
      // Prefer the pages where the printed range is also present (the detailed result), when that narrows it down.
      source: pagesLabel(range && hits.length > 1 && hits.some((h) => rangeIsPrinted(range, [h])) ? hits.filter((h) => rangeIsPrinted(range, [h])) : hits) ?? { method: 'model' },
    });
  }

  /* ── other fields ── */
  const findSimple = (s: string) => (hasLocalText ? findTextPages(s, pages) : []);
  const symptoms = verify(cleanStrings(raw.symptoms), 'Symptom', (s) => s, findSimple);
  const medicines = verify(cleanStrings(raw.medicines), 'Medicine', (s) => s, findSimple);
  const diagnosesMentioned = verify(cleanStrings(raw.diagnosesMentioned), 'Diagnosis', (s) => s, findSimple);
  const dates = verify(cleanStrings(raw.dates), 'Date', (s) => s, findSimple);
  const measurements = verify(
    raw.measurements.map((m) => ({ name: m.name.trim(), value: m.value.trim(), unit: m.unit.trim() })).filter((m) => m.name && m.value),
    'Measurement',
    (m) => `${m.name} ${m.value}${m.unit ? ` ${m.unit}` : ''}`,
    (m) => findLabPages(m.name, m.value, pages)
  );
  const reportRiskScores = verify(
    raw.reportRiskScores.map((r) => ({ name: r.name.trim(), result: r.result.trim() })).filter((r) => r.name && r.result),
    'Risk score',
    (r) => `${r.name}: ${r.result}`,
    (r) => (hasLocalText ? findTextPages(r.name, pages).filter((p) => findTextPages(r.result, [p]).length > 0) : [])
  );
  const doctorNotesRaw = raw.doctorNotes.trim();
  const doctorNotes = doctorNotesRaw
    ? verify([doctorNotesRaw], 'Doctor notes', (s) => s.slice(0, 80), findSimple)[0] ?? ''
    : '';

  if (dropped.length > 0) {
    limitations.push(`${dropped.length} item(s) proposed by the AI were not found in the document text and were discarded (${dropped.slice(0, 5).join('; ')}${dropped.length > 5 ? '; …' : ''}).`);
  }

  const verbatimPages = buildVerbatimPages(docs);
  const localVerbatim = verbatimText(verbatimPages);
  const typed = typedText.trim();
  const imageNarrative = raw.fullNarrative.trim();
  // The document's own words come first. AI-transcribed content is appended, labelled, and only for content the text layer lacks.
  const rawText =
    hasLocalText || typed
      ? [typed, localVerbatim, hasModelOnly && imageNarrative ? `--- Read by AI from images/scans (not verified against the text layer) ---\n${imageNarrative}` : '']
          .filter(Boolean)
          .join('\n\n')
      : imageNarrative;

  const report: StructuredReport = {
    documents: docs.map((d) => ({
      fileIndex: d.fileIndex,
      filename: d.filename,
      mimeType: d.mimeType,
      sourceType: d.sourceType,
      ...(d.sourceType === 'pdf'
        ? {
            pageCount: d.pages.length,
            pages: d.pages.map((p) => ({ page: p.page, status: p.status, imageCount: p.imageCount, tableCount: p.tables.length })),
          }
        : {}),
      tableCount: d.tables.length,
      notes: d.notes,
    })),
    pages: verbatimPages,
    tables: tableRows,
    dates,
    diagnosesMentioned,
    measurements,
    reportRiskScores,
    imagingFindings: cleanStrings(raw.imagingFindings),
    unverifiedFromImages: unverified,
    doctorNotes,
    extractionLimitations: [...docs.flatMap((d) => d.notes), ...limitations, ...cleanStrings(raw.extractionLimitations)],
  };

  return {
    patientDetails: { name, ...(age !== undefined ? { age } : {}), gender },
    symptoms,
    medicines,
    labValues,
    rawText,
    modelReadText: hasLocalText || typed ? (hasModelOnly ? imageNarrative : '') : imageNarrative,
    report,
  };
};

const PDF_MIME = 'application/pdf';

/** Back-compat helper: per-page text of a PDF. */
export const extractPdfPages = async (buffer: Buffer): Promise<string[]> => (await analyzePdf(buffer)).pages.map((p) => p.text);

/** Upper bound on text sent for extraction (~15k tokens). Exceeding it truncates with an explicit flag. */
export const getMaxExtractionChars = (): number => envInt('AI_MAX_EXTRACTION_CHARS', 60_000);

/**
 * Normalises every uploaded file into the common representation and collects the
 * native media parts the multimodal model must read (images, scanned PDFs, audio).
 */
export const collectSources = async (
  mediaFiles: Express.Multer.File[]
): Promise<{ documents: NormalizedDocument[]; mediaParts: Exclude<UserContent, string>; hasAudio: boolean }> => {
  const documents: NormalizedDocument[] = [];
  const mediaParts: Exclude<UserContent, string> = [];
  let hasAudio = false;

  for (const [fileIndex, file] of mediaFiles.entries()) {
    const base = { fileIndex, filename: file.originalname || `file-${fileIndex + 1}`, mimeType: file.mimetype, pages: [] as AnalyzedPage[], tables: [] as ExtractedTable[], notes: [] as string[] };

    if (file.mimetype === PDF_MIME) {
      let pages: AnalyzedPage[] = [];
      try {
        pages = (await analyzePdf(file.buffer)).pages;
      } catch (error) {
        logger.warn('extraction.pdf_parse_failed', { fileIndex, sizeBytes: file.size, ...errorMeta(error) });
        recordFailure('extraction_failure', { operation: 'analyze_pdf', fileIndex });
        base.notes.push(`${base.filename}: PDF structure could not be parsed locally; the file was sent to the vision model instead.`);
      }
      const unreadable = pages.filter((p) => p.status === 'scanned' || p.status === 'mixed' || p.status === 'empty');
      const needsModelReading = pages.length === 0 || unreadable.some((p) => p.status !== 'empty' || p.imageCount > 0) || pages.every((p) => p.status === 'empty');
      for (const p of pages) {
        for (const t of p.tables) {
          if (t.labRows.length === 0 && t.rows.length > 0) {
            recordFailure('table_extraction_failure', { operation: 'analyze_pdf', page: p.page });
            base.notes.push(`${base.filename} page ${p.page}: a table was detected but its rows could not be mapped to test/result columns; kept as plain text.`);
          }
        }
        if (p.status === 'scanned') base.notes.push(`${base.filename} page ${p.page}: no text layer (scanned); read by the vision model.`);
        if (p.status === 'mixed') base.notes.push(`${base.filename} page ${p.page}: contains ${p.imageCount} embedded image(s); image content read by the vision model.`);
        if (p.status === 'empty') base.notes.push(`${base.filename} page ${p.page}: no extractable content.`);
      }
      if (needsModelReading) mediaParts.push({ type: 'file', data: file.buffer, mediaType: PDF_MIME });
      documents.push({ ...base, sourceType: 'pdf', pages, tables: pages.flatMap((p) => p.tables), needsModelReading });
    } else if (file.mimetype === 'text/plain') {
      const text = file.buffer.toString('utf8').trim();
      if (!text) base.notes.push(`${base.filename}: text file is empty.`);
      documents.push({ ...base, sourceType: 'text', text, tables: detectDelimitedTables(text, undefined, `f${fileIndex}-`), needsModelReading: false });
    } else if (file.mimetype.startsWith('image/')) {
      mediaParts.push({ type: 'image', image: file.buffer, mediaType: file.mimetype });
      documents.push({ ...base, sourceType: 'image', needsModelReading: true, notes: [`${base.filename}: image read by the vision model; findings are model-transcribed, not locally verified.`] });
    } else if (file.mimetype.startsWith('audio/')) {
      hasAudio = true;
      mediaParts.push({ type: 'file', data: file.buffer, mediaType: file.mimetype });
      documents.push({ ...base, sourceType: 'audio', needsModelReading: true, notes: [`${base.filename}: audio transcribed by the speech-capable model.`] });
    }
  }
  return { documents, mediaParts, hasAudio };
};

/** Model-facing rendering of the normalised documents (tables kept row-intact with | delimiters). */
export const formatDocumentsForModel = (documents: NormalizedDocument[]): string =>
  documents
    .map((d) => {
      if (d.sourceType === 'pdf') {
        if (d.pages.length === 0) return `[Document ${d.fileIndex + 1}: ${d.filename} — see attached PDF]`;
        return d.pages
          .map((p) =>
            p.status === 'scanned' || (p.status === 'empty' && d.needsModelReading)
              ? `[Document ${d.fileIndex + 1}: ${d.filename}, page ${p.page} — NO TEXT LAYER, read from the attached PDF]`
              : `[Document ${d.fileIndex + 1}: ${d.filename}, page ${p.page}]\n${p.text}`
          )
          .join('\n\n');
      }
      if (d.sourceType === 'text') return d.text ? `[Document ${d.fileIndex + 1}: ${d.filename}]\n${d.text}` : '';
      return `[Document ${d.fileIndex + 1}: ${d.filename} — ${d.sourceType} attached]`;
    })
    .filter(Boolean)
    .join('\n\n');

/**
 * JOB A — extract structured clinical entities from free text and/or uploaded files in
 * ONE structured model call. Local parsing (PDF text layer, tables) happens first; only
 * content that cannot be read locally is sent as native media.
 * Throws AIProcessingError / AIConfigurationError; never returns fabricated data.
 */
export const extractMedicalData = async (
  rawText: string,
  mediaFiles: Express.Multer.File[] = [],
  options: { candidates?: ModelCandidate[] } = {}
): Promise<ExtractionResult> => {
  const { documents, mediaParts, hasAudio } = await collectSources(mediaFiles);

  const sections: string[] = [];
  if (rawText.trim()) sections.push(`[Clinician Input]\n${rawText.trim()}`);
  const formatted = formatDocumentsForModel(documents);
  if (formatted) sections.push(formatted);

  const hasText = sections.join('').replace(/\[[^\]]*\]/g, '').trim().length > 0;
  if (!hasText && mediaParts.length === 0) {
    recordFailure('no_input', { operation: 'extract_medical_data' });
    throw new AIProcessingError('No medical data could be extracted from the provided input.', 'no_input');
  }

  let sourceText = sections.join('\n\n');
  const maxChars = getMaxExtractionChars();
  const truncated = sourceText.length > maxChars;
  if (truncated) {
    logger.warn('extraction.input_truncated', { originalChars: sourceText.length, maxChars });
    sourceText = `${sourceText.slice(0, maxChars)}\n[... input truncated: exceeded ${maxChars} characters ...]`;
  }

  let candidates = options.candidates ?? getModelCandidates();
  if (hasAudio) {
    candidates = candidates.filter((c) => c.provider === 'google');
    if (candidates.length === 0) {
      throw new AIConfigurationError('Audio processing requires a Gemini model (set GEMINI_API_KEY).');
    }
  }

  const { system, prompt } = await renderPrompt(EXTRACTION_PROMPT, {
    data_rules: DATA_HANDLING_RULES,
    material: escapeForPrompt(sourceText || '(see attached files)'),
  });

  const result = await generateStructured({
    operation: 'extract_medical_data',
    schema: ExtractionSchema,
    system,
    messages: [{ role: 'user', content: [{ type: 'text', text: prompt }, ...mediaParts] }],
    temperature: 0.1,
    postValidate: (output) => normalizeExtraction(output, sourceText, documents, rawText),
    candidates,
  });

  if (truncated) result.value.report.extractionLimitations.push(`Input exceeded ${maxChars} characters and was truncated before extraction.`);

  return {
    ...result.value,
    ...(truncated ? { truncated: true } : {}),
    sourceDocuments: documents,
    model: { provider: result.provider, model: result.model, fallbackUsed: result.fallbackUsed },
  };
};

/* ───────────────────────── Shared risk vocabulary ───────────────────────── */

export const DISCLAIMER = 'Generated by AI - Subject to Physician Review';
