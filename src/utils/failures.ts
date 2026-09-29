import { logger } from './logger';

/**
 * Categorised failure taxonomy shared by the AI pipeline, document RAG and the
 * evaluation harness, so failures are measurable rather than opaque 500s.
 */
export const FAILURE_CATEGORIES = [
  'no_input',
  'configuration_error',
  'extraction_failure',
  'schema_validation_failure',
  'provider_failure',
  'provider_timeout',
  'provider_rate_limited',
  'model_unavailable',
  'fallback_exhausted',
  'persistence_failure',
  'no_extractable_text',
  'chunking_failure',
  'embedding_failure',
  'vector_db_failure',
  'retrieval_miss',
  'insufficient_context',
  'invalid_metadata',
  'citation_failure',
  'table_extraction_failure',
  'transcription_failure',
] as const;

export type FailureCategory = (typeof FAILURE_CATEGORIES)[number];

const counts = new Map<FailureCategory, number>();

/** Records a categorised failure (metadata only — never pass clinical content). */
export const recordFailure = (
  category: FailureCategory,
  meta: Record<string, string | number | boolean | undefined> = {}
): void => {
  counts.set(category, (counts.get(category) ?? 0) + 1);
  logger.warn('failure', { category, ...meta });
};

export const getFailureCounts = (): Partial<Record<FailureCategory, number>> => Object.fromEntries(counts);

export const resetFailureCounts = (): void => counts.clear();
