import { AIConfigurationError, AIProcessingError } from '../services/ai.service';
import type { FailureCategory } from '../utils/failures';

/** Maps AI pipeline errors onto HTTP status + a safe, categorised client response. */
export const aiErrorResponse = (
  error: unknown,
  fallbackMessage: string
): { status: number; body: { success: false; error: string; failureCategory: FailureCategory } } => {
  if (error instanceof AIConfigurationError) {
    return { status: 503, body: { success: false, error: 'AI service is not configured', failureCategory: error.category } };
  }
  if (error instanceof AIProcessingError) {
    const status = error.category === 'no_input' ? 422 : 502;
    return { status, body: { success: false, error: fallbackMessage, failureCategory: error.category } };
  }
  return { status: 500, body: { success: false, error: fallbackMessage, failureCategory: 'extraction_failure' } };
};
