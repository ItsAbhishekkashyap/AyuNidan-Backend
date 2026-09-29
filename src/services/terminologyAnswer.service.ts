import { z } from 'zod';
import { generateStructured, type ProviderName } from './ai.service';
import { retrieveTerminologyEvidence, type TerminologyRetrieval } from './terminology.service';
import { assembleEvidence, formatTerminologyEvidence, inlineCitationIds, stripInvalidInlineCitations, toCitation, validateCitations, type Citation, type Evidence } from '../rag/evidence';
import { DATA_HANDLING_RULES, escapeForPrompt, renderPrompt, TERMINOLOGY_PROMPT } from '../prompts';
import { recordFailure } from '../utils/failures';
import { logger, errorMeta } from '../utils/logger';
import { StageTimer } from '../utils/timing';

/**
 * Terminology answer: retrieve dictionary-style entries (exact term/synonym first, then semantic), have
 * the model explain ONLY those entries, and validate its T# citations. Terminology is not clinical
 * guidance and no patient data is involved. With no matching entry there is no model call and no
 * invented definition — the caller gets an explicit "no matching entry" result.
 */

export const NO_TERMINOLOGY_ANSWER =
  'No matching entry was found in the medical terminology knowledge base, so this term cannot be explained from a source. Please ask your clinician or check a trusted medical source.';

export const TerminologyAnswerSchema = z.object({
  answer: z.string().max(3000),
  terminologyCitations: z.array(z.string().max(10)).max(20),
  uncertainty: z.string().max(1000),
  insufficientContext: z.boolean(),
});

export interface TerminologyAnswer {
  status: 'answered' | 'insufficient_context';
  answer: string;
  uncertainty?: string;
  /** Terminology entries actually cited (built only from retrieved metadata). */
  citations: Citation[];
  retrieval: { status: TerminologyRetrieval['status']; retrieved: number; exactMatches: number; semanticMatches: number };
  model?: { provider: ProviderName; model: string; fallbackUsed: boolean };
}

export const buildTerminologyPrompt = (question: string, evidence: Evidence[]) =>
  renderPrompt(TERMINOLOGY_PROMPT, {
    data_rules: DATA_HANDLING_RULES,
    user_query: escapeForPrompt(question),
    terminology_evidence: formatTerminologyEvidence(evidence),
  });

/** Model call + deterministic validation over already-assembled terminology evidence. */
export const answerFromTerminology = async (
  question: string,
  evidence: Evidence[],
  retrieval: TerminologyAnswer['retrieval'],
  timer = new StageTimer()
): Promise<TerminologyAnswer> => {
  const terms = evidence.filter((e) => e.domain === 'terminology');
  const insufficient = (model?: TerminologyAnswer['model'], uncertainty?: string): TerminologyAnswer => ({
    status: 'insufficient_context',
    answer: NO_TERMINOLOGY_ANSWER,
    ...(uncertainty ? { uncertainty } : {}),
    citations: [],
    retrieval,
    ...(model ? { model } : {}),
  });
  if (terms.length === 0) return insufficient();

  const { system, prompt } = await buildTerminologyPrompt(question, terms);
  const result = await timer.time('generation', () =>
    generateStructured({ operation: 'explain_terminology', schema: TerminologyAnswerSchema, system, prompt, temperature: 0.1, postValidate: (o) => o })
  );
  const output = result.value;
  const model = { provider: result.provider, model: result.model, fallbackUsed: result.fallbackUsed };

  if (output.insufficientContext) {
    recordFailure('insufficient_context', { operation: 'explain_terminology', reason: 'model_reported' });
    return insufficient(model);
  }

  // Structured ids AND ids written inline in the prose are checked against the retrieved terminology only.
  const claimed = [...output.terminologyCitations, ...inlineCitationIds(output.answer)];
  const { valid, invalid } = validateCitations(claimed, terms, 'terminology');
  if (invalid.length > 0) recordFailure('citation_failure', { operation: 'explain_terminology', unknownCitations: invalid.length });
  if (valid.length === 0) {
    recordFailure('citation_failure', { operation: 'explain_terminology', reason: 'uncited_answer' });
    return insufficient(model);
  }
  const allowed = new Set(valid.map((e) => e.evidenceId));
  return {
    status: 'answered',
    answer: stripInvalidInlineCitations(output.answer, allowed).trim(),
    ...(output.uncertainty.trim() ? { uncertainty: output.uncertainty.trim() } : {}),
    citations: valid.map(toCitation),
    retrieval,
    model,
  };
};

/** Retrieve (never throws on outages — reports `unavailable`) then answer. */
export const explainTerminology = async (question: string, timer = new StageTimer()): Promise<TerminologyAnswer> => {
  let retrieved: TerminologyRetrieval;
  try {
    retrieved = await retrieveTerminologyEvidence(question, {}, timer);
  } catch (error) {
    logger.warn('terminology.retrieval_failed', errorMeta(error));
    return {
      status: 'insufficient_context',
      answer: NO_TERMINOLOGY_ANSWER,
      citations: [],
      retrieval: { status: 'unavailable', retrieved: 0, exactMatches: 0, semanticMatches: 0 },
    };
  }
  const evidence = assembleEvidence([], [], undefined, retrieved.evidence);
  return answerFromTerminology(
    question,
    evidence,
    { status: retrieved.status, retrieved: evidence.length, exactMatches: retrieved.exactMatches, semanticMatches: retrieved.semanticMatches },
    timer
  );
};
