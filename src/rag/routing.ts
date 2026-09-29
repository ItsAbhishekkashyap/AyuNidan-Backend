import { extractDefinitionTerm } from '../services/terminology.service';

/**
 * Deterministic question routing (no model call, no vector call).
 *
 *   terminology — a pure definition request about a term ("What does HbA1c mean?"): answered from the
 *                 medical terminology domain only (T# evidence). Explains a word; makes no clinical claim.
 *   clinical    — asks for interpretation, risk, cause, treatment or safety: needs VERIFIED medical
 *                 reference evidence (R#) in addition to the patient's own data; without it the answer is
 *                 "insufficient evidence", never an ungrounded interpretation.
 *   report      — a lookup in the patient's own documents ("what is the haemoglobin value?", "what did
 *                 the report conclude?"): patient/structured evidence (P#/F#), verified references optional.
 */

export type QuestionKind = 'terminology' | 'clinical' | 'report';

export interface QuestionRoute {
  kind: QuestionKind;
  /** The term being defined (terminology route only). */
  term?: string;
}

/** Words that tie a question to THIS patient's data — a definition of "my X" is a report lookup. */
const PATIENT_SPECIFIC = /\b(my|mine|his|her|their|patient|patients|patient's|report|reports|document|documents|uploaded|result|results|value|values|reading|readings|this|these|those|last|latest|current)\b/;

/** Interpretation / risk / treatment intent. */
const CLINICAL_INTENT =
  /\b(risk|risks|risky|danger|dangerous|serious|severe|severity|urgent|urgency|emergency|life[- ]threatening|treat|treated|treatment|treatments|therapy|cure|prognosis|complication|complications|should i|do i need|is it safe|what causes|cause of|causes of|guideline|guidelines|recommend|recommended|dose|dosage|prescribe|worried|worry|prevent|prevention|manage|management|side effects?)\b/;

/**
 * "What was the potassium?" / "What is the haemoglobin level?" look up a value in THIS report even though
 * they start like a definition question. Only "what is/are <term>" without "the" (or an explicit
 * "meaning/definition of") is treated as a definition request.
 */
const REPORT_LOOKUP = /^\s*(?:what|whats|what's)\s+(?:(?:was|were)\b|(?:is|are)\s+the\s+(?!(?:meaning|definition)\b))/;

export const classifyQuestion = (question: string): QuestionRoute => {
  const text = question.trim().toLowerCase();
  const term = REPORT_LOOKUP.test(text) ? null : extractDefinitionTerm(text);
  if (term && !PATIENT_SPECIFIC.test(term) && !CLINICAL_INTENT.test(term)) return { kind: 'terminology', term };
  if (CLINICAL_INTENT.test(text)) return { kind: 'clinical' };
  return { kind: 'report' };
};
