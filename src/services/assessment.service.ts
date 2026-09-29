import { z } from 'zod';
import { generateStructured, DISCLAIMER, OutputValidationError, type ModelCandidate, type ProviderName, type StructuredReport } from './ai.service';
import { getAllowedDocuments, retrievePatientEvidence } from './document.service';
import { retrieveReferenceEvidence } from './reference.service';
import {
  assembleEvidence,
  formatPatientEvidence,
  formatReferenceEvidence,
  inlineCitationIds,
  stripInvalidInlineCitations,
  toCitation,
  validateCitations,
  type Citation,
  type Evidence,
  type EvidenceInput,
} from '../rag/evidence';
import { CLINICAL_ASSESSMENT_PROMPT, ASSESSMENT_TASK, DATA_HANDLING_RULES, escapeForPrompt, renderPrompt } from '../prompts';
import type { LabValue, PatientDetails } from '../types';
import { foldText } from '../documents/grounding';
import { recordFailure } from '../utils/failures';
import { logger, errorMeta } from '../utils/logger';
import { StageTimer } from '../utils/timing';

/**
 * JOB B — evidence-grounded risk assessment (separate from Job A extraction).
 *
 *   structured findings → deterministic retrieval queries → local HF embeddings →
 *   patient evidence (user-isolated) + verified reference evidence (shared, multi-source) →
 *   evidence budget → ONE LangChain-prompted structured model call → citation validation →
 *   deterministic consistency rules → result (or explicit insufficient_evidence).
 *
 * This is an AI triage aid for clinician review — not a diagnosis and not a validated
 * clinical scoring system.
 */

export const RISK_LEVELS = ['low', 'medium', 'high', 'insufficient_evidence'] as const;
export type RiskLevel = (typeof RISK_LEVELS)[number];

/** Why a proposed risk level was replaced by insufficient_evidence (deterministic, server-side). */
export type DowngradeReason = 'no_valid_citation' | 'no_reference_evidence' | 'reference_not_cited';

/**
 * A risk level (anything but insufficient_evidence) is a clinical interpretation. It must rest on at least
 * one VERIFIED medical reference (R#) that the assessment actually cites. Patient findings alone can only
 * be restated, not interpreted. Set RAG_REQUIRE_REFERENCE_FOR_ASSESSMENT=false to disable (not recommended).
 */
export const referenceRequiredForAssessment = (): boolean => process.env.RAG_REQUIRE_REFERENCE_FOR_ASSESSMENT !== 'false';

/** Engineering consistency bands for the model-assigned score — NOT a validated clinical scale. */
export const RISK_BANDS: Record<Exclude<RiskLevel, 'insufficient_evidence'>, { min: number; max: number }> = {
  low: { min: 0, max: 39 },
  medium: { min: 40, max: 69 },
  high: { min: 70, max: 100 },
};

export const AssessmentSchema = z.object({
  riskLevel: z.enum(RISK_LEVELS),
  riskScore: z.number().min(0).max(100).nullable(),
  summary: z.string().max(4000),
  keyFindings: z
    .array(z.object({ finding: z.string().max(500), significance: z.string().max(1000), evidenceIds: z.array(z.string().max(10)).max(10) }))
    .max(20),
  supportingEvidence: z.array(z.string().max(10)).max(30),
  patientEvidence: z.array(z.string().max(10)).max(30),
  uncertainty: z.string().max(2000),
  insufficientEvidence: z.boolean(),
});

export type RawAssessment = z.infer<typeof AssessmentSchema>;

export interface AssessmentFindings {
  patientDetails?: PatientDetails;
  symptoms: string[];
  medicines: string[];
  labValues: LabValue[];
  rawText?: string;
  voiceTranscript?: string;
  report?: Partial<StructuredReport>;
}

export interface KeyFinding {
  finding: string;
  significance: string;
  evidenceIds: string[];
}

export interface ClinicalAssessment {
  riskLevel: RiskLevel;
  riskScore?: number;
  summary: string;
  keyFindings: KeyFinding[];
  uncertainty: string;
  insufficientEvidence: boolean;
  /** Findings (F#) the model cited, as supplied to it. */
  findingsCited: { id: string; text: string }[];
  /** Patient evidence actually cited (validated). */
  patientEvidence: Citation[];
  /** Verified medical references actually cited (validated) — the "Medical References Used". */
  medicalReferencesUsed: Citation[];
  /**
   * What retrieval FOUND. Retrieved is not used: only citations that survived validation appear in
   * patientEvidence / medicalReferencesUsed, which is what the UI shows as "used".
   */
  retrieval: {
    queries: number;
    patientStatus: string;
    referenceStatus: string;
    patientRetrieved: number;
    referencesRetrieved: number;
    referencesCited: number;
    topReferenceScore?: number;
  };
  validation: { droppedCitations: number; riskScoreAdjusted: boolean; downgradedToInsufficient: boolean; downgradeReason?: DowngradeReason };
  provider: ProviderName;
  model: string;
  fallbackUsed: boolean;
}

/* ───────────────────────── Findings ───────────────────────── */

export interface NumberedFinding {
  id: string;
  text: string;
}

const labLine = (lab: LabValue): string => {
  const parts = [`${lab.name}: ${lab.value}${lab.unit ? ` ${lab.unit}` : ''}`];
  if (lab.normalRange) parts.push(`reference range printed in the report: ${lab.normalRange}`);
  if (lab.flag) {
    parts.push(
      `${lab.flag}${lab.flagSource === 'printed_range' ? ' (computed by comparing the value with the printed range)' : lab.flagSource === 'table' ? ' (as printed in the report table)' : ''}`
    );
  } else if (lab.isAbnormal) parts.push('marked abnormal (unverified)');
  if (lab.date) parts.push(`date ${lab.date}`);
  if (lab.source?.filename) parts.push(`source ${lab.source.filename}${lab.source.page !== undefined ? ` p.${lab.source.page}` : ''}`);
  return parts.join('; ');
};

const isOutOfRange = (lab: LabValue): boolean => (lab.flag ? lab.flag !== 'Normal' : lab.isAbnormal);

/** Removes the patient's name from free text before it can reach the assessment model (data minimisation). */
export const scrubName = (text: string, name?: string): string => {
  let out = text;
  for (const part of (name ?? '').split(/\s+/).filter((p) => p.length >= 3)) {
    const escaped = part.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    out = out.replace(new RegExp(`\\b${escaped}\\b`, 'gi'), '[patient]');
  }
  return out;
};

/**
 * Numbered, structured findings (F1..Fn). The patient's name is deliberately omitted
 * (data minimisation). Out-of-range labs come first, then the report's own written impressions
 * and printed risk scores. The raw document text is NOT included when structured data exists.
 */
export const buildFindings = (f: AssessmentFindings): NumberedFinding[] => {
  const items: string[] = [];
  const demo = [f.patientDetails?.age ? `age ${f.patientDetails.age}` : '', f.patientDetails?.gender ? `gender ${f.patientDetails.gender}` : '']
    .filter(Boolean)
    .join(', ');
  if (demo) items.push(`Demographics: ${demo}`);

  const abnormal = f.labValues.filter(isOutOfRange);
  const inRange = f.labValues.filter((l) => !isOutOfRange(l));
  if (f.labValues.length > 0) {
    items.push(`Lab overview: ${f.labValues.length} values extracted; ${abnormal.length} outside the range printed in the report`);
  }
  for (const lab of abnormal) items.push(`Lab OUT OF RANGE — ${labLine(lab)}`);
  for (const d of f.report?.diagnosesMentioned ?? []) items.push(`Impression/diagnosis written in the report (by the issuing clinic) — ${d}`);
  for (const r of f.report?.reportRiskScores ?? []) items.push(`Risk score printed in the report (by the issuing clinic) — ${r.name}: ${r.result}`);
  for (const s of f.symptoms) items.push(`Symptom — ${s}`);
  for (const m of f.medicines) items.push(`Medication — ${m}`);
  for (const m of f.report?.measurements ?? []) items.push(`Measurement — ${m.name}: ${m.value}${m.unit ? ` ${m.unit}` : ''}`);
  for (const lab of inRange) items.push(`Lab within range — ${labLine(lab)}`);
  for (const i of f.report?.imagingFindings ?? []) items.push(`Image-derived finding (AI-read, unverified) — ${i}`);

  // Free text is only a fallback for inputs that produced almost no structured findings (e.g. a short typed note).
  if (items.length < 3) {
    const notes = [f.report?.doctorNotes, f.rawText, f.voiceTranscript].filter((t): t is string => Boolean(t?.trim()));
    if (notes.length) items.push(`Clinical notes — ${scrubName(notes.join(' / ').slice(0, 1500), f.patientDetails?.name)}`);
  } else if (f.report?.doctorNotes?.trim()) {
    items.push(`Doctor notes — ${scrubName(f.report.doctorNotes.slice(0, 1000), f.patientDetails?.name)}`);
  }
  return items.slice(0, 90).map((text, i) => ({ id: `F${i + 1}`, text }));
};

/** Deterministic retrieval queries: abnormal labs first, then other labs, symptoms, written diagnoses. */
export const buildFindingQueries = (f: AssessmentFindings, max = 6): string[] => {
  const abnormal = f.labValues.filter((l) => l.isAbnormal || (l.flag && l.flag !== 'Normal'));
  const normal = f.labValues.filter((l) => !abnormal.includes(l));
  const queries = [
    ...abnormal.map((l) => `${l.name} ${l.flag ?? 'abnormal'} ${l.value}${l.unit ? ` ${l.unit}` : ''} clinical significance`),
    ...(f.report?.diagnosesMentioned ?? []).map((d) => d),
    ...f.symptoms.map((s) => `${s} evaluation`),
    ...normal.map((l) => `${l.name} reference range interpretation`),
  ];
  return [...new Set(queries.map((q) => q.trim()).filter(Boolean))].slice(0, max);
};

/* ───────────────────────── Validation ───────────────────────── */

const numericTokens = (tokens: string[]): string[] => tokens.filter((t) => /\d/.test(t));

/**
 * Finds the numbered finding a key-finding restates: ≥60% of its significant words must appear in the
 * finding, and every number it mentions must too. Returns [] when nothing matches — never guesses.
 */
export const traceToFinding = (text: string, findings: NumberedFinding[]): string[] => {
  const words = foldText(text).split(' ').filter((t) => t.length >= 2);
  if (words.length === 0) return [];
  const numbers = numericTokens(words);
  let best: { id: string; score: number } | undefined;
  for (const f of findings) {
    const folded = ` ${foldText(f.text)} `;
    if (!numbers.every((n) => folded.includes(` ${n} `))) continue;
    const score = words.filter((w) => folded.includes(` ${w} `)).length / words.length;
    if (score >= 0.6 && (!best || score > best.score)) best = { id: f.id, score };
  }
  return best ? [best.id] : [];
};

/**
 * Deterministic post-validation:
 *  - citations are checked against the supplied ids; invented ids are dropped
 *  - insufficientEvidence and riskLevel are made consistent (either ⇒ insufficient)
 *  - a low/medium/high result that cites no valid finding/evidence is downgraded to
 *    insufficient_evidence (never to "low")
 *  - the score is clamped into its category band (category authoritative); null when insufficient
 */
const withDisclaimer = (s: string): string => (s.includes(DISCLAIMER) ? s : `${s}\n\n${DISCLAIMER}`);

export const validateAssessment = (
  raw: RawAssessment,
  findings: NumberedFinding[],
  evidence: Evidence[],
  options: { requireReference?: boolean } = {}
) => {
  const requireReference = options.requireReference ?? referenceRequiredForAssessment();
  const summary = raw.summary.trim();
  if (summary.replace(DISCLAIMER, '').replace(/[\s.]/g, '').length < 10) throw new OutputValidationError('Assessment summary is empty or too short');

  const findingIds = new Set(findings.map((f) => f.id));
  let dropped = 0;
  const checkIds = (ids: string[]) => {
    const upper = [...new Set(ids.map((i) => i.trim().toUpperCase()))];
    const ok = upper.filter((id) => findingIds.has(id) || evidence.some((e) => e.evidenceId === id));
    dropped += upper.length - ok.length;
    return ok;
  };

  const patient = validateCitations(raw.patientEvidence, evidence, 'patient');
  const reference = validateCitations(raw.supportingEvidence, evidence, 'reference');
  dropped += patient.invalid.length + reference.invalid.length;
  const keyFindings: KeyFinding[] = raw.keyFindings
    .map((k) => ({ finding: k.finding.trim(), significance: k.significance.trim(), evidenceIds: checkIds(k.evidenceIds) }))
    .filter((k) => k.finding)
    // Models often leave evidence ids empty. Trace such findings to the numbered finding they restate
    // (deterministic text match) instead of discarding a perfectly grounded conclusion.
    .map((k) => (k.evidenceIds.length > 0 ? k : { ...k, evidenceIds: traceToFinding(k.finding, findings) }));

  // Evidence cited inside key findings counts as used as well.
  const citedEvidenceIds = new Set([...patient.valid, ...reference.valid].map((e) => e.evidenceId));
  for (const k of keyFindings) for (const id of k.evidenceIds) if (!findingIds.has(id)) citedEvidenceIds.add(id);
  // ...and so do ids written inline in the prose ("[R1]"): the text and the citation list can never disagree.
  const proseIds = inlineCitationIds([raw.summary, raw.uncertainty, ...raw.keyFindings.flatMap((k) => [k.finding, k.significance])].join('\n'));
  const knownIds = new Set([...findingIds, ...evidence.map((e) => e.evidenceId)]);
  for (const id of proseIds) {
    if (!knownIds.has(id)) dropped++;
    else if (!findingIds.has(id)) citedEvidenceIds.add(id);
  }
  const proseFindingIds = proseIds.filter((id) => findingIds.has(id));
  const usedEvidence = evidence.filter((e) => citedEvidenceIds.has(e.evidenceId));
  const usedFindings = new Set([...keyFindings.flatMap((k) => k.evidenceIds.filter((id) => findingIds.has(id))), ...proseFindingIds]);
  const usedReferences = usedEvidence.filter((e) => e.domain === 'reference');
  const referencesRetrieved = evidence.filter((e) => e.domain === 'reference').length;

  let riskLevel: RiskLevel = raw.insufficientEvidence ? 'insufficient_evidence' : raw.riskLevel;
  let downgraded = false;
  let downgradeReason: DowngradeReason | undefined;
  if (riskLevel !== 'insufficient_evidence' && usedEvidence.length === 0 && usedFindings.size === 0) {
    riskLevel = 'insufficient_evidence';
    downgraded = true;
    downgradeReason = 'no_valid_citation';
  } else if (riskLevel !== 'insufficient_evidence' && requireReference && usedReferences.length === 0) {
    // Retrieved-but-uncited references never count as used; no cited verified reference ⇒ no risk level.
    riskLevel = 'insufficient_evidence';
    downgraded = true;
    downgradeReason = referencesRetrieved === 0 ? 'no_reference_evidence' : 'reference_not_cited';
  }

  let riskScore: number | undefined;
  let adjusted = false;
  if (riskLevel !== 'insufficient_evidence') {
    const band = RISK_BANDS[riskLevel];
    if (raw.riskScore !== null && Number.isFinite(raw.riskScore)) {
      const rounded = Math.round(raw.riskScore);
      riskScore = Math.min(band.max, Math.max(band.min, rounded));
      adjusted = riskScore !== rounded;
    }
  }

  // Prose may only cite ids that are real and of an allowed kind; anything else is removed from the text.
  const allowedProse = new Set([...findingIds, ...usedEvidence.map((e) => e.evidenceId)]);
  const clean = (text: string): string => stripInvalidInlineCitations(text, allowedProse);
  const downgradeNote: Record<DowngradeReason, string> = {
    no_valid_citation: "No risk level is reported: the AI's proposed assessment could not be tied to the supplied evidence. Its text follows for reference only.",
    no_reference_evidence: 'No risk level is reported: no verified medical reference evidence was available for these findings, so they cannot be clinically interpreted. The AI text follows for reference only.',
    reference_not_cited: "No risk level is reported: verified references were retrieved but the AI's assessment did not rely on any of them. Its text follows for reference only.",
  };

  return {
    riskLevel,
    ...(riskScore !== undefined ? { riskScore } : {}),
    summary: withDisclaimer(downgradeReason ? `${downgradeNote[downgradeReason]} ${clean(summary)}` : clean(summary)),
    keyFindings: keyFindings.map((k) => ({ ...k, finding: clean(k.finding), significance: clean(k.significance) })),
    uncertainty: clean(raw.uncertainty.trim()) || (riskLevel === 'insufficient_evidence' ? 'The supplied findings and evidence were not sufficient for an assessment.' : ''),
    insufficientEvidence: riskLevel === 'insufficient_evidence',
    findingsCited: findings.filter((f) => usedFindings.has(f.id)),
    patientEvidence: usedEvidence.filter((e) => e.domain === 'patient').map(toCitation),
    medicalReferencesUsed: usedEvidence.filter((e) => e.domain === 'reference').map(toCitation),
    validation: { droppedCitations: dropped, riskScoreAdjusted: adjusted, downgradedToInsufficient: downgraded, ...(downgradeReason ? { downgradeReason } : {}) },
  };
};

/* ───────────────────────── Assessment ───────────────────────── */

export interface AssessmentContext {
  userId: string;
  documentIds?: string[];
  consultationId?: string;
  candidates?: ModelCandidate[];
}

export const buildAssessmentPrompt = (findings: NumberedFinding[], evidence: Evidence[], task = ASSESSMENT_TASK) =>
  renderPrompt(CLINICAL_ASSESSMENT_PROMPT, {
    data_rules: DATA_HANDLING_RULES,
    patient_findings: findings.length ? findings.map((f) => `[${f.id}] ${escapeForPrompt(f.text)}`).join('\n') : '(no structured findings)',
    patient_evidence: formatPatientEvidence(evidence),
    medical_evidence: formatReferenceEvidence(evidence),
    user_query: escapeForPrompt(task),
  });

export const assessRisk = async (input: AssessmentFindings, ctx: AssessmentContext, timer = new StageTimer()): Promise<ClinicalAssessment> => {
  const findings = buildFindings(input);
  const queries = buildFindingQueries(input);

  // Evidence retrieval (local embeddings + vector search). Retrieval outages degrade to
  // "no evidence retrieved" — stated in the result — never to invented evidence.
  let patientEvidence: EvidenceInput[] = [];
  let patientStatus = 'no_documents';
  if (ctx.documentIds?.length || ctx.consultationId) {
    try {
      const allowed = await getAllowedDocuments(ctx.userId, { documentIds: ctx.documentIds, consultationId: ctx.consultationId });
      const patient = await retrievePatientEvidence(ctx.userId, queries, allowed, {}, timer);
      patientEvidence = patient.evidence;
      patientStatus = patient.status;
    } catch (error) {
      patientStatus = 'unavailable';
      logger.warn('assessment.patient_retrieval_failed', errorMeta(error));
    }
  }

  let referenceEvidence: EvidenceInput[] = [];
  let referenceStatus = 'no_sources';
  let topReferenceScore: number | undefined;
  try {
    const reference = await retrieveReferenceEvidence(queries, {}, timer);
    referenceEvidence = reference.evidence;
    referenceStatus = reference.status;
    topReferenceScore = reference.topScores.length ? Math.max(...reference.topScores) : undefined;
  } catch (error) {
    referenceStatus = 'unavailable';
    logger.warn('assessment.reference_retrieval_failed', errorMeta(error));
  }

  const evidence = assembleEvidence(patientEvidence, referenceEvidence);
  if (findings.length === 0 && evidence.length === 0) recordFailure('insufficient_context', { operation: 'assess_risk', reason: 'no_findings' });

  const { system, prompt } = await buildAssessmentPrompt(findings, evidence);
  const result = await timer.time('assessment', () =>
    generateStructured({
      operation: 'assess_clinical_risk',
      schema: AssessmentSchema,
      system,
      prompt,
      temperature: 0.1,
      postValidate: (output) => validateAssessment(output, findings, evidence),
      candidates: ctx.candidates,
    })
  );

  const v = result.value;
  if (v.validation.droppedCitations > 0) recordFailure('citation_failure', { operation: 'assess_risk', dropped: v.validation.droppedCitations });
  if (v.riskLevel === 'insufficient_evidence') recordFailure('insufficient_context', { operation: 'assess_risk', reason: v.validation.downgradeReason ?? 'model_reported' });

  return {
    ...v,
    retrieval: {
      queries: queries.length,
      patientStatus,
      referenceStatus,
      patientRetrieved: evidence.filter((e) => e.domain === 'patient').length,
      referencesRetrieved: evidence.filter((e) => e.domain === 'reference').length,
      referencesCited: v.medicalReferencesUsed.length,
      ...(topReferenceScore !== undefined ? { topReferenceScore: Number(topReferenceScore.toFixed(4)) } : {}),
    },
    provider: result.provider,
    model: result.model,
    fallbackUsed: result.fallbackUsed,
  };
};
