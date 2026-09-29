import { escapeForPrompt } from '../prompts/common';

/**
 * Evidence = one retrieved chunk handed to the final LLM. Each knowledge domain keeps its own
 * id space so an answer always preserves which domain — and which source — a claim came from:
 *   F#  structured patient findings (assessment input, not retrieved evidence)
 *   P#  patient document evidence (the user's own uploads)
 *   R#  verified medical reference evidence (authorised guideline/reference sources)
 *   T#  medical terminology evidence (NLM MedlinePlus / MeSH definitions — the meaning of a term, NOT guidance)
 * Citations returned by the model are validated against these ids before use.
 */

export type EvidenceDomain = 'patient' | 'reference' | 'terminology';

export const EVIDENCE_PREFIX: Record<EvidenceDomain, string> = { patient: 'P', reference: 'R', terminology: 'T' };

export type TerminologyMatchType = 'exact_term' | 'exact_alias' | 'semantic';

export interface Evidence {
  evidenceId: string;
  domain: EvidenceDomain;
  chunkId: string;
  /** Vector similarity (cosine). A retrieval signal only — never clinical certainty. */
  similarity: number;
  content: string;
  // Patient provenance
  documentId?: string;
  filename?: string;
  page?: number;
  contentType?: string;
  tableId?: string;
  // Reference provenance (only fields present in the source's metadata)
  sourceId?: string;
  title?: string;
  organization?: string;
  publicationDate?: string;
  version?: string;
  url?: string;
  section?: string;
  medicalTopic?: string;
  sourceType?: string;
  // Terminology provenance (dataset-derived only)
  dataset?: string;
  datasetName?: string;
  matchType?: TerminologyMatchType;
  aliases?: string[];
  meshId?: string;
}

export type EvidenceInput = Omit<Evidence, 'evidenceId' | 'domain'>;

export interface EvidenceBudget {
  maxPatientItems: number;
  maxReferenceItems: number;
  maxPatientChars: number;
  maxReferenceChars: number;
  maxTerminologyItems: number;
  maxTerminologyChars: number;
}

export const DEFAULT_EVIDENCE_BUDGET: EvidenceBudget = {
  maxPatientItems: 6,
  maxReferenceItems: 8,
  maxPatientChars: 9000,
  maxReferenceChars: 7000,
  maxTerminologyItems: 4,
  maxTerminologyChars: 7000,
};

const takeWithinBudget = (items: EvidenceInput[], maxItems: number, maxChars: number): EvidenceInput[] => {
  const seen = new Set<string>();
  const out: EvidenceInput[] = [];
  let used = 0;
  for (const item of [...items].sort((a, b) => b.similarity - a.similarity || a.chunkId.localeCompare(b.chunkId))) {
    if (seen.has(item.chunkId) || out.length >= maxItems) continue;
    if (used + item.content.length > maxChars && out.length > 0) continue;
    seen.add(item.chunkId);
    out.push(item);
    used += item.content.length;
  }
  return out;
};

/** Like takeWithinBudget but keeps the caller's order (terminology: exact matches are listed first). */
const takeInOrder = (items: EvidenceInput[], maxItems: number, maxChars: number): EvidenceInput[] => {
  const seen = new Set<string>();
  const out: EvidenceInput[] = [];
  let used = 0;
  for (const item of items) {
    if (seen.has(item.chunkId) || out.length >= maxItems) continue;
    if (used + item.content.length > maxChars && out.length > 0) continue;
    seen.add(item.chunkId);
    out.push(item);
    used += item.content.length;
  }
  return out;
};

/** Deduplicates, ranks, applies the context budget and assigns P#/R#/T# ids. */
export const assembleEvidence = (
  patient: EvidenceInput[],
  reference: EvidenceInput[],
  budget: EvidenceBudget = DEFAULT_EVIDENCE_BUDGET,
  terminology: EvidenceInput[] = []
): Evidence[] => [
  ...takeWithinBudget(patient, budget.maxPatientItems, budget.maxPatientChars).map((e, i) => ({ ...e, evidenceId: `P${i + 1}`, domain: 'patient' as const })),
  ...takeWithinBudget(reference, budget.maxReferenceItems, budget.maxReferenceChars).map((e, i) => ({ ...e, evidenceId: `R${i + 1}`, domain: 'reference' as const })),
  ...takeInOrder(terminology, budget.maxTerminologyItems, budget.maxTerminologyChars).map((e, i) => ({ ...e, evidenceId: `T${i + 1}`, domain: 'terminology' as const })),
];

const attr = (name: string, value: string | number | undefined): string =>
  value === undefined || value === '' ? '' : ` ${name}="${escapeForPrompt(String(value))}"`;

export const formatPatientEvidence = (evidence: Evidence[]): string => {
  const items = evidence.filter((e) => e.domain === 'patient');
  if (items.length === 0) return '(no patient document evidence retrieved)';
  return items
    .map((e) => `<evidence id="${e.evidenceId}"${attr('document', e.filename)}${attr('page', e.page)}${attr('type', e.contentType)}>\n${escapeForPrompt(e.content)}\n</evidence>`)
    .join('\n');
};

export const formatReferenceEvidence = (evidence: Evidence[]): string => {
  const items = evidence.filter((e) => e.domain === 'reference');
  if (items.length === 0) return '(no verified medical reference evidence retrieved)';
  return items
    .map(
      (e) =>
        `<evidence id="${e.evidenceId}"${attr('title', e.title)}${attr('organization', e.organization)}${attr('section', e.section)}${attr('page', e.page)}>\n${escapeForPrompt(e.content)}\n</evidence>`
    )
    .join('\n');
};

export const formatTerminologyEvidence = (evidence: Evidence[]): string => {
  const items = evidence.filter((e) => e.domain === 'terminology');
  if (items.length === 0) return '(no medical terminology evidence retrieved)';
  return items
    .map(
      (e) =>
        `<evidence id="${e.evidenceId}"${attr('term', e.title)}${attr('source', e.datasetName)}${attr('also_called', e.aliases?.slice(0, 8).join('; '))}>\n${escapeForPrompt(e.content)}\n</evidence>`
    )
    .join('\n');
};

/** Splits model-supplied ids into those that exist in the supplied evidence and those that do not. */
export const validateCitations = (
  ids: string[],
  evidence: Evidence[],
  domain?: EvidenceDomain
): { valid: Evidence[]; invalid: string[] } => {
  const byId = new Map(evidence.filter((e) => !domain || e.domain === domain).map((e) => [e.evidenceId, e]));
  const valid: Evidence[] = [];
  const invalid: string[] = [];
  for (const raw of new Set(ids.map((id) => id.trim().toUpperCase()))) {
    const found = byId.get(raw);
    if (found) valid.push(found);
    else invalid.push(raw);
  }
  return { valid, invalid };
};

export const MAX_EXCERPT_CHARS = 600;

/** Client-facing citation built only from retrieved metadata (never from model output). */
export const toCitation = (e: Evidence) => ({
  evidenceId: e.evidenceId,
  domain: e.domain,
  chunkId: e.chunkId,
  similarity: Number(e.similarity.toFixed(4)),
  excerpt: e.content.length > MAX_EXCERPT_CHARS ? `${e.content.slice(0, MAX_EXCERPT_CHARS)}…` : e.content,
  ...(e.documentId ? { documentId: e.documentId } : {}),
  ...(e.filename ? { filename: e.filename } : {}),
  ...(e.page !== undefined ? { page: e.page } : {}),
  ...(e.contentType ? { contentType: e.contentType } : {}),
  ...(e.tableId ? { tableId: e.tableId } : {}),
  ...(e.sourceId ? { sourceId: e.sourceId } : {}),
  ...(e.title ? { title: e.title } : {}),
  ...(e.organization ? { organization: e.organization } : {}),
  ...(e.publicationDate ? { publicationDate: e.publicationDate } : {}),
  ...(e.version ? { version: e.version } : {}),
  ...(e.url ? { url: e.url } : {}),
  ...(e.section ? { section: e.section } : {}),
  ...(e.medicalTopic ? { medicalTopic: e.medicalTopic } : {}),
  ...(e.sourceType ? { sourceType: e.sourceType } : {}),
  ...(e.dataset ? { dataset: e.dataset } : {}),
  ...(e.datasetName ? { datasetName: e.datasetName } : {}),
  ...(e.matchType ? { matchType: e.matchType } : {}),
  ...(e.aliases?.length ? { aliases: e.aliases.slice(0, 12) } : {}),
  ...(e.meshId ? { meshId: e.meshId } : {}),
});

export type Citation = ReturnType<typeof toCitation>;

/* ───────────── Inline citation tokens ("[R1]", "[P2, R3]") written inside model prose ───────────── */

const INLINE_GROUP = /\[\s*([FPRT]\d{1,3}(?:\s*[,;]\s*[FPRT]\d{1,3})*)\s*\]/gi;

/** Ids the model wrote inline in prose, upper-cased and de-duplicated. Deterministic (no model call). */
export const inlineCitationIds = (text: string): string[] => {
  const ids = new Set<string>();
  for (const m of text.matchAll(INLINE_GROUP)) for (const id of m[1].split(/[,;]/)) ids.add(id.trim().toUpperCase());
  return [...ids];
};

/** Removes inline ids that are not in `allowed` (fabricated or wrong-domain) so prose never cites what it may not. */
export const stripInvalidInlineCitations = (text: string, allowed: Set<string>): string =>
  text
    .replace(INLINE_GROUP, (_whole, group: string) => {
      const kept = group
        .split(/[,;]/)
        .map((id) => id.trim().toUpperCase())
        .filter((id) => allowed.has(id));
      return kept.length ? `[${kept.join(', ')}]` : '';
    })
    .replace(/\]\s*\[(?=[FPRT]\d)/gi, ', ') // adjacent groups "[F8][F13]" → "[F8, F13]"
    .replace(/[ \t]{2,}/g, ' ')
    .replace(/\s+([.,;:])/g, '$1');
