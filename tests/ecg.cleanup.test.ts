/**
 * Cleanup after manual ECG test: biomarker names carry no provenance, citations have separators, and
 * unrelated retrieved references cannot make an ECG assessment look grounded. Synthetic data only.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';

const generateText = vi.fn();
vi.mock('ai', async (importOriginal) => ({
  ...(await importOriginal<typeof import('ai')>()),
  generateText: (...args: unknown[]) => generateText(...args),
}));

import { cleanBiomarkerName } from '../src/documents/labName';
import { stripInvalidInlineCitations } from '../src/rag/evidence';
import { ingestReferenceSource, invalidateReferenceRegistry } from '../src/services/reference.service';
import { assessRisk } from '../src/services/assessment.service';
import { assessmentOutput } from './helpers/aiOutputs';
import { resetDb } from './helpers/fakeModels';

const LIPID_REF = '# Lipid management\nSynthetic test passage about statins, LDL cholesterol targets and triglyceride management in adults. High values have clinical significance for heart risk and rate of events.';
const ecg = {
  symptoms: ['palpitations'],
  medicines: [],
  labValues: [
    { name: 'Heart Rate', value: '110', unit: 'bpm', normalRange: '60-100', isAbnormal: true, flag: 'High' },
    { name: 'QTc Interval', value: '480', unit: 'ms', normalRange: '350-450', isAbnormal: true, flag: 'High' },
  ],
};

beforeEach(async () => {
  generateText.mockReset();
  resetDb();
  invalidateReferenceRegistry();
  process.env.GEMINI_API_KEY = 'test-gemini';
  process.env.RAG_REF_MIN_SCORE = '0.01'; // noisy retrieval on purpose: unrelated lipid chunks ARE retrieved
  const ingested = await ingestReferenceSource(
    { sourceId: 'ref-lipid', title: 'Synthetic Lipid Reference', organization: 'AyuNidan Synthetic Test Corpus (fictional)', sourceType: 'synthetic_test', authorization: 'Synthetic test text; not medical guidance.' },
    Buffer.from(LIPID_REF),
    'ref-lipid.md'
  );
  expect(ingested.status).toBe('indexed');
});
afterEach(() => {
  delete process.env.GEMINI_API_KEY;
  delete process.env.RAG_REF_MIN_SCORE;
});

describe('display cleanup', () => {
  it('A. biomarker names contain no file name, date or page metadata', () => {
    expect(cleanBiomarkerName('Heart Rate2/17/2011ECG-Sample-Report.pdf · pp.1, 2, 3')).toBe('Heart Rate');
    for (const [raw, clean] of [['PR Interval ECG-Sample-Report.pdf', 'PR Interval'], ['QRS Duration 2/17/2011', 'QRS Duration'], ['QTc Interval · p.2', 'QTc Interval'], ['Vitamin B12', 'Vitamin B12']]) {
      expect(cleanBiomarkerName(raw)).toBe(clean);
    }
  });

  it('B. adjacent citation groups render with separators', () => {
    expect(stripInvalidInlineCitations('Elevated [F8][F13] and [F2] [F3].', new Set(['F8', 'F13', 'F2', 'F3']))).toBe('Elevated [F8, F13] and [F2, F3].');
  });
});

describe('ECG assessment with only unrelated verified references available', () => {
  it('C. retrieved unrelated lipid chunks cannot be shown as used unless validly cited; fabricated ids are rejected', async () => {
    generateText.mockResolvedValue({ output: assessmentOutput({ riskLevel: 'high', riskScore: 85, summary: 'Prolonged QTc [R9].', keyFindings: [{ finding: 'QTc high', significance: 'x', evidenceIds: ['F4', 'R9'] }], supportingEvidence: ['R9'] }) });
    const result = await assessRisk(ecg, { userId: 'u' });
    expect(result.retrieval.referencesRetrieved).toBeGreaterThan(0); // noise was retrieved…
    expect(result.medicalReferencesUsed).toEqual([]); // …but nothing unvalidated is shown as used
    expect(result.summary).not.toContain('R9');
    expect(result.validation.droppedCitations).toBeGreaterThan(0);
  });

  it('D. stays insufficient_evidence: no risk level or score when the model does not rely on any verified reference', async () => {
    generateText.mockResolvedValue({ output: assessmentOutput({ riskLevel: 'medium', riskScore: 55, keyFindings: [{ finding: 'Heart Rate high', significance: 'x', evidenceIds: ['F3'] }], supportingEvidence: [] }) });
    const result = await assessRisk(ecg, { userId: 'u' });
    expect(result).toMatchObject({ riskLevel: 'insufficient_evidence', insufficientEvidence: true, medicalReferencesUsed: [] });
    expect(result.riskScore).toBeUndefined();
    expect(result.validation.downgradeReason).toBe('reference_not_cited');
  });
});
