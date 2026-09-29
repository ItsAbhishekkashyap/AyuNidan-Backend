/**
 * The document's own words are used verbatim; AI output is verified against them.
 * Regression tests for: AI-rewritten text replacing the document, invented items, model-decided
 * abnormal flags, ambiguous names, and broad questions failing on a similarity threshold.
 * All data is fictional (structure modelled on a lab-report layout).
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import request from 'supertest';

const generateText = vi.fn();
vi.mock('ai', async (importOriginal) => ({
  ...(await importOriginal<typeof import('ai')>()),
  generateText: (...args: unknown[]) => generateText(...args),
}));

import { createApp } from '../src/app';
import { cache } from '../src/middleware/cache';
import { resetRateLimits } from '../src/middleware/rateLimiter';
import { collectSources, normalizeExtraction } from '../src/services/ai.service';
import { buildStructuredEvidence, userNamespace } from '../src/services/document.service';
import { traceToFinding, validateAssessment, buildFindings, scrubName } from '../src/services/assessment.service';
import { flagFromRange, parseRange, parseResultNumber } from '../src/documents/ranges';
import { extractIdentity } from '../src/documents/identity';
import { findLabPages, findTextPages } from '../src/documents/grounding';
import { getVectorStore } from '../src/rag/vectorStore';
import { InMemoryVectorStore } from '../src/rag/inMemoryVectorStore';
import { buildPdf } from '../src/eval/pdfBuilder';
import { FakeConsultation, resetDb } from './helpers/fakeModels';
import { createTestUser } from './helpers/auth';
import { assessmentOutput, extractionOutput, isQACall, labOut } from './helpers/aiOutputs';

const app = createApp();
const store = () => getVectorStore() as InMemoryVectorStore;
const pdfFile = (buffer: Buffer, originalname = 'lab.pdf') => ({ mimetype: 'application/pdf', buffer, size: buffer.length, originalname }) as Express.Multer.File;

const REPORT_LINES = [
  'Health Report',
  'Patient Name (Your name) :  ZORBLAX QUENDI',
  'Age/Gender (Your age/gender) : 58Y/Female',
  'Dear  MIRELA VANTHO (Your name)',
  'TRIGLYCERIDES  199 mg/dL  0 - 149.99 mg/dL',
  'HDL CHOLESTEROL  43 mg/dL  40 - 60 mg/dL',
  'Impressions: PRE DIABETIC STAGE',
];

beforeEach(() => {
  resetDb();
  cache.clear();
  resetRateLimits();
  generateText.mockReset();
  process.env.GEMINI_API_KEY = 'test-gemini';
  process.env.RAG_DOC_MIN_SCORE = '0.05';
});
afterEach(() => {
  delete process.env.GEMINI_API_KEY;
  delete process.env.RAG_DOC_MIN_SCORE;
});

/* ───────────── deterministic helpers ───────────── */

describe('printed-range flags (computed by code, not by the model)', () => {
  it('compares values with the range printed in the report', () => {
    expect(flagFromRange('35.90 %', '36  -  46 %')).toBe('Low');
    expect(flagFromRange('199', '0  -  149.99 mg/dL')).toBe('High');
    expect(flagFromRange('43', '40 - 60 mg/dL')).toBe('Normal');
    expect(flagFromRange('0.83 ng/mL', '0.7 – 2.04 ng/mL')).toBe('Normal');
    expect(flagFromRange('12', '< 10')).toBe('High');
    expect(flagFromRange('5', '≤ 10')).toBe('Normal');
    expect(flagFromRange('20', '> 40')).toBe('Low');
  });

  it('refuses to guess when the value or range is unusable', () => {
    expect(flagFromRange('6.2', '0 - 0 %')).toBeUndefined(); // degenerate range printed by the lab
    expect(flagFromRange('NEGATIVE', '0 - 5')).toBeUndefined();
    expect(flagFromRange('5', undefined)).toBeUndefined();
    expect(parseResultNumber('<5')).toBeUndefined();
    expect(parseRange('no numbers')).toBeUndefined();
  });
});

describe('identity from labelled fields', () => {
  it('reads name, age and gender and reports a different salutation name', () => {
    const id = extractIdentity([{ page: 1, text: 'Patient Name (Your name) :  NI BHASKARAN\nAge/Gender (Your age/gender) : 58Y/Female' }, { page: 2, text: 'Dear  RAMANI BHASKARAN (Your name)\nThank you' }]);
    expect(id).toMatchObject({ name: 'NI BHASKARAN', age: 58, gender: 'Female' });
    // "NI BHASKARAN" is a substring of "RAMANI BHASKARAN" but they are different people/names.
    expect(id.otherNames).toEqual([{ name: 'RAMANI BHASKARAN', page: 2 }]);
  });

  it('does not flag the same person written with a middle initial', () => {
    const id = extractIdentity([{ text: 'Patient Name: John Doe\nDear John A Doe,' }]);
    expect(id.otherNames).toEqual([]);
  });
});

describe('grounding', () => {
  const pages = [{ filename: 'a.pdf', page: 1, text: 'Patient reports coughing. Hemoglobin 10.2 g/dL (13 - 17). Aspirin 75 mg daily.' }];
  it('matches inflected words but requires numbers to match exactly', () => {
    expect(findTextPages('cough', pages)).toHaveLength(1);
    expect(findTextPages('aspirin', pages)).toHaveLength(1);
    expect(findTextPages('pneumonia', pages)).toHaveLength(0);
    expect(findLabPages('Hemoglobin', '10.2', pages)).toHaveLength(1);
    expect(findLabPages('Hemoglobin', '10.3', pages)).toHaveLength(0);
    expect(findLabPages('Ferritin', '10.2', pages)).toHaveLength(0);
  });
});

/* ───────────── extraction: the document's words, verified AI output ───────────── */

describe('extraction uses the document text verbatim and verifies AI output', () => {
  it('never substitutes an AI rewrite for the document text, and discards invented items', async () => {
    const { documents } = await collectSources([pdfFile(buildPdf([{ lines: REPORT_LINES }]))]);
    const result = normalizeExtraction(
      extractionOutput({
        patientDetails: { name: 'Someone Invented', age: 30, gender: 'Male' },
        labValues: [
          labOut('TRIGLYCERIDES', '199', { unit: 'mg/dL', normalRange: '0 - 149.99 mg/dL', isAbnormal: false }), // model says "normal"
          labOut('HDL CHOLESTEROL', '43', { unit: 'mg/dL', normalRange: '40 - 60 mg/dL', isAbnormal: true }), // model says "abnormal"
          labOut('FERRITIN', '8', { unit: 'ng/mL' }), // not in the document
        ],
        diagnosesMentioned: ['PRE DIABETIC STAGE', 'INVENTED PNEUMONIA'],
        fullNarrative: 'AI REWRITE THAT MUST NOT REPLACE THE DOCUMENT',
      }) as Parameters<typeof normalizeExtraction>[0],
      '',
      documents
    );

    expect(result.patientDetails).toEqual({ name: 'ZORBLAX QUENDI', age: 58, gender: 'Female' });
    expect(result.rawText).toContain('--- lab.pdf · page 1 ---');
    expect(result.rawText).toContain('TRIGLYCERIDES');
    expect(result.rawText).not.toContain('AI REWRITE');
    expect(result.report.pages[0].text).toContain('ZORBLAX QUENDI');
    expect(result.modelReadText).toBe(''); // pure-text PDF: nothing was model-read

    const byName = Object.fromEntries(result.labValues.map((l) => [l.name, l]));
    expect(Object.keys(byName)).toEqual(['TRIGLYCERIDES', 'HDL CHOLESTEROL']); // FERRITIN discarded
    expect(byName.TRIGLYCERIDES).toMatchObject({ flag: 'High', isAbnormal: true, flagSource: 'printed_range' });
    expect(byName['HDL CHOLESTEROL']).toMatchObject({ flag: 'Normal', isAbnormal: false, flagSource: 'printed_range' });
    expect(result.report.diagnosesMentioned).toEqual(['PRE DIABETIC STAGE']);
    const notes = result.report.extractionLimitations.join(' ');
    expect(notes).toMatch(/not found in the document text and were discarded/);
    expect(notes).toMatch(/Another name appears in the document \("MIRELA VANTHO", page 1\)/);
  });

  it('keeps AI-read image content apart as unverified when the PDF has images', async () => {
    const { documents } = await collectSources([pdfFile(buildPdf([{ lines: REPORT_LINES, jpeg: true }]), 'scan.pdf')]);
    expect(documents[0].needsModelReading).toBe(true);
    const result = normalizeExtraction(
      extractionOutput({
        diagnosesMentioned: ['PRE DIABETIC STAGE', 'DIAGNOSIS ONLY IN AN IMAGE'],
        fullNarrative: 'Page 1 image: a bar chart labelled Sugar trend',
      }) as Parameters<typeof normalizeExtraction>[0],
      '',
      documents
    );
    expect(result.report.diagnosesMentioned).toEqual(['PRE DIABETIC STAGE']); // verified only
    expect(result.report.unverifiedFromImages).toEqual(['Diagnosis: DIAGNOSIS ONLY IN AN IMAGE']);
    expect(result.rawText).toContain('Read by AI from images/scans (not verified against the text layer)');
    expect(result.rawText).toContain('Sugar trend');
    expect(result.modelReadText).toBe('Page 1 image: a bar chart labelled Sugar trend');
  });

  it('indexes ONLY the AI-read content as an ai_transcription document (never the verbatim text twice)', async () => {
    const user = createTestUser();
    generateText.mockResolvedValue({ output: extractionOutput({ fullNarrative: 'Page 1 image: a bar chart labelled Sugar trend' }) });
    const pdf = buildPdf([{ lines: REPORT_LINES, jpeg: true }]);
    const res = await request(app).post('/api/uploads').set('Authorization', user.auth).attach('files', pdf, { filename: 'scan.pdf', contentType: 'application/pdf' });

    expect(res.status).toBe(200);
    expect(res.body.data.documents).toHaveLength(2);
    const records = store().records(userNamespace(user.id));
    const transcription = records.filter((r) => r.metadata.source === 'ai_transcription');
    expect(transcription).toHaveLength(1);
    expect(transcription[0].metadata.text).toBe('Page 1 image: a bar chart labelled Sugar trend');
    expect(records.filter((r) => r.metadata.source === 'pdf_text').map((r) => r.metadata.text).join(' ')).toContain('TRIGLYCERIDES');
    // Internal fields are never returned to the client.
    expect(res.body.data.sourceDocuments).toBeUndefined();
    expect(res.body.data.modelReadText).toBeUndefined();
    expect(res.body.data.rawText).toContain('--- scan.pdf · page 1 ---');
  });
});

/* ───────────── assessment ───────────── */

describe('assessment stays grounded and consistent', () => {
  const labs = [
    { name: 'TRIGLYCERIDES', value: '199', unit: 'mg/dL', normalRange: '0 - 149.99 mg/dL', isAbnormal: true, flag: 'High', flagSource: 'printed_range' as const },
    { name: 'HDL CHOLESTEROL', value: '43', unit: 'mg/dL', normalRange: '40 - 60 mg/dL', isAbnormal: false, flag: 'Normal', flagSource: 'printed_range' as const },
  ];

  it('lists out-of-range labs first, includes the report impressions/scores, and never the patient name', () => {
    const findings = buildFindings({
      patientDetails: { name: 'ZORBLAX QUENDI', age: 58, gender: 'Female' },
      symptoms: [],
      medicines: [],
      labValues: [labs[1], labs[0]],
      rawText: 'ZORBLAX QUENDI long verbatim document text '.repeat(100),
      report: { diagnosesMentioned: ['PRE DIABETIC STAGE'], reportRiskScores: [{ name: 'AICVD Risk score', result: 'Low Risk' }] },
    });
    const text = findings.map((f) => f.text).join('\n');
    expect(findings[2].text).toMatch(/^Lab OUT OF RANGE — TRIGLYCERIDES/);
    expect(text).toContain('computed by comparing the value with the printed range');
    expect(text).toContain('Impression/diagnosis written in the report (by the issuing clinic) — PRE DIABETIC STAGE');
    expect(text).toContain('Risk score printed in the report (by the issuing clinic) — AICVD Risk score: Low Risk');
    expect(text).not.toMatch(/zorblax|quendi/i);
    expect(text).not.toContain('long verbatim document text'); // no raw dump when structured data exists
  });

  it('scrubs the patient name from free-text fallbacks', () => {
    expect(scrubName('Mr Zorblax Quendi has a cough', 'ZORBLAX QUENDI')).toBe('Mr [patient] [patient] has a cough');
  });

  it('traces key findings that have no ids to the finding they restate — and never guesses', () => {
    const findings = [
      { id: 'F1', text: 'Lab OUT OF RANGE — TRIGLYCERIDES: 199 mg/dL; reference range printed in the report: 0 - 149.99 mg/dL; High' },
      { id: 'F2', text: 'Lab within range — HDL CHOLESTEROL: 43 mg/dL' },
    ];
    expect(traceToFinding('Elevated Triglycerides at 199 mg/dL (reference range: 0 - 149.99 mg/dL)', findings)).toEqual(['F1']);
    expect(traceToFinding('Elevated Triglycerides at 250 mg/dL', findings)).toEqual([]); // wrong number → no match
    expect(traceToFinding('Something unrelated entirely', findings)).toEqual([]);
  });

  it('keeps a level whose findings can be traced even when the model left the ids empty (no contradictory downgrade)', () => {
    const findings = [{ id: 'F1', text: 'Lab OUT OF RANGE — TRIGLYCERIDES: 199 mg/dL; High' }];
    const result = validateAssessment(
      assessmentOutput({ riskLevel: 'medium', riskScore: 50, keyFindings: [{ finding: 'TRIGLYCERIDES 199 mg/dL is High', significance: 'x', evidenceIds: [] }] }) as Parameters<typeof validateAssessment>[0],
      findings,
      [],
      { requireReference: false } // this unit test is about tracing ids to findings, not the reference requirement
    );
    expect(result.riskLevel).toBe('medium');
    expect(result.keyFindings[0].evidenceIds).toEqual(['F1']);
    expect(result.validation.downgradedToInsufficient).toBe(false);
  });

  it('when a level really cannot be tied to any evidence, says so in the summary instead of contradicting itself', () => {
    const result = validateAssessment(
      assessmentOutput({ riskLevel: 'high', riskScore: 90, summary: 'The assessment is high risk overall.', keyFindings: [{ finding: 'Nothing traceable here', significance: 'x', evidenceIds: [] }] }) as Parameters<typeof validateAssessment>[0],
      [{ id: 'F1', text: 'Lab OUT OF RANGE — TRIGLYCERIDES: 199 mg/dL' }],
      []
    );
    expect(result.riskLevel).toBe('insufficient_evidence');
    expect(result.summary).toMatch(/^No risk level is reported/);
  });
});

/* ───────────── Q&A on this report ───────────── */

describe('questions about a report', () => {
  it('answers broad questions from the structured report even when no chunk clears a similarity floor', async () => {
    process.env.RAG_DOC_MIN_SCORE = '0.99'; // nothing from the vector store will pass
    const user = createTestUser();
    const consultation = await FakeConsultation.create({
      userId: user.id,
      status: 'completed',
      riskLevel: 'medium',
      riskScore: 55,
      patientDetails: { name: 'ZORBLAX QUENDI', age: 58, gender: 'Female' },
      symptoms: [],
      medicines: [],
      labValues: [{ name: 'TRIGLYCERIDES', value: '199', unit: 'mg/dL', normalRange: '0 - 149.99 mg/dL', flag: 'High', isAbnormal: true, source: { pages: [4, 5] } }],
      report: { diagnosesMentioned: ['PRE DIABETIC STAGE'], reportRiskScores: [{ name: 'AICVD Risk score', result: 'Low Risk' }] },
    });
    generateText.mockImplementation(async (opts: { system?: string }) =>
      isQACall(opts)
        ? { output: { answer: 'Triglycerides are high (199 mg/dL vs 0 - 149.99) [P1]; the report lists a pre-diabetic stage.', patientCitations: ['P1'], referenceCitations: [], uncertainty: '', insufficientContext: false } }
        : { output: extractionOutput() }
    );

    const res = await request(app)
      .post('/api/documents/query')
      .set('Authorization', user.auth)
      .send({ question: "What are the abnormal results and what is the doctor's impression?", consultationId: consultation._id });

    expect(res.status).toBe(200);
    expect(res.body.data.status).toBe('answered');
    expect(res.body.data.citations[0]).toMatchObject({ evidenceId: 'P1', domain: 'patient', filename: 'Structured report (extracted from your document)' });
    const prompt = generateText.mock.calls.map(([o]) => o).find(isQACall).prompt as string;
    expect(prompt).toContain('TRIGLYCERIDES: 199 mg/dL (printed range 0 - 149.99 mg/dL) — High [pp.4, 5]');
    expect(prompt).toContain('Impressions written in the report: PRE DIABETIC STAGE');
    expect(prompt).not.toMatch(/zorblax|quendi/i);
  });

  it("never exposes another user's structured report", async () => {
    const a = createTestUser();
    const b = createTestUser();
    const consultation = await FakeConsultation.create({ userId: a.id, status: 'completed', labValues: [{ name: 'SECRET LAB', value: '1', flag: 'High', isAbnormal: true }] });
    generateText.mockResolvedValue({ output: extractionOutput() });
    const res = await request(app).post('/api/documents/query').set('Authorization', b.auth).send({ question: 'What are the abnormal results?', consultationId: consultation._id });
    expect(res.body.data.status).toBe('insufficient_context');
    expect(JSON.stringify(generateText.mock.calls)).not.toContain('SECRET LAB');
  });

  it('builds no evidence for an empty consultation and never includes the name', () => {
    expect(buildStructuredEvidence({ _id: 'x', patientDetails: { age: undefined } })).toBeNull();
    const e = buildStructuredEvidence({ _id: 'x', patientDetails: { age: 40, gender: 'Male' }, labValues: [{ name: 'A', value: '1' }] });
    expect(e?.content).toMatch(/Patient: 40 years, Male/);
  });
});
