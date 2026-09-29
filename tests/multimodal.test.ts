/**
 * Phase 8: multimodal document understanding + voice pipeline. Synthetic fixtures only;
 * the model is mocked at generateText (no paid calls).
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
import { retrieveDocumentChunks, userNamespace } from '../src/services/document.service';
import { getVectorStore } from '../src/rag/vectorStore';
import { InMemoryVectorStore } from '../src/rag/inMemoryVectorStore';
import { validateAudio, wavDurationSec, AudioValidationError } from '../src/documents/audio';
import { buildPdf, LAB_COLUMNS } from '../src/eval/pdfBuilder';
import { getFailureCounts, resetFailureCounts } from '../src/utils/failures';
import { resetDb } from './helpers/fakeModels';
import { createTestUser } from './helpers/auth';
import { pngBytes, wavBytes, webmBytes } from './helpers/fixtures';
import { extractionOutput, labOut } from './helpers/aiOutputs';

const app = createApp();
const store = () => getVectorStore() as InMemoryVectorStore;
const file = (mimetype: string, buffer: Buffer, originalname: string) => ({ mimetype, buffer, size: buffer.length, originalname }) as Express.Multer.File;

const CBC_TABLE = { columns: LAB_COLUMNS, rows: [['Test', 'Result', 'Unit', 'Reference Range', 'Flag'], ['Hemoglobin', '10.2', 'g/dL', '13-17', 'Low'], ['Platelets', '450', 'x10^9/L', '150-400', 'High']] };
const MULTI_PAGE_PDF = () =>
  buildPdf([
    { lines: ['FICTIONAL HOSPITAL REPORT', 'Patient: Zorblax Quendi (fictional)', 'Date: 2031-02-03'] },
    { lines: ['Complete blood count'], table: CBC_TABLE },
    { image: true }, // scanned page
    { lines: ['Doctor notes: review in two weeks (fictional).'], image: true }, // text + embedded image
  ]);

let user: ReturnType<typeof createTestUser>;
beforeEach(() => {
  resetDb();
  cache.clear();
  resetRateLimits();
  resetFailureCounts();
  generateText.mockReset();
  generateText.mockResolvedValue({ output: extractionOutput({ fullNarrative: 'Synthetic transcript of the fictional report.' }) });
  process.env.GEMINI_API_KEY = 'test-gemini';
  process.env.RAG_DOC_MIN_SCORE = '0.05';
  user = createTestUser();
});
afterEach(() => {
  for (const k of ['GEMINI_API_KEY', 'RAG_DOC_MIN_SCORE', 'VOICE_MAX_DURATION_SEC']) delete process.env[k];
});

/* ───────────── Normalised representation ───────────── */

describe('multimodal normalisation', () => {
  it('converges PDF, image, text and audio into one representation without dropping pages', async () => {
    const { documents, mediaParts, hasAudio } = await collectSources([
      file('application/pdf', MULTI_PAGE_PDF(), 'report.pdf'),
      file('image/png', pngBytes(), 'photo.png'),
      file('text/plain', Buffer.from('Test | Result | Unit | Reference Range | Flag\nSodium | 139 | mmol/L | 135-145 | N'), 'notes.txt'),
      file('audio/wav', wavBytes(), 'note.wav'),
    ]);

    expect(documents.map((d) => d.sourceType)).toEqual(['pdf', 'image', 'text', 'audio']);
    const pdf = documents[0];
    expect(pdf.pages.map((p) => p.status)).toEqual(['text', 'text', 'scanned', 'mixed']);
    expect(pdf.needsModelReading).toBe(true);
    expect(pdf.notes.join('\n')).toMatch(/page 3: no text layer/);
    expect(pdf.notes.join('\n')).toMatch(/page 4: contains 1 embedded image/);
    expect(pdf.tables[0].labRows[0]).toMatchObject({ test: 'Hemoglobin', page: 2 });
    expect(documents[2].tables[0].labRows[0]).toMatchObject({ test: 'Sodium', flag: 'Normal' });
    expect(mediaParts.map((p) => p.type)).toEqual(['file', 'image', 'file']); // PDF (for scanned pages), image, audio
    expect(hasAudio).toBe(true);
  });

  it('handles scanned PDFs with JPEG (DCT) images without crashing the process', async () => {
    // Regression: pdf.js decodes JPEG streams through the browser-only `Image` class. Without the
    // stub in pdfAnalyzer this raised an uncaught ReferenceError that killed the API on real scans.
    const unhandled: unknown[] = [];
    const onUnhandled = (e: unknown) => unhandled.push(e);
    process.on('uncaughtException', onUnhandled);
    process.on('unhandledRejection', onUnhandled);
    try {
      const jpegPdf = buildPdf([{ lines: ['Fictional cover page with enough text to count as a text page.'] }, { jpeg: true }, { lines: ['Fictional notes page with plenty of readable text.'], jpeg: true }]);
      const { documents, mediaParts } = await collectSources([file('application/pdf', jpegPdf, 'scan.pdf')]);
      await new Promise((r) => setTimeout(r, 50)); // let pdf.js message callbacks run
      expect(documents[0].pages.map((p) => [p.status, p.imageCount])).toEqual([['text', 0], ['scanned', 1], ['mixed', 1]]);
      expect(documents[0].needsModelReading).toBe(true);
      expect(mediaParts).toHaveLength(1);
      expect(unhandled).toEqual([]);
    } finally {
      process.off('uncaughtException', onUnhandled);
      process.off('unhandledRejection', onUnhandled);
    }
  });

  it('does not send a fully text-based PDF to the model as a file (cost control)', async () => {
    const { documents, mediaParts } = await collectSources([file('application/pdf', buildPdf([{ lines: ['Fictional text only report page.'] }]), 'text.pdf')]);
    expect(documents[0].needsModelReading).toBe(false);
    expect(mediaParts).toEqual([]);
  });

  it('reports corrupted PDFs explicitly and falls back to the vision model', async () => {
    const { documents, mediaParts } = await collectSources([file('application/pdf', Buffer.from('%PDF-1.4 corrupted fictional'), 'broken.pdf')]);
    expect(documents[0].notes[0]).toMatch(/could not be parsed locally/);
    expect(mediaParts).toHaveLength(1);
    expect(getFailureCounts().extraction_failure).toBe(1);
  });

  it('attaches verifiable provenance to model-extracted labs (single page only, never guessed)', async () => {
    const { documents } = await collectSources([
      file('application/pdf', buildPdf([{ lines: ['Ferritin 8 ng/mL low (fictional)'] }, { lines: ['Sodium 139 mmol/L'] }, { lines: ['Sodium 139 mmol/L repeated'] }]), 'r.pdf'),
    ]);
    const result = normalizeExtraction(
      extractionOutput({ labValues: [labOut('Ferritin', '8'), labOut('Sodium', '139'), labOut('Vitamin D', '12')] }) as Parameters<typeof normalizeExtraction>[0],
      '',
      documents
    );
    // 'Vitamin D 12' is nowhere in the document → discarded (a text-only PDF has nothing the model could legitimately add).
    expect(result.labValues.map((l) => l.name)).toEqual(['Ferritin', 'Sodium']);
    expect(result.labValues.map((l) => l.source)).toEqual([
      { method: 'text_match', filename: 'r.pdf', page: 1 },
      { method: 'text_match', filename: 'r.pdf', pages: [2, 3] }, // on two pages → no single page claimed; both listed
    ]);
    expect(result.report.extractionLimitations.join(' ')).toMatch(/Lab: Vitamin D 12/);
  });
});

/* ───────────── Upload pipeline (Job A + indexing) ───────────── */

describe('upload: comprehensive report extraction', () => {
  it('extracts a multi-page multimodal upload with ONE model call and indexes structured chunks', async () => {
    generateText.mockResolvedValue({
      output: extractionOutput({
        dates: ['2031-02-03'],
        diagnosesMentioned: ['fictional anaemia (as written)', 'invented pneumonia'],
        imagingFindings: ['Page 3 scan: text illegible'],
        extractionLimitations: ['Page 3 image quality too low to read reliably'],
        doctorNotes: 'Review in two weeks.',
        fullNarrative: 'Fictional transcript including the scanned page.',
      }),
    });
    const res = await request(app)
      .post('/api/uploads')
      .set('Authorization', user.auth)
      .attach('files', MULTI_PAGE_PDF(), { filename: 'report.pdf', contentType: 'application/pdf' })
      .attach('files', pngBytes(), { filename: 'photo.png', contentType: 'image/png' });

    expect(res.status).toBe(200);
    expect(generateText).toHaveBeenCalledTimes(1); // one Job A call for all files; indexing uses no LLM
    const call = generateText.mock.calls[0][0];
    expect(call.messages[0].content[0].text).toContain('page 3 — NO TEXT LAYER');
    expect(call.messages[0].content[0].text).toContain('Hemoglobin | 10.2 | g/dL | 13-17 | Low');

    const report = res.body.data.report;
    expect(report.documents[0]).toMatchObject({ filename: 'report.pdf', sourceType: 'pdf', pageCount: 4 });
    expect(report.documents[0].pages.map((p: { status: string }) => p.status)).toEqual(['text', 'text', 'scanned', 'mixed']);
    expect(report.documents[1]).toMatchObject({ filename: 'photo.png', sourceType: 'image' });
    expect(report.tables).toHaveLength(2);
    // Verbatim page text comes from the file itself, not from the model.
    expect(report.pages.map((p: { page: number }) => p.page)).toEqual([1, 2, 3, 4]);
    expect(report.pages[0].text).toContain('FICTIONAL HOSPITAL REPORT');
    expect(res.body.data.rawText).toContain('--- report.pdf · page 1 ---');
    expect(res.body.data.rawText).toContain('Read by AI from images/scans');
    // The page-3/4 PDF has images, so an item the model reads that is NOT in the text layer is kept apart as unverified.
    expect(report.dates).toEqual(['2031-02-03']);
    expect(report.diagnosesMentioned).toEqual([]);
    expect(report.unverifiedFromImages).toEqual(expect.arrayContaining(['Diagnosis: fictional anaemia (as written)', 'Diagnosis: invented pneumonia']));
    expect(report.doctorNotes).toBe('Review in two weeks.');
    expect(report.extractionLimitations.join(' ')).toMatch(/no text layer/);
    expect(report.extractionLimitations).toContain('Page 3 image quality too low to read reliably');

    // Indexed: PDF text + table chunks with pages, plus an AI-transcription doc for model-read content.
    const records = store().records(userNamespace(user.id));
    expect(records.some((r) => r.metadata.contentType === 'table' && r.metadata.page === 2)).toBe(true);
    expect(records.some((r) => r.metadata.contentType === 'transcription')).toBe(true);
    expect(res.body.data.documents.map((d: { status: string }) => d.status)).toEqual(['indexed', 'indexed']);
  });

  it('retrieves structured table evidence after normalisation', async () => {
    await request(app)
      .post('/api/uploads')
      .set('Authorization', user.auth)
      .attach('files', buildPdf([{ table: CBC_TABLE }]), { filename: 'cbc.pdf', contentType: 'application/pdf' });
    const result = await retrieveDocumentChunks({ userId: user.id, question: 'Hemoglobin result reference range flag low' });
    expect(result.chunks[0]).toMatchObject({ contentType: 'table', page: 1 });
    expect(result.chunks[0].text).toContain('Hemoglobin:\nResult = 10.2 g/dL\nReference range = 13-17 g/dL\nFlag = Low');
  });
});

/* ───────────── Voice ───────────── */

describe('voice pipeline (server fallback)', () => {
  const voice = () => request(app).post('/api/voice/transcribe').set('Authorization', user.auth);

  it('returns a FINAL transcript (never labelled live) for valid audio', async () => {
    generateText.mockResolvedValue({ output: extractionOutput({ symptoms: ['cough'], fullNarrative: 'I have had a cough for three days.' }) });
    const res = await voice().attach('audio', wavBytes(2), { filename: 'note.wav', contentType: 'audio/wav' });
    expect(res.status).toBe(200);
    expect(res.body.data).toMatchObject({ transcript: 'I have had a cough for three days.', transcriptType: 'final', durationSec: 2 });
  });

  it('rejects empty/too-short audio explicitly', async () => {
    const res = await voice().attach('audio', webmBytes().subarray(0, 100), { filename: 'n.webm', contentType: 'audio/webm' });
    expect(res.status).toBe(400);
    expect(res.body.failureCategory).toBe('transcription_failure');
    expect(generateText).not.toHaveBeenCalled();
  });

  it('rejects corrupted WAV headers', async () => {
    const corrupted = Buffer.concat([Buffer.from('RIFF\x00\x00\x00\x00WAVEjunk'), Buffer.alloc(4000, 1)]);
    const res = await voice().attach('audio', corrupted, { filename: 'n.wav', contentType: 'audio/wav' });
    expect(res.status).toBe(400);
    expect(res.body.error).toMatch(/Corrupted WAV/);
  });

  it('enforces the duration limit', async () => {
    process.env.VOICE_MAX_DURATION_SEC = '1';
    const res = await voice().attach('audio', wavBytes(3), { filename: 'long.wav', contentType: 'audio/wav' });
    expect(res.status).toBe(413);
    expect(res.body.failureCategory).toBe('transcription_failure');
  });

  it('treats an empty transcription as an explicit failure, not success', async () => {
    generateText.mockResolvedValue({ output: extractionOutput() });
    const res = await voice().attach('audio', wavBytes(), { filename: 'silence.wav', contentType: 'audio/wav' });
    expect(res.status).toBe(422);
    expect(getFailureCounts().transcription_failure).toBe(1);
  });

  it('surfaces provider failure / timeout as categorised errors', async () => {
    generateText.mockRejectedValue(Object.assign(new Error('timeout'), { name: 'TimeoutError' }));
    const res = await voice().attach('audio', wavBytes(), { filename: 'n.wav', contentType: 'audio/wav' });
    expect(res.status).toBe(502);
    expect(res.body.failureCategory).toBe('provider_timeout');
  });

  it('validates audio deterministically', () => {
    expect(wavDurationSec(wavBytes(2))).toBe(2);
    expect(() => validateAudio({ buffer: Buffer.alloc(10), mimetype: 'audio/webm', size: 10 })).toThrow(AudioValidationError);
    expect(validateAudio({ buffer: webmBytes(), mimetype: 'audio/webm', size: 4100 })).toEqual({});
  });
});
