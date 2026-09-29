import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';

const generateText = vi.fn();
vi.mock('ai', async (importOriginal) => ({
  ...(await importOriginal<typeof import('ai')>()),
  generateText: (...args: unknown[]) => generateText(...args),
}));

import {
  extractMedicalData,
  getModelCandidates,
  normalizeExtraction,
  withModelFallback,
  getOpenCircuits,
  AIConfigurationError,
  AIProcessingError,
  type ModelCandidate,
} from '../src/services/ai.service';
import { extractionOutput, labOut } from './helpers/aiOutputs';

const PROVIDER_KEYS = ['GEMINI_API_KEY', 'GOOGLE_API_KEY', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GEMINI_MODEL'];
const saved: Record<string, string | undefined> = {};

const extraction = extractionOutput({
  symptoms: [' fever ', ''],
  medicines: ['Paracetamol'],
  labValues: [labOut('Hb', '9.1', { unit: 'g/dL', normalRange: '12-16', isAbnormal: true }), labOut('', '')],
  fullNarrative: 'Narrative',
});

const fakeFile = (mimetype: string, content: Buffer | string, originalname = 'file') => {
  const buffer = Buffer.isBuffer(content) ? content : Buffer.from(content);
  return { mimetype, buffer, size: buffer.length, originalname } as Express.Multer.File;
};
const fake = (id: string): ModelCandidate => ({ provider: 'google', modelId: id, model: {} as ModelCandidate['model'] });
const apiError = (statusCode: number) => Object.assign(new Error(`status ${statusCode}`), { name: 'AI_APICallError', statusCode });

beforeEach(() => {
  for (const key of PROVIDER_KEYS) {
    saved[key] = process.env[key];
    delete process.env[key];
  }
  generateText.mockReset();
});

afterEach(() => {
  for (const key of PROVIDER_KEYS) {
    if (saved[key] === undefined) delete process.env[key];
    else process.env[key] = saved[key];
  }
});

describe('provider configuration', () => {
  it('uses only direct providers with a configurable Gemini primary (default gemini-3.5-flash-lite)', () => {
    process.env.GEMINI_API_KEY = 'test-gemini';
    process.env.OPENAI_API_KEY = 'test-openai';
    const candidates = getModelCandidates();
    expect(candidates.map((c) => c.provider)).toEqual(['google', 'google', 'openai']);
    expect(candidates.map((c) => c.modelId)).toEqual(['gemini-3.5-flash-lite', 'gemini-2.5-flash-lite', 'gpt-4o-mini']);

    process.env.GEMINI_MODEL = 'custom-model';
    expect(getModelCandidates()[0].modelId).toBe('custom-model');
  });

  it('fails clearly when no provider key is configured', () => {
    expect(() => getModelCandidates()).toThrow(AIConfigurationError);
  });
});

describe('circuit breaker', () => {
  it('stops calling a model that returned 404 (retired/unknown) and uses the fallback', async () => {
    const run = vi.fn(async ({ candidate }: { candidate: ModelCandidate }) => {
      if (candidate.modelId === 'primary') throw apiError(404);
      return { value: 'ok' };
    });
    await withModelFallback('op', run, [fake('primary'), fake('fallback')]);
    await withModelFallback('op', run, [fake('primary'), fake('fallback')]);
    expect(run.mock.calls.filter(([c]) => c.candidate.modelId === 'primary')).toHaveLength(1);
    expect(getOpenCircuits()[0]).toMatchObject({ model: 'google:primary', category: 'model_unavailable' });
  });

  it('opens on 429 and fails fast when every model is rate limited', async () => {
    const run = vi.fn(async () => {
      throw apiError(429);
    });
    const first = await withModelFallback('op', run, [fake('a')]).catch((e) => e);
    expect(first.category).toBe('provider_rate_limited');
    const second = await withModelFallback('op', run, [fake('a')]).catch((e) => e);
    expect(second).toBeInstanceOf(AIProcessingError);
    expect(second.category).toBe('provider_rate_limited');
    expect(run).toHaveBeenCalledTimes(1);
  });

  it('does not open for transient 5xx errors', async () => {
    const run = vi.fn(async () => {
      throw apiError(503);
    });
    await withModelFallback('op', run, [fake('a')]).catch(() => undefined);
    await withModelFallback('op', run, [fake('a')]).catch(() => undefined);
    expect(run).toHaveBeenCalledTimes(2);
  });
});

describe('extractMedicalData (Job A)', () => {
  it('falls back to the next model when one fails and normalises output', async () => {
    process.env.GEMINI_API_KEY = 'test-gemini';
    generateText.mockRejectedValueOnce(new Error('model down')).mockResolvedValueOnce({ output: extraction });

    const result = await extractMedicalData('Patient Name: John Doe\nAge: 42\nSex: Male\nfever');

    expect(generateText).toHaveBeenCalledTimes(2);
    expect(result.patientDetails).toEqual({ name: 'John Doe', age: 42, gender: 'Male' });
    expect(result.symptoms).toEqual(['fever']);
    // 'Hb 9.1' is not in the clinician's text, so the AI-proposed lab is discarded (never merged as fact).
    expect(result.labValues).toEqual([]);
    // Both the invented lab (Hb) and the invented medicine (Paracetamol) are discarded.
    expect(result.medicines).toEqual([]);
    expect(result.report.extractionLimitations.join(' ')).toMatch(/2 item\(s\) proposed by the AI were not found in the document text/);
    // The clinician's own words are kept verbatim; the model's narrative is not substituted for them.
    expect(result.rawText).toBe('Patient Name: John Doe\nAge: 42\nSex: Male\nfever');
    expect(result.report).toMatchObject({ documents: [], tables: [], diagnosesMentioned: [] });
  });

  it('keeps AI-extracted labs that are present in the text and computes flags from the printed range', async () => {
    process.env.GEMINI_API_KEY = 'test-gemini';
    generateText.mockResolvedValue({
      output: extractionOutput({
        labValues: [
          // The model claims "normal" for both; code must decide from the printed ranges.
          labOut('Hemoglobin', '10.2', { unit: 'g/dL', normalRange: '13 - 17 g/dL', isAbnormal: false }),
          labOut('Sodium', '139', { unit: 'mmol/L', normalRange: '136 - 146 mmol/L', isAbnormal: true }),
          labOut('Glucose', '99', { unit: 'mg/dL', normalRange: '70 - 60 mg/dL' }), // range not printed in the text → ignored
        ],
      }),
    });
    const result = await extractMedicalData('Hemoglobin 10.2 g/dL (13 - 17 g/dL)\nSodium 139 mmol/L (136 - 146 mmol/L)\nGlucose 99 mg/dL');
    const byName = Object.fromEntries(result.labValues.map((l) => [l.name, l]));
    expect(byName.Hemoglobin).toMatchObject({ flag: 'Low', isAbnormal: true, flagSource: 'printed_range' });
    expect(byName.Sodium).toMatchObject({ flag: 'Normal', isAbnormal: false, flagSource: 'printed_range' });
    expect(byName.Glucose).toMatchObject({ flagSource: 'model' });
    expect(byName.Glucose.normalRange).toBeUndefined();
  });

  it('takes name/age/gender from labelled fields, reports a different salutation name, and never trusts a model-invented name', async () => {
    process.env.GEMINI_API_KEY = 'test-gemini';
    generateText.mockResolvedValue({ output: extractionOutput({ patientDetails: { name: 'Invented Person', age: 30, gender: 'Male' } }) });
    const text = 'Health Report\nPatient Name (Your name) :  NI BHASKARAN\nAge/Gender (Your age/gender) : 58Y/Female\nDear  RAMANI BHASKARAN (Your name)';
    const result = await extractMedicalData(text);
    expect(result.patientDetails).toEqual({ name: 'NI BHASKARAN', age: 58, gender: 'Female' });
    expect(result.report.extractionLimitations.join(' ')).toMatch(/Another name appears in the document \("RAMANI BHASKARAN"\)/);

    generateText.mockResolvedValue({ output: extractionOutput({ patientDetails: { name: 'Invented Person', age: null, gender: '' } }) });
    const noLabel = await extractMedicalData('Routine visit, no identifying fields in this note.');
    expect(noLabel.patientDetails.name).toBe(''); // "Invented Person" is not in the text → discarded
  });

  it('throws when all models fail', async () => {
    process.env.GEMINI_API_KEY = 'test-gemini';
    generateText.mockRejectedValue(new Error('down'));
    await expect(extractMedicalData('fever')).rejects.toThrow(/All configured AI models failed/);
  });

  it('rejects empty input without calling a model', async () => {
    process.env.GEMINI_API_KEY = 'test-gemini';
    await expect(extractMedicalData('   ')).rejects.toThrow(/No medical data/);
    expect(generateText).not.toHaveBeenCalled();
  });

  it('sends images natively, includes text files, and uses the LangChain extraction prompt', async () => {
    process.env.OPENAI_API_KEY = 'test-openai';
    generateText.mockResolvedValue({ output: extraction });

    await extractMedicalData('', [fakeFile('image/png', Buffer.from([1, 2, 3]), 'scan.png'), fakeFile('text/plain', 'BP 150/95', 'bp.txt')]);

    const call = generateText.mock.calls[0][0];
    expect(call.system).toContain('clinical document extraction assistant');
    expect(call.system).toContain('DATA HANDLING RULES');
    const content = call.messages[0].content;
    expect(content[0].text).toContain('BP 150/95');
    expect(content[0].text).toContain('<data>');
    expect(content.some((p: { type: string }) => p.type === 'image')).toBe(true);
  });

  it('routes audio only to Gemini and errors if Gemini is not configured', async () => {
    process.env.OPENAI_API_KEY = 'test-openai';
    const audio = fakeFile('audio/wav', 'RIFF', 'note.wav');
    await expect(extractMedicalData('', [audio])).rejects.toBeInstanceOf(AIConfigurationError);

    process.env.GEMINI_API_KEY = 'test-gemini';
    generateText.mockResolvedValue({ output: extraction });
    await extractMedicalData('', [audio]);
    const call = generateText.mock.calls[0][0];
    expect(call.messages[0].content.some((p: { type: string }) => p.type === 'file')).toBe(true);
  });
});

describe('normalizeExtraction', () => {
  it('backfills demographics from source text', () => {
    const result = normalizeExtraction(extraction, 'Age/Sex: 35 / F');
    expect(result.patientDetails.age).toBe(35);
    expect(result.patientDetails.gender).toBe('Female'); // normalised from the labelled "F"
  });
});
