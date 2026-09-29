import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import request from 'supertest';
import fs from 'node:fs';
import path from 'node:path';

const generateText = vi.fn();
vi.mock('ai', async (importOriginal) => ({
  ...(await importOriginal<typeof import('ai')>()),
  generateText: (...args: unknown[]) => generateText(...args),
}));
vi.mock('../src/models/Consultation', async () => ({ Consultation: (await import('./helpers/fakeModels')).FakeConsultation }));
vi.mock('../src/models/User', async () => ({ User: (await import('./helpers/fakeModels')).FakeUser }));

import { createApp } from '../src/app';
import { resetRateLimits } from '../src/middleware/rateLimiter';
import { cache } from '../src/middleware/cache';
import { resetDb } from './helpers/fakeModels';
import { createTestUser } from './helpers/auth';
import { assessmentOutput, extractionOutput, labOut } from './helpers/aiOutputs';

const PHI = ['Jane Sensitive', 'crushing chest pain', 'Warfarin', 'Hemoglobin 6.1'];
const clinicalText = `Patient Name: Jane Sensitive\nComplaint: crushing chest pain\nMeds: Warfarin\nLab: Hemoglobin 6.1`;

let output: string[] = [];
const originalEnv = process.env.NODE_ENV;

beforeEach(() => {
  resetDb();
  cache.clear();
  resetRateLimits();
  generateText.mockReset();
  output = [];
  process.env.NODE_ENV = 'development'; // enable info/warn logging for this suite
  process.env.GEMINI_API_KEY = 'test-gemini';
  for (const method of ['log', 'info', 'warn', 'error', 'debug'] as const) {
    vi.spyOn(console, method).mockImplementation((...args: unknown[]) => {
      output.push(args.map((a) => (typeof a === 'string' ? a : JSON.stringify(a))).join(' '));
    });
  }
});

afterEach(() => {
  process.env.NODE_ENV = originalEnv;
  delete process.env.GEMINI_API_KEY;
});

const expectNoPhi = () => {
  const logs = output.join('\n');
  for (const value of PHI) expect(logs).not.toContain(value);
};

describe('PHI is never logged', () => {
  it('successful extraction + consultation logs metadata only', async () => {
    const app = createApp();
    const user = createTestUser();
    generateText
      .mockResolvedValueOnce({
        output: extractionOutput({
          patientDetails: { name: 'Jane Sensitive', age: 50, gender: 'F' },
          symptoms: ['crushing chest pain'],
          medicines: ['Warfarin'],
          labValues: [labOut('Hemoglobin', '6.1', { unit: 'g/dL', isAbnormal: true })],
          fullNarrative: clinicalText,
        }),
      })
      .mockResolvedValueOnce({ output: assessmentOutput({ summary: 'Jane Sensitive has crushing chest pain', riskLevel: 'high', riskScore: 95 }) });

    const res = await request(app)
      .post(`/api/consultations`)
      .set('Authorization', user.auth)
      .send({ rawText: clinicalText });

    expect(res.status).toBe(201);
    expect(output.join('\n')).toContain('ai.call');
    expectNoPhi();
  });

  it('failures (including provider errors that echo input) log no PHI', async () => {
    const app = createApp();
    const user = createTestUser();
    generateText.mockRejectedValue(new Error(`Invalid request: ${clinicalText}`));

    const upload = await request(app)
      .post('/api/uploads')
      .set('Authorization', user.auth)
      .field('text', clinicalText);
    expect(upload.status).toBe(502);
    expect(JSON.stringify(upload.body)).not.toContain('Jane Sensitive');

    const search = await request(app)
      .get('/api/consultations?search=Jane%20Sensitive')
      .set('Authorization', user.auth);
    expect(search.status).toBe(200);

    expectNoPhi();
  });

  it('backend source has no raw console logging of clinical data', () => {
    const offenders: string[] = [];
    const walk = (dir: string) => {
      for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
        const full = path.join(dir, entry.name);
        if (entry.isDirectory()) walk(full);
        else if (full.endsWith('.ts')) {
          const src = fs.readFileSync(full, 'utf8');
          if (/console\.(log|dir|info|debug)\([^)]*(prompt|text|body|payload|symptom|patient|output|response)/i.test(src)) {
            offenders.push(full);
          }
        }
      }
    };
    walk(path.join(__dirname, '..', 'src'));
    expect(offenders).toEqual([]);
  });
});
