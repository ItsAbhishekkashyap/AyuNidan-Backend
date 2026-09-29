import { describe, it, expect, beforeEach, vi } from 'vitest';
import request from 'supertest';

vi.mock('../src/models/Consultation', async () => ({ Consultation: (await import('./helpers/fakeModels')).FakeConsultation }));
vi.mock('../src/models/User', async () => ({ User: (await import('./helpers/fakeModels')).FakeUser }));
vi.mock('../src/services/ai.service', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../src/services/ai.service')>()),
  extractMedicalData: vi.fn(async () => ({
    patientDetails: { name: 'Uploaded Patient', gender: '' },
    symptoms: ['fever'],
    medicines: [],
    labValues: [],
    rawText: 'narrative',
  })),
}));
vi.mock('../src/services/assessment.service', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../src/services/assessment.service')>()),
  assessRisk: vi.fn(async () => (await import('./helpers/aiOutputs')).fakeClinicalAssessment('medium', 50)),
}));
vi.mock('../src/services/rag.service', () => ({
  seedMedicalKnowledgeBase: vi.fn(async () => 'seeded'),
  explainMedicalTermRAG: vi.fn(async () => ({ explanation: 'explanation', grounded: true, sources: [] })),
}));

import { createApp } from '../src/app';
import { cache } from '../src/middleware/cache';
import { resetRateLimits } from '../src/middleware/rateLimiter';
import { extractMedicalData } from '../src/services/ai.service';
import { seedMedicalKnowledgeBase } from '../src/services/rag.service';
import { db, resetDb } from './helpers/fakeModels';
import { createTestUser } from './helpers/auth';
import { pdfBytes, wavBytes } from './helpers/fixtures';

const app = createApp();

beforeEach(() => {
  resetDb();
  cache.clear();
  resetRateLimits();
  vi.mocked(extractMedicalData).mockClear();
  vi.mocked(seedMedicalKnowledgeBase).mockClear();
});

describe('POST /api/uploads', () => {
  it('rejects unauthenticated uploads before processing files', async () => {
    const res = await request(app).post('/api/uploads').attach('files', pdfBytes(), { filename: 'r.pdf', contentType: 'application/pdf' });
    expect(res.status).toBe(401);
    expect(extractMedicalData).not.toHaveBeenCalled();
  });

  it('rejects invalid and forged tokens', async () => {
    const bad = await request(app).post('/api/uploads').set('Authorization', 'Bearer not-a-jwt').field('text', 'fever');
    expect(bad.status).toBe(401);

    const user = createTestUser();
    db.users.delete(user.id);
    const orphan = await request(app).post('/api/uploads').set('Authorization', user.auth).field('text', 'fever');
    expect(orphan.status).toBe(401);
    expect(extractMedicalData).not.toHaveBeenCalled();
  });

  it('allows authenticated users to upload a legitimate report', async () => {
    const user = createTestUser();
    const res = await request(app)
      .post('/api/uploads')
      .set('Authorization', user.auth)
      .attach('files', pdfBytes(), { filename: 'report.pdf', contentType: 'application/pdf' });

    expect(res.status).toBe(200);
    expect(res.body.data.symptoms).toEqual(['fever']);
    expect(extractMedicalData).toHaveBeenCalledTimes(1);
  });

  it("does not persist or expose one user's extraction to another user", async () => {
    const a = createTestUser();
    const b = createTestUser();
    await request(app).post('/api/uploads').set('Authorization', a.auth).field('text', 'fever');

    expect(db.consultations).toHaveLength(0);
    const list = await request(app).get('/api/consultations').set('Authorization', b.auth);
    expect(JSON.stringify(list.body)).not.toContain('Uploaded Patient');

    // There is no route that serves stored uploads.
    const serve = await request(app).get('/api/uploads').set('Authorization', b.auth);
    expect(serve.status).toBe(404);
  });
});

describe('POST /api/voice/transcribe', () => {
  it('rejects unauthenticated requests', async () => {
    const res = await request(app)
      .post('/api/voice/transcribe')
      .attach('audio', wavBytes(), { filename: 'note.wav', contentType: 'audio/wav' });
    expect(res.status).toBe(401);
    expect(extractMedicalData).not.toHaveBeenCalled();
  });

  it('processes audio for authenticated users', async () => {
    const user = createTestUser();
    const res = await request(app)
      .post('/api/voice/transcribe')
      .set('Authorization', user.auth)
      .attach('audio', wavBytes(), { filename: 'note.wav', contentType: 'audio/wav' });
    expect(res.status).toBe(200);
    expect(extractMedicalData).toHaveBeenCalledTimes(1);
  });
});

describe('debug environment endpoint', () => {
  it('is removed and exposes no key information', async () => {
    process.env.GEMINI_API_KEY = 'AIzaTEST-secret-value';
    const res = await request(app).get('/api/debug-env-tokens-isolated');
    expect(res.status).toBe(404);
    expect(JSON.stringify(res.body)).not.toMatch(/AIza|keyPrefix|keyLength/);
    delete process.env.GEMINI_API_KEY;
  });

  it('health endpoint does not expose process memory or env', async () => {
    const res = await request(app).get('/health');
    expect(res.status).toBe(200);
    expect(res.body.memory).toBeUndefined();
    expect(res.body.nodeEnv).toBeUndefined();
  });
});

describe('RAG seeding', () => {
  it('is not reachable over HTTP, even for authenticated users', async () => {
    const anon = await request(app).post('/api/consultations/seed-rag');
    expect(anon.status).toBe(404);

    const user = createTestUser();
    const authed = await request(app).post('/api/consultations/seed-rag').set('Authorization', user.auth);
    expect(authed.status).toBe(404);

    expect(seedMedicalKnowledgeBase).not.toHaveBeenCalled();
  });
});

describe('CORS', () => {
  const preflight = (origin: string) =>
    request(app).options('/api/uploads').set('Origin', origin).set('Access-Control-Request-Method', 'POST').set('Access-Control-Request-Headers', 'authorization');

  it('allows the local dev frontend on localhost and 127.0.0.1', async () => {
    for (const origin of ['http://localhost:3000', 'http://127.0.0.1:3000', 'http://localhost:3001']) {
      expect((await preflight(origin)).headers['access-control-allow-origin']).toBe(origin);
    }
  });

  it('does not allow arbitrary origins', async () => {
    expect((await preflight('https://evil.example')).headers['access-control-allow-origin']).toBeUndefined();
  });
});
