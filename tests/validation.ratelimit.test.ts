import { describe, it, expect, beforeEach, vi } from 'vitest';
import request from 'supertest';
import bcrypt from 'bcryptjs';

vi.mock('../src/models/Consultation', async () => ({ Consultation: (await import('./helpers/fakeModels')).FakeConsultation }));
vi.mock('../src/models/User', async () => ({ User: (await import('./helpers/fakeModels')).FakeUser }));
vi.mock('../src/services/ai.service', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../src/services/ai.service')>()),
  extractMedicalData: vi.fn(),
}));
vi.mock('../src/services/assessment.service', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../src/services/assessment.service')>()),
  assessRisk: vi.fn(async () => (await import('./helpers/aiOutputs')).fakeClinicalAssessment('low', 10)),
}));
vi.mock('../src/services/rag.service', () => ({
  seedMedicalKnowledgeBase: vi.fn(),
  explainMedicalTermRAG: vi.fn(async () => ({ explanation: 'explanation', grounded: true, sources: [] })),
}));

import { createApp } from '../src/app';
import { cache } from '../src/middleware/cache';
import { resetRateLimits } from '../src/middleware/rateLimiter';
import { db, resetDb, FakeConsultation, FakeUser } from './helpers/fakeModels';
import { createTestUser } from './helpers/auth';

const app = createApp();

beforeEach(() => {
  resetDb();
  cache.clear();
  resetRateLimits();
});

describe('auth input validation', () => {
  it('rejects malformed registration input with a readable message', async () => {
    const badEmail = await request(app).post('/api/auth/register').send({ name: 'A', email: 'nope', password: 'longenough1' });
    expect(badEmail.status).toBe(400);
    expect(badEmail.body.message).toMatch(/email/i);

    const shortPassword = await request(app).post('/api/auth/register').send({ name: 'A', email: 'a@b.co', password: 'short' });
    expect(shortPassword.status).toBe(400);

    const operator = await request(app).post('/api/auth/login').send({ email: { $ne: null }, password: 'x' });
    expect(operator.status).toBe(400);
  });

  it('does not echo passwords in validation errors', async () => {
    const res = await request(app).post('/api/auth/register').send({ name: 'A', email: 'a@b.co', password: 'secretX' });
    expect(JSON.stringify(res.body)).not.toContain('secretX');
  });
});

describe('consultation input validation', () => {
  it('validates ids, pagination and search bounds', async () => {
    const user = createTestUser();
    const get = (url: string) => request(app).get(url).set('Authorization', user.auth);

    expect((await get('/api/consultations/not-an-id')).status).toBe(400);
    expect((await get('/api/consultations?limit=1000')).status).toBe(400);
    expect((await get('/api/consultations?page=0')).status).toBe(400);
    expect((await get(`/api/consultations?search=${'a'.repeat(101)}`)).status).toBe(400);
    expect((await get('/api/consultations?page=2&limit=5')).status).toBe(200);
  });

  it('treats search input as a literal, not a regex', async () => {
    const user = createTestUser();
    await FakeConsultation.create({ userId: user.id, patientDetails: { name: 'Alice' }, summary: 's', riskLevel: 'low', riskScore: 1 });

    const wildcard = await request(app).get('/api/consultations?search=.*').set('Authorization', user.auth);
    expect(wildcard.status).toBe(200);
    expect(wildcard.body.data).toEqual([]);

    const exact = await request(app).get('/api/consultations?search=ali').set('Authorization', user.auth);
    expect(exact.body.data).toHaveLength(1);
  });

  it('validates RAG query input', async () => {
    const user = createTestUser();
    expect((await request(app).get('/api/consultations/explain?term=').set('Authorization', user.auth)).status).toBe(400);
    expect((await request(app).get(`/api/consultations/explain?term=${'x'.repeat(101)}`).set('Authorization', user.auth)).status).toBe(400);
    expect((await request(app).get('/api/consultations/explain?term=tachycardia').set('Authorization', user.auth)).status).toBe(200);
  });

  it('validates consultation creation payloads', async () => {
    const user = createTestUser();
    const post = (body: object) => request(app).post('/api/consultations').set('Authorization', user.auth).send(body);

    expect((await post({})).status).toBe(400);
    expect((await post({ rawText: 'x', symptoms: Array(101).fill('cough') })).status).toBe(400);
    expect((await post({ rawText: 'x', labValues: [{ name: 'Hb' }] })).status).toBe(400);
    expect((await post({ rawText: 'x', patientDetails: { name: 'P', age: 400 } })).status).toBe(400);
    expect((await post({ rawText: 'a'.repeat(300_001) })).status).toBe(400);
  });

  it('ignores client-supplied userId and binds records to the authenticated user', async () => {
    const a = createTestUser();
    const b = createTestUser();
    const res = await request(app)
      .post('/api/consultations')
      .set('Authorization', a.auth)
      .send({ userId: b.id, rawText: 'cough', symptoms: ['cough'], labValues: [{ name: 'Hb', value: 12 }] });

    expect(res.status).toBe(201);
    expect(db.consultations[0].userId).toBe(a.id);
    expect(db.consultations[0].labValues[0].value).toBe('12');
  });

  it('rejects malformed JSON bodies cleanly', async () => {
    const user = createTestUser();
    const res = await request(app)
      .post('/api/consultations')
      .set('Authorization', user.auth)
      .set('Content-Type', 'application/json')
      .send('{"rawText": ');
    expect(res.status).toBe(400);
    expect(res.body.error).toBe('Malformed request body');
  });
});

describe('login rate limiting', () => {
  const seedUser = async () =>
    FakeUser.create({ name: 'Doc', email: 'doc@example.com', password: await bcrypt.hash('right-password', 4), authProvider: 'local' });

  it('throttles repeated login attempts against one account', async () => {
    await seedUser();
    for (let i = 0; i < 5; i++) {
      const res = await request(app).post('/api/auth/login').send({ email: 'doc@example.com', password: 'wrong' });
      expect(res.status).toBe(400);
    }
    const blocked = await request(app).post('/api/auth/login').send({ email: 'DOC@example.com', password: 'right-password' });
    expect(blocked.status).toBe(429);
    expect(blocked.body.message).toMatch(/Too many login attempts/);
    expect(blocked.headers['retry-after']).toBeDefined();

    const otherAccount = await request(app).post('/api/auth/login').send({ email: 'other@example.com', password: 'x' });
    expect(otherAccount.status).toBe(400);
  });

  it('throttles auth attempts per IP across accounts', async () => {
    for (let i = 0; i < 20; i++) {
      await request(app).post('/api/auth/login').send({ email: `u${i}@example.com`, password: 'x' });
    }
    const blocked = await request(app).post('/api/auth/register').send({ name: 'A', email: 'new@example.com', password: 'longenough1' });
    expect(blocked.status).toBe(429);
  });

  it('allows a legitimate login within limits', async () => {
    await seedUser();
    const res = await request(app).post('/api/auth/login').send({ email: 'doc@example.com', password: 'right-password' });
    expect(res.status).toBe(200);
    expect(res.body.token).toBeTypeOf('string');
  });
});
