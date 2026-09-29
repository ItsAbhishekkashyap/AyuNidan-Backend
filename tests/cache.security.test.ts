import { describe, it, expect, beforeEach, vi } from 'vitest';
import request from 'supertest';
import express from 'express';

vi.mock('../src/models/Consultation', async () => ({ Consultation: (await import('./helpers/fakeModels')).FakeConsultation }));
vi.mock('../src/models/User', async () => ({ User: (await import('./helpers/fakeModels')).FakeUser }));
vi.mock('../src/services/ai.service', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../src/services/ai.service')>()),
  extractMedicalData: vi.fn(),
}));
vi.mock('../src/services/assessment.service', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../src/services/assessment.service')>()),
  assessRisk: vi.fn(async () => (await import('./helpers/aiOutputs')).fakeClinicalAssessment('medium', 50)),
}));

import { createApp } from '../src/app';
import { cache, cacheMiddleware } from '../src/middleware/cache';
import { resetRateLimits } from '../src/middleware/rateLimiter';
import { db, resetDb, FakeConsultation } from './helpers/fakeModels';
import { createTestUser } from './helpers/auth';

const app = createApp();

const seedConsultation = (userId: string, name: string) =>
  FakeConsultation.create({ userId, patientDetails: { name }, summary: `${name} summary`, riskLevel: 'high', riskScore: 90 });

const miniApp = () => {
  const mini = express();
  mini.use((req, _res, next) => {
    req.user = { id: 'user-1', email: 'u@test.dev' };
    next();
  });
  return mini;
};

beforeEach(() => {
  resetDb();
  cache.clear();
  resetRateLimits();
});

describe('cross-user cache isolation', () => {
  it("does not serve user A's cached consultation list to user B", async () => {
    const a = createTestUser();
    const b = createTestUser();
    await seedConsultation(a.id, 'Alice Patient');

    const first = await request(app).get('/api/consultations').set('Authorization', a.auth);
    expect(first.status).toBe(200);
    expect(first.body.data).toHaveLength(1);

    const second = await request(app).get('/api/consultations').set('Authorization', a.auth);
    expect(second.headers['x-cache']).toBe('HIT');

    const other = await request(app).get('/api/consultations').set('Authorization', b.auth);
    expect(other.status).toBe(200);
    expect(other.headers['x-cache']).toBe('MISS');
    expect(other.body.data).toEqual([]);
    expect(JSON.stringify(other.body)).not.toContain('Alice Patient');
  });

  it("does not serve user A's cached dashboard to user B", async () => {
    const a = createTestUser();
    const b = createTestUser();
    await seedConsultation(a.id, 'Alice Patient');

    await request(app).get('/api/consultations/dashboard').set('Authorization', a.auth);
    const other = await request(app).get('/api/consultations/dashboard').set('Authorization', b.auth);

    expect(other.headers['x-cache']).toBe('MISS');
    expect(other.body.data.total).toBe(0);
  });

  it('never bypasses ownership for a cached consultation by id', async () => {
    const a = createTestUser();
    const b = createTestUser();
    const record = await seedConsultation(a.id, 'Alice Patient');

    const own = await request(app).get(`/api/consultations/${record._id}`).set('Authorization', a.auth);
    expect(own.status).toBe(200);

    const stolen = await request(app).get(`/api/consultations/${record._id}`).set('Authorization', b.auth);
    expect(stolen.status).toBe(404);
    expect(JSON.stringify(stolen.body)).not.toContain('Alice Patient');
  });

  it('requires authentication before any cache lookup', async () => {
    const a = createTestUser();
    await seedConsultation(a.id, 'Alice Patient');
    await request(app).get('/api/consultations').set('Authorization', a.auth);

    const anon = await request(app).get('/api/consultations');
    expect(anon.status).toBe(401);
    expect(JSON.stringify(anon.body)).not.toContain('Alice Patient');
  });
});

describe('error responses are never cached', () => {
  it('does not cache 404 responses', async () => {
    const a = createTestUser();
    const missingId = '64b7f0f0f0f0f0f0f0f0f0f0';

    const first = await request(app).get(`/api/consultations/${missingId}`).set('Authorization', a.auth);
    expect(first.status).toBe(404);
    expect(cache.size).toBe(0);

    const second = await request(app).get(`/api/consultations/${missingId}`).set('Authorization', a.auth);
    expect(second.headers['x-cache']).toBe('MISS');
  });

  it('does not cache 4xx/5xx or success:false bodies', async () => {
    let calls = 0;
    const mini = miniApp();
    mini.get('/fail', cacheMiddleware(60), (_req, res) => {
      calls += 1;
      res.status(500).json({ success: false, error: 'boom' });
    });
    mini.get('/bad', cacheMiddleware(60), (_req, res) => {
      calls += 1;
      res.status(400).json({ success: false, error: 'bad' });
    });
    mini.get('/soft-fail', cacheMiddleware(60), (_req, res) => {
      calls += 1;
      res.status(200).json({ success: false, error: 'soft' });
    });

    for (const path of ['/fail', '/bad', '/soft-fail']) {
      await request(mini).get(path);
      const again = await request(mini).get(path);
      expect(again.headers['x-cache']).toBe('MISS');
    }
    expect(calls).toBe(6);
    expect(cache.size).toBe(0);
  });

  it('caches successful responses', async () => {
    const mini = miniApp();
    mini.get('/ok', cacheMiddleware(60), (_req, res) => {
      res.status(200).json({ success: true, data: 1 });
    });
    await request(mini).get('/ok');
    const again = await request(mini).get('/ok');
    expect(again.headers['x-cache']).toBe('HIT');
    expect(again.body).toEqual({ success: true, data: 1 });
  });
});

describe('cache invalidation on mutation', () => {
  it('invalidates the list and dashboard after creating a consultation', async () => {
    const a = createTestUser();
    await seedConsultation(a.id, 'First Patient');

    await request(app).get('/api/consultations').set('Authorization', a.auth);
    await request(app).get('/api/consultations/dashboard').set('Authorization', a.auth);

    const created = await request(app)
      .post('/api/consultations')
      .set('Authorization', a.auth)
      .send({ rawText: 'Mild cough', symptoms: ['cough'], patientDetails: { name: 'Second Patient' } });
    expect(created.status).toBe(201);

    const list = await request(app).get('/api/consultations').set('Authorization', a.auth);
    expect(list.headers['x-cache']).toBe('MISS');
    expect(list.body.data).toHaveLength(2);

    const dash = await request(app).get('/api/consultations/dashboard').set('Authorization', a.auth);
    expect(dash.headers['x-cache']).toBe('MISS');
    expect(dash.body.data.total).toBe(2);
  });

  it('invalidates cached records after deletion', async () => {
    const a = createTestUser();
    const record = await seedConsultation(a.id, 'Alice Patient');

    await request(app).get(`/api/consultations/${record._id}`).set('Authorization', a.auth);
    await request(app).get('/api/consultations').set('Authorization', a.auth);

    const del = await request(app).delete(`/api/consultations/${record._id}`).set('Authorization', a.auth);
    expect(del.status).toBe(200);

    const byId = await request(app).get(`/api/consultations/${record._id}`).set('Authorization', a.auth);
    expect(byId.status).toBe(404);
    const list = await request(app).get('/api/consultations').set('Authorization', a.auth);
    expect(list.body.data).toEqual([]);
  });

  it("only invalidates the mutating user's entries", async () => {
    const a = createTestUser();
    const b = createTestUser();
    await request(app).get('/api/consultations').set('Authorization', b.auth);

    await request(app)
      .post('/api/consultations')
      .set('Authorization', a.auth)
      .send({ rawText: 'Headache', symptoms: ['headache'] });

    const bList = await request(app).get('/api/consultations').set('Authorization', b.auth);
    expect(bList.headers['x-cache']).toBe('HIT');
  });

  it("prevents user B from deleting user A's consultation", async () => {
    const a = createTestUser();
    const b = createTestUser();
    const record = await seedConsultation(a.id, 'Alice Patient');

    const del = await request(app).delete(`/api/consultations/${record._id}`).set('Authorization', b.auth);
    expect(del.status).toBe(404);
    expect(db.consultations).toHaveLength(1);
  });
});
