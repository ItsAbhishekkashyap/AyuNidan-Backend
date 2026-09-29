import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import request from 'supertest';
import jwt from 'jsonwebtoken';
import fs from 'node:fs';
import path from 'node:path';

vi.mock('../src/models/Consultation', async () => ({ Consultation: (await import('./helpers/fakeModels')).FakeConsultation }));
vi.mock('../src/models/User', async () => ({ User: (await import('./helpers/fakeModels')).FakeUser }));

import { createApp } from '../src/app';
import { loadConfig, getConfig, resetConfigCache, ConfigError } from '../src/config/env';
import { resetRateLimits } from '../src/middleware/rateLimiter';
import { resetDb } from './helpers/fakeModels';
import { createTestUser } from './helpers/auth';

const app = createApp();
const BASE_ENV = { MONGODB_URI: 'mongodb://localhost/test', JWT_SECRET: 'x'.repeat(40) };

beforeEach(() => {
  resetDb();
  resetRateLimits();
});

describe('configuration validation', () => {
  it('fails fast when JWT_SECRET is missing', () => {
    expect(() => loadConfig({ MONGODB_URI: 'mongodb://localhost/test' })).toThrow(ConfigError);
    expect(() => loadConfig({ MONGODB_URI: 'mongodb://localhost/test' })).toThrow(/JWT_SECRET/);
  });

  it('rejects weak JWT secrets without echoing the value', () => {
    try {
      loadConfig({ ...BASE_ENV, JWT_SECRET: 'short-secret-value' });
      expect.unreachable();
    } catch (error) {
      expect(error).toBeInstanceOf(ConfigError);
      expect((error as Error).message).toMatch(/at least 32 characters/);
      expect((error as Error).message).not.toContain('short-secret-value');
    }
  });

  it('fails fast when MONGODB_URI is missing', () => {
    expect(() => loadConfig({ JWT_SECRET: 'x'.repeat(40) })).toThrow(/MONGODB_URI/);
  });

  it('accepts a valid configuration with defaults', () => {
    const config = loadConfig(BASE_ENV);
    expect(config.JWT_EXPIRES_IN).toBe('7d');
    expect(config.PORT).toBe(8080);
  });

  describe('getConfig without JWT_SECRET', () => {
    const original = process.env.JWT_SECRET;
    afterEach(() => {
      process.env.JWT_SECRET = original;
      resetConfigCache();
    });

    it('throws instead of falling back to a default secret', () => {
      delete process.env.JWT_SECRET;
      resetConfigCache();
      expect(() => getConfig()).toThrow(ConfigError);
    });
  });
});

describe('JWT handling', () => {
  it('has no hardcoded fallback secret in the source', () => {
    const files = ['src/middleware/auth.ts', 'src/controllers/auth.controller.ts', 'src/config/env.ts'];
    for (const file of files) {
      const source = fs.readFileSync(path.join(__dirname, '..', file), 'utf8');
      expect(source).not.toMatch(/fallback_secret|JWT_SECRET\s*\|\|/);
    }
  });

  it('rejects tokens signed with the old fallback secret', async () => {
    const user = createTestUser();
    const forged = jwt.sign({ id: user.id, email: user.email }, 'fallback_secret_key_123');
    const res = await request(app).get('/api/consultations').set('Authorization', `Bearer ${forged}`);
    expect(res.status).toBe(401);
  });

  it('rejects unsigned (alg=none) tokens', async () => {
    const user = createTestUser();
    const unsigned = jwt.sign({ id: user.id, email: user.email }, '', { algorithm: 'none' });
    const res = await request(app).get('/api/consultations').set('Authorization', `Bearer ${unsigned}`);
    expect(res.status).toBe(401);
  });

  it('issues tokens that the auth guard accepts', async () => {
    const reg = await request(app)
      .post('/api/auth/register')
      .send({ name: 'Dr Test', email: 'Doc@Example.com', password: 'correct-horse-1' });
    expect(reg.status).toBe(201);
    expect(reg.body.user.email).toBe('doc@example.com');

    const res = await request(app).get('/api/consultations').set('Authorization', `Bearer ${reg.body.token}`);
    expect(res.status).toBe(200);
  });
});
