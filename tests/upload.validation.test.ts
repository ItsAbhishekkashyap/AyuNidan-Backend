import { describe, it, expect, beforeEach, vi } from 'vitest';
import request from 'supertest';

vi.mock('../src/models/Consultation', async () => ({ Consultation: (await import('./helpers/fakeModels')).FakeConsultation }));
vi.mock('../src/models/User', async () => ({ User: (await import('./helpers/fakeModels')).FakeUser }));
vi.mock('../src/services/ai.service', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../src/services/ai.service')>()),
  extractMedicalData: vi.fn(async () => ({ patientDetails: { name: '' }, symptoms: [], medicines: [], labValues: [], rawText: 'Synthetic transcript' })),
}));

import { createApp } from '../src/app';
import { resetRateLimits } from '../src/middleware/rateLimiter';
import { extractMedicalData } from '../src/services/ai.service';
import { resetDb } from './helpers/fakeModels';
import { createTestUser } from './helpers/auth';
import { jpegBytes, pdfBytes, pngBytes, wavBytes, webmBytes } from './helpers/fixtures';

const app = createApp();
let auth: string;

beforeEach(() => {
  resetDb();
  resetRateLimits();
  vi.mocked(extractMedicalData).mockClear();
  auth = createTestUser().auth;
});

const upload = () => request(app).post('/api/uploads').set('Authorization', auth);

describe('report upload hardening', () => {
  it('accepts supported medical documents', async () => {
    const res = await upload()
      .attach('files', pdfBytes(), { filename: 'report.pdf', contentType: 'application/pdf' })
      .attach('files', pngBytes(), { filename: 'scan.png', contentType: 'image/png' })
      .attach('files', jpegBytes(), { filename: 'scan.JPG', contentType: 'image/jpeg' })
      .attach('files', Buffer.from('BP 120/80'), { filename: 'notes.txt', contentType: 'text/plain' });
    expect(res.status).toBe(200);
    expect(vi.mocked(extractMedicalData).mock.calls[0][1]).toHaveLength(4);
  });

  it('rejects disallowed MIME types', async () => {
    const res = await upload().attach('files', Buffer.from('PK\u0003\u0004'), { filename: 'a.zip', contentType: 'application/zip' });
    expect(res.status).toBe(415);
    expect(extractMedicalData).not.toHaveBeenCalled();
  });

  it('rejects extensions that do not match the declared type', async () => {
    const res = await upload().attach('files', pdfBytes(), { filename: 'report.exe', contentType: 'application/pdf' });
    expect(res.status).toBe(415);
  });

  it('rejects content that does not match the declared MIME type (spoofing)', async () => {
    const res = await upload().attach('files', Buffer.from('MZ\x90\x00 not a pdf'), { filename: 'report.pdf', contentType: 'application/pdf' });
    expect(res.status).toBe(415);
    expect(res.body.error).toMatch(/content does not match/);
    expect(extractMedicalData).not.toHaveBeenCalled();
  });

  it('rejects binary content disguised as text', async () => {
    const res = await upload().attach('files', Buffer.from([0x41, 0x00, 0x42]), { filename: 'notes.txt', contentType: 'text/plain' });
    expect(res.status).toBe(415);
  });

  it('rejects empty files', async () => {
    const res = await upload().attach('files', Buffer.alloc(0), { filename: 'empty.pdf', contentType: 'application/pdf' });
    expect(res.status).toBe(400);
    expect(extractMedicalData).not.toHaveBeenCalled();
  });

  it('rejects too many files', async () => {
    let req = upload();
    for (let i = 0; i < 6; i++) req = req.attach('files', pdfBytes(), { filename: `r${i}.pdf`, contentType: 'application/pdf' });
    const res = await req;
    expect(res.status).toBe(400);
    expect(res.body.error).toBe('Too many files');
  });

  it('rejects oversized files', async () => {
    const big = Buffer.concat([pdfBytes(), Buffer.alloc(16 * 1024 * 1024)]);
    const res = await upload().attach('files', big, { filename: 'big.pdf', contentType: 'application/pdf' });
    expect(res.status).toBe(413);
  });

  it('rejects unexpected file fields', async () => {
    const res = await upload().attach('document', pdfBytes(), { filename: 'r.pdf', contentType: 'application/pdf' });
    expect(res.status).toBe(400);
  });

  it('rejects non-multipart and malformed multipart requests', async () => {
    const json = await upload().send({ text: 'fever' });
    expect(json.status).toBe(400);

    const malformed = await upload()
      .set('Content-Type', 'multipart/form-data; boundary=xyz')
      .send('--xyz\r\nContent-Disposition: form-data; name="files"; filename="a.pdf"\r\n\r\n%PDF-');
    expect(malformed.status).toBe(400);
  });

  it('neutralises path traversal in filenames', async () => {
    const res = await upload().attach('files', pdfBytes(), { filename: '../../etc/passwd.pdf', contentType: 'application/pdf' });
    expect(res.status).toBe(200);
    const file = vi.mocked(extractMedicalData).mock.calls[0][1]![0];
    expect(file.originalname).toBe('passwd.pdf');
  });

  it('bounds free-text fields', async () => {
    const res = await upload().field('text', 'a'.repeat(50_001));
    expect(res.status).toBe(400);
  });
});

describe('voice upload hardening', () => {
  const voice = () => request(app).post('/api/voice/transcribe').set('Authorization', auth);

  it('accepts browser-recorded audio with codec parameters', async () => {
    const res = await voice().attach('audio', webmBytes(), { filename: 'voicenote.wav', contentType: 'audio/webm;codecs=opus' });
    expect(res.status).toBe(200);
    expect(vi.mocked(extractMedicalData).mock.calls[0][1]![0].mimetype).toBe('audio/webm');
  });

  it('accepts wav audio', async () => {
    const res = await voice().attach('audio', wavBytes(), { filename: 'note.wav', contentType: 'audio/wav' });
    expect(res.status).toBe(200);
  });

  it('rejects non-audio content declared as audio', async () => {
    const res = await voice().attach('audio', pdfBytes(), { filename: 'note.wav', contentType: 'audio/wav' });
    expect(res.status).toBe(415);
  });

  it('rejects requests without an audio file', async () => {
    const res = await voice().field('x', 'y');
    expect(res.status).toBe(400);
  });
});
