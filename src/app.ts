import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import morgan from 'morgan';
import compression from 'compression';
import { errorHandler } from './middleware/errorHandler';
import { requestId } from './middleware/requestId';
import { parseTrustProxy } from './config/env';
import { runWithRequestContext } from './utils/requestContext';
import routes from './routes/index';

/** Access log line: path WITHOUT query string (search/term params can contain patient data). */
const accessLog = morgan((tokens, req, res) =>
  JSON.stringify({
    level: 'info',
    event: 'http.request',
    time: new Date().toISOString(),
    requestId: (req as express.Request).requestId,
    method: tokens.method(req, res),
    path: ((req as express.Request).originalUrl || req.url || '').split('?')[0],
    status: Number(tokens.status(req, res)),
    latencyMs: Number(tokens['response-time'](req, res)),
  })
);

/**
 * Browser origins allowed to call the API. Override with CORS_ORIGINS (comma-separated).
 * Development also allows 127.0.0.1 and port 3001 (Next.js falls back to 3001 when 3000 is busy).
 */
export const getAllowedOrigins = (): string[] => {
  const configured = process.env.CORS_ORIGINS?.split(',').map((o) => o.trim()).filter(Boolean);
  if (configured?.length) return configured;
  return process.env.NODE_ENV === 'production'
    ? ['https://ayunidan.vercel.app']
    : ['http://localhost:3000', 'http://127.0.0.1:3000', 'http://localhost:3001', 'http://127.0.0.1:3001'];
};

export const createApp = () => {
  const app = express();

  // Required for correct client IPs (rate limiting) behind Render/Vercel-style proxies.
  app.set('trust proxy', parseTrustProxy(process.env.TRUST_PROXY));

  app.use(requestId);
  app.use(compression());
  app.use(helmet({ crossOriginResourcePolicy: { policy: 'cross-origin' } }));
  app.use(
    cors({
      origin: getAllowedOrigins(),
      methods: ['GET', 'POST', 'DELETE'],
      allowedHeaders: ['Content-Type', 'Authorization'],
      maxAge: 86400,
    })
  );

  if (process.env.NODE_ENV !== 'test') app.use(accessLog);
  app.use(express.json({ limit: '1mb' }));
  app.use(express.urlencoded({ extended: true, limit: '1mb' }));
  // Body parsers resume on stream events, which can drop async context; re-bind the request id.
  app.use((req, _res, next) => runWithRequestContext({ requestId: req.requestId as string }, next));

  app.use('/api', routes);

  app.get('/health', (req, res) => {
    res.json({ status: 'Running', uptime: process.uptime(), timestamp: new Date().toISOString() });
  });

  app.use((req, res) => {
    res.status(404).json({ success: false, error: 'Not found' });
  });

  app.use(errorHandler);
  return app;
};
