import { getRequestId } from './requestContext';

/**
 * Minimal structured logger.
 *
 * PHI RULE: never pass clinical content (patient names, symptoms, medicines,
 * report text, prompts, model output, request bodies) into these functions.
 * Log only operational metadata: ids, operation names, model ids, latency,
 * status codes, sizes, token counts and error categories.
 */
export type LogMeta = Record<string, string | number | boolean | null | undefined>;

const write = (level: 'info' | 'warn' | 'error', event: string, meta: LogMeta = {}): void => {
  if (process.env.NODE_ENV === 'test' && level !== 'error') return;
  const requestId = getRequestId();
  const line = JSON.stringify({ level, event, time: new Date().toISOString(), ...(requestId ? { requestId } : {}), ...meta });
  if (level === 'error') console.error(line);
  else if (level === 'warn') console.warn(line);
  else console.log(line);
};

export const logger = {
  info: (event: string, meta?: LogMeta) => write('info', event, meta),
  warn: (event: string, meta?: LogMeta) => write('warn', event, meta),
  error: (event: string, meta?: LogMeta) => write('error', event, meta),
};

/** Error metadata that is safe to log (no messages, which may embed user data). */
export const errorMeta = (err: unknown): LogMeta => {
  if (err instanceof Error) {
    const withStatus = err as Error & { statusCode?: number; status?: number; code?: string | number; category?: string };
    return {
      errorName: err.name,
      errorStatus: withStatus.statusCode ?? withStatus.status,
      errorCode: typeof withStatus.code === 'string' || typeof withStatus.code === 'number' ? withStatus.code : undefined,
      errorCategory: typeof withStatus.category === 'string' ? withStatus.category : undefined,
    };
  }
  return { errorName: typeof err };
};
