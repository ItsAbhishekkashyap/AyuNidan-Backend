import { Request, Response, NextFunction } from 'express';

/**
 * Fixed-window, IN-MEMORY rate limiter.
 *
 * NOT distributed-safe: counters live in this process only, so limits are per
 * instance and reset on restart. A shared store (e.g. Redis) is required for
 * accurate limits across multiple instances.
 *
 * Client IPs are only accurate behind a proxy when `TRUST_PROXY` is configured
 * (see app.ts); otherwise every request appears to come from the proxy.
 */

interface WindowData {
  count: number;
  resetTime: number;
}

interface RateLimiterOptions {
  name: string;
  maxRequests: number;
  windowMs: number;
  /** Defaults to the client IP. */
  keyGenerator?: (req: Request) => string | undefined;
  message?: string;
}

const stores = new Set<Map<string, WindowData>>();

/** Test helper: clears all limiter counters. */
export const resetRateLimits = (): void => {
  for (const store of stores) store.clear();
};

const clientIp = (req: Request): string => req.ip || req.socket.remoteAddress || 'unknown';

export const createRateLimiter = ({
  name,
  maxRequests,
  windowMs,
  keyGenerator = clientIp,
  message = 'Too many requests. Please try again later.',
}: RateLimiterOptions) => {
  const store = new Map<string, WindowData>();
  stores.add(store);
  let lastSweep = Date.now();

  return (req: Request, res: Response, next: NextFunction): void => {
    const now = Date.now();

    // Periodically drop expired windows so the map does not grow without bound.
    if (now - lastSweep > windowMs) {
      for (const [key, data] of store) if (now > data.resetTime) store.delete(key);
      lastSweep = now;
    }

    const rawKey = keyGenerator(req);
    if (rawKey === undefined) {
      next();
      return;
    }
    const key = `${name}:${rawKey}`;
    const windowData = store.get(key);

    if (!windowData || now > windowData.resetTime) {
      store.set(key, { count: 1, resetTime: now + windowMs });
      next();
      return;
    }

    if (windowData.count >= maxRequests) {
      const retryAfter = Math.ceil((windowData.resetTime - now) / 1000);
      res.setHeader('Retry-After', String(retryAfter));
      res.status(429).json({ success: false, error: message, message, retryAfter });
      return;
    }

    windowData.count++;
    next();
  };
};

/** Backwards-compatible helper. */
export const rateLimiter = (maxRequests: number, windowMs: number, name = `limiter-${stores.size}`) =>
  createRateLimiter({ name, maxRequests, windowMs });

const envInt = (name: string, fallback: number): number => {
  const value = Number.parseInt(process.env[name] ?? '', 10);
  return Number.isFinite(value) && value > 0 ? value : fallback;
};

export const aiRateLimiter = createRateLimiter({
  name: 'ai',
  maxRequests: envInt('AI_RATE_LIMIT_MAX', 10),
  windowMs: envInt('AI_RATE_LIMIT_WINDOW_MS', 60_000),
});

export const generalRateLimiter = createRateLimiter({
  name: 'general',
  maxRequests: envInt('GENERAL_RATE_LIMIT_MAX', 100),
  windowMs: envInt('GENERAL_RATE_LIMIT_WINDOW_MS', 60_000),
});

/**
 * Authentication limiters (brute-force mitigation).
 * - authIpRateLimiter: all auth attempts per client IP.
 * - loginAccountRateLimiter: login attempts per target email, so distributed
 *   guessing against one account is also throttled. Mount after body validation.
 */
export const authIpRateLimiter = createRateLimiter({
  name: 'auth-ip',
  maxRequests: envInt('AUTH_RATE_LIMIT_MAX', 20),
  windowMs: envInt('AUTH_RATE_LIMIT_WINDOW_MS', 15 * 60_000),
  message: 'Too many authentication attempts. Please try again later.',
});

export const loginAccountRateLimiter = createRateLimiter({
  name: 'login-account',
  maxRequests: envInt('LOGIN_ACCOUNT_RATE_LIMIT_MAX', 5),
  windowMs: envInt('LOGIN_ACCOUNT_RATE_LIMIT_WINDOW_MS', 15 * 60_000),
  keyGenerator: (req) => (typeof req.body?.email === 'string' ? req.body.email.toLowerCase() : undefined),
  message: 'Too many login attempts for this account. Please try again later.',
});
