import { Request, Response, NextFunction } from 'express';

/**
 * In-process response cache for authenticated GET endpoints.
 *
 * Security properties:
 * - Every key is scoped to the authenticated user (`user:<id>:<url>`), so one
 *   user's cached response can never be served to another user.
 * - Only successful (2xx) responses are stored; error bodies are never cached.
 * - Mutations call `invalidateUserCache(userId)` to drop that user's entries.
 *
 * Limitation: per-process only. With multiple instances each has its own cache
 * and invalidation does not propagate (a shared store such as Redis would be needed).
 */

interface CachedResponse {
  status: number;
  body: unknown;
}

interface CacheEntry {
  value: CachedResponse;
  expiresAt: number;
}

class MemoryCache {
  private store = new Map<string, CacheEntry>();

  constructor(private readonly maxSize = 500) {}

  set(key: string, value: CachedResponse, ttlSeconds: number): void {
    this.store.delete(key);
    while (this.store.size >= this.maxSize) {
      const oldestKey = this.store.keys().next().value;
      if (oldestKey === undefined) break;
      this.store.delete(oldestKey);
    }
    this.store.set(key, { value, expiresAt: Date.now() + ttlSeconds * 1000 });
  }

  get(key: string): CachedResponse | null {
    const entry = this.store.get(key);
    if (!entry) return null;
    if (Date.now() > entry.expiresAt) {
      this.store.delete(key);
      return null;
    }
    // Refresh recency (LRU)
    this.store.delete(key);
    this.store.set(key, entry);
    return entry.value;
  }

  invalidatePrefix(prefix: string): void {
    for (const key of this.store.keys()) {
      if (key.startsWith(prefix)) this.store.delete(key);
    }
  }

  clear(): void {
    this.store.clear();
  }

  get size(): number {
    return this.store.size;
  }
}

export const cache = new MemoryCache();

const userScope = (userId: string): string => `user:${userId}:`;

export const invalidateUserCache = (userId: string): void => {
  cache.invalidatePrefix(userScope(userId));
};

const isSuccessfulBody = (body: unknown): boolean =>
  !(body && typeof body === 'object' && (body as { success?: unknown }).success === false);

/** Must be mounted AFTER authGuard. Requests without an authenticated user are never cached. */
export const cacheMiddleware = (ttlSeconds: number) => {
  return (req: Request, res: Response, next: NextFunction): void => {
    const userId = req.user?.id;
    if (!userId || req.method !== 'GET') {
      next();
      return;
    }

    res.setHeader('Cache-Control', 'private, no-store');
    const key = `${userScope(userId)}${req.originalUrl}`;
    const cached = cache.get(key);

    if (cached) {
      res.setHeader('X-Cache', 'HIT');
      res.status(cached.status).json(cached.body);
      return;
    }

    res.setHeader('X-Cache', 'MISS');
    const originalJson = res.json.bind(res);
    res.json = (body: unknown) => {
      if (res.statusCode >= 200 && res.statusCode < 300 && isSuccessfulBody(body)) {
        cache.set(key, { status: res.statusCode, body }, ttlSeconds);
      }
      return originalJson(body);
    };
    next();
  };
};
