import { Request, Response, NextFunction } from 'express';


interface CacheEntry {
  data: any;
  expiresAt: number;
}

class MemoryCache {
  private cache = new Map<string, CacheEntry>();
  private maxSize = 100; // Max 100 entries

  set(key: string, data: any, ttlSeconds: number): void {
    // Evict if full (LRU-like)
    if (this.cache.size >= this.maxSize) {
      const firstKey = this.cache.keys().next().value;
      if (firstKey) this.cache.delete(firstKey);
    }
    this.cache.set(key, {
      data,
      expiresAt: Date.now() + ttlSeconds * 1000,
    });
  }

  get(key: string): any | null {
    const entry = this.cache.get(key);
    if (!entry) return null;
    if (Date.now() > entry.expiresAt) {
      this.cache.delete(key);
      return null;
    }
    return entry.data;
  }

  invalidate(pattern: string): void {
    for (const key of this.cache.keys()) {
      if (key.includes(pattern)) this.cache.delete(key);
    }
  }
}

export const cache = new MemoryCache();

export const cacheMiddleware = (ttlSeconds: number) => {
  return (req: Request, res: Response, next: NextFunction): void => {
    const key = `${req.method}:${req.originalUrl}:${JSON.stringify(req.query)}`;
    const cached = cache.get(key);

    if (cached) {
      // CN: Cache hit — no DB query needed
      res.setHeader('X-Cache', 'HIT');
      res.json(cached);
      return;
    }

    res.setHeader('X-Cache', 'MISS');
    const originalJson = res.json.bind(res);
    res.json = (data: any) => {
      cache.set(key, data, ttlSeconds);
      return originalJson(data);
    };
    next();
  };
};