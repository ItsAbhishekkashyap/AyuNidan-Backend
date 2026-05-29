import { Request, Response, NextFunction } from 'express';


const requestCounts = new Map<string, { count: number; resetTime: number }>();

export const rateLimiter = (maxRequests: number, windowMs: number) => {
  return (req: Request, res: Response, next: NextFunction): void => {
    const ip = req.ip || req.socket.remoteAddress || 'unknown';
    const now = Date.now();
    const windowData = requestCounts.get(ip);

    if (!windowData || now > windowData.resetTime) {
      requestCounts.set(ip, { count: 1, resetTime: now + windowMs });
      next();
      return;
    }

    if (windowData.count >= maxRequests) {
     
      res.status(429).json({
        success: false,
        error: 'Too many requests. Please try again later.',
        retryAfter: Math.ceil((windowData.resetTime - now) / 1000),
      });
      return;
    }

    windowData.count++;
    next();
  };
};


export const aiRateLimiter = rateLimiter(10, 60 * 1000);   

export const generalRateLimiter = rateLimiter(100, 60 * 1000); 


