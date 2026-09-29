import 'express';

declare global {
  namespace Express {
    interface Request {
      /** Set by authGuard. */
      user?: {
        id: string;
        email: string;
      };
      /** Set by requestId middleware. */
      requestId?: string;
    }
  }
}

export {};
