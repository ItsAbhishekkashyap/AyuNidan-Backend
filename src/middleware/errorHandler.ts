import { Request, Response, NextFunction } from 'express';
import { ApiResponse } from '../types';
import { logger, errorMeta } from '../utils/logger';

export class AppError extends Error {
  statusCode: number;
  isOperational: boolean;

  constructor(message: string, statusCode: number) {
    super(message);
    this.statusCode = statusCode;
    this.isOperational = true;
    Error.captureStackTrace(this, this.constructor);
  }
}

interface BodyParserError extends Error {
  type?: string;
  status?: number;
}

export const errorHandler = (err: AppError | BodyParserError, req: Request, res: Response, _next: NextFunction): void => {
  if (err instanceof AppError) {
    res.status(err.statusCode).json(<ApiResponse<null>>{ success: false, error: err.message });
    return;
  }

  // express.json / urlencoded failures (malformed or oversized bodies)
  const parserError = err as BodyParserError;
  if (parserError.type === 'entity.parse.failed') {
    res.status(400).json(<ApiResponse<null>>{ success: false, error: 'Malformed request body' });
    return;
  }
  if (parserError.type === 'entity.too.large') {
    res.status(413).json(<ApiResponse<null>>{ success: false, error: 'Request body too large' });
    return;
  }

  logger.error('http.unhandled_error', { requestId: req.requestId, path: req.path, ...errorMeta(err) });
  res.status(500).json(<ApiResponse<null>>{ success: false, error: 'Internal server error' });
};
