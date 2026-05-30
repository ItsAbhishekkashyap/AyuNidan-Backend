import { Request, Response, NextFunction } from 'express';
import { ApiResponse } from '../types';

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


export const errorHandler = (
  err: AppError | Error,
  req: Request,
  res: Response,
  next: NextFunction
): void => {
  if (err instanceof AppError) {
    res.status(err.statusCode).json(<ApiResponse<null>>{
      success: false,
      error: err.message,
    });
    return;
  }

  console.error('Unexpected Error:', err);
  res.status(500).json(<ApiResponse<null>>{
    success: false,
    error: 'Internal server error',
  });
};