import { randomUUID } from 'node:crypto';
import { Request, Response, NextFunction } from 'express';
import { runWithRequestContext } from '../utils/requestContext';

const VALID_ID = /^[\w-]{1,64}$/;

/**
 * Assigns a correlation id (reusing a safe inbound X-Request-Id), echoes it back,
 * and binds it to the async context so every log line in this request carries it.
 */
export const requestId = (req: Request, res: Response, next: NextFunction): void => {
  const inbound = req.header('x-request-id');
  req.requestId = inbound && VALID_ID.test(inbound) ? inbound : randomUUID();
  res.setHeader('X-Request-Id', req.requestId);
  runWithRequestContext({ requestId: req.requestId }, next);
};
