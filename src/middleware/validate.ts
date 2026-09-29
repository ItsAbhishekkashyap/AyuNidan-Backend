import { Request, Response, NextFunction } from 'express';
import { z } from 'zod';

interface RequestSchemas {
  body?: z.ZodType;
  query?: z.ZodType;
  params?: z.ZodType;
}

/**
 * Validates and normalises request input with Zod. Parsed values replace the
 * originals, so controllers receive trimmed, typed and bounded data.
 * Error details contain field paths and rule messages only — never input values.
 */
export const validate = (schemas: RequestSchemas) => {
  return (req: Request, res: Response, next: NextFunction): void => {
    const issues: { location: string; path: string; message: string }[] = [];
    const parsed: Partial<Record<keyof RequestSchemas, unknown>> = {};

    for (const location of ['params', 'query', 'body'] as const) {
      const schema = schemas[location];
      if (!schema) continue;
      const result = schema.safeParse(req[location] ?? {});
      if (result.success) {
        parsed[location] = result.data;
      } else {
        for (const issue of result.error.issues) {
          issues.push({ location, path: issue.path.join('.'), message: issue.message });
        }
      }
    }

    if (issues.length > 0) {
      const first = issues[0];
      const summary = `Invalid ${first.location}${first.path ? ` field "${first.path}"` : ''}: ${first.message}`;
      // `message` keeps compatibility with auth clients that read data.message.
      res.status(400).json({ success: false, error: summary, message: summary, details: issues });
      return;
    }

    if (parsed.body !== undefined) req.body = parsed.body;
    // Express 5 exposes req.query/req.params via getters; shadow them with the parsed values.
    if (parsed.query !== undefined) Object.defineProperty(req, 'query', { value: parsed.query, writable: true });
    if (parsed.params !== undefined) Object.defineProperty(req, 'params', { value: parsed.params, writable: true });
    next();
  };
};
