import { Response } from 'express';

/**
 * Lightweight stage timer. Timings are exposed via the standard Server-Timing
 * header and structured logs (never containing clinical content).
 */
export class StageTimer {
  private readonly startedAt = performance.now();
  readonly stages: Record<string, number> = {};

  async time<T>(stage: string, fn: () => Promise<T>): Promise<T> {
    const start = performance.now();
    try {
      return await fn();
    } finally {
      this.stages[stage] = Math.round((this.stages[stage] ?? 0) + performance.now() - start);
    }
  }

  totalMs(): number {
    return Math.round(performance.now() - this.startedAt);
  }

  /** Flattened log metadata, e.g. { t_extraction_ms: 120, t_total_ms: 300 }. */
  toLogMeta(): Record<string, number> {
    const meta: Record<string, number> = {};
    for (const [stage, ms] of Object.entries(this.stages)) meta[`t_${stage}_ms`] = ms;
    meta.t_total_ms = this.totalMs();
    return meta;
  }

  applyHeader(res: Response): void {
    if (res.headersSent) return;
    const parts = Object.entries(this.stages).map(([stage, ms]) => `${stage};dur=${ms}`);
    parts.push(`total;dur=${this.totalMs()}`);
    res.setHeader('Server-Timing', parts.join(', '));
  }
}
