/**
 * In-process AI call metrics (metadata only). Bounded ring buffer; per-process,
 * reset on restart. Used by the evaluation harness and for debugging — not a
 * substitute for a real metrics backend.
 */
export interface AICallSample {
  operation: string;
  provider: string;
  model: string;
  status: 'ok' | 'error';
  latencyMs: number;
  fallbackUsed: boolean;
  failureCategory?: string;
  inputTokens?: number;
  outputTokens?: number;
}

const MAX_SAMPLES = 2000;
const samples: AICallSample[] = [];

export const recordAICall = (sample: AICallSample): void => {
  samples.push(sample);
  if (samples.length > MAX_SAMPLES) samples.shift();
};

export const percentile = (values: number[], p: number): number | undefined => {
  if (values.length === 0) return undefined;
  const sorted = [...values].sort((a, b) => a - b);
  const index = Math.min(sorted.length - 1, Math.max(0, Math.ceil((p / 100) * sorted.length) - 1));
  return sorted[index];
};

export interface AICallStats {
  operation: string;
  model: string;
  calls: number;
  errors: number;
  fallbackCalls: number;
  p50LatencyMs?: number;
  p95LatencyMs?: number;
  inputTokens: number;
  outputTokens: number;
  failureCategories: Record<string, number>;
}

export const getAICallStats = (): AICallStats[] => {
  const groups = new Map<string, AICallSample[]>();
  for (const s of samples) {
    const key = `${s.operation}|${s.provider}:${s.model}`;
    groups.set(key, [...(groups.get(key) ?? []), s]);
  }
  return [...groups].map(([key, group]) => {
    const [operation, model] = key.split('|');
    const failureCategories: Record<string, number> = {};
    for (const s of group) if (s.failureCategory) failureCategories[s.failureCategory] = (failureCategories[s.failureCategory] ?? 0) + 1;
    const latencies = group.map((s) => s.latencyMs);
    return {
      operation,
      model,
      calls: group.length,
      errors: group.filter((s) => s.status === 'error').length,
      fallbackCalls: group.filter((s) => s.fallbackUsed).length,
      p50LatencyMs: percentile(latencies, 50),
      p95LatencyMs: percentile(latencies, 95),
      inputTokens: group.reduce((sum, s) => sum + (s.inputTokens ?? 0), 0),
      outputTokens: group.reduce((sum, s) => sum + (s.outputTokens ?? 0), 0),
      failureCategories,
    };
  });
};

export const resetAICallStats = (): void => {
  samples.length = 0;
};
