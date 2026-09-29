/**
 * Deterministic reference-range checking.
 *
 * Abnormal flags are computed by comparing a value with the reference range PRINTED IN THE
 * REPORT — the model's opinion is never the source of truth, and no external/medical threshold is
 * introduced. If the value or range cannot be parsed unambiguously, no flag is produced.
 */

export type RangeFlag = 'Low' | 'High' | 'Normal';

const NUM = String.raw`-?\d+(?:[.,]\d+)?`;

const toNumber = (s: string): number => Number(s.replace(/,(?=\d{3}\b)/g, '').replace(',', '.'));

/** Parses a plain numeric result such as "35.90 %", "199", "4.5". Returns undefined for "<5", "NEGATIVE", etc. */
export const parseResultNumber = (value: string): number | undefined => {
  const match = value.trim().match(new RegExp(`^(${NUM})(?:\\s*[^\\d<>=≤≥].*)?$`));
  if (!match) return undefined;
  const n = toNumber(match[1]);
  return Number.isFinite(n) ? n : undefined;
};

export interface ParsedRange {
  low?: number;
  high?: number;
}

/**
 * Parses printed ranges: "36 - 46 %", "0 - 199.99 mg/dL", "0.34 – 5.6 μIU/mL", "< 0.04", "≤ 5", "> 40", "≥ 30".
 * A degenerate range ("0 - 0 %", low ≥ high) is treated as unusable.
 */
export const parseRange = (range: string): ParsedRange | undefined => {
  const text = range.trim();
  const between = text.match(new RegExp(`(${NUM})\\s*(?:-|–|—|to)\\s*(${NUM})`, 'i'));
  if (between) {
    const low = toNumber(between[1]);
    const high = toNumber(between[2]);
    return Number.isFinite(low) && Number.isFinite(high) && low < high ? { low, high } : undefined;
  }
  const upper = text.match(new RegExp(`^(?:<|≤|<=|up to|upto)\\s*(${NUM})`, 'i'));
  if (upper) return { high: toNumber(upper[1]) };
  const lower = text.match(new RegExp(`^(?:>|≥|>=)\\s*(${NUM})`, 'i'));
  if (lower) return { low: toNumber(lower[1]) };
  return undefined;
};

/** Compares a numeric result to a printed range. Returns undefined when either cannot be parsed. */
export const flagFromRange = (value: string, range: string | undefined): RangeFlag | undefined => {
  if (!range) return undefined;
  const n = parseResultNumber(value);
  const r = parseRange(range);
  if (n === undefined || !r) return undefined;
  if (r.low !== undefined && n < r.low) return 'Low';
  if (r.high !== undefined && n > r.high) return 'High';
  return 'Normal';
};
