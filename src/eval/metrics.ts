/** Pure evaluation metrics. No I/O, fully deterministic. */

export const normalize = (value: string): string =>
  value
    .toLowerCase()
    .replace(/[^a-z0-9.%<>/ ]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();

/** Lenient string match: equal after normalisation, or one contains the other (min 3 chars). */
export const fuzzyMatch = (predicted: string, expected: string): boolean => {
  const p = normalize(predicted);
  const e = normalize(expected);
  if (!p || !e) return p === e;
  if (p === e) return true;
  const [short, long] = p.length <= e.length ? [p, e] : [e, p];
  return short.length >= 3 && long.includes(short);
};

export interface PRF {
  tp: number;
  fp: number;
  fn: number;
  precision: number;
  recall: number;
  f1: number;
}

/**
 * Convention for empty sets: with no predictions nothing was invented (precision 1);
 * with no gold items nothing could be missed (recall 1). Inventions are penalised by
 * precision and omissions by recall, so empty-vs-empty scores a perfect 1.
 */
export const prfFromCounts = (tp: number, fp: number, fn: number): PRF => {
  const precision = tp + fp === 0 ? 1 : tp / (tp + fp);
  const recall = tp + fn === 0 ? 1 : tp / (tp + fn);
  const f1 = precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall);
  return { tp, fp, fn, precision, recall, f1 };
};

/** Greedy one-to-one set matching with `match`, returning P/R/F1 and matched pairs. */
export const setPRF = <P, G>(
  predicted: P[],
  expected: G[],
  match: (p: P, g: G) => boolean
): PRF & { pairs: [P, G][] } => {
  const unmatched = [...expected];
  const pairs: [P, G][] = [];
  let fp = 0;
  for (const p of predicted) {
    const index = unmatched.findIndex((g) => match(p, g));
    if (index === -1) fp++;
    else pairs.push([p, unmatched.splice(index, 1)[0]]);
  }
  return { ...prfFromCounts(pairs.length, fp, unmatched.length), pairs };
};

/** Micro-average: sum counts across cases, then compute P/R/F1. */
export const microPRF = (items: Pick<PRF, 'tp' | 'fp' | 'fn'>[]): PRF =>
  prfFromCounts(
    items.reduce((s, i) => s + i.tp, 0),
    items.reduce((s, i) => s + i.fp, 0),
    items.reduce((s, i) => s + i.fn, 0)
  );

export const mean = (values: number[]): number | undefined =>
  values.length === 0 ? undefined : values.reduce((s, v) => s + v, 0) / values.length;

/** Fraction of `values` that are true; undefined for an empty set. */
export const rate = (values: boolean[]): number | undefined => mean(values.map((v) => (v ? 1 : 0)));

/** 1 if any relevant item appears in the top-k ranked items, else 0. */
export const hitAtK = (ranked: string[], relevant: Set<string>, k: number): number =>
  ranked.slice(0, k).some((id) => relevant.has(id)) ? 1 : 0;

/** 1 / rank of the first relevant item (1-based), or 0 if none is retrieved. */
export const reciprocalRank = (ranked: string[], relevant: Set<string>): number => {
  const index = ranked.findIndex((id) => relevant.has(id));
  return index === -1 ? 0 : 1 / (index + 1);
};

export const GENDER_ALIASES: Record<string, string> = { m: 'male', male: 'male', f: 'female', female: 'female' };
export const normalizeGender = (value: string): string => GENDER_ALIASES[normalize(value)] ?? normalize(value);

export const round = (value: number | undefined, digits = 3): number | undefined =>
  value === undefined ? undefined : Number(value.toFixed(digits));
