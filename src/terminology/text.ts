/** Text helpers for the terminology datasets (XML/HTML decoding and deterministic lookup keys). */

const NAMED_ENTITIES: Record<string, string> = {
  amp: '&',
  lt: '<',
  gt: '>',
  quot: '"',
  apos: "'",
  nbsp: ' ',
  ndash: '–',
  mdash: '—',
  rsquo: '’',
  lsquo: '‘',
  rdquo: '”',
  ldquo: '“',
  deg: '°',
  micro: 'µ',
  plusmn: '±',
  times: '×',
  hellip: '…',
};

/** Decodes numeric and the common named entities. Unknown named entities are left as written. */
export const decodeEntities = (text: string): string =>
  text.replace(/&(#x[0-9a-fA-F]+|#\d+|[a-zA-Z]+);/g, (whole, body: string) => {
    if (body[0] === '#') {
      const code = body[1] === 'x' || body[1] === 'X' ? Number.parseInt(body.slice(2), 16) : Number.parseInt(body.slice(1), 10);
      return Number.isFinite(code) && code > 0 && code <= 0x10ffff ? String.fromCodePoint(code) : whole;
    }
    return NAMED_ENTITIES[body] ?? whole;
  });

/**
 * MedlinePlus summaries are HTML stored as escaped XML text. Decode once to get the HTML, turn block
 * structure into line breaks / bullets, drop every tag (links keep their visible text), decode again.
 */
export const htmlToText = (escapedHtml: string): string => {
  const html = decodeEntities(escapedHtml);
  const text = html
    .replace(/<\s*(script|style)[^>]*>[\s\S]*?<\s*\/\s*\1\s*>/gi, ' ')
    .replace(/<\s*li[^>]*>/gi, '\n- ')
    .replace(/<\s*\/\s*(p|div|ul|ol|h[1-6]|li|tr)\s*>/gi, '\n')
    .replace(/<\s*br\s*\/?>/gi, '\n')
    .replace(/<[^>]*>/g, '');
  return decodeEntities(text)
    .split('\n')
    .map((line) => line.replace(/[ \t ]+/g, ' ').trim())
    .filter((line, i, all) => line !== '' || (i > 0 && all[i - 1] !== ''))
    .join('\n')
    .trim();
};

/**
 * Deterministic lookup key: case/diacritic-insensitive, punctuation collapsed to single spaces.
 * "Hemoglobin A1c", "hemoglobin  a1c." and "HEMOGLOBIN-A1C" share one key.
 */
export const normalizeTermKey = (text: string): string =>
  text
    .normalize('NFKD')
    .replace(/[̀-ͯ]/g, '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, ' ')
    .trim();

/** Cuts at a sentence boundary at or before `max` characters (never mid-word). */
export const truncateAtSentence = (text: string, max: number): string => {
  if (text.length <= max) return text;
  const slice = text.slice(0, max);
  const lastStop = Math.max(slice.lastIndexOf('. '), slice.lastIndexOf('.\n'), slice.lastIndexOf('\n'));
  return (lastStop > max * 0.5 ? slice.slice(0, lastStop + 1) : slice.replace(/\s+\S*$/, '')).trim();
};

/** Normalised ISO date from MM/DD/YYYY (MedlinePlus) — undefined when not unambiguous. */
export const usDateToIso = (value?: string): string | undefined => {
  const m = value?.match(/^(\d{1,2})\/(\d{1,2})\/(\d{4})/);
  if (!m) return undefined;
  const [, mm, dd, yyyy] = m;
  return `${yyyy}-${mm.padStart(2, '0')}-${dd.padStart(2, '0')}`;
};
