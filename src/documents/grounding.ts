/**
 * Grounding checks: is a model-extracted item actually present in the document text?
 * Used to keep model output honest — items that cannot be found in the source text are never
 * merged into the verified structured data.
 */

const STOPWORDS = new Set(['the', 'and', 'for', 'with', 'total', 'test', 'level', 'levels', 'count', 'serum', 'blood', 'of', 'in', 'to', 'at', 'a']);

export const foldText = (s: string): string =>
  s
    .toLowerCase()
    .normalize('NFKD')
    .replace(/[̀-ͯ]/g, '')
    .replace(/[^a-z0-9.%]+/g, ' ')
    .replace(/\.+(?=\s|$)/g, '') // sentence-final dots must not glue to the previous token ("98." → "98")
    .replace(/\s+/g, ' ')
    .trim();

const tokens = (s: string): string[] => foldText(s).split(' ').filter((t) => t.length >= 2 && !STOPWORDS.has(t));

export interface SourcePageText {
  filename: string;
  page?: number;
  text: string;
}

interface FoldedPage {
  /** Space-padded folded text, so an exact whole-token match is a substring test. */
  padded: string;
  words: string[];
  wordSet: Set<string>;
}

const cache = new WeakMap<SourcePageText, FoldedPage>();
const fold = (p: SourcePageText): FoldedPage => {
  let f = cache.get(p);
  if (!f) {
    const folded = foldText(p.text);
    const words = folded.split(' ').filter(Boolean);
    f = { padded: ` ${folded} `, words, wordSet: new Set(words) };
    cache.set(p, f);
  }
  return f;
};

/** Exact whole-token match (used for numeric values). */
const hasExact = (f: FoldedPage, token: string): boolean => f.wordSet.has(token);

/** Word match tolerant of inflection ("cough" ~ "coughing"): a shared prefix of ≥4 characters. */
const hasWord = (f: FoldedPage, token: string): boolean => {
  if (f.wordSet.has(token)) return true;
  if (token.length < 4) return false;
  return f.words.some((w) => w.length >= 4 && (w.startsWith(token) || token.startsWith(w)));
};

/** Pages on which a free-text item (symptom, medicine, diagnosis, name…) is found: all significant words present. */
export const findTextPages = (value: string, pages: SourcePageText[]): SourcePageText[] => {
  const needle = foldText(value);
  if (!needle) return [];
  const words = tokens(value);
  return pages.filter((p) => {
    const f = fold(p);
    if (f.padded.includes(` ${needle} `)) return true;
    return words.length > 0 && words.every((w) => hasWord(f, w));
  });
};

export const findText = (value: string, pages: SourcePageText[]): SourcePageText | undefined => findTextPages(value, pages)[0];

/** Pages on which a lab/measurement is grounded: its value appears as a whole token AND its name is mentioned. */
export const findLabPages = (name: string, value: string, pages: SourcePageText[]): SourcePageText[] => {
  const v = foldText(value).split(' ')[0];
  const nameWords = tokens(name);
  if (!v) return [];
  return pages.filter((p) => {
    const f = fold(p);
    return hasExact(f, v) && (nameWords.length === 0 || nameWords.some((w) => hasWord(f, w)));
  });
};
