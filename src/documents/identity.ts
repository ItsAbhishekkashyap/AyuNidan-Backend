/**
 * Deterministic patient-identity extraction from LABELLED fields in the document text.
 * Labelled fields in the source are authoritative over anything the model returns; names that
 * appear elsewhere (e.g. "Dear <someone else>") are reported as a discrepancy, never silently merged.
 */

export interface DocumentIdentity {
  name?: string;
  age?: number;
  gender?: string;
  /** Other person-like names found in salutations, with the page they appear on (if known). */
  otherNames: { name: string; page?: number }[];
}

const clean = (s: string): string => s.replace(/\s+/g, ' ').replace(/[|:]+$/g, '').trim();

const GENDERS: Record<string, string> = { m: 'Male', male: 'Male', f: 'Female', female: 'Female', other: 'Other' };

export const normalizeName = (name: string): string => name.toLowerCase().replace(/[^a-z]/g, '');

export const extractIdentity = (pages: { page?: number; text: string }[]): DocumentIdentity => {
  const identity: DocumentIdentity = { otherNames: [] };

  for (const { page, text } of pages) {
    if (!identity.name) {
      // "Patient Name (Your name) :  NI BHASKARAN"  /  "Patient Name: X"  /  "Name of Patient : X"
      const m = text.match(/(?:Patient\s*Name|Name\s+of\s+(?:the\s+)?Patient)[^:\n]*:\s*([^\n|]{2,80})/i);
      if (m) identity.name = clean(m[1]);
    }
    if (identity.age === undefined) {
      // "Age/Gender (Your age/gender) : 58Y/Female"  /  "Age/Sex: 45 / M"  /  "Age: 45 years"
      const m = text.match(/Age\s*\/\s*(?:Gender|Sex)[^:\n]*:\s*(\d{1,3})\s*(?:Y|yrs?|years?)?\s*\/\s*(Male|Female|Other|M|F)\b/i);
      if (m) {
        identity.age = Number(m[1]);
        identity.gender = GENDERS[m[2].toLowerCase()];
      } else {
        const a = text.match(/\bAge\s*:\s*(\d{1,3})\b/i);
        if (a) identity.age = Number(a[1]);
      }
    }
    if (!identity.gender) {
      const g = text.match(/\b(?:Sex|Gender)\s*:\s*(Male|Female|Other|M|F)\b/i);
      if (g) identity.gender = GENDERS[g[1].toLowerCase()];
    }
    for (const m of text.matchAll(/\bDear\s+([A-Z][A-Za-z.'\- ]{2,60}?)\s*(?:\(|,|\n|$)/g)) {
      identity.otherNames.push({ name: clean(m[1]), ...(page !== undefined ? { page } : {}) });
    }
  }

  if (identity.age !== undefined && (identity.age < 0 || identity.age > 150)) identity.age = undefined;
  // Only keep "other names" that are genuinely different people. Compare whole words: "John Doe" and
  // "John A Doe" are the same person, but "NI BHASKARAN" is NOT "RAMANI BHASKARAN" (a substring match would say so).
  if (identity.name) {
    const words = (s: string) => s.toLowerCase().split(/[^a-z]+/).filter((w) => w.length > 1);
    const patient = new Set(words(identity.name));
    const subset = (a: string[], b: Set<string>) => a.length > 0 && a.every((w) => b.has(w));
    identity.otherNames = identity.otherNames.filter((o) => {
      const other = words(o.name);
      return other.length > 0 && !subset(other, patient) && !subset([...patient], new Set(other));
    });
  }
  return identity;
};
