import path from 'node:path';
import { listZipEntries, streamZipText, type ZipEntryInfo } from './zipReader';
import { decodeEntities, htmlToText, truncateAtSentence, usDateToIso } from './text';
import type { DatasetInfo, TerminologyRecord } from './types';

/**
 * Deterministic parsers for the two supplied NLM datasets. Both stream the ZIP entry and cut it into
 * record blocks; only fields that exist in the source are read, nothing is inferred or invented.
 *
 * MedlinePlus health-topics XML  → one record per ENGLISH topic:
 *   term = @title, aliases = <also-called> + <see-reference>, definition = <full-summary> (HTML → text),
 *   url = @url, meshDescriptor = <mesh-heading>/<descriptor>, dateCreated = @date-created.
 * MeSH descriptor XML (desc20XX.xml) → one record per topical descriptor (DescriptorClass 1) with a scope note:
 *   term = DescriptorName, aliases = entry terms (non-permuted Term/String across the descriptor's concepts),
 *   definition = ScopeNote of the preferred concept, treeNumbers, dateCreated = DateIntroduced, dateRevised = LastUpdated.
 */

export const MAX_DEFINITION_CHARS = 3000;
const ORGANIZATION = 'U.S. National Library of Medicine';

export interface ParseStats {
  blocksSeen: number;
  recordsParsed: number;
  skipped: Record<string, number>;
  duplicatesDropped: number;
}

const newStats = (): ParseStats => ({ blocksSeen: 0, recordsParsed: 0, skipped: {}, duplicatesDropped: 0 });
const skip = (stats: ParseStats, reason: string): void => {
  stats.skipped[reason] = (stats.skipped[reason] ?? 0) + 1;
};

const cleanTerm = (value: string): string => decodeEntities(value).replace(/\s+/g, ' ').trim();

const uniqueAliases = (term: string, candidates: string[]): string[] => {
  const seen = new Set([term.toLowerCase()]);
  const out: string[] = [];
  for (const raw of candidates) {
    const alias = cleanTerm(raw);
    if (!alias || alias.length > 200 || seen.has(alias.toLowerCase())) continue;
    seen.add(alias.toLowerCase());
    out.push(alias);
  }
  return out;
};

/** Yields text blocks that start with `open` and end with `close` (inclusive) from a text stream. */
async function* blocks(source: AsyncIterable<string>, open: string, close: string): AsyncGenerator<string> {
  let buffer = '';
  for await (const chunk of source) {
    buffer += chunk;
    for (;;) {
      const start = buffer.indexOf(open);
      if (start < 0) {
        buffer = buffer.slice(Math.max(0, buffer.length - open.length));
        break;
      }
      const end = buffer.indexOf(close, start);
      if (end < 0) {
        buffer = buffer.slice(start);
        break;
      }
      yield buffer.slice(start, end + close.length);
      buffer = buffer.slice(end + close.length);
    }
  }
}

const attr = (tag: string, name: string): string | undefined => {
  const m = tag.match(new RegExp(`\\b${name}="([^"]*)"`));
  return m ? decodeEntities(m[1]) : undefined;
};

const allTexts = (block: string, tag: string): string[] => [...block.matchAll(new RegExp(`<${tag}(?:\\s[^>]*)?>([\\s\\S]*?)</${tag}>`, 'g'))].map((m) => m[1]);

/* ───────────────────────── Entry lookup ───────────────────────── */

const findEntry = (zipPath: string, pattern: RegExp): ZipEntryInfo => {
  const entry = listZipEntries(zipPath).find((e) => pattern.test(e.name) && !e.name.endsWith('/'));
  if (!entry) throw new Error(`No entry matching ${pattern} in ${path.basename(zipPath)}`);
  return entry;
};

/* ───────────────────────── MedlinePlus ───────────────────────── */

export const parseMedlinePlusTopic = (block: string): TerminologyRecord | 'not_english' | 'no_definition' | 'malformed' => {
  const openTag = block.slice(0, block.indexOf('>') + 1);
  const language = attr(openTag, 'language');
  if (language && language !== 'English') return 'not_english';
  const id = attr(openTag, 'id');
  const title = attr(openTag, 'title');
  if (!id || !title) return 'malformed';

  const summaryRaw = allTexts(block, 'full-summary')[0];
  const definition = summaryRaw
    ? truncateAtSentence(
        htmlToText(summaryRaw)
          .split('\n')
          .filter((line) => !/^NIH(\s+Senior Health)?:/i.test(line)) // source attribution line, not content
          .join('\n')
          .trim(),
        MAX_DEFINITION_CHARS
      )
    : '';
  if (!definition) return 'no_definition';

  const descriptor = block.match(/<mesh-heading>\s*<descriptor id="([^"]+)">([\s\S]*?)<\/descriptor>/);
  const url = attr(openTag, 'url');
  const created = usDateToIso(attr(openTag, 'date-created'));
  return {
    recordId: `mplus:${id}`,
    dataset: 'medlineplus',
    sourceRecordId: id,
    term: cleanTerm(title),
    aliases: uniqueAliases(cleanTerm(title), [...allTexts(block, 'also-called'), ...allTexts(block, 'see-reference')]),
    definition,
    ...(url ? { url } : {}),
    ...(descriptor ? { meshDescriptor: { id: descriptor[1], name: cleanTerm(descriptor[2]) } } : {}),
    ...(created ? { dateCreated: created } : {}),
  };
};

export const readMedlinePlusInfo = async (zipPath: string): Promise<{ info: DatasetInfo; entry: ZipEntryInfo }> => {
  const entry = findEntry(zipPath, /\.xml$/i);
  let head = '';
  for await (const chunk of streamZipText(zipPath, entry)) {
    head += chunk;
    if (head.includes('>')) break;
  }
  const root = head.match(/<health-topics[^>]*>/)?.[0] ?? '';
  const generated = usDateToIso(attr(root, 'date-generated'));
  return {
    entry,
    info: {
      dataset: 'medlineplus',
      sourceName: 'MedlinePlus Health Topics',
      organization: ORGANIZATION,
      version: generated ?? path.basename(entry.name, '.xml'),
      ...(generated ? { publicationDate: generated } : {}),
      sourceFile: path.basename(zipPath),
    },
  };
};

export async function* parseMedlinePlus(zipPath: string, stats = newStats()): AsyncGenerator<TerminologyRecord> {
  const { entry } = await readMedlinePlusInfo(zipPath);
  const seen = new Set<string>();
  for await (const block of blocks(streamZipText(zipPath, entry), '<health-topic ', '</health-topic>')) {
    stats.blocksSeen++;
    const parsed = parseMedlinePlusTopic(block);
    if (typeof parsed === 'string') {
      skip(stats, parsed);
      continue;
    }
    if (seen.has(parsed.recordId)) {
      stats.duplicatesDropped++;
      continue;
    }
    seen.add(parsed.recordId);
    stats.recordsParsed++;
    yield parsed;
  }
}

/* ───────────────────────── MeSH ───────────────────────── */

const ymd = (block: string, tag: string): string | undefined => {
  const m = block.match(new RegExp(`<${tag}>\\s*<Year>(\\d{4})</Year>\\s*<Month>(\\d{1,2})</Month>\\s*<Day>(\\d{1,2})</Day>`));
  return m ? `${m[1]}-${m[2].padStart(2, '0')}-${m[3].padStart(2, '0')}` : undefined;
};

export const parseMeshDescriptor = (block: string): TerminologyRecord | 'not_topical' | 'no_definition' | 'malformed' => {
  const descriptorClass = block.match(/^<DescriptorRecord\s+DescriptorClass\s*=\s*"(\d)"/)?.[1];
  if (descriptorClass !== '1') return 'not_topical';
  const id = block.match(/<DescriptorUI>\s*(D\d+)\s*<\/DescriptorUI>/)?.[1];
  // Header fields come before the first ConceptList; the nested SeeRelated descriptors must not be mistaken for them.
  const conceptStart = block.indexOf('<ConceptList>');
  const header = conceptStart >= 0 ? block.slice(0, conceptStart) : block;
  const term = header.match(/<DescriptorName>\s*<String>([\s\S]*?)<\/String>/)?.[1];
  if (!id || !term || conceptStart < 0) return 'malformed';

  const conceptBlocks = block
    .slice(conceptStart)
    .split(/<Concept\s/)
    .slice(1);
  const preferred = conceptBlocks.find((c) => /^PreferredConceptYN="Y"/.test(c)) ?? conceptBlocks[0];
  const scopeNote = preferred?.match(/<ScopeNote>([\s\S]*?)<\/ScopeNote>/)?.[1];
  const definition = scopeNote ? decodeEntities(scopeNote).replace(/\s+/g, ' ').trim() : '';
  if (!definition) return 'no_definition';

  const entryTerms: string[] = [];
  for (const concept of conceptBlocks) {
    for (const t of concept.matchAll(/<Term\s[^>]*IsPermutedTermYN="N"[^>]*>\s*<TermUI>\s*T\d+\s*<\/TermUI>\s*<String>([\s\S]*?)<\/String>/g)) entryTerms.push(t[1]);
  }
  const preferredName = cleanTerm(term);
  const treeNumbers = allTexts(header, 'TreeNumber').map((t) => t.trim()).filter(Boolean);
  const created = ymd(header, 'DateIntroduced');
  const revised = ymd(header, 'LastUpdated');
  return {
    recordId: `mesh:${id}`,
    dataset: 'mesh',
    sourceRecordId: id,
    term: preferredName,
    aliases: uniqueAliases(preferredName, entryTerms),
    definition: truncateAtSentence(definition, MAX_DEFINITION_CHARS),
    ...(treeNumbers.length ? { treeNumbers } : {}),
    ...(created ? { dateCreated: created } : {}),
    ...(revised ? { dateRevised: revised } : {}),
  };
};

export const readMeshInfo = async (zipPath: string): Promise<{ info: DatasetInfo; entry: ZipEntryInfo }> => {
  const entry = findEntry(zipPath, /\.xml$/i);
  let head = '';
  for await (const chunk of streamZipText(zipPath, entry)) {
    head += chunk;
    if (head.includes('<DescriptorRecordSet')) break;
  }
  // The version label comes from the file itself (release year in the entry name, descYYYY.xml); no date is invented.
  const year = path.basename(entry.name).match(/(\d{4})/)?.[1];
  return {
    entry,
    info: {
      dataset: 'mesh',
      sourceName: 'Medical Subject Headings (MeSH) descriptors',
      organization: ORGANIZATION,
      version: year ? `MeSH ${year}` : path.basename(entry.name, '.xml'),
      sourceFile: path.basename(zipPath),
    },
  };
};

export async function* parseMesh(zipPath: string, stats = newStats()): AsyncGenerator<TerminologyRecord> {
  const { entry } = await readMeshInfo(zipPath);
  const seen = new Set<string>();
  for await (const block of blocks(streamZipText(zipPath, entry), '<DescriptorRecord ', '</DescriptorRecord>')) {
    stats.blocksSeen++;
    const parsed = parseMeshDescriptor(block);
    if (typeof parsed === 'string') {
      skip(stats, parsed);
      continue;
    }
    if (seen.has(parsed.recordId)) {
      stats.duplicatesDropped++;
      continue;
    }
    seen.add(parsed.recordId);
    stats.recordsParsed++;
    yield parsed;
  }
}

export { newStats };
