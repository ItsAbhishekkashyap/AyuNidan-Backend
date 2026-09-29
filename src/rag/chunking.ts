/**
 * Deterministic text normalisation + chunking for document retrieval.
 *
 * Chunks are bounded (`chunkSize` characters) with `overlap` characters of shared
 * context, preferring to break at paragraph → line → sentence → word boundaries
 * within the last 40% of the window. The defaults (1000 / 150) keep each chunk
 * well under the embedding model's input limit (bge-small-en-v1.5: 512 tokens);
 * they are engineering defaults, not tuned for clinical optimality.
 */

export interface ChunkOptions {
  chunkSize: number;
  overlap: number;
}

export interface TextChunk {
  index: number;
  text: string;
  /** Character offsets into the normalised input. */
  start: number;
  end: number;
}

export class ChunkingError extends Error {
  readonly category = 'chunking_failure' as const;
  constructor(message: string) {
    super(message);
    this.name = 'ChunkingError';
  }
}

const envInt = (name: string, fallback: number): number => {
  const value = Number.parseInt(process.env[name] ?? '', 10);
  return Number.isFinite(value) && value > 0 ? value : fallback;
};

export const getChunkOptions = (): ChunkOptions => ({
  chunkSize: envInt('RAG_CHUNK_SIZE', 1000),
  overlap: envInt('RAG_CHUNK_OVERLAP', 150),
});

const BREAKS = ['\n\n', '\n', '. ', ' '];

/** Normalises whitespace/control characters deterministically. */
export const normalizeText = (text: string): string =>
  text
    .replace(/\r\n?/g, '\n')
    // eslint-disable-next-line no-control-regex
    .replace(/[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]/g, '')
    .replace(/[ \t\f\v ]+/g, ' ')
    .replace(/ *\n */g, '\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim();

export const chunkText = (input: string, options: ChunkOptions = getChunkOptions()): TextChunk[] => {
  const { chunkSize, overlap } = options;
  if (!Number.isInteger(chunkSize) || chunkSize < 50) throw new ChunkingError('chunkSize must be an integer >= 50');
  if (!Number.isInteger(overlap) || overlap < 0 || overlap >= chunkSize / 2) {
    throw new ChunkingError('overlap must be an integer in [0, chunkSize / 2)');
  }

  const text = normalizeText(input);
  const chunks: TextChunk[] = [];
  const minBreak = Math.floor(chunkSize * 0.6);
  let start = 0;

  while (start < text.length) {
    let end = Math.min(start + chunkSize, text.length);
    if (end < text.length) {
      const window = text.slice(start, end);
      for (const sep of BREAKS) {
        const idx = window.lastIndexOf(sep);
        if (idx >= minBreak) {
          end = start + idx + sep.length;
          break;
        }
      }
    }

    const piece = text.slice(start, end).trim();
    if (piece) chunks.push({ index: chunks.length, text: piece, start, end });
    if (end >= text.length) break;

    // Step back `overlap` characters, aligned forward to a word boundary.
    let next = end - overlap;
    const space = text.indexOf(' ', next);
    if (space !== -1 && space < end) next = space + 1;
    start = Math.max(next, start + 1);
  }

  return chunks;
};
