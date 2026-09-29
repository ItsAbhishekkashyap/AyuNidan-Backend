import pdfParse from 'pdf-parse/lib/pdf-parse.js';
import { detectLayoutTables, toLines, type ExtractedTable, type PositionedText } from './tables';

/**
 * Local, deterministic PDF structure analysis (no LLM):
 * per page → text lines, layout-detected tables, image count, and a status.
 *   text    – has a usable text layer
 *   mixed   – text layer plus embedded images (image content needs vision)
 *   scanned – images but no usable text (needs OCR/vision)
 *   empty   – nothing extractable
 * No page is dropped: every page is reported with its status.
 */

/**
 * pdf.js (bundled in pdf-parse) decodes JPEG (DCT) images through the browser-only `Image`
 * class while building a page's operator list. In Node that is a ReferenceError thrown from a
 * message callback — outside any try/catch — which would crash the whole API process on any
 * scanned PDF. We only need to COUNT images, not decode them, so an inert stub that reports a
 * load error (pdf.js then simply resolves the image as null) is sufficient and safe.
 */
class InertImage {
  onload: (() => void) | null = null;
  onerror: (() => void) | null = null;
  set src(_value: string) {
    setTimeout(() => this.onerror?.(), 0);
  }
}
const ensureImageStub = (): void => {
  const g = globalThis as { Image?: unknown };
  if (typeof g.Image === 'undefined') g.Image = InertImage;
  // Same class of bug: while building operator lists pdf.js installs page fonts through the browser DOM
  // (`document.createElement('style')`) → uncaught ReferenceError that kills the process on real-world PDFs
  // with embedded fonts (found ingesting the WHO/ACC-AHA guidelines). Text extraction never needs font faces.
  // pdf.js reads its defaults from the global `PDFJS` settings object (created before it loads).
  const settings = globalThis as { PDFJS?: { disableFontFace?: boolean } };
  settings.PDFJS = { ...(settings.PDFJS ?? {}), disableFontFace: true };
};

export type PageStatus = 'text' | 'mixed' | 'scanned' | 'empty';

export interface AnalyzedPage {
  page: number;
  text: string;
  tables: ExtractedTable[];
  imageCount: number;
  status: PageStatus;
}

export interface PdfAnalysis {
  pageCount: number;
  pages: AnalyzedPage[];
}

// pdf.js 1.10 operator codes for painting raster images.
const IMAGE_OPS = new Set([82, 85, 86, 87, 88]);
const MIN_TEXT_CHARS = 20;

interface TextItem {
  str: string;
  transform: number[];
  width: number;
}

interface PageProxy {
  getTextContent(options: { normalizeWhitespace: boolean; disableCombineTextItems: boolean }): Promise<unknown>;
  getOperatorList?: () => Promise<{ fnArray: number[] }>;
}

/**
 * Two-column prose pages (journal-style guidelines) must be read column by column; reading line by line
 * across the gutter interleaves unrelated sentences and would corrupt every retrieved chunk.
 * A page is treated as two-column only when a clear vertical gutter exists AND each side reads as prose
 * (about one cell per visual line) — real tables (several cells per line) are left untouched.
 */
export const splitTwoColumns = (items: PositionedText[]): [PositionedText[], PositionedText[]] | null => {
  const words = items.filter((i) => i.str.trim() && i.width > 0);
  if (words.length < 60) return null;
  const right = Math.max(...words.map((i) => i.x + i.width));
  const left = Math.min(...words.map((i) => i.x));
  const span = right - left;
  if (span < 200) return null;
  let best: { g: number; crossing: number } | undefined;
  for (let g = left + span * 0.35; g <= left + span * 0.65; g += 2) {
    const crossing = words.filter((i) => i.x < g && i.x + i.width > g).length;
    if (!best || crossing < best.crossing) best = { g, crossing };
  }
  if (!best || best.crossing > words.length * 0.03) return null;
  const sideOf = (i: PositionedText): 0 | 1 => (i.x + i.width / 2 < best!.g ? 0 : 1);
  const sides: [PositionedText[], PositionedText[]] = [words.filter((i) => sideOf(i) === 0), words.filter((i) => sideOf(i) === 1)];
  if (sides.some((s) => s.length < words.length * 0.25)) return null;
  for (const side of sides) {
    const lines = toLines(side);
    if (lines.length < 12) return null;
    const cellsPerLine = lines.reduce((n, l) => n + l.length, 0) / lines.length;
    if (cellsPerLine > 1.4) return null;
  }
  return sides;
};

export const classifyPage =(textChars: number, imageCount: number): PageStatus =>
  textChars >= MIN_TEXT_CHARS ? (imageCount > 0 ? 'mixed' : 'text') : imageCount > 0 ? 'scanned' : 'empty';

export const analyzePdf = async (buffer: Buffer | Uint8Array): Promise<PdfAnalysis> => {
  ensureImageStub();
  // Small Node Buffers are views into a shared allocation pool; pdf.js reads/transfers the
  // underlying ArrayBuffer, so hand it an exact-size copy that owns its memory.
  const data = new Uint8Array(buffer.byteLength);
  data.set(buffer);

  const pages: AnalyzedPage[] = [];
  await pdfParse(data, {
    pagerender: async (pageData) => {
      const proxy = pageData as unknown as PageProxy;
      const content = (await proxy.getTextContent({ normalizeWhitespace: false, disableCombineTextItems: false })) as { items: TextItem[] };
      const positioned: PositionedText[] = content.items.map((item) => ({
        str: item.str,
        x: item.transform[4],
        y: item.transform[5],
        width: item.width ?? 0,
      }));
      const columns = splitTwoColumns(positioned);
      const lines = columns ? [...toLines(columns[0]), ...toLines(columns[1])] : toLines(positioned);
      const text = lines.map((cells) => cells.map((c) => c.text).join(cells.length >= 3 ? ' | ' : ' ')).join('\n');

      let imageCount = 0;
      try {
        const ops = await proxy.getOperatorList?.();
        imageCount = ops ? ops.fnArray.filter((op) => IMAGE_OPS.has(op)).length : 0;
      } catch {
        imageCount = 0;
      }

      const pageNumber = pages.length + 1;
      const tables = detectLayoutTables(lines, pageNumber);
      pages.push({ page: pageNumber, text, tables, imageCount, status: classifyPage(text.replace(/\s/g, '').length, imageCount) });
      return text;
    },
  });
  return { pageCount: pages.length, pages };
};
