/**
 * Regression tests for issues found while ingesting the real ACC/AHA and WHO guideline PDFs:
 * two-column journal layout, running headers, bibliography chunks, look-alike headings and a pdf.js
 * font-loader crash. All inputs are synthetic strings/coordinates (no guideline text is stored in tests).
 */
import { describe, it, expect } from 'vitest';
import { splitTwoColumns, analyzePdf } from '../src/documents/pdfAnalyzer';
import { toLines, type PositionedText } from '../src/documents/tables';
import { isCitationDense, looksLikeHeading, removeRunningLines, splitSections } from '../src/services/reference.service';
import { buildPdf } from '../src/eval/pdfBuilder';

const word = (str: string, x: number, y: number, width = 60): PositionedText => ({ str, x, y, width });

describe('two-column pages', () => {
  const column = (x: number, prefix: string): PositionedText[] =>
    Array.from({ length: 20 }, (_, i) => [word(`${prefix}${i} sentence`, x, 700 - i * 12, 90), word('words here', x + 92, 700 - i * 12, 80)]).flat();

  it('reads the left column fully before the right column', () => {
    const items = [...column(50, 'L'), ...column(320, 'R')];
    const split = splitTwoColumns(items);
    expect(split).not.toBeNull();
    const text = [...toLines(split![0]), ...toLines(split![1])].map((l) => l.map((c) => c.text).join(' '));
    expect(text[0]).toContain('L0');
    expect(text[19]).toContain('L19');
    expect(text[20]).toContain('R0');
  });

  it('leaves genuine multi-cell tables alone', () => {
    const rows = Array.from({ length: 20 }, (_, i) => [word('Test', 50, 700 - i * 12, 40), word('Result', 150, 700 - i * 12, 40), word('Unit', 250, 700 - i * 12, 30), word('Range', 350, 700 - i * 12, 40), word('Flag', 450, 700 - i * 12, 30)]).flat();
    expect(splitTwoColumns(rows)).toBeNull();
  });

  it('does not split short pages', () => {
    expect(splitTwoColumns([word('a', 0, 0), word('b', 400, 0)])).toBeNull();
  });
});

describe('reference text cleaning', () => {
  it('drops running headers/footers that repeat across pages but keeps body text', () => {
    const topics = ['lipids', 'statins', 'diet', 'exercise', 'imaging', 'children', 'women', 'diabetes'];
    const pages = topics.map((t, i) => `Journal 2026;153:e${1150 + i} Some Guideline\nUnique body sentence about ${t}.`);
    const cleaned = removeRunningLines(pages);
    expect(cleaned.every((t) => !t.includes('Journal 2026'))).toBe(true);
    expect(cleaned[4]).toContain('Unique body sentence about imaging');
  });

  it('leaves short documents untouched', () => {
    expect(removeRunningLines(['a\nb', 'a\nb'])).toEqual(['a\nb', 'a\nb']);
  });

  it('recognises citation-dense bibliography chunks but not recommendation text', () => {
    const bib = '1. Smith A, et al. Lipids. Circulation. 2012;110:1-5. 2. Jones B, et al. Statins. Lancet. 2015;385:2-9. 3. Lee C, et al. Trial. JAMA. 2019;321:100-9.';
    expect(isCitationDense(bib)).toBe(true);
    expect(isCitationDense('In adults with elevated triglycerides, lifestyle change is recommended (2026;153:e1).')).toBe(false);
  });

  it('accepts real headings and rejects sentence fragments, citations and codes', () => {
    for (const ok of ['4.2.3.6. Selective Imaging of Subclinical Atherosclerosis', '1. Haemoglobin cutoffs to define anaemia', 'INTRODUCTION', '# Markdown heading']) {
      expect(looksLikeHeading(ok)).toBe(true);
    }
    for (const bad of ['3. FCS is a rare monogenic disorder characterized by', '6. Schwartz GG, Steg PG, et al Odyssey Outcomes', '1 B-NR', '1. Net benefit thresholds for moderate- and high-', '2. Among  patients  with  clinical  ASCVD']) {
      expect(looksLikeHeading(bad)).toBe(false);
    }
    expect(splitSections('Intro text\n4.1. Real Section Title\nBody', undefined).map((s) => s.section)).toEqual([undefined, '4.1. Real Section Title']);
  });
});

describe('pdf.js server safety', () => {
  it('disables browser-only font loading (uncaught ReferenceError on PDFs with embedded fonts)', async () => {
    await analyzePdf(buildPdf([{ lines: ['Synthetic page'] }]));
    expect((globalThis as { PDFJS?: { disableFontFace?: boolean } }).PDFJS?.disableFontFace).toBe(true);
  });
});
