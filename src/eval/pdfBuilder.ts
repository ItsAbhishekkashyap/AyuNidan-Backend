/**
 * Minimal, dependency-free PDF writer for SYNTHETIC test/evaluation fixtures.
 * Produces valid PDFs with a real text layer: free-text lines, tables whose cells are
 * positioned in columns (so layout-based table detection is exercised for real), and
 * optional embedded raster images (to exercise scanned/mixed page detection).
 */

export interface PdfTableSpec {
  /** x position of each column. */
  columns: number[];
  rows: string[][];
}

export interface PdfPageSpec {
  lines?: string[];
  table?: PdfTableSpec;
  /** Embed a small raster image on the page. */
  image?: boolean;
  /** Embed the image as a JPEG (DCTDecode) stream, like real scanned PDFs (implies `image`). */
  jpeg?: boolean;
}

/** A minimal, valid 1x1 grayscale JPEG. */
const JPEG_1X1_BASE64 =
  '/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////wgALCAABAAEBAREA/8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPxA=';

const escape = (s: string) => s.replace(/\\/g, '\\\\').replace(/\(/g, '\\(').replace(/\)/g, '\\)');

export const buildPdf = (pages: PdfPageSpec[]): Buffer => {
  const objects: string[] = [];
  const pageIds: number[] = [];
  // 1 catalog, 2 pages, 3 font, 4 shared 1x1 image, then (page, content) pairs
  objects[3] = '<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>';
  // Shared image XObject. Real scanned PDFs use JPEG (DCTDecode) streams, which pdf.js decodes via a
  // browser-only API — `jpeg: true` reproduces that so regressions are caught (see pdfAnalyzer.ts).
  const jpegBytes = Buffer.from(JPEG_1X1_BASE64, 'base64').toString('latin1');
  objects[4] = pages.some((p) => p.jpeg)
    ? `<< /Type /XObject /Subtype /Image /Width 1 /Height 1 /ColorSpace /DeviceGray /BitsPerComponent 8 /Filter /DCTDecode /Length ${jpegBytes.length} >>\nstream\n${jpegBytes}\nendstream`
    : '<< /Type /XObject /Subtype /Image /Width 1 /Height 1 /ColorSpace /DeviceGray /BitsPerComponent 8 /Length 1 >>\nstream\n\x80\nendstream';
  let next = 5;

  for (const spec of pages) {
    const pageId = next++;
    const contentId = next++;
    pageIds.push(pageId);
    const ops: string[] = [];
    let y = 740;
    for (const line of spec.lines ?? []) {
      ops.push(`BT /F1 11 Tf 72 ${y} Td (${escape(line)}) Tj ET`);
      y -= 16;
    }
    if (spec.table) {
      y -= 10;
      for (const row of spec.table.rows) {
        row.forEach((cell, i) => {
          if (cell) ops.push(`BT /F1 10 Tf ${spec.table!.columns[i]} ${y} Td (${escape(cell)}) Tj ET`);
        });
        y -= 16;
      }
    }
    const hasImage = Boolean(spec.image || spec.jpeg);
    if (hasImage) ops.push('q 200 0 0 200 72 200 cm /Im1 Do Q');
    const stream = ops.join('\n');
    objects[pageId] =
      `<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >>${hasImage ? ' /XObject << /Im1 4 0 R >>' : ''} >> /Contents ${contentId} 0 R >>`;
    objects[contentId] = `<< /Length ${Buffer.byteLength(stream, 'latin1')} >>\nstream\n${stream}\nendstream`;
  }
  objects[1] = '<< /Type /Catalog /Pages 2 0 R >>';
  objects[2] = `<< /Type /Pages /Kids [${pageIds.map((id) => `${id} 0 R`).join(' ')}] /Count ${pageIds.length} >>`;

  let body = '%PDF-1.4\n';
  const offsets: number[] = [];
  for (let id = 1; id < next; id++) {
    offsets[id] = Buffer.byteLength(body, 'latin1');
    body += `${id} 0 obj\n${objects[id]}\nendobj\n`;
  }
  const xrefOffset = Buffer.byteLength(body, 'latin1');
  body += `xref\n0 ${next}\n0000000000 65535 f \n`;
  for (let id = 1; id < next; id++) body += `${String(offsets[id]).padStart(10, '0')} 00000 n \n`;
  body += `trailer\n<< /Size ${next} /Root 1 0 R >>\nstartxref\n${xrefOffset}\n%%EOF\n`;
  return Buffer.from(body, 'latin1');
};

/** Standard 5-column lab table layout used by fixtures. */
export const LAB_COLUMNS = [72, 200, 280, 350, 470];
