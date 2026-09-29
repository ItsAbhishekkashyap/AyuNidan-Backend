/** Minimal byte fixtures with valid magic numbers (no real clinical content). */
export const pdfBytes = () => Buffer.from('%PDF-1.4\n1 0 obj << /Type /Catalog >> endobj\ntrailer << >>\n%%EOF\n');

export const pngBytes = () =>
  Buffer.concat([Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]), Buffer.alloc(32, 1)]);

export const jpegBytes = () => Buffer.concat([Buffer.from([0xff, 0xd8, 0xff, 0xe0]), Buffer.alloc(32, 1)]);

/** Valid PCM WAV: 8 kHz, 8-bit mono, `seconds` of silence (header carries a real duration). */
export const wavBytes = (seconds = 1) => {
  const dataSize = 8000 * seconds;
  const header = Buffer.alloc(44);
  header.write('RIFF', 0);
  header.writeUInt32LE(36 + dataSize, 4);
  header.write('WAVE', 8);
  header.write('fmt ', 12);
  header.writeUInt32LE(16, 16); // fmt chunk size
  header.writeUInt16LE(1, 20); // PCM
  header.writeUInt16LE(1, 22); // mono
  header.writeUInt32LE(8000, 24); // sample rate
  header.writeUInt32LE(8000, 28); // byte rate
  header.writeUInt16LE(1, 32); // block align
  header.writeUInt16LE(8, 34); // bits per sample
  header.write('data', 36);
  header.writeUInt32LE(dataSize, 40);
  return Buffer.concat([header, Buffer.alloc(dataSize, 0x80)]);
};

export const webmBytes = () => Buffer.concat([Buffer.from([0x1a, 0x45, 0xdf, 0xa3]), Buffer.alloc(4096, 0)]);

/**
 * Builds a small, valid PDF with a real text layer: one page per entry in `pages`,
 * each page containing the given lines. Used to test per-page extraction/citations.
 */
export const textPdf = (pages: string[][]): Buffer => {
  const escape = (s: string) => s.replace(/\\/g, '\\\\').replace(/\(/g, '\\(').replace(/\)/g, '\\)');
  const objects: string[] = [];
  const pageIds: number[] = [];
  // 1: catalog, 2: pages, 3: font, then (page, content) pairs
  let next = 4;
  const pageObjects: string[] = [];
  for (const lines of pages) {
    const pageId = next++;
    const contentId = next++;
    pageIds.push(pageId);
    const stream = ['BT', '/F1 12 Tf', '14 TL', '72 720 Td', ...lines.map((l) => `(${escape(l)}) Tj T*`), 'ET'].join('\n');
    pageObjects[pageId] = `<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> /Contents ${contentId} 0 R >>`;
    pageObjects[contentId] = `<< /Length ${Buffer.byteLength(stream)} >>\nstream\n${stream}\nendstream`;
  }
  objects[1] = '<< /Type /Catalog /Pages 2 0 R >>';
  objects[2] = `<< /Type /Pages /Kids [${pageIds.map((id) => `${id} 0 R`).join(' ')}] /Count ${pageIds.length} >>`;
  objects[3] = '<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>';
  for (let id = 4; id < next; id++) objects[id] = pageObjects[id];

  let body = '%PDF-1.4\n';
  const offsets: number[] = [];
  for (let id = 1; id < next; id++) {
    offsets[id] = Buffer.byteLength(body);
    body += `${id} 0 obj\n${objects[id]}\nendobj\n`;
  }
  const xrefOffset = Buffer.byteLength(body);
  body += `xref\n0 ${next}\n0000000000 65535 f \n`;
  for (let id = 1; id < next; id++) body += `${String(offsets[id]).padStart(10, '0')} 00000 n \n`;
  body += `trailer\n<< /Size ${next} /Root 1 0 R >>\nstartxref\n${xrefOffset}\n%%EOF\n`;
  return Buffer.from(body, 'latin1');
};
