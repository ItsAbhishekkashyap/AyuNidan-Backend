// pdf-parse's package entry runs a debug self-test when loaded outside a parent
// module (e.g. under some loaders); importing the lib file directly avoids that.
declare module 'pdf-parse/lib/pdf-parse.js' {
  interface PdfPageData {
    getTextContent(options: { normalizeWhitespace: boolean; disableCombineTextItems: boolean }): Promise<unknown>;
  }
  interface PdfParseOptions {
    pagerender?: (pageData: PdfPageData) => Promise<string>;
    max?: number;
  }
  interface PdfParseResult {
    numpages: number;
    text: string;
  }
  function pdfParse(dataBuffer: Buffer | Uint8Array, options?: PdfParseOptions): Promise<PdfParseResult>;
  export default pdfParse;
}
