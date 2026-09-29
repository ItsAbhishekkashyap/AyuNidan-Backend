/**
 * Medical (lab) table handling.
 *
 * Tables are detected deterministically — from PDF text positions or from delimited
 * text (| or tab separated) — and mapped to typed rows only when a header row lets us
 * identify the columns. Rows are never split across chunks, and each row keeps its
 * page/table provenance. Nothing here guesses values: unknown columns stay unmapped.
 */

export type ColumnRole = 'test' | 'result' | 'unit' | 'referenceRange' | 'flag' | 'date';

export interface LabTableRow {
  test: string;
  result: string;
  numericResult?: number;
  unit?: string;
  referenceRange?: string;
  flag?: string;
  date?: string;
  page?: number;
  tableId: string;
  rowIndex: number;
}

export interface ExtractedTable {
  tableId: string;
  page?: number;
  headers: string[];
  roles: (ColumnRole | null)[];
  rows: string[][];
  labRows: LabTableRow[];
  source: 'pdf_layout' | 'delimited_text';
}

const ROLE_PATTERNS: [ColumnRole, RegExp][] = [
  ['referenceRange', /\b(ref(erence)?|normal|biological)\b.*|\brange\b|\binterval\b/i],
  ['unit', /^units?\b/i],
  ['flag', /^(flag|status|remarks?|interpretation|abnormal|h\s*\/\s*l)\b/i],
  ['date', /^(date|collected|reported)\b/i],
  ['result', /^(result|value|observed( value)?|reading|your value)\b/i],
  ['test', /^(test|tests|investigation|parameter|analyte|component|test name|examination)\b/i],
];

export const detectRole = (header: string): ColumnRole | null => {
  const h = header.trim();
  for (const [role, pattern] of ROLE_PATTERNS) if (pattern.test(h)) return role;
  return null;
};

/** A header is usable only if it identifies at least the test and result columns. */
export const isLabHeader = (cells: string[]): boolean => {
  const roles = cells.map(detectRole);
  return roles.includes('test') && roles.includes('result');
};

const FLAGS: [RegExp, string][] = [
  [/^(h|hi|high|↑|\*h|above)$/i, 'High'],
  [/^(l|lo|low|↓|\*l|below)$/i, 'Low'],
  [/^(n|normal|within range|wnl)$/i, 'Normal'],
  [/^(critical|panic|c|hh|ll)$/i, 'Critical'],
];

export const normalizeFlag = (flag: string): string => {
  const f = flag.trim();
  for (const [pattern, label] of FLAGS) if (pattern.test(f)) return label;
  return f;
};

const parseNumber = (value: string): number | undefined => {
  const match = value.replace(/,/g, '').match(/^[<>≤≥]?\s*(-?\d+(\.\d+)?)$/);
  return match ? Number(match[1]) : undefined;
};

export const mapRow = (cells: string[], roles: (ColumnRole | null)[], meta: { tableId: string; rowIndex: number; page?: number }): LabTableRow | null => {
  const get = (role: ColumnRole) => {
    const index = roles.indexOf(role);
    const value = index >= 0 ? cells[index]?.trim() : undefined;
    return value ? value : undefined;
  };
  const test = get('test');
  const result = get('result');
  if (!test || !result) return null;
  const unit = get('unit');
  const referenceRange = get('referenceRange');
  const flag = get('flag');
  const date = get('date');
  const numericResult = parseNumber(result);
  return {
    test,
    result,
    ...(numericResult !== undefined ? { numericResult } : {}),
    ...(unit ? { unit } : {}),
    ...(referenceRange ? { referenceRange } : {}),
    ...(flag ? { flag: normalizeFlag(flag) } : {}),
    ...(date ? { date } : {}),
    ...(meta.page !== undefined ? { page: meta.page } : {}),
    tableId: meta.tableId,
    rowIndex: meta.rowIndex,
  };
};

/** Searchable text for one row, keeping every relationship explicit. */
export const labRowToText = (row: LabTableRow): string =>
  [
    `${row.test}:`,
    `Result = ${row.result}${row.unit ? ` ${row.unit}` : ''}`,
    row.referenceRange ? `Reference range = ${row.referenceRange}${row.unit && !row.referenceRange.includes(row.unit) ? ` ${row.unit}` : ''}` : undefined,
    row.flag ? `Flag = ${row.flag}` : undefined,
    row.date ? `Date = ${row.date}` : undefined,
    row.page !== undefined ? `Page = ${row.page}` : undefined,
  ]
    .filter(Boolean)
    .join('\n');

const buildTable = (
  headerCells: string[],
  body: string[][],
  meta: { tableId: string; page?: number; source: ExtractedTable['source'] }
): ExtractedTable => {
  const roles = headerCells.map(detectRole);
  const labRows = body
    .map((cells, rowIndex) => mapRow(cells, roles, { tableId: meta.tableId, rowIndex, page: meta.page }))
    .filter((r): r is LabTableRow => r !== null);
  return { tableId: meta.tableId, page: meta.page, headers: headerCells, roles, rows: body, labRows, source: meta.source };
};

/* ───────────── Delimited text tables ( | or tab ) ───────────── */

/** Splits a | or tab delimited row. Returns [] for a markdown separator row, null for non-table lines. */
const splitDelimited = (line: string): string[] | null => {
  const delimiter = line.includes('|') ? '|' : line.includes('\t') ? '\t' : null;
  if (!delimiter) return null;
  const cells = line.split(delimiter).map((c) => c.trim());
  // Drop empty leading/trailing cells from "| a | b |" style rows.
  while (cells.length && cells[0] === '') cells.shift();
  while (cells.length && cells[cells.length - 1] === '') cells.pop();
  if (cells.length < 2) return null;
  if (cells.every((c) => /^:?-{2,}:?$/.test(c))) return [];
  return cells;
};

export const detectDelimitedTables = (text: string, page?: number, idPrefix = 't'): ExtractedTable[] => {
  const tables: ExtractedTable[] = [];
  const lines = text.split('\n');
  let i = 0;
  while (i < lines.length) {
    const header = splitDelimited(lines[i]);
    if (header && header.length >= 2 && isLabHeader(header)) {
      const body: string[][] = [];
      let j = i + 1;
      while (j < lines.length) {
        const cells = splitDelimited(lines[j]);
        if (cells === null) break;
        if (cells.length > 0) body.push(cells); // [] = markdown separator row
        j++;
      }
      if (body.length > 0) {
        tables.push(buildTable(header, body, { tableId: `${idPrefix}${page ?? 0}-${tables.length}`, page, source: 'delimited_text' }));
      }
      i = j;
    } else {
      i++;
    }
  }
  return tables;
};

/* ───────────── PDF layout tables (from positioned text items) ───────────── */

export interface PositionedText {
  str: string;
  x: number;
  y: number;
  width: number;
}

interface Cell {
  text: string;
  x: number;
  end: number;
}

/** Groups items into visual lines (same baseline) and cells (separated by wide gaps). */
export const toLines = (items: PositionedText[], gap = 8, yTolerance = 2): Cell[][] => {
  const sorted = items.filter((i) => i.str.trim()).sort((a, b) => b.y - a.y || a.x - b.x);
  const lines: PositionedText[][] = [];
  for (const item of sorted) {
    const line = lines[lines.length - 1];
    if (line && Math.abs(line[0].y - item.y) <= yTolerance) line.push(item);
    else lines.push([item]);
  }
  return lines.map((line) => {
    const cells: Cell[] = [];
    for (const item of line.sort((a, b) => a.x - b.x)) {
      const last = cells[cells.length - 1];
      const distance = last ? item.x - last.end : Infinity;
      // Adjacent items (small positive gap) belong to one cell; wide gaps and overlapping
      // items (separately drawn text colliding, e.g. an over-long header) start a new cell.
      if (last && distance <= gap && distance >= -2) {
        last.text += (item.x - last.end > 1 ? ' ' : '') + item.str;
        last.end = item.x + item.width;
      } else {
        cells.push({ text: item.str, x: item.x, end: item.x + item.width });
      }
    }
    return cells.map((c) => ({ ...c, text: c.text.trim() }));
  });
};

export const detectLayoutTables = (lines: Cell[][], page?: number): ExtractedTable[] => {
  const tables: ExtractedTable[] = [];
  let i = 0;
  while (i < lines.length) {
    const header = lines[i];
    if (header.length >= 3 && isLabHeader(header.map((c) => c.text))) {
      // Column boundaries from header cell positions.
      const starts = header.map((c) => c.x);
      const columnOf = (x: number) => {
        let col = 0;
        for (let k = 0; k < starts.length; k++) if (x >= starts[k] - 12) col = k;
        return col;
      };
      const body: string[][] = [];
      let j = i + 1;
      while (j < lines.length && lines[j].length >= 2) {
        const row = new Array<string>(header.length).fill('');
        for (const cell of lines[j]) {
          const col = columnOf(cell.x);
          row[col] = row[col] ? `${row[col]} ${cell.text}` : cell.text;
        }
        body.push(row);
        j++;
      }
      if (body.length > 0) {
        tables.push(buildTable(header.map((c) => c.text), body, { tableId: `p${page ?? 0}-t${tables.length}`, page, source: 'pdf_layout' }));
      }
      i = j;
    } else {
      i++;
    }
  }
  return tables;
};
