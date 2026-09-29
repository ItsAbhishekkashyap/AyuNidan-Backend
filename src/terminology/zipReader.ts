import fs from 'node:fs';
import zlib from 'node:zlib';
import type { Readable } from 'node:stream';

/**
 * Minimal read-only ZIP reader (central directory + streaming inflate). It only ever OPENS the archive
 * for reading — the source datasets are never modified — and streams entries so the 300 MB MeSH XML is
 * never held in memory as a whole. Supports "stored" and "deflate" entries (ZIP64 is not needed here).
 */

export interface ZipEntryInfo {
  name: string;
  method: number;
  compressedSize: number;
  uncompressedSize: number;
  localHeaderOffset: number;
}

const EOCD_SIGNATURE = 0x06054b50;
const CENTRAL_SIGNATURE = 0x02014b50;
const LOCAL_SIGNATURE = 0x04034b50;

const readAt = (fd: number, length: number, position: number): Buffer => {
  const buffer = Buffer.alloc(length);
  const read = fs.readSync(fd, buffer, 0, length, position);
  return read === length ? buffer : buffer.subarray(0, read);
};

export const listZipEntries = (zipPath: string): ZipEntryInfo[] => {
  const fd = fs.openSync(zipPath, 'r');
  try {
    const size = fs.fstatSync(fd).size;
    const tailLength = Math.min(size, 65_557);
    const tail = readAt(fd, tailLength, size - tailLength);
    let eocd = -1;
    for (let i = tail.length - 22; i >= 0; i--) {
      if (tail.readUInt32LE(i) === EOCD_SIGNATURE) {
        eocd = i;
        break;
      }
    }
    if (eocd < 0) throw new Error('Not a valid ZIP archive (end of central directory not found)');
    const entryCount = tail.readUInt16LE(eocd + 10);
    const directorySize = tail.readUInt32LE(eocd + 12);
    const directoryOffset = tail.readUInt32LE(eocd + 16);
    const directory = readAt(fd, directorySize, directoryOffset);

    const entries: ZipEntryInfo[] = [];
    let cursor = 0;
    for (let i = 0; i < entryCount; i++) {
      if (directory.readUInt32LE(cursor) !== CENTRAL_SIGNATURE) throw new Error('Corrupt ZIP central directory');
      const method = directory.readUInt16LE(cursor + 10);
      const compressedSize = directory.readUInt32LE(cursor + 20);
      const uncompressedSize = directory.readUInt32LE(cursor + 24);
      const nameLength = directory.readUInt16LE(cursor + 28);
      const extraLength = directory.readUInt16LE(cursor + 30);
      const commentLength = directory.readUInt16LE(cursor + 32);
      const localHeaderOffset = directory.readUInt32LE(cursor + 42);
      const name = directory.toString('utf8', cursor + 46, cursor + 46 + nameLength);
      entries.push({ name, method, compressedSize, uncompressedSize, localHeaderOffset });
      cursor += 46 + nameLength + extraLength + commentLength;
    }
    return entries;
  } finally {
    fs.closeSync(fd);
  }
};

/** Streams the (decompressed) bytes of one entry. */
export const openZipEntry = (zipPath: string, entry: ZipEntryInfo): Readable => {
  const fd = fs.openSync(zipPath, 'r');
  let start: number;
  try {
    const header = readAt(fd, 30, entry.localHeaderOffset);
    if (header.readUInt32LE(0) !== LOCAL_SIGNATURE) throw new Error('Corrupt ZIP local header');
    start = entry.localHeaderOffset + 30 + header.readUInt16LE(26) + header.readUInt16LE(28);
  } finally {
    fs.closeSync(fd);
  }
  const raw = fs.createReadStream(zipPath, { start, end: start + entry.compressedSize - 1 });
  if (entry.method === 0) return raw;
  if (entry.method === 8) return raw.pipe(zlib.createInflateRaw());
  throw new Error(`Unsupported ZIP compression method ${entry.method}`);
};

/** Yields decoded text blocks of an entry (UTF-8 safe across chunk boundaries). */
export async function* streamZipText(zipPath: string, entry: ZipEntryInfo): AsyncGenerator<string> {
  const stream = openZipEntry(zipPath, entry);
  stream.setEncoding('utf8');
  for await (const chunk of stream) yield chunk as string;
}
