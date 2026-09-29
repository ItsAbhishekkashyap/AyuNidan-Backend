import path from 'node:path';
import { Request, Response, NextFunction, RequestHandler } from 'express';
import multer from 'multer';
import { runWithRequestContext } from '../utils/requestContext';

/**
 * Hardened in-memory upload handling.
 *
 * - Size/count/field limits enforced while streaming (multer), so oversized
 *   uploads are rejected before being fully buffered.
 * - Declared MIME type must be on the allow-list AND match the file extension.
 * - File content is sniffed (magic bytes) and must match the declared type;
 *   client-supplied MIME types are never trusted alone.
 * - Empty files are rejected; filenames are reduced to a safe basename and are
 *   never used for filesystem access (memory storage only).
 *
 * Worst-case memory per request = maxFiles × maxFileSize.
 */

type Sniffer = (buf: Buffer) => boolean;

const startsWith = (buf: Buffer, bytes: number[], offset = 0): boolean =>
  buf.length >= offset + bytes.length && bytes.every((b, i) => buf[offset + i] === b);

const ascii = (s: string): number[] => [...s].map((c) => c.charCodeAt(0));

const isUtf8Text: Sniffer = (buf) => {
  if (buf.includes(0)) return false;
  try {
    new TextDecoder('utf-8', { fatal: true }).decode(buf);
    return true;
  } catch {
    return false;
  }
};

interface FileType {
  extensions: string[];
  sniff: Sniffer;
}

export const FILE_TYPES: Record<string, FileType> = {
  'application/pdf': { extensions: ['.pdf'], sniff: (b) => startsWith(b, ascii('%PDF-')) },
  'image/jpeg': { extensions: ['.jpg', '.jpeg'], sniff: (b) => startsWith(b, [0xff, 0xd8, 0xff]) },
  'image/png': { extensions: ['.png'], sniff: (b) => startsWith(b, [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]) },
  'image/webp': { extensions: ['.webp'], sniff: (b) => startsWith(b, ascii('RIFF')) && startsWith(b, ascii('WEBP'), 8) },
  'text/plain': { extensions: ['.txt'], sniff: isUtf8Text },
  'audio/mpeg': {
    extensions: ['.mp3'],
    sniff: (b) => startsWith(b, ascii('ID3')) || (b.length > 1 && b[0] === 0xff && (b[1] & 0xe0) === 0xe0),
  },
  'audio/wav': { extensions: ['.wav'], sniff: (b) => startsWith(b, ascii('RIFF')) && startsWith(b, ascii('WAVE'), 8) },
  'audio/webm': { extensions: ['.webm', '.weba'], sniff: (b) => startsWith(b, [0x1a, 0x45, 0xdf, 0xa3]) },
  'audio/mp4': { extensions: ['.m4a', '.mp4'], sniff: (b) => startsWith(b, ascii('ftyp'), 4) },
};

const MIME_ALIASES: Record<string, string> = {
  'image/jpg': 'image/jpeg',
  'image/pjpeg': 'image/jpeg',
  'audio/x-wav': 'audio/wav',
  'audio/wave': 'audio/wav',
  'audio/vnd.wave': 'audio/wav',
  'audio/mp3': 'audio/mpeg',
  'audio/x-m4a': 'audio/mp4',
};

/** Strips codec parameters (e.g. "audio/webm;codecs=opus") and resolves aliases. */
export const normalizeMime = (mime: string): string => {
  const base = mime.split(';')[0].trim().toLowerCase();
  return MIME_ALIASES[base] ?? base;
};

export const safeFilename = (name: string): string => {
  const base = path.basename(name.replace(/\\/g, '/'));
  const cleaned = base.replace(/[^\w.\- ]+/g, '_').replace(/^\.+/, '').slice(-100);
  return cleaned || 'upload';
};

class UploadValidationError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
    this.name = 'UploadValidationError';
  }
}

interface UploadPolicy {
  field: string;
  maxFiles: number;
  maxFileSizeBytes: number;
  allowedMimeTypes: string[];
  /** Require the extension to match the declared MIME type (true) or merely be an allowed extension (false). */
  strictExtension?: boolean;
}

const validateFile = (file: Express.Multer.File, policy: UploadPolicy): void => {
  const mime = normalizeMime(file.mimetype);
  const type = FILE_TYPES[mime];
  if (!type || !policy.allowedMimeTypes.includes(mime)) {
    throw new UploadValidationError('Unsupported file type', 415);
  }

  const ext = path.extname(file.originalname).toLowerCase();
  const allowedExtensions = policy.strictExtension === false
    ? policy.allowedMimeTypes.flatMap((m) => FILE_TYPES[m]?.extensions ?? [])
    : type.extensions;
  if (!allowedExtensions.includes(ext)) {
    throw new UploadValidationError('File extension does not match an allowed file type', 415);
  }

  if (!file.buffer || file.buffer.length === 0) {
    throw new UploadValidationError('Uploaded file is empty', 400);
  }

  if (!type.sniff(file.buffer)) {
    throw new UploadValidationError('File content does not match its declared type', 415);
  }

  file.mimetype = mime;
  file.originalname = safeFilename(file.originalname);
};

const multerErrorResponse = (err: unknown): { status: number; error: string } => {
  if (err instanceof UploadValidationError) return { status: err.status, error: err.message };
  if (err instanceof multer.MulterError) {
    switch (err.code) {
      case 'LIMIT_FILE_SIZE':
        return { status: 413, error: 'File too large' };
      case 'LIMIT_FILE_COUNT':
        return { status: 400, error: 'Too many files' };
      case 'LIMIT_UNEXPECTED_FILE':
        return { status: 400, error: 'Unexpected file field' };
      default:
        return { status: 400, error: 'Malformed upload' };
    }
  }
  return { status: 400, error: 'Malformed upload' };
};

/** Builds a middleware that parses and validates a multipart upload according to `policy`. */
export const secureUpload = (policy: UploadPolicy): RequestHandler => {
  const parser = multer({
    storage: multer.memoryStorage(),
    limits: {
      fileSize: policy.maxFileSizeBytes,
      files: policy.maxFiles,
      fields: 10,
      fieldSize: 100 * 1024,
      parts: policy.maxFiles + 10,
    },
    fileFilter: (_req, file, cb) => {
      const mime = normalizeMime(file.mimetype);
      if (FILE_TYPES[mime] && policy.allowedMimeTypes.includes(mime)) cb(null, true);
      else cb(new UploadValidationError('Unsupported file type', 415));
    },
  }).array(policy.field, policy.maxFiles);

  return (req: Request, res: Response, next: NextFunction): void => {
    const contentType = req.headers['content-type'] ?? '';
    if (!contentType.toLowerCase().startsWith('multipart/form-data')) {
      res.status(400).json({ success: false, error: 'Expected multipart/form-data' });
      return;
    }

    parser(req, res, (err: unknown) => {
      const files = (req.files as Express.Multer.File[] | undefined) ?? [];
      const release = () => files.forEach((f) => (f.buffer = Buffer.alloc(0)));

      if (err) {
        release();
        const { status, error } = multerErrorResponse(err);
        res.status(status).json({ success: false, error });
        return;
      }

      try {
        files.forEach((file) => validateFile(file, policy));
      } catch (validationError) {
        release();
        const { status, error } = multerErrorResponse(validationError);
        res.status(status).json({ success: false, error });
        return;
      }
      // Multer resumes on stream events; re-bind the request id for downstream logs.
      runWithRequestContext({ requestId: req.requestId as string }, next);
    });
  };
};

const mb = (name: string, fallback: number): number => {
  const value = Number(process.env[name]);
  return (Number.isFinite(value) && value > 0 ? value : fallback) * 1024 * 1024;
};

export const reportUpload = secureUpload({
  field: 'files',
  maxFiles: 5,
  maxFileSizeBytes: mb('UPLOAD_MAX_FILE_MB', 15),
  allowedMimeTypes: ['application/pdf', 'image/jpeg', 'image/png', 'image/webp', 'text/plain', 'audio/mpeg', 'audio/wav'],
});

export const voiceUpload = secureUpload({
  field: 'audio',
  maxFiles: 1,
  maxFileSizeBytes: mb('VOICE_MAX_FILE_MB', 15),
  allowedMimeTypes: ['audio/webm', 'audio/mp4', 'audio/wav', 'audio/mpeg'],
  // Browser recorders often produce webm/mp4 blobs with a generic filename.
  strictExtension: false,
});
