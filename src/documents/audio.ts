/**
 * Server-side audio checks for the voice pipeline (fallback path when the browser has
 * no live speech recognition). Audio is never silently ignored: every rejection is an
 * explicit, categorised `transcription_failure`.
 */

export class AudioValidationError extends Error {
  readonly category = 'transcription_failure' as const;
  constructor(
    message: string,
    readonly status: number
  ) {
    super(message);
    this.name = 'AudioValidationError';
  }
}

const envInt = (name: string, fallback: number): number => {
  const value = Number.parseInt(process.env[name] ?? '', 10);
  return Number.isFinite(value) && value > 0 ? value : fallback;
};

export const getAudioLimits = () => ({
  maxDurationSec: envInt('VOICE_MAX_DURATION_SEC', 300),
  /** Below this size a recording cannot contain meaningful speech. */
  minBytes: envInt('VOICE_MIN_BYTES', 2048),
});

/** Parses a RIFF/WAVE header; returns the duration or throws if the header is corrupt. */
export const wavDurationSec = (buffer: Buffer): number => {
  if (buffer.length < 44 || buffer.toString('ascii', 0, 4) !== 'RIFF' || buffer.toString('ascii', 8, 12) !== 'WAVE') {
    throw new AudioValidationError('Corrupted WAV audio (invalid header)', 400);
  }
  let offset = 12;
  let byteRate: number | undefined;
  while (offset + 8 <= buffer.length) {
    const id = buffer.toString('ascii', offset, offset + 4);
    const size = buffer.readUInt32LE(offset + 4);
    if (id === 'fmt ' && offset + 16 <= buffer.length) byteRate = buffer.readUInt32LE(offset + 16);
    if (id === 'data') {
      if (!byteRate) throw new AudioValidationError('Corrupted WAV audio (missing format chunk)', 400);
      const available = Math.min(size, buffer.length - offset - 8);
      return available / byteRate;
    }
    offset += 8 + size + (size % 2);
  }
  throw new AudioValidationError('Corrupted WAV audio (no data chunk)', 400);
};

export interface AudioCheck {
  /** Known only for formats whose header carries it (WAV); otherwise bounded by size limits. */
  durationSec?: number;
}

export const validateAudio = (file: Pick<Express.Multer.File, 'buffer' | 'mimetype' | 'size'>): AudioCheck => {
  const limits = getAudioLimits();
  if (!file.buffer || file.buffer.length < limits.minBytes) {
    throw new AudioValidationError('Audio is empty or too short to contain speech', 400);
  }
  if (file.mimetype === 'audio/wav') {
    const durationSec = wavDurationSec(file.buffer);
    if (durationSec > limits.maxDurationSec) {
      throw new AudioValidationError(`Audio exceeds the ${limits.maxDurationSec}s duration limit`, 413);
    }
    if (durationSec < 0.3) throw new AudioValidationError('Audio is too short to contain speech', 400);
    return { durationSec: Math.round(durationSec * 10) / 10 };
  }
  return {};
};
