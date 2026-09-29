import { Request, Response } from 'express';
import { extractMedicalData } from '../services/ai.service';
import { validateAudio, AudioValidationError } from '../documents/audio';
import { logger, errorMeta } from '../utils/logger';
import { recordFailure } from '../utils/failures';
import { StageTimer } from '../utils/timing';
import { aiErrorResponse } from './aiErrors';

/**
 * Server-side voice fallback (used when the browser has no live speech recognition):
 * audio → validation → speech-capable model (Gemini) transcription + extraction in ONE call.
 * This path returns only a FINAL transcript — it is not live. Every failure is explicit.
 */
export const transcribeVoice = async (req: Request, res: Response): Promise<void> => {
  const audioFile = (req.files as Express.Multer.File[] | undefined)?.[0];
  if (!audioFile) {
    res.status(400).json({ success: false, error: 'No audio file provided', failureCategory: 'transcription_failure' });
    return;
  }

  const timer = new StageTimer();
  try {
    const check = validateAudio(audioFile);
    const { sourceDocuments: _sourceDocuments, modelReadText: _modelReadText, ...entities } = await timer.time('transcription', () => extractMedicalData('', [audioFile]));

    const transcript = entities.rawText.trim();
    if (!transcript && entities.symptoms.length === 0 && entities.medicines.length === 0 && entities.labValues.length === 0) {
      recordFailure('transcription_failure', { operation: 'voice', reason: 'empty_transcript' });
      res.status(422).json({ success: false, error: 'No speech could be transcribed from the recording', failureCategory: 'transcription_failure' });
      return;
    }

    logger.info('voice.transcribed', { sizeBytes: audioFile.size, durationSec: check.durationSec, ...timer.toLogMeta() });
    timer.applyHeader(res);
    res.status(200).json({ success: true, data: { ...entities, transcript, transcriptType: 'final', ...(check.durationSec !== undefined ? { durationSec: check.durationSec } : {}) } });
  } catch (error) {
    if (error instanceof AudioValidationError) {
      recordFailure('transcription_failure', { operation: 'voice', reason: 'invalid_audio' });
      res.status(error.status).json({ success: false, error: error.message, failureCategory: error.category });
      return;
    }
    logger.error('voice.extraction_failed', errorMeta(error));
    const { status, body } = aiErrorResponse(error, 'Audio processing failed');
    res.status(status).json(body);
  } finally {
    audioFile.buffer = Buffer.alloc(0);
  }
};
