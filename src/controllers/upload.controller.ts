import { Request, Response } from 'express';
import { extractMedicalData } from '../services/ai.service';
import { indexUploadedFiles } from '../services/document.service';
import { logger, errorMeta } from '../utils/logger';
import { StageTimer } from '../utils/timing';
import { aiErrorResponse } from './aiErrors';

export const processReport = async (req: Request, res: Response): Promise<void> => {
  const files = (req.files as Express.Multer.File[]) || [];
  const timer = new StageTimer();
  try {
    const bodyText = String(req.body.text || req.body.rawText || req.body.symptoms || '');

    if (files.length === 0 && !bodyText.trim()) {
      res.status(400).json({ success: false, error: 'No files or text provided' });
      return;
    }

    const extracted = await timer.time('extraction', () => extractMedicalData(bodyText, files));
    // sourceDocuments is internal (raw normalised text, used for indexing) and never returned to clients.
    const { sourceDocuments, modelReadText, ...entities } = extracted;

    // Index uploaded files for document RAG. Indexing failures never fail the extraction.
    const documents =
      files.length > 0
        ? await timer.time('indexing', () => indexUploadedFiles(req.user!.id, files, sourceDocuments ?? [], modelReadText ?? ''))
        : [];

    logger.info('upload.extracted', {
      fileCount: files.length,
      documentsIndexed: documents.filter((d) => d.status === 'indexed').length,
      documentsFailed: documents.filter((d) => d.status === 'failed').length,
      ...timer.toLogMeta(),
    });
    timer.applyHeader(res);
    res.status(200).json({ success: true, data: { ...entities, documents } });
  } catch (error) {
    logger.error('upload.extraction_failed', { fileCount: files.length, ...errorMeta(error) });
    const { status, body } = aiErrorResponse(error, 'Failed to extract medical data from the upload');
    res.status(status).json(body);
  } finally {
    for (const file of files) file.buffer = Buffer.alloc(0);
  }
};
