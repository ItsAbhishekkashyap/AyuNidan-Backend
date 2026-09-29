import { Request, Response } from 'express';
import { askDocuments, deleteDocument, listDocuments, RetrievalError } from '../services/document.service';
import { AIConfigurationError, AIProcessingError } from '../services/ai.service';
import { logger, errorMeta } from '../utils/logger';
import { StageTimer } from '../utils/timing';

export const getDocuments = async (req: Request, res: Response): Promise<void> => {
  try {
    const { consultationId } = req.query as { consultationId?: string };
    const docs = await listDocuments(req.user!.id, consultationId);
    res.status(200).json({ success: true, data: docs });
  } catch (error) {
    logger.error('document.list_failed', errorMeta(error));
    res.status(500).json({ success: false, error: 'Failed to list documents' });
  }
};

export const removeDocument = async (req: Request, res: Response): Promise<void> => {
  try {
    const deleted = await deleteDocument(req.user!.id, req.params.id as string);
    if (!deleted) {
      res.status(404).json({ success: false, error: 'Document not found' });
      return;
    }
    res.status(200).json({ success: true, message: 'Document and its index entries removed' });
  } catch (error) {
    logger.error('document.delete_failed', errorMeta(error));
    res.status(500).json({ success: false, error: 'Failed to delete document' });
  }
};

/** Grounded Q&A over the caller's own documents, with citations built from retrieved metadata. */
export const queryDocuments = async (req: Request, res: Response): Promise<void> => {
  const timer = new StageTimer();
  try {
    const { question, documentIds, consultationId, topK } = req.body as {
      question: string;
      documentIds?: string[];
      consultationId?: string;
      topK?: number;
    };
    const answer = await askDocuments({ userId: req.user!.id, question, documentIds, consultationId, topK }, timer);
    logger.info('document.query', {
      status: answer.status,
      retrievalStatus: answer.retrieval.status,
      candidates: answer.retrieval.candidates,
      chunksUsed: answer.retrieval.used,
      citations: answer.citations.length,
      model: answer.model?.model,
      ...timer.toLogMeta(),
    });
    timer.applyHeader(res);
    res.status(200).json({ success: true, data: answer });
  } catch (error) {
    logger.error('document.query_failed', errorMeta(error));
    timer.applyHeader(res);
    if (error instanceof RetrievalError || error instanceof AIProcessingError) {
      res.status(502).json({
        success: false,
        error: 'Document question answering is temporarily unavailable',
        failureCategory: error.category,
      });
      return;
    }
    if (error instanceof AIConfigurationError) {
      res.status(503).json({ success: false, error: 'AI service is not configured', failureCategory: error.category });
      return;
    }
    res.status(500).json({ success: false, error: 'Document question answering failed' });
  }
};
