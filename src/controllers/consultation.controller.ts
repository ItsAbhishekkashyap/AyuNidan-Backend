import { Request, Response } from 'express';
import { Types } from 'mongoose';
import { Consultation, FailureStage } from '../models/Consultation';
import { ApiResponse, LabValue, PatientDetails } from '../types';
import { extractMedicalData, AIProcessingError, AIConfigurationError, type StructuredReport } from '../services/ai.service';
import { assessRisk } from '../services/assessment.service';
import { explainMedicalTermRAG } from '../services/rag.service';
import { deleteDocumentsForConsultation, linkDocumentsToConsultation } from '../services/document.service';
import { invalidateUserCache } from '../middleware/cache';
import { logger, errorMeta } from '../utils/logger';
import { recordFailure, type FailureCategory } from '../utils/failures';
import { StageTimer } from '../utils/timing';

const escapeRegex = (value: string): string => value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

const failure = (res: Response, status: number, error: string): void => {
  res.status(status).json({ success: false, error });
};

/** Consultations that carry a low/medium/high conclusion (legacy records have no status). */
const HAS_CONCLUSION = { status: { $nin: ['processing', 'failed'] }, riskLevel: { $in: ['low', 'medium', 'high'] } };

interface CreateConsultationBody {
  rawText: string;
  voiceTranscript?: string;
  patientDetails?: PatientDetails;
  symptoms: string[];
  medicines: string[];
  labValues: LabValue[];
  documentIds: string[];
  /** Structured report from /uploads (clinician-reviewed), echoed back for display/assessment. */
  report?: Partial<StructuredReport>;
}

export const AI_FAILURE_MESSAGE =
  'AI processing failed: no clinical summary or risk assessment was generated. Your input was kept; please retry.';

export const createConsultation = async (req: Request, res: Response<ApiResponse<any>>): Promise<void> => {
  const userId = req.user!.id;
  const timer = new StageTimer();
  // Body is validated and bounded by createConsultationSchema.
  const body = req.body as CreateConsultationBody;
  const textToAnalyze = body.rawText || body.symptoms.join(' ') || body.voiceTranscript || '';
  const hasClientExtraction =
    Boolean(body.patientDetails?.name) || body.symptoms.length > 0 || body.medicines.length > 0 || body.labValues.length > 0;

  // 1. Persist the clinician's input first, explicitly marked as processing.
  let consultationId: string;
  try {
    const created = await timer.time('persist_input', () =>
      Consultation.create({
        userId,
        status: 'processing',
        patientDetails: body.patientDetails,
        rawText: textToAnalyze,
        voiceTranscript: body.voiceTranscript,
        symptoms: body.symptoms,
        medicines: body.medicines,
        labValues: body.labValues,
      })
    );
    consultationId = String(created._id);
  } catch (error) {
    recordFailure('persistence_failure', { stage: 'create' });
    logger.error('consultation.create_failed', errorMeta(error));
    failure(res, 500, 'Failed to create consultation');
    return;
  }
  invalidateUserCache(userId);

  // Link uploaded documents (only the caller's own; others are dropped). Non-fatal.
  let documentIds: string[] = [];
  if (body.documentIds.length > 0) {
    try {
      documentIds = await linkDocumentsToConsultation(userId, consultationId, body.documentIds);
    } catch (error) {
      logger.warn('consultation.link_documents_failed', errorMeta(error));
    }
  }

  // 2. AI processing. Any failure is recorded explicitly — never as a low-risk result.
  let stage: FailureStage = 'extraction';
  try {
    const extracted = hasClientExtraction
      ? {
          patientDetails: body.patientDetails,
          symptoms: body.symptoms,
          medicines: body.medicines,
          labValues: body.labValues,
          report: body.report,
          model: undefined,
        }
      : await timer.time('extraction', () => extractMedicalData(textToAnalyze));

    // JOB B: evidence-grounded assessment (separate call from extraction).
    stage = 'assessment';
    const assessment = await timer.time('assessment', () =>
      assessRisk(
        {
          patientDetails: extracted.patientDetails,
          symptoms: extracted.symptoms,
          medicines: extracted.medicines,
          labValues: extracted.labValues,
          rawText: textToAnalyze,
          voiceTranscript: body.voiceTranscript,
          report: extracted.report,
        },
        { userId, documentIds }
      )
    );
    const { provider, model, fallbackUsed, summary: assessmentSummary, riskLevel, riskScore, ...assessmentDetail } = assessment;

    stage = 'persistence';
    const extractionMeta = hasClientExtraction
      ? { source: 'client' }
      : {
          source: 'model',
          ...(extracted.model
            ? { provider: extracted.model.provider, model: extracted.model.model, fallbackUsed: extracted.model.fallbackUsed }
            : {}),
        };

    const completed = await timer.time('persist_result', () =>
      Consultation.findOneAndUpdate(
        { _id: consultationId, userId },
        {
          $set: {
            status: 'completed',
            patientDetails: extracted.patientDetails,
            symptoms: extracted.symptoms,
            medicines: extracted.medicines,
            labValues: extracted.labValues,
            summary: assessmentSummary,
            riskLevel,
            ...(riskScore !== undefined ? { riskScore } : {}),
            assessment: assessmentDetail,
            ...(extracted.report ? { report: extracted.report } : {}),
            ai: {
              extraction: extractionMeta,
              assessment: { provider, model, fallbackUsed },
            },
            documentIds,
            processingTime: timer.totalMs(),
          },
        },
        { new: true, runValidators: true }
      )
        .lean()
        .exec()
    );
    if (!completed) throw new Error('Consultation was removed during processing');

    invalidateUserCache(userId);
    logger.info('consultation.completed', {
      consultationId,
      riskLevel,
      assessmentModel: model,
      fallbackUsed,
      referencesUsed: assessment.medicalReferencesUsed.length,
      patientEvidenceUsed: assessment.patientEvidence.length,
      droppedCitations: assessment.validation.droppedCitations,
      ...timer.toLogMeta(),
    });
    timer.applyHeader(res);
    res.status(201).json({ success: true, data: completed });
  } catch (error) {
    const isAIError = error instanceof AIProcessingError || error instanceof AIConfigurationError;
    const category: FailureCategory = isAIError
      ? error.category
      : stage === 'persistence'
        ? 'persistence_failure'
        : 'extraction_failure';
    if (!isAIError) recordFailure(category, { stage });
    const attempts = error instanceof AIProcessingError ? error.attempts.length : undefined;

    const failureInfo = { stage, category, attempts, occurredAt: new Date() };
    let failedDoc: unknown = { _id: consultationId, status: 'failed', failure: failureInfo };
    try {
      failedDoc =
        (await Consultation.findOneAndUpdate(
          { _id: consultationId, userId },
          {
            $set: { status: 'failed', failure: failureInfo, documentIds, processingTime: timer.totalMs() },
            $unset: { summary: 1, riskLevel: 1, riskScore: 1, assessment: 1 },
          },
          { new: true }
        )
          .lean()
          .exec()) ?? failedDoc;
    } catch (persistError) {
      recordFailure('persistence_failure', { stage: 'mark_failed' });
      logger.error('consultation.mark_failed_failed', errorMeta(persistError));
    }
    invalidateUserCache(userId);

    logger.error('consultation.processing_failed', { consultationId, stage, category, attempts, ...timer.toLogMeta() });
    timer.applyHeader(res);
    const status = error instanceof AIConfigurationError ? 503 : isAIError ? 502 : 500;
    res.status(status).json({
      success: false,
      error: isAIError ? AI_FAILURE_MESSAGE : 'Failed to save the consultation result',
      data: failedDoc,
    });
  }
};

export const getConsultations = async (req: Request, res: Response<ApiResponse<any>>): Promise<void> => {
  try {
    const userId = req.user!.id;
    // page/limit/search are validated, bounded and trimmed by listConsultationsQuerySchema.
    const { page, limit, search } = req.query as unknown as { page: number; limit: number; search?: string };
    const skip = (page - 1) * limit;

    const queryFilter: Record<string, unknown> = { userId };
    if (search) {
      queryFilter['patientDetails.name'] = { $regex: escapeRegex(search), $options: 'i' };
    }

    const consultations = await Consultation.find(queryFilter)
      .sort({ createdAt: -1 })
      .skip(skip)
      .limit(limit)
      .lean()
      .exec();

    res.status(200).json({ success: true, data: consultations });
  } catch (error) {
    logger.error('consultation.list_failed', errorMeta(error));
    failure(res, 500, 'Failed to fetch consultations');
  }
};

export const getRiskDashboard = async (req: Request, res: Response<ApiResponse<any>>): Promise<void> => {
  try {
    const userId = req.user!.id;

    // Only consultations with a real AI conclusion contribute to risk statistics.
    const metrics = await Consultation.aggregate([
      { $match: { userId: new Types.ObjectId(userId), ...HAS_CONCLUSION } },
      { $group: { _id: '$riskLevel', count: { $sum: 1 }, averageScore: { $avg: '$riskScore' } } },
      { $project: { riskLevel: '$_id', count: 1, averageScore: { $round: ['$averageScore', 1] }, _id: 0 } },
    ]);

    const total = metrics.reduce((acc, curr) => acc + curr.count, 0);
    const failedCount = await Consultation.countDocuments({ userId, status: 'failed' }).exec();
    const insufficientEvidenceCount = await Consultation.countDocuments({ userId, riskLevel: 'insufficient_evidence' }).exec();
    res.status(200).json({ success: true, data: { total, distribution: metrics, failedCount, insufficientEvidenceCount } });
  } catch (error) {
    logger.error('consultation.dashboard_failed', errorMeta(error));
    failure(res, 500, 'Dashboard aggregation failed');
  }
};

/** Ownership is enforced in the query itself; another user's record is indistinguishable from a missing one. */
export const getConsultationById = async (req: Request, res: Response<ApiResponse<any>>): Promise<void> => {
  try {
    const id = req.params.id as string;
    if (!Types.ObjectId.isValid(id)) {
      failure(res, 400, 'Invalid ID format');
      return;
    }

    const consultation = await Consultation.findOne({ _id: id, userId: req.user!.id }).lean().exec();
    if (!consultation) {
      failure(res, 404, 'Consultation not found');
      return;
    }

    res.status(200).json({ success: true, data: consultation });
  } catch (error) {
    logger.error('consultation.get_failed', errorMeta(error));
    failure(res, 500, 'Failed to fetch consultation');
  }
};

export const deleteConsultation = async (req: Request, res: Response<ApiResponse<any>>): Promise<void> => {
  try {
    const id = req.params.id as string;
    const userId = req.user!.id;
    if (!Types.ObjectId.isValid(id)) {
      failure(res, 400, 'Invalid ID format');
      return;
    }

    const deleted = await Consultation.findOneAndDelete({ _id: id, userId }).lean().exec();
    if (!deleted) {
      failure(res, 404, 'Consultation not found');
      return;
    }

    invalidateUserCache(userId);
    // Data minimisation: remove linked documents and their vectors (best effort).
    const removedDocuments = await deleteDocumentsForConsultation(userId, id).catch((error: unknown) => {
      logger.warn('consultation.delete_documents_failed', errorMeta(error));
      return 0;
    });
    logger.info('consultation.deleted', { consultationId: id, removedDocuments });
    res.status(200).json({ success: true, message: 'Consultation permanently removed' });
  } catch (error) {
    logger.error('consultation.delete_failed', errorMeta(error));
    failure(res, 500, 'Deletion failed');
  }
};

export const explainTerm = async (req: Request, res: Response<ApiResponse<any>>): Promise<void> => {
  try {
    const { term } = req.query as { term: string }; // validated by explainQuerySchema
    const { explanation, grounded, groundedIn, sources, terminology } = await explainMedicalTermRAG(term);
    res.status(200).json({ success: true, data: { term, explanation, grounded, groundedIn, sources, terminology } });
  } catch (error) {
    logger.error('consultation.explain_failed', errorMeta(error));
    failure(res, 500, 'Explanation failed');
  }
};
