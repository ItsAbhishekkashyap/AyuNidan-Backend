import { Request, Response } from 'express';
import { Consultation } from '../models/Consultation';
import { ApiResponse } from '../types';
import { generateClinicalSummary } from '../services/ai.service';
import { Types } from 'mongoose';
import { explainMedicalTermRAG, seedMedicalKnowledgeBase } from '../services/rag.service';
import { extractMedicalData } from '../services/ai.service';


declare module 'express-serve-static-core' {
  interface Request {
    user?: {
      id: string;
      email?: string;
      name?: string;
      role?: string;
    };
  }
}

export const createConsultation = async (
  req: Request,
  res: Response<ApiResponse<any>>
): Promise<void> => {
  try {
  
    const resolvedUserId = req.user?.id || req.body.userId;
    const { rawText, voiceTranscript } = req.body;

    if (!resolvedUserId || !Types.ObjectId.isValid(resolvedUserId)) {
      res.status(400).json({
        success: false,
        error: 'Valid User ID context is required',
      });
      return;
    }

    const startTime = Date.now();
    const textToAnalyze = rawText || (req.body.symptoms && req.body.symptoms.length > 0 ? req.body.symptoms.join(" ") : "") || voiceTranscript || "";

    const extractedData = await extractMedicalData(textToAnalyze);

    const aiResult = await generateClinicalSummary({
      symptoms: extractedData.symptoms,
      medicines: extractedData.medicines,
      labValues: extractedData.labValues,
      rawText: textToAnalyze,
      voiceTranscript
    });
    
    const processingTime = Date.now() - startTime;

    const consultation = await Consultation.create({
      userId: resolvedUserId,
      rawText: textToAnalyze,
      voiceTranscript,
      symptoms: extractedData.symptoms,    
      medicines: extractedData.medicines,  
      labValues: extractedData.labValues,  
      summary: aiResult.summary,
      riskLevel: aiResult.riskLevel,
      riskScore: aiResult.riskScore,
      processingTime
    });

    res.status(201).json({
      success: true,
      data: consultation,
    });
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Creation failed';
    console.error("🔥 CONTROLLER CRASH:", error);
    res.status(500).json({ success: false, error: errorMessage });
  }
};

export const getConsultations = async (
  req: Request,
  res: Response<ApiResponse<any>>
): Promise<void> => {
  try {
    const userId = req.user?.id;
    
   
    if (!userId) {
      res.status(401).json({ success: false, error: 'User token verification context lost.' });
      return;
    }

    const page = parseInt(req.query.page as string, 10) || 1;
    const limit = parseInt(req.query.limit as string, 10) || 10;
    const skip = (page - 1) * limit;

 
    const queryFilter = { userId };

    const consultations = await Consultation.find(queryFilter)
      .sort({ createdAt: -1 })
      .skip(skip)
      .limit(limit)
      .lean()
      .exec();

    res.status(200).json({
      success: true,
      data: consultations,
    });
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Fetch failed';
    res.status(500).json({ success: false, error: errorMessage });
  }
};

export const getRiskDashboard = async (
  req: Request,
  res: Response<ApiResponse<any>>
): Promise<void> => {
  try {
    const userId = req.user?.id;

    if (!userId) {
      res.status(401).json({ success: false, error: 'User token verification context lost.' });
      return;
    }

   
    const matchStage = { $match: { userId: new Types.ObjectId(userId) } };

    const metrics = await Consultation.aggregate([
      matchStage,
      {
        $group: {
          _id: '$riskLevel',
          count: { $sum: 1 },
          averageScore: { $avg: '$riskScore' }
        }
      },
      {
        $project: {
          riskLevel: '$_id',
          count: 1,
          averageScore: { $round: ['$averageScore', 1] },
          _id: 0
        }
      }
    ]);

    const totalConsultations = metrics.reduce((acc, curr) => acc + curr.count, 0);

    res.status(200).json({
      success: true,
      data: {
        total: totalConsultations,
        distribution: metrics
      }
    });
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Dashboard aggregation failed';
    res.status(500).json({ success: false, error: errorMessage });
  }
};

export const getConsultationById = async (
  req: Request,
  res: Response<ApiResponse<any>>
): Promise<void> => {
  try {
    const { id } = req.params;
    const userId = req.user?.id;

    if (!Types.ObjectId.isValid(id as string)) {
      res.status(400).json({ success: false, error: 'Invalid ID format' });
      return;
    }

    const consultation = await Consultation.findById(id as string).lean().exec();

    if (!consultation) {
      res.status(404).json({ success: false, error: 'Consultation not found' });
      return;
    }

 
    if (userId && consultation.userId.toString() !== userId) {
      res.status(403).json({ success: false, error: 'Unauthorized access to clinical diagnostic file' });
      return;
    }

    res.status(200).json({ success: true, data: consultation });
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Fetch failed';
    res.status(500).json({ success: false, error: errorMessage });
  }
};

export const deleteConsultation = async (
  req: Request,
  res: Response<ApiResponse<any>>
): Promise<void> => {
  try {
    const { id } = req.params;
    const userId = req.user?.id;

    if (!Types.ObjectId.isValid(id as string)) {
      res.status(400).json({ success: false, error: 'Invalid ID format' });
      return;
    }


    if (userId) {
      const existingRecord = await Consultation.findById(id as string).select('userId').lean().exec();
      if (existingRecord && existingRecord.userId.toString() !== userId) {
        res.status(403).json({ success: false, error: 'Unauthorized mutation block' });
        return;
      }
    }

    const deleted = await Consultation.findByIdAndDelete(id as string).lean().exec();

    if (!deleted) {
      res.status(404).json({ success: false, error: 'Consultation not found' });
      return;
    }

    res.status(200).json({
      success: true,
      message: 'Consultation permanently removed',
    });
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Deletion failed';
    res.status(500).json({ success: false, error: errorMessage });
  }
};

export const seedDatabase = async (
  req: Request,
  res: Response<ApiResponse<any>>
): Promise<void> => {
  try {
    const message = await seedMedicalKnowledgeBase();
    res.status(200).json({ success: true, data: { message } });
  } catch (error) {
    console.error("Seeding Error Context:", error);
    const errorMessage = error instanceof Error ? error.message : 'Seeding failed';
    res.status(500).json({ success: false, error: errorMessage });
  }
};

export const explainTerm = async (
  req: Request,
  res: Response<ApiResponse<any>>
): Promise<void> => {
  try {
    const { term } = req.query;

    if (!term || typeof term !== 'string') {
      res.status(400).json({ success: false, error: 'A medical term must be provided as a query parameter (e.g., ?term=tachycardia)' });
      return;
    }

    const explanation = await explainMedicalTermRAG(term);

    res.status(200).json({
      success: true,
      data: { term, explanation }
    });
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Explanation failed';
    res.status(500).json({ success: false, error: errorMessage });
  }
};