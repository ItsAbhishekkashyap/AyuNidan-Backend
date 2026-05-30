import { Request, Response } from 'express';
import { extractMedicalData } from '../services/ai.service';
import { ApiResponse, ExtractedEntities } from '../types';

export const transcribeVoice = async (
  req: Request,
  res: Response<ApiResponse<ExtractedEntities>>
): Promise<void> => {
  try {
    console.log("==========================================");
    console.log(" Processing audio note...");
    console.log("==========================================");

    if (!req.file) {
      res.status(400).json({
        success: false,
        error: 'No audio file provided',
      });
      return;
    }

    const audioFile = req.file as Express.Multer.File;
    
    const extractedEntities = await extractMedicalData(
      'Extract patient symptoms, medications, and lab values from this voice note.', 
      [audioFile]
    );

    audioFile.buffer = Buffer.alloc(0);

    res.status(200).json({
      success: true,
      data: extractedEntities,
    });
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Audio processing failed';
    console.error("🔥 Voice Controller Error:", error);
    res.status(500).json({
      success: false,
      error: errorMessage,
    });
  }
};