import { Request, Response } from 'express';
import { extractMedicalData } from '../services/ai.service';

export const processReport = async (
  req: Request,
  res: Response
): Promise<void> => {
  try {
    console.log("==========================================");
    console.log("📥 [UPLOAD CONTROLLER] Request arrived!");
    console.log("==========================================");
    
    const bodyText = req.body.text || req.body.rawText || req.body.symptoms || "";
    const files = (req.files as Express.Multer.File[]) || [];

    if (files.length === 0 && !bodyText) {
      res.status(400).json({ success: false, error: 'No files or text provided' });
      return;
    }

    const mediaFiles: Express.Multer.File[] = [];

    for (const file of files) {
      if (
        file.mimetype === 'application/pdf' ||
        file.mimetype.startsWith('image/') ||
        file.mimetype.startsWith('audio/')
      ) {
        mediaFiles.push(file);
      }
    }
    
    const payloadText = bodyText.trim() ? `User Input Text: ${bodyText}\n` : "Please extract medical data from the provided files.";
    
    const extractedEntities = await extractMedicalData(payloadText, mediaFiles);

    for (const file of files) {
      file.buffer = Buffer.alloc(0);
    }

    res.status(200).json({
      success: true,
      data: extractedEntities,
    });

  } catch (error: any) {
    console.log("\n\n🔥 [UPLOAD CONTROLLER CRASH] 🔥");
    console.dir(error, { depth: null });
    
    const errorMessage = error?.message || error?.toString() || "Unknown API Error";
    res.status(500).json({ success: false, error: errorMessage });
  }
};