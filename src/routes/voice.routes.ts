import { Router, Request, Response, NextFunction } from 'express';
import multer from 'multer';
import { transcribeVoice } from '../controllers/voice.controller';
import { aiRateLimiter } from '../middleware/rateLimiter';

const router = Router();

const audioStorage = multer.memoryStorage();
const audioUpload = multer({
  storage: audioStorage,
  limits: { fileSize: 25 * 1024 * 1024 }, 
  fileFilter: (req, file, cb) => {
    const allowed = ['audio/webm', 'audio/mp4', 'audio/wav', 'audio/mpeg'];
    if (allowed.includes(file.mimetype)) {
      cb(null, true);
    } else {
      cb(new Error('Invalid audio format'));
    }
  },
});

router.post('/transcribe', 
  aiRateLimiter, 
  (req: Request, res: Response, next: NextFunction) => {
    console.log("\n🌐 [VOICE ROUTE] Audio upload incoming...");
    audioUpload.single('audio')(req, res, (err: any) => {
      if (err) {
        console.error("🚨 [VOICE MULTER CRASH]:", err.message);
        return res.status(400).json({ success: false, error: err.message });
      }
      console.log("✅ [VOICE MULTER] Audio buffer parsed safely.");
      next();
    });
  },
  transcribeVoice
);

export default router;