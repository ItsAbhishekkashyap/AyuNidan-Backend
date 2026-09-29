import { Router } from 'express';
import { transcribeVoice } from '../controllers/voice.controller';
import { authGuard } from '../middleware/auth';
import { aiRateLimiter } from '../middleware/rateLimiter';
import { voiceUpload } from '../middleware/uploadValidation';

const router = Router();

router.post('/transcribe', authGuard, aiRateLimiter, voiceUpload, transcribeVoice);

export default router;
