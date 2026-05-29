import { Router } from 'express';
import consultationRoutes from './consultation.routes';
import uploadRoutes from './upload.routes';
import voiceRoutes from './voice.routes';
import userRoutes from './user.routes'; 
import authRoutes from './auth.routes'; 

const router = Router();

router.get('/debug-env-tokens-isolated', (req, res) => {
  const currentKey = process.env.GEMINI_API_KEY || '';
  res.json({
    hasKey: currentKey.length > 0,
    keyPrefix: currentKey ? currentKey.substring(0, 10) + '...' : 'EMPTY_NOT_FOUND',
    keyLength: currentKey.length,
    nodeEnv: process.env.NODE_ENV || 'development',
    systemTimestamp: new Date().toISOString()
  });
});

router.use('/auth', authRoutes);
router.use('/consultations', consultationRoutes);
router.use('/uploads', uploadRoutes);
router.use('/voice', voiceRoutes);
router.use('/users', userRoutes); 

export default router;