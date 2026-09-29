import { Router } from 'express';
import consultationRoutes from './consultation.routes';
import uploadRoutes from './upload.routes';
import voiceRoutes from './voice.routes';
import userRoutes from './user.routes';
import authRoutes from './auth.routes';
import documentRoutes from './document.routes';

const router = Router();

router.use('/auth', authRoutes);
router.use('/consultations', consultationRoutes);
router.use('/uploads', uploadRoutes);
router.use('/voice', voiceRoutes);
router.use('/users', userRoutes);
router.use('/documents', documentRoutes);

export default router;
