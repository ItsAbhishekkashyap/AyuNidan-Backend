import { Router } from 'express';
import consultationRoutes from './consultation.routes';
import uploadRoutes from './upload.routes';
import voiceRoutes from './voice.routes';
import userRoutes from './user.routes'; 

const router = Router();


router.use('/consultations', consultationRoutes);
router.use('/uploads', uploadRoutes);
router.use('/voice', voiceRoutes);
router.use('/users', userRoutes); 

export default router;