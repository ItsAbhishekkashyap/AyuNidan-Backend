import { Router } from 'express';
import {
  createConsultation,
  getConsultations,
  getConsultationById,
  deleteConsultation,
  getRiskDashboard,
  explainTerm,        
  seedDatabase
} from '../controllers/consultation.controller';
import { aiRateLimiter, generalRateLimiter } from '../middleware/rateLimiter';
import { cacheMiddleware } from '../middleware/cache';
import { authGuard } from '../middleware/auth'; // 

const router = Router();


router.get('/dashboard', authGuard, generalRateLimiter, cacheMiddleware(60), getRiskDashboard);
router.get('/explain', authGuard, explainTerm);

router.post('/', authGuard, aiRateLimiter, createConsultation);
router.get('/', authGuard, generalRateLimiter, cacheMiddleware(30), getConsultations);
router.get('/:id', authGuard, cacheMiddleware(120), getConsultationById);
router.delete('/:id', authGuard, generalRateLimiter, deleteConsultation);


router.post('/seed-rag', seedDatabase); 

export default router;