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

const router = Router();

router.get('/dashboard', generalRateLimiter, cacheMiddleware(60), getRiskDashboard);
router.post('/seed-rag', seedDatabase); 
router.get('/explain', explainTerm);

router.post('/', aiRateLimiter, createConsultation);

router.get('/', generalRateLimiter, cacheMiddleware(30), getConsultations);

router.get('/:id', generalRateLimiter, cacheMiddleware(120), getConsultationById);

router.delete('/:id', generalRateLimiter, deleteConsultation);

export default router;