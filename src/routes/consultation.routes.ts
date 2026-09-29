import { Router } from 'express';
import {
  createConsultation,
  getConsultations,
  getConsultationById,
  deleteConsultation,
  getRiskDashboard,
  explainTerm,
} from '../controllers/consultation.controller';
import { aiRateLimiter, generalRateLimiter } from '../middleware/rateLimiter';
import { cacheMiddleware } from '../middleware/cache';
import { authGuard } from '../middleware/auth';
import { validate } from '../middleware/validate';
import {
  createConsultationSchema,
  explainQuerySchema,
  listConsultationsQuerySchema,
  objectIdParamSchema,
} from '../validation/schemas';

const router = Router();

// cacheMiddleware must stay after authGuard: cache keys are scoped to req.user.id.
router.get('/dashboard', authGuard, generalRateLimiter, cacheMiddleware(60), getRiskDashboard);
router.get('/explain', authGuard, aiRateLimiter, validate({ query: explainQuerySchema }), explainTerm);

router.post('/', authGuard, aiRateLimiter, validate({ body: createConsultationSchema }), createConsultation);
router.get('/', authGuard, generalRateLimiter, validate({ query: listConsultationsQuerySchema }), cacheMiddleware(30), getConsultations);
router.get('/:id', authGuard, generalRateLimiter, validate({ params: objectIdParamSchema }), cacheMiddleware(120), getConsultationById);
router.delete('/:id', authGuard, generalRateLimiter, validate({ params: objectIdParamSchema }), deleteConsultation);

export default router;
