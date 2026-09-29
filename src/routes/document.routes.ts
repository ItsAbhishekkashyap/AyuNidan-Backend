import { Router } from 'express';
import { getDocuments, queryDocuments, removeDocument } from '../controllers/document.controller';
import { authGuard } from '../middleware/auth';
import { aiRateLimiter, generalRateLimiter } from '../middleware/rateLimiter';
import { validate } from '../middleware/validate';
import { documentQuerySchema, listDocumentsQuerySchema, objectIdParamSchema } from '../validation/schemas';

const router = Router();

router.get('/', authGuard, generalRateLimiter, validate({ query: listDocumentsQuerySchema }), getDocuments);
router.post('/query', authGuard, aiRateLimiter, validate({ body: documentQuerySchema }), queryDocuments);
router.delete('/:id', authGuard, generalRateLimiter, validate({ params: objectIdParamSchema }), removeDocument);

export default router;
