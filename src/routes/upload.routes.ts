import { Router } from 'express';
import { processReport } from '../controllers/upload.controller';
import { authGuard } from '../middleware/auth';
import { aiRateLimiter } from '../middleware/rateLimiter';
import { reportUpload } from '../middleware/uploadValidation';
import { validate } from '../middleware/validate';
import { uploadBodySchema } from '../validation/schemas';

const router = Router();

// Auth and rate limiting run BEFORE the multipart parser so unauthenticated requests never buffer files.
router.post('/', authGuard, aiRateLimiter, reportUpload, validate({ body: uploadBodySchema }), processReport);

export default router;
