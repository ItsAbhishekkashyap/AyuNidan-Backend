import { Router } from 'express';
import { register, login, googleLogin } from '../controllers/auth.controller';
import { authIpRateLimiter, loginAccountRateLimiter } from '../middleware/rateLimiter';
import { validate } from '../middleware/validate';
import { registerSchema, loginSchema, googleLoginSchema } from '../validation/schemas';

const router = Router();

router.post('/register', authIpRateLimiter, validate({ body: registerSchema }), register);
router.post('/login', authIpRateLimiter, validate({ body: loginSchema }), loginAccountRateLimiter, login);
router.post('/google', authIpRateLimiter, validate({ body: googleLoginSchema }), googleLogin);

export default router;
