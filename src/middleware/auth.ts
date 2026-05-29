import { Request, Response, NextFunction } from 'express';
import jwt from 'jsonwebtoken';
import { User } from '../models/User';


declare global {
  namespace Express {
    interface Request {
      user?: {
        id: string;
        email: string;
      };
    }
  }
}

interface JwtPayload {
  id: string;
  email: string;
}

export const authGuard = async (
  req: Request,
  res: Response,
  next: NextFunction
): Promise<void> => {
  try {
    const authHeader = req.headers.authorization;

    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      res.status(401).json({ message: 'Authorization token missing or malformed' });
      return;
    }

    const token = authHeader.split(' ')[1];
    const jwtSecret = process.env.JWT_SECRET || 'fallback_secret_key_123';

    const decoded = jwt.verify(token, jwtSecret) as JwtPayload;

    
    const userExists = await User.findById(decoded.id).select('_id');
    if (!userExists) {
      res.status(401).json({ message: 'User bound to this token no longer exists' });
      return;
    }

    
    req.user = {
      id: decoded.id,
      email: decoded.email,
    };

    next();
  } catch (error) {
    res.status(401).json({ message: 'Invalid or expired authentication token' });
  }
};