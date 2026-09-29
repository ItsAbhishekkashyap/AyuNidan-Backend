import { Request, Response, NextFunction } from 'express';
import jwt from 'jsonwebtoken';
import { User } from '../models/User';
import { getConfig } from '../config/env';

interface JwtPayload {
  id: string;
  email: string;
}

export const authGuard = async (req: Request, res: Response, next: NextFunction): Promise<void> => {
  try {
    const authHeader = req.headers.authorization;

    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      res.status(401).json({ message: 'Authorization token missing or malformed' });
      return;
    }

    const token = authHeader.slice('Bearer '.length).trim();
    const decoded = jwt.verify(token, getConfig().JWT_SECRET, { algorithms: ['HS256'] }) as JwtPayload;

    if (typeof decoded.id !== 'string' || !/^[a-f\d]{24}$/i.test(decoded.id)) {
      res.status(401).json({ message: 'Invalid or expired authentication token' });
      return;
    }

    const userExists = await User.findById(decoded.id).select('_id');
    if (!userExists) {
      res.status(401).json({ message: 'User bound to this token no longer exists' });
      return;
    }

    req.user = { id: decoded.id, email: decoded.email };
    next();
  } catch {
    res.status(401).json({ message: 'Invalid or expired authentication token' });
  }
};
