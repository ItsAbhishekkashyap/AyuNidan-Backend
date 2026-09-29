import { Request, Response } from 'express';
import bcrypt from 'bcryptjs';
import jwt from 'jsonwebtoken';
import { OAuth2Client } from 'google-auth-library';
import { User } from '../models/User';
import { getConfig } from '../config/env';
import { logger, errorMeta } from '../utils/logger';

const googleClient = new OAuth2Client();

const generateToken = (userId: string, email: string): string => {
  const { JWT_SECRET, JWT_EXPIRES_IN } = getConfig();
  return jwt.sign({ id: userId, email }, JWT_SECRET, {
    algorithm: 'HS256',
    expiresIn: JWT_EXPIRES_IN as jwt.SignOptions['expiresIn'],
  });
};

// Request bodies are validated and normalised (trimmed, lower-cased email) by validate() in auth.routes.ts.

export const register = async (req: Request, res: Response): Promise<void> => {
  try {
    const { name, email, password } = req.body;

    const userExists = await User.findOne({ email });
    if (userExists) {
      res.status(400).json({ message: 'User with this email already exists' });
      return;
    }

    const hashedPassword = await bcrypt.hash(password, await bcrypt.genSalt(10));
    const newUser = await User.create({ name, email, password: hashedPassword, authProvider: 'local' });
    const token = generateToken(newUser._id.toString(), newUser.email);

    res.status(201).json({
      token,
      user: { id: newUser._id, name: newUser.name, email: newUser.email, avatar: newUser.avatar },
    });
  } catch (error) {
    logger.error('auth.register_failed', { requestId: req.requestId, ...errorMeta(error) });
    res.status(500).json({ message: 'Server error during registration' });
  }
};

export const login = async (req: Request, res: Response): Promise<void> => {
  try {
    const { email, password } = req.body;

    const user = await User.findOne({ email });
    if (!user) {
      res.status(400).json({ message: 'Invalid credentials' });
      return;
    }

    if (user.authProvider === 'google') {
      res.status(400).json({ message: 'This email is linked to Google Auth. Please login via Google.' });
      return;
    }

    const isMatch = await bcrypt.compare(password, user.password || '');
    if (!isMatch) {
      res.status(400).json({ message: 'Invalid credentials' });
      return;
    }

    const token = generateToken(user._id.toString(), user.email);
    res.status(200).json({
      token,
      user: { id: user._id, name: user.name, email: user.email, avatar: user.avatar },
    });
  } catch (error) {
    logger.error('auth.login_failed', { requestId: req.requestId, ...errorMeta(error) });
    res.status(500).json({ message: 'Server error during login' });
  }
};

export const googleLogin = async (req: Request, res: Response): Promise<void> => {
  const audience = getConfig().GOOGLE_CLIENT_ID;
  if (!audience) {
    res.status(503).json({ message: 'Google sign-in is not configured' });
    return;
  }

  let payload;
  try {
    const ticket = await googleClient.verifyIdToken({ idToken: req.body.idToken, audience });
    payload = ticket.getPayload();
  } catch (error) {
    logger.warn('auth.google_token_rejected', { requestId: req.requestId, ...errorMeta(error) });
    res.status(401).json({ message: 'Google token authentication failed' });
    return;
  }

  if (!payload || !payload.email || !payload.name) {
    res.status(400).json({ message: 'Invalid token payload received from Google' });
    return;
  }
  if (payload.email_verified === false) {
    res.status(401).json({ message: 'Google account email is not verified' });
    return;
  }

  try {
    const email = payload.email.toLowerCase();
    let user = await User.findOne({ email });

    if (!user) {
      user = await User.create({
        name: payload.name,
        email,
        googleId: payload.sub,
        avatar: payload.picture || '',
        authProvider: 'google',
      });
    } else if (user.authProvider !== 'google') {
      user.authProvider = 'google';
      user.googleId = payload.sub;
      if (payload.picture) user.avatar = payload.picture;
      await user.save();
    }

    const token = generateToken(user._id.toString(), user.email);
    res.status(200).json({
      token,
      user: { id: user._id, name: user.name, email: user.email, avatar: user.avatar },
    });
  } catch (error) {
    logger.error('auth.google_login_failed', { requestId: req.requestId, ...errorMeta(error) });
    res.status(500).json({ message: 'Server error during Google login' });
  }
};
