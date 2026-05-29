import { Request, Response } from 'express';
import bcrypt from 'bcryptjs';
import jwt from 'jsonwebtoken';
import { OAuth2Client } from 'google-auth-library';
import { User } from '../models/User';

const client = new OAuth2Client(process.env.GOOGLE_CLIENT_ID);
const JWT_SECRET = process.env.JWT_SECRET || 'fallback_secret_key_123';

// Helper to sign JWT token uniformly
const generateToken = (userId: string, email: string): string => {
  return jwt.sign({ id: userId, email }, JWT_SECRET, { expiresIn: '7d' });
};

// 1. Traditional Email/Password Register
export const register = async (req: Request, res: Response): Promise<void> => {
  try {
    const { name, email, password } = req.body;

    if (!name || !email || !password) {
      res.status(400).json({ message: 'All fields are mandatory' });
      return;
    }

    const userExists = await User.findOne({ email });
    if (userExists) {
      res.status(400).json({ message: 'User with this email already exists' });
      return;
    }

    // Hash the password securely
    const salt = await bcrypt.genSalt(10);
    const hashedPassword = await bcrypt.hash(password, salt);

    const newUser = await User.create({
      name,
      email,
      password: hashedPassword,
      authProvider: 'local',
    });

const token = generateToken(newUser._id.toString(), newUser.email);

    res.status(201).json({
      token,
      user: { id: newUser._id, name: newUser.name, email: newUser.email, avatar: newUser.avatar },
    });
  } catch (error) {
    res.status(500).json({ message: 'Server error during registration', error });
  }
};

// 2. Traditional Email/Password Login
export const login = async (req: Request, res: Response): Promise<void> => {
  try {
    const { email, password } = req.body;

    if (!email || !password) {
      res.status(400).json({ message: 'Email and password are required' });
      return;
    }

    const user = await User.findOne({ email });
    if (!user) {
      res.status(400).json({ message: 'Invalid credentials' });
      return;
    }

    if (user.authProvider === 'google') {
      res.status(400).json({ message: 'This email is linked to Google Auth. Please login via Google.' });
      return;
    }

    // Validate the password
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
    res.status(500).json({ message: 'Server error during login', error });
  }
};

// 3. Google OAuth 2.0 Ingestion Endpoint
export const googleLogin = async (req: Request, res: Response): Promise<void> => {
  try {
    const { idToken } = req.body;

    if (!idToken) {
      res.status(400).json({ message: 'Google ID Token is missing' });
      return;
    }

    // Verify token directly with Google public servers
    const ticket = await client.verifyIdToken({
      idToken,
      audience: process.env.GOOGLE_CLIENT_ID,
    });

    const payload = ticket.getPayload();
    if (!payload || !payload.email || !payload.name) {
      res.status(400).json({ message: 'Invalid token payload received from Google' });
      return;
    }

    let user = await User.findOne({ email: payload.email });

    if (!user) {
      // Create user context if hitting platform for the first time
      user = await User.create({
        name: payload.name,
        email: payload.email,
        googleId: payload.sub,
        avatar: payload.picture || '',
        authProvider: 'google',
      });
    } else if (user.authProvider !== 'google') {
      // Merge account or update googleId context safely if registered locally before
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
    res.status(400).json({ message: 'Google token authentication failed', error });
  }
};