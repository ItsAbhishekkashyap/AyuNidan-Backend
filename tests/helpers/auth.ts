import jwt from 'jsonwebtoken';
import { Types } from 'mongoose';
import { db } from './fakeModels';

/** Registers a user in the fake DB and returns a valid bearer token for it. */
export const createTestUser = (email = `user-${Math.random().toString(36).slice(2)}@test.dev`) => {
  const id = new Types.ObjectId().toString();
  db.users.set(id, { _id: id, email, name: 'Test User', authProvider: 'local' });
  const token = jwt.sign({ id, email }, process.env.JWT_SECRET as string, { expiresIn: '1h' });
  return { id, email, token, auth: `Bearer ${token}` };
};
