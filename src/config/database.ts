import mongoose, { ConnectOptions } from 'mongoose';
import dns from 'node:dns/promises';
import { getConfig } from './env';
import { logger, errorMeta } from '../utils/logger';

const MONGODB_OPTIONS: ConnectOptions = {
  maxPoolSize: 10,
  minPoolSize: 2,
  socketTimeoutMS: 45000,
  serverSelectionTimeoutMS: 5000,
  heartbeatFrequencyMS: 10000,
};

export const connectDB = async (): Promise<void> => {
  const config = getConfig();
  try {
    if (config.NODE_ENV !== 'production') {
      dns.setServers(['8.8.8.8', '1.1.1.1']);
    }
    mongoose.set('strictQuery', true);

    if (config.NODE_ENV === 'development') {
      // Log collection + operation only; query filters/documents may contain PHI.
      mongoose.set('debug', (collectionName: string, method: string): void => {
        logger.info('db.query', { collection: collectionName, method });
      });
    }

    await mongoose.connect(config.MONGODB_URI, MONGODB_OPTIONS);
    logger.info('db.connected');

    mongoose.connection.on('error', (err: Error) => logger.error('db.runtime_error', errorMeta(err)));
    mongoose.connection.on('disconnected', () => logger.warn('db.disconnected'));
    mongoose.connection.on('reconnected', () => logger.info('db.reconnected'));
  } catch (error: unknown) {
    logger.error('db.connection_failed', errorMeta(error));
    process.exit(1);
  }
};

const gracefulShutdown = async (signal: string): Promise<void> => {
  logger.info('server.shutdown', { signal });
  await mongoose.connection.close();
  process.exit(0);
};

process.on('SIGINT', () => gracefulShutdown('SIGINT'));
process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));
