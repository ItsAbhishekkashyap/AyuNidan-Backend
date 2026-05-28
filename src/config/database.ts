import mongoose, { ConnectOptions } from 'mongoose';
import dns from 'node:dns/promises';

const MONGODB_OPTIONS: ConnectOptions = {
  maxPoolSize: 10,
  minPoolSize: 2,
  socketTimeoutMS: 45000,
  serverSelectionTimeoutMS: 5000,
  heartbeatFrequencyMS: 10000,
};

const getMongoUri = (): string => {
  const uri = process.env['MONGODB_URI'];
  if (!uri) {
    throw new Error('MONGODB_URI is not defined in environment variables');
  }
  return uri;
};

export const connectDB = async (): Promise<void> => {
  try {
    if (process.env.NODE_ENV !== 'production') {
      dns.setServers(['8.8.8.8', '1.1.1.1']); 
    }
    mongoose.set('strictQuery', true);

    if (process.env['NODE_ENV'] === 'development') {
      mongoose.set(
        'debug',
        (
          collectionName: string,
          method: string,
          query: Record<string, unknown>
        ): void => {
          console.log(
            ` ${collectionName}.${method}`,
            JSON.stringify(query)
          );
        }
      );
    }

    await mongoose.connect(getMongoUri(), MONGODB_OPTIONS);
    console.log('MongoDB Connected with connection pooling');

    mongoose.connection.on('error', (err: Error): void => {
      console.error('MongoDB runtime error:', err.message);
    });

    mongoose.connection.on('disconnected', (): void => {
      console.warn('MongoDB disconnected. Attempting reconnect...');
    });

    mongoose.connection.on('reconnected', (): void => {
      console.log(' MongoDB reconnected');
    });

  } catch (error: unknown) {
    if (error instanceof Error) {
      console.error('MongoDB Connection Error:', error.message);
    } else {
      console.error('MongoDB Unknown Error');
    }
    process.exit(1);
  }
};

const gracefulShutdown = async (signal: string): Promise<void> => {
  console.log(`\n${signal} received. Closing MongoDB connection...`);
  await mongoose.connection.close();
  console.log('MongoDB connection closed.');
  process.exit(0);
};

process.on('SIGINT', () => gracefulShutdown('SIGINT'));
process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));