// src/index.ts

import 'dotenv/config';

import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import morgan from 'morgan';
import compression from 'compression';
import { connectDB } from './config/database';
import { errorHandler } from './middleware/errorHandler';
import routes from './routes/index';


const app = express();
const PORT = process.env.PORT || 5001;

app.use(compression());

app.use(helmet({
  crossOriginResourcePolicy: { policy: 'cross-origin' },
}));

app.use(cors({
  origin: process.env.NODE_ENV === 'production'
    ? 'https://ayunidan.vercel.app/'
    : 'http://localhost:3000',
  methods: ['GET', 'POST', 'DELETE'],
  allowedHeaders: ['Content-Type', 'Authorization'],

  maxAge: 86400,
}));


app.use((req, res, next) => {
  res.setHeader('Connection', 'keep-alive');
  next();
});

app.use(morgan('dev'));
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));


app.use('/api', routes);


app.get('/health', (req, res) => {
  res.json({
    status: 'Running',
    uptime: process.uptime(),
    memory: process.memoryUsage(),
    timestamp: new Date().toISOString(),
  });
});


app.use(errorHandler);
app.use((err: any, req: any, res: any, next: any) => {
  console.log("🛑 [GLOBAL CRASH]:", err); // Ye asli error terminal me layega
  res.status(500).json({ success: false, error: err.message || "Internal server error" });
});

connectDB().then(() => {
  app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
});

export default app;