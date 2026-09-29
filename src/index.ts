import 'dotenv/config';

import { getConfig, ConfigError } from './config/env';
import { connectDB } from './config/database';
import { createApp } from './app';
import { logger, errorMeta } from './utils/logger';
import { getEmbeddings } from './rag/embeddings';

// Fail fast on missing/weak required configuration (e.g. JWT_SECRET) before accepting traffic.
let port: number;
try {
  port = getConfig().PORT;
} catch (error) {
  console.error(error instanceof ConfigError ? error.message : 'Failed to load configuration');
  process.exit(1);
}

const app = createApp();

connectDB().then(() => {
  app.listen(port, () => {
    logger.info('server.started', { port });
    // Load the local embedding model in the background so the first upload/query does not
    // pay the one-time model download/load (measured: ~17 s first download, ~0.4 s cached load).
    if (process.env.HF_WARMUP !== 'false') {
      const started = Date.now();
      getEmbeddings()
        .embedQuery('warm-up')
        .then(() => logger.info('embeddings.warm', { model: getEmbeddings().spaceId, loadMs: Date.now() - started }))
        .catch((error: unknown) => logger.warn('embeddings.warmup_failed', errorMeta(error)));
    }
  });
});

export default app;
