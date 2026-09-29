/**
 * Creates the Pinecone serverless index for the local embedding space if it does not exist
 * (cosine metric; dimension from the configured Hugging Face model, 384 for bge-small).
 * The previous `panscience-medical` index (1024-d, e5 embeddings) is left untouched.
 * Usage: npm run vector:setup
 */
import 'dotenv/config';
import dns from 'node:dns';
import { PineconeVectorStore, getPineconeIndexName } from '../rag/vectorStore';
import { getHuggingFaceConfig } from '../rag/embeddings';

const run = async (): Promise<void> => {
  if (process.env.NODE_ENV !== 'production') dns.setServers(['8.8.8.8', '1.1.1.1']);
  const { dimensions, model } = getHuggingFaceConfig();
  const result = await new PineconeVectorStore().ensureIndex(dimensions, {
    region: process.env.PINECONE_REGION?.trim() || undefined,
  });
  console.log(`Index ${getPineconeIndexName()} (${dimensions}-d cosine for ${model}): ${result}`);
};

run().catch((error: unknown) => {
  console.error('Vector index setup failed:', error instanceof Error ? error.message : 'unknown error');
  process.exit(1);
});
