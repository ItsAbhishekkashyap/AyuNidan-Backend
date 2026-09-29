import { beforeEach, vi } from 'vitest';

process.env.NODE_ENV = 'test';
process.env.JWT_SECRET = 'test-jwt-secret-that-is-at-least-32-characters-long';
process.env.MONGODB_URI = 'mongodb://127.0.0.1:27017/ayunidan-test';
// Synthetic fixture sources may be served in tests only.
process.env.RAG_INCLUDE_SYNTHETIC_REFERENCES = 'true';

// No test talks to MongoDB, Pinecone, Hugging Face or an LLM by default: models are
// in-memory fakes, the vector store is in-memory, embeddings are deterministic hashing.
vi.mock('../src/models/Consultation', async () => ({
  ...(await vi.importActual<object>('../src/models/Consultation')),
  Consultation: (await import('./helpers/fakeModels')).FakeConsultation,
}));
vi.mock('../src/models/User', async () => ({ User: (await import('./helpers/fakeModels')).FakeUser }));
vi.mock('../src/models/Document', async () => ({
  ...(await vi.importActual<object>('../src/models/Document')),
  ClinicalDocument: (await import('./helpers/fakeModels')).FakeDocument,
}));
vi.mock('../src/models/ReferenceSource', async () => ({
  ...(await vi.importActual<object>('../src/models/ReferenceSource')),
  ReferenceSource: (await import('./helpers/fakeModels')).FakeReferenceSource,
}));

beforeEach(async () => {
  const { setVectorStore } = await import('../src/rag/vectorStore');
  const { InMemoryVectorStore } = await import('../src/rag/inMemoryVectorStore');
  const { setEmbeddings, HashingEmbeddings } = await import('../src/rag/embeddings');
  const { resetCircuitBreakers } = await import('../src/services/ai.service');
  const { invalidateReferenceRegistry } = await import('../src/services/reference.service');
  setVectorStore(new InMemoryVectorStore());
  setEmbeddings(new HashingEmbeddings());
  resetCircuitBreakers();
  invalidateReferenceRegistry();
});
