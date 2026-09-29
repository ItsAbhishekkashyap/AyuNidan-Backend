import { Pinecone } from '@pinecone-database/pinecone';

/**
 * Vector storage/query abstraction over Pinecone (cosine metric). Embeddings are
 * produced locally (see embeddings.ts) — this layer only stores and searches vectors.
 * Tests and the offline evaluation harness substitute the in-memory implementation.
 *
 * Knowledge domains are logically separated by namespace AND a `domain` metadata field:
 *   - patient documents: one namespace per user (`user-<userId>`), never shared
 *   - verified medical reference KB: `medical-reference`, shared read-only reference data
 *   - medical terminology (NLM MedlinePlus + MeSH definitions/synonyms): `medical-terminology` — explains what a term means; never guidance, never patient data
 *   - explainer glossary: `glossary`
 */

export type MetadataValue = string | number | boolean | string[];

export interface VectorRecord {
  id: string;
  values: number[];
  metadata: Record<string, MetadataValue>;
}

export interface VectorMatch {
  id: string;
  score: number;
  metadata?: Record<string, unknown>;
}

export interface VectorQuery {
  vector: number[];
  topK: number;
  /** Pinecone-style metadata filter, e.g. { userId: { $eq: 'x' }, documentId: { $in: [...] } }. */
  filter?: Record<string, unknown>;
}

export interface VectorStore {
  upsert(namespace: string, records: VectorRecord[]): Promise<void>;
  query(namespace: string, query: VectorQuery): Promise<VectorMatch[]>;
  deleteIds(namespace: string, ids: string[]): Promise<void>;
}

export class VectorStoreError extends Error {
  constructor(
    message: string,
    readonly category: 'embedding_failure' | 'vector_db_failure',
    readonly cause?: unknown
  ) {
    super(message);
    this.name = 'VectorStoreError';
  }
}

export const PATIENT_DOMAIN = 'patient';
export const REFERENCE_DOMAIN = 'reference';
export const REFERENCE_NAMESPACE = 'medical-reference';
export const TERMINOLOGY_DOMAIN = 'terminology';
export const TERMINOLOGY_NAMESPACE = 'medical-terminology';
export const GLOSSARY_NAMESPACE = 'glossary';

const envInt = (name: string, fallback: number): number => {
  const value = Number.parseInt(process.env[name] ?? '', 10);
  return Number.isFinite(value) && value >= 0 ? value : fallback;
};

/** Index for the local bge-small (384-d, cosine) vector space. */
export const getPineconeIndexName = (): string => process.env.PINECONE_INDEX?.trim() || 'ayunidan-bge-small-384';

const UPSERT_BATCH = 100;

export class PineconeVectorStore implements VectorStore {
  private client: Pinecone | null = null;

  constructor(private readonly indexName = getPineconeIndexName()) {}

  getClient(): Pinecone {
    if (!this.client) {
      const apiKey = process.env.PINECONE_API_KEY?.trim();
      if (!apiKey) throw new VectorStoreError('PINECONE_API_KEY is not configured', 'vector_db_failure');
      const timeoutMs = envInt('PINECONE_TIMEOUT_MS', 15_000);
      this.client = new Pinecone({
        apiKey,
        // Bounded: one retry layer (the client's), each request capped by a timeout.
        maxRetries: envInt('PINECONE_MAX_RETRIES', 1),
        fetchApi: ((input: Parameters<typeof fetch>[0], init?: RequestInit) => {
          const signals = [AbortSignal.timeout(timeoutMs), ...(init?.signal ? [init.signal] : [])];
          return fetch(input, { ...init, signal: AbortSignal.any(signals) });
        }) as typeof fetch,
      });
    }
    return this.client;
  }

  private index(namespace: string) {
    return this.getClient().index(this.indexName).namespace(namespace);
  }

  /** Creates the serverless cosine index if missing (used by `npm run vector:setup`). */
  async ensureIndex(dimension: number, options: { cloud?: 'aws' | 'gcp' | 'azure'; region?: string } = {}): Promise<'exists' | 'created'> {
    const existing = await this.getClient().listIndexes();
    const found = existing.indexes?.find((i) => i.name === this.indexName);
    if (found) {
      if (found.dimension !== dimension || found.metric !== 'cosine') {
        throw new VectorStoreError(
          `Index ${this.indexName} exists with dimension ${found.dimension}/${found.metric}; expected ${dimension}/cosine`,
          'vector_db_failure'
        );
      }
      return 'exists';
    }
    await this.getClient().createIndex({
      name: this.indexName,
      dimension,
      metric: 'cosine',
      spec: { serverless: { cloud: options.cloud ?? 'aws', region: options.region ?? 'us-east-1' } },
      waitUntilReady: true,
    });
    return 'created';
  }

  async upsert(namespace: string, records: VectorRecord[]): Promise<void> {
    try {
      for (let i = 0; i < records.length; i += UPSERT_BATCH) {
        await this.index(namespace).upsert(records.slice(i, i + UPSERT_BATCH));
      }
    } catch (error) {
      if (error instanceof VectorStoreError) throw error;
      throw new VectorStoreError('Vector upsert failed', 'vector_db_failure', error);
    }
  }

  async query(namespace: string, query: VectorQuery): Promise<VectorMatch[]> {
    try {
      const response = await this.index(namespace).query({
        vector: query.vector,
        topK: query.topK,
        includeMetadata: true,
        ...(query.filter ? { filter: query.filter } : {}),
      });
      return (response.matches ?? []).map((m) => ({ id: m.id, score: m.score ?? 0, metadata: m.metadata }));
    } catch (error) {
      if (error instanceof VectorStoreError) throw error;
      throw new VectorStoreError('Vector query failed', 'vector_db_failure', error);
    }
  }

  async deleteIds(namespace: string, ids: string[]): Promise<void> {
    if (ids.length === 0) return;
    try {
      for (let i = 0; i < ids.length; i += 1000) {
        await this.index(namespace).deleteMany(ids.slice(i, i + 1000));
      }
    } catch (error) {
      if (error instanceof VectorStoreError) throw error;
      throw new VectorStoreError('Vector delete failed', 'vector_db_failure', error);
    }
  }
}

let active: VectorStore | null = null;

export const getVectorStore = (): VectorStore => {
  if (!active) active = new PineconeVectorStore();
  return active;
};

/** Swap the implementation (tests / offline evaluation). Pass null to restore Pinecone. */
export const setVectorStore = (store: VectorStore | null): void => {
  active = store;
};
