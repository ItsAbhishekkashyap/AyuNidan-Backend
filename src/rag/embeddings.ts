import path from 'node:path';
import { Embeddings } from '@langchain/core/embeddings';

/**
 * Embedding models behind LangChain's `Embeddings` abstraction.
 *
 * Production: a LOCAL open-source Hugging Face model (default BAAI bge-small-en-v1.5,
 * ONNX build `Xenova/bge-small-en-v1.5`, 384 dims) run in-process via transformers.js.
 * No embedding traffic goes to Gemini or any paid API.
 *
 * The same instance embeds documents at indexing time and queries at retrieval time,
 * and every indexed record stores `embeddingSpace` so vectors from a different model
 * are never compared against each other.
 */

export class EmbeddingError extends Error {
  readonly category = 'embedding_failure' as const;
  constructor(message: string, readonly cause?: unknown) {
    super(message);
    this.name = 'EmbeddingError';
  }
}

export interface EmbeddingModel extends Embeddings {
  /** Stable identifier of the vector space, e.g. "Xenova/bge-small-en-v1.5@384". */
  readonly spaceId: string;
  readonly dimensions: number;
}

const envValue = (name: string): string | undefined => process.env[name]?.trim() || undefined;

/* ───────────────────────── Local Hugging Face ───────────────────────── */

type FeatureExtractor = (texts: string[], options: { pooling: 'cls' | 'mean'; normalize: boolean }) => Promise<{ tolist(): number[][] }>;

export interface HuggingFaceConfig {
  model: string;
  dimensions: number;
  pooling: 'cls' | 'mean';
  /** bge v1.5 recommends this instruction for short queries against passages. */
  queryPrefix: string;
  dtype: 'fp32' | 'q8';
  batchSize: number;
  cacheDir: string;
}

export const getHuggingFaceConfig = (): HuggingFaceConfig => ({
  model: envValue('HF_EMBEDDING_MODEL') ?? 'Xenova/bge-small-en-v1.5',
  dimensions: Number(envValue('HF_EMBEDDING_DIMENSIONS') ?? 384),
  pooling: envValue('HF_EMBEDDING_POOLING') === 'mean' ? 'mean' : 'cls',
  queryPrefix: envValue('HF_EMBEDDING_QUERY_PREFIX') ?? 'Represent this sentence for searching relevant passages: ',
  dtype: envValue('HF_EMBEDDING_DTYPE') === 'fp32' ? 'fp32' : 'q8',
  batchSize: Number(envValue('HF_EMBEDDING_BATCH_SIZE') ?? 32),
  cacheDir: envValue('HF_CACHE_DIR') ?? path.resolve(__dirname, '..', '..', '.cache', 'hf-models'),
});

export class LocalHuggingFaceEmbeddings extends Embeddings implements EmbeddingModel {
  readonly spaceId: string;
  readonly dimensions: number;
  private extractor: Promise<FeatureExtractor> | null = null;

  constructor(private readonly config: HuggingFaceConfig = getHuggingFaceConfig()) {
    super({ maxConcurrency: 1 });
    this.dimensions = config.dimensions;
    this.spaceId = `${config.model}@${config.dimensions}`;
  }

  /** Loads the model once per process (first call downloads it into the local cache). */
  private getExtractor(): Promise<FeatureExtractor> {
    if (!this.extractor) {
      this.extractor = (async () => {
        const transformers = await import('@huggingface/transformers');
        transformers.env.cacheDir = this.config.cacheDir;
        const pipe = await transformers.pipeline('feature-extraction', this.config.model, { dtype: this.config.dtype });
        return pipe as unknown as FeatureExtractor;
      })().catch((error: unknown) => {
        this.extractor = null; // allow a later retry
        throw new EmbeddingError('Failed to load local embedding model', error);
      });
    }
    return this.extractor;
  }

  private async embed(texts: string[]): Promise<number[][]> {
    const extractor = await this.getExtractor();
    const vectors: number[][] = [];
    try {
      for (let i = 0; i < texts.length; i += this.config.batchSize) {
        const output = await extractor(texts.slice(i, i + this.config.batchSize), { pooling: this.config.pooling, normalize: true });
        vectors.push(...output.tolist());
      }
    } catch (error) {
      throw new EmbeddingError('Local embedding inference failed', error);
    }
    if (vectors.some((v) => v.length !== this.dimensions)) {
      throw new EmbeddingError(`Embedding dimension mismatch (expected ${this.dimensions})`);
    }
    return vectors;
  }

  async embedDocuments(documents: string[]): Promise<number[][]> {
    return documents.length === 0 ? [] : this.embed(documents);
  }

  async embedQuery(query: string): Promise<number[]> {
    const [vector] = await this.embed([`${this.config.queryPrefix}${query}`]);
    return vector;
  }
}

/* ───────────────────────── Deterministic hashing (tests / offline eval) ───────────────────────── */

/**
 * Hashed bag-of-words embedder. Deterministic, dependency-free, lexical only.
 * Used by unit tests and the fast offline evaluation mode. Its similarity scale
 * differs from bge's, so thresholds are always evaluated per embedding space.
 */
export class HashingEmbeddings extends Embeddings implements EmbeddingModel {
  readonly spaceId: string;
  failNext = false;

  constructor(readonly dimensions = 256) {
    super({});
    this.spaceId = `hashed-bow@${dimensions}`;
  }

  private tokenize(text: string): string[] {
    return text.toLowerCase().match(/[a-z0-9]+/g)?.filter((t) => t.length > 1) ?? [];
  }

  private hash(token: string): number {
    let h = 2166136261;
    for (let i = 0; i < token.length; i++) {
      h ^= token.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }
    return h >>> 0;
  }

  embedOne(text: string): number[] {
    const v = new Array<number>(this.dimensions).fill(0);
    for (const token of this.tokenize(text)) {
      const h = this.hash(token);
      v[h % this.dimensions] += h & 1 ? 1 : -1;
    }
    const norm = Math.hypot(...v) || 1;
    return v.map((x) => x / norm);
  }

  private check(): void {
    if (this.failNext) {
      this.failNext = false;
      throw new EmbeddingError('simulated embedding outage');
    }
  }

  async embedDocuments(documents: string[]): Promise<number[][]> {
    this.check();
    return documents.map((d) => this.embedOne(d));
  }

  async embedQuery(query: string): Promise<number[]> {
    this.check();
    return this.embedOne(query);
  }
}

/* ───────────────────────── Registry ───────────────────────── */

let active: EmbeddingModel | null = null;

export const getEmbeddings = (): EmbeddingModel => {
  if (!active) active = new LocalHuggingFaceEmbeddings();
  return active;
};

/** Swap the embedding model (tests / offline evaluation). Pass null to restore the default. */
export const setEmbeddings = (model: EmbeddingModel | null): void => {
  active = model;
};

/** Cosine similarity for already-normalised or raw vectors. */
export const cosineSimilarity = (a: number[], b: number[]): number => {
  if (a.length !== b.length) throw new EmbeddingError('Cannot compare vectors of different dimensions');
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return na === 0 || nb === 0 ? 0 : dot / Math.sqrt(na * nb);
};
