import { cosineSimilarity } from './embeddings';
import { VectorStoreError, type VectorMatch, type VectorQuery, type VectorRecord, type VectorStore } from './vectorStore';

/**
 * Deterministic in-memory vector store (cosine similarity), used by unit tests and
 * the offline evaluation harness. Supports the Pinecone filter subset we use ($eq, $in).
 */
export class InMemoryVectorStore implements VectorStore {
  private readonly namespaces = new Map<string, Map<string, VectorRecord>>();
  /** Test hooks to simulate outages. */
  failQuery = false;
  failUpsert = false;

  async upsert(namespace: string, records: VectorRecord[]): Promise<void> {
    if (this.failUpsert) throw new VectorStoreError('simulated upsert outage', 'vector_db_failure');
    const ns = this.namespaces.get(namespace) ?? new Map<string, VectorRecord>();
    for (const record of records) ns.set(record.id, record);
    this.namespaces.set(namespace, ns);
  }

  private matchesFilter(metadata: Record<string, unknown>, filter?: Record<string, unknown>): boolean {
    if (!filter) return true;
    return Object.entries(filter).every(([key, condition]) => {
      const value = metadata[key];
      if (condition && typeof condition === 'object') {
        const c = condition as { $eq?: unknown; $in?: unknown[] };
        if ('$eq' in c) return value === c.$eq;
        if ('$in' in c) {
          const wanted = c.$in ?? [];
          // Pinecone semantics: a list-valued metadata field matches when ANY element is in $in.
          return Array.isArray(value) ? value.some((v) => wanted.includes(v)) : wanted.includes(value);
        }
      }
      return value === condition;
    });
  }

  async query(namespace: string, query: VectorQuery): Promise<VectorMatch[]> {
    if (this.failQuery) throw new VectorStoreError('simulated query outage', 'vector_db_failure');
    const ns = this.namespaces.get(namespace);
    if (!ns) return [];
    return [...ns.values()]
      .filter((r) => this.matchesFilter(r.metadata, query.filter) && r.values.length === query.vector.length)
      .map((r) => ({ id: r.id, score: cosineSimilarity(r.values, query.vector), metadata: { ...r.metadata } }))
      .sort((a, b) => b.score - a.score || a.id.localeCompare(b.id))
      .slice(0, query.topK);
  }

  async deleteIds(namespace: string, ids: string[]): Promise<void> {
    const ns = this.namespaces.get(namespace);
    for (const id of ids) ns?.delete(id);
  }

  /** Test helper: direct access to stored records. */
  records(namespace: string): VectorRecord[] {
    return [...(this.namespaces.get(namespace)?.values() ?? [])];
  }

  /** Test helper: inject a raw record (e.g. with malformed metadata). */
  inject(namespace: string, record: VectorRecord): void {
    const ns = this.namespaces.get(namespace) ?? new Map<string, VectorRecord>();
    ns.set(record.id, record);
    this.namespaces.set(namespace, ns);
  }

  namespaceNames(): string[] {
    return [...this.namespaces.keys()];
  }
}
