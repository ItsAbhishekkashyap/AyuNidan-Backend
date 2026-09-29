/**
 * Ingests the locally supplied NLM terminology datasets (MedlinePlus Health Topics, MeSH descriptors)
 * into the separate `medical-terminology` vector namespace, using the LOCAL bge-small embeddings.
 * The source ZIPs are only read — never modified, moved or uploaded anywhere.
 *
 * Usage:
 *   npm run terminology:ingest -- --dataset=medlineplus|mesh|all [--dry-run] [--limit=N] [--force]
 *
 * --dry-run  parse + report only (no embeddings, no vector writes)
 * --limit    ingest only the first N records (smoke test)
 * --force    ignore the manifest and re-embed (ids are deterministic, so this overwrites, never duplicates)
 */
import 'dotenv/config';
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { parseMedlinePlus, parseMesh, readMedlinePlusInfo, readMeshInfo, newStats } from '../terminology/parsers';
import type { TerminologyDataset, TerminologyRecord } from '../terminology/types';
import { ingestTerminologyRecords } from '../services/terminology.service';
import { getEmbeddings } from '../rag/embeddings';

const ROOT = path.resolve(__dirname, '..', '..', 'medical-reference');
const SOURCE_DIR = path.join(ROOT, 'terminology');
const PROCESSED = path.join(ROOT, 'processed');

const arg = (name: string): string | undefined => process.argv.find((a) => a.startsWith(`--${name}=`))?.split('=')[1];
const flag = (name: string): boolean => process.argv.includes(`--${name}`);

const zipFor = (dataset: TerminologyDataset): string => {
  const dir = path.join(SOURCE_DIR, dataset);
  const file = fs.readdirSync(dir).find((f) => f.toLowerCase().endsWith('.zip'));
  if (!file) throw new Error(`No .zip found in ${dir}`);
  return path.join(dir, file);
};

const fileSha256 = (file: string): Promise<string> =>
  new Promise((resolve, reject) => {
    const hash = crypto.createHash('sha256');
    fs.createReadStream(file)
      .on('data', (chunk) => hash.update(chunk))
      .on('end', () => resolve(hash.digest('hex')))
      .on('error', reject);
  });

interface Manifest {
  dataset: string;
  sha256: string;
  embeddingSpace: string;
  version: string;
  publicationDate?: string;
  sourceFile: string;
  recordCount: number;
  skipped: Record<string, number>;
  recordIds: string[];
  ingestedAt: string;
}

const run = async (): Promise<void> => {
  const which = arg('dataset') ?? 'all';
  const datasets: TerminologyDataset[] = which === 'all' ? ['medlineplus', 'mesh'] : [which as TerminologyDataset];
  const dryRun = flag('dry-run');
  const force = flag('force');
  const limit = arg('limit') ? Number(arg('limit')) : undefined;
  fs.mkdirSync(PROCESSED, { recursive: true });

  for (const dataset of datasets) {
    if (dataset !== 'medlineplus' && dataset !== 'mesh') throw new Error(`Unknown dataset "${dataset}"`);
    const zip = zipFor(dataset);
    const { info } = dataset === 'mesh' ? await readMeshInfo(zip) : await readMedlinePlusInfo(zip);
    const sha = await fileSha256(zip);
    const manifestPath = path.join(PROCESSED, `terminology-${dataset}.json`);
    const previous: Manifest | undefined = fs.existsSync(manifestPath) ? (JSON.parse(fs.readFileSync(manifestPath, 'utf8')) as Manifest) : undefined;
    const spaceId = getEmbeddings().spaceId;

    if (!dryRun && !force && !limit && previous && previous.sha256 === sha && previous.embeddingSpace === spaceId) {
      console.log(`UNCHANGED ${dataset} (${previous.recordCount} records, ${info.version}) — same dataset file and embedding space.`);
      continue;
    }

    const stats = newStats();
    const records: TerminologyRecord[] = [];
    for await (const record of dataset === 'mesh' ? parseMesh(zip, stats) : parseMedlinePlus(zip, stats)) {
      records.push(record);
      if (limit && records.length >= limit) break;
    }
    console.log(`PARSED ${dataset}: ${stats.recordsParsed} records from ${stats.blocksSeen} blocks; skipped ${JSON.stringify(stats.skipped)}; duplicates ${stats.duplicatesDropped}; version ${info.version}`);
    if (dryRun) continue;

    const started = Date.now();
    const result = await ingestTerminologyRecords(info, records, {
      previousIds: limit ? [] : previous?.recordIds,
      onProgress: (done, total) => {
        if (done % 1024 < 64 || done === total) console.log(`  ${dataset}: ${done}/${total} (${Math.round((Date.now() - started) / 1000)}s)`);
      },
    });
    console.log(`INDEXED ${dataset}: ${result.upserted} vectors upserted, ${result.deletedStale} stale removed, space ${result.embeddingSpace}`);
    if (!limit) {
      const manifest: Manifest = {
        dataset,
        sha256: sha,
        embeddingSpace: result.embeddingSpace,
        version: info.version,
        ...(info.publicationDate ? { publicationDate: info.publicationDate } : {}),
        sourceFile: info.sourceFile,
        recordCount: result.records,
        skipped: stats.skipped,
        recordIds: result.recordIds,
        ingestedAt: new Date().toISOString(),
      };
      fs.writeFileSync(manifestPath, JSON.stringify(manifest));
    }
  }
};

run().catch((error: unknown) => {
  console.error('Terminology ingestion failed:', error instanceof Error ? `${error.name}: ${error.message}` : 'UnknownError');
  process.exit(1);
});
