/**
 * Ingests authorised reference documents from medical-reference/sources into the shared
 * verified-reference KB. Each document needs a `<name>.meta.json` sidecar (see README).
 * Usage: npm run kb:ingest
 */
import 'dotenv/config';
import fs from 'node:fs';
import path from 'node:path';
import dns from 'node:dns';
import mongoose from 'mongoose';
import { getConfig } from '../config/env';
import { ingestReferenceSource } from '../services/reference.service';
import { ReferenceSource } from '../models/ReferenceSource';

const ROOT = path.resolve(__dirname, '..', '..', 'medical-reference');
// Sidecar-described documents: operator-supplied sources and the real clinical guideline PDFs.
const SOURCE_DIRS = [path.join(ROOT, 'sources'), path.join(ROOT, 'clinical-guidelines')];
const PROCESSED = path.join(ROOT, 'processed');
const SUPPORTED = new Set(['.pdf', '.md', '.txt']);

const run = async (): Promise<void> => {
  if (getConfig().NODE_ENV !== 'production') dns.setServers(['8.8.8.8', '1.1.1.1']);
  await mongoose.connect(getConfig().MONGODB_URI, { serverSelectionTimeoutMS: 8000 });
  fs.mkdirSync(PROCESSED, { recursive: true });

  const files = SOURCE_DIRS.filter((d) => fs.existsSync(d)).flatMap((dir) =>
    fs.readdirSync(dir).filter((f) => SUPPORTED.has(path.extname(f).toLowerCase())).map((f) => ({ dir, file: f }))
  );
  if (files.length === 0) console.log('No reference documents found in medical-reference/sources or clinical-guidelines (see README).');

  for (const { dir, file } of files) {
    const metaPath = path.join(dir, `${path.basename(file, path.extname(file))}.meta.json`);
    if (!fs.existsSync(metaPath)) {
      console.log(`SKIP ${file}: missing sidecar ${path.basename(metaPath)} (provenance is required).`);
      continue;
    }
    try {
      const meta = JSON.parse(fs.readFileSync(metaPath, 'utf8')) as unknown;
      const result = await ingestReferenceSource(meta, fs.readFileSync(path.join(dir, file)), file, undefined, { force: process.argv.includes('--force') });
      console.log(`${result.status.toUpperCase()} ${file} → ${result.sourceId} (${result.chunkCount} chunks)${result.failureCategory ? ` [${result.failureCategory}]` : ''}`);
      const record = await ReferenceSource.findOne({ sourceId: result.sourceId }).lean().exec();
      fs.writeFileSync(path.join(PROCESSED, `${result.sourceId}.json`), JSON.stringify({ result, record }, null, 2));
    } catch (error) {
      console.log(`FAILED ${file}: ${error instanceof Error ? error.message : 'unknown error'}`);
    }
  }
  await mongoose.disconnect();
};

run().catch(async (error: unknown) => {
  console.error('Reference ingestion failed:', error instanceof Error ? error.name : 'UnknownError');
  await mongoose.disconnect().catch(() => undefined);
  process.exit(1);
});
