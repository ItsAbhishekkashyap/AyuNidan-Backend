/**
 * Demo helpers (FICTIONAL / SYNTHETIC data only — never medical guidance).
 *
 *   npm run demo:pdf              writes demo-assets/fictional-cbc-report.pdf (upload this in the UI)
 *   npm run demo:refs             ingests the SYNTHETIC reference passages from eval/synthetic-dataset.json as
 *                                 sources of type "synthetic_test" (sourceIds prefixed "demo-")
 *   npm run demo:refs -- --remove deletes those synthetic sources and their vectors
 *
 * Synthetic sources are only served when the API runs with RAG_INCLUDE_SYNTHETIC_REFERENCES=true and are
 * badged "synthetic test source" in the UI. Real, authorised references go through `npm run kb:ingest`.
 */
import 'dotenv/config';
import fs from 'node:fs';
import path from 'node:path';
import dns from 'node:dns';
import mongoose from 'mongoose';
import { buildPdf, LAB_COLUMNS } from '../eval/pdfBuilder';
import { loadDataset } from '../eval/dataset';
import { getConfig } from '../config/env';
import { ingestReferenceSource } from '../services/reference.service';
import { ReferenceSource } from '../models/ReferenceSource';
import { getVectorStore, REFERENCE_NAMESPACE } from '../rag/vectorStore';

const ROOT = path.resolve(__dirname, '..', '..');

const writePdf = (): void => {
  const dir = path.join(ROOT, 'demo-assets');
  fs.mkdirSync(dir, { recursive: true });
  const pdf = buildPdf([
    { lines: ['FICTIONAL LAB REPORT - NOT A REAL PATIENT', 'Patient: Zorblax Quendi (fictional), Age: 51, Sex: M', 'Complaint: tiredness for two weeks'] },
    {
      lines: ['Complete blood count (fictional)'],
      table: {
        columns: LAB_COLUMNS,
        rows: [
          ['Test', 'Result', 'Unit', 'Reference Range', 'Flag'],
          ['Hemoglobin', '10.2', 'g/dL', '13-17', 'Low'],
          ['Platelets', '450', 'x10^9/L', '150-400', 'High'],
        ],
      },
    },
  ]);
  const file = path.join(dir, 'fictional-cbc-report.pdf');
  fs.writeFileSync(file, pdf);
  console.log(`Wrote ${path.relative(ROOT, file)}`);
};

const withDb = async (fn: () => Promise<void>): Promise<void> => {
  if (getConfig().NODE_ENV !== 'production') dns.setServers(['8.8.8.8', '1.1.1.1']);
  await mongoose.connect(getConfig().MONGODB_URI, { serverSelectionTimeoutMS: 8000 });
  try {
    await fn();
  } finally {
    await mongoose.disconnect();
  }
};

const ingestSynthetic = async (): Promise<void> => {
  const corpus = loadDataset().referenceCorpus;
  if (!corpus) throw new Error('No synthetic reference corpus in the dataset');
  await withDb(async () => {
    for (const source of corpus.sources) {
      const result = await ingestReferenceSource(
        {
          sourceId: `demo-${source.sourceId}`,
          title: source.title,
          organization: 'AyuNidan Synthetic Test Corpus (fictional)',
          sourceType: 'synthetic_test',
          authorization: 'Synthetic demo fixture written for this project; not medical guidance',
        },
        Buffer.from(source.text),
        `${source.sourceId}.md`
      );
      console.log(`${result.status.toUpperCase()} demo-${source.sourceId} (${result.chunkCount} chunks)`);
    }
    console.log('Start the API with RAG_INCLUDE_SYNTHETIC_REFERENCES=true to serve them. Remove with: npm run demo:refs -- --remove');
  });
};

const removeSynthetic = async (): Promise<void> => {
  await withDb(async () => {
    const sources = await ReferenceSource.find({ sourceId: /^demo-/, sourceType: 'synthetic_test' }).lean().exec();
    for (const s of sources) {
      await getVectorStore().deleteIds(REFERENCE_NAMESPACE, Array.from({ length: s.chunkCount }, (_, i) => `${s.sourceId}#${i}`));
      await ReferenceSource.deleteOne({ sourceId: s.sourceId }).exec();
      console.log(`REMOVED ${s.sourceId}`);
    }
    if (sources.length === 0) console.log('No synthetic demo sources found.');
  });
};

const [command] = process.argv.slice(2);
const main = async (): Promise<void> => {
  if (command === 'pdf') return writePdf();
  if (command === 'refs') return process.argv.includes('--remove') ? removeSynthetic() : ingestSynthetic();
  console.log('Usage: demo-assets <pdf|refs> [--remove]');
};

main().catch((error: unknown) => {
  console.error('Demo helper failed:', error instanceof Error ? error.message : 'unknown error');
  process.exitCode = 1;
});
