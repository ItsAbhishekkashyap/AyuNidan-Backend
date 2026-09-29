/**
 * One-off maintenance script.
 *
 * Before V2, a failed AI summary was persisted as riskLevel "low" / riskScore 0 with a
 * placeholder summary. This marks those records as `failed` and removes the fabricated
 * risk fields so they no longer look like legitimate low-risk results.
 *
 * Usage: npm run migrate:legacy-failures            (dry run: counts only)
 *        npm run migrate:legacy-failures -- --apply
 */
import 'dotenv/config';
import mongoose from 'mongoose';
import { getConfig } from '../config/env';
import { Consultation } from '../models/Consultation';

const LEGACY_PLACEHOLDER = 'Unable to generate clinical summary';

const run = async (): Promise<void> => {
  const apply = process.argv.includes('--apply');
  await mongoose.connect(getConfig().MONGODB_URI, { serverSelectionTimeoutMS: 5000 });

  const filter: mongoose.QueryFilter<Record<string, unknown>> = {
    status: { $exists: false },
    summary: { $regex: LEGACY_PLACEHOLDER },
    riskLevel: 'low',
    riskScore: 0,
  };
  const count = await Consultation.countDocuments(filter);
  console.log(`Legacy failed-summary records found: ${count}`);

  if (apply && count > 0) {
    const result = await Consultation.updateMany(filter, {
      $set: { status: 'failed', failure: { stage: 'summary', category: 'provider_failure', occurredAt: new Date() } },
      $unset: { summary: 1, riskLevel: 1, riskScore: 1 },
    });
    console.log(`Marked as failed: ${result.modifiedCount}`);
  } else if (!apply) {
    console.log('Dry run. Re-run with --apply to update.');
  }
  await mongoose.disconnect();
};

run().catch(async (error: unknown) => {
  console.error('Migration failed:', error instanceof Error ? error.name : 'UnknownError');
  await mongoose.disconnect().catch(() => undefined);
  process.exit(1);
});
