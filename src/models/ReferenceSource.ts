import mongoose, { Schema, Document as MongooseDocument } from 'mongoose';

/**
 * Registry of verified medical reference sources ingested into the shared reference KB.
 * Only sources listed here with status `indexed` (and the current embedding space) are
 * retrievable — retiring a source removes it from answers even before vectors are purged.
 * Every provenance field is copied from the operator-supplied metadata; none is inferred.
 */
export const REFERENCE_SOURCE_TYPES = [
  'clinical_guideline',
  'public_health_guidance',
  'medical_reference',
  'professional_society',
  'synthetic_test',
  'other',
] as const;
export type ReferenceSourceType = (typeof REFERENCE_SOURCE_TYPES)[number];

export interface IReferenceSource extends MongooseDocument {
  sourceId: string;
  title: string;
  organization: string;
  sourceType: ReferenceSourceType;
  authorization: string;
  publicationDate?: string;
  version?: string;
  url?: string;
  medicalTopics: string[];
  filename: string;
  sha256: string;
  status: 'indexed' | 'failed' | 'retired';
  chunkCount: number;
  pageCount?: number;
  embeddingSpace: string;
  failure?: { category: string; occurredAt: Date };
  createdAt: Date;
  updatedAt: Date;
}

const ReferenceSourceSchema = new Schema<IReferenceSource>(
  {
    sourceId: { type: String, required: true, unique: true, maxlength: 100 },
    title: { type: String, required: true, maxlength: 500 },
    organization: { type: String, required: true, maxlength: 300 },
    sourceType: { type: String, enum: REFERENCE_SOURCE_TYPES, required: true },
    authorization: { type: String, required: true, maxlength: 1000 },
    publicationDate: { type: String, maxlength: 50 },
    version: { type: String, maxlength: 100 },
    url: { type: String, maxlength: 2000 },
    medicalTopics: [{ type: String, maxlength: 100 }],
    filename: { type: String, required: true },
    sha256: { type: String, required: true },
    status: { type: String, enum: ['indexed', 'failed', 'retired'], required: true },
    chunkCount: { type: Number, default: 0 },
    pageCount: { type: Number },
    embeddingSpace: { type: String, required: true },
    failure: { category: { type: String }, occurredAt: { type: Date } },
  },
  { timestamps: true }
);

export const ReferenceSource = mongoose.model<IReferenceSource>('ReferenceSource', ReferenceSourceSchema);
