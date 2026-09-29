import mongoose, { Schema, Document as MongooseDocument } from 'mongoose';

/**
 * An uploaded clinical document indexed for retrieval.
 * Lifecycle: indexing → indexed | failed. Deleting removes the record and its vectors.
 *
 * Chunk text lives only in the vector store (per-user namespace); this record is
 * the authority for ownership and existence — retrieval ignores vectors whose
 * document record is missing or not `indexed`.
 */
export const DOCUMENT_STATUSES = ['indexing', 'indexed', 'failed'] as const;
export type DocumentStatus = (typeof DOCUMENT_STATUSES)[number];

export const DOCUMENT_SOURCES = ['pdf_text', 'text_file', 'ai_transcription'] as const;

export interface StoredTableRow {
  test: string;
  result: string;
  unit?: string;
  referenceRange?: string;
  flag?: string;
  date?: string;
  page?: number;
  tableId: string;
  rowIndex: number;
}
export type DocumentSource = (typeof DOCUMENT_SOURCES)[number];

export interface IClinicalDocument extends MongooseDocument {
  userId: mongoose.Types.ObjectId;
  consultationId?: mongoose.Types.ObjectId;
  filename: string;
  mimeType: string;
  sizeBytes: number;
  sha256: string;
  source: DocumentSource;
  status: DocumentStatus;
  chunkCount: number;
  pageCount?: number;
  textChars: number;
  truncated: boolean;
  /** Embedding space (model@dims) the chunks were indexed with; retrieval requires a match. */
  embeddingModel: string;
  /** Deterministically extracted table rows (row-intact, with page provenance). */
  tables: StoredTableRow[];
  pages: { page: number; status: string; imageCount: number; tableCount: number }[];
  notes: string[];
  failure?: { category: string; occurredAt: Date };
  createdAt: Date;
  updatedAt: Date;
}

const ClinicalDocumentSchema = new Schema<IClinicalDocument>(
  {
    userId: { type: Schema.Types.ObjectId, ref: 'User', required: true },
    consultationId: { type: Schema.Types.ObjectId, ref: 'Consultation' },
    filename: { type: String, required: true, maxlength: 255 },
    mimeType: { type: String, required: true },
    sizeBytes: { type: Number, required: true, min: 0 },
    sha256: { type: String, required: true },
    source: { type: String, enum: DOCUMENT_SOURCES, required: true },
    status: { type: String, enum: DOCUMENT_STATUSES, required: true },
    chunkCount: { type: Number, default: 0 },
    pageCount: { type: Number },
    textChars: { type: Number, default: 0 },
    truncated: { type: Boolean, default: false },
    embeddingModel: { type: String, required: true },
    tables: {
      type: [
        {
          _id: false,
          test: String,
          result: String,
          unit: String,
          referenceRange: String,
          flag: String,
          date: String,
          page: Number,
          tableId: String,
          rowIndex: Number,
        },
      ],
      default: [],
      validate: [(rows: unknown[]) => rows.length <= 500, 'Too many table rows'],
    },
    pages: { type: [{ _id: false, page: Number, status: String, imageCount: Number, tableCount: Number }], default: [] },
    notes: { type: [{ type: String, maxlength: 300 }], default: [] },
    failure: {
      category: { type: String },
      occurredAt: { type: Date },
    },
  },
  { timestamps: true }
);

ClinicalDocumentSchema.index({ userId: 1, createdAt: -1 });
ClinicalDocumentSchema.index({ userId: 1, consultationId: 1 });
ClinicalDocumentSchema.index({ userId: 1, sha256: 1, source: 1 });

export const ClinicalDocument = mongoose.model<IClinicalDocument>('Document', ClinicalDocumentSchema);
