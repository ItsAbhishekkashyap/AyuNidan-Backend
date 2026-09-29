import mongoose, { Schema, Document } from 'mongoose';
import { LabValue } from '../types';

/**
 * Processing lifecycle:
 *   processing → completed   (validated AI summary + risk stored)
 *   processing → failed      (no clinical conclusion stored; `failure` explains why)
 * Legacy records created before this field existed have no `status` and are treated as completed.
 */
export const CONSULTATION_STATUSES = ['processing', 'completed', 'failed'] as const;
export type ConsultationStatus = (typeof CONSULTATION_STATUSES)[number];

export const FAILURE_STAGES = ['extraction', 'assessment', 'summary', 'persistence'] as const;
export type FailureStage = (typeof FAILURE_STAGES)[number];

export interface IConsultationDocument extends Document {
  userId: mongoose.Types.ObjectId;
  status?: ConsultationStatus;
  patientDetails: {
    name: string;
    age?: number;
    gender?: string;
  };
  symptoms: string[];
  medicines: string[];
  labValues: LabValue[];
  voiceTranscript?: string;
  rawText?: string;
  /** Only present when status is completed (or legacy). Never a placeholder. */
  summary?: string;
  /** insufficient_evidence: the model/evidence could not support low/medium/high. */
  riskLevel?: 'low' | 'medium' | 'high' | 'insufficient_evidence';
  riskScore?: number;
  failure?: {
    stage: FailureStage;
    category: string;
    attempts?: number;
    occurredAt: Date;
  };
  ai?: {
    extraction?: { source: 'client' | 'model'; provider?: string; model?: string; fallbackUsed?: boolean };
    /** Legacy (pre-grounded-assessment) records only; new records use `assessment`. */
    summary?: { provider: string; model: string; fallbackUsed: boolean; riskScoreAdjusted: boolean };
    assessment?: { provider: string; model: string; fallbackUsed: boolean };
  };
  /** Evidence-grounded assessment (key findings, uncertainty, cited patient evidence, medical references used). */
  assessment?: Record<string, unknown>;
  /** Comprehensive structured report from document understanding (Job A). */
  report?: Record<string, unknown>;
  documentIds: mongoose.Types.ObjectId[];
  reportUrls: string[];
  processingTime?: number;
  createdAt: Date;
  updatedAt: Date;
}

const LabValueSchema = new Schema<LabValue>({
  name: { type: String, required: true },
  value: { type: String, required: true },
  unit: { type: String, default: '' },
  normalRange: { type: String },
  isAbnormal: { type: Boolean, default: false },
  flag: { type: String },
  flagSource: { type: String, enum: ['printed_range', 'table', 'model'] },
  date: { type: String },
  source: {
    method: { type: String, enum: ['table', 'text_match', 'model'] },
    filename: { type: String },
    page: { type: Number },
    pages: [{ type: Number }],
    tableId: { type: String },
  },
});

const ConsultationSchema = new Schema<IConsultationDocument>(
  {
    userId: {
      type: Schema.Types.ObjectId,
      ref: 'User',
      required: true,
      index: true,
    },
    status: { type: String, enum: CONSULTATION_STATUSES },
    patientDetails: {
      name: { type: String, default: 'Unknown', trim: true },
      age: { type: Number },
      gender: { type: String, default: 'Unknown', trim: true },
    },
    symptoms: [{ type: String, trim: true }],
    medicines: [{ type: String, trim: true }],
    labValues: [LabValueSchema],
    voiceTranscript: { type: String },
    rawText: { type: String },
    summary: { type: String },
    riskLevel: { type: String, enum: ['low', 'medium', 'high', 'insufficient_evidence'], index: true },
    riskScore: { type: Number, min: 0, max: 100 },
    failure: {
      stage: { type: String, enum: FAILURE_STAGES },
      category: { type: String },
      attempts: { type: Number },
      occurredAt: { type: Date },
    },
    ai: {
      extraction: {
        source: { type: String, enum: ['client', 'model'] },
        provider: { type: String },
        model: { type: String },
        fallbackUsed: { type: Boolean },
      },
      summary: {
        provider: { type: String },
        model: { type: String },
        fallbackUsed: { type: Boolean },
        riskScoreAdjusted: { type: Boolean },
      },
      assessment: {
        provider: { type: String },
        model: { type: String },
        fallbackUsed: { type: Boolean },
      },
    },
    assessment: { type: Schema.Types.Mixed },
    report: { type: Schema.Types.Mixed },
    documentIds: [{ type: Schema.Types.ObjectId, ref: 'Document' }],
    reportUrls: [{ type: String }],
    processingTime: { type: Number },
  },
  {
    timestamps: true,
  }
);

// Added index specifically for the new Patient Name search feature
ConsultationSchema.index({ 'patientDetails.name': 1 });
ConsultationSchema.index({ userId: 1, createdAt: -1 });
ConsultationSchema.index({ userId: 1, riskLevel: 1 });
ConsultationSchema.index({ riskLevel: 1, createdAt: -1 });

ConsultationSchema.index({
  symptoms: 'text',
  medicines: 'text',
  summary: 'text',
});

export const Consultation = mongoose.model<IConsultationDocument>('Consultation', ConsultationSchema);
