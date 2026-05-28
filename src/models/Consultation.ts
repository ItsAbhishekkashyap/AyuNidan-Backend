import mongoose, { Schema, Document } from 'mongoose';
import { LabValue } from '../types';

export interface IConsultationDocument extends Document {
  userId: mongoose.Types.ObjectId;
  symptoms: string[];
  medicines: string[];
  labValues: LabValue[];
  voiceTranscript?: string;
  rawText?: string;
  summary: string;
  riskLevel: 'low' | 'medium' | 'high';
  riskScore: number;
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
});

const ConsultationSchema = new Schema<IConsultationDocument>(
  {
    userId: {
      type: Schema.Types.ObjectId,
      ref: 'User',
      required: true,
      index: true,
    },
    symptoms: [{ type: String, trim: true }],
    medicines: [{ type: String, trim: true }],
    labValues: [LabValueSchema],
    voiceTranscript: { type: String },
    rawText: { type: String },
    summary: { type: String, required: true },
    riskLevel: {
      type: String,
      enum: ['low', 'medium', 'high'],
      default: 'low',
      index: true,
    },
    riskScore: { type: Number, default: 0, min: 0, max: 100 },
    reportUrls: [{ type: String }],
    processingTime: { type: Number },
  },
  {
    timestamps: true,
  }
);

ConsultationSchema.index({ userId: 1, createdAt: -1 });
ConsultationSchema.index({ userId: 1, riskLevel: 1 });
ConsultationSchema.index({ riskLevel: 1, createdAt: -1 });

ConsultationSchema.index({
  symptoms: 'text',
  medicines: 'text',
  summary: 'text',
});

export const Consultation = mongoose.model<IConsultationDocument>(
  'Consultation',
  ConsultationSchema
);