export interface IUser {
  _id: string;
  email: string;
  name: string;
  createdAt: Date;
}

/**
 * Where a lab value came from:
 *  - table: parsed deterministically from a document table (exact page/table)
 *  - text_match: model-extracted, located verbatim on exactly one page of the source text
 *  - model: model-extracted, location not independently verifiable (e.g. images, audio)
 */
export interface LabValueSource {
  method: 'table' | 'text_match' | 'model';
  filename?: string;
  /** Set only when the value was found on exactly one page. */
  page?: number;
  /** All pages where the value was found, when it appears on more than one. */
  pages?: number[];
  tableId?: string;
}

export interface LabValue {
  name: string;
  value: string;
  unit: string;
  normalRange?: string;
  isAbnormal: boolean;
  /** Flag (High / Low / Normal …). See `flagSource` for how it was determined. */
  flag?: string;
  /**
   *  - printed_range: computed by code from the value and the reference range printed in the report
   *  - table: the flag printed in the source table
   *  - model: the AI's opinion (used only when no printed range could be evaluated)
   */
  flagSource?: 'printed_range' | 'table' | 'model';
  date?: string;
  source?: LabValueSource;
}

export interface PatientDetails {
  name: string;
  age?: number;
  gender?: string;
}

export interface IConsultation {
  _id: string;
  userId: string;
  symptoms: string[];
  medicines: string[];
  labValues: LabValue[];
  voiceTranscript?: string;
  summary: string;
  riskLevel?: 'low' | 'medium' | 'high' | 'insufficient_evidence';
  riskScore: number;
  reportUrls: string[];
  createdAt: Date;
}

export interface ExtractedEntities {
  symptoms: string[];
  medicines: string[];
  labValues: LabValue[];
  rawText: string;
}

export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}

export type RiskLevel = 'low' | 'medium' | 'high';

export interface MongooseOptions {
  maxPoolSize: number;
  minPoolSize: number;
  socketTimeoutMS: number;
  serverSelectionTimeoutMS: number;
  heartbeatFrequencyMS: number;
}