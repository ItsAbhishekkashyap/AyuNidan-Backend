export interface IUser {
  _id: string;
  email: string;
  name: string;
  createdAt: Date;
}

export interface LabValue {
  name: string;
  value: string;
  unit: string;
  normalRange?: string;
  isAbnormal: boolean;
}

export interface IConsultation {
  _id: string;
  userId: string;
  symptoms: string[];
  medicines: string[];
  labValues: LabValue[];
  voiceTranscript?: string;
  summary: string;
  riskLevel: 'low' | 'medium' | 'high';
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