import { generateObject } from 'ai';
import { createOpenAI } from '@ai-sdk/openai';
import { createAnthropic } from '@ai-sdk/anthropic';
import { createGoogleGenerativeAI } from '@ai-sdk/google';
import { z } from 'zod'; 
import { ExtractedEntities } from '../types';

export const getModel = () => {
  if (process.env.GEMINI_API_KEY && process.env.GEMINI_API_KEY.trim() !== '') {
    // Keeping the original Vercel AI SDK structure intact so rag.service.ts compiles perfectly
    const google = createGoogleGenerativeAI({ apiKey: process.env.GEMINI_API_KEY });
    return google('gemini-2.5-flash-lite'); 
  }

  if (process.env.OPENAI_API_KEY && process.env.OPENAI_API_KEY.startsWith('sk-')) {
    const openai = createOpenAI({ apiKey: process.env.OPENAI_API_KEY });
    return openai('gpt-4o');
  } 

  if (process.env.ANTHROPIC_API_KEY && process.env.ANTHROPIC_API_KEY.startsWith('sk-ant-')) {
    const anthropic = createAnthropic({ apiKey: process.env.ANTHROPIC_API_KEY });
    return anthropic('claude-3-5-sonnet-20240620');
  }

  throw new Error('No valid AI Provider API Key found.');
};

const ExtractionSchema = z.object({
  symptoms: z.array(z.string()),
  medicines: z.array(z.string()),
  labValues: z.array(z.object({
    name: z.string(),
    value: z.string(),
    unit: z.string().optional(),
    isAbnormal: z.boolean()
  })),
  fullNarrative: z.string()
});

export const extractMedicalData = async (
  rawText: string, 
  mediaFiles: Express.Multer.File[] = [] 
): Promise<ExtractedEntities> => {
  try {
    const model = getModel();
    
    const promptText = rawText ? rawText : "Please extract medical data from the provided document/image.";
    const contentArr: any[] = [
      { 
        type: 'text', 
        text: `You are an expert clinical data extraction assistant. Analyze the provided text, images, or documents. Extract structured symptoms, medicines, and lab values. Additionally, transcribe all clinical notes, demographics, family history, and physician observations completely into the fullNarrative field without shortening. DATA: ${promptText}` 
      }
    ];

    if (mediaFiles && mediaFiles.length > 0) {
      mediaFiles.forEach(file => {
        if (file.mimetype.startsWith('image/')) {
          contentArr.push({ type: 'image', image: file.buffer });
        } else if (file.mimetype === 'application/pdf' || file.mimetype.startsWith('audio/')) {
          contentArr.push({
            type: 'file',
            data: file.buffer,
            mediaType: file.mimetype
          });
        }
      });
    }

    console.log('⚡ [AI SERVICE] Processing generateObject execution thread...');
    const { object } = await generateObject({
      model: model,
      schema: ExtractionSchema,
      messages: [{ role: 'user', content: contentArr }],
      temperature: 0.1,
      maxRetries: 0 
    });

    return {
      symptoms: object.symptoms,
      medicines: object.medicines,
      labValues: object.labValues.map((lab: any) => ({
        name: lab.name,
        value: lab.value,
        unit: lab.unit || '',
        isAbnormal: lab.isAbnormal
      })),
      rawText: object.fullNarrative
    };

  } catch (error: any) {
    // ─── MASSIVE HIGH-DENSITY RADAR LOGGING START ───
    console.log('\n======================================================');
    console.log('🚨 [DEEP TRACE] CRASH INSIDE EXTRACT_MEDICAL_DATA');
    console.log('======================================================');
    console.log('➜ 1. Clean Error Message:', error?.message || error);
    console.log('➜ 2. Error Status Code / Type:', error?.status || error?.statusCode || error?.name || 'N/A');
    console.log('➜ 3. Complete Inner Object Dump (Exposing Hidden Fields):');
    console.dir(error, { depth: null, colors: true });
    if (error?.cause) {
      console.log('➜ 3.1. Underlying Error Cause Stack:');
      console.dir(error.cause, { depth: null, colors: true });
    }
    console.log('➜ 4. Error Stack Trace Pipeline:\n', error?.stack);
    console.log('======================================================\n');
    // ─── MASSIVE HIGH-DENSITY RADAR LOGGING END ───

    console.error('🔥 AI Extraction Error:', error?.message || error);
    throw new Error(error?.message || 'Failed to process medical data via AI.');
  }
};

const SummarySchema = z.object({
  summary: z.string(),
  riskLevel: z.enum(['low', 'medium', 'high']),
  riskScore: z.number().min(0).max(100)
});

export const generateClinicalSummary = async (
  payload: any
): Promise<{ summary: string; riskLevel: 'low' | 'medium' | 'high'; riskScore: number }> => {
  try {
    const model = getModel(); 
    console.log('⚡ [AI SERVICE] Processing generateClinicalSummary execution thread...');
    const { object } = await generateObject({
      model: model,
      schema: SummarySchema,
      prompt: `
        You are an AI Clinical Assistant preparing a preliminary consultation briefing.
        INPUT DATA: ${JSON.stringify(payload, null, 2)}
        TASK:
        1. Write a professional clinical summary. Include 'Generated by AI - Subject to Physician Review'.
        2. Classify Risk Level (low, medium, high).
        3. Assign Risk Score (0-100).
      `,
      temperature: 0.3, 
    });

    return object;
  } catch (error: any) {
    // ─── MASSIVE HIGH-DENSITY RADAR LOGGING START ───
    console.log('\n======================================================');
    console.log('🚨 [DEEP TRACE] CRASH INSIDE GENERATE_CLINICAL_SUMMARY');
    console.log('======================================================');
    console.log('➜ 1. Clean Error Message:', error?.message || error);
    console.log('➜ 2. Complete Inner Object Dump:');
    console.dir(error, { depth: null, colors: true });
    console.log('======================================================\n');
    // ─── MASSIVE HIGH-DENSITY RADAR LOGGING END ───

    console.error("Critical Summary Error:", error);
    throw new Error("Failed to generate clinical summary");
  }
};