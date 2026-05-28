import { Pinecone } from '@pinecone-database/pinecone';
import { PromptTemplate } from '@langchain/core/prompts';
import { generateText } from 'ai';
import { getModel } from './ai.service'; 


if (!process.env.PINECONE_API_KEY) {
  console.warn('⚠️ PINECONE_API_KEY is missing. RAG endpoints will fail.');
}


const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY as string,
});

const indexName = 'panscience-medical';
const pineconeIndex = pinecone.Index(indexName);
const EMBEDDING_MODEL = 'multilingual-e5-large'; 


interface MedicalRecord {
  id: string;
  text: string;
}


export const seedMedicalKnowledgeBase = async (): Promise<string> => {
  try {
    const medicalData: MedicalRecord[] = [
      // Cardiovascular (Heart & Blood)
      { id: '1', text: 'Tachycardia: A heart rate that exceeds the normal resting rate, usually over 100 beats per minute in adults.' },
      { id: '2', text: 'Bradycardia: A slower than normal heart rate, typically under 60 beats per minute in adults.' },
      { id: '3', text: 'Hypertension: High blood pressure, a condition where the force of the blood against the artery walls is consistently too high.' },
      { id: '4', text: 'Hypotension: Abnormally low blood pressure, which can cause dizziness and fainting due to inadequate blood flow to the brain.' },
      { id: '5', text: 'Atherosclerosis: The build-up of fats, cholesterol, and other substances in and on the artery walls, which can restrict blood flow.' },
      { id: '6', text: 'Arrhythmia: An improper or irregular beating of the heart, meaning it beats too fast, too slow, or with an irregular pattern.' },

      // Metabolic & Endocrine (Sugar, Fats, Hormones)
      { id: '7', text: 'Hyperlipidemia: Elevated levels of lipids (fats), such as cholesterol or triglycerides, in the blood.' },
      { id: '8', text: 'Hypoglycemia: A condition caused by a very low level of blood sugar (glucose), which is the body\'s main energy source.' },
      { id: '9', text: 'Hyperglycemia: High blood sugar, commonly associated with diabetes, occurring when the body lacks enough insulin or cannot use it properly.' },
      { id: '10', text: 'Hypothyroidism: A condition where the thyroid gland is underactive and doesn\'t produce enough crucial hormones, often slowing down metabolism.' },

      // Respiratory (Lungs & Breathing)
      { id: '11', text: 'Asthma: A condition in which a person\'s airways narrow, swell, and produce extra mucus, making breathing difficult.' },
      { id: '12', text: 'COPD: Chronic Obstructive Pulmonary Disease, a chronic inflammatory lung disease that causes obstructed airflow from the lungs.' },
      { id: '13', text: 'Apnea: A temporary cessation of breathing, most commonly experienced during sleep (Sleep Apnea).' },

      // Neurological (Brain & Nerves)
      { id: '14', text: 'Migraine: A neurological condition that can cause multiple symptoms, most notably a severe, throbbing headache typically on one side of the head.' },
      { id: '15', text: 'Neuropathy: Damage or dysfunction of one or more nerves that typically results in numbness, tingling, muscle weakness, and pain, often in the hands and feet.' },
      { id: '16', text: 'Vertigo: A sudden sensation of feeling off-balance or spinning, often caused by an inner ear problem.' },
      // Gastrointestinal & General (Stomach, Blood, Bones, Tissues)
      { id: '17', text: 'GERD: Gastroesophageal Reflux Disease, a digestive disorder where stomach acid frequently flows back into the tube connecting your mouth and stomach.' },
      { id: '18', text: 'Anemia: A condition in which the blood lacks enough healthy red blood cells or hemoglobin to carry adequate oxygen to the body\'s tissues.' },
      { id: '19', text: 'Osteoporosis: A disease that weakens bones to the point where they break easily—most often in the hip, backbone (spine), and wrist.' },
      { id: '20', text: 'Edema: Visible swelling caused by an accumulation of excess fluid trapped in the body\'s tissues, most commonly noticed in the hands, arms, feet, and ankles.' }
    ]


    const textsToEmbed = medicalData.map(doc => doc.text);

  
    const embeddingResult = await pinecone.inference.embed(
      EMBEDDING_MODEL,
      textsToEmbed,
      { inputType: 'passage' }
    );


    const vectors = medicalData.map((doc, i) => {
      const rawData = embeddingResult.data[i] as any;
      const vectorValues: number[] = rawData.values;
      
      if (!vectorValues) {
        throw new Error(`Failed to generate vector for document: ${doc.id}`);
      }

      return {
        id: doc.id,
        values: vectorValues,
        metadata: { text: doc.text } 
      };
    });

    await pineconeIndex.upsert(vectors);

    return `Successfully seeded ${vectors.length} medical records into Pinecone!`;
  } catch (error) {
    console.error('Pinecone Seeding Error:', error);
    throw new Error('Failed to seed Pinecone Vector Database');
  }
};



export const explainMedicalTermRAG = async (term: string): Promise<string> => {
  try {
    const queryEmbeddingResult = await pinecone.inference.embed(
      EMBEDDING_MODEL,
      [term],
      { inputType: 'query' }
    );

    const rawQueryData = queryEmbeddingResult.data[0] as any;
    const queryVector: number[] = rawQueryData.values;
    
    if (!queryVector) {
      throw new Error('Query vector generation failed');
    }

    const queryResponse = await pineconeIndex.query({
      vector: queryVector,
      topK: 2, 
      includeMetadata: true 
    });

    let retrievedContext = "No specific data found in Pinecone. Fallback to general AI knowledge.";
    

    if (queryResponse.matches && queryResponse.matches.length > 0) {
    
      const bestMatch = queryResponse.matches[0];
      if (bestMatch.score && bestMatch.score > 0.5) {
        retrievedContext = queryResponse.matches
          .map(match => (match.metadata as Record<string, any>)?.text as string)
          .filter(Boolean)
          .join('\n\n');
      }
    }

   
    const promptTemplate = PromptTemplate.fromTemplate(`
      You are an empathetic AI Clinical Explainer.
      
      PATIENT QUESTION/TERM: {term}
      DATABASE CONTEXT: {context}
      
      TASK:
      1. If the DATABASE CONTEXT contains the definition, explain the term strictly using that context.
      2. If the DATABASE CONTEXT says "No specific data found", use your general medical knowledge to explain the term, but start your response with: "Based on general medical knowledge: "
      3. Keep it simple, professional, non-frightening, and under 3 sentences.
    `);

    const formattedPrompt = await promptTemplate.format({
      term,
      context: retrievedContext,
    });

    const model = getModel(); 
    const { text } = await generateText({
      model: model,
      prompt: formattedPrompt,
      temperature: 0.2, 
    });

    return text;
  } catch (error) {
    console.error('RAG Generation Error:', error);
    throw new Error('Failed to generate medical explanation');
  }
};