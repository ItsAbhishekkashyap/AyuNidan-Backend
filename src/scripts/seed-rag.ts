/**
 * Internal maintenance script: seeds the explainer glossary (20 short, app-authored definitions —
 * NOT the verified reference KB) into the `glossary` namespace of the configured Pinecone index.
 * Not exposed over HTTP. Run with: npm run seed:rag
 */
import 'dotenv/config';
import { seedMedicalKnowledgeBase } from '../services/rag.service';

seedMedicalKnowledgeBase()
  .then((message) => {
    console.log(message);
  })
  .catch((error: unknown) => {
    console.error('Seeding failed:', error instanceof Error ? error.name : 'UnknownError');
    process.exitCode = 1; // let open handles close naturally (process.exit() aborts on Windows)
  });
