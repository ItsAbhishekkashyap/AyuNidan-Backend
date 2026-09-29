> **ARCHIVED — historical pre-V2 snapshot (2026-09-29, before the V2 hardening).** It describes the codebase *before* the security, grounded-RAG and multimodal work and is kept only as an audit record. See `../../ARCHITECTURE.md` and `../../../README.md` for the current system.

# AyuNidan — Preliminary Interview Map (Current Implementation)

> A preliminary map only. Detailed preparation comes after the V2 upgrade.
> Each question is followed by where in the code the answer lives. `BE/` = `AyuNidan backend/`, `FE/` = `AyuNidan frontend/`.

## Project overview
- **What does AyuNidan do, end to end?** → `docs/current-architecture.md` §0, §7.1; `IntakeForm.tsx`, `createConsultation`
- **Why split extraction and consultation creation into two calls?** → Human-in-the-loop review: `IntakeForm.tsx:94-223`, `upload.controller.ts`, `consultation.controller.ts:41-67`
- **What's the most interesting technical problem you hit?** → Multimodal parsing + unreliable LLM JSON (`ai.service.ts:151-197, 436-506`), or the cache isolation bug (`cache.ts:46`)

## Architecture
- **Walk me through the request lifecycle of a consultation.** → `index.ts` middleware order → `consultation.routes.ts:21` → controller → `ai.service.ts` → Mongoose
- **Why Express + Next.js as separate apps, not Next API routes?** → Two repos; `FE/src/lib/api.ts` base URL; CORS in `index.ts:24-32`
- **How is the AI layer abstracted from controllers?** → `services/ai.service.ts` `getModelProvider`/`getModel`; HEAD version's multi-provider `getModel()`
- **What would break with 3 server instances?** → In-memory `cache.ts` and `rateLimiter.ts`

## Backend
- **How does your middleware chain work, and why that order?** → `index.ts:18-45`, per-route composition in `consultation.routes.ts`
- **How do you handle file uploads?** → `upload.routes.ts` (multer memoryStorage, limits, MIME filter)
- **How are errors propagated?** → Controller try/catch, `errorHandler.ts`, unused `AppError`
- **How did you build the cache middleware?** → `cache.ts:44-63` (`res.json` wrapping, TTL, FIFO eviction). Be ready to explain its key-design flaw.

## Database
- **Why MongoDB? Show the schema.** → `models/Consultation.ts`, `models/User.ts`
- **Which indexes serve which queries?** → `Consultation.ts:67-77` vs. `getConsultations` / `getRiskDashboard`
- **Explain the dashboard aggregation.** → `consultation.controller.ts:158-177`
- **Connection pooling and shutdown?** → `config/database.ts`
- **How would you paginate at scale / search by patient name?** → `getConsultations` (skip/limit, `$regex`)

## RAG
- **How does your RAG pipeline work?** → `rag.service.ts:97-162`
- **Why `inputType: 'passage'` vs `'query'`?** → `rag.service.ts:64-68, 99-103` (asymmetric E5 embeddings)
- **How did you pick topK = 2 and threshold 0.5?** → `rag.service.ts:114, 124`. There is no eval yet, so be honest about that.
- **How do you prevent the model from inventing definitions?** → Grounded/fallback prompt `rag.service.ts:133-143`
- **Is the user's own report retrievable via RAG?** → No. Uploaded docs are not indexed (audit §8).

## LLM
- **How do you get structured JSON out of the model?** → Current: regex + `JSON.parse` (`ai.service.ts:436-447, 569-580`). HEAD: `generateObject` + Zod.
- **What happens when a model fails?** → Fallback loop `ai.service.ts:307-517`; silent summary fallback `:603-611`
- **How do you handle images vs PDFs?** → `safelyExtractPDF` vs `extractMedicalDataFromImage` (Gemini Vision)
- **How do you choose temperature?** → Summary/RAG 0.2; notes in `PROMPT_ENGINEERING_NOTES.md`
- **What is FreeLLM and why use or remove it?** → `getModelProvider` `ai.service.ts:276-289`; audit §5
- **How is the risk score computed?** → Purely LLM-assigned, then clamped and normalised (`ai.service.ts:582-601`)

## Authentication
- **Explain your JWT flow.** → `auth.controller.ts:11-13`, `middleware/auth.ts`
- **How does Google sign-in work securely?** → `verifyIdToken` with audience, `auth.controller.ts:93-142`
- **Why check the user in the DB on every request?** → `auth.ts:41` (revocation vs cost)
- **Where is the token stored client-side, and what are the risks?** → `authSlice.ts`, `api.ts:71-83`

## Security
- **How do you isolate one doctor's data from another's?** → `userId` filters and ownership checks, plus the cache leak (audit §7). A strong "I found this in my own code" answer.
- **What unauthenticated endpoints exist, and why is that a problem?** → Audit §9, S2–S4
- **How do you handle PHI in logs?** → Currently not handled (`ai.service.ts` console logs), audit S6
- **Prompt injection via uploaded reports?** → Prompt construction `ai.service.ts:399-426`
- **What happens to uploaded files?** → Memory-only, buffers zeroed (`upload.controller.ts:34-36`)

## Performance
- **Where is the latency in creating a consultation?** → PDF parse + Vision + LLM calls. `processingTime` field (`consultation.controller.ts:36, 78`)
- **What happens if the LLM hangs?** → No timeouts (audit §12)
- **How does your rate limiter work, and what are its limits?** → `rateLimiter.ts` (fixed window, per-IP, in-memory)
- **Memory implications of a 50 MB upload?** → multer memoryStorage × 5 files

## Engineering decisions
- **Why the Vercel AI SDK vs LangChain vs raw SDKs?** → `ai` `generateText`, LangChain only for `PromptTemplate`, many unused SDKs in `package.json`
- **Why Pinecone integrated inference instead of OpenAI embeddings?** → `rag.service.ts:18, 64`
- **Why Redux for this app?** → `store/` slices (auth persistence, consultation status)
- **Why regex fallbacks for demographics?** → `ai.service.ts:453-483`

## Failure cases
- **The summary model is down. What does the doctor see?** → A "low" risk, score-0 record is saved (`ai.service.ts:603-611`). This is a key weakness to discuss.
- **A scanned PDF with no text layer?** → Empty text; it is not routed to Vision (`safelyExtractPDF`)
- **A voice note is uploaded to `/voice/transcribe`?** → Audio is ignored and the LLM gets no data (`ai.service.ts:315-367`)
- **Pinecone is unavailable?** → The explainer returns 500 (`rag.service.ts:158-161`)
- **The dashboard API fails?** → The FE shows demo numbers (`DashboardOverview.tsx:146-155`)
- **Two users open the same list within 30 s?** → Cache leak (`cache.ts:46`)

## Top 15 areas to understand deeply
1. End-to-end consultation flow (both HTTP calls and the human review step)
2. `extractMedicalData`: PDF/image handling, model fallback, JSON parsing
3. `generateClinicalSummary`: risk normalisation and its unsafe failure default
4. Structured output: regex parsing vs `generateObject` + Zod (HEAD vs working tree)
5. The RAG pipeline: E5 asymmetric embeddings, topK/threshold, grounded prompt
6. The FreeLLM proxy: what it is, the OpenAI-compatible and Responses API caveat, how to replace it
7. JWT + Google auth flow and the `authGuard` design
8. Per-user isolation and the cache-key vulnerability
9. Mongo schema and which compound index serves which query
10. The dashboard aggregation pipeline
11. Multer and upload safety (memory, MIME, auth gaps)
12. Rate limiting design and multi-instance limitations
13. Hallucination controls actually in code vs in `PROMPT_ENGINEERING_NOTES.md`
14. PHI handling: logs, third-party LLMs, and in-memory buffers
15. Why there are no tests, and how you would evaluate extraction and risk accuracy
