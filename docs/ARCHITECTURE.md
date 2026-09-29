# AyuNidan — Architecture

AyuNidan is an AI-powered clinical **document-intelligence and evidence-grounded decision-support prototype**.
It is not a diagnostic system and has not been clinically validated.

This document describes the implemented system (Express + TypeScript API, Next.js client, MongoDB, Pinecone,
local Hugging Face embeddings, Gemini via the Vercel AI SDK, LangChain prompt layer).

## 1. System overview

```
                         Browser (Next.js 16, React 19, Redux Toolkit)
   Login / Google ─┐   Upload PDF·image·text   Voice (Web Speech API, live)   Ask a question
                   ↓             ↓                        ↓                          ↓
        ─────────────────────────  HTTPS + Bearer JWT  ───────────────────────────────────
                                   Express 5 API  (helmet · CORS · rate limits · Zod validation)
                                                 │
   ┌─────────────────────────────────────────────┼──────────────────────────────────────────┐
   │ JOB A: Document understanding               │                                          │
   │  PDF / image / text / audio                 │                                          │
   │   ↓ local: per-page text layer, table       │                                          │
   │     detection, scanned/embedded-image       │                                          │
   │     detection                               │                                          │
   │   ↓ only what can't be read locally goes    │                                          │
   │     to the multimodal model                 │                                          │
   │   ↓ ONE structured extraction call (Zod)    │                                          │
   │  Normalized documents + Structured Report   │                                          │
   └──────────┬──────────────────────────────────┘                                          │
              ↓ chunk (text: 1000/150; table rows atomic) → local HF embeddings (bge-small, 384-d)
   ┌──────────────────────────────┐            ┌──────────────────────────────────────────┐
   │ PATIENT VECTOR STORE          │            │ VERIFIED MEDICAL-REFERENCE VECTOR STORE   │
   │ Pinecone ns `user-<userId>`   │            │ Pinecone ns `medical-reference` (shared)  │
   │ + Mongo `documents` (owner)   │            │ + Mongo `referencesources` (registry)     │
   └──────────────┬───────────────┘            └────────────────────┬─────────────────────┘
                  └──────── query / findings → embed (same model) ───┘
                                 ↓ metadata filters · similarity threshold · evidence budget
                        Relevant evidence only:  F# findings · P# patient · R# reference
                                 ↓
                 JOB B: LangChain ChatPromptTemplate  (data ≠ instructions)
                                 ↓  ONE structured call (Zod), fallback chain + circuit breaker
                 Validation: citation ids · risk/score consistency · insufficient_evidence
                                 ↓
                  MongoDB `consultations` (status: processing → completed | failed)
                                 ↓
        Dashboard: structured report · assessment · uncertainty · patient evidence · Medical References Used
```

## 2. Components

### Frontend (`AyuNidan frontend`)
Next.js 16 App Router, React 19, Redux Toolkit (auth + consultation slices), shadcn/Radix UI.
- `lib/api.ts` — single typed client; attaches the JWT from `localStorage`.
- `components/IntakeForm.tsx` — upload, voice, typed notes; the clinician **reviews/edits** extracted data before creating a consultation (human-in-the-loop).
- `hooks/useVoiceTranscription.ts` + `lib/transcript.ts` — live interim/final transcript state machine (unit-tested).
- `app/consultation/[id]` + `components/report/ReportInsights.tsx` — structured report, assessment, evidence, **Medical References Used**, "Ask about this report".
- Failed/processing consultations render as **AI FAILED / Not assessed**, never as low risk; `insufficient_evidence` has its own state.

### Backend API (`AyuNidan backend`)
Express 5 + TypeScript (strict). Layers: `routes → middleware (auth, rate limit, validate, upload, cache) → controllers → services → models`.

| Route | Purpose |
|---|---|
| `POST /api/auth/register · login · google` | Local and Google sign-in |
| `POST /api/uploads` | Job A on files/text; indexes documents; returns structured report |
| `POST /api/voice/transcribe` | Server-side transcription fallback (final transcript only) |
| `POST /api/consultations` | Persist input → Job B assessment → persist result |
| `GET /api/consultations[/:id \| /dashboard]`, `DELETE /:id` | Per-user history, stats, cascade delete |
| `POST /api/documents/query`, `GET /api/documents`, `DELETE /api/documents/:id` | Grounded Q&A; list/delete own documents |
| `GET /api/consultations/explain?term=` | Glossary explainer (separate namespace) |
| `GET /health` | Liveness only (no memory/env data) |

There are no debug endpoints, and the vector-seeding endpoint was removed (seeding is a CLI script).

### Authentication and authorization
- Local: bcrypt (10 rounds); Google: ID token verified server-side (audience, `email_verified`).
- JWT HS256, `JWT_EXPIRES_IN` (default 7d); secret from validated config — the server **refuses to start** without a ≥32-char `JWT_SECRET`; there is no fallback secret.
- `authGuard` verifies the token **and** that the user still exists on every request.
- Authorization is ownership-by-query: every read/write filters on `req.user.id` (never on a body-supplied id).
- Rate limits (in-memory, per instance): AI 10/min, general 100/min, auth 20/15 min per IP, login 5/15 min per account.

### Document ingestion and multimodal processing (`src/documents`, `services/ai.service.ts`)
| Input | Handling |
|---|---|
| PDF with text layer | Local per-page text + layout-based table detection (`pdfAnalyzer.ts`, `tables.ts`); text is sent to the model as text, not as a file |
| Scanned / image-only pages | Detected per page (image operators, no text) → the PDF is attached for the multimodal model; noted in `extractionLimitations` |
| Text + embedded images | Page status `mixed`; text used locally, image content read by the model and flagged model-transcribed |
| Images | Sent natively to the multimodal model; result marked *model-transcribed, unverified*; no local OCR |
| Text files / typed notes | Decoded locally; delimited (`\|`/tab) lab tables parsed deterministically |
| Audio (server fallback) | Validated (size, WAV header/duration), transcribed by Gemini in the same extraction call; final transcript only |

All inputs converge to `NormalizedDocument { pages, tables, needsModelReading, notes }`.
Uploads are validated by size/count, MIME **and** extension **and** magic bytes; filenames are sanitized; buffers are zeroed after use (memory storage only).

### Structured extraction — Job A
One Zod-schema-constrained model call (`Output.object`): demographics, symptoms, medicines, labs (unit, range, flag, date), dates, diagnoses *written in the source*, measurements, image-derived findings, doctor notes, limitations.
Deterministic table rows are **authoritative** over model output for labs and carry exact `{file, page, tableId}` provenance; model-only labs get a `text_match` page only when the value appears on exactly one page, otherwise `model` (no page claimed).

### Embeddings and vector stores
- Model: `Xenova/bge-small-en-v1.5` (ONNX, q8) run **in-process** via `@huggingface/transformers` — no paid embedding API. 384-d, CLS pooling, normalized, cosine. Queries get BGE's retrieval instruction prefix.
- Exposed through LangChain's `Embeddings` abstraction (`LocalHuggingFaceEmbeddings`); the same instance embeds documents and queries.
- Pinecone serverless index `ayunidan-bge-small-384` (cosine). Every chunk stores `embeddingSpace`; mismatching vectors are ignored.
- **Domain A – patient documents**: namespace `user-<userId>`, metadata `{domain:'patient', userId, documentId, chunkId, page, contentType, tableId, source, embeddingSpace, text}`.
- **Domain B – verified medical references**: namespace `medical-reference`, metadata `{domain:'reference', sourceId, title, organization, section, page, publicationDate?, version?, url?, …}`; a Mongo registry decides which sources are servable. The corpus ships empty (see `medical-reference/README.md`).
- **Glossary** (20 app-authored definitions for the explainer): its own `glossary` namespace; never presented as a verified reference.

### Chunking
Deterministic: text 1000 chars / 150 overlap, breaking at paragraph → line → sentence → word; chunks never span pages; **table rows are atomic** and rendered as `Test: Result = 10.2 g/dL, Reference range = 13-17 g/dL, Flag = Low, Page = 2`.

### Retrieval
Embed query → namespace search with metadata filter (`domain`, `userId`, allowed `documentId`s) → per-chunk metadata validation → **per-chunk** similarity threshold (patient 0.68, reference 0.66) → context budget. Assessment issues up to 6 deterministic, abnormal-first queries; reference queries are embedded locally and searched in parallel; the patient namespace is queried once per finding-query (sequentially).

### LangChain prompt layer (`src/prompts`)
`ChatPromptTemplate`s for extraction, clinical assessment and medical Q&A. Untrusted content enters only via escaped, delimited variables (`<patient_findings>`, `<patient_evidence>`, `<verified_medical_evidence>`, `<user_query>`, `<data>`); every system prompt embeds the shared data-handling rules. LangChain is used for prompt templates, `Document`, and the `Embeddings` abstraction only — not chains/agents/vector-store wrappers.

### Grounded assessment — Job B (`services/assessment.service.ts`)
One structured call over numbered findings + budgeted evidence (≤6 patient + ≤8 reference chunks). Output: `riskLevel ∈ {low, medium, high, insufficient_evidence}`, score, summary, key findings with evidence ids, uncertainty. These are engineering triage labels, not a validated clinical scale.

### Citation validation
Model-supplied ids are checked against the supplied evidence (`P#`/`R#`/`F#`). Unknown ids are dropped and counted (`citation_failure`); client-facing citations are rebuilt **only from retrieved metadata**. A result that cites nothing valid is downgraded to `insufficient_evidence` (never to "low"); Q&A without a valid citation becomes an explicit insufficiency.

### Persistence (MongoDB / Mongoose)
`users`, `consultations` (`status: processing|completed|failed`, `failure{stage,category,attempts}`, `assessment`, `report`, `ai` provenance), `documents` (owner, status, embedding space, tables), `referencesources` (registry). Input is persisted **before** AI processing so a failure never loses the clinician's data.

### Dashboard
Risk distribution counts only completed low/medium/high results; failed and insufficient-evidence counts are reported separately.

### Medical terminology domain and evidence contract

Four evidence id spaces keep knowledge domains apart: `F#` structured patient findings, `P#` patient document evidence (`user-<id>` namespace), `R#` verified medical references (`medical-reference`), `T#` medical terminology (`medical-terminology`, NLM MedlinePlus + MeSH, local bge-small embeddings, exact-key + semantic retrieval). `src/rag/routing.ts` routes a question deterministically: definition → terminology only; clinical interpretation → patient + verified reference (needs a cited `R#`); report lookup → patient evidence. Assessment and Q&A validate citations per domain (lists and inline `[R1]` tokens), strip fabricated ids, show only cited sources, and return `insufficient_evidence` when a required verified reference is missing or uncited. Terminology never reaches the assessment prompt. Code: `src/terminology/*` (streaming ZIP reader, parsers), `src/services/terminology*.ts`, `src/scripts/ingest-terminology.ts`.

Real guideline PDFs (`medical-reference/clinical-guidelines/` + `*.meta.json`) use the existing reference pipeline unchanged (`npm run kb:ingest`); added for real-document quality: column-aware page reading (`splitTwoColumns`), running header/footer removal, bibliography-chunk exclusion, stricter heading detection, and disabling pdf.js browser font loading (an uncaught crash on PDFs with embedded fonts).

## 3. Security boundaries
1. **Auth/ownership** on every data route; user id from the JWT only.
2. **Tenant isolation for vectors** (three layers): per-user namespace + metadata filter + post-query validation against the caller's Mongo document records (deleted/foreign/other-embedding-space vectors are dropped).
3. **Cache**: keys are `user:<id>:<url>`, only successful responses, invalidated on mutation, `Cache-Control: private, no-store`.
4. **Prompt-injection**: data/instruction separation, escaping, validated citations, structured output only.
5. **PHI-safe logging**: structured logs contain ids, model names, latencies, token counts, status/category — never text, prompts, responses, queries or tokens (asserted by tests). Mongoose debug logs collection/method only.
6. **Uploads**: authenticated + rate-limited *before* multipart parsing; magic-byte validation.
7. **Config**: Zod-validated; secrets only from env; `.env` git-ignored; no debug/seed HTTP routes.

## 4. Failure handling
- 20 categorized failure types (`utils/failures.ts`), counted and logged: provider failure/timeout/rate-limit, model unavailable, fallback exhausted, schema-validation failure, embedding/vector-DB failure, retrieval miss, insufficient context, invalid metadata, citation failure, table-extraction failure, transcription failure, persistence failure, …
- **LLM calls**: ordered fallback (Gemini primary → Gemini fallback → OpenAI/Anthropic if keyed), per-call timeout (45 s), total budget (120 s), ≤3 models, SDK retries ≤1; schema/validation failures are not retried on the same model.
- **Circuit breaker** (in-process): a model returning 404 is skipped for 30 min, 401/403 for 10 min, 429 for `Retry-After`/60 s.
- **Consultation lifecycle**: any AI/persistence failure marks the record `failed` with stage + category; the UI shows *AI FAILED / Not assessed*; failed records are excluded from risk statistics.
- **Vector outage**: indexing failure never fails extraction (document marked `failed`); reference outage degrades to patient-only evidence, stated in the response; patient-retrieval outage on Q&A returns a categorized 502.
- **Observability**: request-id correlation (AsyncLocalStorage), `Server-Timing` headers, per-call AI metrics (latency, tokens, fallback), in-process p50/p95 stats.

## 5. Known constraints
See the README "Known Limitations". Notably: cache and rate limiters are per-process; the reference corpus is empty until authorised sources are ingested; thresholds and metrics come from small synthetic sets.
