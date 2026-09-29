# AyuNidan

**AI-powered clinical document intelligence and evidence-grounded decision-support prototype.**

Upload a clinical report (PDF, image, text) or dictate a note; AyuNidan extracts a structured report, indexes it,
retrieves relevant evidence from the patient's own documents and from a separately curated medical-reference knowledge
base, and produces a **grounded, cited, preliminary assessment for clinician review**.

> **Not a diagnostic system.** AyuNidan does not diagnose, has **not** been clinically validated, and has not been
> deployed clinically. Risk levels are engineering triage labels produced by an LLM from supplied evidence — not a
> validated clinical score. All evaluation is synthetic engineering validation (see `docs/EVALUATION.md`).

> **Repositories.** The project is split in two: the backend (this repo) and the frontend (`ai-clinical-frontend`). Commands below use the
> sibling folder names `AyuNidan backend/` and `AyuNidan frontend/`; clone both side by side (in the backend repo, `cd` means the repo root).

## What it does
1. A clinician uploads a report or dictates a note; the system extracts patient demographics, symptoms, medicines, lab values (with units, ranges, flags, dates, page/table provenance), dates, written diagnoses, measurements, image-derived findings and notes.
2. The clinician reviews/edits the extraction, then creates a consultation.
3. The system retrieves relevant patient-document evidence and verified medical-reference evidence, and asks the LLM to assess **only** from that evidence, returning `low | medium | high | insufficient_evidence`, key findings, uncertainty, and citations.
4. The dashboard shows the structured report, the assessment, the patient evidence and **Medical References Used** (title, organization, section, page, URL/version/date when known), and lets the user ask cited questions about the report.

## Problem
Clinical documents arrive as scanned PDFs, tables, photos and dictation. Naive LLM summarization has three failure modes
this project is designed around: **hallucinated medical facts and sources**, **silent failure that looks like a normal result**
(an outage recorded as "low risk"), and **cross-patient data leakage**.

## Architecture
```
Document / Image / PDF / Voice
        ↓
Normalization + Extraction (local parsing first; one structured LLM call)
        ↓
Structured Patient Representation ──→ Patient Vector Store (per-user namespace)
        ↓                                        ↓
   Findings / user query ──→ retrieval ←── Verified Medical KB (shared, curated)
                                  ↓
                  Relevant evidence only (budgeted, validated)
                                  ↓
              LangChain prompt → Grounded LLM (structured output)
                                  ↓
     Citation validation → Assessment (or insufficient_evidence / explicit failure)
                                  ↓
                     Dashboard with sources and provenance
```
Details: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Core features
- Multimodal intake: text-layer PDFs, scanned/mixed PDFs, images, typed notes, live voice dictation.
- Structured extraction with per-field **source provenance** (file · page · table).
- Two **separate** vector domains: user-isolated patient documents and a shared verified-reference KB.
- Grounded assessment and Q&A with **server-validated citations** and explicit abstention.
- Failure-aware processing: every consultation is `processing → completed | failed`; failures are visible, categorized and never shown as low risk.
- Security-hardened API (see below) and a reproducible synthetic evaluation harness.

## AI/RAG pipeline
| Stage | Implementation |
|---|---|
| Chunking | text 1000/150 chars (paragraph→sentence boundaries); lab-table rows atomic; never across pages |
| Embeddings | **local** `Xenova/bge-small-en-v1.5` (ONNX q8, 384-d, cosine) via `@huggingface/transformers`, wrapped as a LangChain `Embeddings` |
| Vector DB | Pinecone serverless `ayunidan-bge-small-384`; namespaces `user-<id>`, `medical-reference`, `medical-terminology`, `glossary` |
| Retrieval | metadata filters + per-chunk similarity floors (patient 0.50, reference 0.66, terminology 0.70 with exact-match bypass; all configurable) + context budget |
| Prompting | LangChain `ChatPromptTemplate`s (`src/prompts`): extraction, clinical assessment, medical Q&A |
| LLM | Vercel AI SDK, structured output via Zod; `GEMINI_MODEL` (default `gemini-3.5-flash-lite`) with fallback chain and circuit breaker |
| Jobs | **Job A** document understanding and **Job B** evidence assessment are separate calls (accuracy, debuggability, cost) |

## Multimodal processing
Local per-page PDF analysis (text, tables, image operators) decides what the model must read. Text PDFs are never sent as files;
scanned pages and images go to the multimodal model and are labelled *model-transcribed, unverified*. Voice: live interim/final
transcript in the browser (Web Speech API), only **final** text is used; a validated server fallback transcribes recordings when the
browser has no speech recognition. Limitations are recorded in the report (`extractionLimitations`) instead of hidden.

## Medical Reference Knowledge Base
The verified reference corpus is ingested **separately** from patient data. **The repository does not contain the guideline PDFs or the NLM
datasets** (copyright/licensing and size): you place authorised copies locally. Only the provenance sidecars for the two guidelines used in
development are included (`medical-reference/clinical-guidelines/*.meta.json`); the PDFs are git-ignored, so obtain them from their
publishers (ACC/AHA *Circulation* 2026 dyslipidemia guideline; WHO 2024 haemoglobin-cutoffs guideline) and keep the file names used in the sidecars.
No guideline text was fabricated. Each source needs a `.meta.json` sidecar with mandatory provenance (`sourceId`, `title`, `organization`,
`sourceType`, `authorization`) and optional `publicationDate`, `version`, `url`, `medicalTopics`; sections and pages are extracted
from the document. Ingest with:

```bash
cd "AyuNidan backend"
# put files + <name>.meta.json in medical-reference/sources/ or medical-reference/clinical-guidelines/
npm run kb:ingest               # idempotent; add -- --force to rebuild a source
```
Reference chunks stay in the shared `medical-reference` namespace and are never mixed with patient chunks. Without ingested
sources, assessments are grounded only in the patient's own findings (including reference ranges printed in the report) and return
`insufficient_evidence` (a risk level requires a cited verified reference). Full rules: [`medical-reference/README.md`](medical-reference/README.md).

## Clinical Reference Corpus

Three knowledge sources are kept strictly apart:

| Domain | Contents | Namespace | Evidence id |
|---|---|---|---|
| **Clinical evidence** (verified references) | the real guideline PDFs below | `medical-reference` | `R#` |
| **Terminology** | NLM MedlinePlus + MeSH definitions/synonyms | `medical-terminology` | `T#` |
| **Synthetic references** | fictional passages for tests/demos only (badged "synthetic test source"; served only with `RAG_INCLUDE_SYNTHETIC_REFERENCES=true`; **none are indexed now**) | `medical-reference` (registry-gated) | `R#` |

Real guidelines ingested (from `medical-reference/clinical-guidelines/`, files unmodified; provenance is in the `*.meta.json` sidecars and was read from the documents themselves):

| Guideline | Organization | Version / date (as printed) | Chunks |
|---|---|---|---|
| 2026 ACC/AHA/AACVPR/ABC/ACPM/ADA/AGS/APhA/ASPC/NLA/PCNA Guideline on the Management of Dyslipidemia (123 pp.) | American College of Cardiology / American Heart Association Joint Committee on Clinical Practice Guidelines | Circulation 2026;153:e1154–e1276, DOI 10.1161/CIR.0000000000001423, April 28, 2026 | 631 |
| Guideline on haemoglobin cutoffs to define anaemia in individuals and populations (79 pp.) | World Health Organization | © 2024, ISBN 978-92-4-008854-2 (electronic) | 290 |

Notes: no source URL is stored because neither document prints a stable guideline URL (only publisher/DOI-supplement links); page numbers are PDF page indices, not the printed journal page numbers; section labels are the nearest detected heading, and in recommendation tables may be a numbered recommendation line. Ingestion (`npm run kb:ingest`, `--force` to rebuild) is idempotent: deterministic ids `<sourceId>#<n>`, unchanged files are skipped, a rebuild replaces the previous vectors (verified: 921 vectors = 631 + 290, none duplicated). Two-column journal pages are read column by column, running headers/footers are removed, and bibliography chunks (3+ journal citations) are not indexed. Retrieval is semantic (bge-small, floor 0.66); a floor cannot decide answerability, so the model's abstention and server-side citation validation decide.

**Verified with the real corpus (live Gemini/Atlas/Pinecone run, fictional patient, temporary users removed):** patient facts answered from `F#/P#` without a reference; definitions from `T#`; a guideline question answered with `R#` citations showing title, organization, page and section; an assessment citing two guideline chunks while six other retrieved chunks were not shown as used; a question outside the corpus (pediatric leukemia) and an unrelated assessment returned `insufficient_evidence`; an instruction planted in the patient PDF was ignored; a second user got no data and a 404. This is an engineering verification of grounding behaviour, **not clinical validation**; AyuNidan is a clinical document-intelligence / decision-support prototype.

## Medical Terminology Domain

A separate knowledge domain (own vector namespace `medical-terminology`, evidence ids `T#`) explains what a medical *term means*. It is built from two official NLM datasets supplied locally as ZIPs (not included in this repository: download the MedlinePlus Health Topics XML and the MeSH descriptor XML from the National Library of Medicine, zip them if needed and place one ZIP in each of `medical-reference/terminology/medlineplus/` and `medical-reference/terminology/mesh/`): **MedlinePlus Health Topics** (English topics, consumer-language summaries, "also called"/"see" synonyms, topic URL, MeSH cross-reference) and **MeSH descriptors** (topical descriptors with a scope note, entry-term synonyms, tree numbers). Provenance (dataset, organisation, version, dataset date, URL, match type) comes only from the dataset files; version labels are `2026-09-26` (MedlinePlus `date-generated`) and `MeSH 2026` (file name). Nothing is invented.

- **Ingestion:** `npm run terminology:ingest -- --dataset=all` (`--dry-run`, `--limit=N`, `--force`). The ZIPs are streamed read-only (never modified or extracted), parsed deterministically, embedded with the **local** bge-small model and upserted under deterministic ids (`mplus:<id>`, `mesh:<DescriptorUI>`), so re-running is idempotent; a manifest in `medical-reference/processed/` skips unchanged datasets and removes records that disappeared. Real result: 1,014 MedlinePlus topics (of 2,033 blocks: 1,016 Spanish and 3 without summary skipped) and 30,364 MeSH descriptors (598 non-topical and 148 without scope note skipped).
- **Retrieval:** exact term/synonym match on normalised keys first, then semantic matches above `RAG_TERM_MIN_SCORE` (0.70, measured with `npm run terminology:eval`). Definition questions ("What does HbA1c mean?") and `/consultations/explain` use only this domain; the model explains the retrieved entries and must cite `T#`. No match ⇒ an explicit "no matching entry" message, never a definition from model memory.
- **Not a clinical reference:** terminology is never a "verified medical reference", is never shown in the assessment, and can never justify a risk level or a clinical answer.

**Evidence types.** `F#` patient finding (structured, from the report) · `P#` patient document evidence · `R#` verified medical reference · `T#` medical terminology. Citations (structured lists *and* inline `[R1]` tokens) are validated server-side per domain; fabricated or wrong-domain ids are dropped from lists and prose; UI metadata is built only from retrieved evidence. A risk level requires at least one *cited* verified reference, otherwise the result is `insufficient_evidence` (`RAG_REQUIRE_REFERENCE_FOR_ASSESSMENT`, default on). Clinical Q&A follows the same rule; report lookups (P#) do not need a reference. References that were retrieved but not cited are never shown as used.

**Verified status (2026-09-29):** the `medical-terminology` namespace holds 31,378 vectors (1,014 MedlinePlus + 30,364 MeSH; no duplicates; manifests written). Live retrieval against it returned the expected entries with correct provenance for "What is HbA1c?" (MedlinePlus *A1C*, exact synonym), "What is creatinine?" (MeSH *Creatinine*, exact) and "What is triglyceride?" (MeSH + MedlinePlus *Triglycerides*), and no terminology for unrelated queries. A live end-to-end run on a fictional PDF (synthetic reference stand-ins, temporary users, all removed afterwards) confirmed: terminology answer with `T#` citations; assessment with cited `R#` references; `insufficient_evidence` when no verified reference exists; clinical Q&A refused without a verified reference; `F#`/`P#` evidence for report facts; and tenant isolation (another user got no data and a 404). Automated (final): 270 backend tests, 10 frontend tests, typecheck, builds and lint (0 errors) pass. The reference behaviour was first exercised with synthetic sources and then with the real guidelines (see Clinical Reference Corpus).

**Refreshing the datasets:** replace the ZIP in `medical-reference/terminology/<dataset>/`, run `npm run terminology:ingest -- --dataset=<name>`, re-run `npm run terminology:eval` and review the threshold.

**Non-claims:** definitions are derived from official NLM/MedlinePlus datasets but AyuNidan is a prototype decision-support aid, not a diagnostic system, and has not been clinically validated.

## Security
JWT auth with user re-check per request and **no fallback secret** (server refuses to start without one) · ownership-scoped queries ·
3-layer vector tenant isolation · user-scoped success-only response cache (a cross-user leak found in the initial audit was fixed and
regression-tested) · authenticated + rate-limited upload/AI endpoints · magic-byte upload validation · Zod request validation ·
login brute-force limits · prompt-injection defences (data/instruction separation, escaping, validated citations) · PHI-free
structured logs · no debug/seed HTTP routes · helmet + CORS allow-list.

## Evaluation
Engineering validation on synthetic fixtures — **not clinical validation** ([`docs/EVALUATION.md`](docs/EVALUATION.md)):
270/270 backend tests · 10/10 frontend tests · typecheck and builds clean · 16/16 failure scenarios categorized · table extraction 8/8 rows
and all fields correct on 4 fixtures · local-BGE retrieval Hit@1 = 1.0 (patient, n=10) and source Hit@4 = 1.0 (reference, n=6) on tiny synthetic sets ·
one live fictional end-to-end run with `gemini-3.5-flash-lite`. LLM-quality metrics (extraction F1, risk-class confusion) are **not** reported.

## Performance
Local embedding: 375 ms cached load, 661 ms for 32 chunks. Final fictional HTTP run (n = 1): upload+extraction+indexing 7.0 s, assessment 6.3 s
(≈2.9 s model call, rest retrieval round-trips), cited Q&A 2.5 s. Cost controls: local parsing/tables/embeddings, one call per job, budgeted evidence,
token logging. Details and caveats in the evaluation document.

## Tech stack
**Backend:** Node 24, Express 5, TypeScript (strict), Mongoose 9/MongoDB Atlas, Zod, Vercel AI SDK 6 (Gemini/OpenAI/Anthropic providers), LangChain Core (prompts, documents, embeddings), `@huggingface/transformers`, Pinecone, pdf-parse, multer, JWT/bcrypt/Google Auth, Vitest + Supertest.
**Frontend:** Next.js 16 (App Router), React 19, Redux Toolkit, Tailwind 4, shadcn/Radix, Framer Motion, Vitest.

## Project structure
```
AyuNidan backend/                 (git repo)
  src/
    routes/ controllers/          HTTP layer            middleware/   auth · cache · rate limit · validate · upload
    services/                     ai · assessment · document (patient RAG) · reference (verified KB) · rag (glossary)
    documents/                    pdfAnalyzer · tables · audio            rag/   chunking · embeddings · vectorStore · evidence
    prompts/                      LangChain prompt templates              models/ config/ utils/ validation/
    eval/                         synthetic evaluation harness            scripts/ seed · kb:ingest · demo helpers · migration
  tests/  eval/  medical-reference/  PROMPT_ENGINEERING_NOTES.md
AyuNidan frontend/                (git repo)  src/app · components · hooks · lib · store
docs/                             ARCHITECTURE · EVALUATION · HOW_IT_WORKS · RESUME_PROJECT(_AYUNIDAN) · INTERVIEW_DEFENSE · AYUNIDAN_INTERVIEW_HANDBOOK · DEMO_SCRIPT · archive/
```

## Environment variables
Copy `AyuNidan backend/.env.example` (fully documented). Required: `MONGODB_URI`, `JWT_SECRET` (≥32 chars), at least one of
`GEMINI_API_KEY` / `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`, `PINECONE_API_KEY`. Optional: `GOOGLE_CLIENT_ID` (Google sign-in),
`GEMINI_MODEL` (default `gemini-3.5-flash-lite`), `GEMINI_FALLBACK_MODEL`, `PINECONE_INDEX`, `TRUST_PROXY` (set `1` behind one proxy),
`NODE_ENV=production`, and the tuning knobs for timeouts, thresholds and limits.
Frontend (`AyuNidan frontend/.env`): `NEXT_PUBLIC_API_URL` (e.g. `http://localhost:8080/api`), `NEXT_PUBLIC_GOOGLE_CLIENT_ID`.

## Local setup
```bash
# Backend
cd "AyuNidan backend"
npm install
cp .env.example .env            # then fill in the values
npm run vector:setup            # one-time: create the 384-d cosine Pinecone index
npm run seed:rag                # one-time: seed the 20-term explainer glossary
npm run dev                     # API on http://localhost:8080

# Frontend (second terminal)
cd "AyuNidan frontend"
npm install
npm run dev                     # UI on http://localhost:3000
```
The first request downloads the ~34 MB embedding model (cached in `.cache/hf-models`; the server warms it at start-up).
Demo data (fictional only): `npm run demo:pdf` writes a fictional PDF; `npm run demo:refs` adds clearly-labelled **synthetic** reference passages
(start the API with `RAG_INCLUDE_SYNTHETIC_REFERENCES=true` to serve them; `npm run demo:refs -- --remove` deletes them). See `docs/DEMO_SCRIPT.md`.

## Running tests
```bash
cd "AyuNidan backend"  && npm test && npm run typecheck && npm run build     # 270 tests, no external services needed
cd "AyuNidan frontend" && npm test && npx tsc --noEmit && npm run lint && npm run build
```

## Running evaluation
```bash
cd "AyuNidan backend"
npm run eval                                   # offline: failures, tables, retrieval (hashing embedder)
npm run eval -- --embedder=hf --only=retrieval # local BGE model; sets/justifies retrieval thresholds (free)
npm run eval -- --live --models=google:gemini-3.5-flash-lite   # paid: extraction, risk classification, grounded answers
```
Reports are written to `eval/results/` (git-ignored).

## Known limitations
- **Not clinically validated**; risk labels are LLM-produced triage labels. No claim of clinical accuracy is made.
- **Reference coverage is narrow:** only two real guidelines (dyslipidemia, haemoglobin/anaemia) were ingested; clinical questions outside them return `insufficient_evidence`. The PDFs/datasets are not in the repository.
- **Relevance filtering is a similarity floor only:** unrelated chunks can be retrieved (e.g. lipid-guideline chunks for ECG findings); the model's abstention and citation validation keep them from being shown as used, but they still occupy prompt context.
- **Evaluation is small (hand-written probes and synthetic fixtures)**; retrieval thresholds may not transfer to other documents. LLM-quality metrics on the final pipeline were not completed (API quota).
- **Privacy:** extracted text is sent to the configured LLM provider; chunk text is stored in Pinecone metadata; browser dictation uses the browser vendor's speech service. Not HIPAA/GDPR-assessed.
- **Images/scans** are read by the multimodal model (no local OCR) and are flagged unverified; a PDF with any scanned page is attached whole to the model.
- Response cache and rate limiters are **in-process** (per instance). JWT is kept in `localStorage`; no refresh tokens.
- English-only embedding model. Voice fallback works on Gemini only. Reference ingestion is text-layer only.
- No load testing; no end-to-end browser tests (UI verified by build/typecheck/state-machine tests and manual API-level run).

## Future production improvements
Shared cache/rate-limit store (Redis); asynchronous ingestion queue with job status; parallel patient retrieval; refresh tokens + httpOnly cookies;
encryption/retention policy and audit log for PHI; managed reference-corpus governance and clinician-labelled evaluation; provider data-processing agreements;
secrets manager and CI; end-to-end browser tests.

## License / disclaimer
Portfolio prototype. Not for clinical use.
