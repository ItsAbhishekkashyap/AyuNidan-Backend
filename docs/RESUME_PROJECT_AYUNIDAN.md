# AyuNidan — Resume Content

All statements map to the final repository. No users, deployment, accuracy, business or clinical claims are made — AyuNidan is a portfolio **prototype**, not clinically validated. Numbers used: 31,378 terminology vectors (1,014 MedlinePlus + 30,364 MeSH), 921 guideline vectors (631 ACC/AHA + 290 WHO), 270 backend + 10 frontend tests, 384-d local embeddings.

## A. Project title
**AyuNidan — Evidence-Grounded Clinical Document Intelligence (RAG) Platform**

## B. One-line description
Full-stack GenAI system that extracts structured data from medical PDFs/images/voice and produces clinician-reviewable assessments and Q&A whose every claim is tied to validated citations from patient documents, NLM terminology and ingested clinical guidelines — or returns *insufficient evidence*.

## C. Three-bullet version
- **Built a multimodal document-processing backend** (Node/Express 5, TypeScript, MongoDB): page-aware PDF analysis with table and two-column handling, text-only PDFs are never sent to the model (scans, images and audio are), and extracted values/abnormal flags are verified against the source text or computed in code from printed ranges — so model-invented values are discarded; consultations follow an explicit `processing → completed | failed` lifecycle with categorized failures, timeouts, fallback model and circuit breaker.
- **Designed a three-domain RAG architecture on Pinecone with local BGE-small (384-d) embeddings:** tenant-isolated patient documents (namespace + metadata filter + database ownership check), a NLM MedlinePlus/MeSH terminology base (31,378 vectors, exact-synonym plus semantic retrieval), and verified clinical guidelines (ACC/AHA 2026 dyslipidemia, WHO 2024 — 921 chunks with page/section provenance), all ingested through deterministic-id, idempotent pipelines (streaming ZIP/XML parsing for the terminology datasets).
- **Implemented server-side grounding and citation validation:** evidence ids (F/P/R/T) are checked per domain, citation metadata comes only from retrieval, fabricated ids are stripped from lists and prose, uncited references are hidden, and a risk level requires a cited verified reference — otherwise a deterministic `insufficient_evidence`; verified by 270 backend tests (all external services faked) plus a live Atlas/Pinecone/Gemini run on fictional data.

## D. Four-bullet version
- **Built a multimodal ingestion backend** (Express 5, TypeScript, MongoDB) with page-aware PDF analysis, table extraction, two-column reading and model reading reserved for scans, images and audio; extracted values are verified against the document text and abnormal flags computed in code from printed ranges, so hallucinated values are dropped.
- **Designed a three-domain RAG system** (Pinecone namespaces, local BGE-small 384-d embeddings): tenant-isolated patient documents, a NLM MedlinePlus/MeSH terminology base (31,378 vectors, exact-synonym + semantic retrieval) and verified guidelines (ACC/AHA 2026, WHO 2024 → 921 chunks with page/section provenance), ingested with deterministic ids so re-runs never duplicate vectors.
- **Enforced grounding in code, not prompts:** per-domain F/P/R/T citation validation (structured and inline), metadata built only from retrieved evidence, retrieved-but-uncited references hidden, and a deterministic `insufficient_evidence` unless a verified reference is actually cited; prompt-injection defended by escaping, data blocks and output validation.
- **Hardened reliability and security:** JWT auth with owner-scoped queries (404 on foreign ids), magic-byte upload validation, layered rate limits, PHI-free structured logs, timeouts/fallback model/circuit breaker with an explicit `failed` state; 270 backend + 10 frontend tests with faked externals, plus a live real-stack verification and fixes for real-PDF failures (pdf.js crashes, column interleaving).

## E. Two-bullet compact version
- **Built a grounded clinical-RAG backend** (Express/TypeScript, MongoDB, Pinecone, local BGE-small embeddings, Gemini) over three isolated knowledge domains — patient documents (tenant-isolated), NLM MedlinePlus/MeSH terminology (31,378 vectors) and ingested clinical guidelines (921 chunks with page/section provenance) — with idempotent deterministic-id ingestion and multimodal PDF/image/voice extraction whose values are verified against the source.
- **Enforced evidence grounding server-side:** per-domain citation validation, metadata from retrieval only, hidden uncited references, deterministic `insufficient_evidence` without a cited verified reference, a `processing→completed|failed` lifecycle with fallback/circuit breaker, and 270 backend tests plus a live Atlas/Pinecone/Gemini run.

## F. Tech stack line
TypeScript, Node.js, Express 5, Next.js 16/React 19, MongoDB (Mongoose), Pinecone, Hugging Face transformers.js (BGE-small), Gemini via Vercel AI SDK, LangChain Core (prompts), Zod, pdf.js, JWT, Vitest/Supertest.

## G. Interview defense

### Bullet 1 — Multimodal document-processing backend
- **Means:** uploads are validated, PDFs analysed locally page by page (text, tables, image operators); text-only PDFs are never sent to the model; scans, images and audio are read by the multimodal model. A lab value is kept only if it is found in the document text or a parsed table; flags come from `flagFromRange` against the printed range unless the report prints one. Failure is a state, not a fake result.
- **Where:** `src/documents/{pdfAnalyzer,tables,grounding,ranges,identity}.ts`, `src/services/ai.service.ts` (`collectSources`, `normalizeExtraction`, `generateStructured`, breaker/fallback), `src/controllers/{upload,consultation}.controller.ts`, `src/utils/failures.ts`.
- **They'll ask:** "How do you know the extraction isn't hallucinated?" / "What happens when Gemini is down?"
- **Answer (20–30 s):** "Extraction is one structured call, but I don't trust it. Each value must appear in the document's own text or in a parsed table row; anything else goes into an 'unverified' list and is excluded from the assessment. Abnormal flags are computed in code from the range printed in the report. If the model or provider fails, I retry once, fall back to a second Gemini model behind a circuit breaker, and if that fails the consultation is marked `failed` with a category — never a made-up 'low risk'."

### Bullet 2 — Three-domain RAG on Pinecone with local embeddings
- **Means:** three knowledge domains with different trust levels, each in its own namespace and id space, embedded locally (bge-small, 384-d, cosine), with isolation for patient data and idempotent ingestion for the shared domains.
- **Where:** `src/rag/{vectorStore,embeddings,chunking,evidence}.ts`, `src/services/{document,reference,terminology}.service.ts`, `src/terminology/*` (streaming ZIP reader + parsers), `src/scripts/{ingest-references,ingest-terminology}.ts`.
- **They'll ask:** "Why three domains?" / "How is tenant isolation enforced?" / "Why local embeddings?"
- **Answer:** "A dictionary definition, a patient's lab value and a guideline recommendation have different trust levels, so mixing them lets a definition pass as a guideline. Separate namespaces and ids keep them apart. Patient isolation is three layers: per-user namespace, a metadata filter, and a Mongo ownership check on returned chunks. Embeddings are local so there's no per-call cost and patient text isn't sent out for embedding, and tests run offline. Ingestion uses deterministic ids, so re-running overwrites instead of duplicating — I verified 921 reference vectors after a forced rebuild."

### Bullet 3 — Server-side grounding and citation validation
- **Means:** the model returns evidence ids; the server validates them against exactly what it supplied, per domain, including ids written inline in prose; citation objects are built from stored metadata; risk levels are downgraded deterministically.
- **Where:** `src/rag/evidence.ts` (`validateCitations`, `toCitation`, inline-citation helpers), `src/services/assessment.service.ts` (`validateAssessment`, downgrade reasons), `src/services/document.service.ts` (`answerFromEvidence`), `src/rag/routing.ts`, `tests/{terminology.grounding,ecg.cleanup,reference.assessment}.test.ts`.
- **They'll ask:** "How do you prevent hallucinated citations?" / "What if retrieval returns irrelevant chunks?"
- **Answer:** "I don't rely on the prompt. Ids must exist in the evidence I sent and belong to the right domain; page, section and organization come from retrieval metadata, not the model; invalid ids are stripped from both the lists and the text. A risk level needs at least one *cited* verified reference, otherwise it becomes `insufficient_evidence` with a recorded reason. Irrelevant chunks can still be retrieved — I saw lipid-guideline chunks for ECG findings — but if the model doesn't cite them they're not shown as used and the result is insufficient evidence. A relevance filter is the known next step."

### Bullet 4 (four-bullet version) — Reliability, security, testing
- **Means:** owner-scoped data access, upload sniffing, rate limits, PHI-free logging, timeouts/fallback, and a test strategy that fakes every external service, plus one deliberate live run.
- **Where:** `src/middleware/{auth,rateLimiter,uploadValidation,cache}.ts`, `src/utils/{logger,failures,timing}.ts`, `tests/*` (16 files), `docs/EVALUATION.md`.
- **They'll ask:** "How do you test RAG without paying for models?" / "What did real documents break?"
- **Answer:** "Every external dependency is replaced — in-memory vector store with the same filter semantics, a hashing embedder, mocked model calls — so 270 tests run offline. Then I ran one live pass against Atlas, Pinecone and Gemini with fictional data. Ingesting the real guidelines exposed a pdf.js font crash, two-column text interleaving and bibliography chunks polluting retrieval; each got a fix and a regression test."

### Two-bullet version
Both bullets are compressions of the above; use the answers for Bullets 2 and 3, with Bullet 1's failure-handling answer for the "lifecycle/fallback/breaker" clause.

## H. Final recommendation (best 3 bullets)
- **Built a multimodal clinical-document backend** (Express 5/TypeScript, MongoDB) that parses text-layer PDFs locally (tables, two-column layouts) and never sends text-only PDFs to Gemini (only scans, images and audio); extracted values must be found in the source and abnormal flags are computed from printed ranges, so hallucinated values are discarded, and every consultation ends in an explicit `completed` or categorized `failed` state (timeouts, fallback model, circuit breaker).
- **Designed a three-domain RAG system on Pinecone with local BGE-small embeddings** — tenant-isolated patient documents (namespace + filter + ownership check), a NLM MedlinePlus/MeSH terminology base (31,378 vectors) and ingested clinical guidelines (ACC/AHA 2026, WHO 2024; 921 chunks with page/section provenance) — using deterministic-id, idempotent ingestion (streaming parse for the terminology datasets).
- **Enforced grounding in code rather than prompts:** per-domain citation validation (structured and inline), provenance taken only from retrieved metadata, uncited references hidden, and a deterministic `insufficient_evidence` unless a verified reference is cited; backed by 270 backend tests with faked externals and a live Atlas/Pinecone/Gemini verification on fictional data.
