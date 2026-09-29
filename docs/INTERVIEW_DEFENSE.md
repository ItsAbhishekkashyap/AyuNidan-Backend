# AyuNidan — Interview Defense Pack

Answers are grounded in the actual implementation (see `ARCHITECTURE.md`, `EVALUATION.md`). Where something is a limitation, say so — it
is more credible. Stack reminder: **Express 5 + TypeScript backend, Next.js 16 frontend, MongoDB, Pinecone, local BGE embeddings, Gemini via Vercel AI SDK, LangChain (prompts/embeddings abstractions).**

**Numbers to know:** 201 backend + 7 frontend tests · 16/16 failure scenarios · table fixtures 8/8 rows, all fields correct · patient retrieval Hit@1 = 1.0 (n=10, synthetic) · reference Hit@4 = 1.0 (n=6, synthetic) · thresholds 0.68 patient / 0.66 reference · BGE-small 384-d, ~34 MB, 375 ms cached load, 661 ms per 32 chunks · chunking 1000/150 · evidence budget ≤6 patient + ≤8 reference chunks · timeouts 45 s/call, 120 s total, ≤3 models, ≤1 SDK retry · rate limits AI 10/min, login 5/15 min/account.

---

## Overview

**1. Explain AyuNidan in 60 seconds.**
It's a clinical document-intelligence prototype. A clinician uploads a report — PDF with tables, a scan, a photo — or dictates a note. I parse what I can locally, make one structured LLM call to extract a report with page/table provenance, and the clinician reviews it. Then a second, separate call produces a preliminary assessment, but only from retrieved evidence: the patient's own indexed document chunks plus a separately curated medical-reference knowledge base. Every citation the model returns is validated server-side, weak retrieval or missing evidence produces `insufficient_evidence`, and any failure is stored and shown as a failure — never as "low risk". It's not a diagnostic tool and isn't clinically validated; the engineering focus is grounding, isolation and failure behaviour.

**2. Explain the architecture.**
Next.js client → Express API (auth, rate limits, Zod validation, upload validation) → **Job A**: local PDF/table/scan analysis + one schema-constrained extraction call → normalized documents → chunking → local BGE embeddings → Pinecone (per-user namespace for patients, a shared namespace for verified references) with Mongo holding ownership/registry records → **Job B**: retrieval from both domains, evidence budgeting, LangChain prompt, one structured call, citation validation → Mongo `consultations` (`processing → completed | failed`) → dashboard with sources. See the diagram in `ARCHITECTURE.md`.

**3. Why RAG?**
The model shouldn't be the source of medical knowledge or of patient facts. RAG lets me (a) keep private patient documents out of the model except for the few relevant chunks, (b) make claims attributable to a source the user can inspect, (c) update the knowledge base without retraining, and (d) control cost by sending a small evidence set instead of whole reports.

**4. Why separate patient and medical-reference vector stores?**
Different trust, lifecycle and tenancy. Patient data is private, per-user and deletable; reference data is shared, curated and needs mandatory provenance. Mixing them risks leaking patient text into another user's context, lets a patient document masquerade as a "verified reference", and forces one threshold on two very different corpora. Separate namespaces + a `domain` metadata field + a registry let me retire a reference source or delete a patient document independently, and they show up differently in the UI ("Medical References Used" vs "Patient Evidence").

## Embeddings and retrieval

**5. Why local Hugging Face embeddings?**
No embedding traffic to a paid API: no per-call cost or quota (I hit LLM quota limits during development, which validated the choice), lower latency, deterministic behaviour, and patient text isn't sent to a third-party embedding service. The model runs in-process via `@huggingface/transformers` (ONNX, q8). Honest caveat: chunk text is still stored in Pinecone metadata, and the LLM provider still sees the evidence I send it.

**6. Why BGE-small?**
It's small (~34 MB quantized), loads in ~0.4 s cached, embeds 32 chunks in ~0.66 s on CPU, and is a strong retrieval model for its size with a documented query-instruction prefix I use for queries. The trade-off: English-only and smaller than the top models. I didn't benchmark alternatives on my data, so I don't claim it is the best — I claim it fits the constraints, and the model is configurable via env.

**7. Why 384 dimensions?**
That's BGE-small's native output size; the Pinecone index must match it (cosine, 384-d). Smaller vectors mean cheaper storage and faster search. Because vectors from different models aren't comparable, every chunk stores its `embeddingSpace` (`Xenova/bge-small-en-v1.5@384`) and retrieval ignores mismatches — changing the model means re-ingesting, and the code can't silently mix spaces.

**8. How does retrieval work?**
The query (or up to 6 deterministic, abnormal-first finding queries) is embedded with the same model, sent to Pinecone with a namespace and metadata filter (`domain`, `userId`, allowed `documentId`s), and each returned chunk's metadata is validated. A **per-chunk** similarity threshold is applied — not "top-K is non-empty" — then results are de-duplicated, ranked and cut to a character budget. Patient and reference retrieval are separate queries whose results are merged into one labelled evidence set (P#, R#).

**9. How are similarity thresholds chosen?**
From measurement. I built an evaluation set with answerable and unanswerable questions, ran the real local model, and swept the threshold. For patient documents the answerable top-1 scores started at 0.70 and unanswerable topped out at 0.66, with a zero-error band at 0.66–0.70, so I set 0.68. For references the gap was 0.56–0.77 and I set 0.66. Both are env-configurable. The glossary's 0.7 is only spot-checked.

**10. Why are the current thresholds NOT clinical validation?**
They were tuned on a tiny, synthetic, keyword-friendly set (10 + 6 answerable questions). They show the pipeline separates related from unrelated text, not that retrieved text is clinically relevant or true. Similarity is a retrieval signal, never medical certainty, and thresholds must be re-evaluated on real documents and whenever the embedding model or corpus changes.

**11. How are documents chunked?**
Deterministically: normalize whitespace, then 1000-character windows with 150 overlap, preferring paragraph → line → sentence → word boundaries in the last 40% of the window. Chunks never span pages, so a citation's page number is real. 1000 chars ≈ 250 tokens, comfortably under BGE's 512-token limit. Sizes are engineering defaults, not medically tuned.

**12. How are tables handled?**
Deterministically, not by asking the LLM. PDF text items are grouped into lines by position and into cells by gaps; a header row must map to a *test* and a *result* column (role detection incl. variants like "Observed Value" or "Biological Reference Interval") or I refuse to call it a table. Rows become typed records (test, result, unit, range, flag, date, page). Each row is indexed as `Hemoglobin: Result = 10.2 g/dL, Reference range = 13-17 g/dL, Flag = Low, Page = 2` and rows are never split across chunks. Table values override model-extracted duplicates because they have exact provenance. The evaluation caught one bug — an over-long header overlapped the next column — which I fixed.

**13. How are scanned PDFs handled?**
Locally I count image-paint operators and text characters per page: `text`, `mixed` (text + embedded images), `scanned` (images, no text) or `empty`. If any page can't be read locally, the PDF is attached for the multimodal model, and the prompt marks those pages "NO TEXT LAYER". Notes go to `extractionLimitations`; I never silently drop a page. Limitation: the whole PDF is attached, the output is model-transcribed and unverified, and there's no local OCR.

**14. How are images handled?**
Sent natively to the multimodal model in the extraction call. Findings are flagged "model-transcribed, unverified", indexed as an `ai_transcription` document with no invented page numbers, and if the model can't read something it reports that in `extractionLimitations` instead of guessing. I have no accuracy measurement for image extraction.

## Voice

**15. How does voice transcription work?**
In browsers with the Web Speech API, dictation is genuinely live: interim results stream into a state machine, and the recognizer is restarted after silence until the user stops (5-minute cap). Otherwise the app records audio, the server validates it (size, WAV header, duration) and Gemini transcribes it within the extraction call — final-only, and the UI says it isn't live. Errors (permission, no speech, network, empty audio, provider failure) are explicit.

**16. Why only the final voice transcript?**
Interim text is provisional — words change or vanish as recognition improves — and misheard fragments would pollute clinical input. The UI shows interim text in grey; only finalized text is committed, and cancelling discards everything. (Also: browser dictation goes through the browser vendor's speech service — a privacy point I disclose.)

## LangChain and prompting

**17. Why LangChain?**
For three things it does well: `ChatPromptTemplate` (reusable, sectioned prompts with variables kept separate from instructions), the `Document` type for reference ingestion metadata, and the `Embeddings` interface my local model implements. It's `@langchain/core` only.

**18. Why not LangChain everywhere?**
Where correctness matters I wanted explicit code: tenant filters, metadata validation and citation checks are custom and tested, and the Vercel AI SDK gives me typed structured output with abort signals, token usage and per-call fallback. I removed `langchain`, `@langchain/openai` and `@langchain/pinecone` because nothing used them. Fewer abstractions means fewer places for a tenant-isolation bug to hide.

**19. How does the final LLM receive context?**
Through a LangChain template: a system message (role, task, grounding rules, output requirements, data-handling rules) and a human message with `<patient_findings>` (numbered F#), `<patient_evidence>` (P#), `<verified_medical_evidence>` (R# with title/organization/section/page attributes) and `<user_query>`. All variable content is HTML-escaped. Evidence is budgeted (≤6 patient, ≤8 reference chunks) — the model never sees the whole report, the whole KB or raw files it doesn't need. The patient's name is not included.

**20. How do you prevent prompt injection from retrieved documents?**
Layers: (1) instructions and data are separated, and the system prompt says content in data blocks is never an instruction; (2) escaping means a document can't close or forge a block (tests use `</evidence><evidence id="R9">` payloads); (3) output is schema-constrained; (4) citations are validated so an injected "cite R9" is dropped; (5) the model has no tools and no side effects, so the worst case is a wrong or refused answer. It isn't a guarantee against every attack — it reduces and contains the blast radius.

**21. How do you prevent hallucinated citations?**
The model can only return evidence *ids*. The server checks each id against the evidence it actually supplied; unknown ids are dropped and counted (`citation_failure`). The citations sent to the client are rebuilt from retrieved metadata, so the model can't invent a title, page or URL. If nothing valid is cited, the assessment becomes `insufficient_evidence` and Q&A returns an explicit insufficiency. Limitation: this proves a citation exists in the context, not that the cited text truly supports the claim — that's a faithfulness evaluation I haven't done.

**22. What happens when evidence is insufficient?**
It's a first-class outcome. The prompt tells the model to return `insufficient_evidence`; code enforces it when the model reports it or cites nothing valid. It's stored (`riskLevel: insufficient_evidence`, no score), shown in its own UI state, and counted separately from low/medium/high on the dashboard. In my final run an administrative-note-only consultation returned exactly that.

## Security and multi-tenancy

**23. How is multi-tenancy enforced?**
The user id comes only from the verified JWT. Mongo queries filter on `userId`; foreign records return 404. For vectors, three independent layers: a per-user Pinecone namespace, a metadata filter (`domain`, `userId`, allowed `documentId`s), and post-query validation against the caller's Mongo document records and embedding space — so deleted, forged or planted vectors are dropped. Tests cover forged document ids and planted metadata.

**24. How did you prevent the cache leak?**
My own audit found the response cache keyed on method + URL only, mounted before the ownership check — so user B could receive user A's cached list, dashboard or record. Fixes: key = `user:<id>:<url>`; cache only 2xx and `success !== false`; invalidate on create/delete; `Cache-Control: private, no-store`; authGuard always runs first. A regression test exercises A-then-B on list, dashboard and by-id. Limitation: still per-process (needs Redis to scale).

**25. How is authentication implemented?**
Email/password with bcrypt, or Google Sign-In with server-side ID-token verification (audience + `email_verified`). Sessions are HS256 JWTs (default 7 days). The secret comes from validated config and the server won't boot without a ≥32-character secret — the old fallback string is gone (a test proves tokens signed with it are rejected, as are `alg:none` tokens). The guard also confirms the user still exists on each request. Login has per-IP and per-account rate limits. Weakness: the JWT is in `localStorage` and there's no refresh flow.

## Reliability

**26. How are failed AI jobs handled?**
Input is persisted first as `processing`. On any failure the record becomes `failed` with `{stage, category, attempts}`, summary/risk are unset, the API returns 502/503 with the record, the UI shows "AI FAILED / Not assessed", and failed items are excluded from risk statistics. Nothing is ever defaulted to "low". There's a migration script for legacy records that were saved as fake low-risk results.

**27. How did you handle model failures?**
Ordered fallback (Gemini primary → Gemini fallback → OpenAI/Anthropic if keys exist), per-call timeout (45 s), total budget (120 s), ≤3 models, ≤1 SDK retry, no same-model retry on schema failures, and error classification (timeout, rate limit, model unavailable, schema, provider). When my primary Gemini model turned out to be retired (404) and later quota-limited (429), this is what kept the system honest.

**28. How does the circuit breaker work?**
In-process map keyed by `provider:model`. A 404 opens the circuit for 30 minutes, 401/403 for 10, 429 for the `Retry-After` value or 60 s. While open the model is skipped, so we stop hammering a dead model and fail fast if all are open. 5xx and timeouts don't open it. Resets on restart and isn't shared across instances.

## Cost and performance

**29. How did you reduce API cost?**
Local PDF text/table parsing (text PDFs go as text, never as files), local embeddings, one call per job (extraction, assessment, Q&A) instead of per chunk or per source, an evidence budget instead of whole reports, extraction input cap (60k chars), low temperature, AI endpoint rate limits, and token logging so cost is observable. Honest note: extraction is still one model call per upload.

**30. How did you measure performance?**
A `StageTimer` and `Server-Timing` header on upload/consultation/query responses, structured logs with latency and token counts per model call, an in-process p50/p95 metrics buffer, and the evaluation harness. Real measurements are in `EVALUATION.md` (e.g. embedding 32 chunks 661 ms; upload 7.0 s; assessment 6.3 s; Q&A 2.5 s — single run, not a benchmark).

**31. What are the main bottlenecks?**
Network and model latency, not compute. In the final run, the Pinecone upsert was 2.7 s of the 3.2 s indexing step, and about half of the 5.8 s assessment was retrieval/Mongo round-trips (the model call was 2.9 s). Patient queries run sequentially per finding. Local embedding (~0.27 s) and chunking (~1 ms) are negligible. Pinecone is also eventually consistent, so queries right after upsert can miss.

**32. What would you change at 10x scale?**
Move ingestion to an async queue with job status; Redis for cache, rate limits and circuit-breaker state; parallelize patient retrieval; put embeddings in a separate worker so model load/CPU doesn't share the API process; per-tenant quotas; a real metrics/tracing backend; refresh tokens and httpOnly cookies; and re-evaluate thresholds and chunking on real data.

## Infrastructure and failure modes

**33. Why Pinecone?**
Managed serverless vector search with native namespaces and metadata filters — exactly the primitives for tenant isolation and domain separation — and it was already in the project. Trade-offs: an external service holding chunk text, eventual consistency, and vendor lock-in. Atlas Vector Search or pgvector would co-locate data and simplify PHI governance; I'd evaluate that for production.

**34. What happens if Pinecone is unavailable?**
Uploads still succeed: the document is marked `failed` with a category and extraction is returned. Assessment continues with the missing domain reported as unavailable (it can end as `insufficient_evidence`). Patient-document Q&A returns a categorized 502 rather than an invented answer. Calls have a 15 s timeout and ≤1 retry.

**35. What happens if the LLM is unavailable?**
Fallback models are tried; if all fail (or circuits are open) the consultation is stored as `failed` with the category, the input is preserved, the user sees "AI FAILED / Not assessed", and the request returns 502/503. Extraction endpoints return a categorized error. Nothing is fabricated.

**36. What are the main failure modes?**
Provider outage/timeout/quota/retired model; invalid structured output; embedding-model load failure; vector-DB outage or read-after-write lag; scanned/handwritten content misread by the model; table layouts my detector doesn't recognise (falls back to text and is noted); prompt injection; thresholds that don't transfer to real data; per-instance cache/rate limiter under multiple instances; and PHI exposure to third-party providers.

## Reflection

**37. What did you learn from the security audit?**
Bugs cluster at boundaries you forget are boundaries: a cache in front of authorization, "temporary" debug/seed routes, fallback secrets, and logging that "helps debugging". The fix pattern was to make the safe thing structural — config that won't boot unsafe, keys that can't omit the user, logs that can't take content, validation before parsing — and to write a regression test per finding.

**38. What would you improve for production?**
Clinical governance first (validated reference corpus, clinician-labelled evaluation, PHI retention/audit/encryption policy, provider agreements), then engineering: async ingestion, shared state, parallel retrieval, faithfulness evaluation of grounded answers, load testing, browser E2E tests, secrets management and CI.

**39. Why is this NOT a medical diagnosis system?**
It has no clinical validation, its risk labels are LLM-produced triage labels constrained by supplied evidence, similarity scores aren't certainty, its reference corpus is empty by default, and its evaluation is synthetic. It's designed to assist a clinician's review, always states uncertainty and disclaimer, and abstains when evidence is missing.

**40. What is the strongest engineering decision you made?**
Treating the model as an untrusted component: nothing it returns becomes state or is shown until it is validated — schema, citation ids against supplied evidence, risk/score consistency — and every failure or lack of evidence is an explicit, stored, visible state instead of a default value. It's the difference between a demo that looks right and a system whose failures are detectable, and it's what let me prove properties in tests (no fabricated citations reach the client; an outage never becomes "low risk").

---

### Traps to answer honestly
- *"What's your accuracy?"* — I don't have LLM-quality metrics on the final pipeline (API quota); I have retrieval, table and failure-handling numbers on synthetic data, labelled as such.
- *"Is it FastAPI?"* — No, Express 5 + TypeScript.
- *"Where is the medical knowledge?"* — Nowhere by default: the reference KB is empty until authorised sources are ingested; demo references are synthetic and badged.
- *"Is patient data private?"* — Isolated between users, but the text goes to the LLM provider and chunk text sits in Pinecone metadata; not HIPAA/GDPR-assessed.
