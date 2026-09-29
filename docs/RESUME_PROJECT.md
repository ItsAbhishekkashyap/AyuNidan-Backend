# AyuNidan — Resume Material

Every statement below is backed by the repository or by `docs/EVALUATION.md`. Numbers are from synthetic fixtures — say so if asked.
**Do not claim:** clinical validation, diagnosis, deployment, "FastAPI" (the backend is **Express + TypeScript**), or any LLM-quality metric
(extraction F1 / risk accuracy were not completed on the final pipeline). Keep "synthetic" wherever a retrieval/table number appears.

Format of each bullet: **Problem → Engineering decision → Implementation → Result.**

---

## VERSION A — AI / ML / GenAI

**AyuNidan — Evidence-Grounded Clinical Document Intelligence (RAG prototype)**
*Multimodal extraction and cited, abstention-aware assessment over patient documents and a separate curated reference KB.*

- LLM summaries of medical reports can invent facts and sources, so I made the model a synthesizer over retrieved evidence rather than a knowledge source: two separate vector domains (per-user patient namespaces and a shared, provenance-required reference KB), local BGE-small embeddings, a LangChain prompt layer with delimited and escaped evidence, and server-side validation of every citation id the model returns. Fabricated ids are dropped, an uncited result is downgraded to `insufficient_evidence`, and reference citations are rebuilt only from retrieved metadata.
- Flattening PDFs to text destroys lab tables, so I parse pages locally (text layer, layout-based table detection, scanned/embedded-image detection), keep each table row atomic with `{file, page, table}` provenance, and send only unreadable content to the multimodal model in one schema-constrained call, separate from the assessment call. On four synthetic table fixtures, 8/8 rows and every field (result, unit, range, flag, page) were extracted correctly, and text-layer PDFs are never sent to the LLM as files.
- Retrieval thresholds are usually guessed, so I built a synthetic evaluation harness (retrieval Hit@K/MRR, similarity distributions, false-positive/negative sweeps, table accuracy, 16 categorized failure scenarios) and set thresholds from measured score gaps. On the local BGE model the patient set showed Hit@1 = 1.0 (n = 10) with a zero-error band at 0.66–0.70; the harness also caught a table-column merge bug (0.75 → 1.0 accuracy). Documented explicitly as engineering validation, not clinical validation.
- LLM outages must not look like clinical results, so consultations follow a persisted `processing → completed | failed` lifecycle with categorized failures, timeouts, a model-fallback chain and a circuit breaker (404/401/429), and the UI renders failures as "AI FAILED / Not assessed". A failed AI step is never stored as low risk, and 201 automated tests run without any external service.

## VERSION B — Backend / Software Engineering

**AyuNidan — Secure, Failure-Aware Clinical Document API (Express, TypeScript, MongoDB, Pinecone)**
*Multi-tenant document-processing API with hardened auth, validated uploads and an observable AI pipeline.*

- A security audit of my own first version found a response cache keyed without the user (any logged-in user could receive another user's patient list and cached records), unauthenticated paid-LLM upload endpoints, a debug route leaking key prefixes, a hardcoded JWT fallback secret and PHI in logs. I fixed each: user-scoped, success-only cache with mutation invalidation; auth and rate limiting before multipart parsing; fail-fast Zod-validated config; and structured logs of metadata only. Regression tests for each (201 passing) assert the leak cannot recur.
- Tenant isolation had to hold even if one layer failed, so patient vectors use three independent layers — per-user namespace, metadata filter on user and allowed documents, and post-query validation against Mongo ownership records (deleted, foreign or other-embedding-space vectors are dropped). Tests prove user B cannot retrieve, cite, list or delete user A's documents, even with forged document ids or planted vector metadata.
- Uploads and AI calls are the failure-prone edges, so uploads are validated by size, count, MIME, extension and magic bytes, and every model call has a per-call timeout, a total budget, bounded retries, provider fallback and a circuit breaker. Consultation input is persisted before processing, so a failure keeps the clinician's data and records a stage and category (20 failure categories; 16/16 scenarios classified correctly).
- Latency and cost were opaque, so I added request-id correlation (AsyncLocalStorage), `Server-Timing` headers, per-call token/latency metrics and a stage timer. A measured run showed the Pinecone upsert at 2.7 s of a 3.2 s indexing step and roughly half of the assessment time in retrieval round-trips, while local embedding took under 0.3 s, which identifies where to optimize before scaling.

## VERSION C — Full-Stack + AI

**AyuNidan — Clinical Document Intelligence Dashboard (Next.js 16, Express, RAG)**
*End-to-end product: upload or dictate, review, get a cited assessment, and ask grounded questions about a report.*

- Clinicians need to trust and correct AI output, so the workflow is human-in-the-loop: extraction is a separate step from consultation creation, the clinician edits it, and the dashboard shows the full structured report (tables with page provenance, lab flags, dates, limitations), the assessment with key findings tagged to evidence, patient evidence, and a "Medical References Used" panel listing only sources that were actually retrieved and cited, with title, organization, section, page and URL when known.
- Live dictation is easy to fake, so I built a real interim-vs-final transcript state machine on the browser Web Speech API — interim text is shown in grey but only finalized text enters the clinical input — with a validated server-side fallback for browsers without speech recognition and explicit messages for permission, no-speech, network and cancellation errors plus a 5-minute recording cap. The state machine is covered by 7 unit tests.
- Failure and uncertainty had to be visible in the UI, so consultations carry `completed | failed` state and a fourth risk value, `insufficient_evidence`. The dashboard shows "AI FAILED / Not assessed" and "Insufficient evidence" as distinct states and never as low risk, and it excludes them from the risk distribution.
- The AI pipeline is retrieval-first: local BGE-small embeddings, separate patient and reference vector stores, LangChain prompt templates, one structured LLM call per job, and citation validation. It is backed by a synthetic evaluation harness and 201 backend + 7 frontend tests, with typecheck and production builds clean.

---

### One-line variants
- **A:** Built a multimodal RAG pipeline that grounds clinical-document assessments in retrieved patient and reference evidence, validates every citation server-side, and abstains (`insufficient_evidence`) instead of guessing.
- **B:** Hardened an Express/TypeScript clinical-document API after a self-audit (cross-tenant cache leak, unauthenticated LLM endpoints, PHI logging) with three-layer vector tenant isolation, failure-aware AI processing and 201 tests.
- **C:** Shipped a Next.js + Express clinical document dashboard with live dictation, provenance-tracked structured reports, cited grounded Q&A and visible failure/uncertainty states.

### Skills these bullets evidence
RAG design · embeddings/vector search · structured LLM output · prompt-injection defence · multimodal document parsing · multi-tenant isolation · API security · failure-mode design · observability · evaluation methodology · TypeScript/Node · Next.js/React · Pinecone · MongoDB · LangChain (prompts/embeddings abstractions) · Vitest.
