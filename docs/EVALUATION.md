# AyuNidan — Engineering Evaluation

> ## ENGINEERING VALIDATION — NOT CLINICAL VALIDATION
> Everything below was measured on **synthetic / fictional fixtures** and on small samples. These results show that the
> software behaves as designed (parsing, retrieval, attribution, failure handling). They say **nothing** about clinical
> accuracy, diagnostic performance or real-world safety. Risk labels are engineering triage labels, not a validated
> clinical scale. Similarity scores are retrieval signals, not medical certainty.

All numbers are copied from runs actually performed during development (2026-09-28/29). Nothing is estimated.
Reproduce the offline parts with `npm run eval` and `npm run eval -- --embedder=hf --only=retrieval` (backend).

## 1. Test suites and build health (final state)

| Check | Result |
|---|---|
| Backend tests (Vitest, 12 files) | **201 / 201 pass** (no live Mongo/Pinecone/HF/LLM: in-memory fakes, hashing embedder, mocked AI SDK) |
| Frontend tests (Vitest, transcript state machine) | **7 / 7 pass** |
| Backend typecheck (`src` + `tests`) | clean |
| Frontend typecheck | clean |
| Backend build (`tsc`) | clean |
| Frontend production build (`next build`) | clean |
| Frontend lint | **0 errors, 9 warnings** — all pre-existing unused imports in `login/page.tsx`, `DashboardOverview.tsx`, `Header.tsx` (not touched by this work) |

Coverage themes of the backend tests: cross-user cache isolation and invalidation, error-response caching, endpoint protection,
upload hardening (spoofed MIME, traversal, size/count), JWT/config fail-fast, login rate limiting, request validation,
PHI-free logging, consultation lifecycle and failure semantics, AI fallback/timeout/circuit-breaker, patient RAG and tenant
isolation, prompt-injection escaping, reference ingestion/provenance/multi-source retrieval, citation validation,
grounded assessment, multimodal normalization, table extraction, audio validation, evaluation metrics and harness.

## 2. Failure categorization (offline, through production code)

**16 / 16** failure scenarios are categorized correctly (`runFailureEval`): all providers down, timeout, timeout-then-malformed,
malformed output, empty input, vector query outage, embedding outage, invalid chunking options, unanswerable question,
forged vector metadata, hallucinated citation, rate-limited provider (429), retired model (404), corrupted PDF,
unmappable table, invalid audio.

## 3. Table extraction (offline, deterministic; 4 fixtures, 8 rows)

Two PDF-layout tables (one with non-standard headers such as "Observed Value / Biological Reference Interval / Remarks"),
one delimited-text table (abbreviated flags H/N), and one negative case (no header ⇒ no table guessed).

| Metric | Score |
|---|---|
| Row precision / recall / F1 (8 expected, 8 extracted, 0 spurious) | 1.0 / 1.0 / 1.0 |
| Field accuracy: result, unit, reference range, flag, page | 1.0 each |

The first run scored 0.75 on reference range and flag: an over-long header cell overlapped the next column in the PDF text
layer and was merged. The evaluation exposed it; the cell-splitting rule was fixed and the fixture now passes.

## 4. Retrieval with the production embedding model (local BGE-small, 384-d, cosine)

Run: `npm run eval -- --embedder=hf --only=retrieval` (in-memory vector store, real local model; no paid calls).

**Patient documents** — 5 fictional documents, 10 answerable + 4 unanswerable questions.

| Metric | Value |
|---|---|
| Document Hit@1 / Hit@5 / MRR | 1.0 / 1.0 / 1.0 |
| Page-level hit (PDF-sourced questions) | 1.0 |
| Answerable top-1 similarity (min / median / max) | 0.7035 / 0.7191 / 0.7873 |
| Unanswerable top-1 similarity (min / median / max) | 0.4938 / 0.5856 / 0.6583 |
| Threshold sweep | 0.60–0.64: 1 false positive; **0.66–0.70: 0 FP / 0 FN**; 0.72: 5 FN; 0.78: 9 FN |

**Verified-reference retrieval** — 5 *synthetic* reference sources, 6 answerable (1 needs two sources) + 3 unanswerable questions.

| Metric | Value |
|---|---|
| Source Hit@4 / MRR | 1.0 / 1.0 |
| Multi-source full-recall (1 question) | 1.0 |
| Answerable top-1 similarity (min / median / max) | 0.7673 / 0.7952 / 0.8717 |
| Unanswerable top-1 similarity (min / median / max) | 0.4909 / 0.5216 / 0.5608 |
| Threshold sweep | **0.60–0.76: 0 FP / 0 FN**; 0.78: 2 FN; 0.80: 3 FN |

**Threshold selection.** Defaults were set from these sweeps, not by intuition: patient `RAG_DOC_MIN_SCORE=0.68`
(midpoint of the zero-error band 0.66–0.70) and reference `RAG_REF_MIN_SCORE=0.66` (near the midpoint of the gap between
unanswerable max 0.56 and answerable min 0.77). Both are configurable. The glossary threshold (`RAG_GLOSSARY_MIN_SCORE=0.7`)
was only **spot-checked** on 7 probe queries (e.g. "tachycardia" 0.84 accepted; "knee meniscus tear" 0.61 and "best pizza toppings"
0.41 rejected; "HbA1c" 0.67 rejected by the conservative default) and was *not* formally evaluated.

**Caveats.** The samples are tiny and the questions are keyword-friendly; a clean separation here does not imply the same
on real clinical documents. Thresholds must be re-evaluated whenever real reference documents are ingested or the embedding
model changes (the embedding space is recorded per chunk).

## 5. Timings

**Local embedding model** (BGE-small q8, CPU, measured on the development machine):

| Measurement | Time |
|---|---|
| First-ever load incl. ~34 MB model download | ≈ 16.8 s (one-time) |
| Cached model load | 375 ms (server start-up warm-up logged 840 ms) |
| First inference after load | 8 ms |
| 1 query + 2 short passages (one smoke test) | ≈ 27 ms |
| Batch of 32 chunks (~150 words each) | 661 ms |

**Live fictional end-to-end run #1** (service level, `gemini-3.5-flash-lite`, no fallback used):
extraction 3.0 s · patient indexing 4.3 s · assessment 5.4 s · grounded Q&A 3.0 s · total 18.3 s.
Result: 2 lab values from a PDF table with exact page provenance, 3 reference citations (synthetic sources), 2 patient-evidence citations, 0 dropped citations, other user retrieved 0 chunks.

**Live fictional HTTP run #2** (final ship check; real Express server, MongoDB Atlas, Pinecone, `gemini-3.5-flash-lite`, single run, n = 1 — not a benchmark):

| Step | Wall time | Breakdown (from `Server-Timing` / logs) |
|---|---|---|
| `POST /uploads` (2-page PDF with table) | 6.96 s | extraction 3.69 s · indexing 3.17 s (chunking 1 ms · local embedding 267 ms · Pinecone upsert 2.68 s) |
| `POST /consultations` (assessment) | 6.25 s | persist 0.1 s · assessment 5.84 s (model call 2.87 s + retrieval/Mongo round-trips ≈ 2.96 s) · persist 0.09 s |
| `POST /documents/query` | 2.48 s | embedding 103 ms · retrieval 311 ms · reference retrieval ≈ 325 ms · generation 1.61 s |
| Insufficient-evidence consultation (2 model calls) | 3.35 s | extraction 1.56 s · assessment 1.59 s |

Token usage of the model calls in that run: extraction 500 in / 367 out; assessment 1307 in / 547 out; Q&A 615 in / 139 out.
**Observed bottlenecks:** network round-trips (Pinecone upsert, sequential per-finding patient queries) and the model calls; local embedding and chunking are negligible.

## 6. What was verified end-to-end on the real stack (fictional data)

Register/login → PDF upload → structured report with table provenance → indexing → grounded assessment with **Medical References Used**
(synthetic sources, badged) → cited Q&A → dashboard → insufficient-evidence path → glossary explainer (grounded) → anonymous
upload rejected (401) → debug/seed routes absent (404) → second user got 404 on the first user's consultation, empty list,
`insufficient_context` on a forged document id, 404 on deleting the other user's document → per-user cache HIT/MISS correct →
server log contained **0** occurrences of patient text, tokens, passwords or keys (97 lines scanned). All demo users, records,
vectors and synthetic references were deleted afterwards.

### 6b. Real-reference integration run (2026-09-29, fictional patient, one live pass)

Corpus: ACC/AHA 2026 dyslipidemia guideline (631 chunks) + WHO 2024 haemoglobin/anaemia guideline (290 chunks). Semantic retrieval at floor 0.66 returned the right guideline pages for hypertriglyceridemia (top 0.84), ASCVD risk (0.82), LDL-C (0.82), lipoprotein(a) (0.81), triglyceride-rich remnants (0.73), ApoB (0.69 — close to the floor, kept unchanged) and the WHO adult-male cutoff (0.85); "capital of France" scored 0.46 (nothing returned) and "pediatric leukemia" scored 0.67 against unrelated paediatric-lipid chunks, so the model's abstention (not the floor) produced the refusal. Live behaviour: patient fact, terminology, guideline question, clinical question with reference, out-of-corpus question (`insufficient_context`), retrieved-but-unused references hidden (8 retrieved / 2 cited), planted instruction ignored, unrelated assessment `insufficient_evidence`, cross-user access denied. Single run, hand-written probe questions: an engineering check, not a benchmark or clinical validation.

## 7. Not evaluated (be explicit in interviews)

- **LLM-dependent quality on the final pipeline**: extraction F1 against gold, risk-class agreement/confusion matrix and grounded-answer
  correctness (`--live` harness modes exist) were **not completed** for the final pipeline — the Gemini API quota was exhausted (HTTP 429) during
  the first attempt and the comparison was not repeated. The harness computes these metrics but no numbers are reported here.
- **Image/OCR quality**: images are read by the multimodal model and flagged as model-transcribed; there is no accuracy measurement.
- **Live voice transcription quality**: depends on the browser's speech service; tested only as a state machine (7 unit tests).
- **Real reference corpus, real clinical documents, clinical outcomes**: none.
- **Load / concurrency**: no load testing was performed.

## 8. Post-freeze bug found in manual use (fixed)

Uploading a real scanned PDF crashed the API: pdf.js decodes JPEG images through the browser-only `Image` class, raising an uncaught
`ReferenceError` in Node that killed the process. The original fixtures used raw (uncompressed) images and never exercised that path.
Fix: an inert `Image` stub in `documents/pdfAnalyzer.ts` (we only count images), plus a regression test with a JPEG-(DCT)-in-PDF fixture
and a live upload check (page 2 `scanned`, page 3 `mixed`, server stayed up). Backend suite is now **201 / 201**.
Lesson: synthetic fixtures must reproduce the encodings of real inputs; this one only surfaced with real files.
