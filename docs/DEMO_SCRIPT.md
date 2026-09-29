# AyuNidan — Live Demo Script (≈5 minutes)

Tone: an engineer showing how the system stays honest — not a product pitch. Use **only fictional data**.

## Before you present (one-time, ~5 min; do a dry run first)

```powershell
# Terminal 1 — backend (uses your .env; first run downloads the ~34 MB embedding model)
cd "C:\Users\abhi4\AyuNidan\Ayunidan\AyuNidan backend"
npm run vector:setup            # only if the Pinecone index doesn't exist yet
npm run seed:rag                # only once: explainer glossary
npm run demo:pdf                # writes demo-assets\fictional-cbc-report.pdf
npm run demo:refs               # adds SYNTHETIC reference passages (labelled "synthetic test source")
$env:RAG_INCLUDE_SYNTHETIC_REFERENCES = "true"   # only so the synthetic demo references are served
npm run dev                     # wait for {"event":"server.started"} and {"event":"embeddings.warm"}

# Terminal 2 — frontend
cd "C:\Users\abhi4\AyuNidan\Ayunidan\AyuNidan frontend"
npm run dev                     # http://localhost:3000
```
Create a demo account in the UI (Register). Optional second account in a private window for the isolation moment.
Afterwards: `npm run demo:refs -- --remove` in the backend folder, and delete the demo consultation.

**Backup plan.** If the LLM API is rate-limited, the UI will show **AI FAILED / Not assessed** — that *is* a feature. Say so and move on to §4–5.
Have a recorded screenshot of a successful run ready anyway.

---

## 0:00–0:30 — Problem
"Clinical documents are messy — scanned PDFs, lab tables, photos, dictation. If you just paste them into an LLM you get three failure modes:
it invents medical facts and sources, an outage silently turns into a 'normal' result, and one patient's data can leak to another.
AyuNidan is a prototype built around those three failure modes. It's not a diagnostic tool and it isn't clinically validated."

## 0:30–1:00 — Architecture (show `docs/ARCHITECTURE.md` diagram, or sketch)
"Two jobs, deliberately separate. **Job A** understands the document: local PDF, table and scan analysis first, then one structured extraction call.
**Job B** assesses it — but only from retrieved evidence: the patient's own indexed chunks, plus a **separate** curated reference knowledge base.
Embeddings run locally with BGE-small, so retrieval costs no API calls. The model gets a small, labelled evidence set through a LangChain prompt.
Everything it returns is validated before it's stored or shown."

## 1:00–2:00 — Upload and extraction
1. New Consultation → drop `demo-assets\fictional-cbc-report.pdf`.
2. "The PDF has a text layer and a table. Parsing happened locally: page 2's table was detected and each row keeps its **page and table id**.
   The model was called once, for the narrative fields — not to read the table."
3. Point at the extracted text; edit one word. "The clinician reviews and can correct this before anything is assessed — extraction and assessment are separate steps."
4. Click submit.

## 2:00–2:45 — Evidence-grounded assessment
On the consultation page:
- **Extracted Lab Values**: "Hemoglobin 10.2, flag Low — and here's its provenance: file, page 2, *table*. A value the model extracted on its own would say 'model-extracted · location not verified'."
- **Evidence-Grounded Assessment**: key findings with **F / P / R** chips — finding, patient evidence, reference evidence. Read the *Uncertainty* line.
- **Medical References Used**: "Only sources that were retrieved *and* cited appear here, with title, organization, section — and page, version or URL only when they exist.
  These are clearly-badged **synthetic test sources**; the real reference corpus ships empty because I won't fabricate guidelines. Real sources are ingested with `npm run kb:ingest` and require provenance."
- Say once: "The risk label is an LLM triage label constrained by this evidence — not a validated clinical score."

## 2:45–3:30 — Q&A and citations
In **Ask About This Report**: *"What was the hemoglobin result and what can low hemoglobin be associated with?"*
- Answer separates "From your documents" (P) from "Medical references used" (R).
- "The model can only return evidence *ids*. The server checks them against what it was actually given; invented ids are dropped, and the citation details are rebuilt from retrieved metadata — so it can't invent a page or a URL."
- Then ask something not in the report (*"What is the patient's blood group?"*): "Insufficient evidence — it abstains rather than guesses."

## 3:30–4:00 — Security and isolation
"Isolation is enforced in three layers: a per-user vector namespace, a metadata filter, and a check against Mongo ownership records afterwards."
- Optional live proof (second account or curl): user B requesting user A's consultation → **404**; asking about A's document id → insufficient context.
- "I found a real bug in my own first version — the response cache wasn't keyed by user, so it could serve one user's data to another. It's fixed and has a regression test.
  The audit also removed unauthenticated LLM endpoints, a debug route, a fallback JWT secret and PHI in logs."

## 4:00–5:00 — Engineering decisions and limitations
- **Failure is visible**: "If the model fails, the consultation is stored as *failed* with the stage and category, and the UI shows AI FAILED — never 'low risk'. There's a fallback chain, timeouts and a circuit breaker; my primary Gemini model was actually retired mid-project, and this is what handled it."
- **Measured, not guessed**: "Thresholds came from a sweep on a synthetic set — 201 backend tests, 16/16 failure scenarios categorized, table fixtures fully correct. All of it is *engineering* validation on synthetic data."
- **Limitations, upfront**: no clinical validation; reference KB empty by default; LLM-quality metrics not completed (API quota); cache and rate limits are per-process; text goes to the LLM provider and chunk text sits in Pinecone; images are model-read, not OCR'd.
- **Next**: async ingestion queue, Redis for shared state, a governed reference corpus and clinician-labelled evaluation.

## Likely follow-up questions (answers in `INTERVIEW_DEFENSE.md`)
Why separate stores (Q4) · how are citations validated (Q21) · how do you stop prompt injection (Q20) · what if Pinecone/LLM is down (Q34/35) · how are thresholds chosen (Q9/10) · what would you change at 10× (Q32).
