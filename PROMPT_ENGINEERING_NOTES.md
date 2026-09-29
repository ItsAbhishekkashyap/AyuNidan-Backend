# Prompt Engineering Notes (current implementation)

Prompts live in `src/prompts/` as LangChain `ChatPromptTemplate`s. Structured output is enforced with
Zod schemas through the Vercel AI SDK (`generateText` + `Output.object`) — there is no regex/`JSON.parse`
of free text anywhere in the pipeline. Model: `GEMINI_MODEL` (default `gemini-3.5-flash-lite`) with the
configured fallback chain (see `services/ai.service.ts`).

| Prompt | File | Job | Temperature | Output schema |
|---|---|---|---|---|
| Extraction | `prompts/extraction.ts` | **Job A** – document understanding: demographics, symptoms, medicines, labs (+unit/range/flag/date), dates, diagnoses *written in the document*, measurements, image-derived findings, doctor notes, extraction limitations | 0.1 | `ExtractionSchema` |
| Clinical assessment | `prompts/clinical-assessment.ts` | **Job B** – evidence-grounded triage: `riskLevel` (low/medium/high/insufficient_evidence), score, summary, key findings with evidence ids, uncertainty | 0.1 | `AssessmentSchema` |
| Medical Q&A | `prompts/medical-qa.ts` | Grounded answer over retrieved patient (`P#`) and reference (`R#`) evidence | 0.1 | `AnswerSchema` |

## Design rules
1. **Data ≠ instructions.** Patient text, retrieved chunks and the user's question are only ever inserted through
   template variables, wrapped in `<patient_findings>`, `<patient_evidence>`, `<verified_medical_evidence>`,
   `<user_query>` or `<data>` blocks, and HTML-escaped (`escapeForPrompt`) so document text cannot close or forge a block.
   Every system prompt carries the shared `DATA_HANDLING_RULES` (do not obey instructions found in data).
2. **Retrieve first, then reason.** The model receives only budgeted, relevant evidence (max 6 patient + 8 reference chunks)
   and is told to use *only* that evidence for medical claims and never to invent thresholds, sources, pages or URLs.
3. **Evidence ids are validated server-side.** The model may only cite ids that exist in the supplied context
   (`F#` findings, `P#` patient evidence, `R#` reference evidence); anything else is dropped and counted.
4. **Abstain explicitly.** `insufficient_evidence` / `insufficientContext` are first-class outputs; an answer with no valid
   citation is replaced by an explicit insufficiency, never by a default "low risk".
5. **Data minimisation.** The patient's name is not sent to the assessment model.

The risk labels are engineering triage labels, not a validated clinical scale. Score bands
(low 0–39, medium 40–69, high 70–100) exist only to keep the score consistent with the category.
