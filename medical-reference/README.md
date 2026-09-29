# Verified medical reference knowledge base

This folder feeds **Domain B – verified medical knowledge**, the shared reference corpus the
evidence-grounded assessment and medical Q&A retrieve from. It ships **empty on purpose**:
no guideline text has been fabricated, downloaded or reproduced. Until authorised sources are
ingested, assessments are grounded only in patient data (including reference ranges printed
in the patient's own reports). If that is not enough, they return `insufficient_evidence`.

## Adding a source

1. Place the document in `sources/` (or `clinical-guidelines/`, which holds the real guideline PDFs currently ingested: ACC/AHA 2026 dyslipidemia, WHO 2024 haemoglobin cutoffs — both scanned by `npm run kb:ingest`; `--force` rebuilds). Supported formats are `.pdf` (text layer), `.md` and `.txt`.
2. Next to it, add a sidecar file named `<same-name>.meta.json`:

```json
{
  "sourceId": "example-guideline-2024",
  "title": "Exact document title as published",
  "organization": "Publishing organisation",
  "sourceType": "clinical_guideline",
  "authorization": "Licence or permission under which this copy may be used (e.g. public domain, CC BY 4.0, written permission ref …)",
  "publicationDate": "2024-05",
  "version": "3.1",
  "url": "https://publisher.example/official-page",
  "medicalTopics": ["diabetes"]
}
```

   - `sourceType` must be one of: `clinical_guideline`, `public_health_guidance`, `medical_reference`,
     `professional_society`, `other`.
   - Required fields: `sourceId`, `title`, `organization`, `sourceType`, `authorization`.
   - **Omit any field you cannot verify** (`publicationDate`, `version`, `url`, …). Never guess.
     Missing fields are simply not displayed.
3. Run `npm run kb:ingest`. Re-running it is safe: unchanged files are skipped, and changed files
   replace their previous vectors.

## What ingestion does

The pipeline runs these steps in order:

1. **Loader.** PDFs go through a per-page text layer with layout table detection; Markdown and text are read directly.
2. **Cleaning.**
3. **Section-aware chunking.** Headings become the `section`. Table rows are kept intact and never split.
4. **Provenance metadata** attached to every chunk: `sourceId`, `title`, `organization`, `page`, `section`,
   `chunkId`, `publicationDate`, `version` and `url` (the last three only when provided).
5. **Local Hugging Face embedding** (`bge-small-en-v1.5`; no paid API).
6. **Storage** in the shared `medical-reference` vector namespace.
7. **Registry.** The source is registered in MongoDB (`ReferenceSource`).

Retrieval only serves chunks whose source is registered as `indexed` in the current embedding
space. To retire a source, use `retireReferenceSource(sourceId)`. Pages without a text layer are
reported and skipped; they are never guessed.

`processed/` receives a per-source ingestion report and is git-ignored.

## Guarantees and non-claims

- **Separate from patient data.** Reference chunks live in the shared `medical-reference` vector namespace with
  `domain: "reference"` and no `userId`. Patient chunks live in per-user `user-<id>` namespaces with `domain: "patient"`.
  Retrieval for each domain filters on its own `domain`, and the two are never mixed in one vector query.
- **Provenance is mandatory and never fabricated.** Every displayed reference shows its title, organization and
  source id, plus section, page, publication date, version and URL *only if they were supplied/extracted*.
  Nothing is shown as "used" unless it was retrieved **and** cited by the model, and cited ids are validated
  server-side against the evidence actually supplied.
- **Synthetic test sources are not references.** Sources of type `synthetic_test` (used by the test-suite, the
  evaluation harness and `npm run demo:refs`) are ignored unless `RAG_INCLUDE_SYNTHETIC_REFERENCES=true`
  and are badged "synthetic test source" in the UI.
- **Not clinical validation.** Ingesting a document makes it retrievable; it does not make AyuNidan's output
  clinically validated. Outputs are evidence-grounded preliminary triage aids for clinician review.
- **Embedding-space safety.** Each chunk records its `embeddingSpace` (`Xenova/bge-small-en-v1.5@384`). If the
  embedding model changes, old chunks are ignored until the source is re-ingested (`npm run kb:ingest`).
