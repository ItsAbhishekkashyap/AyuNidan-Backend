# Clinical guideline PDFs (not included in the repository)

The guideline PDFs are git-ignored (copyright/licensing and size). Only the provenance sidecars (`*.meta.json`) are committed.

To reproduce the verified clinical-reference corpus, obtain authorised copies from the publishers and save them here with these exact names:

| File | Source |
|---|---|
| `ACC_AHA_2026_Dyslipidemia_Guideline.pdf` | 2026 ACC/AHA/AACVPR/ABC/ACPM/ADA/AGS/APhA/ASPC/NLA/PCNA Guideline on the Management of Dyslipidemia (*Circulation* 2026;153:e1154–e1276) |
| `WHO_2024_Haemoglobin_Anemia_Guideline.pdf` | WHO, *Guideline on haemoglobin cutoffs to define anaemia in individuals and populations* (2024) |

Then run `npm run kb:ingest` (idempotent). To add another guideline, add the PDF plus a `<name>.meta.json` sidecar (`sourceId`, `title`, `organization`, `sourceType`, `authorization`, optional `publicationDate`, `version`, `url`, `medicalTopics`) — never invent metadata.
