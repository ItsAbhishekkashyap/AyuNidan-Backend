# Terminology datasets (not included in the repository)

Place the official NLM datasets here (the ZIPs are git-ignored and are only ever read, never modified):

- `medlineplus/` — one ZIP containing the MedlinePlus Health Topics XML (`mplus_topics_*.xml`)
- `mesh/` — one ZIP containing the MeSH descriptor XML (`desc20XX.xml`)

Then run `npm run terminology:ingest -- --dataset=all` (use `--dry-run` first). Details: the "Medical Terminology Domain" section of the main README.
