/**
 * Medical TERMINOLOGY knowledge domain (evidence type T#).
 *
 * Built from the official NLM datasets supplied locally: MedlinePlus Health Topics (consumer-language
 * topic summaries) and MeSH descriptors (controlled vocabulary with scope notes / entry terms).
 * This domain explains what a term MEANS. It is NOT a clinical guideline, NOT patient data and is kept
 * apart from both (own namespace, own id space, never used to justify a risk level).
 */

export const TERMINOLOGY_DATASETS = ['medlineplus', 'mesh'] as const;
export type TerminologyDataset = (typeof TERMINOLOGY_DATASETS)[number];

export interface TerminologyRecord {
  /** Deterministic, dataset-native id: `mplus:<topic id>` or `mesh:<DescriptorUI>`. */
  recordId: string;
  dataset: TerminologyDataset;
  /** Native identifier in the source dataset (MedlinePlus topic id / MeSH DescriptorUI). */
  sourceRecordId: string;
  /** Preferred term (MedlinePlus title / MeSH descriptor name). */
  term: string;
  /** Alternate names present in the dataset (MedlinePlus also-called + see-reference, MeSH entry terms). */
  aliases: string[];
  /** Definition / summary as plain text (MedlinePlus full-summary, MeSH scope note). */
  definition: string;
  /** Page of the topic — present in MedlinePlus only; never constructed. */
  url?: string;
  /** Cross-reference to a MeSH descriptor (MedlinePlus mesh-heading) — provenance, not a claim. */
  meshDescriptor?: { id: string; name: string };
  /** MeSH tree numbers (hierarchy locations) — MeSH only. */
  treeNumbers?: string[];
  /** Dataset dates as written in the source (normalised to ISO when they are unambiguous). */
  dateCreated?: string;
  dateRevised?: string;
}

export interface DatasetInfo {
  dataset: TerminologyDataset;
  /** Human-readable source name (from this repository's knowledge of the dataset). */
  sourceName: string;
  organization: string;
  /** Version label derived ONLY from the dataset file (root attribute / DTD / file name). */
  version: string;
  /** Publication / generation date from the dataset itself, ISO. */
  publicationDate?: string;
  sourceFile: string;
}
