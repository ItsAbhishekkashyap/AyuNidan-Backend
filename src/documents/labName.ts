/**
 * A biomarker name is only the name ("Heart Rate"). Extraction sometimes glues the report date, the source
 * file name or a page label onto it ("Heart Rate2/17/2011ECG-Sample-Report.pdf · pp.1, 2"). That provenance
 * belongs in the value's own date/source fields, so it is stripped here. The name is returned unchanged when
 * stripping would leave nothing.
 */
export const cleanBiomarkerName = (raw: string): string => {
  const trimmed = raw.trim();
  const cleaned = trimmed
    .replace(/\s*[·|]\s*(?:pp?\.|page\b).*$/i, '')
    .replace(/\s*\d{1,4}[/.-]\d{1,2}[/.-]\d{1,4}.*$/, '')
    .replace(/\s*\S*\.(?:pdf|png|jpe?g|txt|md)\b.*$/i, '')
    .replace(/\s+/g, ' ')
    .trim();
  return cleaned || trimmed;
};
