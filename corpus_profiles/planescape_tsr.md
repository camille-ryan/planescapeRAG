# Planescape TSR sourcebooks

Corpus profile for TSR-published Planescape sourcebooks (Campaign Setting, Planes of Chaos, etc.). Captures known typography and layout quirks that benefit from corpus-specific guidance.

## OCR addendum

This document is from a TSR-published Planescape sourcebook. Apply these corpus-specific rules in addition to the base instructions:

- **Pull quotes are rendered in a decorative font** For pull quotes, transcribe these with the intended words instead of letter for letter.
- **Pull quotes and sidebar callouts are blockquotes, not headings.** Even when rendered in large or stylized type, render any quoted/attributed text block as `> ...` (with attribution as a trailing `> — Speaker` line if present). Never assign `#`, `##`, or `###` to such blocks.
- **Page-number footers** like `+ 35 +`, `*40*`, or similar decorative page markers at the very bottom of the page should be omitted from output.
- **Heading conventions for this corpus:**
  - `#` — chapter titles
  - `##` — plane names, faction names, etc.(e.g. "THE QUASIELEMENTAL PLANE OF RADIANCE", "THE SOCIETY OF SENSATION").
  - `###` — named sub-sections.
  - `####` — finer subdivisions within a sub-section, only when the text clearly indicates a nested heading.
  - Do not promote a heading level above `#` based on font size alone — multiple pages may share `#`-level titles.
- **Illustrations:** If the content is a full page illustration without text return "\<illustration\>"
- **Maps:** If the content is a full page map, return "\<map of (location)>"

## Ingest addendum

### Canonical names

- Use the form most common or most formal in the source.
- Strip leading "the" except where load-bearing ("The Lady of Pain" — keep).
- Planes: prefer the full form ("Quasielemental Plane of Radiance", not "Plane of Radiance").
- Factions: prefer the full faction name over slang ("Doomguard", not "Sinkers" — but Sinkers goes in aliases).
- Characters: prefer "First Last" if both appear; otherwise the most-used form.
