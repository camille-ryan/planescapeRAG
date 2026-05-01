# Planescape TSR adventure modules

Corpus profile for TSR-published Planescape **adventure modules** (e.g. The Eternal Boundary, The Great Modron March). Distinct from the base TSR profile because modules are narrative artifacts — they reference canonical entities but mostly contain stat blocks, encounter logistics, treasure, and per-room descriptions that aren't lore.

The OCR addendum is identical to the base profile (same publisher, same typography). The Ingest addendum is the meaningful difference.

## OCR addendum

This document is from a TSR-published Planescape adventure module. Apply these corpus-specific rules in addition to the base instructions:

- **Pull quotes are rendered in a decorative font.** For pull quotes, transcribe these with the intended words instead of letter for letter.
- **Pull quotes and sidebar callouts are blockquotes, not headings.** Even when rendered in large or stylized type, render any quoted/attributed text block as `> ...` (with attribution as a trailing `> — Speaker` line if present). Never assign `#`, `##`, or `###` to such blocks.
- **Stat blocks** (lines beginning with abbreviations like `AC`, `MV`, `HD`, `hp`, `THAC0`, `#AT`, `Dmg`, `SA`, `SD`, `MR`, `Int`, `AL`, `XP`, `Str`, `Dex`, `Con`, `Wis`, `Int`, `Cha`) should be transcribed as a fenced code block (` ```stat-block ` ... ` ``` `) so downstream stages can recognize and skip them. Race/class/alignment lines that read as descriptive prose ("Marcus is a hobgoblin fighter (4th level), lawful evil") stay as prose — only mechanical numerics go in the fence.
- **Encounter tables and treasure tables** (numbered or tabular lists tied to specific rooms or random encounters) should be wrapped as fenced `gameplay-table` code blocks.
- **Page-number footers** like `+ 35 +`, `*40*`, or similar decorative page markers at the very bottom of the page should be omitted from output.
- **Heading conventions for this corpus:**
  - `#` — module title or chapter titles
  - `##` — major sections (Adventure Background, Chapter X, Appendices)
  - `###` — encounter areas, scene names, named NPC sections
  - `####` — finer subdivisions
  - Do not promote a heading level above `#` based on font size alone.
- **Illustrations:** If the content is a full page illustration without text return "\<illustration\>"
- **Maps:** If the content is a full page map, return "\<map of (location)>"

## Ingest addendum

This document is an **adventure module**, not a canonical setting reference. Apply these rules:

### What to extract

- **Named NPCs** with lore-bearing details. Keep race, class, level, alignment, profession/role, allegiances, motivation, relationships to other entities, and narrative descriptions. *Skip* combat statistics (HD, AC, HP, THAC0, attack rolls, saving throws, equipment lists, treasure carried).
- **Named locations** explicitly added to the canonical setting (named cities, regions, landmarks, planar features). *Skip* dungeon rooms, generic cells, encounter areas, and module-internal-only locations.
- **Named factions, organizations, or named items** with lore significance beyond a single encounter.

### What to skip

- Stat blocks of generic monsters or unnamed encounter creatures ("3 manes appear", "wandering bariaur").
- Pre-generated PCs intended for player use.
- Treasure lists, magic-item allotments tied to encounters, and per-room loot.
- Encounter tables, random-encounter rolls, and skill-check DCs.
- Designer's notes, sidebar play advice, and "if the players do X" branching paragraphs.
- Per-room boxed-text read-alouds (these describe sensory detail for play, not enduring world fact).

### Bias

Bias toward **fewer, higher-quality entities**. When in doubt about whether something belongs in the canon, skip. The chatbot using this corpus prefers a clean lore graph over a complete inventory of module contents.

### Canonical names

Same conventions as the base TSR profile. If a module introduces an NPC who already exists in a canonical source under a different name, prefer the canonical-source form and add the module's form as an alias.

### Supertypes

Same set as the base TSR profile.

### OCR corrections

(none yet — add as discovered)
