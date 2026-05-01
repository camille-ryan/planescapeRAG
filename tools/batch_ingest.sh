#!/usr/bin/env bash
# Batch-ingest every PDF under planescapeRAG/documents/ that hasn't yet
# produced entity files. Boxed Sets and Accessories ingest as 'canonical';
# Modules ingest as 'adventure' with the module-specific corpus profile.
#
# Pass-2 cost prompts kick in at $5/doc — the script will pause waiting for
# y/N at each ingestion that exceeds that threshold. Answering 'n' aborts
# that doc only; the script continues to the next one.
#
# Usage (from any cwd):
#   bash planescapeRAG/tools/batch_ingest.sh

set -uo pipefail   # NOT -e — one failed doc shouldn't stop the rest

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

PROFILE_CANONICAL="planescapeRAG/corpus_profiles/planescape_tsr.md"
PROFILE_ADVENTURE="planescapeRAG/corpus_profiles/planescape_tsr_module.md"
COST_THRESHOLD="5.00"

# pdf_path | doc_tag | source_type | pretty_title
# doc_tags are forever once entity files exist — pick deliberately and
# adjust below if you want different naming.
DOCS=(
    # Boxed Sets — canonical
    "planescapeRAG/documents/Boxed Sets/TSR 2615 Planes of Conflict.pdf|planes_of_conflict|canonical|Planes of Conflict"
    "planescapeRAG/documents/Boxed Sets/TSR 2621 Hellbound The Blood War.pdf|hellbound|canonical|Hellbound: The Blood War"

    # Accessories — canonical
    "planescapeRAG/documents/Accessories/TSR 2609 In The Cage A Guide to Sigil.pdf|in_the_cage|canonical|In the Cage: A Guide to Sigil"
    "planescapeRAG/documents/Accessories/TSR 2611 The Factol's Manifesto.pdf|factols_manifesto|canonical|The Factol's Manifesto"
    "planescapeRAG/documents/Accessories/TSR 2620 The Planeswalker's Handbook.pdf|planeswalkers_handbook|canonical|The Planeswalker's Handbook"
    "planescapeRAG/documents/Accessories/TSR 2623 On Hallowed Ground.pdf|on_hallowed_ground|canonical|On Hallowed Ground"
    "planescapeRAG/documents/Accessories/TSR 2624 Uncaged Faces of Sigil.pdf|uncaged_faces_of_sigil|canonical|Uncaged: Faces of Sigil"
    "planescapeRAG/documents/Accessories/TSR 2625 A Guide to the Astral Plane.pdf|guide_to_astral|canonical|A Guide to the Astral Plane"
    "planescapeRAG/documents/Accessories/TSR 2630 Faces of Evil the Fiends.pdf|faces_of_evil|canonical|Faces of Evil: The Fiends"
    "planescapeRAG/documents/Accessories/TSR 2633 Planescape A Guide to the Ethereal Plane.pdf|guide_to_ethereal|canonical|A Guide to the Ethereal Plane"
    "planescapeRAG/documents/Accessories/TSR 2634 The Inner Planes.pdf|inner_planes|canonical|The Inner Planes"

    # Modules — adventure
    "planescapeRAG/documents/Modules/TSR 2601 The Eternal Boundary.pdf|eternal_boundary|adventure|The Eternal Boundary"
    "planescapeRAG/documents/Modules/TSR 2604 The Well of Worlds.pdf|well_of_worlds|adventure|The Well of Worlds"
    "planescapeRAG/documents/Modules/TSR 2605 In the Abyss.pdf|in_the_abyss|adventure|In the Abyss"
    "planescapeRAG/documents/Modules/TSR 2606 The Deva Spark.pdf|deva_spark|adventure|The Deva Spark"
    "planescapeRAG/documents/Modules/TSR 2608 Fires of Dis.pdf|fires_of_dis|adventure|Fires of Dis"
    "planescapeRAG/documents/Modules/TSR 2614 Harbinger House.pdf|harbinger_house|adventure|Harbinger House"
    "planescapeRAG/documents/Modules/TSR 2619 Something Wild.pdf|something_wild|adventure|Something Wild"
    "planescapeRAG/documents/Modules/TSR 2626 Doors to the Unknown.pdf|doors_to_the_unknown|adventure|Doors to the Unknown"
    "planescapeRAG/documents/Modules/TSR 2628 The Great Modron March.pdf|great_modron_march|adventure|The Great Modron March"
    "planescapeRAG/documents/Modules/TSR 2629 Faction War.pdf|faction_war|adventure|Faction War"
    "planescapeRAG/documents/Modules/TSR 2631 Dead Gods.pdf|dead_gods|adventure|Dead Gods"
    "planescapeRAG/documents/Modules/TSR 2632 Tales From the Infinite Staircase.pdf|infinite_staircase|adventure|Tales from the Infinite Staircase"
)

# Returns 0 (true) if a doc_tag has already been ingested. Detection: any
# entity file references "## Source: <tag>".
already_ingested() {
    local tag="$1"
    grep -rqlF "## Source: $tag" planescapeRAG/graph/entities/ 2>/dev/null
}

n_total=${#DOCS[@]}
n_processed=0
n_skipped=0
n_failed=0

for spec in "${DOCS[@]}"; do
    IFS='|' read -r pdf tag stype title <<< "$spec"
    n_processed=$((n_processed + 1))

    echo
    echo "====================================================================="
    echo "[$n_processed/$n_total] $tag ($stype)"
    echo "  pdf:   $pdf"
    echo "  title: $title"
    echo "====================================================================="

    if already_ingested "$tag"; then
        echo "  SKIP — entity files already reference '$tag'"
        n_skipped=$((n_skipped + 1))
        continue
    fi

    if [[ ! -f "$pdf" ]]; then
        echo "  FAIL — PDF not found at $pdf"
        n_failed=$((n_failed + 1))
        continue
    fi

    if [[ "$stype" == "adventure" ]]; then
        profile="$PROFILE_ADVENTURE"
    else
        profile="$PROFILE_CANONICAL"
    fi

    python -m planescapeRAG.run_pipeline \
        --pdf "$pdf" \
        --profile "$profile" \
        --doc-tag "$tag" \
        --source-type "$stype" \
        --source-title "$title" \
        --pass2-cost-threshold "$COST_THRESHOLD"

    rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "  FAIL — pipeline exited with $rc (continuing to next doc)"
        n_failed=$((n_failed + 1))
    fi
done

echo
echo "====================================================================="
echo "Batch complete."
echo "  total:     $n_total"
echo "  ingested:  $((n_processed - n_skipped - n_failed))"
echo "  skipped:   $n_skipped (already in graph)"
echo "  failed:    $n_failed"
echo "====================================================================="
