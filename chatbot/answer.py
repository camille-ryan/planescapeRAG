"""Question + retrieved context → cited Claude answer.

Hands the retrieved entity bodies to Claude with a system prompt requiring
inline `(doc_tag, p. N)` citations grounded in the bullets we provide. Defaults
to Haiku 4.5 since RAG-with-citations is well within its capability and cost is
negligible; --model flag bumps to Sonnet for harder queries.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import anthropic

from ..config import Settings
from ..cost import report_call
from .retrieve import RetrievalResult, Retriever


DEFAULT_MODEL = "claude-haiku-4-5-20251001"
SUPPORTED_MODELS = [DEFAULT_MODEL, "claude-sonnet-4-6", "claude-opus-4-7"]

SYSTEM_PROMPT = """You answer questions about the Planescape D&D 2nd-edition setting using ONLY the entity files provided in context.

The entity files are excerpts from TSR sourcebooks. Each entity may have multiple `## Source: <doc_tag>` sections that quote the original sourcebook with page numbers, plus `### Relationships from <doc_tag>` bullets in the form:

    - [[Other Entity]] — "verbatim quote" (p. N)

Rules:
- Ground every factual claim in the provided source text. If the question cannot be answered from the context, say so clearly — do not invent.
- Cite each claim inline as `(<doc_tag>, p. N)`. The doc_tag is shown in section headers; the page number is at the end of each bullet. Use the EXACT doc_tag string as it appears in the headers, with no modifications.
- When you reference an entity that appears in the provided context, write its name as a wikilink: `[[Canonical Name]]`. Use the canonical name from the entity header (the `# Heading` at the top of each entity block). The downstream UI renders these as clickable links into the entity dossier.
- Be concise. Stay close to the source; do not editorialize.
- When the same entity appears across multiple `## Source:` sections, draw on whichever source(s) best answer the question and cite accordingly.
"""


@dataclass
class AnswerResult:
    question: str
    answer: str
    model: str
    cost_usd: Optional[float]
    retrieval: RetrievalResult


def build_context(result: RetrievalResult) -> str:
    """Render the retrieval result as a prompt context block."""
    out: list[str] = []
    if result.primaries:
        out.append("=== Top retrieved entities (by embedding similarity) ===\n")
        for p in result.primaries:
            out.append(f"--- ENTITY: {p.canonical_name} ({p.type}, similarity={p.similarity:.3f}) ---")
            out.append(p.body.rstrip())
            out.append("")
    if result.expansions:
        out.append("=== 1-hop expansions (graph-linked from primaries) ===\n")
        for e in result.expansions:
            out.append(f"--- ENTITY: {e.canonical_name} ({e.type}, linked from {e.expanded_from}) ---")
            out.append(e.body.rstrip())
            out.append("")
    return "\n".join(out)


def answer_question(
    question: str,
    retrieval: RetrievalResult,
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 1500,
) -> AnswerResult:
    settings = Settings()
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    user_message = f"# Context\n\n{build_context(retrieval)}\n\n# Question\n\n{question}"

    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    text = "".join(b.text for b in resp.content if b.type == "text")
    cost = report_call(
        operation="chatbot_answer",
        model=model,
        usage=resp.usage,
        is_batch=False,
        details={
            "question": question[:200],
            "n_primaries": len(retrieval.primaries),
            "n_expansions": len(retrieval.expansions),
        },
    )

    return AnswerResult(
        question=question,
        answer=text,
        model=model,
        cost_usd=cost,
        retrieval=retrieval,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the Planescape graph a question.")
    parser.add_argument("question", type=str)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, choices=SUPPORTED_MODELS)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--max-total", type=int, default=15)
    parser.add_argument("--no-expand", action="store_true")
    parser.add_argument("--entities-dir", type=Path, default=Path("planescapeRAG/graph/entities"))
    parser.add_argument("--index-dir", type=Path, default=Path("planescapeRAG/chatbot/index"))
    args = parser.parse_args()

    retriever = Retriever(index_dir=args.index_dir, entities_dir=args.entities_dir)
    retrieval = retriever.retrieve(
        args.question, k=args.k, max_total=args.max_total, expand=not args.no_expand,
    )
    result = answer_question(args.question, retrieval, model=args.model)

    print(f"\n=== Q: {args.question} ===\n")
    print(result.answer)
    print(f"\n[model: {result.model}]")
    print("\nRetrieved entities:")
    for p in retrieval.primaries:
        print(f"  primary    [{p.similarity:.3f}] {p.canonical_name}")
    for e in retrieval.expansions:
        print(f"  expansion           {e.canonical_name}  (← {e.expanded_from})")


if __name__ == "__main__":
    main()
