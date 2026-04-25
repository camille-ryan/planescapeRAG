import argparse
import json
import re
from pathlib import Path

CODE_FENCE_RE = re.compile(r"^\s*```(?:markdown|md)?\s*\n(.*?)\n\s*```\s*$", re.DOTALL)


def _extract_text(record: dict) -> str:
    body = record["response"]["body"]
    parts = []
    for item in body.get("output", []):
        for piece in item.get("content", []):
            if piece.get("text"):
                parts.append(piece["text"])
    text = "\n".join(parts).strip()
    match = CODE_FENCE_RE.match(text)
    if match:
        text = match.group(1).strip()
    return text


def _page_number(custom_id: str) -> int:
    return int(custom_id.split("-")[-1])


def combine_batch_results(input_jsonl: Path, output_md: Path) -> None:
    pages = []
    with input_jsonl.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            pages.append((_page_number(rec["custom_id"]), _extract_text(rec)))
    pages.sort()

    blocks = [f"<!-- page {n} -->\n{text}" for n, text in pages]
    output_md.write_text("\n\n".join(blocks) + "\n")
    print(f"Combined {len(pages)} pages into {output_md}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine OpenAI batch OCR results into a single markdown document.")
    parser.add_argument("input_jsonl", type=Path, help="Batch results JSONL produced by OpenAI batch API")
    parser.add_argument("output_md", type=Path, help="Path to write the combined markdown")
    args = parser.parse_args()
    combine_batch_results(args.input_jsonl, args.output_md)
