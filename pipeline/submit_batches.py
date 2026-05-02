import argparse
import time
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv
from openai import OpenAI

from .combine_batch import combine_batch_results
from ..config import Settings
from .ocr import OpenAIOCR, load_corpus_profile

POLL_INTERVAL_SECONDS = 60
TERMINAL_STATUSES = {"completed", "failed", "cancelled", "expired"}


def _wait_for_batch(client: OpenAI, batch_id: str):
    while True:
        batch = client.batches.retrieve(batch_id)
        print(f"  [{batch_id}] {batch.status} {batch.request_counts}", flush=True)
        if batch.status in TERMINAL_STATUSES:
            return batch
        time.sleep(POLL_INTERVAL_SECONDS)


def submit_chunked(
    pdf_path: Path,
    output_dir: Path,
    profile: Optional[Dict[str, str]] = None,
    chunk_size: int = 50,
    page_start: int = 1,
    page_end: Optional[int] = None,
    combined_md_path: Optional[Path] = None,
) -> Path:
    settings = Settings()
    ocr = OpenAIOCR(settings)
    client = OpenAI()
    output_dir.mkdir(parents=True, exist_ok=True)

    if page_end is None:
        from pdf2image.pdf2image import pdfinfo_from_path
        page_end = pdfinfo_from_path(str(pdf_path))["Pages"]

    stem = pdf_path.stem.replace(" ", "_")
    result_files = []
    chunk_start = page_start
    chunk_idx = 1

    while chunk_start <= page_end:
        chunk_end = min(chunk_start + chunk_size - 1, page_end)
        tag = f"{stem}_chunk{chunk_idx:02d}_p{chunk_start}-{chunk_end}"
        jsonl_path = output_dir / f"{tag}.jsonl"
        results_path = output_dir / f"{tag}_results.jsonl"

        print(f"\n=== chunk {chunk_idx}: pages {chunk_start}-{chunk_end} ===", flush=True)

        if results_path.exists():
            print(f"  results already exist, skipping: {results_path}", flush=True)
            result_files.append(results_path)
            chunk_start = chunk_end + 1
            chunk_idx += 1
            continue

        if not jsonl_path.exists():
            ocr.create_batch_jsonl_for_document(
                pdf_path, jsonl_path,
                page_start=chunk_start, page_end=chunk_end, profile=profile,
            )
        else:
            print(f"  reusing existing JSONL: {jsonl_path}", flush=True)

        with jsonl_path.open("rb") as f:
            batch_file = client.files.create(file=f, purpose="batch")
        batch = client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/responses",
            completion_window="24h",
            metadata={"description": tag},
        )
        print(f"  submitted: {batch.id}", flush=True)

        batch = _wait_for_batch(client, batch.id)
        if batch.status != "completed" or not batch.output_file_id:
            raise RuntimeError(f"Batch {batch.id} status={batch.status} errors={batch.errors}")

        results_path.write_bytes(client.files.content(batch.output_file_id).read())
        result_files.append(results_path)
        print(f"  saved: {results_path}", flush=True)

        if batch.error_file_id:
            err_path = output_dir / f"{tag}_errors.jsonl"
            err_path.write_bytes(client.files.content(batch.error_file_id).read())
            print(f"  errors: {err_path}", flush=True)

        chunk_start = chunk_end + 1
        chunk_idx += 1

    combined_jsonl = output_dir / f"{stem}_all_results.jsonl"
    with combined_jsonl.open("wb") as out:
        for rp in result_files:
            out.write(rp.read_bytes())
    print(f"\nCombined JSONL: {combined_jsonl}", flush=True)

    # `combined_md_path` is the explicit destination from the orchestrator.
    # Fallback to the legacy in-tmp location for direct CLI use.
    md_path = combined_md_path if combined_md_path is not None else output_dir / f"{stem}_combined.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    combine_batch_results(combined_jsonl, md_path)
    return md_path


if __name__ == "__main__":
    load_dotenv(Path(__file__).parent / ".env")
    parser = argparse.ArgumentParser(description="Submit a PDF as chunked OCR batches under tier-1 token limits.")
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile", type=Path, default=None)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--page-start", type=int, default=1)
    parser.add_argument("--page-end", type=int, default=None)
    args = parser.parse_args()

    profile = load_corpus_profile(args.profile) if args.profile else None
    submit_chunked(
        args.pdf, args.output_dir, profile=profile,
        chunk_size=args.chunk_size, page_start=args.page_start, page_end=args.page_end,
    )
