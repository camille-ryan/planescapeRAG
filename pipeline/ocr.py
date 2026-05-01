import json
import re
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, List, Optional

try:
    import boto3
    from botocore.client import Config as BotoConfig
except ImportError:
    boto3 = None
    BotoConfig = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from ..config import Settings


_OCR_BASE_PROMPT = (
    "Extract all visible text from this document page image as Markdown. "
    "Preserve structure: use #, ##, ### headings for titles and sections, "
    "> for pull quotes and blockquotes, and lists where appropriate. "
    "Return raw markdown — do not wrap the output in a code fence. "
    "Do not add commentary or explanations."
)


def build_ocr_prompt(profile: Optional[Dict[str, str]] = None) -> str:
    if profile and profile.get("ocr"):
        return f"{_OCR_BASE_PROMPT}\n\n{profile['ocr']}"
    return _OCR_BASE_PROMPT


_OCR_CORRECTION_BULLET_RE = re.compile(r"^\s*-\s*(.+?)\s*→\s*(.+?)\s*$")


def parse_ocr_corrections(profile: Optional[Dict[str, str]]) -> Dict[str, str]:
    """Extract OCR corrections from the profile's Ingest addendum.

    Looks for `- old → new` bullets under a `### OCR corrections` heading.
    Returns {old: new}. Empty dict if no profile or no section.
    """
    if not profile or "ingest" not in profile:
        return {}

    in_section = False
    corrections: Dict[str, str] = {}
    for line in profile["ingest"].splitlines():
        if line.startswith("### "):
            heading = line[4:].strip().lower()
            in_section = "ocr correction" in heading or heading.startswith("ocr correction")
            continue
        if not in_section:
            continue
        match = _OCR_CORRECTION_BULLET_RE.match(line)
        if match:
            corrections[match.group(1).strip()] = match.group(2).strip()
    return corrections


def apply_ocr_corrections(text: str, corrections: Dict[str, str]) -> str:
    """Apply each correction as a plain-string replacement, in iteration order."""
    for old, new in corrections.items():
        text = text.replace(old, new)
    return text


def load_corpus_profile(path: Path) -> Dict[str, str]:
    """Parse a corpus profile markdown file into a dict of section text.

    Sections are level-2 headings; the first word of the heading becomes the key
    (e.g. "## OCR addendum" -> "ocr", "## Ingest addendum" -> "ingest").
    """
    text = Path(path).read_text()
    sections: Dict[str, str] = {}
    current_key: Optional[str] = None
    current_lines: List[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            if current_key is not None:
                sections[current_key] = "\n".join(current_lines).strip()
            current_key = line[3:].strip().split()[0].lower()
            current_lines = []
        elif current_key is not None:
            current_lines.append(line)
    if current_key is not None:
        sections[current_key] = "\n".join(current_lines).strip()
    return sections


class OpenAIOCR:
    def __init__(self, settings: Settings):
        self.settings = settings
        if not self.settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY must be set to use OCR.")
        if OpenAI is None:
            raise RuntimeError("Install the openai Python package to use OCR.")
        self.client = OpenAI(api_key=self.settings.openai_api_key)

    def create_batch_jsonl_for_document(
        self,
        pdf_path: Path,
        jsonl_output_path: Path,
        page_start: int = 1,
        page_end: Optional[int] = None,
        profile: Optional[Dict[str, str]] = None,
    ) -> None:
        """Convert PDF pages to images, upload to S3, and create batch JSONL file for OCR."""
        if boto3 is None:
            raise RuntimeError("Install boto3 to use S3 upload functionality.")
        if not self.settings.s3_ocr_bucket:
            raise ValueError("S3_OCR_BUCKET must be set in .env for batch OCR.")
        if not self.settings.aws_access_key_id or not self.settings.aws_secret_access_key:
            raise ValueError("AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set in .env for S3 upload.")

        from pdf2image import convert_from_path
        # Critical: explicit regional endpoint + virtual addressing. Without
        # these, boto3 generates presigned URLs against the global host
        # (BUCKET.s3.amazonaws.com), which S3 sometimes serves directly and
        # sometimes 307-redirects to the regional host. The redirect causes
        # SigV4 SignatureDoesNotMatch because the signature is bound to the
        # original host name. Failure rate was ~30-80% for non-trivial doc
        # uploads. With the regional endpoint, URLs are signed for the
        # bucket's home region directly — no redirect, no signature mismatch.
        s3_client = boto3.client(
            's3',
            aws_access_key_id=self.settings.aws_access_key_id,
            aws_secret_access_key=self.settings.aws_secret_access_key,
            region_name=self.settings.aws_region,
            endpoint_url=f"https://s3.{self.settings.aws_region}.amazonaws.com",
            config=BotoConfig(
                signature_version='s3v4',
                s3={'addressing_style': 'virtual'},
            ),
        )

        prompt = build_ocr_prompt(profile)
        images = convert_from_path(pdf_path, first_page=page_start, last_page=page_end)
        batch_lines = []

        for page_num, pil_image in enumerate(images, start=page_start):
            with NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                pil_image.save(tmp.name, 'PNG')
                tmp_path = Path(tmp.name)

            s3_key = f"planescape/{pdf_path.stem}/page-{page_num}.png"
            s3_client.upload_file(str(tmp_path), self.settings.s3_ocr_bucket, s3_key)
            s3_url = s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.settings.s3_ocr_bucket, 'Key': s3_key},
                ExpiresIn=86400,
            )

            request_line = {
                "custom_id": f"page-{page_num}",
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": self.settings.openai_ocr_model,
                    "input": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": prompt},
                                {"type": "input_image", "image_url": s3_url},
                            ],
                        }
                    ],
                },
            }
            batch_lines.append(json.dumps(request_line))
            tmp_path.unlink()

        with jsonl_output_path.open('w') as f:
            for line in batch_lines:
                f.write(line + '\n')

        print(f"Created batch JSONL with {len(batch_lines)} requests at {jsonl_output_path}")
