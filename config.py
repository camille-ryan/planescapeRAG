import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / '.env')


@dataclass
class Settings:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_ocr_model: str = os.getenv("OPENAI_OCR_MODEL", "gpt-4o")
    s3_ocr_bucket: str = os.getenv("S3_OCR_BUCKET", "")
    aws_access_key_id: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    aws_secret_access_key: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")

    def validate(self):
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY must be set for OCR.")
