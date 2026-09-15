import csv
import io
import json
from pathlib import Path
from pypdf import PdfReader
from backend.models import TestSet

MAX_BYTES = 5 * 1024 * 1024
MAX_TEXT = 150_000


def extract_document(name: str, data: bytes) -> list[dict]:
    if len(data) > MAX_BYTES:
        raise ValueError("Each document must be at most 5 MB")
    suffix = Path(name).suffix.lower()
    name = name.replace("\\", "/").split("/")[-1]
    if suffix == ".pdf":
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise ValueError("Encrypted PDFs are unsupported")
        if len(reader.pages) > 100:
            raise ValueError("PDFs must contain at most 100 pages")
        docs = [{"source": name, "page": i + 1, "text": page.extract_text() or ""}
                for i, page in enumerate(reader.pages)]
    elif suffix in {".txt", ".md"}:
        docs = [{"source": name, "page": 1, "text": data.decode("utf-8-sig")}]
    else:
        raise ValueError("Use PDF, UTF-8 TXT, or Markdown files")
    docs = [d for d in docs if d["text"].strip()]
    if not docs:
        raise ValueError("No extractable text; scanned PDFs require OCR first")
    if sum(len(d["text"]) for d in docs) > MAX_TEXT:
        raise ValueError("Document set is limited to 150,000 extracted characters")
    return docs


def parse_test_set(name, data):
    if len(data) > MAX_BYTES:
        raise ValueError("Test set is too large")
    text = data.decode("utf-8-sig")
    if name.lower().endswith(".csv"):
        rows = list(csv.DictReader(io.StringIO(text)))
    elif name.lower().endswith(".json"):
        parsed = json.loads(text)
        rows = parsed.get("questions", []) if isinstance(parsed, dict) else parsed
    else:
        raise ValueError("Upload CSV or JSON")
    if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
        raise ValueError("Expected a list of question/reference objects")
    return TestSet(name=name, questions=[{"question": r.get("question", ""),
        "reference": r.get("reference", r.get("expected_answer", ""))} for r in rows])
