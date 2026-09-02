#!/usr/bin/env python3
"""Cache a PDF lesson and extract text with pypdf."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.parse
import urllib.request
from pathlib import Path

from pypdf import PdfReader


def is_url(value: str) -> bool:
    parsed = urllib.parse.urlparse(value)
    return parsed.scheme in {"http", "https"}


def safe_name(source: str) -> str:
    parsed = urllib.parse.urlparse(source)
    raw_name = Path(parsed.path).name if parsed.path else ""
    if not raw_name:
        raw_name = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16] + ".pdf"
    if not raw_name.lower().endswith(".pdf"):
        raw_name += ".pdf"
    return "".join(char if char.isalnum() or char in ".-_" else "_" for char in raw_name)


def cache_pdf(source: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / safe_name(source)

    if is_url(source):
        request = urllib.request.Request(source, headers={"User-Agent": "codex-tutor/1.0"})
        with urllib.request.urlopen(request) as response:
            destination.write_bytes(response.read())
        return destination

    source_path = Path(source).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"PDF not found: {source_path}")
    if source_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {source_path}")
    if source_path != destination.resolve():
        shutil.copy2(source_path, destination)
    return destination


def extract_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    chunks: list[str] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        chunks.append(f"\n\n--- Page {page_number} ---\n{text.strip()}")
    return "\n".join(chunks).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Cache a PDF and extract its text with pypdf.")
    parser.add_argument("source", help="Local PDF path or PDF URL.")
    parser.add_argument(
        "--cache-dir",
        default=".codex/lesson-cache",
        help="Directory for cached PDFs. Default: .codex/lesson-cache",
    )
    parser.add_argument("--output", help="Optional path for extracted text.")
    args = parser.parse_args()

    pdf_path = cache_pdf(args.source, Path(args.cache_dir))
    text = extract_text(pdf_path)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
        print(f"Cached PDF: {pdf_path}")
        print(f"Extracted text: {output_path}")
    else:
        print(text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
