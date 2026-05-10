"""PDF → text via pdfplumber."""

from __future__ import annotations

from pathlib import Path

import pdfplumber


def pdf_to_text(path: Path) -> str:
    parts: list[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            if t.strip():
                parts.append(t)
    return "\n\n".join(parts)
