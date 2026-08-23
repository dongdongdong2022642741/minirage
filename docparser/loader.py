"""Load supported files into a normalized Markdown Document object.

Error handling contract (no silent crashes):
- directory does not exist  -> raise FileNotFoundError
- file does not exist       -> raise FileNotFoundError
- empty file (0 bytes)      -> skipped with an Issue recorded in ParseResult
- undecodable bytes         -> skipped with an Issue recorded in ParseResult

Text encodings are probed in a small fallback chain. PDF, DOCX and HTML are
normalized to Markdown before entering the existing structured chunker.
"""

from __future__ import annotations

import hashlib
import html as html_stdlib
import re
from datetime import datetime, timezone
from pathlib import Path

try:
    from .document import Document, Issue, ParseResult
except ImportError:  # allow `python docparser/loader.py` to work directly
    from document import Document, Issue, ParseResult  # type: ignore[no-redef]

SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf", ".docx", ".html", ".htm"}
# Tried in order; later entries are fallbacks for legacy encodings.
ENCODING_FALLBACKS = ("utf-8", "gb18030")


def _stable_doc_id(path: Path) -> str:
    """Deterministic id from the absolute path, stable across runs."""
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()
    return digest[:16]


def _read_text(path: Path) -> tuple[str, str]:
    """Read file bytes and decode with a fallback chain.

    Returns (text, encoding_used). Raises UnicodeError if nothing matches.
    """
    raw = path.read_bytes()
    last_error: UnicodeError | None = None
    for encoding in ENCODING_FALLBACKS:
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError as error:
            last_error = error
    raise UnicodeError(f"could not decode {path.name}: {last_error}")


def _escape_table_cell(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().replace("|", "\\|")


def _table_to_markdown(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    normalized = [row + [""] * (width - len(row)) for row in rows]
    header = normalized[0]
    lines = [
        "| " + " | ".join(_escape_table_cell(cell) for cell in header) + " |",
        "| " + " | ".join("---" for _ in range(width)) + " |",
    ]
    lines.extend(
        "| " + " | ".join(_escape_table_cell(cell) for cell in row) + " |"
        for row in normalized[1:]
    )
    return "\n".join(lines)


def _read_docx(path: Path) -> tuple[str, str]:
    try:
        from docx import Document as DocxDocument
        from docx.table import Table
        from docx.text.paragraph import Paragraph

        document = DocxDocument(path)
        blocks: list[str] = []
        for item in document.iter_inner_content():
            if isinstance(item, Paragraph):
                text = item.text.strip()
                if not text:
                    continue
                style = item.style.name if item.style is not None else ""
                match = re.match(r"(?:Heading|标题)\s*(\d+)", style, re.IGNORECASE)
                blocks.append(f"{'#' * min(int(match.group(1)), 6)} {text}" if match else text)
            elif isinstance(item, Table):
                table = _table_to_markdown(
                    [[cell.text for cell in row.cells] for row in item.rows]
                )
                if table:
                    blocks.append(table)
        return "\n\n".join(blocks), "docx"
    except Exception as error:
        raise ValueError(f"could not parse DOCX {path.name}: {error}") from error


def _read_html(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()
    last_error: UnicodeError | None = None
    decoded = ""
    encoding_used = ""
    for encoding in ENCODING_FALLBACKS:
        try:
            decoded = raw.decode(encoding)
            encoding_used = encoding
            break
        except UnicodeDecodeError as error:
            last_error = error
    if not encoding_used:
        raise UnicodeError(f"could not decode {path.name}: {last_error}")

    try:
        from lxml import html

        root = html.fromstring(decoded)
        for node in root.xpath("//script|//style|//noscript"):
            node.drop_tree()
        blocks: list[str] = []
        for node in root.xpath("//h1|//h2|//h3|//h4|//h5|//h6|//p|//li|//table"):
            tag = node.tag.lower()
            if tag == "table":
                rows = [
                    [" ".join(cell.itertext()) for cell in row.xpath("./th|./td")]
                    for row in node.xpath(".//tr")
                ]
                value = _table_to_markdown(rows)
            else:
                value = html_stdlib.unescape(" ".join(" ".join(node.itertext()).split()))
                if tag.startswith("h") and value:
                    value = f"{'#' * int(tag[1])} {value}"
                elif tag == "li" and value:
                    value = f"- {value}"
            if value:
                blocks.append(value)
        return "\n\n".join(blocks), encoding_used
    except Exception as error:
        raise ValueError(f"could not parse HTML {path.name}: {error}") from error


def _read_pdf(path: Path) -> tuple[str, str]:
    try:
        from pypdf import PdfReader

        reader = PdfReader(path)
        pages: list[str] = []
        for page_number, page in enumerate(reader.pages, 1):
            text = (page.extract_text() or "").replace("\x00", "").strip()
            if text:
                pages.append(f"## 第 {page_number} 页\n\n{text}")
        if not pages:
            raise ValueError("PDF contains no extractable text; scanned PDFs require OCR")
        return "\n\n".join(pages), "pdf"
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(f"could not parse PDF {path.name}: {error}") from error


def _extract_text(path: Path) -> tuple[str, str]:
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt"}:
        return _read_text(path)
    if suffix == ".docx":
        return _read_docx(path)
    if suffix in {".html", ".htm"}:
        return _read_html(path)
    if suffix == ".pdf":
        return _read_pdf(path)
    raise ValueError(f"unsupported file type: {path.name}")


def _build_document(path: Path, text: str, encoding: str) -> Document:
    stat = path.stat()
    return Document(
        doc_id=_stable_doc_id(path),
        filename=path.name,
        path=str(path.resolve()),
        text=text,
        suffix=path.suffix.lstrip(".").lower(),
        encoding=encoding,
        char_count=len(text),
        line_count=text.count("\n") + 1,
        size_bytes=stat.st_size,
        modified_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    )


def parse_file(path: str | Path) -> Document:
    """Parse a single file. Raises FileNotFoundError or ValueError on problems.

    - File does not exist  -> FileNotFoundError
    - Empty file           -> ValueError
    - Undecodable content  -> UnicodeError
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"file does not exist: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"not a regular file: {path}")
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(f"unsupported file type: {path.name}")

    text, encoding = _extract_text(path)
    if not text.strip():
        raise ValueError(f"file is empty: {path.name}")
    return _build_document(path, text, encoding)


def load_documents(directory: str | Path) -> ParseResult:
    """Load every .md / .txt file in *directory* (non-recursive).

    Missing directory raises FileNotFoundError. Per-file problems are
    collected as Issues instead of aborting the whole batch.
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"directory does not exist: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"not a directory: {directory}")

    result = ParseResult()
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        try:
            text, encoding = _extract_text(path)
        except (UnicodeError, ValueError) as error:
            reason = "decode_error" if isinstance(error, UnicodeError) else "parse_error"
            result.issues.append(Issue(path.name, reason, str(error)))
            continue
        if not text.strip():
            result.issues.append(
                Issue(path.name, "empty_file", f"skipped, file is empty ({path.stat().st_size} bytes)")
            )
            continue
        result.documents.append(_build_document(path, text, encoding))
    return result


def main() -> int:
    """Demo entry point: print a summary of what was parsed."""
    import sys

    if len(sys.argv) > 1:
        root = Path(sys.argv[1])
    else:
        root = Path(__file__).resolve().parent.parent / "docs"
    if not root.exists():
        print(f"Error: directory does not exist: {root}", file=sys.stderr)
        return 1

    result = load_documents(root)
    print(f"Loaded {len(result.documents)} document(s) from {root}:")
    for doc in result.documents:
        print(f"  [{doc.doc_id}] {doc.filename} ({doc.suffix}, {doc.char_count} chars, {doc.line_count} lines)")
    for issue in result.issues:
        print(f"  [issue] {issue.filename}: {issue.reason} - {issue.message}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
