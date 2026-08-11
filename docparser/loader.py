"""Load .md / .txt files from a directory into Document objects.

Error handling contract (no silent crashes):
- directory does not exist  -> raise FileNotFoundError
- file does not exist       -> raise FileNotFoundError
- empty file (0 bytes)      -> skipped with an Issue recorded in ParseResult
- undecodable bytes         -> skipped with an Issue recorded in ParseResult

Only the Python standard library is used. Encodings are probed in a small
fallback chain (utf-8, then gb18030 for legacy GBK-ish content).
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

try:
    from .document import Document, Issue, ParseResult
except ImportError:  # allow `python docparser/loader.py` to work directly
    from document import Document, Issue, ParseResult  # type: ignore[no-redef]

SUPPORTED_SUFFIXES = {".md", ".txt"}
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

    text, encoding = _read_text(path)
    if text == "":
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
            text, encoding = _read_text(path)
        except UnicodeError as error:
            result.issues.append(Issue(path.name, "decode_error", str(error)))
            continue
        if text == "":
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
