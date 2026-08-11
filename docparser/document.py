"""Data structures for the MiniRAG document layer.

Why a Document object instead of a bare string?
- Later stages (chunking in W2, storage in W3) need a stable doc_id to
  attach chunks back to their source document. A list of strings carries
  no identity, so W3 would force a rewrite of this module.
- Metadata (path, suffix, size, modified time) is cheap to record now and
  expensive to reconstruct later. It is the anchor for tagging, debugging
  and the evaluation set in W5.

This mirrors RAGFlow's two-layer split: a raw Document here, and
Paragraph/Chunk objects layered on top of it in the W2 chunking step.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Document:
    """A single parsed document.

    Fields:
        doc_id: stable identifier derived from the file path (deterministic
            across runs, so storage and embeddings in later weeks can rely on
            it).
        filename: base name of the source file (e.g. "README.md").
        path: absolute path of the source file.
        suffix: file extension without the dot (e.g. "md", "txt").
        text: raw content of the document.
        encoding: encoding used to decode the source file.
        char_count: length of the text in characters.
        line_count: number of lines.
        size_bytes: size of the source file on disk.
        modified_at: ISO-8601 last-modified timestamp.
    """

    doc_id: str
    filename: str
    path: str
    text: str
    suffix: str
    encoding: str
    char_count: int
    line_count: int
    size_bytes: int
    modified_at: str


@dataclass(frozen=True)
class Issue:
    """A non-fatal problem encountered while parsing (e.g. an empty file)."""

    filename: str
    reason: str
    message: str


@dataclass
class ParseResult:
    """Outcome of loading a directory: parsed documents plus skipped issues."""

    documents: list[Document] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)
