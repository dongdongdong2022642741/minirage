"""MiniRAG document parsing module.

Parses text, PDF, DOCX and HTML files into structured Document objects so that the W2
chunking step and the W3 storage step can rely on stable identities and
metadata instead of bare strings.
"""

from .document import Document, Issue, ParseResult
from .loader import SUPPORTED_SUFFIXES, load_documents, parse_file

__all__ = [
    "Document",
    "Issue",
    "ParseResult",
    "load_documents",
    "parse_file",
    "SUPPORTED_SUFFIXES",
]
