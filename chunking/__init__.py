"""MiniRAG chunking module (W2): turns a Document into retrieval Chunks."""

from .chunk import Chunk, ChunkResult
from .fixed import chunk_fixed
from .structured import chunk_structured

__all__ = ["Chunk", "ChunkResult", "chunk_fixed", "chunk_structured"]
