from typing import cast
from .chunk import Chunk, ChunkResult
from docparser import Document


def chunk_fixed(doc: Document, size: int, overlap: int = 0) -> ChunkResult:
    """
    把 doc.text 按固定长度切成块。

    行为契约：
    - size < 1 或 overlap < 0 或 overlap >= size → ValueError
    - 空文本（doc.text 为空）→ 返回空 ChunkResult，不报错
    - 步长 = size - overlap，最后一块允许不足 size（保留尾部，不丢字）
    - chunk_id = f"{doc_id}#{start_char}"（同一文档内唯一、稳定）
    - heading_path 恒为 ()，parent_id 恒为 None（本策略无父子）
    """
    if size < 1:
        raise ValueError("size must be >= 1")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= size:
        raise ValueError("overlap must be less than size")

    doc_id = cast(str, doc.doc_id)
    full_text = cast(str, doc.text)
    result = ChunkResult()

    if not full_text:
        return result

    step = size - overlap
    text_len = len(full_text)
    start = 0
    while start < text_len:
        end = start + size
        if end > text_len:
            end = text_len
        chunk_text = full_text[start:end]
        cid = f"{doc_id}#{start}"
        chunk = Chunk(
            chunk_id=cid,
            doc_id=doc_id,
            parent_id=None,
            heading_path=tuple(),
            text=chunk_text,
            start_char=start,
            end_char=end
        )
        result.chunks.append(chunk)
        start += step
    return result