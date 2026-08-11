import re
from typing import List, Tuple, Optional, NamedTuple, cast
from .chunk import Chunk, ChunkResult
from docparser import Document


class Heading(NamedTuple):
    level: int
    name: str
    line_start: int


# ATX 标题正则：行首1‑6个#，后面跟空格，再跟标题文本
_RE_ATX_HEADING = re.compile(r"^#{1,6} (.*?)\s*$", re.MULTILINE)


def _extract_headings(text: str) -> List[Heading]:
    """提取ATX标题列表，返回(level, name, line_start)"""
    headings: List[Heading] = []
    for match in _RE_ATX_HEADING.finditer(text):
        line_start = match.start()
        full_line = match.group(0)
        name = match.group(1).strip()
        hash_cnt = len(full_line) - len(full_line.lstrip("#"))
        if not name:
            continue
        headings.append(Heading(level=hash_cnt, name=name, line_start=line_start))
    return headings


def _split_sections(text: str, headings: List[Heading]) -> List[Tuple[int, int, Tuple[str, ...]]]:
    """
    返回 [(section_start, section_end, heading_path)]
    包含引言块；维护标题栈处理层级与跳级。
    """
    sections: List[Tuple[int, int, Tuple[str, ...]]] = []
    if not headings:
        sections.append((0, len(text), tuple()))
        return sections

    # 引言：0 ~ 第一个标题起始位置
    first_h = headings[0]
    if first_h.line_start > 0:
        sections.append((0, first_h.line_start, tuple()))

    stack: List[Heading] = []
    for idx, h in enumerate(headings):
        # pop掉所有 >= 当前层级
        while stack and stack[-1].level >= h.level:
            stack.pop()
        stack.append(h)
        path = tuple(item.name for item in stack)

        sec_start = h.line_start
        if idx + 1 < len(headings):
            sec_end = headings[idx + 1].line_start
        else:
            sec_end = len(text)
        sections.append((sec_start, sec_end, path))
    return sections


def _split_child_chunks(
    parent_chunk: Chunk,
    child_size: int,
    child_overlap: int
) -> List[Chunk]:
    """
    将父块按子块参数切分；子块偏移基于全文；继承heading_path；parent_id指向父块。
    """
    children: List[Chunk] = []
    step = child_size - child_overlap
    parent_text = parent_chunk.text
    p_start = parent_chunk.start_char
    text_len = len(parent_text)
    start = 0
    while start < text_len:
        end = start + child_size
        if end > text_len:
            end = text_len
        sub_text = parent_text[start:end]
        abs_start = p_start + start
        abs_end = p_start + end
        cid = f"{parent_chunk.doc_id}#{abs_start}"
        child = Chunk(
            chunk_id=cid,
            doc_id=parent_chunk.doc_id,
            parent_id=parent_chunk.chunk_id,
            heading_path=parent_chunk.heading_path,
            text=sub_text,
            start_char=abs_start,
            end_char=abs_end
        )
        children.append(child)
        start += step
    return children


def chunk_structured(
    doc: Document,
    max_section_size: int = 2000,
    child_size: int = 500,
    child_overlap: int = 50,
) -> ChunkResult:
    """
    沿 Markdown 标题层级切块，超长章节切子块。

    行为契约：
    - 参数非法（<=0 或 child_overlap >= child_size）→ ValueError
    - 无标题的纯文本 → 整篇一个父块（heading_path=()）
    - 引言（第一个标题之前的内容）→ 独立父块，heading_path=()，
      不并入第一章（避免概述污染章节检索）
    - 每个标题（含标题行本身）到下一个同级或更高级标题之前 → 一个父块，
      heading_path = 当前标题栈路径
    - 标题层级跳级（# 一 直接到 ### 1.1.1）→ 接受跳级，
      路径为 ("一", "1.1.1")，不虚构中间层标题
    - 父块文本长度 > max_section_size → 再按 child_size/child_overlap
      切成子块；子块继承父块 heading_path，parent_id = 父块 chunk_id，
      偏移量 = 父块起始偏移 + 子块在父块内的偏移
    - 父块始终存在（即使被切子块），W4 要拿它当 LLM 上下文
    """
    if max_section_size <= 0:
        raise ValueError("max_section_size must be >0")
    if child_size <= 0:
        raise ValueError("child_size must be >0")
    if child_overlap < 0:
        raise ValueError("child_overlap must be >=0")
    if child_overlap >= child_size:
        raise ValueError("child_overlap must be less than child_size")

    doc_id = cast(str, doc.doc_id)
    full_text = cast(str, doc.text)
    result = ChunkResult()
    if not full_text:
        return result

    headings = _extract_headings(full_text)
    sections = _split_sections(full_text, headings)

    for sec_start, sec_end, path in sections:
        if sec_start >= sec_end:
            continue
        sec_text = full_text[sec_start:sec_end]
        parent_cid = f"{doc_id}#sec{sec_start}"
        parent_chunk = Chunk(
            chunk_id=parent_cid,
            doc_id=doc_id,
            parent_id=None,
            heading_path=path,
            text=sec_text,
            start_char=sec_start,
            end_char=sec_end
        )
        result.chunks.append(parent_chunk)

        # 超长则生成子块
        if len(sec_text) > max_section_size:
            child_list = _split_child_chunks(parent_chunk, child_size, child_overlap)
            result.chunks.extend(child_list)

    return result