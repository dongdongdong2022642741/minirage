from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict


@dataclass(frozen=True)
class Chunk:
    chunk_id: str            # 稳定唯一：f"{doc_id}#{start_char}"（固定分块）
                             # 或 f"{doc_id}#sec{index}"（结构化父块）
    doc_id: str              # 挂回源头：这块来自哪篇文档
    parent_id: Optional[str]    # 父子分块：父块 None，子块指向父块 chunk_id
    heading_path: Tuple[str, ...]  # 标题链，如 ("第一章", "1.1 节")
                             # 空元组 () 表示"引言/无标题正文"
    text: str                # 块内容
    start_char: int          # 在原文中的起始偏移（可溯源）
    end_char: int            # 在原文中的结束偏移（不含，即 [start, end)）


@dataclass
class ChunkResult:
    chunks: List[Chunk] = field(default_factory=list)

    def stats(self) -> Dict[str, int]:
        total = len(self.chunks)
        parents = sum(1 for c in self.chunks if c.parent_id is None)
        children = sum(1 for c in self.chunks if c.parent_id is not None)
        return {
            "total": total,
            "parents": parents,
            "children": children
        }
