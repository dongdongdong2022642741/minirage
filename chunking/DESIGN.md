# MiniRAG W2 分块模块设计文档

> 本文档是设计蓝图：你照着这个写 `chunk.py` / `fixed.py` / `structured.py`。
> 文档给出精确的签名、字段和行为契约，实现逻辑由你自己写。

## 1. 分层定位

RAGFlow 的两层拆分，本项目对应：

```
docparser.Document（W1，原始层：一篇文档 = 一个对象）
        │ 输入
        ▼
chunking.Chunk（W2，分块层：一篇文档 = 多个块）
```

- 输入：W1 的 `docparser.Document` 对象
- 输出：`ChunkResult`（一个结果容器，内含 Chunk 列表 + 统计）

为什么输入用 Document 而不是裸字符串？因为这是分层管线，每层消费上一层的
产出；Document 是项目已有的数据契约。内部实现只使用 `doc.doc_id` 和
`doc.text` 两个字段（薄依赖，将来换数据源不伤筋骨）。

## 2. Chunk 数据结构（写在 chunk.py）

```python
@dataclass(frozen=True)
class Chunk:
    chunk_id: str            # 稳定唯一：f"{doc_id}#{start_char}"（固定分块）
                             # 或 f"{doc_id}#sec{index}"（结构化父块）
    doc_id: str              # 挂回源头：这块来自哪篇文档
    parent_id: str | None    # 父子分块：父块 None，子块指向父块 chunk_id
    heading_path: tuple[str, ...]  # 标题链，如 ("第一章", "1.1 节")
                             # 空元组 () 表示"引言/无标题正文"
    text: str                # 块内容
    start_char: int          # 在原文中的起始偏移（可溯源）
    end_char: int            # 在原文中的结束偏移（不含，即 [start, end)）
```

字段选型理由（面试要说得出）：
- `parent_id: str | None`：None 表示顶级块，非 None 表示子块。子块检索、父块喂 LLM。
- `heading_path` 用 tuple 不用 list：frozen dataclass 要求字段可哈希，
  tuple 不可变且可哈希，list 不行。空元组 () 表示引言。
- `start_char/end_char`：块可溯源。W5 评测回溯证据、W3 存库定位都要靠它。
- frozen dataclass：不可变，W3 存库/embedding 缓存时可安全哈希（W1 先例）。

```python
@dataclass
class ChunkResult:
    chunks: list[Chunk] = field(default_factory=list)

    def stats(self) -> dict[str, int]:
        """返回分块健康统计，至少包含：
           total      = len(chunks) 总块数
           parents    = 父块数（parent_id is None）
           children   = 子块数（parent_id is not None）
        """
        ...
```

为什么装统计：分块完打印 `total=127, parents=38, children=89` 一眼看出
分块是否健康；W5 评测异常时靠它判断"是不是块切得太碎"。只装这几个，
不堆死参数（W1 教训）。

## 3. API 签名（写在 fixed.py / structured.py）

```python
# fixed.py —— 固定长度分块
def chunk_fixed(doc: Document, size: int, overlap: int = 0) -> ChunkResult:
    """把 doc.text 按固定长度切成块。

    行为契约：
    - size < 1 或 overlap < 0 或 overlap >= size → ValueError
    - 空文本（doc.text 为空）→ 返回空 ChunkResult，不报错
    - 步长 = size - overlap，最后一块允许不足 size（保留尾部，不丢字）
    - chunk_id = f"{doc_id}#{start_char}"（同一文档内唯一、稳定）
    - heading_path 恒为 ()，parent_id 恒为 None（本策略无父子）
    """

# structured.py —— 标题栈分块（父子结构）
def chunk_structured(
    doc: Document,
    max_section_size: int = 2000,
    child_size: int = 500,
    child_overlap: int = 50,
) -> ChunkResult:
    """沿 Markdown 标题层级切块，超长章节切子块。

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
```

## 4. 标题栈算法要点（structured.py 的核心）

一个辅助函数负责解析标题，一个负责分节：

```
_extract_headings(text) -> list[Heading]
    Heading = (level: int, name: str, line_start: int)
    - 只认 ATX 标题：行首是 1~6 个 # 且后跟空格，如 "# 一"
    - "正文 # 不是标题"（# 不在行首）不认
    - 无名字的 "#" 单独一行不认

_split_sections(text, headings) -> list[(start, end, path)]
    遍历标题，维护一个栈（list of Heading）：
    - 遇到新标题：先 pop 掉所有 level >= 新标题 level 的栈顶，
      再 push 新标题。（这是跳级处理的关键）
    - 每个标题的正文区间 = [本标题行起始, 下一个标题行起始)
      → 标题行本身属于本节的块，不丢字
    - 区间起始即本标题的 line_start；最后一个标题的区间到文本末尾
    - 引言区间 = [0, 第一个标题的 line_start)，path = ()
```

边界情况自查（写测试时要覆盖）：
- 引言：文本开头不是标题 → 第一个 section 是引言块
- 跳级：`# 一` → `### 1.1.1`，栈从 [一] 直接变 [一, 1.1.1]
- 同级回落：`## 1.1` 后遇 `# 二`，pop 掉 1.1 再 push 二
- 连续标题、空正文（标题后无内容）：区间 start >= end 时跳过该块
- 表内 # 不误判："| # |" 这类行不在行首，不认

## 5. 验收自查清单（写完代码后逐项打勾）

- [ ] `chunk_fixed("", ...)` 返回空结果，不抛错
- [ ] `chunk_fixed` 传 overlap >= size 抛 ValueError
- [ ] overlap=0 时，把 chunks 按 start_char 排序拼回 == doc.text（不丢不重）
- [ ] overlap>0 时，答案骑在边界上的内容至少在一个块里完整出现
- [ ] `chunk_structured` 引言是独立块，heading_path == ()
- [ ] 跳级标题路径 == ("一", "1.1.1")，无虚构层
- [ ] 超长章节：父块存在 + 子块 parent_id 都指向父块 + 子块偏移在父块内
- [ ] 顶级块（parent_id is None）拼回 == doc.text
- [ ] stats() 的 parents + children == total
- [ ] 所有块 char_count > 0

## 6. 运行方式

```powershell
# 在 minirage/ 目录下运行（确保 docparser 可导入）
python -m unittest discover -s tests -t .
```

测试文件 `tests/test_chunker.py` 自己写，覆盖上面的自查清单每一项。
