# RAGFlow 源码深潜（infiniflow/ragflow，main 分支）

> 目标：把 MiniRAG 与工业级 RAG 引擎逐环节对齐，产出能上简历的认知。
> 素材来源：浅克隆 + sparse checkout（rag/deepdoc/api/docs/sdk），精读关键文件，行号可溯源。

## 0. 一句话定位

RAGFlow 是"文档理解（DeepDoc 视觉解析）+ 可插拔检索后端（ES/Infinity/OpenSearch）+ LLM 生成"的企业级 RAG 引擎；检索不是它自己写的，是**把 Lucene 语法、向量检索、加权融合全部下推到搜索引擎一次算完**——这与 MiniRAG 亲手实现每层的路线形成最本质的对照。

## 1. 架构总览（AGENTS.md 为本仓库操作手册）

```
web/ (React+TS) → api/ (Python Quart + Peewee ORM) → rag/ (检索/生成/图谱/提示词)
                                                    → deepdoc/ (视觉解析：OCR/Layout/TSR)
                                                    → agent/ (工作流画布)
Go 侧（新架构主战场，Python 侧逐步收敛）：
  internal/ingestion/ (解析→分块→tokenize→extract 的 DSL 驱动流水线)
  internal/parser/    (markdown/html/pdf/docx/xlsx 原生解析, pdfium/pdf_oxide 静态库)
  internal/engine/    (ES/Infinity 检索后端适配)
  cmd/ (ragflow_main / ragflow-cli)
运行时：MySQL/PostgreSQL + Redis + MinIO + ES/Infinity/OpenSearch（按配置）
```

- 测试分层（Go）：unit（默认）/ integration / e2e / manual（本地专属），build tags 隔离，`bash build.sh --test` 驱动；Python 侧 pytest
- 检索后端可插拔：`rag/utils/es_conn.py` / `infinity_conn.py` / `opensearch_conn.py` / `oceanbase_conn.py` / `serenedb_conn.py` 等——**RAGFlow 不做检索实现，做检索适配**

## 2. 链路逐环节对照

### 2.1 文档解析：DeepDoc（我们缺的整层）

`deepdoc/parser/` 17 种格式解析器（pdf/docx/excel/ppt/html/markdown/epub/json/ocr……）+ `deepdoc/vision/`（OCR、LayoutRecognizer、TableStructureRecognizer）。

PDF 管线（`deepdoc/parser/pdf_parser.py`，2145 行）：
- pdfplumber 文本层抽取 + pypdf 兜底 + 自研 LayoutRecognizer（**xgboost 版面分类**：标题/正文/页眉页脚/图片）+ OCR（图片型 PDF 全文兜底）+ TableStructureRecognizer（表格结构识别，TSR）
- KMeans + silhouette 聚类判多栏；标题树（TOC 检测、`rag/nlp/__init__.py` 的 hierarchical_merge）用于父子分块
- 新版接入 docling / mineru / mistral 外部解析器（`docling_parser.py` 等），且有 Go 原生解析（pdfium/pdf_oxide）

对照：MiniRAG 只做文本容错解析（W1）；RAGFlow 用视觉模型解决"扫描件/版面/表格"——这是企业知识库与纯文本语料的本质差距。

### 2.2 分词：jieba 词 + 2-gram 子词双通道（和我们撞了思路，但他们有词表）

`rag/nlp/rag_tokenizer.py`（44 行，核心在 **infinity.rag_tokenizer，C++/Rust 原生**）：
- `tokenize()`：jieba 式词典分词（主通道，粗粒度）
- `fine_grained_tokenize()`：**2-gram 子词**（细粒度通道）
- 双通道并存：`rag/nlp/__init__.py` 的 `tokenize()` 同时写 `content_ltks`（主）与 `content_sm_ltks`（2-gram 细粒度）
- 繁简转换（tradi2simp）、全半角（strQ2B）

检索侧（`rag/nlp/query.py` FulltextQueryer）把词表能力全用上：**词权重（term_weight）→ 同义词扩展（synonym）→ 短语查询（"tk"~2 前缀模糊）→ 细粒度 2-gram 兜底**，编译成 Lucene 语法一次下推。

对照：MiniRAG 纯 2-gram 无词表（零维护、穷举召回）；RAGFlow 词典词 + 2-gram 兜底（精度 + 召回双保险，但分词器是 C++ 原生、词表靠训练/维护）。我们的"2-gram 首 token 误判"坑，在他们这里由 fine-grained 通道天然规避。

### 2.3 检索与融合：引擎内 weighted_sum，不是 RRF

`rag/nlp/search.py` `Dealer.retrieval()`（549 行起）：
- 候选窗口 `RERANK_LIMIT`（按 page_size/top 对齐分页块，rerank 必须在块内）
- **`FusionExpr("weighted_sum", topk, {"weights": "0.001,1"})`（line 210）**：全文 vs 向量在搜索引擎内部一次融合——不是 Python 里做 RRF
- 用户级权重 `vector_similarity_weight`（默认 0.3，可配 0~1）：`term_similarity_weight = 1 - vector_similarity_weight`，决定 min_match 与 rerank 混合
- 相似度阈值过滤（默认 0.2）；`PAGERANK_FLD` 作为 rank_feature（PageRank 类信号参与打分，默认权重 10）
- 双通道向量：匹配用 Dense + 稀疏（Infinity 原生稀疏检索），`matchText + matchDense + fusionExpr` 一次请求

对照：MiniRAG 在 Python 里显式 RRF/加权融合（可解释、可评测——我们实测出 RRF 排名平权亏损）；RAGFlow 把加权和藏进引擎表达式，**不可归因**。这正是我们 W6 复盘里"可解释性强"论点的源码级实锤。

### 2.4 rerank：外部模型优先，本地加权兜底

`rag/nlp/search.py` + `rag/llm/rerank_model.py`（648 行，**15+ 厂商适配**：Jina/Cohere/BGE/SiliconFlow/Qwen/NVIDIA/Bedrock/Voyage/TogetherAI/GreenPT/XInference/LocalAI/OpenAI……）：
- `rerank_by_model`：外部 reranker 分数归一化 [0,1] 后与 token/向量相似度**加权 blend**（单尺度统一）
- 本地 `rerank`：`hybrid_similarity`（token 0.3 + 向量 0.7 加权）
- ES 路径 `rerank_with_knn`：二次 KNN-only 调用取干净余弦分（chunk 向量不再常驻返回）

对照：MiniRAG 的 rerank（bge-m3 交叉打分集内重排）对应他们的"外部 rerank 模型"路线；"rerank 救不回 recall"坑在他们这由 RERANK_LIMIT 窗口 + top 参数显式管理候选池（同样存在集合天花板）。

### 2.5 生成 + 引用：提示词教 + 后处理注入"双保险"

- 提示词侧（`rag/prompts/citation_prompt.md`，122 行规则）：**8 类必须引用**（数据/时间/因果/比较/术语/归属/预测/争议）、5 类不引（常识/过渡句/自有分析）、格式 [ID:i][ID:j]、每句最多 4 条、句末标点前
- 后处理侧（`rag/nlp/search.py` `insert_citations`，251-328 行）：**句子级 embedding 溯源注入**——答案按多语言标点切句（含阿拉伯文标点）→ 每句 embed → hybrid_similarity（token 0.1 + 向量 0.9）→ 阈值 0.63 起、*0.8 递减直到有命中 → 每句最多 4 个 chunk、去重贴 [ID:c]
- 调用点：`api/db/services/dialog_service.py:869`（0.1/0.9）与 :1790（0.7/0.3）——权重可配
- 生成器（`rag/prompts/generator.py`，1003 行）：Jinja 沙箱模板 + json_repair + 上下文利用度 INPUT_UTILIZATION=0.5 + 多轮引用

对照：MiniRAG 只做了提示词侧（LLM 标 [1][2] + 事后合法性校验）；RAGFlow 加了一层**生成后溯源注入**——LLM 标漏的由 embedding 补，这就是"引用可信度"的工程解。可上简历：我们评测里的"引用合法 15/15"若加这层后处理可以更稳。

### 2.6 分块：DSL 化 + 多种策略 + 多模态上下文

`rag/flow/chunker/token_chunker.py`（439 行，新 DSL 管线）：
- `TokenChunkerParam`：delimiter_mode（delimiter/one）、chunk_token_size=512、delimiters、overlapped_percent、**children_delimiters（子分块）**、**table_context_size / image_context_size（表格/图片上下文拼接）**
- 解析器会打 `@@page\tleft\tright\ttop\tbottom##` 坐标 tag，分块时剥离（Python/Go 双实现一致，`_TAG_RE`）
- 旧管线 `rag/nlp/__init__.py`：`naive_merge`（默认 128 tokens）+ `merge_paragraphs`（MergeStrategy.OVER_CAP 等）+ `hierarchical_merge`（标题树父子分块）+ overlap（无条件 overlap）
- 分块后 tokenize 双通道（content_ltks/content_sm_ltks）随 chunk 入索引

对照：MiniRAG 的"标题栈父子分块 + overlap"对应 hierarchical_merge；表格/图片上下文与子分块是我们没做的（我们评测 5304336 类表格/简称问题在这层有解）。

### 2.7 评测：ranx 检索基准 + 内置问答对

`rag/benchmark.py`：**ranx**（IR 评测库）Qrels/Run 结构、MS MARCO 风格、recall/MRR 指标、tqdm 进度；知识库级相似度阈值/向量权重参与评测配置。
（另有对话/问答对评测走 api，本次 sparse 未覆盖 tools/ 与 web 端。）

对照：MiniRAG 自研召回评测（393 条 NoMIRACL）与 RAGAS 式 e2e（faith/rel/引用合法性）——指标同源（recall/MRR/faith），差异在工程化与规模。ranx 提示我们：评测格式对齐 IR 社区（Qrels/Run）可复现性更强。

## 3. 三个"没想到"（面试可讲的深度点）

1. **融合不发生在应用层**：`weighted_sum` 权重 0.001:1 直接写进 ES/Infinity 表达式。商业引擎把"检索数学"压进数据库——效率优先、可解释性牺牲。MiniRAG 反着来，所以能讲出 RRF 排名平权的亏损（输 66/平 319/赢 8）。
2. **引用是后处理工程而非提示词魔法**：句子级 embedding 注入（0.63 阈值降阶梯 + 每句 4 条去重）。LLM 提示词只是第一道，溯源注入是第二道——"引用可信度"是可工程化的。
3. **分词双通道是标准答案**：jieba 词 + 2-gram 子词并存（ltks/sm_ltks 双字段），检索时词权重/同义词/短语模糊/细粒度兜底四级放大。MiniRAG 的纯 2-gram 是它的"无词表特化版"。

## 4. 对照表（简历素材）

| 环节 | RAGFlow | MiniRAG | 简历落点 |
|---|---|---|---|
| 解析 | DeepDoc 视觉管线（OCR/Layout xgboost/TSR）+ Go 原生 | 文本容错解析（W1） | 认知差距：企业文档 vs 纯文本语料 |
| 分词 | C++ 原生 jieba 词 + 2-gram 子词双通道 | Python 纯 2-gram 无词表 | 双通道取舍、零维护特化 |
| 检索 | ES/Infinity 适配，Lucene 语法下推 | 自研 BM25 倒排 + bge-m3 暴力检索 | 自研 vs 适配；检索数学的位置 |
| 融合 | 引擎内 weighted_sum (0.001:1) | Python 显式 RRF/加权 + 393 条实测 | 可解释性 vs 效率；实测数据背书 |
| rerank | 15+ 厂商外部模型 + 本地加权兜底 | bge-m3 交叉打分集内重排 | 候选窗口管理（集合天花板共识） |
| 引用 | 提示词规则 + 句子级 embedding 注入 | 提示词标注 + 合法性校验 | 双保险思路，可补后处理层 |
| 分块 | DSL 管线 + 标题树 + 表格/图片上下文 | 固定长度 + 标题栈父子分块 | 多模态上下文是扩展方向 |
| 评测 | ranx + MS MARCO 基准 | NoMIRACL 393 条 + RAGAS 式 e2e | 指标同源，格式对齐社区可复现 |

## 5. 简历转化（一句话版）

> 在从零实现 RAG 检索链路（自研 BM25、2-gram 分词、显式 RRF/加权融合、集内 rerank、RAGAS 式评测）后，深读工业级开源引擎 RAGFlow 源码（Python+Go 双栈、DeepDoc 视觉解析、引擎内 weighted_sum 融合、句子级 embedding 引用注入），能逐环节对齐两种实现的取舍——包括商业引擎"融合下推数据库牺牲可解释性"与自研"显式融合可归因实测亏损（RRF 输 66/赢 8）"的对比结论。
