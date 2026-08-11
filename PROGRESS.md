# MiniRAG 进度清单

> 恢复点文件：每次会话从这里开始。

## 当前项目
MiniRAG —— 复刻 RAGFlow 检索链路的带引用问答系统

## 立项决策记录（已定）
- 核心：文档解析 → 分块 → 混合检索(BM25自研+向量) → rerank → LLM 带引用生成
- 向量检索：W2 前用自研 numpy 暴力最近邻，之后换工业库(FAISS)，写成对比实验
- LLM：DeepSeek API
- 评测集：NoMIRACL 中文子集（recall@k / MRR）+ 自研 RAGAS 式端到端评测（faithfulness / answer relevancy）
- 用户 Python 水平：B（numpy/pandas 基础会，没写过完整管线）
- Embedding 模型：BAAI/bge-m3，经硅基流动 API 调用（OpenAI 兼容 /v1/embeddings，1024 维）
- 待决：BGE-M3 自带 sparse 权重，是否仍自研 BM25？

## 已完成模块
- [x] 立项（2026-08-06）
- [x] 地基确认（2026-08-06）：Embedding 概念 / DeepSeek API 调用 / 评测指标定义
- [x] 任务 1：DeepSeek chat 调用跑通（输出见会话记录）
- [x] W1 文档解析（docparser 模块：Document 对象 + 目录/空文件/编码容错 + 测试通过）
- [x] W2 分块策略（chunking 模块：固定长度 size+overlap + 标题栈父子分块 + 24 个测试，总计 39 个通过）
- [x] W3 倒排索引 + 向量检索（index 模块：自研 BM25 + bge-m3 向量 + 混合 demo + 向量缓存，新增 19 测试，总计 58 个通过）
- [x] W4 混合融合 + rerank（fusion.py：RRF + 加权和双算法统一接口；rerank.py：在融合 top-N 内捡回分数重排；demo 升级四列对照 + recall@5 判定，新增 28 测试，总计 86 个通过）
- [x] W5 评测闭环（eval_retrieval.py + eval_e2e.py：NoMIRACL 393 条五路检索评测 + 全量语料 15 query × 2 路端到端，新增 13 测试，总计 99 个通过）
- [x] W6 复盘 + 简历化（docs/W6_RETROSPECTIVE.md：六条坑四段式复盘 + MaxKB 对照 + STAR 简历表）

## 收官状态（2026-08-09）
- 检索评测（全量 31,826 篇 × 393 query）：vector 最优 recall@10 0.9358 / MRR@10 0.7684；RRF 平均低于较优单路（输 66/平 319/赢 8，排名平权所致）；rerank 提升 recall@5 0.6469→0.7209（集合不变只修排序）
- 端到端（全量 + 喂料 top-10 × 15 query）：能答率 15/15（从 60% 修到 100%），faithfulness 0.955（vector）/ 0.844（rrf+rerank），rel 0.781/0.770，引用合法 15/15
- 测试 99 条全绿；评测缓存化后重跑零 API
- 交付物：eval_retrieval.py / eval_e2e.py（可复现）/ docs/W5_REPORT.md / docs/W5_E2E_REPORT.md / docs/W6_RETROSPECTIVE.md

## 待办
- 收官遗留（非本轮范围，见复盘第五节）：e2e 样本扩到 50；relevancy 换 LLM-as-judge；历史缓存文件（e2e_cache_N4000.json 等）可清理
- [x] 评测集下载（NoMIRACL zh）：corpus + dev/test relevant/non_relevant topics/qrels

## 遇到的问题 / 坑
- 标题行本身（如 `# 一`）曾落在所有 chunk 之外导致拼接丢字，修正为 section 从标题行开始
- overlap 的实际效果是块数增多（步长 = size - overlap 变小），测试断言曾写反
- PowerShell 重定向会破坏 UTF-8 中文显示（`�?`），数据本身正常，用 Read 工具直接读 UTF-8 文件确认
- NoMIRACL relevant qrels 有少量 doc_id（dev 32 行/test 63 行）不在随包 corpus 中，加载评测集时需过滤
- 2-gram 把"什么/时候"这类功能词也当稀有词：1000 篇百科切片里它们 df≈5、idf≈5.2，与主题词"二战"同级，短文命中功能词会被抬升排名。idf 度量的是"已索引语料的稀有度"而非人感重要度，全量 37,599 篇会缓解
- 1000 篇切片历史内容占比高，idf 偏斜、代表性差，demo 的定性结论不可外推到全量
- demo 三种 query 的 top-3 重叠极低（0/1/1）：两路召回在互补，混合检索有真实增益
- BM25 分数量纲（11.87~23）约为向量（0.45~0.73）的 30 倍，W4 融合前必须先归一化或用 RRF
- 融合不是免费午餐：向量满分 + BM25 全错时（query 1719936#0，vector recall@5=1.000、bm25=0.000），RRF 把 BM25 噪声并入，相关 doc 被挤出 top-5（rrf 掉到 0.400）
- rerank 只能在融合集合内重排、救不回 recall：相关 doc 若已被融合丢出 top-5，rerank 无法找回（query 1719936#0：rrf=0.400 且 rerank=0.400）
- demo 只有 3 个有标注 query，融合时好时坏（1 条 FAIL、1 条 PASS、1 条平手），**不可据此调参**——调 RRF_K/RAW_K/alpha 必须等 W5 在 393 条全量上做，3 条上调参=过拟合
- 缓存假数字：语料 4,000→31,826 后检索结果全变，答案缓存若复用旧文件=假数字（e2e_cache_N4000.json 即此原型）；修复：query 嵌入缓存按 N+内容 hash 命名，e2e 缓存文件名带 N/topK、内嵌配置指纹加载校验（fail loud，不匹配报错）
- 喂料 top-10 提平均 faith 但掺噪声：7258925 rerank faith 1.0→0.5（无关文档带偏模型），证据规模 ×1.9；净收益为正，做成 CLI 可配参数（`python eval_e2e.py 15 5|10`）
- relevancy 的 0.45 余弦阈值退化：收官轮通过率 100%、最低 0.68，无区分度=没测；应换 LLM-as-judge（复盘第五节）
- e2e 15 条样本下平均指标被单条采样波动主导（5304336 faith 在 0.2~1.0 间摆动），结论仅实验性；扩到 50 条才可靠
- 测试边界硬编码 top-5（[6] 越界）在 TOP_FOR_ANSWER=10 后断言反转失败——边界断言须从配置常量推导，不许写死

## 真实数据
- NoMIRACL 中文子集：`data/nomiracl/chinese/`（verify_nomiracl.py 验证通过）
- corpus：37,599 条 passage（JSONL gzip）
- topics：dev/test × relevant/non_relevant（2 列 TSV）
- qrels：dev/test × relevant/non_relevant（4 列 TSV）；dev.relevant 393 个 query 有标注
