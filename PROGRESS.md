# MiniRAG 进度清单

> 恢复点文件：每次会话从这里开始。

## 当前项目
MiniRAG —— 复刻 RAGFlow 检索链路的带引用问答系统

## 学习与协作原则（2026-08-23 重大调整）

> 目标不是尽快堆完一个“企业级 RAG”，而是训练架构选型思维、工程拆解能力和可验证落地能力。知道为什么、怎么设计、如何验证之后，才使用 AI 辅助编码。

- 后续禁止由 AI 直接完成整个企业化阶段；每次只推进一个可以独立解释、实现和验证的小任务
- 每个任务先由用户回答五个问题：业务问题是什么、现状为什么不够、有哪些可选方案、为什么选择当前方案、如何证明实现正确
- 编码前必须先确定：数据模型、模块边界、关键接口、状态变化、失败处理、测试方案；不能先生成代码再倒推设计理由
- AI 的角色调整为：提供真实项目参考、提出追问、评审设计、解释取舍、指出风险、给局部示例、协助排错和验证；用户负责关键方案选择和核心代码理解
- 分工基线（2026-08-23 明确）：用户确认选型、技术路径、接口与状态机、验收标准，并对 AI 代码逐行审查；AI 在用户确认的设计内编写实现与测试、查证事实、填写取舍对比表供拍板、执行验证。用户手写代码不是默认要求，仅核心难点且用户主动要求时才做；但 AI 写的每一段，用户必须能脱稿复述数据流与失败路径
- Vibe coding 的使用条件：用户能先口述完整数据流，能解释关键字段和失败路径，能预测测试结果，生成代码后能逐段审查
- 每一步完成后必须复盘：实际结果、踩坑、与 RAGFlow/MaxKB 的差异、当前方案的规模边界、下一阶段为什么必要
- 已完成的阶段 1/2 不作废，改为逆向训练材料：重新阅读文档目录、版本管理、多格式适配器和测试，做到能脱离代码讲清设计并修改关键行为
- 详细训练流程见 `docs/ENTERPRISE_UPGRADE_LEARNING_GUIDE.md`

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

## 真实复跑记录（2026-08-17，全量重跑 + 真调 API）

> 背景：恢复点后向量库重建（31,826 篇经 SiliconFlow API，512s），393 query 检索评测、15 query 端到端、30 题题目集评测全部真实执行；e2e 答案缓存与 query 嵌入缓存已落盘，此后重跑零 API。

- 检索评测：vector recall@10 0.9362 / MRR@10 0.7696（与收官记录 ±0.001 内，向量库重建的浮点/API 漂移）；bm25、rrf、rrf+rerank 与记录逐位一致
- RRF 输 66 / 平 320 / 赢 7（记录为 66/319/8，一条平局翻转）；rerank 改善 recall@5 86 个 query（记录 88）——检索侧数字可复现、可信
- 端到端 15 query：两路能答率 15/15，faith 0.951（vector）/ 0.832（rrf+rerank），引用合法 14/14、15/15
- 采样波动实证：单条与收官记录差异 ±0.2（1719936 rrf 0.800→1.000、1269860 rrf 0.667→0.250）；7832551 vector 本次 0 引用（记录 OK(1)）——"单条采样主导平均指标"判断成立
- 30 题题目集（27 可答 + 3 拒答）：可答 27/27 = 100%、拒答 3/3 = 100%、引用合法 27——当前状态实测全过；"66%→100%"提升过程无历史落档，不可验证
- 简历更新：docs 数字已同步进《申令瑞-AI应用开发简历.docx》（MiniRAG 段含 recall@10 0.936 / MRR@10 0.770 / 能答率 100% / 30 题全过）

## 待办
- 收官遗留（非本轮范围，见复盘第五节）：e2e 样本扩到 50；relevancy 换 LLM-as-judge；历史缓存文件（e2e_cache_N4000.json 等）可清理
- [x] 评测集下载（NoMIRACL zh）：corpus + dev/test relevant/non_relevant topics/qrels
- **企业化升级 M10 实施记录（2026-08-22）：**
  - [x] 阶段 1 文档生命周期地基：新增 SQLite 文档目录和不可变 Blob 存储；文档使用稳定 `document_id`，版本使用独立 `version_id`，内容使用 SHA-256；同名同内容幂等，同名内容变化创建新版本，失败版本不替换当前可用版本，删除采用逻辑删除
  - [x] 阶段 1 索引身份升级：BM25/向量索引由展示标签改为版本化 `chunk_id`；问答证据保留 `document_id / version_id / chunk_id / heading_path / start_char / end_char`，解决同名标题覆盖和更新后引用不可追踪问题
  - [x] 阶段 1 兼容迁移：首次启动自动把现有 `data/kb/docs/` 登记为版本 1；本机 6 个历史文档均迁移为 ready；索引 Schema/Embedding 模型进入指纹，旧标签缓存不再误用
  - [x] 阶段 1 验证：新增 8 条文档生命周期测试，总测试数 99→107，`python -m unittest discover -s tests -t . -v` 全绿；`compileall` 通过
  - [x] 阶段 2 多格式解析：PDF/DOCX/HTML/TXT/Markdown 统一转换为 `Document` 和 Markdown 文本后复用结构化分块；DOCX 保留标题/段落/表格，HTML 清理脚本并保留标题/列表/表格，PDF 按页插入页码标题；扫描 PDF 无可提取文本时明确失败并提示 OCR，旧 `.doc` 和图片语义不在本阶段范围
  - [x] 阶段 2 验证：增加 DOCX/HTML/PDF/扫描 PDF/二进制解析失败/HTML 版本链路共 6 条测试，总测试数 107→113；`unittest`、`compileall`、`git diff --check` 通过
  - [x] 阶段 3 增量索引与原子发布（2026-08-23 完成）：
    - Chunk 向量缓存：键 = sha256(model + "\\0" + text)，存储于 `cache/vectors/<key前2位>/<key>.npy`；单向量原子读写，损坏/维度不符文件自动删除自愈；批量 `build_with_cache` 同批次去重，返回 `{total, reused, embedded}` 统计
    - 原子代际发布：`cache/generations/gen_{fp}/` 沙箱构建 bm25.pkl + vec/ + manifest.json，`os.replace` 原子更名发布，内存指针最后热切换；构建失败清理沙箱并保留旧索引继续服务（A1 策略）
    - 代际保留策略：仅保留当前与上一代共 2 代目录（B1 策略），全局 vectors 缓存不删
    - 阶段 3 验证：新增 17 条测试（缓存键 4 + 单向量缓存 5 + 批量增量 4 + 原子发布/失败回滚/代际清理 2 + restore 相关 8 条中 2 条索引相关），总测试数 113→136 全绿；实测文档恢复后指纹回到旧值直接命中磁盘缓存，零 Embedding API 调用
  - [x] 阶段 4 ACL（2026-08-23 完成）：用户-文档白名单（默认拒绝），查询时过滤夹在召回与融合之间并超量取候选 RAW_K×2；允许集与 active 文档求交堵住删除/未重建窗口期泄露；`/api/users` `/api/acl` + 前端身份下拉；权限矩阵题集 `acl_cases.json` + `eval_acl.py` 实测 **34/34、越权泄露率 0/34**；零证据确定性拒答短路（不调 LLM，由矩阵评测 ghost 用例实测发现）
  - [x] 阶段 5 异步与运维（2026-08-23 完成）：版本状态 queued/parsing/ready/failed + build_jobs 台账 chunking/embedding/publishing/done/failed；单飞后台重建（重复触发 409，构建期沿用旧代际继续服务）；仅网络类错误指数退避重试 ≤3 次（HTTP400 类不重试）；append-only audit.jsonl（ask 事件不落问题原文）；`/api/stats` P95（500 窗口）+ 相对成本模型；`/api/builds` 历史
  - [x] 学习模式恢复点（2026-08-23 二更）：阶段 4/5 已按"设计拍板→AI实现→用户审查"流程完成并提交（3b98b3d）；评测三线 C/D/E 首跑落地并提交（12f9d6d，详见文末评测小节）；B 线企业题集 v2 自审定稿上线；当前最大杠杆 = 相关性阈值门控压幻觉率（C 线 non_relevant 幻觉率 56.7% → 目标 <20%，且 relevant 能答率损失可控）
- **评测升级路线（2026-08-22 已定）：**
  - 定位：不以单一总分评价 RAG，拆成检索层、生成层、鲁棒拒答层、企业工程层分别报告；公开基准负责横向可比，企业自建题集负责真实业务验收
  - NoMIRACL 中文继续保留：现有全量检索结果作为基线，负责 Recall@k / MRR；后续补跑官方 relevant/non_relevant 任务，以 Hallucination Rate 和 Error Rate 测“无证据不乱答、有证据能识别”；因缺少完整高质量参考答案，不作为主要端到端答案正确性题集
  - CRUD-RAG 作为新增主要中文端到端官方基准：先接入问答子集并按单文档/双文档/三文档分层，稳定后再扩展总结、续写和幻觉纠正；所有题目必须经过 MiniRAG 自己的解析、分块、BM25、向量、融合、rerank、Top-K 和 DeepSeek 链路，禁止直接使用 CRUD-RAG 自带检索结果冒充本系统结果
  - Ragas 作为评测框架而非题集：把 CRUD-RAG 的 question/reference 与 MiniRAG 实际产生的 retrieved_contexts/response 适配为统一样本，首批计算 Faithfulness、Response Relevancy、Context Precision、Context Recall，视成本再增加 Factual Correctness 和 Noise Sensitivity
  - 企业题集继续复用现有 30 题：负责业务可答率、拒答正确率和引用合法率；ACL 上线后另建权限矩阵题集，为同一问题配置不同用户/角色和 expected_behavior，不能用现有 30 题代替权限测试
  - 指标并列展示：保留 NoMIRACL 确定性检索指标、CRUD-RAG 官方指标、Ragas LLM-as-judge 指标和企业规则指标；禁止跨层平均成一个总分，禁止将现有自研 faithfulness 与 Ragas Faithfulness 混报
  - 执行顺序：①冻结现有 NoMIRACL 基线；②补 relevant/non_relevant 拒答评测；③接 CRUD-RAG 问答 200~500 题固定子集；④接 Ragas 数据适配与四项指标；⑤扩展 CRUD-RAG 其他任务；⑥企业化升级后增加 ACL、版本更新/删除一致性、P95 延迟和成本评测
  - 首阶段验收产物：固定题集清单与版本、可复现命令、检索配置和模型版本、逐题结果、分层汇总、失败案例、运行时间和 API 成本；至少人工抽查 10 题校准 Ragas 中文判断
- **官方 RAGAS 外部校准（不替换现有确定性评测）：**
  - [ ] 最小集成：新增独立评测入口，读取现有 30 题数据和真实检索上下文，不改 `eval_retrieval.py`、`eval_e2e.py` 与线上问答链路
  - [ ] 数据适配：每条样本统一为 `question / answer / contexts / reference / expected_behavior`；27 道可答题提供人工 ground truth，3 道拒答题保留 `expected_behavior=abstain`
  - [ ] 首批只跑四指标：Faithfulness、Answer Relevancy、Context Precision、Context Recall；不把多个指标平均成一个总分
  - [ ] 评测分层：Recall@k/MRR/关键词命中/拒答/引用合法性继续作为确定性门禁；RAGAS 仅作为 LLM-as-judge 外部校准；权限合规率继续由自建权限题集负责
  - [ ] 口径留档：报告必须记录 RAGAS 版本、Judge 模型、Embedding 模型、Prompt/语言、题集版本、检索配置、运行时间和成本；现有自研 `faithfulness` 与 `RAGAS Faithfulness` 分栏展示，禁止混报
  - [ ] 人工校准：抽查至少 10 题，对比人工判断 / 现有规则 / RAGAS；单列“RAGAS 高分但人工判错”和“RAGAS 低分但人工判对”的案例，确认中文同义表达与拒答题是否被误伤
  - [ ] 历史版本对照：在同一冻结题集上比较纯向量、向量+BM25+RRF、增加 rerank、top-k/拒答策略调整，回答“提升来自召回、排序还是模型更愿意回答”
  - [ ] 执行策略：普通提交跑 99 测试+确定性评测；重要版本再跑 RAGAS+人工抽检；RAGAS 未经人工校准前不设 CI 硬门禁
  - [ ] 验收产物：生成三层报告（检索层 / 生成层 / 工程业务层）和版本差异报告，列出新增失败、修复成功、保持不变及失败类型；最终把可复现命令与环境依赖补进 README
- **多模态文档检索（独立任务，先做最小闭环）：**
  - [ ] 样本集：准备至少 10 份同时包含正文、图片、图注和表格的 PDF，冻结 20 道题；题目按“纯文本可答 / 必须读取图片 / 图文联合推理”分类，并为每题标注答案与证据页码
  - [ ] 解析入库：按页提取 PDF 文本、图片和图注；图片经 OCR 获取可检索文本，再由 VLM 生成包含对象、关系、数值和结论的结构化描述；保留 `document_id / page / modality / image_path / caption` 元数据
  - [ ] 统一数据模型：文本块、OCR 块和图片语义块统一进入现有 `Document/Chunk` 链路；图片原文件只作为证据 Artifact 保存，不直接塞进普通文本 Prompt
  - [ ] 检索融合：先复用 BM25 + bge-m3 检索 OCR 与 VLM 描述；若文本跨模态召回无法满足题集，再增加 CLIP 类图文共享向量作为独立召回路，并通过 RRF 或归一化加权融合，禁止预设多路融合一定更优
  - [ ] 上下文生成：命中图片证据时，将缩略图/原图引用、OCR 文本、VLM 描述和相邻正文共同组装；模型支持视觉输入时传图片，否则只使用文本化证据，并在结果中返回页码和图片引用
  - [ ] 评测对照：比较“仅正文”“正文+OCR”“正文+OCR+VLM”“增加图文向量（如实现）”四组配置，分别报告各题型 Recall@k、MRR、答案正确率、引用合法率、平均延迟和单文档处理成本
  - [ ] 失败与缓存：OCR/VLM 调用失败可重试且不阻断整份文档；缓存键包含图片 Hash、模型和 Prompt 版本，文档修改或模型配置变化时只重建受影响图片
  - [ ] 完成标准：20 道冻结题全部可复跑；必须读取图片的题目相较“仅正文”基线有可量化提升；新增解析、缓存失效、检索和引用测试，报告中保留失败案例，不用个别 Demo 代替整体指标
- **企业化升级（M10 场景 A，规划见 面试重点.md 第九节）：**
  - [ ] 文档入库管理：多格式解析（pdf/docx/html/txt）统一为 Document 对象；增量更新 + 块级版本化；删除/替换旧版本
  - [ ] 权限矩阵：文档级 ACL（每个 chunk 继承父文档权限）；检索结果过权限过滤后再进生成；构造权限矩阵测试集（低权限 query 命中高权限文档必须拒答/过滤）
  - [ ] 三层评测落地：检索层命中率@k（复用 eval_retrieval 框架）+ 业务可答率（30 题业务题集 + 人工 ground truth）+ 权限合规率（该拒的拒/该答的答）
  - [ ] 前端可选：简单文件管理 + 检索演示页（时间不够可省，核心是可复现的评测与权限）
  - [ ] 简历落点：每场景一行指标（命中率@k / 可答率 / 权限合规率）+ 一行"从零实现" + 一行评测设计

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

## 评测三线首跑（2026-08-23，均已提交 12f9d6d）

- **C NoMIRACL 拒答评测**（eval_nomiracl_rag.py）：dev 30+30，复用全量索引缓存零嵌入成本，检索走生产同款 RRF+rerank。relevant：能答率 26/30、引用合法 26/26；**non_relevant：幻觉率 17/30 = 56.7%**——检索器总能捞出看似相关的噪声段，模型持噪作答。修复方向：向量 top1 相关性阈值门控（KB_RELEVANCE_GATE）
- **D CRUD-RAG 分层问答**（eval_crud_rag.py + third_party/CRUD_RAG_data/）：官方 QA 按 ID 反查源文档闭世界入库，192 题包含匹配 54.2%（1Doc 61.2% / 2Docs 38.2% / 3Docs 60%，拒答 15）。口径注记：包含匹配严于官方 LLM-judge 且语料为闭世界，数字不可与论文直接对比，定位为内部回归基线；2Docs 凹陷待归因
- **E RAGAS 四指标校准**（eval_ragas.py，ragas==0.2.15 + DeepSeek judge + bge-m3 嵌入）：10 题校准集 faithfulness .95 / answer_relevancy .90 / context_precision .95 / context_recall 1.0；低分人工抽检项：qid1 context_precision=.5、qid14 faithfulness=.5（docs/EVAL_RAGAS.json rows）
- **B 企业题集 v2 上线**：60 题 = 48 可答（数字题全部中文数字双保险词）+ 12 near-miss 拒答；参考答案 questions_answers.json 48 条（RAGAS/人工校准基准）；旧版备份 questions_v1.md。待办：v2 全量 run_eval 新基线

### 幻觉门控实验（2026-08-23，负结果 + 方法论发现）
- 实现：KB_RELEVANCE_GATE 环境变量 + eval_nomiracl_rag.py --gate，向量 top1 低于阈值即清空证据走确定性拒答（机制保留，默认关闭）
- 结果：gate=0.35 时 non_relevant 幻觉率不降反升（14/30 vs 基线 14/30 同批噪声内波动）——实测 dev 60 查询 top1 相似度分布：relevant 中位 0.691 / non_relevant 中位 0.654，重叠严重；bge-m3 余弦对任意中文查询-段落基线偏高，单一绝对阈值原理上不可分
- 方法论发现①：同配置重跑幻觉率 17↔14（±10pp），30 样本 + temperature=0.1 的噪声地板过高；后续结论性评测须 n≥50 且 temperature=0
- 下一步杠杆（按性价比）：①top1 与其余 topk 的相对margin 门控（自适应基线）②引入真正的交叉编码器 reranker 用其分数做门控 ③回答前 LLM 自评证据相关性（多一次调用换拒答精度）
