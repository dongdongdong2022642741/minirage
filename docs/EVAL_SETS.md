# MiniRAG 评测题集总索引

> 五套题集、四个层级。自建两套用 MiniRAG 格式放 `data/kb/`；
> 官方三套保持原始格式（不可改动），由对应评测脚本直接读取。
> 一键聚合入口：`python eval_all.py`（默认只列清单，`--run` 选择执行）。

| # | 题集 | 来源 | 规模 | 位置（格式） | 运行命令 | 最近结果 |
|---|---|---|---|---|---|---|
| 1 | 企业业务 v2 | 自建 | **60**（48可答+12拒答） | `data/kb/questions.md` | Web「跑评测」或 `kb.run_eval()` | **60/60 满分**（docs/EVAL_ENTERPRISE_V2.json） |
| 2 | ACL 权限矩阵 | 自建 | **34 行**（5 身份+状态探针） | `data/kb/acl_cases.json` | `python eval_acl.py --json docs/EVAL_ACL.json` | 34/34，越权泄露率 0 |
| 3 | NoMIRACL zh 官方拒答集 | NoMIRACL 官方 | **3,770 query**（dev 393+1082 / test 920+1375） | `data/nomiracl/chinese/topics/*.tsv`（官方 TSV） | `python eval_nomiracl_rag.py --gate 0 [--n-rel N] [--split test]` | dev 抽样 30+30：能答率 26/30；**幻觉率 14/30**（门控负结果见 PROGRESS） |
| 4 | CRUD-RAG 分层问答 | IAAR-Shanghai 官方 | **10,421 问**（1Doc 2510 / 2Docs 3791 / 3Docs 4120 题组 2400） | `third_party/CRUD_RAG_data/quest_eval/QA*.json` + `meta/split_merged.json`（官方 JSON） | `python eval_crud_rag.py --total N` | 首跑 192 题闭世界切片：包含匹配 54.2% |
| 5 | RAGAS 校准样本 | 复用 #1 的可答题 + 参考答案 | 抽样 n 可配（基准 10） | 样本由 `eval_ragas.py` 现场生成；参考答案 `data/kb/questions_answers.json` | `python eval_ragas.py --n 10` | F .95 / AR .90 / CP .95 / CR 1.00 |

## 口径红线（报告时必须带上）

- #1/#2 是**确定性规则判分**（关键词/泄露断言），可做 CI 门禁。
- #3 官方指标口径 = relevant 能答、non_relevant 必须弃答；我们的"幻觉率"即官方 Hallucination 方向。**单跑噪声地板 ±10pp（n=30, temp=0.1）**，结论性数字需 `--n-rel/--n-nonrel ≥100` 且建议 temperature=0。
- #4 当前为**闭世界切片 + 包含匹配**，严于官方 LLM-judge 与全库检索设定，**禁止与论文数字互比**；定位内部回归基线。扩全量需嵌入约 8 万文档（成本另评）。
- #5 judge=DeepSeek 存在自偏置可能，对外引用前必须完成人工抽检。

## 尚未执行的部分（诚实清单）

- [ ] #3 test split 全量（1475+920）
- [ ] #4 全量 10,421 问与全语料开世界设定
- [ ] #5 低分人工复核（qid=1/qid=14）与 n=50 扩样
