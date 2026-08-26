"""RAGAS 四指标校准评测（E 线）：Faithfulness / AnswerRelevancy /
ContextPrecision / ContextRecall，抽样自企业题集 v1 可答题。

Judge：DeepSeek（OpenAI 兼容）；Embeddings：SiliconFlow BAAI/bge-m3。
参考答案由六册手册原文人工起草（questions_answers.json 前身，先内嵌于本文件）。

Run:
    python eval_ragas.py            # 10 题校准集
    python eval_ragas.py --n 5      # 快速冒烟
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.kb import KnowledgeBase, parse_questions  # noqa: E402

# 参考答案：依据手册原文起草（后续迁移到 data/kb/questions_answers.json）
REFERENCES = {
    "产品 A 一个账号一年多少钱？":
        "产品 A 年付价格为每月 239 元（原价 299 元每月打八折），按年付一年约 2868 元。",
    "企业旗舰版和产品 A 相比多了哪些功能？":
        "企业旗舰版包含产品 A 全部功能，另加多租户隔离、私有化部署、SLA 99.9% 可用性承诺，"
        "支持 20 个数据源接入，并提供专属客户经理与 7×24 小时技术支持。",
    "退款政策是什么？7 天后还能退款吗？":
        "购买后 7 天内可无条件退款；7 天后退款按剩余服务期折算；退款到账时间为 3-5 个工作日。",
    "企业旗舰版的 SLA 是多少？":
        "企业旗舰版承诺 SLA 99.9% 可用性。",
    "入职满 3 年，年假有几天？":
        "入职满 3 年不满 5 年的员工年假为 10 天。",
    "出差住宿标准一线城市每晚多少？":
        "一线城市出差住宿标准为每晚 500 元，其他城市每晚 350 元。",
    "外部培训费用超过多少需要总监审批？":
        "外部培训费用超过 5000 元需部门总监审批。",
    "年假没休完怎么办？":
        "年假未休完可顺延至次年 3 月 31 日，过期作废。",
    "数据库每天几点备份？保留多久？":
        "生产数据库每天凌晨 2:00 进行全量备份，备份保留 30 天。",
    "生产发布窗口是什么时候？":
        "生产发布窗口为每周二、周四的 14:00-16:00。",
}
SAMPLE_QIDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 14]


def build_samples(kb: KnowledgeBase, n: int) -> list[dict]:
    questions = [q for q in parse_questions(
        (ROOT / "data" / "kb" / "questions.md").read_text(encoding="utf-8"))
        if q["kind"] == "answer" and q["qid"] in SAMPLE_QIDS][:n]
    samples = []
    for q in questions:
        reference = REFERENCES.get(q["query"])
        if reference is None:
            print(f"[skip] 未起草参考答案: {q['query']}")
            continue
        result = kb.ask(q["query"], user_id="admin")
        contexts = [e["text"] for e in result["evidence"]] or ["资料不足"]
        samples.append({
            "qid": q["qid"],
            "question": q["query"],
            "answer": result["answer"],
            "contexts": contexts,
            "reference": reference,
        })
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(description="RAGAS 四指标校准")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    try:
        from datasets import Dataset
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas import evaluate
        from ragas.metrics import (answer_relevancy, context_precision,
                                   context_recall, faithfulness)
    except ImportError as error:
        print(f"依赖缺失: {error}\n"
              '安装: pip install "ragas==0.2.15" "langchain-openai==0.3.35" '
              '"langchain-core==0.3.79" "langchain-community==0.3.27"')
        return 2

    if not os.getenv("DEEPSEEK_API_KEY") or not os.getenv("SILICONFLOW_API_KEY"):
        print("缺少 DEEPSEEK_API_KEY 或 SILICONFLOW_API_KEY")
        return 2

    kb = KnowledgeBase(ROOT / "data" / "kb")
    print(f"构建 {args.n} 条样本（真实问答链路）...")
    samples = build_samples(kb, args.n)
    if not samples:
        print("没有可用样本")
        return 1

    dataset = Dataset.from_list([
        {"question": s["question"], "answer": s["answer"],
         "contexts": s["contexts"], "reference": s["reference"]}
        for s in samples])

    judge = ChatOpenAI(model="deepseek-chat",
                       base_url="https://api.deepseek.com/v1",
                       api_key=os.getenv("DEEPSEEK_API_KEY"),
                       temperature=0)
    embeddings = OpenAIEmbeddings(
        model="BAAI/bge-m3",
        base_url="https://api.siliconflow.cn/v1",
        api_key=os.getenv("SILICONFLOW_API_KEY"),
        check_embedding_ctx_length=False,
    )

    print("RAGAS 评估中（每题约 4 类 judge 调用，请耐心等待）...")
    result = evaluate(dataset,
                      metrics=[faithfulness, answer_relevancy,
                               context_precision, context_recall],
                      llm=judge, embeddings=embeddings)

    df = result.to_pandas()
    metric_cols = [c for c in df.columns
                   if c in ("faithfulness", "answer_relevancy",
                            "context_precision", "context_recall")]
    per_metric = {c: round(float(df[c].mean()), 4) for c in metric_cols}

    rows = []
    for i, s in enumerate(samples):
        row = {k: s[k] for k in ("qid", "question")}
        for c in metric_cols:
            v = df[c].iloc[i]
            row[c] = None if v != v else round(float(v), 4)  # NaN 安全
        rows.append(row)

    out = Path(args.json_out) if args.json_out else \
        ROOT / "docs" / "EVAL_RAGAS.json"
    out.write_text(json.dumps({"mean": per_metric, "rows": rows},
                              ensure_ascii=False, indent=2),
                   encoding="utf-8")

    print("\n===== RAGAS 均值 =====")
    print(json.dumps(per_metric, ensure_ascii=False, indent=2))
    low = [(r["qid"], m, r[m]) for r in rows for m in metric_cols
           if r.get(m) is not None and r[m] < 0.6]
    if low:
        print("\n低分条目（人工抽检重点）:")
        for qid, m, v in low:
            print(f"  qid={qid} {m}={v}")
    print(f"报告已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
