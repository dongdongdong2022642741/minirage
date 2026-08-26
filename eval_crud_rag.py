"""CRUD-RAG 中文问答子集接入（D 线）：分层（1Doc/2Docs/3Docs）× MiniRAG 全链路。

数据来源：IAAR-Shanghai/CRUD_RAG 官方仓库（third_party/CRUD_RAG_data/）。
每条记录自带源文档原文（news1..newsN），因此只需把被抽中题目的源文档
入库即可评测，无需全量 8 万文档语料。

指标：抽取式答案包含率（预测与任一 gold 答案双向规范化包含），
分 1Doc / 2Docs / 3Docs 三层报告；拒答单独计数。

Run:
    python eval_crud_rag.py --smoke 6      # 每层抽 2 条冒烟（含真实嵌入）
    python eval_crud_rag.py                # 全量 200 题
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.kb import KnowledgeBase  # noqa: E402
from deepseek_chat import ask_deepseek  # noqa: E402

DATA_DIR = ROOT / "third_party" / "CRUD_RAG_data"
KB_ROOT = ROOT / "data" / "crud_kb"
ANSWER_TEMPERATURE = 0.1
TOP_K = 5
TIERS = ("questanswer_1doc", "questanswer_2docs", "questanswer_3docs")
TIER_LABEL = {"questanswer_1doc": "1Doc",
              "questanswer_2docs": "2Docs",
              "questanswer_3docs": "3Docs"}
PROMPT = (
    "你是检索问答助手。根据提供的资料回答问题，并在每个陈述后标注资料来源编号，如[1][2]。"
    "仅当资料与问题完全无关时才回答\"资料不足\"。\n\n"
    "资料：\n{evidence}\n\n问题：{query}\n回答："
)


def norm(text: str) -> str:
    return re.sub(r"[\W_]+", "", text or "").casefold()


def load_split() -> dict:
    return json.loads((DATA_DIR / "meta" / "split_merged.json")
                      .read_text(encoding="utf-8"))


def load_gt() -> dict[str, dict]:
    merged = {}
    for tier in TIERS:
        gt_path = DATA_DIR / "quest_eval" / {
            "questanswer_1doc": "QA1Doc.json",
            "questanswer_2docs": "QA2Docs.json",
            "questanswer_3docs": "QA3Docs.json"}[tier]
        merged[tier] = json.loads(gt_path.read_text(encoding="utf-8"))
    return merged


def build_questions(split: dict, gt: dict, limit_per_tier: int | None,
                    seed: int = 42) -> list[dict]:
    """展平为题目级列表；同一记录的多问共享同一组源文档。"""
    import random
    rng = random.Random(seed)
    questions = []
    for tier in TIERS:
        records = list(split[tier])
        rng.shuffle(records)
        if limit_per_tier:
            picked_ids = set()
            taken = 0
            for rec in records:
                gt_item = gt[tier].get(rec["ID"])
                if not gt_item:
                    continue
                picked_ids.add(rec["ID"])
                taken += len(gt_item["question"])
                if taken >= limit_per_tier:
                    break
            records = [r for r in records if r["ID"] in picked_ids]
        for rec in records:
            gt_item = gt[tier].get(rec["ID"])
            if not gt_item:
                continue
            news_docs = []
            j = 1
            while True:
                text = rec.get(f"news{j}")
                if not text:
                    break
                news_docs.append((f"crud_{rec['ID']}_news{j}.md", text))
                j += 1
            for qi, (question, gold) in enumerate(
                    zip(gt_item["question"], gt_item["answers"])):
                questions.append({
                    "tier": TIER_LABEL[tier],
                    "qid": f"{rec['ID']}#{qi}",
                    "question": question,
                    "gold": [norm(gold)],
                    "doc_names": [name for name, _text in news_docs],
                    "doc_texts": news_docs,
                })
    return questions


def ensure_docs(kb: KnowledgeBase, questions: list[dict]) -> dict[str, str]:
    """按名字幂等入库所有被引用的源文档；返回 name->document_id。"""
    by_name: dict[str, str] = {}
    for q in questions:
        for name, text in q["doc_texts"]:
            if name in by_name:
                continue
            existing = kb.catalog.get_by_name(name, include_deleted=True)
            doc_id = existing["document_id"] if existing else None
            if existing is None or existing["status"] != "ready":
                kb.catalog.ingest(name, text.encode("utf-8"), "crud-rag")
            by_name[name] = doc_id or kb.catalog.get_by_name(name)["document_id"]
            grant_target = "admin"
            if doc_id is None:
                kb.catalog.grant(grant_target, by_name[name])
    return by_name


def main() -> int:
    parser = argparse.ArgumentParser(description="CRUD-RAG 问答子集评测")
    parser.add_argument("--smoke", type=int, default=None,
                        help="每层抽取的【问题数】上限，用于小成本冒烟")
    parser.add_argument("--total", type=int, default=200)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    split = load_split()
    gt = load_gt()
    if args.smoke:
        questions = build_questions(split, gt, args.smoke)
    else:
        per_tier = args.total // 3
        questions = build_questions(split, gt, per_tier)
    print(f"题目数: {len(questions)} "
          f"(1Doc={sum(1 for q in questions if q['tier']=='1Doc')}, "
          f"2Docs={sum(1 for q in questions if q['tier']=='2Docs')}, "
          f"3Docs={sum(1 for q in questions if q['tier']=='3Docs')})")

    kb = KnowledgeBase(KB_ROOT)
    ensure_docs(kb, questions)

    need_rebuild = kb.status()["built"] is False
    if need_rebuild:
        est_chars = 0
        seen_names = set()
        for q in questions:
            for name, text in q["doc_texts"]:
                if name not in seen_names:
                    seen_names.add(name)
                    est_chars += len(text)
        print(f"需要重建索引：约 {len(seen_names)} 个文档 / {est_chars} 字符 "
              f"将调用 Embedding API")
        kb.rebuild(force=True)
    else:
        print("索引已就绪（指纹命中），零嵌入成本")

    rows = []
    for i, q in enumerate(questions, 1):
        hits, _filtered = kb._retrieve(q["question"], allowed_ids=set(
            d["id"] for d in kb.list_docs()))
        evidence = [(chunk.chunk_id, chunk.text) for chunk, _s in hits][:TOP_K]
        if not evidence:
            answer, refusal = "资料不足", True
        else:
            blocks = "\n".join(f"[{j}] {text}" for j, (_cid, text) in
                               enumerate(evidence, 1))
            answer = ask_deepseek(PROMPT.format(evidence=blocks,
                                                query=q["question"]),
                                  temperature=ANSWER_TEMPERATURE)
            refusal = answer.strip().startswith("资料不足") or \
                "资料不足" in answer[:20]
        pred = norm(answer)
        hit = any(g and (g in pred or pred in g) for g in q["gold"])
        rows.append({"tier": q["tier"], "qid": q["qid"],
                     "refusal": bool(refusal),
                     "hit": bool(hit and not refusal),
                     "question": q["question"],
                     "gold": q["gold"],
                     "pred_head": answer[:50]})
        if i % 20 == 0:
            print(f"  进度 {i}/{len(questions)}")

    summary = {}
    for tier in ("1Doc", "2Docs", "3Docs"):
        tr = [r for r in rows if r["tier"] == tier]
        hits = sum(1 for r in tr if r["hit"])
        refusals = sum(1 for r in tr if r["refusal"])
        summary[tier] = {
            "准确率(包含匹配)": f"{hits}/{len(tr)}",
            "拒答数": refusals,
        }
    overall_hits = sum(1 for r in rows if r["hit"])
    summary["overall"] = {"准确率": f"{overall_hits}/{len(rows)}"}

    report = {"config": {"top_k": TOP_K, "temperature": ANSWER_TEMPERATURE},
              "summary": summary, "rows": rows}
    out = Path(args.json_out) if args.json_out else \
        ROOT / "docs" / "EVAL_CRUD_RAG.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"报告已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
