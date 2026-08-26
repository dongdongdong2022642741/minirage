"""NoMIRACL zh 拒答评测（C 线）：relevant 应作答，non_relevant 应拒答。

复用全量语料索引缓存（bm25_N31826 + vecstore_N31826，零嵌入成本），
检索走生产同款 BM25+向量 RRF+rerank 路径，答案判定复用 app.kb 的
拒答标记与引用解析。指标：

    relevant    : 能答率（非拒答）、引用合法率、误拒率
    non_relevant: 拒答率（越高越好）、幻觉率 = 1 - 拒答率

Run:
    python eval_nomiracl_rag.py                     # dev 30+30
    python eval_nomiracl_rag.py --n-rel 50 --n-nonrel 50 --split test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepseek_chat import ask_deepseek  # noqa: E402
from app.kb import build_prompt, is_refusal, parse_citations  # noqa: E402
from demo_hybrid_search import build_retrievers, load_passages  # noqa: E402
from index import bm25_search, fuse, rerank  # noqa: E402

DATA_DIR = ROOT / "data" / "nomiracl" / "chinese"
ANSWER_TEMPERATURE = 0.1
RETRY_REFUSALS = 1


def load_topics(split: str, kind: str) -> list[tuple[str, str]]:
    path = DATA_DIR / "topics" / f"{split}.{kind}.tsv"
    topics = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.rstrip("\n").split("\t")
        if len(parts) == 2 and parts[1].strip():
            topics.append((parts[0], parts[1]))
    return topics


def stride_sample(items: list, n: int, seed: int = 42) -> list:
    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)
    return shuffled[:n]


def retrieve(index, vector_store, query: str, k: int,
             gate: float = 0.0) -> list[tuple[str, str]]:
    """生产同款：双路召回 -> RRF -> rerank 截断；gate 为向量 top1 门控。"""
    bm25_hits = bm25_search(index, query, k=10)
    vector_hits = vector_store.search(query, k=10)
    if gate > 0 and (not vector_hits or vector_hits[0][1] < gate):
        return []  # 幻觉门控：与 app.kb._retrieve 同一语义
    hits = rerank(fuse(bm25_hits, vector_hits, k=k, method="rrf"),
                  bm25_hits, vector_hits)[:k]
    return hits


def cache_path(split: str, k: int, gate: float = 0.0) -> Path:
    return DATA_DIR / f"rag_cache_{split}_k{k}_g{gate}.json"


def fingerprint(path_name: str) -> str:
    return hashlib.sha1(path_name.encode("utf-8")).hexdigest()[:8]


def load_cache(path: Path, cfg_name: str) -> dict:
    if not path.is_file():
        return {}
    blob = json.loads(path.read_text(encoding="utf-8"))
    if blob.get("_cfg") != cfg_name:
        raise RuntimeError(
            f"{path.name} 配置指纹不符（{blob.get('_cfg')} != {cfg_name}），"
            "删除该文件以强制重跑")
    return blob["answers"]


def save_cache(path: Path, answers: dict, cfg_name: str) -> None:
    path.write_text(json.dumps({"_cfg": cfg_name, "answers": answers},
                               ensure_ascii=False, indent=1), encoding="utf-8")


def answer_query(query: str, evidence: list[tuple[str, str]]) -> tuple[str, bool]:
    """返回 (answer, llm_called)。零证据确定性拒答，与线上 ask 行为一致。"""
    if not evidence:
        return "资料不足", False
    answer = ask_deepseek(build_prompt(query, evidence),
                          temperature=ANSWER_TEMPERATURE)
    for _ in range(RETRY_REFUSALS):
        if not is_refusal(answer):
            break
        answer = ask_deepseek(build_prompt(query, evidence, retry=True),
                              temperature=ANSWER_TEMPERATURE)
    return answer, True


def run_split(index, vector_store, doc_text: dict, split: str, kind: str,
              n: int, k: int, cache: dict, cfg_name: str,
              cpath: Path, gate: float = 0.0) -> list[dict]:
    topics = stride_sample(load_topics(split, kind), n)
    rows = []
    errors = 0
    for qid, query in topics:
        entry = cache.get(qid)
        if entry is None:
            try:
                hits = retrieve(index, vector_store, query, k, gate=gate)
                evidence = [(d, doc_text[d]) for d, _ in hits]
                t0 = time.time()
                answer, llm_called = answer_query(query, evidence)
                entry = {
                    "answer": answer,
                    "llm_called": llm_called,
                    "n_evidence": len(evidence),
                    "latency_s": round(time.time() - t0, 2),
                    "citations": parse_citations(answer),
                    "refusal": is_refusal(answer),
                }
                cache[qid] = entry
                save_cache(cpath, cache, cfg_name)
                if llm_called:
                    time.sleep(0.2)
            except RuntimeError as error:
                print(f"  ERROR {qid}: {error}")
                errors += 1
                continue
        rows.append({
            "qid": qid, "query": query, "kind": kind,
            "refusal": entry["refusal"],
            "answered": not entry["refusal"],
            "n_evidence": entry["n_evidence"],
            "citations": entry["citations"],
            "answer_head": entry["answer"][:60],
        })
    if errors:
        print(f"[warn] {errors} 条查询调用失败，未计入指标")
    return rows


def summarize(rows: list[dict], kind: str) -> dict:
    total = len(rows)
    answered = sum(1 for r in rows if r["answered"])
    refusals = total - answered
    cited = sum(1 for r in rows if r["answered"] and r["citations"])
    out = {"total": total}
    if kind == "relevant":
        out.update({
            "能答率": f"{answered}/{total}",
            "误拒率": f"{refusals}/{total}",
            "引用合法率(有引用即可)": f"{cited}/{answered}" if answered else "-",
        })
    else:
        out.update({
            "拒答率": f"{refusals}/{total}",
            "幻觉率(未拒答)": f"{answered}/{total}",
        })
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="NoMIRACL zh 拒答评测")
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--n-rel", type=int, default=30)
    parser.add_argument("--n-nonrel", type=int, default=30)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--gate", type=float, default=0.0,
                        help="向量 top1 相似度门控，0=关闭")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    cfg_name = f"{args.split}_k{args.k}_g{args.gate}"
    cpath = cache_path(args.split, args.k, args.gate)
    cache = load_cache(cpath, fingerprint(cfg_name))

    print("加载全量语料与索引缓存（首次约 1 分钟，无 API 成本）...")
    docs, _dropped = load_passages(10**9)  # 全量语料（load_passages 需要整数上限）
    doc_text = {doc_id: text for doc_id, text in docs}
    index, vector_store = build_retrievers(docs)

    print(f"\n== relevant ({args.n_rel}) ==")
    rel_rows = run_split(index, vector_store, doc_text, args.split,
                         "relevant", args.n_rel, args.k, cache, cfg_name,
                         cpath, gate=args.gate)
    print(f"== non_relevant ({args.n_nonrel}) ==")
    nonrel_rows = run_split(index, vector_store, doc_text, args.split,
                            "non_relevant", args.n_nonrel, args.k,
                            cache, cfg_name, cpath, gate=args.gate)

    report = {
        "config": {"split": args.split, "k": args.k, "gate": args.gate,
                   "path": "bm25+vector RRF rerank",
                   "temperature": ANSWER_TEMPERATURE},
        "relevant_summary": summarize(rel_rows, "relevant"),
        "non_relevant_summary": summarize(nonrel_rows, "non_relevant"),
        "rows": {"relevant": rel_rows, "non_relevant": nonrel_rows},
    }
    out_path = Path(args.json_out) if args.json_out else \
        ROOT / "docs" / f"EVAL_NOMIRACL_RAG_{args.split}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    print("\n===== 结果 =====")
    print(json.dumps(report["relevant_summary"], ensure_ascii=False))
    print(json.dumps(report["non_relevant_summary"], ensure_ascii=False))
    print(f"报告已写入 {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
