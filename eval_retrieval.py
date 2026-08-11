"""W5 retrieval evaluation: all NoMIRACL zh queries x 5 recall paths.

Measures recall@5, recall@10 and MRR@10 for BM25, vector, RRF, weighted
sum and RRF+rerank, and writes evidence-backed conclusions to
docs/W5_REPORT.md.

Run:  python eval_retrieval.py
Reusable: query embeddings are cached next to the vector store, so
re-runs cost zero API calls.

Data notes (PROGRESS.md):
- some qrels doc_ids are not in the indexed corpus -> filter them out
  per query and count how many were dropped;
- queries whose relevant docs are ALL outside the corpus are counted
  but excluded from the metric averages.

Step 1 (this file grows step by step): data layer + filtering only.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np

from demo_hybrid_search import build_retrievers, load_passages, load_qrels
from index import bm25_search, fuse, rerank
from index.embeddings import embed_texts
from index.fusion import DEFAULT_ALPHA, RRF_K

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "nomiracl" / "chinese"
TOPICS = DATA_DIR / "topics" / "dev.relevant.tsv"

N = None  # indexed corpus cap; None = FULL corpus (all passages)
RAW_K = 10   # candidates per recall path (fusion input space)
FUSED_K = 10  # fused / reranked list length fed to the metrics
METHODS = ("bm25", "vector", "rrf", "weighted", "rrf+rerank")


def load_all_topics() -> list[tuple[str, str]]:
    """All (qid, query) pairs from topics/dev.relevant.tsv, in file order."""
    topics: list[tuple[str, str]] = []
    with TOPICS.open(encoding="utf-8") as file:
        for line in file:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 2 and parts[1].strip():
                topics.append((parts[0], parts[1]))
    return topics


def prepare_eval() -> tuple[list[tuple[str, str]], list[tuple[str, str, set[str]]], dict[str, int]]:
    """Load corpus + qrels + topics, filter, and return the eval-ready queries.

    Returns:
        docs:             indexed corpus as list[(doc_id, text)]
        eval_queries:     kept queries as list[(qid, query, relevant_set)]
        stats:            filtering counts, printed by main()
    """
    docs, _dropped = load_passages(1 << 30) if N is None else load_passages(N)
    indexed_ids = {doc_id for doc_id, _ in docs}
    qrels = load_qrels()
    topics = load_all_topics()

    stats = {
        "topics": len(topics),
        "qrels_queries": len(qrels),
        "topics_without_qrels": 0,
        "qrels_docs_filtered": 0,
        "queries_no_relevant_in_corpus": 0,
        "queries_kept": 0,
        "corpus_docs": len(docs),
        "full_corpus": N is None,
    }

    eval_queries: list[tuple[str, str, set[str]]] = []
    for qid, query in topics:
        if qid not in qrels:
            stats["topics_without_qrels"] += 1
            continue
        relevant = qrels[qid]
        in_corpus = relevant & indexed_ids
        stats["qrels_docs_filtered"] += len(relevant) - len(in_corpus)
        if not in_corpus:
            stats["queries_no_relevant_in_corpus"] += 1
            continue
        eval_queries.append((qid, query, in_corpus))
    stats["queries_kept"] = len(eval_queries)
    return docs, eval_queries, stats


def _query_cache_stem(query_texts: list[str], n_docs: int) -> str:
    digest = hashlib.sha1(
        json.dumps(query_texts, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:8]
    return f"query_embeddings_N{n_docs}_{digest}"


def load_or_embed_queries(query_texts: list[str], n_docs: int) -> np.ndarray:
    """Batch-embed a query set once and cache; later runs are zero-API.

    The cache file name embeds a hash of the query texts, so different
    query sets never clobber each other (one set = one file).
    """
    stem = _query_cache_stem(query_texts, n_docs)
    cache = DATA_DIR / f"{stem}.npy"
    meta = DATA_DIR / f"{stem}.json"
    if cache.is_file() and meta.is_file():
        return np.load(cache)
    print(f"embedding {len(query_texts)} queries via API ...")
    matrix = np.asarray(embed_texts(query_texts), dtype=np.float32)
    np.save(cache, matrix)
    meta.write_text(json.dumps({"queries": query_texts}, ensure_ascii=False), encoding="utf-8")
    return matrix


def retrieve_all(
    bm25_index,
    vector_store,
    eval_queries: list[tuple[str, str, set[str]]],
) -> dict[str, dict[str, list[tuple[str, float]]]]:
    """Run all 5 recall paths for every query.

    Returns {qid: {method: top-FUSED_K list[(doc_id, score)]}}.
    Vector path: one batch matmul for all queries (no per-query API).
    """
    query_texts = [query for _, query, _ in eval_queries]
    query_matrix = load_or_embed_queries(query_texts, bm25_index.N)
    norms = np.linalg.norm(query_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    query_norm = query_matrix / norms
    cos = query_norm @ vector_store.norm_matrix.T
    top = np.argsort(-cos, axis=1)[:, :RAW_K]

    results: dict[str, dict[str, list[tuple[str, float]]]] = {}
    for i, (qid, query, _relevant) in enumerate(eval_queries):
        bm25_hits = bm25_search(bm25_index, query, RAW_K)
        vector_hits = [(vector_store.doc_ids[j], float(cos[i, j])) for j in top[i]]
        rrf_hits = fuse(bm25_hits, vector_hits, k=FUSED_K, method="rrf")
        wsum_hits = fuse(bm25_hits, vector_hits, k=FUSED_K, method="weighted")
        reranked = rerank(rrf_hits, bm25_hits, vector_hits)
        results[qid] = {
            "bm25": bm25_hits[:FUSED_K],
            "vector": vector_hits[:FUSED_K],
            "rrf": rrf_hits,
            "weighted": wsum_hits,
            "rrf+rerank": reranked,
        }
    return results


def recall_at_k(hits: list[tuple[str, float]], relevant: set[str], k: int) -> float:
    """Fraction of relevant docs found in hits[:k]. 0 if relevant empty."""
    if not relevant:
        return 0.0
    found = sum(1 for doc_id, _ in hits[:k] if doc_id in relevant)
    return found / len(relevant)


def mrr_at_10(hits: list[tuple[str, float]], relevant: set[str]) -> float:
    """Inverse rank of the first relevant doc (1/rank), 0 if none in top-10."""
    for rank, (doc_id, _) in enumerate(hits[:10], start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def evaluate_all(
    results: dict[str, dict[str, list[tuple[str, float]]]],
    eval_queries: list[tuple[str, str, set[str]]],
) -> dict[str, dict[str, float]]:
    """Per-method averages of recall@5, recall@10, MRR@10 over kept queries."""
    agg = {method: {"recall@5": [], "recall@10": [], "mrr@10": []} for method in METHODS}
    for qid, _query, relevant in eval_queries:
        for method in METHODS:
            hits = results[qid][method]
            agg[method]["recall@5"].append(recall_at_k(hits, relevant, 5))
            agg[method]["recall@10"].append(recall_at_k(hits, relevant, 10))
            agg[method]["mrr@10"].append(mrr_at_10(hits, relevant))
    count = len(eval_queries)
    return {
        method: {metric: sum(values) / count for metric, values in metrics.items()}
        for method, metrics in agg.items()
    }


def print_table(table: dict[str, dict[str, float]]) -> None:
    header = f"{'method':<10} | {'recall@5':>9} | {'recall@10':>10} | {'MRR@10':>9}"
    print("\n" + header)
    print("-" * len(header))
    for method in METHODS:
        m = table[method]
        print(f"{method:<10} | {m['recall@5']:>9.4f} | {m['recall@10']:>10.4f} | {m['mrr@10']:>9.4f}")


def analyze_groups(
    results: dict[str, dict[str, list[tuple[str, float]]]],
    eval_queries: list[tuple[str, str, set[str]]],
) -> dict:
    """Group queries by which single path wins recall@10 (question c).

    Returns per-group mean recall@10 for every method, plus RRF / weighted
    win-tie-loss counts against the better single path, and how often
    rerank improves recall@5 within the fused set.
    """
    groups: dict[str, list[dict]] = {"bm25_better": [], "vector_better": [], "tie": []}
    rrf = {"win": 0, "tie": 0, "loss": 0, "gaps": []}
    wsum = {"win": 0, "tie": 0, "loss": 0, "gaps": []}
    rerank5 = {"better": 0, "same": 0, "worse": 0}

    for qid, query, relevant in eval_queries:
        per = results[qid]
        b10 = recall_at_k(per["bm25"], relevant, 10)
        v10 = recall_at_k(per["vector"], relevant, 10)
        r10 = recall_at_k(per["rrf"], relevant, 10)
        w10 = recall_at_k(per["weighted"], relevant, 10)
        r5 = recall_at_k(per["rrf"], relevant, 5)
        rr5 = recall_at_k(per["rrf+rerank"], relevant, 5)

        if b10 > v10:
            group = "bm25_better"
        elif v10 > b10:
            group = "vector_better"
        else:
            group = "tie"
        groups[group].append(
            {"qid": qid, "query": query, "b": b10, "v": v10, "r": r10, "w": w10,
             "has_digit": bool(re.search(r"\d", query))}
        )

        best = max(b10, v10)
        for name, counter in (("rrf", rrf), ("weighted", wsum)):
            value = r10 if name == "rrf" else w10
            if value > best:
                counter["win"] += 1
            elif value == best:
                counter["tie"] += 1
            else:
                counter["loss"] += 1
                counter["gaps"].append(best - value)

        if rr5 > r5:
            rerank5["better"] += 1
        elif rr5 == r5:
            rerank5["same"] += 1
        else:
            rerank5["worse"] += 1

    group_means = {}
    for name, items in groups.items():
        n = len(items)
        if n == 0:
            group_means[name] = {"n": 0}
            continue
        group_means[name] = {
            "n": n,
            "bm25": sum(i["b"] for i in items) / n,
            "vector": sum(i["v"] for i in items) / n,
            "rrf": sum(i["r"] for i in items) / n,
            "weighted": sum(i["w"] for i in items) / n,
            "digit_frac": sum(1 for i in items if i["has_digit"]) / n,
        }

    mean = lambda c: (sum(c["gaps"]) / len(c["gaps"])) if c["gaps"] else 0.0
    return {
        "group_means": group_means,
        "rrf": {k: (mean(rrf) if k == "mean_gap" else rrf[k]) for k in ("win", "tie", "loss", "mean_gap")},
        "weighted": {k: (mean(wsum) if k == "mean_gap" else wsum[k]) for k in ("win", "tie", "loss", "mean_gap")},
        "rerank5": rerank5,
    }


def write_report(
    table: dict[str, dict[str, float]],
    analysis: dict,
    stats: dict[str, int],
) -> Path:
    """Write docs/W5_REPORT.md with three number-backed conclusions."""
    gm = analysis["group_means"]
    bm25_group = gm.get("bm25_better", {})
    vec_group = gm.get("vector_better", {})
    rrf = analysis["rrf"]
    wsum = analysis["weighted"]
    r5 = analysis["rerank5"]
    n_total = stats["queries_kept"]

    def g(name, key):
        return gm.get(name, {}).get(key, 0)

    lines = [
        f"# W5 评测报告：{n_total} 条 NoMIRACL zh x 五路召回",
        "",
        "## 运行说明",
        f"- 命令：`python eval_retrieval.py`（重跑零 API，query 嵌入已缓存）",
        f"- 语料：{stats['corpus_docs']} 篇 passage"
        f"{'（全量）' if stats['full_corpus'] else '（前 N 篇切片）'}；"
        f"qrels 中 {stats['qrels_docs_filtered']} 条 doc_id 不在库，已剔除；"
        f"{n_total} 个 query 全部可评"
        f"（{stats['queries_no_relevant_in_corpus']} 个因无相关 doc 被排除）",
        f"- 参数：RAW_K={RAW_K}（每路候选）、RRF_K={RRF_K}（RRF 常数）、"
        f"alpha={DEFAULT_ALPHA}（加权和），均为 W4 默认，本轮未调参",
        "",
        f"## 总表（{n_total} query 平均）",
        "",
        "| 方法 | recall@5 | recall@10 | MRR@10 |",
        "|---|---:|---:|---:|",
    ]
    for method in METHODS:
        m = table[method]
        lines.append(f"| {method} | {m['recall@5']:.4f} | {m['recall@10']:.4f} | {m['mrr@10']:.4f} |")
    lines += [
        "",
        "## 三行结论",
        "",
        "### a) BM25 vs 向量：向量全面胜出（W3 遗留问题）",
        f"向量 recall@10 {table['vector']['recall@10']:.4f} vs BM25 {table['bm25']['recall@10']:.4f}"
        f"（+{table['vector']['recall@10'] - table['bm25']['recall@10']:.4f}），"
        f"recall@5 +{table['vector']['recall@5'] - table['bm25']['recall@5']:.4f}，"
        f"MRR@10 +{table['vector']['mrr@10'] - table['bm25']['mrr@10']:.4f}。"
        f"BM25 仅在 {g('bm25_better', 'n')} 个 query（{g('bm25_better', 'n') / n_total * 100:.1f}%）上 recall@10 更优。",
        "",
        "### b) RRF 平均低于较优单路（W4 验收在全部 query 上 FAIL）",
        f"RRF recall@10 {table['rrf']['recall@10']:.4f} vs 向量 {table['vector']['recall@10']:.4f}"
        f"（-{table['vector']['recall@10'] - table['rrf']['recall@10']:.4f}）。"
        f"逐 query：输 {rrf['loss']} 个、平 {rrf['tie']} 个、赢 {rrf['win']} 个，"
        f"输的 query 平均亏 {rrf['mean_gap']:.4f}。加权和 {table['weighted']['recall@10']:.4f}"
        f"（-{table['vector']['recall@10'] - table['weighted']['recall@10']:.4f}）也未反超。"
        f"rerank 把 recall@5 从 {table['rrf']['recall@5']:.4f} 提到 {table['rrf+rerank']['recall@5']:.4f}"
        f"（+{table['rrf+rerank']['recall@5'] - table['rrf']['recall@5']:.4f}，{r5['better']} 个 query 受益，"
        f"另有 {r5['worse']} 个下降、{r5['same']} 个持平），"
        f"recall@10 不变——集合不变只修排序。输入空间无碍（向量 recall@10 已 "
        f"{table['vector']['recall@10']:.4f}），"
        f"损失来自 RRF 排名平权磨平分数断层，属方法问题，调参方向见下节。",
        "",
        "### c) 融合赢输分型：关键词型 vs 语义型",
        f"按\"单路谁 recall@10 更优\"分组：BM25 优势（关键词型）{g('bm25_better', 'n')} 个，"
        f"向量优势（语义型）{g('vector_better', 'n')} 个，平局 {g('tie', 'n')} 个。",
        "",
        "| 组 | n | bm25 | vector | rrf | weighted | 含数字 query 占比 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| 关键词型 | {g('bm25_better', 'n')} | {g('bm25_better', 'bm25'):.4f} | "
        f"{g('bm25_better', 'vector'):.4f} | {g('bm25_better', 'rrf'):.4f} | "
        f"{g('bm25_better', 'weighted'):.4f} | {g('bm25_better', 'digit_frac'):.1%} |",
        f"| 语义型 | {g('vector_better', 'n')} | {g('vector_better', 'bm25'):.4f} | "
        f"{g('vector_better', 'vector'):.4f} | {g('vector_better', 'rrf'):.4f} | "
        f"{g('vector_better', 'weighted'):.4f} | {g('vector_better', 'digit_frac'):.1%} |",
        f"| 平局 | {g('tie', 'n')} | {g('tie', 'bm25'):.4f} | {g('tie', 'vector'):.4f} | "
        f"{g('tie', 'rrf'):.4f} | {g('tie', 'weighted'):.4f} | {g('tie', 'digit_frac'):.1%} |",
        "",
        f"RRF 输给较优单路的 {rrf['loss']} 个 query 集中在语义型（RRF recall@10 "
        f"{g('vector_better', 'rrf'):.4f} < 向量 {g('vector_better', 'vector'):.4f}）；"
        f"关键词型组融合表现：RRF {g('bm25_better', 'rrf'):.4f} vs BM25 {g('bm25_better', 'bm25'):.4f}"
        f" vs 向量 {g('bm25_better', 'vector'):.4f}。",
        "",
        "## 附：未调参声明与调参方向（问题 F 归因）",
        "- 结论 b 的失败发生在输入空间充足的条件下"
        f"（向量 recall@10={table['vector']['recall@10']:.4f}），归因于 RRF 机制（排名平权），"
        "非 RAW_K 太小。",
        "- 未做参数扫描：RRF_K 调小（如 1~20）可削弱双榜加成、alpha 调大可向向量倾斜，均为候选方向；"
        "若调参须在本次全量 393 条上选一次并重新出表，不得用 demo 3 条或本表反推。",
    ]
    report_path = ROOT / "docs" / "W5_REPORT.md"
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> int:
    docs, eval_queries, stats = prepare_eval()
    print(f"corpus                 : {len(docs)} passages "
          f"(N={'full' if stats['full_corpus'] else N})")
    print(f"topics                 : {stats['topics']} total")
    print(f"qrels queries          : {stats['qrels_queries']}")
    print(f"  topics without qrels : {stats['topics_without_qrels']}")
    print(f"  qrels docs filtered  : {stats['qrels_docs_filtered']} (not in indexed corpus)")
    print(f"  queries, no relevant : {stats['queries_no_relevant_in_corpus']} (excluded from averages)")
    print(f"  queries kept         : {stats['queries_kept']}")
    print("\nfirst 5 kept queries:")
    for qid, query, relevant in eval_queries[:5]:
        print(f"  {qid}  {query}  ({len(relevant)} relevant)")

    bm25_index, vector_store = build_retrievers(docs)
    t0 = time.perf_counter()
    results = retrieve_all(bm25_index, vector_store, eval_queries)
    elapsed = time.perf_counter() - t0
    print(f"\nretrieval done: {len(results)} queries x {len(METHODS)} methods "
          f"({elapsed:.1f}s, top-{FUSED_K} each)")

    qid, query, relevant = eval_queries[0]
    print(f"\nsanity check (query 1: {query}, {len(relevant)} relevant):")
    for method in METHODS:
        top3 = [doc for doc, _ in results[qid][method][:3]]
        print(f"  {method:<10}: {top3}")

    table = evaluate_all(results, eval_queries)
    print_table(table)

    analysis = analyze_groups(results, eval_queries)
    report_path = write_report(table, analysis, stats)
    gm = analysis["group_means"]
    print(f"\n分组: 关键词型(bm25优) {gm.get('bm25_better', {}).get('n', 0)} 个 | "
          f"语义型(vector优) {gm.get('vector_better', {}).get('n', 0)} 个 | "
          f"平局 {gm.get('tie', {}).get('n', 0)} 个")
    rrf = analysis["rrf"]
    print(f"RRF vs 较优单路(recall@10): 输 {rrf['loss']} / 平 {rrf['tie']} / 赢 {rrf['win']} "
          f"(输的平均亏 {rrf['mean_gap']:.4f})")
    print(f"rerank 在 recall@5 上: 改善 {analysis['rerank5']['better']} / 持平 "
          f"{analysis['rerank5']['same']} / 变差 {analysis['rerank5']['worse']}")
    print(f"报告已写入: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
