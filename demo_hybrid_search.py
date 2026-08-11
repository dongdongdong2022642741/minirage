"""Hybrid retrieval demo: BM25 vs vector vs RRF fusion vs weighted sum.

Usage:
    python demo_hybrid_search.py [N] [query...]

N = number of corpus passages to index (default 1000). Without CLI queries,
the demo uses the first few real queries from topics/dev.relevant.tsv.
CLI queries have no qid, so their recall@5 line prints n/a.

Per query the demo prints:
    - a 4-column table: BM25 / vector / RRF / RRF+rerank, top-5 each
    - whether rerank reordered the fused top-5, and how much the two
      fusion algorithms (RRF vs weighted sum) agree
    - recall@5 for every method (weighted sum included) against
      qrels/dev.relevant.tsv, and whether fused recall is >= the better
      single path (W4 acceptance)

The vector store is cached on disk (data/nomiracl/chinese/vecstore_N{N}/):
the first run pays the embedding API cost once and saves the matrix;
every later run loads it via VectorStore.load with zero API spend.
"""

from __future__ import annotations

import gzip
import json
import pickle
import sys
import time
from pathlib import Path

from index import IndexBuilder, VectorStore, Searcher, fuse, rerank

DATA_DIR = Path(__file__).resolve().parent / "data" / "nomiracl" / "chinese"
CORPUS = DATA_DIR / "corpus.jsonl.gz"
TOPICS = DATA_DIR / "topics" / "dev.relevant.tsv"
QRELS = DATA_DIR / "qrels" / "dev.relevant.tsv"
DEFAULT_N = 4000  # >= 3719: all 3 default queries have qrels docs in corpus
DEFAULT_QUERIES = 3
RAW_K = 10      # candidates per recall path
FUSED_K = 5     # top-k after fusion / rerank


def load_passages(n: int) -> tuple[list[tuple[str, str]], int]:
    docs: list[tuple[str, str]] = []
    seen: set[str] = set()
    dropped = 0
    with gzip.open(CORPUS, "rt", encoding="utf-8") as file:
        for line in file:
            if len(docs) >= n:
                break
            record = json.loads(line)
            doc_id = record["docid"]
            if doc_id in seen:
                dropped += 1
                continue
            seen.add(doc_id)
            text = f"{record.get('title', '')}\n{record.get('text', '')}".strip()
            docs.append((doc_id, text))
    if not docs:
        raise RuntimeError(f"corpus is empty: {CORPUS}")
    return docs, dropped


def load_topics(count: int = DEFAULT_QUERIES) -> list[tuple[str, str]]:
    topics: list[tuple[str, str]] = []
    with TOPICS.open(encoding="utf-8") as file:
        for line in file:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 2 and parts[1].strip():
                topics.append((parts[0], parts[1]))
            if len(topics) >= count:
                break
    return topics


def load_qrels() -> dict[str, set[str]]:
    relevant: dict[str, set[str]] = {}
    with QRELS.open(encoding="utf-8") as file:
        for line in file:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4 or parts[3].strip() != "1":
                continue
            qid, doc_id = parts[0], parts[2]
            relevant.setdefault(qid, set()).add(doc_id)
    return relevant


def build_retrievers(docs: list[tuple[str, str]]):
    bm25_cache = DATA_DIR / f"bm25_N{len(docs)}.pkl"
    if bm25_cache.is_file():
        t0 = time.perf_counter()
        bm25_index = pickle.loads(bm25_cache.read_bytes())
        elapsed = time.perf_counter() - t0
        print(f"bm25 index   : loaded from cache {bm25_cache.name} ({elapsed:.1f}s)")
    else:
        t0 = time.perf_counter()
        bm25_index = IndexBuilder().build(docs)
        elapsed = time.perf_counter() - t0
        bm25_cache.write_bytes(pickle.dumps(bm25_index))
        print(f"bm25 index   : built ({elapsed:.1f}s), pickled to {bm25_cache.name}")
    print(f"bm25 index   : {bm25_index.N} docs, {len(bm25_index.postings)} terms, "
          f"avgdl {bm25_index.avgdl:.1f}")

    cache_dir = DATA_DIR / f"vecstore_N{len(docs)}"
    if (cache_dir / "matrix.npy").is_file() and (cache_dir / "doc_ids.json").is_file():
        t0 = time.perf_counter()
        vector_store = VectorStore.load(cache_dir)
        elapsed = time.perf_counter() - t0
        print(f"vector store : loaded from cache {cache_dir.name} ({elapsed:.1f}s, no API cost)")
    else:
        t0 = time.perf_counter()
        vector_store = VectorStore.build(docs)
        elapsed = time.perf_counter() - t0
        vector_store.save(cache_dir)
        print(f"vector store : built via embedding API ({elapsed:.1f}s), saved to {cache_dir.name}")
    return bm25_index, vector_store


def recall_at_k(hits: list[tuple[str, float]], relevant: set[str]) -> float | None:
    if not relevant:
        return None
    found = sum(1 for doc_id, _ in hits[:FUSED_K] if doc_id in relevant)
    return found / len(relevant)


def fmt_hit(pair: tuple[str, float], width: int) -> str:
    doc_id, score = pair
    return f"{doc_id}({score:.4f})".ljust(width)


def main() -> int:
    args = sys.argv[1:]
    n = DEFAULT_N
    cli_queries: list[str] = []
    for arg in args:
        if arg.isdigit():
            n = int(arg)
        else:
            cli_queries.append(arg)

    if cli_queries:
        topics = [(f"cli{i}", q) for i, q in enumerate(cli_queries)]
    else:
        topics = load_topics()

    print(f"indexing first {n} distinct corpus passages from {CORPUS.name} ...")
    docs, dropped = load_passages(n)
    if dropped:
        print(f"{dropped} duplicate docids skipped in corpus")
    doc_text = {doc_id: text for doc_id, text in docs}
    indexed_ids = set(doc_text)
    bm25_index, vector_store = build_retrievers(docs)
    searcher = Searcher(bm25_index, vector_store)
    qrels = load_qrels()
    print(f"qrels: {len(qrels)} queries with relevant judgments")

    summary: list[tuple[str, str, float, float, float, float, float, float]] = []
    for qid, query in topics:
        relevant = qrels.get(qid, set()) & indexed_ids
        if qid not in qrels:
            print("(no qrels judgment for this query -> recall n/a)")
        elif not relevant:
            print("(relevant docs outside the indexed corpus -> recall n/a)")
        else:
            print(f"({len(relevant)} relevant docs in indexed corpus)")

        bm25_hits = searcher.bm25_search(query, k=RAW_K)
        vector_hits = searcher.vector_search(query, k=RAW_K)
        rrf_hits = fuse(bm25_hits, vector_hits, k=FUSED_K, method="rrf")
        wsum_hits = fuse(bm25_hits, vector_hits, k=FUSED_K, method="weighted")
        reranked = rerank(rrf_hits, bm25_hits, vector_hits)[:FUSED_K]

        print("\n" + "=" * 70)
        print(f"QUERY: {query}")
        print("=" * 70)
        width = 24
        header = (f"rank | {'BM25'.ljust(width)} | {'向量'.ljust(width)} "
                  f"| {'RRF'.ljust(width)} | {'RRF+rerank'.ljust(width)}")
        print(header)
        print("-" * len(header))
        for rank in range(FUSED_K):
            cells = [fmt_hit(pair, width)
                     for pair in (bm25_hits[rank], vector_hits[rank], rrf_hits[rank], reranked[rank])]
            print(f"{rank + 1:>4} | {' | '.join(cells)}")

        rrf_ids = [d for d, _ in rrf_hits]
        reranked_ids = [d for d, _ in reranked]
        changed = "（顺序有变化）" if rrf_ids != reranked_ids else "（顺序不变）"
        print(f"\nRRF -> rerank 排序: {changed}")
        print(f"overlap RRF vs 加权和: {len(set(rrf_ids) & set(d for d, _ in wsum_hits))}/{FUSED_K}")

        methods = [("bm25", bm25_hits), ("vector", vector_hits),
                   ("rrf", rrf_hits), ("weighted", wsum_hits), ("rrf+rerank", reranked)]
        recalls = [(name, recall_at_k(hits, relevant)) for name, hits in methods]
        row = "  ".join(f"{name}={r:.3f}" if r is not None else f"{name}=n/a" for name, r in recalls)
        print(f"\nrecall@{FUSED_K}: {row}")

        best_single = -1.0
        for name, r in recalls:
            if r is not None and name in ("bm25", "vector") and r > best_single:
                best_single = r
        if best_single >= 0:
            fused_ok = all(r is not None and r >= best_single
                           for name, r in recalls if name in ("rrf", "rrf+rerank"))
            verdict = "PASS" if fused_ok else "FAIL"
            print(f"fused recall >= better single path: {verdict} (best single={best_single:.3f})")
        summary.append((qid, query, best_single, *[r if r is not None else -1.0 for _, r in recalls]))

    judged = [s for s in summary if s[2] >= 0]
    if judged:
        print("\n" + "=" * 70)
        print("SUMMARY (queries with qrels):")
        print(f"{'query':<10} {'best_single':>11} {'bm25':>8} {'vector':>8} {'rrf':>8} {'wsum':>8} {'rerank':>8}")
        for qid, query, best_single, b, v, r, w, rr in judged:
            print(f"{qid:<10} {best_single:>11.3f} {b:>8.3f} {v:>8.3f} {r:>8.3f} {w:>8.3f} {rr:>8.3f}")
        failed = sum(1 for s in judged if s[5] < s[2] or s[7] < s[2])
        print(f"RRF / RRF+rerank recall never below better single path: "
              f"{'PASS' if failed == 0 else f'FAIL ({failed} query)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
