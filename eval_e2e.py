"""W5 phase 2: RAGAS-style end-to-end answer evaluation.

For a sampled subset of queries, DeepSeek answers using the top-5 docs
fed by two retrieval paths (vector vs rrf+rerank). Answers are scored:

    faithfulness : fraction of answer statements supported by the top-5
                   evidence (bge-m3 cosine >= FAITH_THRESHOLD)
    relevancy    : cosine(query embedding, answer embedding)
    citations    : validity of [n] references (must be 1..TOP_FOR_ANSWER
                   and at least one citation; 0 citations = invalid)

Answers are cached to data/nomiracl/chinese/e2e_cache_N{n_docs}_top{top_k}.json
(the name encodes corpus size and topK; the file also stores the
prompt/temperature/retry config fingerprint, verified on load so a
config change fails loudly instead of reusing stale answers);
re-runs load the cache and cost zero DeepSeek API calls.

Answer-time config: low temperature (ANSWER_TEMPERATURE) and automatic
retry with a nudge prompt (RETRY_REFUSALS) when the model answers a
pure refusal like "资料不足".

Run:  python eval_e2e.py [sample_size] [top_k]
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

from deepseek_chat import ask_deepseek
from demo_hybrid_search import build_retrievers
from eval_retrieval import load_or_embed_queries, prepare_eval, retrieve_all
from index.embeddings import embed_texts

SAMPLE_SIZE = 15
TOP_FOR_ANSWER = 10
FAITH_THRESHOLD = 0.65   # statement-evidence cosine deemed "supported"
RELEVANCY_THRESHOLD = 0.45  # answer-query cosine deemed "relevant"
ANSWER_METHODS = ("vector", "rrf+rerank")
ANSWER_TEMPERATURE = 0.1  # low variance; API default 1.0 flips refusals
RETRY_REFUSALS = 1        # re-ask once when the answer is a pure refusal
META_MARKERS = ("资料不足", "无法", "不能确定", "不清楚", "没有提供")

PROMPT = (
    "你是检索问答助手。根据提供的资料回答问题，并在每个陈述后标注资料来源编号，如[1][2]。"
    "可以组合多篇资料的信息进行推断，但必须基于资料内容并标注依据。"
    "仅当所有资料都与问题完全无关、无法提供任何相关信息时，才回答\"资料不足\"。\n\n"
    "资料：\n{evidence}\n\n问题：{query}\n回答："
)

RETRY_PROMPT = (
    "你是检索问答助手。请再次仔细阅读以下资料：问题通常能在资料中找到全部或部分相关信息，"
    "请尽可能从资料中提取相关内容作答，并标注资料来源编号，如[1][2]；"
    "若只能找到部分信息，请作答那部分并注明。\n\n"
    "资料：\n{evidence}\n\n问题：{query}\n回答："
)


def sample_queries(eval_queries: list, size: int) -> list:
    """Deterministic stride sampling across the topic file order."""
    stride = max(1, len(eval_queries) // size)
    return eval_queries[::stride][:size]


def build_prompt(query: str, evidence: list[tuple[str, str]], retry: bool = False) -> str:
    blocks = [f"[{i}] {text}" for i, (_doc_id, text) in enumerate(evidence, 1)]
    template = RETRY_PROMPT if retry else PROMPT
    return template.format(evidence="\n".join(blocks), query=query)


def split_statements(answer: str) -> list[str]:
    parts = re.split(r"[。！？!?；;\n]+", answer)
    return [p.strip() for p in parts if len(p.strip()) >= 4]


def is_meta(statement: str) -> bool:
    return any(marker in statement for marker in META_MARKERS)


def is_refusal(answer: str) -> bool:
    """Pure refusal: no statements, or every statement is meta ("资料不足。")."""
    statements = [s for s in split_statements(answer) if s]
    return not statements or all(is_meta(s) for s in statements)


def evidence_embeddings(vector_store, doc_ids: list[str]) -> np.ndarray:
    """Normalized bge-m3 rows for the given corpus doc_ids."""
    index_of = {doc_id: i for i, doc_id in enumerate(vector_store.doc_ids)}
    rows = [index_of[doc_id] for doc_id in doc_ids if doc_id in index_of]
    if not rows:
        return np.zeros((0, vector_store.norm_matrix.shape[1]), dtype=np.float32)
    return vector_store.norm_matrix[rows]


def citation_ok(citations: list[int]) -> bool:
    """All citations in range 1..TOP_FOR_ANSWER, and at least one exists.

    An answer with zero citations (e.g. "资料不足") is NOT valid: otherwise
    the empty-set == empty-set case would make "all citations valid"
    trivially true for answers that never cite anything.
    """
    return bool(citations) and all(1 <= n <= TOP_FOR_ANSWER for n in citations)


def score_answer(answer: str, query_emb: np.ndarray, evidence_embs: np.ndarray) -> dict:
    statements = [s for s in split_statements(answer) if not is_meta(s)]
    faithfulness = None
    supported: list[bool] = []
    if statements and evidence_embs.shape[0] > 0:
        embs = np.asarray(embed_texts(statements), dtype=np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embs = embs / norms
        sims = evidence_embs @ embs.T
        supported = [bool(sims[:, i].max() >= FAITH_THRESHOLD) for i in range(len(statements))]
        faithfulness = sum(supported) / len(supported)

    relevancy = None
    if answer.strip():
        answer_emb = np.asarray(embed_texts([answer]), dtype=np.float32)
        norm = float(np.linalg.norm(answer_emb))
        if norm > 0:
            relevancy = float(((answer_emb / norm)[0]) @ query_emb)

    citations = [int(n) for n in re.findall(r"\[(\d{1,2})\]", answer)]
    return {
        "statements": statements,
        "supported": supported,
        "faithfulness": faithfulness,
        "relevancy": relevancy,
        "citations": citations,
        "valid_citations": citation_ok(citations),
    }


def cache_path(n_docs: int) -> Path:
    """E2E answer cache file: explicit per-corpus/per-topK name.

    The corpus size N and TOP_FOR_ANSWER are in the file name (switching
    N or topK must never reuse another config's answers); the actual
    prompt/temperature/retry config is stored inside the file and
    verified on load, so a config change fails loudly instead of
    silently reusing stale answers.
    """
    return Path(__file__).resolve().parent / "data" / "nomiracl" / "chinese" / f"e2e_cache_N{n_docs}_top{TOP_FOR_ANSWER}.json"


def _fingerprint() -> str:
    return hashlib.sha1(
        (PROMPT + RETRY_PROMPT + str(ANSWER_TEMPERATURE) + str(RETRY_REFUSALS)
         + str(TOP_FOR_ANSWER)).encode("utf-8")
    ).hexdigest()[:8]


def load_cache(path: Path) -> dict:
    """Load {qid|method: entry} answers; verify the config fingerprint.

    Raises RuntimeError when the file was written under a different
    config (prompt/temperature/retry/topK changed): the file must be
    deleted to force a re-run.
    """
    if not path.is_file():
        return {}
    blob = json.loads(path.read_text(encoding="utf-8"))
    if blob.get("_fingerprint") != _fingerprint():
        raise RuntimeError(
            f"{path.name} was written under a different config "
            f"(expected fingerprint {_fingerprint()}, found {blob.get('_fingerprint')}); "
            "delete it to re-run with the current config"
        )
    return blob["answers"]


def save_cache(path: Path, answers: dict) -> None:
    blob = {"_fingerprint": _fingerprint(), "answers": answers}
    path.write_text(json.dumps(blob, ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> int:
    args = sys.argv[1:]
    size = int(args[0]) if args else SAMPLE_SIZE
    if len(args) > 1:
        global TOP_FOR_ANSWER
        TOP_FOR_ANSWER = int(args[1])
    docs, eval_queries, stats = prepare_eval()
    n_docs = len(docs)
    full_corpus = stats["full_corpus"]
    doc_text = {doc_id: text for doc_id, text in docs}
    bm25_index, vector_store = build_retrievers(docs)
    results = retrieve_all(bm25_index, vector_store, eval_queries)
    sampled = sample_queries(eval_queries, size)
    print(f"e2e: {len(sampled)} queries x {ANSWER_METHODS} (top-{TOP_FOR_ANSWER} each)")

    query_texts = [query for _qid, query, _rel in sampled]
    query_embs = load_or_embed_queries(query_texts, bm25_index.N)
    norms = np.linalg.norm(query_embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    query_embs = query_embs / norms

    cache = load_cache(cache_path(n_docs))
    rows: list[dict] = []
    errors = 0
    evidence_sizes: list[int] = []
    for i, (qid, query, _relevant) in enumerate(sampled):
        for method in ANSWER_METHODS:
            key = f"{qid}|{method}"
            entry = cache.get(key)
            evidence = [(d, doc_text[d]) for d, _ in results[qid][method][:TOP_FOR_ANSWER]]
            evidence_sizes.append(sum(len(t) for _, t in evidence))
            if entry is None:
                try:
                    answer = ask_deepseek(build_prompt(query, evidence), temperature=ANSWER_TEMPERATURE)
                    for attempt in range(RETRY_REFUSALS):
                        if not is_refusal(answer):
                            break
                        answer = ask_deepseek(build_prompt(query, evidence, retry=True),
                                              temperature=ANSWER_TEMPERATURE)
                except RuntimeError as error:
                    print(f"  ERROR {key}: {error}")
                    errors += 1
                    continue
                entry = {"answer": answer, **score_answer(answer, query_embs[i], evidence_embeddings(vector_store, [d for d, _ in evidence]))}
                cache[key] = entry
                save_cache(cache_path(n_docs), cache)
                time.sleep(0.2)
            f = entry["faithfulness"]
            r = entry["relevancy"]
            rows.append({
                "qid": qid, "query": query, "method": method,
                "faith": f, "rel": r,
                # recompute from cached citations: a stored boolean could
                # be stale (e.g. written by a previous run with different
                # validity rules)
                "valid_cit": citation_ok(entry["citations"]),
                "n_cit": len(entry["citations"]),
                "answer": entry["answer"],
            })
            fstr = f"{f:.3f}" if f is not None else "  n/a"
            rstr = f"{r:.3f}" if r is not None else "n/a"
            print(f"  {key}: faith={fstr} rel={rstr} cites={entry['citations']}")

    if errors:
        print(f"\n{errors} generation errors (skipped)")
    if not rows:
        print("no results")
        return 1

    report = Path(__file__).resolve().parent / "docs" / "W5_E2E_REPORT.md"
    lines = [
        f"# W5 端到端评测：DeepSeek 带引用回答（{len(sampled)} query 样本）",
        "",
        f"- 喂料：rerank top-{TOP_FOR_ANSWER}（融合管线）vs vector top-{TOP_FOR_ANSWER}（单路最优）",
        f"- 打分：faithfulness = 陈述被证据支持的比例（bge-m3 余弦 >= {FAITH_THRESHOLD}，"
        f"陈述拆自答案，含\"资料不足\"等元句剔除）；relevancy = 答案与 query 余弦（>= {RELEVANCY_THRESHOLD} 记相关）",
        f"- 配置：temperature={ANSWER_TEMPERATURE}，纯\"资料不足\"回答自动重试 {RETRY_REFUSALS} 次",
        f"- 证据规模：平均 {sum(evidence_sizes) / len(evidence_sizes):.0f} 字符/回答"
        f"（top-{TOP_FOR_ANSWER}）",
        f"- 缓存：`{cache_path(n_docs).name}`，重跑零 DeepSeek API",
        "",
        "## 逐条结果",
        "",
        "| qid | 方法 | faith | rel | 引用 | 答案摘要 |",
        "|---|---|---:|---:|---|---|",
    ]
    for row in rows:
        f = f"{row['faith']:.3f}" if row["faith"] is not None else "n/a"
        r = f"{row['rel']:.3f}" if row["rel"] is not None else "n/a"
        snippet = row["answer"].replace("\n", " ")[:48]
        lines.append(
            f"| {row['qid']} | {row['method']} | {f} | {r} | "
            f"{'OK' if row['valid_cit'] else 'INVALID'}({row['n_cit']}) | {snippet} |"
        )

    def fmt3(value: float | None) -> str:
        return f"{value:.3f}" if value is not None else "n/a"

    def fmtpct(value: float | None) -> str:
        return f"{value:.1%}" if value is not None else "n/a"

    def aggregate(method: str) -> dict:
        sub = [row for row in rows if row["method"] == method]
        answered = [row for row in sub if row["faith"] is not None]
        fs = [row["faith"] for row in answered]
        rs = [row["rel"] for row in answered]
        cited = [row for row in sub if row["n_cit"] >= 1]
        return {
            "total": len(sub),
            "answered": len(answered),
            "answered_by": {row["qid"] for row in answered},
            "faith_mean": sum(fs) / len(fs) if fs else None,
            "rel_mean": sum(rs) / len(rs) if rs else None,
            "rel_min": min(rs) if rs else None,
            "rel_pass": sum(1 for x in rs if x >= RELEVANCY_THRESHOLD) / len(rs) if rs else None,
            "cit_valid": sum(1 for row in cited if row["valid_cit"]),
            "cit_total": len(cited),
        }

    agg = {method: aggregate(method) for method in ANSWER_METHODS}
    answered_by = {method: agg[method]["answered_by"] for method in ANSWER_METHODS}

    lines += [
        "",
        "## 平均（每方法）",
        "",
        "| 方法 | 样本 | 能答 | 能答率 | faith 平均(可答内) | rel 平均(可答内) | rel 通过(可答内) | 引用合法(有引用) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in ANSWER_METHODS:
        a = agg[method]
        lines.append(
            f"| {method} | {a['total']} | {a['answered']} | {fmtpct(a['answered'] / a['total'])} | "
            f"{fmt3(a['faith_mean'])} | {fmt3(a['rel_mean'])} | {fmtpct(a['rel_pass'])} | "
            f"{a['cit_valid']}/{a['cit_total']} |"
        )

    paired_qids = set.intersection(*answered_by.values())
    paired_faith: dict[str, float | None] = {method: None for method in ANSWER_METHODS}
    if paired_qids:
        for method in ANSWER_METHODS:
            vals = [
                row["faith"] for row in rows
                if row["method"] == method and row["qid"] in paired_qids
            ]
            paired_faith[method] = sum(vals) / len(vals)

    lines += ["", "## 结论"]
    ra, va = agg["rrf+rerank"], agg["vector"]

    diff_answered = ra["answered"] - va["answered"]
    if diff_answered > 0:
        lines.append(f"- 能答率：rrf+rerank {ra['answered']}/{ra['total']} vs "
                     f"vector {va['answered']}/{va['total']}（+{diff_answered} 条，"
                     f"融合喂料让 LLM 更少说\"资料不足\"）")
    elif diff_answered < 0:
        lines.append(f"- 能答率：vector {va['answered']}/{va['total']} vs "
                     f"rrf+rerank {ra['answered']}/{ra['total']}（vector 多答 {-diff_answered} 条）")
    else:
        lines.append(f"- 能答率：两路相同（{ra['answered']}/{ra['total']}）")

    rel_mins = [agg[m]["rel_min"] for m in ANSWER_METHODS if agg[m]["rel_min"] is not None]
    rel_note = ""
    if rel_mins and min(rel_mins) >= RELEVANCY_THRESHOLD:
        rel_note = (f"；可答内 rel 最低 {min(rel_mins):.2f}，阈值 {RELEVANCY_THRESHOLD} 已无区分度，"
                    f"该指标在本次样本上信息量有限")
    lines.append(f"- rel 通过率（可答内）：rrf+rerank {fmtpct(ra['rel_pass'])} vs "
                 f"vector {fmtpct(va['rel_pass'])}{rel_note}。")

    if paired_qids:
        fv, fr = paired_faith["vector"], paired_faith["rrf+rerank"]
        verdict = "两者相当" if abs(fv - fr) < 0.02 else f"{'vector' if fv > fr else 'rrf+rerank'} 更高"
        advantage = "融合的优势在\"多答\"而非\"答得更忠实\"。" if diff_answered else ""
        lines.append(f"- 配对 faith（两边都能答的 {len(paired_qids)} 条）：vector {fv:.3f} vs "
                     f"rrf+rerank {fr:.3f}——{verdict}；{advantage}")

    lines.append(f"- 引用（要求至少 1 条且全在 [1]~[{TOP_FOR_ANSWER}]，0 引用按不合法计）："
                 f"rrf+rerank {ra['cit_valid']}/{ra['cit_total']}，"
                 f"vector {va['cit_valid']}/{va['cit_total']}。")

    never_answered = [
        qid for qid, _query, _relevant in sampled
        if qid not in answered_by["vector"] and qid not in answered_by["rrf+rerank"]
    ]
    if never_answered:
        tail = (f"——全量语料下仍\"资料不足\"；单轮判定随喂料与 LLM 采样存在波动，"
                f"需多轮重复才能区分语料缺失与波动。"
                if full_corpus else
                f"——前 {n_docs} 篇切片证据覆盖不足，扩大语料或能救回一部分。")
        lines.append(f"- 两路均\"资料不足\"：{len(never_answered)} 条"
                     f"（{', '.join(never_answered)}）{tail}")

    lines.append(f"- 声明：阈值 {FAITH_THRESHOLD}/{RELEVANCY_THRESHOLD} 为启发式设定；"
                 f"样本 {len(sampled)} 条，结论为实验性。")
    report.parent.mkdir(exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n报告已写入: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
