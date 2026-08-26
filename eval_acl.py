"""ACL 权限矩阵评测：同一题库 × 多身份，验证授权可答 / 未授权拒答 / 零越权泄露。

Run:
    .venv\\Scripts\\python.exe eval_acl.py                 # 真实跑（调 DeepSeek）
    .venv\\Scripts\\python.exe eval_acl.py --json out.json  # 结果落盘
Exit code: 全部通过 0，存在失败 1。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.kb import KnowledgeBase, matching_keywords  # noqa: E402


def _setup_users(kb: KnowledgeBase, users_cfg: dict) -> None:
    for user_id, spec in users_cfg.items():
        kb.catalog.ensure_user(user_id, spec.get("display_name", user_id))
        for doc_name in spec.get("grant", []):
            record = kb.catalog.get_by_name(doc_name)
            if record is None:
                raise SystemExit(f"[config] 文档不存在: {doc_name}")
            kb.catalog.grant(user_id, record["document_id"])


def _ask_and_check(kb: KnowledgeBase, user_id: str, query: str,
                   expected: str, keywords: list[str]) -> dict:
    """执行一次提问并返回统一结构的判定行。"""
    result = kb.ask(query, user_id=user_id)
    allowed_active = kb.catalog.allowed_document_ids(user_id) & {
        d["id"] for d in kb.list_docs()
    }
    evidence_ids = {e["document_id"] for e in result["evidence"]}
    leak_docs = sorted(evidence_ids - allowed_active)

    answered = (not result["refusal"]) and bool(
        matching_keywords(result["answer"], keywords))
    if expected == "answer":
        ok = answered and not leak_docs
    else:
        ok = bool(result["refusal"]) and not leak_docs

    cited_ok = all(0 < n <= len(result["evidence"]) for n in result["citations"])
    return {
        "user": user_id,
        "query": query,
        "expected": expected,
        "refusal": bool(result["refusal"]),
        "evidence_n": len(result["evidence"]),
        "filtered": (result.get("acl") or {}).get("filtered"),
        "leak_docs": leak_docs,
        "cited_ok": cited_ok,
        "ok": bool(ok and (cited_ok or expected == "refuse")),
    }


def _state_probe_rows(kb: KnowledgeBase, probe: dict) -> list[dict]:
    """删除→隐藏、恢复→可见 的状态机探针：授权关系在软删除后不得泄露。"""
    uid, doc_name = probe["user_id"], probe["doc_name"]
    doc = kb.catalog.get_by_name(doc_name)
    if doc is None:
        raise SystemExit(f"[config] state_probe 文档不存在: {doc_name}")
    kb.catalog.grant(uid, doc["document_id"])

    rows = []
    r1 = _ask_and_check(kb, uid, probe["query"], "answer", probe.get("keywords", []))
    rows.append({**r1, "scenario": "granted_visible"})

    kb.delete_doc(doc["document_id"])
    r2 = _ask_and_check(kb, uid, probe["query"], "refuse", [])
    rows.append({**r2, "scenario": "deleted_hidden"})

    kb.restore_document(doc["document_id"])
    r3 = _ask_and_check(kb, uid, probe["query"], "answer", probe.get("keywords", []))
    rows.append({**r3, "scenario": "restored_visible"})
    return rows


def run_acl_matrix(kb: KnowledgeBase, cfg: dict,
                   include_state_probe: bool = True) -> dict:
    started = time.time()
    _setup_users(kb, cfg.get("users", {}))

    rows = []
    for case in cfg.get("cases", []):
        rows.append(_ask_and_check(
            kb, case["user_id"], case["query"],
            case["expected"], case.get("keywords", [])))

    if include_state_probe and cfg.get("state_probe"):
        rows.extend(_state_probe_rows(kb, cfg["state_probe"]))

    leaks = [r for r in rows if r["leak_docs"]]
    answer_rows = [r for r in rows if r["expected"] == "answer"]
    refuse_rows = [r for r in rows if r["expected"] == "refuse"]
    summary = {
        "total": len(rows),
        "passed": sum(1 for r in rows if r["ok"]),
        "leak_count": len(leaks),
        "越权泄露率": f"{len(leaks)}/{len(rows)}",
        "授权可答率": f"{sum(1 for r in answer_rows if r['ok'])}/{len(answer_rows)}",
        "拒答正确率": f"{sum(1 for r in refuse_rows if r['ok'])}/{len(refuse_rows)}",
        "duration_s": round(time.time() - started, 2),
    }
    return {"rows": rows, "summary": summary}


def main() -> int:
    parser = argparse.ArgumentParser(description="ACL 权限矩阵评测")
    parser.add_argument("--cases", default=str(ROOT / "data" / "kb" / "acl_cases.json"))
    parser.add_argument("--json", dest="json_out",
                        default=str(ROOT / "docs" / "EVAL_ACL.json"))
    parser.add_argument("--no-state-probe", action="store_true")
    args = parser.parse_args()

    cfg = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    kb = KnowledgeBase(ROOT / "data" / "kb")
    report = run_acl_matrix(kb, cfg, include_state_probe=not args.no_state_probe)

    print(f"\n{'结果':<4} {'身份':<8} {'期望':<7} {'证据':>4} {'过滤':>4} 场景/问题")
    for row in report["rows"]:
        mark = "PASS" if row["ok"] else "FAIL"
        topic = row.get("scenario") or row["query"]
        leak = f" 越权:{row['leak_docs']}" if row["leak_docs"] else ""
        print(f"{mark:<4} {row['user']:<8} {row['expected']:<7} "
              f"{row['evidence_n']:>4} {str(row['filtered']):>4} {topic}{leak}")

    s = report["summary"]
    print(f"\n总计 {s['passed']}/{s['total']} 通过 | "
          f"授权可答 {s['授权可答率']} | 拒答正确 {s['拒答正确率']} | "
          f"越权泄露率 {s['越权泄露率']} | 用时 {s['duration_s']}s")

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"报告已写入 {args.json_out}")
    return 0 if s["passed"] == s["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
