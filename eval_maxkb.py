"""CLI 版评测：与 Web 端同一套逻辑（app.kb.KnowledgeBase），题目集从 data/kb/questions.md 读取。

Run:  cd minirage && .venv\\Scripts\\python.exe eval_maxkb.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.kb import KnowledgeBase


def main() -> int:
    kb = KnowledgeBase(ROOT / "data" / "kb")
    print(f"题目集: {kb.questions_path.name}")
    result = kb.run_eval()
    info = result["info"]
    print(f"题目: {info['count']} 题（可答 {info['answer_count']} / 拒答 {info['refuse_count']}）\n")

    print("=" * 76)
    print("逐题结果")
    print("=" * 76)
    for r in result["rows"]:
        cit = f"引用{r['citations']}" if r["citations"] else "无引用"
        kws = f" 命中:{r['hit_kws']}" if r["kind"] == "answer" else ""
        print(f"Q{r['qid']:>2} {r['verdict']:<8} {cit:<12}{kws}")
        print(f"     {r['query']}")
        print(f"     答案: {r['answer'][:70].replace(chr(10), ' ')}")

    s = result["summary"]
    print("\n" + "=" * 76)
    print(f"能答率   : {s['answer_rate']} = "
          f"{sum(1 for r in result['rows'] if r['kind']=='answer' and r['ok']) / info['answer_count']:.1%}")
    print(f"拒答正确率: {s['refuse_rate']} = "
          f"{sum(1 for r in result['rows'] if r['kind']=='refuse' and r['ok']) / info['refuse_count']:.1%}")
    print(f"引用合法 : {s['cited']}")
    print(f"未通过   : {s['fails'] if s['fails'] else '无'}")
    return 0 if not s["fails"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
