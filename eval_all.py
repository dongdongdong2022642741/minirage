"""五套评测题集聚合入口：清单 / 按需执行。

默认（无参数）只打印题集清单与最近报告状态，不产生任何 API 费用。
选择执行：

    python eval_all.py --run enterprise          # 60 题业务集
    python eval_all.py --run acl                 # 权限矩阵
    python eval_all.py --run nomiracl            # NoMIRACL dev 30+30 抽样
    python eval_all.py --run nomiracl:100,100    # 扩样 100+100
    python eval_all.py --run crud:200            # CRUD-RAG 前 200 问切片
    python eval_all.py --run ragas:10            # RAGAS 校准 n=10

成本提示：enterprise/acl 走确定性判分但 ask 会调 DeepSeek；
nomiracl/crud/ragas 均真实调用 LLM/嵌入，执行前请确认额度。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

SETS = {
    "enterprise": {
        "desc": "企业业务 v2 · 60 题（48 可答 + 12 拒答）",
        "latest": "docs/EVAL_ENTERPRISE_V2.json",
        "cmd": None,  # 内嵌执行（见 run_enterprise）
    },
    "acl": {
        "desc": "ACL 权限矩阵 · 34 行 × 5 身份",
        "latest": "docs/EVAL_ACL.json",
        "cmd": [sys.executable, "-X", "utf8", "eval_acl.py",
                "--json", "docs/EVAL_ACL.json"],
    },
    "nomiracl": {
        "desc": "NoMIRACL zh 官方拒答集 · 全量 3,770 query（默认抽样 30+30）",
        "latest": "docs/EVAL_NOMIRACL_RAG_dev.json",
        "cmd": [sys.executable, "-X", "utf8", "eval_nomiracl_rag.py"],
    },
    "crud": {
        "desc": "CRUD-RAG 分层问答 · 官方 10,421 问（默认切片 200）",
        "latest": "docs/EVAL_CRUD_RAG_200.json",
        "cmd": [sys.executable, "-X", "utf8", "eval_crud_rag.py", "--total", "200"],
    },
    "ragas": {
        "desc": "RAGAS 四指标校准 · 基准 n=10",
        "latest": "docs/EVAL_RAGAS.json",
        "cmd": [sys.executable, "-X", "utf8", "eval_ragas.py", "--n", "10"],
    },
}


def run_enterprise() -> int:
    from app.kb import KnowledgeBase

    kb = KnowledgeBase(ROOT / "data" / "kb")
    report = kb.run_eval()
    out = ROOT / "docs" / "EVAL_ENTERPRISE_V2.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False))
    print(f"报告已写入 {out}")
    return 0 if not report["summary"]["fails"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="MiniRAG 五套评测聚合入口")
    parser.add_argument(
        "--run", default="",
        help="逗号分隔的题集名，可带 :参数（如 nomiracl:100,100 crud:200）；"
             "留空仅打印清单")
    args = parser.parse_args()

    if not args.run:
        print("MiniRAG 评测题集总览（详见 docs/EVAL_SETS.md）\n")
        for name, meta in SETS.items():
            latest = ROOT / meta["latest"]
            state = latest.name if latest.is_file() else "尚无报告"
            print(f"  {name:<11} {meta['desc']}\n{'':<15}最近报告: {state}\n")
        print("执行示例: python eval_all.py --run acl,enterprise")
        return 0

    failures = []
    for token in [t for t in args.run.split(",") if t.strip()]:
        name, _, param = token.partition(":")
        name = name.strip()
        if name not in SETS:
            print(f"[skip] 未知题集: {name}")
            failures.append(name)
            continue
        cmd = SETS[name]["cmd"]
        if cmd is None:
            print(f"\n===== {name} =====")
            code = run_enterprise()
        else:
            run_cmd = list(cmd)
            if param:
                if name == "nomiracl":
                    a, b = (param.split(",") + ["30"])[:2]
                    run_cmd += ["--n-rel", a, "--n-nonrel", b]
                elif name == "crud":
                    run_cmd[run_cmd.index("--total") + 1] = param
                elif name == "ragas":
                    run_cmd[run_cmd.index("--n") + 1] = param
            print(f"\n===== {name} =====")
            code = subprocess.call(run_cmd, cwd=ROOT)
        if code != 0:
            failures.append(name)

    print("\n===== 聚合结果 =====")
    print("全部通过" if not failures else f"失败/跳过: {failures}")
    return 0 if not failures else 1


if __name__ == "__main__":
    import json
    raise SystemExit(main())
