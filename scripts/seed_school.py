"""学校场景知识库种子脚本：建库、写入五册手册与专属题集、配置 ACL 并构建索引。

Run: .venv\\Scripts\\python.exe scripts/seed_school.py
幂等：重复执行只补缺失文档/授权；索引仅在指纹未命中时重建。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.kb_registry import DEFAULT_KB_ID, KBRegistry  # noqa: E402

DOCS = {
    "01_选课与学分.md": """# 选课与学分

## 选课时间

- 每学期第 1-2 周为选课期，第 3 周可退改选
- 选课系统开放时间为每天 9:00-22:00

## 学分要求

- 本科生每学期建议修读 20-26 学分，上限 30 学分
- 跨专业选修须经开课学院审核，每学期不超过 8 学分

## 重修

- 挂科课程重修需缴纳重修费，标准为每学分 100 元
""",
    "02_考试与成绩.md": """# 考试与成绩

## 缓考

- 缓考申请须在考试前 3 天提交，因病因事需附证明材料

## 成绩构成

- 课程总评成绩 = 平时成绩 40% + 期末成绩 60%

## 绩点

- 绩点分段：90 分以上 4.0，85-89 计 3.7，80-84 计 3.3

## 成绩复核

- 对成绩有异议可在公布后 5 个工作日内申请复核
""",
    "03_奖助学金.md": """# 奖助学金

## 奖学金

- 国家奖学金 8000 元每年，评选比例约 2%
- 国家励志奖学金 5000 元每年，要求家庭经济困难认定
- 校级奖学金：一等 3000 元、二等 2000 元、三等 1000 元
- 申请时间为每年 9 月 1 日至 15 日，需辅导员推荐

## 助学金

- 困难补助助学金每月 350 元，按 10 个月发放
""",
    "04_图书馆指南.md": """# 图书馆指南

## 开放时间

- 周一至周日 7:30-22:30，考试周延长至 23:30

## 借阅规则

- 本科生最多可借 10 册，借期 30 天，可续借一次加 15 天
- 超期费用 0.1 元每天每册

## 研修间

- 研修间需提前 1 天在图书馆公众号预约，每次最长 4 小时
""",
    "05_宿舍与后勤.md": """# 宿舍与后勤

## 门禁与断电

- 宿舍门禁时间 23:00，周日到周四 23:30 断电，周五周六不断电

## 生活服务

- 热水供应时段 6:00-8:00 与 17:00-23:00
- 设施报修通过后勤公众号提交，48 小时内响应

## 调宿

- 调宿申请每学期第 1-2 周集中受理，需双方辅导员签字
""",
}

QUESTIONS_MD = """# 学校场景评测题集

> 可答题格式：`问题 | 关键词1 | 关键词2`；拒答题只有问题。

## 可答

1. 本科生一学期最多修多少学分 | 30
2. 选课系统每天几点开放 | 9:00
3. 挂科重修的费用是多少 | 100
4. 缓考要提前几天申请 | 3天
5. 课程总评成绩怎么构成 | 40 | 平时
6. 绩点 90 分以上是多少 | 4.0
7. 国家奖学金金额是多少 | 8000
8. 校级一等奖学金多少钱 | 3000
9. 助学金每个月发多少 | 350
10. 图书馆考试周开到几点 | 23:30
11. 本科生最多能借几本书 | 10册
12. 图书超期费用怎么算 | 0.1
13. 研修间最长能用多久 | 4小时
14. 宿舍门禁是几点 | 23:00
15. 周五晚上宿舍断电吗 | 不断电
16. 热水供应到几点结束 | 23:00
17. 报修后多长时间会有响应 | 48小时
18. 跨专业选修每学期上限多少学分 | 8

## 拒答

1. 这周食堂午餐菜单是什么
2. 校车时刻表在哪里查
3. 学费通过什么方式缴纳
4. 毕业照什么时候拍摄
5. 校医院怎么挂号
6. 考研辅导班收费多少钱
"""


def main() -> int:
    registry = KBRegistry(ROOT / "data" / "kbs", ROOT / "data" / "kb")
    meta = registry.get_meta("school")
    if meta is None:
        meta = registry.create("school", "校园服务知识库",
                               "选课/考试/奖助/图书馆/宿舍 场景")
    kb = registry.get("school")

    docs_dir = Path(meta["root"]) / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    for name, text in DOCS.items():
        (docs_dir / name).write_text(text, encoding="utf-8")

    qpath = Path(meta["root"]) / "questions.md"
    if not qpath.exists():
        qpath.write_text(QUESTIONS_MD, encoding="utf-8")

    # 迁移登记 + ACL 种子（学生只见生活类，教师见教务类）
    kb.catalog.migrate_directory(docs_dir, {".md"})
    kb.catalog._seed_default_user()
    grants = {
        "teacher": ["01_选课与学分.md", "02_考试与成绩.md", "03_奖助学金.md"],
        "student": ["04_图书馆指南.md", "05_宿舍与后勤.md"],
    }
    for uid, names in grants.items():
        kb.catalog.ensure_user(uid, {"teacher": "教师", "student": "学生"}[uid])
        for name in names:
            rec = kb.catalog.get_by_name(name)
            if rec:
                kb.catalog.grant(uid, rec["document_id"])

    print(f"school 文档: {[d['name'] for d in kb.list_docs()]}")
    print(f"用户: {kb.catalog.list_users()}")

    status = kb.status()
    if not status["built"]:
        print("构建索引中（真实 Embedding，文档量小）...")
        kb.rebuild(force=True)
    print("索引状态:", {k: v for k, v in kb.status().items()
                        if k in ("built", "chunks", "bm25_terms")})
    print(json.dumps(registry.list(), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
