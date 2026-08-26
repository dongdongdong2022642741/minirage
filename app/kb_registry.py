"""多知识库注册表：每个知识库是独立的数据根目录。

隔离模型（与 RAGFlow 多 dataset 同构）：
    data/kbs/registry.json          # 挂载清单
    data/kb/                        # 默认库 main（历史数据原位保留）
    data/kbs/<kb_id>/               # 其余库各自完整目录树

任何层（catalog/blobs/cache/audit）都以 root 为世界边界，
因此"挂载哪个库就只见哪个库"由实例隔离天然保证。
"""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path

from app.kb import KnowledgeBase

DEFAULT_KB_ID = "main"
_KB_ID_RE = re.compile(r"^[a-z][a-z0-9_-]{1,31}$")


class KBRegistry:
    def __init__(self, base: Path | str, legacy_root: Path | str,
                 default_name: str = "企业知识库") -> None:
        self.base = Path(base)
        self.base.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.base / "registry.json"
        self._lock = threading.Lock()
        self._cache: dict[str, KnowledgeBase] = {}
        self._ensure_default(legacy_root, default_name)

    # ---------- 清单 ----------

    def _read(self) -> dict:
        if not self.manifest_path.is_file():
            return {"kbs": []}
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def _write(self, blob: dict) -> None:
        tmp = self.manifest_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(blob, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        tmp.replace(self.manifest_path)

    def _ensure_default(self, legacy_root: Path | str, name: str) -> None:
        blob = self._read()
        if any(e["kb_id"] == DEFAULT_KB_ID for e in blob["kbs"]):
            return
        blob["kbs"].insert(0, {
            "kb_id": DEFAULT_KB_ID,
            "name": name,
            "description": "默认知识库（历史数据）",
            "root": str(Path(legacy_root).resolve()),
            "created_at": time.time(),
        })
        self._write(blob)

    def list(self) -> list[dict]:
        entries = sorted(self._read()["kbs"], key=lambda e: e["created_at"])
        for e in entries:
            e["docs_ready"] = None  # 占位：前端自行按需拉 /api/status?kb_id=
        return entries

    def get_meta(self, kb_id: str) -> dict | None:
        for e in self._read()["kbs"]:
            if e["kb_id"] == kb_id:
                return e
        return None

    def create(self, kb_id: str, name: str, description: str = "") -> dict:
        kb_id = kb_id.strip()
        if not _KB_ID_RE.match(kb_id):
            raise ValueError(
                "kb_id 仅允许小写字母开头，含小写字母/数字/-/_，长度 2~32")
        with self._lock:
            blob = self._read()
            if any(e["kb_id"] == kb_id for e in blob["kbs"]):
                raise ValueError(f"知识库已存在: {kb_id}")
            entry = {
                "kb_id": kb_id,
                "name": name.strip() or kb_id,
                "description": description.strip(),
                "root": str((self.base / kb_id).resolve()),
                "created_at": time.time(),
            }
            blob["kbs"].append(entry)
            self._write(blob)
        Path(entry["root"]).mkdir(parents=True, exist_ok=True)
        self.get(kb_id)  # 触发初始化（建目录/schema/种子用户）
        return entry

    def root_of(self, kb_id: str) -> Path | None:
        meta = self.get_meta(kb_id)
        return Path(meta["root"]) if meta else None

    # ---------- 实例 ----------

    def get(self, kb_id: str = DEFAULT_KB_ID) -> KnowledgeBase:
        meta = self.get_meta(kb_id)
        if meta is None:
            raise KeyError(f"未挂载的知识库: {kb_id}")
        with self._lock:
            inst = self._cache.get(kb_id)
            if inst is None:
                inst = KnowledgeBase(meta["root"])
                self._cache[kb_id] = inst
            return inst
