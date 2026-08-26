"""Append-only JSONL audit log (J1): replayable, crash-safe, human-readable."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path


class AuditLog:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def record(self, event: str, **fields) -> None:
        entry = {"ts": round(time.time(), 3), "event": event, **fields}
        line = json.dumps(entry, ensure_ascii=False)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def tail(self, n: int = 30) -> list[dict]:
        if not self.path.is_file():
            return []
        with self._lock:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        events = []
        for raw in lines[-max(1, n):]:
            try:
                events.append(json.loads(raw))
            except json.JSONDecodeError:
                continue  # 崩溃残留的半行：跳过不中断回放
        return events
