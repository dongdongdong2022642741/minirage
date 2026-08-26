from __future__ import annotations

import hashlib
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from docparser.loader import parse_file


class DocumentNotFoundError(LookupError):
    """Raised when a document id does not exist in the catalog."""


class DocumentStateError(RuntimeError):
    """Raised when an operation conflicts with the document's current state."""


DEFAULT_USER_ID = "admin"
DEFAULT_USER_NAME = "管理员"


class DocumentCatalog:
    """Persistent document identity and immutable content versions."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.db_path = self.root / "catalog.sqlite3"
        self.blob_dir = self.root / "blobs"
        self.root.mkdir(parents=True, exist_ok=True)
        self.blob_dir.mkdir(parents=True, exist_ok=True)
        self._init_schema()
        self._seed_default_user()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @contextmanager
    def _database(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _init_schema(self) -> None:
        with self._database() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL COLLATE NOCASE UNIQUE,
                    status TEXT NOT NULL,
                    current_version_id TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    deleted_at REAL,
                    last_error TEXT
                );
                CREATE TABLE IF NOT EXISTS document_versions (
                    version_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    version_number INTEGER NOT NULL,
                    content_hash TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    suffix TEXT NOT NULL,
                    stored_path TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    source_uri TEXT,
                    status TEXT NOT NULL,
                    error TEXT,
                    created_at REAL NOT NULL,
                    UNIQUE(document_id, version_number),
                    FOREIGN KEY(document_id) REFERENCES documents(document_id)
                );
                CREATE TABLE IF NOT EXISTS acl_users (
                    user_id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS user_documents (
                    user_id TEXT NOT NULL REFERENCES acl_users(user_id),
                    document_id TEXT NOT NULL REFERENCES documents(document_id),
                    PRIMARY KEY (user_id, document_id)
                );
                CREATE TABLE IF NOT EXISTS build_jobs (
                    job_id TEXT PRIMARY KEY,
                    fingerprint TEXT,
                    state TEXT NOT NULL,
                    started_at REAL NOT NULL,
                    finished_at REAL,
                    chunks INTEGER,
                    embedded INTEGER,
                    reused INTEGER,
                    embed_calls INTEGER,
                    embed_chars INTEGER,
                    error TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_versions_document
                    ON document_versions(document_id, version_number);
                """
            )

    @staticmethod
    def _hash(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def _write_blob(self, content_hash: str, suffix: str, data: bytes) -> Path:
        relative = Path("blobs") / content_hash[:2] / f"{content_hash}{suffix}"
        destination = self.root / relative
        if destination.is_file():
            return destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_bytes(data)
        os.replace(temporary, destination)
        return destination

    def migrate_directory(self, directory: Path | str, supported_suffixes: set[str]) -> None:
        directory = Path(directory)
        for path in sorted(directory.iterdir()):
            if not path.is_file() or path.suffix.lower() not in supported_suffixes:
                continue
            if self.get_by_name(path.name, include_deleted=True) is None:
                self.ingest(path.name, path.read_bytes(), "legacy", str(path.resolve()))

    def ingest(
        self,
        name: str,
        data: bytes,
        source_type: str = "upload",
        source_uri: str | None = None,
    ) -> dict:
        name = Path(name).name
        suffix = Path(name).suffix.lower()
        content_hash = self._hash(data)
        stored_path = self._write_blob(content_hash, suffix, data)
        now = time.time()

        with self._database() as db:
            document = db.execute(
                "SELECT * FROM documents WHERE name = ? COLLATE NOCASE", (name,)
            ).fetchone()
            if document is not None and document["current_version_id"]:
                current = db.execute(
                    "SELECT * FROM document_versions WHERE version_id = ?",
                    (document["current_version_id"],),
                ).fetchone()
                if current is not None and current["content_hash"] == content_hash:
                    db.execute(
                        "UPDATE documents SET status = 'ready', deleted_at = NULL, "
                        "updated_at = ?, last_error = NULL WHERE document_id = ?",
                        (now, document["document_id"]),
                    )
                    db.commit()
                    return self.get(document["document_id"])

            document_id = document["document_id"] if document else uuid.uuid4().hex
            next_version = db.execute(
                "SELECT COALESCE(MAX(version_number), 0) + 1 FROM document_versions "
                "WHERE document_id = ?",
                (document_id,),
            ).fetchone()[0]
            version_id = uuid.uuid4().hex
            if document is None:
                db.execute(
                    "INSERT INTO documents(document_id, name, status, created_at, updated_at) "
                    "VALUES (?, ?, 'parsing', ?, ?)",
                    (document_id, name, now, now),
                )
            else:
                db.execute(
                    "UPDATE documents SET status = 'parsing', deleted_at = NULL, "
                    "updated_at = ?, last_error = NULL WHERE document_id = ?",
                    (now, document_id),
                )
            db.execute(
                "INSERT INTO document_versions(version_id, document_id, version_number, "
                "content_hash, size_bytes, suffix, stored_path, source_type, source_uri, "
                "status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?)",
                (
                    version_id,
                    document_id,
                    next_version,
                    content_hash,
                    len(data),
                    suffix,
                    str(stored_path.relative_to(self.root)),
                    source_type,
                    source_uri,
                    now,
                ),
            )
            db.execute(
                "UPDATE document_versions SET status = 'parsing' WHERE version_id = ?",
                (version_id,),
            )

        try:
            parse_file(stored_path)
        except (ValueError, UnicodeError) as error:
            with self._database() as db:
                db.execute(
                    "UPDATE document_versions SET status = 'failed', error = ? "
                    "WHERE version_id = ?",
                    (str(error), version_id),
                )
                document_status = "ready" if document and document["current_version_id"] else "failed"
                db.execute(
                    "UPDATE documents SET status = ?, last_error = ?, updated_at = ? "
                    "WHERE document_id = ?",
                    (document_status, str(error), time.time(), document_id),
                )
            raise ValueError(f"文档解析失败: {name}: {error}") from error

        with self._database() as db:
            db.execute(
                "UPDATE document_versions SET status = 'ready' WHERE version_id = ?",
                (version_id,),
            )
            db.execute(
                "UPDATE documents SET status = 'ready', current_version_id = ?, "
                "updated_at = ?, last_error = NULL WHERE document_id = ?",
                (version_id, time.time(), document_id),
            )
            db.execute(
                "INSERT OR IGNORE INTO user_documents(user_id, document_id) VALUES (?, ?)",
                (DEFAULT_USER_ID, document_id),
            )
        return self.get(document_id)

    def list_documents(self, include_deleted: bool = False) -> list[dict]:
        where = "" if include_deleted else "WHERE d.deleted_at IS NULL"
        with self._database() as db:
            rows = db.execute(
                f"""
                SELECT d.*, v.version_number, v.content_hash, v.size_bytes, v.suffix,
                       v.stored_path, v.source_type, v.source_uri
                FROM documents d
                LEFT JOIN document_versions v ON v.version_id = d.current_version_id
                {where}
                ORDER BY d.name COLLATE NOCASE
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def get(self, document_id: str) -> dict | None:
        with self._database() as db:
            row = db.execute(
                """
                SELECT d.*, v.version_number, v.content_hash, v.size_bytes, v.suffix,
                       v.stored_path, v.source_type, v.source_uri
                FROM documents d
                LEFT JOIN document_versions v ON v.version_id = d.current_version_id
                WHERE d.document_id = ?
                """,
                (document_id,),
            ).fetchone()
        return dict(row) if row else None

    def get_by_name(self, name: str, include_deleted: bool = False) -> dict | None:
        deleted_filter = "" if include_deleted else "AND deleted_at IS NULL"
        with self._database() as db:
            row = db.execute(
                f"SELECT document_id FROM documents WHERE name = ? COLLATE NOCASE {deleted_filter}",
                (Path(name).name,),
            ).fetchone()
        return self.get(row["document_id"]) if row else None

    def versions(self, document_id: str) -> list[dict]:
        with self._database() as db:
            rows = db.execute(
                "SELECT * FROM document_versions WHERE document_id = ? "
                "ORDER BY version_number DESC",
                (document_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def delete(self, document_id: str) -> bool:
        now = time.time()
        with self._database() as db:
            changed = db.execute(
                "UPDATE documents SET status = 'deleted', deleted_at = ?, updated_at = ? "
                "WHERE document_id = ? AND deleted_at IS NULL",
                (now, now, document_id),
            ).rowcount
        return changed == 1

    def restore(self, document_id: str) -> dict:
        """Revive a soft-deleted document by pointing back to its ready version.

        Guard-first strategy (plan B): a single guarded UPDATE performs the
        state transition atomically; only when it affects no rows do we read
        the current record once to diagnose which precondition failed.
        """
        now = time.time()
        with self._database() as db:
            changed = db.execute(
                "UPDATE documents SET status = 'ready', deleted_at = NULL, "
                "updated_at = ?, last_error = NULL "
                "WHERE document_id = ? AND status = 'deleted' "
                "AND current_version_id IS NOT NULL",
                (now, document_id),
            ).rowcount
        if changed == 0:
            record = self.get(document_id)
            if record is None:
                raise DocumentNotFoundError(f"document not found: {document_id}")
            if record["status"] != "deleted":
                raise DocumentStateError(f"document is not deleted: {document_id}")
            raise DocumentStateError(
                f"deleted document has no restorable version: {document_id}"
            )
        return self.get(document_id)

    # ---------- ACL（C1 白名单 / E1 默认拒绝） ----------

    def _seed_default_user(self) -> None:
        """Bootstrap for default-deny: admin owns every active ready document.

        Re-synced on every startup (self-healing). Trade-off: the ingest
        identity cannot be revoked within a running deployment.
        """
        with self._database() as db:
            db.execute(
                "INSERT OR IGNORE INTO acl_users(user_id, display_name) VALUES (?, ?)",
                (DEFAULT_USER_ID, DEFAULT_USER_NAME),
            )
            db.execute(
                "INSERT OR IGNORE INTO user_documents(user_id, document_id) "
                "SELECT ?, document_id FROM documents "
                "WHERE deleted_at IS NULL AND status = 'ready'",
                (DEFAULT_USER_ID,),
            )

    def ensure_user(self, user_id: str, display_name: str = "") -> None:
        user_id = user_id.strip()
        if not user_id:
            raise ValueError("user_id 不能为空")
        with self._database() as db:
            db.execute(
                "INSERT OR IGNORE INTO acl_users(user_id, display_name) VALUES (?, ?)",
                (user_id, display_name.strip() or user_id),
            )

    def grant(self, user_id: str, document_id: str) -> None:
        with self._database() as db:
            if db.execute("SELECT 1 FROM acl_users WHERE user_id = ?",
                          (user_id,)).fetchone() is None:
                raise ValueError(f"用户不存在: {user_id}")
            if db.execute("SELECT 1 FROM documents WHERE document_id = ?",
                          (document_id,)).fetchone() is None:
                raise ValueError(f"文档不存在: {document_id}")
            db.execute(
                "INSERT OR IGNORE INTO user_documents(user_id, document_id) VALUES (?, ?)",
                (user_id, document_id),
            )

    def revoke(self, user_id: str, document_id: str) -> bool:
        with self._database() as db:
            changed = db.execute(
                "DELETE FROM user_documents WHERE user_id = ? AND document_id = ?",
                (user_id, document_id),
            ).rowcount
        return changed == 1

    def allowed_document_ids(self, user_id: str) -> set[str]:
        with self._database() as db:
            rows = db.execute(
                "SELECT document_id FROM user_documents WHERE user_id = ?", (user_id,)
            ).fetchall()
        return {row["document_id"] for row in rows}

    def list_users(self) -> list[dict]:
        with self._database() as db:
            rows = db.execute(
                "SELECT user_id, display_name FROM acl_users ORDER BY user_id"
            ).fetchall()
        return [dict(row) for row in rows]

    # ---------- 构建任务台账（A1 两段式状态机：chunking/embedding/publishing） ----------

    def start_build_job(self, job_id: str, fingerprint: str) -> None:
        with self._database() as db:
            db.execute(
                "INSERT INTO build_jobs(job_id, fingerprint, state, started_at) "
                "VALUES (?, ?, 'chunking', ?)",
                (job_id, fingerprint, time.time()),
            )

    def update_build_job(self, job_id: str, **fields) -> None:
        if not fields:
            return
        assignments = ", ".join(f"{key} = ?" for key in fields)
        values = list(fields.values()) + [job_id]
        with self._database() as db:
            db.execute(f"UPDATE build_jobs SET {assignments} WHERE job_id = ?", values)

    def finish_build_job(self, job_id: str, state: str, error: str | None = None,
                         **counts) -> None:
        fields = {"state": state, "finished_at": time.time(), "error": error, **counts}
        self.update_build_job(job_id, **fields)

    def recent_build_jobs(self, limit: int = 10) -> list[dict]:
        with self._database() as db:
            rows = db.execute(
                "SELECT * FROM build_jobs ORDER BY started_at DESC LIMIT ?",
                (max(1, limit),),
            ).fetchall()
        return [dict(row) for row in rows]

    def active_versions(self) -> list[dict]:
        return [
            row
            for row in self.list_documents()
            if row["status"] == "ready" and row["current_version_id"]
        ]

    def resolve_path(self, record: dict) -> Path:
        return self.root / record["stored_path"]
