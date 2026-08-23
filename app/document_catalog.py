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


class DocumentCatalog:
    """Persistent document identity and immutable content versions."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.db_path = self.root / "catalog.sqlite3"
        self.blob_dir = self.root / "blobs"
        self.root.mkdir(parents=True, exist_ok=True)
        self.blob_dir.mkdir(parents=True, exist_ok=True)
        self._init_schema()

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
                    "VALUES (?, ?, 'processing', ?, ?)",
                    (document_id, name, now, now),
                )
            else:
                db.execute(
                    "UPDATE documents SET status = 'processing', deleted_at = NULL, "
                    "updated_at = ?, last_error = NULL WHERE document_id = ?",
                    (now, document_id),
                )
            db.execute(
                "INSERT INTO document_versions(version_id, document_id, version_number, "
                "content_hash, size_bytes, suffix, stored_path, source_type, source_uri, "
                "status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'processing', ?)",
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

    def active_versions(self) -> list[dict]:
        return [
            row
            for row in self.list_documents()
            if row["status"] == "ready" and row["current_version_id"]
        ]

    def resolve_path(self, record: dict) -> Path:
        return self.root / record["stored_path"]
