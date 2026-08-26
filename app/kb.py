"""知识库服务：文档管理 + 混合索引构建/缓存 + 带引用问答 + 题目集评测。

知识库目录结构（data/kb/）：
    docs/        随项目提供的旧版文档（首次启动自动登记）
    blobs/       按 SHA-256 保存的不可变文档版本
    catalog.sqlite3  文档身份、版本和状态目录
    questions.md 评测题目集（独立于语料，可上传替换，格式见文件头部注释）
    cache/       按文档指纹命名的 BM25 pickle + 向量矩阵缓存（换文档自动换缓存）

Run:  cd minirage && .venv\\Scripts\\python.exe -m uvicorn app.main:app --port 8000
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import shutil
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path

from app.audit import AuditLog

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepseek_chat import ask_deepseek
from index import IndexBuilder, VectorStore, Searcher, fuse, rerank
from index import embeddings as _embeddings
from index.embeddings import (
    MODEL as EMBEDDING_MODEL,
    build_with_cache,
    embed_texts,
)
from docparser.loader import SUPPORTED_SUFFIXES, parse_file
from chunking.structured import chunk_structured
from app.document_catalog import DEFAULT_USER_ID, DocumentCatalog

TOP_FOR_ANSWER = 5
RAW_K = 10
ACL_OVERFETCH = 2
TEMPERATURE = 0.1
INDEX_SCHEMA_VERSION = "enterprise-v1"


def relevance_gate() -> float:
    """向量通道最高相似度门控（幻觉控制）：top1 低于阈值视为无相关证据。

    运行时读 env 以支持单变量实验（KB_RELEVANCE_GATE=0 关闭）；
    只约束向量分数的语义直觉尺度（cosine 0~1），BM25 无界分不适用。
    """
    raw = os.getenv("KB_RELEVANCE_GATE", "0").strip()
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 0.0


@dataclass(frozen=True)
class IndexedChunk:
    chunk_id: str
    document_id: str
    version_id: str
    label: str
    text: str
    heading_path: tuple[str, ...]
    start_char: int
    end_char: int

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

REFUSAL_MARKERS = ("资料不足", "无法", "不能确定", "不清楚", "没有提供", "未找到", "未提及", "未包含")


def build_prompt(query: str, evidence: list[tuple[str, str]], retry: bool = False) -> str:
    blocks = [f"[{i}] {text}" for i, (_doc_id, text) in enumerate(evidence, 1)]
    template = RETRY_PROMPT if retry else PROMPT
    return template.format(evidence="\n".join(blocks), query=query)


def is_refusal(answer: str) -> bool:
    return any(m in answer for m in REFUSAL_MARKERS)


def parse_citations(answer: str) -> list[int]:
    return [int(n) for n in re.findall(r"\[(\d{1,2})\]", answer)]


def matching_keywords(answer: str, checkpoints: list[str]) -> list[str]:
    normalized_answer = re.sub(r"[\W_]+", "", answer, flags=re.UNICODE).casefold()
    return [
        checkpoint for checkpoint in checkpoints
        if re.sub(r"[\W_]+", "", checkpoint, flags=re.UNICODE).casefold()
        in normalized_answer
    ]


def parse_questions(text: str) -> list[dict]:
    """解析题目集 md：按二级标题分组（可答 / 拒答）。

    可答行格式：`问题 | 关键词1 | 关键词2`（命中任一关键词且不拒答即 PASS）；
    拒答行格式：只有问题（回答为拒答即 PASS）。序号与 > 注释行会被忽略。
    """
    questions: list[dict] = []
    kind: str | None = None
    qid = 0
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            title = line.lstrip("#").strip()
            if title.startswith("可答"):
                kind = "answer"
            elif title.startswith("拒答") or "资料不足" in title:
                kind = "refuse"
            continue
        if line.startswith(">"):
            continue
        line = re.sub(r"^\d+[.、)]\s*", "", line)
        line = re.sub(r"^[-*]\s*", "", line)
        if not line or kind is None:
            continue
        qid += 1
        if kind == "answer":
            parts = [p.strip() for p in line.split("|")]
            query, checkpoints = parts[0], [p for p in parts[1:] if p]
        else:
            query, checkpoints = line, []
        questions.append({
            "qid": qid, "query": query, "kind": kind, "checkpoints": checkpoints,
        })
    return questions


class KnowledgeBase:
    """文档 → 分块 → (BM25 + bge-m3 向量) → RRF+rerank → DeepSeek 带引用回答。"""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.docs_dir = self.root / "docs"
        self.cache_dir = self.root / "cache"
        self.vector_cache_dir = self.cache_dir / "vectors"
        self.generations_dir = self.cache_dir / "generations"
        self.questions_path = self.root / "questions.md"
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.vector_cache_dir.mkdir(parents=True, exist_ok=True)
        self.generations_dir.mkdir(parents=True, exist_ok=True)
        self.catalog = DocumentCatalog(self.root)
        self.catalog.migrate_directory(self.docs_dir, SUPPORTED_SUFFIXES)
        self.audit = AuditLog(self.root / "audit.jsonl")
        self._bm25 = None
        self._vector = None
        self._chunks: dict[str, IndexedChunk] = {}
        self._fingerprint_cache: str | None = None
        # ---- 阶段 5 运行时状态（进程内）----
        self._build_lock = threading.Lock()
        self._current_job: dict | None = None
        self._ask_latencies: deque[float] = deque(maxlen=500)
        self._llm_calls = 0
        self._embed_calls = 0
        self._embed_chars = 0
        # 相对成本模型常量：非精确账单，用于横向比较
        self.EMBED_YUAN_PER_1K_CHARS = 0.0007
        self.LLM_YUAN_PER_CALL = 0.01

    def _chat(self, prompt: str, retry: bool = False) -> str:
        """DeepSeek 调用的唯一入口：计数 + 审计不落问题原文。"""
        self._llm_calls += 1
        return ask_deepseek(prompt, temperature=TEMPERATURE)

    def opstats(self) -> dict:
        latencies = sorted(self._ask_latencies)
        p95 = latencies[int(0.95 * (len(latencies) - 1))] if latencies else None
        last_job = self.catalog.recent_build_jobs(1)
        cost = (
            self._embed_chars / 1000 * self.EMBED_YUAN_PER_1K_CHARS
            + self._llm_calls * self.LLM_YUAN_PER_CALL
        )
        return {
            "asks": len(self._ask_latencies),
            "p95_ms": round(p95 * 1000, 1) if p95 is not None else None,
            "llm_calls": self._llm_calls,
            "embed_calls": self._embed_calls,
            "embed_chars": self._embed_chars,
            "estimated_cost_yuan": round(cost, 4),
            "last_build": last_job[0] if last_job else None,
            "note": "进程内累计；成本为相对模型非精确账单",
        }

    # ---------- 文档管理 ----------

    @staticmethod
    def _public_document(doc: dict) -> dict:
        return {
            "id": doc["document_id"],
            "name": doc["name"],
            "size": doc["size_bytes"],
            "mtime": doc["updated_at"],
            "status": doc["status"],
            "version": doc["version_number"],
            "content_hash": doc["content_hash"],
            "last_error": doc["last_error"],
        }

    def list_docs(self) -> list[dict]:
        return [self._public_document(doc) for doc in self.catalog.list_documents()]

    def add_uploads(self, files: list[tuple[str, bytes]]) -> list[str]:
        saved = []
        for name, data in files:
            name = Path(name).name
            if Path(name).suffix.lower() not in SUPPORTED_SUFFIXES or not data:
                continue
            try:
                self.catalog.ingest(name, data, "upload")
                saved.append(name)
                self.audit.record("upload", name=name, ok=True)
            except ValueError as error:
                self.audit.record("upload", name=name, ok=False, error=str(error))
                raise
        return saved

    def import_dir(self, directory: str) -> list[str]:
        src = Path(directory)
        if not src.is_dir():
            raise FileNotFoundError(f"目录不存在: {directory}")
        imported = []
        for path in sorted(src.iterdir()):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            self.catalog.ingest(path.name, path.read_bytes(), "directory", str(path.resolve()))
            imported.append(path.name)
        return imported

    def delete_doc(self, document_id: str) -> bool:
        deleted = self.catalog.delete(document_id)
        if deleted:
            self._fingerprint_cache = None
            self.audit.record("delete", document_id=document_id)
        return deleted

    def restore_document(self, document_id: str) -> dict:
        restored = self.catalog.restore(document_id)
        self._fingerprint_cache = None
        self.audit.record("restore", document_id=document_id)
        return self._public_document(restored)

    def document_versions(self, document_id: str) -> list[dict]:
        fields = (
            "version_id",
            "version_number",
            "content_hash",
            "size_bytes",
            "suffix",
            "source_type",
            "status",
            "error",
            "created_at",
        )
        return [
            {field: version[field] for field in fields}
            for version in self.catalog.versions(document_id)
        ]

    # ---------- 题目集 ----------

    def load_questions(self) -> list[dict]:
        if not self.questions_path.is_file():
            raise ValueError(
                f"题目集不存在: {self.questions_path.name}，请先在界面上传题目集文件")
        questions = parse_questions(self.questions_path.read_text(encoding="utf-8"))
        if not questions:
            raise ValueError("题目集解析为空：请检查格式（## 可答 / ## 拒答 分组）")
        return questions

    def upload_questions(self, filename: str, data: bytes) -> dict:
        if Path(filename).suffix.lower() not in {".md", ".txt"}:
            raise ValueError("题目集仅支持 .md / .txt 格式")
        text = data.decode("utf-8", errors="replace")
        questions = parse_questions(text)
        if not questions:
            raise ValueError("题目集解析为空：请检查格式（## 可答 / ## 拒答 分组）")
        self.questions_path.write_text(text, encoding="utf-8")
        return self.questions_info(questions)

    def questions_info(self, questions: list[dict] | None = None) -> dict:
        if questions is None:
            questions = self.load_questions()
        answers = [q for q in questions if q["kind"] == "answer"]
        refusals = [q for q in questions if q["kind"] == "refuse"]
        return {
            "exists": True,
            "name": self.questions_path.name,
            "count": len(questions),
            "answer_count": len(answers),
            "refuse_count": len(refusals),
        }

    # ---------- 索引 ----------

    def _fingerprint(self) -> str:
        h = hashlib.sha1()
        h.update(INDEX_SCHEMA_VERSION.encode("utf-8"))
        h.update(EMBEDDING_MODEL.encode("utf-8"))
        for doc in sorted(self.catalog.active_versions(), key=lambda item: item["document_id"]):
            h.update(doc["document_id"].encode("utf-8"))
            h.update(doc["current_version_id"].encode("utf-8"))
            h.update(doc["content_hash"].encode("utf-8"))
        return h.hexdigest()[:16]

    def _load_chunks(self) -> list[IndexedChunk]:
        chunks: list[IndexedChunk] = []
        for record in self.catalog.active_versions():
            parsed = parse_file(self.catalog.resolve_path(record))
            doc = replace(parsed, doc_id=record["current_version_id"], filename=record["name"])
            cr = chunk_structured(doc)
            for chunk in cr.chunks:
                if chunk.parent_id is not None:
                    continue
                heading = chunk.heading_path[-1] if chunk.heading_path else "引言"
                label = f"{doc.filename} · {heading}"
                chunks.append(IndexedChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=record["document_id"],
                    version_id=record["current_version_id"],
                    label=label,
                    text=chunk.text,
                    heading_path=chunk.heading_path,
                    start_char=chunk.start_char,
                    end_char=chunk.end_char,
                ))
        return chunks

    def status(self) -> dict:
        fp = self._fingerprint()
        meta = self._load_meta()
        built = meta.get("fingerprint") == fp
        return {
            "docs": self.list_docs(),
            "fingerprint": fp,
            "built": built,
            "built_at": meta.get("built_at"),
            "chunks": meta.get("chunks") if built else None,
            "bm25_terms": meta.get("bm25_terms") if built else None,
        }

    def _generation_dir(self, fp: str) -> Path:
        return self.generations_dir / f"gen_{fp}"

    def _cleanup_old_generations(self, keep_fps: list[str]) -> None:
        """B1 policy: keep active and immediate prior generation, remove older."""
        keep_dirs = {self._generation_dir(fp).resolve() for fp in keep_fps}
        for gen_path in self.generations_dir.glob("gen_*"):
            if gen_path.is_dir() and gen_path.resolve() not in keep_dirs:
                shutil.rmtree(gen_path, ignore_errors=True)

    def _load_meta(self) -> dict:
        meta_path = self.cache_dir / "meta.json"
        if meta_path.is_file():
            try:
                return json.loads(meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        return {}

    def _save_meta(self, fp: str, chunks: int, terms: int) -> None:
        meta = {
            "fingerprint": fp,
            "built_at": time.time(),
            "chunks": chunks,
            "bm25_terms": terms,
        }
        temp_meta = self.cache_dir / f".meta.{uuid.uuid4().hex}.tmp"
        temp_meta.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        os.replace(temp_meta, self.cache_dir / "meta.json")

    def _counting_embed(self, embed_fn):
        """包装 embed 函数：累计会话级调用次数与字符数（相对成本模型）。"""
        def wrapped(texts):
            self._embed_calls += 1
            self._embed_chars += sum(len(t) for t in texts)
            return embed_fn(texts)
        return wrapped

    def rebuild(self, force: bool = False, embed_fn=None,
                job_id: str | None = None) -> dict:
        """原子代际构建与发布；job_id 非 None 时由后台线程驱动并更新实时状态。"""
        own_thread = job_id is not None
        if not own_thread:
            job_id = uuid.uuid4().hex

        started = time.time()
        fp = self._fingerprint()
        if not force and fp == self._fingerprint_cache:
            if own_thread:
                self._current_job["state"] = "done"
            return self.status()

        def set_state(state: str) -> None:
            if own_thread:
                self._current_job["state"] = state
            self.catalog.update_build_job(job_id, state=state)

        self.catalog.start_build_job(job_id, fp)
        embed_before = (self._embed_calls, self._embed_chars)
        try:
            set_state("chunking")
            indexed_chunks = self._load_chunks()
            if not indexed_chunks:
                raise ValueError("知识库为空：请先上传文档或导入目录")
            chunks = [(chunk.chunk_id, chunk.text) for chunk in indexed_chunks]

            gen_dir = self._generation_dir(fp)
            bm25_file = gen_dir / "bm25.pkl"
            vec_dir = gen_dir / "vec"

            # 1. 检查当前代际目录是否已完整就绪
            if bm25_file.is_file() and (vec_dir / "matrix.npy").is_file():
                new_bm25 = pickle.loads(bm25_file.read_bytes())
                new_vector = VectorStore.load(vec_dir)
                stats = {"total": len(chunks), "reused": len(chunks), "embedded": 0}
                print(f"kb: loaded generation for {fp}")
            else:
                # 2. 隔离沙箱构建：在临时目录生成全量资产
                set_state("embedding")
                tmp_gen = self.generations_dir / f".gen_{fp}.{uuid.uuid4().hex}.tmp"
                tmp_vec = tmp_gen / "vec"
                tmp_gen.mkdir(parents=True, exist_ok=True)
                try:
                    new_bm25 = IndexBuilder().build(chunks)
                    # 动态查模块属性：保证单测 mock.patch("index.embeddings.embed_texts") 生效
                    base_embed = embed_fn or _embeddings.embed_texts
                    counting = self._counting_embed(base_embed)
                    new_vector, stats = build_with_cache(
                        chunks, self.vector_cache_dir, embed_fn=counting)

                    (tmp_gen / "bm25.pkl").write_bytes(pickle.dumps(new_bm25))
                    new_vector.save(tmp_vec)
                    manifest = {
                        "fingerprint": fp,
                        "chunks": len(chunks),
                        "bm25_terms": len(new_bm25.postings),
                        "created_at": time.time(),
                    }
                    (tmp_gen / "manifest.json").write_text(
                        json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

                    # 3. 原子更名发布
                    set_state("publishing")
                    if gen_dir.exists():
                        shutil.rmtree(gen_dir, ignore_errors=True)
                    os.replace(tmp_gen, gen_dir)
                    print(f"kb: built + published atomic generation for {fp}")
                except Exception:
                    # A1 策略：构建异常清理临时沙箱，保留原内存与旧索引不受损
                    shutil.rmtree(tmp_gen, ignore_errors=True)
                    raise

            # 4. 内存指针热切换
            self._bm25 = new_bm25
            self._vector = new_vector
            self._chunks = {chunk.chunk_id: chunk for chunk in indexed_chunks}

            old_meta = self._load_meta()
            prior_fp = old_meta.get("fingerprint")
            keep_fps = [fp]
            if prior_fp and prior_fp != fp:
                keep_fps.append(prior_fp)

            self._save_meta(fp, len(chunks), len(new_bm25.postings))
            self._fingerprint_cache = fp
            self._cleanup_old_generations(keep_fps)

            duration = time.time() - started
            d_calls = self._embed_calls - embed_before[0]
            d_chars = self._embed_chars - embed_before[1]
            counts = {"chunks": stats["total"], "embedded": stats["embedded"],
                      "reused": stats["reused"], "embed_calls": d_calls,
                      "embed_chars": d_chars}
            self.catalog.finish_build_job(job_id, "done", **counts)
            self.audit.record("rebuild_done", job_id=job_id, fingerprint=fp,
                              duration_s=round(duration, 3),
                              **{k: v for k, v in counts.items()})
            if own_thread:
                self._current_job.update(counts, state="done",
                                         duration_s=round(duration, 3))
            return self.status()
        except Exception as error:
            self.catalog.finish_build_job(job_id, "failed", error=str(error))
            self.audit.record("rebuild_failed", job_id=job_id, error=str(error))
            raise

    def start_rebuild(self) -> dict:
        """B1 单飞后台构建：已有任务在跑则抛 RuntimeError（路由映射 409）。"""
        if self._current_job is not None and \
                self._current_job.get("state") not in ("done", "failed"):
            raise RuntimeError(f"构建任务进行中: {self._current_job['job_id']}")
        if not self._build_lock.acquire(blocking=False):
            raise RuntimeError("构建任务进行中")
        job_id = uuid.uuid4().hex
        self._current_job = {"job_id": job_id, "state": "queued",
                             "started_at": time.time()}
        threading.Thread(target=self._run_build_job, args=(job_id,),
                         name=f"kb-build-{job_id[:8]}", daemon=True).start()
        return {"job_id": job_id, "state": "queued"}

    def _run_build_job(self, job_id: str) -> None:
        try:
            self.rebuild(force=True, job_id=job_id)
        finally:
            self._build_lock.release()

    def current_build(self) -> dict | None:
        return dict(self._current_job) if self._current_job else None

    def _ensure_built(self) -> None:
        if self._fingerprint_cache == self._fingerprint():
            return
        if self._current_job is not None and \
                self._current_job.get("state") not in ("done", "failed"):
            return  # 后台构建进行中：沿用旧代际继续服务，避免并发重建
        if self.status()["built"]:
            meta = self._load_meta()
            fp = meta["fingerprint"]
            gen_dir = self._generation_dir(fp)
            bm25_file = gen_dir / "bm25.pkl"
            vec_dir = gen_dir / "vec"
            if bm25_file.is_file() and (vec_dir / "matrix.npy").is_file():
                self._bm25 = pickle.loads(bm25_file.read_bytes())
                self._vector = VectorStore.load(vec_dir)
                self._chunks = {chunk.chunk_id: chunk for chunk in self._load_chunks()}
                self._fingerprint_cache = fp
                return
        self.rebuild()

    # ---------- 问答 ----------

    def _retrieve(self, query: str, allowed_ids: set[str] | None = None):
        """Recall -> optional ACL filter -> fuse/rerank/top-k.

        Returns (hits, filtered_count).
        allowed_ids=None keeps the legacy unrestricted path (internal eval).
        When filtering, each channel over-fetches RAW_K * ACL_OVERFETCH so an
        authorized document crowded out by denied ones still survives; denied
        candidates never influence fusion or rerank ordering.
        """
        self._ensure_built()
        searcher = Searcher(self._bm25, self._vector)
        filtered = 0
        if allowed_ids is None:
            bm25_hits = searcher.bm25_search(query, k=RAW_K)
            vector_hits = searcher.vector_search(query, k=RAW_K)
        else:
            active_ids = {doc["id"] for doc in self.list_docs()}
            allowed = allowed_ids & active_ids  # 软删除/非 ready 即使被授权也不可见
            fetch_k = RAW_K * ACL_OVERFETCH
            bm25_hits = searcher.bm25_search(query, k=fetch_k)
            vector_hits = searcher.vector_search(query, k=fetch_k)

            def keep(hits: list[tuple[str, float]]) -> tuple[list[tuple[str, float]], int]:
                kept, dropped = [], 0
                for chunk_id, score in hits:
                    chunk = self._chunks[chunk_id]
                    if chunk.document_id in allowed:
                        kept.append((chunk_id, score))
                    else:
                        dropped += 1
                return kept, dropped

            bm25_hits, dropped_bm25 = keep(bm25_hits)
            vector_hits, dropped_vec = keep(vector_hits)
            filtered = dropped_bm25 + dropped_vec

        gate = relevance_gate()
        if gate > 0:
            top_vec = max((score for _cid, score in vector_hits), default=0.0)
            if top_vec < gate:
                return [], filtered  # 幻觉门控：最高向量相似度仍低于阈值 -> 视为无证据

        hits = rerank(fuse(bm25_hits, vector_hits, k=TOP_FOR_ANSWER, method="rrf"),
                      bm25_hits, vector_hits)[:TOP_FOR_ANSWER]
        return [(self._chunks[chunk_id], score) for chunk_id, score in hits], filtered

    def ask(self, query: str, user_id: str | None = None) -> dict:
        """user_id=None 仅限内部脚本(评测)；HTTP 层必须携带身份，默认拒绝。"""
        if not query.strip():
            raise ValueError("问题不能为空")
        acl_meta = None
        if user_id is None:
            hits, filtered = self._retrieve(query)
        else:
            user_id = user_id.strip()
            if not user_id:
                raise ValueError("缺少用户身份")
            allowed = self.catalog.allowed_document_ids(user_id)
            hits, filtered = self._retrieve(query, allowed_ids=allowed)
            acl_meta = {"user_id": user_id, "filtered": filtered}
        evidence = [(chunk.chunk_id, chunk.text) for chunk, _score in hits]
        started = time.time()
        if not evidence:
            # 确定性拒答：零证据不进模型，杜绝参数知识幻觉与提示词长尾
            answer = "资料不足"
            retried = False
        else:
            answer = self._chat(build_prompt(query, evidence))
            retried = False
            if is_refusal(answer):
                answer = self._chat(build_prompt(query, evidence, retry=True))
                retried = True
        latency_ms = round((time.time() - started) * 1000, 1)
        self._ask_latencies.append(latency_ms / 1000)
        result = {
            "query": query,
            "answer": answer,
            "refusal": is_refusal(answer),
            "retried": retried,
            "citations": parse_citations(answer),
            "evidence": [
                {
                    "rank": i + 1,
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "version_id": chunk.version_id,
                    "label": chunk.label,
                    "score": score,
                    "text": chunk.text,
                    "heading_path": list(chunk.heading_path),
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                }
                for i, (chunk, score) in enumerate(hits)
            ],
        }
        if acl_meta is not None:
            result["acl"] = acl_meta
        result["latency_ms"] = latency_ms
        self.audit.record(
            "ask",
            user_id=acl_meta["user_id"] if acl_meta else "internal",
            filtered=acl_meta["filtered"] if acl_meta else None,
            latency_ms=latency_ms,
            refusal=result["refusal"],
            retried=retried,
        )  # 不记录问题原文，防止敏感内容落盘
        return result

    # ---------- 题目集评测 ----------

    def run_eval(self) -> dict:
        questions = self.load_questions()
        rows = []
        for q in questions:
            qid, query, kind, checkpoints = q["qid"], q["query"], q["kind"], q["checkpoints"]
            hits, _filtered = self._retrieve(query)
            evidence = [(chunk.chunk_id, chunk.text) for chunk, _score in hits]
            if not evidence:
                answer = "资料不足"
                retried = False
            else:
                answer = self._chat(build_prompt(query, evidence))
                retried = False
                if is_refusal(answer):
                    answer = self._chat(build_prompt(query, evidence, retry=True))
                    retried = True
            refusal = is_refusal(answer)
            cites = parse_citations(answer)
            hit_kws = matching_keywords(answer, checkpoints)
            if kind == "answer":
                ok = (not refusal) and bool(hit_kws)
                verdict = "PASS" if ok else ("REFUSE" if refusal else "MISS")
            else:
                ok = refusal
                verdict = "PASS" if ok else "FAIL"
            rows.append({
                "qid": qid, "query": query, "kind": kind, "ok": ok, "verdict": verdict,
                "refusal": refusal, "retried": retried,
                "citations": cites, "hit_kws": hit_kws,
                "answer": answer,
            })

        answered_qs = [r for r in rows if r["kind"] == "answer"]
        refused_qs = [r for r in rows if r["kind"] == "refuse"]
        n_pass = sum(1 for r in answered_qs if r["ok"])
        n_refuse_ok = sum(1 for r in refused_qs if r["ok"])
        n_cited = sum(1 for r in answered_qs if r["ok"] and r["citations"])
        return {
            "info": self.questions_info(questions),
            "rows": rows,
            "summary": {
                "answer_rate": f"{n_pass}/{len(answered_qs)}",
                "refuse_rate": f"{n_refuse_ok}/{len(refused_qs)}",
                "cited": n_cited,
                "fails": [r["qid"] for r in rows if not r["ok"]],
            },
        }
