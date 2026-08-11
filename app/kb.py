"""知识库服务：文档管理 + 混合索引构建/缓存 + 带引用问答 + 题目集评测。

知识库目录结构（data/kb/）：
    docs/        文档文件（网页上传或本地目录导入后复制到这里）
    questions.md 评测题目集（独立于语料，可上传替换，格式见文件头部注释）
    cache/       按文档指纹命名的 BM25 pickle + 向量矩阵缓存（换文档自动换缓存）

Run:  cd minirage && .venv\\Scripts\\python.exe -m uvicorn app.main:app --port 8000
"""

from __future__ import annotations

import hashlib
import json
import pickle
import re
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepseek_chat import ask_deepseek
from index import IndexBuilder, VectorStore, Searcher, fuse, rerank
from docparser.loader import load_documents
from chunking.structured import chunk_structured

TOP_FOR_ANSWER = 5
RAW_K = 10
TEMPERATURE = 0.1
SUPPORTED_SUFFIXES = {".md", ".txt"}

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
        self.questions_path = self.root / "questions.md"
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._bm25 = None
        self._vector = None
        self._chunks: list[tuple[str, str]] = []
        self._fingerprint_cache: str | None = None

    # ---------- 文档管理 ----------

    def list_docs(self) -> list[dict]:
        docs = []
        for path in sorted(self.docs_dir.iterdir()):
            if path.is_file():
                stat = path.stat()
                docs.append({
                    "name": path.name,
                    "size": stat.st_size,
                    "mtime": stat.st_mtime,
                })
        return docs

    def add_uploads(self, files: list[tuple[str, bytes]]) -> list[str]:
        saved = []
        for name, data in files:
            name = Path(name).name
            if Path(name).suffix.lower() not in SUPPORTED_SUFFIXES or not data:
                continue
            path = self.docs_dir / name
            if path.exists():
                stem, suffix = path.stem, path.suffix
                i = 2
                while (self.docs_dir / f"{stem}_{i}{suffix}").exists():
                    i += 1
                path = self.docs_dir / f"{stem}_{i}{suffix}"
            path.write_bytes(data)
            saved.append(path.name)
        return saved

    def import_dir(self, directory: str) -> list[str]:
        src = Path(directory)
        if not src.is_dir():
            raise FileNotFoundError(f"目录不存在: {directory}")
        imported = []
        for path in sorted(src.iterdir()):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            dst = self.docs_dir / path.name
            if dst.exists():
                continue
            shutil.copy2(path, dst)
            imported.append(path.name)
        return imported

    def delete_doc(self, name: str) -> bool:
        path = self.docs_dir / Path(name).name
        if path.is_file():
            path.unlink()
            return True
        return False

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
        for path in sorted(self.docs_dir.iterdir()):
            if not path.is_file():
                continue
            stat = path.stat()
            h.update(path.name.encode("utf-8"))
            h.update(str(stat.st_size).encode("utf-8"))
            h.update(str(stat.st_mtime_ns).encode("utf-8"))
        return h.hexdigest()[:16]

    def _load_chunks(self) -> list[tuple[str, str]]:
        result = load_documents(self.docs_dir)
        chunks: list[tuple[str, str]] = []
        for doc in result.documents:
            cr = chunk_structured(doc)
            for chunk in cr.chunks:
                if chunk.parent_id is not None:
                    continue
                heading = chunk.heading_path[-1] if chunk.heading_path else "引言"
                label = f"{doc.filename} · {heading}"
                chunks.append((label, chunk.text))
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
        (self.cache_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    def rebuild(self, force: bool = False) -> dict:
        fp = self._fingerprint()
        if not force and fp == self._fingerprint_cache:
            return self.status()

        chunks = self._load_chunks()
        if not chunks:
            raise ValueError("知识库为空：请先上传文档或导入目录")

        bm25_path = self.cache_dir / f"bm25_{fp}.pkl"
        vec_dir = self.cache_dir / f"vec_{fp}"
        if bm25_path.is_file() and (vec_dir / "matrix.npy").is_file():
            self._bm25 = pickle.loads(bm25_path.read_bytes())
            self._vector = VectorStore.load(vec_dir)
            print(f"kb: loaded cached index for {fp}")
        else:
            self._bm25 = IndexBuilder().build(chunks)
            self._vector = VectorStore.build(chunks)
            bm25_path.write_bytes(pickle.dumps(self._bm25))
            self._vector.save(vec_dir)
            print(f"kb: built + cached index for {fp}")

        self._chunks = chunks
        self._fingerprint_cache = fp
        self._save_meta(fp, len(chunks), len(self._bm25.postings))
        return self.status()

    def _ensure_built(self) -> None:
        if self._fingerprint_cache == self._fingerprint():
            return
        if self.status()["built"]:
            meta = self._load_meta()
            fp = meta["fingerprint"]
            bm25_path = self.cache_dir / f"bm25_{fp}.pkl"
            vec_dir = self.cache_dir / f"vec_{fp}"
            self._bm25 = pickle.loads(bm25_path.read_bytes())
            self._vector = VectorStore.load(vec_dir)
            self._chunks = self._load_chunks()
            self._fingerprint_cache = fp
            return
        self.rebuild()

    # ---------- 问答 ----------

    def _retrieve(self, query: str) -> list[tuple[str, float, str]]:
        self._ensure_built()
        searcher = Searcher(self._bm25, self._vector)
        bm25_hits = searcher.bm25_search(query, k=RAW_K)
        vector_hits = searcher.vector_search(query, k=RAW_K)
        hits = rerank(fuse(bm25_hits, vector_hits, k=TOP_FOR_ANSWER, method="rrf"),
                      bm25_hits, vector_hits)[:TOP_FOR_ANSWER]
        text_by_label = dict(self._chunks)
        return [(label, score, text_by_label[label]) for label, score in hits]

    def ask(self, query: str) -> dict:
        if not query.strip():
            raise ValueError("问题不能为空")
        hits = self._retrieve(query)
        evidence = [(label, text) for label, _score, text in hits]
        answer = ask_deepseek(build_prompt(query, evidence), temperature=TEMPERATURE)
        retried = False
        if is_refusal(answer):
            answer = ask_deepseek(build_prompt(query, evidence, retry=True),
                                  temperature=TEMPERATURE)
            retried = True
        return {
            "query": query,
            "answer": answer,
            "refusal": is_refusal(answer),
            "retried": retried,
            "citations": parse_citations(answer),
            "evidence": [
                {"rank": i + 1, "label": label, "score": score, "text": text}
                for i, (label, score, text) in enumerate(hits)
            ],
        }

    # ---------- 题目集评测 ----------

    def run_eval(self) -> dict:
        questions = self.load_questions()
        rows = []
        for q in questions:
            qid, query, kind, checkpoints = q["qid"], q["query"], q["kind"], q["checkpoints"]
            hits = self._retrieve(query)
            evidence = [(label, text) for label, _score, text in hits]
            answer = ask_deepseek(build_prompt(query, evidence), temperature=TEMPERATURE)
            retried = False
            if is_refusal(answer):
                answer = ask_deepseek(build_prompt(query, evidence, retry=True),
                                      temperature=TEMPERATURE)
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
