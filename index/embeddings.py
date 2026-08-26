"""VectorStore: document embedding matrix + numpy cosine search.

Contract with Searcher:
    search(query, k) -> list[(doc_id: str, score: float)] sorted desc.
    Empty query or empty store -> [].
    API errors propagate (never swallowed).
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

API_URL = "https://api.siliconflow.cn/v1/embeddings"
MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024
BATCH_SIZE = 64

# R1 重试策略：仅网络类瞬时错误退避重试，参数类错误立即失败
EMBED_MAX_ATTEMPTS = 3
EMBED_BACKOFF_BASE = 0.5
RETRYABLE_HTTP_CODES = {429, 500, 502, 503, 504}


def embedding_cache_key(model: str, text: str) -> str:
    """Content-addressed cache key for one chunk embedding.

    A vector is a function of (model, text) only: chunk ids and document
    versions must never appear in the key, or unchanged texts would be
    re-embedded after every document update. The \\0 separator keeps
    ("ab", "c") and ("a", "bc") distinct.
    """
    return hashlib.sha256(f"{model}\0{text}".encode("utf-8")).hexdigest()


def _vector_cache_path(root_dir: str | os.PathLike[str], key: str) -> Path:
    return Path(root_dir) / key[:2] / f"{key}.npy"


def load_cached_vector(
    root_dir: str | os.PathLike[str], key: str
) -> np.ndarray | None:
    """Load a single cached embedding vector.

    Returns:
        np.ndarray with shape (1024,) and float32 dtype on hit;
        None on miss or if file is corrupted/invalid.
        Corrupted/invalid content is safely removed to auto-heal cache.
    """
    path = _vector_cache_path(root_dir, key)
    if not path.is_file():
        return None
    try:
        vector = np.load(path, allow_pickle=False)
        if vector.shape != (EMBEDDING_DIM,):
            raise ValueError(f"Unexpected shape: {vector.shape}")
        if not np.issubdtype(vector.dtype, np.floating):
            raise ValueError(f"Unexpected dtype: {vector.dtype}")
        return vector.astype(np.float32, copy=False)
    except (ValueError, OSError):
        # A2: Corrupted or non-vector file -> safely delete to avoid repeated invalid hits
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        return None


def save_cached_vector(
    root_dir: str | os.PathLike[str], key: str, vector: np.ndarray | list[float]
) -> Path:
    """Atomically save a single embedding vector to content-addressed cache."""
    arr = np.asarray(vector, dtype=np.float32)
    if arr.shape != (EMBEDDING_DIM,):
        raise ValueError(
            f"Expected vector shape ({EMBEDDING_DIM},), got {arr.shape}"
        )

    destination = _vector_cache_path(root_dir, key)
    destination.parent.mkdir(parents=True, exist_ok=True)

    # Atomic write: np.save will append .npy if filename doesn't end with .npy
    # So we give temp_file a .npy suffix to prevent unwanted rename by np.save
    temp_file = destination.parent / f".{destination.stem}.{uuid.uuid4().hex}.tmp.npy"
    try:
        np.save(str(temp_file), arr, allow_pickle=False)
        os.replace(temp_file, destination)
    except Exception:
        try:
            temp_file.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return destination


def embed_texts(texts: list[str]) -> list[list[float]]:
    api_key = os.getenv("SILICONFLOW_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing SILICONFLOW_API_KEY. Set it in the environment before running."
        )
    if not texts:
        return []
    vectors: list[list[float]] = []
    for start in range(0, len(texts), BATCH_SIZE):
        vectors.extend(_embed_batch(api_key, texts[start : start + BATCH_SIZE]))
    return vectors


def _embed_batch(api_key: str, texts: list[str]) -> list[list[float]]:
    payload = {
        "model": MODEL,
        "input": texts,
        "encoding_format": "float",
    }
    request = Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    last_error: RuntimeError | None = None
    for attempt in range(EMBED_MAX_ATTEMPTS):
        try:
            with urlopen(request, timeout=60) as response:
                result = json.load(response)
            data = result.get("data")
            if not isinstance(data, list) or len(data) != len(texts):
                # 响应形状错误是永久性问题，不重试
                raise RuntimeError(f"Unexpected embeddings response: {result}")
            vectors = [item["embedding"] for item in data]
            for vec in vectors:
                if len(vec) != EMBEDDING_DIM:
                    raise RuntimeError(f"Expected dim {EMBEDDING_DIM}, got {len(vec)}")
            return vectors
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            if error.code not in RETRYABLE_HTTP_CODES:
                raise RuntimeError(
                    f"SiliconFlow returned HTTP {error.code}: {detail}") from error
            retryable = RuntimeError(
                f"SiliconFlow returned HTTP {error.code} (attempt {attempt + 1}): {detail}"
            )
            retryable.__cause__ = error
            last_error = retryable
        except URLError as error:
            transient = RuntimeError(
                f"Could not reach SiliconFlow API (attempt {attempt + 1}): {error.reason}"
            )
            transient.__cause__ = error
            last_error = transient
        if attempt < EMBED_MAX_ATTEMPTS - 1:
            time.sleep(EMBED_BACKOFF_BASE * (2 ** attempt) * (0.5 + random.random()))
    raise last_error  # type: ignore[misc]


def _row_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class VectorStore:
    def __init__(self, doc_ids: list[str], matrix: np.ndarray, embed_fn=embed_texts) -> None:
        self.doc_ids = list(doc_ids)
        self.matrix = matrix
        self.embed = embed_fn
        self.norm_matrix: np.ndarray | None = None
        if matrix is not None and matrix.shape[0] > 0:
            self.norm_matrix = _row_normalize(matrix)

    @classmethod
    def build(cls, docs: list[tuple[str, str]]) -> "VectorStore":
        if not docs:
            return cls([], np.zeros((0, EMBEDDING_DIM), dtype=np.float32))
        doc_ids = [doc_id for doc_id, _ in docs]
        texts = [text for _, text in docs]
        vectors = embed_texts(texts)
        matrix = np.asarray(vectors, dtype=np.float32)
        return cls(doc_ids, matrix)

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        if not query.strip():
            return []
        if self.norm_matrix is None or not self.doc_ids:
            return []
        query_vec = np.asarray(self.embed([query])[0], dtype=np.float32)
        norm = float(np.linalg.norm(query_vec))
        if norm == 0:
            return []
        query_vec = query_vec / norm
        scores = self.norm_matrix @ query_vec
        k = min(k, len(scores))
        top = np.argsort(-scores)[:k]
        return [(self.doc_ids[i], float(scores[i])) for i in top]

    def save(self, directory: str) -> None:
        if self.matrix is None:
            raise ValueError("cannot save an empty VectorStore")
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "doc_ids.json"), "w", encoding="utf-8") as f:
            json.dump(self.doc_ids, f, ensure_ascii=False)
        np.save(os.path.join(directory, "matrix.npy"), self.matrix)

    @classmethod
    def load(cls, directory: str) -> "VectorStore":
        with open(os.path.join(directory, "doc_ids.json"), "r", encoding="utf-8") as f:
            doc_ids = json.load(f)
        matrix = np.load(os.path.join(directory, "matrix.npy"))
        return cls(doc_ids, matrix)


def build_with_cache(
    docs: list[tuple[str, str]],
    cache_dir: str | os.PathLike[str],
    embed_fn=embed_texts,
) -> tuple[VectorStore, dict[str, int]]:
    """Build a VectorStore using content-addressed cache for chunk vectors.

    - Reuses cached vectors by sha256(model \\0 text)
    - Deduplicates missing texts within the batch to minimize embedding API spend
    - Returns (store, {"total": N, "reused": M, "embedded": K})
    """
    total = len(docs)
    if not docs:
        empty_store = VectorStore([], np.zeros((0, EMBEDDING_DIM), dtype=np.float32), embed_fn=embed_fn)
        return empty_store, {"total": 0, "reused": 0, "embedded": 0}

    keys = [embedding_cache_key(MODEL, text) for _, text in docs]
    vectors_by_key: dict[str, np.ndarray] = {}
    missing_keys_to_text: dict[str, str] = {}
    reused_count = 0

    # 1. 查询缓存并按 key 去重缺失项
    seen_in_batch: set[str] = set()
    for key, (_doc_id, text) in zip(keys, docs):
        if key in vectors_by_key or key in seen_in_batch:
            reused_count += 1
            continue
        cached = load_cached_vector(cache_dir, key)
        if cached is not None:
            vectors_by_key[key] = cached
            reused_count += 1
        elif key not in missing_keys_to_text:
            missing_keys_to_text[key] = text
        seen_in_batch.add(key)

    # 2. 仅对缺失且去重后的文本调用 Embedding API
    if missing_keys_to_text:
        unique_missing_keys = list(missing_keys_to_text.keys())
        unique_missing_texts = [missing_keys_to_text[k] for k in unique_missing_keys]
        raw_vectors = embed_fn(unique_missing_texts)
        if len(raw_vectors) != len(unique_missing_keys):
            raise RuntimeError(
                f"Embedding function returned {len(raw_vectors)} vectors for {len(unique_missing_keys)} texts"
            )

        for key, raw_vec in zip(unique_missing_keys, raw_vectors):
            arr = np.asarray(raw_vec, dtype=np.float32)
            if arr.shape != (EMBEDDING_DIM,):
                raise RuntimeError(f"Expected embedding dim {EMBEDDING_DIM}, got {arr.shape}")
            save_cached_vector(cache_dir, key, arr)
            vectors_by_key[key] = arr

    # 3. 按原始顺序组装矩阵与 doc_ids
    doc_ids = [doc_id for doc_id, _ in docs]
    matrix = np.stack([vectors_by_key[key] for key in keys]).astype(np.float32, copy=False)
    store = VectorStore(doc_ids, matrix, embed_fn=embed_fn)

    stats = {
        "total": total,
        "reused": reused_count,
        "embedded": len(missing_keys_to_text),
    }
    return store, stats


if __name__ == "__main__":
    store = VectorStore.build([("doc1", "北京是中国的首都"), ("doc2", "上海是沿海城市")])
    empty = VectorStore.build([])

    assert empty.search("任何词") == []
    assert store.search("") == []
    assert store.search("   ") == []

    result = store.search("北京")
    print(result)
    assert result and result[0][0] == "doc1"
    scores = [score for _, score in result]
    assert scores == sorted(scores, reverse=True)

    tmp = tempfile.mkdtemp()
    try:
        store.save(tmp)
        loaded = VectorStore.load(tmp)
        assert loaded.search("北京")[0][0] == "doc1"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("VectorStore self-tests passed")
