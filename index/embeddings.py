"""VectorStore: document embedding matrix + numpy cosine search.

Contract with Searcher:
    search(query, k) -> list[(doc_id: str, score: float)] sorted desc.
    Empty query or empty store -> [].
    API errors propagate (never swallowed).
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

API_URL = "https://api.siliconflow.cn/v1/embeddings"
MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024
BATCH_SIZE = 64


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
    try:
        with urlopen(request, timeout=60) as response:
            result = json.load(response)
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"SiliconFlow returned HTTP {error.code}: {detail}") from error
    except URLError as error:
        raise RuntimeError(f"Could not reach SiliconFlow API: {error.reason}") from error

    data = result.get("data")
    if not isinstance(data, list) or len(data) != len(texts):
        raise RuntimeError(f"Unexpected embeddings response: {result}")
    vectors = [item["embedding"] for item in data]
    for vec in vectors:
        if len(vec) != EMBEDDING_DIM:
            raise RuntimeError(f"Expected dim {EMBEDDING_DIM}, got {len(vec)}")
    return vectors


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
