"""Build cached full-corpus indexes: pickled BM25 + full vector store.

One-time cost: embedding every corpus passage via the bge-m3 API
(batched, checkpointed: an interrupted run resumes, never re-embeds).

Produced caches (data/nomiracl/chinese/):
    bm25_N{N}.pkl      pickled InvertedIndex (zero API)
    vecstore_N{N}/     doc_ids.json + matrix.npy (VectorStore format)

Every later run (demo / eval) loads these files with zero API calls.

Run:  python build_full_index.py
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np

from demo_hybrid_search import CORPUS, DATA_DIR, load_passages
from index import IndexBuilder, VectorStore, embed_texts

EMBED_BATCH = 64          # docs per API call
CHECKPOINT_EVERY = 32     # save partial state every N batches (2048 docs)


def load_all_docs() -> list[tuple[str, str]]:
    t0 = time.perf_counter()
    docs, dropped = load_passages(1 << 30)
    note = f", {dropped} duplicate docids skipped" if dropped else ""
    print(f"corpus: {len(docs)} passages loaded from {CORPUS.name} "
          f"({time.perf_counter() - t0:.1f}s){note}")
    return docs


def build_bm25(docs: list[tuple[str, str]]) -> None:
    path = DATA_DIR / f"bm25_N{len(docs)}.pkl"
    if path.is_file():
        print(f"bm25   : {path.name} already cached, skip")
        return
    t0 = time.perf_counter()
    index = IndexBuilder().build(docs)
    path.write_bytes(pickle.dumps(index))
    print(f"bm25   : built {index.N} docs / {len(index.postings)} terms "
          f"in {time.perf_counter() - t0:.1f}s -> {path.name}")


def build_vector_store(docs: list[tuple[str, str]]) -> None:
    cache_dir = DATA_DIR / f"vecstore_N{len(docs)}"
    matrix_file = cache_dir / "matrix.npy"
    ids_file = cache_dir / "doc_ids.json"
    if matrix_file.is_file() and ids_file.is_file():
        print(f"vector : {cache_dir.name} already cached, skip")
        return
    cache_dir.mkdir(parents=True, exist_ok=True)

    doc_ids = [doc_id for doc_id, _ in docs]
    texts = [text for _, text in docs]

    matrix: np.ndarray | None = None
    start = 0
    partial = cache_dir / "partial.npy"
    if partial.is_file() and ids_file.is_file():
        saved_ids = json.loads(ids_file.read_text(encoding="utf-8"))
        if saved_ids and saved_ids == doc_ids[: len(saved_ids)]:
            matrix = np.load(str(partial))
            start = len(saved_ids)
            print(f"vector : resuming from checkpoint {start}/{len(docs)}")

    t0 = time.perf_counter()
    for batch_start in range(start, len(texts), EMBED_BATCH):
        vectors = np.asarray(
            embed_texts(texts[batch_start : batch_start + EMBED_BATCH]),
            dtype=np.float32,
        )
        matrix = vectors if matrix is None else np.concatenate([matrix, vectors], axis=0)
        done = matrix.shape[0]
        if done % (EMBED_BATCH * CHECKPOINT_EVERY) == 0 or done == len(docs):
            np.save(str(partial), matrix)
            ids_file.write_text(json.dumps(doc_ids[:done], ensure_ascii=False), encoding="utf-8")
            print(f"  checkpoint {done}/{len(docs)} ({time.perf_counter() - t0:.0f}s)")

    np.save(str(matrix_file), matrix)
    ids_file.write_text(json.dumps(doc_ids, ensure_ascii=False), encoding="utf-8")
    partial.unlink(missing_ok=True)
    print(f"vector : embedded {len(docs)} docs in {time.perf_counter() - t0:.1f}s "
          f"-> {cache_dir.name} ({matrix.nbytes / 1e6:.0f} MB)")


def main() -> int:
    docs = load_all_docs()
    build_bm25(docs)
    build_vector_store(docs)
    store = VectorStore.load(str(DATA_DIR / f"vecstore_N{len(docs)}"))
    print(f"verify : loaded back {len(store.doc_ids)} docs, "
          f"dim {store.matrix.shape[1]} ({store.matrix.nbytes / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
