import math

try:
    from .tokenizer import tokenize_cn
except ImportError:
    from tokenizer import tokenize_cn

K1 = 1.2
B = 0.75


class InvertedIndex:
    def __init__(self) -> None:
        self.postings: dict[str, dict[str, int]] = {}
        self.doc_len: dict[str, int] = {}
        self.N: int = 0
        self.avgdl: float = 0.0
        self.idf: dict[str, float] = {}


class IndexBuilder:
    def __init__(self) -> None:
        self.index = InvertedIndex()

    def _add_document(self, doc_id: str, text: str) -> None:
        tokens = tokenize_cn(text)
        self.index.doc_len[doc_id] = len(tokens)
        tf: dict[str, int] = {}
        for tok in tokens:
            tf[tok] = tf.get(tok, 0) + 1
        for tok, count in tf.items():
            self.index.postings.setdefault(tok, {})[doc_id] = count

    def finalize(self) -> InvertedIndex:
        idx = self.index
        idx.N = len(idx.doc_len)
        if idx.N == 0:
            idx.avgdl = 0.0
            return idx
        idx.avgdl = sum(idx.doc_len.values()) / idx.N
        for term, postings in idx.postings.items():
            df = len(postings)
            idx.idf[term] = math.log((idx.N - df + 0.5) / (df + 0.5) + 1)
        return idx

    def build(self, docs: list[tuple[str, str]]) -> InvertedIndex:
        for doc_id, text in docs:
            self._add_document(doc_id, text)
        return self.finalize()


def bm25_search(index: InvertedIndex, query: str, k: int = 5) -> list[tuple[str, float]]:
    if index.N == 0:
        return []
    tokens = tokenize_cn(query)
    if not tokens:
        return []
    scores: dict[str, float] = {}
    for term in set(tokens):
        if term not in index.postings:
            continue
        for doc_id, tf in index.postings[term].items():
            norm = index.doc_len[doc_id] / index.avgdl
            denom = tf + K1 * (1 - B + B * norm)
            scores[doc_id] = scores.get(doc_id, 0.0) + index.idf[term] * tf * (K1 + 1) / denom
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:k]


if __name__ == "__main__":
    empty = IndexBuilder().build([])
    assert bm25_search(empty, "任何词") == []

    idx = IndexBuilder().build([("doc1", "北京是首都"), ("doc2", "上海是城市")])
    assert bm25_search(idx, "不存在的词xyz") == []
    assert bm25_search(idx, "") == []

    result = bm25_search(idx, "北京首都")
    print(result)
    assert result and result[0][0] == "doc1" and result[0][1] > 0
    print("4 self-tests passed")
