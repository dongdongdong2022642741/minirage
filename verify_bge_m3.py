"""Verify the BAAI/bge-m3 embedding API on SiliconFlow.

Purpose:
    1. Check the endpoint is reachable and returns 200.
    2. Check the model name and that vectors have the expected dimension (1024).
    3. Sanity-check semantics: similar sentences should have high cosine
       similarity, unrelated ones low.

Setup:
    The API key is read from the SILICONFLOW_API_KEY environment variable.
    - Terminal:  $env:SILICONFLOW_API_KEY = "sk-..."  (session only)
    - Permanent: [Environment]::SetEnvironmentVariable("SILICONFLOW_API_KEY", "sk-...", "User")
    - IDE:       set it in the run configuration, or use the permanent option above.

Usage:
    python verify_bge_m3.py
"""


from __future__ import annotations
import json
import os
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

API_URL = "https://api.siliconflow.cn/v1/embeddings"
MODEL = "BAAI/bge-m3"
EXPECTED_DIM = 1024

TEST_TEXTS = ["今天天气很好", "今天阳光明媚", "数据库连接失败"]
SIMILAR_PAIR = (0, 1)   # two sentences about the weather -> high similarity
UNRELATED_PAIR = (0, 2)  # weather vs database failure -> low similarity


def get_embeddings(texts: list[str]) -> dict:
    """Call the SiliconFlow embeddings API and return the parsed JSON."""
    api_key = os.getenv("SILICONFLOW_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing SILICONFLOW_API_KEY. Set it in the environment or run "
            "configuration before running."
        )

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
            return json.load(response)
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"SiliconFlow returned HTTP {error.code}: {detail}") from error
    except URLError as error:
        raise RuntimeError(f"Could not reach SiliconFlow API: {error.reason}") from error


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def main() -> int:
    print(f"Calling {MODEL} via {API_URL} ...")
    result = get_embeddings(TEST_TEXTS)

    vectors = [item["embedding"] for item in result["data"]]
    dims = {len(v) for v in vectors}

    print(f"model        : {result['model']}")
    print(f"num vectors  : {len(vectors)} (expected {len(TEST_TEXTS)})")
    print(f"dimensions   : {dims} (expected {EXPECTED_DIM})")
    print(f"usage        : {result['usage']}")

    if len(vectors) != len(TEST_TEXTS):
        print("FAIL: number of returned vectors does not match input.", file=sys.stderr)
        return 1
    if dims != {EXPECTED_DIM}:
        print(f"FAIL: expected dimension {EXPECTED_DIM}, got {dims}.", file=sys.stderr)
        return 1

    i, j = SIMILAR_PAIR
    sim = cosine_similarity(vectors[i], vectors[j])
    print(f"similarity({TEST_TEXTS[i]!r}, {TEST_TEXTS[j]!r}) = {sim:.3f}")

    i, j = UNRELATED_PAIR
    unrel = cosine_similarity(vectors[i], vectors[j])
    print(f"similarity({TEST_TEXTS[i]!r}, {TEST_TEXTS[j]!r}) = {unrel:.3f}")
    print(f"similarity gap = {sim - unrel:.3f}")

    if sim > unrel:
        print("OK: semantics check passed (similar pair ranks above unrelated pair).")
        return 0
    print("FAIL: similar pair did not rank above unrelated pair.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
