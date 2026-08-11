"""DeepSeek chat API smoke test for the MiniRAG foundation.

This file intentionally tests only the non-streaming chat call. A chat model
generates an answer token by token; it does not turn text into a stable vector
space that can be indexed and compared for semantic similarity. For retrieval,
use a separate embedding model, such as BGE or sentence-transformers.

Setup (PowerShell):
    $env:DEEPSEEK_API_KEY = "your-api-key"
    python deepseek_chat.py "请用一句话解释什么是向量检索"

The API key is read from the environment and is never stored in this file.
The implementation uses only the Python standard library, so no package
installation is required for this first smoke test.
"""

from __future__ import annotations

import json
import os
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


API_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_MODEL = "deepseek-chat"


def ask_deepseek(question: str, temperature: float = 0.1) -> str:
    """Send one question to DeepSeek and return the assistant's reply.

    temperature is passed through to the API; low values reduce answer
    variance (the API default of 1.0 makes "资料不足" refusals flip
    between runs for the same evidence).
    """
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing DEEPSEEK_API_KEY. Set it in the environment before running."
        )

    if not question.strip():
        raise ValueError("question must not be empty")

    payload = {
        "model": os.getenv("DEEPSEEK_MODEL", DEFAULT_MODEL),
        "messages": [{"role": "user", "content": question}],
        "temperature": temperature,
        "stream": False,
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
        raise RuntimeError(f"DeepSeek API returned HTTP {error.code}: {detail}") from error
    except URLError as error:
        raise RuntimeError(f"Could not reach DeepSeek API: {error.reason}") from error

    try:
        return result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError(f"Unexpected DeepSeek response: {result}") from error


def main() -> int:
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        question = "请用一句话解释什么是向量检索。"

    try:
        reply = ask_deepseek(question)
    except (RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Question: {question}")
    print(f"DeepSeek: {reply}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
