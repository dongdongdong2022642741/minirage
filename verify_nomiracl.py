"""Sanity-check the NoMIRACL Chinese subset before building on it.

Checks:
    1. corpus.jsonl.gz is valid gzip + JSON, and prints the first 100 chars
       of records #1, #100 and #1000.
    2. Total corpus record count (README claims 37,599).
    3. Number of queries with at least one relevance label in
       dev.relevant qrels, and the number of label rows.

If any check fails, stop: every downstream evaluation would be garbage.

Usage:
    python verify_nomiracl.py
"""

from __future__ import annotations

import csv
import gzip
import json
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data" / "nomiracl" / "chinese"
CORPUS = DATA_DIR / "corpus.jsonl.gz"
QRELS = DATA_DIR / "qrels" / "dev.relevant.tsv"
EXPECTED_CORPUS_RECORDS = 37_599


def read_corpus():
    """Yield decoded JSON records from the gzipped corpus file."""
    with gzip.open(CORPUS, "rt", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"corpus line {line_number} is not valid JSON: {error}") from error


def main() -> int:
    if not CORPUS.is_file():
        print(f"FAIL: missing corpus file: {CORPUS}", file=sys.stderr)
        return 1

    # Check 1: spot-check records at the 1st, 100th and 1000th line.
    spot = {1, 100, 1000}
    found: dict[int, dict] = {}
    total = 0
    for line_number, record in read_corpus():
        total += 1
        if line_number in spot:
            found[line_number] = record
        if len(found) == len(spot):
            break

    print(f"corpus file       : {CORPUS}")
    for line_number in sorted(spot):
        record = found.get(line_number)
        if record is None:
            print(f"  record #{line_number}: MISSING", file=sys.stderr)
            return 1
        for field in ("docid", "title", "text"):
            if field not in record:
                print(f"  record #{line_number}: missing field {field!r}", file=sys.stderr)
                return 1
        print(f"  record #{line_number}: docid={record['docid']!r} title={record['title'][:30]!r}")
        print(f"                text: {record['text'][:100]!r}")

    # Check 2: total corpus count.
    # Re-read fully because the first pass stopped early.
    total = sum(1 for _ in read_corpus())
    print(f"corpus total      : {total:,} records")
    if total != EXPECTED_CORPUS_RECORDS:
        print(
            f"FAIL: expected {EXPECTED_CORPUS_RECORDS:,} records, got {total:,}",
            file=sys.stderr,
        )
        return 1

    # Check 3: queries with at least one label in dev.relevant qrels.
    if not QRELS.is_file():
        print(f"FAIL: missing qrels file: {QRELS}", file=sys.stderr)
        return 1
    queries: set[str] = set()
    rows = 0
    labels: set[str] = set()
    with QRELS.open(encoding="utf-8", newline="") as file:
        for row in csv.reader(file, delimiter="\t"):
            if len(row) != 4:
                print(f"FAIL: qrels row has {len(row)} columns, expected 4: {row!r}", file=sys.stderr)
                return 1
            query_id, _, doc_id, label = row
            queries.add(query_id)
            labels.add(label)
            rows += 1
            if not doc_id:
                print(f"FAIL: qrels row has empty doc_id: {row!r}", file=sys.stderr)
                return 1
            if label not in ("0", "1"):
                print(f"FAIL: unexpected relevance label {label!r}", file=sys.stderr)
                return 1

    print(f"dev.relevant qrels: {rows:,} rows, {len(queries):,} queries with labels, labels={sorted(labels)}")
    print("OK: data sanity checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
