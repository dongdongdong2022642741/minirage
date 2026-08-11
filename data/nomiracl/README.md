# NoMIRACL Chinese Subset

Source: https://huggingface.co/datasets/miracl/nomiracl

Downloaded from the `data/chinese` directory on 2026-08-07. The GitHub
repository contains evaluation code; the dataset files are hosted on
Hugging Face.

## Files

- `chinese/corpus.jsonl.gz`: 37,599 passages with `docid`, `title`, and `text`
- `chinese/topics/*.tsv`: query ID and query text
- `chinese/qrels/*.tsv`: query ID, `Q0`, document ID, and relevance label

Both `dev` and `test` splits include `relevant` and `non_relevant` subsets.
See the upstream dataset page and repository for licensing and citation.
