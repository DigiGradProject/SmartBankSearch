# NBE Scrape Cleaner

Transforms `nbe_complete_scrape.zip` into RAG-ready documents.

## Run

```bash
cd nbe-scrape-cleaner
python clean_pipeline.py --zip ../nbe_complete_scrape.zip
```

Outputs:
- `output/documents.jsonl` — one cleaned document per line
- `report.md` — size reduction + near-duplicate review list

## Tests

```bash
python -m pytest tests -q
```

## Why JSONL?

One JSON object per line avoids Windows path-length failures from long ProductDetails folder names, and is easier for downstream ingestion to stream.
