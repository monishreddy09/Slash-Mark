# AI Plagiarism Detector

A lightweight NLP pipeline for detecting duplicated or highly similar text across documents.

## Features
- Text normalization and tokenization
- Word and character n-gram TF-IDF features
- Cosine similarity scoring
- Fuzzy matching with RapidFuzz
- Plagiarism thresholding and human-readable reports
- CSV and JSON export

## Setup

```bash
pip install -r requirements.txt
```

## Run

Scan all `.txt` files in a folder:

```bash
python app.py --input-dir samples --threshold 0.75 --output report.csv
```

Optional JSON report:

```bash
python app.py --input-dir samples --threshold 0.75 --output report.csv --json report.json
```

## How it works

Each document is converted into a sparse TF-IDF vector using word and character n-grams.
For every document pair, the detector computes:

- cosine similarity
- fuzzy token similarity
- a weighted final score

Pairs above the threshold are flagged as potential plagiarism.

## Output fields

- `document_a`
- `document_b`
- `cosine_similarity`
- `fuzzy_similarity`
- `combined_score`
- `flagged`

## Notes

This is a strong baseline plagiarism detection pipeline, not a legal verdict.
It works best on essays, articles, blog posts, and other short-to-medium text documents.
