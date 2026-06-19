from __future__ import annotations

import argparse
from pathlib import Path

from .engine import PlagiarismDetector


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="NLP plagiarism detector using TF-IDF, cosine similarity, and fuzzy matching."
    )
    parser.add_argument("--input-dir", type=str, default="samples", help="Folder containing text files")
    parser.add_argument("--threshold", type=float, default=0.75, help="Combined score threshold")
    parser.add_argument("--csv", type=str, default="plagiarism_report.csv", help="CSV output path")
    parser.add_argument("--json", type=str, default="", help="Optional JSON output path")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    detector = PlagiarismDetector(threshold=args.threshold)
    detector.load_folder(args.input_dir)
    results = detector.detect()

    print(detector.summary())
    print("\nTop matches:")
    for r in results[:10]:
        status = "FLAGGED" if r.flagged else "ok"
        print(
            f"{r.document_a:20s} {r.document_b:20s} "
            f"cosine={r.cosine_similarity:.3f} "
            f"fuzzy={r.fuzzy_similarity:.3f} "
            f"combined={r.combined_score:.3f} "
            f"[{status}]"
        )

    csv_path = Path(args.csv)
    detector.save_reports(csv_path=csv_path, json_path=args.json or None)

    print(f"\nSaved CSV report to: {csv_path.resolve()}")
    if args.json:
        print(f"Saved JSON report to: {Path(args.json).resolve()}")

    return 0
