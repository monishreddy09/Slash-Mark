from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .utils import join_tokens, list_documents, read_text_file, tokenize


@dataclass
class DetectionResult:
    document_a: str
    document_b: str
    cosine_similarity: float
    fuzzy_similarity: float
    combined_score: float
    flagged: bool

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class PlagiarismDetector:
    """
    Baseline plagiarism detector:
    - text normalization
    - n-gram TF-IDF features
    - cosine similarity
    - fuzzy matching
    """

    def __init__(
        self,
        word_ngram_range: Tuple[int, int] = (1, 2),
        char_ngram_range: Tuple[int, int] = (3, 5),
        cosine_weight: float = 0.7,
        fuzzy_weight: float = 0.3,
        threshold: float = 0.75,
    ) -> None:
        self.word_ngram_range = word_ngram_range
        self.char_ngram_range = char_ngram_range
        self.cosine_weight = cosine_weight
        self.fuzzy_weight = fuzzy_weight
        self.threshold = threshold

        self.vectorizer_word = TfidfVectorizer(
            analyzer="word",
            ngram_range=word_ngram_range,
            min_df=1,
            lowercase=False,
        )
        self.vectorizer_char = TfidfVectorizer(
            analyzer="char",
            ngram_range=char_ngram_range,
            min_df=1,
            lowercase=False,
        )

        self.documents: List[Path] = []
        self.raw_texts: List[str] = []
        self.cleaned_texts: List[str] = []

    def load_folder(self, folder: str | Path, extensions: Iterable[str] = (".txt", ".md", ".text")) -> None:
        self.documents = list_documents(folder, extensions=extensions)
        if not self.documents:
            raise FileNotFoundError(f"No documents found in {folder!s}")
        self.raw_texts = [read_text_file(p) for p in self.documents]
        self.cleaned_texts = [join_tokens(tokenize(t)) for t in self.raw_texts]

    def _build_feature_matrix(self) -> np.ndarray:
        """
        Combine word and character TF-IDF matrices to improve robustness.
        """
        word_matrix = self.vectorizer_word.fit_transform(self.cleaned_texts)
        char_matrix = self.vectorizer_char.fit_transform(self.raw_texts)
        # concatenate sparse matrices horizontally
        from scipy.sparse import hstack
        return hstack([word_matrix, char_matrix])

    def _pair_score(self, idx_a: int, idx_b: int, matrix) -> DetectionResult:
        vec_a = matrix[idx_a]
        vec_b = matrix[idx_b]
        cosine = float(cosine_similarity(vec_a, vec_b)[0][0])

        # rapidfuzz scores are 0..100
        text_a = self.raw_texts[idx_a]
        text_b = self.raw_texts[idx_b]
        fuzzy = max(
            fuzz.token_sort_ratio(text_a, text_b),
            fuzz.token_set_ratio(text_a, text_b),
            fuzz.partial_ratio(text_a, text_b),
        ) / 100.0

        combined = (self.cosine_weight * cosine) + (self.fuzzy_weight * fuzzy)
        flagged = combined >= self.threshold

        return DetectionResult(
            document_a=self.documents[idx_a].name,
            document_b=self.documents[idx_b].name,
            cosine_similarity=round(cosine, 6),
            fuzzy_similarity=round(fuzzy, 6),
            combined_score=round(combined, 6),
            flagged=flagged,
        )

    def detect(self) -> List[DetectionResult]:
        if not self.documents:
            raise RuntimeError("No documents loaded. Call load_folder() first.")

        matrix = self._build_feature_matrix()
        results: List[DetectionResult] = []

        for i in range(len(self.documents)):
            for j in range(i + 1, len(self.documents)):
                results.append(self._pair_score(i, j, matrix))

        results.sort(key=lambda r: r.combined_score, reverse=True)
        return results

    def detect_as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([r.to_dict() for r in self.detect()])

    def save_reports(
        self,
        csv_path: Optional[str | Path] = None,
        json_path: Optional[str | Path] = None,
    ) -> None:
        df = self.detect_as_dataframe()
        if csv_path:
            df.to_csv(csv_path, index=False)
        if json_path:
            df.to_json(json_path, orient="records", indent=2)

    def summary(self) -> str:
        results = self.detect()
        flagged = [r for r in results if r.flagged]
        lines = [
            f"Documents scanned: {len(self.documents)}",
            f"Pairs evaluated: {len(results)}",
            f"Flagged pairs: {len(flagged)}",
        ]
        if flagged:
            top = flagged[0]
            lines.append(
                f"Highest risk: {top.document_a} vs {top.document_b} "
                f"(score={top.combined_score:.3f})"
            )
        return "\n".join(lines)
