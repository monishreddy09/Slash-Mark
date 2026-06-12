
"""
Sentiment Analysis Project
Supports positive / negative / neutral classification for social posts, product reviews, and restaurant reviews.

Expected dataset:
- a text column (for example: Review, Text, Comment, Tweet)
- a label column (for example: Sentiment, Label, Liked, Rating)

The script can also infer labels from common formats and includes a small fallback demo dataset.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional, Tuple, Dict

import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# -----------------------------
# Optional NLTK setup
# -----------------------------
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    _NLTK_AVAILABLE = True
except Exception:
    _NLTK_AVAILABLE = False

_LEMMATIZER = None
_STOPWORDS = None

def _ensure_nltk():
    global _LEMMATIZER, _STOPWORDS
    if not _NLTK_AVAILABLE:
        return

    # Use local NLTK resources if available; otherwise fall back gracefully.
    try:
        _STOPWORDS = set(stopwords.words("english"))
    except Exception:
        _STOPWORDS = set(ENGLISH_STOP_WORDS)

    try:
        _LEMMATIZER = WordNetLemmatizer()
        _ = _LEMMATIZER.lemmatize("tests")
    except Exception:
        _LEMMATIZER = None

    if _STOPWORDS is None:
        _STOPWORDS = set(ENGLISH_STOP_WORDS)

    # Keep negations for sentiment
    for w in {"no", "nor", "not", "never"}:
        _STOPWORDS.discard(w)

def _get_stopwords():
    if _STOPWORDS is None:
        _ensure_nltk()
    return _STOPWORDS or set(ENGLISH_STOP_WORDS)

def _get_lemmatizer():
    if _LEMMATIZER is None:
        _ensure_nltk()
    return _LEMMATIZER

def preprocess_text(text: object) -> str:
    """
    Clean text with:
    - lowercasing
    - regex tokenization
    - stopword removal
    - lemmatization (when NLTK wordnet is available)
    """
    text = "" if text is None else str(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+", " ", text)
    text = re.sub(r"[^a-z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = re.findall(r"\b[a-z']+\b", text)
    stop_words = _get_stopwords()
    lemmatizer = _get_lemmatizer()

    cleaned = []
    for tok in tokens:
        if tok in stop_words:
            continue
        if lemmatizer is not None:
            try:
                tok = lemmatizer.lemmatize(tok)
            except Exception:
                pass
        cleaned.append(tok)

    return " ".join(cleaned)

def normalize_label(value: object) -> Optional[str]:
    """
    Convert labels to: negative / neutral / positive
    Handles:
    - strings like pos/neg/neutral
    - binary labels 0/1
    - ratings 1-5 (1-2 negative, 3 neutral, 4-5 positive)
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None

    # Numeric labels / ratings
    if isinstance(value, (int, np.integer, float, np.floating)):
        v = float(value)
        if v in {0.0, 1.0}:
            return "positive" if v == 1.0 else "negative"
        if v <= 2:
            return "negative"
        if v == 3:
            return "neutral"
        return "positive"

    s = str(value).strip().lower()

    # Common string labels
    mapping = {
        "pos": "positive",
        "positive": "positive",
        "neg": "negative",
        "negative": "negative",
        "neu": "neutral",
        "neutral": "neutral",
        "0": "negative",
        "1": "positive",
    }
    if s in mapping:
        return mapping[s]

    # Rating-like strings
    try:
        v = float(s)
        if v <= 2:
            return "negative"
        if v == 3:
            return "neutral"
        return "positive"
    except Exception:
        pass

    return None

def infer_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Try to infer the text and label columns from a dataframe.
    """
    lower_cols = {c.lower(): c for c in df.columns}

    text_candidates = ["review", "text", "comment", "tweet", "sentence", "content", "message"]
    label_candidates = ["sentiment", "label", "liked", "rating", "polarity", "class", "target"]

    text_col = None
    label_col = None

    for cand in text_candidates:
        if cand in lower_cols:
            text_col = lower_cols[cand]
            break
    for cand in label_candidates:
        if cand in lower_cols:
            label_col = lower_cols[cand]
            break

    if text_col is None:
        obj_cols = [c for c in df.columns if df[c].dtype == "object"]
        text_col = obj_cols[0] if obj_cols else df.columns[0]

    if label_col is None:
        non_text_cols = [c for c in df.columns if c != text_col]
        if not non_text_cols:
            raise ValueError("Could not infer label column.")
        label_col = non_text_cols[0]

    return text_col, label_col

def load_dataset(path: Optional[str] = None) -> pd.DataFrame:
    """
    Load a CSV/TSV dataset or fallback to a tiny built-in sample.
    """
    if path and Path(path).exists():
        p = Path(path)
        if p.suffix.lower() in {".tsv", ".txt"}:
            df = pd.read_csv(p, sep="\t")
        else:
            df = pd.read_csv(p)
        return df

    # Fallback sample dataset
    sample = [
        ("I loved the food and the service was excellent.", "positive"),
        ("This was the worst restaurant experience ever.", "negative"),
        ("The product works as expected.", "neutral"),
        ("Amazing quality and super fast delivery!", "positive"),
        ("The update did not fix the issue.", "negative"),
        ("It is okay, nothing special.", "neutral"),
        ("Great value for the money.", "positive"),
        ("I would not recommend this to anyone.", "negative"),
        ("Delivery was on time and the packaging was neat.", "positive"),
        ("The app is usable but has room for improvement.", "neutral"),
        ("Absolutely terrible, the order arrived cold.", "negative"),
        ("The movie was fine, average overall.", "neutral"),
        ("I am very happy with this purchase.", "positive"),
        ("The support team never replied to my message.", "negative"),
        ("The review is mixed; some parts were good, some bad.", "neutral"),
    ]
    return pd.DataFrame(sample, columns=["text", "label"])

def build_dataset(
    df: pd.DataFrame,
    text_col: Optional[str] = None,
    label_col: Optional[str] = None,
) -> pd.DataFrame:
    if text_col is None or label_col is None:
        text_col, label_col = infer_columns(df)

    out = df[[text_col, label_col]].copy()
    out.columns = ["text", "label"]
    out["label"] = out["label"].apply(normalize_label)

    # Handle ratings or messy labels; drop unmapped rows
    out["text"] = out["text"].fillna("").astype(str)
    out = out.dropna(subset=["label"])
    out = out[out["text"].str.strip().ne("")]

    # remove empty rows
    out = out[out["text"].str.len() > 0]
    return out.reset_index(drop=True)

def train_models(
    data: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    max_features: int = 10000,
) -> Dict[str, object]:
    """
    Train Logistic Regression and Linear SVM on TF-IDF features.
    Returns a dictionary with models and metrics.
    """
    # Preprocess text
    data = data.copy()
    data["clean_text"] = data["text"].apply(preprocess_text)

    X_train, X_test, y_train, y_test = train_test_split(
        data["clean_text"],
        data["label"],
        test_size=test_size,
        random_state=random_state,
        stratify=data["label"] if data["label"].nunique() > 1 else None,
    )

    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
    )
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)

    models = {
        "logreg": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "svm": LinearSVC(class_weight="balanced"),
    }

    results = {
        "vectorizer": tfidf,
        "X_test": X_test,
        "y_test": y_test,
        "clean_test": X_test,
        "reports": {},
        "confusion_matrices": {},
        "trained_models": {},
    }

    for name, model in models.items():
        model.fit(X_train_vec, y_train)
        preds = model.predict(X_test_vec)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")
        results["reports"][name] = {
            "accuracy": acc,
            "f1_macro": f1,
            "classification_report": classification_report(y_test, preds, zero_division=0),
        }
        results["confusion_matrices"][name] = confusion_matrix(y_test, preds, labels=sorted(data["label"].unique()))
        results["trained_models"][name] = model

    # pick best by macro F1
    best_name = max(results["reports"], key=lambda n: results["reports"][n]["f1_macro"])
    results["best_name"] = best_name
    results["best_model"] = results["trained_models"][best_name]
    results["labels"] = sorted(data["label"].unique())
    return results

def predict_sentiment(text: str, vectorizer: TfidfVectorizer, model) -> str:
    clean = preprocess_text(text)
    vec = vectorizer.transform([clean])
    return str(model.predict(vec)[0])

def save_artifacts(results: Dict[str, object], out_dir: str = "artifacts") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(results["vectorizer"], out / "tfidf_vectorizer.joblib")
    joblib.dump(results["best_model"], out / f"{results['best_name']}_model.joblib")

def main():
    # Change this path to your dataset:
    # Example: "Restaurant_Reviews.tsv"
    data_path = os.environ.get("SENTIMENT_DATA_PATH", "")

    df_raw = load_dataset(data_path if data_path else None)
    df = build_dataset(df_raw)

    print("Dataset shape:", df.shape)
    print(df.head(), "\n")

    results = train_models(df)

    print("Best model:", results["best_name"])
    for name, metrics in results["reports"].items():
        print(f"\n=== {name} ===")
        print("Accuracy:", round(metrics["accuracy"], 4))
        print("Macro F1:", round(metrics["f1_macro"], 4))
        print(metrics["classification_report"])

    save_artifacts(results)

    demo_texts = [
        "The food was delicious and the staff were kind.",
        "The app keeps crashing and I am frustrated.",
        "It is fine, but nothing special.",
    ]
    for t in demo_texts:
        pred = predict_sentiment(t, results["vectorizer"], results["best_model"])
        print(f"\nText: {t}\nPrediction: {pred}")

if __name__ == "__main__":
    main()
