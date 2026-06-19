from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List

try:
    from nltk.corpus import stopwords
    _STOPWORDS = set(stopwords.words("english"))
except Exception:
    _STOPWORDS = {
        "a","an","and","are","as","at","be","but","by","for","if","in","into","is",
        "it","no","not","of","on","or","such","that","the","their","then","there",
        "these","they","this","to","was","will","with","we","you","your","i","me",
        "my","our","from","or","so","than","too","very","can","could","would","should",
        "about","all","any","do","does","did","done","have","has","had","he","she","them",
        "what","when","where","who","whom","which","why","how"
    }

_word_re = re.compile(r"[a-z0-9]+", re.IGNORECASE)

def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def list_documents(folder: str | Path, extensions: Iterable[str] = (".txt", ".md", ".text")) -> List[Path]:
    folder = Path(folder)
    docs = []
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in set(extensions):
            docs.append(p)
    return docs

def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.replace("\n", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text: str, remove_stopwords: bool = True) -> List[str]:
    tokens = _word_re.findall(normalize_text(text))
    if remove_stopwords:
        tokens = [t for t in tokens if t not in _STOPWORDS]
    return tokens

def join_tokens(tokens: Iterable[str]) -> str:
    return " ".join(tokens)
