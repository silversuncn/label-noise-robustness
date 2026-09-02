"""Data loading, sampling, and TF-IDF utilities for the public pipeline."""

from __future__ import annotations

import csv
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9']+")


@dataclass(frozen=True)
class TextRecord:
    text: str
    label: int


@dataclass(frozen=True)
class DatasetBundle:
    name: str
    train: list[TextRecord]
    test: list[TextRecord]
    num_classes: int


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class TfidfVectorizerLite:
    def __init__(self, max_features: int = 3000) -> None:
        self.max_features = max_features
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray | None = None

    def fit(self, texts: list[str]) -> "TfidfVectorizerLite":
        doc_freq: Counter[str] = Counter()
        term_freq: Counter[str] = Counter()
        for text in texts:
            tokens = tokenize(text)
            term_freq.update(tokens)
            doc_freq.update(set(tokens))
        terms = sorted(term_freq, key=lambda term: (-term_freq[term], term))
        self.vocab = {term: idx for idx, term in enumerate(terms[: self.max_features])}
        n_docs = max(len(texts), 1)
        idf = np.ones(len(self.vocab), dtype=np.float64)
        for term, idx in self.vocab.items():
            idf[idx] = math.log((1.0 + n_docs) / (1.0 + doc_freq[term])) + 1.0
        self.idf = idf
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        if self.idf is None:
            raise RuntimeError("vectorizer is not fitted")
        x = np.zeros((len(texts), len(self.vocab)), dtype=np.float64)
        for row_idx, text in enumerate(texts):
            counts: Counter[int] = Counter()
            for token in tokenize(text):
                idx = self.vocab.get(token)
                if idx is not None:
                    counts[idx] += 1
            total = sum(counts.values())
            if total == 0:
                continue
            for idx, count in counts.items():
                x[row_idx, idx] = (count / total) * self.idf[idx]
        norms = np.linalg.norm(x, axis=1)
        norms[norms == 0.0] = 1.0
        return x / norms[:, None]

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        return self.fit(texts).transform(texts)


def load_csv_records(path: Path, text_column: str = "text", label_column: str = "label") -> list[TextRecord]:
    records: list[TextRecord] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            text = row[text_column].strip()
            if text:
                records.append(TextRecord(text=text, label=int(row[label_column])))
    return records


def load_dataset_bundle(
    data_root: Path,
    dataset: str,
    train_name: str | None = None,
    test_name: str | None = None,
    text_column: str = "text",
    label_column: str = "label",
) -> DatasetBundle:
    train_path = data_root / (train_name or f"{dataset}_train.csv")
    test_path = data_root / (test_name or f"{dataset}_test.csv")
    train = load_csv_records(train_path, text_column=text_column, label_column=label_column)
    test = load_csv_records(test_path, text_column=text_column, label_column=label_column)
    labels = {record.label for record in train} | {record.label for record in test}
    return DatasetBundle(name=dataset, train=train, test=test, num_classes=max(labels) + 1 if labels else 0)


def sample_per_class(records: list[TextRecord], budget_per_class: int, seed: int) -> tuple[list[TextRecord], dict[int, int]]:
    by_label: dict[int, list[TextRecord]] = defaultdict(list)
    for record in records:
        by_label[record.label].append(record)
    rng = random.Random(seed)
    sampled: list[TextRecord] = []
    counts: dict[int, int] = {}
    for label in sorted(by_label):
        rows = list(by_label[label])
        rng.shuffle(rows)
        selected = rows[: min(budget_per_class, len(rows))]
        sampled.extend(selected)
        counts[label] = len(selected)
    rng.shuffle(sampled)
    return sampled, counts


def cap_test_per_class(records: list[TextRecord], cap_per_class: int, seed: int) -> list[TextRecord]:
    sampled, _ = sample_per_class(records, cap_per_class, seed)
    return sampled
