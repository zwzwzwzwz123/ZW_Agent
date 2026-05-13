from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from src.embeddings import tokenize


@dataclass
class KeywordHit:
    document: Document
    score: float


class BM25Index:
    def __init__(self, documents: list[Document], bm25: BM25Okapi) -> None:
        self.documents = documents
        self.bm25 = bm25

    @classmethod
    def from_documents(cls, documents: list[Document]) -> "BM25Index":
        corpus = [tokenize(doc.page_content) for doc in documents]
        return cls(documents=documents, bm25=BM25Okapi(corpus))

    def search(self, query: str, k: int) -> list[KeywordHit]:
        if not self.documents:
            return []
        scores = np.asarray(self.bm25.get_scores(tokenize(query)), dtype=np.float32)
        top_indices = np.argsort(scores)[::-1][:k]
        return [KeywordHit(self.documents[int(i)], float(scores[int(i)])) for i in top_indices if scores[int(i)] > 0]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as file:
            pickle.dump({"documents": self.documents, "bm25": self.bm25}, file)

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        with path.open("rb") as file:
            payload = pickle.load(file)
        return cls(documents=payload["documents"], bm25=payload["bm25"])
