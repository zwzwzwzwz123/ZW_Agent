from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.config import AppConfig, BM25_PATH, SAMPLE_DOCS_DIR, VECTOR_STORE_DIR
from src.document_loader import discover_files, load_documents, split_documents
from src.embedding_factory import build_embeddings
from src.keyword_index import BM25Index
from src.models import RetrievedChunk
from src.reranker import Reranker, minmax


class KnowledgeBase:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.embeddings: Embeddings = build_embeddings(config)
        self.vector_store: FAISS | None = None
        self.keyword_index: BM25Index | None = None
        self.reranker: Reranker | None = None

    def build(self, doc_dir: Path = SAMPLE_DOCS_DIR, reset: bool = True) -> dict[str, Any]:
        files = discover_files(doc_dir)
        if not files:
            raise ValueError(f"No supported documents found in {doc_dir}")
        docs = load_documents(files)
        chunks = split_documents(docs, self.config.chunk_size, self.config.chunk_overlap)
        if reset and VECTOR_STORE_DIR.exists():
            shutil.rmtree(VECTOR_STORE_DIR)
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        self.vector_store.save_local(str(VECTOR_STORE_DIR))
        self.keyword_index = BM25Index.from_documents(chunks)
        self.keyword_index.save(BM25_PATH)
        self.reranker = None
        return {
            "files": len(files),
            "documents": len(docs),
            "chunks": len(chunks),
            "sources": sorted({doc.metadata.get("source", "unknown") for doc in docs}),
        }

    def load(self) -> None:
        if not VECTOR_STORE_DIR.exists() or not BM25_PATH.exists():
            self.build()
            return
        self.vector_store = FAISS.load_local(
            str(VECTOR_STORE_DIR),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        self.keyword_index = BM25Index.load(BM25_PATH)

    def search(self, query: str) -> list[RetrievedChunk]:
        self._ensure_loaded()
        assert self.vector_store is not None
        assert self.keyword_index is not None

        vector_hits = self.vector_store.similarity_search_with_score(query, k=self.config.top_k)
        keyword_hits = self.keyword_index.search(query, k=self.config.top_k)

        by_id: dict[str, RetrievedChunk] = {}
        vector_scores = [1.0 / (1.0 + float(score)) for _, score in vector_hits]
        keyword_scores = [hit.score for hit in keyword_hits]
        normalized_vector = minmax(vector_scores)
        normalized_keyword = minmax(keyword_scores)

        for (doc, _raw_score), score in zip(vector_hits, normalized_vector):
            chunk = by_id.setdefault(self._doc_id(doc), RetrievedChunk(document=doc))
            chunk.vector_score = max(chunk.vector_score, score)
            chunk.reasons.append("dense_vector")

        for hit, score in zip(keyword_hits, normalized_keyword):
            chunk = by_id.setdefault(self._doc_id(hit.document), RetrievedChunk(document=hit.document))
            chunk.keyword_score = max(chunk.keyword_score, score)
            chunk.reasons.append("bm25_keyword")

        chunks = list(by_id.values())
        for chunk in chunks:
            chunk.fused_score = (
                self.config.vector_weight * chunk.vector_score
                + self.config.keyword_weight * chunk.keyword_score
            )

        chunks.sort(key=lambda item: item.fused_score, reverse=True)
        return chunks[: max(self.config.top_k, self.config.rerank_top_k)]

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        chunks = self.search(query)
        if self.reranker is None:
            self.reranker = Reranker(self.config)
        return self.reranker.rerank(query, chunks, self.config.rerank_top_k)

    def corpus_summary(self) -> dict[str, Any]:
        self._ensure_loaded()
        assert self.keyword_index is not None
        docs = self.keyword_index.documents
        sources: dict[str, int] = {}
        for doc in docs:
            source = str(doc.metadata.get("source", "unknown"))
            sources[source] = sources.get(source, 0) + 1
        return {
            "chunks": len(docs),
            "sources": sources,
            "avg_chars": round(sum(len(doc.page_content) for doc in docs) / max(len(docs), 1), 1),
        }

    def _ensure_loaded(self) -> None:
        if self.vector_store is None or self.keyword_index is None:
            self.load()

    @staticmethod
    def _doc_id(doc: Document) -> str:
        return str(doc.metadata.get("chunk_id") or f"{doc.metadata.get('source')}::{hash(doc.page_content)}")
