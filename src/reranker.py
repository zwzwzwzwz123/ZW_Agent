from __future__ import annotations

import math

from sentence_transformers import CrossEncoder

from src.embeddings import cosine_similarity
from src.embedding_factory import build_embeddings
from src.models import RetrievedChunk
from src.config import AppConfig


def minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if math.isclose(low, high):
        return [1.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


class Reranker:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.cross_encoder: CrossEncoder | None = None
        if config.reranker_mode == "cross_encoder":
            self.cross_encoder = CrossEncoder(config.reranker_model, max_length=512)
        self.embeddings = build_embeddings(config) if config.reranker_mode == "mmr" else None

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if not chunks:
            return []
        if self.cross_encoder is not None:
            pairs = [(query, chunk.document.page_content) for chunk in chunks]
            raw_scores = [float(score) for score in self.cross_encoder.predict(pairs)]
            normalized = minmax(raw_scores)
            for chunk, score in zip(chunks, normalized):
                chunk.rerank_score = score
                chunk.reasons.append("cross_encoder_relevance")
            return sorted(chunks, key=lambda item: item.rerank_score, reverse=True)[:top_k]
        return self._mmr(query, chunks, top_k)

    def _mmr(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if self.embeddings is None:
            return sorted(chunks, key=lambda item: item.fused_score, reverse=True)[:top_k]

        query_vector = self.embeddings.embed_query(query)
        doc_vectors = [self.embeddings.embed_query(chunk.document.page_content) for chunk in chunks]
        relevance = [cosine_similarity(query_vector, vector) for vector in doc_vectors]
        selected: list[int] = []
        candidate_indices = list(range(len(chunks)))
        lambda_mult = 0.72

        while candidate_indices and len(selected) < top_k:
            best_idx = candidate_indices[0]
            best_score = -float("inf")
            for idx in candidate_indices:
                diversity_penalty = 0.0
                if selected:
                    diversity_penalty = max(cosine_similarity(doc_vectors[idx], doc_vectors[j]) for j in selected)
                score = lambda_mult * relevance[idx] - (1.0 - lambda_mult) * diversity_penalty
                if score > best_score:
                    best_idx = idx
                    best_score = score
            selected.append(best_idx)
            candidate_indices.remove(best_idx)

        selected_chunks = []
        normalized = minmax([relevance[idx] for idx in selected])
        for idx, score in zip(selected, normalized):
            chunks[idx].rerank_score = score
            chunks[idx].reasons.append("mmr_relevance_diversity")
            selected_chunks.append(chunks[idx])
        return selected_chunks
