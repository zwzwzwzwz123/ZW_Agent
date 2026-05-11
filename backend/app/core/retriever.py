from app.core.schemas import ChunkView, SearchHit
from app.core.text import cosine_similarity, term_vector, tokenize


class LexicalRetriever:
    """关键词检索器。

    Retriever 是 RAG 中最关键的模块之一。
    如果检索不到正确证据，后面的 LLM 再强，也只能基于错误或缺失上下文回答。

    当前类叫 LexicalRetriever，意思是“按字面词匹配”的检索器。
    未来可以增加 EmbeddingRetriever、HybridRetriever、RerankRetriever，
    然后在同一套接口下比较不同策略的效果。
    """

    def search(self, query: str, chunks: list[ChunkView], top_k: int) -> list[SearchHit]:
        """从所有 chunk 中找出和 query 最相关的 top-k。

        输入：
        - query：用户问题。
        - chunks：知识库中所有可检索片段。
        - top_k：最多返回多少个候选证据。

        输出：
        - SearchHit：包含 chunk、相似度分数、命中的 token。

        学习重点：这里展示的是“召回阶段”。它不是最终回答，
        而是为生成器准备证据。
        """
        query_vector = term_vector(query)
        query_terms = set(tokenize(query))
        hits: list[SearchHit] = []

        for chunk in chunks:
            chunk_vector = term_vector(chunk.text)
            score = cosine_similarity(query_vector, chunk_vector)
            if score <= 0:
                continue
            chunk_terms = set(chunk_vector)
            # matched_terms 是可解释性信息：它能帮助我们判断为什么这个 chunk 被召回。
            # 面试中可以用它说明：我不仅返回答案，还能观察检索过程。
            matched_terms = sorted(query_terms & chunk_terms)
            hits.append(SearchHit(chunk=chunk, score=round(score, 4), matched_terms=matched_terms))

        # 分数越高表示 query 和 chunk 在当前表示空间中越相似。
        # 注意：top_k 不是越大越好。太大可能把噪声也交给生成器。
        return sorted(hits, key=lambda hit: hit.score, reverse=True)[:top_k]
