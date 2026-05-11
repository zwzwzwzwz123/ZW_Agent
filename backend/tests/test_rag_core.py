from app.core.retriever import LexicalRetriever
from app.core.schemas import ChunkView
from app.core.text import chunk_text, tokenize


def test_tokenize_handles_chinese_and_english_terms() -> None:
    tokens = tokenize("RAG 可以降低 hallucination")
    assert "rag" in tokens
    assert "可以" in tokens
    assert "降低" in tokens
    assert "hallucination" in tokens


def test_chunk_text_uses_overlap() -> None:
    text = " ".join(f"token{i}" for i in range(20))
    chunks = chunk_text(text, max_tokens=10, overlap=2)
    assert len(chunks) == 3
    assert chunks[1].startswith("token8 token9")


def test_lexical_retriever_ranks_matching_chunk_first() -> None:
    chunks = [
        ChunkView(id="1", document_id="d1", document_title="A", index=0, text="transformer attention rag"),
        ChunkView(id="2", document_id="d2", document_title="B", index=0, text="database transaction index"),
    ]
    hits = LexicalRetriever().search("rag attention", chunks, top_k=2)
    assert hits[0].chunk.id == "1"
    assert hits[0].matched_terms == ["attention", "rag"]


def test_lexical_retriever_matches_chinese_bigrams() -> None:
    chunks = [
        ChunkView(id="1", document_id="d1", document_title="A", index=0, text="RAG 可以降低幻觉"),
        ChunkView(id="2", document_id="d2", document_title="B", index=0, text="Agent 可以调用工具"),
    ]
    hits = LexicalRetriever().search("为什么可以降低幻觉", chunks, top_k=2)
    assert hits[0].chunk.id == "1"
    assert "降低" in hits[0].matched_terms
    assert "幻觉" in hits[0].matched_terms
