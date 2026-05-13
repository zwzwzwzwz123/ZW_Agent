from pathlib import Path

from src.document_loader import discover_files, load_documents, split_documents


def test_sample_docs_can_be_split() -> None:
    files = discover_files(Path("data/sample_docs"))
    docs = load_documents(files)
    chunks = split_documents(docs, chunk_size=300, chunk_overlap=50)
    assert files
    assert docs
    assert chunks
    assert all("chunk_id" in chunk.metadata for chunk in chunks)
