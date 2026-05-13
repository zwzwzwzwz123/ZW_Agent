from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf"}


def load_documents(paths: list[Path]) -> list[Document]:
    docs: list[Document] = []
    for path in paths:
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            continue
        if suffix == ".pdf":
            loaded = PyPDFLoader(str(path)).load()
        else:
            loaded = TextLoader(str(path), encoding="utf-8").load()
        for doc in loaded:
            doc.metadata["source"] = path.name
            doc.metadata["path"] = str(path)
        docs.extend(loaded)
    return docs


def split_documents(
    docs: list[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "；", "，", "、", ";", ".", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    source_counts: dict[str, int] = {}
    for chunk in chunks:
        source = str(chunk.metadata.get("source", "unknown"))
        source_counts[source] = source_counts.get(source, 0) + 1
        chunk.metadata["chunk_id"] = f"{source}::chunk-{source_counts[source]:04d}"
        chunk.metadata["char_len"] = len(chunk.page_content)
    return chunks


def discover_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
