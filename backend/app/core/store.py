from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from app.core.schemas import ChunkView, DocumentCreate, DocumentSummary
from app.core.text import chunk_text, normalize_text


class KnowledgeStore:
    """本地知识库。

    Store 负责保存两类东西：
    - document：用户上传或输入的原始文档。
    - chunk：文档切分后的检索单元。

    在真实生产系统中，这里通常会被数据库替代：
    document 元数据放 PostgreSQL，向量放 pgvector/FAISS/Chroma。
    当前先用 JSON 文件，是为了让数据结构一眼能看懂。
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # 内存中保留两张“表”：documents 和 chunks。
        # 这样写是为了让你提前习惯后续数据库建模。
        self._documents: dict[str, dict] = {}
        self._chunks: dict[str, dict] = {}
        self.load()

    def load(self) -> None:
        """从本地 JSON 文件恢复知识库。"""
        if not self.path.exists():
            return
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        self._documents = raw.get("documents", {})
        self._chunks = raw.get("chunks", {})

    def save(self) -> None:
        """把内存中的知识库写回磁盘。

        ensure_ascii=False 很重要：它保证中文按 UTF-8 原样写入，
        而不是变成一长串 \\uXXXX，便于直接阅读和调试。
        """
        payload = {"documents": self._documents, "chunks": self._chunks}
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def add_document(self, document: DocumentCreate) -> DocumentSummary:
        """新增文档，并立即切分成 chunks。

        RAG 系统通常把“入库阶段”和“查询阶段”分开：
        - 入库阶段：清洗、切分、向量化、存储。
        - 查询阶段：检索、重排、生成。

        当前函数完成的是入库阶段的前半部分。
        """
        document_id = str(uuid4())
        text = normalize_text(document.text)
        chunks = chunk_text(text)

        self._documents[document_id] = {
            "id": document_id,
            "title": document.title.strip(),
            "text": text,
            "chunk_ids": [],
        }

        for index, chunk_text_value in enumerate(chunks):
            # 每个 chunk 都有自己的 id，因为用户最终看到的引用应该指向具体证据片段。
            chunk_id = str(uuid4())
            chunk = {
                "id": chunk_id,
                "document_id": document_id,
                "document_title": document.title.strip(),
                "index": index,
                "text": chunk_text_value,
            }
            self._chunks[chunk_id] = chunk
            self._documents[document_id]["chunk_ids"].append(chunk_id)

        self.save()
        return self._summary(document_id)

    def list_documents(self) -> list[DocumentSummary]:
        """返回文档列表摘要，用于前端侧边栏展示。"""
        return [self._summary(document_id) for document_id in self._documents]

    def list_chunks(self) -> list[ChunkView]:
        """返回所有可检索 chunk。

        v0 直接全量扫描所有 chunk。数据量小的时候足够直观。
        数据量变大后，这里会替换为向量数据库查询。
        """
        return [ChunkView(**chunk) for chunk in self._chunks.values()]

    def clear(self) -> None:
        """清空知识库，方便本地实验时重新开始。"""
        self._documents = {}
        self._chunks = {}
        self.save()

    def _summary(self, document_id: str) -> DocumentSummary:
        """把内部完整文档结构转换成前端需要的摘要。"""
        document = self._documents[document_id]
        return DocumentSummary(
            id=document_id,
            title=document["title"],
            chunk_count=len(document["chunk_ids"]),
        )
