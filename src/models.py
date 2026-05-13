from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document


@dataclass
class RetrievedChunk:
    document: Document
    vector_score: float = 0.0
    keyword_score: float = 0.0
    fused_score: float = 0.0
    rerank_score: float = 0.0
    reasons: list[str] = field(default_factory=list)

    @property
    def source(self) -> str:
        return str(self.document.metadata.get("source", "unknown"))

    @property
    def chunk_id(self) -> str:
        return str(self.document.metadata.get("chunk_id", "unknown"))

    def as_dict(self) -> dict[str, Any]:
        return {
            "来源": self.source,
            "文本块ID": self.chunk_id,
            "向量分": round(self.vector_score, 4),
            "关键词分": round(self.keyword_score, 4),
            "融合分": round(self.fused_score, 4),
            "重排分": round(self.rerank_score, 4),
            "预览": self.document.page_content[:220].replace("\n", " "),
        }


@dataclass
class WebSearchResult:
    title: str
    url: str
    content: str
    score: float = 0.0
    published_date: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "标题": self.title,
            "URL": self.url,
            "搜索分": round(self.score, 4),
            "发布日期": self.published_date or "",
            "摘要": self.content[:220].replace("\n", " "),
        }


@dataclass
class AgentStep:
    tool: str
    input: str
    output: str


@dataclass
class AnswerResult:
    answer: str
    chunks: list[RetrievedChunk]
    web_results: list[WebSearchResult]
    steps: list[AgentStep]
    confidence: str
    diagnostics: dict[str, Any]
