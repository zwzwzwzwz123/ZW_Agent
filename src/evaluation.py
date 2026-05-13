from __future__ import annotations

from dataclasses import dataclass

from src.knowledge_base import KnowledgeBase


@dataclass
class RetrievalCase:
    question: str
    expected_source: str


DEFAULT_CASES = [
    RetrievalCase("为什么 RAG 项目需要混合检索？", "rag_agent_resume_project.md"),
    RetrievalCase("大模型应用算法工程师需要哪些优化能力？", "llm_app_engineer_interview.md"),
    RetrievalCase("Agent 在生产系统里有哪些失败模式？", "llm_app_engineer_interview.md"),
]


def evaluate_retrieval(kb: KnowledgeBase, cases: list[RetrievalCase] | None = None) -> dict[str, float]:
    cases = cases or DEFAULT_CASES
    hits = 0
    reciprocal_ranks = []
    for case in cases:
        chunks = kb.search(case.question)
        sources = [chunk.source for chunk in chunks]
        if case.expected_source in sources:
            hits += 1
            reciprocal_ranks.append(1.0 / (sources.index(case.expected_source) + 1))
        else:
            reciprocal_ranks.append(0.0)
    return {
        "cases": float(len(cases)),
        "recall_at_k": hits / max(len(cases), 1),
        "mrr": sum(reciprocal_ranks) / max(len(reciprocal_ranks), 1),
    }
