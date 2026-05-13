from __future__ import annotations

import json

from langchain_core.prompts import ChatPromptTemplate

from src.config import AppConfig
from src.generator import AnswerGenerator
from src.models import RetrievedChunk
from src.web_search import WebSearchDecision, default_topic, has_freshness_need, rule_based_web_search


ROUTER_SYSTEM_PROMPT = """你是一个 RAG Agent 的工具路由器。
你的任务是判断是否需要联网搜索，以及应该搜索什么。
必须只输出 JSON，不要输出 Markdown。
JSON 字段：
- should_search: boolean
- reason: string
- search_query: string
- topic: general 或 news
- source_preference: any、official_docs、academic、news 之一

判断原则：
1. 如果问题需要最新事实、版本、价格、新闻、政策、公司/产品状态，应联网。
2. 如果本地证据置信度低，应联网补充。
3. 如果问题只需要解释概念，且本地证据可用，不要联网。
4. search_query 要具体、可搜索，必要时偏向官方来源。
"""

ROUTER_USER_PROMPT = """用户问题：
{question}

问题意图：
{intent}

本地证据置信度：
{confidence}

本地检索摘要：
{evidence_summary}

请输出联网搜索决策 JSON。"""


class LLMRouter:
    def __init__(self, config: AppConfig, generator: AnswerGenerator) -> None:
        self.config = config
        self.generator = generator

    def decide(
        self,
        question: str,
        intent: str,
        confidence: str,
        chunks: list[RetrievedChunk],
        web_enabled: bool,
    ) -> WebSearchDecision:
        rule_decision = rule_based_web_search(question, confidence, web_enabled)
        if not web_enabled:
            return rule_decision
        if self.config.router_mode == "rules":
            return rule_decision
        if self.config.router_force_web_on_freshness and has_freshness_need(question):
            return rule_decision
        if confidence.startswith("低"):
            return rule_decision
        if self.generator.llm is None:
            return WebSearchDecision(
                rule_decision.should_search,
                f"{rule_decision.reason}；LLM Router 未启用，使用规则兜底",
                search_query=rule_decision.search_query,
                topic=rule_decision.topic,
                source_preference=rule_decision.source_preference,
                decision_source="rule_fallback_no_llm",
            )

        prompt = ChatPromptTemplate.from_messages(
            [("system", ROUTER_SYSTEM_PROMPT), ("human", ROUTER_USER_PROMPT)]
        )
        chain = prompt | self.generator.llm
        raw = chain.invoke(
            {
                "question": question,
                "intent": intent,
                "confidence": confidence,
                "evidence_summary": self._summarize_chunks(chunks),
            }
        )
        try:
            parsed = self._parse_json(str(raw.content))
            return WebSearchDecision(
                should_search=bool(parsed.get("should_search", False)),
                reason=str(parsed.get("reason") or "LLM Router 给出决策"),
                search_query=str(parsed.get("search_query") or question),
                topic=str(parsed.get("topic") or default_topic(question)),
                source_preference=str(parsed.get("source_preference") or "any"),
                decision_source="llm_router",
            )
        except Exception as exc:
            return WebSearchDecision(
                rule_decision.should_search,
                f"LLM Router 解析失败，使用规则兜底：{exc}",
                search_query=rule_decision.search_query,
                topic=rule_decision.topic,
                source_preference=rule_decision.source_preference,
                decision_source="rule_fallback_parse_error",
            )

    def _summarize_chunks(self, chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return "没有检索到本地证据。"
        lines = []
        for idx, chunk in enumerate(chunks[:5], start=1):
            preview = chunk.document.page_content[:220].replace("\n", " ")
            lines.append(
                f"{idx}. 来源={chunk.source}, 重排分={chunk.rerank_score:.3f}, 内容={preview}"
            )
        return "\n".join(lines)

    def _parse_json(self, text: str) -> dict:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            cleaned = cleaned.removeprefix("json").strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("未找到 JSON 对象")
        return json.loads(cleaned[start : end + 1])
