from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests

from src.config import AppConfig
from src.models import WebSearchResult


FRESHNESS_KEYWORDS = {
    "最新",
    "今天",
    "现在",
    "当前",
    "实时",
    "新闻",
    "价格",
    "版本",
    "发布",
    "更新",
    "latest",
    "today",
    "current",
    "news",
    "price",
    "release",
    "version",
    "2025",
    "2026",
}


@dataclass
class WebSearchDecision:
    should_search: bool
    reason: str
    search_query: str
    topic: str = "general"
    source_preference: str = "any"
    decision_source: str = "rule"

    def as_dict(self) -> dict[str, Any]:
        return {
            "should_search": self.should_search,
            "reason": self.reason,
            "search_query": self.search_query,
            "topic": self.topic,
            "source_preference": self.source_preference,
            "decision_source": self.decision_source,
        }


class WebSearchClient:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def search(self, query: str, topic: str = "general") -> list[WebSearchResult]:
        if not self.config.web_search_enabled:
            return []
        if self.config.web_search_provider != "tavily":
            raise ValueError(f"暂不支持的联网搜索提供方：{self.config.web_search_provider}")
        if not self.config.tavily_api_key:
            raise ValueError("已启用联网搜索，但缺少 TAVILY_API_KEY。")
        return self._search_tavily(query, topic)

    def _search_tavily(self, query: str, topic: str) -> list[WebSearchResult]:
        payload: dict[str, Any] = {
            "query": query,
            "search_depth": self.config.web_search_depth,
            "topic": topic,
            "max_results": self.config.web_search_max_results,
            "include_answer": False,
            "include_raw_content": self.config.web_search_include_raw_content,
            "include_images": False,
            "include_favicon": False,
        }
        response = requests.post(
            "https://api.tavily.com/search",
            headers={
                "Authorization": f"Bearer {self.config.tavily_api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()
        results = []
        for item in data.get("results", []):
            content = item.get("raw_content") or item.get("content") or ""
            results.append(
                WebSearchResult(
                    title=item.get("title") or "未命名网页",
                    url=item.get("url") or "",
                    content=content.strip(),
                    score=float(item.get("score") or 0.0),
                    published_date=item.get("published_date"),
                )
            )
        return results


def has_freshness_need(question: str) -> bool:
    lower = question.lower()
    return any(keyword in lower for keyword in FRESHNESS_KEYWORDS)


def default_topic(question: str) -> str:
    lower = question.lower()
    if any(word in lower for word in ["新闻", "news", "今天", "today"]):
        return "news"
    return "general"


def rule_based_web_search(question: str, confidence: str, enabled: bool) -> WebSearchDecision:
    if not enabled:
        return WebSearchDecision(False, "联网搜索未启用", search_query=question)

    if has_freshness_need(question):
        return WebSearchDecision(
            True,
            "问题包含时效性关键词，需要补充联网证据",
            search_query=question,
            topic=default_topic(question),
            decision_source="rule_freshness",
        )

    if confidence.startswith("低"):
        return WebSearchDecision(
            True,
            "本地知识库证据较弱，尝试联网补充证据",
            search_query=question,
            decision_source="rule_low_confidence",
        )

    return WebSearchDecision(False, "本地知识库证据可用，暂不联网", search_query=question)
