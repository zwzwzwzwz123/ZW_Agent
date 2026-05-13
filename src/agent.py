from __future__ import annotations

from dataclasses import asdict
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from src.generator import AnswerGenerator
from src.knowledge_base import KnowledgeBase
from src.models import AgentStep, AnswerResult, RetrievedChunk, WebSearchResult
from src.router import LLMRouter
from src.web_search import WebSearchClient


CONDENSE_SYSTEM_PROMPT = """你负责把多轮对话中的用户追问改写成独立问题。
如果当前问题本身已经完整，直接返回原问题。
只输出改写后的问题，不要解释。"""

CONDENSE_USER_PROMPT = """历史对话：
{history}

当前问题：
{question}

独立问题："""


class RAGAgent:
    """用于作品展示的透明 Agent 控制器。"""

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        generator: AnswerGenerator,
        web_search: WebSearchClient | None = None,
        router: LLMRouter | None = None,
    ) -> None:
        self.knowledge_base = knowledge_base
        self.generator = generator
        self.web_search = web_search
        self.router = router

    def answer(self, question: str, chat_history: list[dict[str, str]] | None = None) -> AnswerResult:
        chat_history = chat_history or []
        steps: list[AgentStep] = []

        standalone_question = self._condense_question(question, chat_history)
        if standalone_question != question:
            steps.append(AgentStep("多轮问题改写", question, standalone_question))
        else:
            steps.append(AgentStep("多轮问题改写", question, "当前问题已足够独立，无需改写"))

        intent = self._analyze_intent(standalone_question)
        steps.append(AgentStep("分析问题意图", standalone_question, intent))

        corpus = self.knowledge_base.corpus_summary()
        steps.append(AgentStep("检查知识库", "当前索引", self._format_corpus(corpus)))

        retrieved = self.knowledge_base.retrieve(standalone_question)
        steps.append(
            AgentStep(
                "本地混合检索并重排",
                standalone_question,
                f"选中 {len(retrieved)} 个文本块："
                + "，".join(f"{chunk.source}/{chunk.rerank_score:.2f}" for chunk in retrieved),
            )
        )

        confidence = self._estimate_confidence(retrieved)
        steps.append(AgentStep("估计本地证据置信度", "已检索证据", confidence))

        web_results: list[WebSearchResult] = []
        if self.router is None:
            raise RuntimeError("Agent 缺少 Router，请检查初始化逻辑。")

        web_decision = self.router.decide(
            question=standalone_question,
            intent=intent,
            confidence=confidence,
            chunks=retrieved,
            web_enabled=self.web_search is not None,
        )
        steps.append(AgentStep("Router 决定是否联网", standalone_question, str(web_decision.as_dict())))

        if web_decision.should_search and self.web_search is not None:
            try:
                web_results = self.web_search.search(web_decision.search_query, topic=web_decision.topic)
                steps.append(
                    AgentStep(
                        "联网搜索",
                        web_decision.search_query,
                        f"获取 {len(web_results)} 条网页证据："
                        + "，".join(result.title for result in web_results[:3]),
                    )
                )
            except Exception as exc:
                steps.append(AgentStep("联网搜索失败", web_decision.search_query, str(exc)))

        final_confidence = self._estimate_final_confidence(confidence, web_results)
        steps.append(AgentStep("估计最终证据置信度", "本地证据 + 联网证据", final_confidence))

        answer = self.generator.generate(question, retrieved, web_results, chat_history=chat_history)
        return AnswerResult(
            answer=answer,
            chunks=retrieved,
            web_results=web_results,
            steps=steps,
            confidence=final_confidence,
            diagnostics={
                "original_question": question,
                "standalone_question": standalone_question,
                "history_turns": len(chat_history),
                "intent": intent,
                "corpus": corpus,
                "local_confidence": confidence,
                "web_search_decision": web_decision.as_dict(),
                "retrieved": [chunk.as_dict() for chunk in retrieved],
                "web_results": [result.as_dict() for result in web_results],
            },
        )

    def _condense_question(self, question: str, chat_history: list[dict[str, str]]) -> str:
        if not chat_history:
            return question
        if self.generator.llm is None:
            recent_user_context = " ".join(
                message.get("content", "") for message in chat_history[-4:] if message.get("role") == "user"
            )
            return f"{recent_user_context} {question}".strip()

        prompt = ChatPromptTemplate.from_messages(
            [("system", CONDENSE_SYSTEM_PROMPT), ("human", CONDENSE_USER_PROMPT)]
        )
        chain = prompt | self.generator.llm
        result = chain.invoke({"history": self._format_history(chat_history), "question": question})
        condensed = str(result.content).strip()
        return condensed or question

    def _format_history(self, chat_history: list[dict[str, str]]) -> str:
        if not chat_history:
            return "无历史对话。"
        lines = []
        for message in chat_history[-8:]:
            role = "用户" if message.get("role") == "user" else "助手"
            lines.append(f"{role}: {message.get('content', '')}")
        return "\n".join(lines)

    def _analyze_intent(self, question: str) -> str:
        lower = question.lower()
        if any(word in lower for word in ["最新", "今天", "现在", "实时", "新闻", "latest", "today", "current"]):
            return "时效信息类问题"
        if any(word in lower for word in ["why", "为什么", "原理", "tradeoff", "取舍"]):
            return "设计原理类问题"
        if any(word in lower for word in ["how", "怎么", "如何", "步骤", "实现"]):
            return "实现路径类问题"
        if any(word in lower for word in ["compare", "对比", "区别"]):
            return "方案对比类问题"
        return "事实总结类问题"

    def _estimate_confidence(self, chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return "低：没有检索到可用本地证据"
        best = max(chunk.rerank_score for chunk in chunks)
        sources = {chunk.source for chunk in chunks}
        if best >= 0.72 and len(sources) >= 2:
            return "高：本地重排分较高，且有多个来源相互支持"
        if best >= 0.45:
            return "中：找到了可用本地证据，但仍建议核对引用"
        return "低：本地证据较弱，回答应保持保守"

    def _estimate_final_confidence(self, local_confidence: str, web_results: list[WebSearchResult]) -> str:
        if web_results and local_confidence.startswith("低"):
            return "中：本地证据较弱，但联网搜索提供了补充证据"
        if web_results and local_confidence.startswith("中"):
            return "中高：本地证据可用，联网搜索提供了补充来源"
        return local_confidence

    def _format_corpus(self, corpus: dict[str, Any]) -> str:
        return f"{corpus['chunks']} 个文本块，平均 {corpus['avg_chars']} 字符，来源={corpus['sources']}"


def serialize_steps(steps: list[AgentStep]) -> list[dict[str, str]]:
    return [asdict(step) for step in steps]
