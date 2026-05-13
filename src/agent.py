from __future__ import annotations

import json
import re
from dataclasses import asdict
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from src.generator import AnswerGenerator
from src.knowledge_base import KnowledgeBase
from src.models import AgentStep, AnswerResult, PlanStepResult, RetrievedChunk, WebSearchResult
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

PLANNER_SYSTEM_PROMPT = """你是一个 RAG Agent 的任务规划器。
你需要把复杂用户目标拆成 2 到 5 个可执行步骤。
每个步骤必须包含：
- goal: 这个步骤要完成什么
- query: 用于检索知识库的查询
- output: 这个步骤应该产出的中间结果

必须只输出 JSON，不要输出 Markdown。
JSON 格式：
{{
  "task_type": "multi_step",
  "steps": [
    {{"goal": "...", "query": "...", "output": "..."}}
  ]
}}
"""

PLANNER_USER_PROMPT = """用户目标：
{question}

问题意图：
{intent}

知识库概况：
{corpus}

请拆解成适合 RAG 检索执行的计划。"""

SYNTHESIS_SYSTEM_PROMPT = """你是一个严谨的 RAG Agent 汇总器。
你会收到用户原始目标、执行计划、每一步检索到的证据摘要和引用。
请综合所有步骤结果，产出面向用户的最终答案。
必须基于证据回答，关键结论后附引用。
引用必须逐字复制 plan_trace 中的 citation_id，不能缩写、改写或重新组织 citation_id。
本地引用格式必须保持为 [source#chunk_id]。"""

SYNTHESIS_USER_PROMPT = """用户目标：
{question}

历史对话：
{history}

执行计划和中间结果：
{plan_trace}

请给出最终答案。"""


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

        if self._should_plan(standalone_question, intent) or self._should_plan(question, intent):
            return self._answer_with_plan(
                original_question=question,
                standalone_question=standalone_question,
                intent=intent,
                corpus=corpus,
                chat_history=chat_history,
                steps=steps,
            )

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

    def _answer_with_plan(
        self,
        original_question: str,
        standalone_question: str,
        intent: str,
        corpus: dict[str, Any],
        chat_history: list[dict[str, str]],
        steps: list[AgentStep],
    ) -> AnswerResult:
        plan = self._make_plan(standalone_question, intent, corpus)
        steps.append(AgentStep("生成多步执行计划", standalone_question, self._format_plan(plan)))

        plan_results: list[PlanStepResult] = []
        all_chunks: list[RetrievedChunk] = []
        for idx, item in enumerate(plan, start=1):
            query = str(item.get("query") or item.get("goal") or standalone_question)
            goal = str(item.get("goal") or query)
            chunks = self.knowledge_base.retrieve(query)
            summary = self._summarize_step_evidence(goal, chunks)
            plan_results.append(PlanStepResult(idx, goal, query, chunks, summary))
            all_chunks.extend(chunks)
            steps.append(
                AgentStep(
                    f"执行计划步骤 {idx}",
                    f"目标：{goal}\n检索：{query}",
                    summary,
                )
            )

        merged_chunks = self._dedupe_chunks(all_chunks)
        confidence = self._estimate_confidence(merged_chunks)
        steps.append(AgentStep("汇总计划证据置信度", "多步检索证据", confidence))

        web_results: list[WebSearchResult] = []
        if self.router is None:
            raise RuntimeError("Agent 缺少 Router，请检查初始化逻辑。")
        web_decision = self.router.decide(
            question=standalone_question,
            intent=intent,
            confidence=confidence,
            chunks=merged_chunks,
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
        steps.append(AgentStep("估计最终证据置信度", "计划证据 + 联网证据", final_confidence))

        answer = self._synthesize_plan_answer(original_question, plan_results, web_results, chat_history)
        return AnswerResult(
            answer=answer,
            chunks=merged_chunks,
            web_results=web_results,
            steps=steps,
            confidence=final_confidence,
            diagnostics={
                "original_question": original_question,
                "standalone_question": standalone_question,
                "history_turns": len(chat_history),
                "intent": intent,
                "corpus": corpus,
                "planning_enabled": True,
                "plan": plan,
                "plan_results": [result.as_dict() for result in plan_results],
                "local_confidence": confidence,
                "web_search_decision": web_decision.as_dict(),
                "retrieved": [chunk.as_dict() for chunk in merged_chunks],
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

    def _should_plan(self, question: str, intent: str) -> bool:
        lower = question.lower()
        planning_keywords = [
            "整理",
            "总结",
            "归纳",
            "对比",
            "比较",
            "生成",
            "草稿",
            "方案",
            "计划",
            "所有",
            "全部",
            "综述",
            "报告",
            "review",
            "summarize",
            "compare",
            "draft",
            "plan",
        ]
        if any(keyword in lower for keyword in planning_keywords):
            return True
        return intent in {"方案对比类问题"} and len(question) >= 18

    def _make_plan(self, question: str, intent: str, corpus: dict[str, Any]) -> list[dict[str, str]]:
        if self.generator.llm is not None:
            prompt = ChatPromptTemplate.from_messages(
                [("system", PLANNER_SYSTEM_PROMPT), ("human", PLANNER_USER_PROMPT)]
            )
            chain = prompt | self.generator.llm
            raw = chain.invoke(
                {
                    "question": question,
                    "intent": intent,
                    "corpus": self._format_corpus(corpus),
                }
            )
            try:
                parsed = self._parse_json(str(raw.content))
                steps = parsed.get("steps", [])
                if isinstance(steps, list) and steps:
                    return [self._normalize_plan_step(step, question) for step in steps[:5]]
            except Exception:
                pass
        return self._fallback_plan(question)

    def _fallback_plan(self, question: str) -> list[dict[str, str]]:
        if any(word in question for word in ["对比", "比较", "区别"]):
            return [
                {"goal": "检索相关方案和概念定义", "query": question, "output": "列出候选方案"},
                {"goal": "提取各方案的优势、限制和适用场景", "query": f"{question} 优势 限制 适用场景", "output": "形成对比要点"},
                {"goal": "综合证据给出结论", "query": f"{question} 取舍 设计选择", "output": "给出建议"},
            ]
        return [
            {"goal": "检索与目标主题相关的全部核心资料", "query": question, "output": "收集主要证据"},
            {"goal": "归纳主要方法、模块和设计取舍", "query": f"{question} 方法 模块 取舍", "output": "形成结构化总结"},
            {"goal": "生成最终草稿或综述答案", "query": f"{question} 综述 草稿 总结", "output": "输出综合结果"},
        ]

    def _normalize_plan_step(self, step: Any, question: str) -> dict[str, str]:
        if not isinstance(step, dict):
            return {"goal": str(step), "query": question, "output": "完成该步骤"}
        goal = str(step.get("goal") or step.get("name") or question)
        query = str(step.get("query") or goal or question)
        output = str(step.get("output") or "完成该步骤")
        return {"goal": goal, "query": query, "output": output}

    def _format_plan(self, plan: list[dict[str, str]]) -> str:
        return "\n".join(
            f"{idx}. {item['goal']}；检索 query：{item['query']}；产出：{item['output']}"
            for idx, item in enumerate(plan, start=1)
        )

    def _summarize_step_evidence(self, goal: str, chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return f"目标：{goal}\n未检索到可用证据。"
        lines = [f"目标：{goal}", f"命中 {len(chunks)} 个证据块。"]
        for idx, chunk in enumerate(chunks[:3], start=1):
            preview = chunk.document.page_content[:120].replace("\n", " ")
            lines.append(
                f"{idx}. {chunk.source}#{chunk.chunk_id}，重排分={chunk.rerank_score:.3f}，预览：{preview}"
            )
        return "\n".join(lines)

    def _dedupe_chunks(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        by_id: dict[str, RetrievedChunk] = {}
        for chunk in chunks:
            key = f"{chunk.source}#{chunk.chunk_id}"
            existing = by_id.get(key)
            if existing is None or chunk.rerank_score > existing.rerank_score:
                by_id[key] = chunk
        deduped = list(by_id.values())
        deduped.sort(key=lambda item: item.rerank_score, reverse=True)
        return deduped[: max(self.knowledge_base.config.rerank_top_k, 6)]

    def _synthesize_plan_answer(
        self,
        question: str,
        plan_results: list[PlanStepResult],
        web_results: list[WebSearchResult],
        chat_history: list[dict[str, str]],
    ) -> str:
        if self.generator.llm is None:
            return self._fallback_plan_answer(plan_results)
        prompt = ChatPromptTemplate.from_messages(
            [("system", SYNTHESIS_SYSTEM_PROMPT), ("human", SYNTHESIS_USER_PROMPT)]
        )
        chain = prompt | self.generator.llm
        result = chain.invoke(
            {
                "question": question,
                "history": self._format_history(chat_history),
                "plan_trace": self._format_plan_trace(plan_results, web_results),
            }
        )
        return str(result.content)

    def _format_plan_trace(
        self, plan_results: list[PlanStepResult], web_results: list[WebSearchResult]
    ) -> str:
        sections = []
        for result in plan_results:
            lines = [
                f"步骤 {result.step_id}: {result.goal}",
                f"query: {result.query}",
                "evidence:",
            ]
            for chunk in result.chunks:
                citation = f"{chunk.source}#{chunk.chunk_id}"
                lines.append(
                    f"- citation_id: {citation}\n"
                    f"  rerank_score: {chunk.rerank_score:.3f}\n"
                    f"  content: {chunk.document.page_content}"
                )
            sections.append("\n".join(lines))
        if web_results:
            web_lines = ["联网搜索证据:"]
            for idx, item in enumerate(web_results, start=1):
                web_lines.append(
                    f"- citation_id: [网页-{idx}: {item.url}]\n"
                    f"  title: {item.title}\n"
                    f"  content: {item.content}"
                )
            sections.append("\n".join(web_lines))
        return "\n\n".join(sections)

    def _fallback_plan_answer(self, plan_results: list[PlanStepResult]) -> str:
        lines = ["我已按多步计划完成检索，结果如下："]
        for result in plan_results:
            lines.append(f"\n{result.step_id}. {result.goal}")
            for chunk in result.chunks[:2]:
                citation = f"{chunk.source}#{chunk.chunk_id}"
                preview = chunk.document.page_content[:160].replace("\n", " ")
                lines.append(f"- {preview} [{citation}]")
        return "\n".join(lines)

    def _parse_json(self, text: str) -> dict[str, Any]:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            cleaned = cleaned.removeprefix("json").strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("未找到 JSON 对象")
        return json.loads(cleaned[start : end + 1])

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
