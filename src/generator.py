from __future__ import annotations

import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.config import AppConfig
from src.models import RetrievedChunk, WebSearchResult
from src.prompts import ANSWER_PROMPT, SYSTEM_PROMPT


API_GENERATION_MODES = {"api", "llm", "openai", "openai_compatible", "deepseek"}


class AnswerGenerator:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.llm: ChatOpenAI | None = None
        if config.generation_mode in API_GENERATION_MODES and config.llm_api_key:
            self.llm = ChatOpenAI(
                model=config.llm_model,
                temperature=config.llm_temperature,
                api_key=config.llm_api_key,
                base_url=config.llm_base_url,
            )
            if config.llm_provider == "deepseek":
                thinking_type = "enabled" if config.deepseek_thinking else "disabled"
                bind_kwargs = {"extra_body": {"thinking": {"type": thinking_type}}}
                if config.deepseek_thinking:
                    bind_kwargs["reasoning_effort"] = config.deepseek_reasoning_effort
                self.llm = self.llm.bind(**bind_kwargs)

    def generate(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        web_results: list[WebSearchResult] | None = None,
        chat_history: list[dict[str, str]] | None = None,
    ) -> str:
        web_results = web_results or []
        chat_history = chat_history or []
        if self.llm is not None:
            prompt = ChatPromptTemplate.from_messages(
                [("system", SYSTEM_PROMPT), ("human", ANSWER_PROMPT)]
            )
            chain = prompt | self.llm
            result = chain.invoke(
                {
                    "question": question,
                    "history": self._format_history(chat_history),
                    "context": self._format_context(chunks, web_results),
                }
            )
            return str(result.content)
        return self._local_grounded_answer(question, chunks, web_results, chat_history)

    def _format_history(self, chat_history: list[dict[str, str]]) -> str:
        if not chat_history:
            return "无历史对话。"
        recent = chat_history[-8:]
        lines = []
        for message in recent:
            role = "用户" if message.get("role") == "user" else "助手"
            lines.append(f"{role}: {message.get('content', '')}")
        return "\n".join(lines)

    def _format_context(self, chunks: list[RetrievedChunk], web_results: list[WebSearchResult]) -> str:
        lines = []
        if chunks:
            lines.append("## 本地知识库证据")
            for idx, chunk in enumerate(chunks, start=1):
                citation = f"{chunk.source}#{chunk.chunk_id}"
                lines.append(
                    f"本地证据 {idx}\n"
                    f"citation_id: {citation}\n"
                    f"rerank_score: {chunk.rerank_score:.3f}\n"
                    f"content:\n{chunk.document.page_content}"
                )
        if web_results:
            lines.append("## 联网搜索证据")
            for idx, result in enumerate(web_results, start=1):
                lines.append(
                    f"网页证据 {idx}\n"
                    f"citation_id: [网页-{idx}: {result.url}]\n"
                    f"title: {result.title}\n"
                    f"url: {result.url}\n"
                    f"search_score: {result.score:.3f}\n"
                    f"content:\n{result.content}"
                )
        return "\n\n".join(lines)

    def _local_grounded_answer(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        web_results: list[WebSearchResult],
        chat_history: list[dict[str, str]],
    ) -> str:
        if not chunks and not web_results:
            return "当前没有检索到足够证据。建议补充本地文档，或启用联网搜索后重新提问。"

        history_note = ""
        if chat_history:
            history_note = f"已参考最近 {min(len(chat_history), 8)} 条历史消息。\n\n"

        query_terms = {term.lower() for term in re.findall(r"[\w\u4e00-\u9fff]+", question)}
        selected_sentences: list[str] = []

        for chunk in chunks:
            sentences = re.split(r"(?<=[。.!?！？])\s+|\n+", chunk.document.page_content)
            best = self._pick_best_sentence(sentences, query_terms)
            if best:
                selected_sentences.append(f"{self._normalize_display_sentence(best)} [{chunk.source}#{chunk.chunk_id}]")

        for idx, result in enumerate(web_results, start=1):
            sentences = re.split(r"(?<=[。.!?！？])\s+|\n+", result.content)
            best = self._pick_best_sentence(sentences, query_terms)
            if best:
                selected_sentences.append(f"{self._normalize_display_sentence(best)} [网页-{idx}: {result.url}]")

        evidence = "\n".join(f"- {sentence}" for sentence in selected_sentences[:6])
        return (
            f"{history_note}基于当前检索证据，可以得到以下回答：\n\n"
            f"{evidence}\n\n"
            "说明：当前处于 `local_fallback` 模式，系统采用抽取式生成，主要用于验证多轮对话、检索、重排、联网补充、引用和 UI 链路；"
            "配置 DeepSeek 或其他 OpenAI 兼容模型后，会切换为生成式综合回答。"
        )

    def _pick_best_sentence(self, sentences: list[str], query_terms: set[str]) -> str:
        candidates = [sentence for sentence in sentences if self._is_content_sentence(sentence)]
        ranked = sorted(candidates, key=lambda sentence: self._sentence_score(sentence, query_terms), reverse=True)
        return next((sentence.strip() for sentence in ranked if sentence.strip()), "")

    def _normalize_display_sentence(self, sentence: str) -> str:
        cleaned = re.sub(r"^\s{0,3}#{1,6}\s+", "", sentence.strip())
        cleaned = re.sub(r"^\s*[-*+]\s+", "", cleaned)
        cleaned = re.sub(r"^\s*\d+[.)]\s+", "", cleaned)
        return cleaned

    def _is_content_sentence(self, sentence: str) -> bool:
        text = sentence.strip()
        if not text:
            return False
        if re.match(r"^\s{0,3}#{1,6}\s+", text):
            return False
        return len(re.findall(r"[\w\u4e00-\u9fff]", text)) >= 18

    def _sentence_score(self, sentence: str, query_terms: set[str]) -> tuple[int, int]:
        tokens = set(re.findall(r"[\w\u4e00-\u9fff]+", sentence.lower()))
        overlap = len(query_terms.intersection(tokens))
        return overlap, min(len(sentence), 220)
