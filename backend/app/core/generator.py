from app.core.schemas import Citation, SearchHit


class GroundedAnswerGenerator:
    """基于证据的回答生成器。

    当前版本还没有调用真正的 LLM，而是用模板把检索结果组织成答案。
    这不是为了“装作智能”，而是为了先把 RAG 的数据流跑通：
    检索结果如何进入生成阶段？引用如何和答案一起返回？

    未来接入 LLM 时，这个类会变成：
    prompt = query + retrieved_chunks + instruction
    answer = llm.generate(prompt)
    """

    def answer(self, query: str, hits: list[SearchHit]) -> tuple[str, list[Citation]]:
        """根据检索命中的 chunk 生成回答和引用。

        RAG 的一个核心原则是 grounded generation：
        生成内容应该尽量被检索证据支撑，而不是完全依赖模型记忆。
        """
        if not hits:
            # 没有证据时不要硬答，这是降低幻觉的重要策略。
            return (
                "我没有在当前知识库中找到足够相关的内容。你可以补充文档，或换一种更具体的问法。",
                [],
            )

        citations: list[Citation] = []
        answer_lines = [
            f"基于当前知识库，和“{query}”最相关的信息主要来自以下片段：",
        ]

        for index, hit in enumerate(hits[:3], start=1):
            snippet = self._snippet(hit.chunk.text)
            # citation 是回答的证据来源。真实产品里通常还会包含页码、文件名、位置等。
            citations.append(
                Citation(
                    chunk_id=hit.chunk.id,
                    document_title=hit.chunk.document_title,
                    snippet=snippet,
                )
            )
            answer_lines.append(
                f"{index}. 《{hit.chunk.document_title}》中提到：{snippet} "
                f"(相关度 {hit.score:.4f})"
            )

        answer_lines.append("这是一个本地可解释生成器的回答。下一阶段我们会接入 LLM，让它在引用约束下生成更自然的答案。")
        return "\n".join(answer_lines), citations

    def _snippet(self, text: str, max_chars: int = 180) -> str:
        """截取短片段，避免前端引用区域过长。

        注意：这是展示层友好的处理，不等于检索本身。
        """
        if len(text) <= max_chars:
            return text
        return f"{text[:max_chars].rstrip()}..."
