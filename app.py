from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.agent import RAGAgent, serialize_steps
from src.config import SAMPLE_DOCS_DIR, UPLOAD_DIR, load_config
from src.evaluation import evaluate_retrieval
from src.generator import AnswerGenerator
from src.knowledge_base import KnowledgeBase
from src.router import LLMRouter
from src.web_search import WebSearchClient


st.set_page_config(page_title="智能体 RAG 简历项目", page_icon="RAG", layout="wide")


@st.cache_resource(show_spinner=False)
def get_runtime(config_hash: str) -> tuple[KnowledgeBase, RAGAgent]:
    config = load_config()
    kb = KnowledgeBase(config)
    kb.load()
    generator = AnswerGenerator(config)
    web_search = WebSearchClient(config) if config.web_search_enabled else None
    router = LLMRouter(config, generator)
    agent = RAGAgent(kb, generator, web_search=web_search, router=router)
    return kb, agent


def save_uploads(files) -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    for file in files:
        target = UPLOAD_DIR / file.name
        target.write_bytes(file.getbuffer())
    return UPLOAD_DIR


def config_signature() -> str:
    config = load_config()
    return json.dumps(config.__dict__, sort_keys=True, ensure_ascii=False)


config = load_config()
kb, agent = get_runtime(config_signature())

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None

st.title("智能体 RAG 简历项目")
st.caption("LangChain + Streamlit + FAISS + BM25 + BGE reranker + 多轮对话 + 联网补充检索 + 可观测 Agent 轨迹")

with st.sidebar:
    st.header("知识库")
    source_mode = st.radio("文档来源", ["示例文档", "上传文档"], horizontal=True)
    uploaded_files = st.file_uploader(
        "上传 .md / .txt / .pdf 文件",
        type=["md", "txt", "pdf"],
        accept_multiple_files=True,
    )

    if st.button("构建 / 重建索引", type="primary", use_container_width=True):
        with st.spinner("正在构建 Embedding、FAISS 索引和 BM25 索引..."):
            doc_dir = SAMPLE_DOCS_DIR
            if source_mode == "上传文档" and uploaded_files:
                doc_dir = save_uploads(uploaded_files)
            stats = kb.build(doc_dir=doc_dir, reset=True)
            get_runtime.clear()
        st.success(f"已从 {stats['files']} 个文件构建 {stats['chunks']} 个文本块。")
        st.json(stats)

    if st.button("清空多轮对话", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_result = None
        st.rerun()

    st.divider()
    st.header("运行配置")
    st.write(f"LLM 提供方：`{config.llm_provider}`")
    st.write(f"LLM 模型：`{config.llm_model}`")
    st.write(f"Embedding 模型：`{config.embedding_model}`")
    st.write(f"重排模式：`{config.reranker_mode}`")
    st.write(f"重排模型：`{config.reranker_model}`")
    st.write(f"生成模式：`{config.generation_mode}`")
    st.write(f"Router 模式：`{config.router_mode}`")
    st.write(f"联网搜索：`{'开启' if config.web_search_enabled else '关闭'}`")
    if config.web_search_enabled:
        st.write(f"搜索提供方：`{config.web_search_provider}`")
        st.write(f"搜索条数：`{config.web_search_max_results}`")
    st.write(f"召回 Top-k：`{config.top_k}`，重排后保留：`{config.rerank_top_k}`")

tab_chat, tab_index, tab_eval = st.tabs(["多轮问答", "索引诊断", "检索评估"])

with tab_chat:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("输入问题。可以追问上一轮回答，例如：那 rerank 具体怎么做？")
    if question:
        history = list(st.session_state.messages)
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("正在结合历史对话检索、重排、路由并生成回答..."):
                result = agent.answer(question, chat_history=history)
            st.markdown(result.answer)
            st.info(f"置信度判断：{result.confidence}")

        st.session_state.messages.append({"role": "assistant", "content": result.answer})
        st.session_state.last_result = result

    result = st.session_state.last_result
    if result is not None:
        left, right = st.columns([1.15, 0.85])
        with left:
            st.subheader("本地检索证据")
            rows = [chunk.as_dict() for chunk in result.chunks]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            for idx, chunk in enumerate(result.chunks, start=1):
                with st.expander(f"{idx}. {chunk.source} | 重排分={chunk.rerank_score:.3f}"):
                    st.write(chunk.document.page_content)
                    st.json(chunk.document.metadata)

            st.subheader("联网搜索证据")
            if result.web_results:
                st.dataframe(
                    pd.DataFrame([item.as_dict() for item in result.web_results]),
                    use_container_width=True,
                    hide_index=True,
                )
                for idx, item in enumerate(result.web_results, start=1):
                    with st.expander(f"{idx}. {item.title}"):
                        st.write(item.content)
                        st.link_button("打开来源网页", item.url)
            else:
                st.caption("本轮没有使用联网搜索，或联网搜索未返回结果。")

        with right:
            st.subheader("Agent 执行轨迹")
            for step in result.steps:
                with st.expander(step.tool, expanded=True):
                    st.caption(step.input)
                    st.write(step.output)
            st.subheader("诊断 JSON")
            st.json({"steps": serialize_steps(result.steps), "diagnostics": result.diagnostics})

with tab_index:
    st.subheader("语料概览")
    summary = kb.corpus_summary()
    st.metric("文本块数量", summary["chunks"])
    st.metric("平均字符数 / 文本块", summary["avg_chars"])
    st.write("来源文件")
    st.dataframe(
        pd.DataFrame([{"来源": source, "文本块数": count} for source, count in summary["sources"].items()]),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("检索调试器")
    debug_query = st.text_input("调试查询", value="混合检索 重排")
    if st.button("调试检索"):
        chunks = kb.search(debug_query)
        st.dataframe(pd.DataFrame([chunk.as_dict() for chunk in chunks]), use_container_width=True, hide_index=True)

with tab_eval:
    st.subheader("检索冒烟评估")
    st.write("这里内置的是很小的启动评估集。确定你的知识库领域后，应替换成带标准答案的领域问题集。")
    if st.button("运行检索评估"):
        metrics = evaluate_retrieval(kb)
        cols = st.columns(3)
        cols[0].metric("样例数", int(metrics["cases"]))
        cols[1].metric("Recall@K", f"{metrics['recall_at_k']:.2f}")
        cols[2].metric("MRR", f"{metrics['mrr']:.2f}")
