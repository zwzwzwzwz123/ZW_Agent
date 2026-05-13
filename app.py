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


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --app-bg: #f5f7fb;
            --panel-bg: #ffffff;
            --text-main: #202635;
            --text-soft: #667085;
            --text-muted: #98a2b3;
            --line: #e3e8ef;
            --accent: #e5484d;
            --accent-soft: #fff3f3;
            --teal: #0f766e;
            --blue-soft: #edf4ff;
        }

        html, body, .stApp, [class*="css"] {
            font-family: "Inter", "Source Han Sans SC", "Noto Sans CJK SC", "Microsoft YaHei",
                "PingFang SC", "Segoe UI", sans-serif;
        }

        .stApp {
            background: var(--app-bg);
            color: var(--text-main);
        }

        header[data-testid="stHeader"] {
            background: rgba(245, 247, 251, 0.86);
            backdrop-filter: blur(10px);
        }

        .block-container {
            max-width: 1320px;
            padding: 1.35rem 2.25rem 2.5rem;
        }

        [data-testid="stSidebar"] {
            background: #eaf0f7;
            border-right: 1px solid #d6dee9;
        }

        [data-testid="stSidebar"] > div:first-child {
            padding-top: 1.4rem;
        }

        [data-testid="stSidebar"] h2 {
            font-size: 1.05rem;
            line-height: 1.35;
            letter-spacing: 0;
            color: #1f2937;
            margin-bottom: 0.75rem;
        }

        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] .stMarkdown {
            font-size: 0.88rem;
            color: #475467;
        }

        .app-hero {
            display: flex;
            align-items: flex-end;
            justify-content: space-between;
            gap: 2rem;
            padding: 1.1rem 0 1.15rem;
            border-bottom: 1px solid var(--line);
            margin-bottom: 0.8rem;
        }

        .app-kicker {
            color: var(--accent);
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }

        .app-title {
            color: var(--text-main);
            font-size: clamp(2.15rem, 3vw, 3.35rem);
            line-height: 1.08;
            font-weight: 820;
            letter-spacing: 0;
            margin: 0;
        }

        .app-subtitle {
            color: var(--text-soft);
            font-size: 0.98rem;
            line-height: 1.65;
            margin-top: 0.72rem;
            max-width: 760px;
        }

        .hero-pills {
            display: flex;
            flex-wrap: wrap;
            justify-content: flex-end;
            gap: 0.45rem;
            max-width: 360px;
        }

        .hero-pill {
            color: #344054;
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 999px;
            padding: 0.38rem 0.62rem;
            font-size: 0.78rem;
            font-weight: 650;
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
        }

        h1, h2, h3 {
            letter-spacing: 0 !important;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 1.35rem;
            border-bottom: 1px solid var(--line);
        }

        .stTabs [data-baseweb="tab"] {
            padding: 0.7rem 0.1rem 0.78rem;
            font-size: 0.94rem;
            color: #4b5563;
            height: auto;
        }

        .stTabs [aria-selected="true"] {
            color: var(--accent) !important;
            font-weight: 650;
        }

        .stButton > button,
        [data-testid="stFileUploaderDropzone"] button {
            border-radius: 8px;
            font-weight: 600;
        }

        .stTextInput input,
        textarea,
        [data-testid="stChatInput"] textarea {
            border-radius: 10px !important;
            border-color: #d9e0ea !important;
            background: #ffffff !important;
            box-shadow: none !important;
            font-size: 0.96rem !important;
        }

        [data-testid="stChatInput"] {
            background: transparent;
        }

        [data-testid="stChatInput"] > div {
            border-radius: 10px !important;
            border: 1px solid #d9e0ea !important;
            background: #ffffff !important;
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04) !important;
        }

        .stChatMessage {
            background: var(--panel-bg);
            border: 1px solid var(--line);
            border-radius: 12px;
            padding: 1rem 1.15rem;
            box-shadow: 0 10px 24px rgba(16, 24, 40, 0.045);
        }

        .stChatMessage [data-testid="stMarkdownContainer"] {
            color: #243044;
            font-size: 0.98rem;
            line-height: 1.78;
        }

        .stChatMessage [data-testid="stMarkdownContainer"] h1 {
            font-size: 1.16rem !important;
            line-height: 1.35 !important;
            margin: 0.7rem 0 0.35rem !important;
        }

        .stChatMessage [data-testid="stMarkdownContainer"] h2 {
            font-size: 1.08rem !important;
            line-height: 1.4 !important;
            margin: 0.65rem 0 0.3rem !important;
        }

        .stChatMessage [data-testid="stMarkdownContainer"] h3 {
            font-size: 1rem !important;
            line-height: 1.45 !important;
            margin: 0.55rem 0 0.25rem !important;
        }

        .stChatMessage [data-testid="stMarkdownContainer"] ul {
            padding-left: 1.1rem;
        }

        .stChatMessage [data-testid="stMarkdownContainer"] li {
            margin: 0.35rem 0;
            padding-left: 0.1rem;
        }

        code {
            border-radius: 6px;
            padding: 0.1rem 0.35rem;
            color: #047857;
            background: #eef8f3;
            white-space: normal;
            overflow-wrap: anywhere;
            font-size: 0.86em;
        }

        .stAlert {
            border-radius: 10px;
            border: 1px solid #cfe2ff;
            background: var(--blue-soft);
        }

        div[data-testid="stExpander"] {
            border: 1px solid var(--line);
            border-radius: 10px;
            background: var(--panel-bg);
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.035);
        }

        [data-testid="stDataFrame"] {
            border: 1px solid var(--line);
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.035);
        }

        .section-title {
            display: flex;
            align-items: center;
            gap: 0.48rem;
            margin: 1.2rem 0 0.62rem;
            font-size: 0.95rem;
            font-weight: 760;
            color: var(--text-main);
        }

        .section-title::before {
            content: "";
            width: 0.42rem;
            height: 0.42rem;
            border-radius: 999px;
            background: var(--accent);
            box-shadow: 0 0 0 4px var(--accent-soft);
        }

        .config-row {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 0.75rem;
            padding: 0.34rem 0;
            border-bottom: 1px solid rgba(203, 213, 225, 0.72);
            font-size: 0.84rem;
        }

        .config-row span:first-child {
            color: #667085;
            flex: 0 0 auto;
        }

        .config-row code {
            text-align: right;
            line-height: 1.45;
            color: var(--teal);
            background: rgba(255, 255, 255, 0.74);
            border: 1px solid rgba(226, 232, 240, 0.9);
        }

        @media (max-width: 960px) {
            .app-hero {
                display: block;
            }

            .hero-pills {
                justify-content: flex-start;
                margin-top: 1rem;
            }

            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_config_row(label: str, value: str) -> None:
    st.markdown(
        f'<div class="config-row"><span>{label}</span><code>{value}</code></div>',
        unsafe_allow_html=True,
    )


def render_section_title(title: str) -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


def render_header() -> None:
    st.markdown(
        """
        <section class="app-hero">
            <div>
                <div class="app-kicker">RAG Agent Portfolio</div>
                <h1 class="app-title">智能体 RAG 简历项目</h1>
                <div class="app-subtitle">
                    面向大模型应用算法工程师的端到端演示：混合检索、重排、引用生成、多轮问答与可观测 Agent 轨迹。
                </div>
            </div>
            <div class="hero-pills">
                <span class="hero-pill">FAISS</span>
                <span class="hero-pill">BM25</span>
                <span class="hero-pill">BGE Reranker</span>
                <span class="hero-pill">Streamlit</span>
                <span class="hero-pill">DeepSeek Ready</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


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
inject_styles()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None

render_header()

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
    render_config_row("LLM 提供方", config.llm_provider)
    render_config_row("LLM 模型", config.llm_model)
    render_config_row("Embedding", config.embedding_model)
    render_config_row("重排模式", config.reranker_mode)
    render_config_row("重排模型", config.reranker_model)
    render_config_row("生成模式", config.generation_mode)
    render_config_row("Router", config.router_mode)
    render_config_row("联网搜索", "开启" if config.web_search_enabled else "关闭")
    if config.web_search_enabled:
        render_config_row("搜索提供方", config.web_search_provider)
        render_config_row("搜索条数", str(config.web_search_max_results))
    render_config_row("召回 / 重排", f"{config.top_k} / {config.rerank_top_k}")

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
            render_section_title("本地检索证据")
            rows = [chunk.as_dict() for chunk in result.chunks]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            for idx, chunk in enumerate(result.chunks, start=1):
                with st.expander(f"{idx}. {chunk.source} | 重排分={chunk.rerank_score:.3f}"):
                    st.write(chunk.document.page_content)
                    st.json(chunk.document.metadata)

            render_section_title("联网搜索证据")
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
            render_section_title("Agent 执行轨迹")
            for step in result.steps:
                with st.expander(step.tool, expanded=True):
                    st.caption(step.input)
                    st.write(step.output)
            render_section_title("诊断 JSON")
            st.json({"steps": serialize_steps(result.steps), "diagnostics": result.diagnostics})

with tab_index:
    render_section_title("语料概览")
    summary = kb.corpus_summary()
    st.metric("文本块数量", summary["chunks"])
    st.metric("平均字符数 / 文本块", summary["avg_chars"])
    st.write("来源文件")
    st.dataframe(
        pd.DataFrame([{"来源": source, "文本块数": count} for source, count in summary["sources"].items()]),
        use_container_width=True,
        hide_index=True,
    )

    render_section_title("检索调试器")
    debug_query = st.text_input("调试查询", value="混合检索 重排")
    if st.button("调试检索"):
        chunks = kb.search(debug_query)
        st.dataframe(pd.DataFrame([chunk.as_dict() for chunk in chunks]), use_container_width=True, hide_index=True)

with tab_eval:
    render_section_title("检索冒烟评估")
    st.write("这里内置的是很小的启动评估集。确定你的知识库领域后，应替换成带标准答案的领域问题集。")
    if st.button("运行检索评估"):
        metrics = evaluate_retrieval(kb)
        cols = st.columns(3)
        cols[0].metric("样例数", int(metrics["cases"]))
        cols[1].metric("Recall@K", f"{metrics['recall_at_k']:.2f}")
        cols[2].metric("MRR", f"{metrics['mrr']:.2f}")
