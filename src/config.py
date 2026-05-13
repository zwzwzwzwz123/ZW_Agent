from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
SAMPLE_DOCS_DIR = DATA_DIR / "sample_docs"
UPLOAD_DIR = DATA_DIR / "user_docs"
STORAGE_DIR = ROOT_DIR / "storage"
VECTOR_STORE_DIR = STORAGE_DIR / "faiss_index"
BM25_PATH = STORAGE_DIR / "bm25.pkl"


@dataclass(frozen=True)
class AppConfig:
    chunk_size: int = 700
    chunk_overlap: int = 120
    top_k: int = 8
    rerank_top_k: int = 4
    vector_weight: float = 0.62
    keyword_weight: float = 0.38
    generation_mode: str = "local_fallback"
    llm_provider: str = "deepseek"
    llm_api_key: str | None = None
    llm_base_url: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-v4-flash"
    llm_temperature: float = 0.1
    deepseek_thinking: bool = False
    deepseek_reasoning_effort: str = "high"
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_device: str = "cpu"
    reranker_mode: str = "cross_encoder"
    reranker_model: str = "BAAI/bge-reranker-base"
    web_search_enabled: bool = False
    web_search_provider: str = "tavily"
    tavily_api_key: str | None = None
    web_search_max_results: int = 5
    web_search_depth: str = "basic"
    web_search_include_raw_content: bool = False
    router_mode: str = "hybrid"
    router_force_web_on_freshness: bool = True


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def load_config() -> AppConfig:
    load_dotenv(ROOT_DIR / ".env")

    provider = os.getenv("LLM_PROVIDER", os.getenv("MODEL_PROVIDER", "deepseek")).lower()
    default_base_url = "https://api.deepseek.com" if provider == "deepseek" else "https://api.openai.com/v1"
    default_model = "deepseek-v4-flash" if provider == "deepseek" else "gpt-4o-mini"

    return AppConfig(
        chunk_size=int(os.getenv("CHUNK_SIZE", "700")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "120")),
        top_k=int(os.getenv("TOP_K", "8")),
        rerank_top_k=int(os.getenv("RERANK_TOP_K", "4")),
        vector_weight=float(os.getenv("VECTOR_WEIGHT", "0.62")),
        keyword_weight=float(os.getenv("KEYWORD_WEIGHT", "0.38")),
        generation_mode=os.getenv("GENERATION_MODE", "local_fallback"),
        llm_provider=provider,
        llm_api_key=_first_env("LLM_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY"),
        llm_base_url=os.getenv("LLM_BASE_URL", os.getenv("OPENAI_BASE_URL", default_base_url)),
        llm_model=os.getenv("LLM_MODEL", os.getenv("OPENAI_MODEL", default_model)),
        llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
        deepseek_thinking=_env_bool("DEEPSEEK_THINKING", False),
        deepseek_reasoning_effort=os.getenv("DEEPSEEK_REASONING_EFFORT", "high"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"),
        embedding_device=os.getenv("EMBEDDING_DEVICE", "cpu"),
        reranker_mode=os.getenv("RERANKER_MODE", "cross_encoder"),
        reranker_model=os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base"),
        web_search_enabled=_env_bool("WEB_SEARCH_ENABLED", False),
        web_search_provider=os.getenv("WEB_SEARCH_PROVIDER", "tavily").lower(),
        tavily_api_key=os.getenv("TAVILY_API_KEY") or None,
        web_search_max_results=int(os.getenv("WEB_SEARCH_MAX_RESULTS", "5")),
        web_search_depth=os.getenv("WEB_SEARCH_DEPTH", "basic"),
        web_search_include_raw_content=_env_bool("WEB_SEARCH_INCLUDE_RAW_CONTENT", False),
        router_mode=os.getenv("ROUTER_MODE", "hybrid").lower(),
        router_force_web_on_freshness=_env_bool("ROUTER_FORCE_WEB_ON_FRESHNESS", True),
    )
