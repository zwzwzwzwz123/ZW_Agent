# 智能体 RAG 简历项目

这是一个面向“大模型应用算法工程师”求职展示的端到端 RAG + Agent 项目。项目使用 LangChain 和 Streamlit 构建，重点展示从知识库构建、混合检索、重排、引用生成到可观测 Agent 轨迹的完整工程能力。

## 快速启动

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
python scripts\build_index.py
streamlit run app.py
```

如果你的终端用模块方式运行，也可以使用：

```powershell
python -m scripts.build_index
```

默认 `GENERATION_MODE=local_fallback`，即使没有大模型 API key，也可以验证文档入库、检索、重排、引用和 UI 链路。

## 使用 DeepSeek v4

DeepSeek API 兼容 OpenAI ChatCompletions，所以代码中仍然使用 LangChain 的 `ChatOpenAI` 适配器，但运行时配置为 DeepSeek 的 base URL 和模型名。

在 `.env` 中配置：

```env
DEEPSEEK_API_KEY=你的 DeepSeek API Key
LLM_PROVIDER=deepseek
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-v4-flash
GENERATION_MODE=deepseek
```

如果希望使用更强模型，可以把 `LLM_MODEL` 改为 `deepseek-v4-pro`。DeepSeek v4 还支持可选的 thinking 配置，首跑建议关闭，等项目稳定后再打开：

```env
DEEPSEEK_THINKING=true
DEEPSEEK_REASONING_EFFORT=high
```

说明：DeepSeek 官方文档显示旧模型名 `deepseek-chat` 和 `deepseek-reasoner` 会在 2026-07-24 废弃，本项目默认使用 v4 模型名。

## 哪些地方需要 API Key

- 答案生成 LLM：需要。使用 DeepSeek v4 时需要 `DEEPSEEK_API_KEY`。
- 联网搜索：可选。启用联网搜索时需要 `TAVILY_API_KEY`。
- Embedding：默认不需要。本项目默认使用本地 Hugging Face 模型 `BAAI/bge-small-zh-v1.5`，首次运行会下载模型，之后本地计算向量。
- FAISS 向量库：不需要。FAISS 是本地向量索引。
- BM25 关键词检索：不需要。完全本地计算。
- MMR 重排：不需要 API。使用本地 embedding 相似度做相关性和多样性重排。
- Cross-Encoder 重排：默认不需要 API，但首次使用会下载 `BAAI/bge-reranker-base` 模型，本地运行。
- Streamlit UI：不需要 API。

如果你之后想把 embedding 也换成云服务，可以扩展 `embedding_factory.py`，但首版简历项目建议先用本地 BGE embedding：稳定、可控、成本低，也方便解释原理。

## 启用联网搜索

联网搜索默认关闭。开启后，Agent 会先查本地知识库；如果本地证据弱，或者问题包含“最新、今天、实时、新闻、价格、版本”等时效性关键词，就会调用 Tavily 搜索补充网页证据。

在 `.env` 中配置：

```env
WEB_SEARCH_ENABLED=true
WEB_SEARCH_PROVIDER=tavily
TAVILY_API_KEY=你的 Tavily API Key
WEB_SEARCH_MAX_RESULTS=5
WEB_SEARCH_DEPTH=basic
```

当前实现不会把网页结果自动写入长期知识库，而是作为当前问题的临时补充上下文使用。这样更安全，也更容易在 UI 中解释来源。

## LLM Router

联网决策采用“规则兜底 + LLM Router”的混合策略：

- 明确需要时效信息的问题，规则强制联网。
- 本地证据置信度很低时，规则建议联网。
- 其他模糊情况，由 DeepSeek v4 判断是否需要联网、搜索什么 query、偏好什么来源。
- Router 的结构化决策会展示在 Agent 执行轨迹和诊断 JSON 中。

配置项：

```env
ROUTER_MODE=hybrid
ROUTER_FORCE_WEB_ON_FRESHNESS=true
```

如果想节省 LLM 调用成本，可以改成 `ROUTER_MODE=rules`，此时只用规则和关键词路由。

## 项目亮点

- 基于 LangChain 的文档加载、切分、Embedding、FAISS 向量库和 Prompt 链路。
- 混合检索：向量语义检索 + BM25 关键词检索。
- 默认启用真实 BGE Cross-Encoder reranker，MMR 仅作为网络或性能受限时的降级选项。
- Agent 式工具编排：问题意图分析、知识库检查、检索重排、置信度估计。
- UI 中展示引用、检索分数、重排分数、Agent 执行轨迹和诊断 JSON。
- 支持多轮对话：历史对话会参与追问改写、检索和最终回答生成。
- 本地证据不足或问题需要时效信息时，可自动调用联网搜索补充证据。
- 内置检索冒烟评估，后续可扩展成领域评测集。
- `progress.md` 持续记录项目状态，便于新开对话后继续开发。

## 推荐运行顺序

1. 使用 `local_fallback` 和示例文档跑通完整流程。
2. 配置 `DEEPSEEK_API_KEY`，把 `GENERATION_MODE` 改成 `deepseek`。
3. 保持 `RERANKER_MODE=cross_encoder` 演示真实 rerank；如果本地下载模型失败，再临时降级为 `mmr`。
4. 替换为你真正想展示的领域知识库。
5. 增加 20 到 50 条带标准来源的检索评估问题。
6. 调整 chunk 大小、混合检索权重、rerank 模式和 prompt。
7. 增加 query rewrite、多路召回、LangGraph 工作流或线上监控。

## 当前注意事项

- 当前仓库已经包含中文示例知识库，可直接演示。
- 如果依赖安装失败，优先检查 pip 源、代理和 Hugging Face 模型下载网络。
- `local_fallback` 不是最终质量路径，它用于保证没有 API key 时仍能验证完整工程链路。
- 当前代码和文档已统一为 UTF-8 中文文本，并通过 Python 静态编译检查。
