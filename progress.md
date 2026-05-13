# 智能体 RAG 简历项目进度

## 项目目标

构建一个用于求职“大模型应用算法工程师”岗位的作品级 RAG + Agent 项目。

项目不是普通 PDF 聊天 demo，而是要体现算法理解和工程落地能力：文档接入、文本切分、Embedding、FAISS 向量检索、BM25 关键词检索、混合召回、重排、基于证据的回答、引用、Agent 执行轨迹、检索诊断和评估。

## 当前架构

用户文档进入系统后，处理链路如下：

1. `document_loader`：加载 `.md`、`.txt`、`.pdf` 文件，并切分为文本块。
2. `embedding_factory`：通过 LangChain 调用 Hugging Face Embedding 模型，默认本地运行，不需要 API key。
3. `knowledge_base`：构建并持久化 FAISS 向量索引和 BM25 关键词索引。
4. `knowledge_base.search`：执行向量召回和关键词召回，归一化分数并加权融合。
5. `reranker`：默认使用 BGE Cross-Encoder 真实 reranker；MMR 仅作为性能或下载受限时的降级选项。
6. `agent`：分析问题意图、检查知识库、检索证据、估计置信度，并调用答案生成器。
7. `generator`：支持 DeepSeek v4 / OpenAI 兼容模型；没有 API key 时使用本地抽取式 fallback。
8. `web_search`：可选 Tavily 联网搜索工具，本地证据不足或问题需要时效信息时补充网页证据。
9. `router`：规则兜底 + LLM Router，决定是否联网、搜索什么 query、偏好什么来源。
10. `app.py`：Streamlit Web UI，支持建库、问答、证据查看、联网证据查看、Agent 轨迹、索引诊断和检索评估。

## DeepSeek v4 适配状态

已完成 DeepSeek v4 适配。DeepSeek API 兼容 OpenAI ChatCompletions，所以代码使用 `langchain_openai.ChatOpenAI` 作为 OpenAI 兼容客户端，通过环境变量切换 provider、base URL 和模型名。

推荐 `.env` 配置：

```env
DEEPSEEK_API_KEY=你的 DeepSeek API Key
LLM_PROVIDER=deepseek
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-v4-flash
GENERATION_MODE=deepseek
```

可选更强模型：

```env
LLM_MODEL=deepseek-v4-pro
```

可选 thinking 配置：

```env
DEEPSEEK_THINKING=true
DEEPSEEK_REASONING_EFFORT=high
```

说明：DeepSeek 官方文档显示旧模型名 `deepseek-chat` 和 `deepseek-reasoner` 会在 2026-07-24 废弃，因此项目默认使用 `deepseek-v4-flash`。

## 哪些模块需要 API Key

- 答案生成 LLM：需要。DeepSeek v4 需要 `DEEPSEEK_API_KEY`。
- 联网搜索：可选。启用联网搜索需要 `TAVILY_API_KEY`。
- Embedding：默认不需要。当前使用本地 Hugging Face 模型 `BAAI/bge-small-zh-v1.5`，首次运行下载模型，之后本地生成向量。
- FAISS 向量库：不需要 API，本地索引。
- BM25：不需要 API，本地关键词检索。
- MMR 重排：不需要 API，本地计算相似度和多样性。
- Cross-Encoder 重排：默认不需要 API，但首次运行要下载 reranker 模型。
- Streamlit UI：不需要 API。

## 文件说明

- `app.py`：Streamlit Web UI，包含文档上传/建库、问答、索引诊断、检索评估。
- `eval_retrieval.py`：命令行检索评估入口。
- `scripts/build_index.py`：命令行建库脚本，不依赖 UI。
- `requirements.txt`：Python 依赖列表。
- `.env.example`：运行配置模板，默认面向 DeepSeek v4。
- `README.md`：项目说明和快速启动。
- `progress.md`：当前交接文档，新开对话时可直接贴给助手继续推进。
- `data/sample_docs/*.md`：中文示例知识库，保证项目首次运行有可检索内容。
- `src/config.py`：路径和环境变量配置；已抽象 `LLM_PROVIDER`、`LLM_BASE_URL`、`LLM_MODEL`。
- `src/document_loader.py`：文档发现、加载和递归文本切分。
- `src/embedding_factory.py`：通过 `langchain_huggingface` 创建 Hugging Face Embedding。
- `src/embeddings.py`：中文/英文 token 切分、余弦相似度等轻量工具。
- `src/keyword_index.py`：BM25 关键词索引和持久化。
- `src/knowledge_base.py`：核心检索系统，负责 FAISS、BM25、混合融合和重排调用。
- `src/reranker.py`：MMR 重排和 Cross-Encoder 重排。
- `src/generator.py`：基于检索证据生成回答，支持 DeepSeek v4 / OpenAI 兼容模式和本地 fallback。
- `src/web_search.py`：Tavily 联网搜索工具，以及是否联网的路由判断。
- `src/router.py`：LLM Router。模糊场景下由模型输出结构化联网决策，规则负责强制兜底。
- `src/prompts.py`：中文系统提示词和回答提示词。
- `src/agent.py`：透明 Agent 控制器和执行轨迹。
- `src/models.py`：检索结果、Agent 步骤、最终回答的数据结构。
- `src/evaluation.py`：启动版检索评估集和指标计算。
- `tests/test_document_loader.py`：文档加载与切分冒烟测试。

## 当前完成进度

- 已完成项目骨架。
- 已完成中文示例知识库。
- 已完成文档加载和文本切分。
- 已完成 Hugging Face Embedding 接入。
- 已完成 FAISS 向量索引持久化。
- 已完成 BM25 关键词索引持久化。
- 已完成向量检索 + 关键词检索的混合召回。
- 已完成分数归一化和加权融合。
- 已完成 MMR 重排和可选 Cross-Encoder 重排。
- 已将默认重排方式改为 `cross_encoder`，默认模型为 `BAAI/bge-reranker-base`。
- 已完成透明 Agent 控制器和执行轨迹展示。
- 已完成多轮对话支持：UI 保存历史，Agent 会把追问改写成独立问题，历史对话会参与检索和生成。
- 已完成 DeepSeek v4 / OpenAI 兼容生成模式和本地抽取式 fallback。
- 已完成可选 Tavily 联网搜索工具。
- 已完成“规则兜底 + LLM Router”的高级工具路由逻辑。
- 已完成 Router 结构化决策展示。
- 已完成 UI 中联网搜索证据展示。
- 已完成 Streamlit UI。
- 已完成命令行建库脚本。
- 已完成启动版检索评估。
- 已删除未使用的旧版本地向量索引文件，主链路统一使用 FAISS。
- 已加入 `jieba` 中文分词，提升 BM25 在中文知识库上的可用性。

## 下一步计划

1. 在网络正常的环境安装依赖并完整启动项目。
2. 运行 `python scripts\build_index.py` 构建索引。
3. 运行 `streamlit run app.py` 验证 Web UI 和多轮对话。
4. 配置 `DEEPSEEK_API_KEY`，把 `GENERATION_MODE` 改为 `deepseek`，验证真实 LLM 回答。
5. 如需联网搜索，配置 `TAVILY_API_KEY`，并设置 `WEB_SEARCH_ENABLED=true`。
6. 确定真正的知识库领域，并替换示例文档。
7. 构建 20 到 50 条领域检索评估问题，记录期望命中文档或文本块。
8. 增加 query rewrite、多查询召回或 HyDE。
9. 增加更系统的日志、缓存、耗时统计和错误分析。
10. 熟悉基础链路后，可考虑迁移到 LangGraph 展示更复杂的 Agent 工作流。

## 当前验证状态

- 已运行 `python -m compileall app.py src eval_retrieval.py scripts tests`，静态编译通过。
- 当前环境执行 `python -m pip install -r requirements.txt` 时被代理/网络拦截，错误为 `ProxyError: Cannot connect to proxy`，因此尚未在本机完成依赖安装、建库和 Streamlit 启动验证。
- 该阻塞属于环境网络问题，不是当前代码语法问题；网络恢复后优先运行 `pip install -r requirements.txt`、`python scripts\build_index.py`、`streamlit run app.py`。

## 典型问题与面试回答

### Q：为什么生成模型用 DeepSeek，但 embedding 不用 DeepSeek API？

A：生成模型负责根据检索上下文组织最终回答，需要更强的语言理解和表达能力，因此适合调用 DeepSeek v4。Embedding 负责把文本转为向量，本项目默认使用本地 BGE 模型，成本低、可复现、部署简单，也更便于解释向量检索原理。二者是 RAG 中不同模块，不必来自同一个厂商。

### Q：Agent 什么时候会联网搜索？

A：Agent 会先检索本地知识库。如果问题明显需要时效信息，规则会强制联网；如果本地证据置信度低，规则会建议联网；其他模糊场景交给 LLM Router 判断是否联网、搜索什么 query、偏好什么来源。这样比纯关键词更灵活，也比无条件联网更可控。

### Q：为什么要做 LLM Router，而不是直接用关键词？

A：关键词规则稳定、便宜、可解释，但泛化能力有限。LLM Router 能结合用户问题、本地证据摘要和置信度做更细的工具选择，例如判断是否需要官方来源、是否需要最新资料、搜索 query 应该如何改写。高质量实现不应完全依赖 LLM，所以本项目保留规则兜底。

### Q：为什么不用单纯向量检索，而要做混合检索？

A：向量检索擅长语义相似和同义改写，但容易漏掉精确术语、缩写、实体名、公式名和接口名。BM25 擅长关键词精确匹配。混合检索先提高召回覆盖面，再交给重排模块提高精度，是更接近真实业务系统的设计。

### Q：为什么默认 chunk size 设为 700，overlap 设为 120？

A：chunk 大小本质是在“精确性”和“上下文完整性”之间折中。太小会导致上下文断裂，太大又会引入噪声并浪费 prompt token。700 左右适合作为中文/英文混合文档的起点，120 overlap 可以缓解边界信息丢失。最终参数应该通过 recall@k、MRR 和失败样本分析来调。

### Q：重排用了什么方法？

A：项目默认使用 `BAAI/bge-reranker-base` 作为 Cross-Encoder reranker，对“问题-文本块”对进行语义相关性打分，再选择 top-k 证据。MMR 仍然保留，但定位是下载或性能受限时的降级方案，不作为主展示方案。

### Q：多轮对话是怎么实现的？

A：Streamlit 使用 `st.session_state.messages` 保存历史消息。每次新问题进入 Agent 后，Agent 会先结合最近历史把追问改写成独立问题，再用独立问题进行本地检索、Router 决策和联网搜索；最终生成时也会把历史对话传入 prompt。因此它不是只在 UI 上显示历史，而是历史真正影响检索和生成。

### Q：为什么 UI 要展示检索分数和 Agent 轨迹？

A：RAG 的质量很大程度取决于检索是否拿到了正确证据。展示向量分、关键词分、融合分、重排分、引用和 Agent 步骤，可以让系统可调试、可解释，也能让面试官看到候选人理解 RAG 的关键失败点。

### Q：为什么保留 `local_fallback`？

A：它不是最终质量路径，而是工程可运行性的保障。没有 API key 时，仍然可以验证入库、检索、重排、引用、Agent 轨迹和 UI。真正展示生成能力时，应切换到 DeepSeek v4 或其他 OpenAI 兼容模型。

## 新对话如何继续

如果新开对话，把这个 `progress.md` 发给助手，并说明“从当前项目状态继续”。下一步应优先安装依赖、启动项目、修复环境问题，然后再扩展高级功能。
