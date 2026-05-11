# RAG Agent Lab 项目成长档案

> 维护方式：这份文档用于持续记录项目目标、阶段进度、技术决策、踩坑复盘、面试表达和学习笔记。每完成一个阶段，都应该更新“当前进度”“问题复盘”“学习笔记”和“下一步计划”。

## 1. 项目初衷

这个项目的初衷不是做一个简单的 PDF 问答 Demo，而是为“大模型应用算法岗位”准备一个能真正体现能力的作品集项目。

目标岗位希望候选人具备三类能力：

- 懂原理：理解 Transformer、Embedding、RAG、Agent、Tool Calling、评估与优化的工作机制。
- 能动手：能从零搭建一个可运行系统，而不是只会调用现成 Demo。
- 会优化：能分析系统为什么答错、哪里召回失败、如何通过切分、检索、重排、提示词和评估改进效果。

因此，本项目选择从一个可解释 RAG 系统开始，再逐步扩展到 Agent。这样既能覆盖大模型应用开发的核心链路，也能在面试中讲清楚“为什么这样设计、遇到了什么问题、怎么定位和改进”。

## 2. 项目最终目标

最终目标是构建一个面向学习和求职场景的 Knowledge Agent。

它不只是回答“文档里有什么”，而是能够围绕技术学习和岗位准备完成多步骤任务：

- 上传或录入学习资料、技术文档、论文、岗位 JD、项目说明。
- 基于知识库进行问答，并给出引用来源。
- 对多个资料进行对比、总结和结构化梳理。
- 根据岗位 JD 分析能力差距。
- 自动生成学习计划和项目改进建议。
- 对生成答案做证据一致性检查，降低幻觉。
- 记录评估结果，支持持续优化 RAG 策略。

理想情况下，这个项目最终可以作为简历中的核心项目，展示以下能力：

- RAG 系统设计与实现
- 文档处理与 chunking 策略
- 语义检索与向量数据库
- rerank、query rewrite、hybrid search 等优化方法
- LLM 调用与 grounded generation
- RAG 评估体系设计
- Agent 工具调用、任务规划与反思机制
- 前后端完整工程落地能力

## 3. 当前完成情况

### v0.1 已完成

- 创建前后端项目结构。
- 后端实现本地知识库。
- 实现文档录入。
- 实现文本 chunk 切分。
- 实现本地可解释检索器。
- 实现基于检索结果的回答生成。
- 返回引用片段、命中分数、匹配词。
- 创建 React Web UI。
- UI 支持文档入库、提问、查看检索证据。
- 添加 RAG 核心单元测试。
- 编写初版 README。
- 添加标准库开发后备服务，降低环境依赖。

### 当前技术栈

- Backend: Python, Pydantic, FastAPI 设计入口，标准库 HTTP 开发后备服务
- Frontend: React, TypeScript, Vite
- Retrieval v0: lexical term vector + cosine similarity
- Storage v0: local JSON file
- Test: pytest

### 当前运行方式

后端：

```bash
cd backend
python -m app.dev_server
```

前端：

```bash
cd frontend
npm install --cache .\.npm-cache
npm run dev -- --port 5173
```

访问：

```text
http://127.0.0.1:5173
```

### 当前验证结果

- 后端核心测试通过：`4 passed`
- 前端生产构建通过：`npm run build`
- API 联调通过：文档入库、提问、检索结果返回正常
- 中文检索增强通过：可以匹配“可以”“降低”“幻觉”等中文 bigram

## 4. 当前版本的技术设计

### 4.1 为什么第一版先做可解释 RAG

第一版没有直接使用 LangChain、向量数据库或复杂 Agent 框架，而是先手写一个最小 RAG 闭环。

原因是 RAG 的本质链路并不复杂：

1. 文档进入系统。
2. 文档被切分成 chunk。
3. 用户问题被转成可检索表示。
4. 检索器找到相关 chunk。
5. 生成器基于 chunk 组织答案。
6. 系统返回答案和引用来源。

如果一开始就接入大量框架，系统虽然能跑，但学习者很容易不知道每个组件真正解决了什么问题。第一版选择可解释实现，是为了把 RAG 的骨架看清楚。

### 4.2 当前检索方式

当前版本使用关键词向量和余弦相似度：

- 对 query 做 tokenize。
- 对 chunk 做 tokenize。
- 将 token 计数看成一个稀疏向量。
- 用 cosine similarity 衡量 query 和 chunk 的相似度。
- 返回 top-k 结果。

这不是最终方案，但它有一个优点：可解释。我们可以直接看到匹配了哪些词，分数为什么高。

### 4.3 当前中文检索处理

最初的正则 tokenizer 会把连续中文当成一个整体 token。这样会导致“RAG 为什么可以降低幻觉”和“RAG 可以降低幻觉”之间无法有效匹配。

为了解决这个问题，当前版本对中文做了轻量处理：

- 中文字符拆成单字。
- 同时生成相邻二元词，也就是 bigram。
- 例如“降低幻觉”会产生“降”“低”“幻”“觉”“降低”“低幻”“幻觉”。

这不是最优中文分词方案，但足够解释为什么中文检索需要特殊处理。后续可以替换为 jieba、HanLP、embedding 或 hybrid search。

## 5. 创建过程中遇到的问题与解决方式

这一节非常重要。面试中，项目亮点往往不是“我用了什么技术”，而是“我遇到了什么真实问题，如何定位并解决”。

### 问题 1：一开始是否应该直接使用 LangChain 或 LlamaIndex

#### 现象

做 RAG 项目时，很容易直接套 LangChain、LlamaIndex 或现成模板。这样可以很快跑通 Demo，但会导致自己对底层链路理解不深。

#### 分析

RAG 的核心不是框架，而是数据流：

```text
Document -> Chunk -> Retrieve -> Generate -> Cite -> Evaluate
```

如果不理解这些基础步骤，就算使用框架，也很难解释系统为什么召回失败、为什么答案幻觉、为什么引用不准确。

#### 解决

第一版选择手写最小闭环：

- 自己实现 chunking。
- 自己实现 tokenization。
- 自己实现 term vector。
- 自己实现 cosine similarity。
- 自己展示 matched terms 和 score。

#### 面试回答

可以这样回答：

> 我没有一开始就使用 LangChain 搭 Demo，而是先实现了一个可解释的 RAG 最小闭环。这样做是为了把文档切分、检索、相似度计算、引用返回这些核心环节拆开观察。后续再替换为 embedding、向量库或 reranker 时，我能明确知道每个组件是在解决哪个问题，而不是只会调框架。

### 问题 2：中文检索效果不好

#### 现象

用户提问“RAG 为什么可以降低幻觉”，文档中也有“RAG 可以降低幻觉”，但最初检索只稳定匹配到 `rag`，中文部分匹配不足。

#### 原因

英文天然有空格，简单 tokenizer 可以按词切分。但中文没有空格，如果把一整段中文作为一个 token，那么 query 和 chunk 只要表述略有不同，就很难匹配。

#### 解决

当前版本使用轻量中文 bigram：

- 保留英文 token。
- 中文拆成单字。
- 中文再生成相邻二元词。

这样“降低幻觉”可以和“可以降低幻觉”产生交集。

#### 面试回答

可以这样回答：

> 我在 v0 检索阶段遇到了中文匹配不足的问题。原因是英文可以靠空格分词，但中文连续文本如果被当成一个 token，会导致 query 和 chunk 很难产生交集。我先用字符加 bigram 的方式做了一个轻量修复，让系统能匹配“降低”“幻觉”这类短语。这个方案不是最终最优，但它能解释中文检索为什么需要分词或语义向量。后续我会引入 embedding 或 hybrid search 来提升语义召回。

### 问题 3：pip 安装 FastAPI 被网络或代理阻塞

#### 现象

安装后端依赖时，`pip install -r requirements.txt` 连接 PyPI 或镜像源失败，提示代理连接被重置或超时。

#### 原因

这属于本地开发环境问题，不是项目代码问题。真实项目中经常会遇到依赖源、代理、权限或缓存配置问题。

#### 解决

项目中保留 FastAPI 正式入口，同时增加了一个标准库 HTTP 开发后备服务：

- 正式入口：`backend/app/main.py`
- 后备入口：`backend/app/dev_server.py`

这样即使 FastAPI 暂时安装不上，也可以继续运行最小 RAG 系统，学习和开发不会被环境问题阻塞。

#### 面试回答

可以这样回答：

> 我遇到过 pip 依赖安装被代理阻塞的问题。我的处理不是把问题和业务代码混在一起，而是保留正式 FastAPI 入口，同时加了一个标准库 HTTP 后备服务，保证核心 RAG 链路可以继续验证。这个处理体现了我对开发可用性的考虑：环境问题应该被隔离，不能阻塞核心功能验证。

### 问题 4：npm 缓存目录没有权限

#### 现象

执行 `npm install` 时，npm 尝试写入 `C:\Program Files\nodejs\node_cache`，出现 EPERM 权限错误。

#### 原因

全局 npm cache 位于系统目录，普通用户没有写权限。

#### 解决

改用项目内缓存：

```bash
npm install --cache .\.npm-cache
```

并在 `.gitignore` 中忽略 `.npm-cache/`。

#### 面试回答

可以这样回答：

> 前端依赖安装时遇到过 npm 全局缓存目录权限问题。我没有用管理员权限强行解决，而是把 npm cache 指向项目目录，并把缓存目录加入 gitignore。这样项目对系统权限的依赖更少，也更容易在不同机器上复现。

### 问题 5：PowerShell 中文显示乱码

#### 现象

README 或 API 验证输出在 PowerShell 中出现中文乱码。

#### 原因

Windows 终端编码、PowerShell 输出编码和文件 UTF-8 编码之间可能不一致。文件本身可能是 UTF-8，但终端显示时被错误解码。

#### 解决

验证中文逻辑时，不依赖 PowerShell 直接显示中文，而是通过 Python UTF-8 请求或测试用例验证。后续也可以统一设置终端编码为 UTF-8。

#### 面试回答

可以这样回答：

> 我遇到过 Windows 终端中文乱码的问题。这个问题不能直接判断为业务逻辑错误，所以我把显示问题和数据处理问题分开验证：通过测试用例和 UTF-8 请求确认后端实际能处理中文，再单独处理终端编码。这让我意识到调试时要区分“数据真的错了”和“展示层解码错了”。

## 6. 面试中如何介绍这个项目

### 30 秒版本

> 这个项目是我为大模型应用算法岗位准备的 RAG/Agent 作品集。我从零实现了一个可解释 RAG 系统，包含文档切分、检索、生成、引用溯源和 Web UI。第一版没有直接套框架，而是手写了检索和相似度计算，方便理解 RAG 的核心机制。后续我会加入 embedding、向量数据库、rerank、评估体系和 Agent 工具调用，让它从问答系统扩展成一个能做学习规划和岗位差距分析的 Knowledge Agent。

### 2 分钟版本

> 项目的目标是做一个面向学习和求职场景的 Knowledge Agent。第一阶段我先实现 RAG 的最小闭环：文档录入后做 chunk 切分，查询时通过检索器找出相关片段，再基于片段生成回答，并返回引用、分数和匹配词。
>
> 我没有一开始就使用复杂框架，而是先实现可解释版本，因为我想看清楚 RAG 的底层数据流。当前检索使用 term vector 和 cosine similarity，虽然效果不如 embedding，但它能展示为什么 chunk、tokenize、top-k 和相似度会影响回答质量。
>
> 开发中我也遇到了一些真实问题，比如中文没有空格导致关键词检索效果不好，所以我加了字符和 bigram 的轻量中文处理；还有 pip 依赖安装被代理阻塞，所以我保留 FastAPI 正式入口，同时写了标准库 HTTP 后备服务，保证核心功能可以持续验证。
>
> 下一步我会加入 embedding 和向量数据库，把 lexical retrieval 和 semantic retrieval 做对比，再构造 benchmark question set 做评估。最后会把检索、总结、评估、学习计划生成封装成 Agent tools，实现多步骤任务。

## 7. 学习文档

### 7.1 RAG 是什么

RAG 是 Retrieval-Augmented Generation，检索增强生成。

大模型本身的参数中存储了大量知识，但它有几个问题：

- 训练后知识可能过时。
- 对私有文档不了解。
- 容易编造不存在的事实。
- 很难给出可靠来源。

RAG 的思路是：不要只依赖模型记忆，而是在回答前先检索外部知识，把检索到的证据作为上下文交给模型。

核心公式可以理解为：

```text
Answer = LLM(Query + Retrieved Evidence)
```

所以 RAG 的关键不只是生成，而是“检索到什么证据”。

### 7.2 为什么 chunking 很重要

LLM 和检索器通常不能直接处理无限长文档，所以需要把文档拆成 chunk。

chunk 太大：

- 每个片段包含太多无关信息。
- 检索分数可能被稀释。
- 传给模型的上下文浪费 token。

chunk 太小：

- 语义不完整。
- 重要上下文被切断。
- 模型无法理解片段含义。

overlap 的作用是减少边界切断。例如一个关键句刚好跨越两个 chunk，如果没有 overlap，就可能检索不到完整信息。

后续可以做实验：

- chunk size = 100, 300, 500
- overlap = 0, 50, 100
- 对比召回率和回答质量

### 7.3 当前版本的 cosine similarity

余弦相似度衡量两个向量方向是否接近。

在当前项目中：

- query 被表示成词频向量。
- chunk 被表示成词频向量。
- 两者共同出现的词越多，相似度通常越高。

公式：

```text
cosine(A, B) = dot(A, B) / (|A| * |B|)
```

它的优点是简单、可解释。缺点是不能理解语义。例如“降低幻觉”和“减少编造”意思接近，但关键词不一样，lexical retrieval 很难匹配。

这正是后续要引入 embedding 的原因。

### 7.4 Embedding 要解决什么问题

Embedding 是把文本映射到向量空间。

它希望做到：

- 语义相近的文本，向量距离更近。
- 即使字面词不同，也能检索到相关内容。

例如：

```text
query: 如何减少大模型胡说八道
chunk: RAG 可以降低幻觉
```

关键词检索可能匹配不到，但 embedding 检索应该能发现二者语义相关。

### 7.5 Rerank 要解决什么问题

向量检索通常负责从大量 chunk 中快速召回候选结果，但 top-k 的排序不一定最准确。

reranker 的作用是对候选结果进行更精细排序：

```text
Query + Candidate Chunk -> relevance score
```

典型流程：

```text
Vector Search Top 20 -> Rerank -> Select Top 5 -> LLM
```

面试中可以强调：

> Retriever 更关注召回，reranker 更关注精排。RAG 优化时，不能只看最终回答，要拆开看召回阶段和排序阶段分别出了什么问题。

### 7.6 Agent 和 RAG 的关系

RAG 更像一个能力模块，解决“从知识库找证据并回答”的问题。

Agent 更像一个任务执行框架，解决“面对复杂目标时，如何规划步骤、调用工具、检查结果”的问题。

在本项目中，未来 Agent 可以调用这些 tools：

- `search_knowledge_base`
- `summarize_documents`
- `compare_sources`
- `evaluate_answer_grounding`
- `generate_learning_plan`
- `analyze_job_description`

Agent 的核心循环可以理解为：

```text
Goal -> Plan -> Tool Call -> Observation -> Reflection -> Final Answer
```

## 8. 后续计划

### Phase 1: 打磨当前 RAG v0

- 增加文件上传解析。
- 增加 chunk 展示页面。
- 支持删除单个文档。
- 保存每次问答记录。

### Phase 2: 接入 embedding

- 选择 embedding provider。
- 实现 EmbeddingRetriever。
- 对比 lexical retrieval 和 semantic retrieval。
- 加入向量缓存。

### Phase 3: 引入向量数据库

- 本地优先：FAISS 或 Chroma。
- 工程化优先：PostgreSQL + pgvector。
- 记录不同方案的取舍。

### Phase 4: RAG 评估系统

- 构造测试问题集。
- 标注 expected evidence。
- 评估 recall@k。
- 评估 answer groundedness。
- 保存实验配置和结果。

### Phase 5: Agent 扩展

- 定义 tool 接口。
- 实现 planner。
- 实现 reflection/checker。
- 支持根据岗位 JD 生成技能差距报告。

## 9. 项目进度表

| 阶段 | 状态 | 说明 |
| --- | --- | --- |
| 项目定位 | 已完成 | 面向大模型应用算法岗位 |
| 前后端骨架 | 已完成 | React + Python 后端 |
| RAG v0 最小闭环 | 已完成 | 文档、切分、检索、回答、引用 |
| 中文检索增强 | 已完成 | 字符 + bigram |
| 单元测试 | 已完成 | RAG 核心测试通过 |
| Embedding 检索 | 未开始 | 下一阶段重点 |
| 向量数据库 | 未开始 | 待选型 |
| RAG 评估 | 未开始 | 简历亮点重点 |
| Agent 工具调用 | 未开始 | 后续扩展 |

## 10. 更新日志

### 2026-05-11

- 从空目录创建项目。
- 明确项目定位：大模型应用算法岗位作品集。
- 创建 backend/frontend/docs 结构。
- 实现 RAG v0 最小闭环。
- 创建 React Web UI。
- 增加标准库开发后备服务。
- 解决 npm cache 权限问题。
- 发现并修复中文检索匹配不足问题。
- 新增本项目成长档案。

