# RAG Agent Lab

一个面向大模型应用算法岗位的作品集项目。目标不是做一个普通 PDF Chatbot，而是从第一性原理拆解 RAG 与 Agent：先把可解释的检索增强问答系统跑通，再逐步加入 embedding、向量库、rerank、评估、查询改写和工具调用。

## 项目档案与学习文档

- [项目成长档案](docs/PROJECT_JOURNAL.md)：记录项目初衷、最终目标、当前进度、技术决策、问题复盘、面试回答和学习路线。

## 当前能力

- 文档录入与本地知识库存储
- 文本切分 chunking
- 可解释关键词向量检索
- 基于检索证据的回答生成
- 引用片段、命中分数、匹配词展示
- React Web UI + FastAPI 后端
- RAG 核心单元测试

## 为什么第一版不用复杂框架

RAG 的本质链路是：

1. 把文档拆成适合检索的片段
2. 把用户问题变成可检索表示
3. 找到最相关的证据
4. 把证据交给生成模型
5. 约束模型基于证据回答并给出来源

第一版使用可解释的词项向量和余弦相似度，是为了让每一步都能被观察和调试。后续替换成 embedding、向量数据库和 reranker 时，你能清楚知道它们分别解决了什么问题。

## 技术栈

- Backend: Python, FastAPI, Pydantic
- Frontend: React, TypeScript, Vite
- Retrieval v0: lexical term vector + cosine similarity
- Storage v0: local JSON

## 本地运行

### 后端

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

如果当前环境暂时装不上 FastAPI，可以先使用标准库开发后备服务：

```bash
cd backend
python -m app.dev_server
```

### 前端

```bash
cd frontend
npm install
npm run dev
```

打开 `http://127.0.0.1:5173`。

## 测试

```bash
cd backend
pytest
```

## 学习路线

### Phase 1: RAG 基础闭环

- chunk size 和 overlap 对召回的影响
- 关键词检索、向量检索、混合检索的差异
- top-k、相似度分数、引用来源如何影响回答质量

### Phase 2: 语义检索升级

- 接入 embedding model
- 使用 pgvector / Chroma / FAISS
- 对比 lexical retrieval 与 semantic retrieval
- 加入 query rewrite 和 rerank

### Phase 3: RAG 评估与优化

- 构造 benchmark question set
- 评估 recall、faithfulness、answer relevance
- 记录失败案例并归因：切分失败、检索失败、生成失败
- 做 ablation study：不同 chunk size、top-k、reranker 的效果对比

### Phase 4: Agent 扩展

- 把检索、总结、评估、报告生成封装成 tools
- 加入 planner，让 Agent 决定调用哪些工具
- 加入 reflection，检查回答是否被证据支持
- 支持多步骤任务：根据岗位 JD 生成技能差距分析和学习计划

## 简历亮点表达

- 从零实现并迭代一个可解释 RAG 系统，覆盖文档切分、检索、生成、引用溯源与 Web 可视化
- 设计 RAG 评估流程，对 chunk size、top-k、rerank 等策略进行对比实验
- 扩展 Agent 工具调用链路，使系统支持多步骤学习规划与证据一致性检查
