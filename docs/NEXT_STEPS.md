# 当前项目导航

## 完成度估计

距离最终完整版 Agent 项目：约 **15%**。

已完成的是 RAG v0 最小闭环：

- 文档录入
- 文本切分
- 本地关键词检索
- 简单回答生成
- 引用与检索分数展示
- Web UI
- 核心测试

还没完成的核心部分：

- embedding 语义检索
- 向量数据库
- rerank
- RAG 评估
- LLM 真实生成
- Agent 工具调用
- 多步骤任务规划

## 现在优先看哪些文件

按顺序看：

1. `backend/app/core/text.py`
   学 chunking、tokenize、cosine similarity。

2. `backend/app/core/retriever.py`
   学 query 如何找到相关 chunk。

3. `backend/app/core/generator.py`
   学为什么回答要基于证据和引用。

4. `backend/app/core/store.py`
   学 document 和 chunk 怎么存。

5. `frontend/src/main.tsx`
   学前端如何调用后端 API。

暂时不用看：

- `frontend/package-lock.json`
- `frontend/src/styles.css`
- `frontend/index.html`
- `node_modules`
- `dist`

## 下一步具体任务

下一步任务：**做一个 chunk 可视化页面**。

目标：

- 用户添加文档后，可以看到每个 chunk 的内容。
- 每个 chunk 显示所属文档、chunk index、chunk 文本。
- 这样你能直观看到文档是如何被切分的。

需要改的文件：

- `backend/app/dev_server.py`
- `backend/app/main.py`
- `frontend/src/main.tsx`

具体要做：

1. 后端新增 `GET /chunks` 接口。
2. 前端新增 chunks 状态。
3. 前端页面展示 chunk 列表。
4. 添加文档后自动刷新 chunk 列表。
5. 清空知识库后同步清空 chunk 展示。

为什么先做这个：

RAG 的第一个关键问题是 **切分质量**。如果 chunk 切得不好，后面的 embedding、rerank、Agent 都救不回来。

