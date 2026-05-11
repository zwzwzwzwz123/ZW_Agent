from pydantic import BaseModel, Field


# schemas.py 定义前后端和核心模块之间传递的数据形状。
# 学习时可以把它理解成“接口契约”：谁传什么字段、谁返回什么字段，都在这里说清楚。

class DocumentCreate(BaseModel):
    # 用户新建文档时只需要传标题和正文。
    # Field 的约束可以提前挡住空标题、空正文这类无效输入。
    title: str = Field(min_length=1, max_length=160)
    text: str = Field(min_length=1)


class DocumentSummary(BaseModel):
    # 前端侧边栏不需要完整正文，只需要摘要信息。
    id: str
    title: str
    chunk_count: int


class ChunkView(BaseModel):
    # Chunk 是 RAG 检索的基本单位。
    # document_id 负责追溯原文档，index 负责表示它在文档中的顺序。
    id: str
    document_id: str
    document_title: str
    index: int
    text: str


class SearchRequest(BaseModel):
    # top_k 控制检索返回多少个候选证据。
    # 过小可能漏掉答案，过大可能引入噪声。
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=12)


class SearchHit(BaseModel):
    # SearchHit 不只返回 chunk，还返回 score 和 matched_terms。
    # 这是为了让检索过程可解释，方便后续做错误分析。
    chunk: ChunkView
    score: float
    matched_terms: list[str]


class AskRequest(SearchRequest):
    # 当前 ask 和 search 使用相同输入。单独定义 AskRequest 是为了给未来扩展留位置：
    # 例如加入 temperature、answer_style、是否启用 rerank 等参数。
    pass


class Citation(BaseModel):
    # Citation 是“答案可追溯”的基础。
    # 没有 citation 的 RAG 很难让用户信任，也难以评估事实一致性。
    chunk_id: str
    document_title: str
    snippet: str


class AskResponse(BaseModel):
    # 返回 answer 给用户，同时保留 citations 和 hits 供调试与展示。
    answer: str
    citations: list[Citation]
    hits: list[SearchHit]
