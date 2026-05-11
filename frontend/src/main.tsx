import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { BookOpen, Database, FilePlus2, RotateCcw, Search, Sparkles, Trash2 } from "lucide-react";
import "./styles.css";

// 前端只负责交互和展示，RAG 逻辑都在后端。
// 你学习时可以把 main.tsx 看成“用户动作 -> HTTP 请求 -> 展示结果”的连接层。
const API_BASE = "http://127.0.0.1:8000";

type DocumentSummary = {
  // 这个类型对应后端的 DocumentSummary。
  // 前端侧边栏只需要标题和 chunk 数，不需要完整正文。
  id: string;
  title: string;
  chunk_count: number;
};

type SearchHit = {
  // SearchHit 是检索阶段的可解释输出：
  // chunk 告诉我们召回了哪段证据，score 告诉我们相关度，matched_terms 告诉我们为什么命中。
  chunk: {
    id: string;
    document_title: string;
    index: number;
    text: string;
  };
  score: number;
  matched_terms: string[];
};

type AskResponse = {
  // answer 是给用户看的最终回答。
  // citations 和 hits 是给“信任与调试”看的证据链。
  answer: string;
  citations: Array<{
    chunk_id: string;
    document_title: string;
    snippet: string;
  }>;
  hits: SearchHit[];
};

// 默认样例文本让你第一次打开页面时就能快速体验 RAG 链路。
// 它同时覆盖 Transformer、RAG、Agent 三个概念，方便后续继续扩展学习材料。
const sampleText = `Transformer 的核心是 self-attention。它让模型在处理一个 token 时，可以根据相关性关注序列中的其他 token。

RAG 的核心思想是先检索外部知识，再把检索到的证据作为上下文交给生成模型。这样可以降低幻觉，并让回答具备来源引用。

Agent 在大模型应用中通常由规划、记忆、工具调用和反思评估组成。它适合解决多步骤任务，而不是只做单轮问答。`;

function App() {
  // documents：知识库里已有的文档摘要，用来展示侧边栏。
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  // title/text：文档入库表单。点击“切分并入库”后会发给后端。
  const [title, setTitle] = useState("RAG / Transformer / Agent 学习笔记");
  const [text, setText] = useState(sampleText);
  // query：用户问题。点击 Ask 后会进入后端 RAG 查询流程。
  const [query, setQuery] = useState("RAG 为什么可以降低幻觉");
  // answer：后端返回的完整问答结果，包含答案、引用和检索命中。
  const [answer, setAnswer] = useState<AskResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");

  useEffect(() => {
    // 页面首次加载时，先从后端拉取已有知识库状态。
    refreshDocuments();
  }, []);

  async function refreshDocuments() {
    // GET /documents -> 后端返回 DocumentSummary[]。
    const response = await fetch(`${API_BASE}/documents`);
    setDocuments(await response.json());
  }

  async function addDocument() {
    // 文档入库是 RAG 的 ingestion 阶段：
    // 前端发送 title/text，后端负责清洗、切分、保存。
    setLoading(true);
    setMessage("");
    try {
      const response = await fetch(`${API_BASE}/documents`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title, text }),
      });
      if (!response.ok) {
        throw new Error("文档入库失败");
      }
      setMessage("文档已切分并写入知识库");
      await refreshDocuments();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "未知错误");
    } finally {
      setLoading(false);
    }
  }

  async function ask() {
    // 问答是 RAG 的 query 阶段：
    // query -> retrieve top_k chunks -> generate grounded answer。
    setLoading(true);
    setMessage("");
    try {
      const response = await fetch(`${API_BASE}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, top_k: 5 }),
      });
      if (!response.ok) {
        throw new Error("问答请求失败");
      }
      setAnswer(await response.json());
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "未知错误");
    } finally {
      setLoading(false);
    }
  }

  async function clearDocuments() {
    // 清空知识库主要用于本地实验。真实产品里通常会做“删除单个文档”。
    setLoading(true);
    setAnswer(null);
    await fetch(`${API_BASE}/documents`, { method: "DELETE" });
    await refreshDocuments();
    setLoading(false);
  }

  return (
    <main className="app-shell">
      <section className="workspace">
        <aside className="sidebar">
          <div className="brand">
            <Sparkles size={22} />
            <div>
              <h1>RAG Agent Lab</h1>
              <p>面向求职作品集的可解释 RAG 系统</p>
            </div>
          </div>

          <div className="metric-grid">
            <div>
              <span>Documents</span>
              <strong>{documents.length}</strong>
            </div>
            <div>
              <span>Chunks</span>
              {/* chunk 总数能帮助你观察：文档越多、切分越细，检索空间就越大。 */}
              <strong>{documents.reduce((sum, doc) => sum + doc.chunk_count, 0)}</strong>
            </div>
          </div>

          <div className="document-list">
            <div className="section-title">
              <Database size={16} />
              <span>知识库</span>
            </div>
            {documents.length === 0 ? (
              <p className="empty">还没有文档，先添加一份学习笔记。</p>
            ) : (
              documents.map((document) => (
                <div className="document-item" key={document.id}>
                  <BookOpen size={16} />
                  <div>
                    <strong>{document.title}</strong>
                    <span>{document.chunk_count} chunks</span>
                  </div>
                </div>
              ))
            )}
          </div>

          <button className="ghost danger" onClick={clearDocuments} disabled={loading || documents.length === 0}>
            <Trash2 size={16} />
            清空知识库
          </button>
        </aside>

        <section className="main-panel">
          <div className="composer">
            <div className="panel-heading">
              <FilePlus2 size={18} />
              <h2>文档入库</h2>
            </div>
            <input value={title} onChange={(event) => setTitle(event.target.value)} placeholder="文档标题" />
            <textarea value={text} onChange={(event) => setText(event.target.value)} />
            <button onClick={addDocument} disabled={loading || !title.trim() || !text.trim()}>
              <FilePlus2 size={16} />
              切分并入库
            </button>
          </div>

          <div className="qa-panel">
            <div className="panel-heading">
              <Search size={18} />
              <h2>检索增强问答</h2>
            </div>
            <div className="query-row">
              <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="输入你的问题" />
              <button onClick={ask} disabled={loading || !query.trim()}>
                <Sparkles size={16} />
                Ask
              </button>
            </div>

            {message && <p className="status">{message}</p>}

            {answer ? (
              <div className="answer-layout">
                <div className="answer-box">
                  <div className="answer-title">
                    <Sparkles size={16} />
                    <span>生成结果</span>
                  </div>
                  {/* pre 保留换行，方便展示生成器组织出的多条证据说明。 */}
                  <pre>{answer.answer}</pre>
                </div>

                <div className="trace-box">
                  <div className="answer-title">
                    <RotateCcw size={16} />
                    <span>检索证据</span>
                  </div>
                  {answer.hits.map((hit) => (
                    <article className="hit" key={hit.chunk.id}>
                      <div className="hit-head">
                        <strong>{hit.chunk.document_title}</strong>
                        {/* score 是调试 RAG 的关键指标：它帮助判断召回是否可靠。 */}
                        <span>{hit.score.toFixed(4)}</span>
                      </div>
                      <p>{hit.chunk.text}</p>
                      <div className="terms">
                        {/* matched_terms 让检索不再是黑盒：你能看到 query 和 chunk 到底重合了什么。 */}
                        {hit.matched_terms.map((term) => (
                          <span key={term}>{term}</span>
                        ))}
                      </div>
                    </article>
                  ))}
                </div>
              </div>
            ) : (
              <div className="placeholder">添加文档后提问，这里会展示答案、引用和检索分数。</div>
            )}
          </div>
        </section>
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
