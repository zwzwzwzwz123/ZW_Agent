from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json

from pydantic import ValidationError

from app.core.generator import GroundedAnswerGenerator
from app.core.retriever import LexicalRetriever
from app.core.schemas import AskRequest, DocumentCreate, SearchRequest
from app.core.store import KnowledgeStore
from app.settings import DATA_PATH


# dev_server.py 是“开发后备服务”：
# 它不用 FastAPI，只用 Python 标准库暴露 HTTP API。
# 这样即使依赖安装暂时有问题，也能继续验证 RAG 主链路。
#
# 学习重点不是 http.server 的所有细节，而是看清楚 API 如何调用核心模块：
# /documents -> KnowledgeStore.add_document
# /search    -> LexicalRetriever.search
# /ask       -> Retriever + Generator
store = KnowledgeStore(DATA_PATH)
retriever = LexicalRetriever()
generator = GroundedAnswerGenerator()


class DevRequestHandler(BaseHTTPRequestHandler):
    def do_OPTIONS(self) -> None:
        # 浏览器跨域请求前可能会先发 OPTIONS 预检请求。
        # 前端和后端端口不同，所以这里需要正确响应 CORS。
        self._send_empty()

    def do_GET(self) -> None:
        if self.path == "/health":
            # 健康检查接口：只确认服务是否活着，不涉及业务逻辑。
            self._send_json({"status": "ok"})
            return
        if self.path == "/documents":
            # 前端侧边栏刷新知识库列表时会调用这里。
            documents = [document.model_dump() for document in store.list_documents()]
            self._send_json(documents)
            return
        self._send_json({"detail": "Not found"}, status=404)

    def do_DELETE(self) -> None:
        if self.path == "/documents":
            # 本地实验阶段经常需要清空知识库重新测试。
            store.clear()
            self._send_json({"status": "cleared"})
            return
        self._send_json({"detail": "Not found"}, status=404)

    def do_POST(self) -> None:
        try:
            payload = self._read_json()
            if self.path == "/documents":
                # 文档入库：校验输入 -> 保存 document -> 切分 chunks。
                document = DocumentCreate(**payload)
                self._send_json(store.add_document(document).model_dump())
                return
            if self.path == "/search":
                # 只检索不生成，适合调试 retriever 的召回效果。
                request = SearchRequest(**payload)
                hits = retriever.search(request.query, store.list_chunks(), request.top_k)
                self._send_json([hit.model_dump() for hit in hits])
                return
            if self.path == "/ask":
                # 完整 RAG 查询：先检索证据，再基于证据生成回答。
                request = AskRequest(**payload)
                hits = retriever.search(request.query, store.list_chunks(), request.top_k)
                answer, citations = generator.answer(request.query, hits)
                self._send_json(
                    {
                        "answer": answer,
                        "citations": [citation.model_dump() for citation in citations],
                        "hits": [hit.model_dump() for hit in hits],
                    }
                )
                return
            self._send_json({"detail": "Not found"}, status=404)
        except ValidationError as error:
            # Pydantic 校验失败时返回 422，前端可以据此提示输入不合法。
            self._send_json({"detail": error.errors()}, status=422)
        except json.JSONDecodeError:
            self._send_json({"detail": "Invalid JSON"}, status=400)

    def log_message(self, format: str, *args) -> None:
        # 关闭标准库服务器默认日志，避免终端输出太吵。
        return

    def _read_json(self) -> dict:
        """读取请求体，并按 UTF-8 解析 JSON。

        中文乱码问题经常出现在这里：发送方、接收方、终端显示必须都用对编码。
        """
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8")
        return json.loads(body) if body else {}

    def _send_empty(self, status: int = 204) -> None:
        self.send_response(status)
        self._send_cors_headers()
        self.end_headers()

    def _send_json(self, payload, status: int = 200) -> None:
        # ensure_ascii=False 保留中文原文；charset=utf-8 告诉浏览器按 UTF-8 解码。
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_cors_headers(self) -> None:
        # 当前前端运行在 5173，后端运行在 8000。端口不同就是跨域。
        self.send_header("Access-Control-Allow-Origin", "http://127.0.0.1:5173")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,DELETE,OPTIONS")


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", 8000), DevRequestHandler)
    print("Dev API running at http://127.0.0.1:8000")
    server.serve_forever()
