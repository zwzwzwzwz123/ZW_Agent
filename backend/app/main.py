from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.core.generator import GroundedAnswerGenerator
from app.core.retriever import LexicalRetriever
from app.core.schemas import AskRequest, AskResponse, DocumentCreate, DocumentSummary, SearchHit, SearchRequest
from app.core.store import KnowledgeStore
from app.settings import DATA_PATH


app = FastAPI(title="RAG Agent Lab", version="0.1.0")
store = KnowledgeStore(DATA_PATH)
retriever = LexicalRetriever()
generator = GroundedAnswerGenerator()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/documents", response_model=DocumentSummary)
def create_document(document: DocumentCreate) -> DocumentSummary:
    return store.add_document(document)


@app.post("/documents/upload", response_model=DocumentSummary)
async def upload_document(file: UploadFile = File(...)) -> DocumentSummary:
    content = await file.read()
    text = content.decode("utf-8", errors="ignore")
    title = file.filename or "uploaded-document.txt"
    return store.add_document(DocumentCreate(title=title, text=text))


@app.get("/documents", response_model=list[DocumentSummary])
def list_documents() -> list[DocumentSummary]:
    return store.list_documents()


@app.delete("/documents")
def clear_documents() -> dict[str, str]:
    store.clear()
    return {"status": "cleared"}


@app.post("/search", response_model=list[SearchHit])
def search(request: SearchRequest) -> list[SearchHit]:
    return retriever.search(request.query, store.list_chunks(), request.top_k)


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    hits = retriever.search(request.query, store.list_chunks(), request.top_k)
    answer, citations = generator.answer(request.query, hits)
    return AskResponse(answer=answer, citations=citations, hits=hits)
