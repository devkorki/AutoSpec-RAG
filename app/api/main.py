from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.models import AskRequest, AskResponse
from app.ingestion.pdf_loader import load_pdf_pages
from app.ingestion.chunking import chunk_page

from app.retrieval.hybrid import HybridRetriever
from app.retrieval.chroma import ChromaRetriever

from app.services.rag import RAGService
from app.services.langchain_rag import LangChainRAGService
from app.services.llamaindex_rag import LlamaIndexRAGService

PDF_DIR = Path("data/pdfs")


def load_document_data():
    chunks = []
    page_lookup = {}

    for pdf_path in PDF_DIR.glob("*.pdf"):
        pages = load_pdf_pages(pdf_path)

        for page in pages:
            key = (
                page["source"],
                page["page"],
            )

            page_lookup[key] = page["text"]

            chunks.extend(
                chunk_page(
                    text=page["text"],
                    source=page["source"],
                    page=page["page"],
                    chunk_size=700,
                    chunk_overlap=100,
                )
            )

    if not chunks:
        raise RuntimeError(
            "No document chunks were created. " "Add PDF files to data/pdfs."
        )

    return chunks, page_lookup


def build_services():
    chunks, page_lookup = load_document_data()

    print("Building Native + FAISS...")
    native_retriever = HybridRetriever(
        use_reranker=False,
    )
    native_retriever.build(chunks)

    native_service = RAGService(
        retriever=native_retriever,
        page_lookup=page_lookup,
    )

    print("Building Chroma...")
    chroma_retriever = ChromaRetriever()
    chroma_retriever.build(chunks)

    chroma_service = RAGService(
        retriever=chroma_retriever,
        page_lookup=page_lookup,
    )

    print("Building LangChain...")
    langchain_faiss = LangChainRAGService(
        retriever=native_retriever,
        page_lookup=page_lookup,
    )

    langchain_chroma = LangChainRAGService(
        retriever=chroma_retriever,
        page_lookup=page_lookup,
    )

    print("Building LlamaIndex...")
    llamaindex_service = LlamaIndexRAGService(
        chunks=chunks,
    )

    return {
        ("native", "faiss"): native_service,
        ("native", "chroma"): chroma_service,
        ("langchain", "faiss"): langchain_faiss,
        ("langchain", "chroma"): langchain_chroma,
        # LlamaIndex currently owns its own
        # Chroma-backed retrieval pipeline.
        ("llamaindex", "chroma"): llamaindex_service,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Building AutoSpec services...")

    app.state.services = build_services()

    print("AutoSpec is ready.")

    yield


app = FastAPI(
    title="AutoSpec-RAG",
    version="0.3.0",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "AutoSpec-RAG",
    }


@app.post(
    "/ask",
    response_model=AskResponse,
)
def ask(request: AskRequest):
    key = (
        request.engine,
        request.vector_store,
    )

    service = app.state.services.get(key)

    if service is None:
        return {
            "answer": ("That engine and vector-store " "combination is not available."),
            "citations": [],
        }

    return service.ask(
        question=request.question,
        top_k=request.top_k,
    )
