from pydantic import BaseModel, Field
from typing import Literal


class Chunk(BaseModel):
    id: str
    text: str
    source: str
    page: int
    model: str | None = None
    manufacturer: str | None = None
    year: int | None = None


class RetrievedChunk(Chunk):
    score: float
    rank: int | None = None


class AskRequest(BaseModel):
    question: str = Field(min_length=1)

    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
    )

    engine: Literal[
        "native",
        "langchain",
        "llamaindex",
    ] = "native"

    vector_store: Literal[
        "faiss",
        "chroma",
    ] = "faiss"


class Citation(BaseModel):
    source: str
    page: int


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
