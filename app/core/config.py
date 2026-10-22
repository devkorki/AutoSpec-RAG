from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    pdf_dir: Path = Path("data/pdfs")
    index_dir: Path = Path("data/index")
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    chunk_size: int = 700
    chunk_overlap: int = 100
    dense_k: int = 20
    lexical_k: int = 20
    fused_k: int = 20
    final_k: int = 5


settings = Settings()
