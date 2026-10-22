from pathlib import Path

import chromadb
from sentence_transformers import (
    SentenceTransformer,
)

from app.core.models import Chunk


class ChromaRetriever:
    def __init__(
        self,
        persist_directory: str = "data/chroma",
        collection_name: str = "autospec",
        model_name: str = ("sentence-transformers/" "all-MiniLM-L6-v2"),
    ):
        self.model = SentenceTransformer(model_name)

        Path(persist_directory).mkdir(
            parents=True,
            exist_ok=True,
        )

        self.client = chromadb.PersistentClient(
            path=persist_directory,
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
        )

        self.chunks_by_id = {}

    def build(
        self,
        chunks: list[Chunk],
    ) -> None:
        self.chunks_by_id = {chunk.id: chunk for chunk in chunks}

        ids = [chunk.id for chunk in chunks]

        texts = [chunk.text for chunk in chunks]

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
        ).tolist()

        metadatas = [
            {
                "source": chunk.source,
                "page": chunk.page,
            }
            for chunk in chunks
        ]

        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
    ):
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
        ).tolist()

        result = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
        )

        ids = result["ids"][0]

        results = []

        for rank, chunk_id in enumerate(
            ids,
            start=1,
        ):
            chunk = self.chunks_by_id.get(chunk_id)

            if chunk is None:
                continue

            results.append(
                {
                    "chunk": chunk,
                    "score": 1.0 / rank,
                }
            )

        return results
