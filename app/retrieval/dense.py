from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

from app.core.models import Chunk


class DenseRetriever:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.model = SentenceTransformer(model_name)

        self.index = None
        self.chunks: list[Chunk] = []

    def build(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks

        texts = [chunk.text for chunk in chunks]

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
        )

        embeddings = np.asarray(
            embeddings,
            dtype="float32",
        )

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)

        self.index.add(embeddings)

    def search(
        self,
        query: str,
        top_k: int = 5,
    ):
        if self.index is None:
            raise RuntimeError("Index has not been built.")

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32",
        )

        scores, indices = self.index.search(
            query_embedding,
            top_k,
        )

        results = []

        for score, idx in zip(scores[0], indices[0]):
            chunk = self.chunks[idx]

            results.append(
                {
                    "chunk": chunk,
                    "score": float(score),
                }
            )

        return results
