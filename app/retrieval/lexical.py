import re

from rank_bm25 import BM25Okapi

from app.core.models import Chunk


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


class BM25Retriever:
    def __init__(self):
        self.bm25 = None
        self.chunks: list[Chunk] = []

    def build(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks

        tokenized_chunks = [tokenize(chunk.text) for chunk in chunks]

        self.bm25 = BM25Okapi(tokenized_chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
    ):
        if self.bm25 is None:
            raise RuntimeError("BM25 index has not been built.")

        query_tokens = tokenize(query)

        scores = self.bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        results = []

        for idx in ranked_indices:
            results.append(
                {
                    "chunk": self.chunks[idx],
                    "score": float(scores[idx]),
                }
            )

        return results
