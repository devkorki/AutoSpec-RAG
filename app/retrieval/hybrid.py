from app.retrieval.dense import DenseRetriever
from app.retrieval.lexical import BM25Retriever
from app.retrieval.rrf import reciprocal_rank_fusion
from app.retrieval.reranker import CrossEncoderReranker
from app.core.models import Chunk


class HybridRetriever:
    def __init__(self, use_reranker: bool = False):
        self.dense = DenseRetriever()
        self.lexical = BM25Retriever()

        self.reranker = CrossEncoderReranker() if use_reranker else None

        self.chunks_by_id: dict[str, Chunk] = {}

    def build(self, chunks: list[Chunk]) -> None:
        self.chunks_by_id = {chunk.id: chunk for chunk in chunks}

        self.dense.build(chunks)
        self.lexical.build(chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 20,
    ):
        dense_results = self.dense.search(
            query,
            top_k=candidate_k,
        )

        lexical_results = self.lexical.search(
            query,
            top_k=candidate_k,
        )

        dense_ids = [result["chunk"].id for result in dense_results]

        lexical_ids = [result["chunk"].id for result in lexical_results]

        fused = reciprocal_rank_fusion(
            [
                dense_ids,
                lexical_ids,
            ]
        )

        if self.reranker is not None:
            candidate_chunks = [
                self.chunks_by_id[chunk_id] for chunk_id, _ in fused[:candidate_k]
            ]

            return self.reranker.rerank(
                query,
                candidate_chunks,
                top_k=top_k,
            )

        results = []

        for chunk_id, score in fused[:top_k]:
            results.append(
                {
                    "chunk": self.chunks_by_id[chunk_id],
                    "score": float(score),
                }
            )

        return results
