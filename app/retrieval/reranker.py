from sentence_transformers import CrossEncoder

from app.core.models import Chunk


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        chunks: list[Chunk],
        top_k: int = 5,
    ):
        if not chunks:
            return []

        pairs = [[query, chunk.text] for chunk in chunks]

        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(chunks, scores),
            key=lambda x: float(x[1]),
            reverse=True,
        )

        results = []

        for chunk, score in ranked[:top_k]:
            results.append(
                {
                    "chunk": chunk,
                    "score": float(score),
                }
            )

        return results
