from app.core.models import RetrievedChunk


def has_sufficient_evidence(
    results: list[RetrievedChunk],
    min_score: float = 0.0,
) -> bool:
    if not results:
        return False

    return results[0].score >= min_score
