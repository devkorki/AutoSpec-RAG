from collections import defaultdict


def reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Fuse ranked result IDs without trying to compare incompatible raw scores."""
    scores: dict[str, float] = defaultdict(float)
    for ranked_ids in ranked_lists:
        for rank, item_id in enumerate(ranked_ids, start=1):
            scores[item_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
