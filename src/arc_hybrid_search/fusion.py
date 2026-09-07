"""Dense retrieval across multiple Chroma collections, and Reciprocal Rank
Fusion to combine dense + sparse (BM25) rankings into one score.
"""


def dense_retrieve(
    query: str, collection, top_k: int, where: dict | None = None
) -> list[tuple[str, float]]:
    """Retrieve top-k `(doc_id, similarity)` pairs from a Chroma collection."""
    results = collection.query(query_texts=[query], n_results=top_k, where=where)
    ids = results["ids"][0]
    distances = results["distances"][0]
    return [
        (doc_id, 1 - dist) for doc_id, dist in zip(ids, distances)
    ]  # cosine distance -> similarity


def joint_dense_retrieve(
    query: str, collections: list, top_k: int, where: dict | None = None
) -> list[tuple[str, float]]:
    """Query several Chroma collections and keep the max score per doc id."""
    max_scores: dict[str, float] = {}
    for collection in collections:
        for doc_id, score in dense_retrieve(query, collection, top_k, where):
            max_scores[doc_id] = max(max_scores.get(doc_id, score), score)
    return sorted(max_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]], k: int = 10
) -> list[tuple[str, dict[str, float]]]:
    """Merge multiple ranked lists using weighted Reciprocal Rank Fusion.

    RRF score = sum(1 / (k + rank)) across all lists a document appears in.
    """
    rrf_scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, (doc_id, _) in enumerate(ranked, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    num_lists = len(ranked_lists)
    max_theoretical = num_lists * (1.0 / (k + 1.0)) if num_lists else 1.0

    detailed_scores = {
        doc_id: {"rrf_score": score, "normalized_score": score / max_theoretical}
        for doc_id, score in rrf_scores.items()
    }
    return sorted(detailed_scores.items(), key=lambda item: item[1]["rrf_score"], reverse=True)
