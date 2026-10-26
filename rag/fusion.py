from collections import defaultdict
from retriever import BM25Retriever, DenseRetriever


def reciprocal_rank_fusion(
    rankings: list[list[str]], k: int = 60
) -> list[tuple[str, float]]:
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


def hybrid_candidates(
    query: str,
    bm25: BM25Retriever,
    dense: DenseRetriever,
    candidate_k: int = 10,
) -> list[tuple[str, float]]:
    bm25_ids = [doc_id for doc_id, _ in bm25.retrieve(query)]
    dense_ids = [doc_id for doc_id, _ in dense.retrieve(query)]
    return reciprocal_rank_fusion([bm25_ids, dense_ids])[:candidate_k]

if __name__ == "__main__":
    print(hybrid_candidates("hot-swappable switches", BM25Retriever(), DenseRetriever()))