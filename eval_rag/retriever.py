from langchain_core.vectorstores import VectorStore


from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from pydantic import ConfigDict
from sentence_transformers import CrossEncoder

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
_reranker = None


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL, max_length=512)
    return _reranker


class RerankingRetriever(BaseRetriever):
    base_retriever: BaseRetriever
    top_k: int = 3

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        candidates = self.base_retriever.invoke(query)
        if not candidates:
            return []

        scores = _get_reranker().predict(
            [(query, doc.page_content) for doc in candidates]
        )
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[: self.top_k]]


def get_retriever(vector_store: VectorStore, k=3, rerank=False, candidate_k=20):
    if not rerank:
        return vector_store.as_retriever(search_kwargs={"k": k})

    base = vector_store.as_retriever(search_kwargs={"k": candidate_k})
    return RerankingRetriever(base_retriever=base, top_k=k)
