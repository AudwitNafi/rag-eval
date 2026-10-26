from langchain_core.vectorstores import VectorStore


def get_retriever(vector_store: VectorStore):
    return vector_store.as_retriever(
        search_kwargs={"k": 1},
    )