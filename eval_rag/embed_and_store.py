from uuid import NAMESPACE_URL, uuid5

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore


def embed_and_store_chunks(
    vector_store: VectorStore,
    chunks: list[Document],
):
    ids = [
        str(uuid5(NAMESPACE_URL, chunk.metadata["id"]))
        for chunk in chunks
    ]

    vector_store.add_documents(
        documents=chunks,
        ids=ids,
    )

    return vector_store
