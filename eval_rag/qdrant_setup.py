from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

client = QdrantClient(url="http://localhost:6333")


def create_or_load_rag_collection(
    embeddings,
    sparse_embeddings=None,
    retrieval_mode: RetrievalMode = RetrievalMode.DENSE,
    collection_name: str = "wnlp_rag",
):
    if not client.collection_exists(collection_name):
        # Determine the embedding dimension.
        vector_size = len(embeddings.embed_query("dimension check"))

        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )

    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
        sparse_embedding=sparse_embeddings,
        distance=Distance.COSINE,
        retrieval_mode=retrieval_mode,
    )
