from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, SparseVectorParams, Modifier

client = QdrantClient(url="http://localhost:6333")

SPARSE_VECTOR_NAME = "langchain-sparse"  # default name used by QdrantVectorStore
SPARSE_EMBEDDINGS = FastEmbedSparse(model_name="Qdrant/bm25")


def create_rag_collection(
    embeddings,
    sparse_embeddings=SPARSE_EMBEDDINGS,
    retrieval_mode: RetrievalMode = RetrievalMode.HYBRID,
    collection_name: str = "wnlp_rag",
):
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
    # Determine the embedding dimension.
    vector_size = len(embeddings.embed_query("dimension check"))

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE,
        ),
        # BM25 needs the IDF modifier so Qdrant applies inverse document frequency.
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: SparseVectorParams(modifier=Modifier.IDF),
        },
    )
    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
        sparse_embedding=sparse_embeddings,
        sparse_vector_name=SPARSE_VECTOR_NAME,
        distance=Distance.COSINE,
        retrieval_mode=retrieval_mode,
    )


def load_rag_collection(
    embeddings,
    sparse_embeddings=SPARSE_EMBEDDINGS,
    retrieval_mode: RetrievalMode = RetrievalMode.HYBRID,
    collection_name: str = "wnlp_rag",
):
    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
        sparse_embedding=sparse_embeddings,
        sparse_vector_name=SPARSE_VECTOR_NAME,
        distance=Distance.COSINE,
        retrieval_mode=retrieval_mode,
    )