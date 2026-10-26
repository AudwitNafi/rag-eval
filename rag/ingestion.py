from pathlib import Path

from langchain_community.document_loaders import TextLoader
from qdrant_client.models import VectorParams, Distance, Document
from rag.chunking import perform_chunking
from rag.qdrant_setup import client

# client.create_collection(
#     collection_name="northstar_equipment",
#     vectors_config=VectorParams(size=384, distance=Distance.COSINE),
# )

# file_path = Path(__file__).parent.parent / "bm25_test_catalog.md"
#
# loader = TextLoader(file_path)
# docs = loader.load()
# doc_content = docs[0].page_content
#
# chunk_points = perform_chunking(doc_content)


# client.upsert(collection_name='northstar_equipment', points=chunk_points)



if __name__ == "__main__":
    query_text = "hot-swappable switches"

    results = client.query_points(
        collection_name="northstar_equipment",
        query=Document(
            text=query_text,
            model="sentence-transformers/all-MiniLM-L6-v2"
        ),
        with_payload=True,
        limit=3
    )

    for result in results.points:
        print(f"CHUNK: {result.payload}\n")
        print(f"SCORE: {result.score}\n\n")
