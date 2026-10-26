from rag.qdrant_setup import client
from qdrant_client.models import Document

query_text = "vegetarian dishes"

results = client.query_points(
    collection_name="items",
    query=Document(
        text=query_text,
        model="sentence-transformers/all-MiniLM-L6-v2"
    ),
    with_payload=True,
    limit=5
)

for result in results.points:
    print(f"Item: {result.payload.get('item_name', 'N/A')}")
    print(f"Score: {result.score}")
    print(f"Description: {result.payload.get('description', 'N/A')}")
    print(f"Price: {result.payload.get('price', 'N/A')}")
    print("-" * 40)
