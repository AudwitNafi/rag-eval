# from qdrant_client import QdrantClient
# from dotenv import load_dotenv
#
# import os
# load_dotenv()
# from qdrant_client.models import VectorParams, Distance
#
# client = QdrantClient(
#     url=os.getenv("QDRANT_CLUSTER_ENDPOINT"),
#     api_key=os.getenv("QDRANT_API_KEY"),
#     cloud_inference=True
# )
#
# if __name__ == "__main__":
#     client.create_collection(
#         collection_name="items",
#         vectors_config=VectorParams(size=384, distance=Distance.COSINE),
#     )
#
