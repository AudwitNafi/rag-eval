import os
from pathlib import Path

import bm25s
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Document

INDEX_DIR = Path(__file__).parent / "index"

data = pd.read_csv("chunks.csv")

doc_list = data.iloc[:, 1].tolist()

class BM25Retriever:
    def __init__(self):
        self.retriever = bm25s.BM25.load(str(INDEX_DIR))

    def retrieve(self, query, k=10):
        tokens = bm25s.tokenize([query])
        results, scores = self.retriever.retrieve(tokens, k=k)
        return [(data.iloc[doc_id, 1], score) for doc_id, score in zip(results[0], scores[0])]
        # return [(self.doc_ids[i], scores[0][j]) for j, i in enumerate(results[0].tolist())]

class DenseRetriever:
    def __init__(self):
        self.client = QdrantClient(
            url=os.getenv("QDRANT_CLUSTER_ENDPOINT"),
            api_key=os.getenv("QDRANT_API_KEY"),
            cloud_inference=True
        )

    def retrieve(self, query, collection="northstar_equipment", k=10):
        results = self.client.query_points(
            collection_name=collection,
            query=Document(
                text=query,
                model="sentence-transformers/all-MiniLM-L6-v2"
            ),
            with_payload=True,
            limit=10
        )

        return [(result.payload["chunk"], result.score) for result in results.points]
