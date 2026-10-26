import os
import pandas as pd
from pathlib import Path

import bm25s

data = pd.read_csv("chunks.csv")

doc_list = data.iloc[:, 1].tolist()
doc_ids = data.iloc[:, 0].tolist()


tokens = bm25s.tokenize(doc_list)

# print(list(tokens.vocab.items()))

# retriever = bm25s.BM25()
# retriever.index(tokens)

INDEX_DIR = Path(__file__).parent / "index"
os.makedirs(INDEX_DIR, exist_ok=True)

retriever = bm25s.BM25.load(str(INDEX_DIR))

# with open(INDEX_DIR / "doc_id_map.txt", "w") as f:
#     f.write("\n".join(doc_ids))

# retriever.save(str(INDEX_DIR))

if __name__ == "__main__":
    query = "BenchBot RB4"
    query_tokens = bm25s.tokenize([query])
    results, scores = retriever.retrieve(query_tokens, k=1)
    print(f"RESULT IDS: {results[0]}")
    for doc_id in results[0]:
        print(f"Doc ID: {doc_id}, Content: {data.iloc[doc_id, 1]}\n\n")
    print(f"Score: {scores}")

