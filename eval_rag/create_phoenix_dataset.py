import pandas as pd
from phoenix.client import Client

df = pd.read_csv("benchmark_qa.csv").dropna(
    subset=["question", "answer"]
)

examples = [
    {
        "input": {
            "question": str(row.question),
        },
        "output": {
            "reference_answer": str(row.answer),
        },
    }
    for row in df.itertuples(index=False)
]

client = Client()

dataset = client.datasets.create_dataset(
    name="nimbus-rag-qa",
    examples=examples,
)

print("Dataset ID:", dataset.id)
