
import json
import pandas as pd
from phoenix.client import Client

PHOENIX_ENDPOINT = "http://localhost:6006"
DATASET_NAME = "nimbus_dataset"


def main():
    with open("benchmark_qa.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    dataset = pd.DataFrame({
        "question": [item["question"] for item in data["items"]],
        "answer": [item["answer"] for item in data["items"]],
    })

    # Keep a local copy of the benchmark.
    dataset.to_csv("benchmark_qa.csv", index=False)

    # Connect to the existing Docker-hosted Phoenix server.
    client = Client(base_url=PHOENIX_ENDPOINT)

    # Upload examples to Phoenix.
    phoenix_dataset = client.datasets.create_dataset(
        dataframe=dataset,
        name=DATASET_NAME,
        input_keys=["question"],
        output_keys=["answer"],
    )

    print(f"Uploaded dataset: {phoenix_dataset.name}")
    print(f"Examples: {len(phoenix_dataset)}")
    print(f"Open Phoenix: {PHOENIX_ENDPOINT}/datasets")


if __name__ == "__main__":
    main()
