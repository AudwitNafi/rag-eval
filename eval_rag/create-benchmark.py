import json
import pandas as pd



def main():
    with open("benchmark_qa.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    dataset = pd.DataFrame({
        "question": [item["question"] for item in data["items"]],
        "answer": [item["answer"] for item in data["items"]],
    })

    # Keep a local copy of the benchmark.
    dataset.to_csv("benchmark_qa.csv", index=False)


if __name__ == "__main__":
    main()
