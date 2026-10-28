from phoenix.client import Client

from eval_rag.evaluation.evaluator import run_rag
from eval_rag.evaluation.eval_functions import *

DATASET_ID = "RGF0YXNldDox"

client = Client()
dataset = client.datasets.get_dataset(dataset=DATASET_ID)

experiment = client.experiments.run_experiment(
    dataset=dataset,
    task=run_rag,
    evaluators=[
        precision_at_k,
        hit_at_k,
        evaluate_context_coverage,
        evaluate_answer_correctness,
        evaluate_faithfulness,
    ],
    experiment_name="qwen3-emb-hybrid-rerank-200-20",
)

print("Evaluation completed.")
