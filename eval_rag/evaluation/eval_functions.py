from phoenix.evals import create_evaluator

from eval_rag.evaluation.evaluator import answer_correctness_judge, faithfulness_judge, context_coverage_judge
from eval_rag.rag_components import K


@create_evaluator(name="precision_at_3", kind="code")
def precision_at_k(output):
    relevance = output["chunk_relevance"]
    return sum(relevance[:K]) / K if K else 0.0


@create_evaluator(name="hit_at_3", kind="code")
def hit_at_k(output):
    return float(any(score >= 0.5 for score in output["chunk_relevance"][:K]))


@create_evaluator(name="context_recall_proxy", kind="llm")
def evaluate_context_coverage(input, output, expected):
    result = context_coverage_judge.evaluate({
        "question": input["question"],
        "reference_answer": expected["reference_answer"],
        "context": "\n\n".join(output["contexts"]),
    })[0]

    return (
        float(result.score),
        f"{result.label}: {result.explanation}",
    )


@create_evaluator(name="answer_correctness", kind="llm")
def evaluate_answer_correctness(input, output, expected):
    result = answer_correctness_judge.evaluate({
        "question": input["question"],
        "reference_answer": expected["reference_answer"],
        "generated_answer": output["answer"],
    })[0]

    return (
        float(result.score),
        f"{result.label}: {result.explanation}",
    )


@create_evaluator(name="faithfulness", kind="llm")
def evaluate_faithfulness(input, output):
    result = faithfulness_judge.evaluate({
        "input": input["question"],
        "output": output["answer"],
        "context": "\n\n".join(output["contexts"]),
    })[0]

    return (
        float(result.score),
        f"{result.label}: {result.explanation}",
    )
