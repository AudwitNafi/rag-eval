from phoenix.evals import LLM, ClassificationEvaluator
from phoenix.evals.metrics import RetrievalRelevanceEvaluator, FaithfulnessEvaluator

from eval_rag.rag_components import llm, retriever
from eval_rag.helpers import *

JUDGE_MODEL = "llama3.2:3b"

judge_llm = LLM(
    provider="openai",
    model=JUDGE_MODEL,
    api_key="ollama",
    base_url="http://localhost:11434/v1",
)

answer_chain = build_answer_chain(llm, rag_prompt)

retrieval_relevance_judge = RetrievalRelevanceEvaluator(
    llm=judge_llm,
)

faithfulness_judge = FaithfulnessEvaluator(
    llm=judge_llm,
)

def format_doc(doc):
    source = doc.metadata.get("source", "Unknown source")
    pages = doc.metadata.get("pages")

    if pages is None:
        pages = doc.metadata.get("page", [])

    if isinstance(pages, (str, int)):
        pages = [pages]

    page_text = ", ".join(map(str, pages)) if pages else "Unknown"

    return (
        f"Source: {source}\n"
        f"Pages: {page_text}\n"
        f"Content:\n{doc.page_content}"
    )


def run_rag(inputs):
    question = inputs["question"]

    # Retrieve the top K documents.
    docs = retriever.invoke(question)
    contexts = [format_doc(doc) for doc in docs]

    # Generate the answer using exactly those retrieved documents.
    context_text = (
        "\n\n---\n\n".join(contexts)
        if contexts
        else "No documents were retrieved."
    )

    answer = answer_chain.invoke({
        "context": context_text,
        "question": question,
    })

    # Automatically judge each retrieved chunk's relevance.
    # These are estimated labels, not manually verified ground truth.
    chunk_relevance = []

    for context in contexts:
        result = retrieval_relevance_judge.evaluate({
            "input": question,
            "context": context,
        })[0]

        chunk_relevance.append(float(result.score or 0.0))

    return {
        "answer": answer,
        "contexts": contexts,
        "chunk_ids": [
            doc.metadata.get("id", "unknown") for doc in docs
        ],
        "chunk_relevance": chunk_relevance,
    }

answer_correctness_judge = ClassificationEvaluator(
    name="answer_correctness",
    llm=judge_llm,
    choices={
        "incorrect": 0.0,
        "correct": 1.0,
    },
    prompt_template="""
Compare the generated answer with the reference answer.

Question:
{question}

Reference answer:
{reference_answer}

Generated answer:
{generated_answer}

Judge whether the generated answer conveys the same essential facts
as the reference. Ignore harmless differences in wording.
Mark incorrect if it contradicts the reference or misses essential facts.

Return correct or incorrect.
""",
)


context_coverage_judge = ClassificationEvaluator(
    name="context_recall_proxy",
    llm=judge_llm,
    choices={
        "missing": 0.0,
        "partial": 0.5,
        "complete": 1.0,
    },
    prompt_template="""
Evaluate how well the retrieved context supports the reference answer.

Question:
{question}

Reference answer:
{reference_answer}

Retrieved context:
{context}

Classify as:
- complete: The context supports all essential facts in the reference.
- partial: The context supports some but not all essential facts.
- missing: The necessary evidence is absent.

Return only complete, partial, or missing.
""",
)

