from pathlib import Path

from langchain_ollama import OllamaLLM, OllamaEmbeddings

from eval_rag.qdrant_setup import load_rag_collection
from eval_rag.retriever import get_retriever

BASE_DIR = Path(__file__).resolve().parent.parent
KB_PATH = BASE_DIR / "knowledge-base" / "nimbus_harbor_handbook.pdf"

EMBEDDINGS = OllamaEmbeddings(model="qwen3-embedding")
# EMBEDDINGS = OllamaEmbeddings(model="bge-m3")
K = 3
LLM = "llama3.2:3b"

vector_store = load_rag_collection(embeddings=EMBEDDINGS)

# Initialize Groq LLM
llm = OllamaLLM(
    model=LLM,
    temperature=0.0,
    # num_ctx=8192,
    # num_predict=512,
    # repeat_penalty=1.1,
)
retriever = get_retriever(vector_store, K, rerank=True)
