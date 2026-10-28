import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor

from eval_rag.chunker import chunk_document
from eval_rag.doc_loader import load_doc
from eval_rag.embed_and_store import embed_and_store_chunks
from eval_rag.helpers import build_rag_chain, rag_prompt
from eval_rag.qdrant_setup import load_rag_collection, create_rag_collection
from eval_rag.retriever import get_retriever




BASE_DIR = Path(__file__).resolve().parent.parent
KB_PATH = BASE_DIR / "knowledge-base" / "nimbus_harbor_handbook.pdf"

# EMBEDDINGS = OllamaEmbeddings(model="bge-m3")
EMBEDDINGS = OllamaEmbeddings(model="qwen3-embedding")
LLM = "llama3.2:3b"

logger = logging.getLogger(__name__)

# tracing
tracer_provider = register(
    project_name="wnlp-rag",
    endpoint="http://localhost:14317",
    protocol="grpc",
)

LangChainInstrumentor().instrument(
    tracer_provider=tracer_provider
)


logging.basicConfig(level=logging.INFO)

logger.info("Loading vector store...")
vector_store = create_rag_collection(embeddings=EMBEDDINGS, collection_name="wnlp_rag")

logger.info("Loading document...")
document, page_ranges = load_doc(file_path=KB_PATH)


logger.info("Chunking document...")
chunks = chunk_document(document, page_ranges, 200, 20)


logger.info("Embedding and storing chunks...")
embed_and_store_chunks(vector_store, chunks)


logger.info("Loading retriever...")
retriever = get_retriever(vector_store, rerank=True)

# Initialize Groq LLM
llm = OllamaLLM(
    model=LLM,
    temperature=0.0,
    num_ctx=8192,
    num_predict=512,
    repeat_penalty=1.1,
)

chain = build_rag_chain(retriever, llm, rag_prompt)

if __name__ == "__main__":
    query = "What does error code HB-5033 mean, and which runbook should be followed?"
    logger.info(f"Query: {query}")
    print(chain.invoke(query))
