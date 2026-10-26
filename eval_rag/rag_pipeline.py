import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor

from eval_rag.chunker import chunk_document
from eval_rag.doc_loader import load_doc
from eval_rag.embed_and_store import embed_and_store_chunks
from eval_rag.helpers import build_rag_chain, rag_prompt
from eval_rag.qdrant_setup import create_or_load_rag_collection
from eval_rag.retriever import get_retriever




BASE_DIR = Path(__file__).resolve().parent.parent
KB_PATH = BASE_DIR / "knowledge-base" / "nimbus_harbor_handbook.pdf"
EMBEDDINGS = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
LLM = "qwen/qwen3.8-27b"

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
vector_store = create_or_load_rag_collection(embeddings=EMBEDDINGS, collection_name="wnlp_rag")

logger.info("Loading document...")
document, page_ranges = load_doc(file_path=KB_PATH)


logger.info("Chunking document...")
chunks = chunk_document(document, page_ranges)


logger.info("Embedding and storing chunks...")
embed_and_store_chunks(vector_store, chunks)


logger.info("Loading retriever...")
retriever = get_retriever(vector_store)

# Initialize Groq LLM
llm = ChatGroq(
    model=LLM,
    temperature=0.0,
    reasoning_effort='none'
)

chain = build_rag_chain(retriever, llm, rag_prompt)

if __name__ == "__main__":
    query = "What does error code HB-5033 mean, and which runbook should be followed?"
    logger.info(f"Query: {query}")
    print(chain.invoke(query))
