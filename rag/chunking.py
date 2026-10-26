import re
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from qdrant_client.models import Document, PointStruct

from rag.qdrant_setup import client


def perform_chunking(document: str, chunk_size: int = 500, chunk_overlap: int = 50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", ""],
    )

    chunks = text_splitter.split_text(document)

    section_patterns = (
        # r'^\*\s*\*\s*\*\s*',
        r'^#+\s+(.+)$',
        r'^.+\n[=\-]{2,}$',
        r'^[A-Z]+:$'
    )
    points = []

    for i, chunk in enumerate(chunks):
        current_section = None
        lines = chunk.split('\n')
        for line in lines:
            for pattern in section_patterns:
                match = re.match(pattern, line)
                if match:
                    current_section = match.group()
        # print(f"CHUNK START:{chunk}", f"SECTION: {current_section}")
        # print('=' * 50)
        doc = Document(text=chunk, model="sentence-transformers/all-MiniLM-L6-v2")
        point = PointStruct(
            id = i,
            vector = doc,
            payload={
                "chunk": chunk,
                "section": current_section,
                "chunk_size": len(chunk),
            }
        )
        points.append(point)

    return points

def to_dataframe(points):
    data = pd.DataFrame([point.payload["chunk"] for point in points], columns=["chunk"])
    data.to_csv("chunks.csv", header=False)
    return data


def perform_markdown_chunking():
    pass


if __name__ == '__main__':
    from pathlib import Path
    from langchain_community.document_loaders import TextLoader

    file_path = Path(__file__).parent.parent / "bm25_test_catalog.md"

    loader = TextLoader(file_path)
    docs = loader.load()
    points = perform_chunking(docs[0].page_content)
    data = to_dataframe(points)
    print(data)


