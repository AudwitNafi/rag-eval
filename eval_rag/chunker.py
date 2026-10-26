from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_document(document: Document, page_ranges) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
        strip_whitespace=True,
    )

    doc_chunks = text_splitter.split_documents([document])

    for idx, chunk in enumerate(doc_chunks):
        chunk_start = chunk.metadata["start_index"]
        chunk_end = chunk_start + len(chunk.page_content)

        pages = [
            page_num
            for page_start, page_end, page_num in page_ranges
            if chunk_start < page_end and chunk_end > page_start
        ]

        chunk.metadata.update({
            "id": f"{document.metadata['source']}_{idx}",
            "source": document.metadata["source"],
            "tenant_id": document.metadata["tenant_id"],
            "pages": pages,
        })

    return doc_chunks
