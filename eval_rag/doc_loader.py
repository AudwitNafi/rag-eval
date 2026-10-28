import uuid

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document

from eval_rag.helpers import clean_pdf_text


def load_doc(file_path):
    loader = PyMuPDFLoader(
        file_path=str(file_path),
        extract_tables="html"
    )

    docs_list = loader.load()
    offset = 0
    page_contents = []
    page_ranges = []
    for doc in docs_list:
        cleaned_doc = clean_pdf_text(doc.page_content)
        if not cleaned_doc:
            continue
        if page_contents:
            offset += 2
        start = offset
        offset = start + len(cleaned_doc)
        end = offset
        page_ranges.append((start, end, doc.metadata['page'] + 1))
        page_contents.append(cleaned_doc)

    full_text = "\n\n".join(page_contents)
    metadata = {
        "source": file_path.name,
        "total_pages": len(docs_list),
        "tenant_id": uuid.uuid4(),
    }

    return Document(
        page_content=full_text,
        metadata=metadata
    ), page_ranges
