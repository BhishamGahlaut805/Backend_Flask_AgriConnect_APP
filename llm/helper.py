"""
Helper utilities for LLM module
Optimized for Render deployment - all PDF handling with in-memory processing
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List
import io
from PyPDF2 import PdfReader

# -----------------------------------
# GLOBAL CACHE
# -----------------------------------
_embeddings_model = None


def get_embeddings():
    """
    Lazy-load HuggingFace embeddings
    Loads ONLY when first used, not during startup
    """
    global _embeddings_model

    if _embeddings_model is None:
        from langchain_community.embeddings import HuggingFaceEmbeddings

        print("Loading HuggingFace embeddings...")

        _embeddings_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        print("Embeddings loaded successfully")

    return _embeddings_model


def download_hugging_face_embeddings():
    """
    Backward-compatible wrapper for get_embeddings()
    """
    return get_embeddings()


def load_pdf_from_bytes(file_bytes: bytes, filename: str) -> List[Document]:
    """
    Load PDF from bytes and return Document objects
    No local file storage - processes entirely in memory
    """
    try:
        # Use PyPDF2 to read PDF from bytes
        pdf_file = io.BytesIO(file_bytes)
        pdf_reader = PdfReader(pdf_file)

        documents = []
        for page_num, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            if text.strip():
                doc = Document(
                    page_content=text,
                    metadata={
                        "source": filename,
                        "page": page_num + 1,
                        "total_pages": len(pdf_reader.pages)
                    }
                )
                documents.append(doc)

        return documents

    except Exception as e:
        print(f"Error loading PDF from bytes: {e}")
        raise


def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Filter documents to minimal metadata
    """
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source", "unknown")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src, "page": doc.metadata.get("page", 0)}
            )
        )
    return minimal_docs


def text_split(extracted_data: List[Document], chunk_size: int = 800,
               chunk_overlap: int = 50) -> List[Document]:
    """
    Split documents into text chunks
    """
    if not extracted_data:
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        length_function=len
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks


def chunk_pdf_bytes(file_bytes: bytes, filename: str, chunk_size: int = 800) -> List[Document]:
    """
    Complete pipeline: load PDF from bytes, extract text, split into chunks
    Returns list of Document chunks ready for embedding
    """
    try:
        # Load PDF from bytes
        documents = load_pdf_from_bytes(file_bytes, filename)

        if not documents:
            return []

        # Filter to minimal metadata
        minimal_docs = filter_to_minimal_docs(documents)

        # Split into chunks
        chunks = text_split(minimal_docs, chunk_size=chunk_size)

        return chunks

    except Exception as e:
        print(f"Error chunking PDF: {e}")
        return []
    