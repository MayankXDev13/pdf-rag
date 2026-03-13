import hashlib
from io import BytesIO
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def load_pdf(file_data: bytes) -> list[Document]:
    pdf_stream = BytesIO(file_data)
    loader = PyPDFLoader(file=pdf_stream)
    return loader.load()


def get_file_hash(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def get_filename(path: str) -> str:
    import os

    return os.path.basename(path)


def chunk_documents(
    documents: list[Document], chunk_size: int = 800, overlap: int = 100
) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        add_start_index=True,
    )
    return text_splitter.split_documents(documents)
