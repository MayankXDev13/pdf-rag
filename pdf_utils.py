import hashlib
from pathlib import Path
from pypdf import PdfReader


def load_pdf(path: str) -> tuple[list[str], int]:
    reader = PdfReader(path)
    pages = []
    
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    
    return pages, len(reader.pages)


def get_file_hash(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def get_filename(path: str) -> str:
    return Path(path).name


def chunk_text(pages: list[str], chunk_size: int = 800, overlap: int = 100) -> list[dict]:
    chunks = []
    global_chunk_id = 0
    
    for page_num, page_text in enumerate(pages, start=1):
        start = 0
        
        while start < len(page_text):
            end = start + chunk_size
            chunk_text = page_text[start:end]
            
            if chunk_text.strip():
                chunks.append({
                    "id": str(global_chunk_id),
                    "text": chunk_text,
                    "page": page_num,
                    "chunk_index": len(chunks)
                })
                global_chunk_id += 1
            
            start += chunk_size - overlap
    
    return chunks
