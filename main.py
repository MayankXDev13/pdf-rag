from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from typing import Optional
from pydantic import BaseModel
from pdf_utils import load_pdf, chunk_documents, get_file_hash
from vectorstore import (
    add_documents,
    delete_file as delete_from_vectorstore,
    list_indexed_files,
)
from s3_utils import (
    upload_file,
    download_file,
    delete_file as delete_from_s3,
    file_exists,
    list_files as list_s3_files,
)
from rag import ask as rag_ask

app = FastAPI(
    title="PDF RAG API", description="LangChain-powered PDF RAG system with S3 storage"
)


class QueryRequest(BaseModel):
    question: str
    k: Optional[int] = 3
    filename: Optional[str] = None


@app.post("/ingest")
async def ingest(
    file: UploadFile = File(...),
    chunk_size: int = Form(800),
    overlap: int = Form(100),
    rebuild: bool = Form(False),
):

    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    filename = file.filename

    if not rebuild and file_exists(filename):
        raise HTTPException(
            status_code=409,
            detail=f"File '{filename}' already exists. Use rebuild=true to re-index.",
        )

    file_data = await file.read()
    file_hash = get_file_hash(file_data)

    if rebuild:
        delete_from_vectorstore(filename, file_hash)

    upload_file(file_data, filename)

    documents = load_pdf(file_data)
    chunks = chunk_documents(documents, chunk_size=chunk_size, overlap=overlap)

    success = add_documents(chunks, filename, file_hash)

    if not success:
        raise HTTPException(
            status_code=500, detail="Failed to add documents to vector store"
        )

    return {
        "message": "File indexed successfully",
        "filename": filename,
        "pages": len(documents),
        "chunks": len(chunks),
    }


@app.post("/query")
async def query(request: QueryRequest):
    try:
        result = rag_ask(
            question=request.question,
            k=request.k,
            filename=request.filename,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/files")
async def list_files():
    s3_files = list_s3_files()
    pinecone_files = list_indexed_files()

    all_files = list(set(s3_files + pinecone_files))

    return {"files": all_files}


@app.delete("/files/{filename}")
async def delete_file(filename: str):
    from config import INDEX_NAME

    file_hash = "unknown"

    s3_deleted = delete_from_s3(filename)
    vector_deleted = delete_from_vectorstore(filename, file_hash)

    if not s3_deleted and not vector_deleted:
        raise HTTPException(status_code=404, detail="File not found")

    return {"message": f"File '{filename}' deleted from S3 and Pinecone"}
