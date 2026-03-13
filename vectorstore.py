import hashlib
from pinecone import Pinecone, ServerlessSpec
from config import PINECONE_API_KEY, INDEX_NAME
from embeddings import get_embedding

pc = Pinecone(api_key=PINECONE_API_KEY)


def create_index():
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )


def get_index():
    return pc.Index(INDEX_NAME)


def compute_file_id(filename: str, file_hash: str) -> str:
    return f"{filename}_{file_hash}"


def file_exists_in_index(filename: str, file_hash: str) -> bool:
    index = get_index()
    file_id = compute_file_id(filename, file_hash)
    
    try:
        fetch_response = index.fetch(ids=[file_id])
        return file_id in fetch_response.vectors
    except Exception:
        return False


def upsert_chunks(chunks: list[dict], filename: str, file_hash: str, batch_size: int = 100):
    index = get_index()
    file_id = compute_file_id(filename, file_hash)
    
    vectors = []
    for chunk in chunks:
        embedding = get_embedding(chunk["text"])
        vectors.append({
            "id": f"{file_id}_chunk_{chunk['chunk_index']}",
            "values": embedding,
            "metadata": {
                "text": chunk["text"],
                "page": chunk["page"],
                "chunk_index": chunk["chunk_index"],
                "filename": filename,
                "file_hash": file_hash,
                "file_id": file_id
            }
        })
        
        if len(vectors) >= batch_size:
            index.upsert(vectors=vectors)
            vectors = []
    
    if vectors:
        index.upsert(vectors=vectors)


def delete_file(filename: str, file_hash: str):
    index = get_index()
    file_id = compute_file_id(filename, file_hash)
    
    try:
        delete_response = index.delete(filter={"file_id": {"$eq": file_id}})
        return True
    except Exception as e:
        print(f"Error deleting file: {e}")
        return False


def list_indexed_files():
    index = get_index()
    stats = index.describe_index_stats()
    
    filenames = set()
    if "namespaces" in stats and "" in stats["namespaces"]:
        namespace_stats = stats["namespaces"][""]
        if "vectors" in namespace_stats and namespace_stats["vectors"] > 0:
            query_response = index.query(
                vector=[0.0] * 768,
                top_k=10000,
                include_metadata=True
            )
            for match in query_response["matches"]:
                if "filename" in match["metadata"]:
                    filenames.add(match["metadata"]["filename"])
    
    return list(filenames)
