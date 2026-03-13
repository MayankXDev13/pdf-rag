from typing import Optional
from langchain_pinecone import PineconeVectorStore
from embeddings import embeddings
from config import INDEX_NAME


def get_retriever(k: int = 3, filename: Optional[str] = None):
    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings,
        text_key="text",
    )

    if filename:
        return vectorstore.as_retriever(
            search_kwargs={"k": k, "filter": {"filename": {"$eq": filename}}}
        )

    return vectorstore.as_retriever(search_kwargs={"k": k})
