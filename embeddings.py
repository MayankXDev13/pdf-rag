import google.generativeai as genai
from config import GOOGLE_API_KEY, EMBED_MODEL

genai.configure(api_key=GOOGLE_API_KEY)


def get_embedding(text: str) -> list[float]:
    """
    Generate an embedding for a document chunk.
    Used when storing text in the vector database.
    """
    result = genai.embed_content(
        model=EMBED_MODEL, content=text, task_type="retrieval_document"
    )

    return result["embedding"]


def get_query_embedding(text: str) -> list[float]:
    """
    Generate an embedding for a user query.
    Query embeddings use a different task type
    to improve retrieval relevance.
    """
    result = genai.embed_content(
        model=EMBED_MODEL, content=text, task_type="retrieval_query"
    )

    return result["embedding"]


def get_embeddings_batch(texts: list[str], batch_size: int = 100) -> list[list[float]]:
    """
    Generate embeddings for multiple texts efficiently.
    """
    embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        result = genai.embed_content(
            model=EMBED_MODEL, content=batch, task_type="retrieval_document"
        )

        batch_embeddings = [item["embedding"] for item in result["embedding"]]
        embeddings.extend(batch_embeddings)

    return embeddings
