import google.generativeai as genai
from config import GOOGLE_API_KEY, EMBED_MODEL

genai.configure(api_key=GOOGLE_API_KEY)


def get_embedding(text: str) -> list[float]:
    result = genai.embed_content(
        model=EMBED_MODEL,
        content=text
    )
    return result["embedding"]


def get_embeddings_batch(texts: list[str], batch_size: int = 100) -> list[list[float]]:
    embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = []
        
        for text in batch:
            embedding = get_embedding(text)
            batch_embeddings.append(embedding)
        
        embeddings.extend(batch_embeddings)
    
    return embeddings
