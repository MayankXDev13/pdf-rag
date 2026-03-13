import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec

from config import *
from utils import load_pdf, chunk_text

genai.configure(api_key=GOOGLE_API_KEY)

pc = Pinecone(api_key=PINECONE_API_KEY)


def create_index():

    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )


def get_embedding(text):

    result = genai.embed_content(
        model=EMBED_MODEL,
        content=text
    )

    return result["embedding"]


def ingest_pdf(path):

    text = load_pdf(path)

    chunks = chunk_text(text)

    index = pc.Index(INDEX_NAME)

    vectors = []

    for i, chunk in enumerate(chunks):

        embedding = get_embedding(chunk)

        vectors.append({
            "id": str(i),
            "values": embedding,
            "metadata": {"text": chunk}
        })

    index.upsert(vectors=vectors)


if __name__ == "__main__":

    create_index()

    ingest_pdf("pdfs/sample.pdf")

    print("PDF successfully indexed")